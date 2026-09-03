"""Causal exact-URL prediction by lightweight discrete pattern matching.

No embedding or neural model is used.  The frozen score is
``log(rank_count + 0.5) - 1.5*search_age - was_visited``.  Runtime state is
session-local: the current response is unbounded, while history and visited
URL LRUs are bounded independently.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .mapper import write_json_atomic
from .traces import SearchResult


PATTERN_ARTIFACT_SCHEMA = "paste_repro.rank_recency_pattern_predictor"
PATTERN_ARTIFACT_VERSION = 2
PATTERN_POLICY_VERSION = "rank-recency-visited-cache-gate-v2"
FROZEN_TOP_K = 5
FROZEN_HISTORY_CAPACITY = 64
FROZEN_VISITED_CAPACITY = 64
FROZEN_MAX_HISTORY_SEARCH_AGE = 2
FROZEN_SMOOTHING = 0.5
FROZEN_SEARCH_AGE_PENALTY = 1.5
FROZEN_VISITED_PENALTY = 1.0

# Compatibility aliases remain public, but every value is now a policy-v2
# invariant rather than a tunable default.
DEFAULT_HISTORY_CAPACITY = FROZEN_HISTORY_CAPACITY
DEFAULT_VISITED_CAPACITY = FROZEN_VISITED_CAPACITY


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_nonnegative(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{field} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _exact_url(value: Any, field: str) -> str:
    """Validate without stripping, decoding, canonicalizing, or case-folding."""
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise ValueError(f"{field} must be an exact absolute HTTP(S) URL")
    return value


@dataclass(frozen=True)
class GateAbstainRule:
    name: str
    min_query_count: int
    consecutive_search_streak: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("gate rule name must be non-empty")
        _positive_int(self.min_query_count, "min_query_count")
        _positive_int(self.consecutive_search_streak, "consecutive_search_streak")

    def matches(self, query_count: int, search_streak: int) -> bool:
        return (query_count >= self.min_query_count
                and search_streak == self.consecutive_search_streak)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "min_query_count": self.min_query_count,
                "consecutive_search_streak": self.consecutive_search_streak}


FROZEN_GATE_RULES = (GateAbstainRule(
    "many_queries_on_second_consecutive_search", 10, 2
),)
DEFAULT_GATE_RULES = FROZEN_GATE_RULES


@dataclass(frozen=True)
class GateDecision:
    admitted: bool
    reason: str
    matched_rule: str | None
    query_count: int
    consecutive_search_streak: int

    def to_dict(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "reason": self.reason,
                "matched_rule": self.matched_rule, "query_count": self.query_count,
                "consecutive_search_streak": self.consecutive_search_streak,
                "unsupported_pattern_default": "admit"}


@dataclass(frozen=True)
class PatternCandidate:
    url: str
    score: float
    smoothed_rank_probability: float
    rank_count: int
    search_age: int
    was_visited: bool
    current: bool
    appearances: int
    source_rank: int
    source_ordinal: int
    source_query_index: int
    source_search_sequence: int
    source_call_index: int | None
    source_line_number: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "score": self.score,
            "score_terms": {
                "log_rank_count_plus_smoothing": math.log(
                    self.rank_count + FROZEN_SMOOTHING
                ),
                "search_age": self.search_age,
                "search_age_penalty": (
                    FROZEN_SEARCH_AGE_PENALTY * self.search_age
                ),
                "was_visited": self.was_visited,
                "visited_penalty": FROZEN_VISITED_PENALTY * int(self.was_visited),
            },
            "smoothed_rank_probability": self.smoothed_rank_probability,
            "rank_count": self.rank_count, "current": self.current,
            "appearances": self.appearances, "source_rank": self.source_rank,
            "source_ordinal": self.source_ordinal,
            "source_query_index": self.source_query_index,
            "source_search_sequence": self.source_search_sequence,
            "source_call_index": self.source_call_index,
            "source_line_number": self.source_line_number,
        }


@dataclass(frozen=True)
class PatternDecision:
    policy: str
    session_id: str
    search_sequence: int
    gate: GateDecision
    predictions: tuple[PatternCandidate, ...]
    ranked_top_k: tuple[PatternCandidate, ...]
    candidate_count: int
    cache: Mapping[str, Any]

    @property
    def prediction_urls(self) -> tuple[str, ...]:
        return tuple(item.url for item in self.predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy, "session_id": self.session_id,
            "search_sequence": self.search_sequence, "gate": self.gate.to_dict(),
            "prediction_urls": list(self.prediction_urls),
            "predictions": [item.to_dict() for item in self.predictions],
            "ranked_top_k": [item.to_dict() for item in self.ranked_top_k],
            "candidate_count": self.candidate_count, "cache": dict(self.cache),
        }


@dataclass(frozen=True)
class _HistoryEntry:
    result: SearchResult
    search_sequence: int
    appearances: int
    source_call_index: int | None
    source_line_number: int | None


class RankRecencyPatternPredictor:
    """Immutable rank-count table and frozen gate configuration."""

    def __init__(
        self,
        rank_counts: Mapping[int | str, int],
        *,
        gate_rules: Iterable[GateAbstainRule] = FROZEN_GATE_RULES,
        top_k: int = FROZEN_TOP_K,
        history_capacity: int = DEFAULT_HISTORY_CAPACITY,
        visited_capacity: int = DEFAULT_VISITED_CAPACITY,
        max_history_search_age: int = FROZEN_MAX_HISTORY_SEARCH_AGE,
        smoothing: float = FROZEN_SMOOTHING,
        search_age_penalty: float = FROZEN_SEARCH_AGE_PENALTY,
        visited_penalty: float = FROZEN_VISITED_PENALTY,
        artifact_sha256: str = "",
    ) -> None:
        counts: dict[int, int] = {}
        for raw_rank, count in rank_counts.items():
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError) as exc:
                raise ValueError("rank_counts keys must be positive integers") from exc
            if (rank <= 0 or isinstance(count, bool) or not isinstance(count, int)
                    or count < 0):
                raise ValueError("rank_counts must contain non-negative integer counts")
            if count:
                counts[rank] = count
        if not counts:
            raise ValueError("rank_counts must contain a positive count")
        self.rank_counts = MappingProxyType(dict(sorted(counts.items())))
        self.top_k = _positive_int(top_k, "top_k")
        self.history_capacity = _positive_int(history_capacity, "history_capacity")
        self.visited_capacity = _positive_int(visited_capacity, "visited_capacity")
        self.max_history_search_age = _positive_int(max_history_search_age,
                                                    "max_history_search_age")
        self.smoothing = _finite_nonnegative(smoothing, "smoothing", positive=True)
        self.search_age_penalty = _finite_nonnegative(search_age_penalty,
                                                      "search_age_penalty")
        self.visited_penalty = _finite_nonnegative(visited_penalty,
                                                   "visited_penalty")
        if not isinstance(artifact_sha256, str):
            raise ValueError("artifact_sha256 must be a string")
        self.artifact_sha256 = artifact_sha256
        self.gate_rules = tuple(gate_rules)
        self._validate_frozen_policy()
        self._rank_total = sum(self.rank_counts.values())
        self._rank_bucket_count = len(self.rank_counts) + 1

    def _validate_frozen_policy(self) -> None:
        frozen_values = (
            ("top_k", self.top_k, FROZEN_TOP_K),
            ("history_capacity", self.history_capacity, FROZEN_HISTORY_CAPACITY),
            ("visited_capacity", self.visited_capacity, FROZEN_VISITED_CAPACITY),
            (
                "max_history_search_age",
                self.max_history_search_age,
                FROZEN_MAX_HISTORY_SEARCH_AGE,
            ),
            ("smoothing", self.smoothing, FROZEN_SMOOTHING),
            (
                "search_age_penalty",
                self.search_age_penalty,
                FROZEN_SEARCH_AGE_PENALTY,
            ),
            ("visited_penalty", self.visited_penalty, FROZEN_VISITED_PENALTY),
        )
        for field, observed, expected in frozen_values:
            if observed != expected:
                raise ValueError(
                    f"policy v2 {field} is frozen at {expected}, got {observed}"
                )
        if self.gate_rules != FROZEN_GATE_RULES:
            raise ValueError(
                "policy v2 gate_rules are frozen to the unique query_count>=10 "
                "and consecutive_search_streak==2 abstain rule"
            )

    @property
    def policy(self) -> str:
        return PATTERN_POLICY_VERSION

    def start_session(self, session_id: str) -> "RankRecencyPatternSession":
        return RankRecencyPatternSession(self, session_id)

    def smoothed_rank_probability(self, rank: int) -> float:
        return (self.rank_counts.get(max(1, int(rank)), 0) + self.smoothing) / (
            self._rank_total + self.smoothing * self._rank_bucket_count)

    def score(self, rank: int, search_age: int, was_visited: bool) -> float:
        rank_count = self.rank_counts.get(max(1, int(rank)), 0)
        return (math.log(rank_count + self.smoothing)
                - self.search_age_penalty * max(0, search_age)
                - self.visited_penalty * int(was_visited))

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": self.policy, "artifact_sha256": self.artifact_sha256 or None,
            "top_k": self.top_k,
            "rank_counts": {str(k): v for k, v in self.rank_counts.items()},
            "score_formula": "log(rank_count+0.5)-1.5*search_age-1.0*was_visited",
            "smoothing": self.smoothing,
            "search_age_penalty": self.search_age_penalty,
            "visited_penalty": self.visited_penalty,
            "preserve_current_top1": True,
            "history_capacity": self.history_capacity,
            "visited_capacity": self.visited_capacity,
            "current_response_bounded": False,
            "max_history_search_age": self.max_history_search_age,
            "gate_rules": [rule.to_dict() for rule in self.gate_rules],
            "unsupported_gate_pattern_default": "admit",
            "empty_candidate_default": "abstain",
            "exact_raw_url_identity": True, "neural_model": False,
        }

    def to_artifact(self, training_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        # Public attributes are readable for transparent telemetry.  Recheck
        # them here so post-construction mutation cannot serialize a policy
        # that still claims the v2 schema.
        self._validate_frozen_policy()
        manifest = dict(training_manifest or {})
        supplied = manifest.pop("manifest_sha256", None)
        computed = _sha256_json(manifest)
        if supplied not in (None, computed):
            raise ValueError("training manifest checksum mismatch")
        manifest["manifest_sha256"] = computed
        artifact: dict[str, Any] = {
            "schema": PATTERN_ARTIFACT_SCHEMA, "version": PATTERN_ARTIFACT_VERSION,
            "policy": self.policy,
            "rank_counts": {str(k): v for k, v in self.rank_counts.items()},
            "gate_rules": [rule.to_dict() for rule in self.gate_rules],
            "config": {
                "top_k": self.top_k, "history_capacity": self.history_capacity,
                "visited_capacity": self.visited_capacity,
                "max_history_search_age": self.max_history_search_age,
                "smoothing": self.smoothing,
                "search_age_penalty": self.search_age_penalty,
                "visited_penalty": self.visited_penalty,
                "preserve_current_top1": True,
                "unsupported_gate_pattern_default": "admit",
                "empty_candidate_default": "abstain",
            },
            "training_manifest": manifest,
        }
        artifact["artifact_sha256"] = _sha256_json(artifact)
        return artifact

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RankRecencyPatternPredictor":
        if value.get("schema") == PATTERN_ARTIFACT_SCHEMA:
            return cls.from_artifact(value)
        if isinstance(value.get("rank_counts"), Mapping):
            counts = value["rank_counts"]
        elif isinstance(value.get("mapper"), Mapping):
            counts = value["mapper"].get("rank_counts")
        else:
            counts = value
        if not isinstance(counts, Mapping):
            raise ValueError("mapping must contain rank_counts")
        return cls(counts)

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "RankRecencyPatternPredictor":
        raw = dict(artifact)
        checksum = raw.pop("artifact_sha256", None)
        if not isinstance(checksum, str) or checksum != _sha256_json(raw):
            raise ValueError("pattern predictor artifact checksum mismatch")
        if raw.get("schema") != PATTERN_ARTIFACT_SCHEMA:
            raise ValueError("unsupported pattern artifact schema")
        if raw.get("version") != PATTERN_ARTIFACT_VERSION or raw.get("policy") != PATTERN_POLICY_VERSION:
            raise ValueError("unsupported pattern artifact version or policy")
        config = raw.get("config")
        keys = {"top_k", "history_capacity", "visited_capacity",
                "max_history_search_age", "smoothing", "search_age_penalty",
                "visited_penalty", "preserve_current_top1",
                "unsupported_gate_pattern_default", "empty_candidate_default"}
        if not isinstance(config, Mapping) or set(config) != keys:
            raise ValueError("pattern artifact config fields mismatch")
        if (config["preserve_current_top1"] is not True
                or config["unsupported_gate_pattern_default"] != "admit"
                or config["empty_candidate_default"] != "abstain"):
            raise ValueError("pattern artifact frozen policy mismatch")
        rows = raw.get("gate_rules")
        if not isinstance(rows, list):
            raise ValueError("gate_rules must be a list")
        rules = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {
                "name", "min_query_count", "consecutive_search_streak"}:
                raise ValueError(f"gate_rules[{index}] fields mismatch")
            rules.append(GateAbstainRule(row["name"], row["min_query_count"],
                                         row["consecutive_search_streak"]))
        manifest = raw.get("training_manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("training_manifest must be an object")
        unsigned_manifest = dict(manifest)
        manifest_checksum = unsigned_manifest.pop("manifest_sha256", None)
        if not isinstance(manifest_checksum, str) or manifest_checksum != _sha256_json(unsigned_manifest):
            raise ValueError("training manifest checksum mismatch")
        counts = raw.get("rank_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("rank_counts must be an object")
        return cls(counts, gate_rules=rules, artifact_sha256=checksum,
                   top_k=config["top_k"], history_capacity=config["history_capacity"],
                   visited_capacity=config["visited_capacity"],
                   max_history_search_age=config["max_history_search_age"],
                   smoothing=config["smoothing"],
                   search_age_penalty=config["search_age_penalty"],
                   visited_penalty=config["visited_penalty"])


class RankRecencyPatternSession:
    """Mutable state isolated to one causal agent session."""

    def __init__(self, predictor: RankRecencyPatternPredictor, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        self.predictor = predictor
        self.session_id = session_id
        self._history: OrderedDict[str, _HistoryEntry] = OrderedDict()
        self._visited: OrderedDict[str, None] = OrderedDict()
        self._search_sequence = 0
        self._search_streak = 0
        self._closed = False

    @property
    def history_urls(self) -> tuple[str, ...]: return tuple(self._history)
    @property
    def visited_urls(self) -> tuple[str, ...]: return tuple(self._visited)
    @property
    def search_sequence(self) -> int: return self._search_sequence
    @property
    def consecutive_search_streak(self) -> int: return self._search_streak
    @property
    def closed(self) -> bool: return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("pattern predictor session is closed")

    def _candidate(self, entry: _HistoryEntry, *, current: bool,
                   search_age: int) -> PatternCandidate:
        result = entry.result
        visited = result.url in self._visited
        probability = self.predictor.smoothed_rank_probability(result.result_rank)
        return PatternCandidate(
            url=result.url,
            score=self.predictor.score(result.result_rank, search_age, visited),
            smoothed_rank_probability=probability,
            rank_count=self.predictor.rank_counts.get(result.result_rank, 0),
            search_age=search_age, was_visited=visited, current=current,
            appearances=entry.appearances, source_rank=result.result_rank,
            source_ordinal=result.ordinal, source_query_index=result.query_index,
            source_search_sequence=entry.search_sequence,
            source_call_index=entry.source_call_index,
            source_line_number=entry.source_line_number)

    @staticmethod
    def _key(item: PatternCandidate) -> tuple[Any, ...]:
        return (-item.score, not item.current, item.was_visited, item.search_age,
                item.source_ordinal, item.url)

    def observe_search(self, search_results: Sequence[SearchResult], *,
                       query_count: int | None = None,
                       source_call_index: int | None = None,
                       source_line_number: int | None = None) -> PatternDecision:
        """Predict after current search, before observing the next tool label."""
        self._ensure_open()
        if source_call_index is not None and (
            isinstance(source_call_index, bool)
            or not isinstance(source_call_index, int)
            or source_call_index < 0
        ):
            raise ValueError("source_call_index must be non-negative")
        if source_line_number is not None:
            _positive_int(source_line_number, "source_line_number")
        first: OrderedDict[str, SearchResult] = OrderedDict()
        occurrences: Counter[str] = Counter()
        for index, result in enumerate(search_results):
            if not isinstance(result, SearchResult):
                raise ValueError(f"search_results[{index}] must be SearchResult")
            url = _exact_url(result.url, f"search_results[{index}].url")
            first.setdefault(url, result)
            occurrences[url] += 1
        if query_count is None:
            observed_query_count = (max((row.query_index for row in search_results),
                                        default=-1) + 1)
        else:
            observed_query_count = _positive_int(query_count, "query_count")

        self._search_sequence += 1
        self._search_streak += 1
        sequence = self._search_sequence
        history_before = len(self._history)
        visited_before = len(self._visited)
        current_entries: list[_HistoryEntry] = []
        for url, result in first.items():
            prior = self._history.get(url)
            current_entries.append(_HistoryEntry(
                result, sequence, occurrences[url] + (prior.appearances if prior else 0),
                source_call_index, source_line_number))
        current = sorted((self._candidate(item, current=True, search_age=0)
                          for item in current_entries), key=self._key)
        historical = []
        for url, entry in reversed(self._history.items()):
            if url in first:
                continue
            age = sequence - entry.search_sequence
            if age <= self.predictor.max_history_search_age:
                historical.append(self._candidate(entry, current=False,
                                                    search_age=age))
        combined = sorted([*current, *historical], key=self._key)
        if current:
            # Preserve the legacy M0 current-response Top-1 *exactly*.
            # In particular, the recency policy's visited penalty must never
            # change this anchor.  M0 orders by learned rank count, then the
            # original response ordinal and exact raw URL.
            anchor = min(
                current,
                key=lambda item: (
                    -item.rank_count,
                    item.source_ordinal,
                    item.url,
                ),
            )
            combined = [anchor, *(item for item in combined if item.url != anchor.url)]
        ranked = tuple(combined[:self.predictor.top_k])

        matched = next((rule for rule in self.predictor.gate_rules
                        if rule.matches(observed_query_count, self._search_streak)), None)
        if not combined:
            gate = GateDecision(False, "no_candidates", None,
                                observed_query_count, self._search_streak)
        elif matched:
            gate = GateDecision(False, "matched_abstain_pattern", matched.name,
                                observed_query_count, self._search_streak)
        else:
            gate = GateDecision(True, "no_rule_match_admit", None,
                                observed_query_count, self._search_streak)

        evictions = 0
        for url, entry in zip(first, current_entries):
            self._history.pop(url, None)
            self._history[url] = entry
        while len(self._history) > self.predictor.history_capacity:
            self._history.popitem(last=False)
            evictions += 1
        cache = {
            "history_capacity": self.predictor.history_capacity,
            "visited_capacity": self.predictor.visited_capacity,
            "current_response_bounded": False,
            "current_row_count": len(search_results),
            "current_unique_count": len(first),
            "history_size_before": history_before,
            "eligible_history_candidate_count": len(historical),
            "max_history_search_age": self.predictor.max_history_search_age,
            "candidate_union_count": len(combined),
            "history_size_after": len(self._history),
            "history_eviction_count": evictions,
            "visited_size_before": visited_before,
            "visited_size_after": len(self._visited),
            "preserved_current_top1": bool(current),
            "exact_raw_url_identity": True,
        }
        return PatternDecision(self.predictor.policy, self.session_id, sequence, gate,
                               ranked if gate.admitted else (), ranked, len(combined), cache)

    def observe_visit(self, urls: str | Sequence[str]) -> dict[str, Any]:
        """Update visited state only after the authoritative visit commit."""
        self._ensure_open()
        values: Sequence[str] = (urls,) if isinstance(urls, str) else urls
        unique: list[str] = []
        for index, value in enumerate(values):
            url = _exact_url(value, f"visit_urls[{index}]")
            if url not in unique: unique.append(url)
        if not unique: raise ValueError("observe_visit requires at least one URL")
        self._search_streak = 0
        for url in unique:
            self._visited.pop(url, None)
            self._visited[url] = None
        evictions = 0
        while len(self._visited) > self.predictor.visited_capacity:
            self._visited.popitem(last=False); evictions += 1
        return {"tool": "visit", "url_count": len(unique),
                "visited_size_after": len(self._visited),
                "visited_eviction_count": evictions,
                "consecutive_search_streak": self._search_streak,
                "exact_raw_url_identity": True}

    def observe_other_tool(self, tool_name: str) -> None:
        self._ensure_open()
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be non-empty")
        if tool_name in {"search", "visit"}:
            raise ValueError("use observe_search/observe_visit")
        self._search_streak = 0

    def close(self) -> None:
        self._history.clear(); self._visited.clear(); self._search_streak = 0
        self._closed = True


def save_pattern_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    RankRecencyPatternPredictor.from_artifact(artifact)
    write_json_atomic(path, artifact)


def load_pattern_artifact(path: str | Path) -> tuple[RankRecencyPatternPredictor,
                                                      dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pattern artifact root must be an object")
    return RankRecencyPatternPredictor.from_artifact(raw), raw


PatternVisitPredictor = RankRecencyPatternPredictor
PatternPredictionSession = RankRecencyPatternSession
