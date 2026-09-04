"""Serializable, causal runtime form of the Qwen all-Visit Pattern V2 policy.

The historical all-Visit experiment evaluates a nested whole-session OOF
model, but its replay objects also carry the future label and measured timing.
This module keeps only the fitted model parameters.  At runtime it reconstructs
the candidate set from the currently visible tool result and session-local
history; neither an authoritative future invocation nor its duration is an
input to :meth:`PatternV2Session.predict_after_tool`.

The model preserves the published causal candidate generator and scorer: a
Top-20 candidate pool and the rich-logistic/pairwise geometric blend.  Its
strict launch rule is a fixed probability-ranked Top-10.  This is deliberately
named as a strict variant: the historical ``fixed_top10`` ranked by measured
lead-time utility, which is not an admissible runtime input here.  Tool
admission, execution time, and the persistent result cache belong to the
executor and are not represented in this predictor artifact.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .traces import SearchResult, parse_search_results


SCHEMA = "paste_repro.pattern_v2_all_visit_crossfit_predictor.v1"
DEPLOYABLE_SCHEMA = "paste_repro.pattern_v2_all_visit_deployable_predictor.v1"
POLICY = "pattern_v2_all_visit_blend_strict_probability_top10"
FEATURE_SCHEMA = (
    "trigger_visit",
    "visit_depth_log",
    "current",
    "was_visited",
    "search_age",
    "appearances_log",
    "candidate_count_log",
    "query_count_log",
    "position_scaled",
    "ordinal_scaled",
    "source_query_index_scaled",
    "same_trigger_domain",
    "same_trigger_query_group",
    "same_any_visited_domain",
    "same_any_visited_query_group",
    "same_trigger_source_rank",
    "same_any_visited_source_rank",
    "trigger_source_rank_frequency",
    "visited_source_rank_frequency",
    "ordinal_after_trigger",
    "ordinal_distance_trigger_scaled",
    "domain_candidate_frequency",
    "query_group_candidate_frequency",
    "title_query_jaccard",
    "title_task_jaccard",
    "url_query_jaccard",
    "url_task_jaccard",
    "position_1",
    "position_2",
    "position_3",
    "position_4",
    "position_5",
    "position_6_10",
    "position_11_plus",
    "source_rank_1",
    "source_rank_2",
    "source_rank_3",
    "source_rank_4",
    "source_rank_5",
    "source_rank_6_plus",
    "query_group_0",
    "query_group_1",
    "query_group_2",
    "query_group_3",
    "query_group_4",
    "query_group_5_plus",
    "ordinal_0_4",
    "ordinal_5_9",
    "ordinal_10_19",
    "ordinal_20_39",
    "ordinal_40_plus",
)

HISTORY_CAPACITY = 64
VISITED_CAPACITY = 64
MAX_SEARCH_AGE = 2
RANK_SMOOTHING = 0.5
SEARCH_AGE_PENALTY = 1.5
VISITED_PENALTY = 1.0
CANDIDATE_POOL_SIZE = 20
TOP_K = 10


def canonical_sha256(value: Any) -> str:
    wire = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def crossfit_fold(session_id: str) -> int:
    """The published five-fold whole-session assignment."""

    digest = hashlib.sha256(
        f"pattern-cache-grouped-cv-v1\0{session_id}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % 5


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _number_vector(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != len(FEATURE_SCHEMA):
        raise ValueError(f"{label} has the wrong feature width")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (len(FEATURE_SCHEMA),) or not np.isfinite(result).all():
        raise ValueError(f"{label} contains an invalid value")
    return result


def _table(rows: Any, *, key_width: int, label: str) -> dict[tuple[Any, ...], float]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result: dict[tuple[Any, ...], float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"key", "value"}:
            raise ValueError(f"{label}[{index}] is malformed")
        key = row["key"]
        if not isinstance(key, list) or len(key) != key_width:
            raise ValueError(f"{label}[{index}] has the wrong key width")
        normalized = tuple(key)
        if normalized in result:
            raise ValueError(f"{label} contains a duplicate key")
        result[normalized] = _finite_probability(row["value"], label)
    return result


class _CountRuntime:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        if set(payload) != {"visit_global", "visit_query", "visit_detail"}:
            raise ValueError("count-calibrator fields do not match the runtime schema")
        self.visit_global = _finite_probability(
            payload["visit_global"], "visit_global"
        )
        self.visit_query = _table(
            payload["visit_query"], key_width=1, label="visit_query"
        )
        self.visit_detail = _table(
            payload["visit_detail"], key_width=3, label="visit_detail"
        )

    def visit_probability(self, candidate: "_RuntimeCandidate") -> float:
        query_key = (_query_bucket(candidate.query_count),)
        query_prior = self.visit_query.get(query_key, self.visit_global)
        detail_key = (
            _query_bucket(candidate.query_count),
            _streak_bucket(candidate.search_streak),
            _sequence_bucket(candidate.search_sequence),
        )
        return self.visit_detail.get(detail_key, query_prior)


class _LinearRuntime:
    def __init__(self, payload: Mapping[str, Any], *, kind: str) -> None:
        if kind == "rich":
            expected = {"mean", "scale", "weights"}
        elif kind == "pairwise":
            expected = {"scale", "weights"}
        else:
            raise ValueError("unknown linear runtime kind")
        if set(payload) != expected:
            raise ValueError(f"{kind} model fields do not match the runtime schema")
        self.kind = kind
        self.scale = _number_vector(payload["scale"], f"{kind}.scale")
        if np.any(self.scale <= 0.0):
            raise ValueError(f"{kind}.scale must be positive")
        raw_weights = payload["weights"]
        expected_width = len(FEATURE_SCHEMA) + (1 if kind == "rich" else 0)
        if not isinstance(raw_weights, list) or len(raw_weights) != expected_width:
            raise ValueError(f"{kind}.weights has the wrong width")
        self.weights = np.asarray(raw_weights, dtype=np.float64)
        if not np.isfinite(self.weights).all():
            raise ValueError(f"{kind}.weights contains an invalid value")
        self.mean = (
            _number_vector(payload["mean"], "rich.mean")
            if kind == "rich"
            else None
        )

    def probability(self, features: Sequence[float]) -> float:
        row = np.asarray(features, dtype=np.float64)
        if row.shape != (len(FEATURE_SCHEMA),) or not np.isfinite(row).all():
            raise ValueError("runtime feature vector is invalid")
        if self.kind == "rich":
            assert self.mean is not None
            logit = float(
                self.weights[0]
                + ((row - self.mean) / self.scale) @ self.weights[1:]
            )
        else:
            logit = float((row / self.scale) @ self.weights)
        bounded = max(-30.0, min(30.0, logit))
        return 1.0 / (1.0 + math.exp(-bounded))


class _FoldRuntime:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        expected = {
            "outer_fold",
            "training_session_ids",
            "training_session_ids_sha256",
            "rank_counts",
            "global_count",
            "trigger_count",
            "global_rich",
            "trigger_rich",
            "global_pairwise",
            "trigger_pairwise",
        }
        if set(payload) != expected:
            raise ValueError("fold-model fields do not match the runtime schema")
        self.outer_fold = int(payload["outer_fold"])
        ids = payload["training_session_ids"]
        if (
            not isinstance(ids, list)
            or any(not isinstance(value, str) or not value for value in ids)
            or ids != sorted(set(ids))
            or canonical_sha256(ids) != payload["training_session_ids_sha256"]
        ):
            raise ValueError("fold training-root binding is invalid")
        if any(crossfit_fold(value) == self.outer_fold for value in ids):
            raise ValueError("fold model contains a held-out-fold training root")
        self.training_session_ids = frozenset(ids)
        raw_counts = payload["rank_counts"]
        if not isinstance(raw_counts, Mapping):
            raise ValueError("rank_counts must be an object")
        self.rank_counts = {
            int(rank): int(count)
            for rank, count in raw_counts.items()
            if int(count) > 0
        }
        if not self.rank_counts:
            raise ValueError("rank_counts is empty")
        self.global_count = _CountRuntime(payload["global_count"])
        if not isinstance(payload["trigger_count"], Mapping):
            raise ValueError("trigger_count must be an object")
        self.trigger_count = {
            str(key): _CountRuntime(value)
            for key, value in payload["trigger_count"].items()
        }
        self.global_rich = _LinearRuntime(payload["global_rich"], kind="rich")
        if not isinstance(payload["trigger_rich"], Mapping):
            raise ValueError("trigger_rich must be an object")
        self.trigger_rich = {
            str(key): _LinearRuntime(value, kind="rich")
            for key, value in payload["trigger_rich"].items()
        }
        self.global_pairwise = _LinearRuntime(
            payload["global_pairwise"], kind="pairwise"
        )
        if not isinstance(payload["trigger_pairwise"], Mapping):
            raise ValueError("trigger_pairwise must be an object")
        self.trigger_pairwise = {
            str(key): _LinearRuntime(value, kind="pairwise")
            for key, value in payload["trigger_pairwise"].items()
        }

    def score(
        self, trigger_tool: str, candidate: "_RuntimeCandidate", features: Sequence[float]
    ) -> float:
        count = self.trigger_count.get(trigger_tool, self.global_count)
        rich = self.trigger_rich.get(trigger_tool, self.global_rich)
        pairwise = self.trigger_pairwise.get(trigger_tool, self.global_pairwise)
        visit_probability = count.visit_probability(candidate)
        rich_probability = rich.probability(features)
        pairwise_probability = visit_probability * pairwise.probability(features)
        return math.sqrt(
            max(1e-12, rich_probability) * max(1e-12, pairwise_probability)
        )


class PatternV2CrossFitPredictor:
    """Validated five-model artifact with one held-out model per root fold."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        unsigned = dict(artifact)
        supplied = unsigned.pop("artifact_sha256", None)
        if not isinstance(supplied, str) or canonical_sha256(unsigned) != supplied:
            raise ValueError("Pattern V2 cross-fit artifact checksum mismatch")
        if unsigned.get("schema") != SCHEMA or unsigned.get("policy") != POLICY:
            raise ValueError("unsupported Pattern V2 cross-fit artifact")
        expected_config = {
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "top_k": TOP_K,
            "selector_model": "blend",
            "candidate_ranking": "exact_probability_only_no_duration_input",
            "cache_scope": "session_url_infinite_ttl",
        }
        if unsigned.get("configuration") != expected_config:
            raise ValueError("Pattern V2 runtime configuration is not the frozen point")
        if unsigned.get("feature_schema") != list(FEATURE_SCHEMA):
            raise ValueError("Pattern V2 feature schema mismatch")
        if unsigned.get("uses_heldout_root_labels_per_fold") is not False:
            raise ValueError("Pattern V2 fold models must exclude held-out labels")
        if unsigned.get("evaluation_regime") != "retrospective_crossfit":
            raise ValueError("Pattern V2 artifact must disclose cross-fit evaluation")
        if unsigned.get("uses_other_evaluation_root_labels") is not True:
            raise ValueError("Pattern V2 artifact must disclose cross-root labels")
        if unsigned.get("predictor_uses_trace_timing") is not False:
            raise ValueError("Pattern V2 predictor artifact must not use trace timing")
        folds = unsigned.get("folds")
        if not isinstance(folds, Mapping) or set(folds) != {str(i) for i in range(5)}:
            raise ValueError("Pattern V2 artifact must contain exactly five folds")
        self._folds = {int(key): _FoldRuntime(value) for key, value in folds.items()}
        if any(key != model.outer_fold for key, model in self._folds.items()):
            raise ValueError("Pattern V2 outer-fold labels are inconsistent")
        self.artifact_sha256 = supplied

    @classmethod
    def from_path(cls, path: str | Path) -> "PatternV2CrossFitPredictor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Pattern V2 predictor artifact must be an object")
        return cls(payload)

    def start_session(
        self, *, source_session_id: str, runtime_session_id: str
    ) -> "PatternV2Session":
        fold = crossfit_fold(source_session_id)
        model = self._folds[fold]
        if source_session_id in model.training_session_ids:
            raise ValueError("held-out root appears in its fold training set")
        return PatternV2Session(
            model=model,
            source_session_id=source_session_id,
            runtime_session_id=runtime_session_id,
            predictor_artifact_sha256=self.artifact_sha256,
        )


class PatternV2DeployablePredictor:
    """One frozen model fitted before evaluation roots are opened."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        unsigned = dict(artifact)
        supplied = unsigned.pop("artifact_sha256", None)
        if not isinstance(supplied, str) or canonical_sha256(unsigned) != supplied:
            raise ValueError("Pattern V2 deployable artifact checksum mismatch")
        if unsigned.get("schema") != DEPLOYABLE_SCHEMA or unsigned.get("policy") != POLICY:
            raise ValueError("unsupported Pattern V2 deployable artifact")
        expected_config = {
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "top_k": TOP_K,
            "selector_model": "blend",
            "candidate_ranking": "exact_probability_only_no_duration_input",
            "cache_scope": "session_url_infinite_ttl",
        }
        if unsigned.get("configuration") != expected_config:
            raise ValueError("Pattern V2 runtime configuration is not the frozen point")
        if unsigned.get("feature_schema") != list(FEATURE_SCHEMA):
            raise ValueError("Pattern V2 feature schema mismatch")
        if unsigned.get("evaluation_regime") != "frozen_train_eval":
            raise ValueError("deployable artifact must use frozen train/eval")
        if unsigned.get("claim_scope") != "retrospective_internal_holdout":
            raise ValueError("deployable artifact must disclose retrospective scope")
        if unsigned.get("prior_policy_development_used_evaluation_corpus") is not True:
            raise ValueError("deployable artifact must disclose prior corpus use")
        if unsigned.get("uses_evaluation_root_labels") is not False:
            raise ValueError("deployable artifact may not use evaluation-root labels")
        if unsigned.get("predictor_uses_trace_timing") is not False:
            raise ValueError("deployable predictor artifact must not use trace timing")
        if unsigned.get("training_role") not in {
            "calibration_only",
            "calibration_plus_tuning",
        }:
            raise ValueError("deployable artifact has an invalid training role")
        provenance = unsigned.get("training_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("deployable artifact lacks training provenance")
        raw_model = unsigned.get("model")
        if not isinstance(raw_model, Mapping):
            raise ValueError("deployable artifact lacks a model")
        model = _FoldRuntime(raw_model)
        if model.outer_fold != -1:
            raise ValueError("deployable model must use the non-crossfit fold marker")
        if provenance.get("training_session_ids_sha256") != raw_model.get(
            "training_session_ids_sha256"
        ):
            raise ValueError("deployable training provenance/model binding mismatch")
        if provenance.get("training_session_count") != len(
            model.training_session_ids
        ):
            raise ValueError("deployable training-root count mismatch")
        self._model = model
        self.artifact_sha256 = supplied

    @classmethod
    def from_path(cls, path: str | Path) -> "PatternV2DeployablePredictor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Pattern V2 predictor artifact must be an object")
        return cls(payload)

    def start_session(
        self, *, source_session_id: str, runtime_session_id: str
    ) -> "PatternV2Session":
        if source_session_id in self._model.training_session_ids:
            raise ValueError("evaluation root appears in deployable-model training set")
        return PatternV2Session(
            model=self._model,
            source_session_id=source_session_id,
            runtime_session_id=runtime_session_id,
            predictor_artifact_sha256=self.artifact_sha256,
        )


@dataclass(frozen=True)
class PatternV2Prediction:
    url: str
    confidence: float
    source_position: int
    trigger_tool: str


@dataclass(frozen=True)
class _HistoryEntry:
    result: SearchResult
    search_sequence: int
    appearances: int
    lru_order: int


@dataclass(frozen=True)
class _RuntimeCandidate:
    url: str
    result_rank: int
    ordinal: int
    search_sequence: int
    appearances: int
    search_age: int
    was_visited: bool
    current: bool
    source_query_index: int
    title: str
    query: str
    snippet: str
    position: int
    query_count: int
    search_streak: int


def _query_bucket(value: int) -> str:
    if value <= 1:
        return "q1"
    if value == 2:
        return "q2"
    if value <= 4:
        return "q3-4"
    if value <= 9:
        return "q5-9"
    return "q10+"


def _streak_bucket(value: int) -> str:
    if value == 1:
        return "s1"
    if value == 2:
        return "s2"
    return "s3+"


def _sequence_bucket(value: int) -> str:
    if value == 1:
        return "w1"
    if value == 2:
        return "w2"
    if value <= 4:
        return "w3-4"
    return "w5+"


_ASCII_TOKEN = __import__("re").compile(r"[a-z0-9]+")
_CJK_RUN = __import__("re").compile(r"[\u3400-\u9fff]+")


def _text_tokens(value: str) -> set[str]:
    lowered = value.lower()
    result = set(_ASCII_TOKEN.findall(lowered))
    for run in _CJK_RUN.findall(lowered):
        result.update(run)
        result.update(run[index : index + 2] for index in range(len(run) - 1))
    return result


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _text_tokens(left)
    right_tokens = _text_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _feature_vector(
    *,
    trigger_tool: str,
    visit_depth: int,
    task_text: str,
    all_candidates: Sequence[_RuntimeCandidate],
    candidate: _RuntimeCandidate,
    trigger_urls: Sequence[str],
) -> tuple[float, ...]:
    by_url = {row.url: row for row in all_candidates}
    trigger_rows = [by_url[url] for url in trigger_urls if url in by_url]
    trigger_domains = {urlsplit(url).hostname or "" for url in trigger_urls}
    trigger_query_groups = {row.source_query_index for row in trigger_rows}
    trigger_ordinals = [row.ordinal for row in trigger_rows]
    visited_domains = {
        urlsplit(row.url).hostname or "" for row in all_candidates if row.was_visited
    }
    visited_query_groups = {
        row.source_query_index for row in all_candidates if row.was_visited
    }
    trigger_source_ranks = [row.result_rank for row in trigger_rows]
    visited_source_ranks = [
        row.result_rank for row in all_candidates if row.was_visited
    ]
    domain = urlsplit(candidate.url).hostname or ""
    domain_count = sum(
        (urlsplit(row.url).hostname or "") == domain for row in all_candidates
    )
    query_group_count = sum(
        row.source_query_index == candidate.source_query_index
        for row in all_candidates
    )
    if trigger_ordinals:
        ordinal_distance = min(
            abs(candidate.ordinal - value) for value in trigger_ordinals
        )
        ordinal_after = candidate.ordinal > max(trigger_ordinals)
    else:
        ordinal_distance = len(all_candidates)
        ordinal_after = False
    title_context = " ".join((candidate.title, candidate.snippet))
    url_context = " ".join(
        (urlsplit(candidate.url).hostname or "", urlsplit(candidate.url).path)
    )
    position = candidate.position
    position_bins = (
        position == 1,
        position == 2,
        position == 3,
        position == 4,
        position == 5,
        6 <= position <= 10,
        position >= 11,
    )
    rank_bins = tuple(candidate.result_rank == rank for rank in range(1, 6)) + (
        candidate.result_rank >= 6,
    )
    query_bins = tuple(candidate.source_query_index == index for index in range(5)) + (
        candidate.source_query_index >= 5,
    )
    ordinal_bins = (
        candidate.ordinal <= 4,
        5 <= candidate.ordinal <= 9,
        10 <= candidate.ordinal <= 19,
        20 <= candidate.ordinal <= 39,
        candidate.ordinal >= 40,
    )
    values = (
        trigger_tool == "visit",
        math.log1p(visit_depth),
        candidate.current,
        candidate.was_visited,
        candidate.search_age,
        math.log1p(candidate.appearances),
        math.log1p(len(all_candidates)),
        math.log1p(candidate.query_count),
        position / 20.0,
        candidate.ordinal / max(1, len(all_candidates) - 1),
        candidate.source_query_index / max(1, candidate.query_count - 1),
        domain in trigger_domains,
        candidate.source_query_index in trigger_query_groups,
        domain in visited_domains,
        candidate.source_query_index in visited_query_groups,
        candidate.result_rank in trigger_source_ranks,
        candidate.result_rank in visited_source_ranks,
        trigger_source_ranks.count(candidate.result_rank)
        / max(1, len(trigger_source_ranks)),
        visited_source_ranks.count(candidate.result_rank)
        / max(1, len(visited_source_ranks)),
        ordinal_after,
        ordinal_distance / max(1, len(all_candidates)),
        domain_count / max(1, len(all_candidates)),
        query_group_count / max(1, len(all_candidates)),
        _token_jaccard(title_context, candidate.query),
        _token_jaccard(title_context, task_text),
        _token_jaccard(url_context, candidate.query),
        _token_jaccard(url_context, task_text),
        *position_bins,
        *rank_bins,
        *query_bins,
        *ordinal_bins,
    )
    if len(values) != len(FEATURE_SCHEMA):
        raise RuntimeError("Pattern V2 runtime feature width mismatch")
    return tuple(float(value) for value in values)


def _latest_visible_tool_response(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content", "")
        if (
            message.get("role") == "user"
            and isinstance(content, str)
            and "<tool_response>" in content
        ):
            return content
    return ""


def _task_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        content = message.get("content", "")
        if (
            message.get("role") == "user"
            and isinstance(content, str)
            and content
            and "<tool_response>" not in content
        ):
            return content
    return ""


def _search_queries(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    raw = arguments.get("query")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(value for value in raw if isinstance(value, str))
    return ()


def _visit_urls(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    raw = arguments.get("url")
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, list):
        return tuple(
            dict.fromkeys(
                value for value in raw if isinstance(value, str) and value
            )
        )
    return ()


class PatternV2Session:
    """Per-session causal state; one call is made after each completed tool."""

    def __init__(
        self,
        *,
        model: _FoldRuntime,
        source_session_id: str,
        runtime_session_id: str,
        predictor_artifact_sha256: str,
    ) -> None:
        self._model = model
        self.source_session_id = str(source_session_id)
        self.runtime_session_id = str(runtime_session_id)
        self.predictor_artifact_sha256 = str(predictor_artifact_sha256)
        self._history: OrderedDict[str, _HistoryEntry] = OrderedDict()
        self._visited: OrderedDict[str, None] = OrderedDict()
        self._search_sequence = 0
        self._search_streak = 0
        self._visit_depth = 0
        self._query_count = 1
        self._previous_tool: str | None = None
        self._lru_order = 0
        self._task_text = ""
        self._decision_count = 0

    def _rank_score(self, candidate: _RuntimeCandidate) -> float:
        count = self._model.rank_counts.get(candidate.result_rank, 0)
        return (
            math.log(count + RANK_SMOOTHING)
            - SEARCH_AGE_PENALTY * candidate.search_age
            - VISITED_PENALTY * int(candidate.was_visited)
        )

    def _snapshot_history(self) -> list[_RuntimeCandidate]:
        rows: list[_RuntimeCandidate] = []
        for url, entry in reversed(self._history.items()):
            age = self._search_sequence - entry.search_sequence
            if age > MAX_SEARCH_AGE:
                continue
            rows.append(
                _RuntimeCandidate(
                    url=url,
                    result_rank=entry.result.result_rank,
                    ordinal=entry.result.ordinal,
                    # CandidatePattern.search_sequence is a decision-level
                    # feature.  ``search_age`` separately carries the source
                    # result's age.
                    search_sequence=max(1, self._search_sequence),
                    appearances=entry.appearances,
                    search_age=age,
                    was_visited=url in self._visited,
                    current=entry.search_sequence == self._search_sequence,
                    source_query_index=entry.result.query_index,
                    title=entry.result.title,
                    query=entry.result.query,
                    snippet=entry.result.snippet,
                    position=0,
                    query_count=self._query_count,
                    search_streak=max(1, self._search_streak),
                )
            )
        return rows

    def _observe_search(
        self,
        *,
        arguments: Mapping[str, Any],
        current_messages: Sequence[Mapping[str, Any]],
    ) -> list[_RuntimeCandidate]:
        self._search_streak = (
            self._search_streak + 1 if self._previous_tool == "search" else 1
        )
        self._search_sequence += 1
        self._visit_depth = 0
        queries = _search_queries(arguments)
        self._query_count = len(queries)
        response = _latest_visible_tool_response(current_messages)
        current_results = (
            parse_search_results(response, queries=queries) if response else ()
        )
        if self._query_count == 0 and current_results:
            self._query_count = max(row.query_index for row in current_results) + 1
        self._query_count = max(1, self._query_count)
        first: OrderedDict[str, SearchResult] = OrderedDict()
        occurrences: Counter[str] = Counter()
        for result in current_results:
            first.setdefault(result.url, result)
            occurrences[result.url] += 1
        snapshot: list[_RuntimeCandidate] = []
        current_entries: list[tuple[str, _HistoryEntry]] = []
        for url, result in first.items():
            prior = self._history.get(url)
            self._lru_order += 1
            entry = _HistoryEntry(
                result=result,
                search_sequence=self._search_sequence,
                appearances=occurrences[url] + (prior.appearances if prior else 0),
                lru_order=self._lru_order,
            )
            current_entries.append((url, entry))
            snapshot.append(
                _RuntimeCandidate(
                    url=url,
                    result_rank=result.result_rank,
                    ordinal=result.ordinal,
                    search_sequence=self._search_sequence,
                    appearances=entry.appearances,
                    search_age=0,
                    was_visited=url in self._visited,
                    current=True,
                    source_query_index=result.query_index,
                    title=result.title,
                    query=result.query,
                    snippet=result.snippet,
                    position=0,
                    query_count=self._query_count,
                    search_streak=max(1, self._search_streak),
                )
            )
        for url, entry in reversed(self._history.items()):
            if url in first:
                continue
            age = self._search_sequence - entry.search_sequence
            if age > MAX_SEARCH_AGE:
                continue
            snapshot.append(
                _RuntimeCandidate(
                    url=url,
                    result_rank=entry.result.result_rank,
                    ordinal=entry.result.ordinal,
                    search_sequence=max(1, self._search_sequence),
                    appearances=entry.appearances,
                    search_age=age,
                    was_visited=url in self._visited,
                    current=False,
                    source_query_index=entry.result.query_index,
                    title=entry.result.title,
                    query=entry.result.query,
                    snippet=entry.result.snippet,
                    position=0,
                    query_count=self._query_count,
                    search_streak=max(1, self._search_streak),
                )
            )
        for url, entry in current_entries:
            self._history.pop(url, None)
            self._history[url] = entry
        while len(self._history) > HISTORY_CAPACITY:
            self._history.popitem(last=False)
        return snapshot

    def predict_after_tool(
        self,
        *,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
        current_messages: Sequence[Mapping[str, Any]],
    ) -> tuple[PatternV2Prediction, ...]:
        """Update visible state and emit concrete URLs before the next LLM call."""

        trigger_tool = str(tool_name)
        arguments = dict(tool_arguments)
        if not self._task_text:
            self._task_text = _task_text(current_messages)
        trigger_urls: tuple[str, ...] = ()
        if trigger_tool == "search":
            candidates = self._observe_search(
                arguments=arguments, current_messages=current_messages
            )
        elif trigger_tool == "visit":
            self._visit_depth = (
                self._visit_depth + 1 if self._previous_tool == "visit" else 1
            )
            self._search_streak = 0
            trigger_urls = _visit_urls(arguments)
            for url in trigger_urls:
                if not url.startswith(("http://", "https://")):
                    continue
                self._visited.pop(url, None)
                self._visited[url] = None
            while len(self._visited) > VISITED_CAPACITY:
                self._visited.popitem(last=False)
            candidates = self._snapshot_history()
        else:
            self._search_streak = 0
            self._visit_depth = 0
            candidates = self._snapshot_history()
        self._previous_tool = trigger_tool
        if not candidates:
            self._decision_count += 1
            return ()

        if trigger_tool == "visit":
            ordered = sorted(
                candidates,
                key=lambda row: (
                    row.was_visited,
                    -self._rank_score(row),
                    not row.current,
                    row.search_age,
                    row.ordinal,
                    row.url,
                ),
            )
        else:
            ordered = sorted(
                candidates,
                key=lambda row: (
                    -self._rank_score(row),
                    not row.current,
                    row.was_visited,
                    row.search_age,
                    row.ordinal,
                    row.url,
                ),
            )
            current = [row for row in candidates if row.current]
            if current:
                anchor = min(
                    current,
                    key=lambda row: (
                        -self._model.rank_counts.get(row.result_rank, 0),
                        row.ordinal,
                        row.url,
                    ),
                )
                ordered = [anchor, *(row for row in ordered if row.url != anchor.url)]

        pool: list[_RuntimeCandidate] = []
        for position, row in enumerate(ordered[:CANDIDATE_POOL_SIZE], 1):
            if not row.url.startswith(("http://", "https://")):
                continue
            pool.append(
                _RuntimeCandidate(
                    **{**row.__dict__, "position": position}
                )
            )
        scored: list[PatternV2Prediction] = []
        for candidate in pool:
            features = _feature_vector(
                trigger_tool=trigger_tool,
                visit_depth=self._visit_depth,
                task_text=self._task_text,
                all_candidates=candidates,
                candidate=candidate,
                trigger_urls=trigger_urls,
            )
            confidence = self._model.score(trigger_tool, candidate, features)
            scored.append(
                PatternV2Prediction(
                    url=candidate.url,
                    confidence=confidence,
                    source_position=candidate.position,
                    trigger_tool=trigger_tool,
                )
            )
        self._decision_count += 1
        return tuple(
            sorted(
                scored,
                key=lambda row: (-row.confidence, row.source_position, row.url),
            )[:TOP_K]
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "source_session_id_sha256": hashlib.sha256(
                self.source_session_id.encode("utf-8")
            ).hexdigest(),
            "runtime_session_id_sha256": hashlib.sha256(
                self.runtime_session_id.encode("utf-8")
            ).hexdigest(),
            "outer_fold": self._model.outer_fold,
            "decisions": self._decision_count,
            "search_sequence": self._search_sequence,
            "history_size": len(self._history),
            "visited_size": len(self._visited),
            "predictor_artifact_sha256": self.predictor_artifact_sha256,
        }
