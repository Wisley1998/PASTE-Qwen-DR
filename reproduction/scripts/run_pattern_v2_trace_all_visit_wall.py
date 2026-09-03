#!/usr/bin/env python3
"""Predict every causally reachable visit, including visit continuations.

The original Pattern-v2 trace replay creates a prediction window only for the
strict ``search -> one LLM -> visit`` shape.  This runner keeps the old replay
unchanged and adds a broader causal boundary: after every completed search or
visit that is followed by an LLM turn, it snapshots the search-result cache and
predicts the next tool call.  A ``visit -> LLM(s) -> visit`` continuation is
therefore trained, calibrated, and replayed exactly like a search-triggered
visit, while never consulting the future LLM output or authority label when
building candidates.

Candidate ranking, exact-probability calibration, service estimation, and
cross-window budget thresholds remain whole-session OOF.  Rich lexical,
query-group, previous-visit, and session-local rank-history features feed a
ridge-logistic/pairwise ensemble.  A separate cross-fold allocator can spend an
average start budget unevenly while enforcing an explicit per-window burst cap.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any
from urllib.parse import urlsplit

import numpy as np


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.pattern_predictor import RankRecencyPatternPredictor  # noqa: E402
from paste_repro.speculation_policy import (  # noqa: E402
    CandidatePattern,
    CountPatternCalibrator,
    LabeledCandidatePattern,
)
from paste_repro.traces import (  # noqa: E402
    LLMCall,
    OtherEvent,
    SearchResult,
    SessionTrace,
    ToolCall,
    latest_tool_response,
    load_sessions,
    parse_search_results,
)
from run_pattern_cache_evaluation import (  # noqa: E402
    CACHE_CAPACITY,
    MAX_SEARCH_AGE,
    RankPattern,
    VISITED_CAPACITY,
    cv_fold,
    make_frozen_predictor,
)
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredCandidate,
    ScoredWindow,
    inner_fold,
)
from run_pattern_v2_trace_multi_spec_wall import (  # noqa: E402
    build_session_replays,
    candidate_value,
    select_per_task_candidates,
    session_full_walls,
    summarize_width,
)
from run_pattern_v2_trace_timing_net_benefit import (  # noqa: E402
    DecisionTiming,
    build_oof_service_estimates,
    ratio,
    sha256_file,
)


SCHEMA = "paste_repro.pattern_v2_trace_all_visit_wall.v4"
DEFAULT_TRACES = (
    REPOSITORY_ROOT
    / "traces"
    / "my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42"
)
DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT / "results" / "pattern_v2_trace_all_visit_wall"
)


def unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def trace_llm_scale_metadata(traces: Path) -> dict[str, Any]:
    """Read an optional materialized LLM-timing manifest."""

    manifest_path = traces / "LLM_SCALE_MANIFEST.json"
    if not manifest_path.is_file():
        return {
            "materialized_scale": 1.0,
            "manifest": None,
            "manifest_sha256": None,
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    scale = payload.get("duration_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not 0.0 < float(scale) <= 1.0
    ):
        raise ValueError(f"invalid duration_scale in {manifest_path}")
    return {
        "materialized_scale": float(scale),
        "manifest": str(manifest_path.resolve()),
        "manifest_schema": payload.get("schema"),
        "composition": payload.get("composition"),
        "manifest_sha256": sha256_file(manifest_path),
    }


def executable_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def visit_urls(call: ToolCall) -> tuple[str, ...]:
    raw = call.tool_args.get("url")
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, list):
        return unique_strings(tuple(row for row in raw if isinstance(row, str)))
    return ()


def search_queries(call: ToolCall) -> tuple[str, ...]:
    raw = call.tool_args.get("query")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(row for row in raw if isinstance(row, str))
    return ()


@dataclass(frozen=True)
class CacheEntry:
    result: SearchResult
    search_sequence: int
    appearances: int
    lru_order: int


@dataclass(frozen=True)
class CausalCandidate:
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


@dataclass(frozen=True)
class AllVisitDecision:
    session_id: str
    decision_id: str
    trigger_tool: str
    visit_depth: int
    query_count: int
    search_streak: int
    search_sequence: int
    task_text: str
    trigger_urls: tuple[str, ...]
    candidates: tuple[CausalCandidate, ...]
    outcome: str
    authoritative_urls: tuple[str, ...]
    trigger_event_index: int
    target_tool_event_index: int | None
    lead_llm_event_indices: tuple[int, ...]


@dataclass(frozen=True)
class RawCandidate:
    pattern: CandidatePattern
    exact_match: bool
    rich_features: tuple[float, ...]


@dataclass(frozen=True)
class RawWindow:
    decision_id: str
    session_id: str
    trigger_tool: str
    visit_depth: int
    v2_gate: bool
    next_tool_visit: bool
    targets: tuple[str, ...]
    executable_targets: tuple[str, ...]
    candidates: tuple[RawCandidate, ...]


@dataclass(frozen=True)
class GlobalCacheSessionReplay:
    """One session replay with persistent URL-keyed speculative results."""

    session_id: str
    baseline_full_wall_s: float
    treatment_full_wall_s: float
    baseline_segment_wall_s: float
    treatment_segment_wall_s: float
    baseline_visit_stall_s: float
    gross_saved_visit_stall_s: float
    net_saved_visit_stall_s: float
    authoritative_url_calls: int
    selected_speculations: int
    exact_url_hits: int
    visible_url_hits: int
    policy_selected_candidates: int
    deduplicated_speculative_starts: int
    ready_cache_hits: int
    inflight_cache_hits: int
    inflight_wait_s: float
    earlier_decision_cache_hits: int
    incremental_future_cache_hits: int


def following_tool_and_llms(
    events: Sequence[Any], trigger_index: int
) -> tuple[int | None, tuple[int, ...]]:
    """Return the next tool and every causal LLM lead turn before it."""

    llms: list[int] = []
    for index in range(trigger_index + 1, len(events)):
        event = events[index]
        if isinstance(event, ToolCall):
            return index, tuple(llms)
        if isinstance(event, LLMCall):
            llms.append(index)
    return None, tuple(llms)


def causal_task_text(session: SessionTrace) -> str:
    """Return only user text already present before any prediction window."""

    for event in session.events:
        if not isinstance(event, LLMCall):
            continue
        for message in event.messages:
            content = message.get("content", "")
            if (
                message.get("role") == "user"
                and isinstance(content, str)
                and content
                and "<tool_response>" not in content
            ):
                return content
    return ""


def _snapshot_from_history(
    history: OrderedDict[str, CacheEntry],
    visited: OrderedDict[str, None],
    *,
    search_sequence: int,
) -> tuple[CausalCandidate, ...]:
    rows: list[CausalCandidate] = []
    for url, entry in reversed(history.items()):
        age = search_sequence - entry.search_sequence
        if age > MAX_SEARCH_AGE:
            continue
        rows.append(
            CausalCandidate(
                url=url,
                result_rank=entry.result.result_rank,
                ordinal=entry.result.ordinal,
                search_sequence=entry.search_sequence,
                appearances=entry.appearances,
                search_age=age,
                was_visited=url in visited,
                current=entry.search_sequence == search_sequence,
                source_query_index=entry.result.query_index,
                title=entry.result.title,
                query=entry.result.query,
                snippet=entry.result.snippet,
            )
        )
    return tuple(rows)


def extract_all_visit_decisions(
    sessions: Sequence[SessionTrace],
) -> tuple[AllVisitDecision, ...]:
    """Extract one causal next-tool window after every measurable tool result."""

    decisions: list[AllVisitDecision] = []
    for session in sessions:
        task_text = causal_task_text(session)
        history: OrderedDict[str, CacheEntry] = OrderedDict()
        visited: OrderedDict[str, None] = OrderedDict()
        search_sequence = 0
        search_streak = 0
        previous_tool: str | None = None
        query_count = 1
        visit_depth = 0
        lru_order = 0

        for trigger_index, event in enumerate(session.events):
            if not isinstance(event, ToolCall):
                continue
            target_index, llm_indices = following_tool_and_llms(
                session.events, trigger_index
            )

            if event.tool_name == "search":
                search_streak = search_streak + 1 if previous_tool == "search" else 1
                search_sequence += 1
                visit_depth = 0
                queries = search_queries(event)
                query_count = len(queries)
                current_results: tuple[SearchResult, ...] = ()
                if llm_indices:
                    decision_llm = session.events[llm_indices[0]]
                    assert isinstance(decision_llm, LLMCall)
                    current_results = parse_search_results(
                        latest_tool_response(decision_llm), queries=queries
                    )
                if query_count == 0 and current_results:
                    query_count = (
                        max(row.query_index for row in current_results) + 1
                    )
                query_count = max(1, query_count)

                first: OrderedDict[str, SearchResult] = OrderedDict()
                occurrences: Counter[str] = Counter()
                for result in current_results:
                    first.setdefault(result.url, result)
                    occurrences[result.url] += 1

                snapshot: list[CausalCandidate] = []
                current_entries: list[tuple[str, CacheEntry]] = []
                for url, result in first.items():
                    prior = history.get(url)
                    lru_order += 1
                    entry = CacheEntry(
                        result=result,
                        search_sequence=search_sequence,
                        appearances=occurrences[url] + (
                            prior.appearances if prior is not None else 0
                        ),
                        lru_order=lru_order,
                    )
                    current_entries.append((url, entry))
                    snapshot.append(
                        CausalCandidate(
                            url=url,
                            result_rank=result.result_rank,
                            ordinal=result.ordinal,
                            search_sequence=search_sequence,
                            appearances=entry.appearances,
                            search_age=0,
                            was_visited=url in visited,
                            current=True,
                            source_query_index=result.query_index,
                            title=result.title,
                            query=result.query,
                            snippet=result.snippet,
                        )
                    )
                for url, entry in reversed(history.items()):
                    if url in first:
                        continue
                    age = search_sequence - entry.search_sequence
                    if age > MAX_SEARCH_AGE:
                        continue
                    snapshot.append(
                        CausalCandidate(
                            url=url,
                            result_rank=entry.result.result_rank,
                            ordinal=entry.result.ordinal,
                            search_sequence=entry.search_sequence,
                            appearances=entry.appearances,
                            search_age=age,
                            was_visited=url in visited,
                            current=False,
                            source_query_index=entry.result.query_index,
                            title=entry.result.title,
                            query=entry.result.query,
                            snippet=entry.result.snippet,
                        )
                    )
                for url, entry in current_entries:
                    history.pop(url, None)
                    history[url] = entry
                while len(history) > CACHE_CAPACITY:
                    history.popitem(last=False)
                candidates = tuple(snapshot)

            elif event.tool_name == "visit":
                visit_depth = visit_depth + 1 if previous_tool == "visit" else 1
                search_streak = 0
                for url in visit_urls(event):
                    if not executable_url(url):
                        continue
                    visited.pop(url, None)
                    visited[url] = None
                while len(visited) > VISITED_CAPACITY:
                    visited.popitem(last=False)
                candidates = _snapshot_from_history(
                    history, visited, search_sequence=search_sequence
                )
            else:
                search_streak = 0
                visit_depth = 0
                candidates = _snapshot_from_history(
                    history, visited, search_sequence=search_sequence
                )

            previous_tool = event.tool_name
            # No following LLM means the tool is terminal.  It cannot launch a
            # future speculative call and is intentionally not a window.
            if not llm_indices:
                continue
            target = (
                session.events[target_index] if target_index is not None else None
            )
            outcome = target.tool_name if isinstance(target, ToolCall) else "no_next_tool"
            targets = visit_urls(target) if outcome == "visit" else ()
            decisions.append(
                AllVisitDecision(
                    session_id=session.session_id,
                    decision_id=(
                        f"{session.session_id}:after-{event.tool_name}-line-"
                        f"{event.line_number}:{len(decisions)}"
                    ),
                    trigger_tool=event.tool_name,
                    visit_depth=visit_depth,
                    query_count=query_count,
                    search_streak=max(1, search_streak),
                    search_sequence=max(1, search_sequence),
                    task_text=task_text,
                    trigger_urls=(
                        visit_urls(event) if event.tool_name == "visit" else ()
                    ),
                    candidates=candidates,
                    outcome=outcome,
                    authoritative_urls=unique_strings(targets),
                    trigger_event_index=trigger_index,
                    target_tool_event_index=target_index,
                    lead_llm_event_indices=llm_indices,
                )
            )
    return tuple(decisions)


def fit_generalized_rank_pattern(
    decisions: Sequence[AllVisitDecision], fit_ids: set[str]
) -> RankPattern:
    counts: Counter[int] = Counter()
    for decision in decisions:
        if decision.session_id not in fit_ids or decision.outcome != "visit":
            continue
        by_url = {candidate.url: candidate for candidate in decision.candidates}
        for url in decision.authoritative_urls:
            candidate = by_url.get(url)
            if candidate is not None:
                counts[candidate.result_rank] += 1
    normalized = dict(sorted((rank, count) for rank, count in counts.items() if count))
    if not normalized:
        raise RuntimeError("generalized rank fit has no covered visit targets")
    return RankPattern(normalized, sum(normalized.values()))


def rank_candidates(
    decision: AllVisitDecision,
    predictor: RankRecencyPatternPredictor,
    *,
    candidate_pool_size: int,
) -> tuple[CausalCandidate, ...]:
    scored = [(predictor.score(row.result_rank, row.search_age, row.was_visited), row)
              for row in decision.candidates]
    if decision.trigger_tool == "visit":
        # A continuation overwhelmingly moves to a not-yet-consumed search
        # result.  Previously visited URLs remain as fallback for true repeats.
        scored.sort(
            key=lambda item: (
                item[1].was_visited,
                -item[0],
                not item[1].current,
                item[1].search_age,
                item[1].ordinal,
                item[1].url,
            )
        )
    else:
        scored.sort(
            key=lambda item: (
                -item[0],
                not item[1].current,
                item[1].was_visited,
                item[1].search_age,
                item[1].ordinal,
                item[1].url,
            )
        )
        current = [row for row in decision.candidates if row.current]
        if current:
            anchor = min(
                current,
                key=lambda row: (
                    -predictor.rank_counts.get(row.result_rank, 0),
                    row.ordinal,
                    row.url,
                ),
            )
            anchor_score = predictor.score(
                anchor.result_rank, anchor.search_age, anchor.was_visited
            )
            scored = [
                (anchor_score, anchor),
                *(item for item in scored if item[1].url != anchor.url),
            ]
    return tuple(row for _, row in scored[:candidate_pool_size])


_ASCII_TOKEN = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


def text_tokens(value: str) -> set[str]:
    lowered = value.lower()
    result = set(_ASCII_TOKEN.findall(lowered))
    for run in _CJK_RUN.findall(lowered):
        result.update(run)
        result.update(run[index : index + 2] for index in range(len(run) - 1))
    return result


def token_jaccard(left: str, right: str) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


RICH_FEATURE_NAMES = (
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


def rich_candidate_features(
    decision: AllVisitDecision,
    candidate: CausalCandidate,
    *,
    position: int,
) -> tuple[float, ...]:
    by_url = {row.url: row for row in decision.candidates}
    trigger_rows = [
        by_url[url] for url in decision.trigger_urls if url in by_url
    ]
    trigger_domains = {
        urlsplit(url).hostname or "" for url in decision.trigger_urls
    }
    trigger_query_groups = {row.source_query_index for row in trigger_rows}
    trigger_ordinals = [row.ordinal for row in trigger_rows]
    visited_domains = {
        urlsplit(row.url).hostname or ""
        for row in decision.candidates
        if row.was_visited
    }
    visited_query_groups = {
        row.source_query_index
        for row in decision.candidates
        if row.was_visited
    }
    trigger_source_ranks = [row.result_rank for row in trigger_rows]
    visited_source_ranks = [
        row.result_rank for row in decision.candidates if row.was_visited
    ]
    domain = urlsplit(candidate.url).hostname or ""
    domain_count = sum(
        (urlsplit(row.url).hostname or "") == domain
        for row in decision.candidates
    )
    query_group_count = sum(
        row.source_query_index == candidate.source_query_index
        for row in decision.candidates
    )
    if trigger_ordinals:
        ordinal_distance = min(
            abs(candidate.ordinal - value) for value in trigger_ordinals
        )
        ordinal_after = candidate.ordinal > max(trigger_ordinals)
    else:
        ordinal_distance = len(decision.candidates)
        ordinal_after = False
    title_context = " ".join((candidate.title, candidate.snippet))
    url_context = " ".join(
        (urlsplit(candidate.url).hostname or "", urlsplit(candidate.url).path)
    )

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
    query_bins = tuple(
        candidate.source_query_index == index for index in range(5)
    ) + (candidate.source_query_index >= 5,)
    ordinal_bins = (
        candidate.ordinal <= 4,
        5 <= candidate.ordinal <= 9,
        10 <= candidate.ordinal <= 19,
        20 <= candidate.ordinal <= 39,
        candidate.ordinal >= 40,
    )
    values = (
        decision.trigger_tool == "visit",
        math.log1p(decision.visit_depth),
        candidate.current,
        candidate.was_visited,
        candidate.search_age,
        math.log1p(candidate.appearances),
        math.log1p(len(decision.candidates)),
        math.log1p(decision.query_count),
        position / 20.0,
        candidate.ordinal / max(1, len(decision.candidates) - 1),
        candidate.source_query_index / max(1, decision.query_count - 1),
        domain in trigger_domains,
        candidate.source_query_index in trigger_query_groups,
        domain in visited_domains,
        candidate.source_query_index in visited_query_groups,
        candidate.result_rank in trigger_source_ranks,
        candidate.result_rank in visited_source_ranks,
        (
            trigger_source_ranks.count(candidate.result_rank)
            / max(1, len(trigger_source_ranks))
        ),
        (
            visited_source_ranks.count(candidate.result_rank)
            / max(1, len(visited_source_ranks))
        ),
        ordinal_after,
        ordinal_distance / max(1, len(decision.candidates)),
        domain_count / max(1, len(decision.candidates)),
        query_group_count / max(1, len(decision.candidates)),
        token_jaccard(title_context, candidate.query),
        token_jaccard(title_context, decision.task_text),
        token_jaccard(url_context, candidate.query),
        token_jaccard(url_context, decision.task_text),
        *position_bins,
        *rank_bins,
        *query_bins,
        *ordinal_bins,
    )
    if len(values) != len(RICH_FEATURE_NAMES):
        raise RuntimeError("rich feature schema length mismatch")
    return tuple(float(value) for value in values)


class RidgeLogisticCalibrator:
    """Small deterministic NumPy logistic model for causal candidate ranking."""

    def __init__(
        self,
        rows: Sequence[tuple[Sequence[float], bool]],
        *,
        regularization: float = 2.0,
        iterations: int = 30,
    ) -> None:
        if not rows:
            raise ValueError("rich calibrator requires rows")
        matrix = np.asarray([row[0] for row in rows], dtype=np.float64)
        labels = np.asarray([row[1] for row in rows], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(RICH_FEATURE_NAMES):
            raise ValueError("rich calibrator feature shape mismatch")
        self.mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale = np.where(scale > 1e-9, scale, 1.0)
        normalized = (matrix - self.mean) / self.scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        weights = np.zeros(design.shape[1], dtype=np.float64)
        positive_rate = (labels.sum() + 1.0) / (len(labels) + 2.0)
        weights[0] = math.log(positive_rate / (1.0 - positive_rate))
        penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
        penalty[0, 0] = 0.0
        for _ in range(iterations):
            logits = np.clip(design @ weights, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            variance = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
            gradient = design.T @ (probabilities - labels) + penalty @ weights
            hessian = (design.T * variance) @ design + penalty
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if float(np.max(np.abs(step))) < 1e-7:
                break
        self.weights = weights
        self.example_count = len(rows)
        self.positive_count = int(labels.sum())
        self.regularization = regularization

    def probability(self, features: Sequence[float]) -> float:
        row = np.asarray(features, dtype=np.float64)
        if row.shape != (len(RICH_FEATURE_NAMES),):
            raise ValueError("rich probability feature shape mismatch")
        normalized = (row - self.mean) / self.scale
        logit = float(self.weights[0] + normalized @ self.weights[1:])
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

    def summary(self) -> dict[str, Any]:
        coefficients = sorted(
            zip(RICH_FEATURE_NAMES, self.weights[1:], strict=True),
            key=lambda row: abs(float(row[1])),
            reverse=True,
        )
        return {
            "kind": "ridge_logistic_causal_features",
            "example_count": self.example_count,
            "positive_count": self.positive_count,
            "regularization": self.regularization,
            "feature_count": len(RICH_FEATURE_NAMES),
            "largest_standardized_coefficients": [
                {"feature": name, "coefficient": float(value)}
                for name, value in coefficients[:12]
            ],
        }


class PairwiseCausalRanker:
    """Ridge pairwise logistic ranker trained only within causal windows."""

    def __init__(
        self,
        windows: Sequence[RawWindow],
        *,
        regularization: float = 4.0,
        iterations: int = 30,
    ) -> None:
        differences: list[np.ndarray] = []
        labels: list[float] = []
        pair_count = 0
        for window in windows:
            positives = [
                np.asarray(row.rich_features, dtype=np.float64)
                for row in window.candidates
                if row.exact_match
            ]
            negatives = [
                np.asarray(row.rich_features, dtype=np.float64)
                for row in window.candidates
                if not row.exact_match
            ]
            for positive in positives:
                for negative in negatives:
                    difference = positive - negative
                    differences.extend((difference, -difference))
                    labels.extend((1.0, 0.0))
                    pair_count += 1
        if not differences:
            raise ValueError("pairwise ranker requires positive/negative pairs")
        matrix = np.asarray(differences, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        scale = matrix.std(axis=0)
        self.scale = np.where(scale > 1e-9, scale, 1.0)
        design = matrix / self.scale
        weights = np.zeros(design.shape[1], dtype=np.float64)
        identity = np.eye(design.shape[1], dtype=np.float64) * regularization
        for _ in range(iterations):
            logits = np.clip(design @ weights, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            variance = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
            gradient = design.T @ (probabilities - target) + identity @ weights
            hessian = (design.T * variance) @ design + identity
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if float(np.max(np.abs(step))) < 1e-7:
                break
        self.weights = weights
        self.pair_count = pair_count
        self.regularization = regularization

    def score(self, features: Sequence[float]) -> float:
        row = np.asarray(features, dtype=np.float64)
        if row.shape != (len(RICH_FEATURE_NAMES),):
            raise ValueError("pairwise ranker feature shape mismatch")
        return float((row / self.scale) @ self.weights)

    def summary(self) -> dict[str, Any]:
        coefficients = sorted(
            zip(RICH_FEATURE_NAMES, self.weights, strict=True),
            key=lambda row: abs(float(row[1])),
            reverse=True,
        )
        return {
            "kind": "pairwise_ridge_logistic_causal_ranker",
            "pair_count": self.pair_count,
            "regularization": self.regularization,
            "feature_count": len(RICH_FEATURE_NAMES),
            "largest_standardized_coefficients": [
                {"feature": name, "coefficient": float(value)}
                for name, value in coefficients[:12]
            ],
        }


def generate_raw_windows(
    decisions: Sequence[AllVisitDecision],
    *,
    fit_ids: set[str],
    evaluation_ids: set[str],
    candidate_pool_size: int,
    runtime_durations_ms: list[float] | None = None,
) -> list[RawWindow]:
    predictor = make_frozen_predictor(
        fit_generalized_rank_pattern(decisions, fit_ids)
    )
    result: list[RawWindow] = []
    for decision in decisions:
        if decision.session_id not in evaluation_ids:
            continue
        started = time.perf_counter_ns()
        ranked = rank_candidates(
            decision,
            predictor,
            candidate_pool_size=candidate_pool_size,
        )
        target_set = set(decision.authoritative_urls)
        raw_candidates = tuple(
            RawCandidate(
                pattern=CandidatePattern(
                    session_id=decision.session_id,
                    decision_id=decision.decision_id,
                    url=candidate.url,
                    position=position,
                    query_count=decision.query_count,
                    search_streak=decision.search_streak,
                    search_sequence=decision.search_sequence,
                    candidate_count=len(decision.candidates),
                    current_count=sum(row.current for row in decision.candidates),
                    repeated_current=(
                        candidate.current and candidate.appearances >= 2
                    ),
                    source_rank=candidate.result_rank,
                    current=candidate.current,
                    was_visited=candidate.was_visited,
                    search_age=candidate.search_age,
                    appearances=candidate.appearances,
                ),
                exact_match=candidate.url in target_set,
                rich_features=rich_candidate_features(
                    decision, candidate, position=position
                ),
            )
            for position, candidate in enumerate(ranked, 1)
        )
        # The frozen immediate-search abstain rule is not valid for this wider
        # label horizon: a search can still lead to a visit after multiple LLM
        # turns.  Contextual OOF probability and expected value perform the
        # generalized admission instead.
        gate = bool(raw_candidates)
        result.append(
            RawWindow(
                decision_id=decision.decision_id,
                session_id=decision.session_id,
                trigger_tool=decision.trigger_tool,
                visit_depth=decision.visit_depth,
                v2_gate=gate,
                next_tool_visit=decision.outcome == "visit",
                targets=decision.authoritative_urls,
                executable_targets=tuple(
                    url for url in decision.authoritative_urls if executable_url(url)
                ),
                candidates=raw_candidates,
            )
        )
        if runtime_durations_ms is not None:
            runtime_durations_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
    return result


def calibrators_by_trigger(
    windows: Sequence[RawWindow],
) -> tuple[
    CountPatternCalibrator,
    dict[str, CountPatternCalibrator],
    RidgeLogisticCalibrator,
    dict[str, RidgeLogisticCalibrator],
    PairwiseCausalRanker,
    dict[str, PairwiseCausalRanker],
]:
    rows = [
        LabeledCandidatePattern(
            candidate.pattern, window.next_tool_visit, candidate.exact_match
        )
        for window in windows
        for candidate in window.candidates
    ]
    if not rows:
        raise RuntimeError("generalized calibration has no candidate rows")
    global_calibrator = CountPatternCalibrator(rows)
    global_rich = RidgeLogisticCalibrator(
        [
            (candidate.rich_features, candidate.exact_match)
            for window in windows
            for candidate in window.candidates
        ]
    )
    global_ranker = PairwiseCausalRanker(windows)
    contexts: dict[str, CountPatternCalibrator] = {}
    rich_contexts: dict[str, RidgeLogisticCalibrator] = {}
    rank_contexts: dict[str, PairwiseCausalRanker] = {}
    for trigger in sorted({window.trigger_tool for window in windows}):
        trigger_rows = [
            LabeledCandidatePattern(
                candidate.pattern, window.next_tool_visit, candidate.exact_match
            )
            for window in windows
            if window.trigger_tool == trigger
            for candidate in window.candidates
        ]
        if trigger_rows:
            contexts[trigger] = CountPatternCalibrator(trigger_rows)
            rich_contexts[trigger] = RidgeLogisticCalibrator(
                [
                    (candidate.rich_features, candidate.exact_match)
                    for window in windows
                    if window.trigger_tool == trigger
                    for candidate in window.candidates
                ]
            )
            trigger_windows = [
                window for window in windows if window.trigger_tool == trigger
            ]
            if any(
                candidate.exact_match
                for window in trigger_windows
                for candidate in window.candidates
            ):
                rank_contexts[trigger] = PairwiseCausalRanker(trigger_windows)
    return (
        global_calibrator,
        contexts,
        global_rich,
        rich_contexts,
        global_ranker,
        rank_contexts,
    )


def collect_nested_oof_all_visit_windows(
    traces: Path,
    *,
    candidate_pool_size: int = 20,
    selector_model: str = "rich_logistic",
) -> tuple[list[ScoredWindow], dict[str, Any], tuple[AllVisitDecision, ...]]:
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")
    if selector_model not in {"rich_logistic", "pairwise", "blend"}:
        raise ValueError("unknown selector_model")
    sessions = load_sessions(traces)
    decisions = extract_all_visit_decisions(sessions)
    session_ids = {session.session_id for session in sessions}
    result: list[ScoredWindow] = []
    fold_rows: list[dict[str, Any]] = []
    runtime_ms: list[float] = []
    probability_ms: list[float] = []

    for outer in range(5):
        train_ids = {sid for sid in session_ids if cv_fold(sid) != outer}
        validation_ids = session_ids - train_ids
        calibration_windows: list[RawWindow] = []
        for inner in range(4):
            inner_validation = {
                sid for sid in train_ids if inner_fold(sid) == inner
            }
            inner_fit = train_ids - inner_validation
            calibration_windows.extend(
                generate_raw_windows(
                    decisions,
                    fit_ids=inner_fit,
                    evaluation_ids=inner_validation,
                    candidate_pool_size=candidate_pool_size,
                )
            )
        (
            global_calibrator,
            contextual,
            global_rich,
            rich_contextual,
            global_ranker,
            rank_contextual,
        ) = calibrators_by_trigger(calibration_windows)
        validation = generate_raw_windows(
            decisions,
            fit_ids=train_ids,
            evaluation_ids=validation_ids,
            candidate_pool_size=candidate_pool_size,
            runtime_durations_ms=runtime_ms,
        )
        trigger_summary: Counter[str] = Counter()
        fold_hits = 0
        for window in validation:
            calibrator = contextual.get(window.trigger_tool, global_calibrator)
            ranker = rank_contextual.get(window.trigger_tool, global_ranker)
            scored: list[ScoredCandidate] = []
            for candidate in window.candidates:
                started = time.perf_counter_ns()
                visit_probability = calibrator.visit_probability(candidate.pattern)
                rich_probability = rich_contextual.get(
                    window.trigger_tool, global_rich
                ).probability(candidate.rich_features)
                rank_score = max(
                    -30.0,
                    min(30.0, ranker.score(candidate.rich_features)),
                )
                conditional_rank_probability = 1.0 / (
                    1.0 + math.exp(-rank_score)
                )
                pairwise_probability = (
                    visit_probability * conditional_rank_probability
                )
                if selector_model == "rich_logistic":
                    exact_probability = rich_probability
                elif selector_model == "pairwise":
                    exact_probability = pairwise_probability
                else:
                    exact_probability = math.sqrt(
                        max(1e-12, rich_probability)
                        * max(1e-12, pairwise_probability)
                    )
                rank_probability = calibrator.rank_only_probability(
                    candidate.pattern
                )
                probability_ms.append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )
                scored.append(
                    ScoredCandidate(
                        pattern=candidate.pattern,
                        exact_probability=exact_probability,
                        visit_probability=visit_probability,
                        rank_only_probability=rank_probability,
                        exact_match=candidate.exact_match,
                    )
                )
                fold_hits += int(candidate.exact_match)
            mean_targets = ratio(
                sum(
                    len(row.executable_targets)
                    for row in calibration_windows
                    if row.trigger_tool == window.trigger_tool
                    and row.next_tool_visit
                ),
                sum(
                    row.trigger_tool == window.trigger_tool and row.next_tool_visit
                    for row in calibration_windows
                ),
            )
            result.append(
                ScoredWindow(
                    decision_id=window.decision_id,
                    session_id=window.session_id,
                    v2_gate=window.v2_gate,
                    next_tool_visit=window.next_tool_visit,
                    expected_authoritative_calls=(
                        (scored[0].visit_probability if scored else 0.0)
                        * mean_targets
                    ),
                    coarse_expected_authoritative_calls=mean_targets,
                    targets=window.targets,
                    executable_targets=window.executable_targets,
                    candidates=tuple(scored),
                )
            )
            trigger_summary[window.trigger_tool] += 1
        fold_rows.append(
            {
                "outer_fold": outer,
                "train_sessions": len(train_ids),
                "validation_sessions": len(validation_ids),
                "validation_windows": len(validation),
                "validation_candidate_hits": fold_hits,
                "validation_windows_by_trigger": dict(sorted(trigger_summary.items())),
                "global_calibrator": global_calibrator.summary(),
                "trigger_calibrators": {
                    key: value.summary() for key, value in sorted(contextual.items())
                },
                "global_rich_calibrator": global_rich.summary(),
                "trigger_rich_calibrators": {
                    key: value.summary()
                    for key, value in sorted(rich_contextual.items())
                },
                "global_pairwise_ranker": global_ranker.summary(),
                "trigger_pairwise_rankers": {
                    key: value.summary()
                    for key, value in sorted(rank_contextual.items())
                },
            }
        )

    by_id = {decision.decision_id: decision for decision in decisions}
    raw_inventory: Counter[str] = Counter()
    for window in result:
        decision = by_id[window.decision_id]
        raw_inventory[f"windows_{decision.trigger_tool}"] += 1
        if window.next_tool_visit:
            raw_inventory[f"visit_windows_{decision.trigger_tool}"] += 1
            raw_inventory[f"visit_urls_{decision.trigger_tool}"] += len(
                window.executable_targets
            )
        raw_inventory[f"candidate_hits_{decision.trigger_tool}"] += sum(
            candidate.exact_match for candidate in window.candidates
        )

    def duration_summary(values: Sequence[float]) -> dict[str, float | int]:
        return {
            "calls": len(values),
            "total": sum(values),
            "mean": statistics.fmean(values) if values else 0.0,
            "max": max(values, default=0.0),
        }

    return result, {
        "method": "outer-5-fold and inner-4-fold whole-session grouped OOF",
        "window_semantics": "predict after every measurable tool completion",
        "session_count": len(sessions),
        "window_count": len(result),
        "candidate_pool_size": candidate_pool_size,
        "selector_model": selector_model,
        "candidate_count": sum(len(window.candidates) for window in result),
        "candidate_hits": sum(
            candidate.exact_match
            for window in result
            for candidate in window.candidates
        ),
        "inventory": dict(sorted(raw_inventory.items())),
        "folds": fold_rows,
        "runtime_pattern_feature_ms": duration_summary(runtime_ms),
        "runtime_probability_lookup_ms": duration_summary(probability_ms),
    }, decisions


def collect_all_visit_timings(
    traces: Path,
    decisions: Sequence[AllVisitDecision],
    *,
    llm_duration_scale: float,
) -> dict[str, DecisionTiming]:
    sessions = {session.session_id: session for session in load_sessions(traces)}
    timings: dict[str, DecisionTiming] = {}
    for decision in decisions:
        session = sessions[decision.session_id]
        lead_s = sum(
            session.events[index].overlap_window_s
            for index in decision.lead_llm_event_indices
            if isinstance(session.events[index], LLMCall)
        ) * llm_duration_scale
        stall_s = 0.0
        services: tuple[float, ...] = ()
        status = "no_visit"
        if decision.outcome == "visit":
            if decision.target_tool_event_index is None:
                raise RuntimeError("visit label is missing its target event")
            target = session.events[decision.target_tool_event_index]
            if not isinstance(target, ToolCall) or target.tool_name != "visit":
                raise RuntimeError("visit label target is not a visit tool")
            correction = target.timing_correction or {}
            raw_services = correction.get("unit_duration_s")
            if isinstance(raw_services, list) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in raw_services
            ):
                raw_target_urls = visit_urls(target)
                if len(raw_services) != len(raw_target_urls):
                    raise RuntimeError(
                        "corrected visit service count does not match URL count"
                    )
                services = tuple(
                    float(value)
                    for url, value in zip(
                        raw_target_urls, raw_services, strict=True
                    )
                    if executable_url(url)
                )
                stall_s = sum(services)
                status = "corrected_visit_stall"
            else:
                next_llm = next(
                    (
                        event
                        for event in session.events[
                            decision.target_tool_event_index + 1 :
                        ]
                        if isinstance(event, LLMCall)
                    ),
                    None,
                )
                if next_llm is not None:
                    stall_s = max(
                        0.0, next_llm.start_timestamp_s - target.timestamp_s
                    )
                    status = "observed_visit_stall"
                else:
                    marker = next(
                        (
                            event
                            for event in session.events[
                                decision.target_tool_event_index + 1 :
                            ]
                            if isinstance(event, OtherEvent)
                            and event.event_type == "synthetic_tool_completion"
                            and event.payload.get("tool_name") == "visit"
                            and event.payload.get("call_index") == target.call_index
                        ),
                        None,
                    )
                    if marker is not None:
                        stall_s = max(0.0, marker.timestamp_s - target.timestamp_s)
                        status = "synthetic_terminal_visit_stall"
        timings[decision.decision_id] = DecisionTiming(
            decision_id=decision.decision_id,
            session_id=decision.session_id,
            llm_overlap_s=max(0.0, lead_s),
            visit_stall_s=stall_s,
            authoritative_urls=len(decision.authoritative_urls),
            timing_status=status,
            visit_url_service_s=services,
        )
    return timings


def visit_coverage_audit(
    traces: Path, decisions: Sequence[AllVisitDecision]
) -> dict[str, Any]:
    """Prove that every recorded visit has exactly one causal predecessor."""

    sessions = load_sessions(traces)
    all_visit_keys = {
        (session.session_id, index)
        for session in sessions
        for index, event in enumerate(session.events)
        if isinstance(event, ToolCall) and event.tool_name == "visit"
    }
    labeled_keys = [
        (decision.session_id, decision.target_tool_event_index)
        for decision in decisions
        if decision.outcome == "visit"
    ]
    labeled_set = set(labeled_keys)
    if len(labeled_keys) != len(labeled_set):
        raise RuntimeError("one visit was labeled by multiple predecessor windows")
    if labeled_set != all_visit_keys:
        raise RuntimeError("generalized windows do not cover every recorded visit")

    raw_urls = 0
    executable_urls = 0
    executable_service_s = 0.0
    for session in sessions:
        for event in session.events:
            if not isinstance(event, ToolCall) or event.tool_name != "visit":
                continue
            urls = visit_urls(event)
            raw_urls += len(urls)
            executable_urls += sum(executable_url(url) for url in urls)
            correction = event.timing_correction or {}
            raw_services = correction.get("unit_duration_s")
            if isinstance(raw_services, list) and len(raw_services) == len(urls):
                executable_service_s += sum(
                    float(value)
                    for url, value in zip(urls, raw_services, strict=True)
                    if executable_url(url)
                )
    return {
        "recorded_visit_calls": len(all_visit_keys),
        "uniquely_labeled_visit_calls": len(labeled_set),
        "recorded_visit_urls": raw_urls,
        "executable_visit_urls": executable_urls,
        "corrected_executable_visit_service_s": executable_service_s,
    }


def apply_cross_fold_start_budget(
    windows: Sequence[ScoredWindow],
    service_estimates: Mapping[str, Any],
    *,
    average_width: int,
    burst_multiplier: int,
    coordination_cost_s: float,
) -> tuple[list[ScoredWindow], dict[str, Any]]:
    """Allocate starts by OOF score without reading validation labels.

    Each held-out fold receives a threshold estimated from the other folds'
    score distribution.  The mean training budget is ``average_width`` starts
    per window, while one decision may borrow up to ``burst_multiplier`` times
    that width.  This makes the burst-capacity tradeoff explicit.
    """

    if average_width <= 0 or burst_multiplier <= 0:
        raise ValueError("budget widths must be positive")
    burst_cap = average_width * burst_multiplier
    selected: set[tuple[str, int]] = set()
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        training_windows = [
            window for window in windows if cv_fold(window.session_id) != fold
        ]
        training_scores = sorted(
            candidate_value(
                candidate,
                service_estimates[window.decision_id],
                coordination_cost_s,
            )
            for window in training_windows
            for candidate in window.candidates
        )
        budget = min(
            len(training_scores), average_width * len(training_windows)
        )
        threshold = training_scores[-budget] if budget else float("inf")
        validation_selected = 0
        validation_windows = 0
        for window in windows:
            if cv_fold(window.session_id) != fold:
                continue
            validation_windows += 1
            eligible = sorted(
                (
                    (
                        candidate_value(
                            candidate,
                            service_estimates[window.decision_id],
                            coordination_cost_s,
                        ),
                        index,
                    )
                    for index, candidate in enumerate(window.candidates)
                    if candidate_value(
                        candidate,
                        service_estimates[window.decision_id],
                        coordination_cost_s,
                    )
                    >= threshold
                ),
                reverse=True,
            )[:burst_cap]
            selected.update(
                (window.decision_id, index) for _, index in eligible
            )
            validation_selected += len(eligible)
        fold_rows.append(
            {
                "outer_fold": fold,
                "training_windows": len(training_windows),
                "training_target_starts": budget,
                "score_threshold": threshold,
                "validation_windows": validation_windows,
                "validation_selected_starts": validation_selected,
            }
        )

    adjusted = [
        ScoredWindow(
            decision_id=window.decision_id,
            session_id=window.session_id,
            v2_gate=window.v2_gate,
            next_tool_visit=window.next_tool_visit,
            expected_authoritative_calls=window.expected_authoritative_calls,
            coarse_expected_authoritative_calls=(
                window.coarse_expected_authoritative_calls
            ),
            targets=window.targets,
            executable_targets=window.executable_targets,
            candidates=tuple(
                candidate
                if (window.decision_id, index) in selected
                else ScoredCandidate(
                    pattern=candidate.pattern,
                    exact_probability=0.0,
                    visit_probability=candidate.visit_probability,
                    rank_only_probability=candidate.rank_only_probability,
                    exact_match=candidate.exact_match,
                )
                for index, candidate in enumerate(window.candidates)
            ),
        )
        for window in windows
    ]
    starts_per_decision = [
        sum(candidate.exact_probability > 0.0 for candidate in window.candidates)
        for window in adjusted
    ]
    start_histogram = Counter(starts_per_decision)
    ordered_starts = sorted(starts_per_decision)
    p95_index = max(0, math.ceil(0.95 * len(ordered_starts)) - 1)
    return adjusted, {
        "kind": "cross-fold score-quantile start budget",
        "average_width": average_width,
        "burst_multiplier": burst_multiplier,
        "max_starts_per_decision": burst_cap,
        "selected_starts": len(selected),
        "starts_per_decision_mean": (
            statistics.fmean(starts_per_decision) if starts_per_decision else 0.0
        ),
        "starts_per_decision_p95": (
            ordered_starts[p95_index] if ordered_starts else 0
        ),
        "zero_start_decisions": start_histogram.get(0, 0),
        "burst_cap_decisions": start_histogram.get(burst_cap, 0),
        "starts_per_decision_histogram": {
            str(key): value for key, value in sorted(start_histogram.items())
        },
        "folds": fold_rows,
    }


def build_session_global_cache_replays(
    traces: Path,
    windows: Sequence[ScoredWindow],
    decisions: Sequence[AllVisitDecision],
    timings: Mapping[str, DecisionTiming],
    service_estimates: Mapping[str, Any],
    full_walls: Mapping[str, float],
    *,
    per_task_width: int,
    coordination_cost_s: float,
) -> tuple[tuple[GlobalCacheSessionReplay, ...], dict[str, Any]]:
    """Replay an infinite-TTL session URL cache over the treatment timeline.

    Selected speculative URLs are singleflight-deduplicated across decisions.
    A completed result remains reusable for the rest of the session; authority
    may also claim a still-running result and wait only for its remaining tail.
    Only speculative results populate this cache, keeping the comparison scoped
    to speculation rather than adding a separate authority-result cache.

    As in the existing trace replay, a speculative hit uses the corrected
    service realization recorded for its next authority occurrence. This is
    evaluation-time ground truth only and is never consumed by selection.
    """

    sessions = {session.session_id: session for session in load_sessions(traces)}
    windows_by_id = {window.decision_id: window for window in windows}
    decisions_by_session: dict[str, list[AllVisitDecision]] = defaultdict(list)
    future_occurrences: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    selections: dict[str, tuple[ScoredCandidate, ...]] = {}
    for window in windows:
        selections[window.decision_id] = select_per_task_candidates(
            window,
            service_estimates[window.decision_id],
            per_task_width=per_task_width,
            coordination_cost_s=coordination_cost_s,
        )
    for decision in decisions:
        if decision.decision_id not in windows_by_id:
            continue
        decisions_by_session[decision.session_id].append(decision)
        if decision.target_tool_event_index is None:
            continue
        window = windows_by_id[decision.decision_id]
        timing = timings[decision.decision_id]
        if len(window.executable_targets) != len(timing.visit_url_service_s):
            if window.executable_targets:
                raise RuntimeError(
                    "global-cache replay requires per-URL corrected visit service"
                )
            continue
        future_occurrences[decision.session_id].extend(
            (
                decision.target_tool_event_index,
                url,
                float(service_s),
            )
            for url, service_s in zip(
                window.executable_targets,
                timing.visit_url_service_s,
                strict=True,
            )
        )

    result: list[GlobalCacheSessionReplay] = []
    totals: Counter[str] = Counter()
    total_inflight_wait_s = 0.0
    for session_id in sorted(full_walls):
        session = sessions[session_id]
        ordered = sorted(
            decisions_by_session.get(session_id, ()),
            key=lambda decision: decision.trigger_event_index,
        )
        occurrences = sorted(future_occurrences.get(session_id, ()))
        # URL -> (completion timestamp on treatment clock, origin decision).
        # An infinity completion is safe for a URL never used by authority; it
        # still records admission so later prediction epochs do not relaunch it.
        jobs: dict[str, tuple[float, str]] = {}
        cumulative_saved_s = 0.0
        baseline_segment_s = 0.0
        baseline_visit_s = 0.0
        gross_saved_s = 0.0
        authoritative_calls = 0
        physical_starts = 0
        policy_selected = 0
        cache_hits = 0
        visible_hit_decisions = 0
        ready_hits = 0
        inflight_hits = 0
        inflight_wait_s = 0.0
        earlier_hits = 0
        immediate_matches = 0

        for decision in ordered:
            window = windows_by_id[decision.decision_id]
            timing = timings[decision.decision_id]
            selected = selections[decision.decision_id]
            policy_selected += len(selected)
            baseline_segment_s += timing.llm_overlap_s + timing.visit_stall_s
            baseline_visit_s += timing.visit_stall_s
            if not decision.lead_llm_event_indices:
                continue
            first_llm = session.events[decision.lead_llm_event_indices[0]]
            if not isinstance(first_llm, LLMCall):
                raise RuntimeError("global-cache speculation does not start at an LLM")
            speculative_start_s = (
                first_llm.start_timestamp_s - cumulative_saved_s
            )
            immediate_urls = {candidate.pattern.url for candidate in selected}
            for candidate in selected:
                url = candidate.pattern.url
                if url in jobs:
                    continue
                future = next(
                    (
                        (event_index, service_s)
                        for event_index, target_url, service_s in occurrences
                        if event_index > decision.trigger_event_index
                        and target_url == url
                    ),
                    None,
                )
                completion_s = (
                    speculative_start_s + future[1]
                    if future is not None
                    else float("inf")
                )
                jobs[url] = (completion_s, decision.decision_id)
                physical_starts += 1

            if (
                decision.target_tool_event_index is None
                or not window.executable_targets
            ):
                continue
            target = session.events[decision.target_tool_event_index]
            if not isinstance(target, ToolCall) or target.tool_name != "visit":
                raise RuntimeError("global-cache authority target is not a visit")
            authority_start_s = target.timestamp_s - cumulative_saved_s
            authority_now_s = authority_start_s
            decision_hit = False
            for url, service_s in zip(
                window.executable_targets,
                timing.visit_url_service_s,
                strict=True,
            ):
                authoritative_calls += 1
                cached = jobs.get(url)
                if cached is None:
                    authority_now_s += service_s
                    continue
                decision_hit = True
                cache_hits += 1
                completion_s, origin_decision_id = cached
                if url in immediate_urls:
                    immediate_matches += 1
                if origin_decision_id != decision.decision_id:
                    earlier_hits += 1
                if completion_s <= authority_now_s + 1e-12:
                    ready_hits += 1
                else:
                    wait_s = completion_s - authority_now_s
                    inflight_hits += 1
                    inflight_wait_s += wait_s
                    authority_now_s = completion_s
            visible_hit_decisions += int(decision_hit)
            treatment_batch_s = authority_now_s - authority_start_s
            saved_s = timing.visit_stall_s - treatment_batch_s
            if saved_s < -1e-7:
                raise RuntimeError("global-cache replay increased authority visit wall")
            saved_s = max(0.0, saved_s)
            cumulative_saved_s += saved_s
            gross_saved_s += saved_s

        coordination_s = physical_starts * coordination_cost_s
        net_saved_s = gross_saved_s - coordination_s
        incremental_hits = cache_hits - immediate_matches
        replay = GlobalCacheSessionReplay(
            session_id=session_id,
            baseline_full_wall_s=float(full_walls[session_id]),
            treatment_full_wall_s=max(
                0.0, float(full_walls[session_id]) - net_saved_s
            ),
            baseline_segment_wall_s=baseline_segment_s,
            treatment_segment_wall_s=max(
                0.0, baseline_segment_s - net_saved_s
            ),
            baseline_visit_stall_s=baseline_visit_s,
            gross_saved_visit_stall_s=gross_saved_s,
            net_saved_visit_stall_s=net_saved_s,
            authoritative_url_calls=authoritative_calls,
            selected_speculations=physical_starts,
            exact_url_hits=cache_hits,
            visible_url_hits=visible_hit_decisions,
            policy_selected_candidates=policy_selected,
            deduplicated_speculative_starts=policy_selected - physical_starts,
            ready_cache_hits=ready_hits,
            inflight_cache_hits=inflight_hits,
            inflight_wait_s=inflight_wait_s,
            earlier_decision_cache_hits=earlier_hits,
            incremental_future_cache_hits=incremental_hits,
        )
        result.append(replay)
        totals.update(
            {
                "policy_selected_candidates": policy_selected,
                "physical_speculative_starts": physical_starts,
                "deduplicated_speculative_starts": policy_selected
                - physical_starts,
                "cache_hit_occurrences": cache_hits,
                "ready_cache_hits": ready_hits,
                "inflight_cache_hits": inflight_hits,
                "earlier_decision_cache_hits": earlier_hits,
                "immediate_selected_matches": immediate_matches,
                "incremental_future_cache_hits": incremental_hits,
            }
        )
        total_inflight_wait_s += inflight_wait_s

    audit = {
        "kind": "infinite-TTL session URL speculative result cache",
        "scope": "session_url",
        "key": "session_id + executable URL",
        "ttl": "infinite",
        "disk_read_cost_s": 0.0,
        "content_expiration": False,
        "authority_results_populate_cache": False,
        "inflight_singleflight": True,
        "service_realization": (
            "corrected service of the next matching authority occurrence; "
            "evaluation-only and not used by candidate selection"
        ),
        **dict(totals),
        "inflight_wait_s": total_inflight_wait_s,
    }
    return tuple(result), audit


def render_report(payload: Mapping[str, Any]) -> str:
    meta = payload["nested_oof"]
    inventory = meta["inventory"]
    config = payload["configuration"]
    lines = [
        "# Pattern-v2 all-visit causal wall replay",
        "",
        "Prediction windows are created after every measurable search or visit result.",
        "Visit continuations reuse the causal search cache and update visited "
        "state before prediction.",
        f"Selector: `{config['selector_model']}`; allocation: "
        f"`{config['allocation']}`.",
        f"Speculative result cache: `{config['cache_scope']}`.",
        f"Effective LLM duration scale: `{config['effective_llm_duration_scale']}` "
        f"(materialized `{config['materialized_llm_duration_scale']}` × runtime "
        f"`{config['llm_duration_scale']}`).",
        "Wrong-call contention is not modeled; speculative slots remain isolated.",
        "",
        "## Coverage inventory",
        "",
        "| Trigger | Windows | Next visit windows | Visit URLs "
        f"| Top-{meta['candidate_pool_size']} candidate-pool hits |",
        "|---|---:|---:|---:|---:|",
    ]
    for trigger in ("search", "visit"):
        lines.append(
            f"| {trigger} | {inventory.get(f'windows_{trigger}', 0)} "
            f"| {inventory.get(f'visit_windows_{trigger}', 0)} "
            f"| {inventory.get(f'visit_urls_{trigger}', 0)} "
            f"| {inventory.get(f'candidate_hits_{trigger}', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Wall results",
            "",
            "| Budget | Burst cap | C | Selected | No-opt wall | Optimized wall "
            "| Full wall speedup | Mean flow reduction "
            "| Eligible visit reduction | Authority recall | Spec precision "
            "| Call amp. |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for width in payload["width_results"]:
        for row in width["concurrency_results"]:
            lines.append(
                f"| {row['per_task_spec_width']} "
                f"| {row['max_starts_per_decision']} "
                f"| {row['task_concurrency']} "
                f"| {row['selected_speculations']} "
                f"| {row['event_full_baseline_wall_s']:.3f} s "
                f"| {row['event_full_treatment_wall_s']:.3f} s "
                f"| {row['event_full_wall_speedup_fraction']:.2%} "
                f"| {row['mean_task_full_flow_reduction_fraction']:.2%} "
                f"| {row['net_visit_stall_reduction_fraction']:.2%} "
                f"| {row['exact_authority_hit_rate']:.2%} "
                f"| {row['prediction_precision']:.2%} "
                f"| {row['physical_call_amplification']:.3f}x |"
            )
    lines.append("")
    if config["cache_scope"] == "session_url":
        lines.extend(
            [
                "## Persistent speculative cache",
                "",
                "| Budget | Policy selections | Physical starts | Deduplicated "
                "| Cache hits | Ready | In-flight | Earlier-decision hits "
                "| Incremental future hits | Wait tail |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for width in payload["width_results"]:
            cache = payload["cache_rows"][str(width["per_task_spec_width"])]
            lines.append(
                f"| {width['per_task_spec_width']} "
                f"| {cache['policy_selected_candidates']} "
                f"| {cache['physical_speculative_starts']} "
                f"| {cache['deduplicated_speculative_starts']} "
                f"| {cache['cache_hit_occurrences']} "
                f"| {cache['ready_cache_hits']} "
                f"| {cache['inflight_cache_hits']} "
                f"| {cache['earlier_decision_cache_hits']} "
                f"| {cache['incremental_future_cache_hits']} "
                f"| {cache['inflight_wait_s']:.3f} s |"
            )
        lines.extend(
            [
                "",
                "The cache is URL-keyed within one session, has infinite TTL, "
                "zero read cost, and no content expiration. Running jobs are "
                "singleflight-claimed; completed speculative results persist "
                "across later decisions. Authority results do not populate this "
                "cache.",
                "",
            ]
        )
    lines.extend(
        [
            "The no-optimization wall is the same 0.42x-LLM trace replayed without "
            "speculative visits. Authority recall and speculative precision are "
            "fixed replay outcomes, so they remain constant across task concurrency; "
            "concurrency changes only the closed-loop makespan schedule.",
            "Authority multi-URL visits execute serially using corrected per-URL "
            "service durations. Selected speculative visits start concurrently in "
            "isolated slots.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-task-widths", type=int, nargs="+", default=[1, 2, 3, 4, 5]
    )
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--coordination-cost-ms", type=float, default=1.0)
    parser.add_argument("--candidate-pool-size", type=int, default=20)
    parser.add_argument(
        "--selector-model",
        choices=("rich_logistic", "pairwise", "blend"),
        default="rich_logistic",
    )
    parser.add_argument(
        "--allocation",
        choices=("per_decision", "cross_fold_budget"),
        default="per_decision",
    )
    parser.add_argument(
        "--cache-scope",
        choices=("decision", "session_url"),
        default="decision",
    )
    parser.add_argument("--burst-multiplier", type=int, default=2)
    parser.add_argument("--domain-prior-strength", type=float, default=10.0)
    parser.add_argument("--llm-duration-scale", type=float, default=1.0)
    args = parser.parse_args()
    if any(value <= 0 for value in args.per_task_widths):
        parser.error("per-task widths must be positive")
    if any(value <= 0 for value in args.concurrencies):
        parser.error("concurrencies must be positive")
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    if args.coordination_cost_ms < 0.0:
        parser.error("coordination cost must be non-negative")
    if args.candidate_pool_size <= 0:
        parser.error("candidate pool size must be positive")
    if args.burst_multiplier <= 0:
        parser.error("burst multiplier must be positive")
    if args.domain_prior_strength < 0.0:
        parser.error("domain prior strength must be non-negative")
    if not 0.0 < args.llm_duration_scale <= 1.0:
        parser.error("LLM duration scale must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    if args.cache_scope == "session_url" and args.llm_duration_scale != 1.0:
        raise ValueError(
            "session_url cache replay requires timestamp-materialized LLM "
            "timing and --llm-duration-scale 1.0"
        )
    trace_scale = trace_llm_scale_metadata(args.traces)
    windows, nested_oof, decisions = collect_nested_oof_all_visit_windows(
        args.traces,
        candidate_pool_size=args.candidate_pool_size,
        selector_model=args.selector_model,
    )
    timings = collect_all_visit_timings(
        args.traces, decisions, llm_duration_scale=args.llm_duration_scale
    )
    coverage_audit = visit_coverage_audit(args.traces, decisions)
    service_estimates, service_estimator = build_oof_service_estimates(
        windows,
        timings,
        domain_prior_strength=args.domain_prior_strength,
    )
    full_walls = session_full_walls(
        args.traces, llm_duration_scale=args.llm_duration_scale
    )
    width_results = []
    session_rows: dict[str, list[dict[str, Any]]] = {}
    allocation_rows: dict[str, dict[str, Any]] = {}
    cache_rows: dict[str, dict[str, Any]] = {}
    for width in sorted(set(args.per_task_widths)):
        replay_windows = windows
        replay_width = width
        if args.allocation == "cross_fold_budget":
            replay_windows, allocation = apply_cross_fold_start_budget(
                windows,
                service_estimates,
                average_width=width,
                burst_multiplier=args.burst_multiplier,
                coordination_cost_s=args.coordination_cost_ms / 1000.0,
            )
            replay_width = args.candidate_pool_size
            allocation_rows[str(width)] = allocation
        else:
            allocation_rows[str(width)] = {
                "kind": "fixed per-decision width",
                "average_width": width,
                "max_starts_per_decision": width,
            }
        if args.cache_scope == "session_url":
            sessions, cache_audit = build_session_global_cache_replays(
                args.traces,
                replay_windows,
                decisions,
                timings,
                service_estimates,
                full_walls,
                per_task_width=replay_width,
                coordination_cost_s=args.coordination_cost_ms / 1000.0,
            )
            cache_rows[str(width)] = cache_audit
        else:
            sessions = build_session_replays(
                replay_windows,
                timings,
                service_estimates,
                full_walls,
                per_task_width=replay_width,
                coordination_cost_s=args.coordination_cost_ms / 1000.0,
            )
            cache_rows[str(width)] = {
                "kind": "decision-scoped speculative result",
                "scope": "decision",
            }
        session_rows[str(width)] = [asdict(row) for row in sessions]
        concurrency_results = []
        max_starts = int(
            allocation_rows[str(width)]["max_starts_per_decision"]
        )
        for concurrency in args.concurrencies:
            summary = summarize_width(
                sessions,
                per_task_width=width,
                concurrency=concurrency,
                repetitions=args.repetitions,
            )
            summary["allocation"] = args.allocation
            summary["cache_scope"] = args.cache_scope
            summary["average_start_budget"] = width
            summary["max_starts_per_decision"] = max_starts
            summary["isolated_spec_slots_upper_bound"] = (
                min(concurrency, len(sessions)) * max_starts
            )
            concurrency_results.append(summary)
        width_results.append(
            {
                "per_task_spec_width": width,
                "concurrency_results": concurrency_results,
            }
        )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "per_task_widths": sorted(set(args.per_task_widths)),
            "concurrencies": args.concurrencies,
            "repetitions": args.repetitions,
            "coordination_cost_ms": args.coordination_cost_ms,
            "candidate_pool_size": args.candidate_pool_size,
            "selector_model": args.selector_model,
            "allocation": args.allocation,
            "cache_scope": args.cache_scope,
            "burst_multiplier": args.burst_multiplier,
            "domain_prior_strength": args.domain_prior_strength,
            "llm_duration_scale": args.llm_duration_scale,
            "materialized_llm_duration_scale": trace_scale[
                "materialized_scale"
            ],
            "effective_llm_duration_scale": (
                trace_scale["materialized_scale"] * args.llm_duration_scale
            ),
            "llm_timing_manifest": trace_scale["manifest"],
            "llm_timing_manifest_schema": trace_scale.get("manifest_schema"),
            "llm_timing_composition": trace_scale.get("composition"),
            "selection": (
                "per-decision contextual OOF expected-value Top-N"
                if args.allocation == "per_decision"
                else "cross-fold OOF score-threshold budget with burst cap"
            ),
            "prediction_scope": "all measurable search and visit completions",
            "capacity_model": (
                "fixed N isolated slots per task"
                if args.allocation == "per_decision"
                else "cross-fold average start budget with explicit burst cap"
            ),
            "cache_model": (
                "decision-scoped exact result"
                if args.cache_scope == "decision"
                else "infinite-TTL zero-read-cost session URL cache with "
                "in-flight singleflight"
            ),
        },
        "source_sha256": {
            "runner": sha256_file(SCRIPT),
            "multi_spec_wall": sha256_file(
                SCRIPT.parent / "run_pattern_v2_trace_multi_spec_wall.py"
            ),
            "llm_timing_manifest": trace_scale["manifest_sha256"],
        },
        "nested_oof": nested_oof,
        "coverage_audit": coverage_audit,
        "service_estimator": service_estimator,
        "session_rows": session_rows,
        "allocation_rows": allocation_rows,
        "cache_rows": cache_rows,
        "width_results": width_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
