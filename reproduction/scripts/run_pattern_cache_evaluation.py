#!/usr/bin/env python3
"""Develop and one-shot evaluate a causal rank-pattern URL cache policy.

Development is deliberately limited to the historical trace directory.  A
separate ``--evaluate-new`` path restores a checksummed, frozen JSON artifact
and claims its output directory with ``O_EXCL`` before reading any holdout
session.  The policy is intentionally non-neural: displayed-rank counts, a
bounded per-session URL cache, two fixed penalties, and a deterministic
visit/abstain rule.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REPRODUCTION_ROOT.parent
sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from paste_repro.multiturn_collector import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_TYPE,
    TRACE_SCHEMA,
    WORKLOAD_SCHEMA_VERSION,
    FixedWorkload,
    load_fixed_workload,
)
from paste_repro.pattern_predictor import (  # noqa: E402
    PATTERN_ARTIFACT_SCHEMA,
    PATTERN_ARTIFACT_VERSION,
    PATTERN_POLICY_VERSION,
    FROZEN_HISTORY_CAPACITY,
    FROZEN_MAX_HISTORY_SEARCH_AGE,
    FROZEN_SEARCH_AGE_PENALTY,
    FROZEN_SMOOTHING,
    FROZEN_TOP_K,
    FROZEN_VISITED_CAPACITY,
    FROZEN_VISITED_PENALTY,
    RankRecencyPatternPredictor,
    load_pattern_artifact,
    save_pattern_artifact,
)
from paste_repro.tool_prediction import structured_search_results  # noqa: E402
from paste_repro.traces import (  # noqa: E402
    LLMCall,
    OtherEvent,
    SearchResult,
    SessionTrace,
    ToolCall,
    latest_tool_response,
    load_trace,
    load_sessions,
    parse_search_results,
    split_sessions,
)


OUTER_SEED = "paste-repro-v1"
CV_SEED = "pattern-cache-grouped-cv-v1"
BOOTSTRAP_SEED = "pattern-cache-new-holdout-bootstrap-v1"
TOP_KS = (1, 3, 5)
DIAGNOSTIC_TOP_KS = (1, 3, 5, 10, 20)
CACHE_CAPACITY = FROZEN_HISTORY_CAPACITY
VISITED_CAPACITY = FROZEN_VISITED_CAPACITY
MAX_SEARCH_AGE = FROZEN_MAX_HISTORY_SEARCH_AGE
AGE_PENALTY = FROZEN_SEARCH_AGE_PENALTY
VISITED_PENALTY = FROZEN_VISITED_PENALTY
RANK_COUNT_SMOOTHING = FROZEN_SMOOTHING
EXPECTED_DEVELOPMENT_SESSIONS = 100
EXPECTED_NEW_WORKLOAD_ID = "new-whole-session-holdout-v1"
EXPECTED_NEW_WORKLOAD_SHA256 = (
    "88d15dfea2f6e1abbce20086f608bc0f324ef8549a47359b425f60cba0ac7f87"
)
EXPECTED_NEW_SOURCE_COUNT = 30
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_NEW_WORKLOAD = (
    REPRODUCTION_ROOT / "workloads" / "new_whole_session_holdout_v1.json"
)
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "results" / "pattern_cache_development"
ARTIFACT_NAME = "pattern_cache_policy.json"
DEVELOPMENT_METRICS_NAME = "development_metrics.json"
NEW_METRICS_NAME = "new_holdout_metrics.json"
REPORT_NAME = "REPORT.md"
STARTED_NAME = "NEW_HOLDOUT_EVALUATION_STARTED.json"
COMPLETE_NAME = "NEW_HOLDOUT_EVALUATION_COMPLETE.json"


@dataclass(frozen=True)
class CachedCandidate:
    """One exact raw URL visible at or before a search decision."""

    url: str
    result_rank: int
    ordinal: int
    age: int
    visited: bool
    current: bool
    lru_order: int


@dataclass(frozen=True)
class SearchDecision:
    """Causal decision input plus a label used only after prediction."""

    session_id: str
    decision_id: str
    current_results: tuple[SearchResult, ...]
    cache_candidates: tuple[CachedCandidate, ...]
    query_count: int
    consecutive_search_streak: int
    visited_cache_size: int
    prior_tool_updates: tuple[tuple[str, tuple[str, ...]], ...]
    outcome: str
    authoritative_urls: tuple[str, ...]


@dataclass(frozen=True)
class RankPattern:
    rank_counts: dict[int, int]
    total: int


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _visit_urls(call: ToolCall) -> tuple[str, ...]:
    raw = call.tool_args.get("url")
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, list):
        return _unique_strings(tuple(item for item in raw if isinstance(item, str)))
    return ()


def _is_executable_url(url: str) -> bool:
    # Match the runtime's exact, non-normalizing dispatch boundary.
    return url.startswith(("http://", "https://"))


def _queries(call: ToolCall) -> tuple[str, ...]:
    raw = call.tool_args.get("query")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(item for item in raw if isinstance(item, str))
    return ()


def _outcome(events: Sequence[Any], search_index: int) -> tuple[str, tuple[str, ...]]:
    """Read the post-decision label; callers never feed it to prediction."""

    next_index = search_index + 2
    if next_index >= len(events) or not isinstance(events[next_index], ToolCall):
        return "no_next_tool", ()
    next_tool = events[next_index]
    if next_tool.tool_name == "visit":
        return "visit", _visit_urls(next_tool)
    return next_tool.tool_name, ()


def extract_search_decisions(
    sessions: Sequence[SessionTrace],
    *,
    cache_capacity: int = CACHE_CAPACITY,
    visited_capacity: int = VISITED_CAPACITY,
    max_search_age: int = MAX_SEARCH_AGE,
) -> tuple[SearchDecision, ...]:
    """Extract every search decision with a bounded, causal per-session cache.

    Search results come only from the request of the immediately following LLM
    call.  The LLM's generated response and the subsequent tool call are never
    consulted while constructing cache state.  The latter is attached only as
    an evaluation label after the snapshot is complete.
    """

    if cache_capacity <= 0:
        raise ValueError("cache_capacity must be positive")
    if visited_capacity <= 0:
        raise ValueError("visited_capacity must be positive")
    if max_search_age < 0:
        raise ValueError("max_search_age must be non-negative")

    extracted: list[SearchDecision] = []
    for session in sessions:
        # Runtime parity mirror: an independently bounded history LRU and
        # visited LRU.  The current response itself remains unbounded.
        history: OrderedDict[str, tuple[int, int, int, int]] = OrderedDict()
        visited_urls: OrderedDict[str, None] = OrderedDict()
        search_sequence = 0
        lru_serial = 0
        previous_tool_name: str | None = None
        search_streak = 0
        pending_runtime_updates: list[tuple[str, tuple[str, ...]]] = []

        for index, event in enumerate(session.events):
            if not isinstance(event, ToolCall):
                continue
            if event.tool_name == "visit":
                visit_urls = _visit_urls(event)
                pending_runtime_updates.append(("visit", visit_urls))
                for url in visit_urls:
                    if not _is_executable_url(url):
                        continue
                    visited_urls.pop(url, None)
                    visited_urls[url] = None
                while len(visited_urls) > visited_capacity:
                    visited_urls.popitem(last=False)
                previous_tool_name = "visit"
                search_streak = 0
                continue
            if event.tool_name != "search":
                pending_runtime_updates.append((event.tool_name, ()))
                previous_tool_name = event.tool_name
                search_streak = 0
                continue

            search_streak = search_streak + 1 if previous_tool_name == "search" else 1
            previous_tool_name = "search"
            if index + 1 >= len(session.events):
                continue
            decision_llm = session.events[index + 1]
            if not isinstance(decision_llm, LLMCall):
                continue

            queries = _queries(event)
            current_results = parse_search_results(
                latest_tool_response(decision_llm), queries=queries
            )
            search_sequence += 1

            # Runtime keeps every current URL available for this decision,
            # then commits it to the bounded history LRU for future searches.
            current_first: OrderedDict[str, SearchResult] = OrderedDict()
            for result in current_results:
                current_first.setdefault(result.url, result)

            snapshot: list[CachedCandidate] = []
            for result in current_first.values():
                lru_serial += 1
                snapshot.append(
                    CachedCandidate(
                        url=result.url,
                        result_rank=result.result_rank,
                        ordinal=result.ordinal,
                        age=0,
                        visited=result.url in visited_urls,
                        current=True,
                        lru_order=lru_serial,
                    )
                )
            for url, (rank, ordinal, seen_at, order) in reversed(history.items()):
                if url in current_first:
                    continue
                age = search_sequence - seen_at
                if age > max_search_age:
                    continue
                snapshot.append(
                    CachedCandidate(
                        url=url,
                        result_rank=rank,
                        ordinal=ordinal,
                        age=age,
                        visited=url in visited_urls,
                        current=False,
                        lru_order=order,
                    )
                )

            for result in current_first.values():
                # ``lru_serial`` is only a provenance tie-break value; runtime
                # ordering does not depend on it.
                existing = next(item for item in snapshot if item.url == result.url)
                history.pop(result.url, None)
                history[result.url] = (
                    result.result_rank,
                    result.ordinal,
                    search_sequence,
                    existing.lru_order,
                )
            while len(history) > cache_capacity:
                history.popitem(last=False)

            # The label is deliberately read only after every prediction input
            # above has been frozen.
            outcome, targets = _outcome(session.events, index)
            query_count = len(queries)
            if query_count == 0 and current_results:
                query_count = max(result.query_index for result in current_results) + 1
            extracted.append(
                SearchDecision(
                    session_id=session.session_id,
                    decision_id=(
                        f"{session.session_id}:search-line-{event.line_number}:"
                        f"{len(extracted)}"
                    ),
                    current_results=current_results,
                    cache_candidates=tuple(snapshot),
                    query_count=query_count,
                    consecutive_search_streak=search_streak,
                    visited_cache_size=len(visited_urls),
                    prior_tool_updates=tuple(pending_runtime_updates),
                    outcome=outcome,
                    authoritative_urls=_unique_strings(targets),
                )
            )
            pending_runtime_updates.clear()
    return tuple(extracted)


def _validated_tool_result_events(
    session: SessionTrace,
) -> tuple[dict[int, OtherEvent], dict[str, Any]]:
    """Bind collector tool-result commits to the immediately preceding request."""

    committed_by_request_index: dict[int, OtherEvent] = {}
    requested = Counter()
    committed = Counter()
    requested_visit_urls = 0
    committed_visit_urls = 0
    for index, event in enumerate(session.events):
        if isinstance(event, ToolCall):
            requested[event.tool_name] += 1
            if event.tool_name == "visit":
                requested_visit_urls += len(_visit_urls(event))
            continue
        if not isinstance(event, OtherEvent) or event.event_type != "tool_result":
            continue
        if index == 0 or not isinstance(session.events[index - 1], ToolCall):
            raise ValueError(
                f"{session.session_id}: tool_result is not immediately after tool_call"
            )
        request = session.events[index - 1]
        payload = event.payload
        if request.tool_name not in {"search", "visit"}:
            raise ValueError(f"{session.session_id}: unsupported committed tool")
        expected_fields = {
            "event_type",
            "call_index",
            "timestamp",
            "tool_name",
            "commit_status",
            "result_sha256",
            "raw_result",
            "formatted_response",
            "transport",
        }
        if set(payload) != expected_fields:
            raise ValueError(f"{session.session_id}: tool_result fields mismatch")
        if index - 1 in committed_by_request_index:
            raise ValueError(f"{session.session_id}: duplicate tool_result commit")
        if payload.get("call_index") != request.call_index:
            raise ValueError(f"{session.session_id}: tool_result call_index mismatch")
        if payload.get("tool_name") != request.tool_name:
            raise ValueError(f"{session.session_id}: tool_result tool_name mismatch")
        status = payload.get("commit_status")
        if status != "committed":
            raise ValueError(f"{session.session_id}: invalid tool_result commit_status")
        raw_result = payload.get("raw_result")
        if not isinstance(raw_result, Mapping):
            raise ValueError(f"{session.session_id}: tool_result raw_result is invalid")
        result_sha = payload.get("result_sha256")
        if (
            not isinstance(result_sha, str)
            or len(result_sha) != 64
            or any(character not in "0123456789abcdef" for character in result_sha)
            or result_sha != sha256_json(raw_result)
        ):
            raise ValueError(f"{session.session_id}: tool_result result_sha256 mismatch")
        if raw_result.get("tool") != request.tool_name:
            raise ValueError(f"{session.session_id}: committed raw_result tool mismatch")
        if not isinstance(payload.get("formatted_response"), (str, type(None))):
            raise ValueError(
                f"{session.session_id}: tool_result formatted_response is invalid"
            )
        if not isinstance(payload.get("transport"), (Mapping, type(None))):
            raise ValueError(f"{session.session_id}: tool_result transport is invalid")
        committed[request.tool_name] += 1
        if request.tool_name == "visit":
            committed_visit_urls += len(_visit_urls(request))
        committed_by_request_index[index - 1] = event

    return (
        committed_by_request_index,
        {
            "requested_tool_calls": dict(sorted(requested.items())),
            "committed_tool_results": dict(sorted(committed.items())),
            "uncommitted_tool_calls": {
                name: requested[name] - committed[name]
                for name in sorted(requested)
                if requested[name] - committed[name]
            },
            "requested_visit_urls": requested_visit_urls,
            "committed_visit_urls": committed_visit_urls,
        },
    )


def _committed_outcome(
    session: SessionTrace,
    decision_index: int,
    committed_by_request_index: Mapping[int, OtherEvent],
) -> tuple[str, tuple[str, ...]]:
    request_index = decision_index + 1
    if request_index >= len(session.events):
        return "no_next_tool", ()
    request = session.events[request_index]
    if not isinstance(request, ToolCall):
        return "no_next_tool", ()
    if request_index not in committed_by_request_index:
        return f"uncommitted_{request.tool_name}", ()
    if request.tool_name == "visit":
        return "visit", _visit_urls(request)
    return request.tool_name, ()


def extract_committed_search_decisions(
    sessions: Sequence[SessionTrace],
) -> tuple[tuple[SearchDecision, ...], dict[str, Any]]:
    """Extract new-collection decisions from successful tool commits only."""

    decisions: list[SearchDecision] = []
    aggregate_requested: Counter[str] = Counter()
    aggregate_committed: Counter[str] = Counter()
    aggregate_uncommitted: Counter[str] = Counter()
    requested_visit_urls = 0
    committed_visit_urls = 0

    for session in sessions:
        committed_by_request, audit = _validated_tool_result_events(session)
        aggregate_requested.update(audit["requested_tool_calls"])
        aggregate_committed.update(audit["committed_tool_results"])
        aggregate_uncommitted.update(audit["uncommitted_tool_calls"])
        requested_visit_urls += int(audit["requested_visit_urls"])
        committed_visit_urls += int(audit["committed_visit_urls"])

        history: OrderedDict[str, tuple[int, int, int, int]] = OrderedDict()
        visited: OrderedDict[str, None] = OrderedDict()
        pending_updates: list[tuple[str, tuple[str, ...]]] = []
        previous_committed_tool: str | None = None
        search_streak = 0
        search_sequence = 0
        lru_serial = 0

        for request_index, result_event in sorted(committed_by_request.items()):
            request = session.events[request_index]
            assert isinstance(request, ToolCall)
            if request.tool_name == "visit":
                urls = _visit_urls(request)
                pending_updates.append(("visit", urls))
                for url in urls:
                    if not _is_executable_url(url):
                        continue
                    visited.pop(url, None)
                    visited[url] = None
                while len(visited) > VISITED_CAPACITY:
                    visited.popitem(last=False)
                previous_committed_tool = "visit"
                search_streak = 0
                continue
            if request.tool_name != "search":
                pending_updates.append((request.tool_name, ()))
                previous_committed_tool = request.tool_name
                search_streak = 0
                continue

            raw_result = result_event.payload["raw_result"]
            current_results = structured_search_results(raw_result)
            search_sequence += 1
            search_streak = (
                search_streak + 1 if previous_committed_tool == "search" else 1
            )
            previous_committed_tool = "search"
            current_first: OrderedDict[str, SearchResult] = OrderedDict()
            for result in current_results:
                current_first.setdefault(result.url, result)
            snapshot: list[CachedCandidate] = []
            for result in current_first.values():
                lru_serial += 1
                snapshot.append(
                    CachedCandidate(
                        result.url,
                        result.result_rank,
                        result.ordinal,
                        0,
                        result.url in visited,
                        True,
                        lru_serial,
                    )
                )
            for url, (rank, ordinal, seen_at, order) in reversed(history.items()):
                if url in current_first:
                    continue
                age = search_sequence - seen_at
                if age <= MAX_SEARCH_AGE:
                    snapshot.append(
                        CachedCandidate(
                            url,
                            rank,
                            ordinal,
                            age,
                            url in visited,
                            False,
                            order,
                        )
                    )
            for result in current_first.values():
                entry = next(item for item in snapshot if item.url == result.url)
                history.pop(result.url, None)
                history[result.url] = (
                    result.result_rank,
                    result.ordinal,
                    search_sequence,
                    entry.lru_order,
                )
            while len(history) > CACHE_CAPACITY:
                history.popitem(last=False)

            decision_index = request_index + 2
            if (
                decision_index >= len(session.events)
                or not isinstance(session.events[decision_index], LLMCall)
            ):
                # The search committed, but no later model decision exists.
                pending_updates.clear()
                continue
            outcome, targets = _committed_outcome(
                session, decision_index, committed_by_request
            )
            queries = _queries(request)
            query_count = len(queries)
            if query_count == 0 and current_results:
                query_count = max(item.query_index for item in current_results) + 1
            decisions.append(
                SearchDecision(
                    session_id=session.session_id,
                    decision_id=(
                        f"{session.session_id}:committed-search-line-"
                        f"{request.line_number}:{len(decisions)}"
                    ),
                    current_results=current_results,
                    cache_candidates=tuple(snapshot),
                    query_count=query_count,
                    consecutive_search_streak=search_streak,
                    visited_cache_size=len(visited),
                    prior_tool_updates=tuple(pending_updates),
                    outcome=outcome,
                    authoritative_urls=_unique_strings(targets),
                )
            )
            pending_updates.clear()

    return (
        tuple(decisions),
        {
            "semantics": "only commit_status=committed tool_result events",
            "requested_tool_calls": dict(sorted(aggregate_requested.items())),
            "committed_tool_results": dict(sorted(aggregate_committed.items())),
            "uncommitted_tool_calls": dict(sorted(aggregate_uncommitted.items())),
            "requested_visit_urls": requested_visit_urls,
            "committed_visit_urls": committed_visit_urls,
            "committed_search_decision_windows": len(decisions),
        },
    )


def fit_rank_pattern(decisions: Sequence[SearchDecision]) -> RankPattern:
    """Count displayed ranks for exact targets in their current response."""

    counts: Counter[int] = Counter()
    for decision in decisions:
        if decision.outcome != "visit":
            continue
        first_result_by_url: dict[str, SearchResult] = {}
        for result in decision.current_results:
            first_result_by_url.setdefault(result.url, result)
        for url in decision.authoritative_urls:
            source = first_result_by_url.get(url)
            if source is not None:
                counts[source.result_rank] += 1
    normalized = dict(sorted((rank, count) for rank, count in counts.items() if count > 0))
    return RankPattern(normalized, sum(normalized.values()))


def make_frozen_predictor(pattern: RankPattern) -> RankRecencyPatternPredictor:
    """Construct the sole frozen runtime configuration used by every path."""

    return RankRecencyPatternPredictor(
        pattern.rank_counts,
        top_k=FROZEN_TOP_K,
        history_capacity=CACHE_CAPACITY,
        visited_capacity=VISITED_CAPACITY,
        max_history_search_age=MAX_SEARCH_AGE,
        smoothing=RANK_COUNT_SMOOTHING,
        search_age_penalty=AGE_PENALTY,
        visited_penalty=VISITED_PENALTY,
    )


def m0_predictions(
    decision: SearchDecision, pattern: RankPattern, *, top_k: int = 5
) -> tuple[str, ...]:
    """Legacy current-response displayed-rank mapper, invoked blindly."""

    if top_k <= 0 or pattern.total <= 0:
        return ()
    unique: dict[str, SearchResult] = {}
    for result in decision.current_results:
        unique.setdefault(result.url, result)
    eligible = [
        result for result in unique.values() if pattern.rank_counts.get(result.result_rank, 0)
    ]
    eligible.sort(
        key=lambda result: (
            -pattern.rank_counts[result.result_rank],
            result.ordinal,
            result.url,
        )
    )
    return tuple(result.url for result in eligible[:top_k])


def candidate_score(
    candidate: CachedCandidate, predictor: RankRecencyPatternPredictor
) -> float:
    return predictor.score(
        candidate.result_rank, candidate.age, candidate.visited
    )


def ranked_pattern_predictions(
    decision: SearchDecision,
    predictor: RankRecencyPatternPredictor,
    *,
    top_k: int = 5,
) -> tuple[str, ...]:
    """Rank the causal candidate snapshot without applying the visit gate."""

    if top_k <= 0:
        return ()
    scored = [
        (candidate_score(candidate, predictor), candidate)
        for candidate in decision.cache_candidates
    ]
    if not scored:
        return ()

    scored.sort(
        key=lambda item: (
            -item[0],
            not item[1].current,
            item[1].visited,
            item[1].age,
            item[1].ordinal,
            item[1].url,
        )
    )
    current = [item for item in decision.cache_candidates if item.current]
    if current:
        anchor = min(
            current,
            key=lambda item: (
                -predictor.rank_counts.get(item.result_rank, 0),
                item.ordinal,
                item.url,
            ),
        )
        scored = [
            (candidate_score(anchor, predictor), anchor),
            *(item for item in scored if item[1].url != anchor.url),
        ]
    return tuple(item.url for _, item in scored[:top_k])


def pattern_predictions(
    decision: SearchDecision,
    predictor: RankRecencyPatternPredictor,
    *,
    top_k: int = 5,
) -> tuple[tuple[str, ...], bool, str]:
    """Independent gated reference implementation used for runtime parity."""

    ranked = ranked_pattern_predictions(decision, predictor, top_k=top_k)
    if not ranked:
        return (), False, "no_candidates"
    if (
        decision.query_count >= 10
        and decision.consecutive_search_streak == 2
    ):
        return (), False, "matched_abstain_pattern"
    return ranked, True, "no_rule_match_admit"


def _hit_count(targets: Sequence[str], predictions: Sequence[str], top_k: int) -> int:
    predicted = set(predictions[:top_k])
    return sum(url in predicted for url in targets)


def score_decisions(
    decisions: Sequence[SearchDecision], predictor: RankRecencyPatternPredictor
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    rows: list[dict[str, Any]] = []
    durations: dict[str, list[float]] = {"M0_current_blind": [], "pattern_cache": []}
    pattern = RankPattern(dict(predictor.rank_counts), sum(predictor.rank_counts.values()))
    runtime_sessions = {
        decision.session_id: predictor.start_session(decision.session_id)
        for decision in decisions
    }
    for decision in decisions:
        session = runtime_sessions[decision.session_id]
        for tool_name, urls in decision.prior_tool_updates:
            if tool_name == "visit":
                executable = tuple(url for url in urls if _is_executable_url(url))
                if executable:
                    session.observe_visit(executable)
                else:
                    session.observe_other_tool("invalid_visit")
            elif tool_name == "search":  # pragma: no cover - current search is not pending
                raise RuntimeError("a search update cannot precede its own decision")
            else:
                session.observe_other_tool(tool_name)
        started = time.perf_counter_ns()
        baseline = m0_predictions(decision, pattern, top_k=max(TOP_KS))
        durations["M0_current_blind"].append(
            (time.perf_counter_ns() - started) / 1_000_000.0
        )
        started = time.perf_counter_ns()
        runtime = session.observe_search(
            decision.current_results,
            query_count=(decision.query_count if decision.query_count > 0 else None),
        )
        durations["pattern_cache"].append(
            (time.perf_counter_ns() - started) / 1_000_000.0
        )
        candidate = runtime.prediction_urls
        ungated_candidate = tuple(item.url for item in runtime.ranked_top_k)
        admitted = runtime.gate.admitted
        reason = runtime.gate.reason
        if candidate != (ungated_candidate if admitted else ()):
            raise RuntimeError(
                f"runtime gated/ungated relationship failed at {decision.decision_id}"
            )

        # The extractor contains a separately implemented causal state mirror.
        # Assert parity decision by decision so offline numbers cannot silently
        # drift from the delivered runtime API.
        reference = pattern_predictions(
            decision, predictor, top_k=max(TOP_KS)
        )
        observed = (candidate, admitted, reason)
        if reference != observed:
            raise RuntimeError(
                f"runtime/reference prediction parity failed at {decision.decision_id}: "
                f"{observed!r} != {reference!r}"
            )

        # Top-10/20 are evaluation-side ranking-depth diagnostics.  The v2
        # artifact and deployed dispatch remain frozen at Top-5.  Prove that
        # the expanded deterministic ranking is a strict extension of the
        # real runtime output before using its tail for any metric.
        diagnostic_baseline = m0_predictions(
            decision, pattern, top_k=max(DIAGNOSTIC_TOP_KS)
        )
        diagnostic_ungated = ranked_pattern_predictions(
            decision, predictor, top_k=max(DIAGNOSTIC_TOP_KS)
        )
        if diagnostic_ungated[: predictor.top_k] != ungated_candidate:
            raise RuntimeError(
                "expanded/runtime Top-5 prefix parity failed at "
                f"{decision.decision_id}"
            )
        diagnostic_gated = diagnostic_ungated if admitted else ()
        if diagnostic_gated[: predictor.top_k] != candidate:
            raise RuntimeError(
                "expanded/runtime gated Top-5 prefix parity failed at "
                f"{decision.decision_id}"
            )
        if runtime.candidate_count != len(decision.cache_candidates):
            raise RuntimeError(
                f"runtime/cache candidate parity failed at {decision.decision_id}"
            )
        if runtime.gate.consecutive_search_streak != decision.consecutive_search_streak:
            raise RuntimeError(
                f"runtime/search-streak parity failed at {decision.decision_id}"
            )
        if runtime.cache["visited_size_before"] != decision.visited_cache_size:
            raise RuntimeError(
                f"runtime/visited-LRU parity failed at {decision.decision_id}"
            )

        targets = decision.authoritative_urls if decision.outcome == "visit" else ()
        current_urls = {result.url for result in decision.current_results}
        cache_urls = {item.url for item in decision.cache_candidates}
        policy_urls = cache_urls
        rows.append(
            {
                "session_id": decision.session_id,
                "decision_id": decision.decision_id,
                "outcome": decision.outcome,
                "targets": list(targets),
                "target_count": len(targets),
                "nonexecutable_target_count": sum(
                    not _is_executable_url(url) for url in targets
                ),
                "m0_predictions": list(baseline),
                "pattern_ungated_predictions": list(ungated_candidate),
                "pattern_gated_predictions": list(candidate),
                "pattern_predictions": list(candidate),
                "m0_diagnostic_predictions": list(diagnostic_baseline),
                "pattern_ungated_diagnostic_predictions": list(
                    diagnostic_ungated
                ),
                "pattern_gated_diagnostic_predictions": list(diagnostic_gated),
                "gate_admitted": admitted,
                "gate_reason": reason,
                "query_count": decision.query_count,
                "consecutive_search_streak": decision.consecutive_search_streak,
                "current_candidate_count": len(current_urls),
                "cache_candidate_count": len(cache_urls),
                "current_covered_targets": sum(url in current_urls for url in targets),
                "cache_covered_targets": sum(url in policy_urls for url in targets),
                "gated_cache_covered_targets": (
                    sum(url in policy_urls for url in targets) if admitted else 0
                ),
                **{
                    f"m0_hits_at_{top_k}": _hit_count(targets, baseline, top_k)
                    for top_k in TOP_KS
                },
                **{
                    f"pattern_ungated_hits_at_{top_k}": _hit_count(
                        targets, ungated_candidate, top_k
                    )
                    for top_k in TOP_KS
                },
                **{
                    f"pattern_gated_hits_at_{top_k}": _hit_count(
                        targets, candidate, top_k
                    )
                    for top_k in TOP_KS
                },
                **{
                    f"pattern_hits_at_{top_k}": _hit_count(targets, candidate, top_k)
                    for top_k in TOP_KS
                },
                **{
                    f"m0_diagnostic_hits_at_{top_k}": _hit_count(
                        targets, diagnostic_baseline, top_k
                    )
                    for top_k in DIAGNOSTIC_TOP_KS
                },
                **{
                    f"pattern_ungated_diagnostic_hits_at_{top_k}": _hit_count(
                        targets, diagnostic_ungated, top_k
                    )
                    for top_k in DIAGNOSTIC_TOP_KS
                },
                **{
                    f"pattern_gated_diagnostic_hits_at_{top_k}": _hit_count(
                        targets, diagnostic_gated, top_k
                    )
                    for top_k in DIAGNOSTIC_TOP_KS
                },
            }
        )

    return rows, durations


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "measured_calls": len(ordered),
        "mean_ms": statistics.fmean(ordered) if ordered else 0.0,
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "max_ms": max(ordered) if ordered else 0.0,
        "timer": "time.perf_counter_ns",
    }


def _model_metrics(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    visit_rows = [row for row in rows if row["outcome"] == "visit"]
    targets = sum(int(row["target_count"]) for row in visit_rows)
    metrics: dict[str, Any] = {
        "conditional_visit_windows": len(visit_rows),
        "conditional_visit_targets": targets,
        "exact_top_k": {},
    }
    for top_k in TOP_KS:
        hits = sum(int(row[f"{prefix}_hits_at_{top_k}"]) for row in visit_rows)
        predictions = sum(
            min(top_k, len(row[f"{prefix}_predictions"])) for row in visit_rows
        )
        metrics["exact_top_k"][str(top_k)] = {
            "hits": hits,
            "target_recall": hits / targets if targets else 0.0,
            "predictions": predictions,
            "conditional_precision": hits / predictions if predictions else 0.0,
        }

    all_predictions = sum(len(row[f"{prefix}_predictions"]) for row in rows)
    exact_hits = sum(int(row[f"{prefix}_hits_at_5"]) for row in visit_rows)
    nonvisit_predictions = sum(
        len(row[f"{prefix}_predictions"])
        for row in rows
        if row["outcome"] != "visit"
    )
    metrics["all_window_top5"] = {
        "search_decisions": len(rows),
        "predictions": all_predictions,
        "exact_target_hits": exact_hits,
        "precision": exact_hits / all_predictions if all_predictions else 0.0,
        "waste": all_predictions - exact_hits,
        "non_visit_predictions": nonvisit_predictions,
    }
    return metrics


def _diagnostic_model_metrics(
    rows: Sequence[Mapping[str, Any]], prefix: str
) -> dict[str, Any]:
    """Exact-recall metrics for the non-dispatch Top-20 ranking sidecar."""

    visit_rows = [row for row in rows if row["outcome"] == "visit"]
    targets = sum(int(row["target_count"]) for row in visit_rows)
    return {
        "conditional_visit_windows": len(visit_rows),
        "conditional_visit_targets": targets,
        "exact_top_k": {
            str(top_k): {
                "hits": sum(
                    int(row[f"{prefix}_hits_at_{top_k}"]) for row in visit_rows
                ),
                "target_recall": (
                    sum(
                        int(row[f"{prefix}_hits_at_{top_k}"])
                        for row in visit_rows
                    )
                    / targets
                    if targets
                    else 0.0
                ),
                "conditional_predictions": sum(
                    min(top_k, len(row[f"{prefix}_predictions"]))
                    for row in visit_rows
                ),
                "all_window_predictions": sum(
                    min(top_k, len(row[f"{prefix}_predictions"])) for row in rows
                ),
            }
            for top_k in DIAGNOSTIC_TOP_KS
        },
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    durations: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    tp = sum(bool(row["gate_admitted"]) and row["outcome"] == "visit" for row in rows)
    fn = sum(not bool(row["gate_admitted"]) and row["outcome"] == "visit" for row in rows)
    fp = sum(bool(row["gate_admitted"]) and row["outcome"] != "visit" for row in rows)
    tn = sum(not bool(row["gate_admitted"]) and row["outcome"] != "visit" for row in rows)
    visit_rows = [row for row in rows if row["outcome"] == "visit"]
    target_total = sum(int(row["target_count"]) for row in visit_rows)
    baseline_metrics = _model_metrics(rows, "m0")
    ungated_metrics = _model_metrics(rows, "pattern_ungated")
    gated_metrics = _model_metrics(rows, "pattern_gated")
    ungated_dispatches = sum(len(row["pattern_ungated_predictions"]) for row in rows)
    gated_dispatches = sum(len(row["pattern_gated_predictions"]) for row in rows)
    dispatch_reduction = ungated_dispatches - gated_dispatches
    fired_count = sum(
        row["gate_reason"] == "matched_abstain_pattern" for row in rows
    )
    diagnostic_models = {
        "M0_current_blind": _diagnostic_model_metrics(rows, "m0_diagnostic"),
        "pattern_cache_ungated": _diagnostic_model_metrics(
            rows, "pattern_ungated_diagnostic"
        ),
        "pattern_cache_gated": _diagnostic_model_metrics(
            rows, "pattern_gated_diagnostic"
        ),
    }

    def ceiling(field: str) -> dict[str, Any]:
        covered = sum(int(row[field]) for row in visit_rows)
        return {
            "covered_targets": covered,
            "target_coverage": covered / target_total if target_total else 0.0,
            "top_k_oracle_hits": {
                str(top_k): sum(
                    min(top_k, int(row[field])) for row in visit_rows
                )
                for top_k in TOP_KS
            },
        }

    result: dict[str, Any] = {
        "sessions": len({str(row["session_id"]) for row in rows}),
        "search_decisions": len(rows),
        "outcomes": dict(sorted(outcomes.items())),
        "nonexecutable_authoritative_targets": sum(
            int(row.get("nonexecutable_target_count", 0)) for row in rows
        ),
        "models": {
            "M0_current_blind": baseline_metrics,
            "pattern_cache_ungated": ungated_metrics,
            "pattern_cache_gated": gated_metrics,
            # Backward-compatible alias.  New reports always use the explicit
            # gated/ungated names above.
            "pattern_cache": gated_metrics,
        },
        "model_aliases": {"pattern_cache": "pattern_cache_gated"},
        "ranking_depth_diagnostic": {
            "evaluation_only": True,
            "runtime_dispatch_top_k": FROZEN_TOP_K,
            "top_ks": list(DIAGNOSTIC_TOP_KS),
            "ranking": (
                "same exact-URL v2 score and causal bounded cache; expanded "
                "ranking prefix is asserted equal to runtime Top-5"
            ),
            "models": diagnostic_models,
            "candidate_pool": {
                "maximum_union_size": max(
                    (int(row["cache_candidate_count"]) for row in rows),
                    default=0,
                ),
                "decisions_with_at_least_20_candidates": sum(
                    int(row["cache_candidate_count"]) >= 20 for row in rows
                ),
                "search_decisions": len(rows),
            },
            "hypothetical_gate_dispatch_reduction": {
                str(top_k): {
                    "ungated": sum(
                        min(
                            top_k,
                            len(row["pattern_ungated_diagnostic_predictions"]),
                        )
                        for row in rows
                    ),
                    "gated": sum(
                        min(
                            top_k,
                            len(row["pattern_gated_diagnostic_predictions"]),
                        )
                        for row in rows
                    ),
                }
                for top_k in DIAGNOSTIC_TOP_KS
            },
            "candidate_oracle_hits": {
                "current_response": {
                    str(top_k): sum(
                        min(top_k, int(row["current_covered_targets"]))
                        for row in visit_rows
                    )
                    for top_k in DIAGNOSTIC_TOP_KS
                },
                "bounded_cache": {
                    str(top_k): sum(
                        min(top_k, int(row["cache_covered_targets"]))
                        for row in visit_rows
                    )
                    for top_k in DIAGNOSTIC_TOP_KS
                },
                "gated_bounded_cache": {
                    str(top_k): sum(
                        min(top_k, int(row["gated_cache_covered_targets"]))
                        for row in visit_rows
                    )
                    for top_k in DIAGNOSTIC_TOP_KS
                },
            },
        },
        "gate": {
            "positive_label": "next_tool_is_committed_visit",
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "admission_rate": (tp + fp) / len(rows) if rows else 0.0,
            "fired_count": fired_count,
            "abstained_count": fn + tn,
            "ungated_url_dispatches": ungated_dispatches,
            "gated_url_dispatches": gated_dispatches,
            "dispatch_reduction_absolute": dispatch_reduction,
            "dispatch_reduction_fraction": (
                dispatch_reduction / ungated_dispatches
                if ungated_dispatches
                else None
            ),
            "gated_vs_ungated_exact_top_k": {
                str(top_k): {
                    "ungated_hits": ungated_metrics["exact_top_k"][str(top_k)][
                        "hits"
                    ],
                    "gated_hits": gated_metrics["exact_top_k"][str(top_k)]["hits"],
                    "delta_hits": (
                        gated_metrics["exact_top_k"][str(top_k)]["hits"]
                        - ungated_metrics["exact_top_k"][str(top_k)]["hits"]
                    ),
                }
                for top_k in TOP_KS
            },
            "abstain_reasons": dict(
                sorted(Counter(str(row["gate_reason"]) for row in rows).items())
            ),
        },
        "candidate_ceilings": {
            "current_response": ceiling("current_covered_targets"),
            "history_lru64_age_le_2_plus_m0_top1": ceiling("cache_covered_targets"),
            "after_visit_abstain_gate": ceiling("gated_cache_covered_targets"),
            "target_total": target_total,
        },
    }
    if durations is not None:
        result["latency"] = {
            name: latency_summary(values) for name, values in durations.items()
        }
    return result


def new_holdout_acceptance(
    evaluation: Mapping[str, Any],
    latency_benchmark: Mapping[str, Any],
    *,
    total_manifest_sessions: int,
) -> dict[str, Any]:
    """Apply the frozen confirmatory gate without treating low power as failure."""

    models = evaluation["models"]
    baseline = models["M0_current_blind"]
    ungated = models["pattern_cache_ungated"]
    gated = models["pattern_cache_gated"]
    gate = evaluation["gate"]
    target_total = int(baseline["conditional_visit_targets"])
    committed_search_sessions = int(evaluation["sessions"])

    data_conditions = {
        "manifest_sessions_exactly_30": total_manifest_sessions == 30,
        "exact_target_total_at_least_80": target_total >= 80,
        "sessions_with_committed_search_at_least_20": (
            committed_search_sessions >= 20
        ),
    }
    data_adequate = all(data_conditions.values())

    gated_vs_m0 = {
        str(top_k): (
            gated["exact_top_k"][str(top_k)]["hits"]
            >= baseline["exact_top_k"][str(top_k)]["hits"]
        )
        for top_k in TOP_KS
    }
    strict_top3_or_top5 = bool(
        gated["exact_top_k"]["3"]["hits"]
        > baseline["exact_top_k"]["3"]["hits"]
        or gated["exact_top_k"]["5"]["hits"]
        > baseline["exact_top_k"]["5"]["hits"]
    )
    ranker_conditions = {
        "gated_top1_at_least_m0": gated_vs_m0["1"],
        "gated_top3_at_least_m0": gated_vs_m0["3"],
        "gated_top5_at_least_m0": gated_vs_m0["5"],
        "gated_top3_or_top5_strictly_above_m0": strict_top3_or_top5,
    }
    ranker_passed = all(ranker_conditions.values())

    gated_vs_ungated = {
        str(top_k): (
            gated["exact_top_k"][str(top_k)]["hits"]
            >= ungated["exact_top_k"][str(top_k)]["hits"]
        )
        for top_k in TOP_KS
    }
    fired_count = int(gate["fired_count"])
    gate_conditions = {
        "fired_at_least_once": fired_count > 0,
        "committed_visit_window_recall_at_least_0_95": float(gate["recall"]) >= 0.95,
        "strict_url_dispatch_reduction": int(
            gate["dispatch_reduction_absolute"]
        )
        > 0,
        "gated_top1_at_least_ungated": gated_vs_ungated["1"],
        "gated_top3_at_least_ungated": gated_vs_ungated["3"],
        "gated_top5_at_least_ungated": gated_vs_ungated["5"],
    }
    gate_passed = all(gate_conditions.values())

    runtime = latency_benchmark["models"]["pattern_cache"]
    runtime_conditions = {
        "p99_below_100ms": float(runtime["p99_ms"]) < 100.0,
        "max_below_100ms": float(runtime["max_ms"]) < 100.0,
    }
    runtime_passed = all(runtime_conditions.values())

    inconclusive_reasons: list[str] = []
    if not data_adequate:
        inconclusive_reasons.append("data_adequacy_threshold_not_met")
    if fired_count == 0:
        inconclusive_reasons.append("visit_abstain_rule_never_fired")
    conclusive = not inconclusive_reasons
    accepted: bool | None
    if conclusive:
        accepted = bool(ranker_passed and gate_passed and runtime_passed)
        status = "accepted" if accepted else "rejected"
    else:
        accepted = None
        status = "inconclusive"

    return {
        "status": status,
        "accepted": accepted,
        "inconclusive_reasons": inconclusive_reasons,
        "data_adequacy": {
            "passed": data_adequate,
            "conditions": data_conditions,
            "exact_target_total": target_total,
            "sessions_with_committed_search_decision": committed_search_sessions,
            "total_manifest_sessions": total_manifest_sessions,
        },
        "ranker": {
            "passed": ranker_passed,
            "conditions": ranker_conditions,
            "gated_minus_m0_hits": {
                str(top_k): (
                    gated["exact_top_k"][str(top_k)]["hits"]
                    - baseline["exact_top_k"][str(top_k)]["hits"]
                )
                for top_k in TOP_KS
            },
        },
        "visit_abstain_gate": {
            "evaluable": fired_count > 0,
            "passed": gate_passed if fired_count > 0 else None,
            "conditions": gate_conditions,
            "fired_count": fired_count,
            "committed_visit_window_recall": float(gate["recall"]),
            "ungated_url_dispatches": int(gate["ungated_url_dispatches"]),
            "gated_url_dispatches": int(gate["gated_url_dispatches"]),
            "dispatch_reduction_absolute": int(
                gate["dispatch_reduction_absolute"]
            ),
            "dispatch_reduction_fraction": gate["dispatch_reduction_fraction"],
            "gated_minus_ungated_hits": {
                str(top_k): (
                    gated["exact_top_k"][str(top_k)]["hits"]
                    - ungated["exact_top_k"][str(top_k)]["hits"]
                )
                for top_k in TOP_KS
            },
        },
        "runtime": {
            "passed": runtime_passed,
            "conditions": runtime_conditions,
            "p99_ms": float(runtime["p99_ms"]),
            "max_ms": float(runtime["max_ms"]),
        },
        "overall": {
            "evaluated": conclusive,
            "passed": accepted,
            "rule": (
                "accept only when data are adequate and ranker, visit-abstain "
                "gate, and runtime checks all pass"
            ),
        },
    }


def cv_fold(session_id: str) -> int:
    digest = hashlib.sha256(f"{CV_SEED}\0{session_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 5


def grouped_oof(sessions: Sequence[SessionTrace]) -> dict[str, Any]:
    decisions = extract_search_decisions(sessions)
    all_rows: list[dict[str, Any]] = []
    all_durations: dict[str, list[float]] = {
        "M0_current_blind": [],
        "pattern_cache": [],
    }
    folds: list[dict[str, Any]] = []
    for fold in range(5):
        fit_ids = {
            session.session_id for session in sessions if cv_fold(session.session_id) != fold
        }
        validation_ids = {
            session.session_id for session in sessions if cv_fold(session.session_id) == fold
        }
        fit_decisions = [item for item in decisions if item.session_id in fit_ids]
        validation = [item for item in decisions if item.session_id in validation_ids]
        pattern = fit_rank_pattern(fit_decisions)
        predictor = make_frozen_predictor(pattern)
        rows, durations = score_decisions(validation, predictor)
        all_rows.extend(rows)
        for name, values in durations.items():
            all_durations[name].extend(values)
        folds.append(
            {
                "fold": fold,
                "fit_sessions": len(fit_ids),
                "validation_sessions": len(validation_ids),
                "fit_decisions": len(fit_decisions),
                "validation_decisions": len(validation),
                "rank_counts": {str(k): v for k, v in pattern.rank_counts.items()},
                "mapped_current_response_targets": pattern.total,
            }
        )
    if len(all_rows) != len(decisions):
        raise RuntimeError("grouped OOF did not score each decision exactly once")
    metrics = summarize_rows(all_rows, durations=all_durations)
    metrics["whole_sessions"] = len(sessions)
    metrics["sessions_without_search_decisions"] = len(sessions) - int(
        metrics["sessions"]
    )
    return {
        "grouping_unit": "whole session",
        "fold_seed": CV_SEED,
        "fold_count": 5,
        "folds": folds,
        "metrics": metrics,
    }


def trace_manifest(sessions: Sequence[SessionTrace]) -> dict[str, Any]:
    files = [
        {
            "session_id": session.session_id,
            "sha256": sha256_file(session.path),
        }
        for session in sorted(sessions, key=lambda item: item.session_id)
    ]
    manifest: dict[str, Any] = {
        "session_count": len(files),
        "sessions": files,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def load_expected_new_workload(path: Path) -> FixedWorkload:
    workload = load_fixed_workload(path)
    observed = (
        workload.workload_id,
        workload.file_sha256,
        len(workload.sources),
    )
    expected = (
        EXPECTED_NEW_WORKLOAD_ID,
        EXPECTED_NEW_WORKLOAD_SHA256,
        EXPECTED_NEW_SOURCE_COUNT,
    )
    if observed != expected:
        raise ValueError(
            "new holdout workload is not the preregistered immutable workload: "
            f"{observed!r} != {expected!r}"
        )
    return workload


def validate_collection_manifest(
    collection_dir: Path,
    workload: FixedWorkload,
) -> tuple[dict[str, Any], tuple[SessionTrace, ...], dict[str, Any]]:
    """Fail closed on collection identity, order, hashes, and extra traces."""

    manifest_path = collection_dir / "manifest.json"
    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid collection manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest_raw, dict):
        raise ValueError("collection manifest root must be an object")
    manifest = dict(manifest_raw)
    if manifest.get("artifact_type") != MANIFEST_TYPE:
        raise ValueError("collection manifest artifact_type mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("collection manifest schema_version mismatch")
    if manifest.get("trace_schema") != TRACE_SCHEMA:
        raise ValueError("collection manifest trace_schema mismatch")
    status = manifest.get("collection_status")
    if status not in {"complete", "complete_with_failures"}:
        raise ValueError(
            "collection_status must be complete or complete_with_failures, "
            f"not {status!r}"
        )
    if not isinstance(manifest.get("completed_at_utc"), str) or not manifest[
        "completed_at_utc"
    ]:
        raise ValueError("completed collection manifest lacks completed_at_utc")

    workload_record = manifest.get("workload")
    if not isinstance(workload_record, Mapping):
        raise ValueError("collection manifest workload must be an object")
    ordered_ids = [source.source_id for source in workload.sources]
    workload_expected = {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "workload_id": workload.workload_id,
        "file_name": workload.file_name,
        "file_sha256": workload.file_sha256,
        "source_count": len(workload.sources),
        "ordered_source_ids": ordered_ids,
    }
    if dict(workload_record) != workload_expected:
        raise ValueError("collection manifest workload binding mismatch")

    records = manifest.get("sessions")
    if not isinstance(records, list) or len(records) != len(workload.sources):
        raise ValueError("collection manifest must contain exactly 30 session records")
    trace_names: list[str] = []
    trace_paths: list[Path] = []
    failed = 0
    for index, (record, source) in enumerate(zip(records, workload.sources), 1):
        if not isinstance(record, Mapping):
            raise ValueError(f"collection sessions[{index - 1}] must be an object")
        expected_session_id = f"{index:04d}-{source.source_id}"
        if record.get("session_id") != expected_session_id:
            raise ValueError(f"session order/id mismatch at ordinal {index}")
        if record.get("source_id") != source.source_id:
            raise ValueError(f"source_id mismatch at ordinal {index}")
        if record.get("source_sha256") != source.source_sha256:
            raise ValueError(f"source_sha256 mismatch at ordinal {index}")
        if record.get("question_sha256") != source.question_sha256:
            raise ValueError(f"question_sha256 mismatch at ordinal {index}")
        if record.get("provenance") != source.provenance:
            raise ValueError(f"source provenance mismatch at ordinal {index}")
        record_status = record.get("status")
        if record_status not in {"succeeded", "failed"}:
            raise ValueError(f"invalid session status at ordinal {index}")
        failed += record_status != "succeeded"
        for count_field in (
            "llm_calls",
            "tool_calls",
            "committed_tool_results",
            "event_count",
        ):
            value = record.get(count_field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"invalid {count_field} in session record at ordinal {index}"
                )

        trace_name = record.get("trace_file")
        if (
            not isinstance(trace_name, str)
            or not trace_name.endswith(".jsonl")
            or Path(trace_name).name != trace_name
            or trace_name != f"{expected_session_id}.jsonl"
        ):
            raise ValueError(f"unsafe or mismatched trace_file at ordinal {index}")
        if trace_name in trace_names:
            raise ValueError(f"duplicate trace_file: {trace_name}")
        trace_names.append(trace_name)
        trace_path = collection_dir / trace_name
        if not trace_path.is_file():
            raise ValueError(f"missing trace_file: {trace_name}")
        observed_sha = sha256_file(trace_path)
        if record.get("trace_sha256") != observed_sha:
            raise ValueError(f"trace_sha256 mismatch for {trace_name}")
        trace_paths.append(trace_path)

    actual_jsonl = {path.name for path in collection_dir.glob("*.jsonl")}
    expected_jsonl = set(trace_names)
    if actual_jsonl != expected_jsonl:
        raise ValueError(
            "collection trace set mismatch (missing or extra JSONL): "
            f"expected={sorted(expected_jsonl)!r}, actual={sorted(actual_jsonl)!r}"
        )
    summary = manifest.get("summary")
    expected_summary = {
        "session_count": len(workload.sources),
        "succeeded": len(workload.sources) - failed,
        "failed": failed,
    }
    if summary != expected_summary:
        raise ValueError("collection summary does not match session statuses")
    if (status == "complete") != (failed == 0):
        raise ValueError("collection_status does not match failure count")

    sessions = tuple(load_trace(path) for path in trace_paths)
    for index, (session, record, source) in enumerate(
        zip(sessions, records, workload.sources), 1
    ):
        if not session.events:
            raise ValueError(f"trace at ordinal {index} is empty")
        start = session.events[0]
        end = session.events[-1]
        if not isinstance(start, OtherEvent) or start.event_type != "session_start":
            raise ValueError(f"trace at ordinal {index} lacks session_start")
        start_expected = {
            "session_id": record["session_id"],
            "workload_id": workload.workload_id,
            "source_id": source.source_id,
            "source_sha256": source.source_sha256,
            "question_sha256": source.question_sha256,
            "provenance": source.provenance,
        }
        if any(start.payload.get(key) != value for key, value in start_expected.items()):
            raise ValueError(f"trace session_start binding mismatch at ordinal {index}")
        if not isinstance(end, OtherEvent) or end.event_type != "session_end":
            raise ValueError(f"trace at ordinal {index} lacks session_end")
        if end.payload.get("status") != record["status"]:
            raise ValueError(f"trace session_end status mismatch at ordinal {index}")
        observed_counts = {
            "llm_calls": sum(isinstance(event, LLMCall) for event in session.events),
            "tool_calls": sum(isinstance(event, ToolCall) for event in session.events),
            "committed_tool_results": sum(
                isinstance(event, OtherEvent) and event.event_type == "tool_result"
                for event in session.events
            ),
            "event_count": len(session.events),
        }
        if any(record[field] != value for field, value in observed_counts.items()):
            raise ValueError(f"session record event counts mismatch at ordinal {index}")
        for field in ("llm_calls", "tool_calls", "committed_tool_results"):
            if end.payload.get(field) != observed_counts[field]:
                raise ValueError(f"session_end {field} mismatch at ordinal {index}")
    return (
        manifest,
        sessions,
        {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "collection_status": status,
            "source_count": len(workload.sources),
            "failed_sessions": failed,
            "ordered_session_ids": [record["session_id"] for record in records],
            "ordered_trace_files": trace_names,
        },
    )


def visit_executability_inventory(
    sessions: Sequence[SessionTrace],
) -> dict[str, Any]:
    """Inventory all visit calls, including those outside scored search windows."""

    visit_calls = 0
    raw_urls = 0
    executable_urls = 0
    nonexecutable_urls = 0
    calls_with_nonexecutable_urls = 0
    schemes: Counter[str] = Counter()
    for session in sessions:
        for event in session.events:
            if not isinstance(event, ToolCall) or event.tool_name != "visit":
                continue
            visit_calls += 1
            urls = _visit_urls(event)
            raw_urls += len(urls)
            invalid_in_call = 0
            for url in urls:
                if _is_executable_url(url):
                    executable_urls += 1
                else:
                    nonexecutable_urls += 1
                    invalid_in_call += 1
                    schemes[url.split(":", 1)[0] if ":" in url else "missing"] += 1
            calls_with_nonexecutable_urls += invalid_in_call > 0
    return {
        "visit_calls": visit_calls,
        "raw_exact_url_labels": raw_urls,
        "runtime_executable_http_urls": executable_urls,
        "runtime_nonexecutable_url_labels": nonexecutable_urls,
        "visit_calls_with_nonexecutable_urls": calls_with_nonexecutable_urls,
        "nonexecutable_scheme_counts": dict(sorted(schemes.items())),
        "handling": (
            "retain as exact-label misses where scored; never dispatch or insert "
            "into the bounded visited LRU"
        ),
    }


def build_artifact(
    pattern: RankPattern,
    sessions: Sequence[SessionTrace],
) -> dict[str, Any]:
    if pattern.total <= 0:
        raise ValueError("cannot freeze an artifact without mapped rank targets")
    predictor = make_frozen_predictor(pattern)
    manifest = trace_manifest(sessions)
    manifest.update(
        {
            "algorithm": "displayed-rank counts",
            "fit_target_scope": (
                "exact visit targets present in the immediately current search response"
            ),
        }
    )
    # Recompute after adding the fit declaration; ``to_artifact`` validates it.
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_json(manifest)
    return predictor.to_artifact(manifest)


def validate_artifact(
    raw: Mapping[str, Any],
) -> tuple[RankRecencyPatternPredictor, dict[str, Any]]:
    predictor = RankRecencyPatternPredictor.from_artifact(raw)
    expected = {
        "schema": PATTERN_ARTIFACT_SCHEMA,
        "version": PATTERN_ARTIFACT_VERSION,
        "policy": PATTERN_POLICY_VERSION,
        "top_k": FROZEN_TOP_K,
        "history_capacity": CACHE_CAPACITY,
        "visited_capacity": VISITED_CAPACITY,
        "max_history_search_age": MAX_SEARCH_AGE,
        "smoothing": RANK_COUNT_SMOOTHING,
        "search_age_penalty": AGE_PENALTY,
        "visited_penalty": VISITED_PENALTY,
    }
    observed = {
        "schema": raw.get("schema"),
        "version": raw.get("version"),
        "policy": predictor.policy,
        "top_k": predictor.top_k,
        "history_capacity": predictor.history_capacity,
        "visited_capacity": predictor.visited_capacity,
        "max_history_search_age": predictor.max_history_search_age,
        "smoothing": predictor.smoothing,
        "search_age_penalty": predictor.search_age_penalty,
        "visited_penalty": predictor.visited_penalty,
    }
    if observed != expected:
        raise ValueError(
            f"artifact is not the sole frozen pattern-cache configuration: "
            f"{observed!r} != {expected!r}"
        )
    return predictor, dict(raw)


def load_artifact(
    path: Path,
) -> tuple[RankRecencyPatternPredictor, dict[str, Any]]:
    predictor, raw = load_pattern_artifact(path)
    checked, artifact = validate_artifact(raw)
    if predictor.metadata() != checked.metadata():
        raise RuntimeError("pattern artifact loader parity failed")
    return checked, artifact


def benchmark(
    decisions: Sequence[SearchDecision],
    predictor: RankRecencyPatternPredictor,
    *,
    passes: int = 20,
) -> dict[str, Any]:
    if passes <= 0:
        raise ValueError("benchmark passes must be positive")
    score_decisions(decisions, predictor)
    samples: dict[str, list[float]] = {"M0_current_blind": [], "pattern_cache": []}
    for _ in range(passes):
        _, measured = score_decisions(decisions, predictor)
        for name, values in measured.items():
            samples[name].extend(values)
    return {
        "scope": "local gate + exact-URL rank/cache scoring + Top-5 sort",
        "excludes": (
            "trace parsing, artifact fit, network, tools, and external request "
            "scheduling"
        ),
        "passes": passes,
        "decision_inputs": len(decisions),
        "models": {name: latency_summary(values) for name, values in samples.items()},
        "acceptance_threshold_ms": 100.0,
        "pattern_p99_below_100ms": (
            latency_summary(samples["pattern_cache"])["p99_ms"] < 100.0
        ),
        "pattern_max_below_100ms": (
            latency_summary(samples["pattern_cache"])["max_ms"] < 100.0
        ),
    }


def paired_session_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    all_session_ids: Sequence[str],
    *,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Paired percentile bootstrap of gated Pattern minus M0 exact recall."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    session_ids = sorted(dict.fromkeys(all_session_ids))
    if not session_ids:
        raise ValueError("cannot bootstrap an empty holdout")
    per_session: dict[str, dict[str, Any]] = {
        session_id: {
            "targets": 0,
            "m0": {top_k: 0 for top_k in TOP_KS},
            "pattern": {top_k: 0 for top_k in TOP_KS},
        }
        for session_id in session_ids
    }
    for row in rows:
        session_id = str(row["session_id"])
        if session_id not in per_session:
            raise ValueError("prediction row refers to an unknown session")
        if row["outcome"] != "visit":
            continue
        per_session[session_id]["targets"] += int(row["target_count"])
        for top_k in TOP_KS:
            per_session[session_id]["m0"][top_k] += int(row[f"m0_hits_at_{top_k}"])
            per_session[session_id]["pattern"][top_k] += int(
                row[f"pattern_hits_at_{top_k}"]
            )

    target_total = sum(item["targets"] for item in per_session.values())
    rng = random.Random(BOOTSTRAP_SEED)
    samples: dict[int, list[float]] = {top_k: [] for top_k in TOP_KS}
    zero_target_resamples_discarded = 0
    while target_total > 0 and len(samples[TOP_KS[0]]) < replicates:
        selected = [rng.choice(session_ids) for _ in session_ids]
        denominator = sum(per_session[item]["targets"] for item in selected)
        if denominator == 0:
            # Conditional target recall is undefined for this cluster sample.
            # Draw a replacement so a zero-target sample cannot pull the
            # percentile interval artificially toward zero.
            zero_target_resamples_discarded += 1
            continue
        for top_k in TOP_KS:
            delta = sum(
                per_session[item]["pattern"][top_k]
                - per_session[item]["m0"][top_k]
                for item in selected
            )
            samples[top_k].append(delta / denominator)

    result: dict[str, Any] = {
        "estimand": "pattern_cache_gated_minus_M0_exact_target_recall",
        "resampling_unit": "whole heldout session",
        "session_count": len(session_ids),
        "sessions_with_visit_targets": sum(
            item["targets"] > 0 for item in per_session.values()
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": replicates,
        "valid_bootstrap_replicates": len(samples[TOP_KS[0]]),
        "zero_target_resamples_discarded": zero_target_resamples_discarded,
        "conditional_recall_defined": target_total > 0,
        "top_k": {},
    }
    for top_k in TOP_KS:
        baseline_hits = sum(item["m0"][top_k] for item in per_session.values())
        pattern_hits = sum(item["pattern"][top_k] for item in per_session.values())
        ordered = sorted(samples[top_k])
        interval: list[float | None]
        probability_positive: float | None
        if ordered:
            interval = [percentile(ordered, 0.025), percentile(ordered, 0.975)]
            probability_positive = sum(value > 0 for value in ordered) / len(ordered)
        else:
            interval = [None, None]
            probability_positive = None
        result["top_k"][str(top_k)] = {
            "m0_hits": baseline_hits,
            "pattern_hits": pattern_hits,
            "delta_hits": pattern_hits - baseline_hits,
            "delta_target_recall": (
                (pattern_hits - baseline_hits) / target_total
                if target_total
                else None
            ),
            "paired_session_bootstrap_95_percentile_interval": interval,
            "bootstrap_probability_delta_gt_zero": probability_positive,
        }
    return result


def _metric_cell(metrics: Mapping[str, Any], top_k: int) -> str:
    item = metrics["exact_top_k"][str(top_k)]
    return (
        f"{item['hits']}/{metrics['conditional_visit_targets']} "
        f"({item['target_recall'] * 100:.1f}%)"
    )


def render_report(payload: Mapping[str, Any], *, new_holdout: bool) -> str:
    if new_holdout:
        metrics = payload["evaluation"]
        title = "# One-shot new whole-session holdout"
        inference = payload["paired_session_bootstrap"]
        strict_metrics = None
    else:
        metrics = payload["all100_grouped_5fold_oof"]["metrics"]
        title = "# Pattern-cache development evaluation"
        inference = None
        strict_metrics = payload["strict_old70_grouped_5fold_oof"]["metrics"]
    baseline = metrics["models"]["M0_current_blind"]
    ungated = metrics["models"]["pattern_cache_ungated"]
    gated = metrics["models"]["pattern_cache_gated"]
    gate = metrics["gate"]
    lines = [
        title,
        "",
        "| Policy | Top-1 | Top-3 | Top-5 |",
        "|---|---:|---:|---:|",
        f"| M0 current-only blind | {_metric_cell(baseline, 1)} | {_metric_cell(baseline, 3)} | {_metric_cell(baseline, 5)} |",
        f"| Pattern cache, ungated | {_metric_cell(ungated, 1)} | {_metric_cell(ungated, 3)} | {_metric_cell(ungated, 5)} |",
        f"| Pattern cache, gated dispatch | {_metric_cell(gated, 1)} | {_metric_cell(gated, 3)} | {_metric_cell(gated, 5)} |",
        "",
        (
            f"Gate confusion: TP={gate['tp']}, FP={gate['fp']}, "
            f"TN={gate['tn']}, FN={gate['fn']}; precision="
            f"{gate['precision'] * 100:.1f}%, recall={gate['recall'] * 100:.1f}%."
        ),
        (
            f"The abstain rule fired {gate['fired_count']} times and reduced URL "
            f"dispatches {gate['ungated_url_dispatches']}→"
            f"{gate['gated_url_dispatches']} "
            f"(-{gate['dispatch_reduction_absolute']}, "
            f"{(gate['dispatch_reduction_fraction'] or 0.0) * 100:.1f}%)."
        ),
        (
            "Gate gated-minus-ungated exact-hit deltas: "
            + ", ".join(
                f"Top-{top_k} "
                f"{gate['gated_vs_ungated_exact_top_k'][str(top_k)]['delta_hits']:+d}"
                for top_k in TOP_KS
            )
            + "."
        ),
        "",
        (
            "Top-5 all-window precision/waste: M0 "
            f"{baseline['all_window_top5']['precision'] * 100:.1f}%/"
            f"{baseline['all_window_top5']['waste']}; ungated "
            f"{ungated['all_window_top5']['precision'] * 100:.1f}%/"
            f"{ungated['all_window_top5']['waste']}; gated "
            f"{gated['all_window_top5']['precision'] * 100:.1f}%/"
            f"{gated['all_window_top5']['waste']}."
        ),
    ]
    if strict_metrics is not None:
        strict_baseline = strict_metrics["models"]["M0_current_blind"]
        strict_ungated = strict_metrics["models"]["pattern_cache_ungated"]
        strict_gated = strict_metrics["models"]["pattern_cache_gated"]
        lines.extend(
            [
                "",
                "Strict historical 70-session grouped OOF:",
                "",
                "| Policy | Top-1 | Top-3 | Top-5 |",
                "|---|---:|---:|---:|",
                f"| M0 current-only blind | {_metric_cell(strict_baseline, 1)} | {_metric_cell(strict_baseline, 3)} | {_metric_cell(strict_baseline, 5)} |",
                f"| Pattern cache, ungated | {_metric_cell(strict_ungated, 1)} | {_metric_cell(strict_ungated, 3)} | {_metric_cell(strict_ungated, 5)} |",
                f"| Pattern cache, gated dispatch | {_metric_cell(strict_gated, 1)} | {_metric_cell(strict_gated, 3)} | {_metric_cell(strict_gated, 5)} |",
            ]
        )
    if not new_holdout:
        outer = payload["retrospective_outer30_fit70"]
        outer_metrics = outer["metrics"]

        def append_depth_table(
            heading: str, source_metrics: Mapping[str, Any]
        ) -> None:
            diagnostic = source_metrics["ranking_depth_diagnostic"]
            models = diagnostic["models"]
            lines.extend(
                [
                    "",
                    heading,
                    "",
                    "| Policy | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for label, key in (
                ("M0 current-only blind", "M0_current_blind"),
                ("Pattern cache, ungated", "pattern_cache_ungated"),
                ("Pattern cache, gated", "pattern_cache_gated"),
            ):
                model = models[key]
                cells = " | ".join(
                    _metric_cell(model, top_k)
                    for top_k in DIAGNOSTIC_TOP_KS
                )
                lines.append(f"| {label} | {cells} |")

        append_depth_table(
            (
                "Retrospective fixed 70→30 ranking-depth diagnostic "
                "(the split containing the quoted 19.3%/43.2%/55.7%):"
            ),
            outer_metrics,
        )
        append_depth_table(
            "All-100 whole-session grouped-OOF ranking-depth diagnostic:",
            metrics,
        )
        outer_depth = outer_metrics["ranking_depth_diagnostic"]
        outer_oracle = outer_depth["candidate_oracle_hits"]
        outer_dispatch = outer_depth["hypothetical_gate_dispatch_reduction"]
        lines.extend(
            [
                "",
                (
                    "Outer30 Top-10/20 candidate oracles: current-response "
                    f"Top-10={outer_oracle['current_response']['10']}, "
                    f"Top-20={outer_oracle['current_response']['20']}; "
                    "bounded-cache "
                    f"Top-10={outer_oracle['bounded_cache']['10']}, "
                    f"Top-20={outer_oracle['bounded_cache']['20']} "
                    "(out of 88 targets)."
                ),
                (
                    "At diagnostic depth, the gate would reduce all-window URL "
                    f"predictions Top-10 {outer_dispatch['10']['ungated']}→"
                    f"{outer_dispatch['10']['gated']} and Top-20 "
                    f"{outer_dispatch['20']['ungated']}→"
                    f"{outer_dispatch['20']['gated']}."
                ),
                "",
                (
                    "Top-10/20 are offline non-dispatch diagnostics. The frozen "
                    "v2 runtime still emits at most Top-5, and every expanded "
                    "ranking is prefix-checked against that runtime output."
                ),
            ]
        )
    if inference is not None:
        lines.extend(
            [
                "",
                "| K | Gated delta hits | Gated delta recall | Paired session-bootstrap 95% CI |",
                "|---:|---:|---:|---:|",
            ]
        )
        for top_k in TOP_KS:
            item = inference["top_k"][str(top_k)]
            interval = item["paired_session_bootstrap_95_percentile_interval"]
            delta_recall = item["delta_target_recall"]
            delta_cell = (
                f"{delta_recall * 100:+.1f} pp"
                if delta_recall is not None
                else "N/A"
            )
            interval_cell = (
                f"[{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}] pp"
                if interval[0] is not None and interval[1] is not None
                else "N/A"
            )
            lines.append(
                f"| {top_k} | {item['delta_hits']:+d} | "
                f"{delta_cell} | {interval_cell} |"
            )
    ceilings = metrics["candidate_ceilings"]
    latency = payload["latency_benchmark"]["models"]["pattern_cache"]
    lines.extend(
        [
            "",
            "Candidate coverage among authoritative visit targets:",
            "",
            f"- Current response: {ceilings['current_response']['target_coverage'] * 100:.1f}%",
            f"- LRU64, age<=2, plus preserved M0 Top-1: {ceilings['history_lru64_age_le_2_plus_m0_top1']['target_coverage'] * 100:.1f}%",
            f"- After gate: {ceilings['after_visit_abstain_gate']['target_coverage'] * 100:.1f}%",
            "",
            (
                "Local pattern prediction latency: "
                f"p50={latency['p50_ms']:.3f} ms, p95={latency['p95_ms']:.3f} ms, "
                f"p99={latency['p99_ms']:.3f} ms, max={latency['max_ms']:.3f} ms."
            ),
            "",
            (
                "Trace-wide non-executable visit URL labels: "
                f"{payload['trace_visit_executability']['runtime_nonexecutable_url_labels']} "
                "(retained as exact-label misses when scored; never dispatched or cached)."
            ),
            "",
            "The policy uses exact raw URL equality and no embedding, neural network, or backpropagation.",
            "",
        ]
    )
    if new_holdout:
        binding = payload["collection_binding"]
        commits = payload["requested_vs_committed_tools"]
        acceptance = payload["acceptance"]
        data_check = acceptance["data_adequacy"]
        ranker_check = acceptance["ranker"]
        gate_check = acceptance["visit_abstain_gate"]
        runtime_check = acceptance["runtime"]

        def check_label(value: bool | None) -> str:
            if value is None:
                return "INCONCLUSIVE"
            return "PASS" if value else "FAIL"

        lines.extend(
            [
                (
                    f"Collection `{binding['collection_status']}` was bound by manifest "
                    f"SHA256 `{binding['manifest_sha256']}`; all "
                    f"{binding['source_count']} workload sessions, including "
                    f"{binding['failed_sessions']} failures, are bootstrap units."
                ),
                "",
                (
                    "Requested/committed tools: "
                    f"{commits['requested_tool_calls']} / "
                    f"{commits['committed_tool_results']}; uncommitted="
                    f"{commits['uncommitted_tool_calls']}. Requested/committed visit "
                    f"URLs: {commits['requested_visit_urls']}/"
                    f"{commits['committed_visit_urls']}."
                ),
                "",
                "## Confirmatory acceptance",
                "",
                f"Overall status: **{acceptance['status'].upper()}**.",
                "",
                (
                    f"- Data adequacy: {check_label(data_check['passed'])} "
                    f"({data_check['exact_target_total']} exact targets; "
                    f"{data_check['sessions_with_committed_search_decision']}/"
                    f"{data_check['total_manifest_sessions']} sessions with a "
                    "committed search decision)."
                ),
                (
                    f"- Ranker: {check_label(ranker_check['passed'])}; "
                    "gated-minus-M0 exact-hit deltas "
                    + ", ".join(
                        f"Top-{top_k} "
                        f"{ranker_check['gated_minus_m0_hits'][str(top_k)]:+d}"
                        for top_k in TOP_KS
                    )
                    + "."
                ),
                (
                    "- Visit-abstain gate: "
                    f"{check_label(gate_check['passed'])}; fired "
                    f"{gate_check['fired_count']} times, committed-visit recall "
                    f"{gate_check['committed_visit_window_recall'] * 100:.1f}%, "
                    "gated-minus-ungated exact-hit deltas "
                    + ", ".join(
                        f"Top-{top_k} "
                        f"{gate_check['gated_minus_ungated_hits'][str(top_k)]:+d}"
                        for top_k in TOP_KS
                    )
                    + "."
                ),
                (
                    f"- Runtime: {check_label(runtime_check['passed'])} "
                    f"(p99={runtime_check['p99_ms']:.3f} ms, "
                    f"max={runtime_check['max_ms']:.3f} ms; both must be "
                    "strictly below 100 ms)."
                ),
                *(
                    [
                        "",
                        "Inconclusive because: "
                        + ", ".join(acceptance["inconclusive_reasons"])
                        + ".",
                    ]
                    if acceptance["inconclusive_reasons"]
                    else []
                ),
                "",
            ]
        )
    return "\n".join(lines)


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:  # pragma: no cover
            os.close(descriptor)


def run_development(
    traces: Path,
    output: Path,
    *,
    expected_sessions: int | None = EXPECTED_DEVELOPMENT_SESSIONS,
    benchmark_passes: int = 20,
) -> dict[str, Any]:
    sessions = load_sessions(traces)
    if expected_sessions is not None and len(sessions) != expected_sessions:
        raise ValueError(
            f"development requires exactly {expected_sessions} old sessions; "
            f"found {len(sessions)}"
        )
    strict_old70, old_outer30 = split_sessions(
        sessions, train_ratio=0.70, seed=OUTER_SEED
    )
    if expected_sessions == 100 and (len(strict_old70), len(old_outer30)) != (70, 30):
        raise RuntimeError("strict historical split is not 70/30 whole sessions")

    strict_oof = grouped_oof(strict_old70)
    all_oof = grouped_oof(sessions)
    decisions = extract_search_decisions(sessions)
    strict_ids = {session.session_id for session in strict_old70}
    outer_ids = {session.session_id for session in old_outer30}
    strict_fit_decisions = [
        decision for decision in decisions if decision.session_id in strict_ids
    ]
    outer_decisions = [
        decision for decision in decisions if decision.session_id in outer_ids
    ]
    outer_pattern = fit_rank_pattern(strict_fit_decisions)
    outer_rows, outer_durations = score_decisions(
        outer_decisions, make_frozen_predictor(outer_pattern)
    )
    pattern = fit_rank_pattern(decisions)
    predictor = make_frozen_predictor(pattern)
    artifact = build_artifact(pattern, sessions)
    fitted_rows, _ = score_decisions(decisions, predictor)
    payload: dict[str, Any] = {
        "mode": "development_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal_boundary": (
            "Only same-session search responses and visits completed before each "
            "decision are features; generated decision responses and next tools are labels only."
        ),
        "frozen_policy": predictor.metadata(),
        "development_data": trace_manifest(sessions),
        "trace_visit_executability": visit_executability_inventory(sessions),
        "strict_old70": {
            "seed": OUTER_SEED,
            "session_count": len(strict_old70),
            "excluded_old_outer_sessions": len(old_outer30),
            "session_ids": sorted(item.session_id for item in strict_old70),
        },
        "strict_old70_grouped_5fold_oof": strict_oof,
        "retrospective_outer30_fit70": {
            "status": "post_hoc_descriptive_not_confirmatory",
            "reason": (
                "the outer30 outcomes were already inspected during policy "
                "development"
            ),
            "fit_sessions": len(strict_old70),
            "evaluation_sessions": len(old_outer30),
            "fit_rank_counts": {
                str(rank): count
                for rank, count in outer_pattern.rank_counts.items()
            },
            "mapped_current_response_targets": outer_pattern.total,
            "metrics": summarize_rows(outer_rows, durations=outer_durations),
        },
        "all100_grouped_5fold_oof": all_oof,
        "fitted_all100_descriptive": summarize_rows(fitted_rows),
        "latency_benchmark": benchmark(
            decisions, predictor, passes=benchmark_passes
        ),
        "artifact": {
            "filename": ARTIFACT_NAME,
            "artifact_sha256": artifact["artifact_sha256"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    save_pattern_artifact(output / ARTIFACT_NAME, artifact)
    write_json_atomic(output / DEVELOPMENT_METRICS_NAME, payload)
    (output / REPORT_NAME).write_text(
        render_report(payload, new_holdout=False), encoding="utf-8"
    )
    return payload


def run_new_evaluation(
    traces: Path,
    artifact_path: Path,
    output: Path,
    *,
    workload_path: Path = DEFAULT_NEW_WORKLOAD,
    bootstrap_replicates: int = 10_000,
    benchmark_passes: int = 20,
) -> dict[str, Any]:
    # Freeze policy and preregistered workload identity before claiming.  The
    # collection manifest and traces remain unread until O_EXCL succeeds.
    predictor, artifact = load_artifact(artifact_path)
    workload = load_expected_new_workload(workload_path)
    output = output.resolve()
    started_path = output / STARTED_NAME
    generated = (NEW_METRICS_NAME, REPORT_NAME, COMPLETE_NAME)
    if (
        started_path.exists()
        or any((output / name).exists() for name in generated)
        or (output.exists() and any(output.iterdir()))
    ):
        raise FileExistsError(
            f"new holdout evaluation was already started or has outputs in {output}"
        )
    started = {
        "schema": "paste_repro.pattern_cache_new_holdout_started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "workload_path": str(workload_path.resolve()),
        "workload_id": workload.workload_id,
        "workload_sha256": workload.file_sha256,
        "expected_source_count": len(workload.sources),
        "new_trace_directory": str(traces.resolve()),
        "claim": (
            "O_EXCL created before reading collection/manifest.json or any "
            "new holdout trace"
        ),
    }
    write_json_exclusive(started_path, started)

    # Any failure from here intentionally leaves STARTED in place: the data may
    # have been observed and this output directory cannot be reused.
    collection_manifest, sessions, collection_binding = (
        validate_collection_manifest(traces, workload)
    )
    decisions, commit_audit = extract_committed_search_decisions(sessions)
    rows, durations = score_decisions(decisions, predictor)
    evaluation = summarize_rows(rows, durations=durations)
    latency_benchmark = benchmark(
        decisions, predictor, passes=benchmark_passes
    )
    acceptance = new_holdout_acceptance(
        evaluation,
        latency_benchmark,
        total_manifest_sessions=len(sessions),
    )
    payload: dict[str, Any] = {
        "mode": "one_shot_new_whole_session_holdout",
        "artifact": {
            "path": str(artifact_path.resolve()),
            "artifact_sha256": artifact["artifact_sha256"],
            "training_manifest_sha256": artifact["training_manifest"]["manifest_sha256"],
        },
        "frozen_policy": predictor.metadata(),
        "collection_binding": collection_binding,
        "new_holdout_data": trace_manifest(sessions),
        "trace_visit_executability": visit_executability_inventory(sessions),
        "requested_vs_committed_tools": commit_audit,
        "failed_session_ids": [
            str(record["session_id"])
            for record in collection_manifest["sessions"]
            if record["status"] != "succeeded"
        ],
        "evaluation": evaluation,
        "latency_benchmark": latency_benchmark,
        "acceptance": acceptance,
        "paired_session_bootstrap": paired_session_bootstrap(
            rows,
            [session.session_id for session in sessions],
            replicates=bootstrap_replicates,
        ),
    }
    metrics_path = output / NEW_METRICS_NAME
    report_path = output / REPORT_NAME
    write_json_atomic(metrics_path, payload)
    report_path.write_text(render_report(payload, new_holdout=True), encoding="utf-8")
    complete = {
        "schema": "paste_repro.pattern_cache_new_holdout_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_sha256": sha256_file(started_path),
        "artifact_sha256": artifact["artifact_sha256"],
        "new_trace_manifest_sha256": payload["new_holdout_data"]["manifest_sha256"],
        "collection_manifest_sha256": collection_binding["manifest_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "report_sha256": sha256_file(report_path),
    }
    write_json_exclusive(output / COMPLETE_NAME, complete)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--develop", action="store_true")
    mode.add_argument("--evaluate-new", action="store_true")
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--benchmark-passes", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.develop:
        run_development(
            args.traces or DEFAULT_TRACES,
            args.output,
            benchmark_passes=args.benchmark_passes,
        )
        return 0
    if args.artifact is None:
        raise SystemExit("--evaluate-new requires --artifact")
    if args.workload is None:
        raise SystemExit("--evaluate-new requires --workload")
    if args.traces is None:
        raise SystemExit("--evaluate-new requires --traces pointing to collection/")
    run_new_evaluation(
        args.traces,
        args.artifact,
        args.output,
        workload_path=args.workload,
        bootstrap_replicates=args.bootstrap_replicates,
        benchmark_passes=args.benchmark_passes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
