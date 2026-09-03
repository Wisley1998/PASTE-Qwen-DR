#!/usr/bin/env python3
"""Evaluate globally allocated, authority-first Pattern-v2 speculation.

This CPU-only runner uses nested whole-session grouped OOF calibration, the
real shared-capacity LiveToolBroker, and deterministic synthetic tool service.
It starts no model server and issues no network requests.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.invocation import Invocation  # noqa: E402
from paste_repro.live_broker import LiveToolBroker  # noqa: E402
from paste_repro.speculation_policy import (  # noqa: E402
    AuthorityFirstUtilityPolicy,
    AuthorityLoad,
    CandidatePattern,
    CountPatternCalibrator,
    LabeledCandidatePattern,
    SafeGlobalBenefitConfig,
    SafeGlobalBenefitPolicy,
    SafeStartBudget,
    UtilityCandidate,
    UtilityPolicyConfig,
)
from paste_repro.traces import load_sessions  # noqa: E402
from run_pattern_cache_evaluation import (  # noqa: E402
    cv_fold,
    extract_search_decisions,
    fit_rank_pattern,
    make_frozen_predictor,
    sha256_file,
)
from run_pattern_v2_load_robustness import (  # noqa: E402
    DEFAULT_WIDTHS,
    bounded_pool_oracle_metrics,
    canonical_sha256,
    collect_pattern_v2_oof_rows,
    percentile,
    ratio,
    stable_order,
    static_width_metrics,
)


SCHEMA = "paste_repro.pattern_v2_adaptive_load.v2"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "results" / "pattern_v2_adaptive_load"
DEFAULT_CONCURRENCIES = (1, 8, 32, 64, 128)
POLICIES = (
    "rank5_sequential_unreserved",
    "rank5_batch_reserved",
    "rank_budgeted_round_robin_reserved",
    "confidence_global_reserved",
    "utility_global_authority_first",
    "utility_global_risk_limited",
    "safe_global_benefit",
)


@dataclass(frozen=True)
class RawCandidate:
    pattern: CandidatePattern
    next_tool_visit: bool
    exact_match: bool


@dataclass(frozen=True)
class RawWindow:
    decision_id: str
    session_id: str
    v2_gate: bool
    next_tool_visit: bool
    targets: tuple[str, ...]
    executable_targets: tuple[str, ...]
    candidates: tuple[RawCandidate, ...]


@dataclass(frozen=True)
class ScoredCandidate:
    pattern: CandidatePattern
    exact_probability: float
    visit_probability: float
    rank_only_probability: float
    exact_match: bool


@dataclass(frozen=True)
class ScoredWindow:
    decision_id: str
    session_id: str
    v2_gate: bool
    next_tool_visit: bool
    expected_authoritative_calls: float
    coarse_expected_authoritative_calls: float
    targets: tuple[str, ...]
    executable_targets: tuple[str, ...]
    candidates: tuple[ScoredCandidate, ...]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    batch_admission: bool
    max_speculative_workers: int | None
    visit_authoritative_reserve: int
    confidence_threshold: float | None = None
    utility_config: UtilityPolicyConfig | None = None
    requires_isolated_capacity: bool = False


def inner_fold(session_id: str) -> int:
    encoded = f"pattern-confidence-inner-v1\0{session_id}".encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest(), 16) % 4


def executable_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _generate_raw_windows(
    decisions: Sequence[Any],
    *,
    fit_ids: set[str],
    evaluation_ids: set[str],
    runtime_durations_ms: list[float] | None = None,
) -> list[RawWindow]:
    fit = [row for row in decisions if row.session_id in fit_ids]
    validation = [
        row for row in decisions if row.session_id in evaluation_ids
    ]
    predictor = make_frozen_predictor(fit_rank_pattern(fit))
    states = {
        session_id: predictor.start_session(session_id)
        for session_id in evaluation_ids
    }
    windows: list[RawWindow] = []
    for decision in validation:
        runtime_started = time.perf_counter_ns()
        state = states[decision.session_id]
        for tool_name, urls in decision.prior_tool_updates:
            if tool_name == "visit":
                executable = tuple(url for url in urls if executable_url(url))
                if executable:
                    state.observe_visit(executable)
                else:
                    state.observe_other_tool("invalid_visit")
            elif tool_name == "search":
                raise RuntimeError("a search update cannot precede its decision")
            else:
                state.observe_other_tool(tool_name)

        runtime = state.observe_search(
            decision.current_results,
            query_count=(decision.query_count if decision.query_count > 0 else None),
        )
        targets = (
            tuple(decision.authoritative_urls)
            if decision.outcome == "visit"
            else ()
        )
        target_set = set(targets)
        current_repetitions = Counter(
            result.url for result in decision.current_results
        )
        candidates: list[RawCandidate] = []
        for position, candidate in enumerate(runtime.ranked_top_k, 1):
            pattern = CandidatePattern(
                session_id=decision.session_id,
                decision_id=decision.decision_id,
                url=candidate.url,
                position=position,
                query_count=decision.query_count,
                search_streak=decision.consecutive_search_streak,
                search_sequence=runtime.search_sequence,
                candidate_count=runtime.candidate_count,
                current_count=len(
                    {result.url for result in decision.current_results}
                ),
                repeated_current=(
                    current_repetitions[candidate.url] >= 2
                ),
                source_rank=candidate.source_rank,
                current=candidate.current,
                was_visited=candidate.was_visited,
                search_age=candidate.search_age,
                appearances=candidate.appearances,
            )
            candidates.append(
                RawCandidate(
                    pattern=pattern,
                    next_tool_visit=decision.outcome == "visit",
                    exact_match=candidate.url in target_set,
                )
            )
        windows.append(
            RawWindow(
                decision_id=decision.decision_id,
                session_id=decision.session_id,
                v2_gate=runtime.gate.admitted,
                next_tool_visit=decision.outcome == "visit",
                targets=targets,
                executable_targets=tuple(
                    url for url in targets if executable_url(url)
                ),
                candidates=tuple(candidates),
            )
        )
        if runtime_durations_ms is not None:
            runtime_durations_ms.append(
                (time.perf_counter_ns() - runtime_started) / 1_000_000.0
            )
    return windows


def collect_nested_oof_windows(
    traces: Path,
) -> tuple[list[ScoredWindow], dict[str, Any]]:
    sessions = load_sessions(traces)
    decisions = extract_search_decisions(sessions)
    session_ids = {session.session_id for session in sessions}
    result: list[ScoredWindow] = []
    folds: list[dict[str, Any]] = []
    prediction_durations_ms: list[float] = []
    feature_durations_ms: list[float] = []
    for outer in range(5):
        train_ids = {
            session_id
            for session_id in session_ids
            if cv_fold(session_id) != outer
        }
        validation_ids = session_ids - train_ids
        calibration_rows: list[LabeledCandidatePattern] = []
        calibration_windows: list[RawWindow] = []
        for inner in range(4):
            inner_validation = {
                session_id
                for session_id in train_ids
                if inner_fold(session_id) == inner
            }
            inner_fit = train_ids - inner_validation
            inner_windows = _generate_raw_windows(
                decisions,
                fit_ids=inner_fit,
                evaluation_ids=inner_validation,
            )
            calibration_windows.extend(inner_windows)
            calibration_rows.extend(
                LabeledCandidatePattern(
                    candidate.pattern,
                    candidate.next_tool_visit,
                    candidate.exact_match,
                )
                for window in inner_windows
                for candidate in window.candidates
            )
        calibrator = CountPatternCalibrator(calibration_rows)
        calibration_visits = sum(
            window.next_tool_visit for window in calibration_windows
        )
        mean_executable_targets_per_visit = ratio(
            sum(
                len(window.executable_targets)
                for window in calibration_windows
                if window.next_tool_visit
            ),
            calibration_visits,
        )
        coarse_executable_calls_per_window = ratio(
            sum(
                len(window.executable_targets)
                for window in calibration_windows
            ),
            len(calibration_windows),
        )
        validation = _generate_raw_windows(
            decisions,
            fit_ids=train_ids,
            evaluation_ids=validation_ids,
            runtime_durations_ms=feature_durations_ms,
        )
        fold_hits = 0
        for window in validation:
            scored: list[ScoredCandidate] = []
            window_visit_probability = calibrator.visit_global
            for candidate in window.candidates:
                started = time.perf_counter_ns()
                exact_probability = calibrator.exact_probability(
                    candidate.pattern
                )
                visit_probability = calibrator.visit_probability(
                    candidate.pattern
                )
                rank_probability = calibrator.rank_only_probability(
                    candidate.pattern
                )
                prediction_durations_ms.append(
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
                window_visit_probability = visit_probability
                fold_hits += int(candidate.exact_match)
            result.append(
                ScoredWindow(
                    decision_id=window.decision_id,
                    session_id=window.session_id,
                    v2_gate=window.v2_gate,
                    next_tool_visit=window.next_tool_visit,
                    expected_authoritative_calls=(
                        window_visit_probability
                        * mean_executable_targets_per_visit
                    ),
                    coarse_expected_authoritative_calls=(
                        coarse_executable_calls_per_window
                    ),
                    targets=window.targets,
                    executable_targets=window.executable_targets,
                    candidates=tuple(scored),
                )
            )
        folds.append(
            {
                "outer_fold": outer,
                "train_sessions": len(train_ids),
                "validation_sessions": len(validation_ids),
                "inner_calibration_rows": len(calibration_rows),
                "validation_windows": len(validation),
                "validation_candidates": sum(
                    len(window.candidates) for window in validation
                ),
                "validation_candidate_hits": fold_hits,
                "calibrator": calibrator.summary(),
                "mean_executable_targets_per_visit": (
                    mean_executable_targets_per_visit
                ),
                "coarse_executable_calls_per_window": (
                    coarse_executable_calls_per_window
                ),
            }
        )
    if len(result) != len(decisions):
        raise RuntimeError("nested OOF did not score every decision")
    if len({window.decision_id for window in result}) != len(result):
        raise RuntimeError("nested OOF produced duplicate decision ids")
    metadata = {
        "method": "outer-5-fold and inner-4-fold whole-session grouped OOF",
        "session_count": len(sessions),
        "window_count": len(result),
        "candidate_count": sum(len(window.candidates) for window in result),
        "candidate_hits": sum(
            candidate.exact_match
            for window in result
            for candidate in window.candidates
        ),
        "folds": folds,
        "runtime_probability_lookup_ms": {
            "calls": len(prediction_durations_ms),
            "total": sum(prediction_durations_ms),
            "mean": statistics.fmean(prediction_durations_ms),
            "p95": percentile(prediction_durations_ms, 0.95),
            "p99": percentile(prediction_durations_ms, 0.99),
            "max": max(prediction_durations_ms),
        },
        "runtime_pattern_feature_ms": {
            "calls": len(feature_durations_ms),
            "total": sum(feature_durations_ms),
            "mean": statistics.fmean(feature_durations_ms),
            "p95": percentile(feature_durations_ms, 0.95),
            "p99": percentile(feature_durations_ms, 0.99),
            "max": max(feature_durations_ms),
        },
    }
    return result, metadata


def force_all_wrong(windows: Sequence[ScoredWindow]) -> list[ScoredWindow]:
    result: list[ScoredWindow] = []
    for window in windows:
        targets = tuple(
            "https://adaptive-all-wrong.invalid/"
            + hashlib.sha256(
                f"{window.decision_id}\0{index}\0{url}".encode("utf-8")
            ).hexdigest()
            for index, url in enumerate(window.executable_targets)
        )
        if set(targets).intersection(
            candidate.pattern.url for candidate in window.candidates
        ):
            raise RuntimeError("all-wrong target matched a candidate")
        result.append(
            replace(
                window,
                targets=targets,
                executable_targets=targets,
                candidates=tuple(
                    replace(candidate, exact_match=False)
                    for candidate in window.candidates
                ),
            )
        )
    return result


def policy_specs() -> tuple[PolicySpec, ...]:
    return (
        PolicySpec(
            "rank5_sequential_unreserved",
            batch_admission=False,
            max_speculative_workers=2,
            visit_authoritative_reserve=0,
        ),
        PolicySpec(
            "rank5_batch_reserved",
            batch_admission=True,
            max_speculative_workers=1,
            visit_authoritative_reserve=1,
        ),
        PolicySpec(
            "rank_budgeted_round_robin_reserved",
            batch_admission=True,
            max_speculative_workers=1,
            visit_authoritative_reserve=1,
        ),
        PolicySpec(
            "confidence_global_reserved",
            batch_admission=True,
            max_speculative_workers=1,
            visit_authoritative_reserve=1,
            confidence_threshold=0.10,
        ),
        PolicySpec(
            "utility_global_authority_first",
            batch_admission=True,
            max_speculative_workers=1,
            visit_authoritative_reserve=1,
            utility_config=UtilityPolicyConfig(),
        ),
        PolicySpec(
            "utility_global_risk_limited",
            batch_admission=True,
            max_speculative_workers=1,
            visit_authoritative_reserve=1,
            confidence_threshold=0.20,
            utility_config=UtilityPolicyConfig(),
        ),
        PolicySpec(
            "safe_global_benefit",
            batch_admission=True,
            max_speculative_workers=None,
            visit_authoritative_reserve=0,
            confidence_threshold=0.0,
            requires_isolated_capacity=True,
        ),
    )


def _selection_budget(
    spec: PolicySpec,
    *,
    visit_capacity: int,
    service_s: float,
    lead_s: float,
    isolated_speculative_slots: int = 0,
    safe_start_limit: int | None = None,
) -> int:
    if lead_s <= 0.0:
        return 0
    speculative_tool_slots = (
        isolated_speculative_slots
        if spec.requires_isolated_capacity
        else min(
            int(spec.max_speculative_workers or 0),
            max(0, visit_capacity - spec.visit_authoritative_reserve),
        )
    )
    starts_per_slot = max(1, math.ceil(lead_s / service_s))
    budget = speculative_tool_slots * starts_per_slot
    if spec.requires_isolated_capacity and safe_start_limit is not None:
        return min(budget, safe_start_limit)
    return budget


def _select_candidates(
    batch: Sequence[ScoredWindow],
    spec: PolicySpec,
    *,
    visit_capacity: int,
    service_s: float,
    lead_s: float,
    isolated_speculative_slots: int = 0,
    safe_start_limit: int | None = None,
    coordination_cost_s: float = 0.0,
) -> tuple[list[tuple[ScoredCandidate, float]], dict[str, Any]]:
    started = time.perf_counter_ns()
    considered = [
        candidate
        for window in batch
        for candidate in window.candidates
    ]
    probability_candidates_evaluated = 0
    selected: list[tuple[ScoredCandidate, float]]
    load_pressure = 0.0
    shadow_price = 0.0
    positive_utility = 0
    coarse_load_pressure = ratio(
        sum(window.coarse_expected_authoritative_calls for window in batch),
        visit_capacity,
    )
    coarse_load_kill_switch = False
    predictor_windows_evaluated = len(batch)
    selection_reason_counts: Counter[str] = Counter()
    if spec.name in {
        "rank5_sequential_unreserved",
        "rank5_batch_reserved",
    }:
        selected = [
            (candidate, 1.0 / candidate.pattern.position)
            for window in batch
            if window.v2_gate
            for candidate in window.candidates
        ]
    elif spec.name == "rank_budgeted_round_robin_reserved":
        budget = _selection_budget(
            spec,
            visit_capacity=visit_capacity,
            service_s=service_s,
            lead_s=lead_s,
        )
        rank_major = sorted(
            (
                candidate
                for window in batch
                if window.v2_gate
                for candidate in window.candidates
            ),
            key=lambda candidate: (
                candidate.pattern.position,
                hashlib.sha256(
                    (
                        f"{candidate.pattern.session_id}\0"
                        f"{candidate.pattern.decision_id}"
                    ).encode("utf-8")
                ).hexdigest(),
            ),
        )
        selected = [
            (candidate, 1.0 / candidate.pattern.position)
            for candidate in rank_major[:budget]
        ]
    elif spec.name == "confidence_global_reserved":
        probability_candidates_evaluated = len(considered)
        threshold = float(spec.confidence_threshold or 0.0)
        budget = _selection_budget(
            spec,
            visit_capacity=visit_capacity,
            service_s=service_s,
            lead_s=lead_s,
        )
        eligible = [
            candidate
            for candidate in considered
            if candidate.exact_probability >= threshold
        ]
        eligible.sort(
            key=lambda candidate: (
                -candidate.exact_probability,
                hashlib.sha256(
                    (
                        f"{candidate.pattern.session_id}\0"
                        f"{candidate.pattern.decision_id}\0"
                        f"{candidate.pattern.url}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
        )
        selected = [
            (candidate, candidate.exact_probability)
            for candidate in eligible[:budget]
        ]
    elif spec.name in {
        "utility_global_authority_first",
        "utility_global_risk_limited",
    }:
        controller = AuthorityFirstUtilityPolicy(spec.utility_config)
        if coarse_load_pressure > controller.config.high_pressure:
            coarse_load_kill_switch = True
            predictor_windows_evaluated = 0
            probability_candidates_evaluated = 0
            considered = []
            selected = []
            coarse_load = AuthorityLoad(
                expected_authoritative_calls=(
                    coarse_load_pressure * visit_capacity
                ),
                tool_capacity=visit_capacity,
            )
            load_pressure = coarse_load.pressure
            shadow_price = controller.shadow_price(coarse_load)
        else:
            probability_candidates_evaluated = len(considered)
            probability_floor = float(spec.confidence_threshold or 0.0)
            if probability_floor > 0.0:
                considered = [
                    candidate
                    for candidate in considered
                    if candidate.exact_probability >= probability_floor
                ]
            expected_authoritative = sum(
                window.expected_authoritative_calls for window in batch
            )
            load = AuthorityLoad(
                expected_authoritative_calls=expected_authoritative,
                tool_capacity=visit_capacity,
            )
            budget = _selection_budget(
                spec,
                visit_capacity=visit_capacity,
                service_s=service_s,
                lead_s=lead_s,
            )
            by_identity = {
                (
                    candidate.pattern.session_id,
                    candidate.pattern.decision_id,
                    candidate.pattern.url,
                ): candidate
                for candidate in considered
            }
            decision = controller.select(
                tuple(
                    UtilityCandidate(
                        pattern=candidate.pattern,
                        exact_probability=candidate.exact_probability,
                        estimated_service_s=service_s,
                        lead_remaining_s=lead_s,
                    )
                    for candidate in considered
                ),
                load=load,
                start_budget=budget,
            )
            selected = [
                (
                    by_identity[
                        (
                            row.candidate.pattern.session_id,
                            row.candidate.pattern.decision_id,
                            row.candidate.pattern.url,
                        )
                    ],
                    row.utility_density,
                )
                for row in decision.selected
            ]
            load_pressure = decision.load_pressure
            shadow_price = decision.shadow_price
            positive_utility = sum(
                row.net_utility_s > 0.0 for row in decision.decisions
            )
    elif spec.name == "safe_global_benefit":
        budget = _selection_budget(
            spec,
            visit_capacity=visit_capacity,
            service_s=service_s,
            lead_s=lead_s,
            isolated_speculative_slots=isolated_speculative_slots,
            safe_start_limit=safe_start_limit,
        )
        if budget == 0:
            selection_reason_counts["no_safe_capacity"] += len(considered)
            predictor_windows_evaluated = 0
            probability_candidates_evaluated = 0
            considered = []
            selected = []
        else:
            probability_candidates_evaluated = len(considered)
            probability_floor = float(spec.confidence_threshold or 0.0)
            considered = [
                candidate
                for candidate in considered
                if candidate.exact_probability >= probability_floor
            ]
            by_identity = {
                (
                    candidate.pattern.session_id,
                    candidate.pattern.decision_id,
                    candidate.pattern.url,
                ): candidate
                for candidate in considered
            }
            decision = SafeGlobalBenefitPolicy(
                SafeGlobalBenefitConfig(
                    coordination_cost_s=coordination_cost_s,
                )
            ).select(
                tuple(
                    UtilityCandidate(
                        pattern=candidate.pattern,
                        exact_probability=candidate.exact_probability,
                        estimated_service_s=service_s,
                        lead_remaining_s=lead_s,
                    )
                    for candidate in considered
                ),
                safe_budget=SafeStartBudget(budget),
                requested_start_budget=budget,
            )
            selected = [
                (
                    by_identity[
                        (
                            row.candidate.pattern.session_id,
                            row.candidate.pattern.decision_id,
                            row.candidate.pattern.url,
                        )
                    ],
                    row.utility_density,
                )
                for row in decision.selected
            ]
            positive_utility = sum(
                row.net_utility_s > 0.0 for row in decision.decisions
            )
            selection_reason_counts.update(
                row.reason for row in decision.decisions if not row.selected
            )
    else:  # pragma: no cover - protected by CLI validation
        raise ValueError(f"unsupported policy: {spec.name}")

    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    metadata = {
        "considered": len(considered),
        "probability_candidates_evaluated": (
            probability_candidates_evaluated
        ),
        "selected": len(selected),
        "selected_hits": sum(candidate.exact_match for candidate, _ in selected),
        "selected_probability_sum": sum(
            candidate.exact_probability for candidate, _ in selected
        ),
        "selected_positions": dict(
            sorted(
                Counter(
                    candidate.pattern.position for candidate, _ in selected
                ).items()
            )
        ),
        "load_pressure": load_pressure,
        "shadow_price": shadow_price,
        "positive_utility": positive_utility,
        "coordination_cost_s": coordination_cost_s,
        "coarse_load_pressure": coarse_load_pressure,
        "coarse_load_kill_switch": coarse_load_kill_switch,
        "safe_start_budget": (
            _selection_budget(
                spec,
                visit_capacity=visit_capacity,
                service_s=service_s,
                lead_s=lead_s,
                isolated_speculative_slots=isolated_speculative_slots,
                safe_start_limit=safe_start_limit,
            )
            if spec.requires_isolated_capacity
            else None
        ),
        "selection_reason_counts": dict(sorted(selection_reason_counts.items())),
        "predictor_windows_evaluated": predictor_windows_evaluated,
        "compute_ms": elapsed_ms,
    }
    return selected, metadata


def session_stream_batches(
    windows: Sequence[ScoredWindow],
    *,
    offered_concurrency: int,
    seed: int,
) -> list[list[ScoredWindow]]:
    """Create closed-loop task batches without overlapping one source session.

    Each source session is one task stream.  Only its current head decision can
    be active; a replacement task enters when an active source session drains.
    """

    streams: dict[str, deque[ScoredWindow]] = {}
    for window in windows:
        streams.setdefault(window.session_id, deque()).append(window)
    waiting = deque(
        sorted(
            streams,
            key=lambda session_id: (
                stable_order(seed, session_id),
                session_id,
            ),
        )
    )
    active: list[str] = []
    while waiting and len(active) < offered_concurrency:
        active.append(waiting.popleft())
    batches: list[list[ScoredWindow]] = []
    while active:
        batches.append([streams[session_id].popleft() for session_id in active])
        survivors = [
            session_id for session_id in active if streams[session_id]
        ]
        while waiting and len(survivors) < offered_concurrency:
            survivors.append(waiting.popleft())
        active = survivors
    if sum(len(batch) for batch in batches) != len(windows):
        raise RuntimeError("session-stream batching lost decisions")
    if any(
        len({window.session_id for window in batch}) != len(batch)
        for batch in batches
    ):
        raise RuntimeError("one source session appeared twice in a batch")
    return batches


async def _run_sample(
    windows: Sequence[ScoredWindow],
    *,
    policy: PolicySpec | None,
    offered_concurrency: int,
    seed: int,
    workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
    isolated_speculative_slots: int = 0,
) -> dict[str, Any]:
    service_s = service_ms / 1000.0
    lead_s = lead_ms / 1000.0

    async def executor(invocation: Invocation) -> dict[str, Any]:
        physical = asyncio.create_task(asyncio.sleep(service_s))
        try:
            await asyncio.shield(physical)
        except asyncio.CancelledError:
            await asyncio.shield(physical)
            raise
        return {"invocation_key": invocation.key}

    strict_policy = bool(
        policy is not None and policy.requires_isolated_capacity
    )
    certified_isolated_slots = (
        isolated_speculative_slots
        if strict_policy and policy is not None
        else 0
    )
    speculate = bool(
        policy is not None
        and (not strict_policy or certified_isolated_slots > 0)
    )
    strict_isolation = strict_policy and speculate
    safe_epoch_start_budget = (
        certified_isolated_slots
        * max(1, math.ceil(lead_s / service_s))
        if strict_isolation and lead_s > 0.0
        else 0
    )
    broker_speculative_pending = (
        min(max_speculative_pending, max(1, safe_epoch_start_budget))
        if strict_isolation
        else max_speculative_pending
    )
    broker_workers = workers + certified_isolated_slots
    broker_visit_capacity = visit_capacity + certified_isolated_slots
    speculative_workers = (
        certified_isolated_slots
        if strict_isolation
        else (int(policy.max_speculative_workers or 0) if policy else 0)
    )
    # In strict mode the original visit capacity remains an authority-only
    # entitlement. Speculation can consume only the explicitly added slice.
    reserve = (
        visit_capacity
        if strict_isolation
        else (policy.visit_authoritative_reserve if policy else 0)
    )
    broker = LiveToolBroker(
        executor,
        max_workers=broker_workers,
        max_speculative_workers=speculative_workers,
        max_authoritative_workers=(workers if strict_isolation else None),
        min_speculative_workers=0,
        max_speculative_pending=broker_speculative_pending,
        ttl_s=1.0,
        tool_capacities={"visit": broker_visit_capacity},
        authoritative_tool_capacities=(
            {"visit": visit_capacity} if strict_isolation else None
        ),
        authoritative_tool_reserves=(
            {"visit": reserve} if reserve else None
        ),
    )
    batches = session_stream_batches(
        windows,
        offered_concurrency=offered_concurrency,
        seed=seed,
    )
    requested = 0
    admission_results: list[bool] = []
    authoritative_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    admission_ms: list[float] = []
    deadline_offsets_ms: list[float] = []
    deadline_overruns = 0
    deadline_skipped_predictions = 0
    deadline_by_session: dict[str, float] = {}
    batch_sizes: list[int] = []
    deferred_cleanup: list[asyncio.Task[int]] = []
    wall_started = time.perf_counter()
    try:
        for batch in batches:
            batch_sizes.append(len(batch))
            session_ids = [
                f"r{seed}:{window.session_id}:{window.decision_id}"
                for window in batch
            ]
            session_by_decision = {
                (window.session_id, window.decision_id): session_id
                for window, session_id in zip(batch, session_ids)
            }
            decision_started = time.perf_counter()
            start_deadline = decision_started + lead_s
            if speculate and policy is not None:
                selected, selection = _select_candidates(
                    batch,
                    policy,
                    visit_capacity=visit_capacity,
                    service_s=service_s,
                    lead_s=lead_s,
                    isolated_speculative_slots=certified_isolated_slots,
                    safe_start_limit=broker_speculative_pending,
                )
                selection_rows.append(selection)
                requests = [
                    (
                        Invocation(
                            "visit", {"url": candidate.pattern.url}
                        ),
                        session_by_decision[
                            (
                                candidate.pattern.session_id,
                                candidate.pattern.decision_id,
                            )
                        ],
                        priority,
                    )
                    for candidate, priority in selected
                ]
                if time.perf_counter() >= start_deadline:
                    deadline_skipped_predictions += len(requests)
                    requests = []
                requested += len(requests)
                admission_started = time.perf_counter()
                if policy.batch_admission:
                    admission_results.extend(
                        await broker.speculate_batch(
                            tuple(requests),
                            start_deadline=start_deadline,
                            replace_lower_priority_queued=strict_isolation,
                        )
                    )
                else:
                    for invocation, session_id, priority in requests:
                        admission_results.append(
                            await broker.speculate(
                                invocation,
                                session_id=session_id,
                                priority=priority,
                                start_deadline=start_deadline,
                                replace_lower_priority_queued=(
                                    strict_isolation
                                ),
                            )
                        )
                admission_ms.append(
                    (time.perf_counter() - admission_started) * 1000.0
                )

            remaining_lead = start_deadline - time.perf_counter()
            if remaining_lead > 0.0:
                await asyncio.sleep(remaining_lead)
            elif speculate:
                deadline_overruns += 1
            confirmation_offset = time.perf_counter() - decision_started
            deadline_offsets_ms.append(confirmation_offset * 1000.0)
            for session_id in session_ids:
                deadline_by_session[session_id] = decision_started + lead_s

            call_specs = [
                (window, session_id, target_index, target)
                for window, session_id in zip(batch, session_ids)
                for target_index, target in enumerate(window.executable_targets)
            ]
            calls = [
                asyncio.create_task(
                    broker.authoritative(
                        Invocation("visit", {"url": target}),
                        session_id=session_id,
                        reuse_running_speculation=not strict_isolation,
                    )
                )
                for _, session_id, _, target in call_specs
            ]
            early_cleanup = [
                asyncio.create_task(
                    broker.cancel_predictions(
                        session_id=session_id,
                        keep=(
                            Invocation(
                                "visit",
                                {"url": window.executable_targets[0]},
                            )
                            if len(window.executable_targets) == 1
                            else None
                        ),
                    )
                )
                for window, session_id in zip(batch, session_ids)
                if len(window.executable_targets) <= 1
            ]
            if calls:
                results = await asyncio.gather(*calls)
                for spec_row, result in zip(call_specs, results):
                    window, _, target_index, target = spec_row
                    authoritative_rows.append(
                        {
                            "target_id": (
                                f"{window.session_id}:{window.decision_id}:"
                                f"target:{target_index}"
                            ),
                            "target": target,
                            "source": result.source,
                            "exposed_wait_ms": result.exposed_wait_s * 1000.0,
                        }
                    )
            if early_cleanup:
                if strict_isolation:
                    deferred_cleanup.extend(early_cleanup)
                else:
                    await asyncio.gather(*early_cleanup)
            late_cleanup = [
                asyncio.create_task(
                    broker.cancel_predictions(session_id=session_id)
                )
                for window, session_id in zip(batch, session_ids)
                if len(window.executable_targets) > 1
            ]
            if late_cleanup:
                if strict_isolation:
                    deferred_cleanup.extend(late_cleanup)
                else:
                    await asyncio.gather(*late_cleanup)
            global_cleanup = asyncio.create_task(broker.cancel_predictions())
            if strict_isolation:
                deferred_cleanup.append(global_cleanup)
                # Let cancellation atomically detach the current batch, but do
                # not put non-preemptible drain on the next batch's path.
                await asyncio.sleep(0)
            else:
                await global_cleanup

        if deferred_cleanup:
            await asyncio.gather(*deferred_cleanup)
        pending_before_close = broker.pending_speculative_count
        # A speculative race winner returns as soon as its protected backup
        # loses; that backup may still be physically draining in the isolated
        # lane.  Keep that drain off the request path, but finish it before the
        # experiment snapshot so physical work and terminal job state are
        # measured consistently.
        await broker.close()
        snapshot = broker.snapshot()
        records = broker.tool_records()
        stats = broker.stats.to_dict()
    finally:
        await broker.close()
    wall_s = time.perf_counter() - wall_started

    speculative_records = [
        record
        for record in records
        if record.get("speculative") is True and record.get("admitted") is True
    ]
    useful_records = [
        record for record in speculative_records if record.get("committed") is True
    ]
    wrong_records = [
        record
        for record in speculative_records
        if record.get("exact_match") is not True
    ]
    hedged_exact_loser_records = [
        record
        for record in speculative_records
        if record.get("exact_match") is True
        and record.get("committed") is not True
    ]
    wrong_started_records = [
        record for record in wrong_records if record.get("started_at") is not None
    ]
    hedged_exact_loser_started_records = [
        record
        for record in hedged_exact_loser_records
        if record.get("started_at") is not None
    ]
    speculative_lane_started_records = [
        record
        for record in speculative_records
        if record.get("dispatch_lane") == "speculative"
        and record.get("started_at") is not None
    ]
    useful_speculative_lane_starts = sum(
        record.get("committed") is True
        for record in speculative_lane_started_records
    )
    physical_started = sum(
        record.get("started_at") is not None and record.get("admitted") is True
        for record in records
    )
    sources = Counter(row["source"] for row in authoritative_rows)
    overlap_hits = sources["reused"] + sources["promoted_inflight"]
    target_count = len(authoritative_rows)
    wrong_service_ms = sum(
        1000.0 * float(record["service_s"])
        for record in wrong_started_records
    )
    hedged_exact_loser_service_ms = sum(
        1000.0 * float(record["service_s"])
        for record in hedged_exact_loser_started_records
    )
    selected_after_deadline = sum(
        record.get("dispatch_lane") == "speculative"
        and isinstance(record.get("started_at"), (int, float))
        and float(record["started_at"])
        > deadline_by_session.get(str(record["session_id"]), math.inf)
        for record in speculative_records
    )
    reserve_cap = max(0, broker_visit_capacity - reserve)
    safety = {
        "requested_identity": (
            not speculate
            or requested
            == int(stats["speculative_admitted"])
            + int(stats["rejected_speculative_capacity"])
            + int(stats.get("rejected_speculative_deadline", 0))
            + int(stats["duplicate_predictions"])
        ),
        "admission_results_match_requests": (
            not speculate or len(admission_results) == requested
        ),
        "commits_equal_targets": int(stats["commits"]) == target_count,
        "authoritative_state_equal_targets": (
            len(broker.authoritative_state) == target_count
        ),
        "pending_zero": pending_before_close == 0,
        "snapshot_jobs_zero": len(snapshot["jobs"]) == 0,
        "global_cap": int(stats["max_running_total"]) <= broker_workers,
        "speculative_cap": (
            int(stats["max_running_speculative"]) <= speculative_workers
        ),
        "authoritative_worker_cap": (
            int(stats["max_running_authoritative"]) <= workers
            if strict_isolation
            else True
        ),
        "visit_cap": (
            int(stats["max_running_by_tool"].get("visit", 0))
            <= broker_visit_capacity
        ),
        "visit_reserve_cap": (
            int(stats["max_running_speculative_by_tool"].get("visit", 0))
            <= reserve_cap
        ),
        "authoritative_visit_cap": (
            int(
                stats["max_running_authoritative_by_tool"].get("visit", 0)
            )
            <= visit_capacity
            if strict_isolation
            else True
        ),
        "waste_service_reconciles": math.isclose(
            wrong_service_ms + hedged_exact_loser_service_ms,
            float(stats["wasted_speculative_service_s"]) * 1000.0,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ),
        "no_speculative_start_after_deadline": selected_after_deadline == 0,
        "baseline_authority_capacity_preserved": (
            not strict_policy
            or certified_isolated_slots == 0
            or (
                broker_workers - speculative_workers >= workers
                and broker_visit_capacity - reserve_cap >= visit_capacity
            )
        ),
    }
    if not all(safety.values()):
        raise RuntimeError(f"adaptive broker safety failed: {safety}")

    return {
        "policy": policy.name if policy else "demand_only",
        "seed": seed,
        "offered_concurrency": offered_concurrency,
        "baseline_workers": workers,
        "baseline_visit_capacity": visit_capacity,
        "broker_workers": broker_workers,
        "broker_visit_capacity": broker_visit_capacity,
        "certified_isolated_speculative_slots": certified_isolated_slots,
        "broker_speculative_pending": broker_speculative_pending,
        "authoritative_targets": target_count,
        "requested_predictions": requested,
        "deadline_skipped_predictions": deadline_skipped_predictions,
        "admitted_predictions": int(stats["speculative_admitted"]),
        "rejected_predictions": (
            int(stats["rejected_speculative_capacity"])
            + int(stats.get("rejected_speculative_deadline", 0))
        ),
        "rejected_capacity": int(stats["rejected_speculative_capacity"]),
        "rejected_start_deadline": int(
            stats.get("rejected_speculative_deadline", 0)
        ),
        "deadline_expired_before_start": int(
            stats.get("speculative_deadline_expired_before_start", 0)
        ),
        "duplicate_predictions": int(stats["duplicate_predictions"]),
        "replaced_queued_predictions": int(
            stats.get("speculative_replaced_by_priority", 0)
        ),
        "exact_hits": len(useful_records),
        "overlap_hits": overlap_hits,
        "source_counts": dict(sorted(sources.items())),
        "running_speculative_races": int(
            stats.get("running_speculative_races", 0)
        ),
        "speculative_race_wins": int(
            stats.get("speculative_race_wins", 0)
        ),
        "authoritative_race_wins": int(
            stats.get("authoritative_race_wins", 0)
        ),
        "wrong_started": len(wrong_started_records),
        "wrong_never_started": len(wrong_records) - len(wrong_started_records),
        "wrong_service_ms": wrong_service_ms,
        "hedged_exact_losers": len(hedged_exact_loser_records),
        "hedged_exact_loser_started": len(
            hedged_exact_loser_started_records
        ),
        "hedged_exact_loser_service_ms": hedged_exact_loser_service_ms,
        "saved_service_ms": float(stats["saved_service_s"]) * 1000.0,
        "speculative_lane_started": len(speculative_lane_started_records),
        "useful_speculative_lane_started": useful_speculative_lane_starts,
        "physical_started": physical_started,
        "physical_amplification": ratio(physical_started, target_count),
        "authoritative_rows": authoritative_rows,
        "total_exposed_wait_ms": sum(
            row["exposed_wait_ms"] for row in authoritative_rows
        ),
        "mean_exposed_wait_ms": (
            statistics.fmean(
                row["exposed_wait_ms"] for row in authoritative_rows
            )
            if authoritative_rows
            else 0.0
        ),
        "p95_exposed_wait_ms": percentile(
            [row["exposed_wait_ms"] for row in authoritative_rows], 0.95
        ),
        "wall_s": wall_s,
        "selection_compute_ms": sum(
            float(row["compute_ms"]) for row in selection_rows
        ),
        "selection_considered": sum(
            int(row["considered"]) for row in selection_rows
        ),
        "probability_candidates_evaluated": sum(
            int(row["probability_candidates_evaluated"])
            for row in selection_rows
        ),
        "selection_selected": sum(
            int(row["selected"]) for row in selection_rows
        ),
        "selection_selected_hits": sum(
            int(row["selected_hits"]) for row in selection_rows
        ),
        "selection_reason_counts": dict(
            sorted(
                (
                    Counter({"no_safe_capacity": len(windows)})
                    if strict_policy and certified_isolated_slots == 0
                    else sum(
                        (
                            Counter(row.get("selection_reason_counts", {}))
                            for row in selection_rows
                        ),
                        Counter(),
                    )
                ).items()
            )
        ),
        "selection_probability_sum": sum(
            float(row["selected_probability_sum"])
            for row in selection_rows
        ),
        "predictor_windows_evaluated": sum(
            int(row["predictor_windows_evaluated"])
            for row in selection_rows
        ),
        "coarse_load_kill_switch_batches": sum(
            bool(row["coarse_load_kill_switch"])
            for row in selection_rows
        ),
        "selected_position_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            {
                                int(position): int(count)
                                for position, count in row[
                                    "selected_positions"
                                ].items()
                            }
                        )
                        for row in selection_rows
                    ),
                    Counter(),
                ).items()
            )
        ),
        "mean_load_pressure": (
            statistics.fmean(
                float(row["load_pressure"]) for row in selection_rows
            )
            if selection_rows
            else 0.0
        ),
        "mean_shadow_price": (
            statistics.fmean(
                float(row["shadow_price"]) for row in selection_rows
            )
            if selection_rows
            else 0.0
        ),
        "max_queued_authoritative": int(stats["max_queued_authoritative"]),
        "max_queued_speculative": int(stats["max_queued_speculative"]),
        "max_running_total": int(stats["max_running_total"]),
        "max_running_speculative": int(stats["max_running_speculative"]),
        "max_running_speculative_by_tool": dict(
            stats["max_running_speculative_by_tool"]
        ),
        "admission_ms": {
            "total": sum(admission_ms),
            "mean": statistics.fmean(admission_ms) if admission_ms else 0.0,
            "p95": percentile(admission_ms, 0.95),
            "max": max(admission_ms, default=0.0),
        },
        "confirmation_offset_ms": {
            "mean": statistics.fmean(deadline_offsets_ms),
            "p95": percentile(deadline_offsets_ms, 0.95),
            "max": max(deadline_offsets_ms),
            "deadline_overrun_batches": deadline_overruns,
            "batches": len(deadline_offsets_ms),
        },
        "task_stream": {
            "source_sessions": len({window.session_id for window in windows}),
            "batches": len(batch_sizes),
            "max_realized_concurrency": max(batch_sizes, default=0),
            "mean_realized_concurrency": (
                statistics.fmean(batch_sizes) if batch_sizes else 0.0
            ),
            "source_session_unique_within_batch": True,
        },
        "safety": safety,
    }


def _sum(samples: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(float(sample[field]) for sample in samples)


def _average_precision(
    scores: Sequence[float], labels: Sequence[bool]
) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(
        zip(scores, labels, range(len(scores))),
        key=lambda row: (-float(row[0]), row[2]),
    )
    hits = 0
    precision_sum = 0.0
    for rank, (_, label, _) in enumerate(ordered, 1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def calibration_quality(
    windows: Sequence[ScoredWindow],
) -> dict[str, Any]:
    candidates = [
        candidate for window in windows for candidate in window.candidates
    ]
    labels = [candidate.exact_match for candidate in candidates]
    pattern_scores = [candidate.exact_probability for candidate in candidates]
    rank_scores = [candidate.rank_only_probability for candidate in candidates]

    def brier(scores: Sequence[float]) -> float:
        return statistics.fmean(
            (float(score) - float(label)) ** 2
            for score, label in zip(scores, labels)
        )

    threshold_rows = []
    for threshold in (0.20, 0.15, 0.10, 0.075, 0.05, 0.0):
        eligible = [
            candidate
            for candidate in candidates
            if candidate.exact_probability >= threshold
        ]
        hits = sum(candidate.exact_match for candidate in eligible)
        threshold_rows.append(
            {
                "threshold": threshold,
                "candidates": len(eligible),
                "hits": hits,
                "precision": ratio(hits, len(eligible)),
                "absolute_candidate_hit_recall": ratio(hits, sum(labels)),
            }
        )
    abstained = [
        candidate
        for candidate in candidates
        if candidate.pattern.hard_abstain
    ]
    return {
        "candidate_count": len(candidates),
        "positive_candidates": sum(labels),
        "positive_rate": ratio(sum(labels), len(labels)),
        "pattern_average_precision": _average_precision(pattern_scores, labels),
        "rank_only_average_precision": _average_precision(rank_scores, labels),
        "pattern_brier": brier(pattern_scores),
        "rank_only_brier": brier(rank_scores),
        "thresholds": threshold_rows,
        "hard_abstain": {
            "candidates": len(abstained),
            "hits": sum(candidate.exact_match for candidate in abstained),
        },
    }


def _target_rows(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for sample in samples
        for row in sample["authoritative_rows"]
    ]


def aggregate_cell(
    *,
    scenario: str,
    spec: PolicySpec,
    offered_concurrency: int,
    baseline_samples: Sequence[Mapping[str, Any]],
    policy_samples: Sequence[Mapping[str, Any]],
    feature_runtime_ms_per_window: float,
    probability_runtime_ms_per_candidate: float,
    workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
    isolated_speculative_slots: int = 0,
) -> dict[str, Any]:
    if len(baseline_samples) != len(policy_samples):
        raise ValueError("paired sample counts differ")
    repetitions = len(policy_samples)
    baseline_rows = _target_rows(baseline_samples)
    pattern_rows = _target_rows(policy_samples)
    if len(baseline_rows) != len(pattern_rows):
        raise RuntimeError("paired target counts differ")

    benefit_by_target: list[float] = []
    miss_regressions_ms: list[float] = []
    overlap_sources = {"reused", "promoted_inflight"}
    for baseline, pattern in zip(baseline_samples, policy_samples):
        baseline_by_id = {
            str(row["target_id"]): float(row["exposed_wait_ms"])
            for row in baseline["authoritative_rows"]
        }
        pattern_by_id = {
            str(row["target_id"]): row
            for row in pattern["authoritative_rows"]
        }
        if baseline_by_id.keys() != pattern_by_id.keys():
            raise RuntimeError("paired target identifiers differ")
        for target_id, baseline_wait in baseline_by_id.items():
            pattern_row = pattern_by_id[target_id]
            pattern_wait = float(pattern_row["exposed_wait_ms"])
            benefit_by_target.append(baseline_wait - pattern_wait)
            if str(pattern_row["source"]) not in overlap_sources:
                miss_regressions_ms.append(pattern_wait - baseline_wait)

    targets = len(pattern_rows)
    requested = int(_sum(policy_samples, "requested_predictions"))
    admitted = int(_sum(policy_samples, "admitted_predictions"))
    rejected = int(_sum(policy_samples, "rejected_predictions"))
    exact_hits = int(_sum(policy_samples, "exact_hits"))
    overlap_hits = sum(
        str(row["source"]) in overlap_sources for row in pattern_rows
    )
    wrong_started = int(_sum(policy_samples, "wrong_started"))
    wrong_never_started = int(
        _sum(policy_samples, "wrong_never_started")
    )
    wrong_service_ms = _sum(policy_samples, "wrong_service_ms")
    hedged_exact_losers = int(
        _sum(policy_samples, "hedged_exact_losers")
    )
    hedged_exact_loser_service_ms = _sum(
        policy_samples, "hedged_exact_loser_service_ms"
    )
    saved_service_ms = _sum(policy_samples, "saved_service_ms")
    speculative_lane_started = int(
        _sum(policy_samples, "speculative_lane_started")
    )
    useful_speculative_lane_started = int(
        _sum(policy_samples, "useful_speculative_lane_started")
    )
    physical_started = int(_sum(policy_samples, "physical_started"))
    baseline_total_ms = _sum(baseline_samples, "total_exposed_wait_ms")
    pattern_total_ms = _sum(policy_samples, "total_exposed_wait_ms")
    raw_net_ms = baseline_total_ms - pattern_total_ms
    selection_ms = _sum(policy_samples, "selection_compute_ms")
    predictor_windows_evaluated = int(
        _sum(policy_samples, "predictor_windows_evaluated")
    )
    probability_candidates_evaluated = int(
        _sum(policy_samples, "probability_candidates_evaluated")
    )
    probability_charge = (
        probability_runtime_ms_per_candidate
        * probability_candidates_evaluated
        if spec.name in {
            "confidence_global_reserved",
            "utility_global_authority_first",
            "utility_global_risk_limited",
            "safe_global_benefit",
        }
        else 0.0
    )
    precomputed_runtime_ms = (
        feature_runtime_ms_per_window * predictor_windows_evaluated
        + probability_charge
    )
    conservative_overhead_ms = precomputed_runtime_ms + selection_ms
    conservative_net_ms = raw_net_ms - conservative_overhead_ms
    baseline_wall_s = _sum(baseline_samples, "wall_s")
    pattern_wall_s = _sum(policy_samples, "wall_s")
    conservative_pattern_wall_s = (
        pattern_wall_s + precomputed_runtime_ms / 1000.0
    )
    source_counts: Counter[str] = Counter(
        str(row["source"]) for row in pattern_rows
    )
    selected = int(_sum(policy_samples, "selection_selected"))
    selected_hits = int(_sum(policy_samples, "selection_selected_hits"))
    deadline_batches = sum(
        int(sample["confirmation_offset_ms"]["batches"])
        for sample in policy_samples
    )
    deadline_overruns = sum(
        int(sample["confirmation_offset_ms"]["deadline_overrun_batches"])
        for sample in policy_samples
    )
    all_safety = all(
        all(bool(value) for value in sample["safety"].values())
        for sample in (*baseline_samples, *policy_samples)
    )
    repeat_raw_net_ms = [
        float(baseline["total_exposed_wait_ms"])
        - float(pattern["total_exposed_wait_ms"])
        for baseline, pattern in zip(baseline_samples, policy_samples)
    ]
    repeat_conservative_net_ms = []
    for raw_net, pattern in zip(repeat_raw_net_ms, policy_samples):
        evaluated_windows = int(pattern["predictor_windows_evaluated"])
        evaluated_candidates = int(
            pattern["probability_candidates_evaluated"]
        )
        runtime_overhead = (
            feature_runtime_ms_per_window * evaluated_windows
            + float(pattern["selection_compute_ms"])
        )
        if spec.name in {
            "confidence_global_reserved",
            "utility_global_authority_first",
            "utility_global_risk_limited",
            "safe_global_benefit",
        }:
            runtime_overhead += (
                probability_runtime_ms_per_candidate
                * evaluated_candidates
            )
        repeat_conservative_net_ms.append(raw_net - runtime_overhead)
    if scenario == "all_wrong_counterfactual" and (
        exact_hits != 0 or overlap_hits != 0 or saved_service_ms != 0.0
    ):
        raise RuntimeError("all-wrong counterfactual produced a hit")
    positive_repetitions = sum(
        value > 0.0 for value in repeat_conservative_net_ms
    )
    repeat_median_per_target = statistics.median(
        repeat_conservative_net_ms
    ) / (targets / repetitions)
    if selected == 0:
        net_interpretation = "no_op_timing_noise"
    elif conservative_net_ms <= 0.0:
        net_interpretation = "negative"
    elif (
        repeat_median_per_target > 0.0
        and positive_repetitions >= math.ceil(0.75 * repetitions)
    ):
        net_interpretation = "repeat_stable_positive"
    else:
        net_interpretation = "sign_unstable"

    def compact_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in sample.items()
            if key != "authoritative_rows"
        }

    return {
        "scenario": scenario,
        "policy": spec.name,
        "offered_concurrency": offered_concurrency,
        "repetitions": repetitions,
        "workers": workers,
        "visit_capacity": visit_capacity,
        "isolated_speculative_slots": (
            isolated_speculative_slots
            if spec.requires_isolated_capacity
            else 0
        ),
        "broker_workers": (
            workers + isolated_speculative_slots
            if spec.requires_isolated_capacity
            else workers
        ),
        "broker_visit_capacity": (
            visit_capacity + isolated_speculative_slots
            if spec.requires_isolated_capacity
            else visit_capacity
        ),
        "requires_isolated_capacity": spec.requires_isolated_capacity,
        "visit_authoritative_reserve": spec.visit_authoritative_reserve,
        "max_speculative_workers": spec.max_speculative_workers,
        "max_speculative_pending": max_speculative_pending,
        "broker_speculative_pending": max(
            int(sample["broker_speculative_pending"])
            for sample in policy_samples
        ),
        "synthetic_service_ms": service_ms,
        "synthetic_prediction_lead_ms": lead_ms,
        "requested_predictions": requested,
        "admitted_predictions": admitted,
        "rejected_predictions": rejected,
        "rejected_capacity": int(_sum(policy_samples, "rejected_capacity")),
        "replaced_queued_predictions": int(
            _sum(policy_samples, "replaced_queued_predictions")
        ),
        "rejected_start_deadline": int(
            _sum(policy_samples, "rejected_start_deadline")
        ),
        "deadline_expired_before_start": int(
            _sum(policy_samples, "deadline_expired_before_start")
        ),
        "deadline_skipped_predictions": int(
            _sum(policy_samples, "deadline_skipped_predictions")
        ),
        "admission_ratio": ratio(admitted, requested),
        "selection_considered": int(
            _sum(policy_samples, "selection_considered")
        ),
        "probability_candidates_evaluated": (
            probability_candidates_evaluated
        ),
        "predictor_windows_evaluated": predictor_windows_evaluated,
        "coarse_load_kill_switch_batches": int(
            _sum(policy_samples, "coarse_load_kill_switch_batches")
        ),
        "selection_selected": selected,
        "selection_selected_hits": selected_hits,
        "selection_precision": ratio(selected_hits, selected),
        "selected_probability_mean": ratio(
            _sum(policy_samples, "selection_probability_sum"), selected
        ),
        "authoritative_targets": targets,
        "exact_hits": exact_hits,
        "realized_exact_target_coverage": ratio(exact_hits, targets),
        "overlap_producing_hits": overlap_hits,
        "overlap_producing_target_coverage": ratio(overlap_hits, targets),
        "admitted_candidate_precision": ratio(exact_hits, admitted),
        "speculative_started_precision": ratio(
            useful_speculative_lane_started, speculative_lane_started
        ),
        "source_counts": dict(sorted(source_counts.items())),
        "running_speculative_races": int(
            _sum(policy_samples, "running_speculative_races")
        ),
        "speculative_race_wins": int(
            _sum(policy_samples, "speculative_race_wins")
        ),
        "authoritative_race_wins": int(
            _sum(policy_samples, "authoritative_race_wins")
        ),
        "wrong_speculations_started": wrong_started,
        "wrong_speculations_never_started": wrong_never_started,
        "wrong_speculative_service_ms": wrong_service_ms,
        "wasted_speculative_service_ms": (
            wrong_service_ms + hedged_exact_loser_service_ms
        ),
        "hedged_exact_losers": hedged_exact_losers,
        "hedged_exact_loser_service_ms": hedged_exact_loser_service_ms,
        "saved_speculative_service_ms": saved_service_ms,
        "wasted_service_ms_per_authoritative_target": ratio(
            wrong_service_ms + hedged_exact_loser_service_ms, targets
        ),
        "physical_calls_started": physical_started,
        "physical_call_amplification_vs_demand_only": ratio(
            physical_started, targets
        ),
        "baseline_mean_exposed_wait_ms": ratio(baseline_total_ms, targets),
        "pattern_mean_exposed_wait_ms": ratio(pattern_total_ms, targets),
        "baseline_p95_exposed_wait_ms": percentile(
            [float(row["exposed_wait_ms"]) for row in baseline_rows], 0.95
        ),
        "pattern_p95_exposed_wait_ms": percentile(
            [float(row["exposed_wait_ms"]) for row in pattern_rows], 0.95
        ),
        "raw_net_latency_benefit_ms_total": raw_net_ms,
        "raw_net_latency_benefit_ms_per_target": ratio(raw_net_ms, targets),
        "conservative_runtime_overhead_ms_total": conservative_overhead_ms,
        "conservative_net_latency_benefit_ms_total": conservative_net_ms,
        "conservative_net_latency_benefit_ms_per_target": ratio(
            conservative_net_ms, targets
        ),
        "conservative_net_latency_benefit_fraction": ratio(
            conservative_net_ms, baseline_total_ms
        ),
        "paired_target_benefit_ms": {
            "mean": statistics.fmean(benefit_by_target),
            "p05": percentile(benefit_by_target, 0.05),
            "p50": percentile(benefit_by_target, 0.50),
            "p95": percentile(benefit_by_target, 0.95),
        },
        "non_overlap_authority_regression_ms": {
            "targets": len(miss_regressions_ms),
            "mean": (
                statistics.fmean(miss_regressions_ms)
                if miss_regressions_ms
                else 0.0
            ),
            "p95": percentile(miss_regressions_ms, 0.95),
            "max": max(miss_regressions_ms, default=0.0),
            "fraction_over_0_1ms": ratio(
                sum(value > 0.1 for value in miss_regressions_ms),
                len(miss_regressions_ms),
            ),
        },
        "baseline_drained_wall_s": baseline_wall_s,
        "pattern_drained_wall_s": pattern_wall_s,
        "conservative_pattern_wall_s": conservative_pattern_wall_s,
        "conservative_drained_wall_benefit_fraction": ratio(
            baseline_wall_s - conservative_pattern_wall_s,
            baseline_wall_s,
        ),
        "baseline_authoritative_throughput_per_s": ratio(
            targets, baseline_wall_s
        ),
        "pattern_authoritative_throughput_per_s": ratio(
            targets, conservative_pattern_wall_s
        ),
        "selection_compute_ms_total": selection_ms,
        "precomputed_runtime_ms_total": precomputed_runtime_ms,
        "admission_ms_total": _sum(
            [sample["admission_ms"] for sample in policy_samples], "total"
        ),
        "admission_ms_p95_max_across_repeats": max(
            float(sample["admission_ms"]["p95"])
            for sample in policy_samples
        ),
        "deadline_overrun_batches": deadline_overruns,
        "deadline_batches": deadline_batches,
        "max_realized_concurrency": max(
            int(sample["task_stream"]["max_realized_concurrency"])
            for sample in policy_samples
        ),
        "mean_load_pressure": statistics.fmean(
            float(sample["mean_load_pressure"])
            for sample in policy_samples
        ),
        "mean_shadow_price": statistics.fmean(
            float(sample["mean_shadow_price"])
            for sample in policy_samples
        ),
        "max_queued_authoritative": max(
            int(sample["max_queued_authoritative"])
            for sample in policy_samples
        ),
        "max_running_speculative_by_tool": {
            "visit": max(
                int(
                    sample["max_running_speculative_by_tool"].get(
                        "visit", 0
                    )
                )
                for sample in policy_samples
            )
        },
        "all_safety_invariants_passed": all_safety,
        "net_interpretation": net_interpretation,
        "execution_order_counterbalanced_ab_ba": True,
        "repeat_raw_net_latency_benefit_ms": repeat_raw_net_ms,
        "repeat_conservative_net_latency_benefit_ms": (
            repeat_conservative_net_ms
        ),
        "repeat_conservative_summary_ms_per_target": {
            "min": min(repeat_conservative_net_ms) / (targets / repetitions),
            "median": statistics.median(repeat_conservative_net_ms)
            / (targets / repetitions),
            "max": max(repeat_conservative_net_ms) / (targets / repetitions),
            "positive_repetitions": positive_repetitions,
            "repetitions": repetitions,
        },
        "samples": {
            "baseline": [compact_sample(sample) for sample in baseline_samples],
            "policy": [compact_sample(sample) for sample in policy_samples],
        },
    }


async def run_matrix(
    windows: Sequence[ScoredWindow],
    *,
    specs: Sequence[PolicySpec],
    concurrencies: Sequence[int],
    repetitions: int,
    workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
    feature_runtime_ms_per_window: float,
    probability_runtime_ms_per_candidate: float,
    isolated_speculative_slots: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = (
        ("observed_nested_oof", list(windows)),
        ("all_wrong_counterfactual", force_all_wrong(windows)),
    )
    for scenario, scenario_windows in scenarios:
        for concurrency in concurrencies:
            for spec in specs:
                print(
                    f"running scenario={scenario} policy={spec.name} "
                    f"concurrency={concurrency}",
                    flush=True,
                )
                baseline_samples = []
                samples = []
                for repetition in range(repetitions):
                    async def run_one(
                        policy: PolicySpec | None,
                    ) -> dict[str, Any]:
                        return await _run_sample(
                            scenario_windows,
                            policy=policy,
                            offered_concurrency=concurrency,
                            seed=repetition,
                            workers=workers,
                            visit_capacity=visit_capacity,
                            max_speculative_pending=max_speculative_pending,
                            service_ms=service_ms,
                            lead_ms=lead_ms,
                            isolated_speculative_slots=(
                                isolated_speculative_slots
                            ),
                        )

                    if repetition % 2 == 0:
                        baseline = await run_one(None)
                        treatment = await run_one(spec)
                    else:
                        treatment = await run_one(spec)
                        baseline = await run_one(None)
                    baseline_samples.append(baseline)
                    samples.append(treatment)
                rows.append(
                    aggregate_cell(
                        scenario=scenario,
                        spec=spec,
                        offered_concurrency=concurrency,
                        baseline_samples=baseline_samples,
                        policy_samples=samples,
                        feature_runtime_ms_per_window=(
                            feature_runtime_ms_per_window
                        ),
                        probability_runtime_ms_per_candidate=(
                            probability_runtime_ms_per_candidate
                        ),
                        workers=workers,
                        visit_capacity=visit_capacity,
                        max_speculative_pending=max_speculative_pending,
                        service_ms=service_ms,
                        lead_ms=lead_ms,
                        isolated_speculative_slots=(
                            isolated_speculative_slots
                        ),
                    )
                )
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened = []
    fieldnames: list[str] = []
    for row in rows:
        simple = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list, tuple))
        }
        flattened.append(simple)
        for key in simple:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def render_report(payload: Mapping[str, Any]) -> str:
    rows = payload["load_matrix"]
    observed = [row for row in rows if row["scenario"] == "observed_nested_oof"]
    wrong = [row for row in rows if row["scenario"] == "all_wrong_counterfactual"]
    utility = [
        row
        for row in observed
        if row["policy"] == "utility_global_risk_limited"
    ]
    calibration = payload["calibration_quality"]
    static = payload["static_runtime_prefixes"]
    oracle = payload["bounded_pool_oracle"]
    lines = [
        "# Authority-first Pattern-v2 under low predictability and high load",
        "",
        "## Result",
        "",
        (
            "The quoted `Top-1 ≈27.8% / hit rate 93.8%` pair is not reproduced "
            "under the frozen whole-session grouped-OOF protocol. Exact-URL "
            f"Top-1 is {_pct(static[0]['exact_target_recall'])} and Top-5 is "
            f"{_pct(static[-1]['exact_target_recall'])}. The nearby "
            f"{_pct(oracle['target_coverage'])} figure is an evaluation-only "
            f"candidate-union oracle over {oracle['candidate_count_if_all_fired']} "
            "candidates, not a realizable hit rate for a bounded runtime policy."
        ),
        "",
        (
            "This experiment replaces per-task candidate allocation with a global "
            "non-neural empirical-count confidence table and expected-utility "
            "allocator. Authoritative work is dispatch-prioritized, one of the two "
            "visit slots is reserved for it, speculative admission is batched, and "
            "the utility policy abstains as forecast authoritative pressure rises."
        ),
        "",
        (
            "The `safe_global_benefit` row uses a stricter lexicographic "
            "policy: it first requires an isolated-capacity certificate and "
            "then ranks all visible sessions by expected saved latency. With "
            "K=0 it follows the demand-only fast path; with K>0 the original "
            "authority worker/tool caps are preserved and a running exact hit "
            "races a protected fresh authority backup."
        ),
        "",
        (
            f"Nested whole-session grouped OOF candidate AP is "
            f"{calibration['pattern_average_precision']:.4f}, versus "
            f"{calibration['rank_only_average_precision']:.4f} for rank alone; "
            f"Brier is {calibration['pattern_brier']:.5f} versus "
            f"{calibration['rank_only_brier']:.5f}. These are development OOF "
            "estimates, not a new confirmatory holdout."
        ),
        "",
        (
            "The online path is deterministic Pattern-v2 state/feature update plus "
            "empirical table lookup: measured Pattern-v2 feature "
            f"runtime is {payload['nested_oof']['runtime_pattern_feature_ms']['mean']:.3f} "
            "ms/decision on average and "
            f"{payload['nested_oof']['runtime_pattern_feature_ms']['p99']:.3f} ms "
            "at p99. A cheap fold-trained calls/window gate runs first; when "
            "forecast pressure exceeds 2x visit capacity, the utility policy "
            "skips candidate generation and probability lookup entirely."
        ),
        "",
    ]
    if utility:
        low = min(utility, key=lambda row: int(row["offered_concurrency"]))
        high = max(utility, key=lambda row: int(row["offered_concurrency"]))
        high_load_utility = [
            row for row in utility if int(row["offered_concurrency"]) >= 8
        ]
        stable_positive = [
            row
            for row in high_load_utility
            if row["net_interpretation"] == "repeat_stable_positive"
        ]
        pooled_positive = [
            row
            for row in high_load_utility
            if float(row["conservative_net_latency_benefit_ms_per_target"])
            > 0.0
            and int(row["selection_selected"]) > 0
        ]
        lines.extend(
            [
                (
                    "For the risk-limited utility allocator, conservative net is "
                    f"{_signed(low['conservative_net_latency_benefit_ms_per_target'])} "
                    f"ms/target at concurrency {low['offered_concurrency']} and "
                    f"{_signed(high['conservative_net_latency_benefit_ms_per_target'])} "
                    f"ms/target at concurrency {high['offered_concurrency']}; the "
                    "latter selected zero candidates and is timing noise around a "
                    "demand-only fallback, not a speedup."
                ),
                "",
                (
                    f"No C>=8 utility cell met the repeat-stability "
                    f"positive rule (found {len(stable_positive)}). "
                    + (
                        "A pooled positive estimate exists at "
                        + ", ".join(
                            f"C={row['offered_concurrency']}"
                            for row in pooled_positive
                        )
                        + ", but its paired repetitions change sign. "
                        if pooled_positive
                        else ""
                    )
                    + "The defensible high-load result is bounded harm and "
                    "graceful abstention, not demonstrated positive latency benefit."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Observed-label closed-loop burst replay",
            "",
            (
                f"Shared pool: {payload['configuration']['workers']} workers, "
                f"visit capacity {payload['configuration']['visit_capacity']}, "
                f"isolated speculative slots "
                f"{payload['configuration']['isolated_speculative_slots']}, "
                f"service {payload['configuration']['service_ms']:.1f} ms, lead "
                f"{payload['configuration']['lead_ms']:.1f} ms. Positive net means "
                "lower latency than demand-only after charging pattern feature, "
                "confidence lookup, and selection overhead."
            ),
            "",
            "| Policy | Offered / realized C | Exact / overlap hit | Wrong starts | Call amp. | Mean authority wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap authority regression mean / p95 | Drained wall benefit |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in observed:
        miss = row["non_overlap_authority_regression_ms"]
        lines.append(
            f"| {row['policy']} | {row['offered_concurrency']} / "
            f"{row['max_realized_concurrency']} | "
            f"{_pct(row['realized_exact_target_coverage'])} / "
            f"{_pct(row['overlap_producing_target_coverage'])} | "
            f"{row['wrong_speculations_started']} | "
            f"{row['physical_call_amplification_vs_demand_only']:.2f}x | "
            f"{row['baseline_mean_exposed_wait_ms']:.2f}→"
            f"{row['pattern_mean_exposed_wait_ms']:.2f} | "
            f"{_signed(row['conservative_net_latency_benefit_ms_per_target'])} / "
            f"{_signed(row['repeat_conservative_summary_ms_per_target']['median'])} "
            f"({row['repeat_conservative_summary_ms_per_target']['positive_repetitions']}/"
            f"{row['repetitions']}) | "
            f"{miss['mean']:+.3f} / {miss['p95']:+.3f} | "
            f"{row['conservative_drained_wall_benefit_fraction'] * 100:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "`Exact hit` includes queued promotion; `overlap hit` counts only "
            "completed reuse and inflight promotion. Non-overlap regression is "
            "measured only on authority targets that obtained no speculative "
            "overlap. Admission, deadline, source, p95, and per-repeat data are in "
            "`metrics.json`. Repetitions vary scheduling/order only and are not "
            "independent accuracy samples.",
            "",
            "## Deterministic all-wrong worst case",
            "",
            "Every authoritative URL is replaced by a guaranteed non-candidate URL, "
            "while gates, scores, ordering, and load stay fixed.",
            "",
            "| Policy | Offered / realized C | Selected / wrong-started | Call amp. | Mean wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap regression p95 / max | Drained wall benefit |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in wrong:
        miss = row["non_overlap_authority_regression_ms"]
        lines.append(
            f"| {row['policy']} | {row['offered_concurrency']} / "
            f"{row['max_realized_concurrency']} | "
            f"{row['selection_selected']} / {row['wrong_speculations_started']} | "
            f"{row['physical_call_amplification_vs_demand_only']:.2f}x | "
            f"{row['baseline_mean_exposed_wait_ms']:.2f}→"
            f"{row['pattern_mean_exposed_wait_ms']:.2f} | "
            f"{_signed(row['conservative_net_latency_benefit_ms_per_target'])} / "
            f"{_signed(row['repeat_conservative_summary_ms_per_target']['median'])} "
            f"({row['repeat_conservative_summary_ms_per_target']['positive_repetitions']}/"
            f"{row['repetitions']}) | "
            f"{miss['p95']:+.3f} / {miss['max']:+.3f} | "
            f"{row['conservative_drained_wall_benefit_fraction'] * 100:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "A cell is a timing-only no-op only when `Selected=0`. An all-wrong "
            "cell with positive speculative starts has no latency benefit, but "
            "still measures real scheduler/control-plane and cleanup overhead in "
            "addition to paired-run timing noise.",
            "",
            "## What authority-first can and cannot guarantee",
            "",
            "The broker now removes a whole cancellation set atomically before "
            "dispatch, so queued siblings cannot start one-by-one during cleanup. "
            "Batch admission also removes the prior O(N)-sweep-per-candidate control "
            "path, and the per-tool reserve prevents speculation from occupying both "
            "visit slots.",
            "",
            "With fixed shared capacity and non-preemptible tool calls, strict zero "
            "interference is impossible whenever any wrong speculation is running: a "
            "future burst can still need every slot. Absolute isolation requires "
            "extra/dedicated capacity, genuinely preemptible calls, or abstention. "
            "Therefore the intended saturated behavior is graceful fallback to "
            "demand-only, not forced positive speculation at every load.",
            "",
            "`safe_global_benefit` implements the extra-capacity case explicitly: "
            "authority is capped at its original baseline envelope, speculation "
            "is capped at K added slots, and K=0 admits nothing. This is a "
            "structural worker/tool-capacity guarantee; it excludes unisolated "
            "rate limits, predictor CPU, memory bandwidth, and event-loop noise.",
            "",
            "This table is a deterministic synthetic-service, closed-loop burst "
            "experiment. It diagnoses scheduling and resource allocation; it does not "
            "claim production network latency. A sustained/open-loop replay and a new "
            "whole-session confirmatory holdout are separate validation requirements.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCIES),
    )
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visit-capacity", type=int, default=2)
    parser.add_argument(
        "--isolated-speculative-slots",
        type=int,
        default=0,
        help=(
            "extra worker+visit slots certified unavailable to baseline "
            "authority; safe_global_benefit abstains when this is zero"
        ),
    )
    parser.add_argument("--max-speculative-pending", type=int, default=64)
    parser.add_argument("--service-ms", type=float, default=5.0)
    parser.add_argument("--lead-ms", type=float, default=2.5)
    parser.add_argument(
        "--policies", nargs="+", choices=POLICIES, default=list(POLICIES)
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if any(value <= 0 for value in args.concurrencies):
        parser.error("--concurrencies must be positive")
    if args.workers <= 0 or args.visit_capacity <= 0:
        parser.error("worker and visit capacities must be positive")
    if args.visit_capacity > args.workers:
        parser.error("--visit-capacity cannot exceed --workers")
    if args.isolated_speculative_slots < 0:
        parser.error("--isolated-speculative-slots must be non-negative")
    if args.max_speculative_pending <= 0:
        parser.error("--max-speculative-pending must be positive")
    if args.service_ms <= 0.0 or args.lead_ms < 0.0:
        parser.error("service must be positive and lead non-negative")
    return args


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    static_rows, static_oof = collect_pattern_v2_oof_rows(args.traces)
    static = static_width_metrics(static_rows, DEFAULT_WIDTHS)
    oracle = bounded_pool_oracle_metrics(static_rows)
    windows, oof = collect_nested_oof_windows(args.traces)
    quality = calibration_quality(windows)
    specs_by_name = {spec.name: spec for spec in policy_specs()}
    specs = [specs_by_name[name] for name in args.policies]
    if any(
        spec.visit_authoritative_reserve >= args.visit_capacity
        for spec in specs
    ):
        raise ValueError(
            "visit capacity must exceed every nonzero speculative policy reserve"
        )
    rows = await run_matrix(
        windows,
        specs=specs,
        concurrencies=args.concurrencies,
        repetitions=args.repetitions,
        workers=args.workers,
        visit_capacity=args.visit_capacity,
        max_speculative_pending=args.max_speculative_pending,
        service_ms=args.service_ms,
        lead_ms=args.lead_ms,
        feature_runtime_ms_per_window=float(
            oof["runtime_pattern_feature_ms"]["mean"]
        ),
        probability_runtime_ms_per_candidate=float(
            oof["runtime_probability_lookup_ms"]["mean"]
        ),
        isolated_speculative_slots=args.isolated_speculative_slots,
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_not_confirmatory",
        "command": shlex.join([sys.executable, *sys.argv]),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "concurrencies": list(args.concurrencies),
            "repetitions": args.repetitions,
            "workers": args.workers,
            "visit_capacity": args.visit_capacity,
            "isolated_speculative_slots": args.isolated_speculative_slots,
            "max_speculative_pending": args.max_speculative_pending,
            "service_ms": args.service_ms,
            "lead_ms": args.lead_ms,
            "policies": [asdict(spec) for spec in specs],
            "arrival_model": (
                "closed-loop source-session streams; one head decision per "
                "source session per batch"
            ),
            "paired_execution_order": "AB/BA counterbalanced by repetition",
            "vllm_required": False,
            "network_required": False,
            "neural_model": False,
        },
        "nested_oof": oof,
        "static_oof": static_oof,
        "static_runtime_prefixes": static,
        "bounded_pool_oracle": oracle,
        "calibration_quality": quality,
        "load_matrix": rows,
        "source_sha256": {
            "runner": sha256_file(SCRIPT),
            "broker": sha256_file(
                REPRODUCTION_ROOT / "paste_repro" / "live_broker.py"
            ),
            "policy": sha256_file(
                REPRODUCTION_ROOT / "paste_repro" / "speculation_policy.py"
            ),
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(async_main(args))
    write_json(args.output_dir / "metrics.json", payload)
    write_csv(args.output_dir / "load_matrix.csv", payload["load_matrix"])
    (args.output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(f"wrote {args.output_dir.resolve()}")
    print(f"payload_sha256={payload['payload_sha256']}")


if __name__ == "__main__":
    main()
