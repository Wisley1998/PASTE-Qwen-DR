#!/usr/bin/env python3
"""Sustained open-loop stress test for adaptive Pattern-v2 speculation.

Unlike the drained-burst adaptive runner, this experiment fixes every
decision release and authoritative confirmation on an exogenous timeline.
Speculation therefore cannot postpone the next batch and then claim that the
system was lightly loaded.  Baseline and treatment consume the same window
order, authoritative targets, service time, and absolute arrival plan.

The runner is CPU-only.  It uses the real :class:`LiveToolBroker`, a
non-preemptible synthetic executor, and the nested whole-session OOF
probabilities produced by ``run_pattern_v2_adaptive_load.py``.  It starts no
vLLM server and performs no network requests.  Each scored causal prefix is
replayed as an independent cloned task; this does not claim to preserve
within-source-session concurrency.  Latency results are scheduler-marginal:
the already measured Pattern feature and probability-table lookup work is
precomputed rather than charged to the online timeline.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    SafeGlobalBenefitPolicy,
    SafeStartBudget,
    UtilityCandidate,
    UtilityPolicyConfig,
)
from run_pattern_cache_evaluation import sha256_file  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredCandidate,
    ScoredWindow,
    collect_nested_oof_windows,
    force_all_wrong,
)
from run_pattern_v2_load_robustness import (  # noqa: E402
    canonical_sha256,
    percentile,
    ratio,
    stable_order,
)


SCHEMA = "paste_repro.pattern_v2_open_loop_stress.v3"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "results" / "pattern_v2_open_loop_stress"
DEFAULT_LOADS = (0.50, 0.90, 1.20)
POLICY_NAMES = (
    "rank5_unreserved",
    "rank5_reserved",
    "rank_budgeted_reserved",
    "confidence_reserved",
    "utility_authority_first",
    "utility_risk_limited",
    "safe_global_benefit",
)
OVERLAP_SOURCES = frozenset({"reused", "promoted_inflight"})


@dataclass(frozen=True)
class OpenLoopPolicy:
    name: str
    selection: str
    max_speculative_workers: int | None
    visit_authoritative_reserve: int
    confidence_threshold: float = 0.10
    utility_config: UtilityPolicyConfig | None = None
    utility_risk_floor: float = 0.0
    coarse_load_pre_gate: bool = False
    requires_isolated_capacity: bool = False


@dataclass(frozen=True)
class ScheduledWindow:
    instance_id: str
    window: ScoredWindow
    release_offset_s: float
    confirmation_offset_s: float


def top_fraction_share(
    values: Sequence[float | int], fraction: float = 0.10
) -> float:
    """Share of non-negative mass held by the largest fraction of entries."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    nonnegative = [max(0.0, float(value)) for value in values]
    total = sum(nonnegative)
    if not nonnegative or total <= 0.0:
        return 0.0
    count = max(1, math.ceil(len(nonnegative) * fraction))
    return sum(sorted(nonnegative, reverse=True)[:count]) / total


def jain_index(values: Sequence[float | int]) -> float:
    """Jain allocation-breadth index, with all-zero allocation reported as 0."""

    nonnegative = [max(0.0, float(value)) for value in values]
    total = sum(nonnegative)
    squared = sum(value * value for value in nonnegative)
    if not nonnegative or squared <= 0.0:
        return 0.0
    return total * total / (len(nonnegative) * squared)


def policy_specs() -> dict[str, OpenLoopPolicy]:
    return {
        "rank5_unreserved": OpenLoopPolicy(
            "rank5_unreserved", "rank", 2, 0
        ),
        "rank5_reserved": OpenLoopPolicy(
            "rank5_reserved", "rank", 1, 1
        ),
        "rank_budgeted_reserved": OpenLoopPolicy(
            "rank_budgeted_reserved", "rank_budgeted", 1, 1
        ),
        "confidence_reserved": OpenLoopPolicy(
            "confidence_reserved", "confidence", 1, 1
        ),
        "utility_authority_first": OpenLoopPolicy(
            "utility_authority_first",
            "utility",
            1,
            1,
            utility_config=UtilityPolicyConfig(),
        ),
        "utility_risk_limited": OpenLoopPolicy(
            "utility_risk_limited",
            "utility",
            1,
            1,
            utility_config=UtilityPolicyConfig(),
            utility_risk_floor=0.20,
            coarse_load_pre_gate=True,
        ),
        "safe_global_benefit": OpenLoopPolicy(
            "safe_global_benefit",
            "safe_benefit",
            None,
            0,
            confidence_threshold=0.0,
            requires_isolated_capacity=True,
        ),
    }


def parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("loads must be finite and positive")
    return result


def parse_csv_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(result) - set(POLICY_NAMES))
    if not result or unknown:
        raise argparse.ArgumentTypeError(
            "policies must be a non-empty subset of "
            + ",".join(POLICY_NAMES)
        )
    return tuple(dict.fromkeys(result))


def build_schedule(
    windows: Sequence[ScoredWindow],
    *,
    offered_load: float,
    visit_capacity: int,
    service_s: float,
    lead_s: float,
    seed: int,
    cycles: int,
) -> tuple[list[ScheduledWindow], dict[str, Any]]:
    """Create an exogenous timeline of independent cloned prefix tasks."""

    if not windows or cycles <= 0:
        raise ValueError("windows and cycles must be non-empty and positive")
    if offered_load <= 0.0 or visit_capacity <= 0 or service_s <= 0.0:
        raise ValueError("load, capacity, and service must be positive")
    ordered: list[tuple[int, ScoredWindow]] = []
    for cycle in range(cycles):
        cycle_windows = sorted(
            windows,
            key=lambda row: (
                stable_order(seed + 104729 * cycle, row.decision_id),
                row.decision_id,
            ),
        )
        ordered.extend((cycle, row) for row in cycle_windows)

    targets = sum(len(row.executable_targets) for _, row in ordered)
    if targets <= 0:
        raise ValueError("open-loop workload has no authoritative targets")
    # With N decisions separated by delta, the planned demand over N*delta is
    # targets*service/(N*delta*capacity), exactly the requested utilization.
    interarrival_s = (
        targets * service_s
        / (len(ordered) * offered_load * visit_capacity)
    )
    scheduled = [
        ScheduledWindow(
            instance_id=f"c{cycle}:{index}:{window.decision_id}",
            window=window,
            release_offset_s=index * interarrival_s,
            confirmation_offset_s=index * interarrival_s + lead_s,
        )
        for index, (cycle, window) in enumerate(ordered)
    ]
    return scheduled, {
        "decisions": len(scheduled),
        "authoritative_targets": targets,
        "interarrival_ms": interarrival_s * 1000.0,
        "lead_ms": lead_s * 1000.0,
        "planned_arrival_span_s": len(scheduled) * interarrival_s,
        "requested_authority_utilization": offered_load,
        "derived_authority_utilization": ratio(
            targets * service_s,
            len(scheduled) * interarrival_s * visit_capacity,
        ),
        "task_model": "independent_prefix_clones_not_source_session_concurrency",
    }


async def _sleep_until(deadline: float) -> None:
    await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))


def _candidate_identity(candidate: ScoredCandidate) -> tuple[str, str, str]:
    return (
        candidate.pattern.session_id,
        candidate.pattern.decision_id,
        candidate.pattern.url,
    )


def _select_candidates(
    window: ScoredWindow,
    policy: OpenLoopPolicy,
    *,
    snapshot: Mapping[str, Any],
    offered_load: float,
    visit_capacity: int,
    service_s: float,
    lead_remaining_s: float,
    isolated_speculative_slots: int = 0,
) -> tuple[list[tuple[ScoredCandidate, float]], dict[str, Any]]:
    """Select using causal features plus the broker state visible now."""

    started_ns = time.perf_counter_ns()
    candidates = tuple(window.candidates)
    reasons: Counter[str] = Counter()
    selected: list[tuple[ScoredCandidate, float]] = []
    load_pressure = offered_load
    shadow_price = 0.0
    counts = snapshot["counts"]
    coarse_forecast_pressure = max(
        offered_load,
        ratio(
            float(window.coarse_expected_authoritative_calls),
            visit_capacity,
        ),
    )
    coarse_load_pressure = max(
        coarse_forecast_pressure,
        ratio(
            int(counts["running_authoritative"])
            + int(counts["queued_authoritative"]),
            visit_capacity,
        ),
    )
    coarse_load_kill_switch = False
    predictor_windows_evaluated = 1
    safe_start_budget = (
        isolated_speculative_slots
        if policy.requires_isolated_capacity
        else 0
    )

    if policy.requires_isolated_capacity and safe_start_budget == 0:
        # The resource certificate is deliberately checked before charging the
        # predictor. Shared idle capacity is not a certificate.
        predictor_windows_evaluated = 0
        reasons["no_safe_capacity"] += max(1, len(candidates))
    elif not candidates:
        reasons["no_candidates"] += 1
    elif lead_remaining_s <= 0.0:
        reasons["deadline_elapsed"] += len(candidates)
    elif policy.selection == "rank":
        if not window.v2_gate:
            reasons["v2_gate_abstain"] += len(candidates)
        else:
            selected = [
                (candidate, 1.0 / candidate.pattern.position)
                for candidate in candidates
            ]
    elif policy.selection == "rank_budgeted":
        if not window.v2_gate:
            reasons["v2_gate_abstain"] += len(candidates)
        else:
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    candidate.pattern.position,
                    candidate.pattern.url,
                ),
            )
            selected = [
                (candidate, 1.0 / candidate.pattern.position)
                for candidate in ranked[:1]
            ]
            reasons["local_start_budget"] += max(0, len(ranked) - 1)
    elif policy.selection == "confidence":
        eligible = [
            candidate
            for candidate in candidates
            if candidate.exact_probability >= policy.confidence_threshold
        ]
        reasons["below_confidence"] += len(candidates) - len(eligible)
        eligible.sort(
            key=lambda row: (
                -row.exact_probability,
                row.pattern.position,
                row.pattern.url,
            )
        )
        # One reserved speculative visit slot can produce at most one start
        # during this short decision lead when lead<=service.
        selected = [
            (candidate, candidate.exact_probability)
            for candidate in eligible[:1]
        ]
        reasons["local_start_budget"] += max(0, len(eligible) - 1)
    elif policy.selection == "utility":
        controller = AuthorityFirstUtilityPolicy(policy.utility_config)
        if (
            policy.coarse_load_pre_gate
            and coarse_load_pressure > controller.config.high_pressure
        ):
            coarse_load_kill_switch = True
            predictor_windows_evaluated = 0
            reasons["coarse_load_kill_switch"] += len(candidates)
            load_pressure = coarse_load_pressure
            shadow_price = controller.shadow_price(
                AuthorityLoad(
                    expected_authoritative_calls=(
                        coarse_load_pressure * visit_capacity
                    ),
                    tool_capacity=visit_capacity,
                )
            )
        else:
            eligible = tuple(
                candidate
                for candidate in candidates
                if candidate.exact_probability >= policy.utility_risk_floor
            )
            reasons["below_risk_floor"] += len(candidates) - len(eligible)
            load = AuthorityLoad(
                expected_authoritative_calls=offered_load * visit_capacity,
                tool_capacity=visit_capacity,
                authoritative_running=int(counts["running_authoritative"]),
                authoritative_queued=int(counts["queued_authoritative"]),
            )
            utility_rows = tuple(
                UtilityCandidate(
                    pattern=candidate.pattern,
                    exact_probability=candidate.exact_probability,
                    estimated_service_s=service_s,
                    lead_remaining_s=lead_remaining_s,
                )
                for candidate in eligible
            )
            decision = controller.select(utility_rows, load=load, start_budget=1)
            by_identity = {_candidate_identity(row): row for row in eligible}
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
            reasons.update(
                row.reason for row in decision.decisions if not row.selected
            )
            load_pressure = decision.load_pressure
            shadow_price = decision.shadow_price
    elif policy.selection == "safe_benefit":
        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.exact_probability >= policy.confidence_threshold
        )
        reasons["below_confidence"] += len(candidates) - len(eligible)
        decision = SafeGlobalBenefitPolicy().select(
            tuple(
                UtilityCandidate(
                    pattern=candidate.pattern,
                    exact_probability=candidate.exact_probability,
                    estimated_service_s=service_s,
                    lead_remaining_s=lead_remaining_s,
                )
                for candidate in eligible
            ),
            safe_budget=SafeStartBudget(safe_start_budget),
            requested_start_budget=1,
        )
        by_identity = {_candidate_identity(row): row for row in eligible}
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
        reasons.update(
            row.reason for row in decision.decisions if not row.selected
        )
    else:  # pragma: no cover - constructor-controlled
        raise ValueError(f"unknown selection: {policy.selection}")

    return selected, {
        "considered": len(candidates),
        "selected": len(selected),
        "selected_labeled_hits": sum(row.exact_match for row, _ in selected),
        "reasons": dict(sorted(reasons.items())),
        "load_pressure": load_pressure,
        "shadow_price": shadow_price,
        "coarse_load_pressure": coarse_load_pressure,
        "coarse_forecast_pressure": coarse_forecast_pressure,
        "coarse_load_kill_switch": coarse_load_kill_switch,
        "predictor_windows_evaluated": predictor_windows_evaluated,
        "safe_start_budget": safe_start_budget,
        "compute_ms": (time.perf_counter_ns() - started_ns) / 1_000_000.0,
        "authoritative_running_at_selection": int(
            snapshot["counts"]["running_authoritative"]
        ),
        "authoritative_queued_at_selection": int(
            snapshot["counts"]["queued_authoritative"]
        ),
    }


async def _wait_for_authoritative_claims(
    broker: LiveToolBroker,
    *,
    session_id: str,
    expected: int,
) -> None:
    """Wait until every exact key is claimed before cancelling its siblings."""

    for _ in range(1000):
        claimed = sum(
            record.get("session_id") == session_id
            and record.get("authoritative_confirmation_at") is not None
            for record in broker.tool_records()
        )
        if claimed >= expected:
            return
        await asyncio.sleep(0)
    raise RuntimeError(
        f"authoritative claims did not become visible for {session_id}: "
        f"expected {expected}"
    )


async def run_open_loop_sample(
    windows: Sequence[ScoredWindow],
    *,
    policy: OpenLoopPolicy | None,
    offered_load: float,
    seed: int,
    cycles: int,
    workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
    isolated_speculative_slots: int = 0,
) -> dict[str, Any]:
    """Run one baseline or treatment against fixed absolute arrivals."""

    service_s = service_ms / 1000.0
    lead_s = lead_ms / 1000.0
    schedule, plan = build_schedule(
        windows,
        offered_load=offered_load,
        visit_capacity=visit_capacity,
        service_s=service_s,
        lead_s=lead_s,
        seed=seed,
        cycles=cycles,
    )

    physical_tasks: list[asyncio.Task[None]] = []

    async def executor(invocation: Invocation) -> dict[str, Any]:
        # Blocking HTTP work cannot normally be stopped by cancelling its
        # asyncio wrapper. Retain the physical slot until service drains.
        physical = asyncio.create_task(asyncio.sleep(service_s))
        physical_tasks.append(physical)
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
    broker_speculative_pending = (
        min(max_speculative_pending, max(1, 2 * certified_isolated_slots))
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
        ttl_s=max(1.0, 10.0 * lead_s),
        service_time_hints_s={"visit": service_s},
        tool_capacities={"visit": broker_visit_capacity},
        authoritative_tool_capacities=(
            {"visit": visit_capacity} if strict_isolation else None
        ),
        authoritative_tool_reserves=(
            {"visit": reserve} if reserve else None
        ),
    )
    loop = asyncio.get_running_loop()
    timeline_start = loop.time() + 0.010
    selection_rows: list[dict[str, Any]] = []
    admission_ms: list[float] = []
    admission_results: list[bool] = []
    requested = 0
    release_lateness_ms: list[float] = []
    confirmation_lateness_ms: list[float] = []
    authoritative_rows: list[dict[str, Any]] = []
    planned_confirmation_by_session: dict[str, float] = {}
    # A causal, O(1) controller view of real authoritative work. Calling the
    # broker's rich public snapshot for every sub-millisecond arrival would
    # itself scan the entire queue and can starve the event loop under stress.
    # This counter is updated at authoritative confirmation/completion and is
    # deliberately independent of prediction labels.
    authoritative_outstanding = 0
    max_authoritative_outstanding = 0

    async def run_window(row: ScheduledWindow) -> None:
        nonlocal requested, authoritative_outstanding, max_authoritative_outstanding
        release_at = timeline_start + row.release_offset_s
        confirmation_at = timeline_start + row.confirmation_offset_s
        session_id = f"ol:r{seed}:{row.instance_id}"
        planned_confirmation_by_session[session_id] = confirmation_at
        await _sleep_until(release_at)
        released = loop.time()
        release_lateness_ms.append(max(0.0, released - release_at) * 1000.0)

        if speculate and policy is not None:
            observed_running = min(authoritative_outstanding, visit_capacity)
            observed_queued = max(
                0, authoritative_outstanding - visit_capacity
            )
            controller_state = {
                "counts": {
                    "running_authoritative": observed_running,
                    "queued_authoritative": observed_queued,
                }
            }
            selected, selection = _select_candidates(
                row.window,
                policy,
                snapshot=controller_state,
                offered_load=offered_load,
                visit_capacity=visit_capacity,
                service_s=service_s,
                lead_remaining_s=max(0.0, confirmation_at - loop.time()),
                isolated_speculative_slots=certified_isolated_slots,
            )
            selection_rows.append(selection)
            requests = tuple(
                (
                    Invocation("visit", {"url": candidate.pattern.url}),
                    session_id,
                    priority,
                )
                for candidate, priority in selected
            )
            if requests and loop.time() < confirmation_at:
                requested += len(requests)
                admitted_started = time.perf_counter()
                admission_results.extend(
                    await broker.speculate_batch(
                        requests,
                        start_deadline=confirmation_at,
                        replace_lower_priority_queued=strict_isolation,
                    )
                )
                admission_ms.append(
                    (time.perf_counter() - admitted_started) * 1000.0
                )

        await _sleep_until(confirmation_at)
        confirmed = loop.time()
        confirmation_lateness_ms.append(
            max(0.0, confirmed - confirmation_at) * 1000.0
        )
        targets = row.window.executable_targets
        if not targets:
            await broker.cancel_predictions(session_id=session_id)
            return

        authoritative_outstanding += len(targets)
        max_authoritative_outstanding = max(
            max_authoritative_outstanding, authoritative_outstanding
        )

        async def call_authority(target: str) -> tuple[Any, float]:
            nonlocal authoritative_outstanding
            try:
                result = await broker.authoritative(
                    Invocation("visit", {"url": target}),
                    session_id=session_id,
                    reuse_running_speculation=not strict_isolation,
                )
                return result, loop.time()
            finally:
                authoritative_outstanding -= 1

        calls = [asyncio.create_task(call_authority(target)) for target in targets]
        await _wait_for_authoritative_claims(
            broker, session_id=session_id, expected=len(targets)
        )
        cleanup = asyncio.create_task(
            broker.cancel_predictions(session_id=session_id)
        )
        results = await asyncio.gather(*calls)
        await cleanup
        for target_index, (target, result_row) in enumerate(zip(targets, results)):
            result, completed_at = result_row
            authoritative_rows.append(
                {
                    "target_id": f"{row.instance_id}:target:{target_index}",
                    "decision_id": row.window.decision_id,
                    "target": target,
                    "source": result.source,
                    "overlap_producing": result.source in OVERLAP_SOURCES,
                    "exposed_wait_ms": result.exposed_wait_s * 1000.0,
                    "queue_ms": result.queue_s * 1000.0,
                    "service_ms": result.service_s * 1000.0,
                    "scheduled_response_ms": max(
                        0.0, completed_at - confirmation_at
                    )
                    * 1000.0,
                    "confirmation_lateness_ms": max(
                        0.0, confirmed - confirmation_at
                    )
                    * 1000.0,
                }
            )

    await _sleep_until(timeline_start)
    tasks = [asyncio.create_task(run_window(row)) for row in schedule]
    await asyncio.gather(*tasks)
    await broker.cancel_predictions()
    # Use the executor's physical tasks as the drain barrier. A concurrent
    # cleanup can detach an already-cancelling broker job before ``close`` sees
    # it; issuing a second runner.cancel() at that point can interrupt the
    # broker while it is publishing terminal telemetry.
    if physical_tasks:
        await asyncio.gather(*physical_tasks, return_exceptions=True)
    telemetry_deadline = loop.time() + max(1.0, 10.0 * service_s)
    while True:
        provisional_records = broker.tool_records()
        incomplete_started = [
            record
            for record in provisional_records
            if record.get("started_at") is not None
            and not isinstance(record.get("service_s"), (int, float))
        ]
        if not incomplete_started:
            break
        if loop.time() >= telemetry_deadline:
            raise RuntimeError(
                "physical runners did not publish terminal service telemetry: "
                + repr([record.get("job_id") for record in incomplete_started[:10]])
            )
        await asyncio.sleep(min(0.001, service_s / 10.0))
    pending_before_close = broker.pending_speculative_count
    await broker.close()
    drained_at = loop.time()
    snapshot = broker.snapshot()
    records = broker.tool_records()
    stats = broker.stats.to_dict()

    speculative_records = [
        record
        for record in records
        if record.get("speculative") is True and record.get("admitted") is True
    ]
    useful_records = [record for record in speculative_records if record.get("committed")]
    wrong_records = [
        record
        for record in speculative_records
        if record.get("exact_match") is not True
    ]
    hedged_exact_loser_records = [
        record
        for record in speculative_records
        if record.get("exact_match") is True
        and not record.get("committed")
    ]
    wrong_started_records = [
        record for record in wrong_records if record.get("started_at") is not None
    ]
    hedged_exact_loser_started_records = [
        record
        for record in hedged_exact_loser_records
        if record.get("started_at") is not None
    ]
    incomplete_wrong_service = [
        record
        for record in wrong_started_records
        if not isinstance(record.get("service_s"), (int, float))
    ]
    if incomplete_wrong_service:
        raise RuntimeError(
            "started wrong speculation lacks service telemetry: "
            + repr(
                [
                    {
                        key: record.get(key)
                        for key in (
                            "job_id",
                            "session_id",
                            "outcome",
                            "source",
                            "started_at",
                            "finished_at",
                            "service_s",
                            "committed",
                        )
                    }
                    for record in incomplete_wrong_service[:5]
                ]
            )
        )
    wrong_service_ms = sum(
        float(record["service_s"]) * 1000.0 for record in wrong_started_records
    )
    hedged_exact_loser_service_ms = sum(
        float(record["service_s"]) * 1000.0
        for record in hedged_exact_loser_started_records
    )
    saved_service_ms = float(stats["saved_service_s"]) * 1000.0
    physical_started = sum(
        record.get("admitted") is True and record.get("started_at") is not None
        for record in records
    )
    started_speculative_records = [
        record
        for record in speculative_records
        if record.get("dispatch_lane") == "speculative"
        and isinstance(record.get("started_at"), (int, float))
    ]
    late_start_deltas_ms = [
        (
            float(record["started_at"])
            - planned_confirmation_by_session[str(record["session_id"])]
        )
        * 1000.0
        for record in started_speculative_records
        if float(record["started_at"])
        > planned_confirmation_by_session[str(record["session_id"])]
    ]
    start_deadlines_match_plan = all(
        isinstance(record.get("start_deadline"), (int, float))
        and math.isclose(
            float(record["start_deadline"]),
            planned_confirmation_by_session[str(record["session_id"])],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for record in speculative_records
    )
    late_speculative_starts = len(late_start_deltas_ms)
    speculative_overtakes = sum(
        record.get("dispatch_lane") == "speculative"
        and int(record.get("queued_authoritative_same_tool_before") or 0) > 0
        for record in speculative_records
    )
    unsafe_speculative_overtakes = sum(
        record.get("dispatch_lane") == "speculative"
        and int(record.get("queued_authoritative_same_tool_before") or 0) > 0
        and not (
            strict_isolation
            and (
                int(record.get("running_authoritative_before") or 0)
                >= workers
                or int(
                    record.get(
                        "running_authoritative_same_tool_before"
                    )
                    or 0
                )
                >= visit_capacity
            )
        )
        for record in speculative_records
    )
    speculative_starts_by_session = Counter(
        str(record["session_id"])
        for record in speculative_records
        if record.get("dispatch_lane") == "speculative"
        and record.get("started_at") is not None
    )
    start_allocations = [
        speculative_starts_by_session.get(session_id, 0)
        for session_id in planned_confirmation_by_session
    ]
    reasons = sum(
        (Counter(row["reasons"]) for row in selection_rows), Counter()
    )
    if strict_policy and certified_isolated_slots == 0:
        reasons["no_safe_capacity"] += len(schedule)
    target_count = len(authoritative_rows)
    reserve_cap = max(0, broker_visit_capacity - reserve)
    safety = {
        "commits_equal_targets": int(stats["commits"]) == target_count,
        "authoritative_state_equal_targets": (
            len(broker.authoritative_state) == target_count
        ),
        "requested_identity": (
            policy is None
            or requested
            == int(stats["speculative_admitted"])
            + int(stats["rejected_speculative_capacity"])
            + int(stats["rejected_speculative_deadline"])
            + int(stats["duplicate_predictions"])
        ),
        "admission_results_match_requests": (
            policy is None or len(admission_results) == requested
        ),
        "pending_zero": pending_before_close == 0,
        "snapshot_jobs_zero": len(snapshot["jobs"]) == 0,
        "global_capacity": (
            int(stats["max_running_total"]) <= broker_workers
        ),
        "visit_capacity": (
            int(stats["max_running_by_tool"].get("visit", 0))
            <= broker_visit_capacity
        ),
        "speculative_capacity": (
            int(stats["max_running_speculative"]) <= speculative_workers
        ),
        "visit_reserve_capacity": (
            int(stats["max_running_speculative_by_tool"].get("visit", 0))
            <= reserve_cap
        ),
        "authoritative_worker_cap": (
            int(stats["max_running_authoritative"]) <= workers
            if strict_isolation
            else True
        ),
        "authoritative_visit_cap": (
            int(
                stats["max_running_authoritative_by_tool"].get("visit", 0)
            )
            <= visit_capacity
            if strict_isolation
            else True
        ),
        "baseline_authority_capacity_preserved": (
            not strict_policy
            or certified_isolated_slots == 0
            or (
                broker_workers - speculative_workers >= workers
                and broker_visit_capacity - reserve_cap >= visit_capacity
            )
        ),
        # Starting isolated work beside a saturated baseline authority slice
        # is concurrency, not an authority overtake.
        "no_queued_authority_overtake": unsafe_speculative_overtakes == 0,
        "speculative_start_deadlines_match_plan": start_deadlines_match_plan,
        "no_speculative_start_after_deadline": late_speculative_starts == 0,
        "wasted_service_reconciles": math.isclose(
            wrong_service_ms + hedged_exact_loser_service_ms,
            float(stats["wasted_speculative_service_s"]) * 1000.0,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ),
    }
    if not all(safety.values()):
        raise RuntimeError(f"open-loop safety invariant failed: {safety}")

    last_confirmation = timeline_start + schedule[-1].confirmation_offset_s
    wall_s = max(0.0, drained_at - timeline_start)
    return {
        "policy": policy.name if policy else "demand_only",
        "seed": seed,
        "offered_load": offered_load,
        "baseline_workers": workers,
        "baseline_visit_capacity": visit_capacity,
        "broker_workers": broker_workers,
        "broker_visit_capacity": broker_visit_capacity,
        "certified_isolated_speculative_slots": certified_isolated_slots,
        "broker_speculative_pending": broker_speculative_pending,
        "plan": plan,
        "authoritative_targets": target_count,
        "requested_predictions": requested,
        "admitted_predictions": int(stats["speculative_admitted"]),
        "rejected_predictions": int(stats["rejected_speculative_capacity"]),
        "deadline_rejected_predictions": int(
            stats["rejected_speculative_deadline"]
        ),
        "replaced_queued_predictions": int(
            stats.get("speculative_replaced_by_priority", 0)
        ),
        "selection_considered": sum(int(row["considered"]) for row in selection_rows),
        "selection_selected": sum(int(row["selected"]) for row in selection_rows),
        "selection_labeled_hits": sum(
            int(row["selected_labeled_hits"]) for row in selection_rows
        ),
        "selection_reason_counts": dict(sorted(reasons.items())),
        "selection_compute_ms": sum(float(row["compute_ms"]) for row in selection_rows),
        "selection_with_authority_backlog": sum(
            int(row["authoritative_queued_at_selection"] > 0)
            for row in selection_rows
        ),
        "coarse_load_kill_switch_windows": sum(
            bool(row["coarse_load_kill_switch"]) for row in selection_rows
        ),
        "predictor_windows_evaluated": sum(
            int(row["predictor_windows_evaluated"]) for row in selection_rows
        ),
        "mean_load_pressure": (
            statistics.fmean(float(row["load_pressure"]) for row in selection_rows)
            if selection_rows
            else 0.0
        ),
        "mean_shadow_price": (
            statistics.fmean(float(row["shadow_price"]) for row in selection_rows)
            if selection_rows
            else 0.0
        ),
        "exact_hits": len(useful_records),
        "overlap_hits": sum(row["overlap_producing"] for row in authoritative_rows),
        "source_counts": dict(sorted(Counter(row["source"] for row in authoritative_rows).items())),
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
        "saved_service_ms": saved_service_ms,
        "wasted_service_ms_per_target": ratio(
            wrong_service_ms + hedged_exact_loser_service_ms,
            target_count,
        ),
        "physical_started": physical_started,
        "physical_amplification": ratio(physical_started, target_count),
        "speculative_start_allocation": {
            "sessions_with_start_fraction": ratio(
                sum(value > 0 for value in start_allocations),
                len(start_allocations),
            ),
            "top_10pct_session_share": top_fraction_share(start_allocations),
            "jain_all_sessions": jain_index(start_allocations),
            "max_starts_one_session": max(start_allocations, default=0),
        },
        "late_speculative_starts": late_speculative_starts,
        "late_speculative_start_max_ms": max(late_start_deltas_ms, default=0.0),
        "speculative_overtakes": speculative_overtakes,
        "unsafe_speculative_overtakes": unsafe_speculative_overtakes,
        "max_running_total": int(stats["max_running_total"]),
        "max_running_speculative": int(stats["max_running_speculative"]),
        "max_running_speculative_visit": int(
            stats["max_running_speculative_by_tool"].get("visit", 0)
        ),
        "max_queued_authoritative": int(stats["max_queued_authoritative"]),
        "max_authoritative_outstanding": max_authoritative_outstanding,
        "authoritative_rows": sorted(authoritative_rows, key=lambda row: row["target_id"]),
        "mean_exposed_wait_ms": statistics.fmean(
            float(row["exposed_wait_ms"]) for row in authoritative_rows
        ),
        "p95_exposed_wait_ms": percentile(
            [float(row["exposed_wait_ms"]) for row in authoritative_rows], 0.95
        ),
        "p99_exposed_wait_ms": percentile(
            [float(row["exposed_wait_ms"]) for row in authoritative_rows], 0.99
        ),
        "mean_scheduled_response_ms": statistics.fmean(
            float(row["scheduled_response_ms"]) for row in authoritative_rows
        ),
        "p95_scheduled_response_ms": percentile(
            [float(row["scheduled_response_ms"]) for row in authoritative_rows], 0.95
        ),
        "p99_scheduled_response_ms": percentile(
            [float(row["scheduled_response_ms"]) for row in authoritative_rows], 0.99
        ),
        "arrival_lateness_ms": {
            "release_p95": percentile(release_lateness_ms, 0.95),
            "confirmation_p95": percentile(confirmation_lateness_ms, 0.95),
            "confirmation_max": max(confirmation_lateness_ms, default=0.0),
        },
        "admission_ms": {
            "calls": len(admission_ms),
            "mean": statistics.fmean(admission_ms) if admission_ms else 0.0,
            "p95": percentile(admission_ms, 0.95),
            "max": max(admission_ms, default=0.0),
        },
        "wall_s": wall_s,
        "drain_tail_s": max(0.0, drained_at - last_confirmation),
        "authoritative_throughput_per_s": ratio(target_count, wall_s),
        "safety": safety,
    }


def paired_metrics(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_rows = {
        str(row["target_id"]): row for row in baseline["authoritative_rows"]
    }
    treatment_rows = {
        str(row["target_id"]): row for row in treatment["authoritative_rows"]
    }
    if baseline_rows.keys() != treatment_rows.keys():
        raise RuntimeError("baseline/treatment logical target identities differ")
    response_deltas: list[float] = []
    exposed_deltas: list[float] = []
    miss_response_deltas: list[float] = []
    baseline_response_total = 0.0
    treatment_response_total = 0.0
    for target_id in sorted(baseline_rows):
        base = baseline_rows[target_id]
        test = treatment_rows[target_id]
        baseline_response = float(base["scheduled_response_ms"])
        treatment_response = float(test["scheduled_response_ms"])
        delta = treatment_response - baseline_response
        response_deltas.append(delta)
        exposed_deltas.append(
            float(test["exposed_wait_ms"]) - float(base["exposed_wait_ms"])
        )
        baseline_response_total += baseline_response
        treatment_response_total += treatment_response
        if not bool(test["overlap_producing"]):
            miss_response_deltas.append(delta)
    count = len(response_deltas)
    net_ms = baseline_response_total - treatment_response_total
    baseline_exposed_total = sum(
        float(row["exposed_wait_ms"]) for row in baseline_rows.values()
    )
    treatment_exposed_total = sum(
        float(row["exposed_wait_ms"]) for row in treatment_rows.values()
    )
    positive_response_regressions = [
        max(0.0, delta) for delta in response_deltas
    ]
    if treatment["scenario"] == "all_wrong_counterfactual" and (
        int(treatment["exact_hits"]) != 0
        or int(treatment["overlap_hits"]) != 0
    ):
        raise RuntimeError("all-wrong counterfactual produced an overlap hit")
    baseline_samples = baseline["samples"]
    treatment_samples = treatment["samples"]
    if len(baseline_samples) != len(treatment_samples):
        raise RuntimeError("paired repetition counts differ")
    repeat_net_scheduled_ms = []
    repeat_net_exposed_ms = []
    repeat_net_scheduled_ms_per_target = []
    repeat_net_exposed_ms_per_target = []
    for baseline_sample, treatment_sample in zip(
        baseline_samples, treatment_samples
    ):
        repeat_scheduled = (
            sum(
                float(row["scheduled_response_ms"])
                for row in baseline_sample["authoritative_rows"]
            )
            - sum(
                float(row["scheduled_response_ms"])
                for row in treatment_sample["authoritative_rows"]
            )
        )
        repeat_exposed = (
            sum(
                float(row["exposed_wait_ms"])
                for row in baseline_sample["authoritative_rows"]
            )
            - sum(
                float(row["exposed_wait_ms"])
                for row in treatment_sample["authoritative_rows"]
            )
        )
        repeat_targets = int(treatment_sample["authoritative_targets"])
        repeat_net_scheduled_ms.append(repeat_scheduled)
        repeat_net_exposed_ms.append(repeat_exposed)
        repeat_net_scheduled_ms_per_target.append(
            repeat_scheduled / repeat_targets
        )
        repeat_net_exposed_ms_per_target.append(
            repeat_exposed / repeat_targets
        )

    def compact(sample: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in sample.items()
            if key != "authoritative_rows"
        }

    return {
        "scenario": treatment["scenario"],
        "policy": treatment["policy"],
        "offered_load": treatment["offered_load"],
        "repetitions": treatment["repetitions"],
        "baseline_workers": treatment["baseline_workers"],
        "baseline_visit_capacity": treatment["baseline_visit_capacity"],
        "broker_workers": treatment["broker_workers"],
        "broker_visit_capacity": treatment["broker_visit_capacity"],
        "certified_isolated_speculative_slots": treatment[
            "certified_isolated_speculative_slots"
        ],
        "broker_speculative_pending": treatment[
            "broker_speculative_pending"
        ],
        "authoritative_targets": count,
        "requested_predictions": treatment["requested_predictions"],
        "admitted_predictions": treatment["admitted_predictions"],
        "capacity_rejected_predictions": treatment["rejected_predictions"],
        "deadline_rejected_predictions": treatment[
            "deadline_rejected_predictions"
        ],
        "replaced_queued_predictions": treatment[
            "replaced_queued_predictions"
        ],
        "exact_hits": treatment["exact_hits"],
        "overlap_hits": treatment["overlap_hits"],
        "overlap_coverage": ratio(treatment["overlap_hits"], count),
        "wrong_started": treatment["wrong_started"],
        "wrong_service_ms": treatment["wrong_service_ms"],
        "hedged_exact_losers": treatment["hedged_exact_losers"],
        "hedged_exact_loser_service_ms": treatment[
            "hedged_exact_loser_service_ms"
        ],
        "saved_service_ms": treatment["saved_service_ms"],
        "running_speculative_races": treatment[
            "running_speculative_races"
        ],
        "speculative_race_wins": treatment["speculative_race_wins"],
        "authoritative_race_wins": treatment[
            "authoritative_race_wins"
        ],
        "wasted_service_ms_per_target": (
            (
                treatment["wrong_service_ms"]
                + treatment["hedged_exact_loser_service_ms"]
            )
            / count
        ),
        "waste_to_saved_service_ratio": ratio(
            treatment["wrong_service_ms"]
            + treatment["hedged_exact_loser_service_ms"],
            treatment["saved_service_ms"],
        ),
        "physical_amplification": treatment["physical_started"] / count,
        "speculative_start_allocation": treatment[
            "speculative_start_allocation"
        ],
        "selection_reason_counts": treatment["selection_reason_counts"],
        "selection_with_authority_backlog": treatment[
            "selection_with_authority_backlog"
        ],
        "coarse_load_kill_switch_windows": treatment[
            "coarse_load_kill_switch_windows"
        ],
        "predictor_windows_evaluated": treatment[
            "predictor_windows_evaluated"
        ],
        "baseline_mean_scheduled_response_ms": baseline_response_total / count,
        "treatment_mean_scheduled_response_ms": treatment_response_total / count,
        "net_scheduled_response_benefit_ms_total": net_ms,
        "net_scheduled_response_benefit_ms_per_target": net_ms / count,
        "net_scheduled_response_benefit_fraction": ratio(
            net_ms, baseline_response_total
        ),
        "net_exposed_benefit_ms_total": (
            baseline_exposed_total - treatment_exposed_total
        ),
        "net_exposed_benefit_ms_per_target": (
            baseline_exposed_total - treatment_exposed_total
        )
        / count,
        "net_exposed_benefit_fraction": ratio(
            baseline_exposed_total - treatment_exposed_total,
            baseline_exposed_total,
        ),
        "paired_response_regression_ms": {
            "mean": statistics.fmean(response_deltas),
            "p95": percentile(response_deltas, 0.95),
            "p99": percentile(response_deltas, 0.99),
            "max": max(response_deltas),
            "delayed_gt_0_1ms_fraction": ratio(
                sum(delta > 0.1 for delta in response_deltas), count
            ),
            "top_10pct_positive_regression_share": top_fraction_share(
                positive_response_regressions
            ),
        },
        "miss_only_response_regression_ms": {
            "count": len(miss_response_deltas),
            "mean": (
                statistics.fmean(miss_response_deltas)
                if miss_response_deltas
                else 0.0
            ),
            "p95": percentile(miss_response_deltas, 0.95),
            "p99": percentile(miss_response_deltas, 0.99),
            "max": max(miss_response_deltas, default=0.0),
        },
        "paired_exposed_regression_ms": {
            "mean": statistics.fmean(exposed_deltas),
            "p95": percentile(exposed_deltas, 0.95),
            "p99": percentile(exposed_deltas, 0.99),
        },
        "baseline_wall_s": baseline["wall_s"],
        "treatment_wall_s": treatment["wall_s"],
        "drained_wall_benefit_fraction": ratio(
            baseline["wall_s"] - treatment["wall_s"], baseline["wall_s"]
        ),
        "baseline_authoritative_throughput_per_s": baseline[
            "authoritative_throughput_per_s"
        ],
        "treatment_authoritative_throughput_per_s": treatment[
            "authoritative_throughput_per_s"
        ],
        "throughput_ratio": ratio(
            treatment["authoritative_throughput_per_s"],
            baseline["authoritative_throughput_per_s"],
        ),
        "baseline_drain_tail_s": baseline["drain_tail_s"],
        "treatment_drain_tail_s": treatment["drain_tail_s"],
        "late_speculative_starts": treatment["late_speculative_starts"],
        "speculative_overtakes": treatment["speculative_overtakes"],
        "unsafe_speculative_overtakes": treatment[
            "unsafe_speculative_overtakes"
        ],
        "max_running_speculative_visit": treatment[
            "max_running_speculative_visit"
        ],
        "max_queued_authoritative": treatment["max_queued_authoritative"],
        "max_authoritative_outstanding": treatment[
            "max_authoritative_outstanding"
        ],
        "admission_ms": treatment["admission_ms"],
        "arrival_lateness_ms": treatment["arrival_lateness_ms"],
        "all_safety_invariants_passed": all(
            treatment["safety"].values()
        ) and all(baseline["safety"].values()),
        "repeat_net_scheduled_response_benefit_ms": repeat_net_scheduled_ms,
        "repeat_net_exposed_benefit_ms": repeat_net_exposed_ms,
        "repeat_net_scheduled_response_benefit_ms_per_target": (
            repeat_net_scheduled_ms_per_target
        ),
        "repeat_net_exposed_benefit_ms_per_target": (
            repeat_net_exposed_ms_per_target
        ),
        "repeat_net_scheduled_summary_ms_per_target": {
            "median": statistics.median(
                repeat_net_scheduled_ms_per_target
            ),
            "min": min(repeat_net_scheduled_ms_per_target),
            "max": max(repeat_net_scheduled_ms_per_target),
        },
        "positive_no_signal_repeats_treated_as_noise": (
            sum(value > 0.0 for value in repeat_net_scheduled_ms_per_target)
            if treatment["scenario"] == "all_wrong_counterfactual"
            or int(treatment["overlap_hits"]) == 0
            else 0
        ),
        "counterbalance_orders": [
            str(sample.get("pair_order", "unpaired"))
            for sample in treatment_samples
        ],
        "net_interpretation": (
            "positive_zero_signal_estimate_is_timing_noise_not_benefit"
            if (
                treatment["scenario"] == "all_wrong_counterfactual"
                or int(treatment["overlap_hits"]) == 0
            )
            and any(
                value > 0.0
                for value in repeat_net_scheduled_ms_per_target
            )
            else "all_wrong_worst_case_cost_no_positive_repeat"
            if treatment["scenario"] == "all_wrong_counterfactual"
            else "zero_overlap_no_latency_benefit"
            if int(treatment["overlap_hits"]) == 0
            else "scheduler_marginal_development_estimate"
        ),
        "samples": {
            "baseline": [compact(sample) for sample in baseline_samples],
            "treatment": [compact(sample) for sample in treatment_samples],
        },
    }


def _sum_scalars(samples: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(sample[key]) for sample in samples)


def aggregate_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    policy: str,
    offered_load: float,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot aggregate empty samples")
    rows = []
    for sample_index, sample in enumerate(samples):
        for authoritative_row in sample["authoritative_rows"]:
            row = dict(authoritative_row)
            # Repetitions replay the same logical target IDs. Prefix the
            # repetition so paired aggregation retains every observation.
            row["target_id"] = f"rep{sample_index}:{row['target_id']}"
            rows.append(row)
    reasons = sum(
        (Counter(sample["selection_reason_counts"]) for sample in samples),
        Counter(),
    )
    wall_s = _sum_scalars(samples, "wall_s")
    targets = int(_sum_scalars(samples, "authoritative_targets"))
    result = {
        "scenario": scenario,
        "policy": policy,
        "offered_load": offered_load,
        "repetitions": len(samples),
        "baseline_workers": int(samples[0]["baseline_workers"]),
        "baseline_visit_capacity": int(
            samples[0]["baseline_visit_capacity"]
        ),
        "broker_workers": int(samples[0]["broker_workers"]),
        "broker_visit_capacity": int(
            samples[0]["broker_visit_capacity"]
        ),
        "certified_isolated_speculative_slots": int(
            samples[0]["certified_isolated_speculative_slots"]
        ),
        "broker_speculative_pending": int(
            samples[0]["broker_speculative_pending"]
        ),
        "authoritative_targets": targets,
        "requested_predictions": int(_sum_scalars(samples, "requested_predictions")),
        "admitted_predictions": int(_sum_scalars(samples, "admitted_predictions")),
        "rejected_predictions": int(_sum_scalars(samples, "rejected_predictions")),
        "deadline_rejected_predictions": int(
            _sum_scalars(samples, "deadline_rejected_predictions")
        ),
        "replaced_queued_predictions": int(
            _sum_scalars(samples, "replaced_queued_predictions")
        ),
        "exact_hits": int(_sum_scalars(samples, "exact_hits")),
        "overlap_hits": int(_sum_scalars(samples, "overlap_hits")),
        "wrong_started": int(_sum_scalars(samples, "wrong_started")),
        "wrong_service_ms": _sum_scalars(samples, "wrong_service_ms"),
        "hedged_exact_losers": int(
            _sum_scalars(samples, "hedged_exact_losers")
        ),
        "hedged_exact_loser_service_ms": _sum_scalars(
            samples, "hedged_exact_loser_service_ms"
        ),
        "saved_service_ms": _sum_scalars(samples, "saved_service_ms"),
        "running_speculative_races": int(
            _sum_scalars(samples, "running_speculative_races")
        ),
        "speculative_race_wins": int(
            _sum_scalars(samples, "speculative_race_wins")
        ),
        "authoritative_race_wins": int(
            _sum_scalars(samples, "authoritative_race_wins")
        ),
        "physical_started": int(_sum_scalars(samples, "physical_started")),
        "speculative_start_allocation": {
            "sessions_with_start_fraction": statistics.fmean(
                float(
                    sample["speculative_start_allocation"][
                        "sessions_with_start_fraction"
                    ]
                )
                for sample in samples
            ),
            "top_10pct_session_share": statistics.fmean(
                float(
                    sample["speculative_start_allocation"][
                        "top_10pct_session_share"
                    ]
                )
                for sample in samples
            ),
            "jain_all_sessions": statistics.fmean(
                float(
                    sample["speculative_start_allocation"][
                        "jain_all_sessions"
                    ]
                )
                for sample in samples
            ),
            "max_starts_one_session": max(
                int(
                    sample["speculative_start_allocation"][
                        "max_starts_one_session"
                    ]
                )
                for sample in samples
            ),
        },
        "selection_reason_counts": dict(sorted(reasons.items())),
        "selection_with_authority_backlog": int(
            _sum_scalars(samples, "selection_with_authority_backlog")
        ),
        "coarse_load_kill_switch_windows": int(
            _sum_scalars(samples, "coarse_load_kill_switch_windows")
        ),
        "predictor_windows_evaluated": int(
            _sum_scalars(samples, "predictor_windows_evaluated")
        ),
        "wall_s": wall_s,
        "drain_tail_s": _sum_scalars(samples, "drain_tail_s"),
        "authoritative_throughput_per_s": ratio(targets, wall_s),
        "late_speculative_starts": int(
            _sum_scalars(samples, "late_speculative_starts")
        ),
        "speculative_overtakes": int(_sum_scalars(samples, "speculative_overtakes")),
        "unsafe_speculative_overtakes": int(
            _sum_scalars(samples, "unsafe_speculative_overtakes")
        ),
        "max_running_speculative_visit": max(
            int(sample["max_running_speculative_visit"]) for sample in samples
        ),
        "max_queued_authoritative": max(
            int(sample["max_queued_authoritative"]) for sample in samples
        ),
        "max_authoritative_outstanding": max(
            int(sample["max_authoritative_outstanding"]) for sample in samples
        ),
        "admission_ms": {
            "mean": statistics.fmean(float(sample["admission_ms"]["mean"]) for sample in samples),
            "p95_mean_across_repeats": statistics.fmean(
                float(sample["admission_ms"]["p95"]) for sample in samples
            ),
            "max": max(float(sample["admission_ms"]["max"]) for sample in samples),
        },
        "arrival_lateness_ms": {
            "release_p95_mean_across_repeats": statistics.fmean(
                float(sample["arrival_lateness_ms"]["release_p95"])
                for sample in samples
            ),
            "confirmation_p95_mean_across_repeats": statistics.fmean(
                float(sample["arrival_lateness_ms"]["confirmation_p95"])
                for sample in samples
            ),
            "confirmation_max": max(
                float(sample["arrival_lateness_ms"]["confirmation_max"])
                for sample in samples
            ),
        },
        "authoritative_rows": rows,
        "safety": {
            key: all(bool(sample["safety"][key]) for sample in samples)
            for key in samples[0]["safety"]
        },
        "samples": list(samples),
    }
    return result


async def run_matrix(
    windows: Sequence[ScoredWindow],
    *,
    policy_names: Sequence[str],
    loads: Sequence[float],
    repetitions: int,
    cycles: int,
    workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
    isolated_speculative_slots: int = 0,
) -> list[dict[str, Any]]:
    specs = policy_specs()
    scenarios = {
        "observed_nested_oof": list(windows),
        "all_wrong_counterfactual": force_all_wrong(windows),
    }
    cells: list[dict[str, Any]] = []
    for scenario, scenario_windows in scenarios.items():
        for load in loads:
            for name in policy_names:
                baseline_samples: list[dict[str, Any]] = []
                treatment_samples: list[dict[str, Any]] = []
                for repetition in range(repetitions):
                    seed = 7301 + repetition

                    async def replay(
                        replay_policy: OpenLoopPolicy | None,
                    ) -> dict[str, Any]:
                        return await run_open_loop_sample(
                            scenario_windows,
                            policy=replay_policy,
                            offered_load=load,
                            seed=seed,
                            cycles=cycles,
                            workers=workers,
                            visit_capacity=visit_capacity,
                            max_speculative_pending=max_speculative_pending,
                            service_ms=service_ms,
                            lead_ms=lead_ms,
                            isolated_speculative_slots=(
                                isolated_speculative_slots
                            ),
                        )

                    order = "AB" if repetition % 2 == 0 else "BA"
                    if order == "AB":
                        baseline_sample = await replay(None)
                        treatment_sample = await replay(specs[name])
                    else:
                        treatment_sample = await replay(specs[name])
                        baseline_sample = await replay(None)
                    baseline_sample["pair_order"] = order
                    treatment_sample["pair_order"] = order
                    baseline_sample["execution_position"] = (
                        1 if order == "AB" else 2
                    )
                    treatment_sample["execution_position"] = (
                        2 if order == "AB" else 1
                    )
                    baseline_samples.append(baseline_sample)
                    treatment_samples.append(treatment_sample)

                baseline = aggregate_samples(
                    baseline_samples,
                    scenario=scenario,
                    policy="demand_only",
                    offered_load=load,
                )
                treatment = aggregate_samples(
                    treatment_samples,
                    scenario=scenario,
                    policy=name,
                    offered_load=load,
                )
                cells.append(paired_metrics(baseline, treatment))
    return cells


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Pattern-v2 sustained open-loop authority stress",
        "",
        (
            "Every decision release and authoritative confirmation is fixed on "
            "an exogenous timeline shared by demand-only and treatment. Results "
            "use a non-preemptible synthetic executor; no vLLM or network is used."
        ),
        "",
        "| Scenario | Policy | Load / peak authority C | Overlap hit | Wrong starts | Waste/target | Physical amp | Miss p95 regression | Response p99 regression | Net exposed / scheduled per target | Scheduled repeat median / min | Throughput ratio | Top-10% start share | Backlog / coarse abstain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cells"]:
        lines.append(
            f"| {row['scenario']} | {row['policy']} | "
            f"{row['offered_load']:.2f} / {row['max_authoritative_outstanding']} | "
            f"{100.0 * row['overlap_coverage']:.1f}% | {row['wrong_started']} | "
            f"{row['wasted_service_ms_per_target']:.2f} ms | "
            f"{row['physical_amplification']:.2f}x | "
            f"{row['miss_only_response_regression_ms']['p95']:+.2f} ms | "
            f"{row['paired_response_regression_ms']['p99']:+.2f} ms | "
            f"{row['net_exposed_benefit_ms_per_target']:+.2f} / "
            f"{row['net_scheduled_response_benefit_ms_per_target']:+.2f} ms | "
            f"{row['repeat_net_scheduled_summary_ms_per_target']['median']:+.2f} / "
            f"{row['repeat_net_scheduled_summary_ms_per_target']['min']:+.2f} ms | "
            f"{row['throughput_ratio']:.3f} | "
            f"{100.0 * row['speculative_start_allocation']['top_10pct_session_share']:.1f}% | "
            f"{row['selection_reason_counts'].get('authoritative_backlog', 0)} / "
            f"{row['selection_reason_counts'].get('coarse_load_kill_switch', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Offered load is authoritative service demand divided by visit capacity; speculative work is extra.",
            "- Every scored prefix is cloned as an independent task. Peak authority C is not original-source-session concurrency.",
            "- Each policy has its own paired demand-only runs, counterbalanced AB/BA by repetition.",
            "- Scheduled response includes event-loop arrival lateness plus broker exposed wait. It is the primary authority metric.",
            "- Peak authority C is the maximum number of confirmed calls outstanding at once; offered load is exogenous URL-call utilization.",
            "- A queued promotion is not an overlap-producing hit.",
            "- rank5 policies are full-fire legacy controls; rank_budgeted_reserved is the equal one-start rank-only control.",
            "- Top-10% start share measures priority concentration across decision sessions; Jain allocation breadth is retained in metrics.json.",
            "- Reserve bounds simultaneous speculative visit work but cannot preempt a call that already started.",
            "- safe_global_benefit uses only the configured isolated K slots; K=0 is a demand-only fast path, while K>0 preserves the original authority worker and visit caps.",
            "- A running exact safe prediction races a fresh protected authority backup; the first speculative success or backup terminal result decides the request, and loser drain is off the return path.",
            "- Open-loop selection is causal: the broker globally prioritizes currently queued candidates and can replace a lower queued item, but cannot preempt running work or optimize against future arrivals.",
            "- All-wrong preserves candidates, probabilities, and the arrival plan while replacing every authoritative URL; causal backlog feedback can still change later admissions.",
            "- Any positive zero-overlap or all-wrong net estimate is timing noise, not a latency-benefit claim; inspect repeat median/min.",
            "- Net latency is scheduler-marginal: Pattern feature extraction and OOF probability lookup are precomputed and excluded.",
            "- This is development evidence from nested grouped OOF traces, not the untouched confirmatory holdout.",
            "",
            "## Reproduction",
            "",
            "```bash",
            str(payload["reproduction_command"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario": row["scenario"],
        "policy": row["policy"],
        "offered_load": row["offered_load"],
        "authoritative_targets": row["authoritative_targets"],
        "overlap_coverage": row["overlap_coverage"],
        "wrong_started": row["wrong_started"],
        "wrong_service_ms": row["wrong_service_ms"],
        "wasted_service_ms_per_target": row["wasted_service_ms_per_target"],
        "physical_amplification": row["physical_amplification"],
        "miss_p95_regression_ms": row["miss_only_response_regression_ms"]["p95"],
        "response_p99_regression_ms": row["paired_response_regression_ms"]["p99"],
        "net_benefit_ms_per_target": row[
            "net_scheduled_response_benefit_ms_per_target"
        ],
        "net_exposed_benefit_ms_per_target": row[
            "net_exposed_benefit_ms_per_target"
        ],
        "repeat_scheduled_net_median_ms_per_target": row[
            "repeat_net_scheduled_summary_ms_per_target"
        ]["median"],
        "repeat_scheduled_net_min_ms_per_target": row[
            "repeat_net_scheduled_summary_ms_per_target"
        ]["min"],
        "net_interpretation": row["net_interpretation"],
        "throughput_ratio": row["throughput_ratio"],
        "max_authoritative_outstanding": row["max_authoritative_outstanding"],
        "speculative_top_10pct_session_share": row[
            "speculative_start_allocation"
        ]["top_10pct_session_share"],
        "backlog_abstain": row["selection_reason_counts"].get(
            "authoritative_backlog", 0
        ),
        "coarse_load_kill_switch": row["selection_reason_counts"].get(
            "coarse_load_kill_switch", 0
        ),
        "late_speculative_starts": row["late_speculative_starts"],
        "all_safety_invariants_passed": row["all_safety_invariants_passed"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--loads", type=parse_csv_floats, default=DEFAULT_LOADS)
    parser.add_argument(
        "--policies",
        type=parse_csv_names,
        default=(
            "rank5_unreserved",
            "rank5_reserved",
            "rank_budgeted_reserved",
            "confidence_reserved",
            "utility_authority_first",
            "utility_risk_limited",
            "safe_global_benefit",
        ),
    )
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visit-capacity", type=int, default=2)
    parser.add_argument(
        "--isolated-speculative-slots",
        type=int,
        default=0,
        help=(
            "extra worker+visit slots unavailable to baseline authority; "
            "safe_global_benefit abstains when this is zero"
        ),
    )
    parser.add_argument("--max-speculative-pending", type=int, default=128)
    parser.add_argument("--service-ms", type=float, default=5.0)
    parser.add_argument("--lead-ms", type=float, default=2.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repetitions <= 0 or args.cycles <= 0:
        raise SystemExit("repetitions and cycles must be positive")
    if args.workers <= 1 or not 1 <= args.visit_capacity <= args.workers:
        raise SystemExit("invalid worker or visit capacity")
    if args.max_speculative_pending <= 0:
        raise SystemExit("max speculative pending must be positive")
    if args.isolated_speculative_slots < 0:
        raise SystemExit("isolated speculative slots must be non-negative")
    if args.service_ms <= 0.0 or args.lead_ms < 0.0:
        raise SystemExit("service must be positive and lead non-negative")
    if not args.traces.is_dir():
        raise SystemExit(f"trace directory does not exist: {args.traces}")

    windows, oof = collect_nested_oof_windows(args.traces)
    cells = asyncio.run(
        run_matrix(
            windows,
            policy_names=args.policies,
            loads=args.loads,
            repetitions=args.repetitions,
            cycles=args.cycles,
            workers=args.workers,
            visit_capacity=args.visit_capacity,
            max_speculative_pending=args.max_speculative_pending,
            service_ms=args.service_ms,
            lead_ms=args.lead_ms,
            isolated_speculative_slots=args.isolated_speculative_slots,
        )
    )
    if not all(bool(row["all_safety_invariants_passed"]) for row in cells):
        raise RuntimeError("one or more open-loop cells failed safety invariants")

    reproduction_command = " ".join(
        [
            "PYTHONPATH=reproduction:reproduction/scripts",
            "python",
            "reproduction/scripts/run_pattern_v2_open_loop_stress.py",
            "--traces",
            shlex.quote(str(args.traces)),
            "--output",
            shlex.quote(str(args.output)),
            "--loads",
            ",".join(map(str, args.loads)),
            "--policies",
            ",".join(args.policies),
            "--repetitions",
            str(args.repetitions),
            "--cycles",
            str(args.cycles),
            "--workers",
            str(args.workers),
            "--visit-capacity",
            str(args.visit_capacity),
            "--isolated-speculative-slots",
            str(args.isolated_speculative_slots),
            "--max-speculative-pending",
            str(args.max_speculative_pending),
            "--service-ms",
            str(args.service_ms),
            "--lead-ms",
            str(args.lead_ms),
        ]
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_not_confirmatory",
        "oof": oof,
        "configuration": {
            "arrival_model": "sustained open-loop absolute decision and confirmation times",
            "task_model": (
                "each scored prefix is an independent cloned task; this is "
                "not original-source-session concurrency"
            ),
            "pairing": (
                "independent demand-only pair per policy with even AB and odd BA order"
            ),
            "latency_scope": (
                "scheduler_marginal_only; precomputed Pattern feature extraction "
                "and OOF probability lookup are excluded"
            ),
            "excluded_precompute_runtime_ms_per_dataset_replay": {
                "pattern_features": float(
                    oof["runtime_pattern_feature_ms"]["total"]
                ),
                "probability_lookup": float(
                    oof["runtime_probability_lookup_ms"]["total"]
                ),
            },
            "loads": list(args.loads),
            "policies": list(args.policies),
            "repetitions": args.repetitions,
            "cycles": args.cycles,
            "workers": args.workers,
            "visit_capacity": args.visit_capacity,
            "isolated_speculative_slots": args.isolated_speculative_slots,
            "max_speculative_pending": args.max_speculative_pending,
            "synthetic_service_ms": args.service_ms,
            "prediction_lead_ms": args.lead_ms,
            "executor_cancellation": "non_preemptible_until_service_drains",
            "network_requests": 0,
            "vllm_required": False,
        },
        "policies": {
            name: asdict(policy_specs()[name]) for name in args.policies
        },
        "cells": cells,
        "source_files": {
            "runner": {"path": str(SCRIPT), "sha256": sha256_file(SCRIPT)},
            "adaptive_runner": {
                "path": str(SCRIPT.parent / "run_pattern_v2_adaptive_load.py"),
                "sha256": sha256_file(
                    SCRIPT.parent / "run_pattern_v2_adaptive_load.py"
                ),
            },
            "policy": {
                "path": str(REPRODUCTION_ROOT / "paste_repro/speculation_policy.py"),
                "sha256": sha256_file(
                    REPRODUCTION_ROOT / "paste_repro/speculation_policy.py"
                ),
            },
            "broker": {
                "path": str(REPRODUCTION_ROOT / "paste_repro/live_broker.py"),
                "sha256": sha256_file(
                    REPRODUCTION_ROOT / "paste_repro/live_broker.py"
                ),
            },
        },
        "reproduction_command": reproduction_command,
    }
    payload["result_sha256_excluding_self"] = canonical_sha256(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    csv_rows = [_csv_row(row) for row in cells]
    with (args.output / "open_loop.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "cells": len(cells),
                "all_safety_invariants_passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
