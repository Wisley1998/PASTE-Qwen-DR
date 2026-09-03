#!/usr/bin/env python3
"""Paired open-loop no-interference check for the process sidecar.

This supplement deliberately tests only the all-wrong miss path.  For each
repeat it constructs one immutable authority-arrival trace, then replays that
same trace against K=0 and a process sidecar.  Every authority coroutine is
created before the timed origin and sleeps until its own absolute monotonic
deadline; completion of an earlier call can never move a later arrival.

The workload is a deterministic CPU-only replay using synthetic tool sleep.
It neither starts a model server nor exercises shared remote-service quotas.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
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
from paste_repro.live_broker import (  # noqa: E402
    LiveAuthoritativeResult,
    LiveToolBroker,
)
from paste_repro.speculation_sidecar import (  # noqa: E402
    ProcessSpeculativeSidecar,
    choose_authority_sidecar_cpus,
)
from run_pattern_cache_evaluation import sha256_file  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredWindow,
    _select_candidates,
    calibration_quality,
    collect_nested_oof_windows,
    force_all_wrong,
    policy_specs,
    session_stream_batches,
)
from run_pattern_v2_load_robustness import (  # noqa: E402
    canonical_sha256,
    percentile,
    ratio,
)
from run_pattern_v2_sidecar_load import (  # noqa: E402
    _paired_repeat_inference,
    _sidecar_count,
    _snapshot_json,
)


SCHEMA = "paste_repro.pattern_v2_sidecar_open_loop_no_interference.v1"
RAW_SCHEMA = f"{SCHEMA}.raw_repeat_vectors.v1"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT
    / "results"
    / "pattern_v2_sidecar_open_loop_no_interference"
)


@dataclass(frozen=True)
class FixedArrivalEpoch:
    """One decision batch in a trace frozen before either paired replay."""

    index: int
    windows: tuple[ScoredWindow, ...]
    speculation_offset_s: float
    authority_offset_s: float
    speculation_phase_guard_s: float
    target_count: int
    ideal_service_waves: int


@dataclass(frozen=True)
class AuthorityCompletion:
    target_id: str
    planned_offset_s: float
    scheduled_at: float
    first_run_at: float
    terminal_at: float
    result: LiveAuthoritativeResult


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def build_fixed_arrival_trace(
    windows: Sequence[ScoredWindow],
    *,
    task_concurrency: int,
    seed: int,
    visit_capacity: int,
    service_s: float,
    lead_s: float,
    speculation_phase_guard_s: float = 0.0,
) -> tuple[FixedArrivalEpoch, ...]:
    """Freeze the ideal closed-loop cadence into an open-loop trace.

    ``session_stream_batches`` retains the original definition of task
    concurrency.  The next epoch is placed at the *modeled* completion of the
    prior epoch (lead plus enough service waves for its targets), never at an
    observed completion.  Thus the resulting offsets are exogenous to both
    K=0 and K>0 executions.
    """

    if task_concurrency <= 0 or visit_capacity <= 0:
        raise ValueError("task_concurrency and visit_capacity must be positive")
    if service_s <= 0.0 or lead_s < 0.0:
        raise ValueError("service_s must be positive and lead_s non-negative")
    if speculation_phase_guard_s < 0.0:
        raise ValueError("speculation_phase_guard_s must be non-negative")
    if (
        speculation_phase_guard_s > 0.0
        and speculation_phase_guard_s >= lead_s
    ):
        raise ValueError(
            "speculation_phase_guard_s must be smaller than lead_s"
        )
    batches = session_stream_batches(
        windows,
        offered_concurrency=task_concurrency,
        seed=seed,
    )
    cursor_s = 0.0
    epochs: list[FixedArrivalEpoch] = []
    for index, batch in enumerate(batches):
        target_count = sum(len(window.executable_targets) for window in batch)
        service_waves = math.ceil(target_count / visit_capacity)
        authority_offset_s = cursor_s + lead_s
        applied_guard_s = (
            0.0 if index == 0 else speculation_phase_guard_s
        )
        epochs.append(
            FixedArrivalEpoch(
                index=index,
                windows=tuple(batch),
                speculation_offset_s=cursor_s + applied_guard_s,
                authority_offset_s=authority_offset_s,
                speculation_phase_guard_s=applied_guard_s,
                target_count=target_count,
                ideal_service_waves=service_waves,
            )
        )
        cursor_s = authority_offset_s + service_waves * service_s
    return tuple(epochs)


def arrival_trace_rows(
    schedule: Sequence[FixedArrivalEpoch],
) -> list[dict[str, Any]]:
    """Return the exact planned authority-arrival vector for auditing."""

    rows: list[dict[str, Any]] = []
    for epoch in schedule:
        for window_index, window in enumerate(epoch.windows):
            for target_index, target in enumerate(window.executable_targets):
                rows.append(
                    {
                        "target_id": (
                            f"e{epoch.index}:w{window_index}:"
                            f"{window.session_id}:{window.decision_id}:"
                            f"target:{target_index}"
                        ),
                        "epoch": epoch.index,
                        "session_id": window.session_id,
                        "decision_id": window.decision_id,
                        "target_url": target,
                        "authority_offset_s": epoch.authority_offset_s,
                    }
                )
    return rows


def arrival_trace_manifest(
    schedule: Sequence[FixedArrivalEpoch],
) -> dict[str, Any]:
    rows = arrival_trace_rows(schedule)
    epoch_rows = [
        {
            "epoch": epoch.index,
            "speculation_offset_s": epoch.speculation_offset_s,
            "authority_offset_s": epoch.authority_offset_s,
            "speculation_phase_guard_s": (
                epoch.speculation_phase_guard_s
            ),
            "effective_speculation_lead_s": (
                epoch.authority_offset_s - epoch.speculation_offset_s
            ),
            "target_count": epoch.target_count,
            "ideal_service_waves": epoch.ideal_service_waves,
        }
        for epoch in schedule
    ]
    value = {"epochs": epoch_rows, "authority_arrivals": rows}
    return {
        **value,
        "sha256": canonical_sha256(value),
        "authority_targets": len(rows),
    }


async def _sleep_until(deadline: float) -> None:
    await asyncio.sleep(max(0.0, deadline - time.monotonic()))


async def _scheduled_authority_call(
    broker: LiveToolBroker,
    invocation: Invocation,
    *,
    session_id: str,
    target_id: str,
    planned_offset_s: float,
    origin_future: "asyncio.Future[float]",
    armed: "asyncio.Queue[None]",
) -> AuthorityCompletion:
    """Release one authority request at a precomputed absolute deadline."""

    armed.put_nowait(None)
    origin = await origin_future
    scheduled_at = origin + planned_offset_s
    await _sleep_until(scheduled_at)
    first_run_at = time.monotonic()
    result = await broker.authoritative(
        invocation,
        session_id=session_id,
        speculation_eligible=False,
    )
    return AuthorityCompletion(
        target_id=target_id,
        planned_offset_s=planned_offset_s,
        scheduled_at=scheduled_at,
        first_run_at=first_run_at,
        terminal_at=time.monotonic(),
        result=result,
    )


def _assert_all_wrong(schedule: Sequence[FixedArrivalEpoch]) -> None:
    for epoch in schedule:
        for window in epoch.windows:
            targets = set(window.executable_targets)
            candidates = {row.pattern.url for row in window.candidates}
            if targets.intersection(candidates):
                raise ValueError(
                    "supplemental runner accepts only all-wrong windows"
                )


async def run_fixed_arrival_sample(
    schedule: Sequence[FixedArrivalEpoch],
    *,
    task_concurrency: int,
    seed: int,
    workers: int,
    visit_capacity: int,
    service_ms: float,
    lead_ms: float,
    sidecar_slots: int,
    max_sidecar_pending: int,
    probability_threshold: float,
    claim_grace_ms: float = 10.0,
    prestart_ms: float = 50.0,
    cpu_isolation: bool = True,
    authority_control_burst_limit: int = 0,
) -> dict[str, Any]:
    """Replay one trace and restore authority affinity on every exit path."""

    original_affinity = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set()
    )
    authority_affinity: set[int] | None = None
    sidecar_affinity: set[int] | None = None
    pinned = False
    if authority_control_burst_limit < 0:
        raise ValueError("authority_control_burst_limit must be non-negative")
    positive_resource_certificate = (
        sidecar_slots > 0 and authority_control_burst_limit > 0
    )
    if cpu_isolation:
        available = sorted(original_affinity)
        required = 2 if positive_resource_certificate else 1
        if len(available) < required:
            raise RuntimeError(
                "CPU isolation has fewer granted CPUs than required by the "
                "active resource certificate"
            )
        if positive_resource_certificate:
            authority_cpu, sidecar_cpu = choose_authority_sidecar_cpus(
                available
            )
            authority_affinity = {authority_cpu}
            sidecar_affinity = {sidecar_cpu}
        else:
            authority_affinity = {available[0]}
    try:
        if authority_affinity is not None:
            os.sched_setaffinity(0, authority_affinity)
            pinned = True
        return await _run_fixed_arrival_sample_pinned(
            schedule,
            task_concurrency=task_concurrency,
            seed=seed,
            workers=workers,
            visit_capacity=visit_capacity,
            service_ms=service_ms,
            lead_ms=lead_ms,
            sidecar_slots=sidecar_slots,
            max_sidecar_pending=max_sidecar_pending,
            probability_threshold=probability_threshold,
            claim_grace_ms=claim_grace_ms,
            prestart_ms=prestart_ms,
            cpu_isolation=cpu_isolation,
            authority_control_burst_limit=authority_control_burst_limit,
            authority_affinity=authority_affinity,
            sidecar_affinity=sidecar_affinity,
        )
    finally:
        if pinned and original_affinity:
            os.sched_setaffinity(0, original_affinity)


async def _run_fixed_arrival_sample_pinned(
    schedule: Sequence[FixedArrivalEpoch],
    *,
    task_concurrency: int,
    seed: int,
    workers: int,
    visit_capacity: int,
    service_ms: float,
    lead_ms: float,
    sidecar_slots: int,
    max_sidecar_pending: int,
    probability_threshold: float,
    claim_grace_ms: float = 10.0,
    prestart_ms: float = 50.0,
    cpu_isolation: bool = True,
    authority_control_burst_limit: int = 0,
    authority_affinity: set[int] | None,
    sidecar_affinity: set[int] | None,
) -> dict[str, Any]:
    """Implementation entered only after optional authority CPU pinning."""

    if not schedule:
        raise ValueError("schedule must not be empty")
    if sidecar_slots > 0 and prestart_ms <= 0.0:
        raise ValueError("process preload requires positive prestart_ms")
    _assert_all_wrong(schedule)
    service_s = service_ms / 1000.0
    lead_s = lead_ms / 1000.0
    trace = arrival_trace_manifest(schedule)
    if not trace["authority_arrivals"]:
        raise ValueError("schedule must contain at least one authority target")

    async def authority_executor(invocation: Invocation) -> dict[str, Any]:
        await asyncio.sleep(service_s)
        return {"invocation_key": invocation.key}

    async def sidecar_executor(invocation: Invocation) -> dict[str, Any]:
        await asyncio.sleep(service_s)
        return {"invocation_key": invocation.key}

    authority = LiveToolBroker(
        authority_executor,
        max_workers=workers,
        max_speculative_workers=0,
        max_speculative_pending=1,
        ttl_s=1.0,
        tool_capacities={"visit": visit_capacity},
    )
    sidecar: ProcessSpeculativeSidecar | None = None
    authority_tasks: list[asyncio.Task[AuthorityCompletion]] = []
    authority_completions: list[AuthorityCompletion] = []
    preload_plan: list[
        tuple[
            float,
            float,
            tuple[tuple[Invocation, str, str, float, str], ...],
        ]
    ] = []
    preload_requested = 0
    preload_handles_returned = 0
    preload_started_at = time.monotonic()
    preload_done_at = preload_started_at
    sidecar_before_close: Mapping[str, Any] = {}
    sidecar_after_close: Mapping[str, Any] = {}
    bridge_started_before_authority_done = False
    drained_at = time.monotonic()

    safe_spec = replace(
        next(
            spec
            for spec in policy_specs()
            if spec.name == "safe_global_benefit"
        ),
        confidence_threshold=probability_threshold,
    )
    selection_plan: list[list[tuple[Any, float]]] = []
    selection_metadata: list[dict[str, Any]] = []
    burst_latch_open = authority_control_burst_limit > 0
    for epoch in schedule:
        effective_lead_s = max(
            0.0,
            epoch.authority_offset_s - epoch.speculation_offset_s,
        )
        burst_limit_exceeded = (
            authority_control_burst_limit == 0
            or epoch.target_count > authority_control_burst_limit
        )
        if burst_limit_exceeded:
            burst_latch_open = False
        burst_gate_open = burst_latch_open
        selected, metadata = _select_candidates(
            epoch.windows,
            safe_spec,
            visit_capacity=visit_capacity,
            service_s=service_s,
            lead_s=effective_lead_s,
            isolated_speculative_slots=max(0, sidecar_slots),
            safe_start_limit=(
                max(0, sidecar_slots) if burst_gate_open else 0
            ),
        )
        metadata = {
            **metadata,
            "effective_speculation_lead_s": effective_lead_s,
            "speculation_phase_guard_s": (
                epoch.speculation_phase_guard_s
            ),
            "authority_control_burst": epoch.target_count,
            "authority_control_burst_limit": (
                authority_control_burst_limit
            ),
            "authority_control_burst_gate_open": burst_gate_open,
            "authority_control_burst_limit_exceeded": (
                burst_limit_exceeded
            ),
        }
        selection_plan.append(selected if sidecar_slots > 0 else [])
        selection_metadata.append(metadata)
    selection_selected_total = sum(
        len(selected) for selected in selection_plan
    )
    sidecar_activated = sidecar_slots > 0 and selection_selected_total > 0

    try:
        if sidecar_activated:
            sidecar = ProcessSpeculativeSidecar(
                sidecar_executor,
                max_workers=sidecar_slots,
                max_pending=max_sidecar_pending,
                cpu_affinity=sidecar_affinity,
                claim_grace_s=claim_grace_ms / 1000.0,
            )
            sidecar.start()

        # Every authority timer first rendezvous on this gate. Speculation is
        # handed to the child in one bounded packet before the timed origin;
        # the parent performs no submit work at any epoch release.
        loop = asyncio.get_running_loop()
        origin_future: asyncio.Future[float] = loop.create_future()
        armed: asyncio.Queue[None] = asyncio.Queue()
        for epoch in schedule:
            for window_index, window in enumerate(epoch.windows):
                runtime_session_id = (
                    f"open-loop-r{seed}:e{epoch.index}:"
                    f"{window.session_id}:{window.decision_id}"
                )
                for target_index, target in enumerate(
                    window.executable_targets
                ):
                    target_id = (
                        f"e{epoch.index}:w{window_index}:"
                        f"{window.session_id}:{window.decision_id}:"
                        f"target:{target_index}"
                    )
                    authority_tasks.append(
                        asyncio.create_task(
                            _scheduled_authority_call(
                                authority,
                                Invocation("visit", {"url": target}),
                                session_id=runtime_session_id,
                                target_id=target_id,
                                planned_offset_s=epoch.authority_offset_s,
                                origin_future=origin_future,
                                armed=armed,
                            )
                        )
                    )

            selected = selection_plan[epoch.index]
            if sidecar is not None and selected:
                runtime_ids = {
                    (window.session_id, window.decision_id): (
                        f"open-loop-r{seed}:e{epoch.index}:"
                        f"{window.session_id}:{window.decision_id}"
                    )
                    for window in epoch.windows
                }
                entries = tuple(
                    (
                        Invocation(
                            "visit", {"url": candidate.pattern.url}
                        ),
                        runtime_ids[
                            (
                                candidate.pattern.session_id,
                                candidate.pattern.decision_id,
                            )
                        ],
                        candidate.pattern.decision_id,
                        float(priority),
                        "all-wrong-open-loop",
                    )
                    for candidate, priority in selected
                )
                preload_plan.append(
                    (
                        epoch.speculation_offset_s,
                        epoch.authority_offset_s,
                        entries,
                    )
                )

        armed_expected = len(authority_tasks)
        armed_observed = 0
        for _ in range(armed_expected):
            await armed.get()
            armed_observed += 1
        setup_done_at = time.monotonic()
        origin = setup_done_at + prestart_ms / 1000.0
        if sidecar is not None and preload_plan:
            preload_requested = sum(len(row[2]) for row in preload_plan)
            preload_started_at = time.monotonic()
            nested_handles = sidecar.try_schedule_batches(
                tuple(
                    (
                        origin + speculation_offset_s,
                        origin + authority_offset_s,
                        entries,
                    )
                    for speculation_offset_s, authority_offset_s, entries
                    in preload_plan
                )
            )
            preload_done_at = time.monotonic()
            preload_handles_returned = sum(
                len(handles) for handles in nested_handles
            )
            if preload_handles_returned != preload_requested:
                raise RuntimeError(
                    "process sidecar failed open during pre-origin preload: "
                    f"requested={preload_requested}, "
                    f"returned={preload_handles_returned}"
                )
            if preload_done_at >= origin:
                raise RuntimeError(
                    "pre-origin preload exhausted --prestart-ms; increase "
                    "the untimed setup guard"
                )
        else:
            preload_started_at = setup_done_at
            preload_done_at = setup_done_at
        origin_future.set_result(origin)

        authority_completions = list(await asyncio.gather(*authority_tasks))
        authority_done_at = max(
            completion.terminal_at for completion in authority_completions
        )
        bridge_started_before_authority_done = bool(
            sidecar is not None and sidecar.bridge_started
        )
        await authority.close()
        authority_snapshot = authority.snapshot()
        authority_stats = authority.stats.to_dict()
        if sidecar is not None:
            sidecar_before_close = sidecar.snapshot()
            sidecar.close(wait=True)
            sidecar_after_close = sidecar.snapshot()
        drained_at = time.monotonic()
    finally:
        for task in authority_tasks:
            if not task.done():
                task.cancel()
        if authority_tasks:
            await asyncio.gather(
                *authority_tasks,
                return_exceptions=True,
            )
        # The normal path above already reports close failures.  On an
        # exceptional path, cleanup must not mask the original error; the
        # public wrapper restores authority affinity regardless.
        try:
            await authority.close()
        except Exception:
            pass
        if sidecar is not None:
            try:
                sidecar.close(wait=True)
            except Exception:
                pass

    authority_rows = [
        {
            "target_id": completion.target_id,
            "planned_arrival_offset_s": completion.planned_offset_s,
            "scheduled_latency_ms": (
                completion.terminal_at - completion.scheduled_at
            )
            * 1000.0,
            "first_run_lag_ms": (
                completion.first_run_at - completion.scheduled_at
            )
            * 1000.0,
            "broker_exposed_wait_ms": (
                completion.result.exposed_wait_s * 1000.0
            ),
            "queue_ms": completion.result.queue_s * 1000.0,
            "service_ms": completion.result.service_s * 1000.0,
            "source": completion.result.source,
        }
        for completion in authority_completions
    ]
    authority_rows.sort(key=lambda row: str(row["target_id"]))
    final_sidecar = sidecar_after_close or sidecar_before_close
    sidecar_snapshot = _snapshot_json(final_sidecar)
    sidecar_transport = (
        sidecar_snapshot.get("transport", {})
        if isinstance(sidecar_snapshot.get("transport", {}), Mapping)
        else {}
    )
    sidecar_started = _sidecar_count(
        final_sidecar, "started", "jobs_started", "speculative_started"
    )
    sidecar_max_running = _sidecar_count(
        final_sidecar, "max_running", "max_running_total"
    )
    actual_sidecar_affinity = set(
        sidecar_snapshot.get("actual_cpu_affinity") or []
    )
    actual_bridge_affinity = set(
        sidecar_snapshot.get("actual_bridge_cpu_affinity") or []
    )
    idle_priority_certificate = (
        sidecar_slots == 0
        or not sidecar_activated
        or sidecar_snapshot.get("actual_scheduler_policy")
        == getattr(os, "SCHED_IDLE", 5)
    )
    placement_certificate = (
        sidecar_slots == 0
        or not sidecar_activated
        or (
            authority_affinity is not None
            and sidecar_affinity is not None
            and actual_sidecar_affinity == sidecar_affinity
            and authority_affinity.isdisjoint(sidecar_affinity)
        )
    )
    bridge_placement_certificate = (
        sidecar_slots == 0
        or not sidecar_activated
        or (
            sidecar_affinity is not None
            and bool(sidecar_snapshot.get("bridge_affinity_ready", False))
            and sidecar_snapshot.get("bridge_affinity_error") is None
            and actual_bridge_affinity == sidecar_affinity
        )
    )
    cpu_certificate = not cpu_isolation or (
        placement_certificate and bridge_placement_certificate
    )
    target_count = len(authority_rows)
    planned_offsets = [
        float(row["planned_arrival_offset_s"]) for row in authority_rows
    ]
    latency_values = [
        float(row["scheduled_latency_ms"]) for row in authority_rows
    ]
    first_run_values = [
        float(row["first_run_lag_ms"]) for row in authority_rows
    ]
    broker_wait_values = [
        float(row["broker_exposed_wait_ms"]) for row in authority_rows
    ]
    first_arrival_at = origin + min(planned_offsets)
    last_arrival_at = origin + max(planned_offsets)
    authority_trace_makespan_s = authority_done_at - first_arrival_at
    requested_predictions = preload_requested
    handles_returned = preload_handles_returned
    preload_elapsed_s = max(0.0, preload_done_at - preload_started_at)
    planned_by_id = {
        str(row["target_id"]): float(row["authority_offset_s"])
        for row in trace["authority_arrivals"]
    }
    observed_plan_by_id = {
        str(row["target_id"]): float(row["planned_arrival_offset_s"])
        for row in authority_rows
    }
    safety = {
        "all_timer_tasks_armed_before_origin": (
            armed_observed == armed_expected
            and setup_done_at <= origin
            and origin_future.done()
        ),
        "preload_completed_before_origin": (
            sidecar_slots == 0 or preload_done_at < origin
        ),
        "timed_parent_admission_is_zero": (
            int(sidecar_transport.get("transport_submit_packets", 0)) == 0
        ),
        "preload_uses_one_schedule_packet": (
            sidecar_slots == 0
            or (
                preload_requested == 0
                and int(
                    sidecar_transport.get("transport_schedule_packets", 0)
                )
                == 0
            )
            or (
                preload_requested > 0
                and int(
                    sidecar_transport.get("transport_schedule_packets", 0)
                )
                == 1
            )
        ),
        "preload_handoff_is_complete": (
            sidecar_slots == 0
            or (
                preload_requested == 0
                and preload_handles_returned == 0
                and int(sidecar_transport.get("transport_scheduled", 0))
                == 0
            )
            or (
                preload_requested > 0
                and preload_handles_returned == preload_requested
                and int(
                    sidecar_transport.get("transport_scheduled", 0)
                )
                == preload_requested
            )
        ),
        "fixed_arrival_trace_matches_manifest": (
            observed_plan_by_id.keys() == planned_by_id.keys()
            and all(
                math.isclose(
                    observed_plan_by_id[target_id],
                    planned_by_id[target_id],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for target_id in observed_plan_by_id
            )
        ),
        "authority_attempts_equal_targets": (
            int(authority_stats["authoritative_requests"]) == target_count
        ),
        "authority_commits_equal_targets": (
            int(authority_stats["commits"]) == target_count
        ),
        "authority_state_equal_targets": (
            len(authority.authoritative_state) == target_count
        ),
        "authority_jobs_drained": len(authority_snapshot["jobs"]) == 0,
        "authority_worker_cap": (
            int(authority_stats["max_running_authoritative"]) <= workers
        ),
        "authority_tool_cap": (
            int(
                authority_stats["max_running_authoritative_by_tool"].get(
                    "visit", 0
                )
            )
            <= visit_capacity
        ),
        "all_authority_sources_executed": all(
            row["source"] == "executed" for row in authority_rows
        ),
        "sidecar_cap": sidecar_max_running <= sidecar_slots,
        "treatment_exercises_sidecar": (
            sidecar_slots == 0
            or requested_predictions == 0
            or (
                requested_predictions > 0
                and handles_returned > 0
                and sidecar_started > 0
            )
        ),
        "cpu_isolation_certificate": cpu_certificate,
        "result_bridge_cpu_affinity_certificate": (
            not cpu_isolation or bridge_placement_certificate
        ),
        "sidecar_idle_priority_certificate": idle_priority_certificate,
        "all_wrong_has_no_claim_packets": (
            int(sidecar_transport.get("transport_claims", 0)) == 0
        ),
        "all_wrong_has_no_result_packets": (
            int(sidecar_transport.get("transport_results", 0)) == 0
        ),
        "all_wrong_has_no_terminal_packets": (
            int(sidecar_transport.get("transport_terminal", 0)) == 0
        ),
        "lease_cleanup_has_no_tombstone_packets": (
            int(
                sidecar_transport.get("transport_tombstone_packets", 0)
            )
            == 0
        ),
        "bridge_absent_until_authority_done": (
            not bridge_started_before_authority_done
        ),
        "speculation_release_precedes_authority": all(
            not selected
            or epoch.speculation_offset_s < epoch.authority_offset_s
            for epoch, selected in zip(schedule, selection_plan)
        ),
        "first_epoch_has_no_artificial_phase_guard": math.isclose(
            schedule[0].speculation_phase_guard_s,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "authority_control_burst_gate_enforced": all(
            not selected
            or bool(metadata["authority_control_burst_gate_open"])
            for selected, metadata in zip(
                selection_plan, selection_metadata
            )
        ),
    }
    if not all(safety.values()):
        raise RuntimeError(f"open-loop safety failed: {safety}")

    return {
        "seed": seed,
        "task_concurrency": task_concurrency,
        "sidecar_slots": sidecar_slots,
        "sidecar_activated": sidecar_activated,
        "arrival_trace_sha256": trace["sha256"],
        "arrival_epoch_rows": trace["epochs"],
        "arrival_trace_rows": trace["authority_arrivals"],
        "arrival_epochs": len(schedule),
        "authoritative_targets": target_count,
        "timer_tasks_armed": armed_expected,
        "timer_tasks_armed_observed": armed_observed,
        "timer_setup_lead_ms": (origin - setup_done_at) * 1000.0,
        "planned_first_authority_offset_s": min(planned_offsets),
        "planned_last_authority_offset_s": max(planned_offsets),
        "authority_trace_makespan_s": authority_trace_makespan_s,
        "authority_trace_makespan_including_preload_s": (
            authority_trace_makespan_s + preload_elapsed_s
        ),
        "authority_drain_tail_s": authority_done_at - last_arrival_at,
        "authority_done_from_origin_s": authority_done_at - origin,
        "drained_from_origin_s": drained_at - origin,
        "authority_scheduled_latency_ms": _summary(latency_values),
        "authority_first_run_lag_ms": _summary(first_run_values),
        "authority_broker_exposed_wait_ms": _summary(broker_wait_values),
        "authority_rows": authority_rows,
        "requested_predictions": requested_predictions,
        "handles_returned": handles_returned,
        "preload_requested": preload_requested,
        "preload_handles_returned": preload_handles_returned,
        "preload_elapsed_ms": preload_elapsed_s * 1000.0,
        "preload_done_before_origin": preload_done_at < origin,
        "timed_parent_admission_calls": 0,
        "timed_parent_submit_packets": int(
            sidecar_transport.get("transport_submit_packets", 0)
        ),
        "selection_selected": selection_selected_total,
        "selection_compute_ms": sum(
            float(row["compute_ms"]) for row in selection_metadata
        ),
        "sidecar_started": sidecar_started,
        "physical_call_amplification": ratio(
            target_count + sidecar_started, target_count
        ),
        "admission_release_lag_ms": _summary([]),
        "admission_elapsed_ms": _summary([preload_elapsed_s * 1000.0]),
        "admission_deadline_overruns": int(preload_done_at >= origin),
        "authority_cpu_affinity": sorted(authority_affinity or []),
        "sidecar_cpu_affinity": sorted(sidecar_affinity or []),
        "authority_control_burst_limit": authority_control_burst_limit,
        "authority_control_burst_gated_epochs": sum(
            epoch.target_count > authority_control_burst_limit
            for epoch in schedule
        ),
        "authority_control_burst_latch_closed_epochs": sum(
            not bool(row["authority_control_burst_gate_open"])
            for row in selection_metadata
        ),
        "cpu_isolation_certified": cpu_certificate,
        "sidecar_idle_priority_certified": idle_priority_certificate,
        "bridge_started_before_authority_done": (
            bridge_started_before_authority_done
        ),
        "authority_stats": authority_stats,
        "sidecar_snapshot": sidecar_snapshot,
        "sample_configuration": {
            "task_concurrency": task_concurrency,
            "workers": workers,
            "visit_capacity": visit_capacity,
            "service_ms": service_ms,
            "lead_ms": lead_ms,
            "speculation_phase_guard_ms": max(
                (
                    epoch.speculation_phase_guard_s * 1000.0
                    for epoch in schedule
                ),
                default=0.0,
            ),
            "max_sidecar_pending": max_sidecar_pending,
            "probability_threshold": probability_threshold,
            "claim_grace_ms": claim_grace_ms,
            "prestart_ms": prestart_ms,
            "cpu_isolation": cpu_isolation,
            "authority_control_burst_limit": authority_control_burst_limit,
        },
        "safety": safety,
    }


def _target_map(sample: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = list(sample["authority_rows"])
    result = {
        str(row["target_id"]): row for row in sample["authority_rows"]
    }
    if len(result) != len(rows):
        raise RuntimeError("duplicate authority target identifier")
    return result


def _compact_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sample.items()
        if key
        not in {"authority_rows", "arrival_epoch_rows", "arrival_trace_rows"}
    }


def aggregate_paired_samples(
    *,
    task_concurrency: int,
    baseline_samples: Sequence[Mapping[str, Any]],
    treatment_samples: Sequence[Mapping[str, Any]],
    counterbalance_orders: Sequence[str],
    latency_margin_ms: float = 0.10,
    wall_margin_fraction: float = 0.001,
) -> dict[str, Any]:
    """Aggregate only repeat-level paired statistics and retain raw vectors."""

    if not baseline_samples or len(baseline_samples) != len(treatment_samples):
        raise ValueError("paired samples must have equal positive length")
    if len(counterbalance_orders) != len(baseline_samples):
        raise ValueError("counterbalance order count differs from repeats")

    baseline_latency: list[float] = []
    treatment_latency: list[float] = []
    latency_regressions: list[float] = []
    baseline_first_run: list[float] = []
    treatment_first_run: list[float] = []
    first_run_regressions: list[float] = []
    baseline_makespans: list[float] = []
    treatment_makespans: list[float] = []
    wall_log_ratios: list[float] = []
    repeat_records: list[dict[str, Any]] = []

    for repeat, (baseline, treatment, order) in enumerate(
        zip(baseline_samples, treatment_samples, counterbalance_orders)
    ):
        if order not in {"AB", "BA"}:
            raise RuntimeError("paired order must be AB or BA")
        if baseline["seed"] != treatment["seed"]:
            raise RuntimeError("paired samples used different repeat seeds")
        if baseline["sample_configuration"] != treatment[
            "sample_configuration"
        ]:
            raise RuntimeError("paired samples used different configurations")
        for sample in (baseline, treatment):
            recomputed_trace_sha256 = canonical_sha256(
                {
                    "epochs": sample["arrival_epoch_rows"],
                    "authority_arrivals": sample["arrival_trace_rows"],
                }
            )
            if recomputed_trace_sha256 != sample["arrival_trace_sha256"]:
                raise RuntimeError("arrival trace digest is not reproducible")
        if baseline["arrival_trace_sha256"] != treatment["arrival_trace_sha256"]:
            raise RuntimeError("paired samples used different arrival traces")
        if baseline["authority_cpu_affinity"] != treatment[
            "authority_cpu_affinity"
        ]:
            raise RuntimeError("paired samples used different authority CPUs")
        baseline_by_id = _target_map(baseline)
        treatment_by_id = _target_map(treatment)
        if baseline_by_id.keys() != treatment_by_id.keys():
            raise RuntimeError("paired authority target identifiers differ")
        if any(
            not math.isclose(
                float(baseline_by_id[key]["planned_arrival_offset_s"]),
                float(treatment_by_id[key]["planned_arrival_offset_s"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in baseline_by_id
        ):
            raise RuntimeError("paired planned authority arrivals differ")

        base_latency = statistics.fmean(
            float(row["scheduled_latency_ms"])
            for row in baseline_by_id.values()
        )
        treat_latency = statistics.fmean(
            float(row["scheduled_latency_ms"])
            for row in treatment_by_id.values()
        )
        base_first_run = statistics.fmean(
            float(row["first_run_lag_ms"])
            for row in baseline_by_id.values()
        )
        treat_first_run = statistics.fmean(
            float(row["first_run_lag_ms"])
            for row in treatment_by_id.values()
        )
        # Charge the one-time parent preload cost to treatment wall time even
        # though it is intentionally moved before the fixed-arrival origin.
        base_wall = float(
            baseline["authority_trace_makespan_including_preload_s"]
        )
        treat_wall = float(
            treatment["authority_trace_makespan_including_preload_s"]
        )
        if base_wall <= 0.0 or treat_wall <= 0.0:
            raise RuntimeError("authority trace makespan must be positive")

        baseline_latency.append(base_latency)
        treatment_latency.append(treat_latency)
        latency_regressions.append(treat_latency - base_latency)
        baseline_first_run.append(base_first_run)
        treatment_first_run.append(treat_first_run)
        first_run_regressions.append(treat_first_run - base_first_run)
        baseline_makespans.append(base_wall)
        treatment_makespans.append(treat_wall)
        wall_log_ratios.append(math.log(treat_wall / base_wall))
        repeat_records.append(
            {
                "repeat": repeat,
                "seed": int(baseline["seed"]),
                "order": order,
                "arrival_trace_sha256": baseline["arrival_trace_sha256"],
                "arrival_trace": {
                    "epochs": baseline["arrival_epoch_rows"],
                    "authority_arrivals": baseline["arrival_trace_rows"],
                },
                "authoritative_targets": len(baseline_by_id),
                "baseline_authority_rows": list(
                    baseline["authority_rows"]
                ),
                "treatment_authority_rows": list(
                    treatment["authority_rows"]
                ),
                "baseline_mean_authority_scheduled_latency_ms": base_latency,
                "treatment_mean_authority_scheduled_latency_ms": treat_latency,
                "authority_latency_regression_ms_per_target": (
                    treat_latency - base_latency
                ),
                "baseline_mean_authority_first_run_lag_ms": base_first_run,
                "treatment_mean_authority_first_run_lag_ms": treat_first_run,
                "authority_first_run_lag_regression_ms_per_target": (
                    treat_first_run - base_first_run
                ),
                "baseline_authority_trace_makespan_s": base_wall,
                "treatment_authority_trace_makespan_s": treat_wall,
                "authority_trace_makespan_log_ratio": wall_log_ratios[-1],
                "treatment_timed_parent_submit_packets": int(
                    treatment["timed_parent_submit_packets"]
                ),
            }
        )

    latency_inference = _paired_repeat_inference(
        latency_regressions,
        margin=latency_margin_ms,
    )
    latency_inference["scale"] = "ms/target"
    first_run_inference = _paired_repeat_inference(
        first_run_regressions,
        margin=latency_margin_ms,
    )
    first_run_inference["scale"] = "ms/target"
    raw_wall_inference = _paired_repeat_inference(
        wall_log_ratios,
        margin=math.log1p(wall_margin_fraction),
    )
    wall_inference = {
        **raw_wall_inference,
        "scale": "log(treatment_makespan / baseline_makespan)",
        "geometric_mean_regression_fraction": math.expm1(
            float(raw_wall_inference["mean"])
        ),
        "ci90_regression_fraction": [
            math.expm1(float(value))
            for value in raw_wall_inference["ci90"]
        ],
        "margin_fraction": wall_margin_fraction,
    }
    decisions = {
        str(latency_inference["decision"]),
        str(wall_inference["decision"]),
    }
    if "insufficient_repetitions" in decisions:
        overall = "insufficient_repetitions"
    elif "regression" in decisions:
        overall = "regression"
    elif decisions == {"pass"}:
        overall = "pass"
    else:
        overall = "inconclusive"

    raw_vectors = {
        "baseline_mean_authority_scheduled_latency_ms_per_target": (
            baseline_latency
        ),
        "treatment_mean_authority_scheduled_latency_ms_per_target": (
            treatment_latency
        ),
        "authority_latency_regression_ms_per_target": latency_regressions,
        "baseline_mean_authority_first_run_lag_ms_per_target": (
            baseline_first_run
        ),
        "treatment_mean_authority_first_run_lag_ms_per_target": (
            treatment_first_run
        ),
        "authority_first_run_lag_regression_ms_per_target": (
            first_run_regressions
        ),
        "baseline_authority_trace_makespan_s": baseline_makespans,
        "treatment_authority_trace_makespan_s": treatment_makespans,
        "authority_trace_makespan_log_ratio": wall_log_ratios,
    }
    return {
        "scenario": "all_wrong_fixed_arrival_open_loop",
        "task_concurrency": task_concurrency,
        "repetitions": len(baseline_samples),
        "baseline_sidecar_slots": 0,
        "treatment_sidecar_slots": int(treatment_samples[0]["sidecar_slots"]),
        "counterbalance_orders": list(counterbalance_orders),
        "authoritative_targets_per_repeat": int(
            baseline_samples[0]["authoritative_targets"]
        ),
        "mean_authority_latency_regression_ms_per_target": statistics.fmean(
            latency_regressions
        ),
        "mean_authority_first_run_lag_regression_ms_per_target": (
            statistics.fmean(first_run_regressions)
        ),
        "authority_latency_inference": latency_inference,
        "authority_first_run_lag_inference": first_run_inference,
        "authority_trace_makespan_inference": wall_inference,
        "overall_no_interference_decision": overall,
        "all_safety_invariants_passed": all(
            all(bool(value) for value in sample["safety"].values())
            for sample in (*baseline_samples, *treatment_samples)
        ),
        "all_timed_parent_submit_packets_zero": all(
            int(sample["timed_parent_submit_packets"]) == 0
            for sample in treatment_samples
        ),
        "raw_repeat_vectors": raw_vectors,
        "repeat_records": repeat_records,
        "samples": {
            "baseline": [_compact_sample(row) for row in baseline_samples],
            "treatment": [_compact_sample(row) for row in treatment_samples],
        },
    }


async def run_matrix(
    windows: Sequence[ScoredWindow],
    *,
    concurrencies: Sequence[int],
    repetitions: int,
    workers: int,
    visit_capacity: int,
    service_ms: float,
    lead_ms: float,
    speculation_phase_guard_ms: float,
    sidecar_slots: int,
    max_sidecar_pending: int,
    probability_threshold: float,
    claim_grace_ms: float,
    prestart_ms: float,
    cpu_isolation: bool,
    authority_control_burst_limit: int,
) -> list[dict[str, Any]]:
    wrong_windows = force_all_wrong(windows)
    rows: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        print(
            f"running fixed-arrival all-wrong C={concurrency} "
            f"paired K=0/K={sidecar_slots}",
            flush=True,
        )
        baseline_samples: list[dict[str, Any]] = []
        treatment_samples: list[dict[str, Any]] = []
        orders: list[str] = []
        for repetition in range(repetitions):
            schedule = build_fixed_arrival_trace(
                wrong_windows,
                task_concurrency=concurrency,
                seed=repetition,
                visit_capacity=visit_capacity,
                service_s=service_ms / 1000.0,
                lead_s=lead_ms / 1000.0,
                speculation_phase_guard_s=(
                    speculation_phase_guard_ms / 1000.0
                ),
            )

            async def run_one(k: int) -> dict[str, Any]:
                return await run_fixed_arrival_sample(
                    schedule,
                    task_concurrency=concurrency,
                    seed=repetition,
                    workers=workers,
                    visit_capacity=visit_capacity,
                    service_ms=service_ms,
                    lead_ms=lead_ms,
                    sidecar_slots=k,
                    max_sidecar_pending=max_sidecar_pending,
                    probability_threshold=probability_threshold,
                    claim_grace_ms=claim_grace_ms,
                    prestart_ms=prestart_ms,
                    cpu_isolation=cpu_isolation,
                    authority_control_burst_limit=(
                        authority_control_burst_limit
                    ),
                )

            if repetition % 2 == 0:
                baseline = await run_one(0)
                treatment = await run_one(sidecar_slots)
                orders.append("AB")
            else:
                treatment = await run_one(sidecar_slots)
                baseline = await run_one(0)
                orders.append("BA")
            baseline_samples.append(baseline)
            treatment_samples.append(treatment)
        rows.append(
            aggregate_paired_samples(
                task_concurrency=concurrency,
                baseline_samples=baseline_samples,
                treatment_samples=treatment_samples,
                counterbalance_orders=orders,
            )
        )
    return rows


def raw_repeat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema": RAW_SCHEMA,
        "configuration": payload["configuration"],
        "source_sha256": payload["source_sha256"],
        "input_sha256": payload.get("input_sha256", {}),
        "cells": [
            {
                "scenario": row["scenario"],
                "task_concurrency": row["task_concurrency"],
                "counterbalance_orders": row["counterbalance_orders"],
                "raw_repeat_vectors": row["raw_repeat_vectors"],
                "repeat_records": row["repeat_records"],
            }
            for row in payload["cells"]
        ],
    }
    value["sha256_excluding_self"] = canonical_sha256(value)
    return value


def render_report(payload: Mapping[str, Any]) -> str:
    sidecar_slots = int(
        payload["configuration"].get("treatment_sidecar_slots", 4)
    )
    treatment_samples = [
        sample
        for cell in payload.get("cells", ())
        for sample in cell.get("samples", {}).get("treatment", ())
    ]
    any_sidecar_activated = any(
        bool(sample.get("sidecar_activated", False))
        for sample in treatment_samples
    )
    any_sidecar_cpu_reserved = any(
        bool(sample.get("sidecar_cpu_affinity", ()))
        for sample in treatment_samples
    )
    if any_sidecar_activated:
        treatment_protocol = (
            "- Treatment requests "
            f"K={sidecar_slots}. When admitted it uses a forked process "
            "sidecar, finite leases, a lazy result bridge, topology-aware "
            "dedicated CPU placement, and SCHED_IDLE. All future batches are "
            "handed off in one bounded packet before the timed origin; "
            "`timed_parent_submit_packets=0` is enforced. No exact claims, "
            "result packets, terminal packets, or parent tombstone packets "
            "are permitted before the safety gate."
        )
    else:
        cpu_clause = (
            "A sidecar CPU was reserved before selection, but "
            if any_sidecar_cpu_reserved
            else "No sidecar CPU was reserved and "
        )
        treatment_protocol = (
            "- Treatment requests "
            f"K={sidecar_slots}, but the resource gate abstained in every "
            f"sample. {cpu_clause}no sidecar process, preload, bridge, or IPC "
            "was created; selected=started=0, "
            "`timed_parent_submit_packets=0`, and physical call amplification "
            "is exactly 1.0."
        )
    lines = [
        "# Process-sidecar fixed-arrival no-interference supplement",
        "",
        "Each paired repeat replays the same precomputed absolute authority "
        "arrival deadlines against K=0 and process-sidecar treatment. The "
        "scenario is forced all-wrong, so any treatment difference is "
        "interference rather than speculative benefit.",
        "",
        "| C | R | Authority regression ms/target | Latency 90% CI | "
        "Makespan regression | Makespan 90% CI | Decision | Safety |",
        "|---:|---:|---:|:---:|---:|:---:|:---:|:---:|",
    ]
    for row in payload["cells"]:
        latency = row["authority_latency_inference"]
        wall = row["authority_trace_makespan_inference"]
        wall_ci = wall["ci90_regression_fraction"]
        lines.append(
            "| {c} | {r} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            "{wall_mean:+.3%} | [{wall_lo:+.3%}, {wall_hi:+.3%}] | "
            "{decision} | {safety} |".format(
                c=row["task_concurrency"],
                r=row["repetitions"],
                mean=row[
                    "mean_authority_latency_regression_ms_per_target"
                ],
                lo=latency["ci90"][0],
                hi=latency["ci90"][1],
                wall_mean=wall["geometric_mean_regression_fraction"],
                wall_lo=wall_ci[0],
                wall_hi=wall_ci[1],
                decision=row["overall_no_interference_decision"],
                safety=(
                    "pass" if row["all_safety_invariants_passed"] else "FAIL"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- C uses the existing source-session batching definition. The "
            "batch cadence is frozen from modeled lead/service waves before "
            "either paired replay; observed completions never schedule future "
            "arrivals.",
            "- Authority scheduled latency includes timer-release lateness, "
            "broker queueing, and service. Repeat—not target—is the inference "
            "unit; AB/BA order is counterbalanced.",
            treatment_protocol,
            "- A configured speculation phase guard delays only epoch 2+ "
            "sidecar releases beyond the preceding modeled authority "
            "completion boundary. Authority arrivals remain unchanged, and "
            "admission uses the resulting shorter effective lead.",
            "- The authority-control burst gate makes the certified start "
            "budget zero for an epoch whose synchronized authority arrivals "
            "exceed the configured host-calibrated limit. A zero limit means "
            "that no positive resource certificate was supplied. This protects the "
            "single authority event loop even when tool slots are plentiful.",
            "- Makespan inference charges the measured one-time parent preload "
            "cost even though that work is outside the fixed-arrival origin.",
            "- The no-interference margins match the main runner: 0.10 "
            "ms/target and 0.1% trace makespan, with one-sided 95% bounds and "
            "at least eight paired repeats.",
            "",
            "## Scope",
            "",
            "This isolates modeled executor capacity, Python GIL, and logical "
            "CPU placement for lightweight synthetic tools. It does not "
            "certify physical-core/LLC/NUMA isolation or independent network, "
            "connection-pool, and remote-service quotas. A statistically "
            "inconclusive result is not evidence of equivalence.",
            "",
            "Raw repeat vectors are stored in `raw_repeat_vectors.json`; "
            "configuration and source hashes are stored in both JSON outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[1, 16, 64]
    )
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visit-capacity", type=int, default=2)
    parser.add_argument("--service-ms", type=float, default=20.0)
    parser.add_argument("--lead-ms", type=float, default=10.0)
    parser.add_argument(
        "--speculation-phase-guard-ms", type=float, default=0.0
    )
    parser.add_argument("--sidecar-slots", type=int, default=4)
    parser.add_argument("--max-sidecar-pending", type=int, default=8)
    parser.add_argument("--probability-threshold", type=float, default=0.20)
    parser.add_argument("--claim-grace-ms", type=float, default=10.0)
    parser.add_argument("--prestart-ms", type=float, default=50.0)
    parser.add_argument(
        "--authority-control-burst-limit", type=int, default=0
    )
    parser.add_argument(
        "--no-cpu-isolation",
        action="store_false",
        dest="cpu_isolation",
    )
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if any(value <= 0 for value in args.concurrencies):
        parser.error("--concurrencies must be positive")
    if args.workers <= 0 or args.visit_capacity <= 0:
        parser.error("authority capacities must be positive")
    if args.visit_capacity > args.workers:
        parser.error("--visit-capacity cannot exceed --workers")
    if args.sidecar_slots <= 0 or args.max_sidecar_pending < args.sidecar_slots:
        parser.error("sidecar pending capacity must be at least its slots")
    if args.authority_control_burst_limit < 0:
        parser.error("--authority-control-burst-limit must be non-negative")
    if args.service_ms <= 0.0 or args.lead_ms < 0.0:
        parser.error("service must be positive and lead non-negative")
    if args.speculation_phase_guard_ms < 0.0:
        parser.error("--speculation-phase-guard-ms must be non-negative")
    if (
        args.speculation_phase_guard_ms > 0.0
        and args.speculation_phase_guard_ms >= args.lead_ms
    ):
        parser.error(
            "--speculation-phase-guard-ms must be smaller than --lead-ms"
        )
    if args.claim_grace_ms < 0.0 or args.prestart_ms <= 0.0:
        parser.error("grace must be non-negative and prestart positive")
    if not 0.0 <= args.probability_threshold <= 1.0:
        parser.error("--probability-threshold must be in [0, 1]")
    return args


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    # Capture the executed code/input identity before a multi-minute matrix;
    # an editor touching a source during the run must not rewrite provenance.
    source_sha256 = {
        "runner": sha256_file(SCRIPT),
        "broker": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "live_broker.py"
        ),
        "sidecar": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "speculation_sidecar.py"
        ),
        "invocation": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "invocation.py"
        ),
        "policy": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "speculation_policy.py"
        ),
        "adaptive_trace_builder": sha256_file(
            SCRIPT.parent / "run_pattern_v2_adaptive_load.py"
        ),
        "metric_helpers": sha256_file(
            SCRIPT.parent / "run_pattern_v2_load_robustness.py"
        ),
        "repeat_inference": sha256_file(
            SCRIPT.parent / "run_pattern_v2_sidecar_load.py"
        ),
    }
    trace_files = {
        path.name: sha256_file(path)
        for path in sorted(args.traces.glob("*.jsonl"))
    }
    input_sha256 = {
        "trace_files": trace_files,
        "trace_manifest": canonical_sha256(trace_files),
    }
    windows, nested_oof = collect_nested_oof_windows(args.traces)
    cells = await run_matrix(
        windows,
        concurrencies=args.concurrencies,
        repetitions=args.repetitions,
        workers=args.workers,
        visit_capacity=args.visit_capacity,
        service_ms=args.service_ms,
        lead_ms=args.lead_ms,
        speculation_phase_guard_ms=args.speculation_phase_guard_ms,
        sidecar_slots=args.sidecar_slots,
        max_sidecar_pending=args.max_sidecar_pending,
        probability_threshold=args.probability_threshold,
        claim_grace_ms=args.claim_grace_ms,
        prestart_ms=args.prestart_ms,
        cpu_isolation=args.cpu_isolation,
        authority_control_burst_limit=args.authority_control_burst_limit,
    )
    configuration = {
        "traces": str(args.traces.resolve()),
        "scenario": "all_wrong_counterfactual_only",
        "concurrencies": list(args.concurrencies),
        "repetitions": args.repetitions,
        "workers": args.workers,
        "visit_capacity": args.visit_capacity,
        "service_ms": args.service_ms,
        "lead_ms": args.lead_ms,
        "speculation_phase_guard_ms": args.speculation_phase_guard_ms,
        "baseline_sidecar_slots": 0,
        "treatment_sidecar_slots": args.sidecar_slots,
        "max_sidecar_pending": args.max_sidecar_pending,
        "probability_threshold": args.probability_threshold,
        "claim_grace_ms": args.claim_grace_ms,
        "prestart_ms": args.prestart_ms,
        "cpu_isolation": args.cpu_isolation,
        "authority_control_burst_limit": (
            args.authority_control_burst_limit
        ),
        "arrival_model": (
            "fixed absolute monotonic authority deadlines; ideal source-"
            "session cadence frozen before each paired replay"
        ),
        "authority_mode": "always-executed demand-only",
        "sidecar_backend": "linux-fork process",
        "paired_execution_order": "AB/BA counterbalanced by repetition",
        "inference_unit": "paired_repeat",
        "latency_margin_ms_per_target": 0.10,
        "makespan_margin_fraction": 0.001,
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "supplemental_open_loop_no_interference",
        "command": shlex.join([sys.executable, *sys.argv]),
        "configuration": configuration,
        "nested_oof": nested_oof,
        "calibration_quality": calibration_quality(windows),
        "source_sha256": source_sha256,
        "input_sha256": input_sha256,
        "cells": cells,
    }
    raw = raw_repeat_payload(payload)
    payload["raw_repeat_vectors_sha256"] = raw["sha256_excluding_self"]
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_repeat_payload(payload)
    expected = payload.get("raw_repeat_vectors_sha256")
    if expected is not None and raw["sha256_excluding_self"] != expected:
        raise RuntimeError("raw repeat-vector digest changed before write")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "raw_repeat_vectors.json").write_text(
        json.dumps(
            raw,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = asyncio.run(async_main(args))
    write_outputs(args.output_dir, payload)
    print(f"wrote {args.output_dir.resolve()}")
    print(f"payload_sha256={payload['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
