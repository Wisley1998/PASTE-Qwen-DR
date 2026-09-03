#!/usr/bin/env python3
"""Replay all-visit speculation in a shared, preemptible visit pool.

This is the resource-tight companion to ``run_pattern_v2_trace_all_visit_wall``.
It preserves the 0.42x trace timeline, serializes URLs within one authoritative
visit, and shares a bounded visit pool across concurrent sessions.  Speculative
work is preemptible: an arriving authority request promotes an exact in-flight
job, otherwise it cancels the lowest-value running speculation until authority
can dispatch.  Consequently speculation never queues authority behind wrong
speculation; authority can still queue behind other authority work.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.traces import LLMCall, SessionTrace, ToolCall, load_sessions  # noqa: E402
from run_pattern_v2_trace_all_visit_wall import (  # noqa: E402
    AllVisitDecision,
    apply_cross_fold_start_budget,
    build_session_global_cache_replays,
    collect_all_visit_timings,
    collect_nested_oof_all_visit_windows,
    trace_llm_scale_metadata,
)
from run_pattern_v2_trace_multi_spec_wall import (  # noqa: E402
    candidate_value,
    select_per_task_candidates,
    session_full_walls,
)
from run_pattern_v2_trace_timing_net_benefit import (  # noqa: E402
    DecisionTiming,
    ServiceEstimate,
    build_oof_service_estimates,
    sha256_file,
)
from run_pattern_v2_adaptive_load import ScoredCandidate, ScoredWindow  # noqa: E402


SCHEMA = "paste_repro.pattern_v2_trace_all_visit_shared_capacity.v6"
DEFAULT_TRACES = (
    REPOSITORY_ROOT
    / "traces"
    / "my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42"
)
DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT
    / "results"
    / "pattern_v2_trace_all_visit_shared_capacity_preemptible"
)


def stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"shared-capacity-v1\0{seed}\0{value}".encode()).hexdigest()


def speculative_service_s(session_id: str, decision_id: str, url: str) -> float:
    """Stable 2--8 second service sample for an arbitrary speculative URL."""

    digest = hashlib.sha256(
        f"shared-spec-service-v1\0{session_id}\0{decision_id}\0{url}".encode()
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return 2.0 + 6.0 * unit


@dataclass(frozen=True)
class Policy:
    name: str
    candidate_policy: str
    scheduler: str


@dataclass(frozen=True)
class Epoch:
    decision_id: str
    original_start_s: float
    authority_offset_s: float | None
    baseline_authority_done_s: float | None
    targets: tuple[str, ...]
    services_s: tuple[float, ...]
    candidates: tuple[ScoredCandidate, ...]
    candidate_services_s: tuple[float, ...]


@dataclass(frozen=True)
class PreparedSession:
    session_id: str
    full_wall_s: float
    epochs: tuple[Epoch, ...]


@dataclass
class SessionState:
    prepared: PreparedSession
    start_s: float
    epoch_index: int = 0
    shift_s: float = 0.0


@dataclass
class Job:
    job_id: int
    session_id: str
    url: str
    duration_s: float
    score: float
    speculative: bool
    origin_decision_id: str | None
    state: str = "queued"
    started_s: float | None = None
    completion_s: float | None = None
    generation: int = 0
    authority_requested_s: float | None = None
    callbacks: list[Callable[[float], None]] | None = None
    executed_speculative_s: float = 0.0
    ever_cache_hit: bool = False

    def __post_init__(self) -> None:
        if self.callbacks is None:
            self.callbacks = []


class EventLoop:
    def __init__(self) -> None:
        self.now_s = 0.0
        self._sequence = 0
        self._events: list[tuple[float, int, int, Callable[[], None]]] = []

    def schedule(self, when_s: float, priority: int, callback: Callable[[], None]) -> None:
        self._sequence += 1
        heapq.heappush(
            self._events,
            (max(self.now_s, when_s), priority, self._sequence, callback),
        )

    def run(self) -> None:
        while self._events:
            when_s, _, _, callback = heapq.heappop(self._events)
            self.now_s = when_s
            callback()


class PreemptibleVisitPool:
    """Authority-first pool with instantaneous speculative cancellation."""

    def __init__(self, loop: EventLoop, *, capacity: int, speculative_cap: int) -> None:
        self.loop = loop
        self.capacity = capacity
        self.speculative_cap = speculative_cap
        self._next_job_id = 0
        self.running: dict[int, Job] = {}
        self.authority_queue: deque[Job] = deque()
        self.spec_queue: list[tuple[float, int, Job]] = []
        self.cache: dict[tuple[str, str], Job] = {}
        self.jobs: list[Job] = []
        self.metrics: Counter[str] = Counter()
        self.authority_queue_wait_s = 0.0
        self.authority_exposed_s = 0.0
        self.speculative_resource_s = 0.0
        self.preempted_speculative_s = 0.0

    def _job(
        self,
        *,
        session_id: str,
        url: str,
        duration_s: float,
        score: float,
        speculative: bool,
        origin_decision_id: str | None,
    ) -> Job:
        self._next_job_id += 1
        job = Job(
            job_id=self._next_job_id,
            session_id=session_id,
            url=url,
            duration_s=duration_s,
            score=score,
            speculative=speculative,
            origin_decision_id=origin_decision_id,
        )
        self.jobs.append(job)
        return job

    def submit_speculation(
        self,
        *,
        session_id: str,
        decision_id: str,
        candidate: ScoredCandidate,
        duration_s: float,
    ) -> bool:
        key = (session_id, candidate.pattern.url)
        self.metrics["policy_candidates"] += 1
        if key in self.cache:
            self.metrics["cache_deduplicated_candidates"] += 1
            return False
        score = candidate.exact_probability
        job = self._job(
            session_id=session_id,
            url=candidate.pattern.url,
            duration_s=duration_s,
            score=score,
            speculative=True,
            origin_decision_id=decision_id,
        )
        self.cache[key] = job
        heapq.heappush(self.spec_queue, (-score, job.job_id, job))
        self.metrics["speculative_admitted"] += 1
        self._dispatch()
        return True

    def request_authority(
        self,
        *,
        session_id: str,
        url: str,
        duration_s: float,
        on_complete: Callable[[float], None],
    ) -> None:
        self.metrics["authority_requests"] += 1
        requested_s = self.loop.now_s

        def record_completion(done_s: float) -> None:
            self.authority_exposed_s += max(0.0, done_s - requested_s)
            on_complete(done_s)

        key = (session_id, url)
        cached = self.cache.get(key)
        if cached is not None and cached.state == "completed":
            cached.ever_cache_hit = True
            self.metrics["cache_hits"] += 1
            self.metrics["ready_cache_hits"] += 1
            self.loop.schedule(
                self.loop.now_s,
                0,
                lambda: record_completion(self.loop.now_s),
            )
            return
        if cached is not None and cached.state == "running":
            cached.ever_cache_hit = True
            self.metrics["cache_hits"] += 1
            self.metrics["inflight_cache_hits"] += 1
            cached.speculative = False
            cached.authority_requested_s = self.loop.now_s
            assert cached.callbacks is not None
            cached.callbacks.append(record_completion)
            self.metrics["promoted_running_speculations"] += 1
            return
        if cached is not None and cached.state == "queued":
            # No speculative work has happened, so this is not a hit.  Lazily
            # invalidate the queue entry and dispatch the recorded authority SLO.
            cached.state = "cancelled"
            self.cache.pop(key, None)
            self.metrics["queued_predictions_superseded"] += 1

        job = self._job(
            session_id=session_id,
            url=url,
            duration_s=duration_s,
            score=math.inf,
            speculative=False,
            origin_decision_id=None,
        )
        job.authority_requested_s = self.loop.now_s
        assert job.callbacks is not None
        job.callbacks.append(record_completion)
        self.authority_queue.append(job)
        self._make_authority_room()
        self._dispatch()

    def _running_speculations(self) -> list[Job]:
        return [job for job in self.running.values() if job.speculative]

    def _preempt(self, job: Job) -> None:
        if job.state != "running" or not job.speculative:
            return
        assert job.started_s is not None
        elapsed_s = max(0.0, self.loop.now_s - job.started_s)
        self.speculative_resource_s += elapsed_s
        self.preempted_speculative_s += elapsed_s
        job.executed_speculative_s += elapsed_s
        job.state = "cancelled"
        job.generation += 1
        self.running.pop(job.job_id, None)
        self.cache.pop((job.session_id, job.url), None)
        self.metrics["preempted_speculations"] += 1

    def _make_authority_room(self) -> None:
        while self.authority_queue and len(self.running) >= self.capacity:
            victims = self._running_speculations()
            if not victims:
                break
            victim = min(victims, key=lambda job: (job.score, job.job_id))
            self._preempt(victim)

    def _start(self, job: Job) -> None:
        job.state = "running"
        job.started_s = self.loop.now_s
        job.completion_s = self.loop.now_s + job.duration_s
        job.generation += 1
        generation = job.generation
        self.running[job.job_id] = job
        if job.speculative:
            self.metrics["physical_speculative_starts"] += 1
        else:
            self.metrics["physical_authority_starts"] += 1

        def complete() -> None:
            if job.state != "running" or job.generation != generation:
                return
            self.running.pop(job.job_id, None)
            job.state = "completed"
            if job.started_s is not None and job.origin_decision_id is not None:
                executed_s = self.loop.now_s - job.started_s
                self.speculative_resource_s += executed_s
                job.executed_speculative_s += executed_s
            if job.origin_decision_id is not None:
                self.metrics["completed_speculative_jobs"] += 1
            if job.authority_requested_s is not None:
                self.authority_queue_wait_s += max(
                    0.0,
                    (
                        job.started_s
                        if job.started_s is not None
                        else self.loop.now_s
                    )
                    - job.authority_requested_s,
                )
            callbacks = tuple(job.callbacks or ())
            for callback in callbacks:
                callback(self.loop.now_s)
            self._dispatch()

        self.loop.schedule(job.completion_s, 0, complete)

    def _dispatch(self) -> None:
        self._make_authority_room()
        while self.authority_queue and len(self.running) < self.capacity:
            job = self.authority_queue.popleft()
            if job.state != "queued":
                continue
            self._start(job)
        while (
            len(self.running) < self.capacity
            and len(self._running_speculations()) < self.speculative_cap
            and self.spec_queue
        ):
            _, _, job = heapq.heappop(self.spec_queue)
            if job.state != "queued":
                continue
            self._start(job)

    def cancel_session(self, session_id: str) -> None:
        for job in list(self.running.values()):
            if job.session_id == session_id and job.speculative:
                self._preempt(job)
        for key, job in list(self.cache.items()):
            if key[0] == session_id:
                if job.state == "queued":
                    job.state = "cancelled"
                self.cache.pop(key, None)
        self._dispatch()


def prepare_sessions(
    traces: Path,
    windows: Sequence[ScoredWindow],
    decisions: Sequence[AllVisitDecision],
    timings: Mapping[str, DecisionTiming],
    service_estimates: Mapping[str, ServiceEstimate],
    full_walls: Mapping[str, float],
    *,
    candidate_policy: str,
) -> tuple[PreparedSession, ...]:
    selected_windows, width = candidate_policy_windows(
        windows, service_estimates, candidate_policy=candidate_policy
    )

    selected_by_id = {
        window.decision_id: select_per_task_candidates(
            window,
            service_estimates[window.decision_id],
            per_task_width=width,
            coordination_cost_s=0.001,
        )
        for window in selected_windows
    }
    window_by_id = {window.decision_id: window for window in selected_windows}
    decisions_by_session: dict[str, list[AllVisitDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.decision_id in window_by_id:
            decisions_by_session[decision.session_id].append(decision)
    trace_by_session = {session.session_id: session for session in load_sessions(traces)}

    future_occurrences: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for decision in decisions:
        if decision.decision_id not in window_by_id:
            continue
        timing = timings[decision.decision_id]
        if decision.target_tool_event_index is None:
            continue
        window = window_by_id[decision.decision_id]
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
    for rows in future_occurrences.values():
        rows.sort()

    prepared: list[PreparedSession] = []
    for session_id in sorted(full_walls):
        trace = trace_by_session[session_id]
        epochs: list[Epoch] = []
        for decision in sorted(
            decisions_by_session.get(session_id, ()),
            key=lambda row: row.trigger_event_index,
        ):
            if not decision.lead_llm_event_indices:
                continue
            first_llm = trace.events[decision.lead_llm_event_indices[0]]
            if not isinstance(first_llm, LLMCall):
                raise RuntimeError("prediction epoch does not begin with an LLM")
            timing = timings[decision.decision_id]
            window = window_by_id[decision.decision_id]
            selected = selected_by_id[decision.decision_id]
            candidate_services = tuple(
                next(
                    (
                        service_s
                        for event_index, target_url, service_s in future_occurrences[
                            session_id
                        ]
                        if event_index > decision.trigger_event_index
                        and target_url == candidate.pattern.url
                    ),
                    speculative_service_s(
                        session_id, decision.decision_id, candidate.pattern.url
                    ),
                )
                for candidate in selected
            )
            authority_offset_s: float | None = None
            baseline_done_s: float | None = None
            if decision.outcome == "visit":
                if decision.target_tool_event_index is None:
                    raise RuntimeError("visit decision has no target event")
                target = trace.events[decision.target_tool_event_index]
                if not isinstance(target, ToolCall):
                    raise RuntimeError("visit target is not a tool call")
                authority_offset_s = max(0.0, target.timestamp_s - first_llm.start_timestamp_s)
                baseline_done_s = target.timestamp_s + timing.visit_stall_s
            epochs.append(
                Epoch(
                    decision_id=decision.decision_id,
                    original_start_s=first_llm.start_timestamp_s,
                    authority_offset_s=authority_offset_s,
                    baseline_authority_done_s=baseline_done_s,
                    targets=window.executable_targets,
                    services_s=timing.visit_url_service_s,
                    candidates=selected,
                    candidate_services_s=candidate_services,
                )
            )
        prepared.append(
            PreparedSession(
                session_id=session_id,
                full_wall_s=float(full_walls[session_id]),
                epochs=tuple(epochs),
            )
        )
    return tuple(prepared)


def candidate_policy_windows(
    windows: Sequence[ScoredWindow],
    service_estimates: Mapping[str, ServiceEstimate],
    *,
    candidate_policy: str,
) -> tuple[list[ScoredWindow], int]:
    if candidate_policy == "budget_w5_cap10":
        selected_windows, _ = apply_cross_fold_start_budget(
            windows,
            service_estimates,
            average_width=5,
            burst_multiplier=2,
            coordination_cost_s=0.001,
        )
        width = 20
    elif candidate_policy.startswith("fixed_top"):
        try:
            width = int(candidate_policy.removeprefix("fixed_top"))
        except ValueError as exc:
            raise ValueError(
                f"invalid fixed candidate policy: {candidate_policy}"
            ) from exc
        if width <= 0:
            raise ValueError("fixed candidate width must be positive")
        selected_windows = list(windows)
    else:
        raise ValueError(f"unknown candidate policy: {candidate_policy}")
    return list(selected_windows), width


def simulate(
    sessions: Sequence[PreparedSession],
    *,
    policy: Policy | None,
    visit_capacity: int,
    offered_concurrency: int,
    seed: int,
    wrong_fraction: float = 0.0,
) -> dict[str, Any]:
    if visit_capacity <= 0 or offered_concurrency <= 0:
        raise ValueError("capacity and concurrency must be positive")
    if not 0.0 <= wrong_fraction <= 1.0:
        raise ValueError("wrong_fraction must be in [0, 1]")
    if policy is None:
        speculative_cap = 0
    elif policy.scheduler == "fixed_half":
        speculative_cap = max(1, visit_capacity // 2)
    elif policy.scheduler == "fixed_reserve_one":
        speculative_cap = max(1, visit_capacity - 1)
    elif policy.scheduler == "adaptive_idle_fill":
        speculative_cap = visit_capacity
    else:
        raise ValueError(f"unknown scheduler: {policy.scheduler}")

    loop = EventLoop()
    pool = PreemptibleVisitPool(
        loop, capacity=visit_capacity, speculative_cap=speculative_cap
    )
    waiting = deque(
        sorted(
            sessions,
            key=lambda row: (stable_order(seed, row.session_id), row.session_id),
        )
    )
    active = 0
    flow_times: list[float] = []
    completed_sessions = 0
    final_session_completion_s = 0.0

    def start_next(now_s: float) -> None:
        nonlocal active
        if not waiting:
            return
        prepared = waiting.popleft()
        active += 1
        state = SessionState(prepared=prepared, start_s=now_s)
        if not prepared.epochs:
            loop.schedule(now_s + prepared.full_wall_s, 3, lambda: finish(state))
            return
        schedule_epoch(state)

    def finish(state: SessionState) -> None:
        nonlocal active, completed_sessions, final_session_completion_s
        pool.cancel_session(state.prepared.session_id)
        active -= 1
        completed_sessions += 1
        final_session_completion_s = max(final_session_completion_s, loop.now_s)
        flow_times.append(loop.now_s - state.start_s)
        if waiting:
            start_next(loop.now_s)

    def schedule_epoch(state: SessionState) -> None:
        epoch = state.prepared.epochs[state.epoch_index]
        when_s = state.start_s + epoch.original_start_s + state.shift_s
        loop.schedule(when_s, 2, lambda: begin_epoch(state))

    def advance_after_nonvisit(state: SessionState) -> None:
        state.epoch_index += 1
        if state.epoch_index < len(state.prepared.epochs):
            schedule_epoch(state)
        else:
            loop.schedule(
                state.start_s + state.prepared.full_wall_s + state.shift_s,
                3,
                lambda: finish(state),
            )

    def finish_visit(state: SessionState, epoch: Epoch, done_s: float) -> None:
        assert epoch.baseline_authority_done_s is not None
        state.shift_s = done_s - (
            state.start_s + epoch.baseline_authority_done_s
        )
        state.epoch_index += 1
        if state.epoch_index < len(state.prepared.epochs):
            schedule_epoch(state)
        else:
            loop.schedule(
                state.start_s + state.prepared.full_wall_s + state.shift_s,
                3,
                lambda: finish(state),
            )

    def run_authority_chain(
        state: SessionState, epoch: Epoch, index: int, now_s: float
    ) -> None:
        if index >= len(epoch.targets):
            finish_visit(state, epoch, now_s)
            return
        target_url = epoch.targets[index]
        if wrong_fraction > 0.0:
            digest = hashlib.sha256(
                (
                    f"robustness-label-v1\0{state.prepared.session_id}\0"
                    f"{epoch.decision_id}\0{index}"
                ).encode()
            ).digest()
            fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
            if fraction < wrong_fraction:
                target_url = (
                    "https://all-wrong.invalid/"
                    + hashlib.sha256(
                        f"{state.prepared.session_id}\0{epoch.decision_id}\0{index}".encode()
                    ).hexdigest()
                )
        pool.request_authority(
            session_id=state.prepared.session_id,
            url=target_url,
            duration_s=epoch.services_s[index],
            on_complete=lambda done_s: run_authority_chain(
                state, epoch, index + 1, done_s
            ),
        )

    def begin_epoch(state: SessionState) -> None:
        epoch = state.prepared.epochs[state.epoch_index]
        if policy is not None:
            ranked = sorted(
                zip(epoch.candidates, epoch.candidate_services_s, strict=True),
                key=lambda item: (
                    -item[0].exact_probability,
                    item[0].pattern.position,
                    item[0].pattern.url,
                ),
            )
            for candidate, duration_s in ranked:
                pool.submit_speculation(
                    session_id=state.prepared.session_id,
                    decision_id=epoch.decision_id,
                    candidate=candidate,
                    duration_s=duration_s,
                )
        if epoch.authority_offset_s is None:
            advance_after_nonvisit(state)
            return
        arrival_s = loop.now_s + epoch.authority_offset_s
        loop.schedule(
            arrival_s,
            1,
            lambda: run_authority_chain(state, epoch, 0, loop.now_s),
        )

    for _ in range(min(offered_concurrency, len(waiting))):
        start_next(0.0)
    loop.run()
    if completed_sessions != len(sessions) or active != 0:
        raise RuntimeError("simulation did not complete every session")

    metrics = dict(pool.metrics)
    authority_requests = int(metrics.get("authority_requests", 0))
    physical_calls = int(metrics.get("physical_authority_starts", 0)) + int(
        metrics.get("physical_speculative_starts", 0)
    )
    useful_speculative_s = sum(
        job.executed_speculative_s for job in pool.jobs if job.ever_cache_hit
    )
    wasted_speculative_s = max(
        0.0, pool.speculative_resource_s - useful_speculative_s
    )
    # A physical start counts even when its modeled service duration is zero.
    # Every cache-hit speculative job must have physically started, so this
    # gives an exact call-count partition:
    # physical starts = useful starts + wasted starts.
    useful_speculative_starts = sum(
        job.ever_cache_hit
        for job in pool.jobs
        if job.origin_decision_id is not None
    )
    wasted_speculative_starts = (
        int(metrics.get("physical_speculative_starts", 0))
        - useful_speculative_starts
    )
    return {
        "makespan_s": final_session_completion_s,
        "mean_flow_s": statistics.fmean(flow_times) if flow_times else 0.0,
        "p95_flow_s": percentile(flow_times, 0.95),
        "authority_requests": authority_requests,
        "authority_queue_wait_s": pool.authority_queue_wait_s,
        "authority_queue_wait_per_call_s": (
            pool.authority_queue_wait_s / authority_requests
            if authority_requests
            else 0.0
        ),
        "authority_exposed_s": pool.authority_exposed_s,
        "authority_exposed_per_call_s": (
            pool.authority_exposed_s / authority_requests
            if authority_requests
            else 0.0
        ),
        "cache_hits": int(metrics.get("cache_hits", 0)),
        "ready_cache_hits": int(metrics.get("ready_cache_hits", 0)),
        "inflight_cache_hits": int(metrics.get("inflight_cache_hits", 0)),
        "realized_cache_hit_rate": (
            int(metrics.get("cache_hits", 0)) / authority_requests
            if authority_requests
            else 0.0
        ),
        "physical_speculative_starts": int(
            metrics.get("physical_speculative_starts", 0)
        ),
        "preempted_speculations": int(
            metrics.get("preempted_speculations", 0)
        ),
        "completed_speculative_jobs": int(
            metrics.get("completed_speculative_jobs", 0)
        ),
        "speculative_resource_s": pool.speculative_resource_s,
        "useful_speculative_s": useful_speculative_s,
        "wasted_speculative_s": wasted_speculative_s,
        "useful_speculative_starts": useful_speculative_starts,
        "wasted_speculative_starts": wasted_speculative_starts,
        "wasted_speculative_fraction": (
            wasted_speculative_s / pool.speculative_resource_s
            if pool.speculative_resource_s
            else 0.0
        ),
        "preempted_speculative_s": pool.preempted_speculative_s,
        "call_amplification": physical_calls / authority_requests,
        "speculative_cap": speculative_cap,
        "raw_metrics": metrics,
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_runs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "makespan_s",
        "mean_flow_s",
        "p95_flow_s",
        "authority_queue_wait_s",
        "authority_queue_wait_per_call_s",
        "authority_exposed_s",
        "authority_exposed_per_call_s",
        "realized_cache_hit_rate",
        "physical_speculative_starts",
        "preempted_speculations",
        "completed_speculative_jobs",
        "speculative_resource_s",
        "wasted_speculative_s",
        "useful_speculative_s",
        "wasted_speculative_fraction",
        "preempted_speculative_s",
        "useful_speculative_starts",
        "wasted_speculative_starts",
        "call_amplification",
    )
    result = {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in scalar_keys
    }
    result["authority_requests"] = int(rows[0]["authority_requests"])
    result["cache_hits"] = statistics.fmean(float(row["cache_hits"]) for row in rows)
    result["ready_cache_hits"] = statistics.fmean(
        float(row["ready_cache_hits"]) for row in rows
    )
    result["inflight_cache_hits"] = statistics.fmean(
        float(row["inflight_cache_hits"]) for row in rows
    )
    result["speculative_cap"] = int(rows[0]["speculative_cap"])
    return result


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Resource-tight all-visit replay with preemptible speculation",
        "",
        "Speculation shares a bounded Visit pool with authority. On authority arrival, "
        "an exact in-flight job is promoted with progress preserved; otherwise the "
        "lowest-score running speculations are cancelled immediately until authority "
        "can dispatch. Multi-URL authoritative Visits remain serial within a session.",
        "",
        "`Policy hit` is the resource-unconstrained session-cache coverage of the "
        "unchanged selector (W5=55.51%, Top-10=71.94%). `Realized hit` is the part "
        "that obtained execution under the shared-capacity scheduler and was reusable "
        "by authority; capacity affects only the latter.",
        "",
        "| Candidates | Scheduler | Pool/C | Spec cap | Policy hit | Realized hit | E2E speedup "
        "| Mean-flow speedup | Call amp. | Preempted | Wasted spec seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['candidate_policy']} | {row['scheduler']} "
            f"| {row['visit_capacity']}/{row['offered_concurrency']} "
            f"| {row['speculative_cap']} | {row['policy_cache_hit_rate']:.2%} "
            f"| {row['realized_cache_hit_rate']:.2%} "
            f"| {row['e2e_speedup_fraction']:.2%} "
            f"| {row['mean_flow_speedup_fraction']:.2%} "
            f"| {row['call_amplification']:.3f}x "
            f"| {row['preempted_speculations']:.1f} "
            f"| {row['wasted_speculative_s']:.1f} |"
        )
    lines.extend(
        [
            "",
            "`fixed_half` and `fixed_reserve_one` leave a fixed speculative ceiling "
            "even while authority is idle. `adaptive_idle_fill` may use the entire "
            "idle pool and shrinks immediately through preemption when authority "
            "arrives. Authority-to-authority queueing is retained in both baseline "
            "and treatment; wrong speculation is never allowed to add queueing in "
            "front of authority.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visit-capacities", type=int, nargs="+")
    parser.add_argument(
        "--capacity-ratios", type=float, nargs="+", default=[1.0, 1.5, 2.0]
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    if args.visit_capacities and any(value <= 0 for value in args.visit_capacities):
        parser.error("capacities must be positive")
    if any(value <= 0 for value in args.capacity_ratios + args.concurrencies):
        parser.error("capacities and concurrencies must be positive")
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    trace_scale = trace_llm_scale_metadata(args.traces)
    windows, nested_oof, decisions = collect_nested_oof_all_visit_windows(
        args.traces, candidate_pool_size=20, selector_model="blend"
    )
    timings = collect_all_visit_timings(args.traces, decisions, llm_duration_scale=1.0)
    service_estimates, service_estimator = build_oof_service_estimates(
        windows, timings, domain_prior_strength=10.0
    )
    full_walls = session_full_walls(args.traces, llm_duration_scale=1.0)
    prepared = {
        candidate_policy: prepare_sessions(
            args.traces,
            windows,
            decisions,
            timings,
            service_estimates,
            full_walls,
            candidate_policy=candidate_policy,
        )
        for candidate_policy in ("budget_w5_cap10", "fixed_top10")
    }
    potential_cache: dict[str, dict[str, Any]] = {}
    for candidate_policy in prepared:
        policy_windows, width = candidate_policy_windows(
            windows, service_estimates, candidate_policy=candidate_policy
        )
        _, audit = build_session_global_cache_replays(
            args.traces,
            policy_windows,
            decisions,
            timings,
            service_estimates,
            full_walls,
            per_task_width=width,
            coordination_cost_s=0.001,
        )
        potential_cache[candidate_policy] = audit
    policies = [
        Policy(candidate, candidate, scheduler)
        for candidate in prepared
        for scheduler in (
            "fixed_half",
            "fixed_reserve_one",
            "adaptive_idle_fill",
        )
    ]

    results: list[dict[str, Any]] = []
    baseline_rows: dict[str, Any] = {}
    cells = (
        [
            (capacity, concurrency)
            for capacity in args.visit_capacities
            for concurrency in args.concurrencies
        ]
        if args.visit_capacities
        else sorted(
            {
                (max(1, math.ceil(concurrency * ratio)), concurrency)
                for concurrency in args.concurrencies
                for ratio in args.capacity_ratios
            }
        )
    )
    for capacity, concurrency in cells:
            baseline_runs = [
                simulate(
                    prepared["budget_w5_cap10"],
                    policy=None,
                    visit_capacity=capacity,
                    offered_concurrency=concurrency,
                    seed=seed,
                )
                for seed in range(args.repetitions)
            ]
            baseline = aggregate_runs(baseline_runs)
            baseline_rows[f"pool{capacity}_c{concurrency}"] = {
                "aggregate": baseline,
                "runs": baseline_runs,
            }
            for policy in policies:
                runs = [
                    simulate(
                        prepared[policy.candidate_policy],
                        policy=policy,
                        visit_capacity=capacity,
                        offered_concurrency=concurrency,
                        seed=seed,
                    )
                    for seed in range(args.repetitions)
                ]
                aggregate = aggregate_runs(runs)
                aggregate.update(
                    {
                        "candidate_policy": policy.candidate_policy,
                        "scheduler": policy.scheduler,
                        "visit_capacity": capacity,
                        "offered_concurrency": concurrency,
                        "capacity_per_active_agent": capacity / concurrency,
                        "policy_cache_hits": potential_cache[
                            policy.candidate_policy
                        ]["cache_hit_occurrences"],
                        "policy_cache_hit_rate": potential_cache[
                            policy.candidate_policy
                        ]["cache_hit_occurrences"]
                        / aggregate["authority_requests"],
                        "realized_fraction_of_policy_hits": (
                            aggregate["cache_hits"]
                            / potential_cache[policy.candidate_policy][
                                "cache_hit_occurrences"
                            ]
                        ),
                        "e2e_speedup_fraction": 1.0
                        - aggregate["makespan_s"] / baseline["makespan_s"],
                        "mean_flow_speedup_fraction": 1.0
                        - aggregate["mean_flow_s"] / baseline["mean_flow_s"],
                        "authority_queue_wait_delta_s": aggregate[
                            "authority_queue_wait_s"
                        ]
                        - baseline["authority_queue_wait_s"],
                        "runs": runs,
                    }
                )
                results.append(aggregate)

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "effective_llm_duration_scale": trace_scale["materialized_scale"],
            "visit_capacities": args.visit_capacities,
            "capacity_ratios": args.capacity_ratios,
            "concurrencies": args.concurrencies,
            "repetitions": args.repetitions,
            "candidate_policies": ["budget_w5_cap10", "fixed_top10"],
            "schedulers": [
                "fixed_half",
                "fixed_reserve_one",
                "adaptive_idle_fill",
            ],
            "cache": "infinite-TTL session URL; zero read cost; speculative results only",
            "authority_semantics": "preemptive priority with exact in-flight promotion",
            "speculative_service": "stable SHA-256 uniform sample in [2, 8] seconds",
        },
        "source_sha256": {
            "runner": sha256_file(SCRIPT),
            "all_visit_runner": sha256_file(
                SCRIPT.parent / "run_pattern_v2_trace_all_visit_wall.py"
            ),
            "llm_timing_manifest": trace_scale["manifest_sha256"],
        },
        "nested_oof": nested_oof,
        "service_estimator": service_estimator,
        "potential_cache": potential_cache,
        "baseline_rows": baseline_rows,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
