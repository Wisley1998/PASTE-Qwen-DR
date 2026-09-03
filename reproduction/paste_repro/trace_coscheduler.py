"""Runtime primitives for the live all-visit trace experiment.

The analytical all-visit replay is useful for policy development, but it does
not exercise wall-clock overlap.  This module provides the two small runtime
pieces needed by the live experiment:

* a shared, authority-first, preemptible Visit pool; and
* a pre-engine admission queue ranked by exposed-tool gain per LLM pressure.

Both components are deliberately independent of vLLM so their safety and
ordering invariants can be tested without a GPU.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
import heapq
import math
import time
from typing import Any, Callable


@dataclass(frozen=True)
class VisitResult:
    source: str
    exposed_wait_s: float
    service_s: float
    saved_service_s: float


@dataclass
class _VisitJob:
    job_id: int
    session_id: str
    url: str
    duration_s: float
    score: float
    speculative: bool
    future: asyncio.Future[None]
    state: str = "queued"
    created_s: float = 0.0
    started_s: float | None = None
    finished_s: float | None = None
    runner: asyncio.Task[None] | None = None
    generation: int = 0
    authority_requested_s: float | None = None
    origin_decision_id: str | None = None
    ever_claimed: bool = False
    executed_speculative_s: float = 0.0


class AsyncPreemptibleVisitPool:
    """Execute real asynchronous sleeps with shared-capacity semantics.

    An exact running prediction is promoted without losing progress.  On an
    authority miss, the lowest-score running prediction is cancelled until an
    authority slot is available.  Completed speculative results remain in a
    session-scoped infinite-TTL cache until :meth:`close_session`.
    """

    def __init__(
        self,
        *,
        capacity: int,
        speculative_cap: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if speculative_cap is None:
            speculative_cap = capacity
        if not 0 <= speculative_cap <= capacity:
            raise ValueError("speculative_cap must be in [0, capacity]")
        self.capacity = capacity
        self.speculative_cap = speculative_cap
        self._clock = clock
        self._lock = asyncio.Lock()
        self._next_job_id = 0
        self._running: dict[int, _VisitJob] = {}
        self._authority_queue: deque[_VisitJob] = deque()
        self._spec_queue: list[tuple[float, int, _VisitJob]] = []
        self._cache: dict[tuple[str, str], _VisitJob] = {}
        self._jobs: list[_VisitJob] = []
        self._closed = False
        self.metrics: Counter[str] = Counter()
        self.authority_queue_wait_s = 0.0
        self.speculative_resource_s = 0.0
        self.preempted_speculative_s = 0.0

    def _new_job(
        self,
        *,
        session_id: str,
        url: str,
        duration_s: float,
        score: float,
        speculative: bool,
        decision_id: str | None,
    ) -> _VisitJob:
        if not session_id or not url:
            raise ValueError("session_id and url must be non-empty")
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("duration_s must be finite and non-negative")
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        self._next_job_id += 1
        job = _VisitJob(
            job_id=self._next_job_id,
            session_id=session_id,
            url=url,
            duration_s=float(duration_s),
            score=float(score),
            speculative=speculative,
            future=asyncio.get_running_loop().create_future(),
            created_s=self._clock(),
            origin_decision_id=decision_id,
        )
        self._jobs.append(job)
        return job

    def _running_speculations(self) -> list[_VisitJob]:
        return [job for job in self._running.values() if job.speculative]

    def _cancel_running_speculation_locked(self, job: _VisitJob) -> None:
        if job.state != "running" or not job.speculative:
            return
        now = self._clock()
        elapsed = max(0.0, now - (job.started_s or now))
        job.executed_speculative_s += elapsed
        self.speculative_resource_s += elapsed
        self.preempted_speculative_s += elapsed
        job.state = "cancelled"
        job.generation += 1
        self._running.pop(job.job_id, None)
        self._cache.pop((job.session_id, job.url), None)
        if job.runner is not None and not job.runner.done():
            job.runner.cancel()
        if not job.future.done():
            job.future.cancel()
        self.metrics["preempted_speculations"] += 1

    def _make_authority_room_locked(self) -> None:
        while self._authority_queue and len(self._running) >= self.capacity:
            victims = self._running_speculations()
            if not victims:
                break
            victim = min(victims, key=lambda row: (row.score, row.job_id))
            self._cancel_running_speculation_locked(victim)

    def _start_locked(self, job: _VisitJob) -> None:
        job.state = "running"
        job.started_s = self._clock()
        job.generation += 1
        generation = job.generation
        self._running[job.job_id] = job
        self.metrics[
            "physical_speculative_starts"
            if job.speculative
            else "physical_authority_starts"
        ] += 1

        async def run() -> None:
            try:
                await asyncio.sleep(job.duration_s)
            except asyncio.CancelledError:
                return
            async with self._lock:
                if job.state != "running" or job.generation != generation:
                    return
                now = self._clock()
                self._running.pop(job.job_id, None)
                job.state = "completed"
                job.finished_s = now
                if job.origin_decision_id is not None:
                    elapsed = max(0.0, now - (job.started_s or now))
                    job.executed_speculative_s += elapsed
                    self.speculative_resource_s += elapsed
                    self.metrics["completed_speculative_jobs"] += 1
                if (
                    job.authority_requested_s is not None
                    and job.started_s is not None
                ):
                    self.authority_queue_wait_s += max(
                        0.0, job.started_s - job.authority_requested_s
                    )
                if not job.future.done():
                    job.future.set_result(None)
                self._dispatch_locked()

        job.runner = asyncio.create_task(run())

    def _dispatch_locked(self) -> None:
        self._make_authority_room_locked()
        while self._authority_queue and len(self._running) < self.capacity:
            job = self._authority_queue.popleft()
            if job.state == "queued":
                self._start_locked(job)
        while (
            len(self._running) < self.capacity
            and len(self._running_speculations()) < self.speculative_cap
            and self._spec_queue
        ):
            _, _, job = heapq.heappop(self._spec_queue)
            if job.state == "queued":
                self._start_locked(job)

    async def speculate_batch(
        self,
        rows: list[tuple[str, str, float, float, str]],
    ) -> tuple[bool, ...]:
        """Submit ``(session, url, duration, score, decision_id)`` rows."""

        admitted = [False] * len(rows)
        ordered = sorted(enumerate(rows), key=lambda row: (-row[1][3], row[0]))
        async with self._lock:
            if self._closed:
                raise RuntimeError("visit pool is closed")
            for index, (session_id, url, duration_s, score, decision_id) in ordered:
                self.metrics["policy_candidates"] += 1
                key = (session_id, url)
                if key in self._cache:
                    self.metrics["cache_deduplicated_candidates"] += 1
                    continue
                job = self._new_job(
                    session_id=session_id,
                    url=url,
                    duration_s=duration_s,
                    score=score,
                    speculative=True,
                    decision_id=decision_id,
                )
                self._cache[key] = job
                heapq.heappush(self._spec_queue, (-score, job.job_id, job))
                self.metrics["speculative_admitted"] += 1
                admitted[index] = True
            self._dispatch_locked()
        return tuple(admitted)

    async def authoritative(
        self,
        *,
        session_id: str,
        url: str,
        duration_s: float,
    ) -> VisitResult:
        requested_s = self._clock()
        source = "executed"
        async with self._lock:
            if self._closed:
                raise RuntimeError("visit pool is closed")
            self.metrics["authority_requests"] += 1
            key = (session_id, url)
            cached = self._cache.get(key)
            if cached is not None and cached.state == "completed":
                cached.ever_claimed = True
                self.metrics["cache_hits"] += 1
                self.metrics["ready_cache_hits"] += 1
                service_s = max(
                    0.0,
                    (cached.finished_s or requested_s)
                    - (cached.started_s or requested_s),
                )
                return VisitResult("reused", 0.0, service_s, service_s)
            if cached is not None and cached.state == "running":
                cached.ever_claimed = True
                cached.speculative = False
                cached.authority_requested_s = requested_s
                self.metrics["cache_hits"] += 1
                self.metrics["inflight_cache_hits"] += 1
                self.metrics["promoted_running_speculations"] += 1
                source = "promoted_inflight"
                job = cached
            else:
                if cached is not None and cached.state == "queued":
                    cached.state = "cancelled"
                    self._cache.pop(key, None)
                    if not cached.future.done():
                        cached.future.cancel()
                    self.metrics["queued_predictions_superseded"] += 1
                job = self._new_job(
                    session_id=session_id,
                    url=url,
                    duration_s=duration_s,
                    score=0.0,
                    speculative=False,
                    decision_id=None,
                )
                job.authority_requested_s = requested_s
                self._authority_queue.append(job)
                self._make_authority_room_locked()
                self._dispatch_locked()

        await asyncio.shield(job.future)
        finished_s = job.finished_s or self._clock()
        started_s = job.started_s or finished_s
        service_s = max(0.0, finished_s - started_s)
        saved_s = (
            min(service_s, max(0.0, requested_s - started_s))
            if source == "promoted_inflight"
            else 0.0
        )
        return VisitResult(
            source=source,
            exposed_wait_s=max(0.0, finished_s - requested_s),
            service_s=service_s,
            saved_service_s=saved_s,
        )

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            for job in list(self._running.values()):
                if job.session_id == session_id and job.speculative:
                    self._cancel_running_speculation_locked(job)
            for key, job in list(self._cache.items()):
                if key[0] != session_id:
                    continue
                if job.state == "queued":
                    job.state = "cancelled"
                    if not job.future.done():
                        job.future.cancel()
                self._cache.pop(key, None)
            self._dispatch_locked()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            jobs = list(self._jobs)
            for job in jobs:
                if job.state == "running" and job.runner is not None:
                    job.runner.cancel()
                if not job.future.done():
                    job.future.cancel()
                if job.state in {"queued", "running"}:
                    job.state = "cancelled"
            self._running.clear()
            self._authority_queue.clear()
            self._spec_queue.clear()
            self._cache.clear()
        await asyncio.gather(
            *(job.runner for job in jobs if job.runner is not None),
            return_exceptions=True,
        )

    def snapshot(self) -> dict[str, Any]:
        useful_s = sum(
            job.executed_speculative_s for job in self._jobs if job.ever_claimed
        )
        return {
            "capacity": self.capacity,
            "speculative_cap": self.speculative_cap,
            "running": len(self._running),
            "cached": len(self._cache),
            "metrics": dict(self.metrics),
            "authority_queue_wait_s": self.authority_queue_wait_s,
            "speculative_resource_s": self.speculative_resource_s,
            "preempted_speculative_s": self.preempted_speculative_s,
            "useful_speculative_s": useful_s,
            "wasted_speculative_s": max(
                0.0, self.speculative_resource_s - useful_s
            ),
        }


@dataclass(frozen=True)
class AdmissionTurn:
    session_id: str
    cold: bool
    exposed_tool_gain_s: float
    predicted_llm_service_s: float
    context_tokens: int
    kv_load: float = 0.0


@dataclass
class _AdmissionWaiter:
    sequence: int
    turn: AdmissionTurn
    enqueued_s: float
    future: asyncio.Future[None]


class GainPressureAdmissionController:
    """Pre-engine admission implementing gain-efficiency plus aging."""

    def __init__(
        self,
        *,
        pressure_low: int,
        pressure_high: int,
        cold_session_cap: int,
        gain_weight: float = 1.0,
        aging_weight: float = 0.02,
        kv_weight: float = 1.0,
        context_ref_tokens: int = 16_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= pressure_low <= pressure_high:
            raise ValueError("require 1 <= pressure_low <= pressure_high")
        if cold_session_cap <= 0 or context_ref_tokens <= 0:
            raise ValueError("caps must be positive")
        if gain_weight < 0.0 or aging_weight < 0.0 or kv_weight < 0.0:
            raise ValueError("weights must be non-negative")
        self.pressure_low = pressure_low
        self.pressure_high = pressure_high
        self.cold_session_cap = cold_session_cap
        self.gain_weight = gain_weight
        self.aging_weight = aging_weight
        self.kv_weight = kv_weight
        self.context_ref_tokens = context_ref_tokens
        self._clock = clock
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._waiting: list[_AdmissionWaiter] = []
        self._running = 0
        self._started_sessions: set[str] = set()
        self._running_by_session: Counter[str] = Counter()
        self._running_kv_by_session: dict[str, deque[float]] = {}
        self._running_kv_load = 0.0
        self._closed = False
        self.metrics: Counter[str] = Counter()
        self.wait_samples_s: list[float] = []

    def _pressure(self, turn: AdmissionTurn) -> float:
        context_factor = 1.0 + max(0, turn.context_tokens) / self.context_ref_tokens
        projected = self._engine_pressure(turn)
        load_factor = 1.0 + projected / self.pressure_high
        return max(1e-6, turn.predicted_llm_service_s * context_factor * load_factor)

    def _engine_pressure(self, turn: AdmissionTurn | None = None) -> float:
        """DecodeLoad + gamma * KVLoad in request-equivalent units."""

        decode_load = float(self._running)
        kv_load = self._running_kv_load
        if turn is not None:
            decode_load += 1.0
            kv_load += max(0.0, turn.kv_load)
        return decode_load + self.kv_weight * kv_load

    def _priority(self, waiter: _AdmissionWaiter, now: float) -> float:
        aging = self.aging_weight * max(0.0, now - waiter.enqueued_s)
        return (
            self.gain_weight
            * max(0.0, waiter.turn.exposed_tool_gain_s)
            / self._pressure(waiter.turn)
            + aging
        )

    def _eligible(self, waiter: _AdmissionWaiter) -> bool:
        if self._engine_pressure() < self.pressure_low:
            return True
        if not waiter.turn.cold:
            return True
        return len(self._started_sessions) < self.cold_session_cap

    def _dispatch_locked(self) -> None:
        while self._waiting:
            now = self._clock()
            eligible = [
                row
                for row in self._waiting
                if self._eligible(row)
                and self._engine_pressure(row.turn) <= self.pressure_high
            ]
            # An oversized request must not deadlock an otherwise idle engine.
            if not eligible and self._running == 0:
                eligible = [row for row in self._waiting if self._eligible(row)]
            if not eligible:
                break
            selected = max(
                eligible,
                key=lambda row: (
                    self._priority(row, now),
                    -row.enqueued_s,
                    -row.sequence,
                ),
            )
            self._waiting.remove(selected)
            self._running += 1
            self._running_by_session[selected.turn.session_id] += 1
            kv_load = max(0.0, selected.turn.kv_load)
            self._running_kv_load += kv_load
            self._running_kv_by_session.setdefault(
                selected.turn.session_id, deque()
            ).append(kv_load)
            self._started_sessions.add(selected.turn.session_id)
            wait_s = max(0.0, now - selected.enqueued_s)
            self.wait_samples_s.append(wait_s)
            self.metrics["admitted"] += 1
            self.metrics["cold_admitted"] += int(selected.turn.cold)
            self.metrics["max_running"] = max(
                self.metrics["max_running"], self._running
            )
            self.metrics["max_engine_pressure_milli"] = max(
                self.metrics["max_engine_pressure_milli"],
                int(round(1000.0 * self._engine_pressure())),
            )
            self.metrics["max_waiting"] = max(
                self.metrics["max_waiting"], len(self._waiting)
            )
            if not selected.future.done():
                selected.future.set_result(None)

    async def acquire(self, turn: AdmissionTurn) -> None:
        if (
            turn.predicted_llm_service_s < 0.0
            or turn.context_tokens < 0
            or turn.kv_load < 0.0
        ):
            raise ValueError("turn pressure inputs must be non-negative")
        async with self._lock:
            if self._closed:
                raise RuntimeError("admission controller is closed")
            self._sequence += 1
            waiter = _AdmissionWaiter(
                sequence=self._sequence,
                turn=turn,
                enqueued_s=self._clock(),
                future=asyncio.get_running_loop().create_future(),
            )
            self._waiting.append(waiter)
            self.metrics["submitted"] += 1
            self.metrics["max_waiting"] = max(
                self.metrics["max_waiting"], len(self._waiting)
            )
            self._dispatch_locked()
        await waiter.future

    async def release(self, session_id: str) -> None:
        async with self._lock:
            if self._running_by_session[session_id] <= 0:
                raise RuntimeError("release without a matching admission")
            self._running_by_session[session_id] -= 1
            if not self._running_by_session[session_id]:
                del self._running_by_session[session_id]
            self._running -= 1
            kv_rows = self._running_kv_by_session.get(session_id)
            if not kv_rows:
                raise RuntimeError("missing KV load for admitted request")
            remaining_kv_load = self._running_kv_load - kv_rows.popleft()
            self._running_kv_load = (
                0.0 if remaining_kv_load < 1e-12 else remaining_kv_load
            )
            if not kv_rows:
                del self._running_kv_by_session[session_id]
            self.metrics["released"] += 1
            self._dispatch_locked()

    async def finish_session(self, session_id: str) -> None:
        """Retire one foreground session so a queued cold session may enter."""

        async with self._lock:
            if self._running_by_session.get(session_id, 0):
                raise RuntimeError("cannot finish a session with a running turn")
            self._started_sessions.discard(session_id)
            self.metrics["sessions_finished"] += 1
            self._dispatch_locked()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            for waiter in self._waiting:
                if not waiter.future.done():
                    waiter.future.cancel()
            self._waiting.clear()

    def snapshot(self) -> dict[str, Any]:
        ordered = sorted(self.wait_samples_s)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return {
            "configuration": {
                "pressure_low": self.pressure_low,
                "pressure_high": self.pressure_high,
                "cold_session_cap": self.cold_session_cap,
                "gain_weight": self.gain_weight,
                "aging_weight": self.aging_weight,
                "kv_weight": self.kv_weight,
                "context_ref_tokens": self.context_ref_tokens,
            },
            "running": self._running,
            "running_kv_load": self._running_kv_load,
            "engine_pressure": self._engine_pressure(),
            "waiting": len(self._waiting),
            "started_sessions": len(self._started_sessions),
            "metrics": dict(self.metrics),
            "mean_wait_s": (
                sum(self.wait_samples_s) / len(self.wait_samples_s)
                if self.wait_samples_s
                else 0.0
            ),
            "p95_wait_s": ordered[p95_index] if ordered else 0.0,
        }
