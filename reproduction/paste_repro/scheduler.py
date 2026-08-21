"""A bounded, state-isolating asynchronous speculative scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Optional

from .invocation import Invocation


Executor = Callable[[Invocation], Awaitable[Any]]


@dataclass
class SchedulerStats:
    admitted: int = 0
    duplicate_predictions: int = 0
    rejected_capacity: int = 0
    completed_reuse: int = 0
    inflight_promotions: int = 0
    misses: int = 0
    expired: int = 0
    speculative_failures: int = 0
    authoritative_executions: int = 0
    commits: int = 0
    saved_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuthoritativeResult:
    invocation: Invocation
    result: Any
    source: str
    exposed_wait_s: float
    saved_time_s: float


@dataclass(frozen=True)
class _ExecutionRecord:
    result: Any
    error: Optional[BaseException]
    started_at: float
    finished_at: float


@dataclass
class _SpeculativeJob:
    invocation: Invocation
    session_id: str
    task: "asyncio.Task[_ExecutionRecord]"
    created_at: float
    expires_at: float


class SpeculativeScheduler:
    """Run predictions early but commit only an exact authoritative match.

    ``max_concurrency`` bounds active speculative executor calls and
    ``max_pending`` bounds all retained in-flight/completed predictions.
    Authoritative misses bypass the speculative semaphore, so scheduler-owned
    speculative capacity cannot queue the correctness-critical path.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        max_concurrency: int = 4,
        max_pending: int | None = None,
        ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        pending_limit = max_concurrency if max_pending is None else max_pending
        if pending_limit <= 0 or pending_limit < max_concurrency:
            raise ValueError("max_pending must be at least max_concurrency")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self._executor = executor
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_pending = pending_limit
        self._ttl_s = float(ttl_s)
        self._clock = clock
        self._jobs: dict[tuple[str, tuple[str, str]], _SpeculativeJob] = {}
        self._authoritative_state: list[AuthoritativeResult] = []
        self._lock = asyncio.Lock()
        self.stats = SchedulerStats()

    @property
    def pending_count(self) -> int:
        return len(self._jobs)

    @property
    def authoritative_state(self) -> tuple[AuthoritativeResult, ...]:
        return tuple(self._authoritative_state)

    async def _run_speculative(self, invocation: Invocation) -> _ExecutionRecord:
        async with self._semaphore:
            started_at = self._clock()
            try:
                result = await self._executor(invocation)
                error: Optional[BaseException] = None
            except Exception as exc:  # the authoritative path gets a clean fallback
                result = None
                error = exc
            return _ExecutionRecord(result, error, started_at, self._clock())

    @staticmethod
    def _job_key(
        session_id: str, invocation: Invocation
    ) -> tuple[str, tuple[str, str]]:
        return session_id, invocation.key

    async def speculate(
        self, invocation: Invocation, *, session_id: str = "default"
    ) -> bool:
        """Admit a prediction, returning false for duplicate/capacity rejection."""

        await self.sweep()
        key = self._job_key(session_id, invocation)
        async with self._lock:
            if key in self._jobs:
                self.stats.duplicate_predictions += 1
                return False
            if len(self._jobs) >= self._max_pending:
                self.stats.rejected_capacity += 1
                return False
            now = self._clock()
            task = asyncio.create_task(self._run_speculative(invocation))
            self._jobs[key] = _SpeculativeJob(
                invocation=invocation,
                session_id=session_id,
                task=task,
                created_at=now,
                expires_at=now + self._ttl_s,
            )
            self.stats.admitted += 1
            return True

    async def authoritative(
        self, invocation: Invocation, *, session_id: str = "default"
    ) -> AuthoritativeResult:
        """Confirm one invocation and cross the authoritative commit boundary."""

        await self.sweep()
        key = self._job_key(session_id, invocation)
        async with self._lock:
            job = self._jobs.pop(key, None)

        if job is None:
            self.stats.misses += 1
            self.stats.authoritative_executions += 1
            started_at = self._clock()
            result = await self._executor(invocation)
            committed = AuthoritativeResult(
                invocation=invocation,
                result=result,
                source="executed",
                exposed_wait_s=max(0.0, self._clock() - started_at),
                saved_time_s=0.0,
            )
            self._commit(committed)
            return committed

        confirmation_at = self._clock()
        was_complete = job.task.done()
        wait_started = self._clock()
        record = await job.task
        exposed_wait_s = max(0.0, self._clock() - wait_started)
        if record.error is not None:
            self.stats.speculative_failures += 1
            self.stats.misses += 1
            self.stats.authoritative_executions += 1
            fallback_started = self._clock()
            result = await self._executor(invocation)
            committed = AuthoritativeResult(
                invocation=invocation,
                result=result,
                source="executed_after_speculative_failure",
                exposed_wait_s=max(0.0, self._clock() - fallback_started),
                saved_time_s=0.0,
            )
            self._commit(committed)
            return committed

        if was_complete:
            self.stats.completed_reuse += 1
            source = "reused"
        else:
            self.stats.inflight_promotions += 1
            source = "promoted"
        execution_s = max(0.0, record.finished_at - record.started_at)
        performed_before_confirmation_s = max(
            0.0,
            min(confirmation_at, record.finished_at) - record.started_at,
        )
        saved_time_s = min(execution_s, performed_before_confirmation_s)
        self.stats.saved_time_s += saved_time_s
        committed = AuthoritativeResult(
            invocation=invocation,
            result=record.result,
            source=source,
            exposed_wait_s=exposed_wait_s,
            saved_time_s=saved_time_s,
        )
        self._commit(committed)
        return committed

    def _commit(self, result: AuthoritativeResult) -> None:
        # This is the sole commit boundary. Task callbacks never write here.
        self._authoritative_state.append(result)
        self.stats.commits += 1

    async def sweep(
        self, *, now: float | None = None, session_id: str | None = None
    ) -> int:
        """Expire unconfirmed predictions and discard their isolated results."""

        cutoff = self._clock() if now is None else float(now)
        cancelled: list[asyncio.Task[_ExecutionRecord]] = []
        expired_count = 0
        async with self._lock:
            for key, job in tuple(self._jobs.items()):
                if session_id is not None and job.session_id != session_id:
                    continue
                if job.expires_at > cutoff and not math.isinf(cutoff):
                    continue
                self._jobs.pop(key, None)
                expired_count += 1
                if not job.task.done():
                    job.task.cancel()
                cancelled.append(job.task)
            self.stats.expired += expired_count
        if cancelled:
            await asyncio.gather(*cancelled, return_exceptions=True)
        return expired_count

    async def close(self) -> None:
        await self.sweep(now=math.inf)

    async def __aenter__(self) -> "SpeculativeScheduler":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

