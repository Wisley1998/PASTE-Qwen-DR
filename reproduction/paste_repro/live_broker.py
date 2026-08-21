"""Shared, bounded broker for live authoritative and speculative tool calls.

Unlike :mod:`paste_repro.scheduler`, this broker models the tool service as a
real shared resource.  Authoritative and speculative work enter the same
worker pool.  Authoritative work is dispatched first by default; an explicit
one-worker reservation can alternate one speculative start with the next
same-tool authoritative start.  Speculative work is otherwise restricted to
a configurable part of the pool.  Speculative results
remain private until an exact, session-scoped authoritative invocation claims
them.

The broker deliberately does not know how a tool is implemented.  Its
``executor`` may perform real HTTP/API calls, invoke a local service, or use a
deterministic executor in tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
import time
from typing import Any, Literal, Optional

from .invocation import Invocation


Executor = Callable[[Invocation], Awaitable[Any]]
Lane = Literal["authoritative", "speculative"]
JobState = Literal[
    "queued", "running", "cancelling", "completed", "failed", "cancelled"
]


@dataclass
class LiveBrokerStats:
    speculative_admitted: int = 0
    duplicate_predictions: int = 0
    rejected_speculative_capacity: int = 0
    speculative_started: int = 0
    speculative_completed: int = 0
    speculative_failures: int = 0
    speculative_expired: int = 0
    speculative_cancelled: int = 0
    authoritative_requests: int = 0
    authoritative_misses: int = 0
    authoritative_executions: int = 0
    authoritative_started: int = 0
    authoritative_completed: int = 0
    authoritative_failures: int = 0
    queued_promotions: int = 0
    running_promotions: int = 0
    completed_reuse: int = 0
    reserved_speculative_dispatches: int = 0
    authoritative_after_reserved_dispatches: int = 0
    commits: int = 0
    saved_service_s: float = 0.0
    wasted_speculative_service_s: float = 0.0
    max_queued_authoritative: int = 0
    max_queued_speculative: int = 0
    max_running_total: int = 0
    max_running_speculative: int = 0
    max_running_by_tool: dict[str, int] = field(default_factory=dict)
    max_queued_by_tool: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveAuthoritativeResult:
    invocation: Invocation
    result: Any
    source: str
    exposed_wait_s: float
    queue_s: float
    service_s: float
    saved_service_s: float


@dataclass(frozen=True)
class _ExecutionRecord:
    result: Any
    error: Optional[BaseException]
    started_at: float
    finished_at: float


@dataclass
class _BrokerJob:
    job_id: int
    invocation: Invocation
    session_id: str
    lane: Lane
    priority: float
    created_at: float
    expires_at: float
    future: "asyncio.Future[_ExecutionRecord]"
    originally_speculative: bool
    state: JobState = "queued"
    generation: int = 0
    queue_order: int = 0
    confirmed_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    worker_id: int | None = None
    runner: "asyncio.Task[None] | None" = None
    expiry_task: "asyncio.Task[None] | None" = None


class LiveToolBroker:
    """Prioritized tool queue with bounded, correctness-safe speculation.

    ``max_workers`` bounds all live executor calls.  At most
    ``max_speculative_workers`` of those calls may be speculative, leaving the
    rest available to authoritative traffic.  ``min_speculative_workers`` may
    reserve a bounded opportunity for dispatchable speculative work, but it
    must leave at least one global and per-tool slot for authoritative work.
    A zero minimum preserves authoritative-first dispatch.
    ``max_speculative_pending`` bounds queued, running,
    and completed-but-unclaimed predictions; authoritative calls are never
    rejected by that speculative limit.  Optional ``tool_capacities`` impose
    additional shared limits on physical calls of a given tool.  Both lanes
    consume the same per-tool capacity, so speculation cannot bypass a visit
    service's concurrency limit.  ``tool_min_start_intervals_s`` optionally
    adds a shared per-tool start-rate gate.  Rate-limited calls remain in the
    broker queue and do not consume worker or service time while waiting.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        max_workers: int = 8,
        max_speculative_workers: int | None = None,
        min_speculative_workers: int = 0,
        max_speculative_pending: int | None = None,
        ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        service_time_hints_s: Mapping[str, float] | None = None,
        service_ewma_alpha: float = 0.2,
        tool_capacities: Mapping[str, int] | None = None,
        tool_min_start_intervals_s: Mapping[str, float] | None = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if max_speculative_workers is None:
            max_speculative_workers = max(0, max_workers - 1)
        if not 0 <= max_speculative_workers <= max_workers:
            raise ValueError(
                "max_speculative_workers must be between zero and max_workers"
            )
        if (
            isinstance(min_speculative_workers, bool)
            or not isinstance(min_speculative_workers, int)
            or not 0 <= min_speculative_workers <= max_speculative_workers
            or min_speculative_workers > 1
        ):
            raise ValueError(
                "min_speculative_workers must be 0 or 1 and no greater than "
                "max_speculative_workers"
            )
        if max_speculative_pending is None:
            max_speculative_pending = max(max_workers, 2 * max_workers)
        if max_speculative_pending <= 0:
            raise ValueError("max_speculative_pending must be positive")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if not 0.0 < service_ewma_alpha <= 1.0:
            raise ValueError("service_ewma_alpha must be in (0, 1]")

        hints = dict(service_time_hints_s or {})
        if any(value < 0 or not math.isfinite(value) for value in hints.values()):
            raise ValueError("service-time hints must be finite and non-negative")
        capacities: dict[str, int] = {}
        for raw_name, raw_capacity in dict(tool_capacities or {}).items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("tool-capacity names must be non-empty strings")
            if (
                isinstance(raw_capacity, bool)
                or not isinstance(raw_capacity, int)
                or not 1 <= raw_capacity <= max_workers
            ):
                raise ValueError(
                    "each tool capacity must be an integer in [1, max_workers]"
                )
            capacities[raw_name] = raw_capacity
        if min_speculative_workers > 0:
            if min_speculative_workers >= max_workers:
                raise ValueError(
                    "min_speculative_workers must leave at least one global "
                    "worker available to authoritative traffic"
                )
            too_small = sorted(
                name
                for name, capacity in capacities.items()
                if min_speculative_workers >= capacity
            )
            if too_small:
                raise ValueError(
                    "min_speculative_workers must leave at least one per-tool "
                    "slot available to authoritative traffic: "
                    + ", ".join(too_small)
                )
        min_start_intervals: dict[str, float] = {}
        for raw_name, raw_interval in dict(
            tool_min_start_intervals_s or {}
        ).items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(
                    "tool minimum-start-interval names must be non-empty strings"
                )
            if (
                isinstance(raw_interval, bool)
                or not isinstance(raw_interval, (int, float))
                or not math.isfinite(raw_interval)
                or raw_interval < 0
            ):
                raise ValueError(
                    "tool minimum start intervals must be finite, non-negative numbers"
                )
            min_start_intervals[raw_name] = float(raw_interval)

        self._executor = executor
        self._max_workers = int(max_workers)
        self._max_speculative_workers = int(max_speculative_workers)
        self._min_speculative_workers = int(min_speculative_workers)
        self._max_speculative_pending = int(max_speculative_pending)
        self._ttl_s = float(ttl_s)
        self._clock = clock
        self._ewma_alpha = float(service_ewma_alpha)
        self._service_ewma_s: dict[str, float] = {
            str(name): float(value) for name, value in hints.items()
        }
        self._tool_capacities = capacities
        self._tool_min_start_intervals_s = min_start_intervals
        self._tool_next_eligible_at: dict[str, float] = {}

        # Heaps contain lazy-invalidated (priority, order, generation, id)
        # entries.  A promotion bumps generation and adds an authoritative
        # entry, so the stale speculative entry can never run.
        self._authoritative_queue: list[tuple[float, int, int, int]] = []
        self._speculative_queue: list[tuple[float, int, int, int]] = []
        self._jobs: dict[int, _BrokerJob] = {}
        self._predictions: dict[tuple[str, tuple[str, str]], _BrokerJob] = {}
        self._authoritative_state: list[LiveAuthoritativeResult] = []
        self._tool_records: dict[int, dict[str, Any]] = {}
        self._rejected_records: list[dict[str, Any]] = []
        self._next_job_id = 0
        self._next_order = 0
        self._running_total = 0
        self._running_speculative = 0
        self._running_by_tool: dict[str, int] = {}
        self._dispatch_ordinal_by_tool: dict[str, int] = {}
        # A reserved speculative start may overtake at most one already-queued
        # authoritative call of the same tool.  The next competing start token
        # for that tool must repay the authoritative lane before another
        # reservation is allowed.  This matters when a start-rate gate is more
        # restrictive than the nominal worker/tool concurrency.
        self._reserved_speculative_debt_by_tool: set[str] = set()
        self._available_worker_ids = list(range(self._max_workers))
        heapq.heapify(self._available_worker_ids)
        self._lock = asyncio.Lock()
        self._closed = False
        self._revision = 0
        self._changed = asyncio.Event()
        self._rate_wakeup_task: asyncio.Task[None] | None = None
        self._rate_wakeup_deadline: float | None = None
        self.stats = LiveBrokerStats()

    @property
    def pending_speculative_count(self) -> int:
        """Number of unclaimed speculative jobs retained by the broker."""

        return len(self._predictions)

    @property
    def authoritative_state(self) -> tuple[LiveAuthoritativeResult, ...]:
        """Results that crossed the sole authoritative commit boundary."""

        return tuple(self._authoritative_state)

    @staticmethod
    def _prediction_key(
        session_id: str, invocation: Invocation
    ) -> tuple[str, tuple[str, str]]:
        return session_id, invocation.key

    def _touch_locked(self) -> None:
        self._revision += 1
        self._changed.set()

    def _new_job_locked(
        self,
        invocation: Invocation,
        *,
        session_id: str,
        lane: Lane,
        priority: float,
        originally_speculative: bool,
    ) -> _BrokerJob:
        loop = asyncio.get_running_loop()
        now = self._clock()
        self._next_job_id += 1
        job = _BrokerJob(
            job_id=self._next_job_id,
            invocation=invocation,
            session_id=session_id,
            lane=lane,
            priority=float(priority),
            created_at=now,
            expires_at=now + self._ttl_s if originally_speculative else math.inf,
            future=loop.create_future(),
            originally_speculative=originally_speculative,
        )
        self._jobs[job.job_id] = job
        self._tool_records[job.job_id] = {
            "job_id": job.job_id,
            "invocation_id": f"tool-{job.job_id:08d}",
            "session_id": session_id,
            "tool": invocation.tool_name,
            "invocation_digest": self._invocation_digest(invocation),
            "speculative": originally_speculative,
            "authoritative": lane == "authoritative",
            "admitted": True,
            "queue_enter": now,
            "admitted_at": now,
            "queue_enter_at": now,
            "start": None,
            "started_at": None,
            "confirmation": None,
            "authoritative_confirmation_at": None,
            "finish": None,
            "finished_at": None,
            "outcome": "queued",
            "result_digest": None,
            "exact_match": None,
            "source": lane,
            "cancelled": False,
            "priority": float(priority),
            "speculation_eligible": True,
            "canary": False,
            "worker_id": None,
            "response_status": None,
            "bytes_read": None,
            "backend": None,
            "request_host": None,
            "http_attempts": None,
            "http_attempt_log": None,
            "transport_identity_source": None,
            "queue_s": None,
            "service_s": None,
            "exposed_wait_s": None,
            "saved_service_s": None,
            "committed": False,
            "reserved_speculative_dispatch": False,
            "authoritative_after_reserved_dispatch": False,
            "dispatch_lane": None,
            "dispatch_reason": None,
            "running_speculative_before": None,
            "queued_authoritative_same_tool_before": None,
            "reservation_debt_before": None,
            "reservation_debt_after": None,
            "per_tool_dispatch_ordinal": None,
            "worker_pool": {
                "max_workers": self._max_workers,
                "max_speculative_workers": self._max_speculative_workers,
                "min_speculative_workers": self._min_speculative_workers,
                "max_speculative_pending": self._max_speculative_pending,
                "tool_capacities": dict(sorted(self._tool_capacities.items())),
                "tool_min_start_intervals_s": dict(
                    sorted(self._tool_min_start_intervals_s.items())
                ),
            },
            "tool_capacity": self._tool_capacity(invocation.tool_name),
            "tool_min_start_interval_s": self._tool_min_start_interval(
                invocation.tool_name
            ),
            "rate_limit_eligible_at": self._tool_next_eligible_at.get(
                invocation.tool_name, now
            ),
            "rate_limit_next_eligible_at": self._tool_next_eligible_at.get(
                invocation.tool_name, now
            ),
            "rate_limit_wait_s": None,
        }
        return job

    @staticmethod
    def _invocation_digest(invocation: Invocation) -> str:
        value = f"{invocation.tool_name}\0{invocation.canonical_arguments}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _result_digest(result: Any) -> str:
        try:
            value = json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            value = repr(result)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_success_http_attempt_log(
        raw_log: Any,
        *,
        expected_attempts: Any,
        expected_final_status: Any,
    ) -> list[dict[str, Any]]:
        """Validate executor-provided evidence for a successful HTTP call.

        Attempt numbers are local to ``request_index`` because one logical
        tool call may fan out to several HTTP requests.  The executor's
        aggregate attempt count and final response status remain authoritative
        only when they agree exactly with the physical-attempt log.
        """

        label = "executor _paste_transport.http_attempt_log"
        if not isinstance(raw_log, (list, tuple)) or not raw_log:
            raise ValueError(f"{label} must be a non-empty list")
        if (
            isinstance(expected_attempts, bool)
            or not isinstance(expected_attempts, int)
            or expected_attempts < 1
        ):
            raise ValueError(
                "executor _paste_transport.http_attempts must be a positive "
                "integer when http_attempt_log is present"
            )
        if (
            isinstance(expected_final_status, bool)
            or not isinstance(expected_final_status, int)
            or not 100 <= expected_final_status <= 599
        ):
            raise ValueError(
                "executor _paste_transport.response_status must be an HTTP "
                "status integer when http_attempt_log is present"
            )

        normalized: list[dict[str, Any]] = []
        attempts_by_request: dict[int, list[int]] = {}
        retry_flags_by_request: dict[int, list[bool]] = {}
        for index, raw_entry in enumerate(raw_log):
            entry_label = f"{label}[{index}]"
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"{entry_label} must be an object")
            entry = dict(raw_entry)
            request_index = entry.get("request_index")
            attempt = entry.get("attempt")
            status = entry.get("status")
            error_type = entry.get("error_type")
            retried = entry.get("retried")
            started = entry.get("started_monotonic_s")
            gate_wait = entry.get("start_gate_wait_s")
            retry_backoff = entry.get("retry_backoff_s")
            if (
                isinstance(request_index, bool)
                or not isinstance(request_index, int)
                or request_index < 0
            ):
                raise ValueError(
                    f"{entry_label}.request_index must be a non-negative integer"
                )
            if (
                isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or attempt < 1
            ):
                raise ValueError(
                    f"{entry_label}.attempt must be a positive integer"
                )
            if status is not None and (
                isinstance(status, bool)
                or not isinstance(status, int)
                or not 100 <= status <= 599
            ):
                raise ValueError(
                    f"{entry_label}.status must be null or an HTTP status integer"
                )
            if error_type is not None and not isinstance(error_type, str):
                raise ValueError(f"{entry_label}.error_type must be null or a string")
            if not isinstance(retried, bool):
                raise ValueError(f"{entry_label}.retried must be a boolean")
            if (
                isinstance(started, bool)
                or not isinstance(started, (int, float))
                or not math.isfinite(float(started))
            ):
                raise ValueError(
                    f"{entry_label}.started_monotonic_s must be finite"
                )
            for field_name, field_value in (
                ("start_gate_wait_s", gate_wait),
                ("retry_backoff_s", retry_backoff),
            ):
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, (int, float))
                    or not math.isfinite(float(field_value))
                    or float(field_value) < 0.0
                ):
                    raise ValueError(
                        f"{entry_label}.{field_name} must be finite and non-negative"
                    )
            attempts_by_request.setdefault(request_index, []).append(attempt)
            retry_flags_by_request.setdefault(request_index, []).append(retried)
            normalized.append(entry)

        if len(normalized) != expected_attempts:
            raise ValueError(
                "executor _paste_transport.http_attempts does not match "
                "http_attempt_log length"
            )
        for request_index, attempts in attempts_by_request.items():
            if attempts != list(range(1, len(attempts) + 1)):
                raise ValueError(
                    f"{label} attempts for request_index={request_index} "
                    "must be contiguous from 1"
                )
            retry_flags = retry_flags_by_request[request_index]
            if retry_flags[-1] or any(not value for value in retry_flags[:-1]):
                raise ValueError(
                    f"{label} retried flags for request_index={request_index} "
                    "are inconsistent with its final attempt"
                )
        if normalized[-1].get("status") != expected_final_status:
            raise ValueError(
                "executor _paste_transport.response_status does not match "
                "http_attempt_log final status"
            )
        return normalized

    def _record_rejection_locked(
        self,
        invocation: Invocation,
        *,
        session_id: str,
        priority: float,
        reason: str,
    ) -> None:
        now = self._clock()
        self._rejected_records.append(
            {
                "invocation_id": f"rejected-{len(self._rejected_records) + 1:08d}",
                "job_id": None,
                "session_id": session_id,
                "tool": invocation.tool_name,
                "invocation_digest": self._invocation_digest(invocation),
                "speculative": True,
                "authoritative": False,
                "admitted": False,
                "queue_enter": None,
                "admitted_at": None,
                "queue_enter_at": None,
                "start": None,
                "started_at": None,
                "confirmation": None,
                "authoritative_confirmation_at": None,
                "finish": now,
                "finished_at": now,
                "outcome": reason,
                "result_digest": None,
                "exact_match": None,
                "source": "speculative",
                "cancelled": False,
                "priority": float(priority),
                "speculation_eligible": True,
                "canary": False,
                "worker_id": None,
                "response_status": None,
                "bytes_read": None,
                "backend": None,
                "request_host": None,
                "http_attempts": None,
                "http_attempt_log": None,
                "transport_identity_source": None,
                "queue_s": None,
                "service_s": None,
                "exposed_wait_s": None,
                "saved_service_s": None,
                "committed": False,
                "reserved_speculative_dispatch": False,
                "authoritative_after_reserved_dispatch": False,
                "dispatch_lane": None,
                "dispatch_reason": None,
                "running_speculative_before": None,
                "queued_authoritative_same_tool_before": None,
                "reservation_debt_before": None,
                "reservation_debt_after": None,
                "per_tool_dispatch_ordinal": None,
                "worker_pool": {
                    "max_workers": self._max_workers,
                    "max_speculative_workers": self._max_speculative_workers,
                    "min_speculative_workers": self._min_speculative_workers,
                    "max_speculative_pending": self._max_speculative_pending,
                    "tool_capacities": dict(sorted(self._tool_capacities.items())),
                    "tool_min_start_intervals_s": dict(
                        sorted(self._tool_min_start_intervals_s.items())
                    ),
                },
                "tool_capacity": self._tool_capacity(invocation.tool_name),
                "tool_min_start_interval_s": self._tool_min_start_interval(
                    invocation.tool_name
                ),
                "rate_limit_eligible_at": self._tool_next_eligible_at.get(
                    invocation.tool_name, now
                ),
                "rate_limit_next_eligible_at": self._tool_next_eligible_at.get(
                    invocation.tool_name, now
                ),
                "rate_limit_wait_s": None,
            }
        )

    def _finalize_never_started_cancellation_locked(
        self,
        job: _BrokerJob,
        telemetry: dict[str, Any],
    ) -> None:
        """Finalize one admitted job cancelled before worker dispatch.

        Such a row is a physical broker job, but not an HTTP attempt.  Keep
        transport identity absent, retain ``start=None``, account its entire
        lifetime as queue time, and make the zero service/attempt semantics
        explicit.  This helper must never touch a job that reached a worker;
        started cancellations retain the executor's planned/actual attempt
        evidence.
        """

        if job.started_at is not None or telemetry.get("started_at") is not None:
            raise RuntimeError("cannot normalize a started job as pre-start cancelled")
        finish = telemetry.get("finished_at")
        if finish is None:
            finish = self._clock()
            telemetry["finish"] = finish
            telemetry["finished_at"] = finish
        queue_enter = telemetry.get("queue_enter_at")
        if not isinstance(queue_enter, (int, float)) or isinstance(queue_enter, bool):
            raise RuntimeError("pre-start cancellation lacks queue-entry telemetry")
        telemetry["start"] = None
        telemetry["started_at"] = None
        telemetry["queue_s"] = max(0.0, float(finish) - float(queue_enter))
        telemetry["service_s"] = 0.0
        telemetry["saved_service_s"] = 0.0
        telemetry["worker_id"] = None
        telemetry["response_status"] = None
        telemetry["bytes_read"] = None
        telemetry["backend"] = None
        telemetry["request_host"] = None
        telemetry["http_attempts"] = 0
        telemetry["transport_identity_source"] = None

    def _tool_capacity(self, tool_name: str) -> int:
        return self._tool_capacities.get(tool_name, self._max_workers)

    def _tool_min_start_interval(self, tool_name: str) -> float:
        return self._tool_min_start_intervals_s.get(tool_name, 0.0)

    def _tool_rate_eligible_locked(
        self, tool_name: str, *, now: float | None = None
    ) -> bool:
        if now is None:
            now = self._clock()
        return now >= self._tool_next_eligible_at.get(tool_name, -math.inf)

    def _tool_has_capacity_locked(self, tool_name: str) -> bool:
        return self._running_by_tool.get(tool_name, 0) < self._tool_capacity(
            tool_name
        )

    def _push_locked(self, job: _BrokerJob) -> None:
        self._next_order += 1
        job.queue_order = self._next_order
        # Larger utility means earlier speculative dispatch.  Authoritative
        # calls are FIFO because correctness traffic should not be reordered
        # by an untrusted speculation score.
        sort_priority = 0.0 if job.lane == "authoritative" else -job.priority
        item = (sort_priority, self._next_order, job.generation, job.job_id)
        if job.lane == "authoritative":
            heapq.heappush(self._authoritative_queue, item)
        else:
            heapq.heappush(self._speculative_queue, item)

    def _pop_dispatchable_locked(
        self, lane: Lane, *, skip_tools: set[str] | None = None
    ) -> _BrokerJob | None:
        heap = (
            self._authoritative_queue
            if lane == "authoritative"
            else self._speculative_queue
        )
        blocked: list[tuple[float, int, int, int]] = []
        while heap:
            item = heapq.heappop(heap)
            _, _, generation, job_id = item
            job = self._jobs.get(job_id)
            if (
                job is not None
                and job.state == "queued"
                and job.lane == lane
                and job.generation == generation
            ):
                tool_name = job.invocation.tool_name
                if skip_tools and tool_name in skip_tools:
                    blocked.append(item)
                    continue
                if self._tool_has_capacity_locked(
                    tool_name
                ) and self._tool_rate_eligible_locked(tool_name):
                    for blocked_item in blocked:
                        heapq.heappush(heap, blocked_item)
                    return job
                blocked.append(item)
        for blocked_item in blocked:
            heapq.heappush(heap, blocked_item)
        return None

    def _dispatchable_authoritative_tools_locked(self) -> set[str]:
        tools: set[str] = set()
        for job in self._jobs.values():
            if job.state != "queued" or job.lane != "authoritative":
                continue
            tool_name = job.invocation.tool_name
            if self._tool_has_capacity_locked(
                tool_name
            ) and self._tool_rate_eligible_locked(tool_name):
                tools.add(tool_name)
        return tools

    def _valid_queue_count_for_tool_locked(self, lane: Lane, tool_name: str) -> int:
        return sum(
            1
            for job in self._jobs.values()
            if job.state == "queued"
            and job.lane == lane
            and job.invocation.tool_name == tool_name
        )

    def _valid_queue_count_locked(self, lane: Lane) -> int:
        return sum(
            1
            for job in self._jobs.values()
            if job.state == "queued" and job.lane == lane
        )

    def _update_high_watermarks_locked(self) -> None:
        self.stats.max_queued_authoritative = max(
            self.stats.max_queued_authoritative,
            self._valid_queue_count_locked("authoritative"),
        )
        self.stats.max_queued_speculative = max(
            self.stats.max_queued_speculative,
            self._valid_queue_count_locked("speculative"),
        )
        self.stats.max_running_total = max(
            self.stats.max_running_total, self._running_total
        )
        self.stats.max_running_speculative = max(
            self.stats.max_running_speculative, self._running_speculative
        )
        queued_by_tool: dict[str, int] = {}
        for job in self._jobs.values():
            if job.state == "queued":
                name = job.invocation.tool_name
                queued_by_tool[name] = queued_by_tool.get(name, 0) + 1
        for name, count in queued_by_tool.items():
            self.stats.max_queued_by_tool[name] = max(
                self.stats.max_queued_by_tool.get(name, 0), count
            )
        for name, count in self._running_by_tool.items():
            self.stats.max_running_by_tool[name] = max(
                self.stats.max_running_by_tool.get(name, 0), count
            )

    def _earliest_rate_wakeup_locked(self) -> float | None:
        now = self._clock()
        deadlines = [
            self._tool_next_eligible_at.get(job.invocation.tool_name, now)
            for job in self._jobs.values()
            if job.state == "queued"
            and self._tool_min_start_interval(job.invocation.tool_name) > 0.0
            and self._tool_next_eligible_at.get(job.invocation.tool_name, now) > now
        ]
        return min(deadlines) if deadlines else None

    def _arm_rate_wakeup_locked(self) -> None:
        """Arrange one future dispatch without consuming a worker slot."""

        if self._closed:
            return
        deadline = self._earliest_rate_wakeup_locked()
        existing = self._rate_wakeup_task
        if deadline is None:
            if existing is not None and not existing.done():
                existing.cancel()
            self._rate_wakeup_task = None
            self._rate_wakeup_deadline = None
            return
        if (
            existing is not None
            and not existing.done()
            and self._rate_wakeup_deadline is not None
            and self._rate_wakeup_deadline <= deadline
        ):
            return
        if existing is not None and not existing.done():
            existing.cancel()
        self._rate_wakeup_deadline = deadline
        self._rate_wakeup_task = asyncio.create_task(
            self._wake_for_rate_limit(deadline)
        )

    async def _wake_for_rate_limit(self, deadline: float) -> None:
        try:
            await asyncio.sleep(max(0.0, deadline - self._clock()))
        except asyncio.CancelledError:
            return
        async with self._lock:
            if asyncio.current_task() is not self._rate_wakeup_task:
                return
            self._rate_wakeup_task = None
            self._rate_wakeup_deadline = None
            if self._closed:
                return
            self._dispatch_locked()
            self._touch_locked()

    def _dispatch_locked(self) -> None:
        if self._closed:
            return
        while self._running_total < self._max_workers:
            job = None
            minimum_speculative_dispatch = False
            if self._running_speculative < self._min_speculative_workers:
                dispatchable_authoritative_tools = (
                    self._dispatchable_authoritative_tools_locked()
                )
                repay_tools = (
                    self._reserved_speculative_debt_by_tool
                    & dispatchable_authoritative_tools
                )
                job = self._pop_dispatchable_locked(
                    "speculative", skip_tools=repay_tools
                )
                minimum_speculative_dispatch = job is not None
            if job is None:
                job = self._pop_dispatchable_locked("authoritative")
            if job is None and (
                self._running_speculative < self._max_speculative_workers
            ):
                job = self._pop_dispatchable_locked("speculative")
            if job is None:
                break

            tool_name = job.invocation.tool_name
            running_speculative_before = self._running_speculative
            queued_authoritative_same_tool_before = (
                self._valid_queue_count_for_tool_locked(
                    "authoritative", tool_name
                )
            )
            reservation_debt_before = (
                tool_name in self._reserved_speculative_debt_by_tool
            )
            competing_authoritative = (
                minimum_speculative_dispatch
                and queued_authoritative_same_tool_before > 0
            )
            authoritative_repayment = (
                job.lane == "authoritative"
                and reservation_debt_before
            )
            if competing_authoritative and reservation_debt_before:
                raise RuntimeError(
                    "reserved speculative dispatch attempted before repayment"
                )
            if competing_authoritative:
                self._reserved_speculative_debt_by_tool.add(tool_name)
                self.stats.reserved_speculative_dispatches += 1
            if authoritative_repayment:
                self._reserved_speculative_debt_by_tool.discard(tool_name)
                self.stats.authoritative_after_reserved_dispatches += 1
            reservation_debt_after = (
                tool_name in self._reserved_speculative_debt_by_tool
            )
            if competing_authoritative:
                dispatch_reason = "reserved_speculative"
            elif minimum_speculative_dispatch:
                dispatch_reason = "speculative_minimum_uncontended"
            elif authoritative_repayment:
                dispatch_reason = "authoritative_repayment"
            elif job.lane == "authoritative":
                dispatch_reason = "authoritative_priority"
            else:
                dispatch_reason = "speculative_opportunistic"

            job.state = "running"
            job.started_at = self._clock()
            job.worker_id = heapq.heappop(self._available_worker_ids)
            per_tool_dispatch_ordinal = (
                self._dispatch_ordinal_by_tool.get(tool_name, 0) + 1
            )
            self._dispatch_ordinal_by_tool[tool_name] = per_tool_dispatch_ordinal
            telemetry = self._tool_records[job.job_id]
            telemetry["reserved_speculative_dispatch"] = competing_authoritative
            telemetry["authoritative_after_reserved_dispatch"] = (
                authoritative_repayment
            )
            telemetry["dispatch_lane"] = job.lane
            telemetry["dispatch_reason"] = dispatch_reason
            telemetry["running_speculative_before"] = (
                running_speculative_before
            )
            telemetry["queued_authoritative_same_tool_before"] = (
                queued_authoritative_same_tool_before
            )
            telemetry["reservation_debt_before"] = reservation_debt_before
            telemetry["reservation_debt_after"] = reservation_debt_after
            telemetry["per_tool_dispatch_ordinal"] = (
                per_tool_dispatch_ordinal
            )
            interval_s = self._tool_min_start_interval(tool_name)
            eligible_at = self._tool_next_eligible_at.get(
                tool_name, job.started_at
            )
            next_eligible_at = job.started_at + interval_s
            self._tool_next_eligible_at[tool_name] = next_eligible_at
            telemetry["start"] = job.started_at
            telemetry["started_at"] = job.started_at
            telemetry["worker_id"] = job.worker_id
            telemetry["queue_s"] = max(0.0, job.started_at - job.created_at)
            telemetry["outcome"] = "running"
            telemetry["authoritative"] = job.lane == "authoritative"
            telemetry["rate_limit_eligible_at"] = eligible_at
            telemetry["rate_limit_next_eligible_at"] = next_eligible_at
            telemetry["rate_limit_wait_s"] = max(
                0.0, min(job.started_at, eligible_at) - job.created_at
            )
            planner = getattr(self._executor, "transport_plan", None)
            if callable(planner):
                try:
                    planned = planner(job.invocation)
                except Exception:
                    planned = None
                if isinstance(planned, Mapping):
                    planned_backend = planned.get("backend")
                    planned_host = planned.get("request_host")
                    planned_attempts = planned.get("http_attempts")
                    if (
                        isinstance(planned_backend, str)
                        and planned_backend
                        and isinstance(planned_host, str)
                        and planned_host
                        and isinstance(planned_attempts, int)
                        and not isinstance(planned_attempts, bool)
                        and planned_attempts >= 1
                    ):
                        telemetry["backend"] = planned_backend
                        telemetry["request_host"] = planned_host
                        telemetry["http_attempts"] = planned_attempts
                        telemetry["transport_identity_source"] = "planned"
            self._running_total += 1
            self._running_by_tool[tool_name] = (
                self._running_by_tool.get(tool_name, 0) + 1
            )
            if job.lane == "speculative":
                self._running_speculative += 1
                self.stats.speculative_started += 1
            else:
                self.stats.authoritative_started += 1
            job.runner = asyncio.create_task(self._run(job))
            self._touch_locked()
        self._update_high_watermarks_locked()
        self._arm_rate_wakeup_locked()

    async def _run(self, job: _BrokerJob) -> None:
        result: Any = None
        response_status: int | None = None
        bytes_read: int | None = None
        backend: str | None = None
        request_host: str | None = None
        http_attempts: int | None = None
        http_attempt_log: list[dict[str, Any]] | None = None
        transport_identity_source: str | None = None
        error: BaseException | None = None
        try:
            result = await self._executor(job.invocation)
            if isinstance(result, Mapping) and "_paste_transport" in result:
                transport = result.get("_paste_transport")
                result = dict(result)
                result.pop("_paste_transport", None)
                if isinstance(transport, Mapping):
                    has_attempt_log = "http_attempt_log" in transport
                    raw_status = transport.get("response_status")
                    raw_bytes = transport.get("bytes_read")
                    raw_backend = transport.get("backend")
                    raw_host = transport.get("request_host")
                    raw_attempts = transport.get("http_attempts")
                    if isinstance(raw_status, int):
                        response_status = raw_status
                    if isinstance(raw_bytes, int):
                        bytes_read = raw_bytes
                    if isinstance(raw_backend, str):
                        backend = raw_backend
                    if isinstance(raw_host, str):
                        request_host = raw_host
                    if isinstance(raw_attempts, int):
                        http_attempts = raw_attempts
                    if has_attempt_log:
                        http_attempt_log = self._normalize_success_http_attempt_log(
                            transport.get("http_attempt_log"),
                            expected_attempts=raw_attempts,
                            expected_final_status=raw_status,
                        )
                        # These assignments make the record derive both scalar
                        # values from the already cross-checked physical log.
                        http_attempts = len(http_attempt_log)
                        response_status = http_attempt_log[-1]["status"]
                    transport_identity_source = "actual"
        except asyncio.CancelledError as exc:
            error = exc
        except BaseException as exc:  # preserve executor failure for clean fallback
            error = exc
            raw_log = getattr(exc, "paste_http_attempt_log", None)
            if isinstance(raw_log, (list, tuple)) and raw_log:
                normalized_log: list[dict[str, Any]] = []
                for entry in raw_log:
                    if not isinstance(entry, Mapping):
                        normalized_log = []
                        break
                    attempt = entry.get("attempt")
                    if (
                        isinstance(attempt, bool)
                        or not isinstance(attempt, int)
                        or attempt < 1
                    ):
                        normalized_log = []
                        break
                    normalized_log.append(dict(entry))
                if normalized_log:
                    http_attempt_log = normalized_log
                    http_attempts = len(normalized_log)
                    final_status = normalized_log[-1].get("status")
                    if isinstance(final_status, int) and not isinstance(
                        final_status, bool
                    ):
                        response_status = final_status
                    transport_identity_source = "actual_failure"

        finished_at = self._clock()
        async with self._lock:
            was_speculative_lane = job.lane == "speculative"
            self._running_total -= 1
            tool_name = job.invocation.tool_name
            remaining_for_tool = self._running_by_tool.get(tool_name, 0) - 1
            if remaining_for_tool > 0:
                self._running_by_tool[tool_name] = remaining_for_tool
            else:
                self._running_by_tool.pop(tool_name, None)
            if was_speculative_lane:
                self._running_speculative -= 1
            job.finished_at = finished_at
            if job.worker_id is not None:
                heapq.heappush(self._available_worker_ids, job.worker_id)
            telemetry = self._tool_records[job.job_id]
            telemetry["finish"] = finished_at
            telemetry["finished_at"] = finished_at
            telemetry["service_s"] = max(
                0.0, finished_at - (job.started_at or finished_at)
            )
            if response_status is not None:
                telemetry["response_status"] = response_status
            if bytes_read is not None:
                telemetry["bytes_read"] = bytes_read
            if backend is not None:
                telemetry["backend"] = backend
            if request_host is not None:
                telemetry["request_host"] = request_host
            if http_attempts is not None:
                telemetry["http_attempts"] = http_attempts
            if http_attempt_log is not None:
                telemetry["http_attempt_log"] = http_attempt_log
            if transport_identity_source is not None:
                telemetry["transport_identity_source"] = transport_identity_source

            if job.state in {"cancelled", "cancelling"}:
                job.state = "cancelled"
                telemetry["outcome"] = "cancelled"
                telemetry["cancelled"] = True
                if job.originally_speculative:
                    self.stats.wasted_speculative_service_s += max(
                        0.0, finished_at - (job.started_at or finished_at)
                    )
                if not job.future.done():
                    job.future.cancel()
            else:
                record = _ExecutionRecord(
                    result=result,
                    error=error,
                    started_at=job.started_at or finished_at,
                    finished_at=finished_at,
                )
                if error is None:
                    job.state = "completed"
                    telemetry["outcome"] = "completed"
                    telemetry["result_digest"] = self._result_digest(result)
                    if job.originally_speculative:
                        self.stats.speculative_completed += 1
                    if job.lane == "authoritative":
                        self.stats.authoritative_completed += 1
                else:
                    job.state = "failed"
                    telemetry["outcome"] = "failed"
                    if job.originally_speculative:
                        self.stats.speculative_failures += 1
                    if job.lane == "authoritative":
                        self.stats.authoritative_failures += 1
                if not job.future.done():
                    job.future.set_result(record)

            service_s = max(0.0, finished_at - (job.started_at or finished_at))
            if error is None and service_s > 0:
                old = self._service_ewma_s.get(job.invocation.tool_name)
                self._service_ewma_s[job.invocation.tool_name] = (
                    service_s
                    if old is None
                    else self._ewma_alpha * service_s + (1.0 - self._ewma_alpha) * old
                )
            self._dispatch_locked()
            self._touch_locked()

    async def _expire_after(self, job: _BrokerJob) -> None:
        try:
            delay = max(0.0, job.expires_at - self._clock())
            await asyncio.sleep(delay)
            await self._discard_prediction(job, expired=True)
        except asyncio.CancelledError:
            return

    async def speculate(
        self,
        invocation: Invocation,
        *,
        session_id: str = "default",
        priority: float = 0.0,
    ) -> bool:
        """Queue one isolated prediction.

        ``priority`` is a caller-provided expected-utility score used only
        within the speculative lane.  It cannot overtake authoritative work
        unless the explicit one-worker reservation is enabled; after such an
        overtake, the next competing same-tool start is authoritative.
        """

        if not math.isfinite(priority):
            raise ValueError("priority must be finite")
        await self.sweep()
        key = self._prediction_key(session_id, invocation)
        async with self._lock:
            if self._closed:
                raise RuntimeError("broker is closed")
            if key in self._predictions:
                self.stats.duplicate_predictions += 1
                self._record_rejection_locked(
                    invocation,
                    session_id=session_id,
                    priority=priority,
                    reason="duplicate",
                )
                return False
            if len(self._predictions) >= self._max_speculative_pending:
                self.stats.rejected_speculative_capacity += 1
                self._record_rejection_locked(
                    invocation,
                    session_id=session_id,
                    priority=priority,
                    reason="rejected_capacity",
                )
                return False

            job = self._new_job_locked(
                invocation,
                session_id=session_id,
                lane="speculative",
                priority=priority,
                originally_speculative=True,
            )
            self._predictions[key] = job
            self._push_locked(job)
            job.expiry_task = asyncio.create_task(self._expire_after(job))
            self.stats.speculative_admitted += 1
            self._dispatch_locked()
            self._touch_locked()
            return True

    async def authoritative(
        self,
        invocation: Invocation,
        *,
        session_id: str = "default",
        speculation_eligible: bool = True,
    ) -> LiveAuthoritativeResult:
        """Execute and commit one exact authoritative invocation.

        Set ``speculation_eligible=False`` for an unbiased canary call.  Such a
        call never consumes a prediction even if an exact one happens to be
        present.
        """

        await self.sweep()
        confirmation_at = self._clock()
        wait_started = confirmation_at
        key = self._prediction_key(session_id, invocation)

        async with self._lock:
            if self._closed:
                raise RuntimeError("broker is closed")
            self.stats.authoritative_requests += 1
            # Pop makes every prediction single-use even if duplicate
            # authoritative calls for the same key arrive concurrently.
            job = (
                self._predictions.pop(key, None)
                if speculation_eligible
                else None
            )
            source: str
            failed_prediction = job is not None and job.state == "failed"
            if job is None or job.state in {"cancelled", "failed"}:
                if job is not None:
                    failed_telemetry = self._tool_records[job.job_id]
                    failed_telemetry["source"] = "failed_speculation_uncommitted"
                    failed_telemetry["confirmation"] = confirmation_at
                    failed_telemetry["authoritative_confirmation_at"] = (
                        confirmation_at
                    )
                    failed_telemetry["authoritative"] = True
                    failed_telemetry["exact_match"] = failed_prediction
                    self._jobs.pop(job.job_id, None)
                self.stats.authoritative_misses += 1
                self.stats.authoritative_executions += 1
                job = self._new_job_locked(
                    invocation,
                    session_id=session_id,
                    lane="authoritative",
                    priority=0.0,
                    originally_speculative=False,
                )
                source = (
                    "executed_after_speculative_failure"
                    if failed_prediction
                    else "executed"
                )
                self._tool_records[job.job_id]["source"] = source
                self._tool_records[job.job_id]["confirmation"] = confirmation_at
                self._tool_records[job.job_id]["authoritative_confirmation_at"] = (
                    confirmation_at
                )
                self._tool_records[job.job_id]["speculation_eligible"] = (
                    speculation_eligible
                )
                self._tool_records[job.job_id]["canary"] = not speculation_eligible
                self._push_locked(job)
            else:
                job.confirmed_at = confirmation_at
                telemetry = self._tool_records[job.job_id]
                telemetry["confirmation"] = confirmation_at
                telemetry["authoritative_confirmation_at"] = confirmation_at
                telemetry["authoritative"] = True
                telemetry["exact_match"] = True
                if job.expiry_task is not None:
                    job.expiry_task.cancel()
                    job.expiry_task = None
                if job.state == "queued":
                    job.lane = "authoritative"
                    job.expires_at = math.inf
                    job.generation += 1
                    self._push_locked(job)
                    self.stats.queued_promotions += 1
                    source = "promoted_from_queue"
                elif job.state == "running":
                    # It already consumes a shared worker; changing lanes here
                    # would corrupt speculative-running accounting.
                    self.stats.running_promotions += 1
                    source = "promoted_inflight"
                else:
                    self.stats.completed_reuse += 1
                    source = "reused"
                telemetry["source"] = source
            self._dispatch_locked()
            self._touch_locked()

        try:
            record = await asyncio.shield(job.future)
        except asyncio.CancelledError:
            await self._cancel_job(job, expired=False)
            raise

        if record.error is not None:
            if job.originally_speculative:
                # A failed speculative attempt is never committed.  Submit a
                # fresh correctness-critical call through the authoritative
                # lane of the same shared pool.
                async with self._lock:
                    self._tool_records[job.job_id]["source"] = (
                        "failed_speculation_uncommitted"
                    )
                    self._jobs.pop(job.job_id, None)
                    self.stats.authoritative_misses += 1
                    self.stats.authoritative_executions += 1
                    fallback = self._new_job_locked(
                        invocation,
                        session_id=session_id,
                        lane="authoritative",
                        priority=0.0,
                        originally_speculative=False,
                    )
                    fallback_telemetry = self._tool_records[fallback.job_id]
                    fallback_telemetry["confirmation"] = confirmation_at
                    fallback_telemetry["authoritative_confirmation_at"] = (
                        confirmation_at
                    )
                    fallback_telemetry["source"] = (
                        "executed_after_speculative_failure"
                    )
                    self._push_locked(fallback)
                    self._dispatch_locked()
                    self._touch_locked()
                fallback_record = await asyncio.shield(fallback.future)
                if fallback_record.error is not None:
                    async with self._lock:
                        self._jobs.pop(fallback.job_id, None)
                    raise fallback_record.error
                job = fallback
                record = fallback_record
                source = "executed_after_speculative_failure"
            else:
                async with self._lock:
                    self._jobs.pop(job.job_id, None)
                raise record.error

        finished_at = record.finished_at
        service_s = max(0.0, finished_at - record.started_at)
        saved_service_s = 0.0
        if job.originally_speculative:
            saved_service_s = max(
                0.0,
                min(confirmation_at, finished_at) - record.started_at,
            )
            saved_service_s = min(service_s, saved_service_s)
            self.stats.saved_service_s += saved_service_s

        committed = LiveAuthoritativeResult(
            invocation=invocation,
            result=record.result,
            source=source,
            exposed_wait_s=max(0.0, self._clock() - wait_started),
            queue_s=max(0.0, record.started_at - job.created_at),
            service_s=service_s,
            saved_service_s=saved_service_s,
        )
        async with self._lock:
            telemetry = self._tool_records[job.job_id]
            telemetry["source"] = source
            telemetry["authoritative"] = True
            telemetry["exact_match"] = job.originally_speculative
            telemetry["exposed_wait_s"] = committed.exposed_wait_s
            telemetry["saved_service_s"] = committed.saved_service_s
            telemetry["committed"] = True
            telemetry["outcome"] = "committed"
            self._jobs.pop(job.job_id, None)
            self._authoritative_state.append(committed)
            self.stats.commits += 1
            self._touch_locked()
        return committed

    async def _discard_prediction(
        self, job: _BrokerJob, *, expired: bool
    ) -> bool:
        runner: asyncio.Task[None] | None = None
        async with self._lock:
            key = self._prediction_key(job.session_id, job.invocation)
            if self._predictions.get(key) is not job:
                return False
            self._predictions.pop(key, None)
            if expired:
                self.stats.speculative_expired += 1
            else:
                self.stats.speculative_cancelled += 1
            telemetry = self._tool_records[job.job_id]
            telemetry["cancelled"] = True
            telemetry["outcome"] = "expired" if expired else "cancelled"
            telemetry["source"] = "expired" if expired else "cancelled"
            if job.state != "running" and telemetry["finish"] is None:
                telemetry["finish"] = self._clock()
                telemetry["finished_at"] = telemetry["finish"]
            if job.started_at is None:
                self._finalize_never_started_cancellation_locked(job, telemetry)

            if job.started_at is not None and job.state in {"completed", "failed"}:
                end = job.finished_at if job.finished_at is not None else self._clock()
                self.stats.wasted_speculative_service_s += max(
                    0.0, end - job.started_at
                )
            if job.state in {"queued", "running"}:
                job.state = "cancelling" if job.state == "running" else "cancelled"
                job.generation += 1
                runner = job.runner
                if runner is None and not job.future.done():
                    job.future.cancel()
            if runner is None:
                self._jobs.pop(job.job_id, None)
            self._dispatch_locked()
            self._touch_locked()

        current = asyncio.current_task()
        if runner is not None and runner is not current:
            if not runner.done():
                runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
            async with self._lock:
                self._jobs.pop(job.job_id, None)
                self._touch_locked()
        expiry_task = job.expiry_task
        if (
            expiry_task is not None
            and expiry_task is not current
            and not expiry_task.done()
        ):
            expiry_task.cancel()
            await asyncio.gather(expiry_task, return_exceptions=True)
        return True

    async def _cancel_job(self, job: _BrokerJob, *, expired: bool) -> bool:
        # Confirmed/direct authoritative jobs are absent from _predictions, so
        # cancellation must address them directly.
        key = self._prediction_key(job.session_id, job.invocation)
        if self._predictions.get(key) is job:
            return await self._discard_prediction(job, expired=expired)

        runner: asyncio.Task[None] | None = None
        async with self._lock:
            if job.job_id not in self._jobs:
                return False
            telemetry = self._tool_records[job.job_id]
            telemetry["cancelled"] = True
            telemetry["outcome"] = "cancelled"
            telemetry["source"] = "cancelled"
            if job.state != "running" and telemetry["finish"] is None:
                telemetry["finish"] = self._clock()
                telemetry["finished_at"] = telemetry["finish"]
            if job.started_at is None:
                self._finalize_never_started_cancellation_locked(job, telemetry)
            if job.state in {"queued", "running"}:
                job.state = "cancelling" if job.state == "running" else "cancelled"
                job.generation += 1
                runner = job.runner
                if runner is None and not job.future.done():
                    job.future.cancel()
            if runner is None:
                self._jobs.pop(job.job_id, None)
            self._dispatch_locked()
            self._touch_locked()
        if runner is not None:
            if not runner.done():
                runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
            async with self._lock:
                self._jobs.pop(job.job_id, None)
                self._touch_locked()
        return True

    async def cancel_predictions(
        self,
        *,
        session_id: str | None = None,
        keep: Invocation | None = None,
    ) -> int:
        """Cancel unconfirmed predictions, optionally within one session."""

        async with self._lock:
            selected = [
                job
                for job in self._predictions.values()
                if (session_id is None or job.session_id == session_id)
                and (keep is None or job.invocation.key != keep.key)
            ]
        cancelled = 0
        for job in selected:
            cancelled += int(await self._discard_prediction(job, expired=False))
        return cancelled

    async def sweep(self, *, now: float | None = None) -> int:
        """Expire predictions whose TTL elapsed (also useful with fake clocks)."""

        cutoff = self._clock() if now is None else float(now)
        async with self._lock:
            expired = [
                job
                for job in self._predictions.values()
                if job.expires_at <= cutoff or math.isinf(cutoff)
            ]
        removed = 0
        for job in expired:
            removed += int(await self._discard_prediction(job, expired=True))
        async with self._lock:
            self._dispatch_locked()
        return removed

    def snapshot(self, *, session_id: str | None = None) -> dict[str, Any]:
        """Return result-free live queue state for an LLM-side controller."""

        now = self._clock()
        all_jobs = list(self._jobs.values())
        jobs = [
            job
            for job in all_jobs
            if session_id is None or job.session_id == session_id
        ]
        queued_authoritative = sorted(
            (
                job
                for job in all_jobs
                if job.state == "queued" and job.lane == "authoritative"
            ),
            key=lambda item: item.queue_order,
        )
        queued_speculative = sorted(
            (
                job
                for job in all_jobs
                if job.state == "queued" and job.lane == "speculative"
            ),
            key=lambda item: (-item.priority, item.queue_order),
        )
        authoritative_positions = {
            job.job_id: index for index, job in enumerate(queued_authoritative)
        }
        speculative_positions = {
            job.job_id: index for index, job in enumerate(queued_speculative)
        }

        def service_estimate(job: _BrokerJob) -> float | None:
            return self._service_ewma_s.get(job.invocation.tool_name)

        running_work_s = 0.0
        running_work_by_tool_s: dict[str, float] = {}
        for running_job in all_jobs:
            if running_job.state not in {"running", "cancelling"}:
                continue
            estimate = service_estimate(running_job)
            if estimate is None:
                continue
            elapsed = max(0.0, now - (running_job.started_at or now))
            remaining = max(0.0, estimate - elapsed)
            running_work_s += remaining
            tool_name = running_job.invocation.tool_name
            running_work_by_tool_s[tool_name] = (
                running_work_by_tool_s.get(tool_name, 0.0) + remaining
            )

        authoritative_prefix_work_s: dict[int, float] = {}
        authoritative_tool_prefix_work_s: dict[int, float] = {}
        authoritative_tool_positions: dict[int, int] = {}
        prefix_work = 0.0
        tool_prefix_work: dict[str, float] = {}
        tool_prefix_count: dict[str, int] = {}
        for queued_job in queued_authoritative:
            authoritative_prefix_work_s[queued_job.job_id] = prefix_work
            tool_name = queued_job.invocation.tool_name
            authoritative_tool_prefix_work_s[queued_job.job_id] = (
                tool_prefix_work.get(tool_name, 0.0)
            )
            authoritative_tool_positions[queued_job.job_id] = (
                tool_prefix_count.get(tool_name, 0)
            )
            work = service_estimate(queued_job) or 0.0
            prefix_work += work
            tool_prefix_work[tool_name] = tool_prefix_work.get(tool_name, 0.0) + work
            tool_prefix_count[tool_name] = tool_prefix_count.get(tool_name, 0) + 1
        all_authoritative_queue_work_s = prefix_work
        all_authoritative_tool_queue_work_s = dict(tool_prefix_work)
        all_authoritative_tool_queue_count = dict(tool_prefix_count)

        speculative_prefix_work_s: dict[int, float] = {}
        speculative_tool_prefix_work_s: dict[int, float] = {}
        speculative_tool_positions: dict[int, int] = {}
        prefix_work = 0.0
        tool_prefix_work = {}
        tool_prefix_count = {}
        for queued_job in queued_speculative:
            speculative_prefix_work_s[queued_job.job_id] = prefix_work
            tool_name = queued_job.invocation.tool_name
            speculative_tool_prefix_work_s[queued_job.job_id] = (
                tool_prefix_work.get(tool_name, 0.0)
            )
            speculative_tool_positions[queued_job.job_id] = (
                all_authoritative_tool_queue_count.get(tool_name, 0)
                + tool_prefix_count.get(tool_name, 0)
            )
            work = service_estimate(queued_job) or 0.0
            prefix_work += work
            tool_prefix_work[tool_name] = tool_prefix_work.get(tool_name, 0.0) + work
            tool_prefix_count[tool_name] = tool_prefix_count.get(tool_name, 0) + 1

        # A positive start-rate interval serializes starts even when the tool
        # has multiple execution slots.  For those tools, model the same
        # bounded spec/auth turn-taking used by dispatch rather than the legacy
        # "all authoritative, then all speculative" order.  This keeps nw/rtw
        # metadata consistent with the physical broker policy.
        reservation_tool_positions: dict[int, int] = {}
        reservation_tool_prefix_work_s: dict[int, float] = {}
        reservation_global_prefix_work_s: dict[int, float] = {}
        if self._min_speculative_workers > 0:
            queued_tools = sorted(
                {
                    job.invocation.tool_name
                    for job in queued_authoritative + queued_speculative
                    if self._tool_min_start_interval(job.invocation.tool_name) > 0.0
                }
            )
            for queued_tool in queued_tools:
                auth_jobs = [
                    job
                    for job in queued_authoritative
                    if job.invocation.tool_name == queued_tool
                ]
                spec_jobs = [
                    job
                    for job in queued_speculative
                    if job.invocation.tool_name == queued_tool
                ]
                merged: list[_BrokerJob] = []
                auth_index = 0
                spec_index = 0
                reservation_blocked_by_running_speculative = (
                    self._running_speculative
                    >= self._min_speculative_workers
                )
                # Do not assume that currently running speculative work will
                # finish between projected starts.  Until a later snapshot
                # observes that slot drain, keep every queued authoritative
                # job ahead of another reserved turn.  This is conservative
                # rather than a future service-time oracle.
                speculative_turn = (
                    not reservation_blocked_by_running_speculative
                    and queued_tool
                    not in self._reserved_speculative_debt_by_tool
                )
                while auth_index < len(auth_jobs) or spec_index < len(spec_jobs):
                    if speculative_turn and spec_index < len(spec_jobs):
                        merged.append(spec_jobs[spec_index])
                        spec_index += 1
                        speculative_turn = False
                    elif auth_index < len(auth_jobs):
                        merged.append(auth_jobs[auth_index])
                        auth_index += 1
                        speculative_turn = (
                            not reservation_blocked_by_running_speculative
                        )
                    else:
                        merged.append(spec_jobs[spec_index])
                        spec_index += 1
                merged_prefix = 0.0
                for merged_index, merged_job in enumerate(merged):
                    reservation_tool_positions[merged_job.job_id] = merged_index
                    reservation_tool_prefix_work_s[merged_job.job_id] = merged_prefix
                    merged_prefix += service_estimate(merged_job) or 0.0

            # Mirror the same bounded overtake at the global worker pool.  It
            # is an intentionally causal snapshot (not a future service-time
            # oracle): lane order and current debt are known, while EWMA is
            # used only to turn the resulting prefix into seconds.
            auth_remaining = list(queued_authoritative)
            spec_remaining = list(queued_speculative)
            simulated_debt = set(self._reserved_speculative_debt_by_tool)
            reservation_blocked_by_running_speculative = (
                self._running_speculative >= self._min_speculative_workers
            )
            speculative_turn = (
                not reservation_blocked_by_running_speculative
            )
            merged_global: list[_BrokerJob] = []
            while auth_remaining or spec_remaining:
                chosen_spec_index: int | None = None
                if speculative_turn:
                    for index, candidate in enumerate(spec_remaining):
                        candidate_tool = candidate.invocation.tool_name
                        owes_same_tool = (
                            candidate_tool in simulated_debt
                            and any(
                                auth.invocation.tool_name == candidate_tool
                                for auth in auth_remaining
                            )
                        )
                        if not owes_same_tool:
                            chosen_spec_index = index
                            break
                if chosen_spec_index is not None:
                    chosen = spec_remaining.pop(chosen_spec_index)
                    merged_global.append(chosen)
                    chosen_tool = chosen.invocation.tool_name
                    if any(
                        auth.invocation.tool_name == chosen_tool
                        for auth in auth_remaining
                    ):
                        simulated_debt.add(chosen_tool)
                    speculative_turn = False
                elif auth_remaining:
                    chosen = auth_remaining.pop(0)
                    merged_global.append(chosen)
                    simulated_debt.discard(chosen.invocation.tool_name)
                    speculative_turn = (
                        not reservation_blocked_by_running_speculative
                    )
                else:
                    merged_global.extend(spec_remaining)
                    spec_remaining.clear()
            merged_global_prefix = 0.0
            for merged_job in merged_global:
                reservation_global_prefix_work_s[merged_job.job_id] = (
                    merged_global_prefix
                )
                merged_global_prefix += service_estimate(merged_job) or 0.0

        public_jobs = []
        for job in sorted(jobs, key=lambda item: item.job_id):
            estimate = service_estimate(job)
            elapsed = (
                max(0.0, now - job.started_at)
                if job.started_at is not None
                and job.state in {"running", "cancelling"}
                else 0.0
            )
            queue_position: int | None = None
            tool_queue_position: int | None = None
            estimated_queue_s = 0.0
            estimated_global_queue_s = 0.0
            estimated_tool_queue_s = 0.0
            estimated_remaining_s: float | None
            tool_name = job.invocation.tool_name
            tool_capacity = self._tool_capacity(tool_name)
            tool_min_start_interval_s = self._tool_min_start_interval(tool_name)
            rate_limit_next_eligible_at = self._tool_next_eligible_at.get(
                tool_name, now
            )
            rate_limit_eligible_at = rate_limit_next_eligible_at
            rate_limit_wait_s = 0.0
            if job.state == "completed":
                estimated_remaining_s = 0.0
            elif job.state in {"running", "cancelling"}:
                estimated_remaining_s = (
                    None if estimate is None else max(0.0, estimate - elapsed)
                )
            elif job.state == "queued" and job.lane == "authoritative":
                queue_position = authoritative_positions[job.job_id]
                tool_queue_position = reservation_tool_positions.get(
                    job.job_id, authoritative_tool_positions[job.job_id]
                )
                rate_limit_eligible_at = max(
                    now, rate_limit_next_eligible_at
                ) + tool_queue_position * tool_min_start_interval_s
                rate_limit_wait_s = max(0.0, rate_limit_eligible_at - now)
                estimated_global_queue_s = (
                    running_work_s
                    + reservation_global_prefix_work_s.get(
                        job.job_id,
                        authoritative_prefix_work_s.get(job.job_id, 0.0),
                    )
                ) / self._max_workers
                estimated_tool_queue_s = (
                    running_work_by_tool_s.get(tool_name, 0.0)
                    + reservation_tool_prefix_work_s.get(
                        job.job_id,
                        authoritative_tool_prefix_work_s.get(job.job_id, 0.0),
                    )
                ) / tool_capacity
                estimated_tool_queue_s = max(
                    estimated_tool_queue_s, rate_limit_wait_s
                )
                estimated_queue_s = max(
                    estimated_global_queue_s, estimated_tool_queue_s
                )
                estimated_remaining_s = (
                    None if estimate is None else estimated_queue_s + estimate
                )
            elif job.state == "queued" and job.lane == "speculative":
                queue_position = speculative_positions[job.job_id]
                tool_queue_position = reservation_tool_positions.get(
                    job.job_id, speculative_tool_positions[job.job_id]
                )
                rate_limit_eligible_at = max(
                    now, rate_limit_next_eligible_at
                ) + tool_queue_position * tool_min_start_interval_s
                rate_limit_wait_s = max(0.0, rate_limit_eligible_at - now)
                if self._max_speculative_workers <= 0:
                    estimated_remaining_s = None
                    estimated_queue_s = math.inf
                    estimated_global_queue_s = math.inf
                    estimated_tool_queue_s = math.inf
                else:
                    estimated_global_queue_s = (
                        running_work_s
                        + reservation_global_prefix_work_s.get(
                            job.job_id,
                            all_authoritative_queue_work_s
                            + speculative_prefix_work_s.get(job.job_id, 0.0),
                        )
                    ) / self._max_speculative_workers
                    speculative_tool_capacity = min(
                        tool_capacity, self._max_speculative_workers
                    )
                    estimated_tool_queue_s = (
                        running_work_by_tool_s.get(tool_name, 0.0)
                        + reservation_tool_prefix_work_s.get(
                            job.job_id,
                            all_authoritative_tool_queue_work_s.get(
                                tool_name, 0.0
                            )
                            + speculative_tool_prefix_work_s.get(job.job_id, 0.0),
                        )
                    ) / speculative_tool_capacity
                    estimated_tool_queue_s = max(
                        estimated_tool_queue_s, rate_limit_wait_s
                    )
                    estimated_queue_s = max(
                        estimated_global_queue_s, estimated_tool_queue_s
                    )
                    estimated_remaining_s = (
                        None if estimate is None else estimated_queue_s + estimate
                    )
            else:
                estimated_remaining_s = None
            public_jobs.append(
                {
                    "job_id": job.job_id,
                    "session_id": job.session_id,
                    "tool_name": job.invocation.tool_name,
                    "invocation_digest": self._invocation_digest(job.invocation),
                    "lane": job.lane,
                    "state": job.state,
                    "priority": job.priority,
                    "queue_order": job.queue_order,
                    "age_s": max(0.0, now - job.created_at),
                    "service_estimate_s": estimate,
                    "queue_position": queue_position,
                    "tool_queue_position": tool_queue_position,
                    "tool_capacity": tool_capacity,
                    "tool_min_start_interval_s": tool_min_start_interval_s,
                    "rate_limit_eligible_at": rate_limit_eligible_at,
                    "rate_limit_next_eligible_at": rate_limit_next_eligible_at,
                    "rate_limit_wait_s": rate_limit_wait_s,
                    "estimated_global_queue_s": estimated_global_queue_s,
                    "estimated_tool_queue_s": estimated_tool_queue_s,
                    "estimated_queue_s": estimated_queue_s,
                    "estimated_remaining_s": estimated_remaining_s,
                    "confirmed": job.confirmed_at is not None,
                }
            )
        counts = {
            "queued_authoritative": sum(
                job.state == "queued" and job.lane == "authoritative" for job in jobs
            ),
            "queued_speculative": sum(
                job.state == "queued" and job.lane == "speculative" for job in jobs
            ),
            "running_authoritative": sum(
                job.state in {"running", "cancelling"}
                and job.lane == "authoritative"
                for job in jobs
            ),
            "running_speculative": sum(
                job.state in {"running", "cancelling"}
                and job.lane == "speculative"
                for job in jobs
            ),
            "completed_unclaimed_speculative": sum(
                job.state == "completed" and job.originally_speculative for job in jobs
            ),
            "queued_by_tool": {
                tool_name: sum(
                    job.state == "queued"
                    and job.invocation.tool_name == tool_name
                    for job in jobs
                )
                for tool_name in sorted(
                    {job.invocation.tool_name for job in jobs if job.state == "queued"}
                )
            },
            "running_by_tool": {
                tool_name: sum(
                    job.state in {"running", "cancelling"}
                    and job.invocation.tool_name == tool_name
                    for job in jobs
                )
                for tool_name in sorted(
                    {
                        job.invocation.tool_name
                        for job in jobs
                        if job.state in {"running", "cancelling"}
                    }
                )
            },
        }
        return {
            "revision": self._revision,
            "observed_at_monotonic_s": now,
            "capacity": {
                "max_workers": self._max_workers,
                "max_speculative_workers": self._max_speculative_workers,
                "min_speculative_workers": self._min_speculative_workers,
                "max_speculative_pending": self._max_speculative_pending,
                "default_tool_capacity": self._max_workers,
                "tool_capacities": dict(sorted(self._tool_capacities.items())),
                "tool_min_start_intervals_s": dict(
                    sorted(self._tool_min_start_intervals_s.items())
                ),
            },
            "rate_limit": {
                "tool_min_start_intervals_s": dict(
                    sorted(self._tool_min_start_intervals_s.items())
                ),
                "next_eligible_at": dict(
                    sorted(self._tool_next_eligible_at.items())
                ),
                "next_eligible_in_s": {
                    name: max(0.0, eligible_at - now)
                    for name, eligible_at in sorted(
                        self._tool_next_eligible_at.items()
                    )
                },
                "wakeup_deadline": self._rate_wakeup_deadline,
            },
            "reservation": {
                "authoritative_turn_due_by_tool": sorted(
                    self._reserved_speculative_debt_by_tool
                ),
                "reserved_speculative_dispatches": (
                    self.stats.reserved_speculative_dispatches
                ),
                "authoritative_after_reserved_dispatches": (
                    self.stats.authoritative_after_reserved_dispatches
                ),
            },
            "counts": counts,
            "service_ewma_s": dict(sorted(self._service_ewma_s.items())),
            "jobs": public_jobs,
            "stats": self.stats.to_dict(),
        }

    def tool_records(self) -> tuple[dict[str, Any], ...]:
        """Return result-free, validator-ready physical-attempt records."""

        records = [dict(record) for _, record in sorted(self._tool_records.items())]
        records.extend(dict(record) for record in self._rejected_records)
        return tuple(records)

    async def wait_for_change(
        self, revision: int, *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Wait until queue state changes, then return a fresh snapshot."""

        while self._revision <= revision:
            self._changed.clear()
            if self._revision > revision:
                break
            waiter = self._changed.wait()
            if timeout_s is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout=timeout_s)
        return self.snapshot()

    async def close(self) -> None:
        rate_wakeup_task: asyncio.Task[None] | None
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            rate_wakeup_task = self._rate_wakeup_task
            self._rate_wakeup_task = None
            self._rate_wakeup_deadline = None
            if rate_wakeup_task is not None and not rate_wakeup_task.done():
                rate_wakeup_task.cancel()
            jobs = list(self._jobs.values())
        if rate_wakeup_task is not None:
            await asyncio.gather(rate_wakeup_task, return_exceptions=True)
        for job in jobs:
            await self._cancel_job(job, expired=False)

    async def __aenter__(self) -> "LiveToolBroker":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
