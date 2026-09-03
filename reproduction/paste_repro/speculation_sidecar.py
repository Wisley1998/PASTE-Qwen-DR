"""An isolated, bounded control plane for speculative tool execution.

The sidecar deliberately has no authoritative queue.  Its executor, event
loop, scheduling heap, result retention, and cleanup all live on a dedicated
thread.  Calls from the authority process are limited to bounded, non-blocking
message submission and an opportunistic exact-key lookup.

The public API is message-oriented so that the thread transport can later be
replaced by a process without changing callers.  A production process-backed
implementation should preserve the two fail-open properties used here:

* ingress is bounded and ``try_*`` methods never wait for capacity; and
* a claim miss, busy sidecar, or failed speculation simply leaves the already
  submitted authoritative call as the only result path.

``close(wait=True)`` and ``snapshot()`` are lifecycle/diagnostic operations;
they are not intended for an authority request's critical path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
import concurrent.futures
from dataclasses import dataclass
import heapq
import math
import multiprocessing
from multiprocessing.connection import wait as wait_for_connections
import os
from pathlib import Path
import pickle
import queue
import select
import socket
import threading
import time
from typing import Any, Literal

from .invocation import Invocation


SidecarExecutor = Callable[[Invocation], Awaitable[Any]]
ProcessSubmitEntry = tuple[Invocation, str, str, float, str]
ProcessScheduleEntry = tuple[
    float, float, Iterable[ProcessSubmitEntry]
]

# A pull claim must not deserialize an arbitrarily large outer envelope on the
# authority thread.  The result itself has a caller-configured cap; this fixed
# allowance covers the request id, timestamps, error tuple, and pickle framing.
_PULL_STAGED_ENVELOPE_HEADROOM_BYTES = 4 * 1024


@dataclass(frozen=True)
class _CpuTopology:
    cpu: int
    package_id: int | None
    core_id: int | None
    numa_node: int | None
    thread_siblings: frozenset[int] | None


def _read_optional_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_optional_cpu_list(path: Path) -> frozenset[int] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        cpus: set[int] = set()
        for part in text.split(","):
            bounds = part.split("-", 1)
            start = int(bounds[0])
            stop = int(bounds[-1])
            if start < 0 or stop < start:
                raise ValueError
            cpus.update(range(start, stop + 1))
        return frozenset(cpus) if cpus else None
    except (OSError, ValueError):
        return None


def _read_cpu_topology(cpu: int, *, topology_root: Path) -> _CpuTopology:
    cpu_root = topology_root / f"cpu{cpu}"
    topology = cpu_root / "topology"
    numa_nodes = sorted(
        int(path.name[4:])
        for path in cpu_root.glob("node[0-9]*")
        if path.name[4:].isdigit()
    )
    return _CpuTopology(
        cpu=cpu,
        package_id=_read_optional_int(topology / "physical_package_id"),
        core_id=_read_optional_int(topology / "core_id"),
        numa_node=numa_nodes[0] if numa_nodes else None,
        thread_siblings=_read_optional_cpu_list(
            topology / "thread_siblings_list"
        ),
    )


def _same_physical_core(
    authority: _CpuTopology, candidate: _CpuTopology
) -> bool | None:
    if authority.thread_siblings is not None:
        return candidate.cpu in authority.thread_siblings
    if candidate.thread_siblings is not None:
        return authority.cpu in candidate.thread_siblings
    if (
        authority.package_id is not None
        and authority.core_id is not None
        and candidate.package_id is not None
        and candidate.core_id is not None
    ):
        return (
            authority.package_id,
            authority.core_id,
        ) == (candidate.package_id, candidate.core_id)
    return None


def distinct_physical_core_certificate(
    cpus: Iterable[int],
    *,
    topology_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
) -> bool:
    """Return true only when every CPU pair is provably on distinct cores.

    Logical-mask disjointness is insufficient for strict placement: two IDs
    can be SMT siblings, and missing sysfs metadata must fail closed rather
    than be interpreted as independence.
    """

    normalized: set[int] = set()
    for cpu in cpus:
        if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0:
            raise ValueError("CPU ids must be non-negative integers")
        normalized.add(cpu)
    ordered = sorted(normalized)
    if len(ordered) < 2:
        return False
    root = Path(topology_root)
    topology = [
        _read_cpu_topology(cpu, topology_root=root) for cpu in ordered
    ]
    return all(
        _same_physical_core(topology[left], topology[right]) is False
        for left in range(len(topology))
        for right in range(left + 1, len(topology))
    )


def choose_authority_sidecar_cpus(
    available_cpus: Iterable[int],
    *,
    topology_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
) -> tuple[int, int]:
    """Choose disjoint logical CPUs without sending fork memory cross-NUMA.

    The authority keeps the lowest granted logical CPU for stable paired
    replays.  The sidecar preference is: same NUMA node and another physical
    core, same package and another physical core, then any other known-distinct
    core.  Candidates whose physical-core relation cannot be proved come
    next, and a known SMT sibling is used only as the final fallback.

    Missing or unreadable sysfs metadata is deliberately non-fatal: the
    result remains deterministic and disjoint, but loses topology guarantees.
    """

    normalized: set[int] = set()
    for cpu in available_cpus:
        if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0:
            raise ValueError("available CPU ids must be non-negative integers")
        normalized.add(cpu)
    cpus = sorted(normalized)
    if len(cpus) < 2:
        raise ValueError("authority/sidecar isolation requires two CPUs")

    root = Path(topology_root)
    authority = _read_cpu_topology(cpus[0], topology_root=root)

    def rank(cpu: int) -> tuple[int, int, int]:
        candidate = _read_cpu_topology(cpu, topology_root=root)
        same_core = _same_physical_core(authority, candidate)
        same_numa = (
            authority.numa_node is not None
            and candidate.numa_node == authority.numa_node
        )
        same_package = (
            authority.package_id is not None
            and candidate.package_id == authority.package_id
        )
        if same_core is False:
            tier = 0 if same_numa else 1 if same_package else 2
        elif same_core is None:
            tier = 3 if same_numa else 4 if same_package else 5
        else:
            tier = 6
        return tier, abs(cpu - authority.cpu), cpu

    sidecar = min(cpus[1:], key=rank)
    return authority.cpu, sidecar


def choose_authority_control_sidecar_cpus(
    available_cpus: Iterable[int],
    *,
    topology_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
) -> tuple[int, int, int]:
    """Choose stable authority, control, and sidecar CPU roles.

    The first two roles are the same stable prefix that the two-role helper
    would choose.  The third role first avoids a proven SMT relationship with
    either role, then prefers locality.  This helper chooses placement only;
    callers that need a strict physical-core certificate must still reject a
    result whose pairwise topology is unknown or shared.
    """

    normalized: set[int] = set()
    for cpu in available_cpus:
        if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0:
            raise ValueError("available CPU ids must be non-negative integers")
        normalized.add(cpu)
    cpus = sorted(normalized)
    if len(cpus) < 3:
        raise ValueError("three-lane isolation requires three CPUs")

    authority_cpu, control_cpu = choose_authority_sidecar_cpus(
        cpus, topology_root=topology_root
    )
    root = Path(topology_root)
    authority = _read_cpu_topology(authority_cpu, topology_root=root)
    control = _read_cpu_topology(control_cpu, topology_root=root)

    def rank(cpu: int) -> tuple[int, int, int, int]:
        candidate = _read_cpu_topology(cpu, topology_root=root)
        core_relations = (
            _same_physical_core(authority, candidate),
            _same_physical_core(control, candidate),
        )
        if all(relation is False for relation in core_relations):
            core_tier = 0
        elif not any(relation is True for relation in core_relations):
            core_tier = 1
        else:
            core_tier = 2

        same_numa = (
            authority.numa_node is not None
            and authority.numa_node == control.numa_node
            and candidate.numa_node == authority.numa_node
        )
        same_package = (
            authority.package_id is not None
            and authority.package_id == control.package_id
            and candidate.package_id == authority.package_id
        )
        locality_tier = 0 if same_numa else 1 if same_package else 2
        distance = abs(cpu - authority_cpu) + abs(cpu - control_cpu)
        return core_tier, locality_tier, distance, cpu

    sidecar_cpu = min(
        (
            cpu
            for cpu in cpus
            if cpu not in {authority_cpu, control_cpu}
        ),
        key=rank,
    )
    return authority_cpu, control_cpu, sidecar_cpu


class SpeculativeSidecarError(RuntimeError):
    """Base class for sidecar-local terminal outcomes."""


class SidecarRejected(SpeculativeSidecarError):
    """The scheduler rejected a request after bounded ingress accepted it."""


class SidecarExpired(SpeculativeSidecarError):
    """A queued request crossed its latest useful start time."""


class SidecarTombstoned(SpeculativeSidecarError):
    """A request was made unclaimable without cancelling physical work."""


class SidecarClosed(SpeculativeSidecarError):
    """The sidecar closed before a request produced a result."""


@dataclass(frozen=True)
class ExactSpeculationKey:
    """Full identity required to reuse a speculative result.

    ``context_token`` lets callers include a credential, snapshot, or tool
    configuration version.  Omitting it is appropriate only when an
    invocation's canonical arguments fully determine interchangeable results.
    """

    session_id: str
    decision_id: str
    invocation_key: tuple[str, str]
    context_token: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise ValueError("decision_id must be a non-empty string")
        if (
            not isinstance(self.invocation_key, tuple)
            or len(self.invocation_key) != 2
            or not all(isinstance(value, str) for value in self.invocation_key)
        ):
            raise ValueError("invocation_key must be a pair of strings")
        if not isinstance(self.context_token, str):
            raise ValueError("context_token must be a string")

    @classmethod
    def from_invocation(
        cls,
        invocation: Invocation,
        *,
        session_id: str,
        decision_id: str,
        context_token: str = "",
    ) -> "ExactSpeculationKey":
        return cls(
            session_id=session_id,
            decision_id=decision_id,
            invocation_key=invocation.key,
            context_token=context_token,
        )


class SpeculativeHandle:
    """A process-replaceable handle for one exact speculative execution.

    The concurrent future is safe to observe from another thread.  It yields
    the executor's value directly, or a sidecar/executor exception.  Creation
    of a handle means only that bounded ingress accepted the message; later
    scheduler rejection is reported through ``future``.
    """

    def __init__(self, key: ExactSpeculationKey) -> None:
        self.key = key
        self.future: concurrent.futures.Future[Any] = (
            concurrent.futures.Future()
        )
        self._lock = threading.Lock()
        self._claimed = False
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._valid_until: float | None = None
        self._job_id: int | None = None

    @property
    def invocation_key(self) -> tuple[str, str]:
        return self.key.invocation_key

    @property
    def claimed(self) -> bool:
        with self._lock:
            return self._claimed

    @property
    def started_at(self) -> float | None:
        with self._lock:
            return self._started_at

    @property
    def finished_at(self) -> float | None:
        with self._lock:
            return self._finished_at

    def as_asyncio_future(
        self, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> "asyncio.Future[Any]":
        """Wrap the result for an authority loop without moving execution."""

        return asyncio.wrap_future(self.future, loop=loop)

    def _mark_started(self, now: float) -> None:
        with self._lock:
            self._started_at = now

    def _mark_admitted(self, job_id: int) -> None:
        with self._lock:
            self._job_id = job_id

    def _mark_finished(self, now: float, *, valid_until: float) -> None:
        with self._lock:
            self._finished_at = now
            self._valid_until = valid_until

    def _try_claim(self, now: float) -> bool:
        with self._lock:
            if self._claimed:
                return False
            if self._valid_until is not None and self._valid_until <= now:
                return False
            self._claimed = True
            return True

    def _is_ready_and_claimed(self) -> bool:
        with self._lock:
            return self._claimed and self._finished_at is not None

    def _admitted_job_id(self) -> int | None:
        with self._lock:
            return self._job_id


@dataclass(frozen=True)
class SidecarRaceResult:
    """Result of a non-cancelling authority/speculation race."""

    result: Any
    source: Literal["authoritative", "speculative"]


@dataclass(frozen=True)
class _SubmitCommand:
    invocation: Invocation
    handle: SpeculativeHandle
    priority: float
    start_deadline: float | None


@dataclass(frozen=True)
class _TombstoneCommand:
    session_id: str
    decision_id: str | None


@dataclass
class _SidecarJob:
    job_id: int
    invocation: Invocation
    handle: SpeculativeHandle
    priority: float
    start_deadline: float | None
    order: int
    state: Literal["queued", "running", "ready"] = "queued"
    generation: int = 0
    tombstoned: bool = False
    ready_expires_at: float | None = None
    runner: "asyncio.Task[None] | None" = None

    @property
    def decision_key(self) -> tuple[str, str]:
        return self.handle.key.session_id, self.handle.key.decision_id


def _observe_background_future(future: "asyncio.Future[Any]") -> None:
    """Retrieve a detached loser's exception without cancelling the loser."""

    try:
        future.exception()
    except (asyncio.CancelledError, concurrent.futures.CancelledError):
        pass


async def race_authority_with_speculation(
    authoritative: Awaitable[Any],
    handle: SpeculativeHandle | None,
) -> SidecarRaceResult:
    """Return speculative success or the authoritative terminal result.

    The authoritative awaitable must already represent a submitted baseline
    call.  It is never cancelled.  A speculative failure is ignored, whereas
    an authoritative success, failure, or cancellation is terminal.  If both
    sides complete in the same observation turn, authority wins the tie.
    Loser cleanup is deliberately absent from this function.
    """

    authority_future = asyncio.ensure_future(authoritative)
    if handle is None:
        return SidecarRaceResult(
            result=await authority_future,
            source="authoritative",
        )

    speculative_future = handle.as_asyncio_future()
    try:
        while True:
            await asyncio.wait(
                {authority_future, speculative_future},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if authority_future.done():
                speculative_future.add_done_callback(
                    _observe_background_future
                )
                return SidecarRaceResult(
                    result=authority_future.result(),
                    source="authoritative",
                )
            if speculative_future.done():
                try:
                    result = speculative_future.result()
                except BaseException:
                    return SidecarRaceResult(
                        result=await authority_future,
                        source="authoritative",
                    )
                authority_future.add_done_callback(_observe_background_future)
                return SidecarRaceResult(
                    result=result,
                    source="speculative",
                )
    except asyncio.CancelledError:
        # Cancellation of the caller is not authority to cancel either
        # physical call.  Observers merely suppress detached-task warnings.
        authority_future.add_done_callback(_observe_background_future)
        speculative_future.add_done_callback(_observe_background_future)
        raise


class SpeculativeSidecar:
    """Benefit-priority speculative scheduler on an isolated event-loop thread.

    ``priority`` is expected-benefit density supplied by the policy layer;
    larger values dispatch first.  The scheduler permits at most one live job
    for each ``(session_id, decision_id)`` and never runs more than
    ``max_workers`` calls.  ``max_pending`` bounds queued, running, and retained
    ready results.  When full, a new higher-priority request may replace only
    a queued request, never running work.
    """

    def __init__(
        self,
        executor: SidecarExecutor,
        max_workers: int = 1,
        max_pending: int | None = None,
        *,
        ingress_capacity: int | None = None,
        result_ttl_s: float = 30.0,
        autostart: bool = False,
        clock: Callable[[], float] = time.monotonic,
        thread_name: str = "speculative-sidecar",
    ) -> None:
        if not callable(executor):
            raise TypeError("executor must be callable")
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer")
        if max_pending is None:
            max_pending = 2 * max_workers
        if (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending < max_workers
        ):
            raise ValueError("max_pending must be an integer >= max_workers")
        if ingress_capacity is None:
            ingress_capacity = max(4, 2 * max_pending)
        if (
            isinstance(ingress_capacity, bool)
            or not isinstance(ingress_capacity, int)
            or ingress_capacity <= 0
        ):
            raise ValueError("ingress_capacity must be a positive integer")
        if (
            isinstance(result_ttl_s, bool)
            or not isinstance(result_ttl_s, (int, float))
            or not math.isfinite(result_ttl_s)
            or result_ttl_s <= 0.0
        ):
            raise ValueError("result_ttl_s must be finite and positive")

        self._executor = executor
        self._max_workers = max_workers
        self._max_pending = max_pending
        self._ingress_capacity = ingress_capacity
        self._result_ttl_s = float(result_ttl_s)
        self._clock = clock
        self._thread_name = thread_name

        self._commands: "queue.Queue[_SubmitCommand | _TombstoneCommand]" = (
            queue.Queue(maxsize=ingress_capacity)
        )
        self._lifecycle_lock = threading.Lock()
        self._publication_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._accepting = True
        self._started = False
        self._closing = False
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_thread_id: int | None = None
        self._shutdown_started = False

        # The following containers are owned by the sidecar loop thread.
        self._jobs: dict[int, _SidecarJob] = {}
        self._by_decision: dict[tuple[str, str], _SidecarJob] = {}
        self._heap: list[tuple[float, int, int, int]] = []
        self._running = 0
        self._next_job_id = 0
        self._next_order = 0
        self._published: dict[ExactSpeculationKey, SpeculativeHandle] = {}
        self._last_snapshot: dict[str, Any] | None = None
        self._stats: dict[str, int] = {
            "ingress_accepted": 0,
            "ingress_full": 0,
            "rejected_nonpositive_priority": 0,
            "admitted": 0,
            "rejected_deadline": 0,
            "rejected_decision": 0,
            "rejected_pending": 0,
            "replaced_queued": 0,
            "started": 0,
            "completed": 0,
            "failed": 0,
            "expired_queued": 0,
            "expired_before_executor": 0,
            "expired_ready": 0,
            "tombstoned_queued": 0,
            "tombstoned_running": 0,
            "tombstoned_ready": 0,
            "claims": 0,
            "claim_misses": 0,
            "claim_busy": 0,
            "max_running": 0,
            "max_pending": 0,
        }

        if autostart:
            self.start()

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[name] += amount

    def _set_high_watermark(self, name: str, value: int) -> None:
        with self._stats_lock:
            self._stats[name] = max(self._stats[name], value)

    def start(self, *, timeout: float = 5.0) -> None:
        """Start the isolated event-loop thread once."""

        with self._lifecycle_lock:
            if self._closed or self._closing:
                raise SidecarClosed("cannot start a closed sidecar")
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._thread_main,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("speculative sidecar thread did not start")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_thread_id = threading.get_ident()
        self._ready.set()
        loop.call_soon(self._drain_commands)
        try:
            loop.run_forever()
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            self._last_snapshot = self._snapshot_in_loop()
            self._last_snapshot["thread_alive"] = False
            loop.close()
            with self._lifecycle_lock:
                self._closed = True
            self._stopped.set()

    def _notify_loop(self) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(self._drain_commands)
        except RuntimeError:
            # A concurrent lifecycle close is a fail-open condition for the
            # authority path. Shutdown will drain or reject queued messages.
            pass

    def try_submit(
        self,
        invocation: Invocation,
        *,
        session_id: str,
        decision_id: str,
        priority: float,
        start_deadline: float | None = None,
        context_token: str = "",
    ) -> SpeculativeHandle | None:
        """Non-blockingly enqueue one candidate and return its exact handle.

        ``None`` means the bounded ingress was full, the sidecar was closed,
        or the candidate had no positive finite benefit priority.  Admission
        decisions made later on the sidecar thread complete the returned
        handle with :class:`SidecarRejected` or :class:`SidecarExpired`.
        """

        if not isinstance(invocation, Invocation):
            raise TypeError("invocation must be an Invocation")
        if (
            isinstance(priority, bool)
            or not isinstance(priority, (int, float))
            or not math.isfinite(priority)
        ):
            raise ValueError("priority must be a finite number")
        if priority <= 0.0:
            self._increment("rejected_nonpositive_priority")
            return None
        if start_deadline is not None and (
            isinstance(start_deadline, bool)
            or not isinstance(start_deadline, (int, float))
            or not math.isfinite(start_deadline)
        ):
            raise ValueError("start_deadline must be finite or None")
        key = ExactSpeculationKey.from_invocation(
            invocation,
            session_id=session_id,
            decision_id=decision_id,
            context_token=context_token,
        )
        handle = SpeculativeHandle(key)
        command = _SubmitCommand(
            invocation=invocation,
            handle=handle,
            priority=float(priority),
            start_deadline=(
                None if start_deadline is None else float(start_deadline)
            ),
        )
        with self._lifecycle_lock:
            if not self._accepting or self._closing or self._closed:
                return None
            try:
                self._commands.put_nowait(command)
            except queue.Full:
                self._increment("ingress_full")
                return None
            self._increment("ingress_accepted")
        self._notify_loop()
        return handle

    def try_tombstone(
        self,
        *,
        session_id: str,
        decision_id: str | None = None,
    ) -> bool:
        """Best-effort invalidation; running physical work drains lazily."""

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if decision_id is not None and (
            not isinstance(decision_id, str) or not decision_id
        ):
            raise ValueError("decision_id must be a non-empty string or None")
        command = _TombstoneCommand(session_id, decision_id)
        with self._lifecycle_lock:
            if not self._accepting or self._closing or self._closed:
                return False
            try:
                self._commands.put_nowait(command)
            except queue.Full:
                self._increment("ingress_full")
                return False
        self._notify_loop()
        return True

    def try_claim(
        self, key: ExactSpeculationKey
    ) -> SpeculativeHandle | None:
        """Claim one RUNNING/READY exact match without waiting for the sidecar.

        Lock contention is treated as a miss.  Queued predictions are never
        exposed because they have produced no overlap and the baseline
        authority call must already have been submitted before this method.
        """

        if not isinstance(key, ExactSpeculationKey):
            raise TypeError("key must be an ExactSpeculationKey")
        if not self._publication_lock.acquire(blocking=False):
            self._increment("claim_busy")
            return None
        retire_job_id: int | None = None
        expire_job_id: int | None = None
        try:
            handle = self._published.get(key)
            if handle is None or not handle._try_claim(self._clock()):
                if handle is not None:
                    self._published.pop(key, None)
                    expire_job_id = handle._admitted_job_id()
                self._increment("claim_misses")
            else:
                self._published.pop(key, None)
                self._increment("claims")
                if handle._is_ready_and_claimed():
                    retire_job_id = handle._admitted_job_id()
        finally:
            self._publication_lock.release()
        if expire_job_id is not None:
            self._schedule_expire_ready(expire_job_id)
            return None
        if handle is None:
            return None
        if retire_job_id is not None:
            self._schedule_retire_claimed_ready(retire_job_id)
        return handle

    def _schedule_retire_claimed_ready(self, job_id: int) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(self._retire_claimed_ready, job_id)
        except RuntimeError:
            pass

    def _try_retire_ready_for_staging(
        self, key: ExactSpeculationKey
    ) -> bool:
        """Release a transferred ready result without recording a policy hit."""

        if not self._publication_lock.acquire(blocking=False):
            return False
        job_id: int | None = None
        try:
            handle = self._published.get(key)
            if (
                handle is None
                or not handle.future.done()
                or handle.finished_at is None
            ):
                return False
            self._published.pop(key, None)
            job_id = handle._admitted_job_id()
        finally:
            self._publication_lock.release()
        if job_id is None:
            return False
        loop = self._loop
        if loop is None or not loop.is_running():
            return False
        try:
            loop.call_soon_threadsafe(
                self._retire_ready_for_staging,
                job_id,
            )
        except RuntimeError:
            return False
        return True

    def _retire_ready_for_staging(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is not None and job.state == "ready":
            self._retire_job(job, exception=None)

    def _retire_claimed_ready(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is not None and job.state == "ready" and job.handle.claimed:
            self._retire_job(job, exception=None)

    def _schedule_expire_ready(self, job_id: int) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(self._expire_ready, job_id)
        except RuntimeError:
            pass

    def _expire_ready(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if (
            job is not None
            and job.state == "ready"
            and job.ready_expires_at is not None
            and job.ready_expires_at <= self._clock()
        ):
            self._increment("expired_ready")
            self._retire_job(
                job,
                exception=SidecarExpired("unclaimed result TTL expired"),
            )

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, _SubmitCommand):
                self._admit(command)
            else:
                self._tombstone(command)
        self._expire(self._clock())
        self._dispatch()

    def _admit(self, command: _SubmitCommand) -> None:
        now = self._clock()
        if self._closing:
            command.handle.future.set_exception(
                SidecarClosed("sidecar is closing")
            )
            return
        if command.start_deadline is not None and command.start_deadline <= now:
            self._increment("rejected_deadline")
            command.handle.future.set_exception(
                SidecarExpired("candidate crossed start_deadline before admission")
            )
            return

        self._expire(now)
        decision_key = (
            command.handle.key.session_id,
            command.handle.key.decision_id,
        )
        existing = self._by_decision.get(decision_key)
        if existing is not None:
            if existing.state == "queued" and command.priority > existing.priority:
                self._retire_job(
                    existing,
                    exception=SidecarRejected(
                        "replaced by a higher-benefit candidate for the decision"
                    ),
                )
                self._increment("replaced_queued")
            else:
                self._increment("rejected_decision")
                command.handle.future.set_exception(
                    SidecarRejected(
                        "one live candidate already exists for session/decision"
                    )
                )
                return

        if len(self._jobs) >= self._max_pending:
            queued = [job for job in self._jobs.values() if job.state == "queued"]
            victim = min(
                queued,
                key=lambda job: (job.priority, -job.order),
                default=None,
            )
            if victim is None or command.priority <= victim.priority:
                self._increment("rejected_pending")
                command.handle.future.set_exception(
                    SidecarRejected("bounded pending set is full")
                )
                return
            self._retire_job(
                victim,
                exception=SidecarRejected(
                    "replaced by a higher-benefit global candidate"
                ),
            )
            self._increment("replaced_queued")

        self._next_job_id += 1
        self._next_order += 1
        job = _SidecarJob(
            job_id=self._next_job_id,
            invocation=command.invocation,
            handle=command.handle,
            priority=command.priority,
            start_deadline=command.start_deadline,
            order=self._next_order,
        )
        self._jobs[job.job_id] = job
        self._by_decision[job.decision_key] = job
        job.handle._mark_admitted(job.job_id)
        heapq.heappush(
            self._heap,
            (-job.priority, job.order, job.generation, job.job_id),
        )
        self._increment("admitted")
        self._set_high_watermark("max_pending", len(self._jobs))

    def _pop_queued(self) -> _SidecarJob | None:
        while self._heap:
            _, _, generation, job_id = heapq.heappop(self._heap)
            job = self._jobs.get(job_id)
            if (
                job is not None
                and job.state == "queued"
                and job.generation == generation
            ):
                return job
        return None

    def _dispatch(self) -> None:
        if self._closing:
            return
        now = self._clock()
        self._expire(now)
        while self._running < self._max_workers:
            job = self._pop_queued()
            if job is None:
                break
            now = self._clock()
            if job.start_deadline is not None and job.start_deadline <= now:
                self._increment("expired_queued")
                self._retire_job(
                    job,
                    exception=SidecarExpired(
                        "candidate crossed start_deadline while queued"
                    ),
                )
                continue
            job.state = "running"
            self._running += 1
            self._set_high_watermark("max_running", self._running)
            job.runner = asyncio.create_task(self._execute(job))

    async def _execute(self, job: _SidecarJob) -> None:
        result: Any = None
        error: BaseException | None = None
        actual_started_at = self._clock()
        if (
            job.start_deadline is not None
            and job.start_deadline <= actual_started_at
        ):
            error = SidecarExpired(
                "candidate crossed start_deadline before executor entry"
            )
            self._increment("expired_before_executor")
        else:
            job.handle._mark_started(actual_started_at)
            with self._publication_lock:
                self._published[job.handle.key] = job.handle
            self._increment("started")
            try:
                result = await self._executor(job.invocation)
            except BaseException as exc:
                error = exc

        finished_at = self._clock()
        self._running -= 1
        job.handle._mark_finished(
            finished_at,
            valid_until=finished_at + self._result_ttl_s,
        )

        deliver = job.handle.claimed or not job.tombstoned
        if error is None:
            self._increment("completed")
            if deliver and not job.handle.future.done():
                job.handle.future.set_result(result)
        else:
            self._increment("failed")
            if deliver and not job.handle.future.done():
                job.handle.future.set_exception(error)

        if (
            error is not None
            or job.tombstoned
            or job.handle.claimed
            or job.handle.future.cancelled()
        ):
            self._retire_job(job, exception=None)
        else:
            job.state = "ready"
            job.ready_expires_at = finished_at + self._result_ttl_s
        self._dispatch()

    def _expire(self, now: float) -> None:
        for job in tuple(self._jobs.values()):
            if (
                job.state == "queued"
                and job.start_deadline is not None
                and job.start_deadline <= now
            ):
                self._increment("expired_queued")
                self._retire_job(
                    job,
                    exception=SidecarExpired(
                        "candidate crossed start_deadline while queued"
                    ),
                )
            elif (
                job.state == "ready"
                and job.ready_expires_at is not None
                and job.ready_expires_at <= now
            ):
                self._increment("expired_ready")
                self._retire_job(
                    job,
                    exception=SidecarExpired("unclaimed result TTL expired"),
                )

    def _tombstone(self, command: _TombstoneCommand) -> None:
        matches = [
            job
            for job in self._jobs.values()
            if job.handle.key.session_id == command.session_id
            and (
                command.decision_id is None
                or job.handle.key.decision_id == command.decision_id
            )
        ]
        for job in matches:
            if job.state == "running":
                job.tombstoned = True
                with self._publication_lock:
                    if self._published.get(job.handle.key) is job.handle:
                        self._published.pop(job.handle.key, None)
                if not job.handle.claimed and not job.handle.future.done():
                    job.handle.future.set_exception(
                        SidecarTombstoned(
                            "running speculation tombstoned; physical work drains"
                        )
                    )
                self._increment("tombstoned_running")
            else:
                state = job.state
                self._retire_job(
                    job,
                    exception=SidecarTombstoned("speculation tombstoned"),
                )
                self._increment(f"tombstoned_{state}")

    def _retire_job(
        self,
        job: _SidecarJob,
        *,
        exception: BaseException | None,
    ) -> None:
        if self._jobs.get(job.job_id) is not job:
            return
        self._jobs.pop(job.job_id, None)
        if self._by_decision.get(job.decision_key) is job:
            self._by_decision.pop(job.decision_key, None)
        job.generation += 1
        with self._publication_lock:
            if self._published.get(job.handle.key) is job.handle:
                self._published.pop(job.handle.key, None)
        if exception is not None and not job.handle.future.done():
            job.handle.future.set_exception(exception)

    def snapshot(self, *, timeout: float = 2.0) -> dict[str, Any]:
        """Return a diagnostic snapshot; this may block and is not hot-path API."""

        with self._lifecycle_lock:
            started = self._started
            closed = self._closed
            loop = self._loop
        if not started:
            return self._snapshot_before_start()
        if closed or loop is None or not loop.is_running():
            if self._last_snapshot is not None:
                return self._copy_snapshot(self._last_snapshot)
            return self._snapshot_before_start()
        if threading.get_ident() == self._loop_thread_id:
            return self._snapshot_in_loop()
        result: concurrent.futures.Future[dict[str, Any]] = (
            concurrent.futures.Future()
        )

        def publish() -> None:
            if not result.done():
                result.set_result(self._snapshot_in_loop())

        loop.call_soon_threadsafe(publish)
        return result.result(timeout=timeout)

    @staticmethod
    def _copy_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        copied = {
            "capacity": dict(snapshot["capacity"]),
            "counts": dict(snapshot["counts"]),
            "stats": dict(snapshot["stats"]),
            "thread_alive": bool(snapshot["thread_alive"]),
        }
        # Scalar counters are also exposed at the top level so experiment
        # runners can consume either a compact dataclass-like snapshot or the
        # richer nested representation without transport-specific branches.
        copied.update(copied["stats"])
        return copied

    def _snapshot_before_start(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        snapshot = {
            "capacity": {
                "max_workers": self._max_workers,
                "max_pending": self._max_pending,
                "ingress_capacity": self._ingress_capacity,
            },
            "counts": {
                "pending": 0,
                "queued": 0,
                "running": 0,
                "ready": 0,
                "published": 0,
                "ingress": self._commands.qsize(),
            },
            "stats": stats,
            "thread_alive": False,
        }
        snapshot.update(stats)
        return snapshot

    def _snapshot_in_loop(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._publication_lock:
            published = len(self._published)
        counts = {state: 0 for state in ("queued", "running", "ready")}
        for job in self._jobs.values():
            counts[job.state] += 1
        snapshot = {
            "capacity": {
                "max_workers": self._max_workers,
                "max_pending": self._max_pending,
                "ingress_capacity": self._ingress_capacity,
            },
            "counts": {
                "pending": len(self._jobs),
                "queued": counts["queued"],
                "running": counts["running"],
                "ready": counts["ready"],
                "published": published,
                "ingress": self._commands.qsize(),
            },
            "stats": stats,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
        }
        snapshot.update(stats)
        return snapshot

    def close(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        """Stop admission and lazily drain running calls on the sidecar thread.

        This lifecycle method may block when ``wait`` is true.  Authority code
        must never call it while serving a request.
        """

        with self._lifecycle_lock:
            if self._closed:
                return
            self._accepting = False
            self._closing = True
            started = self._started
            loop = self._loop
        if not started:
            self._close_before_start()
            return
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._begin_shutdown)
            except RuntimeError:
                pass
        if wait and not self._stopped.wait(timeout):
            raise TimeoutError(
                "speculative sidecar still has non-preemptible work draining"
            )

    def _close_before_start(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, _SubmitCommand) and not command.handle.future.done():
                command.handle.future.set_exception(
                    SidecarClosed("sidecar closed before start")
                )
        with self._lifecycle_lock:
            self._closed = True
        self._last_snapshot = self._snapshot_before_start()
        self._stopped.set()

    def _begin_shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        asyncio.create_task(self._shutdown())

    async def _shutdown(self) -> None:
        # Reject messages accepted immediately before the lifecycle flag was
        # published. No new physical call is dispatched once closing begins.
        self._drain_commands()
        for job in tuple(self._jobs.values()):
            if job.state == "running":
                job.tombstoned = True
                with self._publication_lock:
                    if self._published.get(job.handle.key) is job.handle:
                        self._published.pop(job.handle.key, None)
                if not job.handle.claimed and not job.handle.future.done():
                    job.handle.future.set_exception(
                        SidecarClosed("sidecar closing while physical work drains")
                    )
            else:
                self._retire_job(
                    job,
                    exception=SidecarClosed("sidecar closed before claim"),
                )

        runners = [
            job.runner
            for job in self._jobs.values()
            if job.runner is not None and not job.runner.done()
        ]
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)
        for job in tuple(self._jobs.values()):
            self._retire_job(job, exception=None)
        self._last_snapshot = self._snapshot_in_loop()
        loop = asyncio.get_running_loop()
        loop.stop()


# Process transport -------------------------------------------------------
#
# The process implementation intentionally reuses the scheduler above inside
# the forked child.  This keeps one policy implementation while moving its
# event loop, executor callbacks, timers, heap maintenance, and cleanup off the
# authority process's GIL.  The parent/child protocol contains only immutable
# messages, which is also the intended boundary for a future native process
# worker.

_MAX_CHILD_SOCKET_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class _ProcessSubmit:
    request_id: int
    invocation: Invocation
    session_id: str
    decision_id: str
    context_token: str
    priority: float
    start_deadline: float | None


@dataclass(frozen=True)
class _ProcessSubmitBatch:
    submissions: tuple[_ProcessSubmit, ...]


@dataclass(frozen=True)
class _ProcessScheduledBatch:
    release_at: float
    start_deadline: float
    submissions: tuple[_ProcessSubmit, ...]


@dataclass(frozen=True)
class _ProcessScheduleBatches:
    batches: tuple[_ProcessScheduledBatch, ...]


@dataclass(frozen=True)
class _ProcessClaim:
    request_id: int


@dataclass(frozen=True)
class _ProcessTombstone:
    session_id: str
    decision_id: str | None


@dataclass(frozen=True)
class _ProcessTombstoneBatch:
    tombstones: tuple[_ProcessTombstone, ...]


@dataclass(frozen=True)
class _ProcessSnapshot:
    request_id: int


@dataclass(frozen=True)
class _ProcessClose:
    drain_timeout_s: float


@dataclass(frozen=True)
class _ProcessEvent:
    kind: Literal[
        "terminal",
        "result",
        "staged",
        "claim_miss",
        "snapshot",
        "closed",
    ]
    request_id: int | None = None
    payload: Any = None


@dataclass(frozen=True)
class _ProcessStagedOutcome:
    result_payload: bytes | None
    error_payload: tuple[str, str] | None
    started_at: float | None
    finished_at: float
    valid_until: float


@dataclass(frozen=True)
class _ParentStagedOutcome:
    result: Any
    error: BaseException | None
    started_at: float | None
    finished_at: float
    valid_until: float


@dataclass
class _ChildTransportRecord:
    handle: SpeculativeHandle
    start_deadline: float | None
    lease_until: float | None
    claim_requested: bool = False
    claimed: bool = False
    ready_unclaimed: bool = False
    eager_transferred: bool = False


@dataclass
class _ChildScheduledRecord:
    submission: _ProcessSubmit
    release_at: float
    claim_requested: bool = False


def _remote_error_payload(error: BaseException) -> tuple[str, str]:
    return type(error).__name__, str(error)


def _remote_error(payload: Any) -> BaseException:
    name = "RemoteSpeculationError"
    message = "remote speculative execution failed"
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and all(isinstance(value, str) for value in payload)
    ):
        name, message = payload
    error_types: dict[str, type[SpeculativeSidecarError]] = {
        "SidecarRejected": SidecarRejected,
        "SidecarExpired": SidecarExpired,
        "SidecarTombstoned": SidecarTombstoned,
        "SidecarClosed": SidecarClosed,
    }
    error_type = error_types.get(name, SpeculativeSidecarError)
    return error_type(f"{name}: {message}")


def _process_sidecar_worker(
    executor: SidecarExecutor,
    max_workers: int,
    max_pending: int,
    internal_ingress_capacity: int,
    result_ttl_s: float,
    command_socket: socket.socket,
    event_socket: socket.socket,
    parent_command_socket: socket.socket,
    parent_event_socket: socket.socket,
    max_packet_bytes: int,
    cpu_affinity: tuple[int, ...] | None,
    claim_grace_s: float,
    max_scheduled_pending: int,
    eager_result_staging: bool,
    pull_result_staging: bool,
    max_staged_result_bytes: int,
) -> None:
    """Fork-child bridge around the isolated scheduler.

    By default, successful values are serialized only after an exact claim.
    The explicit staging modes serialize terminal values before confirmation:
    eager staging has a continuous parent bridge, while pull staging leaves
    them in the bounded kernel socket until an exact parent lookup drains it.
    """

    # Close the authority process's ends inherited by fork. Keeping either end
    # open in the child would prevent EOF/failure detection.
    parent_command_socket.close()
    parent_event_socket.close()
    command_socket.settimeout(None)
    event_socket.settimeout(2.0)

    requested_affinity = list(cpu_affinity) if cpu_affinity is not None else None
    try:
        if cpu_affinity is not None:
            os.sched_setaffinity(0, set(cpu_affinity))
        actual_affinity = sorted(os.sched_getaffinity(0))
    except BaseException as exc:
        startup_snapshot = {
            "started": 0,
            "max_running": 0,
            "counts": {},
            "stats": {},
            "process_pid": os.getpid(),
            "requested_cpu_affinity": requested_affinity,
            "actual_cpu_affinity": None,
            "startup_error": f"cpu affinity certificate failed: {exc}",
        }
        try:
            event_socket.send(
                pickle.dumps(
                    _ProcessEvent("closed", payload=startup_snapshot),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            )
        finally:
            command_socket.close()
            event_socket.close()
        return

    # CPU affinity prevents direct run-queue contention on the paper host.
    # SCHED_IDLE adds a fail-open priority guard for oversubscribed/virtualized
    # hosts: speculative callbacks run normally on an idle CPU but yield to
    # every ordinary authority task if the physical scheduler sees pressure.
    requested_scheduler_policy = "SCHED_IDLE"
    scheduler_priority_error: str | None = None
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError) as exc:
        scheduler_priority_error = repr(exc)
    try:
        actual_scheduler_policy = os.sched_getscheduler(0)
    except (AttributeError, OSError):
        actual_scheduler_policy = None

    # Do not reuse the parent's running-loop policy after fork. The scheduler
    # creates its own event loop on a fresh child-local thread.
    asyncio.set_event_loop(None)
    sidecar = SpeculativeSidecar(
        executor,
        max_workers=max_workers,
        max_pending=max_pending,
        ingress_capacity=internal_ingress_capacity,
        result_ttl_s=result_ttl_s,
        autostart=True,
        thread_name="speculative-sidecar-child-loop",
    )
    records: dict[int, _ChildTransportRecord] = {}
    scheduled_records: dict[int, _ChildScheduledRecord] = {}
    scheduled_heap: list[tuple[float, int, tuple[int, ...]]] = []
    scheduled_order = 0
    scheduled_stats: dict[str, int] = {
        "received_batches": 0,
        "received_candidates": 0,
        "released_batches": 0,
        "released_candidates": 0,
        "deadline_dropped": 0,
        "admission_dropped": 0,
        "capacity_dropped": 0,
        "tombstoned": 0,
        "closed_unreleased": 0,
    }
    lease_stats: dict[str, int] = {
        "expired": 0,
        "tombstones_enqueued": 0,
        "tombstone_retries": 0,
    }
    staging_enabled = eager_result_staging or pull_result_staging
    eager_stats: dict[str, int | bool | str] = {
        "enabled": staging_enabled,
        "mode": (
            "eager"
            if eager_result_staging
            else "pull"
            if pull_result_staging
            else "disabled"
        ),
        "result_events": 0,
        "failure_events": 0,
        "dropped_events": 0,
        "max_staged_result_bytes": max_staged_result_bytes,
    }

    def emit(event: _ProcessEvent, *, best_effort: bool = False) -> bool:
        try:
            encoded = pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL)
            if len(encoded) > max_packet_bytes:
                return False
            if best_effort:
                event_socket.setblocking(False)
                event_socket.send(encoded)
                event_socket.settimeout(2.0)
            else:
                event_socket.send(encoded)
            return True
        except (BlockingIOError, BrokenPipeError, EOFError, OSError):
            if best_effort:
                event_socket.settimeout(2.0)
            return False

    def discard_record(request_id: int) -> None:
        records.pop(request_id, None)

    def scan_records() -> None:
        now = time.monotonic()
        for request_id, record in tuple(records.items()):
            handle = record.handle

            # A provisional parent claim waits entirely in the child until
            # the request actually starts. This avoids START notifications on
            # every wrong speculation while preserving running-exact reuse.
            if record.claim_requested and not record.claimed:
                if handle.started_at is not None:
                    claimed = sidecar.try_claim(handle.key)
                    if claimed is not None:
                        record.claimed = True
                    elif handle.future.done():
                        # The terminal branch below provides the precise error.
                        pass
                    else:
                        emit(
                            _ProcessEvent(
                                "claim_miss",
                                request_id,
                                ("SidecarExpired", "exact handle is no longer claimable"),
                            )
                        )
                        sidecar.try_tombstone(
                            session_id=handle.key.session_id,
                            decision_id=handle.key.decision_id,
                        )
                        discard_record(request_id)
                        continue
                elif (
                    record.start_deadline is not None
                    and record.start_deadline <= now
                ):
                    emit(
                        _ProcessEvent(
                            "claim_miss",
                            request_id,
                            ("SidecarExpired", "exact candidate never started in time"),
                        )
                    )
                    sidecar.try_tombstone(
                        session_id=handle.key.session_id,
                        decision_id=handle.key.decision_id,
                    )
                    discard_record(request_id)
                    continue

            if not handle.future.done():
                continue
            try:
                error = handle.future.exception()
            except concurrent.futures.CancelledError as exc:
                error = exc
            if (
                staging_enabled
                and not record.claim_requested
                and not record.claimed
            ):
                # A finite lease remains the logical visibility fence even if
                # physical execution finished first.  Let the ordinary lease
                # collector retire stale work without exporting it.
                if (
                    record.lease_until is not None
                    and record.lease_until <= now
                ):
                    continue
                if record.eager_transferred:
                    if (
                        error is not None
                        or sidecar._try_retire_ready_for_staging(
                            handle.key
                        )
                        or (
                            handle.finished_at is not None
                            and handle.finished_at + result_ttl_s <= now
                        )
                    ):
                        discard_record(request_id)
                    continue

                finished_at = handle.finished_at or now
                valid_until = finished_at + result_ttl_s
                if error is not None:
                    outcome = _ProcessStagedOutcome(
                        result_payload=None,
                        error_payload=_remote_error_payload(error),
                        started_at=handle.started_at,
                        finished_at=finished_at,
                        valid_until=valid_until,
                    )
                    emitted = emit(
                        _ProcessEvent("staged", request_id, outcome),
                        best_effort=pull_result_staging,
                    )
                    eager_stats[
                        "failure_events" if emitted else "dropped_events"
                    ] += 1
                    # Failed inner jobs are already retired by the scheduler.
                    discard_record(request_id)
                    continue

                try:
                    encoded_result = pickle.dumps(
                        handle.future.result(),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    staged_error = (
                        None
                        if len(encoded_result) <= max_staged_result_bytes
                        else (
                            "ResultTooLarge",
                            "serialized speculative result exceeds the "
                            "staging byte cap",
                        )
                    )
                    if staged_error is not None:
                        encoded_result = None
                except BaseException as exc:
                    encoded_result = None
                    staged_error = (
                        "ResultSerializationError",
                        str(exc),
                    )
                outcome = _ProcessStagedOutcome(
                    result_payload=encoded_result,
                    error_payload=staged_error,
                    started_at=handle.started_at,
                    finished_at=finished_at,
                    valid_until=valid_until,
                )
                emitted = emit(
                    _ProcessEvent("staged", request_id, outcome),
                    best_effort=pull_result_staging,
                )
                eager_stats[
                    (
                        "result_events"
                        if emitted and staged_error is None
                        else "failure_events"
                        if emitted
                        else "dropped_events"
                    )
                ] += 1
                # Ownership has moved out of the child whether or not the
                # bounded event socket accepted the payload.  Claiming only the
                # child-local handle releases scheduler retention; it does not
                # expose the value to the agent.
                record.eager_transferred = True
                if sidecar._try_retire_ready_for_staging(handle.key):
                    discard_record(request_id)
                continue
            if error is not None:
                # An unclaimed failure is just a wrong speculation and must
                # not wake a lazy parent bridge or consume its event buffer.
                # Exact provisional claims still receive a reliable terminal.
                if record.claim_requested or record.claimed:
                    emit(
                        _ProcessEvent(
                            "terminal",
                            request_id,
                            _remote_error_payload(error),
                        )
                    )
                discard_record(request_id)
                continue
            if record.claimed:
                try:
                    encoded = pickle.dumps(
                        handle.future.result(), protocol=pickle.HIGHEST_PROTOCOL
                    )
                except BaseException as exc:
                    emit(
                        _ProcessEvent(
                            "terminal",
                            request_id,
                            ("ResultSerializationError", str(exc)),
                        )
                    )
                else:
                    emit(_ProcessEvent("result", request_id, encoded))
                discard_record(request_id)
            else:
                record.ready_unclaimed = True

    def admit_submission(command: _ProcessSubmit) -> bool:
        """Admit one batch member through the existing bounded scheduler."""

        handle = sidecar.try_submit(
            command.invocation,
            session_id=command.session_id,
            decision_id=command.decision_id,
            context_token=command.context_token,
            priority=command.priority,
            start_deadline=command.start_deadline,
        )
        if handle is None:
            # Parent admission was provisional. Keep this silent until an
            # exact claim (which will receive claim_miss) or parent lease reap.
            return False
        lease_until = (
            None
            if command.start_deadline is None
            else command.start_deadline + claim_grace_s
        )
        if lease_until is not None and not math.isfinite(lease_until):
            # Parent validation normally makes this unreachable. Fail closed
            # as a speculative miss if an incompatible sender bypasses it.
            lease_until = time.monotonic()
        records[command.request_id] = _ChildTransportRecord(
            handle=handle,
            start_deadline=command.start_deadline,
            lease_until=lease_until,
        )
        return True

    def stage_scheduled(command: _ProcessScheduleBatches) -> None:
        """Bound and retain future work without touching the inner scheduler."""

        nonlocal scheduled_order
        candidate_count = sum(
            len(batch.submissions) for batch in command.batches
        )
        scheduled_stats["received_batches"] += len(command.batches)
        scheduled_stats["received_candidates"] += candidate_count
        if (
            candidate_count <= 0
            or len(scheduled_records) + candidate_count
            > max_scheduled_pending
        ):
            scheduled_stats["capacity_dropped"] += candidate_count
            return

        now = time.monotonic()
        for batch in command.batches:
            if (
                not batch.submissions
                or batch.release_at >= batch.start_deadline
                or batch.start_deadline <= now
            ):
                scheduled_stats["deadline_dropped"] += len(
                    batch.submissions
                )
                continue
            request_ids: list[int] = []
            for submission in batch.submissions:
                scheduled_records[submission.request_id] = (
                    _ChildScheduledRecord(
                        submission=submission,
                        release_at=batch.release_at,
                    )
                )
                request_ids.append(submission.request_id)
            if request_ids:
                scheduled_order += 1
                heapq.heappush(
                    scheduled_heap,
                    (batch.release_at, scheduled_order, tuple(request_ids)),
                )

    def prune_scheduled_heap() -> None:
        while scheduled_heap and not any(
            request_id in scheduled_records
            for request_id in scheduled_heap[0][2]
        ):
            heapq.heappop(scheduled_heap)

    def compact_scheduled_heap() -> None:
        """Discard every stale timer node after a pre-release cancellation."""

        compacted: list[tuple[float, int, tuple[int, ...]]] = []
        for release_at, order, request_ids in scheduled_heap:
            live_ids = tuple(
                request_id
                for request_id in request_ids
                if request_id in scheduled_records
            )
            if live_ids:
                compacted.append((release_at, order, live_ids))
        scheduled_heap[:] = compacted
        heapq.heapify(scheduled_heap)

    def next_scheduled_release() -> float | None:
        prune_scheduled_heap()
        return scheduled_heap[0][0] if scheduled_heap else None

    def release_due_scheduled(now: float) -> None:
        """Move due preloads into the ordinary bounded scheduler."""

        prune_scheduled_heap()
        while scheduled_heap and scheduled_heap[0][0] <= now:
            _, _, request_ids = heapq.heappop(scheduled_heap)
            live_ids = [
                request_id
                for request_id in request_ids
                if request_id in scheduled_records
            ]
            if live_ids:
                scheduled_stats["released_batches"] += 1
            for request_id in live_ids:
                scheduled = scheduled_records.pop(request_id)
                submission = scheduled.submission
                deadline = submission.start_deadline
                if deadline is None or deadline <= now:
                    scheduled_stats["deadline_dropped"] += 1
                    if scheduled.claim_requested:
                        emit(
                            _ProcessEvent(
                                "claim_miss",
                                request_id,
                                (
                                    "SidecarExpired",
                                    "scheduled exact candidate missed deadline",
                                ),
                            )
                        )
                    continue
                scheduled_stats["released_candidates"] += 1
                admitted = admit_submission(submission)
                if not admitted:
                    scheduled_stats["admission_dropped"] += 1
                if scheduled.claim_requested:
                    record = records.get(request_id)
                    if admitted and record is not None:
                        record.claim_requested = True
                    else:
                        emit(
                            _ProcessEvent(
                                "claim_miss",
                                request_id,
                                (
                                    "SidecarRejected",
                                    "scheduled exact candidate was not admitted",
                                ),
                            )
                        )
            prune_scheduled_heap()

    def cancel_scheduled_tombstones(
        tombstones: tuple[_ProcessTombstone, ...],
    ) -> None:
        cancelled = 0
        for request_id, scheduled in tuple(scheduled_records.items()):
            submission = scheduled.submission
            matched = any(
                submission.session_id == tombstone.session_id
                and (
                    tombstone.decision_id is None
                    or submission.decision_id == tombstone.decision_id
                )
                for tombstone in tombstones
            )
            if matched and not scheduled.claim_requested:
                scheduled_records.pop(request_id, None)
                cancelled += 1
        scheduled_stats["tombstoned"] += cancelled
        if cancelled:
            # Capacity is a memory bound as well as an admission bound. A
            # live early timer must not pin arbitrarily many stale later heap
            # nodes across repeated schedule/tombstone cycles.
            compact_scheduled_heap()

    def cancel_all_scheduled_on_close() -> None:
        scheduled_stats["closed_unreleased"] += len(scheduled_records)
        scheduled_records.clear()
        scheduled_heap.clear()

    def apply_tombstones(
        tombstones: tuple[_ProcessTombstone, ...],
    ) -> None:
        # Pre-release work is parent-owned provisional state. Once the parent
        # has tombstoned it, cancellation cannot depend on inner ingress room.
        cancel_scheduled_tombstones(tombstones)
        accepted: list[_ProcessTombstone] = []
        for tombstone in tombstones:
            if sidecar.try_tombstone(
                session_id=tombstone.session_id,
                decision_id=tombstone.decision_id,
            ):
                accepted.append(tombstone)
        for request_id, record in tuple(records.items()):
            key = record.handle.key
            matched = any(
                key.session_id == tombstone.session_id
                and (
                    tombstone.decision_id is None
                    or key.decision_id == tombstone.decision_id
                )
                for tombstone in accepted
            )
            if (
                matched
                and not record.claim_requested
                and not record.claimed
            ):
                # Parent already made this unclaimed handle terminal. The
                # child scheduler owns the lazy physical drain.
                discard_record(request_id)

    def expire_unclaimed_leases(now: float) -> bool:
        """Logically retire expired wrong guesses without parent traffic.

        A full child control queue is retried locally. Dropping the transport
        record before the scheduler owns its tombstone could otherwise allow
        queued physical work to start after its lease expired.
        """

        expired_by_decision: dict[
            tuple[str, str], list[int]
        ] = {}
        for request_id, record in records.items():
            lease_until = record.lease_until
            if (
                lease_until is None
                or lease_until > now
                or record.claim_requested
                or record.claimed
            ):
                continue
            key = record.handle.key
            expired_by_decision.setdefault(
                (key.session_id, key.decision_id), []
            ).append(request_id)

        retry_needed = False
        for (session_id, decision_id), request_ids in expired_by_decision.items():
            request_id_set = set(request_ids)
            protected_generation_exists = any(
                request_id not in request_id_set
                and record.handle.key.session_id == session_id
                and record.handle.key.decision_id == decision_id
                and (
                    record.claim_requested
                    or record.claimed
                    or record.lease_until is None
                    or record.lease_until > now
                )
                for request_id, record in records.items()
            )
            if protected_generation_exists:
                # Coarse scheduler tombstones are decision-scoped. Never let
                # an old generation's lease kill a newer/claimed generation.
                unresolved = False
                for request_id in request_ids:
                    record = records.get(request_id)
                    if record is None:
                        continue
                    if record.handle.future.done():
                        lease_stats["expired"] += 1
                        discard_record(request_id)
                    else:
                        unresolved = True
                retry_needed = retry_needed or unresolved
                continue
            if not sidecar.try_tombstone(
                session_id=session_id,
                decision_id=decision_id,
            ):
                lease_stats["tombstone_retries"] += 1
                retry_needed = True
                continue
            lease_stats["tombstones_enqueued"] += 1
            lease_stats["expired"] += len(request_ids)
            for request_id in request_ids:
                discard_record(request_id)
        return retry_needed

    def attach_lease_snapshot(snapshot: dict[str, Any]) -> None:
        finite_unclaimed = [
            record.lease_until
            for record in records.values()
            if record.lease_until is not None
            and not record.claim_requested
            and not record.claimed
        ]
        snapshot["lease"] = {
            **lease_stats,
            "claim_grace_s": claim_grace_s,
            "live_finite": len(finite_unclaimed),
            "next_expiry": (
                min(finite_unclaimed) if finite_unclaimed else None
            ),
        }
        next_release = next_scheduled_release()
        snapshot["scheduled"] = {
            **scheduled_stats,
            "capacity": max_scheduled_pending,
            "pending": len(scheduled_records),
            "heap_nodes": len(scheduled_heap),
            "next_release": next_release,
        }
        snapshot["eager_result_staging"] = dict(eager_stats)
        snapshot["requested_scheduler_policy"] = requested_scheduler_policy
        snapshot["actual_scheduler_policy"] = actual_scheduler_policy
        snapshot["scheduler_priority_error"] = scheduler_priority_error

    closing = False
    close_timeout = 5.0
    lease_retry_needed = False
    deferred_submit: (
        _ProcessSubmitBatch
        | _ProcessSubmit
        | _ProcessScheduleBatches
        | None
    ) = None
    while not closing:
        command: Any | None = None
        if lease_retry_needed:
            # Preserve control ordering: an expired old generation must be
            # handed to the scheduler before a later Submit can be admitted.
            # This wait is child-local and never holds authority resources.
            time.sleep(0.001)
            lease_retry_needed = expire_unclaimed_leases(time.monotonic())
            scan_records()
            if lease_retry_needed:
                continue

        # Do not release before one command receive attempt. Control is
        # linearized when the child dequeues it: a dequeued Close/Tombstone
        # fences later release, while a due timer may beat a later datagram
        # that is still queued behind the command handled this iteration.
        scan_records()

        if deferred_submit is not None:
            command = deferred_submit
            deferred_submit = None

        # Wrong/unclaimed speculation produces no parent result, so it needs
        # no bridge polling at all: submit, tombstone, snapshot, and close
        # datagrams wake this process naturally. Poll briefly only while an
        # exact claim is waiting for a running child result. This removes the
        # prior 1000 wakeups/s from all-wrong runs.
        if command is None:
            claim_waiting = any(
                record.claim_requested for record in records.values()
            )
            eager_transfer_waiting = (
                staging_enabled
                and bool(records)
            )
            finite_unclaimed_leases = [
                record.lease_until
                for record in records.values()
                if record.lease_until is not None
                and not record.claim_requested
                and not record.claimed
            ]
            receive_timeout: float | None = (
                0.001
                if claim_waiting or eager_transfer_waiting
                else None
            )
            if finite_unclaimed_leases:
                lease_delay = max(
                    0.0, min(finite_unclaimed_leases) - time.monotonic()
                )
                receive_timeout = (
                    lease_delay
                    if receive_timeout is None
                    else min(receive_timeout, lease_delay)
                )
            scheduled_release = next_scheduled_release()
            if scheduled_release is not None:
                release_delay = max(
                    0.0, scheduled_release - time.monotonic()
                )
                receive_timeout = (
                    release_delay
                    if receive_timeout is None
                    else min(receive_timeout, release_delay)
                )
            if receive_timeout is not None:
                receive_timeout = min(
                    receive_timeout, _MAX_CHILD_SOCKET_TIMEOUT_S
                )
            command_socket.settimeout(receive_timeout)
            try:
                encoded_command = command_socket.recv(max_packet_bytes)
                if not encoded_command:
                    closing = True
                else:
                    command = pickle.loads(encoded_command)
            except (socket.timeout, BlockingIOError):
                pass
            except (EOFError, OSError, pickle.PickleError):
                closing = True

        snapshot_request_id: int | None = None
        if isinstance(command, _ProcessSubmitBatch):
            lease_retry_needed = expire_unclaimed_leases(time.monotonic())
            if lease_retry_needed:
                deferred_submit = command
            else:
                for submission in command.submissions:
                    admit_submission(submission)
        elif isinstance(command, _ProcessSubmit):
            # Kept for wire compatibility with earlier single-submit callers.
            lease_retry_needed = expire_unclaimed_leases(time.monotonic())
            if lease_retry_needed:
                deferred_submit = command
            else:
                admit_submission(command)
        elif isinstance(command, _ProcessScheduleBatches):
            # Free capacity from older due timers before atomically staging a
            # new future packet. Control messages take their own receive
            # branch and therefore still win that receive iteration.
            release_due_scheduled(time.monotonic())
            lease_retry_needed = expire_unclaimed_leases(time.monotonic())
            if lease_retry_needed:
                deferred_submit = command
            else:
                stage_scheduled(command)
        elif isinstance(command, _ProcessClaim):
            record = records.get(command.request_id)
            scheduled = scheduled_records.get(command.request_id)
            if record is not None:
                record.claim_requested = True
            elif scheduled is not None:
                # Preserve an early exact claim until the timer admits it.
                scheduled.claim_requested = True
            else:
                emit(
                    _ProcessEvent(
                        "claim_miss",
                        command.request_id,
                        ("SidecarExpired", "unknown or retired exact handle"),
                    )
                )
        elif isinstance(command, _ProcessTombstoneBatch):
            apply_tombstones(command.tombstones)
        elif isinstance(command, _ProcessTombstone):
            # Kept for wire compatibility with earlier single-submit callers.
            apply_tombstones((command,))
        elif isinstance(command, _ProcessSnapshot):
            snapshot_request_id = command.request_id
        elif isinstance(command, _ProcessClose):
            close_timeout = command.drain_timeout_s
            # Close is a hard fence for work whose release has not happened.
            cancel_all_scheduled_on_close()
            closing = True

        lease_retry_needed = expire_unclaimed_leases(time.monotonic())
        if not closing and not lease_retry_needed:
            release_due_scheduled(time.monotonic())
        scan_records()
        if snapshot_request_id is not None:
            # The scheduler snapshot is ordered after any lease tombstones
            # successfully enqueued above.
            snapshot = sidecar.snapshot()
            snapshot["process_pid"] = os.getpid()
            snapshot["requested_cpu_affinity"] = requested_affinity
            snapshot["actual_cpu_affinity"] = actual_affinity
            attach_lease_snapshot(snapshot)
            emit(_ProcessEvent("snapshot", snapshot_request_id, snapshot))

    close_error: BaseException | None = None
    try:
        sidecar.close(wait=True, timeout=close_timeout)
    except BaseException as exc:
        close_error = exc
    scan_records()
    try:
        final_snapshot = sidecar.snapshot()
    except BaseException:
        final_snapshot = {
            "started": 0,
            "max_running": 0,
            "counts": {},
            "stats": {},
        }
    final_snapshot["process_pid"] = os.getpid()
    final_snapshot["requested_cpu_affinity"] = requested_affinity
    final_snapshot["actual_cpu_affinity"] = actual_affinity
    attach_lease_snapshot(final_snapshot)
    if close_error is not None:
        final_snapshot["close_error"] = repr(close_error)
    emit(_ProcessEvent("closed", payload=final_snapshot))
    command_socket.close()
    event_socket.close()


class ProcessSpeculativeSidecar:
    """Linux-fork speculative sidecar with an authority-safe parent API.

    The executor and :class:`SpeculativeSidecar` scheduler run in a fork child,
    isolating their GIL and event-loop callbacks from authority. Command and
    event transports are bounded Linux ``AF_UNIX/SOCK_SEQPACKET`` socketpairs,
    so no multiprocessing Queue feeder thread runs in the parent. Parent-side
    ``try_submit``, ``try_claim``, and ``try_tombstone`` never wait for child
    work; queue saturation is a fail-open miss.

    ``try_claim`` is provisional: it returns the exact submitted handle and
    sends a non-blocking claim message. If the child candidate was still
    queued, expired, or rejected, the handle completes with an exception and
    the already-submitted authority call remains the sole result path.
    Successful result payloads cross the process boundary only for provisional
    exact claims. The parent result bridge is also lazy: all-wrong traffic has
    no parent reader thread; the first exact claim, snapshot, or close starts
    it. Unclaimed child failures and finite-lease cleanup are silent. The
    default-off ``eager_result_staging`` experiment instead starts that bridge
    eagerly and transfers completed values into a bounded private parent map.
    The mutually exclusive ``pull_result_staging`` mode leaves best-effort
    staged events in the bounded kernel socket and drains at most
    ``max_pending`` packets non-blockingly. A caller may drain during a guard
    window with :meth:`prefetch_pull_results`, which seals the epoch so later
    exact claims are parent-local O(1) lookups and cannot read the socket. It
    creates no result thread on the request path and sends no claim packet.
    Public handles remain pending until exact confirmation consumes a value.

    A finite ``start_deadline`` gives the child an unclaimed lease ending at
    ``start_deadline + claim_grace_s``. Parent registry state is reaped at the
    next submit, while exact claim checks remain O(1). Custom ``clock`` values
    must share Linux ``time.monotonic``'s domain when finite leases or eager
    result staging are used.

    ``cpu_affinity`` is applied in the fork child before its scheduler thread
    is created, so the executor inherits the same CPU mask. The parent result
    bridge pins itself to that mask before reading events, sharing the sidecar
    CPU budget rather than the authority CPU. Supplying a mask disjoint from
    the authority process turns CPU scheduling into an explicit part of the
    sidecar resource certificate.
    """

    def __init__(
        self,
        executor: SidecarExecutor,
        max_workers: int = 1,
        max_pending: int | None = None,
        *,
        ingress_capacity: int | None = None,
        result_capacity: int | None = None,
        max_scheduled_pending: int | None = None,
        max_packet_bytes: int = 256 * 1024,
        cpu_affinity: set[int] | None = None,
        claim_grace_s: float = 0.010,
        result_ttl_s: float = 30.0,
        eager_result_staging: bool = False,
        pull_result_staging: bool = False,
        max_staged_result_bytes: int | None = None,
        autostart: bool = False,
        clock: Callable[[], float] = time.monotonic,
        process_name: str = "speculative-sidecar-process",
    ) -> None:
        if not callable(executor):
            raise TypeError("executor must be callable")
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer")
        if max_pending is None:
            max_pending = 2 * max_workers
        if (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending < max_workers
        ):
            raise ValueError("max_pending must be an integer >= max_workers")
        if ingress_capacity is None:
            ingress_capacity = max(4, 2 * max_pending)
        if (
            isinstance(ingress_capacity, bool)
            or not isinstance(ingress_capacity, int)
            or ingress_capacity <= 0
        ):
            raise ValueError("ingress_capacity must be a positive integer")
        if result_capacity is None:
            result_capacity = max(16, 4 * ingress_capacity)
        if (
            isinstance(result_capacity, bool)
            or not isinstance(result_capacity, int)
            or result_capacity <= 0
        ):
            raise ValueError("result_capacity must be a positive integer")
        if max_scheduled_pending is None:
            max_scheduled_pending = max(64, 8 * max_pending)
        if (
            isinstance(max_scheduled_pending, bool)
            or not isinstance(max_scheduled_pending, int)
            or max_scheduled_pending <= 0
        ):
            raise ValueError(
                "max_scheduled_pending must be a positive integer"
            )
        if (
            isinstance(max_packet_bytes, bool)
            or not isinstance(max_packet_bytes, int)
            or max_packet_bytes < 4096
        ):
            raise ValueError("max_packet_bytes must be an integer >= 4096")
        if cpu_affinity is not None:
            if not isinstance(cpu_affinity, set) or not cpu_affinity:
                raise ValueError("cpu_affinity must be a non-empty set or None")
            if any(
                isinstance(cpu, bool)
                or not isinstance(cpu, int)
                or cpu < 0
                for cpu in cpu_affinity
            ):
                raise ValueError(
                    "cpu_affinity entries must be non-negative integers"
                )
        if (
            isinstance(claim_grace_s, bool)
            or not isinstance(claim_grace_s, (int, float))
            or not math.isfinite(claim_grace_s)
            or claim_grace_s < 0.0
        ):
            raise ValueError("claim_grace_s must be finite and non-negative")
        if (
            isinstance(result_ttl_s, bool)
            or not isinstance(result_ttl_s, (int, float))
            or not math.isfinite(result_ttl_s)
            or result_ttl_s <= 0.0
        ):
            raise ValueError("result_ttl_s must be finite and positive")
        if not isinstance(eager_result_staging, bool):
            raise TypeError("eager_result_staging must be a bool")
        if not isinstance(pull_result_staging, bool):
            raise TypeError("pull_result_staging must be a bool")
        if eager_result_staging and pull_result_staging:
            raise ValueError(
                "eager_result_staging and pull_result_staging are mutually "
                "exclusive"
            )
        if max_staged_result_bytes is None:
            max_staged_result_bytes = max_packet_bytes
        if (
            isinstance(max_staged_result_bytes, bool)
            or not isinstance(max_staged_result_bytes, int)
            or max_staged_result_bytes <= 0
            or max_staged_result_bytes > max_packet_bytes
        ):
            raise ValueError(
                "max_staged_result_bytes must be a positive integer no "
                "larger than max_packet_bytes"
            )
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError(
                "ProcessSpeculativeSidecar requires the Linux fork start method"
            )

        self._executor = executor
        self._max_workers = max_workers
        self._max_pending = max_pending
        self._ingress_capacity = ingress_capacity
        self._result_capacity = result_capacity
        self._max_scheduled_pending = max_scheduled_pending
        self._max_packet_bytes = max_packet_bytes
        self._cpu_affinity = (
            None if cpu_affinity is None else tuple(sorted(cpu_affinity))
        )
        self._claim_grace_s = float(claim_grace_s)
        self._result_ttl_s = float(result_ttl_s)
        self._eager_result_staging = eager_result_staging
        self._pull_result_staging = pull_result_staging
        self._max_staged_result_bytes = max_staged_result_bytes
        self._max_pull_staged_packet_bytes = min(
            max_packet_bytes,
            max_staged_result_bytes
            + _PULL_STAGED_ENVELOPE_HEADROOM_BYTES,
        )
        self._pull_registry_capacity = max(max_pending, result_capacity)
        self._clock = clock
        self._process_name = process_name
        self._context = multiprocessing.get_context("fork")
        self._command_parent, self._command_child = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        self._event_parent, self._event_child = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        # Kernel socket buffers are the bounded queues. Linux may clamp these
        # hints via net.core.{wmem,rmem}_max; EAGAIN remains the admission
        # signal irrespective of the exact granted size.
        command_buffer = max(16 * 1024, ingress_capacity * 4096)
        event_buffer = max(64 * 1024, result_capacity * 4096)
        self._command_parent.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, command_buffer
        )
        self._command_child.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, command_buffer
        )
        self._event_child.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, event_buffer
        )
        self._event_parent.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, event_buffer
        )
        self._command_parent.setblocking(False)
        self._command_send_lock = threading.Lock()
        # Pull claims never wait for a lifecycle bridge that may own recv().
        # The bridge uses the same lock only around the actual receive call.
        self._event_receive_lock = threading.Lock()

        self._lifecycle_lock = threading.Lock()
        self._registry_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._bridge_lock = threading.Lock()
        self._bridge_affinity_ready = threading.Event()
        self._bridge_actual_cpu_affinity: tuple[int, ...] | None = None
        self._bridge_affinity_error: str | None = None
        self._stopped = threading.Event()
        self._started = False
        self._accepting = True
        self._closing = False
        self._closed = False
        self._close_sent = False
        self._process: multiprocessing.Process | None = None
        self._bridge: threading.Thread | None = None
        self._next_request_id = 0
        self._handles: dict[int, SpeculativeHandle] = {}
        self._available: dict[ExactSpeculationKey, int] = {}
        self._leases: dict[int, float] = {}
        self._scheduled_releases: dict[int, float] = {}
        self._staged_results: dict[int, _ParentStagedOutcome] = {}
        self._snapshot_waiters: dict[
            int, concurrent.futures.Future[dict[str, Any]]
        ] = {}
        self._pull_epoch_lock = threading.Lock()
        self._pull_epoch_sealed = False
        self._last_snapshot = self._empty_snapshot()
        self._transport_stats: dict[str, int] = {
            "transport_submitted": 0,
            "transport_submit_packets": 0,
            "transport_scheduled": 0,
            "transport_schedule_batches": 0,
            "transport_schedule_packets": 0,
            "transport_schedule_capacity_rejected": 0,
            "transport_ingress_full": 0,
            "transport_claims": 0,
            "transport_claim_packets": 0,
            "transport_claim_misses": 0,
            "transport_claim_busy": 0,
            "transport_tombstones": 0,
            "transport_tombstone_packets": 0,
            "transport_lease_reaped": 0,
            "transport_claim_expired": 0,
            "transport_results": 0,
            "transport_terminal": 0,
            "transport_staged_results": 0,
            "transport_staged_failures": 0,
            "transport_stage_dropped": 0,
            "transport_eager_hits": 0,
            "transport_eager_not_ready": 0,
            "transport_pull_packets": 0,
            "transport_pull_prefetch_calls": 0,
            "transport_pull_prefetch_packets": 0,
            "transport_pull_prefetch_busy": 0,
            "transport_pull_decode_dropped": 0,
            "transport_pull_hits": 0,
            "transport_pull_not_ready": 0,
            "transport_pull_registry_full": 0,
        }

        if autostart:
            self.start()

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None else None

    @property
    def bridge_started(self) -> bool:
        """Whether a setup, exact claim, or lifecycle call started the bridge."""

        return self._bridge is not None

    @property
    def pull_epoch_sealed(self) -> bool:
        """Whether exact pull claims are restricted to parent-local state."""

        with self._pull_epoch_lock:
            return self._pull_epoch_sealed

    def _set_pull_epoch_sealed(self, sealed: bool) -> None:
        with self._pull_epoch_lock:
            self._pull_epoch_sealed = sealed

    def start_result_bridge(self, *, timeout: float = 5.0) -> bool:
        """Start the blocking result bridge outside an authority hot path.

        The bridge waits in the kernel on the event socket and child sentinel;
        it has no periodic Python wakeup while the child is quiet. Callers that
        already know a safe start budget is positive can therefore pay thread
        creation during setup instead of on the first exact authority claim.
        Success certifies that bridge-local CPU affinity setup and readback
        completed within ``timeout``.
        """

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0.0
        ):
            raise ValueError("timeout must be finite and non-negative")
        if not self._ensure_bridge():
            return False
        if not self._bridge_affinity_ready.wait(float(timeout)):
            return False
        with self._bridge_lock:
            return self._bridge_affinity_error is None

    def startup_snapshot(self, *, timeout: float = 2.0) -> dict[str, Any]:
        """Read one pre-admission child certificate without starting a bridge.

        This setup-only operation is valid before any speculative handle has
        been registered. It performs one bounded command send and consumes at
        most two response datagrams directly. Unlike :meth:`snapshot`, it does
        not create or certify a parent result thread, so pull staging remains
        pull-only when timed execution begins.
        """

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0.0
        ):
            raise ValueError("timeout must be finite and non-negative")
        deadline = time.monotonic() + float(timeout)

        # Hold lifecycle and bridge creation fences for the whole one-shot
        # exchange. close() and start_result_bridge() may wait here because
        # this API is explicitly outside the authority hot path.
        with self._lifecycle_lock:
            if not self._started or self._closed or self._closing:
                raise SidecarClosed(
                    "startup snapshot requires a live process sidecar"
                )
            with self._registry_lock:
                if self._handles or self._snapshot_waiters:
                    raise SidecarRejected(
                        "startup snapshot is valid only before submissions"
                    )
            remaining = max(0.0, deadline - time.monotonic())
            if not self._bridge_lock.acquire(timeout=remaining):
                raise TimeoutError("startup snapshot bridge fence timed out")
            try:
                if self._bridge is not None:
                    raise SidecarRejected(
                        "startup snapshot requires an unstarted result bridge"
                    )
                remaining = max(0.0, deadline - time.monotonic())
                if not self._event_receive_lock.acquire(timeout=remaining):
                    raise TimeoutError(
                        "startup snapshot result socket was busy"
                    )
                try:
                    request_id = self._next_id()
                    remaining = max(0.0, deadline - time.monotonic())
                    self._send_with_timeout(
                        _ProcessSnapshot(request_id), remaining / 2
                    )
                    raw_snapshot: Any | None = None
                    closed_snapshot: Any | None = None
                    # No submission is allowed before this call, so the only
                    # legal packets are startup CLOSED and our SNAPSHOT. Two
                    # iterations keep even protocol-corrupt peers bounded.
                    for _ in range(2):
                        remaining = max(0.0, deadline - time.monotonic())
                        readable, _, _ = select.select(
                            [self._event_parent], [], [], remaining
                        )
                        if not readable:
                            raise TimeoutError(
                                "startup snapshot response timed out"
                            )
                        encoded = self._event_parent.recv(
                            self._max_packet_bytes,
                            socket.MSG_DONTWAIT,
                        )
                        if not encoded:
                            raise SidecarClosed(
                                "process closed during startup snapshot"
                            )
                        event = pickle.loads(encoded)
                        if not isinstance(event, _ProcessEvent):
                            continue
                        if event.kind == "closed":
                            closed_snapshot = event.payload
                            break
                        if (
                            event.kind == "snapshot"
                            and event.request_id == request_id
                        ):
                            raw_snapshot = event.payload
                            break
                    if raw_snapshot is None and closed_snapshot is None:
                        raise SidecarClosed(
                            "invalid process startup snapshot response"
                        )
                except (BlockingIOError, EOFError, OSError, pickle.PickleError) as exc:
                    raise SidecarClosed(
                        "could not read process startup snapshot"
                    ) from exc
                finally:
                    self._event_receive_lock.release()
            finally:
                self._bridge_lock.release()

        if closed_snapshot is not None:
            self._last_snapshot = self._decorate_snapshot(closed_snapshot)
            self._fail_all(SidecarClosed("process sidecar closed at startup"))
            with self._lifecycle_lock:
                self._closed = True
            self._stopped.set()
            self._close_parent_transport()
            return self._last_snapshot
        return self._decorate_snapshot(raw_snapshot)

    def _increment_transport(self, name: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._transport_stats[name] += amount

    def _empty_snapshot(self) -> dict[str, Any]:
        return {
            "capacity": {
                "max_workers": self._max_workers,
                "max_pending": self._max_pending,
                "ingress_capacity": self._ingress_capacity,
                "result_capacity": self._result_capacity,
                "max_scheduled_pending": self._max_scheduled_pending,
                "max_packet_bytes": self._max_packet_bytes,
                "claim_grace_s": self._claim_grace_s,
                "eager_result_staging": self._eager_result_staging,
                "pull_result_staging": self._pull_result_staging,
                "pull_registry_capacity": self._pull_registry_capacity,
                "max_staged_result_bytes": (
                    self._max_staged_result_bytes
                ),
                "max_pull_staged_packet_bytes": (
                    self._max_pull_staged_packet_bytes
                ),
            },
            "counts": {
                "pending": 0,
                "queued": 0,
                "running": 0,
                "ready": 0,
                "published": 0,
                "ingress": 0,
            },
            "stats": {"started": 0, "max_running": 0},
            "started": 0,
            "max_running": 0,
            "process_alive": False,
            "process_pid": None,
            "requested_cpu_affinity": (
                None
                if self._cpu_affinity is None
                else list(self._cpu_affinity)
            ),
            "actual_cpu_affinity": None,
            "bridge_started": False,
            "requested_bridge_cpu_affinity": (
                None
                if self._cpu_affinity is None
                else list(self._cpu_affinity)
            ),
            "actual_bridge_cpu_affinity": (
                None
                if self._bridge_actual_cpu_affinity is None
                else list(self._bridge_actual_cpu_affinity)
            ),
            "bridge_affinity_ready": self._bridge_affinity_ready.is_set(),
            "bridge_affinity_error": self._bridge_affinity_error,
            "transport": {},
            "parent_staging": {
                "enabled": (
                    self._eager_result_staging
                    or self._pull_result_staging
                ),
                "mode": (
                    "eager"
                    if self._eager_result_staging
                    else "pull"
                    if self._pull_result_staging
                    else "disabled"
                ),
                "ready": 0,
                "capacity": self._result_capacity,
                "max_result_bytes": self._max_staged_result_bytes,
                "pull_epoch_sealed": self.pull_epoch_sealed,
            },
        }

    def start(self, *, timeout: float = 5.0) -> None:
        """Fork the executor process; keep the parent bridge lazy."""

        del timeout  # Process.start() is synchronous; child readiness is FIFO.
        with self._lifecycle_lock:
            if self._closed or self._closing:
                raise SidecarClosed("cannot start a closed process sidecar")
            if self._started:
                return
            self._process = self._context.Process(
                target=_process_sidecar_worker,
                args=(
                    self._executor,
                    self._max_workers,
                    self._max_pending,
                    max(4, 2 * self._max_pending),
                    self._result_ttl_s,
                    self._command_child,
                    self._event_child,
                    self._command_parent,
                    self._event_parent,
                    self._max_packet_bytes,
                    self._cpu_affinity,
                    self._claim_grace_s,
                    self._max_scheduled_pending,
                    self._eager_result_staging,
                    self._pull_result_staging,
                    self._max_staged_result_bytes,
                ),
                name=self._process_name,
                daemon=True,
            )
            self._process.start()
            # The child owns these ends after fork. Closing them in the parent
            # is required for reliable peer-death detection.
            self._command_child.close()
            self._event_child.close()
            self._started = True
        if self._eager_result_staging:
            # This opt-in mode deliberately trades the all-wrong lazy-bridge
            # property for parent-local completions.  Never wait for bridge
            # affinity here; an early confirmation simply observes no staged
            # value and fails open.
            self._ensure_bridge()

    def _ensure_bridge(self) -> bool:
        """Start exactly one result bridge outside the all-wrong hot path."""

        start_error: SidecarClosed | None = None
        with self._bridge_lock:
            if self._stopped.is_set() or self._bridge_affinity_error is not None:
                return False
            if self._bridge is not None:
                return True
            if self._process is None:
                return False
            bridge = threading.Thread(
                target=self._bridge_events,
                name=f"{self._process_name}-result-bridge",
                daemon=True,
            )
            try:
                bridge.start()
            except RuntimeError as exc:
                message = f"could not start process result bridge: {exc}"
                self._bridge_affinity_error = message
                start_error = SidecarClosed(message)
            else:
                self._bridge = bridge
        if start_error is not None:
            self._abort_result_bridge(start_error)
            self._bridge_affinity_ready.set()
            return False
        return True

    def _next_id(self) -> int:
        with self._registry_lock:
            self._next_request_id += 1
            return self._next_request_id

    def _put_nowait(self, command: Any) -> bool:
        try:
            encoded = pickle.dumps(command, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException:
            self._increment_transport("transport_ingress_full")
            return False
        if len(encoded) > self._max_packet_bytes:
            self._increment_transport("transport_ingress_full")
            return False
        if not self._command_send_lock.acquire(blocking=False):
            self._increment_transport("transport_ingress_full")
            return False
        try:
            sent = self._command_parent.send(encoded)
            if sent == len(encoded):
                return True
            self._increment_transport("transport_ingress_full")
            return False
        except (BlockingIOError, BrokenPipeError, EOFError, OSError):
            self._increment_transport("transport_ingress_full")
            return False
        finally:
            self._command_send_lock.release()

    def _send_with_timeout(self, command: Any, timeout: float) -> None:
        """Lifecycle-only blocking send; hot-path methods never call this."""

        encoded = pickle.dumps(command, protocol=pickle.HIGHEST_PROTOCOL)
        if len(encoded) > self._max_packet_bytes:
            raise ValueError("process-sidecar command exceeds max_packet_bytes")
        if not self._command_send_lock.acquire(timeout=max(0.0, timeout)):
            raise TimeoutError("process-sidecar command transport is busy")
        try:
            _, writable, _ = select.select(
                [], [self._command_parent], [], max(0.0, timeout)
            )
            if not writable:
                raise TimeoutError("process-sidecar command socket is full")
            sent = self._command_parent.send(encoded)
            if sent != len(encoded):
                raise OSError("partial SOCK_SEQPACKET command send")
        finally:
            self._command_send_lock.release()

    def _reap_expired_for_submit(self) -> None:
        """Retire old unclaimed parent handles at the next admission point."""

        terminal: list[SpeculativeHandle] = []
        with self._registry_lock:
            now = self._clock()
            for request_id, staged in tuple(self._staged_results.items()):
                if staged.valid_until > now:
                    continue
                self._staged_results.pop(request_id, None)
                handle = self._handles.pop(request_id, None)
                self._leases.pop(request_id, None)
                self._scheduled_releases.pop(request_id, None)
                if handle is not None:
                    if self._available.get(handle.key) == request_id:
                        self._available.pop(handle.key, None)
                    terminal.append(handle)
            for request_id, release_at in tuple(
                self._scheduled_releases.items()
            ):
                if release_at <= now:
                    self._scheduled_releases.pop(request_id, None)
            for request_id, lease_until in tuple(self._leases.items()):
                if lease_until > now:
                    continue
                handle = self._handles.get(request_id)
                if handle is None:
                    self._leases.pop(request_id, None)
                    self._scheduled_releases.pop(request_id, None)
                    continue
                if handle.claimed:
                    # A successful provisional claim is owned by the result
                    # bridge, not by the unclaimed lease collector.
                    self._leases.pop(request_id, None)
                    self._scheduled_releases.pop(request_id, None)
                    continue
                self._leases.pop(request_id, None)
                self._scheduled_releases.pop(request_id, None)
                self._staged_results.pop(request_id, None)
                self._handles.pop(request_id, None)
                if self._available.get(handle.key) == request_id:
                    self._available.pop(handle.key, None)
                terminal.append(handle)
        for handle in terminal:
            if not handle.future.done():
                handle.future.set_exception(
                    SidecarExpired(
                        "unclaimed process speculation lease expired"
                    )
                )
        if terminal:
            self._increment_transport(
                "transport_lease_reaped", len(terminal)
            )

    def try_submit(
        self,
        invocation: Invocation,
        *,
        session_id: str,
        decision_id: str,
        priority: float,
        start_deadline: float | None = None,
        context_token: str = "",
    ) -> SpeculativeHandle | None:
        handles = self.try_submit_batch(
            ((invocation, session_id, decision_id, priority, context_token),),
            start_deadline=start_deadline,
        )
        return handles[0] if handles else None

    def try_submit_batch(
        self,
        entries: Iterable[ProcessSubmitEntry],
        *,
        start_deadline: float | None = None,
    ) -> tuple[SpeculativeHandle, ...]:
        """Atomically hand off candidates in one non-blocking datagram.

        Parent registration is all-or-nothing. An empty tuple means that no
        candidate was handed to the child (lifecycle gate, non-positive
        priority, oversized packet, busy send lock, or full socket). Once the
        packet is accepted, the child admits each member independently through
        the bounded scheduler, with any later rejection reported on that
        member's future.
        """

        if start_deadline is not None and (
            isinstance(start_deadline, bool)
            or not isinstance(start_deadline, (int, float))
            or not math.isfinite(start_deadline)
        ):
            raise ValueError("start_deadline must be finite or None")
        normalized_deadline = (
            None if start_deadline is None else float(start_deadline)
        )
        normalized_lease_until = (
            None
            if normalized_deadline is None
            else normalized_deadline + self._claim_grace_s
        )
        if (
            normalized_lease_until is not None
            and not math.isfinite(normalized_lease_until)
        ):
            raise ValueError("start_deadline + claim_grace_s must be finite")

        normalized: list[
            tuple[Invocation, str, str, float, str, ExactSpeculationKey]
        ] = []
        for index, entry in enumerate(entries):
            try:
                (
                    invocation,
                    session_id,
                    decision_id,
                    priority,
                    context_token,
                ) = entry
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "batch entry must be "
                    "(invocation, session_id, decision_id, priority, "
                    "context_token)"
                ) from exc
            if not isinstance(invocation, Invocation):
                raise TypeError(
                    f"batch entry {index} invocation must be an Invocation"
                )
            if (
                isinstance(priority, bool)
                or not isinstance(priority, (int, float))
                or not math.isfinite(priority)
            ):
                raise ValueError(
                    f"batch entry {index} priority must be finite"
                )
            # A policy-ineligible member makes the entire handoff a fail-open
            # miss, preserving the one-packet/all-registered contract.
            if priority <= 0.0:
                return ()
            key = ExactSpeculationKey.from_invocation(
                invocation,
                session_id=session_id,
                decision_id=decision_id,
                context_token=context_token,
            )
            normalized.append(
                (
                    invocation,
                    session_id,
                    decision_id,
                    float(priority),
                    context_token,
                    key,
                )
            )
        if not normalized:
            return ()

        # Reap the previous epoch at admission time, outside authority
        # confirmation. Claims do only an O(1) exact-key lease check.
        self._reap_expired_for_submit()

        with self._lifecycle_lock:
            if (
                not self._started
                or not self._accepting
                or self._closing
                or self._closed
            ):
                return ()

            with self._registry_lock:
                keys = tuple(values[5] for values in normalized)
                if (
                    len(set(keys)) != len(keys)
                    or any(key in self._available for key in keys)
                ):
                    # Exact lookup is one request id per key.  Reject the
                    # whole packet instead of shadowing a live generation or
                    # orphaning an earlier member of this batch.
                    return ()
                if (
                    self._pull_result_staging
                    and len(self._handles) + len(normalized)
                    > self._pull_registry_capacity
                ):
                    self._increment_transport(
                        "transport_pull_registry_full"
                    )
                    return ()
                handles = tuple(
                    SpeculativeHandle(values[5]) for values in normalized
                )
                first_id = self._next_request_id + 1
                self._next_request_id += len(normalized)
                request_ids = tuple(
                    range(first_id, first_id + len(normalized))
                )
                missing = object()
                previous_available: dict[
                    ExactSpeculationKey, int | object
                ] = {
                    values[5]: self._available.get(values[5], missing)
                    for values in normalized
                }
                submissions: list[_ProcessSubmit] = []
                for request_id, handle, values in zip(
                    request_ids, handles, normalized
                ):
                    (
                        invocation,
                        session_id,
                        decision_id,
                        priority,
                        context_token,
                        key,
                    ) = values
                    self._handles[request_id] = handle
                    self._available[key] = request_id
                    if normalized_lease_until is not None:
                        self._leases[request_id] = normalized_lease_until
                    submissions.append(
                        _ProcessSubmit(
                            request_id=request_id,
                            invocation=invocation,
                            session_id=session_id,
                            decision_id=decision_id,
                            context_token=context_token,
                            priority=priority,
                            start_deadline=normalized_deadline,
                        )
                    )

                if not self._put_nowait(
                    _ProcessSubmitBatch(tuple(submissions))
                ):
                    for request_id in request_ids:
                        self._handles.pop(request_id, None)
                        self._leases.pop(request_id, None)
                    for key, prior_request_id in previous_available.items():
                        if prior_request_id is missing:
                            self._available.pop(key, None)
                        else:
                            assert isinstance(prior_request_id, int)
                            self._available[key] = prior_request_id
                    return ()

        self._increment_transport("transport_submitted", len(handles))
        self._increment_transport("transport_submit_packets")
        if self._pull_result_staging:
            self._set_pull_epoch_sealed(False)
        return handles

    def try_schedule_batch(
        self,
        entries: Iterable[ProcessSubmitEntry],
        *,
        release_at: float,
        start_deadline: float,
    ) -> tuple[SpeculativeHandle, ...]:
        """Preload one future batch without timed parent-side admission."""

        batches = self.try_schedule_batches(
            ((release_at, start_deadline, entries),)
        )
        return batches[0] if batches else ()

    def try_schedule_batches(
        self,
        batches: Iterable[ProcessScheduleEntry],
    ) -> tuple[tuple[SpeculativeHandle, ...], ...]:
        """Atomically preload future batches in one bounded datagram.

        Every ``release_at`` and ``start_deadline`` is an absolute finite
        monotonic timestamp, with ``now < release_at < start_deadline`` at
        handoff. The child retains the candidates in a bounded timer heap and
        touches the ordinary speculative scheduler only at ``release_at``.
        Atomicity covers parent registration plus the single transport packet;
        at release, each candidate is independently admitted to the bounded
        inner scheduler and the child may therefore admit only part of a
        batch. ``scheduled.admission_dropped`` exposes that fail-open path.
        An empty return is a fail-open rejection with no parent registration.
        """

        normalized_batches: list[
            tuple[
                float,
                float,
                float,
                list[
                    tuple[
                        Invocation,
                        str,
                        str,
                        float,
                        str,
                        ExactSpeculationKey,
                    ]
                ],
            ]
        ] = []
        initial_now = self._clock()
        for batch_index, batch in enumerate(batches):
            try:
                release_at, start_deadline, entries = batch
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "scheduled batch must be "
                    "(release_at, start_deadline, entries)"
                ) from exc
            for name, value in (
                ("release_at", release_at),
                ("start_deadline", start_deadline),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"scheduled batch {batch_index} {name} must be finite"
                    )
            normalized_release = float(release_at)
            normalized_deadline = float(start_deadline)
            if normalized_release <= initial_now:
                return ()
            if normalized_release >= normalized_deadline:
                return ()
            lease_until = normalized_deadline + self._claim_grace_s
            if not math.isfinite(lease_until):
                raise ValueError(
                    "start_deadline + claim_grace_s must be finite"
                )

            normalized_entries: list[
                tuple[
                    Invocation,
                    str,
                    str,
                    float,
                    str,
                    ExactSpeculationKey,
                ]
            ] = []
            for entry_index, entry in enumerate(entries):
                try:
                    (
                        invocation,
                        session_id,
                        decision_id,
                        priority,
                        context_token,
                    ) = entry
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "scheduled entry must be (invocation, session_id, "
                        "decision_id, priority, context_token)"
                    ) from exc
                if not isinstance(invocation, Invocation):
                    raise TypeError(
                        f"scheduled batch {batch_index} entry {entry_index} "
                        "invocation must be an Invocation"
                    )
                if (
                    isinstance(priority, bool)
                    or not isinstance(priority, (int, float))
                    or not math.isfinite(priority)
                ):
                    raise ValueError(
                        f"scheduled batch {batch_index} entry {entry_index} "
                        "priority must be finite"
                    )
                if priority <= 0.0:
                    return ()
                key = ExactSpeculationKey.from_invocation(
                    invocation,
                    session_id=session_id,
                    decision_id=decision_id,
                    context_token=context_token,
                )
                normalized_entries.append(
                    (
                        invocation,
                        session_id,
                        decision_id,
                        float(priority),
                        context_token,
                        key,
                    )
                )
            if not normalized_entries:
                return ()
            normalized_batches.append(
                (
                    normalized_release,
                    normalized_deadline,
                    lease_until,
                    normalized_entries,
                )
            )
        if not normalized_batches:
            return ()

        self._reap_expired_for_submit()
        candidate_count = sum(
            len(values[3]) for values in normalized_batches
        )
        with self._lifecycle_lock:
            if (
                not self._started
                or not self._accepting
                or self._closing
                or self._closed
            ):
                return ()
            with self._registry_lock:
                now = self._clock()
                for request_id, scheduled_release in tuple(
                    self._scheduled_releases.items()
                ):
                    if scheduled_release <= now:
                        self._scheduled_releases.pop(request_id, None)
                if any(values[0] <= now for values in normalized_batches):
                    return ()
                if (
                    len(self._scheduled_releases) + candidate_count
                    > self._max_scheduled_pending
                ):
                    self._increment_transport(
                        "transport_schedule_capacity_rejected",
                        candidate_count,
                    )
                    return ()
                if (
                    self._pull_result_staging
                    and len(self._handles) + candidate_count
                    > self._pull_registry_capacity
                ):
                    self._increment_transport(
                        "transport_pull_registry_full"
                    )
                    return ()

                flattened = [
                    entry
                    for _, _, _, entries in normalized_batches
                    for entry in entries
                ]
                flattened_keys = [entry[5] for entry in flattened]
                if (
                    len(set(flattened_keys)) != len(flattened_keys)
                    or any(key in self._available for key in flattened_keys)
                ):
                    # Exact lookup is deliberately O(1), not a generation
                    # queue. Reject the provisional packet rather than let a
                    # later release shadow an earlier/live exact key.
                    return ()
                handles = tuple(
                    SpeculativeHandle(values[5]) for values in flattened
                )
                first_id = self._next_request_id + 1
                self._next_request_id += candidate_count
                request_ids = tuple(
                    range(first_id, first_id + candidate_count)
                )
                missing = object()
                previous_available: dict[
                    ExactSpeculationKey, int | object
                ] = {
                    values[5]: self._available.get(values[5], missing)
                    for values in flattened
                }

                commands: list[_ProcessScheduledBatch] = []
                cursor = 0
                nested_handles: list[tuple[SpeculativeHandle, ...]] = []
                for release, deadline, lease_until, entries in normalized_batches:
                    batch_handles = handles[cursor : cursor + len(entries)]
                    batch_ids = request_ids[cursor : cursor + len(entries)]
                    submissions: list[_ProcessSubmit] = []
                    for request_id, handle, values in zip(
                        batch_ids, batch_handles, entries
                    ):
                        (
                            invocation,
                            session_id,
                            decision_id,
                            priority,
                            context_token,
                            key,
                        ) = values
                        self._handles[request_id] = handle
                        self._available[key] = request_id
                        self._leases[request_id] = lease_until
                        self._scheduled_releases[request_id] = release
                        submissions.append(
                            _ProcessSubmit(
                                request_id=request_id,
                                invocation=invocation,
                                session_id=session_id,
                                decision_id=decision_id,
                                context_token=context_token,
                                priority=priority,
                                start_deadline=deadline,
                            )
                        )
                    commands.append(
                        _ProcessScheduledBatch(
                            release_at=release,
                            start_deadline=deadline,
                            submissions=tuple(submissions),
                        )
                    )
                    nested_handles.append(tuple(batch_handles))
                    cursor += len(entries)

                if not self._put_nowait(
                    _ProcessScheduleBatches(tuple(commands))
                ):
                    for request_id in request_ids:
                        self._handles.pop(request_id, None)
                        self._leases.pop(request_id, None)
                        self._scheduled_releases.pop(request_id, None)
                    for key, prior_request_id in previous_available.items():
                        if prior_request_id is missing:
                            self._available.pop(key, None)
                        else:
                            assert isinstance(prior_request_id, int)
                            self._available[key] = prior_request_id
                    return ()

        self._increment_transport("transport_scheduled", candidate_count)
        self._increment_transport(
            "transport_schedule_batches", len(normalized_batches)
        )
        self._increment_transport("transport_schedule_packets")
        if self._pull_result_staging:
            self._set_pull_epoch_sealed(False)
        return tuple(nested_handles)

    def try_claim(
        self, key: ExactSpeculationKey
    ) -> SpeculativeHandle | None:
        if not isinstance(key, ExactSpeculationKey):
            raise TypeError("key must be an ExactSpeculationKey")
        if self._pull_result_staging:
            return self._try_claim_pulled(key)
        if self._eager_result_staging:
            return self._try_claim_staged(key)
        if not self._registry_lock.acquire(blocking=False):
            self._increment_transport("transport_claim_busy")
            return None
        request_id: int | None = None
        handle: SpeculativeHandle | None = None
        expired_handle: SpeculativeHandle | None = None
        rejected_handle: SpeculativeHandle | None = None
        missed = False
        claimed = False
        try:
            request_id = self._available.get(key)
            handle = (
                self._handles.get(request_id)
                if request_id is not None
                else None
            )
            now = self._clock()
            lease_until = (
                self._leases.get(request_id)
                if request_id is not None
                else None
            )
            scheduled_release = (
                self._scheduled_releases.get(request_id)
                if request_id is not None
                else None
            )
            if handle is None:
                missed = True
            elif scheduled_release is not None and scheduled_release > now:
                # A preload is not execution before its release fence. Keep
                # both exact mapping and capacity reservation so authority can
                # retry the same O(1) lookup at or after release.
                missed = True
            elif lease_until is not None and lease_until <= now:
                self._handles.pop(request_id, None)
                if self._available.get(key) == request_id:
                    self._available.pop(key, None)
                self._leases.pop(request_id, None)
                self._scheduled_releases.pop(request_id, None)
                expired_handle = handle
                missed = True
            elif not handle._try_claim(now):
                missed = True
            elif not self._put_nowait(_ProcessClaim(request_id)):
                self._handles.pop(request_id, None)
                if self._available.get(key) == request_id:
                    self._available.pop(key, None)
                self._leases.pop(request_id, None)
                self._scheduled_releases.pop(request_id, None)
                rejected_handle = handle
            else:
                self._available.pop(key, None)
                self._leases.pop(request_id, None)
                self._scheduled_releases.pop(request_id, None)
                claimed = True
        finally:
            self._registry_lock.release()
        if expired_handle is not None and not expired_handle.future.done():
            expired_handle.future.set_exception(
                SidecarExpired("exact process speculation lease expired")
            )
            self._increment_transport("transport_claim_expired")
        if rejected_handle is not None and not rejected_handle.future.done():
            rejected_handle.future.set_exception(
                SidecarRejected("process claim queue was full")
            )
        if missed:
            self._increment_transport("transport_claim_misses")
            return None
        if not claimed:
            return None
        self._increment_transport("transport_claims")
        self._increment_transport("transport_claim_packets")
        self._ensure_bridge()
        return handle

    def _try_claim_pulled(
        self, key: ExactSpeculationKey
    ) -> SpeculativeHandle | None:
        """Boundedly drain the kernel mailbox, then do an exact local claim.

        This is intentionally lossy. A busy receiver, an event beyond the
        finite drain budget, malformed data, or a not-yet-enqueued terminal is
        a speculative miss; none can delay or cancel authority.
        """

        if not self._registry_lock.acquire(blocking=False):
            self._increment_transport("transport_claim_busy")
            return None
        try:
            target_request_id = self._available.get(key)
        finally:
            self._registry_lock.release()

        # An explicit lifecycle operation may already have started the bridge.
        # In that case it owns recv(), and the claim remains a local map lookup.
        if (
            target_request_id is not None
            and self._bridge is None
            and not self.pull_epoch_sealed
        ):
            self._drain_pull_result_mailbox(target_request_id)
        return self._try_claim_staged(key)

    def prefetch_pull_results(
        self, *, deadline: float | None = None
    ) -> int:
        """Boundedly stage one result epoch, then prohibit timed socket reads.

        This is intended for a pre-authority guard window.  Once it returns,
        exact claims use only the bounded parent-local map; late packets stay
        dormant in the kernel mailbox until a later prefetch or lifecycle
        drain.  Lock contention, a deadline, or malformed data merely reduces
        speculative coverage.
        """

        if not self._pull_result_staging:
            raise SidecarRejected("pull prefetch requires pull_result_staging")
        if deadline is not None and (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise ValueError("deadline must be finite or None")
        self._increment_transport("transport_pull_prefetch_calls")
        packets = self._drain_pull_result_mailbox(
            None,
            deadline=None if deadline is None else float(deadline),
        )
        self._set_pull_epoch_sealed(True)
        self._increment_transport(
            "transport_pull_prefetch_packets", packets
        )
        return packets

    def _drain_pull_result_mailbox(
        self,
        target_request_id: int | None,
        *,
        deadline: float | None = None,
    ) -> int:
        """Consume at most one bounded result epoch without blocking."""

        if not self._event_receive_lock.acquire(blocking=False):
            self._increment_transport("transport_claim_busy")
            if target_request_id is None:
                self._increment_transport("transport_pull_prefetch_busy")
            return 0
        packets = 0
        try:
            # max_pending bounds one live scheduler epoch and therefore the
            # per-claim decode work. Parent retention remains independently
            # bounded by result_capacity. Stop at the desired datagram.
            for _ in range(self._max_pending):
                if deadline is not None and self._clock() >= deadline:
                    break
                try:
                    receive_bytes = min(
                        self._max_packet_bytes,
                        self._max_pull_staged_packet_bytes + 1,
                    )
                    encoded = self._event_parent.recv(
                        receive_bytes,
                        socket.MSG_DONTWAIT,
                    )
                    if not encoded:
                        break
                    if len(encoded) > self._max_pull_staged_packet_bytes:
                        self._increment_transport(
                            "transport_pull_decode_dropped"
                        )
                        continue
                    event = pickle.loads(encoded)
                except BlockingIOError:
                    break
                except Exception:
                    self._increment_transport(
                        "transport_pull_decode_dropped"
                    )
                    break
                self._increment_transport("transport_pull_packets")
                packets += 1
                if not isinstance(event, _ProcessEvent):
                    self._increment_transport(
                        "transport_pull_decode_dropped"
                    )
                    continue
                if event.kind == "staged":
                    try:
                        self._handle_staged_event(
                            event,
                            preferred_request_id=target_request_id,
                            nonblocking=True,
                        )
                    except Exception:
                        self._increment_transport(
                            "transport_pull_decode_dropped"
                        )
                        break
                    if event.request_id == target_request_id:
                        break
                    continue
                # Pull mode's timed path accepts staged outcomes only.  A
                # CLOSED or legacy terminal event may require blocking state
                # cleanup, so consume/drop it here and leave cleanup to the
                # post-authority lifecycle bridge.
                self._increment_transport("transport_pull_decode_dropped")
                break
        finally:
            self._event_receive_lock.release()
        return packets

    def _try_claim_staged(
        self, key: ExactSpeculationKey
    ) -> SpeculativeHandle | None:
        """Consume only a parent-local terminal result, never waiting or IPC."""

        if not self._registry_lock.acquire(blocking=False):
            self._increment_transport("transport_claim_busy")
            return None

        handle: SpeculativeHandle | None = None
        staged: _ParentStagedOutcome | None = None
        terminal_error: BaseException | None = None
        completion_owned = False
        not_ready = False
        claimed = False
        try:
            request_id = self._available.get(key)
            handle = (
                self._handles.get(request_id)
                if request_id is not None
                else None
            )
            now = self._clock()
            lease_until = (
                self._leases.get(request_id)
                if request_id is not None
                else None
            )
            staged = (
                self._staged_results.get(request_id)
                if request_id is not None
                else None
            )
            if handle is not None:
                try:
                    completion_owned = (
                        handle.future.set_running_or_notify_cancel()
                    )
                except RuntimeError:
                    # A caller is allowed to observe/cancel the public future.
                    # Unexpected external completion must also remain a
                    # fail-open miss rather than escape onto authority.
                    completion_owned = False

            if handle is None:
                pass
            elif not completion_owned:
                terminal_error = SidecarRejected(
                    "staged result observer was cancelled or terminal"
                )
            elif lease_until is not None and lease_until <= now:
                terminal_error = SidecarExpired(
                    "exact process speculation lease expired"
                )
            elif staged is None:
                not_ready = True
                terminal_error = SidecarRejected(
                    "staged result was not parent-local at confirmation"
                )
            elif staged.valid_until <= now:
                terminal_error = SidecarExpired(
                    "staged process speculation result TTL expired"
                )
            elif staged.error is not None:
                terminal_error = staged.error
            elif handle._try_claim(now):
                claimed = True
            else:
                terminal_error = SidecarRejected(
                    "staged process speculation is no longer claimable"
                )

            if handle is not None and (claimed or terminal_error is not None):
                assert request_id is not None
                self._handles.pop(request_id, None)
                if self._available.get(key) == request_id:
                    self._available.pop(key, None)
                self._leases.pop(request_id, None)
                self._scheduled_releases.pop(request_id, None)
                self._staged_results.pop(request_id, None)
        finally:
            self._registry_lock.release()

        if handle is None:
            self._increment_transport("transport_claim_misses")
            return None
        if terminal_error is not None:
            if completion_owned:
                try:
                    handle.future.set_exception(terminal_error)
                except concurrent.futures.InvalidStateError:
                    pass
            self._increment_transport("transport_claim_misses")
            if not_ready:
                self._increment_transport(
                    "transport_pull_not_ready"
                    if self._pull_result_staging
                    else "transport_eager_not_ready"
                )
            elif isinstance(terminal_error, SidecarExpired):
                self._increment_transport("transport_claim_expired")
            return None
        if not claimed or staged is None:
            self._increment_transport("transport_claim_misses")
            return None

        if staged.started_at is not None:
            handle._mark_started(staged.started_at)
        handle._mark_finished(
            staged.finished_at,
            valid_until=staged.valid_until,
        )
        try:
            handle.future.set_result(staged.result)
        except concurrent.futures.InvalidStateError:
            self._increment_transport("transport_claim_misses")
            return None
        self._increment_transport("transport_claims")
        self._increment_transport(
            "transport_pull_hits"
            if self._pull_result_staging
            else "transport_eager_hits"
        )
        self._increment_transport("transport_results")
        return handle

    def try_tombstone(
        self,
        *,
        session_id: str,
        decision_id: str | None = None,
    ) -> bool:
        return self.try_tombstone_batch(((session_id, decision_id),))

    def try_tombstone_batch(
        self,
        tombstones: Iterable[tuple[str, str | None]],
    ) -> bool:
        """Retire several decisions locally after one atomic child handoff."""

        normalized: list[tuple[str, str | None]] = []
        for index, tombstone in enumerate(tombstones):
            try:
                session_id, decision_id = tombstone
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "tombstone batch entry must be (session_id, decision_id)"
                ) from exc
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(
                    f"tombstone entry {index} session_id must be non-empty"
                )
            if decision_id is not None and (
                not isinstance(decision_id, str) or not decision_id
            ):
                raise ValueError(
                    f"tombstone entry {index} decision_id must be "
                    "a non-empty string or None"
                )
            normalized.append((session_id, decision_id))
        # Preserve caller order while avoiding redundant work and packet bytes.
        unique_targets = tuple(dict.fromkeys(normalized))
        whole_sessions = {
            session_id
            for session_id, decision_id in unique_targets
            if decision_id is None
        }
        targets = tuple(
            (session_id, decision_id)
            for session_id, decision_id in unique_targets
            if decision_id is None or session_id not in whole_sessions
        )
        if not targets:
            return True

        terminal: list[SpeculativeHandle] = []
        with self._registry_lock:
            command = _ProcessTombstoneBatch(
                tuple(
                    _ProcessTombstone(session_id, decision_id)
                    for session_id, decision_id in targets
                )
            )
            if not self._put_nowait(command):
                return False
            exact_decisions = set(targets)
            for request_id, handle in tuple(self._handles.items()):
                key = handle.key
                if (
                    key.session_id not in whole_sessions
                    and (key.session_id, key.decision_id)
                    not in exact_decisions
                ):
                    continue
                if handle.claimed:
                    continue
                self._handles.pop(request_id, None)
                self._leases.pop(request_id, None)
                self._scheduled_releases.pop(request_id, None)
                self._staged_results.pop(request_id, None)
                if self._available.get(key) == request_id:
                    self._available.pop(key, None)
                terminal.append(handle)
        for handle in terminal:
            if not handle.future.done():
                handle.future.set_exception(
                    SidecarTombstoned(
                        "process-side speculation tombstoned; child drains lazily"
                    )
                )
        self._increment_transport("transport_tombstones", len(targets))
        self._increment_transport("transport_tombstone_packets")
        return True

    def _bridge_events(self) -> None:
        requested_affinity = self._cpu_affinity
        actual_affinity: tuple[int, ...] | None = None
        affinity_error: str | None = None
        try:
            if requested_affinity is not None:
                # On Linux, pid 0 addresses only this calling thread. Keeping
                # the call here avoids changing the authority thread's mask.
                os.sched_setaffinity(0, set(requested_affinity))
            actual_affinity = tuple(sorted(os.sched_getaffinity(0)))
            if (
                requested_affinity is not None
                and actual_affinity != requested_affinity
            ):
                affinity_error = (
                    "process result bridge CPU affinity mismatch: "
                    f"requested={list(requested_affinity)}, "
                    f"actual={list(actual_affinity)}"
                )
        except BaseException as exc:
            affinity_error = (
                "process result bridge CPU affinity certificate failed: "
                f"{exc!r}"
            )
        with self._bridge_lock:
            self._bridge_actual_cpu_affinity = actual_affinity
            self._bridge_affinity_error = affinity_error
        if affinity_error is not None:
            self._abort_result_bridge(SidecarClosed(affinity_error))
            self._bridge_affinity_ready.set()
            return
        self._bridge_affinity_ready.set()

        process = self._process
        if process is None:
            self._fail_all(SidecarClosed("process sidecar was not started"))
            self._stopped.set()
            self._close_parent_transport()
            return
        # Wait in the kernel on the response datagram socket and child
        # sentinel together. There is no Queue feeder thread and no periodic
        # Python/GIL wakeup while the sidecar is quiet.
        event_reader = self._event_parent
        child_sentinel = process.sentinel
        while not self._stopped.is_set():
            try:
                ready = wait_for_connections(
                    [event_reader, child_sentinel]
                )
            except (EOFError, OSError):
                self._fail_all(SidecarClosed("process result bridge closed"))
                self._stopped.set()
                self._close_parent_transport()
                break
            if event_reader not in ready:
                # The child exited and no complete event is readable. A final
                # CLOSED event, when present, makes the reader ready as well
                # and is always consumed before the sentinel branch.
                self._fail_all(
                    SidecarClosed("process sidecar exited without close event")
                )
                self._stopped.set()
                self._close_parent_transport()
                break
            try:
                # Pull claims and the lifecycle bridge share one datagram
                # reader. Recheck readiness under the lock so a pull that won
                # the race cannot leave this thread blocked on authority's CPU.
                with self._event_receive_lock:
                    encoded = event_reader.recv(
                        self._max_packet_bytes,
                        socket.MSG_DONTWAIT,
                    )
                if not encoded:
                    raise EOFError("process response socket closed")
                event = pickle.loads(encoded)
            except BlockingIOError:
                continue
            except (EOFError, OSError, pickle.PickleError):
                self._fail_all(SidecarClosed("process result bridge closed"))
                self._stopped.set()
                self._close_parent_transport()
                break
            if not isinstance(event, _ProcessEvent):
                continue
            if event.kind == "snapshot" and event.request_id is not None:
                with self._registry_lock:
                    waiter = self._snapshot_waiters.pop(event.request_id, None)
                if waiter is not None and not waiter.done():
                    waiter.set_result(self._decorate_snapshot(event.payload))
                continue
            if event.kind == "closed":
                self._last_snapshot = self._decorate_snapshot(event.payload)
                self._fail_all(SidecarClosed("process sidecar closed"))
                with self._lifecycle_lock:
                    self._closed = True
                self._stopped.set()
                self._close_parent_transport()
                break
            if event.request_id is None:
                continue
            if event.kind == "staged":
                self._handle_staged_event(event)
                continue
            self._handle_terminal_event(event)

    def _handle_staged_event(
        self,
        event: _ProcessEvent,
        *,
        preferred_request_id: int | None = None,
        nonblocking: bool = False,
    ) -> None:
        """Decode one child terminal into bounded private parent staging."""

        request_id = event.request_id
        outcome = event.payload
        if request_id is None or not isinstance(
            outcome, _ProcessStagedOutcome
        ):
            self._increment_transport("transport_stage_dropped")
            return
        finished_at = outcome.finished_at
        valid_until = outcome.valid_until
        started_at = outcome.started_at
        result_payload = outcome.result_payload
        error_payload = outcome.error_payload
        finite_finished = (
            isinstance(finished_at, (int, float))
            and not isinstance(finished_at, bool)
            and math.isfinite(float(finished_at))
        )
        finite_valid = (
            isinstance(valid_until, (int, float))
            and not isinstance(valid_until, bool)
            and math.isfinite(float(valid_until))
        )
        finite_started = (
            started_at is None
            or (
                isinstance(started_at, (int, float))
                and not isinstance(started_at, bool)
                and math.isfinite(float(started_at))
            )
        )
        valid_result = (
            isinstance(result_payload, bytes)
            and len(result_payload) <= self._max_staged_result_bytes
            and error_payload is None
        )
        valid_error = (
            result_payload is None
            and isinstance(error_payload, tuple)
            and len(error_payload) == 2
            and all(isinstance(value, str) for value in error_payload)
        )
        if (
            not finite_finished
            or not finite_valid
            or not finite_started
            or float(valid_until) <= float(finished_at)
            or not (valid_result or valid_error)
        ):
            self._increment_transport("transport_stage_dropped")
            return

        # Reject stale/non-exact/capacity-overflow events before decoding a
        # potentially large result.  Confirmation may concurrently retire the
        # handle; the second check below linearizes actual publication.
        expired_handles: list[SpeculativeHandle] = []
        preferred_evicted = False
        pull_claim_path = preferred_request_id is not None or nonblocking
        acquired = self._registry_lock.acquire(
            blocking=not pull_claim_path
        )
        if not acquired:
            self._increment_transport("transport_stage_dropped")
            return
        try:
            now = self._clock()
            if (
                request_id not in self._staged_results
                and len(self._staged_results) >= self._result_capacity
            ):
                for stale_id, stale in tuple(self._staged_results.items()):
                    if stale.valid_until <= now:
                        self._staged_results.pop(stale_id, None)
                        stale_handle = self._handles.pop(stale_id, None)
                        self._leases.pop(stale_id, None)
                        self._scheduled_releases.pop(stale_id, None)
                        if stale_handle is not None:
                            if (
                                self._available.get(stale_handle.key)
                                == stale_id
                            ):
                                self._available.pop(stale_handle.key, None)
                            expired_handles.append(stale_handle)
                # In pull mode, earlier unrelated datagrams must not prevent
                # the exact datagram that caused this bounded drain from being
                # retained. Drop one private payload only; its public handle
                # remains pending and will fail open as not-ready if claimed.
                if (
                    request_id == preferred_request_id
                    and request_id not in self._staged_results
                    and len(self._staged_results) >= self._result_capacity
                ):
                    victim_id = next(
                        (
                            staged_id
                            for staged_id in self._staged_results
                            if staged_id != request_id
                        ),
                        None,
                    )
                    if victim_id is not None:
                        self._staged_results.pop(victim_id, None)
                        preferred_evicted = True
            handle = self._handles.get(request_id)
            lease_until = self._leases.get(request_id)
            eligible = (
                handle is not None
                and self._available.get(handle.key) == request_id
                and (lease_until is None or lease_until > now)
                and outcome.valid_until > now
                and (
                    request_id in self._staged_results
                    or len(self._staged_results) < self._result_capacity
                )
            )
        finally:
            self._registry_lock.release()
        for expired_handle in expired_handles:
            try:
                completion_owned = (
                    expired_handle.future.set_running_or_notify_cancel()
                )
            except RuntimeError:
                completion_owned = False
            if completion_owned:
                try:
                    expired_handle.future.set_exception(
                        SidecarExpired(
                            "staged process speculation result TTL expired"
                        )
                    )
                except concurrent.futures.InvalidStateError:
                    pass
        if expired_handles:
            self._increment_transport(
                "transport_lease_reaped", len(expired_handles)
            )
        if preferred_evicted:
            self._increment_transport("transport_stage_dropped")
        if not eligible:
            self._increment_transport("transport_stage_dropped")
            return

        if outcome.error_payload is not None:
            result = None
            error = _remote_error(outcome.error_payload)
        else:
            try:
                result = pickle.loads(outcome.result_payload)
            except BaseException:
                self._increment_transport("transport_stage_dropped")
                return
            error = None
        staged = _ParentStagedOutcome(
            result=result,
            error=error,
            started_at=outcome.started_at,
            finished_at=outcome.finished_at,
            valid_until=outcome.valid_until,
        )

        stored = False
        acquired = self._registry_lock.acquire(
            blocking=not pull_claim_path
        )
        if not acquired:
            self._increment_transport("transport_stage_dropped")
            return
        try:
            handle = self._handles.get(request_id)
            lease_until = self._leases.get(request_id)
            live_exact = (
                handle is not None
                and self._available.get(handle.key) == request_id
            )
            now = self._clock()
            if (
                live_exact
                and (lease_until is None or lease_until > now)
                and staged.valid_until > now
                and (
                    request_id in self._staged_results
                    or len(self._staged_results) < self._result_capacity
                )
            ):
                self._staged_results[request_id] = staged
                stored = True
        finally:
            self._registry_lock.release()
        if not stored:
            self._increment_transport("transport_stage_dropped")
        elif error is None:
            self._increment_transport("transport_staged_results")
        else:
            self._increment_transport("transport_staged_failures")

    def _handle_terminal_event(self, event: _ProcessEvent) -> None:
        request_id = event.request_id
        assert request_id is not None
        with self._registry_lock:
            handle = self._handles.pop(request_id, None)
            self._leases.pop(request_id, None)
            self._scheduled_releases.pop(request_id, None)
            self._staged_results.pop(request_id, None)
            if handle is not None and self._available.get(handle.key) == request_id:
                self._available.pop(handle.key, None)
        if handle is None or handle.future.done():
            return
        if event.kind == "result":
            try:
                result = pickle.loads(event.payload)
            except BaseException as exc:
                handle.future.set_exception(
                    SpeculativeSidecarError(
                        f"could not decode child result: {exc}"
                    )
                )
            else:
                now = self._clock()
                if handle.started_at is None:
                    handle._mark_started(now)
                handle._mark_finished(
                    now, valid_until=now + self._result_ttl_s
                )
                handle.future.set_result(result)
                self._increment_transport("transport_results")
        else:
            handle.future.set_exception(_remote_error(event.payload))
            if event.kind == "claim_miss":
                self._increment_transport("transport_claim_misses")
            else:
                self._increment_transport("transport_terminal")

    def _fail_all(self, error: BaseException) -> None:
        with self._registry_lock:
            handles = list(self._handles.values())
            self._handles.clear()
            self._available.clear()
            self._leases.clear()
            self._scheduled_releases.clear()
            self._staged_results.clear()
            waiters = list(self._snapshot_waiters.values())
            self._snapshot_waiters.clear()
        for handle in handles:
            if not handle.future.done():
                handle.future.set_exception(error)
        for waiter in waiters:
            if not waiter.done():
                waiter.set_exception(error)

    def _abort_result_bridge(self, error: BaseException) -> None:
        """Stop parent admission and make bridge startup failure reapable."""

        with self._lifecycle_lock:
            self._accepting = False
            self._closing = True
        self._fail_all(error)
        # Command EOF asks the child to drain even though no bridge remains to
        # send a Close packet. Closing the event peer also prevents it from
        # blocking while publishing a final snapshot that nobody can read.
        self._close_parent_transport()
        self._stopped.set()

    def _close_parent_transport(self) -> None:
        for endpoint in (self._command_parent, self._event_parent):
            try:
                endpoint.close()
            except OSError:
                pass

    def _decorate_snapshot(self, raw: Any) -> dict[str, Any]:
        snapshot = dict(raw) if isinstance(raw, dict) else self._empty_snapshot()
        capacity = dict(snapshot.get("capacity", {}))
        capacity.update(
            {
                "result_capacity": self._result_capacity,
                "pull_result_staging": self._pull_result_staging,
                "pull_registry_capacity": self._pull_registry_capacity,
                "max_staged_result_bytes": (
                    self._max_staged_result_bytes
                ),
                "max_pull_staged_packet_bytes": (
                    self._max_pull_staged_packet_bytes
                ),
            }
        )
        snapshot["capacity"] = capacity
        process = self._process
        snapshot["process_alive"] = bool(process and process.is_alive())
        snapshot["process_pid"] = (
            snapshot.get("process_pid")
            if snapshot.get("process_pid") is not None
            else self.pid
        )
        snapshot["bridge_started"] = self._bridge is not None
        with self._bridge_lock:
            snapshot["requested_bridge_cpu_affinity"] = (
                None
                if self._cpu_affinity is None
                else list(self._cpu_affinity)
            )
            snapshot["actual_bridge_cpu_affinity"] = (
                None
                if self._bridge_actual_cpu_affinity is None
                else list(self._bridge_actual_cpu_affinity)
            )
            snapshot["bridge_affinity_ready"] = (
                self._bridge_affinity_ready.is_set()
            )
            snapshot["bridge_affinity_error"] = self._bridge_affinity_error
        with self._stats_lock:
            transport = dict(self._transport_stats)
        snapshot["transport"] = transport
        with self._registry_lock:
            staged_ready = len(self._staged_results)
        snapshot["parent_staging"] = {
            "enabled": (
                self._eager_result_staging or self._pull_result_staging
            ),
            "mode": (
                "eager"
                if self._eager_result_staging
                else "pull"
                if self._pull_result_staging
                else "disabled"
            ),
            "ready": staged_ready,
            "capacity": self._result_capacity,
            "max_result_bytes": self._max_staged_result_bytes,
            "pull_epoch_sealed": self.pull_epoch_sealed,
        }
        return snapshot

    def snapshot(self, *, timeout: float = 2.0) -> dict[str, Any]:
        """Request a child snapshot; diagnostic and intentionally blocking."""

        deadline = time.monotonic() + max(0.0, timeout)
        with self._lifecycle_lock:
            if not self._started or self._closed:
                return self._decorate_snapshot(self._last_snapshot)
            stopped = self._stopped.is_set()
        if stopped:
            with self._bridge_lock:
                affinity_failed = self._bridge_affinity_error is not None
            if affinity_failed:
                return self._decorate_snapshot(self._last_snapshot)
            raise SidecarClosed("process result bridge is stopped")
        if not self._ensure_bridge():
            raise SidecarClosed("process result bridge could not start")
        if not self._bridge_affinity_ready.wait(
            max(0.0, deadline - time.monotonic())
        ):
            raise TimeoutError("process result bridge affinity setup timed out")
        with self._lifecycle_lock:
            if self._closed:
                return self._decorate_snapshot(self._last_snapshot)
            stopped = self._stopped.is_set()
        if stopped:
            with self._bridge_lock:
                affinity_failed = self._bridge_affinity_error is not None
            if affinity_failed:
                return self._decorate_snapshot(self._last_snapshot)
            raise SidecarClosed("process result bridge stopped during snapshot")
        request_id = self._next_id()
        waiter: concurrent.futures.Future[dict[str, Any]] = (
            concurrent.futures.Future()
        )
        with self._registry_lock:
            self._snapshot_waiters[request_id] = waiter
        try:
            remaining = max(0.0, deadline - time.monotonic())
            self._send_with_timeout(_ProcessSnapshot(request_id), remaining / 2)
        except (
            TimeoutError,
            BrokenPipeError,
            EOFError,
            OSError,
            ValueError,
        ) as exc:
            with self._registry_lock:
                self._snapshot_waiters.pop(request_id, None)
            raise TimeoutError("could not enqueue process snapshot") from exc
        try:
            return waiter.result(
                timeout=max(0.0, deadline - time.monotonic())
            )
        finally:
            with self._registry_lock:
                self._snapshot_waiters.pop(request_id, None)

    def close(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        """Request child-local lazy drain; never call from authority hot path."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._accepting = False
            self._closing = True
            if not self._started:
                self._closed = True
                self._stopped.set()
                for endpoint in (
                    self._command_parent,
                    self._command_child,
                    self._event_parent,
                    self._event_child,
                ):
                    try:
                        endpoint.close()
                    except OSError:
                        pass
                return
            send_close = not self._close_sent
            self._close_sent = True
        bridge_ready = self._ensure_bridge()
        if send_close and bridge_ready and not self._stopped.is_set():
            try:
                self._send_with_timeout(
                    _ProcessClose(max(0.1, timeout)), max(0.1, timeout / 2)
                )
            except (
                TimeoutError,
                BrokenPipeError,
                EOFError,
                OSError,
                ValueError,
            ) as exc:
                # A lazily detected child crash is already a completed drain,
                # not a reason to make lifecycle cleanup fail.
                self._stopped.wait(min(0.1, max(0.0, timeout)))
                if not self._stopped.is_set():
                    raise TimeoutError("could not enqueue process close") from exc
        if not wait:
            return
        started = time.monotonic()
        if not self._stopped.wait(timeout):
            raise TimeoutError(
                "process sidecar still has non-preemptible work draining"
            )
        remaining = max(0.0, timeout - (time.monotonic() - started))
        process = self._process
        if process is not None:
            process.join(remaining)
            if process.is_alive():
                raise TimeoutError("process sidecar did not exit after drain")
        bridge = self._bridge
        if bridge is not None and bridge is not threading.current_thread():
            bridge.join(max(0.0, remaining))
