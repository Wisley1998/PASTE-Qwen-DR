"""A fork-isolated, demand-only authority execution lane.

The lane keeps the unchanged :class:`~paste_repro.live_broker.LiveToolBroker`
and its asyncio loop in a child process.  The parent only serializes requests
onto an ordered pipe and has one result-bridge thread.  This separates the
authority scheduler, executor callbacks, and Python GIL from a speculative
control loop in the parent process.

This is deliberately a small experimental primitive, rather than a general
RPC framework.  Its safety contract is narrow and explicit:

* every successful ``submit``/``submit_batch`` is ordered before a later
  ``aclose`` command;
* parent-side future cancellation never cancels or suppresses child work;
* ``aclose`` drains every accepted authority request before closing the
  broker; and
* child CPU affinity is applied and read back before setup is certified.

``start`` forks before it starts the parent result bridge.  Callers that also
own another fork-backed component must fork all such children before starting
any parent helper threads.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from collections.abc import Iterable
from dataclasses import dataclass
import math
import multiprocessing
from multiprocessing.connection import Connection
import os
import threading
import time
import traceback
from typing import Any

from .invocation import Invocation
from .live_broker import LiveAuthoritativeResult, LiveToolBroker


class AuthorityProcessLaneError(RuntimeError):
    """Base class for authority-process lifecycle and transport failures."""


class RemoteAuthorityError(AuthorityProcessLaneError):
    """An authority request failed in the child process."""

    def __init__(self, payload: "_ErrorPayload") -> None:
        self.remote_type = payload.type_name
        self.remote_message = payload.message
        self.remote_repr = payload.representation
        self.remote_traceback = payload.traceback_text
        detail = payload.message or payload.representation
        super().__init__(f"remote {payload.type_name}: {detail}")


@dataclass(frozen=True)
class AuthorityCompletion:
    """Picklable completion payload compatible with the replay runner."""

    result: LiveAuthoritativeResult
    scheduled_at: float
    first_run_at: float
    terminal_at: float
    observed_at: float


class _NonCancellingFuture(ConcurrentFuture[AuthorityCompletion]):
    """A proxy whose cancellation cannot cross the authority boundary."""

    def cancel(self) -> bool:
        # Authority is a correctness path, not an optimization.  A cancelled
        # observer must not make its sole physical request unobservable.
        return False


@dataclass(frozen=True)
class _ErrorPayload:
    type_name: str
    message: str
    representation: str
    traceback_text: str


@dataclass(frozen=True)
class _SubmitEntry:
    request_id: int
    invocation: Invocation
    session_id: str
    scheduled_at: float


@dataclass(frozen=True)
class _SubmitBatchCommand:
    entries: tuple[_SubmitEntry, ...]


@dataclass(frozen=True)
class _BarrierCommand:
    request_id: int


@dataclass(frozen=True)
class _CloseCommand:
    request_id: int


@dataclass(frozen=True)
class _ReadyEvent:
    process_pid: int
    requested_cpu_affinity: tuple[int, ...]
    actual_cpu_affinity: tuple[int, ...]
    error: _ErrorPayload | None = None


@dataclass(frozen=True)
class _ResultEvent:
    request_id: int
    completion: AuthorityCompletion | None = None
    error: _ErrorPayload | None = None


@dataclass(frozen=True)
class _ResultBatchEvent:
    events: tuple[_ResultEvent, ...]


@dataclass(frozen=True)
class _BarrierEvent:
    request_id: int
    error: _ErrorPayload | None = None


@dataclass(frozen=True)
class _ClosedEvent:
    request_id: int
    snapshot: dict[str, Any]
    error: _ErrorPayload | None = None


def _error_payload(exc: BaseException) -> _ErrorPayload:
    return _ErrorPayload(
        type_name=f"{type(exc).__module__}.{type(exc).__qualname__}",
        message=str(exc),
        representation=repr(exc),
        traceback_text="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    )


def _send_event(connection: Connection, event: Any) -> bool:
    """Send one child event, returning false only when the parent is gone."""

    try:
        connection.send(event)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _authority_process_worker(
    executor: Any,
    workers: int,
    visit_capacity: int,
    requested_cpu_affinity: tuple[int, ...],
    batch_results: bool,
    command_reader: Connection,
    event_writer: Connection,
    command_writer_parent_copy: Connection,
    event_reader_parent_copy: Connection,
) -> None:
    """Child entry point; all broker state and asyncio callbacks live here."""

    # Fork inherits both ends.  Closing the parent-only copies is required for
    # reliable EOF detection when either process disappears.
    command_writer_parent_copy.close()
    event_reader_parent_copy.close()

    loop: asyncio.AbstractEventLoop | None = None
    broker: LiveToolBroker | None = None
    request_tasks: dict[int, asyncio.Task[None]] = {}
    pending_result_events: list[_ResultEvent] = []
    result_flush_scheduled = False
    close_started = False
    ready_emitted = False
    actual_cpu_affinity: tuple[int, ...] = ()

    try:
        os.sched_setaffinity(0, set(requested_cpu_affinity))
        actual_cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
        if actual_cpu_affinity != requested_cpu_affinity:
            raise RuntimeError(
                "authority process affinity mismatch: "
                f"requested={list(requested_cpu_affinity)}, "
                f"actual={list(actual_cpu_affinity)}"
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        broker = LiveToolBroker(
            executor,
            max_workers=workers,
            max_speculative_workers=0,
            max_speculative_pending=1,
            ttl_s=1.0,
            tool_capacities={"visit": visit_capacity},
        )
        _send_event(
            event_writer,
            _ReadyEvent(
                process_pid=os.getpid(),
                requested_cpu_affinity=requested_cpu_affinity,
                actual_cpu_affinity=actual_cpu_affinity,
            ),
        )
        ready_emitted = True

        def flush_result_events() -> None:
            """Send completions produced in one loop turn as one IPC packet."""

            nonlocal result_flush_scheduled
            result_flush_scheduled = False
            if not pending_result_events:
                return
            events = tuple(pending_result_events)
            pending_result_events.clear()
            batch = _ResultBatchEvent(events=events)
            try:
                _send_event(event_writer, batch)
                return
            except BaseException:
                # Connection.send serializes the complete object before it
                # writes to the pipe.  A bad result therefore cannot have
                # partially emitted this batch.  Retry entries separately so
                # one unpicklable result cannot hide unrelated authority
                # completions, converting only that request to a remote error.
                pass

            for event in events:
                try:
                    delivered = _send_event(
                        event_writer,
                        _ResultBatchEvent(events=(event,)),
                    )
                except BaseException as exc:
                    delivered = _send_event(
                        event_writer,
                        _ResultBatchEvent(
                            events=(
                                _ResultEvent(
                                    request_id=event.request_id,
                                    error=_error_payload(exc),
                                ),
                            )
                        ),
                    )
                if not delivered:
                    return

        def send_individual_result_event(event: _ResultEvent) -> None:
            """Send one result, isolating any serialization failure."""

            try:
                _send_event(event_writer, event)
            except BaseException as exc:
                # The authority call already committed.  Preserve that fact
                # while failing only this observer if its result cannot cross
                # the process boundary.
                _send_event(
                    event_writer,
                    _ResultEvent(
                        request_id=event.request_id,
                        error=_error_payload(exc),
                    ),
                )

        def queue_result_event(event: _ResultEvent) -> None:
            nonlocal result_flush_scheduled
            if not batch_results:
                # Individual packets deliberately smooth parent bridge/Future
                # wakeups for very light tools.  Batching is an explicit
                # experiment because one coherent completion burst can delay
                # the control loop even when it reduces IPC packet count.
                send_individual_result_event(event)
                return
            pending_result_events.append(event)
            if result_flush_scheduled:
                return
            result_flush_scheduled = True
            assert loop is not None
            # call_soon places the flush after every callback already ready in
            # this event-loop turn, naturally coalescing simultaneous results.
            loop.call_soon(flush_result_events)

        async def run_authority(command: _SubmitEntry) -> None:
            first_run_at = time.perf_counter()
            try:
                assert broker is not None
                result = await broker.authoritative(
                    command.invocation,
                    session_id=command.session_id,
                    speculation_eligible=False,
                )
                terminal_at = time.perf_counter()
                completion = AuthorityCompletion(
                    result=result,
                    scheduled_at=command.scheduled_at,
                    first_run_at=first_run_at,
                    terminal_at=terminal_at,
                    observed_at=terminal_at,
                )
                queue_result_event(
                    _ResultEvent(
                        request_id=command.request_id,
                        completion=completion,
                    )
                )
            except BaseException as exc:
                queue_result_event(
                    _ResultEvent(
                        request_id=command.request_id,
                        error=_error_payload(exc),
                    ),
                )
            finally:
                request_tasks.pop(command.request_id, None)

        async def acknowledge_barrier(command: _BarrierCommand) -> None:
            # Submit tasks were created in FIFO command order.  Yielding once
            # gives those tasks the same first scheduling turn guaranteed by
            # DedicatedAuthorityLane.barrier().
            await asyncio.sleep(0)
            _send_event(
                event_writer,
                _BarrierEvent(request_id=command.request_id),
            )

        async def drain_and_close(command: _CloseCommand) -> None:
            close_error: _ErrorPayload | None = None
            snapshot: dict[str, Any]
            try:
                # No reader remains and Close is FIFO behind every accepted
                # Submit, so this finite set is the complete drain obligation.
                tasks = tuple(request_tasks.values())
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                # A close response must never overtake result callbacks that
                # were queued by the just-drained authority tasks.
                flush_result_events()
                assert broker is not None
                await broker.close()
                snapshot = {
                    "snapshot": broker.snapshot(),
                    "stats": broker.stats.to_dict(),
                    "authoritative_state_count": len(
                        broker.authoritative_state
                    ),
                    "requested_cpu_affinity": list(
                        requested_cpu_affinity
                    ),
                    "actual_cpu_affinity": list(actual_cpu_affinity),
                    "process_pid": os.getpid(),
                }
            except BaseException as exc:
                close_error = _error_payload(exc)
                snapshot = {
                    "snapshot": {"jobs": []},
                    "stats": {},
                    "authoritative_state_count": 0,
                    "requested_cpu_affinity": list(
                        requested_cpu_affinity
                    ),
                    "actual_cpu_affinity": list(actual_cpu_affinity),
                    "process_pid": os.getpid(),
                }
            _send_event(
                event_writer,
                _ClosedEvent(
                    request_id=command.request_id,
                    snapshot=snapshot,
                    error=close_error,
                ),
            )
            assert loop is not None
            loop.call_soon(loop.stop)

        def begin_close(request_id: int) -> None:
            nonlocal close_started
            if close_started:
                return
            close_started = True
            assert loop is not None
            loop.remove_reader(command_reader.fileno())
            loop.create_task(drain_and_close(_CloseCommand(request_id)))

        def receive_one_command() -> None:
            # Read at most one command per callback.  A large authority burst
            # therefore cannot starve the broker tasks that the commands just
            # made runnable.
            try:
                command = command_reader.recv()
            except (EOFError, OSError):
                # Parent death cannot revoke already accepted authority.  Drain
                # it before exiting; no close response is expected to arrive.
                begin_close(-1)
                return
            except BaseException as exc:
                _send_event(
                    event_writer,
                    _BarrierEvent(request_id=-1, error=_error_payload(exc)),
                )
                begin_close(-1)
                return

            if isinstance(command, _SubmitBatchCommand):
                # Creating the complete batch before returning to the loop is
                # the only batching semantic: broker ordering and concurrency
                # remain unchanged.
                for entry in command.entries:
                    task = loop.create_task(run_authority(entry))
                    request_tasks[entry.request_id] = task
            elif isinstance(command, _BarrierCommand):
                loop.create_task(acknowledge_barrier(command))
            elif isinstance(command, _CloseCommand):
                begin_close(command.request_id)
            else:
                error = TypeError(
                    f"unknown authority command: {type(command).__name__}"
                )
                _send_event(
                    event_writer,
                    _BarrierEvent(request_id=-1, error=_error_payload(error)),
                )
                begin_close(-1)

        loop.add_reader(command_reader.fileno(), receive_one_command)
        loop.run_forever()
    except BaseException as exc:
        if not ready_emitted:
            _send_event(
                event_writer,
                _ReadyEvent(
                    process_pid=os.getpid(),
                    requested_cpu_affinity=requested_cpu_affinity,
                    actual_cpu_affinity=actual_cpu_affinity,
                    error=_error_payload(exc),
                ),
            )
    finally:
        if loop is not None:
            # Normal close has already drained the request tasks.  This path is
            # a last-resort drain for loop failures; it still never cancels
            # authority work as a teardown shortcut.
            remaining = tuple(
                task for task in request_tasks.values() if not task.done()
            )
            if remaining:
                try:
                    loop.run_until_complete(
                        asyncio.gather(*remaining, return_exceptions=True)
                    )
                except BaseException:
                    pass
            if broker is not None:
                try:
                    loop.run_until_complete(broker.close())
                except BaseException:
                    pass
            try:
                loop.remove_reader(command_reader.fileno())
            except BaseException:
                pass
            asyncio.set_event_loop(None)
            loop.close()
        command_reader.close()
        event_writer.close()


class ProcessAuthorityLane:
    """Run demand-only authority in a fork child with ordered, lossless IPC."""

    def __init__(
        self,
        executor: Any,
        *,
        workers: int,
        visit_capacity: int,
        cpu_affinity: set[int],
        batch_results: bool = False,
    ) -> None:
        if not callable(executor):
            raise TypeError("executor must be callable")
        if (
            isinstance(workers, bool)
            or not isinstance(workers, int)
            or workers <= 0
        ):
            raise ValueError("workers must be a positive integer")
        if (
            isinstance(visit_capacity, bool)
            or not isinstance(visit_capacity, int)
            or not 1 <= visit_capacity <= workers
        ):
            raise ValueError("visit_capacity must be in [1, workers]")
        if not isinstance(cpu_affinity, set) or not cpu_affinity:
            raise ValueError("cpu_affinity must be a non-empty set")
        if any(
            isinstance(cpu, bool)
            or not isinstance(cpu, int)
            or cpu < 0
            for cpu in cpu_affinity
        ):
            raise ValueError(
                "cpu_affinity entries must be non-negative integers"
            )
        if not isinstance(batch_results, bool):
            raise TypeError("batch_results must be a bool")
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("ProcessAuthorityLane requires fork")
        if not hasattr(os, "sched_setaffinity") or not hasattr(
            os, "sched_getaffinity"
        ):
            raise RuntimeError("ProcessAuthorityLane requires Linux affinity")

        self._executor = executor
        self._workers = workers
        self._visit_capacity = visit_capacity
        self._batch_results = batch_results
        self._requested_affinity = tuple(sorted(cpu_affinity))
        self._actual_affinity: tuple[int, ...] = ()
        self._context = multiprocessing.get_context("fork")
        self._command_reader, self._command_writer = self._context.Pipe(
            duplex=False
        )
        self._event_reader, self._event_writer = self._context.Pipe(
            duplex=False
        )

        self._lifecycle_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._registry_lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._async_close_lock = asyncio.Lock()
        self._process: multiprocessing.Process | None = None
        self._bridge: threading.Thread | None = None
        self._bridge_actual_affinity: tuple[int, ...] = ()
        self._next_request_id = 0
        self._requests: dict[int, _NonCancellingFuture] = {}
        self._barriers: dict[int, ConcurrentFuture[None]] = {}
        self._close_future: ConcurrentFuture[dict[str, Any]] | None = None
        self._startup_error: BaseException | None = None
        self._terminal_error: BaseException | None = None
        self._closed_snapshot: dict[str, Any] | None = None
        self._started = False
        self._accepting = True
        self._close_sent = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._orphan_results = 0
        self._command_packets_sent = 0
        self._event_packets_received = 0
        self._submit_batches = 0
        self._max_submit_batch_size = 0
        self._result_batches_received = 0
        self._result_packets_received = 0
        self._result_events_received = 0
        self._max_result_batch_size = 0
        self._fork_started_before_bridge = False

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    def _next_id_locked(self) -> int:
        self._next_request_id += 1
        return self._next_request_id

    def _send_locked(self, command: Any) -> None:
        try:
            with self._send_lock:
                self._command_writer.send(command)
            self._command_packets_sent += 1
        except BaseException as exc:
            error = AuthorityProcessLaneError(
                f"authority command transport failed: {exc!r}"
            )
            self._terminal_error = error
            self._accepting = False
            raise error from exc

    def start(self, *, timeout: float = 5.0) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0.0
        ):
            raise ValueError("timeout must be finite and non-negative")

        with self._lifecycle_lock:
            if self._closed_snapshot is not None or self._close_sent:
                raise AuthorityProcessLaneError(
                    "cannot start a closed authority lane"
                )
            if self._started:
                if not self._ready.is_set():
                    raise AuthorityProcessLaneError(
                        "authority lane start is already in progress"
                    )
                if self._startup_error is not None:
                    raise AuthorityProcessLaneError(
                        f"authority lane startup failed: {self._startup_error}"
                    ) from self._startup_error
                return

            self._process = self._context.Process(
                target=_authority_process_worker,
                args=(
                    self._executor,
                    self._workers,
                    self._visit_capacity,
                    self._requested_affinity,
                    self._batch_results,
                    self._command_reader,
                    self._event_writer,
                    self._command_writer,
                    self._event_reader,
                ),
                name="authority-process-lane",
                daemon=False,
            )
            # This ordering is an intentional fork-safety invariant.
            self._process.start()
            self._command_reader.close()
            self._event_writer.close()
            self._fork_started_before_bridge = self._bridge is None
            self._bridge = threading.Thread(
                target=self._bridge_events,
                name="authority-process-result-bridge",
                daemon=True,
            )
            self._bridge.start()
            self._started = True

            if not self._ready.wait(float(timeout)):
                self._accepting = False
                raise TimeoutError(
                    "authority process lane did not become ready"
                )
            if self._startup_error is not None:
                self._accepting = False
                process = self._process
                bridge = self._bridge
                if process is not None:
                    process.join(timeout=float(timeout))
                if bridge is not None:
                    bridge.join(timeout=float(timeout))
                raise AuthorityProcessLaneError(
                    f"authority lane startup failed: {self._startup_error}"
                ) from self._startup_error
            if self._actual_affinity != self._requested_affinity:
                self._accepting = False
                raise AuthorityProcessLaneError(
                    "authority CPU affinity was not exactly certified"
                )

    def submit(
        self,
        invocation: Invocation,
        *,
        session_id: str,
        scheduled_at: float,
    ) -> ConcurrentFuture[AuthorityCompletion]:
        """Submit one request through the same path as a size-one batch."""

        return self.submit_batch(
            ((invocation, session_id, scheduled_at),)
        )[0]

    def submit_batch(
        self,
        entries: Iterable[tuple[Invocation, str, float]],
    ) -> tuple[ConcurrentFuture[AuthorityCompletion], ...]:
        """Register and transmit a non-empty request batch atomically.

        Each entry is ``(invocation, session_id, scheduled_at)``.  Validation
        happens before any future is registered, and all registered requests
        are carried by exactly one ordered pipe command.
        """

        try:
            raw_entries = tuple(entries)
        except TypeError as exc:
            raise TypeError(
                "entries must be an iterable of request tuples"
            ) from exc
        if not raw_entries:
            raise ValueError("entries must not be empty")

        normalized: list[tuple[Invocation, str, float]] = []
        for index, entry in enumerate(raw_entries):
            try:
                invocation, session_id, scheduled_at = entry
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"entries[{index}] must be "
                    "(Invocation, session_id, scheduled_at)"
                ) from exc
            if not isinstance(invocation, Invocation):
                raise TypeError(
                    f"entries[{index}] invocation must be an Invocation"
                )
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(
                    f"entries[{index}] session_id must be a non-empty string"
                )
            if (
                isinstance(scheduled_at, bool)
                or not isinstance(scheduled_at, (int, float))
                or not math.isfinite(float(scheduled_at))
            ):
                raise ValueError(
                    f"entries[{index}] scheduled_at must be finite"
                )
            normalized.append(
                (invocation, session_id, float(scheduled_at))
            )

        with self._lifecycle_lock:
            if (
                not self._started
                or not self._ready.is_set()
                or not self._accepting
                or self._startup_error is not None
                or self._terminal_error is not None
            ):
                raise AuthorityProcessLaneError(
                    "authority process lane is not accepting requests"
                )
            commands: list[_SubmitEntry] = []
            futures: list[_NonCancellingFuture] = []
            for invocation, session_id, scheduled_at in normalized:
                request_id = self._next_id_locked()
                commands.append(
                    _SubmitEntry(
                        request_id=request_id,
                        invocation=invocation,
                        session_id=session_id,
                        scheduled_at=scheduled_at,
                    )
                )
                futures.append(_NonCancellingFuture())
            with self._registry_lock:
                self._requests.update(
                    (command.request_id, future)
                    for command, future in zip(commands, futures)
                )
            try:
                self._send_locked(
                    _SubmitBatchCommand(entries=tuple(commands))
                )
            except BaseException:
                with self._registry_lock:
                    for command in commands:
                        self._requests.pop(command.request_id, None)
                raise
            batch_size = len(commands)
            self._submitted += batch_size
            self._submit_batches += 1
            self._max_submit_batch_size = max(
                self._max_submit_batch_size,
                batch_size,
            )
            return tuple(futures)

    async def barrier(self) -> None:
        with self._lifecycle_lock:
            if (
                not self._started
                or not self._accepting
                or self._terminal_error is not None
            ):
                raise AuthorityProcessLaneError(
                    "authority process lane is not accepting barriers"
                )
            request_id = self._next_id_locked()
            future: ConcurrentFuture[None] = ConcurrentFuture()
            with self._registry_lock:
                self._barriers[request_id] = future
            try:
                self._send_locked(_BarrierCommand(request_id=request_id))
            except BaseException:
                with self._registry_lock:
                    self._barriers.pop(request_id, None)
                raise
        await asyncio.shield(asyncio.wrap_future(future))

    async def aclose(self) -> dict[str, Any]:
        async with self._async_close_lock:
            if self._closed_snapshot is not None:
                return self._closed_snapshot

            with self._lifecycle_lock:
                self._accepting = False
                if not self._started:
                    self._close_sent = True
                    self._close_parent_connections()
                    self._closed_snapshot = self._empty_snapshot()
                    return self._closed_snapshot

                if self._close_future is None:
                    request_id = self._next_id_locked()
                    self._close_future = ConcurrentFuture()
                    try:
                        self._send_locked(_CloseCommand(request_id=request_id))
                    except BaseException as exc:
                        if not self._close_future.done():
                            self._close_future.set_exception(exc)
                    self._close_sent = True
                close_future = self._close_future

            assert close_future is not None
            snapshot = await asyncio.shield(
                asyncio.wrap_future(close_future)
            )

            process = self._process
            bridge = self._bridge
            if process is not None:
                await asyncio.to_thread(process.join)
            if bridge is not None:
                await asyncio.to_thread(bridge.join)
            self._close_parent_connections()

            finalized = dict(snapshot)
            finalized.update(
                {
                    "process_alive": (
                        False if process is None else process.is_alive()
                    ),
                    "process_exitcode": (
                        None if process is None else process.exitcode
                    ),
                    "bridge_alive": (
                        False if bridge is None else bridge.is_alive()
                    ),
                    "bridge_actual_cpu_affinity": list(
                        self._bridge_actual_affinity
                    ),
                    "fork_started_before_bridge": (
                        self._fork_started_before_bridge
                    ),
                    "submitted": self._submitted,
                    "completed": self._completed,
                    "failed": self._failed,
                    "orphan_results": self._orphan_results,
                    "ipc_stats": self._ipc_stats(),
                }
            )
            if finalized["process_alive"]:
                raise AuthorityProcessLaneError(
                    "authority process remained alive after close"
                )
            if finalized["process_exitcode"] not in (0, None):
                raise AuthorityProcessLaneError(
                    "authority process exited unsuccessfully: "
                    f"{finalized['process_exitcode']}"
                )
            self._closed_snapshot = finalized
            return finalized

    def _bridge_events(self) -> None:
        try:
            self._bridge_actual_affinity = tuple(
                sorted(os.sched_getaffinity(0))
            )
        except BaseException:
            self._bridge_actual_affinity = ()

        saw_closed = False
        while True:
            try:
                event = self._event_reader.recv()
            except (EOFError, OSError) as exc:
                if not saw_closed:
                    error = AuthorityProcessLaneError(
                        "authority process event transport closed before "
                        f"the close handshake: {exc!r}"
                    )
                    self._terminal_error = error
                    if not self._ready.is_set():
                        self._startup_error = error
                        self._ready.set()
                    self._fail_waiters(error)
                break
            except BaseException as exc:
                error = AuthorityProcessLaneError(
                    f"authority result bridge failed: {exc!r}"
                )
                self._terminal_error = error
                if not self._ready.is_set():
                    self._startup_error = error
                    self._ready.set()
                self._fail_waiters(error)
                break

            self._event_packets_received += 1
            if isinstance(event, _ReadyEvent):
                self._actual_affinity = event.actual_cpu_affinity
                if event.error is not None:
                    self._startup_error = RemoteAuthorityError(event.error)
                    self._terminal_error = self._startup_error
                self._ready.set()
            elif isinstance(event, _ResultBatchEvent):
                batch_size = len(event.events)
                if batch_size <= 0:
                    error = AuthorityProcessLaneError(
                        "authority result batch was empty"
                    )
                    self._terminal_error = error
                    self._fail_waiters(error)
                    break
                self._result_packets_received += 1
                self._result_batches_received += 1
                self._result_events_received += batch_size
                self._max_result_batch_size = max(
                    self._max_result_batch_size,
                    batch_size,
                )
                for result_event in event.events:
                    self._complete_result(result_event)
            elif isinstance(event, _ResultEvent):
                self._result_packets_received += 1
                self._result_events_received += 1
                self._max_result_batch_size = max(
                    self._max_result_batch_size,
                    1,
                )
                self._complete_result(event)
            elif isinstance(event, _BarrierEvent):
                if event.request_id == -1:
                    error = (
                        RemoteAuthorityError(event.error)
                        if event.error is not None
                        else AuthorityProcessLaneError(
                            "authority child reported a protocol failure"
                        )
                    )
                    self._terminal_error = error
                    self._fail_waiters(error)
                    continue
                with self._registry_lock:
                    future = self._barriers.pop(event.request_id, None)
                if future is None or future.done():
                    continue
                if event.error is not None:
                    future.set_exception(RemoteAuthorityError(event.error))
                else:
                    future.set_result(None)
            elif isinstance(event, _ClosedEvent):
                saw_closed = True
                close_future = self._close_future
                if close_future is not None and not close_future.done():
                    if event.error is not None:
                        close_future.set_exception(
                            RemoteAuthorityError(event.error)
                        )
                    else:
                        close_future.set_result(event.snapshot)
                break
            else:
                error = AuthorityProcessLaneError(
                    f"unknown authority event: {type(event).__name__}"
                )
                self._terminal_error = error
                self._fail_waiters(error)
                break

        self._stopped.set()

    def _complete_result(self, event: _ResultEvent) -> None:
        """Complete one request from a possibly batched result packet."""

        with self._registry_lock:
            future = self._requests.pop(event.request_id, None)
        if future is None:
            self._orphan_results += 1
            return
        if event.error is not None:
            self._failed += 1
            future.set_exception(RemoteAuthorityError(event.error))
        elif event.completion is None:
            self._failed += 1
            future.set_exception(
                AuthorityProcessLaneError(
                    "authority result event had no payload"
                )
            )
        else:
            self._completed += 1
            future.set_result(event.completion)

    def _fail_waiters(self, error: BaseException) -> None:
        with self._registry_lock:
            requests = tuple(self._requests.values())
            barriers = tuple(self._barriers.values())
            self._requests.clear()
            self._barriers.clear()
        for future in (*requests, *barriers):
            if not future.done():
                future.set_exception(error)
        close_future = self._close_future
        if close_future is not None and not close_future.done():
            close_future.set_exception(error)

    def _close_parent_connections(self) -> None:
        for connection in (self._command_writer, self._event_reader):
            try:
                connection.close()
            except BaseException:
                pass

    def _ipc_stats(self) -> dict[str, int | bool]:
        return {
            "command_packets_sent": self._command_packets_sent,
            "submit_batches": self._submit_batches,
            "submitted_requests": self._submitted,
            "max_submit_batch_size": self._max_submit_batch_size,
            "event_packets_received": self._event_packets_received,
            "result_batching_enabled": self._batch_results,
            "result_packets_received": self._result_packets_received,
            "result_batches": self._result_batches_received,
            "result_events_received": self._result_events_received,
            "max_result_batch_size": self._max_result_batch_size,
        }

    def _empty_snapshot(self) -> dict[str, Any]:
        return {
            "snapshot": {"jobs": []},
            "stats": {},
            "authoritative_state_count": 0,
            "requested_cpu_affinity": list(self._requested_affinity),
            "actual_cpu_affinity": list(self._actual_affinity),
            "process_pid": None,
            "process_alive": False,
            "process_exitcode": None,
            "bridge_alive": False,
            "bridge_actual_cpu_affinity": [],
            "fork_started_before_bridge": False,
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "orphan_results": 0,
            "ipc_stats": self._ipc_stats(),
        }
