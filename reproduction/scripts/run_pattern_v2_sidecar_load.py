#!/usr/bin/env python3
"""Evaluate control-plane-isolated speculative execution under task load.

The authoritative lane is the unchanged demand-only :class:`LiveToolBroker`.
Prediction execution, deadlines, admission, and lease cleanup live in a
dedicated sidecar process. Every authoritative call is submitted regardless
of a speculative hit (the paper's ``shadow authority`` safety mode), and a
valid speculative success may only shorten the logical result time. The
all-wrong authority path sends no cleanup packet; when a process sidecar is
activated, its blocking result bridge is started during untimed setup.

This is a CPU-only replay with deterministic synthetic tool service.  It does
not start a model server or issue network requests.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
from threading import Event, Thread, get_ident
import time
from typing import Any


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.invocation import Invocation  # noqa: E402
from paste_repro.authority_process_lane import (  # noqa: E402
    ProcessAuthorityLane,
)
from paste_repro.live_broker import LiveAuthoritativeResult, LiveToolBroker  # noqa: E402
from paste_repro.speculation_sidecar import (  # noqa: E402
    ProcessSpeculativeSidecar,
    SpeculativeHandle,
    SpeculativeSidecar,
    choose_authority_control_sidecar_cpus,
    choose_authority_sidecar_cpus,
    distinct_physical_core_certificate,
)
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
from run_pattern_cache_evaluation import sha256_file  # noqa: E402


SCHEMA = "paste_repro.pattern_v2_sidecar_load.v2"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "results" / "pattern_v2_sidecar_load"
STRICT_PULL_MAX_STAGED_RESULT_BYTES = 4 * 1024

_ONE_SIDED_T95 = (
    math.nan,
    6.313752,
    2.919986,
    2.353363,
    2.131847,
    2.015048,
    1.943180,
    1.894579,
    1.859548,
    1.833113,
    1.812461,
    1.795885,
    1.782288,
    1.770933,
    1.761310,
    1.753050,
    1.745884,
    1.739607,
    1.734064,
    1.729133,
    1.724718,
    1.720743,
    1.717144,
    1.713872,
    1.710882,
    1.708141,
    1.705618,
    1.703288,
    1.701131,
    1.699127,
    1.697261,
)


@dataclass(frozen=True)
class AuthorityCompletion:
    result: LiveAuthoritativeResult
    scheduled_at: float
    first_run_at: float
    terminal_at: float
    observed_at: float


@dataclass(frozen=True)
class LogicalCompletion:
    source: str
    terminal_at: float
    speculative_executor_terminal_at: float | None = None


class DedicatedAuthorityLane:
    """Run the unchanged demand-only broker on a private asyncio loop.

    This is an experimental event-loop/run-queue isolation mode.  The lane is
    a thread, not a process, so it deliberately does not claim GIL isolation.
    It must be started only after every fork-backed sidecar has started.
    """

    def __init__(
        self,
        executor: Any,
        *,
        workers: int,
        visit_capacity: int,
        cpu_affinity: set[int],
    ) -> None:
        self._executor = executor
        self._workers = workers
        self._visit_capacity = visit_capacity
        self._requested_affinity = set(cpu_affinity)
        self._actual_affinity: set[int] = set()
        self._ready = Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._broker: LiveToolBroker | None = None
        self._startup_error: BaseException | None = None
        self._closed_snapshot: dict[str, Any] | None = None
        self._thread = Thread(
            target=self._thread_main,
            name="dedicated-authority-loop",
            daemon=True,
        )

    def _thread_main(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            os.sched_setaffinity(0, self._requested_affinity)
            self._actual_affinity = set(os.sched_getaffinity(0))
            if self._actual_affinity != self._requested_affinity:
                raise RuntimeError(
                    "authority affinity mismatch: "
                    f"requested={sorted(self._requested_affinity)}, "
                    f"actual={sorted(self._actual_affinity)}"
                )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            broker = LiveToolBroker(
                self._executor,
                max_workers=self._workers,
                max_speculative_workers=0,
                max_speculative_pending=1,
                ttl_s=1.0,
                tool_capacities={"visit": self._visit_capacity},
            )
            self._loop = loop
            self._broker = broker
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            if loop is not None:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                asyncio.set_event_loop(None)
                loop.close()

    def start(self, *, timeout: float = 5.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("dedicated authority lane did not become ready")
        if self._startup_error is not None:
            self._thread.join(timeout=timeout)
            raise RuntimeError(
                f"dedicated authority lane startup failed: {self._startup_error}"
            ) from self._startup_error
        if self._loop is None or self._broker is None:
            raise RuntimeError("dedicated authority lane has no loop/broker")

    def submit(
        self,
        invocation: Invocation,
        *,
        session_id: str,
        scheduled_at: float,
    ) -> ConcurrentFuture[AuthorityCompletion]:
        if self._loop is None or self._broker is None:
            raise RuntimeError("dedicated authority lane is not running")
        return asyncio.run_coroutine_threadsafe(
            _authority_call(
                self._broker,
                invocation,
                session_id=session_id,
                scheduled_at=scheduled_at,
            ),
            self._loop,
        )

    async def barrier(self) -> None:
        if self._loop is None:
            raise RuntimeError("dedicated authority lane is not running")

        async def authority_turn() -> None:
            await asyncio.sleep(0)

        remote = asyncio.run_coroutine_threadsafe(authority_turn(), self._loop)
        await asyncio.wrap_future(remote)

    async def aclose(self) -> dict[str, Any]:
        if self._closed_snapshot is not None:
            return self._closed_snapshot
        if self._loop is None or self._broker is None:
            if self._thread.is_alive():
                self._thread.join(timeout=5.0)
            return {
                "snapshot": {"jobs": []},
                "stats": {},
                "authoritative_state_count": 0,
                "requested_cpu_affinity": sorted(self._requested_affinity),
                "actual_cpu_affinity": sorted(self._actual_affinity),
                "thread_alive": self._thread.is_alive(),
            }

        async def close_lane() -> dict[str, Any]:
            assert self._broker is not None
            await self._broker.close()
            return {
                "snapshot": self._broker.snapshot(),
                "stats": self._broker.stats.to_dict(),
                "authoritative_state_count": len(
                    self._broker.authoritative_state
                ),
                "requested_cpu_affinity": sorted(self._requested_affinity),
                "actual_cpu_affinity": sorted(self._actual_affinity),
                "native_id": self._thread.native_id,
            }

        remote = asyncio.run_coroutine_threadsafe(close_lane(), self._loop)
        wrapped = asyncio.wrap_future(remote)
        cancelled = False
        try:
            snapshot = await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            cancelled = True
            snapshot = await wrapped
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise TimeoutError("dedicated authority lane did not stop")
        snapshot["thread_alive"] = False
        self._closed_snapshot = snapshot
        if cancelled:
            raise asyncio.CancelledError
        return snapshot


def _sleep_until(deadline: float) -> None:
    """Block a release-clock thread until an absolute monotonic deadline."""

    remaining = deadline - time.monotonic()
    if remaining > 0.0:
        time.sleep(remaining)


@dataclass(frozen=True)
class _PullPrestageRelease:
    """Result of a silent off-loop pull prefetch."""

    packets: int
    elapsed_ms: float
    sealed: bool
    deadline_miss: bool
    worker_thread_id: int
    worker_cpu_affinity: tuple[int, ...]
    finished_at: float
    prefetch_deadline: float
    error: str | None = None


def _prefetch_pull_epoch(
    sidecar: ProcessSpeculativeSidecar,
    completion_cutoff: float,
    prefetch_deadline: float,
) -> _PullPrestageRelease:
    """Seal one pull epoch without registering an asyncio callback.

    The caller submits this through ``ThreadPoolExecutor.submit`` and never
    wraps the raw concurrent Future in asyncio.  Completion therefore does not
    wake the parent event loop at the cutoff.  Authority has a separate,
    baseline-identical release clock and never waits for this worker.
    """

    _sleep_until(completion_cutoff)
    started = time.perf_counter()
    packets = 0
    sealed = False
    error: str | None = None
    try:
        packets = sidecar.prefetch_pull_results(
            deadline=prefetch_deadline,
        )
        sealed = sidecar.pull_epoch_sealed
    except Exception as exc:
        # A speculative transport failure must not release the parent loop
        # early.  Preserve the authority clock and suppress this epoch's claim.
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    worker_cpu_affinity = tuple(
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else ()
    )
    # Take the timestamp after the worker body's final syscall/bookkeeping.
    # ThreadPoolExecutor's internal set_result epilogue remains outside this
    # measurement, so this certifies pull-worker-body quiet time only.
    finished_at = time.monotonic()
    deadline_miss = finished_at > prefetch_deadline or not sealed or error is not None
    return _PullPrestageRelease(
        packets=packets,
        elapsed_ms=elapsed_ms,
        sealed=sealed,
        deadline_miss=deadline_miss,
        worker_thread_id=get_ident(),
        worker_cpu_affinity=worker_cpu_affinity,
        finished_at=finished_at,
        prefetch_deadline=prefetch_deadline,
        error=error,
    )


def _timer_ready() -> None:
    """Warm a release-clock worker before the measured interval."""


def _pin_current_worker(cpu_affinity: set[int] | None) -> tuple[int, ...]:
    """Pin one already-forked helper thread and report its actual mask."""

    if cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpu_affinity)
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return ()


def _poll_pull_prefetch(
    future: ConcurrentFuture[_PullPrestageRelease],
) -> tuple[_PullPrestageRelease | None, bool]:
    """Observe a raw prefetch Future without waiting for unfinished work."""

    done = future.done()
    if not done:
        # This succeeds only before the worker begins.  A running worker is
        # joined during post-authority teardown and invalidates strict timing.
        future.cancel()
        return None, False
    try:
        return future.result(timeout=0.0), True
    except Exception:
        return None, True


def _t95(df: int) -> float:
    if df <= 0:
        return math.inf
    if df < len(_ONE_SIDED_T95):
        return _ONE_SIDED_T95[df]
    # The normal limit is slightly anti-conservative for finite df.  Use the
    # df=30 value for all larger samples; this only widens the interval.
    return _ONE_SIDED_T95[-1]


def _paired_repeat_inference(
    values: Sequence[float],
    *,
    margin: float,
    minimum_repetitions: int = 8,
) -> dict[str, Any]:
    """One-sided non-inferiority inference with paired repeat as the unit."""

    n = len(values)
    mean = statistics.fmean(values) if values else 0.0
    # The decision is always insufficient below the minimum.  Keep the R=1
    # descriptive payload finite so metrics.json remains strict JSON.
    sd = statistics.stdev(values) if n >= 2 else 0.0
    se = sd / math.sqrt(n) if n >= 2 else 0.0
    half_width = _t95(n - 1) * se if n >= 2 else 0.0
    lower = mean - half_width
    upper = mean + half_width
    if n < minimum_repetitions:
        decision = "insufficient_repetitions"
    elif upper <= margin:
        decision = "pass"
    elif lower > margin:
        decision = "regression"
    else:
        decision = "inconclusive"
    return {
        "inference_unit": "paired_repeat",
        "n": n,
        "minimum_repetitions": minimum_repetitions,
        "mean": mean,
        "sd_between_repeats": sd,
        "se": se,
        "ci90": [lower, upper],
        "lower_95": lower,
        "upper_95": upper,
        "margin": margin,
        "decision": decision,
    }


def _paired_benefit_inference(
    values: Sequence[float],
    *,
    minimum_repetitions: int = 8,
) -> dict[str, Any]:
    """Classify positive paired logical-latency benefit at repeat level."""

    result = _paired_repeat_inference(
        values,
        margin=0.0,
        minimum_repetitions=minimum_repetitions,
    )
    if len(values) < minimum_repetitions:
        decision = "insufficient_repetitions"
    elif float(result["lower_95"]) > 0.0:
        decision = "improvement"
    elif float(result["upper_95"]) < 0.0:
        decision = "regression"
    else:
        decision = "inconclusive"
    result["decision"] = decision
    result["scale"] = "ms/target saved"
    return result


def _sidecar_count(snapshot: Any, *names: str) -> int:
    """Read one integer counter from dict or dataclass snapshots."""

    for name in names:
        if isinstance(snapshot, Mapping) and name in snapshot:
            return int(snapshot[name])
        if (
            isinstance(snapshot, Mapping)
            and isinstance(snapshot.get("stats"), Mapping)
            and name in snapshot["stats"]
        ):
            return int(snapshot["stats"][name])
        if hasattr(snapshot, name):
            return int(getattr(snapshot, name))
    return 0


def _snapshot_json(snapshot: Any) -> dict[str, Any]:
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    if hasattr(snapshot, "to_dict"):
        value = snapshot.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if hasattr(snapshot, "__dict__"):
        return {
            key: value
            for key, value in vars(snapshot).items()
            if isinstance(value, (str, int, float, bool, type(None), dict, list))
        }
    return {"repr": repr(snapshot)}


async def _authority_call(
    broker: LiveToolBroker,
    invocation: Invocation,
    *,
    session_id: str,
    scheduled_at: float,
) -> AuthorityCompletion:
    first_run_at = time.perf_counter()
    result = await broker.authoritative(
        invocation,
        session_id=session_id,
        speculation_eligible=False,
    )
    terminal_at = time.perf_counter()
    return AuthorityCompletion(
        result=result,
        scheduled_at=scheduled_at,
        first_run_at=first_run_at,
        terminal_at=terminal_at,
        observed_at=terminal_at,
    )


async def _observe_remote_authority(
    remote: ConcurrentFuture[AuthorityCompletion],
) -> AuthorityCompletion:
    # The proxy is observational only.  Caller cancellation must never
    # propagate across the thread boundary and cancel the authority call.
    completion = await asyncio.shield(asyncio.wrap_future(remote))
    return replace(completion, observed_at=time.perf_counter())


def _arm_shadow_authority_race(
    authority: "asyncio.Task[AuthorityCompletion]",
    handle: SpeculativeHandle | None,
    *,
    invocation: Invocation,
    loop: asyncio.AbstractEventLoop,
    raw_authority: ConcurrentFuture[AuthorityCompletion] | None = None,
    speculative_terminal_cutoff: float | None = None,
) -> "asyncio.Future[LogicalCompletion]":
    """Arm a single-notification logical race outside the authority loop.

    Authority has tie priority and is never cancelled.  Speculative rejection
    or failure simply falls back to the already-running authority attempt.

    ``asyncio.wrap_future`` plus ``asyncio.wait`` creates several callbacks and
    a join Task on the authority event loop for every exact hit.  Lightweight
    tools make that fixed control cost visible.  Instead, validate the process
    result in the concurrent-future callback (the result bridge thread) and
    enqueue only the one notification that can release the logical caller.
    """

    logical: asyncio.Future[LogicalCompletion] = loop.create_future()
    settled = Event()

    def settle_authority(
        completed_authority: "asyncio.Task[AuthorityCompletion]",
    ) -> None:
        if logical.done():
            if not completed_authority.cancelled():
                # Observe a losing shadow-authority exception even if outer
                # teardown skips the normal gather; retrieving it does not
                # prevent a later await from raising the same exception.
                completed_authority.exception()
            return
        if completed_authority.cancelled():
            settled.set()
            logical.cancel()
            return
        try:
            completion = completed_authority.result()
        except BaseException as exc:
            settled.set()
            logical.set_exception(exc)
            return
        settled.set()
        logical.set_result(
            LogicalCompletion("authority", completion.observed_at)
        )

    logical.add_done_callback(lambda _: settled.set())
    authority.add_done_callback(settle_authority)
    if handle is None:
        return logical

    def validated_speculative_terminal(
        result_future: ConcurrentFuture[Any],
    ) -> tuple[bool, float | None]:
        if settled.is_set():
            return False, None
        try:
            result = result_future.result()
            if not isinstance(result, Mapping):
                return False, None
            raw_key = result.get("invocation_key")
            if (
                not isinstance(raw_key, (list, tuple))
                or tuple(raw_key) != invocation.key
            ):
                return False, None
            executor_terminal = result.get("executor_terminal_at")
            if speculative_terminal_cutoff is not None and (
                isinstance(executor_terminal, bool)
                or not isinstance(executor_terminal, (int, float))
                or not math.isfinite(float(executor_terminal))
                or float(executor_terminal) > speculative_terminal_cutoff
            ):
                return False, None
        except BaseException:
            return False, None
        return (
            True,
            (
                float(executor_terminal)
                if isinstance(executor_terminal, (int, float))
                and not isinstance(executor_terminal, bool)
                else None
            ),
        )

    def settle_speculative(result_future: ConcurrentFuture[Any]) -> None:
        # This callback normally runs in the result-bridge thread.  Do all
        # decoding/identity checks here so the authority loop sees at most one
        # small callback for a valid result that can still win.
        valid, executor_terminal = validated_speculative_terminal(
            result_future
        )
        if not valid:
            return

        def deliver_speculative() -> None:
            if logical.done():
                return
            # A terminal authority wins a transport/scheduling tie even when
            # its registered done callback has not run yet.
            if authority.done():
                settle_authority(authority)
                return
            # A cross-loop authority completion may be terminal before its
            # control-loop proxy has received the wakeup.  Consult the raw,
            # thread-safe future so speculation cannot steal that transport
            # race; the proxy's registered callback will publish authority.
            if raw_authority is not None and raw_authority.done():
                return
            settled.set()
            logical.set_result(
                LogicalCompletion(
                    "sidecar",
                    time.perf_counter(),
                    executor_terminal,
                )
            )

        try:
            loop.call_soon_threadsafe(deliver_speculative)
        except RuntimeError:
            # The loop can only be closed during teardown; authority/result
            # cleanup owns the remaining terminal state in that case.
            return

    # A precompleted/staged result needs no cross-thread notification at
    # confirmation.  concurrent.futures would otherwise invoke the registered
    # done callback synchronously here, only for that callback to enqueue a
    # redundant call_soon_threadsafe back onto this same control loop.
    if handle.future.done():
        valid, executor_terminal = validated_speculative_terminal(
            handle.future
        )
        if valid:
            if authority.done():
                settle_authority(authority)
            elif raw_authority is None or not raw_authority.done():
                settled.set()
                logical.set_result(
                    LogicalCompletion(
                        "sidecar",
                        time.perf_counter(),
                        executor_terminal,
                    )
                )
        return logical

    handle.future.add_done_callback(settle_speculative)
    return logical


async def _run_sample_impl(
    windows: Sequence[ScoredWindow],
    *,
    offered_concurrency: int,
    seed: int,
    workers: int,
    visit_capacity: int,
    service_ms: float,
    lead_ms: float,
    sidecar_slots: int,
    max_sidecar_pending: int,
    probability_threshold: float,
    sidecar_backend: str = "process",
    cpu_isolation: bool = True,
    shadow_barrier: bool = False,
    authority_control_burst_limit: int = 0,
    dedicated_authority_thread: bool = False,
    dedicated_authority_process: bool = False,
    cpu_role_assignment: tuple[int, int, int] | None = None,
    require_precompletion: bool = False,
    completion_guard_ms: float = 0.0,
    eager_result_staging: bool = False,
    pull_result_staging: bool = False,
    coordination_cost_ms: float = 0.0,
    certified_exclusive_resources: bool = False,
    unsafe_positive_ablation: bool = False,
) -> dict[str, Any]:
    service_s = service_ms / 1000.0
    lead_s = lead_ms / 1000.0
    if authority_control_burst_limit < 0:
        raise ValueError("authority_control_burst_limit must be non-negative")
    if (
        isinstance(completion_guard_ms, bool)
        or not isinstance(completion_guard_ms, (int, float))
        or not math.isfinite(float(completion_guard_ms))
        or completion_guard_ms < 0.0
    ):
        raise ValueError("completion_guard_ms must be finite and non-negative")
    if (
        isinstance(coordination_cost_ms, bool)
        or not isinstance(coordination_cost_ms, (int, float))
        or not math.isfinite(float(coordination_cost_ms))
        or coordination_cost_ms < 0.0
    ):
        raise ValueError("coordination_cost_ms must be finite and non-negative")
    completion_guard_s = float(completion_guard_ms) / 1000.0
    predicted_precompletion = (
        lead_s >= service_s + completion_guard_s
    )
    pull_prestage_enabled = (
        pull_result_staging
        and require_precompletion
        and completion_guard_s > 0.0
    )
    if dedicated_authority_thread and dedicated_authority_process:
        raise ValueError("authority thread and process modes are exclusive")
    if eager_result_staging and pull_result_staging:
        raise ValueError("eager and pull result staging are mutually exclusive")
    if (
        eager_result_staging or pull_result_staging
    ) and sidecar_backend != "process":
        raise ValueError("result staging requires a process sidecar")
    dedicated_authority = (
        dedicated_authority_thread or dedicated_authority_process
    )
    strict_resource_profile_requested = (
        sidecar_backend == "process"
        and cpu_isolation
        and shadow_barrier
        and require_precompletion
        # Preserve the original in-process authority path only when completed
        # results stay in a silent kernel mailbox until an exact bounded pull.
        # Eager staging still runs a parent GIL/bridge callback for every wrong
        # result and therefore cannot certify the direct authority hot path.
        # A dedicated authority process is a useful topology diagnostic, but
        # migrating the original authority path has a measurable handoff tax
        # and therefore must never satisfy the public strict certificate.
        and pull_result_staging
        and pull_prestage_enabled
        and not dedicated_authority
        and certified_exclusive_resources
    )
    positive_resource_candidate = (
        sidecar_slots > 0
        and authority_control_burst_limit > 0
        and (
            strict_resource_profile_requested
            or unsafe_positive_ablation
        )
    )
    positive_resource_certificate = positive_resource_candidate
    original_cpu_affinity = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set()
    )
    authority_cpu_affinity: set[int] | None = None
    control_cpu_affinity: set[int] | None = None
    sidecar_cpu_affinity: set[int] | None = None
    actual_control_cpu_affinity: set[int] = set()
    if cpu_isolation:
        available_cpus = sorted(original_cpu_affinity)
        required = (
            3
            if dedicated_authority and positive_resource_certificate
            else 2
            if dedicated_authority or positive_resource_certificate
            else 1
        )
        if len(available_cpus) < required:
            raise RuntimeError(
                "CPU isolation has fewer granted CPUs than required by the "
                "active resource certificate"
            )
        if dedicated_authority:
            if cpu_role_assignment is not None:
                authority_cpu, control_cpu, sidecar_cpu = cpu_role_assignment
                if (
                    len({authority_cpu, control_cpu, sidecar_cpu}) != 3
                    or not {authority_cpu, control_cpu, sidecar_cpu}.issubset(
                        available_cpus
                    )
                ):
                    raise RuntimeError("invalid three-lane CPU role assignment")
            elif positive_resource_certificate:
                authority_cpu, control_cpu, sidecar_cpu = (
                    choose_authority_control_sidecar_cpus(available_cpus)
                )
            else:
                authority_cpu, control_cpu = choose_authority_sidecar_cpus(
                    available_cpus
                )
                sidecar_cpu = -1
            authority_cpu_affinity = {authority_cpu}
            control_cpu_affinity = {control_cpu}
            if positive_resource_certificate:
                sidecar_cpu_affinity = {sidecar_cpu}
            os.sched_setaffinity(0, control_cpu_affinity)
        elif positive_resource_certificate:
            authority_cpu, sidecar_cpu = choose_authority_sidecar_cpus(
                available_cpus
            )
            authority_cpu_affinity = {authority_cpu}
            control_cpu_affinity = {authority_cpu}
            sidecar_cpu_affinity = {sidecar_cpu}
            os.sched_setaffinity(0, authority_cpu_affinity)
        else:
            authority_cpu_affinity = {available_cpus[0]}
            control_cpu_affinity = {available_cpus[0]}
            os.sched_setaffinity(0, authority_cpu_affinity)
        actual_control_cpu_affinity = set(os.sched_getaffinity(0))
    elif dedicated_authority:
        raise RuntimeError(
            "dedicated authority thread requires explicit CPU isolation"
        )

    placement_cpus = set().union(
        authority_cpu_affinity or set(),
        control_cpu_affinity or set(),
        sidecar_cpu_affinity or set(),
    )
    physical_core_placement_certified = (
        not positive_resource_candidate
        or distinct_physical_core_certificate(placement_cpus)
    )
    strict_static_resource_certificate = (
        strict_resource_profile_requested
        and physical_core_placement_certified
    )
    if (
        positive_resource_candidate
        and not unsafe_positive_ablation
        and not strict_static_resource_certificate
    ):
        positive_resource_certificate = False
        sidecar_cpu_affinity = None

    authority_executor_started_count = 0

    async def authority_executor(invocation: Invocation) -> dict[str, Any]:
        nonlocal authority_executor_started_count
        authority_executor_started_count += 1
        await asyncio.sleep(service_s)
        return {"invocation_key": invocation.key}

    async def sidecar_executor(invocation: Invocation) -> dict[str, Any]:
        executor_started_at = time.perf_counter()
        await asyncio.sleep(service_s)
        return {
            "invocation_key": invocation.key,
            "executor_started_at": executor_started_at,
            "executor_terminal_at": time.perf_counter(),
        }

    authority: LiveToolBroker | None = None
    authority_lane: DedicatedAuthorityLane | ProcessAuthorityLane | None = None
    if not dedicated_authority:
        authority = LiveToolBroker(
            authority_executor,
            max_workers=workers,
            max_speculative_workers=0,
            max_speculative_pending=1,
            ttl_s=1.0,
            tool_capacities={"visit": visit_capacity},
        )
    sidecar: SpeculativeSidecar | ProcessSpeculativeSidecar | None = None
    loop = asyncio.get_running_loop()
    parent_loop_thread_id = get_ident()

    safe_spec = replace(
        next(spec for spec in policy_specs() if spec.name == "safe_global_benefit"),
        confidence_threshold=probability_threshold,
    )
    batches = session_stream_batches(
        windows,
        offered_concurrency=offered_concurrency,
        seed=seed,
    )
    # Predictor and global selection belong to the isolated control plane.
    # This replay precomputes their immutable decisions before the authority
    # clock starts; their CPU cost is still reported separately below.
    selection_plan: list[
        tuple[list[tuple[Any, float]], dict[str, Any]]
    ] = []
    burst_latch_open = positive_resource_certificate and (
        not require_precompletion or predicted_precompletion
    )
    if sidecar_slots > 0:
        for batch in batches:
            authority_control_burst = sum(
                len(window.executable_targets) for window in batch
            )
            burst_limit_exceeded = (
                authority_control_burst_limit == 0
                or authority_control_burst > authority_control_burst_limit
            )
            if burst_limit_exceeded:
                burst_latch_open = False
            selected, metadata = _select_candidates(
                batch,
                safe_spec,
                visit_capacity=visit_capacity,
                service_s=service_s,
                lead_s=lead_s,
                isolated_speculative_slots=sidecar_slots,
                safe_start_limit=(
                    sidecar_slots if burst_latch_open else 0
                ),
                coordination_cost_s=float(coordination_cost_ms) / 1000.0,
            )
            selection_plan.append(
                (
                    selected,
                    {
                        **metadata,
                        "authority_control_burst": (
                            authority_control_burst
                        ),
                        "authority_control_burst_limit": (
                            authority_control_burst_limit
                        ),
                        "authority_control_burst_gate_open": (
                            burst_latch_open
                        ),
                        "authority_control_burst_limit_exceeded": (
                            burst_limit_exceeded
                        ),
                        "require_precompletion": require_precompletion,
                        "predicted_precompletion": predicted_precompletion,
                        "completion_guard_ms": float(completion_guard_ms),
                        "coordination_cost_ms": float(coordination_cost_ms),
                        "strict_static_resource_certificate": (
                            strict_static_resource_certificate
                        ),
                        "physical_core_placement_certified": (
                            physical_core_placement_certified
                        ),
                        "certified_exclusive_resources": (
                            certified_exclusive_resources
                        ),
                        "unsafe_positive_ablation": (
                            unsafe_positive_ablation
                        ),
                    },
                )
            )
    selection_rows = [metadata for _, metadata in selection_plan]
    selection_selected_total = sum(
        len(selected) for selected, _ in selection_plan
    )
    sidecar_activated = sidecar_slots > 0 and selection_selected_total > 0
    result_bridge_prestarted = False
    sidecar_runtime_certificate_checked = False
    sidecar_runtime_certificate_valid = not sidecar_activated
    sidecar_runtime_certificate_error: str | None = None
    sidecar_startup_snapshot: dict[str, Any] = {}
    release_clock: ThreadPoolExecutor | None = None
    pull_prestage_clock: ThreadPoolExecutor | None = None
    actual_pull_prestage_cpu_affinity: set[int] = set()
    pull_prestage_observations: list[
        tuple[
            ConcurrentFuture[_PullPrestageRelease],
            _PullPrestageRelease | None,
            bool,
        ]
    ] = []
    active_confirmation_release: asyncio.Future[Any] | None = None

    async def cleanup_resources() -> None:
        errors: list[BaseException] = []
        # ``run_in_executor`` cancellation does not stop its worker.  Drain a
        # release/prefetch job before closing the sidecar whose socket it may
        # still be reading.  This keeps cancellation fail-open for authority
        # without introducing an executor-vs-close race.
        if active_confirmation_release is not None:
            try:
                await asyncio.shield(active_confirmation_release)
            except BaseException as exc:
                errors.append(exc)
        if authority_lane is not None:
            try:
                await authority_lane.aclose()
            except BaseException as exc:
                errors.append(exc)
        elif authority is not None:
            try:
                await authority.close()
            except BaseException as exc:
                errors.append(exc)
        # Raw prefetch Futures deliberately have no asyncio callback.  Joining
        # their executor is the final backstop when a wrapper/task is cancelled
        # or a bounded prefetch finishes late.  It must precede sidecar close.
        if pull_prestage_clock is not None:
            try:
                pull_prestage_clock.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
            except BaseException as exc:
                errors.append(exc)
        if release_clock is not None:
            try:
                release_clock.shutdown(wait=True)
            except BaseException as exc:
                errors.append(exc)
        if sidecar is not None:
            try:
                sidecar.close(wait=True)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"sample cleanup failed with {len(errors)} error(s)"
            ) from errors[0]

    async def cleanup_resources_cancellation_safe() -> None:
        cleanup_task = asyncio.create_task(cleanup_resources())
        cancelled = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # A second cancellation must not cancel teardown while a raw
                # prefetch worker can still own the sidecar receive socket.
                cancelled = True
        if cleanup_task.cancelled():
            raise RuntimeError("sample cleanup task was cancelled")
        cleanup_error = cleanup_task.exception()
        if cleanup_error is not None:
            raise cleanup_error
        if cancelled:
            raise asyncio.CancelledError

    try:
        process_sidecar_needs_startup_certificate = False
        if sidecar_activated:
            if dedicated_authority_process and sidecar_backend != "process":
                raise RuntimeError(
                    "authority process mode requires a process sidecar so all "
                    "forks precede parent helper threads"
                )
            sidecar_class = (
                ProcessSpeculativeSidecar
                if sidecar_backend == "process"
                else SpeculativeSidecar
            )
            sidecar_kwargs: dict[str, Any] = {}
            if sidecar_backend == "process":
                sidecar_kwargs["cpu_affinity"] = sidecar_cpu_affinity
                sidecar_kwargs["eager_result_staging"] = (
                    eager_result_staging
                )
                sidecar_kwargs["pull_result_staging"] = (
                    pull_result_staging
                )
                if pull_result_staging:
                    sidecar_kwargs["max_staged_result_bytes"] = (
                        STRICT_PULL_MAX_STAGED_RESULT_BYTES
                    )
                # The latest-safe-start fence moves earlier under the
                # precompletion profile, but an already-completed exact
                # result must remain claimable until authority confirmation.
                # Keep the completed result claimable through confirmation
                # plus a fixed scheduling margin. The next admission occurs
                # only after the shadow authority barrier, so expired parent
                # state is reaped outside the protected demand interval.
                sidecar_kwargs["claim_grace_s"] = max(
                    0.010,
                    (
                        service_s + completion_guard_s + 0.010
                        if require_precompletion
                        else 0.010
                    ),
                )
            sidecar = sidecar_class(
                sidecar_executor,
                max_workers=sidecar_slots,
                max_pending=max_sidecar_pending,
                **sidecar_kwargs,
            )
            sidecar.start()
            if isinstance(sidecar, ProcessSpeculativeSidecar):
                process_sidecar_needs_startup_certificate = True

        if dedicated_authority_process:
            if authority_cpu_affinity is None:
                raise RuntimeError(
                    "dedicated authority lane has no CPU assignment"
                )
            authority_lane = ProcessAuthorityLane(
                authority_executor,
                workers=workers,
                visit_capacity=visit_capacity,
                cpu_affinity=authority_cpu_affinity,
            )
            authority_lane.start()

        if process_sidecar_needs_startup_certificate:
            assert isinstance(sidecar, ProcessSpeculativeSidecar)
            # Validate the child-side placement and scheduler policy before
            # opening the timed authority path.  A failed runtime certificate
            # is a K=0 fallback, not a post-hoc safety error after speculative
            # work has already run.
            sidecar_runtime_certificate_checked = True
            try:
                if pull_result_staging:
                    # Pull staging keeps completed packets dormant in the
                    # kernel mailbox.  Its setup-only handshake must not
                    # create a continuous parent reader before timed work.
                    sidecar_startup_snapshot = sidecar.startup_snapshot()
                else:
                    result_bridge_prestarted = sidecar.start_result_bridge()
                    if not result_bridge_prestarted:
                        raise RuntimeError(
                            "could not prestart process result bridge during "
                            "setup"
                        )
                    sidecar_startup_snapshot = sidecar.snapshot()
                startup_actual_affinity = set(
                    sidecar_startup_snapshot.get("actual_cpu_affinity") or []
                )
                startup_bridge_affinity = set(
                    sidecar_startup_snapshot.get(
                        "actual_bridge_cpu_affinity"
                    )
                    or []
                )
                sidecar_runtime_certificate_valid = (
                    bool(sidecar_startup_snapshot.get("process_alive", False))
                    and sidecar_startup_snapshot.get("startup_error") is None
                    and (
                        not pull_result_staging
                        or (
                            sidecar.bridge_started is False
                            and sidecar_startup_snapshot.get(
                                "bridge_started"
                            )
                            is False
                            and sidecar_startup_snapshot.get(
                                "parent_staging", {}
                            ).get("mode")
                            == "pull"
                            and sidecar_startup_snapshot.get(
                                "capacity", {}
                            ).get("pull_result_staging")
                            is True
                        )
                    )
                    and (
                        not cpu_isolation
                        or (
                            sidecar_cpu_affinity is not None
                            and startup_actual_affinity
                            == sidecar_cpu_affinity
                            and (
                                pull_result_staging
                                or startup_bridge_affinity
                                == sidecar_cpu_affinity
                            )
                        )
                    )
                    and sidecar_startup_snapshot.get(
                        "actual_scheduler_policy"
                    )
                    == getattr(os, "SCHED_IDLE", 5)
                    and (
                        not pull_result_staging
                        or int(
                            sidecar_startup_snapshot.get(
                                "eager_result_staging", {}
                            ).get("max_staged_result_bytes", -1)
                        )
                        == STRICT_PULL_MAX_STAGED_RESULT_BYTES
                    )
                )
                if not sidecar_runtime_certificate_valid:
                    sidecar_runtime_certificate_error = (
                        "sidecar affinity/SCHED_IDLE startup certificate "
                        "did not validate"
                    )
            except BaseException as exc:
                sidecar_runtime_certificate_valid = False
                sidecar_runtime_certificate_error = repr(exc)

            if not sidecar_runtime_certificate_valid:
                sidecar.close(wait=True)
                sidecar = None
                sidecar_activated = False
                selection_plan = [
                    (
                        [],
                        {
                            **metadata,
                            "selected": 0,
                            "selected_hits": 0,
                            "selected_probability_sum": 0.0,
                            "selected_positions": {},
                            "safe_start_budget": 0,
                            "authority_control_burst_gate_open": False,
                            "runtime_resource_certificate_valid": False,
                        },
                    )
                    for _, metadata in selection_plan
                ]
                selection_rows = [
                    metadata for _, metadata in selection_plan
                ]
                selection_selected_total = 0

        if dedicated_authority_thread:
            if authority_cpu_affinity is None:
                raise RuntimeError(
                    "dedicated authority lane has no CPU assignment"
                )
            authority_lane = DedicatedAuthorityLane(
                authority_executor,
                workers=workers,
                visit_capacity=visit_capacity,
                cpu_affinity=authority_cpu_affinity,
            )
            authority_lane.start()

        # Create runner-owned threads only after every fork-backed child.
        release_clock = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="authority-release-clock",
        )
        await loop.run_in_executor(release_clock, _timer_ready)
        if (
            pull_prestage_enabled
            and sidecar_activated
            and isinstance(sidecar, ProcessSpeculativeSidecar)
        ):
            # Use a raw concurrent Future: unlike ``run_in_executor``, its
            # completion does not schedule a callback on the parent loop.
            # The helper shares the speculative CPU, never the authority CPU.
            pull_prestage_clock = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="silent-pull-prefetch",
            )
            actual_pull_prestage_cpu_affinity = set(
                pull_prestage_clock.submit(
                    _pin_current_worker,
                    sidecar_cpu_affinity,
                ).result(timeout=5.0)
            )
    except BaseException:
        try:
            await cleanup_resources_cancellation_safe()
        except BaseException:
            pass
        raise
    if release_clock is None:
        raise RuntimeError("release clock setup did not complete")
    authoritative_tasks: list[
        tuple[str, asyncio.Task[AuthorityCompletion]]
    ] = []
    logical_rows: list[dict[str, Any]] = []
    requested = 0
    handles_returned = 0
    deadline_overruns = 0
    deadline_offsets_ms: list[float] = []
    admission_elapsed_ms: list[float] = []
    claim_elapsed_ms: list[float] = []
    claim_attempts = 0
    pull_prestage_elapsed_ms: list[float] = []
    pull_prestage_required_batches = 0
    pull_prestage_calls = 0
    pull_prestage_packets = 0
    pull_prestage_deadline_misses = 0
    pull_prestage_errors = 0
    pull_prestage_not_ready_at_post_arm_poll_batches = 0
    pull_prestage_worker_thread_ids: set[int] = set()
    pull_prestage_worker_cpu_affinities: set[tuple[int, ...]] = set()
    pull_prestage_realized_quiet_gap_ms: list[float] = []
    pull_runtime_latch_open = True
    pull_runtime_latch_trips = 0
    pull_runtime_latch_closed_batches = 0
    authority_backend_arm_violations = 0
    authority_backend_arm_suppressed_batches = 0
    claims_while_authority_unarmed = 0
    claims_while_prestage_unready = 0
    speculative_claim_suppressed_batches = 0
    retire_elapsed_ms: list[float] = []
    admission_selected_batches = 0
    retire_batches = 0
    wall_started = time.perf_counter()
    logical_done_at = wall_started
    bridge_started_before_authority_done = False
    shadow_barrier_violations = 0

    try:
        for batch_index, batch in enumerate(batches):
            if shadow_barrier and any(
                not task.done() for _, task in authoritative_tasks
            ):
                shadow_barrier_violations += 1
            decision_started = time.perf_counter()
            confirmation_deadline = time.monotonic() + lead_s
            planned_confirmation_at = decision_started + lead_s
            # ``start_deadline`` is the latest safe *executor start*, not the
            # authority confirmation.  With a certified service upper bound,
            # work admitted before this fence has a full service interval and
            # guard in which to finish without overlapping demand execution.
            start_deadline = confirmation_deadline
            if require_precompletion:
                start_deadline -= service_s + completion_guard_s
            handle_by_target: dict[
                tuple[str, str, tuple[str, str]], SpeculativeHandle
            ] = {}
            submitted_handles: list[SpeculativeHandle] = []
            runtime_session_ids = {
                (window.session_id, window.decision_id): (
                    f"r{seed}:{window.session_id}:{window.decision_id}"
                )
                for window in batch
            }

            if (
                sidecar is not None
                and selection_plan[batch_index][0]
                and not pull_runtime_latch_open
            ):
                pull_runtime_latch_closed_batches += 1

            if (
                sidecar is not None
                and selection_plan[batch_index][0]
                and pull_runtime_latch_open
            ):
                selected, _ = selection_plan[batch_index]
                requested += len(selected)
                admission_started = time.perf_counter()
                submit_rows: list[
                    tuple[
                        Invocation,
                        str,
                        str,
                        float,
                    ]
                ] = []
                for candidate, priority in selected:
                    pattern = candidate.pattern
                    invocation = Invocation("visit", {"url": pattern.url})
                    runtime_session_id = runtime_session_ids[
                        (pattern.session_id, pattern.decision_id)
                    ]
                    submit_rows.append(
                        (
                            invocation,
                            runtime_session_id,
                            pattern.decision_id,
                            float(priority),
                        )
                    )

                admitted_rows: list[
                    tuple[
                        tuple[Invocation, str, str, float],
                        SpeculativeHandle,
                    ]
                ] = []
                if submit_rows and isinstance(
                    sidecar, ProcessSpeculativeSidecar
                ):
                    returned_handles = sidecar.try_submit_batch(
                        tuple(
                            (
                                invocation,
                                runtime_session_id,
                                decision_id,
                                priority,
                                "",
                            )
                            for (
                                invocation,
                                runtime_session_id,
                                decision_id,
                                priority,
                            ) in submit_rows
                        ),
                        start_deadline=start_deadline,
                    )
                    admitted_rows.extend(zip(submit_rows, returned_handles))
                elif submit_rows:
                    admitted_rows = []
                    for (
                        invocation,
                        runtime_session_id,
                        decision_id,
                        priority,
                    ) in submit_rows:
                        handle = sidecar.try_submit(
                            invocation,
                            session_id=runtime_session_id,
                            decision_id=decision_id,
                            priority=priority,
                            start_deadline=start_deadline,
                        )
                        if handle is not None:
                            admitted_rows.append(
                                (
                                    (
                                        invocation,
                                        runtime_session_id,
                                        decision_id,
                                        priority,
                                    ),
                                    handle,
                                )
                            )
                if submit_rows:
                    admission_selected_batches += 1
                    admission_elapsed_ms.append(
                        (time.perf_counter() - admission_started) * 1000.0
                    )

                for submit_row, handle in admitted_rows:
                    invocation, runtime_session_id, decision_id, _ = submit_row
                    handles_returned += 1
                    submitted_handles.append(handle)
                    handle_by_target[
                        (
                            runtime_session_id,
                            decision_id,
                            invocation.key,
                        )
                    ] = handle

            if time.monotonic() >= start_deadline and sidecar is not None:
                deadline_overruns += 1
            prestage_claim_ready = True
            pull_prestage_future: (
                ConcurrentFuture[_PullPrestageRelease] | None
            ) = None
            if (
                pull_prestage_enabled
                and submitted_handles
                and isinstance(sidecar, ProcessSpeculativeSidecar)
            ):
                pull_prestage_required_batches += 1
                completion_cutoff = (
                    confirmation_deadline - completion_guard_s
                )
                # Reserve the second half of the completion guard as a true
                # quiet interval.  A late worker can reduce coverage, but it
                # is never allowed to gate the authority release clock.
                prefetch_deadline = (
                    confirmation_deadline - completion_guard_s / 2.0
                )
                if pull_prestage_clock is None:
                    raise RuntimeError("pull prestage clock is unavailable")
                pull_prestage_future = pull_prestage_clock.submit(
                    _prefetch_pull_epoch,
                    sidecar,
                    completion_cutoff,
                    prefetch_deadline,
                )
            confirmation_release = loop.run_in_executor(
                release_clock,
                _sleep_until,
                confirmation_deadline,
            )
            active_confirmation_release = confirmation_release
            await asyncio.shield(confirmation_release)
            active_confirmation_release = None
            confirmation_at = time.perf_counter()
            deadline_offsets_ms.append(
                (confirmation_at - decision_started) * 1000.0
            )

            call_rows: list[
                tuple[
                    str,
                    Invocation,
                    SpeculativeHandle | None,
                    asyncio.Task[AuthorityCompletion],
                    ConcurrentFuture[AuthorityCompletion] | None,
                    float,
                    tuple[str, str, tuple[str, str]],
                ]
            ] = []
            authority_executor_started_before_batch = (
                authority_executor_started_count
            )
            for window in batch:
                runtime_session_id = runtime_session_ids[
                    (window.session_id, window.decision_id)
                ]
                for target_index, target in enumerate(window.executable_targets):
                    target_id = (
                        f"{window.session_id}:{window.decision_id}:"
                        f"target:{target_index}"
                    )
                    invocation = Invocation("visit", {"url": target})
                    scheduled_at = time.perf_counter()
                    raw_authority: ConcurrentFuture[AuthorityCompletion] | None
                    if authority_lane is not None:
                        raw_authority = authority_lane.submit(
                            invocation,
                            session_id=runtime_session_id,
                            scheduled_at=scheduled_at,
                        )
                        authority_task = asyncio.create_task(
                            _observe_remote_authority(raw_authority)
                        )
                    else:
                        raw_authority = None
                        if authority is None:
                            raise RuntimeError("authority broker is unavailable")
                        authority_task = asyncio.create_task(
                            _authority_call(
                                authority,
                                invocation,
                                session_id=runtime_session_id,
                                scheduled_at=scheduled_at,
                            )
                        )
                    authoritative_tasks.append((target_id, authority_task))
                    call_rows.append(
                        (
                            target_id,
                            invocation,
                            None,
                            authority_task,
                            raw_authority,
                            scheduled_at,
                            (
                                runtime_session_id,
                                window.decision_id,
                                invocation.key,
                            ),
                        )
                    )

            # Direct/thread authority needs one first scheduling turn before
            # inspecting a sidecar result.  A process authority request is
            # already irrevocably committed once its ordered Pipe send
            # returns; waiting for a child round-trip here only adds one
            # serialized barrier per batch and cannot improve isolation.
            claimed_keys = set()
            authority_batch_armed = True
            if not call_rows and pull_prestage_future is not None:
                # A false-positive decision may have no eventual authority
                # target. It still owns a raw prefetch job, which must be
                # observed before the next batch admits more sidecar work.
                release_outcome, prefetch_done = _poll_pull_prefetch(
                    pull_prestage_future
                )
                prestage_claim_ready = bool(
                    prefetch_done
                    and release_outcome is not None
                    and not release_outcome.deadline_miss
                )
                if not prestage_claim_ready:
                    if pull_runtime_latch_open:
                        pull_runtime_latch_trips += 1
                    pull_runtime_latch_open = False
                pull_prestage_observations.append(
                    (
                        pull_prestage_future,
                        release_outcome,
                        prefetch_done,
                    )
                )
            if call_rows:
                if isinstance(authority_lane, DedicatedAuthorityLane):
                    await authority_lane.barrier()
                elif authority_lane is None:
                    # Turn 1 admits every authority job and schedules its
                    # broker runner. Turn 2 lets that runner enter the tool
                    # executor and arm the backend/service await. Only then
                    # may treatment-specific exact lookup or claim work run.
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    if (
                        submitted_handles
                        and authority_executor_started_count
                        - authority_executor_started_before_batch
                        < len(call_rows)
                    ):
                        authority_backend_arm_violations += 1
                        authority_backend_arm_suppressed_batches += 1
                        authority_batch_armed = False
                # Only after demand authority has irrevocably entered its
                # backend/service await may treatment inspect the raw Future.
                # Future condition-lock contention can therefore never delay
                # authority submission or backend arm.
                if pull_prestage_future is not None:
                    release_outcome, prefetch_done = _poll_pull_prefetch(
                        pull_prestage_future
                    )
                    prestage_claim_ready = bool(
                        prefetch_done
                        and release_outcome is not None
                        and not release_outcome.deadline_miss
                    )
                    if not prestage_claim_ready:
                        if pull_runtime_latch_open:
                            pull_runtime_latch_trips += 1
                        pull_runtime_latch_open = False
                    pull_prestage_observations.append(
                        (
                            pull_prestage_future,
                            release_outcome,
                            prefetch_done,
                        )
                    )
                claim_epoch_open = (
                    prestage_claim_ready and authority_batch_armed
                )
                if submitted_handles and not claim_epoch_open:
                    speculative_claim_suppressed_batches += 1
                resolved_rows = [
                    (
                        target_id,
                        invocation,
                        (
                            handle_by_target.pop(lookup_key, None)
                            if claim_epoch_open
                            else None
                        ),
                        authority_task,
                        raw_authority,
                        scheduled_at,
                    )
                    for (
                        target_id,
                        invocation,
                        _,
                        authority_task,
                        raw_authority,
                        scheduled_at,
                        lookup_key,
                    ) in call_rows
                ]
                claimed_rows = []
                batch_claim_attempts = sum(
                    handle is not None
                    for _, _, handle, _, _, _ in resolved_rows
                )
                if not authority_batch_armed:
                    claims_while_authority_unarmed += batch_claim_attempts
                if not prestage_claim_ready:
                    claims_while_prestage_unready += batch_claim_attempts
                claim_started = (
                    time.perf_counter()
                    if batch_claim_attempts
                    else None
                )
                for row in resolved_rows:
                    (
                        target_id,
                        invocation,
                        handle,
                        authority_task,
                        raw_authority,
                        scheduled_at,
                    ) = row
                    claimed = (
                        sidecar.try_claim(handle.key)
                        if sidecar is not None and handle is not None
                        else None
                    )
                    if claimed is not None:
                        claimed_keys.add(claimed.key)
                    claimed_rows.append(
                        (
                            target_id,
                            invocation,
                            claimed,
                            authority_task,
                            raw_authority,
                            scheduled_at,
                        )
                    )
                if claim_started is not None:
                    claim_elapsed_ms.append(
                        (time.perf_counter() - claim_started) * 1000.0
                    )
                    claim_attempts += batch_claim_attempts
                # The process backend has finite leases and therefore emits
                # no cleanup command on the authority-miss path.  A thread
                # ablation has no child-local lease collector, but strict
                # shadow mode defers its Python cleanup until this batch's
                # protected authority interval has drained below.
                if (
                    sidecar is not None
                    and not isinstance(sidecar, ProcessSpeculativeSidecar)
                    and not shadow_barrier
                ):
                    tombstones = tuple(
                        (handle.key.session_id, handle.key.decision_id)
                        for handle in submitted_handles
                        if handle.key not in claimed_keys
                    )
                    if tombstones:
                        retire_started = time.perf_counter()
                        for session_id, decision_id in tombstones:
                            sidecar.try_tombstone(
                                session_id=session_id,
                                decision_id=decision_id,
                            )
                        retire_elapsed_ms.append(
                            (time.perf_counter() - retire_started) * 1000.0
                        )
                        retire_batches += 1
                logical_futures = [
                    _arm_shadow_authority_race(
                        authority_task,
                        handle,
                        invocation=invocation,
                        loop=loop,
                        raw_authority=raw_authority,
                        speculative_terminal_cutoff=(
                            planned_confirmation_at - completion_guard_s
                            if require_precompletion
                            else None
                        ),
                    )
                    for (
                        _,
                        invocation,
                        handle,
                        authority_task,
                        raw_authority,
                        _,
                    ) in claimed_rows
                ]
                completions = await asyncio.gather(*logical_futures)
                for row, completion in zip(claimed_rows, completions):
                    target_id, _, _, _, _, scheduled_at = row
                    logical_rows.append(
                        {
                            "target_id": target_id,
                            "source": completion.source,
                            "scheduled_latency_ms": (
                                completion.terminal_at - scheduled_at
                            )
                            * 1000.0,
                            "speculative_executor_terminal_at": (
                                completion.speculative_executor_terminal_at
                            ),
                            "speculative_terminal_cutoff": (
                                planned_confirmation_at - completion_guard_s
                                if require_precompletion
                                else None
                            ),
                        }
                    )
                if shadow_barrier and batch_index + 1 < len(batches):
                    # A speculative hit may unblock useful model work, but it
                    # must not let a later real tool request overlap an
                    # unresolved shadow-authority backup on the protected
                    # broker.  This is the minimal closed-loop no-harm guard.
                    await asyncio.gather(*(row[3] for row in call_rows))
                    if sidecar is not None and not isinstance(
                        sidecar, ProcessSpeculativeSidecar
                    ):
                        tombstones = tuple(
                            (handle.key.session_id, handle.key.decision_id)
                            for handle in submitted_handles
                            if handle.key not in claimed_keys
                        )
                        if tombstones:
                            retire_started = time.perf_counter()
                            for session_id, decision_id in tombstones:
                                sidecar.try_tombstone(
                                    session_id=session_id,
                                    decision_id=decision_id,
                                )
                            retire_elapsed_ms.append(
                                (time.perf_counter() - retire_started)
                                * 1000.0
                            )
                            retire_batches += 1
            elif sidecar is not None and not isinstance(
                sidecar, ProcessSpeculativeSidecar
            ):
                tombstones = tuple(
                    (handle.key.session_id, handle.key.decision_id)
                    for handle in submitted_handles
                )
                if tombstones:
                    retire_started = time.perf_counter()
                    for session_id, decision_id in tombstones:
                        sidecar.try_tombstone(
                            session_id=session_id,
                            decision_id=decision_id,
                        )
                    retire_elapsed_ms.append(
                        (time.perf_counter() - retire_started) * 1000.0
                    )
                    retire_batches += 1
            logical_done_at = time.perf_counter()

        logical_done_at = time.perf_counter()
        authority_completions = (
            await asyncio.gather(
                *(task for _, task in authoritative_tasks)
            )
            if authoritative_tasks
            else []
        )
        authority_done_at = time.perf_counter()
        bridge_started_before_authority_done = (
            sidecar.bridge_started
            if isinstance(sidecar, ProcessSpeculativeSidecar)
            else False
        )
        if authority_lane is not None:
            authority_lane_snapshot = await authority_lane.aclose()
            authority_snapshot = dict(
                authority_lane_snapshot["snapshot"]
            )
            authority_stats = dict(authority_lane_snapshot["stats"])
            authority_state_count = int(
                authority_lane_snapshot["authoritative_state_count"]
            )
        else:
            if authority is None:
                raise RuntimeError("authority broker is unavailable")
            await authority.close()
            authority_snapshot = authority.snapshot()
            authority_stats = authority.stats.to_dict()
            authority_state_count = len(authority.authoritative_state)
            authority_lane_snapshot = {}
        # A raw Future intentionally has no loop callback, so normal teardown
        # must explicitly join it before snapshot starts a lifecycle bridge or
        # close touches the same result socket.
        if pull_prestage_clock is not None:
            pull_prestage_clock.shutdown(
                wait=True,
                cancel_futures=True,
            )
            pull_prestage_clock = None
        sidecar_before_close = sidecar.snapshot() if sidecar is not None else {}
        if sidecar is not None:
            sidecar.close(wait=True)
        sidecar_after_close = sidecar.snapshot() if sidecar is not None else {}
    finally:
        primary_error = sys.exc_info()[1]
        try:
            await cleanup_resources_cancellation_safe()
        except BaseException:
            if primary_error is None:
                raise
    drained_at = time.perf_counter()

    # Collect silent raw-Future telemetry only after the protected authority
    # interval and executor join.  None of these bookkeeping operations can
    # perturb authority admission, first-run ordering, or service timers.
    for future, outcome_at_post_arm_poll, done_at_post_arm_poll in (
        pull_prestage_observations
    ):
        if not done_at_post_arm_poll:
            pull_prestage_not_ready_at_post_arm_poll_batches += 1
        outcome = outcome_at_post_arm_poll
        if outcome is None:
            try:
                outcome = future.result(timeout=0.0)
            except BaseException:
                pull_prestage_errors += 1
                pull_prestage_deadline_misses += 1
                continue
        pull_prestage_calls += 1
        pull_prestage_elapsed_ms.append(outcome.elapsed_ms)
        pull_prestage_packets += outcome.packets
        pull_prestage_worker_thread_ids.add(outcome.worker_thread_id)
        pull_prestage_worker_cpu_affinities.add(
            outcome.worker_cpu_affinity
        )
        pull_prestage_realized_quiet_gap_ms.append(
            (
                outcome.prefetch_deadline
                + completion_guard_s / 2.0
                - outcome.finished_at
            )
            * 1000.0
        )
        if outcome.error is not None:
            pull_prestage_errors += 1
        if outcome.deadline_miss or not done_at_post_arm_poll:
            pull_prestage_deadline_misses += 1

    authority_rows: list[dict[str, Any]] = []
    for (target_id, _), completion in zip(
        authoritative_tasks, authority_completions
    ):
        authority_rows.append(
            {
                "target_id": target_id,
                "scheduled_latency_ms": (
                    completion.terminal_at - completion.scheduled_at
                )
                * 1000.0,
                "observed_latency_ms": (
                    completion.observed_at - completion.scheduled_at
                )
                * 1000.0,
                "return_handoff_ms": (
                    completion.observed_at - completion.terminal_at
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
            }
        )

    target_count = len(authority_rows)
    logical_by_id = {row["target_id"]: row for row in logical_rows}
    if set(logical_by_id) != {row["target_id"] for row in authority_rows}:
        raise RuntimeError("logical and authoritative target sets differ")
    sidecar_snapshot = _snapshot_json(sidecar_after_close or sidecar_before_close)
    actual_sidecar_affinity = set(
        sidecar_snapshot.get("actual_cpu_affinity") or []
    )
    actual_bridge_affinity = set(
        sidecar_snapshot.get("actual_bridge_cpu_affinity") or []
    )
    actual_authority_affinity = set(
        authority_lane_snapshot.get("actual_cpu_affinity") or []
    )
    actual_authority_bridge_affinity = set(
        authority_lane_snapshot.get("bridge_actual_cpu_affinity") or []
    )
    pull_mailbox_no_timed_bridge_certified = (
        not pull_result_staging
        or sidecar_slots == 0
        or not sidecar_activated
        or not bridge_started_before_authority_done
    )
    pull_prestage_before_authority_certified = (
        not pull_prestage_enabled
        or not sidecar_activated
        or (
            pull_prestage_calls == pull_prestage_required_batches
            and pull_prestage_deadline_misses == 0
            and pull_prestage_errors == 0
            and pull_prestage_not_ready_at_post_arm_poll_batches == 0
        )
    )
    pull_prestage_off_parent_loop_certified = (
        not pull_prestage_enabled
        or not sidecar_activated
        or pull_prestage_required_batches == 0
        or (
            pull_prestage_calls == pull_prestage_required_batches
            and bool(pull_prestage_worker_thread_ids)
            and parent_loop_thread_id not in pull_prestage_worker_thread_ids
        )
    )
    pull_prestage_cpu_affinity_certified = (
        not pull_prestage_enabled
        or not sidecar_activated
        or not cpu_isolation
        or (
            sidecar_cpu_affinity is not None
            and actual_pull_prestage_cpu_affinity
            == sidecar_cpu_affinity
            and all(
                set(affinity) == sidecar_cpu_affinity
                for affinity in pull_prestage_worker_cpu_affinities
            )
        )
    )
    pull_prestage_quiet_gap_certified = (
        not pull_prestage_enabled
        or not sidecar_activated
        or pull_prestage_required_batches == 0
        or (
            pull_prestage_before_authority_certified
            and len(pull_prestage_realized_quiet_gap_ms)
            == pull_prestage_required_batches
            and min(pull_prestage_realized_quiet_gap_ms)
            >= completion_guard_ms / 2.0 - 1e-6
        )
    )
    bridge_cpu_affinity_certified = (
        not cpu_isolation
        or sidecar_slots == 0
        or not sidecar_activated
        or (
            pull_result_staging
            and pull_mailbox_no_timed_bridge_certified
        )
        or (
            sidecar_backend == "process"
            and sidecar_cpu_affinity is not None
            and bool(sidecar_snapshot.get("bridge_affinity_ready", False))
            and sidecar_snapshot.get("bridge_affinity_error") is None
            and actual_bridge_affinity == sidecar_cpu_affinity
        )
    )
    cpu_isolation_certified = (
        not cpu_isolation
        or sidecar_slots == 0
        or not sidecar_activated
        or (
            sidecar_backend == "process"
            and authority_cpu_affinity is not None
            and sidecar_cpu_affinity is not None
            and authority_cpu_affinity.isdisjoint(sidecar_cpu_affinity)
            and actual_sidecar_affinity == sidecar_cpu_affinity
            and pull_prestage_cpu_affinity_certified
            and bridge_cpu_affinity_certified
        )
    )
    three_lane_logical_cpu_isolation_certified = (
        not dedicated_authority
        or not cpu_isolation
        or sidecar_slots == 0
        or not sidecar_activated
        or (
            authority_cpu_affinity is not None
            and control_cpu_affinity is not None
            and sidecar_cpu_affinity is not None
            and actual_authority_affinity == authority_cpu_affinity
            and (
                not dedicated_authority_process
                or actual_authority_bridge_affinity
                == control_cpu_affinity
            )
            and actual_control_cpu_affinity == control_cpu_affinity
            and actual_sidecar_affinity == sidecar_cpu_affinity
            and (
                pull_result_staging
                or actual_bridge_affinity == sidecar_cpu_affinity
            )
            and authority_cpu_affinity.isdisjoint(control_cpu_affinity)
            and authority_cpu_affinity.isdisjoint(sidecar_cpu_affinity)
            and control_cpu_affinity.isdisjoint(sidecar_cpu_affinity)
        )
    )
    sidecar_idle_priority_certified = (
        sidecar_backend != "process"
        or sidecar_slots == 0
        or not sidecar_activated
        or sidecar_snapshot.get("actual_scheduler_policy")
        == getattr(os, "SCHED_IDLE", 5)
    )
    sidecar_started = _sidecar_count(
        sidecar_after_close or sidecar_before_close,
        "started",
        "jobs_started",
        "speculative_started",
    )
    sidecar_max_running = _sidecar_count(
        sidecar_after_close or sidecar_before_close,
        "max_running",
        "max_running_total",
    )
    visible_hits = sum(row["source"] == "sidecar" for row in logical_rows)
    authority_commit_count = int(authority_stats["commits"])
    process_transport = (
        sidecar_snapshot.get("transport", {})
        if isinstance(sidecar_snapshot.get("transport", {}), Mapping)
        else {}
    )
    safety = {
        "authority_attempts_equal_targets": (
            int(authority_stats["authoritative_requests"]) == target_count
        ),
        "authority_commits_equal_targets": authority_commit_count == target_count,
        "authority_state_equal_targets": (
            authority_state_count == target_count
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
        "sidecar_cap": sidecar_max_running <= sidecar_slots,
        "wall_order": (
            logical_done_at <= authority_done_at <= drained_at
        ),
        "all_authority_sources_executed": all(
            completion.result.source == "executed"
            for completion in authority_completions
        ),
        "cpu_isolation_certificate": cpu_isolation_certified,
        "three_lane_logical_cpu_isolation_certificate": (
            three_lane_logical_cpu_isolation_certified
        ),
        "result_bridge_cpu_affinity_certificate": (
            bridge_cpu_affinity_certified
        ),
        "pull_mailbox_no_timed_bridge_certificate": (
            pull_mailbox_no_timed_bridge_certified
        ),
        "pull_prestage_timely_or_claim_suppressed": (
            pull_prestage_before_authority_certified
            or claims_while_prestage_unready == 0
        ),
        "pull_prestage_worker_off_parent_loop": (
            parent_loop_thread_id not in pull_prestage_worker_thread_ids
        ),
        "pull_prestage_worker_cpu_affinity_certificate": (
            pull_prestage_cpu_affinity_certified
        ),
        "sidecar_idle_priority_certificate": (
            sidecar_idle_priority_certified
        ),
        "process_miss_cleanup_has_no_parent_packets": (
            sidecar_backend != "process"
            or int(process_transport.get("transport_tombstone_packets", 0))
            == 0
        ),
        "result_bridge_started_off_timed_path": (
            sidecar_backend != "process"
            or not sidecar_activated
            or (
                pull_result_staging
                and not bridge_started_before_authority_done
            )
            or result_bridge_prestarted
        ),
        "strict_shadow_barrier_no_prior_backup": (
            not shadow_barrier or shadow_barrier_violations == 0
        ),
        "authority_control_burst_gate_enforced": all(
            not selected
            or bool(metadata["authority_control_burst_gate_open"])
            for selected, metadata in selection_plan
        ),
        "visible_hits_satisfy_precompletion_gate": (
            not require_precompletion
            or all(
                row["source"] != "sidecar"
                or (
                    row["speculative_executor_terminal_at"] is not None
                    and row["speculative_terminal_cutoff"] is not None
                    and row["speculative_executor_terminal_at"]
                    <= row["speculative_terminal_cutoff"]
                )
                for row in logical_rows
            )
        ),
        "authority_backend_armed_before_spec_claim": (
            not pull_prestage_enabled
            or not sidecar_activated
            or claims_while_authority_unarmed == 0
        ),
        "prestage_ready_before_spec_claim": (
            not pull_prestage_enabled
            or not sidecar_activated
            or claims_while_prestage_unready == 0
        ),
        "process_runtime_certificate_checked_before_timed_path": (
            sidecar_backend != "process"
            or not sidecar_activated
            or (
                sidecar_runtime_certificate_checked
                and sidecar_runtime_certificate_valid
            )
        ),
    }
    if not all(safety.values()):
        raise RuntimeError(f"sidecar safety failed: {safety}")

    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "total": sum(values),
            "mean": statistics.fmean(values) if values else 0.0,
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=0.0),
        }

    logical_latencies = [
        float(row["scheduled_latency_ms"]) for row in logical_rows
    ]
    authority_latencies = [
        float(row["scheduled_latency_ms"]) for row in authority_rows
    ]
    authority_observed_latencies = [
        float(row["observed_latency_ms"]) for row in authority_rows
    ]
    authority_return_handoffs = [
        float(row["return_handoff_ms"]) for row in authority_rows
    ]
    first_run_lags = [float(row["first_run_lag_ms"]) for row in authority_rows]
    broker_waits = [
        float(row["broker_exposed_wait_ms"]) for row in authority_rows
    ]
    source_counts = Counter(str(row["source"]) for row in logical_rows)
    return {
        "seed": seed,
        "offered_concurrency": offered_concurrency,
        "sidecar_slots": sidecar_slots,
        "sidecar_activated": sidecar_activated,
        "result_bridge_prestarted": result_bridge_prestarted,
        "result_bridge_cpu_affinity_certified": (
            bridge_cpu_affinity_certified
        ),
        "pull_mailbox_no_timed_bridge_certified": (
            pull_mailbox_no_timed_bridge_certified
        ),
        "sidecar_backend": sidecar_backend,
        "cpu_isolation_requested": cpu_isolation,
        "cpu_isolation_certified": cpu_isolation_certified,
        "three_lane_logical_cpu_isolation_certified": (
            three_lane_logical_cpu_isolation_certified
        ),
        "dedicated_authority_thread": dedicated_authority_thread,
        "dedicated_authority_process": dedicated_authority_process,
        "original_authority_path_preserved": not dedicated_authority,
        "sidecar_idle_priority_certified": (
            sidecar_idle_priority_certified
        ),
        "authority_cpu_affinity": sorted(authority_cpu_affinity or []),
        "actual_authority_cpu_affinity": sorted(actual_authority_affinity),
        "actual_authority_bridge_cpu_affinity": sorted(
            actual_authority_bridge_affinity
        ),
        "control_cpu_affinity": sorted(control_cpu_affinity or []),
        "actual_control_cpu_affinity": sorted(
            actual_control_cpu_affinity
        ),
        "sidecar_cpu_affinity": sorted(sidecar_cpu_affinity or []),
        "actual_pull_prestage_cpu_affinity": sorted(
            actual_pull_prestage_cpu_affinity
        ),
        "probability_threshold": probability_threshold,
        "require_precompletion": require_precompletion,
        "predicted_precompletion": predicted_precompletion,
        "completion_guard_ms": float(completion_guard_ms),
        "eager_result_staging": eager_result_staging,
        "pull_result_staging": pull_result_staging,
        "pull_prestage_enabled": pull_prestage_enabled,
        "pull_prestage_off_parent_loop_certified": (
            pull_prestage_off_parent_loop_certified
        ),
        "pull_prestage_before_authority_certified": (
            pull_prestage_before_authority_certified
        ),
        "pull_prestage_cpu_affinity_certified": (
            pull_prestage_cpu_affinity_certified
        ),
        "pull_prestage_quiet_gap_certified": (
            pull_prestage_quiet_gap_certified
        ),
        "pull_prestage_never_gates_authority_release": True,
        "authority_backend_arm_violations": (
            authority_backend_arm_violations
        ),
        "authority_backend_arm_suppressed_batches": (
            authority_backend_arm_suppressed_batches
        ),
        "claims_while_authority_unarmed": (
            claims_while_authority_unarmed
        ),
        "claims_while_prestage_unready": (
            claims_while_prestage_unready
        ),
        "speculative_claim_suppressed_batches": (
            speculative_claim_suppressed_batches
        ),
        "pull_runtime_latch_trips": pull_runtime_latch_trips,
        "pull_runtime_latch_closed_batches": (
            pull_runtime_latch_closed_batches
        ),
        "pull_max_staged_result_bytes": (
            STRICT_PULL_MAX_STAGED_RESULT_BYTES
            if pull_result_staging
            else None
        ),
        "coordination_cost_ms": float(coordination_cost_ms),
        "strict_static_resource_certificate": (
            strict_static_resource_certificate
        ),
        "physical_core_placement_certified": (
            physical_core_placement_certified
        ),
        "strict_positive_budget_certificate": (
            strict_static_resource_certificate
            and sidecar_runtime_certificate_checked
            and sidecar_runtime_certificate_valid
            and pull_prestage_before_authority_certified
            and pull_prestage_off_parent_loop_certified
            and pull_prestage_cpu_affinity_certified
            and pull_prestage_quiet_gap_certified
            and authority_backend_arm_violations == 0
        ),
        "certified_exclusive_resources": certified_exclusive_resources,
        "unsafe_positive_ablation": unsafe_positive_ablation,
        "sidecar_runtime_certificate_checked": (
            sidecar_runtime_certificate_checked
        ),
        "sidecar_runtime_certificate_valid": (
            sidecar_runtime_certificate_valid
        ),
        "sidecar_runtime_certificate_error": (
            sidecar_runtime_certificate_error
        ),
        "sidecar_startup_snapshot": _snapshot_json(
            sidecar_startup_snapshot
        ),
        "shadow_barrier": shadow_barrier,
        "shadow_barrier_violations": shadow_barrier_violations,
        "authoritative_targets": target_count,
        "requested_predictions": requested,
        "handles_returned": handles_returned,
        "selection_selected": sum(
            int(row["selected"]) for row in selection_rows
        ),
        "authority_control_burst_limit": authority_control_burst_limit,
        "authority_control_burst_limit_exceeded_batches": sum(
            bool(row["authority_control_burst_limit_exceeded"])
            for row in selection_rows
        ),
        "authority_control_burst_latch_closed_batches": sum(
            not bool(row["authority_control_burst_gate_open"])
            for row in selection_rows
        ),
        "selection_selected_hits": sum(
            int(row["selected_hits"]) for row in selection_rows
        ),
        "selection_probability_sum": sum(
            float(row["selected_probability_sum"])
            for row in selection_rows
        ),
        "selection_expected_gross_benefit_ms": (
            sum(
                float(row["selected_probability_sum"])
                for row in selection_rows
            )
            * min(service_ms, lead_ms)
        ),
        "selection_expected_net_benefit_ms": (
            sum(
                float(row["selected_probability_sum"])
                for row in selection_rows
            )
            * min(service_ms, lead_ms)
            - sum(int(row["selected"]) for row in selection_rows)
            * float(coordination_cost_ms)
        ),
        "selection_compute_ms": sum(
            float(row["compute_ms"]) for row in selection_rows
        ),
        "predictor_windows_evaluated": sum(
            int(row["predictor_windows_evaluated"])
            for row in selection_rows
        ),
        "probability_candidates_evaluated": sum(
            int(row["probability_candidates_evaluated"])
            for row in selection_rows
        ),
        "visible_speculative_hits": visible_hits,
        "source_counts": dict(sorted(source_counts.items())),
        "sidecar_started": sidecar_started,
        "physical_calls_started": target_count + sidecar_started,
        "physical_call_amplification": ratio(
            target_count + sidecar_started, target_count
        ),
        "logical_latency_ms": summary(logical_latencies),
        "authority_scheduled_latency_ms": summary(authority_latencies),
        "authority_observed_latency_ms": summary(
            authority_observed_latencies
        ),
        "authority_return_handoff_ms": summary(authority_return_handoffs),
        "authority_first_run_lag_ms": summary(first_run_lags),
        "authority_broker_exposed_wait_ms": summary(broker_waits),
        "logical_rows": logical_rows,
        "authority_rows": authority_rows,
        "logical_done_wall_s": logical_done_at - wall_started,
        "authority_done_wall_s": authority_done_at - wall_started,
        "drained_wall_s": drained_at - wall_started,
        "sidecar_drain_tail_s": drained_at - authority_done_at,
        "confirmation_offset_ms": {
            "mean": statistics.fmean(deadline_offsets_ms),
            "p95": percentile(deadline_offsets_ms, 0.95),
            "max": max(deadline_offsets_ms, default=0.0),
            "mean_lateness": statistics.fmean(
                [max(0.0, value - lead_ms) for value in deadline_offsets_ms]
            ),
            "p95_lateness": percentile(
                [max(0.0, value - lead_ms) for value in deadline_offsets_ms],
                0.95,
            ),
            "deadline_overrun_batches": deadline_overruns,
            "batches": len(deadline_offsets_ms),
        },
        "sidecar_hot_path_ms": {
            "admission_total": sum(admission_elapsed_ms),
            "admission_mean": (
                statistics.fmean(admission_elapsed_ms)
                if admission_elapsed_ms
                else 0.0
            ),
            "admission_p95": percentile(admission_elapsed_ms, 0.95),
            "admission_selected_batches": admission_selected_batches,
            "claim_total": sum(claim_elapsed_ms),
            "claim_mean": (
                statistics.fmean(claim_elapsed_ms)
                if claim_elapsed_ms
                else 0.0
            ),
            "claim_p95": percentile(claim_elapsed_ms, 0.95),
            "claim_batches": len(claim_elapsed_ms),
            "claim_attempts": claim_attempts,
            "pull_prestage_total": sum(pull_prestage_elapsed_ms),
            "pull_prestage_mean": (
                statistics.fmean(pull_prestage_elapsed_ms)
                if pull_prestage_elapsed_ms
                else 0.0
            ),
            "pull_prestage_p95": percentile(
                pull_prestage_elapsed_ms, 0.95
            ),
            "pull_prestage_required_batches": (
                pull_prestage_required_batches
            ),
            "pull_prestage_calls": pull_prestage_calls,
            "pull_prestage_packets": pull_prestage_packets,
            "pull_prestage_deadline_misses": (
                pull_prestage_deadline_misses
            ),
            "pull_prestage_errors": pull_prestage_errors,
            "pull_prestage_not_ready_at_post_arm_poll_batches": (
                pull_prestage_not_ready_at_post_arm_poll_batches
            ),
            "pull_prestage_planned_quiet_gap_ms": (
                completion_guard_ms / 2.0
                if pull_prestage_enabled
                else 0.0
            ),
            "pull_prestage_realized_quiet_gap_min": min(
                pull_prestage_realized_quiet_gap_ms,
                default=0.0,
            ),
            "pull_prestage_realized_quiet_gap_p95": percentile(
                pull_prestage_realized_quiet_gap_ms,
                0.95,
            ),
            "pull_prestage_worker_threads": len(
                pull_prestage_worker_thread_ids
            ),
            "pull_prestage_off_parent_loop": (
                pull_prestage_required_batches == 0
                or (
                    bool(pull_prestage_worker_thread_ids)
                    and parent_loop_thread_id
                    not in pull_prestage_worker_thread_ids
                )
            ),
            "retire_total": sum(retire_elapsed_ms),
            "retire_mean": (
                statistics.fmean(retire_elapsed_ms)
                if retire_elapsed_ms
                else 0.0
            ),
            "retire_p95": percentile(retire_elapsed_ms, 0.95),
            "retire_batches": retire_batches,
        },
        "authority_stats": authority_stats,
        "authority_lane_snapshot": authority_lane_snapshot,
        "sidecar_snapshot": sidecar_snapshot,
        "bridge_started_before_authority_done": (
            bridge_started_before_authority_done
        ),
        "safety": safety,
    }


async def _run_sample(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run one sample while unconditionally restoring caller CPU affinity."""

    original_affinity = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set()
    )
    try:
        return await _run_sample_impl(*args, **kwargs)
    finally:
        if original_affinity and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, original_affinity)


def _target_map(
    sample: Mapping[str, Any],
    field: str,
    *,
    metric: str = "scheduled_latency_ms",
) -> dict[str, float]:
    return {
        str(row["target_id"]): float(row[metric])
        for row in sample[field]
    }


def _aggregate_cell(
    *,
    scenario: str,
    concurrency: int,
    baseline_samples: Sequence[Mapping[str, Any]],
    sidecar_samples: Sequence[Mapping[str, Any]],
    service_ms: float,
    lead_ms: float,
    sidecar_slots: int,
    probability_threshold: float,
    coordination_cost_ms: float,
    sidecar_backend: str,
) -> dict[str, Any]:
    if len(baseline_samples) != len(sidecar_samples):
        raise ValueError("paired sample counts differ")
    target_benefits: list[float] = []
    authority_regressions: list[float] = []
    authority_observed_regressions: list[float] = []
    repeat_target_benefits: list[float] = []
    repeat_authority_regressions: list[float] = []
    repeat_authority_observed_regressions: list[float] = []
    for baseline, treatment in zip(baseline_samples, sidecar_samples):
        baseline_logical = _target_map(baseline, "logical_rows")
        treatment_logical = _target_map(treatment, "logical_rows")
        baseline_authority = _target_map(baseline, "authority_rows")
        treatment_authority = _target_map(treatment, "authority_rows")
        baseline_authority_observed = _target_map(
            baseline, "authority_rows", metric="observed_latency_ms"
        )
        treatment_authority_observed = _target_map(
            treatment, "authority_rows", metric="observed_latency_ms"
        )
        if not (
            baseline_logical.keys()
            == treatment_logical.keys()
            == baseline_authority.keys()
            == treatment_authority.keys()
        ):
            raise RuntimeError("paired target identifiers differ")
        local_benefit = [
            baseline_logical[key] - treatment_logical[key]
            for key in baseline_logical
        ]
        local_regression = [
            treatment_authority[key] - baseline_authority[key]
            for key in baseline_authority
        ]
        local_observed_regression = [
            treatment_authority_observed[key]
            - baseline_authority_observed[key]
            for key in baseline_authority_observed
        ]
        target_benefits.extend(local_benefit)
        authority_regressions.extend(local_regression)
        authority_observed_regressions.extend(local_observed_regression)
        repeat_target_benefits.append(statistics.fmean(local_benefit))
        repeat_authority_regressions.append(
            statistics.fmean(local_regression)
        )
        repeat_authority_observed_regressions.append(
            statistics.fmean(local_observed_regression)
        )

    targets = sum(
        int(sample["authoritative_targets"]) for sample in sidecar_samples
    )
    visible_hits = sum(
        int(sample["visible_speculative_hits"]) for sample in sidecar_samples
    )
    sidecar_started = sum(
        int(sample["sidecar_started"]) for sample in sidecar_samples
    )
    selected = sum(
        int(sample["selection_selected"]) for sample in sidecar_samples
    )
    selected_hits = sum(
        int(sample["selection_selected_hits"])
        for sample in sidecar_samples
    )
    selected_expected_gross_benefit_ms = sum(
        float(sample["selection_expected_gross_benefit_ms"])
        for sample in sidecar_samples
    )
    selected_expected_net_benefit_ms = sum(
        float(sample["selection_expected_net_benefit_ms"])
        for sample in sidecar_samples
    )
    baseline_logical_wall = sum(
        float(sample["logical_done_wall_s"]) for sample in baseline_samples
    )
    treatment_logical_wall = sum(
        float(sample["logical_done_wall_s"]) for sample in sidecar_samples
    )
    baseline_authority_wall = sum(
        float(sample["authority_done_wall_s"]) for sample in baseline_samples
    )
    treatment_authority_wall = sum(
        float(sample["authority_done_wall_s"]) for sample in sidecar_samples
    )
    repeat_authority_wall_log_ratios = [
        math.log(
            float(treatment["authority_done_wall_s"])
            / float(baseline["authority_done_wall_s"])
        )
        for baseline, treatment in zip(baseline_samples, sidecar_samples)
    ]
    mean_authority_regression = (
        statistics.fmean(authority_regressions)
        if authority_regressions
        else 0.0
    )
    mean_authority_observed_regression = (
        statistics.fmean(authority_observed_regressions)
        if authority_observed_regressions
        else 0.0
    )
    repeat_authority_median = statistics.median(
        repeat_authority_regressions
    )
    no_regression_margin_ms = 0.10
    authority_wall_regression = ratio(
        treatment_authority_wall - baseline_authority_wall,
        baseline_authority_wall,
    )
    legacy_point_no_regression = (
        mean_authority_regression <= no_regression_margin_ms
        and mean_authority_observed_regression <= no_regression_margin_ms
        and repeat_authority_median <= no_regression_margin_ms
        and authority_wall_regression <= 0.001
    )
    authority_latency_inference = _paired_repeat_inference(
        repeat_authority_regressions,
        margin=no_regression_margin_ms,
    )
    authority_latency_inference["scale"] = "ms/target"
    authority_observed_latency_inference = _paired_repeat_inference(
        repeat_authority_observed_regressions,
        margin=no_regression_margin_ms,
    )
    authority_observed_latency_inference["scale"] = "ms/target"
    logical_benefit_inference = _paired_benefit_inference(
        repeat_target_benefits
    )
    authority_wall_log_inference = _paired_repeat_inference(
        repeat_authority_wall_log_ratios,
        margin=math.log1p(0.001),
    )
    authority_wall_inference = {
        **authority_wall_log_inference,
        "scale": "log(treatment_wall / baseline_wall)",
        "geometric_mean_regression_fraction": (
            math.expm1(float(authority_wall_log_inference["mean"]))
        ),
        "ci90_regression_fraction": [
            math.expm1(float(value))
            for value in authority_wall_log_inference["ci90"]
        ],
        "lower_95_regression_fraction": math.expm1(
            float(authority_wall_log_inference["lower_95"])
        ),
        "upper_95_regression_fraction": math.expm1(
            float(authority_wall_log_inference["upper_95"])
        ),
        "margin_fraction": 0.001,
        "repeat_log_ratios": repeat_authority_wall_log_ratios,
    }
    component_decisions = {
        str(authority_latency_inference["decision"]),
        str(authority_observed_latency_inference["decision"]),
        str(authority_wall_inference["decision"]),
    }
    if "insufficient_repetitions" in component_decisions:
        overall_no_regression_decision = "insufficient_repetitions"
    elif "regression" in component_decisions:
        overall_no_regression_decision = "regression"
    elif component_decisions == {"pass"}:
        overall_no_regression_decision = "pass"
    else:
        overall_no_regression_decision = "inconclusive"
    return {
        "scenario": scenario,
        "offered_concurrency": concurrency,
        "repetitions": len(sidecar_samples),
        "strict_shadow_barrier": all(
            bool(sample.get("shadow_barrier", False))
            for sample in (*baseline_samples, *sidecar_samples)
        ),
        "require_precompletion": all(
            bool(sample.get("require_precompletion", False))
            for sample in (*baseline_samples, *sidecar_samples)
        ),
        "strict_positive_budget_certificate": all(
            bool(sample.get("strict_positive_budget_certificate", False))
            for sample in sidecar_samples
            if bool(sample.get("sidecar_activated", False))
        ) and any(
            bool(sample.get("sidecar_activated", False))
            for sample in sidecar_samples
        ),
        "unsafe_positive_ablation": any(
            bool(sample.get("unsafe_positive_ablation", False))
            for sample in sidecar_samples
        ),
        "service_ms": service_ms,
        "lead_ms": lead_ms,
        "sidecar_slots": sidecar_slots,
        "sidecar_backend": sidecar_backend,
        "probability_threshold": probability_threshold,
        "coordination_cost_ms": float(coordination_cost_ms),
        "authoritative_targets": targets,
        "selection_selected": selected,
        "selection_selected_hits": selected_hits,
        "selection_expected_gross_benefit_ms": (
            selected_expected_gross_benefit_ms
        ),
        "selection_expected_net_benefit_ms": (
            selected_expected_net_benefit_ms
        ),
        "selection_precision": ratio(selected_hits, selected),
        "visible_speculative_hits": visible_hits,
        "visible_target_coverage": ratio(visible_hits, targets),
        "sidecar_started": sidecar_started,
        "started_hit_precision": ratio(visible_hits, sidecar_started),
        "physical_call_amplification": ratio(
            targets + sidecar_started, targets
        ),
        "mean_logical_latency_benefit_ms_per_target": (
            statistics.fmean(target_benefits) if target_benefits else 0.0
        ),
        "median_repeat_logical_latency_benefit_ms_per_target": (
            statistics.median(repeat_target_benefits)
        ),
        "positive_logical_benefit_repetitions": sum(
            value > 0.0 for value in repeat_target_benefits
        ),
        "logical_benefit_inference": logical_benefit_inference,
        "logical_wall_speedup_fraction": ratio(
            baseline_logical_wall - treatment_logical_wall,
            baseline_logical_wall,
        ),
        "mean_authority_regression_ms_per_target": mean_authority_regression,
        "mean_authority_observed_regression_ms_per_target": (
            mean_authority_observed_regression
        ),
        "p95_authority_regression_ms_per_target": percentile(
            authority_regressions, 0.95
        ),
        "median_repeat_authority_regression_ms_per_target": (
            repeat_authority_median
        ),
        "authority_wall_regression_fraction": authority_wall_regression,
        "practical_no_regression_margin_ms": no_regression_margin_ms,
        "legacy_point_no_regression": legacy_point_no_regression,
        "practical_authority_no_regression": (
            overall_no_regression_decision == "pass"
        ),
        "gate_version": "paired-repeat-tost-v2-observed-authority",
        "authority_latency_inference": authority_latency_inference,
        "authority_observed_latency_inference": (
            authority_observed_latency_inference
        ),
        "authority_wall_inference": authority_wall_inference,
        "noise_calibration": {
            "kind": "identical_AA",
            "matched_configuration": False,
            "decision": "missing",
        },
        "overall_no_regression_decision": overall_no_regression_decision,
        "all_safety_invariants_passed": all(
            all(bool(value) for value in sample["safety"].values())
            for sample in (*baseline_samples, *sidecar_samples)
        ),
        "repeat_logical_latency_benefit_ms_per_target": repeat_target_benefits,
        "repeat_authority_regression_ms_per_target": (
            repeat_authority_regressions
        ),
        "repeat_authority_observed_regression_ms_per_target": (
            repeat_authority_observed_regressions
        ),
        "repeat_authority_wall_log_ratio": (
            repeat_authority_wall_log_ratios
        ),
        "paired_repeat_records": [
            {
                "repeat": repeat,
                "seed": int(baseline["seed"]),
                "order": "AB" if repeat % 2 == 0 else "BA",
                "baseline_logical_rows": baseline["logical_rows"],
                "treatment_logical_rows": treatment["logical_rows"],
                "baseline_authority_rows": baseline["authority_rows"],
                "treatment_authority_rows": treatment["authority_rows"],
            }
            for repeat, (baseline, treatment) in enumerate(
                zip(baseline_samples, sidecar_samples)
            )
        ],
        "samples": {
            "baseline": [
                {
                    key: value
                    for key, value in sample.items()
                    if key not in {"logical_rows", "authority_rows"}
                }
                for sample in baseline_samples
            ],
            "sidecar": [
                {
                    key: value
                    for key, value in sample.items()
                    if key not in {"logical_rows", "authority_rows"}
                }
                for sample in sidecar_samples
            ],
        },
    }


async def run_matrix(
    windows: Sequence[ScoredWindow],
    *,
    concurrencies: Sequence[int],
    repetitions: int,
    scenarios: Sequence[str],
    workers: int,
    visit_capacity: int,
    service_ms: float,
    lead_ms: float,
    sidecar_slots: int,
    max_sidecar_pending: int,
    probability_threshold: float,
    sidecar_backend: str,
    cpu_isolation: bool,
    shadow_barrier: bool = False,
    authority_control_burst_limit: int = 0,
    dedicated_authority_thread: bool = False,
    dedicated_authority_process: bool = False,
    require_precompletion: bool = False,
    completion_guard_ms: float = 0.0,
    eager_result_staging: bool = False,
    pull_result_staging: bool = False,
    coordination_cost_ms: float = 0.0,
    certified_exclusive_resources: bool = False,
    unsafe_positive_ablation: bool = False,
    direct_authority_baseline: bool = False,
) -> list[dict[str, Any]]:
    if dedicated_authority_thread and dedicated_authority_process:
        raise ValueError("authority thread and process modes are exclusive")
    if direct_authority_baseline and not dedicated_authority_process:
        raise ValueError(
            "direct authority baseline requires process-authority treatment"
        )
    cpu_role_assignment: tuple[int, int, int] | None = None
    if (
        (dedicated_authority_thread or dedicated_authority_process)
        and cpu_isolation
    ):
        cpu_role_assignment = choose_authority_control_sidecar_cpus(
            os.sched_getaffinity(0)
        )
    scenario_rows = []
    if "observed" in scenarios:
        scenario_rows.append(("observed_nested_oof", list(windows)))
    if "all-wrong" in scenarios:
        scenario_rows.append(
            ("all_wrong_counterfactual", force_all_wrong(windows))
        )
    rows: list[dict[str, Any]] = []
    for scenario, scenario_windows in scenario_rows:
        for concurrency in concurrencies:
            print(
                f"running scenario={scenario} concurrency={concurrency} "
                f"K={sidecar_slots} threshold={probability_threshold}",
                flush=True,
            )
            baseline_samples = []
            sidecar_samples = []
            for repetition in range(repetitions):
                async def run_one(
                    k: int, *, is_baseline: bool
                ) -> dict[str, Any]:
                    return await _run_sample(
                        scenario_windows,
                        offered_concurrency=concurrency,
                        seed=repetition,
                        workers=workers,
                        visit_capacity=visit_capacity,
                        service_ms=service_ms,
                        lead_ms=lead_ms,
                        sidecar_slots=k,
                        max_sidecar_pending=max_sidecar_pending,
                        probability_threshold=probability_threshold,
                        sidecar_backend=sidecar_backend,
                        cpu_isolation=cpu_isolation,
                        shadow_barrier=shadow_barrier,
                        authority_control_burst_limit=(
                            authority_control_burst_limit
                        ),
                        dedicated_authority_thread=(
                            dedicated_authority_thread
                        ),
                        dedicated_authority_process=(
                            dedicated_authority_process
                            and not (
                                is_baseline and direct_authority_baseline
                            )
                        ),
                        cpu_role_assignment=cpu_role_assignment,
                        require_precompletion=require_precompletion,
                        completion_guard_ms=completion_guard_ms,
                        eager_result_staging=eager_result_staging,
                        pull_result_staging=pull_result_staging,
                        coordination_cost_ms=coordination_cost_ms,
                        certified_exclusive_resources=(
                            certified_exclusive_resources
                        ),
                        unsafe_positive_ablation=unsafe_positive_ablation,
                    )

                if repetition % 2 == 0:
                    baseline = await run_one(0, is_baseline=True)
                    treatment = await run_one(
                        sidecar_slots, is_baseline=False
                    )
                else:
                    treatment = await run_one(
                        sidecar_slots, is_baseline=False
                    )
                    baseline = await run_one(0, is_baseline=True)
                baseline_samples.append(baseline)
                sidecar_samples.append(treatment)
            rows.append(
                _aggregate_cell(
                    scenario=scenario,
                    concurrency=concurrency,
                    baseline_samples=baseline_samples,
                    sidecar_samples=sidecar_samples,
                    service_ms=service_ms,
                    lead_ms=lead_ms,
                    sidecar_slots=sidecar_slots,
                    probability_threshold=probability_threshold,
                    coordination_cost_ms=coordination_cost_ms,
                    sidecar_backend=sidecar_backend,
                )
            )
    return rows


def render_report(payload: Mapping[str, Any]) -> str:
    if payload["configuration"].get("direct_authority_baseline", False):
        authority_placement = (
            "The K=0 baseline uses the original in-process authority path; "
            "the treatment uses a private authority process/GIL/CPU. This is "
            "an end-to-end topology-migration comparison, not an incremental "
            "sidecar-only comparison. "
        )
    elif payload["configuration"].get(
        "dedicated_authority_process", False
    ):
        authority_placement = (
            "The unchanged demand-only broker runs in a private authority "
            "process/GIL/CPU; the parent loop is a separate control role. "
        )
    elif payload["configuration"].get("dedicated_authority_thread", False):
        authority_placement = (
            "The unchanged demand-only broker runs on a private authority "
            "event-loop thread/CPU; the parent loop is a separate control role. "
        )
    else:
        authority_placement = (
            "Authority uses an unchanged demand-only broker on the parent loop. "
        )
    if payload["configuration"].get(
        "pull_prestage_before_authority", False
    ):
        result_delivery = (
            "A bounded pull is performed in the pre-authority guard window "
            "and then the result epoch is sealed; no parent bridge or socket "
            "read runs during authority, and exact confirmation is an O(1) "
            "parent-local lookup. "
        )
    elif payload["configuration"].get("pull_result_staging", False):
        result_delivery = (
            "Completed results remain in a bounded kernel mailbox; no parent "
            "result bridge runs during the timed authority interval, and an "
            "exact confirmation performs only a bounded non-blocking pull. "
        )
    else:
        result_delivery = (
            "A blocking result bridge is started during untimed setup when "
            "the process sidecar is activated. "
        )
    lines = [
        "# Pattern-v2 isolated speculative sidecar",
        "",
        authority_placement + "Predictor scoring and "
        "selection are precomputed in the parent before the timed wall; only "
        "admission, speculative execution, finite-lease cleanup, and drain use "
        "the dedicated sidecar control plane with batched non-blocking ingress "
        f"(`{payload['configuration']['sidecar_backend']}`). "
        + result_delivery
        + "Every target still submits one shadow authority attempt; speculative "
        "success can only shorten the logical completion.",
        "",
        "| Scenario | C | K | Threshold | Authority lane regression ms/target | "
        "Authority observed regression ms/target | "
        "Logical benefit ms/target | Benefit evidence | Logical wall speedup | Visible coverage | "
        "Started precision | Call amp. | No-regression |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["load_matrix"]:
        lines.append(
            "| {scenario} | {offered_concurrency} | {sidecar_slots} | "
            "{probability_threshold:.3f} | "
            "{mean_authority_regression_ms_per_target:+.3f} | "
            "{mean_authority_observed_regression_ms_per_target:+.3f} | "
            "{mean_logical_latency_benefit_ms_per_target:+.3f} | "
            "{benefit} | "
            "{logical_wall_speedup_fraction:+.1%} | "
            "{visible_target_coverage:.1%} | {started_hit_precision:.1%} | "
            "{physical_call_amplification:.3f}x | {safe} |".format(
                safe=row["overall_no_regression_decision"],
                benefit=row["logical_benefit_inference"]["decision"],
                **row,
            )
        )
    metric_lines = [
            "",
            "## Metric semantics",
            "",
            "- Authority lane regression compares scheduled-to-terminal time "
            "inside the always-executed demand-only path. Authority observed "
            "regression additionally includes return handoff to the control loop.",
            "- Logical benefit is agent-visible scheduled-to-first-valid-result "
            "latency, with authority winning ties.",
            "- Benefit evidence is an improvement only when the repeat-level "
            "one-sided 95% lower bound on saved latency is above zero.",
            "- Logical wall stops after all agent-visible results. Authority wall "
            "then waits for every shadow authority call; drained wall additionally "
            "waits for the isolated sidecar.",
            "- No-regression inference treats one paired AB/BA repetition—not "
            "individual targets—as the independent unit. A cell needs at least "
            "eight repetitions and one-sided 95% upper bounds no larger than "
            "0.10 ms/target and 0.1% authority wall. Otherwise it is reported "
            "as regression, inconclusive, or insufficient rather than a binary "
            "point-estimate failure.",
            "- The authority-control burst circuit breaker sets the safe start "
            "budget to zero once a synchronized authority batch exceeds the "
            "host-calibrated limit; a zero limit means no positive resource "
            "certificate was supplied. The latch remains closed for the rest of "
            "the replay, and a fully abstained treatment never starts a sidecar "
            "process.",
            "",
            "## Scope",
            "",
            "This synthetic replay establishes control-plane behavior, not shared "
            "backend quota isolation. Production use still requires an independent "
            "connection/rate/concurrency entitlement for the sidecar. A formal "
            "sub-millisecond equivalence claim additionally requires a matched "
            "A/A noise calibration; absent that, repeat-level inference may remain "
            "inconclusive. All-wrong cells that do not confirm a practical "
            "regression still do not establish equivalence.",
            "",
        ]
    if payload["configuration"].get("strict_shadow_barrier", False):
        metric_lines[6:6] = [
            "- Strict shadow barrier is enabled: a later batch is not admitted "
            "until the current batch's shadow-authority calls have drained. This "
            "prevents speculative early return from increasing protected-broker "
            "overlap across batches."
        ]
    lines.extend(metric_lines)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[1, 16, 64]
    )
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("observed", "all-wrong"),
        default=["observed", "all-wrong"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visit-capacity", type=int, default=2)
    parser.add_argument("--service-ms", type=float, default=20.0)
    parser.add_argument("--lead-ms", type=float, default=10.0)
    parser.add_argument("--sidecar-slots", type=int, default=4)
    parser.add_argument("--max-sidecar-pending", type=int, default=8)
    parser.add_argument(
        "--sidecar-backend",
        choices=("process", "thread"),
        default="process",
    )
    parser.add_argument(
        "--no-cpu-isolation",
        action="store_false",
        dest="cpu_isolation",
        help="disable authority/sidecar CPU affinity isolation",
    )
    parser.add_argument(
        "--strict-shadow-barrier",
        action="store_true",
        help=(
            "do not admit a later batch until all shadow authority calls "
            "from the current batch have drained"
        ),
    )
    parser.add_argument(
        "--dedicated-authority-thread",
        action="store_true",
        help=(
            "run the demand-only broker on a private asyncio thread/CPU; "
            "requires three granted CPUs for a positive speculative budget"
        ),
    )
    parser.add_argument(
        "--dedicated-authority-process",
        action="store_true",
        help=(
            "run the demand-only broker in a private child process/GIL; "
            "requires three granted CPUs for a positive speculative budget"
        ),
    )
    parser.add_argument("--probability-threshold", type=float, default=0.20)
    parser.add_argument(
        "--require-precompletion",
        action="store_true",
        help=(
            "admit positive speculation only when predicted service plus "
            "guard fits before authority confirmation, and reject late hits"
        ),
    )
    parser.add_argument("--completion-guard-ms", type=float, default=0.0)
    parser.add_argument(
        "--coordination-cost-ms",
        type=float,
        default=0.0,
        help=(
            "conservative fixed per-start IPC/wakeup/delivery cost subtracted "
            "from expected saved latency before global Top-K admission"
        ),
    )
    parser.add_argument(
        "--eager-result-staging",
        action="store_true",
        help=(
            "transfer completed process-side results into a bounded private "
            "parent staging map before confirmation; exact claim is then a "
            "non-blocking parent-local lookup"
        ),
    )
    parser.add_argument(
        "--pull-result-staging",
        action="store_true",
        help=(
            "stage completed process-side results in a bounded kernel "
            "mailbox without a continuous parent bridge; with precompletion "
            "and a positive guard, bounded results are pulled and the epoch "
            "is sealed before authority starts"
        ),
    )
    parser.add_argument(
        "--authority-control-burst-limit", type=int, default=0
    )
    parser.add_argument(
        "--certified-exclusive-resources",
        action="store_true",
        help=(
            "assert that authority/control/sidecar CPUs and backend quota "
            "are exclusively reserved; required for a strict positive budget"
        ),
    )
    parser.add_argument(
        "--unsafe-positive-ablation",
        action="store_true",
        help=(
            "allow positive speculation without the complete strict resource "
            "certificate; results are ablation-only and cannot support a "
            "no-regression claim"
        ),
    )
    parser.add_argument(
        "--direct-authority-baseline",
        action="store_true",
        help=(
            "compare the process-authority treatment against the original "
            "in-process demand-only authority path; this measures topology "
            "migration cost rather than incremental speculation alone"
        ),
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if any(value <= 0 for value in args.concurrencies):
        parser.error("--concurrencies must be positive")
    if args.workers <= 0 or args.visit_capacity <= 0:
        parser.error("authority capacities must be positive")
    if args.visit_capacity > args.workers:
        parser.error("--visit-capacity cannot exceed --workers")
    if args.sidecar_slots <= 0 or args.max_sidecar_pending <= 0:
        parser.error("sidecar capacities must be positive")
    if args.authority_control_burst_limit < 0:
        parser.error("--authority-control-burst-limit must be non-negative")
    if args.dedicated_authority_thread and args.dedicated_authority_process:
        parser.error("authority thread and process modes are exclusive")
    if args.direct_authority_baseline and not args.dedicated_authority_process:
        parser.error(
            "--direct-authority-baseline requires "
            "--dedicated-authority-process"
        )
    if args.eager_result_staging and args.pull_result_staging:
        parser.error(
            "--eager-result-staging and --pull-result-staging are mutually "
            "exclusive"
        )
    if (
        args.eager_result_staging or args.pull_result_staging
    ) and args.sidecar_backend != "process":
        parser.error("result staging requires --sidecar-backend process")
    if (
        args.dedicated_authority_thread
        or args.dedicated_authority_process
    ) and not args.cpu_isolation:
        parser.error(
            "dedicated authority modes require CPU isolation"
        )
    if args.service_ms <= 0.0 or args.lead_ms < 0.0:
        parser.error("service must be positive and lead non-negative")
    if not math.isfinite(args.completion_guard_ms) or args.completion_guard_ms < 0:
        parser.error("--completion-guard-ms must be finite and non-negative")
    if (
        not math.isfinite(args.coordination_cost_ms)
        or args.coordination_cost_ms < 0
    ):
        parser.error("--coordination-cost-ms must be finite and non-negative")
    if not 0.0 <= args.probability_threshold <= 1.0:
        parser.error("--probability-threshold must be in [0, 1]")
    return args


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    trace_files = {
        path.name: sha256_file(path)
        for path in sorted(args.traces.glob("*.jsonl"))
    }
    input_sha256 = {
        "trace_files": trace_files,
        "trace_manifest": canonical_sha256(trace_files),
    }
    source_sha256 = {
        "runner": sha256_file(SCRIPT),
        "broker": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "live_broker.py"
        ),
        "authority_process_lane": sha256_file(
            REPRODUCTION_ROOT
            / "paste_repro"
            / "authority_process_lane.py"
        ),
        "policy": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "speculation_policy.py"
        ),
        "sidecar": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "speculation_sidecar.py"
        ),
        "invocation": sha256_file(
            REPRODUCTION_ROOT / "paste_repro" / "invocation.py"
        ),
        "adaptive_trace_builder": sha256_file(
            SCRIPT.parent / "run_pattern_v2_adaptive_load.py"
        ),
        "metric_helpers": sha256_file(
            SCRIPT.parent / "run_pattern_v2_load_robustness.py"
        ),
    }
    windows, oof = collect_nested_oof_windows(args.traces)
    rows = await run_matrix(
        windows,
        concurrencies=args.concurrencies,
        repetitions=args.repetitions,
        scenarios=args.scenarios,
        workers=args.workers,
        visit_capacity=args.visit_capacity,
        service_ms=args.service_ms,
        lead_ms=args.lead_ms,
        sidecar_slots=args.sidecar_slots,
        max_sidecar_pending=args.max_sidecar_pending,
        probability_threshold=args.probability_threshold,
        sidecar_backend=args.sidecar_backend,
        cpu_isolation=args.cpu_isolation,
        shadow_barrier=args.strict_shadow_barrier,
        authority_control_burst_limit=args.authority_control_burst_limit,
        dedicated_authority_thread=args.dedicated_authority_thread,
        dedicated_authority_process=args.dedicated_authority_process,
        require_precompletion=args.require_precompletion,
        completion_guard_ms=args.completion_guard_ms,
        eager_result_staging=args.eager_result_staging,
        pull_result_staging=args.pull_result_staging,
        coordination_cost_ms=args.coordination_cost_ms,
        certified_exclusive_resources=args.certified_exclusive_resources,
        unsafe_positive_ablation=args.unsafe_positive_ablation,
        direct_authority_baseline=args.direct_authority_baseline,
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_control_plane_replay",
        "command": shlex.join([sys.executable, *sys.argv]),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "concurrencies": list(args.concurrencies),
            "repetitions": args.repetitions,
            "scenarios": list(args.scenarios),
            "workers": args.workers,
            "visit_capacity": args.visit_capacity,
            "service_ms": args.service_ms,
            "lead_ms": args.lead_ms,
            "sidecar_slots": args.sidecar_slots,
            "max_sidecar_pending": args.max_sidecar_pending,
            "probability_threshold": args.probability_threshold,
            "require_precompletion": args.require_precompletion,
            "completion_guard_ms": args.completion_guard_ms,
            "eager_result_staging": args.eager_result_staging,
            "pull_result_staging": args.pull_result_staging,
            "pull_prestage_before_authority": (
                args.pull_result_staging
                and args.require_precompletion
                and args.completion_guard_ms > 0.0
            ),
            "pull_prestage_mode": (
                "raw concurrent Future on the speculative CPU; no asyncio "
                "callback and no authority-release dependency"
                if args.pull_result_staging
                else None
            ),
            "pull_prestage_quiet_gap_ms": (
                args.completion_guard_ms / 2.0
                if args.pull_result_staging
                and args.require_precompletion
                and args.completion_guard_ms > 0.0
                else 0.0
            ),
            "pull_max_staged_result_bytes": (
                STRICT_PULL_MAX_STAGED_RESULT_BYTES
                if args.pull_result_staging
                else None
            ),
            "coordination_cost_ms": args.coordination_cost_ms,
            "certified_exclusive_resources": (
                args.certified_exclusive_resources
            ),
            "unsafe_positive_ablation": args.unsafe_positive_ablation,
            "direct_authority_baseline": args.direct_authority_baseline,
            "sidecar_backend": args.sidecar_backend,
            "cpu_isolation": args.cpu_isolation,
            "arrival_model": "closed-loop source-session streams",
            "authority_mode": "always-executed demand-only shadow authority",
            "dedicated_authority_thread": (
                args.dedicated_authority_thread
            ),
            "dedicated_authority_process": (
                args.dedicated_authority_process
            ),
            "strict_shadow_barrier": args.strict_shadow_barrier,
            "authority_control_burst_limit": (
                args.authority_control_burst_limit
            ),
            "prediction_selection_timing": (
                "precomputed in the parent before timed wall_started"
            ),
            "sidecar_control_plane": (
                (
                    "dedicated process, batched SOCK_SEQPACKET ingress, finite "
                    "leases, SCHED_IDLE priority, and bounded non-blocking "
                    "pull-result mailbox without a timed parent bridge"
                    if args.pull_result_staging
                    else "dedicated process, batched SOCK_SEQPACKET ingress, "
                    "finite leases, SCHED_IDLE priority, and blocking result "
                    "bridge prestarted during untimed setup"
                )
                if args.sidecar_backend == "process"
                else "dedicated event-loop thread"
            ),
            "authority_confirmation_clock": (
                "absolute monotonic deadline released by a dedicated sleeping "
                "timer thread shared by baseline and treatment; speculative "
                "prefetch never runs on or gates this clock"
            ),
            "cpu_isolation_policy": (
                "topology-aware three logical CPU roles for authority, "
                "control, and sidecar when dedicated authority mode is "
                "enabled; otherwise separate authority/sidecar logical CPUs, "
                "with silent prefetch sharing the sidecar CPU, for a positive "
                "resource certificate"
                if args.cpu_isolation
                else "disabled"
            ),
            "paired_execution_order": "AB/BA counterbalanced by repetition",
        },
        "nested_oof": oof,
        "calibration_quality": calibration_quality(windows),
        "load_matrix": rows,
        "input_sha256": input_sha256,
        "source_sha256": source_sha256,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = asyncio.run(async_main(args))
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(f"wrote {args.output_dir.resolve()}")
    print(f"payload_sha256={payload['payload_sha256']}")


if __name__ == "__main__":
    main()
