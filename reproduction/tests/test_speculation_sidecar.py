from __future__ import annotations

import asyncio
import multiprocessing
import os
from pathlib import Path
import select
import tempfile
import threading
import time
import unittest
from unittest import mock

from paste_repro.invocation import Invocation
from paste_repro.speculation_sidecar import (
    ExactSpeculationKey,
    ProcessSpeculativeSidecar,
    SidecarClosed,
    SidecarExpired,
    SidecarRejected,
    SidecarTombstoned,
    SpeculativeSidecarError,
    SpeculativeHandle,
    SpeculativeSidecar,
    choose_authority_control_sidecar_cpus,
    choose_authority_sidecar_cpus,
    distinct_physical_core_certificate,
    race_authority_with_speculation,
)


def _visit(name: str) -> Invocation:
    return Invocation("visit", {"url": f"https://example.test/{name}"})


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not satisfied before timeout")


def _write_cpu_topology(
    root: Path,
    cpu: int,
    *,
    package: int,
    core: int,
    node: int,
    siblings: str,
) -> None:
    cpu_root = root / f"cpu{cpu}"
    topology = cpu_root / "topology"
    topology.mkdir(parents=True)
    (topology / "physical_package_id").write_text(str(package))
    (topology / "core_id").write_text(str(core))
    (topology / "thread_siblings_list").write_text(siblings)
    (cpu_root / f"node{node}").mkdir()


class CpuPlacementTests(unittest.TestCase):
    def test_physical_core_certificate_fails_closed_on_smt_or_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_cpu_topology(
                root, 0, package=0, core=0, node=0, siblings="0,1"
            )
            _write_cpu_topology(
                root, 1, package=0, core=0, node=0, siblings="0,1"
            )
            _write_cpu_topology(
                root, 2, package=0, core=1, node=0, siblings="2"
            )
            self.assertFalse(
                distinct_physical_core_certificate(
                    [0, 1], topology_root=root
                )
            )
            self.assertTrue(
                distinct_physical_core_certificate(
                    [0, 2], topology_root=root
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(
                distinct_physical_core_certificate(
                    [0, 9], topology_root=directory
                )
            )

    def test_three_role_choice_keeps_stable_prefix_and_avoids_smt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_cpu_topology(
                root, 0, package=0, core=0, node=0, siblings="0,1"
            )
            _write_cpu_topology(
                root, 1, package=0, core=0, node=0, siblings="0,1"
            )
            _write_cpu_topology(
                root, 2, package=0, core=1, node=0, siblings="2"
            )
            _write_cpu_topology(
                root, 3, package=0, core=2, node=0, siblings="3"
            )

            pair = choose_authority_sidecar_cpus(
                [3, 1, 2, 0], topology_root=root
            )
            triple = choose_authority_control_sidecar_cpus(
                [3, 1, 2, 0], topology_root=root
            )
            self.assertEqual(triple[:2], pair)
            self.assertEqual(triple, (0, 2, 3))

    def test_three_role_choice_requires_three_logical_cpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                choose_authority_control_sidecar_cpus(
                    [0, 1], topology_root=directory
                )

    def test_prefers_same_numa_distinct_core_over_smt_and_remote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_cpu_topology(
                root, 0, package=0, core=0, node=0, siblings="0-1"
            )
            _write_cpu_topology(
                root, 1, package=0, core=0, node=0, siblings="0-1"
            )
            _write_cpu_topology(
                root, 2, package=0, core=1, node=0, siblings="2"
            )
            _write_cpu_topology(
                root, 3, package=1, core=0, node=1, siblings="3"
            )

            self.assertEqual(
                choose_authority_sidecar_cpus(
                    [3, 1, 2, 0], topology_root=root
                ),
                (0, 2),
            )

    def test_prefers_same_socket_distinct_core_over_remote_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_cpu_topology(
                root, 0, package=0, core=0, node=0, siblings="0"
            )
            _write_cpu_topology(
                root, 4, package=0, core=1, node=1, siblings="4"
            )
            _write_cpu_topology(
                root, 8, package=1, core=0, node=2, siblings="8"
            )

            self.assertEqual(
                choose_authority_sidecar_cpus(
                    [0, 4, 8], topology_root=root
                ),
                (0, 4),
            )

    def test_remote_distinct_core_beats_known_smt_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_cpu_topology(
                root, 0, package=0, core=0, node=0, siblings="0,2"
            )
            _write_cpu_topology(
                root, 2, package=0, core=0, node=0, siblings="0,2"
            )
            _write_cpu_topology(
                root, 9, package=1, core=0, node=1, siblings="9"
            )

            self.assertEqual(
                choose_authority_sidecar_cpus(
                    [0, 2, 9], topology_root=root
                ),
                (0, 9),
            )

    def test_missing_topology_is_deterministic_and_validates_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                choose_authority_sidecar_cpus(
                    [7, 3, 7], topology_root=root
                ),
                (3, 7),
            )
            with self.assertRaises(ValueError):
                choose_authority_sidecar_cpus([3], topology_root=root)


class SpeculativeSidecarTests(unittest.TestCase):
    def test_executor_entry_rechecks_latest_start_deadline(self) -> None:
        executor_calls = 0
        clock_calls = 0

        def clock() -> float:
            nonlocal clock_calls
            clock_calls += 1
            # Admission/drain/dispatch observe an admissible time; the
            # newly-created coroutine then observes that true executor entry
            # is late.
            return 0.0 if clock_calls <= 4 else 2.0

        async def executor(_: Invocation) -> str:
            nonlocal executor_calls
            executor_calls += 1
            return "must-not-run"

        sidecar = SpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            clock=clock,
        )
        try:
            handle = sidecar.try_submit(
                _visit("late-entry"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=1.0,
            )
            self.assertIsNotNone(handle)
            sidecar.start()
            assert handle is not None
            with self.assertRaises(SidecarExpired):
                handle.future.result(timeout=1)
            self.assertEqual(executor_calls, 0)
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["started"], 0)
            self.assertEqual(
                snapshot["stats"]["expired_before_executor"], 1
            )
        finally:
            sidecar.close()

    def test_submit_is_bounded_nonblocking_before_start(self) -> None:
        executor_thread_ids: list[int] = []

        async def executor(invocation: Invocation) -> str:
            executor_thread_ids.append(threading.get_ident())
            return str(invocation.arguments["url"])

        sidecar = SpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            ingress_capacity=1,
        )
        main_thread_id = threading.get_ident()
        started = time.perf_counter()
        accepted = sidecar.try_submit(
            _visit("accepted"),
            session_id="s1",
            decision_id="d1",
            priority=1.0,
        )
        rejected = sidecar.try_submit(
            _visit("ingress-full"),
            session_id="s2",
            decision_id="d2",
            priority=2.0,
        )
        elapsed = time.perf_counter() - started

        self.assertIsNotNone(accepted)
        self.assertIsNone(rejected)
        self.assertLess(elapsed, 0.05)
        before = sidecar.snapshot()
        self.assertEqual(before["counts"]["ingress"], 1)
        self.assertEqual(before["stats"]["ingress_full"], 1)

        sidecar.start()
        try:
            assert accepted is not None
            self.assertIn("accepted", accepted.future.result(timeout=1))
            self.assertEqual(len(executor_thread_ids), 1)
            self.assertNotEqual(executor_thread_ids[0], main_thread_id)
        finally:
            sidecar.close()

    def test_k_limit_and_global_benefit_heap(self) -> None:
        release = threading.Event()
        blocker_started = threading.Event()
        order: list[str] = []
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        async def executor(invocation: Invocation) -> str:
            nonlocal active, max_active
            name = str(invocation.arguments["url"]).rsplit("/", 1)[-1]
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                order.append(name)
            if name == "blocker":
                blocker_started.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0.003)
            with state_lock:
                active -= 1
            return name

        sidecar = SpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=3,
            autostart=True,
        )
        try:
            blocker = sidecar.try_submit(
                _visit("blocker"),
                session_id="blocker-session",
                decision_id="blocker-decision",
                priority=100.0,
            )
            self.assertIsNotNone(blocker)
            self.assertTrue(blocker_started.wait(1))

            low = sidecar.try_submit(
                _visit("low"),
                session_id="low-session",
                decision_id="low-decision",
                priority=1.0,
            )
            high = sidecar.try_submit(
                _visit("high"),
                session_id="high-session",
                decision_id="high-decision",
                priority=10.0,
            )
            self.assertIsNotNone(low)
            self.assertIsNotNone(high)
            _wait_until(lambda: sidecar.snapshot()["counts"]["queued"] == 2)
            release.set()
            assert blocker is not None and low is not None and high is not None
            self.assertEqual(blocker.future.result(timeout=1), "blocker")
            self.assertEqual(high.future.result(timeout=1), "high")
            self.assertEqual(low.future.result(timeout=1), "low")
            self.assertEqual(order, ["blocker", "high", "low"])
            self.assertEqual(max_active, 1)
            self.assertEqual(sidecar.snapshot()["stats"]["max_running"], 1)
        finally:
            release.set()
            sidecar.close()

    def test_one_candidate_per_session_decision_replaces_only_queued(self) -> None:
        release = threading.Event()
        blocker_started = threading.Event()
        invoked: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["url"]).rsplit("/", 1)[-1]
            invoked.append(name)
            if name == "blocker":
                blocker_started.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
            return name

        sidecar = SpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=3,
            autostart=True,
        )
        try:
            blocker = sidecar.try_submit(
                _visit("blocker"),
                session_id="blocker",
                decision_id="d0",
                priority=100.0,
            )
            self.assertIsNotNone(blocker)
            self.assertTrue(blocker_started.wait(1))
            low = sidecar.try_submit(
                _visit("same-decision-low"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
            )
            self.assertIsNotNone(low)
            _wait_until(lambda: sidecar.snapshot()["counts"]["queued"] == 1)
            high = sidecar.try_submit(
                _visit("same-decision-high"),
                session_id="session",
                decision_id="decision",
                priority=10.0,
            )
            other_decision = sidecar.try_submit(
                _visit("other-decision"),
                session_id="session",
                decision_id="other",
                priority=5.0,
            )
            self.assertIsNotNone(high)
            self.assertIsNotNone(other_decision)
            assert low is not None
            with self.assertRaises(SidecarRejected):
                low.future.result(timeout=1)

            release.set()
            assert blocker is not None
            assert high is not None and other_decision is not None
            self.assertEqual(blocker.future.result(timeout=1), "blocker")
            self.assertEqual(high.future.result(timeout=1), "same-decision-high")
            self.assertEqual(other_decision.future.result(timeout=1), "other-decision")
            self.assertNotIn("same-decision-low", invoked)
            self.assertEqual(
                invoked,
                ["blocker", "same-decision-high", "other-decision"],
            )
            self.assertEqual(sidecar.snapshot()["stats"]["replaced_queued"], 1)
        finally:
            release.set()
            sidecar.close()

    def test_tombstone_never_cancels_running_physical_work(self) -> None:
        release = threading.Event()
        running_started = threading.Event()
        physically_finished = threading.Event()
        was_cancelled = threading.Event()
        invoked: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["url"]).rsplit("/", 1)[-1]
            invoked.append(name)
            if name == "running":
                running_started.set()
                try:
                    while not release.is_set():
                        await asyncio.sleep(0.001)
                except asyncio.CancelledError:
                    was_cancelled.set()
                    raise
                physically_finished.set()
            return name

        sidecar = SpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=2,
            autostart=True,
        )
        try:
            running = sidecar.try_submit(
                _visit("running"),
                session_id="running-session",
                decision_id="d1",
                priority=10.0,
            )
            self.assertIsNotNone(running)
            self.assertTrue(running_started.wait(1))
            queued = sidecar.try_submit(
                _visit("queued"),
                session_id="queued-session",
                decision_id="d2",
                priority=1.0,
            )
            self.assertIsNotNone(queued)
            _wait_until(lambda: sidecar.snapshot()["counts"]["queued"] == 1)

            started = time.perf_counter()
            self.assertTrue(
                sidecar.try_tombstone(session_id="queued-session")
            )
            self.assertTrue(
                sidecar.try_tombstone(session_id="running-session")
            )
            self.assertLess(time.perf_counter() - started, 0.05)
            assert running is not None and queued is not None
            with self.assertRaises(SidecarTombstoned):
                queued.future.result(timeout=1)
            with self.assertRaises(SidecarTombstoned):
                running.future.result(timeout=1)

            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["counts"]["running"], 1)
            self.assertEqual(snapshot["counts"]["published"], 0)
            self.assertFalse(physically_finished.is_set())
            self.assertFalse(was_cancelled.is_set())

            release.set()
            self.assertTrue(physically_finished.wait(1))
            _wait_until(lambda: sidecar.snapshot()["counts"]["running"] == 0)
            self.assertEqual(invoked, ["running"])
            self.assertFalse(was_cancelled.is_set())
        finally:
            release.set()
            sidecar.close()


@unittest.skipUnless(
    "fork" in multiprocessing.get_all_start_methods(),
    "process sidecar requires fork",
)
class ProcessSpeculativeSidecarTests(unittest.TestCase):
    def test_process_result_staging_modes_are_mutually_exclusive(self) -> None:
        async def executor(_: Invocation) -> str:
            return "unused"

        with self.assertRaises(ValueError):
            ProcessSpeculativeSidecar(
                executor,
                eager_result_staging=True,
                pull_result_staging=True,
            )

    def test_pull_staging_oversized_result_is_a_fail_open_miss(self) -> None:
        async def executor(_: Invocation) -> bytes:
            return b"x" * 256

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            pull_result_staging=True,
            max_staged_result_bytes=64,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("pull-oversized"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )
            self.assertIsNone(sidecar.try_claim(handle.key))
            with self.assertRaisesRegex(
                SpeculativeSidecarError, "ResultTooLarge"
            ):
                handle.future.result(timeout=0)
            snapshot = sidecar.snapshot()
            self.assertEqual(
                snapshot["capacity"]["max_staged_result_bytes"], 64
            )
            self.assertEqual(
                snapshot["transport"]["transport_claim_packets"], 0
            )
        finally:
            sidecar.close(timeout=3)

    def test_pull_staging_ready_hit_uses_only_bounded_kernel_mailbox(self) -> None:
        async def executor(invocation: Invocation) -> dict[str, object]:
            return {"key": invocation.key, "pid": os.getpid()}

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            result_capacity=2,
            pull_result_staging=True,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("pull-ready"),
                session_id="session",
                decision_id="decision",
                context_token="context-v1",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )

            # Completion is private in the socket. A non-exact lookup neither
            # consumes the mailbox nor completes the public observer.
            self.assertFalse(sidecar.bridge_started)
            self.assertFalse(handle.future.done())
            wrong = ExactSpeculationKey.from_invocation(
                _visit("pull-ready"),
                session_id="session",
                decision_id="decision",
                context_token="context-v2",
            )
            self.assertIsNone(sidecar.try_claim(wrong))
            self.assertFalse(handle.future.done())

            started = time.perf_counter()
            claimed = sidecar.try_claim(handle.key)
            self.assertLess(time.perf_counter() - started, 0.05)
            self.assertIs(claimed, handle)
            self.assertFalse(sidecar.bridge_started)
            self.assertEqual(
                handle.future.result(timeout=0),
                {"key": handle.invocation_key, "pid": sidecar.pid},
            )

            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["parent_staging"]["mode"], "pull")
            self.assertEqual(
                snapshot["transport"]["transport_claim_packets"], 0
            )
            self.assertEqual(snapshot["transport"]["transport_pull_hits"], 1)
            self.assertEqual(
                snapshot["transport"]["transport_pull_packets"], 1
            )
        finally:
            sidecar.close(timeout=3)

    def test_pull_prefetch_seals_claim_to_parent_local_state(self) -> None:
        async def executor(invocation: Invocation) -> tuple[str, str]:
            return invocation.key

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            pull_result_staging=True,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("pull-prefetch"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )

            self.assertEqual(sidecar.prefetch_pull_results(), 1)
            self.assertTrue(sidecar.pull_epoch_sealed)
            claimed = sidecar.try_claim(handle.key)
            self.assertIs(claimed, handle)
            self.assertEqual(handle.future.result(timeout=0), handle.invocation_key)
            snapshot = sidecar.snapshot()
            self.assertEqual(
                snapshot["transport"]["transport_pull_prefetch_calls"], 1
            )
            self.assertEqual(
                snapshot["transport"]["transport_pull_prefetch_packets"], 1
            )
            self.assertEqual(
                snapshot["transport"]["transport_claim_packets"], 0
            )
        finally:
            sidecar.close(timeout=3)

    def test_pull_prefetch_late_result_cannot_reenter_sealed_claim(self) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()
        release = context.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return "late"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            pull_result_staging=True,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("pull-prefetch-late"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            self.assertTrue(started.wait(1))
            self.assertEqual(sidecar.prefetch_pull_results(), 0)
            self.assertTrue(sidecar.pull_epoch_sealed)

            release.set()
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )
            # The packet is now readable, but a sealed authority claim is a
            # local miss and must not deserialize it.
            self.assertIsNone(sidecar.try_claim(handle.key))
            self.assertTrue(
                select.select([sidecar._event_parent], [], [], 0)[0]
            )
            with self.assertRaises(SidecarRejected):
                handle.future.result(timeout=0)
        finally:
            release.set()
            sidecar.close(timeout=3)

    def test_pull_staging_not_ready_is_immediate_non_cancelling_miss(self) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()
        release = context.Event()
        physically_finished = context.Event()
        cancelled = context.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            try:
                while not release.is_set():
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            physically_finished.set()
            return "late"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            pull_result_staging=True,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("pull-running"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            self.assertTrue(started.wait(1))
            assert handle is not None

            began = time.perf_counter()
            self.assertIsNone(sidecar.try_claim(handle.key))
            self.assertLess(time.perf_counter() - began, 0.05)
            self.assertFalse(sidecar.bridge_started)
            with self.assertRaises(SidecarRejected):
                handle.future.result(timeout=0)

            release.set()
            self.assertTrue(physically_finished.wait(1))
            snapshot = sidecar.snapshot()
            self.assertFalse(cancelled.is_set())
            self.assertEqual(
                snapshot["transport"]["transport_claim_packets"], 0
            )
            self.assertEqual(
                snapshot["transport"]["transport_pull_not_ready"], 1
            )
        finally:
            release.set()
            sidecar.close(timeout=3)

    def test_pull_staging_registry_contention_drops_without_waiting(self) -> None:
        async def executor(_: Invocation) -> str:
            return "ready-but-contended"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            pull_result_staging=True,
            autostart=True,
        )
        release_lock = threading.Event()
        lock_held = threading.Event()
        holder: threading.Thread | None = None
        try:
            handle = sidecar.try_submit(
                _visit("pull-lock-contention"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )
            with sidecar._registry_lock:
                request_id = sidecar._available[handle.key]

            def hold_registry() -> None:
                with sidecar._registry_lock:
                    lock_held.set()
                    release_lock.wait(1)

            holder = threading.Thread(target=hold_registry)
            holder.start()
            self.assertTrue(lock_held.wait(1))

            started = time.perf_counter()
            sidecar._drain_pull_result_mailbox(request_id)
            self.assertLess(time.perf_counter() - started, 0.05)
            release_lock.set()
            holder.join(1)

            self.assertIsNone(sidecar.try_claim(handle.key))
            with self.assertRaises(SidecarRejected):
                handle.future.result(timeout=0)
            transport = sidecar.snapshot()["transport"]
            self.assertGreaterEqual(
                transport["transport_stage_dropped"], 1
            )
            self.assertEqual(transport["transport_claim_packets"], 0)
        finally:
            release_lock.set()
            if holder is not None:
                holder.join(1)
            sidecar.close(timeout=3)

    def test_pull_staging_bounded_drain_handles_wrong_and_stale_events(self) -> None:
        async def executor(invocation: Invocation) -> str:
            return str(invocation.arguments["url"])

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=2,
            max_pending=2,
            result_capacity=1,
            result_ttl_s=0.040,
            pull_result_staging=True,
            autostart=True,
        )
        try:
            handles = sidecar.try_submit_batch(
                (
                    (_visit("pull-wrong"), "wrong", "decision", 2.0, ""),
                    (_visit("pull-exact"), "exact", "decision", 1.0, ""),
                ),
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertEqual(len(handles), 2)
            # Both terminals must be in the kernel mailbox before the exact
            # lookup; select is level-triggered, so wait briefly for the second.
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )
            time.sleep(0.015)
            claimed = sidecar.try_claim(handles[1].key)
            self.assertIs(claimed, handles[1])
            self.assertIn("pull-exact", handles[1].future.result(timeout=0))
            self.assertFalse(handles[0].future.done())
            self.assertFalse(sidecar.bridge_started)

            # A retained wrong event is still exact-key fenced. Tombstoning it
            # retires only the observer and never cancels physical work.
            self.assertTrue(
                sidecar.try_tombstone(
                    session_id="wrong", decision_id="decision"
                )
            )
            with self.assertRaises(SidecarTombstoned):
                handles[0].future.result(timeout=0)

            stale = sidecar.try_submit(
                _visit("pull-stale"),
                session_id="stale",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(stale)
            assert stale is not None
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )
            time.sleep(0.055)
            began = time.perf_counter()
            self.assertIsNone(sidecar.try_claim(stale.key))
            self.assertLess(time.perf_counter() - began, 0.05)
            with self.assertRaises(SidecarRejected):
                stale.future.result(timeout=0)
        finally:
            sidecar.close(timeout=3)

    def test_pull_staging_snapshot_and_close_start_lifecycle_bridge(self) -> None:
        async def executor(_: Invocation) -> str:
            return "snapshot-ready"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            pull_result_staging=True,
            autostart=True,
        )
        handle = sidecar.try_submit(
            _visit("pull-snapshot"),
            session_id="session",
            decision_id="decision",
            priority=1.0,
            start_deadline=time.monotonic() + 1.0,
        )
        self.assertIsNotNone(handle)
        assert handle is not None
        _wait_until(
            lambda: bool(select.select([sidecar._event_parent], [], [], 0)[0])
        )
        self.assertFalse(sidecar.bridge_started)
        snapshot = sidecar.snapshot(timeout=2)
        self.assertTrue(snapshot["bridge_started"])
        self.assertEqual(snapshot["parent_staging"]["mode"], "pull")
        claimed = sidecar.try_claim(handle.key)
        self.assertIs(claimed, handle)
        self.assertEqual(handle.future.result(timeout=0), "snapshot-ready")
        sidecar.close(timeout=3)
        self.assertFalse(sidecar._process.is_alive())

    def test_pull_startup_snapshot_certifies_child_without_bridge(self) -> None:
        async def executor(_: Invocation) -> str:
            return "startup-certified"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            pull_result_staging=True,
            autostart=True,
        )
        try:
            certificate = sidecar.startup_snapshot(timeout=2)
            self.assertFalse(sidecar.bridge_started)
            self.assertFalse(certificate["bridge_started"])
            self.assertEqual(certificate["parent_staging"]["mode"], "pull")
            self.assertEqual(certificate["process_pid"], sidecar.pid)
            self.assertIsNotNone(certificate["actual_cpu_affinity"])
            self.assertEqual(
                certificate["requested_scheduler_policy"], "SCHED_IDLE"
            )

            handle = sidecar.try_submit(
                _visit("pull-after-startup-certificate"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            with self.assertRaises(SidecarRejected):
                sidecar.startup_snapshot(timeout=0.1)
            _wait_until(
                lambda: bool(
                    select.select([sidecar._event_parent], [], [], 0)[0]
                )
            )
            self.assertIs(sidecar.try_claim(handle.key), handle)
            self.assertEqual(
                handle.future.result(timeout=0), "startup-certified"
            )
            self.assertFalse(sidecar.bridge_started)
        finally:
            sidecar.close(timeout=3)

    def test_pull_staging_parent_registry_is_hard_bounded(self) -> None:
        async def executor(_: Invocation) -> str:
            return "ready"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            result_capacity=1,
            pull_result_staging=True,
            autostart=True,
        )
        replacement = None
        try:
            first = sidecar.try_submit(
                _visit("pull-capacity-first"),
                session_id="first",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(first)
            self.assertIsNone(
                sidecar.try_submit(
                    _visit("pull-capacity-overflow"),
                    session_id="overflow",
                    decision_id="decision",
                    priority=1.0,
                    start_deadline=time.monotonic() + 1.0,
                )
            )
            self.assertTrue(
                sidecar.try_tombstone(
                    session_id="first", decision_id="decision"
                )
            )
            replacement = sidecar.try_submit(
                _visit("pull-capacity-reused"),
                session_id="replacement",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(replacement)
            snapshot = sidecar.snapshot()
            self.assertEqual(
                snapshot["capacity"]["pull_registry_capacity"], 1
            )
            self.assertEqual(
                snapshot["transport"]["transport_pull_registry_full"], 1
            )
        finally:
            if replacement is not None:
                sidecar.try_tombstone(
                    session_id="replacement", decision_id="decision"
                )
            sidecar.close(timeout=3)

    def test_all_wrong_lease_path_keeps_parent_bridge_lazy(self) -> None:
        context = multiprocessing.get_context("fork")
        completed = context.Event()

        async def executor(_: Invocation) -> str:
            completed.set()
            return "wrong"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            claim_grace_s=0.010,
            autostart=True,
        )
        deadline = time.monotonic() + 0.020
        fresh = None
        try:
            wrong = sidecar.try_submit(
                _visit("lazy-wrong"),
                session_id="wrong-session",
                decision_id="wrong-decision",
                priority=1.0,
                start_deadline=deadline,
            )
            self.assertIsNotNone(wrong)
            self.assertTrue(completed.wait(1))
            while time.monotonic() <= deadline + 0.025:
                time.sleep(0.001)
            self.assertIsNone(sidecar._bridge)

            # The next admission reaps the parent half without starting a
            # result reader or sending a tombstone packet.
            fresh = sidecar.try_submit(
                _visit("after-lazy-reap"),
                session_id="fresh-session",
                decision_id="fresh-decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(fresh)
            self.assertIsNone(sidecar._bridge)
            assert wrong is not None
            with self.assertRaises(SidecarExpired):
                wrong.future.result(timeout=1)

            snapshot = sidecar.snapshot()
            self.assertTrue(snapshot["bridge_started"])
            self.assertEqual(snapshot["lease"]["expired"], 1)
            self.assertEqual(snapshot["transport"]["transport_terminal"], 0)
            self.assertEqual(
                snapshot["transport"]["transport_tombstone_packets"], 0
            )
        finally:
            if fresh is not None:
                sidecar.try_tombstone(
                    session_id=fresh.key.session_id,
                    decision_id=fresh.key.decision_id,
                )
            sidecar.close(timeout=3)

    def test_eager_staging_is_private_until_exact_local_claim(self) -> None:
        async def executor(invocation: Invocation) -> dict[str, object]:
            return {
                "key": invocation.key,
                "pid": os.getpid(),
            }

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            eager_result_staging=True,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("eager-ready"),
                session_id="session",
                decision_id="decision",
                context_token="context-v1",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None
            public_completion = threading.Event()
            handle.future.add_done_callback(
                lambda _: public_completion.set()
            )

            def parent_stage_is_ready() -> bool:
                with sidecar._registry_lock:
                    return len(sidecar._staged_results) == 1

            # Do not use snapshot to wake the child transport: eager transfer
            # must publish promptly on executor completion by itself.
            _wait_until(parent_stage_is_ready, timeout=0.5)

            # Bridge/staging work is private: merely finishing speculation does
            # not wake an observer of the public handle.
            self.assertTrue(sidecar.bridge_started)
            self.assertFalse(handle.future.done())
            self.assertFalse(public_completion.is_set())
            wrong = ExactSpeculationKey.from_invocation(
                _visit("eager-ready"),
                session_id="session",
                decision_id="decision",
                context_token="context-v2",
            )
            self.assertIsNone(sidecar.try_claim(wrong))
            self.assertFalse(handle.future.done())
            self.assertEqual(
                sidecar.snapshot()["parent_staging"]["ready"],
                1,
            )

            started = time.perf_counter()
            claimed = sidecar.try_claim(handle.key)
            elapsed = time.perf_counter() - started
            self.assertIs(claimed, handle)
            self.assertLess(elapsed, 0.05)
            self.assertTrue(public_completion.is_set())
            self.assertEqual(handle.future.result(timeout=0), {
                "key": handle.invocation_key,
                "pid": sidecar.pid,
            })

            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["parent_staging"]["ready"], 0)
            self.assertTrue(snapshot["parent_staging"]["enabled"])
            self.assertEqual(
                snapshot["transport"]["transport_claim_packets"],
                0,
            )
            self.assertEqual(
                snapshot["transport"]["transport_eager_hits"],
                1,
            )
            self.assertEqual(snapshot["stats"]["claims"], 0)
            self.assertEqual(
                snapshot["eager_result_staging"]["result_events"],
                1,
            )
        finally:
            sidecar.close(timeout=3)

    def test_eager_cancelled_observer_is_atomic_fail_open_miss(self) -> None:
        async def executor(_: Invocation) -> str:
            return "privately-staged"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            eager_result_staging=True,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("eager-cancelled-observer"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None

            def parent_stage_is_ready() -> bool:
                with sidecar._registry_lock:
                    return len(sidecar._staged_results) == 1

            _wait_until(parent_stage_is_ready, timeout=0.5)
            self.assertTrue(handle.future.cancel())

            # Cancellation can race confirmation, but must never let
            # InvalidStateError escape onto the authority path.
            self.assertIsNone(sidecar.try_claim(handle.key))
            self.assertTrue(handle.future.cancelled())
            with sidecar._registry_lock:
                self.assertEqual(sidecar._staged_results, {})
                self.assertNotIn(handle.key, sidecar._available)
                self.assertNotIn(handle, sidecar._handles.values())
            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_claim_packets"], 0)
            self.assertEqual(transport["transport_claim_misses"], 1)
        finally:
            sidecar.close(timeout=3)

    def test_eager_staged_result_still_obeys_finite_claim_lease(self) -> None:
        async def executor(_: Invocation) -> str:
            return "staged-before-expiry"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            claim_grace_s=0.010,
            eager_result_staging=True,
            autostart=True,
        )
        deadline = time.monotonic() + 0.030
        try:
            handle = sidecar.try_submit(
                _visit("eager-lease"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=deadline,
            )
            self.assertIsNotNone(handle)
            assert handle is not None

            def parent_stage_is_ready() -> bool:
                with sidecar._registry_lock:
                    return len(sidecar._staged_results) == 1

            _wait_until(parent_stage_is_ready, timeout=0.5)
            self.assertFalse(handle.future.done())
            while time.monotonic() <= deadline + 0.015:
                time.sleep(0.001)

            self.assertIsNone(sidecar.try_claim(handle.key))
            with self.assertRaises(SidecarExpired):
                handle.future.result(timeout=0)
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["parent_staging"]["ready"], 0)
            self.assertEqual(
                snapshot["transport"]["transport_claim_expired"],
                1,
            )
            self.assertEqual(
                snapshot["transport"]["transport_claim_packets"],
                0,
            )
        finally:
            sidecar.close(timeout=3)

    def test_eager_not_ready_claim_is_immediate_non_cancelling_miss(self) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()
        release = context.Event()
        physically_finished = context.Event()
        cancelled = context.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            try:
                while not release.is_set():
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            physically_finished.set()
            return "too-late-for-confirmation"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            eager_result_staging=True,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("eager-running"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            self.assertTrue(started.wait(1))
            assert handle is not None

            began = time.perf_counter()
            self.assertIsNone(sidecar.try_claim(handle.key))
            self.assertLess(time.perf_counter() - began, 0.05)
            with self.assertRaises(SidecarRejected):
                handle.future.result(timeout=0)

            release.set()
            self.assertTrue(physically_finished.wait(1))
            _wait_until(
                lambda: sidecar.snapshot()["counts"]["running"] == 0
            )
            snapshot = sidecar.snapshot()
            self.assertFalse(cancelled.is_set())
            self.assertEqual(snapshot["parent_staging"]["ready"], 0)
            self.assertEqual(
                snapshot["transport"]["transport_claim_packets"],
                0,
            )
            self.assertEqual(
                snapshot["transport"]["transport_eager_not_ready"],
                1,
            )
        finally:
            release.set()
            sidecar.close(timeout=3)

    def test_eager_parent_staging_is_bounded_by_result_capacity(self) -> None:
        async def executor(invocation: Invocation) -> str:
            return str(invocation.arguments["url"])

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=2,
            max_pending=2,
            result_capacity=1,
            eager_result_staging=True,
            autostart=True,
        )
        handles: tuple[SpeculativeHandle, ...] = ()
        try:
            handles = sidecar.try_submit_batch(
                tuple(
                    (
                        _visit(f"bounded-{index}"),
                        f"session-{index}",
                        "decision",
                        1.0,
                        "",
                    )
                    for index in range(2)
                ),
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertEqual(len(handles), 2)
            _wait_until(
                lambda: (
                    sidecar.snapshot()["transport"][
                        "transport_staged_results"
                    ]
                    + sidecar.snapshot()["transport"][
                        "transport_stage_dropped"
                    ]
                    >= 2
                )
            )
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["parent_staging"]["capacity"], 1)
            self.assertEqual(snapshot["parent_staging"]["ready"], 1)
            self.assertTrue(all(not handle.future.done() for handle in handles))

            claims = [sidecar.try_claim(handle.key) for handle in handles]
            self.assertEqual(sum(claim is not None for claim in claims), 1)
            for handle, claim in zip(handles, claims):
                if claim is None:
                    with self.assertRaises(SidecarRejected):
                        handle.future.result(timeout=0)
                else:
                    self.assertIn("bounded-", handle.future.result(timeout=0))
            final = sidecar.snapshot()
            self.assertEqual(final["parent_staging"]["ready"], 0)
            self.assertEqual(
                final["transport"]["transport_staged_results"],
                1,
            )
            self.assertEqual(
                final["transport"]["transport_stage_dropped"],
                1,
            )
            self.assertEqual(
                final["transport"]["transport_claim_packets"],
                0,
            )
        finally:
            sidecar.close(timeout=3)

    def test_eager_capacity_prune_fully_retires_expired_registry(self) -> None:
        context = multiprocessing.get_context("fork")
        second_started = context.Event()
        release_second = context.Event()

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["url"]).rsplit("/", 1)[-1]
            if name == "stage-second":
                second_started.set()
                while not release_second.is_set():
                    await asyncio.sleep(0.001)
            return name

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=2,
            max_pending=2,
            result_capacity=1,
            result_ttl_s=0.030,
            eager_result_staging=True,
            autostart=True,
        )
        handles: tuple[SpeculativeHandle, ...] = ()
        try:
            handles = sidecar.try_submit_batch(
                (
                    (
                        _visit("stage-first"),
                        "session-first",
                        "decision",
                        2.0,
                        "",
                    ),
                    (
                        _visit("stage-second"),
                        "session-second",
                        "decision",
                        1.0,
                        "",
                    ),
                ),
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertEqual(len(handles), 2)
            self.assertTrue(second_started.wait(1))

            def first_stage_is_ready() -> bool:
                with sidecar._registry_lock:
                    return len(sidecar._staged_results) == 1

            _wait_until(first_stage_is_ready, timeout=0.5)
            with sidecar._registry_lock:
                first_valid_until = next(
                    iter(sidecar._staged_results.values())
                ).valid_until
                second_request_id = sidecar._available[handles[1].key]
            while time.monotonic() <= first_valid_until:
                time.sleep(0.001)

            release_second.set()

            def only_second_stage_is_ready() -> bool:
                with sidecar._registry_lock:
                    return tuple(sidecar._staged_results) == (
                        second_request_id,
                    )

            _wait_until(only_second_stage_is_ready, timeout=0.5)
            with self.assertRaises(SidecarExpired):
                handles[0].future.result(timeout=0)
            with sidecar._registry_lock:
                self.assertNotIn(handles[0].key, sidecar._available)
                self.assertNotIn(handles[0], sidecar._handles.values())
                self.assertEqual(len(sidecar._handles), 1)

            claimed = sidecar.try_claim(handles[1].key)
            self.assertIs(claimed, handles[1])
            self.assertEqual(handles[1].future.result(timeout=0), "stage-second")
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["parent_staging"]["ready"], 0)
            self.assertEqual(snapshot["transport"]["transport_claim_packets"], 0)
            self.assertEqual(snapshot["transport"]["transport_lease_reaped"], 1)
        finally:
            release_second.set()
            sidecar.close(timeout=3)

    def test_unclaimed_executor_failure_is_silent_until_parent_reap(self) -> None:
        context = multiprocessing.get_context("fork")
        executed = context.Event()

        async def executor(_: Invocation) -> str:
            executed.set()
            raise RuntimeError("wrong speculation failed")

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            claim_grace_s=0.010,
            autostart=True,
        )
        deadline = time.monotonic() + 0.020
        fresh = None
        try:
            failed = sidecar.try_submit(
                _visit("silent-failure"),
                session_id="failed-session",
                decision_id="failed-decision",
                priority=1.0,
                start_deadline=deadline,
            )
            self.assertIsNotNone(failed)
            self.assertTrue(executed.wait(1))
            while time.monotonic() <= deadline + 0.025:
                time.sleep(0.001)
            self.assertIsNone(sidecar._bridge)

            fresh = sidecar.try_submit(
                _visit("fresh-after-failure"),
                session_id="fresh-session",
                decision_id="fresh-decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            assert failed is not None
            with self.assertRaises(SidecarExpired):
                failed.future.result(timeout=1)
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["transport"]["transport_terminal"], 0)
        finally:
            if fresh is not None:
                sidecar.try_tombstone(
                    session_id=fresh.key.session_id,
                    decision_id=fresh.key.decision_id,
                )
            sidecar.close(timeout=3)

    def test_close_starts_lazy_bridge_and_tolerates_prior_child_crash(self) -> None:
        async def executor(_: Invocation) -> str:
            os._exit(17)

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            autostart=True,
        )
        handle = sidecar.try_submit(
            _visit("crash-child"),
            session_id="session",
            decision_id="decision",
            priority=1.0,
            start_deadline=time.monotonic() + 1.0,
        )
        self.assertIsNotNone(handle)
        process = sidecar._process
        assert process is not None
        _wait_until(lambda: not process.is_alive())
        self.assertIsNone(sidecar._bridge)

        with self.assertRaises((TimeoutError, SidecarClosed)):
            sidecar.snapshot(timeout=0.5)
        sidecar.close(timeout=2)

        self.assertIsNotNone(sidecar._bridge)
        self.assertFalse(process.is_alive())
        assert handle is not None
        with self.assertRaises(SidecarClosed):
            handle.future.result(timeout=1)

    def test_lease_sum_is_finite_and_child_timeout_is_capped(self) -> None:
        async def executor(_: Invocation) -> str:
            return "unused"

        invalid = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            claim_grace_s=1e308,
            autostart=True,
        )
        try:
            with self.assertRaises(ValueError):
                invalid.try_submit(
                    _visit("overflow"),
                    session_id="session",
                    decision_id="decision",
                    priority=1.0,
                    start_deadline=1e308,
                )
            process = invalid._process
            assert process is not None
            self.assertTrue(process.is_alive())
        finally:
            invalid.close(timeout=2)

        huge = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            autostart=True,
        )
        try:
            handle = huge.try_submit(
                _visit("huge-finite"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=1e100,
            )
            self.assertIsNotNone(handle)
            time.sleep(0.020)
            process = huge._process
            assert process is not None
            self.assertTrue(process.is_alive())
        finally:
            huge.close(timeout=2)

    def test_finite_leases_cleanup_without_parent_tombstone_packets(self) -> None:
        async def executor(invocation: Invocation) -> str:
            await asyncio.sleep(0.002)
            return str(invocation.arguments["url"])

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=2,
            max_pending=8,
            claim_grace_s=0.010,
            autostart=True,
        )
        deadline = time.monotonic() + 0.025
        handles = sidecar.try_submit_batch(
            tuple(
                (
                    _visit(f"leased-{index}"),
                    f"session-{index}",
                    "decision",
                    float(8 - index),
                    "",
                )
                for index in range(8)
            ),
            start_deadline=deadline,
        )
        fresh = None
        try:
            self.assertEqual(len(handles), 8)
            _wait_until(
                lambda: sidecar.snapshot()["lease"]["expired"] == 8
            )
            child_snapshot = sidecar.snapshot()
            self.assertEqual(child_snapshot["lease"]["live_finite"], 0)
            self.assertEqual(child_snapshot["counts"]["pending"], 0)
            self.assertEqual(
                child_snapshot["transport"]["transport_tombstone_packets"],
                0,
            )
            self.assertEqual(
                child_snapshot["transport"]["transport_terminal"], 0
            )
            # Child expiry is intentionally silent; parent handles are reaped
            # together at the next admission, outside confirmation.
            self.assertTrue(all(not handle.future.done() for handle in handles))

            fresh = sidecar.try_submit(
                _visit("fresh"),
                session_id="fresh-session",
                decision_id="fresh-decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(fresh)
            for handle in handles:
                with self.assertRaises(SidecarExpired):
                    handle.future.result(timeout=1)
            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_lease_reaped"], 8)
            self.assertEqual(transport["transport_tombstone_packets"], 0)
        finally:
            if fresh is not None:
                sidecar.try_tombstone(
                    session_id=fresh.key.session_id,
                    decision_id=fresh.key.decision_id,
                )
            sidecar.close(timeout=3)

    def test_running_exact_can_claim_during_grace_after_start_deadline(self) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()
        release = context.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return "exact"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            claim_grace_s=0.100,
            autostart=True,
        )
        deadline = time.monotonic() + 0.020
        try:
            handle = sidecar.try_submit(
                _visit("grace"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=deadline,
            )
            self.assertIsNotNone(handle)
            self.assertTrue(started.wait(1))
            while time.monotonic() <= deadline + 0.020:
                time.sleep(0.001)

            assert handle is not None
            self.assertIs(sidecar.try_claim(handle.key), handle)
            release.set()
            self.assertEqual(handle.future.result(timeout=2), "exact")
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["transport"]["transport_claims"], 1)
            self.assertEqual(
                snapshot["transport"]["transport_tombstone_packets"], 0
            )
            self.assertEqual(snapshot["lease"]["expired"], 0)
        finally:
            release.set()
            sidecar.close(timeout=3)

    def test_exact_claim_after_lease_expiry_is_o1_fail_open(self) -> None:
        async def executor(_: Invocation) -> str:
            await asyncio.sleep(0.002)
            return "too-late"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            claim_grace_s=0.010,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("expired-exact"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 0.025,
            )
            self.assertIsNotNone(handle)
            _wait_until(
                lambda: sidecar.snapshot()["lease"]["expired"] == 1
            )

            assert handle is not None
            self.assertIsNone(sidecar.try_claim(handle.key))
            with self.assertRaises(SidecarExpired):
                handle.future.result(timeout=1)
            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_claims"], 0)
            self.assertEqual(transport["transport_claim_expired"], 1)
            self.assertEqual(transport["transport_tombstone_packets"], 0)
            self.assertEqual(transport["transport_terminal"], 0)
        finally:
            sidecar.close(timeout=3)

    def test_submit_batch_uses_one_packet_and_preserves_exact_handles(self) -> None:
        async def executor(invocation: Invocation) -> str:
            await asyncio.sleep(0.005)
            return str(invocation.arguments["url"]).rsplit("/", 1)[-1]

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=2,
            max_pending=3,
            autostart=True,
        )
        handles = ()
        try:
            handles = sidecar.try_submit_batch(
                tuple(
                    (
                        _visit(f"batch-{index}"),
                        f"session-{index}",
                        f"decision-{index}",
                        float(3 - index),
                        "context-v1",
                    )
                    for index in range(3)
                ),
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertEqual(len(handles), 3)
            self.assertEqual(
                [handle.key.session_id for handle in handles],
                ["session-0", "session-1", "session-2"],
            )

            exact = sidecar.try_claim(handles[1].key)
            self.assertIs(exact, handles[1])
            assert exact is not None
            self.assertEqual(exact.future.result(timeout=2), "batch-1")

            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_submitted"], 3)
            self.assertEqual(transport["transport_submit_packets"], 1)
        finally:
            for handle in handles:
                if not handle.claimed:
                    sidecar.try_tombstone(
                        session_id=handle.key.session_id,
                        decision_id=handle.key.decision_id,
                    )
            sidecar.close(timeout=3)

    def test_submit_batch_send_failure_rolls_back_prior_exact_mapping(self) -> None:
        async def executor(invocation: Invocation) -> str:
            await asyncio.sleep(0.005)
            return str(invocation.arguments["url"])

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=2,
            max_packet_bytes=4096,
            autostart=True,
        )
        try:
            invocation = _visit("prior")
            prior = sidecar.try_submit(
                invocation,
                session_id="session",
                decision_id="decision",
                priority=1.0,
            )
            self.assertIsNotNone(prior)
            failed = sidecar.try_submit_batch(
                (
                    (
                        _visit("rolled-back"),
                        "rolled-back-session",
                        "rolled-back-decision",
                        2.0,
                        "",
                    ),
                    (
                        Invocation("visit", {"url": "x" * 8192}),
                        "huge-session",
                        "huge-decision",
                        1.0,
                        "",
                    ),
                )
            )
            self.assertEqual(failed, ())

            assert prior is not None
            exact = sidecar.try_claim(prior.key)
            self.assertIs(exact, prior)
            self.assertIn("prior", prior.future.result(timeout=2))
            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_submitted"], 1)
            self.assertEqual(transport["transport_submit_packets"], 1)
            self.assertEqual(transport["transport_ingress_full"], 1)
        finally:
            sidecar.close(timeout=3)

    def test_submit_batch_rejects_duplicate_or_live_exact_key(self) -> None:
        async def executor(invocation: Invocation) -> str:
            await asyncio.sleep(0.005)
            return str(invocation.arguments["url"])

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=2,
            autostart=True,
        )
        try:
            invocation = _visit("one-exact-generation")
            entry = (
                invocation,
                "session",
                "decision",
                1.0,
                "context",
            )
            original = sidecar.try_submit_batch((entry,))
            self.assertEqual(len(original), 1)

            # Neither a later packet nor a duplicate member in one packet may
            # shadow the O(1) exact-key registry entry.
            self.assertEqual(sidecar.try_submit_batch((entry,)), ())
            duplicate_batch = sidecar.try_submit_batch((entry, entry))
            self.assertEqual(duplicate_batch, ())

            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_submitted"], 1)
            self.assertEqual(transport["transport_submit_packets"], 1)
            self.assertEqual(transport["transport_ingress_full"], 0)

            exact = sidecar.try_claim(original[0].key)
            self.assertIs(exact, original[0])
            self.assertIn(
                "one-exact-generation",
                original[0].future.result(timeout=2),
            )
        finally:
            sidecar.close(timeout=3)

    def test_scheduled_batch_does_not_start_early_and_claims_after_release(
        self,
    ) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()
        started_at = context.Value("d", 0.0)

        async def executor(_: Invocation) -> str:
            with started_at.get_lock():
                started_at.value = time.monotonic()
            started.set()
            return "scheduled-exact"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            max_scheduled_pending=1,
            autostart=True,
        )
        release_at = time.monotonic() + 0.080
        handles: tuple[SpeculativeHandle, ...] = ()
        try:
            handles = sidecar.try_schedule_batch(
                ((_visit("scheduled"), "session", "decision", 1.0, ""),),
                release_at=release_at,
                start_deadline=release_at + 0.5,
            )
            self.assertEqual(len(handles), 1)
            # A preload is not claimable before its execution release fence;
            # the exact mapping remains available for an authority retry.
            self.assertIsNone(sidecar.try_claim(handles[0].key))
            self.assertFalse(sidecar.bridge_started)
            self.assertFalse(started.wait(0.030))
            self.assertTrue(started.wait(1))
            with started_at.get_lock():
                observed_start = started_at.value
            self.assertGreaterEqual(observed_start, release_at)

            claimed = sidecar.try_claim(handles[0].key)
            self.assertIs(claimed, handles[0])
            self.assertEqual(
                handles[0].future.result(timeout=2), "scheduled-exact"
            )
            snapshot = sidecar.snapshot()
            self.assertEqual(
                snapshot["transport"]["transport_schedule_packets"], 1
            )
            self.assertEqual(snapshot["scheduled"]["released_candidates"], 1)
        finally:
            sidecar.close(timeout=3)

    def test_scheduled_tombstone_and_close_are_pre_release_fences(self) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            return "must-not-run"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            max_scheduled_pending=2,
            autostart=True,
        )
        release_at = time.monotonic() + 0.100
        first = sidecar.try_schedule_batch(
            ((_visit("cancelled"), "s1", "d1", 1.0, ""),),
            release_at=release_at,
            start_deadline=release_at + 0.3,
        )
        try:
            self.assertEqual(len(first), 1)
            self.assertTrue(
                sidecar.try_tombstone(session_id="s1", decision_id="d1")
            )
            with self.assertRaises(SidecarTombstoned):
                first[0].future.result(timeout=1)
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["scheduled"]["tombstoned"], 1)
            self.assertEqual(snapshot["scheduled"]["pending"], 0)
            self.assertFalse(started.wait(0.130))

            second_release = time.monotonic() + 0.150
            second = sidecar.try_schedule_batch(
                ((_visit("closed"), "s2", "d2", 1.0, ""),),
                release_at=second_release,
                start_deadline=second_release + 0.3,
            )
            self.assertEqual(len(second), 1)
            sidecar.close(timeout=3)
            with self.assertRaises(SidecarClosed):
                second[0].future.result(timeout=1)
            final = sidecar.snapshot()
            self.assertEqual(final["scheduled"]["closed_unreleased"], 1)
            self.assertEqual(final["scheduled"]["pending"], 0)
            self.assertFalse(started.wait(0.180))
        finally:
            sidecar.close(timeout=3)

    def test_scheduled_deadline_is_enforced_behind_running_work(self) -> None:
        context = multiprocessing.get_context("fork")
        blocker_started = context.Event()
        candidate_started = context.Event()
        release_blocker = context.Event()

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["url"]).rsplit("/", 1)[-1]
            if name == "blocker":
                blocker_started.set()
                while not release_blocker.is_set():
                    await asyncio.sleep(0.001)
            else:
                candidate_started.set()
            return name

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=2,
            max_scheduled_pending=1,
            claim_grace_s=0.010,
            autostart=True,
        )
        try:
            blocker = sidecar.try_submit(
                _visit("blocker"),
                session_id="blocker-session",
                decision_id="blocker-decision",
                priority=2.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(blocker)
            self.assertTrue(blocker_started.wait(1))
            release_at = time.monotonic() + 0.030
            scheduled = sidecar.try_schedule_batch(
                ((_visit("expired"), "session", "decision", 1.0, ""),),
                release_at=release_at,
                start_deadline=release_at + 0.030,
            )
            self.assertEqual(len(scheduled), 1)
            time.sleep(0.090)
            release_blocker.set()
            time.sleep(0.040)
            self.assertFalse(candidate_started.is_set())
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["scheduled"]["released_candidates"], 1)
            self.assertEqual(
                snapshot["stats"]["expired_queued"]
                + snapshot["stats"]["tombstoned_queued"],
                1,
            )
            self.assertEqual(snapshot["lease"]["expired"], 1)
        finally:
            release_blocker.set()
            sidecar.close(timeout=3)

    def test_scheduled_capacity_and_exact_key_registry_are_bounded(self) -> None:
        async def executor(_: Invocation) -> str:
            return "unused"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            max_scheduled_pending=1,
            autostart=True,
        )
        release_at = time.monotonic() + 0.200
        try:
            first = sidecar.try_schedule_batch(
                ((_visit("one"), "s1", "d1", 1.0, ""),),
                release_at=release_at,
                start_deadline=release_at + 0.5,
            )
            self.assertEqual(len(first), 1)
            self.assertIsNone(sidecar.try_claim(first[0].key))
            rejected = sidecar.try_schedule_batch(
                ((_visit("two"), "s2", "d2", 1.0, ""),),
                release_at=release_at + 0.020,
                start_deadline=release_at + 0.5,
            )
            self.assertEqual(rejected, ())
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["scheduled"]["pending"], 1)
            self.assertEqual(snapshot["scheduled"]["capacity_dropped"], 0)
            self.assertEqual(
                snapshot["transport"][
                    "transport_schedule_capacity_rejected"
                ],
                1,
            )

            duplicate = sidecar.try_schedule_batch(
                (
                    (_visit("dup"), "same", "decision", 2.0, ""),
                    (_visit("dup"), "same", "decision", 1.0, ""),
                ),
                release_at=release_at + 0.030,
                start_deadline=release_at + 0.5,
            )
            self.assertEqual(duplicate, ())
        finally:
            sidecar.try_tombstone(session_id="s1", decision_id="d1")
            sidecar.close(timeout=3)

    def test_scheduled_heap_compacts_cancelled_future_batches(self) -> None:
        async def executor(_: Invocation) -> str:
            return "unused"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            max_scheduled_pending=2,
            autostart=True,
        )
        base = time.monotonic()
        try:
            keeper = sidecar.try_schedule_batch(
                ((_visit("keeper"), "keeper", "decision", 1.0, ""),),
                release_at=base + 0.500,
                start_deadline=base + 1.0,
            )
            self.assertEqual(len(keeper), 1)
            for index in range(12):
                transient = sidecar.try_schedule_batch(
                    (
                        (
                            _visit(f"transient-{index}"),
                            f"transient-{index}",
                            "decision",
                            1.0,
                            "",
                        ),
                    ),
                    release_at=base + 0.700,
                    start_deadline=base + 1.0,
                )
                self.assertEqual(len(transient), 1)
                self.assertTrue(
                    sidecar.try_tombstone(
                        session_id=f"transient-{index}",
                        decision_id="decision",
                    )
                )
                snapshot = sidecar.snapshot()
                self.assertEqual(snapshot["scheduled"]["pending"], 1)
                self.assertLessEqual(snapshot["scheduled"]["heap_nodes"], 1)
        finally:
            sidecar.try_tombstone(
                session_id="keeper", decision_id="decision"
            )
            sidecar.close(timeout=3)

    def test_tombstone_batch_is_one_packet_and_retires_locally(self) -> None:
        async def executor(invocation: Invocation) -> str:
            await asyncio.sleep(0.005)
            return str(invocation.arguments["url"]).rsplit("/", 1)[-1]

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=3,
            autostart=True,
        )
        handles = sidecar.try_submit_batch(
            tuple(
                (
                    _visit(f"retire-{index}"),
                    "session",
                    f"decision-{index}",
                    float(3 - index),
                    "",
                )
                for index in range(3)
            )
        )
        try:
            self.assertEqual(len(handles), 3)
            self.assertTrue(
                sidecar.try_tombstone_batch(
                    (("session", "decision-0"), ("session", "decision-1"))
                )
            )
            for handle in handles[:2]:
                with self.assertRaises(SidecarTombstoned):
                    handle.future.result(timeout=1)

            exact = sidecar.try_claim(handles[2].key)
            self.assertIs(exact, handles[2])
            assert exact is not None
            self.assertEqual(exact.future.result(timeout=2), "retire-2")
            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_tombstones"], 2)
            self.assertEqual(transport["transport_tombstone_packets"], 1)
        finally:
            sidecar.close(timeout=3)

    def test_tombstone_batch_send_failure_keeps_handle_claimable(self) -> None:
        async def executor(_: Invocation) -> str:
            await asyncio.sleep(0.005)
            return "kept"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            max_packet_bytes=4096,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("kept"),
                session_id="kept-session",
                decision_id="kept-decision",
                priority=1.0,
            )
            self.assertIsNotNone(handle)
            oversized = tuple(
                (f"unrelated-session-{index}-{'x' * 40}", "decision")
                for index in range(200)
            )
            self.assertFalse(sidecar.try_tombstone_batch(oversized))

            assert handle is not None
            self.assertIs(sidecar.try_claim(handle.key), handle)
            self.assertEqual(handle.future.result(timeout=2), "kept")
            transport = sidecar.snapshot()["transport"]
            self.assertEqual(transport["transport_tombstones"], 0)
            self.assertEqual(transport["transport_tombstone_packets"], 0)
            self.assertEqual(transport["transport_ingress_full"], 1)
        finally:
            sidecar.close(timeout=3)

    def test_tombstone_batch_completes_a_queued_provisional_claim(self) -> None:
        context = multiprocessing.get_context("fork")
        blocker_started = context.Event()
        release = context.Event()

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["url"]).rsplit("/", 1)[-1]
            if name == "blocker":
                blocker_started.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
            return name

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=2,
            autostart=True,
        )
        handles = sidecar.try_submit_batch(
            (
                (_visit("blocker"), "s0", "d0", 2.0, ""),
                (_visit("queued"), "s1", "d1", 1.0, ""),
            ),
            start_deadline=time.monotonic() + 2.0,
        )
        try:
            self.assertEqual(len(handles), 2)
            self.assertTrue(blocker_started.wait(1))
            claimed = sidecar.try_claim(handles[1].key)
            self.assertIs(claimed, handles[1])
            # FIFO snapshot ensures the child has observed the claim while the
            # candidate is still queued behind the blocker.
            self.assertEqual(sidecar.snapshot()["counts"]["queued"], 1)
            self.assertTrue(sidecar.try_tombstone_batch((("s1", "d1"),)))
            assert claimed is not None
            with self.assertRaises(SidecarTombstoned):
                claimed.future.result(timeout=2)
        finally:
            release.set()
            sidecar.close(timeout=3)

    def test_seqpacket_ingress_is_bounded_and_fail_open(self) -> None:
        async def executor(_: Invocation) -> str:
            await asyncio.sleep(0.02)
            return "done"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            ingress_capacity=1,
            autostart=True,
        )
        accepted = 0
        dropped = 0
        try:
            for index in range(1000):
                handle = sidecar.try_submit(
                    _visit(f"bounded-{index}"),
                    session_id=f"session-{index}",
                    decision_id="decision",
                    priority=1.0,
                    start_deadline=time.monotonic() + 0.1,
                )
                if handle is None:
                    dropped += 1
                else:
                    accepted += 1
            self.assertGreater(accepted, 0)
            self.assertGreater(dropped, 0)
            self.assertEqual(
                sidecar.snapshot()["transport"]["transport_ingress_full"],
                dropped,
            )
        finally:
            sidecar.close(timeout=2)

    def test_result_bridge_uses_sidecar_cpu_without_repinning_main(self) -> None:
        original_affinity = set(os.sched_getaffinity(0))
        if len(original_affinity) < 2:
            self.skipTest("bridge affinity isolation requires two CPUs")
        authority_cpu, sidecar_cpu = choose_authority_sidecar_cpus(
            original_affinity
        )

        async def executor(_: Invocation) -> str:
            return "done"

        sidecar = None
        try:
            os.sched_setaffinity(0, {authority_cpu})
            sidecar = ProcessSpeculativeSidecar(
                executor,
                cpu_affinity={sidecar_cpu},
                autostart=True,
            )
            self.assertTrue(sidecar.start_result_bridge(timeout=2.0))
            snapshot = sidecar.snapshot()

            self.assertEqual(os.sched_getaffinity(0), {authority_cpu})
            self.assertEqual(
                snapshot["requested_cpu_affinity"], [sidecar_cpu]
            )
            self.assertEqual(snapshot["actual_cpu_affinity"], [sidecar_cpu])
            self.assertEqual(
                snapshot["requested_bridge_cpu_affinity"], [sidecar_cpu]
            )
            self.assertEqual(
                snapshot["actual_bridge_cpu_affinity"], [sidecar_cpu]
            )
            self.assertTrue(snapshot["bridge_affinity_ready"])
            self.assertIsNone(snapshot["bridge_affinity_error"])

            bridge = sidecar._bridge
            self.assertIsNotNone(bridge)
            assert bridge is not None
            self.assertIsNotNone(bridge.native_id)
            assert bridge.native_id is not None
            self.assertEqual(
                os.sched_getaffinity(bridge.native_id), {sidecar_cpu}
            )
        finally:
            try:
                if sidecar is not None:
                    sidecar.close(timeout=3)
            finally:
                os.sched_setaffinity(0, original_affinity)

    def test_result_bridge_affinity_failure_is_bounded_and_reapable(
        self,
    ) -> None:
        chosen_cpu = min(os.sched_getaffinity(0))

        async def executor(_: Invocation) -> str:
            return "done"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            cpu_affinity={chosen_cpu},
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("bridge-affinity-failure"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            assert handle is not None

            with mock.patch(
                "paste_repro.speculation_sidecar.os.sched_setaffinity",
                side_effect=OSError("bridge affinity denied"),
            ):
                self.assertFalse(sidecar.start_result_bridge(timeout=1.0))

            with self.assertRaises(SidecarClosed):
                handle.future.result(timeout=1.0)
            snapshot = sidecar.snapshot()
            self.assertTrue(snapshot["bridge_affinity_ready"])
            self.assertEqual(
                snapshot["requested_bridge_cpu_affinity"], [chosen_cpu]
            )
            self.assertIsNone(snapshot["actual_bridge_cpu_affinity"])
            self.assertIn(
                "bridge affinity denied", snapshot["bridge_affinity_error"]
            )

            sidecar.close(timeout=3)
            self.assertFalse(sidecar._process.is_alive())
            self.assertFalse(sidecar.snapshot()["process_alive"])
        finally:
            sidecar.close(timeout=3)

    def test_executor_runs_in_fork_child_and_respects_k(self) -> None:
        context = multiprocessing.get_context("fork")
        chosen_cpu = min(os.sched_getaffinity(0))
        release = context.Event()
        active = context.Value("i", 0)
        max_active = context.Value("i", 0)
        executor_pid = context.Value("i", 0)

        async def executor(invocation: Invocation) -> dict[str, object]:
            with active.get_lock():
                active.value += 1
                max_active.value = max(max_active.value, active.value)
                executor_pid.value = os.getpid()
            while not release.is_set():
                await asyncio.sleep(0.001)
            with active.get_lock():
                active.value -= 1
            return {
                "invocation_key": invocation.key,
                "pid": os.getpid(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            }

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=2,
            max_pending=4,
            cpu_affinity={chosen_cpu},
            autostart=True,
        )
        handles = []
        try:
            for index in range(4):
                handle = sidecar.try_submit(
                    _visit(f"process-{index}"),
                    session_id=f"session-{index}",
                    decision_id=f"decision-{index}",
                    priority=float(4 - index),
                    start_deadline=time.monotonic() + 2.0,
                )
                self.assertIsNotNone(handle)
                handles.append(handle)
            _wait_until(lambda: active.value == 2)
            snapshot = sidecar.snapshot()
            self.assertEqual(snapshot["counts"]["running"], 2)
            self.assertEqual(snapshot["max_running"], 2)
            self.assertNotEqual(executor_pid.value, os.getpid())
            self.assertEqual(executor_pid.value, sidecar.pid)

            exact = sidecar.try_claim(handles[0].key)
            self.assertIs(exact, handles[0])
            release.set()
            assert exact is not None
            result = exact.future.result(timeout=2)
            self.assertEqual(result["pid"], sidecar.pid)
            self.assertEqual(result["invocation_key"], exact.invocation_key)
            self.assertEqual(result["cpu_affinity"], [chosen_cpu])
            _wait_until(lambda: sidecar.snapshot()["started"] == 4)
            final_live_snapshot = sidecar.snapshot()
            self.assertEqual(
                final_live_snapshot["requested_cpu_affinity"], [chosen_cpu]
            )
            self.assertEqual(
                final_live_snapshot["actual_cpu_affinity"], [chosen_cpu]
            )
            self.assertEqual(max_active.value, 2)
        finally:
            release.set()
            for handle in handles:
                if not handle.claimed:
                    sidecar.try_tombstone(
                        session_id=handle.key.session_id,
                        decision_id=handle.key.decision_id,
                    )
            sidecar.close(timeout=3)

    def test_unclaimed_success_payload_stays_in_child_until_tombstone(self) -> None:
        async def executor(invocation: Invocation) -> dict[str, object]:
            await asyncio.sleep(0.005)
            return {"invocation_key": invocation.key, "large": "x" * 1000}

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("wrong-unclaimed"),
                session_id="wrong-session",
                decision_id="wrong-decision",
                priority=1.0,
                start_deadline=time.monotonic() + 1.0,
            )
            self.assertIsNotNone(handle)
            _wait_until(lambda: sidecar.snapshot()["completed"] == 1)
            assert handle is not None
            self.assertFalse(handle.future.done())
            self.assertEqual(
                sidecar.snapshot()["transport"]["transport_results"], 0
            )
            self.assertFalse(
                sidecar.snapshot()["parent_staging"]["enabled"]
            )

            self.assertTrue(
                sidecar.try_tombstone(
                    session_id="wrong-session",
                    decision_id="wrong-decision",
                )
            )
            with self.assertRaises(SidecarTombstoned):
                handle.future.result(timeout=1)
            self.assertEqual(
                sidecar.snapshot()["transport"]["transport_results"], 0
            )
        finally:
            sidecar.close(timeout=3)

    def test_process_tombstone_does_not_cancel_running_executor(self) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()
        release = context.Event()
        physically_finished = context.Event()
        cancelled = context.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            try:
                while not release.is_set():
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            physically_finished.set()
            return "finished"

        sidecar = ProcessSpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            autostart=True,
        )
        try:
            handle = sidecar.try_submit(
                _visit("running-process"),
                session_id="session",
                decision_id="decision",
                priority=1.0,
                start_deadline=time.monotonic() + 2.0,
            )
            self.assertIsNotNone(handle)
            self.assertTrue(started.wait(1))
            assert handle is not None
            began = time.perf_counter()
            self.assertTrue(
                sidecar.try_tombstone(
                    session_id="session", decision_id="decision"
                )
            )
            self.assertLess(time.perf_counter() - began, 0.05)
            with self.assertRaises(SidecarTombstoned):
                handle.future.result(timeout=1)
            self.assertEqual(sidecar.snapshot()["counts"]["running"], 1)
            self.assertFalse(physically_finished.is_set())
            self.assertFalse(cancelled.is_set())

            release.set()
            self.assertTrue(physically_finished.wait(1))
            _wait_until(lambda: sidecar.snapshot()["counts"]["running"] == 0)
            self.assertFalse(cancelled.is_set())
        finally:
            release.set()
            sidecar.close(timeout=3)


class SpeculativeSidecarRaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_authority_wins_a_simultaneous_terminal_tie(self) -> None:
        invocation = _visit("tie")
        handle = SpeculativeHandle(
            ExactSpeculationKey.from_invocation(
                invocation,
                session_id="session",
                decision_id="decision",
            )
        )
        handle.future.set_result("speculative-result")
        authority = asyncio.get_running_loop().create_future()
        authority.set_result("authoritative-result")

        result = await race_authority_with_speculation(authority, handle)

        self.assertEqual(result.source, "authoritative")
        self.assertEqual(result.result, "authoritative-result")

    async def test_running_exact_result_wins_without_cancelling_authority(self) -> None:
        release_speculation = threading.Event()
        speculation_started = threading.Event()
        authority_finished = asyncio.Event()

        async def executor(_: Invocation) -> str:
            speculation_started.set()
            while not release_speculation.is_set():
                await asyncio.sleep(0.001)
            return "speculative-result"

        sidecar = SpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=2,
            autostart=True,
        )
        try:
            invocation = _visit("exact")
            submitted = sidecar.try_submit(
                invocation,
                session_id="session",
                decision_id="decision",
                priority=1.0,
                context_token="snapshot-v1",
            )
            self.assertIsNotNone(submitted)
            self.assertTrue(speculation_started.wait(1))

            wrong_key = ExactSpeculationKey.from_invocation(
                invocation,
                session_id="session",
                decision_id="decision",
                context_token="snapshot-v2",
            )
            self.assertIsNone(sidecar.try_claim(wrong_key))
            assert submitted is not None
            claimed = sidecar.try_claim(submitted.key)
            self.assertIs(claimed, submitted)
            self.assertIsNone(sidecar.try_claim(submitted.key))

            async def authority() -> str:
                await asyncio.sleep(0.12)
                authority_finished.set()
                return "authoritative-result"

            authority_task = asyncio.create_task(authority())
            race_task = asyncio.create_task(
                race_authority_with_speculation(authority_task, claimed)
            )
            await asyncio.sleep(0.01)
            release_speculation.set()
            result = await asyncio.wait_for(race_task, timeout=1)
            self.assertEqual(result.source, "speculative")
            self.assertEqual(result.result, "speculative-result")
            self.assertFalse(authority_finished.is_set())

            await asyncio.wait_for(authority_task, timeout=1)
            self.assertTrue(authority_finished.is_set())
        finally:
            release_speculation.set()
            sidecar.close()

    async def test_speculative_failure_waits_for_authority_terminal(self) -> None:
        speculation_started = threading.Event()

        async def executor(_: Invocation) -> str:
            speculation_started.set()
            await asyncio.sleep(0.005)
            raise RuntimeError("speculation failed")

        sidecar = SpeculativeSidecar(
            executor,
            max_workers=1,
            max_pending=1,
            autostart=True,
        )
        try:
            invocation = _visit("failure")
            submitted = sidecar.try_submit(
                invocation,
                session_id="session",
                decision_id="decision",
                priority=1.0,
            )
            self.assertIsNotNone(submitted)
            self.assertTrue(speculation_started.wait(1))
            assert submitted is not None
            claimed = sidecar.try_claim(submitted.key)
            self.assertIsNotNone(claimed)

            async def authority() -> str:
                await asyncio.sleep(0.02)
                return "authoritative-result"

            result = await asyncio.wait_for(
                race_authority_with_speculation(
                    asyncio.create_task(authority()), claimed
                ),
                timeout=1,
            )
            self.assertEqual(result.source, "authoritative")
            self.assertEqual(result.result, "authoritative-result")
        finally:
            sidecar.close()


if __name__ == "__main__":
    unittest.main()
