from __future__ import annotations

import asyncio
import multiprocessing
import os
import time
import unittest

from paste_repro.authority_process_lane import (
    AuthorityProcessLaneError,
    ProcessAuthorityLane,
    RemoteAuthorityError,
)
from paste_repro.invocation import Invocation


def _visit(name: str) -> Invocation:
    return Invocation("visit", {"url": f"https://example.test/{name}"})


@unittest.skipUnless(
    "fork" in multiprocessing.get_all_start_methods()
    and hasattr(os, "sched_getaffinity"),
    "authority process lane requires Linux fork and CPU affinity",
)
class ProcessAuthorityLaneTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.cpu = min(os.sched_getaffinity(0))

    async def test_executes_on_fork_child_and_certifies_affinity(self) -> None:
        parent_pid = os.getpid()

        async def executor(invocation: Invocation) -> dict[str, object]:
            await asyncio.sleep(0.002)
            return {
                "pid": os.getpid(),
                "key": invocation.key,
                "affinity": sorted(os.sched_getaffinity(0)),
            }

        lane = ProcessAuthorityLane(
            executor,
            workers=2,
            visit_capacity=2,
            cpu_affinity={self.cpu},
        )
        lane.start()
        futures = lane.submit_batch(
            (
                (
                    _visit(str(index)),
                    f"session-{index}",
                    time.perf_counter(),
                )
                for index in range(4)
            )
        )
        await lane.barrier()
        completions = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
        snapshot = await lane.aclose()

        self.assertTrue(
            all(completion.result.result["pid"] != parent_pid for completion in completions)
        )
        self.assertTrue(
            all(
                completion.result.result["affinity"] == [self.cpu]
                for completion in completions
            )
        )
        self.assertEqual(snapshot["requested_cpu_affinity"], [self.cpu])
        self.assertEqual(snapshot["actual_cpu_affinity"], [self.cpu])
        self.assertEqual(snapshot["authoritative_state_count"], 4)
        self.assertEqual(snapshot["stats"]["authoritative_requests"], 4)
        self.assertEqual(snapshot["submitted"], 4)
        self.assertEqual(snapshot["completed"], 4)
        self.assertFalse(snapshot["process_alive"])
        self.assertFalse(snapshot["bridge_alive"])
        self.assertTrue(snapshot["fork_started_before_bridge"])
        self.assertEqual(snapshot["ipc_stats"]["submit_batches"], 1)
        self.assertEqual(
            snapshot["ipc_stats"]["submitted_requests"],
            4,
        )
        self.assertEqual(
            snapshot["ipc_stats"]["max_submit_batch_size"],
            4,
        )
        # SubmitBatch, Barrier, Close: request count no longer determines the
        # number of parent-to-child packets.
        self.assertEqual(
            snapshot["ipc_stats"]["command_packets_sent"],
            3,
        )
        self.assertFalse(
            snapshot["ipc_stats"]["result_batching_enabled"]
        )
        self.assertEqual(
            snapshot["ipc_stats"]["result_packets_received"],
            4,
        )
        self.assertEqual(snapshot["ipc_stats"]["result_batches"], 0)

        # Close is idempotent and returns the already-certified snapshot.
        self.assertIs(await lane.aclose(), snapshot)

    async def test_observer_cannot_cancel_authority_and_close_drains_it(self) -> None:
        context = multiprocessing.get_context("fork")
        started = context.Event()
        release = context.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return "committed"

        lane = ProcessAuthorityLane(
            executor,
            workers=1,
            visit_capacity=1,
            cpu_affinity={self.cpu},
        )
        lane.start()
        future = lane.submit(
            _visit("uncancellable"),
            session_id="session",
            scheduled_at=time.perf_counter(),
        )
        self.assertFalse(future.cancel())
        self.assertTrue(await asyncio.to_thread(started.wait, 2.0))

        close_task = asyncio.create_task(lane.aclose())
        await asyncio.sleep(0.020)
        self.assertFalse(close_task.done())
        release.set()

        completion = await asyncio.wrap_future(future)
        snapshot = await close_task
        self.assertEqual(completion.result.result, "committed")
        self.assertEqual(snapshot["authoritative_state_count"], 1)
        self.assertEqual(snapshot["completed"], 1)

    async def test_remote_failure_completes_future_and_lane_continues(self) -> None:
        async def executor(invocation: Invocation) -> str:
            if invocation.arguments["url"].endswith("failure"):
                raise ValueError("expected child failure")
            return "ok"

        lane = ProcessAuthorityLane(
            executor,
            workers=1,
            visit_capacity=1,
            cpu_affinity={self.cpu},
        )
        lane.start()
        failed, succeeded = lane.submit_batch(
            (
                (_visit("failure"), "failed", time.perf_counter()),
                (_visit("success"), "success", time.perf_counter()),
            )
        )

        with self.assertRaises(RemoteAuthorityError) as raised:
            await asyncio.wrap_future(failed)
        self.assertIn("ValueError", raised.exception.remote_type)
        self.assertEqual(
            (await asyncio.wrap_future(succeeded)).result.result,
            "ok",
        )
        snapshot = await lane.aclose()
        self.assertEqual(snapshot["submitted"], 2)
        self.assertEqual(snapshot["completed"], 1)
        self.assertEqual(snapshot["failed"], 1)
        self.assertEqual(snapshot["ipc_stats"]["submit_batches"], 1)
        self.assertEqual(
            snapshot["ipc_stats"]["result_events_received"],
            2,
        )

    async def test_close_fifo_drains_whole_batch_without_cancellation(self) -> None:
        context = multiprocessing.get_context("fork")
        release = context.Event()

        async def executor(invocation: Invocation) -> str:
            while not release.is_set():
                await asyncio.sleep(0.001)
            return invocation.arguments["url"]

        lane = ProcessAuthorityLane(
            executor,
            workers=3,
            visit_capacity=3,
            cpu_affinity={self.cpu},
        )
        lane.start()
        futures = lane.submit_batch(
            tuple(
                (_visit(str(index)), f"session-{index}", time.perf_counter())
                for index in range(3)
            )
        )
        self.assertFalse(futures[0].cancel())

        close_task = asyncio.create_task(lane.aclose())
        await asyncio.sleep(0.020)
        self.assertFalse(close_task.done())
        release.set()

        completions = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
        snapshot = await close_task
        self.assertEqual(len(completions), 3)
        self.assertEqual(snapshot["submitted"], 3)
        self.assertEqual(snapshot["completed"], 3)
        self.assertEqual(snapshot["authoritative_state_count"], 3)
        self.assertEqual(snapshot["ipc_stats"]["submit_batches"], 1)

    async def test_results_are_individual_by_default(self) -> None:
        async def executor(invocation: Invocation) -> str:
            return invocation.arguments["url"]

        lane = ProcessAuthorityLane(
            executor,
            workers=8,
            visit_capacity=8,
            cpu_affinity={self.cpu},
        )
        lane.start()
        futures = lane.submit_batch(
            tuple(
                (_visit(str(index)), f"session-{index}", time.perf_counter())
                for index in range(8)
            )
        )
        await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
        snapshot = await lane.aclose()
        ipc = snapshot["ipc_stats"]

        self.assertEqual(ipc["result_events_received"], 8)
        self.assertFalse(ipc["result_batching_enabled"])
        self.assertEqual(ipc["result_packets_received"], 8)
        self.assertEqual(ipc["result_batches"], 0)
        self.assertEqual(ipc["max_result_batch_size"], 1)
        # Ready + eight individual ResultEvents + Closed.
        self.assertEqual(ipc["event_packets_received"], 10)

    async def test_same_turn_result_batching_is_explicit_opt_in(self) -> None:
        async def executor(invocation: Invocation) -> str:
            return invocation.arguments["url"]

        lane = ProcessAuthorityLane(
            executor,
            workers=8,
            visit_capacity=8,
            cpu_affinity={self.cpu},
            batch_results=True,
        )
        lane.start()
        futures = lane.submit_batch(
            tuple(
                (_visit(str(index)), f"session-{index}", time.perf_counter())
                for index in range(8)
            )
        )
        await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
        snapshot = await lane.aclose()
        ipc = snapshot["ipc_stats"]

        self.assertEqual(ipc["result_events_received"], 8)
        self.assertTrue(ipc["result_batching_enabled"])
        self.assertEqual(ipc["result_packets_received"], 1)
        self.assertEqual(ipc["result_batches"], 1)
        self.assertEqual(ipc["max_result_batch_size"], 8)
        # Ready + ResultBatch + Closed.
        self.assertEqual(ipc["event_packets_received"], 3)

    async def test_unpicklable_result_isolated_within_result_batch(self) -> None:
        async def executor(invocation: Invocation) -> object:
            if invocation.arguments["url"].endswith("unpicklable"):
                return lambda: None
            return "ok"

        lane = ProcessAuthorityLane(
            executor,
            workers=2,
            visit_capacity=2,
            cpu_affinity={self.cpu},
            batch_results=True,
        )
        lane.start()
        unpicklable, succeeded = lane.submit_batch(
            (
                (
                    _visit("unpicklable"),
                    "unpicklable",
                    time.perf_counter(),
                ),
                (_visit("success"), "success", time.perf_counter()),
            )
        )

        with self.assertRaises(RemoteAuthorityError):
            await asyncio.wrap_future(unpicklable)
        self.assertEqual(
            (await asyncio.wrap_future(succeeded)).result.result,
            "ok",
        )
        snapshot = await lane.aclose()
        self.assertEqual(snapshot["completed"], 1)
        self.assertEqual(snapshot["failed"], 1)
        self.assertEqual(
            snapshot["ipc_stats"]["result_events_received"],
            2,
        )

    async def test_batch_validation_is_atomic_and_submit_is_compatible(self) -> None:
        async def executor(_: Invocation) -> str:
            return "ok"

        lane = ProcessAuthorityLane(
            executor,
            workers=1,
            visit_capacity=1,
            cpu_affinity={self.cpu},
        )
        lane.start()
        with self.assertRaises(TypeError):
            lane.submit_batch(
                (
                    (_visit("valid"), "valid", time.perf_counter()),
                    ("not-an-invocation", "invalid", time.perf_counter()),
                )
            )

        future = lane.submit(
            _visit("single"),
            session_id="single",
            scheduled_at=time.perf_counter(),
        )
        self.assertEqual(
            (await asyncio.wrap_future(future)).result.result,
            "ok",
        )
        snapshot = await lane.aclose()
        self.assertEqual(snapshot["submitted"], 1)
        self.assertEqual(snapshot["ipc_stats"]["submit_batches"], 1)
        self.assertEqual(
            snapshot["ipc_stats"]["max_submit_batch_size"],
            1,
        )

    async def test_close_before_start_is_empty_and_prevents_start(self) -> None:
        async def executor(_: Invocation) -> None:
            return None

        lane = ProcessAuthorityLane(
            executor,
            workers=1,
            visit_capacity=1,
            cpu_affinity={self.cpu},
        )
        snapshot = await lane.aclose()
        self.assertEqual(snapshot["submitted"], 0)
        self.assertEqual(snapshot["ipc_stats"]["command_packets_sent"], 0)
        self.assertEqual(snapshot["ipc_stats"]["submit_batches"], 0)
        self.assertFalse(snapshot["process_alive"])
        with self.assertRaises(AuthorityProcessLaneError):
            lane.start()

        with self.assertRaises(TypeError):
            ProcessAuthorityLane(
                executor,
                workers=1,
                visit_capacity=1,
                cpu_affinity={self.cpu},
                batch_results=1,
            )


if __name__ == "__main__":
    unittest.main()
