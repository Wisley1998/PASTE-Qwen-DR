from __future__ import annotations

import asyncio
import base64
import json
import math
import threading
import unittest

from paste_repro.invocation import Invocation
from paste_repro.live_broker import LiveToolBroker
from paste_repro.live_executor import SyncToolMapExecutor, WikipediaLiveExecutor


class LiveToolBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_tool_capacity_is_shared_and_does_not_idle_other_tools(
        self,
    ) -> None:
        release = asyncio.Event()
        four_started = asyncio.Event()
        all_finished = asyncio.Event()
        running = {"search": 0, "visit": 0}
        maximum = {"search": 0, "visit": 0, "total": 0}
        started: list[tuple[str, str]] = []
        finished = 0

        async def executor(invocation: Invocation) -> str:
            nonlocal finished
            tool_name = invocation.tool_name
            name = str(invocation.arguments["name"])
            running[tool_name] += 1
            maximum[tool_name] = max(maximum[tool_name], running[tool_name])
            maximum["total"] = max(maximum["total"], sum(running.values()))
            started.append((tool_name, name))
            if len(started) >= 4:
                four_started.set()
            try:
                await release.wait()
                return name
            finally:
                running[tool_name] -= 1
                finished += 1
                if finished == 6:
                    all_finished.set()

        broker = LiveToolBroker(
            executor,
            max_workers=4,
            max_speculative_workers=4,
            max_speculative_pending=8,
            ttl_s=10,
            service_time_hints_s={"search": 1.0, "visit": 2.0},
            tool_capacities={"search": 3, "visit": 1},
        )
        for index in range(3):
            await broker.speculate(
                Invocation("visit", {"name": f"visit-{index}"}),
                session_id=f"visit-{index}",
            )
        for index in range(3):
            await broker.speculate(
                Invocation("search", {"name": f"search-{index}"}),
                session_id=f"search-{index}",
            )
        await asyncio.wait_for(four_started.wait(), timeout=1)

        snapshot = broker.snapshot()
        self.assertEqual(snapshot["capacity"]["tool_capacities"], {"search": 3, "visit": 1})
        self.assertEqual(snapshot["counts"]["running_by_tool"], {"search": 3, "visit": 1})
        self.assertEqual(snapshot["counts"]["queued_by_tool"], {"visit": 2})
        queued_visits = [
            job
            for job in snapshot["jobs"]
            if job["tool_name"] == "visit" and job["state"] == "queued"
        ]
        self.assertEqual(
            [job["tool_queue_position"] for job in queued_visits], [0, 1]
        )
        self.assertGreater(queued_visits[0]["estimated_tool_queue_s"], 1.9)
        self.assertEqual(
            queued_visits[0]["estimated_queue_s"],
            max(
                queued_visits[0]["estimated_global_queue_s"],
                queued_visits[0]["estimated_tool_queue_s"],
            ),
        )

        release.set()
        await asyncio.wait_for(all_finished.wait(), timeout=1)
        self.assertLessEqual(maximum["total"], 4)
        self.assertLessEqual(maximum["search"], 3)
        self.assertLessEqual(maximum["visit"], 1)
        self.assertEqual(broker.stats.max_running_by_tool, {"search": 3, "visit": 1})
        self.assertGreaterEqual(broker.stats.max_queued_by_tool["visit"], 2)
        records = [record for record in broker.tool_records() if record["admitted"]]
        self.assertTrue(all(record["tool_capacity"] in {1, 3} for record in records))
        self.assertTrue(
            all(
                record["worker_pool"]["tool_capacities"]
                == {"search": 3, "visit": 1}
                for record in records
            )
        )
        await broker.cancel_predictions()
        await broker.close()

    async def test_visit_capacity_is_shared_and_authoritative_is_next(self) -> None:
        release_first = asyncio.Event()
        first_started = asyncio.Event()
        authoritative_started = asyncio.Event()
        order: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            order.append(name)
            if name == "spec-running":
                first_started.set()
                await release_first.wait()
            elif name == "authoritative":
                authoritative_started.set()
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            max_speculative_workers=2,
            max_speculative_pending=4,
            ttl_s=10,
            service_time_hints_s={"visit": 2.0},
            tool_capacities={"visit": 1},
        )
        await broker.speculate(
            Invocation("visit", {"name": "spec-running"}), session_id="running"
        )
        await first_started.wait()
        await broker.speculate(
            Invocation("visit", {"name": "spec-queued"}), session_id="queued"
        )
        authoritative_task = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "authoritative"}), session_id="auth"
            )
        )
        for _ in range(10):
            before = broker.snapshot()
            if before["counts"]["queued_authoritative"] == 1:
                break
            await asyncio.sleep(0)
        self.assertEqual(before["counts"]["running_by_tool"], {"visit": 1})
        self.assertEqual(before["counts"]["queued_by_tool"], {"visit": 2})
        queued_auth = next(
            job
            for job in before["jobs"]
            if job["lane"] == "authoritative" and job["state"] == "queued"
        )
        queued_spec = next(
            job
            for job in before["jobs"]
            if job["lane"] == "speculative" and job["state"] == "queued"
        )
        self.assertEqual(queued_auth["tool_queue_position"], 0)
        self.assertEqual(queued_spec["tool_queue_position"], 1)
        self.assertGreater(queued_spec["estimated_tool_queue_s"], 3.9)

        release_first.set()
        await asyncio.wait_for(authoritative_started.wait(), timeout=1)
        result = await asyncio.wait_for(authoritative_task, timeout=1)
        self.assertEqual(order[:2], ["spec-running", "authoritative"])
        self.assertEqual(result.source, "executed")
        self.assertLessEqual(broker.stats.max_running_by_tool["visit"], 1)
        await broker.cancel_predictions()
        await broker.close()

    async def test_authoritative_lane_uses_reserved_worker_before_queued_speculation(
        self,
    ) -> None:
        release_first = asyncio.Event()
        first_started = asyncio.Event()
        authoritative_started = asyncio.Event()
        calls: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            calls.append(name)
            if name == "spec-running":
                first_started.set()
                await release_first.wait()
            if name == "authoritative":
                authoritative_started.set()
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            max_speculative_workers=1,
            max_speculative_pending=4,
            ttl_s=10,
        )
        await broker.speculate(
            Invocation("visit", {"name": "spec-running"}), session_id="s1"
        )
        await first_started.wait()
        await broker.speculate(
            Invocation("visit", {"name": "spec-queued"}), session_id="s2"
        )
        before = broker.snapshot()
        self.assertEqual(before["counts"]["running_speculative"], 1)
        self.assertEqual(before["counts"]["queued_speculative"], 1)

        committed_task = asyncio.create_task(
            broker.authoritative(
                Invocation("search", {"name": "authoritative"}), session_id="s3"
            )
        )
        await authoritative_started.wait()
        self.assertEqual(calls[:2], ["spec-running", "authoritative"])
        committed = await committed_task
        self.assertEqual(committed.source, "executed")
        self.assertEqual(committed.result, "authoritative")

        release_first.set()
        await broker.cancel_predictions()
        self.assertLessEqual(broker.stats.max_running_total, 2)
        self.assertLessEqual(broker.stats.max_running_speculative, 1)
        await broker.close()

    async def test_minimum_speculative_worker_is_bounded_and_auditable(self) -> None:
        release_auth_one = asyncio.Event()
        release_auth_two = asyncio.Event()
        release_spec = asyncio.Event()
        initial_started = asyncio.Event()
        spec_started = asyncio.Event()
        third_auth_started = asyncio.Event()
        order: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            order.append(name)
            if name == "auth-one":
                if "auth-two" in order:
                    initial_started.set()
                await release_auth_one.wait()
            elif name == "auth-two":
                if "auth-one" in order:
                    initial_started.set()
                await release_auth_two.wait()
            elif name == "spec-reserved":
                spec_started.set()
                await release_spec.wait()
            elif name == "auth-three":
                third_auth_started.set()
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            max_speculative_workers=1,
            min_speculative_workers=1,
            max_speculative_pending=4,
            ttl_s=10,
            tool_capacities={"visit": 2},
        )
        first = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "auth-one"}), session_id="a1"
            )
        )
        second = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "auth-two"}), session_id="a2"
            )
        )
        await asyncio.wait_for(initial_started.wait(), timeout=1)
        third = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "auth-three"}), session_id="a3"
            )
        )
        for _ in range(20):
            if broker.snapshot()["counts"]["queued_authoritative"] == 1:
                break
            await asyncio.sleep(0)
        await broker.speculate(
            Invocation("visit", {"name": "spec-reserved"}), session_id="s1"
        )

        release_auth_one.set()
        await asyncio.wait_for(spec_started.wait(), timeout=1)
        self.assertFalse(third_auth_started.is_set())
        self.assertEqual(
            broker.snapshot()["capacity"]["min_speculative_workers"], 1
        )

        release_spec.set()
        await asyncio.wait_for(third_auth_started.wait(), timeout=1)
        self.assertLess(order.index("spec-reserved"), order.index("auth-three"))
        release_auth_two.set()
        await asyncio.gather(first, second, third)
        await asyncio.sleep(0)
        self.assertEqual(broker.stats.reserved_speculative_dispatches, 1)
        speculative_record = next(
            record
            for record in broker.tool_records()
            if record["session_id"] == "s1"
        )
        self.assertEqual(
            speculative_record["worker_pool"]["min_speculative_workers"], 1
        )
        await broker.cancel_predictions()
        await broker.close()

    async def test_minimum_speculative_worker_preserves_authoritative_capacity(
        self,
    ) -> None:
        async def executor(_: Invocation) -> None:
            return None

        with self.assertRaisesRegex(ValueError, "global worker"):
            LiveToolBroker(
                executor,
                max_workers=1,
                max_speculative_workers=1,
                min_speculative_workers=1,
            )
        with self.assertRaisesRegex(ValueError, "per-tool slot"):
            LiveToolBroker(
                executor,
                max_workers=4,
                max_speculative_workers=2,
                min_speculative_workers=1,
                tool_capacities={"visit": 1},
            )
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            LiveToolBroker(
                executor,
                max_workers=4,
                max_speculative_workers=2,
                min_speculative_workers=2,
            )

    async def test_minimum_zero_dispatch_telemetry_is_backward_compatible(
        self,
    ) -> None:
        async def executor(invocation: Invocation) -> str:
            return str(invocation.arguments["name"])

        broker = LiveToolBroker(
            executor,
            max_workers=1,
            max_speculative_workers=1,
            min_speculative_workers=0,
            max_speculative_pending=2,
            ttl_s=10,
        )
        committed = await broker.authoritative(
            Invocation("visit", {"name": "authoritative"}),
            session_id="authoritative",
        )
        self.assertEqual(committed.result, "authoritative")
        await broker.speculate(
            Invocation("visit", {"name": "speculative"}),
            session_id="speculative",
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        records = {row["session_id"]: row for row in broker.tool_records()}
        authoritative = records["authoritative"]
        self.assertEqual(authoritative["dispatch_lane"], "authoritative")
        self.assertEqual(
            authoritative["dispatch_reason"], "authoritative_priority"
        )
        self.assertEqual(authoritative["running_speculative_before"], 0)
        self.assertEqual(
            authoritative["queued_authoritative_same_tool_before"], 1
        )
        self.assertFalse(authoritative["reservation_debt_before"])
        self.assertFalse(authoritative["reservation_debt_after"])
        self.assertEqual(authoritative["per_tool_dispatch_ordinal"], 1)

        speculative = records["speculative"]
        self.assertEqual(speculative["dispatch_lane"], "speculative")
        self.assertEqual(
            speculative["dispatch_reason"], "speculative_opportunistic"
        )
        self.assertEqual(speculative["running_speculative_before"], 0)
        self.assertEqual(
            speculative["queued_authoritative_same_tool_before"], 0
        )
        self.assertFalse(speculative["reservation_debt_before"])
        self.assertFalse(speculative["reservation_debt_after"])
        self.assertEqual(speculative["per_tool_dispatch_ordinal"], 2)
        self.assertFalse(speculative["reserved_speculative_dispatch"])
        self.assertFalse(
            speculative["authoritative_after_reserved_dispatch"]
        )

        await broker.cancel_predictions()
        await broker.close()

    async def test_queued_exact_match_is_promoted_ahead_of_speculative_queue(
        self,
    ) -> None:
        release = asyncio.Event()
        first_started = asyncio.Event()
        promoted_started = asyncio.Event()
        order: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            order.append(name)
            if name == "first":
                first_started.set()
                await release.wait()
            if name == "promoted":
                promoted_started.set()
            await asyncio.sleep(0)
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=1,
            max_speculative_workers=1,
            max_speculative_pending=3,
            ttl_s=10,
        )
        await broker.speculate(Invocation("visit", {"name": "first"}), session_id="a")
        await first_started.wait()
        await broker.speculate(Invocation("visit", {"name": "other"}), session_id="b")
        exact = Invocation("visit", {"name": "promoted", "goal": "same"})
        await broker.speculate(exact, session_id="c", priority=-100.0)
        authoritative_task = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"goal": "same", "name": "promoted"}),
                session_id="c",
            )
        )
        await asyncio.sleep(0)
        release.set()
        await promoted_started.wait()
        result = await authoritative_task
        self.assertEqual(order[:2], ["first", "promoted"])
        self.assertEqual(result.source, "promoted_from_queue")
        self.assertEqual(broker.stats.queued_promotions, 1)
        self.assertEqual(broker.stats.commits, 1)
        await broker.cancel_predictions()
        await broker.close()

    async def test_result_is_private_until_exact_session_scoped_commit(self) -> None:
        calls: list[Invocation] = []

        async def executor(invocation: Invocation) -> dict[str, object]:
            calls.append(invocation)
            await asyncio.sleep(0)
            return {"secret-result": invocation.arguments}

        broker = LiveToolBroker(executor, max_workers=2, ttl_s=10)
        predicted = Invocation("visit", {"url": "u", "goal": "one"})
        await broker.speculate(predicted, session_id="owner")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(broker.authoritative_state, ())
        self.assertNotIn("secret-result", repr(broker.snapshot()))

        wrong_session = await broker.authoritative(predicted, session_id="other")
        self.assertEqual(wrong_session.source, "executed")
        mismatch = await broker.authoritative(
            Invocation("visit", {"url": "u", "goal": "two"}), session_id="owner"
        )
        self.assertEqual(mismatch.source, "executed")
        exact = await broker.authoritative(
            Invocation("visit", {"goal": "one", "url": "u"}), session_id="owner"
        )
        self.assertEqual(exact.source, "reused")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(broker.authoritative_state), 3)
        committed_records = [
            record for record in broker.tool_records() if record["committed"]
        ]
        self.assertEqual(len(committed_records), 3)
        self.assertTrue(all(record["result_digest"] for record in committed_records))
        self.assertTrue(
            {
                "invocation_id",
                "session_id",
                "tool",
                "queue_enter",
                "start",
                "confirmation",
                "finish",
                "worker_pool",
            }.issubset(committed_records[0])
        )
        await broker.close()

    async def test_expiry_cancels_unclaimed_running_prediction(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def executor(_: Invocation) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        broker = LiveToolBroker(
            executor,
            max_workers=1,
            max_speculative_workers=1,
            ttl_s=0.02,
        )
        await broker.speculate(Invocation("visit", {"url": "slow"}), session_id="s")
        await started.wait()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        self.assertEqual(broker.pending_speculative_count, 0)
        self.assertEqual(broker.stats.speculative_expired, 1)
        self.assertEqual(broker.snapshot()["counts"]["running_speculative"], 0)
        await broker.close()

    async def test_failed_speculation_falls_back_through_authoritative_lane(self) -> None:
        attempts = 0

        async def executor(_: Invocation) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("speculation failed")
            return "fresh"

        broker = LiveToolBroker(executor, max_workers=2, ttl_s=10)
        invocation = Invocation("search", {"query": ["q"]})
        await broker.speculate(invocation, session_id="s")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        result = await broker.authoritative(invocation, session_id="s")
        self.assertEqual(result.source, "executed_after_speculative_failure")
        self.assertEqual(result.result, "fresh")
        self.assertEqual(attempts, 2)
        self.assertEqual(broker.stats.commits, 1)
        await broker.close()

    async def test_ineligible_canary_bypasses_an_exact_prediction(self) -> None:
        calls = 0

        async def executor(_: Invocation) -> str:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return f"result-{calls}"

        broker = LiveToolBroker(executor, max_workers=2, ttl_s=10)
        invocation = Invocation("visit", {"url": "canary"})
        await broker.speculate(invocation, session_id="s")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        result = await broker.authoritative(
            invocation,
            session_id="s",
            speculation_eligible=False,
        )
        self.assertEqual(result.source, "executed")
        self.assertEqual(calls, 2)
        committed = next(record for record in broker.tool_records() if record["committed"])
        self.assertFalse(committed["speculation_eligible"])
        self.assertEqual(broker.pending_speculative_count, 1)
        await broker.cancel_predictions()
        await broker.close()

    async def test_fake_clock_sweep_and_change_notification(self) -> None:
        now = 10.0
        blocker = asyncio.Event()

        async def executor(_: Invocation) -> str:
            await blocker.wait()
            return "done"

        broker = LiveToolBroker(
            executor,
            max_workers=1,
            max_speculative_workers=1,
            ttl_s=5,
            clock=lambda: now,
        )
        initial_revision = broker.snapshot()["revision"]
        waiter = asyncio.create_task(broker.wait_for_change(initial_revision, timeout_s=1))
        await broker.speculate(Invocation("visit", {"url": "u"}), session_id="s")
        changed = await waiter
        self.assertGreater(changed["revision"], initial_revision)
        now = 16.0
        self.assertEqual(await broker.sweep(), 1)
        blocker.set()
        await broker.close()

    async def test_transport_metadata_is_logged_but_not_committed(self) -> None:
        async def executor(_: Invocation) -> dict[str, object]:
            return {
                "value": "semantic-result",
                "_paste_transport": {
                    "response_status": 200,
                    "bytes_read": 123,
                    "backend": "test_http",
                    "request_host": "example.test",
                    "http_attempts": 1,
                },
            }

        broker = LiveToolBroker(executor, max_workers=1, ttl_s=5)
        result = await broker.authoritative(
            Invocation("search", {"query": ["q"]}), session_id="s"
        )
        self.assertEqual(result.result, {"value": "semantic-result"})
        record = next(item for item in broker.tool_records() if item["committed"])
        self.assertEqual(record["response_status"], 200)
        self.assertEqual(record["bytes_read"], 123)
        self.assertEqual(record["backend"], "test_http")
        self.assertEqual(record["request_host"], "example.test")
        self.assertEqual(record["http_attempts"], 1)
        self.assertIsNotNone(record["worker_id"])
        await broker.close()

    async def test_success_transport_preserves_normalized_http_attempt_log(
        self,
    ) -> None:
        attempt_log = [
            {
                "request_index": 0,
                "attempt": 1,
                "status": 429,
                "error_type": "aiohttp.ClientResponseError",
                "retried": True,
                "started_monotonic_s": 10.0,
                "start_gate_wait_s": 0.125,
                "retry_backoff_s": 0.5,
            },
            {
                "request_index": 0,
                "attempt": 2,
                "status": 200,
                "error_type": None,
                "retried": False,
                "started_monotonic_s": 11.0,
                "start_gate_wait_s": 0.25,
                "retry_backoff_s": 0.0,
            },
        ]

        async def executor(_: Invocation) -> dict[str, object]:
            return {
                "value": "semantic-result",
                "_paste_transport": {
                    "response_status": 200,
                    "bytes_read": 123,
                    "backend": "test_http",
                    "request_host": "example.test",
                    "http_attempts": 2,
                    "http_attempt_log": attempt_log,
                },
            }

        broker = LiveToolBroker(executor, max_workers=1, ttl_s=5)
        result = await broker.authoritative(
            Invocation("search", {"query": ["q"]}), session_id="s"
        )

        self.assertEqual(result.result, {"value": "semantic-result"})
        record = next(item for item in broker.tool_records() if item["committed"])
        self.assertEqual(record["http_attempts"], len(record["http_attempt_log"]))
        self.assertEqual(record["http_attempt_log"], attempt_log)
        self.assertEqual(record["response_status"], 200)
        self.assertEqual(
            record["http_attempt_log"][-1]["status"],
            record["response_status"],
        )
        self.assertEqual(record["transport_identity_source"], "actual")
        self.assertEqual(record["http_attempt_log"][0]["start_gate_wait_s"], 0.125)
        self.assertEqual(record["http_attempt_log"][0]["retry_backoff_s"], 0.5)
        await broker.close()

    async def test_inconsistent_success_http_attempt_log_fails_fast(self) -> None:
        async def executor(_: Invocation) -> dict[str, object]:
            return {
                "value": "semantic-result",
                "_paste_transport": {
                    "response_status": 200,
                    "bytes_read": 123,
                    "backend": "test_http",
                    "request_host": "example.test",
                    "http_attempts": 2,
                    "http_attempt_log": [
                        {
                            "request_index": 0,
                            "attempt": 1,
                            "status": 200,
                            "error_type": None,
                            "retried": False,
                            "started_monotonic_s": 10.0,
                            "start_gate_wait_s": 0.0,
                            "retry_backoff_s": 0.0,
                        }
                    ],
                },
            }

        broker = LiveToolBroker(executor, max_workers=1, ttl_s=5)
        with self.assertRaisesRegex(
            ValueError,
            "http_attempts does not match http_attempt_log length",
        ):
            await broker.authoritative(
                Invocation("search", {"query": ["q"]}), session_id="s"
            )
        record = broker.tool_records()[0]
        self.assertEqual(record["outcome"], "failed")
        self.assertFalse(record["committed"])
        await broker.close()

    async def test_session_eta_includes_global_shared_queue_work(self) -> None:
        release = asyncio.Event()
        started = asyncio.Event()

        async def executor(_: Invocation) -> str:
            started.set()
            await release.wait()
            return "done"

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            max_speculative_workers=1,
            max_speculative_pending=3,
            ttl_s=10,
            service_time_hints_s={"visit": 2.0},
        )
        await broker.speculate(
            Invocation("visit", {"url": "first"}), session_id="first"
        )
        await started.wait()
        await broker.speculate(
            Invocation("visit", {"url": "second"}), session_id="second"
        )
        second = broker.snapshot(session_id="second")["jobs"][0]
        self.assertEqual(second["state"], "queued")
        self.assertEqual(second["queue_position"], 0)
        self.assertGreater(second["estimated_remaining_s"], 3.9)
        release.set()
        await broker.cancel_predictions()
        await broker.close()

    async def test_started_cancellation_retains_planned_transport_identity(self) -> None:
        class PlannedExecutor:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            def transport_plan(self, _: Invocation) -> dict[str, object]:
                return {
                    "backend": "wikipedia_mediawiki",
                    "request_host": "en.wikipedia.org",
                    "http_attempts": 1,
                }

            async def __call__(self, _: Invocation) -> None:
                self.started.set()
                await asyncio.Event().wait()

        executor = PlannedExecutor()
        broker = LiveToolBroker(
            executor,
            max_workers=1,
            max_speculative_workers=1,
            ttl_s=10,
        )
        await broker.speculate(
            Invocation("search", {"query": ["queueing"]}), session_id="s"
        )
        await executor.started.wait()
        await broker.cancel_predictions(session_id="s")
        record = broker.tool_records()[0]
        self.assertTrue(record["cancelled"])
        self.assertEqual(record["backend"], "wikipedia_mediawiki")
        self.assertEqual(record["request_host"], "en.wikipedia.org")
        self.assertEqual(record["http_attempts"], 1)
        self.assertEqual(record["transport_identity_source"], "planned")
        self.assertIsNone(record["response_status"])
        self.assertIsNone(record["bytes_read"])
        await broker.close()

    async def test_never_started_cancellation_records_zero_http_and_service(self) -> None:
        class PlannedExecutor:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            def transport_plan(self, _: Invocation) -> dict[str, object]:
                return {
                    "backend": "r.jina.ai",
                    "request_host": "r.jina.ai",
                    "http_attempts": 1,
                }

            async def __call__(self, _: Invocation) -> None:
                self.started.set()
                await asyncio.Event().wait()

        executor = PlannedExecutor()
        broker = LiveToolBroker(
            executor,
            max_workers=1,
            max_speculative_workers=1,
            max_speculative_pending=2,
            ttl_s=10,
        )
        await broker.speculate(
            Invocation("visit", {"url": ["https://example.test/first"]}),
            session_id="running",
        )
        await executor.started.wait()
        await broker.speculate(
            Invocation("visit", {"url": ["https://example.test/queued"]}),
            session_id="queued",
        )
        await asyncio.sleep(0.01)
        self.assertEqual(
            broker.snapshot(session_id="queued")["counts"]["queued_speculative"],
            1,
        )
        self.assertEqual(
            await broker.cancel_predictions(session_id="queued"), 1
        )

        record = next(
            item for item in broker.tool_records() if item["session_id"] == "queued"
        )
        self.assertTrue(record["cancelled"])
        self.assertEqual(record["outcome"], "cancelled")
        self.assertIsNone(record["start"])
        self.assertIsNone(record["started_at"])
        self.assertIsNone(record["worker_id"])
        self.assertEqual(record["http_attempts"], 0)
        self.assertEqual(record["service_s"], 0.0)
        self.assertEqual(record["saved_service_s"], 0.0)
        for field in (
            "dispatch_lane",
            "dispatch_reason",
            "running_speculative_before",
            "queued_authoritative_same_tool_before",
            "reservation_debt_before",
            "reservation_debt_after",
            "per_tool_dispatch_ordinal",
        ):
            self.assertIsNone(record[field])
        self.assertAlmostEqual(
            record["queue_s"],
            record["finished_at"] - record["queue_enter_at"],
        )
        for field in (
            "backend",
            "request_host",
            "response_status",
            "bytes_read",
            "transport_identity_source",
        ):
            self.assertIsNone(record[field])

        await broker.cancel_predictions(session_id="running")
        await broker.close()

    async def test_failed_attempt_retains_planned_transport_identity(self) -> None:
        class PlannedFailure:
            def transport_plan(self, _: Invocation) -> dict[str, object]:
                return {
                    "backend": "direct_http",
                    "request_host": "example.test",
                    "http_attempts": 1,
                }

            async def __call__(self, _: Invocation) -> None:
                raise RuntimeError("network failed before response")

        broker = LiveToolBroker(PlannedFailure(), max_workers=1, ttl_s=10)
        with self.assertRaisesRegex(RuntimeError, "network failed"):
            await broker.authoritative(
                Invocation("visit", {"url": "https://example.test"}),
                session_id="s",
            )
        record = broker.tool_records()[0]
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(record["backend"], "direct_http")
        self.assertEqual(record["request_host"], "example.test")
        self.assertEqual(record["http_attempts"], 1)
        self.assertEqual(record["transport_identity_source"], "planned")
        await broker.close()


class SyncToolMapExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_runs_existing_call_interface(self) -> None:
        class Tool:
            def call(self, arguments: dict[str, object]) -> tuple[int, object]:
                return id(asyncio.get_event_loop_policy()), arguments["value"]

        executor = SyncToolMapExecutor({"search": Tool()}, thread_workers=1)
        _, value = await executor(Invocation("search", {"value": 7}))
        self.assertEqual(value, 7)
        await executor.close()

    async def test_cancel_does_not_release_capacity_before_thread_drains(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class Tool:
            def call(self, _: dict[str, object]) -> str:
                started.set()
                release.wait(timeout=2)
                return "finished"

        executor = SyncToolMapExecutor({"visit": Tool()}, thread_workers=1)
        broker = LiveToolBroker(
            executor,
            max_workers=1,
            max_speculative_workers=1,
            ttl_s=10,
        )
        await broker.speculate(Invocation("visit", {"url": "u"}))
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        cancellation = asyncio.create_task(broker.cancel_predictions())
        await asyncio.sleep(0.01)
        self.assertFalse(cancellation.done())
        self.assertEqual(broker.snapshot()["counts"]["running_speculative"], 1)
        release.set()
        self.assertEqual(await asyncio.wait_for(cancellation, timeout=1), 1)
        self.assertEqual(broker.snapshot()["counts"]["running_speculative"], 0)
        await broker.close()
        await executor.close()

    async def test_per_tool_minimum_start_interval_waits_in_queue(self) -> None:
        now = [10.0]
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()
        starts: list[tuple[str, float]] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            starts.append((name, now[0]))
            if name == "first":
                first_started.set()
                await release_first.wait()
            else:
                second_started.set()
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            clock=lambda: now[0],
            service_time_hints_s={"visit": 1.0},
            tool_min_start_intervals_s={"visit": 2.1},
        )
        first = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "first"}), session_id="first"
            )
        )
        await first_started.wait()
        second = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "second"}), session_id="second"
            )
        )
        for _ in range(10):
            snapshot = broker.snapshot()
            if snapshot["counts"]["queued_authoritative"] == 1:
                break
            await asyncio.sleep(0)

        self.assertEqual(snapshot["counts"]["running_by_tool"], {"visit": 1})
        self.assertEqual(snapshot["counts"]["queued_by_tool"], {"visit": 1})
        queued = next(job for job in snapshot["jobs"] if job["state"] == "queued")
        self.assertEqual(queued["tool_min_start_interval_s"], 2.1)
        self.assertAlmostEqual(queued["rate_limit_eligible_at"], 12.1)
        self.assertAlmostEqual(queued["rate_limit_wait_s"], 2.1)
        self.assertAlmostEqual(queued["estimated_queue_s"], 2.1)
        self.assertEqual(
            snapshot["capacity"]["tool_min_start_intervals_s"], {"visit": 2.1}
        )
        self.assertEqual(snapshot["rate_limit"]["next_eligible_at"], {"visit": 12.1})
        self.assertFalse(second_started.is_set())

        release_first.set()
        self.assertEqual((await first).result, "first")
        await asyncio.sleep(0)
        self.assertFalse(second_started.is_set())
        self.assertEqual(broker.snapshot()["counts"]["running_by_tool"], {})

        now[0] = 12.1
        await broker.sweep()
        await asyncio.wait_for(second_started.wait(), timeout=1)
        self.assertEqual((await second).result, "second")
        self.assertEqual(starts, [("first", 10.0), ("second", 12.1)])
        records = [record for record in broker.tool_records() if record["committed"]]
        self.assertEqual(len(records), 2)
        self.assertTrue(
            all(record["tool_min_start_interval_s"] == 2.1 for record in records)
        )
        self.assertAlmostEqual(records[1]["rate_limit_eligible_at"], 12.1)
        self.assertAlmostEqual(records[1]["rate_limit_next_eligible_at"], 14.2)
        self.assertAlmostEqual(records[1]["rate_limit_wait_s"], 2.1)
        await broker.close()
        self.assertIsNone(broker._rate_wakeup_task)

    async def test_rate_limit_is_shared_and_authoritative_starts_first(self) -> None:
        now = [20.0]
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        order: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            order.append(name)
            if name == "first-spec":
                first_started.set()
                await release_first.wait()
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            max_speculative_workers=2,
            clock=lambda: now[0],
            tool_min_start_intervals_s={"visit": 1.0},
        )
        await broker.speculate(
            Invocation("visit", {"name": "first-spec"}), session_id="first"
        )
        await first_started.wait()
        await broker.speculate(
            Invocation("visit", {"name": "queued-spec"}), session_id="spec"
        )
        authoritative = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "queued-auth"}), session_id="auth"
            )
        )
        for _ in range(10):
            snapshot = broker.snapshot()
            if snapshot["counts"]["queued_authoritative"] == 1:
                break
            await asyncio.sleep(0)
        self.assertEqual(snapshot["counts"]["queued_speculative"], 1)
        self.assertEqual(snapshot["counts"]["queued_authoritative"], 1)
        release_first.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(order, ["first-spec"])

        now[0] = 21.0
        await broker.sweep()
        self.assertEqual((await authoritative).result, "queued-auth")
        self.assertEqual(order[:2], ["first-spec", "queued-auth"])
        self.assertEqual(broker.snapshot()["counts"]["queued_speculative"], 1)
        self.assertEqual(await broker.cancel_predictions(), 2)
        self.assertIsNone(broker._rate_wakeup_task)
        await asyncio.wait_for(broker.close(), timeout=1)

    async def test_reserved_start_is_repaid_before_next_rate_limited_spec(self) -> None:
        now = [30.0]
        order: list[str] = []

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            order.append(name)
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            max_speculative_workers=1,
            min_speculative_workers=1,
            max_speculative_pending=4,
            clock=lambda: now[0],
            service_time_hints_s={"visit": 0.5},
            tool_capacities={"visit": 2},
            tool_min_start_intervals_s={"visit": 1.0},
        )
        primer = await broker.authoritative(
            Invocation("visit", {"name": "primer"}), session_id="primer"
        )
        self.assertEqual(primer.result, "primer")

        authoritative = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "authoritative"}),
                session_id="auth",
            )
        )
        await asyncio.sleep(0)
        await broker.speculate(
            Invocation("visit", {"name": "reserved-one"}), session_id="spec-1"
        )
        before = broker.snapshot()
        queued_before = {
            job["session_id"]: job
            for job in before["jobs"]
            if job["state"] == "queued"
        }
        self.assertEqual(queued_before["spec-1"]["tool_queue_position"], 0)
        self.assertEqual(queued_before["auth"]["tool_queue_position"], 1)
        self.assertAlmostEqual(
            queued_before["spec-1"]["rate_limit_eligible_at"], 31.0
        )
        self.assertAlmostEqual(
            queued_before["auth"]["rate_limit_eligible_at"], 32.0
        )
        self.assertAlmostEqual(queued_before["spec-1"]["estimated_queue_s"], 1.0)
        self.assertAlmostEqual(queued_before["auth"]["estimated_queue_s"], 2.0)

        now[0] = 31.0
        await broker.sweep()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(order[:2], ["primer", "reserved-one"])
        self.assertEqual(
            broker.snapshot()["reservation"]["authoritative_turn_due_by_tool"],
            ["visit"],
        )
        await broker.speculate(
            Invocation("visit", {"name": "reserved-two"}), session_id="spec-2"
        )
        owed = broker.snapshot()
        queued_owed = {
            job["session_id"]: job
            for job in owed["jobs"]
            if job["state"] == "queued"
        }
        self.assertEqual(queued_owed["auth"]["tool_queue_position"], 0)
        self.assertEqual(queued_owed["spec-2"]["tool_queue_position"], 1)
        self.assertAlmostEqual(queued_owed["auth"]["estimated_queue_s"], 1.0)
        self.assertAlmostEqual(queued_owed["spec-2"]["estimated_queue_s"], 2.0)

        now[0] = 32.0
        await broker.sweep()
        self.assertEqual((await authoritative).result, "authoritative")
        self.assertEqual(order[:3], ["primer", "reserved-one", "authoritative"])
        self.assertEqual(broker.stats.reserved_speculative_dispatches, 1)
        self.assertEqual(broker.stats.authoritative_after_reserved_dispatches, 1)
        records = {row["session_id"]: row for row in broker.tool_records()}
        self.assertTrue(records["spec-1"]["reserved_speculative_dispatch"])
        self.assertTrue(
            records["auth"]["authoritative_after_reserved_dispatch"]
        )
        primer_record = records["primer"]
        self.assertEqual(primer_record["dispatch_lane"], "authoritative")
        self.assertEqual(
            primer_record["dispatch_reason"], "authoritative_priority"
        )
        self.assertEqual(primer_record["running_speculative_before"], 0)
        self.assertEqual(
            primer_record["queued_authoritative_same_tool_before"], 1
        )
        self.assertFalse(primer_record["reservation_debt_before"])
        self.assertFalse(primer_record["reservation_debt_after"])
        self.assertEqual(primer_record["per_tool_dispatch_ordinal"], 1)

        reserved_record = records["spec-1"]
        self.assertEqual(reserved_record["dispatch_lane"], "speculative")
        self.assertEqual(
            reserved_record["dispatch_reason"], "reserved_speculative"
        )
        self.assertEqual(reserved_record["running_speculative_before"], 0)
        self.assertEqual(
            reserved_record["queued_authoritative_same_tool_before"], 1
        )
        self.assertFalse(reserved_record["reservation_debt_before"])
        self.assertTrue(reserved_record["reservation_debt_after"])
        self.assertEqual(reserved_record["per_tool_dispatch_ordinal"], 2)

        repayment_record = records["auth"]
        self.assertEqual(repayment_record["dispatch_lane"], "authoritative")
        self.assertEqual(
            repayment_record["dispatch_reason"], "authoritative_repayment"
        )
        self.assertEqual(repayment_record["running_speculative_before"], 0)
        self.assertEqual(
            repayment_record["queued_authoritative_same_tool_before"], 1
        )
        self.assertTrue(repayment_record["reservation_debt_before"])
        self.assertFalse(repayment_record["reservation_debt_after"])
        self.assertEqual(repayment_record["per_tool_dispatch_ordinal"], 3)

        await broker.cancel_predictions()
        never_started = {
            row["session_id"]: row for row in broker.tool_records()
        }["spec-2"]
        self.assertIsNone(never_started["started_at"])
        for field in (
            "dispatch_lane",
            "dispatch_reason",
            "running_speculative_before",
            "queued_authoritative_same_tool_before",
            "reservation_debt_before",
            "reservation_debt_after",
            "per_tool_dispatch_ordinal",
        ):
            self.assertIsNone(never_started[field])
        await broker.close()

    async def test_snapshot_does_not_reserve_second_start_while_spec_is_running(
        self,
    ) -> None:
        now = [40.0]
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            if name == "running-spec":
                first_started.set()
                await release_first.wait()
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=2,
            max_speculative_workers=1,
            min_speculative_workers=1,
            max_speculative_pending=4,
            clock=lambda: now[0],
            service_time_hints_s={"visit": 0.5},
            tool_capacities={"visit": 2},
            tool_min_start_intervals_s={"visit": 1.0},
        )
        await broker.speculate(
            Invocation("visit", {"name": "running-spec"}),
            session_id="running-spec",
        )
        await first_started.wait()
        self.assertEqual(broker.snapshot()["counts"]["running_speculative"], 1)
        self.assertEqual(
            broker.snapshot()["reservation"]["authoritative_turn_due_by_tool"],
            [],
        )

        authoritative = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "queued-auth-one"}),
                session_id="queued-auth-one",
            )
        )
        await asyncio.sleep(0)
        authoritative_two = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "queued-auth-two"}),
                session_id="queued-auth-two",
            )
        )
        await asyncio.sleep(0)
        await broker.speculate(
            Invocation("visit", {"name": "queued-spec"}),
            session_id="queued-spec",
        )
        snapshot = broker.snapshot()
        queued = {
            job["session_id"]: job
            for job in snapshot["jobs"]
            if job["state"] == "queued"
        }
        self.assertEqual(queued["queued-auth-one"]["tool_queue_position"], 0)
        self.assertEqual(queued["queued-auth-two"]["tool_queue_position"], 1)
        self.assertEqual(queued["queued-spec"]["tool_queue_position"], 2)
        self.assertLess(
            queued["queued-auth-one"]["estimated_global_queue_s"],
            queued["queued-auth-two"]["estimated_global_queue_s"],
        )
        self.assertLess(
            queued["queued-auth-two"]["estimated_global_queue_s"],
            queued["queued-spec"]["estimated_global_queue_s"],
        )
        self.assertAlmostEqual(
            queued["queued-auth-one"]["rate_limit_eligible_at"], 41.0
        )
        self.assertAlmostEqual(
            queued["queued-auth-two"]["rate_limit_eligible_at"], 42.0
        )
        self.assertAlmostEqual(
            queued["queued-spec"]["rate_limit_eligible_at"], 43.0
        )

        now[0] = 41.0
        await broker.sweep()
        self.assertEqual((await authoritative).result, "queued-auth-one")
        self.assertFalse(authoritative_two.done())
        now[0] = 42.0
        await broker.sweep()
        self.assertEqual((await authoritative_two).result, "queued-auth-two")
        release_first.set()
        await asyncio.sleep(0)
        await broker.cancel_predictions()
        await broker.close()

    async def test_snapshot_running_speculation_blocks_cross_tool_reservation(
        self,
    ) -> None:
        now = [50.0]
        running_started = asyncio.Event()
        release_running = asyncio.Event()

        async def executor(invocation: Invocation) -> str:
            name = str(invocation.arguments["name"])
            if name == "running-search-spec":
                running_started.set()
                await release_running.wait()
            return name

        broker = LiveToolBroker(
            executor,
            max_workers=3,
            max_speculative_workers=1,
            min_speculative_workers=1,
            max_speculative_pending=4,
            clock=lambda: now[0],
            service_time_hints_s={"search": 2.0, "visit": 0.5},
            tool_capacities={"search": 2, "visit": 2},
            tool_min_start_intervals_s={"visit": 1.0},
        )
        primer = await broker.authoritative(
            Invocation("visit", {"name": "primer"}), session_id="primer"
        )
        self.assertEqual(primer.result, "primer")
        await broker.speculate(
            Invocation("search", {"name": "running-search-spec"}),
            session_id="running-search-spec",
        )
        await running_started.wait()

        authoritative = asyncio.create_task(
            broker.authoritative(
                Invocation("visit", {"name": "queued-visit-auth"}),
                session_id="queued-visit-auth",
            )
        )
        await asyncio.sleep(0)
        await broker.speculate(
            Invocation("visit", {"name": "queued-visit-spec"}),
            session_id="queued-visit-spec",
        )
        snapshot = broker.snapshot()
        queued = {
            job["session_id"]: job
            for job in snapshot["jobs"]
            if job["state"] == "queued"
        }
        self.assertEqual(
            queued["queued-visit-auth"]["tool_queue_position"], 0
        )
        self.assertEqual(
            queued["queued-visit-spec"]["tool_queue_position"], 1
        )
        self.assertLess(
            queued["queued-visit-auth"]["estimated_global_queue_s"],
            queued["queued-visit-spec"]["estimated_global_queue_s"],
        )
        self.assertAlmostEqual(
            queued["queued-visit-auth"]["rate_limit_eligible_at"], 51.0
        )
        self.assertAlmostEqual(
            queued["queued-visit-spec"]["rate_limit_eligible_at"], 52.0
        )

        now[0] = 51.0
        await broker.sweep()
        self.assertEqual((await authoritative).result, "queued-visit-auth")
        release_running.set()
        await asyncio.sleep(0)
        running_record = {
            row["session_id"]: row for row in broker.tool_records()
        }["running-search-spec"]
        self.assertEqual(running_record["dispatch_lane"], "speculative")
        self.assertEqual(
            running_record["dispatch_reason"],
            "speculative_minimum_uncontended",
        )
        self.assertEqual(running_record["running_speculative_before"], 0)
        self.assertEqual(
            running_record["queued_authoritative_same_tool_before"], 0
        )
        self.assertFalse(running_record["reservation_debt_before"])
        self.assertFalse(running_record["reservation_debt_after"])
        self.assertFalse(running_record["reserved_speculative_dispatch"])
        await broker.cancel_predictions()
        await broker.close()

    async def test_minimum_start_interval_validation_and_zero_compatibility(
        self,
    ) -> None:
        async def executor(_: Invocation) -> None:
            return None

        for invalid in (-0.1, math.inf, -math.inf, math.nan, True, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    LiveToolBroker(
                        executor,
                        tool_min_start_intervals_s={"visit": invalid},  # type: ignore[dict-item]
                    )
        with self.assertRaises(ValueError):
            LiveToolBroker(executor, tool_min_start_intervals_s={"": 1.0})

        broker = LiveToolBroker(
            executor, tool_min_start_intervals_s={"visit": 0.0}
        )
        self.assertEqual(
            broker.snapshot()["capacity"]["tool_min_start_intervals_s"],
            {"visit": 0.0},
        )
        await broker.close()


class _FakeContent:
    def __init__(self, value: bytes, error: BaseException | None = None) -> None:
        self._value = value
        self._error = error

    async def iter_chunked(self, _: int):
        if self._error is not None:
            raise self._error
        yield self._value


class _FakeHTTPStatusError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


class _FakeResponse:
    def __init__(
        self,
        *,
        payload: dict[str, object] | None = None,
        body: bytes = b"",
        content_type: str = "application/json",
        status: int = 200,
        read_error: BaseException | None = None,
        content_error: BaseException | None = None,
    ) -> None:
        self._payload = payload
        if payload is not None and not body:
            body = json.dumps(payload).encode("utf-8")
        self._body = body
        self.content = _FakeContent(body, content_error)
        self.headers = {"Content-Type": content_type}
        self.charset = "utf-8"
        self.status = status
        self._read_error = read_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise _FakeHTTPStatusError(self.status)

    async def json(self, **_: object) -> dict[str, object]:
        assert self._payload is not None
        return self._payload

    async def read(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append((url, kwargs))
        if "search/page" in url:
            return _FakeResponse(
                payload={
                    "pages": [
                        {
                            "key": "Queueing_theory",
                            "title": "Queueing theory",
                            "excerpt": "Study of <span>waiting lines</span>",
                        }
                    ]
                }
            )
        if "api.php" in url:
            return _FakeResponse(
                payload={
                    "query": {
                        "search": [
                            {
                                "title": "Queueing theory",
                                "snippet": "Study of <span>waiting lines</span>",
                            }
                        ]
                    }
                }
            )
        return _FakeResponse(
            body=b"<html><head><title>Example</title></head>"
            b"<body><p>Useful content</p></body></html>",
            content_type="text/html; charset=utf-8",
        )


class _ScriptedSession:
    def __init__(self, outcomes: list[_FakeResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class WikipediaLiveExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_aiohttp_internal_connection_retry_is_disabled(self) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            self.assertIs(getattr(session, "_retry_connection"), True)
            executor = WikipediaLiveExecutor(session=session)
            self.assertIs(await executor._ensure_session(), session)
            self.assertIs(getattr(session, "_retry_connection"), False)
            self.assertTrue(executor.http_library_retry_disabled_effective)
            self.assertEqual(executor.http_library_name, "aiohttp")
            self.assertEqual(executor.http_library_version, aiohttp.__version__)
            self.assertEqual(
                executor.HTTP_LIBRARY_RETRY_CONTROL_VERSION,
                "aiohttp-private-retry-connection-v1",
            )
            await executor.close()

    async def test_real_aiohttp_unknown_retry_control_shape_fails_closed(self) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            session._retry_connection = "unknown"  # type: ignore[assignment]
            executor = WikipediaLiveExecutor(session=session)
            with self.assertRaisesRegex(RuntimeError, "retry control is not bool"):
                await executor._ensure_session()
            self.assertFalse(executor.http_library_retry_disabled_effective)
            await executor.close()

    async def test_retry_defaults_and_policy_are_explicit_and_validated(self) -> None:
        executor = WikipediaLiveExecutor(session=_FakeSession())
        self.assertEqual(executor.max_http_attempts, 1)
        self.assertEqual(executor.retry_backoff_s, 1.0)
        self.assertEqual(executor.http_attempt_min_start_intervals_s, {})
        self.assertEqual(
            executor.HTTP_RETRY_POLICY_VERSION,
            "idempotent-get-v1",
        )
        self.assertEqual(
            executor.HTTP_ATTEMPT_START_GATE_VERSION,
            "shared-per-tool-monotonic-v1",
        )
        self.assertEqual(
            executor.RETRYABLE_HTTP_STATUSES,
            (429, 500, 502, 503, 504),
        )
        await executor.close()

        for invalid in (0, -1, 1.5, True):
            with self.subTest(max_http_attempts=invalid):
                with self.assertRaisesRegex(ValueError, "max_http_attempts"):
                    WikipediaLiveExecutor(
                        session=_FakeSession(),
                        max_http_attempts=invalid,  # type: ignore[arg-type]
                    )
        for invalid in (-0.1, float("inf"), float("nan"), True, "1"):
            with self.subTest(retry_backoff_s=invalid):
                with self.assertRaisesRegex(ValueError, "retry_backoff_s"):
                    WikipediaLiveExecutor(
                        session=_FakeSession(),
                        retry_backoff_s=invalid,  # type: ignore[arg-type]
                    )
        for mapping in (
            {"search": -0.1},
            {"visit": float("inf")},
            {"visit": True},
            {"unknown": 1.0},
        ):
            with self.subTest(http_attempt_min_start_intervals_s=mapping):
                with self.assertRaisesRegex(ValueError, "HTTP-attempt"):
                    WikipediaLiveExecutor(
                        session=_FakeSession(),
                        http_attempt_min_start_intervals_s=mapping,
                    )

    async def test_retryable_429_then_success_counts_actual_gets(self) -> None:
        success = _FakeResponse(
            payload={
                "pages": [
                    {
                        "key": "Queueing_theory",
                        "title": "Queueing theory",
                        "excerpt": "Waiting lines",
                    }
                ]
            }
        )
        session = _ScriptedSession(
            [_FakeResponse(status=429, body=b"limited"), success]
        )
        executor = WikipediaLiveExecutor(
            session=session,
            max_http_attempts=2,
            retry_backoff_s=0,
        )
        result = await executor(
            Invocation("search", {"query": ["queueing theory"]})
        )

        transport = result["_paste_transport"]
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(transport["response_status"], 200)
        self.assertEqual(transport["bytes_read"], len(success._body))
        self.assertEqual(transport["backend"], "wikipedia_rest_search")
        self.assertEqual(transport["request_host"], "en.wikipedia.org")
        self.assertEqual(transport["http_attempts"], 2)
        self.assertEqual(transport["http_retries"], 1)
        self.assertEqual(
            [entry["status"] for entry in transport["http_attempt_log"]],
            [429, 200],
        )
        self.assertEqual(
            [entry["retried"] for entry in transport["http_attempt_log"]],
            [True, False],
        )
        for entry in transport["http_attempt_log"]:
            self.assertIsInstance(entry["started_monotonic_s"], float)
            self.assertEqual(entry["start_gate_wait_s"], 0.0)
            self.assertGreaterEqual(entry["retry_backoff_s"], 0.0)
        self.assertNotIn("http_attempt_log", result["results"][0])
        await executor.close()

    async def test_attempt_gate_spaces_concurrent_gets_and_explicit_retry(
        self,
    ) -> None:
        success = _FakeResponse(payload={"pages": []})
        session = _ScriptedSession(
            [
                _FakeResponse(status=429, body=b"limited"),
                success,
                _FakeResponse(payload={"pages": []}),
            ]
        )
        executor = WikipediaLiveExecutor(
            session=session,
            max_http_attempts=2,
            retry_backoff_s=0.003,
            http_attempt_min_start_intervals_s={"search": 0.015},
        )
        result = await executor(
            Invocation("search", {"query": ["first", "second"]})
        )

        attempts = sorted(
            result["_paste_transport"]["http_attempt_log"],
            key=lambda entry: entry["started_monotonic_s"],
        )
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(attempts), 3)
        for earlier, later in zip(attempts, attempts[1:]):
            self.assertGreaterEqual(
                later["started_monotonic_s"] - earlier["started_monotonic_s"],
                0.012,
            )
        self.assertGreaterEqual(attempts[1]["start_gate_wait_s"], 0.010)
        retried_failure = next(entry for entry in attempts if entry["retried"])
        self.assertGreaterEqual(retried_failure["retry_backoff_s"], 0.002)
        await executor.close()

    async def test_attempt_gates_for_search_and_visit_are_independent(self) -> None:
        session = _FakeSession()
        executor = WikipediaLiveExecutor(
            session=session,
            http_attempt_min_start_intervals_s={
                "search": 0.04,
                "visit": 0.04,
            },
        )
        search_task = asyncio.create_task(
            executor(Invocation("search", {"query": ["first", "second"]}))
        )
        await asyncio.sleep(0.005)
        visit = await executor(
            Invocation("visit", {"url": "https://example.test/page"})
        )
        search = await search_task

        search_starts = sorted(
            entry["started_monotonic_s"]
            for entry in search["_paste_transport"]["http_attempt_log"]
        )
        visit_entry = visit["_paste_transport"]["http_attempt_log"][0]
        self.assertLess(search_starts[0], visit_entry["started_monotonic_s"])
        self.assertLess(visit_entry["started_monotonic_s"], search_starts[1])
        self.assertLess(visit_entry["start_gate_wait_s"], 0.01)
        await executor.close()

    async def test_timeout_is_retried_but_non_429_4xx_is_not(self) -> None:
        success = _FakeResponse(
            payload={"pages": []},
        )
        timeout_session = _ScriptedSession([asyncio.TimeoutError("slow"), success])
        executor = WikipediaLiveExecutor(
            session=timeout_session,
            max_http_attempts=2,
            retry_backoff_s=0,
        )
        result = await executor(Invocation("search", {"query": "q"}))
        self.assertEqual(result["_paste_transport"]["http_attempts"], 2)
        self.assertEqual(
            result["_paste_transport"]["http_attempt_log"][0]["status"],
            None,
        )
        await executor.close()

        not_found = _FakeHTTPStatusError(404)
        non_retry_session = _ScriptedSession([not_found, success])
        executor = WikipediaLiveExecutor(
            session=non_retry_session,
            max_http_attempts=2,
            retry_backoff_s=0,
        )
        with self.assertRaises(_FakeHTTPStatusError) as caught:
            await executor(Invocation("search", {"query": "q"}))
        self.assertIs(caught.exception, not_found)
        self.assertEqual(len(non_retry_session.calls), 1)
        attempt_log = caught.exception.paste_http_attempt_log
        self.assertEqual(len(attempt_log), 1)
        self.assertEqual(
            {
                key: attempt_log[0][key]
                for key in (
                    "request_index",
                    "attempt",
                    "status",
                    "error_type",
                    "retried",
                )
            },
            {
                "request_index": 0,
                "attempt": 1,
                "status": 404,
                "error_type": (
                    f"{type(not_found).__module__}."
                    f"{type(not_found).__qualname__}"
                ),
                "retried": False,
            },
        )
        self.assertIsInstance(attempt_log[0]["started_monotonic_s"], float)
        self.assertEqual(attempt_log[0]["start_gate_wait_s"], 0.0)
        self.assertEqual(attempt_log[0]["retry_backoff_s"], 0.0)
        await executor.close()

    async def test_search_retries_transient_body_read_after_http_200(self) -> None:
        success = _FakeResponse(payload={"pages": []})
        session = _ScriptedSession(
            [
                _FakeResponse(
                    status=200,
                    read_error=asyncio.TimeoutError("body stalled"),
                ),
                success,
            ]
        )
        executor = WikipediaLiveExecutor(
            session=session,
            max_http_attempts=2,
            retry_backoff_s=0,
        )
        result = await executor(Invocation("search", {"query": "q"}))

        transport = result["_paste_transport"]
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(transport["http_attempts"], 2)
        self.assertEqual(transport["response_status"], 200)
        self.assertEqual(
            [entry["status"] for entry in transport["http_attempt_log"]],
            [200, 200],
        )
        self.assertTrue(transport["http_attempt_log"][0]["retried"])
        await executor.close()

    async def test_visit_retries_transient_body_disconnect_after_http_200(
        self,
    ) -> None:
        import aiohttp

        body = b"<html><body>complete</body></html>"
        session = _ScriptedSession(
            [
                _FakeResponse(
                    status=200,
                    content_type="text/html",
                    content_error=aiohttp.ClientPayloadError(
                        "body disconnected"
                    ),
                ),
                _FakeResponse(body=body, content_type="text/html"),
            ]
        )
        executor = WikipediaLiveExecutor(
            session=session,
            visit_mode="jina",
            max_http_attempts=2,
            retry_backoff_s=0,
        )
        result = await executor(
            Invocation("visit", {"url": "https://example.test/page"})
        )

        transport = result["_paste_transport"]
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(transport["http_attempts"], 2)
        self.assertEqual(transport["response_status"], 200)
        self.assertEqual(transport["bytes_read"], len(body))
        self.assertEqual(result["pages"][0]["content"], "complete")
        self.assertEqual(
            [entry["status"] for entry in transport["http_attempt_log"]],
            [200, 200],
        )
        self.assertTrue(transport["http_attempt_log"][0]["retried"])
        await executor.close()

    async def test_exhaustion_preserves_last_exception_and_attempt_log(self) -> None:
        first = _FakeHTTPStatusError(503)
        last = _FakeHTTPStatusError(503)
        session = _ScriptedSession([first, last])
        executor = WikipediaLiveExecutor(
            session=session,
            max_http_attempts=2,
            retry_backoff_s=0,
        )
        with self.assertRaises(_FakeHTTPStatusError) as caught:
            await executor(Invocation("visit", {"url": "https://example.test"}))
        self.assertIs(caught.exception, last)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(
            [entry["status"] for entry in caught.exception.paste_http_attempt_log],
            [503, 503],
        )
        self.assertEqual(
            [entry["retried"] for entry in caught.exception.paste_http_attempt_log],
            [True, False],
        )
        await executor.close()

    async def test_broker_records_all_physical_attempts_after_retry_exhaustion(
        self,
    ) -> None:
        first = _FakeHTTPStatusError(503)
        last = _FakeHTTPStatusError(503)
        session = _ScriptedSession([first, last])
        executor = WikipediaLiveExecutor(
            session=session,
            visit_mode="jina",
            max_http_attempts=2,
            retry_backoff_s=0,
        )
        broker = LiveToolBroker(executor, max_workers=1, ttl_s=5)

        with self.assertRaises(_FakeHTTPStatusError):
            await broker.authoritative(
                Invocation("visit", {"url": "https://example.test/page"}),
                session_id="failure",
            )

        records = broker.tool_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["outcome"], "failed")
        self.assertEqual(records[0]["http_attempts"], 2)
        self.assertEqual(records[0]["response_status"], 503)
        self.assertEqual(records[0]["transport_identity_source"], "actual_failure")
        self.assertEqual(
            [entry["status"] for entry in records[0]["http_attempt_log"]],
            [503, 503],
        )
        self.assertEqual(records[0]["backend"], "r.jina.ai")
        self.assertEqual(records[0]["request_host"], "r.jina.ai")
        await broker.close()
        await executor.close()

    async def test_retry_backoff_occupies_one_broker_job_and_service_time(self) -> None:
        body = b"<html><body>ok</body></html>"
        session = _ScriptedSession(
            [
                _FakeResponse(status=503, body=b"temporary"),
                _FakeResponse(body=body, content_type="text/html"),
            ]
        )
        executor = WikipediaLiveExecutor(
            session=session,
            visit_mode="jina",
            max_http_attempts=2,
            retry_backoff_s=0.01,
        )
        broker = LiveToolBroker(executor, max_workers=1, ttl_s=5)
        result = await broker.authoritative(
            Invocation("visit", {"url": "https://example.test/page"}),
            session_id="s",
        )

        self.assertEqual(result.result["pages"][0]["content"], "ok")
        records = broker.tool_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["http_attempts"], 2)
        self.assertEqual(records[0]["response_status"], 200)
        self.assertEqual(records[0]["bytes_read"], len(body))
        self.assertEqual(records[0]["backend"], "r.jina.ai")
        self.assertEqual(records[0]["request_host"], "r.jina.ai")
        self.assertGreaterEqual(records[0]["service_s"], 0.009)
        await broker.close()
        await executor.close()

    async def test_structured_search_and_direct_visit(self) -> None:
        session = _FakeSession()
        executor = WikipediaLiveExecutor(session=session, max_results=3)
        search = await executor(
            Invocation("search", {"query": ["queueing theory"]})
        )
        self.assertEqual(search["tool"], "search")
        self.assertEqual(search["results"][0]["rank"], 1)
        self.assertEqual(search["results"][0]["query_index"], 0)
        self.assertEqual(
            search["_paste_transport"]["backend"], "wikipedia_rest_search"
        )
        self.assertEqual(
            search["_paste_transport"]["request_host"], "en.wikipedia.org"
        )
        self.assertEqual(
            search["results"][0]["url"],
            "https://en.wikipedia.org/wiki/Queueing_theory",
        )

        visit = await executor(
            Invocation(
                "visit",
                {"url": search["results"][0]["url"], "goal": "definition"},
            )
        )
        self.assertEqual(visit["tool"], "visit")
        self.assertEqual(visit["goal"], "definition")
        self.assertEqual(visit["pages"][0]["title"], "Example")
        self.assertIn("Useful content", visit["pages"][0]["content"])
        self.assertEqual(
            visit["_paste_transport"]["backend"], "wikipedia_rest_page"
        )
        self.assertIn("/w/rest.php/v1/page/Queueing_theory/html", session.calls[-1][0])
        await executor.close()

    async def test_bing_search_decodes_redirect_and_filters_wikipedia(self) -> None:
        target = "https://en.wikipedia.org/wiki/Apollo_11"
        encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")

        class BingSession(_FakeSession):
            def get(self, url: str, **kwargs: object) -> _FakeResponse:
                self.calls.append((url, kwargs))
                body = (
                    '<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1'
                    + encoded
                    + '"><strong>Apollo 11</strong></a></h2></li>'
                ).encode()
                return _FakeResponse(body=body, content_type="text/html")

        session = BingSession()
        executor = WikipediaLiveExecutor(session=session, search_mode="bing")
        result = await executor(Invocation("search", {"query": ["Apollo 11"]}))
        self.assertEqual(result["results"][0]["url"], target)
        self.assertEqual(result["results"][0]["title"], "Apollo 11")
        self.assertEqual(result["_paste_transport"]["backend"], "bing_html_search")
        self.assertEqual(
            executor.transport_plan(
                Invocation("search", {"query": ["Apollo 11"]})
            ),
            {
                "backend": "bing_html_search",
                "request_host": "www.bing.com",
                "http_attempts": 1,
            },
        )
        await executor.close()

    async def test_jina_visit_uses_proxy_prefix(self) -> None:
        session = _FakeSession()
        executor = WikipediaLiveExecutor(session=session, visit_mode="jina")
        await executor(
            Invocation("visit", {"url": "https://example.test/page", "goal": "g"})
        )
        self.assertEqual(
            session.calls[-1][0],
            "https://r.jina.ai/https://example.test/page",
        )
        plan = executor.transport_plan(
            Invocation("visit", {"url": "https://example.test/page", "goal": "g"})
        )
        self.assertEqual(
            plan,
            {
                "backend": "r.jina.ai",
                "request_host": "r.jina.ai",
                "http_attempts": 1,
            },
        )
        await executor.close()


if __name__ == "__main__":
    unittest.main()
