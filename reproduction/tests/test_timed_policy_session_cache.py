"""CPU regressions for bounded session cache and cancellation-safe timers."""
import asyncio
import math
import unittest

from paste_repro.invocation import Invocation
from paste_repro.live_broker import LiveToolBroker


async def drain_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 2)


class TimedPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_positive_confidence_and_authority_priority(self):
        starts = []
        release = asyncio.Event()

        async def execute(inv):
            starts.append(inv.arguments["name"])
            if inv.arguments["name"] == "block":
                await release.wait()
            return {"ok": True}

        async with LiveToolBroker(execute, max_workers=1, max_speculative_workers=1) as broker:
            blocker = asyncio.create_task(broker.authoritative(Invocation("visit", {"name": "block"})))
            await drain_until(lambda: starts)
            await broker.speculate_batch([
                (Invocation("visit", {"name": "low"}), "t", 0.1),
                (Invocation("visit", {"name": "high"}), "t", 0.9),
            ])
            demand = asyncio.create_task(broker.authoritative(Invocation("visit", {"name": "authority"})))
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(blocker, demand)
            await drain_until(lambda: len(starts) == 4)
            self.assertEqual(starts, ["block", "authority", "high", "low"])

    async def test_completed_cache_does_not_consume_pending_slots_and_is_bounded(self):
        async def execute(inv):
            return {"value": inv.arguments["n"]}

        async with LiveToolBroker(execute, max_workers=1, max_speculative_workers=1,
                max_speculative_pending=1, max_completed_predictions=2, ttl_s=math.inf) as broker:
            for n in range(4):
                self.assertTrue(await broker.speculate(Invocation("visit", {"n": n})))
                await drain_until(lambda: broker.stats.speculative_completed == n + 1)
            self.assertEqual(broker.pending_speculative_count, 2)
            self.assertEqual(sum(r["outcome"] == "cache_evicted" for r in broker.tool_records()), 2)
            hit = await broker.authoritative(Invocation("visit", {"n": 3}))
            self.assertEqual(hit.source, "reused")

    async def test_repeat_reuse_is_session_scoped_and_cleanup_keeps_cost_once(self):
        starts = []
        async def execute(inv):
            starts.append(inv)
            return {"value": 1, "_paste_transport": {"reuse_valid_until_monotonic_s": 1e300}}

        broker = LiveToolBroker(execute, max_workers=2, max_speculative_workers=2,
            max_speculative_pending=2, max_completed_predictions=8, ttl_s=math.inf,
            retain_completed_predictions=True, require_reuse_validity=True)
        inv = Invocation("visit", {"url": "https://example.test/a"})
        await broker.speculate(inv, session_id="a", priority=.9)
        await drain_until(lambda: broker.stats.speculative_completed == 1)
        for _ in range(3):
            self.assertEqual((await broker.authoritative(inv, session_id="a")).source, "reused")
        self.assertEqual((await broker.authoritative(inv, session_id="b")).source, "executed")
        self.assertEqual(len(starts), 2)
        await broker.cancel_predictions(session_id="a")
        await broker.close()
        rows = broker.tool_records()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["outcome"], "committed")
        self.assertEqual(rows[0]["reuse_count"], 3)
        self.assertEqual(broker.resource_summary()["a"]["useful_speculative_calls"], 1)
        self.assertTrue(broker.resource_summary()["a"]["cleanup_complete"])

    async def test_preemption_releases_worker_without_waiting_for_wrong_timer(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        async def execute(inv):
            if inv.arguments["name"] == "wrong":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return {"ok": True}

        async with LiveToolBroker(execute, max_workers=1, max_speculative_workers=1,
                preempt_running_speculation=True) as broker:
            await broker.speculate(Invocation("visit", {"name": "wrong"}), priority=.1)
            await entered.wait()
            result = await asyncio.wait_for(broker.authoritative(Invocation("visit", {"name": "wanted"})), 2)
            self.assertEqual(result.source, "executed")
            self.assertTrue(cancelled.is_set())
            self.assertTrue(broker.tool_records()[0]["preempted_for_authority"])

    async def test_replacement_does_not_discard_completed_cache(self):
        release = asyncio.Event()
        entered = asyncio.Event()
        async def execute(inv):
            entered.set()
            await release.wait()
            return {"ok": True}
        async with LiveToolBroker(execute, max_workers=1, max_speculative_workers=1,
                max_speculative_pending=2, max_completed_predictions=4) as broker:
            await broker.speculate(Invocation("visit", {"n": 0}), priority=.5)
            await entered.wait()
            await broker.speculate(Invocation("visit", {"n": 1}), priority=.1)
            accepted = await broker.speculate(Invocation("visit", {"n": 2}), priority=.9,
                replace_lower_priority_queued=True)
            self.assertTrue(accepted)
            self.assertEqual(broker.stats.speculative_replaced_by_priority, 1)
            release.set()

    async def test_session_cache_survives_long_model_gap_but_checks_result_validity(self):
        now = [1.0]
        async def execute(inv):
            return {"ok": True, "_paste_transport": {"reuse_valid_until_monotonic_s": 1e300}}
        async with LiveToolBroker(execute, max_workers=1, max_speculative_workers=1,
                max_completed_predictions=4, retain_completed_predictions=True,
                ttl_s=math.inf, clock=lambda: now[0], require_reuse_validity=True) as broker:
            inv = Invocation("visit", {"url": "https://example.test/long-gap"})
            await broker.speculate(inv, priority=.9)
            await drain_until(lambda: broker.stats.speculative_completed == 1)
            now[0] += 600
            await broker.sweep()
            self.assertEqual((await broker.authoritative(inv)).source, "reused")
