from __future__ import annotations

import asyncio
import math
import unittest

from paste_repro.invocation import Invocation
from paste_repro.scheduler import SpeculativeScheduler


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_reuse_and_state_isolation(self) -> None:
        calls = []

        async def executor(invocation: Invocation):
            calls.append(invocation)
            await asyncio.sleep(0)
            return invocation.arguments

        scheduler = SpeculativeScheduler(executor, max_concurrency=1, ttl_s=10)
        predicted = Invocation("visit", {"url": "u", "options": {"b": 2, "a": 1}})
        # Different mapping order canonicalizes to the same complete arguments.
        authoritative = Invocation("visit", {"options": {"a": 1, "b": 2}, "url": "u"})
        await scheduler.speculate(predicted, session_id="s")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(scheduler.authoritative_state, ())

        result = await scheduler.authoritative(authoritative, session_id="s")
        self.assertEqual(result.source, "reused")
        self.assertEqual(len(calls), 1)
        self.assertEqual(scheduler.stats.completed_reuse, 1)
        self.assertEqual(scheduler.stats.commits, 1)
        await scheduler.close()

    async def test_full_argument_mismatch_is_a_miss_then_prediction_expires(self) -> None:
        async def executor(invocation: Invocation):
            await asyncio.sleep(0)
            return invocation.arguments

        scheduler = SpeculativeScheduler(executor, max_concurrency=1, ttl_s=10)
        await scheduler.speculate(
            Invocation("visit", {"url": "same", "goal": "one"}), session_id="s"
        )
        await asyncio.sleep(0)
        committed = await scheduler.authoritative(
            Invocation("visit", {"url": "same", "goal": "two"}), session_id="s"
        )
        self.assertEqual(committed.source, "executed")
        self.assertEqual(scheduler.stats.misses, 1)
        self.assertEqual(len(scheduler.authoritative_state), 1)
        await scheduler.sweep(now=math.inf, session_id="s")
        self.assertEqual(scheduler.stats.expired, 1)
        self.assertEqual(len(scheduler.authoritative_state), 1)

    async def test_inflight_match_is_promoted(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def executor(invocation: Invocation):
            started.set()
            await release.wait()
            return "done"

        scheduler = SpeculativeScheduler(executor, max_concurrency=1, ttl_s=10)
        invocation = Invocation("visit", {"url": "slow"})
        await scheduler.speculate(invocation, session_id="s")
        await started.wait()
        confirmation = asyncio.create_task(
            scheduler.authoritative(invocation, session_id="s")
        )
        await asyncio.sleep(0)
        self.assertFalse(confirmation.done())
        self.assertEqual(scheduler.authoritative_state, ())
        release.set()
        result = await confirmation
        self.assertEqual(result.source, "promoted")
        self.assertEqual(scheduler.stats.inflight_promotions, 1)
        self.assertEqual(scheduler.stats.commits, 1)

    async def test_pending_work_is_bounded(self) -> None:
        release = asyncio.Event()

        async def executor(invocation: Invocation):
            await release.wait()
            return invocation.arguments

        scheduler = SpeculativeScheduler(
            executor, max_concurrency=1, max_pending=1, ttl_s=10
        )
        self.assertTrue(await scheduler.speculate(Invocation("visit", {"url": "a"})))
        self.assertFalse(await scheduler.speculate(Invocation("visit", {"url": "b"})))
        self.assertEqual(scheduler.pending_count, 1)
        self.assertEqual(scheduler.stats.rejected_capacity, 1)
        release.set()
        await scheduler.close()


if __name__ == "__main__":
    unittest.main()

