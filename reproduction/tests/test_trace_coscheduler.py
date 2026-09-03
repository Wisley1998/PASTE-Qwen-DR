from __future__ import annotations

import asyncio

from paste_repro.trace_coscheduler import (
    AdmissionTurn,
    AsyncPreemptibleVisitPool,
    GainPressureAdmissionController,
)


def test_visit_pool_promotes_inflight_and_preserves_progress() -> None:
    async def scenario() -> None:
        pool = AsyncPreemptibleVisitPool(capacity=1, speculative_cap=1)
        admitted = await pool.speculate_batch(
            [("s", "https://example.test/a", 0.06, 0.9, "d")]
        )
        assert admitted == (True,)
        await asyncio.sleep(0.025)
        started = asyncio.get_running_loop().time()
        result = await pool.authoritative(
            session_id="s", url="https://example.test/a", duration_s=0.06
        )
        elapsed = asyncio.get_running_loop().time() - started
        assert result.source == "promoted_inflight"
        assert result.saved_service_s > 0.0
        assert elapsed < 0.055
        snapshot = pool.snapshot()
        assert snapshot["metrics"]["cache_hits"] == 1
        assert snapshot["metrics"]["physical_speculative_starts"] == 1
        assert snapshot["metrics"].get("physical_authority_starts", 0) == 0
        await pool.close()

    asyncio.run(scenario())


def test_visit_pool_preempts_lowest_score_for_authority() -> None:
    async def scenario() -> None:
        pool = AsyncPreemptibleVisitPool(capacity=2, speculative_cap=2)
        await pool.speculate_batch(
            [
                ("s1", "https://example.test/low", 0.20, 0.1, "d1"),
                ("s2", "https://example.test/high", 0.20, 0.9, "d2"),
            ]
        )
        await asyncio.sleep(0.015)
        result = await pool.authoritative(
            session_id="s3", url="https://example.test/auth", duration_s=0.01
        )
        assert result.source == "executed"
        snapshot = pool.snapshot()
        assert snapshot["metrics"]["preempted_speculations"] == 1
        assert snapshot["metrics"]["physical_authority_starts"] == 1
        await pool.close()

    asyncio.run(scenario())


def test_visit_pool_completed_cache_is_session_scoped() -> None:
    async def scenario() -> None:
        pool = AsyncPreemptibleVisitPool(capacity=1, speculative_cap=1)
        url = "https://example.test/a"
        await pool.speculate_batch([("s1", url, 0.01, 1.0, "d")])
        await asyncio.sleep(0.025)
        reused = await pool.authoritative(session_id="s1", url=url, duration_s=0.5)
        assert reused.source == "reused"
        executed = await pool.authoritative(session_id="s2", url=url, duration_s=0.01)
        assert executed.source == "executed"
        await pool.close()

    asyncio.run(scenario())


def test_admission_ranks_gain_efficiency_and_reopens_cold_gate() -> None:
    async def scenario() -> None:
        controller = GainPressureAdmissionController(
            pressure_low=1,
            pressure_high=1,
            cold_session_cap=1,
            gain_weight=1.0,
            aging_weight=0.0,
        )
        await controller.acquire(
            AdmissionTurn("running", True, 0.0, 1.0, 100)
        )
        order: list[str] = []

        async def wait(name: str, gain: float) -> None:
            await controller.acquire(
                AdmissionTurn(name, False, gain, 1.0, 100)
            )
            order.append(name)

        low = asyncio.create_task(wait("low", 0.1))
        high = asyncio.create_task(wait("high", 1.0))
        await asyncio.sleep(0)
        await controller.release("running")
        await asyncio.sleep(0)
        assert order == ["high"]
        await controller.release("high")
        await asyncio.sleep(0)
        assert order == ["high", "low"]
        await controller.release("low")
        await asyncio.gather(low, high)
        await controller.finish_session("running")
        await controller.finish_session("high")
        await controller.finish_session("low")
        await controller.close()

    asyncio.run(scenario())


def test_admission_engine_pressure_includes_weighted_kv_load() -> None:
    async def scenario() -> None:
        controller = GainPressureAdmissionController(
            pressure_low=2,
            pressure_high=3,
            cold_session_cap=3,
            kv_weight=2.0,
        )
        await controller.acquire(
            AdmissionTurn("first", True, 0.0, 1.0, 100, kv_load=1.0)
        )
        blocked = asyncio.create_task(
            controller.acquire(
                AdmissionTurn("second", True, 0.0, 1.0, 100, kv_load=0.1)
            )
        )
        await asyncio.sleep(0)
        assert not blocked.done()
        snapshot = controller.snapshot()
        assert snapshot["engine_pressure"] == 3.0
        await controller.release("first")
        await blocked
        await controller.release("second")
        await controller.finish_session("first")
        await controller.finish_session("second")
        await controller.close()

    asyncio.run(scenario())
