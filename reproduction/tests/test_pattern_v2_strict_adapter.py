from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from paste_repro.pattern_v2_all_visit_online import PatternV2Prediction
from paste_repro.pattern_v2_strict_adapter import (
    HashedUniformSLOClock,
    PatternV2StrictCandidate,
    PatternV2StrictPolicy,
    PersistentPatternV2ToolExecutor,
    PublicSLODurationPredictor,
    new_hashed_slo_clock_artifact,
    new_public_slo_duration_artifact,
)
from paste_repro.trace_coscheduler import AsyncPreemptibleVisitPool
from paste_repro.strict_trace_runtime import (
    CausalSessionState,
    POLICY_NAME as SCHEDULER_METADATA_SCHEMA,
    TailPrediction,
)


def test_hashed_slo_clock_is_invocation_only_and_uses_frozen_ranges() -> None:
    clock = HashedUniformSLOClock(
        new_hashed_slo_clock_artifact(seed_sha256="a" * 64)
    )
    first = clock.service_s(
        tool_name="visit", tool_arguments={"url": "HTTPS://Example.Test:443/a#x"}
    )
    normalized = clock.service_s(
        tool_name="visit", tool_arguments={"url": "https://example.test/a"}
    )
    assert first == normalized
    assert 2.0 <= first <= 8.0
    assert 1.0 <= clock.service_s(
        tool_name="search", tool_arguments={"query": ["causal"]}
    ) <= 3.0
    with pytest.raises(ValueError, match="no frozen physical SLO"):
        clock.service_s(tool_name="unknown", tool_arguments={})


def test_policy_duration_estimate_has_no_clock_seed_or_trace_timing() -> None:
    artifact = new_public_slo_duration_artifact(ewma_alpha=0.5)
    assert "seed_sha256" not in artifact
    assert artifact["uses_clock_seed"] is False
    assert artifact["uses_trace_timing"] is False
    predictor = PublicSLODurationPredictor(artifact)
    assert predictor.estimate("visit").service_s == 5.0
    predictor.observe_completed("visit", 7.0, "https://ignored.test/a")
    estimate = predictor.estimate("visit")
    assert estimate.service_s == 6.0
    assert estimate.source.endswith("completed_job_ewma")


@dataclass
class _FakePatternSession:
    predictor_artifact_sha256: str = "b" * 64

    def predict_after_tool(self, **kwargs):
        assert kwargs["tool_name"] == "visit"
        assert kwargs["tool_arguments"] == {"url": "https://seen.test/a"}
        assert kwargs["current_messages"] == [{"role": "user", "content": "visible"}]
        return (
            PatternV2Prediction(
                url="https://future.test/a",
                confidence=0.75,
                source_position=3,
                trigger_tool="visit",
            ),
        )

    def snapshot(self):
        return {"decisions": 1}


class _FakeCrossFitPredictor:
    artifact_sha256 = "b" * 64

    def start_session(self, *, source_session_id, runtime_session_id):
        assert source_session_id == "source"
        assert runtime_session_id == "runtime"
        return _FakePatternSession()


class _FakeTailPredictor:
    def predict(self, **kwargs):
        assert kwargs["current_call_index"] == 2
        assert kwargs["completed_tool_group_waits_s"] == [4.0]
        return TailPrediction(
            next_tool_wait_s=5.0,
            remaining_tool_wait_s=11.0,
            next_tool_probability=0.8,
            reliability=0.5,
        )


def test_policy_adapter_passes_only_completed_tool_and_current_messages() -> None:
    durations = PublicSLODurationPredictor(new_public_slo_duration_artifact())
    policy = PatternV2StrictPolicy(
        predictor=_FakeCrossFitPredictor(),  # type: ignore[arg-type]
        duration_predictor=durations,
    )
    session = policy.start_session(
        source_session_id="source", runtime_session_id="runtime"
    )
    candidates = session.predict_after_completed_tool(
        tool_name="visit",
        tool_arguments={"url": "https://seen.test/a"},
        current_messages=[{"role": "user", "content": "visible"}],
    )
    assert len(candidates) == 1
    assert candidates[0].url == "https://future.test/a"
    assert candidates[0].admission_score == 0.75
    assert candidates[0].predicted_service_s == 5.0


def test_scheduler_metadata_uses_server_causal_wire_schema() -> None:
    durations = PublicSLODurationPredictor(new_public_slo_duration_artifact())
    policy = PatternV2StrictPolicy(
        predictor=_FakeCrossFitPredictor(),  # type: ignore[arg-type]
        duration_predictor=durations,
        tail_predictor=_FakeTailPredictor(),  # type: ignore[arg-type]
    )
    state = CausalSessionState(
        predicted_output_tokens=73.0,
        completed_tool_group_waits_s=[4.0],
    )

    meta = policy.scheduler_metadata(
        trace_id="runtime",
        request_index=2,
        current_call_index=2,
        prompt_tokens=123,
        max_tokens=128,
        state=state,
        observed_event_seq=7,
        decision_seq=8,
    )

    assert meta["ms"] == SCHEDULER_METADATA_SCHEMA
    assert meta["po_hat"] == 73
    assert meta["tool_eta_s_hat"] == 5.0
    assert meta["tool_hit_probability_hat"] == pytest.approx(0.8)
    assert meta["tool_eta_reliability_hat"] == pytest.approx(0.5)
    assert meta["remaining_tool_wait_s_hat"] == 11.0


class _FastClock:
    artifact_sha256 = "c" * 64

    def __init__(self, duration_s: float) -> None:
        self.duration_s = duration_s

    def service_s(self, *, tool_name, tool_arguments):
        del tool_name, tool_arguments
        return self.duration_s


def _candidate(url: str, confidence: float, predicted_service_s: float = 5.0):
    return PatternV2StrictCandidate(
        url=url,
        confidence=confidence,
        predicted_service_s=predicted_service_s,
        prediction_source="public_SLO_midpoint",
        source_position=1,
        trigger_tool="search",
    )


def test_persistent_executor_keeps_completed_candidate_for_later_authority() -> None:
    async def scenario() -> None:
        url = "https://future.test/a"
        predicted_url = "HTTPS://Future.Test:443/a#fragment"
        outcomes = {
            "outcome": {
                "tool_name": "visit",
                "event_index": 7,
                "visit_units": [{"url": url}],
            }
        }
        durations = PublicSLODurationPredictor(new_public_slo_duration_artifact())
        pool = AsyncPreemptibleVisitPool(capacity=64, speculative_cap=64)
        executor = PersistentPatternV2ToolExecutor(
            sealed_outcomes=outcomes,
            service_clock=_FastClock(0.01),  # type: ignore[arg-type]
            duration_predictor=durations,
            visit_pool=pool,
        )
        assert not hasattr(executor, "expire_prediction_window")
        assert not hasattr(executor, "reveal_prediction_outcome")
        assert await executor.speculate(
            session_id="s",
            candidates=[_candidate(predicted_url, 0.7)],
            decision_id="d0",
        ) == (True,)
        await asyncio.sleep(0.025)
        observation = await executor.execute_authoritative(
            session_id="s",
            descriptor={
                "outcome_id": "outcome",
                "tool_name": "visit",
                "event_index": 7,
                "tool_args": {"url": url},
            },
        )
        assert observation.visit_results[0].source == "reused"
        assert executor.snapshot()["metrics"]["ready_cache_hits"] == 1
        await executor.close_session("s")
        await executor.close()

    asyncio.run(scenario())


def test_persistent_executor_preemption_uses_probability_not_duration_hat() -> None:
    async def scenario() -> None:
        transitions: list[dict] = []
        durations = PublicSLODurationPredictor(new_public_slo_duration_artifact())
        pool = AsyncPreemptibleVisitPool(
            capacity=64,
            speculative_cap=64,
            job_event_callback=transitions.append,
        )
        executor = PersistentPatternV2ToolExecutor(
            sealed_outcomes={
                "authority": {
                    "tool_name": "visit",
                    "event_index": 9,
                    "visit_units": [{"url": "https://authority.test/a"}],
                }
            },
            service_clock=_FastClock(0.2),  # type: ignore[arg-type]
            duration_predictor=durations,
            visit_pool=pool,
        )
        candidates = [
            _candidate(
                f"https://spec.test/{index}",
                confidence=(0.01 if index == 0 else 0.5 + index / 1000.0),
                predicted_service_s=(1000.0 if index == 0 else 0.001),
            )
            for index in range(64)
        ]
        await executor.speculate(
            session_id="s", candidates=candidates, decision_id="decision"
        )
        await asyncio.sleep(0.01)
        observation = await executor.execute_authoritative(
            session_id="s",
            descriptor={
                "outcome_id": "authority",
                "tool_name": "visit",
                "event_index": 9,
                "tool_args": {"url": "https://authority.test/a"},
            },
        )
        assert observation.visit_results[0].source == "executed"
        cancelled = [row for row in transitions if row["event"] == "cancelled_preempted"]
        assert len(cancelled) == 1
        assert cancelled[0]["url"] == "https://spec.test/0"
        await executor.close_session("s")
        await executor.close()

    asyncio.run(scenario())
