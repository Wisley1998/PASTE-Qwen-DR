from __future__ import annotations

import asyncio
import aiohttp
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from paste_repro.pattern_v2_all_visit_online import PatternV2Prediction
from paste_repro.pattern_v2_strict_adapter import (
    PublicSLODurationPredictor,
    new_public_slo_duration_artifact,
)
from paste_repro.strict_trace_runtime import validate_signed_payload


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pattern_v2_strict_abef.py"
)
SPEC = importlib.util.spec_from_file_location("run_pattern_v2_strict_abef_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class _FakePatternSession:
    predictor_artifact_sha256 = "a" * 64

    def __init__(self) -> None:
        self.inputs = []

    def predict_after_tool(self, **kwargs):
        self.inputs.append(kwargs)
        if len(self.inputs) == 1:
            return (
                PatternV2Prediction(
                    url="HTTPS://Future.Test:443/article#fragment",
                    confidence=0.8,
                    source_position=1,
                    trigger_tool="search",
                ),
            )
        return ()

    def snapshot(self):
        return {"decisions": len(self.inputs)}


class _FakePredictor:
    artifact_sha256 = "a" * 64

    def __init__(self) -> None:
        self.sessions = []

    def start_session(self, *, source_session_id, runtime_session_id):
        assert source_session_id == "source-root"
        assert runtime_session_id == "runtime-0"
        session = _FakePatternSession()
        self.sessions.append(session)
        return session


class _FastClock:
    artifact_sha256 = "b" * 64

    def service_s(self, *, tool_name, tool_arguments):
        del tool_name, tool_arguments
        return 0.005


def _request(index: int, content: str) -> dict:
    return {
        "call_index": index,
        "messages": [{"role": "user", "content": content}],
        "prompt_tokens": 10 + index,
        "max_tokens": 4,
    }


def test_single_cell_uses_previous_tool_and_keeps_prediction_across_rounds(
    tmp_path: Path,
) -> None:
    post_attempts = 0

    async def fake_post_llm(session, **kwargs):
        nonlocal post_attempts
        del session
        post_attempts += 1
        if post_attempts == 1:
            raise aiohttp.ServerDisconnectedError()
        request = kwargs["request"]
        await asyncio.sleep(0.012)
        return (
            200,
            {
                "prompt_tokens": request["prompt_tokens"],
                "completion_tokens": request["max_tokens"],
            },
            "fixed live response",
        )

    target = "https://future.test/article"
    steps = [
        {
            "request": _request(0, "question"),
            "tools_after": [
                {
                    "outcome_id": "search-0",
                    "event_index": 1,
                    "call_index": 0,
                    "tool_name": "search",
                    "tool_args": {"query": ["first"]},
                }
            ],
        },
        {
            "request": _request(
                1,
                "<tool_response>1. [article](https://future.test/article)</tool_response>",
            ),
            "tools_after": [
                {
                    "outcome_id": "search-1",
                    "event_index": 3,
                    "call_index": 1,
                    "tool_name": "search",
                    "tool_args": {"query": ["second"]},
                }
            ],
        },
        {
            "request": _request(
                2,
                "<tool_response>no new candidate</tool_response>",
            ),
            "tools_after": [
                {
                    "outcome_id": "visit-2",
                    "event_index": 5,
                    "call_index": 2,
                    "tool_name": "visit",
                    "tool_args": {"url": target},
                }
            ],
        },
    ]
    public = {
        "plan_sha256": "c" * 64,
        "traces": [
            {
                "trace_id": "runtime-0",
                "session_id": "runtime-0",
                "source_session_id": "source-root",
                "release_offset_s": 0.0,
            }
        ],
    }
    sealed = {
        "sealed_sha256": "d" * 64,
        "trace_steps": {"runtime-0": steps},
        "outcomes": {
            "search-0": {"tool_name": "search", "event_index": 1},
            "search-1": {"tool_name": "search", "event_index": 3},
            "visit-2": {
                "tool_name": "visit",
                "event_index": 5,
                "visit_units": [{"url": target}],
            },
        },
    }
    predictor = _FakePredictor()
    loaded = runner.RuntimeInputs(
        public=public,
        sealed=sealed,
        predictor=predictor,
        duration_predictor=PublicSLODurationPredictor(
            new_public_slo_duration_artifact()
        ),
        service_clock=_FastClock(),
        tail_predictor=None,
        predictor_disclosure={"claim_scope": "test_only"},
        workload_contract="smoke_test",
        file_hashes={},
        formal_workload=False,
    )
    args = SimpleNamespace(
        cell="B",
        visit_capacity=64,
        speculative_cap=64,
        max_active_tasks=1,
        default_predicted_output_tokens=4.0,
        server_url="http://unused.test",
        model="fake-model",
        request_timeout_s=1.0,
        max_request_attempts=2,
        allow_usage_mismatch=False,
        output_dir=tmp_path / "result",
    )
    summary = asyncio.run(runner.execute_cell(args, loaded, post_llm=fake_post_llm))
    assert summary["failures"] == 0
    assert summary["tasks"] == 1
    assert summary["requests"] == 3
    assert summary["llm_request_attempts"] == 4
    assert summary["retried_requests"] == 1
    assert summary["visit"]["metrics"]["ready_cache_hits"] == 1
    assert summary["realized_visit_hit_rate"] == 1.0
    assert summary["configuration"]["candidate_ranking"] == "exact_probability_only"
    validate_signed_payload(summary, "result_sha256", label="test result")
    assert summary["claim_scope"] == "test_only"
    assert predictor.sessions[0].inputs[0]["tool_name"] == "search"
    assert predictor.sessions[0].inputs[0]["current_messages"] == steps[1]["request"]["messages"]
    transitions = json.loads(
        (args.output_dir / "speculation_execution_events.json").read_text()
    )
    assert not any("expired" in row["event"] or "resolved" in row["event"] for row in transitions)
    outcomes = json.loads((args.output_dir / "prediction_outcomes.json").read_text())
    assert outcomes[0]["outcome_scope"] == "any_later_same_session_authoritative_visit"
    assert outcomes[0]["decision_hit"] is True
    assert (args.output_dir / "result_manifest.json").is_file()


def test_plan_contract_rejects_multiple_tools_between_llm_turns() -> None:
    public = {
        "schema": runner.PUBLIC_PLAN_SCHEMA,
        "role": "crossfit",
        "call_graph_mode": "trace_replay_causal_reveal",
        "plan_sha256": "a" * 64,
        "independent_source_roots": 1,
        "replicas": 1,
        "traces": [
            {
                "trace_id": "r",
                "source_session_id": "s",
            }
        ],
    }
    sealed = {
        "schema": runner.SEALED_PLAN_SCHEMA,
        "role": "crossfit",
        "public_plan_sha256": "a" * 64,
        "trace_steps": {
            "r": [
                {
                    "request": _request(0, "q"),
                    "tools_after": [
                        {"outcome_id": "a"},
                        {"outcome_id": "b"},
                    ],
                }
            ]
        },
        "outcomes": {"a": {}, "b": {}},
    }
    try:
        runner._validate_plan_contract(
            public,
            sealed,
            predictor_schema=runner.pattern_online.SCHEMA,
            allow_smoke_workload=True,
        )
    except ValueError as exc:
        assert "at most one completed tool" in str(exc)
    else:
        raise AssertionError("multi-tool authority group was accepted")


def test_deployable_formal_contract_is_30_roots_x7() -> None:
    traces = []
    steps = {}
    for replica in range(7):
        for root in range(30):
            trace_id = f"r{replica}-s{root}"
            traces.append(
                {
                    "trace_id": trace_id,
                    "source_session_id": f"source-{root}",
                }
            )
            steps[trace_id] = [{"request": _request(0, "q"), "tools_after": []}]
    public = {
        "schema": runner.PUBLIC_PLAN_SCHEMA,
        "role": "final",
        "call_graph_mode": "trace_replay_causal_reveal",
        "plan_sha256": "e" * 64,
        "independent_source_roots": 30,
        "replicas_per_root": 7,
        "replicas": 210,
        "logical_corpus_sha256": runner.DEPLOYABLE_LOGICAL_CORPUS_SHA256,
        "traces": traces,
    }
    sealed = {
        "schema": runner.SEALED_PLAN_SCHEMA,
        "role": "final",
        "public_plan_sha256": "e" * 64,
        "trace_steps": steps,
        "outcomes": {},
    }
    formal, contract = runner._validate_plan_contract(
        public,
        sealed,
        predictor_schema=runner.pattern_online.DEPLOYABLE_SCHEMA,
        allow_smoke_workload=False,
    )
    assert formal is True
    assert contract == "retrospective_internal_holdout_30_roots_x7"
