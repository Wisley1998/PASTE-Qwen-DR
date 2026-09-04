from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from paste_repro.mapper import URLRankMapper
from paste_repro.strict_trace_runtime import (
    CalibrationHashedServiceClock,
    CausalDurationPredictor,
    CausalSessionState,
    CausalTailPredictor,
    CausalTraceCursor,
    SealedTraceToolExecutor,
    StrictCandidate,
    StrictOnlinePolicy,
    SERVICE_CLOCK_SCHEMA,
    normalize_url,
    signed_payload,
)
from paste_repro.trace_coscheduler import AsyncPreemptibleVisitPool
from paste_repro.traces import LLMCall, SearchResult, SearchVisitTransition, SessionTrace, ToolCall


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_strict_trace_abef.py"
SPEC = importlib.util.spec_from_file_location("run_strict_trace_abef", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_strict_causal_experiment.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_strict_causal_experiment_for_qwen_test", AUDIT_SCRIPT
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
strict_audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(strict_audit)
from trace_experiment_lib import _truncate_messages_to_fit


class TinyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return list(range(sum(len(str(row.get("content", ""))) for row in messages) + 1))


def _correction(total: float, units: list[float] | None = None) -> dict:
    return {"duration_s": total, "unit_duration_s": list(units or [total])}


def _session(path: Path, *, visit_duration: float = 0.02, chosen_rank: int = 2) -> SessionTrace:
    urls = ("https://example.test/one", "https://example.test/two")
    links = "\n".join(
        f"{rank}. [result {rank}]({url})"
        for rank, url in enumerate(urls, start=1)
    )
    path.write_text("fixture\n", encoding="utf-8")
    return SessionTrace(
        path,
        (
            LLMCall(0, 0.0, 0.1, 0.1, ({"role": "user", "content": "question"},), "", 1),
            ToolCall(0, 0.1, "search", {"query": ["q"]}, 2, _correction(0.01)),
            LLMCall(
                1,
                0.2,
                0.1,
                0.1,
                ({"role": "user", "content": f"<tool_response>\n{links}\n</tool_response>"},),
                "",
                3,
            ),
            ToolCall(
                1,
                0.3,
                "visit",
                {"url": urls[chosen_rank - 1]},
                4,
                _correction(visit_duration),
            ),
            LLMCall(2, 0.4, 0.1, 0.1, ({"role": "user", "content": "done"},), "", 5),
        ),
    )


def _transition(rank: int) -> SearchVisitTransition:
    urls = ("https://train.test/one", "https://train.test/two")
    return SearchVisitTransition(
        session_id=f"train-{rank}",
        search=ToolCall(0, 0.0, "search", {"query": ["q"]}, 1),
        decision_llm=LLMCall(1, 0.0, 0.0, 0.0, (), "", 2),
        visit=ToolCall(1, 0.0, "visit", {"url": urls[rank - 1]}, 3),
        completion_llm=None,
        search_results=tuple(
            SearchResult(url, index, index - 1, 0)
            for index, url in enumerate(urls, start=1)
        ),
        authoritative_urls=(urls[rank - 1],),
        baseline_stall_s=1.0,
        overlap_window_s=1.0,
    )


def _policy(tmp_path: Path) -> StrictOnlinePolicy:
    calibration = tuple(
        _session(tmp_path / f"cal-{index}.jsonl", visit_duration=0.01 + index / 1000)
        for index in range(3)
    )
    session_ids = [row.session_id for row in calibration]
    provenance = {
        "session_ids": session_ids,
        "session_ids_sha256": runner.canonical_sha256(sorted(session_ids)),
    }
    duration, _ = CausalDurationPredictor.fit(
        calibration, training_provenance=provenance
    )
    tail, _ = CausalTailPredictor.fit(
        calibration,
        training_provenance=provenance,
        duration_predictor=duration,
    )
    mapper = URLRankMapper().fit([_transition(2), _transition(2), _transition(1)])
    return StrictOnlinePolicy(
        mapper=mapper,
        mapper_artifact_sha256="a" * 64,
        duration_predictor=duration,
        tail_predictor=tail,
        top_k=1,
    )


def _service_clock(samples: dict[str, list[float]] | None = None):
    artifact = signed_payload(
        {
            "schema": SERVICE_CLOCK_SCHEMA,
            "physical_service_clock_mode": "calibration_hashed_empirical_v1",
            "training_role": "calibration",
            "training_provenance": {"session_ids": ["calibration-only"]},
            "uses_evaluation_labels": False,
            "enumerates_evaluation_invocations": False,
            "future_state_accepted_invariant": True,
            "minimum_selection_pool_size": 3,
            "seed_sha256": "b" * 64,
            "canonicalization": "test normalized canonical-json",
            "selection_rule": "test",
            "samples_by_tool_s": samples
            or {
                "visit": [0.02, 0.02, 0.02],
                "search": [0.01, 0.01, 0.01],
                "__global__": [0.01, 0.02, 0.03],
            },
        },
        "artifact_sha256",
    )
    return CalibrationHashedServiceClock(artifact), artifact


def test_policy_uses_current_response_and_emits_only_hat_metadata(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    messages = [
        {
            "role": "user",
            "content": (
                "<tool_response>\n"
                "1. [one](https://current.test/one)\n"
                "2. [two](https://current.test/two)\n"
                "</tool_response>"
            ),
        }
    ]
    candidates = policy.materialize_candidates(
        current_messages=messages, last_completed_tool_name="search"
    )
    assert [row.url for row in candidates] == ["https://current.test/two"]
    assert not policy.materialize_candidates(
        current_messages=messages, last_completed_tool_name="visit"
    )

    meta = policy.scheduler_metadata(
        trace_id="opaque",
        request_index=1,
        current_call_index=1,
        prompt_tokens=100,
        max_tokens=128,
        state=CausalSessionState(predicted_output_tokens=64),
        observed_event_seq=4,
        decision_seq=5,
    )
    forbidden = {"n", "rc", "rlmt", "npt", "nmt", "nw", "nwc", "rtw", "eg", "is_final"}
    assert forbidden.isdisjoint(meta)
    assert meta["ms"] == "paste.schedx.causal_prediction.v1"
    assert meta["po_hat"] == 64
    assert "tool_eta_s_hat" in meta
    assert "remaining_tool_wait_s_hat" in meta
    assert meta["decision_seq"] == 5
    assert meta["observed_event_seq"] == 4
    assert len(meta["policy_sha256"]) == 64


def test_cursor_hides_authority_until_live_llm_completion() -> None:
    cursor = CausalTraceCursor(
        [
            {
                "request": {"call_index": 0, "messages": []},
                "tools_after": [{"tool_name": "visit", "tool_args": {"url": "secret"}}],
            }
        ]
    )
    assert cursor.current_request()["call_index"] == 0
    with pytest.raises(RuntimeError, match="hidden until the LLM completes"):
        cursor.reveal_authoritative_tools()
    cursor.mark_llm_completed()
    assert cursor.reveal_authoritative_tools()[0]["tool_args"]["url"] == "secret"
    cursor.advance()
    assert cursor.done


def test_executor_uses_presealed_service_not_prediction_or_future_hit(tmp_path: Path) -> None:
    async def scenario() -> None:
        policy = _policy(tmp_path)
        duration = policy.duration_predictor
        pool = AsyncPreemptibleVisitPool(capacity=1, speculative_cap=1)
        session_id = "instance"
        url = "https://current.test/two"
        service_clock, _ = _service_clock()
        executor = SealedTraceToolExecutor(
            sealed_outcomes={
                "outcome": {
                    "session_id": session_id,
                    "event_index": 4,
                    "tool_name": "visit",
                    "visit_units": [{"url": url}],
                }
            },
            service_clock=service_clock,
            duration_predictor=duration,
            visit_pool=pool,
        )
        # The policy prediction is intentionally absurd.  Physical completion
        # must still follow the presealed 20 ms service surface.
        candidate = StrictCandidate(url, 1.0, 99.0, "test")
        await executor.speculate(
            session_id=session_id,
            candidates=[candidate],
            after_event_index=2,
            decision_id="d",
        )
        await asyncio.sleep(0.04)
        started = asyncio.get_running_loop().time()
        result = await executor.execute_authoritative(
            session_id=session_id,
            descriptor={
                "outcome_id": "outcome",
                "event_index": 4,
                "tool_name": "visit",
                "tool_args": {"url": url},
            },
        )
        assert asyncio.get_running_loop().time() - started < 0.015
        assert result.saved_service_s > 0
        await executor.close_session(session_id)
        await executor.close()

    asyncio.run(scenario())


def test_generic_service_clock_supports_unseen_counterfactual_candidate(tmp_path: Path) -> None:
    async def scenario() -> None:
        policy = _policy(tmp_path)
        service_clock, _ = _service_clock(
            {"visit": [0.001], "__global__": [0.001, 0.002, 0.003]}
        )
        executor = SealedTraceToolExecutor(
            sealed_outcomes={},
            service_clock=service_clock,
            duration_predictor=policy.duration_predictor,
            visit_pool=AsyncPreemptibleVisitPool(capacity=1, speculative_cap=1),
        )
        admitted = await executor.speculate(
            session_id="s",
            candidates=[StrictCandidate("https://never-enumerated.test", 1.0, 99.0, "test")],
            after_event_index=0,
            decision_id="d",
        )
        assert admitted == (True,)
        await asyncio.sleep(0.01)
        await executor.close()

    asyncio.run(scenario())


def test_service_clock_rejects_zero_cost_work() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _service_clock({"visit": [0.0], "__global__": [0.01, 0.02, 0.03]})


def test_service_clock_rejects_future_acceptance_dependent_contract() -> None:
    _, artifact = _service_clock()
    poisoned = dict(artifact)
    poisoned["future_state_accepted_invariant"] = False
    poisoned = signed_payload(poisoned, "artifact_sha256")
    with pytest.raises(ValueError, match="future acceptance"):
        CalibrationHashedServiceClock(poisoned)


def test_service_clock_singleton_tool_pool_uses_global_fallback() -> None:
    clock, _ = _service_clock(
        {"visit": [999.0], "__global__": [0.01, 0.02, 0.03]}
    )
    assert clock.service_s(
        tool_name="visit", tool_arguments={"url": "https://counterfactual.test"}
    ) in {0.01, 0.02, 0.03}


def test_prepare_service_clock_reuse_retains_complete_signed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = {
        "fixed_split_manifest_sha256": "1" * 64,
        "session_ids": ["calibration-a", "calibration-b"],
        "session_ids_sha256": "2" * 64,
        "raw_source_sha256": {
            "calibration-a": "3" * 64,
            "calibration-b": "4" * 64,
        },
        "execution_trace_sha256": {
            "calibration-a": "5" * 64,
            "calibration-b": "6" * 64,
        },
    }
    samples = {
        "search": (0.01, 0.02, 0.03),
        "visit": (1.0, 2.0, 3.0),
        "__global__": (0.01, 0.02, 0.03, 1.0, 2.0, 3.0),
    }
    artifact = runner._new_service_clock_artifact(
        training_provenance=provenance,
        samples_by_tool_s=samples,
        seed_sha256="a" * 64,
    )
    reuse_path = tmp_path / "old-freeze" / "service_clock.json"
    runner.write_json(reuse_path, artifact)
    monkeypatch.setattr(
        runner.secrets,
        "token_hex",
        lambda _size: pytest.fail("reuse must not sample a new private salt"),
    )

    retained = runner._prepare_service_clock_artifact(
        reuse_path=reuse_path,
        training_provenance=provenance,
        samples_by_tool_s=samples,
    )
    assert retained == artifact
    assert retained["artifact_sha256"] == artifact["artifact_sha256"]
    assert retained["seed_sha256"] == "a" * 64

    new_freeze_path = tmp_path / "new-freeze" / "service_clock.json"
    runner.write_json(new_freeze_path, retained)
    assert runner.read_json(new_freeze_path) == artifact
    assert runner.file_sha256(new_freeze_path) == runner.file_sha256(reuse_path)
    parsed = runner.build_parser().parse_args(
        [
            "prepare",
            "--output-dir",
            str(tmp_path / "unused"),
            "--reuse-service-clock",
            str(reuse_path),
        ]
    )
    assert parsed.reuse_service_clock == reuse_path


def test_prepare_service_clock_default_mints_private_salt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = {
        "fixed_split_manifest_sha256": "1" * 64,
        "session_ids": ["calibration-only"],
        "session_ids_sha256": "2" * 64,
        "raw_source_sha256": {"calibration-only": "3" * 64},
        "execution_trace_sha256": {"calibration-only": "4" * 64},
    }
    samples = {
        "visit": (1.0, 2.0, 3.0),
        "__global__": (1.0, 2.0, 3.0),
    }
    monkeypatch.setattr(runner.secrets, "token_hex", lambda size: "c" * (size * 2))
    artifact = runner._prepare_service_clock_artifact(
        reuse_path=None,
        training_provenance=provenance,
        samples_by_tool_s=samples,
    )
    assert artifact["seed_sha256"] == "c" * 64
    assert CalibrationHashedServiceClock(artifact).artifact_sha256 == artifact[
        "artifact_sha256"
    ]


@pytest.mark.parametrize(
    ("poison", "message"),
    (
        ("provenance", "calibration provenance"),
        ("samples", "sample pools"),
        ("signature", "checksum mismatch"),
    ),
)
def test_prepare_service_clock_reuse_rejects_poisoned_calibration_binding(
    tmp_path: Path, poison: str, message: str
) -> None:
    provenance = {
        "fixed_split_manifest_sha256": "1" * 64,
        "session_ids": ["calibration-only"],
        "session_ids_sha256": "2" * 64,
        "raw_source_sha256": {"calibration-only": "3" * 64},
        "execution_trace_sha256": {"calibration-only": "4" * 64},
    }
    samples = {
        "visit": (1.0, 2.0, 3.0),
        "__global__": (1.0, 2.0, 3.0),
    }
    artifact = runner._new_service_clock_artifact(
        training_provenance=provenance,
        samples_by_tool_s=samples,
        seed_sha256="a" * 64,
    )
    poisoned = json.loads(json.dumps(artifact))
    if poison == "provenance":
        poisoned["training_provenance"]["session_ids"] = ["wrong-root"]
        poisoned = signed_payload(poisoned, "artifact_sha256")
    elif poison == "samples":
        poisoned["samples_by_tool_s"]["visit"][0] = 999.0
        poisoned = signed_payload(poisoned, "artifact_sha256")
    else:
        poisoned["seed_sha256"] = "b" * 64
    reuse_path = tmp_path / f"{poison}.json"
    runner.write_json(reuse_path, poisoned)

    with pytest.raises(ValueError, match=message):
        runner._prepare_service_clock_artifact(
            reuse_path=reuse_path,
            training_provenance=provenance,
            samples_by_tool_s=samples,
        )


def test_prepare_service_clock_reuse_rejects_execution_contract_poison() -> None:
    provenance = {
        "fixed_split_manifest_sha256": "1" * 64,
        "session_ids": ["calibration-only"],
        "session_ids_sha256": "2" * 64,
        "raw_source_sha256": {"calibration-only": "3" * 64},
        "execution_trace_sha256": {"calibration-only": "4" * 64},
    }
    samples = {
        "visit": (1.0, 2.0, 3.0),
        "__global__": (1.0, 2.0, 3.0),
    }
    artifact = runner._new_service_clock_artifact(
        training_provenance=provenance,
        samples_by_tool_s=samples,
        seed_sha256="a" * 64,
    )
    poisons = (
        ("schema", "wrong-schema"),
        ("physical_service_clock_mode", "oracle"),
        ("training_role", "final"),
        ("uses_evaluation_labels", True),
        ("enumerates_evaluation_invocations", True),
        ("future_state_accepted_invariant", False),
        ("minimum_selection_pool_size", 4),
        ("canonicalization", "raw arguments"),
        ("selection_rule", "select future hit"),
    )
    for field, value in poisons:
        poisoned = {**artifact, field: value}
        poisoned = signed_payload(poisoned, "artifact_sha256")
        with pytest.raises(ValueError, match=field):
            runner._validate_service_clock_for_current_calibration(
                poisoned,
                training_provenance=provenance,
                samples_by_tool_s=samples,
            )

    extra_field = {**artifact, "evaluation_invocation_keys": ["future"]}
    extra_field = signed_payload(extra_field, "artifact_sha256")
    with pytest.raises(ValueError, match="fields do not match"):
        runner._validate_service_clock_for_current_calibration(
            extra_field,
            training_provenance=provenance,
            samples_by_tool_s=samples,
        )


def test_duration_evidence_predicts_all_units_before_private_clock() -> None:
    calls: list[str] = []

    class Prediction:
        service_s = 0.25

    class Predictor:
        def estimate(self, _tool_name, _url=None):
            calls.append("predict")
            return Prediction()

    class Clock:
        def service_s(self, **_kwargs):
            calls.append("clock")
            return 0.5

    evidence = runner._causal_tool_duration_evidence(
        duration_predictor=Predictor(),
        service_clock=Clock(),
        descriptor={
            "tool_name": "visit",
            "tool_args": {"url": ["https://one.test", "https://two.test"]},
        },
    )
    assert calls == ["predict", "predict", "clock", "clock"]
    assert evidence == {
        "tool_service_s_hat": 0.5,
        "assigned_service_s": 1.0,
        "duration_prediction_absolute_error_s": 0.5,
    }


def test_eval_duration_poison_does_not_change_service_surface(tmp_path: Path) -> None:
    first = _session(tmp_path / "eval-first.jsonl", visit_duration=0.02)
    kwargs = {
        "role": "final",
        "raw_sha_by_id": {first.session_id: "a" * 64},
        "tokenizer": TinyTokenizer(),
        "max_model_len": 10000,
        "output_cap": 128,
        "arrivals": [{"release_offset_s": 0.0, "arrival_index": 0}],
        "arrival_provenance": {"kind": "test"},
        "service_clock_artifact_sha256": _service_clock()[1]["artifact_sha256"],
    }
    public_a, sealed_a, diagnostics_a = runner.build_role_plans(
        sessions=[first], **kwargs
    )

    def with_visit_correction(correction) -> SessionTrace:
        events = list(first.events)
        visit = events[3]
        assert isinstance(visit, ToolCall)
        events[3] = ToolCall(
            visit.call_index,
            visit.timestamp_s,
            visit.tool_name,
            visit.tool_args,
            visit.line_number,
            correction,
        )
        return SessionTrace(first.path, tuple(events))

    poisons = (
        {"duration_s": 9000.0, "unit_duration_s": [9000.0]},
        None,
        {"duration_s": 0.0, "unit_duration_s": []},
        {"duration_s": 1.0, "unit_duration_s": [0.1, 0.9]},
        {"duration_s": "not-a-number", "unit_duration_s": "not-a-list"},
        {"duration_s": float("nan"), "unit_duration_s": [float("inf")]},
    )
    observed_diagnostics = []
    for poison in poisons:
        public_b, sealed_b, diagnostics_b = runner.build_role_plans(
            sessions=[with_visit_correction(poison)], **kwargs
        )
        assert public_b == public_a
        assert sealed_b == sealed_a
        observed_diagnostics.append(diagnostics_b)

    # The diagnostic sidecar changes, proving the poisons were present, while
    # none can alter the execution graph or physical service-clock binding.
    assert any(value != diagnostics_a for value in observed_diagnostics)
    assert "heldout_diagnostics" not in json.dumps(sealed_a, sort_keys=True)


def test_task_timing_evidence_recomputes_flow_from_raw_monotonic_bounds() -> None:
    row = runner._task_timing_evidence(
        experiment_started_monotonic_s=100.0,
        release_offset_s=2.0,
        released_at_monotonic_s=103.0,
        gate_acquired_at_monotonic_s=104.5,
        task_terminal_monotonic_s=110.0,
    )
    assert row["scheduled_release_monotonic_s"] == 102.0
    assert row["release_lag_s"] == 1.0
    assert row["task_gate_wait_s"] == 1.5
    assert row["flow_s"] == 8.0
    with pytest.raises(RuntimeError, match="order"):
        runner._task_timing_evidence(
            experiment_started_monotonic_s=100.0,
            release_offset_s=2.0,
            released_at_monotonic_s=101.0,
            gate_acquired_at_monotonic_s=104.0,
            task_terminal_monotonic_s=110.0,
        )


def test_model_snapshot_inventory_binds_weights_tokenizer_and_symlink_content(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text('{"v": 1}\n', encoding="utf-8")
    weight = blobs / "model.safetensors"
    weight.write_bytes(b"sealed-weight-bytes")
    (snapshot / "model.safetensors").symlink_to(weight)
    first = runner._model_snapshot_inventory(snapshot)
    assert first["file_count"] == 3
    assert {row["relative_path"] for row in first["files"]} == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }
    weight.write_bytes(b"poisoned-weight-bytes")
    second = runner._model_snapshot_inventory(snapshot)
    assert first["inventory_sha256"] != second["inventory_sha256"]


def test_public_plan_contains_no_execution_duration(tmp_path: Path) -> None:
    session = _session(tmp_path / "eval.jsonl")
    public, sealed, diagnostics = runner.build_role_plans(
        role="final",
        sessions=[session],
        raw_sha_by_id={session.session_id: "a" * 64},
        tokenizer=TinyTokenizer(),
        max_model_len=10000,
        output_cap=128,
        arrivals=[{"release_offset_s": 0.0, "arrival_index": 0}],
        arrival_provenance={"kind": "test"},
        service_clock_artifact_sha256=_service_clock()[1]["artifact_sha256"],
    )
    public_json = json.dumps(public, sort_keys=True)
    assert '"duration_s"' not in public_json
    assert '"seed_sha256"' not in public_json
    assert '"samples_by_tool_s"' not in public_json
    for forbidden in runner.PUBLIC_PLAN_FORBIDDEN_FIELDS:
        assert f'"{forbidden}"' not in public_json
    assert set(public["traces"][0]) == runner.PUBLIC_TRACE_FIELDS
    assert "trace_steps" in sealed
    assert "trace_lineage" in sealed
    trace_id = public["traces"][0]["trace_id"]
    assert sealed["trace_steps"][trace_id][0]["request"]["messages"]
    assert any(
        step["tools_after"] for step in sealed["trace_steps"][trace_id]
    )
    assert sealed["service_clock_artifact_sha256"] == _service_clock()[1]["artifact_sha256"]
    assert diagnostics["runtime_access"] is False


def test_plan_file_permissions_and_transitive_policy_firewall(tmp_path: Path) -> None:
    public = {"schema": runner.PUBLIC_PLAN_SCHEMA, "traces": [{"trace_id": "opaque"}]}
    sealed = {"trace_steps": {"opaque": [{"request": {}, "tools_after": []}]}}
    diagnostics = {"records": {}}
    public_path, sealed_path, diagnostics_path = runner._write_role_plan_files(
        output_dir=tmp_path,
        role="final",
        public=public,
        sealed=sealed,
        diagnostics=diagnostics,
    )
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(sealed_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(diagnostics_path.stat().st_mode) == 0o400
    runner._assert_policy_facing_document_safe(
        {
            "plans": {
                "final": {
                    "public": {"path": "final.public.json", "sha256": "a" * 64},
                    # Binding a private document is allowed; inlining it is not.
                    "sealed": {"path": "final.sealed.json", "sha256": "b" * 64},
                }
            }
        },
        label="fixture bundle",
    )
    with pytest.raises(ValueError, match="future-authority fields"):
        runner._assert_policy_facing_document_safe(
            {"plans": {"final": {"embedded": {"tools_after": []}}}},
            label="fixture bundle",
        )
    with pytest.raises(ValueError, match="state_accepted"):
        runner._assert_policy_facing_document_safe(
            {"candidate": {"state_accepted": True}}, label="fixture bundle"
        )


def test_future_graph_poison_is_absent_from_public_plan(tmp_path: Path) -> None:
    original = _session(tmp_path / "eval.jsonl")
    events = list(original.events)
    search = events[1]
    decision = events[2]
    visit = events[3]
    final = events[4]
    assert isinstance(search, ToolCall)
    assert isinstance(decision, LLMCall)
    assert isinstance(visit, ToolCall)
    assert isinstance(final, LLMCall)
    events[1] = ToolCall(
        search.call_index,
        search.timestamp_s,
        search.tool_name,
        {"query": ["future poison"]},
        search.line_number,
        search.timing_correction,
    )
    events[2] = LLMCall(
        decision.call_index,
        decision.timestamp_s,
        decision.total_time_s,
        decision.inference_time_s,
        ({"role": "user", "content": "future request poison"},),
        decision.response,
        decision.line_number,
    )
    events[3] = ToolCall(
        visit.call_index,
        visit.timestamp_s,
        visit.tool_name,
        {"url": "https://future-poison.invalid/authority"},
        visit.line_number,
        visit.timing_correction,
    )
    events[4] = LLMCall(
        final.call_index,
        final.timestamp_s,
        final.total_time_s,
        final.inference_time_s,
        ({"role": "user", "content": "future suffix poison"},),
        final.response,
        final.line_number,
    )
    poisoned = SessionTrace(original.path, tuple(events))
    kwargs = {
        "role": "final",
        "raw_sha_by_id": {original.session_id: "a" * 64},
        "tokenizer": TinyTokenizer(),
        "max_model_len": 10000,
        "output_cap": 128,
        "arrivals": [{"release_offset_s": 0.0, "arrival_index": 0}],
        "arrival_provenance": {"kind": "test"},
        "service_clock_artifact_sha256": _service_clock()[1]["artifact_sha256"],
    }
    public_original, sealed_original, _ = runner.build_role_plans(
        sessions=[original], **kwargs
    )
    public_poisoned, sealed_poisoned, _ = runner.build_role_plans(
        sessions=[poisoned], **kwargs
    )
    assert public_original == public_poisoned
    assert sealed_original != sealed_poisoned
    trace_id = public_original["traces"][0]["trace_id"]
    original_cursor = CausalTraceCursor(sealed_original["trace_steps"][trace_id])
    poisoned_cursor = CausalTraceCursor(sealed_poisoned["trace_steps"][trace_id])
    # The request visible at the pre-poison boundary is byte-identical; none
    # of the poisoned authority/suffix is reachable until later reveal steps.
    assert original_cursor.current_request() == poisoned_cursor.current_request()


def test_fast_prompt_truncation_matches_legacy_oldest_prefix_rule() -> None:
    tokenizer = TinyTokenizer()
    messages = (
        {"role": "system", "content": "S" * 10},
        {"role": "user", "content": "A" * 20},
        {"role": "assistant", "content": "B" * 20},
        {"role": "user", "content": "C" * 20},
    )
    legacy, legacy_tokens, legacy_truncated = _truncate_messages_to_fit(
        tokenizer=tokenizer,
        messages=messages,
        max_prompt_tokens=40,
    )
    prepared = runner._prepare_request(
        LLMCall(0, 0.0, 0.0, 0.0, messages, "unused future response", 1),
        tokenizer=tokenizer,
        max_model_len=50,
        output_cap=10,
    )
    assert prepared["messages"] == legacy
    assert prepared["prompt_tokens"] == legacy_tokens
    assert prepared["truncated"] is legacy_truncated
    assert prepared["max_tokens"] == 10


def test_llm_workload_digest_binds_content_and_is_cell_independent() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "fixed"},
            {"role": "user", "content": "same request in A/B/E/F"},
        ],
        "prompt_tokens": 37,
        "max_tokens": 128,
    }
    # FCFS and schedx use different transport request IDs, but neither is an
    # input to the workload identity compared across cells.
    digest_a = runner._llm_workload_request_sha256(model="model-revision", request=request)
    digest_f = runner._llm_workload_request_sha256(model="model-revision", request=request)
    assert digest_a == digest_f
    assert len(digest_a) == 64

    changed = json.loads(json.dumps(request))
    changed["messages"][1]["content"] += " changed"
    assert runner._llm_workload_request_sha256(
        model="model-revision", request=changed
    ) != digest_a
    assert runner._llm_workload_request_sha256(
        model="other-model", request=request
    ) != digest_a


def test_policy_freezes_before_final_trace_loader_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _session(tmp_path / "calibration.jsonl")
    tuning = _session(tmp_path / "tuning.jsonl")
    role_sessions = {"calibration": (calibration,), "tuning": (tuning,)}
    fixed = {
        "roles": {
            role: [{"session_id": role, "sha256": role[0] * 64}]
            for role in ("calibration", "tuning", "final")
        },
        "ids": {
            role: {role} for role in ("calibration", "tuning", "final")
        },
        "manifest": {"manifest_sha256": "1" * 64},
        "bundle": {"bundle_sha256": "3" * 64},
        "mapper": object(),
        "mapper_artifact": {"artifact_sha256": "2" * 64},
        "mapper_path": tmp_path / "mapper.json",
    }
    fixed["mapper_path"].write_text("{}\n", encoding="utf-8")
    formal_config = tmp_path / "formal.env"
    scheduler_hook = tmp_path / "hook.py"
    environment_prefix = tmp_path / "env"
    environment_python = environment_prefix / "bin" / "python"
    environment_python.parent.mkdir(parents=True)
    environment_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    environment_python.chmod(0o755)
    hf_home = tmp_path / "hf"
    model_snapshot = (
        hf_home / "models--fixture-model" / "snapshots" / "fixture-revision"
    )
    model_snapshot.mkdir(parents=True)
    (model_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    runtime_home = tmp_path / "home"
    runtime_home.mkdir()
    runtime_tmp = tmp_path / "tmp"
    runtime_tmp.mkdir()
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    topo_file = tmp_path / "topology.xml"
    topo_file.write_text("<system/>\n", encoding="utf-8")
    formal_config.write_text(
        "\n".join(
            (
                'export MODEL_ID="fixture-model"',
                'export MODEL_REVISION="fixture-revision"',
                f'export PASTE_ENV_PREFIX="{environment_prefix}"',
                f'export HF_HOME="{hf_home}"',
                f'export PASTE_RUNTIME_HOME="{runtime_home}"',
                f'export PASTE_RUNTIME_PATH="{environment_prefix}/bin:/usr/bin:/bin"',
                'export PASTE_RUNTIME_LD_LIBRARY_PATH="/tmp"',
                f'export PASTE_RUNTIME_TMPDIR="{runtime_tmp}"',
                'export PASTE_RUNTIME_LANG="C.UTF-8"',
                'export PASTE_RUNTIME_TZ="UTC"',
                'export PYTHONHASHSEED="0"',
                'export PYTHONNOUSERSITE="1"',
                'export PYTHONSAFEPATH="1"',
                'export HF_HUB_OFFLINE="1"',
                'export TRANSFORMERS_OFFLINE="1"',
                'export VLLM_NO_USAGE_STATS="1"',
                'export CUDA_DEVICE_ORDER="PCI_BUS_ID"',
                f'export VLLM_HOOK_DIR="{hook_dir}"',
                f'export NCCL_TOPO_FILE="{topo_file}"',
                'export VLLM_HOST="127.0.0.1"',
                'export VLLM_PORT="8100"',
                'export VLLM_TP_SIZE="4"',
                'export VLLM_DTYPE="bfloat16"',
                'export VLLM_MAX_MODEL_LEN="1000"',
                'export VLLM_GPU_MEMORY_UTILIZATION="0.86"',
                'export VLLM_MAX_NUM_BATCHED_TOKENS="2048"',
                'export VLLM_MAX_NUM_SEQS="48"',
                'export VLLM_CUDA_GRAPH_SIZES="32"',
                    'export VLLM_ENABLE_PREFIX_CACHING="1"',
                    'export VLLM_USE_V1="1"',
                    'export VLLM_ENABLE_V1_MULTIPROCESSING="1"',
                'export PASTE_MAX_ACTIVE_TASKS="1"',
                'export PASTE_VISIT_CAPACITY="2"',
                'export PASTE_SPECULATIVE_CAP="1"',
                'export PASTE_REQUEST_TIMEOUT_S="600"',
                'export PASTE_DEFAULT_PREDICTED_OUTPUT_TOKENS="128"',
                'export PASTE_PUBLIC_OUTPUT_CAP="128"',
                'export PASTE_STRICT_SESSIONS="1"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    scheduler_hook.write_text("# fixture hook\n", encoding="utf-8")
    calibration_service_samples = {
        "visit": (0.01, 0.02, 0.03),
        "search": (0.01, 0.02, 0.03),
        "__global__": (0.01, 0.02, 0.03),
    }
    reused_service_clock = runner._new_service_clock_artifact(
        training_provenance=runner._training_provenance(fixed, (calibration,)),
        samples_by_tool_s=calibration_service_samples,
        seed_sha256="b" * 64,
    )
    reused_service_clock_path = tmp_path / "reused-service-clock.json"
    runner.write_json(reused_service_clock_path, reused_service_clock)
    calls: list[str] = []

    class EvaluationLoaderReached(RuntimeError):
        pass

    def load_role(_fixed, _execution_dir, role):
        calls.append(f"load:{role}")
        if role == "final":
            staging = list(tmp_path.glob(".frozen.prepare-*"))
            assert len(staging) == 1
            required = (
                "duration_predictor.json",
                "tail_predictor.json",
                "service_clock.json",
                "tuning_selection.json",
                "invocation_predictor_provenance.json",
                "policy_freeze.json",
            )
            assert all((staging[0] / name).is_file() for name in required)
            assert all(
                (staging[0] / name).stat().st_mode & 0o777 == 0o400
                for name in required
            )
            assert (staging[0] / "runtime_parameters.json").is_file()
            assert stat.S_IMODE(
                (staging[0] / "runtime_parameters.json").stat().st_mode
            ) == 0o444
            freeze = json.loads(
                (staging[0] / "policy_freeze.json").read_text(encoding="utf-8")
            )
            invocation = json.loads(
                (staging[0] / "invocation_predictor_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            runtime_parameters = json.loads(
                (staging[0] / "runtime_parameters.json").read_text(
                    encoding="utf-8"
                )
            )
            frozen_service_clock = json.loads(
                (staging[0] / "service_clock.json").read_text(encoding="utf-8")
            )
            assert frozen_service_clock == reused_service_clock
            assert (
                frozen_service_clock["artifact_sha256"]
                == reused_service_clock["artifact_sha256"]
            )
            assert runtime_parameters["schema"] == runner.RUNTIME_PARAMETERS_SCHEMA
            assert set(runtime_parameters["parameters"]) == runner.RUNTIME_PARAMETER_KEYS
            assert invocation["input_features"] == [
                "last_completed_tool_name",
                "current_visible_search_result_urls",
                "current_visible_search_result_ranks",
                "current_visible_search_result_ordinals",
                "frozen_top_k",
            ]
            assert set(freeze["frozen_runtime_files"]) == {
                "runner",
                "strict_runtime",
                "tool_pool",
                "mapper_code",
                "matrix_wrapper",
                "smoke_script",
                "start_vllm",
                "stop_vllm",
                "sitecustomize",
                "formal_config",
                "scheduler_hook",
            }
            raise EvaluationLoaderReached
        return role_sessions[role]

    def select(**_kwargs):
        calls.append("select:tuning")
        evidence = {
            "selection_role": "tuning",
            "selected_top_k": 2,
        }
        evidence["selection_sha256"] = runner.canonical_sha256(evidence)
        return 2, evidence

    monkeypatch.setattr(runner, "_load_fixed_split", lambda _path: fixed)
    monkeypatch.setattr(runner, "_load_role_sessions", load_role)
    monkeypatch.setattr(runner, "select_tuning_top_k", select)
    monkeypatch.setattr(runner, "_load_tokenizer", lambda _source: TinyTokenizer())
    monkeypatch.setattr(
        runner,
        "_calibration_service_samples",
        lambda _sessions: calibration_service_samples,
    )
    monkeypatch.setattr(
        runner,
        "_arrival_rows",
        lambda _path, _sessions: ([{"release_offset_s": 0.0}], {"kind": "test"}),
    )
    args = SimpleNamespace(
        output_dir=tmp_path / "frozen",
        formal_config=formal_config,
        scheduler_hook=scheduler_hook,
        fixed_bundle=tmp_path / "fixed.json",
        execution_traces=tmp_path,
        duration_ewma_alpha=0.35,
        reuse_service_clock=reused_service_clock_path,
        max_active_tasks=1,
        visit_capacity=2,
        speculative_cap=1,
        request_timeout_s=600.0,
        default_predicted_output_tokens=128.0,
        max_top_k=5,
        min_prediction_precision=0.4,
        tokenizer="fixture-model",
        arrivals=None,
        sessions=1,
        claim_scope="confirmatory",
        model="fixture-model",
        model_revision="fixture-revision",
        max_model_len=1000,
        output_cap=128,
    )
    with pytest.raises(EvaluationLoaderReached):
        runner.prepare_bundle(args)
    assert calls == ["load:calibration", "load:tuning", "select:tuning", "load:final"]
    assert not args.output_dir.exists()


def test_matrix_contract_rejects_capacity_and_hook_dir_overrides(tmp_path: Path) -> None:
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    config = tmp_path / "formal.env"
    scheduler_hook = hook_dir / "sched_policy_patch.py"
    sitecustomize = hook_dir / "sitecustomize.py"
    start = tmp_path / "start_vllm.sh"
    stop = tmp_path / "stop_vllm.sh"
    for path in (config, scheduler_hook, sitecustomize, start, stop):
        path.write_text("fixture\n", encoding="utf-8")
    loaded = {
        "bundle": {
            "runtime_capacities": {
                "max_active_tasks": 80,
                "visit_capacity": 128,
                "speculative_cap": 32,
            }
        },
        "frozen_runtime_paths": {
            "formal_config": config,
            "scheduler_hook": scheduler_hook,
            "sitecustomize": sitecustomize,
            "start_vllm": start,
            "stop_vllm": stop,
        },
    }
    kwargs = {
        "loaded": loaded,
        "config_path": config,
        "scheduler_hook_path": scheduler_hook,
        "sitecustomize_path": sitecustomize,
        "start_vllm_path": start,
        "stop_vllm_path": stop,
        "hook_dir": hook_dir,
        "max_active_tasks": 80,
        "visit_capacity": 128,
        "speculative_cap": 32,
    }
    runner.validate_matrix_execution_contract(**kwargs)
    with pytest.raises(ValueError, match="runtime capacities"):
        runner.validate_matrix_execution_contract(
            **{**kwargs, "speculative_cap": 31}
        )
    wrong_hook_dir = tmp_path / "unfrozen-hooks"
    wrong_hook_dir.mkdir()
    with pytest.raises(ValueError, match="VLLM_HOOK_DIR"):
        runner.validate_matrix_execution_contract(
            **{**kwargs, "hook_dir": wrong_hook_dir}
        )


def test_frozen_matrix_records_exact_vllm_version_evidence() -> None:
    start_text = runner.START_VLLM_PATH.read_text(encoding="utf-8")
    matrix_text = runner.MATRIX_WRAPPER_PATH.read_text(encoding="utf-8")
    assert 'PINNED_VLLM_VERSION="0.10.1"' in start_text
    assert 'version("vllm")' in start_text
    assert '!= "${PINNED_VLLM_VERSION}"' in start_text
    assert 'vllm_distribution_version=%s' in matrix_text
    assert 'vllm_version_requirement=exactly-0.10.1' in matrix_text
    assert 'run_in_frozen_environment \\\n      "CUDA_VISIBLE_DEVICES=${gpu_group}"' in matrix_text
    assert '"${START_SCRIPT}"' in matrix_text
    assert '"${STOP_SCRIPT}"' in matrix_text
    assert '"${PYTHON_BIN}" -I "${SCRIPT_DIR}/run_strict_trace_abef.py" run-cell' in matrix_text
    assert 'MODEL_SNAPSHOT and PYTHONPATH' not in matrix_text
    assert "write_scheduler_runtime_evidence" in matrix_text
    assert '"scheduler_api": "v1.Scheduler.schedule"' in matrix_text
    assert "patched scheduler did not emit runtime evidence during smoke" in matrix_text
    assert "FCFS cell unexpectedly executed the scheduler hook" in matrix_text


def test_fcfs_runtime_evidence_rejects_any_scheduler_hook_marker(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "vllm_8100.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    marker_path = pid_path.with_suffix(".scheduler_runtime.json")
    hook_path = tmp_path / "sched_policy_patch.py"
    hook_path.write_text("# frozen hook\n", encoding="utf-8")
    evidence_path = tmp_path / "scheduler-after-smoke.json"
    evidence = {
        "schema": runner.SCHEDULER_RUNTIME_EVIDENCE_SCHEMA,
        "cell": "A",
        "phase": "after_standardized_smoke",
        "server_pid": os.getpid(),
        "expected_policy": "fcfs",
        "hook_runtime_use_expected": False,
        "patched_scheduler_invocation_verified": False,
        "no_scheduler_hook_runtime_use_verified": True,
        "scheduler_hook_path": str(hook_path.resolve()),
        "scheduler_hook_sha256": runner.file_sha256(hook_path),
        "runtime_marker_path": str(marker_path.resolve()),
        "runtime_marker_sha256": None,
        "scheduler_calling_pid": None,
        "scheduler_calling_process_relation": None,
        "runtime_marker": None,
    }
    runner.write_json(evidence_path, evidence)
    args = SimpleNamespace(
        cell="A",
        server_pid_file=pid_path,
        scheduler_hook_file=hook_path,
        scheduler_hook_file_sha256=runner.file_sha256(hook_path),
        scheduler_runtime_evidence_file=evidence_path,
        scheduler_runtime_marker_file=marker_path,
    )
    assert runner.validate_scheduler_runtime_evidence(
        args,
        cell=runner.CELL_SPECS["A"],
        server_pid=os.getpid(),
    ) == evidence

    marker_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FCFS cell unexpectedly"):
        runner.validate_scheduler_runtime_evidence(
            args,
            cell=runner.CELL_SPECS["A"],
            server_pid=os.getpid(),
        )


def test_formal_environment_rejects_unregistered_snapshot_override() -> None:
    production_exports = runner._formal_config_exports(runner.DEFAULT_FORMAL_CONFIG)
    assert set(production_exports) == runner.FORMAL_ENVIRONMENT_KEYS
    with pytest.raises(ValueError, match="unregistered environment"):
        runner._validate_formal_environment_contract(
            {"MODEL_SNAPSHOT": "/tmp/poisoned-snapshot"}
        )


def test_formal_config_parser_rejects_shell_expansion_without_executing_it(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-exist"
    config = tmp_path / "poisoned.env"
    config.write_text(
        f'export MODEL_ID="$(touch {marker})"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not literal"):
        runner._formal_config_exports(config)
    assert not marker.exists()


def test_matrix_wrapper_rejects_bash_env_before_accepting_a_run(
    tmp_path: Path,
) -> None:
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text("# even a harmless BASH_ENV is forbidden\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["BASH_ENV"] = str(bash_env)
    completed = subprocess.run(
        [str(runner.MATRIX_WRAPPER_PATH), "final"],
        cwd=runner.REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 1
    assert "BASH_ENV is forbidden" in completed.stdout


def test_matrix_clean_reexec_scrubs_unregistered_runtime_poison(
    tmp_path: Path,
) -> None:
    missing_bundle = tmp_path / "missing-bundle.json"

    def invoke(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("BASH_ENV", None)
        environment.update(
            {
                "PASTE_STRICT_BUNDLE": str(missing_bundle),
                "PASTE_VALIDATE_ONLY": "1",
                **extra,
            }
        )
        return subprocess.run(
            [str(runner.MATRIX_WRAPPER_PATH), "final"],
            cwd=runner.REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )

    clean = invoke({})
    poisoned = invoke(
        {
            "MODEL_SNAPSHOT": "/tmp/other-weights",
            "PYTHONPATH": "/tmp/other-hook",
            "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync",
            "VLLM_ATTENTION_BACKEND": "POISON",
            "PASTE_GPU_GROUPS": "99",
            "CUDA_VISIBLE_DEVICES": "99",
        }
    )
    assert clean.returncode == poisoned.returncode == 1
    assert clean.stdout == poisoned.stdout
    assert "required frozen input is missing" in clean.stdout


def test_matrix_python_preflight_ignores_malicious_caller_working_directory(
    tmp_path: Path,
) -> None:
    malicious_cwd = tmp_path / "caller-cwd"
    malicious_cwd.mkdir()
    marker = tmp_path / "cwd-imported"
    poison = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    (malicious_cwd / "sitecustomize.py").write_text(poison, encoding="utf-8")
    (malicious_cwd / "aiohttp.py").write_text(poison, encoding="utf-8")
    poison_vllm = malicious_cwd / "vllm"
    poison_vllm.mkdir()
    (poison_vllm / "__init__.py").write_text(poison, encoding="utf-8")

    environment_prefix = Path(sys.executable).resolve().parents[1]
    hf_home = tmp_path / "hf"
    revision = "fixture-revision"
    snapshot = (
        hf_home
        / "models--fixture--model"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    runtime_home = tmp_path / "home"
    runtime_home.mkdir()
    runtime_tmp = tmp_path / "tmp"
    runtime_tmp.mkdir()
    topology = tmp_path / "topology.xml"
    topology.write_text("<system/>\n", encoding="utf-8")

    exports = runner._formal_config_exports(runner.DEFAULT_FORMAL_CONFIG)
    exports.update(
        {
            "PASTE_ENV_PREFIX": str(environment_prefix),
            "HF_HOME": str(hf_home),
            "MODEL_ID": "fixture/model",
            "MODEL_REVISION": revision,
            "PASTE_RUNTIME_HOME": str(runtime_home),
            "PASTE_RUNTIME_PATH": (
                f"{environment_prefix}/bin:/usr/local/bin:/usr/bin:/bin"
            ),
            "PASTE_RUNTIME_LD_LIBRARY_PATH": "/tmp",
            "PASTE_RUNTIME_TMPDIR": str(runtime_tmp),
            "VLLM_HOOK_DIR": str(runner.ROOT_SCRIPTS / "pythonhooks"),
            "NCCL_TOPO_FILE": str(topology),
            "PASTE_PROTECTED_PID": "1",
        }
    )
    config = tmp_path / "formal.env"
    config.write_text(
        "".join(
            f"export {name}={json.dumps(value)}\n"
            for name, value in exports.items()
        ),
        encoding="utf-8",
    )
    dummy_bundle = tmp_path / "dummy-bundle.json"
    dummy_bundle.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("BASH_ENV", None)
    environment.update(
        {
            "PASTE_STRICT_CONFIG": str(config),
            "PASTE_STRICT_BUNDLE": str(dummy_bundle),
            "PASTE_VALIDATE_ONLY": "1",
        }
    )
    completed = subprocess.run(
        [str(runner.MATRIX_WRAPPER_PATH), "final"],
        cwd=malicious_cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )
    assert completed.returncode != 0
    assert not marker.exists(), completed.stdout


def test_visit_url_normalization_is_shared_by_cache_and_service_clock() -> None:
    assert normalize_url(" HTTPS://Example.TEST:443#fragment ") == "https://example.test/"
    clock, _ = _service_clock(
        {"visit": [0.01, 0.02, 0.03], "__global__": [0.01, 0.02, 0.03]}
    )
    assert clock.service_s(
        tool_name="visit", tool_arguments={"url": "HTTPS://Example.TEST:443#x"}
    ) == clock.service_s(
        tool_name="visit", tool_arguments={"url": "https://example.test/"}
    )


def test_promoted_worker_accounting_splits_at_authority_claim() -> None:
    async def scenario() -> None:
        transitions: list[dict] = []

        def callback(raw: dict) -> None:
            transitions.append(
                {
                    **raw,
                    "trace_id": "trace",
                    "request_index": 0,
                    "candidate_invocation_digest": "c" * 64,
                }
            )

        pool = AsyncPreemptibleVisitPool(
            capacity=1, speculative_cap=1, job_event_callback=callback
        )
        assert await pool.speculate_batch(
            [("session", "https://example.test/", 0.04, 1.0, "prediction")]
        ) == (True,)
        await asyncio.sleep(0.01)
        result = await pool.authoritative(
            session_id="session",
            url="https://example.test/",
            duration_s=0.04,
        )
        assert result.source == "promoted_inflight"
        rows = runner.aggregate_speculation_execution_events(transitions)
        assert len(rows) == 1
        row = rows[0]
        assert row["claimed_by_authority"] is True
        assert 0.0 < row["speculative_resource_s"] < row["total_worker_service_s"]
        assert 0.0 < row["demand_resource_s"] < row["total_worker_service_s"]
        assert row["total_worker_service_s"] == pytest.approx(
            row["speculative_resource_s"] + row["demand_resource_s"], abs=1e-9
        )
        snapshot = pool.snapshot()
        assert snapshot["speculative_resource_s"] == pytest.approx(
            row["speculative_resource_s"], abs=1e-9
        )
        assert snapshot["promoted_demand_resource_s"] == pytest.approx(
            row["demand_resource_s"], abs=1e-9
        )
        assert snapshot["total_worker_occupancy_s"] == pytest.approx(
            row["total_worker_service_s"], abs=1e-9
        )
        await pool.close_session("session")
        await pool.close()

    asyncio.run(scenario())


def test_promoted_worker_first_claim_survives_later_completed_reuse() -> None:
    async def scenario() -> None:
        transitions: list[dict] = []

        def callback(raw: dict) -> None:
            transitions.append(
                {
                    **raw,
                    "trace_id": "trace",
                    "request_index": 0,
                    "candidate_invocation_digest": "c" * 64,
                }
            )

        pool = AsyncPreemptibleVisitPool(
            capacity=1, speculative_cap=1, job_event_callback=callback
        )
        url = "https://example.test/"
        assert await pool.speculate_batch(
            [("session", url, 0.04, 1.0, "prediction")]
        ) == (True,)
        await asyncio.sleep(0.01)
        promoted = await pool.authoritative(
            session_id="session", url=url, duration_s=0.04
        )
        assert promoted.source == "promoted_inflight"

        first_claim = next(
            row["at_monotonic_s"]
            for row in transitions
            if row["event"] == "authority_claimed_inflight"
        )
        reused = await pool.authoritative(
            session_id="session", url=url, duration_s=0.04
        )
        assert reused.source == "reused"

        rows = runner.aggregate_speculation_execution_events(transitions)
        assert len(rows) == 1
        row = rows[0]
        assert [
            item["event"] for item in row["state_transitions"]
        ].count("authority_claimed_completed") == 1
        assert row["authority_claimed_at_monotonic_s"] == first_claim
        started = row["physical_started_at_monotonic_s"]
        terminal = row["terminal_at_monotonic_s"]
        assert row["speculative_resource_s"] == pytest.approx(
            first_claim - started, abs=1e-9
        )
        assert row["demand_resource_s"] == pytest.approx(
            terminal - first_claim, abs=1e-9
        )
        assert all(
            transition["authority_claimed_at_monotonic_s"] == first_claim
            for transition in transitions
            if transition["authority_claimed_at_monotonic_s"] is not None
        )
        await pool.close_session("session")
        await pool.close()

    asyncio.run(scenario())


def test_concurrent_authority_joins_promoted_job_without_double_charging() -> None:
    async def scenario() -> None:
        transitions: list[dict] = []

        def callback(raw: dict) -> None:
            transitions.append(
                {
                    **raw,
                    "trace_id": "trace",
                    "request_index": 0,
                    "candidate_invocation_digest": "d" * 64,
                }
            )

        pool = AsyncPreemptibleVisitPool(
            capacity=1, speculative_cap=1, job_event_callback=callback
        )
        url = "https://example.test/concurrent"
        assert await pool.speculate_batch(
            [("session", url, 0.05, 1.0, "prediction")]
        ) == (True,)
        await asyncio.sleep(0.01)
        first = asyncio.create_task(
            pool.authoritative(session_id="session", url=url, duration_s=0.05)
        )
        second = asyncio.create_task(
            pool.authoritative(session_id="session", url=url, duration_s=0.05)
        )
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result.source == "promoted_inflight"
        assert second_result.source == "promoted_inflight"

        assert sum(
            row["event"] == "authority_claimed_inflight" for row in transitions
        ) == 1
        assert sum(
            row["event"] == "authority_joined_inflight" for row in transitions
        ) == 1
        claim = next(
            row["authority_claimed_at_monotonic_s"]
            for row in transitions
            if row["event"] == "authority_claimed_inflight"
        )
        join = next(
            row for row in transitions if row["event"] == "authority_joined_inflight"
        )
        assert join["authority_claimed_at_monotonic_s"] == claim

        rows = runner.aggregate_speculation_execution_events(transitions)
        assert len(rows) == 1
        row = rows[0]
        started = row["physical_started_at_monotonic_s"]
        terminal = row["terminal_at_monotonic_s"]
        assert row["authority_claimed_at_monotonic_s"] == claim
        assert row["speculative_resource_s"] == pytest.approx(
            claim - started, abs=1e-9
        )
        assert row["demand_resource_s"] == pytest.approx(
            terminal - claim, abs=1e-9
        )
        assert row["total_worker_service_s"] == pytest.approx(
            terminal - started, abs=1e-9
        )
        snapshot = pool.snapshot()
        assert snapshot["metrics"]["inflight_cache_hits"] == 2
        assert snapshot["metrics"]["promoted_running_speculations"] == 1
        assert snapshot["speculative_resource_s"] == pytest.approx(
            row["speculative_resource_s"], abs=1e-9
        )
        assert snapshot["promoted_demand_resource_s"] == pytest.approx(
            row["demand_resource_s"], abs=1e-9
        )
        await pool.close_session("session")
        await pool.close()

    asyncio.run(scenario())


def test_speculation_ledger_rejects_candidate_service_reassignment() -> None:
    transitions = []
    for job_id, assigned in ((1, 0.1), (2, 0.2)):
        common = {
            "job_id": job_id,
            "prediction_id": f"prediction-{job_id}",
            "trace_id": f"trace-{job_id}",
            "request_index": 0,
            "candidate_invocation_digest": "c" * 64,
            "assigned_service_s": assigned,
            "authority_claimed_at_monotonic_s": None,
            "speculative_resource_s": 0.0,
            "demand_resource_s": 0.0,
            "total_worker_service_s": 0.0,
            "service_s": 0.0,
            "claimed_by_authority": False,
        }
        transitions.extend(
            [
                {
                    **common,
                    "event": "admitted",
                    "state": "queued",
                    "at_monotonic_s": float(job_id),
                },
                {
                    **common,
                    "event": "cancelled_window_expired",
                    "state": "cancelled_window_expired",
                    "at_monotonic_s": float(job_id) + 0.01,
                },
            ]
        )
    with pytest.raises(RuntimeError, match="inconsistent physical service"):
        runner.aggregate_speculation_execution_events(transitions)


def test_prediction_precision_excludes_queued_never_started_candidate() -> None:
    hit = "a" * 64
    miss = "b" * 64
    outcome = {
        "prediction_id": "trace:request:0",
        "trace_id": "trace",
        "request_index": 0,
        "authoritative_invocation_digests": ["c" * 64],
        "authoritative_candidate_invocation_digests": [hit],
        "candidates": [
            {
                "candidate_invocation_digest": hit,
                "admitted": True,
                "broker_accepted": True,
                "matched_authority": True,
            },
            {
                "candidate_invocation_digest": miss,
                "admitted": True,
                "broker_accepted": True,
                "matched_authority": False,
            },
        ],
        "emitted_candidate_count": 2,
        "admitted_candidate_count": 2,
        "broker_accepted_candidate_count": 2,
        "matched_emitted_candidate_count": 1,
        "matched_admitted_candidate_count": 1,
        "matched_broker_accepted_candidate_count": 1,
        "decision_hit": True,
    }
    common = {
        "prediction_id": "trace:request:0",
        "trace_id": "trace",
        "request_index": 0,
    }
    ledger = [
        {
            **common,
            "candidate_invocation_digest": hit,
            "physical_started_at_monotonic_s": None,
            "terminal_state": "cancelled",
        },
        {
            **common,
            "candidate_invocation_digest": miss,
            "physical_started_at_monotonic_s": 2.0,
            "terminal_state": "completed",
        },
    ]
    metrics = runner.prediction_metrics_from_raw_evidence(
        prediction_outcomes=[outcome],
        tool_events=[
            {
                "trace_id": "trace",
                "request_index": 0,
                "authority_candidate_invocation_digests": [hit],
            }
        ],
        speculation_execution_events=ledger,
    )
    assert metrics["emitted_candidate_precision"] == 0.5
    assert metrics["broker_accepted_candidate_precision"] == 0.5
    assert metrics["physical_started_candidate_precision"] == 0.0
    assert metrics["admitted_candidate_precision"] == 0.0
    assert metrics["broker_accepted_candidates"] == 2
    assert metrics["physical_started_candidates"] == 1
    assert metrics["queued_never_started_candidates"] == 1
    assert metrics["cancelled_candidates"] == 1
    assert metrics["physical_started_cancelled_candidates"] == 0


def test_speculation_causal_timing_covers_started_and_queued_jobs() -> None:
    decision = {
        "prediction_id": "trace:request:0",
        "trace_id": "trace",
        "request_index": 0,
        "decided_at_monotonic_s": 1.0,
    }
    request = {
        "trace_id": "trace",
        "request_index": 0,
        "llm_completed_at_monotonic_s": 3.0,
    }
    common = {
        "prediction_id": "trace:request:0",
        "trace_id": "trace",
        "request_index": 0,
    }
    queued = {
        **common,
        "job_id": 1,
        "admitted_at_monotonic_s": 1.1,
        "physical_started_at_monotonic_s": None,
    }
    started = {
        **common,
        "job_id": 2,
        "admitted_at_monotonic_s": 1.2,
        "physical_started_at_monotonic_s": 1.3,
    }
    evidence = runner.validate_speculation_causal_timing(
        prediction_decisions=[decision],
        request_events=[request],
        speculation_execution_events=[queued, started],
    )
    assert evidence["broker_accepted_jobs"] == 2
    assert evidence["physical_started_jobs"] == 1
    assert evidence["minimum_decision_to_admission_s"] == pytest.approx(0.1)
    assert evidence["minimum_admission_to_completion_s"] == pytest.approx(1.8)
    assert evidence["minimum_admission_to_physical_start_s"] == pytest.approx(0.1)
    assert evidence["minimum_physical_start_to_completion_s"] == pytest.approx(1.7)

    poisoned_queued = {**queued, "admitted_at_monotonic_s": 3.0}
    with pytest.raises(RuntimeError, match="broker acceptance fell outside"):
        runner.validate_speculation_causal_timing(
            prediction_decisions=[decision],
            request_events=[request],
            speculation_execution_events=[poisoned_queued],
        )
    poisoned_started = {**started, "physical_started_at_monotonic_s": 1.1}
    with pytest.raises(RuntimeError, match="physical speculative start fell outside"):
        runner.validate_speculation_causal_timing(
            prediction_decisions=[decision],
            request_events=[request],
            speculation_execution_events=[poisoned_started],
        )


def test_fake_f_cell_result_passes_formal_auditor_and_digest_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        policy = _policy(tmp_path)
        link_url = "https://Example.TEST:443/two#fragment"
        raw_url = f" {link_url} "
        session = _session(tmp_path / "eval-integration.jsonl")
        events = list(session.events)
        decision_llm = events[2]
        assert isinstance(decision_llm, LLMCall)
        events[2] = LLMCall(
            decision_llm.call_index,
            decision_llm.timestamp_s,
            decision_llm.total_time_s,
            decision_llm.inference_time_s,
            (
                {
                    "role": "user",
                    "content": (
                        "<tool_response>\n"
                        "1. [one](https://example.test/one)\n"
                        f"2. [two]({link_url})\n"
                        "</tool_response>"
                    ),
                },
            ),
            decision_llm.response,
            decision_llm.line_number,
        )
        visit = events[3]
        assert isinstance(visit, ToolCall)
        events[3] = ToolCall(
            visit.call_index,
            visit.timestamp_s,
            visit.tool_name,
            {"url": raw_url},
            visit.line_number,
            visit.timing_correction,
        )
        session = SessionTrace(session.path, tuple(events))
        service_clock, service_artifact = _service_clock(
            {
                "visit": [0.02, 0.02, 0.02],
                "search": [0.001, 0.001, 0.001],
                "__global__": [0.001, 0.01, 0.02],
            }
        )
        public, sealed, _ = runner.build_role_plans(
            role="final",
            sessions=[session],
            raw_sha_by_id={session.session_id: "a" * 64},
            tokenizer=TinyTokenizer(),
            max_model_len=10000,
            output_cap=8,
            arrivals=[{"release_offset_s": 0.0, "arrival_index": 0}],
            arrival_provenance={"kind": "test"},
            service_clock_artifact_sha256=service_artifact["artifact_sha256"],
        )
        runtime_parameters = runner._validate_runtime_parameters(
            runner.signed_payload(
                {
                    "schema": runner.RUNTIME_PARAMETERS_SCHEMA,
                    "parameters": {
                        "model_id": "fixture-model",
                        "model_revision": "fixture-revision",
                        "server_host": "127.0.0.1",
                        "server_port": 8100,
                        "tensor_parallel_size": 1,
                        "dtype": "bfloat16",
                        "max_model_len": 10000,
                        "gpu_memory_utilization": 0.5,
                        "max_num_batched_tokens": 128,
                        "max_num_seqs": 1,
                        "cuda_graph_sizes": [1],
                        "prefix_caching": True,
                        "vllm_v1": True,
                        "max_active_tasks": 1,
                        "tool_capacity": 1,
                        "configured_speculation_capacity": 1,
                        "request_timeout_s": 1.0,
                        "public_output_cap": 8,
                        "workload_instances": 1,
                        "arrival_schedule_sha256": "f" * 64,
                    },
                },
                "runtime_parameters_sha256",
            )
        )
        invocation_path = tmp_path / "invocation.json"
        duration_path = tmp_path / "duration.json"
        service_path = tmp_path / "service.json"
        runtime_path = tmp_path / "runtime.json"
        config_path = tmp_path / "frozen.env"
        hook_path = tmp_path / "hook.py"
        bundle_path = tmp_path / "bundle.json"
        for path, payload in (
            (invocation_path, {"fixture": "invocation"}),
            (duration_path, {"fixture": "duration"}),
            (service_path, service_artifact),
            (runtime_path, runtime_parameters),
            (config_path, {"fixture": "config"}),
            (hook_path, {"fixture": "hook"}),
            (bundle_path, {"fixture": "bundle"}),
        ):
            runner.write_json(path, payload)
        model_snapshot = tmp_path / "model-snapshot"
        model_snapshot.mkdir()
        (model_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        model_inventory = runner._model_snapshot_inventory(model_snapshot)
        bundle = {
            "bundle_sha256": "b" * 64,
            "claim_scope": "retrospective",
            "model": "fixture-model",
            "model_revision": "fixture-revision",
            "model_snapshot_contract": {
                "path": str(model_snapshot),
                "inventory_sha256": model_inventory["inventory_sha256"],
            },
            "mapper_artifact_sha256": policy.mapper_artifact_sha256,
            "duration_predictor_artifact_sha256": policy.duration_predictor.artifact_sha256,
            "tail_predictor_artifact_sha256": policy.tail_predictor.artifact_sha256,
            "service_clock_artifact_sha256": service_artifact["artifact_sha256"],
            "runtime_parameters": runtime_parameters,
            "runtime_capacities": {
                "max_active_tasks": 1,
                "visit_capacity": 1,
                "speculative_cap": 1,
            },
            "selected_top_k": 1,
        }
        loaded = {
            "bundle": bundle,
            "mapper": policy.mapper,
            "duration": policy.duration_predictor,
            "tail": policy.tail_predictor,
            "service_clock": service_clock,
            "public": public,
            "sealed": sealed,
                "artifact_paths": {
                "invocation_predictor": invocation_path,
                "duration_predictor": duration_path,
                "tail_predictor": duration_path,
                "service_clock": service_path,
                "runtime_parameters": runtime_path,
                "public_plan": tmp_path / "unused-public.json",
                    "sealed_plan": tmp_path / "unused-sealed.json",
                },
                "frozen_runtime_paths": {
                    "runner": runner.SCRIPT,
                    "strict_runtime": runner.STRICT_RUNTIME_PATH,
                    "tool_pool": runner.TOOL_POOL_PATH,
                    "mapper_code": runner.MAPPER_CODE_PATH,
                    "matrix_wrapper": runner.MATRIX_WRAPPER_PATH,
                    "smoke_script": runner.SMOKE_SCRIPT_PATH,
                    "start_vllm": runner.START_VLLM_PATH,
                    "stop_vllm": runner.STOP_VLLM_PATH,
                    "sitecustomize": runner.SITECUSTOMIZE_PATH,
                    "formal_config": config_path,
                    "scheduler_hook": hook_path,
                },
            }
        monkeypatch.setattr(runner, "load_strict_bundle", lambda _path, _role: loaded)

        async def fake_post(_session, *, request, **_kwargs):
            await asyncio.sleep(0.005)
            return (
                200,
                {
                    "prompt_tokens": int(request["prompt_tokens"]),
                    "completion_tokens": int(request["max_tokens"]),
                },
                "fixed response",
            )

        monkeypatch.setattr(runner, "_post_llm", fake_post)
        policy_path = tmp_path / "server.policy"
        policy_path.write_text("online_joint_pacer_v2\n", encoding="utf-8")
        pid_path = tmp_path / "server.pid"
        pid_path.write_text(f"{runner.os.getpid()}\n", encoding="utf-8")
        smoke_path = tmp_path / "smoke.txt"
        smoke_path.write_text("standardized smoke passed\n", encoding="utf-8")
        runtime_environment_path = tmp_path / "runtime-environment.txt"
        runtime_environment_path.write_text(
            "schema=paste.paper.frozen_cell_environment.v1\n",
            encoding="utf-8",
        )
        scheduler_runtime_marker_path = pid_path.with_suffix(
            ".scheduler_runtime.json"
        )
        scheduler_runtime_marker = {
            "schema": "paste.vllm.scheduler_runtime_use.v1",
            "pid": runner.os.getpid(),
            "ppid": runner.os.getppid(),
            "process_start_ticks": 1,
            "policy": "online_joint_pacer_v2",
            "scheduler_api": "v1.Scheduler.schedule",
            "scheduler_hook_path": str(hook_path.resolve()),
            "scheduler_hook_sha256": runner.file_sha256(hook_path),
            "safe_working_directory": str(tmp_path.resolve()),
            "python_safe_path_enforced": True,
            "cwd_import_filter_enforced": True,
            "working_directory": str(tmp_path.resolve()),
            "working_directory_on_sys_path": True,
            "working_directory_importable": False,
            "python_version": "3.10.21",
            "recorded_at_unix_ns": 1,
            "recorded_at_monotonic_ns": 1,
        }
        runner.write_json(scheduler_runtime_marker_path, scheduler_runtime_marker)
        scheduler_runtime_evidence_path = tmp_path / "scheduler-runtime.json"
        runner.write_json(
            scheduler_runtime_evidence_path,
            {
                "schema": runner.SCHEDULER_RUNTIME_EVIDENCE_SCHEMA,
                "cell": "F",
                "phase": "after_standardized_smoke",
                "server_pid": runner.os.getpid(),
                "expected_policy": "online_joint_pacer_v2",
                "hook_runtime_use_expected": True,
                "patched_scheduler_invocation_verified": True,
                "no_scheduler_hook_runtime_use_verified": False,
                "scheduler_hook_path": str(hook_path.resolve()),
                "scheduler_hook_sha256": runner.file_sha256(hook_path),
                "runtime_marker_path": str(scheduler_runtime_marker_path.resolve()),
                "runtime_marker_sha256": runner.file_sha256(
                    scheduler_runtime_marker_path
                ),
                "scheduler_calling_pid": runner.os.getpid(),
                "scheduler_calling_process_relation": "server_descendant",
                "runtime_marker": scheduler_runtime_marker,
            },
        )
        args = SimpleNamespace(
            cell="F",
            role="final",
            bundle=bundle_path,
            output_dir=tmp_path / "result",
            server_url="http://127.0.0.1:8100",
            server_policy_file=policy_path,
            server_pid_file=pid_path,
            server_instance_id=f"fixture-server-pid-{runner.os.getpid()}",
            block_id="cycle-01-block-01",
            order_position=1,
            gpu_ids="0",
            config_file=config_path,
            config_file_sha256=runner.file_sha256(config_path),
            scheduler_hook_file=hook_path,
            scheduler_hook_file_sha256=runner.file_sha256(hook_path),
            smoke_evidence_file=smoke_path,
            smoke_evidence_sha256=runner.file_sha256(smoke_path),
            runtime_environment_evidence_file=runtime_environment_path,
            runtime_environment_evidence_sha256=runner.file_sha256(
                runtime_environment_path
            ),
            scheduler_runtime_evidence_file=scheduler_runtime_evidence_path,
            scheduler_runtime_evidence_sha256=runner.file_sha256(
                scheduler_runtime_evidence_path
            ),
            scheduler_runtime_marker_file=scheduler_runtime_marker_path,
            model="fixture-model",
            max_active_tasks=1,
            visit_capacity=1,
            speculative_cap=1,
            default_predicted_output_tokens=8.0,
            request_timeout_s=1.0,
            claim_scope=None,
        )
        result = await runner.run_cell(args)
        assert strict_audit.audit_result_payload(result) == []
        assert result["paper_protocol"]["physical_speculative_starts"] == 1
        assert result["prediction_decisions"][0]["candidates"][0][
            "candidate_invocation_digest"
        ] == result["speculation_execution_events"][0][
            "candidate_invocation_digest"
        ]
        assert result["speculation_execution_events"][0]["assigned_service_s"] == 0.02
        assert result["prediction_metrics"]["emitted_candidate_precision"] == 1.0
        assert result["prediction_metrics"]["broker_accepted_candidate_precision"] == 1.0
        assert result["prediction_metrics"]["physical_started_candidate_precision"] == 1.0
        assert result["prediction_metrics"]["admitted_candidate_precision"] == 1.0
        assert result["prediction_metrics"]["queued_never_started_candidates"] == 0
        causal_timing = result["speculation_causal_timing"]
        assert causal_timing["broker_accepted_jobs"] == 1
        assert causal_timing["physical_started_jobs"] == 1
        assert causal_timing["minimum_decision_to_admission_s"] >= 0.0
        assert causal_timing["minimum_admission_to_completion_s"] > 0.0
        assert causal_timing["minimum_admission_to_physical_start_s"] >= 0.0
        assert causal_timing["minimum_physical_start_to_completion_s"] > 0.0
        assert result["prediction_decisions"][0]["candidates"][0][
            "broker_accepted"
        ] is True
        assert result["prediction_outcomes"][0]["decision_hit"] is True
        assert result["prediction_outcomes"][0][
            "authoritative_candidate_invocation_digests"
        ] == [
            result["prediction_outcomes"][0]["candidates"][0][
                "candidate_invocation_digest"
            ]
        ]
        assert result["prediction_outcomes"][0][
            "authoritative_candidate_invocation_digests"
        ] == sorted(
            {
                digest
                for event in result["tool_events"]
                for digest in event["authority_candidate_invocation_digests"]
            }
        )
        assert all(
            event["authority_invocation_digest"] for event in result["tool_events"]
        )
        assert all(event["assigned_service_s"] > 0 for event in result["tool_events"])
        assert result["duration_prediction_metrics"]["mean_absolute_error_s"] is not None
        assert result["worker_resource_accounting"]["direct_demand_resource_s"] == pytest.approx(
            sum(
                event["service_s"]
                for event in result["tool_events"]
                if event["tool_name"] != "visit"
            ),
            abs=1e-6,
        )
        assert result["llm_events"][0]["workload_request_sha256"]
        task = result["task_results"][0]
        assert (
            task["flow_s"]
            == task["task_terminal_monotonic_s"]
            - task["scheduled_release_monotonic_s"]
        )
        assert (
            result["experiment_wall_s"]
            == result["experiment_ended_monotonic_s"]
            - result["experiment_started_monotonic_s"]
        )

    asyncio.run(scenario())
