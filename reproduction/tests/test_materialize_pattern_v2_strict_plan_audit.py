from __future__ import annotations

from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from paste_repro import pattern_v2_all_visit_online as pattern_online
from paste_repro.strict_trace_runtime import validate_signed_payload
from paste_repro.traces import LLMCall, ToolCall


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_pattern_v2_strict_plan.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_pattern_v2_strict_plan_audit_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


class _FakePredictor:
    artifact_sha256 = "a" * 64

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []

    def start_session(self, *, source_session_id: str, runtime_session_id: str) -> object:
        self.started.append((source_session_id, runtime_session_id))
        return object()


class _FakeTailPredictor:
    artifact_sha256 = "b" * 64

    def __init__(self, payload: dict[str, Any]) -> None:
        assert payload["artifact_sha256"] == self.artifact_sha256


def _fake_prepare_request(
    event: LLMCall, *, tokenizer: Any, max_model_len: int, output_cap: int
) -> dict[str, Any]:
    del tokenizer
    return {
        "call_index": event.call_index,
        "messages": [dict(message) for message in event.messages],
        "prompt_tokens": 7,
        "original_prompt_tokens": 7,
        "max_tokens": output_cap,
        "truncated": False,
        "max_model_len_for_test": max_model_len,
    }


def _predictor_payload() -> dict[str, Any]:
    return {
        "schema": pattern_online.SCHEMA,
        "evaluation_regime": "retrospective_crossfit",
        "claim_scope": "retrospective_crossfit",
        "uses_other_evaluation_root_labels": True,
        "prior_policy_development_used_evaluation_corpus": True,
        "predictor_uses_trace_timing": False,
        "artifact_sha256": _FakePredictor.artifact_sha256,
    }


def _tail_payload() -> dict[str, Any]:
    return {
        "artifact_sha256": _FakeTailPredictor.artifact_sha256,
        "training_role": "calibration",
        "uses_evaluation_labels": False,
        "training_provenance": {"session_ids": []},
    }


def _args(
    *,
    traces: Path,
    predictor: Path,
    tail: Path,
    output: Path,
    replicas: int,
    allow_smoke: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        traces=traces,
        predictor_artifact=predictor,
        tail_artifact=tail,
        root_ids_artifact=None,
        output_dir=output,
        tokenizer="unused-in-test",
        max_model_len=1024,
        output_cap=16,
        replicas=replicas,
        duration_ewma_alpha=0.35,
        clock_seed_sha256="0" * 64,
        trace_limit=None,
        allow_smoke_workload=allow_smoke,
    )


def _install_lightweight_dependencies(monkeypatch, predictor: _FakePredictor) -> None:
    monkeypatch.setattr(
        materializer,
        "_load_predictor",
        lambda path: (predictor, _predictor_payload()),
    )
    monkeypatch.setattr(materializer, "_tokenizer", lambda source: object())
    monkeypatch.setattr(materializer, "_prepare_request", _fake_prepare_request)
    monkeypatch.setattr(materializer, "CausalTailPredictor", _FakeTailPredictor)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | {
            key for child in value.values() for key in _all_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def _sealed_without_allowed_raw_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    normalized.pop("sealed_sha256")
    for row in normalized["trace_lineage"].values():
        row.pop("raw_source_sha256")
    return normalized


def test_timing_and_recorded_response_poison_cannot_change_execution_plan(
    tmp_path: Path, monkeypatch
) -> None:
    """Only the explicitly diagnostic raw-file digest may notice the poison."""

    predictor_path = tmp_path / "predictor.json"
    predictor_path.write_text(json.dumps(_predictor_payload()), encoding="utf-8")
    tail_path = tmp_path / "tail.json"
    tail_path.write_text(json.dumps(_tail_payload()), encoding="utf-8")
    predictor = _FakePredictor()
    _install_lightweight_dependencies(monkeypatch, predictor)

    base_rows = [
        {
            "event_type": "llm_call",
            "call_index": 0,
            "timestamp": 10.0,
            "total_time_ms": 1000.0,
            "inference_time_ms": 900.0,
            "rtt_ms": 10.0,
            "messages": [{"role": "user", "content": "question"}],
            "response": "recorded answer",
        },
        {
            "event_type": "tool_call",
            "call_index": 0,
            "timestamp": 11.0,
            "tool_name": "search",
            "tool_args": {"query": ["causal query"]},
            "timing_correction": {
                "duration_s": 1.5,
                "unit_duration_s": [1.5],
            },
        },
        {
            "event_type": "synthetic_tool_completion",
            "call_index": 0,
            "timestamp": 12.0,
            "tool_name": "search",
            "timing_correction_schema": "diagnostic-only",
        },
        {
            "event_type": "llm_call",
            "call_index": 1,
            "timestamp": 13.0,
            "total_time_ms": 2000.0,
            "inference_time_ms": 1800.0,
            "rtt_ms": 20.0,
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "current causal context"},
            ],
            "response": "another recorded answer",
        },
    ]
    poison_rows = copy.deepcopy(base_rows)
    poison_rows[0].update(
        {
            "timestamp": {"POISON_TIMESTAMP": True},
            "total_time_ms": ["POISON_TOTAL_TIME"],
            "inference_time_ms": {"POISON_INFERENCE_TIME": True},
            "rtt_ms": "POISON_RTT",
            "response": "POISON_RECORDED_RESPONSE",
        }
    )
    poison_rows[1].update(
        {
            "timestamp": ["POISON_TOOL_TIMESTAMP"],
            "timing_correction": {
                "duration_s": "POISON_DURATION",
                "unit_duration_s": ["POISON_UNIT_DURATION"],
            },
        }
    )
    poison_rows[2].update(
        {
            "timestamp": "POISON_SYNTHETIC_TIMESTAMP",
            "timing_correction_schema": "POISON_SYNTHETIC_CORRECTION",
        }
    )
    poison_rows[3].update(
        {
            "timestamp": None,
            "total_time_ms": {"POISON": True},
            "inference_time_ms": ["POISON"],
            "response": "POISON_SECOND_RECORDED_RESPONSE",
        }
    )

    outputs: list[Path] = []
    loaded_sources = []
    for label, rows in (("base", base_rows), ("poison", poison_rows)):
        trace_dir = tmp_path / label / "traces"
        trace_dir.mkdir(parents=True)
        trace_path = trace_dir / "trace_same_root.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        loaded = materializer._load_causal_source(trace_path)
        loaded_sources.append(loaded)
        output = tmp_path / label / "materialized"
        materializer.materialize(
            _args(
                    traces=trace_dir,
                    predictor=predictor_path,
                    tail=tail_path,
                output=output,
                replicas=1,
                allow_smoke=True,
            )
        )
        outputs.append(output)

    assert loaded_sources[0].events == loaded_sources[1].events
    first_llm = loaded_sources[1].events[0]
    assert isinstance(first_llm, LLMCall)
    assert first_llm.timestamp_s == 0.0
    assert first_llm.total_time_s == 0.0
    assert first_llm.inference_time_s == 0.0
    assert first_llm.response == ""
    first_tool = loaded_sources[1].events[1]
    assert isinstance(first_tool, ToolCall)
    assert first_tool.timestamp_s == 0.0
    assert first_tool.timing_correction is None
    assert materializer._logical_source_sha256(
        loaded_sources[0]
    ) == materializer._logical_source_sha256(loaded_sources[1])

    base_public = _read_json(outputs[0] / "public_plan.json")
    poison_public = _read_json(outputs[1] / "public_plan.json")
    base_sealed = _read_json(outputs[0] / "sealed_plan.json")
    poison_sealed = _read_json(outputs[1] / "sealed_plan.json")
    assert base_public == poison_public
    assert (
        base_sealed["trace_lineage"][next(iter(base_sealed["trace_lineage"]))][
            "raw_source_sha256"
        ]
        != poison_sealed["trace_lineage"][next(iter(poison_sealed["trace_lineage"]))][
            "raw_source_sha256"
        ]
    )
    assert _sealed_without_allowed_raw_provenance(
        base_sealed
    ) == _sealed_without_allowed_raw_provenance(poison_sealed)

    forbidden_trace_fields = {
        "timestamp",
        "timestamp_s",
        "total_time_ms",
        "total_time_s",
        "inference_time_ms",
        "inference_time_s",
        "rtt_ms",
        "timing_correction",
        "response",
        "duration_s",
        "unit_duration_s",
        "llm_overlap_s",
        "overlap_window_s",
    }
    assert not (_all_keys(poison_public) & forbidden_trace_fields)
    assert not (_all_keys(poison_sealed) & forbidden_trace_fields)
    serialized = json.dumps(
        {"public": poison_public, "sealed": poison_sealed}, ensure_ascii=False
    )
    assert "POISON_" not in serialized


def test_formal_materialization_is_100_roots_x2_and_binds_all_four_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    trace_dir = tmp_path / "formal_traces"
    trace_dir.mkdir()
    for index in range(100):
        rows = [
            {
                "event_type": "llm_call",
                "call_index": 0,
                "messages": [{"role": "user", "content": f"question-{index}"}],
                "response": "ignored",
            },
            {
                "event_type": "tool_call",
                "call_index": 0,
                "tool_name": "search",
                "tool_args": {"query": [f"query-{index}"]},
            },
            {
                "event_type": "llm_call",
                "call_index": 1,
                "messages": [{"role": "user", "content": "current context"}],
                "response": "ignored",
            },
            {
                "event_type": "tool_call",
                "call_index": 1,
                "tool_name": "visit",
                "tool_args": {"url": f"https://example.test/{index}"},
            },
        ]
        (trace_dir / f"trace_{index:03d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    predictor_path = tmp_path / "predictor.json"
    predictor_path.write_text(json.dumps(_predictor_payload()), encoding="utf-8")
    tail_path = tmp_path / "tail.json"
    tail_path.write_text(json.dumps(_tail_payload()), encoding="utf-8")
    predictor = _FakePredictor()
    _install_lightweight_dependencies(monkeypatch, predictor)
    monkeypatch.setattr(
        materializer,
        "CROSSFIT_LOGICAL_CORPUS_SHA256",
        materializer._logical_corpus_sha256(materializer._load_sources(trace_dir)),
    )
    output = tmp_path / "formal_materialized"
    manifest = materializer.materialize(
        _args(
            traces=trace_dir,
            predictor=predictor_path,
            tail=tail_path,
            output=output,
            replicas=2,
            allow_smoke=False,
        )
    )

    public = validate_signed_payload(
        _read_json(output / "public_plan.json"),
        "plan_sha256",
        label="audit public plan",
    )
    sealed = validate_signed_payload(
        _read_json(output / "sealed_plan.json"),
        "sealed_sha256",
        label="audit sealed plan",
    )
    validate_signed_payload(manifest, "manifest_sha256", label="audit manifest")
    source_counts = Counter(row["source_session_id"] for row in public["traces"])
    assert public["role"] == "crossfit"
    assert public["independent_source_roots"] == 100
    assert public["replicas_per_root"] == 2
    assert public["replicas"] == 200
    assert public["arrival_process"] == {
        "kind": "closed_burst",
        "tasks": 200,
        "release_span_s": 0.0,
    }
    assert len(public["traces"]) == 200
    assert len({row["trace_id"] for row in public["traces"]}) == 200
    assert len(source_counts) == 100
    assert set(source_counts.values()) == {2}
    assert len(sealed["trace_steps"]) == 200
    assert manifest["formal_workload"] is True
    assert manifest["source_totals"] == {
        "llm_calls": 200,
        "tool_calls": 200,
        "raw_visit_urls": 100,
        "executable_visit_urls": 100,
    }

    duration = _read_json(output / "public_duration_predictor.json")
    service_clock = _read_json(output / "private_service_clock.json")
    assert public["predictor_artifact_sha256"] == predictor.artifact_sha256
    assert (
        public["duration_predictor_artifact_sha256"]
        == duration["artifact_sha256"]
    )
    assert (
        sealed["service_clock_artifact_sha256"]
        == service_clock["artifact_sha256"]
    )
    assert (
        public["tail_predictor_artifact_sha256"]
        == _FakeTailPredictor.artifact_sha256
    )
    for binding in manifest["files"].values():
        assert binding["sha256"] == materializer.file_sha256(output / binding["path"])


def test_checked_in_qwen_trace_has_the_expected_pattern_v2_workload() -> None:
    sources = materializer._load_sources(materializer.DEFAULT_TRACES)
    counts = {
        "llm_calls": 0,
        "tool_calls": 0,
        "raw_visit_urls": 0,
        "executable_visit_urls": 0,
    }
    logical_rows = []
    for source in sources:
        logical_rows.append(
            {
                "source_session_id": source.session_id,
                "logical_no_timing_sha256": materializer._logical_source_sha256(source),
            }
        )
        for event in source.events:
            if isinstance(event, LLMCall):
                counts["llm_calls"] += 1
            elif isinstance(event, ToolCall):
                counts["tool_calls"] += 1
                if event.tool_name == "visit":
                    raw = event.tool_args.get("url")
                    values = [raw] if isinstance(raw, str) else list(raw or [])
                    counts["raw_visit_urls"] += sum(
                        isinstance(value, str) for value in values
                    )
                    counts["executable_visit_urls"] += len(
                        materializer._unique_executable_visit_urls(event.tool_args)
                    )

    assert len(sources) == 100
    assert counts == {
        "llm_calls": 873,
        "tool_calls": 599,
        "raw_visit_urls": 507,
        "executable_visit_urls": 499,
    }
    assert (
        materializer.canonical_sha256(logical_rows)
        == "c8eddcf9376754cc37056a1a1af7a42b5e786d7ed8c4af65d86f904431030fbc"
    )

    allowed_message_keys = {"role", "content"}
    allowed_tool_argument_keys = {
        "search": {"query"},
        "google_scholar": {"query"},
        "visit": {"url", "goal"},
    }
    for source in sources:
        for event in source.events:
            if isinstance(event, LLMCall):
                assert all(set(message) <= allowed_message_keys for message in event.messages)
            elif isinstance(event, ToolCall):
                assert event.tool_name in allowed_tool_argument_keys
                assert set(event.tool_args) <= allowed_tool_argument_keys[event.tool_name]

    freeze = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "strict_causal_no_oracle_v2_20260904"
        / "freeze"
    )
    final_root_path = freeze / "final.root_ids.json"
    selected, root_payload, role, expected_roots, expected_replicas = (
        materializer._select_sources(
            sources,
            predictor_schema=pattern_online.DEPLOYABLE_SCHEMA,
            root_ids_artifact=final_root_path,
        )
    )
    assert root_payload is not None
    assert root_payload["artifact_sha256"] == (
        "7e88c9a78240cb583124e0d6b13defc70c36548718b8229d470ff5bf5dc6e93f"
    )
    assert role == "deployable_final"
    assert (expected_roots, expected_replicas, len(selected)) == (30, 7, 30)
    final_logical_rows = [
        {
            "source_session_id": source.session_id,
            "logical_no_timing_sha256": materializer._logical_source_sha256(source),
        }
        for source in selected
    ]
    assert (
        materializer.canonical_sha256(final_logical_rows)
        == "34857c0cab48aa604db8907face0654e7b892a7a3b626cedd0188d79994030a7"
    )
    tail_payload = _read_json(freeze / "tail_predictor.json")
    tail_training_ids = set(tail_payload["training_provenance"]["session_ids"])
    assert len(tail_training_ids) == 40
    assert tail_training_ids.isdisjoint(source.session_id for source in selected)
