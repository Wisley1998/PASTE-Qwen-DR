from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from trace_experiment_lib import prepare_trace_workload, summarize_workload  # noqa: E402
from reproduction.paste_repro.mapper import URLRankMapper, save_artifact  # noqa: E402
from reproduction.paste_repro.traces import (  # noqa: E402
    LLMCall,
    SearchResult,
    SearchVisitTransition,
    ToolCall,
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        text = "\n".join(str(message.get("content", "")) for message in messages)
        return list(range(max(1, len(text.split()))))

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(max(1, len(str(text).split()))))


def _artifact(path: Path) -> dict:
    search = ToolCall(0, 1.0, "search", {"query": ["training"]}, 1)
    decision = LLMCall(1, 2.0, 0.5, 0.5, (), "", 2)
    visit = ToolCall(1, 2.1, "visit", {"url": ["https://train/rank-two"]}, 3)
    transition = SearchVisitTransition(
        session_id="train.jsonl",
        search=search,
        decision_llm=decision,
        visit=visit,
        completion_llm=None,
        search_results=(
            SearchResult("https://train/rank-one", 1, 0, 0),
            SearchResult("https://train/rank-two", 2, 1, 0),
        ),
        authoritative_urls=("https://train/rank-two",),
        baseline_stall_s=1.0,
        overlap_window_s=0.5,
    )
    mapper = URLRankMapper().fit([transition], searches_seen=1)
    artifact = mapper.to_artifact(
        {
            "algorithm": "unit-test whole-session split",
            "seed": "fixed",
            "train_ratio": 0.5,
            "train_sessions": [{"session_id": "train.jsonl", "sha256": "a"}],
            "held_out_sessions": [{"session_id": "eval.jsonl", "sha256": "b"}],
        }
    )
    save_artifact(path, artifact)
    return artifact


def _events(
    *,
    authoritative_urls: list[str] | None = None,
    completion_timestamp: float = 5.0,
    completion_content: str = "future completion input",
) -> list[dict]:
    if authoritative_urls is None:
        authoritative_urls = ["https://eval/rank-two", "https://eval/unseen"]
    visible_search_response = """<tool_response>
1. [one](https://eval/rank-one)
2. [two](https://eval/rank-two)
</tool_response>"""
    return [
        {
            "event_type": "llm_call",
            "call_index": 0,
            "timestamp": 1.0,
            "total_time_ms": 100.0,
            "inference_time_ms": 100.0,
            "messages": [{"role": "user", "content": "question"}],
            "response": "search decision",
        },
        {
            "event_type": "tool_call",
            "call_index": 0,
            "timestamp": 1.1,
            "tool_name": "search",
            "tool_args": {"query": ["evaluation"]},
        },
        {
            "event_type": "llm_call",
            "call_index": 1,
            "timestamp": 2.0,
            "total_time_ms": 500.0,
            "inference_time_ms": 500.0,
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "user", "content": visible_search_response},
            ],
            "response": "visit decision",
        },
        {
            "event_type": "tool_call",
            "call_index": 1,
            "timestamp": 2.1,
            "tool_name": "visit",
            "tool_args": {"url": authoritative_urls, "goal": "future argument"},
        },
        {
            "event_type": "llm_call",
            "call_index": 2,
            "timestamp": completion_timestamp,
            "total_time_ms": 100.0,
            "inference_time_ms": 100.0,
            "messages": [{"role": "user", "content": completion_content}],
            "response": "future output tokens can change",
        },
    ]


def _write_trace(trace_dir: Path, events: list[dict]) -> None:
    trace_dir.mkdir()
    (trace_dir / "eval.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _prepare(trace_dir: Path, artifact_path: Path, mode: str = "learned") -> dict:
    return prepare_trace_workload(
        trace_dir=trace_dir,
        tokenizer=_FakeTokenizer(),
        target_trace_count=1,
        max_model_len=4096,
        max_output_tokens_cap=64,
        min_output_tokens_floor=16,
        output_token_buffer=4,
        tool_overlap_mode=mode,
        tool_prediction_model=artifact_path if mode == "learned" else None,
        tool_prediction_top_k=1,
    )


def _visit_request(workload: dict) -> dict:
    return next(
        request
        for request in workload["traces"][0]["requests"]
        if request["tool_kind_before"] == "visit"
    )


def test_predictions_are_invariant_to_all_future_authoritative_fields(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "mapper.json"
    artifact = _artifact(artifact_path)

    original_dir = tmp_path / "original"
    _write_trace(original_dir, _events())
    original = _prepare(original_dir, artifact_path)

    changed_url_dir = tmp_path / "changed-url"
    _write_trace(
        changed_url_dir,
        _events(authoritative_urls=["https://eval/rank-one", "https://eval/unseen"]),
    )
    changed_url = _prepare(changed_url_dir, artifact_path)

    changed_future_dir = tmp_path / "changed-future"
    _write_trace(
        changed_future_dir,
        _events(
            completion_timestamp=2.25,
            completion_content="future " * 500,
        ),
    )
    changed_future = _prepare(changed_future_dir, artifact_path)

    original_request = _visit_request(original)
    changed_url_request = _visit_request(changed_url)
    changed_future_request = _visit_request(changed_future)
    expected_candidates = ["https://eval/rank-two"]
    assert original_request["tool_prediction_candidates"] == expected_candidates
    assert changed_url_request["tool_prediction_candidates"] == expected_candidates
    assert changed_future_request["tool_prediction_candidates"] == expected_candidates

    # Only exact authoritative equality realizes a hit; the predictor itself is
    # independent of the future URL and future completion request tokenization.
    assert original_request["tool_prediction_exact_hits"] == 1
    assert changed_url_request["tool_prediction_exact_hits"] == 0
    assert original_request["tool_overlap_saved_s"] == pytest.approx(0.25)
    assert changed_url_request["tool_overlap_saved_s"] == 0.0
    assert changed_future_request["tool_prediction_exact_hits"] == 1
    assert changed_future_request["tool_overlap_saved_s"] == pytest.approx(0.025)

    prediction_summary = summarize_workload(original)["tool_prediction"]
    assert prediction_summary == {
        "candidate_count": 1,
        "exact_hits": 1,
        "waste": 0,
        "artifact_sha256": artifact["artifact_sha256"],
        "top_k": 1,
    }
    assert original["meta"]["tool_prediction_artifact_sha256"] == artifact[
        "artifact_sha256"
    ]


def test_learned_mode_fails_closed_on_missing_or_corrupt_artifact(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, _events())
    with pytest.raises(ValueError, match="tool-prediction-model"):
        prepare_trace_workload(
            trace_dir=trace_dir,
            tokenizer=_FakeTokenizer(),
            target_trace_count=1,
            max_model_len=4096,
            max_output_tokens_cap=64,
            min_output_tokens_floor=16,
            output_token_buffer=4,
            tool_overlap_mode="learned",
        )

    artifact_path = tmp_path / "mapper.json"
    artifact = _artifact(artifact_path)
    corrupt = copy.deepcopy(artifact)
    corrupt["mapper"]["rank_counts"]["2"] = 99
    artifact_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        _prepare(trace_dir, artifact_path)


def test_none_mode_preserves_legacy_request_and_meta_shape(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    _write_trace(trace_dir, _events())
    artifact_path = tmp_path / "unused.json"
    workload = _prepare(trace_dir, artifact_path, mode="none")

    assert not any(
        key.startswith("tool_prediction_") for key in workload["meta"]
    )
    assert all(
        request["max_tokens"] >= 16
        for trace in workload["traces"]
        for request in trace["requests"]
    )
    for request in workload["traces"][0]["requests"]:
        assert not any(key.startswith("tool_prediction_") for key in request)
    summary = summarize_workload(workload)
    assert summary["tool_overlap_mode"] == "none"
    assert "tool_prediction" not in summary
