from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from azure_llm_trace import (  # noqa: E402
    apply_azure_arrivals,
    load_azure_llm_invocations,
)


def _write_azure_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        writer.writerow(["2024-05-12 00:00:00.001163+00:00", "1452", "3"])
        writer.writerow(["2024-05-12 00:00:00.041683+00:00", "584", "7"])
        writer.writerow(["2024-05-12 00:00:01.001163+00:00", "900", "11"])
        writer.writerow(["2024-05-12 00:00:03.001163+00:00", "1200", "13"])


def _base_workload() -> dict:
    def request(call_index: int, wait_s: float, marker: str) -> dict:
        return {
            "call_index": call_index,
            "wait_after_prev_s": wait_s,
            "wait_after_prev_original_s": wait_s,
            "prompt_tokens": 100 + call_index,
            "target_output_tokens": 20,
            "max_tokens": 64,
            "truncated": False,
            "original_prompt_tokens": 100 + call_index,
            "messages": [{"role": "user", "content": marker}],
        }

    return {
        "meta": {"target_trace_count": 2, "tool_overlap_mode": "none"},
        "traces": [
            {
                "trace_id": "trace_000",
                "source_trace": "agent-a.jsonl",
                "variant_index": 0,
                "duplicated": False,
                "prefix_char": "",
                "initial_delay_s": 0.0,
                "truncated_calls": 0,
                "requests": [request(0, 0.0, "agent-a-0"), request(1, 2.5, "agent-a-1")],
            },
            {
                "trace_id": "trace_001",
                "source_trace": "agent-b.jsonl",
                "variant_index": 1,
                "duplicated": False,
                "prefix_char": "",
                "initial_delay_s": 0.0,
                "truncated_calls": 0,
                "requests": [request(0, 0.0, "agent-b-0"), request(1, 4.0, "agent-b-1")],
            },
        ],
    }


def test_load_slice_uses_first_selected_row_as_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "azure.csv"
    _write_azure_csv(csv_path)

    rows = load_azure_llm_invocations(
        csv_path,
        start_time="2024-05-12T00:00:00.020000+00:00",
        duration_s=1.5,
    )

    assert [row.row_number for row in rows] == [3, 4]
    assert rows[0].context_tokens == 584
    assert rows[1].generated_tokens == 11


def test_apply_changes_only_session_arrival_and_preserves_agent_calls(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "azure.csv"
    _write_azure_csv(csv_path)
    invocations = load_azure_llm_invocations(csv_path, max_sessions=3)
    base = _base_workload()
    base_before = copy.deepcopy(base)

    mapped = apply_azure_arrivals(
        base,
        invocations,
        source_file=csv_path,
        dataset_variant="conversation",
        arrival_speedup=2.0,
        mapping="round_robin",
    )

    assert base == base_before
    assert len(mapped["traces"]) == 3
    assert [trace["base_trace_id"] for trace in mapped["traces"]] == [
        "trace_000",
        "trace_001",
        "trace_000",
    ]
    assert [trace["initial_delay_s"] for trace in mapped["traces"]] == pytest.approx(
        [0.0, 0.02026, 0.5]
    )

    # Azure token lengths are provenance only; the Agent request is unchanged.
    first = mapped["traces"][0]
    assert first["azure_arrival"]["context_tokens"] == 1452
    assert first["requests"][0]["prompt_tokens"] == 100
    assert first["requests"][0]["messages"] == [
        {"role": "user", "content": "agent-a-0"}
    ]
    assert first["requests"][1] == base["traces"][0]["requests"][1]
    assert mapped["meta"]["arrival_process"][
        "azure_token_fields_used_for_agent_payload"
    ] is False
    assert mapped["meta"]["arrival_process"]["replay_span_s"] == pytest.approx(0.5)


def test_shuffled_mapping_is_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "azure.csv"
    _write_azure_csv(csv_path)
    rows = load_azure_llm_invocations(csv_path)

    first = apply_azure_arrivals(
        _base_workload(),
        rows,
        source_file=csv_path,
        dataset_variant="code",
        mapping="shuffled_round_robin",
        mapping_seed=17,
    )
    second = apply_azure_arrivals(
        _base_workload(),
        rows,
        source_file=csv_path,
        dataset_variant="code",
        mapping="shuffled_round_robin",
        mapping_seed=17,
    )
    assert [t["agent_template_index"] for t in first["traces"]] == [
        t["agent_template_index"] for t in second["traces"]
    ]


def test_rejects_bad_schema_and_double_application(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("TIMESTAMP,ContextTokens\n2024-05-12T00:00:00+00:00,1\n")
    with pytest.raises(ValueError, match="GeneratedTokens"):
        load_azure_llm_invocations(bad_path)

    csv_path = tmp_path / "azure.csv"
    _write_azure_csv(csv_path)
    rows = load_azure_llm_invocations(csv_path, max_sessions=1)
    once = apply_azure_arrivals(
        _base_workload(),
        rows,
        source_file=csv_path,
        dataset_variant="conversation",
    )
    with pytest.raises(ValueError, match="already has an arrival_process"):
        apply_azure_arrivals(
            once,
            rows,
            source_file=csv_path,
            dataset_variant="conversation",
        )


def test_offline_cli_writes_replayable_workload(tmp_path: Path) -> None:
    csv_path = tmp_path / "azure.csv"
    base_path = tmp_path / "base.json"
    output_path = tmp_path / "mapped.json"
    _write_azure_csv(csv_path)
    base_path.write_text(json.dumps(_base_workload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "prepare_azure_agent_workload.py"),
            "--agent-workload",
            str(base_path),
            "--azure-trace",
            str(csv_path),
            "--output",
            str(output_path),
            "--azure-max-sessions",
            "3",
            "--azure-arrival-speedup",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["traces"]) == 3
    assert payload["meta"]["arrival_process"]["kind"] == "azure_llm_inference_2024"
    assert '"arrival_process"' in completed.stdout


def test_integrated_runner_prepare_only_applies_azure_arrivals(tmp_path: Path) -> None:
    csv_path = tmp_path / "azure.csv"
    base_path = tmp_path / "base.json"
    output_dir = tmp_path / "run"
    _write_azure_csv(csv_path)
    base_path.write_text(json.dumps(_base_workload()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_vllm_trace_experiment.py"),
            "--prepared-workload",
            str(base_path),
            "--azure-arrival-trace",
            str(csv_path),
            "--azure-max-sessions",
            "3",
            "--azure-arrival-speedup",
            "2",
            "--output-dir",
            str(output_dir),
            "--speedup",
            "7",
            "--prepare-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    workload = json.loads(
        (output_dir / "prepared_workload.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (output_dir / "workload_summary.json").read_text(encoding="utf-8")
    )
    assert workload["traces"][2]["initial_delay_s"] == pytest.approx(0.5)
    assert workload["traces"][0]["requests"][1]["wait_after_prev_s"] == 2.5
    assert summary["arrival_process"]["arrival_speedup"] == 2.0
