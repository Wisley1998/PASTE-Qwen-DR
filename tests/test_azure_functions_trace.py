from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REPRO_SCRIPTS = ROOT / "reproduction/scripts"
for directory in (SCRIPTS, REPRO_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from azure_functions_trace import (  # noqa: E402
    load_azure_functions_window,
    sample_release_offsets,
)
from prepare_azure_trace_plans import (  # noqa: E402
    canonical_hash,
    materialize_arrival_plan,
)
from run_azure_arrival_comparison import (  # noqa: E402
    baseline_environment,
    build_cells,
)


def _csv_bytes() -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["HashOwner", "HashApp", "HashFunction", "Trigger", "1", "2", "3"])
    writer.writerow(["o1", "a1", "f1", "http", 3, 0, 2])
    writer.writerow(["o2", "a2", "f2", "timer", 1, 4, 0])
    return output.getvalue().encode("utf-8")


def _write_archive(path: Path) -> None:
    payload = _csv_bytes()
    with tarfile.open(path, "w:xz") as archive:
        info = tarfile.TarInfo("nested/invocations_per_function_md.anon.d01.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def _base_plan() -> dict:
    plan = {
        "schema": "paste_repro.trace_all_visit_live_plan.v1",
        "created_at": "2026-09-02T00:00:00+00:00",
        "configuration": {"candidate_policy": "budget_w5_cap10"},
        "traces": [
            {"trace_id": "trace-a", "session_id": "session-a", "steps": [1]},
            {"trace_id": "trace-b", "session_id": "session-b", "steps": [2]},
        ],
    }
    plan["plan_sha256"] = canonical_hash(plan)
    return plan


def test_loads_real_count_window_from_archive(tmp_path: Path) -> None:
    archive = tmp_path / "azurefunctions.tar.xz"
    _write_archive(archive)
    window = load_azure_functions_window(
        archive, day=1, start_minute=0, duration_minutes=3
    )

    assert window.counts == (4, 4, 2)
    assert window.function_rows == 2
    assert window.raw_invocations == 10
    assert window.csv_member.endswith("invocations_per_function_md.anon.d01.csv")


def test_release_sampling_is_exact_sorted_and_reproducible(tmp_path: Path) -> None:
    archive = tmp_path / "azurefunctions.tar.xz"
    _write_archive(archive)
    window = load_azure_functions_window(
        archive, day=1, start_minute=0, duration_minutes=3
    )

    first, metadata = sample_release_offsets(
        window, session_count=6, time_compression=10, seed=17
    )
    second, _ = sample_release_offsets(
        window, session_count=6, time_compression=10, seed=17
    )

    assert first == second
    assert len(first) == 6
    assert first == sorted(first)
    assert first[0] == 0.0
    assert metadata["raw_invocations_in_window"] == 10
    assert metadata["sampled_without_replacement"] == 6


def test_arrival_plan_clones_call_graph_and_isolates_session_cache() -> None:
    base = _base_plan()
    mapped = materialize_arrival_plan(
        base,
        [0.0, 0.2, 0.7],
        arrival_process={"kind": "azure_functions_2019"},
        mapping_seed=9,
    )

    assert base["traces"][0]["session_id"] == "session-a"
    assert [row["release_offset_s"] for row in mapped["traces"]] == [0.0, 0.2, 0.7]
    assert len({row["session_id"] for row in mapped["traces"]}) == 3
    assert all(row["steps"] in ([1], [2]) for row in mapped["traces"])
    unsigned = dict(mapped)
    expected = unsigned.pop("plan_sha256")
    assert expected == canonical_hash(unsigned)
    assert mapped["arrival_process"]["agent_internal_call_graph_changed"] is False


def test_rejects_more_sessions_than_observed_invocations(tmp_path: Path) -> None:
    archive = tmp_path / "azurefunctions.tar.xz"
    _write_archive(archive)
    window = load_azure_functions_window(
        archive, day=1, start_minute=0, duration_minutes=3
    )
    with pytest.raises(ValueError, match="only 10 raw invocations"):
        sample_release_offsets(window, session_count=11)


def test_baseline_keeps_prefix_cache_and_removes_joint_scheduler(tmp_path: Path) -> None:
    env = baseline_environment(
        {
            "PASTE_ENV_PREFIX": "/tmp/env",
            "VLLM_ENABLE_PREFIX_CACHING": "1",
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
        },
        gpus="0,1,2,3",
        port=8123,
        cell_root=tmp_path,
    )
    assert env["VLLM_ENABLE_PREFIX_CACHING"] == "1"
    assert env["VLLM_SCHED_POLICY"] == "fcfs"
    assert "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION" not in env


def test_matrix_contains_paired_cells_for_each_trace_and_cap(tmp_path: Path) -> None:
    cells = build_cells(
        {"azure_llm": tmp_path / "llm.json", "azure_functions": tmp_path / "fn.json"},
        [96, 72],
        repetitions=1,
    )
    assert len(cells) == 8
    keys = {(cell.trace_name, cell.max_active_tasks, cell.system) for cell in cells}
    assert ("azure_llm", 96, "vllm_baseline") in keys
    assert ("azure_llm", 96, "full") in keys
    assert ("azure_functions", 72, "vllm_baseline") in keys
    assert ("azure_functions", 72, "full") in keys
