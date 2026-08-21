from __future__ import annotations

from pathlib import Path
import sys


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from validate_physical_kv_admission import (  # noqa: E402
    _select_experiment_physical_log,
)
from run_vllm_trace_experiment import parse_vllm_log_segment  # noqa: E402


def _physical_line(
    *,
    running: int,
    waiting: int,
    admit: int,
    effective_cap: int,
    write_count: int,
    committed_tokens: int,
    predicted_admit_tokens: int,
    reason: str = "budget",
) -> str:
    return (
        "[sched_policy_patch:physical_kv] decision=admit "
        f"reason={reason} num_gpu_blocks=100 block_size=16 "
        "capacity_tokens=1600 target_utilization=0.900000 "
        "budget_tokens=1440 usage=0.500000 live_tokens=800 "
        "logical_live_tokens=800 running_growth_tokens=0 reserved_tokens=0 "
        f"committed_tokens={committed_tokens} "
        f"predicted_admit_tokens={predicted_admit_tokens} "
        f"waiting={waiting} running={running} fit_admit={admit} "
        f"admit={admit} effective_cap={effective_cap} native_cap=256 "
        f"capacity_write_source=physical_kv capacity_write_count={write_count} "
        "rescue=0"
    )


def test_raw_scope_excludes_only_warmup_cap_one_and_recovers_legacy_hold() -> None:
    warmup = _physical_line(
        running=0,
        waiting=1,
        admit=1,
        effective_cap=1,
        write_count=1,
        committed_tokens=100,
        predicted_admit_tokens=100,
    )
    first_experiment = _physical_line(
        running=1,
        waiting=2,
        admit=1,
        effective_cap=2,
        write_count=2,
        committed_tokens=200,
        predicted_admit_tokens=100,
    )
    legacy_rejected_hold = _physical_line(
        running=2,
        waiting=1,
        admit=0,
        effective_cap=2,
        write_count=3,
        committed_tokens=1456,
        predicted_admit_tokens=0,
        reason="forecast_hold",
    )
    last_experiment = _physical_line(
        running=2,
        waiting=1,
        admit=1,
        effective_cap=3,
        write_count=4,
        committed_tokens=300,
        predicted_admit_tokens=100,
    )
    raw_text = "\n".join(
        [warmup, first_experiment, legacy_rejected_hold, last_experiment]
    )

    # Parser v1 retained counters 2 and 4, rejected counter 3, and never saw
    # the pre-run warm-up counter 1 because the runner used a byte offset.
    stored = parse_vllm_log_segment(
        first_experiment + "\n" + last_experiment
    )["physical_kv_admission"]
    stored["malformed_sample_count"] = 1
    stored["screening_gates"]["no_malformed_samples"] = False
    stored["screening_gates"]["passed"] = False

    (
        selected,
        selected_lines,
        selected_samples,
        selected_counts,
        prefix_counts,
        suffix_counts,
        full,
    ) = _select_experiment_physical_log(raw_text, stored, label="synthetic raw log")

    assert selected["sample_count"] == 3
    assert selected["malformed_sample_count"] == 0
    assert [sample["capacity_write_count"] for sample in selected_samples] == [
        2,
        3,
        4,
    ]
    assert selected_counts == [2, 3, 4]
    assert len(selected_lines) == 3
    assert prefix_counts == [1]
    assert suffix_counts == []
    assert full["sample_count"] == 4
    assert selected_samples[1]["reason"] == "forecast_hold"
    assert selected_samples[1]["committed_tokens"] > selected_samples[1][
        "budget_tokens"
    ]
    assert selected_samples[1]["admit"] == 0
    assert selected_samples[1]["predicted_admit_tokens"] == 0
