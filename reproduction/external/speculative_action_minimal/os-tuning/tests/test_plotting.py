from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_history(path: Path, *, mode: str, run_kind: str, request_type: str | None, model: str | None) -> None:
    requests = []
    if request_type is not None:
        requests.append(
            {
                "request_id": f"{request_type}_0001",
                "request_type": request_type,
                "role": request_type,
                "model": model,
                "call_timestamp": 1000.1,
                "response_timestamp": 1000.6,
                "apply_timestamp": 1000.7,
                "proposed_parameters": {"min_granularity_ns": 200_000},
                "token_metrics": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "applied": True,
                "error": None,
            }
        )
    payload = {
        "setup_mode": mode,
        "run_kind": run_kind,
        "benchmark": "sysbench_cpu",
        "started_at": 1000.0,
        "initial_apply_timestamp": 1000.0,
        "run_duration_s": 20 if run_kind == "reaction" else 200,
        "window_duration_s": 1,
        "initial_parameters": {"latency_ns": 1000, "min_granularity_ns": 10_000_000},
        "os_defaults": {"latency_ns": 1000, "min_granularity_ns": 3_000_000},
        "history": [
            {
                "iteration": 1,
                "timestamp": 1001.0,
                "window_start_time": 1000.0,
                "window_end_time": 1001.0,
                "parameters": {"latency_ns": 1000, "min_granularity_ns": 10_000_000},
                "benchmark_metrics": {"latency_p95": 50.0},
                "system_metrics": {},
            },
            {
                "iteration": 2,
                "timestamp": 1002.0,
                "window_start_time": 1001.0,
                "window_end_time": 1002.0,
                "parameters": {"latency_ns": 1000, "min_granularity_ns": 200_000},
                "benchmark_metrics": {"latency_p95": 30.0},
                "system_metrics": {},
            },
        ],
        "requests": requests,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_plot_script_generates_outputs(tmp_path):
    conv_actor = tmp_path / "conv_actor.json"
    conv_spec = tmp_path / "conv_spec.json"
    conv_dual = tmp_path / "conv_dual.json"
    react_actor = tmp_path / "react_actor.json"
    react_spec = tmp_path / "react_spec.json"
    react_dual = tmp_path / "react_dual.json"
    paper_untuned = tmp_path / "paper_untuned.json"
    paper_actor = tmp_path / "paper_actor.json"

    write_history(conv_actor, mode="actor_only", run_kind="convergence", request_type="actor", model="gemini-2.5-flash")
    write_history(conv_spec, mode="speculator_only", run_kind="convergence", request_type="speculator", model="gemini-2.5-flash-lite")
    write_history(conv_dual, mode="dual", run_kind="convergence", request_type="actor", model="gemini-2.5-flash")
    write_history(react_actor, mode="actor_only", run_kind="reaction", request_type="actor", model="gemini-2.5-flash")
    write_history(react_spec, mode="speculator_only", run_kind="reaction", request_type="speculator", model="gemini-2.5-flash-lite")
    write_history(react_dual, mode="dual", run_kind="reaction", request_type="speculator", model="gemini-2.5-flash-lite")
    write_history(paper_untuned, mode="untuned", run_kind="reaction", request_type=None, model=None)
    write_history(paper_actor, mode="actor_only", run_kind="reaction", request_type="actor", model="gemini-2.5-flash")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "results_root": str(tmp_path),
                "convergence": {
                    "actor_only": str(conv_actor),
                    "speculator_only": str(conv_spec),
                    "dual": str(conv_dual),
                },
                "reaction": {
                    "actor_only": str(react_actor),
                    "speculator_only": str(react_spec),
                    "dual": str(react_dual),
                },
                "paper_reaction": {
                    "untuned": str(paper_untuned),
                    "actor_only": str(paper_actor),
                    "dual": str(react_dual),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/plot_use_case.py", "--manifest", str(manifest)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    plots_dir = tmp_path / "plots"
    assert (plots_dir / "figure5_use_case.png").exists()
    assert (plots_dir / "figure7_reaction_trace.png").exists()
    assert (plots_dir / "figure9_tokens_cost.png").exists()
