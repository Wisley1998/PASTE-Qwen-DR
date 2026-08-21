#!/usr/bin/env python3
"""Validate and summarize matched FCFS versus PASTE-joint replay runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.mapper import write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate matched live trace replays and enforce comparison invariants."
    )
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--joint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, policy: str) -> dict[str, Any]:
    summary_path = path / "summary.json"
    events_path = path / "request_events.jsonl"
    workload_path = path / "prepared_workload.json"
    for required in (summary_path, events_path, workload_path):
        if not required.is_file():
            raise FileNotFoundError(f"run is incomplete; missing {required}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not events:
        raise ValueError(f"run has no request events: {path}")
    if int(summary.get("requests_failed", -1)) != 0:
        raise ValueError(f"run has failed requests: {path}")
    if sum(bool(event.get("ok")) for event in events) != len(events):
        raise ValueError(f"run contains a non-OK event: {path}")
    if summary.get("metadata_source") != "online":
        raise ValueError(f"run did not use online scheduler metadata: {path}")
    if summary.get("workload", {}).get("tool_overlap_mode") != "learned":
        raise ValueError(f"run did not use learned tool overlap: {path}")

    by_trace: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_trace.setdefault(str(event["trace_id"]), []).append(event)
    task_e2e_s = [
        max(float(event["request_end_offset_s"]) for event in trace_events)
        for trace_events in by_trace.values()
    ]
    request_latency_s = [float(event["latency_s"]) for event in events]

    server_log = path / "server.log"
    server_text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.is_file() else ""
    return {
        "name": path.name,
        "policy": policy,
        "workload_sha256": file_sha256(workload_path),
        "trace_count": len(by_trace),
        "request_count": len(events),
        "requests_failed": 0,
        "task_e2e_s": {
            "mean": statistics.fmean(task_e2e_s),
            "p50": percentile(task_e2e_s, 0.50),
            "p95": percentile(task_e2e_s, 0.95),
            "max": max(task_e2e_s),
        },
        "request_latency_s": {
            "mean": statistics.fmean(request_latency_s),
            "p95": percentile(request_latency_s, 0.95),
        },
        "mean_queue_time_s": float(summary["avg_queue_time_s"]),
        "experiment_wall_time_s": float(summary["experiment_wall_time_s"]),
        "timeline_max_running": float(summary["timeline_max_running"]),
        "timeline_max_waiting": float(summary["timeline_max_waiting"]),
        "tool_prediction": summary["workload"].get("tool_prediction", {}),
        "metadata_source": summary["metadata_source"],
        "scheduler_evidence": {
            "v1_install_lines": server_text.count(
                "installed policy=online_joint_pacer_v2 v0=True v1=True"
            ),
            "runtime_joint_lines": server_text.count("[sched_policy_patch:joint]"),
            "error_lines": sum(
                server_text.count(marker)
                for marker in (
                    "scheduler policy patch error",
                    "unknown VLLM_SCHED_POLICY",
                    "not installed",
                )
            ),
        },
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_at(*keys: str) -> float:
        values: list[float] = []
        for run in runs:
            value: Any = run
            for key in keys:
                value = value[key]
            values.append(float(value))
        return statistics.fmean(values)

    return {
        "run_count": len(runs),
        "task_e2e_s": {
            metric: mean_at("task_e2e_s", metric)
            for metric in ("mean", "p50", "p95", "max")
        },
        "request_latency_s": {
            metric: mean_at("request_latency_s", metric)
            for metric in ("mean", "p95")
        },
        "mean_queue_time_s": mean_at("mean_queue_time_s"),
        "experiment_wall_time_s": mean_at("experiment_wall_time_s"),
    }


def relative_reduction(baseline: float, optimized: float) -> float:
    return (baseline - optimized) / baseline if baseline else 0.0


def main() -> int:
    args = parse_args()
    baseline_paths = [path.resolve() for path in args.baseline]
    joint_paths = [path.resolve() for path in args.joint]
    if len(baseline_paths) != len(set(baseline_paths)) or len(joint_paths) != len(set(joint_paths)):
        raise ValueError("duplicate run directory supplied within one policy")
    if set(baseline_paths) & set(joint_paths):
        raise ValueError("baseline and joint run directories must be disjoint")
    if len(baseline_paths) != len(joint_paths):
        raise ValueError("baseline and joint must contain the same number of runs")

    baseline_runs = [load_run(path, "fcfs") for path in baseline_paths]
    joint_runs = [load_run(path, "online_joint_pacer_v2") for path in joint_paths]
    all_runs = baseline_runs + joint_runs
    workload_hashes = {run["workload_sha256"] for run in all_runs}
    trace_counts = {run["trace_count"] for run in all_runs}
    request_counts = {run["request_count"] for run in all_runs}
    if len(workload_hashes) != 1 or len(trace_counts) != 1 or len(request_counts) != 1:
        raise ValueError("runs do not use the same prepared workload and request set")
    if any(
        run["scheduler_evidence"]["error_lines"]
        or run["scheduler_evidence"]["v1_install_lines"] <= 0
        or run["scheduler_evidence"]["runtime_joint_lines"] <= 0
        for run in joint_runs
    ):
        raise ValueError("joint run lacks clean v1 install/runtime scheduler evidence")

    baseline = aggregate(baseline_runs)
    joint = aggregate(joint_runs)
    comparison = {
        "schema": "paste_repro.joint_ab_summary",
        "version": 1,
        "status": "functional_ab_not_full_paper_reproduction",
        "comparison_invariants": {
            "same_workload_sha256": next(iter(workload_hashes)),
            "trace_count": next(iter(trace_counts)),
            "request_count": next(iter(request_counts)),
            "metadata_source": "online",
            "tool_overlap_mode": "learned",
            "all_requests_succeeded": True,
        },
        "baseline": baseline,
        "joint": joint,
        "relative_reduction": {
            "task_e2e_mean": relative_reduction(
                baseline["task_e2e_s"]["mean"], joint["task_e2e_s"]["mean"]
            ),
            "task_e2e_p50": relative_reduction(
                baseline["task_e2e_s"]["p50"], joint["task_e2e_s"]["p50"]
            ),
            "task_e2e_p95": relative_reduction(
                baseline["task_e2e_s"]["p95"], joint["task_e2e_s"]["p95"]
            ),
            "request_latency_mean": relative_reduction(
                baseline["request_latency_s"]["mean"],
                joint["request_latency_s"]["mean"],
            ),
            "mean_queue_time": relative_reduction(
                baseline["mean_queue_time_s"], joint["mean_queue_time_s"]
            ),
            "experiment_wall_time": relative_reduction(
                baseline["experiment_wall_time_s"], joint["experiment_wall_time_s"]
            ),
        },
        "runs": all_runs,
        "interpretation": (
            "Positive means lower latency. A single functional A/B can establish that "
            "the path runs and has a directional gain on selected metrics; it is not a "
            "statistically replicated paper result."
        ),
    }
    if args.output:
        write_json_atomic(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
