#!/usr/bin/env python3
"""Run paired vLLM-baseline/FULL experiments on materialized Azure arrivals."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
from typing import Any, Mapping, Sequence

from run_trace_all_visit_coscheduling_matrix import (
    CENTER,
    CONFIG,
    RUNNER,
    START,
    STOP,
    cell_environment,
    implementation_mapping,
    load_config,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_BASE = ROOT / "reproduction/artifacts/azure_arrival_comparison/runs"
SYSTEMS = ("vllm_baseline", "full")


def canonical_hash(payload: Any) -> str:
    wire = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


@dataclass(frozen=True)
class ExperimentCell:
    repetition: int
    trace_name: str
    plan: Path
    max_active_tasks: int
    system: str

    @property
    def label(self) -> str:
        return (
            f"r{self.repetition:02d}__{self.trace_name}__"
            f"c{self.max_active_tasks}__{self.system}"
        )


def load_arrival_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "paste_repro.trace_all_visit_live_plan.v1":
        raise ValueError(f"unsupported live plan: {path}")
    if not payload.get("arrival_process"):
        raise ValueError(f"plan has no external arrival process: {path}")
    expected = payload.get("plan_sha256")
    unsigned = dict(payload)
    unsigned.pop("plan_sha256", None)
    if expected != canonical_hash(unsigned):
        raise ValueError(f"plan checksum mismatch: {path}")
    offsets = [float(row.get("release_offset_s", -1.0)) for row in payload["traces"]]
    if not offsets or offsets != sorted(offsets) or offsets[0] < 0.0:
        raise ValueError(f"plan release offsets are invalid: {path}")
    return payload


def parse_named_plans(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"plan must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise ValueError(f"unsupported trace name: {name!r}")
        path = Path(raw_path).resolve()
        if name in result:
            raise ValueError(f"duplicate trace name: {name}")
        load_arrival_plan(path)
        result[name] = path
    if not result:
        raise ValueError("at least one --plan NAME=PATH is required")
    return result


def baseline_environment(
    base: Mapping[str, str], *, gpus: str, port: int, cell_root: Path
) -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("VLLM_SCHED_"):
            env.pop(key)
    env.update({key: value for key, value in base.items() if not key.startswith("VLLM_SCHED_")})
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpus,
            "VLLM_PORT": str(port),
            "VLLM_REQUIRE_NEW": "1",
            "VLLM_STATE_DIR": str(cell_root / "state"),
            "VLLM_LOG_DIR": str(cell_root / "server"),
            "VLLM_SCHED_POLICY": "fcfs",
            "VLLM_ENABLE_PREFIX_CACHING": "1",
        }
    )
    return env


def environment_for_cell(
    base: Mapping[str, str], cell: ExperimentCell, *, gpus: str, port: int, cell_root: Path
) -> dict[str, str]:
    if cell.system == "vllm_baseline":
        return baseline_environment(base, gpus=gpus, port=port, cell_root=cell_root)
    return cell_environment(base, CENTER, gpus=gpus, port=port, cell_root=cell_root)


def runner_command(
    python: Path,
    cell: ExperimentCell,
    *,
    output: Path,
    port: int,
    visit_capacity: int,
    speculative_cap: int,
) -> list[str]:
    treatment = cell.system == "full"
    return [
        str(python),
        str(RUNNER),
        "--prepared-plan",
        str(cell.plan),
        "--output-dir",
        str(output),
        "--mode",
        "coscheduled_speculation" if treatment else "vllm_baseline",
        "--admission-backend",
        "engine_joint",
        "--server-url",
        f"http://127.0.0.1:{port}",
        "--max-active-tasks",
        str(cell.max_active_tasks),
        "--visit-capacity",
        str(visit_capacity),
        "--speculative-cap",
        str(speculative_cap if treatment else 0),
    ]


def build_cells(
    plans: Mapping[str, Path], concurrencies: Sequence[int], repetitions: int
) -> list[ExperimentCell]:
    return [
        ExperimentCell(repetition, trace_name, plan, concurrency, system)
        for repetition in range(1, repetitions + 1)
        for trace_name, plan in plans.items()
        for concurrency in concurrencies
        for system in SYSTEMS
    ]


def aggregate(run_root: Path, cells: Sequence[ExperimentCell]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for cell in cells:
        summary_path = run_root / cell.label / "evidence/summary.json"
        if summary_path.is_file():
            results.append(
                {
                    "cell": asdict(cell) | {"plan": str(cell.plan)},
                    "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                    "evidence": str(summary_path),
                }
            )
    payload = {
        "schema": "paste_repro.azure_arrival_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(results) == len(cells) else "in_progress",
        "expected_cells": len(cells),
        "completed_cells": len(results),
        "results": results,
    }
    write_json(run_root / "aggregate.json", payload)
    (run_root / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def render_report(payload: Mapping[str, Any]) -> str:
    indexed: dict[tuple[int, str, int, str], Mapping[str, Any]] = {}
    for row in payload["results"]:
        cell = row["cell"]
        indexed[(cell["repetition"], cell["trace_name"], cell["max_active_tasks"], cell["system"])] = row["summary"]
    groups = sorted({key[:3] for key in indexed})
    lines = [
        "# Azure arrival trace: vLLM baseline vs FULL",
        "",
        f"Status: `{payload['status']}` ({payload['completed_cells']}/{payload['expected_cells']} cells).",
        "",
        "Baseline is native vLLM FCFS with prefix caching. FULL adds the frozen",
        "Joint-v2 co-scheduler, forecast-aware physical-KV admission, and the",
        "preemptible all-Visit speculative executor.",
        "",
        "| Rep | Arrival trace | Client cap | Baseline mean | FULL mean | Mean speedup | Baseline p95 | FULL p95 | Wall speedup | FULL Visit hit / amp. |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    paired_speedups: dict[tuple[str, int], list[float]] = {}
    for repetition, trace_name, concurrency in groups:
        baseline = indexed.get((repetition, trace_name, concurrency, "vllm_baseline"))
        full = indexed.get((repetition, trace_name, concurrency, "full"))
        if baseline is None or full is None:
            continue
        mean_speedup = baseline["mean_task_flow_s"] / full["mean_task_flow_s"]
        wall_speedup = baseline["experiment_wall_s"] / full["experiment_wall_s"]
        paired_speedups.setdefault((trace_name, concurrency), []).append(mean_speedup)
        lines.append(
            f"| {repetition} | {trace_name} | {concurrency} "
            f"| {baseline['mean_task_flow_s']:.3f} s | {full['mean_task_flow_s']:.3f} s "
            f"| {mean_speedup:.3f}x | {baseline['p95_task_flow_s']:.3f} s "
            f"| {full['p95_task_flow_s']:.3f} s | {wall_speedup:.3f}x "
            f"| {full['realized_visit_hit_rate']:.2%} / {full['visit_call_amplification']:.3f}x |"
        )
    if paired_speedups:
        lines.extend(["", "## Trace sensitivity", ""])
        for (trace_name, concurrency), values in sorted(paired_speedups.items()):
            lines.append(
                f"- `{trace_name}`, cap {concurrency}: mean-flow speedup "
                f"{statistics.fmean(values):.3f}x over {len(values)} repetition(s)."
            )
    lines.extend(
        [
            "",
            "Flow time starts at each external trace release, so it includes any",
            "client-cap queueing. Both systems reuse the identical checksummed",
            "materialized plan for each arrival trace/cap pair.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument("--plan", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[96, 72])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=20260903)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--visit-capacity", type=int, default=16)
    parser.add_argument("--speculative-cap", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_tag):
        parser.error("run_tag contains unsupported characters")
    if args.repetitions <= 0 or not 1 <= args.port <= 65535:
        parser.error("repetitions/port out of range")
    if len(args.gpus.split(",")) != 4:
        parser.error("--gpus must contain four comma-separated GPU IDs")
    if any(value <= 0 or value > 100 for value in args.concurrencies):
        parser.error("every concurrency must be in [1, 100]")
    if not 0 <= args.speculative_cap <= args.visit_capacity:
        parser.error("invalid speculative capacity")
    return args


def main() -> None:
    args = parse_args()
    plans = parse_named_plans(args.plan)
    base = load_config(args.config)
    python = Path(base["PASTE_ENV_PREFIX"]) / "bin/python"
    cells = build_cells(plans, args.concurrencies, args.repetitions)
    order = list(cells)
    random.Random(args.order_seed).shuffle(order)
    run_root = args.run_base / args.run_tag
    contract = {
        "schema": "paste_repro.azure_arrival_comparison_plan.v1",
        "run_tag": args.run_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plans": {
            name: {
                "path": str(path),
                "arrival_process": load_arrival_plan(path)["arrival_process"],
            }
            for name, path in plans.items()
        },
        "concurrencies": args.concurrencies,
        "repetitions": args.repetitions,
        "order_seed": args.order_seed,
        "execution_order": [cell.label for cell in order],
        "fresh_server_per_cell": True,
        "systems": {
            "vllm_baseline": "vLLM FCFS + native prefix caching; no speculation",
            "full": "native prefix + Joint-v2 + physical-KV admission + all-Visit speculation",
        },
        "full_center_mapping": implementation_mapping(CENTER),
        "visit_capacity": args.visit_capacity,
        "speculative_cap": args.speculative_cap,
    }
    if args.check_only:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    if run_root.exists():
        raise SystemExit(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    write_json(run_root / "run_plan.json", contract)

    for cell in order:
        cell_root = run_root / cell.label
        evidence = cell_root / "evidence"
        evidence.mkdir(parents=True)
        env = environment_for_cell(
            base, cell, gpus=args.gpus, port=args.port, cell_root=cell_root
        )
        command = runner_command(
            python,
            cell,
            output=evidence,
            port=args.port,
            visit_capacity=args.visit_capacity,
            speculative_cap=args.speculative_cap,
        )
        write_json(
            cell_root / "cell_contract.json",
            {
                "cell": asdict(cell) | {"plan": str(cell.plan)},
                "environment": {
                    key: value
                    for key, value in sorted(env.items())
                    if key.startswith(("VLLM_", "MODEL_", "CUDA_"))
                },
                "runner_command": command,
            },
        )
        try:
            subprocess.run([str(START)], cwd=ROOT, env=env, check=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        finally:
            subprocess.run([str(STOP)], cwd=ROOT, env=env, check=False)
        server_log = cell_root / "server" / f"vllm_{args.port}.log"
        if server_log.is_file():
            (evidence / "server.log").write_bytes(server_log.read_bytes())
        aggregate(run_root, cells)

    print(json.dumps(aggregate(run_root, cells), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
