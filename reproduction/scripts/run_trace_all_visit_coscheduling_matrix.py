#!/usr/bin/env python3
"""Run the paper-aligned FULL configuration and its sensitivity matrix.

Every cell is FULL. The matrix changes only abstract quantities named in the
paper; each quantity maps to the concrete Joint-v2 implementation bundle at
vLLM's waiting-to-running boundary. The rejected external Python admission
queue is deliberately not part of this experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "reproduction/configs/trace_all_visit_coscheduling.env.example"
RUNNER = ROOT / "reproduction/scripts/run_trace_all_visit_live.py"
START = ROOT / "reproduction/scripts/start_vllm.sh"
STOP = ROOT / "reproduction/scripts/stop_vllm.sh"
DEFAULT_PLAN = (
    ROOT
    / "reproduction/artifacts/trace_all_visit_coscheduling/plan/prepared_plan.json"
)
DEFAULT_RUN_BASE = ROOT / "reproduction/artifacts/full_paper_sensitivity/runs"
EXPORT = re.compile(r'export ([A-Z][A-Z0-9_]*)="([^"\\]*)"\Z')

# Data-supported center inherited from the registered formal-v9 Joint setup.
BASE_TOOL_BETA = 0.9
BASE_REMAINING_TOOL_WEIGHT = 0.35
BASE_FINAL_BONUS_S = 12.0
BASE_PROGRESS_BONUS_S = 8.0
BASE_CONTEXT_ALPHA = 1.4
BASE_AGING_ALPHA = 0.2
BASE_RESCUE_WAIT_S = 40.0
BASE_PRESSURE_HIGH = 0.93


@dataclass(frozen=True)
class Cell:
    label: str
    sensitivity_axis: str = "center"
    exposed_tool_gain_scale: float = 1.0
    aging_scale: float = 1.0
    gamma: float = 1.0
    pressure_high: float = BASE_PRESSURE_HIGH


CENTER = Cell("FULL-center")


def cells_for_suite(suite: str) -> tuple[Cell, ...]:
    """Return FULL-only, one-factor-at-a-time paper sensitivity cells."""

    if suite == "center":
        return (CENTER,)
    sensitivity = (
        replace(
            CENTER,
            label="FULL-gain-0p5",
            sensitivity_axis="ExposedToolGain",
            exposed_tool_gain_scale=0.5,
        ),
        replace(
            CENTER,
            label="FULL-gain-2p0",
            sensitivity_axis="ExposedToolGain",
            exposed_tool_gain_scale=2.0,
        ),
        replace(
            CENTER,
            label="FULL-aging-0p5",
            sensitivity_axis="Aging",
            aging_scale=0.5,
        ),
        replace(
            CENTER,
            label="FULL-aging-2p0",
            sensitivity_axis="Aging",
            aging_scale=2.0,
        ),
        replace(
            CENTER,
            label="FULL-gamma-0p5",
            sensitivity_axis="gamma",
            gamma=0.5,
        ),
        replace(
            CENTER,
            label="FULL-gamma-2p0",
            sensitivity_axis="gamma",
            gamma=2.0,
        ),
        replace(
            CENTER,
            label="FULL-Phigh-0p85",
            sensitivity_axis="P_low/P_high",
            pressure_high=0.85,
        ),
        replace(
            CENTER,
            label="FULL-Phigh-0p97",
            sensitivity_axis="P_low/P_high",
            pressure_high=0.97,
        ),
    )
    if suite == "sensitivity":
        return (CENTER, *sensitivity)
    raise ValueError(f"unknown suite: {suite}")


def load_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = EXPORT.fullmatch(line)
        if match is None:
            raise ValueError(f"unsupported config syntax at {path}:{number}")
        values[match.group(1)] = match.group(2)
    return values


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _fmt(value: float) -> str:
    return format(value, ".6g")


def implementation_mapping(cell: Cell) -> dict[str, str]:
    """Map each paper-level variable to one consistent code-level bundle."""

    gain = cell.exposed_tool_gain_scale
    aging = cell.aging_scale
    return {
        # ExposedToolGain comprises next-tool, task-progress and final-turn gain.
        # The causal stage lanes stay enabled in every FULL cell.
        "VLLM_SCHED_JOINT_V2_TOOL_BETA": _fmt(BASE_TOOL_BETA * gain),
        "VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT": _fmt(
            BASE_REMAINING_TOOL_WEIGHT * gain
        ),
        "VLLM_SCHED_JOINT_V2_FINAL_BONUS_S": _fmt(BASE_FINAL_BONUS_S * gain),
        "VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S": _fmt(
            BASE_PROGRESS_BONUS_S * gain
        ),
        # gamma weights context/logical-KV pressure within LLMPressure.
        "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA": _fmt(BASE_CONTEXT_ALPHA * cell.gamma),
        # Aging comprises continuous waiting credit plus the hard rescue clock.
        "VLLM_SCHED_TIME_AGING_ALPHA": _fmt(BASE_AGING_ALPHA * aging),
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": _fmt(
            BASE_RESCUE_WAIT_S / aging
        ),
        # P_low is the invariant work-conserving progress rule. P_high is the
        # forecast-aware physical-KV utilization boundary.
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": _fmt(
            cell.pressure_high
        ),
    }


def cell_environment(
    base: Mapping[str, str],
    cell: Cell,
    *,
    gpus: str,
    port: int,
    cell_root: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("VLLM_SCHED_"):
            env.pop(key)
    env.update(base)
    env.update(implementation_mapping(cell))
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpus,
            "VLLM_PORT": str(port),
            "VLLM_REQUIRE_NEW": "1",
            "VLLM_STATE_DIR": str(cell_root / "state"),
            "VLLM_LOG_DIR": str(cell_root / "server"),
            "VLLM_SCHED_POLICY": "online_joint_pacer_v2",
            "VLLM_ENABLE_PREFIX_CACHING": "1",
            "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY": "0",
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
        }
    )
    return env


def runner_command(
    python: Path,
    *,
    plan: Path,
    output: Path,
    port: int,
    max_active_tasks: int,
    visit_capacity: int,
    speculative_cap: int,
    trace_limit: int | None,
) -> list[str]:
    command = [
        str(python),
        str(RUNNER),
        "--prepared-plan", str(plan),
        "--output-dir", str(output),
        "--mode", "coscheduled_speculation",
        "--admission-backend", "engine_joint",
        "--server-url", f"http://127.0.0.1:{port}",
        "--max-active-tasks", str(max_active_tasks),
        "--visit-capacity", str(visit_capacity),
        "--speculative-cap", str(speculative_cap),
    ]
    if trace_limit is not None:
        command.extend(["--trace-limit", str(trace_limit)])
    return command


def render_report(payload: Mapping[str, Any]) -> str:
    rows = payload.get("results", [])
    center_rows = [row for row in rows if row["cell"]["label"] == CENTER.label]
    center = (
        statistics.fmean(row["summary"]["mean_task_flow_s"] for row in center_rows)
        if center_rows
        else None
    )
    lines = [
        "# Paper-aligned FULL sensitivity",
        "",
        "Every measured cell is FULL: native vLLM prefix caching, Joint-v2",
        "stage/gain/pressure ordering, forecast-aware physical-KV admission, and",
        "the preemptible all-Visit speculation policy. Explicit prefix-affinity",
        "reordering and the rejected client-side Python admission queue are off.",
        "",
        "| FULL cell | Paper axis | Mean task flow | p95 flow | Change vs FULL center | Visit hit | Call amp. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cell = row["cell"]
        summary = row["summary"]
        change = (
            (summary["mean_task_flow_s"] - center) / center
            if center and center > 0.0
            else None
        )
        change_text = f"{change:+.2%}" if change is not None else "n/a"
        lines.append(
            f"| {cell['label']} | {cell['sensitivity_axis']} "
            f"| {summary['mean_task_flow_s']:.3f} s "
            f"| {summary['p95_task_flow_s']:.3f} s "
            f"| {change_text} "
            f"| {summary['realized_visit_hit_rate']:.2%} "
            f"| {summary['visit_call_amplification']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## Paper-variable interpretation",
            "",
            "- `ExposedToolGain`: jointly scales next-tool, remaining-tool, progress and final-turn gain. Increasing it favors turns that expose or consume tool progress sooner; too much can delay expensive or long-context turns.",
            "- `Aging`: jointly scales waiting credit and the hard rescue deadline. Increasing it improves fairness sooner, but can weaken gain-efficient ordering.",
            "- `gamma`: scales the context/logical-KV part of `LLMPressure`. Increasing it protects the engine more strongly from long-context pressure, but can under-serve long-context tasks.",
            "- `P_low/P_high`: `P_low` is the fixed work-conserving progress rule; this sweep changes physical-KV `P_high`. Lower leaves more headroom but can underfill the batch; higher increases throughput opportunity and overload risk.",
            "",
            "This is sensitivity around one complete system, not a component ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(run_root: Path, cells: Sequence[Cell], repetitions: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for cell in cells:
            path = run_root / f"r{repetition:02d}" / cell.label / "evidence/summary.json"
            if not path.is_file():
                continue
            rows.append(
                {
                    "repetition": repetition,
                    "cell": asdict(cell),
                    "implementation_mapping": implementation_mapping(cell),
                    "summary": json.loads(path.read_text(encoding="utf-8")),
                }
            )
    payload = {
        "schema": "paste_repro.full_paper_sensitivity.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "full_only": True,
        "results": rows,
    }
    write_json(run_root / "aggregate.json", payload)
    (run_root / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def execution_orders(
    cells: Sequence[Cell], repetitions: int, seed: int
) -> tuple[tuple[Cell, ...], ...]:
    """Create reproducible independently shuffled orders for drift control."""

    orders: list[tuple[Cell, ...]] = []
    for repetition in range(1, repetitions + 1):
        order = list(cells)
        random.Random(seed + repetition).shuffle(order)
        orders.append(tuple(order))
    return tuple(orders)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument("--suite", choices=["center", "sensitivity"], default="sensitivity")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20260902)
    parser.add_argument("--max-active-tasks", type=int, default=80)
    # Fixed-half is the conservative shared-pool point with positive replay
    # evidence and less waste than reserve-one/adaptive-idle-fill.
    parser.add_argument("--visit-capacity", type=int, default=16)
    parser.add_argument("--speculative-cap", type=int, default=8)
    parser.add_argument("--trace-limit", type=int)
    parser.add_argument(
        "--cells",
        help="comma-separated FULL cell labels selected from the requested suite",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_tag):
        parser.error("run_tag contains unsupported characters")
    if args.repetitions <= 0 or not 1 <= args.port <= 65535:
        parser.error("repetitions/port out of range")
    if len(args.gpus.split(",")) != 4:
        parser.error("--gpus must contain four comma-separated GPU IDs")
    if not 0 <= args.speculative_cap <= args.visit_capacity:
        parser.error("invalid speculative capacity")
    if args.trace_limit is not None and args.trace_limit <= 0:
        parser.error("--trace-limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    python = Path(base["PASTE_ENV_PREFIX"]) / "bin/python"
    cells = cells_for_suite(args.suite)
    if args.cells:
        requested = tuple(part.strip() for part in args.cells.split(",") if part.strip())
        known = {cell.label: cell for cell in cells}
        unknown = sorted(set(requested) - set(known))
        if unknown:
            raise SystemExit(f"unknown cells for suite {args.suite}: {unknown}")
        cells = tuple(known[label] for label in requested)
        if not cells:
            raise SystemExit("--cells selected no cells")
    run_root = args.run_base / args.run_tag
    orders = execution_orders(cells, args.repetitions, args.order_seed)
    plan = {
        "schema": "paste_repro.full_paper_sensitivity_plan.v1",
        "run_tag": args.run_tag,
        "suite": args.suite,
        "repetitions": args.repetitions,
        "order_seed": args.order_seed,
        "execution_orders": [
            [cell.label for cell in order]
            for order in orders
        ],
        "fresh_server_per_cell": True,
        "full_only": True,
        "admission_backend": "engine_joint",
        "cells": [
            {
                **asdict(cell),
                "implementation_mapping": implementation_mapping(cell),
            }
            for cell in cells
        ],
        "prepared_plan": str(args.plan.resolve()),
        "max_active_tasks": args.max_active_tasks,
        "visit_capacity": args.visit_capacity,
        "speculative_cap": args.speculative_cap,
        "trace_limit": args.trace_limit,
    }
    if not python.is_file() or not args.plan.is_file():
        raise SystemExit("environment Python or prepared plan is missing")
    if args.check_only:
        print(json.dumps(plan, indent=2))
        return
    if run_root.exists():
        raise SystemExit(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    write_json(run_root / "run_plan.json", plan)

    for repetition, order in enumerate(orders, 1):
        for cell in order:
            cell_root = run_root / f"r{repetition:02d}" / cell.label
            evidence = cell_root / "evidence"
            evidence.mkdir(parents=True)
            env = cell_environment(
                base, cell, gpus=args.gpus, port=args.port, cell_root=cell_root
            )
            command = runner_command(
                python,
                plan=args.plan,
                output=evidence,
                port=args.port,
                max_active_tasks=args.max_active_tasks,
                visit_capacity=args.visit_capacity,
                speculative_cap=args.speculative_cap,
                trace_limit=args.trace_limit,
            )
            write_json(
                cell_root / "cell_contract.json",
                {
                    "cell": asdict(cell),
                    "implementation_mapping": implementation_mapping(cell),
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
            aggregate(run_root, cells, args.repetitions)

    payload = aggregate(run_root, cells, args.repetitions)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
