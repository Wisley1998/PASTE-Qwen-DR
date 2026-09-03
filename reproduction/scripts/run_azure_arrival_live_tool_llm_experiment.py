#!/usr/bin/env python3
"""Apply external Agent release offsets to the frozen live experiment kernel."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION = ROOT / "reproduction"
if str(REPRODUCTION) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION))

from paste_repro.live_agent import LiveClosedLoopExperiment  # noqa: E402


def load_kernel() -> Any:
    path = ROOT / "scripts/run_live_tool_llm_experiment.py"
    spec = importlib.util.spec_from_file_location("paste_live_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load live experiment kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arrival-plan", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    plan = json.loads(known.arrival_plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "paste_repro.azure_llm_agent_arrivals.v1":
        raise ValueError("unsupported arrival plan schema")
    arrivals = plan.get("arrivals")
    if not isinstance(arrivals, list) or not arrivals:
        raise ValueError("arrival plan is empty")
    by_source: dict[str, dict[str, Any]] = {}
    for row in arrivals:
        source_id = row.get("source_id")
        offset = row.get("release_offset_s")
        if (
            not isinstance(source_id, str)
            or source_id in by_source
            or not isinstance(offset, (int, float))
            or not math.isfinite(float(offset))
            or float(offset) < 0
        ):
            raise ValueError("invalid or duplicate arrival row")
        by_source[source_id] = dict(row)

    original = LiveClosedLoopExperiment.run_task
    origin: float | None = None

    async def delayed_run_task(
        self: LiveClosedLoopExperiment,
        source: Any,
        *,
        replica: int = 0,
        visit_speculation_eligible: bool = True,
    ) -> dict[str, Any]:
        nonlocal origin
        row = by_source.get(source.source_id)
        if row is None:
            raise ValueError(f"missing arrival for source {source.source_id}")
        if origin is None:
            origin = time.monotonic()
        scheduled = origin + float(row["release_offset_s"])
        await asyncio.sleep(max(0.0, scheduled - time.monotonic()))
        released = time.monotonic()
        result = await original(
            self,
            source,
            replica=replica,
            visit_speculation_eligible=visit_speculation_eligible,
        )
        result["external_arrival"] = {
            **row,
            "release_lag_s": released - scheduled,
        }
        return result

    LiveClosedLoopExperiment.run_task = delayed_run_task
    kernel = load_kernel()
    sys.argv = [str(ROOT / "scripts/run_live_tool_llm_experiment.py"), *remaining]
    args = kernel.parse_args()
    if args.max_active_tasks != 0:
        raise ValueError("external arrival runs require --max-active-tasks 0")
    result = asyncio.run(kernel.async_main(args))
    output = Path(args.output_dir).resolve() / "result.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["external_arrival"] = {
        "plan": str(known.arrival_plan.resolve()),
        "plan_sha256": sha256_file(known.arrival_plan),
        "selection": plan["selection"],
        "source": plan["source"],
        "arrival_span_s": plan["arrival_span_s"],
        "task_gate": "disabled",
    }
    kernel._write_json_atomic(output, payload)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
