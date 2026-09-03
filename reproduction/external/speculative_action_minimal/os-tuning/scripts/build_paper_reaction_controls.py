#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from barebones_optimizer.benchmarks import SysbenchCpuBenchmark
from barebones_optimizer.config import ReplicationConfig
from barebones_optimizer.parameter_manager import SchedulerParameterManager
from barebones_optimizer.reaction import ReactionReplayRunner, actor_replay_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-faithful reaction controls from a dual-loop history.")
    parser.add_argument("--dual-history", required=True, help="Path to the dual reaction history.json file.")
    parser.add_argument("--output-root", required=True, help="Directory where control histories should be written.")
    return parser.parse_args()


def timestamped_run_dir(root: Path, mode: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = root / mode / f"reaction_{mode}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def config_from_history(history_data: dict) -> ReplicationConfig:
    config_data = dict(history_data["config"])
    config_data["gemini_api_key"] = config_data.get("gemini_api_key") or "unused"
    config_data["results_dir"] = str(Path(config_data["results_dir"]))
    return ReplicationConfig(**config_data)


def main() -> int:
    args = parse_args()
    dual_history_path = Path(args.dual_history).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with dual_history_path.open("r", encoding="utf-8") as handle:
        dual_history = json.load(handle)

    config = config_from_history(dual_history)
    parameter_manager = SchedulerParameterManager()

    actor_run_dir = timestamped_run_dir(output_root, "actor_only")
    actor_runner = ReactionReplayRunner(
        config=config,
        benchmark=SysbenchCpuBenchmark(config, run_dir=actor_run_dir),
        parameter_manager=parameter_manager,
        run_dir=actor_run_dir,
        setup_mode="actor_only",
        request_type="actor",
        model_name=(dual_history.get("models") or {}).get("actor"),
        scheduled_events=actor_replay_schedule(dual_history),
        source_history=str(dual_history_path),
    )
    actor_result = actor_runner.run()

    untuned_run_dir = timestamped_run_dir(output_root, "untuned")
    untuned_runner = ReactionReplayRunner(
        config=config,
        benchmark=SysbenchCpuBenchmark(config, run_dir=untuned_run_dir),
        parameter_manager=parameter_manager,
        run_dir=untuned_run_dir,
        setup_mode="untuned",
        request_type=None,
        model_name=None,
        scheduled_events=[],
        source_history=str(dual_history_path),
    )
    untuned_result = untuned_runner.run()

    print("Paper reaction controls")
    print(f"  actor replay history: {actor_result['history_file']}")
    print(f"  untuned history: {untuned_result['history_file']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
