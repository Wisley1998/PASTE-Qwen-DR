#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from barebones_optimizer.benchmarks import SysbenchCpuBenchmark
from barebones_optimizer.config import ReplicationConfig
from barebones_optimizer.parameter_manager import SchedulerParameterManager


DEFAULT_CANDIDATES_NS = [
    50_000,
    100_000,
    150_000,
    200_000,
    300_000,
    500_000,
    1_000_000,
    3_000_000,
    5_000_000,
    10_000_000,
    30_000_000,
    50_000_000,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find a deliberately poor min_granularity for the reaction experiment.")
    parser.add_argument("--config-template", required=True, help="Reaction config JSON to borrow sysbench settings from.")
    parser.add_argument("--output", required=True, help="Path to the calibration JSON output.")
    parser.add_argument("--duration-s", type=int, default=3, help="Seconds per probe benchmark.")
    parser.add_argument(
        "--target-p95-ms",
        type=float,
        default=102.97,
        help="Target untuned p95 from the paper; the closest candidate is selected.",
    )
    return parser.parse_args()


def load_config(template_path: Path) -> ReplicationConfig:
    raw = json.loads(template_path.read_text(encoding="utf-8"))
    raw["gemini_api_key"] = raw.get("gemini_api_key") or "unused"
    raw["results_dir"] = "results/reaction_calibration"
    raw["run_duration_s"] = max(1, int(raw.get("window_duration_s", 1)))
    return ReplicationConfig(**raw)


def choose_candidate(results: list[dict], target_p95_ms: float) -> dict:
    degraded = [item for item in results if item["latency_p95"] >= target_p95_ms * 0.8]
    pool = degraded or results
    return min(pool, key=lambda item: (abs(item["latency_p95"] - target_p95_ms), -item["latency_p95"]))


def main() -> int:
    args = parse_args()
    template_path = Path(args.config_template).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(template_path)
    benchmark = SysbenchCpuBenchmark(config, run_dir=output_path.parent / "reaction_calibration")
    manager = SchedulerParameterManager()
    defaults = manager.snapshot()

    results: list[dict] = []
    try:
        for iteration, candidate in enumerate(DEFAULT_CANDIDATES_NS, start=1):
            manager.apply(latency_ns=config.latency_ns, min_granularity_ns=candidate)
            execution = benchmark.execute_window(iteration, args.duration_s)
            results.append(
                {
                    "min_granularity_ns": candidate,
                    "latency_p95": execution.benchmark_metrics.latency_p95,
                    "latency_avg": execution.benchmark_metrics.latency_avg,
                    "throughput": execution.benchmark_metrics.throughput,
                }
            )
    finally:
        manager.restore(defaults)

    selected = choose_candidate(results, args.target_p95_ms)
    payload = {
        "latency_ns": config.latency_ns,
        "defaults": defaults,
        "probe_duration_s": args.duration_s,
        "target_p95_ms": args.target_p95_ms,
        "selected_min_granularity_ns": selected["min_granularity_ns"],
        "selected_latency_p95_ms": selected["latency_p95"],
        "results": sorted(results, key=lambda item: item["latency_p95"], reverse=True),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Reaction calibration")
    print(f"  selected min_granularity_ns: {selected['min_granularity_ns']:,}")
    print(f"  observed degraded p95: {selected['latency_p95']:.2f} ms")
    print(f"  output: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
