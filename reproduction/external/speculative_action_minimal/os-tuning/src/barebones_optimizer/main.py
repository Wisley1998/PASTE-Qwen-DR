from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .benchmarks import SysbenchCpuBenchmark
from .config import ReplicationConfig
from .optimizer import ReplicationOptimizer
from .parameter_manager import SchedulerParameterManager
from .tuners import GeminiTuner


NOISY_LOGGERS = ("httpx", "google", "google_genai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speculative actions sysbench CPU use-case runner.")
    parser.add_argument("--config", required=True, help="Path to a use-case config JSON file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def build_run_dir(config: ReplicationConfig) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = config.make_run_dir()
    run_dir = run_root / f"{config.run_kind}_{config.mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_tuners(config: ReplicationConfig) -> dict[str, GeminiTuner]:
    tuners: dict[str, GeminiTuner] = {}
    if config.mode in {"actor_only", "dual"}:
        tuners["actor"] = GeminiTuner(
            role="actor",
            mode=config.mode,
            model_name=config.actor_model,
            min_granularity_range_ns=config.min_granularity_range_ns,
            latency_ns=config.latency_ns,
            api_key=config.api_key,
        )
    if config.mode in {"speculator_only", "dual"}:
        tuners["speculator"] = GeminiTuner(
            role="speculator",
            mode=config.mode,
            model_name=config.speculator_model,
            min_granularity_range_ns=config.min_granularity_range_ns,
            latency_ns=config.latency_ns,
            api_key=config.api_key,
        )
    return tuners


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(asctime)s %(name)s:%(lineno)d %(message)s",
    )
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(max(level, logging.WARNING))


def format_scheduler_access(summary: dict[str, str]) -> str:
    return ", ".join(f"{name}={mode}" for name, mode in summary.items())


def print_scheduler_access_warning(summary: dict[str, str]) -> None:
    if all(mode in {"direct", "sudo"} for mode in summary.values()):
        return
    print(
        "Warning: scheduler knobs are not fully writable. Re-run with `sudo -E` and ensure debugfs is mounted.",
        file=sys.stderr,
    )
    print(f"  scheduler access: {format_scheduler_access(summary)}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        config = ReplicationConfig.load(args.config)
        run_dir = build_run_dir(config)
        parameter_manager = SchedulerParameterManager()
        access_summary = parameter_manager.access_summary()
        print(f"Scheduler access: {format_scheduler_access(access_summary)}")
        if os.geteuid() != 0:
            print_scheduler_access_warning(access_summary)
        if any(mode in {"missing", "unavailable"} for mode in access_summary.values()):
            raise RuntimeError(
                "Scheduler knobs are not writable. Use `sudo -E`, mount debugfs, and ensure the kernel exposes the required CFS tunables."
            )
        benchmark = SysbenchCpuBenchmark(config, run_dir=run_dir)
        optimizer = ReplicationOptimizer(
            config=config,
            benchmark=benchmark,
            parameter_manager=parameter_manager,
            tuners=build_tuners(config),
            run_dir=run_dir,
        )
        result = optimizer.run()
    except Exception as exc:  # pragma: no cover - surfaced to caller
        logging.getLogger(__name__).error("Use-case run failed: %s", exc, exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
