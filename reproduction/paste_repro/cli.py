"""Trace-analysis CLI for offline PASTE utilities."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from .mapper import save_artifact, write_json_atomic
from .pipeline import (
    default_trace_directory,
    run_speculative_tool_execution,
    run_tool_only_replay,
    train_and_analyze,
)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--traces",
        type=Path,
        default=default_trace_directory(),
        help="directory containing one JSONL file per session",
    )
    parser.add_argument("--seed", default="paste-repro-v1")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model-out",
        type=Path,
        help="optional path for the checksummed learned mapper JSON",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        help="optional path for an atomically written full JSON report",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m paste_repro.cli",
        description="Trace-analysis utilities (not the final-v9 live benchmark)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="train and evaluate on held-out traces")
    _common_arguments(analyze)

    replay = subparsers.add_parser(
        "run-tool-only", help="run the bounded scheduler over held-out trace calls"
    )
    _common_arguments(replay)
    replay.add_argument("--max-concurrency", type=int, default=4)
    replay.add_argument("--limit", type=int)
    replay.add_argument("--simulation-delay-ms", type=float, default=0.0)

    speculative_tools = subparsers.add_parser(
        "run-speculative-tools",
        help="run trace-learned speculative tool execution without LLM co-design",
    )
    _common_arguments(speculative_tools)
    speculative_tools.add_argument("--max-concurrency", type=int, default=4)
    speculative_tools.add_argument("--limit", type=int)
    speculative_tools.add_argument("--simulation-delay-ms", type=float, default=0.0)
    return parser


def _emit(
    payload: dict[str, Any], model_out: Path | None, report_out: Path | None
) -> None:
    if model_out is not None and report_out is not None:
        if model_out.resolve() == report_out.resolve():
            raise SystemExit("--model-out and --report-out must be different paths")
    if model_out is not None:
        save_artifact(model_out, payload["model_artifact"])
    if report_out is not None:
        write_json_atomic(report_out, payload)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        report, _, _, _, _ = train_and_analyze(
            args.traces,
            seed=args.seed,
            train_ratio=args.train_ratio,
            top_k=args.top_k,
        )
        _emit(report, args.model_out, args.report_out)
        return 0
    if args.command in {"run-tool-only", "run-speculative-tools"}:
        if args.simulation_delay_ms < 0:
            raise SystemExit("--simulation-delay-ms must be non-negative")
        runner = (
            run_speculative_tool_execution
            if args.command == "run-speculative-tools"
            else run_tool_only_replay
        )
        report = asyncio.run(
            runner(
                args.traces,
                seed=args.seed,
                train_ratio=args.train_ratio,
                top_k=args.top_k,
                max_concurrency=args.max_concurrency,
                limit=args.limit,
                simulation_delay_s=args.simulation_delay_ms / 1000.0,
            )
        )
        _emit(report, args.model_out, args.report_out)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
