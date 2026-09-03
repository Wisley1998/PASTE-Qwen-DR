#!/usr/bin/env python3
"""Replay the agent-baseline boundary on held-out PASTE traces."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.baseline_boundary import (  # noqa: E402
    DEFAULT_SPEEDUPS,
    run_agent_baseline_boundary,
)


DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT / "results" / "agent_baseline_boundary" / "replay.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--seed", default="paste-repro-v1")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--inference-speedups",
        type=float,
        nargs="+",
        default=DEFAULT_SPEEDUPS,
        metavar="FACTOR",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = asyncio.run(
        run_agent_baseline_boundary(
            args.trace_directory,
            seed=args.seed,
            train_ratio=args.train_ratio,
            top_k=args.top_k,
            max_concurrency=args.max_concurrency,
            speedups=args.inference_speedups,
        )
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

