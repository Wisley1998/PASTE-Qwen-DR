#!/usr/bin/env python3
"""Run trace-learned online speculative visits with a live Qwen-DR server.

This is a narrow convenience entry point around the general live experiment
driver.  It fixes the causal online contract to an autonomous call graph and
visit speculation learned from a checksummed trace artifact.  The LLM remains
the authoritative URL chooser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVE_RUNNER = (
    REPOSITORY_ROOT / "scripts" / "run_online_trace_learned_experiment.py"
)
DEFAULT_WORKLOAD = (
    REPOSITORY_ROOT / "reproduction" / "workloads" / "live_joint_wikipedia_tune_v1.json"
)
DEFAULT_MODEL_ARTIFACT = (
    REPOSITORY_ROOT
    / "reproduction"
    / "results"
    / "tool_only"
    / "url_rank_mapper.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument(
        "--prediction-model", type=Path, default=DEFAULT_MODEL_ARTIFACT
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--server-url", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    )
    parser.add_argument("--tokenizer")
    parser.add_argument("--source-limit", type=int)
    parser.add_argument(
        "--cell-label", default="online-trace-learned-speculative-tools"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the delegated live-runner command without contacting a server",
    )
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.source_limit is not None and args.source_limit <= 0:
        raise ValueError("--source-limit must be positive")
    for label, path in (
        ("live runner", LIVE_RUNNER),
        ("workload", args.workload),
        ("prediction model", args.prediction_model),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    command = [
        sys.executable,
        str(LIVE_RUNNER),
        "--workload",
        str(args.workload.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--server-url",
        args.server_url,
        "--model",
        args.model,
        "--cell-label",
        args.cell_label,
        "--call-graph-mode",
        "autonomous",
        "--speculation-mode",
        "visit",
        "--tool-signal-policy",
        "legacy",
        "--visit-top-k",
        str(args.top_k),
        "--visit-prediction-model",
        str(args.prediction_model.resolve()),
    ]
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.source_limit is not None:
        command.extend(["--source-limit", str(args.source_limit)])
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = build_command(args)
    if args.dry_run:
        print(json.dumps({"command": command}, ensure_ascii=False, indent=2))
        return 0
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
