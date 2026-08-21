#!/usr/bin/env python3
"""Change only decoding budgets in an already tokenized prepared workload."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
for import_root in (SCRIPTS_ROOT, REPRODUCTION_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from trace_experiment_lib import load_workload, summarize_workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget max_tokens without re-tokenizing prompts or changing trace waits."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--max-output-tokens-cap", type=int, required=True)
    parser.add_argument("--output-token-buffer", type=int, default=8)
    parser.add_argument("--min-output-tokens-floor", type=int, default=64)
    return parser.parse_args()


def retarget_workload(
    workload: dict[str, Any],
    *,
    max_output_tokens_cap: int,
    output_token_buffer: int,
    min_output_tokens_floor: int,
) -> dict[str, Any]:
    if max_output_tokens_cap <= 0:
        raise ValueError("max_output_tokens_cap must be positive")
    if output_token_buffer < 0:
        raise ValueError("output_token_buffer must be non-negative")
    if min_output_tokens_floor <= 0:
        raise ValueError("min_output_tokens_floor must be positive")
    if min_output_tokens_floor > max_output_tokens_cap:
        raise ValueError("min_output_tokens_floor cannot exceed the output cap")

    retargeted = copy.deepcopy(workload)
    meta = retargeted.setdefault("meta", {})
    max_model_len = int(meta.get("max_model_len", 0) or 0)
    request_count = 0
    context_clamped_requests = 0
    for trace in retargeted.get("traces", []):
        for request in trace.get("requests", []):
            target = max(
                1,
                int(request.get("target_output_tokens", request.get("max_tokens", 1))),
            )
            max_tokens = min(
                max_output_tokens_cap,
                max(min_output_tokens_floor, target + output_token_buffer),
            )
            prompt_tokens = int(request.get("prompt_tokens", 0) or 0)
            if max_model_len > 0 and prompt_tokens + max_tokens > max_model_len:
                available_output_tokens = max_model_len - prompt_tokens
                if available_output_tokens < min_output_tokens_floor:
                    raise ValueError(
                        "retargeted request cannot preserve the output floor: "
                        f"{prompt_tokens}+{min_output_tokens_floor}>{max_model_len}"
                    )
                max_tokens = available_output_tokens
                context_clamped_requests += 1
            request["max_tokens"] = max_tokens
            request_count += 1
    if request_count == 0:
        raise ValueError("prepared workload has no requests")

    meta["max_output_tokens_cap"] = max_output_tokens_cap
    meta["output_token_buffer"] = output_token_buffer
    meta["min_output_tokens_floor"] = min_output_tokens_floor
    meta["retargeted_from_max_output_tokens_cap"] = workload.get("meta", {}).get(
        "max_output_tokens_cap"
    )
    meta["retargeted_context_clamped_requests"] = context_clamped_requests
    return retargeted


def main() -> int:
    args = parse_args()
    workload = load_workload(args.input)
    retargeted = retarget_workload(
        workload,
        max_output_tokens_cap=args.max_output_tokens_cap,
        output_token_buffer=args.output_token_buffer,
        min_output_tokens_floor=args.min_output_tokens_floor,
    )
    write_json_atomic(args.output, retargeted)
    summary_path = args.summary_output or args.output.with_name("workload_summary.json")
    write_json_atomic(summary_path, summarize_workload(retargeted))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "summary": str(summary_path.resolve()),
                "request_count": sum(
                    len(trace.get("requests", []))
                    for trace in retargeted.get("traces", [])
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
