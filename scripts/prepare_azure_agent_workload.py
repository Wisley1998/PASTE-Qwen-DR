#!/usr/bin/env python3
"""Prepare an Agent workload driven by Azure LLM 2024 arrival timestamps."""

from __future__ import annotations

import argparse
import json

from azure_llm_trace import apply_azure_arrivals, load_azure_llm_invocations
from trace_experiment_lib import load_workload, save_workload, summarize_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map Azure LLM Inference Trace 2024 timestamps to prepared Agent "
            "sessions without changing their messages, token budgets, or tool waits."
        )
    )
    parser.add_argument("--agent-workload", required=True)
    parser.add_argument("--azure-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--azure-dataset-variant",
        choices=["conversation", "code"],
        default="conversation",
    )
    parser.add_argument(
        "--azure-start-time",
        default=None,
        help="Inclusive ISO-8601 timestamp; offsets start at the first selected row.",
    )
    parser.add_argument("--azure-duration-s", type=float, default=None)
    parser.add_argument(
        "--azure-max-sessions",
        type=int,
        default=128,
        help="Safety limit on selected Azure rows (default: 128).",
    )
    parser.add_argument("--azure-arrival-speedup", type=float, default=1.0)
    parser.add_argument(
        "--azure-session-mapping",
        choices=["round_robin", "shuffled_round_robin"],
        default="round_robin",
    )
    parser.add_argument("--seed", type=int, default=20260417)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_workload = load_workload(args.agent_workload)
    invocations = load_azure_llm_invocations(
        args.azure_trace,
        start_time=args.azure_start_time,
        duration_s=args.azure_duration_s,
        max_sessions=args.azure_max_sessions,
    )
    workload = apply_azure_arrivals(
        base_workload,
        invocations,
        source_file=args.azure_trace,
        dataset_variant=args.azure_dataset_variant,
        arrival_speedup=args.azure_arrival_speedup,
        mapping=args.azure_session_mapping,
        mapping_seed=args.seed,
    )
    save_workload(workload, args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "arrival_process": workload["meta"]["arrival_process"],
                "workload": summarize_workload(workload),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
