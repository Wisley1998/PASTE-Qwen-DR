#!/usr/bin/env python3
"""Collect fresh, whole-session Tongyi DeepResearch traces."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.live_executor import WikipediaLiveExecutor  # noqa: E402
from paste_repro.multiturn_collector import (  # noqa: E402
    CollectorConfig,
    OpenAICompatibleChatClient,
    collect_fixed_workload,
    load_fixed_workload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one legacy-compatible JSONL trace per source using an "
            "OpenAI-compatible vLLM endpoint and keyless Wikipedia tools."
        )
    )
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--search-mode", choices=("rest", "action", "bing"), default="rest")
    parser.add_argument("--visit-mode", choices=("direct", "jina"), default="direct")
    parser.add_argument("--tool-timeout-s", type=float, default=20.0)
    parser.add_argument("--max-search-results", type=int, default=5)
    parser.add_argument("--max-visit-urls", type=int, default=6)
    parser.add_argument("--max-http-attempts", type=int, default=1)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--search-min-start-interval-s", type=float, default=0.0)
    parser.add_argument("--visit-min-start-interval-s", type=float, default=0.0)
    parser.add_argument(
        "--api-key-env",
        default="VLLM_API_KEY",
        help="Environment variable holding an optional API key; its value is never persisted.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and fingerprint the workload without contacting any endpoint.",
    )
    return parser


def _config(args: argparse.Namespace) -> CollectorConfig:
    return CollectorConfig(
        endpoint=args.endpoint,
        model=args.model,
        max_calls=args.max_calls,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        request_timeout_s=args.request_timeout_s,
        search_mode=args.search_mode,
        visit_mode=args.visit_mode,
        tool_timeout_s=args.tool_timeout_s,
        max_search_results=args.max_search_results,
        max_visit_urls=args.max_visit_urls,
        max_http_attempts=args.max_http_attempts,
        retry_backoff_s=args.retry_backoff_s,
        search_min_start_interval_s=args.search_min_start_interval_s,
        visit_min_start_interval_s=args.visit_min_start_interval_s,
    )


async def _run(args: argparse.Namespace) -> dict[str, object]:
    config = _config(args)
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    attempt_intervals = {
        tool_name: interval_s
        for tool_name, interval_s in (
            ("search", config.search_min_start_interval_s),
            ("visit", config.visit_min_start_interval_s),
        )
        if interval_s > 0
    }
    async with OpenAICompatibleChatClient(
        config.endpoint, timeout_s=config.request_timeout_s, api_key=api_key
    ) as client:
        async with WikipediaLiveExecutor(
            visit_mode=config.visit_mode,
            timeout_s=config.tool_timeout_s,
            max_results=config.max_search_results,
            max_visit_urls=config.max_visit_urls,
            search_mode=config.search_mode,
            max_http_attempts=config.max_http_attempts,
            retry_backoff_s=config.retry_backoff_s,
            http_attempt_min_start_intervals_s=attempt_intervals,
        ) as executor:
            return await collect_fixed_workload(
                workload_path=args.workload,
                output_dir=args.output_dir,
                config=config,
                client=client,
                executor=executor,
                authentication_configured=bool(api_key),
                collector_cli_source_path=Path(__file__),
                live_executor_source_path=(
                    REPRODUCTION_ROOT / "paste_repro" / "live_executor.py"
                ),
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args)
    if args.validate_only:
        workload = load_fixed_workload(args.workload)
        print(
            json.dumps(
                {
                    "workload_id": workload.workload_id,
                    "file_sha256": workload.file_sha256,
                    "source_count": len(workload.sources),
                    "config": config.to_manifest(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "collection_status": manifest["collection_status"],
                "summary": manifest["summary"],
                "manifest": str(args.output_dir / "manifest.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest["collection_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
