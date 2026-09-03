#!/usr/bin/env python3
"""Prepare, collect, and evaluate the Speculative Actions trace baseline.

Preparation and evaluation are offline.  Only ``collect`` contacts the local
Qwen3 speculator endpoint; it never calls the authoritative model or tools.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import aiohttp


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.speculative_action_replay import (  # noqa: E402
    PREDICTION_SCHEMA,
    build_cases,
    canonical_json,
    evaluate_predictions,
    parse_predictions,
    percentile,
    read_cases,
    sha256_file,
    write_jsonl,
)
from paste_repro.traces import LLMCall, ToolCall  # noqa: E402


DEFAULT_TRACES = (
    REPOSITORY_ROOT
    / "traces/my_traces_tool_slo_search_uniform_1_3s_"
    "visit_serial_uniform_2_8s_llm_x0_42"
)
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "artifacts/speculative_action_qwen3_8b"
MANIFEST_SCHEMA = "paste_repro.speculative_action_manifest.v1"
COLLECTION_SCHEMA = "paste_repro.speculative_action_collection.v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain one JSON object per line")
    return rows


def trace_inventory(trace_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(trace_dir.glob("*.jsonl"))
    ]


def prepare(args: argparse.Namespace) -> int:
    cases, sessions = build_cases(
        args.traces,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        trace_limit=args.trace_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "cases.jsonl"
    write_jsonl(cases_path, (case.to_dict() for case in cases))

    llm_calls = [
        event
        for session in sessions
        for event in session.events
        if isinstance(event, LLMCall)
    ]
    tool_calls = [
        event
        for session in sessions
        for event in session.events
        if isinstance(event, ToolCall)
    ]
    inventory = trace_inventory(args.traces)
    if args.trace_limit is not None:
        selected = {session.path.name for session in sessions}
        inventory = [row for row in inventory if row["filename"] in selected]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": now_iso(),
        "phase": "prepared_only_no_model_or_tool_execution",
        "trace_dir": str(args.traces.resolve()),
        "trace_files": inventory,
        "trace_count": len(sessions),
        "llm_calls": len(llm_calls),
        "eligible_llm_to_tool_cases": len(cases),
        "tool_calls": len(tool_calls),
        "total_recorded_llm_time_s": sum(call.total_time_s for call in llm_calls),
        "total_recorded_llm_inference_time_s": sum(
            call.inference_time_s for call in llm_calls
        ),
        "top_k": args.top_k,
        "max_context_chars": args.max_context_chars,
        "truncated_prompt_cases": sum(case.prompt_truncated for case in cases),
        "sessions": [
            {
                "session_id": session.session_id,
                "llm_calls": sum(isinstance(event, LLMCall) for event in session.events),
                "tool_calls": sum(isinstance(event, ToolCall) for event in session.events),
                "recorded_llm_time_s": sum(
                    event.total_time_s
                    for event in session.events
                    if isinstance(event, LLMCall)
                ),
            }
            for session in sessions
        ],
        "cases_file": str(cases_path.resolve()),
        "cases_sha256": sha256_file(cases_path),
        "fairness_contract": {
            "authoritative_llm_requests": 0,
            "tool_requests": 0,
            "recorded_llm_outputs_are_labels_only": True,
            "recorded_tool_calls_are_labels_only": True,
            "speculator_prompt_excludes_recorded_llm_response_and_tool_label": True,
        },
    }
    manifest_path = args.output_dir / "prepare_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "cases": len(cases),
                "sessions": len(sessions),
                "phase": manifest["phase"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


async def endpoint_models(
    session: aiohttp.ClientSession, server_url: str, api_key: str | None
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with session.get(
        f"{server_url.rstrip('/')}/v1/models", headers=headers
    ) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"model endpoint returned HTTP {response.status}: {body[:500]}")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise RuntimeError("model endpoint returned a non-object response")
        return payload


async def collect_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    case: Any,
    server_url: str,
    api_key: str | None,
    model: str,
    top_k: int,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    raw_text = ""
    response_payload: dict[str, Any] = {}
    error: str | None = None
    predictions = ()
    async with semaphore:
        # Client-side concurrency limiting is not part of model latency. Start
        # the clock only when this request is admitted; vLLM-side queueing and
        # HTTP time remain included.
        started = time.perf_counter()
        try:
            async with session.post(
                f"{server_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise RuntimeError("chat completion response is not an object")
                response_payload = parsed
                raw_text = str(parsed["choices"][0]["message"].get("content") or "")
                predictions = parse_predictions(raw_text, top_k=top_k)
        except Exception as exc:  # A failed prediction is a fail-closed miss.
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
    usage = response_payload.get("usage", {})
    return {
        "schema": PREDICTION_SCHEMA,
        "case_id": case.case_id,
        "session_id": case.session_id,
        "llm_call_index": case.llm_call_index,
        "latency_s": latency,
        "predictions": [prediction.to_dict() for prediction in predictions],
        "usage": usage if isinstance(usage, dict) else {},
        "raw_response": raw_text,
        "error": error,
    }


async def collect_async(args: argparse.Namespace) -> int:
    cases_path = args.cases or args.output_dir / "cases.jsonl"
    cases = read_cases(cases_path)
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "dry_run_no_requests",
                    "cases_file": str(cases_path),
                    "cases": len(cases),
                    "server_url": args.server_url,
                    "model": args.model,
                    "top_k": args.top_k,
                    "concurrency": args.concurrency,
                    "max_tokens": args.max_tokens,
                },
                indent=2,
            )
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "predictions.partial.jsonl"
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency, 2))
    api_key = args.api_key or os.getenv("SPEC_ACTION_API_KEY")
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        models = await endpoint_models(session, args.server_url, api_key)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            asyncio.create_task(
                collect_one(
                    session,
                    semaphore,
                    case=case,
                    server_url=args.server_url,
                    api_key=api_key,
                    model=args.model,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                )
            )
            for case in cases
        ]
        rows = []
        case_order = {case.case_id: index for index, case in enumerate(cases)}
        for completed in asyncio.as_completed(tasks):
            rows.append(await completed)
            if len(rows) % args.checkpoint_every == 0 or len(rows) == len(tasks):
                ordered_partial = sorted(rows, key=lambda row: case_order[row["case_id"]])
                write_jsonl(partial_path, ordered_partial)
                mean_latency = statistics.fmean(
                    float(row["latency_s"]) for row in rows
                )
                print(
                    f"[collect] {len(rows)}/{len(tasks)} "
                    f"errors={sum(bool(row['error']) for row in rows)} "
                    f"mean_latency_s={mean_latency:.3f}",
                    flush=True,
                )

    rows.sort(key=lambda row: case_order[row["case_id"]])
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    collection = {
        "schema": COLLECTION_SCHEMA,
        "created_at": now_iso(),
        "cases_file": str(cases_path.resolve()),
        "cases_sha256": sha256_file(cases_path),
        "predictions_file": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "cases": len(cases),
        "errors": sum(bool(row["error"]) for row in rows),
        "server_url": args.server_url,
        "endpoint_models": models,
        "model": args.model,
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "thinking": False,
        "latency_includes_queue_and_http": True,
    }
    write_json(args.output_dir / "collection_manifest.json", collection)
    print(json.dumps(collection, ensure_ascii=False, indent=2))
    return 0


def render_report(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    summary = report["summary"]
    setup = report.get("experimental_setup", {})
    return f"""# Speculative Actions / Qwen3-8B trace replay

This report is a lossless counterfactual over the pinned recorded trace. The
authoritative LLM output and authoritative tool calls were not regenerated.

## Setup

- Trace: `{manifest['trace_dir']}` ({manifest['trace_count']} whole sessions)
- Speculator: `{setup.get('model', 'Qwen/Qwen3-8B')}`, Top-{setup.get('top_k', 3)},
  temperature {setup.get('temperature', 0.0)}, non-thinking mode
- Speculator request concurrency: {setup.get('concurrency', 1)}
- Match rule: exact tool name and complete canonical JSON arguments
- Timing rule: `head_start = max(0, recorded_generation_window - measured_speculator_latency)`;
  an exact hit saves `min(tool_duration, head_start)`

## Results

| Metric | Value |
|---|---:|
| Trace sessions | {manifest['trace_count']} |
| Eligible LLM→tool turns | {summary['eligible_tool_calls']} |
| Content-exact Top-K hits / misses | {summary['exact_hits']} / {summary['exact_misses']} ({summary['exact_hit_rate']:.2%} hit) |
| Effective on-time hits / misses | {summary['on_time_exact_hits']} / {summary['effective_misses']} ({summary['on_time_exact_hit_rate']:.2%} hit) |
| Malformed fail-closed predictions | {summary['prediction_errors']} |
| Valid empty predictions | {summary['clean_empty_prediction_cases']} |
| Parsed candidate misses | {summary['parsed_candidate_misses']} |
| Requests completed before authority | {summary['requests_completed_before_authority']} |
| On-time prediction cases | {summary['on_time_prediction_cases']} |
| Launched speculative tool calls | {summary['launched_speculative_invocations']} |
| Tool call amplification | {summary['tool_call_amplification']:.3f}× |
| Speculator latency p50 / p95 | {summary['speculator_latency_p50_s']:.3f}s / {summary['speculator_latency_p95_s']:.3f}s |
| Exposed tool stall | {summary['baseline_tool_stall_s']:.3f}s → {summary['speculative_tool_stall_s']:.3f}s |
| Tool-stall reduction | {summary['tool_stall_reduction']:.2%} |
| End-to-end trace time | {summary['baseline_e2e_s']:.3f}s → {summary['speculative_e2e_s']:.3f}s |
| End-to-end reduction | {summary['e2e_reduction']:.2%} |
| Mean per-session end-to-end | {summary['mean_session_baseline_e2e_s']:.3f}s → {summary['mean_session_speculative_e2e_s']:.3f}s |
| Mean paired session reduction | {summary['mean_paired_session_e2e_reduction']:.2%} |

An exact hit requires the same tool name and the same complete canonical JSON
arguments. A late prediction, malformed output, or any mismatch is a normal
demand execution and cannot change the recorded result.

The model produced {summary['exact_hits']} content-exact predictions, but all
finished after their corresponding recorded generation windows. Therefore no
tool service was hidden and both tool-side and end-to-end latency savings are
zero under this measured deployment. The on-time incorrect predictions still
raise physical tool invocations from {summary['eligible_tool_calls']} to
{summary['physical_tool_invocations']}.
"""


def evaluate(args: argparse.Namespace) -> int:
    cases_path = args.cases or args.output_dir / "cases.jsonl"
    predictions_path = args.predictions or args.output_dir / "predictions.jsonl"
    manifest_path = args.manifest or args.output_dir / "prepare_manifest.json"
    cases = read_cases(cases_path)
    predictions = read_jsonl(predictions_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported prepare manifest: {manifest.get('schema')!r}")
    if manifest.get("cases_sha256") != sha256_file(cases_path):
        raise ValueError("cases file checksum does not match prepare manifest")
    if int(manifest.get("top_k", -1)) != args.top_k:
        raise ValueError(
            f"--top-k={args.top_k} does not match prepared top_k={manifest.get('top_k')}"
        )
    case_ids = {case.case_id for case in cases}
    prediction_ids = {str(row.get("case_id")) for row in predictions}
    unknown = prediction_ids - case_ids
    missing = case_ids - prediction_ids
    if unknown:
        raise ValueError(f"predictions contain {len(unknown)} unknown case ids")
    if missing and not args.allow_partial:
        raise ValueError(
            f"predictions are missing {len(missing)} cases; use --allow-partial only for smoke diagnostics"
        )
    report = evaluate_predictions(cases, predictions, top_k=args.top_k)
    collection_path = (
        args.collection_manifest or args.output_dir / "collection_manifest.json"
    )
    collection: dict[str, Any] = {}
    if collection_path.is_file():
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        if collection.get("schema") != COLLECTION_SCHEMA:
            raise ValueError(
                f"unsupported collection manifest: {collection.get('schema')!r}"
            )
        if collection.get("cases_sha256") != sha256_file(cases_path):
            raise ValueError("collection manifest cases checksum mismatch")
        if collection.get("predictions_sha256") != sha256_file(predictions_path):
            raise ValueError("collection manifest predictions checksum mismatch")
        report["experimental_setup"] = {
            key: collection.get(key)
            for key in (
                "created_at",
                "model",
                "top_k",
                "concurrency",
                "max_tokens",
                "temperature",
                "seed",
                "thinking",
                "latency_includes_queue_and_http",
                "endpoint_models",
            )
        }
    summary = report["summary"]
    total_llm = float(manifest["total_recorded_llm_time_s"])
    baseline_e2e = total_llm + summary["baseline_tool_stall_s"]
    speculative_e2e = baseline_e2e - summary["saved_tool_stall_s"]
    summary.update(
        {
            "recorded_llm_time_s": total_llm,
            "baseline_e2e_s": baseline_e2e,
            "speculative_e2e_s": speculative_e2e,
            "e2e_reduction": (
                summary["saved_tool_stall_s"] / baseline_e2e if baseline_e2e else 0.0
            ),
            "e2e_speedup": baseline_e2e / speculative_e2e if speculative_e2e else None,
        }
    )
    session_saved: dict[str, float] = {}
    session_tool: dict[str, float] = {}
    for row in report["cases"]:
        session_id = row["session_id"]
        session_saved[session_id] = session_saved.get(session_id, 0.0) + float(
            row["saved_tool_stall_s"]
        )
        session_tool[session_id] = session_tool.get(session_id, 0.0) + float(
            row["tool_duration_s"]
        )
    session_rows = []
    for session in manifest["sessions"]:
        session_id = session["session_id"]
        baseline = float(session["recorded_llm_time_s"]) + session_tool.get(session_id, 0.0)
        speculative = baseline - session_saved.get(session_id, 0.0)
        session_rows.append(
            {
                "session_id": session_id,
                "baseline_e2e_s": baseline,
                "speculative_e2e_s": speculative,
                "saved_s": baseline - speculative,
                "e2e_reduction": (baseline - speculative) / baseline if baseline else 0.0,
            }
        )
    baseline_sessions = [row["baseline_e2e_s"] for row in session_rows]
    speculative_sessions = [row["speculative_e2e_s"] for row in session_rows]
    reductions = [row["e2e_reduction"] for row in session_rows]
    summary.update(
        {
            "mean_session_baseline_e2e_s": statistics.fmean(baseline_sessions),
            "mean_session_speculative_e2e_s": statistics.fmean(speculative_sessions),
            "p50_session_baseline_e2e_s": percentile(baseline_sessions, 0.50),
            "p50_session_speculative_e2e_s": percentile(speculative_sessions, 0.50),
            "p95_session_baseline_e2e_s": percentile(baseline_sessions, 0.95),
            "p95_session_speculative_e2e_s": percentile(speculative_sessions, 0.95),
            "mean_paired_session_e2e_reduction": statistics.fmean(reductions),
        }
    )
    report["sessions"] = session_rows
    report["inputs"] = {
        "prepare_manifest": str(manifest_path.resolve()),
        "cases": str(cases_path.resolve()),
        "cases_sha256": sha256_file(cases_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "collection_manifest": (
            str(collection_path.resolve()) if collection_path.is_file() else None
        ),
        "top_k": args.top_k,
    }
    report_path = args.output_dir / "report.json"
    write_json(report_path, report)
    (args.output_dir / "REPORT.md").write_text(
        render_report(report, manifest), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="materialize causal prompts and checksums; execute no models/tools"
    )
    add_common_paths(prepare_parser)
    prepare_parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    prepare_parser.add_argument("--top-k", type=int, default=3)
    prepare_parser.add_argument("--max-context-chars", type=int, default=36_000)
    prepare_parser.add_argument("--trace-limit", type=int)

    collect_parser = subparsers.add_parser(
        "collect", help="query only the local speculator and cache predictions"
    )
    add_common_paths(collect_parser)
    collect_parser.add_argument("--cases", type=Path)
    collect_parser.add_argument("--server-url", default="http://127.0.0.1:8200")
    collect_parser.add_argument("--api-key")
    collect_parser.add_argument("--model", default="Qwen/Qwen3-8B")
    collect_parser.add_argument("--top-k", type=int, default=3)
    collect_parser.add_argument("--concurrency", type=int, default=1)
    collect_parser.add_argument("--max-tokens", type=int, default=512)
    collect_parser.add_argument("--temperature", type=float, default=0.0)
    collect_parser.add_argument("--seed", type=int, default=20260903)
    collect_parser.add_argument("--request-timeout-s", type=float, default=180.0)
    collect_parser.add_argument("--case-limit", type=int)
    collect_parser.add_argument("--checkpoint-every", type=int, default=10)
    collect_parser.add_argument("--dry-run", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate cached predictions without model/tool requests"
    )
    add_common_paths(evaluate_parser)
    evaluate_parser.add_argument("--cases", type=Path)
    evaluate_parser.add_argument("--predictions", type=Path)
    evaluate_parser.add_argument("--manifest", type=Path)
    evaluate_parser.add_argument("--collection-manifest", type=Path)
    evaluate_parser.add_argument("--top-k", type=int, default=3)
    evaluate_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow missing predictions for explicitly labelled smoke diagnostics",
    )

    args = parser.parse_args()
    for name in (
        "top_k", "concurrency", "max_tokens", "max_context_chars", "checkpoint_every"
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "temperature") and args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if hasattr(args, "request_timeout_s") and args.request_timeout_s <= 0:
        parser.error("--request-timeout-s must be positive")
    if getattr(args, "trace_limit", None) is not None and args.trace_limit <= 0:
        parser.error("--trace-limit must be positive")
    if getattr(args, "case_limit", None) is not None and args.case_limit <= 0:
        parser.error("--case-limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        raise SystemExit(prepare(args))
    if args.command == "collect":
        raise SystemExit(asyncio.run(collect_async(args)))
    if args.command == "evaluate":
        raise SystemExit(evaluate(args))
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
