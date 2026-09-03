#!/usr/bin/env python3
"""Run live Qwen-DR with trace-learned online speculative tool execution.

This is separate from the frozen formal-v9 driver so adding the learned model
does not change historical formal runtime hashes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_ROOT = REPO_ROOT / "reproduction"
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.online_learned_agent import (  # noqa: E402
    ApproximateTokenCounter,
    FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
    FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
    FINAL_ANSWER_SCHEMA_POLICY_VERSION,
    FIXED_FINAL_ANSWER_CONTRACT_POLICY_VERSION,
    FIXED_OUTPUT_CONTRACT_POLICY_VERSION,
    LiveClosedLoopExperiment,
    LiveLLMClient,
    TOOL_SIGNAL_POLICY_VERSION,
    TransformersTokenCounter,
    sha256_json,
    summarize_live_run,
    task_to_dict,
    validate_sources,
)
from paste_repro.live_broker import LiveToolBroker  # noqa: E402
from paste_repro.live_executor import WikipediaLiveExecutor  # noqa: E402
from paste_repro.tool_prediction import load_visit_predictor  # noqa: E402


SCHEDULER_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MODEL_ID",
    "MODEL_REVISION",
    "VLLM_PORT",
    "VLLM_TP_SIZE",
    "VLLM_DTYPE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_CUDA_GRAPH_SIZES",
    "VLLM_ENABLE_PREFIX_CACHING",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
    "VLLM_USE_V1",
    "VLLM_SCHED_POLICY",
    "VLLM_SCHED_PRED_OUT_ENABLE",
    "VLLM_SCHED_PRED_OUT_EMA_ALPHA",
    "VLLM_SCHED_DEFAULT_PRED_OUT",
    "VLLM_SCHED_AVG_CALL_SERVICE_S",
    "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2",
    "VLLM_SCHED_DECODE_TOKENS_PER_S_V2",
    "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S",
    "VLLM_SCHED_TIME_AGING_ALPHA",
    "VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS",
    "VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING",
    "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
    "VLLM_SCHED_JOINT_V2_FINAL_LANE",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
    "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING",
    "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING",
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S",
    "VLLM_SCHED_JOINT_V2_TAIL_BETA",
    "VLLM_SCHED_JOINT_V2_TOOL_BETA",
    "VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S",
    "VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT",
    "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA",
    "VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS",
    "VLLM_SCHED_JOINT_V2_FINAL_BONUS_S",
    "VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S",
    "VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S",
    "VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_WEIGHT",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_REFRESH_S",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_LOG_INTERVAL_S",
    "VLLM_SCHED_HBM_MIN_RUNNING_REQS",
    "VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP",
    "VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS",
    "VLLM_SCHED_HBM_MAX_LONG_RUNNING",
    "VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS",
    "VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS",
    "VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS",
    "VLLM_SCHED_HBM_LOW_PRESSURE",
    "VLLM_SCHED_HBM_HIGH_PRESSURE",
    "VLLM_SCHED_HBM_BUDGET_INCREASE",
    "VLLM_SCHED_HBM_BUDGET_DECREASE",
    "VLLM_SCHED_HBM_CONTROL_INTERVAL_S",
    "VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO",
)


def _normalize_tool_attempt_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return serialization-ready physical telemetry without inventing HTTP work.

    The only nullable ``http_attempts`` value that can be derived is an
    admitted job cancelled before dispatch.  Its timestamps already prove
    that the whole lifetime was queueing and that no worker/transport existed.
    Any ambiguous row fails closed.  Started jobs retain—and must provide—the
    executor's positive attempt count.
    """

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"tool record {index} is not an object")
        row = dict(raw)
        if row.get("admitted") is not True:
            normalized.append(row)
            continue

        started = row.get("started_at")
        start_alias = row.get("start")
        attempts = row.get("http_attempts")
        if started is not None:
            if (
                isinstance(started, bool)
                or not isinstance(started, (int, float))
                or not math.isfinite(float(started))
                or isinstance(start_alias, bool)
                or not isinstance(start_alias, (int, float))
                or not math.isclose(
                    float(start_alias), float(started), rel_tol=0.0, abs_tol=1e-9
                )
            ):
                raise RuntimeError(f"tool record {index} has inconsistent start telemetry")
            if (
                isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 1
            ):
                raise RuntimeError(
                    f"started tool record {index} lacks positive HTTP attempts"
                )
            normalized.append(row)
            continue

        if start_alias is not None:
            raise RuntimeError(f"tool record {index} has inconsistent start telemetry")
        if (
            row.get("cancelled") is not True
            or row.get("outcome") not in {"cancelled", "expired"}
            or row.get("committed") is not False
            or row.get("worker_id") is not None
        ):
            raise RuntimeError(
                f"tool record {index} is not a valid never-started cancellation"
            )
        if any(
            row.get(field) is not None
            for field in (
                "backend",
                "request_host",
                "response_status",
                "bytes_read",
                "transport_identity_source",
            )
        ):
            raise RuntimeError(
                f"never-started tool record {index} claims transport evidence"
            )
        if isinstance(attempts, bool) or attempts not in {None, 0}:
            raise RuntimeError(
                f"never-started tool record {index} claims HTTP attempts"
            )
        queued = row.get("queue_enter_at")
        finish = row.get("finished_at")
        queue_alias = row.get("queue_enter")
        finish_alias = row.get("finish")
        timestamps = (queued, finish, queue_alias, finish_alias)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in timestamps
        ):
            raise RuntimeError(
                f"never-started tool record {index} lacks queue/finish timestamps"
            )
        assert queued is not None and finish is not None
        assert queue_alias is not None and finish_alias is not None
        if (
            float(finish) < float(queued)
            or not math.isclose(
                float(queue_alias), float(queued), rel_tol=0.0, abs_tol=1e-9
            )
            or not math.isclose(
                float(finish_alias), float(finish), rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise RuntimeError(
                f"never-started tool record {index} has inconsistent timestamps"
            )
        expected_queue_s = float(finish) - float(queued)
        observed_queue_s = row.get("queue_s")
        if observed_queue_s is not None and (
            isinstance(observed_queue_s, bool)
            or not isinstance(observed_queue_s, (int, float))
            or not math.isclose(
                float(observed_queue_s),
                expected_queue_s,
                rel_tol=0.02,
                abs_tol=0.01,
            )
        ):
            raise RuntimeError(
                f"never-started tool record {index} has inconsistent queue duration"
            )
        for field in ("service_s", "saved_service_s"):
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) != 0.0
            ):
                raise RuntimeError(
                    f"never-started tool record {index} has non-zero {field}"
                )
        row["http_attempts"] = 0
        row["queue_s"] = expected_queue_s
        row["service_s"] = 0.0
        row["saved_service_s"] = 0.0
        normalized.append(row)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    )
    parser.add_argument("--tokenizer")
    parser.add_argument("--cell-label", required=True)
    parser.add_argument("--formal-block-id")
    parser.add_argument("--formal-cell-id", choices=["A", "B", "E", "F"])
    parser.add_argument("--formal-order-index", type=int)
    parser.add_argument("--server-instance-id")
    parser.add_argument(
        "--fresh-server",
        action="store_true",
        help="Assert that this cell was started on a new vLLM server instance.",
    )
    parser.add_argument(
        "--result-cache-empty",
        action="store_true",
        help="Assert that no live-tool result cache was carried into this cell.",
    )
    parser.add_argument(
        "--call-graph-mode",
        choices=["autonomous", "frozen"],
        default="autonomous",
        help=(
            "Use model selection from live search results, or freeze each visit "
            "to the workload's expected_url while still executing live search."
        ),
    )
    parser.add_argument(
        "--speculation-mode",
        choices=["off", "search", "visit", "search_visit"],
        required=True,
    )
    parser.add_argument(
        "--tool-signal-policy",
        choices=["legacy", "execution_aware"],
        default="legacy",
        help=(
            "Gate direct LLM overlap bonuses using live broker execution state; "
            "execution_aware gives no readiness bonus to queued-only predictions."
        ),
    )
    parser.add_argument("--visit-top-k", type=int, default=1)
    parser.add_argument(
        "--visit-prediction-model",
        type=Path,
        help=(
            "Checksummed rank-only or contextual exact-URL predictor artifact. "
            "When supplied, visit speculation is late-bound to the current live "
            "search response; the LLM still chooses authoritatively from all "
            "returned URLs."
        ),
    )
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--max-active-tasks", type=int, default=0)
    parser.add_argument("--tool-workers", type=int, default=8)
    parser.add_argument("--speculative-tool-workers", type=int, default=4)
    parser.add_argument(
        "--min-speculative-tool-workers",
        type=int,
        default=0,
        help=(
            "Enable one bounded speculative start opportunity (0 or 1); a "
            "same-tool authoritative start repays every contested overtake."
        ),
    )
    parser.add_argument(
        "--search-tool-capacity",
        type=int,
        default=0,
        help="Shared search execution cap (0 uses the global tool-worker cap).",
    )
    parser.add_argument(
        "--visit-tool-capacity",
        type=int,
        default=0,
        help="Shared visit execution cap (0 uses the global tool-worker cap).",
    )
    parser.add_argument(
        "--search-min-start-interval-s",
        type=float,
        default=0.0,
        help="Minimum interval between physical search starts in the shared broker.",
    )
    parser.add_argument(
        "--visit-min-start-interval-s",
        type=float,
        default=0.0,
        help="Minimum interval between physical visit starts in the shared broker.",
    )
    parser.add_argument("--max-speculative-pending", type=int, default=128)
    parser.add_argument("--speculative-ttl-s", type=float, default=60.0)
    parser.add_argument("--tool-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--tool-http-max-attempts",
        type=int,
        default=1,
        help=(
            "Maximum real HTTP attempts per live-tool request; values above one "
            "enable bounded transient-failure retries."
        ),
    )
    parser.add_argument(
        "--tool-http-retry-backoff-s",
        type=float,
        default=1.0,
        help="Backoff before each bounded live HTTP retry.",
    )
    parser.add_argument(
        "--tool-http-attempt-start-gate",
        action="store_true",
        help=(
            "Apply each configured per-tool minimum-start interval to every "
            "physical HTTP GET, including concurrent requests and retries. "
            "Disabled by default to preserve prior experiment semantics."
        ),
    )
    parser.add_argument("--tool-service-hint-s", type=float, default=1.5)
    parser.add_argument("--visit-mode", choices=["direct", "jina"], default="direct")
    parser.add_argument(
        "--search-mode", choices=["rest", "action", "bing"], default="rest"
    )
    parser.add_argument("--search-max-results", type=int, default=5)
    parser.add_argument("--visit-max-chars", type=int, default=8000)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--max-tokens-tool", type=int, default=128)
    parser.add_argument("--max-tokens-answer", type=int, default=160)
    parser.add_argument(
        "--fixed-final-completion-tokens",
        type=int,
        choices=[192],
        default=None,
        help=(
            "Opt in to the frozen 192-token guided-grammar final-answer "
            "contract. This requires --tokenizer; tool calls remain compact."
        ),
    )
    parser.add_argument("--predicted-visit-result-tokens", type=int, default=1600)
    parser.add_argument(
        "--context-padding-tokens",
        type=int,
        default=0,
        help="Per-task private agent-history tokens used to create real KV pressure.",
    )
    parser.add_argument("--queue-sample-interval-s", type=float, default=0.2)
    parser.add_argument(
        "--visit-canary-stride",
        type=int,
        default=0,
        help=(
            "Every Nth task bypasses an exact visit prediction at commit time, "
            "providing an unbiased authoritative-latency canary (0 disables)."
        ),
    )
    return parser.parse_args()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


async def _fetch_metrics(
    session: aiohttp.ClientSession, server_url: str
) -> dict[str, float]:
    try:
        from prometheus_client.parser import text_string_to_metric_families

        async with session.get(
            f"{server_url.rstrip('/')}/metrics",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            response.raise_for_status()
            text = await response.text()
        result: dict[str, float] = {}
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                result[sample.name] = result.get(sample.name, 0.0) + float(sample.value)
        return result
    except Exception:
        return {}


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    interesting = (
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_inference_time_seconds_sum",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_decode_time_seconds_sum",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    )
    return {
        key: after[key] - before.get(key, 0.0)
        for key in interesting
        if key in after
    }


def _http_attempt_start_intervals(
    args: argparse.Namespace,
) -> dict[str, float]:
    """Map the explicit opt-in to the exact executor-level attempt gate."""

    if not args.tool_http_attempt_start_gate:
        return {}
    return {
        tool_name: float(interval_s)
        for tool_name, interval_s in (
            ("search", args.search_min_start_interval_s),
            ("visit", args.visit_min_start_interval_s),
        )
        if interval_s > 0
    }


async def _sample_joint_queues(
    *,
    broker: LiveToolBroker,
    session: aiohttp.ClientSession,
    server_url: str,
    stop: asyncio.Event,
    interval_s: float,
    output: list[dict[str, Any]],
) -> None:
    while True:
        snapshot = broker.snapshot()
        metrics = await _fetch_metrics(session, server_url)
        counts = snapshot["counts"]
        output.append(
            {
                "wall_s": time.time(),
                "monotonic_s": time.monotonic(),
                "broker_revision": snapshot["revision"],
                "tool_queued_authoritative": counts["queued_authoritative"],
                "tool_queued_speculative": counts["queued_speculative"],
                "tool_running_authoritative": counts["running_authoritative"],
                "tool_running_speculative": counts["running_speculative"],
                "tool_completed_unclaimed_speculative": counts[
                    "completed_unclaimed_speculative"
                ],
                "llm_running": metrics.get("vllm:num_requests_running"),
                "llm_waiting": metrics.get("vllm:num_requests_waiting"),
                "gpu_cache_usage": metrics.get("vllm:gpu_cache_usage_perc"),
            }
        )
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def _timeline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_llm = [row for row in rows if row["llm_waiting"] is not None]
    tool_auth_queue = [row for row in rows if row["tool_queued_authoritative"] > 0]
    tool_any_queue = [
        row
        for row in rows
        if row["tool_queued_authoritative"] + row["tool_queued_speculative"] > 0
    ]
    joint_pressure = [
        row
        for row in valid_llm
        if row["llm_waiting"] > 0
        and (
            row["tool_queued_authoritative"] > 0
            or row["tool_running_authoritative"] > 0
        )
    ]
    return {
        "sample_count": len(rows),
        "llm_metric_sample_count": len(valid_llm),
        "tool_authoritative_queue_sample_count": len(tool_auth_queue),
        "tool_any_queue_sample_count": len(tool_any_queue),
        "joint_llm_wait_and_live_tool_pressure_sample_count": len(joint_pressure),
        "tool_authoritative_queue_fraction": (
            len(tool_auth_queue) / len(rows) if rows else None
        ),
        "tool_any_queue_fraction": len(tool_any_queue) / len(rows) if rows else None,
        "joint_pressure_fraction": (
            len(joint_pressure) / len(valid_llm) if valid_llm else None
        ),
        "max_tool_queued_authoritative": max(
            (row["tool_queued_authoritative"] for row in rows), default=0
        ),
        "max_tool_queued_speculative": max(
            (row["tool_queued_speculative"] for row in rows), default=0
        ),
        "max_llm_running": max(
            (row["llm_running"] for row in valid_llm), default=None
        ),
        "max_llm_waiting": max(
            (row["llm_waiting"] for row in valid_llm), default=None
        ),
    }


async def async_main(args: argparse.Namespace) -> int:
    if args.replicas <= 0:
        raise ValueError("--replicas must be positive")
    if args.source_limit is not None and args.source_limit <= 0:
        raise ValueError("--source-limit must be positive")
    if args.max_active_tasks < 0:
        raise ValueError("--max-active-tasks cannot be negative")
    if args.visit_top_k <= 0:
        raise ValueError("--visit-top-k must be positive")
    if args.visit_prediction_model is not None and args.speculation_mode not in {
        "visit",
        "search_visit",
    }:
        raise ValueError(
            "--visit-prediction-model requires visit or search_visit speculation"
        )
    if args.visit_canary_stride < 0:
        raise ValueError("--visit-canary-stride cannot be negative")
    if args.queue_sample_interval_s <= 0:
        raise ValueError("--queue-sample-interval-s must be positive")
    if args.context_padding_tokens < 0:
        raise ValueError("--context-padding-tokens cannot be negative")
    if args.context_padding_tokens > 0 and not args.tokenizer:
        raise ValueError(
            "--context-padding-tokens requires --tokenizer so the real model "
            "context cannot be overfilled by an approximate token count"
        )
    if args.fixed_final_completion_tokens is not None and not args.tokenizer:
        raise ValueError(
            "--fixed-final-completion-tokens requires --tokenizer so semantic "
            "and padding tokens use the exact model tokenizer"
        )
    if args.tool_http_max_attempts < 1:
        raise ValueError("--tool-http-max-attempts must be positive")
    if (
        not math.isfinite(args.tool_http_retry_backoff_s)
        or args.tool_http_retry_backoff_s < 0
    ):
        raise ValueError(
            "--tool-http-retry-backoff-s must be finite and non-negative"
        )
    formal_values = (
        args.formal_block_id,
        args.formal_cell_id,
        args.formal_order_index,
        args.server_instance_id,
    )
    if any(value is not None for value in formal_values):
        if any(value is None for value in formal_values):
            raise ValueError(
                "formal evidence requires block, cell, order, and server instance"
            )
        if args.formal_order_index not in range(4):
            raise ValueError("--formal-order-index must be in [0, 3]")
        if not args.fresh_server or not args.result_cache_empty:
            raise ValueError(
                "formal evidence requires --fresh-server and --result-cache-empty"
            )
    elif args.fresh_server or args.result_cache_empty:
        raise ValueError(
            "fresh-server/cache assertions require the complete formal evidence tuple"
        )
    if args.call_graph_mode == "frozen" and args.search_mode != "bing":
        raise ValueError(
            "--call-graph-mode frozen requires --search-mode bing so the "
            "primary cell still executes live Bing search"
        )
    if not 0 <= args.speculative_tool_workers <= args.tool_workers:
        raise ValueError("speculative tool workers must be within total tool capacity")
    if not (
        args.min_speculative_tool_workers in {0, 1}
        and args.min_speculative_tool_workers <= args.speculative_tool_workers
    ):
        raise ValueError(
            "minimum speculative tool workers must be 0 or 1 and within "
            "speculative capacity"
        )
    for name in ("search_tool_capacity", "visit_tool_capacity"):
        value = getattr(args, name)
        if value < 0 or value > args.tool_workers:
            raise ValueError(
                f"{name.replace('_', '-')} must be 0 or in [1, tool-workers]"
            )
    for name in ("search_min_start_interval_s", "visit_min_start_interval_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
    if args.tool_http_attempt_start_gate and not (
        args.search_min_start_interval_s > 0
        or args.visit_min_start_interval_s > 0
    ):
        raise ValueError(
            "--tool-http-attempt-start-gate requires at least one positive "
            "per-tool minimum-start interval"
        )

    visit_predictor = (
        load_visit_predictor(args.visit_prediction_model, top_k=args.visit_top_k)
        if args.visit_prediction_model is not None
        else None
    )
    workload_path = Path(args.workload).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    payload = json.loads(workload_path.read_text(encoding="utf-8"))
    sources = validate_sources(payload, call_graph_mode=args.call_graph_mode)
    if args.source_limit is not None:
        sources = sources[: args.source_limit]
    tasks = [(source, replica) for source in sources for replica in range(args.replicas)]
    workload_bytes = workload_path.read_bytes()

    if args.tokenizer:
        counter = TransformersTokenCounter(args.tokenizer)
    else:
        counter = ApproximateTokenCounter()

    api_key = os.getenv("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=15)
    started_wall = time.time()
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        before_metrics = await _fetch_metrics(session, args.server_url)
        tool_capacities = {
            tool_name: capacity
            for tool_name, capacity in (
                ("search", args.search_tool_capacity),
                ("visit", args.visit_tool_capacity),
            )
            if capacity > 0
        }
        tool_min_start_intervals_s = {
            tool_name: interval_s
            for tool_name, interval_s in (
                ("search", args.search_min_start_interval_s),
                ("visit", args.visit_min_start_interval_s),
            )
            if interval_s > 0
        }
        http_attempt_min_start_intervals_s = _http_attempt_start_intervals(args)
        executor = WikipediaLiveExecutor(
            max_results=args.search_max_results,
            visit_max_chars=args.visit_max_chars,
            timeout_s=args.tool_timeout_s,
            max_http_attempts=args.tool_http_max_attempts,
            retry_backoff_s=args.tool_http_retry_backoff_s,
            http_attempt_min_start_intervals_s=(
                http_attempt_min_start_intervals_s
            ),
            visit_mode=args.visit_mode,
            search_mode=args.search_mode,
        )
        broker = LiveToolBroker(
            executor,
            max_workers=args.tool_workers,
            max_speculative_workers=args.speculative_tool_workers,
            min_speculative_workers=args.min_speculative_tool_workers,
            max_speculative_pending=args.max_speculative_pending,
            ttl_s=args.speculative_ttl_s,
            service_time_hints_s={
                "search": args.tool_service_hint_s,
                "visit": args.tool_service_hint_s,
            },
            tool_capacities=tool_capacities,
            tool_min_start_intervals_s=tool_min_start_intervals_s,
        )
        llm = LiveLLMClient(
            session,
            server_url=args.server_url,
            model=args.model,
            timeout_s=args.request_timeout_s,
        )
        experiment = LiveClosedLoopExperiment(
            broker=broker,
            llm=llm,
            token_counter=counter,
            speculation_mode=args.speculation_mode,
            visit_top_k=args.visit_top_k,
            max_tokens_tool=args.max_tokens_tool,
            max_tokens_answer=args.max_tokens_answer,
            default_tool_service_s=args.tool_service_hint_s,
            predicted_visit_result_tokens=args.predicted_visit_result_tokens,
            call_graph_mode=args.call_graph_mode,
            context_padding_tokens=args.context_padding_tokens,
            tool_signal_policy=args.tool_signal_policy,
            fixed_final_completion_tokens=args.fixed_final_completion_tokens,
            visit_predictor=visit_predictor,
        )
        queue_samples: list[dict[str, Any]] = []
        queue_sampler_stop = asyncio.Event()
        queue_sampler = asyncio.create_task(
            _sample_joint_queues(
                broker=broker,
                session=session,
                server_url=args.server_url,
                stop=queue_sampler_stop,
                interval_s=args.queue_sample_interval_s,
                output=queue_samples,
            )
        )
        semaphore = (
            asyncio.Semaphore(args.max_active_tasks)
            if args.max_active_tasks > 0
            else None
        )

        async def run_one(
            source: Any, replica: int, task_index: int
        ) -> dict[str, Any]:
            eligible = not (
                args.visit_canary_stride > 0
                and task_index % args.visit_canary_stride == 0
            )
            if semaphore is None:
                return await experiment.run_task(
                    source,
                    replica=replica,
                    visit_speculation_eligible=eligible,
                )
            async with semaphore:
                return await experiment.run_task(
                    source,
                    replica=replica,
                    visit_speculation_eligible=eligible,
                )

        task_results = await asyncio.gather(
            *(
                run_one(source, replica, task_index)
                for task_index, (source, replica) in enumerate(tasks)
            )
        )
        tasks_ended_wall = time.time()
        broker_stats = broker.stats.to_dict()
        await broker.close()
        await executor.close()
        if (
            args.formal_block_id is not None
            and not executor.http_library_retry_disabled_effective
        ):
            raise RuntimeError(
                "formal live tool execution did not disable aiohttp's hidden "
                "persistent-connection retry"
            )
        ended_wall = time.time()
        broker_snapshot = broker.snapshot()
        broker_stats = broker.stats.to_dict()
        tool_attempt_records = _normalize_tool_attempt_records(
            broker.tool_records()
        )
        queue_sampler_stop.set()
        await queue_sampler
        after_metrics = await _fetch_metrics(session, args.server_url)

    broker_counts = broker_snapshot.get("counts", {})
    broker_drained = bool(
        isinstance(broker_counts, dict)
        and not broker_snapshot.get("jobs")
        and all(
            broker_counts.get(key) == 0
            for key in (
                "completed_unclaimed_speculative",
                "queued_authoritative",
                "queued_speculative",
                "running_authoritative",
                "running_speculative",
            )
        )
        and broker_counts.get("queued_by_tool") == {}
        and broker_counts.get("running_by_tool") == {}
    )
    if args.formal_block_id is not None and not broker_drained:
        raise RuntimeError("formal cell ended with a non-drained live tool broker")

    summary = summarize_live_run(
        tasks=task_results,
        llm_events=llm.events,
        broker_stats=broker_stats,
        started_wall_s=started_wall,
        ended_wall_s=ended_wall,
    )
    expected_url_eligible = sum(
        1 for source, _replica in tasks if source.expected_url is not None
    )
    expected_url_observations = [
        task.get("search_result_contains_expected_url")
        for task in task_results
        if isinstance(task.get("search_result_contains_expected_url"), bool)
    ]
    expected_url_matches = sum(bool(value) for value in expected_url_observations)
    expected_url_search_coverage = {
        "eligible_task_count": expected_url_eligible,
        "observed_task_count": len(expected_url_observations),
        "matched_task_count": expected_url_matches,
        "fraction_of_observed": (
            expected_url_matches / len(expected_url_observations)
            if expected_url_observations
            else None
        ),
        "fraction_of_eligible": (
            expected_url_matches / expected_url_eligible
            if expected_url_eligible
            else None
        ),
    }
    config = {
        "cell_label": args.cell_label,
        "server_url": args.server_url,
        "model": args.model,
        "call_graph_mode": args.call_graph_mode,
        "expected_url_search_coverage": expected_url_search_coverage,
        "speculation_mode": args.speculation_mode,
        "tool_signal_policy": args.tool_signal_policy,
        "tool_signal_policy_version": TOOL_SIGNAL_POLICY_VERSION,
        "tool_signal_policy_module_sha256": hashlib.sha256(
            (
                REPRODUCTION_ROOT
                / "paste_repro"
                / "online_learned_agent.py"
            ).read_bytes()
        ).hexdigest(),
        "visit_top_k": args.visit_top_k,
        "visit_prediction": (
            visit_predictor.metadata()
            if visit_predictor is not None
            else {
                "policy": "current-result-order-top-k",
                "top_k": args.visit_top_k,
                "artifact_sha256": None,
            }
        ),
        "independent_source_count": len(sources),
        "replicas": args.replicas,
        "task_count": len(tasks),
        "max_active_tasks": args.max_active_tasks or len(tasks),
        "tool_workers": args.tool_workers,
        "speculative_tool_workers": args.speculative_tool_workers,
        "min_speculative_tool_workers": args.min_speculative_tool_workers,
        "search_tool_capacity": args.search_tool_capacity,
        "visit_tool_capacity": args.visit_tool_capacity,
        "search_min_start_interval_s": args.search_min_start_interval_s,
        "visit_min_start_interval_s": args.visit_min_start_interval_s,
        "max_speculative_pending": args.max_speculative_pending,
        "speculative_ttl_s": args.speculative_ttl_s,
        "tool_http_max_attempts": args.tool_http_max_attempts,
        "tool_http_retry_backoff_s": args.tool_http_retry_backoff_s,
        "tool_http_attempt_start_gate_enabled": (
            args.tool_http_attempt_start_gate
        ),
        "tool_http_attempt_start_gate_policy_version": (
            WikipediaLiveExecutor.HTTP_ATTEMPT_START_GATE_VERSION
        ),
        "tool_http_attempt_min_start_intervals_s": (
            executor.http_attempt_min_start_intervals_s
        ),
        "tool_http_retry_policy_version": (
            WikipediaLiveExecutor.HTTP_RETRY_POLICY_VERSION
        ),
        "tool_http_retryable_statuses": list(
            WikipediaLiveExecutor.RETRYABLE_HTTP_STATUSES
        ),
        "tool_http_retryable_exception_types": list(
            WikipediaLiveExecutor.RETRYABLE_HTTP_EXCEPTION_TYPES
        ),
        "tool_http_library_retry_disabled": (
            executor.http_library_retry_disabled_effective
        ),
        "tool_http_library_retry_control_version": (
            WikipediaLiveExecutor.HTTP_LIBRARY_RETRY_CONTROL_VERSION
        ),
        "tool_http_library_name": executor.http_library_name,
        "tool_http_library_version": executor.http_library_version,
        "visit_mode": args.visit_mode,
        "search_mode": args.search_mode,
        "search_max_results": args.search_max_results,
        "visit_max_chars": args.visit_max_chars,
        "max_tokens_tool": args.max_tokens_tool,
        "max_tokens_answer": args.max_tokens_answer,
        "fixed_final_completion_tokens": args.fixed_final_completion_tokens,
        "fixed_final_completion_enabled": (
            args.fixed_final_completion_tokens is not None
        ),
        "final_answer_contract_policy_version": (
            FIXED_FINAL_ANSWER_CONTRACT_POLICY_VERSION
            if args.fixed_final_completion_tokens is not None
            else None
        ),
        "final_answer_schema_policy_version": (
            FINAL_ANSWER_SCHEMA_POLICY_VERSION
            if args.fixed_final_completion_tokens is not None
            else None
        ),
        "final_answer_grammar_policy_version": (
            FINAL_ANSWER_GRAMMAR_POLICY_VERSION
            if args.fixed_final_completion_tokens is not None
            else None
        ),
        "final_answer_grammar_xgrammar_version": (
            FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION
            if args.fixed_final_completion_tokens is not None
            else None
        ),
        "output_contract_policy_version": (
            FIXED_OUTPUT_CONTRACT_POLICY_VERSION
            if args.fixed_final_completion_tokens is not None
            else None
        ),
        "live_agent_sha256": hashlib.sha256(
            (REPRODUCTION_ROOT / "paste_repro" / "live_agent.py").read_bytes()
        ).hexdigest(),
        "tool_call_prompt_encoding": (
            "canonical_json_sort_keys_compact"
            if args.fixed_final_completion_tokens is not None
            else "raw_model_completion"
        ),
        "visit_canary_stride": args.visit_canary_stride,
        "context_padding_tokens": args.context_padding_tokens,
        "queue_sample_interval_s": args.queue_sample_interval_s,
        "token_count_method": getattr(counter, "method", type(counter).__name__),
        "live_tool_execution": True,
        "recorded_tool_sleep": False,
        "controlled_http_retry": args.tool_http_max_attempts > 1,
        "shared_bounded_tool_pool": True,
        "generated_tool_call_controls_next_prompt": True,
        "authoritative_and_speculative_share_capacity": True,
        "workload_path": str(workload_path),
        "workload_file_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "selected_workload_sha256": sha256_json(
            [task_to_dict(source) for source in sources]
        ),
        "scheduler_environment": {
            key: os.environ.get(key) for key in SCHEDULER_ENV_KEYS
        },
        "tool_metadata_is_causal": True,
        "tool_result_private_until_exact_commit": True,
        "future_trace_oracle_used": False,
        "frozen_url_is_workload_input": args.call_graph_mode == "frozen",
        "workload_split_id": payload.get("split_id"),
        "workload_split_role": payload.get("split_role"),
        "workload_formal_eligible": payload.get("formal_eligible"),
    }
    if args.formal_block_id is not None:
        config["formal_run"] = {
            "block_id": args.formal_block_id,
            "cell_id": args.formal_cell_id,
            "order_index": args.formal_order_index,
            "server_instance_id": args.server_instance_id,
            "fresh_server": True,
            "result_cache_empty": True,
            "broker_drained": broker_drained,
        }
    result = {
        "schema_version": 1,
        "config": config,
        "summary": summary,
        "task_completion_makespan_s": tasks_ended_wall - started_wall,
        "tasks": task_results,
        "llm_events": llm.events,
        "tool_attempt_records": tool_attempt_records,
        "broker_final_snapshot": broker_snapshot,
        "vllm_metric_deltas": _metric_delta(before_metrics, after_metrics),
        "queue_timeline_summary": _timeline_summary(queue_samples),
    }
    timeline_path = output_dir / "queue_timeline.jsonl"
    _write_jsonl_atomic(timeline_path, queue_samples)
    result["raw_evidence"] = {
        "queue_timeline": {
            "path": str(timeline_path),
            "sha256": hashlib.sha256(timeline_path.read_bytes()).hexdigest(),
            "sample_count": len(queue_samples),
        }
    }
    _write_json_atomic(output_dir / "result.json", result)
    print(json.dumps({"config": config, "summary": summary}, ensure_ascii=False, indent=2))
    return 0 if summary["all_tasks_succeeded"] else 1


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
