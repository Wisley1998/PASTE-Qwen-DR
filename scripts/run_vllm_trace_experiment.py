#!/usr/bin/env python3
"""
Run a strict trace-driven vLLM replay experiment.

Each trace task replays the recorded message histories against a live vLLM server.
Tool execution is not performed; instead, the original tool-side timing is used to
control when the next LLM request arrives.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from azure_llm_trace import apply_azure_arrivals, load_azure_llm_invocations
from online_session_predictor import OnlineSessionPredictor
from trace_experiment_lib import (
    cap_workload_by_arrival_time,
    load_workload,
    prepare_trace_workload,
    save_workload,
    summarize_workload,
    validate_learned_workload_artifact,
)


_ONLINE_PREDICTOR: Optional[OnlineSessionPredictor] = None

# Increment this only when the accepted physical-KV decision semantics change.
# Revalidation sidecars bind both this identifier and the module SHA256.
PHYSICAL_KV_LOG_PARSER_ID = "paste.physical_kv_admission_log_parser"
PHYSICAL_KV_LOG_PARSER_VERSION = 2

# This is also the immutable runtime-configuration evidence consumed by the
# reproduction validators.  Keep engine-shape and every online_joint_pacer_v2
# control here, not just the fields currently used in its ranking score.  The
# API key and cache paths are deliberately excluded.
_SCHEDULER_ENV_KEYS = (
    "PASTE_STRESS_PROFILE",
    "PASTE_FROZEN_CONFIG_SHA256",
    "PASTE_SCHEDULER_METADATA_MODE",
    "MODEL_ID",
    "MODEL_REVISION",
    "CUDA_VISIBLE_DEVICES",
    "VLLM_HOST",
    "VLLM_PROBE_HOST",
    "VLLM_PORT",
    "VLLM_TP_SIZE",
    "VLLM_DTYPE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_SCHED_POLICY",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_CUDA_GRAPH_SIZES",
    "VLLM_USE_V1",
    "VLLM_SCHED_PRED_OUT_ENABLE",
    "VLLM_SCHED_DEFAULT_PRED_OUT",
    "VLLM_SCHED_PRED_OUT_EMA_ALPHA",
    "VLLM_SCHED_AVG_CALL_SERVICE_S",
    "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2",
    "VLLM_SCHED_DECODE_TOKENS_PER_S_V2",
    "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S",
    "VLLM_SCHED_TIME_AGING_ALPHA",
    "VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS",
    "VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING",
    "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
    "VLLM_SCHED_JOINT_V2_FINAL_LANE",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S",
    "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING",
    "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING",
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


def _predictor() -> Optional[OnlineSessionPredictor]:
    return _ONLINE_PREDICTOR


def _maybe_init_predictor(
    metadata_mode: str,
    calibration_workload_path: Optional[str | Path],
) -> None:
    """Initialize online metadata without consulting the replay workload."""

    global _ONLINE_PREDICTOR
    _ONLINE_PREDICTOR = None
    if metadata_mode == "oracle":
        return
    if metadata_mode != "online":
        raise ValueError(f"unsupported scheduler metadata mode: {metadata_mode}")
    if calibration_workload_path is None:
        raise ValueError(
            "online scheduler metadata requires --scheduler-calibration-workload; "
            "the replay workload is deliberately never used for calibration. "
            "Use --scheduler-metadata-mode oracle only for legacy lookahead replay."
        )
    src = Path(calibration_workload_path)
    if not src.is_file():
        raise FileNotFoundError(f"scheduler calibration workload not found: {src}")
    _ONLINE_PREDICTOR = OnlineSessionPredictor.from_workload(src)
    print(
        f"[metadata] online predictor calibrated from {src} "
        f"next_tool_wait_reliability="
        f"{_ONLINE_PREDICTOR.next_tool_wait_reliability:.6f}"
    )


def _source_session_ids(workload: Dict[str, Any], workload_label: str) -> set[str]:
    """Return stable source-session IDs, failing closed on incomplete provenance."""

    traces = workload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError(f"{workload_label} workload has no traces/source sessions")

    session_ids: set[str] = set()
    for trace_index, trace in enumerate(traces):
        source_trace = trace.get("source_trace")
        if source_trace is None or not str(source_trace).strip():
            raise ValueError(
                f"{workload_label} workload trace {trace_index} is missing source_trace"
            )
        session_id = Path(str(source_trace)).name
        if not session_id:
            raise ValueError(
                f"{workload_label} workload trace {trace_index} has invalid source_trace"
            )
        session_ids.add(session_id)

    if not session_ids:
        raise ValueError(f"{workload_label} workload has no source sessions")
    return session_ids


def _validate_disjoint_calibration_sessions(
    replay_workload: Dict[str, Any],
    calibration_workload: Dict[str, Any],
) -> None:
    """Reject calibration data containing any replay source session."""

    replay_sessions = _source_session_ids(replay_workload, "replay")
    calibration_sessions = _source_session_ids(calibration_workload, "calibration")
    overlap = replay_sessions & calibration_sessions
    if overlap:
        overlap_preview = ", ".join(sorted(overlap)[:10])
        raise ValueError(
            "online scheduler calibration overlaps replay source sessions: "
            f"{overlap_preview}"
        )


def _client_headers(api_key: Optional[str]) -> Dict[str, str]:
    """Build request headers without persisting endpoint credentials."""

    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _is_retryable_transport_error(exc: BaseException) -> bool:
    """Return whether a failure is an explicit non-timeout transport error."""

    return (
        isinstance(exc, aiohttp.ClientConnectionError)
        and not isinstance(
            exc,
            (
                asyncio.TimeoutError,
                aiohttp.ClientSSLError,
                aiohttp.ServerFingerprintMismatch,
            ),
        )
    )


def _transport_delivery_is_ambiguous(exc: BaseException) -> bool:
    """Whether a transport failure may have happened after the POST was sent."""

    return _is_retryable_transport_error(exc) and not isinstance(
        exc, aiohttp.ClientConnectorError
    )


def _summarize_request_attempts(
    request_events: List[Dict[str, Any]],
    configured_max_request_attempts: int,
) -> Dict[str, int]:
    """Validate attempt histories and return transparent retry aggregates."""

    if configured_max_request_attempts <= 0:
        raise ValueError("configured_max_request_attempts must be positive")

    attempts_total = 0
    retry_count = 0
    retried_request_count = 0
    retry_success_count = 0
    ambiguous_retry_count = 0
    final_failure_count = 0

    for event_index, event in enumerate(request_events):
        history = event.get("attempt_history")
        if not isinstance(history, list) or not history:
            raise ValueError(
                f"request event {event_index} has no non-empty attempt_history"
            )
        attempts = event.get("attempts")
        if not isinstance(attempts, int) or isinstance(attempts, bool):
            raise ValueError(f"request event {event_index} has invalid attempts")
        if attempts != len(history):
            raise ValueError(
                f"request event {event_index} attempts/history length mismatch"
            )
        if attempts > configured_max_request_attempts:
            raise ValueError(
                f"request event {event_index} exceeds configured max attempts"
            )

        for attempt_index, record in enumerate(history, start=1):
            if not isinstance(record, dict) or record.get("attempt") != attempt_index:
                raise ValueError(
                    f"request event {event_index} has malformed attempt_history"
                )
            required_fields = {
                "transport",
                "outcome",
                "http_status",
                "error_type",
                "error",
                "duration_s",
                "retryable",
                "will_retry",
                "retry_backoff_s",
                "delivery_ambiguous",
            }
            if not required_fields.issubset(record):
                raise ValueError(
                    f"request event {event_index} has incomplete attempt_history"
                )
            duration_s = record.get("duration_s")
            if (
                not isinstance(duration_s, (int, float))
                or isinstance(duration_s, bool)
                or duration_s < 0
            ):
                raise ValueError(
                    f"request event {event_index} has invalid attempt duration"
                )
            is_last = attempt_index == attempts
            will_retry = record.get("will_retry")
            if not isinstance(will_retry, bool):
                raise ValueError(
                    f"request event {event_index} has invalid will_retry marker"
                )
            if is_last and will_retry:
                raise ValueError(
                    f"request event {event_index} retries after its final attempt"
                )
            if not is_last:
                if not will_retry or record.get("outcome") != "transport_error":
                    raise ValueError(
                        f"request event {event_index} has a non-transport retry"
                    )
                if record.get("retryable") is not True:
                    raise ValueError(
                        f"request event {event_index} retries a non-retryable error"
                    )
                if record.get("delivery_ambiguous") is True:
                    ambiguous_retry_count += 1

        final_record = history[-1]
        final_ok = bool(event.get("ok"))
        history_ok = (
            final_record.get("outcome") == "success"
            and final_record.get("http_status") == 200
        )
        if final_ok != history_ok:
            raise ValueError(
                f"request event {event_index} final outcome/history mismatch"
            )
        if event.get("http_status") != final_record.get("http_status"):
            raise ValueError(
                f"request event {event_index} final HTTP status/history mismatch"
            )

        attempts_total += attempts
        retry_count += attempts - 1
        if attempts > 1:
            retried_request_count += 1
            retry_success_count += 1 if final_ok else 0
        final_failure_count += 0 if final_ok else 1

    if attempts_total != len(request_events) + retry_count:
        raise ValueError("request attempt aggregate identity is inconsistent")

    return {
        "configured_max_request_attempts": configured_max_request_attempts,
        "request_attempts_total": attempts_total,
        "retry_count": retry_count,
        "retried_request_count": retried_request_count,
        "retry_success_count": retry_success_count,
        "ambiguous_retry_count": ambiguous_retry_count,
        "final_failure_count": final_failure_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict trace replay against a live vLLM server")
    parser.add_argument("--trace-dir", default="traces/my_traces")
    parser.add_argument("--prepared-workload", default=None)
    parser.add_argument(
        "--azure-arrival-trace",
        default=None,
        help=(
            "Azure LLM Inference Trace 2024 CSV. When set, its timestamps "
            "replace only the top-level Agent-session arrivals."
        ),
    )
    parser.add_argument(
        "--azure-dataset-variant",
        choices=["conversation", "code"],
        default="conversation",
    )
    parser.add_argument("--azure-start-time", default=None)
    parser.add_argument("--azure-duration-s", type=float, default=None)
    parser.add_argument(
        "--azure-max-sessions",
        type=int,
        default=None,
        help=(
            "Maximum Azure rows to replay. Defaults to --trace-count as a "
            "safety bound."
        ),
    )
    parser.add_argument(
        "--azure-arrival-speedup",
        type=float,
        default=1.0,
        help="Compress only inter-session Azure arrival offsets.",
    )
    parser.add_argument(
        "--azure-session-mapping",
        choices=["round_robin", "shuffled_round_robin"],
        default="round_robin",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    parser.add_argument("--tokenizer", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    parser.add_argument("--trace-count", type=int, default=128)
    parser.add_argument("--speedup", type=float, required=True)
    parser.add_argument(
        "--scheduler-metadata-mode",
        choices=["online", "oracle"],
        default=os.getenv("VLLM_SCHED_METADATA_MODE", "online"),
        help=(
            "online emits causal scheduler metadata estimated from a separate "
            "calibration workload; explicitly select oracle only to preserve the "
            "legacy trace-lookahead path."
        ),
    )
    parser.add_argument(
        "--scheduler-calibration-workload",
        default=(
            os.getenv("VLLM_SCHED_CALIBRATION_WORKLOAD")
            or os.getenv("VLLM_SCHED_PREDICTOR_CALIB")
            or None
        ),
        help=(
            "Prepared workload used only to fit online session metadata. It must "
            "be separate from the workload being replayed."
        ),
    )
    parser.add_argument(
        "--tool-wait-mode",
        choices=["sleep", "measure_only"],
        default="sleep",
        help=(
            "sleep replays tool waits on the critical path. measure_only records "
            "the scaled tool waits but does not sleep, useful for no-queue solo "
            "LLM timing baselines."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-arrival-time-s", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-output-tokens-cap", type=int, default=8192)
    parser.add_argument("--output-token-buffer", type=int, default=64)
    parser.add_argument("--min-output-tokens-floor", type=int, default=1024)
    parser.add_argument(
        "--tool-overlap-mode",
        choices=["none", "native", "oracle", "learned"],
        default=os.getenv("TRACE_TOOL_OVERLAP_MODE", "none"),
        help=(
            "Transform trace tool waits to model tool-LLM overlap. "
            "'native' models the existing search->visit prefetch/cache path; "
            "'learned' late-binds a checksummed training artifact to URLs in the "
            "currently visible search response; 'oracle' is an upper bound."
        ),
    )
    parser.add_argument(
        "--tool-prediction-model",
        default=os.getenv("TRACE_TOOL_PREDICTION_MODEL") or None,
        help=(
            "Checksummed URL-rank mapper artifact. Required for learned overlap; "
            "the replay never retrains it on evaluation traces."
        ),
    )
    parser.add_argument(
        "--tool-prediction-top-k",
        type=int,
        default=int(os.getenv("TRACE_TOOL_PREDICTION_TOP_K", "5")),
        help="Maximum concrete visit predictions admitted after each search.",
    )
    parser.add_argument(
        "--tool-overlap-efficiency",
        type=float,
        default=float(os.getenv("TRACE_TOOL_OVERLAP_EFFICIENCY", "1.0")),
        help="Fraction of the previous LLM inference window usable for speculative tool work.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--presence-penalty", type=float, default=1.1)
    parser.add_argument("--request-timeout-s", type=int, default=3600)
    parser.add_argument(
        "--max-request-attempts",
        type=int,
        default=3,
        help=(
            "Maximum wire attempts per request. Only explicit non-timeout "
            "aiohttp connection errors are retried."
        ),
    )
    parser.add_argument("--metrics-scrape-interval-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument(
        "--prefix-marker-mode",
        choices=["preserve_prefix", "break_prefix"],
        default=os.getenv("TRACE_PREFIX_MARKER_MODE", "preserve_prefix"),
        help=(
            "Where to inject duplicate-trace markers. preserve_prefix keeps prefix-cache sharing; "
            "break_prefix puts the marker first to reduce prefix sharing for stress workloads."
        ),
    )
    parser.add_argument(
        "--max-active-traces",
        type=int,
        default=None,
        help="Limit concurrently replayed trace tasks while keeping the workload fixed.",
    )
    parser.add_argument(
        "--vllm-log-file",
        default="logs/vllm_trace_experiments/vllm_8000.log",
    )
    parser.add_argument(
        "--swap-events-file",
        default="logs/vllm_trace_experiments/vllm_8000_swap_events.jsonl",
    )
    return parser.parse_args()


def _json_dump_line(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _scaled_tool_wait_s(wait_s: float, speedup: float) -> float:
    return wait_s / speedup if speedup > 0 else wait_s


def _bounded_online_output_prediction(
    predicted_output_tokens: float,
    request_max_tokens: int,
) -> float:
    """Keep a causal output estimate inside the request's decoding budget.

    The global cold-start estimate can come from a workload with a different
    output cap.  Passing that larger value to the scheduler makes short smoke
    workloads look much more decode-heavy than they can possibly be.
    """

    return float(min(max(1.0, predicted_output_tokens), max(1, request_max_tokens)))


def _build_sched_request_id(
    trace: Dict[str, Any],
    request: Dict[str, Any],
    request_index: int,
    speedup: float,
    predicted_output_tokens: Optional[float] = None,
    metadata_mode: str = "oracle",
    online_predictor: Optional[OnlineSessionPredictor] = None,
    previous_requests: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Dict[str, Any]]:
    if metadata_mode == "online":
        predictor = online_predictor or _predictor()
        if predictor is None:
            raise RuntimeError(
                "online scheduler metadata requested before calibration predictor initialization"
            )

        # Deliberately index only requests already seen.  Do not call len(),
        # slice the suffix, or inspect any future request in this branch.
        known_previous = previous_requests
        if known_previous is None:
            known_previous = [trace["requests"][index] for index in range(request_index)]
        observed_requests = [*known_previous, request]
        past_tool_waits_s = [
            float(item.get("wait_after_prev_s", 0.0))
            for item in observed_requests
            if int(item.get("call_index", 0)) > 0
        ]
        prediction = predictor.predict(
            current_call_index=int(request["call_index"]),
            past_tool_waits_s=past_tool_waits_s,
        )
        remaining_calls = prediction.remaining_calls
        next_tool_wait_s = _scaled_tool_wait_s(prediction.next_tool_wait_s, speedup)
        remaining_tool_wait_s = _scaled_tool_wait_s(
            prediction.remaining_tool_wait_s,
            speedup,
        )
        meta = {
            "t": trace["trace_id"],
            "c": int(request["call_index"]),
            "i": request_index,
            "n": request_index + 1 + remaining_calls,
            "rc": remaining_calls,
            "nw": next_tool_wait_s,
            "nwc": predictor.next_tool_wait_reliability,
            "rtw": remaining_tool_wait_s,
            "pt": int(request["prompt_tokens"]),
            "mt": int(request["max_tokens"]),
            "ms": "online",
        }
        if predicted_output_tokens is not None:
            predicted_tokens = int(max(1, round(predicted_output_tokens)))
            meta["po"] = predicted_tokens
            if remaining_calls > 0:
                # This is a prediction carried forward from current/past output,
                # not the recorded output length of a future replay request.
                meta["npo"] = predicted_tokens
        meta_full = dict(meta)
        meta_full.update({"metadata_source": "online", "nw_src": "predicted"})
    elif metadata_mode == "oracle":
        requests = trace["requests"]
        total_calls = len(requests)
        remaining_requests = requests[request_index + 1 :]
        next_tool_wait_s = None
        if remaining_requests:
            next_tool_wait_s = _scaled_tool_wait_s(
                float(remaining_requests[0].get("wait_after_prev_s", 0.0)),
                speedup,
            )
        remaining_tool_wait_s = sum(
            _scaled_tool_wait_s(float(item.get("wait_after_prev_s", 0.0)), speedup)
            for item in remaining_requests
        )
        meta = {
            "t": trace["trace_id"],
            "c": int(request["call_index"]),
            "i": request_index,
            "n": total_calls,
            "rc": max(0, total_calls - request_index - 1),
            "nw": next_tool_wait_s,
            "rtw": remaining_tool_wait_s,
            "pt": int(request["prompt_tokens"]),
            "mt": int(request["max_tokens"]),
        }
        if predicted_output_tokens is not None:
            meta["po"] = int(max(1, round(predicted_output_tokens)))
        if remaining_requests:
            next_request = remaining_requests[0]
            meta["npt"] = int(next_request.get("prompt_tokens", 0) or 0)
            meta["nmt"] = int(next_request.get("max_tokens", 0) or 0)
            if predicted_output_tokens is not None:
                meta["npo"] = int(max(1, round(predicted_output_tokens)))
        meta_full = dict(meta)
        meta_full.update(
            {
                "metadata_source": "oracle",
                "nw_src": "oracle",
                "nw_oracle": next_tool_wait_s,
                "rtw_oracle": remaining_tool_wait_s,
            }
        )
    else:
        raise ValueError(f"unsupported scheduler metadata mode: {metadata_mode}")

    encoded = json.dumps(meta, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"schedx{encoded.encode('utf-8').hex()}z", meta_full


def _parse_prometheus_metrics(text: str) -> Dict[str, float]:
    from prometheus_client.parser import text_string_to_metric_families

    parsed: Dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name, labels, value = sample.name, sample.labels, float(sample.value)
            if labels:
                label_suffix = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
                parsed[f"{name}|{label_suffix}"] = value
            parsed[name] = parsed.get(name, 0.0) + value
    return parsed


async def fetch_metrics(session: aiohttp.ClientSession, metrics_url: str) -> Dict[str, Any]:
    async with session.get(metrics_url) as response:
        response.raise_for_status()
        text = await response.text()
    return {
        "timestamp": time.time(),
        "metrics": _parse_prometheus_metrics(text),
    }


async def scrape_metrics(
    session: aiohttp.ClientSession,
    metrics_url: str,
    stop_event: asyncio.Event,
    output_path: Path,
    interval_s: float,
) -> None:
    while not stop_event.is_set():
        try:
            snapshot = await fetch_metrics(session, metrics_url)
            _json_dump_line(output_path, snapshot)
        except Exception as exc:  # pragma: no cover
            _json_dump_line(
                output_path,
                {
                    "timestamp": time.time(),
                    "error": repr(exc),
                },
            )
        await asyncio.sleep(interval_s)


def delta_metric(before: Dict[str, float], after: Dict[str, float], metric_name: str) -> float:
    return float(after.get(metric_name, 0.0) - before.get(metric_name, 0.0))


def delta_metric_candidates(
    before: Dict[str, float],
    after: Dict[str, float],
    metric_names: List[str],
) -> tuple[float, Optional[str]]:
    for metric_name in metric_names:
        if metric_name in before or metric_name in after:
            return delta_metric(before, after, metric_name), metric_name
    return 0.0, None


def _read_text_segment(path: Path, start_offset: int, end_offset: int) -> str:
    if not path.exists() or end_offset <= start_offset:
        return ""
    with path.open("rb") as f:
        f.seek(start_offset)
        return f.read(end_offset - start_offset).decode("utf-8", errors="ignore")


def parse_vllm_log_segment(text: str) -> Dict[str, Any]:
    percentage = r"(?:\d+(?:\.\d*)?|\.\d+)"
    legacy_stats_pattern = re.compile(
        r"Running:\s*(?P<running>\d+)\s+reqs,\s*"
        r"Swapped:\s*(?P<swapped>\d+)\s+reqs,\s*"
        r"Pending:\s*(?P<waiting>\d+)\s+reqs,\s*"
        rf"GPU KV cache usage:\s*(?P<gpu>{percentage})\s*%,\s*"
        rf"CPU KV cache usage:\s*(?P<cpu>{percentage})\s*%"
    )
    modern_stats_pattern = re.compile(
        r"Running:\s*(?P<running>\d+)\s+reqs,\s*"
        r"Waiting:\s*(?P<waiting>\d+)\s+reqs,\s*"
        rf"GPU KV cache usage:\s*(?P<gpu>{percentage})\s*%"
    )
    legacy_prefix_pattern = re.compile(
        rf"Prefix cache hit rate:\s*GPU:\s*(?P<gpu>{percentage})\s*%,\s*"
        rf"CPU:\s*(?P<cpu>{percentage})\s*%"
    )
    modern_prefix_pattern = re.compile(
        rf"Prefix cache hit rate:\s*(?P<gpu>{percentage})\s*%"
    )

    stats_samples: List[Dict[str, Any]] = []
    for match in legacy_stats_pattern.finditer(text):
        stats_samples.append(
            {
                "running": int(match.group("running")),
                "waiting": int(match.group("waiting")),
                "swapped": int(match.group("swapped")),
                "gpu_cache_usage_perc": float(match.group("gpu")) / 100.0,
                "cpu_cache_usage_perc": float(match.group("cpu")) / 100.0,
            }
        )
    for match in modern_stats_pattern.finditer(text):
        stats_samples.append(
            {
                "running": int(match.group("running")),
                "waiting": int(match.group("waiting")),
                # vLLM 0.10.1's v1 log line does not report swap or CPU-KV data.
                "swapped": None,
                "gpu_cache_usage_perc": float(match.group("gpu")) / 100.0,
                "cpu_cache_usage_perc": None,
            }
        )

    prefix_samples: List[Dict[str, Optional[float]]] = []
    for match in legacy_prefix_pattern.finditer(text):
        prefix_samples.append(
            {
                "gpu_prefix_hit_ratio": float(match.group("gpu")) / 100.0,
                "cpu_prefix_hit_ratio": float(match.group("cpu")) / 100.0,
            }
        )
    for match in modern_prefix_pattern.finditer(text):
        prefix_samples.append(
            {
                "gpu_prefix_hit_ratio": float(match.group("gpu")) / 100.0,
                # vLLM 0.10.1 reports a single aggregate prefix-cache ratio.
                "cpu_prefix_hit_ratio": None,
            }
        )

    physical_kv_samples: List[Dict[str, Any]] = []
    physical_kv_fail_closed_reasons: List[str] = []
    physical_kv_malformed_sample_count = 0
    physical_marker = "[sched_policy_patch:physical_kv]"
    integer_fields = {
        "num_gpu_blocks",
        "block_size",
        "capacity_tokens",
        "budget_tokens",
        "live_tokens",
        "logical_live_tokens",
        "running_growth_tokens",
        "reserved_tokens",
        "committed_tokens",
        "predicted_admit_tokens",
        "waiting",
        "running",
        "fit_admit",
        "admit",
        "effective_cap",
        "native_cap",
        "capacity_write_count",
        "rescue",
    }
    float_fields = {"target_utilization", "usage"}
    required_decision_fields = integer_fields | float_fields | {
        "decision",
        "reason",
        "capacity_write_source",
    }
    for line in text.splitlines():
        marker_index = line.find(physical_marker)
        if marker_index < 0:
            continue
        fields = dict(
            re.findall(
                r"([a-z][a-z0-9_]*)=([^\s]+)",
                line[marker_index + len(physical_marker):],
            )
        )
        decision_kind = fields.get("decision")
        if decision_kind == "fail_closed":
            physical_kv_fail_closed_reasons.append(
                fields.get("reason", "missing_reason")
            )
            continue
        if decision_kind != "admit" or not required_decision_fields.issubset(fields):
            physical_kv_malformed_sample_count += 1
            continue
        try:
            sample: Dict[str, Any] = {
                key: int(fields[key]) for key in integer_fields
            }
            sample.update({key: float(fields[key]) for key in float_fields})
        except (TypeError, ValueError, OverflowError):
            physical_kv_malformed_sample_count += 1
            continue
        sample.update(
            {
                "decision": decision_kind,
                "reason": fields["reason"],
                "capacity_write_source": fields["capacity_write_source"],
            }
        )
        valid = (
            sample["num_gpu_blocks"] > 0
            and sample["block_size"] > 0
            and sample["capacity_tokens"]
            == sample["num_gpu_blocks"] * sample["block_size"]
            and 0.0 < sample["target_utilization"] <= 1.0
            and 0.0 <= sample["usage"] <= 1.0
            and 0 <= sample["budget_tokens"] <= sample["capacity_tokens"]
            and 0 <= sample["live_tokens"] <= sample["capacity_tokens"]
            and sample["waiting"] >= 0
            and sample["running"] >= 0
            and sample["fit_admit"] >= 0
            and sample["admit"] >= sample["fit_admit"]
            and sample["effective_cap"]
            == min(
                sample["native_cap"],
                sample["running"] + sample["admit"],
            )
            and sample["native_cap"] > 0
            and sample["capacity_write_source"] == "physical_kv"
            and sample["capacity_write_count"] > 0
            and sample["rescue"] in {0, 1}
            and (
                # A non-rescue positive admission must fit the soft budget.
                # If existing running-growth forecasts already exceed that
                # budget, a zero-admit forecast_hold is the safe response: it
                # adds no KV exposure and must not be rejected as malformed.
                sample["rescue"] == 1
                or (
                    sample["admit"] == 0
                    and sample["predicted_admit_tokens"] == 0
                )
                or (
                    sample["admit"] > 0
                    and sample["committed_tokens"]
                    + sample["predicted_admit_tokens"]
                    <= sample["budget_tokens"]
                )
            )
            and (
                sample["rescue"] == 0
                or sample["live_tokens"]
                + sample["predicted_admit_tokens"]
                <= sample["capacity_tokens"]
            )
        )
        if not valid:
            physical_kv_malformed_sample_count += 1
            continue
        physical_kv_samples.append(sample)

    def avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    running_values = [sample["running"] for sample in stats_samples]
    waiting_values = [sample["waiting"] for sample in stats_samples]
    swapped_values = [
        sample["swapped"] for sample in stats_samples if sample["swapped"] is not None
    ]
    gpu_cache_values = [sample["gpu_cache_usage_perc"] for sample in stats_samples]
    cpu_cache_values = [
        sample["cpu_cache_usage_perc"]
        for sample in stats_samples
        if sample["cpu_cache_usage_perc"] is not None
    ]
    gpu_prefix_values = [sample["gpu_prefix_hit_ratio"] for sample in prefix_samples]
    cpu_prefix_values = [
        sample["cpu_prefix_hit_ratio"]
        for sample in prefix_samples
        if sample["cpu_prefix_hit_ratio"] is not None
    ]

    def numeric_summary(field: str) -> Dict[str, Optional[float]]:
        values = [float(sample[field]) for sample in physical_kv_samples]
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": avg(values),
        }

    effective_caps = {
        int(sample["effective_cap"]) for sample in physical_kv_samples
    }
    cap_changes = [
        int(after["effective_cap"]) - int(before["effective_cap"])
        for before, after in zip(physical_kv_samples, physical_kv_samples[1:])
    ]
    physical_kv_admission = {
        "sample_count": len(physical_kv_samples),
        "malformed_sample_count": physical_kv_malformed_sample_count,
        "fail_closed_count": len(physical_kv_fail_closed_reasons),
        "fail_closed_reasons": sorted(set(physical_kv_fail_closed_reasons)),
        "capacity_tokens": numeric_summary("capacity_tokens"),
        "target_utilization": numeric_summary("target_utilization"),
        "budget_tokens": numeric_summary("budget_tokens"),
        "usage": numeric_summary("usage"),
        "live_tokens": numeric_summary("live_tokens"),
        "logical_live_tokens": numeric_summary("logical_live_tokens"),
        "running_growth_tokens": numeric_summary("running_growth_tokens"),
        "reserved_tokens": numeric_summary("reserved_tokens"),
        "committed_tokens": numeric_summary("committed_tokens"),
        "predicted_admit_tokens": numeric_summary("predicted_admit_tokens"),
        "admit": numeric_summary("admit"),
        "effective_cap": {
            **numeric_summary("effective_cap"),
            "unique_count": len(effective_caps),
        },
        "native_cap": numeric_summary("native_cap"),
        "capacity_write_count": numeric_summary("capacity_write_count"),
        "effective_cap_increase_count": sum(
            1 for change in cap_changes if change > 0
        ),
        "effective_cap_decrease_count": sum(
            1 for change in cap_changes if change < 0
        ),
        "fit_admit_zero_sample_count": sum(
            1 for sample in physical_kv_samples if sample["fit_admit"] == 0
        ),
        "fit_admit_positive_sample_count": sum(
            1 for sample in physical_kv_samples if sample["fit_admit"] > 0
        ),
        "effective_cap_above_64_sample_count": sum(
            1 for sample in physical_kv_samples if sample["effective_cap"] > 64
        ),
        "running_above_64_sample_count": sum(
            1 for sample in physical_kv_samples if sample["running"] > 64
        ),
        "pressure_above_64_sample_count": sum(
            1
            for sample in physical_kv_samples
            if sample["running"] > 64
            and sample["waiting"] > 0
            and sample["effective_cap"] > 64
        ),
        "rescue_sample_count": sum(
            int(sample["rescue"]) for sample in physical_kv_samples
        ),
        "samples": physical_kv_samples,
    }
    physical_gate_checks = {
        "has_samples": len(physical_kv_samples) > 0,
        "no_malformed_samples": physical_kv_malformed_sample_count == 0,
        "no_fail_closed_decisions": not physical_kv_fail_closed_reasons,
        "stable_physical_capacity": (
            bool(physical_kv_samples)
            and len(
                {
                    int(sample["capacity_tokens"])
                    for sample in physical_kv_samples
                }
            )
            == 1
        ),
        "at_least_three_effective_caps": len(effective_caps) >= 3,
        "observed_cap_increase": any(change > 0 for change in cap_changes),
        "observed_cap_decrease": any(change < 0 for change in cap_changes),
        "observed_zero_fit_admit": any(
            sample["fit_admit"] == 0 for sample in physical_kv_samples
        ),
        "observed_positive_fit_admit": any(
            sample["fit_admit"] > 0 for sample in physical_kv_samples
        ),
        "at_least_ten_pressure_samples_above_64": (
            physical_kv_admission["pressure_above_64_sample_count"] >= 10
        ),
    }
    physical_kv_admission["screening_gates"] = {
        **physical_gate_checks,
        "passed": all(physical_gate_checks.values()),
    }

    return {
        "stats_sample_count": len(stats_samples),
        "prefix_sample_count": len(prefix_samples),
        "avg_running_requests": avg(running_values),
        "max_running_requests": max(running_values, default=0),
        "avg_waiting_requests": avg(waiting_values),
        "max_waiting_requests": max(waiting_values, default=0),
        "avg_swapped_requests": avg(swapped_values),
        "max_swapped_requests": max(swapped_values, default=0),
        "avg_log_gpu_cache_usage_perc": avg(gpu_cache_values),
        "max_log_gpu_cache_usage_perc": max(gpu_cache_values, default=0.0),
        "avg_log_cpu_cache_usage_perc": avg(cpu_cache_values),
        "max_log_cpu_cache_usage_perc": max(cpu_cache_values, default=0.0),
        "gpu_prefix_hit_ratio_avg": avg(gpu_prefix_values),
        "gpu_prefix_hit_ratio_max": max(gpu_prefix_values, default=0.0),
        "cpu_prefix_hit_ratio_avg": avg(cpu_prefix_values),
        "cpu_prefix_hit_ratio_max": max(cpu_prefix_values, default=0.0),
        "preemption_warning_count": len(re.findall(r"is preempted by .* mode", text)),
        "physical_kv_admission": physical_kv_admission,
    }


def load_swap_events(path: Path, start_s: float, end_s: float) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = float(payload.get("ts", 0.0))
            if start_s <= ts <= end_s:
                events.append(payload)
    return events


def summarize_swap_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    swap_in = [event for event in events if event.get("op") == "swap_in" and event.get("ok", True)]
    swap_out = [event for event in events if event.get("op") == "swap_out" and event.get("ok", True)]

    def avg_duration(items: List[Dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(float(item.get("duration_s", 0.0)) for item in items) / len(items)

    return {
        "swap_event_count": len(events),
        "swap_in_event_count": len(swap_in),
        "swap_out_event_count": len(swap_out),
        "swap_avg_time_s": avg_duration(events),
        "swap_in_avg_time_s": avg_duration(swap_in),
        "swap_out_avg_time_s": avg_duration(swap_out),
        "swap_total_time_s": sum(float(item.get("duration_s", 0.0)) for item in events),
        "swap_total_blocks": sum(int(item.get("mapping_len", 0)) for item in events),
    }


def compute_summary(
    workload_summary: Dict[str, Any],
    baseline_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
    request_events: List[Dict[str, Any]],
    speedup: float,
    vllm_log_summary: Dict[str, Any],
    swap_summary: Dict[str, Any],
    configured_max_request_attempts: int,
) -> Dict[str, Any]:
    queue_sum, queue_metric_sum = delta_metric_candidates(
        baseline_metrics,
        final_metrics,
        ["vllm:request_queue_time_seconds_sum"],
    )
    queue_count, queue_metric_count = delta_metric_candidates(
        baseline_metrics,
        final_metrics,
        ["vllm:request_queue_time_seconds_count"],
    )
    prompt_tokens, prompt_tokens_metric = delta_metric_candidates(
        baseline_metrics,
        final_metrics,
        ["vllm:request_prompt_tokens_sum", "vllm:prompt_tokens_total"],
    )
    num_preemptions, num_preemptions_metric = delta_metric_candidates(
        baseline_metrics,
        final_metrics,
        ["vllm:num_preemptions_total"],
    )

    success_count = sum(1 for event in request_events if event.get("ok"))
    failure_count = len(request_events) - success_count
    attempt_summary = _summarize_request_attempts(
        request_events,
        configured_max_request_attempts,
    )
    if attempt_summary["final_failure_count"] != failure_count:
        raise ValueError("request failure count disagrees with attempt histories")
    latencies = [event["latency_s"] for event in request_events if event.get("ok")]
    avg_latency_s = sum(latencies) / len(latencies) if latencies else 0.0
    # vLLM V1 normally resolves KV pressure by recompute preemption.  That is
    # materially different from moving KV blocks to CPU, so keep the two
    # signals separate.  Older summaries folded ``num_preemptions`` into the
    # swap boolean; readers retain an explicit compatibility path for those
    # immutable artifacts.
    preemption_total: Optional[float] = (
        num_preemptions if num_preemptions_metric is not None else None
    )
    preemption_happened: Optional[bool] = (
        preemption_total > 0 if preemption_total is not None else None
    )
    kv_swap_happened = bool(
        swap_summary.get("swap_event_count", 0) > 0
        or (vllm_log_summary.get("max_swapped_requests", 0) or 0) > 0
    )

    return {
        "speedup": speedup,
        "workload": workload_summary,
        "requests_total": len(request_events),
        "requests_success": success_count,
        "requests_failed": failure_count,
        **attempt_summary,
        "avg_request_latency_s": avg_latency_s,
        "avg_queue_time_s": (queue_sum / queue_count) if queue_count > 0 else 0.0,
        "queue_time_metric_sum": queue_metric_sum,
        "queue_time_metric_count": queue_metric_count,
        "prefix_hit_ratio": vllm_log_summary.get("gpu_prefix_hit_ratio_avg"),
        "gpu_prefix_hit_ratio_avg": vllm_log_summary.get("gpu_prefix_hit_ratio_avg"),
        "gpu_prefix_hit_ratio_max": vllm_log_summary.get("gpu_prefix_hit_ratio_max"),
        "cpu_prefix_hit_ratio_avg": vllm_log_summary.get("cpu_prefix_hit_ratio_avg"),
        "cpu_prefix_hit_ratio_max": vllm_log_summary.get("cpu_prefix_hit_ratio_max"),
        "prefix_log_sample_count": vllm_log_summary.get("prefix_sample_count", 0),
        "prompt_tokens_total": prompt_tokens,
        "prompt_tokens_metric": prompt_tokens_metric,
        "num_preemptions_total": preemption_total,
        "num_preemptions_metric": num_preemptions_metric,
        "preemption_happened": preemption_happened,
        "kv_swap_happened": kv_swap_happened,
        "kv_swap_happened_semantics": "cpu_swap_only_v2",
        "kv_swap_event_count": swap_summary.get("swap_event_count", 0),
        "kv_swap_avg_time_s": swap_summary.get("swap_avg_time_s", 0.0),
        "kv_swap_total_time_s": swap_summary.get("swap_total_time_s", 0.0),
        "kv_swap_total_blocks": swap_summary.get("swap_total_blocks", 0),
        "kv_swap_in_event_count": swap_summary.get("swap_in_event_count", 0),
        "kv_swap_out_event_count": swap_summary.get("swap_out_event_count", 0),
        "kv_swap_in_avg_time_s": swap_summary.get("swap_in_avg_time_s", 0.0),
        "kv_swap_out_avg_time_s": swap_summary.get("swap_out_avg_time_s", 0.0),
        "avg_swapped_requests": vllm_log_summary.get("avg_swapped_requests"),
        "max_swapped_requests": vllm_log_summary.get("max_swapped_requests"),
        "avg_log_gpu_cache_usage_perc": vllm_log_summary.get("avg_log_gpu_cache_usage_perc"),
        "max_log_gpu_cache_usage_perc": vllm_log_summary.get("max_log_gpu_cache_usage_perc"),
        "avg_log_cpu_cache_usage_perc": vllm_log_summary.get("avg_log_cpu_cache_usage_perc"),
        "max_log_cpu_cache_usage_perc": vllm_log_summary.get("max_log_cpu_cache_usage_perc"),
        "preemption_warning_count": vllm_log_summary.get("preemption_warning_count", 0),
        "physical_kv_admission": vllm_log_summary.get("physical_kv_admission"),
    }


def build_timeline(metrics_log_path: Path, experiment_start_s: float) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    with metrics_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "metrics" not in payload:
                continue
            metrics = payload["metrics"]
            samples.append(
                {
                    "t_s": payload["timestamp"] - experiment_start_s,
                    "running": metrics.get("vllm:num_requests_running", 0.0),
                    "waiting": metrics.get("vllm:num_requests_waiting", 0.0),
                    "gpu_cache_usage_perc": metrics.get(
                        "vllm:gpu_cache_usage_perc",
                        metrics.get("vllm:kv_cache_usage_perc", 0.0),
                    ),
                }
            )
    return {
        "samples": samples,
        "max_running": max((sample["running"] for sample in samples), default=0.0),
        "max_waiting": max((sample["waiting"] for sample in samples), default=0.0),
        "avg_running": (
            sum(sample["running"] for sample in samples) / len(samples) if samples else 0.0
        ),
        "avg_waiting": (
            sum(sample["waiting"] for sample in samples) / len(samples) if samples else 0.0
        ),
        "avg_gpu_cache_usage_perc": (
            sum(sample["gpu_cache_usage_perc"] for sample in samples) / len(samples) if samples else 0.0
        ),
        "max_gpu_cache_usage_perc": max(
            (sample["gpu_cache_usage_perc"] for sample in samples),
            default=0.0,
        ),
    }


async def run_trace_task(
    session: aiohttp.ClientSession,
    trace: Dict[str, Any],
    request_url: str,
    model: str,
    speedup: float,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    request_timeout_s: int,
    max_request_attempts: int,
    tool_wait_mode: str,
    request_log_path: Path,
    experiment_start_s: float,
    metadata_mode: str = "oracle",
    online_predictor: Optional[OnlineSessionPredictor] = None,
) -> List[Dict[str, Any]]:
    if max_request_attempts <= 0:
        raise ValueError("max_request_attempts must be positive")

    events: List[Dict[str, Any]] = []
    trace_id = trace["trace_id"]
    # Online output-length predictor: EMA over past calls' actual
    # completion_tokens in this trace. Cold-start uses an env-provided global
    # mean, but the scheduler-facing estimate is always bounded by the current
    # request's max_tokens budget.
    try:
        default_po = float(os.getenv("VLLM_SCHED_DEFAULT_PRED_OUT", "722"))
    except ValueError:
        default_po = 722.0
    try:
        ema_alpha = float(os.getenv("VLLM_SCHED_PRED_OUT_EMA_ALPHA", "0.5"))
    except ValueError:
        ema_alpha = 0.5
    enable_po = os.getenv("VLLM_SCHED_PRED_OUT_ENABLE", "1").strip() in {"1", "true", "True"}
    po_ema: float = default_po
    previous_requests: List[Dict[str, Any]] = []
    for request_index, request in enumerate(trace["requests"]):
        wait_s = float(request["wait_after_prev_s"])
        # The first request's wait is an absolute session-arrival offset.  It
        # must not be compressed by the independent intra-session tool speedup.
        scaled_wait_s = wait_s if request_index == 0 else _scaled_tool_wait_s(wait_s, speedup)
        if tool_wait_mode == "sleep" and scaled_wait_s > 0:
            await asyncio.sleep(scaled_wait_s)

        predicted_output_tokens: Optional[float] = None
        if enable_po:
            predicted_output_tokens = po_ema
            if metadata_mode == "online":
                predicted_output_tokens = _bounded_online_output_prediction(
                    po_ema,
                    int(request["max_tokens"]),
                )

        sched_request_id, sched_meta = _build_sched_request_id(
            trace=trace,
            request=request,
            request_index=request_index,
            speedup=speedup,
            predicted_output_tokens=predicted_output_tokens,
            metadata_mode=metadata_mode,
            online_predictor=online_predictor,
            previous_requests=previous_requests,
        )

        payload = {
            "model": model,
            "messages": request["messages"],
            "stop": ["\n<tool_response>", "<tool_response>"],
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "max_tokens": int(request["max_tokens"]),
            "request_id": sched_request_id,
        }

        request_start_s = time.time()
        base_event = {
            "trace_id": trace_id,
            "source_trace": trace["source_trace"],
            "duplicated": trace["duplicated"],
            "prefix_char": trace["prefix_char"],
            "call_index": request["call_index"],
            "scheduled_wait_s": scaled_wait_s,
            "tool_wait_mode": tool_wait_mode,
            "scheduled_wait_original_s": request.get(
                "wait_after_prev_original_s",
                request["wait_after_prev_s"],
            ),
            "tool_overlap_saved_s": request.get("tool_overlap_saved_s", 0.0),
            "tool_overlap_window_s": request.get("tool_overlap_window_s", 0.0),
            "tool_kind_before": request.get("tool_kind_before", ""),
            "tool_cache_hit": request.get("tool_cache_hit", False),
            "tool_overlap_mode": request.get("tool_overlap_mode", "none"),
            "tool_prediction_candidate_count": request.get(
                "tool_prediction_candidate_count", 0
            ),
            "tool_prediction_exact_hits": request.get(
                "tool_prediction_exact_hits", 0
            ),
            "tool_prediction_waste": request.get("tool_prediction_waste", 0),
            "tool_prediction_artifact_sha256": request.get(
                "tool_prediction_artifact_sha256", ""
            ),
            "tool_prediction_top_k": request.get("tool_prediction_top_k", 0),
            "prompt_tokens": request["prompt_tokens"],
            "target_output_tokens": request.get("target_output_tokens", request["max_tokens"]),
            "max_tokens": request["max_tokens"],
            "truncated": request["truncated"],
            "request_start_offset_s": request_start_s - experiment_start_s,
            "request_id": sched_request_id,
            "metadata_source": sched_meta["metadata_source"],
            "oracle_next_tool_wait_s": sched_meta.get("nw_oracle"),
            "oracle_remaining_tool_wait_s": sched_meta.get("rtw_oracle"),
            "oracle_remaining_calls_after": (
                sched_meta["rc"] if sched_meta["metadata_source"] == "oracle" else None
            ),
            "oracle_total_calls": (
                sched_meta["n"] if sched_meta["metadata_source"] == "oracle" else None
            ),
            "scheduled_remaining_calls_after": sched_meta["rc"],
            "scheduled_total_calls": sched_meta["n"],
            "scheduled_nw": sched_meta["nw"],
            "scheduled_nw_reliability": sched_meta.get("nwc"),
            "scheduled_rtw": sched_meta["rtw"],
            "nw_source": sched_meta.get("nw_src", "oracle"),
        }

        attempt = 0
        event: Dict[str, Any] = dict(base_event)
        attempt_history: List[Dict[str, Any]] = []
        while attempt < max_request_attempts:
            attempt += 1
            event = dict(base_event)
            event["attempts"] = attempt
            attempt_started_s = time.monotonic()
            attempt_record: Dict[str, Any] = {
                "attempt": attempt,
                "transport": "http",
                "outcome": "unknown",
                "http_status": None,
                "error_type": None,
                "error": None,
                "duration_s": 0.0,
                "retryable": False,
                "will_retry": False,
                "retry_backoff_s": 0.0,
                "delivery_ambiguous": False,
            }
            try:
                async with session.post(
                    request_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=request_timeout_s),
                ) as response:
                    event["http_status"] = response.status
                    attempt_record["http_status"] = response.status
                    event["ok"] = response.status == 200
                    if response.status == 200:
                        body = await response.json(content_type=None)
                        attempt_record["outcome"] = "success"
                        content = (
                            body.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        usage = body.get("usage", {})
                        event["response_chars"] = len(content)
                        event["usage"] = usage
                        # Update per-trace EMA on actual completion tokens.
                        actual_out = int(usage.get("completion_tokens", 0) or 0)
                        if actual_out > 0:
                            po_ema = ema_alpha * actual_out + (1.0 - ema_alpha) * po_ema
                        event["po_predicted"] = sched_meta.get("po")
                        event["po_actual"] = actual_out
                    else:
                        # HTTP failures, including 429/5xx, are experimental
                        # service outcomes and must not be hidden by retries.
                        attempt_record["outcome"] = "http_error"
                        try:
                            body = await response.json(content_type=None)
                        except Exception as body_exc:
                            body = {
                                "response_body_unavailable": True,
                                "parse_error_type": type(body_exc).__name__,
                            }
                        event["error_body"] = body
            except asyncio.TimeoutError as exc:
                # A POST timeout may occur after server-side work began.  It is
                # not safe to retry automatically without idempotency support.
                event["ok"] = False
                event["error"] = repr(exc)
                attempt_record.update(
                    {
                        "transport": "timeout",
                        "outcome": "timeout",
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                        "delivery_ambiguous": True,
                    }
                )
            except aiohttp.ClientConnectionError as exc:
                # Only explicit, non-timeout aiohttp connection failures may
                # be retried.  Reuse the scheduler request ID: changing its
                # wire format risks breaking deployed scheduler metadata.
                event["ok"] = False
                event["error"] = repr(exc)
                retryable = _is_retryable_transport_error(exc)
                will_retry = retryable and attempt < max_request_attempts
                backoff_s = (
                    min(4.0, float(2 ** (attempt - 1))) if will_retry else 0.0
                )
                attempt_record.update(
                    {
                        "transport": "aiohttp_connection",
                        "outcome": "transport_error",
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                        "retryable": retryable,
                        "will_retry": will_retry,
                        "retry_backoff_s": backoff_s,
                        "delivery_ambiguous": _transport_delivery_is_ambiguous(exc),
                    }
                )
            except aiohttp.ClientPayloadError as exc:
                # The response body broke after the request may have executed.
                # Keep it visible and do not risk a duplicate POST.
                event["ok"] = False
                event["error"] = repr(exc)
                attempt_record.update(
                    {
                        "transport": "aiohttp_payload",
                        "outcome": "response_error",
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                        "delivery_ambiguous": True,
                    }
                )
            except aiohttp.ClientError as exc:
                event["ok"] = False
                event["error"] = repr(exc)
                attempt_record.update(
                    {
                        "transport": "aiohttp_client",
                        "outcome": "client_error",
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    }
                )
            except Exception as exc:
                event["ok"] = False
                event["error"] = repr(exc)
                attempt_record.update(
                    {
                        "transport": "exception",
                        "outcome": "unexpected_error",
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    }
                )
            finally:
                attempt_record["duration_s"] = max(
                    0.0,
                    time.monotonic() - attempt_started_s,
                )
                attempt_history.append(attempt_record)

            if not attempt_record["will_retry"]:
                break
            await asyncio.sleep(float(attempt_record["retry_backoff_s"]))

        event["attempts"] = attempt
        event["attempt_history"] = attempt_history
        event["request_end_offset_s"] = time.time() - experiment_start_s
        event["latency_s"] = event["request_end_offset_s"] - event["request_start_offset_s"]
        _json_dump_line(request_log_path, event)
        events.append(event)
        previous_requests.append(request)
    return events


async def main_async(args: argparse.Namespace) -> int:
    if args.tool_prediction_top_k <= 0:
        raise ValueError("--tool-prediction-top-k must be positive")
    if args.max_request_attempts <= 0:
        raise ValueError("--max-request-attempts must be positive")
    if args.tool_overlap_mode == "learned" and not args.tool_prediction_model:
        raise ValueError(
            "--tool-prediction-model is required when --tool-overlap-mode=learned"
        )
    if args.azure_arrival_speedup <= 0:
        raise ValueError("--azure-arrival-speedup must be positive")
    if args.azure_max_sessions is not None and args.azure_max_sessions <= 0:
        raise ValueError("--azure-max-sessions must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_log_path = output_dir / "request_events.jsonl"
    metrics_log_path = output_dir / "metrics_samples.jsonl"
    summary_path = output_dir / "summary.json"
    timeline_path = output_dir / "timeline.json"
    workload_path = output_dir / "prepared_workload.json"
    vllm_log_summary_path = output_dir / "vllm_log_summary.json"
    swap_summary_path = output_dir / "swap_summary.json"

    for path in [
        request_log_path,
        metrics_log_path,
        summary_path,
        timeline_path,
        vllm_log_summary_path,
        swap_summary_path,
    ]:
        if path.exists():
            path.unlink()

    if args.prepared_workload:
        workload = load_workload(args.prepared_workload)
        prepared_overlap_mode = workload.get("meta", {}).get(
            "tool_overlap_mode", "none"
        )
        if args.tool_overlap_mode == "learned" or prepared_overlap_mode == "learned":
            validate_learned_workload_artifact(
                workload, args.tool_prediction_model
            )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        workload = prepare_trace_workload(
            trace_dir=args.trace_dir,
            tokenizer=tokenizer,
            target_trace_count=args.trace_count,
            max_model_len=args.max_model_len,
            max_output_tokens_cap=args.max_output_tokens_cap,
            min_output_tokens_floor=args.min_output_tokens_floor,
            output_token_buffer=args.output_token_buffer,
            duplicate_seed=args.seed,
            tool_overlap_mode=args.tool_overlap_mode,
            tool_overlap_efficiency=args.tool_overlap_efficiency,
            prefix_marker_mode=args.prefix_marker_mode,
            tool_prediction_model=args.tool_prediction_model,
            tool_prediction_top_k=args.tool_prediction_top_k,
        )

    if args.azure_arrival_trace:
        invocations = load_azure_llm_invocations(
            args.azure_arrival_trace,
            start_time=args.azure_start_time,
            duration_s=args.azure_duration_s,
            max_sessions=(
                args.azure_max_sessions
                if args.azure_max_sessions is not None
                else args.trace_count
            ),
        )
        workload = apply_azure_arrivals(
            workload,
            invocations,
            source_file=args.azure_arrival_trace,
            dataset_variant=args.azure_dataset_variant,
            arrival_speedup=args.azure_arrival_speedup,
            mapping=args.azure_session_mapping,
            mapping_seed=args.seed,
        )

    if args.max_arrival_time_s is not None:
        workload = cap_workload_by_arrival_time(
            workload=workload,
            max_arrival_time_s=args.max_arrival_time_s,
        )

    save_workload(workload, workload_path)

    workload_summary = summarize_workload(workload)
    with (output_dir / "workload_summary.json").open("w", encoding="utf-8") as f:
        json.dump(workload_summary, f, ensure_ascii=False, indent=2)

    if args.prepare_only:
        print(json.dumps({"prepared_only": True, "workload": workload_summary}, ensure_ascii=False, indent=2))
        return 0

    if args.scheduler_metadata_mode == "online" and args.scheduler_calibration_workload:
        calibration_path = Path(args.scheduler_calibration_workload).resolve()
        replay_paths = {workload_path.resolve()}
        if args.prepared_workload:
            replay_paths.add(Path(args.prepared_workload).resolve())
        if calibration_path in replay_paths:
            raise ValueError(
                "online scheduler calibration must be separate from the replay workload"
            )
        calibration_workload = load_workload(calibration_path)
        _validate_disjoint_calibration_sessions(workload, calibration_workload)
    _maybe_init_predictor(
        args.scheduler_metadata_mode,
        args.scheduler_calibration_workload,
    )

    server_url = args.server_url.rstrip("/")
    request_url = f"{server_url}/v1/chat/completions"
    metrics_url = f"{server_url}/metrics"
    vllm_log_path = Path(args.vllm_log_file)
    swap_events_path = Path(args.swap_events_file)
    vllm_log_start_offset = vllm_log_path.stat().st_size if vllm_log_path.exists() else 0

    headers = _client_headers(os.environ.get("VLLM_API_KEY"))
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        baseline_snapshot = await fetch_metrics(session, metrics_url)
        stop_event = asyncio.Event()
        metrics_task = asyncio.create_task(
            scrape_metrics(
                session=session,
                metrics_url=metrics_url,
                stop_event=stop_event,
                output_path=metrics_log_path,
                interval_s=args.metrics_scrape_interval_s,
            )
        )

        experiment_start_s = time.time()
        trace_semaphore = (
            asyncio.Semaphore(args.max_active_traces)
            if args.max_active_traces is not None and args.max_active_traces > 0
            else None
        )

        async def run_maybe_gated(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
            if trace_semaphore is None:
                return await run_trace_task(
                    session=session,
                    trace=trace,
                    request_url=request_url,
                    model=args.model,
                    speedup=args.speedup,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    request_timeout_s=args.request_timeout_s,
                    max_request_attempts=args.max_request_attempts,
                    tool_wait_mode=args.tool_wait_mode,
                    request_log_path=request_log_path,
                    experiment_start_s=experiment_start_s,
                    metadata_mode=args.scheduler_metadata_mode,
                )
            async with trace_semaphore:
                return await run_trace_task(
                    session=session,
                    trace=trace,
                    request_url=request_url,
                    model=args.model,
                    speedup=args.speedup,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    request_timeout_s=args.request_timeout_s,
                    max_request_attempts=args.max_request_attempts,
                    tool_wait_mode=args.tool_wait_mode,
                    request_log_path=request_log_path,
                    experiment_start_s=experiment_start_s,
                    metadata_mode=args.scheduler_metadata_mode,
                )

        task_futures = [
            asyncio.create_task(run_maybe_gated(trace))
            for trace in workload["traces"]
        ]

        gathered = await asyncio.gather(*task_futures)
        stop_event.set()
        await asyncio.sleep(args.metrics_scrape_interval_s)
        await metrics_task
        final_snapshot = await fetch_metrics(session, metrics_url)
        experiment_end_s = time.time()

    request_events = [item for group in gathered for item in group]
    vllm_log_end_offset = vllm_log_path.stat().st_size if vllm_log_path.exists() else vllm_log_start_offset
    vllm_log_summary = parse_vllm_log_segment(
        _read_text_segment(vllm_log_path, vllm_log_start_offset, vllm_log_end_offset)
    )
    swap_summary = summarize_swap_events(
        load_swap_events(swap_events_path, experiment_start_s, experiment_end_s)
    )
    summary = compute_summary(
        workload_summary=workload_summary,
        baseline_metrics=baseline_snapshot["metrics"],
        final_metrics=final_snapshot["metrics"],
        request_events=request_events,
        speedup=args.speedup,
        vllm_log_summary=vllm_log_summary,
        swap_summary=swap_summary,
        configured_max_request_attempts=args.max_request_attempts,
    )
    summary["experiment_wall_time_s"] = experiment_end_s - experiment_start_s
    summary["experiment_start_s"] = experiment_start_s
    summary["experiment_end_s"] = experiment_end_s
    summary["baseline_metrics_timestamp"] = baseline_snapshot["timestamp"]
    summary["final_metrics_timestamp"] = final_snapshot["timestamp"]
    timeline = build_timeline(metrics_log_path, experiment_start_s)
    summary["timeline_max_running"] = timeline["max_running"]
    summary["timeline_max_waiting"] = timeline["max_waiting"]
    summary["timeline_avg_running"] = timeline["avg_running"]
    summary["timeline_avg_waiting"] = timeline["avg_waiting"]
    summary["timeline_avg_gpu_cache_usage_perc"] = timeline["avg_gpu_cache_usage_perc"]
    summary["timeline_max_gpu_cache_usage_perc"] = timeline["max_gpu_cache_usage_perc"]
    summary["max_active_traces"] = args.max_active_traces
    summary["tool_wait_mode"] = args.tool_wait_mode
    summary["scheduler_metadata_mode"] = args.scheduler_metadata_mode
    summary["metadata_source"] = args.scheduler_metadata_mode
    summary["scheduler_calibration_workload"] = args.scheduler_calibration_workload
    summary["scheduler_environment"] = {
        key: os.environ[key]
        for key in _SCHEDULER_ENV_KEYS
        if key in os.environ
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with timeline_path.open("w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    with vllm_log_summary_path.open("w", encoding="utf-8") as f:
        json.dump(vllm_log_summary, f, ensure_ascii=False, indent=2)
    with swap_summary_path.open("w", encoding="utf-8") as f:
        json.dump(swap_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if int(summary.get("requests_failed", 0)) else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
