#!/usr/bin/env python3
"""Run the 0.42x trace against live vLLM with real all-visit speculation.

Search and non-Visit tools use their corrected trace duration as an actual
``asyncio.sleep``.  Visit URLs execute as cancellable wall-clock jobs in one
shared, authority-first pool.  In the treatment, generalized all-visit OOF
predictions are submitted after each completed tool and can be completed,
promoted in flight, or preempted when authority arrives.

The LLM responses do not choose the recorded next call: this is a strict trace
replay, just like the repository's original vLLM runner.  The live vLLM server
does perform every recorded LLM turn, while the tool timeline is driven by the
recorded authoritative call graph.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
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
ROOT_SCRIPTS = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(ROOT_SCRIPTS))

from paste_repro.trace_coscheduler import (  # noqa: E402
    AdmissionTurn,
    AsyncPreemptibleVisitPool,
    GainPressureAdmissionController,
)
from paste_repro.traces import LLMCall, ToolCall, load_sessions  # noqa: E402
from run_pattern_cache_evaluation import cv_fold  # noqa: E402
from run_pattern_v2_trace_all_visit_shared_capacity import (  # noqa: E402
    candidate_policy_windows,
    prepare_sessions,
)
from run_pattern_v2_trace_all_visit_wall import (  # noqa: E402
    collect_all_visit_timings,
    collect_nested_oof_all_visit_windows,
    executable_url,
    trace_llm_scale_metadata,
    visit_coverage_audit,
    visit_urls,
)
from run_pattern_v2_trace_multi_spec_wall import (  # noqa: E402
    candidate_value,
    select_per_task_candidates,
    session_full_walls,
)
from run_pattern_v2_trace_timing_net_benefit import (  # noqa: E402
    build_oof_service_estimates,
    sha256_file,
)
from trace_experiment_lib import prepare_trace_workload  # noqa: E402


SCHEMA = "paste_repro.trace_all_visit_live.v1"
PLAN_SCHEMA = "paste_repro.trace_all_visit_live_plan.v1"
DEFAULT_TRACES = (
    REPOSITORY_ROOT
    / "traces/my_traces_tool_slo_search_uniform_1_3s_"
    "visit_serial_uniform_2_8s_llm_x0_42"
)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def canonical_hash(payload: Any) -> str:
    wire = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _tool_duration(call: ToolCall) -> tuple[float, tuple[float, ...]]:
    correction = call.timing_correction or {}
    total = correction.get("duration_s", 0.0)
    units = correction.get("unit_duration_s", [])
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        total = 0.0
    if not isinstance(units, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in units
    ):
        units = []
    return max(0.0, float(total)), tuple(max(0.0, float(v)) for v in units)


def _oof_progress_forecasts(sessions: Sequence[Any]) -> dict[str, dict[str, Any]]:
    totals = {
        session.session_id: sum(isinstance(event, LLMCall) for event in session.events)
        for session in sessions
    }
    tool_durations: dict[str, list[float]] = defaultdict(list)
    visit_rates: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        for event in session.events:
            if not isinstance(event, ToolCall):
                continue
            duration, _ = _tool_duration(event)
            if duration > 0.0:
                tool_durations[session.session_id].append(duration)
            visit_rates[session.session_id].append(float(event.tool_name == "visit"))

    result: dict[str, dict[str, Any]] = {}
    for session in sessions:
        fold = cv_fold(session.session_id)
        training_ids = [sid for sid in totals if cv_fold(sid) != fold]
        training_totals = [totals[sid] for sid in training_ids]
        mean_total = statistics.fmean(training_totals)
        durations = [value for sid in training_ids for value in tool_durations[sid]]
        labels = [value for sid in training_ids for value in visit_rates[sid]]
        result[session.session_id] = {
            "outer_fold": fold,
            "predicted_total_calls": mean_total,
            "mean_tool_service_s": statistics.fmean(durations),
            "visit_probability": statistics.fmean(labels),
            "training_sessions": len(training_ids),
        }
    return result


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    sessions = load_sessions(args.traces)
    tokenizer_source = args.tokenizer
    if "/" in tokenizer_source and not Path(tokenizer_source).exists():
        cache_root = Path(os.getenv("HF_HOME", Path.home() / "hf_cache"))
        snapshots = sorted(
            (
                cache_root / f"models--{tokenizer_source.replace('/', '--')}" / "snapshots"
            ).glob("*")
        )
        if len(snapshots) == 1:
            tokenizer_source = str(snapshots[0])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True, local_files_only=True
    )
    prepared_workload = prepare_trace_workload(
        trace_dir=args.traces,
        tokenizer=tokenizer,
        target_trace_count=len(sessions),
        max_model_len=args.max_model_len,
        max_output_tokens_cap=args.max_output_tokens_cap,
        min_output_tokens_floor=args.min_output_tokens_floor,
        output_token_buffer=args.output_token_buffer,
        duplicate_seed=args.seed,
        tool_overlap_mode="none",
    )
    windows, predictor_summary, decisions = collect_nested_oof_all_visit_windows(
        args.traces,
        candidate_pool_size=args.candidate_pool_size,
        selector_model=args.selector_model,
    )
    timings = collect_all_visit_timings(
        args.traces, decisions, llm_duration_scale=1.0
    )
    service_estimates, service_summary = build_oof_service_estimates(
        windows, timings, domain_prior_strength=args.domain_prior_strength
    )
    selected_windows, selected_width = candidate_policy_windows(
        windows, service_estimates, candidate_policy=args.candidate_policy
    )
    full_walls = session_full_walls(args.traces, llm_duration_scale=1.0)
    prepared_sessions = prepare_sessions(
        args.traces,
        windows,
        decisions,
        timings,
        service_estimates,
        full_walls,
        candidate_policy=args.candidate_policy,
    )
    epochs = {
        epoch.decision_id: epoch
        for session in prepared_sessions
        for epoch in session.epochs
    }
    window_by_id = {window.decision_id: window for window in selected_windows}
    selected_by_id = {
        window.decision_id: select_per_task_candidates(
            window,
            service_estimates[window.decision_id],
            per_task_width=selected_width,
            coordination_cost_s=args.coordination_cost_ms / 1000.0,
        )
        for window in selected_windows
    }
    decision_by_trigger = {
        (decision.session_id, decision.trigger_event_index): decision
        for decision in decisions
    }
    prepared_by_source = {
        Path(trace["source_trace"]).name: trace
        for trace in prepared_workload["traces"]
    }
    forecasts = _oof_progress_forecasts(sessions)
    trace_rows: list[dict[str, Any]] = []
    for session in sessions:
        prepared = prepared_by_source[session.session_id]
        requests = {
            int(request["call_index"]): request for request in prepared["requests"]
        }
        steps: list[dict[str, Any]] = []
        event_list = list(session.events)
        for event_index, event in enumerate(event_list):
            if not isinstance(event, LLMCall):
                continue
            request = requests[event.call_index]
            tools_after: list[dict[str, Any]] = []
            cursor = event_index + 1
            while cursor < len(event_list) and not isinstance(
                event_list[cursor], LLMCall
            ):
                tool = event_list[cursor]
                if isinstance(tool, ToolCall):
                    duration_s, unit_durations_s = _tool_duration(tool)
                    urls = visit_urls(tool) if tool.tool_name == "visit" else ()
                    executable = [
                        (url, unit_durations_s[index])
                        for index, url in enumerate(urls)
                        if executable_url(url) and index < len(unit_durations_s)
                    ]
                    tool_row: dict[str, Any] = {
                        "event_index": cursor,
                        "call_index": tool.call_index,
                        "tool_name": tool.tool_name,
                        "duration_s": duration_s,
                        "visit_units": [
                            {"url": url, "duration_s": service_s}
                            for url, service_s in executable
                        ],
                        "speculation": None,
                    }
                    decision = decision_by_trigger.get((session.session_id, cursor))
                    if decision is not None:
                        epoch = epochs.get(decision.decision_id)
                        selected = selected_by_id[decision.decision_id]
                        if epoch is None or len(selected) != len(
                            epoch.candidate_services_s
                        ):
                            raise RuntimeError(
                                f"candidate service mismatch: {decision.decision_id}"
                            )
                        estimate = service_estimates[decision.decision_id]
                        candidate_rows = [
                            {
                                "url": candidate.pattern.url,
                                "score": candidate.exact_probability,
                                "duration_s": service_s,
                                "expected_overlap_s": estimate.overlap_for_url(
                                    candidate.pattern.url
                                ),
                            }
                            for candidate, service_s in zip(
                                selected,
                                epoch.candidate_services_s,
                                strict=True,
                            )
                        ]
                        window = window_by_id[decision.decision_id]
                        tool_row["speculation"] = {
                            "decision_id": decision.decision_id,
                            "trigger_tool": decision.trigger_tool,
                            "expected_authoritative_calls": (
                                window.expected_authoritative_calls
                            ),
                            "candidates": candidate_rows,
                            "expected_gain_s": sum(
                                row["score"] * row["expected_overlap_s"]
                                for row in candidate_rows
                            ),
                        }
                    tools_after.append(tool_row)
                cursor += 1
            steps.append({"request": request, "tools_after": tools_after})
        trace_rows.append(
            {
                "trace_id": prepared["trace_id"],
                "session_id": session.session_id,
                "source_trace": str(session.path.resolve()),
                "forecast": forecasts[session.session_id],
                "steps": steps,
            }
        )

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "candidate_pool_size": args.candidate_pool_size,
            "selector_model": args.selector_model,
            "candidate_policy": args.candidate_policy,
            "coordination_cost_ms": args.coordination_cost_ms,
            "domain_prior_strength": args.domain_prior_strength,
            "max_model_len": args.max_model_len,
            "max_output_tokens_cap": args.max_output_tokens_cap,
            "min_output_tokens_floor": args.min_output_tokens_floor,
            "output_token_buffer": args.output_token_buffer,
            "seed": args.seed,
        },
        "trace_scale": trace_llm_scale_metadata(args.traces),
        "coverage": visit_coverage_audit(args.traces, decisions),
        "predictor": predictor_summary,
        "service_estimator": service_summary,
        "traces": trace_rows,
    }
    plan["plan_sha256"] = canonical_hash(plan)
    return plan


def _parse_metrics(text: str) -> dict[str, float]:
    from prometheus_client.parser import text_string_to_metric_families

    result: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            result[sample.name] = result.get(sample.name, 0.0) + float(sample.value)
    return result


async def fetch_metrics(session: aiohttp.ClientSession, url: str) -> dict[str, float]:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
        response.raise_for_status()
        return _parse_metrics(await response.text())


def _request_id(meta: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(meta), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"schedx{encoded.hex()}z"


async def _post_llm(
    session: aiohttp.ClientSession,
    *,
    request_url: str,
    model: str,
    request: Mapping[str, Any],
    request_id: str,
    timeout_s: float,
) -> tuple[int, dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": request["messages"],
        "stop": ["\n<tool_response>", "<tool_response>"],
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "max_tokens": int(request["max_tokens"]),
        "request_id": request_id,
    }
    async with session.post(
        request_url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as response:
        body = await response.json(content_type=None)
        if response.status != 200:
            raise RuntimeError(f"vLLM HTTP {response.status}: {body}")
    choice = body.get("choices", [{}])[0]
    return response.status, dict(body.get("usage", {})), str(
        choice.get("message", {}).get("content", "")
    )


async def execute_plan(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("prepared plan has an unsupported schema")
    expected_hash = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if expected_hash != canonical_hash(unsigned):
        raise ValueError("prepared plan checksum mismatch")

    treatment = args.mode == "coscheduled_speculation"
    visit_pool = AsyncPreemptibleVisitPool(
        capacity=args.visit_capacity,
        speculative_cap=(args.speculative_cap if treatment else 0),
    )
    admission = (
        GainPressureAdmissionController(
            pressure_low=args.pressure_low,
            pressure_high=args.pressure_high,
            cold_session_cap=args.cold_session_cap,
            gain_weight=args.gain_weight,
            aging_weight=args.aging_weight,
            kv_weight=args.kv_weight,
            context_ref_tokens=args.context_ref_tokens,
        )
        if treatment and args.admission_backend == "python_gain_pressure"
        else None
    )
    task_gate = asyncio.Semaphore(args.max_active_tasks)
    request_events: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    list_lock = asyncio.Lock()
    experiment_started = time.monotonic()

    async def run_trace(trace: Mapping[str, Any]) -> None:
        release_offset_s = float(trace.get("release_offset_s", 0.0))
        if not math.isfinite(release_offset_s) or release_offset_s < 0.0:
            raise ValueError(
                f"invalid release_offset_s for {trace.get('trace_id')}: "
                f"{release_offset_s!r}"
            )
        scheduled_release = experiment_started + release_offset_s
        release_sleep_s = scheduled_release - time.monotonic()
        if release_sleep_s > 0.0:
            await asyncio.sleep(release_sleep_s)
        released_at = time.monotonic()
        # Flow time is defined from workload release, so time waiting for the
        # client-side concurrency semaphore must not disappear from the task
        # metric.  The old quick runner started the clock inside this block and
        # omitted the wait for sessions beyond max_active_tasks.
        task_started = scheduled_release
        async with task_gate:
            task_gate_acquired = time.monotonic()
            session_id = str(trace["session_id"])
            trace_id = str(trace["trace_id"])
            forecast = trace["forecast"]
            po_ema = float(os.getenv("VLLM_SCHED_DEFAULT_PRED_OUT", "128"))
            observed_tool_s: list[float] = []
            realized_gain_s = 0.0
            future_gain_s = 0.0
            next_tool_wait_s = 0.0
            next_tool_probability = 0.0
            trace_llm_s = 0.0
            trace_admission_s = 0.0
            trace_search_s = 0.0
            trace_other_tool_s = 0.0
            trace_visit_exposed_s = 0.0
            trace_saved_s = 0.0
            failure: str | None = None
            try:
                for request_index, step in enumerate(trace["steps"]):
                    request = step["request"]
                    predicted_total = float(forecast["predicted_total_calls"])
                    remaining_calls = max(
                        0, int(round(predicted_total - request_index - 1))
                    )
                    rolling_tool_s = (
                        statistics.fmean(observed_tool_s)
                        if observed_tool_s
                        else float(forecast["mean_tool_service_s"])
                    )
                    remaining_tool_s = (
                        next_tool_wait_s
                        + max(0, remaining_calls - 1)
                        * rolling_tool_s
                        * float(forecast["visit_probability"])
                    )
                    predicted_service_s = (
                        int(request["prompt_tokens"])
                        / float(os.getenv("VLLM_SCHED_PREFILL_TOKENS_PER_S_V2", "38112"))
                        + max(1.0, po_ema)
                        / float(os.getenv("VLLM_SCHED_DECODE_TOKENS_PER_S_V2", "113.7"))
                    )
                    sched_meta = {
                        "t": trace_id,
                        "c": int(request["call_index"]),
                        "i": request_index,
                        "n": request_index + 1 + remaining_calls,
                        "rc": remaining_calls,
                        "nw": next_tool_wait_s,
                        "nwc": next_tool_probability,
                        "rtw": remaining_tool_s,
                        "pt": int(request["prompt_tokens"]),
                        "mt": int(request["max_tokens"]),
                        "po": int(max(1, min(po_ema, int(request["max_tokens"])))),
                        "eg": realized_gain_s + future_gain_s,
                        "ms": "causal_oof_all_visit",
                    }
                    request_id = _request_id(sched_meta)
                    admission_started = time.monotonic()
                    if admission is not None:
                        await admission.acquire(
                            AdmissionTurn(
                                session_id=session_id,
                                cold=request_index == 0,
                                exposed_tool_gain_s=(
                                    realized_gain_s + future_gain_s
                                ),
                                predicted_llm_service_s=predicted_service_s,
                                context_tokens=int(request["prompt_tokens"]),
                                kv_load=(
                                    int(request["prompt_tokens"])
                                    + int(max(1, min(po_ema, int(request["max_tokens"]))))
                                ) / args.context_ref_tokens,
                            )
                        )
                    admitted_s = time.monotonic()
                    trace_admission_s += admitted_s - admission_started
                    llm_started = admitted_s
                    try:
                        status, usage, content = await _post_llm(
                            http_session,
                            request_url=request_url,
                            model=args.model,
                            request=request,
                            request_id=request_id,
                            timeout_s=args.request_timeout_s,
                        )
                    finally:
                        if admission is not None:
                            await admission.release(session_id)
                    llm_finished = time.monotonic()
                    llm_s = llm_finished - llm_started
                    trace_llm_s += llm_s
                    actual_out = int(usage.get("completion_tokens", 0) or 0)
                    if actual_out > 0:
                        po_ema = 0.5 * actual_out + 0.5 * po_ema
                    async with list_lock:
                        request_events.append(
                            {
                                "trace_id": trace_id,
                                "session_id": session_id,
                                "request_index": request_index,
                                "call_index": request["call_index"],
                                "request_id": request_id,
                                "http_status": status,
                                "latency_s": llm_s,
                                "admission_wait_s": admitted_s - admission_started,
                                "start_offset_s": llm_started - experiment_started,
                                "end_offset_s": llm_finished - experiment_started,
                                "prompt_tokens": request["prompt_tokens"],
                                "max_tokens": request["max_tokens"],
                                "usage": usage,
                                "response_chars": len(content),
                                "scheduler_metadata": sched_meta,
                            }
                        )

                    realized_gain_s = 0.0
                    future_gain_s = 0.0
                    next_tool_wait_s = 0.0
                    next_tool_probability = 0.0
                    for tool in step["tools_after"]:
                        tool_started = time.monotonic()
                        visit_results: list[dict[str, Any]] = []
                        tool_name = str(tool["tool_name"])
                        if tool_name == "visit":
                            for unit in tool["visit_units"]:
                                result = await visit_pool.authoritative(
                                    session_id=session_id,
                                    url=str(unit["url"]),
                                    duration_s=float(unit["duration_s"]),
                                )
                                visit_results.append(asdict(result))
                                trace_visit_exposed_s += result.exposed_wait_s
                                trace_saved_s += result.saved_service_s
                                realized_gain_s += result.saved_service_s
                                observed_tool_s.append(result.service_s)
                        else:
                            duration_s = float(tool["duration_s"])
                            await asyncio.sleep(duration_s)
                            observed_tool_s.append(duration_s)
                            if tool_name == "search":
                                trace_search_s += duration_s
                            else:
                                trace_other_tool_s += duration_s
                        tool_finished = time.monotonic()

                        spec = tool.get("speculation")
                        admitted: tuple[bool, ...] = ()
                        if treatment and isinstance(spec, Mapping):
                            candidates = list(spec["candidates"])
                            admitted = await visit_pool.speculate_batch(
                                [
                                    (
                                        session_id,
                                        str(row["url"]),
                                        float(row["duration_s"]),
                                        float(row["score"]),
                                        str(spec["decision_id"]),
                                    )
                                    for row in candidates
                                ]
                            )
                            future_gain_s = float(spec["expected_gain_s"])
                            next_tool_probability = min(
                                1.0,
                                max(
                                    0.0,
                                    float(spec["expected_authoritative_calls"]),
                                ),
                            )
                            conditional_service = (
                                future_gain_s / next_tool_probability
                                if next_tool_probability > 0.0
                                else 0.0
                            )
                            next_tool_wait_s = conditional_service
                        async with list_lock:
                            tool_events.append(
                                {
                                    "trace_id": trace_id,
                                    "session_id": session_id,
                                    "event_index": tool["event_index"],
                                    "tool_name": tool_name,
                                    "start_offset_s": tool_started - experiment_started,
                                    "end_offset_s": tool_finished - experiment_started,
                                    "duration_s": tool_finished - tool_started,
                                    "visit_results": visit_results,
                                    "speculation_decision_id": (
                                        spec.get("decision_id")
                                        if isinstance(spec, Mapping)
                                        else None
                                    ),
                                    "speculation_admitted": list(admitted),
                                }
                            )
            except Exception as exc:
                failure = repr(exc)
                raise
            finally:
                await visit_pool.close_session(session_id)
                if admission is not None:
                    await admission.finish_session(session_id)
                task_finished = time.monotonic()
                async with list_lock:
                    task_rows.append(
                        {
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "release_offset_s": release_offset_s,
                            "release_lag_s": released_at - scheduled_release,
                            "flow_s": task_finished - task_started,
                            "task_gate_wait_s": task_gate_acquired - task_started,
                            "llm_s": trace_llm_s,
                            "admission_wait_s": trace_admission_s,
                            "search_sleep_s": trace_search_s,
                            "other_tool_sleep_s": trace_other_tool_s,
                            "visit_exposed_s": trace_visit_exposed_s,
                            "saved_visit_service_s": trace_saved_s,
                            "failure": failure,
                        }
                    )

    server_url = args.server_url.rstrip("/")
    request_url = f"{server_url}/v1/chat/completions"
    metrics_url = f"{server_url}/metrics"
    headers = (
        {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
        if os.environ.get("VLLM_API_KEY")
        else {}
    )
    connector = aiohttp.TCPConnector(limit=0)
    error: BaseException | None = None
    async with aiohttp.ClientSession(headers=headers, connector=connector) as http_session:
        before = await fetch_metrics(http_session, metrics_url)
        try:
            trace_rows = list(plan["traces"])
            if args.trace_limit is not None:
                trace_rows = trace_rows[: args.trace_limit]
            await asyncio.gather(*(run_trace(trace) for trace in trace_rows))
        except BaseException as exc:
            error = exc
        after = await fetch_metrics(http_session, metrics_url)
    experiment_finished = time.monotonic()
    visit_snapshot = visit_pool.snapshot()
    await visit_pool.close()
    if admission is not None:
        admission_snapshot = admission.snapshot()
        await admission.close()
    else:
        admission_snapshot = None

    task_flows = [float(row["flow_s"]) for row in task_rows]
    llm_latencies = [float(row["latency_s"]) for row in request_events]
    authority = int(
        visit_snapshot["metrics"].get("authority_requests", 0)
    )
    physical = int(
        visit_snapshot["metrics"].get("physical_authority_starts", 0)
    ) + int(visit_snapshot["metrics"].get("physical_speculative_starts", 0))
    metric_names = sorted(set(before) | set(after))
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "plan_sha256": expected_hash,
        "scheduler_policy": os.getenv("VLLM_SCHED_POLICY", "fcfs"),
        "configuration": {
            "server_url": server_url,
            "model": args.model,
            "max_active_tasks": args.max_active_tasks,
            "visit_capacity": args.visit_capacity,
            "speculative_cap": args.speculative_cap if treatment else 0,
            "admission_backend": args.admission_backend if treatment else "none",
            "pressure_low": args.pressure_low if treatment else None,
            "pressure_high": args.pressure_high if treatment else None,
            "cold_session_cap": args.cold_session_cap if treatment else None,
            "gain_weight": args.gain_weight if treatment else None,
            "aging_weight": args.aging_weight if treatment else None,
            "kv_weight": args.kv_weight if treatment else None,
            "physical_kv_target": os.getenv(
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"
            ),
            "arrival_process": plan.get("arrival_process"),
        },
        "tasks": len(task_rows),
        "requests": len(request_events),
        "tool_events": len(tool_events),
        "failures": sum(row["failure"] is not None for row in task_rows),
        "experiment_wall_s": experiment_finished - experiment_started,
        "release_span_s": max(
            (float(row.get("release_offset_s", 0.0)) for row in plan["traces"]),
            default=0.0,
        ),
        "mean_task_flow_s": statistics.fmean(task_flows) if task_flows else 0.0,
        "p50_task_flow_s": percentile(task_flows, 0.50),
        "p95_task_flow_s": percentile(task_flows, 0.95),
        "max_task_flow_s": max(task_flows, default=0.0),
        "mean_llm_latency_s": statistics.fmean(llm_latencies) if llm_latencies else 0.0,
        "p95_llm_latency_s": percentile(llm_latencies, 0.95),
        "mean_admission_wait_s": statistics.fmean(
            [float(row["admission_wait_s"]) for row in request_events]
        ) if request_events else 0.0,
        "mean_search_sleep_s_per_task": statistics.fmean(
            [float(row["search_sleep_s"]) for row in task_rows]
        ) if task_rows else 0.0,
        "mean_visit_exposed_s_per_task": statistics.fmean(
            [float(row["visit_exposed_s"]) for row in task_rows]
        ) if task_rows else 0.0,
        "mean_saved_visit_service_s_per_task": statistics.fmean(
            [float(row["saved_visit_service_s"]) for row in task_rows]
        ) if task_rows else 0.0,
        "visit": visit_snapshot,
        "realized_visit_hit_rate": (
            float(visit_snapshot["metrics"].get("cache_hits", 0)) / authority
            if authority
            else 0.0
        ),
        "visit_call_amplification": physical / authority if authority else 0.0,
        "admission": admission_snapshot,
        "vllm_metric_deltas": {
            name: after.get(name, 0.0) - before.get(name, 0.0)
            for name in metric_names
            if name.startswith("vllm:")
        },
        "scheduler_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("VLLM_SCHED_")
        },
    }
    output_dir = args.output_dir
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "request_events.json", request_events)
    write_json(output_dir / "tool_events.json", tool_events)
    write_json(output_dir / "task_results.json", task_rows)
    if error is not None:
        raise error
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-plan", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["vllm_baseline", "coscheduled_speculation"],
        default="vllm_baseline",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    parser.add_argument("--tokenizer", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-output-tokens-cap", type=int, default=2048)
    parser.add_argument("--min-output-tokens-floor", type=int, default=128)
    parser.add_argument("--output-token-buffer", type=int, default=32)
    parser.add_argument("--candidate-pool-size", type=int, default=20)
    parser.add_argument("--selector-model", choices=["rich_logistic", "pairwise", "blend"], default="blend")
    parser.add_argument("--candidate-policy", choices=["budget_w5_cap10", "fixed_top10"], default="budget_w5_cap10")
    parser.add_argument("--coordination-cost-ms", type=float, default=1.0)
    parser.add_argument("--domain-prior-strength", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-active-tasks", type=int, default=64)
    parser.add_argument("--trace-limit", type=int)
    parser.add_argument("--visit-capacity", type=int, default=128)
    parser.add_argument("--speculative-cap", type=int, default=127)
    parser.add_argument(
        "--admission-backend",
        choices=["engine_joint", "python_gain_pressure"],
        default="engine_joint",
        help=(
            "engine_joint leaves ready turns in vLLM's native waiting queue so "
            "the in-engine Joint policy ranks and admits them; "
            "python_gain_pressure enables the experimental external gate"
        ),
    )
    parser.add_argument("--pressure-low", type=int, default=24)
    parser.add_argument("--pressure-high", type=int, default=40)
    parser.add_argument("--cold-session-cap", type=int, default=64)
    parser.add_argument("--gain-weight", type=float, default=1.0)
    parser.add_argument("--aging-weight", type=float, default=0.02)
    parser.add_argument("--kv-weight", type=float, default=1.0)
    parser.add_argument("--context-ref-tokens", type=int, default=16000)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    positive = (
        "max_model_len", "max_output_tokens_cap", "min_output_tokens_floor",
        "candidate_pool_size", "max_active_tasks", "visit_capacity",
        "pressure_low", "pressure_high", "cold_session_cap", "context_ref_tokens",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("integer capacities and limits must be positive")
    if args.trace_limit is not None and args.trace_limit <= 0:
        parser.error("--trace-limit must be positive")
    if not 0 <= args.speculative_cap <= args.visit_capacity:
        parser.error("--speculative-cap must be in [0, visit-capacity]")
    if args.pressure_low > args.pressure_high:
        parser.error("--pressure-low cannot exceed --pressure-high")
    if args.gain_weight < 0 or args.aging_weight < 0 or args.kv_weight < 0:
        parser.error("scheduler weights must be non-negative")
    return args


async def main_async(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepared_plan is None:
        plan = build_plan(args)
        plan_path = args.output_dir / "prepared_plan.json"
        write_json(plan_path, plan)
    else:
        plan = json.loads(args.prepared_plan.read_text(encoding="utf-8"))
        plan_path = args.prepared_plan
    if args.prepare_only:
        print(json.dumps({"prepared_plan": str(plan_path), "plan_sha256": plan["plan_sha256"]}, indent=2))
        return 0
    summary = await execute_plan(args, plan)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failures"] else 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
