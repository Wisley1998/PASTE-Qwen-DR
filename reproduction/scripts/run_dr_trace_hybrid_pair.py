#!/usr/bin/env python3
"""Run a paired trace-conditioned replay of real Qwen DR traces.

``--tool-backend live`` uses the existing broker and HTTP executor, with
simulated Search sharing its capacity. Only raw URL fetches with explicit
HTTP freshness are reusable. LLM messages and authoritative calls stay fixed;
this measures execution latency/cost, not autonomous answer quality.

The workload preserves complete, distinct DeepResearch trace sessions and their
recorded messages/token cadence.  LLM calls execute on live vLLM.  Baseline
replays every recorded tool duration; FULL subtracts only exact URL hits from
the frozen, out-of-fold Pattern-v2 session cache.  Both cells share the same
arrival process and tool concurrency limit.

``--preengine-policy gain-pressure`` optionally ranks coalesced cold sessions
by remaining LLM work, pressure-adjusted preserved tool gain, and aging.  Once
admitted, a session holds its slot through every LLM/tool turn; the default
``fifo`` policy retains the historical semaphore path.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PLAN_SCHEMA = "paste_repro.dr_trace_hybrid_plan.v1"
RESULT_SCHEMA = "paste_repro.dr_trace_hybrid_result.v1"
SCHEDULER_METADATA_SCHEMA = "paste.schedx.remaining_llm_work.v1"


@dataclass(frozen=True)
class SessionAdmissionFeatures:
    """Frozen task-level inputs to cold-session gain/pressure admission."""

    remaining_completion_tokens: int
    prompt_pressure_tokens: int
    expected_tool_gain_s: float


@dataclass
class _AdmissionTicket:
    task_id: str
    features: SessionAdmissionFeatures
    arrived_s: float
    sequence: int
    ready: asyncio.Future[None]


def session_admission_features(
    trace: Mapping[str, Any], *, full: bool
) -> SessionAdmissionFeatures:
    """Derive immutable work and saved-tool-gain from the frozen trace."""

    steps = list(trace["steps"])
    if not steps:
        raise ValueError("session admission requires a non-empty fixed trace")
    remaining_completion_tokens = sum(
        int(step["request"]["fixed_completion_tokens"])
        for step in steps
    )
    # Multi-turn prompts are nested context envelopes, so their maximum is a
    # pressure proxy without charging the same prefix repeatedly.
    prompt_pressure_tokens = max(
        int(step["request"]["prompt_tokens"]) for step in steps
    )
    if remaining_completion_tokens < 1 or prompt_pressure_tokens < 1:
        raise ValueError("session admission token work must be positive")
    expected_tool_gain_s = (
        sum(
            max(0.0, float(tool["offline_saved_s"]))
            for step in steps
            for tool in step["tools_after"]
        )
        if full else 0.0
    )
    return SessionAdmissionFeatures(
        remaining_completion_tokens=remaining_completion_tokens,
        prompt_pressure_tokens=prompt_pressure_tokens,
        expected_tool_gain_s=expected_tool_gain_s,
    )


def session_admission_score(
    features: SessionAdmissionFeatures,
    *,
    wait_s: float,
    pressure: float,
    prefill_tokens_per_s: float,
    decode_tokens_per_s: float,
    pressure_weight: float,
    tool_gain_beta: float,
    aging_alpha: float,
) -> float:
    """Return the lower-is-better cold-session gain/pressure score."""

    if prefill_tokens_per_s <= 0 or decode_tokens_per_s <= 0:
        raise ValueError("admission token rates must be positive")
    bounded_pressure = max(0.0, min(1.0, pressure))
    pressure_scale = 1.0 + max(0.0, pressure_weight) * bounded_pressure
    remaining_llm_s = (
        features.prompt_pressure_tokens / prefill_tokens_per_s
        + features.remaining_completion_tokens / decode_tokens_per_s
    )
    exposed_gain_s = (
        max(0.0, tool_gain_beta)
        * features.expected_tool_gain_s
        / pressure_scale
    )
    # Unbounded aging eventually overrides any finite work/gain difference.
    return (
        remaining_llm_s
        - exposed_gain_s
        - max(0.0, aging_alpha) * max(0.0, wait_s)
    )


class AsyncSessionAdmissionPool:
    """Coalescing, priority-ranked, session-persistent admission slots."""

    def __init__(
        self,
        *,
        capacity: int,
        coalesce_s: float,
        prefill_tokens_per_s: float,
        decode_tokens_per_s: float,
        pressure_weight: float,
        tool_gain_beta: float,
        aging_alpha: float,
    ) -> None:
        if capacity < 1:
            raise ValueError("session admission capacity must be positive")
        numeric_options = {
            "coalesce_s": coalesce_s,
            "prefill_tokens_per_s": prefill_tokens_per_s,
            "decode_tokens_per_s": decode_tokens_per_s,
            "pressure_weight": pressure_weight,
            "tool_gain_beta": tool_gain_beta,
            "aging_alpha": aging_alpha,
        }
        if not all(math.isfinite(value) for value in numeric_options.values()):
            raise ValueError("session admission options must be finite")
        if coalesce_s < 0:
            raise ValueError("admission coalescing window cannot be negative")
        if prefill_tokens_per_s <= 0 or decode_tokens_per_s <= 0:
            raise ValueError("admission token rates must be positive")
        self.capacity = capacity
        self.coalesce_s = coalesce_s
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self.decode_tokens_per_s = decode_tokens_per_s
        self.pressure_weight = pressure_weight
        self.tool_gain_beta = tool_gain_beta
        self.aging_alpha = aging_alpha
        self._lock = asyncio.Lock()
        self._pending: dict[str, _AdmissionTicket] = {}
        self._active: set[str] = set()
        self._sequence = 0
        self._dispatch_task: asyncio.Task[None] | None = None

    @property
    def active(self) -> int:
        return len(self._active)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def _schedule_dispatch_locked(self) -> None:
        if not self._pending or len(self._active) >= self.capacity:
            return
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        loop = asyncio.get_running_loop()
        oldest = min(ticket.arrived_s for ticket in self._pending.values())
        delay_s = max(0.0, oldest + self.coalesce_s - loop.time())
        self._dispatch_task = loop.create_task(self._dispatch_after(delay_s))

    async def _dispatch_after(self, delay_s: float) -> None:
        try:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            async with self._lock:
                if self._dispatch_task is asyncio.current_task():
                    self._dispatch_task = None
                try:
                    loop = asyncio.get_running_loop()
                    while self._pending and len(self._active) < self.capacity:
                        now_s = loop.time()
                        pressure = len(self._active) / self.capacity
                        ticket = min(
                            self._pending.values(),
                            key=lambda item: (
                                session_admission_score(
                                    item.features,
                                    wait_s=now_s - item.arrived_s,
                                    pressure=pressure,
                                    prefill_tokens_per_s=self.prefill_tokens_per_s,
                                    decode_tokens_per_s=self.decode_tokens_per_s,
                                    pressure_weight=self.pressure_weight,
                                    tool_gain_beta=self.tool_gain_beta,
                                    aging_alpha=self.aging_alpha,
                                ),
                                item.arrived_s,
                                item.sequence,
                            ),
                        )
                        self._pending.pop(ticket.task_id)
                        self._active.add(ticket.task_id)
                        if not ticket.ready.done():
                            ticket.ready.set_result(None)
                    self._schedule_dispatch_locked()
                except Exception as exc:
                    pending = tuple(self._pending.values())
                    self._pending.clear()
                    for ticket in pending:
                        if not ticket.ready.done():
                            ticket.ready.set_exception(exc)
        except asyncio.CancelledError:
            async with self._lock:
                if self._dispatch_task is asyncio.current_task():
                    self._dispatch_task = None
                pending = tuple(self._pending.values())
                self._pending.clear()
                for ticket in pending:
                    if not ticket.ready.done():
                        ticket.ready.cancel()
            raise

    async def acquire(
        self, task_id: str, features: SessionAdmissionFeatures
    ) -> None:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        async with self._lock:
            if task_id in self._pending or task_id in self._active:
                raise ValueError(f"duplicate admission task id: {task_id}")
            self._pending[task_id] = _AdmissionTicket(
                task_id=task_id,
                features=features,
                arrived_s=loop.time(),
                sequence=self._sequence,
                ready=ready,
            )
            self._sequence += 1
            self._schedule_dispatch_locked()
        try:
            await ready
        except asyncio.CancelledError:
            async with self._lock:
                self._pending.pop(task_id, None)
                self._active.discard(task_id)
                self._schedule_dispatch_locked()
            raise

    async def release(self, task_id: str) -> None:
        async with self._lock:
            if task_id not in self._active:
                raise ValueError(
                    f"session admission task is not active: {task_id}"
                )
            self._active.remove(task_id)
            self._schedule_dispatch_locked()


def canonical_hash(value: Any) -> str:
    wire = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    temporary.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def checked_hash(value: dict[str, Any], field: str, path: Path) -> None:
    expected = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if expected != canonical_hash(unsigned):
        raise ValueError(f"checksum mismatch: {path}")


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def prepare(args: argparse.Namespace) -> int:
    source = read_json(args.source_plan)
    if source.get("schema") != "paste_repro.trace_all_visit_live_plan.v1":
        raise ValueError("unsupported source plan schema")
    checked_hash(source, "plan_sha256", args.source_plan)
    trace_offset = getattr(args, "trace_offset", 0)
    if trace_offset < 0:
        raise ValueError("trace offset must be non-negative")
    source_traces = source["traces"][trace_offset:]
    arrivals = read_json(args.arrivals)
    arrival_rows = arrivals.get("arrivals") or arrivals.get("traces") or arrivals.get("sessions")
    if not isinstance(arrival_rows, list) or len(arrival_rows) < args.sessions:
        raise ValueError("arrival file contains too few rows")
    if len(source_traces) < args.sessions:
        raise ValueError("source plan contains too few distinct traces")

    offsets = [float(row["release_offset_s"]) for row in arrival_rows[:args.sessions]]
    if offsets != sorted(offsets) or not offsets or offsets[0] < 0:
        raise ValueError("arrival offsets must be sorted and non-negative")

    traces: list[dict[str, Any]] = []
    cache_hits = 0
    visit_units = 0
    visit_service_s = 0.0
    all_tool_service_s = 0.0
    offline_saved_s = 0.0
    requests = 0
    prompt_tokens = 0
    completion_tokens = 0
    tools = 0
    for index, (raw_trace, arrival) in enumerate(
        zip(source_traces[:args.sessions], arrival_rows[:args.sessions], strict=True)
    ):
        trace = copy.deepcopy(raw_trace)
        trace["task_id"] = f"dr-{index + 1:03d}-{trace['trace_id']}"
        trace["release_offset_s"] = float(arrival["release_offset_s"])
        trace["arrival_source_id"] = arrival.get("source_id")
        trace["arrival_source_row"] = arrival.get("csv_row_number")
        speculative_cache: set[str] = set()
        for step in trace["steps"]:
            request = step["request"]
            fixed_completion = min(
                int(request["target_output_tokens"]), int(request["max_tokens"])
            )
            if fixed_completion <= 0:
                raise ValueError("fixed completion work must be positive")
            request["fixed_completion_tokens"] = fixed_completion
            requests += 1
            prompt_tokens += int(request["prompt_tokens"])
            completion_tokens += fixed_completion
            for tool in step["tools_after"]:
                tools += 1
                duration_s = float(tool["duration_s"])
                all_tool_service_s += duration_s
                hit_urls: list[str] = []
                saved_s = 0.0
                if tool["tool_name"] == "visit":
                    visit_service_s += duration_s
                    for unit in tool.get("visit_units", []):
                        visit_units += 1
                        url = str(unit["url"])
                        if url in speculative_cache:
                            cache_hits += 1
                            hit_urls.append(url)
                            saved_s += float(unit["duration_s"])
                if saved_s > duration_s + 1e-6:
                    raise ValueError("offline saved service exceeds tool duration")
                tool["offline_cache_hit_urls"] = hit_urls
                tool["offline_saved_s"] = min(duration_s, saved_s)
                offline_saved_s += tool["offline_saved_s"]
                speculation = tool.get("speculation")
                if isinstance(speculation, Mapping):
                    for candidate in speculation.get("candidates", []):
                        speculative_cache.add(str(candidate["url"]))
        traces.append(trace)

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "benchmark": "Qwen DeepResearch real trace replay",
            "sessions": args.sessions,
            "session_identity": f"{args.sessions} distinct DR sessions from offset {trace_offset}; no replication",
            "arrival_process": "unchanged raw Azure 3-second/80-arrival window",
            "llm_clock": "live vLLM Tongyi-DeepResearch-30B-A3B",
            "llm_prompts": "recorded real multi-turn messages",
            "llm_completion_work": "fixed min(target_output_tokens, max_tokens)",
            "tool_clock": "recorded corrected trace service times, shared capacity",
            "full_tool_policy": (
                "frozen all-Visit Pattern-v2 nested-OOF blend, W=5/cap=10, "
                "session URL cache, exact authoritative URL confirmation"
            ),
            "offline_boundary": (
                "tool hit/readiness labels are frozen offline; only LLM service and "
                "queueing plus the residual tool clock are measured live"
            ),
        },
        "sources": {
            "source_plan": str(args.source_plan.resolve()),
            "trace_offset": trace_offset,
            "source_plan_file_sha256": file_sha256(args.source_plan),
            "source_plan_sha256": source["plan_sha256"],
            "arrival_path": str(args.arrivals.resolve()),
            "arrival_sha256": file_sha256(args.arrivals),
            "trace_scale": source.get("trace_scale"),
            "predictor": source.get("predictor"),
            "coverage": source.get("coverage"),
            "source_configuration": source.get("configuration"),
        },
        "summary": {
            "sessions": len(traces),
            "requests": requests,
            "tools": tools,
            "prompt_tokens": prompt_tokens,
            "fixed_completion_tokens": completion_tokens,
            "arrival_span_s": offsets[-1] - offsets[0],
            "all_tool_service_s": all_tool_service_s,
            "visit_service_s": visit_service_s,
            "executable_visit_urls": visit_units,
            "offline_cache_hits": cache_hits,
            "offline_cache_hit_rate": cache_hits / visit_units if visit_units else 0.0,
            "offline_saved_visit_s": offline_saved_s,
            "offline_visit_reduction": (
                offline_saved_s / visit_service_s if visit_service_s else 0.0
            ),
        },
        "traces": traces,
    }
    plan["plan_sha256"] = canonical_hash(plan)
    write_json(args.output, plan)
    print(json.dumps({"output": str(args.output), "plan_sha256": plan["plan_sha256"], **plan["summary"]}, indent=2))
    return 0


def checked_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported hybrid plan schema")
    checked_hash(plan, "plan_sha256", path)
    return plan


def schedx_id(metadata: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(metadata), ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8").hex()
    return f"schedx{encoded}z"


def causal_scheduler_metadata(
    *, task_id: str, call_index: int, request_index: int,
    prompt_tokens: int, generation_cap: int, po_ema: float,
    tool_eta_s: float, tool_confidence: float, expected_gain_s: float = 0.0,
) -> dict[str, Any]:
    """Online signals only: current request, completed-output EMA and tool prediction.

    One further call is a shared fixed prior, not a claim about trace finality.
    Predict the next prompt by persistence. Never accept a trace or outcome here.
    """
    if generation_cap < 1 or prompt_tokens < 1:
        raise ValueError("current prompt and declared generation cap must be positive")
    po = max(1.0, min(float(po_ema), generation_cap))
    eta = max(0.0, tool_eta_s)
    confidence = max(0.0, min(1.0, tool_confidence))
    return {
        "ms": "paste.schedx.causal_prediction.v1",
        "t": task_id, "c": call_index, "i": request_index,
        "pt": prompt_tokens, "mt": generation_cap,
        "po_hat": po, "remaining_calls_hat": 1,
        "remaining_llm_tokens_hat": 2 * po,
        "next_prompt_tokens_hat": prompt_tokens,
        "next_output_tokens_hat": po,
        "tool_eta_s_hat": eta, "tool_hit_probability_hat": confidence,
        "remaining_tool_wait_s_hat": eta,
        "expected_gain_s_hat": max(0.0, expected_gain_s),
    }


def speculation_readiness(snapshot: Mapping[str, Any], tool_eta_s: float) -> dict[str, Any]:
    """Observe current broker state; readiness is not a future authoritative hit.

    Pending jobs get no realized-work credit. Completed valid jobs receive only
    a probability-weighted service estimate, never a known future saved duration.
    """
    ready = pending = unavailable = 0
    gain = confidence = 0.0
    for job in snapshot["jobs"]:
        if not job["prediction_available"]:
            unavailable += 1
            continue
        probability = min(1.0, max(0.0, float(job["priority"])))
        confidence = max(confidence, probability)
        if job["reusable_now"]:
            ready += 1
            service = job.get("service_estimate_s")
            service = tool_eta_s if service is None else max(0.0, float(service))
            gain = max(gain, probability * min(tool_eta_s, service))
        elif job["state"] in {"queued", "running"}:
            pending += 1
        else:
            unavailable += 1
    return dict(ready=ready, pending=pending, unavailable=unavailable,
                confidence=confidence, expected_gain_s=gain)


def build_scheduler_metadata(
    trace: Mapping[str, Any],
    request_index: int,
    *,
    full: bool,
    po_ema: float,
) -> dict[str, Any]:
    """Build the frozen trace-derived signals carried in a request ID.

    ``rlmt`` is the remaining completion-token work for the whole task,
    including the current request.  ``npt`` and ``nmt`` describe the next
    request and are zero on the final request.  These fields are scheduling
    hints only; the live request still uses the recorded messages and fixed
    completion-token count below.
    """

    steps = list(trace["steps"])
    step = steps[request_index]
    request = step["request"]
    remaining_calls = len(steps) - request_index - 1
    remaining_tool_s = sum(
        max(
            0.0,
            float(tool["duration_s"])
            - (float(tool["offline_saved_s"]) if full else 0.0),
        )
        for future in steps[request_index:]
        for tool in future["tools_after"]
    )
    next_tools = list(step["tools_after"])
    next_wait = sum(
        max(
            0.0,
            float(tool["duration_s"])
            - (float(tool["offline_saved_s"]) if full else 0.0),
        )
        for tool in next_tools
    )
    fixed_completion = int(request["fixed_completion_tokens"])
    remaining_llm_tokens = sum(
        int(future["request"]["fixed_completion_tokens"])
        for future in steps[request_index:]
    )
    next_request = (
        steps[request_index + 1]["request"] if remaining_calls > 0 else None
    )
    return {
        "t": str(trace["task_id"]),
        "c": int(request["call_index"]),
        "i": request_index,
        "n": len(steps),
        "rc": remaining_calls,
        "pt": int(request["prompt_tokens"]),
        "mt": fixed_completion,
        "po": int(max(1, min(po_ema, fixed_completion))),
        "rlmt": remaining_llm_tokens,
        "npt": int(next_request["prompt_tokens"]) if next_request else 0,
        "nmt": (
            int(next_request["fixed_completion_tokens"])
            if next_request else 0
        ),
        "nw": next_wait,
        "nwc": 1.0 if next_tools else 0.0,
        "rtw": remaining_tool_s,
        "eg": (
            sum(
                float(tool["offline_saved_s"])
                for future in steps[request_index:]
                for tool in future["tools_after"]
            )
            if full else 0.0
        ),
        "ms": "real_dr_trace_offline_pattern_v2",
    }


def timed_visit_durations(tool: Mapping[str, Any]) -> list[float]:
    """Executor-only URL service demands, preserving the frozen serial sum."""
    urls = tool["arguments"].get("url", [])
    urls = [urls] if isinstance(urls, str) else urls
    units = tool.get("visit_units", [])
    total = float(tool["duration_s"])
    if [unit["url"] for unit in units] == urls and units:
        durations = [float(unit["duration_s"]) for unit in units]
        if any(not math.isfinite(value) or value < 0 for value in durations):
            raise ValueError("invalid recorded URL duration")
        if not math.isclose(sum(durations), total, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("recorded visit_units do not sum to the batch duration")
        return durations
    if units:
        raise ValueError("recorded visit_units do not match the authoritative URL order")
    # Non-HTTP/view-source calls have no executable URL units in this corpus.
    return [total / len(urls)] * len(urls) if urls else []


async def run_cell(args: argparse.Namespace) -> int:
    plan = checked_plan(args.plan)
    full = args.system == "full"
    timed = getattr(args, "tool_backend", "replay") == "timed"
    live = getattr(args, "tool_backend", "replay") in {"live", "timed"}
    broker = executor = predictor = None
    if live:
        from paste_repro.invocation import Invocation
        from paste_repro.live_broker import LiveToolBroker
        from paste_repro.live_executor import WikipediaLiveExecutor
        from paste_repro.pattern_v2_all_visit_online import PatternV2CrossFitPredictor
        if args.preengine_policy != "fifo":
            raise ValueError("live ranker comparisons require a fixed FIFO admission gate")
        if args.predictor is None:
            raise ValueError("live mode requires the frozen predictor, including at zero budget")
        if not 0 <= args.speculation_capacity <= args.tool_capacity:
            raise ValueError("speculation capacity must be in [0, tool capacity]")
        if not 1 <= args.speculation_top_k <= 10:
            raise ValueError("speculation top-k must be in [1, 10]")
        if not timed and (not math.isfinite(args.speculation_ttl_s)
                          or args.session_result_cache or args.preempt_speculation):
            raise ValueError("session-stable retention and timer preemption require --tool-backend timed")
        for trace in plan["traces"]:
            for step in trace["steps"]:
                for tool in step["tools_after"]:
                    if "arguments" not in tool:
                        raise ValueError("rebuild source plan with original prepare-only entry to retain tool arguments")
        # Private executor input: neither the predictor nor scheduler reads
        # these recorded service times before the simulated job completes.
        recorded_services = {}
        assigned_services = {}
        if timed:
            for trace in plan["traces"]:
                for step in trace["steps"]:
                    for tool in step["tools_after"]:
                        if tool["tool_name"] != "visit":
                            continue
                        urls = tool["arguments"].get("url", [])
                        urls = [urls] if isinstance(urls, str) else urls
                        durations = timed_visit_durations(tool)
                        for url, duration in zip(urls, durations):
                            recorded_services.setdefault((trace["task_id"], url),
                                duration)
        executor = None if timed else WikipediaLiveExecutor(timeout_s=20, max_http_attempts=1, max_visit_urls=1)
        async def execute(invocation):
            if timed:
                await asyncio.sleep(assigned_services[id(invocation)])
                return {"simulated_tool": True, "_paste_transport": {
                    "backend": "simulated_trace_tool", "http_attempts": 0, "bytes_read": 0,
                    # The timer model has stable values during the existing TTL.
                    # This is not evidence about HTTP freshness or real outputs.
                    "reuse_valid_until_monotonic_s": (time.monotonic() + args.speculation_ttl_s
                        if math.isfinite(args.speculation_ttl_s) else 1e300)}}

            if invocation.tool_name in {"search", "google_scholar"}:
                await asyncio.sleep(invocation.arguments["_simulated_service_s"])
                return {"simulated_search": True, "_paste_transport": {
                    "backend": "simulated_search", "http_attempts": 0, "bytes_read": 0}}
            return await executor(invocation)
        broker = LiveToolBroker(execute, max_workers=args.tool_capacity,
            max_speculative_workers=args.speculation_capacity if full else 0,
            max_speculative_pending=args.speculation_pending_capacity,
            max_completed_predictions=args.speculation_cache_capacity,
            retain_completed_predictions=args.session_result_cache,
            preempt_running_speculation=args.preempt_speculation,
            ttl_s=args.speculation_ttl_s, require_reuse_validity=True)
        predictor = PatternV2CrossFitPredictor.from_path(args.predictor)
        # The replay request["max_tokens"] was constructed from the recorded
        # completion plus a buffer. It is NOT an online-visible request cap.
        public_llm_config = plan["sources"]["source_configuration"]
        public_generation_limit = int(public_llm_config["max_output_tokens_cap"])
        public_context_limit = int(public_llm_config["max_model_len"])
    task_gate = (
        asyncio.Semaphore(args.max_active_tasks)
        if args.preengine_policy == "fifo" else None
    )
    session_admission = (
        AsyncSessionAdmissionPool(
            capacity=args.max_active_tasks,
            coalesce_s=args.preengine_coalesce_s,
            prefill_tokens_per_s=args.preengine_prefill_tokens_per_s,
            decode_tokens_per_s=args.preengine_decode_tokens_per_s,
            pressure_weight=args.preengine_pressure_weight,
            tool_gain_beta=args.preengine_tool_gain_beta,
            aging_alpha=args.preengine_aging_alpha,
        )
        if args.preengine_policy == "gain-pressure" else None
    )
    tool_gate = asyncio.Semaphore(args.tool_capacity)
    result_lock = asyncio.Lock()
    task_rows: list[dict[str, Any]] = []
    llm_events: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    started_mono = time.monotonic()
    started_wall = time.time()
    request_url = args.base_url.rstrip("/") + "/chat/completions"

    @asynccontextmanager
    async def admitted_session(
        task_id: str, trace: Mapping[str, Any]
    ) -> Any:
        if session_admission is not None:
            await session_admission.acquire(
                task_id,
                session_admission_features(trace, full=full),
            )
        else:
            assert task_gate is not None
            if live:
                await asyncio.sleep(args.preengine_coalesce_s)
            await task_gate.acquire()
        try:
            yield
        finally:
            if session_admission is not None:
                await session_admission.release(task_id)
            else:
                assert task_gate is not None
                task_gate.release()

    async def run_one(trace: Mapping[str, Any], http: aiohttp.ClientSession) -> None:
        release = float(trace["release_offset_s"]) * getattr(args, "arrival_scale", 1.0)
        deadline = started_mono + release
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        released = time.monotonic()
        task_id = str(trace["task_id"])
        session_id = str(trace["session_id"])
        async with admitted_session(task_id, trace):
            acquired = time.monotonic()
            error: str | None = None
            completed_requests = 0
            completed_tools = 0
            task_llm_s = 0.0
            task_tool_wait_s = 0.0
            task_saved_tool_s = 0.0
            po_ema = 128.0
            policy = predictor.start_session(source_session_id=session_id, runtime_session_id=task_id) if predictor else None
            prior_tool = None
            prediction_triggers = 0
            observed_tool_s = 2.0
            failed_tools = 0
            try:
                steps = list(trace["steps"])
                for request_index, step in enumerate(steps):
                    request = step["request"]
                    fixed_completion = int(request["fixed_completion_tokens"])
                    if live:
                        candidates = policy.predict_after_tool(
                            tool_name=prior_tool["tool_name"],
                            tool_arguments=prior_tool["arguments"],
                            current_messages=request["messages"],
                        ) if prior_tool else ()
                        prediction_triggers += int(prior_tool is not None)
                        prior_tool = None
                        if full and args.speculation_capacity:
                            batch = []
                            for candidate in candidates[:args.speculation_top_k]:
                                invocation = Invocation("visit", {"url": candidate.url, "goal": ""})
                                if timed:
                                    assigned_services[id(invocation)] = recorded_services.get(
                                        (task_id, candidate.url), 2.0)
                                batch.append((invocation, task_id, candidate.confidence))
                            await broker.speculate_batch(batch,
                                replace_lower_priority_queued=args.replace_queued_predictions)
                        readiness = speculation_readiness(broker.snapshot(session_id=task_id), observed_tool_s)
                        confidence = max(readiness["confidence"],
                            max((c.confidence for c in candidates), default=0.0))
                        metadata = causal_scheduler_metadata(
                            task_id=task_id, call_index=int(request["call_index"]),
                            request_index=request_index,
                            prompt_tokens=int(request["prompt_tokens"]),
                            generation_cap=min(public_generation_limit,
                                public_context_limit - int(request["prompt_tokens"])),
                            po_ema=po_ema,
                            tool_eta_s=observed_tool_s, tool_confidence=confidence,
                            expected_gain_s=readiness["expected_gain_s"],
                        )
                        metadata["readiness_at_submission"] = readiness
                    else:
                        # Legacy replay is an explicitly oracle diagnostic mode.
                        metadata = build_scheduler_metadata(
                            trace, request_index, full=full, po_ema=po_ema,
                        )
                    request_id = schedx_id(metadata)
                    payload = {
                        "model": args.model,
                        "messages": request["messages"],
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 0,
                        "max_tokens": fixed_completion,
                        "min_tokens": fixed_completion,
                        "ignore_eos": True,
                        "request_id": request_id,
                    }
                    llm_started = time.monotonic()
                    async with http.post(
                        request_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=args.request_timeout_s),
                    ) as response:
                        body = await response.json(content_type=None)
                        if response.status != 200:
                            raise RuntimeError(f"vLLM HTTP {response.status}: {body}")
                    llm_finished = time.monotonic()
                    usage = body.get("usage") or {}
                    observed_prompt = int(usage.get("prompt_tokens", -1))
                    observed_completion = int(usage.get("completion_tokens", -1))
                    if observed_prompt != int(request["prompt_tokens"]):
                        raise RuntimeError(
                            f"prompt-token mismatch {observed_prompt} != {request['prompt_tokens']}"
                        )
                    if observed_completion != fixed_completion:
                        raise RuntimeError(
                            f"completion-token mismatch {observed_completion} != {fixed_completion}"
                        )
                    latency = llm_finished - llm_started
                    task_llm_s += latency
                    po_ema = 0.5 * observed_completion + 0.5 * po_ema
                    completed_requests += 1
                    async with result_lock:
                        llm_events.append(
                            {
                                "task_id": task_id,
                                "session_id": session_id,
                                "request_index": request_index,
                                "call_index": int(request["call_index"]),
                                "request_id_sha256": hashlib.sha256(request_id.encode()).hexdigest(),
                                "start_offset_s": llm_started - started_mono,
                                "end_offset_s": llm_finished - started_mono,
                                "latency_s": latency,
                                "http_status": response.status,
                                "usage": {
                                    "prompt_tokens": observed_prompt,
                                    "completion_tokens": observed_completion,
                                    "total_tokens": int(usage.get("total_tokens", 0)),
                                },
                                "scheduler_metadata": metadata,
                            }
                        )

                    next_tools = list(step["tools_after"])
                    for tool in next_tools:
                        if live:
                            queued_at = time.monotonic()
                            deliveries = []
                            results = []
                            name, arguments = tool["tool_name"], tool["arguments"]
                            if name == "visit":
                                urls = arguments.get("url", [])
                                if isinstance(urls, str):
                                    urls = [urls]
                                invocations = [Invocation("visit", {"url": url, "goal": ""}) for url in urls]
                            elif name in {"search", "google_scholar"}:
                                invocations = [Invocation(name, {**arguments, "_simulated_service_s": tool["duration_s"]})]
                            else:
                                raise ValueError(f"unsupported real tool: {name}")
                            durations = (timed_visit_durations(tool) if name == "visit"
                                         else [float(tool["duration_s"])])
                            saved = 0.0
                            for invocation, duration in zip(invocations, durations):
                                if timed:
                                    assigned_services[id(invocation)] = duration
                                try:
                                    delivery = await broker.authoritative(invocation, session_id=task_id)
                                    deliveries.append(delivery)
                                    results.append(delivery.result)
                                    saved += min(duration, delivery.saved_service_s) if timed else delivery.saved_service_s
                                except Exception as exc:
                                    failed_tools += 1
                                    results.append({"error": type(exc).__name__, "message": str(exc)})
                            finished = time.monotonic()
                            exposed = finished - queued_at
                            observed_tool_s = 0.5 * observed_tool_s + 0.5 * exposed
                            task_tool_wait_s += exposed
                            task_saved_tool_s += saved
                            completed_tools += 1
                            prior_tool = tool
                            # Goal is formatting context; only raw URL fetches are cached.
                            result_digest = canonical_hash({"goal": arguments.get("goal", ""), "results": results})
                            tool_events.append({"task_id": task_id, "session_id": session_id,
                                "event_index": tool["event_index"], "call_index": tool["call_index"],
                                "tool_name": name, "queued_offset_s": queued_at - started_mono,
                                "end_offset_s": finished - started_mono, "exposed_wait_s": exposed,
                                "full_service_s": tool["duration_s"], "executed_service_s": None,
                                "offline_saved_s": 0.0, "offline_cache_hit_urls": [],
                                "saved_service_s": saved,
                                "sources": [d.source for d in deliveries], "result_sha256": result_digest})
                            continue
                        full_service = float(tool["duration_s"])
                        offline_saved = float(tool["offline_saved_s"]) if full else 0.0
                        executed_service = max(0.0, full_service - offline_saved)
                        queued_at = time.monotonic()
                        async with tool_gate:
                            service_started = time.monotonic()
                            await asyncio.sleep(executed_service)
                        finished = time.monotonic()
                        exposed = finished - queued_at
                        task_tool_wait_s += exposed
                        task_saved_tool_s += offline_saved
                        completed_tools += 1
                        result_digest = canonical_hash(
                            {
                                "tool_name": tool["tool_name"],
                                "call_index": tool["call_index"],
                                "visit_units": tool.get("visit_units", []),
                            }
                        )
                        async with result_lock:
                            tool_events.append(
                                {
                                    "task_id": task_id,
                                    "session_id": session_id,
                                    "event_index": int(tool["event_index"]),
                                    "call_index": int(tool["call_index"]),
                                    "tool_name": str(tool["tool_name"]),
                                    "queued_offset_s": queued_at - started_mono,
                                    "service_start_offset_s": service_started - started_mono,
                                    "end_offset_s": finished - started_mono,
                                    "queue_wait_s": service_started - queued_at,
                                    "exposed_wait_s": exposed,
                                    "full_service_s": full_service,
                                    "executed_service_s": executed_service,
                                    "offline_saved_s": offline_saved,
                                    "offline_cache_hit_urls": (
                                        list(tool["offline_cache_hit_urls"]) if full else []
                                    ),
                                    "result_sha256": result_digest,
                                }
                            )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            ended = time.monotonic()
            if broker:
                await broker.cancel_predictions(session_id=task_id)
            cleanup_ended = time.monotonic()
            async with result_lock:
                task_rows.append(
                    {
                        "task_id": task_id,
                        "session_id": session_id,
                        "release_offset_s": release,
                        "release_lag_s": released - deadline,
                        "client_gate_wait_s": acquired - released,
                        "preengine_gate_wait_s": acquired - released,
                        "preengine_policy": args.preengine_policy,
                        "e2e_s": ended - deadline,
                        "llm_s": task_llm_s,
                        "exposed_tool_s": task_tool_wait_s,
                        "saved_tool_s": task_saved_tool_s,
                        "completed_requests": completed_requests,
                        "completed_tools": completed_tools,
                        "cleanup_s": cleanup_ended - ended,
                        "failed_tool_calls": failed_tools,
                        "prediction_triggers": prediction_triggers,
                        "ok": error is None,
                        "error": error,
                    }
                )

    # Tool gaps can outlast vLLM's idle keep-alive timeout. Fresh local LLM
    # connections avoid reusing a socket that the server is concurrently closing.
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=live)
    async with aiohttp.ClientSession(connector=connector) as http:
        await asyncio.gather(*(run_one(trace, http) for trace in plan["traces"]))
    if broker:
        await broker.close()
        if executor is not None:
            await executor.close()
        costs = broker.resource_summary()
        for row in task_rows:
            row["resources"] = costs.get(row["task_id"], {})
    ended_mono = time.monotonic()
    ended_wall = time.time()

    task_rows.sort(key=lambda row: row["task_id"])
    llm_events.sort(key=lambda row: (row["task_id"], row["request_index"]))
    tool_events.sort(key=lambda row: (row["task_id"], row["event_index"]))
    good = [row for row in task_rows if row["ok"]]
    e2e = [float(row["e2e_s"]) for row in good]
    latencies = [float(row["latency_s"]) for row in llm_events]
    gate_waits = [float(row["preengine_gate_wait_s"]) for row in task_rows]
    summary = {
        "tasks": len(task_rows),
        "successful_tasks": len(good),
        "llm_requests": len(llm_events),
        "tool_calls": len(tool_events),
        "mean_e2e_s": statistics.fmean(e2e) if e2e else None,
        "p50_e2e_s": statistics.median(e2e) if e2e else None,
        "p95_e2e_s": percentile(e2e, 0.95),
        "makespan_s": ended_mono - started_mono,
        "mean_llm_request_s": statistics.fmean(latencies) if latencies else None,
        "p95_llm_request_s": percentile(latencies, 0.95),
        "mean_preengine_gate_wait_s": (
            statistics.fmean(gate_waits) if gate_waits else None
        ),
        "p95_preengine_gate_wait_s": percentile(gate_waits, 0.95),
        "full_tool_service_s": sum(float(row["full_service_s"]) for row in tool_events),
        "executed_tool_service_s": sum(float(row["executed_service_s"]) for row in tool_events) if not live else None,
        "exposed_tool_wait_s": sum(float(row["exposed_wait_s"]) for row in tool_events),
        "saved_tool_service_s": sum(float(row.get("saved_service_s", row["offline_saved_s"])) for row in tool_events),
        "offline_url_hits": sum(len(row["offline_cache_hit_urls"]) for row in tool_events),
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": args.system,
        "plan": str(args.plan.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "model": args.model,
        "base_url": args.base_url,
        "started_wall_s": started_wall,
        "ended_wall_s": ended_wall,
        "settings": {
            "max_active_tasks": args.max_active_tasks,
            "tool_capacity": args.tool_capacity,
            "scheduler": "native_fcfs" if not full else "online_joint_pacer_v2",
            "scheduler_metadata_schema": SCHEDULER_METADATA_SCHEMA,
            "preengine_policy": args.preengine_policy,
            "session_persistent_admission": True,
            "preengine_coalesce_s": args.preengine_coalesce_s,
            "preengine_prefill_tokens_per_s": (
                args.preengine_prefill_tokens_per_s
            ),
            "preengine_decode_tokens_per_s": (
                args.preengine_decode_tokens_per_s
            ),
            "preengine_pressure_weight": args.preengine_pressure_weight,
            "preengine_tool_gain_beta": args.preengine_tool_gain_beta,
            "preengine_aging_alpha": args.preengine_aging_alpha,
            "tool_mechanism": (
                "none" if not full else "offline_pattern_v2_oof_session_url_cache_exact_hits"
            ),
        },
        "summary": summary,
        "tasks": task_rows,
        "llm_events": llm_events,
        "tool_events": tool_events,
    }
    if live:
        from paste_repro.resource_accounting import summarize_resources
        result["settings"].update(tool_backend="timed" if timed else "live", speculation_capacity=args.speculation_capacity if full else 0,
            speculation_top_k=args.speculation_top_k,
            speculation_ttl_s=args.speculation_ttl_s if math.isfinite(args.speculation_ttl_s) else "session",
            speculation_pending_capacity=args.speculation_pending_capacity,
            speculation_cache_capacity=args.speculation_cache_capacity,
            session_result_cache=args.session_result_cache,
            preempt_speculation=args.preempt_speculation,
            replace_queued_predictions=args.replace_queued_predictions,
            speculative_priority="larger positive confidence first",
            arrival_scale=args.arrival_scale, predictor_sha256=file_sha256(args.predictor),
            scheduler="externally configured; verify server ranker evidence",
            scheduler_metadata_schema="paste.schedx.causal_prediction.v1",
            scheduler_generation_cap_source="configured max_output_tokens_cap clipped only by current context headroom; never replay max_tokens",
            scheduler_generation_cap=public_generation_limit,
            llm_estimator="completed-output EMA(alpha=0.5, initial=128); remaining calls prior=1; next prompt=current prompt",
            tool_mechanism="raw URL fetch; explicit HTTP freshness validation; simulated Search",
            scope="fixed trace calls/messages/LLM work; not autonomous solve-quality or dynamic-web equality")
        result["tool_jobs"] = list(broker.tool_records())
        result["resources"] = summarize_resources(result["tool_jobs"], now=ended_mono)
        if timed:
            result["settings"].update(
                tool_mechanism="shared broker with asynchronous recorded-duration executor; exact session invocation match; retention/cache/preemption settings explicit above; stable synthetic result",
                timing_scope="original visit_units durations, proportional reconciliation only when units omit non-HTTP entries; speculative matching URL uses first recorded per-session URL service; unmatched candidate2s prior; timings private to executor",
                scope="policy simulation with real LLM; simulated worker-time is not CPU/network cost or output correctness evidence")
            for row in [result, *task_rows]:
                row["resources"]["cpu_core_s"] = None
                row["resources"]["measurement_scope"] = "simulated trace worker occupancy; no physical CPU/network measurement"
                for key in ("http_requests", "http_request_bytes", "http_response_body_bytes"):
                    row["resources"][key] = None
        summary["executed_tool_service_s"] = result["resources"]["worker_occupancy_s"]
        summary["completion_rate"] = len(good) / len(task_rows) if task_rows else 0.0
        summary["p99_e2e_s"] = percentile(e2e, 0.99)
        summary["failed_tool_calls"] = sum(row["failed_tool_calls"] for row in task_rows)
    result["result_sha256"] = canonical_hash(result)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "system": args.system, **summary}, indent=2))
    return 0 if len(good) == len(task_rows) else 2


def reduction(baseline: float, full: float) -> float:
    return (baseline - full) / baseline if baseline else 0.0


def server_log_audit(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    running = [int(value) for value in re.findall(r"Running: (\d+)", text)]
    waiting = [int(value) for value in re.findall(r"Waiting: (\d+)", text)]
    max_num_seqs = re.search(r"'max_num_seqs': (\d+)", text)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "http_200_chat_completions": len(
            re.findall(r'POST /v1/chat/completions HTTP/1\.1" 200', text)
        ),
        "max_num_seqs": int(max_num_seqs.group(1)) if max_num_seqs else None,
        "max_running": max(running, default=0),
        "max_waiting": max(waiting, default=0),
        "joint_hook_installations": text.count(
            "[sched_policy_patch] installed policy=online_joint_pacer_v2"
        ),
        "fail_open_markers": text.count("fail_open"),
    }


def all_validity_checks_pass(validity: Mapping[str, Any]) -> bool:
    """Require every declared comparison invariant to be exactly true."""

    return bool(validity) and all(value is True for value in validity.values())


def compare(args: argparse.Namespace) -> int:
    baseline = read_json(args.baseline)
    full = read_json(args.full)
    if baseline.get("schema") != RESULT_SCHEMA or full.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported result schema")
    if baseline["plan_sha256"] != full["plan_sha256"]:
        raise ValueError("cells used different plans")
    if baseline["system"] != "baseline" or full["system"] != "full":
        raise ValueError("expected baseline and full cell")
    b = baseline["summary"]
    f = full["summary"]
    b_tasks = {row["task_id"]: row for row in baseline["tasks"]}
    f_tasks = {row["task_id"]: row for row in full["tasks"]}
    if set(b_tasks) != set(f_tasks):
        raise ValueError("paired task IDs differ")
    b_llm = [
        (row["task_id"], row["request_index"], row["usage"]["prompt_tokens"],
         row["usage"]["completion_tokens"], row["http_status"])
        for row in baseline["llm_events"]
    ]
    f_llm = [
        (row["task_id"], row["request_index"], row["usage"]["prompt_tokens"],
         row["usage"]["completion_tokens"], row["http_status"])
        for row in full["llm_events"]
    ]
    b_tools = [
        (row["task_id"], row["event_index"], row["tool_name"],
         row["full_service_s"], row["result_sha256"])
        for row in baseline["tool_events"]
    ]
    f_tools = [
        (row["task_id"], row["event_index"], row["tool_name"],
         row["full_service_s"], row["result_sha256"])
        for row in full["tool_events"]
    ]
    metrics = {
        "mean_e2e_reduction": reduction(b["mean_e2e_s"], f["mean_e2e_s"]),
        "p50_e2e_reduction": reduction(b["p50_e2e_s"], f["p50_e2e_s"]),
        "p95_e2e_reduction": reduction(b["p95_e2e_s"], f["p95_e2e_s"]),
        "makespan_reduction": reduction(b["makespan_s"], f["makespan_s"]),
        "mean_llm_request_reduction": reduction(
            b["mean_llm_request_s"], f["mean_llm_request_s"]
        ),
        "paired_tasks_faster": sum(
            f_tasks[key]["e2e_s"] < b_tasks[key]["e2e_s"] for key in b_tasks
        ),
        "paired_tasks": len(b_tasks),
    }
    logs = {
        "baseline": server_log_audit(args.baseline_server_log),
        "full": server_log_audit(args.full_server_log),
    }
    validity = {
        "same_frozen_plan": True,
        "same_model": baseline["model"] == full["model"],
        "all_tasks_successful": (
            b["tasks"]
            == b["successful_tasks"]
            == f["tasks"]
            == f["successful_tasks"]
        ),
        "request_counts_equal": b["llm_requests"] == f["llm_requests"],
        "tool_counts_equal": b["tool_calls"] == f["tool_calls"],
        "llm_token_work_and_status_equal": b_llm == f_llm,
        "tool_trace_work_and_results_equal": b_tools == f_tools,
        "server_http_counts_match_results": (
            logs["baseline"]["http_200_chat_completions"]
            == b["llm_requests"]
            == logs["full"]["http_200_chat_completions"]
        ),
        "same_server_max_num_seqs": (
            logs["baseline"]["max_num_seqs"] is not None
            and logs["baseline"]["max_num_seqs"]
            == logs["full"]["max_num_seqs"]
        ),
        "baseline_joint_hook_absent": (
            logs["baseline"]["joint_hook_installations"] == 0
        ),
        "full_joint_hook_installed": (
            logs["full"]["joint_hook_installations"] > 0
        ),
        "baseline_fail_open_free": logs["baseline"]["fail_open_markers"] == 0,
        "full_fail_open_free": logs["full"]["fail_open_markers"] == 0,
    }
    valid = all_validity_checks_pass(validity)
    invalid_checks = [
        name for name, passed in validity.items() if passed is not True
    ]
    report: dict[str, Any] = {
        "schema": "paste_repro.dr_trace_hybrid_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": baseline["plan_sha256"],
        "baseline": str(args.baseline.resolve()),
        "full": str(args.full.resolve()),
        "baseline_summary": b,
        "full_summary": f,
        "metrics": metrics,
        "server_logs": logs,
        "valid": valid,
        "invalid_checks": invalid_checks,
        "validity": validity,
        "observations": {
            "real_server_queue_observed": (
                logs["baseline"]["max_waiting"] > 0 and logs["full"]["max_waiting"] > 0
            ),
        },
    }
    report["report_sha256"] = canonical_hash(report)
    write_json(args.output, report)

    lines = [
        "# Qwen DeepResearch: real-trace hybrid comparison",
        "",
        "This paired experiment runs 80 distinct real DeepResearch sessions. All LLM requests and queueing are live; tools replay the frozen real-trace clock, and FULL applies only the frozen Pattern-v2 exact-hit offline cache projection.",
        "",
        "| Metric | Baseline | FULL | Reduction |",
        "|---|---:|---:|---:|",
    ]
    for label, key, metric in (
        ("Mean task E2E", "mean_e2e_s", "mean_e2e_reduction"),
        ("p50 task E2E", "p50_e2e_s", "p50_e2e_reduction"),
        ("p95 task E2E", "p95_e2e_s", "p95_e2e_reduction"),
        ("Makespan", "makespan_s", "makespan_reduction"),
        ("Mean LLM request", "mean_llm_request_s", "mean_llm_request_reduction"),
    ):
        lines.append(
            f"| {label} | {b[key]:.3f}s | {f[key]:.3f}s | {metrics[metric] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"FULL: `online_joint_pacer_v2` + frozen Pattern-v2 OOF session URL cache ({full['settings']['tool_capacity']} shared tool slots).",
            f"Offline exact URL hits: {f['offline_url_hits']}; removed Visit service: {f['saved_tool_service_s']:.3f}s; residual executed tool service: {f['executed_tool_service_s']:.3f}s.",
            f"Paired tasks faster: {metrics['paired_tasks_faster']}/{metrics['paired_tasks']}.",
            "",
            (
                f"Validation: PASS. Work equivalence covers {b['llm_requests']} "
                f"live LLM requests and {b['tool_calls']} trace tool calls per "
                "cell; all declared checks passed."
                if valid
                else "Validation: **FAIL**. Failed checks: "
                + ", ".join(invalid_checks)
                + ". Reported metrics are diagnostic only."
            ),
            f"Frozen plan SHA-256: `{baseline['plan_sha256']}`.",
            "",
            f"Live queue: max_num_seqs={logs['baseline']['max_num_seqs']} in both cells; baseline max Running/Waiting={logs['baseline']['max_running']}/{logs['baseline']['max_waiting']}, FULL={logs['full']['max_running']}/{logs['full']['max_waiting']}.",
            f"Scheduler audit: baseline Joint hook={logs['baseline']['joint_hook_installations']}; FULL Joint hook={logs['full']['joint_hook_installations']}; FULL fail-open markers={logs['full']['fail_open_markers']}.",
            "",
            "Boundary: Pattern-v2 hit/readiness labels are frozen offline, while LLM latency, LLM queueing, shared residual-tool contention, and end-to-end wall time are measured online. This is a systems trace replay, not a new answer-quality evaluation.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(args.markdown),
                "valid": valid,
                "invalid_checks": invalid_checks,
                **metrics,
            },
            indent=2,
        )
    )
    return 0 if valid else 2


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-plan", type=Path, required=True)
    prep.add_argument("--arrivals", type=Path, required=True)
    prep.add_argument("--sessions", type=int, default=80)
    prep.add_argument("--trace-offset", type=int, default=0)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=prepare)

    cell = sub.add_parser("run-cell")
    cell.add_argument("--plan", type=Path, required=True)
    cell.add_argument("--system", choices=("baseline", "full"), required=True)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    cell.add_argument("--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    cell.add_argument("--max-active-tasks", type=int, default=80)
    cell.add_argument(
        "--preengine-policy",
        choices=("fifo", "gain-pressure"),
        default="fifo",
        help=(
            "cold-session admission policy; fifo preserves the historical "
            "Semaphore, while gain-pressure ranks a coalesced burst and holds "
            "each selected slot for the complete trace"
        ),
    )
    cell.add_argument("--preengine-coalesce-s", type=float, default=0.25)
    cell.add_argument(
        "--preengine-prefill-tokens-per-s", type=float, default=10_000.0
    )
    cell.add_argument(
        "--preengine-decode-tokens-per-s", type=float, default=500.0
    )
    cell.add_argument("--preengine-pressure-weight", type=float, default=1.0)
    cell.add_argument("--preengine-tool-gain-beta", type=float, default=1.0)
    cell.add_argument("--preengine-aging-alpha", type=float, default=0.05)
    cell.add_argument("--tool-backend", choices=("replay", "live", "timed"), default="replay")
    cell.add_argument("--predictor", type=Path)
    cell.add_argument("--speculation-capacity", type=int, default=4)
    cell.add_argument("--speculation-top-k", type=int, default=10,
        help="Execution prefix of the unchanged frozen predictor ranking")
    cell.add_argument("--speculation-ttl-s", type=float, default=60,
        help="Private candidate retention limit; origin HTTP freshness still bounds reuse")
    cell.add_argument("--speculation-pending-capacity", type=int, default=128)
    cell.add_argument("--speculation-cache-capacity", type=int, default=None,
        help="Separate completed-result capacity; omitted retains legacy combined limit")
    cell.add_argument("--session-result-cache", action="store_true",
        help="Timed backend only: retain successful speculative results across repeated demands; requires infinite TTL")
    cell.add_argument("--preempt-speculation", action="store_true",
        help="Timed backend only: cancel lowest-utility unclaimed running work when authority needs a slot")
    cell.add_argument("--replace-queued-predictions", action="store_true")
    cell.add_argument("--arrival-scale", type=float, default=1.0)
    cell.add_argument("--tool-capacity", type=int, default=16)
    cell.add_argument("--request-timeout-s", type=float, default=900.0)
    cell.set_defaults(func=lambda value: asyncio.run(run_cell(value)))

    comp = sub.add_parser("compare")
    comp.add_argument("--baseline", type=Path, required=True)
    comp.add_argument("--full", type=Path, required=True)
    comp.add_argument("--baseline-server-log", type=Path, required=True)
    comp.add_argument("--full-server-log", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    comp.add_argument("--markdown", type=Path, required=True)
    comp.set_defaults(func=compare)
    return top


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
