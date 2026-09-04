#!/usr/bin/env python3
"""Run a paired live-LLM/offline-tool replay of real Qwen DR traces.

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
import time
from typing import Any, Mapping, Sequence

import aiohttp


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
    arrivals = read_json(args.arrivals)
    arrival_rows = arrivals.get("arrivals") or arrivals.get("traces")
    if not isinstance(arrival_rows, list) or len(arrival_rows) < args.sessions:
        raise ValueError("arrival file contains too few rows")
    if len(source.get("traces", [])) < args.sessions:
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
        zip(source["traces"][:args.sessions], arrival_rows[:args.sessions], strict=True)
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
            "session_identity": "first 80 distinct real DR sessions; no replication",
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


async def run_cell(args: argparse.Namespace) -> int:
    plan = checked_plan(args.plan)
    full = args.system == "full"
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
        release = float(trace["release_offset_s"])
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
            try:
                steps = list(trace["steps"])
                for request_index, step in enumerate(steps):
                    request = step["request"]
                    next_tools = list(step["tools_after"])
                    fixed_completion = int(request["fixed_completion_tokens"])
                    metadata = build_scheduler_metadata(
                        trace,
                        request_index,
                        full=full,
                        po_ema=po_ema,
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

                    for tool in next_tools:
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
                        "ok": error is None,
                        "error": error,
                    }
                )

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as http:
        await asyncio.gather(*(run_one(trace, http) for trace in plan["traces"]))
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
        "executed_tool_service_s": sum(float(row["executed_service_s"]) for row in tool_events),
        "exposed_tool_wait_s": sum(float(row["exposed_wait_s"]) for row in tool_events),
        "saved_tool_service_s": sum(float(row["offline_saved_s"]) for row in tool_events),
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
            "session_persistent_admission": (
                args.preengine_policy == "gain-pressure"
            ),
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
