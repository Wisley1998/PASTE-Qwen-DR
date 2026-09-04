#!/usr/bin/env python3
"""Run one strict Qwen Pattern V2 A/B/E/F live cell.

The runner consumes the existing strict public/sealed plan format and a
separately frozen Pattern V2 predictor artifact.  It intentionally does not
fit a model or inspect recorded tool timing.  A private hashed-uniform SLO
clock determines physical tool service, while policy-visible duration hats use
only the public SLO midpoint and completed-job observations.

Pattern prediction occurs at the beginning of a current request, using only
the immediately preceding completed tool descriptor and the current request's
visible messages.  Predictions are never expired or resolved at the next
authority boundary: completed results remain in the per-session URL cache and
wrong running work is cancelled only when authority needs capacity.

This is a causal-reveal systems trace replay, not an autonomous-agent quality
evaluation: live vLLM executes every recorded request, but its response does
not choose the recorded next tool call.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
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
for import_root in (REPRODUCTION_ROOT, SCRIPT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from paste_repro import pattern_v2_all_visit_online as pattern_online  # noqa: E402
from paste_repro.pattern_v2_strict_adapter import (  # noqa: E402
    HashedUniformSLOClock,
    PatternV2PredictorLike,
    PatternV2StrictPolicy,
    PersistentPatternV2ToolExecutor,
    PublicSLODurationPredictor,
    prediction_evidence,
)
from paste_repro.strict_trace_runtime import (  # noqa: E402
    CausalSessionState,
    CausalTailPredictor,
    CausalTraceCursor,
    normalize_url,
    signed_payload,
    validate_signed_payload,
)
from paste_repro.trace_coscheduler import AsyncPreemptibleVisitPool  # noqa: E402
from run_strict_trace_abef import (  # noqa: E402
    PUBLIC_PLAN_SCHEMA,
    SEALED_PLAN_SCHEMA,
    _atomic_visit_digests,
    _fcfs_request_id,
    _llm_workload_request_sha256,
    _post_llm,
    _scheduler_request_id,
    _tool_invocation_digest,
    file_sha256,
    percentile,
    read_json,
    validate_server_policy,
    write_json,
)


RESULT_SCHEMA = "paste_repro.pattern_v2_strict_abef_result.v1"
CELL_SPECS = {
    "A": {"scheduler": "native_fcfs", "server_policy": "fcfs", "speculation": False},
    "B": {"scheduler": "native_fcfs", "server_policy": "fcfs", "speculation": True},
    "E": {
        "scheduler": "causal_joint",
        "server_policy": "online_joint_pacer_v2",
        "speculation": False,
    },
    "F": {
        "scheduler": "causal_joint",
        "server_policy": "online_joint_pacer_v2",
        "speculation": True,
    },
}
EXACT_SOURCE_ROOTS = 100
EXACT_REPLICAS = 2
EXACT_TASKS = EXACT_SOURCE_ROOTS * EXACT_REPLICAS
DEPLOYABLE_SOURCE_ROOTS = 30
DEPLOYABLE_REPLICAS = 7
DEPLOYABLE_TASKS = DEPLOYABLE_SOURCE_ROOTS * DEPLOYABLE_REPLICAS
EXACT_VISIT_CAPACITY = 64
EXACT_SPECULATIVE_CAP = 64
CROSSFIT_LOGICAL_CORPUS_SHA256 = (
    "c8eddcf9376754cc37056a1a1af7a42b5e786d7ed8c4af65d86f904431030fbc"
)
DEPLOYABLE_LOGICAL_CORPUS_SHA256 = (
    "34857c0cab48aa604db8907face0654e7b892a7a3b626cedd0188d79994030a7"
)


PostLLM = Callable[..., Awaitable[tuple[int, dict[str, Any], str]]]


@dataclass(frozen=True)
class RuntimeInputs:
    public: dict[str, Any]
    sealed: dict[str, Any]
    predictor: PatternV2PredictorLike
    duration_predictor: PublicSLODurationPredictor
    service_clock: HashedUniformSLOClock
    tail_predictor: CausalTailPredictor | None
    predictor_disclosure: dict[str, Any]
    workload_contract: str
    file_hashes: dict[str, str]
    formal_workload: bool


def _load_pattern_predictor(
    path: Path, payload: Mapping[str, Any]
) -> PatternV2PredictorLike:
    schema = payload.get("schema")
    if schema == pattern_online.SCHEMA:
        return pattern_online.PatternV2CrossFitPredictor.from_path(path)
    if schema == pattern_online.DEPLOYABLE_SCHEMA:
        return pattern_online.PatternV2DeployablePredictor.from_path(path)
    raise ValueError(f"unsupported Pattern V2 predictor schema: {schema!r}")


def _forbidden_keys(payload: Any, forbidden: set[str], *, path: str = "$.") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = f"{path}{key}"
            if str(key) in forbidden:
                found.append(child)
            found.extend(_forbidden_keys(value, forbidden, path=f"{child}."))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_forbidden_keys(value, forbidden, path=f"{path}[{index}]."))
    return found


def _validate_plan_contract(
    public: Mapping[str, Any],
    sealed: Mapping[str, Any],
    *,
    predictor_schema: str,
    allow_smoke_workload: bool,
) -> tuple[bool, str]:
    if public.get("schema") != PUBLIC_PLAN_SCHEMA:
        raise ValueError("public plan has an unsupported schema")
    if sealed.get("schema") != SEALED_PLAN_SCHEMA:
        raise ValueError("sealed plan has an unsupported schema")
    if sealed.get("public_plan_sha256") != public.get("plan_sha256"):
        raise ValueError("sealed/public plan binding mismatch")
    if public.get("role") != sealed.get("role"):
        raise ValueError("sealed/public role mismatch")
    if public.get("call_graph_mode") != "trace_replay_causal_reveal":
        raise ValueError("public plan is not a causal-reveal replay")
    traces = public.get("traces")
    steps_by_trace = sealed.get("trace_steps")
    outcomes = sealed.get("outcomes")
    if not isinstance(traces, list) or not isinstance(steps_by_trace, Mapping):
        raise ValueError("strict plans have invalid trace collections")
    if not isinstance(outcomes, Mapping):
        raise ValueError("sealed plan outcomes are invalid")
    trace_ids = [str(row.get("trace_id", "")) for row in traces]
    if not trace_ids or any(not value for value in trace_ids):
        raise ValueError("public plan contains an empty trace ID")
    if len(trace_ids) != len(set(trace_ids)) or set(trace_ids) != set(steps_by_trace):
        raise ValueError("public/sealed trace IDs are not one-to-one")
    source_ids = [str(row.get("source_session_id", "")) for row in traces]
    if any(not value for value in source_ids):
        raise ValueError("public plan contains an empty source-session ID")
    source_counts = Counter(source_ids)
    if predictor_schema == pattern_online.SCHEMA:
        expected_roots = EXACT_SOURCE_ROOTS
        expected_replicas = EXACT_REPLICAS
        expected_tasks = EXACT_TASKS
        workload_contract = "retrospective_crossfit_100_roots_x2"
        expected_logical_corpus_sha256 = CROSSFIT_LOGICAL_CORPUS_SHA256
    elif predictor_schema == pattern_online.DEPLOYABLE_SCHEMA:
        expected_roots = DEPLOYABLE_SOURCE_ROOTS
        expected_replicas = DEPLOYABLE_REPLICAS
        expected_tasks = DEPLOYABLE_TASKS
        workload_contract = "retrospective_internal_holdout_30_roots_x7"
        expected_logical_corpus_sha256 = DEPLOYABLE_LOGICAL_CORPUS_SHA256
    else:
        raise ValueError("unknown predictor schema for workload contract")
    formal = (
        len(traces) == expected_tasks
        and len(source_counts) == expected_roots
        and set(source_counts.values()) == {expected_replicas}
        and int(public.get("independent_source_roots", -1)) == expected_roots
        and int(public.get("replicas", -1)) == expected_tasks
        and int(public.get("replicas_per_root", expected_replicas))
        == expected_replicas
    )
    if not formal and not allow_smoke_workload:
        raise ValueError(
            f"formal Pattern V2 workload violates {workload_contract}"
        )
    if formal and public.get("logical_corpus_sha256") != expected_logical_corpus_sha256:
        raise ValueError("formal plan has the wrong frozen Pattern V2 logical corpus")
    for trace_id, raw_steps in steps_by_trace.items():
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"sealed trace has no steps: {trace_id}")
        for request_index, step in enumerate(raw_steps):
            if not isinstance(step, Mapping) or not isinstance(step.get("request"), Mapping):
                raise ValueError(f"malformed sealed step: {trace_id}:{request_index}")
            tools = step.get("tools_after", [])
            if not isinstance(tools, list):
                raise ValueError(f"malformed authority group: {trace_id}:{request_index}")
            # The online Pattern V2 API uses the next LLM-visible tool response.
            # More than one tool between LLMs would make that response future
            # information for the first tool, so fail closed instead of guessing.
            if len(tools) > 1:
                raise ValueError(
                    "strict Pattern V2 requires at most one completed tool between LLM turns"
                )
            for descriptor in tools:
                outcome_id = str(descriptor.get("outcome_id", ""))
                if not outcome_id or outcome_id not in outcomes:
                    raise ValueError(f"authority descriptor lacks a sealed outcome: {trace_id}")
                tool_name = str(descriptor.get("tool_name", ""))
                if tool_name not in {"search", "google_scholar", "visit"}:
                    raise ValueError(f"authority tool has no frozen SLO: {tool_name!r}")
                outcome = outcomes[outcome_id]
                if (
                    not isinstance(outcome, Mapping)
                    or str(outcome.get("tool_name")) != tool_name
                    or int(outcome.get("event_index", -1))
                    != int(descriptor.get("event_index", -2))
                ):
                    raise ValueError(f"authority/sealed outcome mismatch: {outcome_id}")
                if tool_name == "visit":
                    authority_urls = _visit_urls_from_descriptor(descriptor)
                    if any(
                        not url.startswith(("http://", "https://"))
                        for url in authority_urls
                    ):
                        raise ValueError("non-executable Visit URL entered physical plan")
                    raw_units = outcome.get("visit_units")
                    if not isinstance(raw_units, list) or any(
                        not isinstance(row, Mapping) or not isinstance(row.get("url"), str)
                        for row in raw_units
                    ):
                        raise ValueError(f"sealed Visit units are malformed: {outcome_id}")
                    sealed_urls = tuple(
                        normalize_url(str(row["url"])) for row in raw_units
                    )
                    if sealed_urls != authority_urls:
                        raise ValueError(f"sealed Visit units differ from authority: {outcome_id}")
    forbidden_timing = {
        "duration_s",
        "unit_duration_s",
        "recorded_total_service_diagnostic_s",
        "recorded_unit_service_diagnostic_s",
        "llm_overlap_s",
        "overlap_window_s",
    }
    leaked = _forbidden_keys(sealed, forbidden_timing)
    if leaked:
        raise ValueError(f"sealed execution plan contains trace timing fields: {leaked[:3]}")
    return formal, workload_contract


def load_runtime_inputs(args: argparse.Namespace) -> RuntimeInputs:
    paths = {
        "public_plan": args.public_plan.resolve(),
        "sealed_plan": args.sealed_plan.resolve(),
        "predictor_artifact": args.predictor_artifact.resolve(),
        "duration_artifact": args.duration_artifact.resolve(),
        "service_clock_artifact": args.service_clock_artifact.resolve(),
    }
    if args.tail_artifact is not None:
        paths["tail_artifact"] = args.tail_artifact.resolve()
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    predictor_payload = read_json(paths["predictor_artifact"])
    if not isinstance(predictor_payload, Mapping):
        raise ValueError("Pattern V2 predictor artifact must be an object")
    public = validate_signed_payload(
        read_json(paths["public_plan"]), "plan_sha256", label="strict public plan"
    )
    sealed = validate_signed_payload(
        read_json(paths["sealed_plan"]), "sealed_sha256", label="strict sealed plan"
    )
    formal, workload_contract = _validate_plan_contract(
        public,
        sealed,
        predictor_schema=str(predictor_payload.get("schema", "")),
        allow_smoke_workload=bool(args.allow_smoke_workload),
    )
    predictor = _load_pattern_predictor(
        paths["predictor_artifact"], predictor_payload
    )
    duration_payload = read_json(paths["duration_artifact"])
    service_payload = read_json(paths["service_clock_artifact"])
    duration = PublicSLODurationPredictor(duration_payload)
    service_clock = HashedUniformSLOClock(service_payload)
    if public.get("predictor_artifact_sha256") != predictor.artifact_sha256:
        raise ValueError("public plan is not bound to the supplied Pattern V2 predictor")
    if (
        public.get("duration_predictor_artifact_sha256")
        != duration.artifact_sha256
    ):
        raise ValueError("public plan is not bound to the supplied duration predictor")
    disclosure = {
        key: predictor_payload.get(key)
        for key in (
            "schema",
            "evaluation_regime",
            "claim_scope",
            "uses_other_evaluation_root_labels",
            "prior_policy_development_used_evaluation_corpus",
            "predictor_uses_trace_timing",
        )
        if key in predictor_payload
    }
    if public.get("predictor_disclosure") != disclosure:
        raise ValueError("public plan predictor disclosure differs from the artifact")
    disclosed_scope = disclosure.get("claim_scope") or disclosure.get(
        "evaluation_regime"
    )
    if public.get("claim_scope") != disclosed_scope:
        raise ValueError("public plan claim scope differs from the predictor artifact")
    if sealed.get("service_clock_artifact_sha256") != service_clock.artifact_sha256:
        raise ValueError("sealed plan is not bound to the supplied physical SLO clock")
    if _forbidden_keys(
        predictor_payload,
        {"duration_s", "unit_duration_s", "llm_overlap_s", "overlap_window_s"},
    ):
        raise ValueError("Pattern V2 predictor artifact contains forbidden timing labels")
    tail = None
    if args.tail_artifact is not None:
        tail = CausalTailPredictor(read_json(paths["tail_artifact"]))
        if public.get("tail_predictor_artifact_sha256") != tail.artifact_sha256:
            raise ValueError("public plan is not bound to the supplied tail predictor")
    if CELL_SPECS[args.cell]["scheduler"] == "causal_joint" and tail is None:
        raise ValueError("E/F requires --tail-artifact for causal scheduler metadata")
    # Force every source-to-fold/deployable binding check before touching live
    # execution.  The returned state is discarded; each task gets a fresh one.
    for trace in public["traces"]:
        predictor.start_session(
            source_session_id=str(trace["source_session_id"]),
            runtime_session_id=str(trace["session_id"]),
        )
    return RuntimeInputs(
        public=dict(public),
        sealed=dict(sealed),
        predictor=predictor,
        duration_predictor=duration,
        service_clock=service_clock,
        tail_predictor=tail,
        predictor_disclosure=disclosure,
        workload_contract=workload_contract,
        file_hashes={label: file_sha256(path) for label, path in paths.items()},
        formal_workload=formal,
    )


def _visit_urls_from_descriptor(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    if str(descriptor.get("tool_name")) != "visit":
        return ()
    arguments = descriptor.get("tool_args", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("authority tool arguments must be an object")
    raw = arguments.get("url")
    if isinstance(raw, str):
        return (normalize_url(raw),) if raw else ()
    if isinstance(raw, list):
        return tuple(normalize_url(str(value)) for value in raw if isinstance(value, str) and value)
    return ()


async def execute_cell(
    args: argparse.Namespace,
    loaded: RuntimeInputs,
    *,
    post_llm: PostLLM = _post_llm,
) -> dict[str, Any]:
    """Execute one cell; ``post_llm`` is injectable for a CPU-only test."""

    cell = CELL_SPECS[args.cell]
    treatment = bool(cell["speculation"])
    if args.visit_capacity != EXACT_VISIT_CAPACITY:
        raise ValueError("strict Pattern V2 fixes Visit capacity at 64")
    if args.speculative_cap != EXACT_SPECULATIVE_CAP:
        raise ValueError("strict Pattern V2 fixes speculative cap at 64")
    if args.max_active_tasks <= 0:
        raise ValueError("max active tasks must be positive")
    public = loaded.public
    sealed = loaded.sealed
    policy = PatternV2StrictPolicy(
        predictor=loaded.predictor,
        duration_predictor=loaded.duration_predictor,
        tail_predictor=loaded.tail_predictor,
    )
    decision_context: dict[str, tuple[str, int]] = {}
    job_transitions: list[dict[str, Any]] = []

    def record_job_event(raw: dict[str, Any]) -> None:
        prediction_id = str(raw["prediction_id"])
        trace_id, request_index = decision_context[prediction_id]
        job_transitions.append(
            {
                **{
                    key: value
                    for key, value in raw.items()
                    if key not in {"url", "session_id"}
                },
                "trace_id": trace_id,
                "request_index": request_index,
                "candidate_invocation_digest": _tool_invocation_digest(
                    "visit", {"url": str(raw["url"])}
                ),
            }
        )

    pool = AsyncPreemptibleVisitPool(
        capacity=EXACT_VISIT_CAPACITY,
        speculative_cap=EXACT_SPECULATIVE_CAP,
        job_event_callback=record_job_event,
    )
    executor = PersistentPatternV2ToolExecutor(
        sealed_outcomes=sealed["outcomes"],
        service_clock=loaded.service_clock,
        duration_predictor=loaded.duration_predictor,
        visit_pool=pool,
    )
    request_events: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    prediction_decisions: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    request_attempt_counts: Counter[tuple[str, int]] = Counter()
    # Raw URLs are kept only in memory long enough to derive post-reveal labels.
    prediction_candidates: dict[str, list[dict[str, Any]]] = {}
    predictions_by_session: dict[str, list[str]] = {}
    result_lock = asyncio.Lock()
    gate = asyncio.Semaphore(args.max_active_tasks)
    experiment_started = time.monotonic()

    async def execute_trace(
        trace: Mapping[str, Any], http_session: aiohttp.ClientSession
    ) -> None:
        release = float(trace.get("release_offset_s", 0.0))
        if not math.isfinite(release) or release < 0.0:
            raise ValueError("release offset must be finite and non-negative")
        scheduled = experiment_started + release
        delay = scheduled - time.monotonic()
        if delay > 0.0:
            await asyncio.sleep(delay)
        released = time.monotonic()
        acquired = released
        session_id = str(trace["session_id"])
        source_session_id = str(trace["source_session_id"])
        cursor = CausalTraceCursor(sealed["trace_steps"][session_id])
        pattern_session = (
            policy.start_session(
                source_session_id=source_session_id,
                runtime_session_id=session_id,
            )
            if treatment
            else None
        )
        state = CausalSessionState(
            predicted_output_tokens=float(args.default_predicted_output_tokens)
        )
        pending_completed_tool: dict[str, Any] | None = None
        causal_seq = 0
        task_llm_s = 0.0
        task_tool_s = 0.0
        task_saved_s = 0.0
        task_prediction_s = 0.0
        failure: str | None = None
        try:
            async with gate:
                acquired = time.monotonic()
                while not cursor.done:
                    request_index = cursor.request_index
                    request = cursor.current_request()
                    observed_seq = causal_seq
                    causal_seq += 1
                    decision_seq = causal_seq
                    candidates = ()
                    admitted: tuple[bool, ...] = ()
                    prediction_record: dict[str, Any] | None = None
                    if treatment and pending_completed_tool is not None:
                        assert pattern_session is not None
                        prediction_started = time.monotonic()
                        candidates = pattern_session.predict_after_completed_tool(
                            tool_name=str(pending_completed_tool["tool_name"]),
                            tool_arguments=pending_completed_tool.get("tool_args", {}),
                            current_messages=request["messages"],
                        )
                        decided_at = time.monotonic()
                        prediction_id = (
                            f"{session_id}:after:{pending_completed_tool['event_index']}:"
                            f"before:{request_index}"
                        )
                        decision_context[prediction_id] = (session_id, request_index)
                        admitted = await executor.speculate(
                            session_id=session_id,
                            candidates=candidates,
                            decision_id=prediction_id,
                        )
                        prediction_finished = time.monotonic()
                        task_prediction_s += prediction_finished - prediction_started
                        internal_rows = [
                            {
                                "url": normalize_url(candidate.url),
                                "candidate": candidate,
                                "admitted": bool(was_admitted),
                                "matched_event_index": None,
                            }
                            for candidate, was_admitted in zip(
                                candidates, admitted, strict=True
                            )
                        ]
                        prediction_candidates[prediction_id] = internal_rows
                        predictions_by_session.setdefault(session_id, []).append(
                            prediction_id
                        )
                        prediction_record = {
                                "record_type": "prediction_decision",
                                "prediction_id": prediction_id,
                                "trace_id": session_id,
                                "source_session_id_sha256": hashlib.sha256(
                                    source_session_id.encode("utf-8")
                                ).hexdigest(),
                                "request_index": request_index,
                                "trigger_event_index": int(
                                    pending_completed_tool["event_index"]
                                ),
                                "trigger_tool": str(
                                    pending_completed_tool["tool_name"]
                                ),
                                "observed_event_seq": observed_seq,
                                "decision_seq": decision_seq,
                                "decided_at_monotonic_s": decided_at,
                                "predictor_artifact_sha256": (
                                    pattern_session.predictor_artifact_sha256
                                ),
                                "duration_predictor_artifact_sha256": (
                                    loaded.duration_predictor.artifact_sha256
                                ),
                                "candidate_ranking": "exact_probability_only",
                                "candidates": [
                                    {
                                        **{
                                            key: value
                                            for key, value in prediction_evidence(
                                                candidate, admitted=was_admitted
                                            ).items()
                                            if key != "url"
                                        },
                                        "candidate_invocation_digest": (
                                            _tool_invocation_digest(
                                                "visit", {"url": candidate.url}
                                            )
                                        ),
                                    }
                                    for candidate, was_admitted in zip(
                                        candidates, admitted, strict=True
                                    )
                                ],
                                "prediction_latency_s": (
                                    prediction_finished - prediction_started
                                ),
                            }
                        prediction_decisions.append(prediction_record)
                    meta = None
                    if cell["scheduler"] == "causal_joint":
                        meta = policy.scheduler_metadata(
                            trace_id=session_id,
                            request_index=request_index,
                            current_call_index=int(request["call_index"]),
                            prompt_tokens=int(request["prompt_tokens"]),
                            max_tokens=int(request["max_tokens"]),
                            state=state,
                            observed_event_seq=observed_seq,
                            decision_seq=decision_seq,
                        )
                    llm_started = time.monotonic()
                    if prediction_record is not None:
                        prediction_record["llm_started_at_monotonic_s"] = llm_started
                        prediction_record["llm_started_after_decision_invariant"] = (
                            llm_started
                            >= float(prediction_record["decided_at_monotonic_s"])
                        )
                        if not prediction_record[
                            "llm_started_after_decision_invariant"
                        ]:
                            raise RuntimeError("LLM started before Pattern V2 decision")
                    max_request_attempts = int(
                        getattr(args, "max_request_attempts", 1)
                    )
                    request_attempt = 0
                    while True:
                        request_attempt += 1
                        request_attempt_counts[(session_id, request_index)] += 1
                        try:
                            status, usage, content = await post_llm(
                                http_session,
                                request_url=(
                                    f"{args.server_url.rstrip('/')}"
                                    "/v1/chat/completions"
                                ),
                                model=args.model,
                                request=request,
                                request_id=(
                                    _scheduler_request_id(meta)
                                    if meta is not None
                                    else _fcfs_request_id(
                                        session_id, request_index
                                    )
                                ),
                                timeout_s=args.request_timeout_s,
                            )
                            break
                        except aiohttp.ServerDisconnectedError:
                            if request_attempt >= max_request_attempts:
                                raise
                            # A retry is diagnostic transport recovery only;
                            # it reuses the identical semantic request and is
                            # explicitly counted in the result artifact.
                            await asyncio.sleep(0)
                    llm_finished = time.monotonic()
                    task_llm_s += llm_finished - llm_started
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    prompt_tokens = int(usage.get("prompt_tokens", -1) or -1)
                    if not args.allow_usage_mismatch and (
                        completion_tokens != int(request["max_tokens"])
                        or prompt_tokens != int(request["prompt_tokens"])
                    ):
                        raise RuntimeError("live LLM token work differs from the sealed request")
                    state.observe_llm_completion(completion_tokens)
                    causal_seq += 1
                    llm_completed_seq = causal_seq
                    async with result_lock:
                        request_events.append(
                            {
                                "trace_id": session_id,
                                "request_index": request_index,
                                "call_index": request["call_index"],
                                "workload_request_sha256": _llm_workload_request_sha256(
                                    model=args.model, request=request
                                ),
                                "http_status": status,
                                "request_attempts": request_attempt,
                                "latency_s": llm_finished - llm_started,
                                "prompt_tokens": request["prompt_tokens"],
                                "public_max_tokens": request["max_tokens"],
                                "usage": usage,
                                "response_sha256": hashlib.sha256(
                                    content.encode("utf-8")
                                ).hexdigest(),
                                "llm_completed_seq": llm_completed_seq,
                                "llm_completed_at_monotonic_s": llm_finished,
                                "scheduler_metadata": meta,
                            }
                        )
                    cursor.mark_llm_completed()
                    tools = cursor.reveal_authoritative_tools()
                    pending_completed_tool = None
                    group_exposed = 0.0
                    for descriptor in tools:
                        causal_seq += 1
                        revealed_seq = causal_seq
                        revealed_at = time.monotonic()
                        authority_urls = set(_visit_urls_from_descriptor(descriptor))
                        if authority_urls:
                            for prediction_id in predictions_by_session.get(
                                session_id, ()
                            ):
                                for row in prediction_candidates[prediction_id]:
                                    if (
                                        row["matched_event_index"] is None
                                        and row["url"] in authority_urls
                                    ):
                                        row["matched_event_index"] = int(
                                            descriptor["event_index"]
                                        )
                        observation = await executor.execute_authoritative(
                            session_id=session_id, descriptor=descriptor
                        )
                        completed_at = time.monotonic()
                        causal_seq += 1
                        group_exposed += observation.exposed_wait_s
                        task_tool_s += observation.exposed_wait_s
                        task_saved_s += observation.saved_service_s
                        pending_completed_tool = dict(descriptor)
                        async with result_lock:
                            tool_events.append(
                                {
                                    "trace_id": session_id,
                                    "request_index": request_index,
                                    "event_index": descriptor["event_index"],
                                    "tool_name": descriptor["tool_name"],
                                    "authority_invocation_digest": _tool_invocation_digest(
                                        str(descriptor["tool_name"]),
                                        descriptor.get("tool_args", {}),
                                    ),
                                    "authority_candidate_invocation_digests": sorted(
                                        _atomic_visit_digests(descriptor)
                                    ),
                                    "llm_completed_seq": llm_completed_seq,
                                    "authoritative_revealed_seq": revealed_seq,
                                    "authoritative_revealed_at_monotonic_s": revealed_at,
                                    "tool_completed_seq": causal_seq,
                                    "tool_completed_at_monotonic_s": completed_at,
                                    "exposed_wait_s": observation.exposed_wait_s,
                                    "service_s": observation.service_s,
                                    "saved_service_s": observation.saved_service_s,
                                    "visit_results": [
                                        {
                                            "source": row.source,
                                            "exposed_wait_s": row.exposed_wait_s,
                                            "service_s": row.service_s,
                                            "saved_service_s": row.saved_service_s,
                                        }
                                        for row in observation.visit_results
                                    ],
                                }
                            )
                    if tools:
                        assert pending_completed_tool is not None
                        state.observe_tool_group(
                            tool_name=str(pending_completed_tool["tool_name"]),
                            event_index=int(pending_completed_tool["event_index"]),
                            exposed_wait_s=group_exposed,
                        )
                    cursor.advance()
        except Exception as exc:
            failure = repr(exc)
        finally:
            await executor.close_session(session_id)
            finished = time.monotonic()
            async with result_lock:
                task_rows.append(
                    {
                        "trace_id": session_id,
                        "source_session_id_sha256": hashlib.sha256(
                            source_session_id.encode("utf-8")
                        ).hexdigest(),
                        "release_offset_s": release,
                        "release_lag_s": released - scheduled,
                        "task_gate_wait_s": acquired - scheduled,
                        "flow_s": finished - scheduled,
                        "llm_s": task_llm_s,
                        "tool_exposed_s": task_tool_s,
                        "saved_tool_service_s": task_saved_s,
                        "prediction_overhead_s": task_prediction_s,
                        "failure": failure,
                    }
                )

    connector = aiohttp.TCPConnector(limit=0)
    headers = (
        {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
        if os.environ.get("VLLM_API_KEY")
        else {}
    )
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        await asyncio.gather(
            *(execute_trace(trace, session) for trace in public["traces"])
        )
    experiment_finished = time.monotonic()
    visit_snapshot = executor.snapshot()
    await executor.close()
    prediction_outcomes: list[dict[str, Any]] = []
    for decision in prediction_decisions:
        prediction_id = str(decision["prediction_id"])
        rows = prediction_candidates[prediction_id]
        prediction_outcomes.append(
            {
                "record_type": "prediction_outcome",
                "prediction_id": prediction_id,
                "trace_id": decision["trace_id"],
                "outcome_scope": "any_later_same_session_authoritative_visit",
                "candidates": [
                    {
                        "candidate_invocation_digest": _tool_invocation_digest(
                            "visit", {"url": row["url"]}
                        ),
                        "admitted": row["admitted"],
                        "matched_future_authority": row["matched_event_index"] is not None,
                        "first_matched_event_index": row["matched_event_index"],
                    }
                    for row in rows
                ],
                "decision_hit": any(
                    row["matched_event_index"] is not None for row in rows
                ),
            }
        )
    metrics = visit_snapshot["metrics"]
    authority = int(metrics.get("authority_requests", 0))
    physical = int(metrics.get("physical_authority_starts", 0)) + int(
        metrics.get("physical_speculative_starts", 0)
    )
    flows = [float(row["flow_s"]) for row in task_rows]
    llm_latencies = [float(row["latency_s"]) for row in request_events]
    emitted = sum(len(row["candidates"]) for row in prediction_outcomes)
    matched = sum(
        int(candidate["matched_future_authority"])
        for row in prediction_outcomes
        for candidate in row["candidates"]
    )
    summary = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cell": args.cell,
        "scheduler": cell["scheduler"],
        "speculation": treatment,
        "evaluation_mode": "trace_replay_causal_reveal",
        "formal_workload": loaded.formal_workload,
        "workload_contract": loaded.workload_contract,
        "claim_scope": loaded.predictor_disclosure.get(
            "claim_scope",
            loaded.predictor_disclosure.get("evaluation_regime", "retrospective"),
        ),
        "confirmatory_claim_allowed": False,
        "public_plan_sha256": public["plan_sha256"],
        "sealed_plan_sha256": sealed["sealed_sha256"],
        "predictor_artifact_sha256": loaded.predictor.artifact_sha256,
        "predictor_disclosure": dict(loaded.predictor_disclosure),
        "duration_predictor_artifact_sha256": (
            loaded.duration_predictor.artifact_sha256
        ),
        "tail_predictor_artifact_sha256": (
            loaded.tail_predictor.artifact_sha256
            if loaded.tail_predictor is not None
            else None
        ),
        "physical_service_clock_sha256": loaded.service_clock.artifact_sha256,
        "frozen_input_file_sha256": dict(loaded.file_hashes),
        "configuration": {
            "model": args.model,
            "max_active_tasks": args.max_active_tasks,
            "visit_capacity": EXACT_VISIT_CAPACITY,
            "speculative_cap": EXACT_SPECULATIVE_CAP,
            "candidate_ranking": "exact_probability_only",
            "cache_scope": "session_url_infinite_ttl",
            "wrong_candidate_retirement": (
                "authority_capacity_pressure_or_session_close"
            ),
            "max_request_attempts": int(
                getattr(args, "max_request_attempts", 1)
            ),
        },
        "tasks": len(task_rows),
        "source_roots": len({row["source_session_id"] for row in public["traces"]}),
        "requests": len(request_events),
        "llm_request_attempts": sum(request_attempt_counts.values()),
        "retried_requests": sum(
            value > 1 for value in request_attempt_counts.values()
        ),
        "tool_events": len(tool_events),
        "failures": sum(row["failure"] is not None for row in task_rows),
        "experiment_wall_s": experiment_finished - experiment_started,
        "mean_task_flow_s": statistics.fmean(flows) if flows else 0.0,
        "p50_task_flow_s": percentile(flows, 0.50),
        "p95_task_flow_s": percentile(flows, 0.95),
        "mean_llm_latency_s": (
            statistics.fmean(llm_latencies) if llm_latencies else 0.0
        ),
        "mean_tool_exposed_s_per_task": statistics.fmean(
            [float(row["tool_exposed_s"]) for row in task_rows]
        ) if task_rows else 0.0,
        "mean_saved_tool_service_s_per_task": statistics.fmean(
            [float(row["saved_tool_service_s"]) for row in task_rows]
        ) if task_rows else 0.0,
        "visit": visit_snapshot,
        "realized_visit_hit_rate": (
            float(metrics.get("cache_hits", 0)) / authority if authority else 0.0
        ),
        "visit_call_amplification": physical / authority if authority else 0.0,
        "prediction_candidates": emitted,
        "future_matched_prediction_candidates": matched,
        "future_candidate_precision": matched / emitted if emitted else 0.0,
    }
    summary = signed_payload(summary, "result_sha256")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "request_events.json", request_events)
    write_json(args.output_dir / "tool_events.json", tool_events)
    write_json(args.output_dir / "prediction_decisions.json", prediction_decisions)
    write_json(args.output_dir / "prediction_outcomes.json", prediction_outcomes)
    write_json(args.output_dir / "speculation_execution_events.json", job_transitions)
    write_json(args.output_dir / "task_results.json", task_rows)
    result_paths = sorted(
        path for path in args.output_dir.iterdir() if path.is_file()
    )
    result_manifest = signed_payload(
        {
            "schema": "paste_repro.pattern_v2_strict_result_manifest.v1",
            "cell": args.cell,
            "result_sha256": summary["result_sha256"],
            "files": {
                path.name: file_sha256(path) for path in result_paths
            },
        },
        "manifest_sha256",
    )
    write_json(args.output_dir / "result_manifest.json", result_manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-plan", type=Path, required=True)
    parser.add_argument("--sealed-plan", type=Path, required=True)
    parser.add_argument("--predictor-artifact", type=Path, required=True)
    parser.add_argument("--duration-artifact", type=Path, required=True)
    parser.add_argument("--service-clock-artifact", type=Path, required=True)
    parser.add_argument("--tail-artifact", type=Path)
    parser.add_argument("--cell", choices=sorted(CELL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8100")
    parser.add_argument("--server-policy-file", type=Path, required=True)
    parser.add_argument(
        "--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    )
    parser.add_argument("--max-active-tasks", type=int, default=16)
    parser.add_argument("--visit-capacity", type=int, default=64)
    parser.add_argument("--speculative-cap", type=int, default=64)
    parser.add_argument("--default-predicted-output-tokens", type=float, default=128.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--max-request-attempts",
        type=int,
        default=1,
        help="retry ServerDisconnectedError only; every attempt is recorded",
    )
    parser.add_argument(
        "--allow-smoke-workload",
        action="store_true",
        help="allow fewer than 100 roots x 2 replicas; result is marked non-formal",
    )
    parser.add_argument(
        "--allow-usage-mismatch",
        action="store_true",
        help="diagnostic only; formal runs must leave this disabled",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.max_active_tasks <= 0:
        raise ValueError("--max-active-tasks must be positive")
    if args.visit_capacity != EXACT_VISIT_CAPACITY:
        raise ValueError("--visit-capacity is frozen at 64")
    if args.speculative_cap != EXACT_SPECULATIVE_CAP:
        raise ValueError("--speculative-cap is frozen at 64")
    if args.default_predicted_output_tokens <= 0 or args.request_timeout_s <= 0:
        raise ValueError("prediction and timeout values must be positive")
    if args.max_request_attempts <= 0:
        raise ValueError("--max-request-attempts must be positive")
    if args.allow_usage_mismatch and not args.allow_smoke_workload:
        raise ValueError("usage mismatch is permitted only in a non-formal smoke")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    validate_server_policy(
        args.server_policy_file.resolve(), CELL_SPECS[args.cell]["server_policy"]
    )
    loaded = load_runtime_inputs(args)
    startup_hashes = dict(loaded.file_hashes)
    result = asyncio.run(execute_cell(args, loaded))
    paths = {
        "public_plan": args.public_plan.resolve(),
        "sealed_plan": args.sealed_plan.resolve(),
        "predictor_artifact": args.predictor_artifact.resolve(),
        "duration_artifact": args.duration_artifact.resolve(),
        "service_clock_artifact": args.service_clock_artifact.resolve(),
    }
    if args.tail_artifact is not None:
        paths["tail_artifact"] = args.tail_artifact.resolve()
    end_hashes = {label: file_sha256(path) for label, path in paths.items()}
    if end_hashes != startup_hashes:
        raise RuntimeError("a frozen plan/model/clock artifact changed during the cell")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "cell",
                    "formal_workload",
                    "tasks",
                    "source_roots",
                    "requests",
                    "failures",
                    "experiment_wall_s",
                    "mean_task_flow_s",
                    "p95_task_flow_s",
                    "realized_visit_hit_rate",
                    "visit_call_amplification",
                )
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
