#!/usr/bin/env python3
"""Recompute the registered strict A/B/E/F paper table from bound cell results.

The analyzer is intentionally fail-closed.  It verifies the manifest and every
result binding, requires identical task/request work across cells, folds load
replicas inside source root and block, then folds blocks inside source root.
Confidence intervals resample paired source-root vectors, never turns, replica
instances, requests, or four-cell server processes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

import audit_strict_causal_experiment as strict_audit


SCHEMA = "paste.paper.strict_causal_abef_analysis.v1"
RELATIVE_CONTRASTS: dict[str, tuple[str, str]] = {
    "A_vs_B": ("A", "B"),
    "A_vs_E": ("A", "E"),
    "E_vs_F": ("E", "F"),
    "B_vs_F": ("B", "F"),
    "A_vs_F": ("A", "F"),
}

DESCRIPTIVE_METRIC_FIELDS = (
    "experiment_wall_s",
    "makespan_s",
    "successful_tasks",
    "failures",
    "task_throughput_per_s",
    "requests",
    "request_latency_mean_s",
    "request_latency_p95_s",
    "prompt_tokens",
    "completion_tokens",
    "prediction_decisions_emitted",
    "prediction_candidates_emitted",
    "prediction_candidates_broker_accepted",
    "prediction_candidates_physical_started",
    "prediction_candidates_admitted",
    "physical_speculative_starts",
    "exact_emitted_post_authority_hits",
    "exact_broker_accepted_post_authority_hits",
    "exact_physical_started_post_authority_hits",
    "exact_admitted_post_authority_hits",
    "exact_post_authority_hits",
    "emitted_prediction_precision",
    "broker_accepted_prediction_precision",
    "physical_started_prediction_precision",
    "admitted_prediction_precision",
    "prediction_precision",
    "prediction_abstention_rate",
    "speculative_worker_s",
    "promoted_demand_worker_s",
    "direct_demand_worker_s",
    "total_worker_s",
    "useful_speculative_worker_s",
    "wasted_speculative_worker_s",
    "duration_predictor_mae_s",
)


class AnalysisError(ValueError):
    """A fail-closed protocol or work-equivalence violation."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{label}: expected finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{label}: expected finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise AnalysisError(f"{label}: expected {'positive ' if positive else ''}finite number")
    return number


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise AnalysisError("cannot take a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper or ordered[lower] == ordered[upper]:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _root_id(task: Mapping[str, Any], *, label: str) -> str:
    for field in (
        "source_root_id",
        "source_session_id",
        "root_id",
        "template_session_id",
        "session_id",
    ):
        value = task.get(field)
        if isinstance(value, str) and value:
            return value
    raise AnalysisError(f"{label}: task lacks a source-root identity")


def _instance_id(task: Mapping[str, Any], *, label: str) -> str:
    for field in ("task_id", "root_instance_id", "trace_id"):
        value = task.get(field)
        if isinstance(value, str) and value:
            return value
    raise AnalysisError(f"{label}: task lacks a stable replica/instance identity")


def _task_success(task: Mapping[str, Any]) -> bool:
    if "ok" in task:
        return task.get("ok") is True
    if "failure" in task:
        return task.get("failure") in (None, "")
    if "error" in task:
        return task.get("error") in (None, "")
    return False


def _task_e2e(task: Mapping[str, Any], *, label: str) -> float:
    scheduled = _finite(
        task.get("scheduled_release_monotonic_s"),
        label=f"{label}.scheduled_release_monotonic_s",
    )
    terminal = _finite(
        task.get("task_terminal_monotonic_s"),
        label=f"{label}.task_terminal_monotonic_s",
    )
    derived = terminal - scheduled
    if derived <= 0.0:
        raise AnalysisError(f"{label}: raw terminal-minus-scheduled E2E must be positive")
    # These are redundant runner checksums, never the analysis input.  Retain
    # this local check in addition to the shared result audit so future callers
    # cannot accidentally restore trust in a summary-only duration.
    summaries = [field for field in ("e2e_s", "flow_s") if field in task]
    if not summaries:
        raise AnalysisError(f"{label}: task lacks a redundant e2e_s/flow_s checksum")
    for field in summaries:
        summary = _finite(task[field], label=f"{label}.{field}", positive=True)
        if not math.isclose(summary, derived, rel_tol=1e-6, abs_tol=1e-3):
            raise AnalysisError(
                f"{label}.{field}: does not equal raw terminal-minus-scheduled E2E"
            )
    return derived


def _request_task_id(event: Mapping[str, Any], *, label: str) -> str:
    for field in ("task_id", "trace_id", "root_instance_id"):
        value = event.get(field)
        if isinstance(value, str) and value:
            return value
    raise AnalysisError(f"{label}: request lacks task identity")


def _request_index(event: Mapping[str, Any], *, label: str) -> int:
    for field in ("request_index", "call_index", "response_index"):
        value = event.get(field)
        if type(value) is int and value >= 0:
            return value
    raise AnalysisError(f"{label}: request lacks a non-negative index")


def _request_work_signature(event: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    digest = event.get("workload_request_sha256")
    if strict_audit._is_sha256(digest):
        work: dict[str, Any] = {"workload_request_sha256": digest}
    else:
        usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
        prompt = event.get("prompt_tokens", usage.get("prompt_tokens"))
        public_max = event.get("public_max_tokens", event.get("max_tokens"))
        if type(prompt) is not int or prompt < 0:
            raise AnalysisError(
                f"{label}: need workload_request_sha256 or explicit prompt_tokens"
            )
        if type(public_max) is not int or public_max <= 0:
            raise AnalysisError(
                f"{label}: need workload_request_sha256 or explicit public_max_tokens"
            )
        work = {"prompt_tokens": prompt, "public_max_tokens": public_max}
    usage = event.get("usage")
    if not isinstance(usage, Mapping):
        raise AnalysisError(f"{label}.usage: missing")
    completion = usage.get("completion_tokens")
    if type(completion) is not int or completion < 0:
        raise AnalysisError(f"{label}.usage.completion_tokens: invalid")
    work["model_completion_tokens"] = completion
    return work


def _authority_tool_work_signature(
    event: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Normalize only treatment-invariant authoritative tool work."""

    if strict_audit._is_sha256(event.get("authority_invocation_digest")):
        invocation_digest = str(event["authority_invocation_digest"])
        result_identity = event.get("outcome_id")
        if not strict_audit._is_sha256(result_identity):
            raise AnalysisError(f"{label}.outcome_id: invalid Qwen result identity")
        state_accepted = event.get("state_accepted")
        if state_accepted is not None and type(state_accepted) is not bool:
            raise AnalysisError(f"{label}.state_accepted: expected boolean or absent")
    elif strict_audit._is_sha256(event.get("authority_key_sha256")):
        invocation_digest = str(event["authority_key_sha256"])
        result_identity = event.get("result_sha256")
        if not strict_audit._is_sha256(result_identity):
            raise AnalysisError(f"{label}.result_sha256: invalid Gemini result identity")
        state_accepted = event.get("state_accepted")
        if type(state_accepted) is not bool:
            raise AnalysisError(f"{label}.state_accepted: boolean required")
    else:
        raise AnalysisError(f"{label}: missing authoritative invocation digest")
    tool_name = event.get("tool_name", event.get("tool"))
    if not isinstance(tool_name, str) or not tool_name:
        raise AnalysisError(f"{label}: missing authoritative tool name")
    assigned = event.get("assigned_service_s")
    if assigned is None:
        assigned = event.get("execution_surface_service_s")
    assigned_service = _finite(assigned, label=f"{label}.assigned_service_s")
    if assigned_service < 0.0:
        raise AnalysisError(f"{label}.assigned_service_s: expected non-negative")
    return {
        "authority_invocation_digest": invocation_digest,
        "tool_name": tool_name,
        "result_identity_sha256": str(result_identity),
        "state_accepted": state_accepted,
        "assigned_service_s": assigned_service,
    }


def _optional_finite(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label=label)


def _first_number(
    candidates: Sequence[tuple[Any, str]], *, nonnegative: bool = True
) -> float | None:
    for value, label in candidates:
        if value is None:
            continue
        number = _finite(value, label=label)
        if nonnegative and number < 0.0:
            raise AnalysisError(f"{label}: expected non-negative finite number")
        return number
    return None


def _descriptive_metrics(
    payload: Mapping[str, Any],
    *,
    tasks_raw: Sequence[Any],
    requests_raw: Sequence[Any],
    label: str,
) -> dict[str, Any]:
    """Normalize mechanism/system metrics without inventing missing evidence."""

    summary = payload.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    started_wall = _optional_finite(
        payload.get("started_wall_s"), label=f"{label}.started_wall_s"
    )
    ended_wall = _optional_finite(
        payload.get("ended_wall_s"), label=f"{label}.ended_wall_s"
    )
    wall_interval = (
        ended_wall - started_wall
        if started_wall is not None and ended_wall is not None
        else None
    )
    if wall_interval is not None and wall_interval < 0.0:
        raise AnalysisError(f"{label}: ended_wall_s predates started_wall_s")
    experiment_wall = _first_number(
        (
            (payload.get("experiment_wall_s"), f"{label}.experiment_wall_s"),
            (summary.get("makespan_s"), f"{label}.summary.makespan_s"),
            (wall_interval, f"{label}.wall_interval_s"),
        )
    )
    makespan = _first_number(
        (
            (summary.get("makespan_s"), f"{label}.summary.makespan_s"),
            (payload.get("experiment_wall_s"), f"{label}.experiment_wall_s"),
            (wall_interval, f"{label}.wall_interval_s"),
        )
    )

    successful = sum(
        isinstance(row, Mapping) and _task_success(row) for row in tasks_raw
    )
    failures = len(tasks_raw) - successful
    request_latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    for index, raw in enumerate(requests_raw):
        if not isinstance(raw, Mapping):
            continue
        latency = raw.get("latency_s")
        if latency is not None:
            value = _finite(latency, label=f"{label}.llm_events[{index}].latency_s")
            if value < 0.0:
                raise AnalysisError(
                    f"{label}.llm_events[{index}].latency_s: expected non-negative"
                )
            request_latencies.append(value)
        usage = raw.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        prompt = raw.get("prompt_tokens", usage.get("prompt_tokens"))
        completion = usage.get("completion_tokens")
        if type(prompt) is int and prompt >= 0:
            prompt_tokens += prompt
        if type(completion) is int and completion >= 0:
            completion_tokens += completion

    decisions_raw = payload.get("prediction_decisions")
    decisions = (
        [row for row in decisions_raw if isinstance(row, Mapping)]
        if isinstance(decisions_raw, list)
        else []
    )
    emitted_candidate_keys: set[tuple[str, str]] = set()
    broker_accepted_candidate_keys: set[tuple[str, str]] = set()
    candidate_request_identity: dict[tuple[str, str], tuple[str, int]] = {}
    decision_request_keys: set[tuple[str, int]] = set()
    for decision_index, decision in enumerate(decisions):
        prediction_id = decision.get("prediction_id")
        if not isinstance(prediction_id, str) or not prediction_id:
            raise AnalysisError(
                f"{label}.prediction_decisions[{decision_index}].prediction_id: missing"
            )
        candidates = decision.get("candidates")
        candidate_rows = (
            [row for row in candidates if isinstance(row, Mapping)]
            if isinstance(candidates, list)
            else []
        )
        if not candidate_rows:
            candidate_rows = [decision]
        trace = decision.get("trace_id", decision.get("task_id"))
        request_index = decision.get("request_index")
        if not isinstance(trace, str) or type(request_index) is not int:
            raise AnalysisError(
                f"{label}.prediction_decisions[{decision_index}]: "
                "missing trace/request identity"
            )
        request_identity = (trace, request_index)
        decision_request_keys.add(request_identity)
        for candidate_index, candidate in enumerate(candidate_rows):
            digest = candidate.get("candidate_invocation_digest")
            if not strict_audit._is_sha256(digest):
                raise AnalysisError(
                    f"{label}.prediction_decisions[{decision_index}]."
                    f"candidates[{candidate_index}]: invalid candidate digest"
                )
            candidate_key = (prediction_id, str(digest))
            if candidate_key in emitted_candidate_keys:
                raise AnalysisError(
                    f"{label}.prediction_decisions[{decision_index}]: "
                    "duplicate prediction/candidate identity"
                )
            emitted_candidate_keys.add(candidate_key)
            candidate_request_identity[candidate_key] = request_identity
            # In strict results the legacy candidate-level ``admitted`` flag
            # means broker acceptance only.  It is never the denominator of
            # the physical/admitted precision estimand.
            broker_accepted = candidate.get(
                "broker_accepted", candidate.get("admitted")
            )
            if broker_accepted is True:
                broker_accepted_candidate_keys.add(candidate_key)

    emitted_candidates = len(emitted_candidate_keys)
    broker_accepted_candidates = len(broker_accepted_candidate_keys)

    execution_raw = payload.get("speculation_execution_events")
    executions = (
        [row for row in execution_raw if isinstance(row, Mapping)]
        if isinstance(execution_raw, list)
        else []
    )
    execution_candidate_keys: set[tuple[str, str]] = set()
    physical_started_candidate_keys: set[tuple[str, str]] = set()
    for execution_index, execution in enumerate(executions):
        prediction_id = execution.get("prediction_id")
        digest = execution.get("candidate_invocation_digest")
        if not isinstance(prediction_id, str) or not strict_audit._is_sha256(digest):
            raise AnalysisError(
                f"{label}.speculation_execution_events[{execution_index}]: "
                "missing prediction/candidate identity"
            )
        candidate_key = (prediction_id, str(digest))
        if candidate_key in execution_candidate_keys:
            raise AnalysisError(
                f"{label}.speculation_execution_events[{execution_index}]: "
                "duplicate prediction/candidate identity"
            )
        execution_candidate_keys.add(candidate_key)
        if candidate_key not in broker_accepted_candidate_keys:
            raise AnalysisError(
                f"{label}.speculation_execution_events[{execution_index}]: "
                "candidate was not broker-accepted"
            )
        if execution.get("physical_started_at_monotonic_s") is not None:
            physical_started_candidate_keys.add(candidate_key)
    if execution_candidate_keys != broker_accepted_candidate_keys:
        raise AnalysisError(
            f"{label}: broker-accepted candidates differ from execution ledger"
        )
    physical_starts = len(physical_started_candidate_keys)
    admitted_candidates = physical_starts
    outcomes_raw = payload.get("prediction_outcomes")
    outcomes = (
        [row for row in outcomes_raw if isinstance(row, Mapping)]
        if isinstance(outcomes_raw, list)
        else []
    )
    tools_raw = payload.get("tool_events")
    tools = tools_raw if isinstance(tools_raw, list) else []
    qwen_authority_candidates: dict[tuple[str, int], set[str]] = defaultdict(set)
    gemini_authority_keys: dict[tuple[str, int], set[str]] = defaultdict(set)
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        trace = tool.get("trace_id", tool.get("task_id"))
        request_index = tool.get("request_index")
        if not isinstance(trace, str) or type(request_index) is not int:
            continue
        identity = (trace, request_index)
        candidates = tool.get("authority_candidate_invocation_digests")
        if isinstance(candidates, list):
            qwen_authority_candidates[identity].update(
                str(value)
                for value in candidates
                if strict_audit._is_sha256(value)
            )
        pool_key = tool.get("pool_authority_key_sha256")
        if strict_audit._is_sha256(pool_key):
            gemini_authority_keys[identity].add(str(pool_key))
    if outcomes:
        emitted_exact_hits = 0
        broker_accepted_exact_hits = 0
        admitted_exact_hits = 0
        for outcome_index, outcome in enumerate(outcomes):
            prediction_id = outcome.get("prediction_id")
            if not isinstance(prediction_id, str) or not prediction_id:
                raise AnalysisError(
                    f"{label}.prediction_outcomes[{outcome_index}].prediction_id: missing"
                )
            trace = outcome.get("trace_id", outcome.get("task_id"))
            request_index = outcome.get("request_index")
            if not isinstance(trace, str) or type(request_index) is not int:
                raise AnalysisError(
                    f"{label}.prediction_outcomes[{outcome_index}]: "
                    "missing trace/request identity"
                )
            identity = (trace, request_index)
            nested = outcome.get("candidates")
            if isinstance(nested, list):
                # Qwen: derive every label from the union carried by raw
                # authoritative tool events.  Outcome labels and counts are
                # redundant checksums and are never analysis inputs.
                authority = qwen_authority_candidates.get(identity, set())
                for candidate_index, candidate in enumerate(nested):
                    if not isinstance(candidate, Mapping) or not strict_audit._is_sha256(
                        candidate.get("candidate_invocation_digest")
                    ):
                        raise AnalysisError(
                            f"{label}.prediction_outcomes[{outcome_index}]."
                            f"candidates[{candidate_index}]: invalid candidate digest"
                        )
                    hit = str(candidate["candidate_invocation_digest"]) in authority
                    candidate_key = (
                        prediction_id,
                        str(candidate["candidate_invocation_digest"]),
                    )
                    if candidate_key not in emitted_candidate_keys:
                        raise AnalysisError(
                            f"{label}.prediction_outcomes[{outcome_index}]."
                            f"candidates[{candidate_index}]: no sealed decision candidate"
                        )
                    if candidate_request_identity[candidate_key] != identity:
                        raise AnalysisError(
                            f"{label}.prediction_outcomes[{outcome_index}]."
                            f"candidates[{candidate_index}]: trace/request differs from decision"
                        )
                    emitted_exact_hits += int(hit)
                    broker_accepted_exact_hits += int(
                        hit and candidate_key in broker_accepted_candidate_keys
                    )
                    admitted_exact_hits += int(
                        hit and candidate_key in physical_started_candidate_keys
                    )
            else:
                # Gemini: the immutable candidate digest is compared with the
                # raw post-reveal pool authority key.
                digest = outcome.get("candidate_invocation_digest")
                if not strict_audit._is_sha256(digest):
                    raise AnalysisError(
                        f"{label}.prediction_outcomes[{outcome_index}]: "
                        "invalid candidate digest"
                    )
                hit = str(digest) in gemini_authority_keys.get(identity, set())
                candidate_key = (prediction_id, str(digest))
                if candidate_key not in emitted_candidate_keys:
                    raise AnalysisError(
                        f"{label}.prediction_outcomes[{outcome_index}]: "
                        "no sealed decision candidate"
                    )
                if candidate_request_identity[candidate_key] != identity:
                    raise AnalysisError(
                        f"{label}.prediction_outcomes[{outcome_index}]: "
                        "trace/request differs from decision"
                    )
                emitted_exact_hits += int(hit)
                broker_accepted_exact_hits += int(
                    hit and candidate_key in broker_accepted_candidate_keys
                )
                admitted_exact_hits += int(
                    hit and candidate_key in physical_started_candidate_keys
                )
    elif executions:
        # Both strict executors permit authority claims only for exact matches.
        emitted_exact_hits = None
        broker_accepted_exact_hits = None
        admitted_exact_hits = sum(
            row.get("claimed_by_authority") is True for row in executions
            if row.get("physical_started_at_monotonic_s") is not None
        )
    else:
        emitted_exact_hits = 0 if emitted_candidates == 0 else None
        broker_accepted_exact_hits = (
            0 if broker_accepted_candidates == 0 else None
        )
        admitted_exact_hits = 0 if admitted_candidates == 0 else None

    accounting = payload.get("worker_resource_accounting")
    accounting = accounting if isinstance(accounting, Mapping) else {}
    spec_worker = _first_number(
        ((accounting.get("speculative_resource_s"), f"{label}.worker.speculative"),)
    )
    promoted_worker = _first_number(
        ((accounting.get("promoted_demand_resource_s"), f"{label}.worker.promoted"),)
    )
    direct_worker = _first_number(
        ((accounting.get("direct_demand_resource_s"), f"{label}.worker.direct"),)
    )
    total_worker = _first_number(
        ((accounting.get("total_worker_occupancy_s"), f"{label}.worker.total"),)
    )
    if executions:
        useful_spec = sum(
            _finite(
                row.get("speculative_resource_s", 0.0),
                label=f"{label}.speculation_execution_events.speculative_resource_s",
            )
            for row in executions
            if row.get("claimed_by_authority") is True
        )
        wasted_spec = sum(
            _finite(
                row.get("speculative_resource_s", 0.0),
                label=f"{label}.speculation_execution_events.speculative_resource_s",
            )
            for row in executions
            if row.get("claimed_by_authority") is not True
        )
    elif spec_worker == 0.0:
        useful_spec = 0.0
        wasted_spec = 0.0
    else:
        useful_spec = None
        wasted_spec = _first_number(
            (
                (
                    summary.get("wasted_speculative_s"),
                    f"{label}.summary.wasted_speculative_s",
                ),
            )
        )

    duration_errors = []
    for index, row in enumerate(tools):
        if not isinstance(row, Mapping):
            continue
        predicted = _first_number(
            tuple(
                (row.get(field), f"{label}.tool_events[{index}].{field}")
                for field in ("authority_eta_hat_s", "tool_service_s_hat")
            )
        )
        actual = _first_number(
            tuple(
                (row.get(field), f"{label}.tool_events[{index}].{field}")
                for field in (
                    "execution_surface_service_s",
                    "assigned_service_s",
                )
            )
        )
        if predicted is not None and actual is not None:
            recomputed = abs(actual - predicted)
            absolute = row.get("duration_prediction_absolute_error_s")
            if absolute is not None:
                declared = _finite(
                    absolute,
                    label=(
                        f"{label}.tool_events[{index}]."
                        "duration_prediction_absolute_error_s"
                    ),
                )
                if not math.isclose(declared, recomputed, rel_tol=1e-9, abs_tol=1e-9):
                    raise AnalysisError(
                        f"{label}.tool_events[{index}]: declared duration error "
                        "differs from raw prediction/assigned service"
                    )
            duration_errors.append(recomputed)

    requests_count = len(requests_raw)
    emitted_precision = (
        float(emitted_exact_hits) / emitted_candidates
        if emitted_exact_hits is not None and emitted_candidates > 0
        else None
    )
    broker_accepted_precision = (
        float(broker_accepted_exact_hits) / broker_accepted_candidates
        if broker_accepted_exact_hits is not None
        and broker_accepted_candidates > 0
        else None
    )
    admitted_precision = (
        float(admitted_exact_hits) / admitted_candidates
        if admitted_exact_hits is not None and admitted_candidates > 0
        else None
    )
    abstention = (
        1.0 - min(len(decision_request_keys), requests_count) / requests_count
        if requests_count > 0
        else None
    )
    throughput_denominator = makespan if makespan is not None else experiment_wall
    return {
        "experiment_wall_s": experiment_wall,
        "makespan_s": makespan,
        "successful_tasks": successful,
        "failures": failures,
        "task_throughput_per_s": (
            successful / throughput_denominator
            if throughput_denominator is not None and throughput_denominator > 0.0
            else None
        ),
        "requests": requests_count,
        "request_latency_mean_s": (
            statistics.fmean(request_latencies) if request_latencies else None
        ),
        "request_latency_p95_s": (
            _percentile(request_latencies, 0.95) if request_latencies else None
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prediction_decisions_emitted": len(decisions),
        "prediction_candidates_emitted": emitted_candidates,
        "prediction_candidates_broker_accepted": broker_accepted_candidates,
        "prediction_candidates_physical_started": physical_starts,
        "prediction_candidates_admitted": admitted_candidates,
        "physical_speculative_starts": physical_starts,
        "exact_emitted_post_authority_hits": emitted_exact_hits,
        "exact_broker_accepted_post_authority_hits": broker_accepted_exact_hits,
        "exact_physical_started_post_authority_hits": admitted_exact_hits,
        "exact_admitted_post_authority_hits": admitted_exact_hits,
        # Kept as an explicit compatibility alias for older tables.
        "exact_post_authority_hits": admitted_exact_hits,
        "emitted_prediction_precision": emitted_precision,
        "broker_accepted_prediction_precision": broker_accepted_precision,
        "physical_started_prediction_precision": admitted_precision,
        "admitted_prediction_precision": admitted_precision,
        "prediction_precision": admitted_precision,
        "prediction_abstention_rate": abstention,
        "speculative_worker_s": spec_worker,
        "promoted_demand_worker_s": promoted_worker,
        "direct_demand_worker_s": direct_worker,
        "total_worker_s": total_worker,
        "useful_speculative_worker_s": useful_spec,
        "wasted_speculative_worker_s": wasted_spec,
        "duration_predictor_mae_s": (
            statistics.fmean(duration_errors) if duration_errors else None
        ),
        "_request_latency_values_s": request_latencies,
        "_duration_prediction_absolute_errors_s": duration_errors,
    }


def _distribution_or_null(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    finite = [float(value) for value in values]
    return {
        "observations": len(finite),
        "sum": sum(finite),
        "mean": statistics.fmean(finite),
        "p50": _percentile(finite, 0.50),
        "p95": _percentile(finite, 0.95),
    }


def _system_descriptive_report(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_block_cell = []
    by_cell: dict[str, Any] = {}
    for result in sorted(results, key=lambda row: (row["block_id"], row["cell"])):
        descriptive = result["descriptive"]
        by_block_cell.append(
            {
                "block_id": result["block_id"],
                "cell": result["cell"],
                **{field: descriptive.get(field) for field in DESCRIPTIVE_METRIC_FIELDS},
            }
        )
    for cell in strict_audit.CELLS:
        cell_rows = [
            row["descriptive"] for row in results if row["cell"] == cell
        ]
        distributions = {
            field: _distribution_or_null(
                [float(row[field]) for row in cell_rows if row.get(field) is not None]
            )
            for field in DESCRIPTIVE_METRIC_FIELDS
        }
        requests = sum(int(row["requests"]) for row in cell_rows)
        successes = sum(int(row["successful_tasks"]) for row in cell_rows)
        broker_accepted = sum(
            int(row["prediction_candidates_broker_accepted"])
            for row in cell_rows
        )
        physical_started = sum(
            int(row["prediction_candidates_physical_started"])
            for row in cell_rows
        )
        admitted = sum(int(row["prediction_candidates_admitted"]) for row in cell_rows)
        emitted_candidates = sum(
            int(row["prediction_candidates_emitted"]) for row in cell_rows
        )
        emitted_decisions = sum(
            int(row["prediction_decisions_emitted"]) for row in cell_rows
        )
        exact_hit_values = [
            int(row["exact_post_authority_hits"])
            for row in cell_rows
            if row.get("exact_post_authority_hits") is not None
        ]
        complete_hit_evidence = len(exact_hit_values) == len(cell_rows)
        exact_hits = sum(exact_hit_values) if complete_hit_evidence else None
        emitted_hit_values = [
            int(row["exact_emitted_post_authority_hits"])
            for row in cell_rows
            if row.get("exact_emitted_post_authority_hits") is not None
        ]
        complete_emitted_hit_evidence = len(emitted_hit_values) == len(cell_rows)
        emitted_exact_hits = (
            sum(emitted_hit_values) if complete_emitted_hit_evidence else None
        )
        broker_hit_values = [
            int(row["exact_broker_accepted_post_authority_hits"])
            for row in cell_rows
            if row.get("exact_broker_accepted_post_authority_hits") is not None
        ]
        complete_broker_hit_evidence = len(broker_hit_values) == len(cell_rows)
        broker_exact_hits = (
            sum(broker_hit_values) if complete_broker_hit_evidence else None
        )
        makespans = [
            float(row["makespan_s"])
            for row in cell_rows
            if row.get("makespan_s") is not None
        ]
        request_latencies = [
            float(value)
            for row in cell_rows
            for value in row.get("_request_latency_values_s", [])
        ]
        duration_errors = [
            float(value)
            for row in cell_rows
            for value in row.get("_duration_prediction_absolute_errors_s", [])
        ]
        pooled = {
            "blocks": len(cell_rows),
            "successful_tasks": successes,
            "failures": sum(int(row["failures"]) for row in cell_rows),
            "requests": requests,
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in cell_rows),
            "completion_tokens": sum(
                int(row["completion_tokens"]) for row in cell_rows
            ),
            "task_throughput_per_s": (
                successes / sum(makespans) if makespans and sum(makespans) > 0.0 else None
            ),
            "request_latency_mean_s": (
                statistics.fmean(request_latencies) if request_latencies else None
            ),
            "request_latency_p95_s": (
                _percentile(request_latencies, 0.95) if request_latencies else None
            ),
            "prediction_decisions_emitted": emitted_decisions,
            "prediction_candidates_emitted": emitted_candidates,
            "prediction_candidates_broker_accepted": broker_accepted,
            "prediction_candidates_physical_started": physical_started,
            "prediction_candidates_admitted": admitted,
            "physical_speculative_starts": sum(
                int(row["physical_speculative_starts"]) for row in cell_rows
            ),
            "exact_emitted_post_authority_hits": emitted_exact_hits,
            "exact_broker_accepted_post_authority_hits": broker_exact_hits,
            "exact_physical_started_post_authority_hits": exact_hits,
            "exact_admitted_post_authority_hits": exact_hits,
            "exact_post_authority_hits": exact_hits,
            "emitted_prediction_precision": (
                emitted_exact_hits / emitted_candidates
                if emitted_exact_hits is not None and emitted_candidates > 0
                else None
            ),
            "broker_accepted_prediction_precision": (
                broker_exact_hits / broker_accepted
                if broker_exact_hits is not None and broker_accepted > 0
                else None
            ),
            "physical_started_prediction_precision": (
                exact_hits / physical_started
                if exact_hits is not None and physical_started > 0
                else None
            ),
            "admitted_prediction_precision": (
                exact_hits / admitted
                if exact_hits is not None and admitted > 0
                else None
            ),
            "prediction_precision": (
                exact_hits / admitted
                if exact_hits is not None and admitted > 0
                else None
            ),
            "prediction_abstention_rate": (
                1.0 - min(emitted_decisions, requests) / requests
                if requests > 0
                else None
            ),
            "speculative_worker_s": (
                sum(float(row["speculative_worker_s"]) for row in cell_rows)
                if all(row.get("speculative_worker_s") is not None for row in cell_rows)
                else None
            ),
            "promoted_demand_worker_s": (
                sum(float(row["promoted_demand_worker_s"]) for row in cell_rows)
                if all(row.get("promoted_demand_worker_s") is not None for row in cell_rows)
                else None
            ),
            "direct_demand_worker_s": (
                sum(float(row["direct_demand_worker_s"]) for row in cell_rows)
                if all(row.get("direct_demand_worker_s") is not None for row in cell_rows)
                else None
            ),
            "total_worker_s": (
                sum(float(row["total_worker_s"]) for row in cell_rows)
                if all(row.get("total_worker_s") is not None for row in cell_rows)
                else None
            ),
            "useful_speculative_worker_s": (
                sum(float(row["useful_speculative_worker_s"]) for row in cell_rows)
                if all(
                    row.get("useful_speculative_worker_s") is not None
                    for row in cell_rows
                )
                else None
            ),
            "wasted_speculative_worker_s": (
                sum(float(row["wasted_speculative_worker_s"]) for row in cell_rows)
                if all(
                    row.get("wasted_speculative_worker_s") is not None
                    for row in cell_rows
                )
                else None
            ),
            "duration_predictor_mae_s": (
                statistics.fmean(duration_errors) if duration_errors else None
            ),
            "duration_predictor_evaluated_tools": len(duration_errors),
        }
        by_cell[cell] = {
            "pooled": pooled,
            "across_block_distribution": distributions,
        }
    return {
        "definitions": {
            "emitted_prediction_precision": (
                "all emitted candidates that exactly match the later authority / "
                "all emitted candidates"
            ),
            "broker_accepted_prediction_precision": (
                "broker-accepted candidates that exactly match the later authority / "
                "broker-accepted candidates; queue-conditioned diagnostic only"
            ),
            "physical_started_prediction_precision": (
                "candidates with a non-null raw physical-start timestamp that exactly "
                "match the later authority / candidates with a non-null raw physical-start timestamp"
            ),
            "admitted_prediction_precision": (
                "compatibility alias of physical_started_prediction_precision; "
                "decision-level admitted means broker acceptance and is not this denominator"
            ),
            "prediction_precision": (
                "compatibility alias of admitted_prediction_precision"
            ),
            "prediction_abstention_rate": (
                "one minus requests with an emitted prediction decision / requests"
            ),
            "useful_speculative_worker_s": (
                "pre-authority speculative occupancy of jobs later claimed by exact match"
            ),
            "wasted_speculative_worker_s": (
                "speculative occupancy of jobs never claimed by exact match"
            ),
            "duration_predictor_mae_s": (
                "mean absolute causal ETA error over tool events carrying both prediction and execution service"
            ),
            "null": "the source result did not contain enough raw evidence; no value was imputed",
        },
        "by_block_cell": by_block_cell,
        "by_cell": by_cell,
    }


def _normalize_result(
    payload: Any,
    *,
    block_id: str,
    expected_cell: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AnalysisError(f"{label}: result root must be an object")
    result_errors = strict_audit.audit_result_payload(payload)
    if result_errors:
        preview = "; ".join(result_errors[:5])
        raise AnalysisError(f"{label}: strict result audit failed: {preview}")
    paper = payload["paper_protocol"]
    if paper.get("cell") != expected_cell:
        raise AnalysisError(f"{label}: bound cell/result cell mismatch")

    tasks_raw = payload.get("task_results")
    if not isinstance(tasks_raw, list):
        tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise AnalysisError(f"{label}.tasks: expected non-empty list")
    tasks: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tasks_raw):
        task_label = f"{label}.tasks[{index}]"
        if not isinstance(raw, Mapping):
            raise AnalysisError(f"{task_label}: expected object")
        if not _task_success(raw):
            raise AnalysisError(f"{task_label}: unsuccessful task retained; paper gate fails")
        instance = _instance_id(raw, label=task_label)
        if instance in tasks:
            raise AnalysisError(f"{task_label}: duplicate task instance {instance!r}")
        tasks[instance] = {
            "root_id": _root_id(raw, label=task_label),
            "e2e_s": _task_e2e(raw, label=task_label),
            "release_offset_s": _finite(
                raw.get("release_offset_s"), label=f"{task_label}.release_offset_s"
            ),
        }

    requests_raw = payload.get("llm_events")
    if not isinstance(requests_raw, list) or not requests_raw:
        raise AnalysisError(f"{label}.llm_events: expected non-empty list")
    requests: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for index, raw in enumerate(requests_raw):
        request_label = f"{label}.llm_events[{index}]"
        if not isinstance(raw, Mapping):
            raise AnalysisError(f"{request_label}: expected object")
        instance = _request_task_id(raw, label=request_label)
        if instance not in tasks:
            raise AnalysisError(f"{request_label}: request refers to unknown task")
        request_index = _request_index(raw, label=request_label)
        if request_index in requests[instance]:
            raise AnalysisError(f"{request_label}: duplicate request index")
        requests[instance][request_index] = _request_work_signature(
            raw, label=request_label
        )
    missing_requests = sorted(set(tasks) - set(requests))
    if missing_requests:
        raise AnalysisError(f"{label}: task(s) have no live requests: {missing_requests[:5]}")

    tool_events_raw = payload.get("tool_events", [])
    if not isinstance(tool_events_raw, list):
        raise AnalysisError(f"{label}.tool_events: expected list")
    authority_tools: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    physical_services: dict[str, float] = {}

    def bind_physical_service(key: Any, value: Any, *, service_label: str) -> None:
        if not strict_audit._is_sha256(key):
            raise AnalysisError(f"{service_label}: invalid physical invocation key")
        service = _finite(value, label=f"{service_label}.assigned_service_s")
        if service < 0.0:
            raise AnalysisError(f"{service_label}.assigned_service_s: negative")
        key_string = str(key)
        previous = physical_services.get(key_string)
        if previous is not None and previous != service:
            raise AnalysisError(
                f"{service_label}: same physical invocation key has non-identical service"
            )
        physical_services[key_string] = service

    for index, raw in enumerate(tool_events_raw):
        tool_label = f"{label}.tool_events[{index}]"
        if not isinstance(raw, Mapping):
            raise AnalysisError(f"{tool_label}: expected object")
        instance = _request_task_id(raw, label=tool_label)
        if instance not in tasks:
            raise AnalysisError(f"{tool_label}: tool event refers to unknown task")
        request_index = _request_index(raw, label=tool_label)
        reveal_seq = raw.get("authoritative_revealed_seq")
        if type(reveal_seq) is not int or reveal_seq < 0:
            raise AnalysisError(
                f"{tool_label}.authoritative_revealed_seq: non-negative integer required"
            )
        authority_tools[instance].append(
            (
                request_index,
                reveal_seq,
                _authority_tool_work_signature(raw, label=tool_label),
            )
        )
        physical_key = raw.get("physical_service_key_sha256")
        if physical_key is None:
            physical_key = raw.get("authority_invocation_digest")
        assigned_service = raw.get("execution_surface_service_s")
        if assigned_service is None:
            assigned_service = raw.get("assigned_service_s")
        bind_physical_service(physical_key, assigned_service, service_label=tool_label)

    speculation_raw = payload.get("speculation_execution_events", [])
    if not isinstance(speculation_raw, list):
        raise AnalysisError(f"{label}.speculation_execution_events: expected list")
    for index, raw in enumerate(speculation_raw):
        if not isinstance(raw, Mapping):
            raise AnalysisError(
                f"{label}.speculation_execution_events[{index}]: expected object"
            )
        bind_physical_service(
            raw.get("candidate_invocation_digest"),
            raw.get("assigned_service_s"),
            service_label=f"{label}.speculation_execution_events[{index}]",
        )

    task_work: dict[str, dict[str, Any]] = {}
    for instance, task in tasks.items():
        ordered = [
            {"request_index": index, **work}
            for index, work in sorted(requests[instance].items())
        ]
        ordered_tools = []
        ordinal_by_request: dict[int, int] = defaultdict(int)
        for request_index, _reveal_seq, tool_work in sorted(
            authority_tools.get(instance, []), key=lambda row: (row[0], row[1])
        ):
            ordinal = ordinal_by_request[request_index]
            ordinal_by_request[request_index] += 1
            ordered_tools.append(
                {
                    "request_index": request_index,
                    "tool_ordinal": ordinal,
                    **tool_work,
                }
            )
        task_work[instance] = {
            "root_id": task["root_id"],
            "release_offset_s": task["release_offset_s"],
            "request_count": len(ordered),
            "model_completion_tokens": sum(
                row["model_completion_tokens"] for row in ordered
            ),
            "requests": ordered,
            "authoritative_tools": ordered_tools,
        }

    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise AnalysisError(f"{label}.model: missing")
    descriptive = _descriptive_metrics(
        payload,
        tasks_raw=tasks_raw,
        requests_raw=requests_raw,
        label=label,
    )
    return {
        "block_id": block_id,
        "cell": expected_cell,
        "model": model,
        "tasks": tasks,
        "task_work": task_work,
        "task_multiset_by_root": Counter(task["root_id"] for task in tasks.values()),
        "physical_services": physical_services,
        "descriptive": descriptive,
    }


def _load_bound_results(manifest: Mapping[str, Any], *, base: Path) -> list[dict[str, Any]]:
    evidence = manifest.get("cell_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AnalysisError("$.cell_evidence: completed four-cell bindings are required")
    results: list[dict[str, Any]] = []
    for index, raw in enumerate(evidence):
        label = f"$.cell_evidence[{index}]"
        if not isinstance(raw, Mapping):
            raise AnalysisError(f"{label}: expected object")
        block_id = raw.get("block_id")
        cell = raw.get("cell")
        path_raw = raw.get("result_path")
        expected_sha = raw.get("result_sha256")
        if not isinstance(block_id, str) or not block_id:
            raise AnalysisError(f"{label}.block_id: invalid")
        if cell not in strict_audit.CELLS:
            raise AnalysisError(f"{label}.cell: invalid")
        if not isinstance(path_raw, str) or not path_raw:
            raise AnalysisError(f"{label}.result_path: invalid")
        path = Path(path_raw)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise AnalysisError(f"{label}: result file does not exist: {path}")
        actual_sha = strict_audit.file_sha256(path)
        if expected_sha != actual_sha:
            raise AnalysisError(f"{label}: result SHA-256 mismatch")
        normalized = _normalize_result(
            _load_json(path),
            block_id=block_id,
            expected_cell=str(cell),
            label=label,
        )
        normalized["result_path"] = str(path)
        normalized["result_sha256"] = actual_sha
        results.append(normalized)
    return results


def _bound_role_json(
    manifest: Mapping[str, Any], *, role: str, base: Path
) -> tuple[dict[str, Any], Path]:
    frozen = manifest.get("frozen_files")
    if not isinstance(frozen, list):
        raise AnalysisError("manifest lacks frozen_files")
    rows = [row for row in frozen if isinstance(row, Mapping) and row.get("role") == role]
    if len(rows) != 1:
        raise AnalysisError(f"manifest must bind exactly one {role!r} file")
    binding = rows[0]
    path_raw = binding.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        raise AnalysisError(f"frozen {role!r} path is invalid")
    path = Path(path_raw)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file() or strict_audit.file_sha256(path) != binding.get("sha256"):
        raise AnalysisError(f"frozen {role!r} file binding changed")
    document = _load_json(path)
    if not isinstance(document, dict):
        raise AnalysisError(f"frozen {role!r} must be a JSON object")
    return document, path


def _checked_relative_json(
    binding: Any, *, anchor: Path, label: str
) -> tuple[dict[str, Any], Path]:
    if not isinstance(binding, Mapping):
        raise AnalysisError(f"{label}: file binding missing")
    path_raw = binding.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        raise AnalysisError(f"{label}: path missing")
    path = Path(path_raw)
    if not path.is_absolute():
        path = anchor / path
    path = path.resolve()
    if not path.is_file() or strict_audit.file_sha256(path) != binding.get("sha256"):
        raise AnalysisError(f"{label}: SHA-256 binding mismatch")
    document = _load_json(path)
    if not isinstance(document, dict):
        raise AnalysisError(f"{label}: expected JSON object")
    return document, path


def _signed_identity(document: Mapping[str, Any], field: str, *, ascii_json: bool) -> str:
    declared = document.get(field)
    if not strict_audit._is_sha256(declared):
        raise AnalysisError(f"signed workload document lacks {field}")
    unsigned = dict(document)
    unsigned.pop(field, None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=ascii_json,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != declared:
        raise AnalysisError(f"signed workload document {field} is invalid")
    return str(declared)


def _normalize_registered_tasks(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        label = f"registered workload task[{index}]"
        task_id = row.get("task_id")
        root_id = row.get("root_id")
        release = row.get("release_offset_s")
        request_count = row.get("request_count")
        if not isinstance(task_id, str) or not task_id or task_id in contract:
            raise AnalysisError(f"{label}: task identity is missing or duplicated")
        if not isinstance(root_id, str) or not root_id:
            raise AnalysisError(f"{label}: root identity is missing")
        release_s = _finite(release, label=f"{label}.release_offset_s")
        if release_s < 0.0:
            raise AnalysisError(f"{label}.release_offset_s: expected non-negative")
        if type(request_count) is not int or request_count <= 0:
            raise AnalysisError(f"{label}.request_count: positive integer required")
        contract[task_id] = {
            "root_id": root_id,
            "release_offset_s": release_s,
            "request_count": request_count,
        }
    if not contract:
        raise AnalysisError("registered workload contract is empty")
    return contract


def _registered_workload_contract(
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    base: Path,
) -> dict[str, dict[str, Any]]:
    """Recover the pre-run instance/request-count contract from frozen inputs."""

    policy, policy_path = _bound_role_json(manifest, role="policy_bundle", base=base)
    schema = policy.get("schema")
    if schema == "paste.paper.registered_workload_contract.v1":
        raw_rows = policy.get("tasks")
        if not isinstance(raw_rows, list):
            raise AnalysisError("generic policy bundle lacks registered tasks")
        return _normalize_registered_tasks(raw_rows)

    # Qwen bundle: public metadata registers instances/root/release, while the
    # separately bound sealed plan supplies only the request count for post-run
    # auditing.  No authority value is copied into the analysis manifest.
    plans = policy.get("plans")
    if isinstance(plans, Mapping):
        role_plan = plans.get("final")
        if not isinstance(role_plan, Mapping):
            raise AnalysisError("Qwen policy bundle lacks final plan bindings")
        public, _ = _checked_relative_json(
            role_plan.get("public"), anchor=policy_path.parent, label="Qwen public plan"
        )
        sealed, _ = _checked_relative_json(
            role_plan.get("sealed"), anchor=policy_path.parent, label="Qwen sealed plan"
        )
        if _signed_identity(public, "plan_sha256", ascii_json=False) != role_plan.get(
            "public_plan_sha256"
        ):
            raise AnalysisError("Qwen public plan logical identity differs from bundle")
        if _signed_identity(sealed, "sealed_sha256", ascii_json=False) != role_plan.get(
            "sealed_plan_sha256"
        ):
            raise AnalysisError("Qwen sealed plan logical identity differs from bundle")
        traces = public.get("traces")
        steps_by_task = sealed.get("trace_steps")
        if not isinstance(traces, list) or not isinstance(steps_by_task, Mapping):
            raise AnalysisError("Qwen workload plans lack traces/trace_steps")
        rows: list[dict[str, Any]] = []
        for trace in traces:
            if not isinstance(trace, Mapping):
                raise AnalysisError("Qwen public trace is malformed")
            task_id = trace.get("trace_id", trace.get("session_id"))
            steps = steps_by_task.get(task_id)
            if not isinstance(steps, list) or not steps:
                raise AnalysisError(f"Qwen sealed plan lacks requests for {task_id!r}")
            rows.append(
                {
                    "task_id": task_id,
                    "root_id": trace.get("source_session_id"),
                    "release_offset_s": trace.get("release_offset_s"),
                    "request_count": len(steps),
                }
            )
        return _normalize_registered_tasks(rows)

    # Gemini policy bundle registers the instance mapping.  Its sealed trace is
    # addressed by a signed logical hash in that bundle; a result path is merely
    # a locator and cannot substitute a different document with the same claim.
    sessions = policy.get("sessions")
    templates = policy.get("templates")
    if isinstance(sessions, list) and isinstance(templates, list):
        if not results:
            raise AnalysisError("Gemini workload contract needs a bound result")
        first_result = _load_json(Path(str(results[0]["result_path"])))
        provenance = first_result.get("provenance") if isinstance(first_result, Mapping) else None
        sealed_path_raw = (
            provenance.get("sealed_trace_path") if isinstance(provenance, Mapping) else None
        )
        if not isinstance(sealed_path_raw, str) or not sealed_path_raw:
            raise AnalysisError("Gemini result lacks sealed_trace_path locator")
        sealed_path = Path(sealed_path_raw)
        if not sealed_path.is_absolute():
            sealed_path = Path(str(results[0]["result_path"])).parent / sealed_path
        sealed_path = sealed_path.resolve()
        if not sealed_path.is_file():
            raise AnalysisError("Gemini sealed trace locator does not exist")
        sealed = _load_json(sealed_path)
        if not isinstance(sealed, Mapping):
            raise AnalysisError("Gemini sealed trace is not an object")
        sealed_identity = _signed_identity(
            sealed, "sealed_sha256", ascii_json=True
        )
        if sealed_identity != policy.get("sealed_trace_sha256"):
            raise AnalysisError("Gemini sealed trace differs from frozen policy bundle")
        sealed_templates = sealed.get("templates")
        if not isinstance(sealed_templates, list):
            raise AnalysisError("Gemini sealed trace lacks templates")
        request_counts = {
            int(row["template_index"]): len(row.get("requests", []))
            for row in sealed_templates
            if isinstance(row, Mapping) and type(row.get("template_index")) is int
        }
        root_by_template = {
            int(row["template_index"]): str(row["session_id"])
            for row in templates
            if isinstance(row, Mapping)
            and type(row.get("template_index")) is int
            and isinstance(row.get("session_id"), str)
        }
        rows = []
        for session in sessions:
            if not isinstance(session, Mapping):
                raise AnalysisError("Gemini registered session is malformed")
            template_index = session.get("template_index")
            if type(template_index) is not int or template_index not in root_by_template:
                raise AnalysisError("Gemini session has an unknown template")
            if template_index not in request_counts or request_counts[template_index] <= 0:
                raise AnalysisError("Gemini sealed template has no requests")
            rows.append(
                {
                    "task_id": session.get("task_id"),
                    "root_id": root_by_template[template_index],
                    "release_offset_s": session.get("release_offset_s"),
                    "request_count": request_counts[template_index],
                }
            )
        return _normalize_registered_tasks(rows)

    raise AnalysisError(f"unsupported frozen policy-bundle workload schema: {schema!r}")


def _check_registered_workload(
    results: Sequence[Mapping[str, Any]],
    registered: Mapping[str, Mapping[str, Any]],
) -> str:
    for result in results:
        label = f"block={result['block_id']} cell={result['cell']}"
        tasks = result["tasks"]
        if set(tasks) != set(registered):
            missing = sorted(set(registered) - set(tasks))
            extra = sorted(set(tasks) - set(registered))
            raise AnalysisError(
                f"{label}: tasks differ from frozen workload contract; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        for task_id, expected in registered.items():
            observed = tasks[task_id]
            work = result["task_work"][task_id]
            if observed["root_id"] != expected["root_id"] or not math.isclose(
                float(observed["release_offset_s"]),
                float(expected["release_offset_s"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise AnalysisError(
                    f"{label}: task {task_id!r} root/release differs from frozen workload contract"
                )
            if int(work["request_count"]) != int(expected["request_count"]):
                raise AnalysisError(
                    f"{label}: task {task_id!r} request count differs from frozen workload contract"
                )
            request_indices = [int(row["request_index"]) for row in work["requests"]]
            if request_indices != list(range(int(expected["request_count"]))):
                raise AnalysisError(
                    f"{label}: task {task_id!r} request indices are not the frozen contiguous workload"
                )
    return _sha256(
        [
            {"task_id": task_id, **dict(registered[task_id])}
            for task_id in sorted(registered)
        ]
    )


def _check_work_equivalence(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise AnalysisError("no cell results")
    expected_model = results[0]["model"]
    expected_multiset = results[0]["task_multiset_by_root"]
    expected_instances = set(results[0]["tasks"])
    expected_work = results[0]["task_work"]
    physical_services: dict[str, float] = {}
    for result in results[1:]:
        label = f"block={result['block_id']} cell={result['cell']}"
        if result["model"] != expected_model:
            raise AnalysisError(f"{label}: model differs across cells")
        if result["task_multiset_by_root"] != expected_multiset:
            raise AnalysisError(f"{label}: source-root multiset differs across cells")
        if set(result["tasks"]) != expected_instances:
            raise AnalysisError(f"{label}: task/replica identities differ across cells")
        if result["task_work"] != expected_work:
            raise AnalysisError(
                f"{label}: release schedule, request identity, prompt/public max-token "
                "work, model completion tokens, or authoritative tool/result/service "
                "sequence differ"
            )
    for result in results:
        for key, service in result.get("physical_services", {}).items():
            previous = physical_services.get(str(key))
            if previous is not None and previous != float(service):
                raise AnalysisError(
                    "same normalized physical invocation key received different "
                    "assigned service across A/B/E/F or blocks"
                )
            physical_services[str(key)] = float(service)
    return {
        "passed": True,
        "model": expected_model,
        "source_roots": len(expected_multiset),
        "task_instances": sum(expected_multiset.values()),
        "root_multiset_sha256": _sha256(sorted(expected_multiset.items())),
        "task_request_work_sha256": _sha256(expected_work),
        "authoritative_tool_work_sha256": _sha256(
            {
                instance: row["authoritative_tools"]
                for instance, row in expected_work.items()
            }
        ),
        "physical_service_assignment_count": len(physical_services),
        "physical_service_assignments_sha256": _sha256(physical_services),
        "same_key_service_assignment_passed": True,
        "requests_per_cell": sum(
            int(row["request_count"]) for row in expected_work.values()
        ),
        "model_completion_tokens_per_cell": sum(
            int(row["model_completion_tokens"]) for row in expected_work.values()
        ),
    }


def _fold_root_block_cell(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for result in results:
        block_id = str(result["block_id"])
        cell = str(result["cell"])
        for task in result["tasks"].values():
            grouped[(str(task["root_id"]), block_id, cell)].append(float(task["e2e_s"]))

    roots = sorted({key[0] for key in grouped})
    blocks = sorted({key[1] for key in grouped})
    missing = [
        (root, block, cell)
        for root in roots
        for block in blocks
        for cell in strict_audit.CELLS
        if (root, block, cell) not in grouped
    ]
    if missing:
        raise AnalysisError(f"incomplete paired root/block/cell matrix: {missing[:5]}")

    root_block: dict[str, dict[str, dict[str, float]]] = {}
    replica_counts: Counter[int] = Counter()
    for root in roots:
        root_block[root] = {}
        for block in blocks:
            root_block[root][block] = {}
            for cell in strict_audit.CELLS:
                values = grouped[(root, block, cell)]
                replica_counts[len(values)] += 1
                root_block[root][block][cell] = statistics.fmean(values)
    return root_block, {
        "roots": len(roots),
        "blocks": len(blocks),
        "replica_count_distribution_over_root_block_cell": {
            str(key): value for key, value in sorted(replica_counts.items())
        },
    }


def _root_cell_means(
    root_block: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> dict[str, dict[str, float]]:
    return {
        root: {
            cell: statistics.fmean(
                block_values[cell] for block_values in blocks.values()
            )
            for cell in strict_audit.CELLS
        }
        for root, blocks in root_block.items()
    }


def _contrast_vectors(
    root_cells: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    vectors: dict[str, dict[str, float]] = {}
    for name, (baseline, treatment) in RELATIVE_CONTRASTS.items():
        vector: dict[str, float] = {}
        for root, cells in root_cells.items():
            denominator = cells[baseline]
            if denominator <= 0.0:
                raise AnalysisError(f"root {root!r}: non-positive {baseline} E2E")
            vector[root] = (denominator - cells[treatment]) / denominator
        vectors[name] = vector
    vectors["interaction"] = {
        root: (cells["E"] - cells["F"]) - (cells["A"] - cells["B"])
        for root, cells in root_cells.items()
    }
    return vectors


def _bootstrap_contrasts(
    root_cells: Mapping[str, Mapping[str, float]],
    *,
    resamples: int,
    seed: str,
) -> dict[str, list[float]]:
    roots = sorted(root_cells)
    if not roots:
        raise AnalysisError("bootstrap has no roots")
    seed_int = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest(), "big")
    rng = random.Random(seed_int)
    samples = {name: [] for name in (*RELATIVE_CONTRASTS, "interaction")}
    for _ in range(resamples):
        chosen = [roots[rng.randrange(len(roots))] for _ in roots]
        for name, (baseline, treatment) in RELATIVE_CONTRASTS.items():
            baseline_mean = statistics.fmean(
                root_cells[root][baseline] for root in chosen
            )
            treatment_mean = statistics.fmean(
                root_cells[root][treatment] for root in chosen
            )
            samples[name].append(
                (baseline_mean - treatment_mean) / baseline_mean
            )
        samples["interaction"].append(
            statistics.fmean(
                (root_cells[root]["E"] - root_cells[root]["F"])
                - (root_cells[root]["A"] - root_cells[root]["B"])
                for root in chosen
            )
        )
    return samples


def analyze_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = _load_json(path)
    preflight = strict_audit.audit_manifest(
        payload,
        base=path.parent,
        verify_files=True,
        require_evidence=False,
    )
    if not preflight["valid"]:
        raise AnalysisError(
            "manifest preflight failed: " + "; ".join(preflight["errors"][:8])
        )
    results = _load_bound_results(payload, base=path.parent)
    expected_pairs = {
        (str(block["block_id"]), cell)
        for block in payload["execution"]["blocks"]
        for cell in strict_audit.CELLS
    }
    actual_pairs = {(row["block_id"], row["cell"]) for row in results}
    if actual_pairs != expected_pairs or len(results) != len(expected_pairs):
        raise AnalysisError("cell evidence is not exactly one result per block and A/B/E/F")

    registered_workload = _registered_workload_contract(
        payload, results, base=path.parent
    )
    registered_workload_sha256 = _check_registered_workload(
        results, registered_workload
    )
    work = _check_work_equivalence(results)
    work["registered_workload_contract_passed"] = True
    work["registered_workload_contract_sha256"] = registered_workload_sha256
    observed_roots = set(results[0]["task_multiset_by_root"])
    registered_roots = set(payload["data"]["evaluation_root_ids"])
    if observed_roots != registered_roots:
        missing = sorted(registered_roots - observed_roots)
        extra = sorted(observed_roots - registered_roots)
        raise AnalysisError(
            "executed source roots differ from the registered evaluation set: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    root_block, folding = _fold_root_block_cell(results)
    root_cells = _root_cell_means(root_block)
    vectors = _contrast_vectors(root_cells)
    stats = payload["statistics"]
    resamples = int(stats["paired_bootstrap_resamples"])
    seed = str(stats["paired_bootstrap_seed"])
    samples = _bootstrap_contrasts(root_cells, resamples=resamples, seed=seed)

    contrasts: dict[str, Any] = {}
    for name, vector in vectors.items():
        values = list(vector.values())
        if name == "interaction":
            estimate = statistics.fmean(values)
        else:
            baseline, treatment = RELATIVE_CONTRASTS[name]
            baseline_mean = statistics.fmean(
                cells[baseline] for cells in root_cells.values()
            )
            treatment_mean = statistics.fmean(
                cells[treatment] for cells in root_cells.values()
            )
            estimate = (baseline_mean - treatment_mean) / baseline_mean
        row: dict[str, Any] = {
            "unit": "seconds" if name == "interaction" else "relative_reduction",
            "estimate": estimate,
            "paired_root_bootstrap_95_ci": [
                _percentile(samples[name], 0.025),
                _percentile(samples[name], 0.975),
            ],
            "roots": len(values),
            "fraction_roots_positive": sum(value > 0.0 for value in values) / len(values),
        }
        if name != "interaction":
            row["estimand"] = "ratio_of_paired_root_mean_e2e"
            row["ratio_of_paired_root_mean_e2e"] = estimate
            row["mean_per_root_relative_reduction_secondary"] = statistics.fmean(
                values
            )
            row["paired_bootstrap_95_ci"] = list(
                row["paired_root_bootstrap_95_ci"]
            )
        contrasts[name] = row

    primary = contrasts["A_vs_F"]
    ci_lower = float(primary["paired_root_bootstrap_95_ci"][0])
    speedup_20_pass = (
        float(primary["ratio_of_paired_root_mean_e2e"]) >= 0.20
        and ci_lower > 0.0
    )
    strong_20_claim_pass = ci_lower >= 0.20
    cell_distributions = {
        cell: {
            "mean_root_e2e_s": statistics.fmean(
                values[cell] for values in root_cells.values()
            ),
            "p50_root_e2e_s": _percentile(
                [values[cell] for values in root_cells.values()], 0.50
            ),
            "p95_root_e2e_s": _percentile(
                [values[cell] for values in root_cells.values()], 0.95
            ),
        }
        for cell in strict_audit.CELLS
    }
    block_primary = {}
    for block in sorted(next(iter(root_block.values()))):
        a_values = [root_block[root][block]["A"] for root in sorted(root_block)]
        f_values = [root_block[root][block]["F"] for root in sorted(root_block)]
        values = [
            (a_value - f_value) / a_value
            for a_value, f_value in zip(a_values, f_values, strict=True)
        ]
        block_primary[block] = {
            "A_vs_F_ratio_of_mean_root_e2e": (
                statistics.fmean(a_values) - statistics.fmean(f_values)
            )
            / statistics.fmean(a_values),
            "mean_per_root_relative_reduction_secondary": statistics.fmean(values),
            "fraction_roots_positive": sum(value > 0.0 for value in values) / len(values),
        }
    system_descriptive = _system_descriptive_report(results)

    manifest_outcomes = {
        name: {
            "estimand": "ratio_of_paired_root_mean_e2e",
            "ratio_of_paired_root_mean_e2e": row[
                "ratio_of_paired_root_mean_e2e"
            ],
            "paired_bootstrap_95_ci": row["paired_bootstrap_95_ci"],
        }
        for name, row in contrasts.items()
        if name != "interaction"
    }
    manifest_outcomes["interaction"] = {
        "mean_interaction_s": contrasts["interaction"]["estimate"],
        "paired_bootstrap_95_ci_s": contrasts["interaction"][
            "paired_root_bootstrap_95_ci"
        ],
    }

    unsigned = {
        "schema": SCHEMA,
        "manifest": str(path),
        "manifest_sha256": strict_audit.file_sha256(path),
        "claim_scope": payload["claim_scope"],
        "confirmatory_eligible": preflight["confirmatory_eligible"],
        "call_graph_mode": payload["policy"]["call_graph_mode"],
        "claim_type": payload["policy"]["claim_type"],
        "folding": folding,
        "work_equivalence": work,
        "cell_root_e2e": cell_distributions,
        "block_primary_effects": block_primary,
        "mechanism_and_system_descriptive": system_descriptive,
        "contrasts": contrasts,
        "manifest_outcomes": manifest_outcomes,
        "primary_rule": {
            "contrast": "A_vs_F",
            "threshold": 0.20,
            "rule": "point_estimate_ge_0.20_and_ci_lower_gt_0",
            "speedup_20_pass": speedup_20_pass,
            "strong_20_claim_pass": strong_20_claim_pass,
        },
        "bootstrap": {
            "unit": "paired_source_root",
            "resamples": resamples,
            "seed": seed,
            "ci_level": 0.95,
        },
        "result_bindings": [
            {
                "block_id": row["block_id"],
                "cell": row["cell"],
                "path": row["result_path"],
                "sha256": row["result_sha256"],
            }
            for row in sorted(results, key=lambda item: (item["block_id"], item["cell"]))
        ],
    }
    runtime_provenance = strict_audit.expected_runtime_provenance(payload)
    if all(
        strict_audit._is_sha256(runtime_provenance.get(field))
        for field in strict_audit.GEMINI_LEGACY_COMPATIBILITY_PROVENANCE_FIELDS
    ):
        unsigned["legacy_frozen_compatibility"] = {
            "schema": strict_audit.GEMINI_LEGACY_COMPATIBILITY_SCHEMA,
            "compatibility_mode": (
                strict_audit.GEMINI_LEGACY_COMPATIBILITY_MODE
            ),
            "certificate_file_sha256": runtime_provenance[
                "legacy_compatibility_certificate_file_sha256"
            ],
            "compatibility_sha256": runtime_provenance[
                "legacy_compatibility_certificate_sha256"
            ],
            "independent_verifier_sha256": runtime_provenance[
                "legacy_compatibility_verifier_file_sha256"
            ],
            "independent_verifier_rerun_passed": True,
            "per_decision_service_eta_recomputed": True,
        }
    return {**unsigned, "analysis_sha256": _sha256(unsigned)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("manifest", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        result = analyze_manifest(args.manifest)
    except (AnalysisError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 2
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
