#!/usr/bin/env python3
"""Strictly audit and compare development A/N/V live-tool runs.

The three treatments are:

* A: native FCFS LLM scheduling, demand-only tools;
* N: joint physical-KV LLM scheduling, demand-only tools;
* V: the same joint scheduler plus execution-aware visit speculation.

Unlike a two-file convenience comparator, this module accepts repeated runs of
each cell.  It folds replicas and repeated executions *within source* before
bootstrapping the independent sources.  Every individual run and every run
pair retains explicit eligibility gates.  Consequently, an older run with
missing wire-attempt evidence can remain visible as an order-sensitivity
diagnostic without silently entering a strict causal claim.

The runner summary is never trusted as the performance observation.  Core
task, LLM, tool, and queue evidence is revalidated by
``compare_live_joint_pair._validate_run``.  This module adds the development
contract that was introduced after the original pair validator: exact
48/144/96 success counts, a wire-level log for every started HTTP attempt,
physical attempt-start spacing, canary non-speculation, a narrow treatment
config allowlist, token-balance gates, and E2E = LLM + exposed-tool-wait +
orchestration-residual accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

from compare_live_joint_pair import (  # type: ignore
    BOOTSTRAP_SEED,
    ValidatedRun,
    _distribution,
    _mapping,
    _percentile,
    _validate_run,
)


SCHEMA = "paste_repro.live_joint_dev_triplet"
SCHEMA_VERSION = 1
BOOTSTRAP_RESAMPLES = 10_000

EXPECTED_TASK_COUNT = 48
EXPECTED_LLM_REQUEST_COUNT = 144
EXPECTED_AUTHORITATIVE_COMMIT_COUNT = 96
EXPECTED_SOURCE_COUNT = 16
EXPECTED_REPLICAS = 3

HTTP_ATTEMPT_GATE_POLICY_VERSION = "shared-per-tool-monotonic-v1"
MAX_TOKEN_RELATIVE_DIFFERENCE = 0.01
MIN_E2E_REDUCTION = 0.05
MIN_FASTER_SOURCE_FRACTION = 0.60
MAX_TASK_P95_RATIO = 1.05
MAX_MAKESPAN_RATIO = 1.03
MIN_JOINT_PRESSURE_FRACTION = 0.10
ATTEMPT_SPACING_TOLERANCE_S = 0.01

CELL_IDS = ("A", "N", "V")
CELL_TREATMENTS = {
    "A": {"speculation_mode": "off", "scheduler_policy": "fcfs"},
    "N": {
        "speculation_mode": "off",
        "scheduler_policy": "online_joint_pacer_v2",
    },
    "V": {
        "speculation_mode": "visit",
        "scheduler_policy": "online_joint_pacer_v2",
    },
}
EFFECTS = {
    "A_to_N": ("A", "N"),
    "N_to_V": ("N", "V"),
    "A_to_V": ("A", "V"),
}

# These are the only top-level fields permitted to vary across a treatment
# comparison.  In particular, code/module digests are deliberately not here.
_COMMON_CONFIG_EXCLUSIONS = frozenset(
    {
        "cell_label",
        "speculation_mode",
        "scheduler_environment",
        # This is derived live-search evidence, not a design input.
        "expected_url_search_coverage",
    }
)


@dataclass(frozen=True)
class RunAudit:
    cell: str
    ordinal: int
    run: ValidatedRun
    exact_counts: Mapping[str, Any]
    http_attempts: Mapping[str, Any]
    canary: Mapping[str, Any]

    @property
    def label(self) -> str:
        return f"{self.cell}{self.ordinal}"

    @property
    def strict_evidence_eligible(self) -> bool:
        return bool(
            self.exact_counts["passed"]
            and self.http_attempts["passed"]
            and self.canary["passed"]
        )


def _finite(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _integer(value: Any, label: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _bounded_errors(errors: Sequence[str], limit: int = 20) -> dict[str, Any]:
    return {
        "count": len(errors),
        "first": list(errors[:limit]),
        "truncated": len(errors) > limit,
    }


def _gate(observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _relative_delta(baseline: float, candidate: float) -> float:
    return _ratio(candidate - baseline, baseline)


def _latency_delta(baseline: float, candidate: float) -> dict[str, float]:
    reduction = baseline - candidate
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_reduction_s": reduction,
        "relative_reduction": _ratio(reduction, baseline),
    }


def _validate_exact_counts(
    run: ValidatedRun,
    *,
    expected_task_count: int = EXPECTED_TASK_COUNT,
    expected_llm_request_count: int = EXPECTED_LLM_REQUEST_COUNT,
    expected_authoritative_commit_count: int = (
        EXPECTED_AUTHORITATIVE_COMMIT_COUNT
    ),
    expected_source_count: int = EXPECTED_SOURCE_COUNT,
    expected_replicas: int = EXPECTED_REPLICAS,
) -> dict[str, Any]:
    task_count = len(run.tasks_by_key)
    llm_count = sum(len(rows) for rows in run.llm_by_task.values())
    commit_count = len(run.committed_by_task_tool)
    source_count = len({source_id for source_id, _ in run.tasks_by_key})
    replicas = int(run.config.get("replicas", -1))
    summary = _mapping(run.payload.get("summary"), "summary")
    llm_summary = _mapping(summary.get("llm"), "summary.llm")
    tool_summary = _mapping(summary.get("tool"), "summary.tool")
    observed = {
        "task_count": task_count,
        "successful_task_count": summary.get("successful_task_count"),
        "failed_task_count": summary.get("failed_task_count"),
        "llm_request_count": llm_count,
        "successful_llm_request_count": llm_summary.get(
            "successful_request_count"
        ),
        "llm_exactly_one_attempt_each": llm_summary.get(
            "exactly_one_attempt_each"
        ),
        "authoritative_commit_count": commit_count,
        "reported_authoritative_commit_count": tool_summary.get(
            "authoritative_commit_count"
        ),
        "independent_source_count": source_count,
        "replicas": replicas,
    }
    expected = {
        "task_count": expected_task_count,
        "llm_request_count": expected_llm_request_count,
        "authoritative_commit_count": expected_authoritative_commit_count,
        "independent_source_count": expected_source_count,
        "replicas": expected_replicas,
    }
    passed = (
        task_count == expected_task_count
        and summary.get("successful_task_count") == expected_task_count
        and summary.get("failed_task_count") == 0
        and llm_count == expected_llm_request_count
        and llm_summary.get("successful_request_count")
        == expected_llm_request_count
        and llm_summary.get("exactly_one_attempt_each") is True
        and commit_count == expected_authoritative_commit_count
        and tool_summary.get("authoritative_commit_count")
        == expected_authoritative_commit_count
        and source_count == expected_source_count
        and replicas == expected_replicas
    )
    return {"passed": passed, "observed": observed, "expected": expected}


def _audit_http_attempt_logs(run: ValidatedRun) -> dict[str, Any]:
    """Validate wire-attempt logs and the physical per-tool start gate.

    This is intentionally separate from the older core validator so that old
    artifacts can be reported as ineligible rather than reinterpreted as if
    the subsequently added attempt telemetry had existed.
    """

    config = run.config
    errors: list[str] = []
    if config.get("tool_http_attempt_start_gate_enabled") is not True:
        errors.append("tool_http_attempt_start_gate_enabled is not true")
    if (
        config.get("tool_http_attempt_start_gate_policy_version")
        != HTTP_ATTEMPT_GATE_POLICY_VERSION
    ):
        errors.append("tool HTTP attempt-start gate policy version differs")

    raw_intervals = config.get("tool_http_attempt_min_start_intervals_s")
    if not isinstance(raw_intervals, Mapping):
        errors.append("tool_http_attempt_min_start_intervals_s is not an object")
        intervals: dict[str, float] = {}
    else:
        intervals = {}
        for raw_tool, raw_interval in raw_intervals.items():
            tool = str(raw_tool)
            try:
                interval = _finite(
                    raw_interval,
                    f"tool_http_attempt_min_start_intervals_s.{tool}",
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if tool not in {"search", "visit"}:
                errors.append(f"attempt-start interval has unsupported tool {tool!r}")
            elif interval > 0.0:
                intervals[tool] = interval

    for tool in ("search", "visit"):
        raw = config.get(f"{tool}_min_start_interval_s", 0.0)
        try:
            expected = _finite(raw, f"{tool}_min_start_interval_s")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        recorded = intervals.get(tool, 0.0)
        if not math.isclose(recorded, expected, rel_tol=0.0, abs_tol=1e-9):
            errors.append(
                f"{tool} attempt-start interval {recorded} differs from {expected}"
            )

    starts_by_tool: dict[str, list[float]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    started_job_count = 0
    attempt_count = 0
    retried_job_count = 0
    total_gate_wait_s = 0.0
    total_retry_backoff_s = 0.0
    retryable_statuses = set(config.get("tool_http_retryable_statuses", []))
    retryable_errors = set(config.get("tool_http_retryable_exception_types", []))
    configured_backoff = float(config.get("tool_http_retry_backoff_s", 0.0))

    for record_index, record in enumerate(run.physical_records):
        label = f"tool_attempt_records[{record_index}]"
        started_at = record.get("started_at")
        raw_log = record.get("http_attempt_log")
        http_attempts = record.get("http_attempts")
        if started_at is None:
            if raw_log is not None and raw_log != []:
                errors.append(f"{label} never started but has an HTTP attempt log")
            if http_attempts != 0:
                errors.append(f"{label} never started but http_attempts is not zero")
            continue

        started_job_count += 1
        if not isinstance(raw_log, list) or not raw_log:
            errors.append(f"{label} started without a non-empty http_attempt_log")
            continue
        if not isinstance(http_attempts, int) or isinstance(http_attempts, bool):
            errors.append(f"{label}.http_attempts is not an integer")
            continue
        if len(raw_log) != http_attempts:
            errors.append(f"{label} http_attempt_log length differs from http_attempts")
        attempt_count += len(raw_log)
        retried_job_count += int(len(raw_log) > 1)
        previous_start = -math.inf
        tool = str(record.get("tool"))
        finished_at = record.get("finished_at")
        for attempt_index, raw_attempt in enumerate(raw_log):
            prefix = f"{label}.http_attempt_log[{attempt_index}]"
            if not isinstance(raw_attempt, Mapping):
                errors.append(f"{prefix} is not an object")
                continue
            attempt = raw_attempt.get("attempt")
            request_index = raw_attempt.get("request_index")
            if attempt != attempt_index + 1:
                errors.append(f"{prefix}.attempt is not contiguous from one")
            if (
                not isinstance(request_index, int)
                or isinstance(request_index, bool)
                or request_index < 0
            ):
                errors.append(f"{prefix}.request_index is not non-negative integer")
            try:
                started = _finite(
                    raw_attempt.get("started_monotonic_s"),
                    f"{prefix}.started_monotonic_s",
                )
                gate_wait = _finite(
                    raw_attempt.get("start_gate_wait_s"),
                    f"{prefix}.start_gate_wait_s",
                )
                retry_backoff = _finite(
                    raw_attempt.get("retry_backoff_s"),
                    f"{prefix}.retry_backoff_s",
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if started < previous_start:
                errors.append(f"{prefix}.started_monotonic_s is not monotonic")
            previous_start = started
            if started + 0.02 < float(started_at):
                errors.append(f"{prefix} starts before its broker job")
            if finished_at is not None and started > float(finished_at) + 0.02:
                errors.append(f"{prefix} starts after its broker job finished")
            starts_by_tool[tool].append(started)
            total_gate_wait_s += gate_wait
            total_retry_backoff_s += retry_backoff

            status = raw_attempt.get("status")
            error_type = raw_attempt.get("error_type")
            retried = raw_attempt.get("retried")
            if status is not None and (
                not isinstance(status, int) or isinstance(status, bool)
            ):
                errors.append(f"{prefix}.status is neither integer nor null")
            if error_type is not None and (
                not isinstance(error_type, str) or not error_type
            ):
                errors.append(f"{prefix}.error_type is neither string nor null")
            if not isinstance(retried, bool):
                errors.append(f"{prefix}.retried is not boolean")
                continue
            status_counts[str(status)] += 1
            is_final = attempt_index == len(raw_log) - 1
            if is_final:
                if status != 200 or error_type is not None or retried:
                    errors.append(f"{prefix} is not a final successful HTTP 200")
                if retry_backoff != 0.0:
                    errors.append(f"{prefix} final success reports retry backoff")
            else:
                if not retried:
                    errors.append(f"{prefix} is followed by another attempt but retried=false")
                if status not in retryable_statuses and error_type not in retryable_errors:
                    errors.append(f"{prefix} failure is outside controlled retry policy")
                if configured_backoff > 0.0 and retry_backoff + 0.05 < configured_backoff:
                    errors.append(f"{prefix} retry backoff is shorter than configured")

    spacing: dict[str, Any] = {}
    for tool in ("search", "visit"):
        starts = sorted(starts_by_tool.get(tool, []))
        deltas = [later - earlier for earlier, later in zip(starts, starts[1:])]
        minimum = min(deltas) if deltas else None
        required = intervals.get(tool, 0.0)
        passed = minimum is None or minimum + ATTEMPT_SPACING_TOLERANCE_S >= required
        if not passed:
            errors.append(
                f"{tool} physical HTTP starts violate {required:.6f}s minimum interval"
            )
        spacing[tool] = {
            "attempt_count": len(starts),
            "minimum_observed_start_delta_s": minimum,
            "required_minimum_start_delta_s": required,
            "tolerance_s": ATTEMPT_SPACING_TOLERANCE_S,
            "passed": passed,
        }

    expected_started_attempts = sum(
        int(record.get("http_attempts") or 0)
        for record in run.physical_records
        if record.get("started_at") is not None
    )
    if attempt_count != expected_started_attempts:
        errors.append(
            "wire attempt-log rows do not equal physical http_attempts accounting"
        )
    return {
        "passed": not errors,
        "policy_enabled": config.get("tool_http_attempt_start_gate_enabled"),
        "policy_version": config.get(
            "tool_http_attempt_start_gate_policy_version"
        ),
        "configured_intervals_s": dict(sorted(intervals.items())),
        "started_physical_job_count": started_job_count,
        "wire_http_attempt_count": attempt_count,
        "expected_wire_http_attempt_count": expected_started_attempts,
        "retried_physical_job_count": retried_job_count,
        "status_counts": dict(sorted(status_counts.items())),
        "total_start_gate_wait_s": total_gate_wait_s,
        "total_retry_backoff_s": total_retry_backoff_s,
        "spacing": spacing,
        "errors": _bounded_errors(errors),
    }


def _audit_canary_non_speculation(run: ValidatedRun) -> dict[str, Any]:
    errors: list[str] = []
    canary_task_ids = {
        task_id
        for task_id, task in run.tasks_by_id.items()
        if task.get("visit_canary") is True
    }
    stride = run.config.get("visit_canary_stride")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        errors.append("visit_canary_stride is not a positive integer")
        expected_count = None
    else:
        expected_count = math.ceil(len(run.tasks_by_id) / stride)
        if len(canary_task_ids) != expected_count:
            errors.append(
                f"visit canary count {len(canary_task_ids)} differs from {expected_count}"
            )

    speculative_records = [
        record
        for record in run.physical_records
        if record.get("speculative") is True
    ]
    canary_speculative = [
        record
        for record in speculative_records
        if record.get("session_id") in canary_task_ids
    ]
    if canary_speculative:
        errors.append("one or more canary tasks received physical speculation")

    canary_commits = [
        record
        for (task_id, tool), record in run.committed_by_task_tool.items()
        if task_id in canary_task_ids and tool == "visit"
    ]
    if len(canary_commits) != len(canary_task_ids):
        errors.append("canary visit authoritative commit coverage is incomplete")
    for record in canary_commits:
        if record.get("speculative") is True:
            errors.append("a canary visit commit is speculative")
        if record.get("speculation_eligible") is not False:
            errors.append("a canary visit commit is marked speculation-eligible")

    if run.config.get("speculation_mode") == "off" and speculative_records:
        errors.append("speculation_mode=off contains speculative physical records")
    return {
        "passed": not errors,
        "canary_task_count": len(canary_task_ids),
        "expected_canary_task_count": expected_count,
        "speculative_physical_record_count": len(speculative_records),
        "canary_speculative_physical_record_count": len(canary_speculative),
        "canary_authoritative_visit_commit_count": len(canary_commits),
        "errors": _bounded_errors(errors),
    }


def _scheduler_environment(run: ValidatedRun) -> Mapping[str, Any]:
    return _mapping(run.config.get("scheduler_environment"), "scheduler_environment")


def _treatment_audit(run: ValidatedRun, cell: str) -> dict[str, Any]:
    treatment = CELL_TREATMENTS[cell]
    environment = _scheduler_environment(run)
    observed = {
        "speculation_mode": run.config.get("speculation_mode"),
        "scheduler_policy": environment.get("VLLM_SCHED_POLICY"),
    }
    return {
        "observed": observed,
        "expected": treatment,
        "passed": observed == treatment,
    }


def _config_pair_audit(
    left: ValidatedRun,
    right: ValidatedRun,
    *,
    left_cell: str,
    right_cell: str,
) -> dict[str, Any]:
    left_common = {
        key: value
        for key, value in left.config.items()
        if key not in _COMMON_CONFIG_EXCLUSIONS
    }
    right_common = {
        key: value
        for key, value in right.config.items()
        if key not in _COMMON_CONFIG_EXCLUSIONS
    }
    common_differences = sorted(
        key
        for key in set(left_common) | set(right_common)
        if left_common.get(key) != right_common.get(key)
    )
    left_env = _scheduler_environment(left)
    right_env = _scheduler_environment(right)
    left_runtime = {
        key: value for key, value in left_env.items() if not key.startswith("VLLM_SCHED_")
    }
    right_runtime = {
        key: value for key, value in right_env.items() if not key.startswith("VLLM_SCHED_")
    }
    runtime_differences = sorted(
        key
        for key in set(left_runtime) | set(right_runtime)
        if left_runtime.get(key) != right_runtime.get(key)
    )
    same_scheduler_treatment = (
        CELL_TREATMENTS[left_cell]["scheduler_policy"]
        == CELL_TREATMENTS[right_cell]["scheduler_policy"]
    )
    scheduler_environment_exact = left_env == right_env
    scheduler_passed = not runtime_differences and (
        scheduler_environment_exact or not same_scheduler_treatment
    )
    left_treatment = _treatment_audit(left, left_cell)
    right_treatment = _treatment_audit(right, right_cell)
    passed = (
        not common_differences
        and scheduler_passed
        and left_treatment["passed"]
        and right_treatment["passed"]
    )
    return {
        "passed": passed,
        "allowed_top_level_treatment_fields": sorted(_COMMON_CONFIG_EXCLUSIONS),
        "uncontrolled_top_level_differences": common_differences,
        "non_scheduler_runtime_environment_differences": runtime_differences,
        "same_scheduler_treatment": same_scheduler_treatment,
        "scheduler_environment_exact_match": scheduler_environment_exact,
        "left_treatment": left_treatment,
        "right_treatment": right_treatment,
    }


def _identity_pair_audit(left: ValidatedRun, right: ValidatedRun) -> dict[str, Any]:
    errors: list[str] = []
    if left.call_graph_mode != right.call_graph_mode:
        errors.append("call_graph_mode differs")
    left_keys = set(left.tasks_by_key)
    right_keys = set(right.tasks_by_key)
    if left_keys != right_keys:
        errors.append("source+replica task identity sets differ")
    keys = sorted(left_keys & right_keys)
    search_invocation_matches = 0
    visit_invocation_matches = 0
    search_result_matches = 0
    visit_result_matches = 0
    selected_url_matches = 0
    search_url_list_matches = 0
    frozen_identity_matches = 0
    for key in keys:
        left_task = left.tasks_by_key[key]
        right_task = right.tasks_by_key[key]
        task_id = str(left_task["task_id"])
        if right_task.get("task_id") != task_id:
            errors.append(f"{key} task_id differs")
        for field in ("question_sha256", "search_query"):
            if left_task.get(field) != right_task.get(field):
                errors.append(f"{task_id} {field} differs")
        left_search = left.committed_by_task_tool[(task_id, "search")]
        right_search = right.committed_by_task_tool[(task_id, "search")]
        left_visit = left.committed_by_task_tool[(task_id, "visit")]
        right_visit = right.committed_by_task_tool[(task_id, "visit")]
        search_invocation_matches += int(
            left_search.get("invocation_digest")
            == right_search.get("invocation_digest")
        )
        visit_invocation_match = (
            left_visit.get("invocation_digest")
            == right_visit.get("invocation_digest")
        )
        visit_invocation_matches += int(visit_invocation_match)
        search_result_matches += int(
            left_search.get("result_digest") == right_search.get("result_digest")
        )
        visit_result_matches += int(
            left_visit.get("result_digest") == right_visit.get("result_digest")
        )
        selected_url_matches += int(
            left_task.get("selected_url") == right_task.get("selected_url")
        )
        search_url_list_matches += int(
            left_task.get("search_urls") == right_task.get("search_urls")
        )
        if left.call_graph_mode == "frozen":
            expected_match = (
                left_task.get("expected_url") == right_task.get("expected_url")
            )
            frozen_identity_matches += int(expected_match and visit_invocation_match)
            if not expected_match:
                errors.append(f"{task_id} frozen expected_url differs")
            if not visit_invocation_match:
                errors.append(f"{task_id} frozen visit invocation differs")

    pair_count = len(keys)
    frozen_required = left.call_graph_mode == "frozen"
    claim_identity_passed = not errors and (
        frozen_identity_matches == pair_count
        if frozen_required
        else (
            search_invocation_matches == pair_count
            and visit_invocation_matches == pair_count
            and search_result_matches == pair_count
            and visit_result_matches == pair_count
            and selected_url_matches == pair_count
            and search_url_list_matches == pair_count
        )
    )
    return {
        "passed": claim_identity_passed,
        "claim_identity_basis": (
            "frozen_expected_url_and_exact_visit_invocation"
            if frozen_required
            else "autonomous_full_invocation_result_and_search_selection"
        ),
        "task_pair_count": pair_count,
        "search_invocation_match_count": search_invocation_matches,
        "visit_invocation_match_count": visit_invocation_matches,
        "search_result_match_count": search_result_matches,
        "visit_result_match_count": visit_result_matches,
        "selected_url_match_count": selected_url_matches,
        "search_url_list_match_count": search_url_list_matches,
        "frozen_claim_identity_match_count": frozen_identity_matches,
        "errors": _bounded_errors(errors),
    }


def _task_components(run: ValidatedRun) -> dict[tuple[str, int], dict[str, float]]:
    rows: dict[tuple[str, int], dict[str, float]] = {}
    for key, task in run.tasks_by_key.items():
        task_id = str(task["task_id"])
        search = run.committed_by_task_tool[(task_id, "search")]
        visit = run.committed_by_task_tool[(task_id, "visit")]
        e2e = float(task["e2e_s"])
        llm = float(task["llm_duration_s"])
        search_exposed = float(search["exposed_wait_s"])
        visit_exposed = float(visit["exposed_wait_s"])
        tool_exposed = search_exposed + visit_exposed
        residual = e2e - llm - tool_exposed
        if residual < -0.05:
            raise ValueError(
                f"negative E2E decomposition residual for {task_id}: {residual}"
            )
        rows[key] = {
            "e2e_s": e2e,
            "llm_s": llm,
            "tool_exposed_s": tool_exposed,
            "search_exposed_s": search_exposed,
            "visit_exposed_s": visit_exposed,
            "orchestration_residual_s": residual,
        }
    return rows


def _source_folded_components(
    audits: Sequence[RunAudit],
) -> dict[str, dict[str, float]]:
    observations: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for audit in audits:
        for (source_id, _replica), values in _task_components(audit.run).items():
            for metric, value in values.items():
                observations[source_id][metric].append(value)
    return {
        source_id: {
            metric: statistics.fmean(values)
            for metric, values in sorted(metric_values.items())
        }
        for source_id, metric_values in sorted(observations.items())
    }


def _vllm_metric(run: ValidatedRun, name: str) -> float:
    metrics = _mapping(run.summary["llm"], "llm")["vllm_metric_deltas"]
    return float(_mapping(metrics, "vllm_metric_deltas").get(name, 0.0))


def _run_summary(audit: RunAudit) -> dict[str, Any]:
    run = audit.run
    components = list(_task_components(run).values())
    component_summary = {
        metric: _distribution([row[metric] for row in components])
        for metric in components[0]
    }
    llm = _mapping(run.summary["llm"], "llm")
    task_count = len(run.tasks_by_key)
    prompt_tokens = int(llm["prompt_tokens"])
    completion_tokens = int(llm["completion_tokens"])
    prefix_queries = _vllm_metric(run, "vllm:prefix_cache_queries_total")
    prefix_hits = _vllm_metric(run, "vllm:prefix_cache_hits_total")
    max_model_len = int(_scheduler_environment(run).get("VLLM_MAX_MODEL_LEN"))
    max_prompt_plus_output = 0
    prompt_by_call: dict[str, list[int]] = defaultdict(list)
    for task_events in run.llm_by_task.values():
        for event in task_events:
            usage = _mapping(event.get("usage"), "llm usage")
            prompt = int(usage["prompt_tokens"])
            call_index = int(event["call_index"])
            prompt_by_call[str(call_index)].append(prompt)
            max_output = int(
                run.config[
                    "max_tokens_answer" if call_index == 2 else "max_tokens_tool"
                ]
            )
            max_prompt_plus_output = max(max_prompt_plus_output, prompt + max_output)
    raw_summary = _mapping(run.payload.get("summary"), "summary")
    queue = _mapping(run.summary["queue_timeline"], "queue_timeline")
    max_num_seqs = int(_scheduler_environment(run).get("VLLM_MAX_NUM_SEQS"))
    return {
        "label": audit.label,
        "cell": audit.cell,
        "ordinal": audit.ordinal,
        "input": {"path": str(run.path), "sha256": run.sha256},
        "started_wall_s": raw_summary.get("started_wall_s"),
        "ended_wall_s": raw_summary.get("ended_wall_s"),
        "strict_evidence_eligible": audit.strict_evidence_eligible,
        "gates": {
            "exact_counts": dict(audit.exact_counts),
            "http_attempt_logs": dict(audit.http_attempts),
            "canary_zero_speculation": dict(audit.canary),
            "treatment": _treatment_audit(run, audit.cell),
            "context_length_safe": _gate(
                max_prompt_plus_output,
                f"max(prompt_tokens + configured max output) < {max_model_len}",
                max_prompt_plus_output < max_model_len,
            ),
            "natural_llm_queue_below_runtime_capacity": _gate(
                float(queue["max_llm_waiting"]),
                f"max_llm_waiting < VLLM_MAX_NUM_SEQS ({max_num_seqs})",
                float(queue["max_llm_waiting"]) < max_num_seqs,
            ),
            "joint_pressure": _gate(
                float(queue["joint_pressure_fraction"]),
                f">= {MIN_JOINT_PRESSURE_FRACTION}",
                float(queue["joint_pressure_fraction"])
                >= MIN_JOINT_PRESSURE_FRACTION,
            ),
        },
        "components": component_summary,
        "task_e2e_s": dict(run.summary["task_e2e_s"]),
        "task_window_makespan_s": run.summary["task_window_makespan_s"],
        "task_completion_makespan_s": run.summary["task_completion_makespan_s"],
        "llm": {
            "request_duration_s": dict(llm["request_duration_s"]),
            "request_count": llm["request_count"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_by_call": {
                call: _distribution(values)
                for call, values in sorted(prompt_by_call.items())
            },
            "vllm_queue_time_s": _vllm_metric(
                run, "vllm:request_queue_time_seconds_sum"
            ),
            "vllm_inference_time_s": _vllm_metric(
                run, "vllm:request_inference_time_seconds_sum"
            ),
            "vllm_prefill_time_s": _vllm_metric(
                run, "vllm:request_prefill_time_seconds_sum"
            ),
            "vllm_decode_time_s": _vllm_metric(
                run, "vllm:request_decode_time_seconds_sum"
            ),
            "prefix_cache_queries": prefix_queries,
            "prefix_cache_hits": prefix_hits,
            "prefix_cache_hit_ratio": _ratio(prefix_hits, prefix_queries),
        },
        "tool": dict(run.summary["tool"]),
        "queue_timeline": dict(queue),
        "context": {
            "max_model_len": max_model_len,
            "max_prompt_plus_configured_output_tokens": max_prompt_plus_output,
            "context_padding_actual_tokens": _distribution(
                [float(task["context_padding_actual_tokens"]) for task in run.tasks_by_id.values()]
            ),
        },
    }


def _cell_summary(audits: Sequence[RunAudit]) -> dict[str, Any]:
    if not audits:
        raise ValueError("cannot summarize an empty cell")
    source = _source_folded_components(audits)
    pooled = [
        row
        for audit in audits
        for row in _task_components(audit.run).values()
    ]
    run_summaries = [_run_summary(audit) for audit in audits]
    component_means = {
        metric: statistics.fmean(values[metric] for values in source.values())
        for metric in next(iter(source.values()))
    }
    token_names = ("prompt_tokens", "completion_tokens", "total_tokens")
    tokens = {
        name: statistics.fmean(float(row["llm"][name]) for row in run_summaries)
        for name in token_names
    }
    return {
        "cell": audits[0].cell,
        "run_count": len(audits),
        "all_runs_strict_evidence_eligible": all(
            audit.strict_evidence_eligible for audit in audits
        ),
        "runs": run_summaries,
        "source_folded": {
            "sampling_unit": "independent_source_mean_over_runs_and_replicas",
            "source_count": len(source),
            "by_source": source,
            "component_means_s": component_means,
            "e2e_distribution_s": _distribution(
                [values["e2e_s"] for values in source.values()]
            ),
        },
        "pooled_task": {
            "observation_count": len(pooled),
            "components": {
                metric: _distribution([row[metric] for row in pooled])
                for metric in pooled[0]
            },
        },
        "run_mean_tokens": tokens,
        "run_makespan_s": _distribution(
            [float(row["task_completion_makespan_s"]) for row in run_summaries]
        ),
        "queue": {
            metric: _distribution(
                [float(row["queue_timeline"][metric]) for row in run_summaries]
            )
            for metric in (
                "tool_queue_sample_fraction",
                "max_tool_queued",
                "max_llm_running",
                "max_llm_waiting",
                "joint_pressure_fraction",
            )
        },
    }


def _bootstrap_effect(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    *,
    resamples: int,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("cannot bootstrap unmatched or empty source sets")
    source_ids = sorted(baseline)
    metrics = tuple(next(iter(baseline.values())))
    rng = random.Random(BOOTSTRAP_SEED)
    absolute: dict[str, list[float]] = {metric: [] for metric in metrics}
    relative_e2e: list[float] = []
    for _ in range(resamples):
        sample = [source_ids[rng.randrange(len(source_ids))] for _ in source_ids]
        for metric in metrics:
            base_mean = statistics.fmean(baseline[source][metric] for source in sample)
            candidate_mean = statistics.fmean(
                candidate[source][metric] for source in sample
            )
            absolute[metric].append(base_mean - candidate_mean)
            if metric == "e2e_s":
                relative_e2e.append(_ratio(base_mean - candidate_mean, base_mean))
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": resamples,
        "sampling_unit": "independent_source_mean_over_runs_and_replicas",
        "sample_size": len(source_ids),
        "absolute_reduction_s_95_ci": {
            metric: [
                _percentile(samples, 0.025),
                _percentile(samples, 0.975),
            ]
            for metric, samples in absolute.items()
        },
        "e2e_relative_reduction_95_ci": [
            _percentile(relative_e2e, 0.025),
            _percentile(relative_e2e, 0.975),
        ],
    }


def _token_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    task_count: int,
    configured_decode_tokens_per_s: float | None,
    observed_e2e_reduction_s: float,
    llm_component_reduction_s: float,
    tool_component_reduction_s: float,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    maximum = 0.0
    for metric in ("prompt_tokens", "completion_tokens", "total_tokens"):
        base = float(baseline[metric])
        cand = float(candidate[metric])
        relative = _relative_delta(base, cand)
        maximum = max(maximum, abs(relative))
        rows[metric] = {
            "baseline_run_mean": base,
            "candidate_run_mean": cand,
            "absolute_delta": cand - base,
            "relative_delta": relative,
        }
    decode_equivalent: float | None = None
    if configured_decode_tokens_per_s is not None and configured_decode_tokens_per_s > 0:
        completion_shortfall = max(
            0.0,
            float(baseline["completion_tokens"])
            - float(candidate["completion_tokens"]),
        )
        decode_equivalent = (
            completion_shortfall / task_count / configured_decode_tokens_per_s
        )
    cannot_explain = bool(
        observed_e2e_reduction_s > 0.0
        and float(candidate["completion_tokens"])
        < float(baseline["completion_tokens"])
        and llm_component_reduction_s <= 0.0
        and tool_component_reduction_s > 0.0
    )
    return {
        "metrics": rows,
        "max_absolute_relative_difference": maximum,
        "balance_gate": _gate(
            maximum,
            f"<= {MAX_TOKEN_RELATIVE_DIFFERENCE}",
            maximum <= MAX_TOKEN_RELATIVE_DIFFERENCE,
        ),
        "configured_decode_tokens_per_s": configured_decode_tokens_per_s,
        "completion_shortfall_decode_time_equivalent_s_per_task": decode_equivalent,
        "decode_equivalent_fraction_of_observed_e2e_reduction": (
            _ratio(decode_equivalent, observed_e2e_reduction_s)
            if decode_equivalent is not None and observed_e2e_reduction_s > 0.0
            else None
        ),
        "candidate_llm_component_is_not_faster": llm_component_reduction_s <= 0.0,
        "tool_wait_reduction_is_positive": tool_component_reduction_s > 0.0,
        "token_shortfall_direction_cannot_explain_observed_e2e_gain": cannot_explain,
    }


def _decode_rate(audits: Sequence[RunAudit]) -> float | None:
    values: set[float] = set()
    for audit in audits:
        raw = _scheduler_environment(audit.run).get(
            "VLLM_SCHED_DECODE_TOKENS_PER_S_V2"
        )
        if raw is None:
            continue
        values.add(float(raw))
    return next(iter(values)) if len(values) == 1 else None


def _effect_summary(
    baseline_audits: Sequence[RunAudit],
    candidate_audits: Sequence[RunAudit],
    baseline_cell: Mapping[str, Any],
    candidate_cell: Mapping[str, Any],
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    baseline_source = _mapping(
        _mapping(baseline_cell["source_folded"], "baseline source")["by_source"],
        "baseline by_source",
    )
    candidate_source = _mapping(
        _mapping(candidate_cell["source_folded"], "candidate source")["by_source"],
        "candidate by_source",
    )
    if set(baseline_source) != set(candidate_source):
        raise ValueError("effect source sets differ")
    metrics = tuple(next(iter(baseline_source.values())))
    means: dict[str, dict[str, float]] = {}
    for metric in metrics:
        base = statistics.fmean(
            float(_mapping(values, "source values")[metric])
            for values in baseline_source.values()
        )
        cand = statistics.fmean(
            float(_mapping(values, "source values")[metric])
            for values in candidate_source.values()
        )
        means[metric] = _latency_delta(base, cand)
    e2e_reduction = means["e2e_s"]["absolute_reduction_s"]
    component_sum = sum(
        means[metric]["absolute_reduction_s"]
        for metric in ("llm_s", "tool_exposed_s", "orchestration_residual_s")
    )
    faster_sources = [
        source_id
        for source_id in baseline_source
        if float(_mapping(baseline_source[source_id], "baseline source")["e2e_s"])
        > float(_mapping(candidate_source[source_id], "candidate source")["e2e_s"])
    ]
    base_pooled = _mapping(
        _mapping(baseline_cell["pooled_task"], "baseline pooled")["components"],
        "baseline components",
    )
    cand_pooled = _mapping(
        _mapping(candidate_cell["pooled_task"], "candidate pooled")["components"],
        "candidate components",
    )
    base_e2e = _mapping(base_pooled["e2e_s"], "baseline e2e")
    cand_e2e = _mapping(cand_pooled["e2e_s"], "candidate e2e")
    base_makespan = float(
        _mapping(baseline_cell["run_makespan_s"], "baseline makespan")["mean"]
    )
    cand_makespan = float(
        _mapping(candidate_cell["run_makespan_s"], "candidate makespan")["mean"]
    )
    base_tokens = _mapping(baseline_cell["run_mean_tokens"], "baseline tokens")
    cand_tokens = _mapping(candidate_cell["run_mean_tokens"], "candidate tokens")
    decode_rates = [
        rate
        for rate in (_decode_rate(baseline_audits), _decode_rate(candidate_audits))
        if rate is not None
    ]
    decode_rate = decode_rates[0] if decode_rates and len(set(decode_rates)) == 1 else None
    tokens = _token_comparison(
        base_tokens,
        cand_tokens,
        task_count=EXPECTED_TASK_COUNT,
        configured_decode_tokens_per_s=decode_rate,
        observed_e2e_reduction_s=e2e_reduction,
        llm_component_reduction_s=means["llm_s"]["absolute_reduction_s"],
        tool_component_reduction_s=means["tool_exposed_s"]["absolute_reduction_s"],
    )
    config_pairs = [
        _config_pair_audit(
            left.run,
            right.run,
            left_cell=left.cell,
            right_cell=right.cell,
        )
        for left in baseline_audits
        for right in candidate_audits
    ]
    identity_pairs = [
        _identity_pair_audit(left.run, right.run)
        for left in baseline_audits
        for right in candidate_audits
    ]
    strict_inputs = all(
        audit.strict_evidence_eligible
        for audit in (*baseline_audits, *candidate_audits)
    )
    strict_config = all(pair["passed"] for pair in config_pairs)
    strict_identity = all(pair["passed"] for pair in identity_pairs)
    bootstrap = _bootstrap_effect(
        baseline_source,
        candidate_source,
        resamples=bootstrap_resamples,
    )
    performance_gates = {
        "mean_e2e_reduction": _gate(
            means["e2e_s"]["relative_reduction"],
            f">= {MIN_E2E_REDUCTION}",
            means["e2e_s"]["relative_reduction"] >= MIN_E2E_REDUCTION,
        ),
        "faster_source_fraction": _gate(
            len(faster_sources) / len(baseline_source),
            f">= {MIN_FASTER_SOURCE_FRACTION}",
            len(faster_sources) / len(baseline_source)
            >= MIN_FASTER_SOURCE_FRACTION,
        ),
        "bootstrap_mean_reduction_positive": _gate(
            bootstrap["e2e_relative_reduction_95_ci"][0],
            "> 0",
            bootstrap["e2e_relative_reduction_95_ci"][0] > 0.0,
        ),
        "task_p95_ratio": _gate(
            _ratio(float(cand_e2e["p95"]), float(base_e2e["p95"])),
            f"<= {MAX_TASK_P95_RATIO}",
            _ratio(float(cand_e2e["p95"]), float(base_e2e["p95"]))
            <= MAX_TASK_P95_RATIO,
        ),
        "makespan_ratio": _gate(
            _ratio(cand_makespan, base_makespan),
            f"<= {MAX_MAKESPAN_RATIO}",
            _ratio(cand_makespan, base_makespan) <= MAX_MAKESPAN_RATIO,
        ),
        "token_balance": dict(tokens["balance_gate"]),
    }
    return {
        "design": {
            "baseline_run_count": len(baseline_audits),
            "candidate_run_count": len(candidate_audits),
            "source_count": len(baseline_source),
            "source_folding": "mean over runs and replicas within each source",
        },
        "eligibility": {
            "all_input_runs_strict_evidence": strict_inputs,
            "all_cross_run_config_pairs_pass": strict_config,
            "all_cross_run_identity_pairs_pass": strict_identity,
            "token_balance_pass": tokens["balance_gate"]["passed"],
            "strict_performance_claim_eligible": bool(
                strict_inputs
                and strict_config
                and strict_identity
                and tokens["balance_gate"]["passed"]
            ),
            "config_pair_audits": config_pairs,
            "identity_pair_audits": identity_pairs,
        },
        "source_paired": {
            "component_comparisons": means,
            "faster_source_count": len(faster_sources),
            "faster_source_fraction": len(faster_sources) / len(baseline_source),
            "faster_source_ids": sorted(faster_sources),
            "bootstrap": bootstrap,
        },
        "decomposition": {
            "identity": (
                "e2e_s = llm_s + tool_exposed_s + orchestration_residual_s"
            ),
            "component_comparisons": means,
            "search_plus_visit_consistency_error_s": (
                means["tool_exposed_s"]["absolute_reduction_s"]
                - means["search_exposed_s"]["absolute_reduction_s"]
                - means["visit_exposed_s"]["absolute_reduction_s"]
            ),
            "reduction_accounting_error_s": e2e_reduction - component_sum,
            "share_of_e2e_reduction": {
                metric: _ratio(
                    means[metric]["absolute_reduction_s"], e2e_reduction
                )
                for metric in (
                    "llm_s",
                    "tool_exposed_s",
                    "orchestration_residual_s",
                )
            },
        },
        "tails": {
            "task_e2e_s": {
                metric: _latency_delta(
                    float(base_e2e[metric]), float(cand_e2e[metric])
                )
                for metric in ("mean", "p50", "p95", "p99", "max")
            },
            "task_completion_makespan_s": _latency_delta(
                base_makespan, cand_makespan
            ),
        },
        "tokens": tokens,
        "queue": {
            "baseline": baseline_cell["queue"],
            "candidate": candidate_cell["queue"],
        },
        "performance_gates": performance_gates,
        "all_performance_gates_pass": all(
            gate["passed"] for gate in performance_gates.values()
        ),
    }


def _ordered_pair_summary(
    baseline: RunAudit,
    candidate: RunAudit,
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    baseline_cell = _cell_summary([baseline])
    candidate_cell = _cell_summary([candidate])
    effect = _effect_summary(
        [baseline],
        [candidate],
        baseline_cell,
        candidate_cell,
        bootstrap_resamples=bootstrap_resamples,
    )
    base_started = float(
        _mapping(baseline.run.payload.get("summary"), "baseline summary")[
            "started_wall_s"
        ]
    )
    cand_started = float(
        _mapping(candidate.run.payload.get("summary"), "candidate summary")[
            "started_wall_s"
        ]
    )
    if base_started < cand_started:
        execution_order = f"{baseline.label}_before_{candidate.label}"
    else:
        execution_order = f"{candidate.label}_before_{baseline.label}"
    return {
        "pair": f"{baseline.label}_to_{candidate.label}",
        "execution_order": execution_order,
        "baseline_started_wall_s": base_started,
        "candidate_started_wall_s": cand_started,
        "effect": effect,
    }


def _load_audits(cell: str, paths: Sequence[Path]) -> list[RunAudit]:
    if not paths:
        raise ValueError(f"cell {cell} has no input runs")
    audits: list[RunAudit] = []
    for ordinal, path in enumerate(paths, 1):
        role = "candidate" if cell == "V" else "baseline"
        run = _validate_run(path, role=role)
        audits.append(
            RunAudit(
                cell=cell,
                ordinal=ordinal,
                run=run,
                exact_counts=_validate_exact_counts(run),
                http_attempts=_audit_http_attempt_logs(run),
                canary=_audit_canary_non_speculation(run),
            )
        )
    return audits


def compare_live_joint_dev_triplet(
    *,
    a_results: Sequence[Path],
    n_results: Sequence[Path],
    v_results: Sequence[Path],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    audits = {
        "A": _load_audits("A", a_results),
        "N": _load_audits("N", n_results),
        "V": _load_audits("V", v_results),
    }
    cells = {cell: _cell_summary(rows) for cell, rows in audits.items()}
    effects = {
        effect: _effect_summary(
            audits[baseline],
            audits[candidate],
            cells[baseline],
            cells[candidate],
            bootstrap_resamples=bootstrap_resamples,
        )
        for effect, (baseline, candidate) in EFFECTS.items()
    }
    ordered_pairs = {
        effect: [
            _ordered_pair_summary(
                left,
                right,
                bootstrap_resamples=bootstrap_resamples,
            )
            for left in audits[baseline]
            for right in audits[candidate]
        ]
        for effect, (baseline, candidate) in EFFECTS.items()
    }
    strict_pairs = [
        row["pair"]
        for rows in ordered_pairs.values()
        for row in rows
        if row["effect"]["eligibility"]["strict_performance_claim_eligible"]
    ]
    transport_config_identity_pairs = [
        row["pair"]
        for rows in ordered_pairs.values()
        for row in rows
        if row["effect"]["eligibility"]["all_input_runs_strict_evidence"]
        and row["effect"]["eligibility"]["all_cross_run_config_pairs_pass"]
        and row["effect"]["eligibility"]["all_cross_run_identity_pairs_pass"]
    ]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "design": {
            "cells": CELL_TREATMENTS,
            "expected_exact_success_counts": {
                "tasks": EXPECTED_TASK_COUNT,
                "llm_requests": EXPECTED_LLM_REQUEST_COUNT,
                "authoritative_tool_commits": EXPECTED_AUTHORITATIVE_COMMIT_COUNT,
            },
            "independent_source_count": EXPECTED_SOURCE_COUNT,
            "replicas_per_run": EXPECTED_REPLICAS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": bootstrap_resamples,
            "token_balance_limit": MAX_TOKEN_RELATIVE_DIFFERENCE,
        },
        "cells": cells,
        "effects": effects,
        "ordered_run_pairs": ordered_pairs,
        "claim_summary": {
            "strict_performance_claim_eligible_pairs": strict_pairs,
            "transport_config_identity_eligible_pairs_before_token_gate": (
                transport_config_identity_pairs
            ),
            "diagnostic_only_pairs": sorted(
                {
                    row["pair"]
                    for rows in ordered_pairs.values()
                    for row in rows
                }
                - set(strict_pairs)
            ),
        },
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-result", type=Path, action="append", required=True)
    parser.add_argument("--n-result", type=Path, action="append", required=True)
    parser.add_argument("--v-result", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = compare_live_joint_dev_triplet(
            a_results=args.a_result,
            n_results=args.n_result,
            v_results=args.v_result,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        _write_json_atomic(args.output, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"live A/N/V development comparison failed: {exc}", file=sys.stderr)
        return 2
    compact = {
        name: {
            "relative_reduction": effect["source_paired"][
                "component_comparisons"
            ]["e2e_s"]["relative_reduction"],
            "strict_performance_claim_eligible": effect["eligibility"][
                "strict_performance_claim_eligible"
            ],
            "token_balance_pass": effect["eligibility"]["token_balance_pass"],
        }
        for name, effect in result["effects"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
