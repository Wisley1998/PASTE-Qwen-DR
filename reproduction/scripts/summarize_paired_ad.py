#!/usr/bin/env python3
"""Summarize paired final/heldout/stress A and D runs.

Each ``--pair`` is one replicate block.  Session-level deltas are calculated
inside each block as ``A task flow - D task flow`` so positive values mean the
joint learned path is faster.  Across replicates, deltas are first averaged for
each identical session and only then summarized across sessions; repeated runs
of the same evaluation sessions are therefore not treated as independent samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from summarize_four_cell import (  # noqa: E402
    canonical_sha256,
    load_fixed_manifest,
    load_run,
    percentile,
    repository_display_path,
)


SCHEMA = "paste_repro.paired_ad_summary"
VERSION = 1
KV_SWAP_SEMANTICS_V2 = "cpu_swap_only_v2"
TIE_EPSILON_S = 1e-9
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_RESAMPLES = 10_000
STAT_NAMES = ("mean", "p50", "p95", "max")
METRIC_PATHS: tuple[tuple[str, ...], ...] = (
    *(("task_flow_time_s", statistic) for statistic in STAT_NAMES),
    ("task_makespan_s",),
    *(("request_latency_s", statistic) for statistic in STAT_NAMES),
    ("mean_queue_time_s",),
    ("instrumentation_wall_time_s",),
)
_JOINT_ONLY_SCHEDULER_KEYS = {
    "VLLM_SCHED_AVG_CALL_SERVICE_S",
    "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2",
    "VLLM_SCHED_DECODE_TOKENS_PER_S_V2",
    "VLLM_SCHED_TIME_AGING_ALPHA",
}
_JOINT_ONLY_SCHEDULER_PREFIXES = (
    "VLLM_SCHED_JOINT_",
    "VLLM_SCHED_HBM_",
)


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0 or number != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _optional_nonnegative_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return _nonnegative_integer(payload[key], f"{label} {key}")


def _optional_finite_nonnegative(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> float | None:
    if key not in payload or payload[key] is None:
        return None
    return _finite_nonnegative(payload[key], f"{label} {key}")


def _load_optional_json_object(path: Path, label: str) -> Mapping[str, Any] | None:
    """Load optional raw evidence without treating absence as an empty object."""

    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root is not an object: {path}")
    return payload


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Fixed-seed percentile bootstrap over independent sampling units."""

    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("bootstrap sample is empty")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not all(math.isfinite(value) for value in sample):
        raise ValueError("bootstrap sample must be finite")
    generator = random.Random(seed)
    sample_size = len(sample)
    bootstrap_means = [
        statistics.fmean(
            sample[generator.randrange(sample_size)] for _ in range(sample_size)
        )
        for _ in range(resamples)
    ]
    return {
        "method": "nonparametric_percentile_bootstrap",
        "estimand": "mean_A_minus_D_task_flow_s",
        "sampling_unit": "independent_source_session_mean",
        "confidence_level": 0.95,
        "seed": seed,
        "resamples": resamples,
        "sample_size": sample_size,
        "lower_s": percentile(bootstrap_means, 0.025),
        "upper_s": percentile(bootstrap_means, 0.975),
    }


def _load_raw_execution_accounting(
    run_path: Path,
    validated_public: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract execution counters after ``load_run`` validates the raw files.

    Old immutable runs did not record serving counters or token usage.  Missing
    evidence is therefore represented as unavailable/``None`` rather than being
    silently converted to zero.  Present evidence is validated fail-closed.
    """

    summary_path = run_path / "summary.json"
    events_path = run_path / "request_events.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError(f"summary is not an object: {run_path}")
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(event, Mapping) for event in events):
        raise ValueError(f"request events contain a non-object: {run_path}")

    requests_total = _nonnegative_integer(
        summary.get("requests_total"), "summary requests_total"
    )
    requests_success = _nonnegative_integer(
        summary.get("requests_success"), "summary requests_success"
    )
    requests_failed = _nonnegative_integer(
        summary.get("requests_failed"), "summary requests_failed"
    )
    event_successes = sum(event.get("ok") is True for event in events)
    event_failures = sum(event.get("ok") is not True for event in events)
    if (
        requests_total != len(events)
        or requests_success != event_successes
        or requests_failed != event_failures
        or requests_success + requests_failed != requests_total
        or requests_total != int(validated_public["request_count"])
    ):
        raise ValueError(f"raw request outcome accounting mismatch: {run_path}")

    completion_tokens: list[int] = []
    usage_presence: list[bool] = []
    for event_number, event in enumerate(events):
        usage = event.get("usage")
        present = isinstance(usage, Mapping) and "completion_tokens" in usage
        usage_presence.append(present)
        if present:
            completion_tokens.append(
                _nonnegative_integer(
                    usage["completion_tokens"],
                    f"event {event_number} usage completion_tokens",
                )
            )
        elif usage is not None:
            raise ValueError(
                f"event {event_number} has incomplete usage accounting: {run_path}"
            )
    if any(usage_presence) and not all(usage_presence):
        raise ValueError(f"partial completion-token accounting: {run_path}")
    token_accounting_available = bool(events) and all(usage_presence)
    completion_tokens_total = (
        sum(completion_tokens) if token_accounting_available else None
    )

    num_preemptions_total = _optional_nonnegative_integer(
        summary,
        "num_preemptions_total",
        label="summary",
    )
    preemption_warning_count = _optional_nonnegative_integer(
        summary,
        "preemption_warning_count",
        label="summary",
    )
    preemption_available = num_preemptions_total is not None
    preemption_happened = (
        num_preemptions_total > 0 if num_preemptions_total is not None else None
    )
    if "preemption_happened" in summary:
        recorded_preemption_happened = summary["preemption_happened"]
        if recorded_preemption_happened is not None and type(
            recorded_preemption_happened
        ) is not bool:
            raise ValueError(f"summary preemption_happened must be boolean: {run_path}")
        if (
            preemption_happened is not None
            and recorded_preemption_happened != preemption_happened
        ):
            raise ValueError(f"summary preemption happened/count mismatch: {run_path}")

    swap_integer_fields = (
        "kv_swap_event_count",
        "kv_swap_in_event_count",
        "kv_swap_out_event_count",
        "kv_swap_total_blocks",
        "max_swapped_requests",
    )
    swap_float_fields = (
        "kv_swap_total_time_s",
        "kv_swap_avg_time_s",
        "kv_swap_in_avg_time_s",
        "kv_swap_out_avg_time_s",
    )
    swap_values: dict[str, Any] = {
        key: _optional_nonnegative_integer(summary, key, label="summary")
        for key in swap_integer_fields
    }
    swap_values.update(
        {
            key: _optional_finite_nonnegative(summary, key, label="summary")
            for key in swap_float_fields
        }
    )

    # When raw sidecars are present they are authoritative evidence, not a
    # second set of defaults.  Missing sidecars remain unavailable so legacy
    # runs without them never acquire synthetic zero counters.
    raw_swap = _load_optional_json_object(
        run_path / "swap_summary.json", "swap_summary.json"
    )
    if raw_swap is not None:
        raw_integer_fields = {
            "kv_swap_event_count": "swap_event_count",
            "kv_swap_in_event_count": "swap_in_event_count",
            "kv_swap_out_event_count": "swap_out_event_count",
            "kv_swap_total_blocks": "swap_total_blocks",
        }
        raw_float_fields = {
            "kv_swap_total_time_s": "swap_total_time_s",
            "kv_swap_avg_time_s": "swap_avg_time_s",
            "kv_swap_in_avg_time_s": "swap_in_avg_time_s",
            "kv_swap_out_avg_time_s": "swap_out_avg_time_s",
        }
        for summary_key, raw_key in raw_integer_fields.items():
            raw_value = _nonnegative_integer(
                raw_swap.get(raw_key), f"swap_summary {raw_key}"
            )
            if (
                swap_values[summary_key] is not None
                and swap_values[summary_key] != raw_value
            ):
                raise ValueError(
                    f"summary/raw swap event mismatch for {summary_key}: {run_path}"
                )
            swap_values[summary_key] = raw_value
        for summary_key, raw_key in raw_float_fields.items():
            raw_value = _finite_nonnegative(
                raw_swap.get(raw_key), f"swap_summary {raw_key}"
            )
            if (
                swap_values[summary_key] is not None
                and not math.isclose(
                    swap_values[summary_key], raw_value, rel_tol=1e-9, abs_tol=1e-9
                )
            ):
                raise ValueError(
                    f"summary/raw swap event mismatch for {summary_key}: {run_path}"
                )
            swap_values[summary_key] = raw_value

    raw_log_summary = _load_optional_json_object(
        run_path / "vllm_log_summary.json", "vllm_log_summary.json"
    )
    if raw_log_summary is not None:
        raw_max_swapped = _nonnegative_integer(
            raw_log_summary.get("max_swapped_requests"),
            "vllm_log_summary max_swapped_requests",
        )
        if (
            swap_values["max_swapped_requests"] is not None
            and swap_values["max_swapped_requests"] != raw_max_swapped
        ):
            raise ValueError(f"summary/raw max swapped mismatch: {run_path}")
        swap_values["max_swapped_requests"] = raw_max_swapped
        raw_preemption_warnings = _nonnegative_integer(
            raw_log_summary.get("preemption_warning_count"),
            "vllm_log_summary preemption_warning_count",
        )
        if (
            preemption_warning_count is not None
            and preemption_warning_count != raw_preemption_warnings
        ):
            raise ValueError(f"summary/raw preemption warning mismatch: {run_path}")
        preemption_warning_count = raw_preemption_warnings

    if "kv_swap_happened" in summary:
        if type(summary["kv_swap_happened"]) is not bool:
            raise ValueError(f"summary kv_swap_happened must be boolean: {run_path}")
        recorded_swap_happened: bool | None = summary["kv_swap_happened"]
    else:
        recorded_swap_happened = None
    recorded_semantics = summary.get("kv_swap_happened_semantics")
    if recorded_semantics is not None and recorded_semantics != KV_SWAP_SEMANTICS_V2:
        raise ValueError(f"unknown kv_swap_happened semantics: {run_path}")

    event_evidence_available = swap_values["kv_swap_event_count"] is not None
    swapped_request_evidence_available = (
        swap_values["max_swapped_requests"] is not None
    )
    positive_swap_evidence = bool(
        (swap_values["kv_swap_event_count"] or 0) > 0
        or (swap_values["max_swapped_requests"] or 0) > 0
    )
    if positive_swap_evidence:
        actual_swap_happened: bool | None = True
    elif event_evidence_available and swapped_request_evidence_available:
        actual_swap_happened = False
    else:
        actual_swap_happened = None

    legacy_preemption_conflated_swap_flag: bool | None = (
        False if actual_swap_happened is not None else None
    )
    if recorded_semantics == KV_SWAP_SEMANTICS_V2:
        if (
            actual_swap_happened is None
            or recorded_swap_happened is None
            or recorded_swap_happened != actual_swap_happened
        ):
            raise ValueError(f"summary swap happened/count mismatch: {run_path}")
    elif actual_swap_happened is not None and recorded_swap_happened is not None:
        if recorded_swap_happened != actual_swap_happened:
            legacy_conflation = bool(
                recorded_swap_happened
                and not actual_swap_happened
                and num_preemptions_total is not None
                and num_preemptions_total > 0
            )
            if not legacy_conflation:
                raise ValueError(f"summary swap happened/count mismatch: {run_path}")
            legacy_preemption_conflated_swap_flag = True

    # Always publish the normalized CPU-swap result.  For old summaries that
    # folded a recompute preemption into the flag, preserve the recorded value
    # and mark the compatibility normalization explicitly.
    swap_happened = actual_swap_happened
    swap_available = swap_happened is not None and all(
        swap_values[key] is not None
        for key in (
            "kv_swap_event_count",
            "kv_swap_in_event_count",
            "kv_swap_out_event_count",
            "kv_swap_total_blocks",
            "kv_swap_total_time_s",
        )
    )

    return {
        "request_outcomes": {
            "requests_total": requests_total,
            "requests_success": requests_success,
            "requests_failed": requests_failed,
        },
        "completion_tokens": {
            "available": token_accounting_available,
            "source": "request_events.jsonl[].usage.completion_tokens",
            "requests_with_usage": len(completion_tokens),
            "total": completion_tokens_total,
        },
        "preemption": {
            "available": preemption_available,
            "source": summary.get("num_preemptions_metric"),
            "num_preemptions_total": num_preemptions_total,
            "preemption_happened": preemption_happened,
            "preemption_warning_count": preemption_warning_count,
        },
        "swap": {
            "available": swap_available,
            "kv_swap_happened": swap_happened,
            "recorded_kv_swap_happened": recorded_swap_happened,
            "recorded_kv_swap_happened_semantics": recorded_semantics,
            "normalized_kv_swap_happened_semantics": KV_SWAP_SEMANTICS_V2,
            "legacy_preemption_conflated_swap_flag": (
                legacy_preemption_conflated_swap_flag
            ),
            "swap_event_sidecar_available": raw_swap is not None,
            "swapped_request_log_evidence_available": (
                swapped_request_evidence_available
            ),
            **swap_values,
        },
        "missing_execution_evidence_is_not_zero": True,
    }


def _shared_scheduler_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Return settings that can affect both FCFS A and Joint D.

    Joint score/admission knobs are inert under FCFS and may intentionally
    differ in a tuned A/D comparison.  Unknown keys remain shared by default,
    so newly added engine or replay settings still fail closed.
    """

    return {
        key: value
        for key, value in configuration.items()
        if key not in _JOINT_ONLY_SCHEDULER_KEYS
        and not key.startswith(_JOINT_ONLY_SCHEDULER_PREFIXES)
    }


def _delta_stats(values: Iterable[float]) -> dict[str, float]:
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("paired delta sample is empty")
    return {
        "mean": statistics.fmean(sample),
        "p05": percentile(sample, 0.05),
        "p50": percentile(sample, 0.50),
        "p95": percentile(sample, 0.95),
        "min": min(sample),
        "max": max(sample),
    }


def _outcome(delta_s: float) -> str:
    if delta_s > TIE_EPSILON_S:
        return "joint_faster"
    if delta_s < -TIE_EPSILON_S:
        return "joint_slower"
    return "tie"


def _outcome_counts(values: Iterable[float]) -> dict[str, Any]:
    sample = [float(value) for value in values]
    wins = sum(value > TIE_EPSILON_S for value in sample)
    losses = sum(value < -TIE_EPSILON_S for value in sample)
    ties = len(sample) - wins - losses
    return {
        "joint_faster": wins,
        "tie": ties,
        "joint_slower": losses,
        "joint_faster_fraction": wins / len(sample) if sample else 0.0,
    }


def _collapse_pair_rows_by_source(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_instance_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_session"]), []).append(row)
    if set(grouped) != set(expected_instance_counts):
        raise ValueError("paired rows do not contain the validated source registry")
    collapsed: list[dict[str, Any]] = []
    for source_session in sorted(grouped):
        instances = grouped[source_session]
        expected_count = int(expected_instance_counts[source_session])
        if len(instances) != expected_count:
            raise ValueError(
                f"source {source_session} has {len(instances)} load instances; "
                f"expected {expected_count}"
            )
        delta = statistics.fmean(float(row["delta_s"]) for row in instances)
        collapsed.append(
            {
                "source_session": source_session,
                "trace_ids": sorted(str(row["trace_id"]) for row in instances),
                "load_instance_count": len(instances),
                "a_task_flow_mean_s": statistics.fmean(
                    float(row["a_task_flow_s"]) for row in instances
                ),
                "d_task_flow_mean_s": statistics.fmean(
                    float(row["d_task_flow_s"]) for row in instances
                ),
                "delta_mean_s": delta,
                "outcome": _outcome(delta),
            }
        )
    return collapsed


def _validate_source_multiplicity(
    source_counts: Counter[str],
    *,
    workload_invariants: Mapping[str, Any],
    replicate: int,
) -> None:
    expected_load_count = int(workload_invariants["load_instance_count"])
    expected_source_count = int(
        workload_invariants["independent_source_session_count"]
    )
    if sum(source_counts.values()) != expected_load_count:
        raise ValueError(
            f"replicate {replicate} workload load-instance count differs from manifest"
        )
    if len(source_counts) != expected_source_count:
        raise ValueError(
            f"replicate {replicate} workload source count differs from manifest"
        )

    exact_instances = workload_invariants["instances_per_source"]
    if exact_instances is not None:
        if set(source_counts.values()) != {int(exact_instances)}:
            raise ValueError(
                f"replicate {replicate} workload source multiplicity differs from "
                "manifest"
            )
        return

    minimum = int(workload_invariants["minimum_instances_per_source"])
    maximum = int(workload_invariants["maximum_instances_per_source"])
    sources_with_extra = int(
        workload_invariants["sources_with_one_extra_instance"]
    )
    observed_distribution = Counter(source_counts.values())
    expected_distribution = Counter(
        {
            minimum: expected_source_count - sources_with_extra,
            maximum: sources_with_extra,
        }
    )
    if observed_distribution != expected_distribution:
        raise ValueError(
            f"replicate {replicate} workload source multiplicity is not the "
            "manifest-balanced distribution"
        )


def _get_metric(payload: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _set_metric(payload: dict[str, Any], path: Sequence[str], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _cell_metrics(
    public: Mapping[str, Any],
    execution_accounting: Mapping[str, Any],
) -> dict[str, Any]:
    request_outcomes = execution_accounting["request_outcomes"]
    completion_tokens = execution_accounting["completion_tokens"]
    preemption = execution_accounting["preemption"]
    swap = execution_accounting["swap"]
    return {
        "run_name": public["run_name"],
        "run_path": public["run_path"],
        "policy": public["policy"],
        "tool_overlap_mode": public["tool_overlap_mode"],
        "trace_count": public["trace_count"],
        "source_session_count": public["source_session_count"],
        "request_count": public["request_count"],
        "requests_success": request_outcomes["requests_success"],
        "requests_failed": request_outcomes["requests_failed"],
        "completion_tokens_total": completion_tokens["total"],
        "num_preemptions_total": preemption["num_preemptions_total"],
        "preemption_happened": preemption["preemption_happened"],
        "kv_swap_happened": swap["kv_swap_happened"],
        "kv_swap_event_count": swap["kv_swap_event_count"],
        "task_flow_time_s": dict(public["task_flow_time_s"]),
        "task_makespan_s": public["task_makespan_s"],
        "request_latency_s": dict(public["request_latency_s"]),
        "mean_queue_time_s": public["mean_queue_time_s"],
        "instrumentation_wall_time_s": public["instrumentation_wall_time_s"],
        "instrumentation_overhang_s": public["instrumentation_overhang_s"],
        "prepared_workload_sha256": public["prepared_workload_sha256"],
        "fixed_role": public["fixed_role"],
        "speedup": public["speedup"],
        "max_active_traces": public["max_active_traces"],
        "tool_wait_mode": public["tool_wait_mode"],
        "configured_max_request_attempts": public[
            "configured_max_request_attempts"
        ],
        "retry_accounting": dict(public["retry_accounting"]),
        "scheduler_configuration": public["scheduler_configuration"],
        "scheduler_calibration_workload_sha256": public[
            "scheduler_calibration_workload_sha256"
        ],
        "scheduler_evidence": public["scheduler_evidence"],
        "execution_accounting": dict(execution_accounting),
    }


def _pair_effect(
    a_metrics: Mapping[str, Any],
    d_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    absolute: dict[str, Any] = {}
    relative: dict[str, Any] = {}
    for path in METRIC_PATHS:
        a_value = _get_metric(a_metrics, path)
        d_value = _get_metric(d_metrics, path)
        reduction = a_value - d_value
        _set_metric(absolute, path, reduction)
        _set_metric(relative, path, reduction / a_value if a_value else None)
    return {
        "definition": "A - D; positive means joint+learned is lower/faster",
        "absolute_reduction": absolute,
        "relative_reduction": relative,
    }


def _completion_token_comparison(
    a_metrics: Mapping[str, Any],
    d_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    a_value = a_metrics.get("completion_tokens_mean_per_replicate")
    d_value = d_metrics.get("completion_tokens_mean_per_replicate")
    available = a_value is not None and d_value is not None
    difference = float(d_value) - float(a_value) if available else None
    return {
        "definition": (
            "D - A completion tokens, using each cell's mean total per replicate; "
            "positive means joint+learned generated more tokens"
        ),
        "available": available,
        "a_mean_per_replicate": a_value,
        "d_mean_per_replicate": d_value,
        "d_minus_a": difference,
        "relative_to_a": (
            difference / float(a_value)
            if available and float(a_value) != 0.0
            else None
        ),
    }


def _task_flow_by_trace(
    run_path: Path,
    validated: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    arrival_by_trace: dict[str, float] = {}
    source_by_trace: dict[str, str] = {}
    for row in validated["identity_rows"]:
        trace_id = str(row["trace_id"])
        arrival = _finite_nonnegative(row["initial_delay_s"], f"{trace_id} arrival")
        source = str(row["source_session"])
        if trace_id in arrival_by_trace and (
            arrival_by_trace[trace_id] != arrival or source_by_trace[trace_id] != source
        ):
            raise ValueError(f"inconsistent static trace identity in {run_path}: {trace_id}")
        arrival_by_trace[trace_id] = arrival
        source_by_trace[trace_id] = source

    completion_by_trace: dict[str, float] = {}
    events_path = run_path / "request_events.jsonl"
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        trace_id = str(event["trace_id"])
        end_offset = _finite_nonnegative(
            event["request_end_offset_s"], f"{trace_id} request end"
        )
        completion_by_trace[trace_id] = max(
            completion_by_trace.get(trace_id, 0.0), end_offset
        )
    if set(completion_by_trace) != set(arrival_by_trace):
        raise ValueError(f"task completion identities do not match workload: {run_path}")

    result: dict[str, dict[str, Any]] = {}
    for trace_id in sorted(arrival_by_trace):
        flow = completion_by_trace[trace_id] - arrival_by_trace[trace_id]
        if flow < -TIE_EPSILON_S:
            raise ValueError(f"task completed before arrival: {run_path}: {trace_id}")
        result[trace_id] = {
            "source_session": source_by_trace[trace_id],
            "initial_delay_s": arrival_by_trace[trace_id],
            "completion_offset_s": completion_by_trace[trace_id],
            "task_flow_s": max(0.0, flow),
        }
    return result


def load_pair(
    a_path: Path,
    d_path: Path,
    replicate: int,
    bindings: Mapping[str, Mapping[str, Any]],
    workload_invariants: Mapping[str, Any],
    *,
    require_identical_scheduler_config: bool = False,
) -> dict[str, Any]:
    a_resolved = a_path.resolve()
    d_resolved = d_path.resolve()
    if a_resolved == d_resolved:
        raise ValueError(f"replicate {replicate} A and D run directories must differ")
    a = load_run(a_resolved, "A", bindings["A"])
    d = load_run(d_resolved, "D", bindings["D"])
    if a["identity_rows"] != d["identity_rows"]:
        raise ValueError(
            f"replicate {replicate} A/D request identity, prompt, or messages mismatch"
        )
    if a["source_mapping"] != d["source_mapping"]:
        raise ValueError(f"replicate {replicate} A/D source sessions mismatch")
    role = str(bindings["A"].get("role"))
    source_counts = Counter(a["source_mapping"].values())
    _validate_source_multiplicity(
        source_counts,
        workload_invariants=workload_invariants,
        replicate=replicate,
    )
    for field in (
        "speedup",
        "max_active_traces",
        "tool_wait_mode",
        "configured_max_request_attempts",
    ):
        if a["public"][field] != d["public"][field]:
            raise ValueError(f"replicate {replicate} A/D configuration mismatch: {field}")
    a_scheduler_configuration = a["public"]["scheduler_configuration"]
    d_scheduler_configuration = d["public"]["scheduler_configuration"]
    if require_identical_scheduler_config:
        comparable_a_configuration = a_scheduler_configuration
        comparable_d_configuration = d_scheduler_configuration
        configuration_label = "scheduler_configuration"
    else:
        comparable_a_configuration = _shared_scheduler_configuration(
            a_scheduler_configuration
        )
        comparable_d_configuration = _shared_scheduler_configuration(
            d_scheduler_configuration
        )
        configuration_label = "shared_scheduler_configuration"
    if comparable_a_configuration != comparable_d_configuration:
        raise ValueError(
            f"replicate {replicate} A/D configuration mismatch: "
            f"{configuration_label}"
        )

    a_flows = _task_flow_by_trace(a_resolved, a)
    d_flows = _task_flow_by_trace(d_resolved, d)
    if set(a_flows) != set(d_flows):
        raise ValueError(f"replicate {replicate} A/D task identities mismatch")
    session_rows = []
    for trace_id in sorted(a_flows):
        a_row = a_flows[trace_id]
        d_row = d_flows[trace_id]
        if (
            a_row["source_session"] != d_row["source_session"]
            or a_row["initial_delay_s"] != d_row["initial_delay_s"]
        ):
            raise ValueError(f"replicate {replicate} A/D task pairing mismatch: {trace_id}")
        delta = a_row["task_flow_s"] - d_row["task_flow_s"]
        session_rows.append(
            {
                "trace_id": trace_id,
                "source_session": a_row["source_session"],
                "initial_delay_s": a_row["initial_delay_s"],
                "a_completion_offset_s": a_row["completion_offset_s"],
                "d_completion_offset_s": d_row["completion_offset_s"],
                "a_task_flow_s": a_row["task_flow_s"],
                "d_task_flow_s": d_row["task_flow_s"],
                "delta_s": delta,
                "outcome": _outcome(delta),
            }
        )
    deltas = [row["delta_s"] for row in session_rows]
    source_rows = _collapse_pair_rows_by_source(
        session_rows,
        expected_instance_counts=source_counts,
    )
    source_deltas = [row["delta_mean_s"] for row in source_rows]
    a_execution = _load_raw_execution_accounting(a_resolved, a["public"])
    d_execution = _load_raw_execution_accounting(d_resolved, d["public"])
    a_metrics = _cell_metrics(a["public"], a_execution)
    d_metrics = _cell_metrics(d["public"], d_execution)
    paired_task_flow = {
        "definition": "A task flow - D task flow; positive means joint+learned faster",
        "tie_epsilon_s": TIE_EPSILON_S,
        "session_count": len(session_rows),
        "load_instance_count": len(session_rows),
        "independent_source_session_count": len(source_rows),
        "instances_per_source": workload_invariants["instances_per_source"],
        "duplicates_are_not_independent": workload_invariants[
            "duplicates_are_not_independent"
        ],
        "outcomes": _outcome_counts(deltas),
        "delta_s": _delta_stats(deltas),
        "load_instance_distribution": True,
        "source_session_outcomes": _outcome_counts(source_deltas),
        "source_session_delta_s": _delta_stats(source_deltas),
        "independent_source_mean_bootstrap_95_ci_s": _bootstrap_mean_ci(
            source_deltas
        ),
        "bootstrap_independence_statement": (
            "Bootstrap resamples only independent source-session means; "
            "deterministic duplicates are averaged before resampling."
        ),
        "source_sessions": source_rows,
        "sessions": session_rows,
    }
    if role == "stress" and "minimum_instances_per_source" in workload_invariants:
        for field in (
            "minimum_instances_per_source",
            "maximum_instances_per_source",
            "sources_with_one_extra_instance",
            "source_instances_are_balanced",
        ):
            paired_task_flow[field] = workload_invariants[field]
    return {
        "replicate": replicate,
        "request_identity_sha256": a["public"]["request_identity_sha256"],
        "source_sessions_sha256": a["public"]["source_sessions_sha256"],
        "mapper_artifact_sha256": d["mapper_checksum"],
        "tool_prediction_top_k": d["mapper_top_k"],
        "cells": {"A": a_metrics, "D": d_metrics},
        "effects": _pair_effect(a_metrics, d_metrics),
        "paired_task_flow": paired_task_flow,
        "_identity_rows": a["identity_rows"],
        "_source_mapping": a["source_mapping"],
    }


def _sum_if_all_present(values: Iterable[int | float | None]) -> int | float | None:
    sample = list(values)
    if any(value is None for value in sample):
        return None
    return sum(value for value in sample if value is not None)


def _aggregate_execution_accounting(
    replicates: Sequence[Mapping[str, Any]],
    cell: str,
) -> dict[str, Any]:
    per_replicate = [
        replicate["cells"][cell]["execution_accounting"]
        for replicate in replicates
    ]
    requests = {
        field: sum(
            int(evidence["request_outcomes"][field]) for evidence in per_replicate
        )
        for field in ("requests_total", "requests_success", "requests_failed")
    }

    token_totals = [
        evidence["completion_tokens"]["total"] for evidence in per_replicate
    ]
    completion_total = _sum_if_all_present(token_totals)
    completion_available = completion_total is not None

    preemptions = [
        evidence["preemption"]["num_preemptions_total"]
        for evidence in per_replicate
    ]
    preemption_warnings = [
        evidence["preemption"]["preemption_warning_count"]
        for evidence in per_replicate
    ]
    preemption_total = _sum_if_all_present(preemptions)
    preemption_happened = (
        preemption_total > 0 if preemption_total is not None else None
    )

    swap_integer_fields = (
        "kv_swap_event_count",
        "kv_swap_in_event_count",
        "kv_swap_out_event_count",
        "kv_swap_total_blocks",
    )
    swap_totals = {
        field: _sum_if_all_present(
            evidence["swap"][field] for evidence in per_replicate
        )
        for field in swap_integer_fields
    }
    swap_total_time = _sum_if_all_present(
        evidence["swap"]["kv_swap_total_time_s"] for evidence in per_replicate
    )
    swap_available = all(evidence["swap"]["available"] for evidence in per_replicate)
    swap_happened = (
        any(evidence["swap"]["kv_swap_happened"] for evidence in per_replicate)
        if swap_available
        else None
    )
    max_swapped_values = [
        evidence["swap"]["max_swapped_requests"] for evidence in per_replicate
    ]
    max_swapped_requests = (
        max(int(value) for value in max_swapped_values if value is not None)
        if max_swapped_values and all(value is not None for value in max_swapped_values)
        else None
    )
    legacy_conflation_values = [
        evidence["swap"]["legacy_preemption_conflated_swap_flag"]
        for evidence in per_replicate
    ]
    legacy_preemption_conflated_swap_flag = (
        any(bool(value) for value in legacy_conflation_values)
        if all(value is not None for value in legacy_conflation_values)
        else None
    )
    return {
        "replicate_count": len(per_replicate),
        "counter_aggregation": "sum_across_replicates",
        "request_outcomes": requests,
        "completion_tokens": {
            "all_replicates_available": completion_available,
            "source": "request_events.jsonl[].usage.completion_tokens",
            "per_replicate_totals": token_totals,
            "total_across_replicates": completion_total,
            "mean_per_replicate": (
                statistics.fmean(float(value) for value in token_totals)
                if completion_available
                else None
            ),
        },
        "preemption": {
            "all_replicates_available": preemption_total is not None,
            "per_replicate_totals": preemptions,
            "num_preemptions_total": preemption_total,
            "preemption_happened": preemption_happened,
            "preemption_warning_count": _sum_if_all_present(
                preemption_warnings
            ),
        },
        "swap": {
            "all_replicates_available": swap_available,
            "kv_swap_happened": swap_happened,
            **swap_totals,
            "kv_swap_total_time_s": swap_total_time,
            "max_swapped_requests": max_swapped_requests,
            "normalized_kv_swap_happened_semantics": KV_SWAP_SEMANTICS_V2,
            "legacy_preemption_conflated_swap_flag": (
                legacy_preemption_conflated_swap_flag
            ),
        },
        "missing_execution_evidence_is_not_zero": True,
    }


def _aggregate_cell(replicates: Sequence[Mapping[str, Any]], cell: str) -> dict[str, Any]:
    configured_attempts = {
        replicate["cells"][cell]["configured_max_request_attempts"]
        for replicate in replicates
    }
    if len(configured_attempts) != 1:
        raise ValueError(f"cell {cell} has inconsistent max request attempts")
    execution_accounting = _aggregate_execution_accounting(replicates, cell)
    request_outcomes = execution_accounting["request_outcomes"]
    completion_tokens = execution_accounting["completion_tokens"]
    preemption = execution_accounting["preemption"]
    swap = execution_accounting["swap"]
    aggregate: dict[str, Any] = {
        "policy": replicates[0]["cells"][cell]["policy"],
        "tool_overlap_mode": replicates[0]["cells"][cell]["tool_overlap_mode"],
        "replicate_count": len(replicates),
        "trace_count": replicates[0]["cells"][cell]["trace_count"],
        "source_session_count": replicates[0]["cells"][cell]["source_session_count"],
        "request_count": replicates[0]["cells"][cell]["request_count"],
        "requests_success": request_outcomes["requests_success"],
        "requests_failed": request_outcomes["requests_failed"],
        "completion_tokens_total": completion_tokens["total_across_replicates"],
        "completion_tokens_mean_per_replicate": completion_tokens[
            "mean_per_replicate"
        ],
        "num_preemptions_total": preemption["num_preemptions_total"],
        "preemption_happened": preemption["preemption_happened"],
        "kv_swap_happened": swap["kv_swap_happened"],
        "kv_swap_event_count": swap["kv_swap_event_count"],
        "configured_max_request_attempts": next(iter(configured_attempts)),
        "retry_accounting": {
            "configured_max_request_attempts": next(iter(configured_attempts)),
            **{
                field: sum(
                    replicate["cells"][cell]["retry_accounting"][field]
                    for replicate in replicates
                )
                for field in (
                    "requests_total",
                    "request_attempts_total",
                    "retry_count",
                    "retried_request_count",
                    "retry_success_count",
                    "ambiguous_retry_count",
                    "final_failure_count",
                )
            },
        },
        "execution_accounting": execution_accounting,
    }
    for path in METRIC_PATHS:
        _set_metric(
            aggregate,
            path,
            statistics.fmean(
                _get_metric(replicate["cells"][cell], path)
                for replicate in replicates
            ),
        )
    return aggregate


def _aggregate_paired_sessions(
    replicates: Sequence[Mapping[str, Any]],
    *,
    role: str,
    workload_invariants: Mapping[str, Any],
) -> dict[str, Any]:
    if role == "stress":
        rows_by_replicate = [
            {
                row["source_session"]: row
                for row in replicate["paired_task_flow"]["source_sessions"]
            }
            for replicate in replicates
        ]
        source_sessions = sorted(rows_by_replicate[0])
        if any(
            set(rows) != set(source_sessions) for rows in rows_by_replicate[1:]
        ):
            raise ValueError("stress replicates do not contain the same source sessions")
        aggregate_rows = []
        for source_session in source_sessions:
            observations = [rows[source_session] for rows in rows_by_replicate]
            trace_id_sets = {tuple(row["trace_ids"]) for row in observations}
            if len(trace_id_sets) != 1:
                raise ValueError(
                    f"stress source load identities changed: {source_session}"
                )
            load_instance_counts = {
                int(row["load_instance_count"]) for row in observations
            }
            if len(load_instance_counts) != 1:
                raise ValueError(
                    f"stress source multiplicity changed: {source_session}"
                )
            replicate_deltas = [float(row["delta_mean_s"]) for row in observations]
            mean_delta = statistics.fmean(replicate_deltas)
            aggregate_rows.append(
                {
                    "source_session": source_session,
                    "trace_ids": observations[0]["trace_ids"],
                    "load_instances_per_replicate": next(
                        iter(load_instance_counts)
                    ),
                    "a_task_flow_mean_s": statistics.fmean(
                        float(row["a_task_flow_mean_s"]) for row in observations
                    ),
                    "d_task_flow_mean_s": statistics.fmean(
                        float(row["d_task_flow_mean_s"]) for row in observations
                    ),
                    "delta_mean_s": mean_delta,
                    "outcome": _outcome(mean_delta),
                    "replicate_source_mean_delta_s": replicate_deltas,
                }
            )
        mean_deltas = [row["delta_mean_s"] for row in aggregate_rows]
        load_instance_count = sum(
            int(row["load_instances_per_replicate"]) for row in aggregate_rows
        )
        if load_instance_count != int(workload_invariants["load_instance_count"]):
            raise ValueError(
                "stress aggregate load-instance count differs from validated manifest"
            )
        independent_source_count = len(aggregate_rows)
        if independent_source_count != int(
            workload_invariants["independent_source_session_count"]
        ):
            raise ValueError(
                "stress aggregate source count differs from validated manifest"
            )
        if load_instance_count == 120 and independent_source_count == 60:
            definition = (
                "For each of 60 unique heldout sources, first average its original "
                "and deterministic duplicate A-D deltas within each replicate, then "
                "average across replicates; summarize only those 60 source means."
            )
        else:
            exact_instances = workload_invariants["instances_per_source"]
            multiplicity = (
                f"{exact_instances} deterministic load instances"
                if exact_instances is not None
                else "its balanced deterministic load instances"
            )
            definition = (
                f"For each of {independent_source_count} unique heldout sources, "
                f"first average {multiplicity} A-D deltas within each replicate, "
                "then average across replicates; summarize only those source means."
            )
        aggregate = {
            "definition": definition,
            "independent_session_count": len(aggregate_rows),
            "independent_source_session_count": len(aggregate_rows),
            "load_instance_count_per_replicate": load_instance_count,
            "raw_paired_load_observation_count": load_instance_count
            * len(replicates),
            "raw_source_mean_observation_count": len(aggregate_rows)
            * len(replicates),
            "replicate_count": len(replicates),
            "duplicates_are_not_independent": workload_invariants[
                "duplicates_are_not_independent"
            ],
            "repeated_sessions_are_not_counted_as_independent": True,
            "duplicates_and_replicates_do_not_increase_independent_sample_size": True,
            "effective_independent_sample_size": len(aggregate_rows),
            "bootstrap_resampling_unit": "independent_source_session_mean",
            "tie_epsilon_s": TIE_EPSILON_S,
            "outcomes": _outcome_counts(mean_deltas),
            "delta_s": _delta_stats(mean_deltas),
            "independent_source_mean_bootstrap_95_ci_s": _bootstrap_mean_ci(
                mean_deltas
            ),
            "sessions": aggregate_rows,
        }
        if "minimum_instances_per_source" in workload_invariants:
            for field in (
                "instances_per_source",
                "minimum_instances_per_source",
                "maximum_instances_per_source",
                "sources_with_one_extra_instance",
                "source_instances_are_balanced",
            ):
                aggregate[field] = workload_invariants[field]
        return aggregate

    rows_by_replicate = [
        {
            row["trace_id"]: row
            for row in replicate["paired_task_flow"]["sessions"]
        }
        for replicate in replicates
    ]
    trace_ids = sorted(rows_by_replicate[0])
    if any(set(rows) != set(trace_ids) for rows in rows_by_replicate[1:]):
        raise ValueError("replicates do not contain the same paired task identities")

    aggregate_rows = []
    for trace_id in trace_ids:
        observations = [rows[trace_id] for rows in rows_by_replicate]
        source_sessions = {row["source_session"] for row in observations}
        if len(source_sessions) != 1:
            raise ValueError(f"source session changed across replicates: {trace_id}")
        replicate_deltas = [float(row["delta_s"]) for row in observations]
        mean_delta = statistics.fmean(replicate_deltas)
        aggregate_rows.append(
            {
                "trace_id": trace_id,
                "source_session": observations[0]["source_session"],
                "a_task_flow_mean_s": statistics.fmean(
                    float(row["a_task_flow_s"]) for row in observations
                ),
                "d_task_flow_mean_s": statistics.fmean(
                    float(row["d_task_flow_s"]) for row in observations
                ),
                "delta_mean_s": mean_delta,
                "outcome": _outcome(mean_delta),
                "replicate_delta_s": replicate_deltas,
            }
        )
    mean_deltas = [row["delta_mean_s"] for row in aggregate_rows]
    source_sessions = [str(row["source_session"]) for row in aggregate_rows]
    if len(source_sessions) != len(set(source_sessions)):
        raise ValueError(
            "non-stress paired aggregation has duplicate source sessions"
        )
    return {
        "definition": (
            "For each identical evaluation session, average its A-D delta across "
            "replicates; summarize only those per-session means."
        ),
        "independent_session_count": len(aggregate_rows),
        "raw_paired_observation_count": len(aggregate_rows) * len(replicates),
        "replicate_count": len(replicates),
        "repeated_sessions_are_not_counted_as_independent": True,
        "duplicates_and_replicates_do_not_increase_independent_sample_size": True,
        "effective_independent_sample_size": len(aggregate_rows),
        "bootstrap_resampling_unit": "independent_source_session_mean",
        "tie_epsilon_s": TIE_EPSILON_S,
        "outcomes": _outcome_counts(mean_deltas),
        "delta_s": _delta_stats(mean_deltas),
        "independent_source_mean_bootstrap_95_ci_s": _bootstrap_mean_ci(
            mean_deltas
        ),
        "sessions": aggregate_rows,
    }


def summarize_pairs(
    pairs: Sequence[tuple[Path, Path]],
    *,
    manifest_path: Path,
    role: str = "final",
    require_identical_scheduler_config: bool = False,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one A/D pair is required")
    all_paths = [path.resolve() for pair in pairs for path in pair]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("run directories must be unique across all replicate pairs")

    if role not in {"final", "heldout", "stress"}:
        raise ValueError("paired A/D role must be final, heldout, or stress")
    fixed_manifest = load_fixed_manifest(manifest_path, role)
    loaded = [
        load_pair(
            a_path,
            d_path,
            replicate=index,
            bindings=fixed_manifest["bindings"],
            workload_invariants=fixed_manifest,
            require_identical_scheduler_config=(
                require_identical_scheduler_config
            ),
        )
        for index, (a_path, d_path) in enumerate(pairs, 1)
    ]
    reference_identity = loaded[0]["_identity_rows"]
    reference_sources = loaded[0]["_source_mapping"]
    if len(set(reference_sources.values())) != fixed_manifest[
        "independent_source_session_count"
    ]:
        raise ValueError("paired run source-session count differs from fixed manifest")
    for replicate in loaded[1:]:
        if replicate["_identity_rows"] != reference_identity:
            raise ValueError(
                f"replicate {replicate['replicate']} request identity differs from replicate 1"
            )
        if replicate["_source_mapping"] != reference_sources:
            raise ValueError(
                f"replicate {replicate['replicate']} source sessions differ from replicate 1"
            )
        for cell in ("A", "D"):
            for field in (
                "speedup",
                "max_active_traces",
                "tool_wait_mode",
                "configured_max_request_attempts",
                "scheduler_configuration",
            ):
                if replicate["cells"][cell][field] != loaded[0]["cells"][cell][field]:
                    raise ValueError(
                        f"replicate {replicate['replicate']} configuration differs "
                        f"for cell {cell}: {field}"
                    )
    mapper_checksums = {replicate["mapper_artifact_sha256"] for replicate in loaded}
    mapper_top_ks = {replicate["tool_prediction_top_k"] for replicate in loaded}
    if len(mapper_checksums) != 1 or None in mapper_checksums:
        raise ValueError("D replicates do not use one identical mapper artifact checksum")
    if len(mapper_top_ks) != 1 or None in mapper_top_ks:
        raise ValueError("D replicates do not use one identical tool_prediction_top_k")

    public_replicates = []
    for replicate in loaded:
        public_replicates.append(
            {
                key: value
                for key, value in replicate.items()
                if not key.startswith("_")
            }
        )
    aggregate_cells = {
        cell: _aggregate_cell(public_replicates, cell) for cell in ("A", "D")
    }
    aggregate_effect = _pair_effect(aggregate_cells["A"], aggregate_cells["D"])
    completion_token_comparison = _completion_token_comparison(
        aggregate_cells["A"], aggregate_cells["D"]
    )
    paired_sessions = _aggregate_paired_sessions(
        public_replicates,
        role=role,
        workload_invariants=fixed_manifest,
    )
    effect_mean = aggregate_effect["absolute_reduction"]["task_flow_time_s"]["mean"]
    if (
        fixed_manifest["instances_per_source"] is not None
        and not math.isclose(
            effect_mean, paired_sessions["delta_s"]["mean"], abs_tol=1e-9
        )
    ):
        raise AssertionError("aggregate task mean effect disagrees with paired session mean")
    retry_totals = {
        field: sum(
            aggregate_cells[cell]["retry_accounting"][field]
            for cell in ("A", "D")
        )
        for field in (
            "requests_total",
            "request_attempts_total",
            "retry_count",
            "retried_request_count",
            "retry_success_count",
            "ambiguous_retry_count",
            "final_failure_count",
        )
    }
    request_outcome_totals = {
        field: sum(
            int(aggregate_cells[cell]["execution_accounting"]["request_outcomes"][field])
            for cell in ("A", "D")
        )
        for field in ("requests_total", "requests_success", "requests_failed")
    }
    all_requests_finally_succeeded = (
        retry_totals["final_failure_count"] == 0
        and request_outcome_totals["requests_failed"] == 0
        and request_outcome_totals["requests_success"]
        == request_outcome_totals["requests_total"]
    )
    all_requests_succeeded_exactly_once = (
        all_requests_finally_succeeded
        and retry_totals["request_attempts_total"] == retry_totals["requests_total"]
        and retry_totals["retry_count"] == 0
        and retry_totals["ambiguous_retry_count"] == 0
    )

    if role == "stress":
        stress_count = fixed_manifest["load_instance_count"]
        status = (
            f"paired_stress{stress_count}_"
            "ad_load_sensitivity_not_independent_not_final"
        )
        if stress_count == 120:
            role_interpretation = (
                " Stress120 contains one original and one deterministic break-prefix "
                "duplicate for each of 60 heldout sources. The 120 load instances "
                "are not independent; paired inference is summarized over 60 "
                "per-source means, and this is not a final evaluation."
            )
        else:
            source_count = fixed_manifest["independent_source_session_count"]
            exact_instances = fixed_manifest["instances_per_source"]
            if exact_instances is not None:
                multiplicity = (
                    f"one original and {exact_instances - 1} deterministic "
                    "break-prefix duplicates"
                )
            else:
                minimum = fixed_manifest["minimum_instances_per_source"]
                maximum = fixed_manifest["maximum_instances_per_source"]
                multiplicity = f"a balanced {minimum}-{maximum} instances"
            role_interpretation = (
                f" Stress{stress_count} contains {multiplicity} for each of "
                f"{source_count} heldout sources. The {stress_count} load instances "
                "are not independent; paired inference is summarized over "
                f"{source_count} per-source means, and this is not a final evaluation."
            )
    elif role == "heldout":
        status = "paired_heldout_ad_load_sensitivity"
        role_interpretation = (
            " Heldout is the tuning+previously-inspected-final union and is only "
            "load-sensitivity evidence, not a new untouched final set."
        )
    else:
        status = "paired_final_ad_not_full_paper_reproduction"
        role_interpretation = ""
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": status,
        "comparison_invariants": {
            "replicate_count": len(public_replicates),
            "fixed_workload_manifest": repository_display_path(
                fixed_manifest["path"]
            ),
            "fixed_workload_manifest_sha256": fixed_manifest["manifest_sha256"],
            "fixed_split_manifest_sha256": fixed_manifest[
                "fixed_split_manifest_sha256"
            ],
            "fixed_role": role,
            "evidence_role": fixed_manifest["evidence_role"],
            "heldout_parent_manifest_sha256": fixed_manifest[
                "heldout_parent_manifest_sha256"
            ],
            "stress_parent_manifest_sha256": fixed_manifest[
                "stress_parent_manifest_sha256"
            ],
            "trace_count": public_replicates[0]["cells"]["A"]["trace_count"],
            "load_instance_count": fixed_manifest["load_instance_count"],
            "source_session_count": public_replicates[0]["cells"]["A"][
                "source_session_count"
            ],
            "independent_source_session_count": fixed_manifest[
                "independent_source_session_count"
            ],
            "instances_per_source": fixed_manifest["instances_per_source"],
            "duplicates_are_not_independent": fixed_manifest[
                "duplicates_are_not_independent"
            ],
            "is_final_evaluation": fixed_manifest["is_final_evaluation"],
            "calibration_excluded": fixed_manifest["calibration_excluded"],
            "prefix_marker_mode": fixed_manifest["prefix_marker_mode"],
            "request_count": public_replicates[0]["cells"]["A"]["request_count"],
            "request_identity_sha256": public_replicates[0][
                "request_identity_sha256"
            ],
            "source_sessions_sha256": canonical_sha256(
                sorted(set(reference_sources.values()))
            ),
            "mapper_artifact_sha256": next(iter(mapper_checksums)),
            "tool_prediction_top_k": next(iter(mapper_top_ks)),
            "metadata_source": "online",
            "configured_max_request_attempts": public_replicates[0]["cells"]["A"][
                "configured_max_request_attempts"
            ],
            "all_requests_finally_succeeded": all_requests_finally_succeeded,
            "all_requests_succeeded_exactly_once": (
                all_requests_succeeded_exactly_once
            ),
            "retry_accounting": retry_totals,
            "request_outcomes": request_outcome_totals,
            "joint_hook_install_and_runtime_verified": True,
            "positive_delta_means_joint_learned_faster": True,
        },
        "replicates": public_replicates,
        "aggregate": {
            "aggregation": (
                "Cell metrics are means of replicate-level metrics. Relative effects "
                "are computed after aggregation, not averaged as percentages."
            ),
            "cells": aggregate_cells,
            "effects": aggregate_effect,
            "completion_token_comparison": completion_token_comparison,
            "paired_task_flow": paired_sessions,
        },
        "interpretation": (
            "Paired delta is A-D, so positive is faster under joint+learned. "
            "Delta p05 describes the harmed tail and delta p95 the strongly improved "
            "tail; neither is the same as A task-p95 minus D task-p95."
            " The fixed-seed nonparametric bootstrap resamples only independent "
            "source-session means after averaging deterministic duplicates within "
            "replicate and repeated measurements across replicates; duplicates and "
            "replicates do not enlarge the independent sample size."
            + role_interpretation
        ),
    }
    if role == "stress" and "minimum_instances_per_source" in fixed_manifest:
        for field in (
            "minimum_instances_per_source",
            "maximum_instances_per_source",
            "sources_with_one_extra_instance",
            "source_instances_are_balanced",
        ):
            result["comparison_invariants"][field] = fixed_manifest[field]
    if require_identical_scheduler_config:
        result["comparison_invariants"][
            "identical_scheduler_configuration_required"
        ] = True
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated, strictly paired final/heldout/stress A and D runs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("final", "heldout", "stress"),
        default="final",
        help="evaluation role; final remains the default for backward compatibility",
    )
    parser.add_argument(
        "--pair",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("A_RUN", "D_RUN"),
        help="one replicate block: FCFS+none run directory followed by joint+learned",
    )
    parser.add_argument(
        "--require-identical-scheduler-config",
        action="store_true",
        help=(
            "require all recorded engine and scheduler settings (except the "
            "cell-defining policy) to be byte-for-byte identical between A and D; "
            "the default retains tuned-pair compatibility by comparing only shared "
            "settings"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = summarize_pairs(
        [(pair[0], pair[1]) for pair in args.pair],
        manifest_path=args.manifest,
        role=args.role,
        require_identical_scheduler_config=(
            args.require_identical_scheduler_config
        ),
    )
    if args.output is not None:
        write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
