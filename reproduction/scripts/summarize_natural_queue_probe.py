#!/usr/bin/env python3
"""Validate one replay cell and summarize evidence of a native vLLM queue.

The probe is deliberately read-only: it reads the immutable cell artifacts and
emits one JSON document on stdout.  A sequence-count limit is called
non-binding only when both the configured workload concurrency upper bound and
all sampled running-request counts are strictly below ``max_num_seqs``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "paste_repro.natural_queue_probe"
VERSION = 1
JOINT_POLICY = "online_joint_pacer_v2"
KV_SWAP_SEMANTICS_V2 = "cpu_swap_only_v2"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"cell is incomplete; missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    try:
        exact = float(value) == number
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or number < 0 or (positive and number == 0):
        qualifier = "a positive integer" if positive else "a non-negative integer"
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _environment_flag(value: Any, label: str) -> bool:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be recorded as a string flag")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} has an invalid boolean flag value")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _unit_interval_arg(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return value


def _nonnegative_float_arg(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"cell is incomplete; missing request events: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"request event line {line_number} is not valid JSON: {path}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(f"request event line {line_number} is not an object")
        events.append(event)
    if not events:
        raise ValueError(f"request event log is empty: {path}")
    return events


def _validate_retry_accounting(
    summary: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    cell_dir: Path,
) -> dict[str, Any]:
    configured = _integer(
        summary.get("configured_max_request_attempts"),
        "configured_max_request_attempts",
        positive=True,
    )
    attempts_total = 0
    retried_requests = 0
    retry_successes = 0
    ambiguous_retries = 0
    final_failures = 0

    for event_number, event in enumerate(events):
        if type(event.get("ok")) is not bool:
            raise ValueError(f"event {event_number} ok must be boolean: {cell_dir}")
        attempts = _integer(
            event.get("attempts"), f"event {event_number} attempts", positive=True
        )
        history = event.get("attempt_history")
        if not isinstance(history, list) or len(history) != attempts:
            raise ValueError(
                f"event {event_number} attempts/history length mismatch: {cell_dir}"
            )
        if attempts > configured:
            raise ValueError(
                f"event {event_number} exceeds configured request attempts: {cell_dir}"
            )
        for attempt_number, record in enumerate(history, 1):
            if not isinstance(record, Mapping) or _integer(
                record.get("attempt"),
                f"event {event_number} attempt index",
                positive=True,
            ) != attempt_number:
                raise ValueError(
                    f"event {event_number} has malformed attempt history: {cell_dir}"
                )
            required = {
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
            if not required.issubset(record):
                raise ValueError(
                    f"event {event_number} has incomplete attempt history: {cell_dir}"
                )
            _finite_nonnegative(
                record.get("duration_s"),
                f"event {event_number} attempt {attempt_number} duration_s",
            )
            backoff = _finite_nonnegative(
                record.get("retry_backoff_s"),
                f"event {event_number} attempt {attempt_number} retry_backoff_s",
            )
            will_retry = _boolean(
                record.get("will_retry"),
                f"event {event_number} attempt {attempt_number} will_retry",
            )
            retryable = _boolean(
                record.get("retryable"),
                f"event {event_number} attempt {attempt_number} retryable",
            )
            ambiguous = _boolean(
                record.get("delivery_ambiguous"),
                f"event {event_number} attempt {attempt_number} delivery_ambiguous",
            )
            is_last = attempt_number == attempts
            if (will_retry and backoff <= 0.0) or (
                not will_retry and backoff != 0.0
            ):
                raise ValueError(
                    f"event {event_number} has inconsistent retry backoff: {cell_dir}"
                )
            if is_last and will_retry:
                raise ValueError(
                    f"event {event_number} retries after its final attempt: {cell_dir}"
                )
            if not is_last and (
                not will_retry
                or not retryable
                or record.get("outcome") != "transport_error"
            ):
                raise ValueError(
                    f"event {event_number} has a non-transport retry: {cell_dir}"
                )
            if not is_last and ambiguous:
                ambiguous_retries += 1

        final = history[-1]
        event_ok = event["ok"]
        history_ok = final.get("outcome") == "success" and final.get(
            "http_status"
        ) == 200
        if event_ok != history_ok or event.get("http_status") != final.get(
            "http_status"
        ):
            raise ValueError(
                f"event {event_number} final outcome/history mismatch: {cell_dir}"
            )
        attempts_total += attempts
        retried_requests += int(attempts > 1)
        retry_successes += int(attempts > 1 and event_ok)
        final_failures += int(not event_ok)

    successes = len(events) - final_failures
    observed = {
        "configured_max_request_attempts": configured,
        "requests_total": len(events),
        "requests_success": successes,
        "requests_failed": final_failures,
        "request_attempts_total": attempts_total,
        "retry_count": attempts_total - len(events),
        "retried_request_count": retried_requests,
        "retry_success_count": retry_successes,
        "ambiguous_retry_count": ambiguous_retries,
        "final_failure_count": final_failures,
    }
    for field, expected in observed.items():
        actual = _integer(summary.get(field), f"summary {field}")
        if actual != expected:
            raise ValueError(
                f"summary/event mismatch for {field}: {actual} != {expected}: "
                f"{cell_dir}"
            )
    observed["all_requests_finally_succeeded"] = final_failures == 0
    observed["all_requests_succeeded_exactly_once"] = (
        final_failures == 0 and observed["retry_count"] == 0
    )
    return observed


def _validate_timeline(
    summary: Mapping[str, Any], timeline: Mapping[str, Any], cell_dir: Path
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    raw_samples = timeline.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"timeline must contain a non-empty samples list: {cell_dir}")
    samples: list[dict[str, float]] = []
    previous_t = -1.0
    for sample_number, raw_sample in enumerate(raw_samples):
        sample = _mapping(raw_sample, f"timeline sample {sample_number}")
        values = {
            "t_s": _finite_nonnegative(
                sample.get("t_s"), f"timeline sample {sample_number} t_s"
            ),
            "running": float(
                _integer(
                    sample.get("running"),
                    f"timeline sample {sample_number} running",
                )
            ),
            "waiting": float(
                _integer(
                    sample.get("waiting"),
                    f"timeline sample {sample_number} waiting",
                )
            ),
            "kv": _finite_nonnegative(
                sample.get("gpu_cache_usage_perc"),
                f"timeline sample {sample_number} gpu_cache_usage_perc",
            ),
        }
        if values["t_s"] < previous_t:
            raise ValueError(f"timeline timestamps are not monotonic: {cell_dir}")
        if values["kv"] > 1.0 + 1e-9:
            raise ValueError(
                f"timeline KV cache usage must be a fraction in [0, 1]: {cell_dir}"
            )
        previous_t = values["t_s"]
        samples.append(values)

    running = [sample["running"] for sample in samples]
    waiting = [sample["waiting"] for sample in samples]
    kv_usage = [sample["kv"] for sample in samples]
    aggregates = {
        "max_running": max(running),
        "mean_running": statistics.fmean(running),
        "max_waiting": max(waiting),
        "mean_waiting": statistics.fmean(waiting),
        "max_kv": max(kv_usage),
        "mean_kv": statistics.fmean(kv_usage),
    }
    timeline_fields = {
        "timeline_max_running": aggregates["max_running"],
        "timeline_avg_running": aggregates["mean_running"],
        "timeline_max_waiting": aggregates["max_waiting"],
        "timeline_avg_waiting": aggregates["mean_waiting"],
        "timeline_max_gpu_cache_usage_perc": aggregates["max_kv"],
        "timeline_avg_gpu_cache_usage_perc": aggregates["mean_kv"],
    }
    for field, recomputed in timeline_fields.items():
        recorded = _finite_nonnegative(summary.get(field), f"summary {field}")
        if not _close(recorded, recomputed):
            raise ValueError(
                f"summary/timeline mismatch for {field}: {recorded} != "
                f"{recomputed}: {cell_dir}"
            )
    return aggregates, samples


def _validate_memory_accounting(
    summary: Mapping[str, Any],
    swap: Mapping[str, Any],
    log_summary: Mapping[str, Any],
    cell_dir: Path,
) -> dict[str, Any]:
    preemptions = _integer(
        summary.get("num_preemptions_total"), "summary num_preemptions_total"
    )
    preemption_metric = summary.get("num_preemptions_metric")
    if not isinstance(preemption_metric, str) or not preemption_metric:
        raise ValueError(f"summary lacks preemption metric provenance: {cell_dir}")
    warning_count = _integer(
        summary.get("preemption_warning_count"),
        "summary preemption_warning_count",
    )
    log_warning_count = _integer(
        log_summary.get("preemption_warning_count"),
        "vllm_log_summary preemption_warning_count",
    )
    if warning_count != log_warning_count:
        raise ValueError(f"preemption warning accounting mismatch: {cell_dir}")
    max_swapped = _integer(
        summary.get("max_swapped_requests"), "summary max_swapped_requests"
    )
    log_max_swapped = _integer(
        log_summary.get("max_swapped_requests"),
        "vllm_log_summary max_swapped_requests",
    )
    if max_swapped != log_max_swapped:
        raise ValueError(f"max swapped-request accounting mismatch: {cell_dir}")

    integer_fields = {
        "kv_swap_event_count": "swap_event_count",
        "kv_swap_in_event_count": "swap_in_event_count",
        "kv_swap_out_event_count": "swap_out_event_count",
        "kv_swap_total_blocks": "swap_total_blocks",
    }
    float_fields = {
        "kv_swap_avg_time_s": "swap_avg_time_s",
        "kv_swap_in_avg_time_s": "swap_in_avg_time_s",
        "kv_swap_out_avg_time_s": "swap_out_avg_time_s",
        "kv_swap_total_time_s": "swap_total_time_s",
    }
    validated: dict[str, Any] = {}
    for summary_field, swap_field in integer_fields.items():
        summary_value = _integer(summary.get(summary_field), f"summary {summary_field}")
        swap_value = _integer(swap.get(swap_field), f"swap_summary {swap_field}")
        if summary_value != swap_value:
            raise ValueError(f"swap accounting mismatch for {summary_field}: {cell_dir}")
        validated[summary_field] = summary_value
    for summary_field, swap_field in float_fields.items():
        summary_value = _finite_nonnegative(
            summary.get(summary_field), f"summary {summary_field}"
        )
        swap_value = _finite_nonnegative(
            swap.get(swap_field), f"swap_summary {swap_field}"
        )
        if not _close(summary_value, swap_value):
            raise ValueError(f"swap accounting mismatch for {summary_field}: {cell_dir}")
        validated[summary_field] = summary_value

    preemption_happened = preemptions > 0
    if "preemption_happened" in summary and _boolean(
        summary["preemption_happened"], "summary preemption_happened"
    ) != preemption_happened:
        raise ValueError(f"preemption_happened accounting mismatch: {cell_dir}")

    recorded_happened = _boolean(
        summary.get("kv_swap_happened"), "summary kv_swap_happened"
    )
    recorded_semantics = summary.get("kv_swap_happened_semantics")
    if recorded_semantics is not None and recorded_semantics != KV_SWAP_SEMANTICS_V2:
        raise ValueError(f"unknown kv_swap_happened semantics: {cell_dir}")
    actual_cpu_swap_happened = bool(
        validated["kv_swap_event_count"] > 0 or max_swapped > 0
    )
    legacy_conflation = bool(
        recorded_semantics is None
        and recorded_happened
        and not actual_cpu_swap_happened
        and preemption_happened
    )
    if recorded_happened != actual_cpu_swap_happened and not legacy_conflation:
        raise ValueError(f"kv_swap_happened accounting mismatch: {cell_dir}")
    return {
        "num_preemptions_total": preemptions,
        "num_preemptions_metric": preemption_metric,
        "preemption_happened": preemption_happened,
        "preemption_warning_count": warning_count,
        "kv_swap_happened": actual_cpu_swap_happened,
        "recorded_kv_swap_happened": recorded_happened,
        "recorded_kv_swap_happened_semantics": recorded_semantics,
        "normalized_kv_swap_happened_semantics": KV_SWAP_SEMANTICS_V2,
        "legacy_preemption_conflated_swap_flag": legacy_conflation,
        **validated,
        "max_swapped_requests": max_swapped,
    }


def summarize_probe(cell_dir: Path) -> dict[str, Any]:
    cell_dir = cell_dir.resolve()
    if not cell_dir.is_dir():
        raise FileNotFoundError(f"cell directory does not exist: {cell_dir}")
    summary = _load_json_object(cell_dir / "summary.json", "summary.json")
    timeline = _load_json_object(cell_dir / "timeline.json", "timeline.json")
    swap = _load_json_object(cell_dir / "swap_summary.json", "swap_summary.json")
    log_summary = _load_json_object(
        cell_dir / "vllm_log_summary.json", "vllm_log_summary.json"
    )
    events = _load_events(cell_dir / "request_events.jsonl")

    request_accounting = _validate_retry_accounting(summary, events, cell_dir)
    mean_request_latency = _finite_nonnegative(
        summary.get("avg_request_latency_s"), "summary avg_request_latency_s"
    )
    event_latencies = [
        _finite_nonnegative(
            event.get("latency_s"), f"event {event_number} latency_s"
        )
        for event_number, event in enumerate(events)
        if event.get("ok") is True
    ]
    if not event_latencies:
        raise ValueError(f"cannot calculate request latency without a success: {cell_dir}")
    recomputed_latency = statistics.fmean(event_latencies)
    if not _close(mean_request_latency, recomputed_latency):
        raise ValueError(
            f"summary/event mean request latency mismatch: {cell_dir}"
        )
    mean_queue_time = _finite_nonnegative(
        summary.get("avg_queue_time_s"), "summary avg_queue_time_s"
    )
    if mean_request_latency == 0.0:
        if mean_queue_time != 0.0:
            raise ValueError(f"nonzero queue time with zero request latency: {cell_dir}")
        queue_share = 0.0
    else:
        queue_share = mean_queue_time / mean_request_latency
    queue_sum_source = summary.get("queue_time_metric_sum")
    queue_count_source = summary.get("queue_time_metric_count")
    if not isinstance(queue_sum_source, str) or not queue_sum_source:
        raise ValueError(f"summary lacks queue-time sum metric provenance: {cell_dir}")
    if not isinstance(queue_count_source, str) or not queue_count_source:
        raise ValueError(f"summary lacks queue-time count metric provenance: {cell_dir}")

    timeline_aggregates, samples = _validate_timeline(summary, timeline, cell_dir)
    memory_accounting = _validate_memory_accounting(
        summary, swap, log_summary, cell_dir
    )
    memory_accounting["preemptions_per_logical_request"] = (
        memory_accounting["num_preemptions_total"]
        / request_accounting["requests_total"]
    )

    environment = _mapping(
        summary.get("scheduler_environment"), "summary scheduler_environment"
    )
    policy = environment.get("VLLM_SCHED_POLICY")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"scheduler policy is missing: {cell_dir}")
    configured_max_num_seqs = _integer(
        environment.get("VLLM_MAX_NUM_SEQS"),
        "scheduler VLLM_MAX_NUM_SEQS",
        positive=True,
    )
    configured_max_num_batched_tokens = _integer(
        environment.get("VLLM_MAX_NUM_BATCHED_TOKENS"),
        "scheduler VLLM_MAX_NUM_BATCHED_TOKENS",
        positive=True,
    )
    native_admission_flag = _environment_flag(
        environment.get("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION"),
        "scheduler VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
    )
    workload = _mapping(summary.get("workload"), "summary workload")
    trace_count = _integer(
        workload.get("trace_count"), "summary workload trace_count", positive=True
    )
    workload_request_count = _integer(
        workload.get("request_count"),
        "summary workload request_count",
        positive=True,
    )
    if workload_request_count != request_accounting["requests_total"]:
        raise ValueError(f"workload/request event count mismatch: {cell_dir}")
    max_active_traces = _integer(
        summary.get("max_active_traces"), "summary max_active_traces", positive=True
    )
    offered_concurrency_upper_bound = min(trace_count, max_active_traces)
    if timeline_aggregates["max_running"] > offered_concurrency_upper_bound:
        raise ValueError(
            "timeline running requests exceed the replay concurrency upper bound: "
            f"{cell_dir}"
        )

    waiting_samples = [sample for sample in samples if sample["waiting"] > 0.0]
    natural_queue_samples = [
        sample
        for sample in samples
        if sample["waiting"] > 0.0
        and sample["running"] < configured_max_num_seqs
    ]
    cap_reached_samples = [
        sample for sample in samples if sample["running"] >= configured_max_num_seqs
    ]
    structural_nonbinding = (
        offered_concurrency_upper_bound < configured_max_num_seqs
    )
    observed_nonbinding = not cap_reached_samples
    sequence_cap_nonbinding = structural_nonbinding and observed_nonbinding
    scheduler_admission_is_native = policy == "fcfs" or (
        policy == JOINT_POLICY and native_admission_flag
    )
    native_vllm_queue_proven = bool(
        sequence_cap_nonbinding
        and scheduler_admission_is_native
        and natural_queue_samples
    )
    if native_vllm_queue_proven:
        conclusion = "native_vllm_resource_queue_observed_with_nonbinding_sequence_cap"
    elif not sequence_cap_nonbinding:
        conclusion = "sequence_count_cap_may_bind"
    elif not scheduler_admission_is_native:
        conclusion = "native_admission_not_established"
    else:
        conclusion = "no_waiting_below_sequence_cap_observed"

    sample_count = len(samples)
    waiting_count = len(waiting_samples)
    natural_count = len(natural_queue_samples)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "cell_dir": cell_dir.as_posix(),
        "request_accounting": request_accounting,
        "serving_memory_accounting": memory_accounting,
        "queueing": {
            "mean_request_latency_s": mean_request_latency,
            "mean_queue_time_s": mean_queue_time,
            "queue_time_fraction_of_request_latency": queue_share,
            "queue_time_metric_sum": queue_sum_source,
            "queue_time_metric_count": queue_count_source,
        },
        "timeline": {
            "sample_count": sample_count,
            "running_requests": {
                "mean": timeline_aggregates["mean_running"],
                "max": timeline_aggregates["max_running"],
            },
            "waiting_requests": {
                "mean": timeline_aggregates["mean_waiting"],
                "max": timeline_aggregates["max_waiting"],
            },
            "kv_cache_usage": {
                "unit": "fraction_of_vllm_gpu_kv_capacity",
                "source_field": "gpu_cache_usage_perc",
                "mean": timeline_aggregates["mean_kv"],
                "max": timeline_aggregates["max_kv"],
            },
            "waiting_sample_count": waiting_count,
            "waiting_sample_fraction": waiting_count / sample_count,
            "waiting_below_sequence_cap_sample_count": natural_count,
            "waiting_below_sequence_cap_sample_fraction": natural_count
            / sample_count,
            "waiting_below_sequence_cap_fraction_of_waiting_samples": (
                natural_count / waiting_count if waiting_count else 0.0
            ),
            "sequence_cap_reached_sample_count": len(cap_reached_samples),
        },
        "sequence_capacity": {
            "scheduler_policy": policy,
            "configured_max_num_seqs": configured_max_num_seqs,
            "configured_max_num_batched_tokens": configured_max_num_batched_tokens,
            "workload_trace_count": trace_count,
            "configured_max_active_traces": max_active_traces,
            "offered_concurrency_upper_bound": offered_concurrency_upper_bound,
            "configuration_sequence_headroom": configured_max_num_seqs
            - offered_concurrency_upper_bound,
            "observed_max_running": timeline_aggregates["max_running"],
            "observed_sequence_headroom": configured_max_num_seqs
            - timeline_aggregates["max_running"],
            "nonbinding_by_configuration": structural_nonbinding,
            "nonbinding_in_timeline_samples": observed_nonbinding,
            "sequence_cap_nonbinding": sequence_cap_nonbinding,
            "joint_native_admission_flag": native_admission_flag,
            "scheduler_admission_is_native": scheduler_admission_is_native,
            "natural_vllm_queue_proven": native_vllm_queue_proven,
            "conclusion": conclusion,
            "attribution_limit": (
                "This establishes a native resource queue, not whether token-budget "
                "or physical KV availability was the dominant cause."
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one replay cell, validate its accounting, and summarize "
            "whether waiting occurred below a non-binding sequence cap."
        )
    )
    parser.add_argument("cell_dir", type=Path)
    parser.add_argument(
        "--require-natural-queue",
        action="store_true",
        help="fail unless the artifacts prove a native queue below a non-binding cap",
    )
    parser.add_argument(
        "--require-exactly-once",
        action="store_true",
        help="fail if any logical request retried or finally failed",
    )
    parser.add_argument(
        "--require-no-kv-swap",
        action="store_true",
        help="fail if vLLM performed any CPU KV-cache swap",
    )
    parser.add_argument(
        "--min-waiting-below-cap-sample-fraction",
        type=_unit_interval_arg,
        help="minimum fraction of timeline samples with waiting>0 and running<cap",
    )
    parser.add_argument(
        "--min-queue-time-fraction",
        type=_unit_interval_arg,
        help="minimum mean queue-time / mean request-latency fraction",
    )
    parser.add_argument(
        "--max-preemptions-per-request",
        type=_nonnegative_float_arg,
        help="maximum native recomputation-preemptions / logical requests",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = summarize_probe(args.cell_dir)
    if args.require_natural_queue and not result["sequence_capacity"][
        "natural_vllm_queue_proven"
    ]:
        conclusion = result["sequence_capacity"]["conclusion"]
        raise ValueError(f"natural vLLM queue requirement failed: {conclusion}")
    if args.require_exactly_once and not result["request_accounting"][
        "all_requests_succeeded_exactly_once"
    ]:
        raise ValueError("exactly-once request requirement failed")
    if args.require_no_kv_swap and result["serving_memory_accounting"][
        "kv_swap_happened"
    ]:
        raise ValueError("no-CPU-KV-swap requirement failed")
    waiting_fraction = result["timeline"][
        "waiting_below_sequence_cap_sample_fraction"
    ]
    if (
        args.min_waiting_below_cap_sample_fraction is not None
        and waiting_fraction + 1e-12
        < args.min_waiting_below_cap_sample_fraction
    ):
        raise ValueError(
            "waiting-below-cap sample fraction requirement failed: "
            f"{waiting_fraction} < {args.min_waiting_below_cap_sample_fraction}"
        )
    queue_fraction = result["queueing"][
        "queue_time_fraction_of_request_latency"
    ]
    if (
        args.min_queue_time_fraction is not None
        and queue_fraction + 1e-12 < args.min_queue_time_fraction
    ):
        raise ValueError(
            "queue-time fraction requirement failed: "
            f"{queue_fraction} < {args.min_queue_time_fraction}"
        )
    preemption_rate = result["serving_memory_accounting"][
        "preemptions_per_logical_request"
    ]
    if (
        args.max_preemptions_per_request is not None
        and preemption_rate
        > args.max_preemptions_per_request + 1e-12
    ):
        raise ValueError(
            "preemptions-per-request requirement failed: "
            f"{preemption_rate} > {args.max_preemptions_per_request}"
        )
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
