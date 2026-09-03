#!/usr/bin/env python3
"""Replay Pattern-v2 once per source trace and emit request-indexed metrics.

One numerically ordered ``task<N>`` JSONL trace is one request on the x-axis. Prediction
quality is measured from nested whole-session grouped-OOF Pattern-v2 prefixes;
latency compares the risk-limited authority-first policy with demand-only on
the same deterministic synthetic-service replay.  The runner is CPU-only: it
starts no model server and performs no network requests.

Top-1/3/5 are *available-prefix exact-target recall*: a candidate counts only
when the frozen Pattern-v2 gate fires and its exact URL is in the first k.
``runtime_overall_hit_rate`` is deliberately stricter: it counts only targets
that actually reused a completed speculative result or promoted an in-flight
one.  Keeping these columns separate prevents a Top-k oracle from being
reported as a realizable speculative hit rate.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.traces import load_sessions  # noqa: E402
from run_pattern_cache_evaluation import sha256_file  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    PolicySpec,
    ScoredWindow,
    _run_sample,
    _select_candidates,
    aggregate_cell,
    collect_nested_oof_windows,
    policy_specs,
)
from run_pattern_v2_load_robustness import (  # noqa: E402
    canonical_sha256,
    ratio,
)


SCHEMA = "paste_repro.pattern_v2_per_trace.v1"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "results" / "pattern_v2_per_trace"
POLICY_NAME = "utility_global_risk_limited"
TOP_KS = (1, 3, 5)
OVERLAP_SOURCES = frozenset({"reused", "promoted_inflight"})
TASK_NUMBER = re.compile(r"(?:^|_)task(?P<number>[1-9][0-9]*)(?:_|\.jsonl$)")


def ordered_request_sessions(sessions: Sequence[Any]) -> list[dict[str, Any]]:
    """Map ``_task<N>_`` filenames to Request N, with lexical fallback."""

    parsed: list[tuple[int, str, Any]] = []
    fallback: list[tuple[str, Any]] = []
    seen_numbers: set[int] = set()
    for session in sessions:
        session_id = str(session.session_id)
        match = TASK_NUMBER.search(session_id)
        if match is None:
            fallback.append((session_id, session))
            continue
        number = int(match.group("number"))
        if number in seen_numbers:
            raise ValueError(f"duplicate source task number: {number}")
        seen_numbers.add(number)
        parsed.append((number, session_id, session))
    parsed.sort(key=lambda item: (item[0], item[1]))
    fallback.sort(key=lambda item: item[0])
    next_number = max(seen_numbers, default=0) + 1
    ordered: list[dict[str, Any]] = []
    for number, session_id, session in parsed:
        ordered.append(
            {
                "request_number": number,
                "source_task_number": number,
                "trace_id": session_id,
                "session": session,
                "mapping": "parsed_task_number",
            }
        )
    for offset, (session_id, session) in enumerate(fallback):
        ordered.append(
            {
                "request_number": next_number + offset,
                "source_task_number": None,
                "trace_id": session_id,
                "session": session,
                "mapping": "lexical_fallback",
            }
        )
    for order_index, row in enumerate(ordered, 1):
        row["request_order_index"] = order_index
    return ordered


def _trace_quality(windows: Sequence[ScoredWindow]) -> dict[str, Any]:
    """Return exact executable-target Top-k counts for one source trace."""

    targets = sum(len(window.executable_targets) for window in windows)
    visit_windows = sum(bool(window.executable_targets) for window in windows)
    result: dict[str, Any] = {
        "search_decisions": len(windows),
        "visit_windows": visit_windows,
        "authoritative_targets": targets,
    }
    for width in TOP_KS:
        target_hits = 0
        hit_windows = 0
        predictions = 0
        for window in windows:
            selected = window.candidates[:width] if window.v2_gate else ()
            predictions += len(selected)
            selected_urls = {
                candidate.pattern.url for candidate in selected
            }
            hits = len(set(window.executable_targets) & selected_urls)
            target_hits += hits
            hit_windows += int(hits > 0)
        result[f"top{width}_target_hits"] = target_hits if targets else None
        result[f"top{width}_target_recall"] = (
            ratio(target_hits, targets) if targets else None
        )
        result[f"top{width}_hit_visit_windows"] = (
            hit_windows if visit_windows else None
        )
        result[f"top{width}_visit_window_coverage"] = (
            ratio(hit_windows, visit_windows) if visit_windows else None
        )
        result[f"top{width}_requested_candidates"] = predictions
    return result


def _selection_profiles(
    windows_by_session: Mapping[str, Sequence[ScoredWindow]],
    *,
    policy: PolicySpec,
    visit_capacity: int,
    service_ms: float,
    lead_ms: float,
) -> dict[str, dict[str, float | int]]:
    """Recompute deterministic C=1 selection counts for overhead attribution."""

    profiles: dict[str, dict[str, float | int]] = {}
    for session_id, windows in windows_by_session.items():
        totals: Counter[str] = Counter()
        compute_ms = 0.0
        for window in windows:
            _, metadata = _select_candidates(
                [window],
                policy,
                visit_capacity=visit_capacity,
                service_s=service_ms / 1000.0,
                lead_s=lead_ms / 1000.0,
            )
            for field in (
                "predictor_windows_evaluated",
                "probability_candidates_evaluated",
                "selected",
                "selected_hits",
            ):
                totals[field] += int(metadata[field])
            compute_ms += float(metadata["compute_ms"])
        profiles[session_id] = {
            **dict(totals),
            "allocation_weight": compute_ms,
        }
    return profiles


def _repeat_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "positive_repetitions": sum(value > 0.0 for value in values),
        "repetitions": len(values),
    }


def _paired_drained_wall(
    *,
    baseline_samples: Sequence[Mapping[str, Any]],
    policy_samples: Sequence[Mapping[str, Any]],
    feature_ms_per_window: float,
    probability_ms_per_candidate: float,
) -> dict[str, Any]:
    """Summarize paired replay wall time, including charged lookup overhead."""

    if not baseline_samples or len(baseline_samples) != len(policy_samples):
        raise ValueError("replayed trace requires paired timing samples")
    precomputed_repeat_ms = [
        feature_ms_per_window
        * int(sample["predictor_windows_evaluated"])
        + probability_ms_per_candidate
        * int(sample["probability_candidates_evaluated"])
        for sample in policy_samples
    ]
    baseline_wall_ms = [
        1000.0 * float(sample["wall_s"]) for sample in baseline_samples
    ]
    pattern_wall_ms = [
        1000.0 * float(sample["wall_s"]) + precomputed
        for sample, precomputed in zip(policy_samples, precomputed_repeat_ms)
    ]
    benefits = [
        baseline - pattern
        for baseline, pattern in zip(baseline_wall_ms, pattern_wall_ms)
    ]
    speedups = [
        ratio(baseline, pattern)
        for baseline, pattern in zip(baseline_wall_ms, pattern_wall_ms)
    ]
    baseline_mean = statistics.fmean(baseline_wall_ms)
    pattern_mean = statistics.fmean(pattern_wall_ms)
    benefit_mean = baseline_mean - pattern_mean
    return {
        "repetitions": len(policy_samples),
        "precomputed_repeat_ms": precomputed_repeat_ms,
        "baseline_mean_ms": baseline_mean,
        "pattern_mean_ms": pattern_mean,
        "benefit_mean_ms": benefit_mean,
        "speedup_fraction": ratio(benefit_mean, baseline_mean),
        "speedup_ratio": ratio(baseline_mean, pattern_mean),
        "benefit_summary": _repeat_summary(benefits),
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
    }


def build_isolated_request_row(
    *,
    request: Mapping[str, Any],
    windows: Sequence[ScoredWindow],
    cell: Mapping[str, Any] | None,
    baseline_samples: Sequence[Mapping[str, Any]],
    policy_samples: Sequence[Mapping[str, Any]],
    profile: Mapping[str, float | int],
    feature_ms_per_window: float,
    probability_ms_per_candidate: float,
) -> dict[str, Any]:
    """Combine static quality with paired isolated-trace latency evidence."""

    quality = _trace_quality(windows)
    row: dict[str, Any] = {
        "request_number": int(request["request_number"]),
        "request_order_index": int(request["request_order_index"]),
        "source_task_number": request["source_task_number"],
        "request_number_mapping": request["mapping"],
        "trace_id": str(request["trace_id"]),
        **quality,
        "runtime_policy": POLICY_NAME,
    }
    targets = int(quality["authoritative_targets"])
    if not windows:
        row.update(
            {
                "runtime_repetitions": None,
                "runtime_selected_candidates_per_replay": None,
                "runtime_selected_exact_candidates_per_replay": None,
                "runtime_overlap_hits": None,
                "runtime_target_observations": None,
                "runtime_overall_hit_rate": None,
                "runtime_source_counts": None,
                "wrong_speculations_started": None,
                "wrong_speculations_started_per_replay": None,
                "wasted_speculative_service_ms_per_replay": None,
                "physical_call_amplification_vs_demand_only": None,
                "physical_calls_started_per_replay": None,
                "extra_physical_calls_per_replay": None,
                "baseline_authority_wait_ms_mean_per_request": None,
                "pattern_authority_wait_ms_mean_per_request": None,
                "pattern_runtime_overhead_ms_mean_per_request": None,
                "pattern_conservative_latency_ms_mean_per_request": None,
                "conservative_latency_benefit_ms_mean_per_request": None,
                "conservative_authority_wait_speedup_fraction": None,
                "conservative_authority_wait_speedup_ratio": None,
                "authority_wait_benefit_repeat_min_ms": None,
                "authority_wait_benefit_repeat_median_ms": None,
                "authority_wait_benefit_repeat_max_ms": None,
                "baseline_request_critical_path_proxy_ms_mean": None,
                "pattern_request_critical_path_proxy_ms_mean": None,
                "request_critical_path_benefit_ms_mean": None,
                "request_critical_path_speedup_fraction": None,
                "request_critical_path_speedup_ratio": None,
                "request_critical_path_benefit_repeat_min_ms": None,
                "request_critical_path_benefit_repeat_median_ms": None,
                "request_critical_path_benefit_repeat_max_ms": None,
                "request_critical_path_positive_repetitions": None,
                "baseline_latency_ms": None,
                "pattern_conservative_latency_ms": None,
                "conservative_speedup_fraction": None,
                "conservative_speedup_ratio": None,
                "speedup_factor_min": None,
                "speedup_factor_max": None,
                "timing_status": "not_applicable_no_search_decision",
            }
        )
        return row

    wall = _paired_drained_wall(
        baseline_samples=baseline_samples,
        policy_samples=policy_samples,
        feature_ms_per_window=feature_ms_per_window,
        probability_ms_per_candidate=probability_ms_per_candidate,
    )
    repetitions = int(wall["repetitions"])
    precomputed_repeat_ms = wall["precomputed_repeat_ms"]
    critical_summary = wall["benefit_summary"]
    baseline_critical = float(wall["baseline_mean_ms"])
    pattern_critical = float(wall["pattern_mean_ms"])
    critical_benefit = float(wall["benefit_mean_ms"])
    selected = sum(
        int(sample["selection_selected"]) for sample in policy_samples
    )
    selected_hits = sum(
        int(sample["selection_selected_hits"]) for sample in policy_samples
    )
    wrong_started = sum(
        int(sample["wrong_started"]) for sample in policy_samples
    )
    runtime_overhead = statistics.fmean(
        precomputed
        + float(sample["selection_compute_ms"])
        for sample, precomputed in zip(policy_samples, precomputed_repeat_ms)
    )
    physical_calls_per_replay = statistics.fmean(
        float(sample["physical_started"]) for sample in policy_samples
    )
    extra_physical_calls_per_replay = statistics.fmean(
        float(sample["physical_started"])
        - float(sample["authoritative_targets"])
        for sample in policy_samples
    )
    common_wall = {
        "baseline_request_critical_path_proxy_ms_mean": baseline_critical,
        "pattern_request_critical_path_proxy_ms_mean": pattern_critical,
        "request_critical_path_benefit_ms_mean": critical_benefit,
        "request_critical_path_speedup_fraction": wall["speedup_fraction"],
        "request_critical_path_speedup_ratio": wall["speedup_ratio"],
        "request_critical_path_benefit_repeat_min_ms": critical_summary["min"],
        "request_critical_path_benefit_repeat_median_ms": critical_summary[
            "median"
        ],
        "request_critical_path_benefit_repeat_max_ms": critical_summary["max"],
        "request_critical_path_positive_repetitions": critical_summary[
            "positive_repetitions"
        ],
        # Stable aliases consumed by the renderer.  These are drained-wall
        # proxy values, not production request end-to-end latency.
        "baseline_latency_ms": baseline_critical,
        "pattern_conservative_latency_ms": pattern_critical,
        "conservative_speedup_fraction": wall["speedup_fraction"],
        "conservative_speedup_ratio": wall["speedup_ratio"],
        "speedup_factor_min": wall["speedup_min"],
        "speedup_factor_max": wall["speedup_max"],
    }

    if targets == 0:
        row.update(
            {
                "runtime_repetitions": repetitions,
                "runtime_selected_candidates_per_replay": ratio(
                    selected, repetitions
                ),
                "runtime_selected_exact_candidates_per_replay": ratio(
                    selected_hits, repetitions
                ),
                "runtime_overlap_hits": None,
                "runtime_target_observations": None,
                "runtime_overall_hit_rate": None,
                "runtime_source_counts": None,
                "wrong_speculations_started": wrong_started,
                "wrong_speculations_started_per_replay": ratio(
                    wrong_started, repetitions
                ),
                "wasted_speculative_service_ms_per_replay": (
                    statistics.fmean(
                        float(sample["wrong_service_ms"])
                        for sample in policy_samples
                    )
                ),
                "physical_call_amplification_vs_demand_only": None,
                "physical_calls_started_per_replay": physical_calls_per_replay,
                "extra_physical_calls_per_replay": (
                    extra_physical_calls_per_replay
                ),
                "baseline_authority_wait_ms_mean_per_request": None,
                "pattern_authority_wait_ms_mean_per_request": None,
                "pattern_runtime_overhead_ms_mean_per_request": runtime_overhead,
                "pattern_conservative_latency_ms_mean_per_request": None,
                "conservative_latency_benefit_ms_mean_per_request": None,
                "conservative_authority_wait_speedup_fraction": None,
                "conservative_authority_wait_speedup_ratio": None,
                "authority_wait_benefit_repeat_min_ms": None,
                "authority_wait_benefit_repeat_median_ms": None,
                "authority_wait_benefit_repeat_max_ms": None,
                **common_wall,
                "timing_status": (
                    "measured_isolated_trace_drained_wall_proxy_"
                    "no_authoritative_visit"
                ),
            }
        )
        if int(profile["selected"]) * repetitions != selected:
            raise RuntimeError("isolated selection profile did not reconcile")
        return row

    if cell is None:
        raise ValueError("target-bearing trace requires an aggregate cell")
    authority_benefits = [
        float(value)
        for value in cell["repeat_conservative_net_latency_benefit_ms"]
    ]
    authority_summary = _repeat_summary(authority_benefits)
    baseline_authority_total = statistics.fmean(
        float(sample["total_exposed_wait_ms"])
        for sample in baseline_samples
    )
    pattern_authority_total = statistics.fmean(
        float(sample["total_exposed_wait_ms"])
        for sample in policy_samples
    )
    pattern_conservative = pattern_authority_total + runtime_overhead
    authority_benefit = baseline_authority_total - pattern_conservative
    row.update(
        {
            "runtime_repetitions": repetitions,
            "runtime_selected_candidates_per_replay": ratio(
                int(cell["selection_selected"]), repetitions
            ),
            "runtime_selected_exact_candidates_per_replay": ratio(
                int(cell["selection_selected_hits"]), repetitions
            ),
            "runtime_overlap_hits": int(cell["overlap_producing_hits"]),
            "runtime_target_observations": int(cell["authoritative_targets"]),
            "runtime_overall_hit_rate": float(
                cell["overlap_producing_target_coverage"]
            ),
            "runtime_source_counts": cell["source_counts"],
            "wrong_speculations_started": int(
                cell["wrong_speculations_started"]
            ),
            "wrong_speculations_started_per_replay": ratio(
                int(cell["wrong_speculations_started"]), repetitions
            ),
            "wasted_speculative_service_ms_per_replay": ratio(
                float(cell["wasted_speculative_service_ms"]), repetitions
            ),
            "physical_call_amplification_vs_demand_only": float(
                cell["physical_call_amplification_vs_demand_only"]
            ),
            "physical_calls_started_per_replay": physical_calls_per_replay,
            "extra_physical_calls_per_replay": extra_physical_calls_per_replay,
            "baseline_authority_wait_ms_mean_per_request": (
                baseline_authority_total
            ),
            "pattern_authority_wait_ms_mean_per_request": (
                pattern_authority_total
            ),
            "pattern_runtime_overhead_ms_mean_per_request": runtime_overhead,
            "pattern_conservative_latency_ms_mean_per_request": (
                pattern_conservative
            ),
            "conservative_latency_benefit_ms_mean_per_request": (
                authority_benefit
            ),
            "conservative_authority_wait_speedup_fraction": ratio(
                authority_benefit, baseline_authority_total
            ),
            "conservative_authority_wait_speedup_ratio": ratio(
                baseline_authority_total, pattern_conservative
            ),
            "authority_wait_benefit_repeat_min_ms": authority_summary["min"],
            "authority_wait_benefit_repeat_median_ms": authority_summary[
                "median"
            ],
            "authority_wait_benefit_repeat_max_ms": authority_summary["max"],
            **common_wall,
            "timing_status": "measured_isolated_trace_drained_wall_proxy",
        }
    )
    if int(profile["selected"]) * repetitions != int(
        cell["selection_selected"]
    ):
        raise RuntimeError("isolated selection profile did not reconcile")
    return row


def add_cumulative_metrics(rows: Sequence[dict[str, Any]]) -> None:
    target_total = 0
    top_hits = {width: 0 for width in TOP_KS}
    runtime_hits = 0
    runtime_targets = 0
    baseline_critical = 0.0
    pattern_critical = 0.0
    for row in rows:
        targets = int(row["authoritative_targets"])
        target_total += targets
        for width in TOP_KS:
            value = row[f"top{width}_target_hits"]
            if value is not None:
                top_hits[width] += int(value)
            row[f"cumulative_top{width}_target_recall"] = (
                ratio(top_hits[width], target_total) if target_total else None
            )
        if row["runtime_overlap_hits"] is not None:
            runtime_hits += int(row["runtime_overlap_hits"])
            runtime_targets += int(row["runtime_target_observations"])
        if row["baseline_request_critical_path_proxy_ms_mean"] is not None:
            baseline_critical += float(
                row["baseline_request_critical_path_proxy_ms_mean"]
            )
            pattern_critical += float(
                row["pattern_request_critical_path_proxy_ms_mean"]
            )
        row["cumulative_runtime_overall_hit_rate"] = (
            ratio(runtime_hits, runtime_targets) if runtime_targets else None
        )
        row["cumulative_request_critical_path_speedup_fraction"] = (
            ratio(
                baseline_critical - pattern_critical,
                baseline_critical,
            )
            if baseline_critical > 0.0
            else None
        )


def summarize_requests(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [
        row for row in rows if row["timing_status"].startswith("measured")
    ]
    target_rows = [
        row for row in measured if int(row["authoritative_targets"]) > 0
    ]
    targets = sum(int(row["authoritative_targets"]) for row in rows)
    result: dict[str, Any] = {
        "requests": len(rows),
        "requests_with_search_decisions": sum(
            int(row["search_decisions"]) > 0 for row in rows
        ),
        "requests_with_timing": len(measured),
        "requests_without_modeled_tool_path": len(rows) - len(measured),
        "requests_with_authoritative_visit": len(target_rows),
        "authoritative_targets": targets,
    }
    for width in TOP_KS:
        hits = sum(
            int(row[f"top{width}_target_hits"] or 0) for row in rows
        )
        result[f"top{width}_target_hits"] = hits
        result[f"top{width}_target_recall"] = ratio(hits, targets)
    runtime_hits = sum(
        int(row["runtime_overlap_hits"]) for row in target_rows
    )
    runtime_targets = sum(
        int(row["runtime_target_observations"]) for row in target_rows
    )
    baseline_authority = sum(
        float(row["baseline_authority_wait_ms_mean_per_request"])
        for row in target_rows
    )
    pattern_authority = sum(
        float(row["pattern_conservative_latency_ms_mean_per_request"])
        for row in target_rows
    )
    baseline_critical = sum(
        float(row["baseline_request_critical_path_proxy_ms_mean"])
        for row in measured
    )
    pattern_critical = sum(
        float(row["pattern_request_critical_path_proxy_ms_mean"])
        for row in measured
    )
    result.update(
        {
            "runtime_overlap_hits": runtime_hits,
            "runtime_target_observations": runtime_targets,
            "runtime_overall_hit_rate": ratio(runtime_hits, runtime_targets),
            "baseline_authority_wait_ms_mean_replay_total": baseline_authority,
            "pattern_conservative_authority_latency_ms_mean_replay_total": (
                pattern_authority
            ),
            "conservative_authority_wait_speedup_fraction": ratio(
                baseline_authority - pattern_authority, baseline_authority
            ),
            "conservative_authority_wait_speedup_ratio": ratio(
                baseline_authority, pattern_authority
            ),
            "baseline_request_critical_path_proxy_ms_mean_total": (
                baseline_critical
            ),
            "pattern_request_critical_path_proxy_ms_mean_total": pattern_critical,
            "request_critical_path_speedup_fraction": ratio(
                baseline_critical - pattern_critical, baseline_critical
            ),
            "request_critical_path_speedup_ratio": ratio(
                baseline_critical, pattern_critical
            ),
            "requests_with_positive_authority_wait_benefit": sum(
                float(row["conservative_latency_benefit_ms_mean_per_request"])
                > 0.0
                for row in target_rows
            ),
            "requests_with_positive_critical_path_benefit": sum(
                float(row["request_critical_path_benefit_ms_mean"]) > 0.0
                for row in measured
            ),
            "runtime_selected_candidates_mean_replay_total": sum(
                float(row["runtime_selected_candidates_per_replay"] or 0.0)
                for row in measured
            ),
            "wrong_speculations_started_mean_replay_total": sum(
                float(row["wrong_speculations_started_per_replay"] or 0.0)
                for row in measured
            ),
            "wasted_speculative_service_ms_mean_replay_total": sum(
                float(row["wasted_speculative_service_ms_per_replay"] or 0.0)
                for row in measured
            ),
            "extra_physical_calls_mean_replay_total": sum(
                float(row["extra_physical_calls_per_replay"] or 0.0)
                for row in measured
            ),
        }
    )
    return result


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    windows, oof = collect_nested_oof_windows(args.traces)
    requests = ordered_request_sessions(load_sessions(args.traces))
    if args.limit is not None:
        requests = requests[: args.limit]
    session_ids = [str(request["trace_id"]) for request in requests]
    selected_ids = set(session_ids)
    windows = [window for window in windows if window.session_id in selected_ids]
    windows_by_session: dict[str, list[ScoredWindow]] = {
        session_id: [] for session_id in session_ids
    }
    for window in windows:
        windows_by_session[window.session_id].append(window)

    policy = {
        spec.name: spec for spec in policy_specs()
    }[POLICY_NAME]
    profiles = _selection_profiles(
        windows_by_session,
        policy=policy,
        visit_capacity=args.visit_capacity,
        service_ms=args.service_ms,
        lead_ms=args.lead_ms,
    )
    feature_ms = float(oof["runtime_pattern_feature_ms"]["mean"])
    probability_ms = float(
        oof["runtime_probability_lookup_ms"]["mean"]
    )
    request_rows: list[dict[str, Any]] = []
    replayable_request_count = sum(
        bool(windows_by_session[session_id]) for session_id in session_ids
    )
    measured_index = 0
    for request in requests:
        session_id = str(request["trace_id"])
        trace_windows = windows_by_session[session_id]
        target_count = sum(
            len(window.executable_targets) for window in trace_windows
        )
        if not trace_windows:
            request_rows.append(
                build_isolated_request_row(
                    request=request,
                    windows=trace_windows,
                    cell=None,
                    baseline_samples=(),
                    policy_samples=(),
                    profile=profiles[session_id],
                    feature_ms_per_window=feature_ms,
                    probability_ms_per_candidate=probability_ms,
                )
            )
            continue

        measured_index += 1
        print(
            f"request={request['request_number']} measured="
            f"{measured_index}/{replayable_request_count}",
            flush=True,
        )
        baseline_samples: list[dict[str, Any]] = []
        policy_samples: list[dict[str, Any]] = []
        for repetition in range(args.repetitions):
            seed = int(request["request_number"]) * 10_000 + repetition

            async def run_one(spec: PolicySpec | None) -> dict[str, Any]:
                return await _run_sample(
                    trace_windows,
                    policy=spec,
                    offered_concurrency=1,
                    seed=seed,
                    workers=args.workers,
                    visit_capacity=args.visit_capacity,
                    max_speculative_pending=args.max_speculative_pending,
                    service_ms=args.service_ms,
                    lead_ms=args.lead_ms,
                )

            if repetition % 2 == 0:
                baseline = await run_one(None)
                treatment = await run_one(policy)
            else:
                treatment = await run_one(policy)
                baseline = await run_one(None)
            baseline_samples.append(baseline)
            policy_samples.append(treatment)

        cell = (
            aggregate_cell(
                scenario="observed_nested_oof",
                spec=policy,
                offered_concurrency=1,
                baseline_samples=baseline_samples,
                policy_samples=policy_samples,
                feature_runtime_ms_per_window=feature_ms,
                probability_runtime_ms_per_candidate=probability_ms,
                workers=args.workers,
                visit_capacity=args.visit_capacity,
                max_speculative_pending=args.max_speculative_pending,
                service_ms=args.service_ms,
                lead_ms=args.lead_ms,
            )
            if target_count > 0
            else None
        )
        request_rows.append(
            build_isolated_request_row(
                request=request,
                windows=trace_windows,
                cell=cell,
                baseline_samples=baseline_samples,
                policy_samples=policy_samples,
                profile=profiles[session_id],
                feature_ms_per_window=feature_ms,
                probability_ms_per_candidate=probability_ms,
            )
        )

    add_cumulative_metrics(request_rows)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_nested_whole_session_oof",
        "command": shlex.join([sys.executable, *sys.argv]),
        "definitions": {
            "request": (
                "one source JSONL trace; Request N is parsed from _task<N>_ "
                "with lexical fallback only when no task number exists"
            ),
            "top_k": (
                "exact executable-target recall in the frozen gated "
                "Pattern-v2 prefix; availability metric, not dispatch"
            ),
            "overall_hit_rate": (
                "actual completed reuse or in-flight promotion divided by "
                "authoritative target observations"
            ),
            "primary_latency": (
                "isolated trace drained-wall critical-path proxy from replay "
                "start through final broker cleanup; Pattern-v2 feature and "
                "probability lookup time are conservatively added, while "
                "selection/admission are already inside measured wall time"
            ),
            "secondary_latency": (
                "sum of authoritative exposed wait; feature, probability, and "
                "selection time are conservatively charged"
            ),
            "zero_target_request": (
                "Top-k and runtime-hit rates are null because no authoritative "
                "visit provides a denominator; drained-wall speed is measured "
                "when search decisions exist, including necessarily wrong "
                "speculation and predictor overhead"
            ),
            "top_k_speedup_separation": (
                "Top-k is ranking availability only; reported speedup belongs "
                "to the deployed utility_global_risk_limited runtime policy"
            ),
        },
        "configuration": {
            "traces": str(args.traces.resolve()),
            "request_order": "numeric _task<N>_ then lexical fallback",
            "selected_request_count": len(session_ids),
            "repetitions": args.repetitions,
            "paired_order": "AB/BA counterbalanced",
            "offered_concurrency": 1,
            "workers": args.workers,
            "visit_capacity": args.visit_capacity,
            "max_speculative_pending": args.max_speculative_pending,
            "service_ms": args.service_ms,
            "lead_ms": args.lead_ms,
            "policy": POLICY_NAME,
            "neural_model": False,
            "vllm_required": False,
            "network_required": False,
        },
        "summary": summarize_requests(request_rows),
        "nested_oof_runtime": {
            "method": oof["method"],
            "session_count": oof["session_count"],
            "window_count": oof["window_count"],
            "runtime_pattern_feature_ms": oof[
                "runtime_pattern_feature_ms"
            ],
            "runtime_probability_lookup_ms": oof[
                "runtime_probability_lookup_ms"
            ],
        },
        "requests": request_rows,
        "source_sha256": {
            "runner": sha256_file(SCRIPT),
            "adaptive_runner": sha256_file(
                SCRIPT.parent / "run_pattern_v2_adaptive_load.py"
            ),
            "broker": sha256_file(
                REPRODUCTION_ROOT / "paste_repro" / "live_broker.py"
            ),
            "policy": sha256_file(
                REPRODUCTION_ROOT / "paste_repro" / "speculation_policy.py"
            ),
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    request_rows = list(payload["requests"])
    fieldnames = list(request_rows[0]) if request_rows else []
    with (output_dir / "per_request.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: _csv_value(value)
                for key, value in row.items()
            }
            for row in request_rows
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visit-capacity", type=int, default=2)
    parser.add_argument("--max-speculative-pending", type=int, default=64)
    parser.add_argument("--service-ms", type=float, default=5.0)
    parser.add_argument("--lead-ms", type=float, default=2.5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="diagnostic only: replay the first N numerically ordered traces",
    )
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.workers <= 0 or args.visit_capacity <= 0:
        parser.error("worker and visit capacities must be positive")
    if args.visit_capacity > args.workers:
        parser.error("--visit-capacity cannot exceed --workers")
    if args.visit_capacity <= 1:
        parser.error("--visit-capacity must exceed the authority reserve of one")
    if args.max_speculative_pending <= 0:
        parser.error("--max-speculative-pending must be positive")
    if not math.isfinite(args.service_ms) or args.service_ms <= 0.0:
        parser.error("--service-ms must be finite and positive")
    if not math.isfinite(args.lead_ms) or args.lead_ms < 0.0:
        parser.error("--lead-ms must be finite and non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run_experiment(args))
    write_outputs(args.output_dir, payload)
    print(f"wrote {args.output_dir.resolve()}")
    print(f"payload_sha256={payload['payload_sha256']}")


if __name__ == "__main__":
    main()
