#!/usr/bin/env python3
"""CPU-only robustness evaluation for the frozen Pattern-v2 URL predictor.

The experiment has two evidence tiers and deliberately starts no model server:

1. Whole-session grouped-OOF trace replay measures exact-URL recall, decision
   coverage, candidate precision, and logical waste for every runtime prefix
   K=1..5.  It also reports the bounded-cache oracle separately; the oracle is
   not a runtime dispatch policy.
2. A paired synthetic-service replay invokes :class:`LiveToolBroker`, where
   authoritative and speculative visits share the same bounded worker pool.
   It measures realized reuse, capacity rejection, physically started waste,
   wasted service, and *unclamped* net authoritative latency benefit as offered
   concurrency increases.  A deterministic all-wrong counterfactual preserves
   Pattern-v2's firing decisions and candidate counts while making every exact
   prediction miss.

Network, vLLM, embeddings, and neural inference are outside this experiment.
Synthetic milliseconds characterize queue behavior, not production latency.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.invocation import Invocation  # noqa: E402
from paste_repro.live_broker import LiveToolBroker  # noqa: E402
from paste_repro.pattern_predictor import (  # noqa: E402
    FROZEN_TOP_K,
    PATTERN_ARTIFACT_VERSION,
    PATTERN_POLICY_VERSION,
    load_pattern_artifact,
)
from paste_repro.traces import load_sessions  # noqa: E402
from run_pattern_cache_evaluation import (  # noqa: E402
    CV_SEED,
    cv_fold,
    extract_search_decisions,
    fit_rank_pattern,
    make_frozen_predictor,
    score_decisions,
    sha256_file,
    trace_manifest,
)


SCHEMA = "paste_repro.pattern_v2_load_robustness.v1"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_ARTIFACT = (
    REPRODUCTION_ROOT
    / "results"
    / "pattern_cache_development"
    / "pattern_cache_policy.json"
)
DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT / "results" / "pattern_v2_load_robustness"
)
DEFAULT_WIDTHS = (1, 2, 3, 4, 5)
DEFAULT_CONCURRENCIES = (1, 8, 32, 64, 128)


@dataclass(frozen=True)
class ReplayOpportunity:
    decision_id: str
    predictions: tuple[str, ...]
    executable_targets: tuple[str, ...]


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def stable_order(seed: int, value: str) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def executable_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def collect_pattern_v2_oof_rows(
    traces: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute five-fold whole-session OOF Pattern-v2 decisions."""

    sessions = load_sessions(traces)
    decisions = extract_search_decisions(sessions)
    rows: list[dict[str, Any]] = []
    predictor_durations_ms: list[float] = []
    folds: list[dict[str, Any]] = []
    for fold in range(5):
        fit_ids = {
            session.session_id
            for session in sessions
            if cv_fold(session.session_id) != fold
        }
        validation_ids = {
            session.session_id
            for session in sessions
            if cv_fold(session.session_id) == fold
        }
        fit_decisions = [
            decision for decision in decisions if decision.session_id in fit_ids
        ]
        validation = [
            decision
            for decision in decisions
            if decision.session_id in validation_ids
        ]
        pattern = fit_rank_pattern(fit_decisions)
        predictor = make_frozen_predictor(pattern)
        fold_rows, durations = score_decisions(validation, predictor)
        rows.extend(fold_rows)
        predictor_durations_ms.extend(
            float(value) for value in durations["pattern_cache"]
        )
        folds.append(
            {
                "fold": fold,
                "fit_sessions": len(fit_ids),
                "validation_sessions": len(validation_ids),
                "fit_decisions": len(fit_decisions),
                "validation_decisions": len(validation),
                "rank_counts": {
                    str(rank): count
                    for rank, count in pattern.rank_counts.items()
                },
            }
        )
    if len(rows) != len(decisions):
        raise RuntimeError("grouped OOF did not score every search decision")
    if len({str(row["decision_id"]) for row in rows}) != len(rows):
        raise RuntimeError("grouped OOF produced duplicate decision identifiers")
    metadata = {
        "grouping_unit": "whole session",
        "fold_seed": CV_SEED,
        "folds": folds,
        "session_count": len(sessions),
        "search_decisions": len(rows),
        "predictor_compute": {
            "scope": "gate + bounded cache update + exact-URL Top-5 ranking",
            "measured_calls": len(predictor_durations_ms),
            "total_ms_per_full_replay": sum(predictor_durations_ms),
            "mean_ms": (
                statistics.fmean(predictor_durations_ms)
                if predictor_durations_ms
                else 0.0
            ),
            "p50_ms": percentile(predictor_durations_ms, 0.50),
            "p95_ms": percentile(predictor_durations_ms, 0.95),
            "p99_ms": percentile(predictor_durations_ms, 0.99),
            "max_ms": max(predictor_durations_ms, default=0.0),
        },
        "trace_manifest": trace_manifest(sessions),
    }
    return rows, metadata


def static_width_metrics(
    rows: Sequence[Mapping[str, Any]], widths: Sequence[int]
) -> list[dict[str, Any]]:
    """Measure runtime-prefix recall and all-window logical waste."""

    visit_rows = [row for row in rows if row["outcome"] == "visit"]
    targets = sum(int(row["target_count"]) for row in visit_rows)
    result: list[dict[str, Any]] = []
    for width in widths:
        if not 1 <= width <= FROZEN_TOP_K:
            raise ValueError(
                f"Pattern-v2 runtime prefixes must be in [1,{FROZEN_TOP_K}]"
            )
        requested = 0
        useful = 0
        nonvisit_requested = 0
        hit_windows = 0
        for row in rows:
            selected = tuple(row["pattern_gated_predictions"][:width])
            requested += len(selected)
            if row["outcome"] != "visit":
                nonvisit_requested += len(selected)
                continue
            target_urls = tuple(str(value) for value in row["targets"])
            selected_set = set(selected)
            hits = sum(url in selected_set for url in target_urls)
            useful += hits
            hit_windows += hits > 0
        logical_waste = requested - useful
        physical_calls_if_every_prediction_completes = requested + targets - useful
        result.append(
            {
                "scope": "all100_whole_session_grouped_oof",
                "width": width,
                "runtime_prefix": True,
                "search_decisions": len(rows),
                "visit_windows": len(visit_rows),
                "authoritative_targets": targets,
                "target_hits": useful,
                "exact_target_recall": ratio(useful, targets),
                "hit_visit_windows": hit_windows,
                "visit_window_coverage": ratio(hit_windows, len(visit_rows)),
                "requested_candidates": requested,
                "useful_candidates": useful,
                "candidate_precision": ratio(useful, requested),
                "logical_waste_candidates": logical_waste,
                "logical_waste_fraction": ratio(logical_waste, requested),
                "nonvisit_window_candidates": nonvisit_requested,
                "physical_calls_if_every_prediction_completes": (
                    physical_calls_if_every_prediction_completes
                ),
                "worst_complete_call_amplification_vs_demand_only": ratio(
                    physical_calls_if_every_prediction_completes, targets
                ),
            }
        )
    return result


def bounded_pool_oracle_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report candidate-union coverage without pretending it is dispatched."""

    visit_rows = [row for row in rows if row["outcome"] == "visit"]
    targets = sum(int(row["target_count"]) for row in visit_rows)
    covered = sum(int(row["gated_cache_covered_targets"]) for row in visit_rows)
    hit_windows = sum(
        int(row["gated_cache_covered_targets"]) > 0 for row in visit_rows
    )
    candidates = sum(
        int(row["cache_candidate_count"])
        for row in rows
        if bool(row["gate_admitted"])
    )
    return {
        "evaluation_only": True,
        "runtime_dispatch": False,
        "description": (
            "all exact URLs in the unbounded current response plus admitted "
            "LRU64 historical URLs with search age<=2; no Top-k truncation"
        ),
        "authoritative_targets": targets,
        "covered_targets": covered,
        "target_coverage": ratio(covered, targets),
        "visit_windows": len(visit_rows),
        "covered_visit_windows": hit_windows,
        "visit_window_coverage": ratio(hit_windows, len(visit_rows)),
        "candidate_count_if_all_fired": candidates,
        "candidate_precision_if_all_fired": ratio(covered, candidates),
        "logical_waste_if_all_fired": candidates - covered,
        "mean_candidates_per_admitted_window": ratio(
            candidates, sum(bool(row["gate_admitted"]) for row in rows)
        ),
    }


def build_replay_opportunities(
    rows: Sequence[Mapping[str, Any]],
) -> list[ReplayOpportunity]:
    opportunities: list[ReplayOpportunity] = []
    for row in rows:
        targets = (
            tuple(
                str(url)
                for url in row["targets"]
                if executable_url(str(url))
            )
            if row["outcome"] == "visit"
            else ()
        )
        opportunities.append(
            ReplayOpportunity(
                decision_id=str(row["decision_id"]),
                predictions=tuple(
                    str(url) for url in row["pattern_gated_predictions"]
                ),
                executable_targets=targets,
            )
        )
    return opportunities


def force_all_wrong(
    opportunities: Sequence[ReplayOpportunity],
) -> list[ReplayOpportunity]:
    """Preserve candidate firing but replace every executable target URL."""

    result: list[ReplayOpportunity] = []
    for opportunity in opportunities:
        targets = tuple(
            "https://all-wrong.invalid/"
            + hashlib.sha256(
                f"{opportunity.decision_id}\0{index}\0{url}".encode("utf-8")
            ).hexdigest()
            for index, url in enumerate(opportunity.executable_targets)
        )
        if set(targets).intersection(opportunity.predictions):
            raise RuntimeError("all-wrong target unexpectedly matched a prediction")
        result.append(
            ReplayOpportunity(
                opportunity.decision_id,
                opportunity.predictions,
                targets,
            )
        )
    return result


async def _run_broker_workload(
    opportunities: Sequence[ReplayOpportunity],
    *,
    width: int,
    offered_concurrency: int,
    seed: int,
    speculate: bool,
    workers: int,
    speculative_workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
) -> dict[str, Any]:
    """Replay one ordering through the real shared-capacity broker."""

    async def executor(invocation: Invocation) -> dict[str, Any]:
        # Mirror the production SyncToolMapExecutor contract: cancelling an
        # asyncio wrapper cannot stop an already-running blocking HTTP call,
        # and the broker must retain its physical slot until that call drains.
        physical = asyncio.create_task(asyncio.sleep(service_ms / 1000.0))
        try:
            await asyncio.shield(physical)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(physical)
            except Exception:
                pass
            raise
        return {"invocation_key": invocation.key}

    broker = LiveToolBroker(
        executor,
        max_workers=workers,
        max_speculative_workers=(speculative_workers if speculate else 0),
        min_speculative_workers=0,
        max_speculative_pending=max_speculative_pending,
        ttl_s=60.0,
        tool_capacities={"visit": visit_capacity},
    )
    ordered = sorted(
        opportunities,
        key=lambda item: (stable_order(seed, item.decision_id), item.decision_id),
    )
    requested_predictions = 0
    authoritative_results = []
    admission_submission_ms: list[float] = []
    confirmation_offset_ms: list[float] = []
    decision_deadline_overruns = 0
    wall_started = time.perf_counter()
    try:
        for batch_start in range(0, len(ordered), offered_concurrency):
            batch = ordered[batch_start : batch_start + offered_concurrency]
            session_ids = [
                f"r{seed}:b{batch_start}:i{index}:{item.decision_id}"
                for index, item in enumerate(batch)
            ]
            decision_started = time.perf_counter()
            if speculate:
                # Rank-major submission prevents lower ranks from consuming the
                # pending cap before every opportunity's higher rank.
                for rank in range(width):
                    for item, session_id in zip(batch, session_ids):
                        if rank >= len(item.predictions):
                            continue
                        requested_predictions += 1
                        await broker.speculate(
                            Invocation("visit", {"url": item.predictions[rank]}),
                            session_id=session_id,
                            priority=1.0 / (rank + 1),
                        )
            submitted_at = time.perf_counter()
            submission_ms = (submitted_at - decision_started) * 1000.0
            admission_submission_ms.append(submission_ms)
            remaining_lead_s = lead_ms / 1000.0 - (
                submitted_at - decision_started
            )
            if remaining_lead_s > 0:
                await asyncio.sleep(remaining_lead_s)
            elif speculate:
                decision_deadline_overruns += 1
            confirmation_offset_ms.append(
                (time.perf_counter() - decision_started) * 1000.0
            )

            calls = [
                asyncio.create_task(
                    broker.authoritative(
                        Invocation("visit", {"url": target}),
                        session_id=session_id,
                    )
                )
                for item, session_id in zip(batch, session_ids)
                for target in item.executable_targets
            ]
            # For zero/one-target windows the broker's single ``keep`` key lets
            # us cancel wrong siblings concurrently with authoritative work
            # without racing the correct prediction. Multi-target windows are
            # cleaned immediately after all their exact claims complete.
            early_cleanup = [
                asyncio.create_task(
                    broker.cancel_predictions(
                        session_id=session_id,
                        keep=(
                            Invocation("visit", {"url": item.executable_targets[0]})
                            if len(item.executable_targets) == 1
                            else None
                        ),
                    )
                )
                for item, session_id in zip(batch, session_ids)
                if len(item.executable_targets) <= 1
            ]
            if calls:
                authoritative_results.extend(await asyncio.gather(*calls))
            if early_cleanup:
                await asyncio.gather(*early_cleanup)
            late_cleanup = [
                asyncio.create_task(
                    broker.cancel_predictions(session_id=session_id)
                )
                for item, session_id in zip(batch, session_ids)
                if len(item.executable_targets) > 1
            ]
            if late_cleanup:
                await asyncio.gather(*late_cleanup)
            await broker.cancel_predictions()
        pending_before_close = broker.pending_speculative_count
        snapshot = broker.snapshot()
        records = broker.tool_records()
        stats = broker.stats.to_dict()
    finally:
        await broker.close()
    wall_s = time.perf_counter() - wall_started

    exposed_ms = [result.exposed_wait_s * 1000.0 for result in authoritative_results]
    sources = Counter(result.source for result in authoritative_results)
    speculative_records = [
        record
        for record in records
        if record.get("speculative") is True and record.get("admitted") is True
    ]
    useful_records = [
        record for record in speculative_records if record.get("committed") is True
    ]
    wrong_records = [
        record for record in speculative_records if record.get("committed") is not True
    ]
    wrong_started = sum(record.get("started_at") is not None for record in wrong_records)
    wrong_never_started = len(wrong_records) - wrong_started
    started_records = [
        record
        for record in records
        if record.get("started_at") is not None and record.get("admitted") is True
    ]
    physical_started = len(started_records)
    observed_service_ms = [
        1000.0 * float(record["service_s"])
        for record in started_records
        if isinstance(record.get("service_s"), (int, float))
    ]
    wrong_started_records = [
        record for record in wrong_records if record.get("started_at") is not None
    ]
    wrong_service_ms = [
        1000.0 * float(record["service_s"])
        for record in wrong_started_records
        if isinstance(record.get("service_s"), (int, float))
    ]
    stats_wasted_service_ms = (
        float(stats["wasted_speculative_service_s"]) * 1000.0
    )
    exact_hits = len(useful_records)
    target_count = len(authoritative_results)
    requested_identity_ok = (
        not speculate
        or requested_predictions
        == int(stats["speculative_admitted"])
        + int(stats["rejected_speculative_capacity"])
        + int(stats["duplicate_predictions"])
    )
    safety = {
        "requested_identity_ok": requested_identity_ok,
        "authoritative_commits_equal_targets": (
            int(stats["commits"]) == target_count
        ),
        "no_pending_predictions_at_end": pending_before_close == 0,
        "authoritative_state_equal_targets": (
            len(broker.authoritative_state) == target_count
        ),
        "max_running_total_within_cap": (
            int(stats["max_running_total"]) <= workers
        ),
        "max_running_speculative_within_cap": (
            int(stats["max_running_speculative"])
            <= (speculative_workers if speculate else 0)
        ),
        "max_running_visit_within_cap": (
            int(stats["max_running_by_tool"].get("visit", 0)) <= visit_capacity
        ),
        "snapshot_pending_zero": (
            int(snapshot["counts"]["queued_speculative"])
            + int(snapshot["counts"]["running_speculative"])
            + int(snapshot["counts"]["completed_unclaimed_speculative"])
            == 0
        ),
        "snapshot_authoritative_zero": (
            int(snapshot["counts"]["queued_authoritative"])
            + int(snapshot["counts"]["running_authoritative"])
            == 0
        ),
        "snapshot_has_no_jobs": len(snapshot["jobs"]) == 0,
        "every_started_record_has_service_duration": (
            len(observed_service_ms) == physical_started
        ),
        "every_wrong_started_record_has_service_duration": (
            len(wrong_service_ms) == wrong_started
        ),
        "recorded_waste_matches_broker_stats": math.isclose(
            sum(wrong_service_ms),
            stats_wasted_service_ms,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ),
    }
    if not all(safety.values()):
        raise RuntimeError(f"broker safety invariant failed: {safety}")

    return {
        "requested_predictions": requested_predictions,
        "admitted_predictions": int(stats["speculative_admitted"]),
        "rejected_predictions": int(stats["rejected_speculative_capacity"]),
        "duplicate_predictions": int(stats["duplicate_predictions"]),
        "admission_ratio": ratio(
            int(stats["speculative_admitted"]), requested_predictions
        ),
        "authoritative_targets": target_count,
        "exact_hits": exact_hits,
        "realized_exact_target_coverage": ratio(exact_hits, target_count),
        "source_counts": dict(sorted(sources.items())),
        "logical_waste_admitted": len(wrong_records),
        "wrong_speculations_started": wrong_started,
        "wrong_speculations_never_started": wrong_never_started,
        "wasted_speculative_service_ms": stats_wasted_service_ms,
        "observed_service_ms": {
            "requested_sleep_ms": service_ms,
            "started_calls": len(observed_service_ms),
            "mean": (
                statistics.fmean(observed_service_ms)
                if observed_service_ms
                else 0.0
            ),
            "p95": percentile(observed_service_ms, 0.95),
            "p99": percentile(observed_service_ms, 0.99),
            "max": max(observed_service_ms, default=0.0),
        },
        "observed_wrong_service_ms": {
            "started_calls": len(wrong_service_ms),
            "mean": statistics.fmean(wrong_service_ms) if wrong_service_ms else 0.0,
            "p95": percentile(wrong_service_ms, 0.95),
            "p99": percentile(wrong_service_ms, 0.99),
            "max": max(wrong_service_ms, default=0.0),
        },
        "saved_speculative_service_ms": float(stats["saved_service_s"]) * 1000.0,
        "physical_calls_started": physical_started,
        "physical_call_amplification_vs_demand_only": ratio(
            physical_started, target_count
        ),
        "mean_exposed_wait_ms": (
            statistics.fmean(exposed_ms) if exposed_ms else 0.0
        ),
        "p95_exposed_wait_ms": percentile(exposed_ms, 0.95),
        "p99_exposed_wait_ms": percentile(exposed_ms, 0.99),
        "total_exposed_wait_ms": sum(exposed_ms),
        "wall_s": wall_s,
        "admission_submission_ms": {
            "batches": len(admission_submission_ms),
            "mean": (
                statistics.fmean(admission_submission_ms)
                if admission_submission_ms
                else 0.0
            ),
            "p95": percentile(admission_submission_ms, 0.95),
            "max": max(admission_submission_ms, default=0.0),
            "decision_deadline_overrun_batches": decision_deadline_overruns,
        },
        "confirmation_offset_ms_from_batch_start": {
            "batches": len(confirmation_offset_ms),
            "requested_deadline_ms": lead_ms,
            "mean": (
                statistics.fmean(confirmation_offset_ms)
                if confirmation_offset_ms
                else 0.0
            ),
            "p95": percentile(confirmation_offset_ms, 0.95),
            "max": max(confirmation_offset_ms, default=0.0),
        },
        "max_queued_authoritative": int(stats["max_queued_authoritative"]),
        "max_queued_speculative": int(stats["max_queued_speculative"]),
        "max_running_total": int(stats["max_running_total"]),
        "max_running_speculative": int(stats["max_running_speculative"]),
        "visit_capacity": visit_capacity,
        "safety": safety,
    }


def _sum_sample_field(samples: Sequence[Mapping[str, Any]], name: str) -> float:
    return sum(float(sample[name]) for sample in samples)


async def paired_broker_cell(
    opportunities: Sequence[ReplayOpportunity],
    *,
    scenario: str,
    width: int,
    offered_concurrency: int,
    repetitions: int,
    workers: int,
    speculative_workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
) -> dict[str, Any]:
    baseline_samples: list[dict[str, Any]] = []
    pattern_samples: list[dict[str, Any]] = []
    per_repeat_net_benefit: list[float] = []
    for repetition in range(repetitions):
        baseline = await _run_broker_workload(
            opportunities,
            width=width,
            offered_concurrency=offered_concurrency,
            seed=repetition,
            speculate=False,
            workers=workers,
            speculative_workers=speculative_workers,
            visit_capacity=visit_capacity,
            max_speculative_pending=max_speculative_pending,
            service_ms=service_ms,
            lead_ms=lead_ms,
        )
        pattern = await _run_broker_workload(
            opportunities,
            width=width,
            offered_concurrency=offered_concurrency,
            seed=repetition,
            speculate=True,
            workers=workers,
            speculative_workers=speculative_workers,
            visit_capacity=visit_capacity,
            max_speculative_pending=max_speculative_pending,
            service_ms=service_ms,
            lead_ms=lead_ms,
        )
        if baseline["authoritative_targets"] != pattern["authoritative_targets"]:
            raise RuntimeError("paired cells have different target counts")
        baseline_samples.append(baseline)
        pattern_samples.append(pattern)
        per_repeat_net_benefit.append(
            float(baseline["total_exposed_wait_ms"])
            - float(pattern["total_exposed_wait_ms"])
        )

    targets = int(_sum_sample_field(pattern_samples, "authoritative_targets"))
    requested = int(_sum_sample_field(pattern_samples, "requested_predictions"))
    admitted = int(_sum_sample_field(pattern_samples, "admitted_predictions"))
    rejected = int(_sum_sample_field(pattern_samples, "rejected_predictions"))
    exact_hits = int(_sum_sample_field(pattern_samples, "exact_hits"))
    baseline_total_ms = _sum_sample_field(baseline_samples, "total_exposed_wait_ms")
    pattern_total_ms = _sum_sample_field(pattern_samples, "total_exposed_wait_ms")
    net_ms = baseline_total_ms - pattern_total_ms
    wrong_started = int(
        _sum_sample_field(pattern_samples, "wrong_speculations_started")
    )
    physical_started = int(
        _sum_sample_field(pattern_samples, "physical_calls_started")
    )
    baseline_physical_started = int(
        _sum_sample_field(baseline_samples, "physical_calls_started")
    )
    baseline_observed_service_ms = sum(
        float(sample["observed_service_ms"]["mean"])
        * int(sample["observed_service_ms"]["started_calls"])
        for sample in baseline_samples
    )
    pattern_observed_service_ms = sum(
        float(sample["observed_service_ms"]["mean"])
        * int(sample["observed_service_ms"]["started_calls"])
        for sample in pattern_samples
    )
    all_safety = all(
        all(bool(value) for value in sample["safety"].values())
        for sample in baseline_samples + pattern_samples
    )
    source_counts: Counter[str] = Counter()
    for sample in pattern_samples:
        source_counts.update(
            {
                str(name): int(count)
                for name, count in sample["source_counts"].items()
            }
        )
    saved_speculative_service_ms = _sum_sample_field(
        pattern_samples, "saved_speculative_service_ms"
    )
    overlap_producing_hits = int(source_counts.get("reused", 0)) + int(
        source_counts.get("promoted_inflight", 0)
    )
    baseline_wall_s = _sum_sample_field(baseline_samples, "wall_s")
    pattern_wall_s = _sum_sample_field(pattern_samples, "wall_s")
    wall_net_s = baseline_wall_s - pattern_wall_s
    admission_batches = sum(
        int(sample["admission_submission_ms"]["batches"])
        for sample in pattern_samples
    )
    if scenario == "all_wrong_counterfactual" and (
        exact_hits != 0 or saved_speculative_service_ms != 0.0
    ):
        raise RuntimeError("all-wrong scenario unexpectedly reused speculation")
    return {
        "scenario": scenario,
        "width": width,
        "offered_concurrency": offered_concurrency,
        "repetitions": repetitions,
        "workers": workers,
        "speculative_workers": speculative_workers,
        "visit_capacity": visit_capacity,
        "max_speculative_pending": max_speculative_pending,
        "synthetic_service_ms": service_ms,
        "synthetic_prediction_lead_ms": lead_ms,
        "requested_predictions": requested,
        "admitted_predictions": admitted,
        "rejected_predictions": rejected,
        "admission_ratio": ratio(admitted, requested),
        "authoritative_targets": targets,
        "exact_hits": exact_hits,
        "source_counts": dict(sorted(source_counts.items())),
        "completed_reuse_hits": int(source_counts.get("reused", 0)),
        "inflight_promotion_hits": int(
            source_counts.get("promoted_inflight", 0)
        ),
        "queued_promotion_hits": int(
            source_counts.get("promoted_from_queue", 0)
        ),
        "overlap_producing_hits": overlap_producing_hits,
        "overlap_producing_target_coverage": ratio(
            overlap_producing_hits, targets
        ),
        "realized_exact_target_coverage": ratio(exact_hits, targets),
        "admitted_candidate_precision": ratio(exact_hits, admitted),
        "logical_waste_admitted": int(
            _sum_sample_field(pattern_samples, "logical_waste_admitted")
        ),
        "wrong_speculations_started": wrong_started,
        "wrong_speculations_never_started": int(
            _sum_sample_field(pattern_samples, "wrong_speculations_never_started")
        ),
        "wasted_speculative_service_ms": _sum_sample_field(
            pattern_samples, "wasted_speculative_service_ms"
        ),
        "saved_speculative_service_ms": saved_speculative_service_ms,
        "wasted_service_ms_per_authoritative_target": ratio(
            _sum_sample_field(pattern_samples, "wasted_speculative_service_ms"),
            targets,
        ),
        "observed_wrong_service_ms_per_started_call": ratio(
            _sum_sample_field(pattern_samples, "wasted_speculative_service_ms"),
            wrong_started,
        ),
        "baseline_observed_service_ms_per_started_call": ratio(
            baseline_observed_service_ms, baseline_physical_started
        ),
        "pattern_observed_service_ms_per_started_call": ratio(
            pattern_observed_service_ms, physical_started
        ),
        "baseline_observed_service_p95_ms_mean_across_repeats": (
            statistics.fmean(
                float(sample["observed_service_ms"]["p95"])
                for sample in baseline_samples
            )
        ),
        "pattern_observed_service_p95_ms_mean_across_repeats": (
            statistics.fmean(
                float(sample["observed_service_ms"]["p95"])
                for sample in pattern_samples
            )
        ),
        "physical_calls_started": physical_started,
        "physical_call_amplification_vs_demand_only": ratio(
            physical_started, targets
        ),
        "baseline_mean_exposed_wait_ms": ratio(baseline_total_ms, targets),
        "pattern_mean_exposed_wait_ms": ratio(pattern_total_ms, targets),
        "baseline_p95_exposed_wait_ms_mean_across_repeats": statistics.fmean(
            float(sample["p95_exposed_wait_ms"]) for sample in baseline_samples
        ),
        "pattern_p95_exposed_wait_ms_mean_across_repeats": statistics.fmean(
            float(sample["p95_exposed_wait_ms"]) for sample in pattern_samples
        ),
        "net_latency_benefit_ms_total": net_ms,
        "net_latency_benefit_ms_per_target": ratio(net_ms, targets),
        "net_latency_benefit_fraction": ratio(net_ms, baseline_total_ms),
        "baseline_drained_workload_wall_s": baseline_wall_s,
        "pattern_drained_workload_wall_s": pattern_wall_s,
        "drained_workload_wall_benefit_s": wall_net_s,
        "drained_workload_wall_benefit_fraction": ratio(
            wall_net_s, baseline_wall_s
        ),
        "baseline_authoritative_throughput_per_s": ratio(
            targets, baseline_wall_s
        ),
        "pattern_authoritative_throughput_per_s": ratio(
            targets, pattern_wall_s
        ),
        "admission_deadline_overrun_batches": sum(
            int(sample["admission_submission_ms"]["decision_deadline_overrun_batches"])
            for sample in pattern_samples
        ),
        "admission_batches": admission_batches,
        "admission_submission_mean_ms_across_repeats": statistics.fmean(
            float(sample["admission_submission_ms"]["mean"])
            for sample in pattern_samples
        ),
        "admission_submission_max_ms": max(
            float(sample["admission_submission_ms"]["max"])
            for sample in pattern_samples
        ),
        "confirmation_offset_mean_ms_across_repeats": statistics.fmean(
            float(sample["confirmation_offset_ms_from_batch_start"]["mean"])
            for sample in pattern_samples
        ),
        "confirmation_offset_p95_ms_mean_across_repeats": statistics.fmean(
            float(sample["confirmation_offset_ms_from_batch_start"]["p95"])
            for sample in pattern_samples
        ),
        "confirmation_offset_max_ms": max(
            float(sample["confirmation_offset_ms_from_batch_start"]["max"])
            for sample in pattern_samples
        ),
        "repeat_net_latency_benefit_ms": per_repeat_net_benefit,
        "all_safety_invariants_passed": all_safety,
        "samples": {
            "baseline": baseline_samples,
            "pattern_v2": pattern_samples,
        },
    }


async def run_load_sweep(
    opportunities: Sequence[ReplayOpportunity],
    *,
    widths: Sequence[int],
    concurrencies: Sequence[int],
    repetitions: int,
    workers: int,
    speculative_workers: int,
    visit_capacity: int,
    max_speculative_pending: int,
    service_ms: float,
    lead_ms: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in widths:
        for concurrency in concurrencies:
            rows.append(
                await paired_broker_cell(
                    opportunities,
                    scenario="observed_pattern_v2_oof",
                    width=width,
                    offered_concurrency=concurrency,
                    repetitions=repetitions,
                    workers=workers,
                    speculative_workers=speculative_workers,
                    visit_capacity=visit_capacity,
                    max_speculative_pending=max_speculative_pending,
                    service_ms=service_ms,
                    lead_ms=lead_ms,
                )
            )
    wrong = force_all_wrong(opportunities)
    for concurrency in concurrencies:
        rows.append(
            await paired_broker_cell(
                wrong,
                scenario="all_wrong_counterfactual",
                width=FROZEN_TOP_K,
                offered_concurrency=concurrency,
                repetitions=repetitions,
                workers=workers,
                speculative_workers=speculative_workers,
                visit_capacity=visit_capacity,
                max_speculative_pending=max_speculative_pending,
                service_ms=service_ms,
                lead_ms=lead_ms,
            )
        )
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    flattened: list[dict[str, Any]] = []
    for row in rows:
        simple = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list, tuple))
        }
        flattened.append(simple)
        for key in simple:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def signed_ms(value: float) -> str:
    return f"{value:+.2f}"


def render_report(payload: Mapping[str, Any]) -> str:
    static = payload["static_runtime_prefixes"]
    oracle = payload["bounded_pool_oracle"]
    load_rows = payload["shared_pool_load_sweep"]
    natural = [row for row in load_rows if row["scenario"] == "observed_pattern_v2_oof"]
    wrong = [row for row in load_rows if row["scenario"] == "all_wrong_counterfactual"]
    runtime_rows = [row for row in natural if int(row["width"]) == FROZEN_TOP_K]
    if not runtime_rows:
        widest = max(int(row["width"]) for row in natural)
        runtime_rows = [row for row in natural if int(row["width"]) == widest]
    runtime_low = min(runtime_rows, key=lambda row: int(row["offered_concurrency"]))
    runtime_high = max(runtime_rows, key=lambda row: int(row["offered_concurrency"]))
    lines = [
        "# Pattern-v2 robustness under low predictability and high load",
        "",
        "## Bottom line",
        "",
        (
            "The premise `Top-1 ≈27.8% / hit rate 93.8%` is not a reproducible "
            "Pattern-v2 metric pair. On the 100-session whole-session grouped-OOF "
            f"replay, exact target Top-1 is {pct(static[0]['exact_target_recall'])}; "
            "grouped-OOF recall at the frozen runtime width Top-5 is "
            f"{pct(static[-1]['exact_target_recall'])}."
        ),
        "",
        (
            f"The nearby {pct(oracle['target_coverage'])} number is an "
            "evaluation-only bounded-pool target oracle (visit-window coverage "
            f"{pct(oracle['visit_window_coverage'])}). Firing every admitted candidate "
            f"union would issue {oracle['candidate_count_if_all_fired']} candidates "
            "and expose this ceiling, with only "
            f"{pct(oracle['candidate_precision_if_all_fired'])} candidate precision. "
            "The delivered v2 runtime never does this: it is frozen at Top-5."
        ),
        "",
        (
            "The repository's shared-capacity broker implementation was then stressed "
            "with synthetic service. At the widest tested runtime prefix, conservative "
            "exposed-wait benefit is "
            f"{signed_ms(runtime_low['conservative_net_including_predictor_ms_per_target'])} "
            f"ms/target at burst width {runtime_low['offered_concurrency']} and "
            f"{signed_ms(runtime_high['conservative_net_including_predictor_ms_per_target'])} "
            f"ms/target at width {runtime_high['offered_concurrency']}. The drained "
            "workload-time result is reported separately because wrong-candidate "
            "cleanup can hurt throughput even when confirmation-to-result wait improves."
        ),
        "",
        "## Static Pattern-v2 quality and logical waste",
        "",
        "| K | Exact target recall | Visit-window coverage | Candidates | Candidate precision | Logical waste | Logical invocation-equivalent upper envelope |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in static:
        lines.append(
            f"| {row['width']} | {row['target_hits']}/{row['authoritative_targets']} "
            f"({pct(row['exact_target_recall'])}) | "
            f"{row['hit_visit_windows']}/{row['visit_windows']} "
            f"({pct(row['visit_window_coverage'])}) | "
            f"{row['requested_candidates']} | {pct(row['candidate_precision'])} | "
            f"{row['logical_waste_candidates']} ({pct(row['logical_waste_fraction'])}) | "
            f"{row['worst_complete_call_amplification_vs_demand_only']:.2f}x |"
        )
    lines.extend(
        [
            "",
            (
                "`Candidates` includes all gated search windows, including windows "
                "whose next tool was not `visit`; they are selected candidate demand, "
                "although capacity rejection or cancellation can prevent physical work. "
                "The final column is a logical upper envelope on the historical-label "
                "denominator (which contains one non-executable label), not a measured "
                "physical-call ratio. It assumes every selected candidate completes "
                "before unused work is cancelled."
            ),
            "",
            "## Closed-loop shared-pool burst sweep (CPU-only synthetic service)",
            "",
            (
                f"Configuration: {payload['configuration']['workers']} shared workers, "
                f"at most {payload['configuration']['speculative_workers']} speculative "
                f"workers, visit capacity {payload['configuration']['visit_capacity']}, "
                f"pending cap {payload['configuration']['max_speculative_pending']}, "
                f"executor-requested sleep {payload['configuration']['synthetic_service_ms']:.1f} ms, "
                f"decision deadline {payload['configuration']['synthetic_prediction_lead_ms']:.1f} ms "
                "from batch start. Candidate-submission time consumes that deadline; "
                "observed service also includes event-loop scheduling delay and is "
                "reported explicitly. This is a closed-loop drained-burst stress, not "
                "sustained open-loop traffic. The denominator is executable HTTP(S) "
                "targets only; invalid trace labels are not dispatched."
            ),
            "",
            "| K | Burst width | Admission / deadline misses | Admitted exact match | Overlap-producing hit | Wrong starts (pooled) | Observed wrong service/start | Physical-call amp. | Mean exposed wait baseline→v2 | Conservative exposed net | Conservative drained wall incl. predictor baseline→v2 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in natural:
        repetitions = int(row["repetitions"])
        baseline_wall = float(row["baseline_drained_workload_wall_s"]) / repetitions
        pattern_wall = (
            float(row["conservative_pattern_wall_including_predictor_s"])
            / repetitions
        )
        lines.append(
            f"| {row['width']} | {row['offered_concurrency']} | "
            f"{pct(row['admission_ratio'])} / "
            f"{row['admission_deadline_overrun_batches']}/"
            f"{row['admission_batches']} | "
            f"{pct(row['realized_exact_target_coverage'])} | "
            f"{pct(row['overlap_producing_target_coverage'])} | "
            f"{row['wrong_speculations_started']} | "
            f"{row['observed_wrong_service_ms_per_started_call']:.2f} ms | "
            f"{row['physical_call_amplification_vs_demand_only']:.2f}x | "
            f"{row['baseline_mean_exposed_wait_ms']:.2f}→"
            f"{row['pattern_mean_exposed_wait_ms']:.2f} ms | "
            f"{signed_ms(row['conservative_net_including_predictor_ms_per_target'])} ms "
            f"({row['conservative_net_including_predictor_fraction'] * 100:+.1f}%) | "
            f"{baseline_wall:.3f}→{pattern_wall:.3f} s "
            f"({row['conservative_drained_wall_benefit_including_predictor_fraction'] * 100:+.1f}%) |"
        )
    lines.extend(
        [
            "",
            "`Admitted exact match` includes queued promotion; only completed reuse "
            "and inflight promotion count as an overlap-producing hit. A positive "
            "net value means Pattern-v2 reduced exposed authoritative wait; a negative "
            "value means contention cost more than overlap saved. "
            "Requested-but-rejected candidates do no physical work. The JSON also "
            "separates admitted-never-started waste from wrong calls that actually "
            "started and records p95/p99 waits and every paired repetition. Started "
            f"call counts are pooled over {payload['configuration']['repetitions']} "
            "repetitions. Conservative columns charge the full local predictor runtime "
            "serially; in practice it may overlap. `Drained wall` additionally includes "
            "candidate admission and cancellation tails, so it is the appropriate "
            "closed-loop throughput check rather than a per-session latency claim. "
            "Its percentage is an unclamped benefit: a negative value means wall time "
            "increased by that magnitude. "
            "A deadline miss means serial candidate admission itself exhausted the "
            "batch-start-to-confirmation lead budget; exact confirmation offsets and "
            "submission distributions remain in `metrics.json`.",
            "",
            "## Mostly-wrong worst case",
            "",
            "This counterfactual keeps the exact Pattern-v2 gate, number/order of "
            "candidates, arrival batches, service time, and authoritative target "
            "multiplicity, but deterministically replaces every target so no "
            "candidate can match.",
            "",
            "| Burst width | Admission / deadline misses | Exact / overlap hits | Wrong starts (pooled) | Observed wrong service/start | Physical-call amp. | Mean exposed wait baseline→v2 | Conservative exposed net | Conservative drained wall incl. predictor baseline→v2 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in wrong:
        repetitions = int(row["repetitions"])
        baseline_wall = float(row["baseline_drained_workload_wall_s"]) / repetitions
        pattern_wall = (
            float(row["conservative_pattern_wall_including_predictor_s"])
            / repetitions
        )
        lines.append(
            f"| {row['offered_concurrency']} | {pct(row['admission_ratio'])} / "
            f"{row['admission_deadline_overrun_batches']}/"
            f"{row['admission_batches']} | "
            f"{row['exact_hits']} / {row['overlap_producing_hits']} | "
            f"{row['wrong_speculations_started']} | "
            f"{row['observed_wrong_service_ms_per_started_call']:.2f} ms | "
            f"{row['physical_call_amplification_vs_demand_only']:.2f}x | "
            f"{row['baseline_mean_exposed_wait_ms']:.2f}→"
            f"{row['pattern_mean_exposed_wait_ms']:.2f} ms | "
            f"{signed_ms(row['conservative_net_including_predictor_ms_per_target'])} ms "
            f"({row['conservative_net_including_predictor_fraction'] * 100:+.1f}%) | "
            f"{baseline_wall:.3f}→{pattern_wall:.3f} s "
            f"({row['conservative_drained_wall_benefit_including_predictor_fraction'] * 100:+.1f}%) |"
        )
    lines.extend(
        [
            "",
            "Worst-case behavior is fail-safe for correctness, not free for latency: "
            "speculative results never commit without an exact same-session "
            "authoritative claim. The configured speculative and visit caps are "
            f"{payload['configuration']['speculative_workers']} and "
            f"{payload['configuration']['visit_capacity']} within "
            f"{payload['configuration']['workers']} global workers, and at most "
            f"{payload['configuration']['max_speculative_pending']} predictions can "
            "remain pending. Those caps bound instantaneous occupancy, not cumulative "
            "waste across arriving batches. In the current broker, bulk cancellation "
            "waits for jobs one by one; while it waits for a non-preemptive wrong call, "
            "queued wrong siblings may start and must also drain. Thus even burst width "
            "1 can execute the whole selected wrong set. Pending-cap saturation rejects "
            "new candidates while full, but cumulative wrong work grows again as slots "
            "drain and later batches arrive.",
            "",
            "If an external visit hangs, the finite worst-case latency bound comes from "
            "the visit timeout or backend service bound, not from Pattern-v2. Without "
            "such a timeout there is no finite predictor-only bound. The all-wrong "
            "drained-wall column captures this cleanup tail for the bounded synthetic "
            "executor used here.",
            "",
            "## Scope and reproducibility",
            "",
            "- Prediction evidence is development-only grouped OOF over the existing "
            "100 sessions; no genuinely unseen confirmatory trace set remains.",
            "- Queue results use the repository's real `LiveToolBroker` and exact "
            "session-scoped confirmation. The executor requests a fixed sleep, but "
            "observed service includes event-loop scheduling delay and is reported. "
            "This is a scheduler experiment, not an end-to-end GPU/network claim.",
            "- Load is closed-loop drained visit-window bursts only: there is no "
            "open-loop sustained arrival process, mixed search/LLM traffic, or tool "
            "start-rate gate. Exposed authoritative wait and drained workload wall "
            "answer different latency and throughput questions.",
            "- Each paired repetition runs baseline first and Pattern-v2 second. Three "
            "repetitions are descriptive and no confidence interval is claimed.",
            "- The 27.8% and legacy 93.8% values are not substituted into this run. "
            "All reported Pattern-v2 numerators and denominators are regenerated from "
            "the checked-in Qwen traces.",
            "- No vLLM server, model inference, or network request is used.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            payload["reproduction_command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--widths", type=parse_ints, default=(1, 3, 5))
    parser.add_argument(
        "--concurrencies", type=parse_ints, default=DEFAULT_CONCURRENCIES
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--speculative-workers", type=int, default=2)
    parser.add_argument("--visit-capacity", type=int, default=2)
    parser.add_argument("--max-speculative-pending", type=int, default=128)
    parser.add_argument("--service-ms", type=float, default=10.0)
    parser.add_argument("--lead-ms", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.workers <= 1:
        raise SystemExit("--workers must be greater than one")
    if not 0 <= args.speculative_workers < args.workers:
        raise SystemExit("--speculative-workers must be in [0, workers)")
    if not 1 <= args.visit_capacity <= args.workers:
        raise SystemExit("--visit-capacity must be in [1, workers]")
    if args.max_speculative_pending <= 0:
        raise SystemExit("--max-speculative-pending must be positive")
    if args.service_ms <= 0 or args.lead_ms < 0:
        raise SystemExit("service must be positive and lead non-negative")
    if any(width > FROZEN_TOP_K for width in args.widths):
        raise SystemExit(f"--widths cannot exceed frozen Top-{FROZEN_TOP_K}")
    if not args.traces.is_dir():
        raise SystemExit(f"trace directory does not exist: {args.traces}")
    if not args.artifact.is_file():
        raise SystemExit(f"Pattern-v2 artifact does not exist: {args.artifact}")

    artifact_predictor, artifact_document = load_pattern_artifact(args.artifact)
    artifact_metadata = artifact_predictor.metadata()
    if (
        artifact_document.get("version") != PATTERN_ARTIFACT_VERSION
        or artifact_metadata["policy"] != PATTERN_POLICY_VERSION
        or int(artifact_metadata["top_k"]) != FROZEN_TOP_K
        or artifact_metadata["neural_model"] is not False
    ):
        raise RuntimeError("artifact does not validate as the frozen Pattern-v2 policy")

    rows, oof = collect_pattern_v2_oof_rows(args.traces)
    static = static_width_metrics(rows, DEFAULT_WIDTHS)
    oracle = bounded_pool_oracle_metrics(rows)
    opportunities = build_replay_opportunities(rows)
    load_rows = asyncio.run(
        run_load_sweep(
            opportunities,
            widths=args.widths,
            concurrencies=args.concurrencies,
            repetitions=args.repetitions,
            workers=args.workers,
            speculative_workers=args.speculative_workers,
            visit_capacity=args.visit_capacity,
            max_speculative_pending=args.max_speculative_pending,
            service_ms=args.service_ms,
            lead_ms=args.lead_ms,
        )
    )
    predictor_ms_per_replay = float(
        oof["predictor_compute"]["total_ms_per_full_replay"]
    )
    for row in load_rows:
        serialized_predictor_ms = (
            predictor_ms_per_replay * int(row["repetitions"])
        )
        conservative_net_ms = (
            float(row["net_latency_benefit_ms_total"])
            - serialized_predictor_ms
        )
        baseline_total_ms = (
            float(row["baseline_mean_exposed_wait_ms"])
            * int(row["authoritative_targets"])
        )
        row["serialized_predictor_compute_ms"] = serialized_predictor_ms
        row["conservative_net_including_predictor_ms_total"] = (
            conservative_net_ms
        )
        row["conservative_net_including_predictor_ms_per_target"] = ratio(
            conservative_net_ms, int(row["authoritative_targets"])
        )
        row["conservative_net_including_predictor_fraction"] = ratio(
            conservative_net_ms, baseline_total_ms
        )
        conservative_pattern_wall_s = (
            float(row["pattern_drained_workload_wall_s"])
            + serialized_predictor_ms / 1000.0
        )
        conservative_wall_benefit_s = (
            float(row["baseline_drained_workload_wall_s"])
            - conservative_pattern_wall_s
        )
        row["conservative_pattern_wall_including_predictor_s"] = (
            conservative_pattern_wall_s
        )
        row["conservative_drained_wall_benefit_including_predictor_s"] = (
            conservative_wall_benefit_s
        )
        row["conservative_drained_wall_benefit_including_predictor_fraction"] = ratio(
            conservative_wall_benefit_s,
            float(row["baseline_drained_workload_wall_s"]),
        )
    if not all(bool(row["all_safety_invariants_passed"]) for row in load_rows):
        raise RuntimeError("not every shared-pool cell passed its safety invariants")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_not_confirmatory",
        "metric_audit": {
            "quoted_27_8_percent": (
                "not a reproducible Pattern-v2 Qwen exact-URL metric"
            ),
            "quoted_93_8_percent": (
                "not a Pattern-v2 overall speculative-execution hit rate"
            ),
            "load_target_denominator": (
                "executable HTTP(S) authoritative URL occurrences"
            ),
            "static_target_denominator": (
                "all trace authoritative URL occurrences, preserving historical "
                "exact-label scoring"
            ),
        },
        "policy": {
            "name": PATTERN_POLICY_VERSION,
            "artifact_version": PATTERN_ARTIFACT_VERSION,
            "frozen_runtime_top_k": FROZEN_TOP_K,
            "artifact_path": str(args.artifact.resolve()),
            "artifact_file_sha256": sha256_file(args.artifact),
            "artifact_internal_sha256": artifact_metadata["artifact_sha256"],
            "artifact_runtime_metadata": artifact_metadata,
            "artifact_role": (
                "validates frozen v2 configuration and policy identity only; "
                "grouped-OOF rank counts are refit on each training fold to "
                "prevent validation-session leakage"
            ),
        },
        "oof": oof,
        "static_runtime_prefixes": static,
        "bounded_pool_oracle": oracle,
        "configuration": {
            "widths_in_load_sweep": list(args.widths),
            "offered_concurrencies": list(args.concurrencies),
            "repetitions": args.repetitions,
            "workers": args.workers,
            "speculative_workers": args.speculative_workers,
            "visit_capacity": args.visit_capacity,
            "minimum_speculative_workers": 0,
            "authoritative_priority": True,
            "arrival_model": (
                "closed-loop drained bursts: offered_concurrency is batch width; "
                "each batch reaches the next decision deadline and fully drains "
                "authoritative work plus cancellation before the next batch"
            ),
            "mixed_traffic": False,
            "tool_start_rate_gate": False,
            "max_speculative_pending": args.max_speculative_pending,
            "synthetic_service_ms": args.service_ms,
            "synthetic_prediction_lead_ms": args.lead_ms,
            "network_requests": 0,
            "vllm_required": False,
        },
        "shared_pool_load_sweep": load_rows,
        "source_files": {
            "runner": {
                "path": str(SCRIPT),
                "sha256": sha256_file(SCRIPT),
            },
            "pattern_predictor": {
                "path": str(
                    REPRODUCTION_ROOT / "paste_repro" / "pattern_predictor.py"
                ),
                "sha256": sha256_file(
                    REPRODUCTION_ROOT / "paste_repro" / "pattern_predictor.py"
                ),
            },
            "pattern_evaluator": {
                "path": str(SCRIPT.parent / "run_pattern_cache_evaluation.py"),
                "sha256": sha256_file(
                    SCRIPT.parent / "run_pattern_cache_evaluation.py"
                ),
            },
            "shared_broker": {
                "path": str(REPRODUCTION_ROOT / "paste_repro" / "live_broker.py"),
                "sha256": sha256_file(
                    REPRODUCTION_ROOT / "paste_repro" / "live_broker.py"
                ),
            },
        },
        "reproduction_command": " ".join(
            [
                "PYTHONPATH=reproduction",
                "python",
                "reproduction/scripts/run_pattern_v2_load_robustness.py",
                "--traces",
                shlex.quote(str(args.traces)),
                "--artifact",
                shlex.quote(str(args.artifact)),
                "--output",
                shlex.quote(str(args.output)),
                "--widths",
                ",".join(map(str, args.widths)),
                "--concurrencies",
                ",".join(map(str, args.concurrencies)),
                "--repetitions",
                str(args.repetitions),
                "--workers",
                str(args.workers),
                "--speculative-workers",
                str(args.speculative_workers),
                "--visit-capacity",
                str(args.visit_capacity),
                "--max-speculative-pending",
                str(args.max_speculative_pending),
                "--service-ms",
                str(args.service_ms),
                "--lead-ms",
                str(args.lead_ms),
            ]
        ),
    }
    payload["result_sha256_excluding_self"] = canonical_sha256(payload)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    report_path = args.output / "REPORT.md"
    csv_path = args.output / "load_sweep.csv"
    write_json(metrics_path, payload)
    write_csv(csv_path, load_rows)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "search_decisions": oof["search_decisions"],
                "static_rows": len(static),
                "load_rows": len(load_rows),
                "all_safety_invariants_passed": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
