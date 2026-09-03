#!/usr/bin/env python3
"""Trace-timed net-benefit replay for Pattern-v2 tool speculation.

Unlike the control-plane runner, this replay never replaces tool service with
a fixed synthetic sleep.  It recovers, for every search decision, the causal
LLM overlap window and the user-visible visit stall from the original JSONL
timestamps.  Exact speculative hits remove one atomic authoritative URL call.

Multi-URL visit calls are modeled as serial URL executions. Corrected traces
provide sampled per-URL service; legacy traces fall back to equal share. All
speculations start concurrently before authority, so a later exact URL can
also finish while earlier authority URLs are running.
Search execution is outside the compared interval because its recorded result
is the causal input to both baseline and treatment prediction.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any
from urllib.parse import urlsplit


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.traces import LLMCall, OtherEvent, ToolCall, load_sessions  # noqa: E402
from run_pattern_cache_evaluation import extract_search_decisions  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredCandidate,
    ScoredWindow,
    collect_nested_oof_windows,
    session_stream_batches,
)


SCHEMA = "paste_repro.pattern_v2_trace_timing_net_benefit.v5"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT / "results" / "pattern_v2_trace_timing_net_benefit"
)
DECISION_ID_RE = re.compile(r"^(.*):search-line-(\d+):(\d+)$")


@dataclass(frozen=True)
class DecisionTiming:
    decision_id: str
    session_id: str
    llm_overlap_s: float
    visit_stall_s: float
    authoritative_urls: int
    timing_status: str
    visit_url_service_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class ServiceEstimate:
    """Causal atomic-service estimate for one held-out decision.

    ``expected_overlap_s`` is E[min(S, lead)] over atomic URL service samples
    from the other outer folds.  Retaining the distribution, rather than only
    its median, is important for the heavy-tailed Web visit latencies.
    """

    decision_id: str
    outer_fold: int
    training_atomic_samples: int
    expected_overlap_s: float
    candidate_expected_overlap_s: tuple[tuple[str, float], ...] = ()

    def overlap_for_url(self, url: str) -> float:
        return dict(self.candidate_expected_overlap_s).get(
            url, self.expected_overlap_s
        )


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def serial_visit_hit_saving(
    batch_stall_s: float,
    llm_lead_s: float,
    exact_hit_mask: Sequence[bool],
    url_service_s: Sequence[float] | None = None,
) -> float:
    """Replay serial URL authority against concurrent exact speculations.

    Corrected traces expose sampled per-URL service. Legacy traces expose only
    total batch stall and fall back to equal-share atomic service.
    Exact speculative URLs all start ``llm_lead_s`` before authority. During
    authority, they keep running concurrently, including while earlier miss
    URLs execute serially.
    """

    target_count = len(exact_hit_mask)
    if batch_stall_s <= 0.0 or target_count <= 0 or not any(exact_hit_mask):
        return 0.0
    if url_service_s:
        services = tuple(float(value) for value in url_service_s)
        if len(services) != target_count or any(value < 0.0 for value in services):
            raise ValueError("per-URL visit service does not match authority targets")
        if not math.isclose(
            sum(services), batch_stall_s, rel_tol=1e-9, abs_tol=1e-7
        ):
            raise ValueError("per-URL visit service does not sum to batch stall")
    else:
        services = (batch_stall_s / target_count,) * target_count
    treatment_stall_s = 0.0
    for exact_hit, service_s in zip(exact_hit_mask, services, strict=True):
        if exact_hit:
            # Completion time relative to authority start. It may be negative
            # when this speculative result is ready before authority begins.
            treatment_stall_s = max(
                treatment_stall_s,
                service_s - max(0.0, llm_lead_s),
            )
        else:
            treatment_stall_s += service_s
    return max(0.0, min(batch_stall_s, batch_stall_s - treatment_stall_s))


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_decision_timings(
    traces: Path, *, llm_duration_scale: float = 1.0
) -> dict[str, DecisionTiming]:
    sessions = load_sessions(traces)
    sessions_by_id = {session.session_id: session for session in sessions}
    timings: dict[str, DecisionTiming] = {}
    for decision in extract_search_decisions(sessions):
        match = DECISION_ID_RE.fullmatch(decision.decision_id)
        if match is None:
            raise ValueError(f"unrecognized decision id: {decision.decision_id}")
        session = sessions_by_id[match.group(1)]
        search_line = int(match.group(2))
        search_index = next(
            index
            for index, event in enumerate(session.events)
            if isinstance(event, ToolCall) and event.line_number == search_line
        )
        if search_index + 1 >= len(session.events):
            raise ValueError(f"{decision.decision_id}: missing decision LLM")
        decision_llm = session.events[search_index + 1]
        if not isinstance(decision_llm, LLMCall):
            raise ValueError(f"{decision.decision_id}: decision event is not LLM")

        visit_stall_s = 0.0
        visit_url_service_s: tuple[float, ...] = ()
        timing_status = "no_visit"
        if decision.outcome == "visit":
            visit_index = search_index + 2
            if (
                visit_index >= len(session.events)
                or not isinstance(session.events[visit_index], ToolCall)
                or session.events[visit_index].tool_name != "visit"
            ):
                raise ValueError(
                    f"{decision.decision_id}: labeled visit is not immediate"
                )
            visit = session.events[visit_index]
            correction = visit.timing_correction or {}
            raw_services = correction.get("unit_duration_s")
            if isinstance(raw_services, list) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in raw_services
            ):
                visit_url_service_s = tuple(float(value) for value in raw_services)
            completion = next(
                (
                    event
                    for event in session.events[visit_index + 1 :]
                    if isinstance(event, LLMCall)
                ),
                None,
            )
            if completion is None:
                synthetic_completion = next(
                    (
                        event
                        for event in session.events[visit_index + 1 :]
                        if isinstance(event, OtherEvent)
                        and event.event_type == "synthetic_tool_completion"
                        and event.payload.get("tool_name") == "visit"
                        and event.payload.get("call_index") == visit.call_index
                    ),
                    None,
                )
                if synthetic_completion is None:
                    timing_status = "visit_without_following_llm"
                else:
                    visit_stall_s = max(
                        0.0,
                        synthetic_completion.timestamp_s - visit.timestamp_s,
                    )
                    timing_status = "synthetic_terminal_visit_stall"
            else:
                visit_stall_s = max(
                    0.0,
                    completion.start_timestamp_s - visit.timestamp_s,
                )
                timing_status = "observed_visit_stall"
            if visit_url_service_s and not math.isclose(
                sum(visit_url_service_s),
                visit_stall_s,
                rel_tol=1e-9,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    f"{decision.decision_id}: corrected URL service does not match visit stall"
                )
        timings[decision.decision_id] = DecisionTiming(
            decision_id=decision.decision_id,
            session_id=decision.session_id,
            # Queue removal is a duration transform, not a timestamp rewrite.
            # In particular, changing the next LLM duration must not move the
            # already observed end of the preceding tool call.
            llm_overlap_s=max(
                0.0, decision_llm.overlap_window_s * llm_duration_scale
            ),
            visit_stall_s=visit_stall_s,
            authoritative_urls=len(decision.authoritative_urls),
            timing_status=timing_status,
            visit_url_service_s=visit_url_service_s,
        )
    return timings


def build_oof_service_estimates(
    windows: Sequence[ScoredWindow],
    timings: Mapping[str, DecisionTiming],
    *,
    domain_prior_strength: float = 10.0,
) -> tuple[dict[str, ServiceEstimate], dict[str, Any]]:
    """Estimate atomic visit overlap without using the held-out outer fold."""

    if domain_prior_strength < 0.0:
        raise ValueError("domain prior strength must be non-negative")

    # Importing here keeps the runner's public helpers easy to exercise in
    # isolation while using exactly the predictor's outer-fold identity.
    from run_pattern_cache_evaluation import cv_fold

    atomic_by_fold: dict[int, list[tuple[str, float]]] = {
        fold: [] for fold in range(5)
    }
    for window in windows:
        timing = timings[window.decision_id]
        target_count = len(window.executable_targets)
        if target_count == 0 or timing.visit_stall_s <= 0.0:
            continue
        atomic_service_s = timing.visit_stall_s / target_count
        # The execution and credit unit is one URL, so an n-URL authoritative
        # call contributes n atomic samples rather than one batch sample.
        services = timing.visit_url_service_s or (
            (atomic_service_s,) * target_count
        )
        atomic_by_fold[cv_fold(window.session_id)].extend(
            (urlsplit(url).hostname or "", service_s)
            for url, service_s in zip(
                window.executable_targets, services, strict=True
            )
        )

    estimates: dict[str, ServiceEstimate] = {}
    fold_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        training_rows = [
            row
            for fold, samples in atomic_by_fold.items()
            if fold != outer_fold
            for row in samples
        ]
        training_samples = [sample for _, sample in training_rows]
        if not training_samples:
            raise RuntimeError(
                f"outer fold {outer_fold} has no positive atomic service samples"
            )
        validation = [
            window
            for window in windows
            if cv_fold(window.session_id) == outer_fold
        ]
        domain_samples: dict[str, list[float]] = {}
        for domain, sample in training_rows:
            domain_samples.setdefault(domain, []).append(sample)
        for window in validation:
            lead_s = timings[window.decision_id].llm_overlap_s
            global_overlap_s = statistics.fmean(
                min(sample, lead_s) for sample in training_samples
            )
            candidate_overlaps: list[tuple[str, float]] = []
            for candidate in window.candidates:
                domain = urlsplit(candidate.pattern.url).hostname or ""
                samples = domain_samples.get(domain, [])
                if samples:
                    domain_overlap_s = statistics.fmean(
                        min(sample, lead_s) for sample in samples
                    )
                    weight = len(samples)
                    overlap_s = (
                        weight * domain_overlap_s
                        + domain_prior_strength * global_overlap_s
                    ) / (weight + domain_prior_strength)
                else:
                    overlap_s = global_overlap_s
                candidate_overlaps.append((candidate.pattern.url, overlap_s))
            estimates[window.decision_id] = ServiceEstimate(
                decision_id=window.decision_id,
                outer_fold=outer_fold,
                training_atomic_samples=len(training_samples),
                expected_overlap_s=global_overlap_s,
                candidate_expected_overlap_s=tuple(candidate_overlaps),
            )
        ordered = sorted(training_samples)
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "validation_decisions": len(validation),
                "training_atomic_samples": len(training_samples),
                "training_atomic_service_s": {
                    "mean": statistics.fmean(training_samples),
                    "p50": percentile(ordered, 0.50),
                    "p90": percentile(ordered, 0.90),
                    "p95": percentile(ordered, 0.95),
                    "max": max(ordered),
                },
            }
        )
    if set(estimates) != {window.decision_id for window in windows}:
        raise RuntimeError("OOF service estimator did not cover every decision")
    return estimates, {
        "method": (
            "outer-fold OOF domain-shrunk empirical "
            "E[min(atomic_service, scaled_lead)]"
        ),
        "atomic_unit": "visit_stall / executable URL count",
        "domain_key": "URL hostname",
        "domain_prior_strength": domain_prior_strength,
        "unknown_domain_fallback": "outer-fold global empirical distribution",
        "folds": fold_rows,
    }


def stable_tie(candidate: ScoredCandidate) -> str:
    return hashlib.sha256(
        (
            f"{candidate.pattern.session_id}\0{candidate.pattern.decision_id}\0"
            f"{candidate.pattern.url}"
        ).encode("utf-8")
    ).hexdigest()


def select_batch(
    batch: Sequence[ScoredWindow],
    service_estimates: Mapping[str, ServiceEstimate],
    *,
    global_k: int,
    coordination_cost_s: float,
) -> tuple[dict[str, ScoredCandidate], dict[str, float | int]]:
    per_decision: list[tuple[float, ScoredCandidate]] = []
    considered = 0
    for window in batch:
        if not window.v2_gate or not window.candidates:
            continue
        considered += len(window.candidates)
        estimate = service_estimates[window.decision_id]
        best = max(
            window.candidates,
            key=lambda candidate: (
                candidate.exact_probability
                * estimate.overlap_for_url(candidate.pattern.url),
                candidate.exact_probability,
                -candidate.pattern.position,
                stable_tie(candidate),
            ),
        )
        expected_gross = (
            best.exact_probability * estimate.overlap_for_url(best.pattern.url)
        )
        expected_net = expected_gross - coordination_cost_s
        if expected_net > 0.0:
            per_decision.append((expected_net, best))
    per_decision.sort(key=lambda row: (-row[0], stable_tie(row[1])))
    selected = per_decision[:global_k]
    return (
        {candidate.pattern.decision_id: candidate for _, candidate in selected},
        {
            "considered": considered,
            "eligible_decisions": len(per_decision),
            "selected": len(selected),
            "selected_expected_net_s": sum(value for value, _ in selected),
        },
    )


def replay_once(
    windows: Sequence[ScoredWindow],
    timings: Mapping[str, DecisionTiming],
    service_estimates: Mapping[str, ServiceEstimate],
    *,
    concurrency: int,
    seed: int,
    global_k: int,
    coordination_cost_s: float,
) -> dict[str, float | int]:
    batches = session_stream_batches(
        windows, offered_concurrency=concurrency, seed=seed
    )
    baseline_tool_stall_s = 0.0
    saved_tool_stall_s = 0.0
    baseline_wall_s = 0.0
    treatment_wall_s = 0.0
    authoritative_calls = 0
    selected_total = 0
    selected_exact_hits = 0
    visible_hits = 0
    timing_missing_visits = 0

    for batch in batches:
        selected, selection = select_batch(
            batch,
            service_estimates,
            global_k=global_k,
            coordination_cost_s=coordination_cost_s,
        )
        selected_total += int(selection["selected"])
        baseline_durations: list[float] = []
        treatment_durations: list[float] = []
        for window in batch:
            timing = timings[window.decision_id]
            targets = len(window.executable_targets)
            authoritative_calls += targets
            stall_s = timing.visit_stall_s if targets else 0.0
            baseline_tool_stall_s += stall_s
            candidate = selected.get(window.decision_id)
            exact_hit = bool(
                candidate is not None
                and candidate.exact_match
                and targets > 0
            )
            selected_exact_hits += int(exact_hit)
            exact_url = candidate.pattern.url if exact_hit else None
            exact_hit_mask = tuple(
                exact_url == url for url in window.executable_targets
            )
            gross_saved_s = serial_visit_hit_saving(
                stall_s,
                timing.llm_overlap_s,
                exact_hit_mask,
                timing.visit_url_service_s,
            )
            visible_hits += int(gross_saved_s > 0.0)
            saved_tool_stall_s += gross_saved_s
            if targets and timing.timing_status == "visit_without_following_llm":
                timing_missing_visits += 1
            baseline_duration = timing.llm_overlap_s + stall_s
            treatment_duration = (
                timing.llm_overlap_s
                + max(0.0, stall_s - gross_saved_s)
                + (coordination_cost_s if candidate is not None else 0.0)
            )
            baseline_durations.append(baseline_duration)
            treatment_durations.append(treatment_duration)
        baseline_wall_s += max(baseline_durations, default=0.0)
        treatment_wall_s += max(treatment_durations, default=0.0)

    # A production exact hit reuses the speculative call and suppresses its
    # matching AUTH call. Wrong predictions remain physical extra work.
    production_physical_calls = (
        authoritative_calls - selected_exact_hits + selected_total
    )
    net_saved_tool_s = saved_tool_stall_s - selected_total * coordination_cost_s
    return {
        "seed": seed,
        "batches": len(batches),
        "authoritative_calls": authoritative_calls,
        "selected": selected_total,
        "selected_exact_hits": selected_exact_hits,
        "visible_hits": visible_hits,
        "timing_missing_visits": timing_missing_visits,
        "baseline_tool_stall_s": baseline_tool_stall_s,
        "gross_saved_tool_stall_s": saved_tool_stall_s,
        "net_saved_tool_stall_s": net_saved_tool_s,
        "baseline_wall_s": baseline_wall_s,
        "treatment_wall_s": treatment_wall_s,
        "production_physical_calls": production_physical_calls,
    }


def summarize_runs(
    runs: Sequence[Mapping[str, float | int]], concurrency: int
) -> dict[str, Any]:
    def mean(key: str) -> float:
        return statistics.fmean(float(row[key]) for row in runs)

    baseline_stall = mean("baseline_tool_stall_s")
    saved_stall = mean("gross_saved_tool_stall_s")
    net_saved = mean("net_saved_tool_stall_s")
    baseline_wall = mean("baseline_wall_s")
    treatment_wall = mean("treatment_wall_s")
    authoritative = mean("authoritative_calls")
    selected = mean("selected")
    exact_hits = mean("selected_exact_hits")
    visible_hits = mean("visible_hits")
    return {
        "task_concurrency": concurrency,
        "repetitions": len(runs),
        "authoritative_calls_per_replay": authoritative,
        "selected_per_replay": selected,
        "exact_hits_per_replay": exact_hits,
        "visible_hits_per_replay": visible_hits,
        "exact_authority_hit_rate": ratio(exact_hits, authoritative),
        "visible_authority_hit_rate": ratio(visible_hits, authoritative),
        "prediction_precision": ratio(exact_hits, selected),
        "baseline_tool_stall_s_per_replay": baseline_stall,
        "gross_saved_tool_stall_s_per_replay": saved_stall,
        "net_saved_tool_stall_s_per_replay": net_saved,
        "tool_stall_reduction_fraction": ratio(saved_stall, baseline_stall),
        "net_tool_stall_reduction_fraction": ratio(net_saved, baseline_stall),
        "modeled_baseline_wall_s": baseline_wall,
        "modeled_treatment_wall_s": treatment_wall,
        "modeled_wall_speedup_fraction": ratio(
            baseline_wall - treatment_wall, baseline_wall
        ),
        "production_physical_call_amplification": ratio(
            mean("production_physical_calls"), authoritative
        ),
        "scheduling_sensitivity": {
            "net_saved_tool_stall_s_p05": percentile(
                [float(row["net_saved_tool_stall_s"]) for row in runs], 0.05
            ),
            "net_saved_tool_stall_s_p50": percentile(
                [float(row["net_saved_tool_stall_s"]) for row in runs], 0.50
            ),
            "net_saved_tool_stall_s_p95": percentile(
                [float(row["net_saved_tool_stall_s"]) for row in runs], 0.95
            ),
        },
        "runs": list(runs),
    }


def render_report(payload: Mapping[str, Any]) -> str:
    config = payload["configuration"]
    lines = [
        "# Pattern-v2 Real-Trace Timing Net-Benefit Replay",
        "",
        "This replay scales per-decision LLM overlap while preserving the observed",
        "visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms",
        "synthetic service or shadow AUTH is used.",
        "",
        f"Global K sweep=`{config['global_ks']}`, scheduling seeds=`{config['repetitions']}`, "
        f"coordination cost=`{config['coordination_cost_ms']} ms/start`.",
        f"LLM duration scale=`{config['llm_duration_scale']}`; selection uses "
        "outer-fold OOF empirical atomic-service distributions.",
    ]
    for k_result in payload["k_results"]:
        lines.extend(
            [
                "",
                f"## Global K={k_result['global_k']}",
                "",
                "| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in k_result["concurrency_results"]:
            lines.append(
                f"| {row['task_concurrency']} | {row['net_saved_tool_stall_s_per_replay']:.3f} s "
                f"| {row['net_tool_stall_reduction_fraction']:.2%} "
                f"| {row['modeled_wall_speedup_fraction']:.2%} "
                f"| {row['visible_authority_hit_rate']:.2%} "
                f"| {row['prediction_precision']:.2%} "
                f"| {row['production_physical_call_amplification']:.3f}x |"
            )
    lines.extend(
        [
            "",
            "Multi-URL visits use serial per-URL event replay (equal-share only",
            "for legacy traces); hits keep running during earlier authority URLs. Visits",
            "without a following LLM timestamp receive zero benefit. Repetitions are",
            "deterministic scheduling-order sensitivity runs, not independent traces.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--global-k", type=int, default=4)
    parser.add_argument(
        "--global-k-sweep",
        type=int,
        nargs="+",
        default=None,
        help="evaluate these K values in one OOF pass; overrides --global-k",
    )
    parser.add_argument("--coordination-cost-ms", type=float, default=1.0)
    parser.add_argument("--domain-prior-strength", type=float, default=10.0)
    parser.add_argument(
        "--llm-duration-scale",
        type=float,
        default=0.70,
        help=(
            "multiply LLM inference duration/overlap by this factor while "
            "preserving observed tool gaps (default: remove 30%%)"
        ),
    )
    args = parser.parse_args()
    global_ks = args.global_k_sweep or [args.global_k]
    if args.repetitions <= 0 or any(value <= 0 for value in global_ks):
        parser.error("repetitions and global K must be positive")
    if any(value <= 0 for value in args.concurrencies):
        parser.error("concurrencies must be positive")
    if args.coordination_cost_ms < 0:
        parser.error("coordination cost must be non-negative")
    if args.domain_prior_strength < 0:
        parser.error("domain prior strength must be non-negative")
    if not 0.0 < args.llm_duration_scale <= 1.0:
        parser.error("LLM duration scale must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    windows, oof = collect_nested_oof_windows(args.traces)
    raw_timings = collect_decision_timings(args.traces)
    timings = collect_decision_timings(
        args.traces, llm_duration_scale=args.llm_duration_scale
    )
    if set(timings) != {window.decision_id for window in windows}:
        raise RuntimeError("trace timing and OOF decision identities differ")
    if any(
        timing.visit_stall_s != raw_timings[decision_id].visit_stall_s
        for decision_id, timing in timings.items()
    ):
        raise RuntimeError("LLM scaling changed an observed tool stall")
    service_estimates, service_estimator = build_oof_service_estimates(
        windows,
        timings,
        domain_prior_strength=args.domain_prior_strength,
    )
    positive_stalls = [
        timing.visit_stall_s
        for timing in timings.values()
        if timing.visit_stall_s > 0.0
    ]
    global_ks = sorted(set(args.global_k_sweep or [args.global_k]))
    k_results = []
    for global_k in global_ks:
        concurrency_results = []
        for concurrency in args.concurrencies:
            runs = [
                replay_once(
                    windows,
                    timings,
                    service_estimates,
                    concurrency=concurrency,
                    seed=seed,
                    global_k=global_k,
                    coordination_cost_s=args.coordination_cost_ms / 1000.0,
                )
                for seed in range(args.repetitions)
            ]
            concurrency_results.append(summarize_runs(runs, concurrency))
        k_results.append(
            {"global_k": global_k, "concurrency_results": concurrency_results}
        )

    timing_status_counts: dict[str, int] = {}
    for timing in timings.values():
        timing_status_counts[timing.timing_status] = (
            timing_status_counts.get(timing.timing_status, 0) + 1
        )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "concurrencies": list(args.concurrencies),
            "repetitions": args.repetitions,
            "global_k": args.global_k,
            "global_ks": global_ks,
            "coordination_cost_ms": args.coordination_cost_ms,
            "domain_prior_strength": args.domain_prior_strength,
            "llm_duration_scale": args.llm_duration_scale,
            "selection": (
                "OOF probability * outer-fold OOF expected atomic overlap "
                "global Top-K, at most one URL per decision"
            ),
            "promotion": "exact hit suppresses one matching authority URL call",
            "multi_url_credit": (
                "event replay over serial per-URL authority service (equal-share "
                "legacy fallback); exact speculations run concurrently"
            ),
        },
        "trace_timing": {
            "decisions": len(timings),
            "status_counts": dict(sorted(timing_status_counts.items())),
            "positive_visit_stall_s": {
                "count": len(positive_stalls),
                "mean": statistics.fmean(positive_stalls),
                "p50": percentile(positive_stalls, 0.50),
                "p90": percentile(positive_stalls, 0.90),
                "p95": percentile(positive_stalls, 0.95),
                "p99": percentile(positive_stalls, 0.99),
                "max": max(positive_stalls),
            },
            "rows": [asdict(timing) for timing in timings.values()],
        },
        "service_estimator": {
            **service_estimator,
            "rows": [asdict(estimate) for estimate in service_estimates.values()],
        },
        "nested_oof": oof,
        # Preserve the single-K v1 access path while adding the complete sweep.
        "concurrency_results": k_results[0]["concurrency_results"],
        "k_results": k_results,
        "source_sha256": {
            "runner": sha256_file(SCRIPT),
            "adaptive_load": sha256_file(
                SCRIPT.parent / "run_pattern_v2_adaptive_load.py"
            ),
            "trace_loader": sha256_file(
                REPRODUCTION_ROOT / "paste_repro" / "traces.py"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
