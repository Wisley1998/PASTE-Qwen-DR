#!/usr/bin/env python3
"""Reviewer-facing robustness sweep for all-visit speculative execution.

The sweep separates static selector coverage from load-dependent realized cache
hits, then corrupts 50%, 75%, or 100% of authority URL labels without changing
scores or admission.  This exposes wasted resource and the latency failure mode
when the predictor is miscalibrated or mostly wrong.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from run_pattern_v2_trace_all_visit_shared_capacity import (  # noqa: E402
    DEFAULT_TRACES,
    Policy,
    aggregate_runs,
    candidate_policy_windows,
    prepare_sessions,
    simulate,
)
from run_pattern_v2_trace_all_visit_wall import (  # noqa: E402
    build_session_global_cache_replays,
    collect_all_visit_timings,
    collect_nested_oof_all_visit_windows,
    trace_llm_scale_metadata,
)
from run_pattern_v2_trace_multi_spec_wall import session_full_walls  # noqa: E402
from run_pattern_v2_trace_timing_net_benefit import (  # noqa: E402
    build_oof_service_estimates,
    sha256_file,
)


SCHEMA = "paste_repro.pattern_v2_all_visit_robustness.v3"
DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT
    / "results"
    / "pattern_v2_all_visit_robustness_preemptible"
)
SCENARIOS = {
    "observed": 0.0,
    "wrong_50pct": 0.50,
    "mostly_wrong_75pct": 0.75,
    "all_wrong": 1.0,
}


def static_policy_rows(
    traces: Path,
    windows: Sequence[Any],
    decisions: Sequence[Any],
    timings: Mapping[str, Any],
    service_estimates: Mapping[str, Any],
    full_walls: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    authority_urls = sum(
        len(window.executable_targets) for window in windows
    )
    rows: list[dict[str, Any]] = []
    audits: dict[str, dict[str, Any]] = {}
    for name in (
        "fixed_top1",
        "fixed_top5",
        "budget_w5_cap10",
        "fixed_top10",
        "fixed_top20",
    ):
        policy_windows, width = candidate_policy_windows(
            windows, service_estimates, candidate_policy=name
        )
        _, audit = build_session_global_cache_replays(
            traces,
            policy_windows,
            decisions,
            timings,
            service_estimates,
            full_walls,
            per_task_width=width,
            coordination_cost_s=0.001,
        )
        audits[name] = audit
        immediate_hits = int(audit["immediate_selected_matches"])
        cache_hits = int(audit["cache_hit_occurrences"])
        physical_starts = int(audit["physical_speculative_starts"])
        rows.append(
            {
                "candidate_policy": name,
                "authority_urls": authority_urls,
                "policy_selections": int(audit["policy_selected_candidates"]),
                "physical_speculative_starts": physical_starts,
                "physical_starts_per_authority_call": (
                    physical_starts / authority_urls
                ),
                "immediate_exact_hits": immediate_hits,
                "immediate_exact_recall": immediate_hits / authority_urls,
                "persistent_cache_hits": cache_hits,
                "persistent_cache_coverage": cache_hits / authority_urls,
                "cache_hits_per_physical_start": (
                    cache_hits / physical_starts if physical_starts else 0.0
                ),
            }
        )
    return rows, audits


def run_cell(
    sessions: Sequence[Any],
    *,
    policy: Policy,
    capacity: int,
    concurrency: int,
    wrong_fraction: float,
    repetitions: int,
    baseline_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runs = [
        simulate(
            sessions,
            policy=policy,
            visit_capacity=capacity,
            offered_concurrency=concurrency,
            seed=seed,
            wrong_fraction=wrong_fraction,
        )
        for seed in range(repetitions)
    ]
    aggregate = aggregate_runs(runs)
    speedups = [
        1.0 - float(run["makespan_s"]) / float(baseline["makespan_s"])
        for run, baseline in zip(runs, baseline_runs, strict=True)
    ]
    net_seconds = [
        float(baseline["makespan_s"]) - float(run["makespan_s"])
        for run, baseline in zip(runs, baseline_runs, strict=True)
    ]
    aggregate.update(
        {
            "candidate_policy": policy.candidate_policy,
            "scheduler": policy.scheduler,
            "visit_capacity": capacity,
            "offered_concurrency": concurrency,
            "capacity_per_active_agent": capacity / concurrency,
            "wrong_fraction": wrong_fraction,
            "scenario": next(
                name for name, value in SCENARIOS.items() if value == wrong_fraction
            ),
            "e2e_speedup_fraction": statistics.fmean(speedups),
            "e2e_speedup_min_fraction": min(speedups),
            "e2e_speedup_max_fraction": max(speedups),
            "net_latency_benefit_s": statistics.fmean(net_seconds),
            "wasted_speculative_starts_per_authority_call": (
                aggregate["wasted_speculative_starts"]
                / aggregate["authority_requests"]
            ),
            "wasted_speculative_s_per_authority_call": (
                aggregate["wasted_speculative_s"]
                / aggregate["authority_requests"]
            ),
            "runs": runs,
        }
    )
    return aggregate


def render_report(payload: Mapping[str, Any]) -> str:
    static = payload["static_policy_rows"]
    results = payload["results"]
    lines = [
        "# Robustness under low predictability and high load",
        "",
        "## Metric audit: Top-1 versus firing many candidates",
        "",
        "The quoted 27.8% Top-1 and 93.8% hit rate are not reproduced as one "
        "same-scope metric by the frozen all-visit trace. The table below reports "
        "exact URL targets on the current 0.42x-LLM trace. `Immediate recall` uses "
        "the candidate set selected at that decision; `persistent coverage` also "
        "credits an earlier completed or in-flight session-cache prediction.",
        "",
        "| Candidate budget | Immediate exact hits | Immediate recall | Persistent "
        "cache hits | Persistent coverage | Policy selections | Physical starts "
        "| Hits/start |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in static:
        lines.append(
            f"| {row['candidate_policy']} "
            f"| {row['immediate_exact_hits']}/{row['authority_urls']} "
            f"| {row['immediate_exact_recall']:.2%} "
            f"| {row['persistent_cache_hits']}/{row['authority_urls']} "
            f"| {row['persistent_cache_coverage']:.2%} "
            f"| {row['policy_selections']} "
            f"| {row['physical_speculative_starts']} "
            f"| {row['cache_hits_per_physical_start']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Observed labels: concurrency and load",
            "",
            "Top-10 with preemptible adaptive idle-fill is shown below. Pool capacity "
            "scales with active Agent concurrency. Wasted work includes both cancelled "
            "partial calls and completed speculative calls that never produce a cache hit.",
            "",
            "| Slots/Agent | C | Policy coverage | Realized hit | Wasted calls/auth "
            "| Wasted seconds/auth | Call amp. | Net latency benefit | E2E speedup |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    top10_coverage = next(
        row["persistent_cache_coverage"]
        for row in static
        if row["candidate_policy"] == "fixed_top10"
    )
    observed_top10 = sorted(
        (
            row
            for row in results
            if row["scenario"] == "observed"
            and row["candidate_policy"] == "fixed_top10"
            and row["scheduler"] == "adaptive_idle_fill"
        ),
        key=lambda row: (
            row["capacity_per_active_agent"],
            row["offered_concurrency"],
        ),
    )
    for row in observed_top10:
        lines.append(
            f"| {row['capacity_per_active_agent']:.1f}x "
            f"| {row['offered_concurrency']} "
            f"| {top10_coverage:.2%} "
            f"| {row['realized_cache_hit_rate']:.2%} "
            f"| {row['wasted_speculative_starts_per_authority_call']:.3f} "
            f"| {row['wasted_speculative_s_per_authority_call']:.3f} s "
            f"| {row['call_amplification']:.3f}x "
            f"| {row['net_latency_benefit_s']:+.2f} s "
            f"| {row['e2e_speedup_fraction']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Candidate breadth at representative high load",
            "",
            "This cell uses C=16 and 1.5 Visit slots per active Agent. It shows "
            "whether wider firing remains worthwhile after charging wasted execution.",
            "",
            "| Candidates | Scheduler | Policy coverage | Realized hit | Wasted "
            "calls/auth | Wasted seconds/auth | Call amp. | E2E speedup |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    representative = sorted(
        (
            row
            for row in results
            if row["scenario"] == "observed"
            and row["offered_concurrency"] == 16
            and abs(row["capacity_per_active_agent"] - 1.5) < 1e-12
        ),
        key=lambda row: (row["candidate_policy"], row["scheduler"]),
    )
    coverage_by_policy = {
        row["candidate_policy"]: row["persistent_cache_coverage"] for row in static
    }
    for row in representative:
        lines.append(
            f"| {row['candidate_policy']} | {row['scheduler']} "
            f"| {coverage_by_policy[row['candidate_policy']]:.2%} "
            f"| {row['realized_cache_hit_rate']:.2%} "
            f"| {row['wasted_speculative_starts_per_authority_call']:.3f} "
            f"| {row['wasted_speculative_s_per_authority_call']:.3f} s "
            f"| {row['call_amplification']:.3f}x "
            f"| {row['e2e_speedup_fraction']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Degrading predictability",
            "",
            "For this deterministic negative control, 50%, 75%, or 100% of "
            "authority URLs are replaced by guaranteed non-candidate URLs after "
            "selection. Scores and admission are unchanged. The representative "
            "cell is C=16, 1.5 slots/Agent, adaptive idle-fill.",
            "",
            "| Candidates | Scenario | Realized hit | Wasted calls/auth | Wasted "
            "seconds/auth | Waste fraction | Call amp. | E2E speedup |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    degradation = sorted(
        (
            row
            for row in results
            if row["offered_concurrency"] == 16
            and abs(row["capacity_per_active_agent"] - 1.5) < 1e-12
            and row["scheduler"] == "adaptive_idle_fill"
            and row["candidate_policy"] in {"budget_w5_cap10", "fixed_top10"}
        ),
        key=lambda row: (row["candidate_policy"], row["wrong_fraction"]),
    )
    for row in degradation:
        lines.append(
            f"| {row['candidate_policy']} | {row['scenario']} "
            f"| {row['realized_cache_hit_rate']:.2%} "
            f"| {row['wasted_speculative_starts_per_authority_call']:.3f} "
            f"| {row['wasted_speculative_s_per_authority_call']:.3f} s "
            f"| {row['wasted_speculative_fraction']:.2%} "
            f"| {row['call_amplification']:.3f}x "
            f"| {row['e2e_speedup_fraction']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Deterministic all-wrong worst case",
            "",
            "Top-10 adaptive idle-fill is forced to miss every authority URL. "
            "Because running speculation is synchronously preempted for authority, "
            "the latency path falls back to baseline; the failure cost is wasted "
            "idle resource rather than authority delay or incorrect state commit.",
            "",
            "| Slots/Agent | C | Realized hit | Wasted calls/auth | Wasted "
            "seconds/auth | Waste fraction | Call amp. | E2E speedup |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    all_wrong = sorted(
        (
            row
            for row in results
            if row["scenario"] == "all_wrong"
            and row["candidate_policy"] == "fixed_top10"
            and row["scheduler"] == "adaptive_idle_fill"
        ),
        key=lambda row: (
            row["capacity_per_active_agent"],
            row["offered_concurrency"],
        ),
    )
    for row in all_wrong:
        lines.append(
            f"| {row['capacity_per_active_agent']:.1f}x "
            f"| {row['offered_concurrency']} "
            f"| {row['realized_cache_hit_rate']:.2%} "
            f"| {row['wasted_speculative_starts_per_authority_call']:.3f} "
            f"| {row['wasted_speculative_s_per_authority_call']:.3f} s "
            f"| {row['wasted_speculative_fraction']:.2%} "
            f"| {row['call_amplification']:.3f}x "
            f"| {row['e2e_speedup_fraction']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A wide candidate budget raises static coverage but also lowers useful "
            "work per start. Under load, the relevant quantity is realized hit after "
            "admission, not the unthrottled candidate-union coverage. Preemption makes "
            "the all-wrong latency behavior fail-safe, but it does not make wrong work "
            "free: resource amplification and wasted slot-seconds remain, so a runtime "
            "confidence/load gate should shrink to W5 or abstain as predicted utility "
            "falls.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--capacity-ratios", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    if any(value <= 0 for value in args.concurrencies + args.capacity_ratios):
        parser.error("concurrency and capacity ratios must be positive")
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    trace_scale = trace_llm_scale_metadata(args.traces)
    windows, nested_oof, decisions = collect_nested_oof_all_visit_windows(
        args.traces, candidate_pool_size=20, selector_model="blend"
    )
    timings = collect_all_visit_timings(args.traces, decisions, llm_duration_scale=1.0)
    service_estimates, service_estimator = build_oof_service_estimates(
        windows, timings, domain_prior_strength=10.0
    )
    full_walls = session_full_walls(args.traces, llm_duration_scale=1.0)
    static_rows, static_audits = static_policy_rows(
        args.traces,
        windows,
        decisions,
        timings,
        service_estimates,
        full_walls,
    )
    runtime_policies = ("fixed_top1", "budget_w5_cap10", "fixed_top10")
    prepared = {
        candidate_policy: prepare_sessions(
            args.traces,
            windows,
            decisions,
            timings,
            service_estimates,
            full_walls,
            candidate_policy=candidate_policy,
        )
        for candidate_policy in runtime_policies
    }
    cells = sorted(
        {
            (max(1, math.ceil(concurrency * ratio)), concurrency)
            for concurrency in args.concurrencies
            for ratio in args.capacity_ratios
        }
    )
    baselines: dict[str, list[dict[str, Any]]] = {}
    for capacity, concurrency in cells:
        baselines[f"pool{capacity}_c{concurrency}"] = [
            simulate(
                prepared["fixed_top1"],
                policy=None,
                visit_capacity=capacity,
                offered_concurrency=concurrency,
                seed=seed,
            )
            for seed in range(args.repetitions)
        ]

    results: list[dict[str, Any]] = []
    for capacity, concurrency in cells:
        baseline_runs = baselines[f"pool{capacity}_c{concurrency}"]
        # Observed labels compare breadth and fixed versus dynamic scheduling.
        for candidate_policy in runtime_policies:
            for scheduler in ("fixed_reserve_one", "adaptive_idle_fill"):
                results.append(
                    run_cell(
                        prepared[candidate_policy],
                        policy=Policy(candidate_policy, candidate_policy, scheduler),
                        capacity=capacity,
                        concurrency=concurrency,
                        wrong_fraction=0.0,
                        repetitions=args.repetitions,
                        baseline_runs=baseline_runs,
                    )
                )
        # Negative controls use the adaptive policy whose quota is most exposed
        # to wasted work when confidence is wrong.
        for wrong_fraction in (0.50, 0.75, 1.0):
            for candidate_policy in ("budget_w5_cap10", "fixed_top10"):
                results.append(
                    run_cell(
                        prepared[candidate_policy],
                        policy=Policy(
                            candidate_policy,
                            candidate_policy,
                            "adaptive_idle_fill",
                        ),
                        capacity=capacity,
                        concurrency=concurrency,
                        wrong_fraction=wrong_fraction,
                        repetitions=args.repetitions,
                        baseline_runs=baseline_runs,
                    )
                )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "effective_llm_duration_scale": trace_scale["materialized_scale"],
            "concurrencies": args.concurrencies,
            "capacity_ratios": args.capacity_ratios,
            "repetitions": args.repetitions,
            "authority_semantics": "preemptive priority; exact in-flight promotion",
            "wrong_label_method": (
                "stable SHA-256 target corruption after selection; guaranteed "
                "non-candidate URL; scores and admission unchanged"
            ),
        },
        "source_sha256": {
            "runner": sha256_file(SCRIPT),
            "shared_capacity_runner": sha256_file(
                SCRIPT.parent / "run_pattern_v2_trace_all_visit_shared_capacity.py"
            ),
            "all_visit_runner": sha256_file(
                SCRIPT.parent / "run_pattern_v2_trace_all_visit_wall.py"
            ),
            "llm_timing_manifest": trace_scale["manifest_sha256"],
        },
        "nested_oof": nested_oof,
        "service_estimator": service_estimator,
        "static_policy_rows": static_rows,
        "static_cache_audits": static_audits,
        "baseline_runs": baselines,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
