#!/usr/bin/env python3
"""Run one fixed all-visit speculative policy across a concurrency curve.

The policy is intentionally not re-selected at each load: fixed Top-10 exact-URL
candidates, infinite-TTL session URL cache, adaptive idle-fill admission, and
preemptible authority-first execution all share one fixed global Visit pool.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
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


SCHEMA = "paste_repro.pattern_v2_all_visit_load_curve.v5"
DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT / "results" / "pattern_v2_all_visit_top10_load_curve_c1_128"
)


def paired_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def render_report(payload: Mapping[str, Any]) -> str:
    config = payload["configuration"]
    cache = payload["policy_cache"]
    lines = [
        "# One-policy load curve: Top-10 + full session cache",
        "",
        "Every concurrency uses the same policy: Top-10 candidates, infinite-TTL "
        "zero-read-cost session URL cache, adaptive idle-fill, and immediate "
        "authority preemption in one fixed shared Visit pool. There is no per-load "
        "policy selection.",
        "",
        f"Unconstrained policy coverage is {cache['cache_hit_occurrences']}/"
        f"{payload['source_authority_urls']} = "
        f"{payload['policy_cache_hit_rate']:.2%}. The fixed replay contains "
        f"{payload['replay_sessions']} tasks ({config['workload_replicas']} replicas) "
        f"and the Visit pool has {config['visit_capacity']} slots.",
        "",
        "`Tool stall reduction` is the reduction in summed authority-visible Visit "
        "wait (queue + remaining service). `Overall E2E` is closed-loop makespan "
        "reduction over the complete 0.42x-LLM sessions.",
        "",
        "`Spec calls / auth call` counts all physically started speculative tool "
        "executions. `Unused spec calls / auth call` counts the subset whose result "
        "is never consumed by an authority call. Both use the number of authority "
        "tool calls as the denominator.",
        "",
        "| C | Realized hit | Spec calls / auth call | Unused spec calls / auth "
        "call | Tool stall reduction | Overall E2E |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["load_results"]:
        lines.append(
            f"| {row['task_concurrency']} "
            f"| {row['realized_cache_hit_rate']:.2%} "
            f"| {row['physical_speculative_starts_per_authority_call']:.3f} "
            f"| {row['wasted_speculative_starts_per_authority_call']:.3f} "
            f"| {row['tool_stall_reduction_fraction']:.2%} "
            f"| {row['overall_e2e_reduction_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Mostly-wrong high-load negative control",
            "",
            f"Only the maximum-load C={config['concurrencies'][-1]} cell is used. "
            "For the negative control, 75% or 100% of authority URLs are replaced "
            "after selection by guaranteed non-candidate URLs; prediction scores "
            "and admission remain unchanged.",
            "",
            "| C=128 scenario | Realized hit | Spec calls / auth call | Unused spec "
            "calls / auth call | Tool stall reduction | Overall E2E |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["worst_case_results"]:
        lines.append(
            f"| {row['scenario']} "
            f"| {row['realized_cache_hit_rate']:.2%} "
            f"| {row['physical_speculative_starts_per_authority_call']:.3f} "
            f"| {row['wasted_speculative_starts_per_authority_call']:.3f} "
            f"| {row['tool_stall_reduction_fraction']:.2%} "
            f"| {row['overall_e2e_reduction_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "At 75% corruption the optimization retains only the overlap from the "
            "remaining correct predictions. In the deterministic all-wrong case, "
            "running speculation is preempted immediately for authority: realized "
            "hit and latency benefit both become zero, no wrong result is committed, "
            "and the cost is limited to wasted speculative calls on otherwise idle "
            "tool capacity.",
            "",
            "Realized hit falls only when the fixed pool loses speculative slack. "
            "Wrong running calls are preempted before authority dispatch, so the "
            "curve charges resource waste without allowing speculation to sit in "
            "front of a real Visit.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visit-capacity", type=int, default=64)
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 48, 64, 96, 128],
    )
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--workload-replicas", type=int, default=2)
    args = parser.parse_args()
    if args.visit_capacity <= 0 or any(value <= 0 for value in args.concurrencies):
        parser.error("capacity and concurrencies must be positive")
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    if args.workload_replicas <= 0:
        parser.error("workload replicas must be positive")
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
    policy_windows, width = candidate_policy_windows(
        windows, service_estimates, candidate_policy="fixed_top10"
    )
    _, cache_audit = build_session_global_cache_replays(
        args.traces,
        policy_windows,
        decisions,
        timings,
        service_estimates,
        full_walls,
        per_task_width=width,
        coordination_cost_s=0.001,
    )
    source_sessions = prepare_sessions(
        args.traces,
        windows,
        decisions,
        timings,
        service_estimates,
        full_walls,
        candidate_policy="fixed_top10",
    )
    sessions = tuple(
        replace(session, session_id=f"replica-{replica}:{session.session_id}")
        for replica in range(args.workload_replicas)
        for session in source_sessions
    )
    policy = Policy("fixed_top10", "fixed_top10", "adaptive_idle_fill")
    source_authority_urls = sum(
        len(window.executable_targets) for window in windows
    )
    replay_authority_urls = source_authority_urls * args.workload_replicas
    policy_hit_rate = (
        cache_audit["cache_hit_occurrences"] / source_authority_urls
    )
    source_authority_service_s = sum(
        sum(timing.visit_url_service_s) for timing in timings.values()
    )
    authority_service_s = source_authority_service_s * args.workload_replicas

    load_results: list[dict[str, Any]] = []
    raw_runs: dict[str, Any] = {}
    for concurrency in args.concurrencies:
        baseline_runs = [
            simulate(
                sessions,
                policy=None,
                visit_capacity=args.visit_capacity,
                offered_concurrency=concurrency,
                seed=seed,
            )
            for seed in range(args.repetitions)
        ]
        treatment_runs = [
            simulate(
                sessions,
                policy=policy,
                visit_capacity=args.visit_capacity,
                offered_concurrency=concurrency,
                seed=seed,
            )
            for seed in range(args.repetitions)
        ]
        baseline = aggregate_runs(baseline_runs)
        treatment = aggregate_runs(treatment_runs)
        e2e_reductions = [
            1.0 - float(treatment_run["makespan_s"]) / float(baseline_run["makespan_s"])
            for treatment_run, baseline_run in zip(
                treatment_runs, baseline_runs, strict=True
            )
        ]
        net_benefits = [
            float(baseline_run["makespan_s"]) - float(treatment_run["makespan_s"])
            for treatment_run, baseline_run in zip(
                treatment_runs, baseline_runs, strict=True
            )
        ]
        baseline_tool_s = float(baseline["authority_exposed_s"])
        treatment_tool_s = float(treatment["authority_exposed_s"])
        load_results.append(
            {
                "task_concurrency": concurrency,
                "visit_capacity": args.visit_capacity,
                "load_per_pool_slot": concurrency / args.visit_capacity,
                "authority_service_utilization_fraction": (
                    authority_service_s
                    / (args.visit_capacity * float(baseline["makespan_s"]))
                ),
                "policy_cache_hit_rate": policy_hit_rate,
                "realized_cache_hits": treatment["cache_hits"],
                "realized_cache_hit_rate": treatment[
                    "realized_cache_hit_rate"
                ],
                "policy_hit_retention": (
                    treatment["realized_cache_hit_rate"] / policy_hit_rate
                ),
                "baseline_authority_exposed_s": baseline_tool_s,
                "treatment_authority_exposed_s": treatment_tool_s,
                "tool_stall_reduction_fraction": (
                    1.0 - treatment_tool_s / baseline_tool_s
                ),
                "tool_speedup_factor": baseline_tool_s / treatment_tool_s,
                "baseline_full_makespan_s": baseline["makespan_s"],
                "treatment_full_makespan_s": treatment["makespan_s"],
                "overall_e2e_reduction_fraction": paired_mean(e2e_reductions),
                "overall_e2e_reduction_min_fraction": min(e2e_reductions),
                "overall_e2e_reduction_max_fraction": max(e2e_reductions),
                "net_latency_benefit_s": paired_mean(net_benefits),
                "call_amplification": treatment["call_amplification"],
                "physical_speculative_starts": treatment[
                    "physical_speculative_starts"
                ],
                "physical_speculative_starts_per_authority_call": (
                    treatment["physical_speculative_starts"]
                    / treatment["authority_requests"]
                ),
                "preempted_speculations": treatment["preempted_speculations"],
                "wasted_speculative_s": treatment["wasted_speculative_s"],
                "wasted_speculative_starts": treatment[
                    "wasted_speculative_starts"
                ],
                "wasted_speculative_starts_per_authority_call": (
                    treatment["wasted_speculative_starts"]
                    / treatment["authority_requests"]
                ),
                "wasted_speculative_s_per_authority_call": (
                    treatment["wasted_speculative_s"]
                    / treatment["authority_requests"]
                ),
                "wasted_speculative_starts_per_task": (
                    treatment["wasted_speculative_starts"] / len(sessions)
                ),
                "wasted_speculative_s_per_task": (
                    treatment["wasted_speculative_s"] / len(sessions)
                ),
                "wasted_speculative_fraction": treatment[
                    "wasted_speculative_fraction"
                ],
            }
        )
        raw_runs[str(concurrency)] = {
            "baseline": baseline_runs,
            "treatment": treatment_runs,
        }

    max_concurrency = max(args.concurrencies)
    max_baseline_runs = raw_runs[str(max_concurrency)]["baseline"]
    max_baseline = aggregate_runs(max_baseline_runs)

    def worst_case_row(
        scenario: str,
        wrong_fraction: float,
        treatment_runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        treatment = aggregate_runs(treatment_runs)
        e2e_reductions = [
            1.0 - float(treatment_run["makespan_s"])
            / float(baseline_run["makespan_s"])
            for treatment_run, baseline_run in zip(
                treatment_runs, max_baseline_runs, strict=True
            )
        ]
        baseline_tool_s = float(max_baseline["authority_exposed_s"])
        treatment_tool_s = float(treatment["authority_exposed_s"])
        return {
            "scenario": scenario,
            "wrong_fraction": wrong_fraction,
            "task_concurrency": max_concurrency,
            "realized_cache_hit_rate": treatment[
                "realized_cache_hit_rate"
            ],
            "physical_speculative_starts": treatment[
                "physical_speculative_starts"
            ],
            "physical_speculative_starts_per_authority_call": (
                treatment["physical_speculative_starts"]
                / treatment["authority_requests"]
            ),
            "wasted_speculative_starts_per_authority_call": (
                treatment["wasted_speculative_starts"]
                / treatment["authority_requests"]
            ),
            "tool_stall_reduction_fraction": (
                1.0 - treatment_tool_s / baseline_tool_s
            ),
            "overall_e2e_reduction_fraction": paired_mean(e2e_reductions),
            "call_amplification": treatment["call_amplification"],
            "wasted_speculative_fraction": treatment[
                "wasted_speculative_fraction"
            ],
            "runs": treatment_runs,
        }

    observed_runs = raw_runs[str(max_concurrency)]["treatment"]
    worst_case_results = [
        worst_case_row("observed", 0.0, observed_runs)
    ]
    for scenario, wrong_fraction in (
        ("mostly wrong (75%)", 0.75),
        ("all wrong (100%)", 1.0),
    ):
        scenario_runs = [
            simulate(
                sessions,
                policy=policy,
                visit_capacity=args.visit_capacity,
                offered_concurrency=max_concurrency,
                seed=seed,
                wrong_fraction=wrong_fraction,
            )
            for seed in range(args.repetitions)
        ]
        worst_case_results.append(
            worst_case_row(scenario, wrong_fraction, scenario_runs)
        )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "effective_llm_duration_scale": trace_scale["materialized_scale"],
            "visit_capacity": args.visit_capacity,
            "concurrencies": args.concurrencies,
            "repetitions": args.repetitions,
            "workload_replicas": args.workload_replicas,
            "candidate_policy": "fixed_top10",
            "cache": "infinite-TTL session URL, zero read cost, no expiration",
            "scheduler": "adaptive idle-fill, authority-first preemptible",
            "per_load_policy_selection": False,
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
        "source_sessions": len(source_sessions),
        "replay_sessions": len(sessions),
        "source_authority_urls": source_authority_urls,
        "replay_authority_urls": replay_authority_urls,
        "authority_service_s": authority_service_s,
        "policy_cache": cache_audit,
        "policy_cache_hit_rate": policy_hit_rate,
        "load_results": load_results,
        "worst_case_results": worst_case_results,
        "raw_runs": raw_runs,
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
