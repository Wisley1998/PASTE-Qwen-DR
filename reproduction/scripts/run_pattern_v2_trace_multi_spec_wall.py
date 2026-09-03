#!/usr/bin/env python3
"""Real-trace wall replay with multiple concurrent speculations per task.

This runner answers a different question from
``run_pattern_v2_trace_timing_net_benefit.py``: instead of allowing at most one
URL candidate per decision and sharing a lockstep Top-K across tasks, it starts
the best ``N`` positive-value URL candidates for every admitted decision.

Speculative calls are assumed to use isolated capacity, so wrong calls do not
delay authority.  The required capacity and physical-call amplification are
reported explicitly.  This is therefore an interference-free benefit ceiling,
not a claim about a bounded shared tool pool.

Two wall scopes are reported:

* full trace wall: the complete recorded session span, with all LLM inference
  durations scaled by ``--llm-duration-scale``;
* eligible segment wall: only search-decision LLM lead plus the immediately
  following measurable visit stall, matching the older replay's scope.

Sessions are scheduled independently onto ``C`` task slots with event-driven
list scheduling.  There is no cross-task lockstep barrier.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import statistics
import sys
from typing import Any


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.traces import LLMCall, load_sessions  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredCandidate,
    ScoredWindow,
    collect_nested_oof_windows,
)
from run_pattern_v2_trace_timing_net_benefit import (  # noqa: E402
    DecisionTiming,
    ServiceEstimate,
    build_oof_service_estimates,
    collect_decision_timings,
    percentile,
    ratio,
    serial_visit_hit_saving,
    sha256_file,
    stable_tie,
)


SCHEMA = "paste_repro.pattern_v2_trace_multi_spec_wall.v4"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "results" / "pattern_v2_trace_multi_spec_wall"


@dataclass(frozen=True)
class SessionReplay:
    session_id: str
    baseline_full_wall_s: float
    treatment_full_wall_s: float
    baseline_segment_wall_s: float
    treatment_segment_wall_s: float
    baseline_visit_stall_s: float
    gross_saved_visit_stall_s: float
    net_saved_visit_stall_s: float
    authoritative_url_calls: int
    selected_speculations: int
    exact_url_hits: int
    visible_url_hits: int


def candidate_value(
    candidate: ScoredCandidate,
    estimate: ServiceEstimate,
    coordination_cost_s: float,
) -> float:
    return (
        candidate.exact_probability
        * estimate.overlap_for_url(candidate.pattern.url)
        - coordination_cost_s
    )


def select_per_task_candidates(
    window: ScoredWindow,
    estimate: ServiceEstimate,
    *,
    per_task_width: int,
    coordination_cost_s: float,
) -> tuple[ScoredCandidate, ...]:
    """Return up to N independently useful candidates for one decision."""

    if per_task_width <= 0:
        raise ValueError("per-task width must be positive")
    if not window.v2_gate:
        return ()
    ranked = sorted(
        window.candidates,
        key=lambda candidate: (
            -candidate_value(candidate, estimate, coordination_cost_s),
            -candidate.exact_probability,
            candidate.pattern.position,
            stable_tie(candidate),
        ),
    )
    return tuple(
        candidate
        for candidate in ranked
        if candidate_value(candidate, estimate, coordination_cost_s) > 0.0
    )[:per_task_width]


def session_full_walls(
    traces: Path, *, llm_duration_scale: float
) -> dict[str, float]:
    """Return full session spans after a duration-only LLM counterfactual."""

    walls: dict[str, float] = {}
    for session in load_sessions(traces):
        raw_wall_s = max(
            (float(event.timestamp_s) for event in session.events), default=0.0
        )
        removed_llm_s = (1.0 - llm_duration_scale) * sum(
            event.overlap_window_s
            for event in session.events
            if isinstance(event, LLMCall)
        )
        walls[session.session_id] = max(0.0, raw_wall_s - removed_llm_s)
    return walls


def build_session_replays(
    windows: Sequence[ScoredWindow],
    timings: Mapping[str, DecisionTiming],
    service_estimates: Mapping[str, ServiceEstimate],
    full_walls: Mapping[str, float],
    *,
    per_task_width: int,
    coordination_cost_s: float,
) -> tuple[SessionReplay, ...]:
    accumulators: dict[str, dict[str, float | int]] = {
        session_id: {
            "baseline_segment_wall_s": 0.0,
            "baseline_visit_stall_s": 0.0,
            "gross_saved_visit_stall_s": 0.0,
            "authoritative_url_calls": 0,
            "selected_speculations": 0,
            "exact_url_hits": 0,
            "visible_url_hits": 0,
        }
        for session_id in full_walls
    }
    for window in windows:
        timing = timings[window.decision_id]
        estimate = service_estimates[window.decision_id]
        selected = select_per_task_candidates(
            window,
            estimate,
            per_task_width=per_task_width,
            coordination_cost_s=coordination_cost_s,
        )
        row = accumulators[window.session_id]
        targets = len(window.executable_targets)
        stall_s = timing.visit_stall_s if targets else 0.0
        exact_urls = {
            candidate.pattern.url
            for candidate in selected
            if candidate.exact_match
        }
        exact_hit_mask = tuple(
            url in exact_urls for url in window.executable_targets
        )
        exact_hits = sum(exact_hit_mask)
        gross_saved_s = serial_visit_hit_saving(
            stall_s,
            timing.llm_overlap_s,
            exact_hit_mask,
            timing.visit_url_service_s,
        )
        row["baseline_segment_wall_s"] = float(
            row["baseline_segment_wall_s"]
        ) + timing.llm_overlap_s + stall_s
        row["baseline_visit_stall_s"] = float(
            row["baseline_visit_stall_s"]
        ) + stall_s
        row["gross_saved_visit_stall_s"] = float(
            row["gross_saved_visit_stall_s"]
        ) + gross_saved_s
        row["authoritative_url_calls"] = int(
            row["authoritative_url_calls"]
        ) + targets
        row["selected_speculations"] = int(
            row["selected_speculations"]
        ) + len(selected)
        row["exact_url_hits"] = int(row["exact_url_hits"]) + exact_hits
        row["visible_url_hits"] = int(row["visible_url_hits"]) + int(
            gross_saved_s > 0.0
        )

    result: list[SessionReplay] = []
    for session_id in sorted(full_walls):
        row = accumulators[session_id]
        selected = int(row["selected_speculations"])
        coordination_s = selected * coordination_cost_s
        baseline_segment_s = float(row["baseline_segment_wall_s"])
        gross_saved_s = float(row["gross_saved_visit_stall_s"])
        net_saved_s = gross_saved_s - coordination_s
        result.append(
            SessionReplay(
                session_id=session_id,
                baseline_full_wall_s=float(full_walls[session_id]),
                treatment_full_wall_s=max(
                    0.0, float(full_walls[session_id]) - net_saved_s
                ),
                baseline_segment_wall_s=baseline_segment_s,
                treatment_segment_wall_s=max(
                    0.0, baseline_segment_s - net_saved_s
                ),
                baseline_visit_stall_s=float(row["baseline_visit_stall_s"]),
                gross_saved_visit_stall_s=gross_saved_s,
                net_saved_visit_stall_s=net_saved_s,
                authoritative_url_calls=int(row["authoritative_url_calls"]),
                selected_speculations=selected,
                exact_url_hits=int(row["exact_url_hits"]),
                visible_url_hits=int(row["visible_url_hits"]),
            )
        )
    return tuple(result)


def seeded_order(session_id: str, seed: int) -> str:
    return hashlib.sha256(
        f"multi-spec-event-wall-v1\0{seed}\0{session_id}".encode("utf-8")
    ).hexdigest()


def list_schedule_makespan(
    sessions: Sequence[SessionReplay],
    *,
    concurrency: int,
    seed: int,
    duration_field: str,
) -> float:
    """Closed-loop event-driven scheduling without per-decision barriers."""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    workers = [0.0] * min(concurrency, max(1, len(sessions)))
    heapq.heapify(workers)
    ordered = sorted(
        sessions,
        key=lambda row: (seeded_order(row.session_id, seed), row.session_id),
    )
    for row in ordered:
        available_s = heapq.heappop(workers)
        duration_s = float(getattr(row, duration_field))
        heapq.heappush(workers, available_s + duration_s)
    return max(workers, default=0.0)


def summarize_width(
    sessions: Sequence[SessionReplay],
    *,
    per_task_width: int,
    concurrency: int,
    repetitions: int,
) -> dict[str, Any]:
    runs = []
    for seed in range(repetitions):
        baseline_full = list_schedule_makespan(
            sessions,
            concurrency=concurrency,
            seed=seed,
            duration_field="baseline_full_wall_s",
        )
        treatment_full = list_schedule_makespan(
            sessions,
            concurrency=concurrency,
            seed=seed,
            duration_field="treatment_full_wall_s",
        )
        baseline_segment = list_schedule_makespan(
            sessions,
            concurrency=concurrency,
            seed=seed,
            duration_field="baseline_segment_wall_s",
        )
        treatment_segment = list_schedule_makespan(
            sessions,
            concurrency=concurrency,
            seed=seed,
            duration_field="treatment_segment_wall_s",
        )
        runs.append(
            {
                "seed": seed,
                "baseline_full_wall_s": baseline_full,
                "treatment_full_wall_s": treatment_full,
                "baseline_segment_wall_s": baseline_segment,
                "treatment_segment_wall_s": treatment_segment,
            }
        )

    def mean_run(key: str) -> float:
        return statistics.fmean(float(row[key]) for row in runs)

    authoritative = sum(row.authoritative_url_calls for row in sessions)
    selected = sum(row.selected_speculations for row in sessions)
    exact_hits = sum(row.exact_url_hits for row in sessions)
    visible_hits = sum(row.visible_url_hits for row in sessions)
    baseline_visit = sum(row.baseline_visit_stall_s for row in sessions)
    gross_saved = sum(row.gross_saved_visit_stall_s for row in sessions)
    net_saved = sum(row.net_saved_visit_stall_s for row in sessions)
    baseline_full_sum = sum(row.baseline_full_wall_s for row in sessions)
    treatment_full_sum = sum(row.treatment_full_wall_s for row in sessions)
    baseline_segment_sum = sum(
        row.baseline_segment_wall_s for row in sessions
    )
    treatment_segment_sum = sum(
        row.treatment_segment_wall_s for row in sessions
    )
    baseline_full_wall = mean_run("baseline_full_wall_s")
    treatment_full_wall = mean_run("treatment_full_wall_s")
    baseline_segment_wall = mean_run("baseline_segment_wall_s")
    treatment_segment_wall = mean_run("treatment_segment_wall_s")
    physical_calls = authoritative - exact_hits + selected
    wall_speedups = [
        ratio(
            float(row["baseline_full_wall_s"])
            - float(row["treatment_full_wall_s"]),
            float(row["baseline_full_wall_s"]),
        )
        for row in runs
    ]
    return {
        "per_task_spec_width": per_task_width,
        "task_concurrency": concurrency,
        "repetitions": repetitions,
        "isolated_spec_slots_upper_bound": min(concurrency, len(sessions))
        * per_task_width,
        "authoritative_url_calls": authoritative,
        "selected_speculations": selected,
        "exact_url_hits": exact_hits,
        "visible_hit_decisions": visible_hits,
        "exact_authority_hit_rate": ratio(exact_hits, authoritative),
        "prediction_precision": ratio(exact_hits, selected),
        "physical_call_amplification": ratio(physical_calls, authoritative),
        "baseline_visit_stall_s": baseline_visit,
        "gross_saved_visit_stall_s": gross_saved,
        "net_saved_visit_stall_s": net_saved,
        "gross_visit_stall_reduction_fraction": ratio(
            gross_saved, baseline_visit
        ),
        "net_visit_stall_reduction_fraction": ratio(net_saved, baseline_visit),
        "mean_task_full_flow_reduction_fraction": ratio(
            baseline_full_sum - treatment_full_sum, baseline_full_sum
        ),
        "mean_task_segment_reduction_fraction": ratio(
            baseline_segment_sum - treatment_segment_sum,
            baseline_segment_sum,
        ),
        "event_full_baseline_wall_s": baseline_full_wall,
        "event_full_treatment_wall_s": treatment_full_wall,
        "event_full_wall_speedup_fraction": ratio(
            baseline_full_wall - treatment_full_wall, baseline_full_wall
        ),
        "event_segment_baseline_wall_s": baseline_segment_wall,
        "event_segment_treatment_wall_s": treatment_segment_wall,
        "event_segment_wall_speedup_fraction": ratio(
            baseline_segment_wall - treatment_segment_wall,
            baseline_segment_wall,
        ),
        "event_full_wall_speedup_sensitivity": {
            "p05": percentile(wall_speedups, 0.05),
            "p50": percentile(wall_speedups, 0.50),
            "p95": percentile(wall_speedups, 0.95),
        },
        "runs": runs,
    }


def render_report(payload: Mapping[str, Any]) -> str:
    config = payload["configuration"]
    lines = [
        "# Pattern-v2 per-task multi-spec real-trace wall replay",
        "",
        "Each admitted task decision may start multiple URL candidates concurrently.",
        "Speculation uses isolated capacity; wrong-call contention is not modeled.",
        "Sessions use event-driven list scheduling without lockstep decision barriers.",
        "",
        f"Widths=`{config['per_task_widths']}`, task concurrency=`{config['concurrencies']}`, "
        f"LLM duration scale=`{config['llm_duration_scale']}`, "
        f"coordination cost=`{config['coordination_cost_ms']} ms/start`.",
        "",
    ]
    for width_result in payload["width_results"]:
        width = width_result["per_task_spec_width"]
        lines.extend(
            [
                f"## Per-task speculative width={width}",
                "",
                "| Task C | Full-trace wall speedup | Eligible-segment wall speedup | Mean full task-flow reduction | Visit-stall reduction | Exact hit rate | Call amp. | Slot upper bound |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in width_result["concurrency_results"]:
            lines.append(
                f"| {row['task_concurrency']} "
                f"| {row['event_full_wall_speedup_fraction']:.2%} "
                f"| {row['event_segment_wall_speedup_fraction']:.2%} "
                f"| {row['mean_task_full_flow_reduction_fraction']:.2%} "
                f"| {row['net_visit_stall_reduction_fraction']:.2%} "
                f"| {row['exact_authority_hit_rate']:.2%} "
                f"| {row['physical_call_amplification']:.3f}x "
                f"| {row['isolated_spec_slots_upper_bound']} |"
            )
        lines.append("")
    lines.extend(
        [
            "The full-trace scope includes recorded search waits and every LLM turn.",
            "The eligible segment includes only search-decision LLM lead and immediate",
            "measurable visit stall. Multi-URL authority is replayed serially; concurrent",
            "exact speculations keep progressing during the LLM lead and while earlier",
            "authority URLs execute. Corrected traces provide per-URL service samples;",
            "legacy traces use equal-share atomic service as a fallback.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-task-widths", type=int, nargs="+", default=[1, 2, 3, 4, 5]
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--coordination-cost-ms", type=float, default=1.0)
    parser.add_argument("--domain-prior-strength", type=float, default=10.0)
    parser.add_argument("--llm-duration-scale", type=float, default=0.70)
    args = parser.parse_args()
    if any(value <= 0 for value in args.per_task_widths):
        parser.error("per-task widths must be positive")
    if any(value <= 0 for value in args.concurrencies):
        parser.error("concurrencies must be positive")
    if args.repetitions <= 0:
        parser.error("repetitions must be positive")
    if args.coordination_cost_ms < 0.0:
        parser.error("coordination cost must be non-negative")
    if args.domain_prior_strength < 0.0:
        parser.error("domain prior strength must be non-negative")
    if not 0.0 < args.llm_duration_scale <= 1.0:
        parser.error("LLM duration scale must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    windows, nested_oof = collect_nested_oof_windows(args.traces)
    timings = collect_decision_timings(
        args.traces, llm_duration_scale=args.llm_duration_scale
    )
    service_estimates, service_estimator = build_oof_service_estimates(
        windows,
        timings,
        domain_prior_strength=args.domain_prior_strength,
    )
    full_walls = session_full_walls(
        args.traces, llm_duration_scale=args.llm_duration_scale
    )
    width_results = []
    session_rows: dict[str, list[dict[str, Any]]] = {}
    for width in sorted(set(args.per_task_widths)):
        sessions = build_session_replays(
            windows,
            timings,
            service_estimates,
            full_walls,
            per_task_width=width,
            coordination_cost_s=args.coordination_cost_ms / 1000.0,
        )
        session_rows[str(width)] = [asdict(row) for row in sessions]
        width_results.append(
            {
                "per_task_spec_width": width,
                "concurrency_results": [
                    summarize_width(
                        sessions,
                        per_task_width=width,
                        concurrency=concurrency,
                        repetitions=args.repetitions,
                    )
                    for concurrency in args.concurrencies
                ],
            }
        )

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "traces": str(args.traces.resolve()),
            "per_task_widths": sorted(set(args.per_task_widths)),
            "concurrencies": args.concurrencies,
            "repetitions": args.repetitions,
            "coordination_cost_ms": args.coordination_cost_ms,
            "domain_prior_strength": args.domain_prior_strength,
            "llm_duration_scale": args.llm_duration_scale,
            "selection": "per-decision OOF expected-value Top-N",
            "capacity_model": "N isolated concurrent slots per active task",
            "wall_model": "event-driven closed-loop list scheduling",
            "multi_url_credit": (
                "event replay over serial per-URL authority service (equal-share "
                "legacy fallback); exact speculations run concurrently"
            ),
        },
        "source_sha256": {
            "runner": sha256_file(SCRIPT),
            "adaptive_load": sha256_file(
                SCRIPT.parent / "run_pattern_v2_adaptive_load.py"
            ),
            "trace_timing": sha256_file(
                SCRIPT.parent / "run_pattern_v2_trace_timing_net_benefit.py"
            ),
        },
        "nested_oof": nested_oof,
        "service_estimator": service_estimator,
        "session_rows": session_rows,
        "width_results": width_results,
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
