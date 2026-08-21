#!/usr/bin/env python3
"""Compare one completed Joint candidate against a validated A/reference-D pair.

This is a read-only screening utility.  It is intended for the case where a
previous FCFS A cell is reused while a new Joint D configuration is screened.
Unlike ``summarize_paired_ad.py``, configuration drift between the reference D
and candidate D must match an explicit, exact allowlist.  Reusing A still does
not make the result a fresh-server pair or independent confirmation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from summarize_four_cell import load_fixed_manifest, load_run, percentile  # noqa: E402
from summarize_natural_queue_probe import summarize_probe  # noqa: E402
from summarize_paired_ad import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    TIE_EPSILON_S,
    _bootstrap_mean_ci,
    _load_raw_execution_accounting,
    _task_flow_by_trace,
    _validate_source_multiplicity,
)


SCHEMA = "paste_repro.candidate_d_comparison"
VERSION = 1
_MISSING = object()


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"request event {line_number} is not an object: {path}")
        events.append(value)
    if not events:
        raise ValueError(f"request event file is empty: {path}")
    return events


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    checked = [_finite_nonnegative(value, "sample value") for value in values]
    return {
        "mean": statistics.fmean(checked),
        "p50": percentile(checked, 0.50),
        "p95": percentile(checked, 0.95),
        "p99": percentile(checked, 0.99),
        "max": max(checked),
    }


def _parse_key_value(items: Sequence[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"{option} requires KEY=VALUE, got {item!r}")
        if key in parsed:
            raise ValueError(f"{option} repeats key {key}")
        parsed[key] = value
    return parsed


def _configuration_guard(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    allowed_differences: set[str],
    expected_reference: Mapping[str, str],
    expected_candidate: Mapping[str, str],
) -> dict[str, Any]:
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(reference) | set(candidate)):
        reference_value = reference.get(key, _MISSING)
        candidate_value = candidate.get(key, _MISSING)
        if reference_value == candidate_value:
            continue
        differences[key] = {
            "reference_present": reference_value is not _MISSING,
            "reference_value": None if reference_value is _MISSING else reference_value,
            "candidate_present": candidate_value is not _MISSING,
            "candidate_value": None if candidate_value is _MISSING else candidate_value,
        }
    actual = set(differences)
    if actual != allowed_differences:
        unexpected = sorted(actual - allowed_differences)
        unused = sorted(allowed_differences - actual)
        raise ValueError(
            "candidate/reference scheduler configuration diff does not exactly "
            f"match allowlist; unexpected={unexpected}, unused={unused}"
        )

    def check_expected(
        configuration: Mapping[str, Any],
        expected: Mapping[str, str],
        label: str,
    ) -> None:
        for key, expected_value in expected.items():
            if key not in configuration:
                raise ValueError(f"{label} scheduler configuration lacks {key}")
            if configuration[key] != expected_value:
                raise ValueError(
                    f"{label} scheduler configuration {key}={configuration[key]!r}; "
                    f"expected {expected_value!r}"
                )

    check_expected(reference, expected_reference, "reference D")
    check_expected(candidate, expected_candidate, "candidate D")
    return {
        "exact_allowlist_match": True,
        "allowed_difference_keys": sorted(allowed_differences),
        "actual_difference_keys": sorted(actual),
        "differences": differences,
        "expected_reference_values": dict(sorted(expected_reference.items())),
        "expected_candidate_values": dict(sorted(expected_candidate.items())),
    }


def _cell_metrics(
    run_path: Path,
    validated: Mapping[str, Any],
    flows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    events = _load_events(run_path / "request_events.jsonl")
    latencies = [
        _finite_nonnegative(event.get("latency_s"), "request latency")
        for event in events
    ]
    if any(event.get("ok") is not True for event in events):
        raise ValueError(f"candidate comparison requires all requests to succeed: {run_path}")
    task_flows = [float(row["task_flow_s"]) for row in flows.values()]
    public = validated["public"]
    request_count = int(public["request_count"])
    trace_count = int(public["trace_count"])
    if len(events) != request_count or len(task_flows) != trace_count:
        raise ValueError(f"validated event/task count mismatch: {run_path}")
    mean_latency = statistics.fmean(latencies)
    mean_queue = _finite_nonnegative(public["mean_queue_time_s"], "mean queue time")
    mean_nonqueue = mean_latency - mean_queue
    if mean_nonqueue < -1e-9:
        raise ValueError(f"mean queue time exceeds mean request latency: {run_path}")
    mean_nonqueue = max(0.0, mean_nonqueue)
    requests_per_task = request_count / trace_count
    noninitial_tool_wait_total = sum(
        _finite_nonnegative(event.get("scheduled_wait_s"), "scheduled wait")
        for event in events
        if int(event.get("call_index", -1)) > 0
    )
    request_contribution = mean_latency * requests_per_task
    queue_contribution = mean_queue * requests_per_task
    nonqueue_contribution = mean_nonqueue * requests_per_task
    tool_contribution = noninitial_tool_wait_total / trace_count
    task_mean = statistics.fmean(task_flows)
    residual = task_mean - request_contribution - tool_contribution
    if abs(residual) < 1e-9:
        residual = 0.0
    execution = _load_raw_execution_accounting(run_path, public)
    completion_tokens = execution["completion_tokens"]
    preemption = execution["preemption"]
    swap = execution["swap"]
    return {
        "run_path": run_path.resolve().as_posix(),
        "policy": public["policy"],
        "tool_overlap_mode": public["tool_overlap_mode"],
        "trace_count": trace_count,
        "request_count": request_count,
        "task_flow_time_s": _stats(task_flows),
        "task_makespan_s": _finite_nonnegative(
            public["task_makespan_s"], "task makespan"
        ),
        "instrumentation_wall_time_s": _finite_nonnegative(
            public["instrumentation_wall_time_s"], "instrumentation wall time"
        ),
        "request_latency_s": {
            **_stats(latencies),
            "count_gt_120_s": sum(value > 120.0 for value in latencies),
            "count_gt_240_s": sum(value > 240.0 for value in latencies),
            "fraction_gt_120_s": sum(value > 120.0 for value in latencies)
            / request_count,
            "fraction_gt_240_s": sum(value > 240.0 for value in latencies)
            / request_count,
        },
        "mean_queue_time_s": mean_queue,
        "mean_nonqueue_request_time_s": mean_nonqueue,
        "mean_task_component_s": {
            "request_latency": request_contribution,
            "queue": queue_contribution,
            "nonqueue_request": nonqueue_contribution,
            "noninitial_recorded_tool_wait": tool_contribution,
            "residual_harness_and_timing": residual,
            "identity": (
                "task_mean = queue + nonqueue_request + noninitial_recorded_tool_wait "
                "+ residual_harness_and_timing"
            ),
        },
        "retry_accounting": dict(public["retry_accounting"]),
        "execution": {
            "completion_tokens_total": completion_tokens["total"],
            "num_preemptions_total": preemption["num_preemptions_total"],
            "preemption_happened": preemption["preemption_happened"],
            "kv_swap_happened": swap["kv_swap_happened"],
            "kv_swap_event_count": swap["kv_swap_event_count"],
        },
    }


def _reduction(a: float, d: float) -> dict[str, float | None]:
    return {
        "a_minus_d_s": a - d,
        "relative_reduction": (a - d) / a if a else None,
    }


def _comparison(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    task = {
        statistic: _reduction(
            float(baseline_metrics["task_flow_time_s"][statistic]),
            float(candidate_metrics["task_flow_time_s"][statistic]),
        )
        for statistic in ("mean", "p50", "p95", "p99", "max")
    }
    request = {
        statistic: _reduction(
            float(baseline_metrics["request_latency_s"][statistic]),
            float(candidate_metrics["request_latency_s"][statistic]),
        )
        for statistic in ("mean", "p50", "p95", "p99", "max")
    }
    return {
        "definition": (
            f"{baseline_label} - {candidate_label}; positive means "
            f"{candidate_label} is lower/faster"
        ),
        "task_flow_time_s": task,
        "task_makespan_s": _reduction(
            float(baseline_metrics["task_makespan_s"]),
            float(candidate_metrics["task_makespan_s"]),
        ),
        "request_latency_s": request,
        "mean_queue_time_s": _reduction(
            float(baseline_metrics["mean_queue_time_s"]),
            float(candidate_metrics["mean_queue_time_s"]),
        ),
        "mean_nonqueue_request_time_s": _reduction(
            float(baseline_metrics["mean_nonqueue_request_time_s"]),
            float(candidate_metrics["mean_nonqueue_request_time_s"]),
        ),
    }


def _source_pairing(
    a_flows: Mapping[str, Mapping[str, Any]],
    d_flows: Mapping[str, Mapping[str, Any]],
    source_mapping: Mapping[str, str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    instance_deltas: list[float] = []
    for trace_id in sorted(a_flows):
        delta = float(a_flows[trace_id]["task_flow_s"]) - float(
            d_flows[trace_id]["task_flow_s"]
        )
        instance_deltas.append(delta)
        grouped.setdefault(str(source_mapping[trace_id]), []).append(
            {
                "trace_id": trace_id,
                "a_task_flow_s": float(a_flows[trace_id]["task_flow_s"]),
                "d_task_flow_s": float(d_flows[trace_id]["task_flow_s"]),
                "delta_s": delta,
            }
        )
    source_rows: list[dict[str, Any]] = []
    for source in sorted(grouped):
        instances = grouped[source]
        mean_delta = statistics.fmean(row["delta_s"] for row in instances)
        source_rows.append(
            {
                "source_session": source,
                "trace_ids": [row["trace_id"] for row in instances],
                "load_instance_count": len(instances),
                "a_task_flow_mean_s": statistics.fmean(
                    row["a_task_flow_s"] for row in instances
                ),
                "d_task_flow_mean_s": statistics.fmean(
                    row["d_task_flow_s"] for row in instances
                ),
                "delta_mean_s": mean_delta,
                "outcome": (
                    "d_faster"
                    if mean_delta > TIE_EPSILON_S
                    else "d_slower"
                    if mean_delta < -TIE_EPSILON_S
                    else "tie"
                ),
            }
        )
    source_deltas = [row["delta_mean_s"] for row in source_rows]

    def outcomes(values: Sequence[float]) -> dict[str, int | float]:
        wins = sum(value > TIE_EPSILON_S for value in values)
        losses = sum(value < -TIE_EPSILON_S for value in values)
        ties = len(values) - wins - losses
        return {
            "d_faster": wins,
            "tie": ties,
            "d_slower": losses,
            "d_faster_fraction": wins / len(values),
        }

    return {
        "definition": (
            "A-D task flow; deterministic load instances are averaged within each "
            "independent source before inference"
        ),
        "load_instance_count": len(instance_deltas),
        "independent_source_session_count": len(source_rows),
        "load_instance_outcomes": outcomes(instance_deltas),
        "source_session_outcomes": outcomes(source_deltas),
        "source_mean_saving_s": statistics.fmean(source_deltas),
        "independent_source_mean_bootstrap_95_ci_s": _bootstrap_mean_ci(
            source_deltas,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
        "source_sessions": source_rows,
    }


def _saving_decomposition(
    a_metrics: Mapping[str, Any],
    d_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    a_components = a_metrics["mean_task_component_s"]
    d_components = d_metrics["mean_task_component_s"]
    components = {
        key: float(a_components[key]) - float(d_components[key])
        for key in (
            "queue",
            "nonqueue_request",
            "noninitial_recorded_tool_wait",
            "residual_harness_and_timing",
        )
    }
    total = (
        float(a_metrics["task_flow_time_s"]["mean"])
        - float(d_metrics["task_flow_time_s"]["mean"])
    )
    reconstructed = sum(components.values())
    if not math.isclose(total, reconstructed, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError("task-saving decomposition does not reconstruct total")
    return {
        "definition": "A component - D component; positive contributes to D saving",
        "task_mean_saving_s": total,
        "components_s": components,
        "component_fraction_of_total_saving": {
            key: value / total if total else None for key, value in components.items()
        },
        "reconstructed_task_mean_saving_s": reconstructed,
    }


def summarize_candidate(
    *,
    manifest_path: Path,
    role: str,
    a_run: Path,
    reference_d_run: Path,
    candidate_d_run: Path,
    allowed_config_differences: set[str],
    expected_reference_config: Mapping[str, str] | None = None,
    expected_candidate_config: Mapping[str, str] | None = None,
    include_natural_queue_evidence: bool = True,
    require_natural_queue: bool = False,
) -> dict[str, Any]:
    paths = [a_run.resolve(), reference_d_run.resolve(), candidate_d_run.resolve()]
    if len(set(paths)) != 3:
        raise ValueError("A, reference D, and candidate D directories must be distinct")
    manifest = load_fixed_manifest(manifest_path, role)
    a = load_run(paths[0], "A", manifest["bindings"]["A"])
    reference_d = load_run(paths[1], "D", manifest["bindings"]["D"])
    candidate_d = load_run(paths[2], "D", manifest["bindings"]["D"])
    for label, run in (("reference D", reference_d), ("candidate D", candidate_d)):
        if a["identity_rows"] != run["identity_rows"]:
            raise ValueError(f"A/{label} request identity, prompt, or messages mismatch")
        if a["source_mapping"] != run["source_mapping"]:
            raise ValueError(f"A/{label} source-session mapping mismatch")
    source_counts = Counter(a["source_mapping"].values())
    _validate_source_multiplicity(
        source_counts,
        workload_invariants=manifest,
        replicate=1,
    )
    for field in (
        "speedup",
        "max_active_traces",
        "tool_wait_mode",
        "configured_max_request_attempts",
    ):
        values = {
            a["public"][field],
            reference_d["public"][field],
            candidate_d["public"][field],
        }
        if len(values) != 1:
            raise ValueError(f"A/reference/candidate configuration mismatch: {field}")
    if (
        a["public"]["scheduler_configuration"]
        != reference_d["public"]["scheduler_configuration"]
    ):
        raise ValueError(
            "reference A/D scheduler configurations must be identical for candidate screening"
        )
    if (
        reference_d["public"]["scheduler_calibration_workload_sha256"]
        != candidate_d["public"]["scheduler_calibration_workload_sha256"]
    ):
        raise ValueError("reference/candidate D calibration workload mismatch")
    config_guard = _configuration_guard(
        reference_d["public"]["scheduler_configuration"],
        candidate_d["public"]["scheduler_configuration"],
        allowed_differences=set(allowed_config_differences),
        expected_reference=expected_reference_config or {},
        expected_candidate=expected_candidate_config or {},
    )

    validated_runs = {
        "A": (paths[0], a),
        "reference_D": (paths[1], reference_d),
        "candidate_D": (paths[2], candidate_d),
    }
    flows = {
        name: _task_flow_by_trace(path, validated)
        for name, (path, validated) in validated_runs.items()
    }
    cells = {
        name: _cell_metrics(path, validated, flows[name])
        for name, (path, validated) in validated_runs.items()
    }
    natural_queue: dict[str, Any]
    if include_natural_queue_evidence:
        natural_queue = {
            name: summarize_probe(path) for name, (path, _) in validated_runs.items()
        }
        all_proven = all(
            evidence["sequence_capacity"]["natural_vllm_queue_proven"]
            for evidence in natural_queue.values()
        )
        if require_natural_queue and not all_proven:
            failed = [
                name
                for name, evidence in natural_queue.items()
                if not evidence["sequence_capacity"]["natural_vllm_queue_proven"]
            ]
            raise ValueError(f"natural vLLM queue requirement failed for {failed}")
        natural_queue = {"all_cells_proven": all_proven, "cells": natural_queue}
    else:
        if require_natural_queue:
            raise ValueError("cannot require natural queue when evidence is disabled")
        natural_queue = {"available": False, "reason": "disabled by caller"}

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "candidate_screen_reuses_previous_a_not_fresh_server_pair",
        "comparison_invariants": {
            "fixed_role": role,
            "fixed_workload_manifest_sha256": manifest["manifest_sha256"],
            "load_instance_count": manifest["load_instance_count"],
            "independent_source_session_count": manifest[
                "independent_source_session_count"
            ],
            "instances_per_source": manifest["instances_per_source"],
            "duplicates_are_not_independent": manifest[
                "duplicates_are_not_independent"
            ],
            "reference_a_d_scheduler_configuration_identical": True,
            "candidate_config_guard": config_guard,
        },
        "cells": cells,
        "comparisons": {
            "a_vs_reference_d": _comparison(
                cells["A"],
                cells["reference_D"],
                baseline_label="A",
                candidate_label="reference D",
            ),
            "a_vs_candidate_d": _comparison(
                cells["A"],
                cells["candidate_D"],
                baseline_label="A",
                candidate_label="candidate D",
            ),
            "reference_d_vs_candidate_d": _comparison(
                cells["reference_D"],
                cells["candidate_D"],
                baseline_label="reference D",
                candidate_label="candidate D",
            ),
        },
        "candidate_source_pairing": _source_pairing(
            flows["A"], flows["candidate_D"], a["source_mapping"]
        ),
        "candidate_task_saving_decomposition": _saving_decomposition(
            cells["A"], cells["candidate_D"]
        ),
        "natural_queue_evidence": natural_queue,
        "interpretation": (
            "The candidate is paired by identical deterministic workload identity, "
            "but A is reused. This is tuning/screening evidence, not a fresh-server "
            "paired replicate. Bootstrap resamples independent source-session means; "
            "deterministic duplicate load instances do not increase sample size."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("final", "heldout", "stress"), default="stress")
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--reference-d-run", type=Path, required=True)
    parser.add_argument("--candidate-d-run", type=Path, required=True)
    parser.add_argument(
        "--allow-config-diff",
        action="append",
        default=[],
        metavar="KEY",
        help=(
            "one scheduler_environment key allowed to differ between reference and "
            "candidate D; the actual diff set must equal this repeated allowlist"
        ),
    )
    parser.add_argument(
        "--expect-reference-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--expect-candidate-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--require-natural-queue",
        action="store_true",
        help="fail unless all three cells prove waiting below a non-binding sequence cap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the complete result atomically to this JSON path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    allowed = set(args.allow_config_diff)
    if len(allowed) != len(args.allow_config_diff):
        raise ValueError("--allow-config-diff contains duplicate keys")
    result = summarize_candidate(
        manifest_path=args.manifest,
        role=args.role,
        a_run=args.a_run,
        reference_d_run=args.reference_d_run,
        candidate_d_run=args.candidate_d_run,
        allowed_config_differences=allowed,
        expected_reference_config=_parse_key_value(
            args.expect_reference_config, "--expect-reference-config"
        ),
        expected_candidate_config=_parse_key_value(
            args.expect_candidate_config, "--expect-candidate-config"
        ),
        include_natural_queue_evidence=True,
        require_natural_queue=args.require_natural_queue,
    )
    if args.output is not None:
        write_json_atomic(args.output, result)
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
