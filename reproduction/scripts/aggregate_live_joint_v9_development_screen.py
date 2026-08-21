#!/usr/bin/env python3
"""Aggregate the prospective two-block E/F0/F1 v9 development screen.

Replicas are repeated measurements, not independent samples.  Each cell first
folds five replicas within each of 16 frozen tune sources, then folds the two
fresh-server reverse-order blocks.  Only the resulting 16 source estimates
enter the paired bootstrap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproduction/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_live_joint_pair as pair  # type: ignore
import validate_live_joint_v9_development_screen as validator  # type: ignore


SCHEMA = "paste_repro.live_joint_v9_development_screen"
SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_RESAMPLES = 10_000
CELL_IDS = ("E", "F0", "F1")
EXPECTED_ORDERS = (("E", "F0", "F1"), ("F1", "F0", "E"))
COMMON_CONFIG_EXCLUSIONS = frozenset(
    {
        "cell_label",
        "speculation_mode",
        "min_speculative_tool_workers",
        "expected_url_search_coverage",
        "formal_run",
    }
)


class DevelopmentScreenAggregationError(ValueError):
    """Raw development evidence cannot enter the prospective selection."""


def _gate(observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def _relative_difference(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right))
    return abs(left - right) / denominator if denominator else 0.0


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return pair._distribution([float(value) for value in values])


def _source_values(run: pair.ValidatedRun) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for (source_id, _replica), task in run.tasks_by_key.items():
        values[source_id].append(float(task["e2e_s"]))
    if (
        len(values) != validator.SOURCE_COUNT
        or any(len(rows) != validator.REPLICAS for rows in values.values())
    ):
        raise DevelopmentScreenAggregationError(
            "source estimator requires exactly 16 sources x 5 replicas"
        )
    return {
        source_id: statistics.fmean(observations)
        for source_id, observations in sorted(values.items())
    }


def _task_components(
    run: pair.ValidatedRun,
) -> dict[tuple[str, int], dict[str, float]]:
    rows: dict[tuple[str, int], dict[str, float]] = {}
    for key, task in run.tasks_by_key.items():
        task_id = str(task["task_id"])
        e2e_s = float(task["e2e_s"])
        llm_s = float(task["llm_duration_s"])
        search = run.committed_by_task_tool[(task_id, "search")]
        visit = run.committed_by_task_tool[(task_id, "visit")]
        search_exposed_s = float(search["exposed_wait_s"])
        visit_exposed_s = float(visit["exposed_wait_s"])
        tool_exposed_s = search_exposed_s + visit_exposed_s
        residual_s = e2e_s - llm_s - tool_exposed_s
        if residual_s < -0.05:
            raise DevelopmentScreenAggregationError(
                f"negative E2E component residual for {task_id}: {residual_s}"
            )
        rows[key] = {
            "e2e_s": e2e_s,
            "llm_s": llm_s,
            "tool_exposed_s": tool_exposed_s,
            "search_exposed_s": search_exposed_s,
            "visit_exposed_s": visit_exposed_s,
            "orchestration_residual_s": residual_s,
        }
    return rows


def _fold_source_metric(
    runs: Mapping[str, Mapping[str, pair.ValidatedRun]],
    *,
    cell: str,
    metric: str,
) -> dict[str, float]:
    """Fold five replicas inside block, then two blocks, preserving n=16."""

    block_source: dict[str, dict[str, float]] = {}
    for block_id in sorted(runs):
        observations: dict[str, list[float]] = defaultdict(list)
        if metric == "e2e_s":
            for (source_id, _replica), task in runs[block_id][cell].tasks_by_key.items():
                observations[source_id].append(float(task["e2e_s"]))
        else:
            for (source_id, _replica), components in _task_components(
                runs[block_id][cell]
            ).items():
                observations[source_id].append(float(components[metric]))
        if (
            len(observations) != validator.SOURCE_COUNT
            or any(len(rows) != validator.REPLICAS for rows in observations.values())
        ):
            raise DevelopmentScreenAggregationError(
                f"{block_id}/{cell}/{metric} is not 16x5"
            )
        block_source[block_id] = {
            source_id: statistics.fmean(values)
            for source_id, values in sorted(observations.items())
        }
    source_sets = [set(rows) for rows in block_source.values()]
    if len(source_sets) != 2 or source_sets[0] != source_sets[1]:
        raise DevelopmentScreenAggregationError(
            f"{cell}/{metric} source identities differ across blocks"
        )
    return {
        source_id: statistics.fmean(
            block_source[block_id][source_id] for block_id in sorted(block_source)
        )
        for source_id in sorted(source_sets[0])
    }


def _bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    resamples: int,
) -> dict[str, Any]:
    if set(baseline) != set(candidate) or len(baseline) != validator.SOURCE_COUNT:
        raise DevelopmentScreenAggregationError(
            "bootstrap requires the same 16 independent sources"
        )
    if resamples <= 0:
        raise DevelopmentScreenAggregationError("bootstrap resamples must be positive")
    source_ids = sorted(baseline)
    reductions = {
        source_id: baseline[source_id] - candidate[source_id]
        for source_id in source_ids
    }
    rng = random.Random(BOOTSTRAP_SEED)
    absolute: list[float] = []
    relative: list[float] = []
    for _ in range(resamples):
        sample = [source_ids[rng.randrange(len(source_ids))] for _ in source_ids]
        baseline_mean = statistics.fmean(baseline[source] for source in sample)
        candidate_mean = statistics.fmean(candidate[source] for source in sample)
        absolute.append(baseline_mean - candidate_mean)
        relative.append(
            (baseline_mean - candidate_mean) / baseline_mean
            if baseline_mean
            else 0.0
        )
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": resamples,
        "sampling_unit": (
            "16 independent source means after 5-replica then 2-block folding"
        ),
        "sample_size": len(source_ids),
        "absolute_reduction_s_95_ci": [
            pair._percentile(absolute, 0.025),
            pair._percentile(absolute, 0.975),
        ],
        "relative_reduction_95_ci": [
            pair._percentile(relative, 0.025),
            pair._percentile(relative, 0.975),
        ],
        "source_reduction_s": reductions,
    }


def _effect(
    runs: Mapping[str, Mapping[str, pair.ValidatedRun]],
    aggregate_sources: Mapping[str, Mapping[str, float]],
    *,
    baseline_cell: str,
    candidate_cell: str,
    resamples: int,
) -> dict[str, Any]:
    baseline = aggregate_sources[baseline_cell]
    candidate = aggregate_sources[candidate_cell]
    baseline_mean = statistics.fmean(baseline.values())
    candidate_mean = statistics.fmean(candidate.values())
    blocks: list[dict[str, Any]] = []
    for block_id in sorted(runs):
        block_baseline = _source_values(runs[block_id][baseline_cell])
        block_candidate = _source_values(runs[block_id][candidate_cell])
        left = statistics.fmean(block_baseline.values())
        right = statistics.fmean(block_candidate.values())
        blocks.append(
            {
                "block_id": block_id,
                "baseline_mean_s": left,
                "candidate_mean_s": right,
                "absolute_reduction_s": left - right,
                "relative_reduction": (left - right) / left if left else 0.0,
                "faster_source_count": sum(
                    block_baseline[source] > block_candidate[source]
                    for source in block_baseline
                ),
            }
        )
    reductions = {
        source: baseline[source] - candidate[source] for source in baseline
    }
    return {
        "baseline_cell": baseline_cell,
        "candidate_cell": candidate_cell,
        "baseline_mean_s": baseline_mean,
        "candidate_mean_s": candidate_mean,
        "mean_absolute_reduction_s": baseline_mean - candidate_mean,
        "aggregate_relative_reduction": (
            (baseline_mean - candidate_mean) / baseline_mean
            if baseline_mean
            else 0.0
        ),
        "faster_source_count": sum(value > 0 for value in reductions.values()),
        "faster_source_fraction": sum(value > 0 for value in reductions.values())
        / len(reductions),
        "every_block_mean_reduction_positive": all(
            row["absolute_reduction_s"] > 0 for row in blocks
        ),
        "blocks": blocks,
        "source_reduction_distribution_s": _distribution(list(reductions.values())),
        "bootstrap": _bootstrap(baseline, candidate, resamples=resamples),
    }


def _component_decomposition(
    runs: Mapping[str, Mapping[str, pair.ValidatedRun]],
    *,
    candidate_cell: str,
) -> dict[str, Any]:
    metrics = (
        "e2e_s",
        "llm_s",
        "tool_exposed_s",
        "search_exposed_s",
        "visit_exposed_s",
        "orchestration_residual_s",
    )
    by_cell = {
        cell: {
            metric: _fold_source_metric(runs, cell=cell, metric=metric)
            for metric in metrics
        }
        for cell in ("E", candidate_cell)
    }
    means = {
        cell: {
            metric: statistics.fmean(by_cell[cell][metric].values())
            for metric in metrics
        }
        for cell in by_cell
    }
    saving = {
        metric: means["E"][metric] - means[candidate_cell][metric]
        for metric in metrics
    }
    e_llm = means["E"]["llm_s"]
    net = saving["e2e_s"]
    tool = saving["tool_exposed_s"]
    return {
        "definition": (
            "task E2E = three LLM durations + committed search/visit exposed "
            "waits + residual; fold replicas within block then blocks within source"
        ),
        "source_count": validator.SOURCE_COUNT,
        "mean_components_s": means,
        "mean_saving_E_minus_candidate_s": saving,
        "candidate_llm_component_speedup_fraction": (
            (e_llm - means[candidate_cell]["llm_s"]) / e_llm
            if e_llm
            else None
        ),
        "tool_exposed_wait_saving_to_net_e2e_saving_ratio": (
            tool / net if net > 0 else None
        ),
    }


def _reservation_audit(run: pair.ValidatedRun, *, label: str) -> dict[str, Any]:
    try:
        replay = validator.validate_dispatch_ledger(run, label=label)
    except validator.DevelopmentScreenValidationError as exc:
        raise DevelopmentScreenAggregationError(str(exc)) from exc
    if replay["min_speculative_tool_workers"] != 1:
        raise DevelopmentScreenAggregationError(f"{label} min reservation is not 1")
    return replay


def _no_reservation_audit(run: pair.ValidatedRun, *, label: str) -> None:
    try:
        replay = validator.validate_dispatch_ledger(run, label=label)
    except validator.DevelopmentScreenValidationError as exc:
        raise DevelopmentScreenAggregationError(str(exc)) from exc
    if (
        replay["min_speculative_tool_workers"] != 0
        or replay["reserved_speculative_dispatch_count"] != 0
        or replay["authoritative_repayment_count"] != 0
    ):
        raise DevelopmentScreenAggregationError(f"{label} unexpectedly reserves")


def _common_config_identity(
    runs: Mapping[str, Mapping[str, pair.ValidatedRun]],
) -> dict[str, Any]:
    canonical: str | None = None
    digest: str | None = None
    selected_workload_sha: str | None = None
    rows: dict[str, Any] = {}
    for block_id in sorted(runs):
        for cell in CELL_IDS:
            run = runs[block_id][cell]
            common = {
                key: value
                for key, value in run.config.items()
                if key not in COMMON_CONFIG_EXCLUSIONS
            }
            encoded = json.dumps(
                common,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            current_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if canonical is None:
                canonical, digest = encoded, current_digest
            elif encoded != canonical:
                raise DevelopmentScreenAggregationError(
                    f"{block_id}/{cell} differs outside registered treatment factors"
                )
            current_workload = str(run.config.get("selected_workload_sha256"))
            if selected_workload_sha is None:
                selected_workload_sha = current_workload
            elif current_workload != selected_workload_sha:
                raise DevelopmentScreenAggregationError(
                    "selected workload identity differs across stage-1 cells"
                )
            rows[f"{block_id}/{cell}"] = {
                "common_config_sha256": current_digest,
                "selected_workload_sha256": current_workload,
                "speculation_mode": run.config["speculation_mode"],
                "min_speculative_tool_workers": run.config[
                    "min_speculative_tool_workers"
                ],
            }
    return {
        "passed": True,
        "allowed_differing_fields": sorted(COMMON_CONFIG_EXCLUSIONS),
        "common_config_sha256": digest,
        "selected_workload_sha256": selected_workload_sha,
        "cells": rows,
    }


def _candidate_gates(
    *,
    cell: str,
    effect: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    combined_p95: Mapping[str, float],
    mean_makespan: Mapping[str, float],
    token_difference: float,
    block_token_difference: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    llm_speedup = decomposition["candidate_llm_component_speedup_fraction"]
    savings = decomposition["mean_saving_E_minus_candidate_s"]
    net_saving = float(savings["e2e_s"])
    tool_saving = float(savings["tool_exposed_s"])
    return {
        "mean_reduction_at_least_5pct": _gate(
            effect["aggregate_relative_reduction"],
            ">=0.05",
            float(effect["aggregate_relative_reduction"]) >= 0.05,
        ),
        "both_blocks_positive": _gate(
            [row["absolute_reduction_s"] for row in effect["blocks"]],
            "both >0",
            bool(effect["every_block_mean_reduction_positive"]),
        ),
        "at_least_13_of_16_sources_faster": _gate(
            effect["faster_source_count"],
            ">=13",
            int(effect["faster_source_count"]) >= 13,
        ),
        "bootstrap_absolute_lower_bound_positive": _gate(
            effect["bootstrap"]["absolute_reduction_s_95_ci"][0],
            ">0",
            float(effect["bootstrap"]["absolute_reduction_s_95_ci"][0]) > 0,
        ),
        "task_p95_not_worse": _gate(
            {"E": combined_p95["E"], cell: combined_p95[cell]},
            f"{cell}<=E",
            combined_p95[cell] <= combined_p95["E"],
        ),
        "makespan_at_most_1p03_E": _gate(
            {"E": mean_makespan["E"], cell: mean_makespan[cell]},
            f"{cell}<=1.03*E",
            mean_makespan[cell] <= 1.03 * mean_makespan["E"],
        ),
        "completion_token_difference_below_1pct": _gate(
            {"aggregate": token_difference, "by_block": block_token_difference},
            "aggregate and each block <0.01",
            token_difference < 0.01
            and all(value < 0.01 for value in block_token_difference.values()),
        ),
        "llm_component_not_more_than_1pct_faster": _gate(
            {
                "speedup_fraction": llm_speedup,
                "mean_components_s": decomposition["mean_components_s"],
            },
            "(E_llm-candidate_llm)/E_llm <=0.01",
            llm_speedup is not None and float(llm_speedup) <= 0.01,
        ),
        "tool_exposed_saving_explains_net": _gate(
            {
                "net_e2e_saving_s": net_saving,
                "tool_exposed_wait_saving_s": tool_saving,
                "ratio": decomposition[
                    "tool_exposed_wait_saving_to_net_e2e_saving_ratio"
                ],
            },
            "net>0 and tool exposed-wait saving >= net E2E saving",
            net_saving > 0 and tool_saving >= net_saving,
        ),
    }


def select_policy(
    *,
    f0_passed: bool,
    f1_base_passed: bool,
    f1_incremental_passed: bool,
) -> str | None:
    if f1_base_passed and f1_incremental_passed:
        return "F1"
    if f0_passed:
        return "F0"
    return None


def aggregate_development_screen(
    blocks: Sequence[tuple[str, Mapping[str, Path]]],
    *,
    selected_visit_interval_s: float,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if selected_visit_interval_s not in validator.TRANSPORT_LADDER_S:
        raise DevelopmentScreenAggregationError("selected interval is unregistered")
    if len(blocks) != 2:
        raise DevelopmentScreenAggregationError("screen requires exactly two blocks")
    if bootstrap_resamples <= 0:
        raise DevelopmentScreenAggregationError("bootstrap resamples must be positive")

    runs: dict[str, dict[str, pair.ValidatedRun]] = {}
    validations: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    server_ids: set[str] = set()
    for block_number, (block_id, raw_paths) in enumerate(blocks):
        if set(raw_paths) != set(CELL_IDS):
            raise DevelopmentScreenAggregationError(
                f"{block_id} must contain E/F0/F1 exactly once"
            )
        block_runs: dict[str, pair.ValidatedRun] = {}
        order: dict[int, str] = {}
        validations[block_id] = {}
        inputs[block_id] = {}
        for cell in CELL_IDS:
            result_path = Path(raw_paths[cell]).resolve()
            timeline_path = result_path.parent / "queue_timeline.jsonl"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            config = payload.get("config", {})
            formal_run = config.get("formal_run", {})
            order_index = formal_run.get("order_index")
            server_id = formal_run.get("server_instance_id")
            if (
                not isinstance(order_index, int)
                or isinstance(order_index, bool)
                or order_index not in range(3)
                or not isinstance(server_id, str)
                or not server_id
            ):
                raise DevelopmentScreenAggregationError(
                    f"{block_id}/{cell} lacks stage-1 fresh-server metadata"
                )
            if order_index in order or server_id in server_ids:
                raise DevelopmentScreenAggregationError(
                    f"{block_id}/{cell} reuses order index or server"
                )
            validation = validator.validate_cell_result(
                result_path=result_path,
                timeline_path=timeline_path,
                cell=cell,
                block_id=block_id,
                order_index=order_index,
                server_instance_id=server_id,
                visit_interval_s=selected_visit_interval_s,
                stage="stage1",
            )
            if validation["accepted"] is not True:
                raise DevelopmentScreenAggregationError(
                    f"{block_id}/{cell} strict validation did not accept"
                )
            role = "baseline" if cell == "E" else "candidate"
            run = pair._validate_run(
                result_path, role=role, timeline_override=timeline_path
            )
            block_runs[cell] = run
            validations[block_id][cell] = validation
            inputs[block_id][cell] = {
                "result_path": str(result_path),
                "result_sha256": validation["result_sha256"],
                "timeline_path": str(timeline_path),
                "timeline_sha256": validation["timeline_sha256"],
                "server_instance_id": server_id,
                "order_index": order_index,
            }
            order[order_index] = cell
            server_ids.add(server_id)
        observed_order = tuple(order[index] for index in range(3))
        if observed_order != EXPECTED_ORDERS[block_number]:
            raise DevelopmentScreenAggregationError(
                f"block {block_number + 1} order {observed_order} is not preregistered"
            )
        runs[block_id] = block_runs
    if len(server_ids) != 6:
        raise DevelopmentScreenAggregationError("six unique fresh servers required")

    common_identity = _common_config_identity(runs)
    reservation: dict[str, Any] = {}
    for block_id in sorted(runs):
        _no_reservation_audit(runs[block_id]["E"], label=f"{block_id}/E")
        _no_reservation_audit(runs[block_id]["F0"], label=f"{block_id}/F0")
        reservation[block_id] = _reservation_audit(
            runs[block_id]["F1"], label=f"{block_id}/F1"
        )

    aggregate_sources = {
        cell: _fold_source_metric(runs, cell=cell, metric="e2e_s")
        for cell in CELL_IDS
    }
    effects = {
        cell: _effect(
            runs,
            aggregate_sources,
            baseline_cell="E",
            candidate_cell=cell,
            resamples=bootstrap_resamples,
        )
        for cell in ("F0", "F1")
    }
    incremental = _effect(
        runs,
        aggregate_sources,
        baseline_cell="F0",
        candidate_cell="F1",
        resamples=bootstrap_resamples,
    )
    decomposition = {
        cell: _component_decomposition(runs, candidate_cell=cell)
        for cell in ("F0", "F1")
    }
    combined_task = {
        cell: [
            float(task["e2e_s"])
            for block_id in sorted(runs)
            for task in runs[block_id][cell].tasks_by_key.values()
        ]
        for cell in CELL_IDS
    }
    combined_p95 = {
        cell: pair._percentile(values, 0.95)
        for cell, values in combined_task.items()
    }
    mean_makespan = {
        cell: statistics.fmean(
            float(runs[block_id][cell].summary["task_completion_makespan_s"])
            for block_id in runs
        )
        for cell in CELL_IDS
    }
    completion_tokens = {
        cell: sum(
            sum(
                int(event["usage"]["completion_tokens"])
                for events in runs[block_id][cell].llm_by_task.values()
                for event in events
            )
            for block_id in runs
        )
        for cell in CELL_IDS
    }
    block_completion_tokens = {
        block_id: {
            cell: sum(
                int(event["usage"]["completion_tokens"])
                for events in runs[block_id][cell].llm_by_task.values()
                for event in events
            )
            for cell in CELL_IDS
        }
        for block_id in runs
    }
    token_difference = {
        cell: _relative_difference(completion_tokens["E"], completion_tokens[cell])
        for cell in ("F0", "F1")
    }
    block_token_difference = {
        cell: {
            block_id: _relative_difference(
                block_completion_tokens[block_id]["E"],
                block_completion_tokens[block_id][cell],
            )
            for block_id in runs
        }
        for cell in ("F0", "F1")
    }
    candidate_gates = {
        cell: _candidate_gates(
            cell=cell,
            effect=effects[cell],
            decomposition=decomposition[cell],
            combined_p95=combined_p95,
            mean_makespan=mean_makespan,
            token_difference=token_difference[cell],
            block_token_difference=block_token_difference[cell],
        )
        for cell in ("F0", "F1")
    }
    candidate_passed = {
        cell: all(gate["passed"] for gate in gates.values())
        for cell, gates in candidate_gates.items()
    }
    incremental_gates = {
        "additional_mean_reduction_at_least_2pct": _gate(
            incremental["aggregate_relative_reduction"],
            ">=0.02",
            incremental["aggregate_relative_reduction"] >= 0.02,
        ),
        "both_blocks_positive": _gate(
            [row["absolute_reduction_s"] for row in incremental["blocks"]],
            "both >0",
            incremental["every_block_mean_reduction_positive"],
        ),
        "reservation_dispatch_repayment_and_ready_hits_each_block": _gate(
            reservation,
            (
                "every F1 block: min=1, debt always 0/1 and final 0, "
                ">=6 reserved dispatches with matching repayments, >=6 completed-reuse ready hits"
            ),
            all(
                row["reserved_speculative_dispatch_count"] >= 6
                and row["authoritative_repayment_count"]
                == row["reserved_speculative_dispatch_count"]
                and row["completed_reuse_ready_hit_count"] >= 6
                and row["debt_domain"] == [0, 1]
                and row["final_debt_zero"] is True
                and row["all_dispatch_rows_causally_replayed"] is True
                for row in reservation.values()
            ),
        ),
    }
    incremental_passed = all(
        gate["passed"] for gate in incremental_gates.values()
    )
    selected = select_policy(
        f0_passed=candidate_passed["F0"],
        f1_base_passed=candidate_passed["F1"],
        f1_incremental_passed=incremental_passed,
    )
    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "valid": True,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "formal_promotion_claim": False,
        "selected_policy": selected,
        "development_selection_passed": selected is not None,
        "no_winner": selected is None,
        "selected_visit_interval_s": selected_visit_interval_s,
        "preregistered_orders": [list(order) for order in EXPECTED_ORDERS],
        "inputs": inputs,
        "strict_cell_validations": validations,
        "fresh_server_instance_count": len(server_ids),
        "common_code_and_config_identity": common_identity,
        "estimator": {
            "independent_source_count": validator.SOURCE_COUNT,
            "replicas_per_source_per_block": validator.REPLICAS,
            "fresh_server_block_count": 2,
            "source_estimator": (
                "mean five replicas within source/cell/block, then mean two "
                "blocks within source; bootstrap the 16 paired source means"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": bootstrap_resamples,
            "effective_bootstrap_sample_size": validator.SOURCE_COUNT,
        },
        "aggregate_source_e2e_s": aggregate_sources,
        "effects_E_to_candidate": effects,
        "effect_F0_to_F1": incremental,
        "component_decomposition_E_to_candidate": decomposition,
        "combined_task_p95_s": combined_p95,
        "mean_task_completion_makespan_s": mean_makespan,
        "completion_tokens": completion_tokens,
        "completion_tokens_by_block": block_completion_tokens,
        "completion_token_relative_difference": token_difference,
        "completion_token_relative_difference_by_block": block_token_difference,
        "F1_reservation_audit_by_block": reservation,
        "candidate_gates": candidate_gates,
        "candidate_passed": candidate_passed,
        "F1_incremental_gates": incremental_gates,
        "F1_incremental_passed": incremental_passed,
        "selection_rule": (
            "select F1 only if all E->F1 gates and all F0->F1 incremental/"
            "reservation gates pass; otherwise select F0 if all E->F0 gates "
            "pass; otherwise no winner"
        ),
    }


def _parse_block(raw: Sequence[str]) -> tuple[str, Mapping[str, Path]]:
    if len(raw) != 4:
        raise argparse.ArgumentTypeError("--block requires ID E_RESULT F0_RESULT F1_RESULT")
    return raw[0], {cell: Path(path) for cell, path in zip(CELL_IDS, raw[1:])}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        nargs=4,
        action="append",
        metavar=("ID", "E_RESULT", "F0_RESULT", "F1_RESULT"),
        required=True,
    )
    parser.add_argument(
        "--selected-visit-interval-s",
        type=float,
        choices=validator.TRANSPORT_LADDER_S,
        required=True,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    blocks = [_parse_block(raw) for raw in args.block]
    result = aggregate_development_screen(
        blocks,
        selected_visit_interval_s=args.selected_visit_interval_s,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    return 0 if result["development_selection_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DevelopmentScreenAggregationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
