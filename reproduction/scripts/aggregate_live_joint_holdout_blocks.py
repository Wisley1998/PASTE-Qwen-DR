#!/usr/bin/env python3
"""Strict two-block E/F holdout aggregator for live tool--LLM development.

Expected layout under ``--root``::

    block1/E_off/cell/result.json
    block1/F_visit/cell/result.json
    block2/F_visit/cell/result.json
    block2/E_off/cell/result.json

Block 1 is deliberately E-before-F and block 2 F-before-E.  Each result has 24
independent sources and two replicas.  Replica observations are first folded
inside source and block, then the two block means are folded inside source;
the bootstrap therefore resamples exactly 24 independent sources, never 96
tasks or 48 source/block pseudo-replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence

from compare_live_joint_pair import _distribution, _mapping, _validate_run  # type: ignore
from compare_live_joint_dev_triplet import (  # type: ignore
    BOOTSTRAP_RESAMPLES,
    MAX_TOKEN_RELATIVE_DIFFERENCE,
    RunAudit,
    _audit_canary_non_speculation,
    _audit_http_attempt_logs,
    _bootstrap_effect,
    _cell_summary,
    _config_pair_audit,
    _effect_summary,
    _identity_pair_audit,
    _task_components,
    _validate_exact_counts,
    _write_json_atomic,
)


SCHEMA = "paste_repro.live_joint_holdout_two_block"
SCHEMA_VERSION = 1
BLOCK_IDS = ("block1", "block2")
EXPECTED_ORDER = {
    "block1": ("E", "F"),
    "block2": ("F", "E"),
}
EXPECTED_SOURCE_COUNT = 24
EXPECTED_REPLICAS = 2
EXPECTED_TASK_COUNT = 48
EXPECTED_LLM_COUNT = 144
EXPECTED_COMMIT_COUNT = 96
MIN_BLOCK_E2E_REDUCTION = 0.05
MIN_AGGREGATE_E2E_REDUCTION = 0.05
MIN_SOURCE_FASTER_FRACTION = 0.60
MAX_CANARY_MEAN_RATIO = 1.05
MAX_CANARY_P95_RATIO = 1.10
MAX_WASTE_WORKER_FRACTION = 0.45
MIN_SPEC_HIT_RATE = 0.10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _load_cell(root: Path, block_id: str, cell: str) -> RunAudit:
    directory = "E_off" if cell == "E" else "F_visit"
    path = root / block_id / directory / "cell" / "result.json"
    role = "baseline" if cell == "E" else "candidate"
    run = _validate_run(path, role=role)
    return RunAudit(
        # Reuse the A/N/V comparator's treatment definitions: N is the joint
        # demand-only treatment and V the joint visit-speculation treatment.
        cell="N" if cell == "E" else "V",
        ordinal=1 if block_id == "block1" else 2,
        run=run,
        exact_counts=_validate_exact_counts(
            run,
            expected_task_count=EXPECTED_TASK_COUNT,
            expected_llm_request_count=EXPECTED_LLM_COUNT,
            expected_authoritative_commit_count=EXPECTED_COMMIT_COUNT,
            expected_source_count=EXPECTED_SOURCE_COUNT,
            expected_replicas=EXPECTED_REPLICAS,
        ),
        http_attempts=_audit_http_attempt_logs(run),
        canary=_audit_canary_non_speculation(run),
    )


def _server_log_audit(audit: RunAudit) -> dict[str, Any]:
    path = audit.run.path.parent.parent / "server" / "vllm_8100.log"
    errors: list[str] = []
    if not path.is_file():
        return {
            "passed": False,
            "path": str(path),
            "errors": ["fresh-server log does not exist"],
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    pids = sorted(set(re.findall(r"APIServer pid=(\d+)", text)))
    if len(pids) != 1:
        errors.append("server log does not contain exactly one API server PID")
    required_markers = (
        "vLLM API server version",
        "[sched_policy_patch] installed policy=online_joint_pacer_v2",
        "Resolved architecture: Qwen3MoeForCausalLM",
        "Using max model len 16384",
    )
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        errors.append("server log is missing initialization markers: " + ", ".join(missing))
    return {
        "passed": not errors,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "api_server_pids": pids,
        "required_initialization_markers_present": not missing,
        "errors": errors,
    }


def _block_source_components(audit: RunAudit) -> dict[str, dict[str, float]]:
    observations: dict[str, dict[str, list[float]]] = {}
    for (source_id, _replica), values in _task_components(audit.run).items():
        target = observations.setdefault(source_id, {})
        for metric, value in values.items():
            target.setdefault(metric, []).append(value)
    folded: dict[str, dict[str, float]] = {}
    for source_id, values in sorted(observations.items()):
        counts = {len(rows) for rows in values.values()}
        if counts != {EXPECTED_REPLICAS}:
            raise ValueError(
                f"{audit.label}/{source_id} does not have exactly "
                f"{EXPECTED_REPLICAS} replicas per component"
            )
        folded[source_id] = {
            metric: statistics.fmean(rows)
            for metric, rows in sorted(values.items())
        }
    if len(folded) != EXPECTED_SOURCE_COUNT:
        raise ValueError(f"{audit.label} does not fold to 24 sources")
    return folded


def _aggregate_source_components(
    block_sources: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> dict[str, dict[str, float]]:
    source_sets = [set(rows) for rows in block_sources.values()]
    if not source_sets or any(rows != source_sets[0] for rows in source_sets[1:]):
        raise ValueError("source identities differ across blocks")
    folded: dict[str, dict[str, float]] = {}
    for source_id in sorted(source_sets[0]):
        metric_names = set(
            next(iter(block_sources.values()))[source_id]
        )
        folded[source_id] = {
            metric: statistics.fmean(
                float(block_sources[block_id][source_id][metric])
                for block_id in BLOCK_IDS
            )
            for metric in sorted(metric_names)
        }
    return folded


def _canary_records(audits: Sequence[RunAudit]) -> list[Mapping[str, Any]]:
    return [
        record
        for audit in audits
        for (_task_id, tool), record in audit.run.committed_by_task_tool.items()
        if tool == "visit" and record.get("canary") is True
    ]


def _canary_comparison(
    baseline: Sequence[RunAudit], candidate: Sequence[RunAudit]
) -> dict[str, Any]:
    base = _canary_records(baseline)
    cand = _canary_records(candidate)
    if len(base) != len(cand) or not base:
        raise ValueError("paired canary visit counts are empty or unequal")
    metrics: dict[str, Any] = {}
    for metric in ("exposed_wait_s", "queue_s", "service_s"):
        base_dist = _distribution([float(row[metric]) for row in base])
        cand_dist = _distribution([float(row[metric]) for row in cand])
        metrics[metric] = {
            "baseline": base_dist,
            "candidate": cand_dist,
            "mean_ratio": _ratio(
                float(cand_dist["mean"]), float(base_dist["mean"])
            ),
            "p95_ratio": _ratio(
                float(cand_dist["p95"]), float(base_dist["p95"])
            ),
        }
    exposed = metrics["exposed_wait_s"]
    gates = {
        "mean_ratio": _gate(
            exposed["mean_ratio"],
            f"<= {MAX_CANARY_MEAN_RATIO}",
            exposed["mean_ratio"] <= MAX_CANARY_MEAN_RATIO,
        ),
        "p95_ratio": _gate(
            exposed["p95_ratio"],
            f"<= {MAX_CANARY_P95_RATIO}",
            exposed["p95_ratio"] <= MAX_CANARY_P95_RATIO,
        ),
    }
    return {
        "count_per_treatment": len(base),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()),
    }


def _speculation_summary(audits: Sequence[RunAudit]) -> dict[str, Any]:
    summaries = [
        _mapping(audit.run.summary["tool"], "tool summary") for audit in audits
    ]
    totals = {
        key: sum(float(summary[key]) for summary in summaries)
        for key in (
            "speculative_admitted_count",
            "exact_hit_count",
            "queued_promotion_count",
            "running_promotion_count",
            "completed_reuse_count",
            "saved_service_s",
            "cancelled_physical_count",
            "expired_physical_count",
            "rejected_physical_count",
            "started_physical_job_count",
            "physical_http_attempt_count",
            "retried_physical_job_count",
            "physical_service_s",
            "wasted_speculative_service_s_from_records",
            "wasted_speculative_service_s_broker",
        )
    }
    eligible_all_tools = sum(
        sum(
            record.get("speculation_eligible") is True
            for record in audit.run.committed_by_task_tool.values()
        )
        for audit in audits
    )
    eligible = 0
    for audit in audits:
        mode = str(audit.run.config.get("speculation_mode"))
        enabled_tools = {
            "search": {"search", "search_visit"},
            "visit": {"visit", "search_visit"},
        }
        eligible += sum(
            record.get("speculation_eligible") is True
            and mode in enabled_tools.get(str(record.get("tool")), set())
            for record in audit.run.committed_by_task_tool.values()
        )
    exact_hit_rate = _ratio(totals["exact_hit_count"], float(eligible))
    wasted_fraction = _ratio(
        totals["wasted_speculative_service_s_broker"],
        totals["physical_service_s"],
    )
    gates = {
        "exact_hit_rate": _gate(
            exact_hit_rate,
            f">= {MIN_SPEC_HIT_RATE}",
            exact_hit_rate >= MIN_SPEC_HIT_RATE,
        ),
        "wasted_worker_fraction": _gate(
            wasted_fraction,
            f"<= {MAX_WASTE_WORKER_FRACTION}",
            wasted_fraction <= MAX_WASTE_WORKER_FRACTION,
        ),
        "zero_failed_or_retried_physical_work": _gate(
            {
                "retried": totals["retried_physical_job_count"],
                "cancelled": totals["cancelled_physical_count"],
                "expired": totals["expired_physical_count"],
            },
            "all are zero",
            totals["retried_physical_job_count"] == 0
            and totals["cancelled_physical_count"] == 0
            and totals["expired_physical_count"] == 0,
        ),
    }
    return {
        "totals": totals,
        "eligible_authoritative_commit_count_all_tools": eligible_all_tools,
        "eligible_authoritative_commit_count_for_enabled_speculation_mode": eligible,
        "exact_hit_rate": exact_hit_rate,
        "wasted_worker_fraction": wasted_fraction,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()),
    }


def _block_summary(
    block_id: str,
    e: RunAudit,
    f: RunAudit,
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    e_cell = _cell_summary([e])
    f_cell = _cell_summary([f])
    effect = _effect_summary(
        [e],
        [f],
        e_cell,
        f_cell,
        bootstrap_resamples=bootstrap_resamples,
    )
    starts = {
        "E": float(_mapping(e.run.payload["summary"], "E summary")["started_wall_s"]),
        "F": float(_mapping(f.run.payload["summary"], "F summary")["started_wall_s"]),
    }
    observed_order = tuple(sorted(starts, key=starts.get))
    reduction = effect["source_paired"]["component_comparisons"]["e2e_s"]
    canary = _canary_comparison([e], [f])
    speculation = _speculation_summary([f])
    gates = {
        "expected_execution_order": _gate(
            observed_order,
            str(EXPECTED_ORDER[block_id]),
            observed_order == EXPECTED_ORDER[block_id],
        ),
        "strict_input_config_identity": _gate(
            {
                "inputs": e.strict_evidence_eligible and f.strict_evidence_eligible,
                "config": effect["eligibility"]["all_cross_run_config_pairs_pass"],
                "identity": effect["eligibility"]["all_cross_run_identity_pairs_pass"],
            },
            "all true",
            e.strict_evidence_eligible
            and f.strict_evidence_eligible
            and effect["eligibility"]["all_cross_run_config_pairs_pass"]
            and effect["eligibility"]["all_cross_run_identity_pairs_pass"],
        ),
        "positive_preregistered_scale_e2e_gain": _gate(
            reduction["relative_reduction"],
            f">= {MIN_BLOCK_E2E_REDUCTION}",
            reduction["relative_reduction"] >= MIN_BLOCK_E2E_REDUCTION,
        ),
        "faster_source_fraction": _gate(
            effect["source_paired"]["faster_source_fraction"],
            f">= {MIN_SOURCE_FASTER_FRACTION}",
            effect["source_paired"]["faster_source_fraction"]
            >= MIN_SOURCE_FASTER_FRACTION,
        ),
        "canary": _gate(canary["passed"], "true", canary["passed"]),
        "speculation_quality": _gate(
            speculation["passed"], "true", speculation["passed"]
        ),
    }
    return {
        "block_id": block_id,
        "expected_execution_order": list(EXPECTED_ORDER[block_id]),
        "observed_execution_order": list(observed_order),
        "start_wall_s": starts,
        "E": e_cell["runs"][0],
        "F": f_cell["runs"][0],
        "source_folding": {
            "E": _block_source_components(e),
            "F": _block_source_components(f),
            "replicas_folded_per_source": EXPECTED_REPLICAS,
            "effective_source_count": EXPECTED_SOURCE_COUNT,
        },
        "effect": effect,
        "canary": canary,
        "speculation": speculation,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()),
    }


def aggregate_live_joint_holdout_blocks(
    root: Path,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    root = root.resolve()
    runs = {
        block_id: {
            cell: _load_cell(root, block_id, cell) for cell in ("E", "F")
        }
        for block_id in BLOCK_IDS
    }
    block_summaries = {
        block_id: _block_summary(
            block_id,
            runs[block_id]["E"],
            runs[block_id]["F"],
            bootstrap_resamples=bootstrap_resamples,
        )
        for block_id in BLOCK_IDS
    }
    e_runs = [runs[block_id]["E"] for block_id in BLOCK_IDS]
    f_runs = [runs[block_id]["F"] for block_id in BLOCK_IDS]
    e_cell = _cell_summary(e_runs)
    f_cell = _cell_summary(f_runs)
    aggregate_effect = _effect_summary(
        e_runs,
        f_runs,
        e_cell,
        f_cell,
        bootstrap_resamples=bootstrap_resamples,
    )

    block_e_sources = {
        block_id: _block_source_components(runs[block_id]["E"])
        for block_id in BLOCK_IDS
    }
    block_f_sources = {
        block_id: _block_source_components(runs[block_id]["F"])
        for block_id in BLOCK_IDS
    }
    aggregate_e_sources = _aggregate_source_components(block_e_sources)
    aggregate_f_sources = _aggregate_source_components(block_f_sources)
    explicit_bootstrap = _bootstrap_effect(
        aggregate_e_sources,
        aggregate_f_sources,
        resamples=bootstrap_resamples,
    )
    reported_bootstrap = aggregate_effect["source_paired"]["bootstrap"]
    if explicit_bootstrap != reported_bootstrap:
        raise ValueError("explicit replica-then-block folding differs from aggregate")

    config_audits = []
    identity_audits = []
    flattened = [
        (block_id, cell, runs[block_id][cell])
        for block_id in BLOCK_IDS
        for cell in ("E", "F")
    ]
    for left_index, (left_block, left_cell, left) in enumerate(flattened):
        for right_block, right_cell, right in flattened[left_index + 1 :]:
            config_audits.append(
                {
                    "pair": f"{left_block}/{left_cell}-{right_block}/{right_cell}",
                    "audit": _config_pair_audit(
                        left.run,
                        right.run,
                        left_cell="N" if left_cell == "E" else "V",
                        right_cell="N" if right_cell == "E" else "V",
                    ),
                }
            )
            identity_audits.append(
                {
                    "pair": f"{left_block}/{left_cell}-{right_block}/{right_cell}",
                    "audit": _identity_pair_audit(left.run, right.run),
                }
            )

    server_logs = {
        f"{block_id}/{cell}": _server_log_audit(runs[block_id][cell])
        for block_id in BLOCK_IDS
        for cell in ("E", "F")
    }
    server_pids = [
        entry["api_server_pids"][0]
        for entry in server_logs.values()
        if len(entry.get("api_server_pids", [])) == 1
    ]
    fresh_server_passed = (
        all(entry["passed"] for entry in server_logs.values())
        and len(server_pids) == 4
        and len(set(server_pids)) == 4
    )
    fresh_server = {
        "passed": fresh_server_passed,
        "four_distinct_api_server_pids": len(server_pids) == 4
        and len(set(server_pids)) == 4,
        "api_server_pids": server_pids,
        "logs": server_logs,
    }

    canary = _canary_comparison(e_runs, f_runs)
    speculation = _speculation_summary(f_runs)
    reduction = aggregate_effect["source_paired"]["component_comparisons"]["e2e_s"]
    aggregate_gates = {
        "both_blocks_pass": _gate(
            {block: summary["passed"] for block, summary in block_summaries.items()},
            "all true",
            all(summary["passed"] for summary in block_summaries.values()),
        ),
        "fresh_servers": _gate(fresh_server_passed, "true", fresh_server_passed),
        "same_config_and_code_sha": _gate(
            all(row["audit"]["passed"] for row in config_audits),
            "all six pair audits pass",
            all(row["audit"]["passed"] for row in config_audits),
        ),
        "same_frozen_identity": _gate(
            all(row["audit"]["passed"] for row in identity_audits),
            "all six pair audits pass",
            all(row["audit"]["passed"] for row in identity_audits),
        ),
        "aggregate_e2e_gain": _gate(
            reduction["relative_reduction"],
            f">= {MIN_AGGREGATE_E2E_REDUCTION}",
            reduction["relative_reduction"] >= MIN_AGGREGATE_E2E_REDUCTION,
        ),
        "aggregate_token_balance": _gate(
            aggregate_effect["tokens"]["max_absolute_relative_difference"],
            f"<= {MAX_TOKEN_RELATIVE_DIFFERENCE}",
            aggregate_effect["tokens"]["balance_gate"]["passed"],
        ),
        "aggregate_faster_source_fraction": _gate(
            aggregate_effect["source_paired"]["faster_source_fraction"],
            f">= {MIN_SOURCE_FASTER_FRACTION}",
            aggregate_effect["source_paired"]["faster_source_fraction"]
            >= MIN_SOURCE_FASTER_FRACTION,
        ),
        "aggregate_bootstrap_positive": _gate(
            explicit_bootstrap["e2e_relative_reduction_95_ci"][0],
            "> 0",
            explicit_bootstrap["e2e_relative_reduction_95_ci"][0] > 0.0,
        ),
        "tails_and_makespan": _gate(
            {
                key: value["passed"]
                for key, value in aggregate_effect["performance_gates"].items()
                if key in {"task_p95_ratio", "makespan_ratio"}
            },
            "all true",
            all(
                aggregate_effect["performance_gates"][key]["passed"]
                for key in ("task_p95_ratio", "makespan_ratio")
            ),
        ),
        "canary": _gate(canary["passed"], "true", canary["passed"]),
        "speculation_quality_and_waste": _gate(
            speculation["passed"], "true", speculation["passed"]
        ),
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "design": {
            "blocks": list(BLOCK_IDS),
            "orders": {key: list(value) for key, value in EXPECTED_ORDER.items()},
            "source_count": EXPECTED_SOURCE_COUNT,
            "replicas_per_source_per_block": EXPECTED_REPLICAS,
            "task_count_per_cell": EXPECTED_TASK_COUNT,
            "bootstrap_resamples": bootstrap_resamples,
            "folding_order": "replica_within_source_and_block_then_block_within_source",
            "effective_bootstrap_sample_size": EXPECTED_SOURCE_COUNT,
        },
        "fresh_server_evidence": fresh_server,
        "blocks": block_summaries,
        "cross_run_config_audits": config_audits,
        "cross_run_identity_audits": identity_audits,
        "aggregate": {
            "E": e_cell,
            "F": f_cell,
            "explicit_source_folding": {
                "E_by_source": aggregate_e_sources,
                "F_by_source": aggregate_f_sources,
                "bootstrap": explicit_bootstrap,
            },
            "effect": aggregate_effect,
            "canary": canary,
            "speculation": speculation,
            "gates": aggregate_gates,
            "strict_development_holdout_passed": all(
                gate["passed"] for gate in aggregate_gates.values()
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or (args.root / "strict_holdout_comparison.json")
    try:
        result = aggregate_live_joint_holdout_blocks(
            args.root,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        _write_json_atomic(output, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"strict holdout aggregation failed: {exc}", file=sys.stderr)
        return 2
    compact = {
        "blocks": {
            block: {
                "relative_reduction": summary["effect"]["source_paired"][
                    "component_comparisons"
                ]["e2e_s"]["relative_reduction"],
                "passed": summary["passed"],
            }
            for block, summary in result["blocks"].items()
        },
        "aggregate_relative_reduction": result["aggregate"]["effect"][
            "source_paired"
        ]["component_comparisons"]["e2e_s"]["relative_reduction"],
        "strict_development_holdout_passed": result["aggregate"][
            "strict_development_holdout_passed"
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
