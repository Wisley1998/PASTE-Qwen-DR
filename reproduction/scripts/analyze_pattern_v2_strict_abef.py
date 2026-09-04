#!/usr/bin/env python3
"""Audit and analyze Pattern V2 strict A/B/E/F live results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
for import_root in (REPRODUCTION_ROOT, SCRIPT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from paste_repro.strict_trace_runtime import (  # noqa: E402
    signed_payload,
    validate_signed_payload,
)
from run_pattern_v2_strict_abef import CELL_SPECS, RESULT_SCHEMA, file_sha256  # noqa: E402


SCHEMA = "paste_repro.pattern_v2_strict_abef_analysis.v1"
CONTRASTS = {
    "B_vs_A_tool_only": ("A", "B"),
    "E_vs_A_scheduler_only": ("A", "E"),
    "F_vs_E_tool_incremental": ("E", "F"),
    "F_vs_A_combined": ("A", "F"),
    "F_vs_B_scheduler_incremental": ("B", "F"),
}
COMMON_SUMMARY_FIELDS = (
    "evaluation_mode",
    "formal_workload",
    "workload_contract",
    "claim_scope",
    "confirmatory_claim_allowed",
    "public_plan_sha256",
    "sealed_plan_sha256",
    "predictor_artifact_sha256",
    "predictor_disclosure",
    "duration_predictor_artifact_sha256",
    "tail_predictor_artifact_sha256",
    "physical_service_clock_sha256",
    "frozen_input_file_sha256",
    "tasks",
    "source_roots",
    "requests",
    "tool_events",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _speedup(baseline: float, treatment: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline must be positive")
    return (baseline - treatment) / baseline


def _validate_result_directory(client: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = validate_signed_payload(
        _read_json(client / "summary.json"), "result_sha256", label=str(client)
    )
    manifest = validate_signed_payload(
        _read_json(client / "result_manifest.json"),
        "manifest_sha256",
        label=f"{client} manifest",
    )
    if summary.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"unsupported result schema in {client}")
    if manifest.get("result_sha256") != summary.get("result_sha256"):
        raise ValueError(f"result/manifest binding mismatch in {client}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"invalid result file manifest in {client}")
    for name, expected in files.items():
        path = client / str(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"result file hash mismatch: {path}")
    tasks = _read_json(client / "task_results.json")
    requests = _read_json(client / "request_events.json")
    tools = _read_json(client / "tool_events.json")
    if not all(isinstance(rows, list) for rows in (tasks, requests, tools)):
        raise ValueError(f"invalid event arrays in {client}")
    if (
        summary.get("formal_workload") is not True
        or summary.get("confirmatory_claim_allowed") is not False
        or summary.get("claim_scope") != "retrospective_internal_holdout"
        or summary.get("tasks") != 210
        or summary.get("source_roots") != 30
        or summary.get("requests") != 1785
        or summary.get("tool_events") != 1302
        or summary.get("failures") != 0
        or len(tasks) != 210
        or len(requests) != 1785
        or len(tools) != 1302
    ):
        raise ValueError(f"formal workload/result contract failed in {client}")
    if any(row.get("failure") is not None for row in tasks):
        raise ValueError(f"task failure found in {client}")
    flow_mean = statistics.fmean(float(row["flow_s"]) for row in tasks)
    if not math.isclose(flow_mean, float(summary["mean_task_flow_s"]), abs_tol=1e-9):
        raise ValueError(f"summary task flow mismatch in {client}")
    return dict(summary), {"tasks": tasks, "requests": requests, "tools": tools}


def _request_work(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], tuple[Any, ...]]:
    result: dict[tuple[str, int], tuple[Any, ...]] = {}
    for row in rows:
        key = (str(row["trace_id"]), int(row["request_index"]))
        usage = row.get("usage", {})
        value = (
            row.get("workload_request_sha256"),
            int(row["prompt_tokens"]),
            int(row["public_max_tokens"]),
            int(usage.get("prompt_tokens", -1)),
            int(usage.get("completion_tokens", -1)),
        )
        if key in result:
            raise ValueError(f"duplicate request event: {key}")
        result[key] = value
    return result


def _tool_work(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], tuple[Any, ...]]:
    result: dict[tuple[str, int], tuple[Any, ...]] = {}
    for row in rows:
        key = (str(row["trace_id"]), int(row["event_index"]))
        service_s = float(row.get("service_s", -1.0))
        if not math.isfinite(service_s) or service_s < 0.0:
            raise ValueError(f"invalid measured authority service: {key}")
        value = (
            row.get("tool_name"),
            row.get("authority_invocation_digest"),
            tuple(row.get("authority_candidate_invocation_digests", [])),
        )
        if key in result:
            raise ValueError(f"duplicate authority event: {key}")
        result[key] = value
    return result


def _task_work(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        trace_id = str(row["trace_id"])
        root = str(row["source_session_id_sha256"])
        if trace_id in result:
            raise ValueError(f"duplicate task result: {trace_id}")
        result[trace_id] = root
        counts[root] += 1
    if len(counts) != 30 or set(counts.values()) != {7}:
        raise ValueError("task rows are not 30 roots x7 replicas")
    return result


def _bootstrap_ratio(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired bootstrap vectors are invalid")
    rng = random.Random(seed)
    n = len(baseline)
    draws: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        base = statistics.fmean(baseline[index] for index in indices)
        treat = statistics.fmean(treatment[index] for index in indices)
        draws.append(_speedup(base, treat))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def analyze(run_root: Path, *, bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    discovered = sorted(run_root.glob("block-??/?/client/summary.json"))
    if not discovered:
        raise FileNotFoundError(f"no Pattern V2 cells in {run_root}")
    summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    events: dict[tuple[str, str], dict[str, Any]] = {}
    for path in discovered:
        block = path.parents[2].name
        cell = path.parents[1].name
        if cell not in CELL_SPECS or cell in summaries[block]:
            raise ValueError(f"invalid or duplicate cell path: {path}")
        summary, cell_events = _validate_result_directory(path.parent)
        if summary.get("cell") != cell:
            raise ValueError(f"cell identity mismatch in {path}")
        expected = CELL_SPECS[cell]
        if (
            summary.get("scheduler") != expected["scheduler"]
            or summary.get("speculation") is not expected["speculation"]
        ):
            raise ValueError(f"treatment mismatch in {path}")
        summaries[block][cell] = summary
        events[(block, cell)] = cell_events
    if any(set(rows) != set(CELL_SPECS) for rows in summaries.values()):
        raise ValueError("each block must contain exactly A/B/E/F")

    first = next(iter(next(iter(summaries.values())).values()))
    for block_rows in summaries.values():
        for summary in block_rows.values():
            for field in COMMON_SUMMARY_FIELDS:
                if summary.get(field) != first.get(field):
                    raise ValueError(f"cross-cell frozen field differs: {field}")
            configuration = summary.get("configuration", {})
            if (
                configuration.get("max_active_tasks") != 16
                or configuration.get("visit_capacity") != 64
                or configuration.get("speculative_cap") != 64
            ):
                raise ValueError("runtime capacity contract differs")

    reference_request = None
    reference_tool = None
    reference_task = None
    root_flows: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (block, cell), row in events.items():
        requests = _request_work(row["requests"])
        tools = _tool_work(row["tools"])
        tasks = _task_work(row["tasks"])
        if reference_request is None:
            reference_request, reference_tool, reference_task = requests, tools, tasks
        elif requests != reference_request or tools != reference_tool or tasks != reference_task:
            raise ValueError("A/B/E/F executed different LLM or authority work")
        grouped: dict[str, list[float]] = defaultdict(list)
        for task in row["tasks"]:
            grouped[str(task["source_session_id_sha256"])].append(float(task["flow_s"]))
        root_flows[block][cell] = {
            root: statistics.fmean(values) for root, values in grouped.items()
        }

    blocks = sorted(summaries)
    cells: dict[str, Any] = {}
    for cell in sorted(CELL_SPECS):
        rows = [summaries[block][cell] for block in blocks]
        cells[cell] = {
            "scheduler": rows[0]["scheduler"],
            "speculation": rows[0]["speculation"],
            "blocks": len(rows),
            "mean_makespan_s": statistics.fmean(float(row["experiment_wall_s"]) for row in rows),
            "mean_task_flow_s": statistics.fmean(float(row["mean_task_flow_s"]) for row in rows),
            "mean_p95_task_flow_s": statistics.fmean(float(row["p95_task_flow_s"]) for row in rows),
            "realized_visit_hit_rate": statistics.fmean(float(row["realized_visit_hit_rate"]) for row in rows),
            "visit_call_amplification": statistics.fmean(float(row["visit_call_amplification"]) for row in rows),
            "mean_tool_exposed_s_per_task": statistics.fmean(float(row["mean_tool_exposed_s_per_task"]) for row in rows),
            "mean_saved_tool_service_s_per_task": statistics.fmean(float(row["mean_saved_tool_service_s_per_task"]) for row in rows),
        }

    roots = sorted(next(iter(root_flows.values()))["A"])
    folded: dict[str, list[float]] = {}
    for cell in CELL_SPECS:
        folded[cell] = [
            statistics.fmean(root_flows[block][cell][root] for block in blocks)
            for root in roots
        ]
    contrasts: dict[str, Any] = {}
    for offset, (name, (baseline, treatment)) in enumerate(CONTRASTS.items()):
        base_wall = cells[baseline]["mean_makespan_s"]
        treat_wall = cells[treatment]["mean_makespan_s"]
        base_flow = statistics.fmean(folded[baseline])
        treat_flow = statistics.fmean(folded[treatment])
        low, high = _bootstrap_ratio(
            folded[baseline],
            folded[treatment],
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        contrasts[name] = {
            "baseline": baseline,
            "treatment": treatment,
            "makespan_speedup_fraction": _speedup(base_wall, treat_wall),
            "paired_root_mean_flow_speedup_fraction": _speedup(base_flow, treat_flow),
            "paired_root_bootstrap_95ci": [low, high],
            "independent_roots": len(roots),
            "load_replicas_per_root": 7,
        }

    return signed_payload(
        {
            "schema": SCHEMA,
            "run_root": str(run_root.resolve()),
            "evidence_scope": "retrospective_internal_holdout_live_pilot" if len(blocks) == 1 else "retrospective_internal_holdout_live_williams",
            "confirmatory_claim_allowed": False,
            "blocks": blocks,
            "cells": cells,
            "contrasts": contrasts,
            "bootstrap": {
                "unit": "paired_independent_source_root",
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
            },
            "work_equivalence": {
                "tasks": len(reference_task or {}),
                "requests": len(reference_request or {}),
                "authority_tools": len(reference_tool or {}),
                "identical_across_all_cells": True,
            },
        },
        "analysis_sha256",
    )


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Pattern V2 strict A/B/E/F live analysis",
        "",
        f"Scope: `{result['evidence_scope']}`; confirmatory claim allowed: **no**.",
        "",
        "| Cell | Makespan (s) | Mean flow (s) | P95 flow (s) | Visit hit | Call amp. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in ("A", "B", "E", "F"):
        row = result["cells"][cell]
        lines.append(
            f"| {cell} | {row['mean_makespan_s']:.3f} | {row['mean_task_flow_s']:.3f} "
            f"| {row['mean_p95_task_flow_s']:.3f} | {row['realized_visit_hit_rate']:.2%} "
            f"| {row['visit_call_amplification']:.3f}x |"
        )
    lines += ["", "| Contrast | Makespan speedup | Paired-root mean-flow speedup (95% CI) |", "|---|---:|---:|"]
    for name, row in result["contrasts"].items():
        low, high = row["paired_root_bootstrap_95ci"]
        lines.append(
            f"| {name} | {row['makespan_speedup_fraction']:.2%} | "
            f"{row['paired_root_mean_flow_speedup_fraction']:.2%} "
            f"([{low:.2%}, {high:.2%}]) |"
        )
    lines.append("")
    lines.append("All cells executed identical 210 tasks, 1,785 LLM requests, and 1,302 authoritative tool calls.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite output: {args.output_dir}")
    if args.bootstrap_samples < 1_000:
        parser.error("bootstrap samples must be at least 1000")
    result = analyze(
        args.run_root.resolve(),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(_report(result), encoding="utf-8")
    print(_report(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
