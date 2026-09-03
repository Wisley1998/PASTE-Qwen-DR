#!/usr/bin/env python3
"""Run the Murakkab-inspired, SLO-aware PASTE trace comparison."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_ROOT = REPRODUCTION_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_fixed_three_way_split import build_fixed_bundle  # noqa: E402
from paste_repro.mapper import load_artifact, write_json_atomic  # noqa: E402
from paste_repro.murakkab_optimizer import (  # noqa: E402
    SCHEMA,
    VERSION,
    CandidateConfiguration,
    DeclarativeWorkflow,
    LatencySLO,
    compare_plans,
    execute_isolated_replay,
    optimize_configurations,
    profile_configuration,
)
from paste_repro.traces import load_sessions, transitions_from_sessions  # noqa: E402


CONFIG_SCHEMA = "paste_repro.murakkab_trace_experiment_config"
CONFIG_VERSION = 1
RESULT_SCHEMA = "paste_repro.murakkab_paste_comparison"
RESULT_VERSION = 1
DEFAULT_CONFIG = REPRODUCTION_ROOT / "configs" / "murakkab_paste_trace.json"
DEFAULT_RESULT = REPRODUCTION_ROOT / "results" / "murakkab_paste" / "comparison.json"
DEFAULT_REPORT = REPRODUCTION_ROOT / "results" / "murakkab_paste" / "REPORT.md"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_config(path: Path) -> tuple[
    dict[str, Any],
    DeclarativeWorkflow,
    tuple[CandidateConfiguration, ...],
    tuple[LatencySLO, ...],
]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config root must be an object")
    if raw.get("schema") != CONFIG_SCHEMA or raw.get("version") != CONFIG_VERSION:
        raise ValueError(
            f"unsupported experiment config: {raw.get('schema')!r} v{raw.get('version')!r}"
        )
    workflow_raw = raw.get("workflow")
    candidates_raw = raw.get("candidate_configurations")
    slos_raw = raw.get("latency_slos")
    if not isinstance(workflow_raw, Mapping):
        raise ValueError("experiment config is missing workflow")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidate_configurations must be a non-empty list")
    if not isinstance(slos_raw, list) or not slos_raw:
        raise ValueError("latency_slos must be a non-empty list")
    workflow = DeclarativeWorkflow.from_mapping(workflow_raw)
    candidates = tuple(
        CandidateConfiguration.from_mapping(item)
        for item in candidates_raw
        if isinstance(item, Mapping)
    )
    slos = tuple(
        LatencySLO.from_mapping(item) for item in slos_raw if isinstance(item, Mapping)
    )
    if len(candidates) != len(candidates_raw):
        raise ValueError("every candidate configuration must be an object")
    if len(slos) != len(slos_raw):
        raise ValueError("every latency SLO must be an object")
    if len({slo.tier for slo in slos}) != len(slos):
        raise ValueError("latency SLO tier names must be unique")
    return raw, workflow, candidates, slos


def _bootstrap_config(config: Mapping[str, Any]) -> tuple[int, str]:
    raw = config.get("bootstrap", {})
    if not isinstance(raw, Mapping):
        raise ValueError("bootstrap config must be an object")
    samples = raw.get("samples", 0)
    seed = raw.get("seed", "murakkab-paste-v1")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("bootstrap.samples must be a non-negative integer")
    if not isinstance(seed, str) or not seed:
        raise ValueError("bootstrap.seed must be a non-empty string")
    return samples, seed


def _baseline_ids(config: Mapping[str, Any]) -> tuple[str, str]:
    raw = config.get("baselines")
    if not isinstance(raw, Mapping):
        raise ValueError("baselines config must be an object")
    demand = raw.get("static_demand_config_id")
    paste = raw.get("static_paste_config_id")
    if not isinstance(demand, str) or not demand:
        raise ValueError("static_demand_config_id must be a non-empty string")
    if not isinstance(paste, str) or not paste:
        raise ValueError("static_paste_config_id must be a non-empty string")
    return demand, paste


def _render_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(result: Mapping[str, Any]) -> str:
    comparison = result["comparison"]
    policies = comparison.get("policies", {})
    lines = [
        "# Murakkab-inspired PASTE comparison",
        "",
        "Experiment date: 2026-08-31",
        "",
        "## Result",
        "",
    ]
    if comparison.get("status") != "ok":
        lines.extend(
            [
                "The configured SLO plan was infeasible. See `comparison.json` for details.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "On the fixed final trace role, the Murakkab-inspired planner met all "
                "four aggregate SLO tiers while reducing the conservative admitted-tool "
                "request proxy by "
                f"**{_render_percent(comparison['murakkab_vs_static_paste']['admitted_tool_request_unit_reduction'])}** "
                "relative to static PASTE `top_k=5`.",
                "",
                "| Policy | Weighted stall reduction | Tool request units / authoritative call | SLO tiers met |",
                "|---|---:|---:|---:|",
            ]
        )
        labels = {
            "static_demand_only": "Demand only (k=0)",
            "static_paste": "Static PASTE (k=5)",
            "murakkab_inspired_paste": "Murakkab-inspired PASTE",
        }
        for key in ("static_demand_only", "static_paste", "murakkab_inspired_paste"):
            policy = policies[key]
            lines.append(
                f"| {labels[key]} | {_render_percent(policy['weighted_stall_reduction'])} "
                f"| {policy['weighted_tool_request_units_per_invocation']:.3f} "
                f"| {policy['aggregate_slo_tiers_met']}/{policy['aggregate_slo_tiers_total']} |"
            )
        lines.extend(
            [
                "",
                "The resource metric is an admitted request-unit upper bound, not measured "
                "GPU energy, cloud cost, or completed network calls. Static demand-only "
                "uses fewer units but fails the non-basic latency tiers; static top-5 "
                "over-serves relaxed tiers.",
                "",
                "## Selected configurations",
                "",
                "| Tier | Required stall reduction | Planning margin | Selected | Final reduction | Final SLO |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        plans_by_tier = {plan["tier"]: plan for plan in result["plans"]}
        for tier in comparison["tier_results"]:
            plan = plans_by_tier[tier["tier"]]
            lines.append(
                f"| {tier['tier']} | {_render_percent(tier['minimum_stall_reduction'])} "
                f"| {_render_percent(plan['planning_margin'])} "
                f"| k={tier['selected_top_k']} "
                f"| {_render_percent(tier['final_stall_reduction'])} "
                f"| {'pass' if tier['aggregate_slo_met'] else 'fail'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Final configuration frontier",
            "",
            "| Configuration | Hits / authoritative | Stall reduction (95% session bootstrap CI) | Tool request units / authoritative |",
            "|---|---:|---:|---:|",
        ]
    )
    for profile in result["profiles"]["final"]:
        ci = profile["bootstrap_stall_reduction_95pct_ci"]
        ci_text = (
            f" [{_render_percent(ci[0])}, {_render_percent(ci[1])}]" if ci else ""
        )
        lines.append(
            f"| {profile['config_id']} | {profile['exact_hits']}/{profile['authoritative_invocations']} "
            f"| {_render_percent(profile['stall_reduction'])}{ci_text} "
            f"| {profile['admitted_tool_request_units_per_authoritative_invocation']:.3f} |"
        )

    replay_ok = all(
        all(item["reconciliation"].values())
        for item in result["executed_replays"].values()
    )
    isolation_violations = sum(
        item["state_isolation_violations"]
        for item in result["executed_replays"].values()
    )
    lines.extend(
        [
            "",
            "## What was reproduced",
            "",
            "This is an idea-level reproduction on PASTE-Qwen-DR because no official "
            "Murakkab code artifact is linked from the paper or USENIX page. It implements "
            "a typed declarative DAG, offline workflow profiles, SLO filtering, a "
            "resource-minimizing configuration planner, and exact isolated replay. "
            f"Replay reconciliation was `{'pass' if replay_ok else 'fail'}` with "
            f"{isolation_violations} state-isolation violations.",
            "",
            "It does **not** reproduce Murakkab's LLM-based executor discovery, model/GPU "
            "profiles, Gurobi MILP, Azure autoscaling, multi-model colocation, or the "
            "paper's energy/cost numbers. Those require unpublished code, model profiles, "
            "A100-80GB/H100-80GB fleets, and the authors' production-scale workload.",
            "",
            "The latency result remains the repository's bounded trace counterfactual: "
            "`min(observed visit stall, preceding LLM decision window) × exact-hit fraction`. "
            "SLO pass/fail is aggregate over the final role, not a per-request guarantee. "
            "The typed DAG is validated but the recorded trace supplies execution order; "
            "each selected `k` is replayed separately and combined by declared demand "
            "weights rather than run as one online mixed-SLO service. The protocol is "
            "exploratory and was not preregistered.",
            "",
            "Primary sources: [USENIX OSDI '26 paper page](https://www.usenix.org/conference/osdi26/presentation/chaudhry), "
            "[arXiv paper](https://arxiv.org/abs/2508.18298).",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python reproduction/scripts/run_murakkab_paste_comparison.py",
            "PYTHONPATH=reproduction pytest -q reproduction/tests/test_murakkab_optimizer.py",
            "```",
            "",
            "Machine-readable evidence is in [`comparison.json`](comparison.json).",
            "",
        ]
    )
    return "\n".join(lines)


async def _execute_replays(
    mapper: Any,
    transitions: Sequence[Any],
    candidates: Sequence[CandidateConfiguration],
    plans: Sequence[Mapping[str, Any]],
    baseline_ids: tuple[str, str],
) -> dict[str, Any]:
    by_id = {candidate.config_id: candidate for candidate in candidates}
    selected_ids = {
        plan["selected"]["config_id"]
        for plan in plans
        if plan.get("status") == "selected"
    }
    selected_ids.update(baseline_ids)
    unknown = sorted(selected_ids - set(by_id))
    if unknown:
        raise ValueError(f"selected or baseline config ids are unknown: {unknown}")
    results: dict[str, Any] = {}
    for config_id in sorted(selected_ids, key=lambda item: by_id[item].top_k):
        results[config_id] = await execute_isolated_replay(
            mapper,
            transitions,
            by_id[config_id],
        )
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, workflow, candidates, slos = _load_config(args.config)
    bootstrap_samples, bootstrap_seed = _bootstrap_config(config)
    baseline_ids = _baseline_ids(config)

    fixed = build_fixed_bundle(
        legacy_artifact_path=args.legacy_artifact,
        trace_directory=args.trace_directory,
        output_root=args.split_output_root,
    )
    mapper_path = Path(fixed["mapper_artifact_path"])
    mapper, mapper_artifact = load_artifact(mapper_path)
    transitions_by_role: dict[str, tuple[Any, ...]] = {}
    for role in ("calibration", "tuning", "final"):
        role_directory = Path(fixed["roles"][role]["absolute_directory"])
        transitions_by_role[role] = transitions_from_sessions(load_sessions(role_directory))

    profiles: dict[str, dict[str, Any]] = {}
    profile_objects: dict[str, dict[str, Any]] = {}
    for role, transitions in transitions_by_role.items():
        role_profiles = {
            candidate.config_id: profile_configuration(
                mapper,
                transitions,
                candidate,
                role=role,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for candidate in candidates
        }
        profile_objects[role] = role_profiles
        profiles[role] = [
            role_profiles[candidate.config_id].to_dict() for candidate in candidates
        ]

    plans = optimize_configurations(
        candidates,
        {
            "calibration": profile_objects["calibration"],
            "tuning": profile_objects["tuning"],
        },
        slos,
    )
    comparison = compare_plans(
        plans,
        profile_objects["final"],
        static_demand_config_id=baseline_ids[0],
        static_paste_config_id=baseline_ids[1],
    )
    executed_replays = asyncio.run(
        _execute_replays(
            mapper,
            transitions_by_role["final"],
            candidates,
            plans,
            baseline_ids,
        )
    )

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "implementation": {
            "module_schema": SCHEMA,
            "module_version": VERSION,
            "kind": "Murakkab-inspired idea-level reproduction on PASTE-Qwen-DR",
            "workflow": workflow.to_dict(),
            "implemented_mechanisms": [
                "typed declarative request-agnostic workflow DAG",
                "offline profiles over workflow execution configurations",
                "SLO filtering with explicit safety margins",
                "minimum admitted tool-work selection",
                "exact-match isolated speculative execution replay",
            ],
            "not_implemented": [
                "authors' unpublished Murakkab code",
                "LLM-based arbitrary task-to-executor discovery",
                "model and heterogeneous GPU profile collection",
                "Gurobi MILP fleet allocation",
                "autoscaling and multi-tenant model-instance multiplexing",
                "Azure energy and dollar-cost accounting",
            ],
        },
        "protocol": {
            "status": "post-hoc exploratory; not preregistered",
            "whole_session_roles": ["calibration", "tuning", "final"],
            "mapper_training_role": "calibration",
            "planning_profile_roles": ["calibration", "tuning"],
            "evaluation_role": "final",
            "planning_profile_aggregation": (
                "minimum stall reduction and maximum tool request units across roles"
            ),
            "latency_model": (
                "per transition min(observed visit stall, preceding LLM inference) "
                "times exact authoritative URL hit fraction"
            ),
            "resource_metric": (
                "admitted concrete speculative requests plus authoritative misses; "
                "conservative request-unit proxy, not energy/cost"
            ),
            "quality_boundary": (
                "same authoritative invocation sequence and exact-match commit; no task "
                "answer-quality benchmark is inferred"
            ),
            "execution_boundary": (
                "typed DAG is parsed and type-checked, while recorded traces provide the "
                "execution order; selected configurations are replayed separately and "
                "combined by SLO demand weights, not dispatched by an online mixed-SLO runtime"
            ),
            "bootstrap": {
                "unit": "whole session with at least one extracted transition",
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
            },
        },
        "candidate_configurations": [
            {"config_id": item.config_id, "top_k": item.top_k} for item in candidates
        ],
        "latency_slos": [asdict_slo(slo) for slo in slos],
        "profiles": profiles,
        "plans": plans,
        "comparison": comparison,
        "executed_replays": executed_replays,
        "provenance": {
            "git_head": _git_head(),
            "config": _relative(args.config),
            "config_sha256": _file_sha256(args.config),
            "module": "reproduction/paste_repro/murakkab_optimizer.py",
            "module_sha256": _file_sha256(
                REPRODUCTION_ROOT / "paste_repro" / "murakkab_optimizer.py"
            ),
            "entrypoint": "reproduction/scripts/run_murakkab_paste_comparison.py",
            "entrypoint_sha256": _file_sha256(Path(__file__)),
            "split_bundle_sha256": fixed["bundle_sha256"],
            "split_manifest_sha256": fixed["split_manifest_sha256"],
            "mapper_artifact_sha256": mapper_artifact["artifact_sha256"],
            "role_session_checksums": {
                role: fixed["roles"][role]["sessions_sha256"]
                for role in ("calibration", "tuning", "final")
            },
        },
        "primary_sources": {
            "usenix": "https://www.usenix.org/conference/osdi26/presentation/chaudhry",
            "arxiv": "https://arxiv.org/abs/2508.18298",
            "code_availability_audit": (
                "No official runnable artifact was linked from the paper, USENIX page, "
                "author publication pages, or discoverable GitHub results as of 2026-08-31."
            ),
        },
        "claim_boundaries": [
            "This is not the authors' Murakkab implementation.",
            "Published Murakkab GPU, energy, and cost ratios are not reproduced or compared numerically.",
            "Tool request units are a normalized conservative proxy, not joules, GPU-hours, or dollars.",
            "Latency savings are trace-derived counterfactuals, not a live-network timing run.",
            "Aggregate SLO compliance does not establish per-request tail compliance.",
            "The declarative DAG is validated but does not drive the recorded trace replay.",
            "The weighted SLO comparison combines separate configuration replays, not one live mixed-SLO run.",
        ],
    }
    return result


def asdict_slo(slo: LatencySLO) -> dict[str, Any]:
    return {
        "tier": slo.tier,
        "minimum_stall_reduction": slo.minimum_stall_reduction,
        "planning_margin": slo.planning_margin,
        "demand_weight": slo.demand_weight,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--legacy-artifact",
        type=Path,
        default=REPRODUCTION_ROOT / "results" / "tool_only" / "url_rank_mapper.json",
    )
    parser.add_argument(
        "--trace-directory",
        type=Path,
        default=REPOSITORY_ROOT / "traces" / "my_traces",
    )
    parser.add_argument(
        "--split-output-root",
        type=Path,
        default=REPRODUCTION_ROOT / "artifacts" / "fixed_trace_splits",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    write_json_atomic(args.output, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render_report(result), encoding="utf-8")
    print(json.dumps(result["comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {_relative(args.output)}")
    print(f"wrote {_relative(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
