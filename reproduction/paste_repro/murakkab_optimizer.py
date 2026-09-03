"""Murakkab-inspired profile-guided optimization for PASTE workflows.

The Murakkab paper does not publish a runnable implementation.  This module
therefore reproduces the part of its design that can be evaluated fairly in
this repository: a typed declarative workflow, offline configuration profiles,
SLO filtering, and resource-minimizing configuration selection.  It does not
claim to reproduce Murakkab's Azure fleet manager, model/hardware placement,
or Gurobi formulation.

The configurable workflow knob is PASTE's speculative visit ``top_k``.  Every
configuration retains PASTE's exact authoritative commit rule, so changing the
knob can change exposed tool wait and admitted tool work but cannot change the
authoritative invocation stream in this replay.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import random
from typing import Any

from .mapper import URLRankMapper
from .scheduler import SpeculativeScheduler
from .traces import SearchVisitTransition


SCHEMA = "paste_repro.murakkab_inspired_optimizer"
VERSION = 1


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class WorkflowNode:
    """One typed logical executor in a declarative workflow DAG."""

    node_id: str
    executor: str
    depends_on: tuple[str, ...]
    input_types: dict[str, str]
    output_type: str


@dataclass(frozen=True)
class DeclarativeWorkflow:
    """A request-agnostic typed workflow, decoupled from execution knobs."""

    workflow_id: str
    description: str
    nodes: tuple[WorkflowNode, ...]
    topological_order: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeclarativeWorkflow":
        workflow_id = raw.get("id")
        description = raw.get("description", "")
        nodes_raw = raw.get("nodes")
        if not isinstance(workflow_id, str) or not workflow_id:
            raise ValueError("workflow.id must be a non-empty string")
        if not isinstance(description, str):
            raise ValueError("workflow.description must be a string")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise ValueError("workflow.nodes must be a non-empty list")

        nodes: list[WorkflowNode] = []
        for raw_node in nodes_raw:
            if not isinstance(raw_node, Mapping):
                raise ValueError("each workflow node must be an object")
            node_id = raw_node.get("id")
            executor = raw_node.get("executor")
            dependencies = raw_node.get("depends_on", [])
            input_types = raw_node.get("input_types", {})
            output_type = raw_node.get("output_type")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("workflow node id must be a non-empty string")
            if not isinstance(executor, str) or not executor:
                raise ValueError(f"node {node_id}: executor must be a non-empty string")
            if not isinstance(dependencies, list) or not all(
                isinstance(item, str) and item for item in dependencies
            ):
                raise ValueError(f"node {node_id}: depends_on must contain strings")
            if not isinstance(input_types, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in input_types.items()
            ):
                raise ValueError(f"node {node_id}: input_types must map strings to strings")
            if not isinstance(output_type, str) or not output_type:
                raise ValueError(f"node {node_id}: output_type must be a non-empty string")
            nodes.append(
                WorkflowNode(
                    node_id=node_id,
                    executor=executor,
                    depends_on=tuple(dependencies),
                    input_types=dict(input_types),
                    output_type=output_type,
                )
            )

        by_id = {node.node_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise ValueError("workflow node ids must be unique")
        for node in nodes:
            if set(node.input_types) != set(node.depends_on):
                raise ValueError(
                    f"node {node.node_id}: input_types keys must exactly match depends_on"
                )
            for dependency in node.depends_on:
                source = by_id.get(dependency)
                if source is None:
                    raise ValueError(
                        f"node {node.node_id}: unknown dependency {dependency!r}"
                    )
                expected = node.input_types[dependency]
                if expected != source.output_type:
                    raise ValueError(
                        f"node {node.node_id}: dependency {dependency!r} emits "
                        f"{source.output_type!r}, expected {expected!r}"
                    )

        remaining = {node.node_id: set(node.depends_on) for node in nodes}
        order: list[str] = []
        while remaining:
            ready = sorted(node_id for node_id, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError("workflow dependencies contain a cycle")
            for node_id in ready:
                order.append(node_id)
                remaining.pop(node_id)
            for dependencies in remaining.values():
                dependencies.difference_update(ready)

        return cls(
            workflow_id=workflow_id,
            description=description,
            nodes=tuple(nodes),
            topological_order=tuple(order),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.workflow_id,
            "description": self.description,
            "nodes": [asdict(node) for node in self.nodes],
            "topological_order": list(self.topological_order),
            "type_checked": True,
        }


@dataclass(frozen=True)
class CandidateConfiguration:
    config_id: str
    top_k: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CandidateConfiguration":
        config_id = raw.get("id")
        top_k = raw.get("top_k")
        if not isinstance(config_id, str) or not config_id:
            raise ValueError("candidate configuration id must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError(f"candidate {config_id}: top_k must be a non-negative integer")
        return cls(config_id=config_id, top_k=top_k)


@dataclass(frozen=True)
class LatencySLO:
    tier: str
    minimum_stall_reduction: float
    planning_margin: float
    demand_weight: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LatencySLO":
        tier = raw.get("tier")
        if not isinstance(tier, str) or not tier:
            raise ValueError("SLO tier must be a non-empty string")
        try:
            target = float(raw.get("minimum_stall_reduction"))
            margin = float(raw.get("planning_margin", 0.0))
            weight = float(raw.get("demand_weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SLO tier {tier}: numeric fields are invalid") from exc
        if not 0.0 <= target < 1.0:
            raise ValueError(f"SLO tier {tier}: minimum_stall_reduction must be in [0, 1)")
        if not 0.0 <= margin < 1.0 or target + margin >= 1.0:
            raise ValueError(f"SLO tier {tier}: target plus margin must be in [0, 1)")
        if weight <= 0.0:
            raise ValueError(f"SLO tier {tier}: demand_weight must be positive")
        return cls(tier, target, margin, weight)


@dataclass(frozen=True)
class ConfigurationProfile:
    """Observed latency/resource profile for one PASTE configuration."""

    role: str
    config_id: str
    top_k: int
    sessions_with_transitions: int
    search_visit_transitions: int
    authoritative_invocations: int
    concrete_predictions: int
    exact_hits: int
    authoritative_misses: int
    admitted_tool_request_units: int
    speculative_waste_units: int
    baseline_exposed_tool_stall_s: float
    optimized_exposed_tool_stall_s: float
    saved_tool_stall_s: float
    stall_reduction: float
    exact_invocation_hit_rate: float
    prediction_precision: float
    admitted_tool_request_units_per_authoritative_invocation: float
    bootstrap_stall_reduction_95pct_ci: tuple[float, float] | None
    bootstrap_tool_request_units_per_invocation_95pct_ci: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_configuration(
    mapper: URLRankMapper,
    transitions: Sequence[SearchVisitTransition],
    configuration: CandidateConfiguration,
    *,
    role: str,
    bootstrap_samples: int = 0,
    bootstrap_seed: str = "murakkab-paste-v1",
) -> ConfigurationProfile:
    """Build a workflow profile using exact URL matches and trace timestamps.

    ``admitted_tool_request_units`` is a conservative resource proxy: every
    concrete speculative admission plus every authoritative miss counts as one
    unit.  A live broker may cancel a queued unused prediction before it starts,
    so this is not labeled as measured energy or dollar cost.
    """

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    transition_list = tuple(transitions)
    session_rows: dict[str, list[float]] = {}
    baseline_stall_s = 0.0
    saved_stall_s = 0.0
    authoritative = 0
    predictions = 0
    hits = 0

    for transition in transition_list:
        predicted = mapper.predict(transition.search_results, configuration.top_k)
        predicted_keys = {item.invocation.key for item in predicted}
        transition_authoritative = len(transition.authoritative_invocations)
        transition_hits = sum(
            invocation.key in predicted_keys
            for invocation in transition.authoritative_invocations
        )
        transition_saved = min(
            transition.baseline_stall_s,
            transition.overlap_window_s,
        ) * _ratio(transition_hits, transition_authoritative)
        authoritative += transition_authoritative
        predictions += len(predicted)
        hits += transition_hits
        baseline_stall_s += transition.baseline_stall_s
        saved_stall_s += transition_saved
        row = session_rows.setdefault(transition.session_id, [0.0] * 5)
        row[0] += transition.baseline_stall_s
        row[1] += transition_saved
        row[2] += transition_authoritative
        row[3] += len(predicted)
        row[4] += transition_hits

    misses = authoritative - hits
    tool_units = predictions + misses
    waste = predictions - hits
    stall_samples: list[float] = []
    unit_samples: list[float] = []
    rows = tuple(session_rows.values())
    if bootstrap_samples and rows:
        rng = random.Random(
            f"{bootstrap_seed}\0{role}\0{configuration.config_id}\0{configuration.top_k}"
        )
        for _ in range(bootstrap_samples):
            sample = tuple(rows[rng.randrange(len(rows))] for _ in rows)
            sample_baseline = sum(row[0] for row in sample)
            sample_saved = sum(row[1] for row in sample)
            sample_authoritative = sum(row[2] for row in sample)
            sample_predictions = sum(row[3] for row in sample)
            sample_hits = sum(row[4] for row in sample)
            stall_samples.append(_ratio(sample_saved, sample_baseline))
            unit_samples.append(
                _ratio(
                    sample_predictions + sample_authoritative - sample_hits,
                    sample_authoritative,
                )
            )

    optimized = max(0.0, baseline_stall_s - saved_stall_s)
    return ConfigurationProfile(
        role=role,
        config_id=configuration.config_id,
        top_k=configuration.top_k,
        sessions_with_transitions=len(session_rows),
        search_visit_transitions=len(transition_list),
        authoritative_invocations=authoritative,
        concrete_predictions=predictions,
        exact_hits=hits,
        authoritative_misses=misses,
        admitted_tool_request_units=tool_units,
        speculative_waste_units=waste,
        baseline_exposed_tool_stall_s=baseline_stall_s,
        optimized_exposed_tool_stall_s=optimized,
        saved_tool_stall_s=saved_stall_s,
        stall_reduction=_ratio(saved_stall_s, baseline_stall_s),
        exact_invocation_hit_rate=_ratio(hits, authoritative),
        prediction_precision=_ratio(hits, predictions),
        admitted_tool_request_units_per_authoritative_invocation=_ratio(
            tool_units, authoritative
        ),
        bootstrap_stall_reduction_95pct_ci=(
            (_quantile(stall_samples, 0.025), _quantile(stall_samples, 0.975))
            if stall_samples
            else None
        ),
        bootstrap_tool_request_units_per_invocation_95pct_ci=(
            (_quantile(unit_samples, 0.025), _quantile(unit_samples, 0.975))
            if unit_samples
            else None
        ),
    )


def optimize_configurations(
    candidates: Sequence[CandidateConfiguration],
    profiles_by_role: Mapping[str, Mapping[str, ConfigurationProfile]],
    slos: Sequence[LatencySLO],
) -> list[dict[str, Any]]:
    """Select minimum conservative tool work subject to each latency SLO.

    The planner deliberately uses the worst observed reduction and worst
    admitted-work ratio across all supplied profiling roles.  This corresponds
    to Murakkab's profile-guided filtering, with an explicit user-visible
    safety margin rather than an undocumented tuning heuristic.
    """

    if not candidates:
        raise ValueError("at least one candidate configuration is required")
    if not profiles_by_role:
        raise ValueError("at least one profiling role is required")
    candidate_ids = {candidate.config_id for candidate in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("candidate configuration ids must be unique")
    if len({candidate.top_k for candidate in candidates}) != len(candidates):
        raise ValueError("candidate top_k values must be unique")
    for role, profiles in profiles_by_role.items():
        missing = sorted(candidate_ids - set(profiles))
        if missing:
            raise ValueError(f"profiling role {role!r} is missing candidates: {missing}")

    planning_rows: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        role_profiles = [
            profiles[candidate.config_id] for profiles in profiles_by_role.values()
        ]
        planning_rows[candidate.config_id] = {
            "config_id": candidate.config_id,
            "top_k": candidate.top_k,
            "conservative_stall_reduction": min(
                profile.stall_reduction for profile in role_profiles
            ),
            "conservative_tool_request_units_per_invocation": max(
                profile.admitted_tool_request_units_per_authoritative_invocation
                for profile in role_profiles
            ),
            "profile_roles": {
                profile.role: {
                    "stall_reduction": profile.stall_reduction,
                    "tool_request_units_per_invocation": (
                        profile.admitted_tool_request_units_per_authoritative_invocation
                    ),
                }
                for profile in role_profiles
            },
        }

    plans: list[dict[str, Any]] = []
    for slo in slos:
        threshold = slo.minimum_stall_reduction + slo.planning_margin
        feasible = [
            row
            for row in planning_rows.values()
            if row["conservative_stall_reduction"] + 1e-12 >= threshold
        ]
        if not feasible:
            best_effort = max(
                planning_rows.values(),
                key=lambda row: (row["conservative_stall_reduction"], -row["top_k"]),
            )
            plans.append(
                {
                    "tier": slo.tier,
                    "status": "infeasible",
                    "minimum_stall_reduction": slo.minimum_stall_reduction,
                    "planning_margin": slo.planning_margin,
                    "planning_threshold": threshold,
                    "demand_weight": slo.demand_weight,
                    "best_effort": best_effort,
                }
            )
            continue
        selected = min(
            feasible,
            key=lambda row: (
                row["conservative_tool_request_units_per_invocation"],
                row["top_k"],
                row["config_id"],
            ),
        )
        plans.append(
            {
                "tier": slo.tier,
                "status": "selected",
                "minimum_stall_reduction": slo.minimum_stall_reduction,
                "planning_margin": slo.planning_margin,
                "planning_threshold": threshold,
                "demand_weight": slo.demand_weight,
                "selected": selected,
            }
        )
    return plans


async def execute_isolated_replay(
    mapper: URLRankMapper,
    transitions: Sequence[SearchVisitTransition],
    configuration: CandidateConfiguration,
    *,
    max_concurrency: int = 8,
) -> dict[str, Any]:
    """Execute the scheduler contract and reconcile every replayed operation."""

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    transition_list = tuple(transitions)
    executor_calls = 0

    async def executor(invocation: Any) -> dict[str, Any]:
        nonlocal executor_calls
        executor_calls += 1
        return {"replayed": invocation.to_dict()}

    scheduler = SpeculativeScheduler(
        executor,
        max_concurrency=max(max_concurrency, configuration.top_k, 1),
        max_pending=max(max_concurrency, configuration.top_k, 1),
        ttl_s=30.0,
    )
    isolation_violations = 0
    predictions = 0
    exact_hits = 0
    authoritative = 0
    try:
        for index, transition in enumerate(transition_list):
            session_id = f"{transition.session_id}:{index}"
            predicted = mapper.predict(transition.search_results, configuration.top_k)
            predicted_keys = {item.invocation.key for item in predicted}
            predictions += len(predicted)
            authoritative += len(transition.authoritative_invocations)
            exact_hits += sum(
                invocation.key in predicted_keys
                for invocation in transition.authoritative_invocations
            )
            commits_before = len(scheduler.authoritative_state)
            for prediction in predicted:
                admitted = await scheduler.speculate(
                    prediction.invocation,
                    session_id=session_id,
                )
                if not admitted:
                    raise AssertionError("unbounded test replay unexpectedly rejected work")
            # All test executor calls are zero-delay and concurrency covers top_k.
            await asyncio.sleep(0)
            if len(scheduler.authoritative_state) != commits_before:
                isolation_violations += 1
            for invocation in transition.authoritative_invocations:
                await scheduler.authoritative(invocation, session_id=session_id)
            await scheduler.sweep(now=float("inf"), session_id=session_id)
    finally:
        await scheduler.close()

    stats = scheduler.stats.to_dict()
    stats.pop("saved_time_s", None)
    promoted_hits = stats["completed_reuse"] + stats["inflight_promotions"]
    expected_calls = predictions + authoritative - exact_hits
    reconciliation = {
        "predictions_equal_admissions": predictions == stats["admitted"],
        "exact_hits_equal_reuse_plus_promotions": exact_hits == promoted_hits,
        "commits_equal_authoritative_invocations": stats["commits"] == authoritative,
        "executor_calls_equal_predictions_plus_misses": executor_calls == expected_calls,
        "zero_state_isolation_violations": isolation_violations == 0,
    }
    if not all(reconciliation.values()):
        raise AssertionError(f"replay reconciliation failed: {reconciliation}")
    return {
        "config_id": configuration.config_id,
        "top_k": configuration.top_k,
        "transitions": len(transition_list),
        "authoritative_invocations": authoritative,
        "concrete_predictions": predictions,
        "exact_hits": exact_hits,
        "executor_calls": executor_calls,
        "state_isolation_violations": isolation_violations,
        "scheduler": stats,
        "reconciliation": reconciliation,
    }


def compare_plans(
    plans: Sequence[Mapping[str, Any]],
    final_profiles: Mapping[str, ConfigurationProfile],
    *,
    static_demand_config_id: str,
    static_paste_config_id: str,
) -> dict[str, Any]:
    """Compare an SLO-aware plan with static demand-only and static PASTE."""

    selected_plans = [plan for plan in plans if plan.get("status") == "selected"]
    if len(selected_plans) != len(plans):
        return {
            "status": "infeasible",
            "reason": "one or more SLO tiers have no feasible profiled configuration",
        }
    if static_demand_config_id not in final_profiles:
        raise ValueError("static demand-only profile is missing")
    if static_paste_config_id not in final_profiles:
        raise ValueError("static PASTE profile is missing")

    total_weight = sum(float(plan["demand_weight"]) for plan in selected_plans)

    def weighted_metric(config_for_plan: Any, attribute: str) -> float:
        return sum(
            float(plan["demand_weight"])
            * float(getattr(final_profiles[config_for_plan(plan)], attribute))
            for plan in selected_plans
        ) / total_weight

    adaptive_units = weighted_metric(
        lambda plan: plan["selected"]["config_id"],
        "admitted_tool_request_units_per_authoritative_invocation",
    )
    adaptive_reduction = weighted_metric(
        lambda plan: plan["selected"]["config_id"], "stall_reduction"
    )
    demand = final_profiles[static_demand_config_id]
    paste = final_profiles[static_paste_config_id]

    tier_results: list[dict[str, Any]] = []
    demand_passes = 0
    paste_passes = 0
    adaptive_passes = 0
    for plan in selected_plans:
        target = float(plan["minimum_stall_reduction"])
        selected_id = plan["selected"]["config_id"]
        adaptive_profile = final_profiles[selected_id]
        demand_pass = demand.stall_reduction + 1e-12 >= target
        paste_pass = paste.stall_reduction + 1e-12 >= target
        adaptive_pass = adaptive_profile.stall_reduction + 1e-12 >= target
        demand_passes += demand_pass
        paste_passes += paste_pass
        adaptive_passes += adaptive_pass
        tier_results.append(
            {
                "tier": plan["tier"],
                "demand_weight": plan["demand_weight"],
                "minimum_stall_reduction": target,
                "selected_config_id": selected_id,
                "selected_top_k": adaptive_profile.top_k,
                "final_stall_reduction": adaptive_profile.stall_reduction,
                "final_tool_request_units_per_invocation": (
                    adaptive_profile.admitted_tool_request_units_per_authoritative_invocation
                ),
                "aggregate_slo_met": adaptive_pass,
                "static_demand_only_slo_met": demand_pass,
                "static_paste_slo_met": paste_pass,
            }
        )

    return {
        "status": "ok",
        "demand_mix": "weights normalized across declared SLO tiers",
        "quality_contract": (
            "all policies commit the same authoritative invocations; speculative "
            "results remain isolated until an exact match"
        ),
        "policies": {
            "static_demand_only": {
                "config_id": static_demand_config_id,
                "top_k": demand.top_k,
                "weighted_stall_reduction": demand.stall_reduction,
                "weighted_tool_request_units_per_invocation": (
                    demand.admitted_tool_request_units_per_authoritative_invocation
                ),
                "aggregate_slo_tiers_met": demand_passes,
                "aggregate_slo_tiers_total": len(plans),
            },
            "static_paste": {
                "config_id": static_paste_config_id,
                "top_k": paste.top_k,
                "weighted_stall_reduction": paste.stall_reduction,
                "weighted_tool_request_units_per_invocation": (
                    paste.admitted_tool_request_units_per_authoritative_invocation
                ),
                "aggregate_slo_tiers_met": paste_passes,
                "aggregate_slo_tiers_total": len(plans),
            },
            "murakkab_inspired_paste": {
                "weighted_stall_reduction": adaptive_reduction,
                "weighted_tool_request_units_per_invocation": adaptive_units,
                "aggregate_slo_tiers_met": adaptive_passes,
                "aggregate_slo_tiers_total": len(plans),
            },
        },
        "murakkab_vs_static_paste": {
            "admitted_tool_request_unit_reduction": _ratio(
                paste.admitted_tool_request_units_per_authoritative_invocation
                - adaptive_units,
                paste.admitted_tool_request_units_per_authoritative_invocation,
            ),
            "stall_reduction_tradeoff_pp": 100.0
            * (adaptive_reduction - paste.stall_reduction),
        },
        "murakkab_vs_demand_only": {
            "admitted_tool_request_unit_increase": _ratio(
                adaptive_units
                - demand.admitted_tool_request_units_per_authoritative_invocation,
                demand.admitted_tool_request_units_per_authoritative_invocation,
            ),
            "stall_reduction_gain_pp": 100.0
            * (adaptive_reduction - demand.stall_reduction),
        },
        "tier_results": tier_results,
    }

