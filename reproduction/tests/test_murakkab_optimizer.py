from __future__ import annotations

from dataclasses import replace
import unittest

from paste_repro.mapper import URLRankMapper
from paste_repro.murakkab_optimizer import (
    CandidateConfiguration,
    DeclarativeWorkflow,
    LatencySLO,
    compare_plans,
    execute_isolated_replay,
    optimize_configurations,
    profile_configuration,
)
from paste_repro.traces import (
    LLMCall,
    SearchResult,
    SearchVisitTransition,
    ToolCall,
)


def _transition(
    session_id: str,
    *,
    chosen_rank: int = 1,
    baseline_stall_s: float = 4.0,
    overlap_window_s: float = 2.0,
) -> SearchVisitTransition:
    results = tuple(
        SearchResult(
            url=f"https://example.test/{session_id}/{rank}",
            result_rank=rank,
            ordinal=rank - 1,
            query_index=0,
        )
        for rank in range(1, 4)
    )
    chosen_url = results[chosen_rank - 1].url
    search = ToolCall(0, 1.0, "search", {"query": session_id}, 1)
    decision = LLMCall(1, 2.0, overlap_window_s, overlap_window_s, (), "visit", 2)
    visit = ToolCall(1, 2.1, "visit", {"url": chosen_url}, 3)
    return SearchVisitTransition(
        session_id=session_id,
        search=search,
        decision_llm=decision,
        visit=visit,
        completion_llm=None,
        search_results=results,
        authoritative_urls=(chosen_url,),
        baseline_stall_s=baseline_stall_s,
        overlap_window_s=overlap_window_s,
    )


def _workflow() -> dict:
    return {
        "id": "test",
        "description": "typed test DAG",
        "nodes": [
            {
                "id": "search",
                "executor": "search",
                "depends_on": [],
                "input_types": {},
                "output_type": "results",
            },
            {
                "id": "visit",
                "executor": "visit",
                "depends_on": ["search"],
                "input_types": {"search": "results"},
                "output_type": "documents",
            },
        ],
    }


class DeclarativeWorkflowTests(unittest.TestCase):
    def test_type_checked_topological_order(self) -> None:
        workflow = DeclarativeWorkflow.from_mapping(_workflow())
        self.assertEqual(workflow.topological_order, ("search", "visit"))
        self.assertTrue(workflow.to_dict()["type_checked"])

    def test_type_mismatch_and_cycle_fail_closed(self) -> None:
        mismatch = _workflow()
        mismatch["nodes"][1]["input_types"]["search"] = "wrong"
        with self.assertRaisesRegex(ValueError, "emits 'results', expected 'wrong'"):
            DeclarativeWorkflow.from_mapping(mismatch)

        cycle = _workflow()
        cycle["nodes"][0]["depends_on"] = ["visit"]
        cycle["nodes"][0]["input_types"] = {"visit": "documents"}
        with self.assertRaisesRegex(ValueError, "cycle"):
            DeclarativeWorkflow.from_mapping(cycle)


class ProfileAndOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transitions = (
            _transition("a", chosen_rank=1),
            _transition("b", chosen_rank=1),
            _transition("c", chosen_rank=2),
        )
        self.mapper = URLRankMapper().fit(self.transitions, searches_seen=3)

    def test_profile_accounts_for_predictions_misses_and_overlap(self) -> None:
        demand = profile_configuration(
            self.mapper,
            self.transitions,
            CandidateConfiguration("k0", 0),
            role="tuning",
            bootstrap_samples=100,
        )
        top_one = profile_configuration(
            self.mapper,
            self.transitions,
            CandidateConfiguration("k1", 1),
            role="tuning",
            bootstrap_samples=100,
        )
        self.assertEqual(demand.concrete_predictions, 0)
        self.assertEqual(demand.authoritative_misses, 3)
        self.assertEqual(demand.admitted_tool_request_units, 3)
        self.assertEqual(demand.stall_reduction, 0.0)
        self.assertEqual(top_one.concrete_predictions, 3)
        self.assertEqual(top_one.exact_hits, 2)
        self.assertEqual(top_one.authoritative_misses, 1)
        self.assertEqual(top_one.admitted_tool_request_units, 4)
        self.assertAlmostEqual(top_one.stall_reduction, 1.0 / 3.0)
        self.assertIsNotNone(top_one.bootstrap_stall_reduction_95pct_ci)

    def test_optimizer_filters_slo_then_minimizes_resource_proxy(self) -> None:
        candidates = (
            CandidateConfiguration("k0", 0),
            CandidateConfiguration("k1", 1),
            CandidateConfiguration("k2", 2),
        )
        base_profiles = {
            candidate.config_id: profile_configuration(
                self.mapper,
                self.transitions,
                candidate,
                role="calibration",
            )
            for candidate in candidates
        }
        # Make a controlled frontier without relying on incidental mapper order.
        reductions = {"k0": 0.0, "k1": 0.2, "k2": 0.4}
        units = {"k0": 1.0, "k1": 1.4, "k2": 2.0}
        calibration = {
            key: replace(
                value,
                stall_reduction=reductions[key],
                admitted_tool_request_units_per_authoritative_invocation=units[key],
            )
            for key, value in base_profiles.items()
        }
        tuning = {
            key: replace(
                value,
                role="tuning",
                stall_reduction=reductions[key] - (0.0 if key == "k0" else 0.02),
                admitted_tool_request_units_per_authoritative_invocation=units[key] + 0.1,
            )
            for key, value in base_profiles.items()
        }
        plans = optimize_configurations(
            candidates,
            {"calibration": calibration, "tuning": tuning},
            (
                LatencySLO("basic", 0.0, 0.0, 1.0),
                LatencySLO("strict", 0.25, 0.05, 1.0),
            ),
        )
        self.assertEqual(plans[0]["selected"]["config_id"], "k0")
        self.assertEqual(plans[1]["selected"]["config_id"], "k2")
        self.assertEqual(
            plans[1]["selected"]["conservative_stall_reduction"], 0.38
        )

    def test_comparison_reports_slo_tradeoff(self) -> None:
        candidates = (
            CandidateConfiguration("k0", 0),
            CandidateConfiguration("k1", 1),
        )
        profiles = {
            candidate.config_id: profile_configuration(
                self.mapper, self.transitions, candidate, role="final"
            )
            for candidate in candidates
        }
        plans = [
            {
                "tier": "basic",
                "status": "selected",
                "minimum_stall_reduction": 0.0,
                "planning_margin": 0.0,
                "planning_threshold": 0.0,
                "demand_weight": 0.5,
                "selected": {"config_id": "k0", "top_k": 0},
            },
            {
                "tier": "strict",
                "status": "selected",
                "minimum_stall_reduction": 0.2,
                "planning_margin": 0.0,
                "planning_threshold": 0.2,
                "demand_weight": 0.5,
                "selected": {"config_id": "k1", "top_k": 1},
            },
        ]
        comparison = compare_plans(
            plans,
            profiles,
            static_demand_config_id="k0",
            static_paste_config_id="k1",
        )
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(
            comparison["policies"]["murakkab_inspired_paste"]["aggregate_slo_tiers_met"],
            2,
        )
        self.assertGreater(
            comparison["murakkab_vs_static_paste"][
                "admitted_tool_request_unit_reduction"
            ],
            0.0,
        )


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_preserves_exact_commit_boundary(self) -> None:
        transitions = (
            _transition("a", chosen_rank=1),
            _transition("b", chosen_rank=2),
        )
        mapper = URLRankMapper().fit(transitions, searches_seen=2)
        replay = await execute_isolated_replay(
            mapper,
            transitions,
            CandidateConfiguration("k2", 2),
            max_concurrency=2,
        )
        self.assertEqual(replay["state_isolation_violations"], 0)
        self.assertTrue(all(replay["reconciliation"].values()))
        self.assertEqual(replay["scheduler"]["commits"], 2)


if __name__ == "__main__":
    unittest.main()

