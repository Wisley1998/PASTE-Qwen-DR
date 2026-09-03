from __future__ import annotations

import asyncio
import math
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_open_loop_stress as stress  # noqa: E402
from paste_repro.speculation_policy import CandidatePattern  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredCandidate,
    ScoredWindow,
)


def window(index: int, *, exact: bool = False) -> ScoredWindow:
    decision_id = f"decision-{index}"
    target = f"https://target.test/{index}"
    candidate_url = target if exact else f"https://wrong.test/{index}"
    candidate = ScoredCandidate(
        pattern=CandidatePattern(
            session_id=f"session-{index}",
            decision_id=decision_id,
            url=candidate_url,
            position=1,
            query_count=1,
            search_streak=1,
            search_sequence=1,
            candidate_count=1,
            current_count=1,
            repeated_current=False,
            source_rank=1,
            current=True,
            was_visited=False,
            search_age=0,
            appearances=1,
        ),
        exact_probability=0.8,
        visit_probability=0.9,
        rank_only_probability=0.2,
        exact_match=exact,
    )
    return ScoredWindow(
        decision_id=decision_id,
        session_id=f"session-{index}",
        v2_gate=True,
        next_tool_visit=True,
        expected_authoritative_calls=1.0,
        coarse_expected_authoritative_calls=1.0,
        targets=(target,),
        executable_targets=(target,),
        candidates=(candidate,),
    )


class OpenLoopScheduleTests(unittest.TestCase):
    def test_url_call_utilization_is_derived_exactly(self) -> None:
        schedule, plan = stress.build_schedule(
            [window(index) for index in range(6)],
            offered_load=0.75,
            visit_capacity=2,
            service_s=0.010,
            lead_s=0.005,
            seed=7,
            cycles=2,
        )
        self.assertEqual(len(schedule), 12)
        self.assertTrue(
            math.isclose(
                float(plan["derived_authority_utilization"]),
                0.75,
                rel_tol=1e-12,
            )
        )
        self.assertEqual(len({row.instance_id for row in schedule}), 12)

    def test_concentration_helpers_have_explicit_all_zero_semantics(self) -> None:
        self.assertEqual(stress.top_fraction_share([0, 0, 0]), 0.0)
        self.assertEqual(stress.jain_index([0, 0, 0]), 0.0)
        self.assertEqual(stress.top_fraction_share([10, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(stress.jain_index([1, 1, 1, 1]), 1.0)

    def test_risk_limited_utility_coarse_gate_precedes_scoring(self) -> None:
        selected, metadata = stress._select_candidates(
            window(1),
            stress.policy_specs()["utility_risk_limited"],
            snapshot={
                "counts": {
                    "running_authoritative": 2,
                    "queued_authoritative": 3,
                }
            },
            offered_load=0.5,
            visit_capacity=2,
            service_s=0.010,
            lead_remaining_s=0.005,
        )
        self.assertEqual(selected, [])
        self.assertTrue(metadata["coarse_load_kill_switch"])
        self.assertEqual(metadata["predictor_windows_evaluated"], 0)
        self.assertEqual(metadata["reasons"]["coarse_load_kill_switch"], 1)

    def test_safe_policy_requires_certified_capacity(self) -> None:
        selected, metadata = stress._select_candidates(
            window(1, exact=True),
            stress.policy_specs()["safe_global_benefit"],
            snapshot={
                "counts": {
                    "running_authoritative": 0,
                    "queued_authoritative": 0,
                }
            },
            offered_load=0.5,
            visit_capacity=2,
            service_s=0.010,
            lead_remaining_s=0.005,
            isolated_speculative_slots=0,
        )
        self.assertEqual(selected, [])
        self.assertEqual(metadata["predictor_windows_evaluated"], 0)
        self.assertEqual(metadata["reasons"]["no_safe_capacity"], 1)


class OpenLoopBrokerSmokeTests(unittest.TestCase):
    def test_safe_open_loop_zero_certificate_is_demand_only_fast_path(
        self,
    ) -> None:
        sample = asyncio.run(
            stress.run_open_loop_sample(
                [window(index, exact=True) for index in range(4)],
                policy=stress.policy_specs()["safe_global_benefit"],
                offered_load=0.5,
                seed=9,
                cycles=1,
                workers=2,
                visit_capacity=2,
                max_speculative_pending=8,
                service_ms=4.0,
                lead_ms=2.0,
                isolated_speculative_slots=0,
            )
        )
        self.assertEqual(sample["broker_workers"], 2)
        self.assertEqual(sample["requested_predictions"], 0)
        self.assertEqual(sample["predictor_windows_evaluated"], 0)
        self.assertEqual(sample["physical_started"], 4)
        self.assertEqual(
            sample["selection_reason_counts"]["no_safe_capacity"], 4
        )
        self.assertTrue(all(sample["safety"].values()))

    def test_safe_open_loop_uses_only_isolated_slice(self) -> None:
        sample = asyncio.run(
            stress.run_open_loop_sample(
                [window(index, exact=True) for index in range(6)],
                policy=stress.policy_specs()["safe_global_benefit"],
                offered_load=0.9,
                seed=17,
                cycles=1,
                workers=2,
                visit_capacity=2,
                max_speculative_pending=16,
                service_ms=6.0,
                lead_ms=3.0,
                isolated_speculative_slots=1,
            )
        )
        self.assertEqual(sample["broker_workers"], 3)
        self.assertEqual(sample["broker_visit_capacity"], 3)
        self.assertEqual(sample["certified_isolated_speculative_slots"], 1)
        self.assertGreater(sample["requested_predictions"], 0)
        self.assertGreater(sample["overlap_hits"], 0)
        self.assertGreater(sample["speculative_race_wins"], 0)
        self.assertLessEqual(sample["max_running_speculative_visit"], 1)
        self.assertTrue(all(sample["safety"].values()))

    def test_all_wrong_drains_with_reserve_and_absolute_start_deadline(self) -> None:
        windows = [window(index) for index in range(6)]
        common = dict(
            offered_load=0.5,
            seed=11,
            cycles=1,
            workers=4,
            visit_capacity=2,
            max_speculative_pending=16,
            service_ms=10.0,
            lead_ms=5.0,
        )
        baseline = asyncio.run(
            stress.run_open_loop_sample(windows, policy=None, **common)
        )
        treatment = asyncio.run(
            stress.run_open_loop_sample(
                windows,
                policy=stress.policy_specs()["rank5_reserved"],
                **common,
            )
        )
        self.assertEqual(treatment["overlap_hits"], 0)
        self.assertGreater(treatment["wrong_started"], 0)
        self.assertGreater(treatment["wrong_service_ms"], 0.0)
        self.assertEqual(treatment["late_speculative_starts"], 0)
        self.assertLessEqual(treatment["max_running_speculative_visit"], 1)
        self.assertTrue(all(treatment["safety"].values()))

        baseline_aggregate = stress.aggregate_samples(
            [baseline, baseline],
            scenario="all_wrong_counterfactual",
            policy="demand_only",
            offered_load=0.5,
        )
        treatment_aggregate = stress.aggregate_samples(
            [treatment, treatment],
            scenario="all_wrong_counterfactual",
            policy="rank5_reserved",
            offered_load=0.5,
        )
        paired = stress.paired_metrics(baseline_aggregate, treatment_aggregate)
        self.assertEqual(paired["authoritative_targets"], 12)
        self.assertEqual(
            len(
                {
                    row["target_id"]
                    for row in treatment_aggregate["authoritative_rows"]
                }
            ),
            12,
        )
        self.assertEqual(paired["overlap_hits"], 0)
        self.assertTrue(paired["all_safety_invariants_passed"])

    def test_matrix_counterbalances_each_policy_pair_and_compacts_rows(self) -> None:
        cells = asyncio.run(
            stress.run_matrix(
                [window(index) for index in range(6)],
                policy_names=("utility_risk_limited",),
                loads=(1.2,),
                repetitions=2,
                cycles=1,
                workers=4,
                visit_capacity=2,
                max_speculative_pending=16,
                service_ms=4.0,
                lead_ms=2.0,
            )
        )
        self.assertEqual(len(cells), 2)
        for cell in cells:
            self.assertEqual(cell["counterbalance_orders"], ["AB", "BA"])
            self.assertEqual(
                len(cell["repeat_net_scheduled_response_benefit_ms_per_target"]),
                2,
            )
            for role in ("baseline", "treatment"):
                for sample in cell["samples"][role]:
                    self.assertNotIn("authoritative_rows", sample)


if __name__ == "__main__":
    unittest.main()
