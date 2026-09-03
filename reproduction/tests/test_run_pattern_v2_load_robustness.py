from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_load_robustness as robustness  # noqa: E402


class PatternV2StaticRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.oof = robustness.collect_pattern_v2_oof_rows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )

    def test_grouped_oof_runtime_prefix_counts_and_oracle_are_frozen(self) -> None:
        metrics = robustness.static_width_metrics(
            self.rows, robustness.DEFAULT_WIDTHS
        )
        by_width = {row["width"]: row for row in metrics}
        self.assertEqual(self.oof["search_decisions"], 340)
        self.assertEqual(by_width[1]["target_hits"], 50)
        self.assertEqual(by_width[1]["authoritative_targets"], 236)
        self.assertEqual(by_width[1]["requested_candidates"], 314)
        self.assertEqual(by_width[5]["target_hits"], 134)
        self.assertEqual(by_width[5]["requested_candidates"], 1570)
        self.assertEqual(by_width[5]["logical_waste_candidates"], 1436)

        oracle = robustness.bounded_pool_oracle_metrics(self.rows)
        self.assertFalse(oracle["runtime_dispatch"])
        self.assertEqual(oracle["covered_targets"], 219)
        self.assertEqual(oracle["candidate_count_if_all_fired"], 15486)

    def test_all_wrong_preserves_firing_and_target_count_without_overlap(self) -> None:
        original = robustness.build_replay_opportunities(self.rows)
        wrong = robustness.force_all_wrong(original)
        self.assertEqual(len(wrong), len(original))
        self.assertEqual(
            sum(len(item.predictions) for item in wrong),
            sum(len(item.predictions) for item in original),
        )
        self.assertEqual(
            sum(len(item.executable_targets) for item in wrong),
            sum(len(item.executable_targets) for item in original),
        )
        for item in wrong:
            self.assertFalse(
                set(item.predictions).intersection(item.executable_targets)
            )


class PatternV2SharedBrokerSmokeTests(unittest.TestCase):
    def test_all_wrong_shared_pool_commits_only_authoritative_results(self) -> None:
        opportunities = [
            robustness.ReplayOpportunity(
                decision_id="d1",
                predictions=("https://pred.invalid/1", "https://pred.invalid/2"),
                executable_targets=("https://target.invalid/1",),
            ),
            robustness.ReplayOpportunity(
                decision_id="d2",
                predictions=("https://pred.invalid/3",),
                executable_targets=("https://target.invalid/2",),
            ),
        ]
        row = asyncio.run(
            robustness.paired_broker_cell(
                opportunities,
                scenario="all_wrong_counterfactual",
                width=2,
                offered_concurrency=2,
                repetitions=1,
                workers=2,
                speculative_workers=1,
                visit_capacity=1,
                max_speculative_pending=4,
                service_ms=2.0,
                lead_ms=1.0,
            )
        )
        self.assertEqual(row["authoritative_targets"], 2)
        self.assertEqual(row["exact_hits"], 0)
        self.assertEqual(row["overlap_producing_hits"], 0)
        self.assertEqual(row["saved_speculative_service_ms"], 0.0)
        self.assertGreater(row["wrong_speculations_started"], 0)
        self.assertGreater(row["wasted_speculative_service_ms"], 0.0)
        self.assertGreater(row["baseline_drained_workload_wall_s"], 0.0)
        self.assertGreater(row["pattern_drained_workload_wall_s"], 0.0)
        self.assertEqual(row["admission_batches"], 1)
        self.assertTrue(row["all_safety_invariants_passed"])


if __name__ == "__main__":
    unittest.main()
