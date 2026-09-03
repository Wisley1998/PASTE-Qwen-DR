from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_adaptive_load as adaptive  # noqa: E402


class PatternV2AdaptiveDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.windows, cls.metadata = adaptive.collect_nested_oof_windows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )

    def test_cross_fitted_candidate_counts_are_stable(self) -> None:
        self.assertEqual(len(self.windows), 340)
        self.assertEqual(self.metadata["candidate_count"], 1700)
        self.assertEqual(self.metadata["candidate_hits"], 134)
        quality = adaptive.calibration_quality(self.windows)
        self.assertGreater(
            quality["pattern_average_precision"],
            quality["rank_only_average_precision"],
        )
        self.assertLess(
            quality["pattern_brier"], quality["rank_only_brier"]
        )

    def test_session_stream_batches_preserve_order_and_never_overlap(self) -> None:
        batches = adaptive.session_stream_batches(
            self.windows,
            offered_concurrency=32,
            seed=7,
        )
        self.assertEqual(sum(map(len, batches)), len(self.windows))
        self.assertLessEqual(max(map(len, batches)), 32)
        self.assertTrue(
            all(
                len({window.session_id for window in batch}) == len(batch)
                for batch in batches
            )
        )
        original: dict[str, list[str]] = {}
        replayed: dict[str, list[str]] = {}
        for window in self.windows:
            original.setdefault(window.session_id, []).append(
                window.decision_id
            )
        for batch in batches:
            for window in batch:
                replayed.setdefault(window.session_id, []).append(
                    window.decision_id
                )
        self.assertEqual(replayed, original)

    def test_equal_budget_rank_control_and_global_confidence(self) -> None:
        batch = adaptive.session_stream_batches(
            self.windows,
            offered_concurrency=8,
            seed=3,
        )[0]
        specs = {spec.name: spec for spec in adaptive.policy_specs()}
        rank, _ = adaptive._select_candidates(
            batch,
            specs["rank_budgeted_round_robin_reserved"],
            visit_capacity=2,
            service_s=0.005,
            lead_s=0.0025,
        )
        confidence, _ = adaptive._select_candidates(
            batch,
            specs["confidence_global_reserved"],
            visit_capacity=2,
            service_s=0.005,
            lead_s=0.0025,
        )
        self.assertLessEqual(len(rank), 1)
        self.assertLessEqual(len(confidence), 1)
        self.assertEqual(
            adaptive._selection_budget(
                specs["confidence_global_reserved"],
                visit_capacity=2,
                service_s=0.005,
                lead_s=0.0,
            ),
            0,
        )

    def test_risk_limited_lookup_count_precedes_probability_filter(self) -> None:
        spec = {
            item.name: item for item in adaptive.policy_specs()
        }["utility_global_risk_limited"]
        threshold = float(spec.confidence_threshold)
        window = next(
            window
            for window in self.windows
            if 0
            < sum(
                candidate.exact_probability >= threshold
                for candidate in window.candidates
            )
            < len(window.candidates)
        )

        _, metadata = adaptive._select_candidates(
            [window],
            spec,
            visit_capacity=2,
            service_s=0.005,
            lead_s=0.0025,
        )

        eligible = sum(
            candidate.exact_probability >= threshold
            for candidate in window.candidates
        )
        self.assertEqual(metadata["considered"], eligible)
        self.assertEqual(
            metadata["probability_candidates_evaluated"],
            len(window.candidates),
        )
        self.assertGreater(
            metadata["probability_candidates_evaluated"],
            metadata["considered"],
        )

    def test_safe_policy_fails_closed_before_predictor_without_certificate(
        self,
    ) -> None:
        spec = {item.name: item for item in adaptive.policy_specs()}[
            "safe_global_benefit"
        ]
        batch = adaptive.session_stream_batches(
            self.windows,
            offered_concurrency=8,
            seed=5,
        )[0]
        selected, metadata = adaptive._select_candidates(
            batch,
            spec,
            visit_capacity=2,
            service_s=0.005,
            lead_s=0.0025,
            isolated_speculative_slots=0,
        )
        self.assertEqual(selected, [])
        self.assertEqual(metadata["safe_start_budget"], 0)
        self.assertEqual(metadata["predictor_windows_evaluated"], 0)
        self.assertIn("no_safe_capacity", metadata["selection_reason_counts"])

    def test_safe_policy_uses_one_global_certified_start(self) -> None:
        spec = {item.name: item for item in adaptive.policy_specs()}[
            "safe_global_benefit"
        ]
        batch = adaptive.session_stream_batches(
            self.windows,
            offered_concurrency=32,
            seed=3,
        )[0]
        selected, metadata = adaptive._select_candidates(
            batch,
            spec,
            visit_capacity=2,
            service_s=0.005,
            lead_s=0.0025,
            isolated_speculative_slots=1,
        )
        self.assertLessEqual(len(selected), 1)
        self.assertEqual(metadata["safe_start_budget"], 1)

        selected_two, metadata_two = adaptive._select_candidates(
            batch,
            spec,
            visit_capacity=2,
            service_s=0.005,
            lead_s=0.0025,
            isolated_speculative_slots=2,
        )
        self.assertEqual(len(selected_two), 2)
        self.assertEqual(metadata_two["safe_start_budget"], 2)

        selected_capped, metadata_capped = adaptive._select_candidates(
            batch,
            spec,
            visit_capacity=2,
            service_s=0.001,
            lead_s=0.003,
            isolated_speculative_slots=2,
            safe_start_limit=3,
        )
        self.assertEqual(len(selected_capped), 3)
        self.assertEqual(metadata_capped["safe_start_budget"], 3)


class PatternV2AdaptiveBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def test_safe_policy_zero_certificate_is_demand_only_fast_path(
        self,
    ) -> None:
        windows, _ = adaptive.collect_nested_oof_windows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )
        selected = [
            window for window in windows if window.executable_targets
        ][:3]
        sample = await adaptive._run_sample(
            selected,
            policy={spec.name: spec for spec in adaptive.policy_specs()}[
                "safe_global_benefit"
            ],
            offered_concurrency=3,
            seed=0,
            workers=2,
            visit_capacity=2,
            max_speculative_pending=8,
            service_ms=2.0,
            lead_ms=1.0,
            isolated_speculative_slots=0,
        )
        self.assertEqual(sample["broker_workers"], 2)
        self.assertEqual(sample["broker_visit_capacity"], 2)
        self.assertEqual(sample["requested_predictions"], 0)
        self.assertEqual(sample["predictor_windows_evaluated"], 0)
        self.assertEqual(
            sample["physical_started"], sample["authoritative_targets"]
        )
        self.assertEqual(
            sample["selection_reason_counts"]["no_safe_capacity"], 3
        )
        self.assertTrue(all(sample["safety"].values()))

    async def test_safe_policy_preserves_baseline_caps_with_isolated_slot(
        self,
    ) -> None:
        windows, _ = adaptive.collect_nested_oof_windows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )
        selected = [
            window
            for window in windows
            if window.executable_targets
            and any(
                candidate.exact_probability >= 0.20
                for candidate in window.candidates
            )
        ][:4]
        sample = await adaptive._run_sample(
            adaptive.force_all_wrong(selected),
            policy={spec.name: spec for spec in adaptive.policy_specs()}[
                "safe_global_benefit"
            ],
            offered_concurrency=4,
            seed=0,
            workers=2,
            visit_capacity=2,
            max_speculative_pending=8,
            service_ms=2.0,
            lead_ms=1.0,
            isolated_speculative_slots=1,
        )
        self.assertEqual(sample["broker_workers"], 3)
        self.assertEqual(sample["broker_visit_capacity"], 3)
        self.assertEqual(sample["certified_isolated_speculative_slots"], 1)
        self.assertGreater(sample["requested_predictions"], 0)
        self.assertTrue(all(sample["safety"].values()))

    async def test_safe_exact_race_drains_loser_before_final_snapshot(
        self,
    ) -> None:
        windows, _ = adaptive.collect_nested_oof_windows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )
        policy = {spec.name: spec for spec in adaptive.policy_specs()}[
            "safe_global_benefit"
        ]
        selected_window = None
        for window in windows:
            selected, _ = adaptive._select_candidates(
                [window],
                policy,
                visit_capacity=2,
                service_s=0.020,
                lead_s=0.010,
                isolated_speculative_slots=1,
            )
            if (
                selected
                and len(window.executable_targets) == 1
                and selected[0][0].pattern.url
                == window.executable_targets[0]
            ):
                selected_window = window
                break
        self.assertIsNotNone(selected_window)

        sample = await adaptive._run_sample(
            [selected_window],
            policy=policy,
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            max_speculative_pending=8,
            service_ms=20.0,
            lead_ms=10.0,
            isolated_speculative_slots=1,
        )

        self.assertEqual(sample["running_speculative_races"], 1)
        self.assertEqual(sample["speculative_race_wins"], 1)
        self.assertTrue(all(sample["safety"].values()))

    async def test_zero_target_search_window_returns_zero_wait_metrics(self) -> None:
        windows, _ = adaptive.collect_nested_oof_windows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )
        selected = next(
            window for window in windows if not window.executable_targets
        )
        policy = {spec.name: spec for spec in adaptive.policy_specs()}[
            "utility_global_risk_limited"
        ]

        sample = await adaptive._run_sample(
            [selected],
            policy=policy,
            offered_concurrency=1,
            seed=0,
            workers=4,
            visit_capacity=2,
            max_speculative_pending=8,
            service_ms=1.0,
            lead_ms=0.5,
        )

        self.assertEqual(sample["authoritative_targets"], 0)
        self.assertEqual(sample["mean_exposed_wait_ms"], 0.0)
        self.assertEqual(sample["p95_exposed_wait_ms"], 0.0)
        self.assertGreater(sample["wall_s"], 0.0)
        self.assertTrue(all(sample["safety"].values()))

    async def test_all_wrong_sample_respects_reserve_and_start_deadline(self) -> None:
        windows, _ = adaptive.collect_nested_oof_windows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )
        selected = []
        seen_sessions: set[str] = set()
        for window in windows:
            if (
                window.executable_targets
                and window.candidates
                and window.session_id not in seen_sessions
            ):
                selected.append(window)
                seen_sessions.add(window.session_id)
            if len(selected) == 4:
                break
        sample = await adaptive._run_sample(
            adaptive.force_all_wrong(selected),
            policy={spec.name: spec for spec in adaptive.policy_specs()}[
                "rank_budgeted_round_robin_reserved"
            ],
            offered_concurrency=4,
            seed=0,
            workers=2,
            visit_capacity=2,
            max_speculative_pending=8,
            service_ms=2.0,
            lead_ms=1.0,
        )
        self.assertEqual(sample["exact_hits"], 0)
        self.assertEqual(sample["overlap_hits"], 0)
        self.assertLessEqual(
            sample["max_running_speculative_by_tool"].get("visit", 0),
            1,
        )
        self.assertTrue(all(sample["safety"].values()))


if __name__ == "__main__":
    unittest.main()
