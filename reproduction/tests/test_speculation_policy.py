from __future__ import annotations

import math
import unittest
from dataclasses import replace

from paste_repro.speculation_policy import (
    AuthorityFirstUtilityPolicy,
    AuthorityLoad,
    CandidatePattern,
    CountPatternCalibrator,
    LabeledCandidatePattern,
    SafeGlobalBenefitConfig,
    SafeGlobalBenefitPolicy,
    SafeStartBudget,
    UtilityCandidate,
    UtilityPolicyConfig,
)


def pattern(
    decision_id: str,
    *,
    position: int = 1,
    query_count: int = 1,
    search_streak: int = 1,
    repeated: bool = False,
) -> CandidatePattern:
    return CandidatePattern(
        session_id=f"session-{decision_id}",
        decision_id=decision_id,
        url=f"https://example.test/{decision_id}/{position}",
        position=position,
        query_count=query_count,
        search_streak=search_streak,
        search_sequence=1,
        candidate_count=5,
        current_count=5,
        repeated_current=repeated,
        source_rank=position,
        current=True,
        was_visited=False,
        search_age=0,
        appearances=2 if repeated else 1,
    )


class CountPatternCalibratorTests(unittest.TestCase):
    def test_probability_uses_patterns_and_preserves_hard_abstain(self) -> None:
        rows = []
        for index in range(20):
            high = pattern(f"high-{index}", repeated=True)
            low = pattern(f"low-{index}", position=2, query_count=6)
            rows.extend(
                (
                    LabeledCandidatePattern(
                        high,
                        next_tool_visit=True,
                        exact_match=index < 12,
                    ),
                    LabeledCandidatePattern(
                        low,
                        next_tool_visit=index < 10,
                        exact_match=False,
                    ),
                )
            )
        calibrator = CountPatternCalibrator(rows)
        self.assertGreater(
            calibrator.exact_probability(pattern("new-high", repeated=True)),
            calibrator.exact_probability(
                pattern("new-low", position=2, query_count=6)
            ),
        )
        self.assertEqual(
            calibrator.exact_probability(
                pattern(
                    "hard-abstain",
                    query_count=10,
                    search_streak=2,
                )
            ),
            0.0,
        )
        self.assertFalse(calibrator.summary()["neural_model"])


class AuthorityFirstUtilityPolicyTests(unittest.TestCase):
    def candidate(self, name: str, probability: float) -> UtilityCandidate:
        return UtilityCandidate(
            pattern(name),
            exact_probability=probability,
            estimated_service_s=0.010,
            lead_remaining_s=0.005,
        )

    def test_global_budget_is_not_round_robin(self) -> None:
        policy = AuthorityFirstUtilityPolicy(
            UtilityPolicyConfig(
                idle_shadow_price=0.01,
                medium_shadow_price=0.01,
                high_shadow_price=0.01,
                saturated_shadow_price=0.01,
            )
        )
        selection = policy.select(
            (
                self.candidate("first-low", 0.10),
                self.candidate("second-high", 0.80),
                self.candidate("third-medium", 0.40),
            ),
            load=AuthorityLoad(0.0, tool_capacity=2),
            start_budget=1,
        )
        self.assertEqual(len(selection.selected), 1)
        self.assertEqual(
            selection.selected[0].candidate.pattern.decision_id,
            "second-high",
        )

    def test_authoritative_backlog_is_a_hard_kill_switch(self) -> None:
        policy = AuthorityFirstUtilityPolicy()
        selection = policy.select(
            (self.candidate("very-high", 0.99),),
            load=AuthorityLoad(
                0.0,
                tool_capacity=2,
                authoritative_queued=1,
            ),
            start_budget=1,
        )
        self.assertEqual(selection.selected, ())
        self.assertEqual(
            selection.decisions[0].reason,
            "authoritative_backlog",
        )

    def test_load_pressure_raises_the_cost_of_wrong_work(self) -> None:
        policy = AuthorityFirstUtilityPolicy()
        idle = policy.select(
            (self.candidate("candidate", 0.20),),
            load=AuthorityLoad(0.2, tool_capacity=2),
            start_budget=1,
        )
        saturated = policy.select(
            (self.candidate("candidate", 0.20),),
            load=AuthorityLoad(8.0, tool_capacity=2),
            start_budget=1,
        )
        self.assertEqual(len(idle.selected), 1)
        self.assertEqual(saturated.selected, ())
        self.assertGreater(saturated.shadow_price, idle.shadow_price)


class SafeGlobalBenefitPolicyTests(unittest.TestCase):
    @staticmethod
    def candidate(
        decision_id: str,
        probability: float,
        *,
        session_id: str | None = None,
        position: int = 1,
        service_s: float = 0.010,
        lead_s: float = 0.005,
    ) -> UtilityCandidate:
        candidate_pattern = pattern(decision_id, position=position)
        if session_id is not None:
            candidate_pattern = replace(
                candidate_pattern,
                session_id=session_id,
            )
        return UtilityCandidate(
            candidate_pattern,
            exact_probability=probability,
            estimated_service_s=service_s,
            lead_remaining_s=lead_s,
        )

    def test_zero_safe_capacity_abstains_even_for_certain_hit(self) -> None:
        result = SafeGlobalBenefitPolicy().select(
            (self.candidate("certain", 1.0),),
            safe_budget=SafeStartBudget(0),
        )
        self.assertEqual(result.selected, ())
        self.assertEqual(result.start_budget, 0)
        self.assertEqual(result.decisions[0].reason, "no_safe_capacity")

    def test_coordination_cost_filters_theoretically_positive_benefit(self) -> None:
        policy = SafeGlobalBenefitPolicy(
            SafeGlobalBenefitConfig(coordination_cost_s=0.001)
        )
        result = policy.select(
            (self.candidate("too-small", 0.10, lead_s=0.005),),
            safe_budget=SafeStartBudget(1),
        )
        self.assertEqual(result.selected, ())
        self.assertAlmostEqual(result.decisions[0].net_utility_s, -0.0005)
        self.assertEqual(
            result.decisions[0].reason,
            "nonpositive_net_benefit",
        )

    def test_coordination_cost_keeps_only_positive_net_benefit(self) -> None:
        policy = SafeGlobalBenefitPolicy(
            SafeGlobalBenefitConfig(coordination_cost_s=0.001)
        )
        result = policy.select(
            (
                self.candidate("below", 0.10, lead_s=0.005),
                self.candidate("above", 0.40, lead_s=0.005),
            ),
            safe_budget=SafeStartBudget(2),
        )
        self.assertEqual(
            [row.candidate.pattern.decision_id for row in result.selected],
            ["above"],
        )

    def test_coordination_cost_must_be_finite_and_nonnegative(self) -> None:
        for invalid in (-0.001, math.inf, math.nan, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    SafeGlobalBenefitConfig(coordination_cost_s=invalid)

    def test_invalid_resource_state_fails_closed(self) -> None:
        result = SafeGlobalBenefitPolicy().select(
            (self.candidate("certain", 1.0),),
            safe_budget=SafeStartBudget(1, state_valid=False),
        )
        self.assertEqual(result.selected, ())
        self.assertEqual(
            result.decisions[0].reason,
            "invalid_resource_state",
        )

    def test_global_top_budget_prefers_expected_saved_latency(self) -> None:
        result = SafeGlobalBenefitPolicy().select(
            (
                self.candidate("low", 0.20),
                self.candidate("high", 0.90),
                self.candidate("medium", 0.50),
            ),
            safe_budget=SafeStartBudget(2),
        )
        self.assertEqual(
            [row.candidate.pattern.decision_id for row in result.selected],
            ["high", "medium"],
        )

    def test_cardinality_budget_uses_benefit_not_service_density(self) -> None:
        result = SafeGlobalBenefitPolicy().select(
            (
                self.candidate(
                    "larger-benefit",
                    0.80,
                    service_s=0.010,
                    lead_s=0.005,
                ),
                self.candidate(
                    "larger-density",
                    0.90,
                    service_s=0.003,
                    lead_s=0.003,
                ),
            ),
            safe_budget=SafeStartBudget(1),
        )
        self.assertEqual(
            result.selected[0].candidate.pattern.decision_id,
            "larger-benefit",
        )

    def test_one_candidate_per_session_decision(self) -> None:
        result = SafeGlobalBenefitPolicy().select(
            (
                self.candidate(
                    "same-decision",
                    0.90,
                    session_id="same-session",
                    position=1,
                ),
                self.candidate(
                    "same-decision",
                    0.80,
                    session_id="same-session",
                    position=2,
                ),
                self.candidate(
                    "other-decision",
                    0.70,
                    session_id="other-session",
                ),
            ),
            safe_budget=SafeStartBudget(2),
        )
        self.assertEqual(len(result.selected), 2)
        self.assertEqual(
            {
                (
                    row.candidate.pattern.session_id,
                    row.candidate.pattern.decision_id,
                )
                for row in result.selected
            },
            {
                ("same-session", "same-decision"),
                ("other-session", "other-decision"),
            },
        )
        self.assertIn(
            "per_decision_cap",
            {row.reason for row in result.decisions},
        )

    def test_duplicate_rows_never_exceed_budget(self) -> None:
        duplicate = self.candidate("duplicate", 0.90)
        result = SafeGlobalBenefitPolicy().select(
            (duplicate, duplicate),
            safe_budget=SafeStartBudget(1),
        )
        self.assertEqual(len(result.selected), 1)

    def test_same_decision_id_in_different_sessions_is_independent(self) -> None:
        result = SafeGlobalBenefitPolicy().select(
            (
                self.candidate("shared", 0.90, session_id="session-a"),
                self.candidate("shared", 0.80, session_id="session-b"),
            ),
            safe_budget=SafeStartBudget(2),
        )
        self.assertEqual(len(result.selected), 2)


if __name__ == "__main__":
    unittest.main()
