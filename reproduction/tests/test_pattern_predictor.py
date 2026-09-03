from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from paste_repro.pattern_predictor import (
    FROZEN_HISTORY_CAPACITY,
    FROZEN_SMOOTHING,
    FROZEN_VISITED_CAPACITY,
    GateAbstainRule,
    PATTERN_POLICY_VERSION,
    RankRecencyPatternPredictor,
    load_pattern_artifact,
    save_pattern_artifact,
)
from paste_repro.traces import SearchResult


def _result(url: str, rank: int, ordinal: int = 0, query_index: int = 0) -> SearchResult:
    return SearchResult(
        url=url,
        result_rank=rank,
        ordinal=ordinal,
        query_index=query_index,
    )


def _resign(artifact: dict) -> None:
    artifact.pop("artifact_sha256", None)
    encoded = json.dumps(
        artifact,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()


class RankRecencyPatternPredictorTests(unittest.TestCase):
    def test_current_response_is_unbounded_while_history_is_bounded(self) -> None:
        predictor = RankRecencyPatternPredictor({1: 1, 2: 10})
        session = predictor.start_session("s")
        rows = [
            _result(f"https://example.test/{index}", 2, index)
            for index in range(FROZEN_HISTORY_CAPACITY + 6)
        ]

        decision = session.observe_search(rows, query_count=1)

        self.assertEqual(decision.candidate_count, FROZEN_HISTORY_CAPACITY + 6)
        self.assertEqual(len(decision.ranked_top_k), 5)
        self.assertEqual(len(session.history_urls), FROZEN_HISTORY_CAPACITY)
        self.assertEqual(
            decision.cache["history_eviction_count"],
            6,
        )
        self.assertFalse(decision.cache["current_response_bounded"])

    def test_current_top1_is_preserved_then_history_fills_top5(self) -> None:
        predictor = RankRecencyPatternPredictor({1: 1, 2: 100}, top_k=5)
        session = predictor.start_session("s")
        historical = "https://example.test/historical-rank-two"
        current = "https://example.test/current-rank-one"
        session.observe_search([_result(historical, 2)], query_count=1)

        decision = session.observe_search([_result(current, 1)], query_count=1)

        # History has the higher raw score, but the frozen policy protects the
        # best current-response candidate in the first slot.
        self.assertEqual(decision.prediction_urls, (current, historical))
        self.assertTrue(decision.ranked_top_k[0].current)
        self.assertFalse(decision.ranked_top_k[1].current)
        self.assertEqual(decision.ranked_top_k[1].search_age, 1)

    def test_visited_current_url_cannot_change_legacy_m0_anchor(self) -> None:
        predictor = RankRecencyPatternPredictor({1: 10, 2: 9})
        session = predictor.start_session("s")
        legacy_anchor = "https://example.test/visited-rank-one"
        session.observe_search([_result(legacy_anchor, 1)], query_count=1)
        session.observe_visit(legacy_anchor)

        decision = session.observe_search(
            [
                _result(legacy_anchor, 1, 0),
                _result("https://example.test/unvisited-rank-two", 2, 1),
            ],
            query_count=1,
        )

        self.assertTrue(decision.ranked_top_k[0].was_visited)
        self.assertEqual(decision.prediction_urls[0], legacy_anchor)

    def test_only_preceding_two_searches_are_candidate_history(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10}, top_k=5)
        session = predictor.start_session("s")
        oldest = "https://example.test/oldest"
        session.observe_search([_result(oldest, 2)], query_count=1)
        session.observe_search([_result("https://example.test/middle", 2)], query_count=1)
        third = session.observe_search([_result("https://example.test/current", 2)], query_count=1)
        fourth = session.observe_search([_result("https://example.test/new", 2)], query_count=1)

        self.assertIn(oldest, third.prediction_urls)
        self.assertNotIn(oldest, fourth.prediction_urls)
        self.assertIn(oldest, session.history_urls)  # retained in bounded LRU, not eligible

    def test_visited_is_causal_penalty_not_a_hard_filter(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10}, top_k=5)
        session = predictor.start_session("s")
        url = "https://example.test/raw%2FCase"
        before_commit = session.observe_search([_result(url, 2)], query_count=1)
        self.assertFalse(before_commit.ranked_top_k[0].was_visited)

        session.observe_visit(url)
        after_commit = session.observe_search(
            [_result("https://example.test/current", 2)], query_count=1
        )
        historical = next(item for item in after_commit.ranked_top_k if item.url == url)

        self.assertTrue(historical.was_visited)
        self.assertIn(url, after_commit.prediction_urls)
        expected = predictor.score(rank=2, search_age=1, was_visited=True)
        self.assertAlmostEqual(historical.score, expected)

    def test_frozen_abstain_rule_and_default_admit(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10})
        session = predictor.start_session("s")
        row = [_result("https://example.test/u", 2)]

        first = session.observe_search(row, query_count=10)
        second = session.observe_search(row, query_count=10)
        third = session.observe_search(row, query_count=10)

        self.assertTrue(first.gate.admitted)
        self.assertEqual(first.gate.reason, "no_rule_match_admit")
        self.assertFalse(second.gate.admitted)
        self.assertEqual(second.gate.reason, "matched_abstain_pattern")
        self.assertEqual(second.prediction_urls, ())
        self.assertTrue(second.ranked_top_k)
        self.assertTrue(third.gate.admitted)  # equality rule is streak == 2

        session.observe_visit("https://example.test/u")
        reset = session.observe_search(row, query_count=10)
        self.assertEqual(reset.gate.consecutive_search_streak, 1)
        self.assertTrue(reset.gate.admitted)

    def test_no_url_abstains_even_when_pattern_default_is_admit(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10})
        decision = predictor.start_session("s").observe_search([], query_count=1)
        self.assertFalse(decision.gate.admitted)
        self.assertEqual(decision.gate.reason, "no_candidates")
        self.assertEqual(decision.prediction_urls, ())

    def test_exact_raw_url_identity_and_session_isolation(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10})
        left = predictor.start_session("left")
        right = predictor.start_session("right")
        upper = "https://EXAMPLE.test/A%2Fb"
        lower = "https://example.test/A%2fb"

        decision = left.observe_search(
            [_result(upper, 2, 0), _result(lower, 2, 1)], query_count=1
        )

        self.assertEqual(set(decision.prediction_urls), {upper, lower})
        self.assertEqual(right.history_urls, ())
        right_decision = right.observe_search(
            [_result("https://other.test/u", 2)], query_count=1
        )
        self.assertNotIn(upper, right_decision.prediction_urls)

        left.close()
        self.assertEqual(left.history_urls, ())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            left.observe_search([_result(upper, 2)], query_count=1)

    def test_artifact_round_trip_and_checksum_failure(self) -> None:
        predictor = RankRecencyPatternPredictor({1: 2, 2: 9})
        artifact = predictor.to_artifact(
            {"algorithm": "unit-test-patterns", "sessions": ["a", "b"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pattern.json"
            save_pattern_artifact(path, artifact)
            loaded, raw = load_pattern_artifact(path)
            self.assertEqual(loaded.rank_counts, {1: 2, 2: 9})
            self.assertEqual(loaded.policy, PATTERN_POLICY_VERSION)
            self.assertEqual(loaded.artifact_sha256, raw["artifact_sha256"])

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["rank_counts"]["2"] += 1
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_pattern_artifact(path)

    def test_policy_v2_defaults_and_literal_score(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10})

        self.assertEqual(predictor.smoothing, FROZEN_SMOOTHING)
        self.assertAlmostEqual(
            predictor.score(rank=2, search_age=1, was_visited=True),
            math.log(10.5) - 1.5 - 1.0,
        )
        self.assertEqual(
            predictor.metadata()["score_formula"],
            "log(rank_count+0.5)-1.5*search_age-1.0*was_visited",
        )

    def test_rank_counts_are_read_only(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10})

        with self.assertRaises(TypeError):
            predictor.rank_counts[2] = 99  # type: ignore[index]
        self.assertEqual(dict(predictor.rank_counts), {2: 10})

    def test_constructor_rejects_every_non_v2_policy_parameter(self) -> None:
        cases = (
            {"top_k": 4},
            {"history_capacity": 63},
            {"visited_capacity": 63},
            {"max_history_search_age": 1},
            {"smoothing": 1.0},
            {"search_age_penalty": 1.0},
            {"visited_penalty": 0.5},
            {"gate_rules": ()},
            {
                "gate_rules": (
                    GateAbstainRule("wrong", 9, 2),
                )
            },
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "policy v2"):
                    RankRecencyPatternPredictor({2: 10}, **kwargs)

    def test_resigned_non_v2_artifacts_are_rejected(self) -> None:
        artifact = RankRecencyPatternPredictor({2: 10}).to_artifact()
        config_cases = (
            ("top_k", 4),
            ("history_capacity", 63),
            ("visited_capacity", 63),
            ("max_history_search_age", 1),
            ("smoothing", 1.0),
            ("search_age_penalty", 1.0),
            ("visited_penalty", 0.5),
        )
        for field, value in config_cases:
            tampered = json.loads(json.dumps(artifact))
            tampered["config"][field] = value
            _resign(tampered)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "policy v2"):
                    RankRecencyPatternPredictor.from_artifact(tampered)

        tampered = json.loads(json.dumps(artifact))
        tampered["gate_rules"] = [
            GateAbstainRule("wrong-name-with-same-pattern", 10, 2).to_dict()
        ]
        _resign(tampered)
        with self.assertRaisesRegex(ValueError, "policy v2"):
            RankRecencyPatternPredictor.from_artifact(tampered)

    def test_resigned_v1_artifact_is_rejected(self) -> None:
        artifact = RankRecencyPatternPredictor({2: 10}).to_artifact()
        artifact["version"] = 1
        artifact["policy"] = "rank-recency-visited-cache-gate-v1"
        artifact["config"]["search_age_penalty"] = 1.0
        _resign(artifact)

        with self.assertRaisesRegex(ValueError, "version or policy"):
            RankRecencyPatternPredictor.from_artifact(artifact)

    def test_visited_lru_is_frozen_and_bounded(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10})
        session = predictor.start_session("s")
        urls = [
            f"https://example.test/visited/{index}"
            for index in range(FROZEN_VISITED_CAPACITY + 1)
        ]

        telemetry = session.observe_visit(urls)

        self.assertEqual(len(session.visited_urls), FROZEN_VISITED_CAPACITY)
        self.assertNotIn(urls[0], session.visited_urls)
        self.assertEqual(telemetry["visited_eviction_count"], 1)

    def test_source_call_index_requires_a_nonnegative_integer(self) -> None:
        predictor = RankRecencyPatternPredictor({2: 10})
        row = [_result("https://example.test/u", 2)]
        for value in (True, 1.5, "1", -1):
            session = predictor.start_session(f"s-{value!r}")
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "source_call_index"):
                    session.observe_search(row, query_count=1, source_call_index=value)
                self.assertEqual(session.search_sequence, 0)

    def test_simple_mapping_accepts_legacy_mapper_shape(self) -> None:
        predictor = RankRecencyPatternPredictor.from_mapping(
            {"mapper": {"rank_counts": {"2": 12, "4": 3}}}
        )
        self.assertEqual(predictor.rank_counts, {2: 12, 4: 3})
        self.assertEqual(predictor.metadata()["neural_model"], False)


if __name__ == "__main__":
    unittest.main()
