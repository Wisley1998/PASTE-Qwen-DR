from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import correct_trace_search_slo as correction  # noqa: E402


def llm(timestamp: float, duration_s: float, call_index: int) -> dict:
    return {
        "event_type": "llm_call",
        "call_index": call_index,
        "timestamp": timestamp,
        "total_time_ms": duration_s * 1000.0,
        "inference_time_ms": duration_s * 1000.0,
        "messages": [],
        "response": "",
    }


class SearchSloCorrectionTests(unittest.TestCase):
    def test_long_search_is_resampled_and_later_visit_gap_is_preserved(self) -> None:
        events = [
            llm(5.0, 2.0, 0),
            {
                "event_type": "tool_call",
                "call_index": 0,
                "timestamp": 5.0,
                "tool_name": "search",
                "tool_args": {"query": ["a", "b"]},
            },
            llm(20.0, 5.0, 1),  # search gap = 10 seconds
            {
                "event_type": "tool_call",
                "call_index": 1,
                "timestamp": 20.0,
                "tool_name": "visit",
                "tool_args": {"url": ["https://example.invalid"]},
            },
            llm(27.0, 4.0, 2),  # visit gap = 3 seconds
        ]
        target = correction.uniform_search_duration_s(
            min_search_s=1.0,
            max_search_s=3.0,
            seed="test-seed",
            session_id="session-a",
            event_index=1,
            call_index=0,
        )
        rewritten, audit = correction.correct_events(
            events,
            min_search_s=1.0,
            max_search_s=3.0,
            seed="test-seed",
            session_id="session-a",
        )
        removed = 10.0 - target
        self.assertAlmostEqual(audit["total_removed_s"], removed)
        self.assertAlmostEqual(rewritten[2]["timestamp"], 20.0 - removed)
        self.assertAlmostEqual(rewritten[3]["timestamp"], 20.0 - removed)
        self.assertAlmostEqual(rewritten[4]["timestamp"], 27.0 - removed)
        self.assertAlmostEqual(
            correction.llm_start_s(rewritten[2]) - rewritten[1]["timestamp"],
            target,
        )
        self.assertAlmostEqual(
            correction.llm_start_s(rewritten[4]) - rewritten[3]["timestamp"],
            3.0,
        )

    def test_short_search_is_resampled_without_changing_llm_duration(self) -> None:
        events = [
            {
                "event_type": "tool_call",
                "call_index": 0,
                "timestamp": 1.0,
                "tool_name": "search",
                "tool_args": {"query": "q"},
            },
            llm(3.5, 2.0, 1),  # search gap = 0.5 seconds
        ]
        target = correction.uniform_search_duration_s(
            min_search_s=1.0,
            max_search_s=3.0,
            seed="test-seed",
            session_id="session-b",
            event_index=0,
            call_index=0,
        )
        rewritten, audit = correction.correct_events(
            events,
            min_search_s=1.0,
            max_search_s=3.0,
            seed="test-seed",
            session_id="session-b",
        )
        self.assertAlmostEqual(audit["total_removed_s"], 0.5 - target)
        self.assertAlmostEqual(rewritten[1]["timestamp"], 3.0 + target)
        self.assertEqual(rewritten[1]["total_time_ms"], 2000.0)

    def test_uniform_samples_are_reproducible_and_not_degenerate(self) -> None:
        samples = [
            correction.uniform_search_duration_s(
                min_search_s=1.0,
                max_search_s=3.0,
                seed="stable-seed",
                session_id="session",
                event_index=index,
                call_index=index,
            )
            for index in range(20)
        ]
        repeated = [
            correction.uniform_search_duration_s(
                min_search_s=1.0,
                max_search_s=3.0,
                seed="stable-seed",
                session_id="session",
                event_index=index,
                call_index=index,
            )
            for index in range(20)
        ]
        self.assertEqual(samples, repeated)
        self.assertTrue(all(1.0 <= value <= 3.0 for value in samples))
        self.assertGreater(len(set(samples)), 1)

    def test_terminal_search_is_left_unchanged(self) -> None:
        events = [
            {
                "event_type": "tool_call",
                "call_index": 0,
                "timestamp": 1.0,
                "tool_name": "search",
                "tool_args": {"query": "q"},
            }
        ]
        rewritten, audit = correction.correct_events(
            events, min_search_s=1.0, max_search_s=3.0
        )
        self.assertEqual(rewritten, events)
        self.assertEqual(audit["terminal_searches_without_following_llm"], 1)


if __name__ == "__main__":
    unittest.main()
