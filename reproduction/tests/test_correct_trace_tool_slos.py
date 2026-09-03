from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import correct_trace_tool_slos as correction  # noqa: E402


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


class ToolSloCorrectionTests(unittest.TestCase):
    def test_search_and_serial_visit_are_resampled_with_causal_shifts(self) -> None:
        events = [
            llm(5.0, 2.0, 0),
            {
                "event_type": "tool_call",
                "call_index": 0,
                "timestamp": 5.0,
                "tool_name": "search",
                "tool_args": {"query": ["a", "b"]},
            },
            llm(20.0, 5.0, 1),  # observed search = 10 seconds
            {
                "event_type": "tool_call",
                "call_index": 1,
                "timestamp": 20.0,
                "tool_name": "visit",
                "tool_args": {
                    "url": ["https://a.invalid", "https://b.invalid"],
                    "goal": "g",
                },
            },
            llm(27.0, 4.0, 2),  # observed visit batch = 3 seconds
        ]
        rewritten, audit = correction.correct_events(
            events,
            seed="test-seed",
            session_id="session-a",
        )
        rows = audit["rows"]
        self.assertEqual([row["tool_name"] for row in rows], ["search", "visit"])
        search_s = float(rows[0]["corrected_s"])
        visit_s = float(rows[1]["corrected_s"])
        visit_units = tuple(float(value) for value in rows[1]["unit_duration_s"])
        self.assertTrue(1.0 <= search_s <= 3.0)
        self.assertEqual(len(visit_units), 2)
        self.assertTrue(all(2.0 <= value <= 8.0 for value in visit_units))
        self.assertAlmostEqual(visit_s, sum(visit_units))
        self.assertAlmostEqual(
            correction.llm_start_s(rewritten[2]) - rewritten[1]["timestamp"],
            search_s,
        )
        self.assertAlmostEqual(
            correction.llm_start_s(rewritten[4]) - rewritten[3]["timestamp"],
            visit_s,
        )
        self.assertEqual(rewritten[2]["total_time_ms"], 5000.0)
        self.assertEqual(rewritten[4]["total_time_ms"], 4000.0)
        self.assertEqual(
            rewritten[3]["tool_args"],
            events[3]["tool_args"],
        )
        self.assertEqual(
            rewritten[3]["timing_correction"]["execution"],
            "serial_sum_per_url",
        )

    def test_terminal_visit_adds_explicit_completion(self) -> None:
        events = [
            llm(4.0, 2.0, 0),
            {
                "event_type": "tool_call",
                "call_index": 0,
                "timestamp": 4.0,
                "tool_name": "visit",
                "tool_args": {
                    "url": ["https://a.invalid", "https://b.invalid"],
                    "goal": "g",
                },
            },
        ]
        rewritten, audit = correction.correct_events(
            events,
            seed="test-seed",
            session_id="terminal-session",
        )
        self.assertEqual(audit["terminal_completions_added"], 1)
        self.assertEqual(len(rewritten), 3)
        tool = rewritten[1]
        marker = rewritten[2]
        duration_s = tool["timing_correction"]["duration_s"]
        self.assertEqual(marker["event_type"], "synthetic_tool_completion")
        self.assertEqual(marker["tool_name"], "visit")
        self.assertAlmostEqual(marker["timestamp"], tool["timestamp"] + duration_s)

    def test_uniform_url_samples_are_reproducible_and_non_degenerate(self) -> None:
        samples = [
            correction.stable_uniform_duration_s(
                minimum_s=2.0,
                maximum_s=8.0,
                seed="stable-seed",
                session_id="session",
                event_index=index,
                call_index=index,
                tool_name="visit",
                unit_index=unit,
            )
            for index in range(10)
            for unit in range(3)
        ]
        repeated = [
            correction.stable_uniform_duration_s(
                minimum_s=2.0,
                maximum_s=8.0,
                seed="stable-seed",
                session_id="session",
                event_index=index,
                call_index=index,
                tool_name="visit",
                unit_index=unit,
            )
            for index in range(10)
            for unit in range(3)
        ]
        self.assertEqual(samples, repeated)
        self.assertTrue(all(2.0 <= value <= 8.0 for value in samples))
        self.assertEqual(len(samples), len(set(samples)))


if __name__ == "__main__":
    unittest.main()
