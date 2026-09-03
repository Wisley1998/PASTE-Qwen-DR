from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from paste_repro.speculative_action_replay import (
    SpeculationCase,
    build_cases,
    evaluate_predictions,
    parse_predictions,
)


class SpeculativeActionReplayTests(unittest.TestCase):
    def test_prepare_is_causal_and_uses_corrected_duration(self) -> None:
        events = [
            {
                "event_type": "llm_call",
                "call_index": 0,
                "timestamp": 4.0,
                "total_time_ms": 3000,
                "inference_time_ms": 2500,
                "messages": [
                    {"role": "system", "content": "tools"},
                    {"role": "user", "content": "find evidence"},
                ],
                "response": "SECRET AUTHORITATIVE RESPONSE",
            },
            {
                "event_type": "tool_call",
                "call_index": 0,
                "timestamp": 4.0,
                "tool_name": "search",
                "tool_args": {"query": ["exact query"]},
                "timing_correction": {"duration_s": 2.25},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            cases, sessions = build_cases(Path(directory))
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(cases), 1)
        self.assertNotIn("SECRET AUTHORITATIVE RESPONSE", cases[0].prompt)
        self.assertNotIn("exact query", cases[0].prompt)
        self.assertAlmostEqual(cases[0].overlap_window_s, 2.5)
        self.assertAlmostEqual(cases[0].tool_duration_s, 2.25)

    def test_prediction_parser_is_exact_and_deduplicates(self) -> None:
        response = """```json
        {"predictions": [
          {"tool_name": "visit", "tool_args": {"goal": "g", "url": ["u"]}},
          {"name": "visit", "arguments": {"url": ["u"], "goal": "g"}},
          {"tool_name": "search", "tool_args": {"query": ["q"]}}
        ]}
        ```"""
        predictions = parse_predictions(response, top_k=3)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0].tool_name, "visit")
        self.assertEqual(predictions[1].tool_name, "search")

    def test_late_exact_hit_only_saves_available_head_start(self) -> None:
        case = SpeculationCase(
            case_id="case",
            session_id="session",
            llm_call_index=1,
            llm_line_number=1,
            tool_call_index=1,
            tool_line_number=2,
            overlap_window_s=3.0,
            llm_total_time_s=3.2,
            tool_duration_s=5.0,
            authoritative_tool_name="visit",
            authoritative_tool_args={"url": ["u"], "goal": "g"},
            prompt="causal",
            prompt_truncated=False,
        )
        rows = [
            {
                "case_id": "case",
                "latency_s": 1.0,
                "predictions": [
                    {
                        "tool_name": "visit",
                        "arguments": {"goal": "g", "url": ["u"]},
                    }
                ],
                "error": None,
            }
        ]
        report = evaluate_predictions([case], rows, top_k=3)
        summary = report["summary"]
        self.assertEqual(summary["exact_hits"], 1)
        self.assertAlmostEqual(summary["saved_tool_stall_s"], 2.0)
        self.assertAlmostEqual(summary["speculative_tool_stall_s"], 3.0)
        self.assertAlmostEqual(summary["tool_stall_reduction"], 0.4)


if __name__ == "__main__":
    unittest.main()
