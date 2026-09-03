from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from paste_repro.traces import (
    LLMCall,
    SessionTrace,
    ToolCall,
    TraceFormatError,
    extract_search_visit_transitions,
    load_trace,
    split_sessions,
)


def _event_lines() -> list[dict]:
    search_response = """<tool_response>
A search found 2 results:

1. [first](https://example.test/first)
2. [second](https://example.test/second)
=======
A second search found 1 result:
1. [third](https://example.test/third)
</tool_response>"""
    return [
        {
            "event_type": "llm_call",
            "call_index": 0,
            "timestamp": 1.0,
            "total_time_ms": 1000.0,
            "inference_time_ms": 900.0,
            "messages": [],
            "response": "search",
        },
        {
            "event_type": "tool_call",
            "call_index": 0,
            "timestamp": 1.0,
            "tool_name": "search",
            "tool_args": {"query": ["topic"]},
        },
        {
            "event_type": "llm_call",
            "call_index": 1,
            "timestamp": 5.0,
            "total_time_ms": 2000.0,
            "inference_time_ms": 1500.0,
            "messages": [{"role": "user", "content": search_response}],
            "response": "visit",
        },
        {
            "event_type": "tool_call",
            "call_index": 1,
            "timestamp": 5.0,
            "tool_name": "visit",
            "tool_args": {
                "url": ["https://example.test/second"],
                "goal": "read it",
            },
        },
        {
            "event_type": "llm_call",
            "call_index": 2,
            "timestamp": 10.0,
            "total_time_ms": 1000.0,
            "inference_time_ms": 800.0,
            "messages": [],
            "response": "done",
        },
    ]


class TraceParserTests(unittest.TestCase):
    def test_parser_and_search_visit_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in _event_lines()),
                encoding="utf-8",
            )
            session = load_trace(path)

        self.assertEqual(len(session.events), 5)
        self.assertIsInstance(session.events[0], LLMCall)
        self.assertIsInstance(session.events[1], ToolCall)
        transitions = extract_search_visit_transitions(session)
        self.assertEqual(len(transitions), 1)
        transition = transitions[0]
        self.assertEqual(
            [result.url for result in transition.search_results],
            [
                "https://example.test/first",
                "https://example.test/second",
                "https://example.test/third",
            ],
        )
        self.assertEqual([result.query_index for result in transition.search_results], [0, 0, 1])
        self.assertEqual(
            [result.title for result in transition.search_results],
            ["first", "second", "third"],
        )
        self.assertEqual(
            [result.query for result in transition.search_results],
            ["topic", "topic", ""],
        )
        self.assertEqual(transition.authoritative_urls, ("https://example.test/second",))
        # Completion LLM starts at t=9; visit was issued at t=5.
        self.assertAlmostEqual(transition.baseline_stall_s, 4.0)
        self.assertAlmostEqual(transition.overlap_window_s, 1.5)

    def test_invalid_json_has_file_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text("{broken}\n", encoding="utf-8")
            with self.assertRaisesRegex(TraceFormatError, r"broken\.jsonl:1"):
                load_trace(path)

    def test_split_is_deterministic_and_keeps_whole_sessions(self) -> None:
        sessions = tuple(
            SessionTrace(Path(f"session-{index}.jsonl"), ()) for index in range(10)
        )
        train_a, test_a = split_sessions(sessions, seed="unit-test")
        train_b, test_b = split_sessions(tuple(reversed(sessions)), seed="unit-test")
        self.assertEqual(len(train_a), 7)
        self.assertEqual(len(test_a), 3)
        self.assertEqual(
            [item.session_id for item in train_a],
            [item.session_id for item in train_b],
        )
        self.assertEqual(
            [item.session_id for item in test_a],
            [item.session_id for item in test_b],
        )
        self.assertFalse(
            {item.session_id for item in train_a}
            & {item.session_id for item in test_a}
        )


if __name__ == "__main__":
    unittest.main()
