from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from paste_repro.mapper import URLRankMapper, save_artifact
from paste_repro.tool_prediction import (
    TRACE_LEARNED_VISIT_POLICY_VERSION,
    TraceLearnedVisitPredictor,
    load_visit_predictor,
    structured_search_results,
)
from paste_repro.traces import LLMCall, SearchResult, SearchVisitTransition, ToolCall


def _transition(target_rank: int) -> SearchVisitTransition:
    urls = ("https://train.test/one", "https://train.test/two")
    return SearchVisitTransition(
        session_id=f"train-{target_rank}",
        search=ToolCall(0, 1.0, "search", {"query": ["q"]}, 1),
        decision_llm=LLMCall(1, 2.0, 1.0, 1.0, (), "", 2),
        visit=ToolCall(1, 2.0, "visit", {"url": [urls[target_rank - 1]]}, 3),
        completion_llm=None,
        search_results=tuple(
            SearchResult(url, rank, rank - 1, 0)
            for rank, url in enumerate(urls, start=1)
        ),
        authoritative_urls=(urls[target_rank - 1],),
        baseline_stall_s=1.0,
        overlap_window_s=1.0,
    )


class TraceLearnedVisitPredictorTests(unittest.TestCase):
    def test_live_prediction_late_binds_learned_rank_to_current_url(self) -> None:
        mapper = URLRankMapper().fit([_transition(2), _transition(2), _transition(1)])
        predictor = TraceLearnedVisitPredictor(mapper=mapper, top_k=1)
        live_result = {
            "tool": "search",
            "results": [
                {"url": "https://current.test/a", "rank": 1, "query_index": 0},
                {"url": "https://current.test/b", "rank": 2, "query_index": 0},
            ],
        }

        self.assertEqual(
            predictor.predict_structured_result(live_result),
            ("https://current.test/b",),
        )
        self.assertEqual(predictor.policy, TRACE_LEARNED_VISIT_POLICY_VERSION)
        self.assertNotIn("train.test", repr(predictor.predict_structured_result(live_result)))

    def test_structured_adapter_preserves_rank_and_has_legacy_fallback(self) -> None:
        rows = structured_search_results(
            {
                "tool": "search",
                "results": [
                    {
                        "url": "https://current.test/a",
                        "rank": 3,
                        "query_index": 1,
                        "query": "alpha",
                        "title": "Alpha result",
                        "snippet": "A summary",
                    },
                    {"url": "https://current.test/b"},
                ],
            }
        )
        self.assertEqual((rows[0].result_rank, rows[0].query_index), (3, 1))
        self.assertEqual(
            (rows[0].query, rows[0].title, rows[0].snippet),
            ("alpha", "Alpha result", "A summary"),
        )
        self.assertEqual((rows[1].result_rank, rows[1].query_index), (1, 0))

    def test_visible_text_prediction_uses_only_supplied_response(self) -> None:
        mapper = URLRankMapper().fit([_transition(2)])
        predictor = TraceLearnedVisitPredictor(mapper=mapper, top_k=1)
        visible = "\n".join(
            [
                "1. [one](https://current.test/one)",
                "2. [two](https://current.test/two)",
            ]
        )
        self.assertEqual(
            predictor.predict_visible_response(visible),
            ("https://current.test/two",),
        )

    def test_schema_dispatch_loads_legacy_and_rejects_unknown(self) -> None:
        mapper = URLRankMapper().fit([_transition(2)])
        artifact = mapper.to_artifact(
            {
                "algorithm": "unit-test",
                "seed": "fixed",
                "train_ratio": 0.7,
                "train_sessions": [{"session_id": "train", "sha256": "abc"}],
                "held_out_sessions": [{"session_id": "held", "sha256": "def"}],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy.json"
            unknown_path = Path(directory) / "unknown.json"
            save_artifact(legacy_path, artifact)
            unknown_path.write_text('{"schema":"unknown"}\n', encoding="utf-8")
            loaded = load_visit_predictor(legacy_path, top_k=1)
            with self.assertRaisesRegex(ValueError, "unsupported visit predictor"):
                load_visit_predictor(unknown_path, top_k=1)
        self.assertIsInstance(loaded, TraceLearnedVisitPredictor)


if __name__ == "__main__":
    unittest.main()
