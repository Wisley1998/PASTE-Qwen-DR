from __future__ import annotations

import unittest

from paste_repro.mapper import URLRankMapper
from paste_repro.tool_prediction import (
    TRACE_LEARNED_VISIT_POLICY_VERSION,
    TraceLearnedVisitPredictor,
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
                    {"url": "https://current.test/a", "rank": 3, "query_index": 1},
                    {"url": "https://current.test/b"},
                ],
            }
        )
        self.assertEqual((rows[0].result_rank, rows[0].query_index), (3, 1))
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


if __name__ == "__main__":
    unittest.main()
