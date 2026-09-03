from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.baseline_boundary import inference_speedup_counterfactual  # noqa: E402
from paste_repro.mapper import URLRankMapper  # noqa: E402
from paste_repro.traces import (  # noqa: E402
    LLMCall,
    SearchResult,
    SearchVisitTransition,
    ToolCall,
)


def _transition(
    session_id: str,
    *,
    authoritative_url: str,
    stall_s: float,
    overlap_s: float,
) -> SearchVisitTransition:
    visible = (
        SearchResult("https://example.test/first", 1, 0, 0),
        SearchResult("https://example.test/second", 2, 1, 0),
    )
    return SearchVisitTransition(
        session_id=session_id,
        search=ToolCall(0, 1.0, "search", {"query": ["q"]}, 1),
        decision_llm=LLMCall(1, 2.0, overlap_s, overlap_s, (), "", 2),
        visit=ToolCall(1, 2.0, "visit", {"url": [authoritative_url]}, 3),
        completion_llm=None,
        search_results=visible,
        authoritative_urls=(authoritative_url,),
        baseline_stall_s=stall_s,
        overlap_window_s=overlap_s,
    )


class BaselineBoundaryCounterfactualTests(unittest.TestCase):
    def test_scales_only_decision_time_and_bounds_exact_hit_overlap(self) -> None:
        hit = _transition(
            "hit",
            authoritative_url="https://example.test/second",
            stall_s=4.0,
            overlap_s=2.0,
        )
        miss = _transition(
            "miss",
            authoritative_url="https://outside.test/not-visible",
            stall_s=1.0,
            overlap_s=2.0,
        )
        mapper = URLRankMapper().fit([hit])

        report = inference_speedup_counterfactual(
            mapper, [hit, miss], top_k=1, speedups=(1.0, 2.0)
        )

        self.assertEqual(report["observed_decision_generation_s"], 4.0)
        self.assertEqual(report["observed_demand_only_external_tool_stall_s"], 5.0)
        self.assertEqual(report["rows"][0]["paste_hidden_external_tool_stall_s"], 2.0)
        self.assertEqual(report["rows"][0]["demand_only_segment_s"], 9.0)
        self.assertEqual(report["rows"][0]["paste_segment_s"], 7.0)
        self.assertEqual(report["rows"][1]["paste_hidden_external_tool_stall_s"], 1.0)
        self.assertEqual(report["rows"][1]["demand_only_segment_s"], 7.0)
        self.assertEqual(report["rows"][1]["paste_segment_s"], 6.0)

    def test_rejects_nonpositive_speedup(self) -> None:
        mapper = URLRankMapper()
        with self.assertRaises(ValueError):
            inference_speedup_counterfactual(
                mapper, (), top_k=1, speedups=(0.0,)
            )


if __name__ == "__main__":
    unittest.main()
