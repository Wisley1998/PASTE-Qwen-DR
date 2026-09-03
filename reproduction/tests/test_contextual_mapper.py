from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from paste_repro.contextual_mapper import (
    FEATURE_SCHEMA,
    ContextualURLReranker,
    contextual_candidates,
    load_contextual_artifact,
    save_contextual_artifact,
)
from paste_repro.tool_prediction import (
    ContextualTraceVisitPredictor,
    load_visit_predictor,
)
from paste_repro.traces import LLMCall, SearchResult, SearchVisitTransition, ToolCall


def _transition(
    session: str,
    results: tuple[SearchResult, ...],
    target: str,
    *,
    generated_response: str = "future response",
) -> SearchVisitTransition:
    return SearchVisitTransition(
        session_id=session,
        search=ToolCall(0, 1.0, "search", {"query": ["alpha topic"]}, 1),
        decision_llm=LLMCall(
            1, 2.0, 1.0, 1.0, (), generated_response, 2
        ),
        visit=ToolCall(1, 2.0, "visit", {"url": [target]}, 3),
        completion_llm=None,
        search_results=results,
        authoritative_urls=(target,),
        baseline_stall_s=1.0,
        overlap_window_s=1.0,
    )


def _semantic_training() -> tuple[SearchVisitTransition, ...]:
    rows: list[SearchVisitTransition] = []
    for index in range(12):
        exact = f"https://exact-{index}.test/alpha-topic"
        noise = f"https://noise-{index}.test/unrelated"
        if index % 2 == 0:
            results = (
                SearchResult(exact, 1, 0, 0, "Alpha topic result", "alpha topic"),
                SearchResult(noise, 2, 1, 0, "Completely unrelated", "alpha topic"),
            )
        else:
            results = (
                SearchResult(noise, 1, 0, 0, "Completely unrelated", "alpha topic"),
                SearchResult(exact, 2, 1, 0, "Alpha topic result", "alpha topic"),
            )
        rows.append(_transition(f"train-{index}", results, exact))
    return tuple(rows)


class ContextualMapperTests(unittest.TestCase):
    def test_feature_schema_and_raw_url_dedup_are_stable(self) -> None:
        repeated = "https://current.test/raw?a=1#fragment"
        candidates = contextual_candidates(
            (
                SearchResult(repeated, 2, 0, 0, "Alpha", "alpha"),
                SearchResult(repeated, 4, 1, 1, "Alpha topic", "alpha topic"),
                SearchResult("https://other.test/", 1, 2, 0, "Other", "alpha"),
            )
        )
        self.assertEqual(
            [item.url for item in candidates],
            [repeated, "https://other.test/"],
        )
        self.assertEqual(len(candidates[0].features), len(FEATURE_SCHEMA))

    def test_pairwise_model_learns_lexical_relevance_without_rank_leakage(self) -> None:
        reranker = ContextualURLReranker().fit(_semantic_training())
        current = (
            SearchResult(
                "https://current.test/noise",
                1,
                0,
                0,
                "Unrelated page",
                "alpha topic",
            ),
            SearchResult(
                "https://current.test/exact",
                2,
                1,
                0,
                "Alpha topic result",
                "alpha topic",
            ),
        )
        self.assertTrue(reranker.optimizer_converged)
        self.assertEqual(
            reranker.predict(current, 1)[0].invocation.arguments["url"],
            "https://current.test/exact",
        )

    def test_future_llm_response_and_visit_are_not_prediction_inputs(self) -> None:
        reranker = ContextualURLReranker().fit(_semantic_training())
        visible = (
            SearchResult("https://a.test/", 1, 0, 0, "Alpha", "alpha"),
            SearchResult("https://b.test/", 2, 1, 0, "Beta", "alpha"),
        )
        first = _transition(
            "held",
            visible,
            "https://a.test/",
            generated_response="visit https://a.test/",
        )
        second = _transition(
            "held",
            visible,
            "https://b.test/",
            generated_response="visit https://b.test/",
        )
        self.assertNotEqual(first.decision_llm.response, second.decision_llm.response)
        self.assertNotEqual(first.authoritative_urls, second.authoritative_urls)
        self.assertEqual(
            reranker.predict(first.search_results, 2),
            reranker.predict(second.search_results, 2),
        )

    def test_artifact_round_trip_and_online_adapter(self) -> None:
        reranker = ContextualURLReranker().fit(_semantic_training())
        artifact = reranker.to_artifact(
            {
                "algorithm": "unit-test",
                "seed": "fixed",
                "train_ratio": 0.7,
                "train_sessions": [{"session_id": "train", "sha256": "abc"}],
                "held_out_sessions": [{"session_id": "held", "sha256": "def"}],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contextual.json"
            save_contextual_artifact(path, artifact)
            restored, loaded = load_contextual_artifact(path)
            predictor = ContextualTraceVisitPredictor.from_artifact(path, top_k=1)
            dispatched = load_visit_predictor(path, top_k=1)
        self.assertEqual(restored.weights, reranker.weights)
        self.assertEqual(loaded["artifact_sha256"], artifact["artifact_sha256"])
        self.assertIsInstance(dispatched, ContextualTraceVisitPredictor)
        result = {
            "tool": "search",
            "results": [
                {
                    "url": "https://current.test/noise",
                    "rank": 1,
                    "query_index": 0,
                    "query": "alpha topic",
                    "title": "Unrelated page",
                },
                {
                    "url": "https://current.test/exact",
                    "rank": 2,
                    "query_index": 0,
                    "query": "alpha topic",
                    "title": "Alpha topic result",
                },
            ],
        }
        self.assertEqual(
            predictor.predict_structured_result(result),
            ("https://current.test/exact",),
        )


if __name__ == "__main__":
    unittest.main()
