from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from paste_repro.analysis import evaluate_held_out
from paste_repro.mapper import URLRankMapper, load_artifact, save_artifact
from paste_repro.traces import LLMCall, SearchResult, SearchVisitTransition, ToolCall


def _transition(
    session: str,
    ranked_urls: list[tuple[int, str]],
    target_urls: list[str],
    *,
    stall_s: float = 4.0,
    overlap_s: float = 2.0,
) -> SearchVisitTransition:
    search = ToolCall(0, 1.0, "search", {"query": ["q"]}, 1)
    llm = LLMCall(1, 2.0, overlap_s, overlap_s, (), "", 2)
    visit = ToolCall(1, 2.0, "visit", {"url": target_urls}, 3)
    results = tuple(
        SearchResult(url, rank, ordinal, 0)
        for ordinal, (rank, url) in enumerate(ranked_urls)
    )
    return SearchVisitTransition(
        session_id=session,
        search=search,
        decision_llm=llm,
        visit=visit,
        completion_llm=None,
        search_results=results,
        authoritative_urls=tuple(target_urls),
        baseline_stall_s=stall_s,
        overlap_window_s=overlap_s,
    )


class MapperTests(unittest.TestCase):
    def test_rank_preference_is_learned_and_late_binds_current_urls(self) -> None:
        training = [
            _transition("a", [(1, "train-a"), (2, "train-b")], ["train-b"]),
            _transition("b", [(1, "train-c"), (2, "train-d")], ["train-d"]),
            _transition("c", [(1, "train-e"), (2, "train-f")], ["train-e"]),
        ]
        mapper = URLRankMapper().fit(training, searches_seen=5)
        current = (
            SearchResult("current-one", 1, 0, 0),
            SearchResult("current-two", 2, 1, 0),
        )
        prediction = mapper.predict(current, 1)[0]
        self.assertEqual(mapper.learned_rank_order, (2, 1))
        self.assertEqual(prediction.invocation.arguments, {"url": "current-two"})
        self.assertNotIn("train-", prediction.invocation.canonical_arguments)

    def test_held_out_hit_produces_positive_latency_gain(self) -> None:
        training = [_transition("train", [(1, "a"), (2, "b")], ["b"])]
        held_out = _transition(
            "test",
            [(1, "new-a"), (2, "new-b")],
            ["new-b"],
            stall_s=4.0,
            overlap_s=1.5,
        )
        mapper = URLRankMapper().fit(training)
        report = evaluate_held_out(mapper, [held_out], top_ks=(1,), latency_top_k=1)
        self.assertEqual(
            report["top_k_concrete_invocation_hit"]["1"]["invocation_hits"], 1
        )
        self.assertAlmostEqual(report["baseline_exposed_tool_stall_s"], 4.0)
        self.assertAlmostEqual(report["saved_time_s"], 1.5)
        self.assertAlmostEqual(report["optimized_exposed_tool_stall_s"], 2.5)
        self.assertGreater(report["stall_reduction"], 0.0)

    def test_versioned_artifact_round_trip_and_checksum(self) -> None:
        mapper = URLRankMapper().fit(
            [_transition("train", [(1, "a"), (2, "b")], ["b"])],
            searches_seen=2,
        )
        artifact = mapper.to_artifact(
            {
                "algorithm": "test",
                "seed": "fixed",
                "train_ratio": 0.7,
                "train_sessions": [{"session_id": "train", "sha256": "abc"}],
                "held_out_sessions": [{"session_id": "test", "sha256": "def"}],
            }
        )
        self.assertEqual(artifact["schema"], "paste_repro.url_rank_mapper")
        self.assertEqual(artifact["version"], 1)
        self.assertIn("manifest_sha256", artifact["training_split"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapper.json"
            save_artifact(path, artifact)
            restored, loaded = load_artifact(path)
        self.assertEqual(restored.rank_counts, mapper.rank_counts)
        self.assertEqual(loaded["artifact_sha256"], artifact["artifact_sha256"])

        tampered = dict(artifact)
        tampered["version"] = 2
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            URLRankMapper.from_artifact(tampered)


if __name__ == "__main__":
    unittest.main()
