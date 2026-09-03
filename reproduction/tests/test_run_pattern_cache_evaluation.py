from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from paste_repro.mapper import write_json_atomic
from paste_repro.traces import (
    LLMCall,
    OtherEvent,
    SearchResult,
    SessionTrace,
    ToolCall,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPOSITORY_ROOT
    / "reproduction"
    / "scripts"
    / "run_pattern_cache_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("run_pattern_cache_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _response(*rows: tuple[int, str, str], query: str = "topic") -> str:
    links = "\n".join(f"{rank}. [{title}]({url})" for rank, title, url in rows)
    return (
        "<tool_response>\n"
        f"A SearXNG search for '{query}' found {len(rows)} results:\n\n"
        f"## Web Results\n{links}\n</tool_response>"
    )


def _llm(line: int, tool_response: str, *, generated: str = "future") -> LLMCall:
    return LLMCall(
        call_index=line,
        timestamp_s=float(line),
        total_time_s=0.01,
        inference_time_s=0.01,
        messages=({"role": "user", "content": tool_response},),
        response=generated,
        line_number=line,
    )


def _commit(line: int, call_index: int, tool_name: str, raw: dict) -> OtherEvent:
    return OtherEvent(
        event_type="tool_result",
        timestamp_s=float(line),
        payload={
            "event_type": "tool_result",
            "call_index": call_index,
            "timestamp": float(line),
            "tool_name": tool_name,
            "commit_status": "committed",
            "result_sha256": runner.sha256_json(raw),
            "raw_result": raw,
            "formatted_response": "committed response",
            "transport": None,
        },
        line_number=line,
    )


def _decision(
    *,
    session: str = "session.jsonl",
    current: tuple[SearchResult, ...] = (),
    cached: tuple[object, ...] = (),
    query_count: int = 1,
    streak: int = 1,
    visited_cache_size: int = 0,
    prior_tool_updates: tuple[tuple[str, tuple[str, ...]], ...] = (),
    outcome: str = "visit",
    targets: tuple[str, ...] = (),
) -> object:
    return runner.SearchDecision(
        session_id=session,
        decision_id=f"{session}:decision",
        current_results=current,
        cache_candidates=cached,
        query_count=query_count,
        consecutive_search_streak=streak,
        visited_cache_size=visited_cache_size,
        prior_tool_updates=prior_tool_updates,
        outcome=outcome,
        authoritative_urls=targets,
    )


def _acceptance_inputs(
    *,
    targets: int = 80,
    sessions: int = 20,
    m0_hits: tuple[int, int, int] = (20, 35, 45),
    ungated_hits: tuple[int, int, int] = (20, 36, 46),
    gated_hits: tuple[int, int, int] = (20, 36, 46),
    fired_count: int = 3,
    visit_recall: float = 0.95,
    ungated_dispatches: int = 300,
    gated_dispatches: int = 285,
    p99_ms: float = 1.0,
    max_ms: float = 2.0,
) -> tuple[dict, dict]:
    def model(hits: tuple[int, int, int]) -> dict:
        return {
            "conditional_visit_targets": targets,
            "exact_top_k": {
                str(top_k): {"hits": hit}
                for top_k, hit in zip(runner.TOP_KS, hits)
            },
        }

    evaluation = {
        "sessions": sessions,
        "models": {
            "M0_current_blind": model(m0_hits),
            "pattern_cache_ungated": model(ungated_hits),
            "pattern_cache_gated": model(gated_hits),
        },
        "gate": {
            "fired_count": fired_count,
            "recall": visit_recall,
            "ungated_url_dispatches": ungated_dispatches,
            "gated_url_dispatches": gated_dispatches,
            "dispatch_reduction_absolute": (
                ungated_dispatches - gated_dispatches
            ),
            "dispatch_reduction_fraction": (
                (ungated_dispatches - gated_dispatches) / ungated_dispatches
                if ungated_dispatches
                else None
            ),
        },
    }
    latency = {
        "models": {
            "pattern_cache": {"p99_ms": p99_ms, "max_ms": max_ms}
        }
    }
    return evaluation, latency


class PatternCacheEvaluationTests(unittest.TestCase):
    def test_extraction_is_causal_bounded_and_marks_only_prior_visits(self) -> None:
        first_a = "https://first.test/a"
        first_b = "https://first.test/b"
        second = "https://second.test/current"
        third = "https://third.test/current"
        future_only = "https://future-output.test/forbidden"
        events = (
            ToolCall(0, 1.0, "search", {"query": ["first"]}, 1),
            _llm(
                2,
                _response((1, "A", first_a), (2, "B", first_b), query="first"),
                generated=f"I will visit {future_only}",
            ),
            ToolCall(1, 3.0, "visit", {"url": [first_b]}, 3),
            _llm(4, "<tool_response>visited</tool_response>"),
            ToolCall(2, 5.0, "search", {"query": ["second"]}, 5),
            _llm(6, _response((2, "Second", second), query="second")),
            ToolCall(
                3,
                7.0,
                "search",
                {"query": [f"query-{index}" for index in range(10)]},
                7,
            ),
            _llm(8, _response((1, "Third", third), query="query-0")),
        )
        session = SessionTrace(Path("synthetic.jsonl"), events)
        decisions = runner.extract_search_decisions((session,))

        self.assertEqual(len(decisions), 3)
        self.assertEqual(decisions[0].outcome, "visit")
        self.assertNotIn(
            future_only,
            {item.url for decision in decisions for item in decision.cache_candidates},
        )
        third_snapshot = {item.url: item for item in decisions[2].cache_candidates}
        self.assertEqual(third_snapshot[first_b].age, 2)
        self.assertTrue(third_snapshot[first_b].visited)
        self.assertFalse(third_snapshot[first_a].visited)
        self.assertEqual(third_snapshot[second].age, 1)
        self.assertEqual(third_snapshot[third].age, 0)
        self.assertEqual(decisions[2].query_count, 10)
        self.assertEqual(decisions[2].consecutive_search_streak, 2)

        pattern = runner.fit_rank_pattern(decisions)
        self.assertEqual(pattern.rank_counts, {2: 1})
        predictor = runner.make_frozen_predictor(pattern)
        predictions, admitted, reason = runner.pattern_predictions(
            decisions[2], predictor
        )
        self.assertEqual(predictions, ())
        self.assertFalse(admitted)
        self.assertEqual(reason, "matched_abstain_pattern")

    def test_pattern_score_and_ranking_preserve_current_m0_top1(self) -> None:
        current_top = "https://current.test/top"
        current_second = "https://current.test/second"
        history = "https://history.test/one"
        visited_history = "https://history.test/visited"
        current = (
            SearchResult(current_top, 1, 0, 0),
            SearchResult(current_second, 2, 1, 0),
        )
        cached = (
            runner.CachedCandidate(history, 1, 0, 1, False, False, 1),
            runner.CachedCandidate(visited_history, 1, 1, 1, True, False, 2),
            runner.CachedCandidate(current_top, 1, 0, 0, False, True, 3),
            runner.CachedCandidate(current_second, 2, 1, 0, False, True, 4),
        )
        decision = _decision(current=current, cached=cached, targets=(history,))
        pattern = runner.RankPattern({1: 10, 2: 5}, 15)
        predictor = runner.make_frozen_predictor(pattern)

        expected = math.log(10.0 + 0.5) - runner.AGE_PENALTY
        self.assertAlmostEqual(runner.candidate_score(cached[0], predictor), expected)
        predictions, admitted, reason = runner.pattern_predictions(decision, predictor)
        self.assertTrue(admitted)
        self.assertEqual(reason, "no_rule_match_admit")
        self.assertEqual(predictions[0], current_top)
        self.assertLess(predictions.index(history), predictions.index(visited_history))
        self.assertEqual(len(set(predictions)), len(predictions))

    def test_top20_diagnostic_extends_runtime_top5_without_dispatch_change(self) -> None:
        current = tuple(
            SearchResult(
                f"https://depth.test/{index}",
                index % 5 + 1,
                index,
                index // 5,
            )
            for index in range(25)
        )
        cached = tuple(
            runner.CachedCandidate(
                result.url,
                result.result_rank,
                result.ordinal,
                0,
                False,
                True,
                index,
            )
            for index, result in enumerate(current)
        )
        predictor = runner.make_frozen_predictor(
            runner.RankPattern({1: 5, 2: 4, 3: 3, 4: 2, 5: 1}, 15)
        )
        unlabeled = _decision(current=current, cached=cached, outcome="search")
        expanded = runner.ranked_pattern_predictions(
            unlabeled, predictor, top_k=20
        )
        self.assertEqual(len(expanded), 20)
        target = expanded[15]
        decision = _decision(
            current=current,
            cached=cached,
            targets=(target,),
        )

        rows, durations = runner.score_decisions((decision,), predictor)
        row = rows[0]
        self.assertEqual(len(row["pattern_gated_predictions"]), 5)
        self.assertEqual(len(row["pattern_gated_diagnostic_predictions"]), 20)
        self.assertEqual(
            row["pattern_gated_diagnostic_predictions"][:5],
            row["pattern_gated_predictions"],
        )
        metrics = runner.summarize_rows(rows, durations=durations)
        diagnostic = metrics["ranking_depth_diagnostic"]
        self.assertTrue(diagnostic["evaluation_only"])
        self.assertEqual(diagnostic["runtime_dispatch_top_k"], 5)
        self.assertEqual(
            metrics["models"]["pattern_cache_gated"]["exact_top_k"]["5"][
                "hits"
            ],
            0,
        )
        self.assertEqual(
            diagnostic["models"]["pattern_cache_gated"]["exact_top_k"]["20"][
                "hits"
            ],
            1,
        )

        gated_rows, _ = runner.score_decisions(
            (
                _decision(
                    session="abstain",
                    current=current,
                    cached=cached,
                    outcome="search",
                ),
                _decision(
                    session="abstain",
                    current=current,
                    cached=cached,
                    query_count=10,
                    streak=2,
                    outcome="search",
                ),
            ),
            predictor,
        )
        gated_row = gated_rows[1]
        self.assertEqual(gated_row["pattern_gated_predictions"], [])
        self.assertEqual(
            gated_row["pattern_gated_diagnostic_predictions"], []
        )
        self.assertEqual(
            len(gated_row["pattern_ungated_diagnostic_predictions"]), 20
        )

    def test_metrics_include_nonvisit_windows_gate_confusion_and_waste(self) -> None:
        url = "https://target.test/"
        current = (SearchResult(url, 1, 0, 0),)
        cached = (runner.CachedCandidate(url, 1, 0, 0, False, True, 1),)
        decisions = (
            _decision(session="tp", current=current, cached=cached, targets=(url,)),
            _decision(session="gate", current=current, cached=cached, outcome="search"),
            _decision(
                session="gate",
                current=current,
                cached=cached,
                query_count=10,
                streak=2,
                outcome="search",
            ),
            _decision(session="fn", outcome="visit", targets=(url,)),
        )
        rows, durations = runner.score_decisions(
            decisions,
            runner.make_frozen_predictor(runner.RankPattern({1: 3}, 3)),
        )
        metrics = runner.summarize_rows(rows, durations=durations)
        self.assertEqual(
            {key: metrics["gate"][key] for key in ("tp", "fp", "tn", "fn")},
            {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        )
        self.assertEqual(metrics["outcomes"], {"search": 2, "visit": 2})
        self.assertEqual(
            metrics["models"]["pattern_cache"]["all_window_top5"]["predictions"],
            2,
        )
        self.assertEqual(
            metrics["models"]["pattern_cache"]["all_window_top5"]["waste"],
            1,
        )
        self.assertEqual(
            metrics["candidate_ceilings"]["current_response"]["covered_targets"],
            1,
        )
        self.assertEqual(rows[2]["pattern_ungated_predictions"], [url])
        self.assertEqual(rows[2]["pattern_gated_predictions"], [])
        self.assertEqual(metrics["gate"]["fired_count"], 1)
        self.assertEqual(
            metrics["gate"]["positive_label"],
            "next_tool_is_committed_visit",
        )
        self.assertEqual(metrics["gate"]["ungated_url_dispatches"], 3)
        self.assertEqual(metrics["gate"]["gated_url_dispatches"], 2)
        self.assertEqual(metrics["gate"]["dispatch_reduction_absolute"], 1)
        self.assertAlmostEqual(
            metrics["gate"]["dispatch_reduction_fraction"], 1 / 3
        )
        for top_k in runner.TOP_KS:
            comparison = metrics["gate"]["gated_vs_ungated_exact_top_k"][
                str(top_k)
            ]
            self.assertEqual(comparison["gated_hits"], comparison["ungated_hits"])

    def test_acceptance_passes_all_frozen_ranker_and_gate_rules(self) -> None:
        evaluation, latency = _acceptance_inputs()
        acceptance = runner.new_holdout_acceptance(
            evaluation, latency, total_manifest_sessions=30
        )

        self.assertEqual(acceptance["status"], "accepted")
        self.assertTrue(acceptance["accepted"])
        self.assertTrue(acceptance["ranker"]["passed"])
        self.assertTrue(acceptance["visit_abstain_gate"]["passed"])
        self.assertTrue(acceptance["runtime"]["passed"])

    def test_acceptance_rejects_ranker_regression_or_missing_strict_gain(self) -> None:
        regressed, latency = _acceptance_inputs(gated_hits=(19, 36, 46))
        regression = runner.new_holdout_acceptance(
            regressed, latency, total_manifest_sessions=30
        )
        self.assertEqual(regression["status"], "rejected")
        self.assertFalse(regression["ranker"]["passed"])
        self.assertFalse(
            regression["ranker"]["conditions"]["gated_top1_at_least_m0"]
        )

        tied, latency = _acceptance_inputs(
            ungated_hits=(20, 35, 45), gated_hits=(20, 35, 45)
        )
        no_strict_gain = runner.new_holdout_acceptance(
            tied, latency, total_manifest_sessions=30
        )
        self.assertEqual(no_strict_gain["status"], "rejected")
        self.assertFalse(no_strict_gain["ranker"]["passed"])
        self.assertFalse(
            no_strict_gain["ranker"]["conditions"][
                "gated_top3_or_top5_strictly_above_m0"
            ]
        )

    def test_acceptance_is_inconclusive_when_gate_never_fires(self) -> None:
        evaluation, latency = _acceptance_inputs(
            fired_count=0,
            ungated_dispatches=300,
            gated_dispatches=300,
        )
        acceptance = runner.new_holdout_acceptance(
            evaluation, latency, total_manifest_sessions=30
        )

        self.assertEqual(acceptance["status"], "inconclusive")
        self.assertIsNone(acceptance["accepted"])
        self.assertIsNone(acceptance["visit_abstain_gate"]["passed"])
        self.assertIn(
            "visit_abstain_rule_never_fired",
            acceptance["inconclusive_reasons"],
        )

    def test_acceptance_is_inconclusive_when_data_are_insufficient(self) -> None:
        evaluation, latency = _acceptance_inputs(targets=79, sessions=19)
        acceptance = runner.new_holdout_acceptance(
            evaluation, latency, total_manifest_sessions=30
        )

        self.assertEqual(acceptance["status"], "inconclusive")
        self.assertIsNone(acceptance["accepted"])
        self.assertFalse(acceptance["data_adequacy"]["passed"])
        self.assertIn(
            "data_adequacy_threshold_not_met",
            acceptance["inconclusive_reasons"],
        )

    def test_bootstrap_replaces_zero_target_cluster_draws(self) -> None:
        url = "https://target.test/page"
        current = (SearchResult(url, 1, 0, 0),)
        cached = (runner.CachedCandidate(url, 1, 0, 0, False, True, 1),)
        rows, _ = runner.score_decisions(
            (
                _decision(
                    session="target-session",
                    current=current,
                    cached=cached,
                    targets=(url,),
                ),
            ),
            runner.make_frozen_predictor(runner.RankPattern({1: 1}, 1)),
        )
        inference = runner.paired_session_bootstrap(
            rows,
            ("target-session", "zero-target-session"),
            replicates=100,
        )

        self.assertEqual(
            inference["estimand"],
            "pattern_cache_gated_minus_M0_exact_target_recall",
        )
        self.assertTrue(inference["conditional_recall_defined"])
        self.assertEqual(inference["valid_bootstrap_replicates"], 100)
        self.assertGreater(inference["zero_target_resamples_discarded"], 0)
        self.assertIsNotNone(
            inference["top_k"]["1"][
                "paired_session_bootstrap_95_percentile_interval"
            ][0]
        )

    def test_bootstrap_marks_all_zero_target_holdout_undefined(self) -> None:
        inference = runner.paired_session_bootstrap(
            (), ("failed-session",), replicates=10
        )

        self.assertFalse(inference["conditional_recall_defined"])
        self.assertEqual(inference["valid_bootstrap_replicates"], 0)
        self.assertIsNone(inference["top_k"]["1"]["delta_target_recall"])
        self.assertEqual(
            inference["top_k"]["1"][
                "paired_session_bootstrap_95_percentile_interval"
            ],
            [None, None],
        )

    def test_non_http_visit_labels_remain_misses_but_never_enter_visited_lru(self) -> None:
        visible = "https://visible.test/page"
        invalid = "view-source:https://visible.test/page"
        later = "https://later.test/page"
        session = SessionTrace(
            Path("invalid-visit.jsonl"),
            (
                ToolCall(0, 1.0, "search", {"query": ["first"]}, 1),
                _llm(2, _response((1, "Visible", visible), query="first")),
                ToolCall(1, 3.0, "visit", {"url": [invalid]}, 3),
                _llm(4, "<tool_response>invalid visit</tool_response>"),
                ToolCall(2, 5.0, "search", {"query": ["later"]}, 5),
                _llm(6, _response((1, "Later", later), query="later")),
            ),
        )
        decisions = runner.extract_search_decisions((session,))
        inventory = runner.visit_executability_inventory((session,))
        self.assertEqual(decisions[0].authoritative_urls, (invalid,))
        self.assertEqual(decisions[1].visited_cache_size, 0)
        self.assertEqual(decisions[1].consecutive_search_streak, 1)
        self.assertEqual(inventory["runtime_nonexecutable_url_labels"], 1)
        self.assertEqual(inventory["visit_calls_with_nonexecutable_urls"], 1)

        rows, _ = runner.score_decisions(
            decisions,
            runner.make_frozen_predictor(runner.RankPattern({1: 2}, 2)),
        )
        metrics = runner.summarize_rows(rows)
        self.assertEqual(metrics["nonexecutable_authoritative_targets"], 1)
        self.assertEqual(
            metrics["models"]["pattern_cache"]["exact_top_k"]["5"]["hits"],
            0,
        )

    def test_new_extraction_uses_only_committed_search_and_visit_results(self) -> None:
        selected = "https://committed.test/selected"
        later = "https://committed.test/later"
        search_one = {
            "tool": "search",
            "query": ["first"],
            "results": [
                {
                    "url": selected,
                    "rank": 1,
                    "query_index": 0,
                    "query": "first",
                    "title": "Selected",
                }
            ],
        }
        visit_result = {
            "tool": "visit",
            "goal": "evidence",
            "pages": [{"url": selected, "content": "ok"}],
        }
        search_two = {
            "tool": "search",
            "query": ["second"],
            "results": [
                {
                    "url": later,
                    "rank": 1,
                    "query_index": 0,
                    "query": "second",
                    "title": "Later",
                }
            ],
        }
        session = SessionTrace(
            Path("committed.jsonl"),
            (
                ToolCall(0, 1.0, "search", {"query": ["first"]}, 1),
                _commit(2, 0, "search", search_one),
                _llm(3, "<tool_response>committed search</tool_response>"),
                ToolCall(
                    1,
                    4.0,
                    "visit",
                    {"url": [selected], "goal": "evidence"},
                    4,
                ),
                _commit(5, 1, "visit", visit_result),
                _llm(6, "<tool_response>committed visit</tool_response>"),
                ToolCall(2, 7.0, "search", {"query": ["second"]}, 7),
                _commit(8, 2, "search", search_two),
                _llm(9, "<tool_response>committed search</tool_response>"),
                # Requested but failed: there is deliberately no tool_result.
                ToolCall(
                    3,
                    10.0,
                    "visit",
                    {"url": [later], "goal": "uncommitted"},
                    10,
                ),
                OtherEvent("collector_error", 11.0, {"error_type": "failure"}, 11),
            ),
        )
        decisions, audit = runner.extract_committed_search_decisions((session,))
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0].outcome, "visit")
        self.assertEqual(decisions[0].authoritative_urls, (selected,))
        self.assertEqual(decisions[1].outcome, "uncommitted_visit")
        self.assertEqual(decisions[1].authoritative_urls, ())
        historical = next(
            item for item in decisions[1].cache_candidates if item.url == selected
        )
        self.assertTrue(historical.visited)
        self.assertEqual(audit["requested_tool_calls"], {"search": 2, "visit": 2})
        self.assertEqual(audit["committed_tool_results"], {"search": 2, "visit": 1})
        self.assertEqual(audit["uncommitted_tool_calls"], {"visit": 1})

        rows, _ = runner.score_decisions(
            decisions,
            runner.make_frozen_predictor(runner.RankPattern({1: 3}, 3)),
        )
        self.assertEqual(sum(row["target_count"] for row in rows), 1)

    def test_artifact_is_plain_checksummed_json_with_explicit_smoothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "train.jsonl"
            trace_path.write_text("{}\n", encoding="utf-8")
            session = SessionTrace(trace_path, ())
            artifact = runner.build_artifact(
                runner.RankPattern({1: 2, 3: 1}, 3), (session,)
            )
            path = root / "artifact.json"
            write_json_atomic(path, artifact)
            restored, loaded = runner.load_artifact(path)

            self.assertEqual(restored.rank_counts, {1: 2, 3: 1})
            self.assertEqual(loaded["config"]["smoothing"], 0.5)
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["rank_counts"]["1"] = 99
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                runner.validate_artifact(tampered)

    def test_new_holdout_is_claimed_once_and_bootstraps_whole_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "training.jsonl"
            training_path.write_text("{}\n", encoding="utf-8")
            artifact = runner.build_artifact(
                runner.RankPattern({1: 3}, 3),
                (SessionTrace(training_path, ()),),
            )
            artifact_path = root / "artifact.json"
            write_json_atomic(artifact_path, artifact)

            workload = runner.load_expected_new_workload(runner.DEFAULT_NEW_WORKLOAD)
            holdout = root / "collection"
            holdout.mkdir()
            records = []
            for index, source in enumerate(workload.sources, 1):
                session_id = f"{index:04d}-{source.source_id}"
                trace_name = f"{session_id}.jsonl"
                trace_path = holdout / trace_name
                trace_events = [
                    {
                        "event_type": "session_start",
                        "call_index": 0,
                        "timestamp": 0.0,
                        "session_id": session_id,
                        "workload_id": workload.workload_id,
                        "source_id": source.source_id,
                        "source_sha256": source.source_sha256,
                        "question_sha256": source.question_sha256,
                        "provenance": source.provenance,
                    },
                    {
                        "event_type": "session_end",
                        "call_index": 0,
                        "timestamp": 1.0,
                        "status": "failed",
                        "llm_calls": 0,
                        "tool_calls": 0,
                        "committed_tool_results": 0,
                    },
                ]
                trace_path.write_text(
                    "".join(json.dumps(event) + "\n" for event in trace_events),
                    encoding="utf-8",
                )
                records.append(
                    {
                        "session_id": session_id,
                        "source_id": source.source_id,
                        "source_sha256": source.source_sha256,
                        "question_sha256": source.question_sha256,
                        "provenance": source.provenance,
                        "trace_file": trace_name,
                        "trace_sha256": runner.sha256_file(trace_path),
                        "status": "failed",
                        "llm_calls": 0,
                        "tool_calls": 0,
                        "committed_tool_results": 0,
                        "event_count": 2,
                    }
                )
            manifest = {
                "schema_version": runner.MANIFEST_SCHEMA_VERSION,
                "artifact_type": runner.MANIFEST_TYPE,
                "trace_schema": runner.TRACE_SCHEMA,
                "collection_status": "complete_with_failures",
                "completed_at_utc": "2026-08-31T00:00:00Z",
                "workload": {
                    "schema_version": 1,
                    "workload_id": workload.workload_id,
                    "file_name": workload.file_name,
                    "file_sha256": workload.file_sha256,
                    "source_count": len(workload.sources),
                    "ordered_source_ids": [
                        source.source_id for source in workload.sources
                    ],
                },
                "sessions": records,
                "summary": {
                    "session_count": 30,
                    "succeeded": 0,
                    "failed": 30,
                },
            }
            write_json_atomic(holdout / "manifest.json", manifest)
            extra = holdout / "extra.jsonl"
            extra.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace set mismatch"):
                runner.validate_collection_manifest(holdout, workload)
            extra.unlink()
            bad_manifest = json.loads(json.dumps(manifest))
            bad_manifest["sessions"][0]["trace_sha256"] = "0" * 64
            write_json_atomic(holdout / "manifest.json", bad_manifest)
            with self.assertRaisesRegex(ValueError, "trace_sha256 mismatch"):
                runner.validate_collection_manifest(holdout, workload)
            write_json_atomic(holdout / "manifest.json", manifest)
            output = root / "evaluation"
            payload = runner.run_new_evaluation(
                holdout,
                artifact_path,
                output,
                workload_path=runner.DEFAULT_NEW_WORKLOAD,
                bootstrap_replicates=50,
                benchmark_passes=1,
            )
            self.assertTrue((output / runner.STARTED_NAME).is_file())
            self.assertTrue((output / runner.COMPLETE_NAME).is_file())
            self.assertEqual(
                payload["paired_session_bootstrap"]["session_count"], 30
            )
            self.assertEqual(
                payload["paired_session_bootstrap"]["bootstrap_replicates"], 50
            )
            with self.assertRaises(FileExistsError):
                runner.run_new_evaluation(
                    holdout,
                    artifact_path,
                    output,
                    workload_path=runner.DEFAULT_NEW_WORKLOAD,
                    bootstrap_replicates=1,
                    benchmark_passes=1,
                )

            # A malformed manifest is discovered only after the atomic claim;
            # the retained marker prevents a repaired/replayed second look.
            write_json_atomic(holdout / "manifest.json", {})
            failed_output = root / "failed-evaluation"
            with self.assertRaisesRegex(ValueError, "artifact_type"):
                runner.run_new_evaluation(
                    holdout,
                    artifact_path,
                    failed_output,
                    workload_path=runner.DEFAULT_NEW_WORKLOAD,
                    bootstrap_replicates=1,
                    benchmark_passes=1,
                )
            self.assertTrue((failed_output / runner.STARTED_NAME).is_file())
            write_json_atomic(holdout / "manifest.json", manifest)
            with self.assertRaises(FileExistsError):
                runner.run_new_evaluation(
                    holdout,
                    artifact_path,
                    failed_output,
                    workload_path=runner.DEFAULT_NEW_WORKLOAD,
                    bootstrap_replicates=1,
                    benchmark_passes=1,
                )


if __name__ == "__main__":
    unittest.main()
