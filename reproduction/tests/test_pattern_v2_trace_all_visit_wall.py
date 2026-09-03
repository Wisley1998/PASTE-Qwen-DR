from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_trace_all_visit_wall as all_visit  # noqa: E402
import scale_trace_llm_timing as llm_timing  # noqa: E402

from paste_repro.traces import LLMCall, SessionTrace, ToolCall  # noqa: E402


def llm(index: int, timestamp: float, response: str) -> LLMCall:
    return LLMCall(
        call_index=index,
        timestamp_s=timestamp,
        total_time_s=1.0,
        inference_time_s=0.8,
        messages=(
            {
                "role": "user",
                "content": f"<tool_response>\n{response}\n</tool_response>",
            },
        ),
        response="",
        line_number=index + 1,
    )


class AllVisitWallTests(unittest.TestCase):
    def test_session_url_cache_reuses_ready_and_inflight_future_results(self) -> None:
        first = "https://example.test/a"
        ready = "https://example.test/b"
        inflight = "https://example.test/c"
        session_id = "global-cache.jsonl"
        session = SessionTrace(
            Path(session_id),
            (
                ToolCall(0, 0.0, "search", {"query": ["topic"]}, 1),
                llm(1, 5.0, "search results"),
                ToolCall(
                    1,
                    5.0,
                    "visit",
                    {"url": [first]},
                    3,
                    {"unit_duration_s": [2.0]},
                ),
                llm(2, 8.0, "visited first"),
                ToolCall(
                    2,
                    8.0,
                    "visit",
                    {"url": [ready, inflight]},
                    5,
                    {"unit_duration_s": [3.0, 6.0]},
                ),
                llm(3, 18.0, "visited future URLs"),
            ),
        )
        first_decision = all_visit.AllVisitDecision(
            session_id=session_id,
            decision_id="d1",
            trigger_tool="search",
            visit_depth=0,
            query_count=1,
            search_streak=1,
            search_sequence=1,
            task_text="topic",
            trigger_urls=(),
            candidates=(),
            outcome="visit",
            authoritative_urls=(first,),
            trigger_event_index=0,
            target_tool_event_index=2,
            lead_llm_event_indices=(1,),
        )
        second_decision = all_visit.AllVisitDecision(
            session_id=session_id,
            decision_id="d2",
            trigger_tool="visit",
            visit_depth=1,
            query_count=1,
            search_streak=0,
            search_sequence=1,
            task_text="topic",
            trigger_urls=(first,),
            candidates=(),
            outcome="visit",
            authoritative_urls=(ready, inflight),
            trigger_event_index=2,
            target_tool_event_index=4,
            lead_llm_event_indices=(3,),
        )

        def candidate(url: str, position: int) -> all_visit.ScoredCandidate:
            return all_visit.ScoredCandidate(
                pattern=SimpleNamespace(
                    url=url,
                    position=position,
                    session_id=session_id,
                    decision_id="d1",
                ),
                exact_probability=0.9,
                visit_probability=0.9,
                rank_only_probability=0.9,
                exact_match=False,
            )

        windows = (
            all_visit.ScoredWindow(
                decision_id="d1",
                session_id=session_id,
                v2_gate=True,
                next_tool_visit=True,
                expected_authoritative_calls=1.0,
                coarse_expected_authoritative_calls=1.0,
                targets=(first,),
                executable_targets=(first,),
                candidates=(candidate(ready, 1), candidate(inflight, 2)),
            ),
            all_visit.ScoredWindow(
                decision_id="d2",
                session_id=session_id,
                v2_gate=True,
                next_tool_visit=True,
                expected_authoritative_calls=2.0,
                coarse_expected_authoritative_calls=2.0,
                targets=(ready, inflight),
                executable_targets=(ready, inflight),
                candidates=(),
            ),
        )
        timings = {
            "d1": all_visit.DecisionTiming(
                "d1", session_id, 1.0, 2.0, 1, "corrected", (2.0,)
            ),
            "d2": all_visit.DecisionTiming(
                "d2", session_id, 1.0, 9.0, 2, "corrected", (3.0, 6.0)
            ),
        }
        estimates = {
            decision_id: SimpleNamespace(overlap_for_url=lambda _url: 1.0)
            for decision_id in ("d1", "d2")
        }

        with mock.patch.object(
            all_visit, "load_sessions", return_value=(session,)
        ):
            replays, audit = all_visit.build_session_global_cache_replays(
                Path("unused"),
                windows,
                (first_decision, second_decision),
                timings,
                estimates,
                {session_id: 18.0},
                per_task_width=2,
                coordination_cost_s=0.001,
            )

        replay = replays[0]
        self.assertEqual(replay.selected_speculations, 2)
        self.assertEqual(replay.exact_url_hits, 2)
        self.assertEqual(replay.ready_cache_hits, 1)
        self.assertEqual(replay.inflight_cache_hits, 1)
        self.assertEqual(replay.earlier_decision_cache_hits, 2)
        self.assertEqual(replay.incremental_future_cache_hits, 2)
        self.assertAlmostEqual(replay.inflight_wait_s, 2.0)
        self.assertAlmostEqual(replay.gross_saved_visit_stall_s, 7.0)
        self.assertAlmostEqual(replay.net_saved_visit_stall_s, 6.998)
        self.assertAlmostEqual(replay.treatment_full_wall_s, 11.002)
        self.assertEqual(audit["ttl"], "infinite")

    def test_llm_scaling_materializes_0_42_and_aligns_timestamps(self) -> None:
        events = [
            {
                "event_type": "tool_call",
                "call_index": 0,
                "timestamp": 2.0,
                "tool_name": "search",
                "tool_args": {"query": ["topic"]},
            },
            {
                "event_type": "llm_call",
                "call_index": 1,
                "timestamp": 7.0,
                "total_time_ms": 3000.0,
                "inference_time_ms": 2700.0,
                "rtt_ms": 300.0,
                "messages": [],
                "response": "",
            },
            {
                "event_type": "tool_call",
                "call_index": 1,
                "timestamp": 7.0,
                "tool_name": "visit",
                "tool_args": {"url": ["https://example.test/a"]},
                "timing_correction": {
                    "execution": "serial_sum_per_url",
                    "duration_s": 2.0,
                    "unit_duration_s": [2.0],
                },
            },
            {
                "event_type": "llm_call",
                "call_index": 2,
                "timestamp": 11.0,
                "total_time_ms": 2000.0,
                "inference_time_ms": 1800.0,
                "rtt_ms": 200.0,
                "messages": [],
                "response": "",
            },
        ]

        rewritten, audit = llm_timing.scale_events(events, duration_scale=0.42)

        self.assertAlmostEqual(rewritten[1]["total_time_ms"], 1260.0)
        self.assertAlmostEqual(rewritten[1]["inference_time_ms"], 1134.0)
        self.assertAlmostEqual(rewritten[1]["rtt_ms"], 126.0)
        self.assertAlmostEqual(rewritten[1]["timestamp"], 5.26)
        self.assertAlmostEqual(rewritten[2]["timestamp"], 5.26)
        self.assertAlmostEqual(rewritten[3]["timestamp"], 8.10)
        self.assertAlmostEqual(
            rewritten[3]["timestamp"]
            - rewritten[3]["total_time_ms"] / 1000.0
            - rewritten[2]["timestamp"],
            2.0,
        )
        self.assertEqual(
            rewritten[2]["timing_correction"]["execution"],
            "serial_sum_per_url",
        )
        self.assertAlmostEqual(audit["removed_total_s"], 2.9)

    def test_cross_fold_budget_is_label_independent_and_burst_bounded(self) -> None:
        windows = []
        estimates = {}
        for window_index in range(10):
            decision_id = f"decision-{window_index}"
            session_id = f"session-{window_index}.jsonl"
            candidates = tuple(
                all_visit.ScoredCandidate(
                    pattern=SimpleNamespace(
                        url=f"https://example.test/{window_index}/{index}",
                        position=index + 1,
                        session_id=session_id,
                        decision_id=decision_id,
                    ),
                    exact_probability=0.9 - index * 0.05,
                    visit_probability=0.5,
                    rank_only_probability=0.1,
                    exact_match=False,
                )
                for index in range(6)
            )
            windows.append(
                all_visit.ScoredWindow(
                    decision_id=decision_id,
                    session_id=session_id,
                    v2_gate=True,
                    next_tool_visit=False,
                    expected_authoritative_calls=0.5,
                    coarse_expected_authoritative_calls=0.5,
                    targets=(),
                    executable_targets=(),
                    candidates=candidates,
                )
            )
            estimates[decision_id] = SimpleNamespace(
                overlap_for_url=lambda _url: 1.0
            )

        adjusted, audit = all_visit.apply_cross_fold_start_budget(
            windows,
            estimates,
            average_width=2,
            burst_multiplier=2,
            coordination_cost_s=0.001,
        )

        self.assertEqual(audit["max_starts_per_decision"], 4)
        self.assertEqual(
            sum(audit["starts_per_decision_histogram"].values()), 10
        )
        self.assertLessEqual(audit["starts_per_decision_p95"], 4)
        selected = [
            candidate.pattern.url
            for window in adjusted
            for candidate in window.candidates
            if candidate.exact_probability > 0.0
        ]
        self.assertTrue(selected)
        self.assertTrue(
            all(
                sum(candidate.exact_probability > 0.0 for candidate in window.candidates)
                <= 4
                for window in adjusted
            )
        )
        # Selection consumes scores and folds only; changing future labels
        # cannot alter which candidates receive starts.
        relabeled = [
            all_visit.ScoredWindow(
                decision_id=window.decision_id,
                session_id=window.session_id,
                v2_gate=window.v2_gate,
                next_tool_visit=True,
                expected_authoritative_calls=window.expected_authoritative_calls,
                coarse_expected_authoritative_calls=(
                    window.coarse_expected_authoritative_calls
                ),
                targets=("https://future-label.invalid",),
                executable_targets=("https://future-label.invalid",),
                candidates=tuple(
                    all_visit.ScoredCandidate(
                        pattern=candidate.pattern,
                        exact_probability=candidate.exact_probability,
                        visit_probability=candidate.visit_probability,
                        rank_only_probability=candidate.rank_only_probability,
                        exact_match=True,
                    )
                    for candidate in window.candidates
                ),
            )
            for window in windows
        ]
        relabeled_adjusted, _ = all_visit.apply_cross_fold_start_budget(
            relabeled,
            estimates,
            average_width=2,
            burst_multiplier=2,
            coordination_cost_s=0.001,
        )
        relabeled_selected = [
            candidate.pattern.url
            for window in relabeled_adjusted
            for candidate in window.candidates
            if candidate.exact_probability > 0.0
        ]
        self.assertEqual(selected, relabeled_selected)

    def test_extractor_creates_visit_continuation_window_causally(self) -> None:
        first = "https://example.test/first"
        second = "https://example.test/second"
        search_response = f"1. [First]({first})\n2. [Second]({second})"
        session = SessionTrace(
            Path("continuation.jsonl"),
            (
                ToolCall(0, 1.0, "search", {"query": ["topic"]}, 1),
                llm(1, 2.0, search_response),
                ToolCall(
                    1,
                    2.0,
                    "visit",
                    {"url": [first]},
                    3,
                    {"unit_duration_s": [2.0]},
                ),
                llm(2, 5.0, "visited first"),
                ToolCall(
                    2,
                    5.0,
                    "visit",
                    {"url": [second]},
                    5,
                    {"unit_duration_s": [3.0]},
                ),
                llm(3, 9.0, "visited second"),
            ),
        )

        decisions = all_visit.extract_all_visit_decisions((session,))

        self.assertEqual(len(decisions), 3)
        self.assertEqual(decisions[0].trigger_tool, "search")
        self.assertEqual(decisions[0].authoritative_urls, (first,))
        continuation = decisions[1]
        self.assertEqual(continuation.trigger_tool, "visit")
        self.assertEqual(continuation.authoritative_urls, (second,))
        by_url = {candidate.url: candidate for candidate in continuation.candidates}
        self.assertTrue(by_url[first].was_visited)
        self.assertFalse(by_url[second].was_visited)
        # The terminal visit has an LLM result but no future tool label.
        self.assertEqual(decisions[2].outcome, "no_next_tool")

    def test_timing_assigns_each_visit_to_exactly_one_predecessor(self) -> None:
        first = "https://example.test/first"
        second = "https://example.test/second"
        session = SessionTrace(
            Path("timing.jsonl"),
            (
                ToolCall(0, 1.0, "search", {"query": ["topic"]}, 1),
                llm(1, 2.0, f"1. [First]({first})\n2. [Second]({second})"),
                ToolCall(
                    1,
                    2.0,
                    "visit",
                    {"url": [first]},
                    3,
                    {"unit_duration_s": [2.0]},
                ),
                llm(2, 5.0, "visited first"),
                ToolCall(
                    2,
                    5.0,
                    "visit",
                    {"url": [second]},
                    5,
                    {"unit_duration_s": [3.0]},
                ),
                llm(3, 9.0, "visited second"),
            ),
        )
        decisions = all_visit.extract_all_visit_decisions((session,))

        # Exercise the timing logic without writing a temporary trace by
        # reproducing its direct corrected-service accounting invariants.
        positive = [row for row in decisions if row.outcome == "visit"]
        self.assertEqual(len(positive), 2)
        target_indexes = [row.target_tool_event_index for row in positive]
        self.assertEqual(target_indexes, [2, 4])
        services = [
            sum(session.events[index].timing_correction["unit_duration_s"])
            for index in target_indexes
            if index is not None
            and isinstance(session.events[index], ToolCall)
            and session.events[index].timing_correction is not None
        ]
        self.assertEqual(services, [2.0, 3.0])

    def test_real_corrected_trace_covers_every_executable_visit(self) -> None:
        traces = (
            REPOSITORY_ROOT
            / "traces"
            / "my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s"
        )
        decisions = all_visit.extract_all_visit_decisions(
            all_visit.load_sessions(traces)
        )

        self.assertEqual(len(decisions), 530)
        positives = [row for row in decisions if row.outcome == "visit"]
        self.assertEqual(len(positives), 229)
        self.assertEqual(
            sum(
                all_visit.executable_url(url)
                for row in positives
                for url in row.authoritative_urls
            ),
            499,
        )
        self.assertEqual(
            sum(row.trigger_tool == "visit" for row in positives),
            89,
        )


if __name__ == "__main__":
    unittest.main()
