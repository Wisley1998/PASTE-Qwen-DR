from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction"))

import run_pattern_v2_per_trace as per_trace  # noqa: E402
from paste_repro.speculation_policy import CandidatePattern  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredCandidate,
    ScoredWindow,
)


def candidate(
    decision_id: str,
    url: str,
    position: int,
    *,
    exact: bool,
) -> ScoredCandidate:
    return ScoredCandidate(
        pattern=CandidatePattern(
            session_id="trace-a.jsonl",
            decision_id=decision_id,
            url=url,
            position=position,
            query_count=1,
            search_streak=1,
            search_sequence=1,
            candidate_count=5,
            current_count=5,
            repeated_current=False,
            source_rank=position,
            current=True,
            was_visited=False,
            search_age=0,
            appearances=1,
        ),
        exact_probability=0.25,
        visit_probability=0.5,
        rank_only_probability=0.2,
        exact_match=exact,
    )


def window(
    decision_id: str,
    urls: tuple[str, ...],
    targets: tuple[str, ...],
    *,
    gate: bool = True,
) -> ScoredWindow:
    candidates = tuple(
        candidate(
            decision_id,
            url,
            position,
            exact=url in targets,
        )
        for position, url in enumerate(urls, 1)
    )
    return ScoredWindow(
        decision_id=decision_id,
        session_id="trace-a.jsonl",
        v2_gate=gate,
        next_tool_visit=bool(targets),
        expected_authoritative_calls=0.5,
        coarse_expected_authoritative_calls=0.5,
        targets=targets,
        executable_targets=targets,
        candidates=candidates,
    )


class PerTraceQualityTests(unittest.TestCase):
    def test_request_axis_uses_numeric_task_number_not_lexical_order(self) -> None:
        sessions = [
            SimpleNamespace(session_id="trace_x_task10_example.jsonl"),
            SimpleNamespace(session_id="trace_x_task2_example.jsonl"),
            SimpleNamespace(session_id="trace_x_task1_example.jsonl"),
        ]

        ordered = per_trace.ordered_request_sessions(sessions)

        self.assertEqual(
            [row["request_number"] for row in ordered], [1, 2, 10]
        )
        self.assertEqual(
            [row["request_order_index"] for row in ordered], [1, 2, 3]
        )

    def test_top_k_is_exact_target_recall_and_respects_gate(self) -> None:
        windows = [
            window(
                "d1",
                (
                    "https://candidate/1",
                    "https://candidate/2",
                    "https://candidate/3",
                    "https://target/a",
                    "https://target/b",
                ),
                ("https://target/a", "https://target/b"),
            ),
            window(
                "d2",
                ("https://target/c",),
                ("https://target/c",),
                gate=False,
            ),
        ]

        result = per_trace._trace_quality(windows)

        self.assertEqual(result["authoritative_targets"], 3)
        self.assertEqual(result["top1_target_hits"], 0)
        self.assertEqual(result["top3_target_hits"], 0)
        self.assertEqual(result["top5_target_hits"], 2)
        self.assertAlmostEqual(result["top5_target_recall"], 2 / 3)
        self.assertEqual(result["top5_hit_visit_windows"], 1)

    def test_zero_target_rates_are_null(self) -> None:
        result = per_trace._trace_quality(
            [window("d1", ("https://candidate/1",), (), gate=True)]
        )
        self.assertIsNone(result["top1_target_recall"])
        self.assertIsNone(result["top5_visit_window_coverage"])


class PerTraceRequestRowsTests(unittest.TestCase):
    def test_rows_include_zero_target_trace_and_cumulative_metrics(self) -> None:
        trace_a = "trace-a.jsonl"
        trace_b = "trace-b.jsonl"
        windows = {
            trace_a: [
                window(
                    "d1",
                    ("https://target/a",),
                    ("https://target/a",),
                )
            ],
            trace_b: [],
        }
        profiles = {
            trace_a: {
                "predictor_windows_evaluated": 1,
                "probability_candidates_evaluated": 1,
                "selected": 1,
                "selected_hits": 1,
                "allocation_weight": 1.0,
            },
            trace_b: {
                "predictor_windows_evaluated": 0,
                "probability_candidates_evaluated": 0,
                "selected": 0,
                "selected_hits": 0,
                "allocation_weight": 0.0,
            },
        }
        baseline = {
            "selection_compute_ms": 0.0,
            "wall_s": 0.010,
            "total_exposed_wait_ms": 5.0,
            "authoritative_rows": [
                {
                    "target_id": "d1:target:0",
                    "source": "authoritative",
                    "exposed_wait_ms": 5.0,
                }
            ],
        }
        policy = {
            "selection_compute_ms": 0.2,
            "predictor_windows_evaluated": 1,
            "probability_candidates_evaluated": 1,
            "selection_selected": 1,
            "selection_selected_hits": 1,
            "wrong_started": 0,
            "wrong_service_ms": 0.0,
            "physical_started": 1,
            "authoritative_targets": 1,
            "wall_s": 0.004,
            "total_exposed_wait_ms": 1.0,
            "authoritative_rows": [
                {
                    "target_id": "d1:target:0",
                    "source": "reused",
                    "exposed_wait_ms": 1.0,
                }
            ],
        }

        cell = {
            "repeat_conservative_net_latency_benefit_ms": [3.65],
            "selection_selected": 1,
            "selection_selected_hits": 1,
            "overlap_producing_hits": 1,
            "authoritative_targets": 1,
            "overlap_producing_target_coverage": 1.0,
            "source_counts": {"reused": 1},
            "wrong_speculations_started": 0,
            "wasted_speculative_service_ms": 0.0,
            "physical_call_amplification_vs_demand_only": 1.0,
        }
        request_a = {
            "request_number": 1,
            "request_order_index": 1,
            "source_task_number": 1,
            "trace_id": trace_a,
            "mapping": "parsed_task_number",
        }
        request_b = {
            "request_number": 2,
            "request_order_index": 2,
            "source_task_number": 2,
            "trace_id": trace_b,
            "mapping": "parsed_task_number",
        }
        rows = [
            per_trace.build_isolated_request_row(
                request=request_a,
                windows=windows[trace_a],
                cell=cell,
                baseline_samples=[baseline],
                policy_samples=[policy],
                profile=profiles[trace_a],
                feature_ms_per_window=0.1,
                probability_ms_per_candidate=0.05,
            ),
            per_trace.build_isolated_request_row(
                request=request_b,
                windows=windows[trace_b],
                cell=None,
                baseline_samples=[],
                policy_samples=[],
                profile=profiles[trace_b],
                feature_ms_per_window=0.1,
                probability_ms_per_candidate=0.05,
            ),
        ]
        per_trace.add_cumulative_metrics(rows)

        self.assertEqual([row["request_number"] for row in rows], [1, 2])
        self.assertAlmostEqual(
            rows[0]["conservative_latency_benefit_ms_mean_per_request"],
            3.65,
        )
        self.assertEqual(rows[0]["runtime_overall_hit_rate"], 1.0)
        self.assertEqual(rows[0]["cumulative_top1_target_recall"], 1.0)
        self.assertAlmostEqual(
            rows[0]["request_critical_path_benefit_ms_mean"], 5.85
        )
        self.assertIsNone(rows[1]["conservative_speedup_fraction"])
        self.assertEqual(
            rows[1]["timing_status"],
            "not_applicable_no_search_decision",
        )
        self.assertEqual(rows[1]["cumulative_runtime_overall_hit_rate"], 1.0)

    def test_zero_target_search_trace_keeps_hit_na_but_measures_wall(self) -> None:
        no_target_window = window(
            "d1", ("https://candidate/1",), (), gate=True
        )
        baseline = {
            "wall_s": 0.003,
            "authoritative_targets": 0,
        }
        policy = {
            "wall_s": 0.004,
            "predictor_windows_evaluated": 1,
            "probability_candidates_evaluated": 1,
            "selection_compute_ms": 0.1,
            "selection_selected": 1,
            "selection_selected_hits": 0,
            "wrong_started": 1,
            "wrong_service_ms": 0.5,
            "physical_started": 1,
            "authoritative_targets": 0,
        }

        row = per_trace.build_isolated_request_row(
            request={
                "request_number": 2,
                "request_order_index": 2,
                "source_task_number": 2,
                "trace_id": "trace-a.jsonl",
                "mapping": "parsed_task_number",
            },
            windows=[no_target_window],
            cell=None,
            baseline_samples=[baseline],
            policy_samples=[policy],
            profile={"selected": 1},
            feature_ms_per_window=0.1,
            probability_ms_per_candidate=0.05,
        )

        self.assertIsNone(row["runtime_overall_hit_rate"])
        self.assertIsNone(row["top1_target_recall"])
        self.assertAlmostEqual(row["baseline_latency_ms"], 3.0)
        self.assertAlmostEqual(row["pattern_conservative_latency_ms"], 4.15)
        self.assertEqual(row["wrong_speculations_started_per_replay"], 1.0)
        self.assertTrue(row["timing_status"].startswith("measured"))

    def test_output_is_valid_json_and_flat_csv(self) -> None:
        payload = {
            "requests": [
                {
                    "request_number": 1,
                    "trace_id": "trace-a.jsonl",
                    "runtime_source_counts": {"reused": 1},
                    "top1_target_recall": 1.0,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            per_trace.write_outputs(output, payload)
            loaded = json.loads((output / "metrics.json").read_text())
            with (output / "per_request.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(loaded, payload)
        self.assertEqual(csv_rows[0]["request_number"], "1")
        self.assertEqual(csv_rows[0]["runtime_source_counts"], '{"reused":1}')


if __name__ == "__main__":
    unittest.main()
