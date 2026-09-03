from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_trace_multi_spec_wall as multi  # noqa: E402


class MultiSpecWallTests(unittest.TestCase):
    def test_selection_supports_multiple_candidates_per_task(self) -> None:
        candidates = tuple(
            SimpleNamespace(
                exact_probability=probability,
                pattern=SimpleNamespace(
                    url=f"https://example.invalid/{index}",
                    position=index,
                    session_id="session",
                    decision_id="decision",
                ),
            )
            for index, probability in enumerate((0.8, 0.5, 0.1), 1)
        )
        window = SimpleNamespace(v2_gate=True, candidates=candidates)
        estimate = SimpleNamespace(overlap_for_url=lambda _: 1.0)
        selected = multi.select_per_task_candidates(
            window,
            estimate,
            per_task_width=2,
            coordination_cost_s=0.01,
        )
        self.assertEqual(selected, candidates[:2])

    def test_selection_drops_nonpositive_value(self) -> None:
        candidates = (
            SimpleNamespace(
                exact_probability=0.5,
                pattern=SimpleNamespace(
                    url="https://example.invalid/good",
                    position=1,
                    session_id="session",
                    decision_id="decision",
                ),
            ),
            SimpleNamespace(
                exact_probability=0.001,
                pattern=SimpleNamespace(
                    url="https://example.invalid/bad",
                    position=2,
                    session_id="session",
                    decision_id="decision",
                ),
            ),
        )
        window = SimpleNamespace(v2_gate=True, candidates=candidates)
        estimate = SimpleNamespace(overlap_for_url=lambda _: 1.0)
        selected = multi.select_per_task_candidates(
            window,
            estimate,
            per_task_width=5,
            coordination_cost_s=0.01,
        )
        self.assertEqual(selected, candidates[:1])

    def test_event_driven_list_schedule_has_no_decision_barrier(self) -> None:
        rows = [
            SimpleNamespace(session_id="a", duration=10.0),
            SimpleNamespace(session_id="b", duration=3.0),
            SimpleNamespace(session_id="c", duration=2.0),
        ]
        serial = multi.list_schedule_makespan(
            rows, concurrency=1, seed=0, duration_field="duration"
        )
        parallel = multi.list_schedule_makespan(
            rows, concurrency=3, seed=0, duration_field="duration"
        )
        self.assertEqual(serial, 15.0)
        self.assertEqual(parallel, 10.0)

    def test_width_one_reconciles_with_existing_unbounded_replay(self) -> None:
        traces = REPOSITORY_ROOT / "traces" / "my_traces"
        windows, _ = multi.collect_nested_oof_windows(traces)
        timings = multi.collect_decision_timings(
            traces, llm_duration_scale=0.70
        )
        estimates, _ = multi.build_oof_service_estimates(windows, timings)
        sessions = multi.build_session_replays(
            windows,
            timings,
            estimates,
            multi.session_full_walls(traces, llm_duration_scale=0.70),
            per_task_width=1,
            coordination_cost_s=0.001,
        )
        self.assertEqual(sum(row.selected_speculations for row in sessions), 314)
        self.assertEqual(sum(row.exact_url_hits for row in sessions), 49)
        self.assertAlmostEqual(
            sum(row.net_saved_visit_stall_s for row in sessions),
            20.734167538642886,
        )


if __name__ == "__main__":
    unittest.main()
