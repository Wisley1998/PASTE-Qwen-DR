from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import statistics
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_trace_timing_net_benefit as trace_replay  # noqa: E402
from run_pattern_cache_evaluation import cv_fold  # noqa: E402


class TraceTimingSetupTests(unittest.TestCase):
    def test_serial_visit_credit_includes_earlier_authority_work(self) -> None:
        # Four serial 3-second URLs and 2 seconds of initial lead. All exact
        # jobs finish at authority t=1. When hits come first, authority waits
        # once, then reuses the other completed hits before executing the miss.
        self.assertEqual(
            trace_replay.serial_visit_hit_saving(
                12.0, 2.0, (True, True, True, False)
            ),
            8.0,
        )
        # If the miss is first, its 3 seconds also hide every speculative hit.
        self.assertEqual(
            trace_replay.serial_visit_hit_saving(
                12.0, 2.0, (False, True, True, True)
            ),
            9.0,
        )
        self.assertEqual(
            trace_replay.serial_visit_hit_saving(12.0, 2.0, (True,)), 2.0
        )
        self.assertEqual(
            trace_replay.serial_visit_hit_saving(12.0, 2.0, ()), 0.0
        )
        self.assertEqual(
            trace_replay.serial_visit_hit_saving(
                12.0, 2.0, (False, False, False, False)
            ),
            0.0,
        )

    def test_serial_visit_credit_uses_corrected_per_url_service(self) -> None:
        # The first hit needs 4 seconds and has 2 seconds of LLM lead. It waits
        # 2 seconds; the 3-second miss then hides the final 2-second hit.
        self.assertEqual(
            trace_replay.serial_visit_hit_saving(
                9.0,
                2.0,
                (True, False, True),
                (4.0, 3.0, 2.0),
            ),
            4.0,
        )

    def test_llm_queue_scaling_never_relabels_tool_stall(self) -> None:
        traces = REPOSITORY_ROOT / "traces" / "my_traces"
        raw = trace_replay.collect_decision_timings(traces)
        scaled = trace_replay.collect_decision_timings(
            traces, llm_duration_scale=0.70
        )
        self.assertEqual(raw.keys(), scaled.keys())
        for decision_id, timing in raw.items():
            self.assertEqual(
                timing.visit_stall_s, scaled[decision_id].visit_stall_s
            )
            self.assertAlmostEqual(
                scaled[decision_id].llm_overlap_s,
                0.70 * timing.llm_overlap_s,
            )

    def test_service_distribution_is_atomic_and_outer_fold_oof(self) -> None:
        session_by_fold: dict[int, str] = {}
        candidate = 0
        while len(session_by_fold) < 5:
            session_id = f"synthetic-session-{candidate}"
            session_by_fold.setdefault(cv_fold(session_id), session_id)
            candidate += 1

        windows = []
        timings = {}
        services_by_fold = {fold: float(fold + 1) for fold in range(5)}
        for fold in range(5):
            decision_id = f"decision-{fold}"
            session_id = session_by_fold[fold]
            windows.append(
                SimpleNamespace(
                    decision_id=decision_id,
                    session_id=session_id,
                    executable_targets=("https://a.invalid", "https://b.invalid"),
                    candidates=(
                        SimpleNamespace(
                            pattern=SimpleNamespace(url="https://a.invalid")
                        ),
                    ),
                )
            )
            # Two URLs make the atomic service half the observed batch stall.
            timings[decision_id] = trace_replay.DecisionTiming(
                decision_id=decision_id,
                session_id=session_id,
                llm_overlap_s=100.0,
                visit_stall_s=2.0 * services_by_fold[fold],
                authoritative_urls=2,
                timing_status="observed_visit_stall",
            )

        estimates, metadata = trace_replay.build_oof_service_estimates(
            windows, timings
        )
        for held_out_fold in range(5):
            expected = statistics.fmean(
                service
                for fold, service in services_by_fold.items()
                if fold != held_out_fold
            )
            row = estimates[f"decision-{held_out_fold}"]
            self.assertEqual(row.outer_fold, held_out_fold)
            self.assertEqual(row.training_atomic_samples, 8)
            self.assertAlmostEqual(row.expected_overlap_s, expected)
        self.assertEqual(
            metadata["atomic_unit"], "visit_stall / executable URL count"
        )


if __name__ == "__main__":
    unittest.main()
