from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction"))

from paste_repro.speculation_policy import CandidatePattern  # noqa: E402
from run_pattern_v2_adaptive_load import ScoredCandidate  # noqa: E402
import run_pattern_v2_trace_all_visit_shared_capacity as shared  # noqa: E402


def candidate(url: str, probability: float = 0.5) -> ScoredCandidate:
    return ScoredCandidate(
        pattern=CandidatePattern(
            session_id="s",
            decision_id="d",
            url=url,
            position=1,
            query_count=1,
            search_streak=1,
            search_sequence=1,
            candidate_count=1,
            current_count=1,
            repeated_current=False,
            source_rank=1,
            current=True,
            was_visited=False,
            search_age=0,
            appearances=1,
        ),
        exact_probability=probability,
        visit_probability=1.0,
        rank_only_probability=probability,
        exact_match=False,
    )


class PreemptibleVisitPoolTests(unittest.TestCase):
    def test_wrong_running_speculation_is_preempted_for_authority(self) -> None:
        loop = shared.EventLoop()
        pool = shared.PreemptibleVisitPool(loop, capacity=1, speculative_cap=1)
        completed: list[float] = []
        pool.submit_speculation(
            session_id="s",
            decision_id="d",
            candidate=candidate("https://wrong.example"),
            duration_s=10.0,
        )
        loop.schedule(
            1.0,
            1,
            lambda: pool.request_authority(
                session_id="s",
                url="https://right.example",
                duration_s=2.0,
                on_complete=completed.append,
            ),
        )
        loop.run()

        self.assertEqual(completed, [3.0])
        self.assertEqual(pool.metrics["preempted_speculations"], 1)
        self.assertAlmostEqual(pool.preempted_speculative_s, 1.0)
        self.assertAlmostEqual(pool.authority_queue_wait_s, 0.0)
        self.assertAlmostEqual(pool.authority_exposed_s, 2.0)

    def test_all_wrong_replay_falls_back_to_baseline_and_charges_waste(self) -> None:
        predicted = candidate("https://right.example")
        session = shared.PreparedSession(
            session_id="s",
            full_wall_s=3.0,
            epochs=(
                shared.Epoch(
                    decision_id="d",
                    original_start_s=0.0,
                    authority_offset_s=1.0,
                    baseline_authority_done_s=3.0,
                    targets=("https://right.example",),
                    services_s=(2.0,),
                    candidates=(predicted,),
                    candidate_services_s=(2.0,),
                ),
            ),
        )
        baseline = shared.simulate(
            [session],
            policy=None,
            visit_capacity=1,
            offered_concurrency=1,
            seed=0,
        )
        all_wrong = shared.simulate(
            [session],
            policy=shared.Policy(
                "fixed_top1", "fixed_top1", "adaptive_idle_fill"
            ),
            visit_capacity=1,
            offered_concurrency=1,
            seed=0,
            wrong_fraction=1.0,
        )

        self.assertEqual(all_wrong["cache_hits"], 0)
        self.assertAlmostEqual(all_wrong["makespan_s"], baseline["makespan_s"])
        self.assertAlmostEqual(all_wrong["wasted_speculative_s"], 1.0)
        self.assertEqual(all_wrong["wasted_speculative_starts"], 1)
        self.assertAlmostEqual(all_wrong["wasted_speculative_fraction"], 1.0)
        self.assertAlmostEqual(
            all_wrong["authority_exposed_s"], baseline["authority_exposed_s"]
        )

    def test_zero_duration_speculation_still_counts_as_a_wasted_call(self) -> None:
        session = shared.PreparedSession(
            session_id="s",
            full_wall_s=1.0,
            epochs=(
                shared.Epoch(
                    decision_id="d",
                    original_start_s=0.0,
                    authority_offset_s=1.0,
                    baseline_authority_done_s=1.0,
                    targets=("https://right.example",),
                    services_s=(0.0,),
                    candidates=(candidate("https://wrong.example"),),
                    candidate_services_s=(0.0,),
                ),
            ),
        )
        result = shared.simulate(
            [session],
            policy=shared.Policy(
                "fixed_top1", "fixed_top1", "adaptive_idle_fill"
            ),
            visit_capacity=1,
            offered_concurrency=1,
            seed=0,
        )

        self.assertEqual(result["physical_speculative_starts"], 1)
        self.assertEqual(result["useful_speculative_starts"], 0)
        self.assertEqual(result["wasted_speculative_starts"], 1)

    def test_matching_running_speculation_is_promoted_with_progress(self) -> None:
        loop = shared.EventLoop()
        pool = shared.PreemptibleVisitPool(loop, capacity=1, speculative_cap=1)
        completed: list[float] = []
        url = "https://right.example"
        pool.submit_speculation(
            session_id="s",
            decision_id="d",
            candidate=candidate(url),
            duration_s=2.0,
        )
        loop.schedule(
            1.0,
            1,
            lambda: pool.request_authority(
                session_id="s",
                url=url,
                duration_s=2.0,
                on_complete=completed.append,
            ),
        )
        loop.run()

        self.assertEqual(completed, [2.0])
        self.assertEqual(pool.metrics["inflight_cache_hits"], 1)
        self.assertEqual(pool.metrics["physical_authority_starts"], 0)
        self.assertAlmostEqual(pool.authority_queue_wait_s, 0.0)
        self.assertAlmostEqual(pool.authority_exposed_s, 1.0)


if __name__ == "__main__":
    unittest.main()
