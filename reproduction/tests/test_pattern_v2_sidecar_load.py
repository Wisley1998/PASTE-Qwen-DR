from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
import os
from pathlib import Path
import sys
from threading import Event, get_ident
import time
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_sidecar_load as sidecar_runner  # noqa: E402
import run_pattern_v2_adaptive_load as adaptive  # noqa: E402
from paste_repro.invocation import Invocation  # noqa: E402
from paste_repro.live_broker import LiveAuthoritativeResult  # noqa: E402
from paste_repro.speculation_sidecar import (  # noqa: E402
    ExactSpeculationKey,
    ProcessSpeculativeSidecar,
    SpeculativeHandle,
)


def _completion(invocation: Invocation) -> sidecar_runner.AuthorityCompletion:
    now = time.perf_counter()
    result = LiveAuthoritativeResult(
        invocation=invocation,
        result={"invocation_key": invocation.key},
        source="executed",
        exposed_wait_s=0.001,
        queue_s=0.0,
        service_s=0.001,
        saved_service_s=0.0,
    )
    return sidecar_runner.AuthorityCompletion(
        result=result,
        scheduled_at=now - 0.001,
        first_run_at=now - 0.001,
        terminal_at=now,
        observed_at=now,
    )


class PullReleaseClockTests(unittest.TestCase):
    def test_raw_prefetch_finishes_before_independent_release(self) -> None:
        called = Event()

        class FakeSidecar:
            pull_epoch_sealed = True

            def prefetch_pull_results(self, *, deadline: float) -> int:
                self.deadline = deadline
                self.thread_id = get_ident()
                called.set()
                return 3

        sidecar = FakeSidecar()
        cutoff = time.monotonic() + 0.010
        prefetch_deadline = cutoff + 0.020
        confirmation = cutoff + 0.050
        parent_thread_id = get_ident()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                sidecar_runner._prefetch_pull_epoch,
                sidecar,
                cutoff,
                prefetch_deadline,
            )
            self.assertTrue(called.wait(timeout=1.0))
            outcome = future.result(timeout=1.0)

        self.assertEqual(outcome.packets, 3)
        self.assertTrue(outcome.sealed)
        self.assertFalse(outcome.deadline_miss)
        self.assertIsNone(outcome.error)
        self.assertNotEqual(outcome.worker_thread_id, parent_thread_id)
        self.assertEqual(outcome.worker_thread_id, sidecar.thread_id)
        self.assertLess(outcome.finished_at, confirmation)
        self.assertEqual(outcome.prefetch_deadline, prefetch_deadline)


class PatternV2SidecarIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.windows, _ = adaptive.collect_nested_oof_windows(
            REPOSITORY_ROOT / "traces" / "my_traces"
        )
        policy = next(
            spec
            for spec in adaptive.policy_specs()
            if spec.name == "safe_global_benefit"
        )
        cls.exact_windows = []
        cls.zero_target_window = None
        for window in cls.windows:
            selected, _ = adaptive._select_candidates(
                [window],
                policy,
                visit_capacity=2,
                service_s=0.020,
                lead_s=0.010,
                isolated_speculative_slots=1,
                safe_start_limit=1,
            )
            if (
                selected
                and len(window.executable_targets) == 1
                and selected[0][0].pattern.url
                == window.executable_targets[0]
            ):
                cls.exact_windows.append(window)
                if len(cls.exact_windows) == 2:
                    if cls.zero_target_window is not None:
                        break
            if not window.executable_targets and selected:
                cls.zero_target_window = window
        if len(cls.exact_windows) < 2:
            raise AssertionError("test fixture has fewer than two exact windows")
        if cls.zero_target_window is None:
            raise AssertionError("test fixture has no selected zero-target window")
        cls.exact_window = cls.exact_windows[0]

    async def test_k_zero_is_true_demand_only_path(self) -> None:
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=4.0,
            lead_ms=2.0,
            sidecar_slots=0,
            max_sidecar_pending=2,
            probability_threshold=0.0,
        )
        self.assertEqual(sample["requested_predictions"], 0)
        self.assertEqual(sample["sidecar_started"], 0)
        self.assertEqual(sample["visible_speculative_hits"], 0)
        self.assertEqual(
            sample["physical_calls_started"], sample["authoritative_targets"]
        )
        self.assertFalse(sample["result_bridge_prestarted"])
        self.assertTrue(all(sample["safety"].values()))

    async def test_exact_sidecar_win_keeps_shadow_authority(self) -> None:
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=20.0,
            lead_ms=10.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            unsafe_positive_ablation=True,
        )
        self.assertEqual(sample["visible_speculative_hits"], 1)
        self.assertEqual(sample["source_counts"], {"sidecar": 1})
        self.assertEqual(sample["authority_stats"]["authoritative_requests"], 1)
        self.assertEqual(sample["authority_stats"]["commits"], 1)
        self.assertLess(
            sample["logical_done_wall_s"], sample["authority_done_wall_s"]
        )
        self.assertTrue(sample["bridge_started_before_authority_done"])
        self.assertTrue(sample["result_bridge_prestarted"])
        self.assertTrue(sample["result_bridge_cpu_affinity_certified"])
        self.assertEqual(
            sample["sidecar_snapshot"]["actual_bridge_cpu_affinity"],
            sample["sidecar_cpu_affinity"],
        )
        self.assertTrue(all(sample["safety"].values()))

    async def test_eager_staged_hit_claims_without_child_packet(self) -> None:
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=30.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            shadow_barrier=True,
            authority_control_burst_limit=32,
            require_precompletion=True,
            completion_guard_ms=2.0,
            eager_result_staging=True,
            unsafe_positive_ablation=True,
        )
        transport = sample["sidecar_snapshot"]["transport"]
        self.assertEqual(sample["visible_speculative_hits"], 1)
        self.assertEqual(transport["transport_claim_packets"], 0)
        self.assertEqual(transport["transport_staged_results"], 1)
        self.assertTrue(
            sample["sidecar_snapshot"]["parent_staging"]["enabled"]
        )
        self.assertTrue(all(sample["safety"].values()))

    async def test_pull_staged_hit_has_no_timed_result_bridge(self) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 2:
            self.skipTest("two granted CPUs are required")
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=30.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            shadow_barrier=True,
            authority_control_burst_limit=32,
            require_precompletion=True,
            completion_guard_ms=10.0,
            pull_result_staging=True,
            certified_exclusive_resources=True,
        )
        transport = sample["sidecar_snapshot"]["transport"]
        self.assertTrue(sample["strict_positive_budget_certificate"])
        self.assertEqual(sample["visible_speculative_hits"], 1)
        self.assertFalse(sample["result_bridge_prestarted"])
        self.assertFalse(sample["bridge_started_before_authority_done"])
        self.assertTrue(sample["pull_prestage_enabled"])
        self.assertTrue(
            sample["pull_mailbox_no_timed_bridge_certified"]
        )
        self.assertTrue(
            sample["pull_prestage_before_authority_certified"]
        )
        self.assertTrue(
            sample["pull_prestage_off_parent_loop_certified"]
        )
        self.assertTrue(sample["pull_prestage_cpu_affinity_certified"])
        self.assertTrue(sample["pull_prestage_quiet_gap_certified"])
        self.assertTrue(sample["pull_prestage_never_gates_authority_release"])
        self.assertEqual(transport["transport_claim_packets"], 0)
        self.assertEqual(transport["transport_pull_hits"], 1)
        self.assertEqual(
            transport["transport_pull_prefetch_packets"], 1
        )
        self.assertEqual(sample["authority_backend_arm_violations"], 0)
        self.assertEqual(
            sample["authority_backend_arm_suppressed_batches"], 0
        )
        self.assertEqual(sample["claims_while_authority_unarmed"], 0)
        self.assertEqual(sample["claims_while_prestage_unready"], 0)
        self.assertEqual(sample["speculative_claim_suppressed_batches"], 0)
        self.assertEqual(
            sample["sidecar_hot_path_ms"]["pull_prestage_calls"], 1
        )
        self.assertEqual(
            sample["sidecar_hot_path_ms"][
                "pull_prestage_deadline_misses"
            ],
            0,
        )
        self.assertEqual(
            sample["sidecar_hot_path_ms"]["pull_prestage_errors"], 0
        )
        self.assertEqual(
            sample["sidecar_hot_path_ms"][
                "pull_prestage_not_ready_at_post_arm_poll_batches"
            ],
            0,
        )
        self.assertTrue(
            sample["sidecar_hot_path_ms"][
                "pull_prestage_off_parent_loop"
            ]
        )
        self.assertEqual(
            sample["sidecar_snapshot"]["parent_staging"]["mode"],
            "pull",
        )
        self.assertTrue(all(sample["safety"].values()))

    async def test_late_raw_prefetch_never_blocks_or_claims(self) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 2:
            self.skipTest("two granted CPUs are required")

        def late_prefetch(
            sidecar: ProcessSpeculativeSidecar,
            completion_cutoff: float,
            prefetch_deadline: float,
        ) -> sidecar_runner._PullPrestageRelease:
            del sidecar, completion_cutoff
            time.sleep(max(0.0, prefetch_deadline - time.monotonic()) + 0.020)
            return sidecar_runner._PullPrestageRelease(
                packets=0,
                elapsed_ms=20.0,
                sealed=False,
                deadline_miss=True,
                worker_thread_id=get_ident(),
                worker_cpu_affinity=tuple(sorted(os.sched_getaffinity(0))),
                finished_at=time.monotonic(),
                prefetch_deadline=prefetch_deadline,
            )

        with patch.object(
            sidecar_runner,
            "_prefetch_pull_epoch",
            late_prefetch,
        ):
            sample = await sidecar_runner._run_sample(
                [self.exact_window],
                offered_concurrency=1,
                seed=0,
                workers=2,
                visit_capacity=2,
                service_ms=8.0,
                lead_ms=30.0,
                sidecar_slots=1,
                max_sidecar_pending=2,
                probability_threshold=0.0,
                shadow_barrier=True,
                authority_control_burst_limit=32,
                require_precompletion=True,
                completion_guard_ms=10.0,
                pull_result_staging=True,
                certified_exclusive_resources=True,
            )

        hot = sample["sidecar_hot_path_ms"]
        self.assertEqual(sample["visible_speculative_hits"], 0)
        self.assertEqual(sample["pull_runtime_latch_trips"], 1)
        self.assertEqual(sample["speculative_claim_suppressed_batches"], 1)
        self.assertEqual(sample["claims_while_authority_unarmed"], 0)
        self.assertEqual(sample["claims_while_prestage_unready"], 0)
        self.assertEqual(
            hot["pull_prestage_not_ready_at_post_arm_poll_batches"], 1
        )
        self.assertEqual(hot["pull_prestage_deadline_misses"], 1)
        self.assertFalse(sample["pull_prestage_before_authority_certified"])
        self.assertFalse(sample["pull_prestage_quiet_gap_certified"])
        self.assertFalse(sample["strict_positive_budget_certificate"])
        self.assertTrue(all(sample["safety"].values()))

    async def test_zero_target_prefetch_is_observed_before_next_batch(
        self,
    ) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 2:
            self.skipTest("two granted CPUs are required")
        sample = await sidecar_runner._run_sample(
            [self.zero_target_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=30.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            shadow_barrier=True,
            authority_control_burst_limit=32,
            require_precompletion=True,
            completion_guard_ms=10.0,
            pull_result_staging=True,
            certified_exclusive_resources=True,
        )
        hot = sample["sidecar_hot_path_ms"]
        self.assertEqual(sample["authoritative_targets"], 0)
        self.assertEqual(sample["handles_returned"], 1)
        self.assertEqual(hot["pull_prestage_required_batches"], 1)
        self.assertEqual(hot["pull_prestage_calls"], 1)
        self.assertEqual(
            hot["pull_prestage_not_ready_at_post_arm_poll_batches"], 0
        )
        self.assertEqual(sample["pull_runtime_latch_trips"], 0)
        self.assertTrue(all(sample["safety"].values()))

    async def test_positive_k_without_resource_certificate_abstains(
        self,
    ) -> None:
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=4.0,
            lead_ms=2.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
        )
        self.assertFalse(sample["sidecar_activated"])
        self.assertEqual(sample["selection_selected"], 0)
        self.assertEqual(sample["physical_call_amplification"], 1.0)
        self.assertEqual(sample["sidecar_cpu_affinity"], [])
        self.assertTrue(all(sample["safety"].values()))

    async def test_coordination_cost_can_close_positive_resource_budget(
        self,
    ) -> None:
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=20.0,
            lead_ms=10.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            coordination_cost_ms=11.0,
            unsafe_positive_ablation=True,
        )
        self.assertEqual(sample["coordination_cost_ms"], 11.0)
        self.assertEqual(sample["selection_selected"], 0)
        self.assertFalse(sample["sidecar_activated"])
        self.assertEqual(sample["physical_call_amplification"], 1.0)
        self.assertTrue(all(sample["safety"].values()))

    async def test_all_wrong_has_no_bridge_hit_and_preserves_authority(self) -> None:
        wrong = adaptive.force_all_wrong([self.exact_window])
        sample = await sidecar_runner._run_sample(
            wrong,
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=6.0,
            lead_ms=3.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            unsafe_positive_ablation=True,
        )
        self.assertEqual(sample["visible_speculative_hits"], 0)
        self.assertEqual(sample["source_counts"], {"authority": 1})
        self.assertEqual(sample["authority_stats"]["authoritative_requests"], 1)
        self.assertEqual(sample["authority_stats"]["commits"], 1)
        self.assertLessEqual(
            sample["logical_done_wall_s"], sample["authority_done_wall_s"]
        )
        self.assertLessEqual(
            sample["authority_done_wall_s"], sample["drained_wall_s"]
        )
        self.assertTrue(sample["bridge_started_before_authority_done"])
        self.assertTrue(sample["result_bridge_prestarted"])
        self.assertEqual(
            sample["sidecar_snapshot"]["transport"][
                "transport_tombstone_packets"
            ],
            0,
        )
        self.assertTrue(all(sample["safety"].values()))

    async def test_strict_shadow_barrier_drains_before_next_batch(self) -> None:
        sample = await sidecar_runner._run_sample(
            self.exact_windows,
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=20.0,
            lead_ms=10.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            shadow_barrier=True,
            authority_control_burst_limit=32,
            unsafe_positive_ablation=True,
        )
        self.assertTrue(sample["shadow_barrier"])
        self.assertEqual(sample["shadow_barrier_violations"], 0)
        self.assertEqual(sample["visible_speculative_hits"], 2)
        self.assertTrue(
            sample["safety"]["strict_shadow_barrier_no_prior_backup"]
        )
        self.assertTrue(all(sample["safety"].values()))

    async def test_control_burst_latch_uses_true_k_zero_fallback(self) -> None:
        sample = await sidecar_runner._run_sample(
            self.exact_windows,
            offered_concurrency=2,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=4.0,
            lead_ms=2.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=1,
            unsafe_positive_ablation=True,
        )
        self.assertFalse(sample["sidecar_activated"])
        self.assertFalse(sample["result_bridge_prestarted"])
        self.assertEqual(sample["requested_predictions"], 0)
        self.assertEqual(sample["sidecar_started"], 0)
        self.assertEqual(sample["selection_selected"], 0)
        self.assertGreaterEqual(
            sample["authority_control_burst_latch_closed_batches"], 1
        )
        self.assertTrue(all(sample["safety"].values()))

    async def test_dedicated_authority_thread_uses_three_cpu_roles(self) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 3:
            self.skipTest("three granted CPUs are required")
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=12.0,
            lead_ms=6.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            dedicated_authority_thread=True,
            unsafe_positive_ablation=True,
        )
        roles = {
            tuple(sample["authority_cpu_affinity"]),
            tuple(sample["control_cpu_affinity"]),
            tuple(sample["sidecar_cpu_affinity"]),
        }
        self.assertEqual(len(roles), 3)
        self.assertTrue(sample["three_lane_logical_cpu_isolation_certified"])
        self.assertFalse(sample["authority_lane_snapshot"]["thread_alive"])
        self.assertTrue(all(sample["safety"].values()))

    async def test_dedicated_authority_process_uses_three_cpu_roles(self) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 3:
            self.skipTest("three granted CPUs are required")
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=30.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            dedicated_authority_process=True,
            require_precompletion=True,
            completion_guard_ms=2.0,
            shadow_barrier=True,
            certified_exclusive_resources=True,
            unsafe_positive_ablation=True,
        )
        roles = {
            tuple(sample["authority_cpu_affinity"]),
            tuple(sample["control_cpu_affinity"]),
            tuple(sample["sidecar_cpu_affinity"]),
        }
        lane = sample["authority_lane_snapshot"]
        self.assertEqual(len(roles), 3)
        self.assertTrue(sample["three_lane_logical_cpu_isolation_certified"])
        self.assertFalse(lane["process_alive"])
        self.assertFalse(lane["bridge_alive"])
        self.assertEqual(lane["submitted"], lane["completed"])
        self.assertEqual(lane["orphan_results"], 0)
        self.assertEqual(sample["visible_speculative_hits"], 1)
        self.assertFalse(sample["strict_positive_budget_certificate"])
        self.assertTrue(sample["sidecar_runtime_certificate_checked"])
        self.assertTrue(sample["sidecar_runtime_certificate_valid"])
        self.assertTrue(all(sample["safety"].values()))

    async def test_strict_direct_eager_delivery_fails_closed(
        self,
    ) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 2:
            self.skipTest("two granted CPUs are required")
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=30.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            require_precompletion=True,
            completion_guard_ms=2.0,
            eager_result_staging=True,
            shadow_barrier=True,
            certified_exclusive_resources=True,
        )
        self.assertFalse(sample["sidecar_activated"])
        self.assertFalse(sample["strict_positive_budget_certificate"])
        self.assertEqual(sample["selection_selected"], 0)
        self.assertTrue(all(sample["safety"].values()))

    async def test_strict_direct_lazy_delivery_fails_closed(self) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 2:
            self.skipTest("two granted CPUs are required")
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=30.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            require_precompletion=True,
            completion_guard_ms=2.0,
            shadow_barrier=True,
            certified_exclusive_resources=True,
        )
        self.assertFalse(sample["sidecar_activated"])
        self.assertFalse(sample["strict_positive_budget_certificate"])
        self.assertEqual(sample["selection_selected"], 0)
        self.assertTrue(all(sample["safety"].values()))

    async def test_bmax_alone_is_not_a_positive_resource_certificate(
        self,
    ) -> None:
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=16.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
        )
        self.assertFalse(sample["strict_static_resource_certificate"])
        self.assertFalse(sample["sidecar_activated"])
        self.assertEqual(sample["selection_selected"], 0)
        self.assertEqual(sample["physical_call_amplification"], 1.0)
        self.assertTrue(all(sample["safety"].values()))

    async def test_invalid_sidecar_startup_certificate_falls_back_to_k_zero(
        self,
    ) -> None:
        if not hasattr(os, "sched_getaffinity") or len(
            os.sched_getaffinity(0)
        ) < 2:
            self.skipTest("two granted CPUs are required")
        with patch.object(
            ProcessSpeculativeSidecar,
            "snapshot",
            return_value={
                "process_alive": False,
                "startup_error": "injected certificate failure",
            },
        ):
            sample = await sidecar_runner._run_sample(
                [self.exact_window],
                offered_concurrency=1,
                seed=0,
                workers=2,
                visit_capacity=2,
                service_ms=8.0,
                lead_ms=30.0,
                sidecar_slots=1,
                max_sidecar_pending=2,
                probability_threshold=0.0,
                authority_control_burst_limit=32,
                require_precompletion=True,
                completion_guard_ms=2.0,
                unsafe_positive_ablation=True,
            )
        self.assertFalse(sample["sidecar_activated"])
        self.assertEqual(sample["selection_selected"], 0)
        self.assertEqual(sample["physical_call_amplification"], 1.0)
        self.assertTrue(sample["sidecar_runtime_certificate_checked"])
        self.assertFalse(sample["sidecar_runtime_certificate_valid"])
        self.assertIsNotNone(sample["sidecar_runtime_certificate_error"])
        self.assertTrue(all(sample["safety"].values()))

    async def test_precompletion_gate_abstains_when_lead_is_too_short(self) -> None:
        sample = await sidecar_runner._run_sample(
            [self.exact_window],
            offered_concurrency=1,
            seed=0,
            workers=2,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=4.0,
            sidecar_slots=1,
            max_sidecar_pending=2,
            probability_threshold=0.0,
            authority_control_burst_limit=32,
            require_precompletion=True,
            completion_guard_ms=1.0,
            unsafe_positive_ablation=True,
        )
        self.assertFalse(sample["predicted_precompletion"])
        self.assertFalse(sample["sidecar_activated"])
        self.assertEqual(sample["selection_selected"], 0)
        self.assertEqual(sample["physical_call_amplification"], 1.0)
        self.assertTrue(all(sample["safety"].values()))


class SingleNotificationRaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_precompleted_hit_uses_parent_local_fast_path(self) -> None:
        invocation = Invocation("visit", {"url": "https://example.test/ready"})
        release = asyncio.Event()

        async def authority_call() -> sidecar_runner.AuthorityCompletion:
            await release.wait()
            return _completion(invocation)

        authority = asyncio.create_task(authority_call())
        handle = SpeculativeHandle(
            ExactSpeculationKey("s", "d", invocation.key)
        )
        terminal_at = time.perf_counter()
        handle.future.set_result(
            {
                "invocation_key": invocation.key,
                "executor_terminal_at": terminal_at,
            }
        )
        actual_loop = asyncio.get_running_loop()

        class RecordingLoop:
            def __init__(self) -> None:
                self.threadsafe_calls = 0

            def create_future(self) -> asyncio.Future[object]:
                return actual_loop.create_future()

            def call_soon_threadsafe(self, callback: object) -> None:
                self.threadsafe_calls += 1
                actual_loop.call_soon_threadsafe(callback)  # type: ignore[arg-type]

        recording_loop = RecordingLoop()
        logical = sidecar_runner._arm_shadow_authority_race(
            authority,
            handle,
            invocation=invocation,
            loop=recording_loop,  # type: ignore[arg-type]
            speculative_terminal_cutoff=terminal_at + 1.0,
        )
        winner = await logical
        self.assertEqual(winner.source, "sidecar")
        self.assertEqual(recording_loop.threadsafe_calls, 0)
        release.set()
        await authority

    async def test_remote_authority_completion_has_tie_priority(self) -> None:
        invocation = Invocation("visit", {"url": "https://example.test/a"})
        raw: ConcurrentFuture[sidecar_runner.AuthorityCompletion] = (
            ConcurrentFuture()
        )
        authority = asyncio.create_task(
            sidecar_runner._observe_remote_authority(raw)
        )
        handle = SpeculativeHandle(
            ExactSpeculationKey("s", "d", invocation.key)
        )
        logical = sidecar_runner._arm_shadow_authority_race(
            authority,
            handle,
            invocation=invocation,
            loop=asyncio.get_running_loop(),
            raw_authority=raw,
        )

        raw.set_result(_completion(invocation))
        handle.future.set_result({"invocation_key": invocation.key})
        winner = await logical
        self.assertEqual(winner.source, "authority")
        await authority

    async def test_observer_cancellation_does_not_cancel_raw_authority(self) -> None:
        raw: ConcurrentFuture[sidecar_runner.AuthorityCompletion] = (
            ConcurrentFuture()
        )
        observer = asyncio.create_task(
            sidecar_runner._observe_remote_authority(raw)
        )
        await asyncio.sleep(0)
        observer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await observer
        self.assertFalse(raw.cancelled())
        invocation = Invocation("visit", {"url": "https://example.test/b"})
        raw.set_result(_completion(invocation))

    async def test_malformed_speculative_identity_falls_back(self) -> None:
        invocation = Invocation("visit", {"url": "https://example.test/c"})

        async def authority_call() -> sidecar_runner.AuthorityCompletion:
            await asyncio.sleep(0)
            return _completion(invocation)

        authority = asyncio.create_task(authority_call())
        handle = SpeculativeHandle(
            ExactSpeculationKey("s", "d", invocation.key)
        )
        logical = sidecar_runner._arm_shadow_authority_race(
            authority,
            handle,
            invocation=invocation,
            loop=asyncio.get_running_loop(),
        )
        handle.future.set_result({"invocation_key": None})
        winner = await logical
        self.assertEqual(winner.source, "authority")


class PairedRepeatInferenceTests(unittest.TestCase):
    def test_requires_eight_paired_repetitions(self) -> None:
        result = sidecar_runner._paired_repeat_inference(
            [0.0, 0.0], margin=0.1
        )
        self.assertEqual(result["decision"], "insufficient_repetitions")

    def test_zero_regression_passes_with_repeat_as_unit(self) -> None:
        result = sidecar_runner._paired_repeat_inference(
            [0.0] * 8, margin=0.1
        )
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["upper_95"], 0.0)

    def test_repeat_level_regression_is_not_diluted_by_targets(self) -> None:
        result = sidecar_runner._paired_repeat_inference(
            [0.2] * 8, margin=0.1
        )
        self.assertEqual(result["decision"], "regression")

    def test_logical_benefit_requires_positive_repeat_lower_bound(self) -> None:
        improved = sidecar_runner._paired_benefit_inference([0.2] * 8)
        regressed = sidecar_runner._paired_benefit_inference([-0.2] * 8)
        self.assertEqual(improved["decision"], "improvement")
        self.assertEqual(regressed["decision"], "regression")


if __name__ == "__main__":
    unittest.main()
