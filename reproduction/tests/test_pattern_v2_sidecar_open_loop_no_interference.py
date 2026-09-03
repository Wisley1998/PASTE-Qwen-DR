from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "reproduction" / "scripts"))

import run_pattern_v2_sidecar_open_loop_no_interference as runner  # noqa: E402
from paste_repro.speculation_policy import CandidatePattern  # noqa: E402
from run_pattern_v2_adaptive_load import (  # noqa: E402
    ScoredCandidate,
    ScoredWindow,
)


def _wrong_window(index: int) -> ScoredWindow:
    decision_id = f"decision-{index}"
    candidate = ScoredCandidate(
        pattern=CandidatePattern(
            session_id=f"session-{index}",
            decision_id=decision_id,
            url=f"https://wrong.test/{index}",
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
        exact_probability=0.9,
        visit_probability=0.95,
        rank_only_probability=0.2,
        exact_match=False,
    )
    return ScoredWindow(
        decision_id=decision_id,
        session_id=f"session-{index}",
        v2_gate=True,
        next_tool_visit=True,
        expected_authoritative_calls=1.0,
        coarse_expected_authoritative_calls=1.0,
        targets=(f"https://target.test/{index}",),
        executable_targets=(f"https://target.test/{index}",),
        candidates=(candidate,),
    )


class FixedArrivalTraceTests(unittest.TestCase):
    def test_trace_freezes_modeled_cadence_without_completion_feedback(
        self,
    ) -> None:
        schedule = runner.build_fixed_arrival_trace(
            [_wrong_window(index) for index in range(6)],
            task_concurrency=2,
            seed=7,
            visit_capacity=2,
            service_s=0.004,
            lead_s=0.002,
        )
        self.assertEqual(len(schedule), 3)
        self.assertEqual([row.target_count for row in schedule], [2, 2, 2])
        self.assertEqual(
            [row.ideal_service_waves for row in schedule], [1, 1, 1]
        )
        expected = [0.002, 0.008, 0.014]
        for actual, planned in zip(
            [row.authority_offset_s for row in schedule], expected
        ):
            self.assertTrue(math.isclose(actual, planned, abs_tol=1e-12))
        manifest = runner.arrival_trace_manifest(schedule)
        self.assertEqual(manifest["authority_targets"], 6)
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_cli_defaults_to_requested_paper_matrix(self) -> None:
        args = runner.parse_args([])
        self.assertEqual(args.concurrencies, [1, 16, 64])
        self.assertEqual(args.sidecar_slots, 4)
        self.assertEqual(args.repetitions, 8)
        self.assertEqual(args.probability_threshold, 0.20)
        self.assertEqual(args.speculation_phase_guard_ms, 0.0)
        self.assertEqual(args.authority_control_burst_limit, 0)

    def test_phase_guard_moves_only_later_speculation_releases(self) -> None:
        schedule = runner.build_fixed_arrival_trace(
            [_wrong_window(index) for index in range(6)],
            task_concurrency=2,
            seed=7,
            visit_capacity=2,
            service_s=0.004,
            lead_s=0.002,
            speculation_phase_guard_s=0.001,
        )
        expected_authority = [0.002, 0.008, 0.014]
        expected_speculation = [0.0, 0.007, 0.013]
        for epoch, authority, speculation in zip(
            schedule,
            expected_authority,
            expected_speculation,
        ):
            self.assertTrue(
                math.isclose(
                    epoch.authority_offset_s, authority, abs_tol=1e-12
                )
            )
            self.assertTrue(
                math.isclose(
                    epoch.speculation_offset_s, speculation, abs_tol=1e-12
                )
            )
        self.assertEqual(schedule[0].speculation_phase_guard_s, 0.0)
        self.assertEqual(schedule[1].speculation_phase_guard_s, 0.001)
        manifest = runner.arrival_trace_manifest(schedule)
        self.assertTrue(
            math.isclose(
                manifest["epochs"][1]["effective_speculation_lead_s"],
                0.001,
                abs_tol=1e-12,
            )
        )

    def test_phase_guard_must_leave_positive_speculation_lead(self) -> None:
        with self.assertRaises(ValueError):
            runner.build_fixed_arrival_trace(
                [_wrong_window(0)],
                task_concurrency=1,
                seed=0,
                visit_capacity=1,
                service_s=0.004,
                lead_s=0.002,
                speculation_phase_guard_s=0.002,
            )


class FixedArrivalProcessSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.schedule = runner.build_fixed_arrival_trace(
            [_wrong_window(index) for index in range(4)],
            task_concurrency=2,
            seed=3,
            visit_capacity=2,
            service_s=0.008,
            lead_s=0.004,
        )

    async def _run(
        self,
        slots: int,
        *,
        cpu_isolation: bool = False,
        authority_control_burst_limit: int = 32,
    ) -> dict[str, object]:
        return await runner.run_fixed_arrival_sample(
            self.schedule,
            task_concurrency=2,
            seed=3,
            workers=4,
            visit_capacity=2,
            service_ms=8.0,
            lead_ms=4.0,
            sidecar_slots=slots,
            max_sidecar_pending=8,
            probability_threshold=0.0,
            claim_grace_ms=2.0,
            prestart_ms=30.0,
            cpu_isolation=cpu_isolation,
            authority_control_burst_limit=authority_control_burst_limit,
        )

    async def test_control_burst_gate_fails_closed_to_zero_starts(self) -> None:
        treatment = await self._run(
            4,
            authority_control_burst_limit=1,
        )
        self.assertEqual(treatment["requested_predictions"], 0)
        self.assertEqual(treatment["sidecar_started"], 0)
        self.assertEqual(treatment["authority_control_burst_gated_epochs"], 2)
        self.assertTrue(all(treatment["safety"].values()))

    async def test_missing_certificate_reserves_no_sidecar_cpu(self) -> None:
        treatment = await self._run(
            4,
            cpu_isolation=True,
            authority_control_burst_limit=0,
        )
        self.assertFalse(treatment["sidecar_activated"])
        self.assertEqual(treatment["selection_selected"], 0)
        self.assertEqual(treatment["sidecar_cpu_affinity"], [])
        self.assertEqual(treatment["physical_call_amplification"], 1.0)
        self.assertTrue(all(treatment["safety"].values()))

    async def test_active_sidecar_bridge_uses_sidecar_cpu(self) -> None:
        if len(os.sched_getaffinity(0)) < 2:
            self.skipTest("bridge isolation requires two granted CPUs")
        treatment = await self._run(
            4,
            cpu_isolation=True,
            authority_control_burst_limit=32,
        )
        self.assertTrue(treatment["sidecar_activated"])
        self.assertTrue(
            treatment["safety"][
                "result_bridge_cpu_affinity_certificate"
            ]
        )
        self.assertEqual(
            treatment["sidecar_snapshot"]["actual_bridge_cpu_affinity"],
            treatment["sidecar_cpu_affinity"],
        )
        self.assertTrue(all(treatment["safety"].values()))

    async def test_k0_and_process_k4_replay_identical_absolute_arrivals(
        self,
    ) -> None:
        baseline = await self._run(0)
        treatment = await self._run(4)

        self.assertEqual(
            baseline["arrival_trace_sha256"],
            treatment["arrival_trace_sha256"],
        )
        baseline_arrivals = {
            row["target_id"]: row["planned_arrival_offset_s"]
            for row in baseline["authority_rows"]
        }
        treatment_arrivals = {
            row["target_id"]: row["planned_arrival_offset_s"]
            for row in treatment["authority_rows"]
        }
        self.assertEqual(baseline_arrivals.keys(), treatment_arrivals.keys())
        for target_id, planned in baseline_arrivals.items():
            self.assertTrue(
                math.isclose(
                    planned,
                    treatment_arrivals[target_id],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        self.assertEqual(baseline["requested_predictions"], 0)
        self.assertGreater(treatment["requested_predictions"], 0)
        self.assertGreater(treatment["sidecar_started"], 0)
        self.assertEqual(
            treatment["timer_tasks_armed"],
            treatment["timer_tasks_armed_observed"],
        )
        self.assertGreater(treatment["timer_setup_lead_ms"], 0.0)
        self.assertTrue(treatment["preload_done_before_origin"])
        self.assertEqual(treatment["timed_parent_admission_calls"], 0)
        self.assertEqual(treatment["timed_parent_submit_packets"], 0)
        self.assertEqual(
            treatment["preload_handles_returned"],
            treatment["preload_requested"],
        )
        self.assertFalse(treatment["bridge_started_before_authority_done"])
        transport = treatment["sidecar_snapshot"]["transport"]
        self.assertEqual(transport["transport_submit_packets"], 0)
        self.assertEqual(transport["transport_schedule_packets"], 1)
        self.assertEqual(transport["transport_claims"], 0)
        self.assertEqual(transport["transport_results"], 0)
        self.assertEqual(transport["transport_terminal"], 0)
        self.assertEqual(transport["transport_tombstone_packets"], 0)
        self.assertTrue(all(baseline["safety"].values()))
        self.assertTrue(all(treatment["safety"].values()))

        cell = runner.aggregate_paired_samples(
            task_concurrency=2,
            baseline_samples=[baseline] * 8,
            treatment_samples=[treatment] * 8,
            counterbalance_orders=["AB", "BA"] * 4,
        )
        vectors = cell["raw_repeat_vectors"]
        self.assertEqual(
            len(vectors["authority_latency_regression_ms_per_target"]),
            8,
        )
        self.assertEqual(cell["authority_latency_inference"]["n"], 8)
        self.assertEqual(len(cell["repeat_records"]), 8)
        self.assertTrue(cell["all_timed_parent_submit_packets_zero"])

    async def test_raw_vectors_configuration_hashes_and_report_are_written(
        self,
    ) -> None:
        baseline = await self._run(0)
        treatment = await self._run(4)
        cell = runner.aggregate_paired_samples(
            task_concurrency=2,
            baseline_samples=[baseline],
            treatment_samples=[treatment],
            counterbalance_orders=["AB"],
        )
        payload = {
            "schema": runner.SCHEMA,
            "configuration": {"concurrencies": [2], "repetitions": 1},
            "source_sha256": {"runner": "a" * 64},
            "cells": [cell],
        }
        raw = runner.raw_repeat_payload(payload)
        payload["raw_repeat_vectors_sha256"] = raw[
            "sha256_excluding_self"
        ]
        payload["payload_sha256"] = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner.write_outputs(output, payload)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "REPORT.md").is_file())
            raw_on_disk = json.loads(
                (output / "raw_repeat_vectors.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(raw_on_disk["configuration"], payload["configuration"])
        self.assertEqual(raw_on_disk["source_sha256"], payload["source_sha256"])
        self.assertEqual(len(raw_on_disk["cells"][0]["repeat_records"]), 1)


if __name__ == "__main__":
    unittest.main()
