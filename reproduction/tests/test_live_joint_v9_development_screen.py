from __future__ import annotations

import contextlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_live_joint_v9_development_screen as aggregate  # type: ignore
import run_live_joint_v9_development_screen as runner  # type: ignore
import validate_live_joint_v9_development_screen as validator  # type: ignore


def _stage0_row(
    interval: float,
    *,
    accepted: bool,
    retry_only: bool = False,
    failed: list[str] | None = None,
) -> dict[str, object]:
    return {
        "valid": True,
        "stage": "stage0",
        "cell_id": "A",
        "development_only": True,
        "formal_evidence_eligible": False,
        "visit_interval_s": interval,
        "accepted": accepted,
        "failed_gates": [] if failed is None else failed,
        "retry_only_fallback_eligible": retry_only,
        "server_instance_id": f"server-{interval}",
    }


class ConfigAndWorkloadTests(unittest.TestCase):
    def test_frozen_config_is_exact_development_profile(self) -> None:
        config = validator.load_frozen_config()
        self.assertEqual(config, validator.EXPECTED_CONFIG)
        self.assertEqual(config["PASTE_LIVE_REPLICAS"], "5")
        self.assertEqual(config["PASTE_LIVE_VISIT_MIN_START_INTERVAL_S"], "2.5")
        self.assertEqual(config["VLLM_ENABLE_PREFIX_CACHING"], "1")
        self.assertEqual(config["VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY"], "0")
        self.assertEqual(
            validator.sha256_file(
                ROOT / "reproduction/paste_repro/live_broker.py"
            ),
            validator.EXPECTED_LIVE_BROKER_SHA256,
        )

    def test_workload_is_bound_and_non_formal(self) -> None:
        value = validator.validate_development_workload()
        self.assertTrue(value["development_only"])
        self.assertFalse(value["formal_eligible"])
        self.assertFalse(value["formal_evidence_eligible"])
        self.assertEqual(value["source_count"], 16)
        self.assertEqual(value["replicas"], 5)
        self.assertEqual(value["task_count"], 80)

    def test_formal_workload_path_is_rejected(self) -> None:
        formal = ROOT / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v9.json"
        with self.assertRaisesRegex(
            validator.DevelopmentScreenValidationError,
            "only frozen tune-v1",
        ):
            validator.validate_development_workload(formal)
        self.assertIn(
            "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20",
            validator.FORMAL_WORKLOAD_SHA256S,
        )


class TransportSelectionTests(unittest.TestCase):
    def test_first_passing_interval_stops_ladder(self) -> None:
        selection = validator.select_transport_interval(
            [_stage0_row(2.5, accepted=True)]
        )
        self.assertEqual(selection["selected_visit_interval_s"], 2.5)
        self.assertFalse(selection["candidate_performance_observed_or_used"])

    def test_retry_only_failure_allows_one_fresh_3s_baseline(self) -> None:
        selection = validator.select_transport_interval(
            [
                _stage0_row(
                    2.5,
                    accepted=False,
                    retry_only=True,
                    failed=["transport_zero_retry_and_at_most_2pct"],
                ),
                _stage0_row(3.0, accepted=True),
            ]
        )
        self.assertEqual(selection["selected_visit_interval_s"], 3.0)
        self.assertEqual(selection["selection_input_cells"], ["A", "A"])

    def test_non_transport_failure_forbids_fallback(self) -> None:
        with self.assertRaisesRegex(
            validator.DevelopmentScreenValidationError,
            "non-transport",
        ):
            validator.select_transport_interval(
                [
                    _stage0_row(
                        2.5,
                        accepted=False,
                        retry_only=False,
                        failed=["natural_dual_queue_load"],
                    )
                ]
            )

    def test_second_retry_failure_stops(self) -> None:
        with self.assertRaisesRegex(
            validator.DevelopmentScreenValidationError,
            "did not pass",
        ):
            validator.select_transport_interval(
                [
                    _stage0_row(
                        2.5,
                        accepted=False,
                        retry_only=True,
                        failed=["transport_zero_retry_and_at_most_2pct"],
                    ),
                    _stage0_row(
                        3.0,
                        accepted=False,
                        retry_only=True,
                        failed=["transport_zero_retry_and_at_most_2pct"],
                    ),
                ]
            )

    def test_3s_is_forbidden_after_2p5_passes(self) -> None:
        with self.assertRaisesRegex(
            validator.DevelopmentScreenValidationError,
            "already passed",
        ):
            validator.select_transport_interval(
                [
                    _stage0_row(2.5, accepted=True),
                    _stage0_row(3.0, accepted=True),
                ]
            )


class EstimatorAndSelectionTests(unittest.TestCase):
    def test_policy_precedence_is_preregistered(self) -> None:
        self.assertEqual(
            aggregate.select_policy(
                f0_passed=True,
                f1_base_passed=True,
                f1_incremental_passed=True,
            ),
            "F1",
        )
        self.assertEqual(
            aggregate.select_policy(
                f0_passed=True,
                f1_base_passed=True,
                f1_incremental_passed=False,
            ),
            "F0",
        )
        self.assertIsNone(
            aggregate.select_policy(
                f0_passed=False,
                f1_base_passed=True,
                f1_incremental_passed=False,
            )
        )

    def test_bootstrap_uses_exactly_16_source_means(self) -> None:
        baseline = {f"s{i:02d}": 10.0 + i for i in range(16)}
        candidate = {key: value - 1.0 for key, value in baseline.items()}
        result = aggregate._bootstrap(baseline, candidate, resamples=200)
        self.assertEqual(result["sample_size"], 16)
        self.assertEqual(result["resamples"], 200)
        self.assertGreater(result["absolute_reduction_s_95_ci"][0], 0)

    def test_replica_then_block_folding(self) -> None:
        runs: dict[str, dict[str, object]] = {}
        for block_number, block_id in enumerate(("b1", "b2"), 1):
            tasks = {}
            for source in range(16):
                for replica in range(5):
                    tasks[(f"s{source:02d}", replica)] = {
                        "e2e_s": 100 * block_number + 10 * source + replica
                    }
            runs[block_id] = {"E": SimpleNamespace(tasks_by_key=tasks)}
        folded = aggregate._fold_source_metric(  # type: ignore[arg-type]
            runs, cell="E", metric="e2e_s"
        )
        # source 3: block means are 132 and 232; two-block mean is 182.
        self.assertEqual(folded["s03"], 182.0)


class ReservationTests(unittest.TestCase):
    def test_reservation_flags_replay_as_zero_one_debt(self) -> None:
        records = [
            {
                "started_at": 1.0,
                "tool": "visit",
                "speculative": True,
                "authoritative": True,
                "reserved_speculative_dispatch": True,
                "authoritative_after_reserved_dispatch": False,
                "dispatch_lane": "speculative",
                "dispatch_reason": "reserved_speculative",
                "running_speculative_before": 0,
                "queued_authoritative_same_tool_before": 1,
                "reservation_debt_before": False,
                "reservation_debt_after": True,
                "per_tool_dispatch_ordinal": 1,
            },
            {
                "started_at": 2.0,
                "tool": "visit",
                "speculative": False,
                "authoritative": True,
                "reserved_speculative_dispatch": False,
                "authoritative_after_reserved_dispatch": True,
                "dispatch_lane": "authoritative",
                "dispatch_reason": "authoritative_repayment",
                "running_speculative_before": 0,
                "queued_authoritative_same_tool_before": 1,
                "reservation_debt_before": True,
                "reservation_debt_after": False,
                "per_tool_dispatch_ordinal": 2,
            },
        ]
        run = SimpleNamespace(
            config={
                "min_speculative_tool_workers": 1,
                "speculative_tool_workers": 2,
            },
            physical_records=records,
            payload={
                "broker_final_snapshot": {
                    "reservation": {
                        "authoritative_turn_due_by_tool": [],
                        "reserved_speculative_dispatches": 1,
                        "authoritative_after_reserved_dispatches": 1,
                    },
                    "stats": {
                        "reserved_speculative_dispatches": 1,
                        "authoritative_after_reserved_dispatches": 1,
                        "completed_reuse": 7,
                    },
                }
            },
        )
        audit = aggregate._reservation_audit(run, label="b/F1")  # type: ignore[arg-type]
        self.assertTrue(audit["final_debt_zero"])
        self.assertTrue(audit["all_dispatch_rows_causally_replayed"])

    def test_double_reserved_dispatch_is_rejected(self) -> None:
        records = [
            {
                "started_at": float(index),
                "tool": "visit",
                "speculative": True,
                "authoritative": True,
                "reserved_speculative_dispatch": True,
                "authoritative_after_reserved_dispatch": False,
                "dispatch_lane": "speculative",
                "dispatch_reason": "reserved_speculative",
                "running_speculative_before": 0,
                "queued_authoritative_same_tool_before": 1,
                "reservation_debt_before": False,
                "reservation_debt_after": True,
                "per_tool_dispatch_ordinal": index,
            }
            for index in (1, 2)
        ]
        run = SimpleNamespace(
            config={
                "min_speculative_tool_workers": 1,
                "speculative_tool_workers": 2,
            },
            physical_records=records,
            payload={
                "broker_final_snapshot": {
                    "reservation": {},
                    "stats": {},
                }
            },
        )
        with self.assertRaisesRegex(
            aggregate.DevelopmentScreenAggregationError,
            "debt is not replayable",
        ):
            aggregate._reservation_audit(run, label="b/F1")  # type: ignore[arg-type]


class RunnerTests(unittest.TestCase):
    def test_f1_runner_command_maps_metadata_but_keeps_treatment_label(self) -> None:
        config = runner._derived_config(
            validator.load_frozen_config(), visit_interval_s=3.0, cell="F1"
        )
        command = runner._runner_command(
            python=Path("/python"),
            output=Path("/evidence"),
            cell="F1",
            block_id="block",
            order_index=0,
            server_instance_id="server",
            config=config,
        )
        self.assertEqual(
            command[command.index("--formal-cell-id") + 1], "F"
        )
        self.assertEqual(
            command[command.index("--cell-label") + 1], "block-F1"
        )
        self.assertEqual(
            command[command.index("--min-speculative-tool-workers") + 1], "1"
        )
        self.assertEqual(
            command[command.index("--visit-min-start-interval-s") + 1], "3.0"
        )

    def test_check_only_never_invokes_live_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake_base = Path(temporary) / "runs"
            fake_preflight = {
                "schema": "check",
                "valid": True,
                "workload_validation": {},
                "fixed_final_grammar_feasibility": {},
            }
            output = StringIO()
            with (
                mock.patch.object(runner, "RUN_BASE", fake_base),
                mock.patch.object(runner, "_preflight", return_value=fake_preflight),
                mock.patch.object(runner.formal, "_run_logged") as run_logged,
                contextlib.redirect_stdout(output),
            ):
                code = runner.main(["unit-check", "--check-only"])
            self.assertEqual(code, 0)
            run_logged.assert_not_called()
            self.assertFalse(fake_base.exists())
            self.assertTrue(json.loads(output.getvalue())["valid"])


if __name__ == "__main__":
    unittest.main()
