from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPOSITORY_ROOT
    / "reproduction/scripts/run_scheduler_live_sensitivity.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scheduler_live_sensitivity", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SchedulerLiveSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()
        cls.base = cls.module.formal.load_frozen_config(cls.module.BASE_CONFIG)

    def test_registered_suites_are_small_and_cover_requested_dimensions(self) -> None:
        self.assertEqual(len(self.module.cells_for_suite("target")), 4)
        self.assertEqual(len(self.module.cells_for_suite("high")), 2)
        self.assertEqual(self.module.REGISTERED_SUITES, ("target", "high"))
        self.assertEqual(
            {
                cell.physical_kv_target
                for cell in self.module.cells_for_suite("target")
                if cell.cell == "E" and cell.is_reference_shape
            },
            {0.85, 0.93, 0.97},
        )
        self.assertEqual(
            [cell.context_padding_tokens for cell in self.module.cells_for_suite("high")],
            [12_000, 12_000],
        )
        with self.assertRaisesRegex(self.module.LiveSensitivityError, "unknown suite"):
            self.module.cells_for_suite("shape")
        with self.assertRaisesRegex(self.module.LiveSensitivityError, "unknown suite"):
            self.module.cells_for_suite("all")

    def test_a_e_pairs_change_no_common_workload_configuration(self) -> None:
        for suite, maximum_context, planned_indices in (
            ("target", 10_000, [0, 1, 2, 3]),
            ("high", 12_000, [0, 1]),
        ):
            with self.subTest(suite=suite):
                specs = self.module.cells_for_suite(suite)
                audit = self.module._matrix_invariants(
                    specs, self.base, gpus="4,5,6,7", port=8100
                )
                self.assertTrue(audit["all_cells_use_same_workload"])
                self.assertEqual(len(audit["pair_checks"]), 1)
                self.assertEqual(
                    audit["context_headroom"]["maximum_planned_context_padding"],
                    maximum_context,
                )
                self.assertGreater(
                    audit["context_headroom"]["safe_context_padding_ceiling"],
                    maximum_context,
                )
                self.assertTrue(
                    audit["offered_load_is_strictly_below_native_sequence_ceiling"]
                )
                self.assertTrue(
                    all(not row["common_config_diff"] for row in audit["pair_checks"])
                )
                self.assertEqual(
                    audit["formal_order_index_gate"]["planned_indices"],
                    planned_indices,
                )
                self.assertTrue(
                    audit["formal_order_index_gate"]["all_indices_in_range"]
                )
                self.assertEqual(
                    audit["formal_order_index_gate"]["underlying_runner"]["maximum"],
                    3,
                )

    def test_target_sweep_changes_only_active_physical_kv_target(self) -> None:
        specs = self.module.cells_for_suite("target")
        audit = self.module._matrix_invariants(
            specs, self.base, gpus="4,5,6,7", port=8100
        )
        changed = {
            tuple(row["changed_keys"]) for row in audit["target_checks"]
        }
        self.assertEqual(
            changed,
            {
                (),
                (
                    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
                ),
            },
        )

    def test_fcfs_hides_target_while_joint_receives_it(self) -> None:
        baseline, candidate = self.module.cells_for_suite("high")
        baseline_config = self.module._derived_config(
            self.base, baseline, gpus="4,5,6,7", port=8100
        )
        candidate_config = self.module._derived_config(
            self.base, candidate, gpus="4,5,6,7", port=8100
        )
        baseline_environment = self.module.formal._cell_environment(
            baseline_config, cell="A"
        )
        candidate_environment = self.module.formal._cell_environment(
            candidate_config, cell="E"
        )
        key = "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"
        self.assertNotIn(key, baseline_environment)
        self.assertEqual(candidate_environment[key], "0.93")
        self.assertEqual(baseline_environment["VLLM_SCHED_POLICY"], "fcfs")
        self.assertEqual(
            candidate_environment["VLLM_SCHED_POLICY"],
            "online_joint_pacer_v2",
        )
        self.assertEqual(
            baseline_config["PASTE_LIVE_VISIT_MIN_START_INTERVAL_S"], "3.0"
        )
        self.assertEqual(
            candidate_config["PASTE_LIVE_VISIT_MIN_START_INTERVAL_S"], "3.0"
        )

    def test_preflight_rejects_a_fifth_formal_order_index(self) -> None:
        specs = (
            *self.module.cells_for_suite("target"),
            self.module.cells_for_suite("high")[0],
        )
        with self.assertRaisesRegex(
            self.module.LiveSensitivityError,
            "exceed the underlying runner range",
        ):
            self.module._matrix_invariants(
                specs, self.base, gpus="4,5,6,7", port=8100
            )

    def test_high_plan_binds_and_excludes_shape_r1_failure(self) -> None:
        specs = self.module.cells_for_suite("high")
        preflight = {
            "matrix_invariants": self.module._matrix_invariants(
                specs, self.base, gpus="4,5,6,7", port=8100
            )
        }
        bindings = self.module._bindings(self.module.FAILED_SHAPE_BOUND_PATHS)
        plan = self.module._plan(
            run_tag="comment3-high-fixture",
            suite="high",
            specs=specs,
            base=self.base,
            python=Path(self.base["PASTE_ENV_PREFIX"]) / "bin/python",
            gpus="4,5,6,7",
            port=8100,
            preflight=preflight,
            bindings=bindings,
        )
        self.assertEqual(
            [cell["label"] for cell in plan["cells"]],
            ["a-c12k-l80", "e-c12k-l80-u093"],
        )
        self.assertEqual(
            [cell["order_index"] for cell in plan["cells"]], [0, 1]
        )
        repair = plan["shape_r1_harness_repair"]
        self.assertEqual(repair["rejected_order_index"], 4)
        self.assertEqual(repair["failed_cell_request_count"], 0)
        self.assertFalse(repair["failed_cell_result_present"])
        absence = repair["failed_cell_absence_checks"]
        self.assertEqual(absence["chat_completion_post_count"], 0)
        self.assertTrue(
            all(
                value
                for key, value in absence.items()
                if key != "chat_completion_post_count"
            )
        )
        self.assertEqual(
            repair["runner_bindings"]["historical_sha256"],
            self.module.FAILED_SHAPE_RUNNER_SHA256,
        )
        self.assertTrue(
            repair["runner_bindings"]["replacement_runner_sha_differs"]
        )
        self.assertEqual(repair["excluded_observed_prefix"]["cell_count"], 4)
        self.assertFalse(
            repair["excluded_observed_prefix"]["reused_by_replacement"]
        )
        self.assertFalse(
            repair["excluded_observed_prefix"]["pooled_with_replacement"]
        )
        self.assertTrue(
            repair["replacement"]["no_further_auto_rerun"]
        )
        self.assertTrue(
            all(
                row["equal_after_identity_normalization"]
                for row in repair["replacement"]["configuration_equivalence"]
            )
        )
        self.assertEqual(len(repair["bound_files"]), 7)
        self.assertTrue(set(repair["bound_files"]).issubset(plan["bindings"]))
        self.assertEqual(
            set(repair["replacement"]["allowed_identity_differences"]),
            {
                "run/output path",
                "formal block id",
                "formal order index",
                "server instance id",
                "server URL/port",
            },
        )

    def test_failed_shape_run_tag_cannot_be_resumed(self) -> None:
        with self.assertRaisesRegex(
            self.module.LiveSensitivityError, "immutable failed evidence"
        ):
            self.module.main(
                ["comment3-shape-r1", "--suite", "high", "--check-only"]
            )

    def test_transport_gate_accepts_only_one_clean_attempt_per_invocation(self) -> None:
        clean = {
            "authoritative": True,
            "speculative": False,
            "committed": True,
            "outcome": "committed",
            "http_attempts": 1,
            "response_status": 200,
            "http_attempt_log": [{"status": 200, "retried": False}],
        }
        result = {
            "tool_attempt_records": [
                dict(clean) for _ in range(self.module.EXPECTED_TOOL_COMMITS)
            ]
        }
        audit = self.module._validate_transport_attempts(result)
        self.assertEqual(audit["http_retry_count"], 0)
        self.assertEqual(audit["http_429_count"], 0)

        result["tool_attempt_records"][7] = {
            **clean,
            "http_attempts": 2,
            "http_attempt_log": [
                {"status": 429, "retried": True},
                {"status": 200, "retried": False},
            ],
        }
        with self.assertRaisesRegex(
            self.module.LiveSensitivityError, "retry-free transport gate"
        ):
            self.module._validate_transport_attempts(result)


if __name__ == "__main__":
    unittest.main()
