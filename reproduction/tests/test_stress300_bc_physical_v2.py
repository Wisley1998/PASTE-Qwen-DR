from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from summarize_strict_screening_bc import ALLOWED_CONFIG_DIFFERENCES  # noqa: E402
from summarize_strict_screening_bc_physical_v2 import (  # noqa: E402
    B_CONFIG_SHA256,
    C_CONFIG_SHA256,
    EXPECTED_B_CONFIG,
    EXPECTED_B_MISSING,
    EXPECTED_C_CONFIG,
    _result_boundaries,
)
from validate_native_admission_zero_write_v2 import (  # noqa: E402
    _parse_joint_cap_samples,
)


B_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/"
    "joint_stress300_u86_native320_native_exact_rescue120_b_screen.env.example"
)
C_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/"
    "joint_stress300_u86_native320_physical093_exact_rescue120.env.example"
)
C_SUMMARY = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/stress300_u86_native320_g256_physical093_"
    "exact_rescue120/stress300_c_physical093_r1/"
    "stress300_c_physical093_r1_joint_learned/summary.json"
)
B_RUN_ROOT = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/stress300_u86_native320_g256_native_"
    "exact_rescue120_b_screen/stress300_b_native_r1"
)
B_NATIVE_VALIDATION = B_RUN_ROOT / "native_admission_zero_write_v2.json"
STRICT_BC_RESULT = B_RUN_ROOT / "strict_b_vs_c_physical_v2.json"
EXPORT_PATTERN = re.compile(r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exports(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = EXPORT_PATTERN.fullmatch(line)
        if match is not None:
            result[match.group(1)] = match.group(2)
    return result


def _screen(*, benefit: bool = True, tail: bool = True) -> dict:
    b_mean = 100.0
    c_mean = 80.0 if benefit else 105.0
    b_p95 = 130.0
    c_p95 = 120.0 if tail else 140.0
    source_faster = 40 if benefit else 20
    ci_lower = 5.0 if benefit else -5.0
    return {
        "cells": {
            "B": {
                "task_flow_time_s": {"mean": b_mean, "p95": b_p95},
                "task_makespan_s": 150.0,
                "request_latency_s": {"p99": 20.0, "count_gt_240_s": 0},
            },
            "C": {
                "task_flow_time_s": {"mean": c_mean, "p95": c_p95},
                "task_makespan_s": 149.0,
                "request_latency_s": {"p99": 25.0, "count_gt_240_s": 0},
            },
        },
        "comparison": {
            "task_flow_time_s": {
                "mean": {"relative_reduction": (b_mean - c_mean) / b_mean}
            },
            "execution": {
                "completion_tokens": {"c_relative_to_b": 0.001}
            },
        },
        "source_pairing": {
            "independent_source_session_count": 60,
            "source_session_outcomes": {"c_faster": source_faster},
            "independent_source_mean_bootstrap_95_ci_s": {"lower_s": ci_lower},
        },
    }


class Stress300BCPhysicalV2Tests(unittest.TestCase):
    def test_frozen_b_config_and_exact_runtime_scheduler_delta(self) -> None:
        self.assertEqual(_sha(B_CONFIG), B_CONFIG_SHA256)
        self.assertEqual(_sha(C_CONFIG), C_CONFIG_SHA256)
        b_exports = _exports(B_CONFIG)
        c_environment = json.loads(C_SUMMARY.read_text(encoding="utf-8"))[
            "scheduler_environment"
        ]
        synthesized_b = dict(c_environment)
        synthesized_b.update(EXPECTED_B_CONFIG)
        for key in EXPECTED_B_MISSING:
            synthesized_b.pop(key, None)
        differences = {
            key
            for key in set(synthesized_b) | set(c_environment)
            if synthesized_b.get(key) != c_environment.get(key)
        }
        self.assertEqual(differences, set(ALLOWED_CONFIG_DIFFERENCES))
        self.assertEqual(set(EXPECTED_B_MISSING), {
            key for key in ALLOWED_CONFIG_DIFFERENCES if "PHYSICAL_KV_" in key
        })
        self.assertEqual(b_exports["VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION"], "1")
        self.assertEqual(b_exports["VLLM_MAX_NUM_SEQS"], "320")
        self.assertEqual(b_exports["PASTE_MAX_ACTIVE_TRACES"], "300")
        self.assertEqual(b_exports["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"], "60")
        for key in EXPECTED_B_MISSING:
            self.assertNotIn(key, b_exports)
        for key, value in EXPECTED_C_CONFIG.items():
            self.assertEqual(c_environment.get(key), value)

    def test_joint_cap_parser_requires_exact_schema(self) -> None:
        valid = (
            "prefix [sched_policy_patch:joint] pending_returns=1 reserved_kv=2 "
            "reserved_slots=0 running=99 cap=320 window_s=5.0\n"
        )
        self.assertEqual(
            _parse_joint_cap_samples(valid),
            [{
                "pending_returns": 1,
                "reserved_kv": 2,
                "reserved_slots": 0,
                "running": 99,
                "cap": 320,
                "window_s": 5.0,
            }],
        )
        with self.assertRaisesRegex(ValueError, "duplicate, missing, unknown"):
            _parse_joint_cap_samples(valid.replace(" cap=320", " cap=64 extra=1"))
        with self.assertRaisesRegex(ValueError, "multiple markers"):
            _parse_joint_cap_samples(valid.rstrip() + " " + valid)

    def test_completed_b_screen_evidence_is_frozen_and_promoted(self) -> None:
        self.assertEqual(
            _sha(B_NATIVE_VALIDATION),
            "6138577e44a5eba666877fdd4be4e3e409d8840f5aa5cfdcf4975b853f278977",
        )
        self.assertEqual(
            _sha(STRICT_BC_RESULT),
            "8e9db08d1fa2558ff3fe2a5d8a4de4988ae059470bc375e6dbced1e60a686d4b",
        )
        result = json.loads(STRICT_BC_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(result["schema"], "paste_repro.strict_screening_bc_physical_v2")
        self.assertEqual(result["status"], "valid_incremental_single_screen")
        self.assertEqual(
            result["result_boundaries"]["classification"],
            "accepted_incremental_physical_admission_benefit",
        )
        self.assertTrue(result["result_boundaries"]["promotion_passed"])
        admission = result["admission_evidence"]
        self.assertEqual(admission["exact_scheduler_configuration_difference_count"], 7)
        self.assertEqual(admission["B_physical_controller_capacity_write_count"], 0)
        self.assertEqual(admission["C_capacity_write_source"], "physical_kv")
        self.assertEqual(admission["C_dynamic_cap_min"], 2)
        self.assertEqual(admission["C_dynamic_cap_max"], 300)
        self.assertEqual(admission["C_pressure_above_64_sample_count"], 1950)
        native = admission["B_native_reorder_only"]["native_admission"]
        self.assertEqual(native["joint_cap_observed_unique_count"], 1)
        self.assertEqual(native["joint_cap_observed_min"], 320)
        self.assertEqual(native["joint_cap_observed_max"], 320)
        self.assertEqual(native["physical_capacity_write_count"], 0)
        self.assertEqual(native["vllm_stats_max_running"], 300)
        self.assertGreater(native["vllm_stats_max_waiting"], 0)

    def test_result_boundaries_promote_only_when_tail_and_benefit_pass(self) -> None:
        passed = _result_boundaries(_screen())
        self.assertTrue(passed["promotion_passed"])
        self.assertEqual(
            passed["classification"],
            "accepted_incremental_physical_admission_benefit",
        )

        no_benefit = _result_boundaries(_screen(benefit=False))
        self.assertFalse(no_benefit["promotion_passed"])
        self.assertFalse(
            no_benefit["incremental_physical_admission_benefit"]["passed"]
        )
        self.assertEqual(no_benefit["classification"], "valid_screen_not_promoted")

        bad_tail = _result_boundaries(_screen(tail=False))
        self.assertFalse(bad_tail["promotion_passed"])
        self.assertFalse(bad_tail["comparability_and_tail"]["passed"])


if __name__ == "__main__":
    unittest.main()
