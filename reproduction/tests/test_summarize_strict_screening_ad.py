from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from reproduction.tests.test_summarize_paired_ad import _paired_fixture  # noqa: E402
from summarize_strict_screening_ad import (  # noqa: E402
    _exact_config_guard,
    summarize_strict_screening,
)


class StrictScreeningADTests(unittest.TestCase):
    def test_exact_guard_requires_the_actual_diff_and_two_sided_expectations(
        self,
    ) -> None:
        result = _exact_config_guard(
            {"ENGINE": "same", "PROFILE": "a"},
            {"ENGINE": "same", "PROFILE": "d", "SOFT": "0"},
            allowed_differences={"PROFILE", "SOFT"},
            expected_a={"PROFILE": "a"},
            expected_d={"PROFILE": "d", "SOFT": "0"},
            expected_a_missing={"SOFT"},
            expected_d_missing=set(),
        )
        self.assertTrue(result["exact_allowlist_match"])
        self.assertEqual(result["actual_difference_keys"], ["PROFILE", "SOFT"])

        with self.assertRaisesRegex(ValueError, "unexpected=.*ENGINE"):
            _exact_config_guard(
                {"ENGINE": "a", "PROFILE": "a"},
                {"ENGINE": "d", "PROFILE": "d", "SOFT": "0"},
                allowed_differences={"PROFILE", "SOFT"},
                expected_a={"PROFILE": "a"},
                expected_d={"PROFILE": "d", "SOFT": "0"},
                expected_a_missing={"SOFT"},
                expected_d_missing=set(),
            )
        with self.assertRaisesRegex(ValueError, "exact A expectation"):
            _exact_config_guard(
                {"PROFILE": "a"},
                {"PROFILE": "d"},
                allowed_differences={"PROFILE"},
                expected_a={},
                expected_d={"PROFILE": "d"},
                expected_a_missing=set(),
                expected_d_missing=set(),
            )

    def test_synthetic_pair_is_identity_folded_and_reports_requested_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            a_run, d_run = pairs[0]
            result = summarize_strict_screening(
                manifest_path=manifest,
                role="final",
                a_run=a_run,
                d_run=d_run,
                allowed_config_differences=set(),
                expected_a_config={},
                expected_d_config={},
                expected_a_config_missing=set(),
                expected_d_config_missing=set(),
                expected_a_policy="fcfs",
                expected_d_policy="online_joint_pacer_v2",
                expected_a_overlap="none",
                expected_d_overlap="learned",
                required_engine_keys=(
                    "VLLM_MAX_NUM_SEQS",
                    "VLLM_SCHED_PRED_OUT_ENABLE",
                ),
                include_natural_queue_evidence=False,
                require_natural_queue=False,
                verify_frozen_configs=False,
            )

        self.assertTrue(
            result["comparison_invariants"]["request_identity_exact_match"]
        )
        self.assertTrue(
            result["comparison_invariants"]["engine_shape_guard"]
            ["all_required_keys_present_and_identical"]
        )
        self.assertIn("p99", result["cells"]["A"]["task_flow_time_s"])
        self.assertIn("p99", result["cells"]["D"]["request_latency_s"])
        self.assertIn("execution", result["comparison"])
        self.assertIn("queue", result["task_saving_decomposition"]["components_s"])
        self.assertEqual(result["source_pairing"]["load_instance_count"], 2)
        self.assertEqual(
            result["source_pairing"]["independent_source_session_count"], 2
        )
        self.assertEqual(
            result["source_pairing"]
            ["independent_source_mean_bootstrap_95_ci_s"]["sample_size"],
            2,
        )

    def test_unwhitelisted_engine_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            a_run, d_run = pairs[0]
            summary_path = d_run / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scheduler_environment"]["VLLM_MAX_NUM_SEQS"] = "9"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unexpected=.*MAX_NUM_SEQS"):
                summarize_strict_screening(
                    manifest_path=manifest,
                    role="final",
                    a_run=a_run,
                    d_run=d_run,
                    allowed_config_differences=set(),
                    expected_a_config={},
                    expected_d_config={},
                    expected_a_config_missing=set(),
                    expected_d_config_missing=set(),
                    expected_a_policy="fcfs",
                    expected_d_policy="online_joint_pacer_v2",
                    expected_a_overlap="none",
                    expected_d_overlap="learned",
                    required_engine_keys=("VLLM_MAX_NUM_SEQS",),
                    include_natural_queue_evidence=False,
                    require_natural_queue=False,
                    verify_frozen_configs=False,
                )


if __name__ == "__main__":
    unittest.main()
