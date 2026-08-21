from __future__ import annotations

import json
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from reproduction.tests.test_summarize_four_cell import _write_json  # noqa: E402
from reproduction.tests.test_summarize_paired_ad import _paired_fixture  # noqa: E402
import summarize_candidate_d as candidate_module  # noqa: E402
from summarize_candidate_d import main, parse_args, summarize_candidate  # noqa: E402


def _candidate_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    manifest, pairs = _paired_fixture(root)
    a_run, reference_d = pairs[0]
    candidate_d = root / "candidate_d"
    shutil.copytree(reference_d, candidate_d)
    summary_path = candidate_d / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["scheduler_environment"][
        "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING"
    ] = "48"
    _write_json(summary_path, summary)
    return manifest, a_run, reference_d, candidate_d


class CandidateDSummaryTests(unittest.TestCase):
    def test_exact_config_allowlist_and_extended_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, a_run, reference_d, candidate_d = _candidate_fixture(
                Path(temporary)
            )
            result = summarize_candidate(
                manifest_path=manifest,
                role="final",
                a_run=a_run,
                reference_d_run=reference_d,
                candidate_d_run=candidate_d,
                allowed_config_differences={
                    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING"
                },
                expected_candidate_config={
                    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING": "48"
                },
                include_natural_queue_evidence=False,
            )

        self.assertEqual(
            result["status"],
            "candidate_screen_reuses_previous_a_not_fresh_server_pair",
        )
        guard = result["comparison_invariants"]["candidate_config_guard"]
        self.assertTrue(guard["exact_allowlist_match"])
        self.assertEqual(
            guard["actual_difference_keys"],
            ["VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING"],
        )
        candidate = result["cells"]["candidate_D"]
        self.assertIn("p99", candidate["task_flow_time_s"])
        self.assertIn("p99", candidate["request_latency_s"])
        self.assertEqual(candidate["request_latency_s"]["count_gt_120_s"], 0)
        self.assertEqual(candidate["request_latency_s"]["count_gt_240_s"], 0)
        decomposition = result["candidate_task_saving_decomposition"]
        self.assertAlmostEqual(
            sum(decomposition["components_s"].values()),
            decomposition["task_mean_saving_s"],
        )
        pairing = result["candidate_source_pairing"]
        self.assertEqual(pairing["independent_source_session_count"], 2)
        self.assertEqual(
            pairing["independent_source_mean_bootstrap_95_ci_s"]["sample_size"],
            2,
        )
        self.assertFalse(result["natural_queue_evidence"]["available"])

    def test_unexpected_or_unused_config_allowlist_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, a_run, reference_d, candidate_d = _candidate_fixture(
                Path(temporary)
            )
            with self.assertRaisesRegex(ValueError, "unexpected=.*DEADLINE"):
                summarize_candidate(
                    manifest_path=manifest,
                    role="final",
                    a_run=a_run,
                    reference_d_run=reference_d,
                    candidate_d_run=candidate_d,
                    allowed_config_differences=set(),
                    include_natural_queue_evidence=False,
                )
            with self.assertRaisesRegex(ValueError, "unused=.*TYPO"):
                summarize_candidate(
                    manifest_path=manifest,
                    role="final",
                    a_run=a_run,
                    reference_d_run=reference_d,
                    candidate_d_run=candidate_d,
                    allowed_config_differences={
                        "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
                        "TYPO",
                    },
                    include_natural_queue_evidence=False,
                )

    def test_expected_config_and_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, a_run, reference_d, candidate_d = _candidate_fixture(
                Path(temporary)
            )
            with self.assertRaisesRegex(ValueError, "expected '49'"):
                summarize_candidate(
                    manifest_path=manifest,
                    role="final",
                    a_run=a_run,
                    reference_d_run=reference_d,
                    candidate_d_run=candidate_d,
                    allowed_config_differences={
                        "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING"
                    },
                    expected_candidate_config={
                        "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING": "49"
                    },
                    include_natural_queue_evidence=False,
                )

        with tempfile.TemporaryDirectory() as temporary:
            manifest, a_run, reference_d, candidate_d = _candidate_fixture(
                Path(temporary)
            )
            events_path = candidate_d / "request_events.jsonl"
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            events[0]["trace_id"] = "different-trace"
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identities do not exactly match"):
                summarize_candidate(
                    manifest_path=manifest,
                    role="final",
                    a_run=a_run,
                    reference_d_run=reference_d,
                    candidate_d_run=candidate_d,
                    allowed_config_differences={
                        "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING"
                    },
                    include_natural_queue_evidence=False,
                )

    def test_cli_parses_explicit_guards(self) -> None:
        parsed = parse_args(
            [
                "--manifest",
                "manifest.json",
                "--a-run",
                "a",
                "--reference-d-run",
                "old-d",
                "--candidate-d-run",
                "new-d",
                "--allow-config-diff",
                "PASTE_FROZEN_CONFIG_SHA256",
                "--expect-candidate-config",
                "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING=48",
                "--require-natural-queue",
                "--output",
                "comparison.json",
            ]
        )
        self.assertEqual(parsed.role, "stress")
        self.assertEqual(
            parsed.allow_config_diff, ["PASTE_FROZEN_CONFIG_SHA256"]
        )
        self.assertTrue(parsed.require_natural_queue)
        self.assertEqual(parsed.output, Path("comparison.json"))

    def test_main_atomically_writes_output_and_preserves_stdout(self) -> None:
        payload = {"schema": "synthetic", "value": 7}
        stdout = io.StringIO()
        argv = [
            "--manifest",
            "manifest.json",
            "--a-run",
            "a",
            "--reference-d-run",
            "old-d",
            "--candidate-d-run",
            "new-d",
            "--output",
            "comparison.json",
        ]
        with mock.patch.object(
            candidate_module, "summarize_candidate", return_value=payload
        ), mock.patch.object(
            candidate_module, "write_json_atomic"
        ) as atomic_write, mock.patch.object(sys, "stdout", stdout):
            self.assertEqual(main(argv), 0)

        atomic_write.assert_called_once_with(Path("comparison.json"), payload)
        self.assertEqual(json.loads(stdout.getvalue()), payload)


if __name__ == "__main__":
    unittest.main()
