from __future__ import annotations

import contextlib
import copy
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_live_joint_formal_v9_matrix as v9  # type: ignore


class KernelIsolationMixin:
    def setUp(self) -> None:
        self._kernel_state = {
            "DEFAULT_CONFIG": v9.formal.DEFAULT_CONFIG,
            "PROTOCOL": v9.formal.PROTOCOL,
            "FORMAL_WORKLOAD_SHA256": v9.formal.FORMAL_WORKLOAD_SHA256,
            "FORMAL_CANONICAL_SHA256": v9.formal.FORMAL_CANONICAL_SHA256,
            "FORMAL_SOURCES_SHA256": v9.formal.FORMAL_SOURCES_SHA256,
            "EXPECTED_CONFIG": v9.formal.EXPECTED_CONFIG,
            "FROZEN_JOINT_SCHEDULER_ENV_KEYS": (
                v9.formal.FROZEN_JOINT_SCHEDULER_ENV_KEYS
            ),
            "validate_cell_result": v9.formal.validate_cell_result,
            "write_json_atomic": v9.formal.write_json_atomic,
            "BOUND_CODE_PATHS": v9.formal.BOUND_CODE_PATHS,
        }

    def tearDown(self) -> None:
        for name, value in self._kernel_state.items():
            setattr(v9.formal, name, value)


class FrozenInputTests(KernelIsolationMixin, unittest.TestCase):
    def test_runtime_and_development_selection_match_exact_shas(self) -> None:
        runtime = v9.validate_frozen_runtime()
        selection = v9.validate_development_selection()
        self.assertEqual(
            runtime["reproduction/paste_repro/live_broker.py"],
            "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27",
        )
        self.assertEqual(
            runtime[
                "reproduction/configs/live_joint_formal_v9_matrix.env.example"
            ],
            "946db6793569d6d9c33215d515318a4fffaf869d8a2def65daa43f2e798c09ac",
        )
        self.assertEqual(selection["selected_policy"], "F0")
        self.assertEqual(selection["selected_visit_interval_s"], 2.5)
        self.assertEqual(selection["selected_min_speculative_tool_workers"], 0)
        self.assertFalse(
            selection["candidate_performance_used_for_transport_selection"]
        )

    def test_config_is_exact_v9_f0_treatment(self) -> None:
        v9._configure_frozen_kernel()
        config = v9.formal.load_frozen_config(v9.DEFAULT_CONFIG)
        self.assertEqual(config, v9.formal.EXPECTED_CONFIG)
        self.assertEqual(config["PASTE_LIVE_MAX_ACTIVE_TASKS"], "80")
        self.assertEqual(config["VLLM_MAX_NUM_SEQS"], "96")
        self.assertEqual(config["PASTE_LIVE_FIXED_FINAL_COMPLETION_TOKENS"], "192")
        self.assertEqual(config["PASTE_LIVE_VISIT_MIN_START_INTERVAL_S"], "2.5")
        self.assertEqual(config["PASTE_LIVE_MIN_SPECULATIVE_TOOL_WORKERS"], "0")
        self.assertEqual(config["PASTE_LIVE_FORMAL_SELECTED_POLICY"], "F0")
        self.assertEqual(
            config["PASTE_LIVE_FORMAL_MAX_OBSERVED_HTTP_RETRIES"], "0"
        )
        self.assertEqual(config["PASTE_LIVE_TOOL_HTTP_MAX_ATTEMPTS"], "2")
        self.assertEqual(
            config["PASTE_LIVE_FORMAL_WORKLOAD_SHA256"],
            v9.FORMAL_WORKLOAD_SHA256,
        )

    def test_all_cell_commands_share_interval_and_minimum(self) -> None:
        v9._configure_frozen_kernel()
        config = v9.formal.load_frozen_config(v9.DEFAULT_CONFIG)
        python = Path(config["PASTE_ENV_PREFIX"]) / "bin/python"
        for cell in ("A", "B", "E", "F"):
            command = v9.formal._runner_command(
                python=python,
                workload=v9.V9_WORKLOAD,
                output=ROOT / "unused",
                cell=cell,
                block_id="formal-v9-test-block-1",
                order_index=0,
                server_instance_id="server-test",
                config=config,
            )
            self.assertEqual(
                command[command.index("--visit-min-start-interval-s") + 1],
                "2.5",
            )
            self.assertEqual(
                command[command.index("--min-speculative-tool-workers") + 1],
                "0",
            )
            expected_speculation = "visit" if cell in {"B", "F"} else "off"
            self.assertEqual(
                command[command.index("--speculation-mode") + 1],
                expected_speculation,
            )

    def test_bound_paths_include_all_three_selection_artifacts(self) -> None:
        v9._configure_frozen_kernel()
        bound = {path.resolve() for path in v9.formal.BOUND_CODE_PATHS}
        self.assertIn(v9.COMPLETED_SCREEN.resolve(), bound)
        self.assertIn(v9.STRICT_DEVELOPMENT_SELECTION.resolve(), bound)
        self.assertIn(v9.SELECTED_TRANSPORT.resolve(), bound)
        self.assertIn(Path(v9.__file__).resolve(), bound)

    def test_cli_forbids_ad_hoc_config_and_order_changes(self) -> None:
        v9._configure_frozen_kernel()
        v9._validate_v9_cli(["registered-run"])
        with self.assertRaisesRegex(v9.FormalV9RunError, "orders"):
            v9._validate_v9_cli(
                ["registered-run", "--orders", "A,B,E,F;B,A,E,F;A,B,F,E"]
            )
        with self.assertRaisesRegex(v9.FormalV9RunError, "config"):
            v9._validate_v9_cli(
                ["registered-run", "--config", "/tmp/copied-v9-config.env"]
            )

    def test_semantic_selection_rejects_wrong_f0_minimum(self) -> None:
        completed = json.loads(v9.COMPLETED_SCREEN.read_text(encoding="utf-8"))
        selection = json.loads(
            v9.STRICT_DEVELOPMENT_SELECTION.read_text(encoding="utf-8")
        )
        transport = json.loads(v9.SELECTED_TRANSPORT.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(selection)
        first_f0 = next(
            key
            for key in mutated["common_code_and_config_identity"]["cells"]
            if key.endswith("/F0")
        )
        mutated["common_code_and_config_identity"]["cells"][first_f0][
            "min_speculative_tool_workers"
        ] = 1

        def load(path: Path, _sha: str):
            if path == v9.COMPLETED_SCREEN:
                return completed
            if path == v9.STRICT_DEVELOPMENT_SELECTION:
                return mutated
            return transport

        with mock.patch.object(v9, "_load_exact_json", side_effect=load):
            with self.assertRaisesRegex(v9.FormalV9RunError, "min-spec=0"):
                v9.validate_development_selection()


def _tool_records(*, gap_s: float = 2.5, retry: bool = False) -> list[dict]:
    records: list[dict] = [
        {
            "tool": "search",
            "http_attempts": 1,
            "http_attempt_log": [
                {
                    "attempt": 1,
                    "retried": False,
                    "started_monotonic_s": 10.0,
                }
            ],
        }
    ]
    for index in range(80):
        attempts = 2 if retry and index == 7 else 1
        log = [
            {
                "attempt": 1,
                "retried": attempts == 2,
                "started_monotonic_s": 100.0 + index * gap_s,
            }
        ]
        if attempts == 2:
            log.append(
                {
                    "attempt": 2,
                    "retried": False,
                    "started_monotonic_s": 101.0 + index * gap_s,
                }
            )
        records.append(
            {"tool": "visit", "http_attempts": attempts, "http_attempt_log": log}
        )
    return records


class V9CellValidationTests(unittest.TestCase):
    def _result(self, **record_options: object) -> dict:
        return {
            "config": {
                "visit_min_start_interval_s": 2.5,
                "tool_http_attempt_min_start_intervals_s": {"visit": 2.5},
                "workload_split_id": v9.FORMAL_SPLIT_ID,
                "workload_file_sha256": v9.FORMAL_WORKLOAD_SHA256,
                "min_speculative_tool_workers": 0,
            },
            "tool_attempt_records": _tool_records(**record_options),
        }

    def test_v9_adapter_checks_original_then_reuses_frozen_validator(self) -> None:
        captured: dict = {}

        def legacy(result, **_kwargs):
            captured.update(result["config"])

        with mock.patch.object(v9, "_LEGACY_VALIDATE_CELL_RESULT", legacy):
            v9.validate_v9_cell_result(
                self._result(),
                cell="F",
                block_id="block-1",
                order_index=3,
                server_instance_id="server-1",
            )
        self.assertEqual(captured["visit_min_start_interval_s"], 2.1)
        self.assertEqual(
            captured["workload_split_id"],
            "live-joint-wikipedia-frozen-formal-v8",
        )

    def test_formal_retry_is_fail_closed_at_zero(self) -> None:
        with mock.patch.object(v9, "_LEGACY_VALIDATE_CELL_RESULT"):
            with self.assertRaisesRegex(v9.FormalV9RunError, "zero-retry"):
                v9.validate_v9_cell_result(
                    self._result(retry=True),
                    cell="F",
                    block_id="block-1",
                    order_index=3,
                    server_instance_id="server-1",
                )

    def test_started_uncommitted_speculation_is_rejected_as_waste(self) -> None:
        result = self._result()
        result["tool_attempt_records"][7].update(
            {"speculative": True, "committed": False}
        )
        with mock.patch.object(v9, "_LEGACY_VALIDATE_CELL_RESULT"):
            with self.assertRaisesRegex(
                v9.FormalV9RunError, "zero-wasted-speculative-service"
            ):
                v9.validate_v9_cell_result(
                    result,
                    cell="F",
                    block_id="block-1",
                    order_index=3,
                    server_instance_id="server-1",
                )

    def test_selected_physical_start_gate_is_2p5_with_20ms_tolerance(self) -> None:
        with mock.patch.object(v9, "_LEGACY_VALIDATE_CELL_RESULT"):
            with self.assertRaisesRegex(v9.FormalV9RunError, "2.5s visit gate"):
                v9.validate_v9_cell_result(
                    self._result(gap_s=2.47),
                    cell="F",
                    block_id="block-1",
                    order_index=3,
                    server_instance_id="server-1",
                )


class MetadataTests(KernelIsolationMixin, unittest.TestCase):
    def test_cell_metadata_gets_exact_selection_provenance_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effective_config.json"
            v9.write_json_atomic_v9(
                path,
                {
                    "schema": "paste_repro.live_joint_formal_cell_config",
                    "version": 1,
                },
            )
            value = json.loads(path.read_text(encoding="utf-8"))
        provenance = value["formal_v9_selection"]
        self.assertEqual(value["formal_generation"], "v9")
        self.assertEqual(provenance["selected_policy"], "F0")
        self.assertEqual(provenance["selected_visit_interval_s"], 2.5)
        self.assertEqual(provenance["selected_min_speculative_tool_workers"], 0)
        self.assertEqual(provenance["maximum_observed_http_retries_per_cell"], 0)
        self.assertEqual(
            provenance["live_broker_sha256"],
            "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27",
        )

    def test_check_only_adds_v9_preflight_without_starting_server(self) -> None:
        base = {
            "valid": True,
            "check_only": True,
            "gpu_or_server_touched": False,
        }

        def kernel(_arguments):
            print(json.dumps(base))
            return 0

        stdout = StringIO()
        with (
            mock.patch.object(v9, "validate_frozen_runtime", return_value={"x": "y"}),
            mock.patch.object(
                v9,
                "validate_development_selection",
                return_value={"valid": True, "selected_policy": "F0"},
            ),
            mock.patch.object(v9.formal, "main", side_effect=kernel),
            contextlib.redirect_stdout(stdout),
        ):
            code = v9.main(["offline-test", "--check-only"])
        value = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(value["schema"], "paste_repro.live_joint_formal_v9_check")
        self.assertTrue(value["formal_workload"]["untouched_by_development_screen"])
        self.assertEqual(value["registered_treatment"]["offered_concurrency"], 80)
        self.assertEqual(
            value["registered_treatment"]["visit_min_start_interval_s"], 2.5
        )


if __name__ == "__main__":
    unittest.main()
