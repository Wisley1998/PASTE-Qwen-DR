from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_ROOT = REPRODUCTION_ROOT / "scripts"
for import_root in (REPRODUCTION_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_stress_duplicate_workloads import build_stress_bundle  # noqa: E402
from reproduction.tests.test_build_stress_duplicate_workloads import (  # noqa: E402
    _build_heldout_fixture,
)


WRAPPER = SCRIPT_ROOT / "run_joint_stress_pair.sh"
FOUR_CELL = SCRIPT_ROOT / "run_four_cell.sh"
BASE_CONFIG = REPRODUCTION_ROOT / "configs" / "joint_stress.env.example"
NATIVE256_CONFIG = (
    REPRODUCTION_ROOT / "configs" / "joint_stress180_u86_native256.env.example"
)
EXACT_RESCUE120_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress180_u86_native256_exact_rescue120.env.example"
)
SOFT4_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress180_u86_native256_soft4.env.example"
)
STRESS240_A_PROBE_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress240_u86_native256_a_probe.env.example"
)
STRESS240_D_SCREEN_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress240_u86_native256_exact_rescue120.env.example"
)
STRESS300_A_PROBE_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress300_u86_native320_a_probe.env.example"
)
STRESS300_KEEPALIVE60_A_PROBE_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress300_u86_native320_keepalive60_a_probe.env.example"
)
STRESS300_PHYSICAL093_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress300_u86_native320_physical093_exact_rescue120.env.example"
)
STRESS300_NATIVE_B_SCREEN_CONFIG = (
    REPRODUCTION_ROOT
    / "configs"
    / "joint_stress300_u86_native320_native_exact_rescue120_b_screen.env.example"
)
STRESS300_REFERENCE_C_CELL = (
    REPRODUCTION_ROOT
    / "artifacts/stress300_u86_native320_g256_physical093_exact_rescue120"
    / "stress300_c_physical093_r1"
    / "stress300_c_physical093_r1_joint_learned"
)
STRESS300_ACCEPTED_A_R3_PROBE = (
    REPRODUCTION_ROOT
    / "artifacts"
    / "stress300_u86_native320_g256_keepalive60_a_probe"
    / "stress300_a_probe_r3"
    / "natural_queue_probe.json"
)
SCHEDULER_PATCH = REPOSITORY_ROOT / "scripts" / "pythonhooks" / "sched_policy_patch.py"
TRACE_RUNNER = REPOSITORY_ROOT / "scripts" / "run_vllm_trace_experiment.py"
NATURAL_QUEUE_SUMMARIZER = SCRIPT_ROOT / "summarize_natural_queue_probe.py"

from reproduction.tests.test_validate_accepted_a_probe import (  # noqa: E402
    _write_accepted_probe_fixture,
)


class JointStressPairScriptTests(unittest.TestCase):
    def test_wrapper_uses_python_for_nonexecutable_python_helpers(self) -> None:
        wrapper_lines = WRAPPER.read_text(encoding="utf-8").splitlines()
        helper_lines = [
            line
            for line in wrapper_lines
            if '"${SCRIPT_DIR}/' in line and '.py"' in line
        ]
        self.assertTrue(helper_lines)
        for line in helper_lines:
            self.assertIn('"${PYTHON_BIN}"', line)

        # This is the real deployment mode that exposed the regression: the
        # summarizer intentionally has no executable bit, but succeeds when
        # launched through the configured Python interpreter.
        self.assertEqual(NATURAL_QUEUE_SUMMARIZER.stat().st_mode & 0o111, 0)
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            probe = _write_accepted_probe_fixture(Path(temporary))
            cell = probe.parent / "accepted_a_fcfs_none"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(NATURAL_QUEUE_SUMMARIZER),
                    str(cell),
                    "--require-natural-queue",
                    "--require-exactly-once",
                    "--require-no-kv-swap",
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_formal_runtime_configuration_fields_are_recorded(self) -> None:
        tree = ast.parse(TRACE_RUNNER.read_text(encoding="utf-8"))
        recorded: set[str] | None = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == "_SCHEDULER_ENV_KEYS"
                for target in node.targets
            ):
                recorded = set(ast.literal_eval(node.value))
                break
        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertTrue(
            {
                "PASTE_STRESS_PROFILE",
                "PASTE_FROZEN_CONFIG_SHA256",
                "MODEL_ID",
                "MODEL_REVISION",
                "CUDA_VISIBLE_DEVICES",
                "VLLM_TP_SIZE",
                "VLLM_DTYPE",
                "VLLM_MAX_MODEL_LEN",
                "VLLM_GPU_MEMORY_UTILIZATION",
                "VLLM_MAX_NUM_BATCHED_TOKENS",
                "VLLM_MAX_NUM_SEQS",
                "VLLM_CUDA_GRAPH_SIZES",
                "VLLM_USE_V1",
                "VLLM_SCHED_PRED_OUT_ENABLE",
                "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S",
                "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
                "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
                "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
                "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
                "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY",
                "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S",
                "VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S",
                "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA",
                "VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS",
                "VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S",
                "VLLM_SCHED_HBM_LOW_PRESSURE",
                "VLLM_SCHED_HBM_HIGH_PRESSURE",
                "VLLM_SCHED_HBM_BUDGET_INCREASE",
                "VLLM_SCHED_HBM_BUDGET_DECREASE",
                "VLLM_SCHED_HBM_CONTROL_INTERVAL_S",
                "VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO",
            }.issubset(recorded)
        )

    def test_native_profile_freezes_context_cost_throughput(self) -> None:
        config_text = NATIVE256_CONFIG.read_text(encoding="utf-8")
        wrapper_text = WRAPPER.read_text(encoding="utf-8")
        patch_text = SCHEDULER_PATCH.read_text(encoding="utf-8")
        name = "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S"

        self.assertIn(f'export {name}="6000"', config_text)
        self.assertIn(f'require_exact {name} "6000"', wrapper_text)
        self.assertRegex(
            patch_text,
            rf'os\.getenv\("{name}", "6000"\)',
        )

    def test_native_profile_explicitly_opts_into_coarse_remaining_call_lanes(
        self,
    ) -> None:
        config_text = NATIVE256_CONFIG.read_text(encoding="utf-8")
        wrapper_text = WRAPPER.read_text(encoding="utf-8")
        patch_text = SCHEDULER_PATCH.read_text(encoding="utf-8")
        name = "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES"

        self.assertIn(f'export {name}="1"', config_text)
        self.assertIn(f'require_exact {name} "1"', wrapper_text)
        # The experiment profile opts in explicitly; unrelated profiles keep
        # the implementation's backward-compatible, disabled default.
        self.assertRegex(
            patch_text,
            rf'"{name}", "0"',
        )

    def test_soft4_profile_is_independent_and_exactly_selects_soft_stage(self) -> None:
        coarse_exports = dict(
            re.findall(
                r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                NATIVE256_CONFIG.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        )
        soft_exports = dict(
            re.findall(
                r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                SOFT4_CONFIG.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        )

        self.assertNotEqual(SOFT4_CONFIG, NATIVE256_CONFIG)
        self.assertEqual(
            soft_exports["PASTE_STRESS_PROFILE"],
            "stress180_native256_g256_u86_soft4",
        )
        for name in (
            "PASTE_MAX_ACTIVE_TRACES",
            "VLLM_MAX_NUM_SEQS",
            "VLLM_CUDA_GRAPH_SIZES",
            "VLLM_GPU_MEMORY_UTILIZATION",
            "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
            "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
            "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
            "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY",
        ):
            self.assertEqual(soft_exports[name], coarse_exports[name], name)
        self.assertEqual(
            soft_exports["VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE"], "0"
        )
        self.assertEqual(
            soft_exports["VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES"],
            "0",
        )
        self.assertEqual(
            soft_exports["VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S"],
            "4.0",
        )

    def test_exact_rescue120_profile_changes_only_explicit_screening_controls(
        self,
    ) -> None:
        native_exports = dict(
            re.findall(
                r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                NATIVE256_CONFIG.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        )
        rescue_exports = dict(
            re.findall(
                r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                EXACT_RESCUE120_CONFIG.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        )
        allowed_differences = {
            "PASTE_STRESS_PROFILE",
            "PASTE_STRESS_RUN_BASE",
            "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
            "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
            "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
        }
        actual_differences = {
            name
            for name in set(native_exports) | set(rescue_exports)
            if native_exports.get(name) != rescue_exports.get(name)
        }
        self.assertEqual(actual_differences, allowed_differences)
        self.assertEqual(
            rescue_exports["PASTE_STRESS_PROFILE"],
            "stress180_native256_g256_u86_exact_rescue120",
        )
        self.assertEqual(
            rescue_exports["VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S"], "120"
        )
        self.assertEqual(
            rescue_exports["VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING"], "48"
        )
        self.assertEqual(rescue_exports["VLLM_SCHED_JOINT_V2_FINAL_LANE"], "1")
        self.assertEqual(
            rescue_exports["VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE"], "1"
        )
        self.assertEqual(
            rescue_exports["VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES"],
            "0",
        )
        self.assertEqual(
            rescue_exports["VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S"],
            "0",
        )
        self.assertEqual(
            rescue_exports["VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION"], "1"
        )
        self.assertEqual(
            rescue_exports["VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY"], "0"
        )

    def test_stress240_probe_has_structural_headroom_and_frozen_a_gates(self) -> None:
        exports = dict(
            re.findall(
                r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                STRESS240_A_PROBE_CONFIG.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(exports["PASTE_MAX_ACTIVE_TRACES"], "240")
        self.assertEqual(exports["VLLM_MAX_NUM_SEQS"], "256")
        self.assertGreater(
            int(exports["VLLM_MAX_NUM_SEQS"]),
            int(exports["PASTE_MAX_ACTIVE_TRACES"]),
        )
        self.assertEqual(exports["VLLM_CUDA_GRAPH_SIZES"], "256")
        self.assertEqual(exports["VLLM_GPU_MEMORY_UTILIZATION"], "0.86")
        self.assertEqual(exports["VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION"], "1")
        self.assertEqual(
            exports["PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION"],
            "0.50",
        )
        self.assertEqual(
            exports["PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION"],
            "0.20",
        )
        self.assertEqual(
            exports["PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST"],
            "0.25",
        )

    def test_stress300_probe_has_headroom_graph256_and_frozen_safety_gates(
        self,
    ) -> None:
        exports = dict(
            re.findall(
                r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                STRESS300_A_PROBE_CONFIG.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(
            exports["PASTE_STRESS_PROFILE"],
            "stress300_native320_g256_u86_a_probe",
        )
        self.assertEqual(exports["PASTE_MAX_ACTIVE_TRACES"], "300")
        self.assertEqual(exports["VLLM_MAX_NUM_SEQS"], "320")
        self.assertEqual(
            int(exports["VLLM_MAX_NUM_SEQS"])
            - int(exports["PASTE_MAX_ACTIVE_TRACES"]),
            20,
        )
        self.assertEqual(exports["VLLM_CUDA_GRAPH_SIZES"], "256")
        self.assertEqual(exports["VLLM_GPU_MEMORY_UTILIZATION"], "0.86")
        self.assertEqual(exports["VLLM_MAX_NUM_BATCHED_TOKENS"], "8192")
        self.assertEqual(exports["VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION"], "1")
        self.assertEqual(
            exports["VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS"], "320"
        )
        self.assertEqual(
            exports["VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING"], "320"
        )
        self.assertEqual(
            exports["VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING"], "320"
        )
        self.assertEqual(
            exports["PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION"],
            "0.50",
        )
        self.assertEqual(
            exports["PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION"], "0.20"
        )
        self.assertEqual(
            exports["PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST"],
            "0.25",
        )

    def test_stress300_keepalive60_probe_changes_only_transport_identity(self) -> None:
        def exports(path: Path) -> dict[str, str]:
            return dict(
                re.findall(
                    r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                    path.read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                )
            )

        original = exports(STRESS300_A_PROBE_CONFIG)
        keepalive = exports(STRESS300_KEEPALIVE60_A_PROBE_CONFIG)
        differences = {
            name
            for name in set(original) | set(keepalive)
            if original.get(name) != keepalive.get(name)
        }

        self.assertEqual(
            differences,
            {
                "PASTE_STRESS_PROFILE",
                "PASTE_STRESS_RUN_BASE",
                "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
            },
        )
        self.assertEqual(
            keepalive["PASTE_STRESS_PROFILE"],
            "stress300_native320_g256_u86_keepalive60_a_probe",
        )
        self.assertEqual(keepalive["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"], "60")
        self.assertEqual(keepalive["PASTE_MAX_ACTIVE_TRACES"], "300")
        self.assertEqual(keepalive["VLLM_MAX_NUM_SEQS"], "320")
        self.assertEqual(keepalive["VLLM_CUDA_GRAPH_SIZES"], "256")

    def test_stress300_physical093_config_is_exact_frozen_a_to_c_delta(self) -> None:
        def exports(path: Path) -> dict[str, str]:
            return dict(
                re.findall(
                    r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                    path.read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                )
            )

        a_values = exports(STRESS300_KEEPALIVE60_A_PROBE_CONFIG)
        c_values = exports(STRESS300_PHYSICAL093_CONFIG)
        differences = {
            name
            for name in set(a_values) | set(c_values)
            if a_values.get(name) != c_values.get(name)
        }
        self.assertEqual(
            differences,
            {
                "PASTE_STRESS_PROFILE",
                "PASTE_STRESS_RUN_BASE",
                "PASTE_ACCEPTED_A_PROBE",
                "PASTE_ACCEPTED_A_PROBE_SHA256",
                "PASTE_ACCEPTED_A_CONFIG_SHA256",
                "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
                "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
                "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
            },
        )
        self.assertEqual(c_values["PASTE_MAX_ACTIVE_TRACES"], "300")
        self.assertEqual(c_values["VLLM_MAX_NUM_SEQS"], "320")
        self.assertEqual(c_values["VLLM_CUDA_GRAPH_SIZES"], "256")
        self.assertEqual(c_values["VLLM_MAX_NUM_BATCHED_TOKENS"], "8192")
        self.assertEqual(c_values["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"], "60")
        self.assertEqual(c_values["VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S"], "120")
        self.assertEqual(
            c_values["VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES"], "0"
        )
        self.assertEqual(c_values["VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION"], "0")
        self.assertEqual(c_values["VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION"], "1")
        self.assertEqual(
            c_values["VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"],
            "0.93",
        )
        self.assertEqual(
            c_values["VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S"], "120"
        )
        self.assertEqual(
            hashlib.sha256(STRESS300_PHYSICAL093_CONFIG.read_bytes()).hexdigest(),
            "1ee7dfe9f5831223fb4ff14c1e86154827d32d7835d11b2749c8e07863321d43",
        )

    def test_stress300_physical093_check_only_and_fail_fast_cli_binding(self) -> None:
        environment = os.environ.copy()
        environment["PASTE_ENV_PREFIX"] = str(Path(sys.executable).resolve().parents[1])
        environment["PASTE_ACCEPTED_A_PROBE"] = (
            STRESS300_ACCEPTED_A_R3_PROBE.relative_to(REPOSITORY_ROOT).as_posix()
        )
        run_tag = "cpu_stress300_physical093_check"
        run_root = (
            REPRODUCTION_ROOT
            / "artifacts/stress300_u86_native320_g256_physical093_exact_rescue120"
            / run_tag
        )
        self.assertFalse(run_root.exists())
        completed = self._run(
            run_tag,
            "--config",
            str(STRESS300_PHYSICAL093_CONFIG),
            "--cells",
            "D",
            "--gpus",
            "4,5,6,7",
            "--port",
            "8100",
            "--check-only",
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("physical093 exact-rescue120", completed.stdout)
        self.assertIn("HTTP keep-alive: 60s", completed.stdout)
        self.assertIn("A SHA:   c2a5b098", completed.stdout)
        self.assertIn("no output was created", completed.stdout)
        self.assertFalse(run_root.exists())

        for option, value, expected in (
            ("--gpus", "0,1,2,3", "must match accepted A-r3 GPUs"),
            ("--port", "8000", "must match accepted A-r3 port"),
            ("--cells", "A", "requires --cells D"),
            ("--cells", "A,D", "requires --cells D"),
            ("--cells", "D,A", "requires --cells D"),
        ):
            with self.subTest(option=option, value=value):
                args = [
                    f"reject_{option[2:]}_{value.replace(',', '_')}",
                    "--config",
                    str(STRESS300_PHYSICAL093_CONFIG),
                    "--cells",
                    "D",
                    "--gpus",
                    "4,5,6,7",
                    "--port",
                    "8100",
                    "--check-only",
                ]
                position = args.index(option)
                args[position + 1] = value
                rejected = self._run(*args, env=environment)
                self.assertEqual(rejected.returncode, 2, rejected.stdout)
                self.assertIn(expected, rejected.stdout)

        wrapper = WRAPPER.read_text(encoding="utf-8")
        for frozen_argument in (
            "--expected-requests 2595",
            "--expected-num-gpu-blocks 44178",
            "--expected-block-size 16",
            "--expected-target-utilization 0.93",
            "--expected-keepalive-s 60",
            "--expected-preemptions 0",
            "validate_physical_kv_admission_v2.py",
            "summarize_strict_screening_ac_physical_v2.py",
        ):
            self.assertIn(frozen_argument, wrapper)

    def test_stress300_native_b_screen_check_only_and_fail_closed_inputs(self) -> None:
        environment = os.environ.copy()
        environment["PASTE_ENV_PREFIX"] = str(Path(sys.executable).resolve().parents[1])
        environment["PASTE_REFERENCE_C_RUN"] = (
            STRESS300_REFERENCE_C_CELL.relative_to(REPOSITORY_ROOT).as_posix()
        )
        run_tag = "cpu_stress300_native_b_screen_check"
        run_root = (
            REPRODUCTION_ROOT
            / "artifacts/stress300_u86_native320_g256_native_exact_rescue120_b_screen"
            / run_tag
        )
        self.assertFalse(run_root.exists())
        completed = self._run(
            run_tag,
            "--config",
            str(STRESS300_NATIVE_B_SCREEN_CONFIG),
            "--cells",
            "D",
            "--gpus",
            "4,5,6,7",
            "--port",
            "8100",
            "--check-only",
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("native reorder-only", completed.stdout)
        self.assertIn("C proof: b292c04f", completed.stdout)
        self.assertIn("no output was created", completed.stdout)
        self.assertFalse(run_root.exists())
        self.assertFalse(run_root.with_name(f"{run_root.name}.lock").exists())

        for option, value, expected in (
            ("--gpus", "0,1,2,3", "must match completed C GPUs"),
            ("--port", "8000", "must match completed C port"),
            ("--cells", "A", "requires --cells D"),
            ("--cells", "A,D", "requires --cells D"),
            ("--cells", "D,A", "requires --cells D"),
        ):
            with self.subTest(option=option, value=value):
                args = [
                    f"reject_native_b_{option[2:]}_{value.replace(',', '_')}",
                    "--config",
                    str(STRESS300_NATIVE_B_SCREEN_CONFIG),
                    "--cells",
                    "D",
                    "--gpus",
                    "4,5,6,7",
                    "--port",
                    "8100",
                    "--check-only",
                ]
                position = args.index(option)
                args[position + 1] = value
                rejected = self._run(*args, env=environment)
                self.assertEqual(rejected.returncode, 2, rejected.stdout)
                self.assertIn(expected, rejected.stdout)

        for physical_key in (
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
        ):
            for inherited_value in ("", "1"):
                with self.subTest(
                    physical_key=physical_key, inherited_value=inherited_value
                ):
                    polluted = dict(environment)
                    polluted[physical_key] = inherited_value
                    rejected = self._run(
                        f"reject_native_b_inherited_{physical_key.lower()}_{len(inherited_value)}",
                        "--config",
                        str(STRESS300_NATIVE_B_SCREEN_CONFIG),
                        "--cells",
                        "D",
                        "--gpus",
                        "4,5,6,7",
                        "--port",
                        "8100",
                        "--check-only",
                        env=polluted,
                    )
                    self.assertEqual(rejected.returncode, 2, rejected.stdout)
                    self.assertIn(f"{physical_key} must be absent", rejected.stdout)

        wrong_reference = dict(environment)
        wrong_reference["PASTE_REFERENCE_C_RUN"] = (
            STRESS300_ACCEPTED_A_R3_PROBE.relative_to(REPOSITORY_ROOT).as_posix()
        )
        rejected = self._run(
            "reject_native_b_wrong_c",
            "--config",
            str(STRESS300_NATIVE_B_SCREEN_CONFIG),
            "--cells",
            "D",
            "--gpus",
            "4,5,6,7",
            "--port",
            "8100",
            "--check-only",
            env=wrong_reference,
        )
        self.assertEqual(rejected.returncode, 2, rejected.stdout)
        self.assertIn("must point to the frozen completed stress300 C", rejected.stdout)

        self.assertEqual(
            hashlib.sha256(STRESS300_NATIVE_B_SCREEN_CONFIG.read_bytes()).hexdigest(),
            "e024ab17e6b08c1c1cd3246e4b74b253b681af152138af762bc536f7b513908e",
        )
        wrapper = WRAPPER.read_text(encoding="utf-8")
        for required in (
            "validate_native_admission_zero_write_v2.py",
            "summarize_strict_screening_bc_physical_v2.py",
            "strict_b_vs_c_physical_v2.json",
        ):
            self.assertIn(required, wrapper)

    def test_stress240_d_screen_matches_a_engine_and_selected_rescue_policy(
        self,
    ) -> None:
        def exports(path: Path) -> dict[str, str]:
            return dict(
                re.findall(
                    r'^export ([A-Z][A-Z0-9_]*)="([^"$]*)"$',
                    path.read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                )
            )

        a_values = exports(STRESS240_A_PROBE_CONFIG)
        d_values = exports(STRESS240_D_SCREEN_CONFIG)
        rescue_values = exports(EXACT_RESCUE120_CONFIG)
        for name in (
            "MODEL_ID",
            "MODEL_REVISION",
            "VLLM_TP_SIZE",
            "VLLM_DTYPE",
            "VLLM_MAX_MODEL_LEN",
            "VLLM_GPU_MEMORY_UTILIZATION",
            "VLLM_MAX_NUM_BATCHED_TOKENS",
            "VLLM_MAX_NUM_SEQS",
            "VLLM_CUDA_GRAPH_SIZES",
            "VLLM_USE_V1",
        ):
            self.assertEqual(d_values[name], a_values[name], name)
        for name in (
            "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
            "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
            "VLLM_SCHED_JOINT_V2_FINAL_LANE",
            "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE",
            "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
            "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
            "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY",
            "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S",
            "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
        ):
            self.assertEqual(d_values[name], rescue_values[name], name)
        self.assertEqual(d_values["PASTE_MAX_ACTIVE_TRACES"], "240")
        self.assertEqual(d_values["VLLM_MAX_NUM_SEQS"], "256")
        self.assertEqual(d_values["VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S"], "120")
        self.assertEqual(d_values["VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING"], "48")
        allowed_rescue_differences = {
            "PASTE_STRESS_PROFILE",
            "PASTE_FIXED_WORKLOAD_MANIFEST",
            "PASTE_STRESS_RUN_BASE",
            "PASTE_MAX_ACTIVE_TRACES",
            "PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION",
            "PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION",
            "PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST",
        }
        actual_rescue_differences = {
            name
            for name in set(d_values) | set(rescue_values)
            if d_values.get(name) != rescue_values.get(name)
        }
        self.assertEqual(
            actual_rescue_differences,
            allowed_rescue_differences,
        )

    def test_frozen_predictor_values_and_explicit_defaults_match_sources(self) -> None:
        config_text = BASE_CONFIG.read_text(encoding="utf-8")
        wrapper_text = WRAPPER.read_text(encoding="utf-8")
        exports = dict(
            re.findall(
                r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$',
                config_text,
                flags=re.MULTILINE,
            )
        )
        # The wrapper now has independent profiles with intentionally
        # different exact values.  The target64 validator appears first and is
        # the one whose defaults this test audits.
        requirements: dict[str, str] = {}
        for name, value in re.findall(
            r'^\s*require_exact ([A-Z][A-Z0-9_]*) "([^"]*)"$',
            wrapper_text,
            flags=re.MULTILINE,
        ):
            requirements.setdefault(name, value)

        # These two values are measured target64 tuning inputs, not code defaults.
        tuned = {
            "VLLM_SCHED_DEFAULT_PRED_OUT": "357",
            "VLLM_SCHED_AVG_CALL_SERVICE_S": "3.3",
        }
        for name, expected in tuned.items():
            self.assertEqual(exports[name], expected)
            self.assertEqual(requirements[name], expected)

        # Every other newly-explicit control below intentionally freezes the
        # exact behavior that an unset variable has in the current code.
        default_sources = {
            "VLLM_SCHED_PRED_OUT_EMA_ALPHA": TRACE_RUNNER,
            "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2": SCHEDULER_PATCH,
            "VLLM_SCHED_DECODE_TOKENS_PER_S_V2": SCHEDULER_PATCH,
            "VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S": SCHEDULER_PATCH,
            "VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT": SCHEDULER_PATCH,
            "VLLM_SCHED_JOINT_V2_FINAL_LANE": SCHEDULER_PATCH,
            "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE": SCHEDULER_PATCH,
            "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA": SCHEDULER_PATCH,
            "VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS": SCHEDULER_PATCH,
            "VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S": SCHEDULER_PATCH,
            "VLLM_SCHED_HBM_LOW_PRESSURE": SCHEDULER_PATCH,
            "VLLM_SCHED_HBM_HIGH_PRESSURE": SCHEDULER_PATCH,
            "VLLM_SCHED_HBM_BUDGET_INCREASE": SCHEDULER_PATCH,
            "VLLM_SCHED_HBM_BUDGET_DECREASE": SCHEDULER_PATCH,
            "VLLM_SCHED_HBM_CONTROL_INTERVAL_S": SCHEDULER_PATCH,
            "VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO": SCHEDULER_PATCH,
        }
        source_cache: dict[Path, str] = {}
        for name, source_path in default_sources.items():
            source_text = source_cache.setdefault(
                source_path,
                source_path.read_text(encoding="utf-8"),
            )
            match = re.search(
                rf'os\.getenv\(\s*"{re.escape(name)}",\s*"([^"]+)"',
                source_text,
            )
            self.assertIsNotNone(match, f"missing code default for {name}")
            assert match is not None
            self.assertEqual(exports[name], match.group(1), name)
            self.assertEqual(requirements[name], exports[name], name)

    def _fixture(
        self,
        root: Path,
        *,
        load_instance_count: int = 120,
        stress180_profile: bool = False,
        native256_profile: bool = False,
        exact_rescue120_profile: bool = False,
        soft4_profile: bool = False,
        stress240_a_probe: bool = False,
        stress240_d_screen: bool = False,
        stress300_a_probe: bool = False,
        stress300_keepalive60_a_probe: bool = False,
    ) -> tuple[Path, Path, dict[str, str]]:
        heldout_manifest, tokenizer = _build_heldout_fixture(root)
        stress_manifest = root / f"manifest_stress{load_instance_count}.json"
        build_stress_bundle(
            manifest_path=heldout_manifest,
            output_root=root / f"stress{load_instance_count}",
            output_manifest=stress_manifest,
            tokenizer=tokenizer,
            load_instance_count=load_instance_count,
        )

        relative_manifest = stress_manifest.relative_to(REPOSITORY_ROOT).as_posix()
        relative_run_base = (root / "runs").relative_to(REPOSITORY_ROOT).as_posix()
        config = root / "stress.env"
        overrides = [
            f"export PASTE_FIXED_WORKLOAD_MANIFEST={shlex.quote(relative_manifest)}",
            f"export PASTE_STRESS_RUN_BASE={shlex.quote(relative_run_base)}",
        ]
        if (
            exact_rescue120_profile
            or stress240_d_screen
            or stress300_a_probe
            or stress300_keepalive60_a_probe
        ):
            pass
        elif soft4_profile:
            overrides.extend(
                (
                    "export PASTE_STRESS_PROFILE=stress180_native256_g256_u86_soft4",
                    "export PASTE_MAX_ACTIVE_TRACES=180",
                    "export VLLM_GPU_MEMORY_UTILIZATION=0.86",
                    "export VLLM_MAX_NUM_SEQS=256",
                    "export VLLM_CUDA_GRAPH_SIZES=256",
                    "export VLLM_SCHED_TIME_AGING_ALPHA=0.2",
                    "export VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S=6000",
                    "export VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS=256",
                    "export VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING=48",
                    "export VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S=40",
                    "export VLLM_SCHED_JOINT_V2_FINAL_LANE=1",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE=0",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES=0",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S=4.0",
                    "export VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING=48",
                    "export VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING=256",
                    "export VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING=256",
                    "export VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION=1",
                    "export VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY=0",
                    "export VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S=0",
                )
            )
        elif stress240_a_probe:
            overrides.extend(
                (
                    "export PASTE_STRESS_PROFILE=stress240_native256_g256_u86_a_probe",
                    "export PASTE_MAX_ACTIVE_TRACES=240",
                    "export VLLM_GPU_MEMORY_UTILIZATION=0.86",
                    "export VLLM_MAX_NUM_SEQS=256",
                    "export VLLM_CUDA_GRAPH_SIZES=256",
                    "export VLLM_SCHED_TIME_AGING_ALPHA=0.2",
                    "export VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S=6000",
                    "export VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS=256",
                    "export VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S=40",
                    "export VLLM_SCHED_JOINT_V2_FINAL_LANE=1",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE=1",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES=1",
                    "export VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING=48",
                    "export VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING=256",
                    "export VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING=256",
                    "export VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION=1",
                    "export VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY=0",
                    "export VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S=0",
                    "export PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION=0.50",
                    "export PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION=0.20",
                    "export PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST=0.25",
                )
            )
        elif native256_profile:
            overrides.extend(
                (
                    "export PASTE_STRESS_PROFILE=stress180_native256_g256_u86",
                    "export PASTE_MAX_ACTIVE_TRACES=180",
                    "export VLLM_GPU_MEMORY_UTILIZATION=0.86",
                    "export VLLM_MAX_NUM_SEQS=256",
                    "export VLLM_CUDA_GRAPH_SIZES=256",
                    "export VLLM_SCHED_TIME_AGING_ALPHA=0.2",
                    "export VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S=6000",
                    "export VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS=256",
                    "export VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S=40",
                    "export VLLM_SCHED_JOINT_V2_FINAL_LANE=1",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE=1",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES=1",
                    "export VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING=48",
                    "export VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING=256",
                    "export VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING=256",
                    "export VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION=1",
                    "export VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY=0",
                    "export VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S=0",
                )
            )
        elif stress180_profile:
            overrides.extend(
                (
                    "export PASTE_STRESS_PROFILE=stress180_target64_u86",
                    "export PASTE_MAX_ACTIVE_TRACES=180",
                    "export VLLM_GPU_MEMORY_UTILIZATION=0.86",
                    "export VLLM_SCHED_TIME_AGING_ALPHA=0.2",
                    "export VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S=40",
                    "export VLLM_SCHED_JOINT_V2_FINAL_LANE=1",
                    "export VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE=1",
                )
            )
        if stress300_keepalive60_a_probe:
            source_config = STRESS300_KEEPALIVE60_A_PROBE_CONFIG
        elif stress300_a_probe:
            source_config = STRESS300_A_PROBE_CONFIG
        elif stress240_d_screen:
            accepted_probe = _write_accepted_probe_fixture(root / "accepted_a")
            relative_probe = accepted_probe.relative_to(REPOSITORY_ROOT).as_posix()
            overrides.append(
                f"export PASTE_ACCEPTED_A_PROBE={shlex.quote(relative_probe)}"
            )
            source_config = STRESS240_D_SCREEN_CONFIG
        elif exact_rescue120_profile:
            source_config = EXACT_RESCUE120_CONFIG
        else:
            source_config = BASE_CONFIG
        config.write_text(
            source_config.read_text(encoding="utf-8")
            + "\n"
            + "\n".join(overrides)
            + "\n",
            encoding="utf-8",
        )

        environment_root = root / "env"
        (environment_root / "bin").mkdir(parents=True)
        (environment_root / "bin" / "python").symlink_to(Path(sys.executable).resolve())
        environment = os.environ.copy()
        environment["PASTE_ENV_PREFIX"] = str(environment_root)
        return config, root / "runs", environment

    def _run(
        self,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(WRAPPER), *args],
            cwd="/",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_check_only_validates_real_stress_manifest_without_output(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(root)
            completed = self._run(
                "cpu_reverse",
                "--config",
                str(config),
                "--cell-order",
                "D,A",
                "--gpus",
                "7,6,5,4",
                "--port",
                "18123",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("joint_learned,fcfs_none", completed.stdout)
            self.assertIn(
                "Validated fixed stress manifest and 2 requested cell(s)",
                completed.stdout,
            )
            self.assertIn("no output was created", completed.stdout)
            self.assertFalse((run_base / "cpu_reverse").exists())

    def test_check_only_accepts_stress180_frozen_config_profile(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=180,
                stress180_profile=True,
            )
            completed = self._run(
                "cpu_stress180",
                "--config",
                str(config),
                "--gpus",
                "7,6,5,4",
                "--port",
                "18124",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("stress180 target64/u86 pair", completed.stdout)
            self.assertIn(
                "profile: stress180_target64_u86",
                completed.stdout,
            )
            self.assertIn(
                "Validated fixed stress manifest and 2 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse((run_base / "cpu_stress180").exists())

    def test_check_only_accepts_stress180_native256_profile(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=180,
                native256_profile=True,
            )
            completed = self._run(
                "cpu_native256",
                "--config",
                str(config),
                "--gpus",
                "7,6,5,4",
                "--port",
                "18125",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("stress180 native256/graph256/u86 pair", completed.stdout)
            self.assertIn("profile: stress180_native256_g256_u86", completed.stdout)
            self.assertIn(
                "Validated fixed stress manifest and 2 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse((run_base / "cpu_native256").exists())

    def test_check_only_accepts_exact_rescue120_d_only_screen(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=180,
                exact_rescue120_profile=True,
            )
            completed = self._run(
                "cpu_exact_rescue120",
                "--config",
                str(config),
                "--cells",
                "D",
                "--gpus",
                "7,6,5,4",
                "--port",
                "18128",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("exact-stage rescue120 D-only screen", completed.stdout)
            self.assertIn(
                "profile: stress180_native256_g256_u86_exact_rescue120",
                completed.stdout,
            )
            self.assertIn("cells:   joint_learned", completed.stdout)
            self.assertIn(
                "Validated fixed stress manifest and 1 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse((run_base / "cpu_exact_rescue120").exists())

    def test_check_only_accepts_stress180_native256_soft4_profile(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=180,
                soft4_profile=True,
            )
            completed = self._run(
                "cpu_native256_soft4",
                "--config",
                str(config),
                "--gpus",
                "7,6,5,4",
                "--port",
                "18127",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn(
                "stress180 native256/graph256/u86 soft4 pair",
                completed.stdout,
            )
            self.assertIn(
                "profile: stress180_native256_g256_u86_soft4",
                completed.stdout,
            )
            self.assertIn(
                "Validated fixed stress manifest and 2 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse((run_base / "cpu_native256_soft4").exists())

    def test_check_only_accepts_stress240_a_only_probe(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=240,
                stress240_a_probe=True,
            )
            completed = self._run(
                "cpu_stress240_a",
                "--config",
                str(config),
                "--cells",
                "A",
                "--gpus",
                "7,6,5,4",
                "--port",
                "18126",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn(
                "stress240 native256/graph256/u86 A-only load probe",
                completed.stdout,
            )
            self.assertIn("profile: stress240_native256_g256_u86_a_probe", completed.stdout)
            self.assertIn("cells:   fcfs_none", completed.stdout)
            self.assertIn(
                "Validated fixed stress manifest and 1 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse((run_base / "cpu_stress240_a").exists())

    def test_stress240_probe_rejects_d_and_capacity_or_gate_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=240,
                stress240_a_probe=True,
            )
            pair = self._run(
                "selection_must_not_see_d",
                "--config",
                str(config),
                "--cells",
                "A,D",
                "--check-only",
                env=environment,
            )
            self.assertEqual(pair.returncode, 2, pair.stdout)
            self.assertIn("requires --cells A", pair.stdout)

            for name, value, expected in (
                ("PASTE_MAX_ACTIVE_TRACES", "256", "must be '240'"),
                ("VLLM_MAX_NUM_SEQS", "240", "must be '256'"),
                ("VLLM_CUDA_GRAPH_SIZES", "512", "must be '256'"),
                ("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION", "0", "must be '1'"),
                (
                    "PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION",
                    "0.49",
                    "must be '0.50'",
                ),
                (
                    "PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST",
                    "0.50",
                    "must be '0.25'",
                ),
            ):
                with self.subTest(name=name):
                    drifted = root / f"stress240_drifted_{name}.env"
                    drifted.write_text(
                        config.read_text(encoding="utf-8")
                        + f"\nexport {name}={value}\n",
                        encoding="utf-8",
                    )
                    completed = self._run(
                        f"bad_stress240_{name.lower()}",
                        "--config",
                        str(drifted),
                        "--cells",
                        "A",
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertIn(expected, completed.stdout)

    def test_check_only_accepts_stress300_a_only_probe(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=300,
                stress300_a_probe=True,
            )
            completed = self._run(
                "cpu_stress300_a",
                "--config",
                str(config),
                "--cells",
                "A",
                "--gpus",
                "7,6,5,4",
                "--port",
                "18130",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn(
                "stress300 native320/graph256/u86 A-only load probe",
                completed.stdout,
            )
            self.assertIn(
                "profile: stress300_native320_g256_u86_a_probe",
                completed.stdout,
            )
            self.assertIn("cells:   fcfs_none", completed.stdout)
            self.assertIn(
                "Validated fixed stress manifest and 1 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse((run_base / "cpu_stress300_a").exists())

    def test_check_only_accepts_stress300_keepalive60_a_only_probe(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=300,
                stress300_keepalive60_a_probe=True,
            )
            completed = self._run(
                "cpu_stress300_keepalive60_a",
                "--config",
                str(config),
                "--cells",
                "A",
                "--gpus",
                "7,6,5,4",
                "--port",
                "18130",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn(
                "stress300 native320/graph256/u86 keepalive60 A-only "
                "retry-clean load probe",
                completed.stdout,
            )
            self.assertIn(
                "profile: stress300_native320_g256_u86_keepalive60_a_probe",
                completed.stdout,
            )
            self.assertIn(
                "HTTP keep-alive: 60s (frozen server setting)",
                completed.stdout,
            )
            self.assertIn("cells:   fcfs_none", completed.stdout)
            self.assertIn(
                "Validated fixed stress manifest and 1 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse(
                (run_base / "cpu_stress300_keepalive60_a").exists()
            )

    def test_stress300_keepalive60_probe_rejects_d_or_timeout_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=300,
                stress300_keepalive60_a_probe=True,
            )
            disallowed = self._run(
                "stress300_keepalive60_must_not_see_d",
                "--config",
                str(config),
                "--cells",
                "D",
                "--check-only",
                env=environment,
            )
            self.assertEqual(disallowed.returncode, 2, disallowed.stdout)
            self.assertIn("requires --cells A", disallowed.stdout)

            drifted = root / "stress300_keepalive_drifted.env"
            drifted.write_text(
                config.read_text(encoding="utf-8")
                + "\nexport VLLM_HTTP_TIMEOUT_KEEP_ALIVE=5\n",
                encoding="utf-8",
            )
            rejected = self._run(
                "stress300_keepalive_drift",
                "--config",
                str(drifted),
                "--cells",
                "A",
                "--check-only",
                env=environment,
            )
            self.assertEqual(rejected.returncode, 2, rejected.stdout)
            self.assertIn(
                "VLLM_HTTP_TIMEOUT_KEEP_ALIVE must be '60'", rejected.stdout
            )

    def test_stress300_probe_rejects_d_and_any_frozen_shape_or_gate_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=300,
                stress300_a_probe=True,
            )
            for cells in ("A,D", "D,A", "D"):
                with self.subTest(cells=cells):
                    disallowed = self._run(
                        "stress300_selection_must_not_see_d",
                        "--config",
                        str(config),
                        "--cells",
                        cells,
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(
                        disallowed.returncode, 2, disallowed.stdout
                    )
                    self.assertIn("requires --cells A", disallowed.stdout)

            for name, value, expected in (
                ("PASTE_MAX_ACTIVE_TRACES", "299", "must be '300'"),
                ("VLLM_MAX_NUM_SEQS", "300", "must be '320'"),
                ("VLLM_CUDA_GRAPH_SIZES", "320", "must be '256'"),
                ("VLLM_GPU_MEMORY_UTILIZATION", "0.85", "must be '0.86'"),
                (
                    "VLLM_MAX_NUM_BATCHED_TOKENS",
                    "16384",
                    "must be '8192'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS",
                    "300",
                    "must be '320'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING",
                    "300",
                    "must be '320'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING",
                    "300",
                    "must be '320'",
                ),
                ("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION", "0", "must be '1'"),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
                    "0",
                    "must be '1'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
                    "1",
                    "must be '0'",
                ),
                ("VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY", "1", "must be '0'"),
                (
                    "PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION",
                    "0.49",
                    "must be '0.50'",
                ),
                (
                    "PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION",
                    "0.19",
                    "must be '0.20'",
                ),
                (
                    "PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST",
                    "0.26",
                    "must be '0.25'",
                ),
            ):
                with self.subTest(name=name):
                    drifted = root / f"stress300_drifted_{name}.env"
                    drifted.write_text(
                        config.read_text(encoding="utf-8")
                        + f"\nexport {name}={value}\n",
                        encoding="utf-8",
                    )
                    completed = self._run(
                        f"bad_stress300_{name.lower()}",
                        "--config",
                        str(drifted),
                        "--cells",
                        "A",
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertIn(expected, completed.stdout)

    def test_stress300_probe_rejects_non_five_copy_manifest(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=300,
                stress300_a_probe=True,
            )
            original_manifest = root / "manifest_stress300.json"
            bad_manifest = root / "manifest_stress300_bad_multiplicity.json"
            payload = json.loads(original_manifest.read_text(encoding="utf-8"))
            payload["stress_definition"]["instances_per_source"] = 4
            bad_manifest.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            config.write_text(
                config.read_text(encoding="utf-8")
                + "\nexport PASTE_FIXED_WORKLOAD_MANIFEST="
                + shlex.quote(bad_manifest.relative_to(REPOSITORY_ROOT).as_posix())
                + "\n",
                encoding="utf-8",
            )

            completed = self._run(
                "stress300_bad_multiplicity",
                "--config",
                str(config),
                "--cells",
                "A",
                "--check-only",
                env=environment,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn(
                "exactly five balanced instances of each of 60 heldout sources",
                completed.stdout,
            )

    def test_check_only_accepts_stress240_d_screen_after_a_probe_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=240,
                stress240_d_screen=True,
            )
            completed = self._run(
                "cpu_stress240_d",
                "--config",
                str(config),
                "--cells",
                "D",
                "--gpus",
                "7,6,5,4",
                "--port",
                "18129",
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("accepted-A D-only screen", completed.stdout)
            self.assertIn(
                "profile: stress240_native256_g256_u86_exact_rescue120",
                completed.stdout,
            )
            self.assertIn("cells:   joint_learned", completed.stdout)
            self.assertIn("A probe:", completed.stdout)
            self.assertIn("(accepted)", completed.stdout)
            self.assertIn(
                "Validated fixed stress manifest and 1 requested cell(s)",
                completed.stdout,
            )
            self.assertFalse((run_base / "cpu_stress240_d").exists())

    def test_stress240_d_screen_requires_d_and_a_completed_repo_relative_probe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(
                root,
                load_instance_count=240,
                stress240_d_screen=True,
            )
            pair = self._run(
                "stress240_d_must_be_single",
                "--config",
                str(config),
                "--cells",
                "A,D",
                "--check-only",
                env=environment,
            )
            self.assertEqual(pair.returncode, 2, pair.stdout)
            self.assertIn("requires --cells D", pair.stdout)

            cases = (
                ("", "must be explicitly set"),
                (
                    "reproduction/artifacts/not-finished/natural_queue_probe.json",
                    "missing or incomplete",
                ),
                ("/tmp/natural_queue_probe.json", "must be repository-relative"),
                ("../natural_queue_probe.json", "must stay inside the repository"),
            )
            for index, (probe_value, expected) in enumerate(cases):
                with self.subTest(probe_value=probe_value):
                    drifted = root / f"stress240_bad_probe_{index}.env"
                    drifted.write_text(
                        config.read_text(encoding="utf-8")
                        + "\nexport PASTE_ACCEPTED_A_PROBE="
                        + shlex.quote(probe_value)
                        + "\n",
                        encoding="utf-8",
                    )
                    completed = self._run(
                        f"stress240_bad_probe_{index}",
                        "--config",
                        str(drifted),
                        "--cells",
                        "D",
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertIn(expected, completed.stdout)
                    self.assertFalse(
                        (run_base / f"stress240_bad_probe_{index}").exists()
                    )

    def test_stress240_d_screen_rejects_policy_engine_or_gate_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=240,
                stress240_d_screen=True,
            )
            for name, value, expected in (
                ("PASTE_MAX_ACTIVE_TRACES", "239", "must be '240'"),
                ("VLLM_MAX_NUM_SEQS", "240", "must be '256'"),
                ("VLLM_CUDA_GRAPH_SIZES", "128", "must be '256'"),
                ("VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S", "119", "must be '120'"),
                (
                    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
                    "49",
                    "must be '48'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
                    "1",
                    "must be '0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
                    "1",
                    "must be '0'",
                ),
                ("VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY", "1", "must be '0'"),
                (
                    "PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION",
                    "0.49",
                    "must be '0.50'",
                ),
                (
                    "PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST",
                    "0.26",
                    "must be '0.25'",
                ),
            ):
                with self.subTest(name=name):
                    drifted = root / f"stress240_d_drifted_{name}.env"
                    drifted.write_text(
                        config.read_text(encoding="utf-8")
                        + f"\nexport {name}={value}\n",
                        encoding="utf-8",
                    )
                    completed = self._run(
                        f"bad_stress240_d_{name.lower()}",
                        "--config",
                        str(drifted),
                        "--cells",
                        "D",
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertIn(expected, completed.stdout)

    def test_native256_profile_rejects_graph_or_admission_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=180,
                native256_profile=True,
            )
            for name, value, expected in (
                ("VLLM_CUDA_GRAPH_SIZES", "512", "must be '256'"),
                (
                    "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S",
                    "5999",
                    "must be '6000'",
                ),
                ("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION", "0", "must be '1'"),
                ("VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY", "1", "must be '0'"),
                (
                    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S",
                    "60",
                    "must be '0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
                    "256",
                    "must be '48'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
                    "0",
                    "must be '1'",
                ),
            ):
                with self.subTest(name=name):
                    drifted = root / f"drifted_{name}.env"
                    drifted.write_text(
                        config.read_text(encoding="utf-8")
                        + f"\nexport {name}={value}\n",
                        encoding="utf-8",
                    )
                    completed = self._run(
                        f"bad_{name.lower()}",
                        "--config",
                        str(drifted),
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertIn(expected, completed.stdout)

    def test_exact_rescue120_rejects_pair_and_any_engine_or_score_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=180,
                exact_rescue120_profile=True,
            )
            pair = self._run(
                "rescue120_is_screen_only",
                "--config",
                str(config),
                "--cells",
                "A,D",
                "--check-only",
                env=environment,
            )
            self.assertEqual(pair.returncode, 2, pair.stdout)
            self.assertIn("requires --cells D", pair.stdout)

            for name, value, expected in (
                ("VLLM_CUDA_GRAPH_SIZES", "512", "must be '256'"),
                ("VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S", "119", "must be '120'"),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE",
                    "0",
                    "must be '1'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
                    "1",
                    "must be '0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
                    "0.1",
                    "must be '0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
                    "49",
                    "must be '48'",
                ),
                ("VLLM_SCHED_JOINT_V2_TAIL_BETA", "0.26", "must be '0.25'"),
                ("VLLM_SCHED_HBM_LOW_PRESSURE", "0.81", "must be '0.82'"),
                ("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION", "0", "must be '1'"),
                ("VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY", "1", "must be '0'"),
            ):
                with self.subTest(name=name):
                    drifted = root / f"rescue120_drifted_{name}.env"
                    drifted.write_text(
                        config.read_text(encoding="utf-8")
                        + f"\nexport {name}={value}\n",
                        encoding="utf-8",
                    )
                    completed = self._run(
                        f"bad_rescue120_{name.lower()}",
                        "--config",
                        str(drifted),
                        "--cells",
                        "D",
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertIn(expected, completed.stdout)

    def test_soft4_profile_rejects_stage_or_native_admission_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(
                root,
                load_instance_count=180,
                soft4_profile=True,
            )
            for name, value, expected in (
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
                    "3.9",
                    "must be '4.0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE",
                    "1",
                    "must be '0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
                    "1",
                    "must be '0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
                    "39",
                    "must be '40'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
                    "256",
                    "must be '48'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY",
                    "1",
                    "must be '0'",
                ),
                (
                    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
                    "0",
                    "must be '1'",
                ),
            ):
                with self.subTest(name=name):
                    drifted = root / f"soft4_drifted_{name}.env"
                    drifted.write_text(
                        config.read_text(encoding="utf-8")
                        + f"\nexport {name}={value}\n",
                        encoding="utf-8",
                    )
                    completed = self._run(
                        f"bad_soft4_{name.lower()}",
                        "--config",
                        str(drifted),
                        "--check-only",
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertIn(expected, completed.stdout)

    def test_stress180_profile_rejects_stress120_manifest(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(root)
            config.write_text(
                config.read_text(encoding="utf-8")
                + "\nexport PASTE_STRESS_PROFILE=stress180_target64_u86\n"
                + "export PASTE_MAX_ACTIVE_TRACES=180\n"
                + "export VLLM_GPU_MEMORY_UTILIZATION=0.86\n",
                encoding="utf-8",
            )
            completed = self._run(
                "wrong_manifest_count",
                "--config",
                str(config),
                "--check-only",
                env=environment,
            )

            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("requires 180 load instances", completed.stdout)

    def test_rejects_existing_output_and_configuration_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, run_base, environment = self._fixture(root)
            occupied = run_base / "occupied"
            occupied.mkdir(parents=True)

            existing = self._run(
                "occupied",
                "--config",
                str(config),
                "--check-only",
                env=environment,
            )
            self.assertEqual(existing.returncode, 2, existing.stdout)
            self.assertIn("output or lock already exists", existing.stdout)

            drifted = root / "drifted.env"
            drifted.write_text(
                config.read_text(encoding="utf-8")
                + "\nexport VLLM_MAX_NUM_SEQS=63\n",
                encoding="utf-8",
            )
            drift = self._run(
                "drifted",
                "--config",
                str(drifted),
                "--check-only",
                env=environment,
            )
            self.assertEqual(drift.returncode, 2, drift.stdout)
            self.assertIn("VLLM_MAX_NUM_SEQS must be '64'", drift.stdout)

    def test_rejects_bad_cell_gpu_port_and_path_inputs_before_gpu_use(self) -> None:
        for args, expected in (
            (
                ("bad_cells", "--cells", "A,A"),
                "--cells must contain exactly A and D",
            ),
            (("bad_gpus", "--gpus", "0,1,1,2"), "GPU IDs must be distinct"),
            (("bad_port", "--port", "65536"), "--port must be an integer"),
        ):
            with self.subTest(args=args):
                completed = self._run(*args)
                self.assertEqual(completed.returncode, 2, completed.stdout)
                self.assertIn(expected, completed.stdout)

        with tempfile.TemporaryDirectory(
            prefix=".stress-wrapper-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            config, _, environment = self._fixture(root)
            escaped = root / "escaped.env"
            escaped.write_text(
                config.read_text(encoding="utf-8")
                + "\nexport PASTE_STRESS_RUN_BASE=../outside-repository\n",
                encoding="utf-8",
            )
            completed = self._run(
                "bad_path",
                "--config",
                str(escaped),
                "--check-only",
                env=environment,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("must stay inside the repository", completed.stdout)

    def test_four_cell_validate_only_flag_is_strict(self) -> None:
        environment = os.environ.copy()
        environment["PASTE_VALIDATE_ONLY"] = "sometimes"
        completed = subprocess.run(
            ["bash", str(FOUR_CELL), "stress"],
            cwd="/",
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("PASTE_VALIDATE_ONLY must be 0 or 1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
