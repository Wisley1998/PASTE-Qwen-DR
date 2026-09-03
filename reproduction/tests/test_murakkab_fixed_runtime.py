from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from paste_repro.murakkab_fixed_runtime import (
    EXPECTED_RUNTIME_CONFIG,
    EXPECTED_SCHEDULER_ENV,
    FIXED_WORKFLOW,
    FixedTypedWorkflow,
    MurakkabFixedError,
    build_singleton_plan,
    compute_fixed_metrics,
    validate_dependency_dispatch,
    validate_live_result,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPOSITORY_ROOT / "reproduction/configs/murakkab_fixed_v9_m_only.json"
RUNNER = REPOSITORY_ROOT / "reproduction/scripts/run_murakkab_fixed_live.py"
HISTORICAL_A_DIAGNOSTIC = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/comment3_scheduler"
    / "comment3-target-r2/cells/01-a-c10k-l80/evidence/result.json"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_murakkab_fixed_live", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_record(task_id: str, tool: str, start: float) -> dict:
    return {
        "invocation_id": f"{task_id}-{tool}",
        "session_id": task_id,
        "tool": tool,
        "speculative": False,
        "authoritative": True,
        "admitted": True,
        "committed": True,
        "source": "executed",
        "queue_enter_at": start,
        "started_at": start + 0.1,
        "finished_at": start + 0.8,
        "queue_s": 0.1,
        "service_s": 0.7,
        "http_attempts": 1,
    }


def _result(task_count: int = 2) -> dict:
    config = dict(EXPECTED_RUNTIME_CONFIG)
    config.update(
        {
            "call_graph_mode": "frozen",
            "frozen_url_is_workload_input": True,
            "independent_source_count": task_count,
            "task_count": task_count,
            "workload_file_sha256": (
                "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20"
            ),
            "workload_split_id": "live-joint-wikipedia-frozen-formal-v9",
            "workload_formal_eligible": True,
            "scheduler_environment": dict(EXPECTED_SCHEDULER_ENV),
        }
    )
    tasks = []
    events = []
    records = []
    for index in range(task_count):
        task_id = f"source-{index}__r00"
        base = 100.0 + 10.0 * index
        search = _tool_record(task_id, "search", base + 1.1)
        visit = _tool_record(task_id, "visit", base + 3.1)
        records.extend((search, visit))
        events.extend(
            (
                {
                    "task_id": task_id, "call_index": 0,
                    "request_start_monotonic_s": base, "duration_s": 1.0,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
                {
                    "task_id": task_id, "call_index": 1,
                    "request_start_monotonic_s": base + 2.0, "duration_s": 1.0,
                    "usage": {"prompt_tokens": 200, "completion_tokens": 20},
                },
                {
                    "task_id": task_id, "call_index": 2,
                    "request_start_monotonic_s": base + 4.0, "duration_s": 1.0,
                    "usage": {"prompt_tokens": 300, "completion_tokens": 192},
                },
            )
        )
        tasks.append(
            {
                "task_id": task_id, "ok": True, "e2e_s": 6.0,
                "start_wall_s": 1000.0, "end_wall_s": 1006.0 + index,
                "llm_duration_s": 3.0,
                "tools": [
                    {
                        "invocation": {"tool_name": "search"},
                        "exposed_wait_s": 0.8, "queue_s": 0.1, "service_s": 0.7,
                    },
                    {
                        "invocation": {"tool_name": "visit"},
                        "exposed_wait_s": 0.8, "queue_s": 0.1, "service_s": 0.7,
                    },
                ],
            }
        )
    zero_spec = {
        "speculative_admitted": 0, "speculative_started": 0,
        "speculative_completed": 0, "speculative_failures": 0,
        "queued_promotions": 0, "running_promotions": 0,
        "completed_reuse": 0, "wasted_speculative_service_s": 0.0,
    }
    return {
        "config": config,
        "summary": {
            "all_tasks_succeeded": True, "task_count": task_count,
            "successful_task_count": task_count, "failed_task_count": 0,
            "llm": {
                "request_count": 3 * task_count,
                "successful_request_count": 3 * task_count,
                "exactly_one_attempt_each": True,
            },
            "tool": {
                "broker_stats": {
                    "authoritative_requests": 2 * task_count,
                    "commits": 2 * task_count, "authoritative_failures": 0,
                    **zero_spec,
                }
            },
        },
        "tasks": tasks,
        "llm_events": events,
        "tool_attempt_records": records,
        "broker_final_snapshot": {
            "jobs": [],
            "counts": {
                "completed_unclaimed_speculative": 0,
                "queued_authoritative": 0, "queued_speculative": 0,
                "running_authoritative": 0, "running_speculative": 0,
                "queued_by_tool": {}, "running_by_tool": {},
            },
        },
        "task_completion_makespan_s": 10.0,
    }


class FixedPlanTests(unittest.TestCase):
    def test_manifest_selects_one_typed_registered_workflow(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        plan = build_singleton_plan(protocol)
        self.assertEqual(plan["candidate_count"], 1)
        self.assertFalse(plan["objective_evaluated"])
        self.assertTrue(plan["typed_dag_validated"])
        self.assertEqual(
            plan["workflow"]["topological_order"],
            ["initial_llm", "search", "decision_llm", "visit", "synthesis_llm"],
        )

    def test_fixed_type_checker_rejects_mismatch_and_cycle(self) -> None:
        mismatch = deepcopy(FIXED_WORKFLOW)
        mismatch["nodes"][1]["input_types"]["initial_llm"] = "wrong"
        with self.assertRaisesRegex(MurakkabFixedError, "emits"):
            FixedTypedWorkflow.from_mapping(mismatch)
        cycle = deepcopy(FIXED_WORKFLOW)
        cycle["nodes"][0]["depends_on"] = ["synthesis_llm"]
        cycle["nodes"][0]["input_types"] = {"synthesis_llm": "grounded_answer"}
        with self.assertRaisesRegex(MurakkabFixedError, "cycle"):
            FixedTypedWorkflow.from_mapping(cycle)

    def test_planner_rejects_extra_candidate_dimension(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        protocol["constrained_murakkab"]["model_candidates"] = 2
        with self.assertRaisesRegex(MurakkabFixedError, "not singleton"):
            build_singleton_plan(protocol)


class FixedResultTests(unittest.TestCase):
    def test_result_validates_and_metrics_cover_requested_outputs(self) -> None:
        result = _result()
        evidence = validate_live_result(
            result, call_graph_mode="frozen", expected_task_count=2
        )
        self.assertTrue(evidence["validated"])
        metrics = compute_fixed_metrics(result)
        self.assertEqual(metrics["task_count"], 2)
        self.assertEqual(metrics["throughput"]["completed_tasks_per_s"], 0.2)
        self.assertEqual(metrics["task_e2e"]["p99_s"], 6.0)
        self.assertEqual(metrics["tool"]["physical_http_attempt_count"], 4)

    def test_speculation_dependency_and_broker_leaks_fail_closed(self) -> None:
        speculative = _result()
        speculative["tool_attempt_records"][0]["speculative"] = True
        with self.assertRaisesRegex(MurakkabFixedError, "physical tool"):
            validate_live_result(
                speculative, call_graph_mode="frozen", expected_task_count=2
            )
        reordered = _result()
        reordered["llm_events"][1]["request_start_monotonic_s"] = 101.5
        with self.assertRaisesRegex(MurakkabFixedError, "DAG dependency"):
            validate_live_result(reordered, call_graph_mode="frozen", expected_task_count=2)
        leaked = _result()
        leaked["broker_final_snapshot"]["counts"]["queued_authoritative"] = 1
        with self.assertRaisesRegex(MurakkabFixedError, "non-drained"):
            validate_live_result(leaked, call_graph_mode="frozen", expected_task_count=2)

        wrong_workload = _result()
        wrong_workload["config"]["workload_split_id"] = "some-other-split"
        with self.assertRaisesRegex(MurakkabFixedError, "workload identity"):
            validate_live_result(
                wrong_workload, call_graph_mode="frozen", expected_task_count=2
            )

    @unittest.skipUnless(
        HISTORICAL_A_DIAGNOSTIC.is_file(),
        "optional historical A-schema diagnostic artifact is not installed",
    )
    def test_historical_a_schema_only_dependency_diagnostic(self) -> None:
        # This checks compatibility with real live-runner telemetry only.  It
        # neither enriches nor relabels this historical A result as M evidence.
        historical_a = json.loads(HISTORICAL_A_DIAGNOSTIC.read_text(encoding="utf-8"))
        evidence = validate_dependency_dispatch(historical_a)
        self.assertEqual(evidence["task_count"], 80)
        self.assertFalse(evidence["speculative_tool_execution_observed"])


class FixedRunnerTests(unittest.TestCase):
    def test_environment_purges_joint_knobs_and_command_is_m_only(self) -> None:
        runner = _load_runner()
        env = runner.build_cell_environment(
            state_dir=Path("/tmp/m-state"), log_dir=Path("/tmp/m-log"),
            inherited={
                "PATH": "/bin", "VLLM_SCHED_POLICY": "online_joint_pacer_v2",
                "VLLM_SCHED_JOINT_V2_FINAL_LANE": "1",
            },
        )
        self.assertEqual(env["VLLM_SCHED_POLICY"], "fcfs")
        self.assertNotIn("VLLM_SCHED_JOINT_V2_FINAL_LANE", env)
        command = runner.build_live_command(
            python=Path("/python"), output=Path("/result"),
            run_tag="test-r1", source_limit=None,
        )
        joined = " ".join(command)
        self.assertIn("--call-graph-mode frozen", joined)
        self.assertIn("--speculation-mode off", joined)
        self.assertNotIn("--formal-cell-id", command)
        self.assertNotIn("--fresh-server", command)

    def test_gpu_process_audit_fails_closed(self) -> None:
        runner = _load_runner()
        gpu_stdout = "\n".join(
            f"{index}, GPU-{index}, NVIDIA A100-SXM4-40GB, 40960, 0, 0, 50"
            for index in range(4, 8)
        )
        gpu_result = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout=gpu_stdout, stderr=""
        )
        failed_apps = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=1, stdout="", stderr="query failed"
        )
        with patch.object(runner, "_run_capture", side_effect=(gpu_result, failed_apps)):
            with self.assertRaisesRegex(
                runner.MurakkabLiveRunError, "compute-application query failed"
            ):
                runner.gpu_snapshot()

        malformed_apps = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0,
            stdout="GPU-4, 123, python\n", stderr="",
        )
        with patch.object(
            runner, "_run_capture", side_effect=(gpu_result, malformed_apps)
        ):
            with self.assertRaisesRegex(
                runner.MurakkabLiveRunError, "compute-application row"
            ):
                runner.gpu_snapshot()

    def test_additional_selected_gpu_application_fails_closed(self) -> None:
        runner = _load_runner()
        gpu_stdout = "\n".join(
            f"{index}, GPU-{index}, NVIDIA A100-SXM4-40GB, 40960, 2774, 2, 55"
            for index in range(4, 8)
        )
        app_stdout = "\n".join(
            [
                *(f"GPU-{index}, 2298, python, 2774" for index in range(4, 8)),
                "GPU-4, 9999, python, 100",
            ]
        )
        responses = (
            subprocess.CompletedProcess(["nvidia-smi"], 0, gpu_stdout, ""),
            subprocess.CompletedProcess(["nvidia-smi"], 0, app_stdout, ""),
        )
        with patch.object(runner, "_run_capture", side_effect=responses):
            with self.assertRaisesRegex(runner.MurakkabLiveRunError, "exactly four"):
                runner.gpu_snapshot()

    def test_wrong_registered_child_argv_fails_closed(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as raw_temp:
            proc_root = Path(raw_temp)
            child = proc_root / "42"
            child.mkdir()
            (child / "exe").symlink_to(runner.REGISTERED_BACKGROUND_EXECUTABLE)
            (child / "cwd").symlink_to(runner.REGISTERED_BACKGROUND_CWD)
            (child / "cmdline").write_bytes(b"python\0other.py\0")
            (child / "stat").write_text(
                "42 (python) S 1 " + "0 " * 18 + "1 0\n", encoding="utf-8"
            )
            boot = proc_root / "sys/kernel/random"
            boot.mkdir(parents=True)
            (boot / "boot_id").write_text(
                "00000000-0000-0000-0000-000000000001\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                runner.MurakkabLiveRunError, "registered ResNet child"
            ):
                runner._read_registered_resnet_identity(42, proc_root=proc_root)

    def test_registered_identity_continuity_rejects_pid_change(self) -> None:
        runner = _load_runner()
        identity = {
            "valid": True,
            "policy": runner.REGISTERED_BACKGROUND_POLICY,
            "pid": 42,
            "executable": runner.REGISTERED_BACKGROUND_EXECUTABLE,
            "cwd": runner.REGISTERED_BACKGROUND_CWD,
            "argv": list(runner.REGISTERED_BACKGROUND_ARGV),
            "resolved_script": runner.REGISTERED_BACKGROUND_SCRIPT,
            "resolved_script_sha256": runner.REGISTERED_BACKGROUND_SCRIPT_SHA256,
            "proc_starttime_ticks": 123,
            "boot_id": "00000000-0000-0000-0000-000000000001",
            "selected_gpu_indices": [4, 5, 6, 7],
            "selected_gpu_uuids": ["a", "b", "c", "d"],
            "selected_application_record_count": 4,
            "additional_selected_gpu_compute_apps_observed": False,
        }
        after = deepcopy(identity)
        after["pid"] = 43
        with self.assertRaisesRegex(runner.MurakkabLiveRunError, "identity changed"):
            runner.validate_registered_background_continuity(
                {"registered_background": identity},
                {"registered_background": after},
            )

if __name__ == "__main__":
    unittest.main()
