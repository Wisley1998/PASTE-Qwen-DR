from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import uuid

from reproduction.tests.test_run_live_joint_formal_matrix import (
    _task_fixture,
    _tool_record,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "reproduction/scripts/run_live_joint_v8_load_rehearsal.py"
)
VALIDATOR_PATH = (
    REPOSITORY_ROOT
    / "reproduction/scripts/validate_live_joint_v8_load_rehearsal.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("v8_load_rehearsal_runner_test", RUNNER_PATH)
validator = _load("v8_load_rehearsal_validator_test", VALIDATOR_PATH)


class DevelopmentWorkloadTests(unittest.TestCase):
    def test_only_frozen_tune_v1_is_eligible(self) -> None:
        observed = validator.validate_development_workload()
        self.assertTrue(observed["valid"])
        self.assertTrue(observed["development_only"])
        self.assertFalse(observed["formal_evidence_eligible"])
        self.assertEqual(observed["source_count"], 16)
        self.assertEqual(observed["replicas"], 5)
        self.assertEqual(observed["offered_task_count"], 80)
        self.assertNotEqual(
            observed["file_sha256"],
            observed["formal_v8_workload_sha256_forbidden"],
        )
        formal_workload = (
            REPOSITORY_ROOT
            / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v8.json"
        )
        with self.assertRaisesRegex(
            validator.RehearsalValidationError, "frozen tune-v1"
        ):
            validator.validate_development_workload(formal_workload)

    def test_mutated_tune_workload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / validator.TUNE_WORKLOAD.name
            path.write_bytes(validator.TUNE_WORKLOAD.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                validator.RehearsalValidationError, "repository path"
            ):
                validator.validate_development_workload(path)


class RehearsalRunnerTests(unittest.TestCase):
    def test_derived_command_is_16x5_native_a_and_never_formal_v8(self) -> None:
        config = runner.formal.load_frozen_config(runner.CONFIG)
        derived = runner._derived_runner_config(config)
        command = runner.formal._runner_command(
            python=Path(config["PASTE_ENV_PREFIX"]) / "bin/python",
            workload=runner.TUNE_WORKLOAD,
            output=REPOSITORY_ROOT / "unused-rehearsal",
            cell="A",
            block_id="development-block-1",
            order_index=0,
            server_instance_id="server-1",
            config=derived,
        )
        rendered = " ".join(command)
        self.assertIn(str(runner.TUNE_WORKLOAD), rendered)
        self.assertNotIn("frozen_formal_v8.json", rendered)
        self.assertIn("--replicas 5", rendered)
        self.assertIn("--max-active-tasks 80", rendered)
        self.assertIn("--speculation-mode off", rendered)
        self.assertIn("--fixed-final-completion-tokens 192", rendered)

    def test_check_only_is_offline_and_creates_no_output(self) -> None:
        tag = "unit-rehearsal-check-" + uuid.uuid4().hex
        output = runner.RUN_BASE / tag
        completed = subprocess.run(
            [
                "/home/aiscuser/.conda/envs/paste/bin/python",
                str(RUNNER_PATH),
                tag,
                "--check-only",
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["development_only"])
        self.assertFalse(payload["formal_evidence_eligible"])
        self.assertFalse(payload["selection_uses_formal_v8_performance"])
        self.assertFalse(payload["gpu_or_server_touched"])
        self.assertFalse(payload["network_touched"])
        self.assertEqual(payload["offered_concurrency"], 80)
        self.assertEqual(payload["native_sequence_ceiling"], 96)
        self.assertEqual(
            payload["fixed_final_grammar_feasibility"]["source_count"], 16
        )
        self.assertFalse(output.exists())


class StrictValidatorTests(unittest.TestCase):
    def _fixture(self) -> tuple[dict, list[dict]]:
        payload = json.loads(validator.TUNE_WORKLOAD.read_text(encoding="utf-8"))
        tasks = []
        events = []
        for task_index in range(80):
            source = payload["sources"][task_index // 5]
            replica = task_index % 5
            task, task_events = _task_fixture(
                task_index,
                source_id=source["source_id"],
                replica=replica,
                selected_url=source["expected_url"],
                search_query=source["search_query"],
                question=source["question"],
            )
            tasks.append(task)
            events.extend(task_events)
        config = validator._expected_config("development-block-1", "server-1")
        config["scheduler_environment"] = {
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
            "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
            "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
            "VLLM_PORT": "8100",
            "VLLM_MAX_MODEL_LEN": "16384",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
            "VLLM_MAX_NUM_SEQS": "96",
            "VLLM_ENABLE_PREFIX_CACHING": "1",
            "VLLM_USE_V1": "1",
            "VLLM_SCHED_POLICY": "fcfs",
        }
        result = {
            "config": config,
            "summary": {
                "all_tasks_succeeded": True,
                "task_count": 80,
                "successful_task_count": 80,
                "failed_task_count": 0,
                "llm": {
                    "request_count": 240,
                    "successful_request_count": 240,
                    "exactly_one_attempt_each": True,
                },
                "tool": {
                    "broker_stats": {
                        "authoritative_requests": 160,
                        "authoritative_executions": 160,
                        "authoritative_failures": 0,
                        "commits": 160,
                        "speculative_admitted": 0,
                        "speculative_started": 0,
                        "speculative_failures": 0,
                    }
                },
            },
            "tasks": tasks,
            "llm_events": events,
            "tool_attempt_records": [
                *(_tool_record("search", index) for index in range(80)),
                *(_tool_record("visit", index) for index in range(80)),
            ],
            "broker_final_snapshot": {
                "jobs": [],
                "counts": {
                    "completed_unclaimed_speculative": 0,
                    "queued_authoritative": 0,
                    "queued_speculative": 0,
                    "running_authoritative": 0,
                    "running_speculative": 0,
                    "queued_by_tool": {},
                    "running_by_tool": {},
                },
            },
        }
        timeline = []
        for index in range(40):
            pressure = index < 10
            timeline.append(
                {
                    "monotonic_s": 100.0 + index * 0.2,
                    "wall_s": 1000.0 + index * 0.2,
                    "llm_running": 12,
                    "llm_waiting": 1 if pressure else 0,
                    "tool_queued_authoritative": 1 if pressure else 0,
                }
            )
        return result, timeline

    def _validate(self, result: dict, timeline: list[dict]) -> dict:
        artifacts = REPOSITORY_ROOT / "reproduction/artifacts"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
            root = Path(temporary)
            result_path = root / "result.json"
            timeline_path = root / "queue_timeline.jsonl"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            timeline_path.write_text(
                "".join(json.dumps(row) + "\n" for row in timeline),
                encoding="utf-8",
            )
            return validator.validate_rehearsal_result(
                result_path=result_path,
                timeline_path=timeline_path,
                block_id="development-block-1",
                server_instance_id="server-1",
            )

    def test_full_strict_fixture_and_zero_retry_threshold(self) -> None:
        result, timeline = self._fixture()
        validation = self._validate(result, timeline)
        self.assertTrue(validation["valid"])
        self.assertFalse(validation["formal_evidence_eligible"])
        self.assertTrue(validation["baseline_gate"]["accepted"])
        self.assertTrue(validation["thresholds"]["zero_transport_retries_required"])

        result["tool_attempt_records"][0]["http_attempts"] = 2
        with self.assertRaisesRegex(
            validator.RehearsalValidationError, "one committed live GET"
        ):
            self._validate(result, timeline)

    def test_fixed_final_and_sampling_hole_mutations_fail(self) -> None:
        result, timeline = self._fixture()
        result["llm_events"][2]["usage"]["completion_tokens"] = 191
        with self.assertRaisesRegex(validator.formal.FormalRunError, "fixed-final"):
            self._validate(result, timeline)

        result, timeline = self._fixture()
        for index in range(5, len(timeline)):
            timeline[index]["monotonic_s"] += 3.0
            timeline[index]["wall_s"] += 3.0
        with self.assertRaisesRegex(
            validator.RehearsalValidationError, "dual-queue gate"
        ):
            self._validate(result, timeline)


if __name__ == "__main__":
    unittest.main()
