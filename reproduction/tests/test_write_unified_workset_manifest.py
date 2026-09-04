from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from write_unified_workset_manifest import (  # noqa: E402
    build_manifest,
    canonical_hash,
    main,
    parse_args,
)


class UnifiedWorksetManifestTests(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
        *,
        mismatch_full_tokens: bool = False,
        bad_full_schema: bool = False,
    ) -> list[str]:
        plan_path = root / "plan.json"
        baseline_path = root / "baseline.json"
        full_path = root / "full.json"
        baseline_log = root / "baseline.log"
        full_log = root / "full.log"
        profile = root / "profile.env"
        hook = root / "hook.py"
        runner = root / "runner.py"
        output = root / "manifest.json"

        tool = {
            "event_index": 3,
            "call_index": 0,
            "tool_name": "visit",
            "duration_s": 1.25,
            "offline_saved_s": 0.75,
            "visit_units": [{"url": "https://example.test/"}],
        }
        plan = {
            "schema": "paste_repro.dr_trace_hybrid_plan.v1",
            "summary": {
                "sessions": 1,
                "requests": 1,
                "tools": 1,
                "prompt_tokens": 11,
                "fixed_completion_tokens": 7,
            },
            "traces": [
                {
                    "task_id": "task-1",
                    "steps": [
                        {
                            "request": {
                                "call_index": 0,
                                "prompt_tokens": 11,
                                "fixed_completion_tokens": 7,
                            },
                            "tools_after": [tool],
                        }
                    ],
                }
            ],
        }
        plan["plan_sha256"] = canonical_hash(plan)
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        result_digest = canonical_hash(
            {
                "tool_name": tool["tool_name"],
                "call_index": tool["call_index"],
                "visit_units": tool["visit_units"],
            }
        )

        def result(system: str, completion_tokens: int) -> dict:
            value = {
                "schema": "paste_repro.dr_trace_hybrid_result.v1",
                "system": system,
                "plan": str(plan_path.resolve()),
                "plan_sha256": plan["plan_sha256"],
                "model": "test-model",
                "summary": {
                    "tasks": 1,
                    "successful_tasks": 1,
                    "llm_requests": 1,
                    "tool_calls": 1,
                },
                "tasks": [{"task_id": "task-1", "ok": True}],
                "llm_events": [
                    {
                        "task_id": "task-1",
                        "request_index": 0,
                        "call_index": 0,
                        "http_status": 200,
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": completion_tokens,
                        },
                    }
                ],
                "tool_events": [
                    {
                        "task_id": "task-1",
                        "event_index": 3,
                        "call_index": 0,
                        "tool_name": "visit",
                        "full_service_s": 1.25,
                        "result_sha256": result_digest,
                    }
                ],
            }
            value["result_sha256"] = canonical_hash(value)
            return value

        baseline_path.write_text(
            json.dumps(result("baseline", 7)), encoding="utf-8"
        )
        full_result = result("full", 8 if mismatch_full_tokens else 7)
        if bad_full_schema:
            full_result["schema"] = "unsupported.result.v0"
            full_result.pop("result_sha256")
            full_result["result_sha256"] = canonical_hash(full_result)
        full_path.write_text(json.dumps(full_result), encoding="utf-8")
        common_args = (
            "{'host': '127.0.0.1', 'port': %d, 'model': '/models/test', "
            "'served_model_name': ['test-model'], 'dtype': 'bfloat16', "
            "'max_model_len': 16384, 'tensor_parallel_size': 4, "
            "'gpu_memory_utilization': 0.86, 'max_num_batched_tokens': 2048, "
            "'max_num_seqs': 48, 'enable_prefix_caching': True, "
            "'cuda_graph_sizes': [32], "
            "'api_key': 'must-not-appear'}"
        )
        baseline_log.write_text(
            "non-default args: " + common_args % 8100 + "\n"
            'POST /v1/chat/completions HTTP/1.1" 200\n',
            encoding="utf-8",
        )
        full_log.write_text(
            "[sched_policy_patch] installed policy=online_joint_pacer_v2\n"
            "non-default args: " + common_args % 8200 + "\n"
            'POST /v1/chat/completions HTTP/1.1" 200\n',
            encoding="utf-8",
        )
        profile.write_text(
            'export VLLM_SCHED_POLICY="online_joint_pacer_v2"\n'
            'export PRIVATE_TOKEN="profile-secret-must-not-appear"\n',
            encoding="utf-8",
        )
        hook.write_text("# scheduler hook\n", encoding="utf-8")
        runner.write_text("# paired runner\n", encoding="utf-8")
        return [
            "--baseline",
            str(baseline_path),
            "--full",
            str(full_path),
            "--baseline-server-log",
            str(baseline_log),
            "--full-server-log",
            str(full_log),
            "--profile-config",
            str(profile),
            "--hook",
            str(hook),
            "--runner",
            str(runner),
            "--repo-root",
            str(REPOSITORY_ROOT),
            "--skip-gpu-inventory",
            "--output",
            str(output),
        ]

    def test_valid_pair_records_hashes_without_environment_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            argv = self._write_fixture(Path(temporary))
            args = parse_args(argv)
            manifest = build_manifest(args)

            self.assertTrue(manifest["valid"], manifest["invalid_checks"])
            self.assertEqual(manifest["invalid_checks"], [])
            unsigned = dict(manifest)
            expected = unsigned.pop("manifest_sha256")
            self.assertEqual(expected, canonical_hash(unsigned))
            digests = manifest["validation"]["workload_digests"]
            self.assertEqual(
                digests["plan_llm_token_work_and_status_sha256"],
                digests["full_llm_token_work_and_status_sha256"],
            )
            self.assertEqual(
                digests["plan_tool_trace_and_result_sha256"],
                digests["baseline_tool_trace_and_result_sha256"],
            )
            self.assertIn(
                "dirty_diff_sha256", manifest["provenance"]["git"]
            )
            self.assertIn("sitecustomize", manifest["artifacts"])
            self.assertIn("vllm_launcher", manifest["artifacts"])
            self.assertFalse(manifest["provenance"]["process_environment_recorded"])
            self.assertEqual(
                manifest["provenance"]["runtime_scheduler_environment"]["status"],
                "unknown",
            )
            serialized = json.dumps(manifest)
            self.assertNotIn("profile-secret-must-not-appear", serialized)
            self.assertNotIn("must-not-appear", serialized)
            server_args = manifest["provenance"]["server_argv_evidence"]
            self.assertNotIn(
                "api_key",
                server_args["full"]["vllm_non_default_args_allowlisted"],
            )

    def test_token_mismatch_is_written_as_invalid_and_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = self._write_fixture(root, mismatch_full_tokens=True)
            self.assertEqual(main(argv), 2)
            output = root / "manifest.json"
            self.assertTrue(output.is_file())
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(manifest["valid"])
            self.assertIn(
                "llm_token_work_and_status_match_plan",
                manifest["invalid_checks"],
            )
            self.assertEqual(list(root.glob(".manifest.json.tmp-*")), [])

    def test_valid_cli_write_is_atomic_and_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = self._write_fixture(root)
            self.assertEqual(main(argv), 0)
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["valid"])
            self.assertEqual(list(root.glob(".manifest.json.tmp-*")), [])

    def test_unsupported_result_schema_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = self._write_fixture(root, bad_full_schema=True)
            manifest = build_manifest(parse_args(argv))
            self.assertFalse(manifest["valid"])
            self.assertIn("supported_result_schemas", manifest["invalid_checks"])

    def test_missing_critical_server_argv_key_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = self._write_fixture(root)
            baseline_log = root / "baseline.log"
            baseline_log.write_text(
                baseline_log.read_text(encoding="utf-8").replace(
                    "'dtype': 'bfloat16', ", ""
                ),
                encoding="utf-8",
            )
            manifest = build_manifest(parse_args(argv))
            self.assertFalse(manifest["valid"])
            self.assertIn(
                "server_required_argv_keys_present", manifest["invalid_checks"]
            )


if __name__ == "__main__":
    unittest.main()
