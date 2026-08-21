from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "reproduction/scripts/run_live_joint_formal_matrix.py"
CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_formal_v8_matrix.env.example"
)
LEGACY_V7_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_formal_v7_matrix.env.example"
)
SPEC = importlib.util.spec_from_file_location("run_live_joint_formal_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
formal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_fixture(
    task_index: int,
    *,
    source_id: str | None = None,
    replica: int = 0,
    selected_url: str | None = None,
    search_query: str = "fixture query",
    question: str = "fixture question",
) -> tuple[dict, list[dict]]:
    source_id = source_id or f"formal-v8-{task_index + 1:03d}"
    task_id = f"{source_id}__r{replica:02d}"
    selected_url = selected_url or (
        f"https://en.wikipedia.org/wiki/Formal_{task_index:03d}"
    )
    answer_text = f"Verified answer for source {task_index}."
    answer = {"answer": answer_text, "source_url": selected_url}
    semantic = json.dumps(
        answer,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("/", "\\/")
    padding = " " * 17
    raw = semantic + padding
    answer_sha = _sha(answer_text)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "source_url"],
        "properties": {
            "answer": {"type": "string"},
            "source_url": {"const": selected_url},
        },
    }
    recovery_calls = []
    output_calls = []
    for call_index in range(2):
        raw_sha = _sha(f"guided-{task_index}-{call_index}")
        recovery_calls.append(
            {
                "call_index": call_index,
                "mode": "guided_json",
                "guided_json_requested": True,
                "json_parse_attempted": True,
                "local_wrap_applied": False,
                "contract_succeeded": True,
                "policy_version": formal.GUIDED_JSON_RECOVERY_POLICY_VERSION,
                "recovery_applied": False,
                "parse_succeeded": True,
                "raw_sha256": raw_sha,
            }
        )
        output_calls.append(
            {
                "call_index": call_index,
                "mode": "guided_json",
                "guided_json_requested": True,
                "json_parse_attempted": True,
                "local_wrap_applied": False,
                "contract_succeeded": True,
                "recovery_applied": False,
                "parse_succeeded": True,
                "raw_sha256": raw_sha,
            }
        )
    grammar_sha = formal._fixed_final_grammar_sha256(selected_url)
    final_contract = {
        "call_index": 2,
        "policy_version": formal.FINAL_ANSWER_CONTRACT_POLICY_VERSION,
        "schema_policy_version": formal.FINAL_ANSWER_SCHEMA_POLICY_VERSION,
        "schema_sha256": formal._sha256_json(schema),
        "schema_answer_constraint": "type_only_no_length_or_pattern",
        "mode": "guided_grammar_fixed_completion_strict_raw_decode_local_projection",
        "guided_json_requested": False,
        "guided_grammar_requested": True,
        "json_parse_attempted": True,
        "strict_json_parse": True,
        "strict_json_raw_decode": True,
        "recovery_allowed": False,
        "recovery_applied": False,
        "parse_succeeded": True,
        "local_wrap_applied": True,
        "local_projection_applied": False,
        "object_constructed_locally": True,
        "model_source_url_validated": True,
        "source_url_binding": "exact_committed_selected_url",
        "source_url_sha256": _sha(selected_url),
        "contract_succeeded": True,
        "raw_sha256": _sha(raw),
        "raw_char_count": len(raw),
        "semantic_sha256": _sha(semantic),
        "semantic_char_count": len(semantic),
        "semantic_byte_count": len(semantic.encode("utf-8")),
        "padding_sha256": _sha(padding),
        "padding_char_count": len(padding),
        "padding_byte_count": len(padding),
        "tail_nonempty": True,
        "tail_ascii_space_only": True,
        "tail_validation_succeeded": True,
        "grammar_policy_version": formal.FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
        "grammar_xgrammar_version": formal.FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
        "grammar_sha256": grammar_sha,
        "grammar_semantic_json_whitespace": "compact",
        "tail_policy": "one_or_more_ascii_spaces_only",
        "fixed_completion_tokens": 192,
        "min_tokens": 192,
        "max_tokens": 192,
        "total_completion_tokens": 192,
        "semantic_token_count": 40,
        "padding_token_count": 152,
        "token_partition_method": "server_total_minus_local_semantic_tokenization",
        "token_counter_method": "transformers_chat_template",
        "token_accounting_succeeded": True,
        "finish_reason": "length",
        "finish_reason_validated": True,
        "max_chars": 480,
        "max_words": 60,
        "target_chars": 360,
        "model_answer_sha256": answer_sha,
        "model_answer_char_count": len(answer_text),
        "pre_projection_canonical_sha256": answer_sha,
        "pre_projection_char_count": len(answer_text),
        "pre_projection_word_count": len(answer_text.split(" ")),
        "canonical_sha256": answer_sha,
        "canonicalization_changed": False,
        "canonical_char_count": len(answer_text),
        "canonical_word_count": len(answer_text.split(" ")),
        "word_projection_applied": False,
        "char_projection_applied": False,
    }
    task = {
        "ok": True,
        "task_id": task_id,
        "source_id": source_id,
        "replica": replica,
        "search_query": search_query,
        "question_sha256": _sha(question),
        "visit_canary": task_index % 6 == 0,
        "context_padding_target_tokens": 10000,
        "expected_url": selected_url,
        "selected_url": selected_url,
        "answer": answer,
        "answer_sha256": formal._sha256_json(answer),
        "prompt_tokens": 33,
        "completion_tokens": 203,
        "tools": [
            {
                "invocation": {
                    "tool_name": "visit",
                    "arguments": {"url": [selected_url], "goal": "fixture"},
                }
            }
        ],
        "guided_json_recovery": {
            "policy_version": formal.GUIDED_JSON_RECOVERY_POLICY_VERSION,
            "parsed_call_count": 2,
            "recovery_count": 0,
            "calls": recovery_calls,
        },
        "output_contract": {
            "policy_version": formal.OUTPUT_CONTRACT_POLICY_VERSION,
            "calls": [*output_calls, dict(final_contract)],
        },
        "final_answer_contract": final_contract,
    }
    events = []
    for call_index, (prompt_tokens, completion_tokens) in enumerate(
        ((10, 5), (11, 6), (12, 192))
    ):
        fixed = call_index == 2
        events.append(
            {
                "task_id": task_id,
                "call_index": call_index,
                "request_id": f"request-{task_id}-{call_index}",
                "ok": True,
                "attempts": 1,
                "http_status": 200,
                "output_mode": "guided_grammar" if fixed else "guided_json",
                "guided_json_requested": not fixed,
                "guided_grammar_requested": fixed,
                "guided_grammar_sha256": grammar_sha if fixed else None,
                "min_tokens": 192 if fixed else 0,
                "max_tokens": 192 if fixed else 128,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "finish_reason": "length" if fixed else "stop",
                "response_sha256": _sha(raw) if fixed else _sha("tool"),
            }
        )
    return task, events


def _tool_record(tool: str, index: int) -> dict:
    started = 10.0 + index * 2.1 if tool == "visit" else 1.0 + index * 0.01
    canary = tool == "visit" and index % 6 == 0
    return {
        "tool": tool,
        "admitted": True,
        "authoritative": True,
        "speculative": False,
        "speculation_eligible": not canary,
        "canary": canary,
        "committed": True,
        "cancelled": False,
        "start": started,
        "started_at": started,
        "http_attempts": 1,
        "http_attempt_log": [
            {
                "started_monotonic_s": started,
                "start_gate_wait_s": 0.0,
                "retry_backoff_s": 0.0,
            }
        ],
        "outcome": "committed",
        "service_s": 0.2,
        "backend": "r.jina.ai" if tool == "visit" else "bing_html_search",
        "request_host": "r.jina.ai" if tool == "visit" else "www.bing.com",
        "transport_identity_source": "actual",
        "response_status": 200,
        "bytes_read": 256,
    }


def _result_fixture(cell: str = "A") -> dict:
    policy, speculation = formal.CELL_POLICY[cell]
    tasks = []
    events = []
    for index in range(80):
        task, task_events = _task_fixture(index)
        tasks.append(task)
        events.extend(task_events)
    scheduler = {
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
        "VLLM_PORT": "8100",
        "VLLM_MAX_MODEL_LEN": "16384",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_MAX_NUM_SEQS": "96",
        "VLLM_ENABLE_PREFIX_CACHING": "1",
        "VLLM_USE_V1": "1",
        "VLLM_SCHED_POLICY": policy,
    }
    if policy != "fcfs":
        scheduler.update(
            {
                key: formal.EXPECTED_CONFIG[key]
                for key in formal.FROZEN_JOINT_SCHEDULER_ENV_KEYS
            }
        )
    config = {
        "call_graph_mode": "frozen",
        "speculation_mode": speculation,
        "tool_signal_policy": "execution_aware",
        "tool_signal_policy_version": "exact-session-invocation-running-completed-v1",
        "tool_signal_policy_module_sha256": formal.LIVE_AGENT_SHA256,
        "independent_source_count": 80,
        "replicas": 1,
        "task_count": 80,
        "max_active_tasks": 80,
        "tool_workers": 4,
        "speculative_tool_workers": 2,
        "min_speculative_tool_workers": 0,
        "search_tool_capacity": 3,
        "visit_tool_capacity": 2,
        "search_min_start_interval_s": 0.0,
        "visit_min_start_interval_s": 2.1,
        "max_speculative_pending": 128,
        "speculative_ttl_s": 120.0,
        "tool_http_max_attempts": 2,
        "tool_http_retry_backoff_s": 1.0,
        "tool_http_attempt_start_gate_enabled": True,
        "tool_http_attempt_start_gate_policy_version": "shared-per-tool-monotonic-v1",
        "tool_http_attempt_min_start_intervals_s": {"visit": 2.1},
        "tool_http_retry_policy_version": "idempotent-get-v1",
        "tool_http_retryable_statuses": [429, 500, 502, 503, 504],
        "tool_http_retryable_exception_types": [
            "asyncio.TimeoutError",
            "ConnectionError",
            "aiohttp.ClientConnectionError",
            "aiohttp.ClientPayloadError",
        ],
        "tool_http_library_retry_disabled": True,
        "tool_http_library_retry_control_version": "aiohttp-private-retry-connection-v1",
        "tool_http_library_name": "aiohttp",
        "tool_http_library_version": "3.12.15",
        "visit_mode": "jina",
        "search_mode": "bing",
        "visit_canary_stride": 6,
        "context_padding_tokens": 10000,
        "fixed_final_completion_tokens": 192,
        "fixed_final_completion_enabled": True,
        "final_answer_contract_policy_version": formal.FINAL_ANSWER_CONTRACT_POLICY_VERSION,
        "final_answer_schema_policy_version": formal.FINAL_ANSWER_SCHEMA_POLICY_VERSION,
        "final_answer_grammar_policy_version": formal.FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
        "final_answer_grammar_xgrammar_version": formal.FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
        "output_contract_policy_version": formal.OUTPUT_CONTRACT_POLICY_VERSION,
        "live_agent_sha256": formal.LIVE_AGENT_SHA256,
        "tool_call_prompt_encoding": "canonical_json_sort_keys_compact",
        "token_count_method": "transformers_chat_template",
        "live_tool_execution": True,
        "recorded_tool_sleep": False,
        "controlled_http_retry": True,
        "shared_bounded_tool_pool": True,
        "authoritative_and_speculative_share_capacity": True,
        "tool_metadata_is_causal": True,
        "tool_result_private_until_exact_commit": True,
        "future_trace_oracle_used": False,
        "frozen_url_is_workload_input": True,
        "workload_file_sha256": formal.FORMAL_WORKLOAD_SHA256,
        "workload_split_id": "live-joint-wikipedia-frozen-formal-v8",
        "workload_split_role": "formal_heldout",
        "workload_formal_eligible": True,
        "scheduler_environment": scheduler,
        "formal_run": {
            "block_id": "block-1",
            "cell_id": cell,
            "order_index": 0,
            "server_instance_id": "server-1",
            "fresh_server": True,
            "result_cache_empty": True,
            "broker_drained": True,
        },
    }
    return {
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
                    "commits": 160,
                    "authoritative_failures": 0,
                }
            },
        },
        "tasks": tasks,
        "llm_events": events,
        "tool_attempt_records": [
            *(_tool_record("search", index) for index in range(80)),
            *(_tool_record("visit", index) for index in range(80)),
        ],
    }


class FrozenConfigTests(unittest.TestCase):
    def test_v8_frozen_config_is_exact_and_v7_cannot_drive_wrapper(self) -> None:
        observed = formal.load_frozen_config(CONFIG)
        self.assertEqual(observed, formal.EXPECTED_CONFIG)
        self.assertEqual(observed["PASTE_LIVE_FORMAL_SOURCE_COUNT"], "80")
        self.assertEqual(observed["PASTE_LIVE_MAX_ACTIVE_TASKS"], "80")
        self.assertGreater(int(observed["PASTE_LIVE_MAX_ACTIVE_TASKS"]), 64)
        self.assertLess(
            int(observed["PASTE_LIVE_MAX_ACTIVE_TASKS"]),
            int(observed["VLLM_MAX_NUM_SEQS"]),
        )
        self.assertEqual(observed["PASTE_LIVE_FIXED_FINAL_COMPLETION_TOKENS"], "192")
        self.assertEqual(observed["PASTE_LIVE_VLLM_LIBRARY_VERSION"], "0.10.1")
        self.assertEqual(
            observed["PASTE_LIVE_TRANSFORMERS_LIBRARY_VERSION"], "4.56.1"
        )
        with self.assertRaisesRegex(formal.FormalRunError, "mismatch"):
            formal.load_frozen_config(LEGACY_V7_CONFIG)

    def test_changed_or_executable_config_fails_closed(self) -> None:
        changed = CONFIG.read_text(encoding="utf-8").replace(
            'export PASTE_LIVE_MAX_ACTIVE_TASKS="80"',
            'export PASTE_LIVE_MAX_ACTIVE_TASKS="64"',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.env"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(formal.FormalRunError, "changed"):
                formal.load_frozen_config(path)
            path.write_text(changed + "\nexport BAD=$(touch /tmp/nope)\n")
            with self.assertRaisesRegex(formal.FormalRunError, "literal export"):
                formal.load_frozen_config(path)


class OrderAndEnvironmentTests(unittest.TestCase):
    def test_orders_and_cell_environment_are_frozen(self) -> None:
        self.assertEqual(
            formal.validate_orders(
                "A,B,E,F;B,A,F,E;A,B,F,E", baseline_only=False
            ),
            [["A", "B", "E", "F"], ["B", "A", "F", "E"], ["A", "B", "F", "E"]],
        )
        config = formal.load_frozen_config(CONFIG)
        a_env = formal._cell_environment(
            config,
            cell="A",
            inherited={"VLLM_SCHED_STALE": "bad", "KEEP": "yes"},
        )
        self.assertEqual(a_env["VLLM_SCHED_POLICY"], "fcfs")
        self.assertFalse(
            any(
                key.startswith("VLLM_SCHED_") and key != "VLLM_SCHED_POLICY"
                for key in a_env
            )
        )
        f_env = formal._cell_environment(config, cell="F", inherited={})
        self.assertEqual(f_env["VLLM_SCHED_POLICY"], "online_joint_pacer_v2")
        self.assertEqual(
            {key for key in f_env if key.startswith("VLLM_SCHED_")},
            {*formal.FROZEN_JOINT_SCHEDULER_ENV_KEYS, "VLLM_SCHED_POLICY"},
        )


class ResultContractTests(unittest.TestCase):
    def test_v8_result_contract_accepts_exact_fixture_and_rejects_drift(self) -> None:
        result = _result_fixture("A")
        formal.validate_cell_result(
            result,
            cell="A",
            block_id="block-1",
            order_index=0,
            server_instance_id="server-1",
        )
        event = result["llm_events"][2]
        event["usage"]["completion_tokens"] = 191
        with self.assertRaisesRegex(formal.FormalRunError, "fixed-final"):
            formal.validate_cell_result(
                result,
                cell="A",
                block_id="block-1",
                order_index=0,
                server_instance_id="server-1",
            )
        event["usage"]["completion_tokens"] = 192
        contract = result["tasks"][0]["final_answer_contract"]
        mirrored = result["tasks"][0]["output_contract"]["calls"][2]
        contract["padding_sha256"] = "0" * 64
        mirrored["padding_sha256"] = "0" * 64
        with self.assertRaisesRegex(formal.FormalRunError, "ASCII-space tail"):
            formal.validate_cell_result(
                result,
                cell="A",
                block_id="block-1",
                order_index=0,
                server_instance_id="server-1",
            )

    def test_runner_command_binds_live_fixed_final_controls(self) -> None:
        config = formal.load_frozen_config(CONFIG)
        command = formal._runner_command(
            python=Path(config["PASTE_ENV_PREFIX"]) / "bin/python",
            workload=REPOSITORY_ROOT / config["PASTE_LIVE_FORMAL_WORKLOAD"],
            output=REPOSITORY_ROOT / "unused",
            cell="F",
            block_id="block-1",
            order_index=0,
            server_instance_id="server-1",
            config=config,
        )
        rendered = " ".join(command)
        self.assertIn("--fixed-final-completion-tokens 192", rendered)
        self.assertIn("--max-active-tasks 80", rendered)
        self.assertIn("--replicas 1", rendered)
        self.assertIn("--visit-canary-stride 6", rendered)
        self.assertIn("--speculation-mode visit", rendered)
        self.assertNotIn("--source-limit", command)


class BaselineGateTests(unittest.TestCase):
    def _write_evidence(
        self,
        root: Path,
        *,
        pressure_count: int = 10,
        pressure_gap_after: int | None = None,
    ) -> tuple[Path, Path]:
        result_path = root / "result.json"
        timeline_path = root / "queue_timeline.jsonl"
        result = _result_fixture("A")
        # The load gate consumes only config/summary; keep this unit artifact small.
        result = {"config": result["config"], "summary": result["summary"]}
        result_path.write_text(json.dumps(result), encoding="utf-8")
        rows = []
        monotonic = 100.0
        wall = 1000.0
        for index in range(40):
            if pressure_gap_after is not None and index == pressure_gap_after:
                monotonic += 3.0
                wall += 3.0
            pressure = index < pressure_count
            rows.append(
                {
                    "monotonic_s": monotonic,
                    "wall_s": wall,
                    "llm_running": 12,
                    "llm_waiting": 1 if pressure else 0,
                    "tool_queued_authoritative": 1 if pressure else 0,
                }
            )
            monotonic += 0.2
            wall += 0.2
        timeline_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return result_path, timeline_path

    def test_gate_requires_80_above_64_and_true_continuous_pressure(self) -> None:
        artifacts = REPOSITORY_ROOT / "reproduction/artifacts"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
            result, timeline = self._write_evidence(Path(temporary))
            gate = formal.evaluate_baseline_gate(
                result, timeline, block_id="block-1"
            )
        self.assertTrue(gate["accepted"])
        self.assertEqual(
            gate["observed"]["simultaneous_llm_and_tool_queue_sample_count"], 10
        )
        self.assertGreaterEqual(
            gate["observed"]["longest_consecutive_simultaneous_queue_span_s"],
            1.0,
        )
        self.assertEqual(gate["thresholds"]["offered_concurrency_must_exceed"], 64)
        self.assertNotIn("task_e2e", json.dumps(gate))

    def test_three_second_sampling_hole_breaks_continuity(self) -> None:
        artifacts = REPOSITORY_ROOT / "reproduction/artifacts"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
            result, timeline = self._write_evidence(
                Path(temporary), pressure_gap_after=5
            )
            gate = formal.evaluate_baseline_gate(
                result, timeline, block_id="block-1"
            )
        self.assertFalse(gate["accepted"])
        self.assertEqual(gate["observed"]["simultaneous_gap_reset_count"], 1)
        self.assertGreater(
            gate["observed"]["maximum_adjacent_simultaneous_monotonic_gap_s"],
            0.5,
        )
        self.assertFalse(
            gate["checks"][
                "continuous_simultaneous_llm_and_tool_queue_span_at_least_1s"
            ]
        )


class OfflineCheckTests(unittest.TestCase):
    def test_check_only_compiles_all_80_grammars_without_output(self) -> None:
        tag = "unit-v8-check-" + uuid.uuid4().hex
        output = REPOSITORY_ROOT / "reproduction/artifacts/live_joint/formal" / tag
        completed = subprocess.run(
            [
                "/home/aiscuser/.conda/envs/paste/bin/python",
                str(SCRIPT),
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
        self.assertFalse(payload["gpu_or_server_touched"])
        self.assertEqual(payload["workload_validation"]["source_count"], 80)
        feasibility = payload["fixed_final_grammar_feasibility"]
        self.assertEqual(feasibility["source_count"], 80)
        self.assertEqual(feasibility["fixed_final_completion_tokens"], 192)
        self.assertEqual(feasibility["space_token_id"], 220)
        self.assertEqual(feasibility["vllm_version"], "0.10.1")
        self.assertEqual(feasibility["transformers_version"], "4.56.1")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
