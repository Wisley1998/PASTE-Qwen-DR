from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from compare_live_joint_dev_triplet import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXPECTED_AUTHORITATIVE_COMMIT_COUNT,
    EXPECTED_LLM_REQUEST_COUNT,
    EXPECTED_REPLICAS,
    EXPECTED_SOURCE_COUNT,
    EXPECTED_TASK_COUNT,
    _audit_canary_non_speculation,
    _audit_http_attempt_logs,
    _bootstrap_effect,
    _config_pair_audit,
    _task_components,
    _token_comparison,
    _validate_exact_counts,
)


def _attempt(
    started: float,
    *,
    attempt: int = 1,
    status: int | None = 200,
    error_type: str | None = None,
    retried: bool = False,
    retry_backoff_s: float = 0.0,
) -> dict:
    return {
        "request_index": 0,
        "attempt": attempt,
        "status": status,
        "error_type": error_type,
        "retried": retried,
        "started_monotonic_s": started,
        "start_gate_wait_s": 0.01,
        "retry_backoff_s": retry_backoff_s,
    }


def _physical_record(
    invocation_id: str,
    started: float,
    *,
    tool: str = "visit",
    attempts: list[dict] | None = None,
) -> dict:
    attempt_rows = [_attempt(started)] if attempts is None else attempts
    return {
        "invocation_id": invocation_id,
        "tool": tool,
        "started_at": started - 0.01,
        "finished_at": max(row["started_monotonic_s"] for row in attempt_rows)
        + 0.5,
        "http_attempts": len(attempt_rows),
        "http_attempt_log": attempt_rows,
    }


def _http_run(records: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "tool_http_attempt_start_gate_enabled": True,
            "tool_http_attempt_start_gate_policy_version": (
                "shared-per-tool-monotonic-v1"
            ),
            "tool_http_attempt_min_start_intervals_s": {"visit": 2.1},
            "search_min_start_interval_s": 0.0,
            "visit_min_start_interval_s": 2.1,
            "tool_http_retryable_statuses": [429, 500, 502, 503, 504],
            "tool_http_retryable_exception_types": [
                "asyncio.TimeoutError",
                "ConnectionError",
                "aiohttp.ClientConnectionError",
                "aiohttp.ClientPayloadError",
            ],
            "tool_http_retry_backoff_s": 1.0,
        },
        physical_records=tuple(records),
    )


def _config_run(
    *,
    label: str,
    speculation_mode: str,
    policy: str,
    module_sha: str = "a" * 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "cell_label": label,
            "speculation_mode": speculation_mode,
            "expected_url_search_coverage": {"matched_task_count": 47},
            "tool_signal_policy_module_sha256": module_sha,
            "task_count": 48,
            "scheduler_environment": {
                "MODEL_ID": "test/model",
                "VLLM_MAX_MODEL_LEN": "16384",
                "VLLM_SCHED_POLICY": policy,
                "VLLM_SCHED_JOINT_V2_TOOL_BETA": (
                    "0.9" if policy == "online_joint_pacer_v2" else None
                ),
            },
        }
    )


def _component_row(e2e: float, llm: float, tool: float) -> dict[str, float]:
    return {
        "e2e_s": e2e,
        "llm_s": llm,
        "tool_exposed_s": tool,
        "search_exposed_s": 0.2,
        "visit_exposed_s": tool - 0.2,
        "orchestration_residual_s": e2e - llm - tool,
    }


class CompareLiveJointDevTripletTests(unittest.TestCase):
    def test_http_attempt_audit_accepts_complete_logs_and_physical_spacing(self) -> None:
        run = _http_run(
            [
                _physical_record("v1", 10.0),
                _physical_record("v2", 12.1),
                _physical_record("s1", 10.1, tool="search"),
                _physical_record("s2", 10.2, tool="search"),
            ]
        )
        audit = _audit_http_attempt_logs(run)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["wire_http_attempt_count"], 4)
        self.assertAlmostEqual(
            audit["spacing"]["visit"]["minimum_observed_start_delta_s"],
            2.1,
        )

    def test_http_attempt_audit_rejects_missing_log_and_spacing_violation(self) -> None:
        records = [
            _physical_record("v1", 10.0),
            _physical_record("v2", 11.0),
            _physical_record("v3", 11.5),
        ]
        records[0]["http_attempt_log"] = None
        audit = _audit_http_attempt_logs(_http_run(records))
        self.assertFalse(audit["passed"])
        messages = "\n".join(audit["errors"]["first"])
        self.assertIn("without a non-empty http_attempt_log", messages)
        self.assertIn("minimum interval", messages)

    def test_http_attempt_audit_accepts_one_controlled_retry(self) -> None:
        attempts = [
            _attempt(
                10.0,
                status=429,
                error_type="aiohttp.ClientResponseError",
                retried=True,
                retry_backoff_s=1.0,
            ),
            _attempt(12.1, attempt=2),
        ]
        run = _http_run([_physical_record("v1", 10.0, attempts=attempts)])
        # aiohttp.ClientResponseError is represented by its retryable 429
        # status, so it need not also be in the exception allowlist.
        audit = _audit_http_attempt_logs(run)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["retried_physical_job_count"], 1)
        self.assertEqual(audit["wire_http_attempt_count"], 2)

    def test_canary_audit_requires_zero_physical_speculation(self) -> None:
        tasks = {
            f"s{index}__r00": {"visit_canary": index in {0, 3}}
            for index in range(6)
        }
        committed = {
            (task_id, "visit"): {
                "session_id": task_id,
                "tool": "visit",
                "speculative": False,
                "speculation_eligible": False,
            }
            for task_id, task in tasks.items()
            if task["visit_canary"]
        }
        run = SimpleNamespace(
            config={"visit_canary_stride": 3, "speculation_mode": "visit"},
            tasks_by_id=tasks,
            committed_by_task_tool=committed,
            physical_records=(
                {
                    "session_id": "s1__r00",
                    "tool": "visit",
                    "speculative": True,
                },
            ),
        )
        self.assertTrue(_audit_canary_non_speculation(run)["passed"])
        run.physical_records += (
            {
                "session_id": "s0__r00",
                "tool": "visit",
                "speculative": True,
            },
        )
        audit = _audit_canary_non_speculation(run)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["canary_speculative_physical_record_count"], 1)

    def test_config_allowlist_accepts_treatments_but_not_module_sha_drift(self) -> None:
        n = _config_run(
            label="n-r1",
            speculation_mode="off",
            policy="online_joint_pacer_v2",
        )
        v = _config_run(
            label="v-r1",
            speculation_mode="visit",
            policy="online_joint_pacer_v2",
        )
        self.assertTrue(
            _config_pair_audit(n, v, left_cell="N", right_cell="V")["passed"]
        )
        changed = deepcopy(v)
        changed.config["tool_signal_policy_module_sha256"] = "b" * 64
        audit = _config_pair_audit(
            n, changed, left_cell="N", right_cell="V"
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["uncontrolled_top_level_differences"],
            ["tool_signal_policy_module_sha256"],
        )

    def test_config_allowlist_accepts_only_scheduler_keys_for_a_to_n(self) -> None:
        a = _config_run(label="a", speculation_mode="off", policy="fcfs")
        n = _config_run(
            label="n",
            speculation_mode="off",
            policy="online_joint_pacer_v2",
        )
        audit = _config_pair_audit(a, n, left_cell="A", right_cell="N")
        self.assertTrue(audit["passed"])
        n.config["scheduler_environment"]["MODEL_ID"] = "different/model"
        audit = _config_pair_audit(a, n, left_cell="A", right_cell="N")
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["non_scheduler_runtime_environment_differences"], ["MODEL_ID"]
        )

    def test_exact_48_144_96_contract_is_fail_closed(self) -> None:
        task_keys = {
            (f"source{source}", replica): {}
            for source in range(EXPECTED_SOURCE_COUNT)
            for replica in range(EXPECTED_REPLICAS)
        }
        llm = {
            f"task{index}": ({}, {}, {}) for index in range(EXPECTED_TASK_COUNT)
        }
        commits = {
            (f"task{index}", tool): {}
            for index in range(EXPECTED_TASK_COUNT)
            for tool in ("search", "visit")
        }
        run = SimpleNamespace(
            tasks_by_key=task_keys,
            llm_by_task=llm,
            committed_by_task_tool=commits,
            config={"replicas": EXPECTED_REPLICAS},
            payload={
                "summary": {
                    "successful_task_count": EXPECTED_TASK_COUNT,
                    "failed_task_count": 0,
                    "llm": {
                        "successful_request_count": EXPECTED_LLM_REQUEST_COUNT,
                        "exactly_one_attempt_each": True,
                    },
                    "tool": {
                        "authoritative_commit_count": (
                            EXPECTED_AUTHORITATIVE_COMMIT_COUNT
                        )
                    },
                }
            },
        )
        self.assertTrue(_validate_exact_counts(run)["passed"])
        run.committed_by_task_tool.pop(next(iter(run.committed_by_task_tool)))
        self.assertFalse(_validate_exact_counts(run)["passed"])

    def test_source_bootstrap_is_fixed_and_uses_source_unit(self) -> None:
        baseline = {
            "s1": _component_row(10.0, 4.0, 5.8),
            "s2": _component_row(12.0, 5.0, 6.8),
            "s3": _component_row(14.0, 6.0, 7.8),
        }
        candidate = {
            source: _component_row(
                values["e2e_s"] - 2.0,
                values["llm_s"] - 0.5,
                values["tool_exposed_s"] - 1.5,
            )
            for source, values in baseline.items()
        }
        first = _bootstrap_effect(baseline, candidate, resamples=100)
        second = _bootstrap_effect(baseline, candidate, resamples=100)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], BOOTSTRAP_SEED)
        self.assertEqual(first["sample_size"], 3)
        self.assertEqual(
            first["sampling_unit"],
            "independent_source_mean_over_runs_and_replicas",
        )
        self.assertGreater(first["e2e_relative_reduction_95_ci"][0], 0.0)

    def test_token_gate_flags_1_573_percent_but_mechanism_direction_is_tool(self) -> None:
        result = _token_comparison(
            {
                "prompt_tokens": 1_540_579,
                "completion_tokens": 6_802,
                "total_tokens": 1_547_381,
            },
            {
                "prompt_tokens": 1_540_572,
                "completion_tokens": 6_695,
                "total_tokens": 1_547_267,
            },
            task_count=48,
            configured_decode_tokens_per_s=113.7,
            observed_e2e_reduction_s=3.816,
            llm_component_reduction_s=-2.049,
            tool_component_reduction_s=5.864,
        )
        self.assertFalse(result["balance_gate"]["passed"])
        self.assertAlmostEqual(
            result["metrics"]["completion_tokens"]["relative_delta"],
            -107 / 6802,
        )
        self.assertAlmostEqual(
            result["completion_shortfall_decode_time_equivalent_s_per_task"],
            107 / 48 / 113.7,
        )
        self.assertTrue(
            result["token_shortfall_direction_cannot_explain_observed_e2e_gain"]
        )

    def test_e2e_component_accounting_uses_authoritative_exposed_wait(self) -> None:
        task_id = "source__r00"
        run = SimpleNamespace(
            tasks_by_key={
                ("source", 0): {
                    "task_id": task_id,
                    "e2e_s": 10.0,
                    "llm_duration_s": 4.0,
                }
            },
            committed_by_task_tool={
                (task_id, "search"): {"exposed_wait_s": 1.0},
                (task_id, "visit"): {"exposed_wait_s": 4.8},
            },
        )
        row = _task_components(run)[("source", 0)]
        self.assertAlmostEqual(row["tool_exposed_s"], 5.8)
        self.assertAlmostEqual(row["orchestration_residual_s"], 0.2)


if __name__ == "__main__":
    unittest.main()
