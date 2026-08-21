from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from aggregate_live_joint_holdout_blocks import (  # noqa: E402
    EXPECTED_REPLICAS,
    EXPECTED_SOURCE_COUNT,
    _aggregate_source_components,
    _block_source_components,
    _canary_comparison,
    _server_log_audit,
    _speculation_summary,
)


def _audit_with_tasks(offset: float) -> SimpleNamespace:
    tasks = {}
    commits = {}
    for source in range(EXPECTED_SOURCE_COUNT):
        for replica in range(EXPECTED_REPLICAS):
            task_id = f"s{source}__r{replica:02d}"
            tasks[(f"s{source}", replica)] = {
                "task_id": task_id,
                "e2e_s": offset + source + replica,
                "llm_duration_s": 2.0,
            }
            commits[(task_id, "search")] = {"exposed_wait_s": 0.5}
            commits[(task_id, "visit")] = {"exposed_wait_s": 0.3}
    return SimpleNamespace(
        label="test",
        run=SimpleNamespace(
            tasks_by_key=tasks,
            committed_by_task_tool=commits,
        ),
    )


def _canary_audit(values: tuple[float, float]) -> SimpleNamespace:
    commits = {}
    for index, exposed in enumerate(values):
        commits[(f"task{index}", "visit")] = {
            "tool": "visit",
            "canary": True,
            "exposed_wait_s": exposed,
            "queue_s": exposed - 0.1,
            "service_s": 0.2,
        }
    return SimpleNamespace(
        run=SimpleNamespace(committed_by_task_tool=commits)
    )


class AggregateLiveJointHoldoutBlocksTests(unittest.TestCase):
    def test_block_folding_averages_replicas_before_source_sampling(self) -> None:
        folded = _block_source_components(_audit_with_tasks(10.0))
        self.assertEqual(len(folded), EXPECTED_SOURCE_COUNT)
        self.assertAlmostEqual(folded["s0"]["e2e_s"], 10.5)
        self.assertAlmostEqual(folded["s23"]["e2e_s"], 33.5)

    def test_aggregate_folding_averages_block_means_inside_source(self) -> None:
        first = _block_source_components(_audit_with_tasks(10.0))
        second = _block_source_components(_audit_with_tasks(14.0))
        aggregate = _aggregate_source_components(
            {"block1": first, "block2": second}
        )
        self.assertAlmostEqual(aggregate["s0"]["e2e_s"], 12.5)
        self.assertAlmostEqual(aggregate["s23"]["e2e_s"], 35.5)

    def test_aggregate_folding_rejects_block_source_identity_drift(self) -> None:
        first = _block_source_components(_audit_with_tasks(10.0))
        second = _block_source_components(_audit_with_tasks(14.0))
        second.pop("s23")
        with self.assertRaisesRegex(ValueError, "source identities differ"):
            _aggregate_source_components({"block1": first, "block2": second})

    def test_canary_comparison_reports_non_regression_gates(self) -> None:
        result = _canary_comparison(
            [_canary_audit((10.0, 12.0))],
            [_canary_audit((8.0, 9.0))],
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["count_per_treatment"], 2)
        self.assertLess(
            result["metrics"]["exposed_wait_s"]["mean_ratio"], 1.0
        )

    def test_speculation_summary_uses_only_enabled_tool_for_hit_rate(self) -> None:
        commits = {}
        for index in range(4):
            commits[(f"task{index}", "search")] = {
                "tool": "search",
                "speculation_eligible": True,
            }
            commits[(f"task{index}", "visit")] = {
                "tool": "visit",
                "speculation_eligible": True,
            }
        tool_summary = {
            "speculative_admitted_count": 4,
            "exact_hit_count": 4,
            "queued_promotion_count": 3,
            "running_promotion_count": 0,
            "completed_reuse_count": 1,
            "saved_service_s": 1.0,
            "cancelled_physical_count": 0,
            "expired_physical_count": 0,
            "rejected_physical_count": 0,
            "started_physical_job_count": 8,
            "physical_http_attempt_count": 8,
            "retried_physical_job_count": 0,
            "physical_service_s": 8.0,
            "wasted_speculative_service_s_from_records": 0.0,
            "wasted_speculative_service_s_broker": 0.0,
        }
        audit = SimpleNamespace(
            run=SimpleNamespace(
                config={"speculation_mode": "visit"},
                committed_by_task_tool=commits,
                summary={"tool": tool_summary},
            )
        )
        result = _speculation_summary([audit])
        self.assertEqual(
            result["eligible_authoritative_commit_count_all_tools"], 8
        )
        self.assertEqual(
            result[
                "eligible_authoritative_commit_count_for_enabled_speculation_mode"
            ],
            4,
        )
        self.assertEqual(result["exact_hit_rate"], 1.0)
        self.assertEqual(result["wasted_worker_fraction"], 0.0)

    def test_fresh_server_log_requires_one_api_pid_and_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "cell" / "result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{}\n", encoding="utf-8")
            log_path = root / "server" / "vllm_8100.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                "\n".join(
                    (
                        "(APIServer pid=123) vLLM API server version 0.10.1",
                        "[sched_policy_patch] installed policy=online_joint_pacer_v2",
                        "Resolved architecture: Qwen3MoeForCausalLM",
                        "Using max model len 16384",
                    )
                ),
                encoding="utf-8",
            )
            audit = SimpleNamespace(run=SimpleNamespace(path=result_path))
            result = _server_log_audit(audit)
            self.assertTrue(result["passed"])
            self.assertEqual(result["api_server_pids"], ["123"])


if __name__ == "__main__":
    unittest.main()
