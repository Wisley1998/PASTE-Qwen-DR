from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "reproduction/scripts/run_scheduler_robustness.py"
HIGH_SUMMARY = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-high-r1/summary.json"
)
TARGET_SUMMARY = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-target-r3/summary.json"
)
CENTER_SUMMARY = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/reviewer_comment3_live/center093/summary.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("scheduler_robustness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SchedulerRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    @staticmethod
    def _clean_tool_records() -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for index in range(160):
            tool = "search" if index < 80 else "visit"
            records.append(
                {
                    "authoritative": True,
                    "speculative": False,
                    "committed": True,
                    "outcome": "committed",
                    "http_attempts": 1,
                    "response_status": 200,
                    "transport_identity_source": "actual",
                    "tool": tool,
                    "backend": (
                        "bing_html_search" if tool == "search" else "r.jina.ai"
                    ),
                    "request_host": (
                        "www.bing.com" if tool == "search" else "r.jina.ai"
                    ),
                    "http_attempt_log": [
                        {
                            "attempt": 1,
                            "status": 200,
                            "retried": False,
                            "started_monotonic_s": (
                                float(index)
                                if tool == "search"
                                else 1_000.0 + 3.0 * (index - 80)
                            ),
                        }
                    ],
                }
            )
        return records

    def test_exact_decomposition_matches_production_score(self) -> None:
        hook = self.module._load_hook()
        profile = self.module.PROXY_PROFILES[1]
        result = self.module._evaluate_state(
            hook, profile, "mixed", 1.0, 0.70
        )
        self.assertLessEqual(result["max_formula_error_s"], 1e-9)

    def test_factorial_matrix_has_all_registered_dimensions(self) -> None:
        hook = self.module._load_hook()
        rows = self.module._factorial_sweep(hook)
        self.assertEqual(len(rows), 3 * 4 * 3 * 3)
        self.assertEqual({row["candidate_count"] for row in rows}, {18})
        self.assertTrue(
            all(
                row["physical_admission"]["decision"] == "admit"
                for row in rows
            )
        )

    def test_report_template_documents_only_bounded_live_suites(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Only the bounded `target` and `high` suites", source)
        self.assertIn("comment3-high-r1 --suite high", source)
        self.assertIn("Its four observed prefix cells are excluded", source)
        self.assertNotIn("The all suite additionally runs", source)
        self.assertNotIn("48–96 min` for all", source)
        self.assertIn("Prefix performance loaded/reported", source)
        self.assertIn("comment3-high-r1/summary.json", source)

    def test_completed_high_summary_binds_shape_failure_and_recomputes(self) -> None:
        extracted = self.module._external_live_sensitivity_summary(HIGH_SUMMARY)
        self.assertEqual(extracted["run_tag"], "comment3-high-r1")
        self.assertEqual(extracted["suite"], "high")
        self.assertEqual(
            extracted["execution_order"],
            ["a-c12k-l80", "e-c12k-l80-u093"],
        )
        effect = extracted["a_to_e_effects"][0]
        self.assertAlmostEqual(effect["baseline_mean_s"], 249.02635246185164)
        self.assertAlmostEqual(effect["candidate_mean_s"], 218.00905231226244)
        self.assertAlmostEqual(effect["relative_reduction"], 0.1245542885038271)
        self.assertAlmostEqual(
            effect["task_p95_relative_reduction"], 0.12720960612425355
        )
        self.assertEqual(effect["faster_source_count"], 78)
        telemetry = extracted["cells"]["e-c12k-l80-u093"]
        telemetry = telemetry["physical_kv_telemetry"]
        self.assertEqual(telemetry["sample_count"], 364)
        self.assertAlmostEqual(telemetry["usage_max"], 0.535707)
        self.assertEqual(
            telemetry["target_budget_truncated_waiting_sample_count"], 272
        )
        self.assertEqual(
            telemetry["semantic_required_admission_field_malformed_count"], 0
        )
        self.assertEqual(telemetry["fail_closed_count"], 0)
        self.assertEqual(telemetry["raw_line_interleaving_count"], 2)
        self.assertFalse(telemetry["tail_rescue_parse_clean"])
        self.assertFalse(telemetry["strict_parser_v2_clean"])
        for cell in extracted["cells"].values():
            transport = cell["transport_evidence"]
            self.assertEqual(transport["http_attempt_count"], 160)
            self.assertEqual(transport["http_retry_count"], 0)
            self.assertEqual(transport["http_429_count"], 0)
            self.assertGreaterEqual(
                transport["minimum_adjacent_visit_start_gap_s"], 2.98
            )
            self.assertEqual(transport["broker"]["commits"], 160)
            self.assertEqual(transport["broker"]["authoritative_failures"], 0)
        repair = extracted["shape_r1_harness_repair"]
        self.assertEqual(len(repair["bound_files"]), 7)
        self.assertEqual(repair["failed_cell_request_count"], 0)
        self.assertEqual(
            len(repair["aggregator_additional_absence_checks"]), 3
        )
        self.assertTrue(
            all(repair["aggregator_additional_absence_checks"].values())
        )
        self.assertEqual(
            next(iter(repair["aggregator_additional_source_files"].values())),
            hashlib.sha256(b"").hexdigest(),
        )
        self.assertFalse(
            repair["excluded_observed_prefix"]["performance_loaded_or_reported"]
        )
        self.assertFalse(repair["excluded_observed_prefix"]["reused_by_replacement"])
        self.assertFalse(repair["excluded_observed_prefix"]["pooled_with_replacement"])

    def test_high_shape_provenance_mutations_fail_closed(self) -> None:
        run_root = HIGH_SUMMARY.parent
        plan = json.loads((run_root / "run_plan.json").read_text(encoding="utf-8"))
        completion = json.loads(
            (run_root / "completed_matrix.json").read_text(encoding="utf-8")
        )
        summary = json.loads(HIGH_SUMMARY.read_text(encoding="utf-8"))
        matrix = plan["preflight"]["matrix_invariants"]

        pooled = copy.deepcopy(plan)
        pooled["shape_r1_harness_repair"]["excluded_observed_prefix"][
            "pooled_with_replacement"
        ] = True
        with self.assertRaisesRegex(ValueError, "strictly excluded"):
            self.module._high_shape_r1_provenance(
                plan=pooled,
                completion=completion,
                summary_boundary=summary["evidence_boundary"],
                matrix=matrix,
                plan_cells=pooled["cells"],
                plan_bindings=pooled["bindings"],
                source_files=dict(self.module.HIGH_R1_ROOT_FILES),
            )

        bad_binding = copy.deepcopy(plan)
        bound = bad_binding["shape_r1_harness_repair"]["bound_files"]
        bound[next(iter(bound))] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binding set drifted"):
            self.module._high_shape_r1_provenance(
                plan=bad_binding,
                completion=completion,
                summary_boundary=summary["evidence_boundary"],
                matrix=matrix,
                plan_cells=bad_binding["cells"],
                plan_bindings=bad_binding["bindings"],
                source_files=dict(self.module.HIGH_R1_ROOT_FILES),
            )

        changed_config = copy.deepcopy(plan)
        command = changed_config["cells"][0]["runner_command"]
        command[command.index("--max-active-tasks") + 1] = "79"
        with self.assertRaisesRegex(ValueError, "non-identity config drifted"):
            self.module._high_shape_r1_provenance(
                plan=changed_config,
                completion=completion,
                summary_boundary=summary["evidence_boundary"],
                matrix=matrix,
                plan_cells=changed_config["cells"],
                plan_bindings=changed_config["bindings"],
                source_files=dict(self.module.HIGH_R1_ROOT_FILES),
            )

    def test_combined_formal_report_discloses_raw_interleavings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.module.run(
                Path(directory),
                live_sensitivity_summaries=(TARGET_SUMMARY, HIGH_SUMMARY),
                trace_center_summaries=(CENTER_SUMMARY,),
            )
            self.assertEqual(
                payload["evidence_boundary"][
                    "external_live_sensitivity_summary_count"
                ],
                2,
            )
            runs = {
                row["run_tag"]: row
                for row in payload["external_live_sensitivity_summaries"]
            }
            target_cells = runs["comment3-target-r3"]["cells"]
            expected_interleavings = {
                "e-c10k-l80-u085": 3,
                "e-c10k-l80-u093": 1,
                "e-c10k-l80-u097": 0,
            }
            for label, expected in expected_interleavings.items():
                telemetry = target_cells[label]["physical_kv_telemetry"]
                self.assertEqual(telemetry["raw_line_interleaving_count"], expected)
                self.assertEqual(
                    telemetry[
                        "semantic_required_admission_field_malformed_count"
                    ],
                    0,
                )
                self.assertEqual(
                    telemetry["strict_parser_v2_clean"], expected == 0
                )
            high_telemetry = runs["comment3-high-r1"]["cells"]
            high_telemetry = high_telemetry["e-c12k-l80-u093"][
                "physical_kv_telemetry"
            ]
            self.assertEqual(high_telemetry["raw_line_interleaving_count"], 2)
            self.assertFalse(high_telemetry["strict_parser_v2_clean"])

            report = (Path(directory) / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("249.026 | 218.009 | +12.455%", report)
            self.assertIn("313.284 | 273.431 | +12.721% | 78/80", report)
            self.assertIn("Raw line interleavings", report)
            self.assertIn("must not be described as raw-malformed-free", report)
            self.assertNotIn("| a-c5k-l40 |", report)
            self.assertNotIn("| e-c5k-l40-u093 |", report)

    def test_live_transport_rejects_recovered_retry(self) -> None:
        records = self._clean_tool_records()
        records[0]["http_attempts"] = 2
        records[0]["http_attempt_log"] = [
            {
                "attempt": 1,
                "status": 429,
                "retried": True,
                "started_monotonic_s": 0.0,
            },
            {
                "attempt": 2,
                "status": 200,
                "retried": False,
                "started_monotonic_s": 1.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "raw transport attempt"):
            self.module._clean_live_transport_evidence(records)

    def test_live_transport_rejects_sub_298_visit_gap(self) -> None:
        records = self._clean_tool_records()
        second_visit_log = records[81]["http_attempt_log"]
        assert isinstance(second_visit_log, list)
        assert isinstance(second_visit_log[0], dict)
        second_visit_log[0]["started_monotonic_s"] = 1_002.97
        with self.assertRaisesRegex(ValueError, "visit start gate"):
            self.module._clean_live_transport_evidence(records)

    def test_live_transport_rejects_broker_ledger_mismatch(self) -> None:
        broker = {
            "stats": {
                "authoritative_requests": 160,
                "authoritative_started": 160,
                "authoritative_completed": 160,
                "authoritative_failures": 0,
                "commits": 159,
            }
        }
        with self.assertRaisesRegex(ValueError, "broker completion ledger"):
            self.module._clean_live_broker_evidence(broker)

    def test_generation_records_evidence_boundary_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            payload = self.module.run(output)
            self.assertFalse(
                payload["evidence_boundary"][
                    "cross_model_or_gpu_generalization_proven"
                ]
            )
            self.assertTrue(
                payload["verification"]["formula_equivalent_within_1e_9"]
            )
            for name in (
                "raw_results.json",
                "sensitivity.csv",
                "sensitivity.svg",
                "REPORT.md",
            ):
                self.assertTrue((output / name).is_file(), name)
            raw = json.loads((output / "raw_results.json").read_text())
            self.assertEqual(raw["verification"]["factorial_state_count"], 108)
            self.assertIn("<svg", (output / "sensitivity.svg").read_text())

    def test_completed_live_summary_binds_raw_cells_and_recomputes_effects(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".scheduler-live-summary-test-", dir=REPOSITORY_ROOT
        ) as directory:
            fixture_base = Path(directory)
            run_root = fixture_base / "comment3-target-r3"
            run_root.mkdir()
            relative_root = str(run_root.relative_to(REPOSITORY_ROOT))
            binding = {"fixture.py": "0" * 64}
            specs = {
                "a": {
                    "cell": "A",
                    "context_padding_tokens": 10_000,
                    "label": "a",
                    "max_active_tasks": 80,
                    "pair_group": "c10k-l80",
                    "physical_kv_target": 0.93,
                    "role": "fcfs_reference",
                },
                "e": {
                    "cell": "E",
                    "context_padding_tokens": 10_000,
                    "label": "e",
                    "max_active_tasks": 80,
                    "pair_group": "c10k-l80",
                    "physical_kv_target": 0.93,
                    "role": "joint_candidate",
                },
            }

            def distribution(value: float, count: int) -> dict[str, float | int]:
                return {
                    "count": count,
                    "mean": value,
                    "p50": value,
                    "p95": value,
                    "p99": value,
                    "max": value,
                }

            plan_cells = []
            completion_cells = []
            summary_cells = {}
            result_paths = {}
            for index, label in enumerate(("a", "e")):
                spec = specs[label]
                cell_root = run_root / "cells" / f"{index + 1:02d}-{label}"
                evidence_dir = cell_root / "evidence"
                server_dir = cell_root / "server"
                evidence_dir.mkdir(parents=True)
                server_dir.mkdir()
                task_value = 10.0 if label == "a" else 9.0
                request_value = 3.0 if label == "a" else 2.5
                result = {
                    "tasks": [
                        {
                            "source_id": f"source-{source:03d}",
                            "task_id": f"source-{source:03d}__r00",
                            "replica": 0,
                            "question_sha256": hashlib.sha256(
                                f"question-{source:03d}".encode("utf-8")
                            ).hexdigest(),
                            "expected_url": f"https://example.test/{source:03d}",
                            "e2e_s": task_value,
                            "ok": True,
                        }
                        for source in range(80)
                    ],
                    "llm_events": [
                        {
                            "duration_s": request_value,
                            "ok": True,
                            "attempts": 1,
                            "http_status": 200,
                            "task_id": f"source-{request // 3:03d}__r00",
                            "call_index": request % 3,
                        }
                        for request in range(240)
                    ],
                    "tool_attempt_records": [
                        {
                            "authoritative": True,
                            "speculative": False,
                            "committed": True,
                            "outcome": "committed",
                            "http_attempts": 1,
                            "response_status": 200,
                            "transport_identity_source": "actual",
                            "session_id": f"source-{tool % 80:03d}__r00",
                            "invocation_digest": hashlib.sha256(
                                f"{tool % 80}:{'search' if tool < 80 else 'visit'}".encode(
                                    "utf-8"
                                )
                            ).hexdigest(),
                            "tool": "search" if tool < 80 else "visit",
                            "backend": (
                                "bing_html_search" if tool < 80 else "r.jina.ai"
                            ),
                            "request_host": (
                                "www.bing.com" if tool < 80 else "r.jina.ai"
                            ),
                            "http_attempt_log": [
                                {
                                    "attempt": 1,
                                    "status": 200,
                                    "retried": False,
                                    "started_monotonic_s": (
                                        float(tool)
                                        if tool < 80
                                        else 1_000.0 + 3.0 * (tool - 80)
                                    ),
                                }
                            ],
                        }
                        for tool in range(160)
                    ],
                    "broker_final_snapshot": {
                        "stats": {
                            "authoritative_requests": 160,
                            "authoritative_started": 160,
                            "authoritative_completed": 160,
                            "authoritative_failures": 0,
                            "commits": 160,
                        }
                    },
                }
                result_path = evidence_dir / "result.json"
                result_path.write_text(json.dumps(result), encoding="utf-8")
                result_paths[label] = result_path
                timeline_path = evidence_dir / "queue_timeline.jsonl"
                timeline_path.write_text("{}\n", encoding="utf-8")
                server_path = server_dir / "vllm_8100.log"
                if label == "a":
                    server_text = "FCFS fixture\n"
                    target = None
                    policy = "fcfs"
                else:
                    server_text = (
                        "[sched_policy_patch] installed policy=online_joint_pacer_v2 "
                        "v0=True v1=True\n"
                        "[sched_policy_patch:physical_kv] decision=admit reason=budget "
                        "num_gpu_blocks=100 block_size=16 capacity_tokens=1600 "
                        "target_utilization=0.930000 budget_tokens=1488 "
                        "usage=0.500000 live_tokens=800 logical_live_tokens=700 "
                        "running_growth_tokens=50 reserved_tokens=0 "
                        "committed_tokens=750 predicted_admit_tokens=700 "
                        "waiting=10 running=2 fit_admit=8 admit=8 "
                        "effective_cap=10 native_cap=96 "
                        "capacity_write_source=physical_kv "
                        "capacity_write_count=1 rescue=0\n"
                    )
                    target = "0.93"
                    policy = "online_joint_pacer_v2"
                server_path.write_text(server_text, encoding="utf-8")
                runner_stdout_path = cell_root / "runner.stdout.log"
                runner_stderr_path = cell_root / "runner.stderr.log"
                lifecycle_stdout_path = cell_root / "server_lifecycle.stdout.log"
                lifecycle_stderr_path = cell_root / "server_lifecycle.stderr.log"
                runner_stdout_path.write_text("completed fixture\n", encoding="utf-8")
                runner_stderr_path.write_text("", encoding="utf-8")
                lifecycle_stdout_path.write_text(
                    f"vLLM pid {1000 + index} stopped cleanly.\n",
                    encoding="utf-8",
                )
                lifecycle_stderr_path.write_text("", encoding="utf-8")
                contract = {
                    "schema": "paste_repro.scheduler_live_sensitivity_cell_contract",
                    "version": 1,
                    "development_only": True,
                    "formal_eligible": False,
                    "order_index": index,
                    "server_instance_id": f"server-{label}",
                    "spec": spec,
                    "bindings": binding,
                    "workload": {"sha256": "1" * 64},
                    "treatment": {
                        "policy": policy,
                        "physical_kv_admission": None if label == "a" else "1",
                        "physical_kv_target": target,
                    },
                    "transport_contract": {
                        "remediation_version": "fixture-r2-remediation",
                        "visit_min_start_interval_s": 3.0,
                        "accepted_http_attempts_per_tool_invocation": 1,
                        "zero_retries_required": True,
                    },
                }
                contract_path = cell_root / "cell_contract.json"
                contract_path.write_text(json.dumps(contract), encoding="utf-8")
                validation = {
                    "valid": True,
                    "fresh_server_identity": True,
                    "all_sources_exactly_once": True,
                    "task_count": 80,
                    "llm_request_count": 240,
                    "authoritative_tool_commit_count": 160,
                    "scheduler_policy": policy,
                    "physical_kv_target_visible_to_server": target,
                    "transport_validation": {
                        "remediation_version": "fixture-r2-remediation",
                        "visit_min_start_interval_s": 3.0,
                        "tool_invocation_count": 160,
                        "physical_http_attempt_count": 160,
                        "http_retry_count": 0,
                        "http_429_count": 0,
                        "all_status_200": True,
                    },
                }
                validation_path = cell_root / "strict_development_validation.json"
                validation_path.write_text(json.dumps(validation), encoding="utf-8")
                manifest_evidence = {}
                for evidence_path in (
                    contract_path,
                    validation_path,
                    result_path,
                    timeline_path,
                    server_path,
                    runner_stdout_path,
                    runner_stderr_path,
                    lifecycle_stdout_path,
                    lifecycle_stderr_path,
                ):
                    manifest_evidence[
                        str(evidence_path.relative_to(REPOSITORY_ROOT))
                    ] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                manifest = {
                    "schema": "paste_repro.scheduler_live_sensitivity_cell_evidence",
                    "version": 1,
                    "development_only": True,
                    "cell": label,
                    "evidence": manifest_evidence,
                }
                (cell_root / "cell_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                plan_cells.append({**spec, "order_index": index})
                completion_cells.append(
                    {
                        "label": label,
                        "path": str(cell_root.relative_to(REPOSITORY_ROOT)),
                    }
                )
                summary_cells[label] = {
                    "spec": spec,
                    "task_e2e_s": distribution(task_value, 80),
                    "llm_request_duration_s": distribution(request_value, 240),
                    "transport_validation": validation["transport_validation"],
                }

            summary_path = run_root / "summary.json"
            summary = {
                "schema": "paste_repro.scheduler_live_sensitivity_summary",
                "version": 1,
                "development_only": True,
                "formal_eligible": False,
                "single_run_per_cell": True,
                "confidence_interval_available": False,
                "run_root": relative_root,
                "evidence_boundary": {
                    "cross_gpu_or_cross_model_generalization_proven": False,
                    "transport_remediation_version": "fixture-r2-remediation",
                    "all_cells_rebaselined_after_failed_r2": True,
                    "zero_http_retries_required": True,
                    "descriptive_only_under_fixed_3s_jina_pacing": True,
                    "failed_r2_cells_excluded_without_pooling": True,
                },
                "cells": summary_cells,
                "a_to_e_effects": [
                    {
                        "pair_group": "c10k-l80",
                        "baseline": "a",
                        "candidate": "e",
                        "baseline_mean_s": 10.0,
                        "candidate_mean_s": 9.0,
                        "relative_reduction": 0.1,
                        "faster_source_count": 80,
                    }
                ],
                "physical_kv_target_sensitivity": [
                    {
                        "reference": "e",
                        "candidate": "e",
                        "target": 0.93,
                        "mean_s": 9.0,
                        "relative_change_vs_u093": 0.0,
                    }
                ],
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (run_root / "run_plan.json").write_text(
                json.dumps(
                    {
                        "schema": "paste_repro.scheduler_live_sensitivity_plan",
                        "version": 1,
                        "development_only": True,
                        "formal_eligible": False,
                        "run_tag": "fixture",
                        "suite": "target",
                        "run_root": relative_root,
                        "port": 8100,
                        "cell_count": 2,
                        "gpu_ids": [0, 1, 2, 3],
                        "bindings": binding,
                        "transport_remediation_after_failed_pilot": {
                            "failed_run_tag": "comment3-target-r2",
                            "all_cells_rebaselined": True,
                            "zero_http_retries_required": True,
                            "no_cross_transport_pooling_or_comparison": True,
                            "not_preregistered": True,
                            "failed_run_performance_was_observable": True,
                            "one_shot_replacement": True,
                            "no_further_auto_rerun_or_transport_escalation": True,
                        },
                        "preflight": {
                            "matrix_invariants": {
                                "all_cells_use_same_workload": True,
                                "fresh_server_per_cell": True,
                                "cross_cell_state_reuse": False,
                                "workload_sha256": "1" * 64,
                                "pair_checks": [
                                    {
                                        "only_scheduler_treatment_changes": True,
                                        "common_config_diff": {},
                                    }
                                ],
                                "target_checks": [
                                    {
                                        "only_active_physical_kv_target_changes": True
                                    }
                                ],
                                "transport_contract": {
                                    "remediation_version": "fixture-r2-remediation",
                                    "visit_min_start_interval_s": 3.0,
                                    "accepted_http_attempts_per_tool_invocation": 1,
                                    "zero_retries_required": True,
                                    "same_for_every_a_e_cell": True,
                                },
                            }
                        },
                        "cells": plan_cells,
                        "evidence_boundary": {"same_model_family": "fixture/model"},
                    }
                ),
                encoding="utf-8",
            )
            (run_root / "completed_matrix.json").write_text(
                json.dumps(
                    {
                        "schema": "paste_repro.scheduler_live_sensitivity_completion",
                        "version": 1,
                        "development_only": True,
                        "formal_eligible": False,
                        "summary": {
                            "path": str(
                                summary_path.relative_to(REPOSITORY_ROOT)
                            ),
                            "sha256": hashlib.sha256(
                                summary_path.read_bytes()
                            ).hexdigest()
                        },
                        "completed_cells": completion_cells,
                        "bindings": binding,
                    }
                ),
                encoding="utf-8",
            )

            failed_root = fixture_base / "comment3-target-r2"
            failed_a = failed_root / "cells/01-a-c10k-l80"
            failed_e = failed_root / "cells/02-e-c10k-l80-u085"
            for cell_root in (failed_a, failed_e):
                (cell_root / "evidence").mkdir(parents=True)
                (cell_root / "server").mkdir()
                (cell_root / "server/vllm_8100.log").write_text(
                    "excluded r2 fixture\n", encoding="utf-8"
                )
            (failed_root / "run_plan.json").write_text(
                json.dumps(
                    {
                        "schema": "paste_repro.scheduler_live_sensitivity_plan",
                        "run_tag": "comment3-target-r2",
                    }
                ),
                encoding="utf-8",
            )
            (failed_root / "failure.json").write_text(
                json.dumps(
                    {
                        "schema": "paste_repro.scheduler_live_sensitivity_failure",
                        "error_type": "LiveSensitivityError",
                    }
                ),
                encoding="utf-8",
            )

            def failed_record(*, recovered: bool) -> dict[str, object]:
                return {
                    "outcome": "committed" if recovered else "failed",
                    "http_attempts": 2,
                    "http_attempt_log": [
                        {"status": 429},
                        {"status": 200 if recovered else 429},
                    ],
                }

            clean_failed_provenance_record = {
                "outcome": "committed",
                "http_attempts": 1,
                "http_attempt_log": [{"status": 200}],
            }
            a_records = [failed_record(recovered=True) for _ in range(4)] + [
                dict(clean_failed_provenance_record) for _ in range(156)
            ]
            e_records = [failed_record(recovered=True) for _ in range(4)] + [
                failed_record(recovered=False)
            ] + [dict(clean_failed_provenance_record) for _ in range(155)]
            (failed_a / "evidence/result.json").write_text(
                json.dumps({"tool_attempt_records": a_records}),
                encoding="utf-8",
            )
            (failed_e / "evidence/result.json").write_text(
                json.dumps({"tool_attempt_records": e_records}),
                encoding="utf-8",
            )
            extracted = self.module._external_live_sensitivity_summary(
                summary_path
            )
            self.assertEqual(extracted["run_tag"], "fixture")
            self.assertEqual(extracted["cells"]["a"]["task_count"], 80)
            self.assertTrue(extracted["same_source_keys_across_cells"])
            self.assertEqual(
                extracted["cells"]["e"]["physical_kv_telemetry"]
                ["target_budget_truncated_waiting_sample_count"],
                1,
            )

            original_result = result_paths["e"].read_bytes()
            result_paths["e"].write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                self.module._external_live_sensitivity_summary(summary_path)
            result_paths["e"].write_bytes(original_result)

            summary["single_run_per_cell"] = False
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported live sensitivity"):
                self.module._external_live_sensitivity_summary(summary_path)


if __name__ == "__main__":
    unittest.main()
