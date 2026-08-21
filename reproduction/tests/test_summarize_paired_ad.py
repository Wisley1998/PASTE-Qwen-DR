from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from reproduction.tests.test_summarize_four_cell import (  # noqa: E402
    _heldout_fixture,
    _make_fixed_manifest,
    _write_json,
    _write_run,
)
import summarize_four_cell as summarize_four_cell_module  # noqa: E402
from summarize_paired_ad import parse_args, summarize_pairs  # noqa: E402


def _set_run_measurements(
    run: Path,
    *,
    completion_by_trace: dict[str, float],
    queue_s: float,
) -> None:
    events_path = run / "request_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    for event in events:
        completion = completion_by_trace[event["trace_id"]]
        if event["trace_id"] == "trace_000" and event["call_index"] == 0:
            completion -= 4.0
        event["request_end_offset_s"] = completion
        event["request_start_offset_s"] = completion - event["latency_s"]
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["avg_queue_time_s"] = queue_s
    summary["experiment_wall_time_s"] = max(completion_by_trace.values()) + 0.5
    _write_json(summary_path, summary)


def _paired_fixture(root: Path) -> tuple[Path, list[tuple[Path, Path]]]:
    manifest, workloads, mapper_sha = _make_fixed_manifest(root)
    pairs: list[tuple[Path, Path]] = []
    measurements = (
        (
            {"trace_000": 22.0, "trace_001": 11.0},
            {"trace_000": 12.0, "trace_001": 13.0},
            2.0,
            1.0,
        ),
        (
            {"trace_000": 22.0, "trace_001": 11.0},
            {"trace_000": 28.0, "trace_001": 7.0},
            3.0,
            2.0,
        ),
    )
    for replicate, (a_ends, d_ends, a_queue, d_queue) in enumerate(measurements, 1):
        a_run = _write_run(root, "A", replicate, workloads, mapper_sha)
        d_run = _write_run(root, "D", replicate, workloads, mapper_sha)
        _set_run_measurements(
            a_run,
            completion_by_trace=a_ends,
            queue_s=a_queue,
        )
        _set_run_measurements(
            d_run,
            completion_by_trace=d_ends,
            queue_s=d_queue,
        )
        pairs.append((a_run, d_run))
    return manifest, pairs


def _add_successful_retry(run: Path, *, delivery_ambiguous: bool = False) -> None:
    events_path = run / "request_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    final_success = dict(events[0]["attempt_history"][0])
    final_success["attempt"] = 2
    events[0]["attempts"] = 2
    events[0]["attempt_history"] = [
        {
            "attempt": 1,
            "transport": "aiohttp_connection",
            "outcome": "transport_error",
            "http_status": None,
            "error_type": (
                "ServerDisconnectedError"
                if delivery_ambiguous
                else "ClientConnectorError"
            ),
            "error": "synthetic transport failure",
            "duration_s": 0.001,
            "retryable": True,
            "will_retry": True,
            "retry_backoff_s": 1.0,
            "delivery_ambiguous": delivery_ambiguous,
        },
        final_success,
    ]
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "request_attempts_total": len(events) + 1,
            "retry_count": 1,
            "retried_request_count": 1,
            "retry_success_count": 1,
            "ambiguous_retry_count": int(delivery_ambiguous),
        }
    )
    _write_json(summary_path, summary)


def _add_execution_evidence(
    run: Path,
    *,
    completion_tokens_per_request: int,
    preemptions: int = 0,
    swap_events: int = 0,
) -> None:
    events_path = run / "request_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    for event in events:
        event["usage"] = {"completion_tokens": completion_tokens_per_request}
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "num_preemptions_total": float(preemptions),
            "num_preemptions_metric": "vllm:num_preemptions_total",
            "preemption_warning_count": preemptions,
            "kv_swap_happened": swap_events > 0,
            "kv_swap_event_count": swap_events,
            "kv_swap_in_event_count": swap_events,
            "kv_swap_out_event_count": 0,
            "kv_swap_total_blocks": swap_events * 2,
            "kv_swap_total_time_s": swap_events * 0.25,
            "kv_swap_avg_time_s": 0.25 if swap_events else 0.0,
            "kv_swap_in_avg_time_s": 0.25 if swap_events else 0.0,
            "kv_swap_out_avg_time_s": 0.0,
            "max_swapped_requests": int(swap_events > 0),
        }
    )
    _write_json(summary_path, summary)


class PairedADSummaryTests(unittest.TestCase):
    def test_public_paths_are_repository_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, pairs = _paired_fixture(root)
            with mock.patch.object(
                summarize_four_cell_module,
                "REPOSITORY_ROOT",
                root,
            ):
                result = summarize_pairs(pairs, manifest_path=manifest)

        self.assertEqual(
            result["comparison_invariants"]["fixed_workload_manifest"],
            "manifest.json",
        )
        expected = {
            (replicate, cell): f"{cell.lower()}_r{replicate}"
            for replicate in (1, 2)
            for cell in ("A", "D")
        }
        for replicate in result["replicates"]:
            for cell in ("A", "D"):
                run_path = replicate["cells"][cell]["run_path"]
                self.assertEqual(
                    run_path,
                    expected[(replicate["replicate"], cell)],
                )
                self.assertFalse(Path(run_path).is_absolute())

    def test_heldout_role_preserves_load_sensitivity_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _heldout_fixture(Path(temporary))
            result = summarize_pairs(
                [(groups["A"][0], groups["D"][0])],
                manifest_path=manifest,
                role="heldout",
            )

        invariants = result["comparison_invariants"]
        self.assertEqual(result["status"], "paired_heldout_ad_load_sensitivity")
        self.assertEqual(invariants["fixed_role"], "heldout")
        self.assertEqual(
            invariants["evidence_role"],
            "heldout_load_sensitivity_not_untouched_final",
        )
        self.assertEqual(invariants["source_session_count"], 4)
        self.assertIsNotNone(invariants["heldout_parent_manifest_sha256"])
        self.assertIn("not a new untouched final set", result["interpretation"])

    def test_per_replicate_and_unique_session_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            result = summarize_pairs(pairs, manifest_path=manifest)

        self.assertEqual(result["comparison_invariants"]["replicate_count"], 2)
        self.assertEqual(result["comparison_invariants"]["fixed_role"], "final")
        invariants = result["comparison_invariants"]
        self.assertEqual(invariants["configured_max_request_attempts"], 2)
        self.assertEqual(invariants["retry_accounting"]["requests_total"], 12)
        self.assertEqual(
            invariants["retry_accounting"]["request_attempts_total"], 12
        )
        self.assertTrue(invariants["all_requests_finally_succeeded"])
        self.assertTrue(invariants["all_requests_succeeded_exactly_once"])
        first = result["replicates"][0]["paired_task_flow"]
        second = result["replicates"][1]["paired_task_flow"]
        self.assertEqual(first["delta_s"]["mean"], 4.0)
        self.assertEqual(first["outcomes"]["joint_faster"], 1)
        self.assertEqual(first["outcomes"]["joint_slower"], 1)
        self.assertEqual(second["delta_s"]["mean"], -1.0)

        aggregate = result["aggregate"]
        paired = aggregate["paired_task_flow"]
        self.assertEqual(paired["independent_session_count"], 2)
        self.assertEqual(paired["raw_paired_observation_count"], 4)
        self.assertTrue(paired["repeated_sessions_are_not_counted_as_independent"])
        self.assertEqual(
            [row["delta_mean_s"] for row in paired["sessions"]],
            [2.0, 1.0],
        )
        self.assertEqual(paired["delta_s"]["mean"], 1.5)
        self.assertEqual(paired["delta_s"]["p50"], 1.5)
        self.assertAlmostEqual(paired["delta_s"]["p95"], 1.95)
        self.assertEqual(paired["outcomes"]["joint_faster"], 2)
        self.assertEqual(paired["outcomes"]["joint_slower"], 0)

        cells = aggregate["cells"]
        self.assertEqual(cells["A"]["task_flow_time_s"]["mean"], 15.0)
        self.assertEqual(cells["D"]["task_flow_time_s"]["mean"], 13.5)
        effect = aggregate["effects"]
        self.assertEqual(
            effect["absolute_reduction"]["task_flow_time_s"]["mean"], 1.5
        )
        self.assertEqual(effect["relative_reduction"]["task_flow_time_s"]["mean"], 0.1)
        self.assertEqual(effect["absolute_reduction"]["task_makespan_s"], 1.5)
        self.assertEqual(effect["absolute_reduction"]["mean_queue_time_s"], 1.0)
        self.assertIn("instrumentation_wall_time_s", effect["absolute_reduction"])

        # Legacy artifacts without token/serving counters remain readable, but
        # missing evidence is never silently reported as zero.
        self.assertIsNone(cells["A"]["completion_tokens_total"])
        self.assertIsNone(cells["A"]["num_preemptions_total"])
        self.assertIsNone(cells["A"]["kv_swap_happened"])
        self.assertFalse(
            cells["A"]["execution_accounting"]["completion_tokens"][
                "all_replicates_available"
            ]
        )

    def test_execution_evidence_and_source_bootstrap_are_formalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            for replicate, (a_run, d_run) in enumerate(pairs, 1):
                _add_execution_evidence(
                    a_run,
                    completion_tokens_per_request=10,
                    preemptions=replicate - 1,
                    swap_events=0,
                )
                _add_execution_evidence(
                    d_run,
                    completion_tokens_per_request=11,
                    preemptions=0,
                    swap_events=replicate - 1,
                )
            result = summarize_pairs(pairs, manifest_path=manifest)
            repeated = summarize_pairs(pairs, manifest_path=manifest)

        cells = result["aggregate"]["cells"]
        self.assertEqual(cells["A"]["requests_success"], 6)
        self.assertEqual(cells["A"]["requests_failed"], 0)
        self.assertEqual(cells["A"]["completion_tokens_total"], 60)
        self.assertEqual(cells["A"]["completion_tokens_mean_per_replicate"], 30)
        self.assertEqual(cells["D"]["completion_tokens_total"], 66)
        self.assertEqual(cells["A"]["num_preemptions_total"], 1)
        self.assertEqual(cells["D"]["kv_swap_event_count"], 1)
        self.assertTrue(cells["D"]["kv_swap_happened"])
        self.assertEqual(
            cells["D"]["execution_accounting"]["swap"]["kv_swap_total_blocks"],
            2,
        )

        token_comparison = result["aggregate"]["completion_token_comparison"]
        self.assertTrue(token_comparison["available"])
        self.assertEqual(token_comparison["d_minus_a"], 3)
        self.assertAlmostEqual(token_comparison["relative_to_a"], 0.1)
        self.assertEqual(
            result["comparison_invariants"]["request_outcomes"],
            {"requests_total": 12, "requests_success": 12, "requests_failed": 0},
        )

        paired = result["aggregate"]["paired_task_flow"]
        bootstrap = paired["independent_source_mean_bootstrap_95_ci_s"]
        self.assertEqual(bootstrap["seed"], 20260815)
        self.assertEqual(bootstrap["resamples"], 10_000)
        self.assertEqual(bootstrap["sample_size"], 2)
        self.assertEqual(
            bootstrap["sampling_unit"], "independent_source_session_mean"
        )
        self.assertTrue(
            paired[
                "duplicates_and_replicates_do_not_increase_independent_sample_size"
            ]
        )
        self.assertEqual(paired["effective_independent_sample_size"], 2)
        self.assertEqual(
            bootstrap,
            repeated["aggregate"]["paired_task_flow"][
                "independent_source_mean_bootstrap_95_ci_s"
            ],
        )

    def test_partial_or_malformed_execution_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            events_path = pairs[0][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[0]["usage"] = {"completion_tokens": 10}
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "partial completion-token"):
                summarize_pairs(pairs, manifest_path=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            summary_path = pairs[0][0] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["kv_swap_happened"] = False
            summary["kv_swap_event_count"] = 1
            summary["kv_swap_in_event_count"] = 1
            summary["kv_swap_out_event_count"] = 0
            summary["kv_swap_total_blocks"] = 1
            summary["kv_swap_total_time_s"] = 0.1
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "swap happened/count mismatch"):
                summarize_pairs(pairs, manifest_path=manifest)

    def test_legacy_preemption_conflated_swap_flag_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            run = pairs[0][0]
            _add_execution_evidence(
                run,
                completion_tokens_per_request=10,
                preemptions=3,
                swap_events=0,
            )
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text())
            # This is the immutable pre-v2 runner behavior: recompute
            # preemption alone set the field named kv_swap_happened.
            summary["kv_swap_happened"] = True
            _write_json(summary_path, summary)
            result = summarize_pairs(pairs, manifest_path=manifest)

        evidence = result["replicates"][0]["cells"]["A"][
            "execution_accounting"
        ]
        self.assertTrue(evidence["preemption"]["preemption_happened"])
        self.assertEqual(evidence["preemption"]["num_preemptions_total"], 3)
        self.assertFalse(evidence["swap"]["kv_swap_happened"])
        self.assertTrue(evidence["swap"]["recorded_kv_swap_happened"])
        self.assertTrue(
            evidence["swap"]["legacy_preemption_conflated_swap_flag"]
        )

    def test_v2_or_raw_swap_contradictions_still_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            run = pairs[0][0]
            _add_execution_evidence(
                run,
                completion_tokens_per_request=10,
                preemptions=2,
                swap_events=0,
            )
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["kv_swap_happened"] = True
            summary["kv_swap_happened_semantics"] = "cpu_swap_only_v2"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "swap happened/count mismatch"):
                summarize_pairs(pairs, manifest_path=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            run = pairs[0][0]
            _add_execution_evidence(
                run,
                completion_tokens_per_request=10,
                preemptions=0,
                swap_events=0,
            )
            _write_json(
                run / "swap_summary.json",
                {
                    "swap_event_count": 1,
                    "swap_in_event_count": 1,
                    "swap_out_event_count": 0,
                    "swap_total_blocks": 2,
                    "swap_total_time_s": 0.25,
                    "swap_avg_time_s": 0.25,
                    "swap_in_avg_time_s": 0.25,
                    "swap_out_avg_time_s": 0.0,
                },
            )
            with self.assertRaisesRegex(ValueError, "summary/raw swap event mismatch"):
                summarize_pairs(pairs, manifest_path=manifest)

    def test_ad_pair_allows_joint_only_tuning_but_keeps_it_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            for _, d_run in pairs:
                summary_path = d_run / "summary.json"
                summary = json.loads(summary_path.read_text())
                summary["scheduler_environment"].update(
                    {
                        "VLLM_SCHED_TIME_AGING_ALPHA": "0.2",
                        "VLLM_SCHED_JOINT_V2_FINAL_BONUS_S": "28",
                        "VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS": "12000",
                    }
                )
                _write_json(summary_path, summary)

            result = summarize_pairs(pairs, manifest_path=manifest)

        d_configuration = result["replicates"][0]["cells"]["D"][
            "scheduler_configuration"
        ]
        self.assertEqual(d_configuration["VLLM_SCHED_TIME_AGING_ALPHA"], "0.2")
        self.assertEqual(
            d_configuration["VLLM_SCHED_JOINT_V2_FINAL_BONUS_S"], "28"
        )

    def test_opt_in_strict_scheduler_configuration_rejects_joint_only_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            summary_path = pairs[0][1] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["scheduler_environment"][
                "VLLM_SCHED_JOINT_V2_FINAL_BONUS_S"
            ] = "28"
            _write_json(summary_path, summary)

            with self.assertRaisesRegex(
                ValueError,
                "A/D configuration mismatch: scheduler_configuration",
            ):
                summarize_pairs(
                    pairs,
                    manifest_path=manifest,
                    require_identical_scheduler_config=True,
                )

    def test_opt_in_strict_scheduler_configuration_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            result = summarize_pairs(
                pairs,
                manifest_path=manifest,
                require_identical_scheduler_config=True,
            )

        self.assertTrue(
            result["comparison_invariants"][
                "identical_scheduler_configuration_required"
            ]
        )

    def test_retry_accounting_reports_final_success_without_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            _add_successful_retry(pairs[0][0])
            result = summarize_pairs(pairs, manifest_path=manifest)

        invariants = result["comparison_invariants"]
        accounting = invariants["retry_accounting"]
        self.assertEqual(accounting["requests_total"], 12)
        self.assertEqual(accounting["request_attempts_total"], 13)
        self.assertEqual(accounting["retry_count"], 1)
        self.assertEqual(accounting["ambiguous_retry_count"], 0)
        self.assertTrue(invariants["all_requests_finally_succeeded"])
        self.assertFalse(invariants["all_requests_succeeded_exactly_once"])
        self.assertEqual(
            result["replicates"][0]["cells"]["A"]["retry_accounting"][
                "retry_count"
            ],
            1,
        )
        self.assertEqual(
            result["aggregate"]["cells"]["A"]["retry_accounting"][
                "retry_count"
            ],
            1,
        )

    def test_cli_accepts_repeated_pairs_and_requires_manifest(self) -> None:
        parsed = parse_args(
            [
                "--manifest",
                "manifest.json",
                "--pair",
                "a1",
                "d1",
                "--pair",
                "a2",
                "d2",
            ]
        )
        self.assertEqual(parsed.manifest, Path("manifest.json"))
        self.assertEqual(parsed.role, "final")
        self.assertEqual(parsed.pair, [[Path("a1"), Path("d1")], [Path("a2"), Path("d2")]])
        self.assertFalse(parsed.require_identical_scheduler_config)
        strict = parse_args(
            [
                "--manifest",
                "manifest.json",
                "--require-identical-scheduler-config",
                "--pair",
                "a",
                "d",
            ]
        )
        self.assertTrue(strict.require_identical_scheduler_config)
        heldout = parse_args(
            [
                "--manifest",
                "manifest.json",
                "--role",
                "heldout",
                "--pair",
                "a",
                "d",
            ]
        )
        self.assertEqual(heldout.role, "heldout")
        with self.assertRaises(SystemExit):
            parse_args(["--pair", "a", "d"])

    def test_failures_identity_mapper_and_configuration_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            summary_path = pairs[0][0] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["requests_failed"] = 1
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "failed requests"):
                summarize_pairs(pairs, manifest_path=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            events_path = pairs[0][1] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[0]["trace_id"] = "future-trace"
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "identities do not exactly match"):
                summarize_pairs(pairs, manifest_path=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            summary_path = pairs[1][1] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["workload"]["tool_prediction"]["artifact_sha256"] = "b" * 64
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "mapper checksum mismatch"):
                summarize_pairs(pairs, manifest_path=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            summary_path = pairs[1][0] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["scheduler_environment"]["VLLM_MAX_NUM_SEQS"] = "99"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                summarize_pairs(pairs, manifest_path=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            summary_path = pairs[0][1] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["configured_max_request_attempts"] = 3
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(
                ValueError,
                "A/D configuration mismatch: configured_max_request_attempts",
            ):
                summarize_pairs(pairs, manifest_path=manifest)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            for run in pairs[1]:
                summary_path = run / "summary.json"
                summary = json.loads(summary_path.read_text())
                summary["configured_max_request_attempts"] = 3
                _write_json(summary_path, summary)
            with self.assertRaisesRegex(
                ValueError,
                "configuration differs for cell A: configured_max_request_attempts",
            ):
                summarize_pairs(pairs, manifest_path=manifest)

    def test_run_directories_cannot_be_reused_as_fake_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, pairs = _paired_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "run directories must be unique"):
                summarize_pairs([pairs[0], pairs[0]], manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
