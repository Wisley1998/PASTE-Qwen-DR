from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from summarize_natural_queue_probe import main, summarize_probe  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _success_attempt(attempt: int = 1) -> dict:
    return {
        "attempt": attempt,
        "transport": "http",
        "outcome": "success",
        "http_status": 200,
        "error_type": None,
        "error": None,
        "duration_s": 1.0,
        "retryable": False,
        "will_retry": False,
        "retry_backoff_s": 0.0,
        "delivery_ambiguous": False,
    }


def _event(latency_s: float) -> dict:
    return {
        "ok": True,
        "http_status": 200,
        "attempts": 1,
        "attempt_history": [_success_attempt()],
        "latency_s": latency_s,
    }


def _write_probe_fixture(
    root: Path,
    *,
    max_num_seqs: int = 256,
    policy: str = "fcfs",
    native_admission: bool = True,
    binding_sample: bool = False,
    retry: bool = False,
    preemptions: int = 0,
    swap_events: int = 0,
) -> Path:
    cell = root / "cell"
    cell.mkdir(parents=True)
    events = [_event(8.0), _event(12.0)]
    if retry:
        events[0]["attempts"] = 2
        events[0]["attempt_history"] = [
            {
                "attempt": 1,
                "transport": "aiohttp_connection",
                "outcome": "transport_error",
                "http_status": None,
                "error_type": "ServerDisconnectedError",
                "error": "synthetic disconnect",
                "duration_s": 0.001,
                "retryable": True,
                "will_retry": True,
                "retry_backoff_s": 1.0,
                "delivery_ambiguous": True,
            },
            _success_attempt(2),
        ]
    (cell / "request_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    if binding_sample:
        samples = [
            {"t_s": 0.0, "running": 0.0, "waiting": 0.0, "gpu_cache_usage_perc": 0.0},
            {"t_s": 0.5, "running": 128.0, "waiting": 52.0, "gpu_cache_usage_perc": 0.7},
            {"t_s": 1.0, "running": 127.0, "waiting": 1.0, "gpu_cache_usage_perc": 0.9},
            {"t_s": 1.5, "running": 80.0, "waiting": 0.0, "gpu_cache_usage_perc": 0.4},
        ]
    else:
        samples = [
            {"t_s": 0.0, "running": 0.0, "waiting": 0.0, "gpu_cache_usage_perc": 0.0},
            {"t_s": 0.5, "running": 120.0, "waiting": 60.0, "gpu_cache_usage_perc": 0.7},
            {"t_s": 1.0, "running": 179.0, "waiting": 1.0, "gpu_cache_usage_perc": 0.9},
            {"t_s": 1.5, "running": 80.0, "waiting": 0.0, "gpu_cache_usage_perc": 0.4},
        ]
    timeline = {
        "samples": samples,
        "max_running": max(sample["running"] for sample in samples),
        "max_waiting": max(sample["waiting"] for sample in samples),
        "avg_running": sum(sample["running"] for sample in samples) / len(samples),
        "avg_waiting": sum(sample["waiting"] for sample in samples) / len(samples),
        "avg_gpu_cache_usage_perc": sum(
            sample["gpu_cache_usage_perc"] for sample in samples
        )
        / len(samples),
        "max_gpu_cache_usage_perc": max(
            sample["gpu_cache_usage_perc"] for sample in samples
        ),
    }
    _write_json(cell / "timeline.json", timeline)

    swap_total_time_s = swap_events * 0.25
    swap = {
        "swap_event_count": swap_events,
        "swap_in_event_count": swap_events,
        "swap_out_event_count": 0,
        "swap_avg_time_s": 0.25 if swap_events else 0.0,
        "swap_in_avg_time_s": 0.25 if swap_events else 0.0,
        "swap_out_avg_time_s": 0.0,
        "swap_total_time_s": swap_total_time_s,
        "swap_total_blocks": swap_events * 2,
    }
    _write_json(cell / "swap_summary.json", swap)
    max_swapped = int(swap_events > 0)
    warning_count = preemptions
    _write_json(
        cell / "vllm_log_summary.json",
        {
            "preemption_warning_count": warning_count,
            "max_swapped_requests": max_swapped,
        },
    )

    attempts_total = len(events) + int(retry)
    summary = {
        "requests_total": len(events),
        "requests_success": len(events),
        "requests_failed": 0,
        "configured_max_request_attempts": 2,
        "request_attempts_total": attempts_total,
        "retry_count": int(retry),
        "retried_request_count": int(retry),
        "retry_success_count": int(retry),
        "ambiguous_retry_count": int(retry),
        "final_failure_count": 0,
        "avg_request_latency_s": 10.0,
        "avg_queue_time_s": 2.5,
        "queue_time_metric_sum": "vllm:request_queue_time_seconds_sum",
        "queue_time_metric_count": "vllm:request_queue_time_seconds_count",
        "num_preemptions_total": float(preemptions),
        "num_preemptions_metric": "vllm:num_preemptions_total",
        "preemption_warning_count": warning_count,
        "kv_swap_happened": bool(preemptions or swap_events),
        "kv_swap_event_count": swap_events,
        "kv_swap_in_event_count": swap_events,
        "kv_swap_out_event_count": 0,
        "kv_swap_total_blocks": swap_events * 2,
        "kv_swap_avg_time_s": 0.25 if swap_events else 0.0,
        "kv_swap_in_avg_time_s": 0.25 if swap_events else 0.0,
        "kv_swap_out_avg_time_s": 0.0,
        "kv_swap_total_time_s": swap_total_time_s,
        "max_swapped_requests": max_swapped,
        "timeline_max_running": timeline["max_running"],
        "timeline_max_waiting": timeline["max_waiting"],
        "timeline_avg_running": timeline["avg_running"],
        "timeline_avg_waiting": timeline["avg_waiting"],
        "timeline_avg_gpu_cache_usage_perc": timeline[
            "avg_gpu_cache_usage_perc"
        ],
        "timeline_max_gpu_cache_usage_perc": timeline[
            "max_gpu_cache_usage_perc"
        ],
        "max_active_traces": 180,
        "workload": {"trace_count": 180, "request_count": len(events)},
        "scheduler_environment": {
            "VLLM_SCHED_POLICY": policy,
            "VLLM_MAX_NUM_SEQS": str(max_num_seqs),
            "VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
            "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": (
                "1" if native_admission else "0"
            ),
        },
    }
    _write_json(cell / "summary.json", summary)
    return cell


class NaturalQueueProbeTests(unittest.TestCase):
    def test_reports_native_queue_below_structurally_nonbinding_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(Path(temporary))
            result = summarize_probe(cell)

        self.assertEqual(result["schema"], "paste_repro.natural_queue_probe")
        self.assertEqual(result["request_accounting"]["requests_success"], 2)
        self.assertTrue(
            result["request_accounting"]["all_requests_succeeded_exactly_once"]
        )
        self.assertEqual(
            result["queueing"]["queue_time_fraction_of_request_latency"], 0.25
        )
        timeline = result["timeline"]
        self.assertEqual(timeline["sample_count"], 4)
        self.assertEqual(timeline["running_requests"]["mean"], 94.75)
        self.assertEqual(timeline["running_requests"]["max"], 179.0)
        self.assertEqual(timeline["waiting_requests"]["mean"], 15.25)
        self.assertEqual(timeline["waiting_requests"]["max"], 60.0)
        self.assertEqual(timeline["kv_cache_usage"]["mean"], 0.5)
        self.assertEqual(timeline["kv_cache_usage"]["max"], 0.9)
        self.assertEqual(timeline["waiting_below_sequence_cap_sample_count"], 2)
        self.assertEqual(timeline["waiting_below_sequence_cap_sample_fraction"], 0.5)
        self.assertEqual(
            timeline["waiting_below_sequence_cap_fraction_of_waiting_samples"],
            1.0,
        )
        capacity = result["sequence_capacity"]
        self.assertEqual(capacity["configured_max_num_seqs"], 256)
        self.assertEqual(capacity["offered_concurrency_upper_bound"], 180)
        self.assertEqual(capacity["configuration_sequence_headroom"], 76)
        self.assertTrue(capacity["sequence_cap_nonbinding"])
        self.assertTrue(capacity["natural_vllm_queue_proven"])

    def test_reports_retries_preemptions_and_swap_without_hiding_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(
                Path(temporary), retry=True, preemptions=3, swap_events=1
            )
            result = summarize_probe(cell)

        requests = result["request_accounting"]
        self.assertEqual(requests["retry_count"], 1)
        self.assertEqual(requests["ambiguous_retry_count"], 1)
        self.assertTrue(requests["all_requests_finally_succeeded"])
        self.assertFalse(requests["all_requests_succeeded_exactly_once"])
        memory = result["serving_memory_accounting"]
        self.assertEqual(memory["num_preemptions_total"], 3)
        self.assertEqual(memory["preemptions_per_logical_request"], 1.5)
        self.assertTrue(memory["kv_swap_happened"])
        self.assertEqual(memory["kv_swap_event_count"], 1)
        self.assertEqual(memory["kv_swap_total_blocks"], 2)

    def test_legacy_recompute_preemption_is_not_reported_as_cpu_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(Path(temporary), preemptions=3)
            result = summarize_probe(cell)

        memory = result["serving_memory_accounting"]
        self.assertTrue(memory["preemption_happened"])
        self.assertEqual(memory["num_preemptions_total"], 3)
        self.assertFalse(memory["kv_swap_happened"])
        self.assertTrue(memory["recorded_kv_swap_happened"])
        self.assertTrue(memory["legacy_preemption_conflated_swap_flag"])

    def test_v2_swap_semantics_rejects_a_conflated_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(Path(temporary), preemptions=3)
            summary_path = cell / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["kv_swap_happened_semantics"] = "cpu_swap_only_v2"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(
                ValueError, "kv_swap_happened accounting mismatch"
            ):
                summarize_probe(cell)

    def test_cap_reached_is_not_misreported_as_a_native_resource_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(
                Path(temporary), max_num_seqs=128, binding_sample=True
            )
            result = summarize_probe(cell)
            with self.assertRaisesRegex(
                ValueError, "natural vLLM queue requirement failed"
            ):
                main([str(cell), "--require-natural-queue"])

        capacity = result["sequence_capacity"]
        self.assertFalse(capacity["nonbinding_by_configuration"])
        self.assertFalse(capacity["nonbinding_in_timeline_samples"])
        self.assertFalse(capacity["sequence_cap_nonbinding"])
        self.assertFalse(capacity["natural_vllm_queue_proven"])
        self.assertEqual(capacity["conclusion"], "sequence_count_cap_may_bind")

    def test_joint_requires_explicit_native_admission_for_native_queue_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(
                Path(temporary),
                policy="online_joint_pacer_v2",
                native_admission=False,
            )
            result = summarize_probe(cell)

        capacity = result["sequence_capacity"]
        self.assertTrue(capacity["sequence_cap_nonbinding"])
        self.assertFalse(capacity["scheduler_admission_is_native"])
        self.assertFalse(capacity["natural_vllm_queue_proven"])
        self.assertEqual(capacity["conclusion"], "native_admission_not_established")

    def test_missing_sample_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(Path(temporary))
            timeline_path = cell / "timeline.json"
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            del timeline["samples"][1]["gpu_cache_usage_perc"]
            _write_json(timeline_path, timeline)
            with self.assertRaisesRegex(ValueError, "gpu_cache_usage_perc"):
                summarize_probe(cell)

    def test_cross_file_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(Path(temporary))
            summary_path = cell / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["timeline_max_waiting"] += 1
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(
                ValueError, "summary/timeline mismatch for timeline_max_waiting"
            ):
                summarize_probe(cell)

    def test_cli_emits_json_and_does_not_modify_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = _write_probe_fixture(Path(temporary))
            before = {
                path.name: path.read_bytes()
                for path in sorted(cell.iterdir())
                if path.is_file()
            }
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main([str(cell), "--require-natural-queue"]),
                    0,
                )
            after = {
                path.name: path.read_bytes()
                for path in sorted(cell.iterdir())
                if path.is_file()
            }

        self.assertEqual(before, after)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["sequence_capacity"]["natural_vllm_queue_proven"])

    def test_cli_acceptance_gates_pass_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = _write_probe_fixture(root / "accepted")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            str(accepted),
                            "--require-natural-queue",
                            "--require-exactly-once",
                            "--require-no-kv-swap",
                            "--min-waiting-below-cap-sample-fraction",
                            "0.50",
                            "--min-queue-time-fraction",
                            "0.25",
                            "--max-preemptions-per-request",
                            "0",
                        ]
                    ),
                    0,
                )

            cases = (
                (
                    _write_probe_fixture(root / "retry", retry=True),
                    ["--require-exactly-once"],
                    "exactly-once",
                ),
                (
                    _write_probe_fixture(root / "swap", swap_events=1),
                    ["--require-no-kv-swap"],
                    "no-CPU-KV-swap",
                ),
                (
                    accepted,
                    ["--min-waiting-below-cap-sample-fraction", "0.51"],
                    "waiting-below-cap",
                ),
                (
                    accepted,
                    ["--min-queue-time-fraction", "0.26"],
                    "queue-time fraction",
                ),
                (
                    _write_probe_fixture(root / "preempt", preemptions=3),
                    ["--max-preemptions-per-request", "1.49"],
                    "preemptions-per-request",
                ),
            )
            for cell, flags, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        main([str(cell), *flags])


if __name__ == "__main__":
    unittest.main()
