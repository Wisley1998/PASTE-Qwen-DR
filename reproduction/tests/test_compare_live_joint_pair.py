from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from compare_live_joint_pair import (  # noqa: E402
    BOOTSTRAP_SEED,
    _write_json_atomic,
    compare_live_joint_pair,
    main,
)


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _invocation_sha(tool: str, arguments: dict) -> str:
    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{tool}\0{encoded}".encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_run(
    root: Path,
    *,
    name: str,
    speculation_mode: str,
    e2e_values: tuple[float, float],
    result_suffix: str = "",
) -> Path:
    run_dir = root / name
    timeline_path = run_dir / "queue_timeline.jsonl"
    timeline = [
        {
            "wall_s": 1000.0,
            "monotonic_s": 10.0,
            "broker_revision": 1,
            "tool_queued_authoritative": 1,
            "tool_queued_speculative": 0,
            "tool_running_authoritative": 1,
            "tool_running_speculative": 0,
            "tool_completed_unclaimed_speculative": 0,
            "llm_running": 2.0,
            "llm_waiting": 1.0,
            "gpu_cache_usage": 0.1,
        },
        {
            "wall_s": 1000.2,
            "monotonic_s": 10.2,
            "broker_revision": 2,
            "tool_queued_authoritative": 0,
            "tool_queued_speculative": 0,
            "tool_running_authoritative": 0,
            "tool_running_speculative": 0,
            "tool_completed_unclaimed_speculative": 0,
            "llm_running": 1.0,
            "llm_waiting": 0.0,
            "gpu_cache_usage": 0.1,
        },
    ]
    _write_jsonl(timeline_path, timeline)

    config = {
        "cell_label": name,
        "model": "test/model",
        "speculation_mode": speculation_mode,
        "independent_source_count": 2,
        "replicas": 1,
        "task_count": 2,
        "max_active_tasks": 2,
        "tool_workers": 2,
        "speculative_tool_workers": 1,
        "max_speculative_pending": 8,
        "speculative_ttl_s": 60.0,
        "tool_http_max_attempts": 2,
        "tool_http_retry_backoff_s": 1.0,
        "controlled_http_retry": True,
        "tool_http_retry_policy_version": "idempotent-get-v1",
        "tool_http_retryable_statuses": [429, 500, 502, 503, 504],
        "tool_http_retryable_exception_types": [
            "asyncio.TimeoutError",
            "ConnectionError",
            "aiohttp.ClientConnectionError",
            "aiohttp.ClientPayloadError",
        ],
        "tool_http_library_retry_disabled": True,
        "tool_http_library_retry_control_version": (
            "aiohttp-private-retry-connection-v1"
        ),
        "tool_http_library_name": "aiohttp",
        "tool_http_library_version": "3.12.15",
        "visit_mode": "jina",
        "search_mode": "bing",
        "search_max_results": 5,
        "visit_max_chars": 3000,
        "visit_top_k": 1,
        "visit_canary_stride": 2,
        "live_tool_execution": True,
        "recorded_tool_sleep": False,
        "shared_bounded_tool_pool": True,
        "generated_tool_call_controls_next_prompt": True,
        "authoritative_and_speculative_share_capacity": True,
        "tool_metadata_is_causal": True,
        "tool_result_private_until_exact_commit": True,
        "future_trace_oracle_used": False,
        "workload_file_sha256": "a" * 64,
        "selected_workload_sha256": "b" * 64,
        "scheduler_environment": {"VLLM_SCHED_POLICY": "fcfs"},
    }
    tasks: list[dict] = []
    llm_events: list[dict] = []
    tool_records: list[dict] = []
    saved_total = 0.0
    spec_admitted = 0
    queued_promotions = 0
    for source_index, e2e in enumerate(e2e_values, 1):
        source_id = f"source{source_index}"
        task_id = f"{source_id}__r00"
        canary = source_index == 1
        url = f"https://en.wikipedia.org/wiki/Page_{source_index}"
        tools: list[dict] = []
        for tool_index, tool in enumerate(("search", "visit")):
            arguments = (
                {"query": [f"query {source_index}"]}
                if tool == "search"
                else {"url": [url], "goal": f"question {source_index}"}
            )
            invocation = {"tool_name": tool, "arguments": arguments}
            result_digest = hashlib.sha256(
                f"{source_id}:{tool}:result{result_suffix}".encode("utf-8")
            ).hexdigest()
            speculative = speculation_mode != "off" and (
                tool == "search"
                or (tool == "visit" and speculation_mode in {"visit", "search_visit"} and not canary)
            )
            exact_match = speculative
            source = "promoted_from_queue" if speculative else "executed"
            saved = 0.2 if speculative else 0.0
            saved_total += saved
            spec_admitted += int(speculative)
            queued_promotions += int(speculative)
            queue_s = 0.2 if speculative else 0.5
            service_s = 0.5
            exposed_s = 0.3 if speculative else 1.0
            tools.append(
                {
                    "invocation": invocation,
                    "source": source,
                    "exposed_wait_s": exposed_s,
                    "queue_s": queue_s,
                    "service_s": service_s,
                    "saved_service_s": saved,
                    "result_sha256": result_digest,
                }
            )
            started_at = 100.0 + source_index * 10 + tool_index
            tool_records.append(
                {
                    "job_id": len(tool_records) + 1,
                    "invocation_id": f"{name}-tool-{len(tool_records) + 1}",
                    "session_id": task_id,
                    "tool": tool,
                    "invocation_digest": _invocation_sha(tool, arguments),
                    "speculative": speculative,
                    "authoritative": True,
                    "admitted": True,
                    "queue_enter_at": started_at - queue_s,
                    "admitted_at": started_at - queue_s,
                    "started_at": started_at,
                    "authoritative_confirmation_at": started_at + 0.1,
                    "finished_at": started_at + service_s,
                    "outcome": "committed",
                    "result_digest": result_digest,
                    "exact_match": exact_match,
                    "source": source,
                    "cancelled": False,
                    "speculation_eligible": not (tool == "visit" and canary),
                    "canary": tool == "visit" and canary,
                    "worker_id": 0,
                    "queue_s": queue_s,
                    "service_s": service_s,
                    "exposed_wait_s": exposed_s,
                    "saved_service_s": saved,
                    "committed": True,
                    "response_status": 200,
                    "bytes_read": 1024,
                    "backend": "bing_html_search" if tool == "search" else "r.jina.ai",
                    "request_host": "www.bing.com" if tool == "search" else "r.jina.ai",
                    "http_attempts": 1,
                    "transport_identity_source": "actual",
                }
            )
        answer = {"answer": f"answer {source_index}", "source_url": url}
        tasks.append(
            {
                "task_id": task_id,
                "source_id": source_id,
                "replica": 0,
                "ok": True,
                "visit_canary": canary,
                "start_wall_s": 2000.0,
                "end_wall_s": 2000.0 + e2e,
                "e2e_s": e2e,
                "question_sha256": hashlib.sha256(
                    f"question {source_index}".encode("utf-8")
                ).hexdigest(),
                "search_query": f"query {source_index}",
                "search_urls": [url],
                "selected_url": url,
                "answer": answer,
                "answer_sha256": _canonical_sha(answer),
                "tools": tools,
                "llm_duration_s": 1.2,
                "prompt_tokens": 30,
                "completion_tokens": 6,
            }
        )
        for call_index in range(3):
            llm_events.append(
                {
                    "task_id": task_id,
                    "call_index": call_index,
                    "request_id": f"{name}-llm-{source_index}-{call_index}",
                    "request_start_s": 2000.0 + call_index,
                    "duration_s": 0.4,
                    "prompt_tokens_estimate": 10,
                    "attempts": 1,
                    "ok": True,
                    "http_status": 200,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    "scheduler_meta": {
                        "t": task_id,
                        "c": call_index,
                        "ms": "live_broker",
                        "tqa": 1,
                        "tqs": 0,
                        "tra": 1,
                        "trs": 0,
                    },
                }
            )

    result = {
        "schema_version": 1,
        "config": config,
        "summary": {
            "all_tasks_succeeded": True,
            "task_count": 2,
            "successful_task_count": 2,
            "failed_task_count": 0,
            "llm": {
                "request_count": 6,
                "successful_request_count": 6,
                "exactly_one_attempt_each": True,
            },
            "tool": {"authoritative_commit_count": 4},
        },
        "task_completion_makespan_s": max(e2e_values) + 0.1,
        "tasks": tasks,
        "llm_events": llm_events,
        "tool_attempt_records": tool_records,
        "broker_final_snapshot": {
            "counts": {
                "queued_authoritative": 0,
                "queued_speculative": 0,
                "running_authoritative": 0,
                "running_speculative": 0,
                "completed_unclaimed_speculative": 0,
            },
            "jobs": [],
            "stats": {
                "commits": 4,
                "authoritative_requests": 4,
                "authoritative_failures": 0,
                "saved_service_s": saved_total,
                "wasted_speculative_service_s": 0.0,
                "speculative_admitted": spec_admitted,
                "queued_promotions": queued_promotions,
                "running_promotions": 0,
                "completed_reuse": 0,
            },
        },
        "vllm_metric_deltas": {
            "vllm:prompt_tokens_total": 60.0,
            "vllm:generation_tokens_total": 12.0,
            "vllm:request_queue_time_seconds_sum": 1.0,
            "vllm:num_preemptions_total": 0.0,
        },
        "raw_evidence": {
            "queue_timeline": {
                "path": str(timeline_path.resolve()),
                "sha256": _sha(timeline_path),
                "sample_count": len(timeline),
            }
        },
    }
    result_path = run_dir / "result.json"
    _write_json(result_path, result)
    return result_path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _convert_to_frozen(path: Path, *, omit_expected_from_first_search: bool) -> None:
    payload = _load(path)
    payload["config"]["call_graph_mode"] = "frozen"
    payload["config"]["frozen_url_is_workload_input"] = True
    matched = 0
    for index, task in enumerate(payload["tasks"]):
        expected_url = task["selected_url"]
        task["call_graph_mode"] = "frozen"
        task["expected_url"] = expected_url
        if index == 0 and omit_expected_from_first_search:
            task["search_urls"] = [
                "https://en.wikipedia.org/wiki/Different_live_result"
            ]
        contains = expected_url in task["search_urls"]
        task["search_result_contains_expected_url"] = contains
        matched += int(contains)
    count = len(payload["tasks"])
    payload["config"]["expected_url_search_coverage"] = {
        "eligible_task_count": count,
        "observed_task_count": count,
        "matched_task_count": matched,
        "fraction_of_eligible": matched / count,
        "fraction_of_observed": matched / count,
    }
    _write_json(path, payload)


class CompareLiveJointPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline = _make_run(
            self.root,
            name="baseline",
            speculation_mode="off",
            e2e_values=(10.0, 12.0),
        )
        self.candidate = _make_run(
            self.root,
            name="candidate",
            speculation_mode="search_visit",
            e2e_values=(8.0, 9.0),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def compare(self) -> dict:
        return compare_live_joint_pair(
            self.baseline,
            self.candidate,
            bootstrap_resamples=200,
        )

    def rewrite(self, path: Path, mutate) -> None:
        payload = _load(path)
        mutate(payload)
        _write_json(path, payload)

    def mark_first_committed_retry(self, path: Path) -> None:
        def mutate(payload: dict) -> None:
            row = next(
                record
                for record in payload["tool_attempt_records"]
                if record["committed"] is True
            )
            row["http_attempts"] = 2
            row["service_s"] += 1.0
            row["finished_at"] += 1.0
            task = next(
                task for task in payload["tasks"] if task["task_id"] == row["session_id"]
            )
            tool_index = 0 if row["tool"] == "search" else 1
            task["tools"][tool_index]["service_s"] = row["service_s"]

        self.rewrite(path, mutate)

    def test_valid_pair_recomputes_gain_and_fixed_source_bootstrap(self) -> None:
        result = self.compare()
        self.assertFalse(result["claim_scope"]["screen_only"])
        self.assertTrue(result["claim_scope"]["identity_matched_paired_claim_eligible"])
        paired = result["comparison"]["source_paired"]
        self.assertAlmostEqual(paired["baseline_mean_s"], 11.0)
        self.assertAlmostEqual(paired["candidate_mean_s"], 8.5)
        self.assertAlmostEqual(paired["aggregate_relative_reduction"], 2.5 / 11.0)
        self.assertEqual(paired["faster_source_count"], 2)
        self.assertEqual(paired["bootstrap"]["seed"], BOOTSTRAP_SEED)
        self.assertEqual(result["comparison"]["speculation"]["exact_hit_count"], 3)
        self.assertEqual(result["identity_pairing"]["overall"]["invocation_match_rate"], 1.0)

    def test_controlled_tool_retry_is_accepted_and_reported(self) -> None:
        self.mark_first_committed_retry(self.candidate)
        result = self.compare()
        retry = result["comparison"]["tool"]["authoritative_retry"]
        self.assertEqual(retry["candidate"]["retried_commit_count"], 1)
        self.assertEqual(retry["candidate"]["commit_count"], 4)
        self.assertEqual(retry["candidate"]["rate"], 0.25)
        self.assertEqual(
            result["candidate"]["tool"]["physical_http_attempt_count"], 5
        )

    def test_tool_retry_above_controlled_attempt_limit_fails_closed(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["tool_attempt_records"][0].update(
                http_attempts=3
            ),
        )
        with self.assertRaisesRegex(ValueError, "controlled HTTP attempt limit"):
            self.compare()

    def test_retry_enabled_without_controlled_flag_fails_closed(self) -> None:
        for path in (self.baseline, self.candidate):
            self.rewrite(
                path,
                lambda payload: payload["config"].update(
                    controlled_http_retry=False
                ),
            )
        with self.assertRaisesRegex(ValueError, "disagrees with the attempt limit"):
            self.compare()

    def test_hidden_http_library_retry_must_be_disabled(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["config"].update(
                tool_http_library_retry_disabled=False
            ),
        )
        with self.assertRaisesRegex(ValueError, "hidden HTTP-library retry"):
            self.compare()

    def test_started_job_requires_actual_final_http_200(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["tool_attempt_records"][0].update(
                transport_identity_source="planned"
            ),
        )
        with self.assertRaisesRegex(ValueError, "actual final HTTP evidence"):
            self.compare()

    def test_dynamic_result_mismatch_is_explicitly_screen_only(self) -> None:
        candidate = _make_run(
            self.root,
            name="candidate-mismatch",
            speculation_mode="search",
            e2e_values=(8.0, 9.0),
            result_suffix=":changed",
        )
        result = compare_live_joint_pair(
            self.baseline,
            candidate,
            bootstrap_resamples=20,
        )
        self.assertTrue(result["claim_scope"]["screen_only"])
        self.assertFalse(result["claim_scope"]["identity_matched_paired_claim_eligible"])
        self.assertEqual(result["identity_pairing"]["overall"]["result_match_rate"], 0.0)
        self.assertEqual(result["identity_pairing"]["overall"]["invocation_match_rate"], 1.0)

    def test_frozen_graph_uses_workload_url_identity_not_search_coverage(self) -> None:
        candidate = _make_run(
            self.root,
            name="candidate-frozen",
            speculation_mode="search_visit",
            e2e_values=(8.0, 9.0),
            result_suffix=":dynamic-live-content",
        )
        _convert_to_frozen(self.baseline, omit_expected_from_first_search=True)
        _convert_to_frozen(candidate, omit_expected_from_first_search=True)
        result = compare_live_joint_pair(
            self.baseline,
            candidate,
            bootstrap_resamples=20,
        )
        self.assertFalse(result["claim_scope"]["screen_only"])
        self.assertTrue(result["claim_scope"]["identity_matched_paired_claim_eligible"])
        self.assertFalse(result["claim_scope"]["full_transport_result_identity_matched"])
        self.assertEqual(
            result["claim_scope"]["identity_basis"],
            "frozen_workload_expected_url_and_exact_visit_invocation",
        )
        coverage = result["identity_pairing"]["frozen_search_coverage"]
        self.assertEqual(coverage["identity_eligibility_effect"], "diagnostic_only")
        self.assertEqual(coverage["baseline"]["matched_task_count"], 1)
        self.assertEqual(coverage["candidate"]["matched_task_count"], 1)

    def test_frozen_graph_rejects_false_workload_input_flag(self) -> None:
        _convert_to_frozen(self.baseline, omit_expected_from_first_search=False)
        _convert_to_frozen(self.candidate, omit_expected_from_first_search=False)
        self.rewrite(
            self.candidate,
            lambda payload: payload["config"].update(
                frozen_url_is_workload_input=False
            ),
        )
        with self.assertRaisesRegex(ValueError, "frozen_url_is_workload_input=true"):
            self.compare()

    def test_frozen_graph_rejects_non_https_expected_url(self) -> None:
        _convert_to_frozen(self.baseline, omit_expected_from_first_search=False)
        _convert_to_frozen(self.candidate, omit_expected_from_first_search=False)
        self.rewrite(
            self.candidate,
            lambda payload: payload["tasks"][0].update(
                expected_url="http://en.wikipedia.org/wiki/Page_1",
                selected_url="http://en.wikipedia.org/wiki/Page_1",
            ),
        )
        with self.assertRaisesRegex(ValueError, "absolute HTTPS URL"):
            self.compare()

    def test_frozen_graph_rejects_selected_url_not_expected(self) -> None:
        _convert_to_frozen(self.baseline, omit_expected_from_first_search=False)
        _convert_to_frozen(self.candidate, omit_expected_from_first_search=False)
        self.rewrite(
            self.candidate,
            lambda payload: payload["tasks"][0].update(
                expected_url="https://en.wikipedia.org/wiki/Other"
            ),
        )
        with self.assertRaisesRegex(ValueError, "selected_url differs from expected_url"):
            self.compare()

    def test_frozen_graph_rejects_paired_visit_invocation_difference(self) -> None:
        _convert_to_frozen(self.baseline, omit_expected_from_first_search=False)
        _convert_to_frozen(self.candidate, omit_expected_from_first_search=False)

        def mutate(payload: dict) -> None:
            task = payload["tasks"][0]
            task["tools"][1]["invocation"]["arguments"]["goal"] = "changed goal"
            digest = _invocation_sha(
                "visit", task["tools"][1]["invocation"]["arguments"]
            )
            for record in payload["tool_attempt_records"]:
                if record["session_id"] == task["task_id"] and record["tool"] == "visit":
                    record["invocation_digest"] = digest

        self.rewrite(self.candidate, mutate)
        with self.assertRaisesRegex(ValueError, "visit invocation differs"):
            self.compare()

    def test_frozen_graph_rejects_forged_search_coverage(self) -> None:
        _convert_to_frozen(self.baseline, omit_expected_from_first_search=True)
        _convert_to_frozen(self.candidate, omit_expected_from_first_search=True)
        self.rewrite(
            self.candidate,
            lambda payload: payload["config"][
                "expected_url_search_coverage"
            ].update(matched_task_count=2),
        )
        with self.assertRaisesRegex(ValueError, "search-coverage count"):
            self.compare()

    def test_rejects_failed_task(self) -> None:
        self.rewrite(self.candidate, lambda payload: payload["tasks"][0].update(ok=False))
        with self.assertRaisesRegex(ValueError, "task did not succeed"):
            self.compare()

    def test_rejects_llm_retry(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["llm_events"][0].update(attempts=2),
        )
        with self.assertRaisesRegex(ValueError, "not exactly once"):
            self.compare()

    def test_rejects_missing_authoritative_commit(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["tool_attempt_records"].pop(),
        )
        with self.assertRaisesRegex(ValueError, "exactly one authoritative visit commit"):
            self.compare()

    def test_rejects_forged_http_backend(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["tool_attempt_records"][0].update(
                backend="fake_cache"
            ),
        )
        with self.assertRaisesRegex(ValueError, "backend does not match"):
            self.compare()

    def test_never_started_cancellation_has_explicit_zero_attempt_contract(self) -> None:
        def add_cancelled(payload: dict) -> None:
            row = dict(payload["tool_attempt_records"][1])
            row.update(
                {
                    "job_id": 999,
                    "invocation_id": "never-started-cancelled",
                    "speculative": True,
                    "authoritative": False,
                    "committed": False,
                    "authoritative_confirmation_at": None,
                    "started_at": None,
                    "finished_at": row["queue_enter_at"] + 3.0,
                    "outcome": "cancelled",
                    "result_digest": None,
                    "exact_match": False,
                    "source": "cancelled",
                    "cancelled": True,
                    "worker_id": None,
                    "queue_s": 3.0,
                    "service_s": 0.0,
                    "exposed_wait_s": None,
                    "saved_service_s": 0.0,
                    "response_status": None,
                    "bytes_read": None,
                    "backend": None,
                    "request_host": None,
                    "http_attempts": 0,
                    "transport_identity_source": None,
                }
            )
            payload["tool_attempt_records"].append(row)

        self.rewrite(self.candidate, add_cancelled)
        self.compare()

        self.rewrite(
            self.candidate,
            lambda payload: payload["tool_attempt_records"][-1].update(
                http_attempts=None
            ),
        )
        with self.assertRaisesRegex(ValueError, "http_attempts must be an integer"):
            self.compare()

    def test_rejects_timeline_sha_mismatch(self) -> None:
        payload = _load(self.candidate)
        timeline = Path(payload["raw_evidence"]["queue_timeline"]["path"])
        with timeline.open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.compare()

    def test_rejects_timeline_without_tool_queue_pressure(self) -> None:
        payload = _load(self.candidate)
        timeline_path = Path(payload["raw_evidence"]["queue_timeline"]["path"])
        rows = [json.loads(line) for line in timeline_path.read_text().splitlines()]
        for row in rows:
            row["tool_queued_authoritative"] = 0
            row["tool_queued_speculative"] = 0
        _write_jsonl(timeline_path, rows)
        payload["raw_evidence"]["queue_timeline"]["sha256"] = _sha(timeline_path)
        _write_json(self.candidate, payload)
        with self.assertRaisesRegex(ValueError, "no real tool-queue pressure"):
            self.compare()

    def test_rejects_uncontrolled_config_difference(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["config"].update(tool_workers=3),
        )
        with self.assertRaisesRegex(ValueError, "differs outside"):
            self.compare()

    def test_cli_failure_does_not_create_output(self) -> None:
        self.rewrite(
            self.candidate,
            lambda payload: payload["llm_events"][0].update(attempts=2),
        )
        output = self.root / "comparison.json"
        status = main(
            [
                "--baseline-result",
                str(self.baseline),
                "--candidate-result",
                str(self.candidate),
                "--output",
                str(output),
                "--bootstrap-resamples",
                "10",
            ]
        )
        self.assertEqual(status, 2)
        self.assertFalse(output.exists())

    def test_atomic_writer_replaces_complete_json(self) -> None:
        output = self.root / "comparison.json"
        output.write_text("old\n", encoding="utf-8")
        _write_json_atomic(output, {"ok": True})
        self.assertEqual(_load(output), {"ok": True})
        self.assertEqual(list(output.parent.glob(".comparison.json.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
