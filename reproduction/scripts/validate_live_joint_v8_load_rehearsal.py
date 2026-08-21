#!/usr/bin/env python3
"""Strict validator for the development-only v8 80-offered A rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORMAL_RUNNER = (
    REPOSITORY_ROOT / "reproduction/scripts/run_live_joint_formal_matrix.py"
)
TUNE_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
)
TUNE_WORKLOAD_SHA256 = (
    "e9f63f75bb80c840fbc59f2aa9a581527669c10fc761a4649f50a1bc03eaf1ea"
)
FORMAL_V8_WORKLOAD_SHA256 = (
    "780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4"
)
TUNE_SPLIT_ID = "live-joint-wikipedia-frozen-tune-v1"
TUNE_SOURCE_COUNT = 16
REPLICAS = 5
TASK_COUNT = 80

SPEC = importlib.util.spec_from_file_location("formal_v8_runner", FORMAL_RUNNER)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - installation failure
    raise RuntimeError("cannot import the frozen formal-v8 runner")
formal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal)


class RehearsalValidationError(ValueError):
    """Fail-closed development-rehearsal validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_development_workload(path: Path = TUNE_WORKLOAD) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved != TUNE_WORKLOAD.resolve():
        raise RehearsalValidationError(
            "rehearsal workload must be the frozen tune-v1 repository path"
        )
    file_sha256 = sha256_file(resolved)
    if file_sha256 != TUNE_WORKLOAD_SHA256:
        raise RehearsalValidationError("frozen tune-v1 raw SHA256 mismatch")
    if file_sha256 == FORMAL_V8_WORKLOAD_SHA256:
        raise RehearsalValidationError("formal-v8 workload is forbidden in rehearsal")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if (
        payload.get("schema_version") != 1
        or payload.get("split_id") != TUNE_SPLIT_ID
        or payload.get("split_role") != "tune"
        or payload.get("formal_eligible") is not False
        or not isinstance(sources, list)
        or len(sources) != TUNE_SOURCE_COUNT
    ):
        raise RehearsalValidationError(
            "rehearsal workload is not the exact non-formal 16-source tune split"
        )
    expected_ids = [f"tune-wiki{index:03d}" for index in range(1, 17)]
    observed_ids: list[str] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise RehearsalValidationError(f"tune source {index} is invalid")
        source_id = source.get("source_id")
        question = source.get("question")
        query = source.get("search_query")
        url = source.get("expected_url")
        if (
            not isinstance(source_id, str)
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(url, str)
            or not url.startswith("https://en.wikipedia.org/wiki/")
        ):
            raise RehearsalValidationError(f"tune source {index} is incomplete")
        observed_ids.append(source_id)
    if observed_ids != expected_ids:
        raise RehearsalValidationError("frozen tune source order/identity mismatch")
    return {
        "schema": "paste_repro.live_joint_v8_rehearsal_workload_validation",
        "version": 1,
        "valid": True,
        "development_only": True,
        "formal_evidence_eligible": False,
        "file_sha256": file_sha256,
        "split_id": TUNE_SPLIT_ID,
        "split_role": "tune",
        "source_count": TUNE_SOURCE_COUNT,
        "replicas": REPLICAS,
        "offered_task_count": TASK_COUNT,
        "formal_v8_workload_sha256_forbidden": FORMAL_V8_WORKLOAD_SHA256,
    }


def _expected_config(block_id: str, server_instance_id: str) -> dict[str, Any]:
    return {
        "call_graph_mode": "frozen",
        "speculation_mode": "off",
        "tool_signal_policy": "execution_aware",
        "tool_signal_policy_version": "exact-session-invocation-running-completed-v1",
        "tool_signal_policy_module_sha256": formal.LIVE_AGENT_SHA256,
        "independent_source_count": TUNE_SOURCE_COUNT,
        "replicas": REPLICAS,
        "task_count": TASK_COUNT,
        "max_active_tasks": TASK_COUNT,
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
        "workload_file_sha256": TUNE_WORKLOAD_SHA256,
        "workload_split_id": TUNE_SPLIT_ID,
        "workload_split_role": "tune",
        "workload_formal_eligible": False,
        "formal_run": {
            "block_id": block_id,
            "cell_id": "A",
            "order_index": 0,
            "server_instance_id": server_instance_id,
            "fresh_server": True,
            "result_cache_empty": True,
            "broker_drained": True,
        },
    }


def validate_rehearsal_result(
    *,
    result_path: Path,
    timeline_path: Path,
    block_id: str,
    server_instance_id: str,
) -> dict[str, Any]:
    workload_validation = validate_development_workload()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise RehearsalValidationError("rehearsal result config is missing")
    changed = [
        key
        for key, expected in _expected_config(block_id, server_instance_id).items()
        if config.get(key) != expected
    ]
    if changed:
        raise RehearsalValidationError(
            f"development rehearsal config mismatch: {sorted(changed)}"
        )
    scheduler = config.get("scheduler_environment")
    if not isinstance(scheduler, Mapping):
        raise RehearsalValidationError("rehearsal scheduler evidence is missing")
    required_scheduler = {
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
    if any(scheduler.get(key) != value for key, value in required_scheduler.items()):
        raise RehearsalValidationError("rehearsal native FCFS environment mismatch")
    leaked = [
        key
        for key, value in scheduler.items()
        if key.startswith("VLLM_SCHED_")
        and key != "VLLM_SCHED_POLICY"
        and value is not None
    ]
    if leaked:
        raise RehearsalValidationError(
            f"development FCFS rehearsal leaked Joint knobs: {sorted(leaked)}"
        )

    summary = result.get("summary")
    broker_stats = (
        summary.get("tool", {}).get("broker_stats", {})
        if isinstance(summary, Mapping)
        else {}
    )
    if (
        not isinstance(summary, Mapping)
        or summary.get("all_tasks_succeeded") is not True
        or summary.get("task_count") != TASK_COUNT
        or summary.get("successful_task_count") != TASK_COUNT
        or summary.get("failed_task_count") != 0
        or summary.get("llm", {}).get("request_count") != 240
        or summary.get("llm", {}).get("successful_request_count") != 240
        or summary.get("llm", {}).get("exactly_one_attempt_each") is not True
        or broker_stats.get("authoritative_requests") != 160
        or broker_stats.get("authoritative_executions") != 160
        or broker_stats.get("authoritative_failures") != 0
        or broker_stats.get("commits") != 160
        or broker_stats.get("speculative_admitted") != 0
        or broker_stats.get("speculative_started") != 0
        or broker_stats.get("speculative_failures") != 0
    ):
        raise RehearsalValidationError("rehearsal exact completion counts mismatch")

    tasks = result.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != TASK_COUNT:
        raise RehearsalValidationError("rehearsal task evidence is incomplete")
    tune_payload = json.loads(TUNE_WORKLOAD.read_text(encoding="utf-8"))
    tune_sources = {
        source["source_id"]: source for source in tune_payload["sources"]
    }
    for task_index, task in enumerate(tasks):
        source_number = task_index // REPLICAS + 1
        replica = task_index % REPLICAS
        source_id = f"tune-wiki{source_number:03d}"
        source = tune_sources[source_id]
        if (
            not isinstance(task, Mapping)
            or task.get("ok") is not True
            or task.get("source_id") != source_id
            or task.get("replica") != replica
            or task.get("task_id") != f"{source_id}__r{replica:02d}"
            or task.get("search_query") != source["search_query"]
            or task.get("question_sha256")
            != hashlib.sha256(source["question"].encode("utf-8")).hexdigest()
            or task.get("expected_url") != source["expected_url"]
            or task.get("selected_url") != source["expected_url"]
            or task.get("visit_canary") != (task_index % 6 == 0)
            or task.get("context_padding_target_tokens") != 10000
        ):
            raise RehearsalValidationError(
                f"rehearsal task {task_index} violates frozen identity/canary/padding"
            )
        formal._validate_task_output_contract(
            task, label=f"development-rehearsal task {task_index}"
        )
    formal._validate_fixed_final_llm_events(
        result,
        tasks=tasks,
        label="development-rehearsal/A",
    )

    records = result.get("tool_attempt_records")
    if not isinstance(records, list) or len(records) != 160:
        raise RehearsalValidationError("rehearsal requires exactly 160 live records")
    tool_counts = {"search": 0, "visit": 0}
    visit_starts: list[float] = []
    canary_count = 0
    for index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or record.get("admitted") is not True
            or record.get("authoritative") is not True
            or record.get("speculative") is not False
            or record.get("committed") is not True
            or record.get("cancelled") is not False
            or record.get("outcome") != "committed"
            or record.get("http_attempts") != 1
        ):
            raise RehearsalValidationError(
                f"rehearsal tool record {index} is not one committed live GET"
            )
        tool = record.get("tool")
        if tool not in tool_counts:
            raise RehearsalValidationError(f"rehearsal tool record {index} is unknown")
        tool_counts[tool] += 1
        starts = formal._validate_started_tool_record(
            record,
            label=f"development-rehearsal tool record {index}",
            max_http_attempts=2,
            retry_backoff_s=1.0,
        )
        if tool == "visit":
            visit_starts.extend(starts)
            if record.get("canary") is True:
                canary_count += 1
                if record.get("speculation_eligible") is not False:
                    raise RehearsalValidationError("canary visit was speculation eligible")
    if tool_counts != {"search": 80, "visit": 80} or canary_count != 14:
        raise RehearsalValidationError("rehearsal live tool/canary counts mismatch")
    ordered_starts = sorted(visit_starts)
    if any(
        right - left < 2.08
        for left, right in zip(ordered_starts, ordered_starts[1:])
    ):
        raise RehearsalValidationError("rehearsal violated the 2.1s visit start gate")

    snapshot = result.get("broker_final_snapshot")
    counts = snapshot.get("counts") if isinstance(snapshot, Mapping) else None
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("jobs") != []
        or not isinstance(counts, Mapping)
        or any(
            counts.get(key) != 0
            for key in (
                "completed_unclaimed_speculative",
                "queued_authoritative",
                "queued_speculative",
                "running_authoritative",
                "running_speculative",
            )
        )
        or counts.get("queued_by_tool") != {}
        or counts.get("running_by_tool") != {}
    ):
        raise RehearsalValidationError("rehearsal broker did not drain")

    gate = formal.evaluate_baseline_gate(
        result_path,
        timeline_path,
        block_id=block_id,
    )
    if gate.get("accepted") is not True:
        raise RehearsalValidationError(
            "development rehearsal failed the frozen 80-offered dual-queue gate"
        )
    return {
        "schema": "paste_repro.live_joint_v8_load_rehearsal_validation",
        "version": 1,
        "valid": True,
        "development_only": True,
        "formal_evidence_eligible": False,
        "selection_uses_formal_v8_performance": False,
        "thresholds": {
            "exact_live_http_attempts_per_record": 1,
            "zero_transport_retries_required": True,
            "minimum_native_waiting_below_cap_fraction": 0.05,
            "minimum_authoritative_tool_queue_fraction": 0.05,
            "minimum_dual_queue_samples": 10,
            "minimum_continuous_dual_queue_span_s": 1.0,
            "maximum_adjacent_dual_sample_gap_s": 0.5,
        },
        "workload_validation": workload_validation,
        "result_sha256": sha256_file(result_path),
        "timeline_sha256": sha256_file(timeline_path),
        "baseline_gate": gate,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--server-instance-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validation = validate_rehearsal_result(
        result_path=args.result.resolve(),
        timeline_path=args.timeline.resolve(),
        block_id=args.block_id,
        server_instance_id=args.server_instance_id,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
