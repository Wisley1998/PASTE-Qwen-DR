#!/usr/bin/env python3
"""Fail-closed validator for the prospective v9 development-only screen.

This module validates one fresh-server cell.  Stage 0 deliberately returns a
gate vector instead of immediately rejecting a controlled HTTP retry so the
runner can apply the preregistered, baseline-only 2.5s -> 3.0s transport
fallback.  Every structural, correctness, load, and output-contract failure
still raises.  Stage-1 cells require zero retry/failure/waste outright.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproduction/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import aggregate_live_joint_four_cell as formal_aggregate  # type: ignore
import compare_live_joint_pair as pair  # type: ignore


FORMAL_RUNNER_PATH = SCRIPTS_DIR / "run_live_joint_formal_matrix.py"
FORMAL_SPEC = importlib.util.spec_from_file_location(
    "formal_v8_runner_for_v9_development", FORMAL_RUNNER_PATH
)
if FORMAL_SPEC is None or FORMAL_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import the frozen formal-v8 runner")
formal = importlib.util.module_from_spec(FORMAL_SPEC)
FORMAL_SPEC.loader.exec_module(formal)

CONFIG_PATH = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_v9_development_screen.env.example"
)
TUNE_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
)
TUNE_WORKLOAD_SHA256 = (
    "e9f63f75bb80c840fbc59f2aa9a581527669c10fc761a4649f50a1bc03eaf1ea"
)
TUNE_CANONICAL_SHA256 = (
    "f6062264053c72096e6be3c91753ffc5e7adb6bb30b42a32e028e652e38df63f"
)
TUNE_SOURCES_SHA256 = (
    "d5343d42e19699198788b241c8fd9f9fcb5c0be4435ad6f601ba8bd299c1c450"
)
TUNE_SPLIT_ID = "live-joint-wikipedia-frozen-tune-v1"
EXPECTED_LIVE_BROKER_SHA256 = (
    "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27"
)
FORMAL_WORKLOAD_SHA256S = frozenset(
    {
        "4c71ce9bf72b3cbec8ddc077f7e58270493f10e63f3a45e107e39faff3b1bb76",
        "a8f5de832e7e04e3cbd1b7bb71629207201f99285a0d9f95fbc1e7246f0b6366",
        "e965317225ed0f2d4aec9e8e1a444abd0949521205e705c4daae5e786ce092d5",
        "6b11193c8a0dbbd70f9ae4bc2c72b56737893b4d45dacd1d9970e01ca019ae31",
        "44122877db66b1df4a985316c2a96b71d91d13c4e8be84affb73d405490bd43f",
        "cbf143f59f4d2a05650df68d8fa6f00d7471964a4b257d26dd092ba90c40e6c8",
        "780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4",
        "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20",
    }
)

SOURCE_COUNT = 16
REPLICAS = 5
TASK_COUNT = SOURCE_COUNT * REPLICAS
LLM_REQUEST_COUNT = 3 * TASK_COUNT
AUTHORITATIVE_COMMIT_COUNT = 2 * TASK_COUNT
CANARY_STRIDE = 6
CANARY_COUNT = 14
ELIGIBLE_VISIT_COUNT = TASK_COUNT - CANARY_COUNT
CONTEXT_PADDING_TOKENS = 10_000
FIXED_FINAL_COMPLETION_TOKENS = 192
TRANSPORT_LADDER_S = (2.5, 3.0)
HTTP_SPACING_TOLERANCE_S = 0.02

CELL_TREATMENTS: dict[str, dict[str, Any]] = {
    "A": {
        "scheduler": "fcfs",
        "speculation_mode": "off",
        "min_speculative_tool_workers": 0,
        "formal_cell_id": "A",
    },
    "E": {
        "scheduler": "online_joint_pacer_v2",
        "speculation_mode": "off",
        "min_speculative_tool_workers": 0,
        "formal_cell_id": "E",
    },
    "F0": {
        "scheduler": "online_joint_pacer_v2",
        "speculation_mode": "visit",
        "min_speculative_tool_workers": 0,
        "formal_cell_id": "F",
    },
    "F1": {
        "scheduler": "online_joint_pacer_v2",
        "speculation_mode": "visit",
        "min_speculative_tool_workers": 1,
        "formal_cell_id": "F",
    },
}

EXPECTED_CONFIG: dict[str, str] = dict(formal.EXPECTED_CONFIG)
EXPECTED_CONFIG.update(
    {
        "PASTE_LIVE_FORMAL_PROFILE": (
            "live_joint_v9_development_screen_tune16x5_context10000_"
            "fixedfinal192_transport_ladder_2p5_3p0"
        ),
        "PASTE_LIVE_FORMAL_WORKLOAD": (
            "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
        ),
        "PASTE_LIVE_FORMAL_WORKLOAD_SHA256": TUNE_WORKLOAD_SHA256,
        "PASTE_LIVE_FORMAL_CANONICAL_SHA256": TUNE_CANONICAL_SHA256,
        "PASTE_LIVE_FORMAL_SOURCES_SHA256": TUNE_SOURCES_SHA256,
        "PASTE_LIVE_FORMAL_SOURCE_COUNT": str(SOURCE_COUNT),
        "PASTE_LIVE_FORMAL_DEFAULT_ORDERS": "E,F0,F1;F1,F0,E",
        "PASTE_LIVE_FORMAL_RUN_BASE": (
            "reproduction/artifacts/live_joint/development/v9_screen"
        ),
        "PASTE_LIVE_REPLICAS": str(REPLICAS),
        "PASTE_LIVE_VISIT_MIN_START_INTERVAL_S": "2.5",
    }
)


class DevelopmentScreenValidationError(ValueError):
    """A prospective development-screen contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gate(observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def load_frozen_config(path: Path = CONFIG_PATH) -> dict[str, str]:
    if path.resolve() != CONFIG_PATH.resolve():
        raise DevelopmentScreenValidationError(
            "development screen requires its repository-frozen config path"
        )
    if not path.is_file():
        raise DevelopmentScreenValidationError("development config is missing")
    export_re = re.compile(r'export ([A-Z][A-Z0-9_]*)="([^"\\]*)"\Z')
    values: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = export_re.fullmatch(line)
        if match is None:
            raise DevelopmentScreenValidationError(
                f"config line {line_number} is not a literal export"
            )
        key, value = match.groups()
        if key in values:
            raise DevelopmentScreenValidationError(f"config repeats {key}")
        values[key] = value
    missing = sorted(set(EXPECTED_CONFIG) - set(values))
    extra = sorted(set(values) - set(EXPECTED_CONFIG))
    changed = sorted(
        key
        for key in EXPECTED_CONFIG.keys() & values.keys()
        if values[key] != EXPECTED_CONFIG[key]
    )
    if missing or extra or changed:
        raise DevelopmentScreenValidationError(
            "development config mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    broker_path = REPOSITORY_ROOT / "reproduction/paste_repro/live_broker.py"
    if sha256_file(broker_path) != EXPECTED_LIVE_BROKER_SHA256:
        raise DevelopmentScreenValidationError(
            "live_broker.py differs from the frozen fair-reservation implementation"
        )
    return values


def validate_development_workload(path: Path = TUNE_WORKLOAD) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved != TUNE_WORKLOAD.resolve():
        raise DevelopmentScreenValidationError(
            "only frozen tune-v1 may enter the development screen"
        )
    raw_sha = sha256_file(resolved)
    if raw_sha != TUNE_WORKLOAD_SHA256 or raw_sha in FORMAL_WORKLOAD_SHA256S:
        raise DevelopmentScreenValidationError(
            "development workload SHA is wrong or aliases a formal workload"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if (
        payload.get("schema_version") != 1
        or payload.get("split_id") != TUNE_SPLIT_ID
        or payload.get("split_role") != "tune"
        or payload.get("formal_eligible") is not False
        or not isinstance(sources, list)
        or len(sources) != SOURCE_COUNT
    ):
        raise DevelopmentScreenValidationError(
            "workload is not the exact 16-source non-formal tune split"
        )
    expected_ids = [f"tune-wiki{index:03d}" for index in range(1, 17)]
    if [row.get("source_id") for row in sources] != expected_ids:
        raise DevelopmentScreenValidationError("tune source order differs")
    for index, source in enumerate(sources):
        if (
            not isinstance(source, Mapping)
            or not isinstance(source.get("question"), str)
            or not source["question"].strip()
            or not isinstance(source.get("search_query"), str)
            or not source["search_query"].strip()
            or not isinstance(source.get("expected_url"), str)
            or not source["expected_url"].startswith(
                "https://en.wikipedia.org/wiki/"
            )
        ):
            raise DevelopmentScreenValidationError(
                f"tune source {index} is incomplete"
            )
    if canonical_sha256(payload) != TUNE_CANONICAL_SHA256:
        raise DevelopmentScreenValidationError("tune canonical payload SHA differs")
    if canonical_sha256(sources) != TUNE_SOURCES_SHA256:
        raise DevelopmentScreenValidationError("tune canonical sources SHA differs")
    return {
        "schema": "paste_repro.live_joint_v9_development_workload_validation",
        "version": 1,
        "valid": True,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "split_id": TUNE_SPLIT_ID,
        "split_role": "tune",
        "source_count": SOURCE_COUNT,
        "replicas": REPLICAS,
        "task_count": TASK_COUNT,
        "file_sha256": raw_sha,
        "canonical_json_sha256": TUNE_CANONICAL_SHA256,
        "canonical_sources_sha256": TUNE_SOURCES_SHA256,
        "formal_workload_sha256s_rejected": sorted(FORMAL_WORKLOAD_SHA256S),
    }


def _expected_result_config(
    *, cell: str, block_id: str, order_index: int, server_instance_id: str,
    visit_interval_s: float,
) -> dict[str, Any]:
    treatment = CELL_TREATMENTS[cell]
    return {
        "call_graph_mode": "frozen",
        "speculation_mode": treatment["speculation_mode"],
        "tool_signal_policy": "execution_aware",
        "tool_signal_policy_version": (
            "exact-session-invocation-running-completed-v1"
        ),
        "tool_signal_policy_module_sha256": formal.LIVE_AGENT_SHA256,
        "independent_source_count": SOURCE_COUNT,
        "replicas": REPLICAS,
        "task_count": TASK_COUNT,
        "max_active_tasks": TASK_COUNT,
        "tool_workers": 4,
        "speculative_tool_workers": 2,
        "min_speculative_tool_workers": treatment[
            "min_speculative_tool_workers"
        ],
        "search_tool_capacity": 3,
        "visit_tool_capacity": 2,
        "search_min_start_interval_s": 0.0,
        "visit_min_start_interval_s": visit_interval_s,
        "max_speculative_pending": 128,
        "speculative_ttl_s": 120.0,
        "tool_http_max_attempts": 2,
        "tool_http_retry_backoff_s": 1.0,
        "tool_http_attempt_start_gate_enabled": True,
        "tool_http_attempt_start_gate_policy_version": (
            "shared-per-tool-monotonic-v1"
        ),
        "tool_http_attempt_min_start_intervals_s": {
            "visit": visit_interval_s
        },
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
        "visit_canary_stride": CANARY_STRIDE,
        "context_padding_tokens": CONTEXT_PADDING_TOKENS,
        "fixed_final_completion_tokens": FIXED_FINAL_COMPLETION_TOKENS,
        "fixed_final_completion_enabled": True,
        "final_answer_contract_policy_version": (
            formal.FINAL_ANSWER_CONTRACT_POLICY_VERSION
        ),
        "final_answer_schema_policy_version": (
            formal.FINAL_ANSWER_SCHEMA_POLICY_VERSION
        ),
        "final_answer_grammar_policy_version": (
            formal.FINAL_ANSWER_GRAMMAR_POLICY_VERSION
        ),
        "final_answer_grammar_xgrammar_version": (
            formal.FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION
        ),
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
            "cell_id": treatment["formal_cell_id"],
            "order_index": order_index,
            "server_instance_id": server_instance_id,
            "fresh_server": True,
            "result_cache_empty": True,
            "broker_drained": True,
        },
    }


def _validate_scheduler_environment(
    run: pair.ValidatedRun, *, cell: str, label: str
) -> None:
    scheduler = run.config.get("scheduler_environment")
    if not isinstance(scheduler, Mapping):
        raise DevelopmentScreenValidationError(f"{label} scheduler is missing")
    treatment = CELL_TREATMENTS[cell]
    expected = {
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
        "VLLM_PORT": "8100",
        "VLLM_MAX_MODEL_LEN": "16384",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_MAX_NUM_SEQS": "96",
        "VLLM_ENABLE_PREFIX_CACHING": "1",
        "VLLM_USE_V1": "1",
        "VLLM_SCHED_POLICY": treatment["scheduler"],
    }
    if treatment["scheduler"] == "fcfs":
        leaked = sorted(
            key
            for key, value in scheduler.items()
            if key.startswith("VLLM_SCHED_")
            and key != "VLLM_SCHED_POLICY"
            and value is not None
        )
        if leaked:
            raise DevelopmentScreenValidationError(
                f"{label} native FCFS leaked Joint knobs: {leaked}"
            )
    else:
        expected.update(
            {
                key: EXPECTED_CONFIG[key]
                for key in formal.FROZEN_JOINT_SCHEDULER_ENV_KEYS
            }
        )
    changed = sorted(
        key for key, expected_value in expected.items()
        if scheduler.get(key) != expected_value
    )
    if changed:
        raise DevelopmentScreenValidationError(
            f"{label} scheduler environment mismatch: {changed}"
        )


def _validate_tune_task_identity(run: pair.ValidatedRun, *, label: str) -> None:
    workload = json.loads(TUNE_WORKLOAD.read_text(encoding="utf-8"))
    expected_sources = {
        str(row["source_id"]): row for row in workload["sources"]
    }
    expected_keys = {
        (source_id, replica)
        for source_id in expected_sources
        for replica in range(REPLICAS)
    }
    if set(run.tasks_by_key) != expected_keys:
        raise DevelopmentScreenValidationError(
            f"{label} source/replica identity matrix differs"
        )
    for task_index in range(TASK_COUNT):
        source_index = task_index // REPLICAS + 1
        replica = task_index % REPLICAS
        source_id = f"tune-wiki{source_index:03d}"
        task_id = f"{source_id}__r{replica:02d}"
        task = run.tasks_by_id.get(task_id)
        source = expected_sources[source_id]
        if (
            not isinstance(task, Mapping)
            or task.get("search_query") != source["search_query"]
            or task.get("question_sha256")
            != hashlib.sha256(source["question"].encode("utf-8")).hexdigest()
            or task.get("expected_url") != source["expected_url"]
            or task.get("selected_url") != source["expected_url"]
            or task.get("visit_canary") != (task_index % CANARY_STRIDE == 0)
            or task.get("context_padding_target_tokens")
            != CONTEXT_PADDING_TOKENS
        ):
            raise DevelopmentScreenValidationError(
                f"{label}/{task_id} violates tune identity/canary/padding"
            )


def _validate_broker_drained(run: pair.ValidatedRun, *, label: str) -> None:
    snapshot = run.payload.get("broker_final_snapshot")
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
        raise DevelopmentScreenValidationError(f"{label} broker did not drain")


def _load_gate(run: pair.ValidatedRun, *, label: str) -> dict[str, Any]:
    scheduler = run.config["scheduler_environment"]
    max_sequences = int(scheduler["VLLM_MAX_NUM_SEQS"])
    offered = int(run.config["max_active_tasks"])
    llm_rows = [row for row in run.timeline if row.get("llm_waiting") is not None]
    if not llm_rows:
        raise DevelopmentScreenValidationError(f"{label} has no LLM samples")
    waiting = sum(
        float(row["llm_waiting"]) > 0
        and float(row["llm_running"]) < max_sequences
        for row in llm_rows
    )
    tool_wait = sum(
        int(row.get("tool_queued_authoritative", 0) or 0) > 0
        for row in run.timeline
    )
    longest_count = 0
    longest_span = 0.0
    streak_start: float | None = None
    previous_mono: float | None = None
    previous_wall: float | None = None
    streak_count = 0
    maximum_in_streak_gap = 0.0
    dual_count = 0
    for row in run.timeline:
        dual = (
            row.get("llm_waiting") is not None
            and float(row["llm_waiting"]) > 0
            and int(row.get("tool_queued_authoritative", 0) or 0) > 0
        )
        if not dual:
            streak_start = previous_mono = previous_wall = None
            streak_count = 0
            continue
        dual_count += 1
        mono = float(row["monotonic_s"])
        wall = float(row["wall_s"])
        if not math.isfinite(mono) or not math.isfinite(wall):
            raise DevelopmentScreenValidationError(
                f"{label} has non-finite timeline timestamps"
            )
        if previous_mono is not None and previous_wall is not None:
            mono_gap = mono - previous_mono
            wall_gap = wall - previous_wall
            if mono_gap < 0 or wall_gap < 0:
                raise DevelopmentScreenValidationError(
                    f"{label} timeline timestamps decrease"
                )
            gap = max(mono_gap, wall_gap)
            if gap > 0.5:
                streak_start = None
                streak_count = 0
            else:
                maximum_in_streak_gap = max(maximum_in_streak_gap, gap)
        if streak_start is None:
            streak_start = mono
            streak_count = 1
        else:
            streak_count += 1
        longest_count = max(longest_count, streak_count)
        longest_span = max(longest_span, mono - streak_start)
        previous_mono, previous_wall = mono, wall
    observed = {
        "offered_concurrency": offered,
        "vllm_max_num_seqs": max_sequences,
        "llm_metric_sample_count": len(llm_rows),
        "native_waiting_below_cap_sample_count": waiting,
        "native_waiting_below_cap_fraction": waiting / len(llm_rows),
        "timeline_sample_count": len(run.timeline),
        "authoritative_tool_queue_sample_count": tool_wait,
        "authoritative_tool_queue_sample_fraction": tool_wait / len(run.timeline),
        "dual_queue_pressure_sample_count": dual_count,
        "longest_continuous_dual_sample_count": longest_count,
        "longest_continuous_dual_span_s": longest_span,
        "maximum_adjacent_gap_within_continuous_streak_s": maximum_in_streak_gap,
    }
    observed["passed"] = bool(
        offered == TASK_COUNT
        and 64 < offered < max_sequences == 96
        and observed["native_waiting_below_cap_fraction"] >= 0.05
        and observed["authoritative_tool_queue_sample_fraction"] >= 0.05
        and dual_count >= 10
        and longest_span >= 1.0
        and maximum_in_streak_gap <= 0.5
    )
    return observed


def _broker_stats(run: pair.ValidatedRun, *, label: str) -> Mapping[str, Any]:
    summary = run.payload.get("summary")
    if not isinstance(summary, Mapping):
        raise DevelopmentScreenValidationError(f"{label} summary is missing")
    tool = summary.get("tool")
    stats = tool.get("broker_stats") if isinstance(tool, Mapping) else None
    if not isinstance(stats, Mapping):
        raise DevelopmentScreenValidationError(f"{label} broker stats missing")
    return stats


def validate_dispatch_ledger(
    run: pair.ValidatedRun, *, label: str
) -> dict[str, Any]:
    """Replay every started broker dispatch from causal before/after fields."""

    minimum = int(run.config.get("min_speculative_tool_workers", -1))
    maximum = int(run.config.get("speculative_tool_workers", -1))
    if minimum not in {0, 1} or maximum != 2 or minimum > maximum:
        raise DevelopmentScreenValidationError(
            f"{label} has invalid speculative worker limits"
        )
    by_tool: dict[str, list[tuple[int, int, Mapping[str, Any]]]] = defaultdict(list)
    started_count = 0
    allowed_reasons = {
        "reserved_speculative",
        "speculative_minimum_uncontended",
        "authoritative_repayment",
        "authoritative_priority",
        "speculative_opportunistic",
    }
    reason_counts = {reason: 0 for reason in sorted(allowed_reasons)}
    for record_index, record in enumerate(run.physical_records):
        started = record.get("started_at")
        fields = {
            "dispatch_lane": record.get("dispatch_lane"),
            "dispatch_reason": record.get("dispatch_reason"),
            "running_speculative_before": record.get(
                "running_speculative_before"
            ),
            "queued_authoritative_same_tool_before": record.get(
                "queued_authoritative_same_tool_before"
            ),
            "reservation_debt_before": record.get("reservation_debt_before"),
            "reservation_debt_after": record.get("reservation_debt_after"),
            "per_tool_dispatch_ordinal": record.get(
                "per_tool_dispatch_ordinal"
            ),
        }
        if started is None:
            if any(value is not None for value in fields.values()):
                raise DevelopmentScreenValidationError(
                    f"{label} unstarted row {record_index} has dispatch telemetry"
                )
            continue
        started_count += 1
        lane = fields["dispatch_lane"]
        reason = fields["dispatch_reason"]
        running_before = fields["running_speculative_before"]
        queued_same = fields["queued_authoritative_same_tool_before"]
        debt_before = fields["reservation_debt_before"]
        debt_after = fields["reservation_debt_after"]
        ordinal = fields["per_tool_dispatch_ordinal"]
        if lane not in {"authoritative", "speculative"}:
            raise DevelopmentScreenValidationError(
                f"{label} row {record_index} has invalid dispatch lane"
            )
        if reason not in allowed_reasons:
            raise DevelopmentScreenValidationError(
                f"{label} row {record_index} has invalid dispatch reason"
            )
        if (
            isinstance(running_before, bool)
            or not isinstance(running_before, int)
            or running_before < 0
            or running_before > maximum
            or isinstance(queued_same, bool)
            or not isinstance(queued_same, int)
            or queued_same < 0
            or not isinstance(debt_before, bool)
            or not isinstance(debt_after, bool)
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal <= 0
        ):
            raise DevelopmentScreenValidationError(
                f"{label} row {record_index} has invalid dispatch state"
            )
        expected_reserved = reason == "reserved_speculative"
        expected_repayment = reason == "authoritative_repayment"
        if (
            record.get("reserved_speculative_dispatch") is not expected_reserved
            or record.get("authoritative_after_reserved_dispatch")
            is not expected_repayment
        ):
            raise DevelopmentScreenValidationError(
                f"{label} row {record_index} reason/legacy flags disagree"
            )
        if reason == "reserved_speculative":
            valid_reason = (
                minimum == 1
                and lane == "speculative"
                and record.get("speculative") is True
                and running_before < minimum
                and queued_same > 0
                and debt_before is False
                and debt_after is True
            )
        elif reason == "speculative_minimum_uncontended":
            valid_reason = (
                minimum == 1
                and lane == "speculative"
                and running_before < minimum
                and queued_same == 0
                and debt_before is debt_after
            )
        elif reason == "authoritative_repayment":
            valid_reason = (
                lane == "authoritative"
                and queued_same > 0
                and debt_before is True
                and debt_after is False
            )
        elif reason == "authoritative_priority":
            valid_reason = (
                lane == "authoritative"
                and queued_same > 0
                and debt_before is False
                and debt_after is False
            )
        else:
            valid_reason = (
                lane == "speculative"
                and running_before < maximum
                and debt_before is debt_after
            )
        if not valid_reason:
            raise DevelopmentScreenValidationError(
                f"{label} row {record_index} cannot replay reason {reason}"
            )
        tool = str(record.get("tool"))
        by_tool[tool].append((ordinal, record_index, record))
        reason_counts[str(reason)] += 1

    replay_rows: list[dict[str, Any]] = []
    final_debt_tools: list[str] = []
    for tool, rows in sorted(by_tool.items()):
        debt = False
        ordered = sorted(rows)
        if [ordinal for ordinal, _index, _record in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise DevelopmentScreenValidationError(
                f"{label}/{tool} dispatch ordinals are not contiguous"
            )
        for ordinal, record_index, record in ordered:
            before = bool(record["reservation_debt_before"])
            after = bool(record["reservation_debt_after"])
            if before is not debt:
                raise DevelopmentScreenValidationError(
                    f"{label}/{tool} row {record_index} debt is not replayable"
                )
            debt = after
            replay_rows.append(
                {
                    "tool": tool,
                    "record_index": record_index,
                    "per_tool_dispatch_ordinal": ordinal,
                    "dispatch_lane": record["dispatch_lane"],
                    "dispatch_reason": record["dispatch_reason"],
                    "running_speculative_before": record[
                        "running_speculative_before"
                    ],
                    "queued_authoritative_same_tool_before": record[
                        "queued_authoritative_same_tool_before"
                    ],
                    "reservation_debt_before": before,
                    "reservation_debt_after": after,
                }
            )
        if debt:
            final_debt_tools.append(tool)
    snapshot = run.payload.get("broker_final_snapshot")
    reservation = (
        snapshot.get("reservation") if isinstance(snapshot, Mapping) else None
    )
    stats = snapshot.get("stats") if isinstance(snapshot, Mapping) else None
    if not isinstance(reservation, Mapping) or not isinstance(stats, Mapping):
        raise DevelopmentScreenValidationError(
            f"{label} final reservation snapshot is missing"
        )
    reserved_count = reason_counts["reserved_speculative"]
    repayment_count = reason_counts["authoritative_repayment"]
    if (
        final_debt_tools
        or reservation.get("authoritative_turn_due_by_tool") != []
        or int(reservation.get("reserved_speculative_dispatches", -1))
        != reserved_count
        or int(reservation.get("authoritative_after_reserved_dispatches", -1))
        != repayment_count
        or int(stats.get("reserved_speculative_dispatches", -1))
        != reserved_count
        or int(stats.get("authoritative_after_reserved_dispatches", -1))
        != repayment_count
    ):
        raise DevelopmentScreenValidationError(
            f"{label} replay/stats/final reservation debt disagree"
        )
    if minimum == 0 and (reserved_count or repayment_count):
        raise DevelopmentScreenValidationError(
            f"{label} min=0 produced reservation events"
        )
    return {
        "started_dispatch_count": started_count,
        "min_speculative_tool_workers": minimum,
        "max_speculative_tool_workers": maximum,
        "reason_counts": reason_counts,
        "reserved_speculative_dispatch_count": reserved_count,
        "authoritative_repayment_count": repayment_count,
        "completed_reuse_ready_hit_count": int(stats.get("completed_reuse", 0)),
        "debt_domain": [0, 1],
        "final_debt_tools": final_debt_tools,
        "final_debt_zero": not final_debt_tools,
        "per_tool_ordinals_contiguous": True,
        "all_dispatch_rows_causally_replayed": True,
        "replay_rows": replay_rows,
    }


def validate_cell_result(
    *, result_path: Path, timeline_path: Path, cell: str, block_id: str,
    order_index: int, server_instance_id: str, visit_interval_s: float,
    stage: str,
) -> dict[str, Any]:
    """Validate a stage-0 A or stage-1 E/F0/F1 cell from raw evidence."""

    if cell not in CELL_TREATMENTS:
        raise DevelopmentScreenValidationError(f"unknown cell: {cell}")
    if stage not in {"stage0", "stage1"}:
        raise DevelopmentScreenValidationError(f"unknown stage: {stage}")
    if (stage == "stage0") != (cell == "A"):
        raise DevelopmentScreenValidationError("stage/cell combination is invalid")
    if visit_interval_s not in TRANSPORT_LADDER_S:
        raise DevelopmentScreenValidationError("interval is outside frozen ladder")

    validate_development_workload()
    role = "baseline" if cell in {"A", "E"} else "candidate"
    try:
        run = pair._validate_run(
            result_path.resolve(),
            role=role,
            timeline_override=timeline_path.resolve(),
        )
    except ValueError as exc:
        raise DevelopmentScreenValidationError(str(exc)) from exc
    label = f"{stage}/{block_id}/{cell}"
    expected_config = _expected_result_config(
        cell=cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        visit_interval_s=visit_interval_s,
    )
    changed = sorted(
        key for key, expected in expected_config.items()
        if run.config.get(key) != expected
    )
    if changed:
        raise DevelopmentScreenValidationError(
            f"{label} frozen result config mismatch: {changed}"
        )
    if (
        run.config.get("workload_formal_eligible") is not False
        or run.config.get("workload_split_role") != "tune"
        or run.config.get("workload_file_sha256") in FORMAL_WORKLOAD_SHA256S
    ):
        raise DevelopmentScreenValidationError(
            f"{label} attempted to use formal evidence"
        )
    _validate_scheduler_environment(run, cell=cell, label=label)
    _validate_tune_task_identity(run, label=label)
    _validate_broker_drained(run, label=label)

    task_count = len(run.tasks_by_key)
    llm_count = sum(len(events) for events in run.llm_by_task.values())
    commit_count = len(run.committed_by_task_tool)
    if (
        task_count != TASK_COUNT
        or llm_count != LLM_REQUEST_COUNT
        or commit_count != AUTHORITATIVE_COMMIT_COUNT
        or len(run.physical_records) != AUTHORITATIVE_COMMIT_COUNT
    ):
        raise DevelopmentScreenValidationError(
            f"{label} is not exact 80/240/160 evidence"
        )

    try:
        physical = formal_aggregate._validate_physical_run(
            run, label, require_http_attempt_logs=True
        )
        fixed_final = formal_aggregate._validate_fixed_final_completion_contract(
            run,
            label,
            expected_completion_tokens=FIXED_FINAL_COMPLETION_TOKENS,
        )
        guided = formal_aggregate._validate_zero_guided_json_recovery(
            run, label, expected_parsed_call_count=2
        )
        canary = formal_aggregate._validate_canary_pre_enqueue_skip(
            run, label, expected_count=CANARY_COUNT
        )
    except ValueError as exc:
        raise DevelopmentScreenValidationError(str(exc)) from exc

    if physical["http_attempt_count_by_tool"] is None:
        raise DevelopmentScreenValidationError(f"{label} lacks attempt ledgers")
    visit_starts: list[float] = []
    attempt1_count = 0
    for index, record in enumerate(run.physical_records):
        raw_log = record.get("http_attempt_log")
        attempts = record.get("http_attempts")
        if attempts == 1:
            attempt1_count += 1
        if not isinstance(raw_log, list) or len(raw_log) != attempts:
            raise DevelopmentScreenValidationError(
                f"{label} record {index} attempt ledger differs"
            )
        if record.get("tool") == "visit":
            for attempt in raw_log:
                if not isinstance(attempt, Mapping):
                    raise DevelopmentScreenValidationError(
                        f"{label} record {index} attempt is invalid"
                    )
                started_attempt = attempt.get("started_monotonic_s")
                if (
                    isinstance(started_attempt, bool)
                    or not isinstance(started_attempt, (int, float))
                    or not math.isfinite(float(started_attempt))
                ):
                    raise DevelopmentScreenValidationError(
                        f"{label} record {index} attempt lacks monotonic start"
                    )
                visit_starts.append(float(started_attempt))
    if any(
        right - left < visit_interval_s - HTTP_SPACING_TOLERANCE_S
        for left, right in zip(sorted(visit_starts), sorted(visit_starts)[1:])
    ):
        raise DevelopmentScreenValidationError(
            f"{label} violates the selected visit HTTP-attempt spacing"
        )

    stats = _broker_stats(run, label=label)
    dispatch_replay = validate_dispatch_ledger(run, label=label)
    failed_count = int(physical["failed_physical_job_count"])
    retry_count = int(physical["retried_physical_job_count"])
    authoritative_retry_count = int(
        physical["authoritative_retried_commit_count"]
    )
    waste_s = float(physical["wasted_speculative_worker_s"])
    exact_hits = int(stats.get("queued_promotions", 0)) + int(
        stats.get("running_promotions", 0)
    ) + int(stats.get("completed_reuse", 0))
    expected_speculative_visits = 0 if cell in {"A", "E"} else ELIGIBLE_VISIT_COUNT
    if canary["speculative_visit_record_count"] != expected_speculative_visits:
        raise DevelopmentScreenValidationError(
            f"{label} speculative visit count is not {expected_speculative_visits}"
        )
    if cell in {"F0", "F1"} and exact_hits != ELIGIBLE_VISIT_COUNT:
        raise DevelopmentScreenValidationError(
            f"{label} does not have all {ELIGIBLE_VISIT_COUNT} exact eligible hits"
        )
    if cell in {"A", "E"} and exact_hits != 0:
        raise DevelopmentScreenValidationError(
            f"{label} spec-off cell reports speculative hits"
        )

    correctness_observed = {
        "task_count": task_count,
        "llm_request_count": llm_count,
        "authoritative_commit_count": commit_count,
        "physical_record_count": len(run.physical_records),
        "failed_physical_job_count": failed_count,
        "wasted_speculative_worker_s": waste_s,
        "fixed_final_exact_task_count": fixed_final[
            "exact_completion_token_task_count"
        ],
        "guided_json_recovery_count": guided["recovery_count"],
        "canary_pre_enqueue_speculative_record_count": canary[
            "canary_speculative_record_count"
        ],
        "exact_eligible_visit_hit_count": exact_hits,
    }
    correctness_passed = bool(
        task_count == TASK_COUNT
        and llm_count == LLM_REQUEST_COUNT
        and commit_count == AUTHORITATIVE_COMMIT_COUNT
        and len(run.physical_records) == AUTHORITATIVE_COMMIT_COUNT
        and failed_count == 0
        and math.isclose(waste_s, 0.0, abs_tol=1e-9)
        and fixed_final["all_completion_tokens_exact"] is True
        and guided["recovery_count"] == 0
        and canary["canary_speculative_record_count"] == 0
    )
    if not correctness_passed:
        raise DevelopmentScreenValidationError(
            f"{label} failed a non-transport correctness gate"
        )

    load = _load_gate(run, label=label) if stage == "stage0" else None
    transport_observed = {
        "physical_record_count": len(run.physical_records),
        "attempt1_record_count": attempt1_count,
        "retried_physical_job_count": retry_count,
        "authoritative_retried_commit_count": authoritative_retry_count,
        "authoritative_retry_rate": physical["authoritative_retry_rate"],
        "maximum_allowed_authoritative_retry_rate": 0.02,
    }
    # One composite gate makes the fallback condition unambiguous: stage 0
    # can have exactly one failed gate, and it must be this transport gate.
    transport_passed = bool(
        attempt1_count == len(run.physical_records)
        and retry_count == 0
        and authoritative_retry_count == 0
        and physical["authoritative_retry_rate"] <= 0.02
    )
    gates = {
        "non_transport_correctness": _gate(
            correctness_observed,
            "exact 80/240/160, zero fail/waste, exact fixed-192, zero recovery, canary pre-enqueue skip",
            correctness_passed,
        ),
        "transport_zero_retry_and_at_most_2pct": _gate(
            transport_observed,
            "all physical records attempt1; zero retry; authoritative retry rate <=0.02",
            transport_passed,
        ),
    }
    if stage == "stage0":
        assert load is not None
        gates["natural_dual_queue_load"] = _gate(
            load,
            "wait>=5%, authoritative queue>=5%, dual>=10, continuous>=1s with adjacent gap<=0.5s; 80<96",
            bool(load["passed"]),
        )
    failed_gates = sorted(name for name, gate in gates.items() if not gate["passed"])
    accepted = not failed_gates
    retry_only_fallback_eligible = bool(
        stage == "stage0"
        and failed_gates == ["transport_zero_retry_and_at_most_2pct"]
    )
    if stage == "stage1" and not accepted:
        raise DevelopmentScreenValidationError(
            f"{label} failed strict stage-1 gates: {failed_gates}"
        )

    summary = run.summary
    result = {
        "schema": "paste_repro.live_joint_v9_development_cell_validation",
        "version": 1,
        "valid": True,
        "accepted": accepted,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "stage": stage,
        "cell_id": cell,
        "block_id": block_id,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "visit_interval_s": visit_interval_s,
        "failed_gates": failed_gates,
        "retry_only_fallback_eligible": retry_only_fallback_eligible,
        "gates": gates,
        "workload_validation": validate_development_workload(),
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path.resolve()),
        "timeline_path": str(timeline_path.resolve()),
        "timeline_sha256": sha256_file(timeline_path.resolve()),
        "selected_workload_sha256": run.config.get("selected_workload_sha256"),
        "physical": physical,
        "fixed_final": fixed_final,
        "guided_json": guided,
        "canary": canary,
        "load": load,
        "performance_observed_but_not_used_for_transport_selection": {
            "task_e2e_s": summary["task_e2e_s"],
            "task_completion_makespan_s": summary[
                "task_completion_makespan_s"
            ],
        },
        "broker_mechanism": {
            "exact_hit_count": exact_hits,
            "queued_promotions": int(stats.get("queued_promotions", 0)),
            "running_promotions": int(stats.get("running_promotions", 0)),
            "completed_reuse_ready_hits": int(stats.get("completed_reuse", 0)),
            "reserved_speculative_dispatches": int(
                stats.get("reserved_speculative_dispatches", 0)
            ),
            "authoritative_after_reserved_dispatches": int(
                stats.get("authoritative_after_reserved_dispatches", 0)
            ),
        },
        "dispatch_replay": dispatch_replay,
    }
    return result


def select_transport_interval(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the prospective baseline-only transport fallback ladder."""

    if not attempts or len(attempts) > len(TRANSPORT_LADDER_S):
        raise DevelopmentScreenValidationError("invalid stage-0 attempt count")
    observed_intervals = [float(row.get("visit_interval_s", -1)) for row in attempts]
    if observed_intervals != list(TRANSPORT_LADDER_S[: len(attempts)]):
        raise DevelopmentScreenValidationError(
            "stage-0 attempts do not follow the frozen 2.5s -> 3.0s ladder"
        )
    for index, row in enumerate(attempts):
        if (
            row.get("valid") is not True
            or row.get("stage") != "stage0"
            or row.get("cell_id") != "A"
            or row.get("development_only") is not True
            or row.get("formal_evidence_eligible") is not False
        ):
            raise DevelopmentScreenValidationError(
                f"stage-0 attempt {index + 1} is not development baseline A"
            )
    server_ids = [str(row.get("server_instance_id", "")) for row in attempts]
    if any(not value for value in server_ids) or len(set(server_ids)) != len(server_ids):
        raise DevelopmentScreenValidationError(
            "stage-0 fallback did not use unique fresh server instances"
        )
    first = attempts[0]
    if first.get("accepted") is True:
        if first.get("failed_gates") != [] or first.get(
            "retry_only_fallback_eligible"
        ) is not False:
            raise DevelopmentScreenValidationError(
                "accepted 2.5s attempt has inconsistent gate state"
            )
        if len(attempts) != 1:
            raise DevelopmentScreenValidationError(
                "3.0s fallback ran even though 2.5s already passed"
            )
        selected = 2.5
        reason = "first_zero_retry_load_qualified_baseline"
    elif (
        first.get("retry_only_fallback_eligible") is True
        and first.get("failed_gates")
        == ["transport_zero_retry_and_at_most_2pct"]
    ):
        if len(attempts) != 2:
            raise DevelopmentScreenValidationError(
                "retry-only 2.5s failure requires exactly one fresh 3.0s A"
            )
        second = attempts[1]
        if (
            second.get("accepted") is not True
            or second.get("failed_gates") != []
            or second.get("retry_only_fallback_eligible") is not False
        ):
            raise DevelopmentScreenValidationError(
                "3.0s fallback did not pass every stage-0 gate"
            )
        selected = 3.0
        reason = "retry_only_fallback_then_first_zero_retry_baseline"
    else:
        raise DevelopmentScreenValidationError(
            "2.5s failed a non-transport gate; fallback is forbidden"
        )
    return {
        "schema": "paste_repro.live_joint_v9_development_transport_selection",
        "version": 1,
        "valid": True,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "candidate_performance_observed_or_used": False,
        "selection_input_cells": ["A" for _ in attempts],
        "registered_ladder_s": list(TRANSPORT_LADDER_S),
        "selected_visit_interval_s": selected,
        "selection_reason": reason,
        "attempt_count": len(attempts),
        "attempt_summaries": [
            {
                "visit_interval_s": row["visit_interval_s"],
                "accepted": row["accepted"],
                "failed_gates": row["failed_gates"],
                "retry_only_fallback_eligible": row[
                    "retry_only_fallback_eligible"
                ],
            }
            for row in attempts
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--cell", choices=sorted(CELL_TREATMENTS), required=True)
    parser.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--order-index", type=int, required=True)
    parser.add_argument("--server-instance-id", required=True)
    parser.add_argument(
        "--visit-interval-s", type=float, choices=TRANSPORT_LADDER_S, required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validation = validate_cell_result(
        result_path=args.result,
        timeline_path=args.timeline,
        cell=args.cell,
        block_id=args.block_id,
        order_index=args.order_index,
        server_instance_id=args.server_instance_id,
        visit_interval_s=args.visit_interval_s,
        stage=args.stage,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validation["accepted"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DevelopmentScreenValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
