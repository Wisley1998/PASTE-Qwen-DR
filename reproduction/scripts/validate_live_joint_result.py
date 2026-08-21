#!/usr/bin/env python3
"""Validate a live tool--LLM closed-loop experiment result.

The validator deliberately reads raw task, LLM, and physical tool-job events.
Reported means alone are not accepted.  Performance thresholds are constants in
this module and in the prospective protocol; a result file cannot relax them.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "reproduction"
    / "results"
    / "live_joint"
    / "LIVE_TOOL_LLM_PROTOCOL.md"
)

SCHEMA = "paste_repro.live_joint_experiment"
VERSION = 1
VALIDATION_SCHEMA = "paste_repro.live_joint_validation"
VALIDATION_VERSION = 1

REQUIRED_CELLS = ("A", "B", "E", "F")
OPTIONAL_DIAGNOSTIC_CELLS = ("C", "D")
POLICIES = {
    "A": ("fcfs_native", "demand_only"),
    "B": ("fcfs_native", "resource_aware_speculation"),
    "C": ("joint_native", "demand_only"),
    "D": ("joint_native", "resource_aware_speculation"),
    "E": ("joint_physical_kv", "demand_only"),
    "F": ("joint_physical_kv", "resource_aware_speculation"),
}
EVIDENCE_KINDS = {
    "frozen_config",
    "task_events",
    "llm_events",
    "tool_events",
    "resource_samples",
    "prefix_samples",
    "server_log",
    "tool_server_log",
}
PREFIX_EVIDENCE_KINDS = {
    "frozen_config",
    "task_events",
    "prefix_samples",
    "server_log",
}

FORMAL_SOURCE_COUNT = 60
FORMAL_BLOCK_COUNT = 3
SCREENING_MIN_SOURCES = 12
SCREENING_MAX_SOURCES = 20

MAX_TOKEN_RELATIVE_DIFFERENCE = 0.01
SCREENING_MIN_SPEC_E2E_REDUCTION = 0.03
FORMAL_MIN_SPEC_E2E_REDUCTION = 0.05
SCREENING_MIN_SOURCE_FASTER_FRACTION = 0.60
FORMAL_MIN_SOURCE_FASTER_COUNT = 42
FORMAL_MIN_OVERALL_E2E_REDUCTION = 0.25
FORMAL_MIN_OVERALL_FASTER_COUNT = 48
FORMAL_MAX_REQUEST_P99_RATIO = 1.25
SCREENING_MAX_AUTHORITATIVE_RETRY_RATE = 0.05
FORMAL_MAX_AUTHORITATIVE_RETRY_RATE = 0.02
SCREENING_MAX_RETRY_RATE_DIFFERENCE = 0.03
FORMAL_MAX_RETRY_RATE_DIFFERENCE = 0.01
CONTROLLED_HTTP_MAX_ATTEMPTS = 2
CONTROLLED_HTTP_RETRY_BACKOFF_S = 1.0
CONTROLLED_HTTP_RETRY_POLICY_VERSION = "idempotent-get-v1"
HTTP_LIBRARY_RETRY_CONTROL_VERSION = "aiohttp-private-retry-connection-v1"
FORMAL_HTTP_LIBRARY_NAME = "aiohttp"
FORMAL_HTTP_LIBRARY_VERSION = "3.12.15"
CONTROLLED_HTTP_RETRYABLE_STATUSES = [429, 500, 502, 503, 504]
CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES = [
    "asyncio.TimeoutError",
    "ConnectionError",
    "aiohttp.ClientConnectionError",
    "aiohttp.ClientPayloadError",
]
SCREENING_MAX_WASTE_WORKER_FRACTION = 0.45
FORMAL_MAX_WASTE_WORKER_FRACTION = 0.30
SCREENING_MIN_SPEC_HIT_RATE = 0.10
FORMAL_MIN_SPEC_HIT_RATE = 0.20
SCREENING_MAX_CANARY_MEAN_RATIO = 1.05
SCREENING_MAX_CANARY_P95_RATIO = 1.10
FORMAL_MAX_CANARY_MEAN_RATIO = 1.03
FORMAL_MAX_CANARY_P95_RATIO = 1.05
SCREENING_MAX_TASK_P95_RATIO = 1.05
FORMAL_MAX_TASK_P95_RATIO = 1.00
MAX_MAKESPAN_RATIO = 1.03
MIN_NATIVE_QUEUE_SAMPLE_FRACTION = 0.05
MIN_TOOL_QUEUE_SAMPLE_FRACTION = 0.05
PREFIX_MIN_E2E_REDUCTION = 0.02
PREFIX_MIN_HIT_RATIO_INCREASE = 0.03
PREFIX_MAX_TASK_P95_RATIO = 1.03
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260816


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result != value or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _identifier(value: Any, label: str) -> str:
    """Return a stable textual identifier while accepting broker-native ints."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-empty string or non-negative integer")
    if isinstance(value, int) and value >= 0:
        return str(value)
    return _nonempty_string(value, label)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _relative_difference(left: float, right: float) -> float:
    if left == 0.0:
        return 0.0 if right == 0.0 else math.inf
    return abs(left - right) / left


def _load_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _resolve_evidence(
    entry: Any, *, repository_root: Path, label: str
) -> Path:
    item = _mapping(entry, label)
    relative = Path(_nonempty_string(item.get("path"), f"{label}.path"))
    if relative.is_absolute():
        raise ValueError(f"{label}.path must be repository-relative")
    resolved_root = repository_root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label}.path escapes the repository") from exc
    if not resolved.is_file():
        raise ValueError(f"{label}.path does not exist: {relative}")
    expected = _nonempty_string(item.get("sha256"), f"{label}.sha256")
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(f"{label}.sha256 is not lowercase SHA256")
    observed = _sha256_file(resolved)
    if observed != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return resolved


def _sha256_text(value: Any, label: str) -> str:
    digest = _nonempty_string(value, label)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} is not lowercase SHA256")
    return digest


def _validate_workload_manifest(
    path: Path,
    *,
    source_ids: set[str],
    split_role: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(payload, "workload manifest")
    if root.get("schema") != "paste_repro.live_joint_workload" or root.get("version") != 1:
        raise ValueError("workload manifest schema/version is unsupported")
    if root.get("split_role") != split_role:
        raise ValueError("workload manifest split_role differs from the aggregate")
    formal_eligible = _boolean(
        root.get("formal_eligible"), "workload manifest formal_eligible"
    )
    if formal_eligible != (split_role == "heldout"):
        raise ValueError("workload manifest formal eligibility is inconsistent")
    _nonempty_string(root.get("split_id"), "workload manifest split_id")
    rows = _sequence(root.get("sources"), "workload manifest sources")
    observed: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"workload source {index}")
        source_id = _nonempty_string(row.get("source_id"), f"workload source {index}.source_id")
        if source_id in observed:
            raise ValueError(f"duplicate workload source_id: {source_id}")
        observed.add(source_id)
        _nonempty_string(row.get("question"), f"workload source {index}.question")
        _nonempty_string(row.get("language"), f"workload source {index}.language")
        _nonempty_string(
            row.get("prefix_group_id"), f"workload source {index}.prefix_group_id"
        )
        _sha256_text(
            row.get("system_prompt_sha256"),
            f"workload source {index}.system_prompt_sha256",
        )
        steps = _sequence(row.get("steps"), f"workload source {index}.steps")
        kinds: list[str] = []
        search_indices: set[int] = set()
        for step_index, raw_step in enumerate(steps):
            step = _mapping(raw_step, f"workload source {index}.steps[{step_index}]")
            if _integer(
                step.get("step_index"),
                f"workload source {index}.steps[{step_index}].step_index",
            ) != step_index:
                raise ValueError(f"workload source {index} step indices are not contiguous")
            kind = _nonempty_string(
                step.get("kind"), f"workload source {index}.steps[{step_index}].kind"
            )
            if kind not in {"llm", "search", "visit"}:
                raise ValueError(f"workload source {index} has an unsupported step kind")
            kinds.append(kind)
            if kind == "llm":
                _sha256_text(
                    step.get("request_template_sha256"),
                    f"workload source {index}.steps[{step_index}].request_template_sha256",
                )
            elif kind == "search":
                arguments = _mapping(
                    step.get("arguments"),
                    f"workload source {index}.steps[{step_index}].arguments",
                )
                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"workload source {index} has an invalid search query")
                search_indices.add(step_index)
            else:
                url_from = _mapping(
                    step.get("url_from"),
                    f"workload source {index}.steps[{step_index}].url_from",
                )
                search_step = _integer(
                    url_from.get("search_step_index"),
                    f"workload source {index}.steps[{step_index}].search_step_index",
                )
                rank = _integer(
                    url_from.get("heldout_result_rank"),
                    f"workload source {index}.steps[{step_index}].heldout_result_rank",
                    positive=True,
                )
                if search_step not in search_indices or search_step >= step_index or rank <= 0:
                    raise ValueError(f"workload source {index} visit does not bind an earlier search")
        if kinds.count("llm") < 3 or "search" not in kinds or "visit" not in kinds:
            raise ValueError(f"workload source {index} lacks the required live call graph")
    if observed != source_ids:
        raise ValueError("workload manifest source IDs differ from the aggregate")


def _validate_evidence_map(
    value: Any,
    *,
    required: set[str],
    repository_root: Path,
    label: str,
) -> dict[str, Path]:
    evidence = _mapping(value, label)
    if set(evidence) != required:
        raise ValueError(
            f"{label} kinds must be exact; missing={sorted(required - set(evidence))}, "
            f"extra={sorted(set(evidence) - required)}"
        )
    return {
        kind: _resolve_evidence(
            evidence[kind], repository_root=repository_root, label=f"{label}.{kind}"
        )
        for kind in sorted(required)
    }


def _bootstrap_mean_ci(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap requires observations")
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    estimates = [
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return {
        "lower_s": _percentile(estimates, 0.025),
        "upper_s": _percentile(estimates, 0.975),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _effect(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    base_sources = base["source_task_e2e_s"]
    candidate_sources = candidate["source_task_e2e_s"]
    if set(base_sources) != set(candidate_sources):
        raise ValueError("paired cells do not contain identical source IDs")
    source_differences = [
        mean(base_sources[source]) - mean(candidate_sources[source])
        for source in sorted(base_sources)
    ]
    base_values = [item for values in base_sources.values() for item in values]
    candidate_values = [item for values in candidate_sources.values() for item in values]
    if len(base_values) != len(candidate_values):
        raise ValueError("paired cells do not contain identical task counts")
    base_mean = mean(base_values)
    candidate_mean = mean(candidate_values)
    saving = base_mean - candidate_mean
    return {
        "base_mean_s": base_mean,
        "candidate_mean_s": candidate_mean,
        "saving_s": saving,
        "relative_reduction": saving / base_mean if base_mean else None,
        "base_p95_s": _percentile(base_values, 0.95),
        "candidate_p95_s": _percentile(candidate_values, 0.95),
        "source_faster_count": sum(item > 0.0 for item in source_differences),
        "source_count": len(source_differences),
        "source_mean_saving_s": mean(source_differences),
        "source_bootstrap_95_ci_s": _bootstrap_mean_ci(source_differences),
        "source_differences_s": source_differences,
    }


def _validate_task_events(
    path: Path,
    *,
    source_ids: set[str],
    block_ids: set[str],
    copies_per_source: int,
) -> dict[str, Any]:
    rows = _load_jsonl(path, "task events")
    identities: set[str] = set()
    by_task: dict[str, Mapping[str, Any]] = {}
    source_durations: dict[str, list[float]] = defaultdict(list)
    block_bounds: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_block_source: Counter[tuple[str, str]] = Counter()
    logical_llm = 0
    logical_tool = 0
    for index, row in enumerate(rows):
        prefix = f"task event {index}"
        task_id = _nonempty_string(row.get("task_instance_id"), f"{prefix}.task_instance_id")
        if task_id in identities:
            raise ValueError(f"duplicate task_instance_id: {task_id}")
        identities.add(task_id)
        source_id = _nonempty_string(row.get("source_id"), f"{prefix}.source_id")
        block_id = _nonempty_string(row.get("block_id"), f"{prefix}.block_id")
        if source_id not in source_ids or block_id not in block_ids:
            raise ValueError(f"{prefix} is outside the frozen workload or blocks")
        if _boolean(row.get("success"), f"{prefix}.success") is not True:
            raise ValueError(f"{prefix} did not complete successfully")
        started = _finite(row.get("started_at"), f"{prefix}.started_at")
        finished = _finite(row.get("finished_at"), f"{prefix}.finished_at")
        if finished <= started:
            raise ValueError(f"{prefix} has non-positive E2E")
        llm_count = _integer(row.get("logical_llm_requests"), f"{prefix}.logical_llm_requests", positive=True)
        tool_count = _integer(row.get("logical_tool_calls"), f"{prefix}.logical_tool_calls", positive=True)
        logical_llm += llm_count
        logical_tool += tool_count
        duration = finished - started
        source_durations[source_id].append(duration)
        block_bounds[block_id].append((started, finished))
        per_block_source[(block_id, source_id)] += 1
        by_task[task_id] = row

    for block_id in block_ids:
        for source_id in source_ids:
            if per_block_source[(block_id, source_id)] != copies_per_source:
                raise ValueError(
                    f"block/source copy count mismatch: {block_id}/{source_id}"
                )
    makespans = {
        block: max(finish for _, finish in bounds) - min(start for start, _ in bounds)
        for block, bounds in block_bounds.items()
    }
    return {
        "rows": rows,
        "by_task": by_task,
        "task_count": len(rows),
        "logical_llm_requests": logical_llm,
        "logical_tool_calls": logical_tool,
        "source_task_e2e_s": dict(source_durations),
        "block_makespan_s": makespans,
        "mean_makespan_s": mean(makespans.values()),
    }


def _validate_llm_events(path: Path, tasks: Mapping[str, Any]) -> dict[str, Any]:
    rows = _load_jsonl(path, "LLM events")
    request_ids: set[str] = set()
    per_task_indices: dict[str, set[int]] = defaultdict(set)
    completion_tokens = 0
    latencies: list[float] = []
    queue_times: list[float] = []
    cached_tokens = 0
    prompt_tokens = 0
    task_map = tasks["by_task"]
    for index, row in enumerate(rows):
        prefix = f"LLM event {index}"
        request_id = _nonempty_string(row.get("request_id"), f"{prefix}.request_id")
        if request_id in request_ids:
            raise ValueError(f"duplicate request_id: {request_id}")
        request_ids.add(request_id)
        task_id = _nonempty_string(row.get("task_instance_id"), f"{prefix}.task_instance_id")
        if task_id not in task_map:
            raise ValueError(f"{prefix} references an unknown task")
        if row.get("source_id") != task_map[task_id].get("source_id"):
            raise ValueError(f"{prefix} source_id does not match its task")
        call_index = _integer(row.get("call_index"), f"{prefix}.call_index")
        if call_index in per_task_indices[task_id]:
            raise ValueError(f"{prefix} duplicates a task call_index")
        per_task_indices[task_id].add(call_index)
        if _boolean(row.get("success"), f"{prefix}.success") is not True:
            raise ValueError(f"{prefix} failed")
        if _integer(row.get("http_status"), f"{prefix}.http_status") != 200:
            raise ValueError(f"{prefix} was not HTTP 200")
        if _integer(row.get("attempts"), f"{prefix}.attempts", positive=True) != 1:
            raise ValueError(f"{prefix} was retried")
        submitted = _finite(row.get("submitted_at"), f"{prefix}.submitted_at")
        started = _finite(row.get("started_at"), f"{prefix}.started_at")
        finished = _finite(row.get("finished_at"), f"{prefix}.finished_at")
        if not submitted <= started <= finished:
            raise ValueError(f"{prefix} timestamps are not monotonic")
        prompt = _integer(row.get("prompt_tokens"), f"{prefix}.prompt_tokens", positive=True)
        completion = _integer(row.get("completion_tokens"), f"{prefix}.completion_tokens")
        cached = _integer(row.get("prefix_cached_tokens"), f"{prefix}.prefix_cached_tokens")
        if cached > prompt:
            raise ValueError(f"{prefix} cached tokens exceed prompt tokens")
        prompt_tokens += prompt
        completion_tokens += completion
        cached_tokens += cached
        latencies.append(finished - submitted)
        queue_times.append(started - submitted)

    if len(rows) != tasks["logical_llm_requests"]:
        raise ValueError("LLM event count differs from task accounting")
    for task_id, task in task_map.items():
        expected = _integer(task.get("logical_llm_requests"), "task logical_llm_requests", positive=True)
        if per_task_indices[task_id] != set(range(expected)):
            raise ValueError(f"LLM call indices are incomplete for task {task_id}")
    return {
        "request_count": len(rows),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "prefix_cached_tokens": cached_tokens,
        "mean_latency_s": mean(latencies),
        "p99_latency_s": _percentile(latencies, 0.99),
        "mean_queue_s": mean(queue_times),
    }


def _timestamp_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _close(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= max(0.01, abs(expected) * 0.02)


def _validate_tool_events(
    path: Path,
    *,
    tasks: Mapping[str, Any],
    pool_ids_by_block: Mapping[str, str],
    worker_capacity: int,
    per_tool_capacity: Mapping[str, int],
    external_live: bool,
    max_http_attempts: int,
) -> dict[str, Any]:
    rows = _load_jsonl(path, "tool events")
    required_fields = {
        "job_id", "logical_call_id", "invocation_id", "invocation_digest",
        "result_digest", "task_instance_id", "session_id", "source_id", "tool",
        "admitted", "speculative", "authoritative", "committed",
        "speculation_eligible", "canary",
        "admitted_at", "queue_enter_at", "started_at",
        "authoritative_confirmation_at", "finished_at", "outcome", "exact_match",
        "source", "cancelled", "cross_session_commit", "worker_pool", "worker_id",
        "queue_s", "service_s", "saved_service_s", "response_status",
        "bytes_read", "http_attempts", "backend", "request_host",
        "transport_identity_source",
    }
    job_ids: set[str] = set()
    invocation_ids: set[str] = set()
    logical_ids: set[str] = set()
    per_task_authoritative: Counter[str] = Counter()
    task_map = tasks["by_task"]
    speculative_jobs = 0
    speculative_tool_counts: Counter[str] = Counter()
    speculative_hits = 0
    speculative_wastes = 0
    speculative_worker_s = 0.0
    wasted_worker_s = 0.0
    canary_latencies: list[float] = []
    canary_invocations: Counter[str] = Counter()
    canary_sources: set[str] = set()
    tool_counts: Counter[str] = Counter()
    authoritative_tool_counts: Counter[str] = Counter()
    committed_invocations: Counter[str] = Counter()
    interval_events: dict[str, list[tuple[float, int]]] = defaultdict(list)
    per_tool_interval_events: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    queued_jobs = 0
    http_attempts = 0
    committed_http_attempts = 0
    retried_commits = 0
    failed_physical_jobs = 0
    auth_commits = 0
    physical_http_attempts = 0
    retried_physical_jobs = 0
    auth_commits_by_block: Counter[str] = Counter()
    retried_auth_commits_by_block: Counter[str] = Counter()

    for index, row in enumerate(rows):
        prefix = f"tool event {index}"
        missing = required_fields - set(row)
        if missing:
            raise ValueError(f"{prefix} is missing required fields: {sorted(missing)}")
        if _boolean(row.get("admitted"), f"{prefix}.admitted") is not True:
            raise ValueError(f"{prefix} is a rejected decision, not a physical job")
        job_id = _identifier(row.get("job_id"), f"{prefix}.job_id")
        if job_id in job_ids:
            raise ValueError(f"duplicate tool job_id: {job_id}")
        job_ids.add(job_id)
        task_id = _nonempty_string(row.get("task_instance_id"), f"{prefix}.task_instance_id")
        if task_id not in task_map:
            raise ValueError(f"{prefix} references an unknown task")
        if row.get("source_id") != task_map[task_id].get("source_id"):
            raise ValueError(f"{prefix} source_id does not match its task")
        source_id = str(row.get("source_id"))
        block_id = _nonempty_string(task_map[task_id].get("block_id"), "task block_id")
        invocation_digest = _nonempty_string(row.get("invocation_digest"), f"{prefix}.invocation_digest")
        invocation_id = _identifier(row.get("invocation_id"), f"{prefix}.invocation_id")
        if invocation_id in invocation_ids:
            raise ValueError(f"duplicate tool invocation_id: {invocation_id}")
        invocation_ids.add(invocation_id)
        session_id = _nonempty_string(row.get("session_id"), f"{prefix}.session_id")
        if session_id != task_id:
            raise ValueError(f"{prefix} is not isolated to its task session")
        tool_name = _nonempty_string(row.get("tool"), f"{prefix}.tool")
        if tool_name not in {"search", "visit"}:
            raise ValueError(f"{prefix}.tool is not search or visit")
        tool_counts[tool_name] += 1
        speculative = _boolean(row.get("speculative"), f"{prefix}.speculative")
        authoritative = _boolean(row.get("authoritative"), f"{prefix}.authoritative")
        committed = _boolean(row.get("committed"), f"{prefix}.committed")
        eligible = _boolean(row.get("speculation_eligible"), f"{prefix}.speculation_eligible")
        canary = _boolean(row.get("canary"), f"{prefix}.canary")
        cancelled = _boolean(row.get("cancelled"), f"{prefix}.cancelled")
        exact = _boolean(row.get("exact_match"), f"{prefix}.exact_match")
        cross_session = _boolean(row.get("cross_session_commit"), f"{prefix}.cross_session_commit")
        expected_pool = pool_ids_by_block.get(block_id)
        if row.get("worker_pool") != expected_pool:
            raise ValueError(f"{prefix} did not use the shared worker pool")
        _nonempty_string(row.get("source"), f"{prefix}.source")
        admitted = _finite(row.get("admitted_at"), f"{prefix}.admitted_at")
        queued = _finite(row.get("queue_enter_at"), f"{prefix}.queue_enter_at")
        started = _timestamp_or_none(row.get("started_at"), f"{prefix}.started_at")
        confirmed = _timestamp_or_none(
            row.get("authoritative_confirmation_at"),
            f"{prefix}.authoritative_confirmation_at",
        )
        finished = _timestamp_or_none(row.get("finished_at"), f"{prefix}.finished_at")
        queue_s = _finite(row.get("queue_s"), f"{prefix}.queue_s")
        service_s = _finite(row.get("service_s"), f"{prefix}.service_s")
        saved_s = _finite(row.get("saved_service_s"), f"{prefix}.saved_service_s")
        attempts = _integer(row.get("http_attempts"), f"{prefix}.http_attempts")
        http_attempts += attempts
        bytes_read = row.get("bytes_read")
        if bytes_read is not None:
            _integer(bytes_read, f"{prefix}.bytes_read")

        if admitted > queued:
            raise ValueError(f"{prefix} entered queue before admission")
        if started is None:
            if (
                not cancelled
                or finished is None
                or finished < queued
                or service_s != 0.0
                or saved_s != 0.0
                or attempts != 0
                or row.get("worker_id") is not None
                or row.get("outcome") not in {"cancelled", "expired"}
            ):
                raise ValueError(f"{prefix} is an invalid pre-start cancellation")
            if not _close(queue_s, finished - queued):
                raise ValueError(f"{prefix} pre-start queue duration is inconsistent")
        else:
            if finished is None or not queued <= started < finished:
                raise ValueError(f"{prefix} worker timestamps are not monotonic")
            _identifier(row.get("worker_id"), f"{prefix}.worker_id")
            if not _close(queue_s, started - queued) or not _close(service_s, finished - started):
                raise ValueError(f"{prefix} queue/service duration does not match timestamps")
            interval_events[block_id].append((started, 1))
            interval_events[block_id].append((finished, -1))
            per_tool_interval_events[(block_id, tool_name)].append((started, 1))
            per_tool_interval_events[(block_id, tool_name)].append((finished, -1))
            if attempts <= 0 or attempts > max_http_attempts:
                raise ValueError(
                    f"{prefix} HTTP attempts are outside the controlled range "
                    f"1..{max_http_attempts}"
                )
            if (
                attempts > 1
                and service_s + 0.01 < CONTROLLED_HTTP_RETRY_BACKOFF_S
            ):
                raise ValueError(f"{prefix} service time omits retry backoff")
            physical_http_attempts += attempts
            retried_physical_jobs += attempts > 1
            if queue_s > 0.0:
                queued_jobs += 1
        if canary and eligible:
            raise ValueError(f"{prefix} canary is speculation-eligible")
        if speculative and canary:
            raise ValueError(f"{prefix} speculated a canary")
        if saved_s > service_s + 0.01:
            raise ValueError(f"{prefix} saved service exceeds physical service")

        # External identity is required for every attempt that actually reached
        # a worker.  A queued prediction cancelled before start made no HTTP
        # attempt and therefore legitimately has no backend/status metadata.
        backend = row.get("backend")
        request_host = row.get("request_host")
        loopback_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        if started is None:
            if any(
                row.get(field) is not None
                for field in (
                    "backend",
                    "request_host",
                    "response_status",
                    "bytes_read",
                    "transport_identity_source",
                )
            ):
                raise ValueError(f"{prefix} pre-start cancellation claims HTTP evidence")
        else:
            backend = _nonempty_string(backend, f"{prefix}.backend")
            request_host = _nonempty_string(
                request_host, f"{prefix}.request_host"
            ).lower()
            if row.get("transport_identity_source") != "actual":
                raise ValueError(f"{prefix} lacks actual final HTTP evidence")
            if _integer(
                row.get("response_status"), f"{prefix}.response_status"
            ) != 200:
                raise ValueError(f"{prefix} final response was not HTTP 200")
            _integer(row.get("bytes_read"), f"{prefix}.bytes_read", positive=True)
            if not external_live:
                if backend != "controlled_http" or (
                    request_host not in loopback_hosts and not request_host.startswith("127.")
                ):
                    raise ValueError(f"{prefix} is not a controlled loopback HTTP job")
            elif tool_name == "search":
                if backend != "bing_html_search" or request_host not in {
                    "bing.com",
                    "www.bing.com",
                }:
                    raise ValueError(f"{prefix} is not a live Bing HTML request")
            elif backend == "r.jina.ai":
                if request_host != "r.jina.ai":
                    raise ValueError(f"{prefix} has an invalid Jina request host")
            else:
                raise ValueError(f"{prefix} has an unsupported live visit backend")

        logical_call_id = row.get("logical_call_id")
        if committed:
            if not authoritative:
                raise ValueError(f"{prefix} committed without authoritative confirmation")
            logical_call_id = _nonempty_string(logical_call_id, f"{prefix}.logical_call_id")
            if logical_call_id in logical_ids:
                raise ValueError(f"duplicate authoritative logical_call_id: {logical_call_id}")
            logical_ids.add(logical_call_id)
            per_task_authoritative[task_id] += 1
            auth_commits += 1
            auth_commits_by_block[block_id] += 1
            authoritative_tool_counts[tool_name] += 1
            committed_invocations[invocation_digest] += 1
            if row.get("outcome") not in {"success", "committed"} or cancelled or cross_session:
                raise ValueError(f"{prefix} is not a safe exact authoritative commit")
            if speculative and not exact:
                raise ValueError(f"{prefix} committed a non-exact speculative result")
            _nonempty_string(row.get("result_digest"), f"{prefix}.result_digest")
            committed_http_attempts += attempts
            retried_commits += attempts > 1
            retried_auth_commits_by_block[block_id] += attempts > 1
            if confirmed is None or finished is None or confirmed > finished:
                raise ValueError(f"{prefix} has no valid authoritative confirmation")
            if canary:
                latency = finished - confirmed
                if latency <= 0.0:
                    raise ValueError(f"{prefix} canary has non-positive latency")
                canary_latencies.append(latency)
                canary_invocations[invocation_digest] += 1
                canary_sources.add(source_id)
        else:
            if logical_call_id is not None or cross_session:
                raise ValueError(f"{prefix} uncommitted work has authoritative state")
            if not speculative:
                raise ValueError(f"{prefix} non-speculative physical job was not committed")
            if confirmed is not None and not authoritative:
                raise ValueError(f"{prefix} has a confirmation without authoritative state")
            outcome = row.get("outcome")
            if outcome not in {
                "completed",
                "failed",
                "cancelled",
                "expired",
                "failed_speculation_uncommitted",
            }:
                raise ValueError(f"{prefix} has an invalid speculative-waste outcome")
            failed_physical_jobs += outcome in {
                "failed",
                "failed_speculation_uncommitted",
            }
            if outcome in {"failed", "failed_speculation_uncommitted"}:
                raise ValueError(f"{prefix} is a failed physical job")
            if outcome in {"cancelled", "expired"} and not cancelled:
                raise ValueError(f"{prefix} cancellation outcome is not marked cancelled")
            if outcome == "completed":
                _nonempty_string(row.get("result_digest"), f"{prefix}.result_digest")

        if speculative:
            speculative_jobs += 1
            speculative_tool_counts[tool_name] += 1
            speculative_worker_s += service_s
            if committed:
                speculative_hits += 1
                if saved_s <= 0.0:
                    raise ValueError(f"{prefix} speculative hit saved no service time")
            else:
                speculative_wastes += 1
                wasted_worker_s += service_s
        elif not authoritative:
            raise ValueError(f"{prefix} invalid non-speculative job")

    if auth_commits != tasks["logical_tool_calls"]:
        raise ValueError("authoritative tool commits differ from task accounting")
    for task_id, task in task_map.items():
        expected = _integer(task.get("logical_tool_calls"), "task logical_tool_calls", positive=True)
        if per_task_authoritative[task_id] != expected:
            raise ValueError(f"tool calls are incomplete for task {task_id}")
    if len(rows) != tasks["logical_tool_calls"] + speculative_wastes:
        raise ValueError("physical jobs != logical authoritative calls + speculative wastes")
    if not authoritative_tool_counts["search"] or not authoritative_tool_counts["visit"]:
        raise ValueError("cell did not execute both live search and live visit")

    max_running = 0
    for block_id, events in interval_events.items():
        running = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            running += delta
            if running < 0:
                raise ValueError(
                    f"tool worker interval accounting became negative in {block_id}"
                )
            max_running = max(max_running, running)
        if running != 0:
            raise ValueError(f"tool worker interval accounting did not drain in {block_id}")
    if max_running > worker_capacity:
        raise ValueError("tool worker concurrency exceeded the frozen capacity")
    max_running_by_tool: Counter[str] = Counter()
    for (block_id, tool_name), events in per_tool_interval_events.items():
        running = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            running += delta
            if running < 0:
                raise ValueError(
                    f"{tool_name} interval accounting became negative in {block_id}"
                )
            max_running_by_tool[tool_name] = max(
                max_running_by_tool[tool_name], running
            )
        if running != 0:
            raise ValueError(f"{tool_name} interval accounting did not drain in {block_id}")
    for tool_name in ("search", "visit"):
        if max_running_by_tool[tool_name] > per_tool_capacity[tool_name]:
            raise ValueError(f"{tool_name} concurrency exceeded its frozen shared-pool cap")
    return {
        "physical_jobs": len(rows),
        "authoritative_commits": auth_commits,
        "speculative_jobs": speculative_jobs,
        "speculative_tool_counts": dict(speculative_tool_counts),
        "speculative_hits": speculative_hits,
        "speculative_wastes": speculative_wastes,
        "speculative_worker_s": speculative_worker_s,
        "wasted_speculative_worker_s": wasted_worker_s,
        "canary_calls": len(canary_latencies),
        "canary_mean_latency_s": mean(canary_latencies) if canary_latencies else None,
        "canary_p95_latency_s": _percentile(canary_latencies, 0.95) if canary_latencies else None,
        "canary_invocations": canary_invocations,
        "canary_sources": canary_sources,
        "committed_invocations": committed_invocations,
        "max_running_workers": max_running,
        "max_running_by_tool": dict(max_running_by_tool),
        "jobs_with_positive_queue": queued_jobs,
        "http_attempts": http_attempts,
        "physical_http_attempts": physical_http_attempts,
        "retried_physical_jobs": retried_physical_jobs,
        "committed_http_attempts": committed_http_attempts,
        "authoritative_retried_commits": retried_commits,
        "authoritative_retry_rate": retried_commits / auth_commits,
        "authoritative_retry_by_block": {
            block_id: {
                "retried_commits": retried_auth_commits_by_block[block_id],
                "commits": auth_commits_by_block[block_id],
                "rate": (
                    retried_auth_commits_by_block[block_id]
                    / auth_commits_by_block[block_id]
                ),
            }
            for block_id in sorted(auth_commits_by_block)
        },
        "failed_physical_jobs": failed_physical_jobs,
        "tool_counts": dict(tool_counts),
        "authoritative_tool_counts": dict(authoritative_tool_counts),
    }


def _validate_resource_samples(
    path: Path,
    *,
    block_ids: set[str],
    max_num_seqs: int,
    worker_capacity: int,
) -> dict[str, Any]:
    """Recompute native LLM/tool pressure from the sampled shared state."""

    rows = _load_jsonl(path, "resource samples")
    required = {
        "timestamp",
        "block_id",
        "llm_running",
        "llm_waiting",
        "tool_running_authoritative",
        "tool_running_speculative",
        "tool_queued_authoritative",
        "tool_queued_speculative",
    }
    seen_blocks: set[str] = set()
    timestamps: dict[str, list[float]] = defaultdict(list)
    max_running = 0
    waiting_below_cap = 0
    dual_pressure = 0
    tool_queue_samples = 0
    peak_tool_queue = 0
    for index, row in enumerate(rows):
        prefix = f"resource sample {index}"
        missing = required - set(row)
        if missing:
            raise ValueError(f"{prefix} is missing required fields: {sorted(missing)}")
        block_id = _nonempty_string(row.get("block_id"), f"{prefix}.block_id")
        if block_id not in block_ids:
            raise ValueError(f"{prefix} references an unknown block")
        seen_blocks.add(block_id)
        timestamp = _finite(row.get("timestamp"), f"{prefix}.timestamp")
        timestamps[block_id].append(timestamp)
        llm_running = _integer(row.get("llm_running"), f"{prefix}.llm_running")
        llm_waiting = _integer(row.get("llm_waiting"), f"{prefix}.llm_waiting")
        tool_running_authoritative = _integer(
            row.get("tool_running_authoritative"),
            f"{prefix}.tool_running_authoritative",
        )
        tool_running_speculative = _integer(
            row.get("tool_running_speculative"),
            f"{prefix}.tool_running_speculative",
        )
        tool_queued_authoritative = _integer(
            row.get("tool_queued_authoritative"),
            f"{prefix}.tool_queued_authoritative",
        )
        tool_queued_speculative = _integer(
            row.get("tool_queued_speculative"),
            f"{prefix}.tool_queued_speculative",
        )
        if llm_running >= max_num_seqs:
            raise ValueError(f"{prefix} reached the configured sequence ceiling")
        if tool_running_authoritative + tool_running_speculative > worker_capacity:
            raise ValueError(f"{prefix} exceeded tool worker capacity")
        max_running = max(max_running, llm_running)
        native_llm_waiting = llm_waiting > 0 and llm_running < max_num_seqs
        if native_llm_waiting:
            waiting_below_cap += 1
        tool_queue = tool_queued_authoritative + tool_queued_speculative
        if tool_queue > 0:
            tool_queue_samples += 1
        peak_tool_queue = max(peak_tool_queue, tool_queue)
        if native_llm_waiting and tool_queued_authoritative > 0:
            dual_pressure += 1

    if seen_blocks != block_ids:
        raise ValueError("resource samples do not cover every block")
    for block_id, values in timestamps.items():
        if len(values) != len(set(values)) or values != sorted(values):
            raise ValueError(f"resource sample timestamps are not strictly ordered in {block_id}")
    return {
        "timeline_samples": len(rows),
        "max_observed_running": max_running,
        "waiting_below_cap_samples": waiting_below_cap,
        "dual_pressure_samples": dual_pressure,
        "tool_queue_samples": tool_queue_samples,
        "peak_tool_queue": peak_tool_queue,
    }


def _validate_prefix_samples(path: Path, block_ids: set[str]) -> dict[str, Any]:
    rows = _load_jsonl(path, "prefix samples")
    values: list[float] = []
    seen_blocks: set[str] = set()
    for index, row in enumerate(rows):
        block_id = _nonempty_string(row.get("block_id"), f"prefix sample {index}.block_id")
        if block_id not in block_ids:
            raise ValueError("prefix sample references an unknown block")
        seen_blocks.add(block_id)
        _finite(row.get("timestamp"), f"prefix sample {index}.timestamp")
        value = _finite(row.get("gpu_prefix_hit_ratio"), f"prefix sample {index}.gpu_prefix_hit_ratio")
        if value > 1.0:
            raise ValueError("GPU prefix hit ratio exceeds one")
        values.append(value)
    if seen_blocks != block_ids:
        raise ValueError("prefix samples do not cover every block")
    return {"sample_count": len(values), "gpu_prefix_hit_ratio": mean(values)}


def _validate_runtime(value: Any) -> dict[str, Any]:
    runtime = _mapping(value, "runtime")
    required_true = (
        "live_llm_http",
        "live_search_http",
        "live_visit_http",
        "shared_process_wide_tool_pool",
        "exact_invocation_matching",
        "frozen_call_graph",
        "baseline_only_load_selection",
    )
    required_false = (
        "recorded_wait_replay",
        "synthetic_tool_sleep",
        "future_information_used",
        "cross_cell_tool_cache",
        "generated_text_changes_tool_plan",
    )
    for key in required_true:
        if _boolean(runtime.get(key), f"runtime.{key}") is not True:
            raise ValueError(f"runtime attestation failed: {key}")
    for key in required_false:
        if _boolean(runtime.get(key), f"runtime.{key}") is not False:
            raise ValueError(f"runtime attestation failed: {key}")
    backend_mode = _nonempty_string(runtime.get("backend_mode"), "runtime.backend_mode")
    if backend_mode not in {"external_live", "controlled_http"}:
        raise ValueError("runtime.backend_mode is unsupported")
    if backend_mode == "external_live":
        if runtime.get("search_backend") != "bing_html_search":
            raise ValueError("live search backend is not frozen Bing HTML search")
        if runtime.get("visit_backend") != "r_jina_ai":
            raise ValueError("live visit backend is not frozen r.jina.ai")
    elif runtime.get("search_backend") != "controlled_http" or runtime.get("visit_backend") != "controlled_http":
        raise ValueError("controlled backend labels are inconsistent")
    return {
        "backend_mode": backend_mode,
        "search_backend": str(runtime.get("search_backend")),
        "visit_backend": str(runtime.get("visit_backend")),
        "call_graph": "frozen",
        "full_external_live": backend_mode == "external_live",
    }


def _validate_blocks(value: Any, stage: str, cell_ids: set[str]) -> list[dict[str, Any]]:
    blocks_raw = _sequence(value, "blocks")
    expected_count = FORMAL_BLOCK_COUNT if stage == "formal" else 1
    if stage == "formal" and len(blocks_raw) != expected_count:
        raise ValueError("formal evidence requires exactly three blocks")
    if stage == "screening" and len(blocks_raw) != expected_count:
        raise ValueError("screening requires exactly one fresh block")
    blocks: list[dict[str, Any]] = []
    block_ids: set[str] = set()
    for index, raw in enumerate(blocks_raw):
        block = _mapping(raw, f"blocks[{index}]")
        block_id = _nonempty_string(block.get("block_id"), f"blocks[{index}].block_id")
        if block_id in block_ids:
            raise ValueError("duplicate block_id")
        block_ids.add(block_id)
        order = list(_sequence(block.get("cell_order"), f"blocks[{index}].cell_order"))
        if len(order) != len(cell_ids) or set(order) != cell_ids:
            raise ValueError("each block order must contain every cell exactly once")
        blocks.append({"block_id": block_id, "cell_order": order})
    if stage == "formal":
        for left, right in (("A", "B"), ("E", "F")):
            forward = sum(
                block["cell_order"].index(left) < block["cell_order"].index(right)
                for block in blocks
            )
            reverse = len(blocks) - forward
            if min(forward, reverse) == 0 or abs(forward - reverse) > 1:
                raise ValueError(f"formal run order is not balanced for {left}/{right}")
    return blocks


def _validate_cell(
    cell_id: str,
    value: Any,
    *,
    repository_root: Path,
    source_ids: set[str],
    blocks: Sequence[Mapping[str, Any]],
    copies_per_source: int,
    max_active_sessions: int,
    external_live: bool,
) -> dict[str, Any]:
    cell = _mapping(value, f"cell {cell_id}")
    policy = _mapping(cell.get("policy"), f"cell {cell_id}.policy")
    expected_llm, expected_tool = POLICIES[cell_id]
    if policy.get("llm_scheduler") != expected_llm or policy.get("tool_scheduler") != expected_tool:
        raise ValueError(f"cell {cell_id} policy labels are incorrect")
    prefix_policy = _nonempty_string(policy.get("prefix_policy"), f"cell {cell_id}.prefix_policy")
    if prefix_policy not in {"native", "explicit_affinity"}:
        raise ValueError(f"cell {cell_id} prefix policy is unsupported")
    speculation_scope = _nonempty_string(
        policy.get("speculation_scope"), f"cell {cell_id}.speculation_scope"
    )
    if expected_tool == "demand_only" and speculation_scope != "none":
        raise ValueError(f"cell {cell_id} demand-only policy has speculation enabled")
    if expected_tool == "resource_aware_speculation" and speculation_scope not in {
        "search_only",
        "visit_only",
        "search_visit",
    }:
        raise ValueError(f"cell {cell_id} speculation scope is unsupported")

    block_ids = {str(block["block_id"]) for block in blocks}
    fresh_ids = list(_sequence(cell.get("fresh_server_block_ids"), f"cell {cell_id}.fresh_server_block_ids"))
    if set(fresh_ids) != block_ids or len(fresh_ids) != len(block_ids):
        raise ValueError(f"cell {cell_id} does not have one fresh server per block")
    instances_by_block = _mapping(
        cell.get("server_instance_by_block"),
        f"cell {cell_id}.server_instance_by_block",
    )
    if set(instances_by_block) != block_ids:
        raise ValueError(f"cell {cell_id} server instances do not cover every block")
    instances = [
        _nonempty_string(instances_by_block[block], f"cell {cell_id}.server_instance_by_block.{block}")
        for block in sorted(block_ids)
    ]
    if len(set(instances)) != len(instances):
        raise ValueError(f"cell {cell_id} server instance IDs are not fresh")
    if _boolean(cell.get("result_cache_warm_start"), f"cell {cell_id}.result_cache_warm_start"):
        raise ValueError(f"cell {cell_id} reused a result cache")

    evidence = _validate_evidence_map(
        cell.get("evidence"),
        required=EVIDENCE_KINDS,
        repository_root=repository_root,
        label=f"cell {cell_id}.evidence",
    )
    tasks = _validate_task_events(
        evidence["task_events"],
        source_ids=source_ids,
        block_ids=block_ids,
        copies_per_source=copies_per_source,
    )
    llm = _validate_llm_events(evidence["llm_events"], tasks)

    engine = _mapping(cell.get("engine"), f"cell {cell_id}.engine")
    max_num_seqs = _integer(engine.get("max_num_seqs"), f"cell {cell_id}.engine.max_num_seqs", positive=True)
    offered = _integer(engine.get("max_active_sessions"), f"cell {cell_id}.engine.max_active_sessions", positive=True)
    if offered != max_active_sessions or max_num_seqs <= offered:
        raise ValueError(f"cell {cell_id} has a binding sequence-count ceiling")
    tool_runtime = _mapping(cell.get("tool_runtime"), f"cell {cell_id}.tool_runtime")
    pool_ids_by_block_raw = _mapping(
        tool_runtime.get("worker_pool_by_block"),
        f"cell {cell_id}.tool_runtime.worker_pool_by_block",
    )
    if set(pool_ids_by_block_raw) != block_ids:
        raise ValueError(f"cell {cell_id} worker pools do not cover every block")
    pool_ids_by_block = {
        block: _nonempty_string(
            pool_ids_by_block_raw[block],
            f"cell {cell_id}.tool_runtime.worker_pool_by_block.{block}",
        )
        for block in block_ids
    }
    if len(set(pool_ids_by_block.values())) != len(block_ids):
        raise ValueError(f"cell {cell_id} reused a broker across fresh blocks")
    capacity = _integer(tool_runtime.get("worker_capacity"), f"cell {cell_id}.tool_runtime.worker_capacity", positive=True)
    per_tool_raw = _mapping(
        tool_runtime.get("per_tool_capacity"),
        f"cell {cell_id}.tool_runtime.per_tool_capacity",
    )
    if set(per_tool_raw) != {"search", "visit"}:
        raise ValueError(f"cell {cell_id} must freeze search and visit capacity")
    per_tool_capacity = {
        tool_name: _integer(
            per_tool_raw[tool_name],
            f"cell {cell_id}.tool_runtime.per_tool_capacity.{tool_name}",
            positive=True,
        )
        for tool_name in ("search", "visit")
    }
    if any(value > capacity for value in per_tool_capacity.values()):
        raise ValueError(f"cell {cell_id} per-tool capacity exceeds the shared pool")
    max_speculative_workers = _integer(
        tool_runtime.get("max_speculative_workers"),
        f"cell {cell_id}.tool_runtime.max_speculative_workers",
    )
    if max_speculative_workers > capacity:
        raise ValueError(f"cell {cell_id} speculative capacity exceeds the shared pool")
    max_speculative_pending = _integer(
        tool_runtime.get("max_speculative_pending"),
        f"cell {cell_id}.tool_runtime.max_speculative_pending",
        positive=True,
    )
    speculative_ttl_s = _finite(
        tool_runtime.get("speculative_ttl_s"),
        f"cell {cell_id}.tool_runtime.speculative_ttl_s",
    )
    if speculative_ttl_s <= 0.0:
        raise ValueError(f"cell {cell_id} speculative TTL must be positive")
    max_http_attempts = _integer(
        tool_runtime.get("tool_http_max_attempts"),
        f"cell {cell_id}.tool_runtime.tool_http_max_attempts",
        positive=True,
    )
    if max_http_attempts != CONTROLLED_HTTP_MAX_ATTEMPTS:
        raise ValueError(
            f"cell {cell_id} must freeze tool_http_max_attempts="
            f"{CONTROLLED_HTTP_MAX_ATTEMPTS}"
        )
    retry_backoff_s = _finite(
        tool_runtime.get("tool_http_retry_backoff_s"),
        f"cell {cell_id}.tool_runtime.tool_http_retry_backoff_s",
    )
    if not math.isclose(
        retry_backoff_s, CONTROLLED_HTTP_RETRY_BACKOFF_S, abs_tol=1e-12
    ):
        raise ValueError(
            f"cell {cell_id} must freeze tool_http_retry_backoff_s="
            f"{CONTROLLED_HTTP_RETRY_BACKOFF_S}"
        )
    if _boolean(
        tool_runtime.get("controlled_http_retry"),
        f"cell {cell_id}.tool_runtime.controlled_http_retry",
    ) is not True:
        raise ValueError(f"cell {cell_id} did not enable controlled HTTP retry")
    if (
        tool_runtime.get("tool_http_retry_policy_version")
        != CONTROLLED_HTTP_RETRY_POLICY_VERSION
    ):
        raise ValueError(f"cell {cell_id} has an unsupported HTTP retry policy")
    if (
        tool_runtime.get("tool_http_retryable_statuses")
        != CONTROLLED_HTTP_RETRYABLE_STATUSES
    ):
        raise ValueError(f"cell {cell_id} has an unsupported retryable-status set")
    if (
        tool_runtime.get("tool_http_retryable_exception_types")
        != CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES
    ):
        raise ValueError(f"cell {cell_id} has an unsupported retryable-exception set")
    if _boolean(
        tool_runtime.get("tool_http_library_retry_disabled"),
        f"cell {cell_id}.tool_runtime.tool_http_library_retry_disabled",
    ) is not True:
        raise ValueError(
            f"cell {cell_id} did not disable hidden HTTP-library retry"
        )
    if (
        tool_runtime.get("tool_http_library_retry_control_version")
        != HTTP_LIBRARY_RETRY_CONTROL_VERSION
    ):
        raise ValueError(
            f"cell {cell_id} has an unsupported library-retry control"
        )
    if tool_runtime.get("tool_http_library_name") != FORMAL_HTTP_LIBRARY_NAME:
        raise ValueError(f"cell {cell_id} did not use the audited aiohttp transport")
    if (
        tool_runtime.get("tool_http_library_version")
        != FORMAL_HTTP_LIBRARY_VERSION
    ):
        raise ValueError(
            f"cell {cell_id} must freeze aiohttp {FORMAL_HTTP_LIBRARY_VERSION}"
        )
    resources = _validate_resource_samples(
        evidence["resource_samples"],
        block_ids=block_ids,
        max_num_seqs=max_num_seqs,
        worker_capacity=capacity,
    )
    tool = _validate_tool_events(
        evidence["tool_events"],
        tasks=tasks,
        pool_ids_by_block=pool_ids_by_block,
        worker_capacity=capacity,
        per_tool_capacity=per_tool_capacity,
        external_live=external_live,
        max_http_attempts=max_http_attempts,
    )
    if tool["max_running_workers"] > capacity:
        raise ValueError(f"cell {cell_id} exceeded tool worker capacity")
    if expected_tool == "demand_only" and tool["speculative_jobs"] != 0:
        raise ValueError(f"cell {cell_id} demand-only policy executed speculation")
    if expected_tool == "resource_aware_speculation" and tool["speculative_jobs"] == 0:
        raise ValueError(f"cell {cell_id} spec-on policy executed no speculation")
    observed_speculative_tools = set(tool["speculative_tool_counts"])
    expected_speculative_tools = {
        "none": set(),
        "search_only": {"search"},
        "visit_only": {"visit"},
        "search_visit": {"search", "visit"},
    }[speculation_scope]
    if observed_speculative_tools != expected_speculative_tools:
        raise ValueError(f"cell {cell_id} physical speculation differs from its scope")

    prefix = _validate_prefix_samples(evidence["prefix_samples"], block_ids)
    if tool["canary_sources"] != source_ids:
        raise ValueError(f"cell {cell_id} lacks authoritative canary coverage")
    return {
        "cell_id": cell_id,
        "policy": dict(policy),
        "server_instance_ids": instances,
        "engine": {
            "max_num_seqs": max_num_seqs,
            "max_active_sessions": offered,
            "max_observed_running": resources["max_observed_running"],
            "llm_timeline_samples": resources["timeline_samples"],
            "waiting_below_cap_samples": resources["waiting_below_cap_samples"],
            "native_queue_sample_fraction": (
                resources["waiting_below_cap_samples"] / resources["timeline_samples"]
            ),
            "dual_pressure_samples": resources["dual_pressure_samples"],
        },
        "tool_runtime": {
            "worker_pool_ids": list(pool_ids_by_block.values()),
            "worker_capacity": capacity,
            "per_tool_capacity": per_tool_capacity,
            "max_speculative_workers": max_speculative_workers,
            "max_speculative_pending": max_speculative_pending,
            "speculative_ttl_s": speculative_ttl_s,
            "tool_http_max_attempts": max_http_attempts,
            "tool_http_retry_backoff_s": retry_backoff_s,
            "controlled_http_retry": True,
            "tool_http_retry_policy_version": (
                CONTROLLED_HTTP_RETRY_POLICY_VERSION
            ),
            "tool_http_retryable_statuses": list(
                CONTROLLED_HTTP_RETRYABLE_STATUSES
            ),
            "tool_http_retryable_exception_types": list(
                CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES
            ),
            "tool_http_library_retry_disabled": True,
            "tool_http_library_retry_control_version": (
                HTTP_LIBRARY_RETRY_CONTROL_VERSION
            ),
            "tool_http_library_name": FORMAL_HTTP_LIBRARY_NAME,
            "tool_http_library_version": FORMAL_HTTP_LIBRARY_VERSION,
            "timeline_samples": resources["timeline_samples"],
            "queue_samples": resources["tool_queue_samples"],
            "queue_sample_fraction": (
                resources["tool_queue_samples"] / resources["timeline_samples"]
            ),
            "peak_queue": resources["peak_tool_queue"],
        },
        "source_task_e2e_s": tasks["source_task_e2e_s"],
        "task_count": tasks["task_count"],
        "mean_makespan_s": tasks["mean_makespan_s"],
        "llm": llm,
        "tool": tool,
        "prefix": prefix,
    }


def _validate_pair(base: Mapping[str, Any], candidate: Mapping[str, Any], label: str) -> dict[str, Any]:
    if base["task_count"] != candidate["task_count"]:
        raise ValueError(f"{label} task counts differ")
    if base["llm"]["request_count"] != candidate["llm"]["request_count"]:
        raise ValueError(f"{label} logical LLM counts differ")
    if base["tool"]["authoritative_commits"] != candidate["tool"]["authoritative_commits"]:
        raise ValueError(f"{label} logical tool counts differ")
    if base["tool"]["committed_invocations"] != candidate["tool"]["committed_invocations"]:
        raise ValueError(f"{label} authoritative invocation sets differ")
    if base["tool"]["canary_invocations"] != candidate["tool"]["canary_invocations"]:
        raise ValueError(f"{label} canary invocation sets differ")
    if base["tool_runtime"]["worker_capacity"] != candidate["tool_runtime"]["worker_capacity"]:
        raise ValueError(f"{label} tool capacities differ")
    if base["tool_runtime"]["per_tool_capacity"] != candidate["tool_runtime"]["per_tool_capacity"]:
        raise ValueError(f"{label} per-tool shared-pool capacities differ")
    token_difference = _relative_difference(
        base["llm"]["completion_tokens"], candidate["llm"]["completion_tokens"]
    )
    if token_difference >= MAX_TOKEN_RELATIVE_DIFFERENCE:
        raise ValueError(f"{label} completion-token difference is at least 1%")
    retry_rate_difference = abs(
        base["tool"]["authoritative_retry_rate"]
        - candidate["tool"]["authoritative_retry_rate"]
    )
    return {
        "completion_token_relative_difference": token_difference,
        "authoritative_retry_rate_difference": retry_rate_difference,
    }


def _validate_prefix_ablation(
    value: Any,
    *,
    repository_root: Path,
    formal_source_ids: set[str],
) -> dict[str, Any]:
    ablation = _mapping(value, "prefix_ablation")
    source_ids = set(_sequence(ablation.get("source_ids"), "prefix_ablation.source_ids"))
    if not SCREENING_MIN_SOURCES <= len(source_ids) <= SCREENING_MAX_SOURCES:
        raise ValueError("prefix ablation must use 12--20 independent tune sources")
    if source_ids & formal_source_ids:
        raise ValueError("prefix tune sources overlap formal sources")
    block_list = list(_sequence(ablation.get("block_ids"), "prefix_ablation.block_ids"))
    if (
        not block_list
        or any(not isinstance(item, str) or not item for item in block_list)
        or len(set(block_list)) != len(block_list)
    ):
        raise ValueError("prefix ablation block_ids must be unique non-empty strings")
    block_ids = set(block_list)
    copies = _integer(
        ablation.get("copies_per_source"),
        "prefix_ablation.copies_per_source",
        positive=True,
    )
    cells = _mapping(ablation.get("cells"), "prefix_ablation.cells")
    if set(cells) != {"P0", "P1", "P2"}:
        raise ValueError("prefix ablation requires exactly P0/P1/P2")
    expected_policies = {"P0": "disabled", "P1": "native", "P2": "explicit_affinity"}
    parsed: dict[str, Any] = {}
    all_server_instances: list[str] = []
    for cell_id, expected_policy in expected_policies.items():
        cell = _mapping(cells[cell_id], f"prefix_ablation.{cell_id}")
        if cell.get("prefix_policy") != expected_policy:
            raise ValueError(f"prefix ablation {cell_id} policy is incorrect")
        fresh = set(
            _sequence(
                cell.get("fresh_server_block_ids"),
                f"prefix_ablation.{cell_id}.fresh_server_block_ids",
            )
        )
        if fresh != block_ids or len(fresh) != len(block_ids):
            raise ValueError(f"prefix ablation {cell_id} lacks a fresh server per block")
        instances_by_block = _mapping(
            cell.get("server_instance_by_block"),
            f"prefix_ablation.{cell_id}.server_instance_by_block",
        )
        if set(instances_by_block) != block_ids:
            raise ValueError(f"prefix ablation {cell_id} server IDs do not cover blocks")
        instances = [
            _nonempty_string(
                instances_by_block[block],
                f"prefix_ablation.{cell_id}.server_instance_by_block.{block}",
            )
            for block in sorted(block_ids)
        ]
        if len(set(instances)) != len(instances):
            raise ValueError(f"prefix ablation {cell_id} reused a server")
        all_server_instances.extend(instances)
        evidence = _validate_evidence_map(
            cell.get("evidence"),
            required=PREFIX_EVIDENCE_KINDS,
            repository_root=repository_root,
            label=f"prefix_ablation.{cell_id}.evidence",
        )
        tasks = _validate_task_events(
            evidence["task_events"],
            source_ids=source_ids,
            block_ids=block_ids,
            copies_per_source=copies,
        )
        prefix = _validate_prefix_samples(evidence["prefix_samples"], block_ids)
        hit = prefix["gpu_prefix_hit_ratio"]
        if hit > 1.0 or (cell_id == "P0" and hit != 0.0):
            raise ValueError(f"prefix ablation {cell_id} hit ratio is inconsistent")
        parsed[cell_id] = {
            "source_task_e2e_s": tasks["source_task_e2e_s"],
            "gpu_prefix_hit_ratio": hit,
            "task_count": tasks["task_count"],
        }
    if len(set(all_server_instances)) != len(all_server_instances):
        raise ValueError("prefix ablation reused a server across cells")
    if len({parsed[cell]["task_count"] for cell in parsed}) != 1:
        raise ValueError("prefix ablation task counts differ")
    native_effect = _effect(parsed["P0"], parsed["P1"])
    explicit_effect = _effect(parsed["P1"], parsed["P2"])
    explicit_gates = {
        "mean_task_e2e_reduction_at_least_2pct": explicit_effect["relative_reduction"] >= PREFIX_MIN_E2E_REDUCTION,
        "gpu_prefix_hit_ratio_increase_at_least_3pp": (
            parsed["P2"]["gpu_prefix_hit_ratio"] - parsed["P1"]["gpu_prefix_hit_ratio"]
            >= PREFIX_MIN_HIT_RATIO_INCREASE
        ),
        "source_bootstrap_lower_above_zero": explicit_effect["source_bootstrap_95_ci_s"]["lower_s"] > 0.0,
        "task_p95_within_3pct": explicit_effect["candidate_p95_s"] <= PREFIX_MAX_TASK_P95_RATIO * explicit_effect["base_p95_s"],
    }
    explicit_passed = all(explicit_gates.values())
    selected = _nonempty_string(ablation.get("selected_policy"), "prefix_ablation.selected_policy")
    if selected not in {"native", "explicit_affinity"}:
        raise ValueError("prefix selected_policy is unsupported")
    selection_valid = (selected == "explicit_affinity" and explicit_passed) or (
        selected == "native" and not explicit_passed
    )
    return {
        "selected_policy": selected,
        "selection_valid": selection_valid,
        "native_cache_effect_P0_to_P1": native_effect,
        "explicit_affinity_effect_P1_to_P2": explicit_effect,
        "explicit_affinity_gates": {**explicit_gates, "passed": explicit_passed},
    }


def validate_live_joint_result(
    payload: Mapping[str, Any],
    *,
    stage: str,
    repository_root: Path = REPOSITORY_ROOT,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    if stage not in {"screening", "formal"}:
        raise ValueError("stage must be screening or formal")
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise ValueError("unsupported live-joint result schema/version")
    if payload.get("stage") != stage:
        raise ValueError("result stage does not match the requested validation stage")
    protocol_sha = _nonempty_string(payload.get("protocol_sha256"), "protocol_sha256")
    if not protocol_path.is_file() or protocol_sha != _sha256_file(protocol_path):
        raise ValueError("result is not bound to the current prospective protocol")
    runtime = _validate_runtime(payload.get("runtime"))

    workload = _mapping(payload.get("workload"), "workload")
    source_list = list(_sequence(workload.get("source_ids"), "workload.source_ids"))
    if any(not isinstance(item, str) or not item for item in source_list) or len(set(source_list)) != len(source_list):
        raise ValueError("workload source_ids must be unique non-empty strings")
    source_ids = set(source_list)
    if stage == "formal":
        if len(source_ids) != FORMAL_SOURCE_COUNT or workload.get("split_role") != "heldout":
            raise ValueError("formal workload must contain exactly 60 heldout sources")
        if _integer(workload.get("tuning_source_overlap_count"), "workload.tuning_source_overlap_count") != 0:
            raise ValueError("formal workload overlaps tuning sources")
    else:
        if not SCREENING_MIN_SOURCES <= len(source_ids) <= SCREENING_MAX_SOURCES or workload.get("split_role") != "tune":
            raise ValueError("screening workload must contain 12--20 tune sources")
    manifest_path = _resolve_evidence(
        workload.get("manifest"),
        repository_root=repository_root,
        label="workload.manifest",
    )
    _validate_workload_manifest(
        manifest_path,
        source_ids=source_ids,
        split_role=str(workload.get("split_role")),
    )
    copies = _integer(workload.get("copies_per_source"), "workload.copies_per_source", positive=True)
    max_active = _integer(workload.get("max_active_sessions"), "workload.max_active_sessions", positive=True)
    if max_active != len(source_ids) * copies:
        raise ValueError("max_active_sessions does not equal sources times copies")

    cells_raw = _mapping(payload.get("cells"), "cells")
    cell_ids = set(cells_raw)
    if not set(REQUIRED_CELLS).issubset(cell_ids):
        raise ValueError("A/B/E/F cells are required")
    if not cell_ids.issubset(set(REQUIRED_CELLS) | set(OPTIONAL_DIAGNOSTIC_CELLS)):
        raise ValueError("result contains an unsupported matrix cell")
    if ("C" in cell_ids) != ("D" in cell_ids):
        raise ValueError("diagnostic C/D cells must appear together")
    blocks = _validate_blocks(payload.get("blocks"), stage, cell_ids)

    cells = {
        cell_id: _validate_cell(
            cell_id,
            cells_raw[cell_id],
            repository_root=repository_root,
            source_ids=source_ids,
            blocks=blocks,
            copies_per_source=copies,
            max_active_sessions=max_active,
            external_live=runtime["full_external_live"],
        )
        for cell_id in sorted(cell_ids)
    }
    all_server_instances = [item for cell in cells.values() for item in cell["server_instance_ids"]]
    if len(set(all_server_instances)) != len(all_server_instances):
        raise ValueError("a vLLM server instance was reused across cells")
    all_pool_ids = [
        pool_id
        for cell in cells.values()
        for pool_id in cell["tool_runtime"]["worker_pool_ids"]
    ]
    if len(set(all_pool_ids)) != len(all_pool_ids):
        raise ValueError("a broker/result cache instance was reused across cells")
    capacity_signatures = {
        (
            cell["tool_runtime"]["worker_capacity"],
            cell["tool_runtime"]["per_tool_capacity"]["search"],
            cell["tool_runtime"]["per_tool_capacity"]["visit"],
            cell["tool_runtime"]["max_speculative_workers"],
            cell["tool_runtime"]["max_speculative_pending"],
            cell["tool_runtime"]["speculative_ttl_s"],
            cell["tool_runtime"]["tool_http_max_attempts"],
            cell["tool_runtime"]["tool_http_retry_backoff_s"],
            cell["tool_runtime"]["controlled_http_retry"],
            cell["tool_runtime"]["tool_http_retry_policy_version"],
            tuple(cell["tool_runtime"]["tool_http_retryable_statuses"]),
            tuple(
                cell["tool_runtime"]["tool_http_retryable_exception_types"]
            ),
            cell["tool_runtime"]["tool_http_library_retry_disabled"],
            cell["tool_runtime"][
                "tool_http_library_retry_control_version"
            ],
            cell["tool_runtime"]["tool_http_library_name"],
            cell["tool_runtime"]["tool_http_library_version"],
        )
        for cell in cells.values()
    }
    if len(capacity_signatures) != 1:
        raise ValueError("tool capacity was not frozen identically across cells")

    pair_ab = _validate_pair(cells["A"], cells["B"], "A/B")
    pair_ef = _validate_pair(cells["E"], cells["F"], "E/F")
    pair_af = _validate_pair(cells["A"], cells["F"], "A/F")
    if cells["A"]["policy"]["prefix_policy"] != cells["B"]["policy"]["prefix_policy"]:
        raise ValueError("A/B prefix policies differ")
    if cells["E"]["policy"]["prefix_policy"] != cells["F"]["policy"]["prefix_policy"]:
        raise ValueError("E/F prefix policies differ")
    if cells["B"]["policy"]["speculation_scope"] != cells["F"]["policy"]["speculation_scope"]:
        raise ValueError("B/F speculation scopes differ")

    baseline_queue_gates = {
        "native_llm_queue_fraction_at_least_5pct": (
            cells["A"]["engine"]["native_queue_sample_fraction"] >= MIN_NATIVE_QUEUE_SAMPLE_FRACTION
        ),
        "tool_queue_fraction_at_least_5pct": (
            cells["A"]["tool_runtime"]["queue_sample_fraction"] >= MIN_TOOL_QUEUE_SAMPLE_FRACTION
        ),
        "positive_tool_queue": cells["A"]["tool_runtime"]["peak_queue"] > 0,
        "simultaneous_llm_tool_pressure_observed": cells["A"]["engine"]["dual_pressure_samples"] > 0,
    }
    if not all(baseline_queue_gates.values()):
        raise ValueError("baseline-only native dual-queue load-selection proof failed")

    effect_ef = _effect(cells["E"], cells["F"])
    effect_af = _effect(cells["A"], cells["F"])
    effect_ab = _effect(cells["A"], cells["B"])
    interaction_by_source = [
        ef - ab
        for ef, ab in zip(effect_ef["source_differences_s"], effect_ab["source_differences_s"])
    ]
    interaction = {
        "mean_s": mean(interaction_by_source),
        "source_bootstrap_95_ci_s": _bootstrap_mean_ci(interaction_by_source),
        "positive_point_estimate": mean(interaction_by_source) > 0.0,
    }

    f_tool = cells["F"]["tool"]
    e_tool = cells["E"]["tool"]
    spec_hit_rate = f_tool["speculative_hits"] / f_tool["speculative_jobs"]
    waste_fraction = (
        f_tool["wasted_speculative_worker_s"] / f_tool["speculative_worker_s"]
        if f_tool["speculative_worker_s"] > 0.0
        else math.inf
    )
    if e_tool["canary_mean_latency_s"] is None or f_tool["canary_mean_latency_s"] is None:
        raise ValueError("E/F canary metrics are missing")
    canary_mean_ratio = f_tool["canary_mean_latency_s"] / e_tool["canary_mean_latency_s"]
    canary_p95_ratio = f_tool["canary_p95_latency_s"] / e_tool["canary_p95_latency_s"]
    p95_ratio = effect_ef["candidate_p95_s"] / effect_ef["base_p95_s"]
    makespan_ratio = cells["F"]["mean_makespan_s"] / cells["E"]["mean_makespan_s"]

    if stage == "formal":
        min_effect = FORMAL_MIN_SPEC_E2E_REDUCTION
        min_hit = FORMAL_MIN_SPEC_HIT_RATE
        max_waste = FORMAL_MAX_WASTE_WORKER_FRACTION
        max_canary_mean = FORMAL_MAX_CANARY_MEAN_RATIO
        max_canary_p95 = FORMAL_MAX_CANARY_P95_RATIO
        max_task_p95 = FORMAL_MAX_TASK_P95_RATIO
        max_retry_rate = FORMAL_MAX_AUTHORITATIVE_RETRY_RATE
        max_retry_difference = FORMAL_MAX_RETRY_RATE_DIFFERENCE
        source_gate = effect_ef["source_faster_count"] >= FORMAL_MIN_SOURCE_FASTER_COUNT
        source_gate_label = "at_least_42_of_60_sources_faster"
        bootstrap_gate = effect_ef["source_bootstrap_95_ci_s"]["lower_s"] > 0.0
    else:
        min_effect = SCREENING_MIN_SPEC_E2E_REDUCTION
        min_hit = SCREENING_MIN_SPEC_HIT_RATE
        max_waste = SCREENING_MAX_WASTE_WORKER_FRACTION
        max_canary_mean = SCREENING_MAX_CANARY_MEAN_RATIO
        max_canary_p95 = SCREENING_MAX_CANARY_P95_RATIO
        max_task_p95 = SCREENING_MAX_TASK_P95_RATIO
        max_retry_rate = SCREENING_MAX_AUTHORITATIVE_RETRY_RATE
        max_retry_difference = SCREENING_MAX_RETRY_RATE_DIFFERENCE
        source_gate = (
            effect_ef["source_faster_count"] / effect_ef["source_count"]
            >= SCREENING_MIN_SOURCE_FASTER_FRACTION
        )
        source_gate_label = "at_least_60pct_sources_faster"
        bootstrap_gate = True

    live_speculation_gates = {
        "external_live_backend": runtime["full_external_live"],
        "mean_task_e2e_reduction": effect_ef["relative_reduction"] >= min_effect,
        source_gate_label: source_gate,
        "source_bootstrap_lower_above_zero": bootstrap_gate,
        "speculative_hit_rate": spec_hit_rate >= min_hit,
        "wasted_speculative_worker_fraction": waste_fraction <= max_waste,
        "canary_authoritative_mean_slowdown": canary_mean_ratio <= max_canary_mean,
        "canary_authoritative_p95_slowdown": canary_p95_ratio <= max_canary_p95,
        "authoritative_retry_rate": (
            e_tool["authoritative_retry_rate"] <= max_retry_rate
            and f_tool["authoritative_retry_rate"] <= max_retry_rate
        ),
        "all_cells_authoritative_retry_rate": all(
            cell["tool"]["authoritative_retry_rate"] <= max_retry_rate
            for cell in cells.values()
        ),
        "all_block_cells_authoritative_retry_rate": all(
            block_retry["rate"] <= max_retry_rate
            for cell in cells.values()
            for block_retry in cell["tool"][
                "authoritative_retry_by_block"
            ].values()
        ),
        "authoritative_retry_rate_balance": (
            pair_ef["authoritative_retry_rate_difference"] <= max_retry_difference
        ),
        "zero_failed_physical_jobs": all(
            cell["tool"]["failed_physical_jobs"] == 0
            for cell in cells.values()
        ),
        "task_p95_safety": p95_ratio <= max_task_p95,
        "makespan_safety": makespan_ratio <= MAX_MAKESPAN_RATIO,
    }
    live_speculation_passed = all(live_speculation_gates.values())

    overall_gates = {
        "mean_task_e2e_reduction_at_least_25pct": (
            effect_af["relative_reduction"] >= FORMAL_MIN_OVERALL_E2E_REDUCTION
            if stage == "formal" else True
        ),
        "at_least_48_of_60_sources_faster": (
            effect_af["source_faster_count"] >= FORMAL_MIN_OVERALL_FASTER_COUNT
            if stage == "formal" else True
        ),
        "request_p99_no_more_than_1_25x_A": (
            cells["F"]["llm"]["p99_latency_s"]
            <= FORMAL_MAX_REQUEST_P99_RATIO * cells["A"]["llm"]["p99_latency_s"]
            if stage == "formal" else True
        ),
        "authoritative_retry_rate_A_F": (
            cells["A"]["tool"]["authoritative_retry_rate"]
            <= FORMAL_MAX_AUTHORITATIVE_RETRY_RATE
            and cells["F"]["tool"]["authoritative_retry_rate"]
            <= FORMAL_MAX_AUTHORITATIVE_RETRY_RATE
            if stage == "formal" else True
        ),
        "authoritative_retry_rate_balance_A_F": (
            pair_af["authoritative_retry_rate_difference"]
            <= FORMAL_MAX_RETRY_RATE_DIFFERENCE
            if stage == "formal" else True
        ),
    }
    overall_passed = all(overall_gates.values())

    prefix_value = payload.get("prefix_ablation")
    if stage == "formal" and prefix_value is None:
        raise ValueError("formal result is missing the three-cell prefix ablation")
    prefix = (
        _validate_prefix_ablation(
            prefix_value, repository_root=repository_root, formal_source_ids=source_ids
        )
        if prefix_value is not None
        else None
    )
    if prefix is not None:
        selected = prefix["selected_policy"]
        if cells["E"]["policy"]["prefix_policy"] != selected:
            raise ValueError("E/F do not use the prefix policy selected by the ablation")
    prefix_passed = prefix is None or prefix["selection_valid"]

    promotion_passed = live_speculation_passed and overall_passed and prefix_passed
    return {
        "schema": VALIDATION_SCHEMA,
        "version": VALIDATION_VERSION,
        "stage": stage,
        "protocol_sha256": protocol_sha,
        "runtime": runtime,
        "matrix_cells": sorted(cell_ids),
        "independent_source_count": len(source_ids),
        "block_count": len(blocks),
        "comparability": {"A_vs_B": pair_ab, "E_vs_F": pair_ef, "A_vs_F": pair_af},
        "baseline_dual_queue_gates": {**baseline_queue_gates, "passed": True},
        "effects": {
            "live_speculation_E_vs_F": effect_ef,
            "overall_A_vs_F": effect_af,
            "fcfs_speculation_A_vs_B": effect_ab,
            "interaction": interaction,
        },
        "tool_resource_metrics": {
            "speculative_hit_rate": spec_hit_rate,
            "wasted_speculative_worker_fraction": waste_fraction,
            "canary_mean_latency_ratio_F_over_E": canary_mean_ratio,
            "canary_p95_latency_ratio_F_over_E": canary_p95_ratio,
        },
        "tool_retry_accounting": {
            cell_id: {
                "authoritative_commit_count": cell["tool"][
                    "authoritative_commits"
                ],
                "authoritative_retried_commit_count": cell["tool"][
                    "authoritative_retried_commits"
                ],
                "authoritative_retry_rate": cell["tool"][
                    "authoritative_retry_rate"
                ],
                "authoritative_retry_by_block": cell["tool"][
                    "authoritative_retry_by_block"
                ],
                "physical_job_count": cell["tool"]["physical_jobs"],
                "physical_http_attempt_count": cell["tool"][
                    "physical_http_attempts"
                ],
                "retried_physical_job_count": cell["tool"][
                    "retried_physical_jobs"
                ],
                "failed_physical_job_count": cell["tool"][
                    "failed_physical_jobs"
                ],
            }
            for cell_id, cell in cells.items()
        },
        "tool_retry_policy": {
            key: cells["A"]["tool_runtime"][key]
            for key in (
                "tool_http_max_attempts",
                "tool_http_retry_backoff_s",
                "controlled_http_retry",
                "tool_http_retry_policy_version",
                "tool_http_retryable_statuses",
                "tool_http_retryable_exception_types",
                "tool_http_library_retry_disabled",
                "tool_http_library_retry_control_version",
                "tool_http_library_name",
                "tool_http_library_version",
            )
        },
        "live_speculation_gates": {**live_speculation_gates, "passed": live_speculation_passed},
        "overall_system_gates": {**overall_gates, "passed": overall_passed},
        "prefix_ablation": prefix,
        "thirty_percent_claim_permitted": effect_af["relative_reduction"] >= 0.30,
        "synergy_claim_permitted": (
            interaction["source_bootstrap_95_ci_s"]["lower_s"] > 0.0
        ),
        "promotion_passed": promotion_passed,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--stage", required=True, choices=("screening", "formal"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-promotion", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("result root must be an object")
    validation = validate_live_joint_result(payload, stage=args.stage)
    encoded = json.dumps(validation, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.require_promotion and not validation["promotion_passed"]:
        raise SystemExit("live-joint promotion gates failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
