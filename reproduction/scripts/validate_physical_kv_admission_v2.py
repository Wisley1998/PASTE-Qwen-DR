#!/usr/bin/env python3
"""Validate a fresh parser-v2 adaptive physical-KV screening cell.

This validator is intentionally separate from ``validate_physical_kv_admission.py``.
The older module's SHA256 is bound by an immutable stress240 revalidation sidecar.
Keeping this implementation in a new file lets later stress300 evidence add strict
raw-log and transport checks without invalidating that earlier artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUNNER_DIRECTORY = REPOSITORY_ROOT / "scripts"
for import_path in (SCRIPT_DIRECTORY, RUNNER_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from run_vllm_trace_experiment import (  # noqa: E402
    PHYSICAL_KV_LOG_PARSER_ID,
    PHYSICAL_KV_LOG_PARSER_VERSION,
)
from validate_physical_kv_admission import (  # noqa: E402
    _capacity_write_counts,
    _load_object,
    _select_experiment_physical_log,
    _strict_sample_list,
)


SCHEMA = "paste_repro.physical_kv_admission_validation_v2"
VERSION = 1
STATUS = "accepted_fresh_parser_v2_physical_kv_telemetry"
PHYSICAL_MARKER = "[sched_policy_patch:physical_kv]"
PARSER_MODULE = RUNNER_DIRECTORY / "run_vllm_trace_experiment.py"
DEPENDENCY_MODULE = SCRIPT_DIRECTORY / "validate_physical_kv_admission.py"
VALIDATOR_MODULE = Path(__file__).resolve()
ENGINE_SHAPE_KEYS = (
    "MODEL_ID",
    "MODEL_REVISION",
    "CUDA_VISIBLE_DEVICES",
    "VLLM_HOST",
    "VLLM_PROBE_HOST",
    "VLLM_PORT",
    "VLLM_TP_SIZE",
    "VLLM_DTYPE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_CUDA_GRAPH_SIZES",
    "VLLM_USE_V1",
)
PHYSICAL_INTEGER_FIELDS = (
    "num_gpu_blocks",
    "block_size",
    "capacity_tokens",
    "budget_tokens",
    "live_tokens",
    "logical_live_tokens",
    "running_growth_tokens",
    "reserved_tokens",
    "committed_tokens",
    "predicted_admit_tokens",
    "waiting",
    "running",
    "fit_admit",
    "admit",
    "effective_cap",
    "native_cap",
    "capacity_write_count",
    "rescue",
)
PHYSICAL_RAW_FIELD_ORDER = (
    "decision",
    "reason",
    "num_gpu_blocks",
    "block_size",
    "capacity_tokens",
    "target_utilization",
    "budget_tokens",
    "usage",
    "live_tokens",
    "logical_live_tokens",
    "running_growth_tokens",
    "reserved_tokens",
    "committed_tokens",
    "predicted_admit_tokens",
    "waiting",
    "running",
    "fit_admit",
    "admit",
    "effective_cap",
    "native_cap",
    "capacity_write_source",
    "capacity_write_count",
    "rescue",
)
BLOCK_ALIGNED_TOKEN_FIELDS = (
    "capacity_tokens",
    "budget_tokens",
    "live_tokens",
    "logical_live_tokens",
    "running_growth_tokens",
    "reserved_tokens",
    "committed_tokens",
    "predicted_admit_tokens",
)
REQUIRED_SCREENING_GATES = (
    "has_samples",
    "no_malformed_samples",
    "no_fail_closed_decisions",
    "stable_physical_capacity",
    "at_least_three_effective_caps",
    "observed_cap_increase",
    "observed_cap_decrease",
    "observed_zero_fit_admit",
    "observed_positive_fit_admit",
    "at_least_ten_pressure_samples_above_64",
    "passed",
)
EXPORT_PATTERN = re.compile(r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$')
GPU_KV_CACHE_TOKENS_PATTERN = re.compile(
    r"GPU KV cache size:\s*([0-9][0-9,]*) tokens"
)
SCHEDULER_GPU_BLOCKS_PATTERN = re.compile(
    r"cache_config_info with initialization after num_gpu_blocks is:\s*([0-9]+)"
)
RAW_FIELD_PATTERN = re.compile(r"([a-z][a-z0-9_]*)=([^\s]+)")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside the repository: {resolved}") from exc


def _strict_nonnegative_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number != value or number < 0 or (positive and number == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return number


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _parse_key_values(items: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"--expect-engine-shape requires NAME=VALUE: {item!r}")
        if name in result:
            raise ValueError(f"duplicate engine-shape expectation: {name}")
        result[name] = value
    missing = sorted(set(ENGINE_SHAPE_KEYS) - set(result))
    extra = sorted(set(result) - set(ENGINE_SHAPE_KEYS))
    if missing or extra:
        raise ValueError(
            "engine-shape expectations must be exact; "
            f"missing={missing}, extra={extra}"
        )
    return result


def _parse_frozen_exports(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = EXPORT_PATTERN.fullmatch(line)
        if match is None:
            continue
        name, value = match.groups()
        if name in values:
            raise ValueError(f"frozen config repeats export {name} at line {line_number}")
        values[name] = value
    return values


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.resolve()
    if not output.parent.is_dir():
        raise ValueError(f"output parent is missing: {output.parent}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_raw_physical_line_schema(raw_text: str) -> int:
    """Reject marker/key collapsing before invoking the legacy parser helper."""

    marker_occurrences = raw_text.count(PHYSICAL_MARKER)
    marker_lines = [line for line in raw_text.splitlines() if PHYSICAL_MARKER in line]
    if marker_occurrences != len(marker_lines):
        raise ValueError("raw physical log has multiple markers on one line")
    for index, line in enumerate(marker_lines):
        if line.count(PHYSICAL_MARKER) != 1:
            raise ValueError(f"raw physical marker line {index} is ambiguous")
        payload = line.split(PHYSICAL_MARKER, 1)[1].strip()
        raw_tokens = payload.split()
        parsed_tokens = [RAW_FIELD_PATTERN.fullmatch(token) for token in raw_tokens]
        if any(match is None for match in parsed_tokens):
            raise ValueError(f"raw physical marker line {index} has malformed tokens")
        names = tuple(match.group(1) for match in parsed_tokens if match is not None)
        if names != PHYSICAL_RAW_FIELD_ORDER:
            raise ValueError(
                f"raw physical marker line {index} has duplicate, missing, "
                "unknown, or reordered fields"
            )
    return marker_occurrences


def _validate_request_events(path: Path, expected_requests: int) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"request event evidence is missing: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request event line {line_number} is invalid JSON") from exc
        if not isinstance(event, dict):
            raise ValueError(f"request event line {line_number} is not an object")
        events.append(event)
    if len(events) != expected_requests:
        raise ValueError(
            f"request event count {len(events)} does not equal {expected_requests}"
        )
    identities: list[tuple[str, int]] = []
    completion_tokens = 0
    for index, event in enumerate(events):
        trace_id = event.get("trace_id")
        call_index = _strict_nonnegative_int(
            event.get("call_index"), f"event {index} call_index"
        )
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"event {index} has no trace_id")
        identities.append((trace_id, call_index))
        if event.get("ok") is not True or event.get("http_status") != 200:
            raise ValueError(f"event {index} is not a successful HTTP 200 request")
        if _strict_nonnegative_int(
            event.get("attempts"), f"event {index} attempts", positive=True
        ) != 1:
            raise ValueError(f"event {index} was retried")
        history = event.get("attempt_history")
        if not isinstance(history, list) or len(history) != 1:
            raise ValueError(f"event {index} attempt history is not exactly once")
        attempt = history[0]
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt") != 1
            or attempt.get("outcome") != "success"
            or attempt.get("http_status") != 200
            or attempt.get("will_retry") is not False
            or attempt.get("delivery_ambiguous") is not False
        ):
            raise ValueError(f"event {index} has unsafe attempt accounting")
        usage = event.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError(f"event {index} lacks token usage")
        completion_tokens += _strict_nonnegative_int(
            usage.get("completion_tokens"), f"event {index} completion_tokens"
        )
    if len(set(identities)) != expected_requests:
        raise ValueError("request identities are not unique")
    return {
        "request_count": expected_requests,
        "unique_request_identity_count": len(set(identities)),
        "all_requests_succeeded_exactly_once": True,
        "completion_tokens_total": completion_tokens,
        "request_events_sha256": _sha256_file(path),
    }


def validate_fresh_physical_kv_cell(
    cell: Path,
    *,
    expected_profile: str,
    expected_load: int,
    expected_requests: int,
    expected_config_sha256: str,
    expected_engine_shape: Mapping[str, str],
    expected_num_gpu_blocks: int,
    expected_block_size: int,
    expected_target_utilization: float,
    expected_keepalive_s: int,
    expected_preemptions: int = 0,
) -> dict[str, Any]:
    if PHYSICAL_KV_LOG_PARSER_VERSION != 2:
        raise ValueError("fresh physical-KV validation requires parser version 2")
    cell_path = cell.resolve()
    if not cell_path.is_dir():
        raise ValueError(f"physical-KV cell directory is missing: {cell_path}")
    if not expected_profile:
        raise ValueError("expected profile must be non-empty")
    for value, label in (
        (expected_load, "expected load"),
        (expected_requests, "expected requests"),
        (expected_num_gpu_blocks, "expected num_gpu_blocks"),
        (expected_block_size, "expected block_size"),
        (expected_keepalive_s, "expected keepalive"),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if expected_preemptions < 0:
        raise ValueError("expected preemptions must be non-negative")
    if not 0.0 < expected_target_utilization <= 1.0:
        raise ValueError("expected target utilization must be in (0, 1]")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256):
        raise ValueError("expected config SHA256 must be lowercase hex")

    summary_path = cell_path / "summary.json"
    log_summary_path = cell_path / "vllm_log_summary.json"
    events_path = cell_path / "request_events.jsonl"
    frozen_config_path = cell_path.parent / "frozen_config.env"
    frozen_sidecar_path = cell_path.parent / "frozen_config.sha256"
    summary = _load_object(summary_path)
    log_summary = _load_object(log_summary_path)
    for path in (frozen_config_path, frozen_sidecar_path):
        if not path.is_file():
            raise ValueError(f"frozen configuration evidence is missing: {path}")
    config_sha256 = _sha256_file(frozen_config_path)
    if config_sha256 != expected_config_sha256:
        raise ValueError("frozen config SHA does not match the preregistered value")
    sidecar_fields = frozen_sidecar_path.read_text(encoding="utf-8").split()
    if sidecar_fields != [expected_config_sha256, "frozen_config.env"]:
        raise ValueError("frozen config checksum sidecar is invalid")
    frozen_exports = _parse_frozen_exports(frozen_config_path)
    if frozen_exports.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE") != str(expected_keepalive_s):
        raise ValueError("frozen config does not prove the expected HTTP keep-alive")

    environment = summary.get("scheduler_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("summary lacks scheduler_environment")
    expected_environment = {
        **expected_engine_shape,
        "PASTE_STRESS_PROFILE": expected_profile,
        "PASTE_FROZEN_CONFIG_SHA256": expected_config_sha256,
        "VLLM_SCHED_POLICY": "online_joint_pacer_v2",
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "0",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": (
            f"{expected_target_utilization:g}"
        ),
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": "120",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S": "1",
        "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY": "0",
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ValueError(
                f"scheduler environment {name}={environment.get(name)!r}; "
                f"expected {expected!r}"
            )

    raw_native_sequence_cap = expected_engine_shape.get("VLLM_MAX_NUM_SEQS")
    if (
        not isinstance(raw_native_sequence_cap, str)
        or re.fullmatch(r"[1-9][0-9]*", raw_native_sequence_cap) is None
    ):
        raise ValueError("expected VLLM_MAX_NUM_SEQS must be a positive integer string")
    native_sequence_cap = int(raw_native_sequence_cap)
    if native_sequence_cap <= expected_load:
        raise ValueError(
            "physical-KV screen requires a nonbinding native sequence cap "
            f"({native_sequence_cap} <= active load {expected_load})"
        )

    if _strict_nonnegative_int(summary.get("max_active_traces"), "max_active_traces") != expected_load:
        raise ValueError("max_active_traces does not equal expected load")
    workload = summary.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("summary lacks workload evidence")
    if _strict_nonnegative_int(workload.get("trace_count"), "trace_count") != expected_load:
        raise ValueError("workload trace count does not equal expected load")
    if _strict_nonnegative_int(workload.get("request_count"), "request_count") != expected_requests:
        raise ValueError("workload request count does not equal expected requests")
    if workload.get("tool_overlap_mode") != "learned":
        raise ValueError("physical-KV candidate must use learned tool overlap")
    for name, expected in (
        ("requests_total", expected_requests),
        ("requests_success", expected_requests),
        ("requests_failed", 0),
        ("request_attempts_total", expected_requests),
        ("retry_count", 0),
        ("retried_request_count", 0),
        ("retry_success_count", 0),
        ("ambiguous_retry_count", 0),
        ("final_failure_count", 0),
        ("num_preemptions_total", expected_preemptions),
    ):
        if _strict_nonnegative_int(summary.get(name), name) != expected:
            raise ValueError(f"summary {name} is not exactly {expected}")
    if summary.get("kv_swap_happened") is not False:
        raise ValueError("physical-KV cell reports CPU KV swap or ambiguous evidence")
    event_audit = _validate_request_events(events_path, expected_requests)

    port = expected_engine_shape.get("VLLM_PORT")
    if not isinstance(port, str) or re.fullmatch(r"[1-9][0-9]*", port) is None:
        raise ValueError("expected engine shape has no safe VLLM_PORT")
    raw_log_path = cell_path / "server" / f"vllm_{port}.log"
    if not raw_log_path.is_file():
        raise ValueError(f"canonical raw vLLM log is missing: {raw_log_path}")
    stored_telemetry = log_summary.get("physical_kv_admission")
    if not isinstance(stored_telemetry, Mapping):
        raise ValueError("vLLM log summary lacks physical-KV telemetry")
    if summary.get("physical_kv_admission") != stored_telemetry:
        raise ValueError("summary and vLLM-log physical telemetry disagree")
    # Evidence is decoded strictly: deleting corrupt bytes before marker/token
    # accounting could turn an unsafe line into an apparently valid one.
    raw_text = raw_log_path.read_text(encoding="utf-8", errors="strict")
    marker_count = _validate_raw_physical_line_schema(raw_text)
    (
        recomputed,
        experiment_lines,
        samples,
        write_counts,
        prefix_counts,
        suffix_counts,
        full_telemetry,
    ) = _select_experiment_physical_log(
        raw_text,
        stored_telemetry,
        label="canonical parser-v2 raw server log",
    )
    if recomputed != stored_telemetry:
        raise ValueError("stored telemetry does not exactly match raw experiment scope")
    if prefix_counts != [1] or suffix_counts:
        raise ValueError("raw scope is not exactly one warm-up write-count=1 prefix")
    full_samples = _strict_sample_list(full_telemetry, "full raw telemetry")
    full_write_counts = _capacity_write_counts(full_samples, "full raw telemetry")
    if marker_count != len(full_samples) or marker_count != len(samples) + 1:
        raise ValueError("raw marker/sample/warm-up accounting is inconsistent")
    if full_write_counts[0] != 1 or write_counts[0] != 2:
        raise ValueError("raw physical write-counter scope does not start at warm-up=1/run=2")

    malformed = _strict_nonnegative_int(
        recomputed.get("malformed_sample_count"), "malformed_sample_count"
    )
    fail_closed = _strict_nonnegative_int(
        recomputed.get("fail_closed_count"), "fail_closed_count"
    )
    gates = recomputed.get("screening_gates")
    if malformed != 0 or fail_closed != 0:
        raise ValueError("physical-KV telemetry is malformed or fail-closed")
    if not isinstance(gates, Mapping) or any(
        gates.get(name) is not True for name in REQUIRED_SCREENING_GATES
    ):
        raise ValueError("physical-KV screening gates did not all pass")

    expected_capacity = expected_num_gpu_blocks * expected_block_size
    expected_budget_blocks = max(
        1, math.floor(expected_num_gpu_blocks * expected_target_utilization)
    )
    expected_budget = min(
        expected_capacity, expected_budget_blocks * expected_block_size
    )
    rank_capacity_tokens = [
        int(raw.replace(",", ""))
        for raw in GPU_KV_CACHE_TOKENS_PATTERN.findall(raw_text)
    ]
    scheduler_gpu_blocks = [
        int(raw) for raw in SCHEDULER_GPU_BLOCKS_PATTERN.findall(raw_text)
    ]
    if not rank_capacity_tokens or min(rank_capacity_tokens) != expected_capacity:
        raise ValueError(
            "scheduler physical capacity is not the conservative minimum "
            "reported rank KV capacity"
        )
    if not scheduler_gpu_blocks or any(
        value != expected_num_gpu_blocks for value in scheduler_gpu_blocks
    ):
        raise ValueError("raw vLLM scheduler num_gpu_blocks evidence drifted")
    effective_caps: list[int] = []
    fit_admits: list[int] = []
    pressure_above_64 = 0
    rescue_count = 0
    over_soft_zero_admit_count = 0
    for index, sample in enumerate(samples):
        integers = {
            field: _strict_nonnegative_int(
                sample.get(field), f"physical sample {index} {field}"
            )
            for field in PHYSICAL_INTEGER_FIELDS
        }
        target = _finite_float(
            sample.get("target_utilization"),
            f"physical sample {index} target_utilization",
        )
        usage = _finite_float(sample.get("usage"), f"physical sample {index} usage")
        if not 0.0 <= usage <= 1.0:
            raise ValueError(f"physical sample {index} usage is outside [0, 1]")
        if not math.isclose(target, expected_target_utilization, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"physical sample {index} target utilization drifted")
        if (
            integers["num_gpu_blocks"] != expected_num_gpu_blocks
            or integers["block_size"] != expected_block_size
            or integers["capacity_tokens"] != expected_capacity
            or integers["budget_tokens"] != expected_budget
        ):
            raise ValueError(f"physical sample {index} capacity or budget drifted")
        if any(
            integers[field] % integers["block_size"] != 0
            for field in BLOCK_ALIGNED_TOKEN_FIELDS
        ):
            raise ValueError(f"physical sample {index} has non-block-aligned tokens")
        live_usage_upper = integers["live_tokens"] / integers["capacity_tokens"]
        usage_tolerance = 1.0 / integers["num_gpu_blocks"] + 0.5e-6 + 1e-12
        if abs(usage - live_usage_upper) > usage_tolerance:
            raise ValueError(f"physical sample {index} usage/live-token evidence disagrees")
        if integers["committed_tokens"] != (
            max(integers["live_tokens"], integers["logical_live_tokens"])
            + integers["running_growth_tokens"]
            + integers["reserved_tokens"]
        ):
            raise ValueError(f"physical sample {index} committed equation is invalid")
        if integers["live_tokens"] > expected_capacity:
            raise ValueError(f"physical sample {index} live KV exceeds capacity")
        if integers["native_cap"] != native_sequence_cap:
            raise ValueError(f"physical sample {index} native cap drifted")
        if integers["running"] + integers["admit"] > integers["native_cap"]:
            raise ValueError(f"physical sample {index} admission exceeds native cap")
        if integers["effective_cap"] != integers["running"] + integers["admit"]:
            raise ValueError(f"physical sample {index} effective-cap equation is invalid")
        if integers["admit"] > integers["waiting"]:
            raise ValueError(f"physical sample {index} admits more than waiting")
        if integers["admit"] < integers["fit_admit"]:
            raise ValueError(f"physical sample {index} admit is below fit_admit")
        if sample.get("capacity_write_source") != "physical_kv":
            raise ValueError(f"physical sample {index} has the wrong write source")
        rescue = integers["rescue"]
        if rescue not in {0, 1}:
            raise ValueError(f"physical sample {index} rescue flag is invalid")
        expected_fit_admit = integers["admit"] - rescue
        if expected_fit_admit < 0 or integers["fit_admit"] != expected_fit_admit:
            raise ValueError(f"physical sample {index} fit/rescue equation is invalid")
        if rescue == 0 and integers["admit"] > 0 and (
            integers["committed_tokens"] + integers["predicted_admit_tokens"]
            > integers["budget_tokens"]
        ):
            raise ValueError(f"physical sample {index} positive admission exceeds soft budget")
        if rescue == 0 and integers["admit"] > 0:
            if (
                integers["predicted_admit_tokens"] <= 0
                or sample.get("reason") != "budget"
            ):
                raise ValueError(f"physical sample {index} has invalid budget admission")
        if rescue == 0 and integers["admit"] == 0:
            if integers["fit_admit"] != 0 or integers["predicted_admit_tokens"] != 0:
                raise ValueError(f"physical sample {index} zero admission is inconsistent")
            expected_hold_reason = (
                "no_waiting"
                if integers["waiting"] == 0
                else "native_full"
                if integers["running"] == integers["native_cap"]
                else "forecast_hold"
            )
            if sample.get("reason") != expected_hold_reason:
                raise ValueError(f"physical sample {index} zero-admit state is invalid")
            over_soft = integers["committed_tokens"] > integers["budget_tokens"]
            if over_soft:
                if sample.get("decision") != "admit" or expected_hold_reason != "forecast_hold":
                    raise ValueError(
                        f"physical sample {index} has an unsafe over-soft exemption"
                    )
                over_soft_zero_admit_count += 1
        if rescue == 1:
            rescue_count += 1
            if (
                integers["admit"] != 1
                or integers["fit_admit"] != 0
                or integers["predicted_admit_tokens"] <= 0
                or sample.get("reason") not in {"aged_rescue", "empty_progress"}
            ):
                raise ValueError(f"physical sample {index} has an invalid rescue reason")
            if (
                integers["live_tokens"] + integers["predicted_admit_tokens"]
                > expected_capacity
            ):
                raise ValueError(f"physical sample {index} rescue exceeds physical capacity")
        effective_caps.append(integers["effective_cap"])
        fit_admits.append(integers["fit_admit"])
        pressure_above_64 += int(
            integers["running"] > 64
            and integers["waiting"] > 0
            and integers["effective_cap"] > 64
        )
    if len(set(effective_caps)) < 3:
        raise ValueError("physical cap did not vary across at least three values")
    changes = [after - before for before, after in zip(effective_caps, effective_caps[1:])]
    if not any(change > 0 for change in changes) or not any(change < 0 for change in changes):
        raise ValueError("physical cap did not move in both directions")
    if not any(value == 0 for value in fit_admits) or not any(value > 0 for value in fit_admits):
        raise ValueError("physical telemetry lacks zero/positive fit decisions")
    if max(effective_caps) <= 64 or pressure_above_64 < 10:
        raise ValueError("physical telemetry does not prove pressure above cap 64")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": STATUS,
        "cell": _repo_relative(cell_path),
        "source": {
            "summary": {"path": _repo_relative(summary_path), "sha256": _sha256_file(summary_path)},
            "vllm_log_summary": {"path": _repo_relative(log_summary_path), "sha256": _sha256_file(log_summary_path)},
            "request_events": {"path": _repo_relative(events_path), "sha256": _sha256_file(events_path)},
            "canonical_raw_log": {
                "path": _repo_relative(raw_log_path),
                "sha256": _sha256_file(raw_log_path),
                "size_bytes": raw_log_path.stat().st_size,
                "marker_count": marker_count,
                "scope": "full_server_lifecycle",
            },
            "frozen_config": {"path": _repo_relative(frozen_config_path), "sha256": config_sha256},
            "frozen_config_sidecar": {"path": _repo_relative(frozen_sidecar_path), "sha256": _sha256_file(frozen_sidecar_path)},
        },
        "code_binding": {
            "parser": {
                "id": PHYSICAL_KV_LOG_PARSER_ID,
                "version": PHYSICAL_KV_LOG_PARSER_VERSION,
                "path": _repo_relative(PARSER_MODULE),
                "sha256": _sha256_file(PARSER_MODULE),
            },
            "scope_dependency": {
                "path": _repo_relative(DEPENDENCY_MODULE),
                "sha256": _sha256_file(DEPENDENCY_MODULE),
            },
            "validator": {
                "path": _repo_relative(VALIDATOR_MODULE),
                "sha256": _sha256_file(VALIDATOR_MODULE),
            },
        },
        "configuration": {
            "profile": expected_profile,
            "load": expected_load,
            "request_count": expected_requests,
            "engine_shape": dict(sorted(expected_engine_shape.items())),
            "frozen_config_sha256": config_sha256,
            "http_timeout_keep_alive_s": expected_keepalive_s,
            "native_sequence_cap_nonbinding_by_configuration": True,
        },
        "execution": {
            **event_audit,
            "retry_count": 0,
            "failure_count": 0,
            "preemption_count": expected_preemptions,
            "kv_swap_happened": False,
        },
        "physical_kv": {
            "num_gpu_blocks": expected_num_gpu_blocks,
            "block_size": expected_block_size,
            "capacity_tokens": expected_capacity,
            "raw_rank_capacity_tokens": rank_capacity_tokens,
            "raw_rank_capacity_tokens_min": min(rank_capacity_tokens),
            "raw_scheduler_num_gpu_blocks": scheduler_gpu_blocks,
            "capacity_equals_minimum_reported_rank_capacity": True,
            "target_utilization": expected_target_utilization,
            "budget_blocks": expected_budget_blocks,
            "budget_tokens": expected_budget,
            "experiment_sample_count": len(samples),
            "full_raw_sample_count": len(full_samples),
            "experiment_capacity_write_count_first": write_counts[0],
            "experiment_capacity_write_count_last": write_counts[-1],
            "full_capacity_write_count_first": full_write_counts[0],
            "full_capacity_write_count_last": full_write_counts[-1],
            "excluded_warmup_capacity_write_counts": prefix_counts,
            "effective_cap_min": min(effective_caps),
            "effective_cap_max": max(effective_caps),
            "effective_cap_unique_count": len(set(effective_caps)),
            "effective_cap_increase_count": sum(change > 0 for change in changes),
            "effective_cap_decrease_count": sum(change < 0 for change in changes),
            "pressure_above_64_sample_count": pressure_above_64,
            "rescue_sample_count": rescue_count,
            "over_soft_zero_admit_forecast_hold_count": over_soft_zero_admit_count,
            "capacity_write_source": "physical_kv",
            "capacity_write_counts_strictly_increasing": True,
            "stored_vs_raw_experiment_exact_match": True,
            "telemetry_canonical_sha256": _canonical_json_sha256(recomputed),
            "screening_gates": dict(gates),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--expected-load", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-num-gpu-blocks", type=int, required=True)
    parser.add_argument("--expected-block-size", type=int, required=True)
    parser.add_argument("--expected-target-utilization", type=float, required=True)
    parser.add_argument("--expected-keepalive-s", type=int, required=True)
    parser.add_argument("--expected-preemptions", type=int, default=0)
    parser.add_argument(
        "--expect-engine-shape", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_fresh_physical_kv_cell(
            args.cell,
            expected_profile=args.expected_profile,
            expected_load=args.expected_load,
            expected_requests=args.expected_requests,
            expected_config_sha256=args.expected_config_sha256,
            expected_engine_shape=_parse_key_values(args.expect_engine_shape),
            expected_num_gpu_blocks=args.expected_num_gpu_blocks,
            expected_block_size=args.expected_block_size,
            expected_target_utilization=args.expected_target_utilization,
            expected_keepalive_s=args.expected_keepalive_s,
            expected_preemptions=args.expected_preemptions,
        )
        if args.output is not None:
            output = args.output.resolve()
            if output.parent != args.cell.resolve().parent:
                raise ValueError("--output must be a direct child of the C run root")
            _write_json_atomic(output, result)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
