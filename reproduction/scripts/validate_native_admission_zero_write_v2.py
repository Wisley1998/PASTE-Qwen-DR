#!/usr/bin/env python3
"""Validate a stress300 Joint native-admission, zero-capacity-write cell."""

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
    parse_vllm_log_segment,
)
from validate_physical_kv_admission_v2 import (  # noqa: E402
    ENGINE_SHAPE_KEYS,
    _parse_key_values,
    _validate_request_events,
    _write_json_atomic,
)


SCHEMA = "paste_repro.native_admission_zero_write_validation_v2"
VERSION = 1
STATUS = "accepted_native_reorder_only_zero_capacity_writes"
PHYSICAL_MARKER = "[sched_policy_patch:physical_kv]"
JOINT_MARKER = "[sched_policy_patch:joint]"
PHYSICAL_ENV_KEYS = (
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
)
EXPORT_PATTERN = re.compile(r'^export ([A-Z][A-Z0-9_]*)="([^"]*)"$')
JOINT_FIELD_PATTERN = re.compile(r"([a-z][a-z0-9_]*)=([^\s]+)")
JOINT_FIELD_ORDER = (
    "pending_returns",
    "reserved_kv",
    "reserved_slots",
    "running",
    "cap",
    "window_s",
)
PARSER_MODULE = RUNNER_DIRECTORY / "run_vllm_trace_experiment.py"
HOOK_MODULE = RUNNER_DIRECTORY / "pythonhooks" / "sched_policy_patch.py"
DEPENDENCY_MODULE = SCRIPT_DIRECTORY / "validate_physical_kv_admission_v2.py"
VALIDATOR_MODULE = Path(__file__).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside the repository: {resolved}") from exc


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _nonnegative_int(value: Any, label: str, *, positive: bool = False) -> int:
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


def _parse_frozen_exports(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), 1
    ):
        match = EXPORT_PATTERN.fullmatch(line)
        if match is None:
            continue
        name, value = match.groups()
        if name in values:
            raise ValueError(f"frozen config repeats {name} at line {line_number}")
        values[name] = value
    return values


def _parse_joint_cap_samples(raw_text: str) -> list[dict[str, int | float]]:
    marker_occurrences = raw_text.count(JOINT_MARKER)
    marker_lines = [line for line in raw_text.splitlines() if JOINT_MARKER in line]
    if marker_occurrences != len(marker_lines):
        raise ValueError("raw Joint telemetry has multiple markers on one line")
    samples: list[dict[str, int | float]] = []
    for index, line in enumerate(marker_lines):
        if line.count(JOINT_MARKER) != 1:
            raise ValueError(f"raw Joint telemetry line {index} is ambiguous")
        payload = line.split(JOINT_MARKER, 1)[1].strip()
        tokens = payload.split()
        matches = [JOINT_FIELD_PATTERN.fullmatch(token) for token in tokens]
        if any(match is None for match in matches):
            raise ValueError(f"raw Joint telemetry line {index} has malformed tokens")
        names = tuple(match.group(1) for match in matches if match is not None)
        if names != JOINT_FIELD_ORDER:
            raise ValueError(
                f"raw Joint telemetry line {index} has duplicate, missing, "
                "unknown, or reordered fields"
            )
        fields = {
            match.group(1): match.group(2)
            for match in matches
            if match is not None
        }
        sample: dict[str, int | float] = {}
        for name in JOINT_FIELD_ORDER[:-1]:
            raw = fields[name]
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"raw Joint telemetry line {index} {name} is not an integer"
                ) from exc
            if value < 0:
                raise ValueError(
                    f"raw Joint telemetry line {index} {name} is negative"
                )
            sample[name] = value
        try:
            window_s = float(fields["window_s"])
        except ValueError as exc:
            raise ValueError(
                f"raw Joint telemetry line {index} window_s is not numeric"
            ) from exc
        if not math.isfinite(window_s) or window_s < 0:
            raise ValueError(
                f"raw Joint telemetry line {index} window_s is invalid"
            )
        sample["window_s"] = window_s
        samples.append(sample)
    return samples


def _empty_physical_evidence(evidence: Any, label: str) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{label} physical-KV telemetry is missing")
    sample_count = _nonnegative_int(evidence.get("sample_count"), f"{label} sample_count")
    malformed = _nonnegative_int(
        evidence.get("malformed_sample_count"), f"{label} malformed_sample_count"
    )
    fail_closed = _nonnegative_int(
        evidence.get("fail_closed_count"), f"{label} fail_closed_count"
    )
    samples = evidence.get("samples")
    reasons = evidence.get("fail_closed_reasons")
    writes = evidence.get("capacity_write_count")
    gates = evidence.get("screening_gates")
    if not isinstance(samples, list) or not isinstance(reasons, list):
        raise ValueError(f"{label} physical-KV lists are malformed")
    if not isinstance(writes, Mapping) or not isinstance(gates, Mapping):
        raise ValueError(f"{label} physical-KV aggregates are malformed")
    if (
        sample_count != 0
        or malformed != 0
        or fail_closed != 0
        or samples
        or reasons
        or any(writes.get(name) is not None for name in ("min", "max", "mean"))
        or gates.get("has_samples") is not False
        or gates.get("passed") is not False
    ):
        raise ValueError(f"{label} unexpectedly contains physical capacity writes")
    return dict(evidence)


def validate_native_zero_write_cell(
    cell: Path,
    *,
    expected_profile: str,
    expected_load: int,
    expected_requests: int,
    expected_config_sha256: str,
    expected_engine_shape: Mapping[str, str],
    expected_keepalive_s: int,
) -> dict[str, Any]:
    if PHYSICAL_KV_LOG_PARSER_VERSION != 2:
        raise ValueError("native zero-write validation requires parser version 2")
    if set(expected_engine_shape) != set(ENGINE_SHAPE_KEYS):
        raise ValueError("engine-shape expectations are not exact")
    if expected_load <= 0 or expected_requests <= 0 or expected_keepalive_s <= 0:
        raise ValueError("load, request count, and keep-alive must be positive")
    if re.fullmatch(r"[0-9a-f]{64}", expected_config_sha256) is None:
        raise ValueError("expected config SHA256 must be lowercase hex")

    cell_path = cell.resolve()
    if not cell_path.is_dir():
        raise ValueError(f"native B cell directory is missing: {cell_path}")
    summary_path = cell_path / "summary.json"
    log_summary_path = cell_path / "vllm_log_summary.json"
    events_path = cell_path / "request_events.jsonl"
    frozen_config_path = cell_path.parent / "frozen_config.env"
    frozen_sidecar_path = cell_path.parent / "frozen_config.sha256"
    summary = _load_object(summary_path, "summary")
    log_summary = _load_object(log_summary_path, "vLLM log summary")
    for path in (frozen_config_path, frozen_sidecar_path):
        if not path.is_file():
            raise ValueError(f"frozen configuration evidence is missing: {path}")
    config_sha = _sha256_file(frozen_config_path)
    if config_sha != expected_config_sha256:
        raise ValueError("frozen config SHA does not match the preregistered value")
    if frozen_sidecar_path.read_text(encoding="utf-8", errors="strict").split() != [
        expected_config_sha256,
        "frozen_config.env",
    ]:
        raise ValueError("frozen config checksum sidecar is invalid")
    frozen_exports = _parse_frozen_exports(frozen_config_path)
    if frozen_exports.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE") != str(expected_keepalive_s):
        raise ValueError("frozen config does not prove the expected HTTP keep-alive")
    if frozen_exports.get("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION") != "1":
        raise ValueError("frozen config does not enable native reorder-only admission")
    present_physical = sorted(set(PHYSICAL_ENV_KEYS) & set(frozen_exports))
    if present_physical:
        raise ValueError(f"frozen native config contains physical keys: {present_physical}")

    environment = summary.get("scheduler_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("summary lacks scheduler_environment")
    expected_environment = {
        **expected_engine_shape,
        "PASTE_STRESS_PROFILE": expected_profile,
        "PASTE_FROZEN_CONFIG_SHA256": expected_config_sha256,
        "VLLM_SCHED_POLICY": "online_joint_pacer_v2",
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "1",
        "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY": "0",
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ValueError(
                f"scheduler environment {name}={environment.get(name)!r}; "
                f"expected {expected!r}"
            )
    leaked_physical = sorted(set(PHYSICAL_ENV_KEYS) & set(environment))
    if leaked_physical:
        raise ValueError(f"native scheduler environment contains physical keys: {leaked_physical}")

    raw_native_cap = expected_engine_shape.get("VLLM_MAX_NUM_SEQS")
    if not isinstance(raw_native_cap, str) or not raw_native_cap.isdigit():
        raise ValueError("expected VLLM_MAX_NUM_SEQS must be an integer string")
    native_cap = int(raw_native_cap)
    if native_cap <= expected_load:
        raise ValueError("native sequence cap must be nonbinding by configuration")
    if _nonnegative_int(summary.get("max_active_traces"), "max_active_traces") != expected_load:
        raise ValueError("max_active_traces does not equal expected load")
    workload = summary.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("summary lacks workload evidence")
    if _nonnegative_int(workload.get("trace_count"), "trace_count") != expected_load:
        raise ValueError("workload trace count does not equal expected load")
    if _nonnegative_int(workload.get("request_count"), "request_count") != expected_requests:
        raise ValueError("workload request count does not equal expected requests")
    if workload.get("tool_overlap_mode") != "learned":
        raise ValueError("native B must use learned tool overlap")
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
    ):
        if _nonnegative_int(summary.get(name), name) != expected:
            raise ValueError(f"summary {name} is not exactly {expected}")
    if summary.get("kv_swap_happened") is not False:
        raise ValueError("native B reports CPU KV swap or ambiguous evidence")
    preemptions = _nonnegative_int(
        summary.get("num_preemptions_total"), "num_preemptions_total"
    )
    event_audit = _validate_request_events(events_path, expected_requests)

    port = expected_engine_shape.get("VLLM_PORT")
    if not isinstance(port, str) or not port.isdigit():
        raise ValueError("expected engine shape has no safe VLLM_PORT")
    raw_log_path = cell_path / "server" / f"vllm_{port}.log"
    if not raw_log_path.is_file():
        raise ValueError(f"canonical raw vLLM log is missing: {raw_log_path}")
    raw_text = raw_log_path.read_text(encoding="utf-8", errors="strict")
    physical_marker_count = raw_text.count(PHYSICAL_MARKER)
    physical_write_token_count = raw_text.count("capacity_write_source=physical_kv")
    if physical_marker_count != 0 or physical_write_token_count != 0:
        raise ValueError("native B raw log contains physical capacity writes")
    stored_summary_physical = _empty_physical_evidence(
        summary.get("physical_kv_admission"), "summary"
    )
    stored_log_physical = _empty_physical_evidence(
        log_summary.get("physical_kv_admission"), "vLLM log summary"
    )
    if stored_summary_physical != stored_log_physical:
        raise ValueError("summary and vLLM-log zero-write telemetry disagree")
    recomputed_physical = _empty_physical_evidence(
        parse_vllm_log_segment(raw_text).get("physical_kv_admission"),
        "canonical raw log",
    )
    if recomputed_physical != stored_log_physical:
        raise ValueError("stored zero-write telemetry does not match canonical raw log")

    joint_samples = _parse_joint_cap_samples(raw_text)
    if not joint_samples:
        raise ValueError("native B lacks Joint cap-observation telemetry")
    observed_caps = [int(sample["cap"]) for sample in joint_samples]
    observed_running = [int(sample["running"]) for sample in joint_samples]
    if set(observed_caps) != {native_cap}:
        raise ValueError("native reorder-only Joint telemetry changed the engine cap")
    if max(observed_running) <= 64:
        raise ValueError("native B did not demonstrate running concurrency above 64")

    max_running = _nonnegative_int(
        log_summary.get("max_running_requests"), "max_running_requests"
    )
    max_waiting = _nonnegative_int(
        log_summary.get("max_waiting_requests"), "max_waiting_requests"
    )
    if max_running <= 64 or max_running > expected_load:
        raise ValueError("native B max running does not prove uncapped stress300 load")
    if max_waiting <= 0:
        raise ValueError("native B did not form a vLLM waiting queue")

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
                "scope": "full_server_lifecycle",
            },
            "frozen_config": {"path": _repo_relative(frozen_config_path), "sha256": config_sha},
            "frozen_config_sidecar": {"path": _repo_relative(frozen_sidecar_path), "sha256": _sha256_file(frozen_sidecar_path)},
        },
        "code_binding": {
            "parser": {
                "id": PHYSICAL_KV_LOG_PARSER_ID,
                "version": PHYSICAL_KV_LOG_PARSER_VERSION,
                "path": _repo_relative(PARSER_MODULE),
                "sha256": _sha256_file(PARSER_MODULE),
            },
            "scheduler_hook": {
                "path": _repo_relative(HOOK_MODULE),
                "sha256": _sha256_file(HOOK_MODULE),
            },
            "dependency": {
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
            "frozen_config_sha256": config_sha,
            "http_timeout_keep_alive_s": expected_keepalive_s,
            "native_sequence_cap": native_cap,
            "native_sequence_cap_nonbinding_by_configuration": True,
            "physical_environment_keys_absent": list(PHYSICAL_ENV_KEYS),
        },
        "execution": {
            **event_audit,
            "retry_count": 0,
            "failure_count": 0,
            "preemption_count": preemptions,
            "kv_swap_happened": False,
        },
        "native_admission": {
            "mode": "joint_reorder_only_native_admission",
            "physical_log_marker_count": physical_marker_count,
            "physical_capacity_write_token_count": physical_write_token_count,
            "physical_capacity_write_count": 0,
            "stored_vs_raw_empty_telemetry_exact_match": True,
            "joint_cap_observation_sample_count": len(joint_samples),
            "joint_cap_observed_min": min(observed_caps),
            "joint_cap_observed_max": max(observed_caps),
            "joint_cap_observed_unique_count": len(set(observed_caps)),
            "joint_cap_always_native": True,
            "joint_running_observed_max": max(observed_running),
            "vllm_stats_max_running": max_running,
            "vllm_stats_max_waiting": max_waiting,
            "observed_running_above_64": True,
            "vllm_formed_waiting_queue": True,
            "zero_capacity_writes_passed": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--expected-load", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-keepalive-s", type=int, required=True)
    parser.add_argument(
        "--expect-engine-shape", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_native_zero_write_cell(
            args.cell,
            expected_profile=args.expected_profile,
            expected_load=args.expected_load,
            expected_requests=args.expected_requests,
            expected_config_sha256=args.expected_config_sha256,
            expected_engine_shape=_parse_key_values(args.expect_engine_shape),
            expected_keepalive_s=args.expected_keepalive_s,
        )
        if args.output is not None:
            output = args.output.resolve()
            if output.parent != args.cell.resolve().parent:
                raise ValueError("--output must be a direct child of the B run root")
            _write_json_atomic(output, result)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
