#!/usr/bin/env python3
"""Fail-closed validation for a completed physical-KV Joint screening cell."""

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
RUNNER_DIRECTORY = REPOSITORY_ROOT / "scripts"
if str(RUNNER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIRECTORY))

from run_vllm_trace_experiment import (  # noqa: E402
    PHYSICAL_KV_LOG_PARSER_ID,
    PHYSICAL_KV_LOG_PARSER_VERSION,
    parse_vllm_log_segment,
)


SCHEMA = "paste_repro.physical_kv_admission_validation"
VERSION = 1
RAW_REVALIDATION_SCHEMA = "paste_repro.physical_kv_raw_log_revalidation"
RAW_REVALIDATION_VERSION = 1
PHYSICAL_MARKER = "[sched_policy_patch:physical_kv]"
EXPECTED_ENVIRONMENT = {
    "VLLM_SCHED_POLICY": "online_joint_pacer_v2",
    "VLLM_MAX_NUM_SEQS": "256",
    "VLLM_CUDA_GRAPH_SIZES": "256",
    "VLLM_USE_V1": "1",
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "0",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": "0.93",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": "120",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S": "1",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY": "0",
}
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
REFERENCE_B_ENVIRONMENT = {
    "PASTE_STRESS_PROFILE": "stress240_native256_g256_u86_exact_rescue120",
    "VLLM_SCHED_POLICY": "online_joint_pacer_v2",
    "VLLM_MAX_NUM_SEQS": "256",
    "VLLM_CUDA_GRAPH_SIZES": "256",
    "VLLM_USE_V1": "1",
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "1",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY": "0",
}


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required evidence file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside the repository: {path}") from exc


def _physical_marker_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if PHYSICAL_MARKER in line]


def _strict_sample_list(telemetry: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    samples = telemetry.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label} has no physical-KV decision samples")
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError(f"{label} contains a non-object decision sample")
    return samples


def _capacity_write_counts(
    samples: Sequence[Mapping[str, Any]], label: str
) -> list[int]:
    counts = [
        _strict_int(sample.get("capacity_write_count"), f"{label} write counter")
        for sample in samples
    ]
    if any(after <= before for before, after in zip(counts, counts[1:])):
        raise ValueError(f"{label} capacity-write counters are not strictly increasing")
    return counts


def _parse_complete_physical_log(
    text: str, label: str
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], list[int]]:
    marker_lines = _physical_marker_lines(text)
    telemetry = parse_vllm_log_segment(text).get("physical_kv_admission")
    if not isinstance(telemetry, dict):
        raise ValueError(f"{label} lacks parsed physical-KV telemetry")
    malformed = _strict_int(
        telemetry.get("malformed_sample_count"), f"{label} malformed count"
    )
    fail_closed = _strict_int(
        telemetry.get("fail_closed_count"), f"{label} fail-closed count"
    )
    samples = _strict_sample_list(telemetry, label)
    if malformed != 0 or fail_closed != 0 or len(samples) != len(marker_lines):
        raise ValueError(
            f"{label} is not a complete one-marker/one-safe-sample log: "
            f"markers={len(marker_lines)} samples={len(samples)} "
            f"malformed={malformed} fail_closed={fail_closed}"
        )
    counts = _capacity_write_counts(samples, label)
    return telemetry, marker_lines, samples, counts


def _select_experiment_physical_log(
    raw_text: str,
    stored_telemetry: Mapping[str, Any],
    *,
    label: str,
) -> tuple[
    dict[str, Any],
    list[str],
    list[dict[str, Any]],
    list[int],
    list[int],
    list[int],
    dict[str, Any],
]:
    """Recover the runner's byte-offset scope from its stored write-counter span.

    The runner intentionally parses only the log bytes written after its warm-up
    probe.  The copied ``server.log`` and final raw server log cover the whole
    server lifecycle.  Since scheduler-local capacity writes are strictly
    increasing, the first/last counters retained in the stored telemetry define
    the experiment scope without relying on timestamps or mutable logs.
    """

    (
        full_telemetry,
        marker_lines,
        full_samples,
        full_counts,
    ) = _parse_complete_physical_log(raw_text, label)
    stored_samples = _strict_sample_list(stored_telemetry, "stored telemetry")
    stored_counts = _capacity_write_counts(stored_samples, "stored telemetry")
    first_count, last_count = stored_counts[0], stored_counts[-1]
    selected_pairs = [
        (line, sample)
        for line, sample in zip(marker_lines, full_samples)
        if first_count <= sample["capacity_write_count"] <= last_count
    ]
    prefix_counts = [count for count in full_counts if count < first_count]
    suffix_counts = [count for count in full_counts if count > last_count]
    if not selected_pairs:
        raise ValueError("raw log has no samples in the stored experiment span")
    selected_text = "\n".join(line for line, _ in selected_pairs)
    selected_telemetry = parse_vllm_log_segment(selected_text).get(
        "physical_kv_admission"
    )
    if not isinstance(selected_telemetry, dict):
        raise ValueError("could not parse the selected experiment telemetry")
    selected_samples = _strict_sample_list(
        selected_telemetry, "selected experiment telemetry"
    )
    selected_counts = _capacity_write_counts(
        selected_samples, "selected experiment telemetry"
    )
    if selected_counts[0] != first_count or selected_counts[-1] != last_count:
        raise ValueError("selected experiment write-counter boundaries changed")
    if len(marker_lines) != len(prefix_counts) + len(selected_pairs) + len(suffix_counts):
        raise ValueError("raw physical marker accounting is inconsistent")
    return (
        selected_telemetry,
        [line for line, _ in selected_pairs],
        selected_samples,
        selected_counts,
        prefix_counts,
        suffix_counts,
        full_telemetry,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.resolve()
    if not output.parent.is_dir():
        raise ValueError(f"revalidation output parent is missing: {output.parent}")
    if output.exists():
        raise ValueError(f"revalidation output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary revalidation path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0 or number != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_stat_value(
    telemetry: Mapping[str, Any],
    field: str,
    statistic: str,
) -> float:
    value = telemetry.get(field)
    if not isinstance(value, Mapping) or statistic not in value:
        raise ValueError(f"physical telemetry lacks {field}.{statistic}")
    return _finite(value[statistic], f"physical telemetry {field}.{statistic}")


def validate_reference_b_cell(
    cell: Path,
    *,
    manifest_path: Path,
    expected_config_sha256: str,
    expected_engine_shape: Mapping[str, str],
    expected_load: int = 240,
    expected_requests: int = 2076,
) -> dict[str, Any]:
    """Validate and fingerprint the completed native-admission reference B."""

    cell_path = cell.resolve()
    manifest = manifest_path.resolve()
    if not cell_path.is_dir():
        raise ValueError(f"reference B cell directory is missing: {cell_path}")
    if not manifest.is_file():
        raise ValueError(f"reference B manifest is missing: {manifest}")
    if len(expected_config_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_config_sha256
    ):
        raise ValueError("reference B config SHA256 must be 64 lowercase hex characters")
    if expected_load <= 0 or expected_requests <= 0:
        raise ValueError("reference B expected load/request counts must be positive")

    summary_path = cell_path / "summary.json"
    events_path = cell_path / "request_events.jsonl"
    server_log_path = cell_path / "server.log"
    frozen_config_path = cell_path.parent / "frozen_config.env"
    frozen_sidecar_path = cell_path.parent / "frozen_config.sha256"
    summary = _load_object(summary_path)
    for required_path in (
        events_path,
        server_log_path,
        frozen_config_path,
        frozen_sidecar_path,
    ):
        if not required_path.is_file():
            raise ValueError(f"reference B evidence is missing: {required_path}")

    actual_config_sha256 = _sha256_file(frozen_config_path)
    if actual_config_sha256 != expected_config_sha256:
        raise ValueError(
            "reference B frozen config SHA mismatch: "
            f"{actual_config_sha256} != {expected_config_sha256}"
        )
    sidecar_fields = frozen_sidecar_path.read_text(encoding="utf-8").split()
    if (
        len(sidecar_fields) != 2
        or sidecar_fields[0] != expected_config_sha256
        or sidecar_fields[1] != "frozen_config.env"
    ):
        raise ValueError("reference B frozen-config sidecar is invalid")

    environment = summary.get("scheduler_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("reference B summary lacks scheduler_environment")
    expected_environment = {**REFERENCE_B_ENVIRONMENT, **expected_engine_shape}
    expected_environment["PASTE_FROZEN_CONFIG_SHA256"] = expected_config_sha256
    for name, expected_value in expected_environment.items():
        actual_value = environment.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"reference B environment {name}={actual_value!r}; "
                f"expected {expected_value!r}"
            )
    physical_flag = environment.get(
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION", "0"
    )
    if physical_flag != "0":
        raise ValueError("reference B did not use disabled physical-KV admission")

    workload = summary.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("reference B summary lacks workload evidence")
    if _strict_int(summary.get("max_active_traces"), "reference B max_active_traces") != expected_load:
        raise ValueError("reference B max_active_traces does not match expected load")
    if _strict_int(workload.get("trace_count"), "reference B trace_count") != expected_load:
        raise ValueError("reference B trace count does not match expected load")
    if _strict_int(workload.get("request_count"), "reference B request_count") != expected_requests:
        raise ValueError("reference B workload request count is not exact")
    if workload.get("tool_overlap_mode") != "learned":
        raise ValueError("reference B must use learned tool overlap")
    for name, expected_value in (
        ("requests_total", expected_requests),
        ("requests_success", expected_requests),
        ("requests_failed", 0),
        ("retry_count", 0),
        ("retried_request_count", 0),
        ("final_failure_count", 0),
    ):
        if _strict_int(summary.get(name), f"reference B {name}") != expected_value:
            raise ValueError(f"reference B {name} is not exactly {expected_value}")
    if summary.get("kv_swap_happened") is not False:
        raise ValueError("reference B reports CPU KV swap or ambiguous swap evidence")
    if summary.get("scheduler_metadata_mode") != "online":
        raise ValueError("reference B scheduler metadata mode is not online")

    manifest_payload = _load_object(manifest)
    calibration_relative = (
        manifest_payload.get("four_cell_inputs", {})
        .get("stress", {})
        .get("joint_learned", {})
        .get("online_calibration_workload")
    )
    if not isinstance(calibration_relative, str) or not calibration_relative:
        raise ValueError("stress manifest lacks Joint-learned calibration path")
    expected_calibration = (manifest.parent / calibration_relative).resolve()
    recorded_calibration = summary.get("scheduler_calibration_workload")
    if not isinstance(recorded_calibration, str) or Path(recorded_calibration).resolve() != expected_calibration:
        raise ValueError("reference B calibration workload does not match manifest")

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"reference B request event {line_number} is invalid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(f"reference B request event {line_number} is not an object")
        events.append(event)
    if len(events) != expected_requests:
        raise ValueError("reference B raw request-event count is not exact")
    identities = [
        (str(event.get("trace_id", "")), event.get("call_index"))
        for event in events
    ]
    if (
        len(set(identities)) != expected_requests
        or any(not trace_id or call_index is None for trace_id, call_index in identities)
        or any(event.get("ok") is not True for event in events)
        or any(event.get("attempts") != 1 for event in events)
    ):
        raise ValueError("reference B requests are not unique exactly-once successes")

    server_log = server_log_path.read_text(encoding="utf-8", errors="ignore")
    if (
        "[sched_policy_patch] installed policy=online_joint_pacer_v2 "
        not in server_log
        or "v1=True" not in server_log
    ):
        raise ValueError("reference B lacks Joint-v2/v1 scheduler install evidence")
    if "[sched_policy_patch:physical_kv]" in server_log:
        raise ValueError("reference B unexpectedly contains physical-KV decisions")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "accepted_native_reference_b",
        "cell": cell_path.as_posix(),
        "manifest": manifest.as_posix(),
        "manifest_sha256": _sha256_file(manifest),
        "frozen_config_sha256": actual_config_sha256,
        "physical_kv_admission_effective_value": physical_flag,
        "execution": {
            "trace_count": expected_load,
            "request_count": expected_requests,
            "exactly_once_success": True,
            "tool_overlap_mode": "learned",
            "scheduler_policy": "online_joint_pacer_v2",
            "native_admission": "1",
            "kv_swap_happened": False,
            "num_preemptions_total": summary.get("num_preemptions_total"),
        },
        "engine_shape": dict(sorted(expected_engine_shape.items())),
        "calibration_workload": expected_calibration.as_posix(),
        "evidence_sha256": {
            "summary.json": _sha256_file(summary_path),
            "request_events.jsonl": _sha256_file(events_path),
            "server.log": _sha256_file(server_log_path),
            "frozen_config.env": actual_config_sha256,
            "frozen_config.sha256": _sha256_file(frozen_sidecar_path),
        },
    }


def validate_physical_kv_cell(
    cell: Path,
    *,
    expected_profile: str,
    expected_load: int = 240,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    cell_path = cell.resolve()
    if not cell_path.is_dir():
        raise ValueError(f"physical-KV cell directory is missing: {cell_path}")
    if not expected_profile:
        raise ValueError("expected profile must be non-empty")
    if expected_load <= 0:
        raise ValueError("expected load must be positive")
    if expected_config_sha256 is not None and (
        len(expected_config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_config_sha256)
    ):
        raise ValueError("expected config SHA256 must be 64 lowercase hex characters")

    summary_path = cell_path / "summary.json"
    log_summary_path = cell_path / "vllm_log_summary.json"
    server_log_path = cell_path / "server.log"
    summary = _load_object(summary_path)
    log_summary = _load_object(log_summary_path)
    if not server_log_path.is_file():
        raise ValueError(f"raw vLLM server log is missing: {server_log_path}")

    environment = summary.get("scheduler_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("summary lacks scheduler_environment")
    expected_environment = {
        **EXPECTED_ENVIRONMENT,
        "PASTE_STRESS_PROFILE": expected_profile,
    }
    if expected_config_sha256 is not None:
        expected_environment["PASTE_FROZEN_CONFIG_SHA256"] = expected_config_sha256
    for name, expected_value in expected_environment.items():
        actual_value = environment.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"scheduler environment {name}={actual_value!r}; "
                f"expected {expected_value!r}"
            )

    if _strict_int(summary.get("max_active_traces"), "max_active_traces") != expected_load:
        raise ValueError("physical-KV cell max_active_traces does not match expected load")
    workload = summary.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("summary lacks workload evidence")
    trace_count = _strict_int(workload.get("trace_count"), "workload trace_count")
    request_count = _strict_int(workload.get("request_count"), "workload request_count")
    if trace_count != expected_load:
        raise ValueError("physical-KV workload trace count does not match expected load")
    if workload.get("tool_overlap_mode") != "learned":
        raise ValueError("physical-KV screening cell must use learned overlap")
    requests_total = _strict_int(summary.get("requests_total"), "requests_total")
    requests_success = _strict_int(summary.get("requests_success"), "requests_success")
    requests_failed = _strict_int(summary.get("requests_failed"), "requests_failed")
    final_failures = _strict_int(
        summary.get("final_failure_count"), "final_failure_count"
    )
    if (
        requests_total != request_count
        or requests_success != request_count
        or requests_failed != 0
        or final_failures != 0
    ):
        raise ValueError("physical-KV cell does not have complete successful execution")
    if summary.get("kv_swap_happened") is not False:
        raise ValueError("physical-KV cell reports CPU KV swap or ambiguous swap evidence")

    stored_telemetry = log_summary.get("physical_kv_admission")
    summary_telemetry = summary.get("physical_kv_admission")
    if not isinstance(stored_telemetry, Mapping):
        raise ValueError("vLLM log summary lacks physical-KV telemetry")
    if summary_telemetry != stored_telemetry:
        raise ValueError("summary and vLLM-log physical telemetry disagree")
    (
        recomputed,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _select_experiment_physical_log(
        server_log_path.read_text(encoding="utf-8", errors="ignore"),
        stored_telemetry,
        label="copied server log",
    )
    if recomputed != stored_telemetry:
        raise ValueError(
            "stored physical telemetry does not match its raw experiment span"
        )

    gates = stored_telemetry.get("screening_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("physical telemetry lacks screening_gates")
    failed_gates = [name for name in REQUIRED_SCREENING_GATES if gates.get(name) is not True]
    if failed_gates:
        raise ValueError(f"physical telemetry screening gates failed: {failed_gates}")
    if _strict_int(stored_telemetry.get("fail_closed_count"), "fail_closed_count") != 0:
        raise ValueError("physical telemetry contains fail-closed decisions")
    if _strict_int(
        stored_telemetry.get("malformed_sample_count"), "malformed_sample_count"
    ) != 0:
        raise ValueError("physical telemetry contains malformed samples")

    samples = stored_telemetry.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("physical telemetry has no decision samples")
    write_counts = [
        _strict_int(sample.get("capacity_write_count"), "capacity_write_count")
        for sample in samples
        if isinstance(sample, Mapping)
    ]
    if len(write_counts) != len(samples) or any(
        after <= before for before, after in zip(write_counts, write_counts[1:])
    ):
        raise ValueError("physical capacity-write counters are not strictly increasing")
    if any(
        not isinstance(sample, Mapping)
        or sample.get("capacity_write_source") != "physical_kv"
        for sample in samples
    ):
        raise ValueError("physical telemetry has a non-physical capacity write source")

    capacity_min = _require_stat_value(stored_telemetry, "capacity_tokens", "min")
    capacity_max = _require_stat_value(stored_telemetry, "capacity_tokens", "max")
    utilization_min = _require_stat_value(
        stored_telemetry, "target_utilization", "min"
    )
    utilization_max = _require_stat_value(
        stored_telemetry, "target_utilization", "max"
    )
    native_cap_min = _require_stat_value(stored_telemetry, "native_cap", "min")
    native_cap_max = _require_stat_value(stored_telemetry, "native_cap", "max")
    effective_cap_max = _require_stat_value(
        stored_telemetry, "effective_cap", "max"
    )
    if capacity_min <= 0 or capacity_min != capacity_max:
        raise ValueError("physical token capacity is non-positive or changed during the cell")
    if utilization_min != 0.93 or utilization_max != 0.93:
        raise ValueError("physical telemetry did not use target utilization 0.93")
    if native_cap_min != 256 or native_cap_max != 256:
        raise ValueError("physical telemetry did not retain native cap 256")
    if effective_cap_max <= 64:
        raise ValueError("physical telemetry never demonstrated a cap above 64")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "accepted_physical_kv_telemetry",
        "cell": cell_path.as_posix(),
        "expected_profile": expected_profile,
        "expected_load": expected_load,
        "expected_config_sha256": expected_config_sha256,
        "evidence_sha256": {
            "summary.json": _sha256_file(summary_path),
            "vllm_log_summary.json": _sha256_file(log_summary_path),
            "server.log": _sha256_file(server_log_path),
        },
        "execution": {
            "trace_count": trace_count,
            "request_count": request_count,
            "requests_success": requests_success,
            "retry_count": _strict_int(summary.get("retry_count"), "retry_count"),
            "num_preemptions_total": summary.get("num_preemptions_total"),
            "kv_swap_happened": False,
        },
        "physical_kv": {
            "capacity_tokens": int(capacity_min),
            "target_utilization": utilization_min,
            "native_cap": int(native_cap_min),
            "effective_cap_min": int(
                _require_stat_value(stored_telemetry, "effective_cap", "min")
            ),
            "effective_cap_max": int(effective_cap_max),
            "effective_cap_unique_count": _strict_int(
                stored_telemetry["effective_cap"].get("unique_count"),
                "effective_cap unique_count",
            ),
            "decision_sample_count": _strict_int(
                stored_telemetry.get("sample_count"), "sample_count"
            ),
            "pressure_above_64_sample_count": _strict_int(
                stored_telemetry.get("pressure_above_64_sample_count"),
                "pressure_above_64_sample_count",
            ),
            "capacity_write_count_first": write_counts[0],
            "capacity_write_count_last": write_counts[-1],
            "screening_gates": dict(gates),
        },
    }


def revalidate_physical_kv_raw_log(
    cell: Path,
    *,
    expected_profile: str,
    expected_load: int,
    expected_requests: int,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Revalidate a legacy-failed cell from immutable full-lifecycle raw logs.

    This is deliberately separate from ``validate_physical_kv_cell``.  It does
    not rewrite either stored summary.  Instead, it proves both why parser v1
    rejected the cell and why parser v2 accepts every decision in the runner's
    original experiment scope.
    """

    cell_path = cell.resolve()
    if not cell_path.is_dir():
        raise ValueError(f"physical-KV cell directory is missing: {cell_path}")
    if not expected_profile:
        raise ValueError("expected profile must be non-empty")
    if expected_load <= 0 or expected_requests <= 0:
        raise ValueError("expected load/request counts must be positive")
    if len(expected_config_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_config_sha256
    ):
        raise ValueError("expected config SHA256 must be 64 lowercase hex characters")

    summary_path = cell_path / "summary.json"
    log_summary_path = cell_path / "vllm_log_summary.json"
    summary = _load_object(summary_path)
    log_summary = _load_object(log_summary_path)
    environment = summary.get("scheduler_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("summary lacks scheduler_environment")
    expected_environment = {
        **EXPECTED_ENVIRONMENT,
        "PASTE_STRESS_PROFILE": expected_profile,
        "PASTE_FROZEN_CONFIG_SHA256": expected_config_sha256,
    }
    for name, expected_value in expected_environment.items():
        actual_value = environment.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"scheduler environment {name}={actual_value!r}; "
                f"expected {expected_value!r}"
            )

    port = environment.get("VLLM_PORT")
    if not isinstance(port, str) or re.fullmatch(r"[1-9][0-9]*", port) is None:
        raise ValueError("summary has no safe numeric VLLM_PORT")
    raw_log_path = cell_path / "server" / f"vllm_{port}.log"
    if not raw_log_path.is_file():
        raise ValueError(f"canonical raw vLLM log is missing: {raw_log_path}")

    workload = summary.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("summary lacks workload evidence")
    if _strict_int(summary.get("max_active_traces"), "max_active_traces") != expected_load:
        raise ValueError("physical-KV max_active_traces does not match expected load")
    if _strict_int(workload.get("trace_count"), "trace_count") != expected_load:
        raise ValueError("physical-KV trace count does not match expected load")
    if _strict_int(workload.get("request_count"), "request_count") != expected_requests:
        raise ValueError("physical-KV request count does not match expected count")
    if workload.get("tool_overlap_mode") != "learned":
        raise ValueError("physical-KV cell did not use learned overlap")
    for name, expected_value in (
        ("requests_total", expected_requests),
        ("requests_success", expected_requests),
        ("requests_failed", 0),
        ("retry_count", 0),
        ("retried_request_count", 0),
        ("final_failure_count", 0),
        ("num_preemptions_total", 0),
    ):
        if _strict_int(summary.get(name), name) != expected_value:
            raise ValueError(f"physical-KV {name} is not exactly {expected_value}")
    if summary.get("kv_swap_happened") is not False:
        raise ValueError("physical-KV cell reports CPU KV swap or ambiguous evidence")

    stored_telemetry = log_summary.get("physical_kv_admission")
    if not isinstance(stored_telemetry, Mapping):
        raise ValueError("vLLM log summary lacks physical-KV telemetry")
    if summary.get("physical_kv_admission") != stored_telemetry:
        raise ValueError("original summary and vLLM-log telemetry disagree")
    original_malformed = _strict_int(
        stored_telemetry.get("malformed_sample_count"),
        "original malformed sample count",
    )
    original_fail_closed = _strict_int(
        stored_telemetry.get("fail_closed_count"),
        "original fail-closed count",
    )
    original_gates = stored_telemetry.get("screening_gates")
    if (
        original_malformed <= 0
        or original_fail_closed != 0
        or not isinstance(original_gates, Mapping)
        or original_gates.get("passed") is not False
    ):
        raise ValueError("cell is not the expected legacy parser rejection")

    raw_text = raw_log_path.read_text(encoding="utf-8", errors="ignore")
    (
        experiment_telemetry,
        experiment_lines,
        experiment_samples,
        experiment_counts,
        prefix_counts,
        suffix_counts,
        full_telemetry,
    ) = _select_experiment_physical_log(
        raw_text,
        stored_telemetry,
        label="canonical full-lifecycle raw server log",
    )
    full_samples = _strict_sample_list(
        full_telemetry, "full-lifecycle raw telemetry"
    )
    full_counts = _capacity_write_counts(
        full_samples, "full-lifecycle raw telemetry"
    )
    marker_count = len(_physical_marker_lines(raw_text))
    if prefix_counts != [1] or suffix_counts:
        raise ValueError(
            "experiment scope is not exactly one write-count=1 warm-up prefix"
        )
    if len(experiment_lines) + len(prefix_counts) + len(suffix_counts) != marker_count:
        raise ValueError("raw/experiment marker accounting is not exact")

    legacy_rejected_pairs = [
        (line, sample)
        for line, sample in zip(experiment_lines, experiment_samples)
        if sample["rescue"] == 0
        and sample["committed_tokens"] + sample["predicted_admit_tokens"]
        > sample["budget_tokens"]
    ]
    legacy_accepted_lines = [
        line
        for line, sample in zip(experiment_lines, experiment_samples)
        if not (
            sample["rescue"] == 0
            and sample["committed_tokens"] + sample["predicted_admit_tokens"]
            > sample["budget_tokens"]
        )
    ]
    safe_legacy_shape = lambda sample: (
        sample["decision"] == "admit"
        and sample["reason"] == "forecast_hold"
        and sample["rescue"] == 0
        and sample["admit"] == 0
        and sample["fit_admit"] == 0
        and sample["predicted_admit_tokens"] == 0
    )
    if len(legacy_rejected_pairs) != original_malformed or any(
        not safe_legacy_shape(sample) for _, sample in legacy_rejected_pairs
    ):
        raise ValueError(
            "legacy rejected samples are not exclusively safe zero-admit holds"
        )

    legacy_reconstructed = parse_vllm_log_segment(
        "\n".join(legacy_accepted_lines)
    )["physical_kv_admission"]
    legacy_accepted_samples_exact_match = (
        legacy_reconstructed.get("samples") == stored_telemetry.get("samples")
    )
    legacy_reconstructed["malformed_sample_count"] = len(legacy_rejected_pairs)
    legacy_gates = legacy_reconstructed.get("screening_gates")
    if not isinstance(legacy_gates, dict):
        raise ValueError("could not reconstruct legacy screening gates")
    legacy_gates["no_malformed_samples"] = False
    legacy_gates["passed"] = all(
        value for name, value in legacy_gates.items() if name != "passed"
    )
    legacy_aggregate_exact_match = legacy_reconstructed == stored_telemetry
    if not legacy_accepted_samples_exact_match or not legacy_aggregate_exact_match:
        raise ValueError("raw log cannot exactly reconstruct the stored legacy telemetry")

    gates = experiment_telemetry.get("screening_gates")
    if not isinstance(gates, Mapping) or any(
        gates.get(name) is not True for name in REQUIRED_SCREENING_GATES
    ):
        raise ValueError("parser-v2 experiment telemetry fails screening gates")
    sample_count = len(experiment_samples)
    capacity_equation_count = sum(
        sample["capacity_tokens"]
        == sample["num_gpu_blocks"] * sample["block_size"]
        for sample in experiment_samples
    )
    cap_equation_count = sum(
        sample["effective_cap"]
        == min(sample["native_cap"], sample["running"] + sample["admit"])
        for sample in experiment_samples
    )
    live_safe_count = sum(
        0 <= sample["live_tokens"] <= sample["capacity_tokens"]
        for sample in experiment_samples
    )
    nonrescue_positive = [
        sample
        for sample in experiment_samples
        if sample["rescue"] == 0 and sample["admit"] > 0
    ]
    nonrescue_positive_safe_count = sum(
        sample["committed_tokens"] + sample["predicted_admit_tokens"]
        <= sample["budget_tokens"]
        for sample in nonrescue_positive
    )
    nonrescue_zero = [
        sample
        for sample in experiment_samples
        if sample["rescue"] == 0 and sample["admit"] == 0
    ]
    forecast_hold_over_soft_count = sum(
        safe_legacy_shape(sample)
        and sample["committed_tokens"] > sample["budget_tokens"]
        for sample in experiment_samples
    )
    rescue_samples = [
        sample for sample in experiment_samples if sample["rescue"] == 1
    ]
    rescue_safe_count = sum(
        sample["live_tokens"] + sample["predicted_admit_tokens"]
        <= sample["capacity_tokens"]
        for sample in rescue_samples
    )
    physical_source_count = sum(
        sample["capacity_write_source"] == "physical_kv"
        for sample in experiment_samples
    )
    native_bound_count = sum(
        0 <= sample["effective_cap"] <= sample["native_cap"]
        for sample in experiment_samples
    )
    all_invariants_passed = all(
        (
            sample_count > 0,
            capacity_equation_count == sample_count,
            cap_equation_count == sample_count,
            live_safe_count == sample_count,
            nonrescue_positive_safe_count == len(nonrescue_positive),
            all(sample["predicted_admit_tokens"] == 0 for sample in nonrescue_zero),
            forecast_hold_over_soft_count == len(legacy_rejected_pairs),
            rescue_safe_count == len(rescue_samples),
            physical_source_count == sample_count,
            native_bound_count == sample_count,
        )
    )
    if not all_invariants_passed:
        raise ValueError("independent physical-KV sample audit failed")

    parser_path = RUNNER_DIRECTORY / "run_vllm_trace_experiment.py"
    validator_path = Path(__file__).resolve()
    line_shape = (
        "decision=admit reason=forecast_hold rescue=0 admit=0 "
        "fit_admit=0 predicted_admit_tokens=0"
    )
    return {
        "schema": RAW_REVALIDATION_SCHEMA,
        "version": RAW_REVALIDATION_VERSION,
        "status": "accepted_raw_log_revalidation",
        "source": {
            "raw_log": {
                "path": _repo_relative(raw_log_path),
                "sha256": _sha256_file(raw_log_path),
                "size_bytes": raw_log_path.stat().st_size,
                "scope": "full_server_lifecycle",
                "marker_count": marker_count,
            },
            "summary": {
                "path": _repo_relative(summary_path),
                "sha256": _sha256_file(summary_path),
            },
            "vllm_log_summary": {
                "path": _repo_relative(log_summary_path),
                "sha256": _sha256_file(log_summary_path),
            },
        },
        "parser": {
            "id": PHYSICAL_KV_LOG_PARSER_ID,
            "version": PHYSICAL_KV_LOG_PARSER_VERSION,
            "module_path": _repo_relative(parser_path),
            "module_sha256": _sha256_file(parser_path),
        },
        "validator": {
            "path": _repo_relative(validator_path),
            "sha256": _sha256_file(validator_path),
        },
        "original_post_run_validation": {
            "status": "failed",
            "sample_count": _strict_int(
                stored_telemetry.get("sample_count"), "original sample count"
            ),
            "malformed_sample_count": original_malformed,
            "fail_closed_count": original_fail_closed,
            "screening_gates": dict(original_gates),
        },
        "recomputed": {
            "physical_kv_admission": experiment_telemetry,
            "independent_sample_audit": {
                "experiment_scope": {
                    "derivation": "legacy_stored_telemetry_exact_match",
                    "marker_count": sample_count,
                    "selected_capacity_write_count_first": experiment_counts[0],
                    "selected_capacity_write_count_last": experiment_counts[-1],
                    "excluded_prefix_marker_count": len(prefix_counts),
                    "excluded_prefix_capacity_write_counts": prefix_counts,
                    "excluded_suffix_marker_count": len(suffix_counts),
                    "excluded_suffix_capacity_write_counts": suffix_counts,
                    "raw_marker_accounting_exact": (
                        marker_count
                        == sample_count + len(prefix_counts) + len(suffix_counts)
                    ),
                    "capacity_write_counts_strictly_increasing": all(
                        after > before
                        for before, after in zip(
                            experiment_counts, experiment_counts[1:]
                        )
                    ),
                    "legacy_accepted_sample_count": len(legacy_accepted_lines),
                    "legacy_rejected_sample_count": len(legacy_rejected_pairs),
                    "legacy_accepted_samples_exact_match": (
                        legacy_accepted_samples_exact_match
                    ),
                    "legacy_aggregate_exact_match": legacy_aggregate_exact_match,
                },
                "full_raw_scope": {
                    "marker_count": marker_count,
                    "sample_count": len(full_samples),
                    "malformed_sample_count": _strict_int(
                        full_telemetry.get("malformed_sample_count"),
                        "full raw malformed count",
                    ),
                    "fail_closed_count": _strict_int(
                        full_telemetry.get("fail_closed_count"),
                        "full raw fail-closed count",
                    ),
                    "capacity_write_count_first": full_counts[0],
                    "capacity_write_count_last": full_counts[-1],
                    "capacity_write_counts_strictly_increasing": all(
                        after > before
                        for before, after in zip(full_counts, full_counts[1:])
                    ),
                },
                "legacy_rejection_reason_counts": {
                    "nonrescue_committed_plus_predicted_exceeds_soft_budget": len(
                        legacy_rejected_pairs
                    )
                },
                "legacy_rejection_line_shape_counts": {
                    line_shape: len(legacy_rejected_pairs)
                },
                "invariants": {
                    "sample_count": sample_count,
                    "capacity_equation_pass_count": capacity_equation_count,
                    "effective_cap_equation_pass_count": cap_equation_count,
                    "live_within_physical_capacity_pass_count": live_safe_count,
                    "nonrescue_positive_admit_count": len(nonrescue_positive),
                    "nonrescue_positive_admit_within_soft_budget_count": (
                        nonrescue_positive_safe_count
                    ),
                    "nonrescue_zero_admit_count": len(nonrescue_zero),
                    "forecast_hold_over_soft_budget_zero_admit_count": (
                        forecast_hold_over_soft_count
                    ),
                    "rescue_count": len(rescue_samples),
                    "rescue_within_physical_capacity_count": rescue_safe_count,
                    "capacity_write_source_physical_count": physical_source_count,
                    "native_cap_bound_pass_count": native_bound_count,
                    "all_passed": all_invariants_passed,
                },
                "conclusion": "all_experiment_samples_safe",
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--reference-b", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-profile")
    parser.add_argument("--expected-load", type=int, default=240)
    parser.add_argument("--expected-requests", type=int, default=2076)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument(
        "--revalidation-output",
        type=Path,
        help=(
            "atomically write a parser-v2 raw-log revalidation sidecar; "
            "the original summary files remain untouched"
        ),
    )
    parser.add_argument(
        "--expect-engine-shape",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser.parse_args(argv)


def _parse_expected_engine_shape(items: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise ValueError(f"--expect-engine-shape requires KEY=VALUE, got {item!r}")
        if name in values:
            raise ValueError(f"--expect-engine-shape repeats {name}")
        values[name] = value
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.revalidation_output is not None:
            if args.reference_b:
                raise ValueError("--revalidation-output cannot be used with --reference-b")
            if not args.expected_profile:
                raise ValueError("raw-log revalidation requires --expected-profile")
            if args.expected_config_sha256 is None:
                raise ValueError(
                    "raw-log revalidation requires --expected-config-sha256"
                )
            result = revalidate_physical_kv_raw_log(
                args.cell,
                expected_profile=args.expected_profile,
                expected_load=args.expected_load,
                expected_requests=args.expected_requests,
                expected_config_sha256=args.expected_config_sha256,
            )
            output_relative = _repo_relative(args.revalidation_output)
            _write_json_atomic(args.revalidation_output, result)
            print(
                json.dumps(
                    {
                        "output": output_relative,
                        "sha256": _sha256_file(args.revalidation_output),
                        "status": result["status"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.reference_b:
            if args.manifest is None:
                raise ValueError("--reference-b requires --manifest")
            if args.expected_config_sha256 is None:
                raise ValueError("--reference-b requires --expected-config-sha256")
            result = validate_reference_b_cell(
                args.cell,
                manifest_path=args.manifest,
                expected_config_sha256=args.expected_config_sha256,
                expected_engine_shape=_parse_expected_engine_shape(
                    args.expect_engine_shape
                ),
                expected_load=args.expected_load,
                expected_requests=args.expected_requests,
            )
        else:
            if not args.expected_profile:
                raise ValueError("physical validation requires --expected-profile")
            result = validate_physical_kv_cell(
                args.cell,
                expected_profile=args.expected_profile,
                expected_load=args.expected_load,
                expected_config_sha256=args.expected_config_sha256,
            )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
