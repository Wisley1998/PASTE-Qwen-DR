#!/usr/bin/env python3
"""Validate and aggregate M-only fixed-deployment Murakkab live runs.

The input is the existing live runner's ``result.json`` schema.  Performance
numbers are recomputed from raw task, LLM, and physical tool records; embedded
means are never used as observations.  This module additionally fails closed
unless the result is the constrained M cell: the fixed Tongyi deployment,
native FCFS, demand-only tools, a singleton typed DAG plan, and no speculative
physical work.

This is intentionally a descriptive M-only aggregator.  It does not estimate a
PASTE treatment effect and does not infer GPU savings, energy, or cloud cost.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from compare_live_joint_pair import (  # type: ignore
    ValidatedRun,
    _distribution,
    _validate_run,
)


SCHEMA = "paste_repro.murakkab_fixed_live_aggregate"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "reproduction"
    / "configs"
    / "murakkab_fixed_v9_m_only.json"
)
REQUIRED_PREFLIGHT_BINDING_KEYS = frozenset(
    {
        "reproduction/configs/murakkab_fixed_v9_m_only.json",
        "reproduction/workloads/live_joint_wikipedia_frozen_formal_v9.json",
        "scripts/run_live_tool_llm_experiment.py",
        "reproduction/scripts/start_vllm.sh",
        "reproduction/scripts/stop_vllm.sh",
        "reproduction/paste_repro/live_agent.py",
        "reproduction/paste_repro/live_broker.py",
        "reproduction/paste_repro/live_executor.py",
        "reproduction/paste_repro/murakkab_fixed_runtime.py",
        "scripts/pythonhooks/sched_policy_patch.py",
        "reproduction/scripts/validate_live_joint_formal_workload.py",
        "reproduction/results/live_joint/LIVE_TOOL_LLM_PROTOCOL.md",
        "reproduction/scripts/run_murakkab_fixed_live.py",
    }
)


@dataclass(frozen=True)
class FixedSetup:
    """Fields that define the same-deployment M cell.

    ``task_count`` is replaceable in unit tests, but the command-line entry
    point always uses this frozen 80-task setup.
    """

    model: str = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    model_revision: str = "4b0ac5767427a55d08a254f0367e2934976598e0"
    tensor_parallelism: int = 4
    gpu_count: int = 4
    gpu_indices: tuple[int, ...] = (4, 5, 6, 7)
    gpu_type: str = "NVIDIA A100-SXM4-40GB"
    background_policy: str = "registered_shared_resnet_background_v1"
    background_executable: str = "/opt/conda/envs/ptca/bin/python3.10"
    background_cwd: str = "/home/aiscuser/gpu_occupy"
    background_argv: tuple[str, ...] = ("python", "resnet.py")
    background_resolved_script: str = "/home/aiscuser/gpu_occupy/resnet.py"
    background_script_sha256: str = "3239df3d117271605971a2db4b7f6251b42e06a13cac3509c118b2cc16df09a2"
    context_tokens: int = 16_384
    gpu_memory_utilization: float = 0.86
    max_num_batched_tokens: int = 2_048
    max_num_sequences: int = 96
    repetitions: int = 3
    task_count: int = 80
    context_padding_tokens: int = 10_000
    llm_calls_per_task: int = 3
    fixed_final_completion_tokens: int = 192
    tool_workers: int = 4
    search_capacity: int = 3
    visit_capacity: int = 2
    visit_minimum_start_interval_s: float = 2.5
    maximum_speculative_workers: int = 2
    minimum_speculative_workers: int = 0
    maximum_speculative_pending: int = 128
    speculative_ttl_s: float = 120.0
    workload_split_id: str = "live-joint-wikipedia-frozen-formal-v9"
    workload_file_sha256: str = "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20"
    selected_workload_sha256: str = "750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c"


FIXED_SETUP = FixedSetup()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    lower = sys.float_info.min if positive else 0.0
    if not math.isfinite(result) or result < lower:
        qualifier = "positive" if positive else "finite and non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _sha256_text(value: Any, label: str) -> str:
    digest = _nonempty_string(value, label)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be lowercase SHA256")
    return digest


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    return _mapping(value, label)


def _utc_epoch(value: Any, label: str) -> float:
    text = _nonempty_string(value, label)
    if not text.endswith("Z"):
        raise ValueError(f"{label} must be an explicit UTC timestamp ending in Z")
    body = text[:-1]
    if "." in body:
        prefix, fraction = body.rsplit(".", 1)
        if not fraction.isdigit():
            raise ValueError(f"{label} is not a valid ISO-8601 timestamp")
        # datetime supports microseconds; retain enough precision for the
        # millisecond-tolerance evidence checks while accepting ns captures.
        body = prefix + "." + fraction[:6].ljust(6, "0")
    try:
        parsed = datetime.fromisoformat(body + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must use UTC")
    return parsed.timestamp()


def _expect_near(
    observed: Any,
    expected: float,
    label: str,
    *,
    tolerance: float = 1e-3,
) -> float:
    value = _finite(observed, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"{label} must be within {tolerance}s of {expected}, observed {value}"
        )
    return value


def _expect_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} must be {expected!r}, observed {observed!r}")


def _expect_float(observed: Any, expected: float, label: str) -> None:
    value = _finite(observed, label)
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} must be {expected}, observed {value}")


def _seconds_distribution(values: Sequence[float]) -> dict[str, float | int]:
    raw = _distribution(values)
    return {
        "count": raw["count"],
        "mean_s": raw["mean"],
        "p50_s": raw["p50"],
        "p95_s": raw["p95"],
        "p99_s": raw["p99"],
        "max_s": raw["max"],
    }


def _rate_distribution(values: Sequence[float]) -> dict[str, float | int]:
    raw = _distribution(values)
    return {
        "count": raw["count"],
        "mean_tasks_per_s": raw["mean"],
        "p50_tasks_per_s": raw["p50"],
        "p95_tasks_per_s": raw["p95"],
        "p99_tasks_per_s": raw["p99"],
        "max_tasks_per_s": raw["max"],
    }


def _repetition_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty repetition values")
    numeric = [float(value) for value in values]
    lower = min(numeric)
    upper = max(numeric)
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "min": lower,
        "max": upper,
        "range": upper - lower,
    }


def _validate_protocol(setup: FixedSetup) -> str:
    protocol = _load_json(PROTOCOL_PATH, "M-only execution protocol")
    exact = {
        "schema": "paste_repro.murakkab_fixed_v9_m_only_execution",
        "version": 1,
        "status": "fixed_engineering_execution",
        "evidence_class": "fixed-v9-setup-engineering",
        "confirmatory_eligible": False,
        "repetitions": setup.repetitions,
        "fresh_server_per_repetition": True,
        "result_cache_empty_per_repetition": True,
    }
    for key, expected in exact.items():
        _expect_equal(protocol.get(key), expected, f"protocol.{key}")
    implementation = _mapping(protocol.get("implementation"), "protocol.implementation")
    implementation_exact = {
        "kind": "constrained_murakkab_style_emulation",
        "official_code_used": False,
        "official_runtime_reproduced": False,
        "timed_runtime_semantics": "A-equivalent native-FCFS plus demand-only execution",
    }
    for key, expected in implementation_exact.items():
        _expect_equal(implementation.get(key), expected, f"protocol.implementation.{key}")
    attempt_policy = _mapping(protocol.get("attempt_policy"), "protocol.attempt_policy")
    attempt_exact = {
        "planned_performance_repetitions": setup.repetitions,
        "development_smoke_is_not_a_performance_repetition": True,
        "failed_or_contaminated_repetitions_are_retained": True,
        "failed_or_contaminated_repetitions_must_not_be_silently_replaced": True,
        "performance_based_parameter_changes_between_repetitions_allowed": False,
    }
    for key, expected in attempt_exact.items():
        _expect_equal(attempt_policy.get(key), expected, f"protocol.attempt_policy.{key}")
    workload = _mapping(protocol.get("workload"), "protocol.workload")
    _expect_equal(workload.get("sha256"), setup.workload_file_sha256, "protocol.workload.sha256")
    _expect_equal(workload.get("source_count"), FIXED_SETUP.task_count, "protocol.workload.source_count")
    _expect_equal(workload.get("offered_concurrency"), FIXED_SETUP.task_count, "protocol.workload.offered_concurrency")
    shared = _mapping(protocol.get("shared_setup"), "protocol.shared_setup")
    _expect_equal(shared.get("model"), setup.model, "protocol.shared_setup.model")
    _expect_equal(shared.get("model_revision"), setup.model_revision, "protocol.shared_setup.model_revision")
    _expect_equal(shared.get("tensor_parallelism"), setup.tensor_parallelism, "protocol.shared_setup.tensor_parallelism")
    gpu = _mapping(shared.get("gpu"), "protocol.shared_setup.gpu")
    _expect_equal(gpu.get("count"), setup.gpu_count, "protocol.shared_setup.gpu.count")
    _expect_equal(gpu.get("type"), setup.gpu_type, "protocol.shared_setup.gpu.type")
    _expect_equal(gpu.get("visible_indices"), list(setup.gpu_indices), "protocol.shared_setup.gpu.visible_indices")
    _expect_equal(gpu.get("background_policy"), setup.background_policy, "protocol.shared_setup.gpu.background_policy")
    background = _mapping(
        gpu.get("registered_background"),
        "protocol.shared_setup.gpu.registered_background",
    )
    background_exact = {
        "required": True,
        "user_confirmed_prior_paste_same_condition": True,
        "selected_gpu_compute_app_records": setup.gpu_count,
        "same_pid_on_every_selected_gpu": True,
        "additional_selected_gpu_compute_apps_allowed": False,
        "executable": setup.background_executable,
        "cwd": setup.background_cwd,
        "argv": list(setup.background_argv),
        "resolved_script": setup.background_resolved_script,
        "resolved_script_sha256": setup.background_script_sha256,
        "identity_must_match_before_and_after": True,
    }
    for key, expected in background_exact.items():
        _expect_equal(background.get(key), expected, f"protocol.registered_background.{key}")
    return _sha256_file(PROTOCOL_PATH)


def _normalize_artifact_roots(values: Sequence[Path]) -> tuple[Path, ...]:
    raw_values = tuple(values) if values else (REPOSITORY_ROOT,)
    roots: list[Path] = []
    for index, raw in enumerate(raw_values):
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise ValueError(f"artifact root {index} must be an absolute path")
        root = candidate.resolve()
        if not root.is_dir():
            raise ValueError(f"artifact root is not a directory: {root}")
        for marker in ("reproduction", "scripts"):
            if not (root / marker).is_dir():
                raise ValueError(
                    f"artifact root lacks repository marker {marker!r}: {root}"
                )
        if root in roots:
            raise ValueError(f"artifact root was supplied more than once: {root}")
        roots.append(root)
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if root in other.parents or other in root.parents:
                raise ValueError(
                    f"artifact roots overlap and would make mapping ambiguous: {root}, {other}"
                )
    return tuple(roots)


def _artifact_key(path: Path, *, artifact_roots: Sequence[Path]) -> str:
    resolved = path.resolve()
    matches: list[tuple[Path, Path]] = []
    for root in artifact_roots:
        try:
            matches.append((root, resolved.relative_to(root)))
        except ValueError:
            continue
    if not matches:
        raise ValueError(
            f"artifact is outside every configured artifact root: {resolved}"
        )
    if len(matches) != 1:
        raise ValueError(f"artifact maps to multiple artifact roots: {resolved}")
    relative = matches[0][1]
    if relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact does not have a safe repository-relative key: {resolved}")
    return relative.as_posix()


def _validate_manifest_artifact_keys(artifacts: Mapping[str, Any]) -> None:
    for raw_key in artifacts:
        if not isinstance(raw_key, str) or not raw_key or "\\" in raw_key:
            raise ValueError("completion manifest artifact keys must be non-empty POSIX paths")
        key_path = Path(raw_key)
        normalized = key_path.as_posix()
        if (
            key_path.is_absolute()
            or ".." in key_path.parts
            or "." in key_path.parts
            or normalized != raw_key
        ):
            raise ValueError(
                "completion manifest artifact keys must be canonical repository-relative paths"
            )


def _validate_hardware_snapshot(
    payload: Mapping[str, Any], *, label: str, setup: FixedSetup
) -> dict[str, Any]:
    indices = payload.get("selected_gpu_indices")
    if indices != list(setup.gpu_indices):
        raise ValueError(f"{label} does not bind GPUs {list(setup.gpu_indices)}")
    selected = payload.get("selected_gpus")
    if not isinstance(selected, list) or len(selected) != setup.gpu_count:
        raise ValueError(f"{label}.selected_gpus must contain exactly {setup.gpu_count} rows")
    uuids: list[str] = []
    for expected_index, raw in zip(setup.gpu_indices, selected, strict=True):
        row = _mapping(raw, f"{label}.selected_gpus")
        _expect_equal(row.get("index"), expected_index, f"{label}.gpu.index")
        _expect_equal(row.get("name"), setup.gpu_type, f"{label}.gpu.name")
        uuids.append(_nonempty_string(row.get("uuid"), f"{label}.gpu.uuid"))
    if len(set(uuids)) != setup.gpu_count:
        raise ValueError(f"{label} does not contain four distinct GPU UUIDs")
    registered = _mapping(
        payload.get("registered_background"),
        f"{label}.registered_background",
    )
    registered_exact = {
        "valid": True,
        "policy": setup.background_policy,
        "executable": setup.background_executable,
        "cwd": setup.background_cwd,
        "argv": list(setup.background_argv),
        "resolved_script": setup.background_resolved_script,
        "resolved_script_sha256": setup.background_script_sha256,
        "user_confirmed_prior_paste_same_condition": True,
        "selected_gpu_indices": list(setup.gpu_indices),
        "selected_gpu_uuids": uuids,
        "selected_application_record_count": setup.gpu_count,
        "additional_selected_gpu_compute_apps_observed": False,
    }
    for key, expected in registered_exact.items():
        _expect_equal(registered.get(key), expected, f"{label}.registered_background.{key}")
    pid = _integer(
        registered.get("pid"),
        f"{label}.registered_background.pid",
        positive=True,
    )
    proc_starttime_ticks = _integer(
        registered.get("proc_starttime_ticks"),
        f"{label}.registered_background.proc_starttime_ticks",
        positive=True,
    )
    boot_id = _nonempty_string(
        registered.get("boot_id"),
        f"{label}.registered_background.boot_id",
    )
    per_gpu = registered.get("per_gpu_rows")
    if not isinstance(per_gpu, list) or len(per_gpu) != setup.gpu_count:
        raise ValueError(f"{label}.registered_background.per_gpu_rows must contain four rows")
    normalized_rows: list[dict[str, Any]] = []
    for index, (gpu_index, gpu_uuid, raw) in enumerate(
        zip(setup.gpu_indices, uuids, per_gpu, strict=True)
    ):
        row = _mapping(raw, f"{label}.registered_background.per_gpu_rows[{index}]")
        _expect_equal(row.get("gpu_index"), gpu_index, f"{label}.per_gpu[{index}].gpu_index")
        _expect_equal(row.get("gpu_uuid"), gpu_uuid, f"{label}.per_gpu[{index}].gpu_uuid")
        _expect_equal(row.get("pid"), pid, f"{label}.per_gpu[{index}].pid")
        _expect_equal(row.get("process_name"), "python", f"{label}.per_gpu[{index}].process_name")
        used_memory = _finite(
            row.get("used_memory_mib"),
            f"{label}.per_gpu[{index}].used_memory_mib",
            positive=True,
        )
        normalized_rows.append(
            {
                "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": "python",
                "used_memory_mib": used_memory,
            }
        )
    _expect_equal(
        payload.get("selected_gpu_compute_applications"),
        per_gpu,
        f"{label}.selected_gpu_compute_applications",
    )
    _expect_equal(
        payload.get("selected_gpu_background_process_count"),
        setup.gpu_count,
        f"{label}.selected_gpu_background_process_count",
    )

    # Recompute the "no additional selected-GPU application" assertion from
    # the complete nvidia-smi application table instead of trusting the
    # wrapper's boolean.  Applications on non-selected GPUs are out of scope.
    all_applications = payload.get("all_compute_applications")
    if not isinstance(all_applications, list):
        raise ValueError(f"{label}.all_compute_applications must be an array")
    all_application_rows = [
        _mapping(row, f"{label}.all_compute_applications[{index}]")
        for index, row in enumerate(all_applications)
    ]
    selected_uuid_set = set(uuids)
    selected_applications = [
        row for row in all_application_rows if row.get("gpu_uuid") in selected_uuid_set
    ]
    if len(selected_applications) != setup.gpu_count:
        raise ValueError(
            f"{label} must contain exactly one compute application on each selected GPU"
        )
    by_uuid: dict[str, list[Mapping[str, Any]]] = {
        gpu_uuid: [
            row for row in selected_applications if row.get("gpu_uuid") == gpu_uuid
        ]
        for gpu_uuid in uuids
    }
    if any(len(rows) != 1 for rows in by_uuid.values()):
        raise ValueError(
            f"{label} contains an additional or missing selected-GPU compute application"
        )
    for index, normalized in enumerate(normalized_rows):
        raw = by_uuid[normalized["gpu_uuid"]][0]
        for key in ("gpu_uuid", "pid", "process_name"):
            _expect_equal(
                raw.get(key),
                normalized[key],
                f"{label}.all_compute_applications[{index}].{key}",
            )
        observed_memory = _finite(
            raw.get("used_memory_mib"),
            f"{label}.all_compute_applications[{index}].used_memory_mib",
            positive=True,
        )
        _expect_float(
            observed_memory,
            float(normalized["used_memory_mib"]),
            f"{label}.all_compute_applications[{index}].used_memory_mib",
        )
    return {
        "pid": pid,
        "proc_starttime_ticks": proc_starttime_ticks,
        "boot_id": boot_id,
        "executable": registered["executable"],
        "cwd": registered["cwd"],
        "argv": list(registered["argv"]),
        "resolved_script": registered["resolved_script"],
        "resolved_script_sha256": registered["resolved_script_sha256"],
        "selected_gpu_indices": list(setup.gpu_indices),
        "selected_gpu_uuids": uuids,
        "per_gpu_rows": normalized_rows,
        "registered_background": dict(registered),
    }


def _validate_background_continuity(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    setup: FixedSetup,
) -> dict[str, Any]:
    stable_fields = (
        "pid",
        "proc_starttime_ticks",
        "boot_id",
        "executable",
        "cwd",
        "argv",
        "resolved_script",
        "resolved_script_sha256",
        "selected_gpu_indices",
        "selected_gpu_uuids",
    )
    changed = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in stable_fields
        if before.get(key) != after.get(key)
    }
    if changed:
        raise ValueError(
            "registered ResNet process identity changed between hardware snapshots: "
            + ", ".join(sorted(changed))
        )
    return {
        "valid": True,
        "policy": setup.background_policy,
        "same_process_identity_before_after": True,
        "pid": before["pid"],
        "proc_starttime_ticks": before["proc_starttime_ticks"],
        "boot_id": before["boot_id"],
        "resolved_script_sha256": before["resolved_script_sha256"],
        "selected_gpu_indices": list(before["selected_gpu_indices"]),
        "selected_gpu_uuids": list(before["selected_gpu_uuids"]),
        "user_confirmed_prior_paste_same_condition": True,
        "load_intensity_equivalence_claimed": False,
    }


def _validate_run_sidecars(
    run: ValidatedRun,
    *,
    murakkab: Mapping[str, Any],
    engineering: Mapping[str, Any],
    setup: FixedSetup,
    artifact_roots: Sequence[Path],
) -> dict[str, Any]:
    """Bind the enriched result to the plan, raw result, GPU audit, and manifest."""

    if run.path.parent.name != "evidence":
        raise ValueError("M enriched result must be stored at <run_root>/evidence/result.json")
    run_root = run.path.parent.parent
    preflight_path = run_root / "preflight.json"
    plan_path = run_root / "run_plan.json"
    raw_result_path = run_root / "runner_raw" / "result.json"
    hardware_before_path = run_root / "hardware_before.json"
    hardware_after_path = run_root / "hardware_after.json"
    completion_path = run_root / "completed_run.json"
    for path, label in (
        (preflight_path, "preflight evidence"),
        (plan_path, "run plan"),
        (raw_result_path, "unmodified runner result"),
        (hardware_before_path, "hardware-before audit"),
        (hardware_after_path, "hardware-after audit"),
        (completion_path, "completion manifest"),
    ):
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")

    _expect_equal(
        _sha256_file(plan_path),
        murakkab.get("plan_sha256"),
        "murakkab_fixed.plan_sha256",
    )
    _expect_equal(
        _sha256_file(raw_result_path),
        murakkab.get("raw_runner_result_sha256"),
        "murakkab_fixed.raw_runner_result_sha256",
    )
    plan = _load_json(plan_path, "run plan")
    preflight = _load_json(preflight_path, "preflight evidence")
    _expect_equal(preflight.get("valid"), True, "preflight.valid")
    _expect_equal(
        preflight.get("protocol_sha256"),
        _sha256_file(PROTOCOL_PATH),
        "preflight.protocol_sha256",
    )
    _expect_equal(
        preflight.get("workload_sha256"),
        setup.workload_file_sha256,
        "preflight.workload_sha256",
    )
    bindings_raw = _mapping(preflight.get("bindings"), "preflight.bindings")
    bindings = {str(key): value for key, value in bindings_raw.items()}
    missing_bindings = sorted(REQUIRED_PREFLIGHT_BINDING_KEYS - set(bindings))
    if missing_bindings:
        raise ValueError(
            "preflight.bindings is missing required inputs: "
            + ", ".join(missing_bindings)
        )
    for relative, raw_digest in sorted(bindings.items()):
        digest = _sha256_text(raw_digest, f"preflight.bindings[{relative!r}]")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"preflight binding path is not repository-relative: {relative}")
        bound_path = (REPOSITORY_ROOT / relative_path).resolve()
        try:
            bound_path.relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"preflight binding escapes repository: {relative}") from exc
        if not bound_path.is_file():
            raise ValueError(f"preflight bound input is missing: {relative}")
        _expect_equal(_sha256_file(bound_path), digest, f"preflight binding {relative}")
    bindings_sha = hashlib.sha256(
        json.dumps(
            bindings,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    for key in ("workflow_sha256", "registry_sha256", "selected_candidate_id"):
        _expect_equal(plan.get(key), murakkab.get(key), f"run plan {key}")
    plan_exact = {
        "evidence_tier": "fixed-v9-setup-engineering",
        "confirmatory_eligible": False,
        "performance_comparable": True,
        "source_limit": None,
        "run_tag": engineering.get("run_tag"),
        "repetition": engineering.get("repetition"),
        "server_instance_id": engineering.get("server_instance_id"),
        "registered_background_policy": setup.background_policy,
        "candidate_count": 1,
        "planner": "singleton_constrained_selection",
        "optimizer_outside_timed_path": True,
        "typed_dag_validated": True,
        "dependency_ready_dispatch": True,
    }
    for key, expected in plan_exact.items():
        _expect_equal(plan.get(key), expected, f"run plan {key}")

    hardware_before = _load_json(hardware_before_path, "hardware-before audit")
    hardware_after = _load_json(hardware_after_path, "hardware-after audit")
    before_identity = _validate_hardware_snapshot(
        hardware_before,
        label="hardware_before",
        setup=setup,
    )
    after_identity = _validate_hardware_snapshot(
        hardware_after,
        label="hardware_after",
        setup=setup,
    )
    continuity = _validate_background_continuity(
        before_identity,
        after_identity,
        setup=setup,
    )
    _expect_equal(
        plan.get("registered_background"),
        hardware_before.get("registered_background"),
        "run plan registered_background",
    )

    hardware_evidence = _mapping(
        murakkab.get("hardware_evidence"),
        "murakkab_fixed.hardware_evidence",
    )
    hardware_evidence_exact = {
        "selected_gpu_indices": list(setup.gpu_indices),
        "selected_gpu_names": [setup.gpu_type] * setup.gpu_count,
        "selected_gpu_uuids": before_identity["selected_gpu_uuids"],
        "registered_background_policy": setup.background_policy,
        "registered_background_before": hardware_before["registered_background"],
        "registered_background_after": hardware_after["registered_background"],
        "registered_background_continuity": continuity,
        "before_path": str(hardware_before_path),
        "before_sha256": _sha256_file(hardware_before_path),
        "after_path": str(hardware_after_path),
        "after_sha256": _sha256_file(hardware_after_path),
    }
    for key, expected in hardware_evidence_exact.items():
        _expect_equal(
            hardware_evidence.get(key),
            expected,
            f"murakkab_fixed.hardware_evidence.{key}",
        )

    provenance = _mapping(
        run.payload.get("murakkab_provenance"),
        "murakkab_provenance",
    )
    provenance_exact = {
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "hardware_before_path": str(hardware_before_path),
        "hardware_before_sha256": _sha256_file(hardware_before_path),
        "hardware_after_path": str(hardware_after_path),
        "hardware_after_sha256": _sha256_file(hardware_after_path),
    }
    for key, expected in provenance_exact.items():
        _expect_equal(provenance.get(key), expected, f"murakkab_provenance.{key}")
    raw_provenance = _mapping(
        provenance.get("unmodified_runner_result"),
        "murakkab_provenance.unmodified_runner_result",
    )
    _expect_equal(
        raw_provenance.get("path"),
        str(raw_result_path),
        "murakkab_provenance.unmodified_runner_result.path",
    )
    _expect_equal(
        raw_provenance.get("sha256"),
        _sha256_file(raw_result_path),
        "murakkab_provenance.unmodified_runner_result.sha256",
    )

    completion = _load_json(completion_path, "completion manifest")
    completion_exact = {
        "schema": "paste_repro.murakkab_fixed_live_completion",
        "version": 1,
        "completed": True,
        "evidence_tier": "fixed-v9-setup-engineering",
        "confirmatory_eligible": False,
        "run_tag": engineering.get("run_tag"),
        "repetition": engineering.get("repetition"),
    }
    for key, expected in completion_exact.items():
        _expect_equal(completion.get(key), expected, f"completion manifest {key}")
    _expect_equal(
        completion.get("registered_background"),
        continuity,
        "completion manifest registered_background",
    )
    artifacts = _mapping(completion.get("artifacts"), "completion manifest artifacts")
    _validate_manifest_artifact_keys(artifacts)
    bound_paths = (
        preflight_path,
        plan_path,
        raw_result_path,
        hardware_before_path,
        hardware_after_path,
        run.path,
        run.path.parent / "queue_timeline.jsonl",
    )
    for path in bound_paths:
        key = _artifact_key(path, artifact_roots=artifact_roots)
        expected = artifacts.get(key)
        _sha256_text(expected, f"completion manifest artifact {key}")
        _expect_equal(_sha256_file(path), expected, f"completion manifest artifact {key}")

    return {
        "run_root": str(run_root),
        "run_plan_path": str(plan_path),
        "preflight_sha256": _sha256_file(preflight_path),
        "preflight_bindings": dict(sorted(bindings.items())),
        "preflight_bindings_sha256": bindings_sha,
        "hardware_before_sha256": _sha256_file(hardware_before_path),
        "hardware_after_sha256": _sha256_file(hardware_after_path),
        "registered_background": continuity,
        "registered_background_pid": before_identity["pid"],
        "registered_background_proc_starttime_ticks": before_identity[
            "proc_starttime_ticks"
        ],
        "registered_background_boot_id": before_identity["boot_id"],
        "completion_manifest_sha256": _sha256_file(completion_path),
        "unmodified_runner_result_path": str(raw_result_path),
    }


def _validate_fixed_config(
    run: ValidatedRun,
    setup: FixedSetup,
    *,
    artifact_roots: Sequence[Path] = (REPOSITORY_ROOT,),
) -> dict[str, Any]:
    config = run.config
    label = str(run.path)

    _expect_equal(config.get("model"), setup.model, f"{label}.config.model")
    for key, expected in (
        ("task_count", setup.task_count),
        ("independent_source_count", setup.task_count),
        ("replicas", 1),
        ("max_active_tasks", setup.task_count),
        ("context_padding_tokens", setup.context_padding_tokens),
        ("fixed_final_completion_tokens", setup.fixed_final_completion_tokens),
        ("tool_workers", setup.tool_workers),
        ("search_tool_capacity", setup.search_capacity),
        ("visit_tool_capacity", setup.visit_capacity),
        ("speculative_tool_workers", setup.maximum_speculative_workers),
        ("min_speculative_tool_workers", setup.minimum_speculative_workers),
        ("max_speculative_pending", setup.maximum_speculative_pending),
    ):
        _expect_equal(config.get(key), expected, f"{label}.config.{key}")
    for key, expected in (
        ("visit_min_start_interval_s", setup.visit_minimum_start_interval_s),
        ("speculative_ttl_s", setup.speculative_ttl_s),
    ):
        _expect_float(config.get(key), expected, f"{label}.config.{key}")

    _expect_equal(config.get("speculation_mode"), "off", f"{label}.config.speculation_mode")
    _expect_equal(config.get("fixed_final_completion_enabled"), True, f"{label}.config.fixed_final_completion_enabled")
    _expect_equal(config.get("shared_bounded_tool_pool"), True, f"{label}.config.shared_bounded_tool_pool")

    scheduler = _mapping(config.get("scheduler_environment"), f"{label}.config.scheduler_environment")
    scheduler_exact = {
        "MODEL_ID": setup.model,
        "MODEL_REVISION": setup.model_revision,
        "VLLM_DTYPE": "bfloat16",
        "VLLM_TP_SIZE": str(setup.tensor_parallelism),
        "VLLM_MAX_MODEL_LEN": str(setup.context_tokens),
        "VLLM_GPU_MEMORY_UTILIZATION": str(setup.gpu_memory_utilization),
        "VLLM_MAX_NUM_BATCHED_TOKENS": str(setup.max_num_batched_tokens),
        "VLLM_MAX_NUM_SEQS": str(setup.max_num_sequences),
        "VLLM_ENABLE_PREFIX_CACHING": "1",
        "VLLM_CUDA_GRAPH_SIZES": "32",
        "VLLM_USE_V1": "1",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
        "VLLM_SCHED_POLICY": "fcfs",
    }
    for key, expected in scheduler_exact.items():
        _expect_equal(scheduler.get(key), expected, f"{label}.scheduler_environment.{key}")
    visible = _nonempty_string(
        scheduler.get("CUDA_VISIBLE_DEVICES"),
        f"{label}.scheduler_environment.CUDA_VISIBLE_DEVICES",
    )
    expected_visible = ",".join(str(index) for index in setup.gpu_indices)
    _expect_equal(visible, expected_visible, f"{label}.scheduler_environment.CUDA_VISIBLE_DEVICES")
    leaked_scheduler_knobs = {
        key: value
        for key, value in scheduler.items()
        if key.startswith("VLLM_SCHED_")
        and key != "VLLM_SCHED_POLICY"
        and value is not None
    }
    if leaked_scheduler_knobs:
        raise ValueError(
            f"{label} leaked non-native VLLM scheduler knobs: "
            + ", ".join(sorted(leaked_scheduler_knobs))
        )

    config_exact = {
        "call_graph_mode": "frozen",
        "frozen_url_is_workload_input": True,
        "visit_top_k": 1,
        "search_min_start_interval_s": 0.0,
        "search_max_results": 5,
        "visit_max_chars": 3000,
        "max_tokens_tool": 128,
        "max_tokens_answer": 256,
        "visit_canary_stride": 6,
        "queue_sample_interval_s": 0.2,
        "tool_signal_policy": "execution_aware",
        "tool_signal_policy_version": "exact-session-invocation-running-completed-v1",
        "search_mode": "bing",
        "visit_mode": "jina",
        "controlled_http_retry": True,
        "tool_http_max_attempts": 2,
        "tool_http_retry_backoff_s": 1.0,
        "tool_http_attempt_start_gate_enabled": True,
        "tool_http_attempt_start_gate_policy_version": "shared-per-tool-monotonic-v1",
        "tool_http_library_retry_disabled": True,
        "tool_http_library_retry_control_version": "aiohttp-private-retry-connection-v1",
        "tool_http_library_name": "aiohttp",
        "tool_http_library_version": "3.12.15",
    }
    for key, expected in config_exact.items():
        if isinstance(expected, float):
            _expect_float(config.get(key), expected, f"{label}.config.{key}")
        else:
            _expect_equal(config.get(key), expected, f"{label}.config.{key}")
    attempt_intervals = _mapping(
        config.get("tool_http_attempt_min_start_intervals_s"),
        f"{label}.config.tool_http_attempt_min_start_intervals_s",
    )
    _expect_float(
        attempt_intervals.get("visit"),
        setup.visit_minimum_start_interval_s,
        f"{label}.config.tool_http_attempt_min_start_intervals_s.visit",
    )

    murakkab = _mapping(config.get("murakkab_fixed"), f"{label}.config.murakkab_fixed")
    exact_metadata = {
        "enabled": True,
        "cell_id": "M",
        "evidence_class": "fixed-v9-setup-engineering",
        "implementation_kind": "constrained_murakkab_style_emulation",
        "optimizer_candidate_count": 1,
        "typed_dag_validated": True,
        "dependency_ready_dispatch": True,
        "optimizer_outside_timed_path": True,
        "gpu_count": setup.gpu_count,
        "gpu_type": setup.gpu_type,
        "scheduler": "native_fcfs",
        "tool_execution": "demand_only",
        "official_code_used": False,
        "official_runtime_reproduced": False,
        "runtime_semantics": "A-equivalent",
    }
    for key, expected in exact_metadata.items():
        _expect_equal(murakkab.get(key), expected, f"{label}.murakkab_fixed.{key}")
    for key in ("selected_candidate_id", "workflow_id", "execution_boundary"):
        _nonempty_string(murakkab.get(key), f"{label}.murakkab_fixed.{key}")
    for key in ("plan_sha256", "registry_sha256", "raw_runner_result_sha256"):
        _sha256_text(murakkab.get(key), f"{label}.murakkab_fixed.{key}")

    engineering = _mapping(
        murakkab.get("engineering_run"),
        f"{label}.murakkab_fixed.engineering_run",
    )
    for key in ("fresh_server", "result_cache_empty", "broker_drained"):
        _expect_equal(engineering.get(key), True, f"{label}.engineering_run.{key}")
    _expect_equal(
        engineering.get("evidence_tier"),
        "fixed-v9-setup-engineering",
        f"{label}.engineering_run.evidence_tier",
    )
    _expect_equal(engineering.get("confirmatory_eligible"), False, f"{label}.engineering_run.confirmatory_eligible")
    _expect_equal(engineering.get("performance_comparable"), True, f"{label}.engineering_run.performance_comparable")
    _expect_equal(engineering.get("source_limit"), None, f"{label}.engineering_run.source_limit")
    engineering_background_exact = {
        "performance_comparability_scope": (
            "same fixed model/hardware/workload runtime setup only; this field "
            "does not assert a fresh causal comparison with historical PASTE"
        ),
        "registered_background_policy": setup.background_policy,
        "registered_background_same_identity_before_after": True,
        "user_confirmed_prior_paste_same_condition": True,
        "registered_background_load_intensity_equivalence_claimed": False,
    }
    for key, expected in engineering_background_exact.items():
        _expect_equal(
            engineering.get(key),
            expected,
            f"{label}.engineering_run.{key}",
        )
    run_tag = _nonempty_string(engineering.get("run_tag"), f"{label}.engineering_run.run_tag")
    repetition = _integer(
        engineering.get("repetition"),
        f"{label}.engineering_run.repetition",
        positive=True,
    )
    server_instance_id = _nonempty_string(
        engineering.get("server_instance_id"),
        f"{label}.engineering_run.server_instance_id",
    )
    _nonempty_string(
        engineering.get("assertion_owner"),
        f"{label}.engineering_run.assertion_owner",
    )

    _expect_equal(run.call_graph_mode, "frozen", f"{label}.call_graph_mode")
    _expect_equal(
        config.get("workload_split_id"),
        setup.workload_split_id,
        f"{label}.config.workload_split_id",
    )
    _expect_equal(
        config.get("workload_file_sha256"),
        setup.workload_file_sha256,
        f"{label}.config.workload_file_sha256",
    )
    _expect_equal(
        config.get("selected_workload_sha256"),
        setup.selected_workload_sha256,
        f"{label}.config.selected_workload_sha256",
    )

    sidecars = _validate_run_sidecars(
        run,
        murakkab=murakkab,
        engineering=engineering,
        setup=setup,
        artifact_roots=artifact_roots,
    )

    return {
        "evidence_tier": engineering["evidence_tier"],
        "run_tag": run_tag,
        "repetition": repetition,
        "server_instance_id": server_instance_id,
        "cell_label": _nonempty_string(config.get("cell_label"), f"{label}.config.cell_label"),
        "call_graph_mode": run.call_graph_mode,
        "workload_split_id": config.get("workload_split_id"),
        "selected_workload_sha256": config.get("selected_workload_sha256"),
        "workflow_id": murakkab["workflow_id"],
        "selected_candidate_id": murakkab["selected_candidate_id"],
        "plan_sha256": murakkab["plan_sha256"],
        "registry_sha256": murakkab["registry_sha256"],
        "workflow_sha256": murakkab["workflow_sha256"],
        "raw_runner_result_sha256": murakkab["raw_runner_result_sha256"],
        "gpu_count_verified_from_visible_device_ids": setup.gpu_count,
        "gpu_type_verified_from_hardware_sidecars": setup.gpu_type,
        **sidecars,
    }


def _validate_task_contract(run: ValidatedRun, setup: FixedSetup) -> None:
    for task_id, task in run.tasks_by_id.items():
        _expect_equal(
            task.get("context_padding_target_tokens"),
            setup.context_padding_tokens,
            f"{task_id}.context_padding_target_tokens",
        )
        actual_padding = _integer(
            task.get("context_padding_actual_tokens"),
            f"{task_id}.context_padding_actual_tokens",
            positive=True,
        )
        if not setup.context_padding_tokens <= actual_padding <= setup.context_padding_tokens + 256:
            raise ValueError(f"{task_id} context padding is outside the fixed tolerance")

        final_contract = _mapping(task.get("final_answer_contract"), f"{task_id}.final_answer_contract")
        _expect_equal(final_contract.get("contract_succeeded"), True, f"{task_id}.final_answer_contract.contract_succeeded")
        _expect_equal(final_contract.get("fixed_completion_tokens"), setup.fixed_final_completion_tokens, f"{task_id}.final_answer_contract.fixed_completion_tokens")
        _expect_equal(final_contract.get("total_completion_tokens"), setup.fixed_final_completion_tokens, f"{task_id}.final_answer_contract.total_completion_tokens")

        events = run.llm_by_task[task_id]
        if len(events) != setup.llm_calls_per_task:
            raise ValueError(f"{task_id} does not contain exactly {setup.llm_calls_per_task} LLM calls")
        final_event = events[-1]
        _expect_equal(final_event.get("call_index"), 2, f"{task_id}.final.call_index")
        _expect_equal(final_event.get("min_tokens"), setup.fixed_final_completion_tokens, f"{task_id}.final.min_tokens")
        _expect_equal(final_event.get("max_tokens"), setup.fixed_final_completion_tokens, f"{task_id}.final.max_tokens")
        usage = _mapping(final_event.get("usage"), f"{task_id}.final.usage")
        _expect_equal(usage.get("completion_tokens"), setup.fixed_final_completion_tokens, f"{task_id}.final.completion_tokens")


def _validate_demand_only(run: ValidatedRun, setup: FixedSetup) -> None:
    expected_jobs = 2 * setup.task_count
    if len(run.physical_records) != expected_jobs:
        raise ValueError(
            f"M demand-only must have exactly {expected_jobs} physical jobs; "
            f"observed {len(run.physical_records)}"
        )
    for index, record in enumerate(run.physical_records):
        prefix = f"physical tool record {index}"
        if record.get("speculative") is not False:
            raise ValueError(f"{prefix} contains speculative work")
        if record.get("authoritative") is not True or record.get("committed") is not True:
            raise ValueError(f"{prefix} is not an authoritative commit")
        if record.get("source") != "executed":
            raise ValueError(f"{prefix} was not demand-executed")
        if record.get("exact_match") is not False:
            raise ValueError(f"{prefix} claims speculative exact-match reuse")
        if _finite(record.get("saved_service_s"), f"{prefix}.saved_service_s") != 0.0:
            raise ValueError(f"{prefix} reports speculative saved service")
        if _integer(record.get("http_attempts"), f"{prefix}.http_attempts", positive=True) != 1:
            raise ValueError(f"{prefix} was retried; the fixed engineering run requires zero retries")

    visit_attempt_starts: list[float] = []
    for index, record in enumerate(run.physical_records):
        if record.get("tool") != "visit":
            continue
        attempt_log = record.get("http_attempt_log")
        if not isinstance(attempt_log, list) or len(attempt_log) != 1:
            raise ValueError(f"visit record {index} lacks one-attempt wire-start evidence")
        attempt = _mapping(attempt_log[0], f"visit record {index}.http_attempt_log[0]")
        _expect_equal(attempt.get("attempt"), 1, f"visit record {index}.attempt")
        _expect_equal(attempt.get("status"), 200, f"visit record {index}.status")
        visit_attempt_starts.append(
            _finite(
                attempt.get("started_monotonic_s"),
                f"visit record {index}.started_monotonic_s",
                positive=True,
            )
        )
    visit_attempt_starts.sort()
    tolerance_s = 0.02
    for previous, current in zip(visit_attempt_starts, visit_attempt_starts[1:]):
        if current - previous + tolerance_s < setup.visit_minimum_start_interval_s:
            raise ValueError("physical visit HTTP attempts violate the fixed 2.5-second start gate")

    stats = _mapping(
        _mapping(run.payload.get("broker_final_snapshot"), "broker_final_snapshot").get("stats"),
        "broker_final_snapshot.stats",
    )
    for key in (
        "speculative_admitted",
        "speculative_started",
        "speculative_completed",
        "queued_promotions",
        "running_promotions",
        "completed_reuse",
    ):
        if _integer(stats.get(key, 0), f"broker.stats.{key}") != 0:
            raise ValueError(f"broker.stats.{key} must be zero in the M cell")
    if _finite(stats.get("wasted_speculative_service_s", 0.0), "broker.stats.wasted_speculative_service_s") != 0.0:
        raise ValueError("M cell contains wasted speculative service")


def _task_components(run: ValidatedRun) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for task_id, task in run.tasks_by_id.items():
        llm_s = sum(float(event["duration_s"]) for event in run.llm_by_task[task_id])
        search = run.committed_by_task_tool[(task_id, "search")]
        visit = run.committed_by_task_tool[(task_id, "visit")]
        search_wait_s = float(search["exposed_wait_s"])
        visit_wait_s = float(visit["exposed_wait_s"])
        e2e_s = float(task["e2e_s"])
        rows[task_id] = {
            "e2e_s": e2e_s,
            "llm_s": llm_s,
            "search_exposed_wait_s": search_wait_s,
            "visit_exposed_wait_s": visit_wait_s,
            "tool_exposed_wait_s": search_wait_s + visit_wait_s,
            "unattributed_residual_s": e2e_s - llm_s - search_wait_s - visit_wait_s,
        }
    return rows


def _per_run_metrics(run: ValidatedRun, fixed: Mapping[str, Any]) -> dict[str, Any]:
    tasks = list(run.tasks_by_id.values())
    components = _task_components(run)
    embedded_summary = _mapping(run.payload.get("summary"), "summary")
    experiment_start_s = _finite(embedded_summary.get("started_wall_s"), "summary.started_wall_s", positive=True)
    experiment_end_s = _finite(embedded_summary.get("ended_wall_s"), "summary.ended_wall_s", positive=True)
    if experiment_end_s < experiment_start_s:
        raise ValueError("summary experiment end precedes start")

    completion_offsets = [float(task["end_wall_s"]) - experiment_start_s for task in tasks]
    if min(completion_offsets) < 0.0:
        raise ValueError("a task completed before the experiment start")
    completion_makespan_s = _finite(
        run.payload.get("task_completion_makespan_s"),
        "task_completion_makespan_s",
        positive=True,
    )
    if not math.isclose(max(completion_offsets), completion_makespan_s, rel_tol=0.0, abs_tol=0.25):
        raise ValueError("task_completion_makespan_s disagrees with raw task completion times")
    task_window_s = max(float(task["end_wall_s"]) for task in tasks) - min(
        float(task["start_wall_s"]) for task in tasks
    )
    task_count = len(tasks)
    throughput = task_count / completion_makespan_s
    release_window_throughput = task_count / task_window_s

    physical = list(run.physical_records)
    committed_http_attempts = sum(int(record["http_attempts"]) for record in physical)
    retried_jobs = sum(int(record["http_attempts"]) > 1 for record in physical)
    service_total = sum(float(record["service_s"]) for record in physical)

    by_tool: dict[str, Any] = {}
    for tool_name in ("search", "visit"):
        records = [record for record in physical if record["tool"] == tool_name]
        by_tool[tool_name] = {
            "commit_count": len(records),
            "http_attempt_count": sum(int(record["http_attempts"]) for record in records),
            "exposed_wait_s": _seconds_distribution([float(record["exposed_wait_s"]) for record in records]),
            "queue_s": _seconds_distribution([float(record["queue_s"]) for record in records]),
            "service_s": _seconds_distribution([float(record["service_s"]) for record in records]),
            "worker_service_total_s": sum(float(record["service_s"]) for record in records),
        }

    llm_events = [event for events in run.llm_by_task.values() for event in events]
    vllm = _mapping(run.payload.get("vllm_metric_deltas"), "vllm_metric_deltas")
    llm_by_call = {
        str(call_index): _seconds_distribution(
            [float(event["duration_s"]) for event in llm_events if int(event["call_index"]) == call_index]
        )
        for call_index in range(3)
    }

    component_values = list(components.values())
    component_summary = {
        key: _seconds_distribution([row[key] for row in component_values])
        for key in (
            "llm_s",
            "search_exposed_wait_s",
            "visit_exposed_wait_s",
            "tool_exposed_wait_s",
            "unattributed_residual_s",
        )
    }
    mean_e2e = statistics.fmean(row["e2e_s"] for row in component_values)
    component_summary["mean_fraction_of_e2e"] = {
        "llm": statistics.fmean(row["llm_s"] for row in component_values) / mean_e2e,
        "search_exposed_wait": statistics.fmean(row["search_exposed_wait_s"] for row in component_values) / mean_e2e,
        "visit_exposed_wait": statistics.fmean(row["visit_exposed_wait_s"] for row in component_values) / mean_e2e,
        "unattributed_residual": statistics.fmean(row["unattributed_residual_s"] for row in component_values) / mean_e2e,
    }

    return {
        "input": {
            "result_path": str(run.path),
            "result_sha256": run.sha256,
            **dict(fixed),
        },
        "throughput": {
            "definition": (
                "successful task completions / task_completion_makespan_s; the "
                "runner denominator starts before metrics fetch/client setup and "
                "ends at the last task completion"
            ),
            "successful_task_count": task_count,
            "task_completion_makespan_s": completion_makespan_s,
            "tasks_per_s": throughput,
            "tasks_per_minute": throughput * 60.0,
            "llm_requests_per_s": len(llm_events) / completion_makespan_s,
            "tool_commits_per_s": len(physical) / completion_makespan_s,
            "release_window_tasks_per_s": release_window_throughput,
            "release_window_definition": "task count / (max task end - min task start)",
        },
        "latency": {
            "task_e2e_s": _seconds_distribution([float(task["e2e_s"]) for task in tasks]),
            "source_e2e_s": run.summary["source_e2e_s"],
        },
        "task_completion": {
            "definition": "completion wall time minus experiment start wall time",
            "offset_from_experiment_start_s": _seconds_distribution(completion_offsets),
            "task_window_makespan_s": task_window_s,
            "task_completion_makespan_s": completion_makespan_s,
            "experiment_lifecycle_makespan_s": experiment_end_s - experiment_start_s,
        },
        "decomposition": component_summary,
        "llm": {
            "request_count": len(llm_events),
            "request_duration_s": _seconds_distribution([float(event["duration_s"]) for event in llm_events]),
            "per_task_duration_s": component_summary["llm_s"],
            "request_duration_by_call_index_s": llm_by_call,
            "prompt_tokens": sum(int(_mapping(event["usage"], "usage")["prompt_tokens"]) for event in llm_events),
            "completion_tokens": sum(int(_mapping(event["usage"], "usage")["completion_tokens"]) for event in llm_events),
            "vllm_request_queue_time_total_s": _finite(vllm.get("vllm:request_queue_time_seconds_sum"), "vllm queue time"),
            "vllm_request_inference_time_total_s": (
                _finite(vllm["vllm:request_inference_time_seconds_sum"], "vllm inference time")
                if "vllm:request_inference_time_seconds_sum" in vllm
                else None
            ),
        },
        "tool": {
            "authoritative_commit_count": len(physical),
            "physical_job_count": len(physical),
            "physical_http_attempt_count": committed_http_attempts,
            "retried_physical_job_count": retried_jobs,
            "authoritative_retry_rate": retried_jobs / len(physical),
            "worker_service_total_s": service_total,
            "speculative_job_count": 0,
            "wasted_speculative_service_s": 0.0,
            "by_tool": by_tool,
        },
        "integrity": {
            "valid": True,
            "all_tasks_successful": True,
            "successful_task_count": task_count,
            "failed_task_count": 0,
            "exactly_three_successful_llm_calls_per_task": True,
            "exactly_one_authoritative_search_and_visit_commit_per_task": True,
            "demand_only_zero_speculative_physical_jobs": True,
            "broker_drained": True,
            "raw_timeline_sha256_verified": True,
            "real_http_transport_and_final_200_verified": True,
            "fixed_deployment_contract_verified": True,
            "typed_dag_and_singleton_plan_provenance_verified": True,
        },
        "per_task_components": dict(sorted(components.items())),
    }


def _validate_common_runs(runs: Sequence[ValidatedRun], fixed: Sequence[Mapping[str, Any]]) -> None:
    first = runs[0]
    first_task_keys = set(first.tasks_by_key)
    identity_fields = (
        "selected_workload_sha256",
        "workload_file_sha256",
        "workload_split_id",
        "call_graph_mode",
        "model",
    )
    for run in runs[1:]:
        for key in identity_fields:
            if run.config.get(key) != first.config.get(key):
                raise ValueError(f"M runs differ in common identity field: {key}")
        if set(run.tasks_by_key) != first_task_keys:
            raise ValueError("M runs do not contain the same source/replica identities")

    result_hashes = [run.sha256 for run in runs]
    if len(set(result_hashes)) != len(result_hashes):
        raise ValueError("the same result.json was supplied more than once")
    run_tags = [str(row["run_tag"]) for row in fixed]
    if len(set(run_tags)) != len(run_tags):
        raise ValueError("M runs contain duplicate engineering run tags")
    repetitions = sorted(int(row["repetition"]) for row in fixed)
    if repetitions != list(range(1, len(runs) + 1)):
        raise ValueError("M engineering repetitions must be exactly 1..N")
    server_ids = [str(row["server_instance_id"]) for row in fixed]
    if len(set(server_ids)) != len(server_ids):
        raise ValueError("M runs reused a server instance")
    plan_hashes = {str(row["plan_sha256"]) for row in fixed}
    if len(plan_hashes) != len(runs):
        raise ValueError("M repetitions unexpectedly reuse a run-specific plan SHA")
    for key in (
        "workflow_id",
        "workflow_sha256",
        "registry_sha256",
        "selected_candidate_id",
    ):
        values = {str(row[key]) for row in fixed}
        if len(values) != 1:
            raise ValueError(f"M runs differ in singleton workflow provenance: {key}")
    binding_maps = [row["preflight_bindings"] for row in fixed]
    if any(bindings != binding_maps[0] for bindings in binding_maps[1:]):
        raise ValueError("preflight code/input bindings differ across M repetitions")
    binding_hashes = {str(row["preflight_bindings_sha256"]) for row in fixed}
    if len(binding_hashes) != 1:
        raise ValueError("preflight binding fingerprints differ across M repetitions")
    selected_gpu_uuid_sets = {
        tuple(_mapping(row["registered_background"], "registered_background")["selected_gpu_uuids"])
        for row in fixed
    }
    if len(selected_gpu_uuid_sets) != 1:
        raise ValueError("M runs used different selected GPU UUIDs")


def _validate_host_coload_exclusion(
    manifest_path: Path,
    *,
    setup: FixedSetup,
    artifact_roots: Sequence[Path] = (REPOSITORY_ROOT,),
) -> tuple[ValidatedRun, Mapping[str, Any], dict[str, Any]]:
    """Validate one disclosed post-hoc operational contamination exclusion."""

    manifest = _load_json(manifest_path, "host co-load exclusion manifest")
    _expect_equal(
        manifest.get("schema"),
        "paste_repro.host_coload_observation",
        "host co-load manifest schema",
    )
    _expect_equal(manifest.get("version"), 1, "host co-load manifest version")
    _expect_equal(
        manifest.get("captured_after_run"),
        True,
        "host co-load manifest captured_after_run",
    )
    observation_scope = _nonempty_string(
        manifest.get("observation_scope"),
        "host co-load manifest observation_scope",
    )
    if "not a continuous host-load trace" not in observation_scope:
        raise ValueError(
            "host co-load observation_scope must disclose that evidence is not continuous"
        )

    affected = _mapping(manifest.get("affected_run"), "affected_run")
    normalized_roots = _normalize_artifact_roots(artifact_roots)
    result_sha = _sha256_text(
        affected.get("evidence_result_sha256"),
        "affected_run.evidence_result_sha256",
    )

    def candidates_for(value: Any, label: str) -> list[tuple[Path, Path]]:
        text = _nonempty_string(value, label)
        raw = Path(text)
        candidates: list[tuple[Path, Path]] = []
        if raw.is_absolute():
            resolved = raw.resolve()
            for root in normalized_roots:
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                candidates.append((root, resolved))
        else:
            if ".." in raw.parts or raw == Path("."):
                raise ValueError(f"{label} is not a safe repository-relative path")
            for root in normalized_roots:
                resolved = (root / raw).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"{label} escapes artifact root {root}") from exc
                if resolved.is_file():
                    candidates.append((root, resolved))
        return candidates

    result_candidates = [
        (root, path)
        for root, path in candidates_for(
            affected.get("evidence_result_path"),
            "affected_run.evidence_result_path",
        )
        if path.is_file() and _sha256_file(path) == result_sha
    ]
    if not result_candidates:
        raise ValueError(
            "affected_run.evidence_result_sha256 has no matching result under the "
            "configured artifact roots"
        )

    validated_candidates: list[
        tuple[ValidatedRun, Mapping[str, Any], Path, Path, Path, Path, Path]
    ] = []
    candidate_errors: list[str] = []
    for candidate_root, candidate_result_path in result_candidates:
        try:
            candidate_run = _validate_run(candidate_result_path, role="baseline")
            candidate_fixed = _validate_fixed_config(
                candidate_run,
                setup,
                artifact_roots=normalized_roots,
            )
            _validate_task_contract(candidate_run, setup)
            _validate_demand_only(candidate_run, setup)
            _expect_equal(
                affected.get("run_tag"),
                candidate_fixed["run_tag"],
                "affected_run.run_tag",
            )
            _expect_equal(
                affected.get("repetition"),
                candidate_fixed["repetition"],
                "affected_run.repetition",
            )
            candidate_run_root = Path(str(candidate_fixed["run_root"]))

            def path_for_candidate(value: Any, label: str) -> Path:
                raw = Path(_nonempty_string(value, label))
                resolved = raw.resolve() if raw.is_absolute() else (candidate_root / raw).resolve()
                if not resolved.is_file():
                    raise ValueError(f"{label} is missing for artifact root {candidate_root}")
                return resolved

            candidate_raw = path_for_candidate(
                affected.get("runner_raw_result_path"),
                "affected_run.runner_raw_result_path",
            )
            candidate_completion = path_for_candidate(
                affected.get("completion_manifest_path"),
                "affected_run.completion_manifest_path",
            )
            candidate_before = path_for_candidate(
                affected.get("hardware_before_path"),
                "affected_run.hardware_before_path",
            )
            candidate_after = path_for_candidate(
                affected.get("hardware_after_path"),
                "affected_run.hardware_after_path",
            )
            expected_paths = {
                "evidence result": (
                    candidate_result_path,
                    candidate_run.path.resolve(),
                ),
                "runner raw result": (
                    candidate_raw,
                    (candidate_run_root / "runner_raw/result.json").resolve(),
                ),
                "completion manifest": (
                    candidate_completion,
                    (candidate_run_root / "completed_run.json").resolve(),
                ),
                "hardware before": (
                    candidate_before,
                    (candidate_run_root / "hardware_before.json").resolve(),
                ),
                "hardware after": (
                    candidate_after,
                    (candidate_run_root / "hardware_after.json").resolve(),
                ),
            }
            for label, (observed, expected) in expected_paths.items():
                if observed != expected:
                    raise ValueError(
                        f"affected_run {label} path does not belong to the excluded result"
                    )
            bound_hashes = {
                "runner_raw_result_sha256": (
                    candidate_raw,
                    candidate_fixed["raw_runner_result_sha256"],
                ),
                "completion_manifest_sha256": (
                    candidate_completion,
                    candidate_fixed["completion_manifest_sha256"],
                ),
                "hardware_before_sha256": (
                    candidate_before,
                    candidate_fixed["hardware_before_sha256"],
                ),
                "hardware_after_sha256": (
                    candidate_after,
                    candidate_fixed["hardware_after_sha256"],
                ),
            }
            for key, (path, expected_digest) in bound_hashes.items():
                digest = _sha256_text(affected.get(key), f"affected_run.{key}")
                _expect_equal(digest, expected_digest, f"affected_run.{key}")
                _expect_equal(_sha256_file(path), digest, f"affected_run.{key}")
            validated_candidates.append(
                (
                    candidate_run,
                    candidate_fixed,
                    candidate_result_path,
                    candidate_raw,
                    candidate_completion,
                    candidate_before,
                    candidate_after,
                )
            )
        except (OSError, ValueError) as exc:
            candidate_errors.append(f"{candidate_root}: {exc}")
    if len(validated_candidates) != 1:
        detail = "; ".join(candidate_errors) or "multiple candidates passed"
        raise ValueError(
            "host co-load exclusion must resolve to exactly one fully validated "
            f"artifact root; observed {len(validated_candidates)}: {detail}"
        )
    (
        run,
        fixed,
        result_path,
        raw_result_path,
        completion_path,
        hardware_before_path,
        hardware_after_path,
    ) = validated_candidates[0]

    summary = _mapping(run.payload.get("summary"), "excluded result summary")
    timed_start = _finite(
        summary.get("started_wall_s"),
        "excluded result summary.started_wall_s",
        positive=True,
    )
    timed_end = _finite(
        summary.get("ended_wall_s"),
        "excluded result summary.ended_wall_s",
        positive=True,
    )
    _expect_near(
        affected.get("timed_start_wall_s"),
        timed_start,
        "affected_run.timed_start_wall_s",
    )
    _expect_near(
        affected.get("timed_end_wall_s"),
        timed_end,
        "affected_run.timed_end_wall_s",
    )
    _expect_near(
        _utc_epoch(affected.get("timed_start_utc"), "affected_run.timed_start_utc"),
        timed_start,
        "affected_run.timed_start_utc",
    )
    _expect_near(
        _utc_epoch(affected.get("timed_end_utc"), "affected_run.timed_end_utc"),
        timed_end,
        "affected_run.timed_end_utc",
    )
    captured_at = _utc_epoch(
        manifest.get("captured_at_utc"),
        "host co-load manifest captured_at_utc",
    )
    if captured_at <= timed_end:
        raise ValueError("host co-load evidence was not captured after the affected run")

    hardware_before = _load_json(hardware_before_path, "excluded hardware_before")
    hardware_after = _load_json(hardware_after_path, "excluded hardware_after")
    _expect_near(
        affected.get("hardware_before_query_wall_s"),
        _finite(hardware_before.get("query_wall_s"), "hardware_before.query_wall_s"),
        "affected_run.hardware_before_query_wall_s",
    )
    _expect_near(
        affected.get("hardware_after_query_wall_s"),
        _finite(hardware_after.get("query_wall_s"), "hardware_after.query_wall_s"),
        "affected_run.hardware_after_query_wall_s",
    )
    if _finite(hardware_after.get("query_wall_s"), "hardware_after.query_wall_s") <= timed_end:
        raise ValueError("bound hardware-after observation does not follow the timed run")

    external = _mapping(manifest.get("external_vllm"), "external_vllm")
    _expect_equal(
        external.get("managed_by_this_murakkab_run"),
        False,
        "external_vllm.managed_by_this_murakkab_run",
    )
    _expect_equal(
        external.get("boot_id"),
        fixed["registered_background_boot_id"],
        "external_vllm.boot_id",
    )
    clock_ticks = _integer(
        external.get("clock_ticks_per_second"),
        "external_vllm.clock_ticks_per_second",
        positive=True,
    )
    boot_wall = _finite(
        external.get("kernel_boot_time_wall_s"),
        "external_vllm.kernel_boot_time_wall_s",
        positive=True,
    )
    api_pid = _integer(external.get("api_pid"), "external_vllm.api_pid", positive=True)
    api_ticks = _integer(
        external.get("api_process_starttime_ticks"),
        "external_vllm.api_process_starttime_ticks",
        positive=True,
    )
    api_start = _expect_near(
        external.get("api_process_start_wall_s"),
        boot_wall + api_ticks / clock_ticks,
        "external_vllm.api_process_start_wall_s",
        tolerance=0.02,
    )
    _expect_near(
        _utc_epoch(
            external.get("api_process_start_utc"),
            "external_vllm.api_process_start_utc",
        ),
        api_start,
        "external_vllm.api_process_start_utc",
    )
    if not (timed_start + (timed_end - timed_start) / 2.0 <= api_start < timed_end):
        raise ValueError("external vLLM API did not start in the timed run's second half")
    api_overlap = timed_end - api_start
    _expect_near(
        external.get("api_process_overlap_with_timed_run_s"),
        api_overlap,
        "external_vllm.api_process_overlap_with_timed_run_s",
    )

    engine_pid = _integer(
        external.get("engine_pid"),
        "external_vllm.engine_pid",
        positive=True,
    )
    if engine_pid == api_pid:
        raise ValueError("external vLLM API and engine PIDs must be distinct")
    engine_ticks = _integer(
        external.get("engine_process_starttime_ticks"),
        "external_vllm.engine_process_starttime_ticks",
        positive=True,
    )
    engine_start = _expect_near(
        external.get("engine_process_start_wall_s"),
        boot_wall + engine_ticks / clock_ticks,
        "external_vllm.engine_process_start_wall_s",
        tolerance=0.02,
    )
    if not api_start <= engine_start < timed_end:
        raise ValueError("external vLLM engine start is outside the overlap interval")

    worker_pids_raw = external.get("worker_pids")
    worker_ticks_raw = external.get("worker_process_starttime_ticks")
    if not isinstance(worker_pids_raw, list) or len(worker_pids_raw) != setup.gpu_count:
        raise ValueError("external_vllm.worker_pids must contain four rows")
    if not isinstance(worker_ticks_raw, list) or len(worker_ticks_raw) != setup.gpu_count:
        raise ValueError("external_vllm.worker_process_starttime_ticks must contain four rows")
    worker_pids = [
        _integer(value, f"external_vllm.worker_pids[{index}]", positive=True)
        for index, value in enumerate(worker_pids_raw)
    ]
    worker_ticks = [
        _integer(
            value,
            f"external_vllm.worker_process_starttime_ticks[{index}]",
            positive=True,
        )
        for index, value in enumerate(worker_ticks_raw)
    ]
    if len(set(worker_pids)) != setup.gpu_count:
        raise ValueError("external vLLM worker PIDs are not distinct")
    if api_pid in worker_pids or engine_pid in worker_pids:
        raise ValueError("external vLLM process roles reuse a PID")
    first_worker_start = _expect_near(
        external.get("first_worker_start_wall_s"),
        boot_wall + min(worker_ticks) / clock_ticks,
        "external_vllm.first_worker_start_wall_s",
        tolerance=0.02,
    )
    _expect_near(
        _utc_epoch(
            external.get("first_worker_start_utc"),
            "external_vllm.first_worker_start_utc",
        ),
        first_worker_start,
        "external_vllm.first_worker_start_utc",
    )
    if not engine_start <= first_worker_start < timed_end:
        raise ValueError("external vLLM workers did not start during the timed run")
    worker_overlap = timed_end - first_worker_start
    _expect_near(
        external.get("worker_overlap_with_timed_run_s"),
        worker_overlap,
        "external_vllm.worker_overlap_with_timed_run_s",
    )

    argv_raw = external.get("argv")
    if not isinstance(argv_raw, list) or not all(
        isinstance(value, str) and value for value in argv_raw
    ):
        raise ValueError("external_vllm.argv must be a non-empty string array")
    argv = list(argv_raw)

    def argv_value(flag: str) -> str:
        if argv.count(flag) != 1:
            raise ValueError(f"external_vllm.argv must contain exactly one {flag}")
        index = argv.index(flag)
        if index + 1 >= len(argv):
            raise ValueError(f"external_vllm.argv lacks a value for {flag}")
        return argv[index + 1]

    if "vllm.entrypoints.openai.api_server" not in argv:
        raise ValueError("external process is not a vLLM OpenAI API server")
    _expect_equal(argv_value("--port"), "8200", "external_vllm.argv --port")
    _expect_equal(
        argv_value("--served-model-name"),
        setup.model,
        "external_vllm.argv --served-model-name",
    )
    _expect_equal(
        argv_value("--tensor-parallel-size"),
        str(setup.tensor_parallelism),
        "external_vllm.argv --tensor-parallel-size",
    )
    _expect_equal(
        external.get("listen_endpoint"),
        "127.0.0.1:8200",
        "external_vllm.listen_endpoint",
    )
    _nonempty_string(external.get("executable"), "external_vllm.executable")
    _nonempty_string(external.get("cwd"), "external_vllm.cwd")
    for path_key, sha_key in (
        ("state_pid_path", "state_pid_sha256_at_capture"),
        ("state_policy_path", "state_policy_sha256_at_capture"),
        ("server_log_path", "server_log_sha256_at_capture"),
    ):
        _nonempty_string(external.get(path_key), f"external_vllm.{path_key}")
        _sha256_text(external.get(sha_key), f"external_vllm.{sha_key}")
    _integer(
        external.get("server_log_size_bytes_at_capture"),
        "external_vllm.server_log_size_bytes_at_capture",
        positive=True,
    )
    log_mtime = _utc_epoch(
        external.get("server_log_last_modified_at_capture_utc"),
        "external_vllm.server_log_last_modified_at_capture_utc",
    )
    if log_mtime > captured_at:
        raise ValueError("external server log mtime follows the evidence capture")
    ready_at = _utc_epoch(
        external.get("server_became_ready_utc_from_log"),
        "external_vllm.server_became_ready_utc_from_log",
    )
    _expect_equal(
        external.get("server_became_ready_after_affected_run_ended"),
        True,
        "external_vllm.server_became_ready_after_affected_run_ended",
    )
    if ready_at <= timed_end:
        raise ValueError("external vLLM ready timestamp does not follow the affected run")

    selected_uuids = set(
        _mapping(fixed["registered_background"], "registered_background")[
            "selected_gpu_uuids"
        ]
    )

    def validate_external_gpu_rows(value: Any, label: str) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) != setup.gpu_count:
            raise ValueError(f"{label} must contain four rows")
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(value):
            row = _mapping(raw, f"{label}[{index}]")
            gpu_index = _integer(row.get("gpu_index"), f"{label}[{index}].gpu_index")
            gpu_uuid = _nonempty_string(row.get("gpu_uuid"), f"{label}[{index}].gpu_uuid")
            pid = _integer(row.get("pid"), f"{label}[{index}].pid", positive=True)
            memory = _finite(
                row.get("used_memory_mib"),
                f"{label}[{index}].used_memory_mib",
                positive=True,
            )
            rows.append(
                {
                    "gpu_index": gpu_index,
                    "gpu_uuid": gpu_uuid,
                    "pid": pid,
                    "used_memory_mib": memory,
                }
            )
        if {row["gpu_index"] for row in rows} != {0, 1, 2, 3}:
            raise ValueError(f"{label} does not bind external GPUs 0,1,2,3")
        if len({row["gpu_uuid"] for row in rows}) != setup.gpu_count:
            raise ValueError(f"{label} does not contain four distinct GPU UUIDs")
        if {row["gpu_uuid"] for row in rows} & selected_uuids:
            raise ValueError(f"{label} overlaps the M run's selected GPUs")
        if {row["pid"] for row in rows} != set(worker_pids):
            raise ValueError(f"{label} does not bind the external worker PIDs")
        return rows

    bound_after_rows = validate_external_gpu_rows(
        manifest.get("external_gpu_compute_applications_in_bound_run_after_snapshot"),
        "external_gpu_compute_applications_in_bound_run_after_snapshot",
    )
    validate_external_gpu_rows(
        manifest.get("gpu_compute_applications_at_capture"),
        "gpu_compute_applications_at_capture",
    )
    raw_after_apps = hardware_after.get("all_compute_applications")
    if not isinstance(raw_after_apps, list):
        raise ValueError("hardware_after.all_compute_applications must be an array")
    for row in bound_after_rows:
        matches = [
            raw
            for raw in raw_after_apps
            if isinstance(raw, Mapping)
            and raw.get("gpu_uuid") == row["gpu_uuid"]
            and raw.get("pid") == row["pid"]
            and float(raw.get("used_memory_mib", -1.0)) == row["used_memory_mib"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "bound hardware-after snapshot does not contain an external worker row"
            )
    before_unregistered = affected.get("hardware_before_unregistered_compute_apps")
    if before_unregistered != []:
        raise ValueError(
            "affected_run.hardware_before_unregistered_compute_apps must be empty"
        )
    registered_pid = int(fixed["registered_background_pid"])
    raw_before_apps = hardware_before.get("all_compute_applications")
    if not isinstance(raw_before_apps, list) or any(
        not isinstance(row, Mapping) or row.get("pid") != registered_pid
        for row in raw_before_apps
    ):
        raise ValueError("hardware-before snapshot contains an unregistered compute app")

    selected_capture = _mapping(
        manifest.get("selected_gpu_observation_at_capture"),
        "selected_gpu_observation_at_capture",
    )
    _expect_equal(
        selected_capture.get("indices"),
        list(setup.gpu_indices),
        "selected_gpu_observation_at_capture.indices",
    )
    _expect_equal(
        selected_capture.get("additional_compute_apps_beyond_registered_resnet"),
        False,
        "selected_gpu_observation_at_capture.additional_compute_apps_beyond_registered_resnet",
    )
    _expect_equal(
        selected_capture.get("registered_resnet_pid"),
        registered_pid,
        "selected_gpu_observation_at_capture.registered_resnet_pid",
    )

    interpretation = _mapping(manifest.get("interpretation"), "interpretation")
    interpretation_exact = {
        "functional_integrity_invalidated": False,
        "performance_cleanliness_invalidated": True,
        "exclusion_based_on_performance_value": False,
        "performance_values_inspected_before_exclusion_decision": True,
        "decision_characterization": (
            "Post-run operational contamination exclusion, not a "
            "performance-threshold exclusion."
        ),
    }
    for key, expected in interpretation_exact.items():
        _expect_equal(interpretation.get(key), expected, f"interpretation.{key}")
    reason = _nonempty_string(interpretation.get("reason"), "interpretation.reason")

    metrics = _per_run_metrics(run, fixed)
    return run, fixed, {
        "classification": "host_co_load_contaminated",
        "disposition": "excluded_from_primary_supplementary_only",
        "post_hoc_operational_exclusion": True,
        "exclusion_based_on_performance_value": False,
        "performance_values_inspected_before_exclusion_decision": True,
        "functional_integrity_invalidated": False,
        "performance_cleanliness_invalidated": True,
        "reason": reason,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "result_path": str(result_path),
        "result_sha256": result_sha,
        "run_tag": fixed["run_tag"],
        "repetition": fixed["repetition"],
        "timestamp_evidence": {
            "timed_start_wall_s": timed_start,
            "timed_end_wall_s": timed_end,
            "captured_at_utc": manifest["captured_at_utc"],
            "external_api_pid": api_pid,
            "external_api_start_wall_s": api_start,
            "external_api_overlap_s": api_overlap,
            "external_first_worker_start_wall_s": first_worker_start,
            "external_worker_overlap_s": worker_overlap,
            "external_worker_pids": worker_pids,
            "bound_hardware_after_sha256": fixed["hardware_after_sha256"],
        },
        "performance": metrics,
    }


def aggregate_murakkab_fixed_results(
    result_paths: Sequence[Path],
    *,
    setup: FixedSetup = FIXED_SETUP,
    exclusion_manifest_paths: Sequence[Path] = (),
    artifact_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Validate three primary M runs plus disclosed supplementary exclusions."""

    if not result_paths:
        raise ValueError("at least one M result is required")
    if len(result_paths) != setup.repetitions:
        raise ValueError(
            f"the fixed engineering protocol requires exactly {setup.repetitions} "
            f"fresh-server repetitions; observed {len(result_paths)}"
        )
    normalized_artifact_roots = _normalize_artifact_roots(artifact_roots)
    protocol_sha = _validate_protocol(setup)
    runs: list[ValidatedRun] = []
    fixed_rows: list[Mapping[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    for index, path in enumerate(result_paths):
        run = _validate_run(path, role="baseline")
        fixed = _validate_fixed_config(
            run,
            setup,
            artifact_roots=normalized_artifact_roots,
        )
        _validate_task_contract(run, setup)
        _validate_demand_only(run, setup)
        runs.append(run)
        fixed_rows.append(fixed)
        per_run.append(_per_run_metrics(run, fixed))
    _validate_common_runs(runs, fixed_rows)

    manifest_resolved = [Path(path).resolve() for path in exclusion_manifest_paths]
    if len(set(manifest_resolved)) != len(manifest_resolved):
        raise ValueError("the same exclusion manifest was supplied more than once")
    excluded_runs: list[ValidatedRun] = []
    excluded_fixed: list[Mapping[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    for path in manifest_resolved:
        excluded_run, fixed, record = _validate_host_coload_exclusion(
            path,
            setup=setup,
            artifact_roots=normalized_artifact_roots,
        )
        excluded_runs.append(excluded_run)
        excluded_fixed.append(fixed)
        excluded_records.append(record)

    primary_hashes = {run.sha256 for run in runs}
    primary_tags = {str(row["run_tag"]) for row in fixed_rows}
    primary_server_ids = {str(row["server_instance_id"]) for row in fixed_rows}
    excluded_hashes = [run.sha256 for run in excluded_runs]
    excluded_tags = [str(row["run_tag"]) for row in excluded_fixed]
    if len(set(excluded_hashes)) != len(excluded_hashes):
        raise ValueError("the same contaminated result was disclosed more than once")
    if len(set(excluded_tags)) != len(excluded_tags):
        raise ValueError("contamination manifests contain duplicate run tags")
    if primary_hashes & set(excluded_hashes):
        raise ValueError("a disclosed contaminated result cannot appear in the primary aggregate")
    if primary_tags & set(excluded_tags):
        raise ValueError("a disclosed contaminated run tag cannot appear in the primary aggregate")
    if primary_server_ids & {
        str(row["server_instance_id"]) for row in excluded_fixed
    }:
        raise ValueError("a disclosed contaminated attempt reused a primary server identity")
    primary_task_keys = set(runs[0].tasks_by_key)
    compatibility_keys = (
        "workflow_id",
        "workflow_sha256",
        "registry_sha256",
        "selected_candidate_id",
        "preflight_bindings",
        "selected_workload_sha256",
    )
    primary_identity = fixed_rows[0]
    for excluded_run, fixed in zip(excluded_runs, excluded_fixed, strict=True):
        if not 1 <= int(fixed["repetition"]) <= setup.repetitions:
            raise ValueError("excluded repetition is outside the planned repetition range")
        if set(excluded_run.tasks_by_key) != primary_task_keys:
            raise ValueError("excluded run does not use the primary source identities")
        for key in compatibility_keys:
            if fixed[key] != primary_identity[key]:
                raise ValueError(f"excluded run differs from the primary setup: {key}")
        excluded_uuids = _mapping(
            fixed["registered_background"],
            "excluded registered_background",
        )["selected_gpu_uuids"]
        primary_uuids = _mapping(
            primary_identity["registered_background"],
            "primary registered_background",
        )["selected_gpu_uuids"]
        if excluded_uuids != primary_uuids:
            raise ValueError("excluded run used different selected GPU UUIDs")

    source_values: dict[str, list[float]] = defaultdict(list)
    all_e2e: list[float] = []
    all_components: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for (source_id, _replica), task in run.tasks_by_key.items():
            source_values[source_id].append(float(task["e2e_s"]))
            all_e2e.append(float(task["e2e_s"]))
        for row in _task_components(run).values():
            for key, value in row.items():
                if key != "e2e_s":
                    all_components[key].append(value)
    source_means = {
        source_id: statistics.fmean(values)
        for source_id, values in sorted(source_values.items())
    }

    completion_makespans = [float(row["throughput"]["task_completion_makespan_s"]) for row in per_run]
    throughputs = [float(row["throughput"]["tasks_per_s"]) for row in per_run]
    release_window_throughputs = [
        float(row["throughput"]["release_window_tasks_per_s"]) for row in per_run
    ]
    llm_throughputs = [float(row["throughput"]["llm_requests_per_s"]) for row in per_run]
    tool_throughputs = [float(row["throughput"]["tool_commits_per_s"]) for row in per_run]
    e2e_metric_names = ("mean_s", "p50_s", "p95_s", "p99_s", "max_s")
    per_run_e2e = {
        metric: [float(row["latency"]["task_e2e_s"][metric]) for row in per_run]
        for metric in e2e_metric_names
    }
    all_completion_offsets = [
        float(task["end_wall_s"])
        - float(_mapping(run.payload["summary"], "summary")["started_wall_s"])
        for run in runs
        for task in run.tasks_by_id.values()
    ]
    total_success = sum(int(row["throughput"]["successful_task_count"]) for row in per_run)
    total_runtime = sum(completion_makespans)
    total_release_window = sum(
        float(row["task_completion"]["task_window_makespan_s"])
        for row in per_run
    )

    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "treatment": {
            "cell_id": "M",
            "label": "constrained Murakkab-style emulation (M)",
            "llm_scheduler": "native FCFS",
            "tool_execution": "demand only",
            "optimizer_candidate_count": 1,
            "comparison_effect_estimated": False,
            "evidence_class": "fixed-v9-setup-engineering",
            "official_code_used": False,
            "official_runtime_reproduced": False,
            "runtime_semantics": "A-equivalent",
        },
        "fixed_setup": asdict(setup),
        "provenance": {
            "protocol_path": str(PROTOCOL_PATH),
            "protocol_sha256": protocol_sha,
            "aggregator_path": str(Path(__file__).resolve()),
            "aggregator_sha256": _sha256_file(Path(__file__).resolve()),
            "artifact_roots": [str(root) for root in normalized_artifact_roots],
            "raw_result_sha256s": [run.sha256 for run in runs],
            "exclusion_manifest_sha256s": [
                record["manifest_sha256"] for record in excluded_records
            ],
        },
        "run_count": len(runs),
        "per_run": per_run,
        "attempt_accounting": {
            "planned_primary_clean_repetitions": setup.repetitions,
            "primary_clean_result_count": len(runs),
            "supplementary_operationally_excluded_count": len(excluded_records),
            "total_disclosed_completed_attempt_count": len(runs) + len(excluded_records),
            "excluded_results_in_primary_aggregate": 0,
            "post_hoc_exclusion_rule": (
                "external host co-load timestamp overlap; never a performance-value threshold"
            ),
        },
        "supplementary": {
            "included_in_primary_aggregate": False,
            "operationally_excluded_contaminated_runs": excluded_records,
        },
        "aggregate": {
            "statistical_unit": "source",
            "independent_source_count": len(source_means),
            "source_observations_per_source": len(runs),
            "throughput": {
                "definition": "sum(successful task completions) / sum(task_completion_makespan_s)",
                "successful_task_count": total_success,
                "observed_runtime_s": total_runtime,
                "time_weighted_tasks_per_s": total_success / total_runtime,
                "time_weighted_tasks_per_minute": 60.0 * total_success / total_runtime,
                "time_weighted_llm_requests_per_s": sum(int(row["llm"]["request_count"]) for row in per_run) / total_runtime,
                "time_weighted_tool_commits_per_s": sum(int(row["tool"]["authoritative_commit_count"]) for row in per_run) / total_runtime,
                "time_weighted_release_window_tasks_per_s": total_success / total_release_window,
                "per_run_tasks_per_s": _rate_distribution(throughputs),
                "per_run_release_window_tasks_per_s": _rate_distribution(release_window_throughputs),
            },
            "latency": {
                "source_mean_e2e_across_runs_s": {
                    "by_source": source_means,
                    "distribution": _seconds_distribution(list(source_means.values())),
                },
                "pooled_task_e2e_s": _seconds_distribution(all_e2e),
            },
            "task_completion": {
                "per_run_task_completion_makespan_s": _seconds_distribution(completion_makespans),
                "pooled_completion_offset_from_experiment_start_s": _seconds_distribution(all_completion_offsets),
            },
            "across_repetitions": {
                "descriptive_only_no_significance_test": True,
                "tasks_per_s": _repetition_summary(throughputs),
                "release_window_tasks_per_s": _repetition_summary(release_window_throughputs),
                "llm_requests_per_s": _repetition_summary(llm_throughputs),
                "tool_commits_per_s": _repetition_summary(tool_throughputs),
                "task_completion_makespan_s": _repetition_summary(completion_makespans),
                "task_e2e_s": {
                    metric: _repetition_summary(values)
                    for metric, values in per_run_e2e.items()
                },
            },
            "decomposition_pooled": {
                key: _seconds_distribution(values)
                for key, values in sorted(all_components.items())
            },
            "llm": {
                "request_count": sum(int(row["llm"]["request_count"]) for row in per_run),
                "prompt_tokens": sum(int(row["llm"]["prompt_tokens"]) for row in per_run),
                "completion_tokens": sum(int(row["llm"]["completion_tokens"]) for row in per_run),
            },
            "tool": {
                "authoritative_commit_count": sum(int(row["tool"]["authoritative_commit_count"]) for row in per_run),
                "physical_http_attempt_count": sum(int(row["tool"]["physical_http_attempt_count"]) for row in per_run),
                "retried_physical_job_count": sum(int(row["tool"]["retried_physical_job_count"]) for row in per_run),
                "worker_service_total_s": sum(float(row["tool"]["worker_service_total_s"]) for row in per_run),
                "speculative_job_count": 0,
                "wasted_speculative_service_s": 0.0,
            },
            "integrity": {
                "all_runs_valid": True,
                "all_tasks_successful": True,
                "failed_task_count": 0,
                "zero_observed_http_retries": True,
                "fixed_deployment_contract_verified_every_run": True,
                "fresh_distinct_server_per_run": True,
                "same_workload_and_source_identities_every_run": True,
                "same_preflight_code_and_input_bindings_every_run": True,
                "same_singleton_workflow_registry_and_candidate_every_run": True,
                "distinct_bound_run_plan_every_run": True,
                "demand_only_zero_speculation_every_run": True,
                "registered_resnet_coload_endpoint_identity_verified_every_run": True,
                "one_registered_resnet_application_per_selected_gpu_at_each_endpoint": True,
                "registered_resnet_code_identity_fixed_across_repetitions": True,
                "continuous_background_monitoring_performed": False,
                "background_load_intensity_equivalence_claimed": False,
                "disclosed_host_coload_contaminated_results_excluded_from_primary": len(
                    excluded_records
                ),
            },
        },
        "claim_boundary": {
            "descriptive_m_only_result": True,
            "evidence_class": "fixed-v9-setup-engineering; not a formal-v9 matrix result",
            "system_identity": "constrained Murakkab-style emulation; not the official Murakkab runtime",
            "paste_treatment_effect_or_speedup": "not estimated",
            "gpu_count_saving": "not identifiable; every run provisions four GPUs",
            "gpu_type_evidence": "nvidia-smi hardware snapshots bind four distinct A100 UUIDs before and after each run",
            "measurement_condition": (
                "absolute system-level M metrics observed with the registered ResNet "
                "co-load present on GPUs 4,5,6,7 at both run endpoints"
            ),
            "registered_background_evidence": (
                "within each repetition, before/after snapshots bind the same ResNet "
                "PID, process start time, boot ID, executable, cwd, argv, script SHA, "
                "and one positive-memory application row per selected GPU"
            ),
            "registered_background_monitoring_limit": (
                "endpoint snapshots only; the background process and load intensity "
                "were not continuously monitored during the timed run"
            ),
            "historical_paste_context": (
                "the user confirms prior PASTE used the same registered ResNet setup; "
                "this is retrospective context, not a fresh paired comparison"
            ),
            "historical_background_load_intensity_equivalence": (
                "not established; matching process/code identity and positive GPU memory "
                "do not prove equal utilization, power, or training intensity"
            ),
            "isolated_qwen_capacity": "not estimated because the registered ResNet co-load remained active",
            "contamination_exclusion": (
                "disclosed external-host co-load attempts are shown only as supplementary "
                "results and are excluded post hoc by timestamp overlap, not by throughput "
                "or latency thresholds"
            ),
            "performance_values_inspected_before_known_exclusion": any(
                bool(record["performance_values_inspected_before_exclusion_decision"])
                for record in excluded_records
            ),
            "energy": "not measured",
            "cloud_cost": "not measured",
            "throughput_denominator": (
                "runner experiment start (before metrics fetch/client setup) to "
                "last task completion; release-window throughput is separate"
            ),
            "control_plane_timing": (
                "singleton planning/type checking ran outside the timed path; "
                "control-plane planning overhead was not measured"
            ),
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    aggregate = _mapping(result["aggregate"], "aggregate")
    throughput = _mapping(aggregate["throughput"], "throughput")
    latency = _mapping(
        _mapping(aggregate["latency"], "latency")["source_mean_e2e_across_runs_s"],
        "source latency",
    )["distribution"]
    latency = _mapping(latency, "source latency distribution")
    completion = _mapping(aggregate["task_completion"], "task completion")["per_run_task_completion_makespan_s"]
    completion = _mapping(completion, "completion distribution")
    tool = _mapping(aggregate["tool"], "tool")
    decomposition = _mapping(aggregate["decomposition_pooled"], "decomposition")
    supplementary = _mapping(result["supplementary"], "supplementary")
    excluded = supplementary["operationally_excluded_contaminated_runs"]
    if not isinstance(excluded, list):
        raise ValueError("supplementary contaminated runs must be an array")
    provenance = _mapping(result["provenance"], "provenance")
    artifact_roots = provenance.get("artifact_roots")
    if not isinstance(artifact_roots, list) or not artifact_roots:
        raise ValueError("provenance.artifact_roots must be a non-empty array")
    lines = [
        "# Constrained Murakkab-style emulation: M-only live result",
        "",
        "Evidence class: `fixed-v9-setup-engineering` (not a formal-v9 matrix result).",
        f"Validated repetitions: {result['run_count']}; independent sources: {aggregate['independent_source_count']}.",
        "Artifact roots used only for completion-manifest key mapping: "
        + ", ".join(f"`{root}`" for root in artifact_roots)
        + ".",
        "These are absolute system-level measurements with the registered ResNet "
        "co-load present on GPUs 4,5,6,7 at both endpoint snapshots; they are not "
        "isolated-Qwen capacity measurements.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Runner-window throughput | {float(throughput['time_weighted_tasks_per_s']):.6f} tasks/s |",
        f"| Release-window throughput | {float(throughput['time_weighted_release_window_tasks_per_s']):.6f} tasks/s |",
        f"| LLM throughput | {float(throughput['time_weighted_llm_requests_per_s']):.6f} requests/s |",
        f"| Tool throughput | {float(throughput['time_weighted_tool_commits_per_s']):.6f} commits/s |",
        f"| Source-mean E2E | {float(latency['mean_s']):.3f} s |",
        f"| Source E2E p50 | {float(latency['p50_s']):.3f} s |",
        f"| Source E2E p95 | {float(latency['p95_s']):.3f} s |",
        f"| Source E2E p99 | {float(latency['p99_s']):.3f} s |",
        f"| Per-run task-completion makespan mean | {float(completion['mean_s']):.3f} s |",
        f"| Physical HTTP attempts | {tool['physical_http_attempt_count']} |",
        f"| Tool worker service | {float(tool['worker_service_total_s']):.3f} s |",
        "| Successful tasks | 100% |",
        "| Speculative jobs | 0 |",
        "",
        "## Per-repetition measurements",
        "",
        "| Rep | Tasks/s | Release tasks/s | LLM req/s | Tool commits/s | Makespan (s) | E2E mean | p50 | p95 | p99 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_run"]:
        per_run = _mapping(row, "per_run")
        run_input = _mapping(per_run["input"], "per_run.input")
        rates = _mapping(per_run["throughput"], "per_run.throughput")
        e2e = _mapping(_mapping(per_run["latency"], "per_run.latency")["task_e2e_s"], "per_run.e2e")
        lines.append(
            f"| {run_input['repetition']} | {float(rates['tasks_per_s']):.6f} | "
            f"{float(rates['release_window_tasks_per_s']):.6f} | "
            f"{float(rates['llm_requests_per_s']):.6f} | "
            f"{float(rates['tool_commits_per_s']):.6f} | "
            f"{float(rates['task_completion_makespan_s']):.3f} | "
            f"{float(e2e['mean_s']):.3f} | {float(e2e['p50_s']):.3f} | "
            f"{float(e2e['p95_s']):.3f} | {float(e2e['p99_s']):.3f} | "
            f"{float(e2e['max_s']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pooled task-level decomposition",
            "",
            "| Component | Mean (s) | p50 | p95 | p99 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("llm_s", "LLM request time per task"),
        ("search_exposed_wait_s", "Search exposed wait"),
        ("visit_exposed_wait_s", "Visit exposed wait"),
        ("unattributed_residual_s", "Unattributed residual"),
    ):
        values = _mapping(decomposition[key], f"decomposition.{key}")
        lines.append(
            f"| {label} | {float(values['mean_s']):.3f} | "
            f"{float(values['p50_s']):.3f} | {float(values['p95_s']):.3f} | "
            f"{float(values['p99_s']):.3f} |"
        )
    if excluded:
        lines.extend(
            [
                "",
                "## Supplementary operationally excluded attempts",
                "",
                "These measurements are disclosed for transparency but are not included "
                "in any primary aggregate above.",
                "",
                "| Run | Rep | Classification | Tasks/s | E2E mean (s) | API overlap (s) | Worker overlap (s) |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for raw in excluded:
            row = _mapping(raw, "supplementary contaminated run")
            performance = _mapping(row["performance"], "supplementary performance")
            rates = _mapping(performance["throughput"], "supplementary throughput")
            e2e = _mapping(
                _mapping(performance["latency"], "supplementary latency")["task_e2e_s"],
                "supplementary E2E",
            )
            timestamps = _mapping(row["timestamp_evidence"], "timestamp evidence")
            lines.append(
                f"| {row['run_tag']} | {row['repetition']} | `{row['classification']}` | "
                f"{float(rates['tasks_per_s']):.6f} | {float(e2e['mean_s']):.3f} | "
                f"{float(timestamps['external_api_overlap_s']):.3f} | "
                f"{float(timestamps['external_worker_overlap_s']):.3f} |"
            )
        lines.extend(
            [
                "",
                "The exclusion is a post-run operational decision based on independently "
                "validated external-host vLLM start/worker timestamps overlapping the timed "
                "window. Performance values had already been inspected, which is disclosed; "
                "the rule did not use a throughput or latency threshold.",
            ]
        )
    lines.extend(
        [
            "",
            "All values are recomputed from raw task, LLM, physical-tool, hardware, "
            "and bound sidecar evidence. Across three repetitions, mean/median/range "
            "are descriptive only; no significance test is claimed.",
            "",
            "This is a constrained Murakkab-style emulation with A-equivalent runtime "
            "semantics, not official Murakkab code or runtime. Singleton planning ran "
            "outside the timed path, and its overhead was not measured. This M-only "
            "run does not estimate a PASTE speedup or GPU, energy, or cost saving.",
            "",
            "For each repetition, the before/after snapshots verify the same registered "
            "ResNet PID, process start time, boot ID, executable, argv, working directory, "
            "script SHA, and one positive-memory application row on each selected GPU. "
            "No continuous in-run background monitor was added, so this establishes "
            "endpoint process/code identity—not constant or historically equivalent "
            "utilization, power, or training intensity. The user reports that historical "
            "PASTE used the same registered ResNet setup; that remains retrospective "
            "context rather than a fresh causal comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        action="append",
        required=True,
        help="M-cell evidence/result.json; repeat for fresh-server blocks",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--exclusion-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "post-hoc host co-load observation for a contaminated completed attempt; "
            "repeat as needed (supplementary only)"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "absolute repository root used only to map run artifact paths to "
            "repository-relative completion-manifest keys; repeat for mixed worktrees"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = aggregate_murakkab_fixed_results(
            args.result,
            exclusion_manifest_paths=args.exclusion_manifest,
            artifact_roots=args.artifact_root,
        )
        _write_json_atomic(args.output, result)
        if args.report is not None:
            report = args.report.resolve()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(render_markdown(result), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "aggregate": result["aggregate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
