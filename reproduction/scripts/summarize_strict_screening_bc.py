#!/usr/bin/env python3
"""Strictly compare Joint native-admission B with physical-KV-admission C.

Both inputs are read-only completed stress cells.  They must use the same
learned workload, Joint ordering policy, engine shape, calibration artifact,
and mapper artifact.  The scheduler-environment difference is fixed to the
pre-registered admission-control keys below, with exact two-sided values (or
explicit absence) supplied by the caller.  Physical-KV telemetry is checked
again rather than trusting its saved ``screening_gates.passed`` flag alone.

The derived JSON is written atomically as a direct child of the C run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUNNER_DIRECTORY = REPOSITORY_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY, RUNNER_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from run_vllm_trace_experiment import (  # noqa: E402
    PHYSICAL_KV_LOG_PARSER_ID,
    PHYSICAL_KV_LOG_PARSER_VERSION,
    parse_vllm_log_segment,
)
from summarize_candidate_d import _cell_metrics  # noqa: E402
from summarize_four_cell import load_fixed_manifest, load_run  # noqa: E402
from summarize_paired_ad import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    TIE_EPSILON_S,
    _bootstrap_mean_ci,
    _load_raw_execution_accounting,
    _task_flow_by_trace,
    _validate_source_multiplicity,
)
from summarize_strict_screening_ad import (  # noqa: E402
    DEFAULT_ENGINE_KEYS,
    _engine_shape_guard,
    _verify_frozen_config,
)


SCHEMA = "paste_repro.strict_screening_bc"
VERSION = 2
EXPECTED_POLICY = "online_joint_pacer_v2"
EXPECTED_OVERLAP = "learned"
PHYSICAL_MARKER = "[sched_policy_patch:physical_kv]"
PHYSICAL_REVALIDATION_FILENAME = "physical_kv_revalidation.json"
PHYSICAL_REVALIDATION_SCHEMA = "paste_repro.physical_kv_raw_log_revalidation"
PHYSICAL_REVALIDATION_VERSION = 1
PHYSICAL_REVALIDATION_STATUS = "accepted_raw_log_revalidation"
PHYSICAL_PARSER_MODULE = "scripts/run_vllm_trace_experiment.py"
PHYSICAL_VALIDATOR_MODULE = (
    "reproduction/scripts/validate_physical_kv_admission.py"
)
LEGACY_REJECTION_REASON = (
    "nonrescue_committed_plus_predicted_exceeds_soft_budget"
)
LEGACY_REJECTION_SHAPE = (
    "decision=admit reason=forecast_hold rescue=0 admit=0 fit_admit=0 "
    "predicted_admit_tokens=0"
)
ALLOWED_CONFIG_DIFFERENCES = frozenset(
    {
        "PASTE_FROZEN_CONFIG_SHA256",
        "PASTE_STRESS_PROFILE",
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
    }
)
PHYSICAL_CONFIG_KEYS = frozenset(
    key
    for key in ALLOWED_CONFIG_DIFFERENCES
    if key.startswith("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_")
)
_MISSING = object()


def _parse_key_value(items: Sequence[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"{option} requires KEY=VALUE, got {item!r}")
        if key in result:
            raise ValueError(f"{option} repeats key {key}")
        result[key] = value
    return result


def _exact_config_guard(
    b_config: Mapping[str, Any],
    c_config: Mapping[str, Any],
    *,
    expected_b: Mapping[str, str],
    expected_c: Mapping[str, str],
    expected_b_missing: set[str],
    expected_c_missing: set[str],
) -> dict[str, Any]:
    allowed = set(ALLOWED_CONFIG_DIFFERENCES)
    if set(expected_b) & expected_b_missing or set(expected_c) & expected_c_missing:
        raise ValueError("a config key cannot be both expected present and missing")
    if (set(expected_b) | expected_b_missing) != allowed:
        raise ValueError("every allowed config difference needs an exact B expectation")
    if (set(expected_c) | expected_c_missing) != allowed:
        raise ValueError("every allowed config difference needs an exact C expectation")

    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(b_config) | set(c_config)):
        b_value = b_config.get(key, _MISSING)
        c_value = c_config.get(key, _MISSING)
        if b_value == c_value:
            continue
        differences[key] = {
            "b_present": b_value is not _MISSING,
            "b_value": None if b_value is _MISSING else b_value,
            "c_present": c_value is not _MISSING,
            "c_value": None if c_value is _MISSING else c_value,
        }
    actual = set(differences)
    if actual != allowed:
        raise ValueError(
            "B/C scheduler configuration diff does not exactly match the "
            "physical-admission allowlist; "
            f"unexpected={sorted(actual - allowed)}, "
            f"unused={sorted(allowed - actual)}"
        )

    def validate_side(
        config: Mapping[str, Any],
        expected: Mapping[str, str],
        missing: set[str],
        label: str,
    ) -> None:
        for key, value in expected.items():
            if config.get(key, _MISSING) != value:
                raise ValueError(
                    f"{label} scheduler configuration {key}="
                    f"{config.get(key, _MISSING)!r}; expected {value!r}"
                )
        for key in missing:
            if key in config:
                raise ValueError(
                    f"{label} scheduler configuration unexpectedly has {key}"
                )

    validate_side(b_config, expected_b, expected_b_missing, "B")
    validate_side(c_config, expected_c, expected_c_missing, "C")
    if b_config.get("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION") != "1":
        raise ValueError("B must use reorder-only native admission")
    if c_config.get("VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION") != "0":
        raise ValueError("C must disable reorder-only native admission")
    if b_config.get("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION", "0") != "0":
        raise ValueError("B must not enable physical-KV admission")
    if c_config.get("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION") != "1":
        raise ValueError("C must enable physical-KV admission")
    return {
        "exact_allowlist_match": True,
        "allowed_difference_keys": sorted(allowed),
        "actual_difference_keys": sorted(actual),
        "differences": differences,
        "all_nonwhitelisted_keys_identical": True,
        "expected_b_values": dict(sorted(expected_b.items())),
        "expected_c_values": dict(sorted(expected_c.items())),
        "expected_b_missing": sorted(expected_b_missing),
        "expected_c_missing": sorted(expected_c_missing),
        "admission_mode_transition": {
            "B": "joint_reorder_only_native_admission",
            "C": "joint_adaptive_physical_kv_admission",
        },
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}: {path}")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{label} contains non-finite JSON constant {value}: {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys are not exact; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _strict_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative JSON integer")
    return value


def _strict_boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def _repository_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.as_posix() != value:
        raise ValueError(f"{label} must be a canonical repository-relative path")
    resolved = (REPOSITORY_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    return resolved


def _expected_repository_relative(path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc


def _numeric_summary(
    samples: Sequence[Mapping[str, Any]], field: str
) -> dict[str, float | None]:
    values = [float(sample[field]) for sample in samples]
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def _summarize_physical_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    malformed_sample_count: int,
    fail_closed_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Rebuild exactly the physical subsection emitted by parser v2.

    This deliberately does not trust aggregate fields in a revalidation
    sidecar.  The comparison reconstructs them from the raw-log samples.
    """

    copied_samples = [dict(sample) for sample in samples]
    effective_caps = {int(sample["effective_cap"]) for sample in copied_samples}
    cap_changes = [
        int(after["effective_cap"]) - int(before["effective_cap"])
        for before, after in zip(copied_samples, copied_samples[1:])
    ]
    evidence: dict[str, Any] = {
        "sample_count": len(copied_samples),
        "malformed_sample_count": malformed_sample_count,
        "fail_closed_count": len(fail_closed_reasons),
        "fail_closed_reasons": sorted(set(fail_closed_reasons)),
        "capacity_tokens": _numeric_summary(copied_samples, "capacity_tokens"),
        "target_utilization": _numeric_summary(
            copied_samples, "target_utilization"
        ),
        "budget_tokens": _numeric_summary(copied_samples, "budget_tokens"),
        "usage": _numeric_summary(copied_samples, "usage"),
        "live_tokens": _numeric_summary(copied_samples, "live_tokens"),
        "logical_live_tokens": _numeric_summary(
            copied_samples, "logical_live_tokens"
        ),
        "running_growth_tokens": _numeric_summary(
            copied_samples, "running_growth_tokens"
        ),
        "reserved_tokens": _numeric_summary(copied_samples, "reserved_tokens"),
        "committed_tokens": _numeric_summary(copied_samples, "committed_tokens"),
        "predicted_admit_tokens": _numeric_summary(
            copied_samples, "predicted_admit_tokens"
        ),
        "admit": _numeric_summary(copied_samples, "admit"),
        "effective_cap": {
            **_numeric_summary(copied_samples, "effective_cap"),
            "unique_count": len(effective_caps),
        },
        "native_cap": _numeric_summary(copied_samples, "native_cap"),
        "capacity_write_count": _numeric_summary(
            copied_samples, "capacity_write_count"
        ),
        "effective_cap_increase_count": sum(change > 0 for change in cap_changes),
        "effective_cap_decrease_count": sum(change < 0 for change in cap_changes),
        "fit_admit_zero_sample_count": sum(
            sample["fit_admit"] == 0 for sample in copied_samples
        ),
        "fit_admit_positive_sample_count": sum(
            sample["fit_admit"] > 0 for sample in copied_samples
        ),
        "effective_cap_above_64_sample_count": sum(
            sample["effective_cap"] > 64 for sample in copied_samples
        ),
        "running_above_64_sample_count": sum(
            sample["running"] > 64 for sample in copied_samples
        ),
        "pressure_above_64_sample_count": sum(
            sample["running"] > 64
            and sample["waiting"] > 0
            and sample["effective_cap"] > 64
            for sample in copied_samples
        ),
        "rescue_sample_count": sum(
            int(sample["rescue"]) for sample in copied_samples
        ),
        "samples": copied_samples,
    }
    checks = {
        "has_samples": bool(copied_samples),
        "no_malformed_samples": malformed_sample_count == 0,
        "no_fail_closed_decisions": not fail_closed_reasons,
        "stable_physical_capacity": bool(copied_samples)
        and len({int(sample["capacity_tokens"]) for sample in copied_samples}) == 1,
        "at_least_three_effective_caps": len(effective_caps) >= 3,
        "observed_cap_increase": any(change > 0 for change in cap_changes),
        "observed_cap_decrease": any(change < 0 for change in cap_changes),
        "observed_zero_fit_admit": any(
            sample["fit_admit"] == 0 for sample in copied_samples
        ),
        "observed_positive_fit_admit": any(
            sample["fit_admit"] > 0 for sample in copied_samples
        ),
        "at_least_ten_pressure_samples_above_64": (
            evidence["pressure_above_64_sample_count"] >= 10
        ),
    }
    evidence["screening_gates"] = {**checks, "passed": all(checks.values())}
    return evidence


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _physical_marker_count(run_path: Path) -> int:
    paths = [run_path / "server.log"]
    server_directory = run_path / "server"
    if server_directory.is_dir():
        paths.extend(sorted(server_directory.glob("*.log")))
    return sum(
        path.read_text(encoding="utf-8", errors="replace").count(PHYSICAL_MARKER)
        for path in paths
        if path.is_file()
    )


def _load_physical_evidence(run_path: Path) -> tuple[dict[str, Any], int]:
    summary = _load_json_object(run_path / "summary.json", "summary")
    sidecar = _load_json_object(
        run_path / "vllm_log_summary.json", "vllm log summary"
    )
    summary_evidence = summary.get("physical_kv_admission")
    sidecar_evidence = sidecar.get("physical_kv_admission")
    if not isinstance(summary_evidence, Mapping) or not isinstance(
        sidecar_evidence, Mapping
    ):
        raise ValueError(f"physical-KV evidence is missing: {run_path}")
    if not _exact_json_equal(summary_evidence, sidecar_evidence):
        raise ValueError(f"summary/sidecar physical-KV evidence mismatch: {run_path}")
    return dict(sidecar_evidence), _physical_marker_count(run_path)


def _validate_source_file_binding(
    binding: Any,
    *,
    expected_path: Path,
    label: str,
) -> dict[str, Any]:
    fields = _require_exact_keys(binding, {"path", "sha256"}, label)
    bound_path = _repository_relative_path(fields["path"], f"{label}.path")
    if bound_path != expected_path.resolve():
        raise ValueError(f"{label}.path does not bind the expected C artifact")
    expected_relative = _expected_repository_relative(expected_path, label)
    if fields["path"] != expected_relative:
        raise ValueError(f"{label}.path is not the exact canonical C path")
    actual_sha = _sha256_file(bound_path)
    if fields["sha256"] != actual_sha:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": expected_relative, "sha256": actual_sha}


def _validate_raw_log_binding(
    binding: Any,
    *,
    expected_path: Path,
) -> tuple[dict[str, Any], str]:
    label = "physical revalidation source.raw_log"
    fields = _require_exact_keys(
        binding,
        {"path", "sha256", "size_bytes", "scope", "marker_count"},
        label,
    )
    bound_path = _repository_relative_path(fields["path"], f"{label}.path")
    if bound_path != expected_path.resolve():
        raise ValueError("physical revalidation raw log is not canonical C server log")
    expected_relative = _expected_repository_relative(expected_path, label)
    if fields["path"] != expected_relative:
        raise ValueError("physical revalidation raw-log path is not canonical")
    if fields["scope"] != "full_server_lifecycle":
        raise ValueError("physical revalidation raw-log scope is not full lifecycle")
    actual_sha = _sha256_file(bound_path)
    if fields["sha256"] != actual_sha:
        raise ValueError("physical revalidation raw-log SHA256 mismatch")
    actual_size = bound_path.stat().st_size
    if _strict_nonnegative_integer(fields["size_bytes"], f"{label}.size_bytes") != actual_size:
        raise ValueError("physical revalidation raw-log size mismatch")
    text = bound_path.read_bytes().decode("utf-8", errors="ignore")
    marker_count = text.count(PHYSICAL_MARKER)
    if _strict_nonnegative_integer(
        fields["marker_count"], f"{label}.marker_count"
    ) != marker_count:
        raise ValueError("physical revalidation raw-log marker count mismatch")
    return (
        {
            "path": expected_relative,
            "sha256": actual_sha,
            "size_bytes": actual_size,
            "scope": "full_server_lifecycle",
            "marker_count": marker_count,
        },
        text,
    )


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(after > before for before, after in zip(values, values[1:]))


def _validate_recomputed_invariants(
    samples: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]
) -> dict[str, Any]:
    expected_keys = {
        "sample_count",
        "capacity_equation_pass_count",
        "effective_cap_equation_pass_count",
        "live_within_physical_capacity_pass_count",
        "nonrescue_positive_admit_count",
        "nonrescue_positive_admit_within_soft_budget_count",
        "nonrescue_zero_admit_count",
        "forecast_hold_over_soft_budget_zero_admit_count",
        "rescue_count",
        "rescue_within_physical_capacity_count",
        "capacity_write_source_physical_count",
        "native_cap_bound_pass_count",
        "all_passed",
    }
    fields = _require_exact_keys(audit, expected_keys, "revalidation invariants")
    positive_nonrescue = [
        sample
        for sample in samples
        if int(sample["rescue"]) == 0 and int(sample["admit"]) > 0
    ]
    zero_nonrescue = [
        sample
        for sample in samples
        if int(sample["rescue"]) == 0 and int(sample["admit"]) == 0
    ]
    rescue_samples = [sample for sample in samples if int(sample["rescue"]) == 1]
    computed: dict[str, Any] = {
        "sample_count": len(samples),
        "capacity_equation_pass_count": sum(
            int(sample["capacity_tokens"])
            == int(sample["num_gpu_blocks"]) * int(sample["block_size"])
            for sample in samples
        ),
        "effective_cap_equation_pass_count": sum(
            int(sample["effective_cap"])
            == min(
                int(sample["native_cap"]),
                int(sample["running"]) + int(sample["admit"]),
            )
            for sample in samples
        ),
        "live_within_physical_capacity_pass_count": sum(
            int(sample["live_tokens"]) <= int(sample["capacity_tokens"])
            for sample in samples
        ),
        "nonrescue_positive_admit_count": len(positive_nonrescue),
        "nonrescue_positive_admit_within_soft_budget_count": sum(
            int(sample["committed_tokens"])
            + int(sample["predicted_admit_tokens"])
            <= int(sample["budget_tokens"])
            for sample in positive_nonrescue
        ),
        "nonrescue_zero_admit_count": len(zero_nonrescue),
        "forecast_hold_over_soft_budget_zero_admit_count": sum(
            sample["reason"] == "forecast_hold"
            and int(sample["fit_admit"]) == 0
            and int(sample["predicted_admit_tokens"]) == 0
            and int(sample["committed_tokens"])
            + int(sample["predicted_admit_tokens"])
            > int(sample["budget_tokens"])
            for sample in zero_nonrescue
        ),
        "rescue_count": len(rescue_samples),
        "rescue_within_physical_capacity_count": sum(
            int(sample["live_tokens"])
            + int(sample["predicted_admit_tokens"])
            <= int(sample["capacity_tokens"])
            for sample in rescue_samples
        ),
        "capacity_write_source_physical_count": sum(
            sample["capacity_write_source"] == "physical_kv" for sample in samples
        ),
        "native_cap_bound_pass_count": sum(
            int(sample["effective_cap"]) <= int(sample["native_cap"])
            for sample in samples
        ),
    }
    sample_count = len(samples)
    computed["all_passed"] = (
        computed["capacity_equation_pass_count"] == sample_count
        and computed["effective_cap_equation_pass_count"] == sample_count
        and computed["live_within_physical_capacity_pass_count"] == sample_count
        and computed["nonrescue_positive_admit_within_soft_budget_count"]
        == computed["nonrescue_positive_admit_count"]
        and computed["rescue_within_physical_capacity_count"]
        == computed["rescue_count"]
        and computed["capacity_write_source_physical_count"] == sample_count
        and computed["native_cap_bound_pass_count"] == sample_count
    )
    for key, expected in computed.items():
        if key == "all_passed":
            actual = _strict_boolean(fields[key], f"revalidation invariants.{key}")
        else:
            actual = _strict_nonnegative_integer(
                fields[key], f"revalidation invariants.{key}"
            )
        if actual != expected:
            raise ValueError(f"revalidation invariant {key} mismatch")
    if computed["all_passed"] is not True:
        raise ValueError("revalidation per-sample safety invariants did not pass")
    return computed


def _load_revalidated_c_physical(
    *,
    c_path: Path,
    revalidation_path: Path,
    original_evidence: Mapping[str, Any],
    raw_log_path: Path,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    # Revalidation is derived evidence and intentionally lives beside, not
    # inside, the immutable completed cell.
    expected_sidecar = c_path.parent / PHYSICAL_REVALIDATION_FILENAME
    if revalidation_path.resolve() != expected_sidecar.resolve():
        raise ValueError(
            "C physical revalidation must be physical_kv_revalidation.json "
            "directly under the C cell parent run root"
        )
    payload = _load_json_object(expected_sidecar, "C physical revalidation")
    _require_exact_keys(
        payload,
        {
            "schema",
            "version",
            "status",
            "source",
            "parser",
            "validator",
            "original_post_run_validation",
            "recomputed",
        },
        "C physical revalidation",
    )
    if payload["schema"] != PHYSICAL_REVALIDATION_SCHEMA:
        raise ValueError("C physical revalidation schema mismatch")
    if _strict_nonnegative_integer(
        payload["version"], "C physical revalidation version"
    ) != PHYSICAL_REVALIDATION_VERSION:
        raise ValueError("C physical revalidation version mismatch")
    if payload["status"] != PHYSICAL_REVALIDATION_STATUS:
        raise ValueError("C physical revalidation is not accepted")

    source = _require_exact_keys(
        payload["source"],
        {"raw_log", "summary", "vllm_log_summary"},
        "C physical revalidation source",
    )
    raw_binding, raw_text = _validate_raw_log_binding(
        source["raw_log"], expected_path=raw_log_path
    )
    summary_binding = _validate_source_file_binding(
        source["summary"],
        expected_path=c_path / "summary.json",
        label="physical revalidation source.summary",
    )
    log_summary_binding = _validate_source_file_binding(
        source["vllm_log_summary"],
        expected_path=c_path / "vllm_log_summary.json",
        label="physical revalidation source.vllm_log_summary",
    )

    parser = _require_exact_keys(
        payload["parser"],
        {"id", "version", "module_path", "module_sha256"},
        "C physical revalidation parser",
    )
    if (
        parser["id"] != PHYSICAL_KV_LOG_PARSER_ID
        or parser["id"] != "paste.physical_kv_admission_log_parser"
        or parser["module_path"] != PHYSICAL_PARSER_MODULE
    ):
        raise ValueError("C physical revalidation parser identity/version mismatch")
    parser_version = _strict_nonnegative_integer(
        parser["version"], "C physical revalidation parser.version"
    )
    if parser_version != PHYSICAL_KV_LOG_PARSER_VERSION or parser_version != 2:
        raise ValueError("C physical revalidation parser identity/version mismatch")
    parser_path = _repository_relative_path(
        parser["module_path"], "physical revalidation parser.module_path"
    )
    if parser_path != (REPOSITORY_ROOT / PHYSICAL_PARSER_MODULE).resolve():
        raise ValueError("C physical revalidation parser module path mismatch")
    if parser["module_sha256"] != _sha256_file(parser_path):
        raise ValueError("C physical revalidation parser module SHA256 mismatch")

    validator = _require_exact_keys(
        payload["validator"],
        {"path", "sha256"},
        "C physical revalidation validator",
    )
    if validator["path"] != PHYSICAL_VALIDATOR_MODULE:
        raise ValueError("C physical revalidation validator path mismatch")
    validator_path = _repository_relative_path(
        validator["path"], "physical revalidation validator.path"
    )
    if validator_path != (REPOSITORY_ROOT / PHYSICAL_VALIDATOR_MODULE).resolve():
        raise ValueError("C physical revalidation validator path mismatch")
    if validator["sha256"] != _sha256_file(validator_path):
        raise ValueError("C physical revalidation validator SHA256 mismatch")

    original = _require_exact_keys(
        payload["original_post_run_validation"],
        {
            "status",
            "sample_count",
            "malformed_sample_count",
            "fail_closed_count",
            "screening_gates",
        },
        "C original post-run validation snapshot",
    )
    original_sample_count = _strict_nonnegative_integer(
        original["sample_count"], "original post-run sample_count"
    )
    original_malformed = _strict_nonnegative_integer(
        original["malformed_sample_count"],
        "original post-run malformed_sample_count",
    )
    original_fail_closed = _strict_nonnegative_integer(
        original["fail_closed_count"], "original post-run fail_closed_count"
    )
    original_samples = original_evidence.get("samples")
    original_gates = original_evidence.get("screening_gates")
    if not isinstance(original_samples, list) or not isinstance(original_gates, Mapping):
        raise ValueError("original C physical evidence is malformed")
    if (
        original["status"] != "failed"
        or original_sample_count
        != _strict_nonnegative_integer(
            original_evidence.get("sample_count"), "stored original sample_count"
        )
        or original_malformed
        != _strict_nonnegative_integer(
            original_evidence.get("malformed_sample_count"),
            "stored original malformed_sample_count",
        )
        or original_fail_closed
        != _strict_nonnegative_integer(
            original_evidence.get("fail_closed_count"),
            "stored original fail_closed_count",
        )
        or not _exact_json_equal(original["screening_gates"], original_gates)
    ):
        raise ValueError("C original post-run failure snapshot does not match artifacts")
    if len(original_samples) != original_sample_count:
        raise ValueError("stored original C physical sample count is inconsistent")
    if original_malformed <= 0 or original_fail_closed != 0:
        raise ValueError(
            "raw-log revalidation is only allowed for the recorded parser-malformed "
            "failure with no fail-closed decision"
        )
    false_original_gates: set[str] = set()
    for key, value in original_gates.items():
        if not isinstance(value, bool):
            raise ValueError("original post-run screening gates must be booleans")
        if not value:
            false_original_gates.add(str(key))
    if false_original_gates != {"no_malformed_samples", "passed"}:
        raise ValueError("original post-run validation failed for an unapproved reason")

    recomputed = _require_exact_keys(
        payload["recomputed"],
        {"physical_kv_admission", "independent_sample_audit"},
        "C physical revalidation recomputed",
    )
    recomputed_evidence = recomputed["physical_kv_admission"]
    if not isinstance(recomputed_evidence, Mapping):
        raise ValueError("recomputed physical-KV evidence must be an object")
    audit = _require_exact_keys(
        recomputed["independent_sample_audit"],
        {
            "experiment_scope",
            "full_raw_scope",
            "legacy_rejection_reason_counts",
            "legacy_rejection_line_shape_counts",
            "invariants",
            "conclusion",
        },
        "C physical revalidation independent audit",
    )

    parsed = parse_vllm_log_segment(raw_text)
    full_evidence = parsed.get("physical_kv_admission")
    if not isinstance(full_evidence, Mapping):
        raise ValueError("parser v2 did not return physical-KV evidence")
    full_samples = full_evidence.get("samples")
    if not isinstance(full_samples, list):
        raise ValueError("parser v2 raw-log samples are missing")
    full_malformed = _strict_nonnegative_integer(
        full_evidence.get("malformed_sample_count"),
        "parser v2 full raw malformed_sample_count",
    )
    full_fail_closed = _strict_nonnegative_integer(
        full_evidence.get("fail_closed_count"),
        "parser v2 full raw fail_closed_count",
    )
    if full_malformed != 0 or full_fail_closed != 0:
        raise ValueError(
            "parser v2 found real malformed or fail-closed physical-KV telemetry"
        )
    if len(full_samples) != raw_binding["marker_count"]:
        raise ValueError("full raw physical marker/sample accounting is not exact")
    full_write_counts = [
        _strict_nonnegative_integer(
            sample.get("capacity_write_count"),
            f"full raw sample {index} capacity_write_count",
        )
        for index, sample in enumerate(full_samples)
        if isinstance(sample, Mapping)
    ]
    if len(full_write_counts) != len(full_samples) or not _strictly_increasing(
        full_write_counts
    ):
        raise ValueError("full raw capacity-write counts are not strictly increasing")

    full_scope = _require_exact_keys(
        audit["full_raw_scope"],
        {
            "marker_count",
            "sample_count",
            "malformed_sample_count",
            "fail_closed_count",
            "capacity_write_count_first",
            "capacity_write_count_last",
            "capacity_write_counts_strictly_increasing",
        },
        "C revalidation full_raw_scope",
    )
    computed_full_scope = {
        "marker_count": raw_binding["marker_count"],
        "sample_count": len(full_samples),
        "malformed_sample_count": full_malformed,
        "fail_closed_count": full_fail_closed,
        "capacity_write_count_first": full_write_counts[0],
        "capacity_write_count_last": full_write_counts[-1],
        "capacity_write_counts_strictly_increasing": True,
    }
    if not _exact_json_equal(dict(full_scope), computed_full_scope):
        raise ValueError("C revalidation full_raw_scope does not match raw log")

    experiment_scope = _require_exact_keys(
        audit["experiment_scope"],
        {
            "derivation",
            "marker_count",
            "selected_capacity_write_count_first",
            "selected_capacity_write_count_last",
            "excluded_prefix_marker_count",
            "excluded_prefix_capacity_write_counts",
            "excluded_suffix_marker_count",
            "excluded_suffix_capacity_write_counts",
            "raw_marker_accounting_exact",
            "capacity_write_counts_strictly_increasing",
            "legacy_accepted_sample_count",
            "legacy_rejected_sample_count",
            "legacy_accepted_samples_exact_match",
            "legacy_aggregate_exact_match",
        },
        "C revalidation experiment_scope",
    )
    if experiment_scope["derivation"] != "legacy_stored_telemetry_exact_match":
        raise ValueError("C revalidation experiment scope has wrong derivation")
    first_count = _strict_nonnegative_integer(
        experiment_scope["selected_capacity_write_count_first"],
        "experiment scope first capacity-write count",
    )
    last_count = _strict_nonnegative_integer(
        experiment_scope["selected_capacity_write_count_last"],
        "experiment scope last capacity-write count",
    )
    try:
        first_index = full_write_counts.index(first_count)
        last_index = full_write_counts.index(last_count)
    except ValueError as exc:
        raise ValueError("experiment scope boundary is absent from raw log") from exc
    if first_index > last_index:
        raise ValueError("experiment scope capacity-write boundary is reversed")
    selected_samples = full_samples[first_index : last_index + 1]
    selected_write_counts = full_write_counts[first_index : last_index + 1]
    excluded_prefix = full_write_counts[:first_index]
    excluded_suffix = full_write_counts[last_index + 1 :]
    if not _strictly_increasing(selected_write_counts):
        raise ValueError("experiment capacity-write counts are not strictly increasing")

    def was_rejected_by_legacy_parser(sample: Mapping[str, Any]) -> bool:
        return (
            int(sample["rescue"]) == 0
            and int(sample["committed_tokens"])
            + int(sample["predicted_admit_tokens"])
            > int(sample["budget_tokens"])
        )

    legacy_rejected = [
        sample for sample in selected_samples if was_rejected_by_legacy_parser(sample)
    ]
    legacy_accepted = [
        sample
        for sample in selected_samples
        if not was_rejected_by_legacy_parser(sample)
    ]
    # The only permitted v1-parser false rejection is the known safe hold:
    # no admission, no predicted addition, and no rescue override.
    for index, sample in enumerate(legacy_rejected):
        if not (
            sample.get("decision") == "admit"
            and sample.get("reason") == "forecast_hold"
            and int(sample["rescue"]) == 0
            and int(sample["admit"]) == 0
            and int(sample["fit_admit"]) == 0
            and int(sample["predicted_admit_tokens"]) == 0
        ):
            raise ValueError(
                f"legacy-rejected raw sample {index} is a real malformed shape"
            )
    if not _exact_json_equal(legacy_accepted, original_samples):
        raise ValueError(
            "legacy-accepted raw samples do not exactly match stored telemetry"
        )
    legacy_aggregate = _summarize_physical_samples(
        legacy_accepted,
        malformed_sample_count=len(legacy_rejected),
    )
    if not _exact_json_equal(legacy_aggregate, original_evidence):
        raise ValueError("legacy raw-log aggregate does not exactly match stored telemetry")
    rebuilt_recomputed = _summarize_physical_samples(
        selected_samples,
        malformed_sample_count=0,
    )
    if not _exact_json_equal(rebuilt_recomputed, recomputed_evidence):
        raise ValueError("recomputed physical-KV aggregate does not match raw samples")

    computed_experiment_scope = {
        "derivation": "legacy_stored_telemetry_exact_match",
        "marker_count": len(selected_samples),
        "selected_capacity_write_count_first": selected_write_counts[0],
        "selected_capacity_write_count_last": selected_write_counts[-1],
        "excluded_prefix_marker_count": len(excluded_prefix),
        "excluded_prefix_capacity_write_counts": excluded_prefix,
        "excluded_suffix_marker_count": len(excluded_suffix),
        "excluded_suffix_capacity_write_counts": excluded_suffix,
        "raw_marker_accounting_exact": (
            len(selected_samples) + len(excluded_prefix) + len(excluded_suffix)
            == raw_binding["marker_count"]
        ),
        "capacity_write_counts_strictly_increasing": True,
        "legacy_accepted_sample_count": len(legacy_accepted),
        "legacy_rejected_sample_count": len(legacy_rejected),
        "legacy_accepted_samples_exact_match": True,
        "legacy_aggregate_exact_match": True,
    }
    if not _exact_json_equal(dict(experiment_scope), computed_experiment_scope):
        raise ValueError("C revalidation experiment_scope does not match raw evidence")
    if len(legacy_rejected) != original_malformed:
        raise ValueError("legacy rejection count does not match original malformed count")
    reason_counts = {LEGACY_REJECTION_REASON: len(legacy_rejected)}
    shape_counts = {LEGACY_REJECTION_SHAPE: len(legacy_rejected)}
    if not _exact_json_equal(audit["legacy_rejection_reason_counts"], reason_counts):
        raise ValueError("C revalidation legacy rejection reason counts mismatch")
    if not _exact_json_equal(
        audit["legacy_rejection_line_shape_counts"], shape_counts
    ):
        raise ValueError("C revalidation legacy rejection shape counts mismatch")
    invariant_audit = _validate_recomputed_invariants(
        selected_samples,
        _require_exact_keys(
            audit["invariants"],
            {
                "sample_count",
                "capacity_equation_pass_count",
                "effective_cap_equation_pass_count",
                "live_within_physical_capacity_pass_count",
                "nonrescue_positive_admit_count",
                "nonrescue_positive_admit_within_soft_budget_count",
                "nonrescue_zero_admit_count",
                "forecast_hold_over_soft_budget_zero_admit_count",
                "rescue_count",
                "rescue_within_physical_capacity_count",
                "capacity_write_source_physical_count",
                "native_cap_bound_pass_count",
                "all_passed",
            },
            "C revalidation invariants",
        ),
    )
    if audit["conclusion"] != "all_experiment_samples_safe":
        raise ValueError("C revalidation safety conclusion is not accepted")

    return (
        dict(recomputed_evidence),
        len(selected_samples),
        {
            "schema": PHYSICAL_REVALIDATION_SCHEMA,
            "version": PHYSICAL_REVALIDATION_VERSION,
            "status": PHYSICAL_REVALIDATION_STATUS,
            "accepted": True,
            "sidecar": _expected_repository_relative(
                expected_sidecar, "C physical revalidation sidecar"
            ),
            "source": {
                "raw_log": raw_binding,
                "summary": summary_binding,
                "vllm_log_summary": log_summary_binding,
            },
            "parser": dict(parser),
            "validator": dict(validator),
            "original_post_run_validation": dict(original),
            "experiment_scope": computed_experiment_scope,
            "full_raw_scope": computed_full_scope,
            "legacy_rejection_reason_counts": reason_counts,
            "legacy_rejection_line_shape_counts": shape_counts,
            "invariants": invariant_audit,
            "conclusion": "all_experiment_samples_safe",
            "original_artifacts_preserved": True,
        },
    )


def _require_b_has_no_capacity_writes(
    evidence: Mapping[str, Any], marker_count: int
) -> dict[str, Any]:
    sample_count = _nonnegative_integer(evidence.get("sample_count"), "B sample_count")
    malformed = _nonnegative_integer(
        evidence.get("malformed_sample_count"), "B malformed_sample_count"
    )
    fail_closed = _nonnegative_integer(
        evidence.get("fail_closed_count"), "B fail_closed_count"
    )
    samples = evidence.get("samples")
    writes = evidence.get("capacity_write_count")
    fail_closed_reasons = evidence.get("fail_closed_reasons")
    gates = evidence.get("screening_gates")
    if not isinstance(samples, list) or not isinstance(writes, Mapping):
        raise ValueError("B physical-KV evidence is malformed")
    if not isinstance(fail_closed_reasons, list) or not isinstance(gates, Mapping):
        raise ValueError("B physical-KV evidence is malformed")
    write_values = (writes.get("min"), writes.get("max"), writes.get("mean"))
    if (
        sample_count != 0
        or samples
        or marker_count != 0
        or malformed != 0
        or fail_closed != 0
        or fail_closed_reasons
        or gates.get("passed") is not False
        or any(value is not None for value in write_values)
    ):
        raise ValueError("B unexpectedly contains physical-KV capacity writes")
    return {
        "passed": True,
        "sample_count": 0,
        "physical_log_marker_count": 0,
        "capacity_write_count": 0,
        "interpretation": "native-admission B contains no physical-KV cap write",
    }


def _require_c_physical_gates(
    evidence: Mapping[str, Any],
    *,
    marker_count: int,
    expected_target: float,
    expected_native_cap: int,
) -> dict[str, Any]:
    sample_count = _nonnegative_integer(evidence.get("sample_count"), "C sample_count")
    malformed = _nonnegative_integer(
        evidence.get("malformed_sample_count"), "C malformed_sample_count"
    )
    fail_closed = _nonnegative_integer(
        evidence.get("fail_closed_count"), "C fail_closed_count"
    )
    gates = evidence.get("screening_gates")
    samples = evidence.get("samples")
    if not isinstance(gates, Mapping) or not isinstance(samples, list):
        raise ValueError("C physical-KV evidence is malformed")
    if sample_count <= 0 or len(samples) != sample_count:
        raise ValueError("C physical-KV sample count is empty or inconsistent")
    if malformed != 0:
        raise ValueError("C physical-KV telemetry contains malformed samples")
    if fail_closed != 0:
        raise ValueError("C physical-KV controller entered fail-closed")
    fail_closed_reasons = evidence.get("fail_closed_reasons")
    if not isinstance(fail_closed_reasons, list) or fail_closed_reasons:
        raise ValueError("C physical-KV fail-closed reason accounting is inconsistent")
    if gates.get("passed") is not True:
        raise ValueError("C physical-KV screening gates did not pass")
    if any(value is not True for key, value in gates.items() if key != "passed"):
        raise ValueError("C physical-KV screening gate map is internally inconsistent")
    if marker_count < sample_count:
        raise ValueError("C raw logs contain fewer physical markers than parsed samples")

    capacities: set[int] = set()
    targets: set[float] = set()
    native_caps: set[int] = set()
    effective_caps: list[int] = []
    fit_admits: list[int] = []
    write_counts: list[int] = []
    pressure_above_64 = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"C physical-KV sample {index} is not an object")
        integer_fields = {
            key: _nonnegative_integer(sample.get(key), f"C sample {index} {key}")
            for key in (
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
        }
        target = _finite_float(
            sample.get("target_utilization"),
            f"C sample {index} target_utilization",
        )
        usage = _finite_float(sample.get("usage"), f"C sample {index} usage")
        if not 0.0 <= usage <= 1.0 or not 0.0 < target <= 1.0:
            raise ValueError(f"C physical-KV sample {index} has invalid utilization")
        if sample.get("capacity_write_source") != "physical_kv":
            raise ValueError(f"C physical-KV sample {index} has wrong write source")
        if integer_fields["capacity_write_count"] <= 0:
            raise ValueError(f"C physical-KV sample {index} has no capacity write")
        if integer_fields["rescue"] not in {0, 1}:
            raise ValueError(f"C physical-KV sample {index} has invalid rescue marker")
        if integer_fields["capacity_tokens"] != (
            integer_fields["num_gpu_blocks"] * integer_fields["block_size"]
        ):
            raise ValueError(f"C physical-KV sample {index} has invalid capacity")
        expected_cap = min(
            integer_fields["native_cap"],
            integer_fields["running"] + integer_fields["admit"],
        )
        if integer_fields["effective_cap"] != expected_cap:
            raise ValueError(f"C physical-KV sample {index} has invalid effective cap")
        if integer_fields["admit"] < integer_fields["fit_admit"]:
            raise ValueError(f"C physical-KV sample {index} admits below fit count")
        if integer_fields["rescue"] == 0 and integer_fields["admit"] > 0 and (
            integer_fields["committed_tokens"]
            + integer_fields["predicted_admit_tokens"]
            > integer_fields["budget_tokens"]
        ):
            raise ValueError(
                f"C physical-KV positive-admit sample {index} exceeds target budget"
            )
        if integer_fields["rescue"] == 0 and integer_fields["admit"] == 0:
            if integer_fields["predicted_admit_tokens"] != 0:
                raise ValueError(
                    f"C physical-KV zero-admit sample {index} predicts an admission"
                )
            over_soft_budget = (
                integer_fields["committed_tokens"]
                + integer_fields["predicted_admit_tokens"]
                > integer_fields["budget_tokens"]
            )
            if over_soft_budget and not (
                sample.get("decision") == "admit"
                and sample.get("reason") == "forecast_hold"
                and integer_fields["fit_admit"] == 0
            ):
                raise ValueError(
                    f"C physical-KV zero-admit sample {index} has an unsafe "
                    "over-budget exemption"
                )
        if integer_fields["rescue"] == 1 and (
            integer_fields["live_tokens"]
            + integer_fields["predicted_admit_tokens"]
            > integer_fields["capacity_tokens"]
        ):
            raise ValueError(f"C physical-KV rescue sample {index} exceeds capacity")
        capacities.add(integer_fields["capacity_tokens"])
        targets.add(target)
        native_caps.add(integer_fields["native_cap"])
        effective_caps.append(integer_fields["effective_cap"])
        fit_admits.append(integer_fields["fit_admit"])
        write_counts.append(integer_fields["capacity_write_count"])
        pressure_above_64 += int(
            integer_fields["running"] > 64
            and integer_fields["waiting"] > 0
            and integer_fields["effective_cap"] > 64
        )

    if len(capacities) != 1:
        raise ValueError("C physical KV capacity changed during the run")
    if len(targets) != 1 or not math.isclose(
        next(iter(targets)), expected_target, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("C physical-KV telemetry target differs from configuration")
    if native_caps != {expected_native_cap}:
        raise ValueError("C physical-KV telemetry native cap differs from engine shape")
    if len(set(effective_caps)) < 3:
        raise ValueError("C physical-KV cap did not take at least three values")
    changes = [after - before for before, after in zip(effective_caps, effective_caps[1:])]
    if not any(change > 0 for change in changes) or not any(
        change < 0 for change in changes
    ):
        raise ValueError("C physical-KV cap lacks bidirectional variation")
    if 0 not in fit_admits or not any(value > 0 for value in fit_admits):
        raise ValueError("C physical-KV admission never both bound and released")
    if pressure_above_64 < 10:
        raise ValueError("C lacks ten pressure samples with running/cap above 64")
    if not _strictly_increasing(write_counts):
        raise ValueError(
            "C physical-KV capacity-write counter is not strictly increasing"
        )

    def require_numeric_summary(
        field: str,
        *,
        expected_min: float,
        expected_max: float,
        expected_unique: int | None = None,
    ) -> None:
        summary = evidence.get(field)
        if not isinstance(summary, Mapping):
            raise ValueError(f"C physical-KV {field} summary is missing")
        observed_min = _finite_float(summary.get("min"), f"C {field} min")
        observed_max = _finite_float(summary.get("max"), f"C {field} max")
        if not math.isclose(
            observed_min, expected_min, rel_tol=0.0, abs_tol=1e-9
        ) or not math.isclose(
            observed_max, expected_max, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"C physical-KV {field} summary/sample mismatch")
        if expected_unique is not None and _nonnegative_integer(
            summary.get("unique_count"), f"C {field} unique_count"
        ) != expected_unique:
            raise ValueError(f"C physical-KV {field} unique-count mismatch")

    require_numeric_summary(
        "capacity_tokens",
        expected_min=float(min(capacities)),
        expected_max=float(max(capacities)),
    )
    require_numeric_summary(
        "target_utilization",
        expected_min=min(targets),
        expected_max=max(targets),
    )
    require_numeric_summary(
        "native_cap",
        expected_min=float(min(native_caps)),
        expected_max=float(max(native_caps)),
    )
    require_numeric_summary(
        "effective_cap",
        expected_min=float(min(effective_caps)),
        expected_max=float(max(effective_caps)),
        expected_unique=len(set(effective_caps)),
    )
    require_numeric_summary(
        "capacity_write_count",
        expected_min=float(min(write_counts)),
        expected_max=float(max(write_counts)),
    )
    expected_counts = {
        "effective_cap_increase_count": sum(change > 0 for change in changes),
        "effective_cap_decrease_count": sum(change < 0 for change in changes),
        "fit_admit_zero_sample_count": sum(value == 0 for value in fit_admits),
        "fit_admit_positive_sample_count": sum(value > 0 for value in fit_admits),
        "pressure_above_64_sample_count": pressure_above_64,
    }
    for field, expected in expected_counts.items():
        if _nonnegative_integer(evidence.get(field), f"C {field}") != expected:
            raise ValueError(f"C physical-KV {field} summary/sample mismatch")

    return {
        "passed": True,
        "physical_log_marker_count": marker_count,
        "sample_count": sample_count,
        "capacity_tokens": next(iter(capacities)),
        "target_utilization": next(iter(targets)),
        "native_cap": next(iter(native_caps)),
        "effective_cap_min": min(effective_caps),
        "effective_cap_max": max(effective_caps),
        "effective_cap_unique_count": len(set(effective_caps)),
        "effective_cap_increase_count": expected_counts[
            "effective_cap_increase_count"
        ],
        "effective_cap_decrease_count": expected_counts[
            "effective_cap_decrease_count"
        ],
        "fit_admit_zero_sample_count": expected_counts[
            "fit_admit_zero_sample_count"
        ],
        "fit_admit_positive_sample_count": expected_counts[
            "fit_admit_positive_sample_count"
        ],
        "pressure_above_64_sample_count": pressure_above_64,
        "capacity_write_count_min": min(write_counts),
        "capacity_write_count_max": max(write_counts),
        "all_capacity_writes_from_physical_kv": True,
    }


def _reduction(baseline: float, candidate: float) -> dict[str, float | None]:
    return {
        "b_minus_c_s": baseline - candidate,
        "relative_reduction": (
            (baseline - candidate) / baseline if baseline else None
        ),
    }


def _comparison(
    b_metrics: Mapping[str, Any], c_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "definition": "B - C; positive means adaptive physical-KV C is lower/faster",
        "task_flow_time_s": {
            statistic: _reduction(
                float(b_metrics["task_flow_time_s"][statistic]),
                float(c_metrics["task_flow_time_s"][statistic]),
            )
            for statistic in ("mean", "p50", "p95", "p99", "max")
        },
        "task_makespan_s": _reduction(
            float(b_metrics["task_makespan_s"]),
            float(c_metrics["task_makespan_s"]),
        ),
        "request_latency_s": {
            statistic: _reduction(
                float(b_metrics["request_latency_s"][statistic]),
                float(c_metrics["request_latency_s"][statistic]),
            )
            for statistic in ("mean", "p50", "p95", "p99", "max")
        },
        "request_tail_counts": {
            threshold: {
                "b": int(b_metrics["request_latency_s"][threshold]),
                "c": int(c_metrics["request_latency_s"][threshold]),
                "b_minus_c": int(b_metrics["request_latency_s"][threshold])
                - int(c_metrics["request_latency_s"][threshold]),
            }
            for threshold in ("count_gt_120_s", "count_gt_240_s")
        },
        "mean_queue_time_s": _reduction(
            float(b_metrics["mean_queue_time_s"]),
            float(c_metrics["mean_queue_time_s"]),
        ),
        "mean_nonqueue_request_time_s": _reduction(
            float(b_metrics["mean_nonqueue_request_time_s"]),
            float(c_metrics["mean_nonqueue_request_time_s"]),
        ),
    }


def _execution_comparison(
    b_metrics: Mapping[str, Any], c_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    b_execution = b_metrics["execution_accounting"]
    c_execution = c_metrics["execution_accounting"]
    b_tokens = b_execution["completion_tokens"]["total"]
    c_tokens = c_execution["completion_tokens"]["total"]
    tokens = None
    if b_tokens is not None and c_tokens is not None:
        tokens = {
            "b_total": b_tokens,
            "c_total": c_tokens,
            "c_minus_b": c_tokens - b_tokens,
            "c_relative_to_b": (c_tokens - b_tokens) / b_tokens if b_tokens else None,
        }
    b_preempt = b_execution["preemption"]["num_preemptions_total"]
    c_preempt = c_execution["preemption"]["num_preemptions_total"]
    return {
        "completion_tokens": tokens,
        "retry_accounting": {
            "B": b_metrics["retry_accounting"],
            "C": c_metrics["retry_accounting"],
        },
        "preemption": {
            "b_total": b_preempt,
            "c_total": c_preempt,
            "b_minus_c": (
                b_preempt - c_preempt
                if b_preempt is not None and c_preempt is not None
                else None
            ),
        },
        "swap": {
            "B": b_execution["swap"],
            "C": c_execution["swap"],
        },
    }


def _source_pairing(
    b_flows: Mapping[str, Mapping[str, Any]],
    c_flows: Mapping[str, Mapping[str, Any]],
    source_mapping: Mapping[str, str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    instance_deltas: list[float] = []
    for trace_id in sorted(b_flows):
        delta = float(b_flows[trace_id]["task_flow_s"]) - float(
            c_flows[trace_id]["task_flow_s"]
        )
        instance_deltas.append(delta)
        grouped.setdefault(str(source_mapping[trace_id]), []).append(
            {
                "trace_id": trace_id,
                "b_task_flow_s": float(b_flows[trace_id]["task_flow_s"]),
                "c_task_flow_s": float(c_flows[trace_id]["task_flow_s"]),
                "delta_s": delta,
            }
        )
    source_rows: list[dict[str, Any]] = []
    for source in sorted(grouped):
        instances = grouped[source]
        mean_delta = statistics.fmean(row["delta_s"] for row in instances)
        source_rows.append(
            {
                "source_session": source,
                "trace_ids": [row["trace_id"] for row in instances],
                "load_instance_count": len(instances),
                "b_task_flow_mean_s": statistics.fmean(
                    row["b_task_flow_s"] for row in instances
                ),
                "c_task_flow_mean_s": statistics.fmean(
                    row["c_task_flow_s"] for row in instances
                ),
                "delta_mean_s": mean_delta,
                "outcome": (
                    "c_faster"
                    if mean_delta > TIE_EPSILON_S
                    else "c_slower"
                    if mean_delta < -TIE_EPSILON_S
                    else "tie"
                ),
            }
        )
    source_deltas = [row["delta_mean_s"] for row in source_rows]
    source_bootstrap = _bootstrap_mean_ci(
        source_deltas,
        seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES,
    )
    source_bootstrap["estimand"] = "mean_B_minus_C_task_flow_s"

    def outcomes(values: Sequence[float]) -> dict[str, int | float]:
        wins = sum(value > TIE_EPSILON_S for value in values)
        losses = sum(value < -TIE_EPSILON_S for value in values)
        ties = len(values) - wins - losses
        return {
            "c_faster": wins,
            "tie": ties,
            "c_slower": losses,
            "c_faster_fraction": wins / len(values),
        }

    return {
        "definition": (
            "B-C task flow; deterministic load instances are averaged within each "
            "independent source before inference"
        ),
        "load_instance_count": len(instance_deltas),
        "independent_source_session_count": len(source_rows),
        "load_instance_outcomes": outcomes(instance_deltas),
        "source_session_outcomes": outcomes(source_deltas),
        "source_mean_saving_s": statistics.fmean(source_deltas),
        "independent_source_mean_bootstrap_95_ci_s": source_bootstrap,
        "source_sessions": source_rows,
    }


def _saving_decomposition(
    b_metrics: Mapping[str, Any], c_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    b_components = b_metrics["mean_task_component_s"]
    c_components = c_metrics["mean_task_component_s"]
    components = {
        key: float(b_components[key]) - float(c_components[key])
        for key in (
            "queue",
            "nonqueue_request",
            "noninitial_recorded_tool_wait",
            "residual_harness_and_timing",
        )
    }
    total = float(b_metrics["task_flow_time_s"]["mean"]) - float(
        c_metrics["task_flow_time_s"]["mean"]
    )
    reconstructed = sum(components.values())
    if not math.isclose(total, reconstructed, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError("B/C task-saving decomposition does not reconstruct total")
    return {
        "definition": "B component - C component; positive contributes to C saving",
        "task_mean_saving_s": total,
        "components_s": components,
        "component_fraction_of_total_saving": {
            key: value / total if total else None for key, value in components.items()
        },
        "reconstructed_task_mean_saving_s": reconstructed,
    }


def summarize_strict_screening_bc(
    *,
    manifest_path: Path,
    role: str,
    b_run: Path,
    c_run: Path,
    expected_b_config: Mapping[str, str],
    expected_c_config: Mapping[str, str],
    expected_b_config_missing: set[str],
    expected_c_config_missing: set[str],
    c_physical_revalidation: Path | None = None,
    required_engine_keys: Sequence[str] = DEFAULT_ENGINE_KEYS,
    verify_frozen_configs: bool = True,
) -> dict[str, Any]:
    b_path = b_run.resolve()
    c_path = c_run.resolve()
    if b_path == c_path:
        raise ValueError("B and C run directories must be distinct")
    manifest = load_fixed_manifest(manifest_path, role)
    b = load_run(b_path, "D", manifest["bindings"]["D"])
    c = load_run(c_path, "D", manifest["bindings"]["D"])

    if b["identity_rows"] != c["identity_rows"]:
        raise ValueError("B/C request identity, prompts, or messages mismatch")
    if b["source_mapping"] != c["source_mapping"]:
        raise ValueError("B/C source-session mapping mismatch")
    source_counts = Counter(b["source_mapping"].values())
    _validate_source_multiplicity(source_counts, workload_invariants=manifest, replicate=1)
    for field in (
        "speedup",
        "max_active_traces",
        "tool_wait_mode",
        "configured_max_request_attempts",
    ):
        if b["public"][field] != c["public"][field]:
            raise ValueError(f"B/C replay configuration mismatch: {field}")
    for label, run in (("B", b), ("C", c)):
        if run["public"]["policy"] != EXPECTED_POLICY:
            raise ValueError(f"{label} must use {EXPECTED_POLICY}")
        if run["public"]["tool_overlap_mode"] != EXPECTED_OVERLAP:
            raise ValueError(f"{label} must use learned tool overlap")

    identity_hash_fields = (
        "prepared_workload_sha256",
        "scheduler_calibration_workload_sha256",
        "mapper_artifact_sha256",
        "tool_prediction_top_k",
        "request_identity_sha256",
        "source_sessions_sha256",
    )
    identical_hashes: dict[str, Any] = {}
    for field in identity_hash_fields:
        b_value = b["public"][field]
        c_value = c["public"][field]
        if b_value != c_value:
            raise ValueError(f"B/C {field} mismatch")
        identical_hashes[field] = b_value
    if identical_hashes["mapper_artifact_sha256"] is None:
        raise ValueError("B/C learned runs must bind a mapper artifact")

    b_config = b["public"]["scheduler_configuration"]
    c_config = c["public"]["scheduler_configuration"]
    config_guard = _exact_config_guard(
        b_config,
        c_config,
        expected_b=expected_b_config,
        expected_c=expected_c_config,
        expected_b_missing=expected_b_config_missing,
        expected_c_missing=expected_c_config_missing,
    )
    engine_guard = _engine_shape_guard(
        b_config,
        c_config,
        required_keys=required_engine_keys,
        allowed_differences=set(ALLOWED_CONFIG_DIFFERENCES),
    )
    frozen_evidence: dict[str, Any] | None = None
    if verify_frozen_configs:
        frozen_evidence = {
            "B": _verify_frozen_config(
                b_path, b_config.get("PASTE_FROZEN_CONFIG_SHA256")
            ),
            "C": _verify_frozen_config(
                c_path, c_config.get("PASTE_FROZEN_CONFIG_SHA256")
            ),
        }

    b_flows = _task_flow_by_trace(b_path, b)
    c_flows = _task_flow_by_trace(c_path, c)
    if set(b_flows) != set(c_flows):
        raise ValueError("B/C task identities do not exactly match")
    for trace_id in b_flows:
        if (
            b_flows[trace_id]["source_session"]
            != c_flows[trace_id]["source_session"]
            or b_flows[trace_id]["initial_delay_s"]
            != c_flows[trace_id]["initial_delay_s"]
        ):
            raise ValueError(f"B/C task pairing mismatch: {trace_id}")

    cells: dict[str, Any] = {}
    for label, path, run, flows in (
        ("B", b_path, b, b_flows),
        ("C", c_path, c, c_flows),
    ):
        metrics = _cell_metrics(path, run, flows)
        metrics["execution_accounting"] = _load_raw_execution_accounting(
            path, run["public"]
        )
        cells[label] = metrics

    b_physical, b_marker_count = _load_physical_evidence(b_path)
    c_original_physical, c_original_marker_count = _load_physical_evidence(c_path)
    b_zero_write = _require_b_has_no_capacity_writes(
        b_physical, b_marker_count
    )
    target = _finite_float(
        c_config.get("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"),
        "C physical-KV target configuration",
    )
    native_cap = _nonnegative_integer(
        c_config.get("VLLM_MAX_NUM_SEQS"), "C VLLM_MAX_NUM_SEQS"
    )
    revalidation_audit: dict[str, Any] | None = None
    if c_physical_revalidation is None:
        c_physical = c_original_physical
        c_marker_count = c_original_marker_count
        c_evidence_basis = "original_post_run_validation"
        c_original_validation = {
            "status": "passed",
            "sample_count": c_original_physical.get("sample_count"),
            "malformed_sample_count": c_original_physical.get(
                "malformed_sample_count"
            ),
            "fail_closed_count": c_original_physical.get("fail_closed_count"),
            "screening_gates": c_original_physical.get("screening_gates"),
        }
    else:
        port = c_config.get("VLLM_PORT")
        if not isinstance(port, str) or not port.isdigit():
            raise ValueError("C VLLM_PORT is required to bind the canonical raw log")
        c_physical, c_marker_count, revalidation_audit = (
            _load_revalidated_c_physical(
                c_path=c_path,
                revalidation_path=c_physical_revalidation,
                original_evidence=c_original_physical,
                raw_log_path=c_path / "server" / f"vllm_{port}.log",
            )
        )
        c_evidence_basis = "accepted_raw_log_revalidation"
        c_original_validation = revalidation_audit[
            "original_post_run_validation"
        ]
    c_gate_audit = _require_c_physical_gates(
        c_physical,
        marker_count=c_marker_count,
        expected_target=target,
        expected_native_cap=native_cap,
    )

    source_pairing = _source_pairing(b_flows, c_flows, b["source_mapping"])
    if source_pairing["independent_source_session_count"] != manifest[
        "independent_source_session_count"
    ]:
        raise AssertionError("source-folded sample count differs from manifest")
    comparison = _comparison(cells["B"], cells["C"])
    comparison["execution"] = _execution_comparison(cells["B"], cells["C"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "strict_b_vs_c_physical_kv_candidate_screen",
        "comparison_invariants": {
            "fixed_role": role,
            "fixed_workload_manifest": manifest["path"].as_posix(),
            "fixed_workload_manifest_sha256": manifest["manifest_sha256"],
            "load_instance_count": manifest["load_instance_count"],
            "independent_source_session_count": manifest[
                "independent_source_session_count"
            ],
            "instances_per_source": manifest["instances_per_source"],
            "duplicates_are_not_independent": manifest[
                "duplicates_are_not_independent"
            ],
            "request_identity_exact_match": True,
            "source_mapping_exact_match": True,
            "policy_and_overlap_identical": True,
            "mode": {
                "policy": EXPECTED_POLICY,
                "tool_overlap": EXPECTED_OVERLAP,
            },
            "identical_workload_calibration_mapper_evidence": identical_hashes,
            "engine_shape_guard": engine_guard,
            "scheduler_configuration_guard": config_guard,
            "frozen_config_evidence": frozen_evidence,
        },
        "cells": cells,
        "comparison": comparison,
        "source_pairing": source_pairing,
        "task_saving_decomposition": _saving_decomposition(
            cells["B"], cells["C"]
        ),
        "physical_kv_admission_evidence": {
            "B": b_physical,
            "C": c_physical,
            "C_original_physical_kv_artifact": c_original_physical,
            "C_original_post_run_validation": c_original_validation,
            "C_evidence_basis": c_evidence_basis,
            "C_raw_log_revalidation": revalidation_audit,
            "independent_gate_audit": {
                "B_zero_capacity_writes": b_zero_write,
                "C_dynamic_physical_kv_gates": c_gate_audit,
                "passed": True,
            },
        },
        "interpretation": (
            "B and C use identical Joint ordering, learned overlap, deterministic "
            "request/source identities, engine shape, calibration, and mapper. "
            "Their only configured difference is the explicit native-to-physical "
            "admission allowlist. B-C therefore estimates the incremental adaptive "
            "physical-KV admission bundle in this one screening pair; it is not an "
            "independent replicated estimate. Bootstrap inference folds deterministic "
            "load copies into independent source-session means. If raw-log "
            "revalidation is selected, the original failed post-run validation is "
            "retained verbatim and only a hash-bound parser-v2 reclassification of "
            "the known zero-admit forecast holds is used; no original artifact is "
            "overwritten."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("final", "heldout", "stress"), default="stress")
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--c-run", type=Path, required=True)
    parser.add_argument("--expect-b-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--expect-c-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--expect-b-config-missing", action="append", default=[], metavar="KEY")
    parser.add_argument("--expect-c-config-missing", action="append", default=[], metavar="KEY")
    parser.add_argument(
        "--c-physical-revalidation",
        type=Path,
        help=(
            "optional hash-bound raw-log revalidation sidecar for an original "
            "parser-malformed-only C failure"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    b_missing = set(args.expect_b_config_missing)
    c_missing = set(args.expect_c_config_missing)
    if len(b_missing) != len(args.expect_b_config_missing):
        raise ValueError("--expect-b-config-missing contains duplicate keys")
    if len(c_missing) != len(args.expect_c_config_missing):
        raise ValueError("--expect-c-config-missing contains duplicate keys")
    c_path = args.c_run.resolve()
    output = args.output.resolve()
    if output.parent != c_path or output.suffix != ".json":
        raise ValueError("--output must be a JSON file directly under the C run root")
    result = summarize_strict_screening_bc(
        manifest_path=args.manifest,
        role=args.role,
        b_run=args.b_run,
        c_run=args.c_run,
        expected_b_config=_parse_key_value(
            args.expect_b_config, "--expect-b-config"
        ),
        expected_c_config=_parse_key_value(
            args.expect_c_config, "--expect-c-config"
        ),
        expected_b_config_missing=b_missing,
        expected_c_config_missing=c_missing,
        c_physical_revalidation=args.c_physical_revalidation,
        verify_frozen_configs=True,
    )
    write_json_atomic(output, result)
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
