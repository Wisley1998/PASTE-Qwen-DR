#!/usr/bin/env python3
"""Validate and summarize a matched four-cell trace replay experiment.

Cells are fixed as:

* A: FCFS + no tool overlap
* B: FCFS + learned tool overlap
* C: joint scheduler + no tool overlap
* D: joint scheduler + learned tool overlap

Positive effects always mean a reduction in a lower-is-better metric.  Task
flow time is completion offset minus that trace's initial arrival delay.  The
primary makespan comes directly from request completion events, while the
runner's instrumentation wall clock is retained as a separate diagnostic.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
RUNNER_SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, RUNNER_SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from paste_repro.mapper import load_artifact, write_json_atomic  # noqa: E402
from online_session_predictor import OnlineSessionPredictor  # noqa: E402
from trace_experiment_lib import duplicate_variant_marker  # noqa: E402


SCHEMA = "paste_repro.four_cell_summary"
VERSION = 1
JOINT_POLICY = "online_joint_pacer_v2"
JOINT_INSTALL_MARKER = f"[sched_policy_patch] installed policy={JOINT_POLICY}"
JOINT_RUNTIME_MARKER = "[sched_policy_patch:joint]"
PATCH_ERROR_MARKERS = (
    "scheduler policy patch error",
    "unknown VLLM_SCHED_POLICY",
    "policy=online_joint_pacer_v2 not installed",
)
WORKLOAD_MANIFEST_SCHEMA = "paste_repro.fixed_workload_bundle"
WORKLOAD_MANIFEST_VERSION = 1
SPLIT_MANIFEST_SCHEMA = "paste_repro.fixed_three_way_split"
SPLIT_MANIFEST_VERSION = 1
SCHED_REQUEST_PREFIX = "schedx"
SCHED_REQUEST_SUFFIX = "z"
STRESS_ROLE = "stress"
STRESS_UNIQUE_SOURCE_COUNT = 60
STRESS_LOAD_INSTANCE_COUNT = 120
STRESS_INSTANCES_PER_SOURCE = 2
STRESS_EVIDENCE_ROLE = "stress120_load_sensitivity_not_independent_not_final"


def _stress_evidence_role(load_instance_count: int) -> str:
    if load_instance_count == STRESS_LOAD_INSTANCE_COUNT:
        return STRESS_EVIDENCE_ROLE
    return (
        f"stress{load_instance_count}_"
        "load_sensitivity_not_independent_not_final"
    )


def _stress_replication_metadata(
    load_instance_count: int,
    unique_source_count: int = STRESS_UNIQUE_SOURCE_COUNT,
) -> dict[str, Any]:
    minimum, sources_with_extra = divmod(
        load_instance_count, unique_source_count
    )
    maximum = minimum + int(sources_with_extra > 0)
    result: dict[str, Any] = {
        "instances_per_source": minimum if sources_with_extra == 0 else None,
    }
    if load_instance_count != STRESS_LOAD_INSTANCE_COUNT:
        result.update(
            {
                "minimum_instances_per_source": minimum,
                "maximum_instances_per_source": maximum,
                "sources_with_one_extra_instance": sources_with_extra,
                "source_instances_are_balanced": True,
            }
        )
    return result

CELL_SPECS: dict[str, dict[str, str]] = {
    "A": {"name": "fcfs_none", "policy": "fcfs", "tool_overlap_mode": "none"},
    "B": {"name": "fcfs_learned", "policy": "fcfs", "tool_overlap_mode": "learned"},
    "C": {"name": "joint_none", "policy": JOINT_POLICY, "tool_overlap_mode": "none"},
    "D": {"name": "joint_learned", "policy": JOINT_POLICY, "tool_overlap_mode": "learned"},
}

STAT_NAMES = ("mean", "p50", "p95", "max")
METRIC_PATHS: tuple[tuple[str, ...], ...] = (
    *(("task_flow_time_s", statistic) for statistic in STAT_NAMES),
    ("task_makespan_s",),
    *(("request_latency_s", statistic) for statistic in STAT_NAMES),
    ("mean_queue_time_s",),
    ("instrumentation_wall_time_s",),
)


def repository_display_path(path: Path) -> str:
    """Render repository paths portably without changing path validation.

    Real reproduction artifacts are shown relative to the repository root.
    External paths, including temporary test fixtures, retain an explicit
    absolute POSIX representation so callers can still locate them.
    """

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile quantile must be in [0, 1]")
    # Hyndman-Fan type 7 (used by NumPy/R defaults).  In particular p50 is
    # the ordinary median for even-sized samples, rather than the lower item.
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("metric sample is empty")
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number != value:
        raise ValueError(f"{label} must be an integer")
    return number


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"run is incomplete; missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return payload


def _validate_embedded_checksum(
    payload: Mapping[str, Any],
    *,
    checksum_field: str,
    label: str,
) -> str:
    unsigned = dict(payload)
    supplied = unsigned.pop(checksum_field, None)
    computed = canonical_sha256(unsigned)
    if not isinstance(supplied, str) or supplied != computed:
        raise ValueError(f"{label} checksum mismatch")
    return supplied


def _resolve_manifest_file(
    manifest_path: Path,
    raw_path: Any,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"fixed workload manifest has invalid {label} path")
    resolved = (manifest_path.parent / raw_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"fixed workload manifest is missing {label}: {resolved}")
    return resolved


def _manifest_source_sequence(workload: Mapping[str, Any], label: str) -> list[str]:
    traces = workload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError(f"{label} prepared workload has no traces")
    result: list[str] = []
    for trace_number, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            raise ValueError(f"{label} trace {trace_number} is not an object")
        source = trace.get("source_trace")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{label} trace {trace_number} has invalid source_trace")
        result.append(Path(source).name)
    return result


def _validate_heldout_derivation(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> str:
    guards = manifest.get("contamination_guards")
    if not isinstance(guards, Mapping) or guards.get("heldout_is_not_new_final") is not True:
        raise ValueError("heldout manifest must label heldout as not a new final set")
    if guards.get("heldout_union_sessions") != (
        "tuning plus previously inspected final; calibration excluded"
    ):
        raise ValueError("heldout manifest has invalid union provenance label")
    derived_path = _resolve_manifest_file(
        manifest_path,
        manifest.get("derived_from_manifest"),
        "heldout parent manifest",
    )
    parent = _load_json_object(derived_path, "heldout parent manifest")
    if parent.get("schema") != WORKLOAD_MANIFEST_SCHEMA:
        raise ValueError("heldout parent manifest has unsupported schema")
    parent_sha256 = _validate_embedded_checksum(
        parent,
        checksum_field="manifest_sha256",
        label="heldout parent manifest",
    )
    if parent_sha256 != manifest.get("derived_from_manifest_sha256"):
        raise ValueError("heldout parent manifest checksum mismatch")
    for field in (
        "schema",
        "version",
        "fixed_split_manifest",
        "fixed_split_manifest_sha256",
        "calibration_only_mapper",
        "calibration_only_mapper_sha256",
        "source_mapper_artifact_sha256",
    ):
        if manifest.get(field) != parent.get(field):
            raise ValueError(f"heldout manifest changed parent field: {field}")
    current_workloads = manifest.get("workloads")
    parent_workloads = parent.get("workloads")
    current_inputs = manifest.get("four_cell_inputs")
    parent_inputs = parent.get("four_cell_inputs")
    if not isinstance(current_workloads, Mapping) or not isinstance(
        parent_workloads, Mapping
    ):
        raise ValueError("heldout/parent workload registries are invalid")
    if not isinstance(current_inputs, Mapping) or not isinstance(parent_inputs, Mapping):
        raise ValueError("heldout/parent four-cell registries are invalid")
    for base_role in ("calibration", "tuning", "final"):
        if current_workloads.get(base_role) != parent_workloads.get(base_role):
            raise ValueError(f"heldout manifest changed parent workload role: {base_role}")
    for base_role in ("tuning", "final"):
        if current_inputs.get(base_role) != parent_inputs.get(base_role):
            raise ValueError(f"heldout manifest changed parent cell role: {base_role}")
    current_parameters = manifest.get("parameters")
    parent_parameters = parent.get("parameters")
    if not isinstance(current_parameters, Mapping) or not isinstance(
        parent_parameters, Mapping
    ):
        raise ValueError("heldout/parent parameters are invalid")
    current_parameters_without_heldout = json.loads(_canonical_json(current_parameters))
    counts = current_parameters_without_heldout.get("target_trace_counts")
    if not isinstance(counts, dict):
        raise ValueError("heldout manifest target counts are invalid")
    counts.pop("heldout", None)
    if current_parameters_without_heldout != parent_parameters:
        raise ValueError("heldout manifest changed frozen parent parameters")
    return parent_sha256


def _validate_heldout_workload_union(
    *,
    mode: str,
    tuning: Mapping[str, Any],
    final: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> None:
    tuning_traces = tuning.get("traces")
    final_traces = final.get("traces")
    heldout_traces = heldout.get("traces")
    if not isinstance(tuning_traces, list) or not isinstance(final_traces, list):
        raise ValueError(f"heldout {mode} source traces are invalid")
    if not isinstance(heldout_traces, list):
        raise ValueError(f"heldout {mode} traces are invalid")
    source_traces = [*tuning_traces, *final_traces]
    if len(heldout_traces) != len(source_traces):
        raise ValueError(f"heldout {mode} trace count is not tuning plus final")
    for index, (source, observed) in enumerate(zip(source_traces, heldout_traces, strict=True)):
        if not isinstance(source, Mapping) or not isinstance(observed, Mapping):
            raise ValueError(f"heldout {mode} trace {index} is invalid")
        expected = dict(source)
        expected.update(
            {
                "trace_id": f"heldout_{index:03d}",
                "variant_index": index,
                "duplicated": False,
                "prefix_char": "",
            }
        )
        if dict(observed) != expected:
            raise ValueError(
                f"heldout {mode} trace {index} is not an exact retagged source trace"
            )
    tuning_meta = tuning.get("meta")
    observed_meta = heldout.get("meta")
    if not isinstance(tuning_meta, Mapping) or not isinstance(observed_meta, Mapping):
        raise ValueError(f"heldout {mode} metadata is invalid")
    expected_meta = dict(tuning_meta)
    expected_meta.update(
        {
            "source_trace_dir": None,
            "source_roles": ["tuning", "final"],
            "evidence_role": "heldout_load_sensitivity_not_untouched_final",
            "target_trace_count": len(source_traces),
            "duplicates_added": 0,
            "total_truncated_calls": sum(
                int(trace.get("truncated_calls", 0))
                for trace in source_traces
                if isinstance(trace, Mapping)
            ),
        }
    )
    if dict(observed_meta) != expected_meta:
        raise ValueError(f"heldout {mode} metadata is not the exact union metadata")


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_stress_derivation(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a balanced stress workload to a validated heldout60 parent."""

    parent_path = _resolve_manifest_file(
        manifest_path,
        manifest.get("stress_derived_from_manifest"),
        "stress parent heldout60 manifest",
    )
    parent = _load_json_object(parent_path, "stress parent heldout60 manifest")
    parent_sha256 = _validate_embedded_checksum(
        parent,
        checksum_field="manifest_sha256",
        label="stress parent heldout60 manifest",
    )
    if parent_sha256 != manifest.get("stress_derived_from_manifest_sha256"):
        raise ValueError("stress parent heldout60 manifest checksum mismatch")
    parent_verified = load_fixed_manifest(parent_path, "heldout")
    if parent_verified["manifest_sha256"] != parent_sha256:
        raise AssertionError("validated stress parent checksum changed")

    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("stress manifest has no parameters")
    target_counts = parameters.get("target_trace_counts")
    if not isinstance(target_counts, Mapping):
        raise ValueError("stress manifest has no target counts")
    target_count = _integer(target_counts.get(STRESS_ROLE), "stress target count")
    stress_seed = _integer(
        parameters.get("stress_duplicate_seed"), "stress duplicate seed"
    )

    definition = manifest.get("stress_definition")
    if not isinstance(definition, Mapping):
        raise ValueError("stress manifest has no stress_definition")
    unique_source_count = _integer(
        definition.get("unique_source_session_count"),
        "stress unique source session count",
    )
    if unique_source_count != STRESS_UNIQUE_SOURCE_COUNT:
        raise ValueError("stress definition must contain exactly 60 source sessions")
    load_instance_count = _integer(
        definition.get("load_instance_count"), "stress load instance count"
    )
    if load_instance_count < unique_source_count:
        raise ValueError(
            "stress load instance count must represent every heldout60 source"
        )
    if target_count != load_instance_count:
        raise ValueError("stress target count does not match stress definition")
    evidence_role = _stress_evidence_role(load_instance_count)
    replication_metadata = _stress_replication_metadata(
        load_instance_count, unique_source_count
    )
    duplicates_are_not_independent = load_instance_count > unique_source_count
    expected_definition = {
        "schema": "paste_repro.heldout_duplicate_stress",
        "version": 1,
        "evidence_role": evidence_role,
        "source_role": "heldout",
        "source_manifest_sha256": parent_sha256,
        "unique_source_session_count": unique_source_count,
        "load_instance_count": load_instance_count,
        **replication_metadata,
        "independent_sample_count": unique_source_count,
        "duplicates_are_not_independent": duplicates_are_not_independent,
        "is_final_evaluation": False,
        "calibration_excluded": True,
        "mapper_retrained": False,
        "prefix_marker_mode": "break_prefix",
        "duplicate_seed": stress_seed,
    }
    for field, expected in expected_definition.items():
        if definition.get(field) != expected:
            raise ValueError(f"stress definition mismatch: {field}")
    if not _is_sha256(definition.get("load_identity_sha256")):
        raise ValueError("stress definition has invalid load identity checksum")
    source_registry = definition.get("source_sessions")
    if not isinstance(source_registry, list):
        raise ValueError("stress definition has no source session registry")
    if definition.get("source_sessions_sha256") != canonical_sha256(source_registry):
        raise ValueError("stress source session registry checksum mismatch")

    guards = manifest.get("contamination_guards")
    expected_guards = {
        "stress_source_role": "heldout tuning+final union only",
        "stress_calibration_excluded": True,
        "stress_mapper_retrained": False,
        "stress_unique_source_sessions": unique_source_count,
        "stress_load_instances": load_instance_count,
        "stress_instances_per_source": replication_metadata[
            "instances_per_source"
        ],
        "stress_duplicates_are_not_independent": (
            duplicates_are_not_independent
        ),
        "stress_is_not_final": True,
        "stress_prefix_marker_mode": "break_prefix",
        "stress_evidence_role": evidence_role,
    }
    if load_instance_count != STRESS_LOAD_INSTANCE_COUNT:
        expected_guards.update(
            {
                "stress_minimum_instances_per_source": replication_metadata[
                    "minimum_instances_per_source"
                ],
                "stress_maximum_instances_per_source": replication_metadata[
                    "maximum_instances_per_source"
                ],
                "stress_sources_with_one_extra_instance": replication_metadata[
                    "sources_with_one_extra_instance"
                ],
                "stress_source_instances_are_balanced": True,
            }
        )
    if not isinstance(guards, Mapping):
        raise ValueError("stress manifest has no contamination guards")
    for field, expected in expected_guards.items():
        if guards.get(field) != expected:
            raise ValueError(f"stress manifest has invalid guard: {field}")

    # The child may add only the stress definition and stress workload/cells.
    # Removing exactly those additions must recover the heldout60 parent.
    stripped = json.loads(_canonical_json(manifest))
    stripped.pop("manifest_sha256", None)
    stripped.pop("stress_derived_from_manifest", None)
    stripped.pop("stress_derived_from_manifest_sha256", None)
    stripped.pop("stress_definition", None)
    stripped_workloads = stripped.get("workloads")
    stripped_inputs = stripped.get("four_cell_inputs")
    stripped_parameters = stripped.get("parameters")
    stripped_guards = stripped.get("contamination_guards")
    if not all(
        isinstance(value, dict)
        for value in (
            stripped_workloads,
            stripped_inputs,
            stripped_parameters,
            stripped_guards,
        )
    ):
        raise ValueError("stress manifest registries are invalid")
    stripped_workloads.pop(STRESS_ROLE, None)
    stripped_inputs.pop(STRESS_ROLE, None)
    stripped_counts = stripped_parameters.get("target_trace_counts")
    if not isinstance(stripped_counts, dict):
        raise ValueError("stress manifest target counts are invalid")
    stripped_counts.pop(STRESS_ROLE, None)
    stripped_parameters.pop("stress_duplicate_seed", None)
    for field in expected_guards:
        stripped_guards.pop(field, None)
    parent_unsigned = json.loads(_canonical_json(parent))
    parent_unsigned.pop("manifest_sha256", None)
    if stripped != parent_unsigned:
        raise ValueError("stress manifest changed frozen heldout60 parent content")

    authoritative_sources_by_mode: dict[str, dict[str, Path]] = {}
    parent_traces_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, cell in (("none", "A"), ("learned", "B")):
        parent_workload_path = parent_verified["bindings"][cell][
            "evaluation_workload"
        ]
        parent_workload = _load_json_object(
            parent_workload_path,
            f"stress parent heldout60/{mode} workload",
        )
        source_paths: dict[str, Path] = {}
        source_traces: dict[str, dict[str, Any]] = {}
        for trace in parent_workload["traces"]:
            source_path = Path(str(trace["source_trace"])).resolve()
            source_id = source_path.name
            if source_id in source_paths or not source_path.is_file():
                raise ValueError(
                    f"stress parent heldout60/{mode} source path is invalid: {source_id}"
                )
            source_paths[source_id] = source_path
            source_traces[source_id] = dict(trace)
        authoritative_sources_by_mode[mode] = source_paths
        parent_traces_by_mode[mode] = source_traces
    if authoritative_sources_by_mode["none"] != authoritative_sources_by_mode[
        "learned"
    ]:
        raise ValueError("stress parent heldout60 none/learned source paths differ")

    return {
        "parent_path": parent_path,
        "parent_manifest_sha256": parent_sha256,
        "heldout_parent_manifest_sha256": parent_verified[
            "heldout_parent_manifest_sha256"
        ],
        "stress_seed": stress_seed,
        "evidence_role": evidence_role,
        "unique_source_count": unique_source_count,
        "load_instance_count": load_instance_count,
        "replication_metadata": replication_metadata,
        "duplicates_are_not_independent": duplicates_are_not_independent,
        "definition": definition,
        "authoritative_sources": authoritative_sources_by_mode["none"],
        "parent_traces_by_mode": parent_traces_by_mode,
    }


def _stress_static_identity(workload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in workload["traces"]:
        rows.append(
            {
                "trace_id": trace["trace_id"],
                "source_session": Path(trace["source_trace"]).name,
                "variant_index": trace["variant_index"],
                "duplicated": trace["duplicated"],
                "prefix_char": trace["prefix_char"],
                "initial_delay_s": trace["initial_delay_s"],
                "requests": [
                    {
                        field: request[field]
                        for field in (
                            "call_index",
                            "prompt_tokens",
                            "original_prompt_tokens",
                            "target_output_tokens",
                            "max_tokens",
                            "truncated",
                            "messages",
                        )
                    }
                    for request in trace["requests"]
                ],
            }
        )
    return rows


def _validate_stress_workload(
    *,
    mode: str,
    workload: Mapping[str, Any],
    record: Mapping[str, Any],
    expected_source_ids: set[str],
    stress_seed: int,
    mapper_checksum: str,
    top_k: int,
    definition: Mapping[str, Any],
    authoritative_sources: Mapping[str, Path],
    parent_traces: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = workload.get("meta")
    traces = workload.get("traces")
    if not isinstance(metadata, Mapping) or not isinstance(traces, list):
        raise ValueError(f"stress {mode} workload shape is invalid")
    unique_source_count = _integer(
        definition.get("unique_source_session_count"),
        "stress unique source session count",
    )
    load_instance_count = _integer(
        definition.get("load_instance_count"), "stress load instance count"
    )
    evidence_role = str(definition.get("evidence_role", ""))
    replication_metadata = _stress_replication_metadata(
        load_instance_count, unique_source_count
    )
    duplicates_added = load_instance_count - unique_source_count
    duplicates_are_not_independent = duplicates_added > 0
    expected_metadata = {
        "source_trace_dir": None,
        "source_role": "heldout",
        "evidence_role": evidence_role,
        "target_trace_count": load_instance_count,
        "duplicates_added": duplicates_added,
        "prefix_marker_mode": "break_prefix",
        "unique_source_session_count": unique_source_count,
        "load_instance_count": load_instance_count,
        **replication_metadata,
        "independent_sample_count": unique_source_count,
        "duplicates_are_not_independent": duplicates_are_not_independent,
        "stress_is_not_final": True,
        "duplicate_seed": stress_seed,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"stress {mode} metadata mismatch: {field}")
    if len(traces) != load_instance_count:
        raise ValueError(
            f"stress {mode} must contain exactly {load_instance_count} load instances"
        )
    if len(expected_source_ids) != unique_source_count:
        raise ValueError(
            f"stress source registry must contain exactly {unique_source_count} sessions"
        )

    source_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    load_rows: list[dict[str, Any]] = []
    truncated_calls = 0
    sorted_sources = sorted(expected_source_ids)
    duplicate_cycle = list(sorted_sources)
    random.Random(stress_seed).shuffle(duplicate_cycle)
    expected_sequence = [
        *sorted_sources,
        *(
            duplicate_cycle[index % unique_source_count]
            for index in range(duplicates_added)
        ),
    ]
    for index, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            raise ValueError(f"stress {mode} trace {index} is invalid")
        variant = _integer(trace.get("variant_index"), "stress variant index")
        if variant != index or trace.get("trace_id") != f"trace_{index:03d}":
            raise ValueError(f"stress {mode} trace identities are not deterministic")
        duplicated = trace.get("duplicated")
        if type(duplicated) is not bool:
            raise ValueError(f"stress {mode} trace duplicated flag must be boolean")
        expected_duplicated = index >= unique_source_count
        if duplicated is not expected_duplicated:
            raise ValueError(f"stress {mode} original/duplicate positions are invalid")
        source_path = Path(str(trace.get("source_trace", ""))).resolve()
        source = source_path.name
        if source != expected_sequence[index] or source not in expected_source_ids:
            raise ValueError(
                f"stress {mode} source order is not deterministic and balanced"
            )
        if source_path != authoritative_sources.get(source):
            raise ValueError(
                f"stress {mode} source path does not match authoritative heldout60 source"
            )
        if not duplicated:
            expected_original = dict(parent_traces[source])
            expected_original.update(
                {
                    "trace_id": trace["trace_id"],
                    "source_trace": str(authoritative_sources[source]),
                    "variant_index": variant,
                    "duplicated": False,
                    "prefix_char": "",
                }
            )
            if dict(trace) != expected_original:
                raise ValueError(
                    f"stress {mode} original is not an exact heldout60 source replay"
                )
        marker = trace.get("prefix_char")
        expected_marker = (
            duplicate_variant_marker(index - unique_source_count)
            if duplicated
            else ""
        )
        if marker != expected_marker:
            raise ValueError(f"stress {mode} prefix marker is not deterministic")
        source_counts[source] += 1
        (duplicate_counts if duplicated else original_counts)[source] += 1
        requests = trace.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError(f"stress {mode} trace has no requests")
        for request in requests:
            if not isinstance(request, Mapping):
                raise ValueError(f"stress {mode} request is invalid")
            messages = request.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"stress {mode} request messages are invalid")
            if duplicated and (
                not messages
                or not isinstance(messages[0], Mapping)
                or dict(messages[0])
                != {"role": "system", "content": expected_marker}
            ):
                raise ValueError(
                    f"stress {mode} duplicate marker must be the exact leading system message"
                )
            truncated_calls += int(request.get("truncated") is True)
            if mode == "learned":
                if request.get("tool_prediction_artifact_sha256") != mapper_checksum:
                    raise ValueError("stress learned request mapper checksum mismatch")
                if _integer(
                    request.get("tool_prediction_top_k"),
                    "stress learned request top_k",
                ) != top_k:
                    raise ValueError("stress learned request top_k mismatch")
                candidates = request.get("tool_prediction_candidates")
                if not isinstance(candidates, list):
                    raise ValueError("stress learned request candidates are invalid")
                candidate_count = _integer(
                    request.get("tool_prediction_candidate_count"),
                    "stress learned request candidate count",
                )
                if candidate_count != len(candidates):
                    raise ValueError("stress learned request candidate count mismatch")
            elif any(str(key).startswith("tool_prediction_") for key in request):
                raise ValueError("stress none request unexpectedly has prediction fields")
        load_rows.append(
            {
                "trace_id": trace["trace_id"],
                "source_session": source,
                "variant_index": variant,
                "duplicated": duplicated,
                "prefix_char": marker,
            }
        )

    expected_original_counts = Counter(expected_sequence[:unique_source_count])
    expected_duplicate_counts = Counter(expected_sequence[unique_source_count:])
    if source_counts != Counter(expected_sequence):
        raise ValueError(f"stress {mode} source multiplicity is not balanced")
    if original_counts != expected_original_counts:
        raise ValueError(f"stress {mode} must contain one original per source")
    if duplicate_counts != expected_duplicate_counts:
        raise ValueError(f"stress {mode} duplicate source multiplicity is not balanced")
    if _integer(metadata.get("total_truncated_calls"), "stress truncated calls") != (
        truncated_calls
    ):
        raise ValueError(f"stress {mode} total truncation count mismatch")
    if mode == "learned":
        if metadata.get("tool_prediction_artifact_sha256") != mapper_checksum:
            raise ValueError("stress learned metadata mapper checksum mismatch")
        if _integer(metadata.get("tool_prediction_top_k"), "stress learned top_k") != top_k:
            raise ValueError("stress learned metadata top_k mismatch")
    elif any(str(key).startswith("tool_prediction_") for key in metadata):
        raise ValueError("stress none metadata unexpectedly has prediction fields")

    sequence = [row["source_session"] for row in load_rows]
    load_identity = canonical_sha256(load_rows)
    record_fields = {
        "trace_count": load_instance_count,
        "load_instance_count": load_instance_count,
        "unique_source_session_count": unique_source_count,
        "independent_sample_count": unique_source_count,
        **replication_metadata,
        "source_sequence_sha256": canonical_sha256(sequence),
        "source_set_sha256": canonical_sha256(sorted(set(sequence))),
        "load_identity_sha256": load_identity,
    }
    for field, expected in record_fields.items():
        if record.get(field) != expected:
            raise ValueError(f"stress {mode} workload record mismatch: {field}")
    if definition.get("load_identity_sha256") != load_identity:
        raise ValueError(f"stress {mode} load identity differs from stress definition")
    return {
        "source_sequence": sequence,
        "load_identity_sha256": load_identity,
        "static_identity": _stress_static_identity(workload),
    }


def load_fixed_manifest(manifest_path: Path, role: str) -> dict[str, Any]:
    """Load and cryptographically bind a tuning/final/heldout/stress definition."""

    path = manifest_path.resolve()
    manifest = _load_json_object(path, "fixed workload manifest")
    if manifest.get("schema") != WORKLOAD_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported fixed workload manifest schema: {path}")
    if manifest.get("version") != WORKLOAD_MANIFEST_VERSION:
        raise ValueError(f"unsupported fixed workload manifest version: {path}")
    manifest_sha256 = _validate_embedded_checksum(
        manifest,
        checksum_field="manifest_sha256",
        label="fixed workload manifest",
    )
    if role not in {"tuning", "final", "heldout", STRESS_ROLE}:
        raise ValueError(
            "fixed workload role must be tuning, final, heldout, or stress"
        )
    stress_derivation = (
        _validate_stress_derivation(path, manifest)
        if role == STRESS_ROLE
        else None
    )
    heldout_parent_sha256 = (
        _validate_heldout_derivation(path, manifest)
        if role == "heldout"
        else (
            stress_derivation["heldout_parent_manifest_sha256"]
            if stress_derivation is not None
            else None
        )
    )

    split_path = _resolve_manifest_file(
        path,
        manifest.get("fixed_split_manifest"),
        "fixed split manifest",
    )
    split = _load_json_object(split_path, "fixed split manifest")
    if split.get("schema") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError("unsupported fixed split manifest schema")
    if split.get("version") != SPLIT_MANIFEST_VERSION:
        raise ValueError("unsupported fixed split manifest version")
    split_sha256 = _validate_embedded_checksum(
        split,
        checksum_field="manifest_sha256",
        label="fixed split manifest",
    )
    if split_sha256 != manifest.get("fixed_split_manifest_sha256"):
        raise ValueError("fixed workload/split manifest checksum mismatch")

    split_role_sessions: dict[str, list[str]] = {}
    for split_role in ("calibration", "tuning", "final"):
        entries = split.get(f"{split_role}_sessions")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"fixed split manifest has no {split_role} sessions")
        sessions: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"fixed split {split_role} session is not an object")
            session_id = entry.get("session_id")
            checksum = entry.get("sha256")
            if (
                not isinstance(session_id, str)
                or not session_id
                or Path(session_id).name != session_id
            ):
                raise ValueError(f"fixed split has unsafe {split_role} session id")
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise ValueError(f"fixed split has invalid checksum for {session_id}")
            sessions.append(session_id)
        if len(sessions) != len(set(sessions)):
            raise ValueError(f"fixed split has duplicate {split_role} sessions")
        split_role_sessions[split_role] = sessions
    if any(
        set(split_role_sessions[left]) & set(split_role_sessions[right])
        for left, right in (
            ("calibration", "tuning"),
            ("calibration", "final"),
            ("tuning", "final"),
        )
    ):
        raise ValueError("fixed split roles overlap")
    if stress_derivation is not None:
        expected_stress_registry = sorted(
            [
                {
                    "session_id": str(entry["session_id"]),
                    "sha256": str(entry["sha256"]),
                }
                for entry in [
                    *split["tuning_sessions"],
                    *split["final_sessions"],
                ]
            ],
            key=lambda entry: str(entry["session_id"]),
        )
        definition = stress_derivation["definition"]
        if definition.get("source_sessions") != expected_stress_registry:
            raise ValueError(
                "stress source registry is not exactly the tuning+final split registry"
            )
        for entry in expected_stress_registry:
            source_path = stress_derivation["authoritative_sources"].get(
                entry["session_id"]
            )
            if source_path is None or file_sha256(source_path) != entry["sha256"]:
                raise ValueError(
                    f"stress authoritative source checksum mismatch: {entry['session_id']}"
                )
        if set(split_role_sessions["calibration"]) & {
            str(entry["session_id"]) for entry in expected_stress_registry
        }:
            raise ValueError("calibration sessions leaked into stress source registry")

    mapper_path = _resolve_manifest_file(
        path,
        manifest.get("calibration_only_mapper"),
        "calibration-only mapper",
    )
    _, mapper_artifact = load_artifact(mapper_path)
    mapper_checksum = mapper_artifact.get("artifact_sha256")
    if mapper_checksum != manifest.get("calibration_only_mapper_sha256"):
        raise ValueError("fixed workload mapper checksum mismatch")
    mapper_split = mapper_artifact.get("training_split")
    if not isinstance(mapper_split, Mapping):
        raise ValueError("calibration-only mapper has no training split")
    for split_role, artifact_fields in {
        "calibration": ("train_sessions", "calibration_sessions"),
        "tuning": ("tuning_sessions",),
        "final": ("final_sessions",),
    }.items():
        expected_entries = {
            (str(entry["session_id"]), str(entry["sha256"]))
            for entry in split[f"{split_role}_sessions"]
        }
        for artifact_field in artifact_fields:
            raw_entries = mapper_split.get(artifact_field)
            if not isinstance(raw_entries, list):
                raise ValueError(
                    f"calibration-only mapper has no {artifact_field} registry"
                )
            observed_entries = {
                (str(entry.get("session_id")), str(entry.get("sha256")))
                for entry in raw_entries
                if isinstance(entry, Mapping)
            }
            if len(observed_entries) != len(raw_entries) or observed_entries != expected_entries:
                raise ValueError(
                    f"calibration-only mapper {artifact_field} does not match fixed split"
                )

    parameters = manifest.get("parameters")
    workloads = manifest.get("workloads")
    four_cell_inputs = manifest.get("four_cell_inputs")
    if not isinstance(parameters, Mapping):
        raise ValueError("fixed workload manifest has no parameters")
    if not isinstance(workloads, Mapping):
        raise ValueError("fixed workload manifest has no workloads")
    if not isinstance(four_cell_inputs, Mapping):
        raise ValueError("fixed workload manifest has no four_cell_inputs")
    role_inputs = four_cell_inputs.get(role)
    if not isinstance(role_inputs, Mapping):
        raise ValueError(f"fixed workload manifest has no {role} cell inputs")

    target_counts = parameters.get("target_trace_counts")
    if not isinstance(target_counts, Mapping):
        raise ValueError("fixed workload manifest has no target trace counts")
    speedup = _finite_nonnegative(parameters.get("speedup"), "manifest speedup")
    if speedup <= 0:
        raise ValueError("manifest speedup must be positive")
    expected_count = _integer(target_counts.get(role), f"manifest {role} trace count")
    top_k = _integer(parameters.get("tool_prediction_top_k"), "manifest top_k")
    if expected_count <= 0 or top_k <= 0:
        raise ValueError("manifest trace count and top_k must be positive")
    shared_parameter_fields = {
        "max_model_len": _integer(parameters.get("max_model_len"), "manifest max_model_len"),
        "max_output_tokens_cap": _integer(
            parameters.get("max_output_tokens_cap"),
            "manifest max_output_tokens_cap",
        ),
        "output_token_buffer": _integer(
            parameters.get("output_token_buffer"),
            "manifest output_token_buffer",
        ),
        "min_output_tokens_floor": _integer(
            parameters.get("min_output_tokens_floor"),
            "manifest min_output_tokens_floor",
        ),
    }
    base_duplicate_seed = _integer(parameters.get("seed"), "manifest seed")
    if any(value <= 0 for value in (*shared_parameter_fields.values(), base_duplicate_seed)):
        raise ValueError("fixed workload manifest numeric parameters must be positive")
    for split_role in ("calibration", "tuning", "final"):
        role_count = _integer(
            target_counts.get(split_role),
            f"manifest {split_role} trace count",
        )
        if role_count <= 0 or len(split_role_sessions[split_role]) != role_count:
            raise ValueError(f"fixed split/{split_role} count does not match parameters")

    workload_bindings: dict[tuple[str, str], dict[str, Any]] = {}
    workload_roles = (
        ("calibration", "tuning", "final", "heldout")
        if role == "heldout"
        else ("calibration", STRESS_ROLE)
        if role == STRESS_ROLE
        else ("calibration", role)
    )
    for workload_role in workload_roles:
        role_records = workloads.get(workload_role)
        if not isinstance(role_records, Mapping):
            raise ValueError(f"fixed workload manifest has no {workload_role} workloads")
        for mode in ("none", "learned"):
            record = role_records.get(mode)
            if not isinstance(record, Mapping):
                raise ValueError(f"fixed workload manifest has no {workload_role}/{mode}")
            workload_path = _resolve_manifest_file(
                path,
                record.get("prepared_workload"),
                f"{workload_role}/{mode} prepared workload",
            )
            expected_sha256 = record.get("prepared_workload_sha256")
            if (
                not isinstance(expected_sha256, str)
                or file_sha256(workload_path) != expected_sha256
            ):
                raise ValueError(f"fixed {workload_role}/{mode} workload checksum mismatch")
            summary_path = _resolve_manifest_file(
                path,
                record.get("workload_summary"),
                f"{workload_role}/{mode} workload summary",
            )
            expected_summary_sha256 = record.get("workload_summary_sha256")
            if (
                not isinstance(expected_summary_sha256, str)
                or file_sha256(summary_path) != expected_summary_sha256
            ):
                raise ValueError(
                    f"fixed {workload_role}/{mode} workload summary checksum mismatch"
                )
            workload = _load_json_object(
                workload_path,
                f"fixed {workload_role}/{mode} prepared workload",
            )
            metadata = workload.get("meta")
            if not isinstance(metadata, Mapping) or metadata.get("tool_overlap_mode") != mode:
                raise ValueError(f"fixed {workload_role}/{mode} workload mode mismatch")
            if record.get("tool_overlap_mode") != mode:
                raise ValueError(f"fixed {workload_role}/{mode} record mode mismatch")
            workload_parameter_fields = {
                **shared_parameter_fields,
                "duplicate_seed": (
                    stress_derivation["stress_seed"]
                    if workload_role == STRESS_ROLE
                    else base_duplicate_seed
                ),
            }
            for field, expected_value in workload_parameter_fields.items():
                if _integer(
                    metadata.get(field),
                    f"fixed {workload_role}/{mode} {field}",
                ) != expected_value:
                    raise ValueError(
                        f"fixed {workload_role}/{mode} configuration mismatch: {field}"
                    )
            if not _numbers_close(metadata.get("tool_overlap_efficiency"), 1.0):
                raise ValueError(
                    f"fixed {workload_role}/{mode} overlap efficiency is not one"
                )
            sequence = _manifest_source_sequence(
                workload,
                f"fixed {workload_role}/{mode}",
            )
            if canonical_sha256(sequence) != record.get("source_sequence_sha256"):
                raise ValueError(f"fixed {workload_role}/{mode} source sequence mismatch")
            expected_sessions = (
                set(split_role_sessions["tuning"])
                | set(split_role_sessions["final"])
                if workload_role in {"heldout", STRESS_ROLE}
                else set(split_role_sessions[workload_role])
            )
            if set(sequence) != expected_sessions:
                raise ValueError(f"fixed {workload_role}/{mode} sessions do not match split role")
            if len(sequence) != _integer(record.get("trace_count"), "manifest trace count"):
                raise ValueError(f"fixed {workload_role}/{mode} trace count mismatch")
            if len(set(sequence)) != _integer(
                record.get("unique_source_session_count"),
                "manifest unique source session count",
            ):
                raise ValueError(
                    f"fixed {workload_role}/{mode} unique source count mismatch"
                )
            if len(sequence) != _integer(
                target_counts.get(workload_role),
                f"manifest {workload_role} target count",
            ):
                raise ValueError(f"fixed {workload_role}/{mode} target count mismatch")
            if workload_role != STRESS_ROLE and len(set(sequence)) != len(sequence):
                raise ValueError(f"fixed {workload_role}/{mode} is not one trace per session")
            if mode == "learned":
                if record.get("mapper_artifact_sha256") != mapper_checksum:
                    raise ValueError(f"fixed {workload_role}/learned mapper mismatch")
                if _integer(record.get("tool_prediction_top_k"), "record top_k") != top_k:
                    raise ValueError(f"fixed {workload_role}/learned top_k mismatch")
            workload_bindings[(workload_role, mode)] = {
                "path": workload_path,
                "sha256": expected_sha256,
                "content_sha256": canonical_sha256(workload),
                "source_sequence": sequence,
                "workload": workload,
            }
        if (
            workload_bindings[(workload_role, "none")]["source_sequence"]
            != workload_bindings[(workload_role, "learned")]["source_sequence"]
        ):
            raise ValueError(
                f"fixed {workload_role} none/learned source sequence mismatch"
            )

    if role == "heldout":
        for mode in ("none", "learned"):
            expected_sequence = [
                *workload_bindings[("tuning", mode)]["source_sequence"],
                *workload_bindings[("final", mode)]["source_sequence"],
            ]
            if workload_bindings[("heldout", mode)][
                "source_sequence"
            ] != expected_sequence:
                raise ValueError(
                    f"heldout {mode} source sequence is not tuning followed by final"
                )
            _validate_heldout_workload_union(
                mode=mode,
                tuning=workload_bindings[("tuning", mode)]["workload"],
                final=workload_bindings[("final", mode)]["workload"],
                heldout=workload_bindings[("heldout", mode)]["workload"],
            )
    stress_validation: dict[str, dict[str, Any]] | None = None
    if role == STRESS_ROLE:
        if stress_derivation is None:
            raise AssertionError("stress derivation was not validated")
        expected_stress_sources = set(split_role_sessions["tuning"]) | set(
            split_role_sessions["final"]
        )
        stress_validation = {
            mode: _validate_stress_workload(
                mode=mode,
                workload=workload_bindings[(STRESS_ROLE, mode)]["workload"],
                record=workloads[STRESS_ROLE][mode],
                expected_source_ids=expected_stress_sources,
                stress_seed=stress_derivation["stress_seed"],
                mapper_checksum=str(mapper_checksum),
                top_k=top_k,
                definition=stress_derivation["definition"],
                authoritative_sources=stress_derivation["authoritative_sources"],
                parent_traces=stress_derivation["parent_traces_by_mode"][mode],
            )
            for mode in ("none", "learned")
        }
        if stress_validation["none"]["source_sequence"] != stress_validation[
            "learned"
        ]["source_sequence"]:
            raise ValueError("stress none/learned source sequences differ")
        if stress_validation["none"]["load_identity_sha256"] != stress_validation[
            "learned"
        ]["load_identity_sha256"]:
            raise ValueError("stress none/learned load identities differ")
        if stress_validation["none"]["static_identity"] != stress_validation[
            "learned"
        ]["static_identity"]:
            raise ValueError("stress none/learned static request identities differ")

    bindings: dict[str, dict[str, Any]] = {}
    expected_input_names = {spec["name"] for spec in CELL_SPECS.values()}
    if set(role_inputs) != expected_input_names:
        raise ValueError(f"fixed {role} manifest must define exactly four cells")
    for cell, spec in CELL_SPECS.items():
        cell_input = role_inputs.get(spec["name"])
        if not isinstance(cell_input, Mapping):
            raise ValueError(f"fixed {role} manifest is missing cell {spec['name']}")
        mode = spec["tool_overlap_mode"]
        if cell_input.get("policy") != spec["policy"]:
            raise ValueError(f"fixed {role}/{spec['name']} policy mismatch")
        if cell_input.get("tool_overlap_mode") != mode:
            raise ValueError(f"fixed {role}/{spec['name']} overlap mode mismatch")
        evaluation = _resolve_manifest_file(
            path,
            cell_input.get("evaluation_workload"),
            f"{role}/{spec['name']} evaluation workload",
        )
        calibration = _resolve_manifest_file(
            path,
            cell_input.get("online_calibration_workload"),
            f"{role}/{spec['name']} calibration workload",
        )
        if evaluation != workload_bindings[(role, mode)]["path"]:
            raise ValueError(f"fixed {role}/{spec['name']} evaluation binding mismatch")
        if calibration != workload_bindings[("calibration", mode)]["path"]:
            raise ValueError(f"fixed {role}/{spec['name']} calibration binding mismatch")
        bindings[cell] = {
            "role": role,
            "policy": spec["policy"],
            "tool_overlap_mode": mode,
            "evaluation_workload": evaluation,
            "evaluation_workload_sha256": workload_bindings[(role, mode)]["sha256"],
            "evaluation_workload_content_sha256": workload_bindings[(role, mode)][
                "content_sha256"
            ],
            "calibration_workload": calibration,
            "calibration_workload_sha256": workload_bindings[("calibration", mode)][
                "sha256"
            ],
            "speedup": speedup,
            "trace_count": expected_count,
            "mapper_artifact_sha256": mapper_checksum if mode == "learned" else None,
            "tool_prediction_top_k": top_k if mode == "learned" else None,
        }
    verified_manifest = {
        "path": path,
        "manifest_sha256": manifest_sha256,
        "fixed_split_manifest_sha256": split_sha256,
        "role": role,
        "evidence_role": (
            stress_derivation["evidence_role"]
            if stress_derivation is not None
            else "heldout_load_sensitivity_not_untouched_final"
            if role == "heldout"
            else role
        ),
        "heldout_parent_manifest_sha256": heldout_parent_sha256,
        "stress_parent_manifest_sha256": (
            stress_derivation["parent_manifest_sha256"]
            if stress_derivation is not None
            else None
        ),
        "load_instance_count": (
            stress_derivation["load_instance_count"]
            if stress_derivation is not None
            else expected_count
        ),
        "independent_source_session_count": (
            stress_derivation["unique_source_count"]
            if stress_derivation is not None
            else expected_count
        ),
        "instances_per_source": (
            stress_derivation["replication_metadata"]["instances_per_source"]
            if stress_derivation is not None
            else 1
        ),
        "duplicates_are_not_independent": (
            stress_derivation["duplicates_are_not_independent"]
            if stress_derivation is not None
            else False
        ),
        "is_final_evaluation": role == "final",
        "prefix_marker_mode": "break_prefix" if role == STRESS_ROLE else None,
        "calibration_excluded": role in {"tuning", "final", "heldout", STRESS_ROLE},
        "mapper_artifact": mapper_path,
        "bindings": bindings,
    }
    if (
        stress_derivation is not None
        and stress_derivation["load_instance_count"] != STRESS_LOAD_INSTANCE_COUNT
    ):
        verified_manifest.update(stress_derivation["replication_metadata"])
    return verified_manifest


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"run is incomplete; missing request events: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: event must be an object")
        events.append(payload)
    if not events:
        raise ValueError(f"run has no request events: {path.parent}")
    return events


def _request_key(trace_id: Any, call_index: Any, label: str) -> tuple[str, int]:
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError(f"{label} has invalid trace_id")
    index = _integer(call_index, f"{label} call_index")
    return trace_id, index


def _static_workload_identity(
    workload: Mapping[str, Any],
    run_path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[str, float]]:
    traces = workload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError(f"prepared workload has no traces: {run_path}")
    identity_rows: list[dict[str, Any]] = []
    requests_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    initial_delays: dict[str, float] = {}
    for trace_number, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            raise ValueError(f"workload trace {trace_number} is not an object: {run_path}")
        trace_id = trace.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"workload trace {trace_number} has invalid trace_id: {run_path}")
        if trace_id in initial_delays:
            raise ValueError(f"duplicate workload trace_id {trace_id}: {run_path}")
        source_trace = trace.get("source_trace")
        if not isinstance(source_trace, str) or not source_trace:
            raise ValueError(f"workload trace {trace_id} has invalid source_trace: {run_path}")
        initial_delay = _finite_nonnegative(
            trace.get("initial_delay_s", 0.0),
            f"workload trace {trace_id} initial_delay_s",
        )
        initial_delays[trace_id] = initial_delay
        requests = trace.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError(f"workload trace {trace_id} has no requests: {run_path}")
        trace_identity = {
            "trace_id": trace_id,
            "source_session": Path(source_trace).name,
            "variant_index": trace.get("variant_index"),
            "duplicated": bool(trace.get("duplicated", False)),
            "prefix_char": trace.get("prefix_char", ""),
            "initial_delay_s": initial_delay,
        }
        for request_number, request in enumerate(requests):
            if not isinstance(request, Mapping):
                raise ValueError(
                    f"workload request {trace_id}/{request_number} is not an object: {run_path}"
                )
            key = _request_key(
                trace_id,
                request.get("call_index"),
                f"workload request {trace_id}/{request_number}",
            )
            if key in requests_by_key:
                raise ValueError(
                    f"duplicate workload request identity {key[0]}/{key[1]}: {run_path}"
                )
            static_request = {
                **trace_identity,
                "call_index": key[1],
                "prompt_tokens": request.get("prompt_tokens"),
                "original_prompt_tokens": request.get("original_prompt_tokens"),
                "target_output_tokens": request.get("target_output_tokens"),
                "max_tokens": request.get("max_tokens"),
                "truncated": bool(request.get("truncated", False)),
                "messages": request.get("messages"),
            }
            if not isinstance(static_request["messages"], list):
                raise ValueError(f"workload request {key} has invalid messages: {run_path}")
            for field in (
                "prompt_tokens",
                "original_prompt_tokens",
                "target_output_tokens",
                "max_tokens",
            ):
                _integer(static_request[field], f"workload request {key} {field}")
            stored = dict(request)
            stored["_static_identity"] = static_request
            stored["_source_session"] = trace_identity["source_session"]
            stored["_duplicated"] = trace_identity["duplicated"]
            stored["_prefix_char"] = trace_identity["prefix_char"]
            stored["_request_index"] = request_number
            requests_by_key[key] = stored
            identity_rows.append(static_request)
    identity_rows.sort(key=lambda row: (row["trace_id"], row["call_index"]))
    return identity_rows, requests_by_key, initial_delays


def _event_identity_map(
    events: Sequence[Mapping[str, Any]],
    run_path: Path,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for event_number, event in enumerate(events):
        key = _request_key(
            event.get("trace_id"),
            event.get("call_index"),
            f"event {event_number}",
        )
        if key in by_key:
            raise ValueError(f"duplicate request event identity {key[0]}/{key[1]}: {run_path}")
        by_key[key] = event
    return by_key


def _expected_online_tail_metadata(
    requests_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    predictor: OnlineSessionPredictor,
    speedup: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    by_trace: dict[str, list[tuple[tuple[str, int], Mapping[str, Any]]]] = {}
    for key, request in requests_by_key.items():
        by_trace.setdefault(key[0], []).append((key, request))
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for trace_id, rows in by_trace.items():
        rows.sort(key=lambda row: int(row[1]["_request_index"]))
        observed_waits: list[float] = []
        for request_index, (key, request) in enumerate(rows):
            if request["_request_index"] != request_index:
                raise ValueError(f"trace {trace_id} has non-contiguous request positions")
            if key[1] > 0:
                observed_waits.append(
                    _finite_nonnegative(
                        request.get("wait_after_prev_s"),
                        f"request {key} observed tool wait",
                    )
                )
            prediction = predictor.predict(
                current_call_index=key[1],
                past_tool_waits_s=observed_waits,
            )
            expected[key] = {
                "n": request_index + 1 + prediction.remaining_calls,
                "rc": prediction.remaining_calls,
                "nw": prediction.next_tool_wait_s / speedup,
                "nwc": predictor.next_tool_wait_reliability,
                "rtw": prediction.remaining_tool_wait_s / speedup,
            }
    return expected


def _numbers_close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-7)
    except (TypeError, ValueError):
        return False


def _decode_online_request_id(
    request_id: Any,
    *,
    key: tuple[str, int],
    request: Mapping[str, Any],
    event: Mapping[str, Any],
    run_path: Path,
) -> dict[str, Any]:
    if (
        not isinstance(request_id, str)
        or not request_id.startswith(SCHED_REQUEST_PREFIX)
        or not request_id.endswith(SCHED_REQUEST_SUFFIX)
    ):
        raise ValueError(f"request {key} has invalid scheduler request_id: {run_path}")
    encoded = request_id[len(SCHED_REQUEST_PREFIX) : -len(SCHED_REQUEST_SUFFIX)]
    try:
        raw_json = bytes.fromhex(encoded).decode("utf-8")
        metadata = json.loads(raw_json)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"request {key} scheduler request_id cannot be decoded: {run_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"request {key} scheduler metadata is not an object: {run_path}")
    if metadata.get("ms") != "online":
        raise ValueError(f"request {key} request_id is not online metadata: {run_path}")
    allowed_fields = {
        "t",
        "c",
        "i",
        "n",
        "rc",
        "nw",
        "nwc",
        "rtw",
        "pt",
        "mt",
        "ms",
        "po",
        "npo",
    }
    if set(metadata) - allowed_fields:
        raise ValueError(f"request {key} scheduler metadata has unexpected fields: {run_path}")

    next_wait_reliability: float | None = None
    if "nwc" in metadata:
        next_wait_reliability = _finite_nonnegative(
            metadata["nwc"], f"request {key} scheduler nwc"
        )
        if next_wait_reliability > 1.0:
            raise ValueError(
                f"request {key} scheduler nwc must be at most 1: {run_path}"
            )
    canonical_id = (
        SCHED_REQUEST_PREFIX
        + _canonical_json(metadata).encode("utf-8").hex()
        + SCHED_REQUEST_SUFFIX
    )
    if canonical_id != request_id:
        raise ValueError(f"request {key} scheduler request_id is not canonical: {run_path}")
    static = request["_static_identity"]
    exact = {
        "t": key[0],
        "c": key[1],
        "i": request["_request_index"],
        "pt": static["prompt_tokens"],
        "mt": static["max_tokens"],
    }
    for field, expected in exact.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"request {key} scheduler metadata {field} mismatch: {run_path}"
            )
    total_calls = _integer(metadata.get("n"), f"request {key} scheduler n")
    remaining_calls = _integer(metadata.get("rc"), f"request {key} scheduler rc")
    if remaining_calls < 0 or total_calls != request["_request_index"] + 1 + remaining_calls:
        raise ValueError(f"request {key} scheduler call-count metadata is invalid: {run_path}")
    next_wait = _finite_nonnegative(metadata.get("nw"), f"request {key} scheduler nw")
    remaining_wait = _finite_nonnegative(
        metadata.get("rtw"), f"request {key} scheduler rtw"
    )
    event_fields = {
        "scheduled_total_calls": total_calls,
        "scheduled_remaining_calls_after": remaining_calls,
        "scheduled_nw": next_wait,
        "scheduled_rtw": remaining_wait,
    }
    for field, expected in event_fields.items():
        if not _numbers_close(event.get(field), expected):
            raise ValueError(f"request {key} event/{field} mismatch: {run_path}")
    event_reliability = event.get("scheduled_nw_reliability")
    if next_wait_reliability is None:
        if event_reliability is not None:
            raise ValueError(
                f"request {key} event has reliability without scheduler nwc: {run_path}"
            )
    else:
        checked_event_reliability = _finite_nonnegative(
            event_reliability,
            f"request {key} event scheduled_nw_reliability",
        )
        if checked_event_reliability > 1.0:
            raise ValueError(
                f"request {key} event scheduled_nw_reliability must be at most 1: "
                f"{run_path}"
            )
        if not _numbers_close(checked_event_reliability, next_wait_reliability):
            raise ValueError(
                f"request {key} event/reliability mismatch: {run_path}"
            )
    predicted_output = _integer(metadata.get("po"), f"request {key} scheduler po")
    if predicted_output <= 0 or predicted_output > int(static["max_tokens"]):
        raise ValueError(f"request {key} scheduler output prediction is invalid: {run_path}")
    if _integer(event.get("po_predicted"), f"request {key} po_predicted") != predicted_output:
        raise ValueError(f"request {key} event/output prediction mismatch: {run_path}")
    if remaining_calls > 0:
        if _integer(metadata.get("npo"), f"request {key} scheduler npo") != predicted_output:
            raise ValueError(f"request {key} scheduler npo mismatch: {run_path}")
    elif "npo" in metadata:
        raise ValueError(f"request {key} terminal metadata unexpectedly has npo: {run_path}")
    if event.get("nw_source") != "predicted":
        raise ValueError(f"request {key} next-wait metadata is not predicted: {run_path}")
    for field in (
        "oracle_next_tool_wait_s",
        "oracle_remaining_tool_wait_s",
        "oracle_remaining_calls_after",
        "oracle_total_calls",
    ):
        if event.get(field) is not None:
            raise ValueError(f"request {key} exposes oracle field {field}: {run_path}")
    return metadata


def _validate_tool_metadata(
    key: tuple[str, int],
    request: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    mode: str,
    mapper_checksum: str | None,
    mapper_top_k: int | None,
    speedup: float,
    run_path: Path,
) -> None:
    if request.get("tool_overlap_mode") != mode or event.get("tool_overlap_mode") != mode:
        raise ValueError(f"request {key} tool overlap mode mismatch: {run_path}")
    original_wait = _finite_nonnegative(
        request.get("wait_after_prev_original_s"), f"request {key} original wait"
    )
    realized_wait = _finite_nonnegative(
        request.get("wait_after_prev_s"), f"request {key} realized wait"
    )
    saved_wait = _finite_nonnegative(
        request.get("tool_overlap_saved_s"), f"request {key} saved wait"
    )
    overlap_window = _finite_nonnegative(
        request.get("tool_overlap_window_s"), f"request {key} overlap window"
    )
    if not _numbers_close(realized_wait, max(0.0, original_wait - saved_wait)):
        raise ValueError(f"request {key} wait/saving accounting mismatch: {run_path}")
    expected_scheduled = realized_wait if key[1] == 0 else realized_wait / speedup
    event_comparisons = {
        "scheduled_wait_original_s": original_wait,
        "scheduled_wait_s": expected_scheduled,
        "tool_overlap_saved_s": saved_wait,
        "tool_overlap_window_s": overlap_window,
    }
    for field, expected in event_comparisons.items():
        if not _numbers_close(event.get(field), expected):
            raise ValueError(f"request {key} event/{field} mismatch: {run_path}")
    if event.get("tool_wait_mode") != "sleep":
        raise ValueError(f"request {key} did not execute tool waits by sleeping: {run_path}")

    if mode == "none":
        if saved_wait != 0.0 or not _numbers_close(realized_wait, original_wait):
            raise ValueError(f"none request {key} unexpectedly changes tool wait: {run_path}")
        absent_defaults = {
            "tool_prediction_candidate_count": 0,
            "tool_prediction_exact_hits": 0,
            "tool_prediction_waste": 0,
            "tool_prediction_top_k": 0,
        }
        for field, default in absent_defaults.items():
            if request.get(field, default) != default or event.get(field, default) != default:
                raise ValueError(f"none request {key} has prediction field {field}: {run_path}")
        if request.get("tool_prediction_artifact_sha256", "") not in {None, ""} or event.get(
            "tool_prediction_artifact_sha256", ""
        ) not in {None, ""}:
            raise ValueError(f"none request {key} unexpectedly binds a mapper: {run_path}")
        if request.get("tool_prediction_candidates", []) not in (None, []):
            raise ValueError(f"none request {key} has prediction candidates: {run_path}")
        return

    if mapper_checksum is None or mapper_top_k is None:
        raise AssertionError("learned validation requires a mapper binding")
    if request.get("tool_prediction_artifact_sha256") != mapper_checksum or event.get(
        "tool_prediction_artifact_sha256"
    ) != mapper_checksum:
        raise ValueError(f"learned request {key} mapper checksum mismatch: {run_path}")
    if request.get("tool_prediction_top_k") != mapper_top_k or event.get(
        "tool_prediction_top_k"
    ) != mapper_top_k:
        raise ValueError(f"learned request {key} top_k mismatch: {run_path}")
    candidates = request.get("tool_prediction_candidates")
    if not isinstance(candidates, list) or any(
        not isinstance(candidate, str) or not candidate for candidate in candidates
    ):
        raise ValueError(f"learned request {key} has invalid candidates: {run_path}")
    counts: dict[str, int] = {}
    for field in (
        "tool_prediction_candidate_count",
        "tool_prediction_exact_hits",
        "tool_prediction_waste",
    ):
        counts[field] = _integer(request.get(field), f"learned request {key} {field}")
        if counts[field] < 0 or event.get(field) != counts[field]:
            raise ValueError(f"learned request {key} invalid event/{field}: {run_path}")
    if counts["tool_prediction_candidate_count"] != len(candidates):
        raise ValueError(f"learned request {key} candidate count mismatch: {run_path}")
    if counts["tool_prediction_waste"] > len(candidates):
        raise ValueError(f"learned request {key} waste exceeds candidates: {run_path}")
    if saved_wait > overlap_window + 1e-7 or saved_wait > original_wait + 1e-7:
        raise ValueError(f"learned request {key} saved wait exceeds causal bound: {run_path}")


def _validate_event_against_workload(
    key: tuple[str, int],
    event: Mapping[str, Any],
    request: Mapping[str, Any],
    run_path: Path,
    *,
    mode: str,
    mapper_checksum: str | None,
    mapper_top_k: int | None,
    speedup: float,
    expected_online_tail: Mapping[str, Any],
) -> None:
    if not bool(event.get("ok")):
        raise ValueError(f"non-OK request event {key[0]}/{key[1]}: {run_path}")
    if event.get("metadata_source") != "online":
        raise ValueError(f"request event did not use online metadata {key}: {run_path}")
    static = request["_static_identity"]
    if Path(str(event.get("source_trace", ""))).name != request["_source_session"]:
        raise ValueError(f"event source session mismatch for request {key}: {run_path}")
    comparisons = {
        "duplicated": (bool(event.get("duplicated", False)), request["_duplicated"]),
        "prefix_char": (event.get("prefix_char", ""), request["_prefix_char"]),
        "prompt_tokens": (event.get("prompt_tokens"), static["prompt_tokens"]),
        "target_output_tokens": (
            event.get("target_output_tokens"),
            static["target_output_tokens"],
        ),
        "max_tokens": (event.get("max_tokens"), static["max_tokens"]),
        "truncated": (bool(event.get("truncated", False)), static["truncated"]),
    }
    for field, (observed, expected) in comparisons.items():
        if observed != expected:
            raise ValueError(f"event/workload {field} mismatch for request {key}: {run_path}")
    decoded_metadata = _decode_online_request_id(
        event.get("request_id"),
        key=key,
        request=request,
        event=event,
        run_path=run_path,
    )
    expected_fields = ["n", "rc", "nw", "rtw"]
    # Older causal runs predate reliability metadata.  Validate it against the
    # calibration-only backtest whenever it is present, while retaining the
    # ability to summarize those immutable legacy artifacts.
    if "nwc" in decoded_metadata:
        expected_fields.append("nwc")
    for field in expected_fields:
        if not _numbers_close(decoded_metadata.get(field), expected_online_tail.get(field)):
            raise ValueError(
                f"request {key} online metadata differs from calibration-only "
                f"causal prediction for {field}: {run_path}"
            )
    _validate_tool_metadata(
        key,
        request,
        event,
        mode=mode,
        mapper_checksum=mapper_checksum,
        mapper_top_k=mapper_top_k,
        speedup=speedup,
        run_path=run_path,
    )
    start = _finite_nonnegative(
        event.get("request_start_offset_s"), f"request {key} request_start_offset_s"
    )
    end = _finite_nonnegative(
        event.get("request_end_offset_s"), f"request {key} request_end_offset_s"
    )
    latency = _finite_nonnegative(event.get("latency_s"), f"request {key} latency_s")
    if end + 1e-9 < start or not _numbers_close(latency, end - start):
        raise ValueError(f"request {key} latency does not equal end minus start: {run_path}")


def _validate_retry_accounting(
    summary: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    run_path: Path,
) -> dict[str, int]:
    """Recompute retry totals from events and reject summary drift."""

    configured = _integer(
        summary.get("configured_max_request_attempts"),
        "configured_max_request_attempts",
    )
    if configured <= 0:
        raise ValueError(f"configured max request attempts is not positive: {run_path}")

    attempts_total = 0
    retried_requests = 0
    retry_successes = 0
    ambiguous_retries = 0
    final_failures = 0
    for event_number, event in enumerate(events):
        attempts = _integer(event.get("attempts"), "event attempts")
        history = event.get("attempt_history")
        if not isinstance(history, list) or not history or len(history) != attempts:
            raise ValueError(f"event attempt history mismatch at {event_number}: {run_path}")
        if attempts > configured:
            raise ValueError(f"event exceeds configured attempts at {event_number}: {run_path}")
        for attempt_number, record in enumerate(history, 1):
            if not isinstance(record, Mapping) or _integer(
                record.get("attempt"), "attempt history index"
            ) != attempt_number:
                raise ValueError(f"malformed attempt history at {event_number}: {run_path}")
            required = {
                "transport",
                "outcome",
                "http_status",
                "error_type",
                "error",
                "duration_s",
                "retryable",
                "will_retry",
                "retry_backoff_s",
                "delivery_ambiguous",
            }
            if not required.issubset(record):
                raise ValueError(f"incomplete attempt history at {event_number}: {run_path}")
            _finite_nonnegative(
                record.get("duration_s"),
                f"event {event_number} attempt duration",
            )
            backoff_s = _finite_nonnegative(
                record.get("retry_backoff_s"),
                f"event {event_number} retry backoff",
            )
            will_retry = record.get("will_retry")
            if (
                type(will_retry) is not bool
                or type(record.get("retryable")) is not bool
                or type(record.get("delivery_ambiguous")) is not bool
            ):
                raise ValueError(f"invalid retry marker at {event_number}: {run_path}")
            if (will_retry and backoff_s <= 0.0) or (
                not will_retry and backoff_s != 0.0
            ):
                raise ValueError(f"invalid retry backoff at {event_number}: {run_path}")
            is_last = attempt_number == attempts
            if is_last and will_retry:
                raise ValueError(f"final attempt retries at {event_number}: {run_path}")
            if not is_last:
                if (
                    not will_retry
                    or record.get("retryable") is not True
                    or record.get("outcome") != "transport_error"
                ):
                    raise ValueError(f"non-transport retry at {event_number}: {run_path}")
                ambiguous_retries += int(record.get("delivery_ambiguous") is True)

        final = history[-1]
        event_ok = bool(event.get("ok"))
        history_ok = final.get("outcome") == "success" and final.get(
            "http_status"
        ) == 200
        if event_ok != history_ok or event.get("http_status") != final.get(
            "http_status"
        ):
            raise ValueError(f"event final attempt mismatch at {event_number}: {run_path}")
        attempts_total += attempts
        retried_requests += int(attempts > 1)
        retry_successes += int(attempts > 1 and event_ok)
        final_failures += int(not event_ok)

    observed = {
        "configured_max_request_attempts": configured,
        "requests_total": len(events),
        "request_attempts_total": attempts_total,
        "retry_count": attempts_total - len(events),
        "retried_request_count": retried_requests,
        "retry_success_count": retry_successes,
        "ambiguous_retry_count": ambiguous_retries,
        "final_failure_count": final_failures,
    }
    for field, expected in observed.items():
        if _integer(summary.get(field), f"summary {field}") != expected:
            raise ValueError(f"summary/event retry mismatch for {field}: {run_path}")
    if observed["final_failure_count"] != _integer(
        summary.get("requests_failed"), "requests_failed"
    ):
        raise ValueError(f"summary failure/retry mismatch: {run_path}")
    return observed


def _scheduler_evidence(run_path: Path, policy: str) -> dict[str, int]:
    server_log = run_path / "server.log"
    if not server_log.is_file():
        raise FileNotFoundError(f"run is missing server.log: {run_path}")
    text = server_log.read_text(encoding="utf-8", errors="replace")
    if "vLLM API server version" not in text:
        raise ValueError(f"server.log lacks vLLM server startup evidence: {run_path}")
    install_lines = [
        line for line in text.splitlines() if "[sched_policy_patch] installed policy=" in line
    ]
    policy_install_lines = [
        line for line in install_lines if f"installed policy={policy} " in line
    ]
    evidence = {
        "install_lines": len(policy_install_lines),
        "install_v1_true_lines": sum("v1=True" in line for line in policy_install_lines),
        "runtime_joint_lines": text.count(JOINT_RUNTIME_MARKER),
        "patch_error_lines": sum(text.count(marker) for marker in PATCH_ERROR_MARKERS),
        "unexpected_policy_install_lines": len(install_lines) - len(policy_install_lines),
        "server_startup_lines": text.count("vLLM API server version"),
    }
    if evidence["patch_error_lines"] != 0 or evidence["unexpected_policy_install_lines"] != 0:
        raise ValueError(f"run has scheduler patch error/policy mismatch evidence: {run_path}")
    if policy == JOINT_POLICY:
        if (
            evidence["install_lines"] <= 0
            or evidence["install_v1_true_lines"] != evidence["install_lines"]
            or evidence["runtime_joint_lines"] <= 0
        ):
            raise ValueError(f"joint run lacks clean v1 install/runtime hook evidence: {run_path}")
    elif (
        evidence["install_lines"] != 0
        or evidence["runtime_joint_lines"] != 0
        or install_lines
    ):
        raise ValueError(f"FCFS run contains scheduler patch/runtime evidence: {run_path}")
    return evidence


def load_run(
    run_path: Path,
    cell: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = run_path.resolve()
    spec = CELL_SPECS[cell]
    if expected_binding.get("policy") != spec["policy"] or expected_binding.get(
        "tool_overlap_mode"
    ) != spec["tool_overlap_mode"]:
        raise ValueError(f"fixed manifest binding does not match cell {cell}")
    summary = _load_json_object(path / "summary.json", "summary")
    workload_path = path / "prepared_workload.json"
    workload = _load_json_object(workload_path, "prepared workload")
    events = _load_events(path / "request_events.jsonl")

    workload_sha256 = file_sha256(workload_path)
    if canonical_sha256(workload) != expected_binding.get(
        "evaluation_workload_content_sha256"
    ):
        raise ValueError(f"run workload is not the checksummed fixed-role workload: {path}")

    if _integer(summary.get("requests_failed", -1), "requests_failed") != 0:
        raise ValueError(f"run has failed requests: {path}")
    if summary.get("metadata_source") != "online" or summary.get(
        "scheduler_metadata_mode"
    ) != "online":
        raise ValueError(f"run did not use online scheduler metadata: {path}")
    if _integer(summary.get("requests_total", -1), "requests_total") != len(events):
        raise ValueError(f"summary request total does not match events: {path}")
    if _integer(summary.get("requests_success", -1), "requests_success") != len(events):
        raise ValueError(f"summary success total does not match events: {path}")
    retry_accounting = _validate_retry_accounting(summary, events, path)
    speedup = _finite_nonnegative(summary.get("speedup"), "summary speedup")
    if speedup <= 0 or not _numbers_close(speedup, expected_binding.get("speedup")):
        raise ValueError(f"run speedup does not match fixed manifest: {path}")
    if summary.get("tool_wait_mode") != "sleep":
        raise ValueError(f"run did not execute configured tool waits: {path}")
    calibration_raw = summary.get("scheduler_calibration_workload")
    if (
        not isinstance(calibration_raw, str)
        or Path(calibration_raw).resolve()
        != expected_binding.get("calibration_workload")
    ):
        raise ValueError(f"run online calibration workload does not match manifest: {path}")
    scheduler_environment = summary.get("scheduler_environment")
    if not isinstance(scheduler_environment, Mapping):
        raise ValueError(f"run summary has no scheduler environment: {path}")
    if scheduler_environment.get("VLLM_SCHED_POLICY") != spec["policy"]:
        raise ValueError(f"run scheduler policy does not match cell {cell}: {path}")

    metadata = workload.get("meta")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"prepared workload has no metadata: {path}")
    expected_mode = spec["tool_overlap_mode"]
    if metadata.get("tool_overlap_mode") != expected_mode:
        raise ValueError(f"cell {cell} prepared workload mode mismatch: {path}")
    workload_summary = summary.get("workload")
    if not isinstance(workload_summary, Mapping):
        raise ValueError(f"summary has no workload section: {path}")
    if workload_summary.get("tool_overlap_mode") != expected_mode:
        raise ValueError(f"cell {cell} summary workload mode mismatch: {path}")
    if _integer(workload_summary.get("request_count", -1), "workload request_count") != len(
        events
    ):
        raise ValueError(f"summary workload request count does not match events: {path}")
    expected_trace_count = _integer(expected_binding.get("trace_count"), "binding trace count")
    if (
        _integer(workload_summary.get("trace_count", -1), "workload trace_count")
        != expected_trace_count
    ):
        raise ValueError(f"summary workload trace count does not match manifest: {path}")
    max_active_traces = _integer(summary.get("max_active_traces"), "max_active_traces")
    if max_active_traces <= 0:
        raise ValueError(f"run max_active_traces must be positive: {path}")

    identity_rows, requests_by_key, initial_delays = _static_workload_identity(
        workload, path
    )
    if len(initial_delays) != expected_trace_count:
        raise ValueError(f"prepared workload trace count does not match manifest: {path}")
    online_predictor = OnlineSessionPredictor.from_workload(
        expected_binding["calibration_workload"]
    )
    expected_online_tails = _expected_online_tail_metadata(
        requests_by_key,
        predictor=online_predictor,
        speedup=speedup,
    )
    mapper_checksum: str | None = None
    mapper_top_k: int | None = None
    if expected_mode == "learned":
        raw_checksum = metadata.get("tool_prediction_artifact_sha256")
        if (
            not isinstance(raw_checksum, str)
            or len(raw_checksum) != 64
            or any(character not in "0123456789abcdef" for character in raw_checksum)
        ):
            raise ValueError(f"learned workload has invalid mapper checksum: {path}")
        mapper_checksum = raw_checksum
        mapper_top_k = _integer(
            metadata.get("tool_prediction_top_k"), "tool_prediction_top_k"
        )
        if (
            mapper_checksum != expected_binding.get("mapper_artifact_sha256")
            or mapper_top_k != expected_binding.get("tool_prediction_top_k")
            or mapper_top_k <= 0
        ):
            raise ValueError(f"learned workload mapper/top_k differs from manifest: {path}")
        summary_prediction = workload_summary.get("tool_prediction")
        if not isinstance(summary_prediction, Mapping):
            raise ValueError(f"learned summary has no tool_prediction section: {path}")
        if summary_prediction.get("artifact_sha256") != mapper_checksum:
            raise ValueError(f"summary/workload mapper checksum mismatch: {path}")
        if _integer(summary_prediction.get("top_k"), "summary top_k") != mapper_top_k:
            raise ValueError(f"summary/workload top_k mismatch: {path}")
    else:
        if any(str(key).startswith("tool_prediction_") for key in metadata):
            raise ValueError(f"none workload metadata unexpectedly binds prediction: {path}")
        if "tool_prediction" in workload_summary:
            raise ValueError(f"none summary unexpectedly contains tool_prediction: {path}")

    events_by_key = _event_identity_map(events, path)
    workload_keys = set(requests_by_key)
    event_keys = set(events_by_key)
    if event_keys != workload_keys:
        missing = sorted(workload_keys - event_keys)[:5]
        unexpected = sorted(event_keys - workload_keys)[:5]
        raise ValueError(
            f"request event identities do not exactly match workload: {path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in sorted(workload_keys):
        _validate_event_against_workload(
            key,
            events_by_key[key],
            requests_by_key[key],
            path,
            mode=expected_mode,
            mapper_checksum=mapper_checksum,
            mapper_top_k=mapper_top_k,
            speedup=speedup,
            expected_online_tail=expected_online_tails[key],
        )
    if expected_mode == "learned":
        summary_prediction = workload_summary["tool_prediction"]
        prediction_fields = {
            "candidate_count": "tool_prediction_candidate_count",
            "exact_hits": "tool_prediction_exact_hits",
            "waste": "tool_prediction_waste",
        }
        for summary_field, request_field in prediction_fields.items():
            expected_total = sum(
                _integer(request.get(request_field), f"request {request_field}")
                for request in requests_by_key.values()
            )
            if _integer(
                summary_prediction.get(summary_field),
                f"summary prediction {summary_field}",
            ) != expected_total:
                raise ValueError(
                    f"summary learned prediction {summary_field} does not match workload: {path}"
                )

    completion_by_trace: dict[str, float] = {}
    request_latencies: list[float] = []
    for key, event in events_by_key.items():
        end_offset = _finite_nonnegative(
            event.get("request_end_offset_s"), f"request {key} end offset"
        )
        completion_by_trace[key[0]] = max(completion_by_trace.get(key[0], 0.0), end_offset)
        request_latencies.append(
            _finite_nonnegative(event.get("latency_s"), f"request {key} latency")
        )
    task_flows: list[float] = []
    for trace_id, initial_delay in initial_delays.items():
        completion = completion_by_trace.get(trace_id)
        if completion is None:
            raise ValueError(f"trace has no completion event {trace_id}: {path}")
        flow = completion - initial_delay
        if flow < -1e-9:
            raise ValueError(f"task completed before its arrival {trace_id}: {path}")
        task_flows.append(max(0.0, flow))

    queue_time = _finite_nonnegative(summary.get("avg_queue_time_s"), "avg_queue_time_s")
    instrumentation_wall = _finite_nonnegative(
        summary.get("experiment_wall_time_s"), "experiment_wall_time_s"
    )
    task_makespan = max(completion_by_trace.values())
    if instrumentation_wall + 1e-9 < task_makespan:
        raise ValueError(f"instrumentation wall ends before the last request event: {path}")
    hook_evidence = _scheduler_evidence(path, spec["policy"])
    source_mapping = {
        row["trace_id"]: row["source_session"]
        for row in identity_rows
    }
    public_metrics = {
        "run_name": path.name,
        "run_path": repository_display_path(path),
        "policy": spec["policy"],
        "tool_overlap_mode": expected_mode,
        "trace_count": len(initial_delays),
        "source_session_count": len(set(source_mapping.values())),
        "request_count": len(events),
        "task_flow_time_s": _stats(task_flows),
        "task_makespan_s": task_makespan,
        "request_latency_s": _stats(request_latencies),
        "mean_queue_time_s": queue_time,
        "instrumentation_wall_time_s": instrumentation_wall,
        "instrumentation_overhang_s": instrumentation_wall - task_makespan,
        "prepared_workload_sha256": workload_sha256,
        "fixed_role": expected_binding["role"],
        "speedup": speedup,
        "max_active_traces": max_active_traces,
        "tool_wait_mode": "sleep",
        "configured_max_request_attempts": retry_accounting[
            "configured_max_request_attempts"
        ],
        "retry_accounting": retry_accounting,
        "scheduler_configuration": {
            key: scheduler_environment[key]
            for key in sorted(scheduler_environment)
            if key != "VLLM_SCHED_POLICY"
        },
        "scheduler_calibration_workload_sha256": expected_binding[
            "calibration_workload_sha256"
        ],
        "request_identity_sha256": canonical_sha256(identity_rows),
        "source_sessions_sha256": canonical_sha256(source_mapping),
        "mapper_artifact_sha256": mapper_checksum,
        "tool_prediction_top_k": mapper_top_k,
        "scheduler_evidence": hook_evidence,
    }
    return {
        "public": public_metrics,
        "identity_rows": identity_rows,
        "source_mapping": source_mapping,
        "mapper_checksum": mapper_checksum,
        "mapper_top_k": mapper_top_k,
    }


def _get_metric(payload: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _set_metric(payload: dict[str, Any], path: Sequence[str], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def aggregate_cell(runs: Sequence[dict[str, Any]], cell: str) -> dict[str, Any]:
    if not runs:
        raise ValueError(f"cell {cell} has no runs")
    aggregate: dict[str, Any] = {
        "name": CELL_SPECS[cell]["name"],
        "policy": CELL_SPECS[cell]["policy"],
        "tool_overlap_mode": CELL_SPECS[cell]["tool_overlap_mode"],
        "run_count": len(runs),
        "trace_count": runs[0]["public"]["trace_count"],
        "source_session_count": runs[0]["public"]["source_session_count"],
        "request_count": runs[0]["public"]["request_count"],
    }
    configured_attempts = {
        run["public"]["configured_max_request_attempts"] for run in runs
    }
    if len(configured_attempts) != 1:
        raise ValueError(f"cell {cell} has inconsistent max request attempts")
    aggregate["retry_accounting"] = {
        "configured_max_request_attempts": next(iter(configured_attempts)),
        **{
            field: sum(run["public"]["retry_accounting"][field] for run in runs)
            for field in (
                "requests_total",
                "request_attempts_total",
                "retry_count",
                "retried_request_count",
                "retry_success_count",
                "ambiguous_retry_count",
                "final_failure_count",
            )
        },
    }
    for path in METRIC_PATHS:
        _set_metric(
            aggregate,
            path,
            statistics.fmean(_get_metric(run["public"], path) for run in runs),
        )
    aggregate["runs"] = [run["public"] for run in runs]
    return aggregate


def _pair_effect(
    baseline: Mapping[str, Any],
    optimized: Mapping[str, Any],
    *,
    definition: str,
) -> dict[str, Any]:
    absolute: dict[str, Any] = {}
    relative: dict[str, Any] = {}
    for path in METRIC_PATHS:
        before = _get_metric(baseline, path)
        after = _get_metric(optimized, path)
        reduction = before - after
        _set_metric(absolute, path, reduction)
        _set_metric(relative, path, reduction / before if before else None)
    return {
        "definition": definition,
        "absolute_reduction": absolute,
        "relative_reduction": relative,
    }


def _interaction_effect(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    absolute: dict[str, Any] = {}
    relative_to_a: dict[str, Any] = {}
    for path in METRIC_PATHS:
        a = _get_metric(cells["A"], path)
        b = _get_metric(cells["B"], path)
        c = _get_metric(cells["C"], path)
        d = _get_metric(cells["D"], path)
        interaction = (b - d) - (a - c)
        _set_metric(absolute, path, interaction)
        _set_metric(relative_to_a, path, interaction / a if a else None)
    return {
        "definition": "(B-D) - (A-C), equivalently (C-D) - (A-B)",
        "absolute_reduction": absolute,
        "relative_to_a": relative_to_a,
    }


def compute_effects(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "tool_only_A_to_B": _pair_effect(
            cells["A"], cells["B"], definition="A - B: learned tool overlap under FCFS"
        ),
        "scheduler_none_A_to_C": _pair_effect(
            cells["A"], cells["C"], definition="A - C: joint scheduler without overlap"
        ),
        "scheduler_increment_B_to_D": _pair_effect(
            cells["B"], cells["D"], definition="B - D: joint scheduler on learned path"
        ),
        "full_A_to_D": _pair_effect(
            cells["A"], cells["D"], definition="A - D: full joint + learned path"
        ),
        "tool_under_joint_C_to_D": _pair_effect(
            cells["C"], cells["D"], definition="C - D: learned tool overlap under joint"
        ),
        "interaction": _interaction_effect(cells),
    }


def summarize_four_cell(
    path_groups: Mapping[str, Sequence[Path]],
    *,
    manifest_path: Path,
    role: str,
) -> dict[str, Any]:
    if set(path_groups) != set(CELL_SPECS):
        raise ValueError(f"four-cell inputs must contain exactly {sorted(CELL_SPECS)}")
    counts = {cell: len(path_groups[cell]) for cell in CELL_SPECS}
    if any(count <= 0 for count in counts.values()) or len(set(counts.values())) != 1:
        raise ValueError(f"all four cells must have the same positive replicate count: {counts}")
    resolved_paths = [
        path.resolve()
        for cell in CELL_SPECS
        for path in path_groups[cell]
    ]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("run directories must be unique across all four cells and replicates")

    fixed_manifest = load_fixed_manifest(manifest_path, role)
    loaded = {
        cell: [
            load_run(path, cell, fixed_manifest["bindings"][cell])
            for path in path_groups[cell]
        ]
        for cell in CELL_SPECS
    }
    reference = loaded["A"][0]
    for cell, runs in loaded.items():
        for replicate_index, run in enumerate(runs, 1):
            if run["identity_rows"] != reference["identity_rows"]:
                raise ValueError(
                    "request identity/prompt/messages/max_tokens mismatch: "
                    f"cell={cell}, replicate={replicate_index}"
                )
            if run["source_mapping"] != reference["source_mapping"]:
                raise ValueError(
                    f"source sessions mismatch: cell={cell}, replicate={replicate_index}"
                )
            reference_config = reference["public"]
            for field in (
                "speedup",
                "max_active_traces",
                "tool_wait_mode",
                "configured_max_request_attempts",
                "scheduler_configuration",
            ):
                if run["public"][field] != reference_config[field]:
                    raise ValueError(
                        f"run configuration mismatch for {field}: "
                        f"cell={cell}, replicate={replicate_index}"
                    )

    learned_checksums = {
        run["mapper_checksum"]
        for cell in ("B", "D")
        for run in loaded[cell]
    }
    if len(learned_checksums) != 1 or None in learned_checksums:
        raise ValueError("learned cells do not use one identical mapper artifact checksum")
    learned_top_ks = {
        run["mapper_top_k"]
        for cell in ("B", "D")
        for run in loaded[cell]
    }
    if len(learned_top_ks) != 1:
        raise ValueError("learned cells do not use one identical tool_prediction_top_k")

    cells = {cell: aggregate_cell(loaded[cell], cell) for cell in CELL_SPECS}
    replicate_effects = []
    for replicate_index in range(counts["A"]):
        replicate_cells = {
            cell: loaded[cell][replicate_index]["public"]
            for cell in CELL_SPECS
        }
        replicate_effects.append(
            {
                "replicate": replicate_index + 1,
                "run_names": {
                    cell: replicate_cells[cell]["run_name"] for cell in CELL_SPECS
                },
                "effects": compute_effects(replicate_cells),
            }
        )

    source_sessions = sorted(set(reference["source_mapping"].values()))
    if len(source_sessions) != fixed_manifest["independent_source_session_count"]:
        raise ValueError("run source-session count differs from fixed manifest")
    retry_totals = {
        field: sum(cells[cell]["retry_accounting"][field] for cell in CELL_SPECS)
        for field in (
            "requests_total",
            "request_attempts_total",
            "retry_count",
            "retried_request_count",
            "retry_success_count",
            "ambiguous_retry_count",
            "final_failure_count",
        )
    }
    if role == STRESS_ROLE:
        stress_count = fixed_manifest["load_instance_count"]
        stress_status = (
            f"stress{stress_count}_"
            "four_cell_load_sensitivity_not_independent_not_final"
        )
        if stress_count == STRESS_LOAD_INSTANCE_COUNT:
            stress_interpretation = (
                " Stress task percentiles describe 120 load instances derived from "
                "60 unique heldout sources (one original plus one deterministic "
                "break-prefix duplicate each); the duplicates are not independent, "
                "and this is load-sensitivity evidence rather than a final evaluation."
            )
        else:
            source_count = fixed_manifest["independent_source_session_count"]
            exact_instances = fixed_manifest["instances_per_source"]
            if exact_instances is not None:
                multiplicity = (
                    f"{exact_instances} deterministic instances per source: one "
                    f"original plus {exact_instances - 1} break-prefix duplicates"
                )
            else:
                minimum = fixed_manifest["minimum_instances_per_source"]
                maximum = fixed_manifest["maximum_instances_per_source"]
                extra_sources = fixed_manifest["sources_with_one_extra_instance"]
                multiplicity = (
                    f"a balanced {minimum}-{maximum} instances per source "
                    f"({extra_sources} sources receive one extra instance)"
                )
            stress_interpretation = (
                f" Stress task percentiles describe {stress_count} load instances "
                f"derived from {source_count} unique heldout sources ({multiplicity}); "
                "the duplicates are not independent, and this is load-sensitivity "
                "evidence rather than a final evaluation."
            )
    else:
        stress_status = "functional_four_cell_not_full_paper_reproduction"
        stress_interpretation = ""
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": stress_status,
        "comparison_invariants": {
            "replicate_count_per_cell": counts["A"],
            "fixed_workload_manifest": repository_display_path(
                fixed_manifest["path"]
            ),
            "fixed_workload_manifest_sha256": fixed_manifest["manifest_sha256"],
            "fixed_split_manifest_sha256": fixed_manifest[
                "fixed_split_manifest_sha256"
            ],
            "fixed_role": role,
            "evidence_role": fixed_manifest["evidence_role"],
            "heldout_parent_manifest_sha256": fixed_manifest[
                "heldout_parent_manifest_sha256"
            ],
            "stress_parent_manifest_sha256": fixed_manifest[
                "stress_parent_manifest_sha256"
            ],
            "trace_count": reference["public"]["trace_count"],
            "load_instance_count": fixed_manifest["load_instance_count"],
            "source_session_count": len(source_sessions),
            "independent_source_session_count": fixed_manifest[
                "independent_source_session_count"
            ],
            "instances_per_source": fixed_manifest["instances_per_source"],
            "duplicates_are_not_independent": fixed_manifest[
                "duplicates_are_not_independent"
            ],
            "is_final_evaluation": fixed_manifest["is_final_evaluation"],
            "calibration_excluded": fixed_manifest["calibration_excluded"],
            "prefix_marker_mode": fixed_manifest["prefix_marker_mode"],
            "request_count": reference["public"]["request_count"],
            "request_identity_sha256": reference["public"]["request_identity_sha256"],
            "source_sessions_sha256": canonical_sha256(source_sessions),
            "learned_mapper_artifact_sha256": next(iter(learned_checksums)),
            "tool_prediction_top_k": next(iter(learned_top_ks)),
            "metadata_source": "online",
            "configured_max_request_attempts": reference["public"][
                "configured_max_request_attempts"
            ],
            "all_requests_finally_succeeded": (
                retry_totals["final_failure_count"] == 0
            ),
            "all_requests_succeeded_exactly_once": (
                retry_totals["final_failure_count"] == 0
                and retry_totals["request_attempts_total"]
                == retry_totals["requests_total"]
                and retry_totals["retry_count"] == 0
                and retry_totals["ambiguous_retry_count"] == 0
            ),
            "retry_accounting": retry_totals,
            "joint_hook_install_and_runtime_verified": True,
            "task_flow_time_definition": (
                "max request_end_offset_s per trace minus prepared initial_delay_s"
            ),
            "task_makespan_definition": "max request_end_offset_s across all events",
            "instrumentation_wall_is_diagnostic_only": True,
            "task_distribution_unit": (
                "load_instance_not_independent_source_session"
                if role == STRESS_ROLE
                else "source_session"
            ),
        },
        "cells": cells,
        "effects": compute_effects(cells),
        "replicate_effects": replicate_effects,
        "interpretation": (
            "Every effect is baseline minus optimized, so positive means a lower "
            "latency, makespan, queue time, or instrumentation wall. Interaction is "
            "positive when the joint scheduler and learned overlap provide additional "
            "reduction together beyond their FCFS/no-overlap effects."
            + stress_interpretation
        ),
    }
    if role == STRESS_ROLE and stress_count != STRESS_LOAD_INSTANCE_COUNT:
        for field in (
            "minimum_instances_per_source",
            "maximum_instances_per_source",
            "sources_with_one_extra_instance",
            "source_instances_are_balanced",
        ):
            result["comparison_invariants"][field] = fixed_manifest[field]
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize matched A/B/C/D four-cell live trace runs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=("tuning", "final", "heldout", STRESS_ROLE),
        required=True,
    )
    parser.add_argument("--a", "--fcfs-none", dest="A", type=Path, action="append", required=True)
    parser.add_argument(
        "--b", "--fcfs-learned", dest="B", type=Path, action="append", required=True
    )
    parser.add_argument("--c", "--joint-none", dest="C", type=Path, action="append", required=True)
    parser.add_argument(
        "--d", "--joint-learned", dest="D", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path_groups = {cell: getattr(args, cell) for cell in CELL_SPECS}
    result = summarize_four_cell(
        path_groups,
        manifest_path=args.manifest,
        role=args.role,
    )
    if args.output is not None:
        write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
