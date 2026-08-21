#!/usr/bin/env python3
"""Build a deterministic, balanced stress workload from heldout60.

The default remains the original 120-instance workload: every heldout source
session appears exactly twice, once unchanged and once as a deterministic
duplicate with a unique prefix marker.  Larger counts repeat a seeded shuffle
of all 60 sources, so source multiplicities differ by at most one while every
duplicate keeps a globally unique prefix marker.  Both none and learned
workloads are rebuilt from the authoritative raw traces through the existing
preparation path, so prompt token counts, truncation, output budgets, waits,
and learned predictions are recomputed rather than copied.  Calibration data
is never admitted and the calibration-only mapper is loaded, never retrained.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import tempfile
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
RUNNER_SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (REPRODUCTION_ROOT, RUNNER_SCRIPT_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_fixed_three_way_split import canonical_sha256, file_sha256  # noqa: E402
from paste_repro.mapper import load_artifact, write_json_atomic  # noqa: E402
from prepare_fixed_workloads import _default_tokenizer  # noqa: E402
from summarize_four_cell import load_fixed_manifest  # noqa: E402
from trace_experiment_lib import (  # noqa: E402
    _build_chat_tokens,
    duplicate_variant_marker,
    prepare_trace_workload,
    summarize_workload,
)


STRESS_ROLE = "stress"
UNIQUE_SOURCE_COUNT = 60
LOAD_INSTANCE_COUNT = 120
INSTANCES_PER_SOURCE = 2
EVIDENCE_ROLE = "stress120_load_sensitivity_not_independent_not_final"


def _normalize_load_instance_count(load_instance_count: int) -> int:
    if isinstance(load_instance_count, bool) or not isinstance(load_instance_count, int):
        raise ValueError("load_instance_count must be an integer")
    if load_instance_count < UNIQUE_SOURCE_COUNT:
        raise ValueError(
            f"load_instance_count must be at least {UNIQUE_SOURCE_COUNT} so every "
            "heldout source is represented"
        )
    return load_instance_count


def _evidence_role(load_instance_count: int) -> str:
    if load_instance_count == LOAD_INSTANCE_COUNT:
        return EVIDENCE_ROLE
    return (
        f"stress{load_instance_count}_"
        "load_sensitivity_not_independent_not_final"
    )


def _replication_metadata(load_instance_count: int) -> dict[str, Any]:
    """Describe exact or balanced source multiplicity.

    The stress120 result deliberately retains its original one-field shape so
    rebuilding it produces the same workload and manifest hashes.  General
    counts additionally expose the bounds and partial-cycle size; for a count
    divisible by 60, ``instances_per_source`` remains an exact integer.
    """

    minimum, sources_with_extra = divmod(load_instance_count, UNIQUE_SOURCE_COUNT)
    maximum = minimum + int(sources_with_extra > 0)
    result: dict[str, Any] = {
        "instances_per_source": minimum if sources_with_extra == 0 else None,
    }
    if load_instance_count != LOAD_INSTANCE_COUNT:
        result.update(
            {
                "minimum_instances_per_source": minimum,
                "maximum_instances_per_source": maximum,
                "sources_with_one_extra_instance": sources_with_extra,
                "source_instances_are_balanced": True,
            }
        )
    return result


def _expected_source_sequence(
    source_ids: set[str], load_instance_count: int, seed: int
) -> list[str]:
    originals = sorted(source_ids)
    duplicate_cycle = list(originals)
    random.Random(seed).shuffle(duplicate_cycle)
    duplicates_needed = load_instance_count - len(originals)
    duplicate_sources = [
        duplicate_cycle[index % len(duplicate_cycle)]
        for index in range(duplicates_needed)
    ]
    return [*originals, *duplicate_sources]


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return payload


def _resolve(anchor: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"invalid {label} path")
    path = (anchor / raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _relative(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _source_sequence(workload: Mapping[str, Any]) -> list[str]:
    traces = workload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("prepared workload has no traces")
    sequence: list[str] = []
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("prepared workload trace is not an object")
        source = trace.get("source_trace")
        if not isinstance(source, str) or not source:
            raise ValueError("prepared workload trace has no source_trace")
        sequence.append(Path(source).name)
    return sequence


def _load_parent_inputs(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Path], str]:
    verified = load_fixed_manifest(manifest_path, "heldout")
    manifest = _load_object(manifest_path, "heldout60 manifest")
    anchor = manifest_path.parent
    workloads: dict[str, dict[str, Any]] = {}
    workload_paths: dict[str, Path] = {}
    for mode in ("none", "learned"):
        record = manifest["workloads"]["heldout"][mode]
        path = _resolve(anchor, record.get("prepared_workload"), f"heldout/{mode}")
        if file_sha256(path) != record.get("prepared_workload_sha256"):
            raise ValueError(f"heldout/{mode} workload checksum mismatch")
        workloads[mode] = _load_object(path, f"heldout/{mode} workload")
        workload_paths[mode] = path
    if _source_sequence(workloads["none"]) != _source_sequence(workloads["learned"]):
        raise ValueError("heldout none/learned source sequences differ")
    if len(set(_source_sequence(workloads["none"]))) != UNIQUE_SOURCE_COUNT:
        raise ValueError("heldout source registry must contain exactly 60 unique sessions")
    return manifest, workloads, workload_paths, str(verified["manifest_sha256"])


def _authoritative_sources(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    heldout_workload: Mapping[str, Any],
) -> dict[str, Path]:
    split_path = _resolve(
        manifest_path.parent,
        manifest.get("fixed_split_manifest"),
        "fixed split manifest",
    )
    split = _load_object(split_path, "fixed split manifest")
    checksums: dict[str, str] = {}
    for role in ("tuning", "final"):
        entries = split.get(f"{role}_sessions")
        if not isinstance(entries, list):
            raise ValueError(f"fixed split has no {role} registry")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"fixed split {role} entry is invalid")
            session_id = str(entry.get("session_id", ""))
            checksum = str(entry.get("sha256", ""))
            if not session_id or session_id in checksums:
                raise ValueError(f"invalid/duplicate evaluation source: {session_id}")
            checksums[session_id] = checksum
    calibration_entries = split.get("calibration_sessions")
    if not isinstance(calibration_entries, list):
        raise ValueError("fixed split has no calibration registry")
    calibration_ids = {
        str(entry.get("session_id"))
        for entry in calibration_entries
        if isinstance(entry, Mapping)
    }

    sources: dict[str, Path] = {}
    traces = heldout_workload.get("traces")
    if not isinstance(traces, list):
        raise ValueError("heldout workload traces are invalid")
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("heldout workload trace is invalid")
        raw_path = Path(str(trace.get("source_trace", ""))).resolve()
        session_id = raw_path.name
        if session_id in calibration_ids:
            raise ValueError(f"calibration source leaked into stress inputs: {session_id}")
        expected_checksum = checksums.get(session_id)
        if expected_checksum is None:
            raise ValueError(f"stress source is outside tuning/final: {session_id}")
        if not raw_path.is_file():
            raise FileNotFoundError(f"authoritative stress source is missing: {raw_path}")
        actual_checksum = file_sha256(raw_path)
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"source checksum mismatch for {session_id}: "
                f"{actual_checksum} != {expected_checksum}"
            )
        if session_id in sources and sources[session_id] != raw_path:
            raise ValueError(f"source session resolves to multiple paths: {session_id}")
        sources[session_id] = raw_path
    if len(sources) != UNIQUE_SOURCE_COUNT or set(sources) != set(checksums):
        raise ValueError("stress sources must be exactly the 60 tuning+final sessions")
    return sources


def _messages_start_with_marker(messages: Any, marker: str) -> bool:
    """Require a distinct leading system prefix to defeat prefix-cache sharing."""
    return bool(
        isinstance(messages, list)
        and messages
        and isinstance(messages[0], Mapping)
        and messages[0].get("role") == "system"
        and messages[0].get("content") == marker
    )


def _static_identity(workload: Mapping[str, Any]) -> list[dict[str, Any]]:
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


def _validate_mode_workload(
    workload: dict[str, Any],
    *,
    mode: str,
    tokenizer: Any,
    source_ids: set[str],
    mapper_checksum: str,
    top_k: int,
    load_instance_count: int,
    seed: int,
    validate_prompt_tokens: bool = True,
) -> dict[str, Any]:
    metadata = workload.get("meta")
    traces = workload.get("traces")
    if not isinstance(metadata, Mapping) or not isinstance(traces, list):
        raise ValueError(f"stress {mode} workload shape is invalid")
    if len(traces) != load_instance_count:
        raise ValueError(
            f"stress {mode} must contain exactly {load_instance_count} load instances"
        )
    if int(metadata.get("target_trace_count", -1)) != load_instance_count:
        raise ValueError(f"stress {mode} target trace count mismatch")
    if int(metadata.get("load_instance_count", -1)) != load_instance_count:
        raise ValueError(f"stress {mode} load instance count mismatch")
    if metadata.get("tool_overlap_mode") != mode:
        raise ValueError(f"stress {mode} overlap mode mismatch")
    if metadata.get("prefix_marker_mode") != "break_prefix":
        raise ValueError(f"stress {mode} must use break_prefix markers")
    max_model_len = int(metadata["max_model_len"])
    output_cap = int(metadata["max_output_tokens_cap"])
    output_floor = int(metadata["min_output_tokens_floor"])
    output_buffer = int(metadata["output_token_buffer"])
    source_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    trace_ids: set[str] = set()
    variants: set[int] = set()
    duplicate_markers: set[str] = set()
    request_count = 0
    truncated_count = 0
    load_rows: list[dict[str, Any]] = []
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError(f"stress {mode} trace is invalid")
        trace_id = str(trace.get("trace_id", ""))
        variant = int(trace.get("variant_index", -1))
        source = Path(str(trace.get("source_trace", ""))).name
        duplicated = trace.get("duplicated") is True
        marker = str(trace.get("prefix_char", ""))
        if trace_id != f"trace_{variant:03d}" or trace_id in trace_ids:
            raise ValueError(f"stress {mode} has invalid/duplicate trace identity")
        if variant < 0 or variant in variants:
            raise ValueError(f"stress {mode} has invalid/duplicate variant index")
        if source not in source_ids:
            raise ValueError(f"stress {mode} contains source outside heldout: {source}")
        trace_ids.add(trace_id)
        variants.add(variant)
        source_counts[source] += 1
        if duplicated:
            duplicate_counts[source] += 1
            if not marker or marker in duplicate_markers:
                raise ValueError(f"stress {mode} duplicate marker is empty/non-unique")
            duplicate_markers.add(marker)
        else:
            original_counts[source] += 1
            if marker:
                raise ValueError(f"stress {mode} original unexpectedly has a marker")
        requests = trace.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError(f"stress {mode} trace has no requests")
        trace_truncated = 0
        for request in requests:
            if not isinstance(request, Mapping):
                raise ValueError(f"stress {mode} request is invalid")
            request_count += 1
            prompt_tokens = int(request.get("prompt_tokens", -1))
            if validate_prompt_tokens:
                rebuilt_prompt_tokens = _build_chat_tokens(
                    tokenizer, request.get("messages", [])
                )
                if rebuilt_prompt_tokens != prompt_tokens:
                    raise ValueError(f"stress {mode} prompt token count is stale")
            if duplicated and not _messages_start_with_marker(
                request.get("messages"), marker
            ):
                raise ValueError(
                    f"stress {mode} duplicate marker is not the leading system message"
                )
            truncated = request.get("truncated") is True
            trace_truncated += int(truncated)
            truncated_count += int(truncated)
            target = max(1, int(request.get("target_output_tokens", 1)))
            expected_max_tokens = min(
                output_cap,
                max(1, max_model_len - prompt_tokens),
                max(output_floor, target + output_buffer),
            )
            if int(request.get("max_tokens", -1)) != expected_max_tokens:
                raise ValueError(f"stress {mode} max_tokens is inconsistent")
            if prompt_tokens + expected_max_tokens > max_model_len:
                raise ValueError(f"stress {mode} request exceeds model context")
            if mode == "learned":
                if request.get("tool_prediction_artifact_sha256") != mapper_checksum:
                    raise ValueError("stress learned request mapper checksum mismatch")
                if int(request.get("tool_prediction_top_k", -1)) != top_k:
                    raise ValueError("stress learned request top_k mismatch")
            elif any(str(key).startswith("tool_prediction_") for key in request):
                raise ValueError("stress none request unexpectedly has prediction fields")
        if int(trace.get("truncated_calls", -1)) != trace_truncated:
            raise ValueError(f"stress {mode} per-trace truncation count mismatch")
        load_rows.append(
            {
                "trace_id": trace_id,
                "source_session": source,
                "variant_index": variant,
                "duplicated": duplicated,
                "prefix_char": marker,
            }
        )
    expected_sequence = _expected_source_sequence(
        source_ids, load_instance_count, seed
    )
    expected_original_counts = Counter(expected_sequence[:UNIQUE_SOURCE_COUNT])
    expected_duplicate_counts = Counter(expected_sequence[UNIQUE_SOURCE_COUNT:])
    if variants != set(range(load_instance_count)):
        raise ValueError(f"stress {mode} variant indices are not contiguous")
    if _source_sequence(workload) != expected_sequence:
        raise ValueError(f"stress {mode} source order is not deterministic and balanced")
    if source_counts != Counter(expected_sequence):
        raise ValueError(f"stress {mode} source instances are not balanced")
    if original_counts != expected_original_counts:
        raise ValueError(f"stress {mode} must have one original per source")
    if duplicate_counts != expected_duplicate_counts:
        raise ValueError(f"stress {mode} duplicate source allocation is not balanced")
    duplicates_added = load_instance_count - UNIQUE_SOURCE_COUNT
    expected_markers = {
        duplicate_variant_marker(index) for index in range(duplicates_added)
    }
    if duplicate_markers != expected_markers:
        raise ValueError(f"stress {mode} duplicate markers are not globally unique")
    if int(metadata.get("duplicates_added", -1)) != duplicates_added:
        raise ValueError(f"stress {mode} metadata duplicate count mismatch")
    for field, expected in _replication_metadata(load_instance_count).items():
        if metadata.get(field) != expected:
            raise ValueError(f"stress {mode} metadata {field} mismatch")
    if int(metadata.get("total_truncated_calls", -1)) != truncated_count:
        raise ValueError(f"stress {mode} total truncation count mismatch")
    if mode == "learned":
        if metadata.get("tool_prediction_artifact_sha256") != mapper_checksum:
            raise ValueError("stress learned metadata mapper checksum mismatch")
        if int(metadata.get("tool_prediction_top_k", -1)) != top_k:
            raise ValueError("stress learned metadata top_k mismatch")
    return {
        "request_count": request_count,
        "load_identity_sha256": canonical_sha256(load_rows),
        "source_sequence": _source_sequence(workload),
    }


def _prepare_stress_mode(
    *,
    source_directory: Path,
    sources: Mapping[str, Path],
    tokenizer: Any,
    parameters: Mapping[str, Any],
    mode: str,
    mapper_path: Path,
    seed: int,
    load_instance_count: int,
    evidence_role: str,
) -> dict[str, Any]:
    workload = prepare_trace_workload(
        trace_dir=source_directory,
        tokenizer=tokenizer,
        target_trace_count=load_instance_count,
        max_model_len=int(parameters["max_model_len"]),
        max_output_tokens_cap=int(parameters["max_output_tokens_cap"]),
        min_output_tokens_floor=int(parameters["min_output_tokens_floor"]),
        output_token_buffer=int(parameters["output_token_buffer"]),
        duplicate_seed=seed,
        tool_overlap_mode=mode,
        tool_overlap_efficiency=1.0,
        prefix_marker_mode="break_prefix",
        tool_prediction_model=mapper_path if mode == "learned" else None,
        tool_prediction_top_k=int(parameters["tool_prediction_top_k"]),
    )
    for trace in workload["traces"]:
        session_id = Path(trace["source_trace"]).name
        trace["source_trace"] = str(sources[session_id])
    workload["meta"].update(
        {
            "source_trace_dir": None,
            "source_role": "heldout",
            "evidence_role": evidence_role,
            "unique_source_session_count": UNIQUE_SOURCE_COUNT,
            "load_instance_count": load_instance_count,
            **_replication_metadata(load_instance_count),
            "independent_sample_count": UNIQUE_SOURCE_COUNT,
            "duplicates_are_not_independent": (
                load_instance_count > UNIQUE_SOURCE_COUNT
            ),
            "stress_is_not_final": True,
        }
    )
    return workload


def build_stress_bundle(
    *,
    manifest_path: Path,
    output_root: Path,
    output_manifest: Path,
    tokenizer: Any,
    seed: int | None = None,
    load_instance_count: int = LOAD_INSTANCE_COUNT,
) -> dict[str, Any]:
    load_instance_count = _normalize_load_instance_count(load_instance_count)
    evidence_role = _evidence_role(load_instance_count)
    replication_metadata = _replication_metadata(load_instance_count)
    duplicates_are_not_independent = load_instance_count > UNIQUE_SOURCE_COUNT
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    output_manifest = output_manifest.resolve()
    parent, heldout_workloads, _, parent_sha256 = _load_parent_inputs(manifest_path)
    sources = _authoritative_sources(manifest_path, parent, heldout_workloads["none"])
    source_registry = [
        {"session_id": session_id, "sha256": file_sha256(source_path)}
        for session_id, source_path in sorted(sources.items())
    ]
    learned_paths = {
        Path(str(trace["source_trace"])).name: Path(str(trace["source_trace"])).resolve()
        for trace in heldout_workloads["learned"]["traces"]
    }
    if learned_paths != sources:
        raise ValueError("heldout none/learned authoritative source paths differ")
    parameters = parent.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("heldout manifest has no parameters")
    duplicate_seed = int(parameters.get("seed")) if seed is None else int(seed)
    mapper_path = _resolve(
        manifest_path.parent,
        parent.get("calibration_only_mapper"),
        "calibration-only mapper",
    )
    _, mapper_artifact = load_artifact(mapper_path)
    mapper_checksum = str(mapper_artifact["artifact_sha256"])
    if mapper_checksum != parent.get("calibration_only_mapper_sha256"):
        raise ValueError("heldout manifest mapper checksum mismatch")
    top_k = int(parameters["tool_prediction_top_k"])

    with tempfile.TemporaryDirectory(prefix="paste-stress-sources-") as temporary:
        source_directory = Path(temporary)
        for session_id, source_path in sorted(sources.items()):
            (source_directory / session_id).symlink_to(source_path)
        workloads = {
            mode: _prepare_stress_mode(
                source_directory=source_directory,
                sources=sources,
                tokenizer=tokenizer,
                parameters=parameters,
                mode=mode,
                mapper_path=mapper_path,
                seed=duplicate_seed,
                load_instance_count=load_instance_count,
                evidence_role=evidence_role,
            )
            for mode in ("none", "learned")
        }

    validation = {
        "none": _validate_mode_workload(
            workloads["none"],
            mode="none",
            tokenizer=tokenizer,
            source_ids=set(sources),
            mapper_checksum=mapper_checksum,
            top_k=top_k,
            load_instance_count=load_instance_count,
            seed=duplicate_seed,
        )
    }
    none_static_identity = _static_identity(workloads["none"])
    learned_static_identity = _static_identity(workloads["learned"])
    if none_static_identity != learned_static_identity:
        raise ValueError("stress none/learned static request identities differ")
    # Messages and every token-budget input are now proven identical to the
    # fully retokenized none workload.  Learned validation can safely reuse
    # that result instead of running the tokenizer over the same prompts again.
    validation["learned"] = _validate_mode_workload(
        workloads["learned"],
        mode="learned",
        tokenizer=tokenizer,
        source_ids=set(sources),
        mapper_checksum=mapper_checksum,
        top_k=top_k,
        load_instance_count=load_instance_count,
        seed=duplicate_seed,
        validate_prompt_tokens=False,
    )
    if validation["none"]["load_identity_sha256"] != validation["learned"][
        "load_identity_sha256"
    ]:
        raise ValueError("stress none/learned load identities differ")

    records: dict[str, dict[str, Any]] = {}
    for mode in ("none", "learned"):
        directory = output_root / mode
        workload_path = directory / "prepared_workload.json"
        summary_path = directory / "workload_summary.json"
        write_json_atomic(workload_path, workloads[mode])
        write_json_atomic(summary_path, summarize_workload(workloads[mode]))
        source_sequence = validation[mode]["source_sequence"]
        records[mode] = {
            "prepared_workload": _relative(workload_path, output_manifest.parent),
            "prepared_workload_sha256": file_sha256(workload_path),
            "workload_summary": _relative(summary_path, output_manifest.parent),
            "workload_summary_sha256": file_sha256(summary_path),
            "trace_count": load_instance_count,
            "load_instance_count": load_instance_count,
            "unique_source_session_count": UNIQUE_SOURCE_COUNT,
            "independent_sample_count": UNIQUE_SOURCE_COUNT,
            **replication_metadata,
            "source_sequence_sha256": canonical_sha256(source_sequence),
            "source_set_sha256": canonical_sha256(sorted(set(source_sequence))),
            "load_identity_sha256": validation[mode]["load_identity_sha256"],
            "tool_overlap_mode": mode,
            **(
                {
                    "mapper_artifact_sha256": mapper_checksum,
                    "tool_prediction_top_k": top_k,
                }
                if mode == "learned"
                else {}
            ),
        }

    combined = json.loads(json.dumps(parent, ensure_ascii=False))
    combined["stress_derived_from_manifest"] = _relative(
        manifest_path, output_manifest.parent
    )
    combined["stress_derived_from_manifest_sha256"] = parent_sha256
    combined["workloads"][STRESS_ROLE] = records
    combined["parameters"]["target_trace_counts"][STRESS_ROLE] = load_instance_count
    combined["parameters"]["stress_duplicate_seed"] = duplicate_seed
    combined["stress_definition"] = {
        "schema": "paste_repro.heldout_duplicate_stress",
        "version": 1,
        "evidence_role": evidence_role,
        "source_role": "heldout",
        "source_manifest_sha256": parent_sha256,
        "unique_source_session_count": UNIQUE_SOURCE_COUNT,
        "load_instance_count": load_instance_count,
        **replication_metadata,
        "independent_sample_count": UNIQUE_SOURCE_COUNT,
        "duplicates_are_not_independent": duplicates_are_not_independent,
        "is_final_evaluation": False,
        "calibration_excluded": True,
        "mapper_retrained": False,
        "prefix_marker_mode": "break_prefix",
        "duplicate_seed": duplicate_seed,
        "source_sessions": source_registry,
        "source_sessions_sha256": canonical_sha256(source_registry),
        "load_identity_sha256": validation["none"]["load_identity_sha256"],
    }

    def _cell_inputs() -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for policy_name, policy in (
            ("fcfs", "fcfs"),
            ("joint", "online_joint_pacer_v2"),
        ):
            for mode in ("none", "learned"):
                result[f"{policy_name}_{mode}"] = {
                    "policy": policy,
                    "tool_overlap_mode": mode,
                    "evaluation_workload": records[mode]["prepared_workload"],
                    "online_calibration_workload": parent["workloads"]["calibration"][
                        mode
                    ]["prepared_workload"],
                }
        return result

    combined["four_cell_inputs"][STRESS_ROLE] = _cell_inputs()
    combined["contamination_guards"].update(
        {
            "stress_source_role": "heldout tuning+final union only",
            "stress_calibration_excluded": True,
            "stress_mapper_retrained": False,
            "stress_unique_source_sessions": UNIQUE_SOURCE_COUNT,
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
    )
    if load_instance_count != LOAD_INSTANCE_COUNT:
        combined["contamination_guards"].update(
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
    combined.pop("manifest_sha256", None)
    combined["manifest_sha256"] = canonical_sha256(combined)
    write_json_atomic(output_manifest, combined)
    return combined


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic balanced heldout60 stress workloads without "
            "inference."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--tokenizer", default=_default_tokenizer())
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--load-instance-count",
        type=int,
        default=LOAD_INSTANCE_COUNT,
        help=(
            "total concurrent load instances; must be at least 60 and defaults "
            "to the manifest-compatible stress120 workload"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_instance_count = _normalize_load_instance_count(args.load_instance_count)
    evidence_role = _evidence_role(load_instance_count)
    replication_metadata = _replication_metadata(load_instance_count)
    manifest_path = args.manifest.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else manifest_path.parent / f"stress{load_instance_count}"
    )
    output_manifest = (
        args.manifest_out.resolve()
        if args.manifest_out is not None
        else manifest_path.parent / f"manifest_stress{load_instance_count}.json"
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    result = build_stress_bundle(
        manifest_path=manifest_path,
        output_root=output_root,
        output_manifest=output_manifest,
        tokenizer=tokenizer,
        seed=args.seed,
        load_instance_count=load_instance_count,
    )
    print(
        json.dumps(
            {
                "manifest": str(output_manifest),
                "manifest_sha256": result["manifest_sha256"],
                "unique_source_sessions": UNIQUE_SOURCE_COUNT,
                "load_instances": load_instance_count,
                **replication_metadata,
                "independent_sample_count": UNIQUE_SOURCE_COUNT,
                "evidence_role": evidence_role,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
