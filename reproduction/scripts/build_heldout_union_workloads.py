#!/usr/bin/env python3
"""Build a 60-session sensitivity workload from tuning + final sessions.

The fixed 40/30/30 protocol keeps calibration isolated from both evaluation
roles.  This helper combines the two disjoint 30-session roles without fitting
or updating either predictor.  It is intentionally labelled ``heldout`` (not
``final``): once final has been inspected, the union is only a load-sensitivity
check, not a new untouched test set.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for import_root in (REPRODUCTION_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from trace_experiment_lib import summarize_workload  # noqa: E402

from build_fixed_three_way_split import canonical_sha256, file_sha256  # noqa: E402
from prepare_fixed_workloads import WORKLOAD_MANIFEST_SCHEMA  # noqa: E402


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return value


def _verified_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_object(path, "fixed workload manifest")
    if manifest.get("schema") != WORKLOAD_MANIFEST_SCHEMA:
        raise ValueError("unsupported fixed workload manifest schema")
    supplied = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if supplied != canonical_sha256(payload):
        raise ValueError("fixed workload manifest checksum mismatch")
    for role in ("calibration", "tuning", "final"):
        if role not in manifest.get("workloads", {}):
            raise ValueError(f"fixed workload manifest is missing role: {role}")
    return manifest


def _resolve(anchor: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid {label} path in manifest")
    path = (anchor / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _source_sequence(workload: Mapping[str, Any]) -> list[str]:
    traces = workload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("prepared workload has no traces")
    result: list[str] = []
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("prepared workload trace must be an object")
        source = trace.get("source_trace")
        if not isinstance(source, str) or not source:
            raise ValueError("prepared workload trace is missing source_trace")
        result.append(Path(source).name)
    return result


def _merge_mode(
    *,
    tuning: Mapping[str, Any],
    final: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    tuning_meta = tuning.get("meta")
    final_meta = final.get("meta")
    if not isinstance(tuning_meta, Mapping) or not isinstance(final_meta, Mapping):
        raise ValueError(f"{mode} source workload metadata is missing")
    stable_meta_fields = (
        "max_model_len",
        "max_output_tokens_cap",
        "min_output_tokens_floor",
        "output_token_buffer",
        "tool_overlap_mode",
        "tool_overlap_efficiency",
        "prefix_marker_mode",
        "tool_prediction_artifact_sha256",
        "tool_prediction_top_k",
    )
    for field in stable_meta_fields:
        if tuning_meta.get(field) != final_meta.get(field):
            raise ValueError(f"{mode} tuning/final metadata mismatch: {field}")
    if tuning_meta.get("tool_overlap_mode") != mode:
        raise ValueError(f"source workload mode mismatch: expected {mode}")

    merged_traces: list[dict[str, Any]] = []
    source_sessions: set[str] = set()
    for role, workload in (("tuning", tuning), ("final", final)):
        traces = workload.get("traces")
        if not isinstance(traces, list):
            raise ValueError(f"{mode}/{role} workload traces are invalid")
        for trace in traces:
            copied = copy.deepcopy(trace)
            source = Path(str(copied.get("source_trace", ""))).name
            if not source or source in source_sessions:
                raise ValueError(f"heldout source session is missing or duplicated: {source}")
            source_sessions.add(source)
            copied["trace_id"] = f"heldout_{len(merged_traces):03d}"
            copied["variant_index"] = len(merged_traces)
            copied["duplicated"] = False
            copied["prefix_char"] = ""
            merged_traces.append(copied)

    if len(merged_traces) != 60 or len(source_sessions) != 60:
        raise ValueError("heldout union must contain exactly 60 unique source sessions")
    meta = copy.deepcopy(dict(tuning_meta))
    meta.update(
        {
            "source_trace_dir": None,
            "source_roles": ["tuning", "final"],
            "evidence_role": "heldout_load_sensitivity_not_untouched_final",
            "target_trace_count": 60,
            "duplicates_added": 0,
            "total_truncated_calls": sum(
                int(trace.get("truncated_calls", 0)) for trace in merged_traces
            ),
        }
    )
    return {"meta": meta, "traces": merged_traces}


def build_heldout_union(
    *,
    manifest_path: Path,
    output_root: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _verified_manifest(manifest_path)
    anchor = manifest_path.parent
    output_root = output_root.resolve()
    output_manifest = output_manifest.resolve()

    records: dict[str, dict[str, Any]] = {}
    source_sequences: dict[str, list[str]] = {}
    for mode in ("none", "learned"):
        inputs: dict[str, dict[str, Any]] = {}
        for role in ("tuning", "final"):
            record = manifest["workloads"][role][mode]
            path = _resolve(anchor, record["prepared_workload"], f"{role}/{mode} workload")
            if file_sha256(path) != record.get("prepared_workload_sha256"):
                raise ValueError(f"{role}/{mode} workload checksum mismatch")
            inputs[role] = _load_object(path, f"{role}/{mode} workload")
        merged = _merge_mode(tuning=inputs["tuning"], final=inputs["final"], mode=mode)
        mode_root = output_root / mode
        workload_path = mode_root / "prepared_workload.json"
        summary_path = mode_root / "workload_summary.json"
        write_json_atomic(workload_path, merged)
        write_json_atomic(summary_path, summarize_workload(merged))
        sequence = _source_sequence(merged)
        source_sequences[mode] = sequence
        record = {
            "prepared_workload": Path(
                os.path.relpath(workload_path, output_manifest.parent)
            ).as_posix(),
            "prepared_workload_sha256": file_sha256(workload_path),
            "workload_summary": Path(
                os.path.relpath(summary_path, output_manifest.parent)
            ).as_posix(),
            "workload_summary_sha256": file_sha256(summary_path),
            "source_sequence_sha256": canonical_sha256(sequence),
            "tool_overlap_mode": mode,
            "trace_count": 60,
            "unique_source_session_count": 60,
        }
        if mode == "learned":
            record["mapper_artifact_sha256"] = manifest[
                "calibration_only_mapper_sha256"
            ]
            record["tool_prediction_top_k"] = manifest["parameters"][
                "tool_prediction_top_k"
            ]
        records[mode] = record

    if source_sequences["none"] != source_sequences["learned"]:
        raise ValueError("heldout none/learned source sequences differ")

    combined = copy.deepcopy(manifest)
    combined["derived_from_manifest"] = Path(
        os.path.relpath(manifest_path, output_manifest.parent)
    ).as_posix()
    combined["derived_from_manifest_sha256"] = manifest["manifest_sha256"]
    combined["workloads"]["heldout"] = records
    combined["parameters"]["target_trace_counts"]["heldout"] = 60

    def cells() -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for policy_name, policy in (
            ("fcfs", "fcfs"),
            ("joint", "online_joint_pacer_v2"),
        ):
            for mode in ("none", "learned"):
                result[f"{policy_name}_{mode}"] = {
                    "evaluation_workload": records[mode]["prepared_workload"],
                    "online_calibration_workload": manifest["workloads"][
                        "calibration"
                    ][mode]["prepared_workload"],
                    "policy": policy,
                    "tool_overlap_mode": mode,
                }
        return result

    combined["four_cell_inputs"]["heldout"] = cells()
    combined["contamination_guards"]["heldout_union_sessions"] = (
        "tuning plus previously inspected final; calibration excluded"
    )
    combined["contamination_guards"]["heldout_is_not_new_final"] = True
    combined.pop("manifest_sha256", None)
    combined["manifest_sha256"] = canonical_sha256(combined)
    write_json_atomic(output_manifest, combined)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a checksummed 60-session tuning+final load sensitivity bundle."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_heldout_union(
        manifest_path=args.manifest,
        output_root=args.output_root,
        output_manifest=args.manifest_out,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_out.resolve()),
                "manifest_sha256": result["manifest_sha256"],
                "trace_count": 60,
                "evidence_role": "heldout_load_sensitivity_not_untouched_final",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
