#!/usr/bin/env python3
"""Prepare matched none/learned workloads for a fixed three-way split.

This command performs no live inference.  It first builds the contamination-
aware split and calibration-only mapper, then invokes the existing trace runner
with ``--prepare-only`` for every role/overlap pair.  The resulting manifest
binds every workload to its source-session set and records the calibration
workload that an online predictor must use for each four-cell comparison.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from build_fixed_three_way_split import (  # noqa: E402
    DEFAULT_SALT,
    build_fixed_bundle,
    canonical_sha256,
    file_sha256,
)
from paste_repro.mapper import load_artifact, write_json_atomic  # noqa: E402


WORKLOAD_MANIFEST_SCHEMA = "paste_repro.fixed_workload_bundle"
WORKLOAD_MANIFEST_VERSION = 1
DEFAULT_MODEL_REVISION = "4b0ac5767427a55d08a254f0367e2934976598e0"


def _default_tokenizer() -> str:
    configured = os.getenv("MODEL_SNAPSHOT")
    if configured:
        return configured
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / "hf_cache")))
    revision = os.getenv("MODEL_REVISION", DEFAULT_MODEL_REVISION)
    return str(
        hf_home
        / "models--Alibaba-NLP--Tongyi-DeepResearch-30B-A3B"
        / "snapshots"
        / revision
    )


def _positive_int(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def build_prepare_command(
    *,
    python_executable: str,
    runner: Path,
    trace_directory: Path,
    trace_count: int,
    output_directory: Path,
    tokenizer: str,
    speedup: float,
    max_model_len: int,
    output_cap: int,
    output_buffer: int,
    min_output_floor: int,
    overlap_mode: str,
    mapper_artifact: Path,
    top_k: int,
    seed: int,
) -> list[str]:
    if overlap_mode not in {"none", "learned"}:
        raise ValueError(f"unsupported fixed workload overlap mode: {overlap_mode}")
    command = [
        python_executable,
        str(runner),
        "--trace-dir",
        str(trace_directory),
        "--trace-count",
        str(trace_count),
        "--output-dir",
        str(output_directory),
        "--tokenizer",
        tokenizer,
        "--speedup",
        str(speedup),
        "--prepare-only",
        "--max-model-len",
        str(max_model_len),
        "--max-output-tokens-cap",
        str(output_cap),
        "--output-token-buffer",
        str(output_buffer),
        "--min-output-tokens-floor",
        str(min_output_floor),
        "--tool-overlap-mode",
        overlap_mode,
        "--tool-overlap-efficiency",
        "1.0",
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--presence-penalty",
        "0",
        "--seed",
        str(seed),
    ]
    if overlap_mode == "learned":
        command.extend(
            [
                "--tool-prediction-model",
                str(mapper_artifact),
                "--tool-prediction-top-k",
                str(top_k),
            ]
        )
    return command


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return payload


def _source_session_sequence(workload: Mapping[str, Any]) -> list[str]:
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


def _relative_path(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def build_workload_manifest(
    *,
    fixed_bundle: Mapping[str, Any],
    workload_root: Path,
    output_manifest: Path,
    output_cap: int,
    speedup: float,
    max_model_len: int,
    output_buffer: int,
    min_output_floor: int,
    top_k: int,
    target_counts: Mapping[str, int],
    seed: int,
) -> dict[str, Any]:
    split_manifest_path = Path(str(fixed_bundle["split_manifest_path"]))
    mapper_artifact_path = Path(str(fixed_bundle["mapper_artifact_path"]))
    split_manifest = _load_json_object(split_manifest_path, "fixed split manifest")
    _, mapper_artifact = load_artifact(mapper_artifact_path)
    if split_manifest.get("manifest_sha256") != fixed_bundle.get(
        "split_manifest_sha256"
    ):
        raise ValueError("fixed split manifest checksum does not match build result")
    split_without_checksum = dict(split_manifest)
    supplied_split_checksum = split_without_checksum.pop("manifest_sha256", None)
    if supplied_split_checksum != canonical_sha256(split_without_checksum):
        raise ValueError("fixed split manifest checksum mismatch")
    if mapper_artifact.get("artifact_sha256") != fixed_bundle.get(
        "mapper_artifact_sha256"
    ):
        raise ValueError("calibration-only mapper checksum does not match build result")

    expected_role_sessions = {
        role: {
            str(entry["session_id"])
            for entry in split_manifest[f"{role}_sessions"]
        }
        for role in ("calibration", "tuning", "final")
    }
    if any(
        expected_role_sessions[left] & expected_role_sessions[right]
        for left, right in (
            ("calibration", "tuning"),
            ("calibration", "final"),
            ("tuning", "final"),
        )
    ):
        raise ValueError("fixed split roles overlap")

    workload_records: dict[str, dict[str, Any]] = {}
    role_sequences: dict[str, dict[str, list[str]]] = {}
    manifest_anchor = output_manifest.parent
    for role in ("calibration", "tuning", "final"):
        workload_records[role] = {}
        role_sequences[role] = {}
        for mode in ("none", "learned"):
            output_directory = workload_root / role / mode
            workload_path = output_directory / "prepared_workload.json"
            summary_path = output_directory / "workload_summary.json"
            workload = _load_json_object(workload_path, f"{role}/{mode} workload")
            summary = _load_json_object(summary_path, f"{role}/{mode} workload summary")
            metadata = workload.get("meta")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{role}/{mode} workload has no metadata")
            if metadata.get("tool_overlap_mode") != mode:
                raise ValueError(f"{role}/{mode} workload overlap mode mismatch")
            if int(metadata.get("max_output_tokens_cap", -1)) != output_cap:
                raise ValueError(f"{role}/{mode} workload output cap mismatch")
            if int(metadata.get("max_model_len", -1)) != max_model_len:
                raise ValueError(f"{role}/{mode} workload model length mismatch")
            if int(metadata.get("target_trace_count", -1)) != target_counts[role]:
                raise ValueError(f"{role}/{mode} workload target count mismatch")
            if summary.get("tool_overlap_mode") != mode:
                raise ValueError(f"{role}/{mode} workload summary mode mismatch")
            if int(summary.get("trace_count", -1)) != target_counts[role]:
                raise ValueError(f"{role}/{mode} workload trace count mismatch")
            if mode == "learned":
                if metadata.get("tool_prediction_artifact_sha256") != mapper_artifact[
                    "artifact_sha256"
                ]:
                    raise ValueError(f"{role}/learned mapper checksum mismatch")
                if int(metadata.get("tool_prediction_top_k", -1)) != top_k:
                    raise ValueError(f"{role}/learned top_k mismatch")
            elif any(str(key).startswith("tool_prediction_") for key in metadata):
                raise ValueError(f"{role}/none workload unexpectedly binds a mapper")

            source_sequence = _source_session_sequence(workload)
            if len(source_sequence) != target_counts[role]:
                raise ValueError(f"{role}/{mode} source sequence count mismatch")
            unexpected = sorted(set(source_sequence) - expected_role_sessions[role])
            if unexpected:
                raise ValueError(
                    f"{role}/{mode} workload contains sessions outside its role: {unexpected}"
                )
            role_sequences[role][mode] = source_sequence
            workload_records[role][mode] = {
                "prepared_workload": _relative_path(workload_path, manifest_anchor),
                "prepared_workload_sha256": file_sha256(workload_path),
                "workload_summary": _relative_path(summary_path, manifest_anchor),
                "workload_summary_sha256": file_sha256(summary_path),
                "trace_count": len(source_sequence),
                "unique_source_session_count": len(set(source_sequence)),
                "source_sequence_sha256": canonical_sha256(source_sequence),
                "tool_overlap_mode": mode,
                **(
                    {
                        "mapper_artifact_sha256": mapper_artifact["artifact_sha256"],
                        "tool_prediction_top_k": top_k,
                    }
                    if mode == "learned"
                    else {}
                ),
            }
        if role_sequences[role]["none"] != role_sequences[role]["learned"]:
            raise ValueError(f"{role} none/learned workloads do not replay the same sessions")

    def cell_inputs(eval_role: str) -> dict[str, Any]:
        return {
            "fcfs_none": {
                "policy": "fcfs",
                "tool_overlap_mode": "none",
                "evaluation_workload": workload_records[eval_role]["none"][
                    "prepared_workload"
                ],
                "online_calibration_workload": workload_records["calibration"]["none"][
                    "prepared_workload"
                ],
            },
            "fcfs_learned": {
                "policy": "fcfs",
                "tool_overlap_mode": "learned",
                "evaluation_workload": workload_records[eval_role]["learned"][
                    "prepared_workload"
                ],
                "online_calibration_workload": workload_records["calibration"]["learned"][
                    "prepared_workload"
                ],
            },
            "joint_none": {
                "policy": "online_joint_pacer_v2",
                "tool_overlap_mode": "none",
                "evaluation_workload": workload_records[eval_role]["none"][
                    "prepared_workload"
                ],
                "online_calibration_workload": workload_records["calibration"]["none"][
                    "prepared_workload"
                ],
            },
            "joint_learned": {
                "policy": "online_joint_pacer_v2",
                "tool_overlap_mode": "learned",
                "evaluation_workload": workload_records[eval_role]["learned"][
                    "prepared_workload"
                ],
                "online_calibration_workload": workload_records["calibration"]["learned"][
                    "prepared_workload"
                ],
            },
        }

    manifest: dict[str, Any] = {
        "schema": WORKLOAD_MANIFEST_SCHEMA,
        "version": WORKLOAD_MANIFEST_VERSION,
        "fixed_split_manifest": _relative_path(split_manifest_path, manifest_anchor),
        "fixed_split_manifest_sha256": split_manifest["manifest_sha256"],
        "source_mapper_artifact_sha256": fixed_bundle[
            "source_mapper_artifact_sha256"
        ],
        "calibration_only_mapper": _relative_path(mapper_artifact_path, manifest_anchor),
        "calibration_only_mapper_sha256": mapper_artifact["artifact_sha256"],
        "parameters": {
            "max_output_tokens_cap": output_cap,
            "speedup": speedup,
            "max_model_len": max_model_len,
            "output_token_buffer": output_buffer,
            "min_output_tokens_floor": min_output_floor,
            "tool_prediction_top_k": top_k,
            "target_trace_counts": dict(target_counts),
            "seed": seed,
        },
        "workloads": workload_records,
        "four_cell_inputs": {
            "tuning": cell_inputs("tuning"),
            "final": cell_inputs("final"),
        },
        "contamination_guards": {
            "mapper_fit_sessions": "calibration only",
            "online_predictor_calibration": "matching calibration workload only",
            "tuning_and_final_not_used_for_online_calibration": True,
            "final_workloads_prepared_without_live_inference": True,
            "configuration_must_freeze_before_any_final_cell": True,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def prepare_fixed_workloads(args: argparse.Namespace) -> dict[str, Any]:
    if not math.isfinite(args.speedup) or args.speedup <= 0:
        raise ValueError("speedup must be positive")
    output_cap = _positive_int(args.output_cap, "output_cap")
    top_k = _positive_int(args.top_k, "top_k")
    max_model_len = _positive_int(args.max_model_len, "max_model_len")
    output_buffer = _positive_int(args.output_buffer, "output_buffer")
    min_output_floor = (
        output_cap
        if args.min_output_floor is None
        else _positive_int(args.min_output_floor, "min_output_floor")
    )
    if min_output_floor >= max_model_len:
        raise ValueError("min_output_floor must be smaller than max_model_len")
    target_counts = {
        "calibration": _positive_int(args.calibration_count, "calibration_count"),
        "tuning": _positive_int(args.target_tuning_count, "target_tuning_count"),
        "final": _positive_int(args.target_final_count, "target_final_count"),
    }
    if not args.runner.is_file():
        raise FileNotFoundError(f"trace workload runner is missing: {args.runner}")

    fixed_result_path = args.workload_root.resolve() / "fixed_split_result.json"
    fixed_bundle = build_fixed_bundle(
        legacy_artifact_path=args.legacy_artifact,
        trace_directory=args.trace_dir,
        output_root=args.split_output_root,
        salt=args.salt,
        calibration_count=args.calibration_count,
        tuning_count=args.tuning_count,
        final_count=args.final_count,
        result_out=fixed_result_path,
    )

    mapper_path = Path(str(fixed_bundle["mapper_artifact_path"]))
    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    jobs = _positive_int(int(getattr(args, "jobs", 1)), "jobs")
    commands: list[list[str]] = []
    if not bool(getattr(args, "manifest_only", False)):
        for role in ("calibration", "tuning", "final"):
            trace_directory = Path(
                str(fixed_bundle["roles"][role]["absolute_directory"])
            )
            for mode in ("none", "learned"):
                output_directory = args.workload_root.resolve() / role / mode
                command = build_prepare_command(
                    python_executable=sys.executable,
                    runner=args.runner,
                    trace_directory=trace_directory,
                    trace_count=target_counts[role],
                    output_directory=output_directory,
                    tokenizer=args.tokenizer,
                    speedup=args.speedup,
                    max_model_len=max_model_len,
                    output_cap=output_cap,
                    output_buffer=output_buffer,
                    min_output_floor=min_output_floor,
                    overlap_mode=mode,
                    mapper_artifact=mapper_path,
                    top_k=top_k,
                    seed=args.seed,
                )
                commands.append(command)

    def run_prepare(command: list[str]) -> None:
        subprocess.run(command, check=True, env=environment)

    if commands:
        with ThreadPoolExecutor(max_workers=min(jobs, len(commands))) as executor:
            list(executor.map(run_prepare, commands))

    output_manifest = args.manifest_out or (args.workload_root.resolve() / "manifest.json")
    manifest = build_workload_manifest(
        fixed_bundle=fixed_bundle,
        workload_root=args.workload_root.resolve(),
        output_manifest=output_manifest,
        output_cap=output_cap,
        speedup=args.speedup,
        max_model_len=max_model_len,
        output_buffer=output_buffer,
        min_output_floor=min_output_floor,
        top_k=top_k,
        target_counts=target_counts,
        seed=args.seed,
    )
    write_json_atomic(output_manifest, manifest)
    return {**manifest, "manifest_path": str(output_manifest.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare none/learned calibration, tuning, and final workloads from "
            "a fixed contamination-aware session split. No live inference is run."
        )
    )
    parser.add_argument(
        "--legacy-artifact",
        type=Path,
        default=REPRODUCTION_ROOT / "results" / "tool_only" / "url_rank_mapper.json",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=REPOSITORY_ROOT / "traces" / "my_traces",
    )
    parser.add_argument(
        "--split-output-root",
        type=Path,
        default=REPRODUCTION_ROOT / "artifacts" / "fixed_trace_splits",
    )
    parser.add_argument(
        "--workload-root",
        type=Path,
        default=REPRODUCTION_ROOT / "artifacts" / "workloads" / "fixed_three_way",
    )
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument(
        "--runner",
        type=Path,
        default=REPOSITORY_ROOT / "scripts" / "run_vllm_trace_experiment.py",
    )
    parser.add_argument("--tokenizer", default=_default_tokenizer())
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--calibration-count", type=int, default=40)
    parser.add_argument("--tuning-count", type=int, default=30)
    parser.add_argument("--final-count", type=int, default=30)
    parser.add_argument("--target-tuning-count", type=int, default=30)
    parser.add_argument("--target-final-count", type=int, default=30)
    parser.add_argument("--output-cap", type=int, default=128)
    parser.add_argument("--speedup", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--output-buffer", type=int, default=8)
    parser.add_argument("--min-output-floor", type=int)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of independent prepare-only workloads to tokenize in parallel",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="validate existing prepared workloads and rebuild only their manifest",
    )
    return parser.parse_args()


def main() -> int:
    result = prepare_fixed_workloads(parse_args())
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
