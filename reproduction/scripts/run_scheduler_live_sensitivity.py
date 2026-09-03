#!/usr/bin/env python3
"""Run a post-hoc, frozen-workload A/E scheduler sensitivity matrix.

This reviewer-requested experiment is deliberately development-only.  Every
cell uses the byte-identical frozen formal-v8 source workload and a fresh vLLM
server.  Within an A/E pair only the scheduler treatment changes; within the
E utilization sweep only the active physical-KV utilization target changes.

Use ``--check-only`` first.  It validates the pinned environment, model,
workload, grammar, exact cell matrix, and one-variable invariants without
creating an output directory, starting a server, touching a GPU, or using the
network.

Only the bounded ``target`` and ``high`` suites are executable.  The failed
six-cell ``comment3-shape-r1`` artifact remains immutable evidence under its
original runner SHA; this runner cannot resume it or reuse its observed cells.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
import uuid
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproduction/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_live_joint_formal_matrix as formal  # type: ignore  # noqa: E402


BASE_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_formal_v8_matrix.env.example"
)
WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v8.json"
)
UNDERLYING_EXPERIMENT_RUNNER = (
    REPOSITORY_ROOT / "scripts/run_live_tool_llm_experiment.py"
)
RUN_BASE = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/comment3_scheduler"
)
LOW_CONTEXT_TOKENS = 5_000
REFERENCE_CONTEXT_TOKENS = 10_000
HIGH_CONTEXT_TOKENS = 12_000
LOW_LOAD = 40
REFERENCE_LOAD = 80
REFERENCE_TARGET = 0.93
TARGET_VALUES = (0.85, 0.93, 0.97)
EXPECTED_SOURCE_COUNT = 80
EXPECTED_LLM_REQUESTS = 240
EXPECTED_TOOL_COMMITS = 160
DEFAULT_GPUS = "4,5,6,7"
DEFAULT_PORT = 8100
ROBUST_VISIT_MIN_START_INTERVAL_S = 3.0
TRANSPORT_REMEDIATION_VERSION = "post-r2-jina-429-remediation-v1"
SHAPE_HARNESS_REPAIR_VERSION = "post-shape-r1-formal-order-range-repair-v1"
FAILED_SHAPE_RUNNER_SHA256 = (
    "abcb8c67d2bb72a640663951dcc67e69d53269d1bb284f6579bcd0530299772c"
)
FORMAL_ORDER_INDEX_MIN = 0
FORMAL_ORDER_INDEX_MAX = 3
MAX_FORMAL_CELLS = FORMAL_ORDER_INDEX_MAX - FORMAL_ORDER_INDEX_MIN + 1
REGISTERED_SUITES = ("target", "high")
FAILED_SHAPE_RUN_ROOT = RUN_BASE / "comment3-shape-r1"
FAILED_SHAPE_BOUND_PATHS = (
    FAILED_SHAPE_RUN_ROOT / "run_plan.json",
    FAILED_SHAPE_RUN_ROOT / "failure.json",
    FAILED_SHAPE_RUN_ROOT / "cells/05-a-c12k-l80/cell_contract.json",
    FAILED_SHAPE_RUN_ROOT / "cells/05-a-c12k-l80/runner.stderr.log",
    FAILED_SHAPE_RUN_ROOT / "cells/05-a-c12k-l80/server/vllm_8000.log",
    FAILED_SHAPE_RUN_ROOT / "cells/05-a-c12k-l80/server_lifecycle.stdout.log",
    FAILED_SHAPE_RUN_ROOT / "cells/05-a-c12k-l80/server_lifecycle.stderr.log",
)


class LiveSensitivityError(RuntimeError):
    """Fail-closed error for the post-hoc live scheduler experiment."""


@dataclass(frozen=True)
class CellSpec:
    label: str
    cell: str
    context_padding_tokens: int
    max_active_tasks: int
    physical_kv_target: float
    pair_group: str
    role: str

    @property
    def is_reference_shape(self) -> bool:
        return (
            self.context_padding_tokens == REFERENCE_CONTEXT_TOKENS
            and self.max_active_tasks == REFERENCE_LOAD
        )


def _a(
    label: str,
    context: int,
    load: int,
    pair_group: str,
) -> CellSpec:
    return CellSpec(
        label,
        "A",
        context,
        load,
        REFERENCE_TARGET,
        pair_group,
        "fcfs_reference",
    )


def _e(
    label: str,
    context: int,
    load: int,
    target: float,
    pair_group: str,
    role: str = "joint_candidate",
) -> CellSpec:
    return CellSpec(label, "E", context, load, target, pair_group, role)


def cells_for_suite(suite: str) -> tuple[CellSpec, ...]:
    center_a = _a("a-c10k-l80", 10_000, 80, "c10k-l80")
    center_e = _e(
        "e-c10k-l80-u093", 10_000, 80, 0.93, "c10k-l80"
    )
    high_pair = (
        _a("a-c12k-l80", 12_000, 80, "c12k-l80"),
        _e("e-c12k-l80-u093", 12_000, 80, 0.93, "c12k-l80"),
    )
    target_extremes = (
        _e(
            "e-c10k-l80-u085",
            10_000,
            80,
            0.85,
            "c10k-l80",
            "active_physical_kv_target_sensitivity",
        ),
        _e(
            "e-c10k-l80-u097",
            10_000,
            80,
            0.97,
            "c10k-l80",
            "active_physical_kv_target_sensitivity",
        ),
    )
    if suite == "target":
        return (center_a, target_extremes[0], center_e, target_extremes[1])
    if suite == "high":
        return high_pair
    raise LiveSensitivityError(f"unknown suite: {suite}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_gpus(raw: str) -> tuple[int, int, int, int]:
    parts = raw.split(",")
    if len(parts) != 4 or any(re.fullmatch(r"0|[1-9][0-9]*", p) is None for p in parts):
        raise LiveSensitivityError("--gpus must contain four comma-separated GPU IDs")
    values = tuple(int(part) for part in parts)
    if len(set(values)) != 4:
        raise LiveSensitivityError("--gpus must contain four distinct GPU IDs")
    return values  # type: ignore[return-value]


def _derived_config(
    base: Mapping[str, str],
    spec: CellSpec,
    *,
    gpus: str,
    port: int,
) -> dict[str, str]:
    values = dict(base)
    values.update(
        {
            "CUDA_VISIBLE_DEVICES": gpus,
            "VLLM_PORT": str(port),
            # The original 2.1 s formal-v8 transport gate produced four
            # recovered 429s in the FCFS cell and one unrecovered 429 in the
            # first Joint cell of comment3-target-r2.  That run is retained as
            # failed evidence.  Every replacement A/E cell is rebaselined with
            # the same conservative gate; scheduler treatments are unchanged.
            "PASTE_LIVE_VISIT_MIN_START_INTERVAL_S": format(
                ROBUST_VISIT_MIN_START_INTERVAL_S, ".1f"
            ),
            "PASTE_LIVE_MAX_ACTIVE_TASKS": str(spec.max_active_tasks),
            "PASTE_LIVE_CONTEXT_PADDING_TOKENS": str(
                spec.context_padding_tokens
            ),
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": format(
                spec.physical_kv_target, ".2f"
            ),
        }
    )
    return values


def _config_diff(
    left: Mapping[str, str], right: Mapping[str, str]
) -> dict[str, tuple[str | None, str | None]]:
    return {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def _underlying_formal_order_contract() -> dict[str, Any]:
    text = UNDERLYING_EXPERIMENT_RUNNER.read_text(encoding="utf-8")
    if (
        "if args.formal_order_index not in range(4):" not in text
        or 'ValueError("--formal-order-index must be in [0, 3]")' not in text
    ):
        raise LiveSensitivityError(
            "underlying formal-order validation no longer matches [0, 3]"
        )
    return {
        "path": formal.repository_relative(UNDERLYING_EXPERIMENT_RUNNER),
        "sha256": _sha256(UNDERLYING_EXPERIMENT_RUNNER),
        "minimum": FORMAL_ORDER_INDEX_MIN,
        "maximum": FORMAL_ORDER_INDEX_MAX,
        "maximum_cell_count": MAX_FORMAL_CELLS,
    }


def _matrix_invariants(
    specs: Sequence[CellSpec],
    base: Mapping[str, str],
    *,
    gpus: str,
    port: int,
) -> dict[str, Any]:
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise LiveSensitivityError("cell labels are not unique")
    underlying_order_contract = _underlying_formal_order_contract()
    planned_order_indices = list(range(len(specs)))
    invalid_order_indices = [
        index
        for index in planned_order_indices
        if not FORMAL_ORDER_INDEX_MIN <= index <= FORMAL_ORDER_INDEX_MAX
    ]
    if invalid_order_indices:
        raise LiveSensitivityError(
            "planned formal order indices exceed the underlying runner range "
            f"[{FORMAL_ORDER_INDEX_MIN}, {FORMAL_ORDER_INDEX_MAX}]: "
            f"{invalid_order_indices}"
        )
    configs = {
        spec.label: _derived_config(base, spec, gpus=gpus, port=port)
        for spec in specs
    }
    max_model_len = int(base["VLLM_MAX_MODEL_LEN"])
    fixed_final_tokens = int(base["PASTE_LIVE_FIXED_FINAL_COMPLETION_TOKENS"])
    predicted_visit_tokens = int(
        base["PASTE_LIVE_PREDICTED_VISIT_RESULT_TOKENS"]
    )
    conservative_prompt_margin = 512
    safe_context_padding_ceiling = (
        max_model_len
        - fixed_final_tokens
        - predicted_visit_tokens
        - conservative_prompt_margin
    )
    if any(
        spec.context_padding_tokens > safe_context_padding_ceiling
        for spec in specs
    ):
        raise LiveSensitivityError(
            "context suite exceeds the conservative fixed-final model headroom"
        )
    native_sequence_ceiling = int(base["VLLM_MAX_NUM_SEQS"])
    if any(spec.max_active_tasks >= native_sequence_ceiling for spec in specs):
        raise LiveSensitivityError(
            "offered load must remain strictly below native max-num-seqs"
        )

    pair_checks: list[dict[str, Any]] = []
    for group in sorted({spec.pair_group for spec in specs}):
        members = [spec for spec in specs if spec.pair_group == group]
        a_cells = [spec for spec in members if spec.cell == "A"]
        e093_cells = [
            spec
            for spec in members
            if spec.cell == "E"
            and math.isclose(spec.physical_kv_target, REFERENCE_TARGET)
        ]
        if a_cells and e093_cells:
            if len(a_cells) != 1 or len(e093_cells) != 1:
                raise LiveSensitivityError(f"pair {group} is not unique")
            left, right = a_cells[0], e093_cells[0]
            diff = _config_diff(configs[left.label], configs[right.label])
            if diff:
                raise LiveSensitivityError(
                    f"A/E pair {group} changes non-treatment config: {diff}"
                )
            pair_checks.append(
                {
                    "pair_group": group,
                    "baseline": left.label,
                    "candidate": right.label,
                    "common_config_diff": diff,
                    "only_scheduler_treatment_changes": True,
                }
            )

    target_cells = [
        spec
        for spec in specs
        if spec.cell == "E" and spec.is_reference_shape
    ]
    target_checks: list[dict[str, Any]] = []
    reference = next(
        (
            spec
            for spec in target_cells
            if math.isclose(spec.physical_kv_target, REFERENCE_TARGET)
        ),
        None,
    )
    if reference is not None:
        for spec in target_cells:
            diff = _config_diff(configs[reference.label], configs[spec.label])
            expected = (
                set()
                if spec.label == reference.label
                else {
                    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"
                }
            )
            if set(diff) != expected:
                raise LiveSensitivityError(
                    f"target sweep {spec.label} has non-isolated diff: {diff}"
                )
            target_checks.append(
                {
                    "reference": reference.label,
                    "candidate": spec.label,
                    "changed_keys": list(diff),
                    "only_active_physical_kv_target_changes": True,
                }
            )

    environment_checks: list[dict[str, Any]] = []
    for spec in specs:
        environment = formal._cell_environment(configs[spec.label], cell=spec.cell)
        scheduler_keys = sorted(
            key for key in environment if key.startswith("VLLM_SCHED_")
        )
        if spec.cell == "A":
            if scheduler_keys != ["VLLM_SCHED_POLICY"]:
                raise LiveSensitivityError(
                    f"A cell leaked scheduler extensions: {scheduler_keys}"
                )
            target_value: str | None = None
        else:
            if (
                environment.get("VLLM_SCHED_POLICY")
                != "online_joint_pacer_v2"
                or environment.get(
                    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION"
                )
                != "1"
            ):
                raise LiveSensitivityError(f"E cell {spec.label} is not physical Joint-v2")
            target_value = environment.get(
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"
            )
            if target_value != format(spec.physical_kv_target, ".2f"):
                raise LiveSensitivityError(
                    f"E cell {spec.label} target did not reach server environment"
                )
        environment_checks.append(
            {
                "cell": spec.label,
                "policy": environment["VLLM_SCHED_POLICY"],
                "physical_kv_target_visible_to_server": target_value,
                "legacy_pressure_band_is_swept": False,
            }
        )

    return {
        "all_cells_use_same_workload": True,
        "workload_path": formal.repository_relative(WORKLOAD),
        "workload_sha256": _sha256(WORKLOAD),
        "fresh_server_per_cell": True,
        "cross_cell_state_reuse": False,
        "formal_order_index_gate": {
            "underlying_runner": underlying_order_contract,
            "underlying_runner_range": [
                FORMAL_ORDER_INDEX_MIN,
                FORMAL_ORDER_INDEX_MAX,
            ],
            "planned_indices": planned_order_indices,
            "all_indices_in_range": True,
            "cell_count": len(specs),
            "maximum_cell_count": MAX_FORMAL_CELLS,
        },
        "context_headroom": {
            "max_model_len": max_model_len,
            "fixed_final_tokens": fixed_final_tokens,
            "predicted_visit_result_tokens": predicted_visit_tokens,
            "conservative_prompt_margin_tokens": conservative_prompt_margin,
            "safe_context_padding_ceiling": safe_context_padding_ceiling,
            "maximum_planned_context_padding": max(
                spec.context_padding_tokens for spec in specs
            ),
        },
        "offered_load_is_strictly_below_native_sequence_ceiling": True,
        "native_sequence_ceiling": native_sequence_ceiling,
        "pair_checks": pair_checks,
        "target_checks": target_checks,
        "environment_checks": environment_checks,
        "transport_contract": {
            "remediation_version": TRANSPORT_REMEDIATION_VERSION,
            "visit_min_start_interval_s": ROBUST_VISIT_MIN_START_INTERVAL_S,
            "http_max_attempts": int(
                base["PASTE_LIVE_TOOL_HTTP_MAX_ATTEMPTS"]
            ),
            "accepted_http_attempts_per_tool_invocation": 1,
            "zero_retries_required": True,
            "same_for_every_a_e_cell": True,
            "failed_r2_cells_reused": False,
            "one_shot_replacement": True,
            "no_further_auto_rerun_or_transport_escalation": True,
        },
    }


def _bindings(extra_paths: Sequence[Path] = ()) -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        BASE_CONFIG,
        WORKLOAD,
        *formal.BOUND_CODE_PATHS,
        *extra_paths,
    )
    unique = {path.resolve() for path in paths}
    return {
        formal.repository_relative(path): _sha256(path)
        for path in sorted(unique, key=str)
    }


def _verify_bindings(bindings: Mapping[str, str]) -> None:
    for relative, expected in bindings.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise LiveSensitivityError(f"bound input changed: {relative}")


def _canonical_json_sha256(payload: Any) -> str:
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _normalized_replacement_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only run/block/order/server identity from a planned cell."""

    command = cell.get("runner_command")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise LiveSensitivityError("shape replacement cell lacks a runner command")
    identity_options = {
        "--output-dir": "<RUN_OUTPUT_DIRECTORY>",
        "--server-url": "<SERVER_IDENTITY_URL>",
        "--cell-label": "<RUN_CELL_LABEL>",
        "--formal-block-id": "<FORMAL_BLOCK_ID>",
        "--formal-order-index": "<FORMAL_ORDER_INDEX>",
        "--server-instance-id": "<SERVER_INSTANCE_ID>",
    }
    normalized_command: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        normalized_command.append(item)
        if item in identity_options:
            if index + 1 >= len(command):
                raise LiveSensitivityError(
                    f"shape replacement command lacks a value for {item}"
                )
            normalized_command.append(identity_options[item])
            index += 2
        else:
            index += 1
    normalized_cell = {
        key: value
        for key, value in cell.items()
        if key not in {"order_index", "runner_command", "server_state_directory"}
    }
    normalized_cell["runner_command"] = normalized_command
    return normalized_cell


def _shape_r1_harness_failure_provenance(
    replacement_cells: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Bind and disclose the excluded deterministic shape-r1 harness failure."""

    if any(not path.is_file() for path in FAILED_SHAPE_BOUND_PATHS):
        missing = [str(path) for path in FAILED_SHAPE_BOUND_PATHS if not path.is_file()]
        raise LiveSensitivityError(f"shape-r1 provenance is incomplete: {missing}")
    try:
        old_plan = json.loads(FAILED_SHAPE_BOUND_PATHS[0].read_text(encoding="utf-8"))
        failure = json.loads(FAILED_SHAPE_BOUND_PATHS[1].read_text(encoding="utf-8"))
        failed_contract = json.loads(
            FAILED_SHAPE_BOUND_PATHS[2].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveSensitivityError("shape-r1 provenance is not valid JSON") from exc
    stderr_text = FAILED_SHAPE_BOUND_PATHS[3].read_text(
        encoding="utf-8", errors="replace"
    )
    if not all(
        isinstance(item, Mapping) for item in (old_plan, failure, failed_contract)
    ):
        raise LiveSensitivityError("shape-r1 provenance JSON is not an object")
    old_cells = old_plan.get("cells")
    if (
        old_plan.get("schema") != "paste_repro.scheduler_live_sensitivity_plan"
        or old_plan.get("version") != 1
        or old_plan.get("run_tag") != "comment3-shape-r1"
        or old_plan.get("suite") != "shape"
        or old_plan.get("cell_count") != 6
        or not isinstance(old_cells, list)
        or len(old_cells) != 6
        or failure.get("schema")
        != "paste_repro.scheduler_live_sensitivity_failure"
        or failure.get("version") != 1
        or failure.get("error_type") != "LiveSensitivityError"
        or failure.get("error") != "a-c12k-l80 live runner failed"
    ):
        raise LiveSensitivityError("shape-r1 plan/failure identity drifted")
    failed_spec = failed_contract.get("spec")
    if (
        failed_contract.get("schema")
        != "paste_repro.scheduler_live_sensitivity_cell_contract"
        or failed_contract.get("version") != 1
        or failed_contract.get("order_index") != 4
        or not isinstance(failed_spec, Mapping)
        or failed_spec.get("label") != "a-c12k-l80"
        or failed_spec.get("cell") != "A"
        or failed_spec.get("context_padding_tokens") != 12_000
        or failed_spec.get("max_active_tasks") != 80
        or "ValueError: --formal-order-index must be in [0, 3]" not in stderr_text
    ):
        raise LiveSensitivityError("shape-r1 deterministic failure evidence drifted")

    failed_cell_root = FAILED_SHAPE_RUN_ROOT / "cells/05-a-c12k-l80"
    failed_result = failed_cell_root / "evidence/result.json"
    failed_timeline = failed_cell_root / "evidence/queue_timeline.jsonl"
    failed_manifest = failed_cell_root / "cell_manifest.json"
    failed_stdout = failed_cell_root / "runner.stdout.log"
    failed_server = FAILED_SHAPE_BOUND_PATHS[4]
    lifecycle_stdout = FAILED_SHAPE_BOUND_PATHS[5]
    lifecycle_stderr = FAILED_SHAPE_BOUND_PATHS[6]
    server_text = failed_server.read_text(encoding="utf-8", errors="replace")
    lifecycle_text = lifecycle_stdout.read_text(encoding="utf-8", errors="replace")
    if (
        failed_result.exists()
        or failed_timeline.exists()
        or failed_manifest.exists()
        or not failed_stdout.is_file()
        or failed_stdout.stat().st_size != 0
        or 'POST /v1/chat/completions' in server_text
        or re.search(r"vLLM pid [0-9]+ stopped cleanly\.", lifecycle_text) is None
        or lifecycle_stderr.stat().st_size != 0
    ):
        raise LiveSensitivityError("shape-r1 failed cell was not a zero-request failure")

    excluded_prefix = old_cells[:4]
    excluded_labels = [str(cell.get("label")) for cell in excluded_prefix]
    if excluded_labels != [
        "a-c5k-l40",
        "e-c5k-l40-u093",
        "a-c10k-l80",
        "e-c10k-l80-u093",
    ]:
        raise LiveSensitivityError("shape-r1 observed-prefix identity drifted")
    for index, label in enumerate(excluded_labels):
        cell_root = FAILED_SHAPE_RUN_ROOT / "cells" / f"{index + 1:02d}-{label}"
        manifest_path = cell_root / "cell_manifest.json"
        validation_path = cell_root / "strict_development_validation.json"
        result_path = cell_root / "evidence/result.json"
        if not all(path.is_file() for path in (manifest_path, validation_path, result_path)):
            raise LiveSensitivityError(f"shape-r1 observed cell {label} is incomplete")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            raise LiveSensitivityError(f"shape-r1 observed cell {label} was not valid")

    old_high = {
        str(cell.get("label")): cell
        for cell in old_cells[4:]
        if isinstance(cell, Mapping)
    }
    new_high = {
        str(cell.get("label")): cell
        for cell in replacement_cells
        if isinstance(cell, Mapping)
    }
    expected_high_labels = {"a-c12k-l80", "e-c12k-l80-u093"}
    if set(old_high) != expected_high_labels or set(new_high) != expected_high_labels:
        raise LiveSensitivityError("shape high-pair replacement identities drifted")
    equivalence: list[dict[str, Any]] = []
    for label in ("a-c12k-l80", "e-c12k-l80-u093"):
        old_normalized = _normalized_replacement_cell(old_high[label])
        new_normalized = _normalized_replacement_cell(new_high[label])
        old_sha = _canonical_json_sha256(old_normalized)
        new_sha = _canonical_json_sha256(new_normalized)
        if old_sha != new_sha or old_normalized != new_normalized:
            raise LiveSensitivityError(
                f"shape high-pair replacement changed non-identity config: {label}"
            )
        equivalence.append(
            {
                "cell": label,
                "old_normalized_sha256": old_sha,
                "replacement_normalized_sha256": new_sha,
                "equal_after_identity_normalization": True,
            }
        )

    bound_files = {
        formal.repository_relative(path): _sha256(path)
        for path in FAILED_SHAPE_BOUND_PATHS
    }
    if any(bindings.get(relative) != digest for relative, digest in bound_files.items()):
        raise LiveSensitivityError("shape-r1 provenance is not in the run bindings")
    runner_binding_key = formal.repository_relative(Path(__file__).resolve())
    historical_runner_sha = old_plan.get("bindings", {}).get(runner_binding_key)
    replacement_runner_sha = bindings.get(runner_binding_key)
    if historical_runner_sha != FAILED_SHAPE_RUNNER_SHA256:
        raise LiveSensitivityError("shape-r1 historical runner binding drifted")
    if (
        not isinstance(replacement_runner_sha, str)
        or replacement_runner_sha == historical_runner_sha
    ):
        raise LiveSensitivityError("shape-r1 replacement runner binding is invalid")
    return {
        "version": SHAPE_HARNESS_REPAIR_VERSION,
        "failed_run_tag": "comment3-shape-r1",
        "failure_class": "deterministic_formal_order_index_harness_failure",
        "underlying_formal_order_range": [
            FORMAL_ORDER_INDEX_MIN,
            FORMAL_ORDER_INDEX_MAX,
        ],
        "rejected_order_index": 4,
        "failed_cell": "a-c12k-l80",
        "failed_cell_request_count": 0,
        "failed_cell_result_present": False,
        "failed_cell_absence_checks": {
            "chat_completion_post_count": 0,
            "result_absent": True,
            "queue_timeline_absent": True,
            "cell_manifest_absent": True,
            "runner_stdout_empty": True,
            "server_stopped_cleanly": True,
            "lifecycle_stderr_empty": True,
        },
        "bound_files": bound_files,
        "runner_bindings": {
            "path": runner_binding_key,
            "historical_sha256": historical_runner_sha,
            "replacement_sha256": replacement_runner_sha,
            "historical_artifact_requires_historical_sha256": True,
            "replacement_runner_sha_differs": True,
        },
        "excluded_observed_prefix": {
            "cell_count": 4,
            "cells": excluded_labels,
            "reused_by_replacement": False,
            "pooled_with_replacement": False,
        },
        "replacement": {
            "suite": "high",
            "fixed_order": ["a-c12k-l80", "e-c12k-l80-u093"],
            "cell_count": 2,
            "first_four_cells_rerun": False,
            "failed_shape_run_resumed": False,
            "historical_shape_artifact_requires_original_bound_runner_sha": True,
            "new_runner_may_not_resume_historical_shape_artifact": True,
            "selection_or_tuning_from_observed_prefix": False,
            "allowed_identity_differences": [
                "run/output path",
                "formal block id",
                "formal order index",
                "server instance id",
                "server URL/port",
            ],
            "configuration_equivalence": equivalence,
            "one_shot_replacement": True,
            "no_further_auto_rerun": True,
        },
    }


def _preflight(
    specs: Sequence[CellSpec], *, gpus: str, port: int
) -> tuple[dict[str, str], Path, Path, dict[str, Any]]:
    _parse_gpus(gpus)
    if not 1 <= port <= 65_535:
        raise LiveSensitivityError("--port must be in 1..65535")
    base = formal.load_frozen_config(BASE_CONFIG)
    python = Path(base["PASTE_ENV_PREFIX"]) / "bin/python"
    if not python.is_file():
        raise LiveSensitivityError(f"pinned Python is missing: {python}")
    formal.validate_entrypoints(python=python)
    if not WORKLOAD.is_file() or _sha256(WORKLOAD) != formal.FORMAL_WORKLOAD_SHA256:
        raise LiveSensitivityError("frozen formal-v8 workload SHA256 mismatch")
    workload_validation = formal.validate_formal_workload(
        python=python, workload=WORKLOAD
    )
    model_snapshot = formal._model_snapshot(base)
    if not model_snapshot.is_dir() or not (model_snapshot / "config.json").is_file():
        raise LiveSensitivityError(f"pinned model snapshot is missing: {model_snapshot}")
    grammar = formal.validate_fixed_final_grammar_feasibility(
        workload=WORKLOAD,
        model_snapshot=model_snapshot,
        expected_source_count=EXPECTED_SOURCE_COUNT,
    )
    invariants = _matrix_invariants(specs, base, gpus=gpus, port=port)
    return base, python, model_snapshot, {
        "workload_validation": workload_validation,
        "fixed_final_grammar_feasibility": grammar,
        "matrix_invariants": invariants,
    }


def _planned_command(
    *,
    spec: CellSpec,
    index: int,
    config: Mapping[str, str],
    python: Path,
    run_root: Path,
) -> list[str]:
    return formal._runner_command(
        python=python,
        workload=WORKLOAD,
        output=run_root / "cells" / f"{index + 1:02d}-{spec.label}" / "evidence",
        cell=spec.cell,
        block_id=f"{run_root.name}-{spec.label}",
        order_index=index,
        server_instance_id=f"check-only-{index + 1:02d}-{spec.label}",
        config=config,
    )


def _plan(
    *,
    run_tag: str,
    suite: str,
    specs: Sequence[CellSpec],
    base: Mapping[str, str],
    python: Path,
    gpus: str,
    port: int,
    preflight: Mapping[str, Any],
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    run_root = RUN_BASE / run_tag
    cells: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        config = _derived_config(base, spec, gpus=gpus, port=port)
        cells.append(
            {
                **asdict(spec),
                "order_index": index,
                "fresh_server": True,
                "server_state_directory": formal.repository_relative(
                    run_root
                    / "cells"
                    / f"{index + 1:02d}-{spec.label}"
                    / "state"
                ),
                "runner_command": _planned_command(
                    spec=spec,
                    index=index,
                    config=config,
                    python=python,
                    run_root=run_root,
                ),
                "effective_overrides": {
                    "PASTE_LIVE_VISIT_MIN_START_INTERVAL_S": format(
                        ROBUST_VISIT_MIN_START_INTERVAL_S, ".1f"
                    ),
                    "PASTE_LIVE_MAX_ACTIVE_TASKS": str(spec.max_active_tasks),
                    "PASTE_LIVE_CONTEXT_PADDING_TOKENS": str(
                        spec.context_padding_tokens
                    ),
                    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": format(
                        spec.physical_kv_target, ".2f"
                    ),
                },
            }
        )
    plan: dict[str, Any] = {
        "schema": "paste_repro.scheduler_live_sensitivity_plan",
        "version": 1,
        "run_tag": run_tag,
        "suite": suite,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "reviewer_requested_posthoc_robustness": True,
        "selection_or_tuning_allowed": False,
        "transport_remediation_after_failed_pilot": {
            "version": TRANSPORT_REMEDIATION_VERSION,
            "failed_run_tag": "comment3-target-r2",
            "excluded_predecessor_run_tags": [
                "comment3-target-r1",
                "comment3-target-r2",
            ],
            "failure_class": "external_jina_http_429",
            "failed_run_reused": False,
            "failed_run_performance_was_observable": True,
            "all_cells_rebaselined": True,
            "scheduler_parameter_selected_from_failed_run": False,
            "not_preregistered": True,
            "no_cross_transport_pooling_or_comparison": True,
            "one_shot_replacement": True,
            "no_further_auto_rerun_or_transport_escalation": True,
            "zero_http_retries_required": True,
        },
        "gpu_or_server_touched_by_check_only": False,
        "network_touched_by_check_only": False,
        "run_root": formal.repository_relative(run_root),
        "gpu_ids": list(_parse_gpus(gpus)),
        "port": port,
        "cell_count": len(cells),
        "estimated_duration": {
            "historical_task_phase_makespan_s": [197, 237],
            "budget_minutes_per_fresh_server_cell": [6, 12],
            "note": "server load and live HTTP variance dominate; estimate is not a measured promise",
        },
        "evidence_boundary": {
            "same_model_family": base["MODEL_ID"],
            "same_gpu_sku_unless_operator_changes_hardware": True,
            "context_load_and_capacity_shape_sensitivity_not_cross_gpu_proof": True,
        },
        "preflight": dict(preflight),
        "bindings": dict(bindings),
        "cells": cells,
    }
    if suite == "high":
        plan["shape_r1_harness_repair"] = _shape_r1_harness_failure_provenance(
            cells,
            bindings,
        )
    return plan


def _load_workload_source_ids() -> set[str]:
    payload = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    return {str(row["source_id"]) for row in payload["sources"]}


def _validate_transport_attempts(result: Mapping[str, Any]) -> dict[str, Any]:
    """Require a clean, retry-free external-tool transport in every cell."""

    attempts = result.get("tool_attempt_records")
    if not isinstance(attempts, list) or len(attempts) != EXPECTED_TOOL_COMMITS:
        raise LiveSensitivityError("tool-attempt identity matrix is incomplete")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise LiveSensitivityError(f"tool attempt {index} is malformed")
        attempt_log = attempt.get("http_attempt_log")
        clean_log = (
            isinstance(attempt_log, list)
            and len(attempt_log) == 1
            and isinstance(attempt_log[0], Mapping)
            and attempt_log[0].get("status") == 200
            and attempt_log[0].get("retried") is False
        )
        if (
            attempt.get("authoritative") is not True
            or attempt.get("speculative") is not False
            or attempt.get("committed") is not True
            or attempt.get("outcome") != "committed"
            or attempt.get("http_attempts") != 1
            or attempt.get("response_status") != 200
            or not clean_log
        ):
            raise LiveSensitivityError(
                f"tool attempt {index} violates the retry-free transport gate"
            )
    return {
        "remediation_version": TRANSPORT_REMEDIATION_VERSION,
        "visit_min_start_interval_s": ROBUST_VISIT_MIN_START_INTERVAL_S,
        "tool_invocation_count": EXPECTED_TOOL_COMMITS,
        "physical_http_attempt_count": EXPECTED_TOOL_COMMITS,
        "http_retry_count": 0,
        "http_429_count": 0,
        "all_status_200": True,
    }


def _validate_result(
    result: Mapping[str, Any],
    *,
    spec: CellSpec,
    config: Mapping[str, str],
    block_id: str,
    order_index: int,
    server_instance_id: str,
    timeline_path: Path,
) -> dict[str, Any]:
    result_config = result.get("config")
    summary = result.get("summary")
    tasks = result.get("tasks")
    events = result.get("llm_events")
    broker = result.get("broker_final_snapshot")
    if not all(
        isinstance(item, Mapping)
        for item in (result_config, summary, broker)
    ) or not isinstance(tasks, list) or not isinstance(events, list):
        raise LiveSensitivityError(f"{spec.label} result structure is incomplete")
    assert isinstance(result_config, Mapping)
    assert isinstance(summary, Mapping)
    assert isinstance(broker, Mapping)
    expected_config = {
        "model": config["MODEL_ID"],
        "call_graph_mode": "frozen",
        "speculation_mode": "off",
        "independent_source_count": EXPECTED_SOURCE_COUNT,
        "replicas": 1,
        "task_count": EXPECTED_SOURCE_COUNT,
        "max_active_tasks": spec.max_active_tasks,
        "context_padding_tokens": spec.context_padding_tokens,
        "visit_min_start_interval_s": ROBUST_VISIT_MIN_START_INTERVAL_S,
        "tool_http_max_attempts": int(
            config["PASTE_LIVE_TOOL_HTTP_MAX_ATTEMPTS"]
        ),
        "tool_http_attempt_start_gate_enabled": True,
        "workload_file_sha256": formal.FORMAL_WORKLOAD_SHA256,
        "workload_split_id": "live-joint-wikipedia-frozen-formal-v8",
        "workload_formal_eligible": True,
    }
    drift = {
        key: (expected, result_config.get(key))
        for key, expected in expected_config.items()
        if result_config.get(key) != expected
    }
    if drift:
        raise LiveSensitivityError(f"{spec.label} result config drift: {drift}")
    formal_run = result_config.get("formal_run")
    expected_formal = {
        "block_id": block_id,
        "cell_id": spec.cell,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "fresh_server": True,
        "result_cache_empty": True,
        "broker_drained": True,
    }
    if formal_run != expected_formal:
        raise LiveSensitivityError(f"{spec.label} fresh-server identity drift")
    scheduler = result_config.get("scheduler_environment")
    if not isinstance(scheduler, Mapping):
        raise LiveSensitivityError(f"{spec.label} lacks scheduler environment")
    if scheduler.get("VLLM_SCHED_POLICY") != (
        "fcfs" if spec.cell == "A" else "online_joint_pacer_v2"
    ):
        raise LiveSensitivityError(f"{spec.label} policy mismatch")
    target = scheduler.get(
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"
    )
    if spec.cell == "A":
        if target is not None:
            raise LiveSensitivityError(f"{spec.label} baseline leaked target control")
    elif target != format(spec.physical_kv_target, ".2f"):
        raise LiveSensitivityError(f"{spec.label} active target mismatch")

    required_summary = {
        "task_count": EXPECTED_SOURCE_COUNT,
        "successful_task_count": EXPECTED_SOURCE_COUNT,
        "failed_task_count": 0,
        "all_tasks_succeeded": True,
    }
    if any(summary.get(key) != value for key, value in required_summary.items()):
        raise LiveSensitivityError(f"{spec.label} task completion gate failed")
    llm = summary.get("llm")
    tool = summary.get("tool")
    if not isinstance(llm, Mapping) or not isinstance(tool, Mapping):
        raise LiveSensitivityError(f"{spec.label} lacks LLM/tool summaries")
    if (
        llm.get("request_count") != EXPECTED_LLM_REQUESTS
        or llm.get("successful_request_count") != EXPECTED_LLM_REQUESTS
        or llm.get("exactly_one_attempt_each") is not True
        or tool.get("authoritative_commit_count") != EXPECTED_TOOL_COMMITS
    ):
        raise LiveSensitivityError(f"{spec.label} request/tool gate failed")
    transport_validation = _validate_transport_attempts(result)
    expected_sources = _load_workload_source_ids()
    observed_sources = {
        str(task.get("source_id"))
        for task in tasks
        if isinstance(task, Mapping) and task.get("ok") is True
    }
    if observed_sources != expected_sources or len(tasks) != EXPECTED_SOURCE_COUNT:
        raise LiveSensitivityError(f"{spec.label} source identity gate failed")
    if any(
        not isinstance(task, Mapping)
        or task.get("context_padding_target_tokens")
        != spec.context_padding_tokens
        for task in tasks
    ):
        raise LiveSensitivityError(f"{spec.label} context padding gate failed")
    counts = broker.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(key, -1)) != 0
        for key in (
            "queued_authoritative",
            "queued_speculative",
            "running_authoritative",
            "running_speculative",
            "completed_unclaimed_speculative",
        )
    ):
        raise LiveSensitivityError(f"{spec.label} broker did not drain")
    raw = result.get("raw_evidence")
    timeline = raw.get("queue_timeline") if isinstance(raw, Mapping) else None
    if (
        not isinstance(timeline, Mapping)
        or timeline.get("sha256") != _sha256(timeline_path)
    ):
        raise LiveSensitivityError(f"{spec.label} queue timeline binding failed")
    return {
        "valid": True,
        "task_count": EXPECTED_SOURCE_COUNT,
        "llm_request_count": EXPECTED_LLM_REQUESTS,
        "authoritative_tool_commit_count": EXPECTED_TOOL_COMMITS,
        "all_sources_exactly_once": True,
        "fresh_server_identity": True,
        "scheduler_policy": scheduler["VLLM_SCHED_POLICY"],
        "physical_kv_target_visible_to_server": target,
        "transport_validation": transport_validation,
    }


def _run_cell(
    *,
    run_root: Path,
    spec: CellSpec,
    order_index: int,
    base: Mapping[str, str],
    python: Path,
    model_snapshot: Path,
    gpus: str,
    port: int,
    bindings: Mapping[str, str],
) -> Path:
    _verify_bindings(bindings)
    cell_root = run_root / "cells" / f"{order_index + 1:02d}-{spec.label}"
    cell_root.mkdir(parents=True, exist_ok=False)
    server_dir = cell_root / "server"
    state_dir = cell_root / "state"
    evidence_dir = cell_root / "evidence"
    server_dir.mkdir()
    state_dir.mkdir()
    lifecycle_stdout = cell_root / "server_lifecycle.stdout.log"
    lifecycle_stderr = cell_root / "server_lifecycle.stderr.log"
    runner_stdout = cell_root / "runner.stdout.log"
    runner_stderr = cell_root / "runner.stderr.log"
    server_instance_id = str(uuid.uuid4())
    block_id = f"{run_root.name}-{spec.label}"
    config = _derived_config(base, spec, gpus=gpus, port=port)
    environment = formal._cell_environment(config, cell=spec.cell)
    environment.update(
        {
            "VLLM_REQUIRE_NEW": "1",
            "VLLM_STATE_DIR": str(state_dir),
            "VLLM_LOG_DIR": str(server_dir),
            "VLLM_HOOK_DIR": str(REPOSITORY_ROOT / "scripts/pythonhooks"),
            "MODEL_SNAPSHOT": str(model_snapshot),
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = formal._runner_command(
        python=python,
        workload=WORKLOAD,
        output=evidence_dir,
        cell=spec.cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        config=config,
    )
    contract = {
        "schema": "paste_repro.scheduler_live_sensitivity_cell_contract",
        "version": 1,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "spec": asdict(spec),
        "order_index": order_index,
        "block_id": block_id,
        "server_instance_id": server_instance_id,
        "fresh_server_required": True,
        "result_cache_empty_required": True,
        "state_directory": formal.repository_relative(state_dir),
        "workload": {
            "path": formal.repository_relative(WORKLOAD),
            "sha256": formal.FORMAL_WORKLOAD_SHA256,
        },
        "treatment": {
            "policy": environment["VLLM_SCHED_POLICY"],
            "physical_kv_admission": environment.get(
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION"
            ),
            "physical_kv_target": environment.get(
                "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"
            ),
            "context_padding_tokens": spec.context_padding_tokens,
            "max_active_tasks": spec.max_active_tasks,
        },
        "transport_contract": {
            "remediation_version": TRANSPORT_REMEDIATION_VERSION,
            "visit_min_start_interval_s": ROBUST_VISIT_MIN_START_INTERVAL_S,
            "http_max_attempts": int(
                config["PASTE_LIVE_TOOL_HTTP_MAX_ATTEMPTS"]
            ),
            "accepted_http_attempts_per_tool_invocation": 1,
            "zero_retries_required": True,
        },
        "runner_command": command,
        "bindings": dict(bindings),
    }
    formal.write_json_atomic(cell_root / "cell_contract.json", contract)
    print(
        f"[{order_index + 1}] starting {spec.label}: policy={spec.cell}, "
        f"context={spec.context_padding_tokens}, load={spec.max_active_tasks}, "
        f"target={spec.physical_kv_target:.2f}",
        flush=True,
    )
    started_wall = time.time()
    server_started = False
    primary_error: BaseException | None = None
    try:
        start_code = formal._run_logged(
            [str(formal.START_SERVER)],
            env=environment,
            stdout_path=lifecycle_stdout,
            stderr_path=lifecycle_stderr,
        )
        if start_code != 0:
            raise LiveSensitivityError(f"{spec.label} fresh vLLM start failed")
        server_started = True
        runner_code = formal._run_logged(
            command,
            env=environment,
            stdout_path=runner_stdout,
            stderr_path=runner_stderr,
        )
        if runner_code != 0:
            raise LiveSensitivityError(f"{spec.label} live runner failed")
    except BaseException as exc:
        primary_error = exc
    finally:
        if server_started:
            stop_code = formal._run_logged(
                [str(formal.STOP_SERVER)],
                env=environment,
                stdout_path=lifecycle_stdout,
                stderr_path=lifecycle_stderr,
            )
            if stop_code != 0 and primary_error is None:
                primary_error = LiveSensitivityError(
                    f"{spec.label} vLLM did not stop cleanly"
                )
    if primary_error is not None:
        raise primary_error

    result_path = evidence_dir / "result.json"
    timeline_path = evidence_dir / "queue_timeline.jsonl"
    server_log = server_dir / f"vllm_{port}.log"
    for required in (result_path, timeline_path, server_log):
        if not required.is_file():
            raise LiveSensitivityError(f"{spec.label} missing evidence: {required}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = _validate_result(
        result,
        spec=spec,
        config=config,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        timeline_path=timeline_path,
    )
    validation["elapsed_wall_s_including_server_lifecycle"] = time.time() - started_wall
    formal.write_json_atomic(cell_root / "strict_development_validation.json", validation)
    evidence_files = (
        cell_root / "cell_contract.json",
        cell_root / "strict_development_validation.json",
        result_path,
        timeline_path,
        server_log,
        lifecycle_stdout,
        lifecycle_stderr,
        runner_stdout,
        runner_stderr,
    )
    manifest = {
        "schema": "paste_repro.scheduler_live_sensitivity_cell_evidence",
        "version": 1,
        "development_only": True,
        "cell": spec.label,
        "evidence": {
            formal.repository_relative(path): _sha256(path)
            for path in evidence_files
        },
    }
    formal.write_json_atomic(cell_root / "cell_manifest.json", manifest)
    _verify_bindings(bindings)
    print(f"[{order_index + 1}] completed {spec.label}; server stopped", flush=True)
    return cell_root


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cell_values(path: Path) -> tuple[dict[str, float], list[float]]:
    result = json.loads((path / "evidence/result.json").read_text(encoding="utf-8"))
    by_source = {
        str(task["source_id"]): float(task["e2e_s"])
        for task in result["tasks"]
        if task.get("ok") is True
    }
    request = [
        float(event["duration_s"])
        for event in result["llm_events"]
        if event.get("ok") is True
    ]
    return by_source, request


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _summarize(
    run_root: Path, specs: Sequence[CellSpec], cell_paths: Mapping[str, Path]
) -> dict[str, Any]:
    sources: dict[str, dict[str, float]] = {}
    cells: dict[str, Any] = {}
    for spec in specs:
        source_values, request_values = _cell_values(cell_paths[spec.label])
        sources[spec.label] = source_values
        cells[spec.label] = {
            "spec": asdict(spec),
            "task_e2e_s": _distribution(list(source_values.values())),
            "llm_request_duration_s": _distribution(request_values),
            "transport_validation": json.loads(
                (
                    cell_paths[spec.label]
                    / "strict_development_validation.json"
                ).read_text(encoding="utf-8")
            )["transport_validation"],
        }
    effects: list[dict[str, Any]] = []
    for group in sorted({spec.pair_group for spec in specs}):
        baseline = next(
            (spec for spec in specs if spec.pair_group == group and spec.cell == "A"),
            None,
        )
        if baseline is None:
            continue
        for candidate in (
            spec
            for spec in specs
            if spec.pair_group == group and spec.cell == "E"
        ):
            base = sources[baseline.label]
            observed = sources[candidate.label]
            if set(base) != set(observed):
                raise LiveSensitivityError(f"source mismatch in {baseline.label}/{candidate.label}")
            base_mean = statistics.fmean(base.values())
            candidate_mean = statistics.fmean(observed.values())
            effects.append(
                {
                    "pair_group": group,
                    "baseline": baseline.label,
                    "candidate": candidate.label,
                    "baseline_mean_s": base_mean,
                    "candidate_mean_s": candidate_mean,
                    "relative_reduction": (base_mean - candidate_mean) / base_mean,
                    "faster_source_count": sum(
                        base[source] > observed[source] for source in base
                    ),
                    "single_run_per_cell_no_confidence_interval": True,
                }
            )
    target_reference = next(
        (
            spec
            for spec in specs
            if spec.cell == "E"
            and spec.is_reference_shape
            and math.isclose(spec.physical_kv_target, REFERENCE_TARGET)
        ),
        None,
    )
    target_sensitivity: list[dict[str, Any]] = []
    if target_reference is not None:
        reference_values = sources[target_reference.label]
        reference_mean = statistics.fmean(reference_values.values())
        for candidate in (
            spec for spec in specs if spec.cell == "E" and spec.is_reference_shape
        ):
            candidate_values = sources[candidate.label]
            candidate_mean = statistics.fmean(candidate_values.values())
            target_sensitivity.append(
                {
                    "reference": target_reference.label,
                    "candidate": candidate.label,
                    "target": candidate.physical_kv_target,
                    "mean_s": candidate_mean,
                    "relative_change_vs_u093": (
                        candidate_mean - reference_mean
                    )
                    / reference_mean,
                }
            )
    evidence_boundary: dict[str, Any] = {
        "same_model_and_gpu_shape": True,
        "cross_gpu_or_cross_model_generalization_proven": False,
        "transport_remediation_version": TRANSPORT_REMEDIATION_VERSION,
        "all_cells_rebaselined_after_failed_r2": True,
        "zero_http_retries_required": True,
        "descriptive_only_under_fixed_3s_jina_pacing": True,
        "failed_r2_cells_excluded_without_pooling": True,
    }
    if [spec.label for spec in specs] == [
        "a-c12k-l80",
        "e-c12k-l80-u093",
    ]:
        evidence_boundary.update(
            {
                "shape_harness_repair_version": SHAPE_HARNESS_REPAIR_VERSION,
                "failed_shape_r1_resumed": False,
                "failed_shape_r1_observed_prefix_pooled": False,
                "high_pair_one_shot_replacement": True,
                "no_further_auto_rerun": True,
            }
        )
    return {
        "schema": "paste_repro.scheduler_live_sensitivity_summary",
        "version": 1,
        "development_only": True,
        "formal_eligible": False,
        "single_run_per_cell": True,
        "confidence_interval_available": False,
        "cells": cells,
        "a_to_e_effects": effects,
        "physical_kv_target_sensitivity": target_sensitivity,
        "evidence_boundary": evidence_boundary,
        "run_root": formal.repository_relative(run_root),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument(
        "--suite",
        choices=REGISTERED_SUITES,
        default="target",
        help=(
            "bounded matrix: target has four cells; high has the two-cell "
            "c12k/l80 shape-r1 harness replacement"
        ),
    )
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the exact plan without touching GPU/server/network.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_tag) is None:
        raise LiveSensitivityError("RUN_TAG contains unsupported characters")
    if args.run_tag == "comment3-shape-r1":
        raise LiveSensitivityError(
            "comment3-shape-r1 is immutable failed evidence and cannot be resumed"
        )
    specs = cells_for_suite(args.suite)
    base, python, model_snapshot, preflight = _preflight(
        specs, gpus=args.gpus, port=args.port
    )
    bindings = _bindings(
        FAILED_SHAPE_BOUND_PATHS if args.suite == "high" else ()
    )
    plan = _plan(
        run_tag=args.run_tag,
        suite=args.suite,
        specs=specs,
        base=base,
        python=python,
        gpus=args.gpus,
        port=args.port,
        preflight=preflight,
        bindings=bindings,
    )
    if args.check_only:
        plan["check_only"] = True
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    run_root = RUN_BASE / args.run_tag
    lock_path = RUN_BASE / f".{args.run_tag}.lock"
    if run_root.exists() or lock_path.exists():
        raise LiveSensitivityError(f"run tag already exists: {args.run_tag}")
    RUN_BASE.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise LiveSensitivityError("another process reserved this run tag") from exc
    try:
        run_root.mkdir()
        formal.write_json_atomic(run_root / "run_plan.json", plan)
        completed: dict[str, Path] = {}
        for index, spec in enumerate(specs):
            completed[spec.label] = _run_cell(
                run_root=run_root,
                spec=spec,
                order_index=index,
                base=base,
                python=python,
                model_snapshot=model_snapshot,
                gpus=args.gpus,
                port=args.port,
                bindings=bindings,
            )
        summary = _summarize(run_root, specs, completed)
        formal.write_json_atomic(run_root / "summary.json", summary)
        _verify_bindings(bindings)
        formal.write_json_atomic(
            run_root / "completed_matrix.json",
            {
                "schema": "paste_repro.scheduler_live_sensitivity_completion",
                "version": 1,
                "development_only": True,
                "formal_eligible": False,
                "completed_wall_s": time.time(),
                "summary": {
                    "path": formal.repository_relative(run_root / "summary.json"),
                    "sha256": _sha256(run_root / "summary.json"),
                },
                "completed_cells": [
                    {
                        "label": spec.label,
                        "path": formal.repository_relative(completed[spec.label]),
                    }
                    for spec in specs
                ],
                "bindings": bindings,
                "shape_harness_repair_version": (
                    SHAPE_HARNESS_REPAIR_VERSION
                    if args.suite == "high"
                    else None
                ),
            },
        )
        print(f"Scheduler live sensitivity completed: {run_root}", flush=True)
        return 0
    except BaseException as exc:
        if run_root.is_dir():
            formal.write_json_atomic(
                run_root / "failure.json",
                {
                    "schema": "paste_repro.scheduler_live_sensitivity_failure",
                    "version": 1,
                    "failed_wall_s": time.time(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "shape_harness_repair_version": (
                        SHAPE_HARNESS_REPAIR_VERSION
                        if args.suite == "high"
                        else None
                    ),
                },
            )
        raise
    finally:
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LiveSensitivityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
