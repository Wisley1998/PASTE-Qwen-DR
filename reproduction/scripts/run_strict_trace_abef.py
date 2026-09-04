#!/usr/bin/env python3
"""Prepare and run the oracle-free Qwen DeepResearch A/B/E/F trace matrix.

``prepare`` fits every predictor on calibration40, selects URL Top-k on
tuning30 with a preregistered precision rule, and writes separate public and
sealed plans.  ``run-cell`` consumes one immutable plan against an already
started fresh vLLM server.  It never subtracts offline savings and never gives
the policy a recorded duration, future URL, hit label, or trace suffix.

This is a causal-reveal systems replay: the current recorded request is sent
to live vLLM and the recorded authoritative tool call is revealed only after
that live turn completes.  The live response does not choose the next call,
so results must not be described as an autonomous-agent quality evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import sys
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import aiohttp


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
ROOT_SCRIPTS = REPOSITORY_ROOT / "scripts"
for import_root in (REPRODUCTION_ROOT, ROOT_SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from paste_repro.analysis import evaluate_held_out  # noqa: E402
from paste_repro.mapper import load_artifact  # noqa: E402
from paste_repro.strict_trace_runtime import (  # noqa: E402
    CalibrationHashedServiceClock,
    CausalDurationPredictor,
    CausalSessionState,
    CausalTailPredictor,
    CausalTraceCursor,
    SealedTraceToolExecutor,
    StrictOnlinePolicy,
    canonical_sha256,
    corrected_tool_outcome,
    normalized_tool_arguments,
    SERVICE_CLOCK_SCHEMA,
    serialize_observation,
    signed_payload,
    validate_signed_payload,
    visit_urls,
)
from paste_repro.trace_coscheduler import AsyncPreemptibleVisitPool  # noqa: E402
from paste_repro.traces import (  # noqa: E402
    LLMCall,
    OtherEvent,
    SessionTrace,
    ToolCall,
    load_trace,
    transitions_from_sessions,
)
from trace_experiment_lib import (  # noqa: E402
    _build_chat_tokens,
)


BUNDLE_SCHEMA = "paste_repro.strict_trace_abef_bundle.v1"
PUBLIC_PLAN_SCHEMA = "paste_repro.strict_trace_public_plan.v2"
SEALED_PLAN_SCHEMA = "paste_repro.strict_trace_sealed_plan.v2"
HELDOUT_DIAGNOSTICS_SCHEMA = "paste_repro.strict_trace_heldout_diagnostics.v1"
RESULT_SCHEMA = "paste_repro.strict_trace_abef_result.v1"
INVOCATION_PROVENANCE_SCHEMA = "paste_repro.strict_invocation_predictor_provenance.v1"
MODEL_SNAPSHOT_INVENTORY_SCHEMA = "paste_repro.model_snapshot_inventory.v1"
RUNTIME_PARAMETERS_SCHEMA = "paste.paper.treatment_neutral_runtime.v1"
SCHEDULER_RUNTIME_EVIDENCE_SCHEMA = "paste.paper.scheduler_runtime_evidence.v1"
SERVICE_CLOCK_MODE = "calibration_hashed_empirical_v1"
SERVICE_CLOCK_MINIMUM_SELECTION_POOL_SIZE = 3
SERVICE_CLOCK_CANONICALIZATION = (
    "visit URL: trim, lowercase scheme/authority, remove default port/fragment, "
    "ensure root slash; then canonical-json({tool,arguments}); utf-8; sha256"
)
SERVICE_CLOCK_SELECTION_RULE = (
    "digest-prefix modulo sorted tool-matched calibration samples; "
    "calibration-global fallback"
)
SERVICE_CLOCK_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "physical_service_clock_mode",
        "training_role",
        "training_provenance",
        "uses_evaluation_labels",
        "enumerates_evaluation_invocations",
        "future_state_accepted_invariant",
        "minimum_selection_pool_size",
        "seed_sha256",
        "canonicalization",
        "selection_rule",
        "samples_by_tool_s",
        "artifact_sha256",
    }
)
RUNTIME_PARAMETER_KEYS = frozenset(
    {
        "model_id",
        "model_revision",
        "server_host",
        "server_port",
        "tensor_parallel_size",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "cuda_graph_sizes",
        "prefix_caching",
        "vllm_v1",
        "max_active_tasks",
        "tool_capacity",
        "configured_speculation_capacity",
        "request_timeout_s",
        "public_output_cap",
        "workload_instances",
        "arrival_schedule_sha256",
    }
)
FORMAL_ENVIRONMENT_KEYS = frozenset(
    {
        "PASTE_ENV_PREFIX", "HF_HOME", "MODEL_ID", "MODEL_REVISION",
        "PASTE_RUNTIME_HOME", "PASTE_RUNTIME_PATH",
        "PASTE_RUNTIME_LD_LIBRARY_PATH", "PASTE_RUNTIME_TMPDIR",
        "PASTE_RUNTIME_LANG", "PASTE_RUNTIME_TZ", "PYTHONHASHSEED",
        "PYTHONNOUSERSITE", "PYTHONSAFEPATH",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_NO_USAGE_STATS",
        "CUDA_DEVICE_ORDER", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
        "NCCL_IB_PCI_RELAXED_ORDERING", "NCCL_NET_GDR_LEVEL",
        "NCCL_SOCKET_IFNAME", "NCCL_TOPO_FILE", "VLLM_HOST",
        "VLLM_PROBE_HOST", "VLLM_PORT", "VLLM_HOOK_DIR", "VLLM_TP_SIZE",
        "VLLM_DTYPE", "VLLM_MAX_MODEL_LEN", "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_MAX_NUM_BATCHED_TOKENS", "VLLM_MAX_NUM_SEQS",
        "VLLM_CUDA_GRAPH_SIZES", "VLLM_ENABLE_PREFIX_CACHING", "VLLM_USE_V1",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "VLLM_REQUIRE_NEW", "VLLM_READY_TIMEOUT",
        "VLLM_START_CLEANUP_TIMEOUT", "VLLM_SHUTDOWN_TIMEOUT",
        "PASTE_STRICT_SESSIONS", "PASTE_PUBLIC_OUTPUT_CAP",
        "PASTE_MAX_ACTIVE_TASKS", "PASTE_VISIT_CAPACITY",
        "PASTE_SPECULATIVE_CAP", "PASTE_DEFAULT_PREDICTED_OUTPUT_TOKENS",
        "PASTE_REQUEST_TIMEOUT_S", "PASTE_GPU_GROUPS", "PASTE_PROTECTED_PID",
        "VLLM_SCHED_PRED_OUT_ENABLE", "VLLM_SCHED_PRED_OUT_EMA_ALPHA",
        "VLLM_SCHED_DEFAULT_PRED_OUT", "VLLM_SCHED_AVG_CALL_SERVICE_S",
        "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2",
        "VLLM_SCHED_DECODE_TOKENS_PER_S_V2",
        "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S",
        "VLLM_SCHED_TIME_AGING_ALPHA",
        "VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS",
        "VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING",
        "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
        "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
        "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING",
        "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING",
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_RESPECT_JOINT_LIMITS",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
        "VLLM_SCHED_JOINT_V2_FINAL_LANE",
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE",
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S",
        "VLLM_SCHED_JOINT_V2_REMAINING_LLM_WEIGHT",
        "VLLM_SCHED_JOINT_V2_REALIZED_GAIN_WEIGHT",
        "VLLM_SCHED_JOINT_V2_FINAL_BONUS_S",
        "VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S",
        "VLLM_SCHED_JOINT_RETURN_WINDOW_S", "VLLM_SCHED_JOINT_RESERVE_KV_SCALE",
        "VLLM_SCHED_JOINT_RESERVE_SLOT_SCALE", "VLLM_SCHED_JOINT_V2_TAIL_BETA",
        "VLLM_SCHED_JOINT_V2_TOOL_BETA",
        "VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S",
        "VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT",
        "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA",
        "VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS",
        "VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S",
        "VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S",
        "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY",
        "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY",
        "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S",
        "VLLM_SCHED_HBM_MIN_RUNNING_REQS", "VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP",
        "VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS",
        "VLLM_SCHED_HBM_MAX_LONG_RUNNING",
        "VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS",
        "VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS",
        "VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS",
        "VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO",
    }
)
CALL_GRAPH_MODE = "trace_replay_causal_reveal"
PUBLIC_PLAN_FORBIDDEN_FIELDS = frozenset(
    {
        "requests",
        "steps",
        "messages",
        "tools_after",
        "tool_name",
        "tool_args",
        "outcome_id",
        "authority_key",
        "runtime_key",
        "state_accepted",
    }
)
PUBLIC_TRACE_FIELDS = frozenset(
    {
        "trace_id",
        "session_id",
        "source_session_id",
        "source_root_index",
        "release_offset_s",
        "arrival",
    }
)
DEFAULT_FIXED_BUNDLE = (
    REPRODUCTION_ROOT
    / "artifacts/fixed_trace_splits/"
    "30a0cb7c58b3-1ff2b2e2feb5-c40-t30-f30/bundle.json"
)
DEFAULT_EXECUTION_TRACES = (
    REPOSITORY_ROOT
    / "traces/my_traces_tool_slo_search_uniform_1_3s_"
    "visit_serial_uniform_2_8s_llm_x0_42"
)
DEFAULT_ARRIVALS = (
    REPRODUCTION_ROOT
    / "artifacts/azure_live_full_comparison/plans/azure_llm_3s_raw_80.json"
)
DEFAULT_FORMAL_CONFIG = REPRODUCTION_ROOT / "configs/strict_trace_abef.env.example"
DEFAULT_SCHEDULER_HOOK = REPOSITORY_ROOT / "scripts/pythonhooks/sched_policy_patch.py"
STRICT_RUNTIME_PATH = REPRODUCTION_ROOT / "paste_repro/strict_trace_runtime.py"
TOOL_POOL_PATH = REPRODUCTION_ROOT / "paste_repro/trace_coscheduler.py"
MAPPER_CODE_PATH = REPRODUCTION_ROOT / "paste_repro/mapper.py"
MATRIX_WRAPPER_PATH = REPRODUCTION_ROOT / "scripts/run_strict_trace_abef_matrix.sh"
SMOKE_SCRIPT_PATH = REPRODUCTION_ROOT / "scripts/smoke_vllm.py"
START_VLLM_PATH = REPRODUCTION_ROOT / "scripts/start_vllm.sh"
STOP_VLLM_PATH = REPRODUCTION_ROOT / "scripts/stop_vllm.sh"
SITECUSTOMIZE_PATH = ROOT_SCRIPTS / "pythonhooks/sitecustomize.py"
CELL_SPECS = {
    "A": {"scheduler": "native_fcfs", "server_policy": "fcfs", "speculation": False},
    "B": {"scheduler": "native_fcfs", "server_policy": "fcfs", "speculation": True},
    "E": {
        "scheduler": "causal_joint",
        "server_policy": "online_joint_pacer_v2",
        "speculation": False,
    },
    "F": {
        "scheduler": "causal_joint",
        "server_policy": "online_joint_pacer_v2",
        "speculation": True,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_snapshot_inventory(snapshot: Path) -> dict[str, Any]:
    """Hash every regular file reachable from one pinned HF snapshot.

    Hugging Face snapshots commonly contain symlinks into a blob store.  The
    inventory deliberately hashes resolved content rather than trusting either
    the revision directory name or the symlink/blob filename.
    """

    snapshot = snapshot.resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError(f"model snapshot is not a directory: {snapshot}")
    files: list[dict[str, Any]] = []
    for entry in sorted(snapshot.rglob("*"), key=lambda value: value.as_posix()):
        relative = entry.relative_to(snapshot).as_posix()
        if entry.is_symlink() and entry.resolve(strict=True).is_dir():
            raise ValueError(
                f"model snapshot contains an unsupported directory symlink: {relative}"
            )
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError(f"model snapshot contains a non-regular entry: {relative}")
        resolved = entry.resolve(strict=True)
        before = resolved.stat()
        digest = file_sha256(resolved)
        after = resolved.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or entry.resolve(strict=True) != resolved:
            raise RuntimeError(
                f"model snapshot entry changed while it was inventoried: {relative}"
            )
        files.append(
            {
                "relative_path": relative,
                "size_bytes": before.st_size,
                "content_sha256": digest,
            }
        )
    if not files or "config.json" not in {
        row["relative_path"] for row in files
    }:
        raise ValueError("model snapshot inventory is empty or lacks config.json")
    identity = {
        "schema": MODEL_SNAPSHOT_INVENTORY_SCHEMA,
        "files": files,
    }
    return {
        **identity,
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "inventory_sha256": canonical_sha256(identity),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _future_authority_fields(payload: Any) -> set[str]:
    """Return forbidden keys recursively embedded in a policy-facing artifact."""

    leaked: set[str] = set()
    if isinstance(payload, Mapping):
        leaked.update(PUBLIC_PLAN_FORBIDDEN_FIELDS & set(payload))
        for value in payload.values():
            leaked.update(_future_authority_fields(value))
    elif isinstance(payload, list):
        for value in payload:
            leaked.update(_future_authority_fields(value))
    return leaked


def _assert_policy_facing_document_safe(payload: Any, *, label: str) -> None:
    leaked = _future_authority_fields(payload)
    if leaked:
        raise ValueError(f"{label} exposes future-authority fields: {sorted(leaked)}")


def _write_role_plan_files(
    *,
    output_dir: Path,
    role: str,
    public: Mapping[str, Any],
    sealed: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    """Write the public plan read-only and private documents owner-read-only."""

    public_path = output_dir / f"{role}.public.json"
    sealed_path = output_dir / f"{role}.sealed.json"
    diagnostics_path = output_dir / f"{role}.heldout_diagnostics.json"
    write_json(public_path, public)
    write_json(sealed_path, sealed)
    write_json(diagnostics_path, diagnostics)
    public_path.chmod(0o444)
    sealed_path.chmod(0o400)
    diagnostics_path.chmod(0o400)
    return public_path, sealed_path, diagnostics_path


def _formal_config_exports(path: Path) -> dict[str, str]:
    """Parse literal exported values from the frozen, non-executable contract."""

    exports: dict[str, str] = {}
    pattern = re.compile(
        r"^\s*export\s+([A-Z][A-Z0-9_]*)=(?:\"([^\"]*)\"|'([^']*)'|([^\s#]+))\s*$"
    )
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(
                f"formal config must use literal export assignments: {path}:{line_number}"
            )
        if match.group(4) is not None:
            raise ValueError(
                f"formal config values must be quoted literals: {path}:{line_number}"
            )
        if match.group(2) is not None and any(
            marker in match.group(2) for marker in ("$", "`", "\\")
        ):
            raise ValueError(
                f"formal config double-quoted value is not literal: {path}:{line_number}"
            )
        key = match.group(1)
        if key in exports:
            raise ValueError(f"formal config exports {key} more than once")
        exports[key] = next(
            value for value in match.groups()[1:] if value is not None
        )
    return exports


def _validate_formal_environment_contract(
    exports: Mapping[str, str],
) -> Path:
    """Validate the clean-room host/model environment frozen by the config."""

    unknown = sorted(set(exports) - FORMAL_ENVIRONMENT_KEYS)
    if unknown:
        raise ValueError(
            "formal config exports unregistered environment variables: "
            + ", ".join(unknown)
        )

    forbidden = {
        "MODEL_SNAPSHOT",
        "PYTHONPATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_VISIBLE_DEVICES",
        "VLLM_SCHED_POLICY",
        "VLLM_STATE_DIR",
        "VLLM_LOG_DIR",
        "VLLM_API_KEY",
    }
    present_forbidden = sorted(forbidden.intersection(exports))
    if present_forbidden:
        raise ValueError(
            "formal config exports runtime-derived or unregistered variables: "
            + ", ".join(present_forbidden)
        )

    exact = {
        "PASTE_RUNTIME_LANG": "C.UTF-8",
        "PASTE_RUNTIME_TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    for name, expected in exact.items():
        if exports.get(name) != expected:
            raise ValueError(
                f"formal config {name} must be exactly {expected!r}"
            )

    required_paths = (
        "PASTE_ENV_PREFIX",
        "HF_HOME",
        "PASTE_RUNTIME_HOME",
        "PASTE_RUNTIME_TMPDIR",
        "VLLM_HOOK_DIR",
        "NCCL_TOPO_FILE",
    )
    for name in required_paths:
        raw = exports.get(name)
        if not raw or not Path(raw).is_absolute():
            raise ValueError(f"formal config {name} must be an absolute path")
        path = Path(raw)
        if not path.exists() or path.resolve() != path:
            raise ValueError(
                f"formal config {name} must be an existing canonical path: {path}"
            )

    runtime_path = exports.get("PASTE_RUNTIME_PATH", "")
    path_entries = runtime_path.split(":")
    if (
        not runtime_path
        or any(not entry or not Path(entry).is_absolute() for entry in path_entries)
        or path_entries[0] != str(Path(exports["PASTE_ENV_PREFIX"]) / "bin")
    ):
        raise ValueError(
            "formal config PASTE_RUNTIME_PATH must be absolute and begin with PASTE_ENV_PREFIX/bin"
        )
    library_path = exports.get("PASTE_RUNTIME_LD_LIBRARY_PATH", "")
    if not library_path or any(
        not entry or not Path(entry).is_absolute()
        for entry in library_path.split(":")
    ):
        raise ValueError(
            "formal config PASTE_RUNTIME_LD_LIBRARY_PATH must contain only absolute paths"
        )

    env_python = Path(exports["PASTE_ENV_PREFIX"]) / "bin/python"
    if not env_python.is_file() or not os.access(env_python, os.X_OK):
        raise ValueError(f"formal environment Python is not executable: {env_python}")
    model_id = exports.get("MODEL_ID", "")
    revision = exports.get("MODEL_REVISION", "")
    if (
        not model_id
        or model_id.startswith("/")
        or ".." in model_id.split("/")
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", model_id)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", revision)
    ):
        raise ValueError("formal model ID/revision cannot form a safe snapshot path")
    cache_key = f"models--{model_id.replace('/', '--')}"
    snapshot = (
        Path(exports["HF_HOME"]) / cache_key / "snapshots" / revision
    )
    if not snapshot.is_dir() or snapshot.resolve() != snapshot:
        raise ValueError(
            "pinned model snapshot must exist at the exact canonical "
            "HF_HOME/MODEL_ID/MODEL_REVISION path"
        )
    if not (snapshot / "config.json").is_file():
        raise ValueError(f"pinned model snapshot has no config.json: {snapshot}")
    return snapshot


def _build_runtime_parameters(
    *,
    args: argparse.Namespace,
    arrival_rows: Sequence[Mapping[str, Any]],
    arrival_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the exact treatment-neutral settings before evaluation opens."""

    exports = _formal_config_exports(args.formal_config.resolve())
    _validate_formal_environment_contract(exports)

    def required(name: str) -> str:
        value = exports.get(name)
        if value is None or value == "":
            raise ValueError(f"formal config lacks required export {name}")
        return value

    def integer(name: str, *, minimum: int = 1) -> int:
        raw = required(name)
        if not re.fullmatch(r"[0-9]+", raw) or int(raw) < minimum:
            raise ValueError(f"formal config {name} must be an integer >= {minimum}")
        return int(raw)

    def number(name: str, *, positive: bool = True) -> float:
        try:
            value = float(required(name))
        except ValueError as exc:
            raise ValueError(f"formal config {name} must be numeric") from exc
        if not math.isfinite(value) or (positive and value <= 0.0):
            raise ValueError(f"formal config {name} must be finite and positive")
        return value

    def boolean(name: str) -> bool:
        raw = required(name)
        if raw not in {"0", "1"}:
            raise ValueError(f"formal config {name} must be 0 or 1")
        return raw == "1"

    cuda_graph_sizes = [
        int(value) for value in required("VLLM_CUDA_GRAPH_SIZES").split(",")
    ]
    if not cuda_graph_sizes or min(cuda_graph_sizes) <= 0:
        raise ValueError("formal config VLLM_CUDA_GRAPH_SIZES must be positive")
    configured = {
        "model": required("MODEL_ID"),
        "model_revision": required("MODEL_REVISION"),
        "max_model_len": integer("VLLM_MAX_MODEL_LEN"),
        "max_active_tasks": integer("PASTE_MAX_ACTIVE_TASKS"),
        "visit_capacity": integer("PASTE_VISIT_CAPACITY"),
        "speculative_cap": integer("PASTE_SPECULATIVE_CAP", minimum=0),
        "request_timeout_s": number("PASTE_REQUEST_TIMEOUT_S"),
        "default_predicted_output_tokens": number(
            "PASTE_DEFAULT_PREDICTED_OUTPUT_TOKENS"
        ),
        "output_cap": integer("PASTE_PUBLIC_OUTPUT_CAP"),
        "sessions": integer("PASTE_STRICT_SESSIONS"),
    }
    expected = {
        "model": args.model,
        "model_revision": args.model_revision,
        "max_model_len": args.max_model_len,
        "max_active_tasks": args.max_active_tasks,
        "visit_capacity": args.visit_capacity,
        "speculative_cap": args.speculative_cap,
        "request_timeout_s": float(args.request_timeout_s),
        "default_predicted_output_tokens": float(
            args.default_predicted_output_tokens
        ),
        "output_cap": args.output_cap,
        "sessions": args.sessions,
    }
    if configured != expected:
        raise ValueError(
            f"prepare arguments differ from frozen formal config: {configured!r} != {expected!r}"
        )
    if configured["sessions"] != len(arrival_rows):
        raise ValueError("frozen workload instance count differs from arrival rows")
    if configured["speculative_cap"] > configured["visit_capacity"]:
        raise ValueError("frozen speculation capacity exceeds tool capacity")
    arrival_sha256 = arrival_provenance.get("source_sha256")
    if not isinstance(arrival_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", arrival_sha256
    ):
        arrival_sha256 = canonical_sha256(
            {"arrival_rows": [dict(row) for row in arrival_rows]}
        )
    parameters = {
        "model_id": configured["model"],
        "model_revision": configured["model_revision"],
        "server_host": required("VLLM_HOST"),
        "server_port": integer("VLLM_PORT"),
        "tensor_parallel_size": integer("VLLM_TP_SIZE"),
        "dtype": required("VLLM_DTYPE"),
        "max_model_len": configured["max_model_len"],
        "gpu_memory_utilization": number("VLLM_GPU_MEMORY_UTILIZATION"),
        "max_num_batched_tokens": integer("VLLM_MAX_NUM_BATCHED_TOKENS"),
        "max_num_seqs": integer("VLLM_MAX_NUM_SEQS"),
        "cuda_graph_sizes": cuda_graph_sizes,
        "prefix_caching": boolean("VLLM_ENABLE_PREFIX_CACHING"),
        "vllm_v1": boolean("VLLM_USE_V1"),
        "max_active_tasks": configured["max_active_tasks"],
        "tool_capacity": configured["visit_capacity"],
        "configured_speculation_capacity": configured["speculative_cap"],
        "request_timeout_s": configured["request_timeout_s"],
        "public_output_cap": configured["output_cap"],
        "workload_instances": configured["sessions"],
        "arrival_schedule_sha256": arrival_sha256,
    }
    result = signed_payload(
        {"schema": RUNTIME_PARAMETERS_SCHEMA, "parameters": parameters},
        "runtime_parameters_sha256",
    )
    return _validate_runtime_parameters(result)


def _validate_runtime_parameters(payload: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_signed_payload(
        payload,
        "runtime_parameters_sha256",
        label="treatment-neutral runtime parameters",
    )
    if checked.get("schema") != RUNTIME_PARAMETERS_SCHEMA:
        raise ValueError("unsupported treatment-neutral runtime schema")
    parameters = checked.get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != RUNTIME_PARAMETER_KEYS:
        raise ValueError("treatment-neutral runtime parameter keys are invalid")
    for field in (
        "server_port",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_active_tasks",
        "tool_capacity",
        "public_output_cap",
        "workload_instances",
    ):
        if type(parameters[field]) is not int or int(parameters[field]) <= 0:
            raise ValueError(f"runtime parameter {field} must be a positive integer")
    if (
        type(parameters["configured_speculation_capacity"]) is not int
        or int(parameters["configured_speculation_capacity"]) < 0
        or int(parameters["configured_speculation_capacity"])
        > int(parameters["tool_capacity"])
    ):
        raise ValueError("configured speculation capacity is invalid")
    for field in ("gpu_memory_utilization", "request_timeout_s"):
        value = parameters[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"runtime parameter {field} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"runtime parameter {field} must be finite and positive")
    for field in ("model_id", "model_revision", "server_host", "dtype"):
        if not isinstance(parameters[field], str) or not parameters[field]:
            raise ValueError(f"runtime parameter {field} must be a non-empty string")
    if not re.fullmatch(r"[0-9a-f]{64}", str(parameters["arrival_schedule_sha256"])):
        raise ValueError("runtime arrival schedule SHA-256 is invalid")
    if type(parameters["prefix_caching"]) is not bool or type(
        parameters["vllm_v1"]
    ) is not bool:
        raise ValueError("runtime boolean parameters are invalid")
    graph_sizes = parameters["cuda_graph_sizes"]
    if (
        not isinstance(graph_sizes, list)
        or not graph_sizes
        or any(type(value) is not int or value <= 0 for value in graph_sizes)
    ):
        raise ValueError("runtime CUDA graph sizes are invalid")
    return checked


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _task_timing_evidence(
    *,
    experiment_started_monotonic_s: float,
    release_offset_s: float,
    released_at_monotonic_s: float,
    gate_acquired_at_monotonic_s: float,
    task_terminal_monotonic_s: float,
) -> dict[str, float]:
    """Build fail-closed raw timing evidence for one end-to-end task."""

    values = (
        experiment_started_monotonic_s,
        release_offset_s,
        released_at_monotonic_s,
        gate_acquired_at_monotonic_s,
        task_terminal_monotonic_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("task timing evidence contains a non-finite value")
    if release_offset_s < 0.0:
        raise RuntimeError("task release offset must be non-negative")
    scheduled = experiment_started_monotonic_s + release_offset_s
    if not (
        experiment_started_monotonic_s
        <= scheduled
        <= released_at_monotonic_s
        <= gate_acquired_at_monotonic_s
        <= task_terminal_monotonic_s
    ):
        raise RuntimeError("task monotonic timing order is invalid")
    return {
        "release_offset_s": release_offset_s,
        "scheduled_release_monotonic_s": scheduled,
        "released_at_monotonic_s": released_at_monotonic_s,
        "task_terminal_monotonic_s": task_terminal_monotonic_s,
        "scheduled_release_offset_s": scheduled
        - experiment_started_monotonic_s,
        "released_at_offset_s": released_at_monotonic_s
        - experiment_started_monotonic_s,
        "task_terminal_offset_s": task_terminal_monotonic_s
        - experiment_started_monotonic_s,
        "release_lag_s": released_at_monotonic_s - scheduled,
        "task_gate_wait_s": gate_acquired_at_monotonic_s
        - released_at_monotonic_s,
        "flow_s": task_terminal_monotonic_s - scheduled,
    }


def _relative(path: Path, anchor: Path) -> str:
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(anchor: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else anchor / path


def _bound_file(path: Path, anchor: Path) -> dict[str, str]:
    return {"path": _relative(path, anchor), "sha256": file_sha256(path)}


def _verify_bound_file(entry: Mapping[str, Any], anchor: Path, label: str) -> Path:
    raw = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(raw, str) or not isinstance(expected, str):
        raise ValueError(f"{label} file binding is invalid")
    path = _resolve(anchor, raw)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file binding mismatch: {path}")
    return path


def _logical_trace_hash(session: SessionTrace) -> str:
    """Hash call graph/content while excluding every timing field."""

    rows: list[dict[str, Any]] = []
    for event in session.events:
        if isinstance(event, LLMCall):
            rows.append(
                {
                    "type": "llm",
                    "call_index": event.call_index,
                    "messages": list(event.messages),
                    "response": event.response,
                }
            )
        elif isinstance(event, ToolCall):
            rows.append(
                {
                    "type": "tool",
                    "call_index": event.call_index,
                    "tool_name": event.tool_name,
                    "tool_args": event.tool_args,
                }
            )
        elif isinstance(event, OtherEvent):
            # Corrected corpora append clock-accounting markers which are not
            # part of the agent-visible call graph.
            if event.event_type == "synthetic_tool_completion":
                continue
            payload = dict(event.payload)
            for key in tuple(payload):
                if "time" in key.lower() or key == "timestamp":
                    payload.pop(key, None)
            rows.append({"type": event.event_type, "payload": payload})
    return canonical_sha256(rows)


def _load_fixed_split(fixed_bundle_path: Path) -> dict[str, Any]:
    fixed_bundle = validate_signed_payload(
        read_json(fixed_bundle_path), "bundle_sha256", label="fixed split bundle"
    )
    fixed_anchor = fixed_bundle_path.parent
    split_path = _resolve(fixed_anchor, str(fixed_bundle["split_manifest"]))
    split = validate_signed_payload(
        read_json(split_path), "manifest_sha256", label="fixed split manifest"
    )
    if split["manifest_sha256"] != fixed_bundle["split_manifest_sha256"]:
        raise ValueError("fixed bundle/manifest checksum mismatch")
    expected_counts = {"calibration": 40, "tuning": 30, "final": 30, "total": 100}
    if split.get("counts") != expected_counts:
        raise ValueError(f"strict experiment requires fixed 40/30/30 split: {split.get('counts')}")
    roles: dict[str, list[dict[str, Any]]] = {}
    for role in ("calibration", "tuning", "final"):
        rows = split.get(f"{role}_sessions")
        if not isinstance(rows, list) or len(rows) != expected_counts[role]:
            raise ValueError(f"fixed {role} role is invalid")
        roles[role] = [dict(row) for row in rows]
    ids = {role: {str(row["session_id"]) for row in rows} for role, rows in roles.items()}
    if ids["calibration"] & ids["tuning"] or ids["calibration"] & ids["final"] or ids["tuning"] & ids["final"]:
        raise ValueError("fixed split roles overlap")
    mapper_path = _resolve(fixed_anchor, str(fixed_bundle["mapper_artifact"]))
    mapper, mapper_artifact = load_artifact(mapper_path)
    if mapper_artifact.get("artifact_sha256") != fixed_bundle["mapper_artifact_sha256"]:
        raise ValueError("fixed mapper artifact checksum mismatch")
    mapper_train = {
        str(row["session_id"])
        for row in mapper_artifact.get("training_split", {}).get("train_sessions", [])
    }
    if mapper_train != ids["calibration"]:
        raise ValueError("URL mapper training sessions are not exactly calibration40")
    return {
        "bundle": fixed_bundle,
        "manifest": split,
        "roles": roles,
        "ids": ids,
        "mapper": mapper,
        "mapper_artifact": mapper_artifact,
        "mapper_path": mapper_path,
        "fixed_anchor": fixed_anchor,
    }


def _load_role_sessions(
    fixed: Mapping[str, Any], execution_dir: Path, role: str
) -> tuple[SessionTrace, ...]:
    if not execution_dir.is_dir():
        raise FileNotFoundError(f"corrected execution trace directory is missing: {execution_dir}")
    sessions: list[SessionTrace] = []
    raw_role_dir = fixed["fixed_anchor"] / str(fixed["bundle"]["roles"][role]["directory"])
    for entry in fixed["roles"][role]:
        session_id = str(entry["session_id"])
        raw_path = raw_role_dir / session_id
        execution_path = execution_dir / session_id
        if not raw_path.is_file() or file_sha256(raw_path) != str(entry["sha256"]):
            raise ValueError(f"raw source lineage mismatch: {session_id}")
        if not execution_path.is_file():
            raise FileNotFoundError(f"corrected execution trace is missing: {execution_path}")
        raw_session = load_trace(raw_path)
        execution_session = load_trace(execution_path)
        if _logical_trace_hash(raw_session) != _logical_trace_hash(execution_session):
            raise ValueError(f"raw/corrected logical trace mismatch: {session_id}")
        sessions.append(execution_session)
    return tuple(sessions)


def _training_provenance(
    fixed: Mapping[str, Any], sessions: Sequence[SessionTrace]
) -> dict[str, Any]:
    raw_by_id = {
        str(row["session_id"]): str(row["sha256"])
        for row in fixed["roles"]["calibration"]
    }
    return {
        "fixed_split_manifest_sha256": fixed["manifest"]["manifest_sha256"],
        "session_ids": sorted(raw_by_id),
        "session_ids_sha256": canonical_sha256(sorted(raw_by_id)),
        "raw_source_sha256": dict(sorted(raw_by_id.items())),
        "execution_trace_sha256": {
            session.session_id: file_sha256(session.path)
            for session in sorted(sessions, key=lambda row: row.session_id)
        },
    }


def _calibration_service_samples(
    sessions: Sequence[SessionTrace],
) -> dict[str, tuple[float, ...]]:
    by_tool: dict[str, list[float]] = {}
    all_values: list[float] = []
    for session in sessions:
        for event in session.events:
            if not isinstance(event, ToolCall):
                continue
            outcome = corrected_tool_outcome(event)
            if event.tool_name == "visit":
                values = [float(row["duration_s"]) for row in outcome["visit_units"]]
            elif outcome["duration_s"] is not None:
                values = [float(outcome["duration_s"])]
            else:
                values = []
            by_tool.setdefault(event.tool_name, []).extend(values)
            all_values.extend(values)
    if not all_values or not by_tool.get("visit"):
        raise ValueError("calibration40 contains no usable tool service samples")
    return {
        **{key: tuple(sorted(values)) for key, values in sorted(by_tool.items()) if values},
        "__global__": tuple(sorted(all_values)),
    }


def _service_clock_sample_payload(
    samples_by_tool_s: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    return {
        str(key): [float(value) for value in values]
        for key, values in sorted(samples_by_tool_s.items())
    }


def _new_service_clock_artifact(
    *,
    training_provenance: Mapping[str, Any],
    samples_by_tool_s: Mapping[str, Sequence[float]],
    seed_sha256: str,
) -> dict[str, Any]:
    return signed_payload(
        {
            "schema": SERVICE_CLOCK_SCHEMA,
            "physical_service_clock_mode": SERVICE_CLOCK_MODE,
            "training_role": "calibration",
            "training_provenance": dict(training_provenance),
            "uses_evaluation_labels": False,
            "enumerates_evaluation_invocations": False,
            "future_state_accepted_invariant": True,
            "minimum_selection_pool_size": (
                SERVICE_CLOCK_MINIMUM_SELECTION_POOL_SIZE
            ),
            "seed_sha256": seed_sha256,
            "canonicalization": SERVICE_CLOCK_CANONICALIZATION,
            "selection_rule": SERVICE_CLOCK_SELECTION_RULE,
            "samples_by_tool_s": _service_clock_sample_payload(samples_by_tool_s),
        },
        "artifact_sha256",
    )


def _validate_service_clock_for_current_calibration(
    artifact: Any,
    *,
    training_provenance: Mapping[str, Any],
    samples_by_tool_s: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Bind a private clock artifact to this exact frozen calibration split."""

    if not isinstance(artifact, Mapping):
        raise ValueError("reused service clock artifact must be an object")
    checked = validate_signed_payload(
        artifact,
        "artifact_sha256",
        label="reused calibration service clock artifact",
    )
    if set(checked) != SERVICE_CLOCK_ARTIFACT_FIELDS:
        raise ValueError("reused service clock artifact fields do not match the contract")
    expected_contract = {
        "schema": SERVICE_CLOCK_SCHEMA,
        "physical_service_clock_mode": SERVICE_CLOCK_MODE,
        "training_role": "calibration",
        "uses_evaluation_labels": False,
        "enumerates_evaluation_invocations": False,
        "future_state_accepted_invariant": True,
        "minimum_selection_pool_size": SERVICE_CLOCK_MINIMUM_SELECTION_POOL_SIZE,
        "canonicalization": SERVICE_CLOCK_CANONICALIZATION,
        "selection_rule": SERVICE_CLOCK_SELECTION_RULE,
    }
    for field, expected in expected_contract.items():
        actual = checked.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"reused service clock {field} does not match the current contract"
            )
    reused_provenance = checked.get("training_provenance")
    if not isinstance(reused_provenance, Mapping) or canonical_sha256(
        reused_provenance
    ) != canonical_sha256(dict(training_provenance)):
        raise ValueError(
            "reused service clock calibration provenance does not match the "
            "current fixed split"
        )
    reused_samples = checked.get("samples_by_tool_s")
    expected_samples = _service_clock_sample_payload(samples_by_tool_s)
    if not isinstance(reused_samples, Mapping) or canonical_sha256(
        reused_samples
    ) != canonical_sha256(expected_samples):
        raise ValueError(
            "reused service clock sample pools do not match current calibration"
        )
    # Retain the complete signed artifact, including its private one-time salt,
    # only after the runtime implementation independently accepts it.
    CalibrationHashedServiceClock(checked)
    return checked


def _prepare_service_clock_artifact(
    *,
    reuse_path: Path | None,
    training_provenance: Mapping[str, Any],
    samples_by_tool_s: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    if reuse_path is None:
        candidate = _new_service_clock_artifact(
            training_provenance=training_provenance,
            samples_by_tool_s=samples_by_tool_s,
            seed_sha256=secrets.token_hex(32),
        )
    else:
        resolved = reuse_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"reused service clock is missing: {resolved}")
        candidate = read_json(resolved)
    return _validate_service_clock_for_current_calibration(
        candidate,
        training_provenance=training_provenance,
        samples_by_tool_s=samples_by_tool_s,
    )


def select_tuning_top_k(
    *,
    mapper: Any,
    tuning_sessions: Sequence[SessionTrace],
    max_top_k: int,
    min_precision: float,
) -> tuple[int, dict[str, Any]]:
    """Apply a frozen rule to tuning30 only; final30 is not an argument."""

    if max_top_k <= 0 or not 0.0 <= min_precision <= 1.0:
        raise ValueError("invalid tuning selection rule")
    transitions = transitions_from_sessions(tuning_sessions)
    widths = tuple(range(1, max_top_k + 1))
    evaluation = evaluate_held_out(
        mapper, transitions, top_ks=widths, latency_top_k=max_top_k
    )
    eligible = [
        width
        for width in widths
        if float(
            evaluation["top_k_concrete_invocation_hit"][str(width)][
                "prediction_precision"
            ]
        )
        >= min_precision
    ]
    if not eligible:
        raise ValueError("no tuning Top-k satisfies the preregistered precision gate")
    selected = max(eligible)
    evidence = {
        "selection_role": "tuning",
        "selection_rule": "largest k <= max_top_k with concrete prediction precision >= min_precision",
        "max_top_k": max_top_k,
        "min_precision": min_precision,
        "selected_top_k": selected,
        "tuning_session_ids": sorted(session.session_id for session in tuning_sessions),
        "tuning_session_ids_sha256": canonical_sha256(
            sorted(session.session_id for session in tuning_sessions)
        ),
        "metrics": evaluation,
    }
    evidence["selection_sha256"] = canonical_sha256(evidence)
    return selected, evidence


def _load_tokenizer(source: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(source), trust_remote_code=True, local_files_only=True
    )


def _heldout_duration_diagnostic(call: ToolCall) -> dict[str, Any]:
    """Sanitize recorded timing solely for a post-run diagnostic sidecar.

    Evaluation timing is adversarial input here: missing, malformed, non-finite,
    negative, or length-mismatched unit data must never make plan construction
    fail and must never define the number of physical invocations.  The latter
    is derived only from the authoritative tool arguments in ``build_role_plans``.
    """

    correction = call.timing_correction
    if not isinstance(correction, Mapping):
        correction = {}

    def optional_nonnegative(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result) or result < 0.0:
            return None
        return result

    raw_units = correction.get("unit_duration_s")
    unit_values = (
        [optional_nonnegative(value) for value in raw_units]
        if isinstance(raw_units, list)
        else []
    )
    authoritative_units = len(visit_urls(call)) if call.tool_name == "visit" else 1
    return {
        "recorded_total_service_diagnostic_s": optional_nonnegative(
            correction.get("duration_s")
        ),
        "recorded_unit_service_diagnostic_s": unit_values,
        "recorded_unit_field_was_list": isinstance(raw_units, list),
        "recorded_unit_count_matches_authority": (
            isinstance(raw_units, list) and len(raw_units) == authoritative_units
        ),
    }


def _prepare_request(
    event: LLMCall,
    *,
    tokenizer: Any,
    max_model_len: int,
    output_cap: int,
) -> dict[str, Any]:
    messages = [dict(message) for message in event.messages]
    original_prompt_tokens = _build_chat_tokens(tokenizer, messages)
    max_prompt_tokens = max_model_len - output_cap
    if original_prompt_tokens <= max_prompt_tokens:
        trimmed = messages
        prompt_tokens = original_prompt_tokens
        truncated = False
    else:
        # The legacy helper re-tokenizes the full, often multi-megabyte prompt
        # once per removed message.  Find the same oldest-prefix removal point
        # by binary search.  System messages are retained exactly.
        removable = sum(message.get("role") != "system" for message in messages)

        def remove_oldest(count: int) -> list[dict[str, Any]]:
            remaining = count
            result: list[dict[str, Any]] = []
            for message in messages:
                if remaining and message.get("role") != "system":
                    remaining -= 1
                    continue
                result.append(dict(message))
            return result

        smallest, largest = 1, removable
        while smallest < largest:
            middle = (smallest + largest) // 2
            candidate = remove_oldest(middle)
            if _build_chat_tokens(tokenizer, candidate) <= max_prompt_tokens:
                largest = middle
            else:
                smallest = middle + 1
        trimmed = remove_oldest(smallest)
        prompt_tokens = _build_chat_tokens(tokenizer, trimmed)
        if prompt_tokens > max_prompt_tokens:
            raise ValueError(
                "system-only prompt exceeds the public model budget; refusing "
                "content-dependent character truncation"
            )
        truncated = True
    # This budget is a public, bundle-wide constant.  The trace response and
    # its token count are never inspected.
    if prompt_tokens + output_cap > max_model_len:
        raise ValueError("prompt truncation failed to reserve the uniform output cap")
    max_tokens = output_cap
    return {
        "call_index": event.call_index,
        "messages": trimmed,
        "prompt_tokens": prompt_tokens,
        "original_prompt_tokens": original_prompt_tokens,
        "max_tokens": max_tokens,
        "truncated": bool(truncated),
    }


def _arrival_rows(path: Path | None, sessions: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        count = sessions or 0
        return (
            [{"release_offset_s": 0.0, "arrival_index": index} for index in range(count)],
            {"kind": "closed_burst", "sessions": count},
        )
    payload = read_json(path)
    rows = payload.get("arrivals") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("arrival plan has no arrivals")
    target = len(rows) if sessions is None else sessions
    if target <= 0 or target > len(rows):
        raise ValueError("--sessions must be within the frozen arrival plan")
    selected: list[dict[str, Any]] = []
    previous = -1.0
    for index, raw in enumerate(rows[:target]):
        release = float(raw["release_offset_s"])
        if not math.isfinite(release) or release < previous:
            raise ValueError("arrival releases must be finite, non-negative, and sorted")
        previous = release
        selected.append(
            {
                "release_offset_s": release,
                "arrival_index": index,
                "arrival_source_id": str(raw.get("source_id", index)),
            }
        )
    return selected, {
        "kind": "frozen_arrival_plan",
        "source": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "sessions": target,
        "release_span_s": previous,
    }


def build_role_plans(
    *,
    role: str,
    sessions: Sequence[SessionTrace],
    raw_sha_by_id: Mapping[str, str],
    tokenizer: Any,
    max_model_len: int,
    output_cap: int,
    arrivals: Sequence[Mapping[str, Any]],
    arrival_provenance: Mapping[str, Any],
    service_clock_artifact_sha256: str,
    claim_scope: str = "retrospective",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if role not in {"tuning", "final"}:
        raise ValueError("only tuning/final evaluation plans are allowed")
    if not sessions:
        raise ValueError(f"{role} sessions are empty")
    if claim_scope not in {"retrospective", "confirmatory"}:
        raise ValueError("claim_scope must be retrospective or confirmatory")
    release_rows = list(arrivals) or [
        {"release_offset_s": 0.0, "arrival_index": index}
        for index in range(len(sessions))
    ]
    public_traces: list[dict[str, Any]] = []
    sealed_trace_steps: dict[str, list[dict[str, Any]]] = {}
    sealed_trace_lineage: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    heldout_diagnostics: dict[str, dict[str, Any]] = {}
    # Arrival replication must not repeat expensive deterministic tokenization.
    # Cache by immutable source/event identity, then clone into each instance.
    prepared_request_cache: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for instance_index, arrival in enumerate(release_rows):
        source = sessions[instance_index % len(sessions)]
        opaque_source = hashlib.sha256(source.session_id.encode("utf-8")).hexdigest()[:12]
        instance_id = f"{role}-{instance_index:03d}-{opaque_source}"
        steps: list[dict[str, Any]] = []
        events = list(source.events)
        for event_index, event in enumerate(events):
            if not isinstance(event, LLMCall):
                continue
            tools_after: list[dict[str, Any]] = []
            cursor = event_index + 1
            while cursor < len(events) and not isinstance(events[cursor], LLMCall):
                tool = events[cursor]
                if isinstance(tool, ToolCall):
                    outcome_id = hashlib.sha256(
                        f"{instance_id}\0{cursor}\0{tool.call_index}".encode("utf-8")
                    ).hexdigest()
                    tools_after.append(
                        {
                            "outcome_id": outcome_id,
                            "event_index": cursor,
                            "call_index": tool.call_index,
                            "tool_name": tool.tool_name,
                            "tool_args": tool.tool_args,
                        }
                    )
                    # The physical invocation graph comes only from the
                    # authoritative tool name/arguments.  In particular, the
                    # held-out trace's recorded timing and unit-duration list
                    # cannot add, remove, or reject a Visit invocation.
                    visit_rows = (
                        [{"url": str(url)} for url in visit_urls(tool)]
                        if tool.tool_name == "visit"
                        else []
                    )
                    # Evaluation durations are diagnostics only for every tool.
                    # Physical service is assigned by the calibration-only
                    # invocation surface before any cell starts.
                    execution_outcome = {
                        "visit_units": visit_rows,
                        "source": "calibration_only_counterfactual_surface",
                    }
                    heldout_diagnostics[outcome_id] = _heldout_duration_diagnostic(tool)
                    outcomes[outcome_id] = {
                        "session_id": instance_id,
                        "source_session_id": source.session_id,
                        "event_index": cursor,
                        "call_index": tool.call_index,
                        "tool_name": tool.tool_name,
                        **execution_outcome,
                    }
                cursor += 1
            request_cache_key = (
                source.session_id,
                event_index,
                id(tokenizer),
                max_model_len,
                output_cap,
            )
            cached_request = prepared_request_cache.get(request_cache_key)
            if cached_request is None:
                cached_request = _prepare_request(
                    event,
                    tokenizer=tokenizer,
                    max_model_len=max_model_len,
                    output_cap=output_cap,
                )
                prepared_request_cache[request_cache_key] = cached_request
            prepared_request = json.loads(json.dumps(cached_request))
            steps.append({"request": prepared_request, "tools_after": tools_after})
        # The public plan is deliberately metadata-only.  In particular it
        # does not contain even the current request: the private causal cursor
        # releases that request when its task reaches the corresponding turn.
        public_arrival = {
            "release_offset_s": float(arrival["release_offset_s"]),
            "arrival_index": int(arrival["arrival_index"]),
        }
        if "arrival_source_id" in arrival:
            public_arrival["arrival_source_id"] = str(arrival["arrival_source_id"])
        public_traces.append(
            {
                "trace_id": instance_id,
                "session_id": instance_id,
                "source_session_id": source.session_id,
                "source_root_index": instance_index % len(sessions),
                "release_offset_s": float(arrival["release_offset_s"]),
                "arrival": public_arrival,
            }
        )
        sealed_trace_steps[instance_id] = steps
        sealed_trace_lineage[instance_id] = {
            "source_session_id": source.session_id,
            "source_root_index": instance_index % len(sessions),
            "raw_source_sha256": raw_sha_by_id[source.session_id],
            "execution_trace_sha256": file_sha256(source.path),
            "logical_trace_sha256": _logical_trace_hash(source),
        }
    public = signed_payload(
        {
            "schema": PUBLIC_PLAN_SCHEMA,
            "role": role,
            "claim_scope": claim_scope,
            "call_graph_mode": CALL_GRAPH_MODE,
            "output_budget_policy": "uniform_public_cap_no_trace_response_length",
            "max_model_len": max_model_len,
            "output_cap": output_cap,
            "arrival_process": dict(arrival_provenance),
            "independent_source_roots": len(sessions),
            "replicas": len(public_traces),
            "traces": public_traces,
        },
        "plan_sha256",
    )
    _assert_policy_facing_document_safe(public, label=f"{role} public plan")
    sealed = signed_payload(
        {
            "schema": SEALED_PLAN_SCHEMA,
            "role": role,
            "claim_scope": claim_scope,
            "public_plan_sha256": public["plan_sha256"],
            "access_contract": (
                "CausalTraceCursor releases one current request and its following "
                "authority only after live LLM completion; SealedTraceToolExecutor "
                "alone reads outcomes"
            ),
            "trace_steps": sealed_trace_steps,
            "trace_lineage": sealed_trace_lineage,
            "outcomes": outcomes,
            "service_clock_artifact_sha256": service_clock_artifact_sha256,
        },
        "sealed_sha256",
    )
    diagnostics = signed_payload(
        {
            "schema": HELDOUT_DIAGNOSTICS_SCHEMA,
            "role": role,
            "runtime_access": False,
            "public_plan_sha256": public["plan_sha256"],
            "sealed_plan_sha256": sealed["sealed_sha256"],
            "records": heldout_diagnostics,
        },
        "diagnostics_sha256",
    )
    return public, sealed, diagnostics


def prepare_bundle(args: argparse.Namespace) -> dict[str, Any]:
    requested_output_dir = args.output_dir.resolve()
    if requested_output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen preparation: {requested_output_dir}"
        )
    formal_config_path = args.formal_config.resolve()
    scheduler_hook_path = args.scheduler_hook.resolve()
    for label, path in (
        ("formal config", formal_config_path),
        ("scheduler hook", scheduler_hook_path),
        ("strict runner", SCRIPT),
        ("strict runtime", STRICT_RUNTIME_PATH),
        ("tool pool", TOOL_POOL_PATH),
        ("mapper code", MAPPER_CODE_PATH),
        ("matrix wrapper", MATRIX_WRAPPER_PATH),
        ("smoke script", SMOKE_SCRIPT_PATH),
        ("vLLM start script", START_VLLM_PATH),
        ("vLLM stop script", STOP_VLLM_PATH),
        ("scheduler sitecustomize", SITECUSTOMIZE_PATH),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    formal_exports = _formal_config_exports(formal_config_path)
    pinned_model_snapshot = _validate_formal_environment_contract(formal_exports)
    pinned_model_inventory = _model_snapshot_inventory(pinned_model_snapshot)
    if args.tokenizer != formal_exports.get("MODEL_ID"):
        raise ValueError(
            "--tokenizer must equal the frozen MODEL_ID; the exact revision "
            "snapshot is derived from the formal config"
        )
    fixed = _load_fixed_split(args.fixed_bundle.resolve())
    # The evaluation corpus remains unopened until all train/tune-dependent
    # policy choices and the physical clock have been serialized and hashed.
    calibration_sessions = _load_role_sessions(
        fixed, args.execution_traces.resolve(), "calibration"
    )
    tuning_sessions = _load_role_sessions(
        fixed, args.execution_traces.resolve(), "tuning"
    )
    provenance = _training_provenance(fixed, calibration_sessions)
    duration_predictor, duration_artifact = CausalDurationPredictor.fit(
        calibration_sessions,
        training_provenance=provenance,
        ewma_alpha=args.duration_ewma_alpha,
    )
    _, tail_artifact = CausalTailPredictor.fit(
        calibration_sessions,
        training_provenance=provenance,
        duration_predictor=duration_predictor,
    )
    selected_top_k, tuning_selection = select_tuning_top_k(
        mapper=fixed["mapper"],
        tuning_sessions=tuning_sessions,
        max_top_k=args.max_top_k,
        min_precision=args.min_prediction_precision,
    )
    tokenizer = _load_tokenizer(pinned_model_snapshot)
    calibration_service_samples_s = _calibration_service_samples(
        calibration_sessions
    )
    # A new experiment receives a fresh 256-bit private salt.  A code-only
    # rerun may instead reuse the complete previously frozen clock artifact;
    # its signature, calibration lineage, sample pools, and execution contract
    # are all checked here, before held-out evaluation content is opened.
    service_clock_artifact = _prepare_service_clock_artifact(
        reuse_path=getattr(args, "reuse_service_clock", None),
        training_provenance=provenance,
        samples_by_tool_s=calibration_service_samples_s,
    )
    arrival_rows, arrival_provenance = _arrival_rows(args.arrivals, args.sessions)
    runtime_parameters = _build_runtime_parameters(
        args=args,
        arrival_rows=arrival_rows,
        arrival_provenance=arrival_provenance,
    )
    output_dir = requested_output_dir.with_name(
        f".{requested_output_dir.name}.prepare-{uuid.uuid4()}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    duration_path = output_dir / "duration_predictor.json"
    tail_path = output_dir / "tail_predictor.json"
    service_clock_path = output_dir / "service_clock.json"
    selection_path = output_dir / "tuning_selection.json"
    invocation_provenance_path = output_dir / "invocation_predictor_provenance.json"
    runtime_parameters_path = output_dir / "runtime_parameters.json"
    invocation_provenance = signed_payload(
        {
            "schema": INVOCATION_PROVENANCE_SCHEMA,
            "training_role": "calibration",
            "training_root_ids_sha256": provenance["session_ids_sha256"],
            "uses_evaluation_labels": False,
            "input_features": [
                "last_completed_tool_name",
                "current_visible_search_result_urls",
                "current_visible_search_result_ranks",
                "current_visible_search_result_ordinals",
                "frozen_top_k",
            ],
            "fit_code_sha256": file_sha256(MAPPER_CODE_PATH),
            "source_artifact": _bound_file(fixed["mapper_path"], output_dir),
            # The wrapper's logical predictor identity remains the immutable
            # mapper model hash; provenance_sha256 authenticates this wrapper.
            "artifact_sha256": fixed["mapper_artifact"]["artifact_sha256"],
            "invocation_predictor_artifact_sha256": fixed["mapper_artifact"][
                "artifact_sha256"
            ],
        },
        "provenance_sha256",
    )
    write_json(duration_path, duration_artifact)
    write_json(tail_path, tail_artifact)
    write_json(service_clock_path, service_clock_artifact)
    service_clock_path.chmod(0o600)
    write_json(selection_path, tuning_selection)
    write_json(invocation_provenance_path, invocation_provenance)
    write_json(runtime_parameters_path, runtime_parameters)
    runtime_parameters_path.chmod(0o444)

    frozen_runtime_files = {
        "runner": _bound_file(SCRIPT, output_dir),
        "strict_runtime": _bound_file(STRICT_RUNTIME_PATH, output_dir),
        "tool_pool": _bound_file(TOOL_POOL_PATH, output_dir),
        "mapper_code": _bound_file(MAPPER_CODE_PATH, output_dir),
        "matrix_wrapper": _bound_file(MATRIX_WRAPPER_PATH, output_dir),
        "smoke_script": _bound_file(SMOKE_SCRIPT_PATH, output_dir),
        "start_vllm": _bound_file(START_VLLM_PATH, output_dir),
        "stop_vllm": _bound_file(STOP_VLLM_PATH, output_dir),
        "sitecustomize": _bound_file(SITECUSTOMIZE_PATH, output_dir),
        "formal_config": _bound_file(formal_config_path, output_dir),
        "scheduler_hook": _bound_file(scheduler_hook_path, output_dir),
    }
    model_snapshot_contract = {
        "path": str(pinned_model_snapshot),
        "model_id": formal_exports["MODEL_ID"],
        "model_revision": formal_exports["MODEL_REVISION"],
        "config_json_sha256": file_sha256(pinned_model_snapshot / "config.json"),
        "inventory": pinned_model_inventory,
        "inventory_sha256": pinned_model_inventory["inventory_sha256"],
        "derivation": "canonical HF_HOME/models--MODEL_ID/snapshots/MODEL_REVISION",
    }

    policy_freeze = signed_payload(
        {
            "schema": "paste_repro.strict_policy_freeze.v1",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "claim_scope": args.claim_scope,
            "model": args.model,
            "model_revision": args.model_revision,
            "model_snapshot_contract": model_snapshot_contract,
            "max_model_len": args.max_model_len,
            "output_cap": args.output_cap,
            "mapper_artifact_sha256": fixed["mapper_artifact"]["artifact_sha256"],
            "invocation_predictor_provenance_sha256": invocation_provenance[
                "provenance_sha256"
            ],
            "duration_predictor_artifact_sha256": duration_artifact["artifact_sha256"],
            "tail_predictor_artifact_sha256": tail_artifact["artifact_sha256"],
            "service_clock_artifact_sha256": service_clock_artifact["artifact_sha256"],
            "tuning_selection_sha256": tuning_selection["selection_sha256"],
            "selected_top_k": selected_top_k,
            "runtime_capacities": {
                "max_active_tasks": args.max_active_tasks,
                "visit_capacity": args.visit_capacity,
                "speculative_cap": args.speculative_cap,
            },
            "runtime_parameters": runtime_parameters,
            "runtime_parameters_artifact": _bound_file(
                runtime_parameters_path, output_dir
            ),
            "selection_completed_before_evaluation_open": True,
            "arrival_process": dict(arrival_provenance),
            "fixed_split_bundle_sha256": fixed["bundle"]["bundle_sha256"],
            "fixed_split_manifest_sha256": fixed["manifest"]["manifest_sha256"],
            "frozen_runtime_files": frozen_runtime_files,
        },
        "freeze_sha256",
    )
    policy_freeze_path = output_dir / "policy_freeze.json"
    write_json(policy_freeze_path, policy_freeze)
    for frozen_path in (
        duration_path,
        tail_path,
        service_clock_path,
        selection_path,
        invocation_provenance_path,
        policy_freeze_path,
    ):
        frozen_path.chmod(0o400)

    root_id_bindings: dict[str, Any] = {}
    for role in ("calibration", "tuning", "final"):
        root_ids = sorted(fixed["ids"][role])
        root_id_artifact = signed_payload(
            {
                "schema": "paste_repro.strict_root_ids.v1",
                "role": role,
                "root_ids": root_ids,
                "source_session_ids": root_ids,
                "source_session_ids_sha256": canonical_sha256(root_ids),
            },
            "artifact_sha256",
        )
        root_id_path = output_dir / f"{role}.root_ids.json"
        write_json(root_id_path, root_id_artifact)
        root_id_path.chmod(0o444)
        root_id_bindings[role] = _bound_file(root_id_path, output_dir)
    if args.claim_scope == "retrospective":
        root_id_bindings["previously_observed_evaluation"] = dict(
            root_id_bindings["final"]
        )

    # This is the first read of held-out evaluation trace content.
    evaluation_opened_at = datetime.now(timezone.utc).isoformat()
    final_sessions = _load_role_sessions(
        fixed, args.execution_traces.resolve(), "final"
    )
    role_sessions = {"tuning": tuning_sessions, "final": final_sessions}

    plan_bindings: dict[str, Any] = {}
    for role in ("tuning", "final"):
        raw_sha = {
            str(row["session_id"]): str(row["sha256"])
            for row in fixed["roles"][role]
        }
        public, sealed, diagnostics = build_role_plans(
            role=role,
            sessions=role_sessions[role],
            raw_sha_by_id=raw_sha,
            tokenizer=tokenizer,
            max_model_len=args.max_model_len,
            output_cap=args.output_cap,
            arrivals=arrival_rows,
            arrival_provenance=arrival_provenance,
            service_clock_artifact_sha256=service_clock_artifact["artifact_sha256"],
            claim_scope=args.claim_scope,
        )
        public_path, sealed_path, diagnostics_path = _write_role_plan_files(
            output_dir=output_dir,
            role=role,
            public=public,
            sealed=sealed,
            diagnostics=diagnostics,
        )
        plan_bindings[role] = {
            "public": _bound_file(public_path, output_dir),
            "sealed": _bound_file(sealed_path, output_dir),
            "heldout_diagnostics": _bound_file(diagnostics_path, output_dir),
            "public_plan_sha256": public["plan_sha256"],
            "sealed_plan_sha256": sealed["sealed_sha256"],
            "heldout_diagnostics_sha256": diagnostics["diagnostics_sha256"],
            "service_clock_artifact_sha256": service_clock_artifact["artifact_sha256"],
            "source_roots": len(role_sessions[role]),
            "replicas": len(public["traces"]),
        }

    bundle = signed_payload(
        {
            "schema": BUNDLE_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "model_revision": args.model_revision,
            "model_snapshot_contract": model_snapshot_contract,
            "fixed_split": _bound_file(args.fixed_bundle.resolve(), output_dir),
            "fixed_split_bundle_sha256": fixed["bundle"]["bundle_sha256"],
            "fixed_split_manifest_sha256": fixed["manifest"]["manifest_sha256"],
            "claim_scope": args.claim_scope,
            "roles": {role: sorted(fixed["ids"][role]) for role in fixed["ids"]},
            "execution_trace_directory": str(args.execution_traces.resolve()),
            "execution_trace_directory_role": "authority graph and held-out diagnostics only",
            "mapper": _bound_file(fixed["mapper_path"], output_dir),
            "mapper_artifact_sha256": fixed["mapper_artifact"]["artifact_sha256"],
            "invocation_predictor_provenance": _bound_file(
                invocation_provenance_path, output_dir
            ),
            "invocation_predictor_provenance_sha256": invocation_provenance[
                "provenance_sha256"
            ],
            "duration_predictor": _bound_file(duration_path, output_dir),
            "duration_predictor_artifact_sha256": duration_artifact["artifact_sha256"],
            "tail_predictor": _bound_file(tail_path, output_dir),
            "tail_predictor_artifact_sha256": tail_artifact["artifact_sha256"],
            "service_clock": _bound_file(service_clock_path, output_dir),
            "service_clock_artifact_sha256": service_clock_artifact["artifact_sha256"],
            "root_id_artifacts": root_id_bindings,
            "policy_freeze": _bound_file(policy_freeze_path, output_dir),
            "policy_freeze_sha256": policy_freeze["freeze_sha256"],
            "frozen_runtime_files": frozen_runtime_files,
            "selection_completed_before_evaluation_open": True,
            "policy_frozen_at": policy_freeze["frozen_at"],
            "evaluation_opened_at": evaluation_opened_at,
            "tuning_selection": _bound_file(selection_path, output_dir),
            "tuning_selection_sha256": tuning_selection["selection_sha256"],
            "selected_top_k": selected_top_k,
            "runtime_capacities": dict(policy_freeze["runtime_capacities"]),
            "runtime_parameters": runtime_parameters,
            "runtime_parameters_artifact": _bound_file(
                runtime_parameters_path, output_dir
            ),
            "plans": plan_bindings,
            "policy_contract": {
                "call_graph_mode": CALL_GRAPH_MODE,
                "candidate_materialization": "timed current search response only",
                "duration_prediction": "calibration40 plus completed-job EWMA",
                "actual_duration_access": (
                    "unavailable to policy runtime and service assignment; held-out "
                    "trace durations exist only in separate offline diagnostics"
                ),
                "tool_execution_service": (
                    "generic calibration-only hashed empirical clock; no evaluation "
                    "invocation enumeration; evaluation durations are diagnostics only"
                ),
                "offline_tool_credit_s": 0,
                "forbidden_metadata": [
                    "n", "rc", "rlmt", "npt", "nmt", "nw", "nwc", "rtw", "eg", "is_final"
                ],
                "output_budget": "uniform public cap",
            },
        },
        "bundle_sha256",
    )
    _assert_policy_facing_document_safe(bundle, label="strict policy bundle")
    bundle_path = output_dir / "bundle.json"
    write_json(bundle_path, bundle)
    bundle_path.chmod(0o444)
    if requested_output_dir.exists():
        raise FileExistsError(
            f"preparation target appeared during freeze: {requested_output_dir}"
        )
    output_dir.rename(requested_output_dir)
    return {
        "bundle": str(requested_output_dir / "bundle.json"),
        "bundle_sha256": bundle["bundle_sha256"],
    }


def load_strict_bundle(path: Path, role: str) -> dict[str, Any]:
    bundle = validate_signed_payload(
        read_json(path), "bundle_sha256", label="strict A/B/E/F bundle"
    )
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported strict bundle schema")
    if role not in {"tuning", "final"}:
        raise ValueError("evaluation role must be tuning or final")
    anchor = path.parent
    mapper_path = _verify_bound_file(bundle["mapper"], anchor, "mapper")
    invocation_provenance_path = _verify_bound_file(
        bundle["invocation_predictor_provenance"],
        anchor,
        "invocation predictor provenance",
    )
    policy_freeze_path = _verify_bound_file(
        bundle["policy_freeze"], anchor, "policy freeze"
    )
    duration_path = _verify_bound_file(
        bundle["duration_predictor"], anchor, "duration predictor"
    )
    tail_path = _verify_bound_file(bundle["tail_predictor"], anchor, "tail predictor")
    service_clock_path = _verify_bound_file(
        bundle["service_clock"], anchor, "service clock"
    )
    runtime_parameters_path = _verify_bound_file(
        bundle["runtime_parameters_artifact"],
        anchor,
        "treatment-neutral runtime parameters",
    )
    policy_freeze = validate_signed_payload(
        read_json(policy_freeze_path), "freeze_sha256", label="strict policy freeze"
    )
    if policy_freeze.get("freeze_sha256") != bundle.get("policy_freeze_sha256"):
        raise ValueError("policy-freeze identity mismatch")
    if bundle.get("frozen_runtime_files") != policy_freeze.get(
        "frozen_runtime_files"
    ):
        raise ValueError("bundle/policy-freeze runtime bindings differ")
    snapshot_contract = bundle.get("model_snapshot_contract")
    if (
        not isinstance(snapshot_contract, Mapping)
        or snapshot_contract != policy_freeze.get("model_snapshot_contract")
        or snapshot_contract.get("model_id") != bundle.get("model")
        or snapshot_contract.get("model_revision") != bundle.get("model_revision")
    ):
        raise ValueError("bundle/policy-freeze model snapshot contract differs")
    snapshot_path = Path(str(snapshot_contract.get("path", "")))
    observed_snapshot_inventory = (
        _model_snapshot_inventory(snapshot_path)
        if snapshot_path.is_absolute() and snapshot_path.is_dir()
        else None
    )
    if (
        not snapshot_path.is_absolute()
        or not snapshot_path.is_dir()
        or snapshot_path.resolve() != snapshot_path
        or not (snapshot_path / "config.json").is_file()
        or file_sha256(snapshot_path / "config.json")
        != snapshot_contract.get("config_json_sha256")
        or observed_snapshot_inventory != snapshot_contract.get("inventory")
        or (
            observed_snapshot_inventory is not None
            and observed_snapshot_inventory.get("inventory_sha256")
            != snapshot_contract.get("inventory_sha256")
        )
    ):
        raise ValueError("frozen model snapshot path/config binding is invalid")
    if policy_freeze.get("selection_completed_before_evaluation_open") is not True:
        raise ValueError("policy freeze does not precede evaluation opening")
    if bundle.get("runtime_capacities") != policy_freeze.get("runtime_capacities"):
        raise ValueError("bundle/policy-freeze runtime capacities differ")
    runtime_parameters = _validate_runtime_parameters(
        read_json(runtime_parameters_path)
    )
    if (
        bundle.get("runtime_parameters") != runtime_parameters
        or policy_freeze.get("runtime_parameters") != runtime_parameters
        or policy_freeze.get("runtime_parameters_artifact")
        != bundle.get("runtime_parameters_artifact")
    ):
        raise ValueError("runtime parameter artifact/bundle/policy-freeze binding differs")
    capacities = bundle.get("runtime_capacities")
    if (
        not isinstance(capacities, Mapping)
        or type(capacities.get("max_active_tasks")) is not int
        or type(capacities.get("visit_capacity")) is not int
        or type(capacities.get("speculative_cap")) is not int
        or int(capacities["max_active_tasks"]) <= 0
        or int(capacities["visit_capacity"]) <= 0
        or not 0 <= int(capacities["speculative_cap"]) <= int(
            capacities["visit_capacity"]
        )
    ):
        raise ValueError("frozen runtime capacities are invalid")
    runtime_values = runtime_parameters["parameters"]
    if dict(capacities) != {
        "max_active_tasks": runtime_values["max_active_tasks"],
        "visit_capacity": runtime_values["tool_capacity"],
        "speculative_cap": runtime_values["configured_speculation_capacity"],
    }:
        raise ValueError("runtime capacity aliases differ from normalized parameters")
    frozen_runtime_paths = {
        label: _verify_bound_file(entry, anchor, f"frozen runtime {label}")
        for label, entry in policy_freeze.get("frozen_runtime_files", {}).items()
    }
    if set(frozen_runtime_paths) != {
        "runner",
        "strict_runtime",
        "tool_pool",
        "mapper_code",
        "matrix_wrapper",
        "smoke_script",
        "start_vllm",
        "stop_vllm",
        "sitecustomize",
        "formal_config",
        "scheduler_hook",
    }:
        raise ValueError("policy freeze lacks the complete runtime/config binding")
    formal_exports = _formal_config_exports(frozen_runtime_paths["formal_config"])
    derived_snapshot_path = _validate_formal_environment_contract(formal_exports)
    if (
        derived_snapshot_path != snapshot_path
        or formal_exports.get("MODEL_ID") != snapshot_contract.get("model_id")
        or formal_exports.get("MODEL_REVISION")
        != snapshot_contract.get("model_revision")
    ):
        raise ValueError(
            "frozen formal config derives a different model snapshot contract"
        )
    invocation_provenance = validate_signed_payload(
        read_json(invocation_provenance_path),
        "provenance_sha256",
        label="invocation predictor provenance",
    )
    if invocation_provenance.get("schema") != INVOCATION_PROVENANCE_SCHEMA:
        raise ValueError("unsupported invocation predictor provenance schema")
    if invocation_provenance.get("uses_evaluation_labels") is not False:
        raise ValueError("invocation predictor provenance uses evaluation labels")
    calibration_root_hash = canonical_sha256(sorted(bundle["roles"]["calibration"]))
    if invocation_provenance.get("training_role") != "calibration" or invocation_provenance.get(
        "training_root_ids_sha256"
    ) != calibration_root_hash:
        raise ValueError("invocation predictor calibration-root binding is invalid")
    if invocation_provenance.get("input_features") != [
        "last_completed_tool_name",
        "current_visible_search_result_urls",
        "current_visible_search_result_ranks",
        "current_visible_search_result_ordinals",
        "frozen_top_k",
    ]:
        raise ValueError("invocation predictor input feature schema is invalid")
    if invocation_provenance.get("fit_code_sha256") != file_sha256(
        frozen_runtime_paths["mapper_code"]
    ):
        raise ValueError("invocation predictor fit-code binding is invalid")
    if invocation_provenance.get("provenance_sha256") != bundle.get(
        "invocation_predictor_provenance_sha256"
    ):
        raise ValueError("invocation predictor provenance identity mismatch")
    provenance_mapper_path = _verify_bound_file(
        invocation_provenance["source_artifact"],
        invocation_provenance_path.parent,
        "invocation predictor source artifact",
    )
    if provenance_mapper_path.resolve() != mapper_path.resolve():
        raise ValueError("invocation provenance points at a different mapper")
    public_path = _verify_bound_file(bundle["plans"][role]["public"], anchor, "public plan")
    sealed_path = _verify_bound_file(bundle["plans"][role]["sealed"], anchor, "sealed plan")
    public = validate_signed_payload(
        read_json(public_path), "plan_sha256", label="public plan"
    )
    sealed = validate_signed_payload(
        read_json(sealed_path), "sealed_sha256", label="sealed plan"
    )
    if public.get("schema") != PUBLIC_PLAN_SCHEMA or sealed.get("schema") != SEALED_PLAN_SCHEMA:
        raise ValueError("strict plan schema mismatch")
    if public.get("role") != role or sealed.get("role") != role:
        raise ValueError("strict plan role mismatch")
    if public.get("claim_scope") != bundle.get("claim_scope") or sealed.get(
        "claim_scope"
    ) != bundle.get("claim_scope"):
        raise ValueError("strict plan claim scope mismatch")
    if sealed.get("public_plan_sha256") != public["plan_sha256"]:
        raise ValueError("sealed outcomes are not bound to the public plan")
    _assert_policy_facing_document_safe(public, label="public plan")
    public_traces = public.get("traces")
    if not isinstance(public_traces, list) or not public_traces:
        raise ValueError("public plan must contain non-empty opaque trace metadata")
    public_trace_ids: list[str] = []
    for index, trace in enumerate(public_traces):
        if not isinstance(trace, Mapping):
            raise ValueError(f"public trace {index} is not an object")
        if set(trace) != PUBLIC_TRACE_FIELDS:
            raise ValueError(
                f"public trace {index} has non-metadata fields: "
                f"{sorted(set(trace) - PUBLIC_TRACE_FIELDS)}"
            )
        trace_id = str(trace.get("trace_id", ""))
        if not trace_id or trace.get("session_id") != trace_id:
            raise ValueError(f"public trace {index} has invalid opaque identity")
        public_trace_ids.append(trace_id)
    if len(set(public_trace_ids)) != len(public_trace_ids):
        raise ValueError("public plan has duplicate trace identities")
    sealed_steps = sealed.get("trace_steps")
    sealed_lineage = sealed.get("trace_lineage")
    if not isinstance(sealed_steps, Mapping) or set(sealed_steps) != set(
        public_trace_ids
    ):
        raise ValueError("sealed trace steps do not exactly cover public trace identities")
    if not isinstance(sealed_lineage, Mapping) or set(sealed_lineage) != set(
        public_trace_ids
    ):
        raise ValueError("sealed trace lineage does not exactly cover public identities")
    for trace_id in public_trace_ids:
        steps = sealed_steps[trace_id]
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"sealed trace {trace_id!r} has no causal steps")
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping) or set(step) != {"request", "tools_after"}:
                raise ValueError(
                    f"sealed trace {trace_id!r} step {step_index} is malformed"
                )
            if not isinstance(step["request"], Mapping) or not isinstance(
                step["tools_after"], list
            ):
                raise ValueError(
                    f"sealed trace {trace_id!r} step {step_index} is malformed"
                )
    if (
        sealed.get("service_clock_artifact_sha256")
        != bundle.get("service_clock_artifact_sha256")
    ):
        raise ValueError("sealed plan/service clock identity mismatch")
    mapper, mapper_artifact = load_artifact(mapper_path)
    duration_artifact = read_json(duration_path)
    tail_artifact = read_json(tail_path)
    service_clock_artifact = read_json(service_clock_path)
    duration = CausalDurationPredictor.from_artifact(duration_artifact)
    tail = CausalTailPredictor(tail_artifact)
    service_clock = CalibrationHashedServiceClock(service_clock_artifact)
    if mapper_artifact["artifact_sha256"] != bundle["mapper_artifact_sha256"]:
        raise ValueError("mapper artifact identity mismatch")
    if invocation_provenance.get("invocation_predictor_artifact_sha256") != bundle[
        "mapper_artifact_sha256"
    ] or invocation_provenance.get("artifact_sha256") != bundle[
        "mapper_artifact_sha256"
    ]:
        raise ValueError("invocation provenance/mapper logical identity mismatch")
    if duration.artifact_sha256 != bundle["duration_predictor_artifact_sha256"]:
        raise ValueError("duration artifact identity mismatch")
    if tail.artifact_sha256 != bundle["tail_predictor_artifact_sha256"]:
        raise ValueError("tail artifact identity mismatch")
    if service_clock.artifact_sha256 != bundle["service_clock_artifact_sha256"]:
        raise ValueError("service clock artifact identity mismatch")
    return {
        "bundle": bundle,
        "mapper": mapper,
        "duration": duration,
        "tail": tail,
        "service_clock": service_clock,
        "public": public,
        "sealed": sealed,
        "artifact_paths": {
            "invocation_predictor": invocation_provenance_path,
            "invocation_predictor_source": mapper_path,
            "duration_predictor": duration_path,
            "tail_predictor": tail_path,
            "service_clock": service_clock_path,
            "runtime_parameters": runtime_parameters_path,
            "public_plan": public_path,
            "sealed_plan": sealed_path,
        },
        "frozen_runtime_paths": frozen_runtime_paths,
    }


def _verify_runtime_file(path: Path, expected_sha256: str, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    observed = file_sha256(resolved)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def validate_matrix_execution_contract(
    *,
    loaded: Mapping[str, Any],
    config_path: Path,
    scheduler_hook_path: Path,
    sitecustomize_path: Path,
    start_vllm_path: Path,
    stop_vllm_path: Path,
    hook_dir: Path,
    max_active_tasks: int,
    visit_capacity: int,
    speculative_cap: int,
) -> None:
    """Bind shell-selected launch paths/capacities to the pre-evaluation freeze."""

    frozen = loaded.get("frozen_runtime_paths")
    if not isinstance(frozen, Mapping):
        raise ValueError("loaded bundle lacks frozen runtime paths")
    supplied_paths = {
        "formal_config": config_path,
        "scheduler_hook": scheduler_hook_path,
        "sitecustomize": sitecustomize_path,
        "start_vllm": start_vllm_path,
        "stop_vllm": stop_vllm_path,
    }
    for role, supplied in supplied_paths.items():
        expected = frozen.get(role)
        if not isinstance(expected, Path) or supplied.resolve() != expected.resolve():
            raise ValueError(f"matrix {role} path differs from the policy freeze")
    resolved_hook_dir = hook_dir.resolve()
    if (resolved_hook_dir / "sched_policy_patch.py").resolve() != scheduler_hook_path.resolve():
        raise ValueError("VLLM_HOOK_DIR does not select the frozen scheduler hook")
    if (resolved_hook_dir / "sitecustomize.py").resolve() != sitecustomize_path.resolve():
        raise ValueError("VLLM_HOOK_DIR does not select the frozen sitecustomize")
    actual_capacities = {
        "max_active_tasks": int(max_active_tasks),
        "visit_capacity": int(visit_capacity),
        "speculative_cap": int(speculative_cap),
    }
    if actual_capacities != loaded.get("bundle", {}).get("runtime_capacities"):
        raise ValueError("matrix runtime capacities differ from the policy freeze")


def _scheduler_request_id(meta: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(meta), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"schedx{encoded.hex()}z"


def _fcfs_request_id(trace_id: str, request_index: int) -> str:
    """Return a deterministic opaque ID which cannot activate the schedx hook."""

    digest = hashlib.sha256(
        f"strict-fcfs\0{trace_id}\0{int(request_index)}".encode("utf-8")
    ).hexdigest()
    return f"strict-fcfs-{digest}"


def _tool_invocation_digest(
    tool_name: str, tool_arguments: Mapping[str, Any]
) -> str:
    return canonical_sha256(
        {
            "tool": str(tool_name),
            "arguments": normalized_tool_arguments(tool_name, tool_arguments),
        }
    )


def _atomic_visit_digests(descriptor: Mapping[str, Any]) -> tuple[str, ...]:
    if str(descriptor.get("tool_name")) != "visit":
        return ()
    arguments = descriptor.get("tool_args", {})
    if not isinstance(arguments, Mapping):
        return ()
    raw_urls = arguments.get("url")
    urls = [raw_urls] if isinstance(raw_urls, str) else list(raw_urls or [])
    return tuple(
        _tool_invocation_digest("visit", {"url": str(url)}) for url in urls
    )


def _causal_tool_duration_evidence(
    *,
    duration_predictor: CausalDurationPredictor,
    service_clock: CalibrationHashedServiceClock,
    descriptor: Mapping[str, Any],
) -> dict[str, float]:
    """Predict first, then reveal the independent clock assignment post-authority."""

    tool_name = str(descriptor["tool_name"])
    arguments = descriptor.get("tool_args", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("sealed authority tool arguments must be an object")
    if tool_name == "visit":
        raw_urls = arguments.get("url")
        urls = [raw_urls] if isinstance(raw_urls, str) else list(raw_urls or [])
        # Complete every policy prediction before reading any private clock
        # realization, including for multi-URL Visit calls.
        predicted_units = [
            duration_predictor.estimate("visit", str(url)).service_s
            for url in urls
        ]
        assigned_units = [
            service_clock.service_s(
                tool_name="visit", tool_arguments={"url": str(url)}
            )
            for url in urls
        ]
        predicted = sum(predicted_units)
        assigned = sum(assigned_units)
    else:
        predicted = duration_predictor.estimate(tool_name).service_s
        assigned = service_clock.service_s(
            tool_name=tool_name, tool_arguments=arguments
        )
    return {
        "tool_service_s_hat": float(predicted),
        "assigned_service_s": float(assigned),
        "duration_prediction_absolute_error_s": abs(float(predicted) - float(assigned)),
    }


def aggregate_speculation_execution_events(
    transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse executor callbacks into one auditable row per physical job.

    ``authority_claimed_at_monotonic_s`` is the first resource-classification
    boundary, not the most recent access to a completed cached result.  Keep
    the claim value carried by every raw callback in ``state_transitions`` so
    the independent auditor can reconstruct that boundary without trusting
    this function's top-level projection.
    """

    jobs: dict[int, dict[str, Any]] = {}
    terminal_events = {
        "completed",
        "cancelled_preempted",
        "cancelled_authority_superseded",
        "cancelled_prediction_miss",
        "cancelled_window_expired",
        "cancelled_session_close",
        "cancelled_pool_close",
    }
    for transition in sorted(
        transitions,
        key=lambda row: (float(row["at_monotonic_s"]), int(row["job_id"])),
    ):
        job_id = int(transition["job_id"])
        identity = {
            "prediction_id": str(transition["prediction_id"]),
            "trace_id": str(transition["trace_id"]),
            "request_index": int(transition["request_index"]),
            "candidate_invocation_digest": str(
                transition["candidate_invocation_digest"]
            ),
        }
        row = jobs.setdefault(
            job_id,
            {
                "job_id": job_id,
                **identity,
                "admitted_at_monotonic_s": None,
                "physical_started_at_monotonic_s": None,
                "terminal_at_monotonic_s": None,
                "terminal_state": None,
                "authority_claimed_at_monotonic_s": None,
                "assigned_service_s": float(transition["assigned_service_s"]),
                "speculative_resource_s": 0.0,
                "demand_resource_s": 0.0,
                "total_worker_service_s": 0.0,
                "service_s": 0.0,
                "claimed_by_authority": False,
                "state_transitions": [],
            },
        )
        if any(row[key] != value for key, value in identity.items()):
            raise RuntimeError(f"speculative job identity changed: {job_id}")
        if not math.isclose(
            float(row["assigned_service_s"]),
            float(transition["assigned_service_s"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(f"speculative job assignment changed: {job_id}")
        event = str(transition["event"])
        at_s = float(transition["at_monotonic_s"])
        if "authority_claimed_at_monotonic_s" not in transition:
            raise RuntimeError(
                "speculative callback lacks explicit first authority-claim "
                f"evidence: {job_id}"
            )
        transition_claimed_at = transition[
            "authority_claimed_at_monotonic_s"
        ]
        if transition_claimed_at is not None:
            transition_claimed_at = float(transition_claimed_at)
            if not math.isfinite(transition_claimed_at) or transition_claimed_at < 0.0:
                raise RuntimeError(
                    f"speculative job has an invalid authority claim: {job_id}"
                )
            prior_claimed_at = row["authority_claimed_at_monotonic_s"]
            if prior_claimed_at is None:
                row["authority_claimed_at_monotonic_s"] = transition_claimed_at
            elif not math.isclose(
                float(prior_claimed_at),
                transition_claimed_at,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "speculative job's first authority claim changed across "
                    f"callbacks: {job_id}"
                )
        row["state_transitions"].append(
            {
                "event": event,
                "at_monotonic_s": at_s,
                "authority_claimed_at_monotonic_s": transition_claimed_at,
            }
        )
        if event == "admitted":
            row["admitted_at_monotonic_s"] = at_s
        elif event == "physical_started":
            row["physical_started_at_monotonic_s"] = at_s
        if event in terminal_events:
            row["terminal_at_monotonic_s"] = at_s
            row["terminal_state"] = str(transition["state"])
        for field in (
            "speculative_resource_s",
            "demand_resource_s",
            "total_worker_service_s",
            "service_s",
        ):
            row[field] = max(row[field], float(transition[field]))
        row["claimed_by_authority"] = bool(
            row["claimed_by_authority"] or transition["claimed_by_authority"]
        )
    result = [jobs[key] for key in sorted(jobs)]
    assignment_by_candidate: dict[str, float] = {}
    for row in result:
        admitted_at = row["admitted_at_monotonic_s"]
        started_at = row["physical_started_at_monotonic_s"]
        terminal_at = row["terminal_at_monotonic_s"]
        if admitted_at is None or terminal_at is None:
            raise RuntimeError(f"speculative job lacks admission/terminal evidence: {row['job_id']}")
        if started_at is None:
            if float(row["total_worker_service_s"]) != 0.0:
                raise RuntimeError(f"unstarted speculative job consumed service: {row['job_id']}")
        elif not admitted_at <= started_at <= terminal_at:
            raise RuntimeError(f"invalid speculative job timeline: {row['job_id']}")
        if not math.isfinite(float(row["assigned_service_s"])) or float(
            row["assigned_service_s"]
        ) <= 0.0:
            raise RuntimeError(f"invalid assigned speculative service: {row['job_id']}")
        if abs(
            float(row["total_worker_service_s"])
            - float(row["speculative_resource_s"])
            - float(row["demand_resource_s"])
        ) > 1e-9:
            raise RuntimeError(f"speculative job resource split is invalid: {row['job_id']}")
        nested_claims = [
            float(transition["authority_claimed_at_monotonic_s"])
            for transition in row["state_transitions"]
            if transition["authority_claimed_at_monotonic_s"] is not None
        ]
        first_claimed_at = row["authority_claimed_at_monotonic_s"]
        if bool(nested_claims) != bool(row["claimed_by_authority"]):
            raise RuntimeError(
                f"speculative job claim flag differs from raw callbacks: {row['job_id']}"
            )
        if nested_claims and any(
            not math.isclose(
                value,
                float(first_claimed_at),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for value in nested_claims
        ):
            raise RuntimeError(
                f"speculative job has non-immutable claim evidence: {row['job_id']}"
            )
        if started_at is not None:
            elapsed = float(terminal_at) - float(started_at)
            if first_claimed_at is None:
                expected_speculative_s = elapsed
                expected_demand_s = 0.0
            elif float(first_claimed_at) <= float(started_at):
                expected_speculative_s = 0.0
                expected_demand_s = elapsed
            elif float(first_claimed_at) >= float(terminal_at):
                expected_speculative_s = elapsed
                expected_demand_s = 0.0
            else:
                expected_speculative_s = float(first_claimed_at) - float(started_at)
                expected_demand_s = float(terminal_at) - float(first_claimed_at)
            if not math.isclose(
                float(row["speculative_resource_s"]),
                expected_speculative_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                float(row["demand_resource_s"]),
                expected_demand_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "speculative job resource split differs from its immutable "
                    f"first authority claim: {row['job_id']}"
                )
        candidate_digest = str(row["candidate_invocation_digest"])
        assigned_service_s = float(row["assigned_service_s"])
        previous_assignment = assignment_by_candidate.setdefault(
            candidate_digest, assigned_service_s
        )
        if not math.isclose(
            previous_assignment,
            assigned_service_s,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "the same candidate invocation received inconsistent physical "
                f"service assignments: {candidate_digest}"
            )
    return result


def validate_speculation_causal_timing(
    *,
    prediction_decisions: Sequence[Mapping[str, Any]],
    request_events: Sequence[Mapping[str, Any]],
    speculation_execution_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed on broker/start timing and expose auditable margins.

    Broker acceptance is distinct from physical execution.  Consequently all
    accepted jobs, including jobs cancelled while still queued, must be
    admitted inside their originating decision's causal LLM window.  Jobs
    which occupy a worker additionally need an ordered physical-start event in
    that same window.
    """

    decision_by_id: dict[str, Mapping[str, Any]] = {}
    for row in prediction_decisions:
        prediction_id = str(row["prediction_id"])
        if prediction_id in decision_by_id:
            raise RuntimeError(f"duplicate prediction decision: {prediction_id}")
        decision_by_id[prediction_id] = row
    completion_by_request: dict[tuple[str, int], float] = {}
    for row in request_events:
        request_key = (str(row["trace_id"]), int(row["request_index"]))
        if request_key in completion_by_request:
            raise RuntimeError(f"duplicate LLM request evidence: {request_key}")
        completion = float(row["llm_completed_at_monotonic_s"])
        if not math.isfinite(completion):
            raise RuntimeError(f"non-finite LLM completion timestamp: {request_key}")
        completion_by_request[request_key] = completion

    decision_to_admission: list[float] = []
    admission_to_completion: list[float] = []
    admission_to_start: list[float] = []
    start_to_completion: list[float] = []
    for row in speculation_execution_events:
        prediction_id = str(row["prediction_id"])
        decision = decision_by_id.get(prediction_id)
        if decision is None:
            raise RuntimeError(
                f"speculative execution lacks its prediction decision: {prediction_id}"
            )
        request_key = (str(row["trace_id"]), int(row["request_index"]))
        decision_key = (
            str(decision["trace_id"]),
            int(decision["request_index"]),
        )
        if request_key != decision_key:
            raise RuntimeError(
                f"speculative execution/decision request identity differs: {prediction_id}"
            )
        completion_at = completion_by_request.get(request_key)
        if completion_at is None:
            raise RuntimeError(
                f"speculative execution lacks target LLM completion: {prediction_id}"
            )
        decision_at = float(decision["decided_at_monotonic_s"])
        admitted_at = float(row["admitted_at_monotonic_s"])
        if not math.isfinite(decision_at) or not math.isfinite(admitted_at):
            raise RuntimeError(
                f"non-finite decision/admission timestamp: {prediction_id}"
            )
        if not decision_at <= admitted_at < completion_at:
            raise RuntimeError(
                "broker acceptance fell outside its target LLM causal window: "
                f"job {row['job_id']}"
            )
        decision_to_admission.append(admitted_at - decision_at)
        admission_to_completion.append(completion_at - admitted_at)

        started_raw = row.get("physical_started_at_monotonic_s")
        if started_raw is None:
            continue
        started_at = float(started_raw)
        if not math.isfinite(started_at) or not admitted_at <= started_at < completion_at:
            raise RuntimeError(
                "physical speculative start fell outside its broker/LLM causal window: "
                f"job {row['job_id']}"
            )
        admission_to_start.append(started_at - admitted_at)
        start_to_completion.append(completion_at - started_at)

    def minimum_or_none(values: Sequence[float]) -> float | None:
        return min(values) if values else None

    return {
        "schema": "paste_repro.speculation_causal_timing.v1",
        "broker_accepted_jobs": len(speculation_execution_events),
        "physical_started_jobs": len(start_to_completion),
        "minimum_decision_to_admission_s": minimum_or_none(
            decision_to_admission
        ),
        "minimum_admission_to_completion_s": minimum_or_none(
            admission_to_completion
        ),
        "minimum_admission_to_physical_start_s": minimum_or_none(
            admission_to_start
        ),
        "minimum_physical_start_to_completion_s": minimum_or_none(
            start_to_completion
        ),
    }


def prediction_metrics_from_raw_evidence(
    *,
    prediction_outcomes: Sequence[Mapping[str, Any]],
    tool_events: Sequence[Mapping[str, Any]],
    speculation_execution_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute prediction metrics without conflating enqueue with execution.

    ``candidate.admitted`` is a legacy compatibility field meaning only that
    the broker accepted/queued the candidate.  The historical aggregate names
    ``admitted_candidates``/``admitted_candidate_precision`` are deliberately
    defined here from jobs with a non-null physical-start timestamp.  This
    makes queued-then-expired work cost-free and excludes it from the mechanism
    headline while retaining separate broker-acceptance diagnostics.
    """

    raw_authority_candidates: dict[tuple[str, int], set[str]] = {}
    for event in tool_events:
        request_key = (str(event["trace_id"]), int(event["request_index"]))
        raw_authority_candidates.setdefault(request_key, set()).update(
            str(value)
            for value in event.get("authority_candidate_invocation_digests", [])
        )

    candidate_identity: dict[tuple[str, str], tuple[str, int]] = {}
    broker_accepted_keys: set[tuple[str, str]] = set()
    emitted_candidates = 0
    matched_emitted = 0
    matched_broker_accepted = 0
    decision_hits = 0
    for outcome in prediction_outcomes:
        prediction_id = str(outcome["prediction_id"])
        request_key = (str(outcome["trace_id"]), int(outcome["request_index"]))
        authoritative = raw_authority_candidates.get(request_key, set())
        if outcome["authoritative_candidate_invocation_digests"] != sorted(
            authoritative
        ):
            raise RuntimeError(
                "prediction outcome authority digests differ from raw tool events"
            )
        recomputed_matches: list[bool] = []
        for row in outcome["candidates"]:
            digest = str(row["candidate_invocation_digest"])
            candidate_key = (prediction_id, digest)
            if candidate_key in candidate_identity:
                raise RuntimeError("prediction outcome repeats a candidate invocation")
            candidate_identity[candidate_key] = request_key
            broker_accepted = bool(row["admitted"])
            if row.get("broker_accepted") is not broker_accepted:
                raise RuntimeError(
                    "candidate admitted/broker_accepted compatibility fields differ"
                )
            if broker_accepted:
                broker_accepted_keys.add(candidate_key)
            matched = digest in authoritative
            recomputed_matches.append(matched)
            if bool(row["matched_authority"]) != matched:
                raise RuntimeError(
                    "prediction outcome label differs from raw authority digests"
                )
        emitted_candidates += len(recomputed_matches)
        matched_emitted += sum(recomputed_matches)
        matched_broker_accepted += sum(
            bool(row["broker_accepted"]) and matched
            for row, matched in zip(
                outcome["candidates"], recomputed_matches, strict=True
            )
        )
        decision_hit = any(recomputed_matches)
        decision_hits += int(decision_hit)
        broker_accepted_count = sum(
            bool(row["broker_accepted"]) for row in outcome["candidates"]
        )
        expected_checksums = {
            "emitted_candidate_count": len(recomputed_matches),
            "admitted_candidate_count": broker_accepted_count,
            "broker_accepted_candidate_count": broker_accepted_count,
            "matched_emitted_candidate_count": sum(recomputed_matches),
            "matched_admitted_candidate_count": sum(
                bool(row["broker_accepted"]) and matched
                for row, matched in zip(
                    outcome["candidates"], recomputed_matches, strict=True
                )
            ),
            "matched_broker_accepted_candidate_count": sum(
                bool(row["broker_accepted"]) and matched
                for row, matched in zip(
                    outcome["candidates"], recomputed_matches, strict=True
                )
            ),
            "decision_hit": decision_hit,
        }
        if any(outcome.get(field) != expected for field, expected in expected_checksums.items()):
            raise RuntimeError(
                "prediction outcome precision checksums differ from raw evidence"
            )

    execution_by_candidate: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in speculation_execution_events:
        candidate_key = (
            str(row["prediction_id"]),
            str(row["candidate_invocation_digest"]),
        )
        if candidate_key in execution_by_candidate:
            raise RuntimeError("multiple speculative jobs claim one decision candidate")
        if candidate_key not in broker_accepted_keys:
            raise RuntimeError("physical execution is not bound to a broker-accepted candidate")
        execution_by_candidate[candidate_key] = row
    if set(execution_by_candidate) != broker_accepted_keys:
        raise RuntimeError(
            "broker-accepted candidates do not exactly match the execution ledger"
        )

    physically_started_keys = {
        key
        for key, row in execution_by_candidate.items()
        if row.get("physical_started_at_monotonic_s") is not None
    }
    matched_physical_started = sum(
        key[1] in raw_authority_candidates.get(candidate_identity[key], set())
        for key in physically_started_keys
    )
    queued_never_started = len(execution_by_candidate) - len(physically_started_keys)
    cancelled_candidates = sum(
        str(row.get("terminal_state")) == "cancelled"
        for row in execution_by_candidate.values()
    )
    physically_started_cancelled = sum(
        key in physically_started_keys and str(row.get("terminal_state")) == "cancelled"
        for key, row in execution_by_candidate.items()
    )
    broker_accepted_candidates = len(broker_accepted_keys)
    physically_started_candidates = len(physically_started_keys)
    return {
        "decisions_with_candidates": len(prediction_outcomes),
        "decision_hits": decision_hits,
        "emitted_candidates": emitted_candidates,
        "matched_emitted_candidates": matched_emitted,
        "emitted_candidate_precision": (
            matched_emitted / emitted_candidates if emitted_candidates else None
        ),
        "broker_accepted_candidates": broker_accepted_candidates,
        "matched_broker_accepted_candidates": matched_broker_accepted,
        "broker_accepted_candidate_precision": (
            matched_broker_accepted / broker_accepted_candidates
            if broker_accepted_candidates
            else None
        ),
        # Compatibility aggregate: in strict v1 results, admitted means the
        # candidate actually occupied a worker, never merely entered a queue.
        "admitted_metric_semantics": "physical_started_at_monotonic_s_is_not_null",
        "admitted_candidates": physically_started_candidates,
        "matched_admitted_candidates": matched_physical_started,
        "admitted_candidate_precision": (
            matched_physical_started / physically_started_candidates
            if physically_started_candidates
            else None
        ),
        "physical_started_candidates": physically_started_candidates,
        "matched_physical_started_candidates": matched_physical_started,
        "physical_started_candidate_precision": (
            matched_physical_started / physically_started_candidates
            if physically_started_candidates
            else None
        ),
        "queued_never_started_candidates": queued_never_started,
        "cancelled_candidates": cancelled_candidates,
        "physical_started_cancelled_candidates": physically_started_cancelled,
    }


async def _post_llm(
    session: aiohttp.ClientSession,
    *,
    request_url: str,
    model: str,
    request: Mapping[str, Any],
    request_id: str,
    timeout_s: float,
) -> tuple[int, dict[str, Any], str]:
    controls = _llm_generation_controls(request)
    payload = {
        "model": model,
        "messages": request["messages"],
        **controls,
        "request_id": request_id,
    }
    async with session.post(
        request_url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as response:
        body = await response.json(content_type=None)
        if response.status != 200:
            raise RuntimeError(f"vLLM HTTP {response.status}: {body}")
    choice = body.get("choices", [{}])[0]
    return response.status, dict(body.get("usage", {})), str(
        choice.get("message", {}).get("content", "")
    )


def _llm_generation_controls(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact public generation controls shared by every cell."""

    max_tokens = int(request["max_tokens"])
    return {
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
    }


def _llm_workload_request_sha256(
    *, model: str, request: Mapping[str, Any]
) -> str:
    """Bind semantic LLM work while excluding cell-specific request IDs."""

    return canonical_sha256(
        {
            "api": "openai_chat_completions",
            "model": str(model),
            "messages": request["messages"],
            "prompt_tokens": int(request["prompt_tokens"]),
            "generation_controls": _llm_generation_controls(request),
        }
    )


def validate_server_policy(policy_file: Path, expected: str) -> None:
    if not policy_file.is_file():
        raise FileNotFoundError(f"managed server policy file is missing: {policy_file}")
    observed = policy_file.read_text(encoding="utf-8").strip()
    if observed != expected:
        raise ValueError(f"server policy mismatch: expected {expected}, observed {observed}")


def validate_live_cell_identity(args: argparse.Namespace) -> tuple[int, list[int]]:
    if not re.fullmatch(r"cycle-[0-9]{2}-block-[0-9]{2}", args.block_id):
        raise ValueError("--block-id must use cycle-NN-block-NN")
    if not args.server_instance_id:
        raise ValueError("--server-instance-id is required")
    if not args.server_pid_file.is_file():
        raise FileNotFoundError(f"managed server PID file is missing: {args.server_pid_file}")
    try:
        server_pid = int(args.server_pid_file.read_text(encoding="utf-8").strip())
        os.kill(server_pid, 0)
    except (OSError, ValueError) as exc:
        raise ValueError("managed server PID is not live") from exc
    if str(server_pid) not in args.server_instance_id:
        raise ValueError("server instance ID is not bound to its live PID")
    try:
        gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be a comma-separated integer list") from exc
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise ValueError("--gpu-ids must be a non-empty unique non-negative list")
    if not 1 <= args.order_position <= 4:
        raise ValueError("--order-position must be in 1..4")
    return server_pid, gpu_ids


def validate_scheduler_runtime_evidence(
    args: argparse.Namespace,
    *,
    cell: Mapping[str, Any],
    server_pid: int,
) -> dict[str, Any]:
    """Fail closed on real scheduler-call evidence, not an install marker."""

    evidence_path = args.scheduler_runtime_evidence_file.resolve()
    marker_path = args.scheduler_runtime_marker_file.resolve()
    expected_marker_path = args.server_pid_file.resolve().with_suffix(
        ".scheduler_runtime.json"
    )
    if marker_path != expected_marker_path:
        raise ValueError("scheduler runtime marker path is not bound to server state")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("scheduler runtime evidence is unreadable") from exc
    if not isinstance(evidence, Mapping):
        raise ValueError("scheduler runtime evidence must be a JSON object")
    expected_use = str(cell["server_policy"]) != "fcfs"
    expected = {
        "schema": SCHEDULER_RUNTIME_EVIDENCE_SCHEMA,
        "cell": args.cell,
        "phase": "after_standardized_smoke",
        "server_pid": server_pid,
        "expected_policy": str(cell["server_policy"]),
        "hook_runtime_use_expected": expected_use,
        "patched_scheduler_invocation_verified": expected_use,
        "no_scheduler_hook_runtime_use_verified": not expected_use,
        "scheduler_hook_path": str(args.scheduler_hook_file.resolve()),
        "scheduler_hook_sha256": args.scheduler_hook_file_sha256,
        "runtime_marker_path": str(marker_path),
    }
    for name, value in expected.items():
        if evidence.get(name) != value:
            raise ValueError(
                f"scheduler runtime evidence {name} mismatch: "
                f"{evidence.get(name)!r} != {value!r}"
            )
    marker_payload = evidence.get("runtime_marker")
    if expected_use:
        if not marker_path.is_file():
            raise FileNotFoundError("patched scheduler runtime marker is missing")
        marker_sha256 = file_sha256(marker_path)
        if evidence.get("runtime_marker_sha256") != marker_sha256:
            raise ValueError("scheduler runtime marker SHA-256 mismatch")
        try:
            current_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("scheduler runtime marker is unreadable") from exc
        if marker_payload != current_marker:
            raise ValueError("scheduler runtime marker differs from smoke evidence")
        if (
            not isinstance(marker_payload, Mapping)
            or marker_payload.get("schema") != "paste.vllm.scheduler_runtime_use.v1"
            or marker_payload.get("policy") != str(cell["server_policy"])
            or marker_payload.get("scheduler_api") != "v1.Scheduler.schedule"
            or marker_payload.get("scheduler_hook_sha256")
            != args.scheduler_hook_file_sha256
            or marker_payload.get("scheduler_hook_path")
            != str(args.scheduler_hook_file.resolve())
            or marker_payload.get("python_safe_path_enforced") is not True
            or marker_payload.get("cwd_import_filter_enforced") is not True
            or marker_payload.get("working_directory_importable") is not False
            or not isinstance(
                marker_payload.get("safe_working_directory"), str
            )
            or marker_payload.get("working_directory")
            != marker_payload.get("safe_working_directory")
            or marker_payload.get("pid") != evidence.get("scheduler_calling_pid")
            or evidence.get("scheduler_calling_process_relation")
            != "server_descendant"
        ):
            raise ValueError("scheduler runtime marker does not prove the expected call")
    else:
        if marker_path.exists():
            raise ValueError("FCFS cell unexpectedly executed the scheduler hook")
        if (
            marker_payload is not None
            or evidence.get("runtime_marker_sha256") is not None
            or evidence.get("scheduler_calling_pid") is not None
            or evidence.get("scheduler_calling_process_relation") is not None
        ):
            raise ValueError("FCFS scheduler evidence contains a runtime-use claim")
    return dict(evidence)


async def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    cell_started_wall_s = time.time()
    cell = CELL_SPECS[args.cell]
    validate_server_policy(args.server_policy_file, str(cell["server_policy"]))
    server_pid, gpu_ids = validate_live_cell_identity(args)
    startup_file_hashes = {
        "runner": file_sha256(SCRIPT),
        "policy_bundle": file_sha256(args.bundle.resolve()),
        "config": _verify_runtime_file(
            args.config_file, args.config_file_sha256, "frozen config"
        ),
        "scheduler_hook": _verify_runtime_file(
            args.scheduler_hook_file,
            args.scheduler_hook_file_sha256,
            "frozen scheduler hook",
        ),
        "smoke_evidence": _verify_runtime_file(
            args.smoke_evidence_file,
            args.smoke_evidence_sha256,
            "standardized smoke evidence",
        ),
        "runtime_environment_evidence": _verify_runtime_file(
            args.runtime_environment_evidence_file,
            args.runtime_environment_evidence_sha256,
            "frozen runtime-environment evidence",
        ),
        "scheduler_runtime_evidence": _verify_runtime_file(
            args.scheduler_runtime_evidence_file,
            args.scheduler_runtime_evidence_sha256,
            "scheduler runtime evidence",
        ),
    }
    scheduler_runtime_evidence = validate_scheduler_runtime_evidence(
        args,
        cell=cell,
        server_pid=server_pid,
    )
    broker_instance_id = f"strict-broker-{uuid.uuid4()}-pid-{os.getpid()}"
    loaded = load_strict_bundle(args.bundle.resolve(), args.role)
    bundle = loaded["bundle"]
    startup_model_inventory_sha256 = str(
        bundle["model_snapshot_contract"]["inventory_sha256"]
    )
    runtime_parameters = bundle["runtime_parameters"]
    runtime_values = runtime_parameters["parameters"]
    cell_capacities = {
        "max_active_tasks": args.max_active_tasks,
        "visit_capacity": args.visit_capacity,
        "speculative_cap": args.speculative_cap,
    }
    if cell_capacities != bundle.get("runtime_capacities"):
        raise ValueError("run-cell runtime capacities differ from the policy freeze")
    if not math.isclose(
        float(args.request_timeout_s),
        float(runtime_values["request_timeout_s"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        float(args.default_predicted_output_tokens),
        float(loaded["public"]["output_cap"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("run-cell timeout/output prediction differs from the freeze")
    endpoint = urlsplit(args.server_url)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname != runtime_values["server_host"]
        or endpoint.port != int(runtime_values["server_port"])
        or endpoint.path not in {"", "/"}
    ):
        raise ValueError("run-cell server endpoint differs from the policy freeze")
    if (
        args.model != runtime_values["model_id"]
        or bundle.get("model_revision") != runtime_values["model_revision"]
        or int(loaded["public"]["max_model_len"])
        != int(runtime_values["max_model_len"])
        or int(loaded["public"]["output_cap"])
        != int(runtime_values["public_output_cap"])
        or len(loaded["public"]["traces"])
        != int(runtime_values["workload_instances"])
    ):
        raise ValueError("run-cell workload/model parameters differ from the freeze")
    frozen_runtime_paths = loaded["frozen_runtime_paths"]
    if args.config_file.resolve() != frozen_runtime_paths["formal_config"].resolve():
        raise ValueError("run-cell config path differs from the policy freeze")
    if args.scheduler_hook_file.resolve() != frozen_runtime_paths[
        "scheduler_hook"
    ].resolve():
        raise ValueError("run-cell scheduler-hook path differs from the policy freeze")
    if startup_file_hashes["config"] != file_sha256(
        frozen_runtime_paths["formal_config"]
    ):
        raise ValueError("run-cell config hash differs from the policy freeze")
    if startup_file_hashes["scheduler_hook"] != file_sha256(
        frozen_runtime_paths["scheduler_hook"]
    ):
        raise ValueError("run-cell scheduler-hook hash differs from the policy freeze")
    claim_scope = str(bundle["claim_scope"])
    if args.claim_scope is not None and args.claim_scope != claim_scope:
        raise ValueError(
            f"--claim-scope {args.claim_scope} differs from sealed bundle {claim_scope}"
        )
    if args.model != bundle.get("model"):
        raise ValueError(f"--model {args.model} differs from sealed bundle model")
    duration: CausalDurationPredictor = loaded["duration"]
    policy = StrictOnlinePolicy(
        mapper=loaded["mapper"],
        mapper_artifact_sha256=bundle["mapper_artifact_sha256"],
        duration_predictor=duration,
        tail_predictor=loaded["tail"],
        top_k=int(bundle["selected_top_k"]),
    )
    speculation = bool(cell["speculation"])
    decision_context: dict[str, tuple[str, int]] = {}
    speculative_job_transitions: list[dict[str, Any]] = []

    def record_job_event(raw: dict[str, Any]) -> None:
        """Normalize broker callbacks without exposing raw invocation values."""

        prediction_id = str(raw["prediction_id"])
        try:
            trace_id, request_index = decision_context[prediction_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown speculative decision: {prediction_id}") from exc
        speculative_job_transitions.append(
            {
                **{key: value for key, value in raw.items() if key not in {"url", "session_id"}},
                "prediction_id": prediction_id,
                "trace_id": trace_id,
                "request_index": request_index,
                "candidate_invocation_digest": _tool_invocation_digest(
                    "visit", {"url": str(raw["url"])}
                ),
            }
        )

    visit_pool = AsyncPreemptibleVisitPool(
        capacity=args.visit_capacity,
        speculative_cap=args.speculative_cap if speculation else 0,
        job_event_callback=record_job_event,
    )
    executor = SealedTraceToolExecutor(
        sealed_outcomes=loaded["sealed"]["outcomes"],
        service_clock=loaded["service_clock"],
        duration_predictor=duration,
        visit_pool=visit_pool,
    )
    gate = asyncio.Semaphore(args.max_active_tasks)
    experiment_started = time.monotonic()
    request_events: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    prediction_decisions: list[dict[str, Any]] = []
    prediction_outcomes: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    result_lock = asyncio.Lock()

    async def execute_trace(trace: Mapping[str, Any]) -> None:
        release = float(trace["release_offset_s"])
        scheduled = experiment_started + release
        delay = scheduled - time.monotonic()
        if delay > 0.0:
            await asyncio.sleep(delay)
        released_at = time.monotonic()
        failure: str | None = None
        task_llm_s = 0.0
        task_prediction_s = 0.0
        task_tool_exposed_s = 0.0
        task_saved_s = 0.0
        completion_tokens = 0
        session_id = str(trace["session_id"])
        cursor = CausalTraceCursor(loaded["sealed"]["trace_steps"][session_id])
        state = CausalSessionState(predicted_output_tokens=float(args.default_predicted_output_tokens))
        causal_seq = 0
        gate_acquired = released_at
        try:
            async with gate:
                gate_acquired = time.monotonic()
                async with aiohttp.ClientSession(
                    headers=(
                        {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
                        if os.environ.get("VLLM_API_KEY")
                        else {}
                    ),
                    connector=aiohttp.TCPConnector(limit=0),
                ) as http_session:
                    while not cursor.done:
                        request_index = cursor.request_index
                        request = cursor.current_request()
                        observed_seq = causal_seq
                        causal_seq += 1
                        decision_seq = causal_seq
                        meta = None
                        if cell["scheduler"] == "causal_joint":
                            meta = policy.scheduler_metadata(
                                trace_id=session_id,
                                request_index=request_index,
                                current_call_index=int(request["call_index"]),
                                prompt_tokens=int(request["prompt_tokens"]),
                                max_tokens=int(request["max_tokens"]),
                                state=state,
                                observed_event_seq=observed_seq,
                                decision_seq=decision_seq,
                            )
                        if speculation:
                            prediction_started = time.monotonic()
                            candidates = policy.materialize_candidates(
                                current_messages=request["messages"],
                                last_completed_tool_name=state.last_completed_tool_name,
                            )
                            decided_at = time.monotonic()
                            prediction_id = f"{session_id}:request:{request_index}"
                            decision_context[prediction_id] = (session_id, request_index)
                            admitted = await executor.speculate(
                                session_id=session_id,
                                candidates=candidates,
                                after_event_index=state.last_completed_event_index,
                                decision_id=prediction_id,
                            )
                            prediction_finished = time.monotonic()
                            prediction_elapsed = prediction_finished - prediction_started
                            task_prediction_s += prediction_elapsed
                            if candidates:
                                async with result_lock:
                                    prediction_decisions.append(
                                        {
                                            "record_type": "prediction_decision",
                                            "prediction_id": prediction_id,
                                            "trace_id": session_id,
                                            "request_index": request_index,
                                            "decision_seq": decision_seq,
                                            "observed_event_seq": observed_seq,
                                            "decided_at_monotonic_s": decided_at,
                                            "candidate_invocation_digest": canonical_sha256(
                                                [
                                                    _tool_invocation_digest(
                                                        "visit", {"url": row.url}
                                                    )
                                                    for row in candidates
                                                ]
                                            ),
                                            "predictor_artifact_sha256": policy.predictor_artifact_sha256,
                                            "duration_predictor_artifact_sha256": duration.artifact_sha256,
                                            "admitted_semantics": "broker_accepted_not_physical_start",
                                            "input": {
                                                "current_call_index": int(request["call_index"]),
                                                "current_prompt_tokens": int(request["prompt_tokens"]),
                                                "committed_tool_names": [state.last_completed_tool_name],
                                            },
                                            "candidates": [
                                                {
                                                    "candidate_invocation_digest": _tool_invocation_digest(
                                                        "visit", {"url": row.url}
                                                    ),
                                                    "confidence_hat": row.confidence,
                                                    "tool_service_s_hat": row.predicted_service_s,
                                                    "prediction_source": row.prediction_source,
                                                    "admitted": bool(was_admitted),
                                                    "broker_accepted": bool(was_admitted),
                                                }
                                                for row, was_admitted in zip(candidates, admitted, strict=True)
                                            ],
                                            "prediction_latency_s": prediction_elapsed,
                                        }
                                    )
                        llm_started = time.monotonic()
                        status, usage, content = await _post_llm(
                            http_session,
                            request_url=f"{args.server_url.rstrip('/')}/v1/chat/completions",
                            model=args.model,
                            request=request,
                            request_id=(
                                _scheduler_request_id(meta)
                                if meta is not None
                                else _fcfs_request_id(session_id, request_index)
                            ),
                            timeout_s=args.request_timeout_s,
                        )
                        # Seal the authoritative LLM-completion boundary at the
                        # instant the live response returns.  Queue expiry must
                        # not push this timestamp later and make post-response
                        # speculative starts look causal.
                        llm_finished = time.monotonic()
                        if speculation:
                            executor.expire_prediction_window(
                                f"{session_id}:request:{request_index}"
                            )
                        causal_seq += 1
                        llm_completed_seq = causal_seq
                        llm_s = llm_finished - llm_started
                        task_llm_s += llm_s
                        actual_completion = int(usage.get("completion_tokens", 0) or 0)
                        if actual_completion != int(request["max_tokens"]):
                            raise RuntimeError(
                                "completion-token work mismatch: "
                                f"{actual_completion} != {request['max_tokens']}"
                            )
                        observed_prompt = int(usage.get("prompt_tokens", -1) or -1)
                        if observed_prompt != int(request["prompt_tokens"]):
                            raise RuntimeError(
                                "prompt-token work mismatch: "
                                f"{observed_prompt} != {request['prompt_tokens']}"
                            )
                        completion_tokens += actual_completion
                        state.observe_llm_completion(actual_completion)
                        request_event = {
                                    "trace_id": session_id,
                                    "source_session_id": trace["source_session_id"],
                                    "request_index": request_index,
                                    "call_index": request["call_index"],
                                    "workload_request_sha256": _llm_workload_request_sha256(
                                        model=args.model, request=request
                                    ),
                                    "http_status": status,
                                    "latency_s": llm_s,
                                    "prompt_tokens": request["prompt_tokens"],
                                    "public_max_tokens": request["max_tokens"],
                                    "usage": usage,
                                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                                    "llm_completed_seq": llm_completed_seq,
                                    "llm_completed_at_monotonic_s": llm_finished,
                                }
                        if meta is not None:
                            request_event["scheduler_metadata"] = meta
                        async with result_lock:
                            request_events.append(request_event)
                        cursor.mark_llm_completed()
                        tools = cursor.reveal_authoritative_tools()
                        revealed_tools: list[tuple[Mapping[str, Any], int, float]] = []
                        for descriptor in tools:
                            causal_seq += 1
                            revealed_tools.append(
                                (descriptor, causal_seq, time.monotonic())
                            )
                        if speculation:
                            await executor.reveal_prediction_outcome(
                                decision_id=f"{session_id}:request:{request_index}",
                                authoritative_descriptors=tools,
                            )
                            if candidates:
                                authority_full_digests = [
                                    _tool_invocation_digest(
                                        str(descriptor["tool_name"]),
                                        descriptor.get("tool_args", {}),
                                    )
                                    for descriptor in tools
                                ]
                                authority_candidate_digests = {
                                    digest
                                    for descriptor in tools
                                    for digest in _atomic_visit_digests(descriptor)
                                }
                                resolution_rows = [
                                    {
                                        "candidate_invocation_digest": _tool_invocation_digest(
                                            "visit", {"url": candidate.url}
                                        ),
                                        "admitted": bool(was_admitted),
                                        "broker_accepted": bool(was_admitted),
                                        "matched_authority": (
                                            _tool_invocation_digest(
                                                "visit", {"url": candidate.url}
                                            )
                                            in authority_candidate_digests
                                        ),
                                    }
                                    for candidate, was_admitted in zip(
                                        candidates, admitted, strict=True
                                    )
                                ]
                                authoritative_candidate_invocation_digests = sorted(
                                    authority_candidate_digests
                                )
                                if any(
                                    bool(row["matched_authority"])
                                    != (
                                        row["candidate_invocation_digest"]
                                        in authority_candidate_digests
                                    )
                                    for row in resolution_rows
                                ):
                                    raise RuntimeError(
                                        "prediction outcome labels differ from raw "
                                        "authoritative candidate digests"
                                    )
                                async with result_lock:
                                    prediction_outcomes.append(
                                        {
                                            "record_type": "prediction_outcome",
                                            "prediction_id": f"{session_id}:request:{request_index}",
                                            "trace_id": session_id,
                                            "request_index": request_index,
                                            "resolved_at_monotonic_s": time.monotonic(),
                                            "authority_present": bool(
                                                authority_candidate_digests
                                            ),
                                            "admitted_semantics": "broker_accepted_not_physical_start",
                                            "authoritative_invocation_digests": authority_full_digests,
                                            "authoritative_candidate_invocation_digests": (
                                                authoritative_candidate_invocation_digests
                                            ),
                                            "candidates": resolution_rows,
                                            "emitted_candidate_count": len(resolution_rows),
                                            "admitted_candidate_count": sum(
                                                row["admitted"] for row in resolution_rows
                                            ),
                                            "broker_accepted_candidate_count": sum(
                                                row["broker_accepted"]
                                                for row in resolution_rows
                                            ),
                                            "matched_emitted_candidate_count": sum(
                                                row["matched_authority"]
                                                for row in resolution_rows
                                            ),
                                            "matched_admitted_candidate_count": sum(
                                                row["admitted"] and row["matched_authority"]
                                                for row in resolution_rows
                                            ),
                                            "matched_broker_accepted_candidate_count": sum(
                                                row["broker_accepted"]
                                                and row["matched_authority"]
                                                for row in resolution_rows
                                            ),
                                            "decision_hit": any(
                                                row["matched_authority"]
                                                for row in resolution_rows
                                            ),
                                        }
                                    )
                        group_exposed = 0.0
                        last_name = state.last_completed_tool_name
                        last_event_index = state.last_completed_event_index
                        for descriptor, revealed_seq, revealed_at in revealed_tools:
                            duration_evidence = _causal_tool_duration_evidence(
                                duration_predictor=duration,
                                service_clock=loaded["service_clock"],
                                descriptor=descriptor,
                            )
                            observation = await executor.execute_authoritative(
                                session_id=session_id, descriptor=descriptor
                            )
                            completed_at = time.monotonic()
                            causal_seq += 1
                            tool_completed_seq = causal_seq
                            group_exposed += observation.exposed_wait_s
                            task_tool_exposed_s += observation.exposed_wait_s
                            task_saved_s += observation.saved_service_s
                            last_name = observation.tool_name
                            last_event_index = observation.event_index
                            async with result_lock:
                                tool_events.append(
                                    {
                                        "trace_id": session_id,
                                        "source_session_id": trace["source_session_id"],
                                        "request_index": request_index,
                                        "outcome_id": descriptor["outcome_id"],
                                        "authority_invocation_digest": _tool_invocation_digest(
                                            str(descriptor["tool_name"]),
                                            descriptor.get("tool_args", {}),
                                        ),
                                        "authority_candidate_invocation_digests": sorted(
                                            _atomic_visit_digests(descriptor)
                                        ),
                                        **duration_evidence,
                                        "llm_completed_seq": llm_completed_seq,
                                        "llm_completed_at_monotonic_s": llm_finished,
                                        "authoritative_revealed_seq": revealed_seq,
                                        "authoritative_revealed_at_monotonic_s": revealed_at,
                                        "tool_completed_seq": tool_completed_seq,
                                        "tool_completed_at_monotonic_s": completed_at,
                                        **serialize_observation(observation),
                                    }
                                )
                        if tools:
                            state.observe_tool_group(
                                tool_name=last_name,
                                event_index=last_event_index,
                                exposed_wait_s=group_exposed,
                            )
                        cursor.advance()
        except Exception as exc:
            failure = repr(exc)
        finally:
            await executor.close_session(session_id)
            finished = time.monotonic()
            timing_evidence = _task_timing_evidence(
                experiment_started_monotonic_s=experiment_started,
                release_offset_s=release,
                released_at_monotonic_s=released_at,
                gate_acquired_at_monotonic_s=gate_acquired,
                task_terminal_monotonic_s=finished,
            )
            async with result_lock:
                task_rows.append(
                    {
                        "trace_id": session_id,
                        "source_session_id": trace["source_session_id"],
                        "source_root_index": trace["source_root_index"],
                        **timing_evidence,
                        "llm_s": task_llm_s,
                        "prediction_overhead_s": task_prediction_s,
                        "tool_exposed_s": task_tool_exposed_s,
                        "saved_tool_service_s": task_saved_s,
                        "completion_tokens": completion_tokens,
                        "failure": failure,
                    }
                )

    await asyncio.gather(*(execute_trace(trace) for trace in loaded["public"]["traces"]))
    experiment_finished = time.monotonic()
    if not math.isfinite(experiment_started) or not math.isfinite(experiment_finished):
        raise RuntimeError("experiment monotonic bounds are non-finite")
    if experiment_finished < experiment_started:
        raise RuntimeError("experiment monotonic end precedes start")
    for row in task_rows:
        if not (
            experiment_started
            <= float(row["scheduled_release_monotonic_s"])
            <= float(row["released_at_monotonic_s"])
            <= float(row["task_terminal_monotonic_s"])
            <= experiment_finished
        ):
            raise RuntimeError("task timing evidence falls outside experiment bounds")
        if not math.isclose(
            float(row["flow_s"]),
            float(row["task_terminal_monotonic_s"])
            - float(row["scheduled_release_monotonic_s"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("task flow does not equal terminal minus scheduled release")
    broker_snapshot = executor.snapshot()
    await executor.close()
    cell_ended_wall_s = time.time()
    speculation_execution_events = aggregate_speculation_execution_events(
        speculative_job_transitions
    )
    end_file_hashes = {
        "runner": file_sha256(SCRIPT),
        "policy_bundle": file_sha256(args.bundle.resolve()),
        "config": file_sha256(args.config_file.resolve()),
        "scheduler_hook": file_sha256(args.scheduler_hook_file.resolve()),
        "smoke_evidence": file_sha256(args.smoke_evidence_file.resolve()),
        "runtime_environment_evidence": file_sha256(
            args.runtime_environment_evidence_file.resolve()
        ),
        "scheduler_runtime_evidence": file_sha256(
            args.scheduler_runtime_evidence_file.resolve()
        ),
    }
    if end_file_hashes != startup_file_hashes:
        raise RuntimeError("a frozen runtime artifact changed during the cell")
    if validate_scheduler_runtime_evidence(
        args,
        cell=cell,
        server_pid=server_pid,
    ) != scheduler_runtime_evidence:
        raise RuntimeError("scheduler runtime evidence changed during the cell")
    end_model_inventory = _model_snapshot_inventory(
        Path(str(bundle["model_snapshot_contract"]["path"]))
    )
    if end_model_inventory["inventory_sha256"] != startup_model_inventory_sha256:
        raise RuntimeError("the pinned model snapshot changed during the cell")
    flows = [float(row["flow_s"]) for row in task_rows]
    latencies = [float(row["latency_s"]) for row in request_events]
    failures = sum(row["failure"] is not None for row in task_rows)
    physical_speculative_starts = int(
        broker_snapshot["metrics"].get("physical_speculative_starts", 0)
    )
    evidenced_physical_starts = sum(
        row["physical_started_at_monotonic_s"] is not None
        for row in speculation_execution_events
    )
    if evidenced_physical_starts != physical_speculative_starts:
        raise RuntimeError(
            "speculative start evidence/metric mismatch: "
            f"{evidenced_physical_starts} != {physical_speculative_starts}"
        )
    speculation_causal_timing = validate_speculation_causal_timing(
        prediction_decisions=prediction_decisions,
        request_events=request_events,
        speculation_execution_events=speculation_execution_events,
    )
    ledger_speculative_s = sum(
        float(row["speculative_resource_s"])
        for row in speculation_execution_events
    )
    ledger_promoted_demand_s = sum(
        float(row["demand_resource_s"])
        for row in speculation_execution_events
    )
    if abs(ledger_speculative_s - float(broker_snapshot["speculative_resource_s"])) > 1e-6:
        raise RuntimeError("speculative execution ledger differs from broker accounting")
    if abs(
        ledger_promoted_demand_s
        - float(broker_snapshot["promoted_demand_resource_s"])
    ) > 1e-6:
        raise RuntimeError("promoted-demand ledger differs from broker accounting")
    worker_resource_accounting = {
        "speculative_resource_s": float(broker_snapshot["speculative_resource_s"]),
        "promoted_demand_resource_s": float(
            broker_snapshot["promoted_demand_resource_s"]
        ),
        "direct_demand_resource_s": float(
            broker_snapshot["direct_demand_resource_s"]
        ),
        "total_worker_occupancy_s": float(
            broker_snapshot["total_worker_service_s"]
        ),
    }
    if abs(
        worker_resource_accounting["total_worker_occupancy_s"]
        - worker_resource_accounting["speculative_resource_s"]
        - worker_resource_accounting["promoted_demand_resource_s"]
        - worker_resource_accounting["direct_demand_resource_s"]
    ) > 1e-6:
        raise RuntimeError("broker worker-resource accounting does not close")
    direct_from_tool_events_s = 0.0
    for event in tool_events:
        if str(event["tool_name"]) != "visit":
            direct_from_tool_events_s += float(event["service_s"])
            continue
        direct_from_tool_events_s += sum(
            float(result["service_s"])
            for result in event.get("visit_results", [])
            if result.get("source") == "executed"
        )
    if abs(
        direct_from_tool_events_s
        - worker_resource_accounting["direct_demand_resource_s"]
    ) > 1e-6:
        raise RuntimeError(
            "direct-demand accounting differs from raw authoritative tool events"
        )
    broker_drained = not broker_snapshot["running"] and not broker_snapshot["cached"]
    prediction_metrics = prediction_metrics_from_raw_evidence(
        prediction_outcomes=prediction_outcomes,
        tool_events=tool_events,
        speculation_execution_events=speculation_execution_events,
    )
    duration_errors = [
        float(row["duration_prediction_absolute_error_s"]) for row in tool_events
    ]
    duration_prediction_metrics = {
        "authoritative_tool_calls": len(duration_errors),
        "mean_absolute_error_s": (
            statistics.fmean(duration_errors) if duration_errors else None
        ),
    }
    artifact_paths = loaded["artifact_paths"]
    frozen_artifacts = {
        "runner": {"file_sha256": startup_file_hashes["runner"]},
        "policy_bundle": {
            "file_sha256": startup_file_hashes["policy_bundle"],
            "identity_sha256": bundle["bundle_sha256"],
        },
        "config": {"file_sha256": startup_file_hashes["config"]},
        "scheduler_hook": {
            "file_sha256": startup_file_hashes["scheduler_hook"]
        },
        "invocation_predictor": {
            "file_sha256": file_sha256(artifact_paths["invocation_predictor"]),
            "identity_sha256": bundle["mapper_artifact_sha256"],
        },
        "duration_predictor": {
            "file_sha256": file_sha256(artifact_paths["duration_predictor"]),
            "identity_sha256": bundle["duration_predictor_artifact_sha256"],
        },
        "service_clock": {
            "file_sha256": file_sha256(artifact_paths["service_clock"]),
            "identity_sha256": bundle["service_clock_artifact_sha256"],
        },
        "runtime_parameters": {
            "file_sha256": file_sha256(artifact_paths["runtime_parameters"]),
            "identity_sha256": runtime_parameters[
                "runtime_parameters_sha256"
            ],
        },
    }
    provenance = {
        "runner_file_sha256": frozen_artifacts["runner"]["file_sha256"],
        "policy_bundle_file_sha256": frozen_artifacts["policy_bundle"][
            "file_sha256"
        ],
        "config_file_sha256": frozen_artifacts["config"]["file_sha256"],
        "scheduler_hook_file_sha256": frozen_artifacts["scheduler_hook"][
            "file_sha256"
        ],
        "invocation_predictor_file_sha256": frozen_artifacts[
            "invocation_predictor"
        ]["file_sha256"],
        "invocation_predictor_artifact_sha256": frozen_artifacts[
            "invocation_predictor"
        ]["identity_sha256"],
        "duration_predictor_file_sha256": frozen_artifacts[
            "duration_predictor"
        ]["file_sha256"],
        "duration_predictor_artifact_sha256": frozen_artifacts[
            "duration_predictor"
        ]["identity_sha256"],
        "service_clock_file_sha256": frozen_artifacts["service_clock"][
            "file_sha256"
        ],
        "service_clock_artifact_sha256": frozen_artifacts["service_clock"][
            "identity_sha256"
        ],
        "runtime_parameters_file_sha256": frozen_artifacts[
            "runtime_parameters"
        ]["file_sha256"],
        "runtime_parameters_artifact_sha256": frozen_artifacts[
            "runtime_parameters"
        ]["identity_sha256"],
        "model_snapshot_inventory_sha256": startup_model_inventory_sha256,
    }
    summary = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": bundle["model_revision"],
        "runtime_parameters": runtime_parameters,
        "block_id": args.block_id,
        "order_position": args.order_position,
        "started_wall_s": cell_started_wall_s,
        "ended_wall_s": cell_ended_wall_s,
        "gpu_ids": gpu_ids,
        "server_instance_id": args.server_instance_id,
        "server_pid": server_pid,
        "broker_instance_id": broker_instance_id,
        "broker_pid": os.getpid(),
        "cache_state_contract": {
            "fresh_server_process": True,
            "standardized_smoke_warmed_prefix_state": True,
            "evaluation_workload_cache_empty_before_cell": True,
            "broker_result_cache_empty_before_cell": True,
            "smoke_evidence_sha256": startup_file_hashes["smoke_evidence"],
        },
        "runtime_environment_contract": {
            "evidence_sha256": startup_file_hashes[
                "runtime_environment_evidence"
            ],
            "model_snapshot_inventory_sha256": startup_model_inventory_sha256,
            "environment_scrubbed_before_cell": True,
            "server_and_client_launched_via_env_i": True,
        },
        "scheduler_runtime_contract": {
            "evidence_sha256": startup_file_hashes[
                "scheduler_runtime_evidence"
            ],
            "hook_runtime_use_expected": scheduler_runtime_evidence[
                "hook_runtime_use_expected"
            ],
            "patched_scheduler_invocation_verified": scheduler_runtime_evidence[
                "patched_scheduler_invocation_verified"
            ],
            "no_scheduler_hook_runtime_use_verified": scheduler_runtime_evidence[
                "no_scheduler_hook_runtime_use_verified"
            ],
            "expected_policy": scheduler_runtime_evidence["expected_policy"],
            "scheduler_calling_pid": scheduler_runtime_evidence[
                "scheduler_calling_pid"
            ],
            "scheduler_calling_process_relation": scheduler_runtime_evidence[
                "scheduler_calling_process_relation"
            ],
            "runtime_marker_sha256": scheduler_runtime_evidence[
                "runtime_marker_sha256"
            ],
        },
        "frozen_artifacts": frozen_artifacts,
        "provenance": provenance,
        "settings": {
            "scheduler": cell["scheduler"],
            "server_policy": cell["server_policy"],
            "tool_mechanism": "online_causal_speculation" if speculation else "demand_only",
            "call_graph_mode": CALL_GRAPH_MODE,
            "role": args.role,
            "public_output_cap": loaded["public"]["output_cap"],
            "max_active_tasks": args.max_active_tasks,
            "visit_capacity": args.visit_capacity,
            "configured_speculative_cap": args.speculative_cap,
            "effective_speculative_cap": args.speculative_cap if speculation else 0,
        },
        "paper_protocol": {
            "cell": args.cell,
            "scheduler": cell["scheduler"],
            "speculation": "online_causal" if speculation else "off",
            "offline_credit_s": 0,
            "all_tasks_successful": failures == 0,
            "broker_drained": broker_drained,
            "physical_speculative_starts": physical_speculative_starts,
            "authoritative_calls_revealed_after_live_llm": True,
            "prediction_sealed_before_authoritative_reveal": True,
            "authoritative_call_hidden_until_live_llm_completion": True,
            "call_graph_mode": CALL_GRAPH_MODE,
            "claim_type": "systems_trace_replay",
            "claim_scope": claim_scope,
            "physical_service_clock_mode": "calibration_hashed_empirical_v1",
            "service_assignment_policy_independent": True,
            "service_assignment_future_poison_invariant": True,
            "same_invocation_service_clock_all_cells": True,
            "evaluation_trace_duration_role": "diagnostic_only",
            "service_clock_artifact_sha256": loaded["sealed"][
                "service_clock_artifact_sha256"
            ],
            "uniform_public_llm_budget": True,
            "min_tokens_equals_max_tokens": True,
            "ignore_eos": True,
            "future_state_accepted_poison_invariance_test_passed": True,
        },
        "bundle_sha256": bundle["bundle_sha256"],
        "policy_bundle_file_sha256": startup_file_hashes["policy_bundle"],
        "public_plan_sha256": loaded["public"]["plan_sha256"],
        "sealed_plan_sha256": loaded["sealed"]["sealed_sha256"],
        "mapper_artifact_sha256": bundle["mapper_artifact_sha256"],
        "invocation_predictor_artifact_sha256": bundle["mapper_artifact_sha256"],
        "duration_predictor_artifact_sha256": bundle["duration_predictor_artifact_sha256"],
        "config_file_sha256": startup_file_hashes["config"],
        "scheduler_hook_file_sha256": startup_file_hashes["scheduler_hook"],
        "tail_predictor_artifact_sha256": bundle["tail_predictor_artifact_sha256"],
        "selected_top_k": bundle["selected_top_k"],
        "tasks": len(task_rows),
        "source_roots": len({row["source_session_id"] for row in task_rows}),
        "requests": len(request_events),
        "failures": failures,
        "experiment_started_monotonic_s": experiment_started,
        "experiment_ended_monotonic_s": experiment_finished,
        "experiment_wall_s": experiment_finished - experiment_started,
        "mean_task_flow_s": statistics.fmean(flows) if flows else 0.0,
        "p50_task_flow_s": percentile(flows, 0.50),
        "p95_task_flow_s": percentile(flows, 0.95),
        "p99_task_flow_s": percentile(flows, 0.99),
        "max_task_flow_s": max(flows, default=0.0),
        "mean_llm_latency_s": statistics.fmean(latencies) if latencies else 0.0,
        "p95_llm_latency_s": percentile(latencies, 0.95),
        "prediction_overhead_s": sum(float(row["prediction_overhead_s"]) for row in task_rows),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in task_rows),
        "broker": broker_snapshot,
        "worker_resource_accounting": worker_resource_accounting,
        "duration_predictor_runtime": duration.snapshot(),
        "llm_events": request_events,
        "tool_events": tool_events,
        "prediction_decisions": prediction_decisions,
        "prediction_outcomes": prediction_outcomes,
        "prediction_metrics": prediction_metrics,
        "speculation_causal_timing": speculation_causal_timing,
        "duration_prediction_metrics": duration_prediction_metrics,
        "speculation_execution_events": speculation_execution_events,
        "task_results": task_rows,
    }
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "result.json", summary)
    write_json(output_dir / "request_events.json", request_events)
    write_json(output_dir / "tool_events.json", tool_events)
    write_json(output_dir / "prediction_decisions.json", prediction_decisions)
    write_json(output_dir / "prediction_outcomes.json", prediction_outcomes)
    write_json(
        output_dir / "speculation_execution_events.json",
        speculation_execution_events,
    )
    write_json(output_dir / "task_results.json", task_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="fit calibration-only artifacts and seal plans")
    prepare.add_argument("--fixed-bundle", type=Path, default=DEFAULT_FIXED_BUNDLE)
    prepare.add_argument("--execution-traces", type=Path, default=DEFAULT_EXECUTION_TRACES)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--formal-config", type=Path, default=DEFAULT_FORMAL_CONFIG
    )
    prepare.add_argument(
        "--scheduler-hook", type=Path, default=DEFAULT_SCHEDULER_HOOK
    )
    prepare.add_argument("--tokenizer", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    prepare.add_argument("--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    prepare.add_argument(
        "--model-revision", default="4b0ac5767427a55d08a254f0367e2934976598e0"
    )
    prepare.add_argument("--max-model-len", type=int, default=16384)
    prepare.add_argument("--output-cap", type=int, default=128)
    prepare.add_argument("--max-top-k", type=int, default=5)
    prepare.add_argument("--min-prediction-precision", type=float, default=0.40)
    prepare.add_argument("--duration-ewma-alpha", type=float, default=0.35)
    prepare.add_argument(
        "--reuse-service-clock",
        type=Path,
        help=(
            "reuse one complete private calibration-only service_clock.json "
            "after exact split, sample-pool, signature, and contract validation"
        ),
    )
    prepare.add_argument("--max-active-tasks", type=int, default=80)
    prepare.add_argument("--visit-capacity", type=int, default=128)
    prepare.add_argument("--speculative-cap", type=int, default=32)
    prepare.add_argument("--default-predicted-output-tokens", type=float, default=128.0)
    prepare.add_argument("--request-timeout-s", type=float, default=600.0)
    prepare.add_argument("--arrivals", type=Path, default=DEFAULT_ARRIVALS)
    prepare.add_argument("--sessions", type=int, default=80)
    prepare.add_argument(
        "--claim-scope",
        choices=["retrospective", "confirmatory"],
        default="retrospective",
    )

    run = subparsers.add_parser("run-cell", help="run one immutable A/B/E/F cell")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--role", choices=["tuning", "final"], required=True)
    run.add_argument("--cell", choices=sorted(CELL_SPECS), required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--server-url", default="http://127.0.0.1:8100")
    run.add_argument("--server-policy-file", type=Path, required=True)
    run.add_argument("--server-pid-file", type=Path, required=True)
    run.add_argument("--server-instance-id", required=True)
    run.add_argument("--block-id", required=True)
    run.add_argument("--order-position", type=int, required=True)
    run.add_argument("--gpu-ids", required=True)
    run.add_argument("--config-file", type=Path, required=True)
    run.add_argument("--config-file-sha256", required=True)
    run.add_argument("--scheduler-hook-file", type=Path, required=True)
    run.add_argument("--scheduler-hook-file-sha256", required=True)
    run.add_argument("--smoke-evidence-file", type=Path, required=True)
    run.add_argument("--smoke-evidence-sha256", required=True)
    run.add_argument(
        "--runtime-environment-evidence-file", type=Path, required=True
    )
    run.add_argument("--runtime-environment-evidence-sha256", required=True)
    run.add_argument(
        "--scheduler-runtime-evidence-file", type=Path, required=True
    )
    run.add_argument("--scheduler-runtime-evidence-sha256", required=True)
    run.add_argument("--scheduler-runtime-marker-file", type=Path, required=True)
    run.add_argument("--model", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    run.add_argument("--max-active-tasks", type=int, default=80)
    run.add_argument("--visit-capacity", type=int, default=128)
    run.add_argument("--speculative-cap", type=int, default=32)
    run.add_argument("--default-predicted-output-tokens", type=float, default=128.0)
    run.add_argument("--request-timeout-s", type=float, default=600.0)
    run.add_argument(
        "--claim-scope", choices=["retrospective", "confirmatory"]
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "prepare":
        if args.max_model_len <= 1 or not 0 < args.output_cap < args.max_model_len:
            raise ValueError("output cap must be positive and smaller than max model length")
        if args.sessions is not None and args.sessions <= 0:
            raise ValueError("sessions must be positive")
        if not 0.0 < args.duration_ewma_alpha <= 1.0:
            raise ValueError("duration EWMA alpha must be in (0,1]")
        if min(args.max_active_tasks, args.visit_capacity) <= 0:
            raise ValueError("runtime capacities must be positive")
        if not 0 <= args.speculative_cap <= args.visit_capacity:
            raise ValueError("speculative cap must be in [0, visit capacity]")
        if args.default_predicted_output_tokens <= 0 or args.request_timeout_s <= 0:
            raise ValueError("prediction and timeout values must be positive")
    else:
        if min(args.max_active_tasks, args.visit_capacity) <= 0:
            raise ValueError("runtime capacities must be positive")
        if not 0 <= args.speculative_cap <= args.visit_capacity:
            raise ValueError("speculative cap must be in [0, visit capacity]")
        if args.default_predicted_output_tokens <= 0 or args.request_timeout_s <= 0:
            raise ValueError("prediction and timeout values must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    if args.command == "prepare":
        result = prepare_bundle(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = asyncio.run(run_cell(args))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "settings", "bundle_sha256", "tasks", "source_roots", "requests",
                    "failures", "experiment_wall_s", "mean_task_flow_s", "p95_task_flow_s"
                )
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
