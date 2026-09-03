#!/usr/bin/env python3
"""Run one fail-closed, development-only cross-model A/E portability pair.

This protocol is intentionally post-hoc and one-shot.  It transfers the
registered Qwen scheduler knobs *without calibration* to one explicitly named
local model snapshot, then runs FCFS A followed by physical-KV Joint-v2 E on
the already-observed 80-source formal-v8 workload.  The default profile keeps
the registered c12k/l80 shape.  A separately named c5k/l80 profile is an
explicit pre-live, cross-architecture compatibility fallback; it is not
comparable to c12k and must never be selected from a live A/E outcome.  All
profiles are descriptive stress evidence, not formal evidence, a model
comparison, a cross-GPU result, or a source of confidence intervals.

``--check-only`` is the required first step.  It hashes every local snapshot
file, loads the tokenizer offline, renders all three agent turns for all 80
sources, compiles all 80 fixed-final grammars, and checks deterministic prompts
plus declared response stress probes against 16k; live response-derived text
remains runtime-gated.  It prints the exact immutable plan and neither creates
output nor probes a port, GPU, server, or network endpoint.  A real execution additionally needs
``--execute-one-shot`` plus the exact ``preflight_plan_sha256`` printed by
``--check-only``, and permanently consumes the content-addressed attempt key
even if a cell fails; there is no automatic retry or outcome-driven rerun.

The runner accepts Mistral-7B-Instruct-v0.3 or any other explicit local
snapshot, but compatibility is never assumed.  In particular, a tokenizer
whose native chat template rejects the repository's unchanged message roles,
or whose fixed grammar/static stress probes are incompatible, fails closed in
preflight.  A live title/result that exceeds the remaining context fails the
80/240/160 runtime gates, consumes the one-shot attempt, and yields no summary.
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
import statistics
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence


# Make every Hugging Face/Transformers operation in this process offline.  The
# tokenizer calls below also pass local_files_only=True; these variables cover
# model-specific local code that may delegate back into the hub libraries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproduction/scripts"
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
for import_root in (SCRIPTS_DIR, REPRODUCTION_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_live_joint_formal_matrix as formal  # type: ignore  # noqa: E402
import run_scheduler_live_sensitivity as live  # type: ignore  # noqa: E402


PROTOCOL_VERSION = "cross-model-ae-portability-one-shot-v1"
BASE_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_formal_v8_matrix.env.example"
)
WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v8.json"
)
RUN_BASE = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/comment3_cross_model"
)
ATTEMPT_BASE = RUN_BASE / ".one_shot_attempts"
LIVE_SENSITIVITY_RUNNER = (
    REPOSITORY_ROOT
    / "reproduction/scripts/run_scheduler_live_sensitivity.py"
)

MODEL_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
RUN_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

MAX_ACTIVE_TASKS = 80
PHYSICAL_KV_TARGET = 0.93
MAX_MODEL_LEN = 16_384
TENSOR_PARALLEL_SIZE = 4
DTYPE = "bfloat16"
VISIT_MIN_START_INTERVAL_S = 3.0
EXPECTED_SOURCE_COUNT = 80
EXPECTED_LLM_REQUESTS = 240
EXPECTED_TOOL_COMMITS = 160
REGISTERED_FORECAST_MARGIN_TOKENS = 512
SEARCH_TITLE_PREFLIGHT_CHARS = 256
DEFAULT_PORT = 8200
RESERVED_ACTIVE_PORTS = frozenset({8000, 8100})


class CrossModelProtocolError(RuntimeError):
    """Fail-closed error for the cross-model portability protocol."""


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    profile_role: str
    context_padding_tokens: int
    max_active_tasks: int
    baseline_label: str
    candidate_label: str
    cross_architecture_fallback: bool


@dataclass(frozen=True)
class CellSpec:
    label: str
    cell: str
    role: str
    context_padding_tokens: int
    max_active_tasks: int = MAX_ACTIVE_TASKS
    physical_kv_target: float = PHYSICAL_KV_TARGET


DEFAULT_PROFILE_ID = "c12k-l80-primary"
CROSS_ARCHITECTURE_FALLBACK_PROFILE_ID = "c5k-l80-cross-architecture-fallback"
PROFILES = {
    DEFAULT_PROFILE_ID: ProfileSpec(
        profile_id=DEFAULT_PROFILE_ID,
        profile_role="registered_qwen_shape_primary",
        context_padding_tokens=12_000,
        max_active_tasks=MAX_ACTIVE_TASKS,
        baseline_label="a-c12k-l80-cross-model",
        candidate_label="e-c12k-l80-u093-cross-model",
        cross_architecture_fallback=False,
    ),
    CROSS_ARCHITECTURE_FALLBACK_PROFILE_ID: ProfileSpec(
        profile_id=CROSS_ARCHITECTURE_FALLBACK_PROFILE_ID,
        profile_role="pre_live_cross_architecture_compatibility_fallback",
        context_padding_tokens=5_000,
        max_active_tasks=MAX_ACTIVE_TASKS,
        baseline_label="a-c5k-l80-cross-architecture-fallback",
        candidate_label="e-c5k-l80-u093-cross-architecture-fallback",
        cross_architecture_fallback=True,
    ),
}
DEFAULT_PROFILE = PROFILES[DEFAULT_PROFILE_ID]


def _cells(profile: ProfileSpec) -> tuple[CellSpec, CellSpec]:
    return (
        CellSpec(
            profile.baseline_label,
            "A",
            "fcfs_reference",
            context_padding_tokens=profile.context_padding_tokens,
            max_active_tasks=profile.max_active_tasks,
        ),
        CellSpec(
            profile.candidate_label,
            "E",
            "joint_uncalibrated_transfer",
            context_padding_tokens=profile.context_padding_tokens,
            max_active_tasks=profile.max_active_tasks,
        ),
    )


# Backwards-compatible module constant for callers inspecting the default
# registered shape.  Execution code always derives cells from the selected
# immutable profile.
CELLS = _cells(DEFAULT_PROFILE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _profile_record(profile: ProfileSpec) -> dict[str, Any]:
    return {
        **asdict(profile),
        "selected_and_hashed_before_any_live_execution": True,
        "profile_switch_after_any_execution_attempt_allowed": False,
        "cross_profile_comparison_or_pooling_allowed": False,
        "context_shape_is_not_scheduler_recalibration": True,
    }


def _validate_profile(profile: ProfileSpec) -> None:
    if (
        not isinstance(profile, ProfileSpec)
        or PROFILES.get(profile.profile_id) != profile
    ):
        raise CrossModelProtocolError("selected portability profile is not registered")


def _profile_from_record(value: Any) -> ProfileSpec:
    if not isinstance(value, Mapping):
        raise CrossModelProtocolError("artifact lacks a portability profile")
    for profile in PROFILES.values():
        if dict(value) == _profile_record(profile):
            return profile
    raise CrossModelProtocolError("artifact portability profile is invalid")


def _run_root(run_tag: str, profile: ProfileSpec) -> Path:
    # Preserve the legacy c12k artifact path exactly.  Only the new fallback is
    # namespaced, so an old c12k reservation/report can never be reopened or
    # mistaken for the new c5k protocol shape.
    if profile == DEFAULT_PROFILE:
        return RUN_BASE / run_tag
    return RUN_BASE / profile.profile_id / run_tag


def _attempt_root(attempt_key: str, profile: ProfileSpec) -> Path:
    if profile == DEFAULT_PROFILE:
        return ATTEMPT_BASE / attempt_key
    return ATTEMPT_BASE / profile.profile_id / attempt_key


def _block_id(run_tag: str, profile: ProfileSpec) -> str:
    if profile == DEFAULT_PROFILE:
        return f"{run_tag}-cross-model-ae"
    return f"{run_tag}-{profile.profile_id}-cross-model-ae"


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise CrossModelProtocolError(f"repository binding escaped root: {path}") from exc


def _validate_identity(model_id: str, revision: str, snapshot: Path) -> Path:
    if MODEL_ID_RE.fullmatch(model_id) is None:
        raise CrossModelProtocolError(
            "--model-id must be an exact namespace/name identifier"
        )
    if REVISION_RE.fullmatch(revision) is None:
        raise CrossModelProtocolError(
            "--model-revision must be exactly 40 lowercase hexadecimal characters"
        )
    if not snapshot.is_absolute():
        raise CrossModelProtocolError("--model-snapshot must be an absolute local path")
    if not snapshot.exists():
        raise CrossModelProtocolError(f"local model snapshot is missing: {snapshot}")
    if not snapshot.is_dir():
        raise CrossModelProtocolError(f"local model snapshot is not a directory: {snapshot}")
    resolved = snapshot.resolve(strict=True)
    if resolved.name != revision:
        raise CrossModelProtocolError(
            "local snapshot directory basename must equal --model-revision"
        )
    return resolved


def _snapshot_category(relative: str) -> str:
    name = Path(relative).name.lower()
    if name == "config.json" or name.endswith("_config.json"):
        return "config"
    if name.endswith((".safetensors", ".bin", ".pt", ".pth")):
        return "weight"
    if name.endswith(".index.json") and (
        "model" in name or "weight" in name or "pytorch" in name
    ):
        return "weight_index"
    if (
        "tokenizer" in name
        or "special_tokens" in name
        or "added_tokens" in name
        or name in {
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "spiece.model",
            "sentencepiece.model",
            "tekken.json",
        }
    ):
        return "tokenizer"
    if name.endswith(".py"):
        return "local_code"
    return "other"


def _validate_weight_indexes(snapshot: Path, relative_files: set[str]) -> None:
    indexes = sorted(
        relative
        for relative in relative_files
        if _snapshot_category(relative) == "weight_index"
    )
    for relative in indexes:
        path = snapshot / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CrossModelProtocolError(f"invalid model weight index: {relative}") from exc
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise CrossModelProtocolError(f"weight index has no weight_map: {relative}")
        referenced = set()
        for tensor, shard in weight_map.items():
            if not isinstance(tensor, str) or not isinstance(shard, str):
                raise CrossModelProtocolError(
                    f"weight index contains non-string entries: {relative}"
                )
            shard_path = Path(shard)
            logical_path = Path(relative).parent / shard_path
            if shard_path.is_absolute() or ".." in logical_path.parts:
                raise CrossModelProtocolError(
                    f"weight index shard escapes snapshot: {relative}: {shard}"
                )
            logical = logical_path.as_posix()
            if logical not in relative_files:
                raise CrossModelProtocolError(
                    f"weight index references a missing shard: {relative}: {shard}"
                )
            referenced.add(logical)
        if not referenced:
            raise CrossModelProtocolError(f"weight index references no shards: {relative}")


def _snapshot_manifest(
    snapshot: Path, *, model_id: str, revision: str
) -> dict[str, Any]:
    """Hash every file in one already-local snapshot without following dirs."""

    if not snapshot.is_dir():
        raise CrossModelProtocolError(f"local model snapshot is missing: {snapshot}")
    entries: list[dict[str, Any]] = []
    broken_links: list[str] = []
    directory_links: list[str] = []
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(snapshot).as_posix()
        if path.is_symlink() and not path.exists():
            broken_links.append(relative)
            continue
        if path.is_symlink() and path.is_dir():
            directory_links.append(relative)
            continue
        if not path.is_file():
            continue
        entries.append(
            {
                "path": relative,
                "category": _snapshot_category(relative),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if broken_links:
        raise CrossModelProtocolError(
            f"snapshot contains broken file symlinks: {broken_links}"
        )
    if directory_links:
        raise CrossModelProtocolError(
            "snapshot contains directory symlinks that cannot be completely "
            f"content-addressed: {directory_links}"
        )
    if not entries:
        raise CrossModelProtocolError("local model snapshot contains no files")
    relatives = {entry["path"] for entry in entries}
    required = {"config.json", "tokenizer_config.json"}
    missing = sorted(required - relatives)
    if missing:
        raise CrossModelProtocolError(f"snapshot lacks required files: {missing}")
    tokenizer_payloads = {
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.model",
        "vocab.json",
        "vocab.txt",
        "tekken.json",
    }
    if not (relatives & tokenizer_payloads):
        raise CrossModelProtocolError(
            "snapshot lacks a recognized local tokenizer payload"
        )
    if not any(entry["category"] == "weight" for entry in entries):
        raise CrossModelProtocolError("snapshot lacks recognized local model weights")
    _validate_weight_indexes(snapshot, relatives)
    counts: dict[str, int] = {}
    bytes_by_category: dict[str, int] = {}
    for entry in entries:
        category = str(entry["category"])
        counts[category] = counts.get(category, 0) + 1
        bytes_by_category[category] = bytes_by_category.get(category, 0) + int(
            entry["size_bytes"]
        )
    core_without_content_hash = {
        "schema": "paste_repro.cross_model_snapshot_manifest",
        "version": 1,
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "category_counts": counts,
        "category_size_bytes": bytes_by_category,
        "files": entries,
    }
    content_identity = {
        key: value
        for key, value in core_without_content_hash.items()
        if key != "snapshot_path"
    }
    core = {
        **core_without_content_hash,
        "content_sha256": _sha256_json(content_identity),
    }
    return {**core, "manifest_sha256": _sha256_json(core)}


def _verify_snapshot_manifest(snapshot: Path, manifest: Mapping[str, Any]) -> None:
    recorded_hash = manifest.get("manifest_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(recorded_hash, str) or _sha256_json(core) != recorded_hash:
        raise CrossModelProtocolError("model snapshot manifest SHA is invalid")
    recorded_content_hash = core.get("content_sha256")
    content_identity = {
        key: value
        for key, value in core.items()
        if key not in {"snapshot_path", "content_sha256"}
    }
    if (
        not isinstance(recorded_content_hash, str)
        or _sha256_json(content_identity) != recorded_content_hash
    ):
        raise CrossModelProtocolError("model snapshot content SHA is invalid")
    if (
        not isinstance(core.get("model_id"), str)
        or MODEL_ID_RE.fullmatch(str(core["model_id"])) is None
        or core.get("revision") != snapshot.name
        or REVISION_RE.fullmatch(str(core.get("revision", ""))) is None
    ):
        raise CrossModelProtocolError("model snapshot manifest identity is invalid")
    if core.get("snapshot_path") != str(snapshot.resolve()):
        raise CrossModelProtocolError("model snapshot manifest path changed")
    files = core.get("files")
    if not isinstance(files, list) or not files:
        raise CrossModelProtocolError("model snapshot manifest has no file entries")
    observed_paths: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise CrossModelProtocolError(f"model snapshot entry {index} is malformed")
        relative = entry.get("path")
        expected_sha = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_sha, str)
            or SHA256_RE.fullmatch(expected_sha) is None
            or type(expected_size) is not int
            or expected_size < 0
            or relative in observed_paths
        ):
            raise CrossModelProtocolError(f"model snapshot entry {index} is invalid")
        observed_paths.add(relative)
        path = snapshot / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_sha
        ):
            raise CrossModelProtocolError(f"model snapshot file changed: {relative}")
    current_entries = list(snapshot.rglob("*"))
    unsupported_links = [
        path.relative_to(snapshot).as_posix()
        for path in current_entries
        if path.is_symlink() and (not path.exists() or path.is_dir())
    ]
    if unsupported_links:
        raise CrossModelProtocolError(
            f"model snapshot acquired unsupported symlinks: {unsupported_links}"
        )
    current_files = {
        path.relative_to(snapshot).as_posix()
        for path in current_entries
        if path.is_file()
    }
    if current_files != observed_paths:
        raise CrossModelProtocolError("model snapshot file set changed")


def _dependency_bindings() -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        BASE_CONFIG.resolve(),
        WORKLOAD.resolve(),
        LIVE_SENSITIVITY_RUNNER.resolve(),
        *(path.resolve() for path in formal.BOUND_CODE_PATHS),
    }
    bindings: dict[str, str] = {}
    for path in sorted(paths, key=str):
        if not path.is_file():
            raise CrossModelProtocolError(f"bound dependency is missing: {path}")
        bindings[_repository_relative(path)] = _sha256(path)
    return bindings


def _verify_bindings(bindings: Mapping[str, str]) -> None:
    for relative, expected in bindings.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or SHA256_RE.fullmatch(expected) is None
        ):
            raise CrossModelProtocolError("dependency binding is malformed")
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise CrossModelProtocolError(f"bound dependency changed: {relative}")


def _parse_gpus(raw: str) -> tuple[int, int, int, int]:
    parts = raw.split(",")
    if len(parts) != 4 or any(re.fullmatch(r"0|[1-9][0-9]*", part) is None for part in parts):
        raise CrossModelProtocolError(
            "--gpus must contain exactly four comma-separated physical GPU IDs"
        )
    values = tuple(int(part) for part in parts)
    if len(set(values)) != 4:
        raise CrossModelProtocolError("--gpus must contain four distinct GPU IDs")
    return values  # type: ignore[return-value]


def _validate_port(port: int) -> None:
    if not 1 <= port <= 65_535:
        raise CrossModelProtocolError("--port must be in 1..65535")
    if port in RESERVED_ACTIVE_PORTS:
        raise CrossModelProtocolError(
            f"port {port} is reserved for other reviewer experiments"
        )


def _derived_config(
    frozen: Mapping[str, str],
    *,
    model_id: str,
    revision: str,
    gpus: str,
    port: int,
    profile: ProfileSpec = DEFAULT_PROFILE,
) -> dict[str, str]:
    _validate_profile(profile)
    values = dict(frozen)
    values.update(
        {
            "MODEL_ID": model_id,
            "MODEL_REVISION": revision,
            "CUDA_VISIBLE_DEVICES": gpus,
            "VLLM_PORT": str(port),
            "VLLM_TP_SIZE": str(TENSOR_PARALLEL_SIZE),
            "VLLM_DTYPE": DTYPE,
            "VLLM_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "PASTE_LIVE_CONTEXT_PADDING_TOKENS": str(
                profile.context_padding_tokens
            ),
            "PASTE_LIVE_MAX_ACTIVE_TASKS": str(profile.max_active_tasks),
            "PASTE_LIVE_VISIT_MIN_START_INTERVAL_S": format(
                VISIT_MIN_START_INTERVAL_S, ".1f"
            ),
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": format(
                PHYSICAL_KV_TARGET, ".2f"
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    return values


def _environment_audit(config: Mapping[str, str]) -> dict[str, Any]:
    a_env = formal._cell_environment(config, cell="A", inherited={})
    e_env = formal._cell_environment(config, cell="E", inherited={})
    return _validate_pair_environments(a_env, e_env, config=config)


def _validate_pair_environments(
    a_env: Mapping[str, str],
    e_env: Mapping[str, str],
    *,
    config: Mapping[str, str],
) -> dict[str, Any]:
    a_scheduler = {
        key: value for key, value in a_env.items() if key.startswith("VLLM_SCHED_")
    }
    e_scheduler = {
        key: value for key, value in e_env.items() if key.startswith("VLLM_SCHED_")
    }
    if a_scheduler != {"VLLM_SCHED_POLICY": "fcfs"}:
        raise CrossModelProtocolError(
            f"FCFS baseline leaked scheduler treatment: {sorted(a_scheduler)}"
        )
    expected_e = {
        key: value for key, value in config.items() if key.startswith("VLLM_SCHED_")
    }
    expected_e["VLLM_SCHED_POLICY"] = "online_joint_pacer_v2"
    if e_scheduler != expected_e:
        changed = sorted(
            key
            for key in set(e_scheduler) | set(expected_e)
            if e_scheduler.get(key) != expected_e.get(key)
        )
        raise CrossModelProtocolError(
            f"Joint treatment differs from frozen Qwen knobs: {changed}"
        )
    a_common = {
        key: value for key, value in a_env.items() if not key.startswith("VLLM_SCHED_")
    }
    e_common = {
        key: value for key, value in e_env.items() if not key.startswith("VLLM_SCHED_")
    }
    if a_common != e_common:
        changed = sorted(
            key
            for key in set(a_common) | set(e_common)
            if a_common.get(key) != e_common.get(key)
        )
        raise CrossModelProtocolError(f"A/E common configuration differs: {changed}")
    if (
        e_scheduler.get("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION") != "1"
        or e_scheduler.get(
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"
        )
        != "0.93"
    ):
        raise CrossModelProtocolError("Joint physical-KV treatment is not target 0.93")
    return {
        "fixed_order": ["A", "E"],
        "common_config_identical": True,
        "baseline_scheduler_environment": a_scheduler,
        "candidate_scheduler_environment": e_scheduler,
        "only_scheduler_policy_and_registered_joint_controls_differ": True,
        "qwen_scheduler_knobs_transferred_without_recalibration": True,
        "physical_kv_target": PHYSICAL_KV_TARGET,
    }


class _TokenizerCounter:
    method = "transformers_chat_template_offline_preflight"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(token_ids, list) or not token_ids:
            raise CrossModelProtocolError("tokenizer produced no IDs for nonempty text")
        return len(token_ids)


def _render_chat(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    try:
        token_ids = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True
        )
    except Exception as exc:
        raise CrossModelProtocolError(
            "local tokenizer chat template rejects the unchanged agent message sequence"
        ) from exc
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int for token_id in token_ids)
    ):
        raise CrossModelProtocolError("chat template did not produce a flat token-ID list")
    return token_ids


def _offline_chat_context_preflight(
    snapshot: Path, *, profile: ProfileSpec = DEFAULT_PROFILE
) -> dict[str, Any]:
    _validate_profile(profile)
    try:
        from transformers import AutoTokenizer
        from paste_repro import live_agent
    except ImportError as exc:
        raise CrossModelProtocolError("offline tokenizer preflight dependencies are missing") from exc

    config_path = snapshot / "config.json"
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossModelProtocolError("local model config.json is invalid") from exc
    tokenizer_config_path = snapshot / "tokenizer_config.json"
    try:
        tokenizer_config = json.loads(
            tokenizer_config_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossModelProtocolError("local tokenizer_config.json is invalid") from exc
    if model_config.get("auto_map") or tokenizer_config.get("auto_map"):
        raise CrossModelProtocolError(
            "snapshot requires dynamic remote-code loading; this offline protocol "
            "accepts only built-in Transformers model/tokenizer classes"
        )
    model_type = model_config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise CrossModelProtocolError("local model config lacks a model_type")
    if profile.cross_architecture_fallback and model_type.lower().startswith("qwen"):
        raise CrossModelProtocolError(
            "cross-architecture fallback profile rejects Qwen-family architectures"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), trust_remote_code=False, local_files_only=True
        )
    except Exception as exc:
        raise CrossModelProtocolError("failed to load exact local tokenizer offline") from exc
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise CrossModelProtocolError("local tokenizer has no explicit chat template")
    maximum_positions = model_config.get("max_position_embeddings")
    if type(maximum_positions) is not int or maximum_positions < MAX_MODEL_LEN:
        raise CrossModelProtocolError(
            f"local model config does not support the required {MAX_MODEL_LEN}-token context"
        )

    try:
        payload = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossModelProtocolError("frozen workload is not valid JSON") from exc
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_SOURCE_COUNT:
        raise CrossModelProtocolError("chat preflight requires exactly 80 frozen sources")

    counter = _TokenizerCounter(tokenizer)
    visit_max_chars = int(formal.EXPECTED_CONFIG["PASTE_LIVE_VISIT_MAX_CHARS"])
    probe_patterns = {
        "alternating_ascii": ("x " * (visit_max_chars // 2 + 1))[
            :visit_max_chars
        ],
        "punctuation_ascii": (
            "!@#$%^&*()_+-=[]{};:,.<>/?|~" * visit_max_chars
        )[:visit_max_chars],
        "cjk": "漢" * visit_max_chars,
        "replacement_unicode": "�" * visit_max_chars,
        "astral_unicode": "😀" * visit_max_chars,
        "json_backslash_escape": "\\" * visit_max_chars,
        "json_quote_escape": '"' * visit_max_chars,
    }
    probe_counts = {
        name: counter.count_text(
            live_agent.canonical_json({"content": text})
        )
        for name, text in probe_patterns.items()
    }
    visit_probe_name = max(probe_counts, key=probe_counts.get)
    visit_probe = probe_patterns[visit_probe_name]
    visit_probe_tokens = probe_counts[visit_probe_name]
    truncated_suffix = "\n\n[Content truncated...]"
    visit_content_probe = visit_probe + truncated_suffix
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise CrossModelProtocolError(f"workload source {index} is malformed")
        source_id = source.get("source_id")
        question = source.get("question")
        search_query = source.get("search_query")
        expected_url = source.get("expected_url")
        if not all(isinstance(item, str) and item for item in (
            source_id,
            question,
            search_query,
            expected_url,
        )):
            raise CrossModelProtocolError(f"workload source {index} lacks prompt fields")
        context, context_actual = live_agent._unique_context_padding(
            token_counter=counter,
            task_id=f"{source_id}__r00",
            target_tokens=profile.context_padding_tokens,
        )
        question_message = (
            "TASK\n"
            f"RESEARCH_GOAL: {question}\n"
            f"SEARCH_QUERY: {search_query}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": live_agent.SYSTEM_PROMPT},
            {"role": "user", "content": context},
            {"role": "user", "content": question_message},
        ]
        phase0 = _render_chat(tokenizer, messages)
        search_call = {
            "name": "search",
            "arguments": {"query": [search_query]},
        }
        messages.append(
            {"role": "assistant", "content": live_agent.canonical_json(search_call)}
        )
        synthetic_results = {
            "tool": "search",
            "query": [search_query],
            "results": [
                {
                    "query": search_query,
                    "query_index": 0,
                    "rank": rank,
                    "title": (
                        f"Bounded frozen result {rank} "
                        + visit_probe[:SEARCH_TITLE_PREFLIGHT_CHARS]
                    ),
                    "url": expected_url,
                    "snippet": "",
                }
                for rank in range(1, 6)
            ],
            "_paste_transport": {
                "response_status": 200,
                "bytes_read": 512 * 1024,
                "backend": "bing_html_search",
                "request_host": "www.bing.com",
                "http_attempts": 1,
                "http_retries": 0,
                "http_attempt_log": [
                    {
                        "request_index": 0,
                        "attempt": 1,
                        "status": 200,
                        "error_type": None,
                        "retried": False,
                        "started_monotonic_s": 1.0,
                        "start_gate_wait_s": 0.0,
                        "retry_backoff_s": 0.0,
                    }
                ],
            },
        }
        messages.append(
            {
                "role": "user",
                "content": live_agent._tool_result_message(
                    "search", synthetic_results, goal=question
                ),
            }
        )
        phase1 = _render_chat(tokenizer, messages)
        visit_call = {
            "name": "visit",
            "arguments": {"url": [expected_url], "goal": question},
        }
        messages.append(
            {"role": "assistant", "content": live_agent.canonical_json(visit_call)}
        )
        messages.append(
            {
                "role": "user",
                "content": live_agent._tool_result_message(
                    "visit",
                    {
                        "tool": "visit",
                        "goal": question,
                        "pages": [
                            {
                                "url": expected_url,
                                "title": visit_probe[
                                    :SEARCH_TITLE_PREFLIGHT_CHARS
                                ],
                                "content": visit_content_probe,
                            }
                        ],
                        "_paste_transport": {
                            "response_status": 200,
                            "bytes_read": 512 * 1024,
                            "backend": "r.jina.ai",
                            "request_host": "r.jina.ai",
                            "http_attempts": 1,
                            "http_retries": 0,
                            "http_attempt_log": [
                                {
                                    "request_index": 0,
                                    "attempt": 1,
                                    "status": 200,
                                    "error_type": None,
                                    "retried": False,
                                    "started_monotonic_s": 4.0,
                                    "start_gate_wait_s": 0.0,
                                    "retry_backoff_s": 0.0,
                                }
                            ],
                        },
                    },
                    goal=question,
                ),
            }
        )
        phase2 = _render_chat(tokenizer, messages)
        phase_counts = [len(phase0), len(phase1), len(phase2)]
        completion_budgets = [
            int(formal.EXPECTED_CONFIG["PASTE_LIVE_MAX_TOKENS_TOOL"]),
            int(formal.EXPECTED_CONFIG["PASTE_LIVE_MAX_TOKENS_TOOL"]),
            formal.FIXED_FINAL_COMPLETION_TOKENS,
        ]
        remaining = [
            MAX_MODEL_LEN - prompt - completion
            for prompt, completion in zip(phase_counts, completion_budgets)
        ]
        if min(remaining) < 0:
            raise CrossModelProtocolError(
                f"{source_id} exceeds 16k tokenizer/chat context headroom"
            )
        rows.append(
            {
                "source_id": source_id,
                "context_padding_actual_tokens": context_actual,
                "visit_content_stress_probe_name": visit_probe_name,
                "visit_content_stress_probe_chars": len(visit_content_probe),
                "visit_content_stress_probe_raw_tokens": counter.count_text(
                    visit_content_probe
                ),
                "prompt_tokens_by_phase": phase_counts,
                "completion_budget_tokens_by_phase": completion_budgets,
                "remaining_tokens_by_phase": remaining,
                "message_sequence_rendered_unchanged": True,
            }
        )

    registered_forecast_required = (
        profile.context_padding_tokens
        + int(formal.EXPECTED_CONFIG["PASTE_LIVE_PREDICTED_VISIT_RESULT_TOKENS"])
        + formal.FIXED_FINAL_COMPLETION_TOKENS
        + REGISTERED_FORECAST_MARGIN_TOKENS
    )
    if registered_forecast_required > MAX_MODEL_LEN:
        raise CrossModelProtocolError("registered context forecast exceeds 16k")
    return {
        "schema": "paste_repro.cross_model_chat_context_preflight",
        "version": 1,
        "deterministic_probe_valid": True,
        "universal_live_context_fit_proven": False,
        "external_text_fit_is_runtime_only": True,
        "offline_only": True,
        "network_touched": False,
        "gpu_or_server_touched": False,
        "profile": _profile_record(profile),
        "source_count": len(rows),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "model_max_position_embeddings": maximum_positions,
        "served_max_model_len": MAX_MODEL_LEN,
        "context_padding_target_tokens": profile.context_padding_tokens,
        "visit_max_chars": visit_max_chars,
        "visit_truncation_suffix_chars": len(truncated_suffix),
        "visit_content_stress_probe_name": visit_probe_name,
        "visit_content_stress_probe_json_wrapped_tokens": visit_probe_tokens,
        "visit_content_stress_probe_candidate_json_wrapped_tokens": probe_counts,
        "search_title_preflight_chars_per_result": SEARCH_TITLE_PREFLIGHT_CHARS,
        "synthetic_tool_results_match_live_executor_shape": True,
        "rendered_visit_content_stress_probe_is_a_hard_gate": True,
        "registered_1600_token_forecast_is_not_the_only_headroom_gate": True,
        "titles_not_separately_length_capped": True,
        "titles_bounded_only_by_512k_response_body": True,
        "title_fit_within_16k_not_proven": True,
        "search_title_probe_is_stress_not_a_universal_external_text_bound": True,
        "registered_forecast_required_tokens": registered_forecast_required,
        "registered_forecast_remaining_tokens": (
            MAX_MODEL_LEN - registered_forecast_required
        ),
        "minimum_rendered_remaining_tokens": min(
            remaining for row in rows for remaining in row["remaining_tokens_by_phase"]
        ),
        "maximum_rendered_prompt_tokens": max(
            prompt for row in rows for prompt in row["prompt_tokens_by_phase"]
        ),
        "rows_sha256": _sha256_json(rows),
        "rows": rows,
    }


def _full_preflight(
    *,
    model_id: str,
    revision: str,
    snapshot: Path,
    gpus: str,
    port: int,
    snapshot_manifest: Mapping[str, Any],
    profile: ProfileSpec = DEFAULT_PROFILE,
) -> tuple[dict[str, str], Path, dict[str, Any]]:
    _validate_profile(profile)
    _parse_gpus(gpus)
    _validate_port(port)
    _verify_snapshot_manifest(snapshot, snapshot_manifest)
    try:
        frozen = formal.load_frozen_config(BASE_CONFIG)
    except formal.FormalRunError as exc:
        raise CrossModelProtocolError(str(exc)) from exc
    python = Path(frozen["PASTE_ENV_PREFIX"]) / "bin/python"
    if not python.is_file():
        raise CrossModelProtocolError(f"pinned environment Python is missing: {python}")
    if Path(sys.executable).resolve() != python.resolve():
        raise CrossModelProtocolError(
            f"run this protocol with the pinned Python: {python}"
        )
    try:
        formal.validate_entrypoints(python=python)
        workload_validation = formal.validate_formal_workload(
            python=python, workload=WORKLOAD
        )
    except formal.FormalRunError as exc:
        raise CrossModelProtocolError(str(exc)) from exc
    if _sha256(WORKLOAD) != formal.FORMAL_WORKLOAD_SHA256:
        raise CrossModelProtocolError("frozen 80-source workload SHA mismatch")

    chat_context = _offline_chat_context_preflight(snapshot, profile=profile)
    try:
        grammar = formal.validate_fixed_final_grammar_feasibility(
            workload=WORKLOAD,
            model_snapshot=snapshot,
            expected_source_count=EXPECTED_SOURCE_COUNT,
        )
    except formal.FormalRunError as exc:
        raise CrossModelProtocolError(str(exc)) from exc
    if (
        grammar.get("valid") is not True
        or grammar.get("source_count") != EXPECTED_SOURCE_COUNT
        or chat_context.get("source_count") != EXPECTED_SOURCE_COUNT
        or chat_context.get("deterministic_probe_valid") is not True
    ):
        raise CrossModelProtocolError("80-source tokenizer/grammar preflight is incomplete")
    # Tokenizer/config loading must not materialize or mutate files inside the
    # exact snapshot after its content-addressed manifest was frozen.
    _verify_snapshot_manifest(snapshot, snapshot_manifest)

    config = _derived_config(
        frozen,
        model_id=model_id,
        revision=revision,
        gpus=gpus,
        port=port,
        profile=profile,
    )
    if int(config["VLLM_MAX_NUM_SEQS"]) <= profile.max_active_tasks:
        raise CrossModelProtocolError(
            "offered load must remain strictly below native max-num-seqs"
        )
    environment_audit = _environment_audit(config)
    return config, python, {
        "profile": _profile_record(profile),
        "workload_validation": workload_validation,
        "snapshot_manifest_validation": {
            "valid": True,
            "file_count": snapshot_manifest["file_count"],
            "manifest_sha256": snapshot_manifest["manifest_sha256"],
            "content_sha256": snapshot_manifest["content_sha256"],
        },
        "tokenizer_chat_context": chat_context,
        "fixed_final_grammar": grammar,
        "a_e_environment": environment_audit,
        "hardware_validation": {
            "deferred_until_execution": True,
            "required_gpu_count": 4,
            "required_gpu_family": "NVIDIA A100",
            "required_memory_class": "40GB",
            "check_only_invoked_nvidia_smi": False,
        },
    }


def _runner_command(
    *,
    python: Path,
    snapshot: Path,
    output: Path,
    spec: CellSpec,
    block_id: str,
    order_index: int,
    server_instance_id: str,
    config: Mapping[str, str],
) -> list[str]:
    command = formal._runner_command(
        python=python,
        workload=WORKLOAD,
        output=output,
        cell=spec.cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        config=config,
    )
    try:
        tokenizer_index = command.index("--tokenizer") + 1
    except ValueError as exc:
        raise CrossModelProtocolError("bound live runner command lost --tokenizer") from exc
    command[tokenizer_index] = str(snapshot)
    if command.count("--tokenizer") != 1 or command[tokenizer_index] != str(snapshot):
        raise CrossModelProtocolError("explicit local tokenizer path was not isolated")
    return command


def _one_shot_key(
    *,
    model_id: str,
    revision: str,
    snapshot_manifest: Mapping[str, Any],
    config: Mapping[str, str],
    profile: ProfileSpec = DEFAULT_PROFILE,
) -> str:
    _validate_profile(profile)
    if (
        config.get("PASTE_LIVE_CONTEXT_PADDING_TOKENS")
        != str(profile.context_padding_tokens)
        or config.get("PASTE_LIVE_MAX_ACTIVE_TASKS")
        != str(profile.max_active_tasks)
    ):
        raise CrossModelProtocolError("selected profile and derived config differ")
    qwen_knobs = {
        key: value
        for key, value in config.items()
        if key.startswith("VLLM_SCHED_")
    }
    # Do not add, remove, or rename fields in the default branch: this is the
    # exact legacy c12k attempt-key payload.  The new fallback gets a distinct
    # explicit profile binding without reopening any legacy reservation.
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": model_id,
        "revision": revision,
        "snapshot_content_sha256": snapshot_manifest["content_sha256"],
        "workload_sha256": formal.FORMAL_WORKLOAD_SHA256,
        "fixed_order": ["A", "E"],
        "required_gpu_sku": "NVIDIA A100 40GB",
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "dtype": DTYPE,
        "max_model_len": MAX_MODEL_LEN,
        "context_padding_tokens": profile.context_padding_tokens,
        "max_active_tasks": profile.max_active_tasks,
        "physical_kv_target": PHYSICAL_KV_TARGET,
        "visit_min_start_interval_s": VISIT_MIN_START_INTERVAL_S,
        "qwen_scheduler_knobs": qwen_knobs,
    }
    if profile != DEFAULT_PROFILE:
        payload["profile"] = asdict(profile)
    return _sha256_json(payload)


def _plan(
    *,
    run_tag: str,
    model_id: str,
    revision: str,
    snapshot: Path,
    snapshot_manifest: Mapping[str, Any],
    config: Mapping[str, str],
    python: Path,
    gpus: str,
    port: int,
    preflight: Mapping[str, Any],
    bindings: Mapping[str, str],
    profile: ProfileSpec = DEFAULT_PROFILE,
) -> dict[str, Any]:
    _validate_profile(profile)
    run_root = _run_root(run_tag, profile)
    block_id = _block_id(run_tag, profile)
    selected_cells = _cells(profile)
    cells = []
    for index, spec in enumerate(selected_cells):
        placeholder_instance = f"runtime-uuid-{index + 1:02d}"
        cell_root = run_root / "cells" / f"{index + 1:02d}-{spec.label}"
        cells.append(
            {
                **asdict(spec),
                "order_index": index,
                "fresh_server": True,
                "server_state_reuse": False,
                "runner_command": _runner_command(
                    python=python,
                    snapshot=snapshot,
                    output=cell_root / "evidence",
                    spec=spec,
                    block_id=block_id,
                    order_index=index,
                    server_instance_id=placeholder_instance,
                    config=config,
                ),
            }
        )
    attempt_key = _one_shot_key(
        model_id=model_id,
        revision=revision,
        snapshot_manifest=snapshot_manifest,
        config=config,
        profile=profile,
    )
    core = {
        "schema": "paste_repro.scheduler_cross_model_portability_plan",
        "version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "run_tag": run_tag,
        "run_root": _repository_relative(run_root),
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "posthoc": True,
        "workload_already_observed_with_qwen": True,
        "fixed_order": ["A", "E"],
        "one_shot": True,
        "profile": _profile_record(profile),
        "attempt_key": attempt_key,
        "attempt_key_consumed_on_any_execution_attempt": True,
        "automatic_retry_allowed": False,
        "outcome_driven_parameter_change_or_rerun_allowed": False,
        "model": {
            "model_id": model_id,
            "revision": revision,
            "snapshot_path": str(snapshot),
            "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
            "snapshot_content_sha256": snapshot_manifest["content_sha256"],
            "snapshot_file_count": snapshot_manifest["file_count"],
        },
        "deployment": {
            "gpu_ids": list(_parse_gpus(gpus)),
            "required_gpu_sku": "NVIDIA A100 40GB; verified only at execution",
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "context_padding_tokens": profile.context_padding_tokens,
            "max_active_tasks": profile.max_active_tasks,
            "port": port,
        },
        "transport": {
            "visit_mode": "jina",
            "visit_min_start_interval_s": VISIT_MIN_START_INTERVAL_S,
            "zero_retries_required": True,
            "accepted_physical_http_attempts_per_tool_invocation": 1,
            "expected_authoritative_tool_commits": EXPECTED_TOOL_COMMITS,
        },
        "completion_gates": {
            "successful_tasks": EXPECTED_SOURCE_COUNT,
            "successful_llm_requests": EXPECTED_LLM_REQUESTS,
            "authoritative_tool_commits": EXPECTED_TOOL_COMMITS,
            "failed_tasks": 0,
            "http_retries": 0,
        },
        "scheduler_interpretation": {
            "registered_qwen_knobs_copied_byte_for_byte": True,
            "cross_model_recalibration_performed": False,
            "registered_qwen_context_shape_copied": (
                profile.profile_id == DEFAULT_PROFILE_ID
            ),
            "cross_architecture_fallback_profile": (
                profile.cross_architecture_fallback
            ),
            "candidate_is_deliberately_uncalibrated_stress": True,
            "baseline_has_no_scheduler_extension_leakage": True,
            "candidate_physical_kv_target": PHYSICAL_KV_TARGET,
        },
        "evidence_boundary": {
            "single_run_per_cell": True,
            "confidence_interval_available": False,
            "cross_gpu_generalization_proven": False,
            "cross_model_generalization_proven_by_plan_alone": False,
            "model_quality_comparison_allowed": False,
            "pooling_with_qwen_or_other_runs_allowed": False,
            "cross_profile_comparison_or_pooling_allowed": False,
            "profile_switch_after_any_execution_attempt_allowed": False,
            "fixed_order_confound_acknowledged": True,
            "context_preflight_covers_all_80_sources_and_three_phases": True,
            "context_preflight_is_not_a_universal_bound_on_external_text": True,
            "titles_not_separately_length_capped": True,
            "titles_bounded_only_by_512k_response_body": True,
            "title_fit_within_16k_not_proven": True,
            "runtime_context_overflow_invalidates_the_one_shot_pair": True,
        },
        "check_only_contract": {
            "output_created": False,
            "network_touched": False,
            "gpu_or_server_touched": False,
            "port_probed": False,
        },
        "preflight": dict(preflight),
        "dependency_bindings": dict(bindings),
        "cells": cells,
    }
    return {**core, "preflight_plan_sha256": _sha256_json(core)}


def _validate_transport(result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        audit = live._validate_transport_attempts(result)
    except live.LiveSensitivityError as exc:
        raise CrossModelProtocolError(str(exc)) from exc
    if (
        audit.get("tool_invocation_count") != EXPECTED_TOOL_COMMITS
        or audit.get("physical_http_attempt_count") != EXPECTED_TOOL_COMMITS
        or audit.get("http_retry_count") != 0
        or audit.get("http_429_count") != 0
        or audit.get("all_status_200") is not True
    ):
        raise CrossModelProtocolError("cross-model zero-retry transport gate failed")
    attempts = result.get("tool_attempt_records")
    if not isinstance(attempts, list):
        raise CrossModelProtocolError("transport attempt ledger is missing")
    tool_counts = {"search": 0, "visit": 0}
    visit_starts: list[float] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise CrossModelProtocolError(f"tool attempt {index} is malformed")
        tool = attempt.get("tool")
        if tool not in tool_counts:
            raise CrossModelProtocolError(
                f"tool attempt {index} has an unexpected tool identity"
            )
        tool_counts[str(tool)] += 1
        if attempt.get("transport_identity_source") != "actual":
            raise CrossModelProtocolError(
                f"tool attempt {index} lacks actual transport identity"
            )
        worker_pool = attempt.get("worker_pool")
        intervals = (
            worker_pool.get("tool_min_start_intervals_s")
            if isinstance(worker_pool, Mapping)
            else None
        )
        if (
            not isinstance(intervals, Mapping)
            or intervals.get("visit") != VISIT_MIN_START_INTERVAL_S
        ):
            raise CrossModelProtocolError(
                f"tool attempt {index} lacks the frozen 3.0s visit pacing"
            )
        if tool == "visit":
            log = attempt.get("http_attempt_log")
            start = (
                log[0].get("started_monotonic_s")
                if isinstance(log, list)
                and len(log) == 1
                and isinstance(log[0], Mapping)
                else None
            )
            if not isinstance(start, (int, float)) or isinstance(start, bool):
                raise CrossModelProtocolError(
                    f"visit attempt {index} lacks its physical HTTP start"
                )
            visit_starts.append(float(start))
    if tool_counts != {"search": 80, "visit": 80}:
        raise CrossModelProtocolError(
            f"transport tool identity counts are incomplete: {tool_counts}"
        )
    ordered_visit_starts = sorted(visit_starts)
    visit_gaps = [
        right - left
        for left, right in zip(ordered_visit_starts, ordered_visit_starts[1:])
    ]
    # The configured attempt gate is exactly 3.0 s.  The 20 ms measurement
    # tolerance is inherited from the strict post-hoc transport audit and is
    # only for monotonic timestamp sampling; configuration equality above is
    # exact and no HTTP retry is accepted.
    minimum_visit_gap = min(visit_gaps) if visit_gaps else None
    if minimum_visit_gap is None or minimum_visit_gap < 2.98:
        raise CrossModelProtocolError(
            f"observed visit HTTP pacing fell below 2.98s: {minimum_visit_gap}"
        )

    broker = result.get("broker_final_snapshot")
    stats = broker.get("stats") if isinstance(broker, Mapping) else None
    if not isinstance(stats, Mapping):
        raise CrossModelProtocolError("transport lacks final broker ledger")
    required_stats = {
        "authoritative_requests": EXPECTED_TOOL_COMMITS,
        "authoritative_started": EXPECTED_TOOL_COMMITS,
        "authoritative_completed": EXPECTED_TOOL_COMMITS,
        "authoritative_executions": EXPECTED_TOOL_COMMITS,
        "authoritative_failures": 0,
        "commits": EXPECTED_TOOL_COMMITS,
        "speculative_started": 0,
        "speculative_completed": 0,
        "speculative_failures": 0,
    }
    ledger_drift = {
        key: (expected, stats.get(key))
        for key, expected in required_stats.items()
        if stats.get(key) != expected
    }
    if ledger_drift:
        raise CrossModelProtocolError(
            f"transport broker ledger mismatch: {ledger_drift}"
        )
    return {
        **audit,
        "tool_identity_counts": tool_counts,
        "transport_identity_source": "actual",
        "minimum_observed_visit_http_start_gap_s": minimum_visit_gap,
        "configured_visit_min_start_interval_s": VISIT_MIN_START_INTERVAL_S,
        "broker_ledger_validated": True,
    }


def _physical_kv_log_summary(
    server_text: str, *, expected_target: float | None
) -> dict[str, Any]:
    physical_lines = [
        line
        for line in server_text.splitlines()
        if "[sched_policy_patch:physical_kv]" in line
    ]
    if expected_target is None:
        if physical_lines:
            raise CrossModelProtocolError(
                "FCFS cell unexpectedly emitted physical-KV telemetry"
            )
        return {
            "sample_count": 0,
            "target_utilization": None,
            "controller_was_active": False,
            "malformed_sample_count": 0,
            "fail_closed_count": 0,
        }
    if not physical_lines:
        raise CrossModelProtocolError("Joint cell lacks physical-KV telemetry")
    usage: list[float] = []
    malformed = 0
    fail_closed = 0
    wrong_target = 0
    for line in physical_lines:
        fields = dict(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))
        if fields.get("decision") == "fail_closed":
            fail_closed += 1
        try:
            target = float(fields["target_utilization"])
            current_usage = float(fields["usage"])
            waiting = int(fields["waiting"])
            fit_admit = int(fields["fit_admit"])
            admit = int(fields["admit"])
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        if not math.isclose(target, expected_target, rel_tol=0.0, abs_tol=1e-9):
            wrong_target += 1
        if not 0.0 <= current_usage <= 1.0 or min(waiting, fit_admit, admit) < 0:
            malformed += 1
            continue
        usage.append(current_usage)
    if malformed or fail_closed or wrong_target or len(usage) != len(physical_lines):
        raise CrossModelProtocolError(
            "Joint physical-KV telemetry failed closed: "
            f"samples={len(physical_lines)} malformed={malformed} "
            f"wrong_target={wrong_target} fail_closed={fail_closed}"
        )
    return {
        "sample_count": len(usage),
        "target_utilization": expected_target,
        "maximum_observed_usage": max(usage),
        "controller_was_active": True,
        "malformed_sample_count": 0,
        "fail_closed_count": 0,
    }


def _validate_recorded_environment(
    scheduler: Mapping[str, Any],
    *,
    spec: CellSpec,
    config: Mapping[str, str],
) -> dict[str, bool]:
    """Compare every recorded launch setting, not only policy and target."""

    expected_environment = formal._cell_environment(
        config, cell=spec.cell, inherited={}
    )
    expected_scheduler_map = {
        key: value
        for key, value in expected_environment.items()
        if key.startswith("VLLM_SCHED_")
    }
    observed_scheduler_map = {
        key: value
        for key, value in scheduler.items()
        if key.startswith("VLLM_SCHED_") and value is not None
    }
    if observed_scheduler_map != expected_scheduler_map:
        changed = sorted(
            key
            for key in set(observed_scheduler_map) | set(expected_scheduler_map)
            if observed_scheduler_map.get(key) != expected_scheduler_map.get(key)
        )
        raise CrossModelProtocolError(
            f"{spec.label} full scheduler environment drift: {changed}"
        )
    recorded_common_drift = {
        key: (config.get(key), value)
        for key, value in scheduler.items()
        if not key.startswith("VLLM_SCHED_")
        and (key not in config or value != config.get(key))
    }
    if recorded_common_drift:
        raise CrossModelProtocolError(
            f"{spec.label} full common environment drift: {recorded_common_drift}"
        )
    return {
        "full_scheduler_environment_validated": True,
        "full_recorded_common_environment_validated": True,
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
    server_log: Path,
    profile: ProfileSpec = DEFAULT_PROFILE,
) -> dict[str, Any]:
    _validate_profile(profile)
    if spec not in _cells(profile):
        raise CrossModelProtocolError("cell specification does not match profile")
    try:
        base_validation = live._validate_result(
            result,
            spec=spec,
            config=config,
            block_id=block_id,
            order_index=order_index,
            server_instance_id=server_instance_id,
            timeline_path=timeline_path,
        )
    except live.LiveSensitivityError as exc:
        raise CrossModelProtocolError(str(exc)) from exc
    result_config = result.get("config")
    scheduler = (
        result_config.get("scheduler_environment")
        if isinstance(result_config, Mapping)
        else None
    )
    if not isinstance(result_config, Mapping):
        raise CrossModelProtocolError(f"{spec.label} lacks result configuration")
    required_result_config = {
        "token_count_method": "transformers_chat_template",
        "fixed_final_completion_tokens": formal.FIXED_FINAL_COMPLETION_TOKENS,
        "fixed_final_completion_enabled": True,
        "max_tokens_tool": int(
            formal.EXPECTED_CONFIG["PASTE_LIVE_MAX_TOKENS_TOOL"]
        ),
        "visit_mode": "jina",
        "search_mode": "bing",
        "context_padding_tokens": spec.context_padding_tokens,
        "max_active_tasks": spec.max_active_tasks,
        "visit_min_start_interval_s": VISIT_MIN_START_INTERVAL_S,
    }
    result_drift = {
        key: (expected, result_config.get(key))
        for key, expected in required_result_config.items()
        if result_config.get(key) != expected
    }
    if result_drift:
        raise CrossModelProtocolError(
            f"{spec.label} tokenizer/workload runtime drift: {result_drift}"
        )
    if not isinstance(scheduler, Mapping):
        raise CrossModelProtocolError(f"{spec.label} lacks scheduler environment")
    environment_validation = _validate_recorded_environment(
        scheduler, spec=spec, config=config
    )
    expected_common = {
        "CUDA_VISIBLE_DEVICES": config["CUDA_VISIBLE_DEVICES"],
        "MODEL_ID": config["MODEL_ID"],
        "MODEL_REVISION": config["MODEL_REVISION"],
        "VLLM_PORT": config["VLLM_PORT"],
        "VLLM_TP_SIZE": str(TENSOR_PARALLEL_SIZE),
        "VLLM_DTYPE": DTYPE,
        "VLLM_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "VLLM_MAX_NUM_SEQS": config["VLLM_MAX_NUM_SEQS"],
    }
    drift = {
        key: (expected, scheduler.get(key))
        for key, expected in expected_common.items()
        if scheduler.get(key) != expected
    }
    if drift:
        raise CrossModelProtocolError(f"{spec.label} model/deployment drift: {drift}")
    transport = _validate_transport(result)
    physical = _physical_kv_log_summary(
        server_log.read_text(encoding="utf-8", errors="replace"),
        expected_target=None if spec.cell == "A" else spec.physical_kv_target,
    )
    return {
        **base_validation,
        **environment_validation,
        "profile": _profile_record(profile),
        "model_identity_validated": True,
        "deployment_shape_validated": True,
        "transport_validation": transport,
        "physical_kv_telemetry": physical,
    }


def _relative_file_manifest(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in paths:
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise CrossModelProtocolError(f"evidence path escaped cell root: {path}") from exc
        if not path.is_file():
            raise CrossModelProtocolError(f"evidence file is missing: {path}")
        values[relative] = _sha256(path)
    return values


def _verify_relative_file_manifest(
    root: Path, manifest: Mapping[str, Any]
) -> None:
    evidence = manifest.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise CrossModelProtocolError("evidence manifest is incomplete")
    for relative, expected in evidence.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or SHA256_RE.fullmatch(expected) is None
        ):
            raise CrossModelProtocolError("evidence manifest entry is malformed")
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise CrossModelProtocolError(f"evidence manifest SHA mismatch: {relative}")


def _run_cell(
    *,
    run_root: Path,
    spec: CellSpec,
    order_index: int,
    config: Mapping[str, str],
    python: Path,
    snapshot: Path,
    snapshot_manifest: Mapping[str, Any],
    bindings: Mapping[str, str],
    profile: ProfileSpec = DEFAULT_PROFILE,
) -> Path:
    _validate_profile(profile)
    if spec not in _cells(profile):
        raise CrossModelProtocolError("cell specification does not match profile")
    _verify_bindings(bindings)
    _verify_snapshot_manifest(snapshot, snapshot_manifest)
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
    block_id = _block_id(run_root.name, profile)
    environment = formal._cell_environment(config, cell=spec.cell)
    environment.update(
        {
            "VLLM_REQUIRE_NEW": "1",
            "VLLM_STATE_DIR": str(state_dir),
            "VLLM_LOG_DIR": str(server_dir),
            "VLLM_HOOK_DIR": str(REPOSITORY_ROOT / "scripts/pythonhooks"),
            "MODEL_SNAPSHOT": str(snapshot),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = _runner_command(
        python=python,
        snapshot=snapshot,
        output=evidence_dir,
        spec=spec,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        config=config,
    )
    contract = {
        "schema": "paste_repro.scheduler_cross_model_cell_contract",
        "version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "profile": _profile_record(profile),
        "development_only": True,
        "formal_eligible": False,
        "spec": asdict(spec),
        "order_index": order_index,
        "block_id": block_id,
        "server_instance_id": server_instance_id,
        "fresh_server_required": True,
        "result_cache_empty_required": True,
        "one_shot_no_retry": True,
        "model": {
            "model_id": config["MODEL_ID"],
            "revision": config["MODEL_REVISION"],
            "snapshot_path": str(snapshot),
            "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        },
        "transport": {
            "visit_min_start_interval_s": VISIT_MIN_START_INTERVAL_S,
            "zero_retries_required": True,
        },
        "runner_command": command,
        "dependency_bindings": dict(bindings),
    }
    _write_json_atomic(cell_root / "cell_contract.json", contract)
    print(
        f"[{order_index + 1}/2] starting {spec.label}; fresh server; no retry",
        flush=True,
    )
    started = time.time()
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
            raise CrossModelProtocolError(f"{spec.label} fresh vLLM start failed")
        server_started = True
        runner_code = formal._run_logged(
            command,
            env=environment,
            stdout_path=runner_stdout,
            stderr_path=runner_stderr,
        )
        if runner_code != 0:
            raise CrossModelProtocolError(f"{spec.label} live runner failed")
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
                primary_error = CrossModelProtocolError(
                    f"{spec.label} vLLM did not stop cleanly"
                )
    if primary_error is not None:
        raise primary_error

    result_path = evidence_dir / "result.json"
    timeline_path = evidence_dir / "queue_timeline.jsonl"
    server_log = server_dir / f"vllm_{config['VLLM_PORT']}.log"
    for required in (result_path, timeline_path, server_log):
        if not required.is_file():
            raise CrossModelProtocolError(f"{spec.label} missing evidence: {required}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CrossModelProtocolError(f"{spec.label} result JSON is invalid") from exc
    validation = _validate_result(
        result,
        spec=spec,
        config=config,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        timeline_path=timeline_path,
        server_log=server_log,
        profile=profile,
    )
    validation["elapsed_wall_s_including_server_lifecycle"] = time.time() - started
    validation_path = cell_root / "strict_development_validation.json"
    _write_json_atomic(validation_path, validation)
    evidence_files = (
        cell_root / "cell_contract.json",
        validation_path,
        result_path,
        timeline_path,
        server_log,
        lifecycle_stdout,
        lifecycle_stderr,
        runner_stdout,
        runner_stderr,
    )
    manifest = {
        "schema": "paste_repro.scheduler_cross_model_cell_evidence",
        "version": 2,
        "profile": _profile_record(profile),
        "cell": spec.label,
        "evidence": _relative_file_manifest(cell_root, evidence_files),
    }
    manifest_path = cell_root / "cell_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    _verify_relative_file_manifest(cell_root, manifest)
    _verify_bindings(bindings)
    _verify_snapshot_manifest(snapshot, snapshot_manifest)
    print(f"[{order_index + 1}/2] completed {spec.label}; server stopped", flush=True)
    return cell_root


def _cell_values(cell_root: Path) -> tuple[dict[str, float], list[float]]:
    result = json.loads(
        (cell_root / "evidence/result.json").read_text(encoding="utf-8")
    )
    by_source = {
        str(task["source_id"]): float(task["e2e_s"])
        for task in result["tasks"]
        if task.get("ok") is True
    }
    requests = [
        float(event["duration_s"])
        for event in result["llm_events"]
        if event.get("ok") is True
    ]
    if len(by_source) != EXPECTED_SOURCE_COUNT or len(requests) != EXPECTED_LLM_REQUESTS:
        raise CrossModelProtocolError("completed cell metric identity is incomplete")
    return by_source, requests


def _summarize(
    run_root: Path,
    completed: Mapping[str, Path],
    *,
    model_id: str,
    revision: str,
    snapshot_manifest: Mapping[str, Any],
    profile: ProfileSpec = DEFAULT_PROFILE,
) -> dict[str, Any]:
    _validate_profile(profile)
    selected_cells = _cells(profile)
    values: dict[str, dict[str, float]] = {}
    cells: dict[str, Any] = {}
    for spec in selected_cells:
        tasks, requests = _cell_values(completed[spec.label])
        values[spec.label] = tasks
        validation = json.loads(
            (completed[spec.label] / "strict_development_validation.json").read_text(
                encoding="utf-8"
            )
        )
        cells[spec.label] = {
            "spec": asdict(spec),
            "task_e2e_s": live._distribution(list(tasks.values())),
            "llm_request_duration_s": live._distribution(requests),
            "transport_validation": validation["transport_validation"],
            "physical_kv_telemetry": validation["physical_kv_telemetry"],
        }
    baseline = values[selected_cells[0].label]
    candidate = values[selected_cells[1].label]
    if set(baseline) != set(candidate):
        raise CrossModelProtocolError("A/E source identities differ")
    baseline_mean = statistics.fmean(baseline.values())
    candidate_mean = statistics.fmean(candidate.values())
    return {
        "schema": "paste_repro.scheduler_cross_model_portability_summary",
        "version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "development_only": True,
        "formal_eligible": False,
        "profile": _profile_record(profile),
        "model": {
            "model_id": model_id,
            "revision": revision,
            "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        },
        "cells": cells,
        "a_to_e_effect": {
            "baseline": selected_cells[0].label,
            "candidate": selected_cells[1].label,
            "baseline_mean_s": baseline_mean,
            "candidate_mean_s": candidate_mean,
            "relative_reduction": (baseline_mean - candidate_mean) / baseline_mean,
            "faster_source_count": sum(
                baseline[source] > candidate[source] for source in baseline
            ),
            "paired_source_count": len(baseline),
        },
        "interpretation": {
            "qwen_knobs_were_uncalibrated_for_this_model": True,
            "single_run_per_cell": True,
            "confidence_interval_available": False,
            "fixed_a_then_e_order": True,
            "order_effect_excluded": False,
            "cross_gpu_generalization_proven": False,
            "pooling_with_other_models_allowed": False,
            "cross_profile_comparison_or_pooling_allowed": False,
            "descriptive_portability_stress_only": True,
        },
        "run_root": _repository_relative(run_root),
    }


def _verify_completion_manifest(
    run_root: Path, completion: Mapping[str, Any]
) -> None:
    profile = _profile_from_record(completion.get("profile"))
    expected_profile = _profile_record(profile)
    bindings = completion.get("dependency_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise CrossModelProtocolError("completion lacks dependency bindings")
    _verify_bindings(bindings)
    required_paths = {
        "run_plan": "run_plan.json",
        "model_snapshot_manifest": "model_snapshot_manifest.json",
        "execution_hardware": "execution_hardware.json",
        "summary": "summary.json",
    }
    bound_paths: dict[str, Path] = {}
    for field, required_relative in required_paths.items():
        binding = completion.get(field)
        if not isinstance(binding, Mapping):
            raise CrossModelProtocolError(f"completion lacks {field} binding")
        relative = binding.get("path")
        expected = binding.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or SHA256_RE.fullmatch(expected) is None
        ):
            raise CrossModelProtocolError(f"completion {field} binding is malformed")
        if relative != required_relative:
            raise CrossModelProtocolError(
                f"completion {field} path does not match the protocol"
            )
        path = run_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise CrossModelProtocolError(f"completion {field} SHA mismatch")
        bound_paths[field] = path

    loaded: dict[str, Mapping[str, Any]] = {}
    for field in ("run_plan", "execution_hardware", "summary"):
        try:
            value = json.loads(bound_paths[field].read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CrossModelProtocolError(
                f"completion {field} JSON is invalid"
            ) from exc
        if not isinstance(value, Mapping):
            raise CrossModelProtocolError(f"completion {field} JSON is malformed")
        loaded[field] = value
        if value.get("profile") != expected_profile:
            raise CrossModelProtocolError(
                f"completion {field} portability profile mismatch"
            )
    plan = loaded["run_plan"]
    summary = loaded["summary"]
    if (
        not isinstance(completion.get("attempt_key"), str)
        or completion.get("attempt_key") != plan.get("attempt_key")
    ):
        raise CrossModelProtocolError("completion attempt key does not match run plan")

    selected_cells = _cells(profile)
    plan_cells = plan.get("cells")
    if (
        not isinstance(plan_cells, list)
        or [cell.get("label") for cell in plan_cells if isinstance(cell, Mapping)]
        != [spec.label for spec in selected_cells]
    ):
        raise CrossModelProtocolError("run plan cell labels do not match profile")
    summary_cells = summary.get("cells")
    if (
        not isinstance(summary_cells, Mapping)
        or list(summary_cells) != [spec.label for spec in selected_cells]
    ):
        raise CrossModelProtocolError("summary cell labels do not match profile")
    cells = completion.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise CrossModelProtocolError("completion must bind exactly two cell manifests")
    for index, binding in enumerate(cells):
        spec = selected_cells[index]
        if not isinstance(binding, Mapping):
            raise CrossModelProtocolError(f"completion cell {index} is malformed")
        relative = binding.get("path")
        expected = binding.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or SHA256_RE.fullmatch(expected) is None
        ):
            raise CrossModelProtocolError(f"completion cell {index} is malformed")
        expected_relative = (
            Path("cells")
            / f"{index + 1:02d}-{spec.label}"
            / "cell_manifest.json"
        ).as_posix()
        if binding.get("label") != spec.label or relative != expected_relative:
            raise CrossModelProtocolError(
                f"completion cell {index} label/path does not match profile"
            )
        path = run_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise CrossModelProtocolError(f"completion cell {index} SHA mismatch")
        try:
            cell_manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CrossModelProtocolError(
                f"completion cell {index} manifest JSON is invalid"
            ) from exc
        if not isinstance(cell_manifest, Mapping):
            raise CrossModelProtocolError(
                f"completion cell {index} manifest is malformed"
            )
        if (
            cell_manifest.get("profile") != expected_profile
            or cell_manifest.get("cell") != spec.label
        ):
            raise CrossModelProtocolError(
                f"completion cell {index} manifest profile/label mismatch"
            )
        _verify_relative_file_manifest(path.parent, cell_manifest)


def _validate_execution_hardware(gpus: str) -> dict[str, Any]:
    selected = set(_parse_gpus(gpus))
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise CrossModelProtocolError(
            "execution-only A100 hardware validation failed: " + completed.stderr.strip()
        )
    rows: dict[int, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
            memory_mib = int(parts[2])
        except ValueError:
            continue
        if index in selected:
            rows[index] = {
                "index": index,
                "name": parts[1],
                "memory_mib": memory_mib,
            }
    if set(rows) != selected:
        raise CrossModelProtocolError("selected four GPU IDs were not all reported")
    names = {row["name"] for row in rows.values()}
    memories = {row["memory_mib"] for row in rows.values()}
    if (
        len(names) != 1
        or not all("A100" in name for name in names)
        or len(memories) != 1
        or not all(40_000 <= memory <= 42_000 for memory in memories)
    ):
        raise CrossModelProtocolError(
            "execution requires four identical NVIDIA A100 40GB GPUs"
        )
    return {
        "validated_at_execution": True,
        "gpu_count": 4,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "rows": [rows[index] for index in sorted(rows)],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default=DEFAULT_PROFILE_ID,
        help=(
            "Fixed portability shape. Default preserves legacy c12k/l80; "
            "c5k/l80 is an explicit pre-live cross-architecture fallback."
        ),
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="Exactly four physical A100 GPU IDs; not probed by --check-only.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Offline CPU-only preflight and exact plan; creates no files.",
    )
    parser.add_argument(
        "--execute-one-shot",
        action="store_true",
        help="Consume the attempt key and execute fixed A then E exactly once.",
    )
    parser.add_argument(
        "--expected-preflight-plan-sha256",
        help=(
            "Required for execution; copy preflight_plan_sha256 from the "
            "byte-identical --check-only plan."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = PROFILES[args.profile]
    if RUN_TAG_RE.fullmatch(args.run_tag) is None:
        raise CrossModelProtocolError("RUN_TAG contains unsupported characters")
    if args.check_only and args.execute_one_shot:
        raise CrossModelProtocolError(
            "--check-only and --execute-one-shot are mutually exclusive"
        )
    # Identity/layout checks intentionally precede every subprocess call.  A
    # missing snapshot or malformed revision therefore fails without touching
    # a GPU, a port, the network, or even the pinned environment entrypoints.
    snapshot = _validate_identity(
        args.model_id, args.model_revision, args.model_snapshot
    )
    snapshot_manifest = _snapshot_manifest(
        snapshot, model_id=args.model_id, revision=args.model_revision
    )
    # Freeze repository dependencies before the potentially long tokenizer,
    # grammar, and snapshot preflight, then recheck them before emitting the
    # plan.  This prevents a concurrent edit from being hashed after Python
    # already imported a different implementation.
    bindings = _dependency_bindings()
    config, python, preflight = _full_preflight(
        model_id=args.model_id,
        revision=args.model_revision,
        snapshot=snapshot,
        gpus=args.gpus,
        port=args.port,
        snapshot_manifest=snapshot_manifest,
        profile=profile,
    )
    _verify_bindings(bindings)
    plan = _plan(
        run_tag=args.run_tag,
        model_id=args.model_id,
        revision=args.model_revision,
        snapshot=snapshot,
        snapshot_manifest=snapshot_manifest,
        config=config,
        python=python,
        gpus=args.gpus,
        port=args.port,
        preflight=preflight,
        bindings=bindings,
        profile=profile,
    )
    if args.check_only:
        plan["check_only"] = True
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.execute_one_shot:
        raise CrossModelProtocolError(
            "refusing execution without --execute-one-shot; use --check-only first"
        )
    if (
        not isinstance(args.expected_preflight_plan_sha256, str)
        or SHA256_RE.fullmatch(args.expected_preflight_plan_sha256) is None
        or args.expected_preflight_plan_sha256
        != plan["preflight_plan_sha256"]
    ):
        raise CrossModelProtocolError(
            "execution requires the exact preflight_plan_sha256 emitted by "
            "--check-only; snapshot, dependencies, and plan must be unchanged"
        )

    run_root = _run_root(args.run_tag, profile)
    attempt_key = str(plan["attempt_key"])
    attempt_root = _attempt_root(attempt_key, profile)
    if run_root.exists() or attempt_root.exists():
        raise CrossModelProtocolError(
            "run tag or content-addressed one-shot attempt was already consumed"
        )
    RUN_BASE.mkdir(parents=True, exist_ok=True)
    ATTEMPT_BASE.mkdir(parents=True, exist_ok=True)
    attempt_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempt_root.mkdir()
    except FileExistsError as exc:
        raise CrossModelProtocolError("one-shot attempt was already consumed") from exc
    # The reservation is deliberately never removed, including on failure.
    _write_json_atomic(
        attempt_root / "reservation.json",
        {
            "schema": "paste_repro.cross_model_one_shot_reservation",
            "version": 1,
            "attempt_key": attempt_key,
            "profile": _profile_record(profile),
            "run_tag": args.run_tag,
            "reserved_wall_s": time.time(),
            "rerun_allowed": False,
        },
    )
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir()
    try:
        # The durable reservation above precedes the first GPU-touching
        # operation.  A bad hardware declaration therefore consumes this
        # one-shot attempt just like any later execution failure.
        hardware = _validate_execution_hardware(args.gpus)
        hardware["profile"] = _profile_record(profile)
        snapshot_manifest_path = run_root / "model_snapshot_manifest.json"
        hardware_path = run_root / "execution_hardware.json"
        plan_path = run_root / "run_plan.json"
        _write_json_atomic(snapshot_manifest_path, snapshot_manifest)
        _write_json_atomic(hardware_path, hardware)
        _write_json_atomic(plan_path, plan)
        completed: dict[str, Path] = {}
        selected_cells = _cells(profile)
        for index, spec in enumerate(selected_cells):
            completed[spec.label] = _run_cell(
                run_root=run_root,
                spec=spec,
                order_index=index,
                config=config,
                python=python,
                snapshot=snapshot,
                snapshot_manifest=snapshot_manifest,
                bindings=bindings,
                profile=profile,
            )
        _verify_snapshot_manifest(snapshot, snapshot_manifest)
        _verify_bindings(bindings)
        summary = _summarize(
            run_root,
            completed,
            model_id=args.model_id,
            revision=args.model_revision,
            snapshot_manifest=snapshot_manifest,
            profile=profile,
        )
        summary_path = run_root / "summary.json"
        _write_json_atomic(summary_path, summary)
        completion = {
            "schema": "paste_repro.scheduler_cross_model_portability_completion",
            "version": 2,
            "protocol_version": PROTOCOL_VERSION,
            "development_only": True,
            "formal_eligible": False,
            "profile": _profile_record(profile),
            "attempt_key": attempt_key,
            "completed_wall_s": time.time(),
            "run_plan": {"path": "run_plan.json", "sha256": _sha256(plan_path)},
            "model_snapshot_manifest": {
                "path": "model_snapshot_manifest.json",
                "sha256": _sha256(snapshot_manifest_path),
            },
            "execution_hardware": {
                "path": "execution_hardware.json",
                "sha256": _sha256(hardware_path),
            },
            "summary": {"path": "summary.json", "sha256": _sha256(summary_path)},
            "cells": [
                {
                    "label": spec.label,
                    "path": (
                        Path("cells")
                        / f"{index + 1:02d}-{spec.label}"
                        / "cell_manifest.json"
                    ).as_posix(),
                    "sha256": _sha256(
                        completed[spec.label] / "cell_manifest.json"
                    ),
                }
                for index, spec in enumerate(selected_cells)
            ],
            "dependency_bindings": bindings,
        }
        _verify_completion_manifest(run_root, completion)
        _write_json_atomic(run_root / "completed_pair.json", completion)
        print(
            f"Cross-model one-shot A/E pair completed "
            f"[{profile.profile_id}]: {run_root}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        _write_json_atomic(
            run_root / "failure.json",
            {
                "schema": "paste_repro.scheduler_cross_model_portability_failure",
                "version": 2,
                "attempt_key": attempt_key,
                "profile": _profile_record(profile),
                "rerun_allowed": False,
                "failed_wall_s": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CrossModelProtocolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
