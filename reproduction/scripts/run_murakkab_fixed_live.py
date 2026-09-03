#!/usr/bin/env python3
"""Run one fresh-server constrained-Murakkab cell on the fixed PASTE setup.

The script is a thin, fail-closed wrapper around the existing live closed-loop
runner.  It selects one typed executable DAG before timing, launches native
FCFS with no PASTE scheduler extension, runs demand-only tools, validates the
observed dependency order, and preserves both raw and enriched evidence.

This is an engineering repetition on the already-observed frozen-v9 workload.
It is deliberately not labeled a new formal-v9 or confirmatory result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.murakkab_fixed_runtime import (  # noqa: E402
    SELECTED_CANDIDATE_ID,
    WORKFLOW_ID,
    MurakkabFixedError,
    build_singleton_plan,
    canonical_sha256,
    compute_fixed_metrics,
    sha256_file,
    validate_live_result,
)


PROTOCOL = REPRODUCTION_ROOT / "configs/murakkab_fixed_v9_m_only.json"
WORKLOAD = REPRODUCTION_ROOT / "workloads/live_joint_wikipedia_frozen_formal_v9.json"
WORKLOAD_SHA256 = "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20"
LIVE_RUNNER = REPOSITORY_ROOT / "scripts/run_live_tool_llm_experiment.py"
START_SERVER = REPRODUCTION_ROOT / "scripts/start_vllm.sh"
STOP_SERVER = REPRODUCTION_ROOT / "scripts/stop_vllm.sh"
LIVE_AGENT = REPRODUCTION_ROOT / "paste_repro/live_agent.py"
LIVE_BROKER = REPRODUCTION_ROOT / "paste_repro/live_broker.py"
LIVE_EXECUTOR = REPRODUCTION_ROOT / "paste_repro/live_executor.py"
THIS_RUNTIME = REPRODUCTION_ROOT / "paste_repro/murakkab_fixed_runtime.py"
SCHEDULER_HOOK = REPOSITORY_ROOT / "scripts/pythonhooks/sched_policy_patch.py"
FORMAL_WORKLOAD_VALIDATOR = (
    REPRODUCTION_ROOT / "scripts/validate_live_joint_formal_workload.py"
)
LIVE_TOOL_PROTOCOL = REPRODUCTION_ROOT / "results/live_joint/LIVE_TOOL_LLM_PROTOCOL.md"
DEFAULT_OUTPUT_BASE = REPRODUCTION_ROOT / "artifacts/murakkab_fixed/live"
RUN_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SELECTED_GPU_INDICES = (4, 5, 6, 7)
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-40GB"
REGISTERED_BACKGROUND_POLICY = "registered_shared_resnet_background_v1"
REGISTERED_BACKGROUND_EXECUTABLE = "/opt/conda/envs/ptca/bin/python3.10"
REGISTERED_BACKGROUND_CWD = "/home/aiscuser/gpu_occupy"
REGISTERED_BACKGROUND_ARGV = ("python", "resnet.py")
REGISTERED_BACKGROUND_SCRIPT = "/home/aiscuser/gpu_occupy/resnet.py"
REGISTERED_BACKGROUND_SCRIPT_SHA256 = (
    "3239df3d117271605971a2db4b7f6251b42e06a13cac3509c118b2cc16df09a2"
)
EXPECTED_MODEL = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
EXPECTED_REVISION = "4b0ac5767427a55d08a254f0367e2934976598e0"
EXPECTED_CANONICAL_WORKLOAD_SHA256 = (
    "de588fcbd46c1181156f5a6e49e0264c785c00c43e0d8c2a62698fb6217e3ce7"
)
EXPECTED_SELECTED_WORKLOAD_SHA256 = (
    "750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c"
)

# Prospective hashes copied from the registered formal-v9 runtime allowlist.
# Unlike the per-run binding map, these are not sampled from the current tree.
EXPECTED_FROZEN_RUNTIME_SHA256: dict[str, str] = {
    "reproduction/paste_repro/live_agent.py": (
        "6dab494fa65749b1d60a5b5cbfbb4d0eed3c804b91b3646e0388c707cb7ade8f"
    ),
    "reproduction/paste_repro/live_broker.py": (
        "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27"
    ),
    "reproduction/paste_repro/live_executor.py": (
        "1605c6a3f0002979d11e70c765684b38c6228bf5d69316cd223436aae7179956"
    ),
    "reproduction/results/live_joint/LIVE_TOOL_LLM_PROTOCOL.md": (
        "5ffb2b20582d798a7350f78c42e975e2e516b890486c76148f0edd3ab2c295b6"
    ),
    "reproduction/scripts/start_vllm.sh": (
        "45154b12d870e319781f153c588a9944bfdbb655999e3139f394c7f656eb6a40"
    ),
    "reproduction/scripts/stop_vllm.sh": (
        "90f174e526c26190e927597ee5ff7c32f1f89a62760a937d8b619cebee34f7dd"
    ),
    "reproduction/scripts/validate_live_joint_formal_workload.py": (
        "88d75a7f00d8c8495e0612f93321a26b2193ba6b7bdf89a0674567f0581b0ff4"
    ),
    "scripts/pythonhooks/sched_policy_patch.py": (
        "9acd2316dddddd6a879614336550d2097c47958ef0b56a7da786b55ecf7b8791"
    ),
    "scripts/run_live_tool_llm_experiment.py": (
        "2672bd58a06de204e0a6a92622b688c453cfca36422660c49e32afae5b70afa3"
    ),
}


class MurakkabLiveRunError(RuntimeError):
    """A preflight, lifecycle, execution, or evidence gate failed."""


FIXED_ENVIRONMENT: dict[str, str] = {
    "PASTE_ENV_PREFIX": "/home/aiscuser/.conda/envs/paste",
    "HF_HOME": "/home/aiscuser/hf_cache",
    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    "MODEL_ID": EXPECTED_MODEL,
    "MODEL_REVISION": EXPECTED_REVISION,
    "VLLM_HOST": "127.0.0.1",
    "VLLM_PROBE_HOST": "127.0.0.1",
    "VLLM_PORT": "8100",
    "VLLM_TP_SIZE": "4",
    "VLLM_DTYPE": "bfloat16",
    "VLLM_MAX_MODEL_LEN": "16384",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.86",
    "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
    "VLLM_MAX_NUM_SEQS": "96",
    "VLLM_CUDA_GRAPH_SIZES": "32",
    "VLLM_ENABLE_PREFIX_CACHING": "1",
    "VLLM_USE_V1": "1",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
    "VLLM_READY_TIMEOUT": "3600",
    "VLLM_SHUTDOWN_TIMEOUT": "60",
    "VLLM_SCHED_POLICY": "fcfs",
}


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise MurakkabLiveRunError(f"path is outside the repository: {path}") from exc


def model_snapshot() -> Path:
    key = "models--" + EXPECTED_MODEL.replace("/", "--")
    return Path(FIXED_ENVIRONMENT["HF_HOME"]) / key / "snapshots" / EXPECTED_REVISION


def build_cell_environment(
    *, state_dir: Path, log_dir: Path, inherited: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Clear every scheduler experiment variable before installing FCFS."""

    values = dict(os.environ if inherited is None else inherited)
    for key in tuple(values):
        if key.startswith("VLLM_SCHED_"):
            values.pop(key)
    values.update(FIXED_ENVIRONMENT)
    # Keep this second purge explicit: a future common profile must not leak
    # Joint knobs into M.  Reinstall only the native scheduler selector.
    for key in tuple(values):
        if key.startswith("VLLM_SCHED_") and key != "VLLM_SCHED_POLICY":
            values.pop(key)
    values.update(
        {
            "VLLM_SCHED_POLICY": "fcfs",
            "VLLM_REQUIRE_NEW": "1",
            "VLLM_STATE_DIR": str(state_dir),
            "VLLM_LOG_DIR": str(log_dir),
            "VLLM_HOOK_DIR": str(REPOSITORY_ROOT / "scripts/pythonhooks"),
            "MODEL_SNAPSHOT": str(model_snapshot()),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return values


def build_live_command(
    *, python: Path, output: Path, run_tag: str, source_limit: int | None
) -> list[str]:
    command = [
        str(python), str(LIVE_RUNNER),
        "--workload", str(WORKLOAD),
        "--output-dir", str(output),
        "--server-url", "http://127.0.0.1:8100",
        "--model", EXPECTED_MODEL,
        "--tokenizer", str(model_snapshot()),
        "--cell-label", f"{run_tag}-M",
        "--call-graph-mode", "frozen",
        "--speculation-mode", "off",
        "--tool-signal-policy", "execution_aware",
        "--visit-top-k", "1",
        "--replicas", "1",
        "--max-active-tasks", "80",
        "--tool-workers", "4",
        "--speculative-tool-workers", "2",
        "--min-speculative-tool-workers", "0",
        "--search-tool-capacity", "3",
        "--visit-tool-capacity", "2",
        "--search-min-start-interval-s", "0",
        "--visit-min-start-interval-s", "2.5",
        "--max-speculative-pending", "128",
        "--speculative-ttl-s", "120",
        "--tool-timeout-s", "60",
        "--tool-http-max-attempts", "2",
        "--tool-http-retry-backoff-s", "1.0",
        "--tool-http-attempt-start-gate",
        "--tool-service-hint-s", "2.0",
        "--visit-mode", "jina",
        "--search-mode", "bing",
        "--search-max-results", "5",
        "--visit-max-chars", "3000",
        "--request-timeout-s", "300",
        "--max-tokens-tool", "128",
        "--max-tokens-answer", "256",
        "--fixed-final-completion-tokens", "192",
        "--predicted-visit-result-tokens", "1600",
        "--context-padding-tokens", "10000",
        "--queue-sample-interval-s", "0.2",
        "--visit-canary-stride", "6",
    ]
    if source_limit is not None:
        command.extend(("--source-limit", str(source_limit)))
    return command


def _run_capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=REPOSITORY_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def _read_proc_stat_fields(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, int]:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError as exc:
        raise MurakkabLiveRunError(f"cannot read /proc stat for PID {pid}") from exc
    close = raw.rfind(")")
    fields = raw[close + 2 :].split() if close >= 0 else []
    if len(fields) < 20:
        raise MurakkabLiveRunError(f"malformed /proc stat for PID {pid}")
    try:
        return {"ppid": int(fields[1]), "starttime_ticks": int(fields[19])}
    except (ValueError, IndexError) as exc:
        raise MurakkabLiveRunError(f"malformed /proc stat for PID {pid}") from exc


def _read_boot_id(*, proc_root: Path = Path("/proc")) -> str:
    try:
        value = (proc_root / "sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise MurakkabLiveRunError("cannot read kernel boot_id") from exc
    if re.fullmatch(r"[0-9a-fA-F-]{36}", value) is None:
        raise MurakkabLiveRunError("kernel boot_id is malformed")
    return value


def _read_registered_resnet_identity(
    pid: int, *, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    """Read only the registered child process's exe, cwd, and argv."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise MurakkabLiveRunError("registered ResNet PID must be positive")
    process_root = proc_root / str(pid)
    try:
        executable = str((process_root / "exe").resolve(strict=True))
        cwd = str((process_root / "cwd").resolve(strict=True))
        raw_cmdline = (process_root / "cmdline").read_bytes()
    except OSError as exc:
        raise MurakkabLiveRunError(
            f"cannot read registered ResNet child identity for PID {pid}"
        ) from exc
    try:
        argv = [
            field.decode("utf-8", errors="strict")
            for field in raw_cmdline.split(b"\0")
            if field
        ]
    except UnicodeDecodeError as exc:
        raise MurakkabLiveRunError(
            f"registered ResNet child PID {pid} has non-UTF-8 argv"
        ) from exc
    resolved_script = (
        str((Path(cwd) / argv[1]).resolve()) if len(argv) == 2 else None
    )
    observed = {
        "pid": pid,
        "executable": executable,
        "cwd": cwd,
        "argv": argv,
        "resolved_script": resolved_script,
        "resolved_script_sha256": (
            sha256_file(Path(resolved_script))
            if isinstance(resolved_script, str) and Path(resolved_script).is_file()
            else None
        ),
        "proc_starttime_ticks": _read_proc_stat_fields(
            pid, proc_root=proc_root
        )["starttime_ticks"],
        "boot_id": _read_boot_id(proc_root=proc_root),
    }
    expected = {
        "executable": REGISTERED_BACKGROUND_EXECUTABLE,
        "cwd": REGISTERED_BACKGROUND_CWD,
        "argv": list(REGISTERED_BACKGROUND_ARGV),
        "resolved_script": REGISTERED_BACKGROUND_SCRIPT,
        "resolved_script_sha256": REGISTERED_BACKGROUND_SCRIPT_SHA256,
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise MurakkabLiveRunError(
            f"selected GPUs do not have the registered ResNet child: {mismatches}"
        )
    return observed


def _validate_registered_background(
    *,
    selected_gpus: Sequence[Mapping[str, Any]],
    applications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_by_uuid = {
        str(row["uuid"]): int(row["index"]) for row in selected_gpus
    }
    selected_apps = [
        dict(row) for row in applications if row.get("gpu_uuid") in selected_by_uuid
    ]
    if len(selected_apps) != len(SELECTED_GPU_INDICES):
        raise MurakkabLiveRunError(
            "selected GPUs must have exactly four registered ResNet compute-app rows"
        )
    per_uuid_counts = {
        gpu_uuid: sum(row.get("gpu_uuid") == gpu_uuid for row in selected_apps)
        for gpu_uuid in selected_by_uuid
    }
    if any(count != 1 for count in per_uuid_counts.values()):
        raise MurakkabLiveRunError(
            "each selected GPU must have exactly one registered compute-app row"
        )
    pids = {row.get("pid") for row in selected_apps}
    if len(pids) != 1:
        raise MurakkabLiveRunError(
            "selected GPUs do not share one registered ResNet PID"
        )
    pid = next(iter(pids))
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise MurakkabLiveRunError("registered ResNet compute-app PID is invalid")
    if any(
        row.get("process_name") != "python"
        or not isinstance(row.get("used_memory_mib"), (int, float))
        or isinstance(row.get("used_memory_mib"), bool)
        or float(row["used_memory_mib"]) <= 0.0
        for row in selected_apps
    ):
        raise MurakkabLiveRunError(
            "registered ResNet compute-app rows have invalid name or memory"
        )
    identity = _read_registered_resnet_identity(pid)
    per_gpu_rows = [
        {
            **next(
                row for row in selected_apps
                if row["gpu_uuid"] == str(gpu["uuid"])
            ),
            "gpu_index": int(gpu["index"]),
        }
        for gpu in sorted(selected_gpus, key=lambda row: int(row["index"]))
    ]
    return {
        "valid": True,
        "policy": REGISTERED_BACKGROUND_POLICY,
        **identity,
        "user_confirmed_prior_paste_same_condition": True,
        "selected_gpu_indices": list(SELECTED_GPU_INDICES),
        "selected_gpu_uuids": [
            str(row["uuid"])
            for row in sorted(selected_gpus, key=lambda item: int(item["index"]))
        ],
        "selected_application_record_count": len(selected_apps),
        "additional_selected_gpu_compute_apps_observed": False,
        "per_gpu_rows": per_gpu_rows,
    }


def _query_gpu_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return strict nvidia-smi GPU and compute-app rows."""

    gpu_query = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_query.returncode != 0:
        raise MurakkabLiveRunError("nvidia-smi GPU query failed: " + gpu_query.stderr.strip())
    gpus: list[dict[str, Any]] = []
    for raw in gpu_query.stdout.splitlines():
        if not raw.strip():
            raise MurakkabLiveRunError("nvidia-smi GPU query emitted a blank row")
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 7 or not all(parts):
            raise MurakkabLiveRunError(f"unexpected nvidia-smi GPU row: {raw!r}")
        try:
            row = {
                "index": int(parts[0]), "uuid": parts[1], "name": parts[2],
                "memory_total_mib": float(parts[3]), "memory_used_mib": float(parts[4]),
                "utilization_gpu_percent": float(parts[5]), "power_draw_w": float(parts[6]),
            }
        except ValueError as exc:
            raise MurakkabLiveRunError(
                f"unexpected nvidia-smi GPU row: {raw!r}"
            ) from exc
        gpus.append(row)
    app_query = _run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if app_query.returncode != 0:
        raise MurakkabLiveRunError(
            "nvidia-smi compute-application query failed: "
            + app_query.stderr.strip()
        )
    applications: list[dict[str, Any]] = []
    for raw in app_query.stdout.splitlines():
        if not raw.strip():
            raise MurakkabLiveRunError(
                "nvidia-smi compute-application query emitted a blank row"
            )
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4 or not all(parts):
            raise MurakkabLiveRunError(
                f"unexpected nvidia-smi compute-application row: {raw!r}"
            )
        try:
            pid = int(parts[1])
            used_memory_mib = float(parts[3])
        except ValueError as exc:
            raise MurakkabLiveRunError(
                f"unexpected nvidia-smi compute-application row: {raw!r}"
            ) from exc
        if pid <= 0 or used_memory_mib < 0:
            raise MurakkabLiveRunError(
                f"unexpected nvidia-smi compute-application row: {raw!r}"
            )
        applications.append(
            {
                "gpu_uuid": parts[0], "pid": pid,
                "process_name": parts[2], "used_memory_mib": used_memory_mib,
            }
        )
    known_uuids = {row["uuid"] for row in gpus}
    unknown_uuids = sorted(
        {row["gpu_uuid"] for row in applications} - known_uuids
    )
    if unknown_uuids:
        raise MurakkabLiveRunError(
            f"compute-application query contains unknown GPU UUIDs: {unknown_uuids}"
        )
    return gpus, applications


def _selected_gpus(gpus: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_index = {int(row["index"]): dict(row) for row in gpus}
    selected = [by_index.get(index) for index in SELECTED_GPU_INDICES]
    if any(row is None for row in selected):
        raise MurakkabLiveRunError("selected GPUs 4,5,6,7 are not all visible")
    narrowed = [row for row in selected if row is not None]
    if any(row["name"] != EXPECTED_GPU_NAME for row in narrowed):
        raise MurakkabLiveRunError("selected GPU type is not the fixed A100-SXM4-40GB setup")
    return narrowed


def gpu_snapshot() -> dict[str, Any]:
    gpus, applications = _query_gpu_state()
    selected = _selected_gpus(gpus)
    registered_background = _validate_registered_background(
        selected_gpus=selected,
        applications=applications,
    )
    return {
        "query_wall_s": time.time(),
        "selected_gpu_indices": list(SELECTED_GPU_INDICES),
        "selected_gpus": selected,
        "all_compute_applications": applications,
        "selected_gpu_compute_applications": registered_background["per_gpu_rows"],
        "selected_gpu_background_process_count": 4,
        "registered_background": registered_background,
    }


def validate_registered_background_continuity(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    before_identity = before.get("registered_background")
    after_identity = after.get("registered_background")
    if not isinstance(before_identity, Mapping) or not isinstance(
        after_identity, Mapping
    ):
        raise MurakkabLiveRunError("registered background identity is missing")
    stable_fields = (
        "valid", "policy", "pid", "executable", "cwd", "argv",
        "resolved_script", "resolved_script_sha256", "proc_starttime_ticks",
        "boot_id", "selected_gpu_indices", "selected_gpu_uuids",
        "selected_application_record_count",
        "additional_selected_gpu_compute_apps_observed",
    )
    changed = {
        key: {"before": before_identity.get(key), "after": after_identity.get(key)}
        for key in stable_fields
        if before_identity.get(key) != after_identity.get(key)
    }
    if changed:
        raise MurakkabLiveRunError(
            f"registered ResNet identity changed during the run: {changed}"
        )
    if (
        before_identity.get("valid") is not True
        or before_identity.get("policy") != REGISTERED_BACKGROUND_POLICY
        or before_identity.get("selected_application_record_count") != 4
        or before_identity.get("additional_selected_gpu_compute_apps_observed")
        is not False
    ):
        raise MurakkabLiveRunError("registered ResNet continuity contract failed")
    return {
        "valid": True,
        "policy": REGISTERED_BACKGROUND_POLICY,
        "same_process_identity_before_after": True,
        "pid": before_identity["pid"],
        "proc_starttime_ticks": before_identity["proc_starttime_ticks"],
        "boot_id": before_identity["boot_id"],
        "resolved_script_sha256": before_identity["resolved_script_sha256"],
        "selected_gpu_indices": before_identity["selected_gpu_indices"],
        "selected_gpu_uuids": before_identity["selected_gpu_uuids"],
        "user_confirmed_prior_paste_same_condition": True,
        "load_intensity_equivalence_claimed": False,
    }


def validate_inputs() -> dict[str, Any]:
    required = (
        PROTOCOL, WORKLOAD, LIVE_RUNNER, START_SERVER, STOP_SERVER,
        LIVE_AGENT, LIVE_BROKER, LIVE_EXECUTOR, THIS_RUNTIME, SCHEDULER_HOOK,
        FORMAL_WORKLOAD_VALIDATOR, LIVE_TOOL_PROTOCOL, Path(__file__).resolve(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise MurakkabLiveRunError(f"required inputs are missing: {missing}")
    if sha256_file(WORKLOAD) != WORKLOAD_SHA256:
        raise MurakkabLiveRunError("frozen-v9 workload SHA256 mismatch")
    frozen_mismatches = {
        relative: {
            "expected": expected,
            "observed": (
                sha256_file(REPOSITORY_ROOT / relative)
                if (REPOSITORY_ROOT / relative).is_file()
                else None
            ),
        }
        for relative, expected in EXPECTED_FROZEN_RUNTIME_SHA256.items()
        if not (REPOSITORY_ROOT / relative).is_file()
        or sha256_file(REPOSITORY_ROOT / relative) != expected
    }
    if frozen_mismatches:
        raise MurakkabLiveRunError(
            f"registered formal-v9 runtime SHA256 mismatch: {frozen_mismatches}"
        )
    payload = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list) or len(sources) != 80:
        raise MurakkabLiveRunError("fixed workload must contain exactly 80 sources")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    required_protocol_fields = {
        "schema": "paste_repro.murakkab_fixed_v9_m_only_execution",
        "status": "fixed_engineering_execution",
        "evidence_class": "fixed-v9-setup-engineering",
        "confirmatory_eligible": False,
        "repetitions": 3,
        "fresh_server_per_repetition": True,
        "result_cache_empty_per_repetition": True,
    }
    protocol_mismatches = {
        key: {"expected": value, "observed": protocol.get(key)}
        for key, value in required_protocol_fields.items()
        if protocol.get(key) != value
    }
    cell = protocol.get("cell")
    workload_contract = protocol.get("workload")
    shared_contract = protocol.get("shared_setup")
    gpu_contract = (
        shared_contract.get("gpu") if isinstance(shared_contract, Mapping) else None
    )
    registered_contract = (
        gpu_contract.get("registered_background")
        if isinstance(gpu_contract, Mapping)
        else None
    )
    expected_registered_contract = {
        "required": True,
        "user_confirmed_prior_paste_same_condition": True,
        "selected_gpu_compute_app_records": 4,
        "same_pid_on_every_selected_gpu": True,
        "additional_selected_gpu_compute_apps_allowed": False,
        "executable": REGISTERED_BACKGROUND_EXECUTABLE,
        "cwd": REGISTERED_BACKGROUND_CWD,
        "argv": list(REGISTERED_BACKGROUND_ARGV),
        "resolved_script": REGISTERED_BACKGROUND_SCRIPT,
        "resolved_script_sha256": REGISTERED_BACKGROUND_SCRIPT_SHA256,
        "identity_must_match_before_and_after": True,
    }
    if (
        protocol_mismatches
        or not isinstance(cell, Mapping)
        or cell.get("id") != "M"
        or cell.get("llm_scheduler") != "native FCFS"
        or cell.get("tool_execution") != "demand only"
        or cell.get("call_graph_mode") != "frozen"
        or not isinstance(workload_contract, Mapping)
        or workload_contract.get("sha256") != WORKLOAD_SHA256
        or workload_contract.get("source_count") != 80
        or workload_contract.get("retrospective_existing_workload") is not True
        or not isinstance(gpu_contract, Mapping)
        or gpu_contract.get("visible_indices") != list(SELECTED_GPU_INDICES)
        or gpu_contract.get("background_policy") != REGISTERED_BACKGROUND_POLICY
        or registered_contract != expected_registered_contract
    ):
        raise MurakkabLiveRunError(
            "M-only execution manifest is not the registered engineering design"
        )
    plan = build_singleton_plan(protocol)
    snapshot = model_snapshot()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise MurakkabLiveRunError(f"pinned model snapshot is missing: {snapshot}")
    python = Path(FIXED_ENVIRONMENT["PASTE_ENV_PREFIX"]) / "bin/python"
    if not python.is_file():
        raise MurakkabLiveRunError(f"pinned environment Python is missing: {python}")
    workload_validation_process = _run_capture(
        [str(python), str(FORMAL_WORKLOAD_VALIDATOR), "--workload", str(WORKLOAD)]
    )
    if workload_validation_process.returncode != 0:
        raise MurakkabLiveRunError(
            "frozen workload validator failed: "
            + workload_validation_process.stderr.strip()
        )
    try:
        workload_validation = json.loads(workload_validation_process.stdout)
    except json.JSONDecodeError as exc:
        raise MurakkabLiveRunError("frozen workload validator emitted invalid JSON") from exc
    expected_workload_validation = {
        "valid": True,
        "source_count": 80,
        "formal_eligible": True,
        "file_sha256": WORKLOAD_SHA256,
        "canonical_json_sha256": EXPECTED_CANONICAL_WORKLOAD_SHA256,
        "canonical_sources_sha256": EXPECTED_SELECTED_WORKLOAD_SHA256,
    }
    validation_mismatches = {
        key: {"expected": value, "observed": workload_validation.get(key)}
        for key, value in expected_workload_validation.items()
        if workload_validation.get(key) != value
    }
    if validation_mismatches:
        raise MurakkabLiveRunError(
            f"frozen-v9 canonical workload mismatch: {validation_mismatches}"
        )
    versions = _run_capture(
        [
            str(python), "-c",
            (
                "from importlib.metadata import version; "
                "print(version('vllm'), version('transformers'), "
                "version('aiohttp'), version('xgrammar'))"
            ),
        ]
    )
    if versions.returncode != 0 or versions.stdout.strip().split() != [
        "0.10.1", "4.56.1", "3.12.15", "0.1.21"
    ]:
        raise MurakkabLiveRunError(
            "pinned live stack version mismatch: " + versions.stdout.strip()
        )
    return {
        "valid": True,
        "workload_source_count": len(sources),
        "workload_sha256": WORKLOAD_SHA256,
        "workload_validation": workload_validation,
        "prospective_frozen_runtime_sha256": EXPECTED_FROZEN_RUNTIME_SHA256,
        "protocol_sha256": sha256_file(PROTOCOL),
        "model_snapshot": str(snapshot),
        "model_snapshot_config_sha256": sha256_file(snapshot / "config.json"),
        "python": str(python),
        "versions": {
            "vllm": "0.10.1", "transformers": "4.56.1",
            "aiohttp": "3.12.15", "xgrammar": "0.1.21",
        },
        "singleton_plan": plan,
        "bindings": {repository_relative(path): sha256_file(path) for path in required},
    }


def verify_bindings(bindings: Mapping[str, str]) -> None:
    for relative, expected in bindings.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise MurakkabLiveRunError(f"bound input changed during run: {relative}")


def _run_logged(
    command: Sequence[str], *, environment: Mapping[str, str], stdout: Path, stderr: Path
) -> int:
    with stdout.open("ab") as out, stderr.open("ab") as err:
        completed = subprocess.run(
            list(command), cwd=REPOSITORY_ROOT, env=dict(environment),
            stdout=out, stderr=err, check=False,
        )
    return completed.returncode


def _broker_drained(raw_result: Mapping[str, Any]) -> bool:
    snapshot = raw_result.get("broker_final_snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("jobs") != []:
        return False
    counts = snapshot.get("counts")
    if not isinstance(counts, Mapping):
        return False
    return all(
        counts.get(key) == 0
        for key in (
            "completed_unclaimed_speculative", "queued_authoritative",
            "queued_speculative", "running_authoritative", "running_speculative",
        )
    ) and counts.get("queued_by_tool") == {} and counts.get("running_by_tool") == {}


def _validate_no_observed_retries(raw_result: Mapping[str, Any]) -> None:
    records = raw_result.get("tool_attempt_records")
    if not isinstance(records, list) or not records:
        raise MurakkabLiveRunError("tool attempt evidence is missing")
    retried = [
        record.get("invocation_id")
        for record in records
        if isinstance(record, Mapping) and record.get("http_attempts") != 1
    ]
    if retried:
        raise MurakkabLiveRunError(
            f"fixed-v9 zero-observed-retry gate failed for {len(retried)} tools"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--source-limit", type=int,
        help="Development smoke only; labels the result non-comparable.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate inputs, plan, hardware, and registered ResNet without starting vLLM.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if RUN_TAG_RE.fullmatch(args.run_tag) is None:
        raise MurakkabLiveRunError("run_tag contains unsupported characters")
    if args.source_limit is not None and not 1 <= args.source_limit <= 80:
        raise MurakkabLiveRunError("--source-limit must be in [1, 80]")

    preflight = validate_inputs()
    hardware = gpu_snapshot()
    if args.check_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "check_only": True,
                    "gpu_or_server_touched": False,
                    "ready_for_performance_run": True,
                    "registered_background": hardware["registered_background"],
                    "preflight": preflight,
                    "hardware": hardware,
                },
                ensure_ascii=False, indent=2, sort_keys=True,
            )
        )
        return 0

    output_base = args.output_base.resolve()
    repository_relative(output_base)
    run_root = output_base / args.run_tag
    lock = output_base / f".{args.run_tag}.lock"
    if run_root.exists() or lock.exists():
        raise MurakkabLiveRunError(f"run output or lock already exists: {run_root}")
    output_base.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise MurakkabLiveRunError(f"run tag is already reserved: {args.run_tag}") from exc

    started = False
    failure: BaseException | None = None
    try:
        run_root.mkdir()
        state_dir = run_root / "state"
        server_dir = run_root / "server"
        raw_dir = run_root / "runner_raw"
        evidence_dir = run_root / "evidence"
        state_dir.mkdir()
        server_dir.mkdir()
        evidence_dir.mkdir()
        environment = build_cell_environment(state_dir=state_dir, log_dir=server_dir)
        server_instance_id = str(uuid.uuid4())
        plan = deepcopy(preflight["singleton_plan"])
        plan.update(
            {
                "evidence_tier": "fixed-v9-setup-engineering",
                "implementation_kind": "constrained_murakkab_style_emulation",
                "official_code_used": False,
                "official_runtime_reproduced": False,
                "runtime_semantics": "A-equivalent",
                "runtime_semantics_detail": (
                    "existing native-FCFS+demand-only A backend with outside-timed "
                    "singleton typed-DAG control plane"
                ),
                "confirmatory_eligible": False,
                "reason_not_confirmatory": (
                    "the frozen-v9 workload was observed before this M-only repetition"
                ),
                "run_tag": args.run_tag,
                "repetition": args.repetition,
                "server_instance_id": server_instance_id,
                "call_graph_mode": "frozen",
                "source_limit": args.source_limit,
                "performance_comparable": args.source_limit is None,
                "registered_background_policy": REGISTERED_BACKGROUND_POLICY,
                "registered_background": hardware["registered_background"],
                "fresh_server_required": True,
                "result_cache_empty_required": True,
            }
        )
        write_json_atomic(run_root / "preflight.json", preflight)
        write_json_atomic(run_root / "hardware_before.json", hardware)
        write_json_atomic(run_root / "run_plan.json", plan)
        effective_config = {
            "schema": "paste_repro.murakkab_fixed_live_effective_config",
            "version": 1,
            "cell_id": "M",
            "label": "Murakkab-fixed singleton DAG / native FCFS / demand-only",
            "evidence_tier": "fixed-v9-setup-engineering",
            "environment": {
                key: environment[key]
                for key in sorted(environment)
                if key.startswith(("CUDA_", "MODEL_", "VLLM_", "PASTE_ENV_", "HF_HOME"))
            },
            "runner_command": build_live_command(
                python=Path(preflight["python"]), output=raw_dir,
                run_tag=args.run_tag, source_limit=args.source_limit,
            ),
        }
        write_json_atomic(run_root / "effective_config.json", effective_config)
        verify_bindings(preflight["bindings"])

        lifecycle_out = run_root / "server_lifecycle.stdout.log"
        lifecycle_err = run_root / "server_lifecycle.stderr.log"
        runner_out = run_root / "runner.stdout.log"
        runner_err = run_root / "runner.stderr.log"
        start_code = _run_logged(
            [str(START_SERVER)], environment=environment,
            stdout=lifecycle_out, stderr=lifecycle_err,
        )
        if start_code != 0:
            raise MurakkabLiveRunError("fresh vLLM start failed; inspect lifecycle logs")
        started = True
        command = effective_config["runner_command"]
        run_code = _run_logged(
            command, environment=environment, stdout=runner_out, stderr=runner_err
        )
        if run_code != 0:
            raise MurakkabLiveRunError("M live runner failed; inspect runner logs")
    except BaseException as exc:
        failure = exc
    finally:
        if started:
            stop_code = _run_logged(
                [str(STOP_SERVER)], environment=environment,
                stdout=lifecycle_out, stderr=lifecycle_err,
            )
            if stop_code != 0 and failure is None:
                failure = MurakkabLiveRunError("fresh vLLM did not stop cleanly")
    if failure is not None:
        write_json_atomic(
            run_root / "failed_run.json",
            {
                "failed": True, "error_type": type(failure).__name__,
                "error": repr(failure), "server_stop_attempted": started,
            },
        )
        raise failure

    raw_result_path = raw_dir / "result.json"
    raw_timeline_path = raw_dir / "queue_timeline.jsonl"
    server_log_path = server_dir / "vllm_8100.log"
    for path in (raw_result_path, raw_timeline_path, server_log_path):
        if not path.is_file():
            raise MurakkabLiveRunError(f"run evidence is missing: {path}")
    raw_result = json.loads(raw_result_path.read_text(encoding="utf-8"))
    expected_tasks = args.source_limit or 80
    try:
        dependency_evidence = validate_live_result(
            raw_result, call_graph_mode="frozen", expected_task_count=expected_tasks
        )
        _validate_no_observed_retries(raw_result)
    except MurakkabFixedError as exc:
        raise MurakkabLiveRunError(str(exc)) from exc
    drained = _broker_drained(raw_result)
    if not drained:
        raise MurakkabLiveRunError("M live broker was not drained at cell end")
    verify_bindings(preflight["bindings"])

    hardware_after = gpu_snapshot()
    write_json_atomic(run_root / "hardware_after.json", hardware_after)
    background_continuity = validate_registered_background_continuity(
        hardware, hardware_after
    )
    evidence_timeline = evidence_dir / "queue_timeline.jsonl"
    shutil.copy2(raw_timeline_path, evidence_timeline)
    plan_sha = sha256_file(run_root / "run_plan.json")
    raw_sha = sha256_file(raw_result_path)
    enriched = deepcopy(raw_result)
    enriched_config = enriched["config"]
    enriched_config["murakkab_fixed"] = {
        "enabled": True,
        "cell_id": "M",
        "evidence_class": "fixed-v9-setup-engineering",
        "implementation_kind": "constrained_murakkab_style_emulation",
        "official_code_used": False,
        "official_runtime_reproduced": False,
        "runtime_semantics": "A-equivalent",
        "runtime_semantics_detail": (
            "existing native-FCFS+demand-only A backend with outside-timed "
            "singleton typed-DAG control plane"
        ),
        "workflow_id": WORKFLOW_ID,
        "optimizer_candidate_count": 1,
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "typed_dag_validated": True,
        "dependency_ready_dispatch": True,
        "optimizer_outside_timed_path": True,
        "scheduler": "native_fcfs",
        "tool_execution": "demand_only",
        "gpu_type": EXPECTED_GPU_NAME,
        "gpu_count": 4,
        "hardware_evidence": {
            "selected_gpu_indices": list(SELECTED_GPU_INDICES),
            "selected_gpu_names": [row["name"] for row in hardware["selected_gpus"]],
            "selected_gpu_uuids": [row["uuid"] for row in hardware["selected_gpus"]],
            "registered_background_policy": REGISTERED_BACKGROUND_POLICY,
            "registered_background_before": hardware["registered_background"],
            "registered_background_after": hardware_after["registered_background"],
            "registered_background_continuity": background_continuity,
            "before_path": str(run_root / "hardware_before.json"),
            "before_sha256": sha256_file(run_root / "hardware_before.json"),
            "after_path": str(run_root / "hardware_after.json"),
            "after_sha256": sha256_file(run_root / "hardware_after.json"),
        },
        "plan_sha256": plan_sha,
        "registry_sha256": plan["registry_sha256"],
        "workflow_sha256": plan["workflow_sha256"],
        "raw_runner_result_sha256": raw_sha,
        "execution_boundary": (
            "outside-timed singleton planning; existing sequential live agent is "
            "the dependency-ready execution backend"
        ),
        "engineering_run": {
            "evidence_tier": "fixed-v9-setup-engineering",
            "confirmatory_eligible": False,
            "run_id": args.run_tag,
            "run_tag": args.run_tag,
            "repetition": args.repetition,
            "server_instance_id": server_instance_id,
            "fresh_server": True,
            "result_cache_empty": True,
            "broker_drained": True,
            "assertion_owner": "run_murakkab_fixed_live.py",
            "performance_comparable": args.source_limit is None,
            "performance_comparability_scope": (
                "same fixed model/hardware/workload runtime setup only; this field "
                "does not assert a fresh causal comparison with historical PASTE"
            ),
            "source_limit": args.source_limit,
            "registered_background_policy": REGISTERED_BACKGROUND_POLICY,
            "registered_background_same_identity_before_after": True,
            "user_confirmed_prior_paste_same_condition": True,
            "registered_background_load_intensity_equivalence_claimed": False,
        },
    }
    enriched["murakkab_execution_evidence"] = dependency_evidence
    enriched["raw_evidence"]["queue_timeline"] = {
        "path": str(evidence_timeline),
        "sha256": sha256_file(evidence_timeline),
        "sample_count": raw_result["raw_evidence"]["queue_timeline"]["sample_count"],
    }
    enriched["murakkab_provenance"] = {
        "plan_path": str(run_root / "run_plan.json"),
        "plan_sha256": plan_sha,
        "hardware_before_path": str(run_root / "hardware_before.json"),
        "hardware_before_sha256": sha256_file(run_root / "hardware_before.json"),
        "hardware_after_path": str(run_root / "hardware_after.json"),
        "hardware_after_sha256": sha256_file(run_root / "hardware_after.json"),
        "registered_background_continuity": background_continuity,
        "unmodified_runner_result": {
            "path": str(raw_result_path), "sha256": raw_sha,
        },
    }
    evidence_result = evidence_dir / "result.json"
    write_json_atomic(evidence_result, enriched)
    metrics = compute_fixed_metrics(enriched)
    metrics["evidence_tier"] = "fixed-v9-setup-engineering"
    metrics["performance_comparable"] = args.source_limit is None
    metrics["registered_background"] = background_continuity
    write_json_atomic(evidence_dir / "metrics.json", metrics)

    manifest_paths = (
        run_root / "preflight.json", run_root / "hardware_before.json",
        run_root / "hardware_after.json", run_root / "run_plan.json",
        run_root / "effective_config.json", raw_result_path, raw_timeline_path,
        evidence_result, evidence_timeline, evidence_dir / "metrics.json",
        server_log_path, lifecycle_out, lifecycle_err, runner_out, runner_err,
    )
    manifest = {
        "schema": "paste_repro.murakkab_fixed_live_completion",
        "version": 1,
        "completed": True,
        "evidence_tier": "fixed-v9-setup-engineering",
        "confirmatory_eligible": False,
        "run_tag": args.run_tag,
        "repetition": args.repetition,
        "registered_background": background_continuity,
        "result": repository_relative(evidence_result),
        "metrics": repository_relative(evidence_dir / "metrics.json"),
        "artifacts": {repository_relative(path): sha256_file(path) for path in manifest_paths},
    }
    write_json_atomic(run_root / "completed_run.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    try:
        lock.rmdir()
    except OSError as exc:
        raise MurakkabLiveRunError(f"could not release run lock: {lock}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
