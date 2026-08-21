#!/usr/bin/env python3
"""Run the v2 single-token P0/P1 native-prefix causal development matrix."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Sequence

from run_native_prefix_prompt_cell import (
    OUTPUT_CONSTRAINT,
    build_task_fixtures,
    load_sources,
    validate_single_token_sentinel,
)
from validate_native_prefix_causal_dev import (
    CELL_MANIFEST_SCHEMA,
    EFFECTIVE_CONFIG_SCHEMA,
    EXACT_MATRIX,
    EXACT_THRESHOLDS,
    PLAN_SCHEMA,
    ValidationError,
    validate_run,
    write_json_atomic,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/native_prefix_causal_dev.env.example"
)
CELL_RUNNER = (
    REPOSITORY_ROOT / "reproduction/scripts/run_native_prefix_prompt_cell.py"
)
VALIDATOR = (
    REPOSITORY_ROOT / "reproduction/scripts/validate_native_prefix_causal_dev.py"
)
START_SERVER = REPOSITORY_ROOT / "reproduction/scripts/start_vllm.sh"
STOP_SERVER = REPOSITORY_ROOT / "reproduction/scripts/stop_vllm.sh"
PROTOCOL = (
    REPOSITORY_ROOT
    / "reproduction/results/live_joint/NATIVE_PREFIX_CAUSAL_DEV_PROTOCOL.md"
)
WORKLOAD_SHA256 = "e9f63f75bb80c840fbc59f2aa9a581527669c10fc761a4649f50a1bc03eaf1ea"
ORDERS = (("P0", "P1"), ("P1", "P0"))
EXPORT_RE = re.compile(r'export ([A-Z][A-Z0-9_]*)="([^"\\]*)"\Z')
SERVER_IDENTITY_SCHEMA = "paste_repro.native_prefix_causal_server_identity_v2"

EXPECTED_CONFIG = {
    "PASTE_PREFIX_CAUSAL_PROFILE": (
        "native_prefix_causal_dev_v2_context10000_fcfs_local48_single_token"
    ),
    "PASTE_PREFIX_CAUSAL_WORKLOAD": (
        "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
    ),
    "PASTE_PREFIX_CAUSAL_WORKLOAD_SHA256": WORKLOAD_SHA256,
    "PASTE_PREFIX_CAUSAL_SOURCE_COUNT": "16",
    "PASTE_PREFIX_CAUSAL_REPLICAS": "3",
    "PASTE_PREFIX_CAUSAL_TASK_COUNT": "48",
    "PASTE_PREFIX_CAUSAL_CALLS_PER_TASK": "3",
    "PASTE_PREFIX_CAUSAL_ORDERS": "P0,P1;P1,P0",
    "PASTE_PREFIX_CAUSAL_RUN_BASE": (
        "reproduction/artifacts/live_joint/prefix_native_causal_dev_v2"
    ),
    "PASTE_ENV_PREFIX": "/home/aiscuser/.conda/envs/paste",
    "HF_HOME": "/home/aiscuser/hf_cache",
    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
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
    "VLLM_USE_V1": "1",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
    "VLLM_READY_TIMEOUT": "3600",
    "VLLM_SHUTDOWN_TIMEOUT": "60",
    "PASTE_PREFIX_CONTEXT_PADDING_TOKENS": "10000",
    "PASTE_PREFIX_VISIT_FIXTURE_TOKENS": "900",
    "PASTE_PREFIX_MAX_ACTIVE_TASKS": "48",
    "PASTE_PREFIX_REQUEST_TIMEOUT_S": "300",
    "PASTE_PREFIX_QUEUE_SAMPLE_INTERVAL_S": "0.2",
    "PASTE_PREFIX_SENTINEL": "A",
    "PASTE_PREFIX_OUTPUT_CONSTRAINT": "guided_choice_singleton_v1",
    "PASTE_PREFIX_MAX_TOKENS_CALL0": "1",
    "PASTE_PREFIX_MAX_TOKENS_CALL1": "1",
    "PASTE_PREFIX_MAX_TOKENS_CALL2": "1",
    "PASTE_PREFIX_MIN_NATIVE_HIT_RATIO": "0.60",
    "PASTE_PREFIX_MIN_PREFILL_REDUCTION": "0.15",
    "PASTE_PREFIX_MIN_MEAN_REQUEST_REDUCTION": "0.03",
    "PASTE_PREFIX_MIN_MEAN_TASK_E2E_REDUCTION": "0.03",
    "PASTE_PREFIX_MAX_TASK_P95_RATIO": "1.03",
    "PASTE_PREFIX_MAX_COMPLETION_TOKEN_RELATIVE_DIFFERENCE": "0.01",
    "PASTE_PREFIX_BOOTSTRAP_SAMPLES": "10000",
    "PASTE_PREFIX_BOOTSTRAP_SEED": "20260816",
}


class RunnerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RunnerError(f"path is outside repository: {path}") from exc


def load_frozen_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RunnerError(f"frozen config is missing: {path}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = EXPORT_RE.fullmatch(line)
        if match is None:
            raise RunnerError(f"unsupported config syntax at line {line_number}")
        key, value = match.groups()
        if key in values:
            raise RunnerError(f"duplicate config key: {key}")
        values[key] = value
    if values != EXPECTED_CONFIG:
        missing = sorted(set(EXPECTED_CONFIG) - set(values))
        extra = sorted(set(values) - set(EXPECTED_CONFIG))
        changed = sorted(
            key
            for key in set(values) & set(EXPECTED_CONFIG)
            if values[key] != EXPECTED_CONFIG[key]
        )
        raise RunnerError(
            f"frozen config mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    return values


def _model_snapshot(config: Mapping[str, str]) -> Path:
    cache_key = "models--" + config["MODEL_ID"].replace("/", "--")
    return (
        Path(config["HF_HOME"])
        / cache_key
        / "snapshots"
        / config["MODEL_REVISION"]
    )


def _matrix_from_config(config: Mapping[str, str]) -> dict[str, Any]:
    matrix = {
        "source_count": int(config["PASTE_PREFIX_CAUSAL_SOURCE_COUNT"]),
        "replicas": int(config["PASTE_PREFIX_CAUSAL_REPLICAS"]),
        "task_count": int(config["PASTE_PREFIX_CAUSAL_TASK_COUNT"]),
        "calls_per_task": int(config["PASTE_PREFIX_CAUSAL_CALLS_PER_TASK"]),
        "context_padding_tokens": int(
            config["PASTE_PREFIX_CONTEXT_PADDING_TOKENS"]
        ),
        "visit_fixture_tokens": int(
            config["PASTE_PREFIX_VISIT_FIXTURE_TOKENS"]
        ),
        "max_active_tasks": int(config["PASTE_PREFIX_MAX_ACTIVE_TASKS"]),
        "max_tokens_by_call": [
            int(config["PASTE_PREFIX_MAX_TOKENS_CALL0"]),
            int(config["PASTE_PREFIX_MAX_TOKENS_CALL1"]),
            int(config["PASTE_PREFIX_MAX_TOKENS_CALL2"]),
        ],
        "sentinel": config["PASTE_PREFIX_SENTINEL"],
        "output_constraint": config["PASTE_PREFIX_OUTPUT_CONSTRAINT"],
    }
    if matrix != EXACT_MATRIX:
        raise RunnerError("frozen matrix values differ from validator contract")
    return matrix


def _thresholds_from_config(config: Mapping[str, str]) -> dict[str, Any]:
    thresholds = {
        "min_native_hit_ratio": float(
            config["PASTE_PREFIX_MIN_NATIVE_HIT_RATIO"]
        ),
        "min_prefill_reduction": float(
            config["PASTE_PREFIX_MIN_PREFILL_REDUCTION"]
        ),
        "min_mean_request_reduction": float(
            config["PASTE_PREFIX_MIN_MEAN_REQUEST_REDUCTION"]
        ),
        "min_mean_task_e2e_reduction": float(
            config["PASTE_PREFIX_MIN_MEAN_TASK_E2E_REDUCTION"]
        ),
        "max_task_p95_ratio": float(
            config["PASTE_PREFIX_MAX_TASK_P95_RATIO"]
        ),
        "max_completion_token_relative_difference": float(
            config["PASTE_PREFIX_MAX_COMPLETION_TOKEN_RELATIVE_DIFFERENCE"]
        ),
        "bootstrap_samples": int(config["PASTE_PREFIX_BOOTSTRAP_SAMPLES"]),
        "bootstrap_seed": int(config["PASTE_PREFIX_BOOTSTRAP_SEED"]),
    }
    if thresholds != EXACT_THRESHOLDS:
        raise RunnerError("frozen thresholds differ from validator contract")
    return thresholds


def _preflight_fixture(
    *,
    config: Mapping[str, str],
    sources: Sequence[Any],
    model_snapshot: Path,
) -> dict[str, Any]:
    # This is deliberately CPU/local-only.  It catches tokenizer, chat-template,
    # prompt-shape, context-window, and fixed-completion-cap failures before a
    # four-GPU server is started.
    from transformers import AutoTokenizer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_snapshot),
        trust_remote_code=True,
        local_files_only=True,
    )
    tasks, manifest_sha256 = build_task_fixtures(
        tokenizer,
        sources,
        replicas=int(config["PASTE_PREFIX_CAUSAL_REPLICAS"]),
        context_padding_tokens=int(config["PASTE_PREFIX_CONTEXT_PADDING_TOKENS"]),
        visit_fixture_tokens=int(config["PASTE_PREFIX_VISIT_FIXTURE_TOKENS"]),
        max_tokens_by_call=(
            int(config["PASTE_PREFIX_MAX_TOKENS_CALL0"]),
            int(config["PASTE_PREFIX_MAX_TOKENS_CALL1"]),
            int(config["PASTE_PREFIX_MAX_TOKENS_CALL2"]),
        ),
        max_model_len=int(config["VLLM_MAX_MODEL_LEN"]),
        sentinel=config["PASTE_PREFIX_SENTINEL"],
    )
    sentinel_contract = validate_single_token_sentinel(
        tokenizer, config["PASTE_PREFIX_SENTINEL"]
    )
    if config["PASTE_PREFIX_OUTPUT_CONSTRAINT"] != OUTPUT_CONSTRAINT:
        raise RunnerError("frozen output constraint differs from cell contract")
    if len(tasks) != int(config["PASTE_PREFIX_CAUSAL_TASK_COUNT"]):
        raise RunnerError("CPU fixture preflight task count mismatch")
    by_call = [
        [task.calls[call_index].prompt_tokens for task in tasks]
        for call_index in range(3)
    ]
    completion_by_call = [
        [task.calls[call_index].expected_completion_tokens for task in tasks]
        for call_index in range(3)
    ]
    return {
        "fixture_manifest_sha256": manifest_sha256,
        "task_count": len(tasks),
        "call_count": sum(len(task.calls) for task in tasks),
        "prompt_tokens_by_call": [
            {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for values in by_call
        ],
        "completion_tokens_by_call": [
            {"min": min(values), "max": max(values)}
            for values in completion_by_call
        ],
        "max_prompt_plus_generation_cap": max(
            call.prompt_tokens + call.max_tokens
            for task in tasks
            for call in task.calls
        ),
        "sentinel_contract": sentinel_contract,
    }


def _engine_plan(config: Mapping[str, str]) -> dict[str, str]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "MODEL_ID",
        "MODEL_REVISION",
        "VLLM_HOST",
        "VLLM_PROBE_HOST",
        "VLLM_PORT",
        "VLLM_TP_SIZE",
        "VLLM_DTYPE",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_CUDA_GRAPH_SIZES",
        "VLLM_USE_V1",
    )
    result = {key: config[key] for key in keys}
    result["VLLM_SCHED_POLICY"] = "fcfs"
    result["VLLM_ENABLE_PREFIX_CACHING"] = "per_cell_P0_0_P1_1"
    return result


def _relative_bindings(paths: Sequence[Path]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise RunnerError(f"bound file is missing: {path}")
        bindings[repository_relative(path)] = sha256_file(path)
    return bindings


def _verify_bindings(bindings: Mapping[str, str]) -> None:
    for relative, expected in bindings.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RunnerError(f"bound input changed during run: {relative}")


def _cell_environment(
    config: Mapping[str, str],
    *,
    cell_id: str,
    state_dir: Path,
    server_dir: Path,
    model_snapshot: Path,
    native_pythonpath: Path,
) -> dict[str, str]:
    if cell_id not in {"P0", "P1"}:
        raise RunnerError("cell must be P0 or P1")
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("VLLM_") or key in {"PYTHONHOME", "PYTHONPATH"}:
            env.pop(key, None)
    for key in (
        "PASTE_ENV_PREFIX",
        "HF_HOME",
        "CUDA_VISIBLE_DEVICES",
        "MODEL_ID",
        "MODEL_REVISION",
        "VLLM_HOST",
        "VLLM_PROBE_HOST",
        "VLLM_PORT",
        "VLLM_TP_SIZE",
        "VLLM_DTYPE",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_CUDA_GRAPH_SIZES",
        "VLLM_USE_V1",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
        "VLLM_READY_TIMEOUT",
        "VLLM_SHUTDOWN_TIMEOUT",
    ):
        env[key] = config[key]
    env.update(
        {
            "VLLM_ENABLE_PREFIX_CACHING": "1" if cell_id == "P1" else "0",
            "VLLM_SCHED_POLICY": "fcfs",
            "VLLM_REQUIRE_NEW": "1",
            "VLLM_STATE_DIR": str(state_dir),
            "VLLM_LOG_DIR": str(server_dir),
            # start_vllm prepends this directory to PYTHONPATH.  Keeping it
            # empty makes native FCFS structural: sitecustomize and the local
            # scheduler hook cannot be imported in these cells.
            "VLLM_HOOK_DIR": str(native_pythonpath),
            "MODEL_SNAPSHOT": str(model_snapshot),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "PYTHONPATH": "",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _recorded_environment(env: Mapping[str, str]) -> dict[str, str]:
    keys = sorted(
        {
            "CUDA_VISIBLE_DEVICES",
            "MODEL_ID",
            "MODEL_REVISION",
            "PYTHONPATH",
            *(key for key in env if key.startswith("VLLM_")),
        }
    )
    result = {key: env[key] for key in keys}
    scheduler_keys = {key for key in env if key.startswith("VLLM_SCHED_")}
    if scheduler_keys != {"VLLM_SCHED_POLICY"}:
        raise RunnerError(f"cell environment leaked scheduler keys: {scheduler_keys}")
    return result


def _capture_server_identity(
    *,
    state_dir: Path,
    port: str,
    server_instance_id: str,
) -> dict[str, Any]:
    pid_path = state_dir / f"vllm_{port}.pid"
    try:
        pid_text = pid_path.read_text(encoding="utf-8").strip()
        pid = int(pid_text)
    except (OSError, ValueError) as exc:
        raise RunnerError("fresh server PID evidence is missing or invalid") from exc
    if pid <= 0:
        raise RunnerError("fresh server PID must be positive")
    proc_root = Path("/proc") / str(pid)
    try:
        stat_fields = (proc_root / "stat").read_text(encoding="utf-8").split()
        start_ticks = int(stat_fields[21])
        executable = str((proc_root / "exe").resolve(strict=True))
        cmdline = (proc_root / "cmdline").read_bytes()
    except (OSError, IndexError, ValueError) as exc:
        raise RunnerError("cannot bind the fresh server process identity") from exc
    if start_ticks <= 0 or not cmdline:
        raise RunnerError("fresh server process identity is incomplete")
    process_identity = {
        "pid": pid,
        "proc_start_ticks": start_ticks,
        "executable": executable,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }
    return {
        "schema": SERVER_IDENTITY_SCHEMA,
        "version": 2,
        "server_instance_id": server_instance_id,
        "captured_wall_s": time.time(),
        **process_identity,
        "process_identity_sha256": hashlib.sha256(
            json.dumps(
                process_identity,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _cell_command(
    *,
    python: Path,
    config: Mapping[str, str],
    workload: Path,
    model_snapshot: Path,
    output_dir: Path,
    block_id: str,
    order_index: int,
    cell_id: str,
    server_instance_id: str,
) -> list[str]:
    enabled_flag = (
        "--prefix-cache-enabled" if cell_id == "P1" else "--no-prefix-cache-enabled"
    )
    return [
        str(python),
        str(CELL_RUNNER),
        "--workload",
        str(workload),
        "--workload-sha256",
        config["PASTE_PREFIX_CAUSAL_WORKLOAD_SHA256"],
        "--output-dir",
        str(output_dir),
        "--server-url",
        f"http://127.0.0.1:{config['VLLM_PORT']}",
        "--model",
        config["MODEL_ID"],
        "--tokenizer",
        str(model_snapshot),
        "--cell-id",
        cell_id,
        "--block-id",
        block_id,
        "--order-index",
        str(order_index),
        "--server-instance-id",
        server_instance_id,
        enabled_flag,
        "--expected-source-count",
        config["PASTE_PREFIX_CAUSAL_SOURCE_COUNT"],
        "--expected-task-count",
        config["PASTE_PREFIX_CAUSAL_TASK_COUNT"],
        "--replicas",
        config["PASTE_PREFIX_CAUSAL_REPLICAS"],
        "--max-active-tasks",
        config["PASTE_PREFIX_MAX_ACTIVE_TASKS"],
        "--context-padding-tokens",
        config["PASTE_PREFIX_CONTEXT_PADDING_TOKENS"],
        "--visit-fixture-tokens",
        config["PASTE_PREFIX_VISIT_FIXTURE_TOKENS"],
        "--sentinel",
        config["PASTE_PREFIX_SENTINEL"],
        "--output-constraint",
        config["PASTE_PREFIX_OUTPUT_CONSTRAINT"],
        "--max-tokens-call0",
        config["PASTE_PREFIX_MAX_TOKENS_CALL0"],
        "--max-tokens-call1",
        config["PASTE_PREFIX_MAX_TOKENS_CALL1"],
        "--max-tokens-call2",
        config["PASTE_PREFIX_MAX_TOKENS_CALL2"],
        "--max-model-len",
        config["VLLM_MAX_MODEL_LEN"],
        "--request-timeout-s",
        config["PASTE_PREFIX_REQUEST_TIMEOUT_S"],
        "--queue-sample-interval-s",
        config["PASTE_PREFIX_QUEUE_SAMPLE_INTERVAL_S"],
    ]


def _run_logged(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        completed = subprocess.run(
            list(command),
            cwd=REPOSITORY_ROOT,
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return completed.returncode


def _run_cell(
    *,
    run_root: Path,
    config_path: Path,
    config: Mapping[str, str],
    bindings: Mapping[str, str],
    workload: Path,
    model_snapshot: Path,
    python: Path,
    block_number: int,
    order_index: int,
    cell_id: str,
) -> None:
    _verify_bindings(bindings)
    block_id = f"{run_root.name}-block-{block_number}"
    cell_root = run_root / f"block-{block_number:02d}" / cell_id
    cell_root.mkdir(parents=True, exist_ok=False)
    state_dir = cell_root / "state"
    server_dir = cell_root / "server"
    state_dir.mkdir()
    server_dir.mkdir()
    native_pythonpath = cell_root / "native_pythonpath"
    native_pythonpath.mkdir()
    evidence_dir = cell_root / "evidence"
    lifecycle_stdout = cell_root / "server_lifecycle.stdout.log"
    lifecycle_stderr = cell_root / "server_lifecycle.stderr.log"
    runner_stdout = cell_root / "runner.stdout.log"
    runner_stderr = cell_root / "runner.stderr.log"
    server_instance_id = str(uuid.uuid4())
    env = _cell_environment(
        config,
        cell_id=cell_id,
        state_dir=state_dir,
        server_dir=server_dir,
        model_snapshot=model_snapshot,
        native_pythonpath=native_pythonpath,
    )
    command = _cell_command(
        python=python,
        config=config,
        workload=workload,
        model_snapshot=model_snapshot,
        output_dir=evidence_dir,
        block_id=block_id,
        order_index=order_index,
        cell_id=cell_id,
        server_instance_id=server_instance_id,
    )
    effective = {
        "schema": EFFECTIVE_CONFIG_SCHEMA,
        "version": 2,
        "block_id": block_id,
        "block_number": block_number,
        "cell_id": cell_id,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "fresh_server_required": True,
        "native_prefix_cache_enabled": cell_id == "P1",
        "scheduler_policy": "fcfs",
        "native_pythonpath_isolated": True,
        "native_pythonpath": str(native_pythonpath),
        "explicit_prefix_locality_enabled": False,
        "external_network_allowed": False,
        "external_tools_allowed": False,
        "result_cache_reuse_allowed": False,
        "environment": _recorded_environment(env),
        "runner_arguments": command[2:],
        "frozen_config": {
            "path": repository_relative(config_path),
            "sha256": sha256_file(config_path),
        },
    }
    write_json_atomic(cell_root / "effective_config.json", effective)

    print(
        f"[{block_id}] starting {cell_id} ({order_index + 1}/2), "
        f"native_prefix_cache={cell_id == 'P1'}",
        flush=True,
    )
    started = False
    primary_error: RunnerError | None = None
    try:
        if _run_logged(
            [str(START_SERVER)],
            env=env,
            stdout_path=lifecycle_stdout,
            stderr_path=lifecycle_stderr,
        ) != 0:
            raise RunnerError(f"{block_id}/{cell_id} fresh vLLM start failed")
        started = True
        server_identity = _capture_server_identity(
            state_dir=state_dir,
            port=config["VLLM_PORT"],
            server_instance_id=server_instance_id,
        )
        write_json_atomic(cell_root / "server_identity.json", server_identity)
        if _run_logged(
            command,
            env=env,
            stdout_path=runner_stdout,
            stderr_path=runner_stderr,
        ) != 0:
            raise RunnerError(f"{block_id}/{cell_id} prompt cell failed")
    except RunnerError as exc:
        primary_error = exc
    finally:
        if started:
            if _run_logged(
                [str(STOP_SERVER)],
                env=env,
                stdout_path=lifecycle_stdout,
                stderr_path=lifecycle_stderr,
            ) != 0 and primary_error is None:
                primary_error = RunnerError(
                    f"{block_id}/{cell_id} vLLM did not stop cleanly"
                )
    if primary_error is not None:
        raise primary_error

    required = (
        cell_root / "effective_config.json",
        cell_root / "server_identity.json",
        evidence_dir / "result.json",
        evidence_dir / "queue_timeline.jsonl",
        evidence_dir / "metrics_before.prom",
        evidence_dir / "metrics_after.prom",
        server_dir / f"vllm_{config['VLLM_PORT']}.log",
        lifecycle_stdout,
        lifecycle_stderr,
        runner_stdout,
        runner_stderr,
    )
    for path in required:
        if not path.is_file():
            raise RunnerError(f"{block_id}/{cell_id} evidence is missing: {path}")
    evidence_names = (
        "effective_config.json",
        "server_identity.json",
        "evidence/result.json",
        "evidence/queue_timeline.jsonl",
        "evidence/metrics_before.prom",
        "evidence/metrics_after.prom",
        "server/vllm_8100.log",
        "server_lifecycle.stdout.log",
        "server_lifecycle.stderr.log",
        "runner.stdout.log",
        "runner.stderr.log",
    )
    manifest = {
        "schema": CELL_MANIFEST_SCHEMA,
        "version": 2,
        "block_id": block_id,
        "cell_id": cell_id,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "evidence": {
            name: sha256_file(cell_root / name) for name in evidence_names
        },
    }
    write_json_atomic(cell_root / "cell_manifest.json", manifest)
    _verify_bindings(bindings)
    print(f"[{block_id}] {cell_id} completed; fresh server stopped", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate frozen inputs without creating output or touching server/GPU.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_tag) is None:
        raise RunnerError("run tag contains unsupported characters")
    config_path = args.config.resolve()
    repository_relative(config_path)
    config = load_frozen_config(config_path)
    if config["PASTE_PREFIX_CAUSAL_ORDERS"] != "P0,P1;P1,P0":
        raise RunnerError("reverse-block order changed")
    matrix = _matrix_from_config(config)
    thresholds = _thresholds_from_config(config)
    workload = (REPOSITORY_ROOT / config["PASTE_PREFIX_CAUSAL_WORKLOAD"]).resolve()
    sources, payload = load_sources(
        workload,
        expected_sha256=WORKLOAD_SHA256,
        expected_count=matrix["source_count"],
    )
    if len(sources) != 16 or payload.get("formal_eligible") is not False:
        raise RunnerError("development workload validation failed")
    python = Path(config["PASTE_ENV_PREFIX"]) / "bin/python"
    if not python.is_file():
        raise RunnerError(f"reproduction Python is missing: {python}")
    model_snapshot = _model_snapshot(config)
    if not model_snapshot.is_dir() or not (model_snapshot / "config.json").is_file():
        raise RunnerError(f"pinned local model snapshot is missing: {model_snapshot}")
    fixture_preflight = _preflight_fixture(
        config=config,
        sources=sources,
        model_snapshot=model_snapshot,
    )
    if matrix["max_active_tasks"] >= int(config["VLLM_MAX_NUM_SEQS"]):
        raise RunnerError("max-num-seqs must be strictly above offered concurrency")
    bound_paths = (
        config_path,
        workload,
        Path(__file__).resolve(),
        CELL_RUNNER,
        VALIDATOR,
        START_SERVER,
        STOP_SERVER,
        PROTOCOL,
    )
    bindings = _relative_bindings(bound_paths)
    engine = _engine_plan(config)
    contract_bindings = {
        "protocol": {
            "version": "native-prefix-causal-v2",
            "path": repository_relative(PROTOCOL),
            "sha256": bindings[repository_relative(PROTOCOL)],
        },
        "validator": {
            "schema": "paste_repro.native_prefix_causal_validation_v2",
            "path": repository_relative(VALIDATOR),
            "sha256": bindings[repository_relative(VALIDATOR)],
        },
        "cell_runner": {
            "schema": "paste_repro.native_prefix_prompt_cell_v2",
            "path": repository_relative(CELL_RUNNER),
            "sha256": bindings[repository_relative(CELL_RUNNER)],
        },
        "prior_r1_disposition": "rejected_diagnostic_not_validatable_as_v2",
    }
    if args.check_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "check_only": True,
                    "gpu_or_server_touched": False,
                    "external_network_touched": False,
                    "output_created": False,
                    "benchmark_output_created": False,
                    "run_tag": args.run_tag,
                    "profile": config["PASTE_PREFIX_CAUSAL_PROFILE"],
                    "orders": [list(order) for order in ORDERS],
                    "matrix": matrix,
                    "thresholds": thresholds,
                    "engine": engine,
                    "workload": {
                        "path": repository_relative(workload),
                        "sha256": WORKLOAD_SHA256,
                        "source_count": len(sources),
                        "formal_eligible": False,
                    },
                    "config_sha256": sha256_file(config_path),
                    "bindings": bindings,
                    "contract_bindings": contract_bindings,
                    "fixture_preflight": fixture_preflight,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    run_base = (REPOSITORY_ROOT / config["PASTE_PREFIX_CAUSAL_RUN_BASE"]).resolve()
    repository_relative(run_base)
    run_root = run_base / args.run_tag
    lock_path = run_base / f".{args.run_tag}.lock"
    if run_root.exists() or lock_path.exists():
        raise RunnerError(f"run output or lock already exists: {run_root}")
    run_base.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise RunnerError(f"another process reserved run tag {args.run_tag}") from exc
    try:
        run_root.mkdir()
        shutil.copy2(config_path, run_root / "frozen_config.env")
        plan = {
            "schema": PLAN_SCHEMA,
            "version": 2,
            "created_wall_s": time.time(),
            "run_tag": args.run_tag,
            "profile": config["PASTE_PREFIX_CAUSAL_PROFILE"],
            "development_only": True,
            "formal_evidence": False,
            "prospective_version": "native-prefix-causal-v2",
            "prior_r1_disposition": "rejected_diagnostic_not_validatable_as_v2",
            "orders": [list(order) for order in ORDERS],
            "fresh_server_per_cell": True,
            "cross_cell_cache_reuse": False,
            "external_network_allowed": False,
            "external_tools_allowed": False,
            "only_treatment_variable": "VLLM_ENABLE_PREFIX_CACHING",
            "native_pythonpath_isolated": True,
            "explicit_prefix_locality_enabled": False,
            "pinned_vllm_version": "0.10.1",
            "fixture_preflight": fixture_preflight,
            "generation_contract": fixture_preflight["sentinel_contract"],
            "matrix": matrix,
            "thresholds": thresholds,
            "engine": engine,
            "workload": {
                "path": repository_relative(workload),
                "sha256": WORKLOAD_SHA256,
            },
            "bindings": bindings,
            "contract_bindings": contract_bindings,
        }
        write_json_atomic(run_root / "run_plan.json", plan)
        for block_number, order in enumerate(ORDERS, 1):
            for order_index, cell_id in enumerate(order):
                _run_cell(
                    run_root=run_root,
                    config_path=config_path,
                    config=config,
                    bindings=bindings,
                    workload=workload,
                    model_snapshot=model_snapshot,
                    python=python,
                    block_number=block_number,
                    order_index=order_index,
                    cell_id=cell_id,
                )
        _verify_bindings(bindings)
        validation = validate_run(run_root)
        write_json_atomic(run_root / "strict_validation.json", validation)
        print(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "promotion_passed": validation["promotion_passed"],
                    "selected_policy": validation["selected_policy"],
                    "effects_P0_to_P1": validation["effects_P0_to_P1"],
                    "promotion_gates": validation["promotion_gates"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if lock_path.is_dir():
            lock_path.rmdir()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunnerError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
