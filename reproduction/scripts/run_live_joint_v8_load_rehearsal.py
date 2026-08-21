#!/usr/bin/env python3
"""Run one fresh native-FCFS v8 load rehearsal on frozen tune-v1 only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORMAL_RUNNER = (
    REPOSITORY_ROOT / "reproduction/scripts/run_live_joint_formal_matrix.py"
)
VALIDATOR = (
    REPOSITORY_ROOT
    / "reproduction/scripts/validate_live_joint_v8_load_rehearsal.py"
)
CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_formal_v8_matrix.env.example"
)
TUNE_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
)
RUN_BASE = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/v8_load_rehearsal"
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formal = _load_module("formal_v8_runner_for_rehearsal", FORMAL_RUNNER)
validator = _load_module("v8_rehearsal_validator", VALIDATOR)


class RehearsalRunError(RuntimeError):
    """Fail-closed rehearsal runner error."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate frozen local inputs without creating output or touching GPUs.",
    )
    return parser.parse_args(argv)


def _derived_runner_config(config: Mapping[str, str]) -> dict[str, str]:
    derived = dict(config)
    derived.update(
        {
            "PASTE_LIVE_FORMAL_WORKLOAD": formal.repository_relative(TUNE_WORKLOAD),
            "PASTE_LIVE_FORMAL_WORKLOAD_SHA256": validator.TUNE_WORKLOAD_SHA256,
            "PASTE_LIVE_FORMAL_SOURCE_COUNT": "16",
            "PASTE_LIVE_REPLICAS": "5",
            "PASTE_LIVE_MAX_ACTIVE_TASKS": "80",
        }
    )
    return derived


def _bindings(paths: Sequence[Path]) -> dict[str, str]:
    return {
        formal.repository_relative(path): formal.sha256_file(path)
        for path in paths
    }


def _verify_bindings(bindings: Mapping[str, str]) -> None:
    for relative, expected in bindings.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or formal.sha256_file(path) != expected:
            raise RehearsalRunError(f"rehearsal input changed: {relative}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_tag) is None:
        raise RehearsalRunError("invalid rehearsal RUN_TAG")

    config = formal.load_frozen_config(CONFIG)
    python = Path(config["PASTE_ENV_PREFIX"]) / "bin/python"
    if not python.is_file():
        raise RehearsalRunError(f"reproduction Python is missing: {python}")
    formal.validate_entrypoints(python=python)
    workload_validation = validator.validate_development_workload(TUNE_WORKLOAD)
    if (
        workload_validation["formal_evidence_eligible"] is not False
        or workload_validation["file_sha256"]
        == validator.FORMAL_V8_WORKLOAD_SHA256
    ):
        raise RehearsalRunError("formal-v8 workload cannot enter rehearsal")
    model_snapshot = formal._model_snapshot(config)
    if not model_snapshot.is_dir() or not (model_snapshot / "config.json").is_file():
        raise RehearsalRunError(f"pinned model snapshot is missing: {model_snapshot}")
    grammar_feasibility = formal.validate_fixed_final_grammar_feasibility(
        workload=TUNE_WORKLOAD,
        model_snapshot=model_snapshot,
        expected_source_count=16,
    )
    derived = _derived_runner_config(config)
    if not (
        int(derived["PASTE_LIVE_MAX_ACTIVE_TASKS"]) == 80
        and 64 < 80 < int(derived["VLLM_MAX_NUM_SEQS"])
        and int(derived["PASTE_LIVE_REPLICAS"]) == 5
    ):
        raise RehearsalRunError("rehearsal is not the frozen 16x5 nonbinding load")

    bound_paths = (
        Path(__file__).resolve(),
        CONFIG,
        TUNE_WORKLOAD,
        FORMAL_RUNNER,
        VALIDATOR,
        *formal.BOUND_CODE_PATHS[1:],
    )
    bindings = _bindings(bound_paths)
    run_root = RUN_BASE / args.run_tag
    lock_path = RUN_BASE / f".{args.run_tag}.lock"
    if run_root.exists() or lock_path.exists():
        raise RehearsalRunError(f"rehearsal output or lock already exists: {run_root}")

    if args.check_only:
        print(
            json.dumps(
                {
                    "schema": "paste_repro.live_joint_v8_load_rehearsal_check",
                    "version": 1,
                    "valid": True,
                    "check_only": True,
                    "development_only": True,
                    "formal_evidence_eligible": False,
                    "selection_uses_formal_v8_performance": False,
                    "gpu_or_server_touched": False,
                    "network_touched": False,
                    "run_tag": args.run_tag,
                    "workload_validation": workload_validation,
                    "fixed_final_grammar_feasibility": grammar_feasibility,
                    "offered_concurrency": 80,
                    "former_threshold_exceeded": True,
                    "native_sequence_ceiling": 96,
                    "native_sequence_ceiling_nonbinding": True,
                    "bindings": bindings,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    RUN_BASE.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise RehearsalRunError("another process reserved this rehearsal tag") from exc
    try:
        run_root.mkdir()
        server_dir = run_root / "server"
        state_dir = run_root / "state"
        evidence_dir = run_root / "evidence"
        server_dir.mkdir()
        state_dir.mkdir()
        lifecycle_stdout = run_root / "server_lifecycle.stdout.log"
        lifecycle_stderr = run_root / "server_lifecycle.stderr.log"
        runner_stdout = run_root / "runner.stdout.log"
        runner_stderr = run_root / "runner.stderr.log"
        block_id = f"{args.run_tag}-development-block-1"
        server_instance_id = str(uuid.uuid4())
        command = formal._runner_command(
            python=python,
            workload=TUNE_WORKLOAD,
            output=evidence_dir,
            cell="A",
            block_id=block_id,
            order_index=0,
            server_instance_id=server_instance_id,
            config=derived,
        )
        cell_env = formal._cell_environment(config, cell="A")
        cell_env.update(
            {
                "VLLM_REQUIRE_NEW": "1",
                "VLLM_STATE_DIR": str(state_dir),
                "VLLM_LOG_DIR": str(server_dir),
                "VLLM_HOOK_DIR": str(
                    REPOSITORY_ROOT / "scripts/pythonhooks"
                ),
                "MODEL_SNAPSHOT": str(model_snapshot),
                "PYTHONUNBUFFERED": "1",
            }
        )
        effective = {
            "schema": "paste_repro.live_joint_v8_load_rehearsal_plan",
            "version": 1,
            "created_wall_s": time.time(),
            "run_tag": args.run_tag,
            "development_only": True,
            "formal_evidence_eligible": False,
            "selection_uses_formal_v8_performance": False,
            "forbidden_formal_v8_workload_sha256": (
                validator.FORMAL_V8_WORKLOAD_SHA256
            ),
            "cell_id": "A",
            "llm_scheduler": "fcfs",
            "speculation_mode": "off",
            "source_count": 16,
            "replicas": 5,
            "task_count": 80,
            "max_active_tasks": 80,
            "native_sequence_ceiling": 96,
            "fresh_server_required": True,
            "result_cache_empty_required": True,
            "workload_validation": workload_validation,
            "fixed_final_grammar_feasibility": grammar_feasibility,
            "block_id": block_id,
            "server_instance_id": server_instance_id,
            "runner_arguments": command[2:],
            "bindings": bindings,
        }
        formal.write_json_atomic(run_root / "effective_plan.json", effective)
        shutil.copy2(CONFIG, run_root / "frozen_formal_v8_config.env")

        _verify_bindings(bindings)
        started = False
        primary_error: BaseException | None = None
        try:
            start_code = formal._run_logged(
                [str(formal.START_SERVER)],
                env=cell_env,
                stdout_path=lifecycle_stdout,
                stderr_path=lifecycle_stderr,
            )
            if start_code != 0:
                raise RehearsalRunError("fresh vLLM start failed; see logs")
            started = True
            runner_code = formal._run_logged(
                command,
                env=cell_env,
                stdout_path=runner_stdout,
                stderr_path=runner_stderr,
            )
            if runner_code != 0:
                raise RehearsalRunError("live rehearsal runner failed; see logs")
        except BaseException as exc:
            primary_error = exc
        finally:
            if started:
                stop_code = formal._run_logged(
                    [str(formal.STOP_SERVER)],
                    env=cell_env,
                    stdout_path=lifecycle_stdout,
                    stderr_path=lifecycle_stderr,
                )
                if stop_code != 0 and primary_error is None:
                    primary_error = RehearsalRunError("vLLM did not stop cleanly")
        if primary_error is not None:
            raise primary_error

        result_path = evidence_dir / "result.json"
        timeline_path = evidence_dir / "queue_timeline.jsonl"
        server_log = server_dir / f"vllm_{config['VLLM_PORT']}.log"
        for required in (result_path, timeline_path, server_log):
            if not required.is_file():
                raise RehearsalRunError(f"rehearsal evidence is missing: {required}")
        validation = validator.validate_rehearsal_result(
            result_path=result_path,
            timeline_path=timeline_path,
            block_id=block_id,
            server_instance_id=server_instance_id,
        )
        formal.write_json_atomic(run_root / "strict_validation.json", validation)
        _verify_bindings(bindings)
        completion = {
            "schema": "paste_repro.live_joint_v8_load_rehearsal_completion",
            "version": 1,
            "completed_wall_s": time.time(),
            "valid": True,
            "development_only": True,
            "formal_evidence_eligible": False,
            "selection_uses_formal_v8_performance": False,
            "strict_validation": {
                "path": formal.repository_relative(
                    run_root / "strict_validation.json"
                ),
                "sha256": formal.sha256_file(
                    run_root / "strict_validation.json"
                ),
            },
            "evidence": {
                formal.repository_relative(path): formal.sha256_file(path)
                for path in (
                    run_root / "effective_plan.json",
                    result_path,
                    timeline_path,
                    server_log,
                    lifecycle_stdout,
                    lifecycle_stderr,
                    runner_stdout,
                    runner_stderr,
                )
            },
            "bindings": bindings,
        }
        formal.write_json_atomic(run_root / "completed_rehearsal.json", completion)
        print(f"Development-only v8 load rehearsal completed: {run_root}")
        return 0
    except BaseException as exc:
        if run_root.is_dir():
            formal.write_json_atomic(
                run_root / "failure.json",
                {
                    "schema": "paste_repro.live_joint_v8_load_rehearsal_failure",
                    "version": 1,
                    "failed_wall_s": time.time(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
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
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
