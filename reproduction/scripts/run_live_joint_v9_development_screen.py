#!/usr/bin/env python3
"""Run the prospective two-stage v9 development-only live-joint screen."""

from __future__ import annotations

import argparse
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
SCRIPTS_DIR = REPOSITORY_ROOT / "reproduction/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import aggregate_live_joint_v9_development_screen as aggregator  # type: ignore
import run_live_joint_formal_matrix as formal  # type: ignore
import validate_live_joint_v9_development_screen as validator  # type: ignore


CONFIG_PATH = validator.CONFIG_PATH
WORKLOAD = validator.TUNE_WORKLOAD
PROTOCOL = (
    REPOSITORY_ROOT
    / "reproduction/results/live_joint/V9_DEVELOPMENT_SCREEN_PROTOCOL.md"
)
FORMAL_AGGREGATOR = SCRIPTS_DIR / "aggregate_live_joint_four_cell.py"
PAIR_VALIDATOR = SCRIPTS_DIR / "compare_live_joint_pair.py"
AGGREGATOR_PATH = SCRIPTS_DIR / "aggregate_live_joint_v9_development_screen.py"
VALIDATOR_PATH = SCRIPTS_DIR / "validate_live_joint_v9_development_screen.py"
RUN_BASE = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/v9_screen"
)


class DevelopmentScreenRunError(RuntimeError):
    """The prospective runner must stop without widening its protocol."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stage0-only",
        action="store_true",
        help="Run only the baseline transport ladder and persist its selection.",
    )
    mode.add_argument(
        "--resume-stage1",
        action="store_true",
        help="Resume a SHA-bound run whose Stage 0 already selected transport.",
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all local inputs without touching a GPU, server, or network.",
    )
    return parser.parse_args(argv)


def _bindings(paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise DevelopmentScreenRunError(f"bound file is missing: {resolved}")
        relative = formal.repository_relative(resolved)
        result[relative] = formal.sha256_file(resolved)
    return dict(sorted(result.items()))


def _verify_bindings(bindings: Mapping[str, str]) -> None:
    for relative, expected in bindings.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or formal.sha256_file(path) != expected:
            raise DevelopmentScreenRunError(
                f"development-screen binding changed: {relative}"
            )


def _bound_paths() -> tuple[Path, ...]:
    paths = (
        Path(__file__).resolve(),
        CONFIG_PATH,
        WORKLOAD,
        PROTOCOL,
        VALIDATOR_PATH,
        AGGREGATOR_PATH,
        FORMAL_AGGREGATOR,
        PAIR_VALIDATOR,
        *formal.BOUND_CODE_PATHS,
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _derived_config(
    base: Mapping[str, str], *, visit_interval_s: float, cell: str
) -> dict[str, str]:
    values = dict(base)
    values["PASTE_LIVE_VISIT_MIN_START_INTERVAL_S"] = format(
        visit_interval_s, ".1f"
    )
    values["PASTE_LIVE_MIN_SPECULATIVE_TOOL_WORKERS"] = str(
        validator.CELL_TREATMENTS[cell]["min_speculative_tool_workers"]
    )
    return values


def _runner_command(
    *, python: Path, output: Path, cell: str, block_id: str, order_index: int,
    server_instance_id: str, config: Mapping[str, str],
) -> list[str]:
    underlying = str(validator.CELL_TREATMENTS[cell]["formal_cell_id"])
    command = formal._runner_command(
        python=python,
        workload=WORKLOAD,
        output=output,
        cell=underlying,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        config=config,
    )
    label_index = command.index("--cell-label") + 1
    command[label_index] = f"{block_id}-{cell}"
    return command


def _cell_environment(
    base: Mapping[str, str], *, cell: str, state_dir: Path, server_dir: Path,
    model_snapshot: Path,
) -> dict[str, str]:
    underlying = str(validator.CELL_TREATMENTS[cell]["formal_cell_id"])
    environment = formal._cell_environment(base, cell=underlying)
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
    return environment


def _run_cell(
    *, cell_root: Path, stage: str, cell: str, block_id: str,
    order_index: int, visit_interval_s: float, base_config: Mapping[str, str],
    python: Path, model_snapshot: Path, bindings: Mapping[str, str],
) -> dict[str, Any]:
    _verify_bindings(bindings)
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
    config = _derived_config(
        base_config, visit_interval_s=visit_interval_s, cell=cell
    )
    command = _runner_command(
        python=python,
        output=evidence_dir,
        cell=cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        config=config,
    )
    environment = _cell_environment(
        config,
        cell=cell,
        state_dir=state_dir,
        server_dir=server_dir,
        model_snapshot=model_snapshot,
    )
    effective = {
        "schema": "paste_repro.live_joint_v9_development_cell_plan",
        "version": 1,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "stage": stage,
        "cell_id": cell,
        "block_id": block_id,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "fresh_server_required": True,
        "result_cache_empty_required": True,
        "selected_visit_interval_s": visit_interval_s,
        "llm_scheduler": validator.CELL_TREATMENTS[cell]["scheduler"],
        "speculation_mode": validator.CELL_TREATMENTS[cell]["speculation_mode"],
        "min_speculative_tool_workers": validator.CELL_TREATMENTS[cell][
            "min_speculative_tool_workers"
        ],
        "workload": {
            "path": formal.repository_relative(WORKLOAD),
            "sha256": validator.TUNE_WORKLOAD_SHA256,
            "formal_eligible": False,
        },
        "runner_arguments": command[2:],
        "bindings": dict(bindings),
    }
    formal.write_json_atomic(cell_root / "effective_config.json", effective)
    print(
        f"[{block_id}] starting {stage} cell {cell}: "
        f"interval={visit_interval_s:.1f}s, "
        f"min_spec={effective['min_speculative_tool_workers']}",
        flush=True,
    )
    started = False
    primary_error: BaseException | None = None
    try:
        start_code = formal._run_logged(
            [str(formal.START_SERVER)],
            env=environment,
            stdout_path=lifecycle_stdout,
            stderr_path=lifecycle_stderr,
        )
        if start_code != 0:
            raise DevelopmentScreenRunError(
                f"{block_id}/{cell} fresh vLLM start failed"
            )
        started = True
        runner_code = formal._run_logged(
            command,
            env=environment,
            stdout_path=runner_stdout,
            stderr_path=runner_stderr,
        )
        if runner_code != 0:
            raise DevelopmentScreenRunError(
                f"{block_id}/{cell} live runner failed"
            )
    except BaseException as exc:
        primary_error = exc
    finally:
        if started:
            stop_code = formal._run_logged(
                [str(formal.STOP_SERVER)],
                env=environment,
                stdout_path=lifecycle_stdout,
                stderr_path=lifecycle_stderr,
            )
            if stop_code != 0 and primary_error is None:
                primary_error = DevelopmentScreenRunError(
                    f"{block_id}/{cell} vLLM did not stop cleanly"
                )
    if primary_error is not None:
        raise primary_error

    result_path = evidence_dir / "result.json"
    timeline_path = evidence_dir / "queue_timeline.jsonl"
    server_log = server_dir / f"vllm_{config['VLLM_PORT']}.log"
    for required in (result_path, timeline_path, server_log):
        if not required.is_file():
            raise DevelopmentScreenRunError(
                f"{block_id}/{cell} evidence is missing: {required}"
            )
    validation = validator.validate_cell_result(
        result_path=result_path,
        timeline_path=timeline_path,
        cell=cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        visit_interval_s=visit_interval_s,
        stage=stage,
    )
    validation_path = cell_root / "strict_validation.json"
    formal.write_json_atomic(validation_path, validation)
    manifest = {
        "schema": "paste_repro.live_joint_v9_development_cell_evidence",
        "version": 1,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "stage": stage,
        "cell_id": cell,
        "block_id": block_id,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "visit_interval_s": visit_interval_s,
        "accepted": validation["accepted"],
        "evidence": {
            formal.repository_relative(path): formal.sha256_file(path)
            for path in (
                cell_root / "effective_config.json",
                result_path,
                timeline_path,
                validation_path,
                server_log,
                lifecycle_stdout,
                lifecycle_stderr,
                runner_stdout,
                runner_stderr,
            )
        },
        "bindings": dict(bindings),
    }
    formal.write_json_atomic(cell_root / "cell_manifest.json", manifest)
    _verify_bindings(bindings)
    print(f"[{block_id}] cell {cell} completed and server stopped", flush=True)
    return validation


def _run_stage0(
    *, run_root: Path, run_tag: str, base_config: Mapping[str, str],
    python: Path, model_snapshot: Path, bindings: Mapping[str, str],
) -> dict[str, Any]:
    stage_root = run_root / "stage-0"
    stage_root.mkdir(exist_ok=False)
    validations: list[dict[str, Any]] = []
    first = _run_cell(
        cell_root=stage_root / "attempt-01-interval-2p5" / "A",
        stage="stage0",
        cell="A",
        block_id=f"{run_tag}-stage0-a-2p5",
        order_index=0,
        visit_interval_s=2.5,
        base_config=base_config,
        python=python,
        model_snapshot=model_snapshot,
        bindings=bindings,
    )
    validations.append(first)
    if first["accepted"] is not True:
        if first["retry_only_fallback_eligible"] is not True:
            raise DevelopmentScreenRunError(
                "2.5s A failed a non-transport gate; 3.0s fallback forbidden"
            )
        second = _run_cell(
            cell_root=stage_root / "attempt-02-interval-3p0" / "A",
            stage="stage0",
            cell="A",
            block_id=f"{run_tag}-stage0-a-3p0",
            order_index=0,
            visit_interval_s=3.0,
            base_config=base_config,
            python=python,
            model_snapshot=model_snapshot,
            bindings=bindings,
        )
        validations.append(second)
    selection = validator.select_transport_interval(validations)
    formal.write_json_atomic(stage_root / "selected_transport.json", selection)
    attempt_manifests = sorted(stage_root.glob("attempt-*/A/cell_manifest.json"))
    formal.write_json_atomic(
        stage_root / "completed_stage0.json",
        {
            "schema": "paste_repro.live_joint_v9_development_stage0_completion",
            "version": 1,
            "development_only": True,
            "formal_eligible": False,
            "formal_evidence_eligible": False,
            "candidate_performance_observed_or_used": False,
            "selected_transport": {
                "path": formal.repository_relative(
                    stage_root / "selected_transport.json"
                ),
                "sha256": formal.sha256_file(
                    stage_root / "selected_transport.json"
                ),
            },
            "attempt_count": len(validations),
            "attempt_manifests": [
                {
                    "path": formal.repository_relative(path),
                    "sha256": formal.sha256_file(path),
                }
                for path in attempt_manifests
            ],
            "bindings": dict(bindings),
        },
    )
    print(
        "Stage 0 accepted visit interval "
        f"{selection['selected_visit_interval_s']:.1f}s",
        flush=True,
    )
    return selection


def _load_stage0_selection(
    *, run_root: Path, bindings: Mapping[str, str]
) -> dict[str, Any]:
    completion_path = run_root / "stage-0/completed_stage0.json"
    selection_path = run_root / "stage-0/selected_transport.json"
    if not completion_path.is_file() or not selection_path.is_file():
        raise DevelopmentScreenRunError("accepted Stage 0 evidence is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        completion.get("development_only") is not True
        or completion.get("formal_evidence_eligible") is not False
        or completion.get("bindings") != dict(bindings)
        or completion.get("selected_transport", {}).get("sha256")
        != formal.sha256_file(selection_path)
    ):
        raise DevelopmentScreenRunError("Stage 0 completion binding differs")
    attempt_paths = sorted(
        (run_root / "stage-0").glob("attempt-*/A/strict_validation.json")
    )
    manifest_paths = sorted(
        (run_root / "stage-0").glob("attempt-*/A/cell_manifest.json")
    )
    expected_manifest_refs = [
        {
            "path": formal.repository_relative(path),
            "sha256": formal.sha256_file(path),
        }
        for path in manifest_paths
    ]
    if (
        len(manifest_paths) != completion.get("attempt_count")
        or completion.get("attempt_manifests") != expected_manifest_refs
    ):
        raise DevelopmentScreenRunError("Stage 0 attempt-manifest binding differs")
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bindings") != dict(bindings):
            raise DevelopmentScreenRunError("Stage 0 cell code binding differs")
        evidence = manifest.get("evidence")
        if not isinstance(evidence, Mapping):
            raise DevelopmentScreenRunError("Stage 0 cell evidence map is missing")
        for relative, expected_sha in evidence.items():
            path = REPOSITORY_ROOT / str(relative)
            if not path.is_file() or formal.sha256_file(path) != expected_sha:
                raise DevelopmentScreenRunError(
                    f"Stage 0 evidence binding changed: {relative}"
                )
    attempts = [json.loads(path.read_text(encoding="utf-8")) for path in attempt_paths]
    recomputed = validator.select_transport_interval(attempts)
    if selection != recomputed:
        raise DevelopmentScreenRunError("Stage 0 transport selection is not replayable")
    return selection


def _run_stage1(
    *, run_root: Path, run_tag: str, selection: Mapping[str, Any],
    base_config: Mapping[str, str], python: Path, model_snapshot: Path,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    stage_root = run_root / "stage-1"
    if stage_root.exists():
        raise DevelopmentScreenRunError("Stage 1 output already exists")
    stage_root.mkdir()
    interval = float(selection["selected_visit_interval_s"])
    block_inputs: list[tuple[str, Mapping[str, Path]]] = []
    for block_number, order in enumerate(aggregator.EXPECTED_ORDERS, 1):
        block_id = f"{run_tag}-stage1-block-{block_number}"
        paths: dict[str, Path] = {}
        for order_index, cell in enumerate(order):
            cell_root = stage_root / f"block-{block_number:02d}" / cell
            validation = _run_cell(
                cell_root=cell_root,
                stage="stage1",
                cell=cell,
                block_id=block_id,
                order_index=order_index,
                visit_interval_s=interval,
                base_config=base_config,
                python=python,
                model_snapshot=model_snapshot,
                bindings=bindings,
            )
            if validation["accepted"] is not True:
                raise DevelopmentScreenRunError(
                    f"{block_id}/{cell} strict validation did not accept"
                )
            paths[cell] = cell_root / "evidence/result.json"
        block_inputs.append((block_id, paths))
    _verify_bindings(bindings)
    result = aggregator.aggregate_development_screen(
        block_inputs,
        selected_visit_interval_s=interval,
    )
    aggregate_path = run_root / "strict_development_selection.json"
    formal.write_json_atomic(aggregate_path, result)
    _verify_bindings(bindings)
    formal.write_json_atomic(
        run_root / "completed_screen.json",
        {
            "schema": "paste_repro.live_joint_v9_development_screen_completion",
            "version": 1,
            "completed_wall_s": time.time(),
            "development_only": True,
            "formal_eligible": False,
            "formal_evidence_eligible": False,
            "selected_policy": result["selected_policy"],
            "development_selection_passed": result[
                "development_selection_passed"
            ],
            "selected_transport": {
                "path": formal.repository_relative(
                    run_root / "stage-0/selected_transport.json"
                ),
                "sha256": formal.sha256_file(
                    run_root / "stage-0/selected_transport.json"
                ),
            },
            "strict_development_selection": {
                "path": formal.repository_relative(aggregate_path),
                "sha256": formal.sha256_file(aggregate_path),
            },
            "bindings": dict(bindings),
        },
    )
    return result


def _preflight(
    *, run_tag: str, config: Mapping[str, str], python: Path,
    model_snapshot: Path, bindings: Mapping[str, str],
) -> dict[str, Any]:
    formal.validate_entrypoints(python=python)
    workload_validation = validator.validate_development_workload(WORKLOAD)
    grammar = formal.validate_fixed_final_grammar_feasibility(
        workload=WORKLOAD,
        model_snapshot=model_snapshot,
        expected_source_count=validator.SOURCE_COUNT,
    )
    if not (
        int(config["PASTE_LIVE_MAX_ACTIVE_TASKS"]) == validator.TASK_COUNT
        and 64 < validator.TASK_COUNT < int(config["VLLM_MAX_NUM_SEQS"]) == 96
        and int(config["PASTE_LIVE_REPLICAS"]) == validator.REPLICAS
        and config["VLLM_ENABLE_PREFIX_CACHING"] == "1"
        and config["VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY"] == "0"
    ):
        raise DevelopmentScreenRunError("80-offered native-prefix load differs")
    return {
        "schema": "paste_repro.live_joint_v9_development_screen_check",
        "version": 1,
        "valid": True,
        "check_only": True,
        "development_only": True,
        "formal_eligible": False,
        "formal_evidence_eligible": False,
        "gpu_or_server_touched": False,
        "network_touched": False,
        "run_tag": run_tag,
        "stage0_transport_ladder_s": list(validator.TRANSPORT_LADDER_S),
        "stage0_fallback_trigger": (
            "only the composite transport retry gate failed at 2.5s"
        ),
        "stage1_orders": [list(order) for order in aggregator.EXPECTED_ORDERS],
        "workload_validation": workload_validation,
        "fixed_final_grammar_feasibility": grammar,
        "offered_concurrency": validator.TASK_COUNT,
        "native_sequence_ceiling": 96,
        "native_prefix_caching": True,
        "explicit_prefix_locality": False,
        "frozen_live_broker_sha256": validator.EXPECTED_LIVE_BROKER_SHA256,
        "bindings": dict(bindings),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_tag) is None:
        raise DevelopmentScreenRunError("invalid RUN_TAG")
    config = validator.load_frozen_config(CONFIG_PATH)
    python = Path(config["PASTE_ENV_PREFIX"]) / "bin/python"
    if not python.is_file():
        raise DevelopmentScreenRunError(f"reproduction Python missing: {python}")
    model_snapshot = formal._model_snapshot(config)
    if not model_snapshot.is_dir() or not (model_snapshot / "config.json").is_file():
        raise DevelopmentScreenRunError(
            f"pinned model snapshot missing: {model_snapshot}"
        )
    bindings = _bindings(_bound_paths())
    preflight = _preflight(
        run_tag=args.run_tag,
        config=config,
        python=python,
        model_snapshot=model_snapshot,
        bindings=bindings,
    )
    run_root = RUN_BASE / args.run_tag
    lock_path = RUN_BASE / f".{args.run_tag}.lock"
    if args.check_only:
        if run_root.exists() or lock_path.exists():
            raise DevelopmentScreenRunError(
                f"check-only tag already exists or is reserved: {run_root}"
            )
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    RUN_BASE.mkdir(parents=True, exist_ok=True)
    if args.resume_stage1:
        if not run_root.is_dir():
            raise DevelopmentScreenRunError("resume run root does not exist")
        plan_path = run_root / "run_plan.json"
        if not plan_path.is_file():
            raise DevelopmentScreenRunError("resume run plan is missing")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("bindings") != bindings:
            raise DevelopmentScreenRunError(
                "resume code/config bindings differ from Stage 0"
            )
        if (run_root / "completed_screen.json").exists():
            raise DevelopmentScreenRunError("screen already completed")
    else:
        if run_root.exists() or lock_path.exists():
            raise DevelopmentScreenRunError(
                f"run output or lock already exists: {run_root}"
            )
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise DevelopmentScreenRunError("another process reserved this tag") from exc
    try:
        if not args.resume_stage1:
            run_root.mkdir()
            shutil.copy2(CONFIG_PATH, run_root / "frozen_development_config.env")
            formal.write_json_atomic(
                run_root / "run_plan.json",
                {
                    "schema": "paste_repro.live_joint_v9_development_screen_plan",
                    "version": 1,
                    "created_wall_s": time.time(),
                    "run_tag": args.run_tag,
                    "development_only": True,
                    "formal_eligible": False,
                    "formal_evidence_eligible": False,
                    "stage0_transport_ladder_s": list(
                        validator.TRANSPORT_LADDER_S
                    ),
                    "transport_selection_uses_candidate_performance": False,
                    "stage1_orders": [
                        list(order) for order in aggregator.EXPECTED_ORDERS
                    ],
                    "fresh_server_per_cell": True,
                    "cross_cell_result_cache": False,
                    "workload_validation": preflight["workload_validation"],
                    "fixed_final_grammar_feasibility": preflight[
                        "fixed_final_grammar_feasibility"
                    ],
                    "bindings": bindings,
                },
            )
            selection = _run_stage0(
                run_root=run_root,
                run_tag=args.run_tag,
                base_config=config,
                python=python,
                model_snapshot=model_snapshot,
                bindings=bindings,
            )
            if args.stage0_only:
                print(f"Stage-0-only run completed: {run_root}", flush=True)
                return 0
        else:
            _verify_bindings(bindings)
            selection = _load_stage0_selection(
                run_root=run_root, bindings=bindings
            )
        result = _run_stage1(
            run_root=run_root,
            run_tag=args.run_tag,
            selection=selection,
            base_config=config,
            python=python,
            model_snapshot=model_snapshot,
            bindings=bindings,
        )
        print(
            f"Development screen completed: {run_root}; "
            f"selected={result['selected_policy']}",
            flush=True,
        )
        return 0 if result["development_selection_passed"] else 1
    except BaseException as exc:
        if run_root.is_dir():
            formal.write_json_atomic(
                run_root / "failure.json",
                {
                    "schema": "paste_repro.live_joint_v9_development_screen_failure",
                    "version": 1,
                    "failed_wall_s": time.time(),
                    "development_only": True,
                    "formal_eligible": False,
                    "formal_evidence_eligible": False,
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
    except (DevelopmentScreenRunError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
