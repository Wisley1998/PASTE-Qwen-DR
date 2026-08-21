#!/usr/bin/env python3
"""Export the supported reproduction as a standalone PASTE-Qwen-DR repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION = SOURCE_ROOT.parent / "PASTE-Qwen-DR"
EXCLUDED_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache"}
EXCLUDED_REPRODUCTION_TOP_LEVEL = {"artifacts", "logs", "run"}
ROOT_FILES = ("LICENSE", "requirements.txt")
ROOT_SCRIPT_FILES = (
    "scripts/online_session_predictor.py",
    "scripts/run_live_tool_llm_experiment.py",
    "scripts/run_online_trace_learned_experiment.py",
    "scripts/run_vllm_trace_experiment.py",
    "scripts/trace_experiment_lib.py",
    "scripts/pythonhooks/sched_policy_patch.py",
    "scripts/pythonhooks/sitecustomize.py",
)
ROOT_TEST_FILES = ("tests/test_learned_tool_overlap.py",)
CHECKED_IN_EVIDENCE = (
    "reproduction/artifacts/live_joint/development/v9_screen/v9-screen-r1/completed_screen.json",
    "reproduction/artifacts/live_joint/development/v9_screen/v9-screen-r1/stage-0/selected_transport.json",
    "reproduction/artifacts/live_joint/development/v9_screen/v9-screen-r1/strict_development_selection.json",
    "reproduction/artifacts/live_joint/formal/formal-v9-context10k-live-r1/completed_matrix.json",
    "reproduction/artifacts/live_joint/formal/formal-v9-context10k-live-r1/strict_four_cell_aggregate.json",
    "reproduction/artifacts/live_joint/prefix_native_causal_dev_v2/native-prefix-v2-r1/strict_validation.json",
)
TEMPLATE_MAPPINGS = {
    "reproduction/standalone/README.md": "README.md",
    "reproduction/standalone/.gitignore": ".gitignore",
    "reproduction/standalone/requirements-cpu.txt": "requirements-cpu.txt",
    "reproduction/standalone/pyproject.toml": "pyproject.toml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"standalone source file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _tree_files(
    root: Path, *, excluded_top_level: set[str] | None = None
) -> Iterable[Path]:
    top_level = excluded_top_level or set()
    for current_raw, directory_names, file_names in os.walk(root):
        current = Path(current_raw)
        relative_directory = current.relative_to(root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in EXCLUDED_DIRECTORY_NAMES
            and not (relative_directory == Path(".") and name in top_level)
        )
        for name in sorted(file_names):
            path = current / name
            if path.is_symlink() or path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def _copy_reproduction(destination: Path) -> None:
    source = SOURCE_ROOT / "reproduction"
    for path in _tree_files(
        source, excluded_top_level=EXCLUDED_REPRODUCTION_TOP_LEVEL
    ):
        relative = path.relative_to(source)
        _copy_file(path, destination / "reproduction" / relative)


def _copy_trace_snapshot(destination: Path) -> None:
    source = SOURCE_ROOT / "traces" / "my_traces"
    trace_files = sorted(source.glob("*.jsonl"))
    if len(trace_files) != 100:
        raise RuntimeError(f"expected 100 trace files, found {len(trace_files)}")
    for path in trace_files:
        _copy_file(path, destination / "traces" / "my_traces" / path.name)


def _write_manifest(destination: Path) -> None:
    rows = []
    for path in _tree_files(destination):
        if ".git" in path.relative_to(destination).parts:
            continue
        relative = path.relative_to(destination).as_posix()
        if relative == "standalone-manifest.json":
            continue
        rows.append(
            {"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    payload = {
        "schema": "paste_qwen_dr.standalone_manifest",
        "version": 1,
        "repository_name": "PASTE-Qwen-DR",
        "file_count": len(rows),
        "files": rows,
    }
    (destination / "standalone-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _remove_generated_caches(destination: Path) -> None:
    for name in ("__pycache__", ".pytest_cache"):
        for path in destination.rglob(name):
            if path.is_dir():
                shutil.rmtree(path)


def export(destination: Path, *, initialize_git: bool, validate: bool) -> None:
    destination = destination.resolve()
    source = SOURCE_ROOT.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("destination must be outside the source repository")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=destination.parent)
    )
    completed = False
    try:
        _copy_reproduction(temporary)
        _copy_trace_snapshot(temporary)
        for relative in (*ROOT_FILES, *ROOT_SCRIPT_FILES, *ROOT_TEST_FILES):
            _copy_file(SOURCE_ROOT / relative, temporary / relative)
        for relative in CHECKED_IN_EVIDENCE:
            _copy_file(SOURCE_ROOT / relative, temporary / relative)
        for source_relative, destination_relative in TEMPLATE_MAPPINGS.items():
            _copy_file(
                SOURCE_ROOT / source_relative, temporary / destination_relative
            )
        _write_manifest(temporary)

        if validate:
            subprocess.run(
                [
                    sys.executable,
                    str(temporary / "reproduction/scripts/validate_standalone_repo.py"),
                    "--repository-root",
                    str(temporary),
                    "--require-manifest",
                    "--smoke",
                ],
                cwd=temporary,
                check=True,
            )
            _remove_generated_caches(temporary)
        if initialize_git:
            subprocess.run(
                ["git", "init", "--initial-branch=main", str(temporary)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        os.replace(temporary, destination)
        completed = True
    finally:
        if not completed:
            shutil.rmtree(temporary, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--no-git", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export(
        args.output,
        initialize_git=not args.no_git,
        validate=not args.no_validate,
    )
    print(f"Exported standalone repository: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
