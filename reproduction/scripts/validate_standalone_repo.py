#!/usr/bin/env python3
"""Validate a standalone PASTE-Qwen-DR export and optionally run CPU smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence


DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "requirements-cpu.txt",
    "pyproject.toml",
    "reproduction/README.md",
    "reproduction/paste_repro/tool_prediction.py",
    "reproduction/paste_repro/online_learned_agent.py",
    "reproduction/paste_repro/live_agent.py",
    "reproduction/paste_repro/live_broker.py",
    "reproduction/paste_repro/pipeline.py",
    "reproduction/results/tool_only/url_rank_mapper.json",
    "reproduction/scripts/run_speculative_tool_execution.sh",
    "reproduction/scripts/run_online_speculative_execution.py",
    "reproduction/scripts/run_live_joint_formal_v9_matrix.py",
    "scripts/run_live_tool_llm_experiment.py",
    "scripts/run_online_trace_learned_experiment.py",
    "scripts/run_vllm_trace_experiment.py",
    "scripts/trace_experiment_lib.py",
    "scripts/online_session_predictor.py",
    "scripts/pythonhooks/sched_policy_patch.py",
)
RUNTIME_SCAN_ROOTS = (
    "reproduction/paste_repro",
    "reproduction/scripts",
    "reproduction/configs",
    "scripts",
)
FORBIDDEN_SOURCE_PATH = "/".join(
    ("", "home", "aiscuser", "Qwen-DeepResearch-PASTE")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.as_posix() == "standalone-manifest.json":
            continue
        yield path


def _validate_manifest(root: Path) -> dict[str, Any]:
    path = root / "standalone-manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "paste_qwen_dr.standalone_manifest":
        raise ValueError("unexpected standalone manifest schema")
    if raw.get("version") != 1 or raw.get("repository_name") != "PASTE-Qwen-DR":
        raise ValueError("unexpected standalone manifest identity")
    rows = raw.get("files")
    if not isinstance(rows, list):
        raise ValueError("standalone manifest files must be a list")
    expected: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("standalone manifest entry must be an object")
        relative = row.get("path")
        size = row.get("size")
        checksum = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(size, int)
            or not isinstance(checksum, str)
            or relative in expected
        ):
            raise ValueError("invalid or duplicate standalone manifest entry")
        expected[relative] = (size, checksum)
    actual = {
        path.relative_to(root).as_posix(): path for path in _manifest_files(root)
    }
    if set(actual) != set(expected):
        raise ValueError(
            "standalone manifest file set mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for relative, path in actual.items():
        size, checksum = expected[relative]
        if path.stat().st_size != size or _sha256(path) != checksum:
            raise ValueError(f"standalone manifest checksum mismatch: {relative}")
    if raw.get("file_count") != len(rows):
        raise ValueError("standalone manifest file_count mismatch")
    return {"file_count": len(rows), "verified": True}


def _validate_runtime_paths(root: Path) -> int:
    scanned = 0
    for relative_root in RUNTIME_SCAN_ROOTS:
        scan_root = root / relative_root
        if not scan_root.is_dir():
            raise FileNotFoundError(f"runtime source directory missing: {scan_root}")
        for path in scan_root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".sh", ".example"}:
                continue
            scanned += 1
            if FORBIDDEN_SOURCE_PATH in path.read_text(encoding="utf-8"):
                raise ValueError(f"source-repository absolute path leaked into {path}")
    return scanned


def _run(command: Sequence[str], *, root: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        list(command),
        cwd=root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _smoke(root: Path) -> dict[str, Any]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root / "reproduction") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    _run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "reproduction/paste_repro",
            "reproduction/scripts/run_online_speculative_execution.py",
            "scripts/run_live_tool_llm_experiment.py",
            "scripts/run_online_trace_learned_experiment.py",
            "scripts/trace_experiment_lib.py",
        ],
        root=root,
        env=env,
    )
    for pattern in ("test_mapper.py", "test_tool_prediction.py", "test_scheduler.py"):
        _run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "reproduction/tests",
                "-p",
                pattern,
                "-q",
            ],
            root=root,
            env=env,
        )
    with tempfile.TemporaryDirectory(prefix="paste-standalone-smoke-") as directory:
        result_path = Path(directory) / "result.json"
        model_path = Path(directory) / "mapper.json"
        _run(
            [
                sys.executable,
                "-m",
                "paste_repro.cli",
                "run-speculative-tools",
                "--limit",
                "2",
                "--report-out",
                str(result_path),
                "--model-out",
                str(model_path),
            ],
            root=root,
            env=env,
        )
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if report.get("schema") != "paste_repro.speculative_tool_execution":
            raise ValueError("standalone trace smoke emitted an unexpected schema")
        dry_run = _run(
            [
                sys.executable,
                "reproduction/scripts/run_online_speculative_execution.py",
                "--output-dir",
                str(Path(directory) / "online"),
                "--dry-run",
            ],
            root=root,
            env=env,
        )
        delegated = json.loads(dry_run)
        command = delegated.get("command", [])
        if "--visit-prediction-model" not in command or "autonomous" not in command:
            raise ValueError("online smoke did not preserve the learned causal contract")
    return {
        "compile": True,
        "unit_test_modules": 3,
        "trace_examples": report.get("replayed_examples"),
        "online_dry_run": True,
    }


def validate(root: Path, *, require_manifest: bool, smoke: bool) -> dict[str, Any]:
    root = root.resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"standalone required files missing: {missing}")
    trace_count = len(list((root / "traces" / "my_traces").glob("*.jsonl")))
    if trace_count != 100:
        raise ValueError(f"expected 100 standalone traces, found {trace_count}")
    manifest_path = root / "standalone-manifest.json"
    if require_manifest and not manifest_path.is_file():
        raise FileNotFoundError("standalone-manifest.json is required")
    result: dict[str, Any] = {
        "schema": "paste_qwen_dr.standalone_validation",
        "version": 1,
        "repository_root": str(root),
        "required_files": len(REQUIRED_FILES),
        "trace_count": trace_count,
        "runtime_files_scanned": _validate_runtime_paths(root),
        "manifest": (
            _validate_manifest(root) if manifest_path.is_file() else {"verified": False}
        ),
    }
    result["smoke"] = _smoke(root) if smoke else {"run": False}
    result["valid"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate(
        args.repository_root,
        require_manifest=args.require_manifest,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
