#!/usr/bin/env python3
"""Verify the checksummed session split and expose it as read-only symlinks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.mapper import load_artifact, write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify every raw trace against the learned mapper manifest and create "
            "non-overlapping train/eval symlink directories."
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=REPOSITORY_ROOT / "traces" / "my_traces",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRODUCTION_ROOT / "artifacts" / "trace_splits",
    )
    parser.add_argument(
        "--held-out-limit",
        type=int,
        default=16,
        help="number of held-out sessions exposed for the live smoke (0 means all)",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_entries(raw: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"artifact {label} must be a non-empty list")
    entries: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"artifact {label} entry must be an object")
        session_id = item.get("session_id")
        checksum = item.get("sha256")
        if (
            not isinstance(session_id, str)
            or not session_id
            or Path(session_id).name != session_id
        ):
            raise ValueError(f"unsafe session id in artifact {label}: {session_id!r}")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError(f"invalid SHA-256 for {session_id}")
        entries.append({"session_id": session_id, "sha256": checksum})
    return entries


def verify_sources(
    trace_dir: Path, entries: Iterable[dict[str, str]]
) -> list[tuple[dict[str, str], Path]]:
    verified: list[tuple[dict[str, str], Path]] = []
    for entry in entries:
        source = trace_dir / entry["session_id"]
        if not source.is_file():
            raise FileNotFoundError(f"trace from artifact is missing: {source}")
        actual = file_sha256(source)
        if actual != entry["sha256"]:
            raise ValueError(
                f"trace checksum mismatch for {entry['session_id']}: "
                f"{actual} != {entry['sha256']}"
            )
        verified.append((entry, source.resolve()))
    return verified


def ensure_symlink_directory(
    directory: Path, verified: Iterable[tuple[dict[str, str], Path]]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    for entry, source in verified:
        name = entry["session_id"]
        expected.add(name)
        destination = directory / name
        if destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"existing symlink points elsewhere: {destination}")
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to replace existing path: {destination}")
        destination.symlink_to(source)

    unexpected = sorted(path.name for path in directory.iterdir() if path.name not in expected)
    if unexpected:
        raise RuntimeError(
            f"split directory contains unexpected entries: {directory}: {unexpected}"
        )


def main() -> int:
    args = parse_args()
    if args.held_out_limit < 0:
        raise SystemExit("--held-out-limit must be non-negative")

    _, artifact = load_artifact(args.artifact)
    split = artifact["training_split"]
    train_entries = validate_entries(split.get("train_sessions"), "train_sessions")
    held_out_entries = validate_entries(
        split.get("held_out_sessions"), "held_out_sessions"
    )
    train_ids = {entry["session_id"] for entry in train_entries}
    held_out_ids = {entry["session_id"] for entry in held_out_entries}
    overlap = sorted(train_ids & held_out_ids)
    if overlap:
        raise ValueError(f"train and held-out sessions overlap: {overlap}")

    selected_held_out = (
        held_out_entries
        if args.held_out_limit == 0
        else held_out_entries[: args.held_out_limit]
    )
    if not selected_held_out:
        raise ValueError("held-out selection is empty")

    trace_dir = args.trace_dir.resolve()
    verified_train = verify_sources(trace_dir, train_entries)
    verified_eval = verify_sources(trace_dir, selected_held_out)
    artifact_sha256 = artifact["artifact_sha256"]
    split_name = f"{artifact_sha256[:16]}-eval{len(selected_held_out)}"
    split_root = args.output_root.resolve() / split_name
    train_dir = split_root / "train"
    eval_dir = split_root / "eval"
    ensure_symlink_directory(train_dir, verified_train)
    ensure_symlink_directory(eval_dir, verified_eval)

    manifest = {
        "schema": "paste_repro.materialized_trace_split",
        "version": 1,
        "mapper_artifact_sha256": artifact_sha256,
        "training_manifest_sha256": split["manifest_sha256"],
        "train_sessions": train_entries,
        "held_out_sessions": selected_held_out,
        "train_directory": str(train_dir),
        "held_out_directory": str(eval_dir),
    }
    manifest_path = split_root / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
