#!/usr/bin/env python3
"""Download the pinned Tongyi checkpoint and emit a secret-free manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile


DEFAULT_MODEL_ID = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
DEFAULT_REVISION = "4b0ac5767427a55d08a254f0367e2934976598e0"
REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    default_cache = Path(os.environ.get("HF_HOME", "~/hf_cache")).expanduser()
    default_manifest = repository_root() / "reproduction" / "artifacts" / "model_manifest.json"
    parser = argparse.ArgumentParser(
        description="Download the revision-pinned Tongyi model into the HF cache."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Verify/use an existing snapshot without network access.",
    )
    return parser.parse_args()


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)
    path.chmod(0o644)


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))

    # Import after argument parsing so `--help` works before the environment is set up.
    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            cache_dir=str(cache_dir),
            local_files_only=args.local_files_only,
        )
    ).resolve()

    if re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        resolved_revision = snapshot_path.name.lower()
        if resolved_revision != args.revision.lower():
            raise RuntimeError(
                "resolved snapshot does not match the requested immutable revision: "
                f"{resolved_revision} != {args.revision}"
            )
    else:
        resolved_revision = snapshot_path.name

    missing = [name for name in REQUIRED_FILES if not (snapshot_path / name).is_file()]
    if missing:
        raise RuntimeError(f"snapshot is incomplete; missing: {', '.join(missing)}")

    # Keep this allow-listed payload deliberately small: credentials and process
    # environment values are never copied into the manifest.
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot_path),
        "cache_dir": str(cache_dir),
        "verified_files": list(REQUIRED_FILES),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Manifest written to {args.manifest.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
