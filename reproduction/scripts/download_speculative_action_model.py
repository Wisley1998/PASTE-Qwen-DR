#!/usr/bin/env python3
"""Download or validate the pinned local Qwen3 Speculative Actions model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def snapshot_path(cache_dir: Path, model: str, revision: str) -> Path:
    return cache_dir / f"models--{model.replace('/', '--')}" / "snapshots" / revision


def validate(path: Path) -> dict[str, object]:
    required = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    if not shards:
        missing.append("model-*-of-*.safetensors")
    if missing:
        raise FileNotFoundError(f"incomplete model snapshot {path}: missing {missing}")
    return {
        "snapshot": str(path.resolve()),
        "weight_shards": len(shards),
        "weight_bytes": sum(item.stat().st_size for item in shards),
        "status": "ready",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("HF_HOME", Path.home() / "hf_cache")),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    expected = snapshot_path(args.cache_dir, args.model, args.revision)
    if not args.check_only:
        downloaded = snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        if Path(downloaded).resolve() != expected.resolve():
            raise RuntimeError(
                f"download resolved to unexpected snapshot: {downloaded} != {expected}"
            )
    print(json.dumps(validate(expected), indent=2))


if __name__ == "__main__":
    main()
