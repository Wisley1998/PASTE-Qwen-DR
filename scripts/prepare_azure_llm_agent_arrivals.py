#!/usr/bin/env python3
"""Select a trace-only Azure LLM arrival window for an Agent workload."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "paste_repro.azure_llm_agent_arrivals.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(row_number: int, timestamp: str, divisor: int) -> bool:
    material = f"{row_number}:{timestamp}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value % divisor == 0


def select_first_exact_window(
    trace: Path, *, target: int, divisor: int, window_s: int
) -> tuple[int, list[dict[str, Any]], int]:
    current_start: int | None = None
    sampled: list[dict[str, Any]] = []
    raw_rows = 0
    with trace.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Azure trace is missing columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            timestamp = row["TIMESTAMP"]
            parsed = datetime.fromisoformat(timestamp)
            epoch_s = int(parsed.timestamp())
            window_start = epoch_s - epoch_s % window_s
            if current_start is None:
                current_start = window_start
            if window_start != current_start:
                if len(sampled) == target:
                    return current_start, sampled, raw_rows
                current_start = window_start
                sampled = []
                raw_rows = 0
            raw_rows += 1
            if selected(row_number, timestamp, divisor):
                sampled.append(
                    {
                        "csv_row_number": row_number,
                        "timestamp_utc": timestamp,
                        "context_tokens": int(row["ContextTokens"]),
                        "generated_tokens": int(row["GeneratedTokens"]),
                    }
                )
    if current_start is not None and len(sampled) == target:
        return current_start, sampled, raw_rows
    raise ValueError("no aligned window has exactly the requested sampled arrivals")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--divisor", type=int, default=40)
    parser.add_argument("--window-s", type=int, default=120)
    args = parser.parse_args()
    if args.target <= 0 or args.divisor <= 0 or args.window_s <= 0:
        parser.error("target, divisor, and window-s must be positive")

    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    sources = workload.get("sources")
    if not isinstance(sources, list) or len(sources) != args.target:
        raise ValueError("workload source count must equal target")
    source_ids = [row.get("source_id") for row in sources]
    if any(not isinstance(value, str) or not value for value in source_ids):
        raise ValueError("every workload source needs a source_id")

    start_epoch, arrivals, raw_rows = select_first_exact_window(
        args.trace, target=args.target, divisor=args.divisor, window_s=args.window_s
    )
    first_ts = datetime.fromisoformat(arrivals[0]["timestamp_utc"])
    for source_id, arrival in zip(source_ids, arrivals, strict=True):
        offset = (datetime.fromisoformat(arrival["timestamp_utc"]) - first_ts).total_seconds()
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("invalid arrival offset")
        arrival["source_id"] = source_id
        arrival["release_offset_s"] = offset

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "policy": "first_utc_aligned_window_with_exact_target_after_sha256_mod_sampling",
            "outcome_independent": True,
            "hash_material": "csv_row_number:TIMESTAMP",
            "hash_prefix_bits": 64,
            "modulus_divisor": args.divisor,
            "accepted_remainder": 0,
            "window_s": args.window_s,
            "target_arrivals": args.target,
            "aligned_window_start_utc": datetime.fromtimestamp(
                start_epoch, timezone.utc
            ).isoformat(),
            "aligned_window_end_utc": datetime.fromtimestamp(
                start_epoch + args.window_s, timezone.utc
            ).isoformat(),
            "raw_rows_in_window": raw_rows,
        },
        "source": {
            "azure_trace": str(args.trace.resolve()),
            "azure_trace_sha256": sha256_file(args.trace),
            "workload": str(args.workload.resolve()),
            "workload_sha256": sha256_file(args.workload),
        },
        "arrival_span_s": arrivals[-1]["release_offset_s"],
        "arrivals": arrivals,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
