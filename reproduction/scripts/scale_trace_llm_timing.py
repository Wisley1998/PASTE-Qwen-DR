#!/usr/bin/env python3
"""Materialize a timestamp-consistent LLM-duration counterfactual trace.

Each LLM call keeps its causal start boundary, while ``total_time_ms``,
``inference_time_ms``, and ``rtt_ms`` are multiplied by ``--duration-scale``.
The LLM completion and every later event are shifted earlier by the removed
duration. Tool timings and their serial/parallel execution annotations are
left unchanged.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCHEMA = "paste_repro.llm_duration_scaled_trace.v1"
SCRIPT = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT.parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "traces"
    / "my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "traces"
    / "my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42"
)


def number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum_s": sum(values),
        "mean_s": statistics.fmean(values) if values else 0.0,
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
        "max_s": max(values, default=0.0),
        "min_s": min(values, default=0.0),
    }


def scale_events(
    events: Sequence[Mapping[str, Any]], *, duration_scale: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scale LLM calls and causally shift all completion timestamps."""

    if not 0.0 < duration_scale <= 1.0:
        raise ValueError("duration scale must be in (0, 1]")

    rewritten: list[dict[str, Any]] = []
    cumulative_removed_s = 0.0
    previous_timestamp_s = 0.0
    llm_rows: list[dict[str, float | int]] = []
    for index, source in enumerate(events):
        event = dict(source)
        original_timestamp_s = number(event.get("timestamp", 0.0), "timestamp")
        removed_s = 0.0
        if event.get("event_type") == "llm_call":
            original_total_ms = number(
                event.get("total_time_ms", 0.0), "total_time_ms"
            )
            original_inference_ms = number(
                event.get("inference_time_ms", 0.0), "inference_time_ms"
            )
            if original_total_ms < 0.0 or original_inference_ms < 0.0:
                raise ValueError("LLM durations must be non-negative")
            scaled_total_ms = original_total_ms * duration_scale
            scaled_inference_ms = original_inference_ms * duration_scale
            removed_s = (original_total_ms - scaled_total_ms) / 1000.0
            event["total_time_ms"] = scaled_total_ms
            event["inference_time_ms"] = scaled_inference_ms
            if "rtt_ms" in event:
                original_rtt_ms = number(event["rtt_ms"], "rtt_ms")
                if original_rtt_ms < 0.0:
                    raise ValueError("rtt_ms must be non-negative")
                event["rtt_ms"] = original_rtt_ms * duration_scale
            llm_rows.append(
                {
                    "event_index": index,
                    "original_total_s": original_total_ms / 1000.0,
                    "scaled_total_s": scaled_total_ms / 1000.0,
                    "removed_s": removed_s,
                }
            )

        # LLM timestamps are completion timestamps, so their own removed time
        # shifts the event itself. Later events inherit the cumulative shift.
        timestamp_s = original_timestamp_s - cumulative_removed_s - removed_s
        if timestamp_s < -1e-9:
            raise ValueError("LLM scaling produced a negative timestamp")
        timestamp_s = max(0.0, timestamp_s)
        if timestamp_s + 1e-9 < previous_timestamp_s:
            raise ValueError("LLM scaling produced decreasing timestamps")
        event["timestamp"] = timestamp_s
        rewritten.append(event)
        previous_timestamp_s = timestamp_s
        cumulative_removed_s += removed_s

    return rewritten, {
        "llm_calls": len(llm_rows),
        "original_total": distribution(
            [float(row["original_total_s"]) for row in llm_rows]
        ),
        "scaled_total": distribution(
            [float(row["scaled_total_s"]) for row in llm_rows]
        ),
        "removed_total_s": cumulative_removed_s,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(value)
    return events


def write_jsonl(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for event in events:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-scale", type=float, default=0.42)
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("input and output directories must differ")
    if not 0.0 < args.duration_scale <= 1.0:
        parser.error("duration scale must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    inputs = sorted(args.input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not inputs:
        raise FileNotFoundError(f"no JSONL traces found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    files: list[dict[str, Any]] = []
    total_llm_calls = 0
    total_removed_s = 0.0
    for source in inputs:
        rewritten, audit = scale_events(
            load_jsonl(source), duration_scale=args.duration_scale
        )
        destination = args.output_dir / source.name
        write_jsonl(destination, rewritten)
        total_llm_calls += int(audit["llm_calls"])
        total_removed_s += float(audit["removed_total_s"])
        files.append(
            {
                "filename": source.name,
                "input_sha256": sha256_file(source),
                "output_sha256": sha256_file(destination),
                "llm_calls": audit["llm_calls"],
                "removed_total_s": audit["removed_total_s"],
            }
        )

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "duration_scale": args.duration_scale,
        "composition": "0.70 * 0.60 = 0.42",
        "scaled_fields": ["total_time_ms", "inference_time_ms", "rtt_ms"],
        "timestamp_policy": (
            "preserve each LLM causal start after prior shifts; move its "
            "completion and every later event earlier by removed total duration"
        ),
        "llm_calls": total_llm_calls,
        "removed_total_s": total_removed_s,
        "file_count": len(files),
        "files": files,
    }
    manifest_path = args.output_dir / "LLM_SCALE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
