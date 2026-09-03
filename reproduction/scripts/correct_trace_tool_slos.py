#!/usr/bin/env python3
"""Build a timestamp-consistent trace with realistic search and visit SLOs.

Search is sampled once per tool call. Visit is sampled independently per URL,
and a multi-URL visit call is the serial sum of those samples. Every following
event timestamp is shifted by the observed-minus-corrected duration, preserving
causal gaps and all LLM durations.

Legacy terminal tool calls have no following LLM and therefore no recorded
completion boundary. For those calls, this derivative appends an explicit
``synthetic_tool_completion`` event so the session wall includes their sampled
service instead of treating them as zero-duration work.
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


SCHEMA = "paste_repro.tool_slo_corrected_trace.v1"
SCRIPT = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT.parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "traces"
    / "my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s"
)
DEFAULT_SEED = "qwen-tool-slo-uniform-v1"


def number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def llm_start_s(event: Mapping[str, Any]) -> float:
    return max(
        0.0,
        number(event.get("timestamp", 0.0), "timestamp")
        - number(event.get("total_time_ms", 0.0), "total_time_ms") / 1000.0,
    )


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
    lower = math.floor(position)
    upper = math.ceil(position)
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


def stable_uniform_duration_s(
    *,
    minimum_s: float,
    maximum_s: float,
    seed: str,
    session_id: str,
    event_index: int,
    call_index: Any,
    tool_name: str,
    unit_index: int,
) -> float:
    """Return one call-order-independent SHA-256 uniform sample."""

    if minimum_s < 0.0 or maximum_s < minimum_s:
        raise ValueError("duration bounds are invalid")
    if minimum_s == maximum_s:
        return minimum_s
    digest = hashlib.sha256(
        (
            f"{seed}\0{session_id}\0{event_index}\0{call_index}"
            f"\0{tool_name}\0{unit_index}"
        ).encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return minimum_s + (maximum_s - minimum_s) * unit


def visit_urls(event: Mapping[str, Any]) -> tuple[str, ...]:
    arguments = event.get("tool_args", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("visit tool_args must be an object")
    raw = arguments.get("url")
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return tuple(item for item in raw if item)
    raise ValueError("visit url must be a string or string list")


def corrected_tool_duration_s(
    event: Mapping[str, Any],
    *,
    search_min_s: float,
    search_max_s: float,
    visit_min_s: float,
    visit_max_s: float,
    seed: str,
    session_id: str,
    event_index: int,
) -> tuple[float, tuple[float, ...], str] | None:
    if event.get("event_type") != "tool_call":
        return None
    tool_name = event.get("tool_name")
    if tool_name == "search":
        samples = (
            stable_uniform_duration_s(
                minimum_s=search_min_s,
                maximum_s=search_max_s,
                seed=seed,
                session_id=session_id,
                event_index=event_index,
                call_index=event.get("call_index"),
                tool_name="search",
                unit_index=0,
            ),
        )
        return samples[0], samples, "one_sample_per_call"
    if tool_name == "visit":
        urls = visit_urls(event)
        if not urls:
            raise ValueError("visit call has no URLs")
        samples = tuple(
            stable_uniform_duration_s(
                minimum_s=visit_min_s,
                maximum_s=visit_max_s,
                seed=seed,
                session_id=session_id,
                event_index=event_index,
                call_index=event.get("call_index"),
                tool_name="visit",
                unit_index=unit_index,
            )
            for unit_index in range(len(urls))
        )
        return sum(samples), samples, "serial_sum_per_url"
    return None


def correct_events(
    events: Sequence[Mapping[str, Any]],
    *,
    search_min_s: float = 1.0,
    search_max_s: float = 3.0,
    visit_min_s: float = 2.0,
    visit_max_s: float = 8.0,
    seed: str = DEFAULT_SEED,
    session_id: str = "synthetic-session",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Correct complete tools, append terminal completions, and return audit."""

    for minimum, maximum, label in (
        (search_min_s, search_max_s, "search"),
        (visit_min_s, visit_max_s, "visit"),
    ):
        if minimum < 0.0 or maximum < minimum:
            raise ValueError(f"{label} duration bounds are invalid")

    copied = [dict(event) for event in events]
    adjustments_at: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    terminal_indexes: list[int] = []
    for index, event in enumerate(copied):
        corrected = corrected_tool_duration_s(
            event,
            search_min_s=search_min_s,
            search_max_s=search_max_s,
            visit_min_s=visit_min_s,
            visit_max_s=visit_max_s,
            seed=seed,
            session_id=session_id,
            event_index=index,
        )
        if corrected is None:
            continue
        corrected_s, unit_samples_s, execution = corrected
        tool_name = str(event["tool_name"])
        event["timing_correction"] = {
            "schema": SCHEMA,
            "sampling": "deterministic_uniform_sha256",
            "execution": execution,
            "duration_s": corrected_s,
            "unit_duration_s": list(unit_samples_s),
        }
        next_llm_index = next(
            (
                candidate
                for candidate in range(index + 1, len(copied))
                if copied[candidate].get("event_type") == "llm_call"
            ),
            None,
        )
        observed_s: float | None = None
        if next_llm_index is None:
            terminal_indexes.append(index)
        else:
            if next_llm_index != index + 1:
                raise ValueError(
                    f"{tool_name} correction requires the following event to be its LLM"
                )
            observed_s = max(
                0.0,
                llm_start_s(copied[next_llm_index])
                - number(event.get("timestamp", 0.0), "tool timestamp"),
            )
            adjustments_at[next_llm_index] = (
                adjustments_at.get(next_llm_index, 0.0)
                + observed_s
                - corrected_s
            )
        rows.append(
            {
                "event_index": index,
                "call_index": event.get("call_index"),
                "tool_name": tool_name,
                "unit_count": len(unit_samples_s),
                "unit_duration_s": list(unit_samples_s),
                "observed_s": observed_s,
                "corrected_s": corrected_s,
                "terminal": next_llm_index is None,
            }
        )

    cumulative_removed_s = 0.0
    rewritten: list[dict[str, Any]] = []
    previous_timestamp_s = 0.0
    for index, event in enumerate(copied):
        cumulative_removed_s += adjustments_at.get(index, 0.0)
        timestamp_s = max(
            0.0,
            number(event.get("timestamp", 0.0), "timestamp")
            - cumulative_removed_s,
        )
        if timestamp_s + 1e-9 < previous_timestamp_s:
            raise ValueError("tool correction produced decreasing timestamps")
        event["timestamp"] = timestamp_s
        rewritten.append(event)
        previous_timestamp_s = timestamp_s

    for index in terminal_indexes:
        tool = copied[index]
        correction = tool["timing_correction"]
        completion_timestamp_s = number(tool["timestamp"], "tool timestamp") + number(
            correction["duration_s"], "corrected duration"
        )
        if index != len(copied) - 1:
            raise ValueError("terminal tool is not the final legacy event")
        marker = {
            "event_type": "synthetic_tool_completion",
            "call_index": tool.get("call_index", 0),
            "timestamp": completion_timestamp_s,
            "tool_name": tool["tool_name"],
            "source_event_index": index,
            "timing_correction_schema": SCHEMA,
        }
        if completion_timestamp_s + 1e-9 < previous_timestamp_s:
            raise ValueError("terminal completion timestamp is not monotonic")
        rewritten.append(marker)
        previous_timestamp_s = completion_timestamp_s

    return rewritten, {"rows": rows, "terminal_completions_added": len(terminal_indexes)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            result.append(value)
    return result


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
    parser.add_argument("--search-min-s", type=float, default=1.0)
    parser.add_argument("--search-max-s", type=float, default=3.0)
    parser.add_argument("--visit-min-s", type=float, default=2.0)
    parser.add_argument("--visit-max-s", type=float, default=8.0)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("input and output directories must differ")
    return args


def main() -> None:
    args = parse_args()
    inputs = sorted(args.input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not inputs:
        raise FileNotFoundError(f"no JSONL traces found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    file_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for source in inputs:
        rewritten, audit = correct_events(
            load_jsonl(source),
            search_min_s=args.search_min_s,
            search_max_s=args.search_max_s,
            visit_min_s=args.visit_min_s,
            visit_max_s=args.visit_max_s,
            seed=args.seed,
            session_id=source.name,
        )
        destination = args.output_dir / source.name
        write_jsonl(destination, rewritten)
        rows = list(audit["rows"])
        all_rows.extend(rows)
        file_rows.append(
            {
                "filename": source.name,
                "input_sha256": sha256_file(source),
                "output_sha256": sha256_file(destination),
                "corrected_search_calls": sum(
                    row["tool_name"] == "search" for row in rows
                ),
                "corrected_visit_calls": sum(
                    row["tool_name"] == "visit" for row in rows
                ),
                "corrected_visit_urls": sum(
                    int(row["unit_count"])
                    for row in rows
                    if row["tool_name"] == "visit"
                ),
                "terminal_completions_added": audit[
                    "terminal_completions_added"
                ],
            }
        )

    def selected(tool_name: str, *, terminal: bool | None = None) -> list[dict[str, Any]]:
        return [
            row
            for row in all_rows
            if row["tool_name"] == tool_name
            and (terminal is None or bool(row["terminal"]) == terminal)
        ]

    search_rows = selected("search")
    visit_rows = selected("visit")
    measurable_search = selected("search", terminal=False)
    measurable_visit = selected("visit", terminal=False)
    visit_units = [
        float(sample)
        for row in visit_rows
        for sample in row["unit_duration_s"]
    ]
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "sampling": "deterministic_uniform_sha256",
        "seed": args.seed,
        "search": {
            "bounds_s": [args.search_min_s, args.search_max_s],
            "execution": "one_sample_per_call",
            "calls": len(search_rows),
            "terminal_calls": sum(bool(row["terminal"]) for row in search_rows),
            "before_measurable": distribution(
                [float(row["observed_s"]) for row in measurable_search]
            ),
            "after_all": distribution(
                [float(row["corrected_s"]) for row in search_rows]
            ),
        },
        "visit": {
            "bounds_per_url_s": [args.visit_min_s, args.visit_max_s],
            "execution": "serial_sum_per_url",
            "calls": len(visit_rows),
            "urls": sum(int(row["unit_count"]) for row in visit_rows),
            "terminal_calls": sum(bool(row["terminal"]) for row in visit_rows),
            "before_measurable_batch": distribution(
                [float(row["observed_s"]) for row in measurable_visit]
            ),
            "after_all_batch": distribution(
                [float(row["corrected_s"]) for row in visit_rows]
            ),
            "after_all_url": distribution(visit_units),
        },
        "terminal_completions_added": sum(
            int(row["terminal_completions_added"]) for row in file_rows
        ),
        "file_count": len(file_rows),
        "files": file_rows,
    }
    manifest_path = args.output_dir / "CORRECTION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
