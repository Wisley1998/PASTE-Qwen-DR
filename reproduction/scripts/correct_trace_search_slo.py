#!/usr/bin/env python3
"""Create a causal trace derivative with search gaps sampled inside an SLO.

The historical Qwen traces record a search ``tool_call`` start but no explicit
tool completion.  Its exposed duration is therefore recovered as the next LLM
request start minus the search timestamp.  Old batch searches contain a clear
serial-per-query timing artifact (often about ten seconds per query).

This tool assigns each measurable complete search invocation a deterministic
uniform pseudo-random duration in ``[min, max]`` and shifts the following LLM
plus every later event by the same delta.  The sample key includes a fixed seed,
session filename, and event identity, making the derivative reproducible while
avoiding a degenerate corpus in which every search lasts exactly the SLO cap.
LLM durations, visit gaps, event payloads, filenames, and causal ordering are
otherwise preserved.  The input corpus is never modified.
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


SCHEMA = "paste_repro.search_slo_corrected_trace.v1"
SCRIPT = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT.parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "traces" / "my_traces_search_slo_uniform_1_3s"
DEFAULT_SEED = "qwen-search-slo-uniform-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    }


def uniform_search_duration_s(
    *,
    min_search_s: float,
    max_search_s: float,
    seed: str,
    session_id: str,
    event_index: int,
    call_index: Any,
) -> float:
    """Return a stable uniform sample without depending on RNG call order."""

    if min_search_s < 0.0 or max_search_s < min_search_s:
        raise ValueError("search SLO bounds are invalid")
    if min_search_s == max_search_s:
        return min_search_s
    digest = hashlib.sha256(
        (
            f"{seed}\0{session_id}\0{event_index}\0{call_index}"
        ).encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return min_search_s + (max_search_s - min_search_s) * unit


def correct_events(
    events: Sequence[Mapping[str, Any]],
    *,
    min_search_s: float,
    max_search_s: float,
    seed: str = DEFAULT_SEED,
    session_id: str = "synthetic-session",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resample complete search gaps and return rewritten events plus audit."""

    if min_search_s < 0.0 or max_search_s < min_search_s:
        raise ValueError("search SLO bounds are invalid")
    copied = [dict(event) for event in events]
    adjustments_at: dict[int, float] = {}
    search_rows: list[dict[str, Any]] = []
    terminal_searches = 0
    for index, event in enumerate(copied):
        if not (
            event.get("event_type") == "tool_call"
            and event.get("tool_name") == "search"
        ):
            continue
        next_llm_index = next(
            (
                candidate
                for candidate in range(index + 1, len(copied))
                if copied[candidate].get("event_type") == "llm_call"
            ),
            None,
        )
        if next_llm_index is None:
            terminal_searches += 1
            continue
        if next_llm_index != index + 1:
            raise ValueError(
                "search correction requires the following event to be its LLM"
            )
        observed_s = max(
            0.0,
            llm_start_s(copied[next_llm_index])
            - number(event.get("timestamp", 0.0), "search timestamp"),
        )
        corrected_s = uniform_search_duration_s(
            min_search_s=min_search_s,
            max_search_s=max_search_s,
            seed=seed,
            session_id=session_id,
            event_index=index,
            call_index=event.get("call_index"),
        )
        removed_s = observed_s - corrected_s
        adjustments_at[next_llm_index] = (
            adjustments_at.get(next_llm_index, 0.0) + removed_s
        )
        raw_queries = event.get("tool_args", {}).get("query")
        query_count = len(raw_queries) if isinstance(raw_queries, list) else 1
        search_rows.append(
            {
                "event_index": index,
                "call_index": event.get("call_index"),
                "query_count": query_count,
                "observed_s": observed_s,
                "corrected_s": corrected_s,
                "removed_s": removed_s,
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
            raise ValueError("search correction produced decreasing timestamps")
        event["timestamp"] = timestamp_s
        rewritten.append(event)
        previous_timestamp_s = timestamp_s

    before = [float(row["observed_s"]) for row in search_rows]
    after = [float(row["corrected_s"]) for row in search_rows]
    return rewritten, {
        "measurable_searches": len(search_rows),
        "terminal_searches_without_following_llm": terminal_searches,
        "original_below_slo": sum(value < min_search_s for value in before),
        "original_above_slo": sum(value > max_search_s for value in before),
        "original_inside_slo": sum(
            min_search_s <= value <= max_search_s for value in before
        ),
        "total_removed_s": sum(before) - sum(after),
        "before": distribution(before),
        "after": distribution(after),
        "rows": search_rows,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
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
    parser.add_argument("--min-search-s", type=float, default=1.0)
    parser.add_argument("--max-search-s", type=float, default=3.0)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.min_search_s < 0.0 or args.max_search_s < args.min_search_s:
        parser.error("search SLO bounds are invalid")
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

    file_rows = []
    all_before: list[float] = []
    all_after: list[float] = []
    terminal_searches = 0
    for source in inputs:
        rewritten, audit = correct_events(
            load_jsonl(source),
            min_search_s=args.min_search_s,
            max_search_s=args.max_search_s,
            seed=args.seed,
            session_id=source.name,
        )
        destination = args.output_dir / source.name
        write_jsonl(destination, rewritten)
        before = [float(row["observed_s"]) for row in audit["rows"]]
        after = [float(row["corrected_s"]) for row in audit["rows"]]
        all_before.extend(before)
        all_after.extend(after)
        terminal_searches += int(audit["terminal_searches_without_following_llm"])
        file_rows.append(
            {
                "filename": source.name,
                "input_sha256": sha256_file(source),
                "output_sha256": sha256_file(destination),
                "measurable_searches": len(before),
                "terminal_searches_without_following_llm": audit[
                    "terminal_searches_without_following_llm"
                ],
                "total_removed_s": sum(before) - sum(after),
            }
        )

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "min_search_s": args.min_search_s,
        "max_search_s": args.max_search_s,
        "sampling": "deterministic_uniform_sha256",
        "seed": args.seed,
        "file_count": len(file_rows),
        "terminal_searches_without_following_llm": terminal_searches,
        "original_below_slo": sum(
            value < args.min_search_s for value in all_before
        ),
        "original_above_slo": sum(
            value > args.max_search_s for value in all_before
        ),
        "original_inside_slo": sum(
            args.min_search_s <= value <= args.max_search_s
            for value in all_before
        ),
        "total_removed_s": sum(all_before) - sum(all_after),
        "before": distribution(all_before),
        "after": distribution(all_after),
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
