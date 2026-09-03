#!/usr/bin/env python3
"""Materialize Agent-session arrivals from Azure Functions Dataset 2019.

The public trace contains per-function invocation counts in one-minute bins.
This module aggregates a fixed window and samples real invocation mass without
replacement.  Only the unobserved position *inside* each minute is randomized
uniformly.  It never substitutes Azure function payloads for Agent requests.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO


ARRIVAL_PROCESS_KIND = "azure_functions_2019"
MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class AzureFunctionsWindow:
    counts: tuple[int, ...]
    source_file: str
    source_sha256: str
    csv_member: str
    day: int
    start_minute: int
    function_rows: int

    @property
    def raw_invocations(self) -> int:
        return sum(self.counts)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day_filename(day: int) -> str:
    if not 1 <= day <= 14:
        raise ValueError("Azure Functions day must be in [1, 14]")
    return f"invocations_per_function_md.anon.d{day:02d}.csv"


def _open_tar_member(path: Path, filename: str) -> tuple[TextIO, str, tarfile.TarFile]:
    archive = tarfile.open(path, mode="r:*")
    try:
        for member in archive:
            if Path(member.name).name != filename:
                continue
            raw = archive.extractfile(member)
            if raw is None:
                raise ValueError(f"Azure Functions member is not a file: {member.name}")
            return io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), member.name, archive
    except Exception:
        archive.close()
        raise
    archive.close()
    raise FileNotFoundError(f"{filename} not found in Azure Functions archive: {path}")


def _open_source(source: Path, day: int) -> tuple[TextIO, str, object | None]:
    filename = _day_filename(day)
    if source.is_dir():
        candidates = (
            source / filename,
            source / f"{filename}.gz",
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.suffix == ".gz":
                import gzip

                return gzip.open(candidate, "rt", encoding="utf-8-sig", newline=""), candidate.name, None
            return candidate.open("r", encoding="utf-8-sig", newline=""), candidate.name, None
        raise FileNotFoundError(f"{filename} not found under {source}")
    if not source.is_file():
        raise FileNotFoundError(f"Azure Functions trace not found: {source}")
    if source.name.endswith((".tar", ".tar.xz", ".tar.gz", ".tgz")):
        return _open_tar_member(source, filename)
    return source.open("r", encoding="utf-8-sig", newline=""), source.name, None


def load_azure_functions_window(
    source: str | Path,
    *,
    day: int = 1,
    start_minute: int = 0,
    duration_minutes: int = 20,
) -> AzureFunctionsWindow:
    """Aggregate one chronological window of real per-minute counts."""

    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")
    if not 0 <= start_minute < MINUTES_PER_DAY:
        raise ValueError("start_minute must be in [0, 1439]")
    if start_minute + duration_minutes > MINUTES_PER_DAY:
        raise ValueError("Azure Functions window crosses the end of the day")

    source_path = Path(source)
    handle, member_name, owner = _open_source(source_path, day)
    counts = [0] * duration_minutes
    function_rows = 0
    try:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Azure Functions CSV is empty") from exc
        minute_columns: list[int] = []
        positions = {name: index for index, name in enumerate(header)}
        for minute in range(start_minute + 1, start_minute + duration_minutes + 1):
            column = positions.get(str(minute))
            if column is None:
                raise ValueError(f"Azure Functions CSV is missing minute column {minute}")
            minute_columns.append(column)
        for row_number, row in enumerate(reader, start=2):
            function_rows += 1
            for output_index, column in enumerate(minute_columns):
                value = row[column].strip() if column < len(row) else ""
                if not value:
                    continue
                try:
                    parsed = int(value)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid invocation count at row {row_number}, "
                        f"minute {start_minute + output_index + 1}: {value!r}"
                    ) from exc
                if parsed < 0:
                    raise ValueError("Azure Functions invocation counts must be non-negative")
                counts[output_index] += parsed
    finally:
        handle.close()
        if owner is not None:
            owner.close()

    if not function_rows:
        raise ValueError("Azure Functions CSV contains no function rows")
    if not sum(counts):
        raise ValueError("selected Azure Functions window contains no invocations")
    return AzureFunctionsWindow(
        counts=tuple(counts),
        source_file=str(source_path),
        source_sha256=sha256_file(source_path),
        csv_member=member_name,
        day=day,
        start_minute=start_minute,
        function_rows=function_rows,
    )


def sample_release_offsets(
    window: AzureFunctionsWindow,
    *,
    session_count: int,
    time_compression: float = 20.0,
    seed: int = 20260903,
) -> tuple[list[float], dict[str, object]]:
    """Sample actual count mass and return normalized replay-second offsets."""

    if session_count <= 0:
        raise ValueError("session_count must be positive")
    if time_compression <= 0:
        raise ValueError("time_compression must be positive")
    total = window.raw_invocations
    if session_count > total:
        raise ValueError(
            f"requested {session_count} sessions from only {total} raw invocations"
        )

    rng = random.Random(seed)
    ranks = sorted(rng.sample(range(total), session_count))
    offsets: list[float] = []
    cumulative = 0
    rank_index = 0
    for minute_index, count in enumerate(window.counts):
        upper = cumulative + count
        while rank_index < len(ranks) and ranks[rank_index] < upper:
            within_minute_s = rng.random() * 60.0
            offsets.append((minute_index * 60.0 + within_minute_s) / time_compression)
            rank_index += 1
        cumulative = upper
    offsets.sort()
    first = offsets[0]
    offsets = [value - first for value in offsets]
    metadata: dict[str, object] = {
        "kind": ARRIVAL_PROCESS_KIND,
        "semantics": "top_level_agent_session_arrivals_only",
        "source_file": window.source_file,
        "source_sha256": window.source_sha256,
        "csv_member": window.csv_member,
        "day": window.day,
        "start_minute_zero_based": window.start_minute,
        "duration_minutes": len(window.counts),
        "function_rows": window.function_rows,
        "raw_invocations_in_window": total,
        "sampled_without_replacement": session_count,
        "intra_minute_assumption": "uniform_conditioned_on_observed_count",
        "time_compression": time_compression,
        "seed": seed,
        "replay_span_s": offsets[-1],
    }
    return offsets, metadata

