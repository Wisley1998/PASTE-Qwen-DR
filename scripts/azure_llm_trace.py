#!/usr/bin/env python3
"""Map the Azure LLM Inference Trace (2024) to Agent-session arrivals.

The Azure dataset describes independent LLM invocations.  It does not contain
Agent conversations or tool calls.  This adapter therefore uses only each CSV
row's timestamp as the arrival time of a complete, already-prepared Agent
session.  ContextTokens and GeneratedTokens are retained as provenance and are
never substituted for the Agent session's own per-call token counts.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO


REQUIRED_COLUMNS = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")
ARRIVAL_PROCESS_KIND = "azure_llm_inference_2024"


@dataclass(frozen=True)
class AzureLLMInvocation:
    """One selected row from an Azure LLM Inference Trace CSV."""

    timestamp: datetime
    context_tokens: int
    generated_tokens: int
    row_number: int


def _open_csv(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def parse_azure_timestamp(value: str) -> datetime:
    """Parse an Azure timestamp and normalize it to UTC."""

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid Azure TIMESTAMP: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Azure TIMESTAMP must include a UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def _positive_integer(value: str, column: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Azure row {row_number} has invalid {column}: {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"Azure row {row_number} has negative {column}: {parsed}"
        )
    return parsed


def load_azure_llm_invocations(
    csv_file: str | Path,
    *,
    start_time: str | datetime | None = None,
    duration_s: float | None = None,
    max_sessions: int | None = None,
) -> List[AzureLLMInvocation]:
    """Load a chronological slice of the official 2024 Azure CSV.

    ``duration_s`` is measured from the first selected row.  The first selected
    invocation is replayed at offset zero, so a start time between two rows does
    not introduce an artificial silent prefix.
    """

    path = Path(csv_file)
    if not path.is_file():
        raise FileNotFoundError(f"Azure LLM trace not found: {path}")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if max_sessions is not None and max_sessions <= 0:
        raise ValueError("max_sessions must be positive")

    if isinstance(start_time, datetime):
        start = start_time
        if start.tzinfo is None:
            raise ValueError("start_time must include a UTC offset")
        start = start.astimezone(timezone.utc)
    elif start_time is not None:
        start = parse_azure_timestamp(start_time)
    else:
        start = None

    selected: List[AzureLLMInvocation] = []
    first_selected: Optional[datetime] = None
    previous_timestamp: Optional[datetime] = None

    with _open_csv(path) as source:
        reader = csv.DictReader(source)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                "Azure LLM trace is missing required columns: " + ", ".join(missing)
            )

        for row_number, row in enumerate(reader, start=2):
            timestamp = parse_azure_timestamp(row["TIMESTAMP"])
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    f"Azure trace must be chronological; row {row_number} is out of order"
                )
            previous_timestamp = timestamp

            if start is not None and timestamp < start:
                continue
            if first_selected is None:
                first_selected = timestamp
            if duration_s is not None:
                window_end = first_selected + timedelta(seconds=duration_s)
                if timestamp >= window_end:
                    break

            selected.append(
                AzureLLMInvocation(
                    timestamp=timestamp,
                    context_tokens=_positive_integer(
                        row["ContextTokens"], "ContextTokens", row_number
                    ),
                    generated_tokens=_positive_integer(
                        row["GeneratedTokens"], "GeneratedTokens", row_number
                    ),
                    row_number=row_number,
                )
            )
            if max_sessions is not None and len(selected) >= max_sessions:
                break

    if not selected:
        raise ValueError("the requested Azure trace slice contains no invocations")
    return selected


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _template_indices(
    template_count: int,
    invocation_count: int,
    mapping: str,
    seed: int,
) -> Iterable[int]:
    if mapping == "round_robin":
        for index in range(invocation_count):
            yield index % template_count
        return
    if mapping != "shuffled_round_robin":
        raise ValueError(f"unsupported Agent-session mapping: {mapping!r}")

    rng = random.Random(seed)
    emitted = 0
    while emitted < invocation_count:
        cycle = list(range(template_count))
        rng.shuffle(cycle)
        for index in cycle:
            if emitted >= invocation_count:
                return
            yield index
            emitted += 1


def apply_azure_arrivals(
    agent_workload: Dict[str, Any],
    invocations: List[AzureLLMInvocation],
    *,
    source_file: str | Path,
    dataset_variant: str,
    arrival_speedup: float = 1.0,
    mapping: str = "round_robin",
    mapping_seed: int = 20260417,
) -> Dict[str, Any]:
    """Create one Agent session per Azure row, preserving Agent internals."""

    if arrival_speedup <= 0:
        raise ValueError("arrival_speedup must be positive")
    if dataset_variant not in {"conversation", "code"}:
        raise ValueError("dataset_variant must be 'conversation' or 'code'")
    if not invocations:
        raise ValueError("invocations must not be empty")

    existing_process = agent_workload.get("meta", {}).get("arrival_process")
    if existing_process:
        raise ValueError(
            "workload already has an arrival_process; refusing to apply Azure twice"
        )

    templates = agent_workload.get("traces")
    if not isinstance(templates, list) or not templates:
        raise ValueError("Agent workload must contain at least one trace")
    for index, template in enumerate(templates):
        if not isinstance(template.get("requests"), list) or not template["requests"]:
            raise ValueError(f"Agent template {index} has no requests")

    first_timestamp = invocations[0].timestamp
    previous_timestamp = first_timestamp
    mapped_traces: List[Dict[str, Any]] = []
    indices = _template_indices(
        len(templates), len(invocations), mapping, mapping_seed
    )

    for azure_index, (invocation, template_index) in enumerate(zip(invocations, indices)):
        if invocation.timestamp < previous_timestamp:
            raise ValueError("invocations must be chronological")
        previous_timestamp = invocation.timestamp

        original_offset_s = (invocation.timestamp - first_timestamp).total_seconds()
        replay_offset_s = original_offset_s / arrival_speedup
        trace = copy.deepcopy(templates[template_index])
        base_trace_id = str(trace.get("trace_id", f"template_{template_index:03d}"))
        trace["trace_id"] = f"azure_{azure_index:06d}__{base_trace_id}"
        trace["base_trace_id"] = base_trace_id
        trace["agent_template_index"] = template_index
        trace["agent_template_cycle"] = azure_index // len(templates)
        trace["azure_arrival"] = {
            "dataset_variant": dataset_variant,
            "csv_row_number": invocation.row_number,
            "timestamp_utc": invocation.timestamp.isoformat(),
            "context_tokens": invocation.context_tokens,
            "generated_tokens": invocation.generated_tokens,
            "arrival_offset_original_s": original_offset_s,
            "arrival_offset_replay_s": replay_offset_s,
        }
        trace["initial_delay_s"] = replay_offset_s
        first_request = trace["requests"][0]
        first_request["wait_after_prev_s"] = replay_offset_s
        first_request["wait_after_prev_original_s"] = original_offset_s
        mapped_traces.append(trace)

    source_path = Path(source_file)
    original_span_s = (
        invocations[-1].timestamp - invocations[0].timestamp
    ).total_seconds()
    output = copy.deepcopy(agent_workload)
    output["traces"] = mapped_traces
    output.setdefault("meta", {})
    prior_target = output["meta"].get("target_trace_count", len(templates))
    output["meta"]["base_workload_target_trace_count"] = prior_target
    output["meta"]["target_trace_count"] = len(mapped_traces)
    output["meta"]["arrival_process"] = {
        "kind": ARRIVAL_PROCESS_KIND,
        "semantics": "top_level_agent_session_arrivals_only",
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "dataset_variant": dataset_variant,
        "first_timestamp_utc": invocations[0].timestamp.isoformat(),
        "last_timestamp_utc": invocations[-1].timestamp.isoformat(),
        "original_span_s": original_span_s,
        "replay_span_s": original_span_s / arrival_speedup,
        "arrival_speedup": arrival_speedup,
        "session_mapping": mapping,
        "mapping_seed": mapping_seed,
        "azure_invocation_count": len(invocations),
        "agent_template_count": len(templates),
        "azure_token_fields_used_for_agent_payload": False,
    }
    return output
