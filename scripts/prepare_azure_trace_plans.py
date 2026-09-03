#!/usr/bin/env python3
"""Apply Azure LLM and Azure Functions arrivals to one frozen FULL replay plan."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from azure_functions_trace import load_azure_functions_window, sample_release_offsets
from azure_llm_trace import load_azure_llm_invocations, sha256_file


PLAN_SCHEMA = "paste_repro.trace_all_visit_live_plan.v1"


def canonical_hash(payload: Any) -> str:
    wire = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_base_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported base plan schema: {plan.get('schema')!r}")
    expected = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if expected != canonical_hash(unsigned):
        raise ValueError("base plan checksum mismatch")
    traces = plan.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("base plan contains no traces")
    if plan.get("arrival_process"):
        raise ValueError("base plan already contains an arrival process")
    return plan


def shuffled_round_robin_indices(
    template_count: int, session_count: int, seed: int
) -> list[int]:
    rng = random.Random(seed)
    indices: list[int] = []
    while len(indices) < session_count:
        cycle = list(range(template_count))
        rng.shuffle(cycle)
        indices.extend(cycle[: session_count - len(indices)])
    return indices


def materialize_arrival_plan(
    base_plan: Mapping[str, Any],
    release_offsets_s: Sequence[float],
    *,
    arrival_process: Mapping[str, Any],
    arrival_rows: Sequence[Mapping[str, Any]] | None = None,
    mapping_seed: int = 20260417,
) -> dict[str, Any]:
    """Clone plan templates without changing any per-session call graph."""

    offsets = [float(value) for value in release_offsets_s]
    if not offsets or offsets != sorted(offsets) or offsets[0] < 0.0:
        raise ValueError("release offsets must be a non-empty sorted non-negative list")
    if arrival_rows is not None and len(arrival_rows) != len(offsets):
        raise ValueError("arrival row metadata length does not match offsets")
    templates = list(base_plan["traces"])
    indices = shuffled_round_robin_indices(len(templates), len(offsets), mapping_seed)
    kind = str(arrival_process["kind"])
    traces: list[dict[str, Any]] = []
    for arrival_index, (offset, template_index) in enumerate(zip(offsets, indices)):
        template = copy.deepcopy(templates[template_index])
        base_trace_id = str(template["trace_id"])
        base_session_id = str(template["session_id"])
        prefix = f"{kind}_{arrival_index:06d}"
        template.update(
            {
                "trace_id": f"{prefix}__{base_trace_id}",
                "session_id": f"{prefix}__{base_session_id}",
                "base_trace_id": base_trace_id,
                "base_session_id": base_session_id,
                "agent_template_index": template_index,
                "release_offset_s": offset,
                "arrival": (
                    dict(arrival_rows[arrival_index])
                    if arrival_rows is not None
                    else {"release_offset_s": offset}
                ),
            }
        )
        traces.append(template)

    plan = copy.deepcopy(dict(base_plan))
    base_hash = str(plan.pop("plan_sha256"))
    process = dict(arrival_process)
    process.update(
        {
            "session_mapping": "shuffled_round_robin",
            "mapping_seed": mapping_seed,
            "session_count": len(traces),
            "agent_template_count": len(templates),
            "base_plan_sha256": base_hash,
            "agent_internal_call_graph_changed": False,
        }
    )
    plan["created_at"] = datetime.now(timezone.utc).isoformat()
    plan["arrival_process"] = process
    plan["traces"] = traces
    plan.setdefault("configuration", {})
    plan["configuration"]["arrival_process_kind"] = kind
    plan["plan_sha256"] = canonical_hash(plan)
    return plan


def prepare_azure_llm_plan(
    base_plan: Mapping[str, Any],
    csv_path: Path,
    *,
    dataset_variant: str,
    start_time: str | None,
    session_count: int,
    arrival_speedup: float,
    mapping_seed: int,
) -> dict[str, Any]:
    rows = load_azure_llm_invocations(
        csv_path, start_time=start_time, max_sessions=session_count
    )
    if len(rows) != session_count:
        raise ValueError(
            f"Azure LLM slice has {len(rows)} rows, expected {session_count}"
        )
    first = rows[0].timestamp
    offsets = [
        (row.timestamp - first).total_seconds() / arrival_speedup for row in rows
    ]
    process = {
        "kind": "azure_llm_inference_2024",
        "semantics": "top_level_agent_session_arrivals_only",
        "source_file": str(csv_path),
        "source_sha256": sha256_file(csv_path),
        "dataset_variant": dataset_variant,
        "first_timestamp_utc": rows[0].timestamp.isoformat(),
        "last_timestamp_utc": rows[-1].timestamp.isoformat(),
        "original_span_s": (rows[-1].timestamp - first).total_seconds(),
        "arrival_speedup": arrival_speedup,
        "replay_span_s": offsets[-1],
        "azure_token_fields_used_for_agent_payload": False,
    }
    arrival_rows = [
        {
            "csv_row_number": row.row_number,
            "timestamp_utc": row.timestamp.isoformat(),
            "context_tokens": row.context_tokens,
            "generated_tokens": row.generated_tokens,
            "release_offset_s": offset,
        }
        for row, offset in zip(rows, offsets)
    ]
    return materialize_arrival_plan(
        base_plan,
        offsets,
        arrival_process=process,
        arrival_rows=arrival_rows,
        mapping_seed=mapping_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--azure-llm-trace", type=Path, required=True)
    parser.add_argument(
        "--azure-llm-variant", choices=["conversation", "code"], default="conversation"
    )
    parser.add_argument("--azure-llm-start-time")
    parser.add_argument("--azure-llm-arrival-speedup", type=float, default=10.0)
    parser.add_argument("--azure-functions-trace", type=Path, required=True)
    parser.add_argument("--azure-functions-day", type=int, default=1)
    parser.add_argument("--azure-functions-start-minute", type=int, default=480)
    parser.add_argument("--azure-functions-duration-minutes", type=int, default=20)
    parser.add_argument("--azure-functions-time-compression", type=float, default=20.0)
    parser.add_argument("--session-count", type=int, default=100)
    parser.add_argument("--mapping-seed", type=int, default=20260417)
    parser.add_argument("--functions-sampling-seed", type=int, default=20260903)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.session_count <= 0:
        parser.error("--session-count must be positive")
    if args.azure_llm_arrival_speedup <= 0:
        parser.error("--azure-llm-arrival-speedup must be positive")
    if args.azure_functions_time_compression <= 0:
        parser.error("--azure-functions-time-compression must be positive")
    return args


def main() -> None:
    args = parse_args()
    base_plan = load_base_plan(args.base_plan)
    llm_plan = prepare_azure_llm_plan(
        base_plan,
        args.azure_llm_trace,
        dataset_variant=args.azure_llm_variant,
        start_time=args.azure_llm_start_time,
        session_count=args.session_count,
        arrival_speedup=args.azure_llm_arrival_speedup,
        mapping_seed=args.mapping_seed,
    )
    functions_window = load_azure_functions_window(
        args.azure_functions_trace,
        day=args.azure_functions_day,
        start_minute=args.azure_functions_start_minute,
        duration_minutes=args.azure_functions_duration_minutes,
    )
    function_offsets, function_process = sample_release_offsets(
        functions_window,
        session_count=args.session_count,
        time_compression=args.azure_functions_time_compression,
        seed=args.functions_sampling_seed,
    )
    functions_plan = materialize_arrival_plan(
        base_plan,
        function_offsets,
        arrival_process=function_process,
        mapping_seed=args.mapping_seed,
    )
    llm_path = args.output_dir / "azure_llm_conversation_plan.json"
    functions_path = args.output_dir / "azure_functions_plan.json"
    write_json(llm_path, llm_plan)
    write_json(functions_path, functions_plan)
    print(
        json.dumps(
            {
                "azure_llm": {
                    "plan": str(llm_path),
                    "plan_sha256": llm_plan["plan_sha256"],
                    "arrival_process": llm_plan["arrival_process"],
                },
                "azure_functions": {
                    "plan": str(functions_path),
                    "plan_sha256": functions_plan["plan_sha256"],
                    "arrival_process": functions_plan["arrival_process"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

