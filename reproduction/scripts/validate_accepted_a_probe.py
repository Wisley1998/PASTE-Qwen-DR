#!/usr/bin/env python3
"""Fail-closed validation for an A-only natural-queue selection artifact.

The accepted JSON is not trusted as a detached assertion.  Its referenced A
cell is required to be a sibling in the same run directory, and the complete
probe is recomputed from that cell before any acceptance condition is checked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from summarize_natural_queue_probe import SCHEMA, VERSION, summarize_probe


ENGINE_SHAPE_KEYS = (
    "MODEL_ID",
    "MODEL_REVISION",
    "VLLM_TP_SIZE",
    "VLLM_DTYPE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_CUDA_GRAPH_SIZES",
    "VLLM_USE_V1",
)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing or incomplete: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
        exact = float(value) == number
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not exact or number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository: {resolved}") from exc
    return resolved


def _parse_engine_shape(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        name, separator, value = raw.partition("=")
        if not separator or not name or not value:
            raise ValueError(
                "--expect-engine-shape entries must use non-empty NAME=VALUE"
            )
        if name in parsed:
            raise ValueError(f"duplicate engine-shape expectation: {name}")
        parsed[name] = value
    missing = sorted(set(ENGINE_SHAPE_KEYS) - set(parsed))
    extra = sorted(set(parsed) - set(ENGINE_SHAPE_KEYS))
    if missing or extra:
        raise ValueError(
            "engine-shape expectations must contain exactly "
            f"{list(ENGINE_SHAPE_KEYS)} (missing={missing}, extra={extra})"
        )
    return parsed


def validate_accepted_probe(
    probe_path: Path,
    *,
    repository_root: Path,
    expected_profile: str,
    expected_load: int,
    expected_max_num_seqs: int,
    minimum_waiting_fraction: float,
    minimum_queue_fraction: float,
    maximum_preemptions_per_request: float,
    expected_engine_shape: Mapping[str, str],
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    probe_path = _within(probe_path, repository_root, "accepted A probe")
    if probe_path.name != "natural_queue_probe.json":
        raise ValueError(
            "accepted A probe must be a completed natural_queue_probe.json"
        )
    saved = _load_object(probe_path, "accepted A probe")
    if saved.get("schema") != SCHEMA or saved.get("version") != VERSION:
        raise ValueError("accepted A probe has an unsupported schema or version")

    raw_cell_dir = saved.get("cell_dir")
    if not isinstance(raw_cell_dir, str) or not raw_cell_dir:
        raise ValueError("accepted A probe has no cell_dir")
    cell_dir = _within(Path(raw_cell_dir), repository_root, "accepted A cell")
    if cell_dir.parent != probe_path.parent or not cell_dir.name.endswith("_fcfs_none"):
        raise ValueError(
            "accepted A cell must be the fcfs_none sibling of natural_queue_probe.json"
        )

    recomputed = summarize_probe(cell_dir)
    if saved != recomputed:
        raise ValueError(
            "accepted A probe does not exactly match a fresh validation of its cell"
        )

    capacity = _mapping(saved.get("sequence_capacity"), "sequence_capacity")
    requests = _mapping(saved.get("request_accounting"), "request_accounting")
    memory = _mapping(
        saved.get("serving_memory_accounting"), "serving_memory_accounting"
    )
    timeline = _mapping(saved.get("timeline"), "timeline")
    queueing = _mapping(saved.get("queueing"), "queueing")

    if capacity.get("scheduler_policy") != "fcfs":
        raise ValueError("accepted A probe must come from the FCFS A cell")
    load_fields = {
        "workload_trace_count": capacity.get("workload_trace_count"),
        "configured_max_active_traces": capacity.get(
            "configured_max_active_traces"
        ),
        "offered_concurrency_upper_bound": capacity.get(
            "offered_concurrency_upper_bound"
        ),
    }
    for name, value in load_fields.items():
        if _integer(value, f"sequence_capacity {name}") != expected_load:
            raise ValueError(
                f"accepted A probe {name} does not match load {expected_load}"
            )
    if (
        _integer(
            capacity.get("configured_max_num_seqs"),
            "sequence_capacity configured_max_num_seqs",
        )
        != expected_max_num_seqs
    ):
        raise ValueError(
            "accepted A probe configured_max_num_seqs does not match "
            f"{expected_max_num_seqs}"
        )
    if capacity.get("natural_vllm_queue_proven") is not True:
        raise ValueError("accepted A probe does not prove a natural vLLM queue")
    if capacity.get("sequence_cap_nonbinding") is not True:
        raise ValueError("accepted A probe does not prove a non-binding sequence cap")
    if requests.get("all_requests_succeeded_exactly_once") is not True:
        raise ValueError("accepted A probe fails the exactly-once gate")
    if memory.get("kv_swap_happened") is not False:
        raise ValueError("accepted A probe fails the no-CPU-KV-swap gate")

    waiting_fraction = _finite(
        timeline.get("waiting_below_sequence_cap_sample_fraction"),
        "waiting-below-cap sample fraction",
    )
    if waiting_fraction + 1e-12 < minimum_waiting_fraction:
        raise ValueError(
            "accepted A probe fails the waiting-below-cap gate: "
            f"{waiting_fraction} < {minimum_waiting_fraction}"
        )
    queue_fraction = _finite(
        queueing.get("queue_time_fraction_of_request_latency"),
        "queue-time fraction",
    )
    if queue_fraction + 1e-12 < minimum_queue_fraction:
        raise ValueError(
            "accepted A probe fails the queue-time gate: "
            f"{queue_fraction} < {minimum_queue_fraction}"
        )
    preemption_rate = _finite(
        memory.get("preemptions_per_logical_request"),
        "preemptions per logical request",
    )
    if preemption_rate > maximum_preemptions_per_request + 1e-12:
        raise ValueError(
            "accepted A probe fails the preemption gate: "
            f"{preemption_rate} > {maximum_preemptions_per_request}"
        )

    summary = _load_object(cell_dir / "summary.json", "accepted A summary.json")
    environment = _mapping(
        summary.get("scheduler_environment"),
        "accepted A scheduler_environment",
    )
    if environment.get("PASTE_STRESS_PROFILE") != expected_profile:
        raise ValueError(
            "accepted A scheduler profile does not match "
            f"{expected_profile}"
        )
    for name in ENGINE_SHAPE_KEYS:
        actual = environment.get(name)
        expected = expected_engine_shape.get(name)
        if actual != expected:
            raise ValueError(
                f"accepted A engine shape mismatch for {name}: "
                f"{actual!r} != {expected!r}"
            )

    return {
        "probe_path": probe_path.as_posix(),
        "cell_dir": cell_dir.as_posix(),
        "profile": expected_profile,
        "load": expected_load,
        "max_num_seqs": expected_max_num_seqs,
        "waiting_below_cap_sample_fraction": waiting_fraction,
        "queue_time_fraction_of_request_latency": queue_fraction,
        "preemptions_per_logical_request": preemption_rate,
        "engine_shape": dict(expected_engine_shape),
    }


def _unit_interval(raw: str) -> float:
    value = _finite(raw, "threshold")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return value


def _nonnegative(raw: str) -> float:
    value = _finite(raw, "threshold")
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed A-only natural-queue probe before a D-only "
            "screen is allowed to start."
        )
    )
    parser.add_argument("probe_path", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--expected-load", type=int, required=True)
    parser.add_argument("--expected-max-num-seqs", type=int, required=True)
    parser.add_argument("--min-waiting-fraction", type=_unit_interval, required=True)
    parser.add_argument("--min-queue-fraction", type=_unit_interval, required=True)
    parser.add_argument(
        "--max-preemptions-per-request", type=_nonnegative, required=True
    )
    parser.add_argument(
        "--expect-engine-shape",
        action="append",
        default=[],
        metavar="NAME=VALUE",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_accepted_probe(
        args.probe_path,
        repository_root=args.repository_root,
        expected_profile=args.expected_profile,
        expected_load=args.expected_load,
        expected_max_num_seqs=args.expected_max_num_seqs,
        minimum_waiting_fraction=args.min_waiting_fraction,
        minimum_queue_fraction=args.min_queue_fraction,
        maximum_preemptions_per_request=args.max_preemptions_per_request,
        expected_engine_shape=_parse_engine_shape(args.expect_engine_shape),
    )
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
