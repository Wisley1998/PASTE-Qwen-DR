#!/usr/bin/env python3
"""Strict A/C screening with fresh parser-v2 physical-KV validation.

The replay wrapper names a Joint+learned cell ``D`` for manifest compatibility.
For this screen that technical D slot is the conceptual C candidate: FCFS A is
compared with Joint+learned+adaptive-physical-KV C.  A is a previously accepted
fresh-server probe, so this remains screening evidence rather than a fresh
paired replicate.

No saved physical-KV sidecar is trusted.  The candidate's canonical raw vLLM
log, stored telemetry, request events, frozen configuration, and copied A-probe
evidence are all validated again before result boundaries are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from summarize_strict_screening_ad import (  # noqa: E402
    DEFAULT_ENGINE_KEYS,
    summarize_strict_screening,
)
from validate_accepted_a_probe import (  # noqa: E402
    ENGINE_SHAPE_KEYS as A_PROBE_ENGINE_SHAPE_KEYS,
    validate_accepted_probe,
)
from validate_physical_kv_admission_v2 import (  # noqa: E402
    _parse_frozen_exports,
    validate_fresh_physical_kv_cell,
)


SCHEMA = "paste_repro.strict_screening_ac_physical_v2"
VERSION = 1
STATUS = "screening_reuses_previous_a_not_fresh_server_pair"
COMPARATOR_MODULE = Path(__file__).resolve()
STRICT_COMPARATOR_MODULE = SCRIPT_DIRECTORY / "summarize_strict_screening_ad.py"
ACCEPTED_A_VALIDATOR_MODULE = SCRIPT_DIRECTORY / "validate_accepted_a_probe.py"
NATURAL_QUEUE_MODULE = SCRIPT_DIRECTORY / "summarize_natural_queue_probe.py"
CANDIDATE_METRICS_MODULE = SCRIPT_DIRECTORY / "summarize_candidate_d.py"
PAIRED_INFERENCE_MODULE = SCRIPT_DIRECTORY / "summarize_paired_ad.py"
FOUR_CELL_LOADER_MODULE = SCRIPT_DIRECTORY / "summarize_four_cell.py"
MAPPER_MODULE = REPRODUCTION_ROOT / "paste_repro" / "mapper.py"

STRESS300_MANIFEST = (
    REPRODUCTION_ROOT
    / "artifacts/workloads/fixed_three_way_cap512_floor64/manifest_stress300.json"
)
STRESS300_MANIFEST_FILE_SHA256 = (
    "43f6d9dee3f12c4d31f7195e1616fa0ffd21ac98e8a7bdbffe3089be378318fa"
)
STRESS300_A_PROFILE = "stress300_native320_g256_u86_keepalive60_a_probe"
STRESS300_C_PROFILE = (
    "stress300_native320_g256_u86_physical093_exact_rescue120"
)
STRESS300_A_CONFIG_SHA256 = (
    "c1c043836601203c4f49284daf8b7e925bab450747482e486eed83897dda2d06"
)
STRESS300_A_PROBE_SHA256 = (
    "c2a5b098a178e7e9d899ea88995f0f591bb24ec70380c2d5242bc734d2c247bd"
)
STRESS300_LOAD = 300
STRESS300_REQUESTS = 2595
STRESS300_MAX_NUM_SEQS = 320
STRESS300_INSTANCES_PER_SOURCE = 5
STRESS300_NUM_GPU_BLOCKS = 44178
STRESS300_BLOCK_SIZE = 16
STRESS300_KEEPALIVE_S = 60

# Exact A -> conceptual-C configuration delta.  All other recorded scheduler
# configuration keys, including max-num-seqs and legacy HBM estimates, must be
# byte-for-byte identical.
ALLOWED_CONFIG_DIFFERENCES = frozenset(
    {
        "PASTE_FROZEN_CONFIG_SHA256",
        "PASTE_STRESS_PROFILE",
        "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S",
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES",
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
    }
)
PHYSICAL_CONFIG_KEYS = frozenset(
    key
    for key in ALLOWED_CONFIG_DIFFERENCES
    if key.startswith("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_")
)

# Frozen selection and reporting boundaries.  These determine interpretation,
# never whether a completed run is retained as evidence.
MINIMUM_WAITING_FRACTION = 0.50
MINIMUM_QUEUE_FRACTION = 0.20
MAXIMUM_A_PREEMPTIONS_PER_REQUEST = 0.25
MAXIMUM_COMPLETION_TOKEN_RELATIVE_DIFFERENCE = 0.01
MAXIMUM_REQUEST_P99_RATIO = 1.5
MAXIMUM_MAKESPAN_RATIO = 1.03
MINIMUM_MEAN_TASK_REDUCTION = 0.15
MINIMUM_FASTER_SOURCE_COUNT = 48
EXPECTED_INDEPENDENT_SOURCE_COUNT = 60


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside the repository: {resolved}") from exc


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number != value or number < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _validate_accepted_a_binding(
    *,
    probe_path: Path,
    expected_probe_sha256: str,
    a_run: Path,
    c_run: Path,
    expected_a_profile: str,
    expected_a_config_sha256: str,
    expected_load: int,
    expected_max_num_seqs: int,
    expected_keepalive_s: int,
    engine_shape: Mapping[str, str],
) -> dict[str, Any]:
    source = probe_path.resolve()
    expected_sha = _require_sha256(expected_probe_sha256, "accepted A probe SHA256")
    if not source.is_file() or _sha256_file(source) != expected_sha:
        raise ValueError("accepted A probe does not match its preregistered SHA256")
    probe_engine_shape = {
        name: engine_shape[name] for name in A_PROBE_ENGINE_SHAPE_KEYS
    }
    validation = validate_accepted_probe(
        source,
        repository_root=REPOSITORY_ROOT,
        expected_profile=expected_a_profile,
        expected_load=expected_load,
        expected_max_num_seqs=expected_max_num_seqs,
        minimum_waiting_fraction=MINIMUM_WAITING_FRACTION,
        minimum_queue_fraction=MINIMUM_QUEUE_FRACTION,
        maximum_preemptions_per_request=MAXIMUM_A_PREEMPTIONS_PER_REQUEST,
        expected_engine_shape=probe_engine_shape,
    )
    if Path(validation["cell_dir"]).resolve() != a_run.resolve():
        raise ValueError("accepted A probe does not bind the compared A cell")

    a_frozen_config = a_run.resolve().parent / "frozen_config.env"
    a_frozen_sidecar = a_run.resolve().parent / "frozen_config.sha256"
    expected_a_config_sha = _require_sha256(
        expected_a_config_sha256, "accepted A frozen-config SHA256"
    )
    if (
        not a_frozen_config.is_file()
        or _sha256_file(a_frozen_config) != expected_a_config_sha
    ):
        raise ValueError("accepted A frozen config does not match its preregistered SHA256")
    if a_frozen_sidecar.read_text(encoding="utf-8").split() != [
        expected_a_config_sha,
        "frozen_config.env",
    ]:
        raise ValueError("accepted A frozen-config checksum sidecar is invalid")
    if _parse_frozen_exports(a_frozen_config).get(
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE"
    ) != str(expected_keepalive_s):
        raise ValueError("accepted A frozen config does not prove HTTP keep-alive=60")

    run_root = c_run.resolve().parent
    snapshot = run_root / "accepted_a_probe.json"
    checksum = run_root / "accepted_a_probe.sha256"
    validation_snapshot = run_root / "accepted_a_probe_validation.json"
    if not snapshot.is_file() or snapshot.read_bytes() != source.read_bytes():
        raise ValueError("C run-root accepted-A snapshot is missing or differs")
    if checksum.read_text(encoding="utf-8").split() != [
        expected_sha,
        "accepted_a_probe.json",
    ]:
        raise ValueError("C run-root accepted-A checksum sidecar is invalid")
    if _load_object(validation_snapshot, "accepted-A validation snapshot") != validation:
        raise ValueError("C run-root accepted-A validation snapshot differs on recomputation")
    return {
        "source": {
            "path": _repo_relative(source),
            "sha256": expected_sha,
        },
        "copied_snapshot": {
            "path": _repo_relative(snapshot),
            "sha256": _sha256_file(snapshot),
            "byte_exact_match": True,
        },
        "checksum_sidecar": {
            "path": _repo_relative(checksum),
            "sha256": _sha256_file(checksum),
            "exact_match": True,
        },
        "validation_snapshot": {
            "path": _repo_relative(validation_snapshot),
            "sha256": _sha256_file(validation_snapshot),
            "fresh_recomputation_exact_match": True,
        },
        "frozen_config": {
            "path": _repo_relative(a_frozen_config),
            "sha256": expected_a_config_sha,
            "http_timeout_keep_alive_s": expected_keepalive_s,
        },
        "frozen_config_sidecar": {
            "path": _repo_relative(a_frozen_sidecar),
            "sha256": _sha256_file(a_frozen_sidecar),
            "exact_match": True,
        },
        "fresh_validation": validation,
    }


def _reduction(a_value: Any, c_value: Any) -> dict[str, float | None]:
    baseline = _finite(a_value, "A metric")
    candidate = _finite(c_value, "C metric")
    return {
        "a": baseline,
        "c": candidate,
        "a_minus_c_s": baseline - candidate,
        "relative_reduction": (baseline - candidate) / baseline if baseline else None,
    }


def _comparison(
    a_metrics: Mapping[str, Any],
    c_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = {
        "definition": "A - C; positive means conceptual physical-KV C is lower/faster",
        "task_flow_time_s": {
            statistic: _reduction(
                a_metrics["task_flow_time_s"][statistic],
                c_metrics["task_flow_time_s"][statistic],
            )
            for statistic in ("mean", "p50", "p95", "p99", "max")
        },
        "task_makespan_s": _reduction(
            a_metrics["task_makespan_s"], c_metrics["task_makespan_s"]
        ),
        "request_latency_s": {
            statistic: _reduction(
                a_metrics["request_latency_s"][statistic],
                c_metrics["request_latency_s"][statistic],
            )
            for statistic in ("mean", "p50", "p95", "p99", "max")
        },
        "request_tail_counts": {},
        "mean_queue_time_s": _reduction(
            a_metrics["mean_queue_time_s"], c_metrics["mean_queue_time_s"]
        ),
        "mean_nonqueue_request_time_s": _reduction(
            a_metrics["mean_nonqueue_request_time_s"],
            c_metrics["mean_nonqueue_request_time_s"],
        ),
    }
    for field in ("count_gt_120_s", "count_gt_240_s"):
        a_count = _nonnegative_int(a_metrics["request_latency_s"][field], f"A {field}")
        c_count = _nonnegative_int(c_metrics["request_latency_s"][field], f"C {field}")
        comparison["request_tail_counts"][field] = {
            "a": a_count,
            "c": c_count,
            "a_minus_c": a_count - c_count,
        }

    a_execution = a_metrics["execution_accounting"]
    c_execution = c_metrics["execution_accounting"]
    a_tokens = _nonnegative_int(
        a_execution["completion_tokens"]["total"], "A completion tokens"
    )
    c_tokens = _nonnegative_int(
        c_execution["completion_tokens"]["total"], "C completion tokens"
    )
    comparison["execution"] = {
        "completion_tokens": {
            "a_total": a_tokens,
            "c_total": c_tokens,
            "c_minus_a": c_tokens - a_tokens,
            "absolute_relative_difference": (
                abs(c_tokens - a_tokens) / a_tokens if a_tokens else None
            ),
        },
        "retry_accounting": {
            "A": a_metrics["retry_accounting"],
            "C": c_metrics["retry_accounting"],
        },
        "preemption": {
            "a_total": a_execution["preemption"]["num_preemptions_total"],
            "c_total": c_execution["preemption"]["num_preemptions_total"],
        },
        "swap": {
            "A": a_execution["swap"],
            "C": c_execution["swap"],
        },
    }
    return comparison


def _conceptual_source_pairing(source: Mapping[str, Any]) -> dict[str, Any]:
    source_outcomes = source["source_session_outcomes"]
    load_outcomes = source["load_instance_outcomes"]

    def outcomes(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "c_faster": values["d_faster"],
            "tie": values["tie"],
            "c_slower": values["d_slower"],
            "c_faster_fraction": values["d_faster_fraction"],
        }

    rows = []
    for row in source["source_sessions"]:
        copied = dict(row)
        copied["c_task_flow_mean_s"] = copied.pop("d_task_flow_mean_s")
        outcome = copied.get("outcome")
        copied["outcome"] = {
            "d_faster": "c_faster",
            "d_slower": "c_slower",
        }.get(outcome, outcome)
        rows.append(copied)
    bootstrap = dict(source["independent_source_mean_bootstrap_95_ci_s"])
    bootstrap["estimand"] = "mean_A_minus_C_task_flow_s"
    return {
        "definition": (
            "A-C task flow; deterministic load instances are averaged within "
            "each independent source before inference"
        ),
        "load_instance_count": source["load_instance_count"],
        "independent_source_session_count": source[
            "independent_source_session_count"
        ],
        "load_instance_outcomes": outcomes(load_outcomes),
        "source_session_outcomes": outcomes(source_outcomes),
        "source_mean_saving_s": source["source_mean_saving_s"],
        "independent_source_mean_bootstrap_95_ci_s": bootstrap,
        "source_sessions": rows,
    }


def _result_boundaries(
    *,
    cells: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Any],
    source_pairing: Mapping[str, Any],
) -> dict[str, Any]:
    a_metrics, c_metrics = cells["A"], cells["C"]
    token_difference = comparison["execution"]["completion_tokens"][
        "absolute_relative_difference"
    ]
    if token_difference is None:
        raise ValueError("completion-token comparison is undefined")
    a_request_p99 = _finite(a_metrics["request_latency_s"]["p99"], "A request p99")
    c_request_p99 = _finite(c_metrics["request_latency_s"]["p99"], "C request p99")
    a_task_p95 = _finite(a_metrics["task_flow_time_s"]["p95"], "A task p95")
    c_task_p95 = _finite(c_metrics["task_flow_time_s"]["p95"], "C task p95")
    a_makespan = _finite(a_metrics["task_makespan_s"], "A makespan")
    c_makespan = _finite(c_metrics["task_makespan_s"], "C makespan")
    if a_makespan <= 0.0 or c_makespan <= 0.0:
        raise ValueError("A/C makespans must be positive")
    a_gt_240 = _nonnegative_int(
        a_metrics["request_latency_s"]["count_gt_240_s"], "A requests >240s"
    )
    c_gt_240 = _nonnegative_int(
        c_metrics["request_latency_s"]["count_gt_240_s"], "C requests >240s"
    )
    source_count = _nonnegative_int(
        source_pairing["independent_source_session_count"], "source count"
    )
    if source_count != EXPECTED_INDEPENDENT_SOURCE_COUNT:
        raise ValueError(
            f"stress300 promotion requires exactly {EXPECTED_INDEPENDENT_SOURCE_COUNT} "
            f"independent sources, got {source_count}"
        )
    faster_sources = _nonnegative_int(
        source_pairing["source_session_outcomes"]["c_faster"],
        "faster source count",
    )
    bootstrap_lower = _finite(
        source_pairing["independent_source_mean_bootstrap_95_ci_s"]["lower_s"],
        "source bootstrap lower bound",
    )
    mean_reduction = comparison["task_flow_time_s"]["mean"]["relative_reduction"]
    if mean_reduction is None:
        raise ValueError("mean task relative reduction is undefined")

    comparability = {
        "completion_token_absolute_relative_difference_lt_1pct": {
            "passed": token_difference < MAXIMUM_COMPLETION_TOKEN_RELATIVE_DIFFERENCE,
            "observed": token_difference,
            "operator": "<",
            "threshold": MAXIMUM_COMPLETION_TOKEN_RELATIVE_DIFFERENCE,
        },
        "request_p99_not_above_1_5x_a": {
            "passed": c_request_p99 <= MAXIMUM_REQUEST_P99_RATIO * a_request_p99,
            "a_s": a_request_p99,
            "c_s": c_request_p99,
            "ratio": c_request_p99 / a_request_p99 if a_request_p99 else None,
            "maximum_ratio": MAXIMUM_REQUEST_P99_RATIO,
        },
        "request_count_gt_240s_not_increased": {
            "passed": c_gt_240 <= a_gt_240,
            "a": a_gt_240,
            "c": c_gt_240,
        },
        "task_p95_not_regressed": {
            "passed": c_task_p95 <= a_task_p95,
            "a_s": a_task_p95,
            "c_s": c_task_p95,
        },
        "makespan_not_regressed_over_3pct": {
            "passed": c_makespan <= MAXIMUM_MAKESPAN_RATIO * a_makespan,
            "a_s": a_makespan,
            "c_s": c_makespan,
            "ratio": c_makespan / a_makespan if a_makespan else None,
            "maximum_ratio": MAXIMUM_MAKESPAN_RATIO,
        },
    }
    promotion_effect = {
        "mean_task_e2e_reduction_at_least_15pct": {
            "passed": mean_reduction >= MINIMUM_MEAN_TASK_REDUCTION,
            "observed": mean_reduction,
            "minimum": MINIMUM_MEAN_TASK_REDUCTION,
        },
        "at_least_48_of_60_sources_faster": {
            "passed": faster_sources >= MINIMUM_FASTER_SOURCE_COUNT,
            "observed": faster_sources,
            "source_count": source_count,
            "minimum": MINIMUM_FASTER_SOURCE_COUNT,
        },
        "source_bootstrap_95pct_lower_above_zero": {
            "passed": bootstrap_lower > 0.0,
            "observed_lower_s": bootstrap_lower,
            "operator": ">",
            "threshold_s": 0.0,
        },
    }
    comparability_passed = all(item["passed"] for item in comparability.values())
    effect_passed = all(item["passed"] for item in promotion_effect.values())
    makespan_regression = c_makespan / a_makespan - 1.0
    a_throughput = a_metrics["request_count"] / a_makespan
    c_throughput = c_metrics["request_count"] / c_makespan
    throughput_regression = (a_throughput - c_throughput) / a_throughput
    followup_095_permitted = (
        makespan_regression > 0.03 or throughput_regression > 0.03
    )
    return {
        "classification_only_not_run_completion_gates": True,
        "comparability_and_tail": {
            **comparability,
            "passed": comparability_passed,
        },
        "promotion_effect": {
            **promotion_effect,
            "passed": effect_passed,
        },
        "promotion_passed": comparability_passed and effect_passed,
        "followup_095": {
            "physical_093_safety_passed": True,
            "makespan_regression_relative_to_a": makespan_regression,
            "throughput_regression_relative_to_a": throughput_regression,
            "strict_regression_threshold": 0.03,
            "regression_triggered": followup_095_permitted,
            "followup_095_permitted": followup_095_permitted,
            "policy": (
                "A separately preregistered 0.95 screen is permitted only when "
                "this validated-safe 0.93 candidate has makespan or throughput "
                "strictly more than 3% worse than A."
            ),
        },
    }


def summarize_strict_screening_ac_physical_v2(
    *,
    manifest_path: Path,
    a_run: Path,
    c_run: Path,
    accepted_a_probe: Path,
    expected_a_probe_sha256: str,
    expected_a_profile: str,
    expected_c_profile: str,
    expected_a_config_sha256: str,
    expected_c_config_sha256: str,
    expected_load: int,
    expected_requests: int,
    expected_num_gpu_blocks: int,
    expected_block_size: int,
    expected_max_num_seqs: int,
    expected_keepalive_s: int,
) -> dict[str, Any]:
    a_path = a_run.resolve()
    c_path = c_run.resolve()
    a_config_sha = _require_sha256(expected_a_config_sha256, "A config SHA256")
    c_config_sha = _require_sha256(expected_c_config_sha256, "C config SHA256")
    frozen_scope = {
        "manifest_path": manifest_path.resolve() == STRESS300_MANIFEST.resolve(),
        "manifest_file_sha256": (
            manifest_path.is_file()
            and _sha256_file(manifest_path) == STRESS300_MANIFEST_FILE_SHA256
        ),
        "a_profile": expected_a_profile == STRESS300_A_PROFILE,
        "c_profile": expected_c_profile == STRESS300_C_PROFILE,
        "a_config_sha256": a_config_sha == STRESS300_A_CONFIG_SHA256,
        "a_probe_sha256": expected_a_probe_sha256 == STRESS300_A_PROBE_SHA256,
        "load": expected_load == STRESS300_LOAD,
        "requests": expected_requests == STRESS300_REQUESTS,
        "max_num_seqs": expected_max_num_seqs == STRESS300_MAX_NUM_SEQS,
        "num_gpu_blocks": expected_num_gpu_blocks == STRESS300_NUM_GPU_BLOCKS,
        "block_size": expected_block_size == STRESS300_BLOCK_SIZE,
        "keepalive_s": expected_keepalive_s == STRESS300_KEEPALIVE_S,
    }
    failed_scope = [name for name, passed in frozen_scope.items() if not passed]
    if failed_scope:
        raise ValueError(f"stress300 preregistered scope drifted: {failed_scope}")

    expected_a_config = {
        "PASTE_FROZEN_CONFIG_SHA256": a_config_sha,
        "PASTE_STRESS_PROFILE": expected_a_profile,
        "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S": "40",
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES": "1",
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "1",
    }
    expected_c_config = {
        "PASTE_FROZEN_CONFIG_SHA256": c_config_sha,
        "PASTE_STRESS_PROFILE": expected_c_profile,
        "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S": "120",
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES": "0",
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "0",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": "0.93",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": "120",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S": "1",
    }
    strict = summarize_strict_screening(
        manifest_path=manifest_path,
        role="stress",
        a_run=a_path,
        d_run=c_path,
        allowed_config_differences=set(ALLOWED_CONFIG_DIFFERENCES),
        expected_a_config=expected_a_config,
        expected_d_config=expected_c_config,
        expected_a_config_missing=set(PHYSICAL_CONFIG_KEYS),
        expected_d_config_missing=set(),
        expected_a_policy="fcfs",
        expected_d_policy="online_joint_pacer_v2",
        expected_a_overlap="none",
        expected_d_overlap="learned",
        required_engine_keys=DEFAULT_ENGINE_KEYS,
        include_natural_queue_evidence=True,
        # The existing helper intentionally defines natural queueing only for
        # native admission.  C uses dynamic physical admission, so require the
        # native natural-queue gate only for A and use C's raw physical pressure
        # evidence below instead of creating a contradictory prerequisite.
        require_natural_queue=False,
        verify_frozen_configs=True,
    )
    comparison_invariants = strict["comparison_invariants"]
    if (
        comparison_invariants["load_instance_count"] != STRESS300_LOAD
        or comparison_invariants["independent_source_session_count"]
        != EXPECTED_INDEPENDENT_SOURCE_COUNT
        or comparison_invariants["instances_per_source"]
        != STRESS300_INSTANCES_PER_SOURCE
    ):
        raise ValueError("loaded manifest does not have the preregistered stress300 shape")
    if strict["natural_queue_evidence"]["cells"]["A"]["sequence_capacity"].get(
        "natural_vllm_queue_proven"
    ) is not True:
        raise ValueError("accepted A does not prove a native vLLM queue")
    engine_shape = strict["comparison_invariants"]["engine_shape_guard"]["values"]
    if int(engine_shape["VLLM_MAX_NUM_SEQS"]) != expected_max_num_seqs:
        raise ValueError("strict engine shape does not match expected max-num-seqs")
    physical = validate_fresh_physical_kv_cell(
        c_path,
        expected_profile=expected_c_profile,
        expected_load=expected_load,
        expected_requests=expected_requests,
        expected_config_sha256=c_config_sha,
        expected_engine_shape=engine_shape,
        expected_num_gpu_blocks=expected_num_gpu_blocks,
        expected_block_size=expected_block_size,
        expected_target_utilization=0.93,
        expected_keepalive_s=expected_keepalive_s,
        expected_preemptions=0,
    )
    accepted_a = _validate_accepted_a_binding(
        probe_path=accepted_a_probe,
        expected_probe_sha256=expected_a_probe_sha256,
        a_run=a_path,
        c_run=c_path,
        expected_a_profile=expected_a_profile,
        expected_a_config_sha256=a_config_sha,
        expected_load=expected_load,
        expected_max_num_seqs=expected_max_num_seqs,
        expected_keepalive_s=expected_keepalive_s,
        engine_shape=engine_shape,
    )

    cells = {"A": strict["cells"]["A"], "C": strict["cells"]["D"]}
    comparison = _comparison(cells["A"], cells["C"])
    source_pairing = _conceptual_source_pairing(strict["source_pairing"])
    decomposition = dict(strict["task_saving_decomposition"])
    decomposition["definition"] = (
        "A component - C component; positive contributes to conceptual C saving"
    )
    boundaries = _result_boundaries(
        cells=cells,
        comparison=comparison,
        source_pairing=source_pairing,
    )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": STATUS,
        "runner_role_mapping": {
            "manifest_baseline_slot": "A",
            "manifest_candidate_slot": "D",
            "conceptual_candidate_label": "C",
            "meaning": "C = Joint + learned overlap + adaptive physical-KV admission",
        },
        "comparison_invariants": strict["comparison_invariants"],
        "preregistered_stress300_scope": {
            "passed": True,
            "checks": frozen_scope,
            "manifest_file_sha256": STRESS300_MANIFEST_FILE_SHA256,
            "load_instance_count": STRESS300_LOAD,
            "independent_source_session_count": EXPECTED_INDEPENDENT_SOURCE_COUNT,
            "instances_per_source": STRESS300_INSTANCES_PER_SOURCE,
            "request_count": STRESS300_REQUESTS,
            "max_num_seqs": STRESS300_MAX_NUM_SEQS,
            "num_gpu_blocks": STRESS300_NUM_GPU_BLOCKS,
            "block_size": STRESS300_BLOCK_SIZE,
            "keepalive_s": STRESS300_KEEPALIVE_S,
        },
        "accepted_a_probe_evidence": accepted_a,
        "physical_kv_admission_evidence": physical,
        "cells": cells,
        "comparison": comparison,
        "source_pairing": source_pairing,
        "task_saving_decomposition": decomposition,
        "natural_queue_evidence": {
            "A_native_natural_queue_proven": True,
            "C_dynamic_physical_queue_pressure_proven": (
                physical["physical_kv"]["pressure_above_64_sample_count"] >= 10
            ),
            "definitions_are_intentionally_distinct": True,
            "cells": {
                "A": strict["natural_queue_evidence"]["cells"]["A"],
                "C": strict["natural_queue_evidence"]["cells"]["D"],
            },
        },
        "result_boundaries": boundaries,
        "code_binding": {
            "comparator": {
                "path": _repo_relative(COMPARATOR_MODULE),
                "sha256": _sha256_file(COMPARATOR_MODULE),
            },
            "physical_validator": physical["code_binding"],
            "result_dependencies": {
                name: {
                    "path": _repo_relative(path),
                    "sha256": _sha256_file(path),
                }
                for name, path in {
                    "strict_comparator": STRICT_COMPARATOR_MODULE,
                    "accepted_a_validator": ACCEPTED_A_VALIDATOR_MODULE,
                    "natural_queue": NATURAL_QUEUE_MODULE,
                    "candidate_metrics": CANDIDATE_METRICS_MODULE,
                    "paired_inference": PAIRED_INFERENCE_MODULE,
                    "four_cell_loader": FOUR_CELL_LOADER_MODULE,
                    "atomic_mapper_writer": MAPPER_MODULE,
                }.items()
            },
        },
        "interpretation": (
            "A and C have identical deterministic workload identities and engine "
            "shape, but A is reused from its accepted fresh-server probe. This is "
            "a strict A/C screen of the total Joint+learned+physical-admission "
            "bundle, not a fresh-server paired replicate and not an estimate of "
            "physical admission alone. Promotion boundaries classify the retained "
            "result and do not suppress a completed safe run."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--c-run", type=Path, required=True)
    parser.add_argument("--accepted-a-probe", type=Path, required=True)
    parser.add_argument("--expected-a-probe-sha256", required=True)
    parser.add_argument("--expected-a-profile", required=True)
    parser.add_argument("--expected-c-profile", required=True)
    parser.add_argument("--expected-a-config-sha256", required=True)
    parser.add_argument("--expected-c-config-sha256", required=True)
    parser.add_argument("--expected-load", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-num-gpu-blocks", type=int, required=True)
    parser.add_argument("--expected-block-size", type=int, required=True)
    parser.add_argument("--expected-max-num-seqs", type=int, required=True)
    parser.add_argument("--expected-keepalive-s", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    c_path = args.c_run.resolve()
    output = args.output.resolve()
    if output.parent != c_path.parent or output.suffix != ".json":
        raise ValueError("--output must be a JSON file directly under the C run root")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing comparison evidence: {output}")
    result = summarize_strict_screening_ac_physical_v2(
        manifest_path=args.manifest,
        a_run=args.a_run,
        c_run=c_path,
        accepted_a_probe=args.accepted_a_probe,
        expected_a_probe_sha256=args.expected_a_probe_sha256,
        expected_a_profile=args.expected_a_profile,
        expected_c_profile=args.expected_c_profile,
        expected_a_config_sha256=args.expected_a_config_sha256,
        expected_c_config_sha256=args.expected_c_config_sha256,
        expected_load=args.expected_load,
        expected_requests=args.expected_requests,
        expected_num_gpu_blocks=args.expected_num_gpu_blocks,
        expected_block_size=args.expected_block_size,
        expected_max_num_seqs=args.expected_max_num_seqs,
        expected_keepalive_s=args.expected_keepalive_s,
    )
    write_json_atomic(output, result)
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
