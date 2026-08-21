#!/usr/bin/env python3
"""Strictly validate the v2 single-token native-prefix causal matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "paste_repro.native_prefix_causal_validation_v2"
PLAN_SCHEMA = "paste_repro.native_prefix_causal_plan_v2"
CELL_SCHEMA = "paste_repro.native_prefix_prompt_cell_v2"
CELL_MANIFEST_SCHEMA = "paste_repro.native_prefix_causal_cell_evidence_v2"
EFFECTIVE_CONFIG_SCHEMA = "paste_repro.native_prefix_causal_cell_config_v2"
SERVER_IDENTITY_SCHEMA = "paste_repro.native_prefix_causal_server_identity_v2"
FIXTURE_VERSION = "three-call-local-prefix-single-token-fixture-v2"
PROTOCOL_VERSION = "native-prefix-causal-v2"
OUTPUT_CONSTRAINT = "guided_choice_singleton_v1"
SENTINEL = "A"
SENTINEL_TOKEN_ID = 32
PROTOCOL_PATH = "reproduction/results/live_joint/NATIVE_PREFIX_CAUSAL_DEV_PROTOCOL.md"
VALIDATOR_PATH = "reproduction/scripts/validate_native_prefix_causal_dev.py"
CELL_RUNNER_PATH = "reproduction/scripts/run_native_prefix_prompt_cell.py"
EXPECTED_ORDERS = (("P0", "P1"), ("P1", "P0"))
EXPECTED_CELLS = frozenset({"P0", "P1"})
REQUIRED_METRICS = (
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)
EXACT_ENGINE = {
    "VLLM_MAX_MODEL_LEN": "16384",
    "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
    "VLLM_MAX_NUM_SEQS": "96",
    "VLLM_USE_V1": "1",
    "VLLM_SCHED_POLICY": "fcfs",
}
EXACT_MATRIX = {
    "source_count": 16,
    "replicas": 3,
    "task_count": 48,
    "calls_per_task": 3,
    "context_padding_tokens": 10000,
    "visit_fixture_tokens": 900,
    "max_active_tasks": 48,
    "max_tokens_by_call": [1, 1, 1],
    "sentinel": SENTINEL,
    "output_constraint": OUTPUT_CONSTRAINT,
}
EXACT_THRESHOLDS = {
    "min_native_hit_ratio": 0.60,
    "min_prefill_reduction": 0.15,
    "min_mean_request_reduction": 0.03,
    "min_mean_task_e2e_reduction": 0.03,
    "max_task_p95_ratio": 1.03,
    "max_completion_token_relative_difference": 0.01,
    "bootstrap_samples": 10000,
    "bootstrap_seed": 20260816,
}
EXACT_VLLM_ENV_KEYS = frozenset(
    {
        "VLLM_CUDA_GRAPH_SIZES",
        "VLLM_DTYPE",
        "VLLM_ENABLE_PREFIX_CACHING",
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_HOOK_DIR",
        "VLLM_HOST",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
        "VLLM_LOG_DIR",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_NO_USAGE_STATS",
        "VLLM_PORT",
        "VLLM_PROBE_HOST",
        "VLLM_READY_TIMEOUT",
        "VLLM_REQUIRE_NEW",
        "VLLM_SCHED_POLICY",
        "VLLM_SHUTDOWN_TIMEOUT",
        "VLLM_STATE_DIR",
        "VLLM_TP_SIZE",
        "VLLM_USE_V1",
    }
)


class ValidationError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sentinel_contract() -> dict[str, Any]:
    return {
        "contract": OUTPUT_CONSTRAINT,
        "sentinel": SENTINEL,
        "sentinel_utf8_sha256": hashlib.sha256(SENTINEL.encode("utf-8")).hexdigest(),
        "token_id": SENTINEL_TOKEN_ID,
        "token_ids_sha256": sha256_json([SENTINEL_TOKEN_ID]),
        "token_count": 1,
        "allowed_choice_count": 1,
        "guided_choice": [SENTINEL],
        "guided_choice_sha256": sha256_json([SENTINEL]),
        "max_tokens": 1,
        "round_trip_exact": True,
        "special_token": False,
    }


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValidationError(f"{label} must be positive")
    return value


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValidationError(f"{label} must be finite and non-negative")
    return result


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValidationError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is invalid JSON: {path}") from exc
    return _mapping(value, label)


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValidationError("cannot take a percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValidationError("relative-reduction baseline must be positive")
    return (baseline - candidate) / baseline


def _relative_difference(left: int | float, right: int | float) -> float:
    denominator = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left) - float(right)) / denominator


def _bootstrap_source_savings(
    source_savings: Mapping[str, float], *, samples: int, seed: int
) -> dict[str, Any]:
    ordered = sorted(source_savings)
    if not ordered:
        raise ValidationError("source bootstrap has no sources")
    values = [float(source_savings[source]) for source in ordered]
    rng = random.Random(seed)
    draws = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return {
        "method": "paired_source_nonparametric_bootstrap",
        "samples": samples,
        "seed": seed,
        "source_count": len(values),
        "lower_s": _percentile(draws, 0.025),
        "upper_s": _percentile(draws, 0.975),
    }


def _resolve_bound_path(relative: str) -> Path:
    path = (REPOSITORY_ROOT / relative).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValidationError(f"binding escapes repository: {relative}") from exc
    return path


def _validate_plan(run_root: Path) -> Mapping[str, Any]:
    plan = _load_json(run_root / "run_plan.json", "run plan")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("version") != 2:
        raise ValidationError("run plan schema mismatch")
    if (
        plan.get("prospective_version") != PROTOCOL_VERSION
        or plan.get("prior_r1_disposition")
        != "rejected_diagnostic_not_validatable_as_v2"
        or plan.get("native_pythonpath_isolated") is not True
        or plan.get("explicit_prefix_locality_enabled") is not False
        or plan.get("pinned_vllm_version") != "0.10.1"
        or plan.get("only_treatment_variable") != "VLLM_ENABLE_PREFIX_CACHING"
    ):
        raise ValidationError("run plan does not guarantee native isolated FCFS")
    orders = tuple(tuple(row) for row in _sequence(plan.get("orders"), "orders"))
    if orders != EXPECTED_ORDERS:
        raise ValidationError("matrix must use exactly P0,P1 then P1,P0")
    if dict(_mapping(plan.get("matrix"), "matrix")) != EXACT_MATRIX:
        raise ValidationError("frozen matrix shape mismatch")
    if dict(_mapping(plan.get("thresholds"), "thresholds")) != EXACT_THRESHOLDS:
        raise ValidationError("prospective threshold set mismatch")
    engine = _mapping(plan.get("engine"), "engine")
    for key, expected in EXACT_ENGINE.items():
        if engine.get(key) != expected:
            raise ValidationError(f"run-plan engine mismatch: {key}")
    if engine.get("VLLM_ENABLE_PREFIX_CACHING") != "per_cell_P0_0_P1_1":
        raise ValidationError("run plan does not isolate the prefix-cache flag")
    if int(engine.get("VLLM_MAX_NUM_SEQS", "0")) <= EXACT_MATRIX["task_count"]:
        raise ValidationError("max-num-seqs is not above offered task concurrency")
    preflight = _mapping(plan.get("fixture_preflight"), "fixture preflight")
    if (
        preflight.get("task_count") != 48
        or preflight.get("call_count") != 144
        or _integer(
            preflight.get("max_prompt_plus_generation_cap"),
            "fixture maximum context",
            positive=True,
        )
        >= 16384
        or re.fullmatch(
            r"[0-9a-f]{64}",
            _string(
                preflight.get("fixture_manifest_sha256"),
                "fixture preflight manifest",
            ),
        )
        is None
    ):
        raise ValidationError("CPU fixture preflight evidence mismatch")
    sentinel_contract = dict(
        _mapping(preflight.get("sentinel_contract"), "preflight sentinel contract")
    )
    if sentinel_contract != exact_sentinel_contract():
        raise ValidationError("CPU tokenizer sentinel proof mismatch")
    if dict(_mapping(plan.get("generation_contract"), "generation contract")) != (
        sentinel_contract
    ):
        raise ValidationError("run plan generation contract differs from preflight")
    workload = _mapping(plan.get("workload"), "workload")
    workload_path = _resolve_bound_path(_string(workload.get("path"), "workload.path"))
    if sha256_file(workload_path) != _string(workload.get("sha256"), "workload.sha256"):
        raise ValidationError("workload binding changed")
    workload_payload = _load_json(workload_path, "frozen tune workload")
    if (
        workload_payload.get("split_id")
        != "live-joint-wikipedia-frozen-tune-v1"
        or workload_payload.get("split_role") != "tune"
        or workload_payload.get("formal_eligible") is not False
        or len(_sequence(workload_payload.get("sources"), "workload sources")) != 16
    ):
        raise ValidationError("workload is not the frozen non-formal tune split")
    bindings = _mapping(plan.get("bindings"), "bindings")
    if len(bindings) < 6:
        raise ValidationError("run plan lacks complete code/config bindings")
    for relative, expected in bindings.items():
        path = _resolve_bound_path(_string(relative, "binding path"))
        if not path.is_file() or sha256_file(path) != _string(
            expected, f"binding {relative}"
        ):
            raise ValidationError(f"bound input changed: {relative}")
    contract_bindings = _mapping(plan.get("contract_bindings"), "contract bindings")
    if set(contract_bindings) != {
        "protocol",
        "validator",
        "cell_runner",
        "prior_r1_disposition",
    }:
        raise ValidationError("contract binding set mismatch")
    if (
        contract_bindings.get("prior_r1_disposition")
        != "rejected_diagnostic_not_validatable_as_v2"
    ):
        raise ValidationError("r1 rejection disposition changed")
    expected_contracts = {
        "protocol": (PROTOCOL_PATH, PROTOCOL_VERSION, None),
        "validator": (VALIDATOR_PATH, None, SCHEMA),
        "cell_runner": (CELL_RUNNER_PATH, None, CELL_SCHEMA),
    }
    for name, (expected_path, version, schema) in expected_contracts.items():
        row = _mapping(contract_bindings.get(name), f"contract binding {name}")
        relative = _string(row.get("path"), f"contract binding {name}.path")
        digest = _string(row.get("sha256"), f"contract binding {name}.sha256")
        if relative != expected_path or bindings.get(relative) != digest:
            raise ValidationError(f"contract binding {name} is not SHA-bound")
        if version is not None and row.get("version") != version:
            raise ValidationError(f"contract binding {name} version mismatch")
        if schema is not None and row.get("schema") != schema:
            raise ValidationError(f"contract binding {name} schema mismatch")
    return plan


def _validate_evidence_manifest(
    cell_root: Path,
    *,
    block_id: str,
    cell_id: str,
    order_index: int,
) -> Mapping[str, Any]:
    manifest = _load_json(cell_root / "cell_manifest.json", "cell manifest")
    if (
        manifest.get("schema") != CELL_MANIFEST_SCHEMA
        or manifest.get("version") != 2
        or manifest.get("block_id") != block_id
        or manifest.get("cell_id") != cell_id
        or manifest.get("order_index") != order_index
    ):
        raise ValidationError(f"{block_id}/{cell_id} cell manifest mismatch")
    evidence = _mapping(manifest.get("evidence"), "cell evidence")
    required_names = {
        "effective_config.json",
        "server_identity.json",
        "evidence/result.json",
        "evidence/queue_timeline.jsonl",
        "evidence/metrics_before.prom",
        "evidence/metrics_after.prom",
        "server/vllm_8100.log",
        "server_lifecycle.stdout.log",
        "server_lifecycle.stderr.log",
        "runner.stdout.log",
        "runner.stderr.log",
    }
    if set(evidence) != required_names:
        raise ValidationError(f"{block_id}/{cell_id} evidence set mismatch")
    for relative, expected_sha in evidence.items():
        path = (cell_root / relative).resolve()
        try:
            path.relative_to(cell_root.resolve())
        except ValueError as exc:
            raise ValidationError("cell evidence path escapes cell root") from exc
        if not path.is_file() or sha256_file(path) != _string(
            expected_sha, f"evidence SHA {relative}"
        ):
            raise ValidationError(
                f"{block_id}/{cell_id} evidence hash mismatch: {relative}"
            )
    return manifest


def _validate_server_log(
    log_path: Path,
    *,
    prefix_enabled: bool,
    block_id: str,
    cell_id: str,
) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    expected_bool = "True" if prefix_enabled else "False"
    api_line = re.search(r"non-default args: \{[^\n]*", text)
    engine_line = re.search(
        r"Initializing a V1 LLM engine \(v0\.10\.1\)[^\n]*", text
    )
    if api_line is None or engine_line is None:
        raise ValidationError(
            f"{block_id}/{cell_id} lacks pinned API/engine startup evidence"
        )
    api_text = api_line.group(0)
    engine_text = engine_line.group(0)
    if (
        f"'enable_prefix_caching': {expected_bool}" not in api_text
        or f"enable_prefix_caching={expected_bool}" not in engine_text
    ):
        raise ValidationError(
            f"{block_id}/{cell_id} effective prefix-cache log mismatch"
        )
    required_api = {
        "api_max_model_len": "'max_model_len': 16384",
        "api_max_batched_tokens": "'max_num_batched_tokens': 2048",
        "api_max_num_seqs": "'max_num_seqs': 96",
    }
    missing = [name for name, marker in required_api.items() if marker not in api_text]
    if "max_seq_len=16384" not in engine_text:
        missing.append("engine_max_model_len")
    if missing:
        raise ValidationError(
            f"{block_id}/{cell_id} server log lacks engine evidence: {missing}"
        )
    forbidden = (
        "[sched_policy_patch]",
        "[sched_policy_patch:",
        "online_joint_pacer",
        "VLLM_SCHED_JOINT",
        "VLLM_SCHED_HBM",
    )
    present_forbidden = [marker for marker in forbidden if marker in text]
    if present_forbidden:
        raise ValidationError(
            f"{block_id}/{cell_id} native FCFS log contains Joint patch evidence: "
            f"{present_forbidden}"
        )
    return {
        "api_prefix_cache_enabled": prefix_enabled,
        "engine_prefix_cache_enabled": prefix_enabled,
        "scheduler_patch_markers": 0,
        "max_model_len": 16384,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 96,
        "vllm_v1": True,
        "vllm_version": "0.10.1",
    }


def _load_queue_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"queue sample {line_number} is invalid JSON") from exc
        rows.append(_mapping(value, f"queue sample {line_number}"))
    if not rows:
        raise ValidationError("queue timeline is empty")
    return rows


def _validate_server_identity(
    cell_root: Path, *, server_instance_id: str
) -> dict[str, Any]:
    identity = _load_json(cell_root / "server_identity.json", "server identity")
    if (
        identity.get("schema") != SERVER_IDENTITY_SCHEMA
        or identity.get("version") != 2
        or identity.get("server_instance_id") != server_instance_id
    ):
        raise ValidationError("fresh server process identity schema mismatch")
    process = {
        "pid": _integer(identity.get("pid"), "server PID", positive=True),
        "proc_start_ticks": _integer(
            identity.get("proc_start_ticks"), "server start ticks", positive=True
        ),
        "executable": _string(identity.get("executable"), "server executable"),
        "cmdline_sha256": _string(
            identity.get("cmdline_sha256"), "server command SHA"
        ),
    }
    if (
        not Path(process["executable"]).is_absolute()
        or re.fullmatch(r"[0-9a-f]{64}", process["cmdline_sha256"]) is None
        or _finite(
            identity.get("captured_wall_s"),
            "server identity capture time",
            nonnegative=True,
        )
        <= 0
    ):
        raise ValidationError("fresh server process identity is malformed")
    expected_sha = sha256_json(process)
    if identity.get("process_identity_sha256") != expected_sha:
        raise ValidationError("fresh server process identity digest mismatch")
    return {**process, "process_identity_sha256": expected_sha}


def _parse_prometheus_snapshot(path: Path) -> dict[str, list[dict[str, Any]]]:
    from prometheus_client.parser import text_string_to_metric_families

    text = path.read_text(encoding="utf-8")
    parsed: dict[str, list[dict[str, Any]]] = {}
    try:
        families = text_string_to_metric_families(text)
        for family in families:
            for sample in family.samples:
                if sample.name not in REQUIRED_METRICS:
                    continue
                parsed.setdefault(sample.name, []).append(
                    {
                        "labels": dict(sample.labels),
                        "value": float(sample.value),
                    }
                )
    except Exception as exc:
        raise ValidationError(f"invalid raw Prometheus snapshot: {path}") from exc
    return parsed


def _validate_raw_metrics(
    cell_root: Path,
    *,
    raw_evidence: Mapping[str, Any],
    reported_deltas: Mapping[str, float],
    model_name: str,
) -> dict[str, Any]:
    snapshots: dict[str, dict[str, list[dict[str, Any]]]] = {}
    hashes: dict[str, str] = {}
    for phase in ("before", "after"):
        path = cell_root / f"evidence/metrics_{phase}.prom"
        record = _mapping(raw_evidence.get(f"metrics_{phase}"), f"metrics {phase}")
        digest = sha256_file(path)
        if record.get("path") != str(path) or record.get("sha256") != digest:
            raise ValidationError(f"raw Prometheus {phase} hash mismatch")
        hashes[phase] = digest
        snapshots[phase] = _parse_prometheus_snapshot(path)
    selected: dict[str, dict[str, Any]] = {}
    for metric in REQUIRED_METRICS:
        phase_rows: dict[str, dict[str, Any]] = {}
        for phase in ("before", "after"):
            rows = snapshots[phase].get(metric, [])
            if len(rows) != 1:
                raise ValidationError(
                    f"raw metric {metric} must expose exactly one engine series"
                )
            row = rows[0]
            labels = _mapping(row.get("labels"), f"raw metric labels {metric}")
            if labels.get("engine") != "0" or labels.get("model_name") != model_name:
                raise ValidationError(f"raw metric {metric} labels changed")
            phase_rows[phase] = row
        if phase_rows["before"]["labels"] != phase_rows["after"]["labels"]:
            raise ValidationError(f"raw metric {metric} label identity changed")
        delta = float(phase_rows["after"]["value"]) - float(
            phase_rows["before"]["value"]
        )
        if delta < 0 or not math.isclose(
            delta,
            reported_deltas[metric],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValidationError(f"raw metric {metric} delta mismatch or reset")
        selected[metric] = {
            "labels": dict(phase_rows["after"]["labels"]),
            "before": phase_rows["before"]["value"],
            "after": phase_rows["after"]["value"],
            "delta": delta,
        }
    return {"sha256": hashes, "series": selected}


def _validate_cell(
    cell_root: Path,
    *,
    plan: Mapping[str, Any],
    block_number: int,
    cell_id: str,
    order_index: int,
) -> dict[str, Any]:
    run_tag = _string(plan.get("run_tag"), "run_tag")
    block_id = f"{run_tag}-block-{block_number}"
    manifest = _validate_evidence_manifest(
        cell_root,
        block_id=block_id,
        cell_id=cell_id,
        order_index=order_index,
    )
    server_instance_id = _string(
        manifest.get("server_instance_id"), "server_instance_id"
    )
    server_process = _validate_server_identity(
        cell_root, server_instance_id=server_instance_id
    )
    effective = _load_json(cell_root / "effective_config.json", "effective config")
    prefix_enabled = cell_id == "P1"
    if (
        effective.get("schema") != EFFECTIVE_CONFIG_SCHEMA
        or effective.get("version") != 2
        or effective.get("block_id") != block_id
        or effective.get("block_number") != block_number
        or effective.get("cell_id") != cell_id
        or effective.get("order_index") != order_index
        or effective.get("server_instance_id") != server_instance_id
        or effective.get("fresh_server_required") is not True
        or effective.get("native_prefix_cache_enabled") is not prefix_enabled
        or effective.get("scheduler_policy") != "fcfs"
        or effective.get("native_pythonpath_isolated") is not True
        or effective.get("explicit_prefix_locality_enabled") is not False
        or effective.get("external_network_allowed") is not False
        or effective.get("external_tools_allowed") is not False
    ):
        raise ValidationError(f"{block_id}/{cell_id} effective config mismatch")
    environment = _mapping(effective.get("environment"), "effective environment")
    vllm_keys = {key for key in environment if key.startswith("VLLM_")}
    if vllm_keys != EXACT_VLLM_ENV_KEYS:
        raise ValidationError(
            f"{block_id}/{cell_id} inherited or missing vLLM environment: "
            f"{sorted(vllm_keys ^ EXACT_VLLM_ENV_KEYS)}"
        )
    scheduler_keys = {key for key in environment if key.startswith("VLLM_SCHED_")}
    if scheduler_keys != {"VLLM_SCHED_POLICY"}:
        raise ValidationError(f"{block_id}/{cell_id} leaked Joint scheduler variables")
    for key, expected in EXACT_ENGINE.items():
        if environment.get(key) != expected:
            raise ValidationError(f"{block_id}/{cell_id} engine env mismatch: {key}")
    if environment.get("VLLM_ENABLE_PREFIX_CACHING") != (
        "1" if prefix_enabled else "0"
    ):
        raise ValidationError(f"{block_id}/{cell_id} prefix env mismatch")
    native_pythonpath = (cell_root / "native_pythonpath").resolve()
    if (
        environment.get("PYTHONPATH") != ""
        or environment.get("VLLM_HOOK_DIR") != str(native_pythonpath)
        or effective.get("native_pythonpath") != str(native_pythonpath)
        or not native_pythonpath.is_dir()
        or any(native_pythonpath.iterdir())
    ):
        raise ValidationError(f"{block_id}/{cell_id} native PYTHONPATH isolation failed")

    log_evidence = _validate_server_log(
        cell_root / "server/vllm_8100.log",
        prefix_enabled=prefix_enabled,
        block_id=block_id,
        cell_id=cell_id,
    )
    result = _load_json(cell_root / "evidence/result.json", "cell result")
    if result.get("schema") != CELL_SCHEMA or result.get("version") != 2:
        raise ValidationError(f"{block_id}/{cell_id} result schema mismatch")
    config = _mapping(result.get("config"), "cell result config")
    expected_result_config = {
        "fixture_version": FIXTURE_VERSION,
        "output_constraint": OUTPUT_CONSTRAINT,
        "cell_id": cell_id,
        "block_id": block_id,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "fresh_server": True,
        "prefix_cache_enabled": prefix_enabled,
        "scheduler_policy": "fcfs",
        "native_pythonpath_isolated": True,
        "explicit_prefix_locality_enabled": False,
        "external_network_used": False,
        "external_tools_executed": False,
        "deterministic_local_fixture": True,
        "later_prompts_use_runtime_completion": False,
        "calls_per_task": 3,
        "source_count": 16,
        "replicas": 3,
        "task_count": 48,
        "max_active_tasks": 48,
        "context_padding_tokens": 10000,
        "visit_fixture_tokens": 900,
        "max_tokens_by_call": [1, 1, 1],
        "max_model_len": 16384,
        "server_url": "http://127.0.0.1:8100",
        "model": _mapping(plan.get("engine"), "engine")["MODEL_ID"],
        "workload_sha256": _mapping(plan.get("workload"), "workload")["sha256"],
        "workload_split_id": "live-joint-wikipedia-frozen-tune-v1",
        "workload_formal_eligible": False,
    }
    changed = [
        key for key, expected in expected_result_config.items() if config.get(key) != expected
    ]
    if changed:
        raise ValidationError(
            f"{block_id}/{cell_id} frozen result config mismatch: {changed}"
        )
    if dict(_mapping(config.get("sentinel_contract"), "cell sentinel contract")) != (
        exact_sentinel_contract()
    ):
        raise ValidationError(f"{block_id}/{cell_id} sentinel contract mismatch")
    if config.get("scheduler_environment") != {"VLLM_SCHED_POLICY": "fcfs"}:
        raise ValidationError(f"{block_id}/{cell_id} result leaked scheduler knobs")
    fixture_manifest_sha = _string(
        config.get("fixture_manifest_sha256"), "fixture manifest SHA"
    )
    if re.fullmatch(r"[0-9a-f]{64}", fixture_manifest_sha) is None:
        raise ValidationError("fixture manifest identity is not SHA256")
    if fixture_manifest_sha != _mapping(
        plan.get("fixture_preflight"), "fixture preflight"
    ).get("fixture_manifest_sha256"):
        raise ValidationError("runtime fixture differs from CPU preflight")
    if config.get("engine_environment") != {
        key: environment[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "MODEL_ID",
            "MODEL_REVISION",
            "VLLM_PORT",
            "VLLM_MAX_MODEL_LEN",
            "VLLM_MAX_NUM_BATCHED_TOKENS",
            "VLLM_MAX_NUM_SEQS",
            "VLLM_ENABLE_PREFIX_CACHING",
            "VLLM_USE_V1",
            "VLLM_SCHED_POLICY",
        )
    }:
        raise ValidationError(f"{block_id}/{cell_id} process environment mismatch")

    tasks = _sequence(result.get("tasks"), "tasks")
    events = _sequence(result.get("llm_events"), "llm_events")
    if len(tasks) != 48 or len(events) != 144:
        raise ValidationError(f"{block_id}/{cell_id} exact completion counts mismatch")
    task_map: dict[str, Mapping[str, Any]] = {}
    for index, task_raw in enumerate(tasks):
        task = _mapping(task_raw, f"task {index}")
        task_id = _string(task.get("task_id"), f"task {index}.task_id")
        if task_id in task_map:
            raise ValidationError("task IDs are not unique")
        if (
            task.get("ok") is not True
            or task.get("completed_call_indices") != [0, 1, 2]
            or _integer(task.get("replica"), f"task {index}.replica") not in range(3)
            or _integer(
                task.get("context_padding_actual_tokens"),
                f"task {index}.padding",
            ) < 10000
        ):
            raise ValidationError(f"{block_id}/{cell_id} task {task_id} is incomplete")
        _string(task.get("source_id"), f"task {index}.source_id")
        _finite(task.get("e2e_s"), f"task {index}.e2e", nonnegative=True)
        started = _finite(task.get("started_wall_s"), f"task {index}.start")
        ended = _finite(task.get("ended_wall_s"), f"task {index}.end")
        if ended < started:
            raise ValidationError("task timestamps are not monotonic")
        task_map[task_id] = task
    workload_path = _resolve_bound_path(
        _string(
            _mapping(plan.get("workload"), "workload").get("path"),
            "workload.path",
        )
    )
    workload_payload = _load_json(workload_path, "frozen tune workload")
    expected_source_ids = {
        _string(row.get("source_id"), "workload source_id")
        for row in (
            _mapping(value, "workload source")
            for value in _sequence(workload_payload.get("sources"), "workload sources")
        )
    }
    observed_source_ids = {str(task["source_id"]) for task in task_map.values()}
    if observed_source_ids != expected_source_ids or len(observed_source_ids) != 16:
        raise ValidationError("cell does not contain exactly 16 independent sources")
    for task_id, task in task_map.items():
        expected_task_id = f"{task['source_id']}__r{int(task['replica']):02d}"
        if task_id != expected_task_id:
            raise ValidationError("task ID is not the frozen source/replica identity")

    event_map: dict[tuple[str, int], Mapping[str, Any]] = {}
    event_durations_by_task: dict[str, float] = {task_id: 0.0 for task_id in task_map}
    prompt_tokens_total = 0
    completion_tokens_total = 0
    for index, event_raw in enumerate(events):
        event = _mapping(event_raw, f"event {index}")
        task_id = _string(event.get("task_id"), f"event {index}.task_id")
        call_index = _integer(event.get("call_index"), f"event {index}.call_index")
        key = (task_id, call_index)
        if task_id not in task_map or call_index not in range(3) or key in event_map:
            raise ValidationError(f"{block_id}/{cell_id} invalid request identity")
        if (
            event.get("ok") is not True
            or event.get("attempts") != 1
            or event.get("http_status") != 200
        ):
            raise ValidationError(f"{block_id}/{cell_id} request was not exactly once")
        for digest_field in (
            "messages_sha256",
            "prompt_token_ids_sha256",
            "request_payload_sha256",
            "guided_choice_sha256",
            "expected_completion_sha256",
            "response_sha256",
            "semantic_response_sha256",
        ):
            digest = _string(event.get(digest_field), f"event {index}.{digest_field}")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValidationError(f"event {index}.{digest_field} is not SHA256")
        expected_completion = _string(
            event.get("expected_completion"), f"event {index}.expected_completion"
        )
        if expected_completion != SENTINEL:
            raise ValidationError("expected completion is not the frozen sentinel")
        if event["guided_choice_sha256"] != sha256_json([SENTINEL]):
            raise ValidationError("guided choice is not the frozen singleton")
        if sha256_json(expected_completion) != event["expected_completion_sha256"]:
            raise ValidationError("expected completion digest mismatch")
        response = _string(event.get("response"), f"event {index}.response")
        if hashlib.sha256(response.encode("utf-8")).hexdigest() != event["response_sha256"]:
            raise ValidationError("raw completion digest mismatch")
        if response != expected_completion:
            raise ValidationError("guided completion differs from one-token sentinel")
        if sha256_json(response) != event["semantic_response_sha256"]:
            raise ValidationError("semantic completion digest mismatch")
        usage = _mapping(event.get("usage"), f"event {index}.usage")
        prompt_tokens = _integer(
            usage.get("prompt_tokens"), f"event {index}.prompt_tokens", positive=True
        )
        completion_tokens = _integer(
            usage.get("completion_tokens"),
            f"event {index}.completion_tokens",
            positive=True,
        )
        total_tokens = _integer(
            usage.get("total_tokens"), f"event {index}.total_tokens", positive=True
        )
        if total_tokens != prompt_tokens + completion_tokens:
            raise ValidationError("server usage token identity failed")
        if completion_tokens != 1:
            raise ValidationError("sentinel completion was not exactly one token")
        if event.get("prompt_tokens_estimate") != prompt_tokens:
            raise ValidationError("local and server prompt token counts differ")
        max_tokens = _integer(event.get("max_tokens"), f"event {index}.max_tokens")
        if max_tokens != 1:
            raise ValidationError("per-call generation cap changed")
        expected_completion_tokens = _integer(
            event.get("expected_completion_tokens_estimate"),
            f"event {index}.expected completion tokens",
            positive=True,
        )
        if expected_completion_tokens != 1:
            raise ValidationError("local sentinel was not exactly one token")
        if prompt_tokens + max_tokens > 16384:
            raise ValidationError("request exceeds max model length")
        duration = _finite(
            event.get("duration_s"), f"event {index}.duration", nonnegative=True
        )
        event_durations_by_task[task_id] += duration
        prompt_tokens_total += prompt_tokens
        completion_tokens_total += completion_tokens
        event_map[key] = event
    if len(event_map) != 144:
        raise ValidationError("request identities are incomplete")
    if completion_tokens_total != 144:
        raise ValidationError("cell completion-token total is not exactly 144")
    for task_id, task in task_map.items():
        prompts = [
            int(event_map[(task_id, call)]["usage"]["prompt_tokens"])
            for call in range(3)
        ]
        if not (
            10000 <= prompts[0] <= 10768
            and 64 <= prompts[1] - prompts[0] <= 768
            and 640 <= prompts[2] - prompts[1] <= 1536
        ):
            raise ValidationError(
                f"{block_id}/{cell_id} prompt topology changed for {task_id}: {prompts}"
            )
        if float(task["e2e_s"]) + 0.01 < event_durations_by_task[task_id]:
            raise ValidationError("task flow omits request duration")

    summary = _mapping(result.get("summary"), "summary")
    exact_summary = {
        "task_count": 48,
        "successful_task_count": 48,
        "failed_task_count": 0,
        "all_tasks_succeeded": True,
        "request_count": 144,
        "successful_request_count": 144,
        "failed_request_count": 0,
        "exactly_one_attempt_each": True,
    }
    if any(summary.get(key) != expected for key, expected in exact_summary.items()):
        raise ValidationError(f"{block_id}/{cell_id} summary completion mismatch")
    summary_llm = _mapping(summary.get("llm"), "summary.llm")
    if (
        summary_llm.get("prompt_tokens") != prompt_tokens_total
        or summary_llm.get("completion_tokens") != completion_tokens_total
    ):
        raise ValidationError("summary token totals mismatch")
    task_e2e_values = [float(task["e2e_s"]) for task in task_map.values()]
    request_durations = [float(event["duration_s"]) for event in event_map.values()]
    recomputed = {
        "mean_task_e2e_s": statistics.fmean(task_e2e_values),
        "p95_task_e2e_s": _percentile(task_e2e_values, 0.95),
        "mean_request_s": statistics.fmean(request_durations),
    }
    reported_task = _mapping(summary.get("task_e2e"), "summary.task_e2e")
    for observed, reported, label in (
        (recomputed["mean_task_e2e_s"], reported_task.get("mean_s"), "mean task"),
        (recomputed["p95_task_e2e_s"], reported_task.get("p95_s"), "p95 task"),
        (recomputed["mean_request_s"], summary_llm.get("mean_request_s"), "mean request"),
    ):
        if not math.isclose(
            observed,
            _finite(reported, label, nonnegative=True),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValidationError(f"{block_id}/{cell_id} {label} is inconsistent")

    metrics = _mapping(result.get("vllm_metric_deltas"), "vllm metrics")
    presence = _mapping(result.get("vllm_metric_presence"), "metric presence")
    if set(metrics) != set(REQUIRED_METRICS) or set(presence) != set(REQUIRED_METRICS):
        raise ValidationError("vLLM metric evidence set mismatch")
    parsed_metrics = {
        key: _finite(metrics[key], f"metric {key}", nonnegative=True)
        for key in REQUIRED_METRICS
    }
    raw = _mapping(result.get("raw_evidence"), "raw evidence")
    raw_metrics = _validate_raw_metrics(
        cell_root,
        raw_evidence=raw,
        reported_deltas=parsed_metrics,
        model_name=str(config["model"]),
    )
    for key in REQUIRED_METRICS:
        states = _mapping(presence[key], f"metric presence {key}")
        if states.get("after") is not True:
            raise ValidationError(f"vLLM metric missing after cell: {key}")
    if not math.isclose(
        parsed_metrics["vllm:prompt_tokens_total"],
        prompt_tokens_total,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        parsed_metrics["vllm:generation_tokens_total"],
        completion_tokens_total,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValidationError("server metric token totals do not equal request usage")
    if parsed_metrics["vllm:num_preemptions_total"] != 0:
        raise ValidationError("native prefix cell had a preemption")
    if parsed_metrics["vllm:request_prefill_time_seconds_sum"] <= 0:
        raise ValidationError("cell records no prefill time")
    queries = parsed_metrics["vllm:prefix_cache_queries_total"]
    hits = parsed_metrics["vllm:prefix_cache_hits_total"]
    if prefix_enabled:
        if queries != prompt_tokens_total or hits <= 0 or hits > queries:
            raise ValidationError("P1 native prefix counters are inconsistent")
        hit_ratio = hits / queries
        if hit_ratio < EXACT_THRESHOLDS["min_native_hit_ratio"]:
            raise ValidationError("P1 native prefix opportunity gate failed")
    else:
        if queries != 0 or hits != 0:
            raise ValidationError("P0 produced native prefix queries or hits")
        hit_ratio = 0.0

    queue_path = cell_root / "evidence/queue_timeline.jsonl"
    queue_rows = _load_queue_rows(queue_path)
    raw_queue = _mapping(raw.get("queue_timeline"), "raw queue evidence")
    if (
        raw_queue.get("sha256") != sha256_file(queue_path)
        or raw_queue.get("sample_count") != len(queue_rows)
    ):
        raise ValidationError("queue evidence binding mismatch")
    valid_queue = [
        row
        for row in queue_rows
        if row.get("ok") is True
        and isinstance(row.get("llm_running"), (int, float))
        and isinstance(row.get("llm_waiting"), (int, float))
    ]
    if not valid_queue:
        raise ValidationError("cell has no valid local vLLM queue sample")
    if any(
        float(row["llm_running"]) >= int(EXACT_ENGINE["VLLM_MAX_NUM_SEQS"])
        for row in valid_queue
    ):
        raise ValidationError("cell reached the configured sequence ceiling")

    source_task_values: dict[str, list[float]] = {}
    for task in task_map.values():
        source_task_values.setdefault(str(task["source_id"]), []).append(
            float(task["e2e_s"])
        )
    if any(len(values) != 3 for values in source_task_values.values()):
        raise ValidationError("source replica accounting mismatch")
    return {
        "block_id": block_id,
        "block_number": block_number,
        "cell_id": cell_id,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "server_process_identity": server_process,
        "prefix_cache_enabled": prefix_enabled,
        "fixture_manifest_sha256": fixture_manifest_sha,
        "event_identity": {
            f"{task_id}/{call_index}": {
                field: event_map[(task_id, call_index)][field]
                for field in (
                    "messages_sha256",
                    "prompt_tokens_estimate",
                    "prompt_token_ids_sha256",
                    "request_payload_sha256",
                    "guided_choice_sha256",
                    "expected_completion_sha256",
                    "response_sha256",
                    "semantic_response_sha256",
                    "expected_completion_tokens_estimate",
                    "usage",
                )
            }
            for task_id, call_index in sorted(event_map)
        },
        "task_e2e_s": {
            task_id: float(task["e2e_s"]) for task_id, task in task_map.items()
        },
        "source_task_e2e_s": {
            source: statistics.fmean(values)
            for source, values in source_task_values.items()
        },
        "request_durations_s": request_durations,
        "metrics": parsed_metrics,
        "native_hit_ratio": hit_ratio,
        "mean_task_e2e_s": recomputed["mean_task_e2e_s"],
        "p95_task_e2e_s": recomputed["p95_task_e2e_s"],
        "mean_request_s": recomputed["mean_request_s"],
        "completion_tokens": completion_tokens_total,
        "prompt_tokens": prompt_tokens_total,
        "server_log": log_evidence,
        "raw_metrics": raw_metrics,
        "max_llm_running": max(float(row["llm_running"]) for row in valid_queue),
        "max_llm_waiting": max(float(row["llm_waiting"]) for row in valid_queue),
    }


def validate_run(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    plan = _validate_plan(run_root)
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    server_ids: set[str] = set()
    server_process_ids: set[str] = set()
    for block_number, order in enumerate(EXPECTED_ORDERS, 1):
        block_root = run_root / f"block-{block_number:02d}"
        for order_index, cell_id in enumerate(order):
            cell = _validate_cell(
                block_root / cell_id,
                plan=plan,
                block_number=block_number,
                cell_id=cell_id,
                order_index=order_index,
            )
            if cell["server_instance_id"] in server_ids:
                raise ValidationError("fresh server instance ID was reused")
            process_identity = cell["server_process_identity"][
                "process_identity_sha256"
            ]
            if process_identity in server_process_ids:
                raise ValidationError("fresh server OS process identity was reused")
            server_ids.add(cell["server_instance_id"])
            server_process_ids.add(process_identity)
            parsed[(block_number, cell_id)] = cell

    reference = parsed[(1, "P0")]
    for key, cell in parsed.items():
        if cell["fixture_manifest_sha256"] != reference["fixture_manifest_sha256"]:
            raise ValidationError("prompt fixture manifest differs across cells")
        if cell["event_identity"] != reference["event_identity"]:
            raise ValidationError(
                f"prompt or exact completion identity differs in block/cell {key}"
            )

    block_effects: list[dict[str, Any]] = []
    for block_number in (1, 2):
        p0 = parsed[(block_number, "P0")]
        p1 = parsed[(block_number, "P1")]
        block_effects.append(
            {
                "block_number": block_number,
                "order": list(EXPECTED_ORDERS[block_number - 1]),
                "prefill_reduction": _relative_reduction(
                    p0["metrics"]["vllm:request_prefill_time_seconds_sum"],
                    p1["metrics"]["vllm:request_prefill_time_seconds_sum"],
                ),
                "mean_request_reduction": _relative_reduction(
                    p0["mean_request_s"], p1["mean_request_s"]
                ),
                "mean_task_e2e_reduction": _relative_reduction(
                    p0["mean_task_e2e_s"], p1["mean_task_e2e_s"]
                ),
                "p0_mean_task_e2e_s": p0["mean_task_e2e_s"],
                "p1_mean_task_e2e_s": p1["mean_task_e2e_s"],
                "p0_prefill_s": p0["metrics"][
                    "vllm:request_prefill_time_seconds_sum"
                ],
                "p1_prefill_s": p1["metrics"][
                    "vllm:request_prefill_time_seconds_sum"
                ],
            }
        )

    cells_by_policy = {
        cell_id: [parsed[(block, cell_id)] for block in (1, 2)]
        for cell_id in ("P0", "P1")
    }
    aggregate: dict[str, dict[str, Any]] = {}
    for cell_id, cells in cells_by_policy.items():
        task_values = [
            value for cell in cells for value in cell["task_e2e_s"].values()
        ]
        request_values = [
            value for cell in cells for value in cell["request_durations_s"]
        ]
        aggregate[cell_id] = {
            "task_count": len(task_values),
            "request_count": len(request_values),
            "mean_task_e2e_s": statistics.fmean(task_values),
            "p95_task_e2e_s": _percentile(task_values, 0.95),
            "mean_request_s": statistics.fmean(request_values),
            "prefill_s": sum(
                cell["metrics"]["vllm:request_prefill_time_seconds_sum"]
                for cell in cells
            ),
            "prompt_tokens": sum(cell["prompt_tokens"] for cell in cells),
            "completion_tokens": sum(cell["completion_tokens"] for cell in cells),
            "native_hits": sum(
                cell["metrics"]["vllm:prefix_cache_hits_total"] for cell in cells
            ),
            "native_queries": sum(
                cell["metrics"]["vllm:prefix_cache_queries_total"] for cell in cells
            ),
        }
        aggregate[cell_id]["native_hit_ratio"] = (
            aggregate[cell_id]["native_hits"]
            / aggregate[cell_id]["native_queries"]
            if aggregate[cell_id]["native_queries"]
            else 0.0
        )

    source_savings: dict[str, float] = {}
    source_ids = sorted(parsed[(1, "P0")]["source_task_e2e_s"])
    for source_id in source_ids:
        p0_value = statistics.fmean(
            parsed[(block, "P0")]["source_task_e2e_s"][source_id]
            for block in (1, 2)
        )
        p1_value = statistics.fmean(
            parsed[(block, "P1")]["source_task_e2e_s"][source_id]
            for block in (1, 2)
        )
        source_savings[source_id] = p0_value - p1_value
    bootstrap = _bootstrap_source_savings(
        source_savings,
        samples=EXACT_THRESHOLDS["bootstrap_samples"],
        seed=EXACT_THRESHOLDS["bootstrap_seed"],
    )
    p0_agg = aggregate["P0"]
    p1_agg = aggregate["P1"]
    effects = {
        "prefill_reduction": _relative_reduction(
            p0_agg["prefill_s"], p1_agg["prefill_s"]
        ),
        "mean_request_reduction": _relative_reduction(
            p0_agg["mean_request_s"], p1_agg["mean_request_s"]
        ),
        "mean_task_e2e_reduction": _relative_reduction(
            p0_agg["mean_task_e2e_s"], p1_agg["mean_task_e2e_s"]
        ),
        "task_p95_ratio": p1_agg["p95_task_e2e_s"] / p0_agg["p95_task_e2e_s"],
        "completion_token_relative_difference": _relative_difference(
            p0_agg["completion_tokens"], p1_agg["completion_tokens"]
        ),
        "paired_source_mean_saving_s": statistics.fmean(source_savings.values()),
        "paired_source_bootstrap_95_ci_s": bootstrap,
        "sources_faster": sum(value > 0 for value in source_savings.values()),
        "source_count": len(source_savings),
    }
    gates = {
        "effective_prefix_flags_and_counters_valid": True,
        "prompt_and_completion_identity_exact": True,
        "native_fcfs_without_joint_patch": True,
        "p1_hit_ratio_at_least_60pct": (
            p1_agg["native_hit_ratio"] >= EXACT_THRESHOLDS["min_native_hit_ratio"]
        ),
        "prefill_reduction_at_least_15pct": (
            effects["prefill_reduction"]
            >= EXACT_THRESHOLDS["min_prefill_reduction"]
        ),
        "prefill_lower_in_both_reverse_blocks": all(
            block["prefill_reduction"] > 0 for block in block_effects
        ),
        "mean_request_reduction_at_least_3pct": (
            effects["mean_request_reduction"]
            >= EXACT_THRESHOLDS["min_mean_request_reduction"]
        ),
        "mean_task_e2e_reduction_at_least_3pct": (
            effects["mean_task_e2e_reduction"]
            >= EXACT_THRESHOLDS["min_mean_task_e2e_reduction"]
        ),
        "task_e2e_lower_in_both_reverse_blocks": all(
            block["mean_task_e2e_reduction"] > 0 for block in block_effects
        ),
        "paired_source_bootstrap_lower_above_zero": bootstrap["lower_s"] > 0,
        "task_p95_within_3pct": (
            effects["task_p95_ratio"]
            <= EXACT_THRESHOLDS["max_task_p95_ratio"]
        ),
        "completion_token_difference_below_1pct": (
            effects["completion_token_relative_difference"]
            < EXACT_THRESHOLDS["max_completion_token_relative_difference"]
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "schema": SCHEMA,
        "version": 2,
        "valid": True,
        "development_only": True,
        "formal_evidence": False,
        "prospective_version": PROTOCOL_VERSION,
        "prior_r1_disposition": "rejected_diagnostic_not_validatable_as_v2",
        "run_root": str(run_root),
        "run_plan_sha256": sha256_file(run_root / "run_plan.json"),
        "contract_bindings": plan["contract_bindings"],
        "run_tag": plan["run_tag"],
        "orders": [list(order) for order in EXPECTED_ORDERS],
        "unique_fresh_server_instances": len(server_ids),
        "unique_fresh_server_processes": len(server_process_ids),
        "matrix": dict(EXACT_MATRIX),
        "thresholds": dict(EXACT_THRESHOLDS),
        "cells": {
            f"block-{block:02d}/{cell}": {
                key: value
                for key, value in parsed[(block, cell)].items()
                if key not in {
                    "event_identity",
                    "task_e2e_s",
                    "source_task_e2e_s",
                    "request_durations_s",
                }
            }
            for block in (1, 2)
            for cell in ("P0", "P1")
        },
        "aggregate": aggregate,
        "block_effects": block_effects,
        "effects_P0_to_P1": effects,
        "promotion_gates": gates,
        "promotion_passed": gates["passed"],
        "selected_policy": "native" if gates["passed"] else "no_latency_claim",
        "stopping_rule": (
            "stop_prefix_exploration_native_cache_causal_gain_confirmed"
            if gates["passed"]
            else "stop_without_native_latency_claim_no_optional_third_block"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Return status 2 when evidence is valid but promotion gates fail.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_run(args.run_root)
    except (OSError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output = args.output or args.run_root.resolve() / "strict_validation.json"
    write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_promotion and not result["promotion_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
