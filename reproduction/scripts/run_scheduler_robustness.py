#!/usr/bin/env python3
"""Audit and replay the production Joint-v2 co-scheduler without a GPU.

This experiment has two deliberately separate evidence tiers:

1. It re-extracts checked-in live-A100 results.  Those numbers are empirical,
   but they all use the pinned Qwen/A100 setup recorded by their source files.
2. It directly invokes ``scripts/pythonhooks/sched_policy_patch.py`` over a
   deterministic factorial set of synthetic scheduler states.  This proves
   implementation algebra, ordering/admission behavior, and parameter
   sensitivity; it is not a replacement for cross-model or cross-GPU runs.

Only Python's standard library is required.  The generated JSON retains every
scenario row and the source SHA256s used to make the report.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_RUNNER_PATH = Path(__file__).resolve()
LIVE_SENSITIVITY_RUNNER_PATH = (
    REPOSITORY_ROOT
    / "reproduction/scripts/run_scheduler_live_sensitivity.py"
)
HOOK_PATH = REPOSITORY_ROOT / "scripts/pythonhooks/sched_policy_patch.py"
LIVE_AGENT_PATH = REPOSITORY_ROOT / "reproduction/paste_repro/live_agent.py"
FORMAL_RUNNER_PATH = (
    REPOSITORY_ROOT / "reproduction/scripts/run_live_joint_formal_matrix.py"
)
FINAL_REPORT_PATH = (
    REPOSITORY_ROOT
    / "reproduction/results/live_joint/PREFIX_AND_LIVE_CLOSED_LOOP_FINAL_REPORT.md"
)
SHAPE_HARNESS_REPAIR_VERSION = "post-shape-r1-formal-order-range-repair-v1"
HISTORICAL_SHAPE_RUNNER_SHA256 = (
    "abcb8c67d2bb72a640663951dcc67e69d53269d1bb284f6579bcd0530299772c"
)
HIGH_REPLACEMENT_RUNNER_SHA256 = (
    "df1286308096455e53de31520db0fd73663f5b0d27cff7c77d92bc62d0e25180"
)
HIGH_R1_ROOT_FILES = {
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-high-r1/run_plan.json": (
        "1a1da1a28ad9fd46dd7e5a957b13c6e57b5f9aab9c21a3525c1cddbf06498298"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-high-r1/completed_matrix.json": (
        "6480f1b4d92c901058c819e0e0325e8940a05127c83ad02474a944e8bb7341ec"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-high-r1/summary.json": (
        "532b396bfc02e2c2e6c0b7ad8a009d51de84f44f4be7475a939b2ec330b11067"
    ),
}
SHAPE_R1_BOUND_FILES = {
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-shape-r1/run_plan.json": (
        "e46917943bb19609ea1531d03a61a6f5d72af7adc7208bdc5c788c798a58a14f"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-shape-r1/failure.json": (
        "22fb3ea0998fcca97dcdaf6a9ba6c2f188b40e00cd97a8fcd9e4c41f89266e7d"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-shape-r1/cells/05-a-c12k-l80/cell_contract.json": (
        "2fae388d8aebcc5b92061c2b9e3eccc4f8a796046d5423896ea5a41323da99e2"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-shape-r1/cells/05-a-c12k-l80/runner.stderr.log": (
        "335682335977d253ce075ce4be93d24cafbbc486d667fa2ece9d5c74e8ec3234"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-shape-r1/cells/05-a-c12k-l80/server/vllm_8000.log": (
        "cd3c19c49b791ef006e3370ad68a936726bb993b19fba427d570f82e4b6b6c5d"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-shape-r1/cells/05-a-c12k-l80/server_lifecycle.stdout.log": (
        "8e565959a584997b4fd5561e5b31ed4ff847868e13845c3fc3c6559ed3eb95c8"
    ),
    "reproduction/artifacts/live_joint/development/comment3_scheduler/"
    "comment3-shape-r1/cells/05-a-c12k-l80/server_lifecycle.stderr.log": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
}


REFERENCE_ENV = {
    "VLLM_SCHED_POLICY": "online_joint_pacer_v2",
    "VLLM_SCHED_DEFAULT_PRED_OUT": "128",
    "VLLM_SCHED_AVG_CALL_SERVICE_S": "2.0",
    "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2": "38112",
    "VLLM_SCHED_DECODE_TOKENS_PER_S_V2": "113.7",
    "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S": "6000",
    "VLLM_SCHED_TIME_AGING_ALPHA": "0.2",
    "VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS": "96",
    "VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING": "48",
    "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S": "40",
    "VLLM_SCHED_JOINT_V2_FINAL_LANE": "1",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE": "1",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES": "0",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S": "0",
    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING": "48",
    "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING": "96",
    "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING": "96",
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "0",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": "0.93",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": "40",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S": "0",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY": "0",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S": "0",
    "VLLM_SCHED_JOINT_V2_TAIL_BETA": "0.25",
    "VLLM_SCHED_JOINT_V2_TOOL_BETA": "0.9",
    "VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S": "80",
    "VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT": "0.35",
    "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA": "1.4",
    "VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS": "8000",
    "VLLM_SCHED_JOINT_V2_FINAL_BONUS_S": "12",
    "VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S": "8",
    "VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S": "4",
    "VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S": "120",
    "VLLM_SCHED_HBM_MIN_RUNNING_REQS": "16",
    "VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP": "16",
    "VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS": "16000",
    "VLLM_SCHED_HBM_MAX_LONG_RUNNING": "16",
    "VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS": "786432",
    "VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS": "524288",
    "VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS": "1048576",
    "VLLM_SCHED_HBM_LOW_PRESSURE": "0.82",
    "VLLM_SCHED_HBM_HIGH_PRESSURE": "1.02",
    "VLLM_SCHED_HBM_BUDGET_INCREASE": "1.02",
    "VLLM_SCHED_HBM_BUDGET_DECREASE": "0.97",
    "VLLM_SCHED_HBM_CONTROL_INTERVAL_S": "5",
    "VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO": "0.96",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY": "0",
}


@dataclass(frozen=True)
class ProxyProfile:
    name: str
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    physical_kv_tokens: int
    native_max_running: int
    decode_target_running: int
    long_context_tokens: int
    evidence_role: str = "counterfactual_proxy_not_measured_hardware"


PROXY_PROFILES = (
    ProxyProfile("small_slow_proxy", 20_000.0, 70.0, 393_216, 48, 48, 8_000),
    ProxyProfile("registered_a100_shape", 38_112.0, 113.7, 786_432, 96, 96, 16_000),
    ProxyProfile("large_fast_proxy", 68_000.0, 190.0, 1_572_864, 192, 192, 32_000),
)


@dataclass
class DummyRequest:
    request_id: str
    arrival_time: float
    num_prompt_tokens: int
    max_tokens: int
    num_tokens: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hook() -> Any:
    spec = importlib.util.spec_from_file_location(
        "paste_scheduler_hook_robustness_audit", HOOK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduler hook: {HOOK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _patched_environment(values: Mapping[str, str]) -> Iterator[None]:
    keys = set(values)
    before = {key: os.environ.get(key) for key in keys}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _request_id(meta: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(meta), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8").hex()
    return f"schedx{encoded}z"


def _candidate_id(hook: Any, request: DummyRequest) -> str:
    return str(hook._decode_meta(request).get("t", request.request_id))


def _scenario_environment(profile: ProxyProfile, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(REFERENCE_ENV)
    environment.update(
        {
            "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2": str(profile.prefill_tokens_per_s),
            "VLLM_SCHED_DECODE_TOKENS_PER_S_V2": str(profile.decode_tokens_per_s),
            "VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS": str(
                profile.native_max_running
            ),
            "VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING": str(
                max(1, profile.decode_target_running // 2)
            ),
            "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING": str(
                max(1, profile.decode_target_running // 2)
            ),
            "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING": str(
                profile.decode_target_running
            ),
            "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING": str(
                profile.decode_target_running
            ),
            "VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS": str(
                profile.physical_kv_tokens
            ),
            "VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS": str(
                max(1, int(profile.physical_kv_tokens * 2 / 3))
            ),
            "VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS": str(
                int(profile.physical_kv_tokens * 4 / 3)
            ),
            "VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS": str(
                profile.long_context_tokens
            ),
            "VLLM_SCHED_HBM_MAX_LONG_RUNNING": str(
                max(1, profile.native_max_running // 6)
            ),
        }
    )
    if overrides:
        environment.update({key: str(value) for key, value in overrides.items()})
    return environment


def _make_waiting(
    workload: str,
    context_scale: float,
    *,
    now_s: float,
) -> list[DummyRequest]:
    prompt_cycle = (2_048, 4_096, 6_144, 8_192, 12_288, 15_360)
    output_cycle = (64, 96, 128, 192, 256, 384)
    remaining_cycle = (0, 1, 2, 2, 1, 0)
    waited_cycle = (0.0, 2.0, 6.0, 12.0, 24.0, 45.0)
    workload_signals = {
        "tool_heavy": ((0.0, 24.0, 48.0, 72.0, 36.0, 0.0), (1.0, 0.9, 0.8, 0.95, 0.75, 1.0)),
        "mixed": ((0.0, 6.0, 18.0, 32.0, 12.0, 0.0), (1.0, 0.65, 0.75, 0.55, 0.8, 1.0)),
        "llm_heavy": ((0.0, 0.8, 2.0, 4.0, 1.0, 0.0), (1.0, 0.25, 0.4, 0.3, 0.2, 1.0)),
        "bursty_mixed": ((0.0, 60.0, 1.0, 45.0, 4.0, 0.0), (1.0, 0.9, 0.25, 0.75, 0.45, 1.0)),
    }
    if workload not in workload_signals:
        raise ValueError(f"unknown workload: {workload}")
    waits, confidences = workload_signals[workload]
    requests: list[DummyRequest] = []
    for index in range(18):
        slot = index % len(prompt_cycle)
        remaining_calls = remaining_cycle[slot]
        prompt_tokens = max(256, int(round(prompt_cycle[slot] * context_scale)))
        output_tokens = output_cycle[(index * 5 + slot) % len(output_cycle)]
        next_wait = waits[slot] if remaining_calls > 0 else 0.0
        confidence = confidences[slot]
        waited_s = waited_cycle[(index * 2 + slot) % len(waited_cycle)]
        meta = {
            "t": f"{workload}-{index:02d}",
            "c": max(0, 2 - remaining_calls),
            "i": max(0, 2 - remaining_calls),
            "n": 3,
            "rc": remaining_calls,
            "nw": next_wait,
            "nwc": confidence,
            "rtw": next_wait * max(1, remaining_calls),
            "pt": prompt_tokens,
            "mt": output_tokens,
            "po": output_tokens,
            "ms": "deterministic_proxy",
        }
        requests.append(
            DummyRequest(
                request_id=_request_id(meta),
                arrival_time=now_s - waited_s,
                num_prompt_tokens=prompt_tokens,
                max_tokens=output_tokens,
                num_tokens=prompt_tokens,
            )
        )
    return requests


def _make_running(
    profile: ProxyProfile,
    load_ratio: float,
    *,
    now_s: float,
) -> list[DummyRequest]:
    running_count = max(1, int(round(profile.decode_target_running * load_ratio)))
    live_tokens = max(running_count * 128, int(profile.physical_kv_tokens * load_ratio))
    per_request = max(128, live_tokens // running_count)
    requests: list[DummyRequest] = []
    for index in range(running_count):
        active = per_request + (1 if index < live_tokens % running_count else 0)
        output = min(64, max(1, active // 4))
        prompt = max(1, active - output)
        meta = {
            "t": f"running-{index:03d}",
            "c": 1,
            "i": 1,
            "n": 3,
            "rc": 1,
            "nw": 4.0,
            "nwc": 0.5,
            "rtw": 4.0,
            "pt": prompt,
            "mt": output,
            "po": output,
        }
        requests.append(
            DummyRequest(
                request_id=_request_id(meta),
                arrival_time=now_s - float(index % 9),
                num_prompt_tokens=prompt,
                max_tokens=output,
                num_tokens=active,
            )
        )
    return requests


def _score_components(
    hook: Any,
    feature: Mapping[str, Any],
    *,
    live_tokens: float,
    virtual_tokens: float,
    live_long_count: int,
    virtual_long_count: int,
    is_new_session: bool,
) -> dict[str, float | bool]:
    """Expand the exact production Joint-v2 score into named terms."""

    meta = feature["meta"]
    prompt_tokens = max(0, int(feature["prompt_tokens"]))
    kv_tokens = max(0, int(feature["kv_tokens"]))
    cached_tokens = max(0, int(feature.get("cached_tokens", 0)))
    raw_marginal_kv_tokens = max(
        0, int(feature.get("marginal_kv_tokens", kv_tokens))
    )
    prefix_weight = hook._joint_v2_prefix_locality_weight()
    prefix_discount_tokens = min(
        float(prompt_tokens), float(cached_tokens) * prefix_weight
    )
    cached_kv_tokens = max(0, kv_tokens - raw_marginal_kv_tokens)
    marginal_kv_tokens = max(
        0.0,
        float(kv_tokens)
        - min(float(kv_tokens), float(cached_kv_tokens) * prefix_weight),
    )
    marginal_prefill_tokens = max(
        0.0, float(prompt_tokens) - prefix_discount_tokens
    )
    remaining_calls = hook._meta_int(meta, "rc", 10**9)
    soft_weight = hook._joint_v2_remaining_call_soft_weight_s()
    soft_remaining_calls = (
        hook._joint_v2_soft_remaining_calls(meta)
        if soft_weight > 0.0
        else remaining_calls
    )
    next_tool_wait = max(0.0, float(feature["next_tool_wait"]))
    remaining_tool_wait = max(
        0.0,
        hook._meta_float(
            meta, "rtw", next_tool_wait * max(0, soft_remaining_calls)
        ),
    )
    prompt_len = int(feature["prompt_len"])
    max_tokens = int(feature["max_tokens"])
    target_tokens = hook._hbm_target_context_tokens()
    fill_target = target_tokens * hook._hbm_virtual_fill_ratio()
    projected_tokens = live_tokens + virtual_tokens + float(kv_tokens)
    projected_pressure = max(0.0, projected_tokens / max(1.0, target_tokens))
    marginal_projected_tokens = (
        live_tokens + virtual_tokens + float(marginal_kv_tokens)
    )
    marginal_pressure = max(
        0.0, marginal_projected_tokens / max(1.0, target_tokens)
    )
    service_s = max(
        0.0,
        hook._service_estimate_v2_s(meta, prompt_len, max_tokens)
        - prefix_discount_tokens / hook._prefill_tokens_per_s_v2(),
    )
    prompt_cost_s = marginal_prefill_tokens / hook._oas_v3_context_tokens_per_s()
    context_penalty_s = (
        hook._joint_v2_context_alpha()
        * (marginal_pressure**1.35)
        * prompt_cost_s
    )
    capped_remaining_tool_s = min(
        remaining_tool_wait,
        hook._joint_v2_tool_wait_cap_s() * max(1, soft_remaining_calls),
    )
    legacy_remaining_service_s = (
        max(0, remaining_calls) * hook._avg_call_service_s()
        if soft_weight <= 0.0
        else 0.0
    )
    task_tail_s = (
        service_s
        + legacy_remaining_service_s
        + hook._joint_v2_remaining_tool_weight() * capped_remaining_tool_s
    )
    remaining_call_soft_cost_s = (
        soft_weight * soft_remaining_calls if soft_weight > 0.0 else 0.0
    )
    context_ref = hook._joint_v2_context_ref_tokens()
    context_damp = 1.0 + prompt_tokens / context_ref
    final_bonus_s = (
        hook._joint_v2_final_bonus_s() / context_damp
        if soft_remaining_calls == 0
        else 0.0
    )
    progress_bonus_s = (
        hook._joint_v2_progress_bonus_s()
        / float(max(1, remaining_calls + 1))
        / context_damp
        if soft_weight <= 0.0
        else 0.0
    )
    capped_next_tool_s = (
        min(next_tool_wait, hook._joint_v2_tool_wait_cap_s())
        if soft_remaining_calls > 0
        else 0.0
    )
    tool_damp = 1.0 + projected_pressure * prompt_tokens / context_ref
    exposed_tool_gain_s = (
        hook._joint_v2_tool_beta()
        * hook._next_tool_wait_reliability(meta)
        * capped_next_tool_s
        / tool_damp
    )
    is_long = prompt_tokens >= hook._hbm_long_context_tokens()
    long_over_cap = (
        is_long
        and hook._hbm_max_long_running() > 0
        and live_long_count + virtual_long_count >= hook._hbm_max_long_running()
    )
    token_over_budget = projected_tokens > fill_target
    over_budget = token_over_budget or long_over_cap
    over_penalty_s = 0.0
    if token_over_budget:
        over_ratio = max(0.0, projected_tokens / max(1.0, fill_target) - 1.0)
        over_penalty_s += hook._joint_v2_over_budget_penalty_s() * (
            1.0 + over_ratio
        ) ** 2
    if long_over_cap:
        over_penalty_s += hook._joint_v2_over_budget_penalty_s()
    new_session_penalty_s = (
        hook._joint_v2_new_session_penalty_s() if is_new_session else 0.0
    )
    task_tail_cost_s = hook._joint_v2_tail_beta() * task_tail_s
    aging_gain_s = hook._time_aging_alpha() * float(feature["waited_s"])
    llm_pressure_s = (
        service_s
        + context_penalty_s
        + task_tail_cost_s
        + remaining_call_soft_cost_s
        + over_penalty_s
        + new_session_penalty_s
    )
    task_progress_gain_s = final_bonus_s + progress_bonus_s
    score_s = (
        llm_pressure_s
        - exposed_tool_gain_s
        - task_progress_gain_s
        - aging_gain_s
    )
    production_score_s, production_over_budget = hook._joint_v2_score_s(
        dict(feature),
        live_tokens=live_tokens,
        virtual_tokens=virtual_tokens,
        live_long_count=live_long_count,
        virtual_long_count=virtual_long_count,
        is_new_session=is_new_session,
    )
    return {
        "service_s": service_s,
        "context_penalty_s": context_penalty_s,
        "task_tail_cost_s": task_tail_cost_s,
        "remaining_call_soft_cost_s": remaining_call_soft_cost_s,
        "over_budget_penalty_s": over_penalty_s,
        "new_session_penalty_s": new_session_penalty_s,
        "llm_pressure_surrogate_s": llm_pressure_s,
        "exposed_tool_gain_surrogate_s": exposed_tool_gain_s,
        "task_progress_gain_s": task_progress_gain_s,
        "aging_gain_s": aging_gain_s,
        "score_s": score_s,
        "production_score_s": production_score_s,
        "absolute_formula_error_s": abs(score_s - production_score_s),
        "over_budget": over_budget,
        "production_over_budget": production_over_budget,
        "projected_pressure": projected_pressure,
        "marginal_projected_pressure": marginal_pressure,
    }


def _pairwise_agreement(reference: Sequence[str], observed: Sequence[str]) -> float:
    if set(reference) != set(observed):
        raise ValueError("orders contain different candidate identities")
    positions = {item: index for index, item in enumerate(observed)}
    agreements = 0
    pairs = 0
    for left_index, left in enumerate(reference):
        for right in reference[left_index + 1 :]:
            pairs += 1
            agreements += int(positions[left] < positions[right])
    return agreements / pairs if pairs else 1.0


def _top_k_overlap(reference: Sequence[str], observed: Sequence[str], k: int = 5) -> float:
    count = min(k, len(reference), len(observed))
    if count <= 0:
        return 1.0
    return len(set(reference[:count]) & set(observed[:count])) / count


def _reset_hook_state(hook: Any) -> None:
    hook._META_CACHE.clear()
    hook._pending_returns.clear()
    hook._v2_started_sessions.clear()
    hook._v2_completed_sessions.clear()


def _evaluate_state(
    hook: Any,
    profile: ProxyProfile,
    workload: str,
    context_scale: float,
    load_ratio: float,
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    now_s = 10_000.0
    environment = _scenario_environment(profile, overrides)
    with _patched_environment(environment):
        _reset_hook_state(hook)
        waiting = _make_waiting(workload, context_scale, now_s=now_s)
        running = _make_running(profile, load_ratio, now_s=now_s)
        live_tokens = float(sum(item.num_tokens for item in running))
        long_threshold = hook._hbm_long_context_tokens()
        live_long_count = sum(
            1 for item in running if item.num_tokens >= long_threshold
        )
        features = {
            id(item): hook._hbm_feature(
                item, now_s, hook._prompt_len_v1, joint_pacer=True
            )
            for item in waiting
        }
        components: dict[str, dict[str, Any]] = {}
        for item in waiting:
            feature = features[id(item)]
            components[_candidate_id(hook, item)] = _score_components(
                hook,
                feature,
                live_tokens=live_tokens,
                virtual_tokens=0.0,
                live_long_count=live_long_count,
                virtual_long_count=0,
                is_new_session=hook._joint_v2_is_new_session(feature["meta"]),
            )
        continuous_order = sorted(
            components,
            key=lambda candidate: (
                components[candidate]["production_score_s"], candidate
            ),
        )
        ordered, logical_admissible_count, logical_budget = (
            hook._order_joint_pacer_v2_waiting(
                waiting_items=waiting,
                running_items=running,
                now_s=now_s,
                prompt_len_fn=hook._prompt_len_v1,
            )
        )
        production_order = [_candidate_id(hook, item) for item in ordered]
        block_size = 16
        num_blocks = max(1, profile.physical_kv_tokens // block_size)
        physical_usage = min(1.0, live_tokens / (num_blocks * block_size))
        scheduler = SimpleNamespace(
            max_num_running_reqs=profile.native_max_running,
            cache_config=SimpleNamespace(
                num_gpu_blocks=num_blocks,
                block_size=block_size,
            ),
            kv_cache_manager=SimpleNamespace(usage=physical_usage),
        )
        physical_order = list(ordered)
        # The production helper logs whenever the synthetic capacity changes.
        # Preserve its behavior and decision while keeping this offline replay's
        # machine-readable stdout/stderr deterministic and compact.
        with redirect_stderr(io.StringIO()):
            admission = hook._apply_joint_v2_physical_kv_admission(
                scheduler,
                ordered=physical_order,
                running_items=running,
                prompt_len_fn=hook._prompt_len_v1,
                reserved_kv=0.0,
                now_s=now_s,
            )
        component_means = {
            key: statistics.fmean(float(row[key]) for row in components.values())
            for key in (
                "llm_pressure_surrogate_s",
                "exposed_tool_gain_surrogate_s",
                "task_progress_gain_s",
                "aging_gain_s",
                "projected_pressure",
            )
        }
        return {
            "profile": profile.name,
            "profile_evidence_role": profile.evidence_role,
            "workload": workload,
            "context_scale": context_scale,
            "load_ratio": load_ratio,
            "candidate_count": len(waiting),
            "running_count": len(running),
            "live_tokens": int(live_tokens),
            "physical_kv_tokens": num_blocks * block_size,
            "decode_load_count_ratio": len(running)
            / max(1, profile.decode_target_running),
            "physical_kv_load_ratio": physical_usage,
            "production_order": production_order,
            "continuous_score_order": continuous_order,
            "production_top5": production_order[:5],
            "continuous_top5": continuous_order[:5],
            "logical_admissible_count": logical_admissible_count,
            "logical_budget_tokens": int(logical_budget),
            "physical_admission": admission,
            "component_means": component_means,
            "max_formula_error_s": max(
                float(row["absolute_formula_error_s"])
                for row in components.values()
            ),
            "component_rows": components,
        }


def _factorial_sweep(hook: Any) -> list[dict[str, Any]]:
    raw: dict[tuple[str, float, float, str], dict[str, Any]] = {}
    for workload in ("tool_heavy", "mixed", "llm_heavy", "bursty_mixed"):
        for context_scale in (0.5, 1.0, 2.0):
            for load_ratio in (0.35, 0.70, 0.90):
                for profile in PROXY_PROFILES:
                    state = _evaluate_state(
                        hook,
                        profile,
                        workload,
                        context_scale,
                        load_ratio,
                    )
                    raw[(workload, context_scale, load_ratio, profile.name)] = state

    rows: list[dict[str, Any]] = []
    reference_name = "registered_a100_shape"
    for key, state in raw.items():
        workload, context_scale, load_ratio, profile_name = key
        reference = raw[(workload, context_scale, load_ratio, reference_name)]
        row = dict(state)
        row.update(
            {
                "production_pairwise_agreement_vs_registered": _pairwise_agreement(
                    reference["production_order"], state["production_order"]
                ),
                "continuous_pairwise_agreement_vs_registered": _pairwise_agreement(
                    reference["continuous_score_order"],
                    state["continuous_score_order"],
                ),
                "production_top5_overlap_vs_registered": _top_k_overlap(
                    reference["production_order"], state["production_order"]
                ),
                "continuous_top5_overlap_vs_registered": _top_k_overlap(
                    reference["continuous_score_order"],
                    state["continuous_score_order"],
                ),
                "reference_profile": reference_name,
            }
        )
        rows.append(row)
    return rows


def _parameter_sweep(hook: Any) -> list[dict[str, Any]]:
    profile = next(
        item for item in PROXY_PROFILES if item.name == "registered_a100_shape"
    )
    base = _evaluate_state(hook, profile, "mixed", 1.0, 0.70)
    definitions = {
        "context_alpha_gamma_analogue": (
            "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA",
            ("0", "0.7", "1.4", "2.8"),
        ),
        "aging_alpha": (
            "VLLM_SCHED_TIME_AGING_ALPHA",
            ("0", "0.1", "0.2", "0.4"),
        ),
        "tool_gain_beta": (
            "VLLM_SCHED_JOINT_V2_TOOL_BETA",
            ("0", "0.45", "0.9", "1.8"),
        ),
        "physical_kv_target_utilization": (
            "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
            ("0.85", "0.90", "0.93", "0.97"),
        ),
        # These are intentionally included to demonstrate that the registered
        # physical-KV branch bypasses the legacy low/high HBM controller.
        "legacy_pressure_band": (
            "VLLM_SCHED_HBM_LOW_PRESSURE,VLLM_SCHED_HBM_HIGH_PRESSURE",
            ("0.60,0.80", "0.82,1.02", "0.95,1.20"),
        ),
    }
    rows: list[dict[str, Any]] = []
    for parameter, (environment_key, values) in definitions.items():
        for value in values:
            if "," in environment_key:
                keys = environment_key.split(",")
                parts = value.split(",")
                overrides = dict(zip(keys, parts))
            else:
                overrides = {environment_key: value}
            state = _evaluate_state(
                hook,
                profile,
                "mixed",
                1.0,
                0.70,
                overrides=overrides,
            )
            rows.append(
                {
                    "parameter": parameter,
                    "environment_key": environment_key,
                    "value": value,
                    "production_pairwise_agreement_vs_registered": _pairwise_agreement(
                        base["production_order"], state["production_order"]
                    ),
                    "continuous_pairwise_agreement_vs_registered": _pairwise_agreement(
                        base["continuous_score_order"],
                        state["continuous_score_order"],
                    ),
                    "production_top5_overlap_vs_registered": _top_k_overlap(
                        base["production_order"], state["production_order"]
                    ),
                    "continuous_top5_overlap_vs_registered": _top_k_overlap(
                        base["continuous_score_order"],
                        state["continuous_score_order"],
                    ),
                    "physical_admit_count": int(
                        state["physical_admission"].get("admit", 0)
                    ),
                    "physical_effective_cap": int(
                        state["physical_admission"].get("effective_cap", 0)
                    ),
                    "max_formula_error_s": state["max_formula_error_s"],
                }
            )
    return rows


def _json_pair_point(
    relative_path: str,
    *,
    label: str,
    load_instances: int,
    controller: str,
    evidence_role: str,
) -> dict[str, Any]:
    path = REPOSITORY_ROOT / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    relative = payload["aggregate"]["effects"]["relative_reduction"]
    return {
        "label": label,
        "load_instances": load_instances,
        "controller": controller,
        "mean_task_reduction_pct": 100.0
        * float(relative["task_flow_time_s"]["mean"]),
        "task_p95_reduction_pct": 100.0
        * float(relative["task_flow_time_s"]["p95"]),
        "source": relative_path,
        "source_sha256": _sha256(path),
        "evidence_role": evidence_role,
    }


def _extract_markdown_percent(path: Path, pattern: str) -> float:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"cannot extract empirical value from {path}: {pattern}")
    return float(match.group(1).replace("−", "-").replace(",", ""))


def _empirical_evidence() -> list[dict[str, Any]]:
    points = [
        _json_pair_point(
            "reproduction/results/joint/paired_heldout60_cap512_m64_t32_r1.json",
            label="heldout60 target32",
            load_instances=60,
            controller="legacy_count_target32",
            evidence_role="one_pair_load_sensitivity",
        ),
        _json_pair_point(
            "reproduction/results/joint/paired_heldout60_cap512_m64_t56_pair_r1.json",
            label="heldout60 target56",
            load_instances=60,
            controller="legacy_count_target56",
            evidence_role="one_pair_load_sensitivity",
        ),
        _json_pair_point(
            "reproduction/results/joint/paired_stress120_cap512_m64_t56.json",
            label="stress120 target56",
            load_instances=120,
            controller="legacy_count_target56",
            evidence_role="two_replicate_development",
        ),
        _json_pair_point(
            "reproduction/results/joint/paired_stress120_cap512_m64_t64.json",
            label="stress120 target64",
            load_instances=120,
            controller="legacy_count_target64",
            evidence_role="three_replicate_load_sensitivity",
        ),
    ]
    stress180_path = (
        REPOSITORY_ROOT / "reproduction/results/joint/summary_stress180_u86_stage_3x.json"
    )
    stress180 = json.loads(stress180_path.read_text(encoding="utf-8"))
    points.append(
        {
            "label": "stress180 target64 stage-aware",
            "load_instances": 180,
            "controller": "legacy_count_target64_stage_lane",
            "mean_task_reduction_pct": 100.0
            * float(stress180["aggregate"]["relative_reduction"]["task_flow_time"]["mean"]),
            "task_p95_reduction_pct": 100.0
            * float(stress180["aggregate"]["relative_reduction"]["task_flow_time"]["p95"]),
            "source": str(stress180_path.relative_to(REPOSITORY_ROOT)),
            "source_sha256": _sha256(stress180_path),
            "evidence_role": "three_replicate_development_load_sensitivity",
        }
    )
    for load in (240, 300):
        path = (
            REPOSITORY_ROOT
            / f"reproduction/results/joint/PHYSICAL_KV_STRESS{load}_REPORT.md"
        )
        text = path.read_text(encoding="utf-8")
        row = next(
            line for line in text.splitlines() if line.startswith("| Mean task E2E |")
        )
        percentages = [
            float(token.replace("−", "-").replace("-", "-").replace("%", ""))
            for token in re.findall(r"\*\*([−-]?[0-9.]+%)\*\*", row)
        ]
        if load == 240:
            full_reduction, _incremental = percentages
        else:
            _a_to_b, _b_to_c, full_reduction = percentages
        p95_row = next(
            line for line in text.splitlines() if line.startswith("| P95 task E2E |")
        )
        p95_percentages = [
            float(token.replace("−", "-").replace("%", ""))
            for token in re.findall(r"\*\*([−-]?[0-9.]+%)\*\*", p95_row)
        ]
        p95_reduction = p95_percentages[0] if load == 240 else p95_percentages[2]
        points.append(
            {
                "label": f"stress{load} physical-KV",
                "load_instances": load,
                "controller": "physical_kv_target_0.93",
                "mean_task_reduction_pct": full_reduction,
                "task_p95_reduction_pct": p95_reduction,
                "source": str(path.relative_to(REPOSITORY_ROOT)),
                "source_sha256": _sha256(path),
                "evidence_role": "single_screen_load_sensitivity",
            }
        )
    return points


def _external_live_aggregate(path: Path) -> dict[str, Any]:
    """Extract a compact, SHA-bound view of a strict four-cell aggregate.

    The upstream aggregator performs the raw-evidence validation.  This loader
    deliberately accepts only its stable schema and never treats the supplied
    file as a GPU run performed by this robustness script.
    """

    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"live aggregate does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"live aggregate is not valid JSON: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"live aggregate is not an object: {resolved}")
    if (
        payload.get("schema") != "paste_repro.live_joint_four_cell_formal"
        or payload.get("schema_version") != 1
    ):
        raise ValueError(f"unsupported live aggregate schema: {resolved}")
    design = payload.get("design")
    effects = payload.get("effects")
    cells = payload.get("aggregate_cells")
    if not all(isinstance(item, Mapping) for item in (design, effects, cells)):
        raise ValueError(f"live aggregate lacks design/effects/cells: {resolved}")
    assert isinstance(design, Mapping)
    assert isinstance(effects, Mapping)
    assert isinstance(cells, Mapping)
    required_effects = ("A_to_E", "B_to_F", "E_to_F", "A_to_F")
    required_cells = ("A", "B", "E", "F")
    if any(not isinstance(effects.get(name), Mapping) for name in required_effects):
        raise ValueError(f"live aggregate lacks a required effect: {resolved}")
    if any(not isinstance(cells.get(name), Mapping) for name in required_cells):
        raise ValueError(f"live aggregate lacks a required cell: {resolved}")
    formal_workload = design.get("formal_workload")
    formal_load = design.get("formal_load")
    if not isinstance(formal_workload, Mapping) or not isinstance(
        formal_load, Mapping
    ):
        raise ValueError(f"live aggregate lacks workload/load identity: {resolved}")

    effect_summary: dict[str, dict[str, Any]] = {}
    for name in required_effects:
        effect = effects[name]
        assert isinstance(effect, Mapping)
        effect_summary[name] = {
            "baseline_cell": effect.get("baseline_cell"),
            "candidate_cell": effect.get("candidate_cell"),
            "baseline_mean_s": float(effect["baseline_mean_s"]),
            "candidate_mean_s": float(effect["candidate_mean_s"]),
            "aggregate_relative_reduction": float(
                effect["aggregate_relative_reduction"]
            ),
            "faster_source_count": int(effect["faster_source_count"]),
            "every_block_mean_reduction_positive": bool(
                effect.get("every_block_mean_reduction_positive")
            ),
        }

    cell_summary: dict[str, dict[str, float | int]] = {}
    for name in required_cells:
        cell = cells[name]
        assert isinstance(cell, Mapping)
        task_e2e = cell.get("task_e2e_s")
        request = cell.get("llm_request_duration_s")
        if not isinstance(task_e2e, Mapping) or not isinstance(request, Mapping):
            raise ValueError(f"live aggregate cell {name} lacks tails: {resolved}")
        cell_summary[name] = {
            "task_count": int(task_e2e["count"]),
            "task_mean_s": float(task_e2e["mean"]),
            "task_p95_s": float(task_e2e["p95"]),
            "request_p95_s": float(request["p95"]),
            "request_p99_s": float(request["p99"]),
        }

    try:
        display_path = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _sha256(resolved),
        "upstream_schema": payload["schema"],
        "formal_promotion_passed": bool(payload.get("formal_promotion_passed")),
        "formal_profile": design.get("formal_profile"),
        "workload_split_id": formal_workload.get("split_id"),
        "workload_sha256": formal_workload.get("file_sha256"),
        "independent_source_count": int(design["independent_source_count"]),
        "block_count": int(design["block_count"]),
        "load": dict(formal_load),
        "effects": effect_summary,
        "cells": cell_summary,
        "evidence_role": "sha_bound_reanalysis_of_external_gpu_aggregate",
    }


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    """Match the live runner's deterministic linear percentile definition."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty live distribution")
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _observed_live_distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("completed live cell has no successful observations")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _linear_percentile(values, 0.50),
        "p95": _linear_percentile(values, 0.95),
        "p99": _linear_percentile(values, 0.99),
        "max": max(values),
    }


def _physical_kv_log_summary(
    server_text: str, *, expected_target: float | None
) -> dict[str, Any]:
    """Validate controller semantics while disclosing raw stdout interleaving."""

    physical_lines = [
        line
        for line in server_text.splitlines()
        if "[sched_policy_patch:physical_kv]" in line
    ]
    if expected_target is None:
        if physical_lines:
            raise ValueError("FCFS live cell unexpectedly emitted physical-KV telemetry")
        return {
            "sample_count": 0,
            "raw_marker_count": 0,
            "target_utilization": None,
            "usage_max": None,
            "fit_admit_min": None,
            "admit_min": None,
            "fit_admit_zero_sample_count": 0,
            "admit_zero_sample_count": 0,
            "target_budget_truncated_waiting_sample_count": 0,
            "semantic_required_admission_field_malformed_count": 0,
            "fail_closed_count": 0,
            "raw_line_interleaving_count": 0,
            "tail_rescue_parse_clean": True,
            "strict_parser_v2_clean": True,
            "controller_was_active": False,
        }
    if not physical_lines:
        raise ValueError("Joint live cell lacks physical-KV telemetry")

    usage: list[float] = []
    fit_admit: list[int] = []
    admit: list[int] = []
    truncated = 0
    malformed = 0
    fail_closed = 0
    wrong_target = 0
    raw_line_interleaving_count = 0
    capacity_write_counts: list[int] = []
    integer_fields = {
        "num_gpu_blocks",
        "block_size",
        "capacity_tokens",
        "budget_tokens",
        "live_tokens",
        "logical_live_tokens",
        "running_growth_tokens",
        "reserved_tokens",
        "committed_tokens",
        "predicted_admit_tokens",
        "waiting",
        "running",
        "fit_admit",
        "admit",
        "effective_cap",
        "native_cap",
        "capacity_write_count",
        "rescue",
    }
    float_fields = {"target_utilization", "usage"}
    required_fields = integer_fields | float_fields | {
        "decision",
        "reason",
        "capacity_write_source",
    }
    for line in physical_lines:
        fields = dict(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)", line))
        if fields.get("decision") == "fail_closed":
            fail_closed += 1
            continue
        raw_rescue = fields.get("rescue")
        if raw_rescue not in {"0", "1"}:
            interleaved = re.fullmatch(
                r"([01])(?:INFO:|WARNING:|ERROR:|\x1b\[[0-9;]*m\(APIServer)",
                str(raw_rescue),
            )
            if interleaved is None:
                malformed += 1
                continue
            fields["rescue"] = interleaved.group(1)
            raw_line_interleaving_count += 1
        if fields.get("decision") != "admit" or not required_fields.issubset(fields):
            malformed += 1
            continue
        try:
            sample: dict[str, Any] = {
                key: int(fields[key]) for key in integer_fields
            }
            sample.update({key: float(fields[key]) for key in float_fields})
            sample.update(
                {
                    "decision": fields["decision"],
                    "reason": fields["reason"],
                    "capacity_write_source": fields["capacity_write_source"],
                }
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            malformed += 1
            continue
        observed_target = float(sample["target_utilization"])
        observed_usage = float(sample["usage"])
        observed_waiting = int(sample["waiting"])
        observed_fit = int(sample["fit_admit"])
        observed_admit = int(sample["admit"])
        if not math.isclose(
            observed_target, expected_target, rel_tol=0.0, abs_tol=1e-9
        ):
            wrong_target += 1
        valid = (
            sample["num_gpu_blocks"] > 0
            and sample["block_size"] > 0
            and sample["capacity_tokens"]
            == sample["num_gpu_blocks"] * sample["block_size"]
            and 0.0 < observed_target <= 1.0
            and 0.0 <= observed_usage <= 1.0
            and 0 <= sample["budget_tokens"] <= sample["capacity_tokens"]
            and 0 <= sample["live_tokens"] <= sample["capacity_tokens"]
            and observed_waiting >= 0
            and sample["running"] >= 0
            and observed_fit >= 0
            and observed_admit >= observed_fit
            and sample["effective_cap"]
            == min(sample["native_cap"], sample["running"] + observed_admit)
            and sample["native_cap"] > 0
            and sample["capacity_write_source"] == "physical_kv"
            and sample["capacity_write_count"] > 0
            and sample["rescue"] in {0, 1}
            and (
                sample["rescue"] == 1
                or (
                    observed_admit == 0
                    and sample["predicted_admit_tokens"] == 0
                )
                or (
                    observed_admit > 0
                    and sample["committed_tokens"]
                    + sample["predicted_admit_tokens"]
                    <= sample["budget_tokens"]
                )
            )
            and (
                sample["rescue"] == 0
                or sample["live_tokens"] + sample["predicted_admit_tokens"]
                <= sample["capacity_tokens"]
            )
        )
        if not valid:
            malformed += 1
            continue
        usage.append(observed_usage)
        fit_admit.append(observed_fit)
        admit.append(observed_admit)
        capacity_write_counts.append(int(sample["capacity_write_count"]))
        if observed_fit < observed_waiting:
            truncated += 1
    if any(
        after <= before
        for before, after in zip(capacity_write_counts, capacity_write_counts[1:])
    ):
        malformed += 1
    if malformed or wrong_target or fail_closed or len(usage) != len(physical_lines):
        raise ValueError(
            "Joint live physical-KV telemetry failed closed: "
            f"samples={len(physical_lines)} malformed={malformed} "
            f"wrong_target={wrong_target} fail_closed={fail_closed}"
        )
    return {
        "sample_count": len(physical_lines),
        "raw_marker_count": len(physical_lines),
        "target_utilization": expected_target,
        "usage_max": max(usage),
        "fit_admit_min": min(fit_admit),
        "admit_min": min(admit),
        "fit_admit_zero_sample_count": sum(value == 0 for value in fit_admit),
        "admit_zero_sample_count": sum(value == 0 for value in admit),
        "target_budget_truncated_waiting_sample_count": truncated,
        "semantic_required_admission_field_malformed_count": 0,
        "fail_closed_count": 0,
        "raw_line_interleaving_count": raw_line_interleaving_count,
        "tail_rescue_parse_clean": raw_line_interleaving_count == 0,
        "strict_parser_v2_clean": raw_line_interleaving_count == 0,
        "controller_was_active": True,
    }


def _failed_r2_transport_provenance(run_root: Path) -> dict[str, Any]:
    """Bind the excluded partial r2 pilot without treating it as a result cell."""

    failed_root = run_root.parent / "comment3-target-r2"
    plan_path = failed_root / "run_plan.json"
    failure_path = failed_root / "failure.json"
    a_result_path = failed_root / "cells/01-a-c10k-l80/evidence/result.json"
    a_server_path = failed_root / "cells/01-a-c10k-l80/server/vllm_8100.log"
    e_result_path = (
        failed_root / "cells/02-e-c10k-l80-u085/evidence/result.json"
    )
    e_server_path = (
        failed_root / "cells/02-e-c10k-l80-u085/server/vllm_8100.log"
    )
    required = (
        plan_path,
        failure_path,
        a_result_path,
        a_server_path,
        e_result_path,
        e_server_path,
    )
    if any(not item.is_file() for item in required):
        missing = [str(item) for item in required if not item.is_file()]
        raise ValueError(f"excluded r2 provenance is incomplete: {missing}")
    try:
        failed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        results = {
            "A": json.loads(a_result_path.read_text(encoding="utf-8")),
            "E_u085": json.loads(e_result_path.read_text(encoding="utf-8")),
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("excluded r2 provenance is not valid JSON") from exc
    if (
        not isinstance(failed_plan, Mapping)
        or failed_plan.get("schema")
        != "paste_repro.scheduler_live_sensitivity_plan"
        or failed_plan.get("run_tag") != "comment3-target-r2"
        or not isinstance(failure, Mapping)
        or failure.get("schema")
        != "paste_repro.scheduler_live_sensitivity_failure"
        or failure.get("error_type") != "LiveSensitivityError"
        or not all(isinstance(item, Mapping) for item in results.values())
    ):
        raise ValueError("excluded r2 plan/failure identity drifted")

    transport_counts: dict[str, dict[str, int]] = {}
    for label, result in results.items():
        records = result.get("tool_attempt_records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"excluded r2 {label} lacks transport records")
        logs = [
            entry
            for record in records
            if isinstance(record, Mapping)
            for entry in (
                record.get("http_attempt_log")
                if isinstance(record.get("http_attempt_log"), list)
                else []
            )
            if isinstance(entry, Mapping)
        ]
        transport_counts[label] = {
            "tool_record_count": len(records),
            "http_attempt_count": len(logs),
            "http_retry_count": sum(
                int(record.get("http_attempts", 0)) - 1
                for record in records
                if isinstance(record, Mapping)
            ),
            "http_429_count": sum(entry.get("status") == 429 for entry in logs),
            "failed_tool_record_count": sum(
                record.get("outcome") == "failed"
                for record in records
                if isinstance(record, Mapping)
            ),
        }
    expected_counts = {
        "A": {
            "tool_record_count": 160,
            "http_attempt_count": 164,
            "http_retry_count": 4,
            "http_429_count": 4,
            "failed_tool_record_count": 0,
        },
        "E_u085": {
            "tool_record_count": 160,
            "http_attempt_count": 165,
            "http_retry_count": 5,
            "http_429_count": 6,
            "failed_tool_record_count": 1,
        },
    }
    if transport_counts != expected_counts:
        raise ValueError("excluded r2 does not reproduce its disclosed 429 failure")
    return {
        "run_tag": "comment3-target-r2",
        "role": "excluded_partial_transport_pilot_not_aggregated",
        "failure_class": "external_jina_http_429",
        "partial_performance_was_observable": True,
        "partial_performance_used_in_effects": False,
        "transport_counts": transport_counts,
        "source_files": {
            str(item.relative_to(REPOSITORY_ROOT)): _sha256(item)
            for item in required
        },
    }


def _clean_live_transport_evidence(tool_attempts: Any) -> dict[str, Any]:
    """Validate the complete retry-free 80-search/80-visit transport ledger."""

    if not isinstance(tool_attempts, list) or len(tool_attempts) != 160:
        raise ValueError("live raw transport ledger is incomplete")
    tool_counts: dict[str, int] = {"search": 0, "visit": 0}
    expected_identity = {
        "search": ("bing_html_search", "www.bing.com"),
        "visit": ("r.jina.ai", "r.jina.ai"),
    }
    visit_starts: list[float] = []
    for index, attempt in enumerate(tool_attempts):
        attempt_log = (
            attempt.get("http_attempt_log")
            if isinstance(attempt, Mapping)
            else None
        )
        entry = attempt_log[0] if isinstance(attempt_log, list) and attempt_log else None
        started = entry.get("started_monotonic_s") if isinstance(entry, Mapping) else None
        tool = attempt.get("tool") if isinstance(attempt, Mapping) else None
        identity = expected_identity.get(str(tool))
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("authoritative") is not True
            or attempt.get("speculative") is not False
            or attempt.get("committed") is not True
            or attempt.get("outcome") != "committed"
            or attempt.get("http_attempts") != 1
            or attempt.get("response_status") != 200
            or attempt.get("transport_identity_source") != "actual"
            or identity is None
            or (attempt.get("backend"), attempt.get("request_host")) != identity
            or not isinstance(attempt_log, list)
            or len(attempt_log) != 1
            or not isinstance(entry, Mapping)
            or entry.get("attempt") != 1
            or entry.get("status") != 200
            or entry.get("retried") is not False
            or isinstance(started, bool)
            or not isinstance(started, (int, float))
            or not math.isfinite(float(started))
        ):
            raise ValueError(f"live raw transport attempt {index} failed")
        tool_name = str(tool)
        tool_counts[tool_name] += 1
        if tool_name == "visit":
            visit_starts.append(float(started))
    if tool_counts != {"search": 80, "visit": 80}:
        raise ValueError("live search/visit identity matrix drifted")
    visit_starts.sort()
    visit_start_gaps = [
        right - left for left, right in zip(visit_starts, visit_starts[1:])
    ]
    if len(visit_starts) != 80 or min(visit_start_gaps) < 2.98:
        raise ValueError("live 3 s visit start gate failed")
    return {
        "tool_record_count": 160,
        "search_record_count": 80,
        "visit_record_count": 80,
        "http_attempt_count": 160,
        "http_retry_count": 0,
        "http_429_count": 0,
        "actual_transport_identity_count": 160,
        "minimum_adjacent_visit_start_gap_s": min(visit_start_gaps),
    }


def _clean_live_broker_evidence(broker: Any) -> dict[str, int]:
    expected = {
        "authoritative_requests": 160,
        "authoritative_started": 160,
        "authoritative_completed": 160,
        "authoritative_failures": 0,
        "commits": 160,
    }
    stats = broker.get("stats") if isinstance(broker, Mapping) else None
    if not isinstance(stats, Mapping) or any(
        stats.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("live broker completion ledger drifted")
    return expected


def _canonical_json_sha256(payload: Any) -> str:
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _normalized_shape_replacement_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the identity fields allowed by the frozen repair plan."""

    command = cell.get("runner_command")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError("shape replacement cell lacks a runner command")
    identity_options = {
        "--output-dir": "<RUN_OUTPUT_DIRECTORY>",
        "--server-url": "<SERVER_IDENTITY_URL>",
        "--cell-label": "<RUN_CELL_LABEL>",
        "--formal-block-id": "<FORMAL_BLOCK_ID>",
        "--formal-order-index": "<FORMAL_ORDER_INDEX>",
        "--server-instance-id": "<SERVER_INSTANCE_ID>",
    }
    normalized_command: list[str] = []
    index = 0
    while index < len(command):
        option = command[index]
        normalized_command.append(option)
        if option in identity_options:
            if index + 1 >= len(command):
                raise ValueError(
                    f"shape replacement command lacks a value for {option}"
                )
            normalized_command.append(identity_options[option])
            index += 2
        else:
            index += 1
    normalized = {
        key: value
        for key, value in cell.items()
        if key not in {"order_index", "runner_command", "server_state_directory"}
    }
    normalized["runner_command"] = normalized_command
    return normalized


def _high_shape_r1_provenance(
    *,
    plan: Mapping[str, Any],
    completion: Mapping[str, Any],
    summary_boundary: Mapping[str, Any],
    matrix: Mapping[str, Any],
    plan_cells: Sequence[Mapping[str, Any]],
    plan_bindings: Mapping[str, Any],
    source_files: dict[str, str],
) -> dict[str, Any]:
    """Fail closed over the excluded shape-r1 failure and high replacement."""

    repair = plan.get("shape_r1_harness_repair")
    order_gate = matrix.get("formal_order_index_gate")
    expected_labels = ["a-c12k-l80", "e-c12k-l80-u093"]
    expected_specs = [
        ("A", 12_000, 80, 0.93, "c12k-l80", "fcfs_reference", 0),
        ("E", 12_000, 80, 0.93, "c12k-l80", "joint_candidate", 1),
    ]
    if (
        plan.get("run_tag") != "comment3-high-r1"
        or plan.get("suite") != "high"
        or plan.get("cell_count") != 2
        or completion.get("shape_harness_repair_version")
        != SHAPE_HARNESS_REPAIR_VERSION
        or not isinstance(repair, Mapping)
        or not isinstance(order_gate, Mapping)
        or order_gate.get("underlying_runner_range") != [0, 3]
        or order_gate.get("planned_indices") != [0, 1]
        or order_gate.get("all_indices_in_range") is not True
        or order_gate.get("cell_count") != 2
        or order_gate.get("maximum_cell_count") != 4
        or len(plan_cells) != 2
        or [str(cell.get("label")) for cell in plan_cells] != expected_labels
    ):
        raise ValueError("high replacement plan/order identity drifted")
    if any(
        source_files.get(relative) != expected_sha
        for relative, expected_sha in HIGH_R1_ROOT_FILES.items()
    ):
        raise ValueError("high-r1 plan/completion/summary root hash drifted")
    for cell, expected in zip(plan_cells, expected_specs):
        observed = (
            cell.get("cell"),
            cell.get("context_padding_tokens"),
            cell.get("max_active_tasks"),
            cell.get("physical_kv_target"),
            cell.get("pair_group"),
            cell.get("role"),
            cell.get("order_index"),
        )
        if observed != expected or cell.get("fresh_server") is not True:
            raise ValueError("high replacement cell configuration drifted")

    expected_summary_flags = {
        "shape_harness_repair_version": SHAPE_HARNESS_REPAIR_VERSION,
        "failed_shape_r1_resumed": False,
        "failed_shape_r1_observed_prefix_pooled": False,
        "high_pair_one_shot_replacement": True,
        "no_further_auto_rerun": True,
    }
    if any(
        summary_boundary.get(key) != value
        for key, value in expected_summary_flags.items()
    ):
        raise ValueError("high replacement summary disclosure drifted")

    bound_files = repair.get("bound_files")
    if bound_files != SHAPE_R1_BOUND_FILES:
        raise ValueError("shape-r1 provenance binding set drifted")
    for relative, expected_sha in SHAPE_R1_BOUND_FILES.items():
        artifact = (REPOSITORY_ROOT / relative).resolve()
        if (
            plan_bindings.get(relative) != expected_sha
            or not artifact.is_file()
            or _sha256(artifact) != expected_sha
        ):
            raise ValueError(f"shape-r1 provenance SHA256 mismatch: {relative}")
        source_files[relative] = expected_sha

    runner_path = str(LIVE_SENSITIVITY_RUNNER_PATH.relative_to(REPOSITORY_ROOT))
    runner_bindings = repair.get("runner_bindings")
    if (
        not isinstance(runner_bindings, Mapping)
        or runner_bindings.get("path") != runner_path
        or runner_bindings.get("historical_sha256")
        != HISTORICAL_SHAPE_RUNNER_SHA256
        or runner_bindings.get("replacement_sha256")
        != HIGH_REPLACEMENT_RUNNER_SHA256
        or runner_bindings.get("historical_artifact_requires_historical_sha256")
        is not True
        or runner_bindings.get("replacement_runner_sha_differs") is not True
        or plan_bindings.get(runner_path) != HIGH_REPLACEMENT_RUNNER_SHA256
        or _sha256(LIVE_SENSITIVITY_RUNNER_PATH)
        != HIGH_REPLACEMENT_RUNNER_SHA256
    ):
        raise ValueError("shape-r1 historical/replacement runner binding drifted")

    shape_root = (
        REPOSITORY_ROOT
        / "reproduction/artifacts/live_joint/development/comment3_scheduler/"
        "comment3-shape-r1"
    )
    failed_cell_root = shape_root / "cells/05-a-c12k-l80"
    old_plan_path = shape_root / "run_plan.json"
    failure_path = shape_root / "failure.json"
    failed_contract_path = failed_cell_root / "cell_contract.json"
    stderr_path = failed_cell_root / "runner.stderr.log"
    server_path = failed_cell_root / "server/vllm_8000.log"
    lifecycle_stdout_path = failed_cell_root / "server_lifecycle.stdout.log"
    lifecycle_stderr_path = failed_cell_root / "server_lifecycle.stderr.log"
    try:
        old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        failed_contract = json.loads(
            failed_contract_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("shape-r1 provenance is not valid JSON") from exc
    if not all(
        isinstance(item, Mapping) for item in (old_plan, failure, failed_contract)
    ):
        raise ValueError("shape-r1 provenance JSON is not an object")
    old_cells = old_plan.get("cells")
    old_bindings = old_plan.get("bindings")
    if (
        old_plan.get("schema")
        != "paste_repro.scheduler_live_sensitivity_plan"
        or old_plan.get("version") != 1
        or old_plan.get("run_tag") != "comment3-shape-r1"
        or old_plan.get("suite") != "shape"
        or old_plan.get("cell_count") != 6
        or not isinstance(old_cells, list)
        or len(old_cells) != 6
        or not isinstance(old_bindings, Mapping)
        or old_bindings.get(runner_path) != HISTORICAL_SHAPE_RUNNER_SHA256
        or failure.get("schema")
        != "paste_repro.scheduler_live_sensitivity_failure"
        or failure.get("version") != 1
        or failure.get("error_type") != "LiveSensitivityError"
        or failure.get("error") != "a-c12k-l80 live runner failed"
    ):
        raise ValueError("shape-r1 historical plan/failure identity drifted")
    failed_spec = failed_contract.get("spec")
    if (
        failed_contract.get("schema")
        != "paste_repro.scheduler_live_sensitivity_cell_contract"
        or failed_contract.get("version") != 1
        or failed_contract.get("order_index") != 4
        or failed_contract.get("bindings") != old_bindings
        or not isinstance(failed_spec, Mapping)
        or failed_spec.get("label") != "a-c12k-l80"
        or failed_spec.get("cell") != "A"
        or failed_spec.get("context_padding_tokens") != 12_000
        or failed_spec.get("max_active_tasks") != 80
    ):
        raise ValueError("shape-r1 failed cell contract drifted")

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    server_text = server_path.read_text(encoding="utf-8", errors="replace")
    lifecycle_stdout = lifecycle_stdout_path.read_text(
        encoding="utf-8", errors="replace"
    )
    failed_stdout_path = failed_cell_root / "runner.stdout.log"
    actual_absence = {
        "chat_completion_post_count": server_text.count(
            "POST /v1/chat/completions"
        ),
        "result_absent": not (failed_cell_root / "evidence/result.json").exists(),
        "queue_timeline_absent": not (
            failed_cell_root / "evidence/queue_timeline.jsonl"
        ).exists(),
        "cell_manifest_absent": not (failed_cell_root / "cell_manifest.json").exists(),
        "runner_stdout_empty": (
            failed_stdout_path.is_file() and failed_stdout_path.stat().st_size == 0
        ),
        "server_stopped_cleanly": re.search(
            r"vLLM pid [0-9]+ stopped cleanly\.", lifecycle_stdout
        )
        is not None,
        "lifecycle_stderr_empty": lifecycle_stderr_path.stat().st_size == 0,
    }
    if (
        repair.get("version") != SHAPE_HARNESS_REPAIR_VERSION
        or repair.get("failed_run_tag") != "comment3-shape-r1"
        or repair.get("failure_class")
        != "deterministic_formal_order_index_harness_failure"
        or repair.get("rejected_order_index") != 4
        or repair.get("failed_cell") != "a-c12k-l80"
        or repair.get("failed_cell_request_count") != 0
        or repair.get("failed_cell_result_present") is not False
        or repair.get("failed_cell_absence_checks") != actual_absence
        or "ValueError: --formal-order-index must be in [0, 3]"
        not in stderr_text
        or actual_absence["chat_completion_post_count"] != 0
        or any(
            value is not True
            for key, value in actual_absence.items()
            if key != "chat_completion_post_count"
        )
    ):
        raise ValueError("shape-r1 zero-request deterministic failure drifted")
    failed_stdout_relative = str(failed_stdout_path.relative_to(REPOSITORY_ROOT))
    source_files[failed_stdout_relative] = _sha256(failed_stdout_path)
    additional_absence_paths = [
        shape_root / "summary.json",
        shape_root / "completed_matrix.json",
        shape_root / "cells/06-e-c12k-l80-u093",
    ]
    if any(path.exists() for path in additional_absence_paths):
        raise ValueError("shape-r1 immutable partial-run absence boundary drifted")

    excluded = repair.get("excluded_observed_prefix")
    expected_excluded = [
        "a-c5k-l40",
        "e-c5k-l40-u093",
        "a-c10k-l80",
        "e-c10k-l80-u093",
    ]
    if (
        not isinstance(excluded, Mapping)
        or excluded.get("cell_count") != 4
        or excluded.get("cells") != expected_excluded
        or excluded.get("reused_by_replacement") is not False
        or excluded.get("pooled_with_replacement") is not False
        or set(expected_excluded) & set(expected_labels)
    ):
        raise ValueError("shape-r1 observed prefix was not strictly excluded")

    replacement = repair.get("replacement")
    expected_allowed_differences = [
        "run/output path",
        "formal block id",
        "formal order index",
        "server instance id",
        "server URL/port",
    ]
    if (
        not isinstance(replacement, Mapping)
        or replacement.get("suite") != "high"
        or replacement.get("cell_count") != 2
        or replacement.get("fixed_order") != expected_labels
        or replacement.get("first_four_cells_rerun") is not False
        or replacement.get("failed_shape_run_resumed") is not False
        or replacement.get("historical_shape_artifact_requires_original_bound_runner_sha")
        is not True
        or replacement.get("new_runner_may_not_resume_historical_shape_artifact")
        is not True
        or replacement.get("selection_or_tuning_from_observed_prefix") is not False
        or replacement.get("allowed_identity_differences")
        != expected_allowed_differences
        or replacement.get("one_shot_replacement") is not True
        or replacement.get("no_further_auto_rerun") is not True
    ):
        raise ValueError("shape-r1 replacement disclosure drifted")

    old_high = {
        str(cell.get("label")): cell
        for cell in old_cells[4:]
        if isinstance(cell, Mapping)
    }
    new_high = {str(cell.get("label")): cell for cell in plan_cells}
    recorded_equivalence = replacement.get("configuration_equivalence")
    if (
        set(old_high) != set(expected_labels)
        or set(new_high) != set(expected_labels)
        or not isinstance(recorded_equivalence, list)
        or len(recorded_equivalence) != 2
    ):
        raise ValueError("shape-r1 high-pair equivalence matrix drifted")
    verified_equivalence: list[dict[str, Any]] = []
    for label, recorded in zip(expected_labels, recorded_equivalence):
        if not isinstance(recorded, Mapping) or recorded.get("cell") != label:
            raise ValueError("shape-r1 equivalence row identity drifted")
        old_normalized = _normalized_shape_replacement_cell(old_high[label])
        new_normalized = _normalized_shape_replacement_cell(new_high[label])
        old_sha = _canonical_json_sha256(old_normalized)
        new_sha = _canonical_json_sha256(new_normalized)
        if (
            old_normalized != new_normalized
            or old_sha != new_sha
            or recorded.get("old_normalized_sha256") != old_sha
            or recorded.get("replacement_normalized_sha256") != new_sha
            or recorded.get("equal_after_identity_normalization") is not True
        ):
            raise ValueError("shape-r1 high-pair non-identity config drifted")
        verified_equivalence.append(
            {
                "cell": label,
                "normalized_sha256": old_sha,
                "equal_after_identity_normalization": True,
            }
        )

    return {
        "version": SHAPE_HARNESS_REPAIR_VERSION,
        "failed_run_tag": "comment3-shape-r1",
        "failure_class": "deterministic_formal_order_index_harness_failure",
        "failed_cell": "a-c12k-l80",
        "rejected_order_index": 4,
        "failed_cell_request_count": 0,
        "failed_cell_absence_checks": actual_absence,
        "aggregator_additional_absence_checks": {
            str(path.relative_to(REPOSITORY_ROOT)): True
            for path in additional_absence_paths
        },
        "aggregator_additional_source_files": {
            failed_stdout_relative: source_files[failed_stdout_relative]
        },
        "bound_files": dict(sorted(SHAPE_R1_BOUND_FILES.items())),
        "runner_bindings": dict(runner_bindings),
        "excluded_observed_prefix": {
            "cell_count": 4,
            "cells": expected_excluded,
            "reused_by_replacement": False,
            "pooled_with_replacement": False,
            "performance_loaded_or_reported": False,
        },
        "replacement": {
            "run_tag": "comment3-high-r1",
            "fixed_order": expected_labels,
            "one_shot": True,
            "no_further_auto_rerun": True,
            "configuration_equivalence": verified_equivalence,
        },
    }


def _external_live_sensitivity_summary(path: Path) -> dict[str, Any]:
    """Fail closed over the plan -> completion -> manifest -> raw-evidence chain."""

    summary_path = path.resolve()
    run_root = summary_path.parent
    plan_path = run_root / "run_plan.json"
    completion_path = run_root / "completed_matrix.json"
    required = (summary_path, plan_path, completion_path)
    if any(not item.is_file() for item in required):
        missing = [str(item) for item in required if not item.is_file()]
        raise ValueError(f"live sensitivity evidence is incomplete: {missing}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("live sensitivity evidence is not valid JSON") from exc
    if not all(isinstance(item, Mapping) for item in (summary, plan, completion)):
        raise ValueError("live sensitivity evidence must contain JSON objects")
    if (
        summary.get("schema")
        != "paste_repro.scheduler_live_sensitivity_summary"
        or summary.get("version") != 1
        or summary.get("development_only") is not True
        or summary.get("formal_eligible") is not False
        or summary.get("single_run_per_cell") is not True
        or summary.get("confidence_interval_available") is not False
    ):
        raise ValueError("unsupported live sensitivity summary schema")
    if (
        plan.get("schema") != "paste_repro.scheduler_live_sensitivity_plan"
        or plan.get("version") != 1
        or plan.get("development_only") is not True
        or plan.get("formal_eligible") is not False
        or completion.get("schema")
        != "paste_repro.scheduler_live_sensitivity_completion"
        or completion.get("version") != 1
        or completion.get("development_only") is not True
        or completion.get("formal_eligible") is not False
    ):
        raise ValueError("live sensitivity plan/completion schema mismatch")

    relative_run_root = str(run_root.relative_to(REPOSITORY_ROOT))
    if (
        summary.get("run_root") != relative_run_root
        or plan.get("run_root") != relative_run_root
    ):
        raise ValueError("live summary/plan run-root identity mismatch")
    summary_ref = completion.get("summary")
    expected_summary_path = str(summary_path.relative_to(REPOSITORY_ROOT))
    if (
        not isinstance(summary_ref, Mapping)
        or summary_ref.get("path") != expected_summary_path
        or summary_ref.get("sha256") != _sha256(summary_path)
    ):
        raise ValueError("completion does not bind the supplied live summary")
    plan_bindings = plan.get("bindings")
    if (
        not isinstance(plan_bindings, Mapping)
        or not plan_bindings
        or completion.get("bindings") != plan_bindings
    ):
        raise ValueError("completion does not preserve the planned code bindings")

    plan_cells = plan.get("cells")
    completed_cells = completion.get("completed_cells")
    cells = summary.get("cells")
    effects = summary.get("a_to_e_effects")
    targets = summary.get("physical_kv_target_sensitivity")
    if (
        not isinstance(plan_cells, list)
        or not isinstance(completed_cells, list)
        or not isinstance(cells, Mapping)
        or not isinstance(effects, list)
        or not isinstance(targets, list)
        or len(plan_cells) != int(plan.get("cell_count", -1))
        or len(completed_cells) != len(plan_cells)
    ):
        raise ValueError("live sensitivity matrix structure is incomplete")
    plan_by_label = {
        str(row.get("label")): row
        for row in plan_cells
        if isinstance(row, Mapping) and isinstance(row.get("label"), str)
    }
    completed_by_label = {
        str(row.get("label")): row
        for row in completed_cells
        if isinstance(row, Mapping) and isinstance(row.get("label"), str)
    }
    if (
        len(plan_by_label) != len(plan_cells)
        or len(completed_by_label) != len(completed_cells)
        or set(plan_by_label) != set(completed_by_label)
        or set(cells) != set(plan_by_label)
    ):
        raise ValueError("live plan/completion/summary cell identities differ")

    preflight = plan.get("preflight")
    matrix = preflight.get("matrix_invariants") if isinstance(preflight, Mapping) else None
    transport_contract = (
        matrix.get("transport_contract") if isinstance(matrix, Mapping) else None
    )
    remediation = plan.get("transport_remediation_after_failed_pilot")
    summary_boundary = summary.get("evidence_boundary")
    if (
        not isinstance(matrix, Mapping)
        or matrix.get("all_cells_use_same_workload") is not True
        or matrix.get("fresh_server_per_cell") is not True
        or matrix.get("cross_cell_state_reuse") is not False
        or not re.fullmatch(r"[0-9a-f]{64}", str(matrix.get("workload_sha256")))
        or any(
            check.get("only_scheduler_treatment_changes") is not True
            or check.get("common_config_diff") != {}
            for check in matrix.get("pair_checks", [])
            if isinstance(check, Mapping)
        )
        or any(
            check.get("only_active_physical_kv_target_changes") is not True
            for check in matrix.get("target_checks", [])
            if isinstance(check, Mapping)
        )
        or not isinstance(transport_contract, Mapping)
        or transport_contract.get("visit_min_start_interval_s") != 3.0
        or transport_contract.get("accepted_http_attempts_per_tool_invocation") != 1
        or transport_contract.get("zero_retries_required") is not True
        or transport_contract.get("same_for_every_a_e_cell") is not True
        or not isinstance(remediation, Mapping)
        or remediation.get("failed_run_tag") != "comment3-target-r2"
        or remediation.get("all_cells_rebaselined") is not True
        or remediation.get("zero_http_retries_required") is not True
        or remediation.get("not_preregistered") is not True
        or remediation.get("failed_run_performance_was_observable") is not True
        or remediation.get("one_shot_replacement") is not True
        or remediation.get("no_further_auto_rerun_or_transport_escalation")
        is not True
        or remediation.get("no_cross_transport_pooling_or_comparison") is not True
        or not isinstance(summary_boundary, Mapping)
        or summary_boundary.get("transport_remediation_version")
        != transport_contract.get("remediation_version")
        or summary_boundary.get("all_cells_rebaselined_after_failed_r2") is not True
        or summary_boundary.get("zero_http_retries_required") is not True
        or summary_boundary.get("descriptive_only_under_fixed_3s_jina_pacing")
        is not True
        or summary_boundary.get("failed_r2_cells_excluded_without_pooling") is not True
    ):
        raise ValueError("live plan comparison invariants are missing or failed")
    workload_sha = str(matrix["workload_sha256"])
    failed_r2_provenance = _failed_r2_transport_provenance(run_root)

    compact_cells: dict[str, dict[str, Any]] = {}
    source_values: dict[str, dict[str, float]] = {}
    task_identity_values: dict[str, dict[str, tuple[Any, ...]]] = {}
    tool_invocation_values: dict[str, set[tuple[str, str, str]]] = {}
    server_instance_ids: set[str] = set()
    source_files = {
        str(item.relative_to(REPOSITORY_ROOT)): _sha256(item) for item in required
    }
    suite = plan.get("suite")
    if suite not in {"target", "high"}:
        raise ValueError("unsupported live sensitivity suite")
    shape_r1_provenance: dict[str, Any] | None = None
    if suite == "high":
        shape_r1_provenance = _high_shape_r1_provenance(
            plan=plan,
            completion=completion,
            summary_boundary=summary_boundary,
            matrix=matrix,
            plan_cells=plan_cells,
            plan_bindings=plan_bindings,
            source_files=source_files,
        )
    elif (
        plan.get("shape_r1_harness_repair") is not None
        or completion.get("shape_harness_repair_version") is not None
    ):
        raise ValueError("target suite unexpectedly contains shape-r1 repair data")
    spec_keys = (
        "cell",
        "context_padding_tokens",
        "label",
        "max_active_tasks",
        "pair_group",
        "physical_kv_target",
        "role",
    )
    distribution_keys = ("count", "mean", "p50", "p95", "p99", "max")
    for label, cell in cells.items():
        if not isinstance(label, str) or not isinstance(cell, Mapping):
            raise ValueError("live summary contains an invalid cell")
        plan_cell = plan_by_label[label]
        completion_cell = completed_by_label[label]
        assert isinstance(plan_cell, Mapping)
        assert isinstance(completion_cell, Mapping)
        expected_cell_root = (
            run_root
            / "cells"
            / f'{int(plan_cell["order_index"]) + 1:02d}-{label}'
        ).resolve()
        completion_relative = completion_cell.get("path")
        if not isinstance(completion_relative, str):
            raise ValueError(f"live completion cell {label} lacks a path")
        completion_root = (REPOSITORY_ROOT / completion_relative).resolve()
        if completion_root != expected_cell_root:
            raise ValueError(f"live completion cell {label} path drifted")
        contract_path = completion_root / "cell_contract.json"
        validation_path = completion_root / "strict_development_validation.json"
        manifest_path = completion_root / "cell_manifest.json"
        result_path = completion_root / "evidence/result.json"
        timeline_path = completion_root / "evidence/queue_timeline.jsonl"
        server_path = completion_root / "server" / f'vllm_{int(plan["port"])}.log'
        runner_stdout_path = completion_root / "runner.stdout.log"
        runner_stderr_path = completion_root / "runner.stderr.log"
        lifecycle_stdout_path = completion_root / "server_lifecycle.stdout.log"
        lifecycle_stderr_path = completion_root / "server_lifecycle.stderr.log"
        cell_required = (
            contract_path,
            validation_path,
            manifest_path,
            result_path,
            timeline_path,
            server_path,
            runner_stdout_path,
            runner_stderr_path,
            lifecycle_stdout_path,
            lifecycle_stderr_path,
        )
        if any(not item.is_file() for item in cell_required):
            missing = [str(item) for item in cell_required if not item.is_file()]
            raise ValueError(f"live cell {label} evidence is incomplete: {missing}")
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"live cell {label} evidence is not valid JSON") from exc
        if not all(
            isinstance(item, Mapping)
            for item in (contract, validation, manifest, result)
        ):
            raise ValueError(f"live cell {label} JSON evidence is not an object")
        evidence = manifest.get("evidence")
        if (
            manifest.get("schema")
            != "paste_repro.scheduler_live_sensitivity_cell_evidence"
            or manifest.get("version") != 1
            or manifest.get("development_only") is not True
            or manifest.get("cell") != label
            or not isinstance(evidence, Mapping)
        ):
            raise ValueError(f"live cell {label} manifest contract failed")
        expected_evidence_paths = {
            str(item.relative_to(REPOSITORY_ROOT))
            for item in (
                contract_path,
                validation_path,
                result_path,
                timeline_path,
                server_path,
                runner_stdout_path,
                runner_stderr_path,
                lifecycle_stdout_path,
                lifecycle_stderr_path,
            )
        }
        if not expected_evidence_paths.issubset(evidence):
            raise ValueError(f"live cell {label} manifest omits required raw evidence")
        for relative, digest in evidence.items():
            if not isinstance(relative, str) or not re.fullmatch(
                r"[0-9a-f]{64}", str(digest)
            ):
                raise ValueError(f"live cell {label} manifest has an invalid binding")
            evidence_path = (REPOSITORY_ROOT / relative).resolve()
            try:
                evidence_path.relative_to(completion_root)
            except ValueError as exc:
                raise ValueError(
                    f"live cell {label} manifest escapes its cell directory"
                ) from exc
            if not evidence_path.is_file() or _sha256(evidence_path) != digest:
                raise ValueError(f"live cell {label} manifest SHA256 mismatch: {relative}")
            source_files[relative] = str(digest)
        source_files[str(manifest_path.relative_to(REPOSITORY_ROOT))] = _sha256(
            manifest_path
        )

        spec = cell.get("spec")
        task = cell.get("task_e2e_s")
        request = cell.get("llm_request_duration_s")
        summary_transport = cell.get("transport_validation")
        contract_spec = contract.get("spec")
        if not all(
            isinstance(item, Mapping)
            for item in (spec, task, request, contract_spec, summary_transport)
        ):
            raise ValueError(f"live summary cell {label} is incomplete")
        assert isinstance(spec, Mapping)
        assert isinstance(task, Mapping)
        assert isinstance(request, Mapping)
        assert isinstance(contract_spec, Mapping)
        assert isinstance(summary_transport, Mapping)
        if any(
            spec.get(key) != plan_cell.get(key)
            or contract_spec.get(key) != plan_cell.get(key)
            for key in spec_keys
        ):
            raise ValueError(f"live cell {label} spec drifted across evidence layers")
        if (
            contract.get("schema")
            != "paste_repro.scheduler_live_sensitivity_cell_contract"
            or contract.get("version") != 1
            or contract.get("development_only") is not True
            or contract.get("formal_eligible") is not False
            or contract.get("order_index") != plan_cell.get("order_index")
            or contract.get("bindings") != plan_bindings
            or contract.get("workload", {}).get("sha256") != workload_sha
            or validation.get("valid") is not True
            or validation.get("fresh_server_identity") is not True
            or validation.get("all_sources_exactly_once") is not True
            or int(validation.get("task_count", -1)) != 80
            or int(validation.get("llm_request_count", -1)) != 240
            or int(validation.get("authoritative_tool_commit_count", -1)) != 160
        ):
            raise ValueError(f"live cell {label} contract/strict validation failed")
        server_instance_id = contract.get("server_instance_id")
        if not isinstance(server_instance_id, str) or not server_instance_id:
            raise ValueError(f"live cell {label} lacks a server instance identity")
        if server_instance_id in server_instance_ids:
            raise ValueError("live cells reused a server instance identity")
        server_instance_ids.add(server_instance_id)

        validation_transport = validation.get("transport_validation")
        cell_transport_contract = contract.get("transport_contract")
        if (
            not isinstance(validation_transport, Mapping)
            or not isinstance(cell_transport_contract, Mapping)
            or summary_transport != validation_transport
            or cell_transport_contract.get("remediation_version")
            != transport_contract.get("remediation_version")
            or cell_transport_contract.get("visit_min_start_interval_s") != 3.0
            or cell_transport_contract.get(
                "accepted_http_attempts_per_tool_invocation"
            )
            != 1
            or cell_transport_contract.get("zero_retries_required") is not True
            or validation_transport.get("visit_min_start_interval_s") != 3.0
            or validation_transport.get("tool_invocation_count") != 160
            or validation_transport.get("physical_http_attempt_count") != 160
            or validation_transport.get("http_retry_count") != 0
            or validation_transport.get("http_429_count") != 0
            or validation_transport.get("all_status_200") is not True
        ):
            raise ValueError(f"live cell {label} transport remediation gate failed")

        cell_kind = str(plan_cell["cell"])
        target = float(plan_cell["physical_kv_target"])
        expected_policy = "fcfs" if cell_kind == "A" else "online_joint_pacer_v2"
        expected_target = None if cell_kind == "A" else format(target, ".2f")
        treatment = contract.get("treatment")
        if (
            not isinstance(treatment, Mapping)
            or treatment.get("policy") != expected_policy
            or validation.get("scheduler_policy") != expected_policy
            or validation.get("physical_kv_target_visible_to_server")
            != expected_target
            or treatment.get("physical_kv_target") != expected_target
            or (
                cell_kind == "A" and treatment.get("physical_kv_admission") is not None
            )
            or (
                cell_kind == "E" and treatment.get("physical_kv_admission") != "1"
            )
        ):
            raise ValueError(f"live cell {label} scheduler treatment drifted")

        tasks = result.get("tasks")
        requests = result.get("llm_events")
        tool_attempts = result.get("tool_attempt_records")
        if (
            not isinstance(tasks, list)
            or len(tasks) != 80
            or not isinstance(requests, list)
            or len(requests) != 240
            or not isinstance(tool_attempts, list)
            or len(tool_attempts) != 160
            or any(not isinstance(row, Mapping) or row.get("ok") is not True for row in tasks)
            or any(
                not isinstance(row, Mapping)
                or row.get("ok") is not True
                or row.get("attempts") != 1
                or row.get("http_status") != 200
                or not isinstance(row.get("task_id"), str)
                or row.get("call_index") not in {0, 1, 2}
                for row in requests
            )
        ):
            raise ValueError(f"live cell {label} raw result completion gate failed")
        try:
            transport_evidence = _clean_live_transport_evidence(tool_attempts)
            broker_evidence = _clean_live_broker_evidence(
                result.get("broker_final_snapshot")
            )
        except ValueError as exc:
            raise ValueError(f"live cell {label}: {exc}") from exc
        transport_evidence["broker"] = broker_evidence
        observed_sources = {
            str(row["source_id"]): float(row["e2e_s"]) for row in tasks
        }
        if len(observed_sources) != 80:
            raise ValueError(f"live cell {label} source keys are not unique")
        task_identities = {
            str(row["source_id"]): (
                row.get("task_id"),
                row.get("replica"),
                row.get("question_sha256"),
                row.get("expected_url"),
            )
            for row in tasks
        }
        if (
            len(task_identities) != 80
            or any(
                not isinstance(task_id, str)
                or not task_id
                or replica != 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(question_sha))
                or not isinstance(expected_url, str)
                or not expected_url
                for task_id, replica, question_sha, expected_url in task_identities.values()
            )
        ):
            raise ValueError(f"live cell {label} task identity ledger is malformed")
        observed_request_keys = {
            (str(row["task_id"]), int(row["call_index"])) for row in requests
        }
        expected_request_keys = {
            (str(identity[0]), call_index)
            for identity in task_identities.values()
            for call_index in range(3)
        }
        if observed_request_keys != expected_request_keys:
            raise ValueError(f"live cell {label} LLM call identity matrix drifted")
        tool_invocations = {
            (
                str(row.get("session_id")),
                str(row.get("tool")),
                str(row.get("invocation_digest")),
            )
            for row in tool_attempts
            if isinstance(row, Mapping)
        }
        if (
            len(tool_invocations) != 160
            or any(
                session not in {str(identity[0]) for identity in task_identities.values()}
                or tool not in {"search", "visit"}
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for session, tool, digest in tool_invocations
            )
        ):
            raise ValueError(f"live cell {label} tool invocation identity drifted")
        source_values[label] = observed_sources
        task_identity_values[label] = task_identities
        tool_invocation_values[label] = tool_invocations
        observed_task = _observed_live_distribution(list(observed_sources.values()))
        observed_request = _observed_live_distribution(
            [float(row["duration_s"]) for row in requests]
        )
        for reported, observed, name in (
            (task, observed_task, "task"),
            (request, observed_request, "request"),
        ):
            if any(
                int(reported[key]) != int(observed[key])
                if key == "count"
                else not math.isclose(
                    float(reported[key]),
                    float(observed[key]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for key in distribution_keys
            ):
                raise ValueError(f"live cell {label} {name} summary arithmetic drifted")

        server_text = server_path.read_text(encoding="utf-8", errors="replace")
        lifecycle_stdout = lifecycle_stdout_path.read_text(
            encoding="utf-8", errors="replace"
        )
        if (
            "Traceback (most recent call last)" in server_text
            or runner_stderr_path.stat().st_size != 0
            or lifecycle_stderr_path.stat().st_size != 0
            or re.search(
                r"vLLM pid [0-9]+ stopped cleanly\.", lifecycle_stdout
            )
            is None
        ):
            raise ValueError(f"live cell {label} process lifecycle was not clean")
        if cell_kind == "E" and "installed policy=online_joint_pacer_v2" not in server_text:
            raise ValueError(f"live cell {label} lacks Joint policy installation evidence")
        physical = _physical_kv_log_summary(
            server_text, expected_target=None if cell_kind == "A" else target
        )
        compact_cells[label] = {
            "cell": cell_kind,
            "context_padding_tokens": int(spec["context_padding_tokens"]),
            "max_active_tasks": int(spec["max_active_tasks"]),
            "physical_kv_target": target,
            "task_count": int(task["count"]),
            "task_mean_s": float(task["mean"]),
            "task_p95_s": float(task["p95"]),
            "request_p95_s": float(request["p95"]),
            "request_p99_s": float(request["p99"]),
            "transport_evidence": transport_evidence,
            "physical_kv_telemetry": physical,
        }

    source_key_sets = {label: set(values) for label, values in source_values.items()}
    if len({frozenset(keys) for keys in source_key_sets.values()}) != 1:
        raise ValueError("live sensitivity cells do not have source-key parity")
    if len(
        {
            tuple(sorted(identities.items()))
            for identities in task_identity_values.values()
        }
    ) != 1:
        raise ValueError("live sensitivity cells do not have task-identity parity")
    if len(
        {frozenset(invocations) for invocations in tool_invocation_values.values()}
    ) != 1:
        raise ValueError("live sensitivity cells do not have tool-invocation parity")
    if len(server_instance_ids) != len(plan_cells):
        raise ValueError("live sensitivity did not use one fresh server per cell")

    compact_effects: list[dict[str, Any]] = []
    observed_effect_pairs: set[tuple[str, str]] = set()
    for row in effects:
        if not isinstance(row, Mapping):
            raise ValueError("live summary contains an invalid A/E effect")
        baseline = str(row["baseline"])
        candidate = str(row["candidate"])
        if baseline not in source_values or candidate not in source_values:
            raise ValueError("live A/E effect references an unknown cell")
        base = source_values[baseline]
        observed = source_values[candidate]
        base_mean = statistics.fmean(base.values())
        candidate_mean = statistics.fmean(observed.values())
        reduction = (base_mean - candidate_mean) / base_mean
        faster = sum(base[source] > observed[source] for source in base)
        if (
            not math.isclose(float(row["baseline_mean_s"]), base_mean, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(float(row["candidate_mean_s"]), candidate_mean, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(float(row["relative_reduction"]), reduction, rel_tol=0.0, abs_tol=1e-12)
            or int(row["faster_source_count"]) != faster
        ):
            raise ValueError("live A/E effect arithmetic drifted")
        observed_effect_pairs.add((baseline, candidate))
        compact_effects.append(
            {
                "pair_group": str(row["pair_group"]),
                "baseline": baseline,
                "candidate": candidate,
                "baseline_mean_s": base_mean,
                "candidate_mean_s": candidate_mean,
                "relative_reduction": reduction,
                "faster_source_count": faster,
                "baseline_task_p95_s": compact_cells[baseline]["task_p95_s"],
                "candidate_task_p95_s": compact_cells[candidate]["task_p95_s"],
                "task_p95_relative_reduction": (
                    compact_cells[baseline]["task_p95_s"]
                    - compact_cells[candidate]["task_p95_s"]
                )
                / compact_cells[baseline]["task_p95_s"],
            }
        )
    expected_effect_pairs: set[tuple[str, str]] = set()
    for group in {str(row["pair_group"]) for row in plan_cells}:
        baselines = [
            str(row["label"])
            for row in plan_cells
            if row["pair_group"] == group and row["cell"] == "A"
        ]
        candidates = [
            str(row["label"])
            for row in plan_cells
            if row["pair_group"] == group and row["cell"] == "E"
        ]
        if len(baselines) == 1:
            expected_effect_pairs.update((baselines[0], item) for item in candidates)
    if observed_effect_pairs != expected_effect_pairs:
        raise ValueError("live A/E effect rows are incomplete or duplicated")

    compact_targets: list[dict[str, Any]] = []
    if suite == "high":
        if targets:
            raise ValueError("high suite unexpectedly reports a target sweep")
    else:
        reference_cells = [
            row
            for row in plan_cells
            if row["cell"] == "E"
            and int(row["context_padding_tokens"]) == 10_000
            and int(row["max_active_tasks"]) == 80
            and math.isclose(float(row["physical_kv_target"]), 0.93)
        ]
        target_candidates = [
            row
            for row in plan_cells
            if row["cell"] == "E"
            and int(row["context_padding_tokens"]) == 10_000
            and int(row["max_active_tasks"]) == 80
        ]
        expected_target_candidates = {
            str(row["label"]) for row in target_candidates
        }
        if len(reference_cells) != 1:
            raise ValueError("live target sensitivity lacks a unique .93 reference")
        reference_label = str(reference_cells[0]["label"])
        reference_mean = compact_cells[reference_label]["task_mean_s"]
        observed_target_candidates: set[str] = set()
        for row in targets:
            if not isinstance(row, Mapping):
                raise ValueError("live summary contains an invalid target effect")
            candidate = str(row["candidate"])
            if (
                row.get("reference") != reference_label
                or candidate not in compact_cells
            ):
                raise ValueError("live target effect references an unknown cell")
            candidate_mean = compact_cells[candidate]["task_mean_s"]
            relative_change = (candidate_mean - reference_mean) / reference_mean
            if (
                not math.isclose(
                    float(row["target"]),
                    compact_cells[candidate]["physical_kv_target"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["mean_s"]),
                    candidate_mean,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(row["relative_change_vs_u093"]),
                    relative_change,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("live target effect arithmetic drifted")
            observed_target_candidates.add(candidate)
            compact_targets.append(
                {
                    "reference": reference_label,
                    "candidate": candidate,
                    "target": compact_cells[candidate]["physical_kv_target"],
                    "mean_s": candidate_mean,
                    "relative_change_vs_u093": relative_change,
                }
            )
        if observed_target_candidates != expected_target_candidates:
            raise ValueError("live target effect rows are incomplete or duplicated")

    return {
        "run_tag": plan.get("run_tag"),
        "suite": plan.get("suite"),
        "model_id": plan.get("evidence_boundary", {}).get("same_model_family"),
        "gpu_ids": plan.get("gpu_ids"),
        "workload_sha256": workload_sha,
        "same_source_keys_across_cells": True,
        "same_task_identities_across_cells": True,
        "same_tool_invocations_across_cells": True,
        "execution_order": [
            str(row["label"])
            for row in sorted(plan_cells, key=lambda item: int(item["order_index"]))
        ],
        "fresh_unique_server_instance_per_cell": True,
        "single_run_per_cell": True,
        "confidence_interval_available": False,
        "cells": compact_cells,
        "a_to_e_effects": compact_effects,
        "physical_kv_target_sensitivity": compact_targets,
        "excluded_r2_transport_provenance": failed_r2_provenance,
        "shape_r1_harness_repair": shape_r1_provenance,
        "evidence_boundary": summary.get("evidence_boundary"),
        "source_files": dict(sorted(source_files.items())),
        "planned_bindings": dict(plan_bindings),
        "evidence_role": "sha_bound_reanalysis_of_comment3_live_sensitivity",
    }


def _external_trace_center_summary(path: Path) -> dict[str, Any]:
    """Fail-closed reanalysis of the real-trace functional A/E center point."""

    aggregate_path = path.resolve()
    root = aggregate_path.parent
    cell_names = ("fcfs", "joint_target093")
    cell_summary_paths = {name: root / name / "summary.json" for name in cell_names}
    workload_paths = {
        name: root / name / "prepared_workload.json" for name in cell_names
    }
    request_paths = {
        name: root / name / "request_events.jsonl" for name in cell_names
    }
    server_paths = {name: root / name / "server.log" for name in cell_names}
    required = (
        aggregate_path,
        *cell_summary_paths.values(),
        *workload_paths.values(),
        *request_paths.values(),
        *server_paths.values(),
    )
    if any(not item.is_file() for item in required):
        missing = [str(item) for item in required if not item.is_file()]
        raise ValueError(f"trace-center evidence is incomplete: {missing}")
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        cell_summaries = {
            name: json.loads(item.read_text(encoding="utf-8"))
            for name, item in cell_summary_paths.items()
        }
        request_rows = {
            name: [
                json.loads(line)
                for line in item.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for name, item in request_paths.items()
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("trace-center evidence is not valid JSON/JSONL") from exc
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("schema") != "paste_repro.joint_ab_summary"
        or aggregate.get("version") != 1
        or aggregate.get("status")
        != "functional_ab_not_full_paper_reproduction"
    ):
        raise ValueError("unsupported trace-center aggregate schema/status")
    invariants = aggregate.get("comparison_invariants")
    if not isinstance(invariants, Mapping):
        raise ValueError("trace-center comparison invariants are missing")
    workload_sha = str(invariants.get("same_workload_sha256"))
    expected_invariants = {
        "all_requests_succeeded": True,
        "metadata_source": "online",
        "request_count": 264,
        "trace_count": 30,
        "tool_overlap_mode": "learned",
    }
    invariant_drift = {
        key: (expected, invariants.get(key))
        for key, expected in expected_invariants.items()
        if invariants.get(key) != expected
    }
    if invariant_drift or not re.fullmatch(r"[0-9a-f]{64}", workload_sha):
        raise ValueError(f"trace-center invariant drift: {invariant_drift}")
    if any(_sha256(item) != workload_sha for item in workload_paths.values()):
        raise ValueError("trace-center prepared workloads are not byte-identical")

    runs = aggregate.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("trace-center aggregate must contain exactly two runs")
    runs_by_name = {
        str(row.get("name")): row for row in runs if isinstance(row, Mapping)
    }
    if set(runs_by_name) != set(cell_names):
        raise ValueError("trace-center aggregate run identities drifted")
    expected_policies = {
        "fcfs": "fcfs",
        "joint_target093": "online_joint_pacer_v2",
    }
    request_keys: dict[str, set[tuple[str, int]]] = {}
    compact_cells: dict[str, dict[str, Any]] = {}
    for name in cell_names:
        run = runs_by_name[name]
        summary = cell_summaries[name]
        if not isinstance(summary, Mapping):
            raise ValueError(f"trace-center {name} summary is not an object")
        policy = expected_policies[name]
        if (
            run.get("policy") != policy
            or run.get("request_count") != 264
            or run.get("requests_failed") != 0
            or run.get("metadata_source") != "online"
            or run.get("workload_sha256") != workload_sha
        ):
            raise ValueError(f"trace-center aggregate run {name} drifted")
        environment = summary.get("scheduler_environment")
        physical = summary.get("physical_kv_admission")
        workload = summary.get("workload")
        if not all(
            isinstance(item, Mapping)
            for item in (environment, physical, workload)
        ):
            raise ValueError(f"trace-center {name} lacks environment/telemetry")
        assert isinstance(environment, Mapping)
        assert isinstance(physical, Mapping)
        assert isinstance(workload, Mapping)
        if (
            summary.get("requests_total") != 264
            or summary.get("requests_success") != 264
            or summary.get("requests_failed") != 0
            or summary.get("final_failure_count") != 0
            or summary.get("metadata_source") != "online"
            or summary.get("scheduler_metadata_mode") != "online"
            or summary.get("max_active_traces") != 30
            or environment.get("VLLM_SCHED_POLICY") != policy
            or workload.get("request_count") != 264
            or workload.get("trace_count") != 30
            or workload.get("tool_overlap_mode") != "learned"
        ):
            raise ValueError(f"trace-center {name} result contract failed")
        rows = request_rows[name]
        if (
            len(rows) != 264
            or any(
                not isinstance(row, Mapping)
                or row.get("ok") is not True
                or row.get("metadata_source") != "online"
                for row in rows
            )
        ):
            raise ValueError(f"trace-center {name} request evidence failed")
        keys = {
            (str(row["trace_id"]), int(row["call_index"]))
            for row in rows
        }
        if len(keys) != 264:
            raise ValueError(f"trace-center {name} request keys are not unique")
        request_keys[name] = keys
        server_text = server_paths[name].read_text(
            encoding="utf-8", errors="replace"
        )
        if "Traceback (most recent call last)" in server_text:
            raise ValueError(f"trace-center {name} server log contains a traceback")
        physical_lines = [
            line
            for line in server_text.splitlines()
            if "[sched_policy_patch:physical_kv]" in line
        ]
        fail_closed_lines = [
            line for line in physical_lines if "decision=fail_closed" in line
        ]
        wrong_target_lines = [
            line
            for line in physical_lines
            if "target_utilization=0.930000" not in line
        ]
        if name == "fcfs":
            if physical_lines:
                raise ValueError("FCFS trace-center log unexpectedly used physical admission")
        elif (
            len(physical_lines) < 155
            or fail_closed_lines
            or wrong_target_lines
            or "installed policy=online_joint_pacer_v2" not in server_text
        ):
            raise ValueError("Joint trace-center server telemetry contract failed")
        compact_cells[name] = {
            "policy": policy,
            "request_count": 264,
            "trace_count": 30,
            "experiment_wall_time_s": float(run["experiment_wall_time_s"]),
            "task_e2e_s": {
                key: float(value)
                for key, value in run["task_e2e_s"].items()
            },
            "request_latency_s": {
                key: float(value)
                for key, value in run["request_latency_s"].items()
            },
            "mean_queue_time_s": float(run["mean_queue_time_s"]),
            "server_physical_line_count": len(physical_lines),
            "server_fail_closed_line_count": len(fail_closed_lines),
        }
    if request_keys["fcfs"] != request_keys["joint_target093"]:
        raise ValueError("trace-center A/E request identities differ")

    joint_physical = cell_summaries["joint_target093"][
        "physical_kv_admission"
    ]
    assert isinstance(joint_physical, Mapping)
    target = joint_physical.get("target_utilization")
    usage = joint_physical.get("usage")
    if not isinstance(target, Mapping) or not isinstance(usage, Mapping):
        raise ValueError("Joint trace-center physical distributions are missing")
    if (
        joint_physical.get("sample_count") != 155
        or joint_physical.get("malformed_sample_count") != 0
        or joint_physical.get("fail_closed_count") != 0
        or float(target.get("min", -1.0)) != 0.93
        or float(target.get("max", -1.0)) != 0.93
        or float(usage.get("max", 2.0)) >= 0.93
        or joint_physical.get("fit_admit_zero_sample_count") != 0
    ):
        raise ValueError("Joint trace-center physical target gate failed")
    maximum_usage = float(usage["max"])
    # The first physical line is emitted before the metrics baseline.  The
    # post-baseline result therefore has 155 samples while this log has 156.
    if compact_cells["joint_target093"]["server_physical_line_count"] not in {
        155,
        156,
    }:
        raise ValueError("Joint trace-center log/result sample alignment drifted")

    baseline = aggregate.get("baseline")
    joint = aggregate.get("joint")
    reductions = aggregate.get("relative_reduction")
    if not all(isinstance(item, Mapping) for item in (baseline, joint, reductions)):
        raise ValueError("trace-center aggregate effects are missing")
    assert isinstance(baseline, Mapping)
    assert isinstance(joint, Mapping)
    assert isinstance(reductions, Mapping)
    recomputed = {
        "task_e2e_mean": (
            float(baseline["task_e2e_s"]["mean"])
            - float(joint["task_e2e_s"]["mean"])
        )
        / float(baseline["task_e2e_s"]["mean"]),
        "task_e2e_p95": (
            float(baseline["task_e2e_s"]["p95"])
            - float(joint["task_e2e_s"]["p95"])
        )
        / float(baseline["task_e2e_s"]["p95"]),
        "mean_queue_time": (
            float(baseline["mean_queue_time_s"])
            - float(joint["mean_queue_time_s"])
        )
        / float(baseline["mean_queue_time_s"]),
        "request_latency_mean": (
            float(baseline["request_latency_s"]["mean"])
            - float(joint["request_latency_s"]["mean"])
        )
        / float(baseline["request_latency_s"]["mean"]),
    }
    if any(
        not math.isclose(float(reductions[key]), value, rel_tol=0.0, abs_tol=1e-12)
        for key, value in recomputed.items()
    ):
        raise ValueError("trace-center aggregate reduction arithmetic drifted")
    return {
        "label": root.name,
        "evidence_role": "real_trace_central_functional_ae_single_run",
        "not_a_statistically_replicated_paper_result": True,
        "not_a_target_sensitivity_result": True,
        "same_model_and_gpu_shape": True,
        "workload_sha256": workload_sha,
        "same_request_keys": True,
        "request_count_per_cell": 264,
        "trace_count_per_cell": 30,
        "cells": compact_cells,
        "relative_reduction": recomputed,
        "physical_kv": {
            "post_baseline_sample_count": 155,
            "server_log_line_count": compact_cells["joint_target093"][
                "server_physical_line_count"
            ],
            "target_utilization": 0.93,
            "maximum_observed_usage": maximum_usage,
            "malformed_sample_count": 0,
            "fail_closed_count": 0,
            "fit_admit_zero_sample_count": 0,
            "target_was_binding": False,
        },
        "source_files": {
            str(item.relative_to(REPOSITORY_ROOT)): _sha256(item)
            for item in required
        },
    }


def _static_mapping() -> dict[str, Any]:
    hook_text = HOOK_PATH.read_text(encoding="utf-8")
    agent_text = LIVE_AGENT_PATH.read_text(encoding="utf-8")
    return {
        "paper_priority_formula_is_literal_in_hook": (
            "ExposedToolGain" in hook_text or "LLMPressure" in hook_text
        ),
        "paper_engine_pressure_formula_is_literal_in_hook": (
            "DecodeLoad" in hook_text or "KVLoad" in hook_text
        ),
        "literal_gamma_parameter_in_joint_v2": bool(
            re.search(r"JOINT_V2.*GAMMA", hook_text)
        ),
        "deployed_surrogate": {
            "direction": "lower_score_first",
            "formula": "C_i = P_llm,i - G_tool,i - G_progress,i - A_i",
            "ExposedToolGain": (
                "G_tool = tool_beta * nwc * min(nw, tool_wait_cap) / "
                "(1 + projected_pressure * prompt_tokens/context_ref)"
            ),
            "LLMPressure": (
                "P_llm = service + nonlinear_context_penalty + tail_beta*task_tail "
                "+ over_budget_penalty + new_session_penalty"
            ),
            "Aging": "A_i = time_aging_alpha * scheduler_wait_seconds",
            "hard_lanes": (
                "final-call lane and exact remaining-call lane precede the continuous score"
            ),
        },
        "DecodeLoad": {
            "actual_proxy": "len(engine_running_requests)",
            "normalization": "decode_target_running",
            "active_in_registered_physical_kv_path": False,
            "reason": (
                "physical-KV admission is selected instead of "
                "_apply_hbm_capacity_with_reserve, which owns the decode band"
            ),
        },
        "KVLoad": {
            "ranking_proxy": "(live + virtual + predicted request KV tokens) / configured HBM target tokens",
            "admission_measurement": "ceil(kv_cache_manager.usage * num_gpu_blocks) * block_size",
            "physical_capacity": "cache_config.num_gpu_blocks * cache_config.block_size",
            "active_in_registered_physical_kv_path": True,
        },
        "pressure_band": {
            "legacy_low_high": [0.82, 1.02],
            "active_in_registered_physical_kv_path": False,
            "registered_active_limit": "physical KV utilization target 0.93",
        },
        "gamma": {
            "literal_implementation": None,
            "nearest_non_equivalent_term": (
                "context_alpha=1.4 multiplying projected_KV_pressure^1.35 * prompt_cost"
            ),
        },
        "aging": {
            "linear_alpha": 0.2,
            "gate_or_rescue_seconds": 40.0,
            "physical_rescue_never_crosses": "100% physical KV capacity",
        },
        "speculation_budget": {
            "global_workers": 4,
            "maximum_speculative_workers": 2,
            "minimum_reserved_speculative_workers": 0,
            "visit_capacity": 2,
            "search_capacity": 3,
            "maximum_pending_predictions": 128,
            "ttl_seconds": 120,
            "selection_evidence": (
                "F1 min=1 improved only 0.2079% over F0 min=0; 6/16 sources "
                "were faster and the bootstrap interval crossed zero, so F0 was selected"
            ),
        },
        "telemetry_contract": {
            "agent_emits": [
                field
                for field in ("nw", "nwc", "rtw", "pt", "po", "rc", "nrg", "nps", "tqa", "tqs", "tra", "trs")
                if f'"{field}"' in agent_text
            ],
            "hook_reads_directly": [
                field
                for field in ("nw", "nwc", "rtw", "pt", "po", "rc", "nrg", "nps", "tqa", "tqs", "tra", "trs")
                if re.search(rf'["\']{field}["\']', hook_text)
            ],
            "important_boundary": (
                "nrg/nps and global tool queue counts are recorded but not direct "
                "Joint-v2 score inputs; nw/nwc/rtw are the score inputs"
            ),
        },
    }


def _validate_reference_bindings() -> None:
    source = FORMAL_RUNNER_PATH.read_text(encoding="utf-8")
    required = {
        "VLLM_SCHED_TIME_AGING_ALPHA": "0.2",
        "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING": "96",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": "0.93",
        "VLLM_SCHED_JOINT_V2_TOOL_BETA": "0.9",
        "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA": "1.4",
        "VLLM_SCHED_HBM_LOW_PRESSURE": "0.82",
        "VLLM_SCHED_HBM_HIGH_PRESSURE": "1.02",
        "PASTE_LIVE_TOOL_WORKERS": "4",
        "PASTE_LIVE_SPECULATIVE_TOOL_WORKERS": "2",
        "PASTE_LIVE_MIN_SPECULATIVE_TOOL_WORKERS": "0",
        "PASTE_LIVE_MAX_SPECULATIVE_PENDING": "128",
    }
    missing = [
        f'{key}={value}'
        for key, value in required.items()
        if f'"{key}": "{value}"' not in source
    ]
    if missing:
        raise RuntimeError(
            "reference configuration drifted: " + ", ".join(missing)
        )


def _summarize_proxy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for profile in (item.name for item in PROXY_PROFILES):
        selected = [row for row in rows if row["profile"] == profile]
        summary[profile] = {
            "scenario_count": len(selected),
            "production_pairwise_agreement_mean": statistics.fmean(
                float(row["production_pairwise_agreement_vs_registered"])
                for row in selected
            ),
            "production_pairwise_agreement_min": min(
                float(row["production_pairwise_agreement_vs_registered"])
                for row in selected
            ),
            "continuous_pairwise_agreement_mean": statistics.fmean(
                float(row["continuous_pairwise_agreement_vs_registered"])
                for row in selected
            ),
            "continuous_pairwise_agreement_min": min(
                float(row["continuous_pairwise_agreement_vs_registered"])
                for row in selected
            ),
            "production_top5_overlap_mean": statistics.fmean(
                float(row["production_top5_overlap_vs_registered"])
                for row in selected
            ),
            "continuous_top5_overlap_mean": statistics.fmean(
                float(row["continuous_top5_overlap_vs_registered"])
                for row in selected
            ),
            "physical_admit_count_min": min(
                int(row["physical_admission"].get("admit", 0)) for row in selected
            ),
            "physical_admit_count_max": max(
                int(row["physical_admission"].get("admit", 0)) for row in selected
            ),
        }
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "profile",
        "workload",
        "context_scale",
        "load_ratio",
        "running_count",
        "live_tokens",
        "physical_kv_tokens",
        "decode_load_count_ratio",
        "physical_kv_load_ratio",
        "production_pairwise_agreement_vs_registered",
        "continuous_pairwise_agreement_vs_registered",
        "production_top5_overlap_vs_registered",
        "continuous_top5_overlap_vs_registered",
        "logical_admissible_count",
        "physical_admit_count",
        "physical_effective_cap",
        "max_formula_error_s",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames}
            flat["physical_admit_count"] = row["physical_admission"].get("admit", 0)
            flat["physical_effective_cap"] = row["physical_admission"].get(
                "effective_cap", 0
            )
            writer.writerow(flat)


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _write_svg(
    path: Path,
    empirical: Sequence[Mapping[str, Any]],
    proxy_summary: Mapping[str, Mapping[str, Any]],
) -> None:
    width, height = 1220, 620
    left_x, top_y, chart_w, chart_h = 80, 75, 690, 430
    minimum, maximum = -35.0, 35.0
    zero_y = top_y + chart_h * maximum / (maximum - minimum)
    bar_gap = chart_w / len(empirical)
    colors = {
        "legacy_count_target32": "#d95f02",
        "legacy_count_target56": "#e6ab02",
        "legacy_count_target64": "#7570b3",
        "legacy_count_target64_stage_lane": "#1b9e77",
        "physical_kv_target_0.93": "#2c7fb8",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:700}.label{font-size:12px}.small{font-size:11px;fill:#444}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#444;stroke-width:1.4}</style>',
        '<text x="50" y="34" class="title">Scheduler robustness: checked-in A100 evidence vs CPU policy replay</text>',
        '<text x="80" y="60" class="label">A. Historical mean-task reduction (different development configurations; not one causal load curve)</text>',
    ]
    for tick in (-30, -20, -10, 0, 10, 20, 30):
        y = top_y + chart_h * (maximum - tick) / (maximum - minimum)
        lines.append(f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x+chart_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left_x-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick}%</text>')
    lines.append(f'<line x1="{left_x}" y1="{zero_y:.1f}" x2="{left_x+chart_w}" y2="{zero_y:.1f}" class="axis"/>')
    for index, point in enumerate(empirical):
        value = float(point["mean_task_reduction_pct"])
        x = left_x + index * bar_gap + bar_gap * 0.18
        y_value = top_y + chart_h * (maximum - value) / (maximum - minimum)
        y = min(y_value, zero_y)
        bar_height = max(1.0, abs(zero_y - y_value))
        color = colors.get(str(point["controller"]), "#666")
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_gap*0.64:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
        value_y = y - 6 if value >= 0 else y + bar_height + 15
        lines.append(f'<text x="{x+bar_gap*0.32:.1f}" y="{value_y:.1f}" text-anchor="middle" class="small">{value:+.1f}%</text>')
        label = _svg_escape(str(point["label"]))
        lines.append(f'<text transform="translate({x+bar_gap*0.32:.1f},{top_y+chart_h+14}) rotate(52)" class="small">{label}</text>')

    panel_x = 820
    lines.extend(
        [
            f'<text x="{panel_x}" y="60" class="label">B. Mean ordering agreement vs registered proxy</text>',
            f'<line x1="{panel_x}" y1="{top_y+chart_h}" x2="1165" y2="{top_y+chart_h}" class="axis"/>',
        ]
    )
    profile_colors = ("#d95f02", "#2c7fb8", "#1b9e77")
    for index, (profile, summary) in enumerate(proxy_summary.items()):
        value = float(summary["continuous_pairwise_agreement_mean"])
        x = panel_x + index * 110 + 22
        bar_height = value * chart_h
        y = top_y + chart_h - bar_height
        lines.append(f'<rect x="{x}" y="{y:.1f}" width="62" height="{bar_height:.1f}" fill="{profile_colors[index]}"/>')
        lines.append(f'<text x="{x+31}" y="{y-7:.1f}" text-anchor="middle" class="small">{value:.3f}</text>')
        lines.append(f'<text transform="translate({x+31},{top_y+chart_h+14}) rotate(48)" class="small">{_svg_escape(profile)}</text>')
    lines.extend(
        [
            '<text x="80" y="592" class="small">Positive bars mean lower mean task time under the compared Joint bundle. Proxy agreement is a scheduler-decision metric, not latency.</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_report(payload: Mapping[str, Any]) -> str:
    mapping = payload["implementation_mapping"]
    proxy = payload["proxy_summary"]
    empirical = payload["empirical_a100_evidence"]
    external = payload["external_live_aggregates"]
    live_sensitivity = payload["external_live_sensitivity_summaries"]
    trace_centers = payload["external_trace_center_results"]
    parameters = payload["parameter_sensitivity"]
    max_error = float(payload["verification"]["max_formula_error_s"])
    legacy_band_rows = [
        row for row in parameters if row["parameter"] == "legacy_pressure_band"
    ]
    legacy_band_admits = sorted({row["physical_admit_count"] for row in legacy_band_rows})
    utilization_rows = [
        row
        for row in parameters
        if row["parameter"] == "physical_kv_target_utilization"
    ]
    utilization_admits = ", ".join(
        f'{row["value"]}→{row["physical_admit_count"]}'
        for row in utilization_rows
    )
    parameter_rows = "\n".join(
        f'| {row["parameter"]} | `{row["value"]}` '
        f'| {row["production_pairwise_agreement_vs_registered"]:.3f} '
        f'| {row["continuous_pairwise_agreement_vs_registered"]:.3f} '
        f'| {row["physical_admit_count"]} |'
        for row in parameters
    )
    empirical_rows = "\n".join(
        "| {label} | {load_instances} | {controller} | {mean_task_reduction_pct:+.3f}% | {task_p95_reduction_pct:+.3f}% | {evidence_role} |".format(
            **point
        )
        for point in empirical
    )
    proxy_rows = "\n".join(
        f'| {name} | {row["scenario_count"]} | {row["production_pairwise_agreement_mean"]:.3f} '
        f'| {row["continuous_pairwise_agreement_mean"]:.3f} | {row["continuous_pairwise_agreement_min"]:.3f} '
        f'| {row["physical_admit_count_min"]}–{row["physical_admit_count_max"]} |'
        for name, row in proxy.items()
    )
    if external:
        external_rows = "\n".join(
            f'| `{row["path"]}` | {row["formal_profile"]} '
            f'| {row["load"].get("max_active_tasks", "?")} '
            f'| {row["load"].get("context_padding_target_tokens", "?")} '
            f'| {100.0 * row["effects"]["A_to_E"]["aggregate_relative_reduction"]:+.3f}% '
            f'| {100.0 * row["effects"]["A_to_F"]["aggregate_relative_reduction"]:+.3f}% '
            f'| {row["cells"]["E"]["task_p95_s"]:.3f} '
            f'| {row["cells"]["E"]["request_p99_s"]:.3f} |'
            for row in external
        )
        external_section = f"""
## Supplied strict live aggregates

These rows were re-extracted from upstream strict four-cell aggregate files;
their SHA256s and full extracted fields are in `raw_results.json`.  Supplying a
file does not mean this CPU script launched its GPU experiment.

| Aggregate | Profile | Offered | Context padding | A→E mean reduction | A→F mean reduction | E task P95 (s) | E request P99 (s) |
|---|---|---:|---:|---:|---:|---:|---:|
{external_rows}
"""
    else:
        external_section = """
## Supplied strict live aggregates

None were supplied to this invocation.  A completed formal aggregate can be
SHA-bound and re-extracted with repeatable `--live-aggregate PATH` arguments.
"""
    if live_sensitivity:
        target_live_runs = [
            run for run in live_sensitivity if run.get("suite") == "target"
        ]
        high_live_runs = [
            run for run in live_sensitivity if run.get("suite") == "high"
        ]
        live_effect_rows = "\n".join(
            f'| {run["run_tag"]} | {effect["pair_group"]} '
            f'| {effect["baseline"]} | {effect["candidate"]} '
            f'| {effect["baseline_mean_s"]:.3f} '
            f'| {effect["candidate_mean_s"]:.3f} '
            f'| {100.0 * effect["relative_reduction"]:+.3f}% '
            f'| {effect["baseline_task_p95_s"]:.3f} '
            f'| {effect["candidate_task_p95_s"]:.3f} '
            f'| {100.0 * effect["task_p95_relative_reduction"]:+.3f}% '
            f'| {effect["faster_source_count"]}/80 |'
            for run in live_sensitivity
            for effect in run["a_to_e_effects"]
        )
        live_target_rows = "\n".join(
            f'| {run["run_tag"]} | {row["target"]:.2f} '
            f'| {row["mean_s"]:.3f} '
            f'| {run["cells"][row["candidate"]]["task_p95_s"]:.3f} '
            f'| {100.0 * row["relative_change_vs_u093"]:+.3f}% |'
            for run in live_sensitivity
            for row in run["physical_kv_target_sensitivity"]
        )
        live_telemetry_rows = "\n".join(
            f'| {run["run_tag"]} | {label} '
            f'| {telemetry["target_utilization"]:.2f} '
            f'| {telemetry["sample_count"]} '
            f'| {telemetry["usage_max"]:.3f} '
            f'| {telemetry["fit_admit_min"]} '
            f'| {telemetry["admit_min"]} '
            f'| {telemetry["target_budget_truncated_waiting_sample_count"]} '
            f'| {telemetry["fit_admit_zero_sample_count"]} '
            f'| {telemetry["semantic_required_admission_field_malformed_count"]} '
            f'| {telemetry["fail_closed_count"]} '
            f'| {telemetry["raw_line_interleaving_count"]} '
            f'| {str(telemetry["tail_rescue_parse_clean"]).lower()} '
            f'| {str(telemetry["strict_parser_v2_clean"]).lower()} |'
            for run in live_sensitivity
            for label, cell in run["cells"].items()
            if cell["cell"] == "E"
            for telemetry in (cell["physical_kv_telemetry"],)
        )
        live_transport_rows = "\n".join(
            f'| {run["run_tag"]} | {label} '
            f'| {transport["search_record_count"]} '
            f'| {transport["visit_record_count"]} '
            f'| {transport["http_attempt_count"]} '
            f'| {transport["http_retry_count"]} '
            f'| {transport["http_429_count"]} '
            f'| {transport["minimum_adjacent_visit_start_gap_s"]:.3f} '
            f'| {transport["broker"]["commits"]} '
            f'| {transport["broker"]["authoritative_failures"]} |'
            for run in live_sensitivity
            for label, cell in run["cells"].items()
            for transport in (cell["transport_evidence"],)
        )
        excluded_r2_rows = "\n".join(
            f'| {run["run_tag"]} '
            f'| {provenance["transport_counts"]["A"]["http_429_count"]} '
            f'| {provenance["transport_counts"]["E_u085"]["http_429_count"]} '
            f'| {provenance["transport_counts"]["E_u085"]["failed_tool_record_count"]} '
            f'| {len(provenance["source_files"])} |'
            for run in live_sensitivity
            for provenance in (run["excluded_r2_transport_provenance"],)
        )
        live_execution_order = "; ".join(
            f'`{run["run_tag"]}`: '
            + " → ".join(f'`{label}`' for label in run["execution_order"])
            for run in live_sensitivity
        )
        target_effects = [
            effect
            for run in target_live_runs
            for effect in run["a_to_e_effects"]
        ]
        all_target_effects_positive = len(target_effects) == 3 and all(
            effect["relative_reduction"] > 0.0
            for effect in target_effects
        )
        directional_statement = (
            "All three completed target cells are directionally faster than "
            "their common A cell in this one execution."
            if all_target_effects_positive
            else (
                "The completed target cells do not all move in the same direction."
                if target_effects
                else "No target-sweep run was supplied."
            )
        )
        high_shape_rows = "\n".join(
            f'| {run["run_tag"]} '
            f'| {repair["failed_run_tag"]} '
            f'| {repair["rejected_order_index"]} '
            f'| {repair["failed_cell_request_count"]} '
            f'| {len(repair["bound_files"])} '
            f'| {repair["excluded_observed_prefix"]["cell_count"]} '
            f'| {str(repair["excluded_observed_prefix"]["reused_by_replacement"]).lower()} '
            f'| {str(repair["excluded_observed_prefix"]["pooled_with_replacement"]).lower()} '
            f'| {str(repair["excluded_observed_prefix"]["performance_loaded_or_reported"]).lower()} |'
            for run in high_live_runs
            for repair in (run["shape_r1_harness_repair"],)
        )
        high_equivalence_rows = "\n".join(
            f'| {run["run_tag"]} | {row["cell"]} '
            f'| `{row["normalized_sha256"]}` '
            f'| {str(row["equal_after_identity_normalization"]).lower()} |'
            for run in high_live_runs
            for row in run["shape_r1_harness_repair"]["replacement"]
            ["configuration_equivalence"]
        )
        if high_live_runs:
            high_effects_positive = all(
                effect["relative_reduction"] > 0.0
                and effect["task_p95_relative_reduction"] > 0.0
                for run in high_live_runs
                for effect in run["a_to_e_effects"]
            )
            high_direction_sentence = (
                "Both mean task latency and task P95 are lower in the observed "
                "E cell."
                if high_effects_positive
                else "Mean and task-P95 effects do not both move positively."
            )
            high_joint_cell = next(
                cell
                for cell in high_live_runs[0]["cells"].values()
                if cell["cell"] == "E"
            )
            high_physical = high_joint_cell["physical_kv_telemetry"]
            high_shape_section = f"""
### High-shape one-shot replacement and failed-harness boundary

The completed `high` suite is a separate `12k/80` A/E pair.  Its plan binds
the immutable `comment3-shape-r1` plan, failure, rejected cell-5 contract and
stderr, and server lifecycle evidence.  The loader independently confirms
that formal order index 4 failed deterministically before any chat-completion
request, the server stopped cleanly, and no result, timeline, or manifest was
created for that cell.  It also recomputes old→new high-pair configuration
equality after normalizing only the disclosed run/block/order/server identity
fields.

| Replacement | Failed run | Rejected index | Failed-cell requests | Bound failure files | Excluded observed prefix | Reused | Pooled | Prefix performance loaded/reported |
|---|---|---:|---:|---:|---:|---|---|---|
{high_shape_rows}

| Replacement | Cell | Normalized config SHA256 | Equal |
|---|---|---|---|
{high_equivalence_rows}

The four observed prefix cells from the failed six-cell run are excluded: no
performance value from them is loaded, shown, pooled, or used to select the
replacement.  {high_direction_sentence}  The high pair remains one
descriptive run without repeats or a confidence interval.

High E's maximum current usage was `{high_physical["usage_max"]:.3f}` versus
target `{high_physical["target_utilization"]:.2f}`, while predicted-token
budgeting truncated the waiting fit in
`{high_physical["target_budget_truncated_waiting_sample_count"]}` marker
samples.  This establishes active controller telemetry; it neither shows that
current usage reached the target nor isolates the target as the cause of the
observed end-to-end effect.
"""
        else:
            high_shape_section = """
### High-shape one-shot replacement and failed-harness boundary

No completed high-shape replacement was supplied to this invocation.
"""
        live_sensitivity_section = f"""
## Completed comment-3 live sensitivity reanalysis

Each run is bound through its plan, completion record, every cell manifest,
raw result/timeline, and server log SHA.  Source keys and reported effects are
independently recomputed; task identities and all 160 tool-invocation digests
must match across cells.  Each cell has one unique fresh server instance.
The fixed execution order was {live_execution_order}.  These are single-run
development effects without repeats or confidence intervals.

| Run | Shape | A | E | A mean (s) | E mean (s) | Mean reduction | A P95 (s) | E P95 (s) | P95 reduction | Faster sources |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{live_effect_rows}

| Run | Physical target | E mean (s) | E task P95 (s) | Mean change vs `.93` |
|---|---:|---:|---:|---:|
{live_target_rows}

{directional_statement}  This supports functional execution and a positive
direction relative to one common A observation, not an optimum-target claim.
In particular, the lower `.85` descriptive mean cannot establish that `.85`
is optimal: order was fixed rather than randomized, every cell ran once, and
external HTTP service conditions may drift over wall-clock time.  No post-hoc
significance test is reported, and this one Qwen/A100 run supplies no
cross-model or cross-GPU generalization.

{high_shape_section}

| Run | Cell | Search records | Visit records | Physical HTTP attempts | Retries | HTTP 429 | Min visit-start gap (s) | Broker commits | Broker failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{live_transport_rows}

Every completed cell must have exactly 80 search plus 80 visit records, one
actual status-200 transport attempt per record, a minimum adjacent visit-start
gap of 2.98 s, and a 160/160/160/160 request/start/complete/commit broker
ledger with zero failure.  A recovered retry is rejected, not normalized away.

| Run | Joint cell | Target | Physical markers | Max current usage | Min fit/admit | Min actual admit | Budget-truncated samples | Fit=0 | Semantic required-field malformed | Controller fail-closed | Raw line interleavings | Tail `rescue` parse-clean | Strict parser-v2 clean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{live_telemetry_rows}

`Budget-truncated` counts samples where `fit_admit < waiting`: the configured
target budget actively limited how many waiting requests fit.  It is kept
separate from maximum current physical usage; usage below the target does not
imply that the controller was inactive, because committed and predicted tokens
also enter admission.

`Semantic required-field malformed=0` means that, after isolating a known
stdout concatenation suffix, every controller admission field and safety
equation was present and valid; it is not a claim that every raw line was
parse-clean.  `Raw line interleavings` counts marker lines whose terminal
`rescue=0` token was immediately concatenated with an API-server log prefix.
Those lines are SHA-bound and disclosed, but their tail token is not clean for
the repository's strict parser-v2.  Consequently any row with a nonzero count
must not be described as raw-malformed-free or strict-parser-v2 clean.

The replacement is explicitly post-hoc.  The partial r2 pilot is SHA-bound
only as excluded provenance and is never pooled into an effect:

| Replacement | Excluded r2 A 429s | Excluded r2 E(.85) 429s | Excluded r2 failed tool records | Bound r2 files |
|---|---:|---:|---:|---:|
{excluded_r2_rows}
"""
    else:
        live_sensitivity_section = """
## Completed comment-3 live sensitivity reanalysis

No completed live-sensitivity summary was supplied to this invocation.
"""
    if trace_centers:
        trace_center_rows = "\n".join(
            f'| {row["label"]} '
            f'| {row["cells"]["fcfs"]["task_e2e_s"]["mean"]:.4f}→'
            f'{row["cells"]["joint_target093"]["task_e2e_s"]["mean"]:.4f} '
            f'| {100.0 * row["relative_reduction"]["task_e2e_mean"]:+.3f}% '
            f'| {row["cells"]["fcfs"]["task_e2e_s"]["p95"]:.4f}→'
            f'{row["cells"]["joint_target093"]["task_e2e_s"]["p95"]:.4f} '
            f'| {100.0 * row["relative_reduction"]["task_e2e_p95"]:+.3f}% '
            f'| {100.0 * row["relative_reduction"]["mean_queue_time"]:+.3f}% '
            f'| {100.0 * row["relative_reduction"]["request_latency_mean"]:+.3f}% |'
            for row in trace_centers
        )
        trace_center_details = "\n".join(
            f'- `{row["label"]}`: 264/264 source-call request keys match per '
            f'cell; Joint has {row["physical_kv"]["post_baseline_sample_count"]} '
            f'post-baseline physical samples at target '
            f'`{row["physical_kv"]["target_utilization"]:.2f}`, '
            f'{row["physical_kv"]["malformed_sample_count"]} malformed and '
            f'{row["physical_kv"]["fail_closed_count"]} fail-closed.  Maximum '
            f'usage was only `{row["physical_kv"]["maximum_observed_usage"]:.3f}`, '
            'so the target was not binding.'
            for row in trace_centers
        )
        trace_center_section = f"""
## Real-trace central functional A/E (separate evidence tier)

This is a single real-trace center run on one Qwen/A100 shape.  It establishes
that FCFS A and physical-Joint E both execute the same 264 online-metadata
requests and gives a directional functional effect.  It is **not** a
replicated paper result, a `.85/.93/.97` target-sensitivity result, or
cross-model/cross-GPU evidence.

| Point | Task mean A→E (s) | Mean reduction | Task P95 A→E (s) | P95 reduction | Queue reduction | Request-mean reduction |
|---|---:|---:|---:|---:|---:|---:|
{trace_center_rows}

{trace_center_details}
"""
    else:
        trace_center_section = """
## Real-trace central functional A/E (separate evidence tier)

No real-trace center summary was supplied to this invocation.
"""
    return f"""# Co-Scheduler Specification and Robustness Audit

Date: 2026-08-30

## Bottom line

The Qwen reproduction now has an executable, exact decomposition of its
deployed Joint-v2 score and a deterministic 108-state sensitivity replay.  It
also has meaningful checked-in A100 load evidence.  These artifacts **do not
yet close the reviewer's cross-model/cross-GPU generalization concern**:
all GPU points re-extracted by this script use the same Qwen/A100 family, and
this command itself never launches a model server.  The CPU sweep tests
scheduler decisions under throughput/KV proxies, not model latency.

There is also a specification mismatch that should be fixed in the paper.  The
paper's abstract `ExposedToolGain / LLMPressure + Aging` and
`DecodeLoad + gamma * KVLoad` equations are not literal expressions in the
Qwen hook.  The registered formal path uses an additive cost plus independent
physical-KV admission.  It would be inaccurate to claim that the code directly
implements the paper variables under those names.

## Exact deployed policy

Candidates are lower-is-better.  With prefix locality disabled in the formal
configuration, the continuous score is:

```text
C_i = P_llm,i - G_tool,i - G_progress,i - A_i

service_i = pt_i / prefill_rate + po_i / decode_rate
G_tool,i = beta * confidence_i * min(next_tool_wait_i, 80 s)
           / (1 + projected_KV_pressure_i * pt_i / context_ref)
A_i = aging_alpha * scheduler_wait_i
```

`P_llm` is the sum of service time, a nonlinear context/KV contention term,
task-tail cost, any over-budget penalty, and a cold-session penalty.
`G_progress` contains final-call and reciprocal-progress bonuses.  In the
registered configuration, a final-call lane and an exact remaining-call lane
are lexicographic keys *before* the continuous score.  Thus this is not a
literal gain/pressure ratio.

The audit reimplements those terms only for observability, then compares the
sum against production `_joint_v2_score_s`.  Maximum absolute disagreement
across all replayed candidate states is `{max_error:.3e}` seconds.

### Reviewer term to implementation mapping

| Reviewer term | What this Qwen implementation actually uses | Active in formal physical-KV path? |
|---|---|---|
| ExposedToolGain | execution-aware `nwc * min(nw, 80)` bonus, damped by prompt length and projected logical KV pressure | yes, as a surrogate |
| LLMPressure | additive service/context/task-tail/over-budget/cold-session cost in seconds | yes, as a surrogate |
| DecodeLoad | engine running-request count relative to configured target/max | **no**; physical-KV admission bypasses the decode-band helper |
| KVLoad | logical projected tokens for ranking; physical block usage and forecast footprint for admission | yes |
| pressure band | legacy `.82–1.02` HBM controller | **no**; formal uses physical target utilization `.93` |
| gamma | no literal Joint-v2 gamma; nearest non-equivalent knob is context alpha `1.4` with pressure exponent `1.35` | no literal gamma |
| aging | `0.2 * wait_seconds`, plus a 40-second physical rescue deadline | yes |
| speculation budget | 4 global workers, max 2 speculative, visit cap 2, pending cap 128, TTL 120 s, min reservation 0 | tool-side broker |

The live agent records `nrg/nps` and global queue counts for evidence, but the
Joint-v2 score directly reads `nw/nwc/rtw`; it does not directly read the
completed-ready flag or global broker queue counts.  In particular, a completed
prediction has estimated remaining wait zero, so this checkout should not
claim a fully realized-gain feedback term without adding and validating that
side channel.

## Parameter selection evidence

- Prefill/decode rates (`38112` and `113.7` token/s) are calibration constants,
  not universal model/GPU constants.
- The active physical-KV target is `0.93`; predicted footprints are rounded to
  native block size and one request older than 40 seconds may consume the 7%
  reserve, but never cross 100% physical capacity.
- The formal aging coefficient is `0.2`, so 40 seconds of scheduler wait lowers
  continuous cost by 8 seconds.  The 40-second rescue is the hard progress
  mechanism under physical admission.
- The legacy low/high pressure-band sweep produced the same physical admit
  count(s) `{legacy_band_admits}`, confirming that those variables are inactive
  in this path.  In contrast, the active utilization sweep changed admit counts
  as follows: `{utilization_admits}`.
- The tool budget was bounded structurally at two speculative workers because
  there are four global workers and visit capacity is two.  Development F1
  (`min_speculative_workers=1`) improved only `0.2079%` over F0, only `6/16`
  sources were faster, and the bootstrap interval crossed zero; therefore F0
  (`min=0`) was frozen.  `max=2` and pending cap `128` were not independently
  hardware-swept and should not be described as universally optimal.

The single-factor replay below holds the registered A100-shape proxy, mixed
workload, `1x` context, and `0.70` load fixed.  “Full” includes lexicographic
stage lanes; “continuous” isolates the additive score.  This distinction is
why some large coefficient changes alter the continuous ranking while the
full-policy ranking remains unchanged.

| Parameter | Value | Full pairwise agreement | Continuous agreement | Physical admits |
|---|---:|---:|---:|---:|
{parameter_rows}

## Checked-in A100 load evidence

These points are re-extracted from checked-in JSON/reports.  They use different
development configurations, so they are evidence of sensitivity, not one
causal load curve.  Positive reduction is faster under the indicated Joint
bundle.

| Point | Offered sessions | Controller/config | Mean-task reduction | Task-P95 reduction | Evidence role |
|---|---:|---|---:|---:|---|
{empirical_rows}

The current physical-KV controller is directionally consistent at 240 and 300
offered sessions (`25.385%` and `23.852%` mean-task reductions), but each is a
single development screen.  Both improve task P95, while their source reports
also disclose request-P95 regressions versus FCFS.  Earlier count-target
results range from regressions to large gains.  This is exactly why a frozen
configuration and fresh cross-hardware matrix are required.

{trace_center_section}

{external_section}

{live_sensitivity_section}

## CPU policy replay

The sweep crosses three throughput/KV proxy profiles, four tool/LLM workload
mixes, three context scales (`0.5x/1x/2x`), and three physical-load ratios
(`0.35/0.70/0.90`): 108 states, each with 18 waiting candidates.  Agreement is
against the registered A100-shape proxy for the same workload/context/load.

| Proxy profile | States | Full-policy pairwise agreement | Continuous-score agreement | Worst continuous agreement | Physical admits/state |
|---|---:|---:|---:|---:|---:|
{proxy_rows}

Full-policy agreement is partly protected by hard stage lanes.  Continuous
agreement is the more informative parameter-sensitivity measure.  Neither
number is an E2E latency result, and naming a proxy `small` or `large` does not
associate it with a measured GPU SKU.

![Sensitivity summary](sensitivity.svg)

## What can and cannot be said to the reviewer

Supported now:

1. Every active Qwen scheduling term, unit, default, metadata source, and
   physical-admission rule can be specified exactly.
2. The score decomposition is numerically identical to production code.
3. Physical-KV admission adapts to native block geometry rather than a fixed
   token capacity, and checked-in stress240/300 results are positive on mean
   task latency.
4. Aging/rescue and speculative worker caps are bounded and auditable.
5. The completed `12k/80` high pair is positive on mean task latency and task
   P95 in one SHA-bound development run, with the failed shape prefix excluded.

Not supported now:

1. Cross-model or cross-GPU E2E generalization.
2. A claim that `.82–1.02` or a literal gamma controls the registered formal
   physical-KV experiment.
3. Universal optimality of `aging=.2`, target utilization `.93`, or max two
   speculative workers.
4. A fully implemented realized-completed-tool-gain side channel.
5. Raw-malformed-free or strict-parser-v2-clean physical telemetry for the
   target `.85/.93` and high `.93` cells whose server logs contain disclosed
   stdout line interleavings.

To close the remaining concern, rerun a preregistered matrix with at least two
model families and two GPU memory/throughput shapes.  Calibrate only rates and
physical capacity per deployment; keep dimensionless policy values frozen,
and report mean/task-P95/request-P95 plus starvation and admission telemetry.

## Live reviewer-follow-up runner

`run_scheduler_live_sensitivity.py` is a post-hoc, development-only bridge for
the hardware currently available.  It uses the byte-identical frozen 80-source
workload and a fresh four-GPU server per cell.  Its target suite is one FCFS A
cell plus three Joint E cells at physical-KV targets `.85/.93/.97`; the
invariant checker verifies that only this active target key changes among E.
The completed r3 replacement uses a common 3.0-second visit-start gate and
rejects every retry (including a recovered 429) in every A/E cell.  It is a
transport remediation after the excluded r2 pilot, not scheduler tuning.
Only the bounded `target` and `high` suites are executable.  The historical
six-cell `comment3-shape-r1` run is immutable failed evidence under its bound
old runner SHA: cell 5 deterministically rejected formal order index 4 before
issuing any request.  Its four observed prefix cells are excluded and cannot
be resumed, reused, or pooled.  The `high` suite is the one-shot replacement:
one source-identical A/E pair at `12k/80`, fixed A→E order indices 0/1, with
the old and replacement cell configs required to be byte-equivalent after
normalizing only run/block/order/server identity.  This is context/load
robustness on one model/GPU family, not cross-model or cross-GPU proof.

```bash
/home/aiscuser/.conda/envs/paste/bin/python \\
  reproduction/scripts/run_scheduler_live_sensitivity.py \\
  comment3-target-r3 --suite target --gpus 4,5,6,7 --port 8100 --check-only

# This records the completed r3 preflight; its tag/artifacts are immutable.

/home/aiscuser/.conda/envs/paste/bin/python \\
  reproduction/scripts/run_scheduler_live_sensitivity.py \\
  comment3-high-r1 --suite high --gpus 0,1,2,3 --port 8000 --check-only

# Check-only for the bounded high-pair replacement; it neither resumes nor
# writes into comment3-shape-r1.
```

The checked-in historical task phase took roughly `197–237 s` per cell.  The
runner budgets `6–12 min` per cell including fresh model load, shutdown, and
live-HTTP variance: `24–48 min` for target (4 cells) or `12–24 min` for high
(2 cells).  These are planning estimates, not newly measured durations.

## Reproduction

```bash
python3 reproduction/scripts/run_scheduler_robustness.py \\
  --output-dir reproduction/results/scheduler_robustness \\
  --trace-center-summary \\
    reproduction/artifacts/reviewer_comment3_live/center093/summary.json \\
  --live-sensitivity-summary \\
    reproduction/artifacts/live_joint/development/comment3_scheduler/comment3-target-r3/summary.json \\
  --live-sensitivity-summary \\
    reproduction/artifacts/live_joint/development/comment3_scheduler/comment3-high-r1/summary.json

python3 -m unittest \\
  reproduction.tests.test_scheduler_robustness \\
  reproduction.tests.test_scheduler_live_sensitivity
```

Outputs:

- `raw_results.json`: complete states, score components, parameter sweep, and
  source hashes;
- `sensitivity.csv`: flat 108-state table;
- `sensitivity.svg`: historical evidence and proxy decision summary;
- this report.

Use `--live-aggregate reproduction/.../strict_four_cell_aggregate.json` to
bind and re-extract a newly completed strict GPU aggregate.
Use `--live-sensitivity-summary reproduction/.../summary.json` to bind the
new comment-3 plan, completion record, and live summary together.
Use `--trace-center-summary reproduction/.../center093/summary.json` for the
separate real-trace functional center evidence tier.
"""


def run(
    output_dir: Path,
    live_aggregates: Sequence[Path] = (),
    live_sensitivity_summaries: Sequence[Path] = (),
    trace_center_summaries: Sequence[Path] = (),
) -> dict[str, Any]:
    _validate_reference_bindings()
    hook = _load_hook()
    factorial = _factorial_sweep(hook)
    parameters = _parameter_sweep(hook)
    empirical = _empirical_evidence()
    proxy_summary = _summarize_proxy(factorial)
    external_live_aggregates = [
        _external_live_aggregate(path) for path in live_aggregates
    ]
    external_live_sensitivity_summaries = [
        _external_live_sensitivity_summary(path)
        for path in live_sensitivity_summaries
    ]
    live_run_tags = [
        str(row["run_tag"]) for row in external_live_sensitivity_summaries
    ]
    if len(live_run_tags) != len(set(live_run_tags)):
        raise ValueError("duplicate live sensitivity run tag supplied")
    if external_live_sensitivity_summaries:
        if len(
            {
                str(row["workload_sha256"])
                for row in external_live_sensitivity_summaries
            }
        ) != 1:
            raise ValueError("live sensitivity summaries use different workloads")
        if len(
            {
                str(row["model_id"])
                for row in external_live_sensitivity_summaries
            }
        ) != 1:
            raise ValueError("live sensitivity summaries use different models")
    external_trace_center_results = [
        _external_trace_center_summary(path) for path in trace_center_summaries
    ]
    max_formula_error = max(
        [float(row["max_formula_error_s"]) for row in factorial]
        + [float(row["max_formula_error_s"]) for row in parameters]
    )
    payload = {
        "schema": "paste_repro.scheduler_robustness",
        "version": 1,
        "generated_date_utc": "2026-08-30",
        "evidence_boundary": {
            "gpu_executed_by_this_script": False,
            "network_used_by_this_script": False,
            "proxy_results_are_latency_measurements": False,
            "empirical_results_are_reextracted_checked_in_a100_evidence": True,
            "external_live_aggregates_are_reanalysis_not_script_execution": True,
            "external_live_aggregate_count": len(external_live_aggregates),
            "external_live_sensitivity_summary_count": len(
                external_live_sensitivity_summaries
            ),
            "external_trace_center_result_count": len(
                external_trace_center_results
            ),
            "cross_model_or_gpu_generalization_proven": False,
        },
        "source_bindings": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
            for path in (
                ROBUSTNESS_RUNNER_PATH,
                LIVE_SENSITIVITY_RUNNER_PATH,
                HOOK_PATH,
                LIVE_AGENT_PATH,
                FORMAL_RUNNER_PATH,
                FINAL_REPORT_PATH,
            )
        },
        "reference_environment": REFERENCE_ENV,
        "implementation_mapping": _static_mapping(),
        "verification": {
            "formula_checked_against": "_joint_v2_score_s",
            "max_formula_error_s": max_formula_error,
            "formula_equivalent_within_1e_9": max_formula_error <= 1e-9,
            "factorial_state_count": len(factorial),
            "candidate_evaluations": sum(
                int(row["candidate_count"]) for row in factorial
            ),
        },
        "proxy_profiles": [profile.__dict__ for profile in PROXY_PROFILES],
        "proxy_summary": proxy_summary,
        "factorial_states": factorial,
        "parameter_sensitivity": parameters,
        "empirical_a100_evidence": empirical,
        "external_live_aggregates": external_live_aggregates,
        "external_live_sensitivity_summaries": (
            external_live_sensitivity_summaries
        ),
        "external_trace_center_results": external_trace_center_results,
    }
    if not payload["verification"]["formula_equivalent_within_1e_9"]:
        raise RuntimeError(
            f"score decomposition drift: max error={max_formula_error}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.json"
    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "sensitivity.csv", factorial)
    _write_svg(output_dir / "sensitivity.svg", empirical, proxy_summary)
    (output_dir / "REPORT.md").write_text(
        _render_report(payload), encoding="utf-8"
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPOSITORY_ROOT / "reproduction/results/scheduler_robustness"
        ),
    )
    parser.add_argument(
        "--live-aggregate",
        action="append",
        default=[],
        type=Path,
        help=(
            "Repeatable strict four-cell aggregate to SHA-bind and re-extract; "
            "this never launches its GPU experiment."
        ),
    )
    parser.add_argument(
        "--live-sensitivity-summary",
        action="append",
        default=[],
        type=Path,
        help=(
            "Repeatable completed comment-3 summary; its sibling plan and "
            "completion record are also SHA-validated."
        ),
    )
    parser.add_argument(
        "--trace-center-summary",
        action="append",
        default=[],
        type=Path,
        help=(
            "Repeatable real-trace functional A/E center summary; both cell "
            "results, prepared workloads, request JSONL, and server logs are "
            "validated and SHA-bound."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(
        args.output_dir.resolve(),
        args.live_aggregate,
        args.live_sensitivity_summary,
        args.trace_center_summary,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "factorial_state_count": payload["verification"][
                    "factorial_state_count"
                ],
                "candidate_evaluations": payload["verification"][
                    "candidate_evaluations"
                ],
                "max_formula_error_s": payload["verification"][
                    "max_formula_error_s"
                ],
                "cross_model_or_gpu_generalization_proven": payload[
                    "evidence_boundary"
                ]["cross_model_or_gpu_generalization_proven"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
