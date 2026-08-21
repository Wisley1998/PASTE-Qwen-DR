"""Monkey-patch vLLM request scheduling order.

The patch is intentionally narrow: by default it only reorders requests that
are still in the scheduler waiting queue before vLLM admits new prefills.  An
explicit Joint-v2/native-admission opt-in may also reorder the v1 FCFS running
list once, immediately before vLLM schedules it.  Chunked-prefill budgeting,
prefix-cache allocation, and the actual preemption operation remain vLLM's.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import deque
from functools import wraps
from typing import Any, Callable, Iterable, Optional


_META_RE = re.compile(r"schedx([0-9a-f]+)z")
_META_CACHE: dict[str, dict[str, Any]] = {}

# State for tool-return-aware policies.
# Maps trace_id (str) -> (expected_return_wall_time_s, kv_tokens_to_reserve).
# Populated when a request leaves the running set (entering tool wait in the driver)
# and cleared when its next turn re-appears in waiting/running.
_pending_returns: dict[str, tuple[float, int]] = {}
_prev_running_ids: set[str] = set()
_oracle_last_log_s: float = 0.0
_v2_started_sessions: set[str] = set()
_v2_completed_sessions: set[str] = set()

_SUPPORTED_POLICIES = {
    "fcfs",
    "sjf",
    "sjf_aging",
    "srpt",
    "srpt_aging",
    "ljf",
    "random",
    "oracle_next",
    "oracle_short_tail",
    "oracle_long_tail",
    "oracle_last",
    "oracle_turns",
    "oracle_critical",
    "oracle_next_long",
    "oracle_next_long_aging",
    "oracle_task_srpt",
    "oracle_task_srpt_aging",
    "oracle_service_srpt",
    "oracle_service_srpt_aging",
    "oracle_wspt",
    "oracle_wspt_aging",
    "oracle_final_wspt",
    "oracle_overlap_srpt",
    "oracle_overlap_srpt_aging",
    "online_critical",
    "online_overlap_srpt_aging",
    "online_overlap_srpt_aging_v2",
    "online_oas_v3_no_nw",
    "online_oas_v3_nw_bonus",
    "online_oas_v3_nw_delay",
    "online_oas_v3_g025_nw_bonus",
    "online_oas_v3_g025_nw_delay",
    "online_oas_v3_g050_nw_bonus",
    "online_oas_v3_g050_nw_delay",
    "online_oas_v3_g075_nw_delay",
    "online_oas_v4",
    "online_oas_v5_tool_hbm",
    "online_hbm_controller",
    "online_tool_queue",
    "online_hbm_tool_split",
    "online_joint_pacer_v1",
    "online_joint_pacer_v2",
    "oracle_tool_return_admission",
}


def _prefill_tokens_per_s_v2() -> float:
    raw = os.getenv("VLLM_SCHED_PREFILL_TOKENS_PER_S_V2", "38112")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 38112.0


def _decode_tokens_per_s_v2() -> float:
    raw = os.getenv("VLLM_SCHED_DECODE_TOKENS_PER_S_V2", "113.7")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 113.7


def _default_predicted_output_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_DEFAULT_PRED_OUT", "722")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 722.0


def _oas_v3_context_tokens_per_s() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S", "6000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 6000.0


def _oas_v3_context_gamma() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V3_CONTEXT_GAMMA", "1.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def _oas_v4_context_gamma() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_CONTEXT_GAMMA", "0.5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.5


def _oas_v4_target_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_TARGET_CONTEXT_TOKENS", "900000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 900000.0


def _oas_v4_contention_alpha() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_CONTENTION_ALPHA", "1.5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.5


def _oas_v4_pressure_power() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_PRESSURE_POWER", "2.0")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 2.0


def _oas_v4_long_prompt_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_LONG_PROMPT_TOKENS", "64000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 64000.0


def _oas_v4_medium_bucket_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_MEDIUM_BUCKET_TOKENS", "32000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 32000.0


def _oas_v4_long_bucket_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_LONG_BUCKET_TOKENS", "96000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 96000.0


def _oas_v4_bucket_pattern() -> tuple[str, ...]:
    raw = os.getenv(
        "VLLM_SCHED_OAS_V4_BUCKET_PATTERN",
        "short,medium,short,long,medium,short",
    )
    items = tuple(
        item.strip().lower()
        for item in raw.split(",")
        if item.strip().lower() in {"short", "medium", "long"}
    )
    return items or ("short", "medium", "short", "long", "medium", "short")


def _oas_v4_nw_damp() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V4_NW_DAMP", "1.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def _oas_v5_target_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_TARGET_CONTEXT_TOKENS", "760000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 760000.0


def _oas_v5_hbm_alpha() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_HBM_ALPHA", "2.5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 2.5


def _oas_v5_hbm_power() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_HBM_POWER", "2.0")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 2.0


def _oas_v5_over_budget_penalty_s() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_OVER_BUDGET_PENALTY_S", "300")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 300.0


def _oas_v5_tool_beta() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_TOOL_BETA", "0.9")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.9


def _oas_v5_tool_wait_cap_s() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_TOOL_WAIT_CAP_S", "80")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 80.0


def _oas_v5_tool_damp() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_TOOL_DAMP", "1.25")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.25


def _oas_v5_long_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_LONG_CONTEXT_TOKENS", "96000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 96000.0


def _oas_v5_medium_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_MEDIUM_CONTEXT_TOKENS", "32000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 32000.0


def _oas_v5_max_long_running() -> int:
    raw = os.getenv("VLLM_SCHED_OAS_V5_MAX_LONG_RUNNING", "2")
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _oas_v5_final_bonus_s() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_FINAL_BONUS_S", "8")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 8.0


def _oas_v5_virtual_fill_ratio() -> float:
    raw = os.getenv("VLLM_SCHED_OAS_V5_VIRTUAL_FILL_RATIO", "0.92")
    try:
        return min(1.0, max(0.05, float(raw)))
    except ValueError:
        return 0.92


def _hbm_target_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS", "1050000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 1050000.0


def _hbm_min_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS", "760000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 760000.0


def _hbm_max_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS", "1350000")
    try:
        return max(_hbm_min_context_tokens(), float(raw))
    except ValueError:
        return 1350000.0


def _hbm_low_pressure() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_LOW_PRESSURE", "0.82")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.82


def _hbm_high_pressure() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_HIGH_PRESSURE", "1.02")
    try:
        return max(_hbm_low_pressure(), float(raw))
    except ValueError:
        return 1.02


def _hbm_budget_increase() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_BUDGET_INCREASE", "1.02")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 1.02


def _hbm_budget_decrease() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_BUDGET_DECREASE", "0.97")
    try:
        return min(1.0, max(0.5, float(raw)))
    except ValueError:
        return 0.97


def _hbm_control_interval_s() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_CONTROL_INTERVAL_S", "5")
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 5.0


def _hbm_virtual_fill_ratio() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO", "0.96")
    try:
        return min(1.0, max(0.05, float(raw)))
    except ValueError:
        return 0.96


def _hbm_long_context_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS", "96000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 96000.0


def _hbm_max_long_running() -> int:
    raw = os.getenv("VLLM_SCHED_HBM_MAX_LONG_RUNNING", "3")
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def _hbm_max_admit_per_step() -> int:
    raw = os.getenv("VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP", "16")
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def _hbm_min_running_reqs() -> int:
    raw = os.getenv("VLLM_SCHED_HBM_MIN_RUNNING_REQS", "24")
    try:
        return max(1, int(raw))
    except ValueError:
        return 24


def _tool_queue_beta() -> float:
    raw = os.getenv("VLLM_SCHED_TOOL_QUEUE_BETA", "0.7")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.7


def _tool_queue_wait_cap_s() -> float:
    raw = os.getenv("VLLM_SCHED_TOOL_QUEUE_WAIT_CAP_S", "90")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 90.0


def _tool_queue_final_bonus_s() -> float:
    raw = os.getenv("VLLM_SCHED_TOOL_QUEUE_FINAL_BONUS_S", "12")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 12.0


def _tool_queue_progress_bonus_s() -> float:
    raw = os.getenv("VLLM_SCHED_TOOL_QUEUE_PROGRESS_BONUS_S", "4")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 4.0


def _joint_return_window_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_RETURN_WINDOW_S", "5.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def _joint_reserve_kv_scale() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_RESERVE_KV_SCALE", "1.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def _joint_reserve_slot_scale() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_RESERVE_SLOT_SCALE", "0.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _joint_max_reserved_slots() -> int:
    raw = os.getenv("VLLM_SCHED_JOINT_MAX_RESERVED_SLOTS", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _joint_quick_return_window_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_QUICK_RETURN_WINDOW_S", "8.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 8.0


def _joint_quick_return_penalty_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_QUICK_RETURN_PENALTY_S", "2.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 2.0


def _joint_extra_final_bonus_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_EXTRA_FINAL_BONUS_S", "8.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 8.0


def _joint_extra_progress_bonus_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_EXTRA_PROGRESS_BONUS_S", "4.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 4.0


def _joint_v2_foreground_max_sessions() -> int:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS", "220")
    try:
        return max(1, int(raw))
    except ValueError:
        return 220


def _joint_v2_gate_min_running() -> int:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING", "96")
    try:
        return max(1, int(raw))
    except ValueError:
        return 96


def _joint_v2_gate_max_wait_s() -> float:
    """Deadline before a waiting request becomes fairness-eligible.

    For a cold session this permits the oldest eligible request to bypass the
    foreground gate.  HBM and native engine capacity still apply, so this is
    an eligibility deadline rather than a wall-clock start-time guarantee.
    Zero or a negative value disables the escape.
    """
    raw = os.getenv("VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S", "6")
    try:
        value = float(raw)
    except ValueError:
        return 6.0
    if not math.isfinite(value):
        return 6.0
    return value if value > 0.0 else 0.0


def _joint_v2_deadline_min_running() -> int:
    """Running-load floor for pinning one deadline-expired request.

    The historical behavior only pins at the configured decode target.  Keep
    that behavior when this override is unset, while allowing a deployment to
    make the deadline effective on the admission tick immediately below the
    target (for example, 63 for a target of 64).
    """

    target = _joint_v2_decode_target_running()
    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING",
        str(target),
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return target


def _joint_v2_final_lane_enabled() -> bool:
    return os.getenv("VLLM_SCHED_JOINT_V2_FINAL_LANE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _joint_v2_remaining_call_lane_enabled() -> bool:
    return os.getenv(
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _joint_v2_remaining_call_coarse_lanes_enabled() -> bool:
    """Bucket non-final remaining calls instead of using exact stages.

    This only affects the remaining-call lane when that lane is enabled.  The
    separate final-call lane remains responsible for distinguishing ``rc=0``;
    the coarse remaining key groups ``rc<=2`` and ``rc>=3`` so the continuous
    Joint score can order requests within each broad task stage.
    """

    return os.getenv(
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _joint_v2_remaining_call_soft_weight_s() -> float:
    """Seconds of continuous cost per causally predicted remaining call.

    A positive value selects the soft-stage formulation in place of the exact
    or coarse remaining-call sort lane.  Zero is deliberately the default so
    existing profiles retain byte-for-byte-equivalent ordering keys.
    """

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S", "0"
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


# The online predictor used by this hook has an eight-call forecast horizon in
# the stress workload.  Missing or malformed metadata must not look like a
# final call, but an unbounded sentinel would silently recreate a hard lane.
_JOINT_V2_UNKNOWN_REMAINING_CALLS = 8


def _joint_v2_soft_remaining_calls(meta: dict[str, Any]) -> int:
    """Return a finite, conservative stage estimate for the soft cost."""

    raw = meta.get("rc")
    try:
        remaining_calls = int(raw)
    except (TypeError, ValueError, OverflowError):
        return _JOINT_V2_UNKNOWN_REMAINING_CALLS
    if remaining_calls < 0:
        return _JOINT_V2_UNKNOWN_REMAINING_CALLS
    return remaining_calls


def _joint_v2_native_admission_enabled() -> bool:
    """Leave the engine's native running-request capacity unchanged.

    This is deliberately opt-in so existing Joint v2 configurations preserve
    their HBM/decode admission behavior.  When enabled, Joint still reorders
    the waiting queue, but vLLM alone decides how many requests to admit.
    """

    return os.getenv(
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _joint_v2_physical_kv_admission_enabled() -> bool:
    """Opt into forecast-aware admission against vLLM's physical KV cache.

    This mode is intentionally independent of ``NATIVE_ADMISSION`` and is
    disabled by default.  If both flags are set, native admission wins: the
    call sites never invoke the capacity-mutating controller, preserving the
    documented reorder-only behavior of ``NATIVE_ADMISSION=1``.
    """

    return os.getenv(
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _joint_v2_physical_kv_target_utilization() -> float:
    """Fraction of profiled physical KV tokens available to forecasts."""

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION", "0.93"
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.93
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        return 0.93
    return value


def _joint_v2_physical_kv_rescue_wait_s() -> float:
    """Age at which one request may consume the utilization reserve."""

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S", "120"
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 120.0
    if not math.isfinite(value):
        return 120.0
    return value if value > 0.0 else 0.0


def _joint_v2_physical_kv_log_interval_s() -> float:
    """Minimum interval between structured admission evidence records."""

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S", "1"
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return max(0.0, value)


def _joint_v2_prefix_locality_enabled() -> bool:
    """Enable a causal, local-prefix-cache signal for Joint v2 ordering.

    The signal is deliberately opt-in.  It is currently supported only by
    the vLLM v1 wrapper, where the live KV-cache coordinator can be queried
    before native admission without allocating or touching cache blocks.
    """

    return os.getenv(
        "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _joint_v2_prefix_locality_weight() -> float:
    """Strength of the cached-prefill discount in the ordering score."""

    raw = os.getenv("VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_WEIGHT", "1")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return min(4.0, max(0.0, value))


def _joint_v2_prefix_locality_log_interval_s() -> float:
    """Minimum interval between structured prefix-locality records."""

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_LOG_INTERVAL_S", "1"
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return max(0.0, value)


def _joint_v2_prefix_locality_refresh_s() -> float:
    """Age limit for a causal per-request prefix snapshot.

    Prefix lookup walks a request's block-hash chain.  Repeating that walk for
    every queued request on every decode tick can dominate scheduler CPU time,
    so a short scheduler-local cache bounds the observer overhead.  Cached
    observations contain only past state, and stale hits can affect ordering
    but never physical admission or native allocation safety.  Set this to
    zero for an exact lookup on every tick.
    """

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_REFRESH_S", "0.25"
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.25
    if not math.isfinite(value):
        return 0.25
    return max(0.0, value)


def _joint_v2_running_priority_enabled() -> bool:
    """Enable the v1 FCFS running-order experiment.

    This is separate from native admission and defaults off.  The call site
    additionally requires Joint v2, native admission, a non-empty waiting
    queue, and vLLM's underlying FCFS policy.
    """

    return os.getenv(
        "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _joint_v2_running_priority_max_wait_s() -> float:
    """Age at which one running request receives a fairness pin.

    Zero disables the pin.  Only the single oldest expired request is pinned,
    so a saturated set of expired requests does not collapse back to FCFS.
    """

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S", "0"
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _joint_v2_decode_target_running() -> int:
    """Preferred number of engine-running requests (a decode-load proxy).

    The foreground-session limit is a conservative default for existing v2
    configurations.  The value is clamped to vLLM's native capacity at the
    admission-control call site.
    """

    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING",
        str(_joint_v2_foreground_max_sessions()),
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return _joint_v2_foreground_max_sessions()


def _joint_v2_decode_max_running() -> int:
    """Configured decode-pressure ceiling, with one fairness lane by default."""

    target = _joint_v2_decode_target_running()
    raw = os.getenv(
        "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING",
        str(target + 1),
    )
    try:
        return max(target, int(raw))
    except ValueError:
        return target + 1


def _joint_v2_tail_beta() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_TAIL_BETA", "0.25")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.25


def _joint_v2_tool_beta() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_TOOL_BETA", "0.45")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.45


def _joint_v2_tool_wait_cap_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S", "80")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 80.0


def _joint_v2_remaining_tool_weight() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT", "0.35")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.35


def _joint_v2_context_alpha() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA", "1.4")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.4


def _joint_v2_context_ref_tokens() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS", "16000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 16000.0


def _joint_v2_final_bonus_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_FINAL_BONUS_S", "28")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 28.0


def _joint_v2_progress_bonus_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S", "18")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 18.0


def _joint_v2_new_session_penalty_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S", "8")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 8.0


def _joint_v2_over_budget_penalty_s() -> float:
    raw = os.getenv("VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S", "240")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 240.0


def _policy() -> str:
    return os.getenv("VLLM_SCHED_POLICY", "fcfs").strip().lower() or "fcfs"


def _aging_alpha() -> float:
    raw = os.getenv("VLLM_SCHED_AGING_ALPHA", "100")
    try:
        return float(raw)
    except ValueError:
        return 100.0


def _time_aging_alpha() -> float:
    raw = os.getenv("VLLM_SCHED_TIME_AGING_ALPHA", "0.05")
    try:
        return float(raw)
    except ValueError:
        return 0.05


def _avg_call_service_s() -> float:
    raw = os.getenv("VLLM_SCHED_AVG_CALL_SERVICE_S", "25")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 25.0


def _prefill_tokens_per_s() -> float:
    raw = os.getenv("VLLM_SCHED_PREFILL_TOKENS_PER_S", "6000")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 6000.0


def _decode_tokens_per_s() -> float:
    raw = os.getenv("VLLM_SCHED_DECODE_TOKENS_PER_S", "450")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 450.0


def _progress_weight() -> float:
    raw = os.getenv("VLLM_SCHED_PROGRESS_WEIGHT", "4")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 4.0


def _final_weight() -> float:
    raw = os.getenv("VLLM_SCHED_FINAL_WEIGHT", "8")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 8.0


def _overlap_beta() -> float:
    raw = os.getenv("VLLM_SCHED_OVERLAP_BETA", "0.5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.5


def _request_id(obj: Any) -> str:
    return str(getattr(obj, "request_id", ""))


def _arrival_time(obj: Any) -> float:
    value = getattr(obj, "arrival_time", None)
    if value is None:
        metrics = getattr(obj, "metrics", None)
        value = getattr(metrics, "arrival_time", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sampling_max_tokens(obj: Any) -> int:
    value = getattr(obj, "max_tokens", None)
    if value is None:
        sampling_params = getattr(obj, "sampling_params", None)
        value = getattr(sampling_params, "max_tokens", None)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _prompt_len_v0(seq_group: Any) -> int:
    prompt_token_ids = getattr(seq_group, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)

    try:
        seqs = seq_group.get_seqs()
    except Exception:
        seqs = []
    if not seqs:
        return 0

    seq = seqs[0]
    get_prompt_len = getattr(seq, "get_prompt_len", None)
    if callable(get_prompt_len):
        try:
            return int(get_prompt_len())
        except Exception:
            pass

    for candidate in (
        getattr(seq, "prompt_token_ids", None),
        getattr(getattr(seq, "data", None), "prompt_token_ids", None),
    ):
        if candidate is not None:
            return len(candidate)

    get_len = getattr(seq, "get_len", None)
    if callable(get_len):
        try:
            return int(get_len())
        except Exception:
            pass
    return 0


def _prompt_len_v1(request: Any) -> int:
    value = getattr(request, "num_prompt_tokens", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    return len(prompt_token_ids) if prompt_token_ids is not None else 0


def _decode_meta(obj: Any) -> dict[str, Any]:
    rid = _request_id(obj)
    if rid in _META_CACHE:
        return _META_CACHE[rid]

    meta: dict[str, Any] = {}
    match = _META_RE.search(rid)
    if match:
        try:
            meta = json.loads(bytes.fromhex(match.group(1)).decode("utf-8"))
        except Exception:
            meta = {}
    _META_CACHE[rid] = meta
    return meta


def _meta_float(meta: dict[str, Any], key: str, default: float) -> float:
    value = meta.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _meta_int(meta: dict[str, Any], key: str, default: int) -> int:
    value = meta.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _next_tool_wait_reliability(meta: dict[str, Any]) -> float:
    """Return a bounded confidence gate, preserving legacy metadata behavior."""

    reliability = _meta_float(meta, "nwc", 1.0)
    if not math.isfinite(reliability):
        return 0.0
    return min(1.0, max(0.0, reliability))


def _stable_random(obj: Any) -> float:
    seed = os.getenv("VLLM_SCHED_RANDOM_SEED", "20260429")
    payload = f"{seed}:{_request_id(obj)}".encode("utf-8", errors="ignore")
    value = int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16)
    return value / float(2**64)


def _service_estimate_s(
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
) -> float:
    prompt_tokens = _meta_int(meta, "pt", prompt_len)
    output_tokens = _meta_int(meta, "mt", max_tokens)
    return (max(0, prompt_tokens) / _prefill_tokens_per_s()) + (
        max(0, output_tokens) / _decode_tokens_per_s()
    )


def _service_estimate_v2_s(
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
) -> float:
    po = _meta_float(meta, "po", -1.0)
    if po < 0:
        po = float(max_tokens) if max_tokens > 0 else _default_predicted_output_tokens()
    prompt_tokens = _meta_int(meta, "pt", prompt_len)
    return (
        max(0, prompt_tokens) / _prefill_tokens_per_s_v2()
        + max(0.0, po) / _decode_tokens_per_s_v2()
    )


def _oas_v3_context_gamma_for_policy(policy: str) -> float:
    if "_g025_" in policy:
        return 0.25
    if "_g050_" in policy:
        return 0.50
    if "_g075_" in policy:
        return 0.75
    return _oas_v3_context_gamma()


def _oas_v3_base_cost_s(
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
    policy: str,
) -> float:
    prompt_tokens = _meta_int(meta, "pt", prompt_len)
    isolated_service_s = _service_estimate_v2_s(meta, prompt_len, max_tokens)
    context_penalty_s = (
        _oas_v3_context_gamma_for_policy(policy)
        * max(0, prompt_tokens)
        / _oas_v3_context_tokens_per_s()
    )
    return isolated_service_s + context_penalty_s


def _active_context_tokens(
    obj: Any,
    prompt_len_fn: Callable[[Any], int],
) -> int:
    try:
        return max(0, int(getattr(obj, "num_tokens", 0) or 0))
    except (TypeError, ValueError):
        return max(0, prompt_len_fn(obj))


def _estimated_kv_tokens(
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
) -> int:
    prompt_tokens = _meta_int(meta, "pt", prompt_len)
    po = _meta_float(meta, "po", -1.0)
    if po < 0:
        po = float(max_tokens) if max_tokens > 0 else _default_predicted_output_tokens()
    return max(0, int(prompt_tokens + max(0.0, po)))


def _oas_v4_base_key_s(
    *,
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
    next_tool_wait: float,
    waited_s: float,
    context_pressure: float,
) -> float:
    prompt_tokens = _meta_int(meta, "pt", prompt_len)
    isolated_service_s = _service_estimate_v2_s(meta, prompt_len, max_tokens)
    prompt_cost_s = max(0, prompt_tokens) / _oas_v3_context_tokens_per_s()
    pressure = max(0.0, context_pressure)
    base_context_penalty_s = _oas_v4_context_gamma() * prompt_cost_s
    marginal_contention_s = (
        _oas_v4_contention_alpha()
        * (pressure ** _oas_v4_pressure_power())
        * prompt_cost_s
    )
    nw_denom = 1.0 + (
        _oas_v4_nw_damp()
        * pressure
        * max(0, prompt_tokens)
        / _oas_v4_long_prompt_tokens()
    )
    overlap_bonus_s = _overlap_beta() * max(0.0, next_tool_wait) / nw_denom
    return (
        isolated_service_s
        + base_context_penalty_s
        + marginal_contention_s
        - overlap_bonus_s
        - _time_aging_alpha() * waited_s
    )


def _oas_v5_score_s(
    *,
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
    next_tool_wait: float,
    waited_s: float,
    live_tokens: float,
    virtual_tokens: float,
    live_long_count: int,
    virtual_long_count: int,
) -> tuple[float, bool]:
    prompt_tokens = _meta_int(meta, "pt", prompt_len)
    kv_tokens = _estimated_kv_tokens(meta, prompt_len, max_tokens)
    target_tokens = _oas_v5_target_context_tokens()
    fill_target = target_tokens * _oas_v5_virtual_fill_ratio()
    projected_tokens = live_tokens + virtual_tokens + kv_tokens
    projected_pressure = max(0.0, projected_tokens / target_tokens)

    service_s = _service_estimate_v2_s(meta, prompt_len, max_tokens)
    prompt_cost_s = max(0, prompt_tokens) / _oas_v3_context_tokens_per_s()
    hbm_penalty_s = (
        _oas_v5_hbm_alpha()
        * (projected_pressure ** _oas_v5_hbm_power())
        * prompt_cost_s
    )

    is_long = prompt_tokens >= _oas_v5_long_context_tokens()
    long_over_cap = (
        is_long
        and _oas_v5_max_long_running() > 0
        and live_long_count + virtual_long_count >= _oas_v5_max_long_running()
    )
    token_over_budget = projected_tokens > fill_target
    over_budget = token_over_budget or long_over_cap

    over_penalty_s = 0.0
    if token_over_budget:
        over_ratio = max(0.0, projected_tokens / max(1.0, fill_target) - 1.0)
        over_penalty_s += _oas_v5_over_budget_penalty_s() * (1.0 + over_ratio) ** 2
    if long_over_cap:
        over_penalty_s += _oas_v5_over_budget_penalty_s()

    remaining_calls = _meta_int(meta, "rc", 10**9)
    capped_tool_wait_s = 0.0
    if remaining_calls > 0:
        capped_tool_wait_s = min(max(0.0, next_tool_wait), _oas_v5_tool_wait_cap_s())
    hbm_damp = 1.0 + (
        _oas_v5_tool_damp()
        * projected_pressure
        * max(0, prompt_tokens)
        / _oas_v5_long_context_tokens()
    )
    tool_overlap_bonus_s = _oas_v5_tool_beta() * capped_tool_wait_s / hbm_damp
    final_bonus_s = _oas_v5_final_bonus_s() if remaining_calls == 0 else 0.0

    aged_score_s = (
        service_s
        + hbm_penalty_s
        + over_penalty_s
        - tool_overlap_bonus_s
        - final_bonus_s
        - _time_aging_alpha() * waited_s
    )
    return aged_score_s, over_budget


def _tool_queue_key_s(
    *,
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
    next_tool_wait: float,
    waited_s: float,
) -> float:
    remaining_calls = _meta_int(meta, "rc", 10**9)
    service_s = _service_estimate_v2_s(meta, prompt_len, max_tokens)
    capped_tool_wait_s = (
        min(max(0.0, next_tool_wait), _tool_queue_wait_cap_s())
        if remaining_calls > 0 else 0.0
    )
    final_bonus_s = _tool_queue_final_bonus_s() if remaining_calls == 0 else 0.0
    progress_bonus_s = _tool_queue_progress_bonus_s() / float(max(1, remaining_calls + 1))
    return (
        service_s
        - _tool_queue_beta() * capped_tool_wait_s
        - final_bonus_s
        - progress_bonus_s
        - _time_aging_alpha() * waited_s
    )


def _joint_tool_queue_key_s(
    *,
    meta: dict[str, Any],
    prompt_len: int,
    max_tokens: int,
    next_tool_wait: float,
    waited_s: float,
) -> float:
    key_s = _tool_queue_key_s(
        meta=meta,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        next_tool_wait=next_tool_wait,
        waited_s=waited_s,
    )
    remaining_calls = _meta_int(meta, "rc", 10**9)
    window_s = _joint_quick_return_window_s()
    if remaining_calls > 0 and window_s > 0 and 0.0 <= next_tool_wait <= window_s:
        proximity = (window_s - next_tool_wait) / window_s
        key_s += _joint_quick_return_penalty_s() * (0.25 + 0.75 * proximity)
    if remaining_calls == 0:
        key_s -= _joint_extra_final_bonus_s()
    key_s -= _joint_extra_progress_bonus_s() / float(max(1, remaining_calls + 1))
    return key_s


def _hbm_feature(
    obj: Any,
    now_s: float,
    prompt_len_fn: Callable[[Any], int],
    *,
    joint_pacer: bool = False,
) -> dict[str, Any]:
    arrival = _arrival_time(obj)
    waited_s = max(0.0, now_s - arrival) if arrival > 0 else 0.0
    prompt_len = prompt_len_fn(obj)
    max_tokens = _sampling_max_tokens(obj)
    meta = _decode_meta(obj)
    prompt_tokens = _meta_int(meta, "pt", prompt_len)
    remaining_calls = _meta_int(meta, "rc", 10**9)
    next_tool_wait = _meta_float(
        meta,
        "nw",
        0.0 if remaining_calls == 0 else _tool_queue_wait_cap_s(),
    )
    return {
        "arrival": arrival,
        "rid": _request_id(obj),
        "waited_s": waited_s,
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
        "meta": meta,
        "prompt_tokens": prompt_tokens,
        "kv_tokens": _estimated_kv_tokens(meta, prompt_len, max_tokens),
        "next_tool_wait": next_tool_wait,
        "tool_key": (
            _joint_tool_queue_key_s if joint_pacer else _tool_queue_key_s
        )(
            meta=meta,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            next_tool_wait=next_tool_wait,
            waited_s=waited_s,
        ),
    }


def _hbm_cost_key(f: dict[str, Any]) -> tuple[Any, ...]:
    prompt_tokens = max(0, int(f["prompt_tokens"]))
    kv_tokens = max(0, int(f["kv_tokens"]))
    return (
        prompt_tokens >= _hbm_long_context_tokens(),
        kv_tokens,
        f["arrival"],
        f["rid"],
    )


def _get_hbm_budget(self: Any, live_tokens: float, waiting_count: int, running_count: int) -> float:
    budget = getattr(self, "_oas_hbm_context_budget", None)
    if budget is None:
        budget = _hbm_target_context_tokens()
        setattr(self, "_oas_hbm_context_budget", budget)
        setattr(self, "_oas_hbm_last_update_s", 0.0)

    now_s = time.time()
    last_update_s = float(getattr(self, "_oas_hbm_last_update_s", 0.0) or 0.0)
    if now_s - last_update_s < _hbm_control_interval_s():
        return float(budget)

    pressure = live_tokens / max(1.0, float(budget))
    if pressure > _hbm_high_pressure():
        budget *= _hbm_budget_decrease()
    elif waiting_count > 0 and running_count >= _hbm_min_running_reqs() and pressure < _hbm_low_pressure():
        budget *= _hbm_budget_increase()
    elif waiting_count > _hbm_max_admit_per_step() and pressure < 0.95:
        budget *= min(_hbm_budget_increase(), 1.01)

    budget = min(_hbm_max_context_tokens(), max(_hbm_min_context_tokens(), float(budget)))
    setattr(self, "_oas_hbm_context_budget", budget)
    setattr(self, "_oas_hbm_last_update_s", now_s)
    return float(budget)


def _order_tool_queue_waiting(
    *,
    waiting_items: Iterable[Any],
    now_s: float,
    prompt_len_fn: Callable[[Any], int],
) -> list[Any]:
    features = {
        id(item): _hbm_feature(item, now_s, prompt_len_fn)
        for item in waiting_items
    }
    return sorted(
        list(waiting_items),
        key=lambda item: (
            features[id(item)]["tool_key"],
            features[id(item)]["arrival"],
            features[id(item)]["rid"],
        ),
    )


def _order_hbm_split_waiting(
    *,
    waiting_items: Iterable[Any],
    running_items: Iterable[Any],
    now_s: float,
    prompt_len_fn: Callable[[Any], int],
    use_tool_queue: bool,
    joint_pacer: bool = False,
) -> tuple[list[Any], int, float]:
    waiting = list(waiting_items)
    running = list(running_items)
    live_tokens = float(
        sum(_active_context_tokens(item, prompt_len_fn) for item in running)
    )
    running_count = len(running)
    budget = _hbm_target_context_tokens()
    long_threshold = _hbm_long_context_tokens()
    live_long_count = sum(
        1 for item in running
        if _active_context_tokens(item, prompt_len_fn) >= long_threshold
    )

    features = {
        id(item): _hbm_feature(
            item,
            now_s,
            prompt_len_fn,
            joint_pacer=joint_pacer,
        )
        for item in waiting
    }
    if use_tool_queue:
        base_order = sorted(
            waiting,
            key=lambda item: (
                features[id(item)]["tool_key"],
                _hbm_cost_key(features[id(item)]),
            ),
        )
    else:
        base_order = sorted(waiting, key=lambda item: _hbm_cost_key(features[id(item)]))

    # budget is adapted outside this pure ordering helper by _apply_hbm_capacity.
    virtual_tokens = 0.0
    virtual_long_count = 0
    admissible: list[Any] = []
    deferred: list[Any] = []
    fill_budget = budget * _hbm_virtual_fill_ratio()
    max_long = _hbm_max_long_running()
    max_admit = _hbm_max_admit_per_step()

    for item in base_order:
        f = features[id(item)]
        projected_tokens = live_tokens + virtual_tokens + float(f["kv_tokens"])
        is_long = float(f["prompt_tokens"]) >= long_threshold
        fits_tokens = projected_tokens <= fill_budget
        fits_long = not (
            is_long and max_long > 0
            and live_long_count + virtual_long_count >= max_long
        )
        if len(admissible) < max_admit and (fits_tokens and fits_long):
            admissible.append(item)
            virtual_tokens += float(f["kv_tokens"])
            virtual_long_count += 1 if is_long else 0
        else:
            deferred.append(item)

    if not admissible and waiting:
        # Work-conserving floor: do not stall admission completely merely
        # because all candidates are large. Let exactly one best request through.
        admissible = [base_order[0]]
        deferred = base_order[1:]

    return admissible + deferred, len(admissible), budget


def _joint_v2_trace_id(meta: dict[str, Any]) -> Optional[str]:
    trace_id = meta.get("t")
    return str(trace_id) if trace_id is not None else None


def _joint_v2_active_session_count() -> int:
    return len(_v2_started_sessions - _v2_completed_sessions)


def _joint_v2_waited_s(item: Any, now_s: float) -> float:
    arrival = _arrival_time(item)
    return max(0.0, now_s - arrival) if arrival > 0 else 0.0


def _joint_v2_decode_admit_allowance(
    *,
    running_count: int,
    waiting_items: Iterable[Any],
    now_s: float,
    native_max_running: int,
) -> int:
    """Return the causal count allowance imposed by the decode-pressure band.

    Below the target the controller is work-conserving.  At the target, only
    the next-to-admit request may use one deadline-triggered fairness slot.
    The effective upper bound is at most target + 1 even when a larger value
    is configured; this prevents a synchronized backlog from filling several
    fairness slots over consecutive scheduler ticks.  This uses only the
    current engine-running count (which can include chunked prefills) and the
    head request's arrival timestamp; the existing HBM controller independently
    applies causal current-token limits.
    """

    if native_max_running <= 0:
        return 0
    waiting = list(waiting_items)
    if not waiting:
        return 0

    target = min(native_max_running, _joint_v2_decode_target_running())
    upper = min(
        native_max_running,
        target + 1,
        max(target, _joint_v2_decode_max_running()),
    )
    running = max(0, int(running_count))
    if running < target:
        return target - running
    if running >= upper:
        return 0

    fairness_wait_s = _joint_v2_gate_max_wait_s()
    if fairness_wait_s <= 0.0:
        return 0
    head_wait_s = _joint_v2_waited_s(waiting[0], now_s)
    return 1 if head_wait_s >= fairness_wait_s else 0


def _joint_v2_is_new_session(meta: dict[str, Any]) -> bool:
    trace_id = _joint_v2_trace_id(meta)
    if trace_id is None or trace_id in _v2_started_sessions:
        return False
    return _meta_int(meta, "c", _meta_int(meta, "i", 0)) <= 0


def _joint_v2_prefix_fail_closed(
    waiting_count: int,
    reason: str,
) -> tuple[None, dict[str, Any]]:
    """Build one atomic fail-closed prefix-locality result."""

    return None, {
        "decision": "fail_closed",
        "reason": reason,
        "waiting": waiting_count,
        "lookup_requests": 0,
        "hit_requests": 0,
        "cached_tokens": 0,
    }


def _joint_v2_prefix_cache_snapshot(
    self: Any,
    waiting_items: Iterable[Any],
    now_s: Optional[float] = None,
) -> tuple[Optional[dict[int, int]], dict[str, Any]]:
    """Read one causal local-prefix snapshot from vLLM 0.10.1 v1.

    ``KVCacheManager.get_computed_blocks`` updates native prefix-cache
    counters.  Calling it here and again during native admission would double
    count every lookup.  Its vLLM 0.10.1 implementation delegates the actual
    read to ``coordinator.find_longest_cache_hit`` before updating counters,
    so this hook uses that read-only coordinator method directly.  It neither
    touches block refcounts nor allocates KV.

    The result is atomic across the waiting batch.  Any missing private API,
    malformed request, non-native waiting state, exception, or invalid return
    disables the signal for the whole tick; no subset receives a bonus.
    """

    waiting = list(waiting_items)
    waiting_count = len(waiting)
    if not _joint_v2_prefix_locality_enabled():
        return None, {
            "decision": "disabled",
            "reason": "default_off",
            "waiting": waiting_count,
        }

    kv_cache_manager = getattr(self, "kv_cache_manager", None)
    if kv_cache_manager is None:
        return _joint_v2_prefix_fail_closed(
            waiting_count, "kv_cache_manager_unavailable"
        )
    if getattr(kv_cache_manager, "enable_caching", None) is not True:
        return _joint_v2_prefix_fail_closed(
            waiting_count, "prefix_cache_disabled"
        )
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    lookup = getattr(coordinator, "find_longest_cache_hit", None)
    if not callable(lookup):
        return _joint_v2_prefix_fail_closed(
            waiting_count, "lookup_api_unavailable"
        )

    raw_block_size = getattr(kv_cache_manager, "block_size", None)
    if isinstance(raw_block_size, bool):
        return _joint_v2_prefix_fail_closed(waiting_count, "invalid_block_size")
    try:
        block_size = int(raw_block_size)
    except (TypeError, ValueError, OverflowError):
        return _joint_v2_prefix_fail_closed(waiting_count, "invalid_block_size")
    if block_size <= 0:
        return _joint_v2_prefix_fail_closed(waiting_count, "invalid_block_size")

    tick_s = time.time() if now_s is None else float(now_s)
    refresh_s = _joint_v2_prefix_locality_refresh_s()
    prior_cache = getattr(self, "_joint_v2_prefix_observations", {})
    if not isinstance(prior_cache, dict):
        prior_cache = {}
    next_cache: dict[str, tuple[float, int, int]] = {}
    snapshot: dict[int, int] = {}
    lookup_requests = 0
    reused_requests = 0
    hit_requests = 0
    cached_tokens_total = 0
    prompt_tokens_total = 0
    for item in waiting:
        raw_num_tokens = getattr(item, "num_tokens", None)
        raw_num_computed_tokens = getattr(item, "num_computed_tokens", None)
        if isinstance(raw_num_tokens, bool) or isinstance(
            raw_num_computed_tokens, bool
        ):
            return _joint_v2_prefix_fail_closed(
                waiting_count, "invalid_request_tokens"
            )
        try:
            num_tokens = int(raw_num_tokens)
            num_computed_tokens = int(raw_num_computed_tokens)
        except (TypeError, ValueError, OverflowError):
            return _joint_v2_prefix_fail_closed(
                waiting_count, "invalid_request_tokens"
            )
        if num_tokens < 0 or num_computed_tokens != 0:
            # vLLM bypasses the local prefix lookup when a connector has put a
            # waiting request into a partially-computed state.  Treating that
            # state as a local hit would mix two different mechanisms.
            return _joint_v2_prefix_fail_closed(
                waiting_count, "non_native_waiting_state"
            )
        prompt_tokens_total += num_tokens

        sampling_params = getattr(item, "sampling_params", None)
        prompt_logprobs = getattr(sampling_params, "prompt_logprobs", None)
        if prompt_logprobs is not None or num_tokens <= 1:
            cached_tokens = 0
            observed_s = tick_s
        else:
            block_hashes = getattr(item, "block_hashes", None)
            if not isinstance(block_hashes, (list, tuple)):
                return _joint_v2_prefix_fail_closed(
                    waiting_count, "block_hashes_unavailable"
                )
            request_key = _request_id(item)
            cached_record = prior_cache.get(request_key)
            reuse_record = False
            if refresh_s > 0.0 and isinstance(cached_record, tuple) and len(
                cached_record
            ) == 3:
                try:
                    observed_s = float(cached_record[0])
                    observed_num_tokens = int(cached_record[1])
                    cached_tokens = int(cached_record[2])
                    age_s = tick_s - observed_s
                    reuse_record = (
                        math.isfinite(observed_s)
                        and observed_num_tokens == num_tokens
                        and 0.0 <= age_s <= refresh_s
                    )
                except (TypeError, ValueError, OverflowError):
                    reuse_record = False
            if reuse_record:
                reused_requests += 1
            else:
                try:
                    result = lookup(block_hashes, num_tokens - 1)
                except Exception:
                    return _joint_v2_prefix_fail_closed(
                        waiting_count, "lookup_error"
                    )
                if not isinstance(result, tuple) or len(result) != 2:
                    return _joint_v2_prefix_fail_closed(
                        waiting_count, "invalid_lookup_result"
                    )
                cached_tokens = result[1]
                observed_s = tick_s
                lookup_requests += 1

        raw_cached_tokens = cached_tokens
        if isinstance(raw_cached_tokens, bool):
            return _joint_v2_prefix_fail_closed(
                waiting_count, "invalid_cached_tokens"
            )
        try:
            cached_tokens = int(raw_cached_tokens)
        except (TypeError, ValueError, OverflowError):
            return _joint_v2_prefix_fail_closed(
                waiting_count, "invalid_cached_tokens"
            )
        if (
            cached_tokens < 0
            or cached_tokens > num_tokens - 1
            or cached_tokens % block_size != 0
        ):
            return _joint_v2_prefix_fail_closed(
                waiting_count, "invalid_cached_tokens"
            )
        snapshot[id(item)] = cached_tokens
        next_cache[_request_id(item)] = (observed_s, num_tokens, cached_tokens)
        if cached_tokens > 0:
            hit_requests += 1
            cached_tokens_total += cached_tokens

    setattr(self, "_joint_v2_prefix_observations", next_cache)
    return snapshot, {
        "decision": "active",
        "reason": "causal_local_snapshot",
        "waiting": waiting_count,
        "lookup_requests": lookup_requests,
        "reused_requests": reused_requests,
        "hit_requests": hit_requests,
        "cached_tokens": cached_tokens_total,
        "prompt_tokens": prompt_tokens_total,
        "marginal_prefill_tokens": max(
            0, prompt_tokens_total - cached_tokens_total
        ),
        "refresh_s": f"{refresh_s:.6f}",
    }


def _joint_v2_apply_prefix_features(
    features: dict[int, dict[str, Any]],
    cached_tokens_by_id: Optional[dict[int, int]],
) -> bool:
    """Atomically attach cached and marginal token costs to score features."""

    if (
        not _joint_v2_prefix_locality_enabled()
        or cached_tokens_by_id is None
    ):
        return False
    if set(cached_tokens_by_id) != set(features):
        return False

    updates: dict[int, tuple[int, int, int]] = {}
    for item_id, f in features.items():
        raw_cached_tokens = cached_tokens_by_id[item_id]
        if isinstance(raw_cached_tokens, bool):
            return False
        try:
            cached_tokens = int(raw_cached_tokens)
            prompt_tokens = max(0, int(f["prompt_tokens"]))
            kv_tokens = max(0, int(f["kv_tokens"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if cached_tokens < 0:
            return False
        scored_cached_tokens = min(cached_tokens, prompt_tokens)
        updates[item_id] = (
            scored_cached_tokens,
            prompt_tokens - scored_cached_tokens,
            max(0, kv_tokens - scored_cached_tokens),
        )

    for item_id, (cached, marginal_prefill, marginal_kv) in updates.items():
        features[item_id]["cached_tokens"] = cached
        features[item_id]["marginal_prefill_tokens"] = marginal_prefill
        features[item_id]["marginal_kv_tokens"] = marginal_kv
    return True


def _maybe_log_joint_v2_prefix_locality(
    self: Any,
    evidence: dict[str, Any],
    now_s: float,
) -> None:
    """Emit bounded, machine-readable evidence for the prefix experiment."""

    if not _joint_v2_prefix_locality_enabled():
        return
    interval_s = _joint_v2_prefix_locality_log_interval_s()
    last_s = getattr(self, "_joint_v2_prefix_locality_last_log_s", None)
    state = (evidence.get("decision"), evidence.get("reason"))
    last_state = getattr(self, "_joint_v2_prefix_locality_last_state", None)
    if (
        state == last_state
        and last_s is not None
        and now_s - float(last_s) < interval_s
    ):
        return
    setattr(self, "_joint_v2_prefix_locality_last_log_s", now_s)
    setattr(self, "_joint_v2_prefix_locality_last_state", state)

    ordered_keys = (
        "decision",
        "reason",
        "waiting",
        "lookup_requests",
        "reused_requests",
        "hit_requests",
        "cached_tokens",
        "prompt_tokens",
        "marginal_prefill_tokens",
        "refresh_s",
        "input_head",
        "output_head",
        "head_changed",
    )
    fields = " ".join(
        f"{key}={evidence[key]}" for key in ordered_keys if key in evidence
    )
    print(
        f"[sched_policy_patch:prefix_locality] {fields}",
        file=sys.stderr,
        flush=True,
    )


def _joint_v2_score_s(
    f: dict[str, Any],
    *,
    live_tokens: float,
    virtual_tokens: float,
    live_long_count: int,
    virtual_long_count: int,
    is_new_session: bool,
) -> tuple[float, bool]:
    meta = f["meta"]
    prompt_tokens = max(0, int(f["prompt_tokens"]))
    kv_tokens = max(0, int(f["kv_tokens"]))
    cached_tokens = max(0, int(f.get("cached_tokens", 0)))
    raw_marginal_kv_tokens = max(
        0,
        int(f.get("marginal_kv_tokens", kv_tokens)),
    )
    prefix_weight = _joint_v2_prefix_locality_weight()
    prefix_discount_tokens = min(
        float(prompt_tokens),
        float(cached_tokens) * prefix_weight,
    )
    cached_kv_tokens = max(0, kv_tokens - raw_marginal_kv_tokens)
    marginal_kv_tokens = max(
        0.0,
        float(kv_tokens) - min(
            float(kv_tokens),
            float(cached_kv_tokens) * prefix_weight,
        ),
    )
    marginal_prefill_tokens = max(
        0.0,
        float(prompt_tokens) - prefix_discount_tokens,
    )
    remaining_calls = _meta_int(meta, "rc", 10**9)
    remaining_call_soft_weight_s = _joint_v2_remaining_call_soft_weight_s()
    soft_remaining_calls = (
        _joint_v2_soft_remaining_calls(meta)
        if remaining_call_soft_weight_s > 0.0
        else remaining_calls
    )
    next_tool_wait = max(0.0, float(f["next_tool_wait"]))
    remaining_tool_wait = max(
        0.0,
        _meta_float(
            meta,
            "rtw",
            next_tool_wait * max(0, soft_remaining_calls),
        ),
    )
    prompt_len = int(f["prompt_len"])
    max_tokens = int(f["max_tokens"])

    target_tokens = _hbm_target_context_tokens()
    fill_target = target_tokens * _hbm_virtual_fill_ratio()
    projected_tokens = live_tokens + virtual_tokens + float(kv_tokens)
    projected_pressure = max(0.0, projected_tokens / max(1.0, target_tokens))
    marginal_projected_tokens = (
        live_tokens + virtual_tokens + float(marginal_kv_tokens)
    )
    marginal_projected_pressure = max(
        0.0,
        marginal_projected_tokens / max(1.0, target_tokens),
    )

    # The baseline estimate prices the complete prompt.  A causal local cache
    # hit avoids prefill compute and new allocation for already-resident
    # blocks; it does not erase the request's logical context or its
    # conservative physical-KV forecast.  The marginal pressure below is only
    # an ordering cost.  Admission and over-budget checks retain the full
    # ``kv_tokens`` footprint.
    service_s = max(
        0.0,
        _service_estimate_v2_s(meta, prompt_len, max_tokens)
        - prefix_discount_tokens / _prefill_tokens_per_s_v2(),
    )
    prompt_cost_s = marginal_prefill_tokens / _oas_v3_context_tokens_per_s()
    context_penalty_s = (
        _joint_v2_context_alpha()
        * (marginal_projected_pressure ** 1.35)
        * prompt_cost_s
    )

    capped_remaining_tool_s = min(
        remaining_tool_wait,
        _joint_v2_tool_wait_cap_s() * max(1, soft_remaining_calls),
    )
    # In legacy mode the task-tail and reciprocal progress terms both encode
    # remaining-call stage.  A positive soft weight replaces both of those
    # stage components, rather than stacking a third progress signal on top.
    # The observed remaining-tool duration is retained because it represents
    # downstream work, not merely a second spelling of the call count.
    legacy_remaining_service_s = (
        max(0, remaining_calls) * _avg_call_service_s()
        if remaining_call_soft_weight_s <= 0.0
        else 0.0
    )
    task_tail_s = (
        service_s
        + legacy_remaining_service_s
        + _joint_v2_remaining_tool_weight() * capped_remaining_tool_s
    )
    remaining_call_soft_cost_s = (
        remaining_call_soft_weight_s * soft_remaining_calls
        if remaining_call_soft_weight_s > 0.0
        else 0.0
    )

    context_ref = _joint_v2_context_ref_tokens()
    context_damp = 1.0 + prompt_tokens / context_ref
    final_bonus_s = (
        _joint_v2_final_bonus_s() / context_damp
        if soft_remaining_calls == 0 else 0.0
    )
    progress_bonus_s = (
        _joint_v2_progress_bonus_s()
        / float(max(1, remaining_calls + 1))
        / context_damp
        if remaining_call_soft_weight_s <= 0.0
        else 0.0
    )
    capped_next_tool_s = (
        min(next_tool_wait, _joint_v2_tool_wait_cap_s())
        if soft_remaining_calls > 0 else 0.0
    )
    tool_damp = 1.0 + projected_pressure * prompt_tokens / context_ref
    tool_bonus_s = (
        _joint_v2_tool_beta()
        * _next_tool_wait_reliability(meta)
        * capped_next_tool_s
        / tool_damp
    )

    long_threshold = _hbm_long_context_tokens()
    is_long = prompt_tokens >= long_threshold
    long_over_cap = (
        is_long
        and _hbm_max_long_running() > 0
        and live_long_count + virtual_long_count >= _hbm_max_long_running()
    )
    token_over_budget = projected_tokens > fill_target
    over_budget = token_over_budget or long_over_cap
    over_penalty_s = 0.0
    if token_over_budget:
        over_ratio = max(0.0, projected_tokens / max(1.0, fill_target) - 1.0)
        over_penalty_s += _joint_v2_over_budget_penalty_s() * (1.0 + over_ratio) ** 2
    if long_over_cap:
        over_penalty_s += _joint_v2_over_budget_penalty_s()

    new_session_penalty_s = _joint_v2_new_session_penalty_s() if is_new_session else 0.0
    aged_score_s = (
        service_s
        + context_penalty_s
        + _joint_v2_tail_beta() * task_tail_s
        + remaining_call_soft_cost_s
        + over_penalty_s
        + new_session_penalty_s
        - tool_bonus_s
        - final_bonus_s
        - progress_bonus_s
        - _time_aging_alpha() * float(f["waited_s"])
    )
    return aged_score_s, over_budget


def _order_joint_pacer_v2_waiting(
    *,
    waiting_items: Iterable[Any],
    running_items: Iterable[Any],
    now_s: float,
    prompt_len_fn: Callable[[Any], int],
    prefix_cached_tokens_by_id: Optional[dict[int, int]] = None,
) -> tuple[list[Any], int, float]:
    waiting = list(waiting_items)
    running = list(running_items)
    live_tokens = float(
        sum(_active_context_tokens(item, prompt_len_fn) for item in running)
    )
    running_count = len(running)
    budget = _hbm_target_context_tokens()
    long_threshold = _hbm_long_context_tokens()
    live_long_count = sum(
        1 for item in running
        if _active_context_tokens(item, prompt_len_fn) >= long_threshold
    )

    features = {
        id(item): _hbm_feature(
            item,
            now_s,
            prompt_len_fn,
            joint_pacer=True,
        )
        for item in waiting
    }
    _joint_v2_apply_prefix_features(features, prefix_cached_tokens_by_id)
    gate_max_wait_s = _joint_v2_gate_max_wait_s()
    active_sessions = _joint_v2_active_session_count()
    max_foreground = _joint_v2_foreground_max_sessions()
    gate_min_running = _joint_v2_gate_min_running()

    def deadline_key(item: Any) -> tuple[float, str]:
        f = features[id(item)]
        return float(f["arrival"]), str(f["rid"])

    due_items = [
        item
        for item in waiting
        if gate_max_wait_s > 0.0
        and float(features[id(item)]["waited_s"]) >= gate_max_wait_s
    ]

    # Pin only one deadline-expired request.  Promoting every expired request
    # would turn a long, saturated run into EDF/FCFS and erase v2's score.
    decode_urgent_id: Optional[int] = None
    if running_count >= _joint_v2_deadline_min_running() and due_items:
        decode_urgent_id = id(min(due_items, key=deadline_key))

    foreground_gate_active = (
        active_sessions >= max_foreground
        and running_count >= gate_min_running
    )
    due_cold_items = [
        item
        for item in due_items
        if _joint_v2_is_new_session(features[id(item)]["meta"])
    ]
    gate_urgent_id: Optional[int] = None
    if foreground_gate_active and due_cold_items:
        gate_urgent_id = id(min(due_cold_items, key=deadline_key))

    pinned_urgent_id = (
        decode_urgent_id if decode_urgent_id is not None else gate_urgent_id
    )
    final_lane_enabled = _joint_v2_final_lane_enabled()
    soft_remaining_call_enabled = (
        _joint_v2_remaining_call_soft_weight_s() > 0.0
    )
    # A soft stage cost is an alternative to, not an extra tie-breaker inside,
    # the exact/coarse stage lane.  FINAL_LANE remains independently available.
    remaining_call_lane_enabled = (
        _joint_v2_remaining_call_lane_enabled()
        and not soft_remaining_call_enabled
    )
    coarse_remaining_lanes_enabled = (
        remaining_call_lane_enabled
        and _joint_v2_remaining_call_coarse_lanes_enabled()
    )

    def base_key(item: Any) -> tuple[Any, ...]:
        f = features[id(item)]
        is_new = _joint_v2_is_new_session(f["meta"])
        score_s, over_budget = _joint_v2_score_s(
            f,
            live_tokens=live_tokens,
            virtual_tokens=0.0,
            live_long_count=live_long_count,
            virtual_long_count=0,
            is_new_session=is_new,
        )
        if id(item) == pinned_urgent_id:
            # HBM checks below still decide whether this one urgent request is
            # actually admissible; pinning changes priority, not capacity.
            return (0, f["arrival"], f["rid"])
        if final_lane_enabled or remaining_call_lane_enabled:
            predicted_remaining_calls = _meta_int(f["meta"], "rc", 10**9)
            if predicted_remaining_calls < 0:
                predicted_remaining_calls = 10**9
            lane_key: tuple[int, ...] = ()
            if final_lane_enabled:
                lane_key += (0 if predicted_remaining_calls == 0 else 1,)
            if remaining_call_lane_enabled:
                remaining_lane = predicted_remaining_calls
                if coarse_remaining_lanes_enabled:
                    # The final lane above separates rc=0 when enabled.
                    # Within each non-final coarse bucket, over-budget state
                    # and the continuous Joint score remain authoritative.
                    remaining_lane = 0 if predicted_remaining_calls <= 2 else 1
                lane_key += (remaining_lane,)
            return (
                1,
                *lane_key,
                1 if over_budget else 0,
                score_s,
                1 if is_new else 0,
                f["arrival"],
                f["rid"],
            )
        # Preserve the exact legacy sort tuple when both causal lanes are off.
        return (
            1,
            1 if over_budget else 0,
            score_s,
            1 if is_new else 0,
            f["arrival"],
            f["rid"],
        )

    base_order = sorted(waiting, key=base_key)

    virtual_new_sessions = 0
    virtual_tokens = 0.0
    virtual_long_count = 0
    admissible: list[Any] = []
    deferred: list[Any] = []
    gated_new: list[Any] = []
    fill_budget = budget * _hbm_virtual_fill_ratio()
    max_long = _hbm_max_long_running()
    max_admit = _hbm_max_admit_per_step()

    for item in base_order:
        f = features[id(item)]
        is_new = _joint_v2_is_new_session(f["meta"])
        escaped_foreground_gate = is_new and id(item) == gate_urgent_id
        gate_new = (
            is_new
            and not escaped_foreground_gate
            and active_sessions + virtual_new_sessions >= max_foreground
            and running_count >= gate_min_running
        )
        if gate_new:
            gated_new.append(item)
            continue
        score_s, over_budget = _joint_v2_score_s(
            f,
            live_tokens=live_tokens,
            virtual_tokens=virtual_tokens,
            live_long_count=live_long_count,
            virtual_long_count=virtual_long_count,
            is_new_session=is_new,
        )
        projected_tokens = live_tokens + virtual_tokens + float(f["kv_tokens"])
        prompt_tokens = float(f["prompt_tokens"])
        is_long = prompt_tokens >= long_threshold
        fits_tokens = projected_tokens <= fill_budget
        fits_long = not (
            is_long and max_long > 0
            and live_long_count + virtual_long_count >= max_long
        )
        if len(admissible) < max_admit and fits_tokens and fits_long and not over_budget:
            admissible.append(item)
            virtual_tokens += float(f["kv_tokens"])
            virtual_long_count += 1 if is_long else 0
            virtual_new_sessions += 1 if is_new else 0
        else:
            deferred.append(item)

    if not admissible and deferred:
        # Keep progress for existing sessions even when the virtual long/token
        # budget is already full. Do not pierce the foreground gate just to
        # launch another cold session while enough foreground work is active.
        admissible = [deferred[0]]
        deferred = deferred[1:]

    ordered = admissible + deferred + gated_new
    return ordered, len(admissible), budget


def _v1_scheduler_uses_fcfs(self: Any) -> bool:
    """Return whether the native v1 scheduler's own policy is FCFS."""

    policy = getattr(self, "policy", None)
    value = getattr(policy, "value", policy)
    return isinstance(value, str) and value.strip().lower() == "fcfs"


def _joint_v2_running_priority_key(item: Any) -> Optional[tuple[int, int, int]]:
    """Return a causal v1 running-order key, or ``None`` to fail closed.

    vLLM v1 FCFS visits ``self.running`` from the front and, on KV allocation
    failure, preempts with ``self.running.pop()``.  Consequently this one key
    must serve both purposes: task-near requests go to the front for service,
    while the least advanced request in the least advanced task stage goes to
    the tail as the native victim candidate.

    The fields are deliberately limited to causal request state available at
    the scheduling tick:

    1. final-call lane (``rc == 0``),
    2. ascending remaining-call count,
    3. descending already-computed tokens (sunk-KV protection).

    Python's stable sort preserves native FCFS order for exact ties.  Missing
    or invalid stage metadata aborts the entire reorder rather than making an
    unrelated request an accidental preemption victim.
    """

    meta = _decode_meta(item)
    if "rc" not in meta:
        return None
    try:
        remaining_calls = int(meta["rc"])
        computed_tokens = max(
            0,
            int(getattr(item, "num_computed_tokens", 0) or 0),
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if remaining_calls < 0:
        return None
    return (
        0 if remaining_calls == 0 else 1,
        remaining_calls,
        -computed_tokens,
    )


def _maybe_reorder_v1_running(
    self: Any,
    waiting_items: Iterable[Any],
    now_s: float,
) -> bool:
    """Stably reorder v1 FCFS running requests under the strict opt-in.

    This only changes list order before vLLM enters ``schedule()``.  It never
    changes ``max_num_running_reqs``, allocates/frees KV blocks, or directly
    preempts a request.  vLLM therefore retains all admission and preemption
    state transitions.
    """

    if not (
        _joint_v2_running_priority_enabled()
        and _joint_v2_native_admission_enabled()
        and _v1_scheduler_uses_fcfs(self)
    ):
        return False
    try:
        if len(waiting_items) <= 0:
            return False
    except (TypeError, AttributeError):
        return False

    running = getattr(self, "running", None)
    if not isinstance(running, list) or len(running) <= 1:
        return False

    keyed: list[tuple[tuple[int, int, int], Any]] = []
    for item in running:
        key = _joint_v2_running_priority_key(item)
        if key is None:
            return False
        keyed.append((key, item))

    urgent_item: Optional[Any] = None
    max_wait_s = _joint_v2_running_priority_max_wait_s()
    if max_wait_s > 0.0:
        due = [
            (index, item)
            for index, item in enumerate(running)
            if _joint_v2_waited_s(item, now_s) >= max_wait_s
        ]
        if due:
            # Pin exactly one request.  The original index is the final tie
            # break, retaining native FCFS order for equal arrival times.
            _, urgent_item = min(
                due,
                key=lambda pair: (_arrival_time(pair[1]), pair[0]),
            )

    ordered = [
        item
        for _, item in sorted(
            keyed,
            key=lambda pair: (
                0 if pair[1] is urgent_item else 1,
                *pair[0],
            ),
        )
    ]
    changed = any(before is not after for before, after in zip(running, ordered))
    if changed:
        running[:] = ordered
    return changed


def _oracle_return_window_s() -> float:
    try:
        return float(os.getenv("ORACLE_RETURN_WINDOW_S", "5.0"))
    except (TypeError, ValueError):
        return 5.0


def _oracle_reserve_kv_scale() -> float:
    try:
        return float(os.getenv("ORACLE_RESERVE_KV_SCALE", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def _oracle_log_interval_s() -> float:
    try:
        return float(os.getenv("ORACLE_LOG_INTERVAL_S", "5.0"))
    except (TypeError, ValueError):
        return 5.0


def _pending_return_kv_from_meta(meta: dict[str, Any]) -> int:
    next_prompt_tokens = _meta_int(meta, "npt", -1)
    if next_prompt_tokens < 0:
        next_prompt_tokens = _meta_int(meta, "pt", 0)

    next_pred_out = _meta_float(meta, "npo", -1.0)
    if next_pred_out < 0:
        next_max_tokens = _meta_int(meta, "nmt", 0)
        next_pred_out = (
            float(next_max_tokens)
            if next_max_tokens > 0 else _default_predicted_output_tokens()
        )
    return max(0, int(next_prompt_tokens + max(0.0, next_pred_out)))


def _update_pending_returns(self: Any, running_items: Iterable[Any], now_s: float) -> None:
    """Track which sessions just entered tool-wait and when they will return.

    Runs every schedule() tick. Diffs the running set vs the previous tick;
    for each departed request, decodes its meta and, if the trace has more
    turns to come (rc>0) and a predicted next-tool-wait (nw) is present,
    records (return_time, next_turn_kv_tokens) under the trace_id.
    Entries are cleared when the next turn re-appears in waiting/running,
    and stale entries (long past their expected return) are dropped.
    """
    global _prev_running_ids
    running_list = list(running_items)
    current_running_ids = {_request_id(o) for o in running_list}
    for obj in running_list:
        meta = _decode_meta(obj)
        trace_id = meta.get("t")
        if trace_id is not None:
            _v2_started_sessions.add(str(trace_id))
    departed = _prev_running_ids - current_running_ids
    if departed:
        for rid in departed:
            meta = _META_CACHE.get(rid)
            if not meta:
                continue
            try:
                rc_val = int(meta.get("rc", 0) or 0)
            except (TypeError, ValueError):
                rc_val = 0
            trace_id = meta.get("t")
            if rc_val <= 0:
                if trace_id is not None:
                    _v2_completed_sessions.add(str(trace_id))
                continue
            nw_val = meta.get("nw")
            if nw_val is None:
                continue
            try:
                nw_s = float(nw_val)
            except (TypeError, ValueError):
                continue
            if nw_s < 0:
                continue
            if trace_id is None:
                continue
            _pending_returns[str(trace_id)] = (
                now_s + nw_s,
                _pending_return_kv_from_meta(meta),
            )

    # Drop pending entries for traces whose next turn already arrived.
    waiting_items = getattr(self, "waiting", None) or []
    for obj in list(running_list) + list(waiting_items):
        m = _decode_meta(obj)
        trace_id = m.get("t")
        if trace_id is not None:
            _pending_returns.pop(str(trace_id), None)

    # Drop entries that are far past their predicted return (would imply
    # the session terminated or the next turn never came).
    horizon_s = now_s - 60.0
    stale = [t for t, (rt, _) in _pending_returns.items() if rt < horizon_s]
    for t in stale:
        _pending_returns.pop(t, None)

    _prev_running_ids = current_running_ids


def _compute_reserved_kv(
    now_s: float,
    *,
    window_s: Optional[float] = None,
    scale: Optional[float] = None,
) -> float:
    window = _oracle_return_window_s() if window_s is None else float(window_s)
    reserve_scale = _oracle_reserve_kv_scale() if scale is None else float(scale)
    if window <= 0 or reserve_scale <= 0:
        return 0.0
    total = 0.0
    for return_time, kv in _pending_returns.values():
        delta_s = return_time - now_s
        if 0.0 <= delta_s <= window:
            total += float(max(0, kv))
    return total * reserve_scale


def _compute_reserved_slots(
    now_s: float,
    *,
    window_s: float,
    scale: float,
    max_slots: int,
) -> int:
    if window_s <= 0 or scale <= 0 or max_slots <= 0:
        return 0
    count = 0
    for return_time, _ in _pending_returns.values():
        delta_s = return_time - now_s
        if 0.0 <= delta_s <= window_s:
            count += 1
    return min(max_slots, int(count * scale + 0.999))


def _joint_v2_native_running_cap(self: Any) -> Optional[int]:
    """Capture vLLM's configured sequence ceiling without inventing one."""

    existing = getattr(self, "_oas_orig_max_num_running_reqs", None)
    raw = (
        existing
        if existing is not None
        else getattr(self, "max_num_running_reqs", None)
    )
    if isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if value <= 0:
        return None
    if existing is None:
        setattr(self, "_oas_orig_max_num_running_reqs", value)
    return value


def _joint_v2_physical_kv_state(
    self: Any,
) -> Optional[tuple[int, int, int, float, int]]:
    """Return physical block shape and live usage from the v1 scheduler.

    The capacity definition intentionally mirrors vLLM's scheduler API:
    ``cache_config.num_gpu_blocks * cache_config.block_size``.  A missing or
    malformed API returns ``None`` rather than substituting a configured token
    target, because such a substitution would make a physical-capacity claim
    unverifiable.
    """

    cache_config = getattr(self, "cache_config", None)
    if cache_config is None:
        return None
    raw_num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    raw_block_size = getattr(cache_config, "block_size", None)
    if isinstance(raw_num_gpu_blocks, bool) or isinstance(raw_block_size, bool):
        return None
    try:
        num_gpu_blocks = int(raw_num_gpu_blocks)
        block_size = int(raw_block_size)
    except (TypeError, ValueError, OverflowError):
        return None
    if num_gpu_blocks <= 0 or block_size <= 0:
        return None

    kv_cache_manager = getattr(self, "kv_cache_manager", None)
    if kv_cache_manager is None:
        return None
    try:
        usage = float(getattr(kv_cache_manager, "usage"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(usage) or not 0.0 <= usage <= 1.0:
        return None

    capacity_tokens = num_gpu_blocks * block_size
    # Convert utilization back to whole physical blocks conservatively.  The
    # small epsilon avoids turning an exact integer into the following block
    # solely because of binary floating-point representation.
    used_blocks = min(
        num_gpu_blocks,
        max(0, int(math.ceil(usage * num_gpu_blocks - 1e-9))),
    )
    live_tokens = used_blocks * block_size
    return num_gpu_blocks, block_size, capacity_tokens, usage, live_tokens


def _joint_v2_round_kv_tokens(tokens: float, block_size: int) -> int:
    if not math.isfinite(float(tokens)) or tokens <= 0.0 or block_size <= 0:
        return 0
    return int(math.ceil(float(tokens) / block_size)) * block_size


def _joint_v2_predicted_kv_tokens(
    item: Any,
    prompt_len_fn: Callable[[Any], int],
    block_size: int,
) -> int:
    """Conservatively predict a request's eventual block-rounded footprint."""

    prompt_len = prompt_len_fn(item)
    meta = _decode_meta(item)
    predicted = _estimated_kv_tokens(
        meta,
        prompt_len,
        _sampling_max_tokens(item),
    )
    # A resumed request may already contain more generated tokens than an
    # online output forecast.  Never predict less than its causal live length.
    predicted = max(predicted, _active_context_tokens(item, prompt_len_fn))
    return _joint_v2_round_kv_tokens(float(predicted), block_size)


def _joint_v2_physical_kv_running_forecast(
    running_items: Iterable[Any],
    prompt_len_fn: Callable[[Any], int],
    block_size: int,
) -> tuple[int, int]:
    """Return logical live tokens and forecast growth for running requests."""

    logical_live_tokens = 0
    predicted_growth_tokens = 0
    for item in running_items:
        active_tokens = _joint_v2_round_kv_tokens(
            float(_active_context_tokens(item, prompt_len_fn)),
            block_size,
        )
        predicted_tokens = _joint_v2_predicted_kv_tokens(
            item,
            prompt_len_fn,
            block_size,
        )
        logical_live_tokens += active_tokens
        predicted_growth_tokens += max(0, predicted_tokens - active_tokens)
    return logical_live_tokens, predicted_growth_tokens


def _maybe_log_joint_v2_physical_kv(
    self: Any,
    decision: dict[str, Any],
    now_s: float,
    *,
    force: bool = False,
) -> None:
    interval_s = _joint_v2_physical_kv_log_interval_s()
    last_s = getattr(self, "_joint_v2_physical_kv_last_log_s", None)
    current_cap = decision.get("effective_cap")
    last_cap = getattr(self, "_joint_v2_physical_kv_last_logged_cap", None)
    cap_changed = current_cap is not None and current_cap != last_cap
    if (
        not force
        and not cap_changed
        and last_s is not None
        and now_s - float(last_s) < interval_s
    ):
        return
    setattr(self, "_joint_v2_physical_kv_last_log_s", now_s)
    if current_cap is not None:
        setattr(self, "_joint_v2_physical_kv_last_logged_cap", current_cap)

    ordered_keys = (
        "decision",
        "reason",
        "num_gpu_blocks",
        "block_size",
        "capacity_tokens",
        "target_utilization",
        "budget_tokens",
        "usage",
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
        "capacity_write_source",
        "capacity_write_count",
        "rescue",
    )
    fields = " ".join(
        f"{key}={decision[key]}" for key in ordered_keys if key in decision
    )
    print(
        f"[sched_policy_patch:physical_kv] {fields}",
        file=sys.stderr,
        flush=True,
    )


def _write_joint_v2_physical_kv_cap(self: Any, value: int) -> int:
    """Write the dynamic cap and return a scheduler-local audit counter."""

    setattr(self, "max_num_running_reqs", value)
    count = int(
        getattr(self, "_joint_v2_physical_kv_capacity_write_count", 0) or 0
    ) + 1
    setattr(self, "_joint_v2_physical_kv_capacity_write_count", count)
    return count


def _apply_joint_v2_physical_kv_admission(
    self: Any,
    *,
    ordered: list[Any],
    running_items: Iterable[Any],
    prompt_len_fn: Callable[[Any], int],
    reserved_kv: float,
    now_s: Optional[float] = None,
) -> dict[str, Any]:
    """Set a forecast-driven admission cap from vLLM's physical KV shape.

    There is no configured running-count target, minimum, or per-step maximum.
    Every request whose predicted footprint fits the physical-token budget is
    eligible, bounded only by vLLM's native ``max_num_seqs`` safety ceiling.
    One aged request may consume the utilization reserve (but never projected
    physical capacity), and an empty engine admits one physically feasible
    request to guarantee progress.

    The function is an explicit Joint-v2 opt-in.  ``NATIVE_ADMISSION=1`` takes
    precedence and returns before any write to ``max_num_running_reqs``.
    """

    tick_s = time.time() if now_s is None else float(now_s)
    running = list(running_items)
    native_cap = _joint_v2_native_running_cap(self)
    common: dict[str, Any] = {
        "waiting": len(ordered),
        "running": len(running),
        "native_cap": native_cap if native_cap is not None else 0,
    }

    if _joint_v2_native_admission_enabled():
        # In normal call paths this helper is not invoked at all when native
        # admission is enabled.  The direct guard makes that invariant robust
        # to future refactors and, crucially, performs no capacity write.
        return {
            **common,
            "decision": "skipped",
            "reason": "native_admission",
        }
    if not _joint_v2_physical_kv_admission_enabled():
        return {
            **common,
            "decision": "skipped",
            "reason": "disabled",
        }

    if native_cap is None or len(running) > native_cap:
        decision = {
            **common,
            "decision": "fail_closed",
            "reason": "invalid_native_cap",
        }
        _maybe_log_joint_v2_physical_kv(self, decision, tick_s, force=True)
        return decision

    physical_state = _joint_v2_physical_kv_state(self)
    if physical_state is None:
        # If a prior valid tick installed a smaller dynamic cap, restore the
        # captured engine ceiling rather than leaving a stale private limit.
        write_count = _write_joint_v2_physical_kv_cap(self, native_cap)
        decision = {
            **common,
            "decision": "fail_closed",
            "reason": "physical_kv_api_unavailable",
            "effective_cap": native_cap,
            "capacity_write_source": "physical_kv",
            "capacity_write_count": write_count,
        }
        _maybe_log_joint_v2_physical_kv(self, decision, tick_s, force=True)
        return decision

    (
        num_gpu_blocks,
        block_size,
        capacity_tokens,
        usage,
        live_tokens,
    ) = physical_state
    utilization = _joint_v2_physical_kv_target_utilization()
    budget_blocks = max(1, int(math.floor(num_gpu_blocks * utilization)))
    budget_tokens = min(capacity_tokens, budget_blocks * block_size)
    logical_live_tokens, running_growth_tokens = (
        _joint_v2_physical_kv_running_forecast(
            running,
            prompt_len_fn,
            block_size,
        )
    )
    reserved_tokens = _joint_v2_round_kv_tokens(
        max(0.0, float(reserved_kv)),
        block_size,
    )
    committed_tokens = (
        max(live_tokens, logical_live_tokens)
        + running_growth_tokens
        + reserved_tokens
    )
    remaining_tokens = max(0, budget_tokens - committed_tokens)
    native_slots = max(0, native_cap - len(running))

    candidates: list[tuple[Any, int]] = []
    try:
        for item in ordered:
            predicted_tokens = _joint_v2_predicted_kv_tokens(
                item,
                prompt_len_fn,
                block_size,
            )
            if predicted_tokens <= 0:
                raise ValueError("non-positive request KV forecast")
            candidates.append((item, predicted_tokens))
    except Exception:
        write_count = _write_joint_v2_physical_kv_cap(self, native_cap)
        decision = {
            **common,
            "decision": "fail_closed",
            "reason": "invalid_request_forecast",
            "num_gpu_blocks": num_gpu_blocks,
            "block_size": block_size,
            "capacity_tokens": capacity_tokens,
            "target_utilization": f"{utilization:.6f}",
            "budget_tokens": budget_tokens,
            "usage": f"{usage:.6f}",
            "live_tokens": live_tokens,
            "effective_cap": native_cap,
            "capacity_write_source": "physical_kv",
            "capacity_write_count": write_count,
        }
        _maybe_log_joint_v2_physical_kv(self, decision, tick_s, force=True)
        return decision

    selected: list[tuple[Any, int]] = []
    selected_ids: set[int] = set()
    rescue = False
    reason = "budget"
    rescue_wait_s = _joint_v2_physical_kv_rescue_wait_s()
    due_candidates = [
        pair
        for pair in candidates
        if rescue_wait_s > 0.0
        and _joint_v2_waited_s(pair[0], tick_s) >= rescue_wait_s
    ]
    oldest_due = (
        min(
            due_candidates,
            key=lambda pair: (_arrival_time(pair[0]), _request_id(pair[0])),
        )
        if due_candidates
        else None
    )

    if native_slots > 0 and oldest_due is not None:
        due_item, due_tokens = oldest_due
        if due_tokens <= remaining_tokens:
            selected.append(oldest_due)
            selected_ids.add(id(due_item))
            remaining_tokens -= due_tokens
        elif live_tokens + due_tokens <= capacity_tokens:
            # Bypass only the configured utilization reserve/forecasts.  The
            # physical 100% boundary remains non-negotiable.
            selected.append(oldest_due)
            selected_ids.add(id(due_item))
            rescue = True
            reason = "aged_rescue"

    if not rescue:
        for item, predicted_tokens in candidates:
            if len(selected) >= native_slots:
                break
            if id(item) in selected_ids:
                continue
            if predicted_tokens <= remaining_tokens:
                selected.append((item, predicted_tokens))
                selected_ids.add(id(item))
                remaining_tokens -= predicted_tokens

    if (
        not selected
        and native_slots > 0
        and not running
        and candidates
    ):
        # A forecast may consume all reserve even while no request can release
        # KV.  Admit the smallest physically feasible request to avoid a
        # controller-created deadlock.
        feasible = [
            pair for pair in candidates if live_tokens + pair[1] <= capacity_tokens
        ]
        if feasible:
            progress_item = min(
                feasible,
                key=lambda pair: (pair[1], _arrival_time(pair[0]), _request_id(pair[0])),
            )
            selected = [progress_item]
            selected_ids = {id(progress_item[0])}
            rescue = True
            reason = "empty_progress"

    if selected:
        ordered[:] = [item for item, _ in selected] + [
            item for item, _ in candidates if id(item) not in selected_ids
        ]
    admit_count = len(selected)
    predicted_admit_tokens = sum(tokens for _, tokens in selected)
    effective_cap = min(native_cap, len(running) + admit_count)
    write_count = _write_joint_v2_physical_kv_cap(self, effective_cap)

    if not candidates:
        reason = "no_waiting"
    elif native_slots <= 0:
        reason = "native_full"
    elif not selected:
        reason = "forecast_hold"
    decision = {
        **common,
        "decision": "admit",
        "reason": reason,
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": block_size,
        "capacity_tokens": capacity_tokens,
        "target_utilization": f"{utilization:.6f}",
        "budget_tokens": budget_tokens,
        "usage": f"{usage:.6f}",
        "live_tokens": live_tokens,
        "logical_live_tokens": logical_live_tokens,
        "running_growth_tokens": running_growth_tokens,
        "reserved_tokens": reserved_tokens,
        "committed_tokens": committed_tokens,
        "predicted_admit_tokens": predicted_admit_tokens,
        "fit_admit": admit_count - (1 if rescue else 0),
        "admit": admit_count,
        "effective_cap": effective_cap,
        "capacity_write_source": "physical_kv",
        "capacity_write_count": write_count,
        "rescue": 1 if rescue else 0,
    }
    _maybe_log_joint_v2_physical_kv(self, decision, tick_s)
    return decision


def _apply_hbm_capacity_with_reserve(
    self: Any,
    *,
    ordered: list[Any],
    admissible_count: int,
    running_items: Iterable[Any],
    prompt_len_fn: Callable[[Any], int],
    reserved_kv: float,
    reserved_slots: int = 0,
    joint_v2_decode_band: bool = False,
    now_s: Optional[float] = None,
) -> None:
    """Same as _apply_hbm_capacity but treats `reserved_kv` as already-claimed
    by sessions that will return shortly. We subtract it from the live budget
    before computing the dynamic admission count, so new admissions cannot
    consume KV that we want to keep cached for returning sessions."""
    if not hasattr(self, "_oas_orig_max_num_running_reqs"):
        setattr(
            self,
            "_oas_orig_max_num_running_reqs",
            int(getattr(self, "max_num_running_reqs", 0) or 0),
        )
    orig_max = int(getattr(self, "_oas_orig_max_num_running_reqs", 0) or 0)
    if orig_max <= 0:
        return
    if joint_v2_decode_band and _joint_v2_native_admission_enabled():
        # The flag may be enabled after an earlier Joint tick tightened this
        # mutable scheduler field.  Restore the captured engine value as well
        # as avoiding any new HBM/decode cap.
        setattr(self, "max_num_running_reqs", orig_max)
        return

    running = list(running_items)
    live_tokens = float(
        sum(_active_context_tokens(item, prompt_len_fn) for item in running)
    )
    budget = _get_hbm_budget(self, live_tokens, len(ordered), len(running))
    # The reservation is the only difference from _apply_hbm_capacity: shrink
    # the effective fill_budget by the KV we want to preserve.
    fill_budget = budget * _hbm_virtual_fill_ratio() - float(reserved_kv)
    fill_budget = max(_hbm_min_context_tokens() * _hbm_virtual_fill_ratio(), fill_budget)

    allowance = max(0.0, fill_budget - live_tokens)
    if allowance <= 0 and len(running) >= _hbm_min_running_reqs():
        dynamic_admit = 0
    else:
        avg_waiting_kv = 0.0
        if ordered:
            sample = ordered[: min(len(ordered), _hbm_max_admit_per_step())]
            kvs = []
            for item in sample:
                prompt_len = prompt_len_fn(item)
                meta = _decode_meta(item)
                kvs.append(_estimated_kv_tokens(meta, prompt_len, _sampling_max_tokens(item)))
            avg_waiting_kv = sum(kvs) / len(kvs) if kvs else 0.0
        dynamic_admit = _hbm_max_admit_per_step()
        if avg_waiting_kv > 0:
            dynamic_admit = max(1, int(allowance / avg_waiting_kv))
            dynamic_admit = min(_hbm_max_admit_per_step(), dynamic_admit)

    if len(running) < _hbm_min_running_reqs():
        dynamic_admit = max(
            dynamic_admit,
            min(_hbm_max_admit_per_step(), _hbm_min_running_reqs() - len(running)),
        )
    elif reserved_slots > 0:
        slot_allowance = max(0, orig_max - len(running) - int(reserved_slots))
        dynamic_admit = min(dynamic_admit, slot_allowance)

    if not ordered:
        admit_count = 0
    elif admissible_count <= 0:
        admit_count = 0
    elif dynamic_admit <= 0 and len(running) >= _hbm_min_running_reqs():
        admit_count = 0
    else:
        admit_count = min(max(1, dynamic_admit), max(1, admissible_count))
    if joint_v2_decode_band:
        band_allowance = _joint_v2_decode_admit_allowance(
            running_count=len(running),
            # Only requests in the HBM-admissible prefix may consume decode
            # headroom.  A deadline-expired over-budget request becomes gate-
            # eligible, but it cannot lend its fairness slot to another item.
            waiting_items=ordered[:admissible_count],
            now_s=time.time() if now_s is None else float(now_s),
            native_max_running=orig_max,
        )
        admit_count = min(admit_count, band_allowance)
    new_cap = min(orig_max, max(len(running), len(running) + admit_count))
    setattr(self, "max_num_running_reqs", new_cap)


def _maybe_log_return_reserve(
    *,
    label: str,
    now_s: float,
    reserved_kv: float,
    reserved_slots: int,
    running_count: int,
    cap: int,
    window_s: float,
) -> None:
    global _oracle_last_log_s
    interval = _oracle_log_interval_s()
    if interval <= 0:
        return
    if now_s - _oracle_last_log_s < interval:
        return
    _oracle_last_log_s = now_s
    print(
        f"[sched_policy_patch:{label}] pending_returns={len(_pending_returns)} "
        f"reserved_kv={reserved_kv:.0f} reserved_slots={reserved_slots} "
        f"running={running_count} cap={cap} "
        f"window_s={window_s:.1f}",
        file=sys.stderr,
        flush=True,
    )


def _apply_hbm_capacity(
    self: Any,
    *,
    ordered: list[Any],
    admissible_count: int,
    running_items: Iterable[Any],
    prompt_len_fn: Callable[[Any], int],
) -> None:
    if not hasattr(self, "_oas_orig_max_num_running_reqs"):
        setattr(self, "_oas_orig_max_num_running_reqs", int(getattr(self, "max_num_running_reqs", 0) or 0))
    orig_max = int(getattr(self, "_oas_orig_max_num_running_reqs", 0) or 0)
    if orig_max <= 0:
        return

    running = list(running_items)
    live_tokens = float(
        sum(_active_context_tokens(item, prompt_len_fn) for item in running)
    )
    budget = _get_hbm_budget(self, live_tokens, len(ordered), len(running))
    fill_budget = budget * _hbm_virtual_fill_ratio()

    # Recompute a soft admit allowance from the adapted budget. The ordering
    # helper used a static target; this second pass is what makes the controller
    # adaptive without mutating queue order twice.
    allowance = max(0.0, fill_budget - live_tokens)
    if allowance <= 0 and len(running) >= _hbm_min_running_reqs():
        dynamic_admit = 0
    else:
        avg_waiting_kv = 0.0
        if ordered:
            sample = ordered[: min(len(ordered), _hbm_max_admit_per_step())]
            kvs = []
            for item in sample:
                prompt_len = prompt_len_fn(item)
                meta = _decode_meta(item)
                kvs.append(_estimated_kv_tokens(meta, prompt_len, _sampling_max_tokens(item)))
            avg_waiting_kv = sum(kvs) / len(kvs) if kvs else 0.0
        dynamic_admit = _hbm_max_admit_per_step()
        if avg_waiting_kv > 0:
            dynamic_admit = max(1, int(allowance / avg_waiting_kv))
            dynamic_admit = min(_hbm_max_admit_per_step(), dynamic_admit)

    if len(running) < _hbm_min_running_reqs():
        dynamic_admit = max(dynamic_admit, min(_hbm_max_admit_per_step(), _hbm_min_running_reqs() - len(running)))

    if not ordered:
        admit_count = 0
    elif admissible_count <= 0:
        admit_count = 0
    elif dynamic_admit <= 0 and len(running) >= _hbm_min_running_reqs():
        admit_count = 0
    else:
        admit_count = min(max(1, dynamic_admit), max(1, admissible_count))
    new_cap = min(orig_max, max(len(running), len(running) + admit_count))
    setattr(self, "max_num_running_reqs", new_cap)


def _critical_path_s(
    *,
    current_service_s: float,
    remaining_calls: int,
    remaining_tool_wait: float,
    include_current_service: bool,
) -> float:
    current = current_service_s if include_current_service else 0.0
    return current + remaining_tool_wait + max(0, remaining_calls) * _avg_call_service_s()


def _wspt_weight(remaining_calls: int, is_final: int) -> float:
    progress = _progress_weight() / float(max(1, remaining_calls + 1))
    final = _final_weight() if is_final else 0.0
    return max(1e-6, 1.0 + progress + final)


def _key_for(
    obj: Any,
    now_s: float,
    prompt_len_fn: Callable[[Any], int],
) -> tuple[Any, ...]:
    policy = _policy()
    arrival = _arrival_time(obj)
    waited_s = max(0.0, now_s - arrival) if arrival > 0 else 0.0
    prompt_len = prompt_len_fn(obj)
    max_tokens = _sampling_max_tokens(obj)
    rid = _request_id(obj)
    meta = _decode_meta(obj)
    tie = (arrival, rid)

    if policy == "sjf":
        return (prompt_len, *tie)
    if policy == "sjf_aging":
        return (prompt_len - _aging_alpha() * waited_s, *tie)
    if policy == "srpt":
        return (prompt_len + max_tokens, *tie)
    if policy == "srpt_aging":
        return (prompt_len + max_tokens - _aging_alpha() * waited_s, *tie)
    if policy == "ljf":
        return (-prompt_len, *tie)
    if policy == "random":
        return (_stable_random(obj), *tie)

    remaining_calls = _meta_int(meta, "rc", 10**9)
    total_calls = _meta_int(meta, "n", remaining_calls + 1)
    next_tool_wait = _meta_float(meta, "nw", 0.0 if remaining_calls == 0 else 10**9)
    remaining_tool_wait = _meta_float(meta, "rtw", 0.0 if remaining_calls == 0 else 10**9)
    is_final = 1 if remaining_calls == 0 else 0
    service_s = _service_estimate_s(meta, prompt_len, max_tokens)
    tail_srpt_s = _critical_path_s(
        current_service_s=service_s,
        remaining_calls=remaining_calls,
        remaining_tool_wait=remaining_tool_wait,
        include_current_service=False,
    )
    service_srpt_s = _critical_path_s(
        current_service_s=service_s,
        remaining_calls=remaining_calls,
        remaining_tool_wait=remaining_tool_wait,
        include_current_service=True,
    )
    aged_tail_srpt_s = tail_srpt_s - _time_aging_alpha() * waited_s
    aged_service_srpt_s = service_srpt_s - _time_aging_alpha() * waited_s
    wspt_key = service_s / _wspt_weight(remaining_calls, is_final)
    aged_wspt_key = wspt_key - _time_aging_alpha() * waited_s
    overlap_key = service_srpt_s - _overlap_beta() * max(0.0, next_tool_wait)
    aged_overlap_key = overlap_key - _time_aging_alpha() * waited_s

    if policy == "oracle_next":
        # Prefer calls whose next tool completion will return soon; final calls
        # have an effective next wait of zero.
        return (next_tool_wait, remaining_calls, *tie)
    if policy == "oracle_short_tail":
        # Shortest known downstream tool tail first. This uses tool-side trace
        # data instead of using prompt length as a runtime proxy.
        return (remaining_tool_wait, remaining_calls, *tie)
    if policy == "oracle_long_tail":
        # Opposite tail ordering: starts long known tool waits earlier so they
        # can overlap with other LLM work.
        return (-remaining_tool_wait, remaining_calls, *tie)
    if policy == "oracle_last":
        return (0 if is_final else 1, remaining_tool_wait, *tie)
    if policy == "oracle_turns":
        return (remaining_calls, remaining_tool_wait, *tie)
    if policy == "oracle_critical":
        # Complete final calls first, otherwise favor short upcoming tool gaps
        # and shorter known tails to reduce task-level completion time.
        return (0 if is_final else 1, next_tool_wait, remaining_tool_wait,
                remaining_calls, total_calls, *tie)
    if policy == "oracle_next_long":
        # A/B for tool-overlap intuition: for non-final calls, start long
        # known tool waits earlier so they can overlap with other LLM work.
        return (0 if is_final else 1, -next_tool_wait, remaining_calls,
                -remaining_tool_wait, *tie)
    if policy == "oracle_next_long_aging":
        return (0 if is_final else 1,
                -next_tool_wait - _time_aging_alpha() * waited_s,
                remaining_calls, -remaining_tool_wait, *tie)
    if policy == "oracle_task_srpt":
        # Task-level SRPT using only task tail metadata: future tool wait plus
        # an average service estimate for each remaining LLM call.
        return (0 if is_final else 1, tail_srpt_s, remaining_calls, *tie)
    if policy == "oracle_task_srpt_aging":
        return (0 if is_final else 1, aged_tail_srpt_s, remaining_calls, *tie)
    if policy == "oracle_service_srpt":
        # SRPT variant that also accounts for this request's prefill/decode
        # service estimate from prompt tokens and max output tokens.
        return (0 if is_final else 1, service_srpt_s, remaining_calls, *tie)
    if policy == "oracle_service_srpt_aging":
        return (0 if is_final else 1, aged_service_srpt_s, remaining_calls,
                *tie)
    if policy == "oracle_wspt":
        # Smith/WSPT-style value-per-cost: short service first, but tasks close
        # to completion receive more weight.
        return (wspt_key, remaining_calls, remaining_tool_wait, *tie)
    if policy == "oracle_wspt_aging":
        return (aged_wspt_key, remaining_calls, remaining_tool_wait, *tie)
    if policy == "oracle_final_wspt":
        return (0 if is_final else 1, wspt_key, remaining_calls,
                remaining_tool_wait, *tie)
    if policy == "oracle_overlap_srpt":
        # SRPT with a bonus for exposing long next tool waits early.
        return (0 if is_final else 1, overlap_key, remaining_calls, *tie)
    if policy == "oracle_overlap_srpt_aging":
        return (0 if is_final else 1, aged_overlap_key, remaining_calls, *tie)

    if policy == "online_critical":
        # Degraded oracle_critical for PASTE-only online setting: only the
        # predicted next tool wait is available. No trajectory lookahead
        # (is_final / rc / rtw / n) and no current request service estimate.
        return (next_tool_wait, *tie)
    if policy == "online_overlap_srpt_aging":
        # Degraded oracle_overlap_srpt_aging for PASTE-only online setting.
        # Uses only signals available without trajectory oracle:
        #   - current request service estimate from prompt tokens / max output
        #   - PASTE-predicted next tool wait (next_tool_wait, used as bonus)
        #   - locally observed waited time for aging
        online_key = (
            service_s
            - _overlap_beta() * max(0.0, next_tool_wait)
            - _time_aging_alpha() * waited_s
        )
        return (online_key, *tie)
    if policy == "online_overlap_srpt_aging_v2":
        # V2 service estimator: uses
        #   - fitted prefill/decode rates from offline trace calibration
        #   - meta["po"]: predicted output tokens (EMA over past calls in
        #     same trace, computed by sidecar predictor in the driver)
        # Falls back to mt when "po" missing.
        service_v2 = _service_estimate_v2_s(meta, prompt_len, max_tokens)
        online_key = (
            service_v2
            - _overlap_beta() * max(0.0, next_tool_wait)
            - _time_aging_alpha() * waited_s
        )
        return (online_key, *tie)
    if policy in {
        "online_oas_v3_no_nw",
        "online_oas_v3_nw_bonus",
        "online_oas_v3_nw_delay",
        "online_oas_v3_g025_nw_bonus",
        "online_oas_v3_g025_nw_delay",
        "online_oas_v3_g050_nw_bonus",
        "online_oas_v3_g050_nw_delay",
        "online_oas_v3_g075_nw_delay",
    }:
        # V3 separates isolated inference time from scheduling cost. The V2
        # fitted+EMA service estimate is retained, then an explicit long-context
        # penalty is added so the scheduler accounts for KV/HBM contention when
        # many long requests decode concurrently.
        base_cost = _oas_v3_base_cost_s(meta, prompt_len, max_tokens, policy)
        if policy.endswith("_nw_bonus"):
            base_cost -= _overlap_beta() * max(0.0, next_tool_wait)
        elif policy.endswith("_nw_delay"):
            base_cost += _overlap_beta() * max(0.0, next_tool_wait)
        online_key = base_cost - _time_aging_alpha() * waited_s
        return (online_key, *tie)
    if policy == "online_oas_v4":
        online_key = _oas_v4_base_key_s(
            meta=meta,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            next_tool_wait=next_tool_wait,
            waited_s=waited_s,
            context_pressure=0.0,
        )
        return (online_key, *tie)
    if policy == "online_oas_v5_tool_hbm":
        next_tool_wait = _meta_float(
            meta, "nw", 0.0 if remaining_calls == 0 else _oas_v5_tool_wait_cap_s()
        )
        online_key, over_budget = _oas_v5_score_s(
            meta=meta,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            next_tool_wait=next_tool_wait,
            waited_s=waited_s,
            live_tokens=0.0,
            virtual_tokens=0.0,
            live_long_count=0,
            virtual_long_count=0,
        )
        return (1 if over_budget else 0, online_key, *tie)
    if policy == "online_tool_queue":
        next_tool_wait = _meta_float(
            meta,
            "nw",
            0.0 if remaining_calls == 0 else _tool_queue_wait_cap_s(),
        )
        online_key = _tool_queue_key_s(
            meta=meta,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            next_tool_wait=next_tool_wait,
            waited_s=waited_s,
        )
        return (online_key, *tie)

    return tie


def _sorted_deque(items: Iterable[Any], key_fn: Callable[[Any], tuple[Any, ...]]) -> deque[Any]:
    return deque(sorted(items, key=key_fn))


def _order_oas_v4_waiting(
    *,
    waiting_items: Iterable[Any],
    running_items: Iterable[Any],
    now_s: float,
    prompt_len_fn: Callable[[Any], int],
) -> list[Any]:
    live_context_tokens = sum(
        _active_context_tokens(item, prompt_len_fn) for item in running_items
    )
    context_pressure = live_context_tokens / _oas_v4_target_context_tokens()

    def key_fn(obj: Any) -> tuple[Any, ...]:
        arrival = _arrival_time(obj)
        waited_s = max(0.0, now_s - arrival) if arrival > 0 else 0.0
        prompt_len = prompt_len_fn(obj)
        max_tokens = _sampling_max_tokens(obj)
        rid = _request_id(obj)
        meta = _decode_meta(obj)
        remaining_calls = _meta_int(meta, "rc", 10**9)
        next_tool_wait = _meta_float(
            meta, "nw", 0.0 if remaining_calls == 0 else 10**9
        )
        return (
            _oas_v4_base_key_s(
                meta=meta,
                prompt_len=prompt_len,
                max_tokens=max_tokens,
                next_tool_wait=next_tool_wait,
                waited_s=waited_s,
                context_pressure=context_pressure,
            ),
            arrival,
            rid,
        )

    def bucket_name(obj: Any) -> str:
        prompt_len = prompt_len_fn(obj)
        meta = _decode_meta(obj)
        prompt_tokens = _meta_int(meta, "pt", prompt_len)
        if prompt_tokens >= _oas_v4_long_bucket_tokens():
            return "long"
        if prompt_tokens >= _oas_v4_medium_bucket_tokens():
            return "medium"
        return "short"

    buckets: dict[str, list[Any]] = {"short": [], "medium": [], "long": []}
    for item in waiting_items:
        buckets[bucket_name(item)].append(item)
    for bucket in buckets.values():
        bucket.sort(key=key_fn)

    ordered: list[Any] = []
    pattern = _oas_v4_bucket_pattern()
    pattern_index = 0
    while any(buckets.values()):
        chosen_name: Optional[str] = None
        for _ in range(len(pattern)):
            candidate_name = pattern[pattern_index % len(pattern)]
            pattern_index += 1
            if buckets[candidate_name]:
                chosen_name = candidate_name
                break
        if chosen_name is None:
            chosen_name = min(
                (name for name, bucket in buckets.items() if bucket),
                key=lambda name: key_fn(buckets[name][0]),
            )
        ordered.append(buckets[chosen_name].pop(0))
    return ordered


def _order_oas_v5_waiting(
    *,
    waiting_items: Iterable[Any],
    running_items: Iterable[Any],
    now_s: float,
    prompt_len_fn: Callable[[Any], int],
) -> list[Any]:
    live_tokens = float(
        sum(_active_context_tokens(item, prompt_len_fn) for item in running_items)
    )
    long_threshold = _oas_v5_long_context_tokens()
    live_long_count = sum(
        1
        for item in running_items
        if _active_context_tokens(item, prompt_len_fn) >= long_threshold
    )

    remaining = list(waiting_items)
    features: dict[int, dict[str, Any]] = {}
    for item in remaining:
        arrival = _arrival_time(item)
        waited_s = max(0.0, now_s - arrival) if arrival > 0 else 0.0
        prompt_len = prompt_len_fn(item)
        max_tokens = _sampling_max_tokens(item)
        meta = _decode_meta(item)
        prompt_tokens = _meta_int(meta, "pt", prompt_len)
        remaining_calls = _meta_int(meta, "rc", 10**9)
        next_tool_wait = _meta_float(
            meta, "nw", 0.0 if remaining_calls == 0 else _oas_v5_tool_wait_cap_s()
        )
        features[id(item)] = {
            "arrival": arrival,
            "rid": _request_id(item),
            "waited_s": waited_s,
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "meta": meta,
            "prompt_tokens": prompt_tokens,
            "kv_tokens": _estimated_kv_tokens(meta, prompt_len, max_tokens),
            "is_long": prompt_tokens >= long_threshold,
            "next_tool_wait": next_tool_wait,
        }

    virtual_tokens = 0.0
    virtual_long_count = 0
    ordered: list[Any] = []
    deferred: list[Any] = []
    fill_target = _oas_v5_target_context_tokens() * _oas_v5_virtual_fill_ratio()

    def score_for(item: Any) -> tuple[Any, ...]:
        f = features[id(item)]
        score_s, over_budget = _oas_v5_score_s(
            meta=f["meta"],
            prompt_len=f["prompt_len"],
            max_tokens=f["max_tokens"],
            next_tool_wait=f["next_tool_wait"],
            waited_s=f["waited_s"],
            live_tokens=live_tokens,
            virtual_tokens=virtual_tokens,
            live_long_count=live_long_count,
            virtual_long_count=virtual_long_count,
        )
        return (
            1 if over_budget else 0,
            score_s,
            f["arrival"],
            f["rid"],
        )

    for item in sorted(remaining, key=score_for):
        f = features[id(item)]
        fits_token_budget = live_tokens + virtual_tokens + f["kv_tokens"] <= fill_target
        fits_long_budget = not (
            f["is_long"]
            and _oas_v5_max_long_running() > 0
            and live_long_count + virtual_long_count >= _oas_v5_max_long_running()
        )
        if fits_token_budget and fits_long_budget:
            ordered.append(item)
            virtual_tokens += float(f["kv_tokens"])
            virtual_long_count += 1 if f["is_long"] else 0
        else:
            deferred.append(item)

    if not ordered:
        # If the live set is already above budget, still let vLLM make progress
        # with the least harmful request instead of leaving the order unchanged.
        return sorted(remaining, key=score_for)

    ordered.extend(sorted(deferred, key=score_for))
    return ordered


def _install_v0(policy: str) -> bool:
    try:
        from vllm.core.scheduler import Scheduler
    except Exception:
        return False

    if getattr(Scheduler, "_vllm_sched_policy_patch_installed", False):
        return True

    original = Scheduler._schedule_prefills

    @wraps(original)
    def wrapped_schedule_prefills(self: Any, *args: Any, **kwargs: Any) -> Any:
        waiting = getattr(self, "waiting", None)
        current_policy = _policy()
        # Ordering needs two requests, but v2 admission control must also run
        # for a single waiter or that request can bypass the pressure band.
        should_patch = waiting is not None and (
            len(waiting) > 1
            or (len(waiting) == 1 and current_policy == "online_joint_pacer_v2")
        )
        if should_patch:
            now_s = time.time()
            if current_policy == "online_oas_v4":
                running = getattr(self, "running", [])
                self.waiting = deque(
                    _order_oas_v4_waiting(
                        waiting_items=waiting,
                        running_items=running,
                        now_s=now_s,
                        prompt_len_fn=_prompt_len_v0,
                    )
                )
            elif current_policy == "online_oas_v5_tool_hbm":
                running = getattr(self, "running", [])
                self.waiting = deque(
                    _order_oas_v5_waiting(
                        waiting_items=waiting,
                        running_items=running,
                        now_s=now_s,
                        prompt_len_fn=_prompt_len_v0,
                    )
                )
            elif current_policy in {
                "online_hbm_controller",
                "online_hbm_tool_split",
                "online_joint_pacer_v1",
                "online_joint_pacer_v2",
            }:
                running = getattr(self, "running", [])
                if current_policy == "online_joint_pacer_v2":
                    ordered, admissible_count, _ = _order_joint_pacer_v2_waiting(
                        waiting_items=waiting,
                        running_items=running,
                        now_s=now_s,
                        prompt_len_fn=_prompt_len_v0,
                    )
                else:
                    ordered, admissible_count, _ = _order_hbm_split_waiting(
                        waiting_items=waiting,
                        running_items=running,
                        now_s=now_s,
                        prompt_len_fn=_prompt_len_v0,
                        use_tool_queue=current_policy in {
                            "online_hbm_tool_split",
                            "online_joint_pacer_v1",
                        },
                        joint_pacer=current_policy == "online_joint_pacer_v1",
                    )
                if current_policy in {"online_joint_pacer_v1", "online_joint_pacer_v2"}:
                    # Native-admission v2 is strictly reorder-only: do not even
                    # enter the helper that owns the mutable engine cap.
                    if not (
                        current_policy == "online_joint_pacer_v2"
                        and _joint_v2_native_admission_enabled()
                    ):
                        reserved_kv = _compute_reserved_kv(
                            now_s,
                            window_s=_joint_return_window_s(),
                            scale=_joint_reserve_kv_scale(),
                        )
                        reserved_slots = _compute_reserved_slots(
                            now_s,
                            window_s=_joint_return_window_s(),
                            scale=_joint_reserve_slot_scale(),
                            max_slots=_joint_max_reserved_slots(),
                        )
                        if (
                            current_policy == "online_joint_pacer_v2"
                            and _joint_v2_physical_kv_admission_enabled()
                        ):
                            _apply_joint_v2_physical_kv_admission(
                                self,
                                ordered=ordered,
                                running_items=running,
                                prompt_len_fn=_prompt_len_v0,
                                reserved_kv=reserved_kv,
                                now_s=now_s,
                            )
                        else:
                            _apply_hbm_capacity_with_reserve(
                                self,
                                ordered=ordered,
                                admissible_count=admissible_count,
                                running_items=running,
                                prompt_len_fn=_prompt_len_v0,
                                reserved_kv=reserved_kv,
                                reserved_slots=reserved_slots,
                                joint_v2_decode_band=current_policy == "online_joint_pacer_v2",
                                now_s=now_s,
                            )
                else:
                    _apply_hbm_capacity(
                        self,
                        ordered=ordered,
                        admissible_count=admissible_count,
                        running_items=running,
                        prompt_len_fn=_prompt_len_v0,
                    )
                self.waiting = deque(ordered)
            elif current_policy == "online_tool_queue":
                self.waiting = deque(
                    _order_tool_queue_waiting(
                        waiting_items=waiting,
                        now_s=now_s,
                        prompt_len_fn=_prompt_len_v0,
                    )
                )
            else:
                self.waiting = _sorted_deque(
                    waiting,
                    lambda seq_group: _key_for(seq_group, now_s, _prompt_len_v0),
                )
        return original(self, *args, **kwargs)

    Scheduler._schedule_prefills = wrapped_schedule_prefills
    Scheduler._vllm_sched_policy_patch_installed = True
    Scheduler._vllm_sched_policy = policy
    return True


def _install_v1(policy: str) -> bool:
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return False

    if getattr(Scheduler, "_vllm_sched_policy_patch_installed", False):
        return True

    original = Scheduler.schedule

    @wraps(original)
    def wrapped_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        waiting = getattr(self, "waiting", None)
        current_policy = _policy()

        # Tool-return-aware policies: registry must update on every tick
        # regardless of waiting-queue length, otherwise we miss departures.
        if current_policy in {
            "oracle_tool_return_admission",
            "online_joint_pacer_v1",
            "online_joint_pacer_v2",
        }:
            now_s = time.time()
            running = getattr(self, "running", [])
            _update_pending_returns(self, running, now_s)
            if waiting is not None and len(waiting) >= 1:
                is_v2 = current_policy == "online_joint_pacer_v2"
                waiting_for_order = list(waiting)
                prefix_snapshot: Optional[dict[int, int]] = None
                prefix_evidence: dict[str, Any] = {}
                if is_v2:
                    prefix_snapshot, prefix_evidence = (
                        _joint_v2_prefix_cache_snapshot(
                            self,
                            waiting_for_order,
                            now_s=now_s,
                        )
                    )
                is_joint = current_policy in {
                    "online_joint_pacer_v1",
                    "online_joint_pacer_v2",
                }
                window_s = _joint_return_window_s() if is_joint else _oracle_return_window_s()
                reserve_scale = (
                    _joint_reserve_kv_scale() if is_joint else _oracle_reserve_kv_scale()
                )
                reserved_kv = _compute_reserved_kv(
                    now_s,
                    window_s=window_s,
                    scale=reserve_scale,
                )
                reserved_slots = (
                    _compute_reserved_slots(
                        now_s,
                        window_s=window_s,
                        scale=_joint_reserve_slot_scale(),
                        max_slots=_joint_max_reserved_slots(),
                    )
                    if is_joint else 0
                )
                if is_v2:
                    ordered, admissible_count, _ = _order_joint_pacer_v2_waiting(
                        waiting_items=waiting_for_order,
                        running_items=running,
                        now_s=now_s,
                        prompt_len_fn=_prompt_len_v1,
                        prefix_cached_tokens_by_id=prefix_snapshot,
                    )
                else:
                    ordered, admissible_count, _ = _order_hbm_split_waiting(
                        waiting_items=waiting_for_order,
                        running_items=running,
                        now_s=now_s,
                        prompt_len_fn=_prompt_len_v1,
                        use_tool_queue=is_joint,
                        joint_pacer=is_joint,
                    )
                # Native-admission v2 is strictly reorder-only: do not even
                # enter the helper that owns the mutable engine cap.
                if not (is_v2 and _joint_v2_native_admission_enabled()):
                    if is_v2 and _joint_v2_physical_kv_admission_enabled():
                        _apply_joint_v2_physical_kv_admission(
                            self,
                            ordered=ordered,
                            running_items=running,
                            prompt_len_fn=_prompt_len_v1,
                            reserved_kv=reserved_kv,
                            now_s=now_s,
                        )
                    else:
                        _apply_hbm_capacity_with_reserve(
                            self,
                            ordered=ordered,
                            admissible_count=admissible_count,
                            running_items=running,
                            prompt_len_fn=_prompt_len_v1,
                            reserved_kv=reserved_kv,
                            reserved_slots=reserved_slots,
                            joint_v2_decode_band=is_v2,
                            now_s=now_s,
                        )
                if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
                    waiting.clear()
                    waiting.extend(ordered)
                if is_v2:
                    if _joint_v2_prefix_locality_enabled():
                        input_head = (
                            _request_id(waiting_for_order[0])
                            if waiting_for_order else "none"
                        )
                        output_head = _request_id(ordered[0]) if ordered else "none"
                        prefix_evidence.update(
                            {
                                "input_head": input_head,
                                "output_head": output_head,
                                "head_changed": (
                                    1 if input_head != output_head else 0
                                ),
                            }
                        )
                        _maybe_log_joint_v2_prefix_locality(
                            self,
                            prefix_evidence,
                            now_s,
                        )
                    # One pre-iteration stable sort only.  In native FCFS the
                    # tail is also the allocation-failure victim, so never
                    # mutate this list from inside vLLM's scheduling loop.
                    _maybe_reorder_v1_running(self, waiting, now_s)
                _maybe_log_return_reserve(
                    label="joint" if is_joint else "oracle",
                    now_s=now_s,
                    reserved_kv=reserved_kv,
                    reserved_slots=reserved_slots,
                    running_count=len(list(running)),
                    cap=int(getattr(self, "max_num_running_reqs", 0) or 0),
                    window_s=window_s,
                )
            return original(self, *args, **kwargs)

        if waiting is not None and len(waiting) > 1:
            now_s = time.time()
            if current_policy == "online_oas_v4":
                running = getattr(self, "running", [])
                ordered = _order_oas_v4_waiting(
                    waiting_items=waiting,
                    running_items=running,
                    now_s=now_s,
                    prompt_len_fn=_prompt_len_v1,
                )
            elif current_policy == "online_oas_v5_tool_hbm":
                running = getattr(self, "running", [])
                ordered = _order_oas_v5_waiting(
                    waiting_items=waiting,
                    running_items=running,
                    now_s=now_s,
                    prompt_len_fn=_prompt_len_v1,
                )
            elif current_policy in {"online_hbm_controller", "online_hbm_tool_split"}:
                running = getattr(self, "running", [])
                ordered, admissible_count, _ = _order_hbm_split_waiting(
                    waiting_items=waiting,
                    running_items=running,
                    now_s=now_s,
                    prompt_len_fn=_prompt_len_v1,
                    use_tool_queue=current_policy == "online_hbm_tool_split",
                )
                _apply_hbm_capacity(
                    self,
                    ordered=ordered,
                    admissible_count=admissible_count,
                    running_items=running,
                    prompt_len_fn=_prompt_len_v1,
                )
            elif current_policy == "online_tool_queue":
                ordered = _order_tool_queue_waiting(
                    waiting_items=waiting,
                    now_s=now_s,
                    prompt_len_fn=_prompt_len_v1,
                )
            else:
                ordered = sorted(
                    list(waiting),
                    key=lambda request: _key_for(request, now_s, _prompt_len_v1),
                )
            # v1 FCFSRequestQueue subclasses deque. PriorityRequestQueue does
            # not expose clear/extend; leave it alone if the queue is not deque-
            # compatible.
            if hasattr(waiting, "clear") and hasattr(waiting, "extend"):
                waiting.clear()
                waiting.extend(ordered)
        return original(self, *args, **kwargs)

    Scheduler.schedule = wrapped_schedule
    Scheduler._vllm_sched_policy_patch_installed = True
    Scheduler._vllm_sched_policy = policy
    return True


def install() -> None:
    policy = _policy()
    if policy in {"", "fcfs", "default"}:
        return
    if policy not in _SUPPORTED_POLICIES:
        print(
            f"[sched_policy_patch] unknown VLLM_SCHED_POLICY={policy!r}; "
            "leaving scheduler unchanged",
            file=sys.stderr,
        )
        return

    installed_v0 = _install_v0(policy)
    installed_v1 = _install_v1(policy)
    if installed_v0 or installed_v1:
        print(
            f"[sched_policy_patch] installed policy={policy} "
            f"v0={installed_v0} v1={installed_v1}",
            file=sys.stderr,
        )
    else:
        print(
            f"[sched_policy_patch] vLLM scheduler not importable; "
            f"policy={policy} not installed",
            file=sys.stderr,
        )
