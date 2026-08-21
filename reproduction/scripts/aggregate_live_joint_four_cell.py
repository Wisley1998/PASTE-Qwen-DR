#!/usr/bin/env python3
"""Strict multi-block A/B/E/F aggregator for formal-v3 through v9 experiments.

Each ``--block`` supplies four independently produced ``result.json`` files:

* A: FCFS, demand-only tools
* B: FCFS, speculative tools
* E: Joint LLM scheduling, demand-only tools
* F: Joint LLM scheduling, speculative tools

The script deliberately revalidates the embedded task, LLM, physical-tool and
raw queue-timeline evidence.  It selects a frozen protocol profile from the
common workload split (or verifies ``--formal-workload``), including v4's
execution-aware code binding and wire-attempt ledger.  The formal load contains
one r00 task per source; the three repeated fresh-server blocks are folded
within source, so the effective statistical sample is profile-dependent: 60
for v3--v7 and 80 for v8--v9.  It never accepts a runner's aggregate latency as an
observation.  Structural or identity errors fail closed; preregistered
performance thresholds are explicit gates.  The fixed-final token partition is
validated as exact-tokenizer telemetry arithmetic; raw bytes independently
establish the semantic JSON and ASCII-space tail, not internal BPE boundaries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

from compare_live_joint_pair import (  # type: ignore
    CONTROLLED_HTTP_MAX_ATTEMPTS,
    CONTROLLED_HTTP_RETRY_BACKOFF_S,
    CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES,
    CONTROLLED_HTTP_RETRYABLE_STATUSES,
    ValidatedRun,
    _distribution,
    _percentile,
    _validate_retry_config,
    _validate_run,
)
from validate_live_joint_formal_workload import (  # type: ignore
    FORMAL_V3_WORKLOAD,
    FORMAL_V4_WORKLOAD,
    FORMAL_V5_WORKLOAD,
    FORMAL_V6_WORKLOAD,
    FORMAL_V7_WORKLOAD,
    FORMAL_V8_WORKLOAD,
    FORMAL_V9_WORKLOAD,
    validate_formal_workload,
)


SCHEMA = "paste_repro.live_joint_four_cell_formal"
SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_RESAMPLES = 10_000
FORMAL_BLOCK_COUNT = 3
FORMAL_SOURCE_COUNT = 60
FORMAL_REPLICAS = 1
FORMAL_TASKS_PER_CELL = FORMAL_SOURCE_COUNT * FORMAL_REPLICAS
FORMAL_MAX_ACTIVE_TASKS = 60
FORMAL_VLLM_MAX_NUM_SEQS = 96
FORMAL_CONTEXT_PADDING_TOKENS = 5_600
FORMAL_CONTEXT_PADDING_MAX_OVERSHOOT = 256
FORMAL_VISIT_CAPACITY = 1
FORMAL_VISIT_MIN_START_INTERVAL_S = 2.1
FORMAL_MAX_AUTHORITATIVE_RETRY_RATE = 0.02
FORMAL_MAX_RETRY_RATE_DIFFERENCE = 0.01
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVE_AGENT_MODULE = REPOSITORY_ROOT / "reproduction" / "paste_repro" / "live_agent.py"
LIVE_BROKER_MODULE = (
    REPOSITORY_ROOT / "reproduction" / "paste_repro" / "live_broker.py"
)
V9_DEVELOPMENT_ROOT = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/v9_screen/v9-screen-r1"
)
V9_COMPLETED_SCREEN = V9_DEVELOPMENT_ROOT / "completed_screen.json"
V9_STRICT_DEVELOPMENT_SELECTION = (
    V9_DEVELOPMENT_ROOT / "strict_development_selection.json"
)
V9_SELECTED_TRANSPORT = V9_DEVELOPMENT_ROOT / "stage-0/selected_transport.json"
EXECUTION_AWARE_POLICY_VERSION = (
    "exact-session-invocation-running-completed-v1"
)
HTTP_ATTEMPT_GATE_POLICY_VERSION = "shared-per-tool-monotonic-v1"
HTTP_ATTEMPT_SPACING_TOLERANCE_S = 0.02
GUIDED_JSON_RECOVERY_POLICY_VERSION = "escape-unescaped-string-controls-v1"
OUTPUT_CONTRACT_POLICY_VERSION = "guided-tool-json-plain-final-local-wrap-v1"
FINAL_ANSWER_CONTRACT_POLICY_VERSION = (
    "plain-text-unicode-whitespace-local-wrap-v1"
)
FINAL_ANSWER_MAX_CHARS = 480
FINAL_ANSWER_MAX_WORDS = 60
V7_OUTPUT_CONTRACT_POLICY_VERSION = (
    "guided-tool-and-final-json-strict-local-projection-v2"
)
V7_FINAL_ANSWER_CONTRACT_POLICY_VERSION = (
    "guided-json-strict-local-whitespace-bounded-prefix-v2"
)
V7_FINAL_ANSWER_SCHEMA_POLICY_VERSION = "xgrammar-unbounded-answer-exact-url-v1"
V8_FIXED_FINAL_COMPLETION_TOKENS = 192
V8_FIXED_FINAL_CONTRACT_POLICY_VERSION = (
    "guided-grammar-fixed-192-token-strict-tail-local-projection-v1"
)
V8_FINAL_GRAMMAR_POLICY_VERSION = (
    "xgrammar-compact-unbounded-answer-exact-url-ascii-space-tail-v1"
)
V8_FINAL_GRAMMAR_XGRAMMAR_VERSION = "0.1.21"
V8_OUTPUT_CONTRACT_POLICY_VERSION = (
    "guided-tool-json-and-fixed-final-grammar-strict-local-projection-v1"
)
V8_FINAL_ANSWER_SCHEMA_POLICY_VERSION = "xgrammar-unbounded-answer-exact-url-v1"
V9_COMPLETED_SCREEN_SHA256 = (
    "40b4a8033529883f26c1f298d54a92a69e4fcfb6cb942a8d5f70c98fc86481f3"
)
V9_STRICT_DEVELOPMENT_SELECTION_SHA256 = (
    "7f7c9de71f341741192de78ab8596b9cb01721fe211ec3faed79ee33bd7dc7cc"
)
V9_SELECTED_TRANSPORT_SHA256 = (
    "3c44458963c65deb55b35dfa5a2ff888d5e1ec4cb6c0ff350ebe41e53612dc0d"
)
V9_LIVE_BROKER_SHA256 = (
    "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27"
)
V9_SELECTED_POLICY = "F0"
V9_SELECTED_VISIT_INTERVAL_S = 2.5
V9_SELECTED_MIN_SPECULATIVE_TOOL_WORKERS = 0
FINAL_ANSWER_TARGET_CHARS = 360
CELL_IDS = ("A", "B", "E", "F")
EFFECTS = {
    "A_to_B": ("A", "B"),
    "E_to_F": ("E", "F"),
    "A_to_E": ("A", "E"),
    "B_to_F": ("B", "F"),
    "A_to_F": ("A", "F"),
}
_COMMON_CONFIG_EXCLUSIONS = frozenset(
    {
        "cell_label",
        "speculation_mode",
        "scheduler_environment",
        "expected_url_search_coverage",
        "formal_run",
    }
)


@dataclass(frozen=True)
class FormalProfile:
    """Frozen load/runtime contract selected by the workload split identity."""

    name: str
    split_id: str
    default_workload: Path
    context_padding_tokens: int
    visit_tool_capacity: int
    visit_canary_stride: int
    expected_canary_count: int
    source_count: int = FORMAL_SOURCE_COUNT
    replicas_per_source: int = FORMAL_REPLICAS
    max_active_tasks: int = FORMAL_MAX_ACTIVE_TASKS
    vllm_max_num_seqs: int = FORMAL_VLLM_MAX_NUM_SEQS
    min_native_waiting_below_cap_fraction: float = 0.05
    min_authoritative_tool_queue_fraction: float = 0.05
    min_dual_queue_pressure_samples: int = 1
    min_dual_queue_pressure_consecutive_s: float = 0.0
    max_dual_queue_adjacent_sample_gap_s: float | None = None
    min_ef_faster_sources: int = 42
    min_af_faster_sources: int = 48
    expected_final_completion_tokens: int | None = None
    require_strict_semantic_ascii_space_tail: bool = False
    require_ef_component_decomposition: bool = False
    max_ef_llm_component_speedup: float = 0.01
    min_ef_tool_saving_to_net_saving_ratio: float = 1.0
    require_modern_live_evidence: bool = False
    require_zero_guided_json_recovery: bool = False
    guided_json_parsed_call_count: int | None = None
    require_plain_final_output_contract: bool = False
    require_strict_guided_final_output_contract: bool = False
    vllm_max_model_len: int | None = None
    vllm_max_num_batched_tokens: int | None = None
    expected_workload_file_sha256: str | None = None
    expected_workload_canonical_sha256: str | None = None
    expected_workload_sources_sha256: str | None = None
    expected_live_agent_sha256: str | None = None
    require_current_live_agent_binding: bool = False
    visit_min_start_interval_s: float = FORMAL_VISIT_MIN_START_INTERVAL_S
    min_speculative_tool_workers: int = 0
    expected_live_broker_sha256: str | None = None
    require_current_live_broker_binding: bool = False
    require_v9_selection_provenance: bool = False
    bootstrap_seed: int = BOOTSTRAP_SEED

    @property
    def tasks_per_cell(self) -> int:
        return self.source_count * self.replicas_per_source


FORMAL_PROFILES = {
    "live-joint-wikipedia-frozen-formal-v3": FormalProfile(
        name="formal-v3",
        split_id="live-joint-wikipedia-frozen-formal-v3",
        default_workload=FORMAL_V3_WORKLOAD,
        context_padding_tokens=5_600,
        visit_tool_capacity=1,
        visit_canary_stride=10,
        expected_canary_count=6,
    ),
    "live-joint-wikipedia-frozen-formal-v4": FormalProfile(
        name="formal-v4",
        split_id="live-joint-wikipedia-frozen-formal-v4",
        default_workload=FORMAL_V4_WORKLOAD,
        context_padding_tokens=10_000,
        visit_tool_capacity=2,
        visit_canary_stride=6,
        expected_canary_count=10,
        require_modern_live_evidence=True,
        vllm_max_model_len=16_384,
        vllm_max_num_batched_tokens=2_048,
        expected_live_agent_sha256=(
            "d523800ff6caa06e5727b28294b2041b7e44f4856b5ebb67e159057709d66be3"
        ),
    ),
    "live-joint-wikipedia-frozen-formal-v5": FormalProfile(
        name="formal-v5",
        split_id="live-joint-wikipedia-frozen-formal-v5",
        default_workload=FORMAL_V5_WORKLOAD,
        context_padding_tokens=10_000,
        visit_tool_capacity=2,
        visit_canary_stride=6,
        expected_canary_count=10,
        require_modern_live_evidence=True,
        require_zero_guided_json_recovery=True,
        guided_json_parsed_call_count=3,
        vllm_max_model_len=16_384,
        vllm_max_num_batched_tokens=2_048,
        expected_workload_file_sha256=(
            "6b11193c8a0dbbd70f9ae4bc2c72b56737893b4d45dacd1d9970e01ca019ae31"
        ),
        expected_workload_canonical_sha256=(
            "7e89dea02bf2dfc5bf2b7dd2669c0d753097d5e2e351b26f018eb3df02268fbe"
        ),
        expected_workload_sources_sha256=(
            "478310accbd16ce623a4684465dd029a01efa80bfd299f3522943e90bf2cba46"
        ),
        expected_live_agent_sha256=(
            "678864a738084076bb21a181cf15baa24c5839599fc5547303b269bb9e8c8455"
        ),
    ),
    "live-joint-wikipedia-frozen-formal-v6": FormalProfile(
        name="formal-v6",
        split_id="live-joint-wikipedia-frozen-formal-v6",
        default_workload=FORMAL_V6_WORKLOAD,
        context_padding_tokens=10_000,
        visit_tool_capacity=2,
        visit_canary_stride=6,
        expected_canary_count=10,
        require_modern_live_evidence=True,
        require_zero_guided_json_recovery=True,
        guided_json_parsed_call_count=2,
        require_plain_final_output_contract=True,
        vllm_max_model_len=16_384,
        vllm_max_num_batched_tokens=2_048,
        expected_workload_file_sha256=(
            "44122877db66b1df4a985316c2a96b71d91d13c4e8be84affb73d405490bd43f"
        ),
        expected_workload_canonical_sha256=(
            "019fbc5177e45b4cc8cb752ccc28a7070ae1c70a1faeded787a1989dc262a96b"
        ),
        expected_workload_sources_sha256=(
            "e07a94c9485205e2fb864d65a6339ac5885b0821d0b2123113107bfed988f4e0"
        ),
        expected_live_agent_sha256=(
            "719b34c36b5bf4f30d2a6bd4c47e37fe23fdea66a6ad7a5ea8128bdfbb50c28f"
        ),
    ),
    "live-joint-wikipedia-frozen-formal-v7": FormalProfile(
        name="formal-v7",
        split_id="live-joint-wikipedia-frozen-formal-v7",
        default_workload=FORMAL_V7_WORKLOAD,
        context_padding_tokens=10_000,
        visit_tool_capacity=2,
        visit_canary_stride=6,
        expected_canary_count=10,
        require_modern_live_evidence=True,
        require_zero_guided_json_recovery=True,
        guided_json_parsed_call_count=2,
        require_strict_guided_final_output_contract=True,
        vllm_max_model_len=16_384,
        vllm_max_num_batched_tokens=2_048,
        expected_workload_file_sha256=(
            "cbf143f59f4d2a05650df68d8fa6f00d7471964a4b257d26dd092ba90c40e6c8"
        ),
        expected_workload_canonical_sha256=(
            "09e88d67f4aeb1994a566e11678fceb8f374f3b86f667da112f901209e0ef393"
        ),
        expected_workload_sources_sha256=(
            "710cc4f8d62f6c2b8ab78ec3d61d79be1ba7db25f47559accd407e7d0ddc810c"
        ),
        expected_live_agent_sha256=(
            "6fa736aa4e56657874834841c8a60b18c53e31f48ffbe741cc2e93f1c750432f"
        ),
        # v7 remains bound to the exact historical module SHA recorded in its
        # results.  Once v8 extends the module, only the newest profile binds
        # that SHA to the current checkout.
        require_current_live_agent_binding=False,
    ),
    "live-joint-wikipedia-frozen-formal-v8": FormalProfile(
        name="formal-v8",
        split_id="live-joint-wikipedia-frozen-formal-v8",
        default_workload=FORMAL_V8_WORKLOAD,
        context_padding_tokens=10_000,
        visit_tool_capacity=2,
        visit_canary_stride=6,
        expected_canary_count=14,
        source_count=80,
        replicas_per_source=1,
        max_active_tasks=80,
        vllm_max_num_seqs=96,
        min_native_waiting_below_cap_fraction=0.05,
        min_authoritative_tool_queue_fraction=0.05,
        min_dual_queue_pressure_samples=10,
        min_dual_queue_pressure_consecutive_s=1.0,
        max_dual_queue_adjacent_sample_gap_s=0.5,
        min_ef_faster_sources=56,
        min_af_faster_sources=64,
        expected_final_completion_tokens=V8_FIXED_FINAL_COMPLETION_TOKENS,
        require_strict_semantic_ascii_space_tail=True,
        require_ef_component_decomposition=True,
        max_ef_llm_component_speedup=0.01,
        min_ef_tool_saving_to_net_saving_ratio=1.0,
        require_modern_live_evidence=True,
        require_zero_guided_json_recovery=True,
        guided_json_parsed_call_count=2,
        vllm_max_model_len=16_384,
        vllm_max_num_batched_tokens=2_048,
        expected_workload_file_sha256=(
            "780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4"
        ),
        expected_workload_canonical_sha256=(
            "93b8cfad78b76c42101f7d0f23583911b01bc8c075260ae3d85bce45456a9ec7"
        ),
        expected_workload_sources_sha256=(
            "01b029c3427f5f04d4f1b83b4f9b13e5decd705e773ffdeaeebb15970150f0df"
        ),
        expected_live_agent_sha256=(
            "6dab494fa65749b1d60a5b5cbfbb4d0eed3c804b91b3646e0388c707cb7ade8f"
        ),
        require_current_live_agent_binding=True,
    ),
    "live-joint-wikipedia-frozen-formal-v9": FormalProfile(
        name="formal-v9",
        split_id="live-joint-wikipedia-frozen-formal-v9",
        default_workload=FORMAL_V9_WORKLOAD,
        context_padding_tokens=10_000,
        visit_tool_capacity=2,
        visit_canary_stride=6,
        expected_canary_count=14,
        source_count=80,
        replicas_per_source=1,
        max_active_tasks=80,
        vllm_max_num_seqs=96,
        min_native_waiting_below_cap_fraction=0.05,
        min_authoritative_tool_queue_fraction=0.05,
        min_dual_queue_pressure_samples=10,
        min_dual_queue_pressure_consecutive_s=1.0,
        max_dual_queue_adjacent_sample_gap_s=0.5,
        min_ef_faster_sources=56,
        min_af_faster_sources=64,
        expected_final_completion_tokens=V8_FIXED_FINAL_COMPLETION_TOKENS,
        require_strict_semantic_ascii_space_tail=True,
        require_ef_component_decomposition=True,
        max_ef_llm_component_speedup=0.01,
        min_ef_tool_saving_to_net_saving_ratio=1.0,
        require_modern_live_evidence=True,
        require_zero_guided_json_recovery=True,
        guided_json_parsed_call_count=2,
        vllm_max_model_len=16_384,
        vllm_max_num_batched_tokens=2_048,
        expected_workload_file_sha256=(
            "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20"
        ),
        expected_workload_canonical_sha256=(
            "de588fcbd46c1181156f5a6e49e0264c785c00c43e0d8c2a62698fb6217e3ce7"
        ),
        expected_workload_sources_sha256=(
            "750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c"
        ),
        expected_live_agent_sha256=(
            "6dab494fa65749b1d60a5b5cbfbb4d0eed3c804b91b3646e0388c707cb7ade8f"
        ),
        require_current_live_agent_binding=True,
        visit_min_start_interval_s=V9_SELECTED_VISIT_INTERVAL_S,
        min_speculative_tool_workers=V9_SELECTED_MIN_SPECULATIVE_TOOL_WORKERS,
        expected_live_broker_sha256=V9_LIVE_BROKER_SHA256,
        require_current_live_broker_binding=True,
        require_v9_selection_provenance=True,
        bootstrap_seed=20260817,
    ),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _lower_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _strict_true(value: Any, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label} must be true")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError as exc:
        raise ValueError(f"formal-v9 evidence is outside the repository: {path}") from exc


def _load_exact_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the frozen binding")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    return _mapping(value, label)


def _validate_v9_development_selection(
    *,
    profile: FormalProfile,
    workload_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the SHA-bound development decision before accepting v9 cells."""

    if not profile.require_v9_selection_provenance:
        raise ValueError("formal-v9 profile lacks its selection-provenance contract")
    completed = _load_exact_json(
        V9_COMPLETED_SCREEN,
        V9_COMPLETED_SCREEN_SHA256,
        "formal-v9 completed development screen",
    )
    selection = _load_exact_json(
        V9_STRICT_DEVELOPMENT_SELECTION,
        V9_STRICT_DEVELOPMENT_SELECTION_SHA256,
        "formal-v9 strict development selection",
    )
    transport = _load_exact_json(
        V9_SELECTED_TRANSPORT,
        V9_SELECTED_TRANSPORT_SHA256,
        "formal-v9 selected transport",
    )
    completed_ref = {
        "path": _repository_relative(V9_COMPLETED_SCREEN),
        "sha256": V9_COMPLETED_SCREEN_SHA256,
    }
    selection_ref = {
        "path": _repository_relative(V9_STRICT_DEVELOPMENT_SELECTION),
        "sha256": V9_STRICT_DEVELOPMENT_SELECTION_SHA256,
    }
    transport_ref = {
        "path": _repository_relative(V9_SELECTED_TRANSPORT),
        "sha256": V9_SELECTED_TRANSPORT_SHA256,
    }
    if (
        completed.get("schema")
        != "paste_repro.live_joint_v9_development_screen_completion"
        or completed.get("version") != 1
        or completed.get("development_only") is not True
        or completed.get("formal_eligible") is not False
        or completed.get("formal_evidence_eligible") is not False
        or completed.get("development_selection_passed") is not True
        or completed.get("selected_policy") != V9_SELECTED_POLICY
        or completed.get("selected_transport") != transport_ref
        or completed.get("strict_development_selection") != selection_ref
    ):
        raise ValueError("formal-v9 completed screen is not the frozen F0 winner")
    completed_bindings = _mapping(
        completed.get("bindings"), "formal-v9 completed screen bindings"
    )
    if (
        completed_bindings.get("reproduction/paste_repro/live_broker.py")
        != V9_LIVE_BROKER_SHA256
    ):
        raise ValueError("formal-v9 development screen is not bound to live_broker.py")

    candidate_passed = _mapping(
        selection.get("candidate_passed"),
        "formal-v9 development candidate_passed",
    )
    common_identity = _mapping(
        selection.get("common_code_and_config_identity"),
        "formal-v9 development common identity",
    )
    development_cells = _mapping(
        common_identity.get("cells"), "formal-v9 development cells"
    )
    f0_rows = [
        _mapping(row, f"formal-v9 development {label}")
        for label, row in development_cells.items()
        if isinstance(label, str) and label.endswith("/F0")
    ]
    if (
        selection.get("schema") != "paste_repro.live_joint_v9_development_screen"
        or selection.get("version") != 1
        or selection.get("valid") is not True
        or selection.get("development_only") is not True
        or selection.get("formal_eligible") is not False
        or selection.get("formal_evidence_eligible") is not False
        or selection.get("development_selection_passed") is not True
        or candidate_passed.get(V9_SELECTED_POLICY) is not True
        or selection.get("F1_incremental_passed") is not False
        or selection.get("selected_policy") != V9_SELECTED_POLICY
        or selection.get("selected_visit_interval_s")
        != V9_SELECTED_VISIT_INTERVAL_S
        or len(f0_rows) != 2
        or any(
            row.get("speculation_mode") != "visit"
            or row.get("min_speculative_tool_workers")
            != V9_SELECTED_MIN_SPECULATIVE_TOOL_WORKERS
            for row in f0_rows
        )
    ):
        raise ValueError("formal-v9 strict development selection is not F0/2.5s")
    attempt_summaries = transport.get("attempt_summaries")
    if (
        transport.get("schema")
        != "paste_repro.live_joint_v9_development_transport_selection"
        or transport.get("version") != 1
        or transport.get("valid") is not True
        or transport.get("development_only") is not True
        or transport.get("formal_eligible") is not False
        or transport.get("formal_evidence_eligible") is not False
        or transport.get("selected_visit_interval_s")
        != V9_SELECTED_VISIT_INTERVAL_S
        or transport.get("candidate_performance_observed_or_used") is not False
        or transport.get("selection_input_cells") != ["A"]
        or transport.get("selection_reason")
        != "first_zero_retry_load_qualified_baseline"
        or transport.get("attempt_count") != 1
        or not isinstance(attempt_summaries, list)
        or attempt_summaries
        != [
            {
                "accepted": True,
                "failed_gates": [],
                "retry_only_fallback_eligible": False,
                "visit_interval_s": V9_SELECTED_VISIT_INTERVAL_S,
            }
        ]
    ):
        raise ValueError("formal-v9 transport selection is not baseline-only zero-retry")

    expected_broker_sha = profile.expected_live_broker_sha256
    if expected_broker_sha != V9_LIVE_BROKER_SHA256:
        raise ValueError("formal-v9 profile has the wrong live-broker binding")
    return {
        "completed_screen": completed_ref,
        "strict_development_selection": selection_ref,
        "selected_transport": transport_ref,
        "selected_policy": V9_SELECTED_POLICY,
        "selected_visit_interval_s": V9_SELECTED_VISIT_INTERVAL_S,
        "selected_min_speculative_tool_workers": (
            V9_SELECTED_MIN_SPECULATIVE_TOOL_WORKERS
        ),
        "maximum_observed_http_retries_per_cell": 0,
        "zero_wasted_speculative_service_required": True,
        "live_broker_sha256": V9_LIVE_BROKER_SHA256,
        "workload": {
            "path": _repository_relative(profile.default_workload),
            "raw_sha256": workload_validation["file_sha256"],
            "canonical_sha256": workload_validation["canonical_json_sha256"],
            "sources_sha256": workload_validation["canonical_sources_sha256"],
            "source_count": profile.source_count,
        },
    }


def _validate_v9_cell_provenance(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    *,
    expected_selection: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind every raw v9 result to coordinator-owned effective/manifest bytes."""

    result: dict[str, dict[str, Any]] = {}
    expected_scheduler = {
        "A": "fcfs",
        "B": "fcfs",
        "E": "online_joint_pacer_v2",
        "F": "online_joint_pacer_v2",
    }
    for block_id in sorted(runs):
        result[block_id] = {}
        for cell in CELL_IDS:
            run = runs[block_id][cell]
            if run.path.name != "result.json" or run.path.parent.name != "evidence":
                raise ValueError(
                    f"{block_id}/{cell} formal-v9 result is outside the cell evidence layout"
                )
            cell_root = run.path.parent.parent
            effective_path = cell_root / "effective_config.json"
            manifest_path = cell_root / "cell_manifest.json"
            if not effective_path.is_file() or not manifest_path.is_file():
                raise ValueError(
                    f"{block_id}/{cell} lacks effective_config.json/cell_manifest.json"
                )
            try:
                effective = _mapping(
                    json.loads(effective_path.read_text(encoding="utf-8")),
                    f"{block_id}/{cell}.effective_config",
                )
                manifest = _mapping(
                    json.loads(manifest_path.read_text(encoding="utf-8")),
                    f"{block_id}/{cell}.cell_manifest",
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{block_id}/{cell} coordinator evidence is not valid JSON"
                ) from exc
            formal = _mapping(
                run.config.get("formal_run"), f"{block_id}/{cell}.formal_run"
            )
            expected_effective_identity = {
                "schema": "paste_repro.live_joint_formal_cell_config",
                "version": 1,
                "formal_generation": "v9",
                "block_id": block_id,
                "cell_id": cell,
                "order_index": formal.get("order_index"),
                "server_instance_id": formal.get("server_instance_id"),
                "llm_scheduler": expected_scheduler[cell],
                "speculation_mode": run.config.get("speculation_mode"),
                "call_graph_mode": "frozen",
                "min_speculative_tool_workers": (
                    V9_SELECTED_MIN_SPECULATIVE_TOOL_WORKERS
                ),
            }
            changed = sorted(
                key
                for key, expected in expected_effective_identity.items()
                if effective.get(key) != expected
            )
            if changed or effective.get("formal_v9_selection") != expected_selection:
                detail = changed or ["formal_v9_selection"]
                raise ValueError(
                    f"{block_id}/{cell} formal-v9 effective config differs: "
                    + ", ".join(detail)
                )
            workload = _mapping(
                effective.get("workload"), f"{block_id}/{cell}.effective workload"
            )
            if workload != {
                "path": expected_selection["workload"]["path"],
                "sha256": expected_selection["workload"]["raw_sha256"],
            }:
                raise ValueError(f"{block_id}/{cell} effective workload binding differs")

            expected_manifest_identity = {
                "schema": "paste_repro.live_joint_formal_cell_evidence",
                "version": 1,
                "block_id": block_id,
                "cell_id": cell,
                "order_index": formal.get("order_index"),
                "server_instance_id": formal.get("server_instance_id"),
            }
            manifest_changed = sorted(
                key
                for key, expected in expected_manifest_identity.items()
                if manifest.get(key) != expected
            )
            if manifest_changed:
                raise ValueError(
                    f"{block_id}/{cell} cell manifest identity differs: "
                    + ", ".join(manifest_changed)
                )
            timeline = _mapping(
                _mapping(
                    run.payload.get("raw_evidence"),
                    f"{block_id}/{cell}.raw_evidence",
                ).get("queue_timeline"),
                f"{block_id}/{cell}.queue_timeline",
            )
            timeline_path = Path(_nonempty(
                timeline.get("path"), f"{block_id}/{cell}.queue_timeline.path"
            )).resolve()
            if timeline_path != (run.path.parent / "queue_timeline.jsonl").resolve():
                raise ValueError(f"{block_id}/{cell} timeline is outside its evidence dir")
            required_manifest_evidence = {
                _repository_relative(effective_path): _sha256_file(effective_path),
                _repository_relative(run.path): run.sha256,
                _repository_relative(timeline_path): _lower_sha256(
                    timeline.get("sha256"), f"{block_id}/{cell}.timeline.sha256"
                ),
            }
            manifest_evidence = _mapping(
                manifest.get("evidence"), f"{block_id}/{cell}.manifest.evidence"
            )
            differing_evidence = sorted(
                path
                for path, sha256 in required_manifest_evidence.items()
                if manifest_evidence.get(path) != sha256
            )
            if differing_evidence:
                raise ValueError(
                    f"{block_id}/{cell} manifest evidence SHA differs: "
                    + ", ".join(differing_evidence)
                )
            result[block_id][cell] = {
                "effective_config": {
                    "path": _repository_relative(effective_path),
                    "sha256": _sha256_file(effective_path),
                },
                "cell_manifest": {
                    "path": _repository_relative(manifest_path),
                    "sha256": _sha256_file(manifest_path),
                },
                "result_manifest_binding": True,
                "timeline_manifest_binding": True,
                "formal_v9_selection": dict(expected_selection),
            }
    return result


@lru_cache(maxsize=512)
def _fixed_final_grammar_sha256(url: str) -> str:
    try:
        import xgrammar
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise ValueError("fixed-final grammar validation requires xgrammar") from exc
    try:
        xgrammar_version = importlib.metadata.version("xgrammar")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("fixed-final validation cannot resolve the xgrammar version") from exc
    if xgrammar_version != V8_FINAL_GRAMMAR_XGRAMMAR_VERSION:
        raise ValueError(
            "fixed-final xgrammar version differs from the frozen contract"
        )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "source_url"],
        "properties": {
            "answer": {"type": "string"},
            "source_url": {"const": url},
        },
    }
    semantic = xgrammar.Grammar.from_json_schema(
        schema,
        any_whitespace=False,
        separators=(",", ":"),
        strict_mode=True,
    )
    tail = xgrammar.Grammar.from_ebnf('root ::= " "+')
    grammar = str(xgrammar.Grammar.concat(semantic, tail))
    return hashlib.sha256(grammar.encode("utf-8")).hexdigest()


def _bounded_answer_prefix(canonical: str) -> tuple[str, bool, bool]:
    words = canonical.split(" ") if canonical else []
    word_limited = " ".join(words[:FINAL_ANSWER_MAX_WORDS])
    word_projection = len(words) > FINAL_ANSWER_MAX_WORDS
    char_projection = len(word_limited) > FINAL_ANSWER_MAX_CHARS
    projected = word_limited
    if char_projection:
        projected = word_limited[:FINAL_ANSWER_MAX_CHARS].rstrip()
        if (
            projected
            and len(word_limited) > FINAL_ANSWER_MAX_CHARS
            and not word_limited[FINAL_ANSWER_MAX_CHARS].isspace()
            and " " in projected
        ):
            whole_words = projected.rsplit(" ", 1)[0]
            if whole_words:
                projected = whole_words
    return projected, word_projection, char_projection


def _json_has_whitespace_outside_strings(value: str) -> bool:
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character.isspace():
            return True
    return False


def _relative_difference(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return abs(candidate - baseline) / baseline


def _resolve_formal_profile(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    requested_workload: Path | None,
) -> tuple[FormalProfile, Path, Mapping[str, Any]]:
    """Resolve one frozen profile from raw result identity, then bind its bytes.

    Auto-detection is intentionally limited to known immutable formal splits.
    Supplying ``requested_workload`` additionally prevents an operator from
    aggregating a different known split by accident.
    """

    split_ids = {
        run.config.get("workload_split_id")
        for block in runs.values()
        for run in block.values()
    }
    if len(split_ids) != 1:
        raise ValueError("formal cells do not share one workload split ID")
    split_id = next(iter(split_ids))
    if not isinstance(split_id, str) or split_id not in FORMAL_PROFILES:
        raise ValueError(f"unsupported formal workload split ID: {split_id!r}")
    profile = FORMAL_PROFILES[split_id]
    workload_path = (
        Path(requested_workload).resolve()
        if requested_workload is not None
        else profile.default_workload.resolve()
    )
    workload_validation = validate_formal_workload(workload_path)
    if workload_validation.get("split_id") != profile.split_id:
        raise ValueError(
            "requested formal workload split differs from result split identity"
        )
    expected_default_sha = _sha256_file(profile.default_workload.resolve())
    observed_sha = str(workload_validation.get("file_sha256"))
    if observed_sha != expected_default_sha:
        raise ValueError(
            f"{profile.name} workload bytes differ from the frozen repository binding"
        )
    pinned_hashes = {
        "file_sha256": profile.expected_workload_file_sha256,
        "canonical_json_sha256": profile.expected_workload_canonical_sha256,
        "canonical_sources_sha256": profile.expected_workload_sources_sha256,
    }
    differing_pins = sorted(
        key
        for key, expected in pinned_hashes.items()
        if expected is not None and workload_validation.get(key) != expected
    )
    if differing_pins:
        raise ValueError(
            f"{profile.name} workload differs from pinned protocol SHA: "
            + ", ".join(differing_pins)
        )
    return profile, workload_path, workload_validation


def _source_values(run: ValidatedRun) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for (source_id, _replica), task in run.tasks_by_key.items():
        values[source_id].append(float(task["e2e_s"]))
    return {
        source_id: statistics.fmean(observations)
        for source_id, observations in sorted(values.items())
    }


def _task_components(
    run: ValidatedRun,
) -> dict[tuple[str, int], dict[str, float]]:
    """Reconstruct non-overlapping task-wall-clock components from raw rows."""

    rows: dict[tuple[str, int], dict[str, float]] = {}
    for key, task in run.tasks_by_key.items():
        task_id = str(task["task_id"])
        llm_s = _finite(task.get("llm_duration_s"), f"{task_id}.llm_duration_s")
        search = run.committed_by_task_tool[(task_id, "search")]
        visit = run.committed_by_task_tool[(task_id, "visit")]
        search_exposed_s = _finite(
            search.get("exposed_wait_s"), f"{task_id}.search.exposed_wait_s"
        )
        visit_exposed_s = _finite(
            visit.get("exposed_wait_s"), f"{task_id}.visit.exposed_wait_s"
        )
        e2e_s = _finite(task.get("e2e_s"), f"{task_id}.e2e_s")
        tool_exposed_s = search_exposed_s + visit_exposed_s
        residual_s = e2e_s - llm_s - tool_exposed_s
        if residual_s < -0.05:
            raise ValueError(
                f"negative E2E decomposition residual for {task_id}: {residual_s}"
            )
        rows[key] = {
            "e2e_s": e2e_s,
            "llm_s": llm_s,
            "tool_exposed_s": tool_exposed_s,
            "search_exposed_s": search_exposed_s,
            "visit_exposed_s": visit_exposed_s,
            "orchestration_residual_s": residual_s,
        }
    return rows


def _aggregate_source_components(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    cell: str,
) -> dict[str, dict[str, float]]:
    """Fold replicas inside block, then repeated fresh-server blocks."""

    by_block: dict[str, dict[str, dict[str, float]]] = {}
    for block_id in sorted(runs):
        observations: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (source_id, _replica), components in _task_components(
            runs[block_id][cell]
        ).items():
            for metric, value in components.items():
                observations[source_id][metric].append(value)
        by_block[block_id] = {
            source_id: {
                metric: statistics.fmean(values)
                for metric, values in sorted(metrics.items())
            }
            for source_id, metrics in sorted(observations.items())
        }
    source_sets = [set(values) for values in by_block.values()]
    if not source_sets or any(values != source_sets[0] for values in source_sets[1:]):
        raise ValueError(f"{cell} source identities differ across blocks")
    return {
        source_id: {
            metric: statistics.fmean(
                by_block[block_id][source_id][metric]
                for block_id in sorted(by_block)
            )
            for metric in sorted(next(iter(by_block.values()))[source_id])
        }
        for source_id in sorted(source_sets[0])
    }


def _ef_component_decomposition(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
) -> dict[str, Any]:
    by_cell = {
        cell: _aggregate_source_components(runs, cell) for cell in ("E", "F")
    }
    if set(by_cell["E"]) != set(by_cell["F"]):
        raise ValueError("E/F decomposition source identities differ")
    means = {
        cell: {
            metric: statistics.fmean(
                source[metric] for source in by_cell[cell].values()
            )
            for metric in next(iter(by_cell[cell].values()))
        }
        for cell in ("E", "F")
    }
    savings = {
        metric: means["E"][metric] - means["F"][metric]
        for metric in means["E"]
    }
    net_saving = savings["e2e_s"]
    tool_saving = savings["tool_exposed_s"]
    e_llm = means["E"]["llm_s"]
    return {
        "definition": (
            "per task E2E = sum(three LLM durations) + committed search/visit "
            "exposed waits + orchestration residual; replicas fold within block "
            "before the three block means fold within source"
        ),
        "source_count": len(by_cell["E"]),
        "source_components": by_cell,
        "mean_components_s": means,
        "mean_saving_E_minus_F_s": savings,
        "F_llm_component_speedup_fraction": (
            (e_llm - means["F"]["llm_s"]) / e_llm if e_llm else None
        ),
        "tool_exposed_wait_saving_to_net_e2e_saving_ratio": (
            tool_saving / net_saving if net_saving > 0.0 else None
        ),
    }


def _bootstrap_reductions(
    reductions: Mapping[str, float],
    baseline: Mapping[str, float] | None,
    candidate: Mapping[str, float] | None,
    *,
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    source_ids = sorted(reductions)
    if not source_ids:
        raise ValueError("cannot bootstrap an empty source set")
    rng = random.Random(seed)
    absolute: list[float] = []
    relative: list[float] = []
    for _ in range(resamples):
        sample = [source_ids[rng.randrange(len(source_ids))] for _ in source_ids]
        absolute.append(statistics.fmean(reductions[source] for source in sample))
        if baseline is not None and candidate is not None:
            base_mean = statistics.fmean(baseline[source] for source in sample)
            candidate_mean = statistics.fmean(candidate[source] for source in sample)
            relative.append(
                (base_mean - candidate_mean) / base_mean if base_mean else 0.0
            )
    result: dict[str, Any] = {
        "seed": seed,
        "resamples": resamples,
        "sampling_unit": "independent_source_mean_over_blocks_and_replicas",
        "sample_size": len(source_ids),
        "absolute_reduction_s_95_ci": [
            _percentile(absolute, 0.025),
            _percentile(absolute, 0.975),
        ],
    }
    if relative:
        result["relative_reduction_95_ci"] = [
            _percentile(relative, 0.025),
            _percentile(relative, 0.975),
        ]
    return result


def _gate(
    observed: Any,
    requirement: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def _validate_formal_metadata(
    run: ValidatedRun,
    *,
    expected_block_id: str,
    expected_cell_id: str,
) -> tuple[int, str]:
    config = run.config
    formal = _mapping(config.get("formal_run"), "config.formal_run")
    if formal.get("block_id") != expected_block_id:
        raise ValueError(
            f"{expected_block_id}/{expected_cell_id} formal block_id differs"
        )
    if formal.get("cell_id") != expected_cell_id:
        raise ValueError(
            f"{expected_block_id}/{expected_cell_id} formal cell_id differs"
        )
    order_index = formal.get("order_index")
    if isinstance(order_index, bool) or not isinstance(order_index, int):
        raise ValueError(
            f"{expected_block_id}/{expected_cell_id} order_index must be integer"
        )
    if order_index not in range(4):
        raise ValueError(
            f"{expected_block_id}/{expected_cell_id} order_index is outside [0, 3]"
        )
    server_instance_id = _nonempty(
        formal.get("server_instance_id"),
        f"{expected_block_id}/{expected_cell_id}.server_instance_id",
    )
    for field in ("fresh_server", "result_cache_empty", "broker_drained"):
        _strict_true(
            formal.get(field),
            f"{expected_block_id}/{expected_cell_id}.formal_run.{field}",
        )
    return order_index, server_instance_id


def _scheduler_environment(run: ValidatedRun, label: str) -> Mapping[str, Any]:
    return _mapping(run.config.get("scheduler_environment"), f"{label} scheduler")


def _validate_config_factorial(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    *,
    profile: FormalProfile,
) -> dict[str, Any]:
    flattened = [runs[block][cell] for block in sorted(runs) for cell in CELL_IDS]
    retry_policies = [
        _validate_retry_config(run.config, label="formal result config")
        for run in flattened
    ]
    if any(
        policy["max_attempts"] != CONTROLLED_HTTP_MAX_ATTEMPTS
        or policy["enabled"] is not True
        or not math.isclose(
            float(policy["backoff_s"]),
            CONTROLLED_HTTP_RETRY_BACKOFF_S,
            abs_tol=1e-12,
        )
        for policy in retry_policies
    ):
        raise ValueError(
            "formal cells must use the preregistered controlled HTTP retry policy"
        )
    template = {
        key: value
        for key, value in flattened[0].config.items()
        if key not in _COMMON_CONFIG_EXCLUSIONS
    }
    for run in flattened[1:]:
        common = {
            key: value
            for key, value in run.config.items()
            if key not in _COMMON_CONFIG_EXCLUSIONS
        }
        if common != template:
            differing = sorted(
                key
                for key in set(template) | set(common)
                if template.get(key) != common.get(key)
            )
            raise ValueError(
                "formal cells differ outside the two factorial treatments: "
                + ", ".join(differing)
            )

    reference_by_cell: dict[str, Mapping[str, Any]] = {}
    for block_id in sorted(runs):
        block = runs[block_id]
        if block["A"].config.get("speculation_mode") != "off":
            raise ValueError(f"{block_id}/A must use speculation_mode=off")
        if block["E"].config.get("speculation_mode") != "off":
            raise ValueError(f"{block_id}/E must use speculation_mode=off")
        b_mode = block["B"].config.get("speculation_mode")
        f_mode = block["F"].config.get("speculation_mode")
        allowed_speculation_modes = (
            {"visit"}
            if profile.require_modern_live_evidence
            else {"search", "visit", "search_visit"}
        )
        if b_mode not in allowed_speculation_modes or b_mode != f_mode:
            raise ValueError(f"{block_id} B/F speculation treatments differ")

        environments = {
            cell: _scheduler_environment(block[cell], f"{block_id}/{cell}")
            for cell in CELL_IDS
        }
        if environments["A"] != environments["B"]:
            raise ValueError(f"{block_id} A/B scheduler environments differ")
        if environments["E"] != environments["F"]:
            raise ValueError(f"{block_id} E/F scheduler environments differ")
        if environments["A"].get("VLLM_SCHED_POLICY") != "fcfs":
            raise ValueError(f"{block_id} A/B are not FCFS")
        if environments["E"].get("VLLM_SCHED_POLICY") != "online_joint_pacer_v2":
            raise ValueError(f"{block_id} E/F are not online_joint_pacer_v2")
        base_runtime = {
            key: value
            for key, value in environments["A"].items()
            if not key.startswith("VLLM_SCHED_")
        }
        joint_runtime = {
            key: value
            for key, value in environments["E"].items()
            if not key.startswith("VLLM_SCHED_")
        }
        if base_runtime != joint_runtime:
            raise ValueError(f"{block_id} non-scheduler server environments differ")
        for cell in CELL_IDS:
            previous = reference_by_cell.setdefault(cell, environments[cell])
            if environments[cell] != previous:
                raise ValueError(f"scheduler environment for cell {cell} differs by block")
    return {
        "same_non_factor_config_all_cells": True,
        "same_scheduler_environment_within_A_B_and_E_F": True,
        "same_cell_environment_across_blocks": True,
        "fcfs_policy": "fcfs",
        "joint_policy": "online_joint_pacer_v2",
        "speculation_mode": runs[sorted(runs)[0]]["B"].config["speculation_mode"],
        "controlled_http_retry_policy": retry_policies[0],
    }


def _validate_modern_runtime_contract(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    *,
    profile: FormalProfile,
) -> dict[str, Any]:
    """Bind modern results to the selected execution-aware live profile."""

    if not LIVE_AGENT_MODULE.is_file():
        raise ValueError(f"execution-aware policy module is missing: {LIVE_AGENT_MODULE}")
    current_policy_module_sha = _sha256_file(LIVE_AGENT_MODULE)
    policy_module_sha = (
        profile.expected_live_agent_sha256 or current_policy_module_sha
    )
    if (
        profile.require_current_live_agent_binding
        and current_policy_module_sha != policy_module_sha
    ):
        raise ValueError(
            f"{profile.name} current live_agent.py SHA differs from frozen binding"
        )
    current_broker_sha: str | None = None
    if profile.expected_live_broker_sha256 is not None:
        if not LIVE_BROKER_MODULE.is_file():
            raise ValueError(
                f"execution-aware broker module is missing: {LIVE_BROKER_MODULE}"
            )
        current_broker_sha = _sha256_file(LIVE_BROKER_MODULE)
        if (
            profile.require_current_live_broker_binding
            and current_broker_sha != profile.expected_live_broker_sha256
        ):
            raise ValueError(
                f"{profile.name} current live_broker.py SHA differs from frozen binding"
            )
    expected_config: dict[str, Any] = {
        "tool_signal_policy": "execution_aware",
        "tool_signal_policy_version": EXECUTION_AWARE_POLICY_VERSION,
        "tool_signal_policy_module_sha256": policy_module_sha,
        "min_speculative_tool_workers": profile.min_speculative_tool_workers,
        "search_tool_capacity": 3,
        "visit_tool_capacity": profile.visit_tool_capacity,
        "search_min_start_interval_s": 0.0,
        "visit_min_start_interval_s": profile.visit_min_start_interval_s,
        "visit_canary_stride": profile.visit_canary_stride,
        "tool_http_attempt_start_gate_enabled": True,
        "tool_http_attempt_start_gate_policy_version": (
            HTTP_ATTEMPT_GATE_POLICY_VERSION
        ),
        "tool_http_attempt_min_start_intervals_s": {
            "visit": profile.visit_min_start_interval_s
        },
    }
    if profile.require_strict_semantic_ascii_space_tail:
        expected_config.update(
            {
                "fixed_final_completion_tokens": V8_FIXED_FINAL_COMPLETION_TOKENS,
                "fixed_final_completion_enabled": True,
                "final_answer_contract_policy_version": (
                    V8_FIXED_FINAL_CONTRACT_POLICY_VERSION
                ),
                "final_answer_schema_policy_version": (
                    V8_FINAL_ANSWER_SCHEMA_POLICY_VERSION
                ),
                "final_answer_grammar_policy_version": (
                    V8_FINAL_GRAMMAR_POLICY_VERSION
                ),
                "final_answer_grammar_xgrammar_version": (
                    V8_FINAL_GRAMMAR_XGRAMMAR_VERSION
                ),
                "output_contract_policy_version": (
                    V8_OUTPUT_CONTRACT_POLICY_VERSION
                ),
                "live_agent_sha256": policy_module_sha,
                "tool_call_prompt_encoding": "canonical_json_sort_keys_compact",
                "token_count_method": "transformers_chat_template",
            }
        )
    for block_id in sorted(runs):
        for cell in CELL_IDS:
            run = runs[block_id][cell]
            changed = sorted(
                key
                for key, expected in expected_config.items()
                if run.config.get(key) != expected
            )
            if changed:
                raise ValueError(
                    f"{block_id}/{cell} {profile.name} runtime contract differs: "
                    + ", ".join(changed)
                )
            environment = _scheduler_environment(run, f"{block_id}/{cell}")
            expected_environment = {
                "VLLM_MAX_MODEL_LEN": str(profile.vllm_max_model_len),
                "VLLM_MAX_NUM_BATCHED_TOKENS": str(
                    profile.vllm_max_num_batched_tokens
                ),
                "VLLM_MAX_NUM_SEQS": str(profile.vllm_max_num_seqs),
                "VLLM_ENABLE_PREFIX_CACHING": "1",
            }
            environment_differences = sorted(
                key
                for key, expected in expected_environment.items()
                if environment.get(key) != expected
            )
            if environment_differences:
                raise ValueError(
                    f"{block_id}/{cell} {profile.name} vLLM profile differs: "
                    + ", ".join(environment_differences)
                )
            if cell in {"A", "B"}:
                leaked = sorted(
                    key
                    for key, value in environment.items()
                    if key.startswith("VLLM_SCHED_")
                    and key != "VLLM_SCHED_POLICY"
                    and value is not None
                )
                if leaked:
                    raise ValueError(
                        f"{block_id}/{cell} native FCFS leaked Joint knobs: "
                        + ", ".join(leaked)
                    )
            elif environment.get("VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY") != "0":
                raise ValueError(
                    f"{block_id}/{cell} {profile.name} explicit prefix locality "
                    "must be off"
                )
    return {
        "tool_signal_policy": "execution_aware",
        "tool_signal_policy_version": EXECUTION_AWARE_POLICY_VERSION,
        "tool_signal_policy_module_path": str(LIVE_AGENT_MODULE),
        "tool_signal_policy_module_sha256": policy_module_sha,
        "current_tool_signal_policy_module_sha256": current_policy_module_sha,
        "requires_current_module_sha_match": (
            profile.require_current_live_agent_binding
        ),
        "http_attempt_start_gate_policy_version": (
            HTTP_ATTEMPT_GATE_POLICY_VERSION
        ),
        "http_attempt_min_start_intervals_s": {
            "visit": profile.visit_min_start_interval_s
        },
        **(
            {
                "live_broker_module_path": str(LIVE_BROKER_MODULE),
                "live_broker_sha256": profile.expected_live_broker_sha256,
                "current_live_broker_sha256": current_broker_sha,
                "requires_current_live_broker_sha_match": (
                    profile.require_current_live_broker_binding
                ),
            }
            if profile.expected_live_broker_sha256 is not None
            else {}
        ),
        "canary_prediction_policy": "visit-skip-before-enqueue",
        "native_prefix_cache_enabled": True,
        "explicit_joint_prefix_locality_enabled": False,
    }


def _validate_formal_source_identity(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    *,
    profile: FormalProfile,
    workload_path: Path,
    workload_validation: Mapping[str, Any],
) -> dict[str, Any]:
    workload = _mapping(
        json.loads(workload_path.read_text(encoding="utf-8")),
        "formal workload",
    )
    expected_rows = {
        str(row["source_id"]): row for row in workload["sources"]
    }
    expected_source_ids = set(expected_rows)
    if len(expected_source_ids) != profile.source_count:
        raise ValueError(
            f"{profile.name} workload does not contain exactly "
            f"{profile.source_count} sources"
        )

    reference_task_keys: set[tuple[str, int]] | None = None
    reference_invocations: dict[tuple[str, int, str], str] = {}
    padding_actual_values: list[int] = []
    max_prompt_plus_output = 0
    expected_canary_sources = {
        str(row["source_id"])
        for index, row in enumerate(workload["sources"])
        if index % profile.visit_canary_stride == 0
    }
    if len(expected_canary_sources) != profile.expected_canary_count:
        raise ValueError(
            f"{profile.name} workload does not yield the expected canary count"
        )
    for block_id in sorted(runs):
        for cell in CELL_IDS:
            run = runs[block_id][cell]
            config = run.config
            if run.call_graph_mode != "frozen":
                raise ValueError(f"{block_id}/{cell} is not a frozen call graph")
            if config.get("workload_split_id") != workload_validation["split_id"]:
                raise ValueError(f"{block_id}/{cell} uses the wrong workload split")
            if config.get("workload_split_role") != "formal_heldout":
                raise ValueError(f"{block_id}/{cell} is not formal_heldout")
            _strict_true(
                config.get("workload_formal_eligible"),
                f"{block_id}/{cell}.workload_formal_eligible",
            )
            if config.get("workload_file_sha256") != workload_validation["file_sha256"]:
                raise ValueError(f"{block_id}/{cell} workload file SHA differs")
            if (
                config.get("selected_workload_sha256")
                != workload_validation["canonical_sources_sha256"]
            ):
                raise ValueError(f"{block_id}/{cell} selected workload SHA differs")
            if int(config.get("independent_source_count", -1)) != profile.source_count:
                raise ValueError(
                    f"{block_id}/{cell} source count must be {profile.source_count}"
                )
            if int(config.get("replicas", -1)) != profile.replicas_per_source:
                raise ValueError(
                    f"{block_id}/{cell} replicas must be "
                    f"{profile.replicas_per_source}"
                )
            if int(config.get("task_count", -1)) != profile.tasks_per_cell:
                raise ValueError(
                    f"{block_id}/{cell} task_count must be {profile.tasks_per_cell}"
                )
            if int(config.get("max_active_tasks", -1)) != profile.max_active_tasks:
                raise ValueError(
                    f"{block_id}/{cell} max_active_tasks must be "
                    f"{profile.max_active_tasks}"
                )
            if (
                int(config.get("context_padding_tokens", -1))
                != profile.context_padding_tokens
            ):
                raise ValueError(
                    f"{block_id}/{cell} context_padding_tokens must be "
                    f"{profile.context_padding_tokens}"
                )
            if int(config.get("visit_tool_capacity", -1)) != profile.visit_tool_capacity:
                raise ValueError(
                    f"{block_id}/{cell} visit_tool_capacity must be "
                    f"{profile.visit_tool_capacity}"
                )
            if int(config.get("visit_canary_stride", -1)) != profile.visit_canary_stride:
                raise ValueError(
                    f"{block_id}/{cell} visit_canary_stride must be "
                    f"{profile.visit_canary_stride}"
                )
            if not math.isclose(
                _finite(
                    config.get("visit_min_start_interval_s"),
                    f"{block_id}/{cell}.visit_min_start_interval_s",
                ),
                profile.visit_min_start_interval_s,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{block_id}/{cell} visit_min_start_interval_s must be "
                    f"{profile.visit_min_start_interval_s}"
                )
            if profile.require_modern_live_evidence and (
                int(config.get("min_speculative_tool_workers", -1))
                != profile.min_speculative_tool_workers
            ):
                raise ValueError(
                    f"{block_id}/{cell} min_speculative_tool_workers must be "
                    f"{profile.min_speculative_tool_workers}"
                )
            environment = _scheduler_environment(run, f"{block_id}/{cell}")
            try:
                max_num_seqs = int(environment.get("VLLM_MAX_NUM_SEQS"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{block_id}/{cell} lacks integer VLLM_MAX_NUM_SEQS"
                ) from exc
            if max_num_seqs != profile.vllm_max_num_seqs:
                raise ValueError(
                    f"{block_id}/{cell} VLLM_MAX_NUM_SEQS must be "
                    f"{profile.vllm_max_num_seqs}"
                )
            source_ids = {source for source, _ in run.tasks_by_key}
            if source_ids != expected_source_ids:
                raise ValueError(f"{block_id}/{cell} formal source IDs differ")
            keys = set(run.tasks_by_key)
            expected_keys = {
                (source_id, replica)
                for source_id in expected_source_ids
                for replica in range(profile.replicas_per_source)
            }
            if keys != expected_keys:
                raise ValueError(
                    f"{block_id}/{cell} must contain canonical r00 tasks only"
                )
            if reference_task_keys is None:
                reference_task_keys = keys
            elif keys != reference_task_keys:
                raise ValueError(f"{block_id}/{cell} source/replica identities differ")
            for key, task in run.tasks_by_key.items():
                expected = expected_rows[key[0]]
                expected_question_sha = hashlib.sha256(
                    str(expected["question"]).encode("utf-8")
                ).hexdigest()
                if task.get("question_sha256") != expected_question_sha:
                    raise ValueError(f"{block_id}/{cell}/{key[0]} question differs")
                if task.get("search_query") != expected["search_query"]:
                    raise ValueError(f"{block_id}/{cell}/{key[0]} search query differs")
                if task.get("expected_url") != expected["expected_url"]:
                    raise ValueError(f"{block_id}/{cell}/{key[0]} expected URL differs")
                target_padding = task.get("context_padding_target_tokens")
                actual_padding = task.get("context_padding_actual_tokens")
                if (
                    isinstance(target_padding, bool)
                    or not isinstance(target_padding, int)
                    or target_padding != profile.context_padding_tokens
                ):
                    raise ValueError(
                        f"{block_id}/{cell}/{task['task_id']} padding target differs"
                    )
                if (
                    isinstance(actual_padding, bool)
                    or not isinstance(actual_padding, int)
                    or not profile.context_padding_tokens
                    <= actual_padding
                    <= profile.context_padding_tokens
                    + FORMAL_CONTEXT_PADDING_MAX_OVERSHOOT
                ):
                    raise ValueError(
                        f"{block_id}/{cell}/{task['task_id']} padding actual is invalid"
                    )
                llm_events = run.llm_by_task[str(task["task_id"])]
                if any(
                    int(_mapping(event["usage"], "LLM usage")["prompt_tokens"])
                    < actual_padding
                    for event in llm_events
                ):
                    raise ValueError(
                        f"{block_id}/{cell}/{task['task_id']} prompt omits private padding"
                    )
                expected_canary = key[0] in expected_canary_sources
                if task.get("visit_canary") is not expected_canary:
                    raise ValueError(
                        f"{block_id}/{cell}/{task['task_id']} canary identity differs"
                    )
                if profile.vllm_max_model_len is not None:
                    for event in llm_events:
                        call_index = int(event["call_index"])
                        max_output = int(
                            config[
                                "max_tokens_answer"
                                if call_index == 2
                                else "max_tokens_tool"
                            ]
                        )
                        prompt_tokens = int(
                            _mapping(event["usage"], "LLM usage")["prompt_tokens"]
                        )
                        prompt_plus_output = prompt_tokens + max_output
                        max_prompt_plus_output = max(
                            max_prompt_plus_output, prompt_plus_output
                        )
                        if prompt_plus_output >= profile.vllm_max_model_len:
                            raise ValueError(
                                f"{block_id}/{cell}/{task['task_id']} LLM request "
                                "can exceed the frozen model context"
                            )
                padding_actual_values.append(actual_padding)
                for tool_name in ("search", "visit"):
                    digest = str(
                        run.committed_by_task_tool[(str(task["task_id"]), tool_name)][
                            "invocation_digest"
                        ]
                    )
                    identity_key = (key[0], key[1], tool_name)
                    reference = reference_invocations.setdefault(identity_key, digest)
                    if digest != reference:
                        raise ValueError(
                            f"{block_id}/{cell}/{task['task_id']} {tool_name} invocation differs"
                        )
    return {
        "source_count": profile.source_count,
        "task_identity_count": len(reference_task_keys or ()),
        "replicas_per_source": profile.replicas_per_source,
        "tasks_per_cell": profile.tasks_per_cell,
        "context_padding_target_tokens": profile.context_padding_tokens,
        "context_padding_actual_tokens": {
            "minimum": min(padding_actual_values),
            "maximum": max(padding_actual_values),
            "observation_count": len(padding_actual_values),
        },
        "split_id": workload_validation["split_id"],
        "formal_profile": profile.name,
        "workload_path": str(workload_path),
        "workload_file_sha256": workload_validation["file_sha256"],
        "workload_canonical_json_sha256": workload_validation[
            "canonical_json_sha256"
        ],
        "selected_workload_sha256": workload_validation[
            "canonical_sources_sha256"
        ],
        "frozen_question_query_expected_url_identity": True,
        "exact_search_and_visit_invocation_identity": True,
        "expected_canary_source_count": len(expected_canary_sources),
        "max_prompt_plus_configured_output_tokens": max_prompt_plus_output,
        "vllm_max_model_len": profile.vllm_max_model_len,
        "context_length_safe": (
            profile.vllm_max_model_len is None
            or max_prompt_plus_output < profile.vllm_max_model_len
        ),
    }


def _validate_http_attempt_log(
    record: Mapping[str, Any],
    *,
    label: str,
    started_at: float,
    finished_at: float,
    expected_attempts: int,
) -> tuple[list[float], float, float]:
    raw_log = record.get("http_attempt_log")
    if not isinstance(raw_log, list) or len(raw_log) != expected_attempts:
        raise ValueError(f"{label} lacks an exact physical HTTP-attempt ledger")
    starts: list[float] = []
    total_gate_wait_s = 0.0
    total_retry_backoff_s = 0.0
    attempt_by_request: dict[int, int] = defaultdict(int)
    for index, raw_entry in enumerate(raw_log):
        entry = _mapping(raw_entry, f"{label}.http_attempt_log[{index}]")
        request_index = entry.get("request_index")
        attempt = entry.get("attempt")
        if (
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or request_index < 0
        ):
            raise ValueError(f"{label} HTTP attempt {index} has invalid request_index")
        attempt_by_request[request_index] += 1
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt != attempt_by_request[request_index]
        ):
            raise ValueError(
                f"{label} HTTP attempt {index} is not contiguous within request"
            )
        started = _finite(
            entry.get("started_monotonic_s"),
            f"{label}.http_attempt_log[{index}].started_monotonic_s",
        )
        gate_wait = _finite(
            entry.get("start_gate_wait_s"),
            f"{label}.http_attempt_log[{index}].start_gate_wait_s",
        )
        retry_backoff = _finite(
            entry.get("retry_backoff_s"),
            f"{label}.http_attempt_log[{index}].retry_backoff_s",
        )
        if started + HTTP_ATTEMPT_SPACING_TOLERANCE_S < started_at:
            raise ValueError(f"{label} HTTP attempt {index} starts before its job")
        if started > finished_at + HTTP_ATTEMPT_SPACING_TOLERANCE_S:
            raise ValueError(f"{label} HTTP attempt {index} starts after its job")
        status = entry.get("status")
        error_type = entry.get("error_type")
        retried = entry.get("retried")
        if status is not None and (
            isinstance(status, bool) or not isinstance(status, int)
        ):
            raise ValueError(f"{label} HTTP attempt {index} status is invalid")
        if error_type is not None and (
            not isinstance(error_type, str) or not error_type
        ):
            raise ValueError(f"{label} HTTP attempt {index} error_type is invalid")
        if not isinstance(retried, bool):
            raise ValueError(f"{label} HTTP attempt {index} retried is not boolean")

        same_request_follows = any(
            isinstance(following, Mapping)
            and following.get("request_index") == request_index
            for following in raw_log[index + 1 :]
        )
        if same_request_follows:
            if (
                retried is not True
                or (
                    status not in CONTROLLED_HTTP_RETRYABLE_STATUSES
                    and error_type not in CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES
                )
                or retry_backoff + 0.05 < CONTROLLED_HTTP_RETRY_BACKOFF_S
            ):
                raise ValueError(
                    f"{label} HTTP attempt {index} violates controlled retry policy"
                )
        elif status != 200 or error_type is not None or retried is not False:
            raise ValueError(
                f"{label} HTTP request {request_index} lacks final successful HTTP 200"
            )
        elif retry_backoff != 0.0:
            raise ValueError(
                f"{label} final successful HTTP attempt reports retry backoff"
            )
        starts.append(started)
        total_gate_wait_s += gate_wait
        total_retry_backoff_s += retry_backoff
    if len(attempt_by_request) != 1:
        raise ValueError(f"{label} formal workload job must issue exactly one HTTP GET")
    return starts, total_gate_wait_s, total_retry_backoff_s


def _validate_zero_guided_json_recovery(
    run: ValidatedRun,
    label: str,
    *,
    expected_parsed_call_count: int,
) -> dict[str, Any]:
    recovered_task_count = 0
    parsed_call_count = 0
    for task_id, task in sorted(run.tasks_by_id.items()):
        recovery = _mapping(
            task.get("guided_json_recovery"),
            f"{label}/{task_id}.guided_json_recovery",
        )
        if recovery.get("policy_version") != GUIDED_JSON_RECOVERY_POLICY_VERSION:
            raise ValueError(
                f"{label}/{task_id} guided JSON recovery policy version differs"
            )
        raw_recovery_count = recovery.get("recovery_count")
        if (
            isinstance(raw_recovery_count, bool)
            or not isinstance(raw_recovery_count, int)
            or raw_recovery_count != 0
        ):
            raise ValueError(
                f"{label}/{task_id} guided JSON recovery_count must be zero"
            )
        raw_parsed_count = recovery.get("parsed_call_count")
        if (
            isinstance(raw_parsed_count, bool)
            or not isinstance(raw_parsed_count, int)
            or raw_parsed_count != expected_parsed_call_count
        ):
            raise ValueError(
                f"{label}/{task_id} guided JSON parsed_call_count must be "
                f"{expected_parsed_call_count}"
            )
        calls = recovery.get("calls")
        if (
            not isinstance(calls, list)
            or len(calls) != expected_parsed_call_count
        ):
            raise ValueError(
                f"{label}/{task_id} guided JSON call evidence must contain "
                f"{expected_parsed_call_count} rows"
            )
        for call_index, raw_call in enumerate(calls):
            call = _mapping(
                raw_call,
                f"{label}/{task_id}.guided_json_recovery.calls[{call_index}]",
            )
            observed_call_index = call.get("call_index")
            if (
                isinstance(observed_call_index, bool)
                or not isinstance(observed_call_index, int)
                or observed_call_index != call_index
                or call.get("policy_version")
                != GUIDED_JSON_RECOVERY_POLICY_VERSION
                or call.get("recovery_applied") is not False
                or call.get("parse_succeeded") is not True
            ):
                raise ValueError(
                    f"{label}/{task_id} guided JSON call {call_index} "
                    "is not strict-parse-only"
                )
        recovered_task_count += int(raw_recovery_count > 0)
        parsed_call_count += raw_parsed_count
    return {
        "task_count": len(run.tasks_by_id),
        "parsed_call_count": parsed_call_count,
        "recovery_count": 0,
        "recovered_task_count": recovered_task_count,
        "policy_version": GUIDED_JSON_RECOVERY_POLICY_VERSION,
        "expected_parsed_call_count_per_task": expected_parsed_call_count,
        "all_calls_parsed_without_recovery": True,
    }


def _validate_plain_final_output_contract(
    run: ValidatedRun,
    label: str,
) -> dict[str, Any]:
    guided_call_count = 0
    plain_final_call_count = 0
    canonicalization_changed_count = 0
    guided_output_keys = {
        "call_index",
        "mode",
        "guided_json_requested",
        "json_parse_attempted",
        "local_wrap_applied",
        "parse_succeeded",
        "contract_succeeded",
        "recovery_applied",
        "raw_sha256",
    }
    final_contract_keys = {
        "call_index",
        "policy_version",
        "mode",
        "guided_json_requested",
        "json_parse_attempted",
        "local_wrap_applied",
        "object_constructed_locally",
        "source_url_binding",
        "source_url_sha256",
        "contract_succeeded",
        "raw_sha256",
        "raw_char_count",
        "max_chars",
        "max_words",
        "canonical_sha256",
        "canonicalization_changed",
        "canonical_char_count",
        "canonical_word_count",
    }
    for task_id, task in sorted(run.tasks_by_id.items()):
        output_contract = _mapping(
            task.get("output_contract"), f"{label}/{task_id}.output_contract"
        )
        if set(output_contract) != {"policy_version", "calls"} or (
            output_contract.get("policy_version") != OUTPUT_CONTRACT_POLICY_VERSION
        ):
            raise ValueError(f"{label}/{task_id} output contract differs")
        output_calls = output_contract.get("calls")
        if not isinstance(output_calls, list) or len(output_calls) != 3:
            raise ValueError(
                f"{label}/{task_id} output contract must contain exactly three calls"
            )
        recovery = _mapping(
            task.get("guided_json_recovery"),
            f"{label}/{task_id}.guided_json_recovery",
        )
        recovery_calls = recovery.get("calls")
        if not isinstance(recovery_calls, list) or len(recovery_calls) != 2:
            raise ValueError(
                f"{label}/{task_id} must contain exactly two guided parse records"
            )
        llm_events = run.llm_by_task[task_id]
        for call_index in (0, 1):
            call = _mapping(
                output_calls[call_index],
                f"{label}/{task_id}.output_contract.calls[{call_index}]",
            )
            recovery_call = _mapping(
                recovery_calls[call_index],
                f"{label}/{task_id}.guided_json_recovery.calls[{call_index}]",
            )
            observed_call_index = call.get("call_index")
            if set(call) != guided_output_keys or any(
                (
                    isinstance(observed_call_index, bool),
                    not isinstance(observed_call_index, int),
                    observed_call_index != call_index,
                    call.get("mode") != "guided_json",
                    call.get("guided_json_requested") is not True,
                    call.get("json_parse_attempted") is not True,
                    call.get("local_wrap_applied") is not False,
                    call.get("parse_succeeded") is not True,
                    call.get("contract_succeeded") is not True,
                    call.get("recovery_applied") is not False,
                )
            ):
                raise ValueError(
                    f"{label}/{task_id} output call {call_index} is not exact guided JSON"
                )
            raw_sha = _lower_sha256(
                call.get("raw_sha256"),
                f"{label}/{task_id}.output_contract.calls[{call_index}].raw_sha256",
            )
            if recovery_call.get("raw_sha256") != raw_sha:
                raise ValueError(
                    f"{label}/{task_id} guided output/recovery SHA differs"
                )
            event = llm_events[call_index]
            if (
                event.get("output_mode") != "guided_json"
                or event.get("guided_json_requested") is not True
            ):
                raise ValueError(
                    f"{label}/{task_id} LLM event {call_index} output mode differs"
                )
            guided_call_count += 1

        final_contract = _mapping(
            task.get("final_answer_contract"),
            f"{label}/{task_id}.final_answer_contract",
        )
        output_final = _mapping(
            output_calls[2], f"{label}/{task_id}.output_contract.calls[2]"
        )
        if output_final != final_contract or set(final_contract) != final_contract_keys:
            raise ValueError(
                f"{label}/{task_id} final-answer/output contract evidence differs"
            )
        expected_scalars: Mapping[str, Any] = {
            "call_index": 2,
            "policy_version": FINAL_ANSWER_CONTRACT_POLICY_VERSION,
            "mode": "plain_text_local_wrap",
            "guided_json_requested": False,
            "json_parse_attempted": False,
            "local_wrap_applied": True,
            "object_constructed_locally": True,
            "source_url_binding": "exact_committed_selected_url",
            "contract_succeeded": True,
            "max_chars": FINAL_ANSWER_MAX_CHARS,
            "max_words": FINAL_ANSWER_MAX_WORDS,
        }
        changed = sorted(
            key
            for key, expected in expected_scalars.items()
            if final_contract.get(key) != expected
        )
        if changed:
            raise ValueError(
                f"{label}/{task_id} final-answer contract differs: "
                + ", ".join(changed)
            )
        selected_url = _nonempty(
            task.get("selected_url"), f"{label}/{task_id}.selected_url"
        )
        expected_url_sha = hashlib.sha256(selected_url.encode("utf-8")).hexdigest()
        if (
            _lower_sha256(
                final_contract.get("source_url_sha256"),
                f"{label}/{task_id}.final_answer_contract.source_url_sha256",
            )
            != expected_url_sha
        ):
            raise ValueError(
                f"{label}/{task_id} final-answer contract URL binding differs"
            )
        answer = _mapping(task.get("answer"), f"{label}/{task_id}.answer")
        answer_text = answer.get("answer")
        if not isinstance(answer_text, str):
            raise ValueError(f"{label}/{task_id} final answer text is not a string")
        words = answer_text.split(" ")
        if (
            not answer_text
            or answer_text != answer_text.strip()
            or "  " in answer_text
            or any(
                character.isspace() and character != " "
                for character in answer_text
            )
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in answer_text
            )
            or len(answer_text) > FINAL_ANSWER_MAX_CHARS
            or len(words) > FINAL_ANSWER_MAX_WORDS
            or "http://" in answer_text.lower()
            or "https://" in answer_text.lower()
            or answer.get("source_url") != selected_url
        ):
            raise ValueError(f"{label}/{task_id} final answer violates local-wrap bounds")
        canonical_sha = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        if (
            _lower_sha256(
                final_contract.get("canonical_sha256"),
                f"{label}/{task_id}.final_answer_contract.canonical_sha256",
            )
            != canonical_sha
        ):
            raise ValueError(
                f"{label}/{task_id} final-answer canonical SHA differs"
            )
        raw_sha = _lower_sha256(
            final_contract.get("raw_sha256"),
            f"{label}/{task_id}.final_answer_contract.raw_sha256",
        )
        raw_char_count = final_contract.get("raw_char_count")
        canonical_char_count = final_contract.get("canonical_char_count")
        canonical_word_count = final_contract.get("canonical_word_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                raw_char_count,
                canonical_char_count,
                canonical_word_count,
            )
        ):
            raise ValueError(
                f"{label}/{task_id} final-answer contract counts are invalid"
            )
        if (
            canonical_char_count != len(answer_text)
            or canonical_word_count != len(words)
            or raw_char_count < canonical_char_count
        ):
            raise ValueError(
                f"{label}/{task_id} final-answer contract counts differ"
            )
        canonicalization_changed = final_contract.get("canonicalization_changed")
        if (
            not isinstance(canonicalization_changed, bool)
            or canonicalization_changed != (raw_sha != canonical_sha)
            or (
                not canonicalization_changed
                and raw_char_count != canonical_char_count
            )
        ):
            raise ValueError(
                f"{label}/{task_id} final-answer canonicalization evidence differs"
            )
        final_event = llm_events[2]
        if (
            final_event.get("output_mode") != "plain_text"
            or final_event.get("guided_json_requested") is not False
        ):
            raise ValueError(f"{label}/{task_id} final LLM event is not plain text")
        plain_final_call_count += 1
        canonicalization_changed_count += int(canonicalization_changed)
    return {
        "task_count": len(run.tasks_by_id),
        "output_call_count": guided_call_count + plain_final_call_count,
        "guided_json_output_call_count": guided_call_count,
        "plain_text_local_wrap_call_count": plain_final_call_count,
        "exact_url_binding_count": plain_final_call_count,
        "contract_success_count": plain_final_call_count,
        "canonicalization_changed_count": canonicalization_changed_count,
        "output_contract_policy_version": OUTPUT_CONTRACT_POLICY_VERSION,
        "final_answer_contract_policy_version": (
            FINAL_ANSWER_CONTRACT_POLICY_VERSION
        ),
        "all_output_and_final_answer_contracts_exact": True,
    }


def _validate_strict_guided_final_output_contract(
    run: ValidatedRun,
    label: str,
) -> dict[str, Any]:
    guided_tool_call_count = 0
    guided_final_call_count = 0
    local_projection_count = 0
    word_projection_count = 0
    char_projection_count = 0
    guided_output_keys = {
        "call_index",
        "mode",
        "guided_json_requested",
        "json_parse_attempted",
        "local_wrap_applied",
        "parse_succeeded",
        "contract_succeeded",
        "recovery_applied",
        "raw_sha256",
    }
    guided_recovery_keys = guided_output_keys | {"policy_version"}
    final_contract_keys = {
        "call_index",
        "policy_version",
        "schema_policy_version",
        "schema_sha256",
        "schema_answer_constraint",
        "mode",
        "guided_json_requested",
        "json_parse_attempted",
        "strict_json_parse",
        "recovery_allowed",
        "recovery_applied",
        "parse_succeeded",
        "local_wrap_applied",
        "local_projection_applied",
        "object_constructed_locally",
        "source_url_binding",
        "source_url_sha256",
        "contract_succeeded",
        "raw_sha256",
        "raw_char_count",
        "max_chars",
        "max_words",
        "target_chars",
        "model_answer_sha256",
        "model_answer_char_count",
        "model_source_url_validated",
        "pre_projection_canonical_sha256",
        "pre_projection_char_count",
        "pre_projection_word_count",
        "canonical_sha256",
        "canonicalization_changed",
        "canonical_char_count",
        "canonical_word_count",
        "word_projection_applied",
        "char_projection_applied",
    }
    for task_id, task in sorted(run.tasks_by_id.items()):
        output_contract = _mapping(
            task.get("output_contract"), f"{label}/{task_id}.output_contract"
        )
        if set(output_contract) != {"policy_version", "calls"} or (
            output_contract.get("policy_version")
            != V7_OUTPUT_CONTRACT_POLICY_VERSION
        ):
            raise ValueError(f"{label}/{task_id} v7 output contract differs")
        output_calls = output_contract.get("calls")
        if not isinstance(output_calls, list) or len(output_calls) != 3:
            raise ValueError(
                f"{label}/{task_id} v7 output contract must contain three calls"
            )
        recovery = _mapping(
            task.get("guided_json_recovery"),
            f"{label}/{task_id}.guided_json_recovery",
        )
        recovery_calls = recovery.get("calls")
        if (
            set(recovery)
            != {"policy_version", "parsed_call_count", "recovery_count", "calls"}
            or not isinstance(recovery_calls, list)
            or len(recovery_calls) != 2
        ):
            raise ValueError(
                f"{label}/{task_id} v7 must contain two tool recovery records"
            )
        llm_events = run.llm_by_task[task_id]
        for call_index in (0, 1):
            call = _mapping(
                output_calls[call_index],
                f"{label}/{task_id}.output_contract.calls[{call_index}]",
            )
            recovery_call = _mapping(
                recovery_calls[call_index],
                f"{label}/{task_id}.guided_json_recovery.calls[{call_index}]",
            )
            observed_call_index = call.get("call_index")
            if set(call) != guided_output_keys or any(
                (
                    isinstance(observed_call_index, bool),
                    not isinstance(observed_call_index, int),
                    observed_call_index != call_index,
                    call.get("mode") != "guided_json",
                    call.get("guided_json_requested") is not True,
                    call.get("json_parse_attempted") is not True,
                    call.get("local_wrap_applied") is not False,
                    call.get("parse_succeeded") is not True,
                    call.get("contract_succeeded") is not True,
                    call.get("recovery_applied") is not False,
                )
            ):
                raise ValueError(
                    f"{label}/{task_id} v7 tool call {call_index} contract differs"
                )
            raw_sha = _lower_sha256(
                call.get("raw_sha256"),
                f"{label}/{task_id}.output_contract.calls[{call_index}].raw_sha256",
            )
            if (
                set(recovery_call) != guided_recovery_keys
                or recovery_call.get("policy_version")
                != GUIDED_JSON_RECOVERY_POLICY_VERSION
                or {
                    key: value
                    for key, value in recovery_call.items()
                    if key != "policy_version"
                }
                != call
                or recovery_call.get("raw_sha256") != raw_sha
            ):
                raise ValueError(
                    f"{label}/{task_id} v7 tool output/recovery telemetry differs"
                )
            event = llm_events[call_index]
            if (
                event.get("output_mode") != "guided_json"
                or event.get("guided_json_requested") is not True
            ):
                raise ValueError(
                    f"{label}/{task_id} v7 tool LLM output mode differs"
                )
            guided_tool_call_count += 1

        final_contract = _mapping(
            task.get("final_answer_contract"),
            f"{label}/{task_id}.final_answer_contract",
        )
        final_output = _mapping(
            output_calls[2], f"{label}/{task_id}.output_contract.calls[2]"
        )
        if final_output != final_contract or set(final_contract) != final_contract_keys:
            raise ValueError(
                f"{label}/{task_id} v7 final/output telemetry is not exactly mirrored"
            )
        expected_scalars: Mapping[str, Any] = {
            "call_index": 2,
            "policy_version": V7_FINAL_ANSWER_CONTRACT_POLICY_VERSION,
            "schema_policy_version": V7_FINAL_ANSWER_SCHEMA_POLICY_VERSION,
            "schema_answer_constraint": "type_only_no_length_or_pattern",
            "mode": "guided_json_strict_local_projection",
            "guided_json_requested": True,
            "json_parse_attempted": True,
            "strict_json_parse": True,
            "recovery_allowed": False,
            "recovery_applied": False,
            "parse_succeeded": True,
            "local_wrap_applied": True,
            "object_constructed_locally": True,
            "source_url_binding": "exact_committed_selected_url",
            "contract_succeeded": True,
            "max_chars": FINAL_ANSWER_MAX_CHARS,
            "max_words": FINAL_ANSWER_MAX_WORDS,
            "target_chars": FINAL_ANSWER_TARGET_CHARS,
            "model_source_url_validated": True,
        }
        differing = sorted(
            key
            for key, expected in expected_scalars.items()
            if final_contract.get(key) != expected
        )
        if differing:
            raise ValueError(
                f"{label}/{task_id} v7 strict final contract differs: "
                + ", ".join(differing)
            )
        selected_url = _nonempty(
            task.get("selected_url"), f"{label}/{task_id}.selected_url"
        )
        if task.get("expected_url") != selected_url:
            raise ValueError(f"{label}/{task_id} v7 selected/expected URL differs")
        answer = _mapping(task.get("answer"), f"{label}/{task_id}.answer")
        answer_text = answer.get("answer")
        if (
            set(answer) != {"answer", "source_url"}
            or answer.get("source_url") != selected_url
            or not isinstance(answer_text, str)
        ):
            raise ValueError(f"{label}/{task_id} v7 final answer URL binding differs")
        task_visit = _mapping(
            _mapping(task["tools"][1], f"{label}/{task_id}.visit").get(
                "invocation"
            ),
            f"{label}/{task_id}.visit.invocation",
        )
        visit_arguments = _mapping(
            task_visit.get("arguments"), f"{label}/{task_id}.visit.arguments"
        )
        if visit_arguments.get("url") != [selected_url]:
            raise ValueError(f"{label}/{task_id} v7 committed visit URL differs")
        expected_url_sha = hashlib.sha256(selected_url.encode("utf-8")).hexdigest()
        if (
            _lower_sha256(
                final_contract.get("source_url_sha256"),
                f"{label}/{task_id}.final_answer_contract.source_url_sha256",
            )
            != expected_url_sha
        ):
            raise ValueError(f"{label}/{task_id} v7 final URL SHA differs")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "source_url"],
            "properties": {
                "answer": {"type": "string"},
                "source_url": {"const": selected_url},
            },
        }
        if (
            _lower_sha256(
                final_contract.get("schema_sha256"),
                f"{label}/{task_id}.final_answer_contract.schema_sha256",
            )
            != _canonical_json_sha256(schema)
        ):
            raise ValueError(f"{label}/{task_id} v7 final schema SHA differs")
        sha_fields = {
            key: _lower_sha256(
                final_contract.get(key),
                f"{label}/{task_id}.final_answer_contract.{key}",
            )
            for key in (
                "raw_sha256",
                "model_answer_sha256",
                "pre_projection_canonical_sha256",
                "canonical_sha256",
            )
        }
        count_fields: dict[str, int] = {}
        for key in (
            "raw_char_count",
            "model_answer_char_count",
            "pre_projection_char_count",
            "pre_projection_word_count",
            "canonical_char_count",
            "canonical_word_count",
        ):
            value = final_contract.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label}/{task_id} v7 {key} is invalid")
            count_fields[key] = value
        canonical_sha = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        words = answer_text.split(" ")
        if (
            not answer_text
            or answer_text != answer_text.strip()
            or "  " in answer_text
            or any(
                character.isspace() and character != " "
                for character in answer_text
            )
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in answer_text
            )
            or len(answer_text) > FINAL_ANSWER_MAX_CHARS
            or len(words) > FINAL_ANSWER_MAX_WORDS
            or "http://" in answer_text.lower()
            or "https://" in answer_text.lower()
            or sha_fields["canonical_sha256"] != canonical_sha
            or count_fields["canonical_char_count"] != len(answer_text)
            or count_fields["canonical_word_count"] != len(words)
        ):
            raise ValueError(f"{label}/{task_id} v7 bounded answer evidence differs")
        if (
            count_fields["raw_char_count"] < count_fields["model_answer_char_count"]
            or count_fields["model_answer_char_count"]
            < count_fields["pre_projection_char_count"]
            or count_fields["pre_projection_char_count"]
            < count_fields["canonical_char_count"]
            or count_fields["pre_projection_word_count"]
            < count_fields["canonical_word_count"]
        ):
            raise ValueError(f"{label}/{task_id} v7 projection counts are not monotonic")
        canonicalization_changed = final_contract.get("canonicalization_changed")
        word_projection = final_contract.get("word_projection_applied")
        char_projection = final_contract.get("char_projection_applied")
        local_projection = final_contract.get("local_projection_applied")
        if not all(
            isinstance(value, bool)
            for value in (
                canonicalization_changed,
                word_projection,
                char_projection,
                local_projection,
            )
        ):
            raise ValueError(f"{label}/{task_id} v7 projection flags are invalid")
        if (
            canonicalization_changed
            != (
                sha_fields["model_answer_sha256"]
                != sha_fields["pre_projection_canonical_sha256"]
            )
            or (
                not canonicalization_changed
                and count_fields["model_answer_char_count"]
                != count_fields["pre_projection_char_count"]
            )
            or word_projection
            != (count_fields["pre_projection_word_count"] > FINAL_ANSWER_MAX_WORDS)
            or local_projection != (word_projection or char_projection)
            or (
                not word_projection
                and char_projection
                != (
                    count_fields["pre_projection_char_count"]
                    > FINAL_ANSWER_MAX_CHARS
                )
            )
        ):
            raise ValueError(f"{label}/{task_id} v7 projection invariants differ")
        if not local_projection:
            if (
                sha_fields["pre_projection_canonical_sha256"] != canonical_sha
                or count_fields["pre_projection_char_count"] != len(answer_text)
                or count_fields["pre_projection_word_count"] != len(words)
            ):
                raise ValueError(
                    f"{label}/{task_id} v7 no-projection evidence differs"
                )
        elif sha_fields["pre_projection_canonical_sha256"] == canonical_sha:
            raise ValueError(f"{label}/{task_id} v7 projection did not change answer")
        if word_projection and not char_projection and len(words) != FINAL_ANSWER_MAX_WORDS:
            raise ValueError(f"{label}/{task_id} v7 word projection count differs")
        final_event = llm_events[2]
        if (
            final_event.get("output_mode") != "guided_json"
            or final_event.get("guided_json_requested") is not True
        ):
            raise ValueError(f"{label}/{task_id} v7 final LLM call is not guided JSON")
        guided_final_call_count += 1
        local_projection_count += int(local_projection)
        word_projection_count += int(word_projection)
        char_projection_count += int(char_projection)
    return {
        "task_count": len(run.tasks_by_id),
        "output_call_count": guided_tool_call_count + guided_final_call_count,
        "guided_tool_call_count": guided_tool_call_count,
        "strict_guided_final_call_count": guided_final_call_count,
        "exact_url_binding_count": guided_final_call_count,
        "strict_parse_success_count": guided_final_call_count,
        "recovery_applied_count": 0,
        "local_projection_count": local_projection_count,
        "word_projection_count": word_projection_count,
        "char_projection_count": char_projection_count,
        "projection_allowed": True,
        "output_contract_policy_version": V7_OUTPUT_CONTRACT_POLICY_VERSION,
        "final_answer_contract_policy_version": (
            V7_FINAL_ANSWER_CONTRACT_POLICY_VERSION
        ),
        "final_answer_schema_policy_version": (
            V7_FINAL_ANSWER_SCHEMA_POLICY_VERSION
        ),
        "all_v7_contracts_exact": True,
    }


def _validate_fixed_final_completion_contract(
    run: ValidatedRun,
    label: str,
    *,
    expected_completion_tokens: int,
) -> dict[str, Any]:
    """Independently revalidate the complete v8 fixed-final wire contract."""

    if expected_completion_tokens <= 0:
        raise ValueError("expected fixed-final completion tokens must be positive")

    guided_output_keys = {
        "call_index",
        "mode",
        "guided_json_requested",
        "json_parse_attempted",
        "local_wrap_applied",
        "parse_succeeded",
        "contract_succeeded",
        "recovery_applied",
        "raw_sha256",
    }
    guided_recovery_keys = guided_output_keys | {"policy_version"}
    final_contract_keys = {
        "call_index",
        "policy_version",
        "schema_policy_version",
        "schema_sha256",
        "schema_answer_constraint",
        "mode",
        "guided_json_requested",
        "guided_grammar_requested",
        "json_parse_attempted",
        "strict_json_parse",
        "strict_json_raw_decode",
        "recovery_allowed",
        "recovery_applied",
        "parse_succeeded",
        "local_wrap_applied",
        "local_projection_applied",
        "object_constructed_locally",
        "source_url_binding",
        "source_url_sha256",
        "contract_succeeded",
        "raw_sha256",
        "raw_char_count",
        "max_chars",
        "max_words",
        "target_chars",
        "grammar_policy_version",
        "grammar_xgrammar_version",
        "grammar_sha256",
        "grammar_semantic_json_whitespace",
        "tail_policy",
        "tail_validation_succeeded",
        "fixed_completion_tokens",
        "min_tokens",
        "max_tokens",
        "total_completion_tokens",
        "finish_reason",
        "finish_reason_validated",
        "token_accounting_succeeded",
        "semantic_sha256",
        "semantic_char_count",
        "semantic_byte_count",
        "padding_sha256",
        "padding_char_count",
        "padding_byte_count",
        "tail_nonempty",
        "tail_ascii_space_only",
        "token_counter_method",
        "semantic_token_count",
        "padding_token_count",
        "token_partition_method",
        "model_answer_sha256",
        "model_answer_char_count",
        "model_source_url_validated",
        "pre_projection_canonical_sha256",
        "pre_projection_char_count",
        "pre_projection_word_count",
        "canonical_sha256",
        "canonicalization_changed",
        "canonical_char_count",
        "canonical_word_count",
        "word_projection_applied",
        "char_projection_applied",
    }

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    mismatch_task_ids: list[str] = []
    semantic_char_counts: list[int] = []
    tail_char_counts: list[int] = []
    final_completion_tokens: list[int] = []
    projection_count = 0
    for task_id, task in sorted(run.tasks_by_id.items()):
        llm_events = run.llm_by_task[task_id]
        if [int(event.get("call_index", -1)) for event in llm_events] != [0, 1, 2]:
            raise ValueError(f"{label}/{task_id} LLM calls are not exactly 0,1,2")
        recovery = _mapping(
            task.get("guided_json_recovery"),
            f"{label}/{task_id}.guided_json_recovery",
        )
        recovery_calls = recovery.get("calls")
        output_contract = _mapping(
            task.get("output_contract"), f"{label}/{task_id}.output_contract"
        )
        output_calls = output_contract.get("calls")
        if (
            set(output_contract) != {"policy_version", "calls"}
            or output_contract.get("policy_version")
            != V8_OUTPUT_CONTRACT_POLICY_VERSION
            or not isinstance(output_calls, list)
            or len(output_calls) != 3
            or not isinstance(recovery_calls, list)
            or len(recovery_calls) != 2
        ):
            raise ValueError(f"{label}/{task_id} v8 output contract differs")
        for call_index in (0, 1):
            call = _mapping(
                output_calls[call_index],
                f"{label}/{task_id}.output_contract.calls[{call_index}]",
            )
            recovery_call = _mapping(
                recovery_calls[call_index],
                f"{label}/{task_id}.guided_json_recovery.calls[{call_index}]",
            )
            expected_guided = {
                "call_index": call_index,
                "mode": "guided_json",
                "guided_json_requested": True,
                "json_parse_attempted": True,
                "local_wrap_applied": False,
                "parse_succeeded": True,
                "contract_succeeded": True,
                "recovery_applied": False,
            }
            if set(call) != guided_output_keys or any(
                call.get(key) != value for key, value in expected_guided.items()
            ):
                raise ValueError(
                    f"{label}/{task_id} v8 tool call {call_index} contract differs"
                )
            _lower_sha256(
                call.get("raw_sha256"),
                f"{label}/{task_id}.output_contract.calls[{call_index}].raw_sha256",
            )
            if (
                set(recovery_call) != guided_recovery_keys
                or recovery_call.get("policy_version")
                != GUIDED_JSON_RECOVERY_POLICY_VERSION
                or {
                    key: value
                    for key, value in recovery_call.items()
                    if key != "policy_version"
                }
                != call
            ):
                raise ValueError(
                    f"{label}/{task_id} v8 tool output/recovery telemetry differs"
                )
            tool_event = llm_events[call_index]
            if (
                tool_event.get("output_mode") != "guided_json"
                or tool_event.get("guided_json_requested") is not True
                or tool_event.get("guided_grammar_requested") is not False
                or tool_event.get("guided_grammar_sha256") is not None
                or tool_event.get("min_tokens") != 0
            ):
                raise ValueError(
                    f"{label}/{task_id} v8 tool LLM event contract differs"
                )

        final_contract = _mapping(
            task.get("final_answer_contract"),
            f"{label}/{task_id}.final_answer_contract",
        )
        if (
            set(final_contract) != final_contract_keys
            or _mapping(output_calls[2], f"{label}/{task_id}.output.calls[2]")
            != final_contract
        ):
            raise ValueError(
                f"{label}/{task_id} v8 final/output telemetry is not exactly mirrored"
            )
        expected_scalars: Mapping[str, Any] = {
            "call_index": 2,
            "policy_version": V8_FIXED_FINAL_CONTRACT_POLICY_VERSION,
            "schema_policy_version": V8_FINAL_ANSWER_SCHEMA_POLICY_VERSION,
            "schema_answer_constraint": "type_only_no_length_or_pattern",
            "mode": "guided_grammar_fixed_completion_strict_raw_decode_local_projection",
            "guided_json_requested": False,
            "guided_grammar_requested": True,
            "json_parse_attempted": True,
            "strict_json_parse": True,
            "strict_json_raw_decode": True,
            "recovery_allowed": False,
            "recovery_applied": False,
            "parse_succeeded": True,
            "local_wrap_applied": True,
            "object_constructed_locally": True,
            "source_url_binding": "exact_committed_selected_url",
            "contract_succeeded": True,
            "max_chars": FINAL_ANSWER_MAX_CHARS,
            "max_words": FINAL_ANSWER_MAX_WORDS,
            "target_chars": FINAL_ANSWER_TARGET_CHARS,
            "grammar_policy_version": V8_FINAL_GRAMMAR_POLICY_VERSION,
            "grammar_xgrammar_version": V8_FINAL_GRAMMAR_XGRAMMAR_VERSION,
            "grammar_semantic_json_whitespace": "compact",
            "tail_policy": "one_or_more_ascii_spaces_only",
            "tail_validation_succeeded": True,
            "tail_nonempty": True,
            "tail_ascii_space_only": True,
            "fixed_completion_tokens": expected_completion_tokens,
            "min_tokens": expected_completion_tokens,
            "max_tokens": expected_completion_tokens,
            "finish_reason": "length",
            "finish_reason_validated": True,
            "token_accounting_succeeded": True,
            "token_partition_method": (
                "server_total_minus_local_semantic_tokenization"
            ),
            "token_counter_method": "transformers_chat_template",
            "model_source_url_validated": True,
        }
        differing = sorted(
            key
            for key, expected in expected_scalars.items()
            if final_contract.get(key) != expected
        )
        if differing:
            raise ValueError(
                f"{label}/{task_id} v8 fixed-final contract differs: "
                + ", ".join(differing)
            )

        selected_url = _nonempty(
            task.get("selected_url"), f"{label}/{task_id}.selected_url"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "source_url"],
            "properties": {
                "answer": {"type": "string"},
                "source_url": {"const": selected_url},
            },
        }
        expected_schema_sha = _canonical_json_sha256(schema)
        expected_grammar_sha = _fixed_final_grammar_sha256(selected_url)
        expected_url_sha = hashlib.sha256(selected_url.encode("utf-8")).hexdigest()
        if (
            _lower_sha256(
                final_contract.get("schema_sha256"),
                f"{label}/{task_id}.final.schema_sha256",
            )
            != expected_schema_sha
            or _lower_sha256(
                final_contract.get("grammar_sha256"),
                f"{label}/{task_id}.final.grammar_sha256",
            )
            != expected_grammar_sha
            or _lower_sha256(
                final_contract.get("source_url_sha256"),
                f"{label}/{task_id}.final.source_url_sha256",
            )
            != expected_url_sha
        ):
            raise ValueError(f"{label}/{task_id} v8 schema/grammar/URL SHA differs")

        event = llm_events[2]
        usage = _mapping(event.get("usage"), f"{label}/{task_id}.call2.usage")
        raw_tokens = usage.get("completion_tokens")
        if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int):
            raise ValueError(
                f"{label}/{task_id} call-2 completion_tokens is not an integer"
            )
        if (
            event.get("output_mode") != "guided_grammar"
            or event.get("guided_json_requested") is not False
            or event.get("guided_grammar_requested") is not True
            or event.get("guided_grammar_sha256") != expected_grammar_sha
            or event.get("min_tokens") != expected_completion_tokens
            or event.get("max_tokens") != expected_completion_tokens
            or event.get("finish_reason") != "length"
        ):
            raise ValueError(f"{label}/{task_id} call-2 LLM event contract differs")
        final_completion_tokens.append(raw_tokens)
        if raw_tokens != expected_completion_tokens:
            mismatch_task_ids.append(task_id)

        response = event.get("response")
        if not isinstance(response, str) or not response:
            raise ValueError(f"{label}/{task_id} call-2 response is empty or absent")
        if event.get("response_sha256") != hashlib.sha256(
            response.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"{label}/{task_id} call-2 response SHA differs")
        try:
            semantic, semantic_end = decoder.raw_decode(response)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"{label}/{task_id} call-2 semantic JSON is not strict"
            ) from exc
        semantic_wire = response[:semantic_end]
        if _json_has_whitespace_outside_strings(semantic_wire):
            raise ValueError(
                f"{label}/{task_id} call-2 semantic JSON is not compact"
            )
        if (
            not isinstance(semantic, Mapping)
            or set(semantic) != {"answer", "source_url"}
            or not isinstance(semantic.get("answer"), str)
            or not semantic["answer"]
            or semantic.get("source_url") != selected_url
        ):
            raise ValueError(
                f"{label}/{task_id} call-2 semantic JSON fields differ"
            )
        tail = response[semantic_end:]
        if not tail or any(character != " " for character in tail):
            raise ValueError(
                f"{label}/{task_id} call-2 tail is not non-empty ASCII spaces only"
            )
        semantic_sha = hashlib.sha256(semantic_wire.encode("utf-8")).hexdigest()
        padding_sha = hashlib.sha256(tail.encode("utf-8")).hexdigest()
        integer_evidence = {
            "raw_char_count": len(response),
            "semantic_char_count": len(semantic_wire),
            "semantic_byte_count": len(semantic_wire.encode("utf-8")),
            "padding_char_count": len(tail),
            "padding_byte_count": len(tail.encode("utf-8")),
            "total_completion_tokens": raw_tokens,
        }
        if any(
            isinstance(final_contract.get(key), bool)
            or not isinstance(final_contract.get(key), int)
            or final_contract.get(key) != expected
            for key, expected in integer_evidence.items()
        ):
            raise ValueError(f"{label}/{task_id} v8 final count evidence differs")
        if (
            final_contract.get("raw_sha256")
            != hashlib.sha256(response.encode("utf-8")).hexdigest()
            or final_contract.get("semantic_sha256") != semantic_sha
            or final_contract.get("padding_sha256") != padding_sha
        ):
            raise ValueError(f"{label}/{task_id} v8 raw semantic/padding SHA differs")
        semantic_tokens = final_contract.get("semantic_token_count")
        padding_tokens = final_contract.get("padding_token_count")
        if (
            isinstance(semantic_tokens, bool)
            or not isinstance(semantic_tokens, int)
            or semantic_tokens <= 0
            or isinstance(padding_tokens, bool)
            or not isinstance(padding_tokens, int)
            or padding_tokens <= 0
            or semantic_tokens + padding_tokens != raw_tokens
        ):
            raise ValueError(f"{label}/{task_id} v8 token partition differs")

        raw_answer = str(semantic["answer"])
        canonical = " ".join(raw_answer.split())
        projected, word_projection, char_projection = _bounded_answer_prefix(
            canonical
        )
        answer = _mapping(task.get("answer"), f"{label}/{task_id}.answer")
        if (
            not projected
            or set(answer) != {"answer", "source_url"}
            or answer.get("answer") != projected
            or answer.get("source_url") != selected_url
        ):
            raise ValueError(f"{label}/{task_id} v8 projected answer differs")
        projection_scalars: Mapping[str, Any] = {
            "model_answer_sha256": hashlib.sha256(
                raw_answer.encode("utf-8")
            ).hexdigest(),
            "model_answer_char_count": len(raw_answer),
            "pre_projection_canonical_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
            "pre_projection_char_count": len(canonical),
            "pre_projection_word_count": len(canonical.split(" ")),
            "canonical_sha256": hashlib.sha256(projected.encode("utf-8")).hexdigest(),
            "canonicalization_changed": canonical != raw_answer,
            "canonical_char_count": len(projected),
            "canonical_word_count": len(projected.split(" ")),
            "word_projection_applied": word_projection,
            "char_projection_applied": char_projection,
            "local_projection_applied": word_projection or char_projection,
        }
        if any(
            final_contract.get(key) != expected
            for key, expected in projection_scalars.items()
        ):
            raise ValueError(f"{label}/{task_id} v8 projection evidence differs")
        if (
            len(projected) > FINAL_ANSWER_MAX_CHARS
            or len(projected.split(" ")) > FINAL_ANSWER_MAX_WORDS
            or any(character.isspace() and character != " " for character in projected)
            or "  " in projected
            or projected != projected.strip()
            or "http://" in projected.lower()
            or "https://" in projected.lower()
        ):
            raise ValueError(f"{label}/{task_id} v8 projected answer is invalid")
        projection_count += int(word_projection or char_projection)
        semantic_char_counts.append(len(semantic_wire))
        tail_char_counts.append(len(tail))

    return {
        "task_count": len(run.tasks_by_id),
        "expected_completion_tokens_per_task": expected_completion_tokens,
        "exact_completion_token_task_count": (
            len(run.tasks_by_id) - len(mismatch_task_ids)
        ),
        "completion_token_mismatch_count": len(mismatch_task_ids),
        "completion_token_mismatch_task_ids": mismatch_task_ids,
        "completion_tokens": _distribution(
            [float(value) for value in final_completion_tokens]
        ),
        "semantic_json_object_count": len(semantic_char_counts),
        "ascii_space_tail_count": len(tail_char_counts),
        "semantic_char_count": _distribution(
            [float(value) for value in semantic_char_counts]
        ),
        "ascii_space_tail_char_count": _distribution(
            [float(value) for value in tail_char_counts]
        ),
        "local_projection_count": projection_count,
        "output_contract_policy_version": V8_OUTPUT_CONTRACT_POLICY_VERSION,
        "final_answer_contract_policy_version": (
            V8_FIXED_FINAL_CONTRACT_POLICY_VERSION
        ),
        "final_answer_schema_policy_version": (
            V8_FINAL_ANSWER_SCHEMA_POLICY_VERSION
        ),
        "final_answer_grammar_policy_version": V8_FINAL_GRAMMAR_POLICY_VERSION,
        "final_answer_grammar_xgrammar_version": (
            V8_FINAL_GRAMMAR_XGRAMMAR_VERSION
        ),
        "all_semantic_json_and_ascii_space_tails_exact": True,
        "all_completion_tokens_exact": not mismatch_task_ids,
    }


def _validate_physical_run(
    run: ValidatedRun,
    label: str,
    *,
    require_http_attempt_logs: bool = False,
) -> dict[str, Any]:
    records = list(run.physical_records)
    running_events: list[tuple[float, int, bool, str]] = []
    uncontrolled_retries = 0
    failed_physical_jobs = 0
    started_physical_jobs = 0
    physical_http_attempts = 0
    retried_physical_jobs = 0
    authoritative_commits = 0
    authoritative_retried_commits = 0
    rejected_decisions = 0
    speculative_worker_s = 0.0
    wasted_speculative_worker_s = 0.0
    starts_by_tool: dict[str, list[float]] = defaultdict(list)
    intervals_by_worker: dict[int, list[tuple[float, float]]] = defaultdict(list)
    http_attempt_starts_by_tool: dict[str, list[float]] = defaultdict(list)
    http_attempt_gate_wait_s = 0.0
    http_retry_backoff_s = 0.0
    configured_intervals = {
        "search": _finite(
            run.config.get("search_min_start_interval_s", 0.0),
            f"{label}.search_min_start_interval_s",
        ),
        "visit": _finite(
            run.config.get("visit_min_start_interval_s", 0.0),
            f"{label}.visit_min_start_interval_s",
        ),
    }
    for index, record in enumerate(records):
        admitted = record.get("admitted") is True
        if not admitted:
            rejected_decisions += 1
            continue
        outcome = str(record.get("outcome"))
        if "failed" in outcome:
            failed_physical_jobs += 1
        started_raw = record.get("started_at")
        if started_raw is None:
            queued = _finite(
                record.get("queue_enter_at"),
                f"{label}.tool[{index}].queue_enter_at",
            )
            finished = _finite(
                record.get("finished_at"),
                f"{label}.tool[{index}].finished_at",
            )
            queue_s = _finite(
                record.get("queue_s"), f"{label}.tool[{index}].queue_s"
            )
            service_s = _finite(
                record.get("service_s"), f"{label}.tool[{index}].service_s"
            )
            saved_s = _finite(
                record.get("saved_service_s"),
                f"{label}.tool[{index}].saved_service_s",
            )
            attempts = record.get("http_attempts")
            if (
                record.get("speculative") is not True
                or record.get("committed") is not False
                or record.get("cancelled") is not True
                or record.get("outcome") not in {"cancelled", "expired"}
                or record.get("worker_id") is not None
                or isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts != 0
                or finished < queued
                or not math.isclose(
                    queue_s,
                    finished - queued,
                    rel_tol=0.02,
                    abs_tol=0.01,
                )
                or service_s != 0.0
                or saved_s != 0.0
                or any(
                    record.get(field) is not None
                    for field in (
                        "backend",
                        "request_host",
                        "response_status",
                        "bytes_read",
                        "transport_identity_source",
                    )
                )
            ):
                raise ValueError(
                    f"{label} tool record {index} has invalid never-started "
                    "cancellation telemetry"
                )
            if require_http_attempt_logs:
                raw_attempt_log = record.get("http_attempt_log")
                if raw_attempt_log is not None and raw_attempt_log != []:
                    raise ValueError(
                        f"{label} tool record {index} never started but has HTTP attempts"
                    )
            continue
        started = _finite(started_raw, f"{label}.tool[{index}].started_at")
        finished = _finite(record.get("finished_at"), f"{label}.tool[{index}].finished_at")
        queued = _finite(record.get("queue_enter_at"), f"{label}.tool[{index}].queue_enter_at")
        if not queued <= started <= finished:
            raise ValueError(f"{label} tool record {index} timestamps are not monotonic")
        queue_s = _finite(record.get("queue_s"), f"{label}.tool[{index}].queue_s")
        if not math.isclose(queue_s, started - queued, rel_tol=0.02, abs_tol=0.01):
            raise ValueError(f"{label} tool record {index} queue duration differs")
        attempts = record.get("http_attempts")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
            or attempts > CONTROLLED_HTTP_MAX_ATTEMPTS
        ):
            raise ValueError(f"{label} started tool record {index} has invalid HTTP attempts")
        if record.get("transport_identity_source") != "actual":
            raise ValueError(
                f"{label} started tool record {index} lacks actual final HTTP evidence"
            )
        if record.get("response_status") != 200:
            raise ValueError(
                f"{label} started tool record {index} final response is not HTTP 200"
            )
        bytes_read = record.get("bytes_read")
        if (
            isinstance(bytes_read, bool)
            or not isinstance(bytes_read, int)
            or bytes_read <= 0
        ):
            raise ValueError(
                f"{label} started tool record {index} has invalid response bytes"
            )
        started_physical_jobs += 1
        physical_http_attempts += attempts
        retried_physical_jobs += attempts > 1
        if record.get("committed") is True:
            authoritative_commits += 1
            authoritative_retried_commits += attempts > 1
        speculative = record.get("speculative") is True
        # A queued prediction promoted before it starts runs in the
        # authoritative lane and therefore does not consume the speculative
        # worker reservation.  Inflight promotions and completed reuse did
        # physically start as speculative work.
        physically_speculative = (
            speculative and record.get("source") != "promoted_from_queue"
        )
        tool_name = str(record.get("tool"))
        if require_http_attempt_logs:
            attempt_starts, gate_wait_s, retry_backoff_s = _validate_http_attempt_log(
                record,
                label=f"{label}.tool[{index}]",
                started_at=started,
                finished_at=finished,
                expected_attempts=attempts,
            )
            http_attempt_starts_by_tool[tool_name].extend(attempt_starts)
            http_attempt_gate_wait_s += gate_wait_s
            http_retry_backoff_s += retry_backoff_s
        service_s = _finite(record.get("service_s"), f"{label}.tool[{index}].service_s")
        if (
            attempts > 1
            and service_s + 0.01 < CONTROLLED_HTTP_RETRY_BACKOFF_S
        ):
            raise ValueError(
                f"{label} tool record {index} service time omits retry backoff"
            )
        worker_id = record.get("worker_id")
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or worker_id not in range(int(run.config["tool_workers"]))
        ):
            raise ValueError(f"{label} tool record {index} has invalid worker_id")
        intervals_by_worker[worker_id].append((started, finished))
        worker_pool = _mapping(
            record.get("worker_pool"), f"{label}.tool[{index}].worker_pool"
        )
        if int(worker_pool.get("max_workers", -1)) != int(run.config["tool_workers"]):
            raise ValueError(f"{label} tool record {index} worker pool size differs")
        if int(worker_pool.get("max_speculative_workers", -1)) != int(
            run.config["speculative_tool_workers"]
        ):
            raise ValueError(f"{label} tool record {index} speculative pool size differs")
        expected_tool_capacities = {
            name: int(run.config[f"{name}_tool_capacity"])
            for name in ("search", "visit")
            if int(run.config.get(f"{name}_tool_capacity", 0)) > 0
        }
        if worker_pool.get("tool_capacities") != expected_tool_capacities:
            raise ValueError(f"{label} tool record {index} per-tool capacities differ")
        expected_intervals = {
            name: value
            for name, value in configured_intervals.items()
            if value > 0.0
        }
        if worker_pool.get("tool_min_start_intervals_s") != expected_intervals:
            raise ValueError(f"{label} tool record {index} start-rate config differs")
        configured_tool_capacity = expected_tool_capacities.get(
            tool_name, int(run.config["tool_workers"])
        )
        if int(record.get("tool_capacity", -1)) != configured_tool_capacity:
            raise ValueError(f"{label} tool record {index} tool capacity differs")
        running_events.extend(
            [
                (started, 1, physically_speculative, tool_name),
                (finished, -1, physically_speculative, tool_name),
            ]
        )
        starts_by_tool[tool_name].append(started)
        if physically_speculative:
            speculative_worker_s += service_s
            if record.get("committed") is not True:
                wasted_speculative_worker_s += service_s
        interval = configured_intervals.get(tool_name, 0.0)
        observed_interval = _finite(
            record.get("tool_min_start_interval_s", 0.0),
            f"{label}.tool[{index}].tool_min_start_interval_s",
        )
        if not math.isclose(observed_interval, interval, abs_tol=1e-6):
            raise ValueError(f"{label} tool record {index} interval telemetry differs")
        if interval > 0.0:
            next_eligible = _finite(
                record.get("rate_limit_next_eligible_at"),
                f"{label}.tool[{index}].rate_limit_next_eligible_at",
            )
            if not math.isclose(next_eligible, started + interval, abs_tol=0.01):
                raise ValueError(f"{label} tool record {index} next-eligible telemetry differs")

    running = running_speculative = 0
    running_by_tool: dict[str, int] = defaultdict(int)
    maxima_by_tool: dict[str, int] = defaultdict(int)
    max_running = max_running_speculative = 0
    # Finish before start at an equal timestamp.
    for _timestamp, delta, speculative, tool_name in sorted(
        running_events, key=lambda event: (event[0], event[1])
    ):
        running += delta
        running_by_tool[tool_name] += delta
        if speculative:
            running_speculative += delta
        max_running = max(max_running, running)
        max_running_speculative = max(max_running_speculative, running_speculative)
        maxima_by_tool[tool_name] = max(
            maxima_by_tool[tool_name], running_by_tool[tool_name]
        )
        if min(running, running_speculative, *running_by_tool.values()) < 0:
            raise ValueError(f"{label} physical concurrency reconstruction went negative")
    worker_capacity = int(run.config["tool_workers"])
    speculative_capacity = int(run.config["speculative_tool_workers"])
    if max_running > worker_capacity or max_running_speculative > speculative_capacity:
        raise ValueError(f"{label} exceeded shared physical worker capacity")
    for tool_name in ("search", "visit"):
        configured = int(run.config.get(f"{tool_name}_tool_capacity", 0))
        capacity = configured or worker_capacity
        if maxima_by_tool.get(tool_name, 0) > capacity:
            raise ValueError(f"{label} exceeded {tool_name} physical capacity")
        interval = configured_intervals[tool_name]
        starts = sorted(starts_by_tool.get(tool_name, ()))
        for left, right in zip(starts, starts[1:]):
            if right - left + 0.01 < interval:
                raise ValueError(f"{label} violated {tool_name} minimum start interval")
        if require_http_attempt_logs:
            http_starts = sorted(http_attempt_starts_by_tool.get(tool_name, ()))
            for left, right in zip(http_starts, http_starts[1:]):
                if right - left + HTTP_ATTEMPT_SPACING_TOLERANCE_S < interval:
                    raise ValueError(
                        f"{label} violated {tool_name} physical HTTP-attempt start gate"
                    )
    for worker_id, intervals in intervals_by_worker.items():
        ordered = sorted(intervals)
        for previous, following in zip(ordered, ordered[1:]):
            if following[0] + 0.01 < previous[1]:
                raise ValueError(f"{label} worker {worker_id} executed overlapping jobs")

    speculation_mode = str(run.config["speculation_mode"])
    speculative_records = sum(record.get("speculative") is True for record in records)
    if speculation_mode == "off" and speculative_records:
        raise ValueError(f"{label} spec-off cell contains speculative work")
    enabled_speculative_tools = {
        "search": {"search"},
        "visit": {"visit"},
        "search_visit": {"search", "visit"},
    }.get(speculation_mode, set())
    unexpected_speculative_tools = sorted(
        {
            str(record.get("tool"))
            for record in records
            if record.get("speculative") is True
            and record.get("tool") not in enabled_speculative_tools
        }
    )
    if unexpected_speculative_tools:
        raise ValueError(
            f"{label} speculative records violate speculation_mode: "
            + ", ".join(unexpected_speculative_tools)
        )
    if speculation_mode != "off" and speculative_worker_s <= 0.0:
        raise ValueError(f"{label} spec-on cell has no physical speculative execution")
    waste_fraction = (
        wasted_speculative_worker_s / speculative_worker_s
        if speculative_worker_s
        else 0.0
    )
    broker_stats = _mapping(
        _mapping(
            run.payload.get("broker_final_snapshot"),
            f"{label}.broker_final_snapshot",
        ).get("stats"),
        f"{label}.broker_final_snapshot.stats",
    )
    reported_waste = _finite(
        broker_stats.get("wasted_speculative_service_s"),
        f"{label}.broker wasted_speculative_service_s",
    )
    if not math.isclose(
        reported_waste,
        wasted_speculative_worker_s,
        rel_tol=0.02,
        abs_tol=0.01,
    ):
        raise ValueError(f"{label} broker/physical speculative waste differs")
    return {
        "physical_record_count": len(records),
        "rejected_prediction_decision_count": rejected_decisions,
        "uncontrolled_retry_count": uncontrolled_retries,
        "failed_physical_job_count": failed_physical_jobs,
        "started_physical_job_count": started_physical_jobs,
        "physical_http_attempt_count": physical_http_attempts,
        "retried_physical_job_count": retried_physical_jobs,
        "authoritative_commit_count": authoritative_commits,
        "authoritative_retried_commit_count": authoritative_retried_commits,
        "authoritative_retry_rate": (
            authoritative_retried_commits / authoritative_commits
            if authoritative_commits
            else math.inf
        ),
        "authoritative_retry_rate_definition": (
            "committed logical calls with http_attempts>1 / authoritative commits"
        ),
        "service_time_accounting": (
            "started_at..finished_at includes every HTTP attempt and retry backoff"
        ),
        "max_running": max_running,
        "max_running_speculative": max_running_speculative,
        "max_running_by_tool": dict(sorted(maxima_by_tool.items())),
        "speculative_worker_s": speculative_worker_s,
        "wasted_speculative_worker_s": wasted_speculative_worker_s,
        "broker_reported_wasted_speculative_worker_s": reported_waste,
        "wasted_speculative_worker_fraction": waste_fraction,
        "minimum_start_intervals_s": configured_intervals,
        "http_attempt_log_required": require_http_attempt_logs,
        "http_attempt_log_count": (
            sum(len(values) for values in http_attempt_starts_by_tool.values())
            if require_http_attempt_logs
            else None
        ),
        "http_attempt_count_by_tool": (
            {
                tool: len(values)
                for tool, values in sorted(http_attempt_starts_by_tool.items())
            }
            if require_http_attempt_logs
            else None
        ),
        "http_attempt_start_gate_wait_s": (
            http_attempt_gate_wait_s if require_http_attempt_logs else None
        ),
        "http_retry_backoff_s_from_attempt_logs": (
            http_retry_backoff_s if require_http_attempt_logs else None
        ),
    }


def _load_qualification(
    run: ValidatedRun,
    label: str,
    *,
    profile: FormalProfile,
) -> dict[str, Any]:
    environment = _scheduler_environment(run, label)
    raw_max_sequences = environment.get("VLLM_MAX_NUM_SEQS")
    try:
        max_sequences = int(raw_max_sequences)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} lacks integer VLLM_MAX_NUM_SEQS") from exc
    offered = int(run.config["max_active_tasks"])
    if offered >= max_sequences:
        raise ValueError(f"{label} max_active_tasks must be below VLLM_MAX_NUM_SEQS")
    llm_rows = [row for row in run.timeline if row.get("llm_waiting") is not None]
    if not llm_rows:
        raise ValueError(f"{label} has no LLM queue samples")
    waiting_below_cap = sum(
        float(row["llm_waiting"]) > 0.0
        and float(row["llm_running"]) < max_sequences
        for row in llm_rows
    )
    authoritative_queue = sum(
        int(row["tool_queued_authoritative"]) > 0 for row in run.timeline
    )
    dual_flags = [
        row.get("llm_waiting") is not None
        and float(row["llm_waiting"]) > 0.0
        and int(row["tool_queued_authoritative"]) > 0
        for row in run.timeline
    ]
    dual_pressure = sum(dual_flags)
    longest_dual_count = 0
    longest_dual_elapsed_s = 0.0
    streak_start: float | None = None
    streak_count = 0
    previous_dual_monotonic: float | None = None
    previous_dual_wall: float | None = None
    maximum_adjacent_dual_monotonic_gap_s = 0.0
    maximum_adjacent_dual_wall_gap_s = 0.0
    dual_gap_reset_count = 0
    for row, pressured in zip(run.timeline, dual_flags):
        if not pressured:
            streak_start = None
            streak_count = 0
            previous_dual_monotonic = None
            previous_dual_wall = None
            continue
        monotonic_value = _finite(
            row.get("monotonic_s"), f"{label} dual timeline monotonic_s"
        )
        wall_value = _finite(row.get("wall_s"), f"{label} dual timeline wall_s")
        if previous_dual_monotonic is not None and previous_dual_wall is not None:
            monotonic_gap = monotonic_value - previous_dual_monotonic
            wall_gap = wall_value - previous_dual_wall
            if monotonic_gap < 0.0 or wall_gap < 0.0:
                raise ValueError(f"{label} dual timeline timestamps decrease")
            maximum_adjacent_dual_monotonic_gap_s = max(
                maximum_adjacent_dual_monotonic_gap_s, monotonic_gap
            )
            maximum_adjacent_dual_wall_gap_s = max(
                maximum_adjacent_dual_wall_gap_s, wall_gap
            )
            threshold = profile.max_dual_queue_adjacent_sample_gap_s
            if threshold is not None and (
                monotonic_gap > threshold or wall_gap > threshold
            ):
                streak_start = None
                streak_count = 0
                dual_gap_reset_count += 1
        if streak_start is None:
            streak_start = monotonic_value
            streak_count = 1
        else:
            streak_count += 1
        longest_dual_count = max(longest_dual_count, streak_count)
        longest_dual_elapsed_s = max(
            longest_dual_elapsed_s, monotonic_value - streak_start
        )
        previous_dual_monotonic = monotonic_value
        previous_dual_wall = wall_value
    return {
        "offered_concurrency": offered,
        "vllm_max_num_seqs": max_sequences,
        "nonbinding_max_num_seqs": offered < max_sequences,
        "llm_metric_sample_count": len(llm_rows),
        "native_waiting_below_cap_sample_count": waiting_below_cap,
        "native_waiting_below_cap_fraction": waiting_below_cap / len(llm_rows),
        "timeline_sample_count": len(run.timeline),
        "authoritative_tool_queue_sample_count": authoritative_queue,
        "authoritative_tool_queue_sample_fraction": authoritative_queue
        / len(run.timeline),
        "dual_queue_pressure_sample_count": dual_pressure,
        "longest_consecutive_dual_queue_pressure_sample_count": (
            longest_dual_count
        ),
        "longest_consecutive_dual_queue_pressure_elapsed_s": (
            longest_dual_elapsed_s
        ),
        "continuity_max_adjacent_sample_gap_s": (
            profile.max_dual_queue_adjacent_sample_gap_s
        ),
        "maximum_adjacent_simultaneous_monotonic_gap_s": (
            maximum_adjacent_dual_monotonic_gap_s
        ),
        "maximum_adjacent_simultaneous_wall_gap_s": (
            maximum_adjacent_dual_wall_gap_s
        ),
        "simultaneous_gap_reset_count": dual_gap_reset_count,
        "longest_continuous_dual_sample_count": longest_dual_count,
        "longest_continuous_dual_span_s": longest_dual_elapsed_s,
        "passed": (
            offered < max_sequences
            and waiting_below_cap / len(llm_rows)
            >= profile.min_native_waiting_below_cap_fraction
            and authoritative_queue / len(run.timeline)
            >= profile.min_authoritative_tool_queue_fraction
            and dual_pressure >= profile.min_dual_queue_pressure_samples
            and longest_dual_elapsed_s
            >= profile.min_dual_queue_pressure_consecutive_s
        ),
    }


def _validate_canary_pre_enqueue_skip(
    run: ValidatedRun,
    label: str,
    *,
    expected_count: int,
) -> dict[str, Any]:
    canary_task_ids = {
        task_id
        for task_id, task in run.tasks_by_id.items()
        if task.get("visit_canary") is True
    }
    if len(canary_task_ids) != expected_count:
        raise ValueError(
            f"{label} expected exactly {expected_count} visit canary tasks"
        )
    canary_all_records = [
        record
        for record in run.physical_records
        if record.get("session_id") in canary_task_ids
    ]
    canary_speculative_records = [
        record
        for record in canary_all_records
        if record.get("speculative") is True
    ]
    if canary_speculative_records:
        raise ValueError(
            f"{label} canary prediction was enqueued instead of skipped"
        )
    canary_visit_records = [
        record for record in canary_all_records if record.get("tool") == "visit"
    ]
    authoritative_canary_commits = [
        run.committed_by_task_tool[(task_id, "visit")]
        for task_id in sorted(canary_task_ids)
    ]
    if len(canary_visit_records) != expected_count:
        raise ValueError(
            f"{label} canary visit physical record count is not exactly {expected_count}"
        )
    for record in authoritative_canary_commits:
        if (
            record.get("canary") is not True
            or record.get("speculative") is not False
            or record.get("speculation_eligible") is not False
            or record.get("committed") is not True
            or record.get("authoritative") is not True
        ):
            raise ValueError(f"{label} canary visit is not authoritative-only")
    speculative_visit_count = sum(
        record.get("tool") == "visit" and record.get("speculative") is True
        for record in run.physical_records
    )
    if run.config.get("speculation_mode") == "visit" and speculative_visit_count == 0:
        raise ValueError(f"{label} visit-speculation cell did not execute speculation")
    return {
        "canary_task_count": len(canary_task_ids),
        "authoritative_canary_visit_commit_count": len(
            authoritative_canary_commits
        ),
        "canary_speculative_record_count": len(canary_speculative_records),
        "canary_speculative_visit_record_count": sum(
            record.get("tool") == "visit" for record in canary_speculative_records
        ),
        "all_canaries_skipped_before_speculative_enqueue": True,
        "speculative_visit_record_count": speculative_visit_count,
    }


def _effect(
    name: str,
    baseline_cell: str,
    candidate_cell: str,
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    aggregate_sources: Mapping[str, Mapping[str, float]],
    *,
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    baseline = aggregate_sources[baseline_cell]
    candidate = aggregate_sources[candidate_cell]
    reductions = {
        source: baseline[source] - candidate[source] for source in sorted(baseline)
    }
    block_rows: list[dict[str, Any]] = []
    for block_id in sorted(runs):
        base_run = runs[block_id][baseline_cell]
        candidate_run = runs[block_id][candidate_cell]
        base_sources = _source_values(base_run)
        candidate_sources = _source_values(candidate_run)
        base_mean = statistics.fmean(base_sources.values())
        candidate_mean = statistics.fmean(candidate_sources.values())
        base_task = _mapping(base_run.summary["task_e2e_s"], "base task summary")
        candidate_task = _mapping(
            candidate_run.summary["task_e2e_s"], "candidate task summary"
        )
        block_rows.append(
            {
                "block_id": block_id,
                "baseline_mean_s": base_mean,
                "candidate_mean_s": candidate_mean,
                "mean_absolute_reduction_s": base_mean - candidate_mean,
                "relative_reduction": (
                    (base_mean - candidate_mean) / base_mean if base_mean else 0.0
                ),
                "faster_source_count": sum(
                    base_sources[source] > candidate_sources[source]
                    for source in base_sources
                ),
                "baseline_p50_s": float(base_task["p50"]),
                "candidate_p50_s": float(candidate_task["p50"]),
                "baseline_p95_s": float(base_task["p95"]),
                "candidate_p95_s": float(candidate_task["p95"]),
                "baseline_makespan_s": float(
                    base_run.summary["task_completion_makespan_s"]
                ),
                "candidate_makespan_s": float(
                    candidate_run.summary["task_completion_makespan_s"]
                ),
            }
        )
    baseline_mean = statistics.fmean(baseline.values())
    candidate_mean = statistics.fmean(candidate.values())
    return {
        "name": name,
        "baseline_cell": baseline_cell,
        "candidate_cell": candidate_cell,
        "baseline_mean_s": baseline_mean,
        "candidate_mean_s": candidate_mean,
        "mean_absolute_reduction_s": statistics.fmean(reductions.values()),
        "aggregate_relative_reduction": (
            (baseline_mean - candidate_mean) / baseline_mean if baseline_mean else 0.0
        ),
        "faster_source_count": sum(value > 0.0 for value in reductions.values()),
        "faster_source_fraction": sum(value > 0.0 for value in reductions.values())
        / len(reductions),
        "source_reduction_s": dict(sorted(reductions.items())),
        "source_reduction_distribution_s": _distribution(list(reductions.values())),
        "bootstrap": _bootstrap_reductions(
            reductions, baseline, candidate, resamples=resamples, seed=seed
        ),
        "every_block_mean_reduction_positive": all(
            row["mean_absolute_reduction_s"] > 0.0 for row in block_rows
        ),
        "blocks": block_rows,
    }


def _interaction(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    aggregate_sources: Mapping[str, Mapping[str, float]],
    *,
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = {
        source: (
            aggregate_sources["E"][source]
            - aggregate_sources["F"][source]
            - aggregate_sources["A"][source]
            + aggregate_sources["B"][source]
        )
        for source in sorted(aggregate_sources["A"])
    }
    blocks = []
    for block_id in sorted(runs):
        source = {cell: _source_values(runs[block_id][cell]) for cell in CELL_IDS}
        block_values = [
            source["E"][item]
            - source["F"][item]
            - source["A"][item]
            + source["B"][item]
            for item in sorted(source["A"])
        ]
        blocks.append(
            {
                "block_id": block_id,
                "mean_interaction_s": statistics.fmean(block_values),
                "positive_source_count": sum(value > 0.0 for value in block_values),
            }
        )
    return {
        "definition": "(E - F) - (A - B); positive means speculation helps more under Joint",
        "mean_interaction_s": statistics.fmean(values.values()),
        "positive_source_count": sum(value > 0.0 for value in values.values()),
        "source_interaction_s": dict(sorted(values.items())),
        "source_interaction_distribution_s": _distribution(list(values.values())),
        "bootstrap": _bootstrap_reductions(
            values, None, None, resamples=resamples, seed=seed
        ),
        "blocks": blocks,
        "acceptance_effect": "reported_only",
    }


def _combined_llm_durations(
    runs: Mapping[str, Mapping[str, ValidatedRun]], cell: str
) -> list[float]:
    return [
        float(event["duration_s"])
        for block_id in sorted(runs)
        for events in runs[block_id][cell].llm_by_task.values()
        for event in events
    ]


def _combined_task_e2e(
    runs: Mapping[str, Mapping[str, ValidatedRun]], cell: str
) -> list[float]:
    return [
        float(task["e2e_s"])
        for block_id in sorted(runs)
        for task in runs[block_id][cell].tasks_by_key.values()
    ]


def _canary_comparison(
    runs: Mapping[str, Mapping[str, ValidatedRun]],
    baseline_cell: str,
    candidate_cell: str,
) -> dict[str, Any]:
    def rows(cell: str) -> dict[tuple[str, int, str], float]:
        result: dict[tuple[str, int, str], float] = {}
        for block_id in sorted(runs):
            run = runs[block_id][cell]
            for key, task in run.tasks_by_key.items():
                record = run.committed_by_task_tool[(str(task["task_id"]), "visit")]
                if record.get("canary") is True:
                    confirmation = _finite(
                        record.get("authoritative_confirmation_at"),
                        f"{block_id}/{cell}/{task['task_id']} canary confirmation",
                    )
                    finished = _finite(
                        record.get("finished_at"),
                        f"{block_id}/{cell}/{task['task_id']} canary finish",
                    )
                    if finished <= confirmation:
                        raise ValueError(
                            f"{block_id}/{cell}/{task['task_id']} canary has non-positive raw latency"
                        )
                    raw_latency = finished - confirmation
                    exposed = _finite(
                        record.get("exposed_wait_s"),
                        f"{block_id}/{cell}/{task['task_id']} canary exposed wait",
                    )
                    if not math.isclose(
                        exposed, raw_latency, rel_tol=0.02, abs_tol=0.05
                    ):
                        raise ValueError(
                            f"{block_id}/{cell}/{task['task_id']} canary latency telemetry differs"
                        )
                    result[(key[0], key[1], block_id)] = raw_latency
        return result

    baseline = rows(baseline_cell)
    candidate = rows(candidate_cell)
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError(
            f"{baseline_cell}/{candidate_cell} canary identities are empty or differ"
        )
    baseline_values = list(baseline.values())
    candidate_values = list(candidate.values())
    baseline_mean = statistics.fmean(baseline_values)
    candidate_mean = statistics.fmean(candidate_values)
    baseline_p95 = _percentile(baseline_values, 0.95)
    candidate_p95 = _percentile(candidate_values, 0.95)
    return {
        "count": len(baseline),
        "baseline_cell": baseline_cell,
        "candidate_cell": candidate_cell,
        "baseline_mean_s": baseline_mean,
        "candidate_mean_s": candidate_mean,
        "mean_ratio": candidate_mean / baseline_mean if baseline_mean else math.inf,
        "baseline_p95_s": baseline_p95,
        "candidate_p95_s": candidate_p95,
        "p95_ratio": candidate_p95 / baseline_p95 if baseline_p95 else math.inf,
    }


def aggregate_live_joint_four_cell(
    blocks: Sequence[tuple[str, Path, Path, Path, Path]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    formal_workload: Path | None = None,
) -> dict[str, Any]:
    if len(blocks) != FORMAL_BLOCK_COUNT:
        raise ValueError(f"formal aggregation requires exactly {FORMAL_BLOCK_COUNT} blocks")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    block_ids = [item[0] for item in blocks]
    if len(set(block_ids)) != len(block_ids) or any(not item for item in block_ids):
        raise ValueError("formal block IDs must be unique and non-empty")

    runs: dict[str, dict[str, ValidatedRun]] = {}
    inputs: dict[str, Any] = {}
    server_ids: set[str] = set()
    block_orders: dict[str, list[str]] = {}
    for raw in blocks:
        block_id = raw[0]
        paths = dict(zip(CELL_IDS, raw[1:]))
        if len({Path(path).resolve() for path in paths.values()}) != 4:
            raise ValueError(f"{block_id} does not contain four distinct result paths")
        validated: dict[str, ValidatedRun] = {}
        order: dict[int, str] = {}
        block_input: dict[str, Any] = {}
        for cell in CELL_IDS:
            role = "baseline" if cell in {"A", "E"} else "candidate"
            run = _validate_run(Path(paths[cell]), role=role)
            order_index, server_id = _validate_formal_metadata(
                run, expected_block_id=block_id, expected_cell_id=cell
            )
            if order_index in order:
                raise ValueError(f"{block_id} has duplicate order_index {order_index}")
            if server_id in server_ids:
                raise ValueError(f"server_instance_id is reused: {server_id}")
            order[order_index] = cell
            server_ids.add(server_id)
            validated[cell] = run
            block_input[cell] = {
                "path": str(run.path),
                "sha256": run.sha256,
                "server_instance_id": server_id,
                "order_index": order_index,
            }
        if set(order) != set(range(4)):
            raise ValueError(f"{block_id} order indices are not a permutation of 0..3")
        runs[block_id] = validated
        inputs[block_id] = block_input
        block_orders[block_id] = [order[index] for index in range(4)]

    ab_forward = sum(order.index("A") < order.index("B") for order in block_orders.values())
    ef_forward = sum(order.index("E") < order.index("F") for order in block_orders.values())
    if abs(ab_forward - (len(blocks) - ab_forward)) > 1:
        raise ValueError("A/B forward and reverse block orders are not balanced")
    if abs(ef_forward - (len(blocks) - ef_forward)) > 1:
        raise ValueError("E/F forward and reverse block orders are not balanced")

    profile, workload_path, workload_validation = _resolve_formal_profile(
        runs, formal_workload
    )
    v9_development_selection = (
        _validate_v9_development_selection(
            profile=profile,
            workload_validation=workload_validation,
        )
        if profile.require_v9_selection_provenance
        else None
    )
    config_validation = _validate_config_factorial(runs, profile=profile)
    modern_runtime_validation = (
        _validate_modern_runtime_contract(runs, profile=profile)
        if profile.require_modern_live_evidence
        else None
    )
    identity_validation = _validate_formal_source_identity(
        runs,
        profile=profile,
        workload_path=workload_path,
        workload_validation=workload_validation,
    )
    v9_cell_provenance = (
        _validate_v9_cell_provenance(
            runs,
            expected_selection=v9_development_selection,
        )
        if v9_development_selection is not None
        else None
    )
    if v9_cell_provenance is not None:
        for block_id, block in v9_cell_provenance.items():
            for cell, provenance in block.items():
                inputs[block_id][cell]["coordinator_provenance"] = provenance
    physical: dict[str, dict[str, Any]] = {}
    load: dict[str, dict[str, Any]] = {}
    canary_skip: dict[str, dict[str, Any]] = {}
    guided_json_recovery: dict[str, dict[str, Any]] = {}
    output_contract: dict[str, dict[str, Any]] = {}
    strict_guided_final_contract: dict[str, dict[str, Any]] = {}
    fixed_final_completion_contract: dict[str, dict[str, Any]] = {}
    for block_id in sorted(runs):
        physical[block_id] = {
            cell: _validate_physical_run(
                runs[block_id][cell],
                f"{block_id}/{cell}",
                require_http_attempt_logs=profile.require_modern_live_evidence,
            )
            for cell in CELL_IDS
        }
        load[block_id] = {
            cell: _load_qualification(
                runs[block_id][cell],
                f"{block_id}/{cell}",
                profile=profile,
            )
            for cell in CELL_IDS
        }
        if profile.require_modern_live_evidence:
            canary_skip[block_id] = {
                cell: _validate_canary_pre_enqueue_skip(
                    runs[block_id][cell],
                    f"{block_id}/{cell}",
                    expected_count=profile.expected_canary_count,
                )
                for cell in CELL_IDS
            }
        if profile.require_zero_guided_json_recovery:
            if profile.guided_json_parsed_call_count is None:
                raise ValueError(
                    f"{profile.name} lacks a guided JSON parsed-call contract"
                )
            guided_json_recovery[block_id] = {
                cell: _validate_zero_guided_json_recovery(
                    runs[block_id][cell],
                    f"{block_id}/{cell}",
                    expected_parsed_call_count=(
                        profile.guided_json_parsed_call_count
                    ),
                )
                for cell in CELL_IDS
            }
        if profile.require_plain_final_output_contract:
            output_contract[block_id] = {
                cell: _validate_plain_final_output_contract(
                    runs[block_id][cell], f"{block_id}/{cell}"
                )
                for cell in CELL_IDS
            }
        if profile.require_strict_guided_final_output_contract:
            strict_guided_final_contract[block_id] = {
                cell: _validate_strict_guided_final_output_contract(
                    runs[block_id][cell], f"{block_id}/{cell}"
                )
                for cell in CELL_IDS
            }
        if profile.require_strict_semantic_ascii_space_tail:
            if profile.expected_final_completion_tokens is None:
                raise ValueError(
                    f"{profile.name} lacks a fixed final completion-token contract"
                )
            fixed_final_completion_contract[block_id] = {
                cell: _validate_fixed_final_completion_contract(
                    runs[block_id][cell],
                    f"{block_id}/{cell}",
                    expected_completion_tokens=(
                        profile.expected_final_completion_tokens
                    ),
                )
                for cell in CELL_IDS
            }

    aggregate_sources: dict[str, dict[str, float]] = {}
    for cell in CELL_IDS:
        by_block = {
            block_id: _source_values(runs[block_id][cell]) for block_id in sorted(runs)
        }
        aggregate_sources[cell] = {
            source: statistics.fmean(by_block[block_id][source] for block_id in by_block)
            for source in sorted(next(iter(by_block.values())))
        }
    effects = {
        name: _effect(
            name,
            baseline,
            candidate,
            runs,
            aggregate_sources,
            resamples=bootstrap_resamples,
            seed=profile.bootstrap_seed,
        )
        for name, (baseline, candidate) in EFFECTS.items()
    }
    interaction = _interaction(
        runs,
        aggregate_sources,
        resamples=bootstrap_resamples,
        seed=profile.bootstrap_seed,
    )
    ef_component_decomposition = (
        _ef_component_decomposition(runs)
        if profile.require_ef_component_decomposition
        else None
    )

    completion_tokens = {
        cell: sum(
            int(runs[block][cell].summary["llm"]["completion_tokens"])
            for block in runs
        )
        for cell in CELL_IDS
    }
    token_differences = {
        name: _relative_difference(completion_tokens[baseline], completion_tokens[candidate])
        for name, (baseline, candidate) in EFFECTS.items()
    }
    block_token_differences = {
        name: {
            block: _relative_difference(
                float(runs[block][baseline].summary["llm"]["completion_tokens"]),
                float(runs[block][candidate].summary["llm"]["completion_tokens"]),
            )
            for block in sorted(runs)
        }
        for name, (baseline, candidate) in EFFECTS.items()
    }
    uncontrolled_retry_counts = {
        cell: sum(physical[block][cell]["uncontrolled_retry_count"] for block in runs)
        for cell in CELL_IDS
    }
    failure_counts = {
        cell: sum(physical[block][cell]["failed_physical_job_count"] for block in runs)
        for cell in CELL_IDS
    }
    authoritative_retry = {}
    for cell in CELL_IDS:
        commits = sum(
            physical[block][cell]["authoritative_commit_count"] for block in runs
        )
        retried = sum(
            physical[block][cell]["authoritative_retried_commit_count"]
            for block in runs
        )
        authoritative_retry[cell] = {
            "retried_commit_count": retried,
            "commit_count": commits,
            "rate": retried / commits if commits else math.inf,
            "by_block": {
                block: {
                    "retried_commit_count": physical[block][cell][
                        "authoritative_retried_commit_count"
                    ],
                    "commit_count": physical[block][cell][
                        "authoritative_commit_count"
                    ],
                    "rate": physical[block][cell]["authoritative_retry_rate"],
                }
                for block in sorted(runs)
            },
        }
    ef_retry_difference = abs(
        authoritative_retry["E"]["rate"] - authoritative_retry["F"]["rate"]
    )
    af_retry_difference = abs(
        authoritative_retry["A"]["rate"] - authoritative_retry["F"]["rate"]
    )
    speculative = {
        cell: {
            "worker_s": sum(
                physical[block][cell]["speculative_worker_s"] for block in runs
            ),
            "wasted_worker_s": sum(
                physical[block][cell]["wasted_speculative_worker_s"] for block in runs
            ),
        }
        for cell in ("B", "F")
    }
    for value in speculative.values():
        value["wasted_worker_fraction"] = (
            value["wasted_worker_s"] / value["worker_s"]
            if value["worker_s"]
            else math.inf
        )
    exact_hits = {
        cell: sum(
            int(runs[block][cell].summary["tool"]["exact_hit_count"])
            for block in runs
        )
        for cell in ("B", "F")
    }
    eligible_commits = {
        cell: sum(
            sum(
                record.get("speculation_eligible") is True
                for record in runs[block][cell].committed_by_task_tool.values()
            )
            for block in runs
        )
        for cell in ("B", "F")
    }
    hit_rates = {
        cell: exact_hits[cell] / eligible_commits[cell]
        if eligible_commits[cell]
        else 0.0
        for cell in ("B", "F")
    }
    canary = {
        "A_to_B": _canary_comparison(runs, "A", "B"),
        "E_to_F": _canary_comparison(runs, "E", "F"),
    }

    ef = effects["E_to_F"]
    af = effects["A_to_F"]
    combined_task = {cell: _combined_task_e2e(runs, cell) for cell in CELL_IDS}
    combined_llm = {cell: _combined_llm_durations(runs, cell) for cell in CELL_IDS}
    mean_makespan = {
        cell: statistics.fmean(
            float(runs[block][cell].summary["task_completion_makespan_s"])
            for block in runs
        )
        for cell in CELL_IDS
    }

    gates: dict[str, dict[str, Any]] = {}
    total_tasks = sum(len(runs[block][cell].tasks_by_key) for block in runs for cell in CELL_IDS)
    total_llm_requests = sum(
        sum(len(events) for events in runs[block][cell].llm_by_task.values())
        for block in runs
        for cell in CELL_IDS
    )
    total_tool_commits = sum(
        len(runs[block][cell].committed_by_task_tool)
        for block in runs
        for cell in CELL_IDS
    )
    gates["raw_success_exactly_once_and_exact_commit_identity"] = _gate(
        {
            "successful_tasks": total_tasks,
            "successful_logical_llm_requests": total_llm_requests,
            "exact_authoritative_tool_commits": total_tool_commits,
        },
        "all raw tasks and 3 LLM + search/visit calls per task validated exactly once",
        (
            total_tasks
            == FORMAL_BLOCK_COUNT * len(CELL_IDS) * profile.tasks_per_cell
            and total_llm_requests == 3 * total_tasks
            and total_tool_commits == 2 * total_tasks
        ),
    )
    gates["fresh_server_empty_cache_and_drained_every_cell"] = _gate(
        {
            "unique_server_instances": len(server_ids),
            "cell_instances": FORMAL_BLOCK_COUNT * len(CELL_IDS),
        },
        "12 unique instances; every formal_run attestation true and raw broker empty",
        len(server_ids) == FORMAL_BLOCK_COUNT * len(CELL_IDS),
    )
    gates[f"all_{profile.source_count}_frozen_source_identities"] = _gate(
        identity_validation,
        (
            f"exact {profile.name} workload SHA, question/query/expected-URL, "
            "and invocation identity"
        ),
        identity_validation["source_count"] == profile.source_count,
    )
    gates["formal_workload_profile_and_sha_binding"] = _gate(
        {
            "profile": profile.name,
            "split_id": profile.split_id,
            "path": str(workload_path),
            "file_sha256": workload_validation["file_sha256"],
            "canonical_sources_sha256": workload_validation[
                "canonical_sources_sha256"
            ],
        },
        "one supported repository-frozen workload; every cell binds its exact SHA",
        True,
    )
    if profile.require_modern_live_evidence:
        gate_profile_prefix = profile.name.replace("-", "_")
        gates[
            f"{gate_profile_prefix}_execution_aware_policy_and_code_binding"
        ] = _gate(
            modern_runtime_validation,
            (
                "execution_aware exact-session/invocation running-or-completed "
                "policy with the profile-frozen live_agent.py SHA in every cell; "
                "current module equality when required by the profile"
            ),
            True,
        )
        gates[f"{gate_profile_prefix}_http_attempt_gate_and_success_ledgers"] = _gate(
            {
                block: {
                    cell: {
                        "physical_http_attempt_count": physical[block][cell][
                            "physical_http_attempt_count"
                        ],
                        "http_attempt_log_count": physical[block][cell][
                            "http_attempt_log_count"
                        ],
                        "http_attempt_count_by_tool": physical[block][cell][
                            "http_attempt_count_by_tool"
                        ],
                    }
                    for cell in CELL_IDS
                }
                for block in sorted(runs)
            },
            (
                "every physical GET has an exact final-success attempt ledger; "
                "every visit attempt start obeys the shared "
                f"{profile.visit_min_start_interval_s}s gate"
            ),
            all(
                physical[block][cell]["physical_http_attempt_count"]
                == physical[block][cell]["http_attempt_log_count"]
                for block in runs
                for cell in CELL_IDS
            ),
        )
        canary_count_label = {
            10: "ten",
            14: "fourteen",
        }.get(profile.expected_canary_count, str(profile.expected_canary_count))
        gates[
            f"{gate_profile_prefix}_{canary_count_label}_canaries_"
            "skip_visit_speculation_before_enqueue"
        ] = _gate(
            canary_skip,
            (
                f"each block/cell has exactly {profile.expected_canary_count} "
                "authoritative canary visits and "
                "zero speculative record for any canary session"
            ),
            all(
                row["canary_task_count"] == profile.expected_canary_count
                and row["authoritative_canary_visit_commit_count"]
                == profile.expected_canary_count
                and row["canary_speculative_record_count"] == 0
                and (
                    profile.name not in {"formal-v8", "formal-v9"}
                    or row["speculative_visit_record_count"]
                    == (
                        profile.source_count - profile.expected_canary_count
                        if cell in {"B", "F"}
                        else 0
                    )
                )
                for block in canary_skip.values()
                for cell, row in block.items()
            ),
        )
    if profile.require_zero_guided_json_recovery:
        assert profile.guided_json_parsed_call_count is not None
        guided_count = profile.guided_json_parsed_call_count
        gates[
            f"{profile.name.replace('-', '_')}_zero_guided_json_recovery"
        ] = _gate(
            guided_json_recovery,
            (
                "every task uses escape-unescaped-string-controls-v1, parses "
                f"exactly {guided_count} guided calls, and has recovery_count=0"
            ),
            all(
                row["task_count"] == profile.tasks_per_cell
                and row["parsed_call_count"]
                == guided_count * profile.tasks_per_cell
                and row["recovery_count"] == 0
                and row["recovered_task_count"] == 0
                for block in guided_json_recovery.values()
                for row in block.values()
            ),
        )
    if profile.require_plain_final_output_contract:
        gates["formal_v6_plain_final_output_contract"] = _gate(
            output_contract,
            (
                "each task has exactly two guided JSON calls and one call-2 "
                "plain_text_local_wrap with exact URL/SHA/count evidence"
            ),
            all(
                row["task_count"] == profile.tasks_per_cell
                and row["output_call_count"] == 3 * profile.tasks_per_cell
                and row["guided_json_output_call_count"]
                == 2 * profile.tasks_per_cell
                and row["plain_text_local_wrap_call_count"]
                == profile.tasks_per_cell
                and row["exact_url_binding_count"] == profile.tasks_per_cell
                and row["contract_success_count"] == profile.tasks_per_cell
                for block in output_contract.values()
                for row in block.values()
            ),
        )
    if profile.require_strict_guided_final_output_contract:
        gates["formal_v7_strict_guided_final_output_contract"] = _gate(
            strict_guided_final_contract,
            (
                "each task has two zero-recovery guided tool calls and one "
                "strict unbounded-schema guided final call with exact URL and "
                "valid local-projection telemetry"
            ),
            all(
                row["task_count"] == profile.tasks_per_cell
                and row["output_call_count"] == 3 * profile.tasks_per_cell
                and row["guided_tool_call_count"]
                == 2 * profile.tasks_per_cell
                and row["strict_guided_final_call_count"]
                == profile.tasks_per_cell
                and row["exact_url_binding_count"] == profile.tasks_per_cell
                and row["strict_parse_success_count"] == profile.tasks_per_cell
                and row["recovery_applied_count"] == 0
                for block in strict_guided_final_contract.values()
                for row in block.values()
            ),
        )
    if profile.require_strict_semantic_ascii_space_tail:
        assert profile.expected_final_completion_tokens is not None
        gates[
            f"{profile.name.replace('-', '_')}_strict_semantic_json_ascii_space_tail"
        ] = _gate(
            fixed_final_completion_contract,
            (
                "every call-2 wire response reparses as exactly one strict "
                "{answer,source_url} JSON object followed by a non-empty "
                "ASCII-space-only tail"
            ),
            all(
                row["all_semantic_json_and_ascii_space_tails_exact"] is True
                for block in fixed_final_completion_contract.values()
                for row in block.values()
            ),
        )
        gates[
            f"{profile.name.replace('-', '_')}_call2_completion_tokens_exact"
        ] = _gate(
            fixed_final_completion_contract,
            (
                "every task call-2 completion_tokens == "
                f"{profile.expected_final_completion_tokens}"
            ),
            all(
                row["all_completion_tokens_exact"] is True
                and row["task_count"] == profile.tasks_per_cell
                for block in fixed_final_completion_contract.values()
                for row in block.values()
            ),
        )
    gates["all_A_blocks_have_native_llm_and_authoritative_tool_queue"] = _gate(
        {block: load[block]["A"] for block in sorted(load)},
        (
            "each A block: offered<max_num_seqs, preregistered native LLM "
            f"waiting>={profile.min_native_waiting_below_cap_fraction:.3f} "
            "while running<max_num_seqs, authoritative-tool queue fraction>="
            f"{profile.min_authoritative_tool_queue_fraction:.3f}, dual-pressure "
            f"samples>={profile.min_dual_queue_pressure_samples}, and longest "
            "continuous dual-pressure elapsed_s>="
            f"{profile.min_dual_queue_pressure_consecutive_s:.3f}; a streak "
            "continues only across adjacent sample gaps<="
            f"{profile.max_dual_queue_adjacent_sample_gap_s}"
        ),
        all(load[block]["A"]["passed"] for block in load),
    )
    if profile.name in {"formal-v8", "formal-v9"}:
        gates[
            f"{profile.name.replace('-', '_')}_offered_concurrency_"
            "above_64_below_max_num_seqs"
        ] = _gate(
            {block: load[block]["A"] for block in sorted(load)},
            "every A block offers 80 tasks: 80 > 64 and 80 < max_num_seqs=96",
            all(
                load[block]["A"]["offered_concurrency"] == 80
                and load[block]["A"]["offered_concurrency"] > 64
                and load[block]["A"]["offered_concurrency"]
                < load[block]["A"]["vllm_max_num_seqs"]
                for block in load
            ),
        )
    if profile.name == "formal-v9":
        assert v9_development_selection is not None
        assert v9_cell_provenance is not None
        gates["formal_v9_frozen_development_selection_and_cell_provenance"] = _gate(
            {
                "selection": v9_development_selection,
                "cells": v9_cell_provenance,
            },
            (
                "exact completed-screen/strict-selection/transport SHAs; F0, "
                "visit interval 2.5s, min speculative workers 0, exact live-broker "
                "SHA; every effective config/result/timeline is manifest-bound"
            ),
            True,
        )
        gates["formal_v9_physical_visit_attempt_gate_2p5s"] = _gate(
            {
                block: {
                    cell: {
                        "configured_interval_s": physical[block][cell][
                            "minimum_start_intervals_s"
                        ]["visit"],
                        "attempt_log_count": physical[block][cell][
                            "http_attempt_log_count"
                        ],
                    }
                    for cell in CELL_IDS
                }
                for block in sorted(runs)
            },
            (
                "every block/cell is configured for 2.5s and every adjacent "
                "physical visit-attempt start passed the frozen 20ms tolerance"
            ),
            all(
                math.isclose(
                    physical[block][cell]["minimum_start_intervals_s"]["visit"],
                    V9_SELECTED_VISIT_INTERVAL_S,
                    abs_tol=1e-12,
                )
                for block in runs
                for cell in CELL_IDS
            ),
        )
    gates["zero_uncontrolled_http_retries"] = _gate(
        uncontrolled_retry_counts,
        (
            "zero; every observed retry is bounded by max_attempts=2, "
            "backoff=1.0s, and idempotent-get-v1"
        ),
        all(value == 0 for value in uncontrolled_retry_counts.values()),
    )
    gates["zero_failed_physical_jobs"] = _gate(
        failure_counts,
        "zero across A/B/E/F",
        all(value == 0 for value in failure_counts.values()),
    )
    if profile.name == "formal-v9":
        gates["formal_v9_every_block_cell_zero_http_retry"] = _gate(
            {
                block: {
                    cell: physical[block][cell]["retried_physical_job_count"]
                    for cell in CELL_IDS
                }
                for block in sorted(runs)
            },
            "exactly zero started physical jobs with http_attempts>1 per block/cell",
            all(
                physical[block][cell]["retried_physical_job_count"] == 0
                for block in runs
                for cell in CELL_IDS
            ),
        )
        gates["formal_v9_every_block_cell_zero_wasted_speculative_service"] = _gate(
            {
                block: {
                    cell: physical[block][cell]["wasted_speculative_worker_s"]
                    for cell in CELL_IDS
                }
                for block in sorted(runs)
            },
            "exactly 0.0 seconds in every block/cell",
            all(
                physical[block][cell]["wasted_speculative_worker_s"] == 0.0
                for block in runs
                for cell in CELL_IDS
            ),
        )
    gates["all_cells_authoritative_retry_rate_at_most_2pct"] = _gate(
        authoritative_retry,
        "every aggregate cell and every block/cell <=0.02",
        all(
            row["rate"] <= FORMAL_MAX_AUTHORITATIVE_RETRY_RATE
            and all(
                block_row["rate"] <= FORMAL_MAX_AUTHORITATIVE_RETRY_RATE
                for block_row in row["by_block"].values()
            )
            for row in authoritative_retry.values()
        ),
    )
    gates["E_to_F_authoritative_retry_rate_difference_at_most_1pp"] = _gate(
        {
            "E": authoritative_retry["E"]["rate"],
            "F": authoritative_retry["F"]["rate"],
            "absolute_difference": ef_retry_difference,
        },
        "absolute difference <=0.01",
        ef_retry_difference <= FORMAL_MAX_RETRY_RATE_DIFFERENCE,
    )
    gates["A_to_F_authoritative_retry_rate_difference_at_most_1pp"] = _gate(
        {
            "A": authoritative_retry["A"]["rate"],
            "F": authoritative_retry["F"]["rate"],
            "absolute_difference": af_retry_difference,
        },
        "absolute difference <=0.01",
        af_retry_difference <= FORMAL_MAX_RETRY_RATE_DIFFERENCE,
    )
    for effect_name in ("A_to_B", "E_to_F", "A_to_F"):
        gates[f"{effect_name}_completion_token_difference_below_1pct"] = _gate(
            {
                "aggregate": token_differences[effect_name],
                "by_block": block_token_differences[effect_name],
            },
            "aggregate and every block <0.01",
            token_differences[effect_name] < 0.01
            and all(
                value < 0.01
                for value in block_token_differences[effect_name].values()
            ),
        )
    gates["F_speculative_hit_rate"] = _gate(
        hit_rates["F"], ">=0.20", hit_rates["F"] >= 0.20
    )
    gates["F_wasted_speculative_worker_fraction"] = _gate(
        speculative["F"]["wasted_worker_fraction"],
        "<=0.30",
        speculative["F"]["wasted_worker_fraction"] <= 0.30,
    )
    gates["E_to_F_canary_mean_ratio"] = _gate(
        canary["E_to_F"]["mean_ratio"],
        "<=1.03",
        canary["E_to_F"]["mean_ratio"] <= 1.03,
    )
    gates["E_to_F_canary_p95_ratio"] = _gate(
        canary["E_to_F"]["p95_ratio"],
        "<=1.05",
        canary["E_to_F"]["p95_ratio"] <= 1.05,
    )
    gates["E_to_F_every_block_mean_direction"] = _gate(
        [row["mean_absolute_reduction_s"] for row in ef["blocks"]],
        "all >0",
        bool(ef["every_block_mean_reduction_positive"]),
    )
    gates["E_to_F_mean_reduction"] = _gate(
        ef["aggregate_relative_reduction"],
        ">=0.05",
        ef["aggregate_relative_reduction"] >= 0.05,
    )
    if profile.require_ef_component_decomposition:
        assert ef_component_decomposition is not None
        llm_speedup = ef_component_decomposition[
            "F_llm_component_speedup_fraction"
        ]
        gates["E_to_F_LLM_component_not_more_than_1pct_faster"] = _gate(
            {
                "mean_components_s": ef_component_decomposition[
                    "mean_components_s"
                ],
                "F_llm_component_speedup_fraction": llm_speedup,
            },
            (
                "(E_llm-F_llm)/E_llm <= "
                f"{profile.max_ef_llm_component_speedup:.3f}"
            ),
            llm_speedup is not None
            and llm_speedup <= profile.max_ef_llm_component_speedup,
        )
        component_savings = ef_component_decomposition[
            "mean_saving_E_minus_F_s"
        ]
        net_saving = component_savings["e2e_s"]
        tool_saving = component_savings["tool_exposed_s"]
        gates["E_to_F_tool_exposed_wait_explains_net_saving"] = _gate(
            {
                "net_e2e_saving_s": net_saving,
                "tool_exposed_wait_saving_s": tool_saving,
                "tool_saving_to_net_saving_ratio": (
                    ef_component_decomposition[
                        "tool_exposed_wait_saving_to_net_e2e_saving_ratio"
                    ]
                ),
            },
            (
                "tool exposed-wait saving >= "
                f"{profile.min_ef_tool_saving_to_net_saving_ratio:.3f} * "
                "net E2E saving"
            ),
            net_saving > 0.0
            and tool_saving
            >= profile.min_ef_tool_saving_to_net_saving_ratio * net_saving,
        )
    gates["E_to_F_bootstrap_lower_bound"] = _gate(
        ef["bootstrap"]["absolute_reduction_s_95_ci"][0],
        ">0",
        ef["bootstrap"]["absolute_reduction_s_95_ci"][0] > 0.0,
    )
    gates["E_to_F_faster_sources"] = _gate(
        ef["faster_source_count"],
        f">={profile.min_ef_faster_sources} of {profile.source_count}",
        ef["faster_source_count"] >= profile.min_ef_faster_sources,
    )
    e_p95 = _percentile(combined_task["E"], 0.95)
    f_p95 = _percentile(combined_task["F"], 0.95)
    gates["E_to_F_task_p95"] = _gate(
        {"E": e_p95, "F": f_p95}, "F<=E", f_p95 <= e_p95
    )
    gates["E_to_F_makespan"] = _gate(
        {"E": mean_makespan["E"], "F": mean_makespan["F"]},
        "F<=1.03*E",
        mean_makespan["F"] <= 1.03 * mean_makespan["E"],
    )
    gates["A_to_F_every_block_mean_direction"] = _gate(
        [row["mean_absolute_reduction_s"] for row in af["blocks"]],
        "all >0",
        bool(af["every_block_mean_reduction_positive"]),
    )
    gates["A_to_F_mean_reduction"] = _gate(
        af["aggregate_relative_reduction"],
        ">=0.25",
        af["aggregate_relative_reduction"] >= 0.25,
    )
    gates["A_to_F_faster_sources"] = _gate(
        af["faster_source_count"],
        f">={profile.min_af_faster_sources} of {profile.source_count}",
        af["faster_source_count"] >= profile.min_af_faster_sources,
    )
    a_request_p99 = _percentile(combined_llm["A"], 0.99)
    f_request_p99 = _percentile(combined_llm["F"], 0.99)
    gates["A_to_F_request_p99"] = _gate(
        {"A": a_request_p99, "F": f_request_p99},
        "F<=1.25*A",
        f_request_p99 <= 1.25 * a_request_p99,
    )

    block_cell_summary = {
        block: {
            cell: {
                "task_e2e_s": runs[block][cell].summary["task_e2e_s"],
                "task_completion_makespan_s": runs[block][cell].summary[
                    "task_completion_makespan_s"
                ],
                "llm_request_duration_s": runs[block][cell].summary["llm"][
                    "request_duration_s"
                ],
                "completion_tokens": runs[block][cell].summary["llm"][
                    "completion_tokens"
                ],
                "tool": runs[block][cell].summary["tool"],
                "queue_timeline": runs[block][cell].summary["queue_timeline"],
                "physical_validation": physical[block][cell],
                "load_qualification": load[block][cell],
            }
            for cell in CELL_IDS
        }
        for block in sorted(runs)
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal_promotion_passed": passed,
        "inputs": inputs,
        "design": {
            "formal_profile": profile.name,
            "formal_workload": {
                "path": str(workload_path),
                "split_id": profile.split_id,
                "file_sha256": workload_validation["file_sha256"],
                "canonical_json_sha256": workload_validation[
                    "canonical_json_sha256"
                ],
                "canonical_sources_sha256": workload_validation[
                    "canonical_sources_sha256"
                ],
            },
            "block_count": len(blocks),
            "independent_source_count": profile.source_count,
            "replicas_per_source": profile.replicas_per_source,
            "tasks_per_cell_per_block": profile.tasks_per_cell,
            "logical_llm_requests_per_cell_per_block": 3 * profile.tasks_per_cell,
            "authoritative_tool_commits_per_cell_per_block": 2
            * profile.tasks_per_cell,
            "effective_bootstrap_sample_size": profile.source_count,
            "replicas_are_not_independent_samples": True,
            "fresh_server_blocks_are_repeated_measurements_not_independent_sources": True,
            "formal_load": {
                "max_active_tasks": profile.max_active_tasks,
                "vllm_max_num_seqs": profile.vllm_max_num_seqs,
                "context_padding_target_tokens": profile.context_padding_tokens,
                "visit_tool_capacity": profile.visit_tool_capacity,
                "visit_min_start_interval_s": profile.visit_min_start_interval_s,
                **(
                    {
                        "min_speculative_tool_workers": (
                            profile.min_speculative_tool_workers
                        )
                    }
                    if profile.name == "formal-v9"
                    else {}
                ),
                **(
                    {
                        "min_native_waiting_below_cap_fraction": (
                            profile.min_native_waiting_below_cap_fraction
                        ),
                        "min_authoritative_tool_queue_fraction": (
                            profile.min_authoritative_tool_queue_fraction
                        ),
                        "min_dual_queue_pressure_samples": (
                            profile.min_dual_queue_pressure_samples
                        ),
                        "min_dual_queue_pressure_consecutive_s": (
                            profile.min_dual_queue_pressure_consecutive_s
                        ),
                        "max_dual_queue_adjacent_sample_gap_s": (
                            profile.max_dual_queue_adjacent_sample_gap_s
                        ),
                    }
                    if profile.name in {"formal-v8", "formal-v9"}
                    else {}
                ),
                **(
                    {
                        "vllm_max_model_len": profile.vllm_max_model_len,
                        "vllm_max_num_batched_tokens": (
                            profile.vllm_max_num_batched_tokens
                        ),
                        "visit_canary_stride": profile.visit_canary_stride,
                        "expected_canary_count": profile.expected_canary_count,
                    }
                    if profile.require_modern_live_evidence
                    else {}
                ),
            },
            "source_estimator": (
                "one r00 observation per source/cell/block; then mean the "
                "three block-level source observations"
            ),
            "bootstrap_seed": profile.bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "block_orders": block_orders,
            "A_B_forward_count": ab_forward,
            "A_B_reverse_count": len(blocks) - ab_forward,
            "E_F_forward_count": ef_forward,
            "E_F_reverse_count": len(blocks) - ef_forward,
            "unique_fresh_server_instance_count": len(server_ids),
            "config_validation": config_validation,
            "identity_validation": identity_validation,
            "modern_runtime_validation": modern_runtime_validation,
            "v4_runtime_validation": (
                modern_runtime_validation if profile.name == "formal-v4" else None
            ),
            "v5_runtime_validation": (
                modern_runtime_validation if profile.name == "formal-v5" else None
            ),
            "v6_runtime_validation": (
                modern_runtime_validation if profile.name == "formal-v6" else None
            ),
            "v7_runtime_validation": (
                modern_runtime_validation if profile.name == "formal-v7" else None
            ),
            "v8_runtime_validation": (
                modern_runtime_validation if profile.name == "formal-v8" else None
            ),
            **(
                {
                    "v9_runtime_validation": modern_runtime_validation,
                    "v9_development_selection": v9_development_selection,
                    "v9_cell_provenance": v9_cell_provenance,
                }
                if profile.name == "formal-v9"
                else {}
            ),
        },
        "blocks": block_cell_summary,
        "aggregate_cells": {
            cell: {
                "source_e2e_s": aggregate_sources[cell],
                "source_distribution_s": _distribution(
                    list(aggregate_sources[cell].values())
                ),
                "task_e2e_s": _distribution(combined_task[cell]),
                "llm_request_duration_s": _distribution(combined_llm[cell]),
                "mean_task_completion_makespan_s": mean_makespan[cell],
                "completion_tokens": completion_tokens[cell],
                "uncontrolled_retry_count": uncontrolled_retry_counts[cell],
                "authoritative_retry": authoritative_retry[cell],
                "failed_physical_job_count": failure_counts[cell],
                "physical_http_attempt_count": sum(
                    physical[block][cell]["physical_http_attempt_count"]
                    for block in runs
                ),
                "retried_physical_job_count": sum(
                    physical[block][cell]["retried_physical_job_count"]
                    for block in runs
                ),
                "physical_service_s_including_retry_backoff": sum(
                    sum(
                        float(record.get("service_s") or 0.0)
                        for record in runs[block][cell].physical_records
                    )
                    for block in runs
                ),
            }
            for cell in CELL_IDS
        },
        "effects": effects,
        "interaction": interaction,
        "diagnostics": {
            "completion_token_relative_differences": token_differences,
            "completion_token_relative_differences_by_block": block_token_differences,
            "authoritative_retry": {
                "definition": (
                    "committed logical calls with http_attempts>1 / "
                    "authoritative commits"
                ),
                "cells": authoritative_retry,
                "E_to_F_absolute_rate_difference": ef_retry_difference,
                "A_to_F_absolute_rate_difference": af_retry_difference,
                "service_and_waste_include_attempts_and_fixed_backoff": True,
            },
            "speculation": {
                cell: {
                    **speculative[cell],
                    "exact_hit_count": exact_hits[cell],
                    "eligible_commit_count": eligible_commits[cell],
                    "exact_hit_rate": hit_rates[cell],
                }
                for cell in ("B", "F")
            },
            "canary": canary,
            "canary_pre_enqueue_skip": canary_skip,
            "guided_json_recovery": guided_json_recovery,
            "output_contract": output_contract,
            "strict_guided_final_contract": strict_guided_final_contract,
            "fixed_final_completion_contract": fixed_final_completion_contract,
            "E_to_F_component_decomposition": ef_component_decomposition,
            "AB_AE_BF_effects_are_reported_but_not_promotion_gates": True,
            "interaction_is_reported_but_not_a_promotion_gate": True,
        },
        "formal_gates": gates,
        "failed_gate_names": sorted(
            name for name, gate in gates.items() if not gate["passed"]
        ),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block",
        action="append",
        nargs=5,
        required=True,
        metavar=("BLOCK_ID", "A_RESULT", "B_RESULT", "E_RESULT", "F_RESULT"),
        help="Repeat exactly three times; paths are ordered A B E F.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--formal-workload",
        type=Path,
        help=(
            "Optional exact workload binding. If omitted, select the frozen v3-v9 "
            "file from the common workload_split_id embedded in all cells."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        blocks = [
            (row[0], *(Path(value) for value in row[1:])) for row in args.block
        ]
        result = aggregate_live_joint_four_cell(
            blocks,
            bootstrap_resamples=args.bootstrap_resamples,
            formal_workload=args.formal_workload,
        )
        _write_json_atomic(args.output, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"formal live four-cell aggregation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "formal_promotion_passed": result["formal_promotion_passed"],
                "failed_gate_names": result["failed_gate_names"],
                "E_to_F": result["effects"]["E_to_F"],
                "A_to_F": result["effects"]["A_to_F"],
                "interaction": result["interaction"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["formal_promotion_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
