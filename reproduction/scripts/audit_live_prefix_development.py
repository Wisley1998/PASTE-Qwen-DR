#!/usr/bin/env python3
"""Recompute prefix evidence for frozen development-only prefix cells.

This is deliberately a read-only, development-only audit.  It never discovers
workloads or runs: every allowed artifact directory is enumerated below, and
the report is emitted to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


LEGACY_CELLS = {
    "p0_off": "frozen_dev_s24_joint_physical_p0_off_r1",
    "p1_off": "frozen_dev_s24_joint_physical_p1_off_r1",
    "p0_searchvisit": "frozen_dev_s24_joint_physical_p0_searchvisit_r1",
    "p1_searchvisit": "frozen_dev_s24_joint_physical_p1_searchvisit_r1",
}
BLOCK2_CELLS = {
    "p0_nativeoff": "prefix_dev_block2_p0_nativeoff",
    "p1_nativeon": "prefix_dev_block2_p1_nativeon",
    "p2_affinity": "prefix_dev_block2_p2_affinity",
}
CORRECTED_P0_CELL = "prefix_dev_block3_p0_nativeoff"
PREFIX_MARKER = "[sched_policy_patch:prefix_locality]"
PREFIX_NUMERIC_FIELDS = (
    "waiting",
    "lookup_requests",
    "reused_requests",
    "hit_requests",
    "cached_tokens",
    "prompt_tokens",
    "marginal_prefill_tokens",
    "head_changed",
)
VLLM_METRICS = {
    "native_hits": "vllm:prefix_cache_hits_total",
    "native_queries": "vllm:prefix_cache_queries_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "inference_s": "vllm:request_inference_time_seconds_sum",
    "queue_s": "vllm:request_queue_time_seconds_sum",
    "preemptions": "vllm:num_preemptions_total",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def _parse_prefix_samples(log_text: str) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(log_text.splitlines(), 1):
        if PREFIX_MARKER not in line:
            continue
        sample: dict[str, Any] = dict(re.findall(r"(\w+)=([^ ]+)", line))
        for field in PREFIX_NUMERIC_FIELDS:
            sample[field] = int(sample[field])
        sample["line_number"] = line_number
        samples.append(sample)

    if not samples:
        return {
            "logged_sample_count": 0,
            "note": "No records; the explicit prefix-locality feature is off.",
        }

    prompt_tokens = sum(row["prompt_tokens"] for row in samples)
    cached_tokens = sum(row["cached_tokens"] for row in samples)
    return {
        # These are rate-limited snapshots, not exhaustive decisions and not
        # additive native-cache savings.
        "logged_sample_count": len(samples),
        "decision_counts": {
            value: sum(row["decision"] == value for row in samples)
            for value in sorted({row["decision"] for row in samples})
        },
        "reason_counts": {
            value: sum(row["reason"] == value for row in samples)
            for value in sorted({row["reason"] for row in samples})
        },
        "waiting_distribution": {
            str(value): sum(row["waiting"] == value for row in samples)
            for value in sorted({row["waiting"] for row in samples})
        },
        "lookup_requests_sum": sum(row["lookup_requests"] for row in samples),
        "reused_requests_sum": sum(row["reused_requests"] for row in samples),
        "hit_requests_sum": sum(row["hit_requests"] for row in samples),
        "samples_with_hit": sum(row["hit_requests"] > 0 for row in samples),
        "cached_tokens_sum": cached_tokens,
        "prompt_tokens_sum": prompt_tokens,
        "marginal_prefill_tokens_sum": sum(
            row["marginal_prefill_tokens"] for row in samples
        ),
        "cached_over_prompt_in_logged_samples": cached_tokens / prompt_tokens,
        "logged_head_changed_count": sum(row["head_changed"] for row in samples),
        "logged_heads_differ_count": sum(
            row["input_head"] != row["output_head"] for row in samples
        ),
        "changed_log_line_numbers": [
            row["line_number"]
            for row in samples
            if row["head_changed"] or row["input_head"] != row["output_head"]
        ],
        "marginal_token_identity_violations": sum(
            row["marginal_prefill_tokens"]
            != row["prompt_tokens"] - row["cached_tokens"]
            for row in samples
        ),
        "note": (
            "Rate-limited observations only; the configured log interval is "
            "reported separately. Sums must not be presented as unique requests "
            "or additional saved tokens."
        ),
    }


def _cell(
    artifact_root: Path,
    directory: str,
    *,
    log_directory: str = "server_log",
) -> dict[str, Any]:
    cell_root = artifact_root / directory
    result_path = cell_root / "cell" / "result.json"
    log_path = cell_root / log_directory / "vllm_8100.log"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    config = result["config"]
    effective_prefix_values = re.findall(
        r"enable_prefix_caching=(True|False)", log_text
    )
    declared_prefix_enabled = (
        config["scheduler_environment"].get("VLLM_ENABLE_PREFIX_CACHING") == "1"
    )
    effective_prefix_enabled = (
        effective_prefix_values[-1] == "True" if effective_prefix_values else None
    )
    native_log_rates = [
        float(value)
        for value in re.findall(r"Prefix cache hit rate: ([0-9.]+)%", log_text)
    ]
    gpu_kv_log_usage = [
        float(value)
        for value in re.findall(r"GPU KV cache usage: ([0-9.]+)%", log_text)
    ]
    summary = result["summary"]
    deltas = result["vllm_metric_deltas"]
    metrics = {name: float(deltas[key]) for name, key in VLLM_METRICS.items()}
    metrics["native_hit_fraction"] = (
        metrics["native_hits"] / metrics["native_queries"]
        if metrics["native_queries"]
        else None
    )
    metrics["uncached_prompt_tokens"] = (
        metrics["native_queries"] - metrics["native_hits"]
        if metrics["native_queries"]
        else metrics["prompt_tokens"]
        if effective_prefix_enabled is False
        else None
    )
    metrics["native_counter_denominator_available"] = bool(
        metrics["native_queries"]
    )
    metrics["queries_equal_prompt_tokens"] = (
        metrics["native_queries"] == metrics["prompt_tokens"]
    )

    committed = [
        row
        for row in result["tool_attempt_records"]
        if row.get("authoritative") and row.get("committed")
    ]
    tool: dict[str, Any] = {}
    for tool_name in ("search", "visit"):
        rows = [row for row in committed if row["tool"] == tool_name]
        tool[tool_name] = {
            field: _distribution([float(row[field]) for row in rows])
            for field in ("queue_s", "service_s", "exposed_wait_s")
        }
        tool[tool_name]["http_status_counts"] = {
            str(status): sum(row.get("response_status") == status for row in rows)
            for status in sorted({row.get("response_status") for row in rows})
        }

    native_samples: dict[str, Any] = {
        "count": len(native_log_rates),
        "values_percent": native_log_rates,
        "gpu_kv_usage_values_percent": gpu_kv_log_usage,
        "note": (
            "Periodic rounded log samples, so observed GPU KV maximum is not "
            "a continuous-time upper bound. The cumulative metric deltas are "
            "the authoritative whole-cell endpoint."
        ),
    }
    if native_log_rates:
        native_samples.update(
            {
                "first_percent": native_log_rates[0],
                "last_percent": native_log_rates[-1],
                "min_percent": min(native_log_rates),
                "max_percent": max(native_log_rates),
            }
        )
    if gpu_kv_log_usage:
        native_samples["gpu_kv_usage_observed_max_percent"] = max(
            gpu_kv_log_usage
        )

    llm_calls_per_task = summary["llm"]["request_count"] / summary["task_count"]
    tool_calls_per_task = len(committed) / summary["task_count"]
    mean_llm_component_s = summary["llm"]["mean_request_s"] * llm_calls_per_task
    mean_tool_component_s = (
        summary["tool"]["mean_exposed_wait_s"] * tool_calls_per_task
    )

    return {
        "directory": directory,
        "result_sha256": _sha256(result_path),
        "server_log_sha256": _sha256(log_path),
        "declared_native_prefix_enabled": declared_prefix_enabled,
        "effective_native_prefix_enabled": effective_prefix_enabled,
        "effective_native_prefix_log_values": effective_prefix_values,
        "declared_matches_effective_native_prefix": (
            declared_prefix_enabled == effective_prefix_enabled
            if effective_prefix_enabled is not None
            else False
        ),
        "prefix_locality_enabled": config["scheduler_environment"][
            "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY"
        ]
        == "1",
        "prefix_log_interval_s": float(
            config["scheduler_environment"].get(
                "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_LOG_INTERVAL_S", "1"
            )
        ),
        "speculation_mode": config["speculation_mode"],
        "all_tasks_succeeded": summary["all_tasks_succeeded"],
        "success_counts": {
            "tasks": summary["successful_task_count"],
            "llm_requests": summary["llm"]["successful_request_count"],
            "llm_exactly_one_attempt_each": summary["llm"][
                "exactly_one_attempt_each"
            ],
            "authoritative_tool_commits": summary["tool"][
                "authoritative_commit_count"
            ],
        },
        "task_e2e": {
            **summary["task_e2e"],
            "makespan_s": summary["makespan_s"],
        },
        "llm_mean_request_s": summary["llm"]["mean_request_s"],
        "llm_prompt_tokens": summary["llm"]["prompt_tokens"],
        "llm_completion_tokens": summary["llm"]["completion_tokens"],
        "mean_e2e_component_reconstruction": {
            "llm_calls_per_task": llm_calls_per_task,
            "tool_calls_per_task": tool_calls_per_task,
            "llm_s": mean_llm_component_s,
            "tool_exposed_wait_s": mean_tool_component_s,
            "residual_s": summary["task_e2e"]["mean_s"]
            - mean_llm_component_s
            - mean_tool_component_s,
        },
        "vllm_metric_deltas": metrics,
        "native_prefix_log_samples": native_samples,
        "prefix_locality_log": _parse_prefix_samples(log_text),
        "authoritative_tool": tool,
        "tool_mean_exposed_wait_s": summary["tool"]["mean_exposed_wait_s"],
        "tool_mean_queue_s": summary["tool"]["mean_queue_s"],
        "tool_mean_service_s": summary["tool"]["mean_service_s"],
        "queue_timeline_summary": result["queue_timeline_summary"],
        "expected_url_search_coverage": config["expected_url_search_coverage"],
        "workload_identity": {
            key: config.get(key)
            for key in (
                "workload_file_sha256",
                "selected_workload_sha256",
                "workload_split_id",
                "workload_split_role",
                "workload_formal_eligible",
                "independent_source_count",
                "task_count",
                "replicas",
                "call_graph_mode",
            )
        },
    }


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_task = baseline["task_e2e"]
    cand_task = candidate["task_e2e"]
    base_native = baseline["vllm_metric_deltas"]
    cand_native = candidate["vllm_metric_deltas"]
    base_components = baseline["mean_e2e_component_reconstruction"]
    cand_components = candidate["mean_e2e_component_reconstruction"]
    native_hit_rate_delta_pp = None
    if (
        base_native["native_hit_fraction"] is not None
        and cand_native["native_hit_fraction"] is not None
    ):
        native_hit_rate_delta_pp = 100.0 * (
            cand_native["native_hit_fraction"]
            - base_native["native_hit_fraction"]
        )
    return {
        "mean_task_saving_s": base_task["mean_s"] - cand_task["mean_s"],
        "mean_task_speedup_fraction": (
            base_task["mean_s"] - cand_task["mean_s"]
        )
        / base_task["mean_s"],
        "p50_speedup_fraction": (base_task["p50_s"] - cand_task["p50_s"])
        / base_task["p50_s"],
        "p95_speedup_fraction": (base_task["p95_s"] - cand_task["p95_s"])
        / base_task["p95_s"],
        "makespan_speedup_fraction": (
            base_task["makespan_s"] - cand_task["makespan_s"]
        )
        / base_task["makespan_s"],
        "llm_mean_request_speedup_fraction": (
            baseline["llm_mean_request_s"] - candidate["llm_mean_request_s"]
        )
        / baseline["llm_mean_request_s"],
        "completion_token_change_fraction": (
            candidate["llm_completion_tokens"] - baseline["llm_completion_tokens"]
        )
        / baseline["llm_completion_tokens"],
        "native_hit_rate_delta_percentage_points": native_hit_rate_delta_pp,
        "prefill_s_reduction_fraction": (
            base_native["prefill_s"] - cand_native["prefill_s"]
        )
        / base_native["prefill_s"],
        "inference_s_reduction_fraction": (
            base_native["inference_s"] - cand_native["inference_s"]
        )
        / base_native["inference_s"],
        "llm_queue_s_reduction_fraction": (
            base_native["queue_s"] - cand_native["queue_s"]
        )
        / base_native["queue_s"],
        "mean_e2e_component_delta_candidate_minus_baseline_s": {
            "llm_s": cand_components["llm_s"] - base_components["llm_s"],
            "tool_exposed_wait_s": cand_components["tool_exposed_wait_s"]
            - base_components["tool_exposed_wait_s"],
            "residual_s": cand_components["residual_s"]
            - base_components["residual_s"],
        },
    }


def _raw_result(artifact_root: Path, directory: str) -> dict[str, Any]:
    path = artifact_root / directory / "cell" / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_source_stats(
    artifact_root: Path,
    baseline_directory: str,
    candidate_directory: str,
) -> dict[str, Any]:
    baseline = _raw_result(artifact_root, baseline_directory)
    candidate = _raw_result(artifact_root, candidate_directory)
    baseline_tasks = {row["source_id"]: row for row in baseline["tasks"]}
    candidate_tasks = {row["source_id"]: row for row in candidate["tasks"]}
    if set(baseline_tasks) != set(candidate_tasks):
        raise ValueError("paired cells have different source IDs")
    savings = [
        float(baseline_tasks[source_id]["e2e_s"])
        - float(candidate_tasks[source_id]["e2e_s"])
        for source_id in sorted(baseline_tasks)
    ]
    generator = random.Random(20260816)
    bootstrap_means = [
        statistics.mean(
            savings[generator.randrange(len(savings))] for _ in savings
        )
        for _ in range(100_000)
    ]
    return {
        "source_count": len(savings),
        "faster_source_count": sum(value > 0.0 for value in savings),
        "slower_source_count": sum(value < 0.0 for value in savings),
        "tied_source_count": sum(value == 0.0 for value in savings),
        "mean_source_saving_s": statistics.mean(savings),
        "median_source_saving_s": statistics.median(savings),
        "bootstrap_seed": 20260816,
        "bootstrap_replicates": 100_000,
        "bootstrap_mean_saving_95pct_s": [
            _quantile(bootstrap_means, 0.025),
            _quantile(bootstrap_means, 0.975),
        ],
        "warning": (
            "This source bootstrap does not capture shared cell-level live-tool "
            "slowdowns or one-block run-order effects."
        ),
    }


def _config_differences(
    baseline: Any,
    candidate: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(baseline) | set(candidate)):
            differences.extend(
                _config_differences(
                    baseline.get(key, "<missing>"),
                    candidate.get(key, "<missing>"),
                    f"{path}/{key}",
                )
            )
        return differences
    if baseline == candidate:
        return []
    return [{"path": path, "baseline": baseline, "candidate": candidate}]


def _one_factor_config_audit(
    artifact_root: Path,
    baseline_directory: str,
    candidate_directory: str,
    expected_factor_path: str,
) -> dict[str, Any]:
    baseline_config = _raw_result(artifact_root, baseline_directory)["config"]
    candidate_config = _raw_result(artifact_root, candidate_directory)["config"]
    differences = _config_differences(baseline_config, candidate_config)
    runtime_observation_prefix = "/expected_url_search_coverage/"
    ignored_paths = {
        "/cell_label",
    }
    controlled_input_differences = [
        row
        for row in differences
        if row["path"] not in ignored_paths
        and not row["path"].startswith(runtime_observation_prefix)
    ]
    return {
        "all_result_config_differences": differences,
        "runtime_observation_differences": [
            row
            for row in differences
            if row["path"].startswith(runtime_observation_prefix)
        ],
        "controlled_input_differences": controlled_input_differences,
        "expected_only_factor_path": expected_factor_path,
        "declared_one_factor_match": [
            row["path"] for row in controlled_input_differences
        ]
        == [expected_factor_path],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("reproduction/artifacts/live_joint"),
    )
    args = parser.parse_args()
    legacy_cells = {
        name: _cell(args.artifacts_root, path)
        for name, path in LEGACY_CELLS.items()
    }
    block2_cells = {
        name: _cell(args.artifacts_root, path, log_directory="server")
        for name, path in BLOCK2_CELLS.items()
    }
    corrected_p0 = _cell(
        args.artifacts_root,
        CORRECTED_P0_CELL,
        log_directory="server",
    )
    p0_p1 = _comparison(
        block2_cells["p0_nativeoff"], block2_cells["p1_nativeon"]
    )
    p1_p2 = _comparison(
        block2_cells["p1_nativeon"], block2_cells["p2_affinity"]
    )
    p0_p1_sources = _paired_source_stats(
        args.artifacts_root,
        BLOCK2_CELLS["p0_nativeoff"],
        BLOCK2_CELLS["p1_nativeon"],
    )
    p1_p2_sources = _paired_source_stats(
        args.artifacts_root,
        BLOCK2_CELLS["p1_nativeon"],
        BLOCK2_CELLS["p2_affinity"],
    )
    corrected_p0_p1 = _comparison(corrected_p0, block2_cells["p1_nativeon"])
    corrected_p0_p1_sources = _paired_source_stats(
        args.artifacts_root,
        CORRECTED_P0_CELL,
        BLOCK2_CELLS["p1_nativeon"],
    )
    p0_p1_config = _one_factor_config_audit(
        args.artifacts_root,
        BLOCK2_CELLS["p0_nativeoff"],
        BLOCK2_CELLS["p1_nativeon"],
        "/scheduler_environment/VLLM_ENABLE_PREFIX_CACHING",
    )
    p1_p2_config = _one_factor_config_audit(
        args.artifacts_root,
        BLOCK2_CELLS["p1_nativeon"],
        BLOCK2_CELLS["p2_affinity"],
        "/scheduler_environment/VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY",
    )
    corrected_p0_p1_config = _one_factor_config_audit(
        args.artifacts_root,
        CORRECTED_P0_CELL,
        BLOCK2_CELLS["p1_nativeon"],
        "/scheduler_environment/VLLM_ENABLE_PREFIX_CACHING",
    )
    block2_results = {
        name: _raw_result(args.artifacts_root, directory)
        for name, directory in BLOCK2_CELLS.items()
    }
    source_id_sets = {
        name: sorted({row["source_id"] for row in result["tasks"]})
        for name, result in block2_results.items()
    }
    source_set_hashes = {
        name: hashlib.sha256(
            json.dumps(values, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for name, values in source_id_sets.items()
    }

    p2_mean_gate = p1_p2["mean_task_speedup_fraction"] >= 0.02
    p2_hit_gate = p1_p2["native_hit_rate_delta_percentage_points"] >= 3.0
    p2_bootstrap_gate = (
        p1_p2_sources["bootstrap_mean_saving_95pct_s"][0] > 0.0
    )
    p2_p95_gate = (
        block2_cells["p2_affinity"]["task_e2e"]["p95_s"]
        <= 1.03 * block2_cells["p1_nativeon"]["task_e2e"]["p95_s"]
    )
    p2_gates = {
        "mean_task_e2e_reduction_at_least_2pct": p2_mean_gate,
        "native_prefix_hit_rate_increase_at_least_3pp": p2_hit_gate,
        "paired_source_bootstrap_lower_bound_positive": p2_bootstrap_gate,
        "task_p95_within_3pct_of_p1": p2_p95_gate,
    }
    p0_effective_off = (
        block2_cells["p0_nativeoff"]["effective_native_prefix_enabled"] is False
    )
    report = {
        "schema_version": "live_prefix_development_audit_v3",
        "scope": (
            "enumerated frozen development cells only; no formal workload is read"
        ),
        "legacy_block1": {
            "cells": legacy_cells,
            "comparisons_p0_to_p1": {
                "speculation_off": _comparison(
                    legacy_cells["p0_off"], legacy_cells["p1_off"]
                ),
                "searchvisit": _comparison(
                    legacy_cells["p0_searchvisit"],
                    legacy_cells["p1_searchvisit"],
                ),
            },
            "attribution_boundary": (
                "Both legacy P0 and P1 retain native vLLM prefix caching."
            ),
        },
        "prefix_dev_block2": {
            "cells": block2_cells,
            "comparability": {
                "all_cells_24_of_24_tasks_succeeded": all(
                    cell["success_counts"]["tasks"] == 24
                    and cell["all_tasks_succeeded"]
                    for cell in block2_cells.values()
                ),
                "all_cells_72_of_72_llm_requests_succeeded_once": all(
                    cell["success_counts"]["llm_requests"] == 72
                    and cell["success_counts"]["llm_exactly_one_attempt_each"]
                    for cell in block2_cells.values()
                ),
                "all_cells_48_authoritative_commits": all(
                    cell["success_counts"]["authoritative_tool_commits"] == 48
                    for cell in block2_cells.values()
                ),
                "source_count_by_cell": {
                    name: len(values) for name, values in source_id_sets.items()
                },
                "source_set_sha256_by_cell": source_set_hashes,
                "identical_source_sets": len(set(source_set_hashes.values())) == 1,
                "identical_workload_identity": len(
                    {
                        json.dumps(
                            cell["workload_identity"],
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        for cell in block2_cells.values()
                    }
                )
                == 1,
                "p0_vs_p1_declared_config": p0_p1_config,
                "p1_vs_p2_declared_config": p1_p2_config,
            },
            "p0_effective_native_cache_off": p0_effective_off,
            "p0_native_ablation_valid": p0_effective_off
            and p0_p1_config["declared_one_factor_match"],
            "p0_native_ablation_failure": (
                None
                if p0_effective_off
                else (
                    "P0 declared VLLM_ENABLE_PREFIX_CACHING=0, but the engine "
                    "startup log reports enable_prefix_caching=True and native "
                    "hit counters are nonzero. P0->P1 is not a native-off/on "
                    "comparison."
                )
            ),
            "comparisons": {
                "p0_to_p1": {
                    **p0_p1,
                    "paired_sources": p0_p1_sources,
                    "valid_native_cache_effect": False,
                },
                "p1_to_p2": {
                    **p1_p2,
                    "paired_sources": p1_p2_sources,
                },
            },
            "p2_protocol_gates": p2_gates,
            "p2_all_protocol_gates_pass": all(p2_gates.values()),
            "selected_prefix_policy": (
                "P2_native_plus_affinity"
                if all(p2_gates.values())
                else "P1_native"
            ),
            "selection_reason": (
                "P2 is rejected because the native hit-rate gate and paired "
                "bootstrap gate fail. Its point E2E gain is explained by lower "
                "live-tool exposed wait while its LLM component regresses."
            ),
            "attribution_boundary": (
                "This is one sequential development block with an external live "
                "tool backend. Source bootstrap does not absorb the shared tool "
                "queue or run-order drift. P2 logs contain no observed head "
                "change, so the point E2E difference is not attributable to "
                "explicit affinity."
            ),
        },
        "corrected_native_off_diagnostic": {
            "cell": corrected_p0,
            "engine_config_valid": (
                corrected_p0["declared_native_prefix_enabled"] is False
                and corrected_p0["effective_native_prefix_enabled"] is False
                and corrected_p0["declared_matches_effective_native_prefix"]
            ),
            "native_counters_expected_for_disabled_cache": (
                corrected_p0["vllm_metric_deltas"]["native_hits"] == 0.0
                and corrected_p0["vllm_metric_deltas"]["native_queries"] == 0.0
            ),
            "versus_block2_p1_declared_config": corrected_p0_p1_config,
            "versus_block2_p1_descriptive_only": {
                **corrected_p0_p1,
                "paired_sources": corrected_p0_p1_sources,
            },
            "cross_run_boundary": (
                "The corrected P0 was run in block3, while P1 was run earlier "
                "in block2. This validates an effective native-off cell and "
                "allows descriptive LLM-counter comparison, but it is not a "
                "balanced same-block causal P0/P1 estimate. Live-tool queue and "
                "backend state differ across runs."
            ),
        },
        "historical_comparisons_p0_to_p1": {
            "speculation_off": _comparison(
                legacy_cells["p0_off"], legacy_cells["p1_off"]
            ),
            "searchvisit": _comparison(
                legacy_cells["p0_searchvisit"], legacy_cells["p1_searchvisit"]
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
