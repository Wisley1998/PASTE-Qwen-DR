#!/usr/bin/env python3
"""Build the fixed stress300 native-B versus physical-C causal screen.

The completed C is immutable.  This comparator revalidates both raw cells,
reuses the existing strict B/C pairing logic, replaces its legacy physical-log
subsection with parser-v2 evidence, and writes derived evidence only beside B.
Performance thresholds classify the retained result; they never decide whether
the completed run itself is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from summarize_strict_screening_bc import (  # noqa: E402
    ALLOWED_CONFIG_DIFFERENCES,
    summarize_strict_screening_bc,
)
from validate_native_admission_zero_write_v2 import (  # noqa: E402
    validate_native_zero_write_cell,
)
from validate_physical_kv_admission_v2 import (  # noqa: E402
    validate_fresh_physical_kv_cell,
)


SCHEMA = "paste_repro.strict_screening_bc_physical_v2"
VERSION = 1
B_PROFILE = "stress300_native320_g256_u86_native_exact_rescue120_b_screen"
C_PROFILE = "stress300_native320_g256_u86_physical093_exact_rescue120"
B_CONFIG_SHA256 = "e024ab17e6b08c1c1cd3246e4b74b253b681af152138af762bc536f7b513908e"
C_CONFIG_SHA256 = "1ee7dfe9f5831223fb4ff14c1e86154827d32d7835d11b2749c8e07863321d43"
C_PHYSICAL_VALIDATION_SHA256 = (
    "b292c04f0bdaf53ec9bea4ff290a8517f19cdc277d2eca908eb055c24dbf252e"
)
C_AC_SCREENING_SHA256 = (
    "906df1cd484311c3acbf701720d49cc3c0f516f5b48bf78e9e51ec1b5fcc7771"
)
C_SUMMARY_SHA256 = "15f42aa950ce16e0a40a114ce0e70fee52f32f7d70402c5c3a7a554d70d06742"
C_RAW_LOG_SHA256 = "c2eb67a5f6bb737991e485487fe08124a630a4c2f1d57db6e19ac37c34d9a17e"
MANIFEST_RELATIVE = (
    "reproduction/artifacts/workloads/fixed_three_way_cap512_floor64/"
    "manifest_stress300.json"
)
MANIFEST_SHA256 = "43f6d9dee3f12c4d31f7195e1616fa0ffd21ac98e8a7bdbffe3089be378318fa"
C_CELL_RELATIVE = (
    "reproduction/artifacts/stress300_u86_native320_g256_physical093_"
    "exact_rescue120/stress300_c_physical093_r1/"
    "stress300_c_physical093_r1_joint_learned"
)
EXPECTED_REQUESTS = 2595
EXPECTED_LOAD = 300
EXPECTED_ENGINE_SHAPE = {
    "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    "VLLM_HOST": "127.0.0.1",
    "VLLM_PROBE_HOST": "127.0.0.1",
    "VLLM_PORT": "8100",
    "VLLM_TP_SIZE": "4",
    "VLLM_DTYPE": "bfloat16",
    "VLLM_MAX_MODEL_LEN": "16384",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.86",
    "VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
    "VLLM_MAX_NUM_SEQS": "320",
    "VLLM_CUDA_GRAPH_SIZES": "256",
    "VLLM_USE_V1": "1",
}
EXPECTED_B_CONFIG = {
    "PASTE_FROZEN_CONFIG_SHA256": B_CONFIG_SHA256,
    "PASTE_STRESS_PROFILE": B_PROFILE,
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "1",
}
EXPECTED_C_CONFIG = {
    "PASTE_FROZEN_CONFIG_SHA256": C_CONFIG_SHA256,
    "PASTE_STRESS_PROFILE": C_PROFILE,
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "0",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": "0.93",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": "120",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S": "1",
}
EXPECTED_B_MISSING = {
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S",
}
BC_DEPENDENCY = SCRIPT_DIRECTORY / "summarize_strict_screening_bc.py"
NATIVE_VALIDATOR = SCRIPT_DIRECTORY / "validate_native_admission_zero_write_v2.py"
PHYSICAL_VALIDATOR = SCRIPT_DIRECTORY / "validate_physical_kv_admission_v2.py"
COMPARATOR_MODULE = Path(__file__).resolve()
CODE_PATHS = {
    "bc_pairing_dependency": BC_DEPENDENCY,
    "native_zero_write_validator": NATIVE_VALIDATOR,
    "physical_v2_validator": PHYSICAL_VALIDATOR,
    "candidate_metrics_dependency": SCRIPT_DIRECTORY / "summarize_candidate_d.py",
    "fixed_manifest_loader_dependency": SCRIPT_DIRECTORY / "summarize_four_cell.py",
    "pairing_bootstrap_dependency": SCRIPT_DIRECTORY / "summarize_paired_ad.py",
    "engine_guard_dependency": SCRIPT_DIRECTORY / "summarize_strict_screening_ad.py",
    "atomic_writer_dependency": REPRODUCTION_ROOT / "paste_repro" / "mapper.py",
    "runner_parser_dependency": REPOSITORY_ROOT / "scripts" / "run_vllm_trace_experiment.py",
    "scheduler_hook_dependency": REPOSITORY_ROOT / "scripts/pythonhooks/sched_policy_patch.py",
}
EXPECTED_CODE_SHA256 = {
    "bc_pairing_dependency": "90d0e21e4de3d89f437d2329b971b5d8d465dd7fdfd6c508a4367b0f52360daa",
    "native_zero_write_validator": "3f41ea555eb95f559c907a7c13308cc4c413a2c417ea0c0a4cdd1b669be0abd3",
    "physical_v2_validator": "434b65be5713a87238f7d60a7590632252ce585ac0e06c80ac57509bfc078a7d",
    "candidate_metrics_dependency": "16ad902059f04628a49cd6fcc84801681084be0e67beb70df899d75820d1a611",
    "fixed_manifest_loader_dependency": "0b0e49006ebe1f2b1ef1c12190290372b9e6826de9b612c860384bfb53a5c516",
    "pairing_bootstrap_dependency": "077f5c55677562b879b2cd5fd6563217827a39465e1ad032fb621182192ed1bf",
    "engine_guard_dependency": "d8b424595f2c4e3b6f691f4e117fd4d65f77190ce8becc02684b36966dd7b3d0",
    "atomic_writer_dependency": "e25ceef7e789c04c4592f06aaed5cc7323e1a7c5b9a79b978e20750856852358",
    "runner_parser_dependency": "f84b67254f57172a967fb81c973d0c0dfb4083869bf0b0ed9a8129000efa72b5",
    "scheduler_hook_dependency": "1636486cb440fb5bf85d1dced0dc5c3ace907c88aaaa52d5cef7741adbbdc342",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside the repository: {resolved}") from exc


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8", errors="strict"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require_saved_exact(
    path: Path,
    recomputed: Mapping[str, Any],
    label: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    saved = _load_object(path, label)
    actual_sha = _sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    if _canonical_json(saved) != _canonical_json(recomputed):
        raise ValueError(f"{label} differs from fresh recomputation")
    return {
        "path": _repo_relative(path),
        "sha256": actual_sha,
        "fresh_recomputation_exact_match": True,
    }


def _require_fixed_file(path: Path, expected_sha256: str, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": _repo_relative(path), "sha256": actual}


def _gate(observed: Any, threshold: Any, operator: str, passed: bool) -> dict[str, Any]:
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _result_boundaries(strict: Mapping[str, Any]) -> dict[str, Any]:
    cells = strict["cells"]
    comparison = strict["comparison"]
    source = strict["source_pairing"]
    b_metrics = cells["B"]
    c_metrics = cells["C"]

    token_comparison = comparison["execution"]["completion_tokens"]
    if not isinstance(token_comparison, Mapping):
        raise ValueError("B/C completion-token comparison is unavailable")
    raw_token_delta = token_comparison.get("c_relative_to_b")
    if not isinstance(raw_token_delta, (int, float)) or isinstance(raw_token_delta, bool):
        raise ValueError("B/C completion-token delta is not numeric")
    token_abs_delta = abs(float(raw_token_delta))

    b_request_p99 = float(b_metrics["request_latency_s"]["p99"])
    c_request_p99 = float(c_metrics["request_latency_s"]["p99"])
    b_count_gt_240 = int(b_metrics["request_latency_s"]["count_gt_240_s"])
    c_count_gt_240 = int(c_metrics["request_latency_s"]["count_gt_240_s"])
    b_task_p95 = float(b_metrics["task_flow_time_s"]["p95"])
    c_task_p95 = float(c_metrics["task_flow_time_s"]["p95"])
    b_makespan = float(b_metrics["task_makespan_s"])
    c_makespan = float(c_metrics["task_makespan_s"])
    comparability = {
        "completion_token_absolute_relative_difference_lt_1pct": _gate(
            token_abs_delta, 0.01, "<", token_abs_delta < 0.01
        ),
        "request_p99_not_above_1_5x_b": {
            "b_s": b_request_p99,
            "c_s": c_request_p99,
            "ratio": c_request_p99 / b_request_p99 if b_request_p99 else None,
            "maximum_ratio": 1.5,
            "passed": c_request_p99 <= 1.5 * b_request_p99,
        },
        "request_count_gt_240s_not_increased": {
            "b": b_count_gt_240,
            "c": c_count_gt_240,
            "passed": c_count_gt_240 <= b_count_gt_240,
        },
        "task_p95_not_regressed": {
            "b_s": b_task_p95,
            "c_s": c_task_p95,
            "passed": c_task_p95 <= b_task_p95,
        },
        "makespan_not_regressed_over_3pct": {
            "b_s": b_makespan,
            "c_s": c_makespan,
            "ratio": c_makespan / b_makespan if b_makespan else None,
            "maximum_ratio": 1.03,
            "passed": c_makespan <= 1.03 * b_makespan,
        },
    }
    comparability["passed"] = all(
        item["passed"] for item in comparability.values() if isinstance(item, Mapping)
    )

    mean_reduction = float(
        comparison["task_flow_time_s"]["mean"]["relative_reduction"]
    )
    source_outcomes = source["source_session_outcomes"]
    c_faster = int(source_outcomes["c_faster"])
    source_count = int(source["independent_source_session_count"])
    ci_lower = float(source["independent_source_mean_bootstrap_95_ci_s"]["lower_s"])
    benefit = {
        "mean_task_e2e_reduction_above_zero": _gate(
            mean_reduction, 0.0, ">", mean_reduction > 0.0
        ),
        "strict_majority_of_independent_sources_faster": {
            "observed": c_faster,
            "minimum": source_count // 2 + 1,
            "source_count": source_count,
            "passed": c_faster >= source_count // 2 + 1,
        },
        "source_bootstrap_95pct_lower_above_zero": _gate(
            ci_lower, 0.0, ">", ci_lower > 0.0
        ),
    }
    benefit["passed"] = all(
        item["passed"] for item in benefit.values() if isinstance(item, Mapping)
    )
    promotion = bool(comparability["passed"] and benefit["passed"])
    return {
        "classification_only_not_run_completion_gates": True,
        "comparability_and_tail": comparability,
        "incremental_physical_admission_benefit": benefit,
        "promotion_passed": promotion,
        "classification": (
            "accepted_incremental_physical_admission_benefit"
            if promotion
            else "valid_screen_not_promoted"
        ),
    }


def summarize_fixed_stress300_bc(
    *, manifest_path: Path, b_run: Path, c_run: Path
) -> dict[str, Any]:
    manifest = manifest_path.resolve()
    expected_manifest = (REPOSITORY_ROOT / MANIFEST_RELATIVE).resolve()
    if manifest != expected_manifest:
        raise ValueError("manifest must be the frozen stress300 five-copy manifest")
    manifest_binding = _require_fixed_file(manifest, MANIFEST_SHA256, "manifest")

    b_path = b_run.resolve()
    c_path = c_run.resolve()
    expected_c_path = (REPOSITORY_ROOT / C_CELL_RELATIVE).resolve()
    if c_path != expected_c_path:
        raise ValueError("C must be the frozen completed physical093 reference cell")
    if b_path == c_path or not b_path.is_dir():
        raise ValueError("B must be a distinct completed cell")
    output_root = b_path.parent
    b_validation_path = output_root / "native_admission_zero_write_v2.json"
    c_root = c_path.parent
    c_validation_path = c_root / "physical_kv_validation_v2.json"
    c_ac_path = c_root / "strict_a_vs_c_physical_v2.json"
    c_summary_path = c_path / "summary.json"
    c_raw_path = c_path / "server" / "vllm_8100.log"

    current_code = {
        name: _sha256_file(path) for name, path in CODE_PATHS.items()
    }
    if current_code != EXPECTED_CODE_SHA256:
        raise ValueError("preregistered validator/comparator dependency code drifted")

    b_validation = validate_native_zero_write_cell(
        b_path,
        expected_profile=B_PROFILE,
        expected_load=EXPECTED_LOAD,
        expected_requests=EXPECTED_REQUESTS,
        expected_config_sha256=B_CONFIG_SHA256,
        expected_engine_shape=EXPECTED_ENGINE_SHAPE,
        expected_keepalive_s=60,
    )
    b_binding = _require_saved_exact(
        b_validation_path, b_validation, "saved native B zero-write validation"
    )
    c_validation = validate_fresh_physical_kv_cell(
        c_path,
        expected_profile=C_PROFILE,
        expected_load=EXPECTED_LOAD,
        expected_requests=EXPECTED_REQUESTS,
        expected_config_sha256=C_CONFIG_SHA256,
        expected_engine_shape=EXPECTED_ENGINE_SHAPE,
        expected_num_gpu_blocks=44178,
        expected_block_size=16,
        expected_target_utilization=0.93,
        expected_keepalive_s=60,
        expected_preemptions=0,
    )
    c_binding = _require_saved_exact(
        c_validation_path,
        c_validation,
        "saved physical C parser-v2 validation",
        expected_sha256=C_PHYSICAL_VALIDATION_SHA256,
    )
    c_ac_binding = _require_fixed_file(
        c_ac_path, C_AC_SCREENING_SHA256, "frozen C strict A/C evidence"
    )
    c_ac = _load_object(c_ac_path, "frozen C strict A/C evidence")
    if (
        c_ac.get("schema") != "paste_repro.strict_screening_ac_physical_v2"
        or c_ac.get("version") != 1
        or c_ac.get("status") != "screening_reuses_previous_a_not_fresh_server_pair"
    ):
        raise ValueError("frozen C strict A/C evidence identity is invalid")
    c_summary_binding = _require_fixed_file(
        c_summary_path, C_SUMMARY_SHA256, "frozen C summary"
    )
    c_raw_binding = _require_fixed_file(
        c_raw_path, C_RAW_LOG_SHA256, "frozen C canonical raw log"
    )

    strict = summarize_strict_screening_bc(
        manifest_path=manifest,
        role="stress",
        b_run=b_path,
        c_run=c_path,
        expected_b_config=EXPECTED_B_CONFIG,
        expected_c_config=EXPECTED_C_CONFIG,
        expected_b_config_missing=set(EXPECTED_B_MISSING),
        expected_c_config_missing=set(),
        c_physical_revalidation=None,
        verify_frozen_configs=True,
    )
    config_guard = strict["comparison_invariants"]["scheduler_configuration_guard"]
    if set(config_guard["actual_difference_keys"]) != set(ALLOWED_CONFIG_DIFFERENCES):
        raise ValueError("B/C scheduler diff is not the exact seven-key allowlist")
    result_boundaries = _result_boundaries(strict)

    # The legacy strict helper double-counts copied and canonical C log markers.
    # Keep its pairing/metrics result, but replace that entire evidence branch
    # with independently recomputed, canonical parser-v2 artifacts.
    strict.pop("physical_kv_admission_evidence", None)
    strict.update(
        {
            "schema": SCHEMA,
            "version": VERSION,
            "status": "valid_incremental_single_screen",
            "fixed_artifact_bindings": {
                "manifest": manifest_binding,
                "B_native_zero_write_validation": b_binding,
                "C_physical_v2_validation": c_binding,
                "C_strict_A_vs_C_context": c_ac_binding,
                "C_summary": c_summary_binding,
                "C_canonical_raw_log": c_raw_binding,
            },
            "code_binding": {
                "comparator": {
                    "path": _repo_relative(COMPARATOR_MODULE),
                    "sha256": _sha256_file(COMPARATOR_MODULE),
                },
                "preregistered_dependencies": {
                    name: {
                        "path": _repo_relative(path),
                        "sha256": current_code[name],
                    }
                    for name, path in sorted(CODE_PATHS.items())
                },
            },
            "admission_evidence": {
                "B_native_reorder_only": b_validation,
                "C_adaptive_physical_kv": c_validation,
                "exact_scheduler_configuration_difference_count": 7,
                "exact_scheduler_configuration_difference_keys": sorted(
                    ALLOWED_CONFIG_DIFFERENCES
                ),
                "B_physical_controller_capacity_write_count": 0,
                "C_capacity_write_source": "physical_kv",
                "C_dynamic_cap_min": c_validation["physical_kv"]["effective_cap_min"],
                "C_dynamic_cap_max": c_validation["physical_kv"]["effective_cap_max"],
                "C_pressure_above_64_sample_count": c_validation["physical_kv"][
                    "pressure_above_64_sample_count"
                ],
                "passed": True,
            },
            "result_boundaries": result_boundaries,
            "interpretation": (
                "B and C have the same fixed 60-source/300-instance workload, "
                "Joint ordering, learned overlap, keepalive60 transport, engine "
                "shape, calibration, mapper, and request identities. Their exact "
                "seven-key scheduler diff changes only native reorder-only versus "
                "adaptive physical-KV admission. B-C is therefore an incremental "
                "single-screen estimate of physical admission for this workload. "
                "B was run later against an immutable historical C, not as a fresh "
                "contemporaneous randomized pair; deterministic copies are folded "
                "to 60 independent source-session means for bootstrap inference."
            ),
        }
    )
    return strict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--c-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        b_path = args.b_run.resolve()
        output = args.output.resolve()
        if output != b_path.parent / "strict_b_vs_c_physical_v2.json":
            raise ValueError(
                "--output must be strict_b_vs_c_physical_v2.json directly under B root"
            )
        result = summarize_fixed_stress300_bc(
            manifest_path=args.manifest,
            b_run=b_path,
            c_run=args.c_run,
        )
        write_json_atomic(output, result)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
