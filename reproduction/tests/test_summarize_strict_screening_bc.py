from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from reproduction.tests.test_summarize_paired_ad import _paired_fixture  # noqa: E402
from summarize_strict_screening_bc import (  # noqa: E402
    ALLOWED_CONFIG_DIFFERENCES,
    LEGACY_REJECTION_REASON,
    LEGACY_REJECTION_SHAPE,
    PHYSICAL_PARSER_MODULE,
    PHYSICAL_CONFIG_KEYS,
    PHYSICAL_REVALIDATION_SCHEMA,
    PHYSICAL_REVALIDATION_STATUS,
    PHYSICAL_REVALIDATION_VERSION,
    PHYSICAL_VALIDATOR_MODULE,
    REPOSITORY_ROOT,
    summarize_strict_screening_bc,
)
from run_vllm_trace_experiment import (  # noqa: E402
    PHYSICAL_KV_LOG_PARSER_ID,
    PHYSICAL_KV_LOG_PARSER_VERSION,
    parse_vllm_log_segment,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _physical_line(sample: dict[str, object]) -> str:
    order = (
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
    return "[sched_policy_patch:physical_kv] " + " ".join(
        f"{key}={sample[key]}" for key in order
    )


def _empty_physical_evidence() -> dict[str, object]:
    return {
        "sample_count": 0,
        "malformed_sample_count": 0,
        "fail_closed_count": 0,
        "fail_closed_reasons": [],
        "capacity_write_count": {"min": None, "max": None, "mean": None},
        "samples": [],
        "screening_gates": {"has_samples": False, "passed": False},
    }


def _passing_physical_evidence() -> dict[str, object]:
    caps = [80, 82, 79, 70, 81, 80, 83, 78, 82, 80]
    samples = []
    for index, cap in enumerate(caps, 1):
        admit = cap - 70
        samples.append(
            {
                "decision": "admit",
                "reason": "forecast_hold" if admit == 0 else "budget",
                "num_gpu_blocks": 45119,
                "block_size": 16,
                "capacity_tokens": 721904,
                "target_utilization": 0.93,
                "budget_tokens": 671360,
                "usage": 0.90,
                "live_tokens": 600000,
                "logical_live_tokens": 580000,
                "running_growth_tokens": 10000,
                "reserved_tokens": 10000,
                "committed_tokens": 600000,
                "predicted_admit_tokens": admit * 1000,
                "waiting": 100 - index,
                "running": 70,
                "fit_admit": admit,
                "admit": admit,
                "effective_cap": cap,
                "native_cap": 256,
                "capacity_write_source": "physical_kv",
                "capacity_write_count": index,
                "rescue": 0,
            }
        )
    checks = {
        "has_samples": True,
        "no_malformed_samples": True,
        "no_fail_closed_decisions": True,
        "stable_physical_capacity": True,
        "at_least_three_effective_caps": True,
        "observed_cap_increase": True,
        "observed_cap_decrease": True,
        "observed_zero_fit_admit": True,
        "observed_positive_fit_admit": True,
        "at_least_ten_pressure_samples_above_64": True,
        "passed": True,
    }
    return {
        "sample_count": len(samples),
        "malformed_sample_count": 0,
        "fail_closed_count": 0,
        "fail_closed_reasons": [],
        "capacity_tokens": {"min": 721904.0, "max": 721904.0, "mean": 721904.0},
        "target_utilization": {"min": 0.93, "max": 0.93, "mean": 0.93},
        "effective_cap": {"min": 70.0, "max": 83.0, "mean": 79.5, "unique_count": 7},
        "native_cap": {"min": 256.0, "max": 256.0, "mean": 256.0},
        "capacity_write_count": {"min": 1.0, "max": 10.0, "mean": 5.5},
        "effective_cap_increase_count": 4,
        "effective_cap_decrease_count": 5,
        "fit_admit_zero_sample_count": 1,
        "fit_admit_positive_sample_count": 9,
        "pressure_above_64_sample_count": 10,
        "samples": samples,
        "screening_gates": checks,
    }


def _set_physical_evidence(run: Path, evidence: dict[str, object]) -> None:
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["physical_kv_admission"] = copy.deepcopy(evidence)
    _write_json(summary_path, summary)

    sidecar_path = run / "vllm_log_summary.json"
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    else:
        sidecar = {"max_swapped_requests": 0, "preemption_warning_count": 0}
    sidecar["physical_kv_admission"] = copy.deepcopy(evidence)
    _write_json(sidecar_path, sidecar)


def _bc_fixture(
    root: Path,
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, str],
    dict[str, str],
    set[str],
    set[str],
]:
    manifest, pairs = _paired_fixture(root)
    source_d = pairs[0][1]
    b_run = root / "b_run"
    c_run = root / "c_run"
    shutil.copytree(source_d, b_run)
    shutil.copytree(source_d, c_run)

    b_summary_path = b_run / "summary.json"
    c_summary_path = c_run / "summary.json"
    b_summary = json.loads(b_summary_path.read_text(encoding="utf-8"))
    c_summary = json.loads(c_summary_path.read_text(encoding="utf-8"))
    b_environment = b_summary["scheduler_environment"]
    c_environment = c_summary["scheduler_environment"]
    b_environment["VLLM_MAX_NUM_SEQS"] = "256"
    c_environment["VLLM_MAX_NUM_SEQS"] = "256"
    b_environment["VLLM_PORT"] = "8100"
    c_environment["VLLM_PORT"] = "8100"
    b_values = {
        "PASTE_STRESS_PROFILE": "stress240_joint_native_b",
        "PASTE_FROZEN_CONFIG_SHA256": "b" * 64,
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "1",
    }
    c_values = {
        "PASTE_STRESS_PROFILE": "stress240_joint_physical_c93",
        "PASTE_FROZEN_CONFIG_SHA256": "c" * 64,
        "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "0",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": "0.93",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": "120",
        "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S": "1",
    }
    for key in ALLOWED_CONFIG_DIFFERENCES:
        b_environment.pop(key, None)
        c_environment.pop(key, None)
    b_environment.update(b_values)
    c_environment.update(c_values)
    _write_json(b_summary_path, b_summary)
    _write_json(c_summary_path, c_summary)

    b_evidence = _empty_physical_evidence()
    c_evidence = _passing_physical_evidence()
    _set_physical_evidence(b_run, b_evidence)
    _set_physical_evidence(c_run, c_evidence)
    with (c_run / "server.log").open("a", encoding="utf-8") as handle:
        for _ in c_evidence["samples"]:
            handle.write("[sched_policy_patch:physical_kv] synthetic=1\n")

    return (
        manifest,
        b_run,
        c_run,
        b_values,
        c_values,
        set(PHYSICAL_CONFIG_KEYS),
        set(),
    )


def _revalidation_fixture(
    root: Path,
) -> tuple[
    tuple[
        Path,
        Path,
        Path,
        dict[str, str],
        dict[str, str],
        set[str],
        set[str],
    ],
    Path,
    Path,
]:
    fixture = _bc_fixture(root)
    _, _, c_run, _, _, _, _ = fixture
    accepted = copy.deepcopy(_passing_physical_evidence()["samples"])
    for index, sample in enumerate(accepted, 2):
        sample["capacity_write_count"] = index

    rejected = copy.deepcopy(next(sample for sample in accepted if sample["admit"] == 0))
    rejected.update(
        {
            "reason": "forecast_hold",
            "committed_tokens": int(rejected["budget_tokens"]) + 1,
            "predicted_admit_tokens": 0,
            "capacity_write_count": len(accepted) + 2,
        }
    )
    selected = [*accepted, rejected]
    selected_lines = [_physical_line(sample) for sample in selected]
    recomputed = parse_vllm_log_segment("\n".join(selected_lines))[
        "physical_kv_admission"
    ]
    legacy_lines = [
        *selected_lines[:-1],
        selected_lines[-1].replace("decision=admit", "decision=legacy_rejected", 1),
    ]
    original = parse_vllm_log_segment("\n".join(legacy_lines))[
        "physical_kv_admission"
    ]
    if original["malformed_sample_count"] != 1 or recomputed["sample_count"] != 11:
        raise AssertionError("synthetic revalidation fixture did not exercise parser v2")
    _set_physical_evidence(c_run, original)

    prefix = copy.deepcopy(selected[0])
    prefix["capacity_write_count"] = 1
    raw_lines = [_physical_line(prefix), *selected_lines]
    raw_path = c_run / "server" / "vllm_8100.log"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    positive = [
        sample
        for sample in selected
        if sample["rescue"] == 0 and sample["admit"] > 0
    ]
    zero = [
        sample
        for sample in selected
        if sample["rescue"] == 0 and sample["admit"] == 0
    ]
    rescue = [sample for sample in selected if sample["rescue"] == 1]
    sample_count = len(selected)
    invariants = {
        "sample_count": sample_count,
        "capacity_equation_pass_count": sample_count,
        "effective_cap_equation_pass_count": sample_count,
        "live_within_physical_capacity_pass_count": sample_count,
        "nonrescue_positive_admit_count": len(positive),
        "nonrescue_positive_admit_within_soft_budget_count": len(positive),
        "nonrescue_zero_admit_count": len(zero),
        "forecast_hold_over_soft_budget_zero_admit_count": 1,
        "rescue_count": len(rescue),
        "rescue_within_physical_capacity_count": len(rescue),
        "capacity_write_source_physical_count": sample_count,
        "native_cap_bound_pass_count": sample_count,
        "all_passed": True,
    }
    summary_path = c_run / "summary.json"
    vllm_summary_path = c_run / "vllm_log_summary.json"
    parser_path = REPOSITORY_ROOT / PHYSICAL_PARSER_MODULE
    validator_path = REPOSITORY_ROOT / PHYSICAL_VALIDATOR_MODULE
    sidecar = {
        "schema": PHYSICAL_REVALIDATION_SCHEMA,
        "version": PHYSICAL_REVALIDATION_VERSION,
        "status": PHYSICAL_REVALIDATION_STATUS,
        "source": {
            "raw_log": {
                "path": _repo_relative(raw_path),
                "sha256": _sha256(raw_path),
                "size_bytes": raw_path.stat().st_size,
                "scope": "full_server_lifecycle",
                "marker_count": len(raw_lines),
            },
            "summary": {
                "path": _repo_relative(summary_path),
                "sha256": _sha256(summary_path),
            },
            "vllm_log_summary": {
                "path": _repo_relative(vllm_summary_path),
                "sha256": _sha256(vllm_summary_path),
            },
        },
        "parser": {
            "id": PHYSICAL_KV_LOG_PARSER_ID,
            "version": PHYSICAL_KV_LOG_PARSER_VERSION,
            "module_path": PHYSICAL_PARSER_MODULE,
            "module_sha256": _sha256(parser_path),
        },
        "validator": {
            "path": PHYSICAL_VALIDATOR_MODULE,
            "sha256": _sha256(validator_path),
        },
        "original_post_run_validation": {
            "status": "failed",
            "sample_count": original["sample_count"],
            "malformed_sample_count": original["malformed_sample_count"],
            "fail_closed_count": original["fail_closed_count"],
            "screening_gates": copy.deepcopy(original["screening_gates"]),
        },
        "recomputed": {
            "physical_kv_admission": recomputed,
            "independent_sample_audit": {
                "experiment_scope": {
                    "derivation": "legacy_stored_telemetry_exact_match",
                    "marker_count": sample_count,
                    "selected_capacity_write_count_first": 2,
                    "selected_capacity_write_count_last": sample_count + 1,
                    "excluded_prefix_marker_count": 1,
                    "excluded_prefix_capacity_write_counts": [1],
                    "excluded_suffix_marker_count": 0,
                    "excluded_suffix_capacity_write_counts": [],
                    "raw_marker_accounting_exact": True,
                    "capacity_write_counts_strictly_increasing": True,
                    "legacy_accepted_sample_count": len(accepted),
                    "legacy_rejected_sample_count": 1,
                    "legacy_accepted_samples_exact_match": True,
                    "legacy_aggregate_exact_match": True,
                },
                "full_raw_scope": {
                    "marker_count": len(raw_lines),
                    "sample_count": len(raw_lines),
                    "malformed_sample_count": 0,
                    "fail_closed_count": 0,
                    "capacity_write_count_first": 1,
                    "capacity_write_count_last": sample_count + 1,
                    "capacity_write_counts_strictly_increasing": True,
                },
                "legacy_rejection_reason_counts": {LEGACY_REJECTION_REASON: 1},
                "legacy_rejection_line_shape_counts": {LEGACY_REJECTION_SHAPE: 1},
                "invariants": invariants,
                "conclusion": "all_experiment_samples_safe",
            },
        },
    }
    sidecar_path = c_run.parent / "physical_kv_revalidation.json"
    _write_json(sidecar_path, sidecar)
    return fixture, sidecar_path, raw_path


def _run_fixture(
    fixture: tuple[
        Path,
        Path,
        Path,
        dict[str, str],
        dict[str, str],
        set[str],
        set[str],
    ],
    *,
    revalidation: Path | None = None,
) -> dict[str, object]:
    manifest, b_run, c_run, b_values, c_values, b_missing, c_missing = fixture
    return summarize_strict_screening_bc(
        manifest_path=manifest,
        role="final",
        b_run=b_run,
        c_run=c_run,
        expected_b_config=b_values,
        expected_c_config=c_values,
        expected_b_config_missing=b_missing,
        expected_c_config_missing=c_missing,
        c_physical_revalidation=revalidation,
        required_engine_keys=("VLLM_MAX_NUM_SEQS",),
        verify_frozen_configs=False,
    )


class StrictScreeningBCTests(unittest.TestCase):
    def test_strict_pair_reports_metrics_source_bootstrap_and_physical_gates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (
                manifest,
                b_run,
                c_run,
                b_values,
                c_values,
                b_missing,
                c_missing,
            ) = _bc_fixture(Path(temporary))
            result = summarize_strict_screening_bc(
                manifest_path=manifest,
                role="final",
                b_run=b_run,
                c_run=c_run,
                expected_b_config=b_values,
                expected_c_config=c_values,
                expected_b_config_missing=b_missing,
                expected_c_config_missing=c_missing,
                required_engine_keys=(
                    "VLLM_MAX_NUM_SEQS",
                    "VLLM_SCHED_PRED_OUT_ENABLE",
                ),
                verify_frozen_configs=False,
            )

        invariants = result["comparison_invariants"]
        self.assertTrue(invariants["request_identity_exact_match"])
        self.assertTrue(invariants["policy_and_overlap_identical"])
        self.assertTrue(
            invariants["scheduler_configuration_guard"]["exact_allowlist_match"]
        )
        self.assertIn("p99", result["cells"]["B"]["task_flow_time_s"])
        self.assertIn("max", result["cells"]["C"]["request_latency_s"])
        self.assertIn(
            "count_gt_120_s", result["comparison"]["request_tail_counts"]
        )
        self.assertIn("execution", result["comparison"])
        self.assertIn("queue", result["task_saving_decomposition"]["components_s"])
        self.assertEqual(result["source_pairing"]["load_instance_count"], 2)
        self.assertEqual(
            result["source_pairing"]["independent_source_session_count"], 2
        )
        self.assertEqual(
            result["source_pairing"]
            ["independent_source_mean_bootstrap_95_ci_s"]["sample_size"],
            2,
        )
        self.assertEqual(
            result["source_pairing"]
            ["independent_source_mean_bootstrap_95_ci_s"]["estimand"],
            "mean_B_minus_C_task_flow_s",
        )
        audit = result["physical_kv_admission_evidence"]["independent_gate_audit"]
        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["B_zero_capacity_writes"]["capacity_write_count"], 0
        )
        self.assertGreater(
            audit["C_dynamic_physical_kv_gates"]["effective_cap_max"], 64
        )

    def test_unwhitelisted_configuration_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bc_fixture(Path(temporary))
            manifest, b_run, c_run, b_values, c_values, b_missing, c_missing = fixture
            summary_path = c_run / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scheduler_environment"]["VLLM_MAX_NUM_BATCHED_TOKENS"] = "9999"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "unexpected=.*BATCHED_TOKENS"):
                summarize_strict_screening_bc(
                    manifest_path=manifest,
                    role="final",
                    b_run=b_run,
                    c_run=c_run,
                    expected_b_config=b_values,
                    expected_c_config=c_values,
                    expected_b_config_missing=b_missing,
                    expected_c_config_missing=c_missing,
                    required_engine_keys=("VLLM_MAX_NUM_SEQS",),
                    verify_frozen_configs=False,
                )

    def test_c_malformed_or_failed_physical_gates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bc_fixture(Path(temporary))
            manifest, b_run, c_run, b_values, c_values, b_missing, c_missing = fixture
            evidence = _passing_physical_evidence()
            evidence["malformed_sample_count"] = 1
            evidence["screening_gates"]["no_malformed_samples"] = False
            evidence["screening_gates"]["passed"] = False
            _set_physical_evidence(c_run, evidence)
            with self.assertRaisesRegex(ValueError, "malformed samples"):
                summarize_strict_screening_bc(
                    manifest_path=manifest,
                    role="final",
                    b_run=b_run,
                    c_run=c_run,
                    expected_b_config=b_values,
                    expected_c_config=c_values,
                    expected_b_config_missing=b_missing,
                    expected_c_config_missing=c_missing,
                    required_engine_keys=("VLLM_MAX_NUM_SEQS",),
                    verify_frozen_configs=False,
                )

    def test_b_capacity_write_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _bc_fixture(Path(temporary))
            manifest, b_run, c_run, b_values, c_values, b_missing, c_missing = fixture
            evidence = _empty_physical_evidence()
            evidence["capacity_write_count"] = {"min": 1, "max": 1, "mean": 1}
            _set_physical_evidence(b_run, evidence)
            with self.assertRaisesRegex(ValueError, "unexpectedly contains"):
                summarize_strict_screening_bc(
                    manifest_path=manifest,
                    role="final",
                    b_run=b_run,
                    c_run=c_run,
                    expected_b_config=b_values,
                    expected_c_config=c_values,
                    expected_b_config_missing=b_missing,
                    expected_c_config_missing=c_missing,
                    required_engine_keys=("VLLM_MAX_NUM_SEQS",),
                    verify_frozen_configs=False,
                )

    def test_raw_log_revalidation_preserves_original_failure(self) -> None:
        test_root = REPOSITORY_ROOT / "reproduction" / "tests"
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            fixture, sidecar, _ = _revalidation_fixture(Path(temporary))
            result = _run_fixture(fixture, revalidation=sidecar)

        physical = result["physical_kv_admission_evidence"]
        self.assertEqual(
            physical["C_evidence_basis"], "accepted_raw_log_revalidation"
        )
        self.assertEqual(physical["C"]["sample_count"], 11)
        self.assertEqual(
            physical["C_original_physical_kv_artifact"][
                "malformed_sample_count"
            ],
            1,
        )
        self.assertEqual(
            physical["C_original_post_run_validation"]["status"], "failed"
        )
        revalidation = physical["C_raw_log_revalidation"]
        self.assertTrue(revalidation["accepted"])
        self.assertEqual(revalidation["status"], PHYSICAL_REVALIDATION_STATUS)
        self.assertEqual(
            revalidation["original_post_run_validation"]["status"], "failed"
        )
        self.assertEqual(
            revalidation["original_post_run_validation"][
                "malformed_sample_count"
            ],
            1,
        )
        self.assertTrue(revalidation["original_artifacts_preserved"])

    def test_original_malformed_failure_without_revalidation_still_fails(self) -> None:
        test_root = REPOSITORY_ROOT / "reproduction" / "tests"
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            fixture, _, _ = _revalidation_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "malformed samples"):
                _run_fixture(fixture)

    def test_revalidation_hash_and_parser_bindings_fail_closed(self) -> None:
        mutations = (
            (
                "raw SHA",
                lambda value: value["source"]["raw_log"].__setitem__(
                    "sha256", "0" * 64
                ),
                "raw-log SHA256 mismatch",
            ),
            (
                "parser version",
                lambda value: value["parser"].__setitem__("version", 1),
                "parser identity/version mismatch",
            ),
            (
                "validator SHA",
                lambda value: value["validator"].__setitem__(
                    "sha256", "0" * 64
                ),
                "validator SHA256 mismatch",
            ),
        )
        test_root = REPOSITORY_ROOT / "reproduction" / "tests"
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=test_root
            ) as temporary:
                fixture, sidecar_path, _ = _revalidation_fixture(Path(temporary))
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                mutate(sidecar)
                _write_json(sidecar_path, sidecar)
                with self.assertRaisesRegex(ValueError, message):
                    _run_fixture(fixture, revalidation=sidecar_path)

    def test_revalidation_does_not_hide_real_malformed_shape(self) -> None:
        test_root = REPOSITORY_ROOT / "reproduction" / "tests"
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            fixture, sidecar_path, raw_path = _revalidation_fixture(Path(temporary))
            lines = raw_path.read_text(encoding="utf-8").splitlines()
            changed = 0
            for index, line in enumerate(lines):
                if "committed_tokens=671361" in line:
                    lines[index] = line.replace(
                        "reason=forecast_hold", "reason=unsafe_hold", 1
                    )
                    changed += 1
            self.assertEqual(changed, 1)
            raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["source"]["raw_log"]["sha256"] = _sha256(raw_path)
            sidecar["source"]["raw_log"]["size_bytes"] = raw_path.stat().st_size
            _write_json(sidecar_path, sidecar)
            with self.assertRaisesRegex(ValueError, "real malformed shape"):
                _run_fixture(fixture, revalidation=sidecar_path)

    def test_revalidation_rejects_other_original_failure_or_audit_tamper(self) -> None:
        test_root = REPOSITORY_ROOT / "reproduction" / "tests"
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            fixture, sidecar_path, _ = _revalidation_fixture(Path(temporary))
            _, _, c_run, _, _, _, _ = fixture
            for filename in ("summary.json", "vllm_log_summary.json"):
                path = c_run / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["physical_kv_admission"]["screening_gates"][
                    "stable_physical_capacity"
                ] = False
                _write_json(path, payload)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["original_post_run_validation"]["screening_gates"][
                "stable_physical_capacity"
            ] = False
            sidecar["source"]["summary"]["sha256"] = _sha256(
                c_run / "summary.json"
            )
            sidecar["source"]["vllm_log_summary"]["sha256"] = _sha256(
                c_run / "vllm_log_summary.json"
            )
            _write_json(sidecar_path, sidecar)
            with self.assertRaisesRegex(ValueError, "unapproved reason"):
                _run_fixture(fixture, revalidation=sidecar_path)

        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            fixture, sidecar_path, _ = _revalidation_fixture(Path(temporary))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["recomputed"]["independent_sample_audit"]["invariants"][
                "sample_count"
            ] += 1
            _write_json(sidecar_path, sidecar)
            with self.assertRaisesRegex(ValueError, "invariant sample_count mismatch"):
                _run_fixture(fixture, revalidation=sidecar_path)


if __name__ == "__main__":
    unittest.main()
