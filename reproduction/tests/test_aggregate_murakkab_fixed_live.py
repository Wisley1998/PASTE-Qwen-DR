from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
TEST_ROOT = REPOSITORY_ROOT / "reproduction" / "tests"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from aggregate_murakkab_fixed_live import (  # noqa: E402
    FIXED_SETUP,
    PROTOCOL_PATH,
    REQUIRED_PREFLIGHT_BINDING_KEYS,
    _validate_common_runs,
    aggregate_murakkab_fixed_results,
    render_markdown,
)
from test_compare_live_joint_pair import (  # noqa: E402
    _convert_to_frozen,
    _make_run,
    _write_json,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixed_m_result(
    root: Path,
    *,
    repetition: int,
    e2e_values: tuple[float, float],
    run_tag_override: str | None = None,
) -> Path:
    run_tag = run_tag_override or f"m-repetition-{repetition}"
    raw_path = _make_run(
        root,
        name=f"{run_tag}/runner_raw",
        speculation_mode="off",
        e2e_values=e2e_values,
        result_suffix=f"-r{repetition}",
    )
    _convert_to_frozen(raw_path, omit_expected_from_first_search=False)
    run_root = raw_path.parent.parent
    evidence_dir = run_root / "evidence"
    evidence_dir.mkdir()
    evidence_timeline = evidence_dir / "queue_timeline.jsonl"
    shutil.copy2(raw_path.parent / "queue_timeline.jsonl", evidence_timeline)

    payload = deepcopy(_load(raw_path))
    config = payload["config"]
    registry_sha = "d" * 64
    workflow_sha = "e" * 64
    selected_candidate = "tongyi-30b-tp4-a100x4-singleton"
    server_instance = f"fresh-server-{run_tag}"
    background_pid = 42_000 + repetition
    background_starttime = 900_000 + repetition
    background_boot_id = f"00000000-0000-0000-0000-{repetition:012d}"
    gpu_uuids = [f"GPU-test-{index}" for index in FIXED_SETUP.gpu_indices]

    def hardware_snapshot(*, after: bool) -> dict:
        per_gpu_rows = [
            {
                "gpu_uuid": gpu_uuid,
                "pid": background_pid,
                "process_name": "python",
                "used_memory_mib": float(1_024 + offset + (8 if after else 0)),
                "gpu_index": gpu_index,
            }
            for offset, (gpu_index, gpu_uuid) in enumerate(
                zip(FIXED_SETUP.gpu_indices, gpu_uuids, strict=True)
            )
        ]
        registered = {
            "valid": True,
            "policy": FIXED_SETUP.background_policy,
            "pid": background_pid,
            "executable": FIXED_SETUP.background_executable,
            "cwd": FIXED_SETUP.background_cwd,
            "argv": list(FIXED_SETUP.background_argv),
            "resolved_script": FIXED_SETUP.background_resolved_script,
            "resolved_script_sha256": FIXED_SETUP.background_script_sha256,
            "proc_starttime_ticks": background_starttime,
            "boot_id": background_boot_id,
            "user_confirmed_prior_paste_same_condition": True,
            "selected_gpu_indices": list(FIXED_SETUP.gpu_indices),
            "selected_gpu_uuids": gpu_uuids,
            "selected_application_record_count": FIXED_SETUP.gpu_count,
            "additional_selected_gpu_compute_apps_observed": False,
            "per_gpu_rows": per_gpu_rows,
        }
        return {
            "query_wall_s": 1_700_000_000.0 + repetition + (0.5 if after else 0.0),
            "selected_gpu_indices": list(FIXED_SETUP.gpu_indices),
            "selected_gpus": [
                {
                    "index": index,
                    "uuid": gpu_uuid,
                    "name": FIXED_SETUP.gpu_type,
                }
                for index, gpu_uuid in zip(
                    FIXED_SETUP.gpu_indices, gpu_uuids, strict=True
                )
            ],
            "all_compute_applications": [
                {key: value for key, value in row.items() if key != "gpu_index"}
                for row in per_gpu_rows
            ],
            "selected_gpu_background_process_count": FIXED_SETUP.gpu_count,
            "selected_gpu_compute_applications": per_gpu_rows,
            "registered_background": registered,
        }

    hardware_before = hardware_snapshot(after=False)
    hardware_after = hardware_snapshot(after=True)
    background_continuity = {
        "valid": True,
        "policy": FIXED_SETUP.background_policy,
        "same_process_identity_before_after": True,
        "pid": background_pid,
        "proc_starttime_ticks": background_starttime,
        "boot_id": background_boot_id,
        "resolved_script_sha256": FIXED_SETUP.background_script_sha256,
        "selected_gpu_indices": list(FIXED_SETUP.gpu_indices),
        "selected_gpu_uuids": gpu_uuids,
        "user_confirmed_prior_paste_same_condition": True,
        "load_intensity_equivalence_claimed": False,
    }
    plan = {
        "schema": "paste_repro.murakkab_fixed_runtime",
        "version": 1,
        "planner": "singleton_constrained_selection",
        "candidate_count": 1,
        "selected_candidate_id": selected_candidate,
        "workflow_sha256": workflow_sha,
        "registry_sha256": registry_sha,
        "optimizer_outside_timed_path": True,
        "typed_dag_validated": True,
        "dependency_ready_dispatch": True,
        "evidence_tier": "fixed-v9-setup-engineering",
        "confirmatory_eligible": False,
        "performance_comparable": True,
        "source_limit": None,
        "run_tag": run_tag,
        "repetition": repetition,
        "server_instance_id": server_instance,
        "registered_background_policy": FIXED_SETUP.background_policy,
        "registered_background": hardware_before["registered_background"],
    }
    plan_path = run_root / "run_plan.json"
    _write_json(plan_path, plan)
    _write_json(
        run_root / "preflight.json",
        {
            "valid": True,
            "protocol_sha256": _sha(PROTOCOL_PATH),
            "workload_sha256": FIXED_SETUP.workload_file_sha256,
            "bindings": {
                relative: _sha(REPOSITORY_ROOT / relative)
                for relative in sorted(REQUIRED_PREFLIGHT_BINDING_KEYS)
            },
        },
    )
    _write_json(run_root / "hardware_before.json", hardware_before)
    _write_json(run_root / "hardware_after.json", hardware_after)

    config.update(
        {
            "cell_label": f"murakkab-fixed-engineering-r{repetition}",
            "model": FIXED_SETUP.model,
            "independent_source_count": 2,
            "replicas": 1,
            "task_count": 2,
            "max_active_tasks": 2,
            "context_padding_tokens": FIXED_SETUP.context_padding_tokens,
            "fixed_final_completion_tokens": FIXED_SETUP.fixed_final_completion_tokens,
            "fixed_final_completion_enabled": True,
            "tool_workers": FIXED_SETUP.tool_workers,
            "search_tool_capacity": FIXED_SETUP.search_capacity,
            "visit_tool_capacity": FIXED_SETUP.visit_capacity,
            "speculative_tool_workers": FIXED_SETUP.maximum_speculative_workers,
            "min_speculative_tool_workers": FIXED_SETUP.minimum_speculative_workers,
            "max_speculative_pending": FIXED_SETUP.maximum_speculative_pending,
            "visit_min_start_interval_s": FIXED_SETUP.visit_minimum_start_interval_s,
            "speculative_ttl_s": FIXED_SETUP.speculative_ttl_s,
            "workload_split_id": FIXED_SETUP.workload_split_id,
            "workload_file_sha256": FIXED_SETUP.workload_file_sha256,
            "selected_workload_sha256": FIXED_SETUP.selected_workload_sha256,
            "visit_top_k": 1,
            "search_min_start_interval_s": 0.0,
            "search_max_results": 5,
            "visit_max_chars": 3000,
            "max_tokens_tool": 128,
            "max_tokens_answer": 256,
            "visit_canary_stride": 6,
            "queue_sample_interval_s": 0.2,
            "tool_signal_policy": "execution_aware",
            "tool_signal_policy_version": "exact-session-invocation-running-completed-v1",
            "tool_http_attempt_start_gate_enabled": True,
            "tool_http_attempt_start_gate_policy_version": "shared-per-tool-monotonic-v1",
            "tool_http_attempt_min_start_intervals_s": {"visit": 2.5},
            "murakkab_fixed": {
                "enabled": True,
                "cell_id": "M",
                "evidence_class": "fixed-v9-setup-engineering",
                "implementation_kind": "constrained_murakkab_style_emulation",
                "official_code_used": False,
                "official_runtime_reproduced": False,
                "runtime_semantics": "A-equivalent",
                "optimizer_candidate_count": 1,
                "selected_candidate_id": selected_candidate,
                "workflow_id": "tongyi_deepresearch_fixed_linear_v1",
                "typed_dag_validated": True,
                "dependency_ready_dispatch": True,
                "optimizer_outside_timed_path": True,
                "scheduler": "native_fcfs",
                "tool_execution": "demand_only",
                "plan_sha256": _sha(plan_path),
                "registry_sha256": registry_sha,
                "workflow_sha256": workflow_sha,
                "raw_runner_result_sha256": _sha(raw_path),
                "execution_boundary": "wrapper typed DAG delegates node bodies to frozen live runner",
                "gpu_count": FIXED_SETUP.gpu_count,
                "gpu_type": FIXED_SETUP.gpu_type,
                "hardware_evidence": {
                    "selected_gpu_indices": list(FIXED_SETUP.gpu_indices),
                    "selected_gpu_names": [
                        FIXED_SETUP.gpu_type
                    ] * FIXED_SETUP.gpu_count,
                    "selected_gpu_uuids": gpu_uuids,
                    "registered_background_policy": FIXED_SETUP.background_policy,
                    "registered_background_before": hardware_before[
                        "registered_background"
                    ],
                    "registered_background_after": hardware_after[
                        "registered_background"
                    ],
                    "registered_background_continuity": background_continuity,
                    "before_path": str(run_root / "hardware_before.json"),
                    "before_sha256": _sha(run_root / "hardware_before.json"),
                    "after_path": str(run_root / "hardware_after.json"),
                    "after_sha256": _sha(run_root / "hardware_after.json"),
                },
                "engineering_run": {
                    "evidence_tier": "fixed-v9-setup-engineering",
                    "confirmatory_eligible": False,
                    "run_id": run_tag,
                    "run_tag": run_tag,
                    "repetition": repetition,
                    "server_instance_id": server_instance,
                    "fresh_server": True,
                    "result_cache_empty": True,
                    "broker_drained": True,
                    "assertion_owner": "run_murakkab_fixed_live.py",
                    "performance_comparable": True,
                    "performance_comparability_scope": (
                        "same fixed model/hardware/workload runtime setup only; this field "
                        "does not assert a fresh causal comparison with historical PASTE"
                    ),
                    "source_limit": None,
                    "registered_background_policy": FIXED_SETUP.background_policy,
                    "registered_background_same_identity_before_after": True,
                    "user_confirmed_prior_paste_same_condition": True,
                    "registered_background_load_intensity_equivalence_claimed": False,
                },
            },
        }
    )
    config["scheduler_environment"].update(
        {
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
            "MODEL_ID": FIXED_SETUP.model,
            "MODEL_REVISION": FIXED_SETUP.model_revision,
            "VLLM_DTYPE": "bfloat16",
            "VLLM_TP_SIZE": "4",
            "VLLM_MAX_MODEL_LEN": "16384",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.86",
            "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
            "VLLM_MAX_NUM_SEQS": "96",
            "VLLM_ENABLE_PREFIX_CACHING": "1",
            "VLLM_CUDA_GRAPH_SIZES": "32",
            "VLLM_USE_V1": "1",
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
            "VLLM_SCHED_POLICY": "fcfs",
            "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY": None,
        }
    )

    for task in payload["tasks"]:
        task["context_padding_target_tokens"] = FIXED_SETUP.context_padding_tokens
        task["context_padding_actual_tokens"] = FIXED_SETUP.context_padding_tokens + 12
        task["final_answer_contract"] = {
            "contract_succeeded": True,
            "fixed_completion_tokens": FIXED_SETUP.fixed_final_completion_tokens,
            "total_completion_tokens": FIXED_SETUP.fixed_final_completion_tokens,
        }
        task["completion_tokens"] = 196
    for event in payload["llm_events"]:
        if event["call_index"] == 2:
            event["min_tokens"] = FIXED_SETUP.fixed_final_completion_tokens
            event["max_tokens"] = FIXED_SETUP.fixed_final_completion_tokens
            event["usage"]["completion_tokens"] = FIXED_SETUP.fixed_final_completion_tokens
            event["usage"]["total_tokens"] = (
                event["usage"]["prompt_tokens"] + FIXED_SETUP.fixed_final_completion_tokens
            )
    payload["vllm_metric_deltas"]["vllm:generation_tokens_total"] = 392.0
    for record in payload["tool_attempt_records"]:
        record["http_attempt_log"] = [
            {
                "attempt": 1,
                "started_monotonic_s": record["started_at"],
                "status": 200,
            }
        ]

    experiment_start = 1999.9
    last_completion = max(float(task["end_wall_s"]) for task in payload["tasks"])
    payload["summary"].update(
        {
            "started_wall_s": experiment_start,
            "ended_wall_s": last_completion + 0.05,
            "makespan_s": last_completion + 0.05 - experiment_start,
            # Deliberately wrong: the aggregator must ignore embedded latency
            # summaries and recompute them from raw task rows.
            "task_e2e": {"available": True, "mean_s": 999999.0},
        }
    )
    payload["raw_evidence"]["queue_timeline"] = {
        "path": str(evidence_timeline.resolve()),
        "sha256": _sha(evidence_timeline),
        "sample_count": 2,
    }
    payload["murakkab_provenance"] = {
        "plan_path": str(plan_path),
        "plan_sha256": _sha(plan_path),
        "hardware_before_path": str(run_root / "hardware_before.json"),
        "hardware_before_sha256": _sha(run_root / "hardware_before.json"),
        "hardware_after_path": str(run_root / "hardware_after.json"),
        "hardware_after_sha256": _sha(run_root / "hardware_after.json"),
        "unmodified_runner_result": {
            "path": str(raw_path),
            "sha256": _sha(raw_path),
        },
    }
    evidence_path = evidence_dir / "result.json"
    _write_json(evidence_path, payload)

    artifact_paths = (
        run_root / "preflight.json",
        plan_path,
        raw_path,
        run_root / "hardware_before.json",
        run_root / "hardware_after.json",
        evidence_path,
        evidence_timeline,
    )
    _write_json(
        run_root / "completed_run.json",
        {
            "schema": "paste_repro.murakkab_fixed_live_completion",
            "version": 1,
            "completed": True,
            "evidence_tier": "fixed-v9-setup-engineering",
            "confirmatory_eligible": False,
            "run_tag": run_tag,
            "repetition": repetition,
            "registered_background": background_continuity,
            "artifacts": {
                artifact.resolve().relative_to(root.resolve()).as_posix(): _sha(
                    artifact
                )
                for artifact in artifact_paths
            },
        },
    )
    return evidence_path


def _utc(wall_s: float) -> str:
    return (
        datetime.fromtimestamp(wall_s, timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _make_host_coload_observation(root: Path, result_path: Path) -> Path:
    run_root = result_path.parent.parent
    hardware_after_path = run_root / "hardware_after.json"
    hardware_after = _load(hardware_after_path)
    worker_pids = [71_001, 71_002, 71_003, 71_004]
    external_rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-external-{index}",
            "pid": pid,
            "used_memory_mib": float(16_000 + index),
        }
        for index, pid in zip(range(4), worker_pids, strict=True)
    ]
    hardware_after["all_compute_applications"].extend(
        {
            "gpu_uuid": row["gpu_uuid"],
            "pid": row["pid"],
            "process_name": f"vllm-worker-{row['gpu_index']}",
            "used_memory_mib": row["used_memory_mib"],
        }
        for row in external_rows
    )
    _write_json(hardware_after_path, hardware_after)

    result = _load(result_path)
    hardware_evidence = result["config"]["murakkab_fixed"]["hardware_evidence"]
    hardware_evidence["after_sha256"] = _sha(hardware_after_path)
    result["murakkab_provenance"]["hardware_after_sha256"] = _sha(
        hardware_after_path
    )
    _write_json(result_path, result)

    completion_path = run_root / "completed_run.json"
    completion = _load(completion_path)
    completion["artifacts"][
        hardware_after_path.resolve().relative_to(root.resolve()).as_posix()
    ] = _sha(
        hardware_after_path
    )
    completion["artifacts"][
        result_path.resolve().relative_to(root.resolve()).as_posix()
    ] = _sha(result_path)
    _write_json(completion_path, completion)

    result = _load(result_path)
    summary = result["summary"]
    timed_start = float(summary["started_wall_s"])
    timed_end = float(summary["ended_wall_s"])
    api_start = round(timed_start + 0.70 * (timed_end - timed_start), 2)
    engine_start = round(api_start + 0.10, 2)
    worker_start = round(engine_start + 0.20, 2)
    boot_wall = 1_000.0
    clock_ticks = 100
    api_ticks = round((api_start - boot_wall) * clock_ticks)
    engine_ticks = round((engine_start - boot_wall) * clock_ticks)
    worker_ticks = round((worker_start - boot_wall) * clock_ticks)
    captured_at = float(hardware_after["query_wall_s"]) + 10.0
    hardware_before_path = run_root / "hardware_before.json"
    raw_path = run_root / "runner_raw/result.json"
    registered_pid = _load(hardware_before_path)["registered_background"]["pid"]
    observation = {
        "schema": "paste_repro.host_coload_observation",
        "version": 1,
        "captured_at_utc": _utc(captured_at),
        "captured_after_run": True,
        "observation_scope": (
            "Read-only process, nvidia-smi, state-file, and server-log evidence "
            "collected after the external co-load was noticed; this is not a "
            "continuous host-load trace."
        ),
        "affected_run": {
            "run_tag": result["config"]["murakkab_fixed"]["engineering_run"][
                "run_tag"
            ],
            "repetition": result["config"]["murakkab_fixed"]["engineering_run"][
                "repetition"
            ],
            "timed_start_wall_s": timed_start,
            "timed_start_utc": _utc(timed_start),
            "timed_end_wall_s": timed_end,
            "timed_end_utc": _utc(timed_end),
            "runner_raw_result_path": str(raw_path),
            "runner_raw_result_sha256": _sha(raw_path),
            "evidence_result_path": str(result_path),
            "evidence_result_sha256": _sha(result_path),
            "completion_manifest_path": str(completion_path),
            "completion_manifest_sha256": _sha(completion_path),
            "hardware_before_path": str(hardware_before_path),
            "hardware_before_sha256": _sha(hardware_before_path),
            "hardware_before_query_wall_s": _load(hardware_before_path)[
                "query_wall_s"
            ],
            "hardware_before_unregistered_compute_apps": [],
            "hardware_after_path": str(hardware_after_path),
            "hardware_after_sha256": _sha(hardware_after_path),
            "hardware_after_query_wall_s": hardware_after["query_wall_s"],
        },
        "external_vllm": {
            "managed_by_this_murakkab_run": False,
            "api_pid": 70_000,
            "api_process_starttime_ticks": api_ticks,
            "api_process_start_wall_s": api_start,
            "api_process_start_utc": _utc(api_start),
            "api_process_overlap_with_timed_run_s": timed_end - api_start,
            "engine_pid": 70_100,
            "engine_process_starttime_ticks": engine_ticks,
            "engine_process_start_wall_s": engine_start,
            "worker_pids": worker_pids,
            "worker_process_starttime_ticks": [worker_ticks] * 4,
            "first_worker_start_wall_s": worker_start,
            "first_worker_start_utc": _utc(worker_start),
            "worker_overlap_with_timed_run_s": timed_end - worker_start,
            "boot_id": result["config"]["murakkab_fixed"]["hardware_evidence"][
                "registered_background_before"
            ]["boot_id"],
            "clock_ticks_per_second": clock_ticks,
            "kernel_boot_time_wall_s": boot_wall,
            "executable": "/test/python3.10",
            "cwd": str(REPOSITORY_ROOT),
            "argv": [
                "/test/python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--served-model-name",
                FIXED_SETUP.model,
                "--port",
                "8200",
                "--tensor-parallel-size",
                "4",
            ],
            "listen_endpoint": "127.0.0.1:8200",
            "state_pid_path": "captured/vllm_8200.pid",
            "state_pid_sha256_at_capture": "a" * 64,
            "state_policy_path": "captured/vllm_8200.policy",
            "state_policy_sha256_at_capture": "b" * 64,
            "server_log_path": "captured/vllm_8200.log",
            "server_log_size_bytes_at_capture": 1_024,
            "server_log_sha256_at_capture": "c" * 64,
            "server_log_last_modified_at_capture_utc": _utc(captured_at - 2.0),
            "server_became_ready_utc_from_log": _utc(timed_end + 1.0),
            "server_became_ready_after_affected_run_ended": True,
        },
        "external_gpu_compute_applications_in_bound_run_after_snapshot": external_rows,
        "gpu_compute_applications_at_capture": [
            {**row, "used_memory_mib": row["used_memory_mib"] + 10_000.0}
            for row in external_rows
        ],
        "selected_gpu_observation_at_capture": {
            "indices": list(FIXED_SETUP.gpu_indices),
            "additional_compute_apps_beyond_registered_resnet": False,
            "registered_resnet_pid": registered_pid,
        },
        "interpretation": {
            "functional_integrity_invalidated": False,
            "performance_cleanliness_invalidated": True,
            "reason": "Unplanned external vLLM cold start overlapped the timed run.",
            "exclusion_based_on_performance_value": False,
            "performance_values_inspected_before_exclusion_decision": True,
            "decision_characterization": (
                "Post-run operational contamination exclusion, not a "
                "performance-threshold exclusion."
            ),
        },
    }
    path = root / "host_coload_observation.json"
    _write_json(path, observation)
    return path


class AggregateMurakkabFixedLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact_root = self.root / "artifact-repo-a"
        (self.artifact_root / "reproduction").mkdir(parents=True)
        (self.artifact_root / "scripts").mkdir()
        self.paths = [
            _make_fixed_m_result(
                self.artifact_root,
                repetition=repetition,
                e2e_values=(9.0 + repetition, 11.0 + repetition),
            )
            for repetition in (1, 2, 3)
        ]
        self.setup = replace(FIXED_SETUP, task_count=2)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self, exclusion_manifests=()) -> dict:
        return aggregate_murakkab_fixed_results(
            self.paths,
            setup=self.setup,
            exclusion_manifest_paths=exclusion_manifests,
            artifact_roots=[self.artifact_root],
        )

    def make_exclusion(self) -> tuple[Path, Path]:
        result = _make_fixed_m_result(
            self.artifact_root,
            repetition=2,
            e2e_values=(30.0, 32.0),
            run_tag_override="m-repetition-2-host-coload-contaminated",
        )
        return result, _make_host_coload_observation(self.artifact_root, result)

    def artifact_key(self, path: Path, *, root: Path | None = None) -> str:
        return path.resolve().relative_to((root or self.artifact_root).resolve()).as_posix()

    def make_artifact_root(self, name: str) -> Path:
        root = self.root / name
        (root / "reproduction").mkdir(parents=True)
        (root / "scripts").mkdir()
        return root

    def rewrite(self, path: Path, mutate) -> None:
        payload = _load(path)
        mutate(payload)
        _write_json(path, payload)
        manifest_path = path.parent.parent / "completed_run.json"
        manifest = _load(manifest_path)
        manifest["artifacts"][self.artifact_key(path)] = _sha(path)
        _write_json(manifest_path, manifest)

    def rewrite_sidecar(self, result_path: Path, name: str, mutate) -> None:
        run_root = result_path.parent.parent
        sidecar_path = run_root / name
        payload = _load(sidecar_path)
        mutate(payload)
        _write_json(sidecar_path, payload)
        manifest_path = run_root / "completed_run.json"
        manifest = _load(manifest_path)
        manifest["artifacts"][self.artifact_key(sidecar_path)] = _sha(sidecar_path)
        _write_json(manifest_path, manifest)

    def test_recomputes_three_repetition_metrics_from_raw_evidence(self) -> None:
        result = self.aggregate()
        self.assertEqual(result["run_count"], 3)
        self.assertEqual(result["treatment"]["evidence_class"], "fixed-v9-setup-engineering")
        self.assertEqual(result["aggregate"]["independent_source_count"], 2)
        self.assertEqual(result["aggregate"]["llm"]["request_count"], 18)
        self.assertEqual(result["aggregate"]["tool"]["authoritative_commit_count"], 12)
        self.assertEqual(result["aggregate"]["tool"]["physical_http_attempt_count"], 12)
        self.assertEqual(result["aggregate"]["tool"]["retried_physical_job_count"], 0)
        self.assertEqual(
            len(
                {
                    row["input"]["registered_background_pid"]
                    for row in result["per_run"]
                }
            ),
            3,
            "a fresh/restarted registered process across repetitions is allowed",
        )
        self.assertTrue(
            result["aggregate"]["integrity"][
                "registered_resnet_coload_endpoint_identity_verified_every_run"
            ]
        )
        self.assertFalse(
            result["aggregate"]["integrity"][
                "background_load_intensity_equivalence_claimed"
            ]
        )

        # Per-source means are (11, 13), so the raw-evidence mean is 12 rather
        # than the deliberately forged embedded summary value.
        source_latency = result["aggregate"]["latency"]["source_mean_e2e_across_runs_s"]
        self.assertAlmostEqual(source_latency["distribution"]["mean_s"], 12.0)
        self.assertEqual(source_latency["by_source"], {"source1": 11.0, "source2": 13.0})

        repetitions = result["aggregate"]["across_repetitions"]
        self.assertTrue(repetitions["descriptive_only_no_significance_test"])
        self.assertEqual(repetitions["tasks_per_s"]["count"], 3)
        self.assertGreater(repetitions["tasks_per_s"]["range"], 0.0)
        for per_run in result["per_run"]:
            self.assertEqual(per_run["llm"]["request_count"], 6)
            self.assertEqual(per_run["tool"]["authoritative_commit_count"], 4)
            self.assertEqual(per_run["tool"]["speculative_job_count"], 0)
            self.assertAlmostEqual(
                per_run["throughput"]["llm_requests_per_s"],
                3.0 * per_run["throughput"]["tasks_per_s"],
            )
            self.assertAlmostEqual(
                per_run["throughput"]["tool_commits_per_s"],
                2.0 * per_run["throughput"]["tasks_per_s"],
            )

        report = render_markdown(result)
        self.assertIn("constrained Murakkab-style", report)
        self.assertIn("does not estimate a PASTE speedup", report)
        self.assertIn("registered ResNet", report)
        self.assertIn("No continuous in-run background monitor", report)
        self.assertIn("not isolated-Qwen capacity", report)

    def test_rejects_additional_selected_gpu_application(self) -> None:
        def add_application(payload: dict) -> None:
            extra = deepcopy(payload["all_compute_applications"][0])
            extra["pid"] += 1
            payload["all_compute_applications"].append(extra)

        self.rewrite_sidecar(
            self.paths[0],
            "hardware_before.json",
            add_application,
        )
        with self.assertRaisesRegex(ValueError, "exactly one compute application"):
            self.aggregate()

    def test_rejects_zero_memory_registered_background_row(self) -> None:
        def remove_memory_residency(payload: dict) -> None:
            payload["registered_background"]["per_gpu_rows"][0][
                "used_memory_mib"
            ] = 0.0
            payload["selected_gpu_compute_applications"][0][
                "used_memory_mib"
            ] = 0.0
            payload["all_compute_applications"][0]["used_memory_mib"] = 0.0

        self.rewrite_sidecar(
            self.paths[0],
            "hardware_before.json",
            remove_memory_residency,
        )
        with self.assertRaisesRegex(ValueError, "used_memory_mib must be positive"):
            self.aggregate()

    def test_rejects_background_process_replacement_between_endpoints(self) -> None:
        self.rewrite_sidecar(
            self.paths[0],
            "hardware_after.json",
            lambda payload: payload["registered_background"].__setitem__(
                "proc_starttime_ticks",
                payload["registered_background"]["proc_starttime_ticks"] + 1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "process identity changed"):
            self.aggregate()

    def test_rejects_forged_completion_background_claim(self) -> None:
        manifest_path = self.paths[0].parent.parent / "completed_run.json"
        manifest = _load(manifest_path)
        manifest["registered_background"][
            "load_intensity_equivalence_claimed"
        ] = True
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            ValueError,
            "completion manifest registered_background",
        ):
            self.aggregate()

    def test_rejects_non_fcfs_scheduler(self) -> None:
        self.rewrite(
            self.paths[0],
            lambda payload: payload["config"]["scheduler_environment"].__setitem__(
                "VLLM_SCHED_POLICY", "online_joint_pacer_v2"
            ),
        )
        with self.assertRaisesRegex(ValueError, "VLLM_SCHED_POLICY"):
            self.aggregate()

    def test_rejects_any_observed_http_retry(self) -> None:
        def add_retry(payload: dict) -> None:
            record = payload["tool_attempt_records"][0]
            record["http_attempts"] = 2
            record["service_s"] += 1.0
            record["finished_at"] += 1.0
            task = next(
                task for task in payload["tasks"] if task["task_id"] == record["session_id"]
            )
            tool_index = 0 if record["tool"] == "search" else 1
            task["tools"][tool_index]["service_s"] = record["service_s"]

        self.rewrite(self.paths[0], add_retry)
        with self.assertRaisesRegex(ValueError, "requires zero retries"):
            self.aggregate()

    def test_rejects_server_identity_that_differs_from_bound_plan(self) -> None:
        reused = _load(self.paths[0])["config"]["murakkab_fixed"]["engineering_run"][
            "server_instance_id"
        ]
        self.rewrite(
            self.paths[1],
            lambda payload: payload["config"]["murakkab_fixed"]["engineering_run"].__setitem__(
                "server_instance_id", reused
            ),
        )
        with self.assertRaisesRegex(ValueError, "run plan server_instance_id"):
            self.aggregate()

    def test_requires_all_three_repetitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            aggregate_murakkab_fixed_results(self.paths[:2], setup=self.setup)

    def test_rejects_cross_repetition_binding_map_change(self) -> None:
        run_root = self.paths[1].parent.parent
        preflight_path = run_root / "preflight.json"
        preflight = _load(preflight_path)
        preflight["bindings"]["README.md"] = _sha(REPOSITORY_ROOT / "README.md")
        _write_json(preflight_path, preflight)
        manifest_path = run_root / "completed_run.json"
        manifest = _load(manifest_path)
        manifest["artifacts"][self.artifact_key(preflight_path)] = _sha(
            preflight_path
        )
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "bindings differ across M repetitions"):
            self.aggregate()

    def test_rejects_cross_repetition_selected_gpu_uuid_change(self) -> None:
        common_config = {
            "selected_workload_sha256": FIXED_SETUP.selected_workload_sha256,
            "workload_file_sha256": FIXED_SETUP.workload_file_sha256,
            "workload_split_id": FIXED_SETUP.workload_split_id,
            "call_graph_mode": "frozen",
            "model": FIXED_SETUP.model,
        }
        runs = [
            SimpleNamespace(
                config=common_config,
                tasks_by_key={("source", 0): {}},
                sha256=character * 64,
            )
            for character in ("1", "2", "3")
        ]
        fixed = [
            {
                "run_tag": f"m-{repetition}",
                "repetition": repetition,
                "server_instance_id": f"server-{repetition}",
                "plan_sha256": str(repetition) * 64,
                "workflow_id": "workflow",
                "workflow_sha256": "a" * 64,
                "registry_sha256": "b" * 64,
                "selected_candidate_id": "candidate",
                "preflight_bindings": {"input": "c" * 64},
                "preflight_bindings_sha256": "d" * 64,
                "registered_background": {
                    "selected_gpu_uuids": list(gpu_uuids),
                },
            }
            for repetition, gpu_uuids in (
                (1, ("gpu4", "gpu5", "gpu6", "gpu7")),
                (2, ("gpu4", "gpu5", "gpu6", "gpu7")),
                (3, ("different4", "gpu5", "gpu6", "gpu7")),
            )
        ]
        with self.assertRaisesRegex(ValueError, "different selected GPU UUIDs"):
            _validate_common_runs(runs, fixed)

    def test_keeps_host_coload_attempt_supplementary_only(self) -> None:
        excluded_result, observation = self.make_exclusion()
        result = self.aggregate([observation])
        self.assertEqual(result["run_count"], 3)
        self.assertEqual(
            result["attempt_accounting"]["primary_clean_result_count"],
            3,
        )
        self.assertEqual(
            result["attempt_accounting"][
                "supplementary_operationally_excluded_count"
            ],
            1,
        )
        self.assertEqual(
            result["attempt_accounting"]["total_disclosed_completed_attempt_count"],
            4,
        )
        excluded = result["supplementary"][
            "operationally_excluded_contaminated_runs"
        ][0]
        self.assertEqual(excluded["result_sha256"], _sha(excluded_result))
        self.assertEqual(excluded["classification"], "host_co_load_contaminated")
        self.assertEqual(
            excluded["disposition"],
            "excluded_from_primary_supplementary_only",
        )
        self.assertFalse(excluded["exclusion_based_on_performance_value"])
        self.assertTrue(
            excluded["performance_values_inspected_before_exclusion_decision"]
        )
        self.assertEqual(result["aggregate"]["llm"]["request_count"], 18)
        report = render_markdown(result)
        self.assertIn("Supplementary operationally excluded attempts", report)
        self.assertIn("not included in any primary aggregate", report)
        self.assertIn("Performance values had already been inspected", report)

    def test_rejects_contaminated_result_in_primary_aggregate(self) -> None:
        excluded_result, observation = self.make_exclusion()
        with self.assertRaisesRegex(
            ValueError,
            "contaminated result cannot appear in the primary",
        ):
            aggregate_murakkab_fixed_results(
                [self.paths[0], excluded_result, self.paths[2]],
                setup=self.setup,
                exclusion_manifest_paths=[observation],
                artifact_roots=[self.artifact_root],
            )

    def test_rejects_exclusion_manifest_result_sha_mismatch(self) -> None:
        _excluded_result, observation = self.make_exclusion()
        payload = _load(observation)
        payload["affected_run"]["evidence_result_sha256"] = "f" * 64
        _write_json(observation, payload)
        with self.assertRaisesRegex(ValueError, "evidence_result_sha256"):
            self.aggregate([observation])

    def test_rejects_exclusion_manifest_run_tag_mismatch(self) -> None:
        _excluded_result, observation = self.make_exclusion()
        payload = _load(observation)
        payload["affected_run"]["run_tag"] = "forged-run-tag"
        _write_json(observation, payload)
        with self.assertRaisesRegex(ValueError, "affected_run.run_tag"):
            self.aggregate([observation])

    def test_rejects_inconsistent_host_overlap_timestamp(self) -> None:
        _excluded_result, observation = self.make_exclusion()
        payload = _load(observation)
        payload["external_vllm"]["api_process_overlap_with_timed_run_s"] += 5.0
        _write_json(observation, payload)
        with self.assertRaisesRegex(ValueError, "api_process_overlap"):
            self.aggregate([observation])

    def test_rejects_performance_value_based_exclusion(self) -> None:
        _excluded_result, observation = self.make_exclusion()
        payload = _load(observation)
        payload["interpretation"]["exclusion_based_on_performance_value"] = True
        _write_json(observation, payload)
        with self.assertRaisesRegex(
            ValueError,
            "exclusion_based_on_performance_value",
        ):
            self.aggregate([observation])

    def test_accepts_results_from_two_explicit_artifact_roots(self) -> None:
        second_root = self.make_artifact_root("artifact-repo-b")
        second_paths = [
            _make_fixed_m_result(
                second_root,
                repetition=repetition,
                e2e_values=(9.0 + repetition, 11.0 + repetition),
            )
            for repetition in (2, 3)
        ]
        result = aggregate_murakkab_fixed_results(
            [self.paths[0], *second_paths],
            setup=self.setup,
            artifact_roots=[self.artifact_root, second_root],
        )
        expected_roots = [
            str(self.artifact_root.resolve()),
            str(second_root.resolve()),
        ]
        self.assertEqual(result["provenance"]["artifact_roots"], expected_roots)
        report = render_markdown(result)
        for root in expected_roots:
            self.assertIn(root, report)

    def test_relative_exclusion_paths_select_unique_strict_sidecar_root(self) -> None:
        excluded_result, observation = self.make_exclusion()
        second_root = self.make_artifact_root("artifact-repo-b")
        relative_run_root = excluded_result.parent.parent.relative_to(
            self.artifact_root
        )
        shutil.copytree(
            excluded_result.parent.parent,
            second_root / relative_run_root,
        )
        payload = _load(observation)
        for key in (
            "runner_raw_result_path",
            "evidence_result_path",
            "completion_manifest_path",
            "hardware_before_path",
            "hardware_after_path",
        ):
            payload["affected_run"][key] = (
                Path(payload["affected_run"][key])
                .resolve()
                .relative_to(self.artifact_root.resolve())
                .as_posix()
            )
        _write_json(observation, payload)
        result = aggregate_murakkab_fixed_results(
            self.paths,
            setup=self.setup,
            exclusion_manifest_paths=[observation],
            artifact_roots=[self.artifact_root, second_root],
        )
        excluded = result["supplementary"][
            "operationally_excluded_contaminated_runs"
        ][0]
        self.assertEqual(excluded["result_path"], str(excluded_result.resolve()))

    def test_rejects_run_outside_configured_artifact_roots(self) -> None:
        second_root = self.make_artifact_root("artifact-repo-b")
        second_paths = [
            _make_fixed_m_result(
                second_root,
                repetition=repetition,
                e2e_values=(9.0 + repetition, 11.0 + repetition),
            )
            for repetition in (2, 3)
        ]
        with self.assertRaisesRegex(ValueError, "outside every configured artifact root"):
            aggregate_murakkab_fixed_results(
                [self.paths[0], *second_paths],
                setup=self.setup,
                artifact_roots=[self.artifact_root],
            )

    def test_rejects_absolute_completion_manifest_artifact_key(self) -> None:
        manifest_path = self.paths[0].parent.parent / "completed_run.json"
        manifest = _load(manifest_path)
        key, digest = next(iter(manifest["artifacts"].items()))
        del manifest["artifacts"][key]
        manifest["artifacts"][str((self.artifact_root / key).resolve())] = digest
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "canonical repository-relative"):
            self.aggregate()

    def test_rejects_non_repository_artifact_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks repository marker"):
            aggregate_murakkab_fixed_results(
                self.paths,
                setup=self.setup,
                artifact_roots=[self.root],
            )


if __name__ == "__main__":
    unittest.main()
