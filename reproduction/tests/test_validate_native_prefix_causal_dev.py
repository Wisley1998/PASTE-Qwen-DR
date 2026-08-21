from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY_ROOT / "reproduction/scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_native_prefix_causal_dev as validator  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _bindings() -> dict[str, str]:
    paths = (
        REPOSITORY_ROOT / "reproduction/configs/native_prefix_causal_dev.env.example",
        REPOSITORY_ROOT / "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json",
        REPOSITORY_ROOT / "reproduction/results/live_joint/NATIVE_PREFIX_CAUSAL_DEV_PROTOCOL.md",
        REPOSITORY_ROOT / "reproduction/scripts/run_native_prefix_prompt_cell.py",
        REPOSITORY_ROOT / "reproduction/scripts/run_native_prefix_causal_dev.py",
        REPOSITORY_ROOT / "reproduction/scripts/validate_native_prefix_causal_dev.py",
        REPOSITORY_ROOT / "reproduction/scripts/start_vllm.sh",
        REPOSITORY_ROOT / "reproduction/scripts/stop_vllm.sh",
    )
    return {_relative(path): validator.sha256_file(path) for path in paths}


def _source_ids() -> list[str]:
    path = REPOSITORY_ROOT / "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row["source_id"] for row in payload["sources"]]


def _engine_environment(cell_id: str, cell_root: Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
        "PYTHONPATH": "",
        "VLLM_CUDA_GRAPH_SIZES": "32",
        "VLLM_DTYPE": "bfloat16",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.86",
        "VLLM_HOOK_DIR": str((cell_root / "native_pythonpath").resolve()),
        "VLLM_HOST": "127.0.0.1",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
        "VLLM_LOG_DIR": str(cell_root / "server"),
        "VLLM_PORT": "8100",
        "VLLM_PROBE_HOST": "127.0.0.1",
        "VLLM_READY_TIMEOUT": "3600",
        "VLLM_REQUIRE_NEW": "1",
        "VLLM_SHUTDOWN_TIMEOUT": "60",
        "VLLM_STATE_DIR": str(cell_root / "state"),
        "VLLM_TP_SIZE": "4",
        "VLLM_MAX_MODEL_LEN": "16384",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_MAX_NUM_SEQS": "96",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_ENABLE_PREFIX_CACHING": "1" if cell_id == "P1" else "0",
        "VLLM_USE_V1": "1",
        "VLLM_SCHED_POLICY": "fcfs",
    }


def _make_cell(
    run_root: Path,
    *,
    block_number: int,
    cell_id: str,
    order_index: int,
    task_e2e_s: float,
    request_s: float,
    prefill_s: float,
) -> None:
    run_tag = run_root.name
    block_id = f"{run_tag}-block-{block_number}"
    cell_root = run_root / f"block-{block_number:02d}" / cell_id
    evidence_dir = cell_root / "evidence"
    server_dir = cell_root / "server"
    evidence_dir.mkdir(parents=True)
    server_dir.mkdir()
    (cell_root / "native_pythonpath").mkdir()
    server_id = f"server-{block_number}-{cell_id}"
    environment = _engine_environment(cell_id, cell_root)
    effective = {
        "schema": validator.EFFECTIVE_CONFIG_SCHEMA,
        "version": 2,
        "block_id": block_id,
        "block_number": block_number,
        "cell_id": cell_id,
        "order_index": order_index,
        "server_instance_id": server_id,
        "fresh_server_required": True,
        "native_prefix_cache_enabled": cell_id == "P1",
        "scheduler_policy": "fcfs",
        "native_pythonpath_isolated": True,
        "native_pythonpath": str((cell_root / "native_pythonpath").resolve()),
        "explicit_prefix_locality_enabled": False,
        "external_network_allowed": False,
        "external_tools_allowed": False,
        "environment": environment,
    }
    _write_json(cell_root / "effective_config.json", effective)
    process_identity = {
        "pid": 10000 + block_number * 10 + (1 if cell_id == "P1" else 0),
        "proc_start_ticks": 50000 + block_number * 10 + (1 if cell_id == "P1" else 0),
        "executable": "/home/aiscuser/.conda/envs/paste/bin/python3.10",
        "cmdline_sha256": _sha_text(f"cmdline/{block_number}/{cell_id}"),
    }
    _write_json(
        cell_root / "server_identity.json",
        {
            "schema": validator.SERVER_IDENTITY_SCHEMA,
            "version": 2,
            "server_instance_id": server_id,
            "captured_wall_s": 100.0,
            **process_identity,
            "process_identity_sha256": validator.sha256_json(process_identity),
        },
    )

    tasks = []
    events = []
    prompt_by_call = [10240, 10400, 11400]
    for source_id in _source_ids():
        for replica in range(3):
            task_id = f"{source_id}__r{replica:02d}"
            tasks.append(
                {
                    "task_id": task_id,
                    "source_id": source_id,
                    "replica": replica,
                    "ok": True,
                    "started_wall_s": 100.0,
                    "ended_wall_s": 100.0 + task_e2e_s,
                    "e2e_s": task_e2e_s,
                    "completed_call_indices": [0, 1, 2],
                    "context_padding_actual_tokens": 10016,
                }
            )
            for call_index, prompt_tokens in enumerate(prompt_by_call):
                completion = validator.SENTINEL
                response = completion
                completion_tokens = 1
                events.append(
                    {
                        "task_id": task_id,
                        "source_id": source_id,
                        "replica": replica,
                        "call_index": call_index,
                        "request_id": f"prefixcausal-{task_id}-c{call_index}",
                        "attempts": 1,
                        "http_status": 200,
                        "ok": True,
                        "duration_s": request_s,
                        "messages_sha256": _sha_text(f"messages/{task_id}/{call_index}"),
                        "prompt_token_ids_sha256": _sha_text(
                            f"tokens/{task_id}/{call_index}"
                        ),
                        "prompt_tokens_estimate": prompt_tokens,
                        "max_tokens": 1,
                        "request_payload_sha256": _sha_text(
                            f"payload/{task_id}/{call_index}"
                        ),
                        "guided_choice_sha256": validator.sha256_json(
                            [validator.SENTINEL]
                        ),
                        "expected_completion": completion,
                        "expected_completion_sha256": validator.sha256_json(completion),
                        "expected_completion_tokens_estimate": 1,
                        "response": response,
                        "response_sha256": _sha_text(response),
                        "semantic_response_sha256": validator.sha256_json(completion),
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                )
    prompt_total = sum(event["usage"]["prompt_tokens"] for event in events)
    completion_total = sum(event["usage"]["completion_tokens"] for event in events)
    task_values = [task["e2e_s"] for task in tasks]
    request_values = [event["duration_s"] for event in events]
    summary = {
        "task_count": 48,
        "successful_task_count": 48,
        "failed_task_count": 0,
        "all_tasks_succeeded": True,
        "request_count": 144,
        "successful_request_count": 144,
        "failed_request_count": 0,
        "exactly_one_attempt_each": True,
        "makespan_s": task_e2e_s,
        "task_completion_makespan_s": task_e2e_s,
        "task_e2e": {
            "mean_s": statistics.fmean(task_values),
            "p50_s": task_e2e_s,
            "p95_s": task_e2e_s,
            "max_s": task_e2e_s,
        },
        "llm": {
            "mean_request_s": statistics.fmean(request_values),
            "p95_request_s": request_s,
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
        },
    }
    metrics = {
        "vllm:request_queue_time_seconds_sum": 50.0,
        "vllm:request_inference_time_seconds_sum": 300.0,
        "vllm:request_prefill_time_seconds_sum": prefill_s,
        "vllm:request_decode_time_seconds_sum": 200.0,
        "vllm:prompt_tokens_total": float(prompt_total),
        "vllm:generation_tokens_total": float(completion_total),
        "vllm:num_preemptions_total": 0.0,
        "vllm:prefix_cache_queries_total": (
            float(prompt_total) if cell_id == "P1" else 0.0
        ),
        "vllm:prefix_cache_hits_total": (
            float(math.floor(prompt_total * 0.65)) if cell_id == "P1" else 0.0
        ),
    }
    metric_labels = (
        'engine="0",model_name="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"'
    )
    metrics_before_path = evidence_dir / "metrics_before.prom"
    metrics_after_path = evidence_dir / "metrics_after.prom"
    metrics_before_path.write_text(
        "".join(f"{name}{{{metric_labels}}} 0\n" for name in metrics),
        encoding="utf-8",
    )
    metrics_after_path.write_text(
        "".join(
            f"{name}{{{metric_labels}}} {value}\n"
            for name, value in metrics.items()
        ),
        encoding="utf-8",
    )
    queue_rows = [
        {
            "ok": True,
            "wall_s": 100.0,
            "monotonic_s": 10.0,
            "llm_running": 40.0,
            "llm_waiting": 8.0,
            "gpu_cache_usage": 0.7,
        }
    ]
    queue_path = evidence_dir / "queue_timeline.jsonl"
    queue_path.write_text(
        "".join(validator.canonical_json(row) + "\n" for row in queue_rows),
        encoding="utf-8",
    )
    result = {
        "schema": validator.CELL_SCHEMA,
        "version": 2,
        "config": {
            "fixture_version": validator.FIXTURE_VERSION,
            "fixture_manifest_sha256": _sha_text("shared-fixture-manifest"),
            "output_constraint": validator.OUTPUT_CONSTRAINT,
            "sentinel_contract": validator.exact_sentinel_contract(),
            "cell_id": cell_id,
            "block_id": block_id,
            "order_index": order_index,
            "server_instance_id": server_id,
            "fresh_server": True,
            "prefix_cache_enabled": cell_id == "P1",
            "scheduler_policy": "fcfs",
            "scheduler_environment": {"VLLM_SCHED_POLICY": "fcfs"},
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
            "model": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
            "workload_sha256": (
                "e9f63f75bb80c840fbc59f2aa9a581527669c10fc761a4649f50a1bc03eaf1ea"
            ),
            "workload_split_id": "live-joint-wikipedia-frozen-tune-v1",
            "workload_formal_eligible": False,
            "engine_environment": {
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
            },
        },
        "summary": summary,
        "tasks": tasks,
        "llm_events": events,
        "vllm_metric_deltas": metrics,
        "vllm_metric_presence": {
            key: {"before": True, "after": True} for key in metrics
        },
        "raw_evidence": {
            "queue_timeline": {
                "path": str(queue_path),
                "sha256": validator.sha256_file(queue_path),
                "sample_count": 1,
            },
            "metrics_before": {
                "path": str(metrics_before_path),
                "sha256": validator.sha256_file(metrics_before_path),
            },
            "metrics_after": {
                "path": str(metrics_after_path),
                "sha256": validator.sha256_file(metrics_after_path),
            },
        },
    }
    _write_json(evidence_dir / "result.json", result)
    enabled = "True" if cell_id == "P1" else "False"
    (server_dir / "vllm_8100.log").write_text(
        "non-default args: {'max_model_len': 16384, "
        f"'enable_prefix_caching': {enabled}, "
        "'max_num_batched_tokens': 2048, 'max_num_seqs': 96}\n"
        "Initializing a V1 LLM engine (v0.10.1) with max_seq_len=16384 "
        f"enable_prefix_caching={enabled}\n",
        encoding="utf-8",
    )
    for name in (
        "server_lifecycle.stdout.log",
        "server_lifecycle.stderr.log",
        "runner.stdout.log",
        "runner.stderr.log",
    ):
        (cell_root / name).write_text("synthetic evidence\n", encoding="utf-8")
    _rehash_manifest(cell_root, block_id, cell_id, order_index, server_id)


def _rehash_manifest(
    cell_root: Path,
    block_id: str | None = None,
    cell_id: str | None = None,
    order_index: int | None = None,
    server_id: str | None = None,
) -> None:
    old_path = cell_root / "cell_manifest.json"
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    names = (
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
    )
    manifest = {
        "schema": validator.CELL_MANIFEST_SCHEMA,
        "version": 2,
        "block_id": block_id or old["block_id"],
        "cell_id": cell_id or old["cell_id"],
        "order_index": order_index if order_index is not None else old["order_index"],
        "server_instance_id": server_id or old["server_instance_id"],
        "evidence": {
            name: validator.sha256_file(cell_root / name) for name in names
        },
    }
    _write_json(old_path, manifest)


def _make_run(
    tmp_path: Path,
    *,
    p1_task_e2e_s: float = 5.5,
    p1_request_s: float = 1.8,
    p1_prefill_s: float = 70.0,
) -> Path:
    run_root = tmp_path / "synthetic"
    run_root.mkdir()
    workload = REPOSITORY_ROOT / "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
    plan = {
        "schema": validator.PLAN_SCHEMA,
        "version": 2,
        "run_tag": "synthetic",
        "prospective_version": validator.PROTOCOL_VERSION,
        "prior_r1_disposition": "rejected_diagnostic_not_validatable_as_v2",
        "only_treatment_variable": "VLLM_ENABLE_PREFIX_CACHING",
        "native_pythonpath_isolated": True,
        "explicit_prefix_locality_enabled": False,
        "pinned_vllm_version": "0.10.1",
        "fixture_preflight": {
            "fixture_manifest_sha256": _sha_text("shared-fixture-manifest"),
            "task_count": 48,
            "call_count": 144,
            "max_prompt_plus_generation_cap": 11401,
            "sentinel_contract": validator.exact_sentinel_contract(),
        },
        "generation_contract": validator.exact_sentinel_contract(),
        "orders": [["P0", "P1"], ["P1", "P0"]],
        "matrix": dict(validator.EXACT_MATRIX),
        "thresholds": dict(validator.EXACT_THRESHOLDS),
        "engine": {
            **validator.EXACT_ENGINE,
            "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
            "VLLM_ENABLE_PREFIX_CACHING": "per_cell_P0_0_P1_1",
        },
        "workload": {
            "path": _relative(workload),
            "sha256": validator.sha256_file(workload),
        },
        "bindings": _bindings(),
    }
    bindings = plan["bindings"]
    protocol_path = "reproduction/results/live_joint/NATIVE_PREFIX_CAUSAL_DEV_PROTOCOL.md"
    validator_path = "reproduction/scripts/validate_native_prefix_causal_dev.py"
    cell_path = "reproduction/scripts/run_native_prefix_prompt_cell.py"
    plan["contract_bindings"] = {
        "protocol": {
            "version": validator.PROTOCOL_VERSION,
            "path": protocol_path,
            "sha256": bindings[protocol_path],
        },
        "validator": {
            "schema": validator.SCHEMA,
            "path": validator_path,
            "sha256": bindings[validator_path],
        },
        "cell_runner": {
            "schema": validator.CELL_SCHEMA,
            "path": cell_path,
            "sha256": bindings[cell_path],
        },
        "prior_r1_disposition": "rejected_diagnostic_not_validatable_as_v2",
    }
    _write_json(run_root / "run_plan.json", plan)
    for block_number, order in enumerate(validator.EXPECTED_ORDERS, 1):
        for order_index, cell_id in enumerate(order):
            _make_cell(
                run_root,
                block_number=block_number,
                cell_id=cell_id,
                order_index=order_index,
                task_e2e_s=6.1 if cell_id == "P0" else p1_task_e2e_s,
                request_s=2.0 if cell_id == "P0" else p1_request_s,
                prefill_s=100.0 if cell_id == "P0" else p1_prefill_s,
            )
    return run_root


def test_valid_two_block_native_prefix_matrix_passes_all_gates(tmp_path: Path) -> None:
    result = validator.validate_run(_make_run(tmp_path))

    assert result["valid"] is True
    assert result["development_only"] is True
    assert result["unique_fresh_server_instances"] == 4
    assert result["promotion_passed"] is True
    assert result["selected_policy"] == "native"
    assert all(result["promotion_gates"].values())
    assert result["aggregate"]["P0"]["native_queries"] == 0
    assert result["aggregate"]["P1"]["native_hit_ratio"] >= 0.60


def test_prompt_identity_difference_is_rejected(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    cell_root = run_root / "block-01/P1"
    result_path = cell_root / "evidence/result.json"
    result = json.loads(result_path.read_text())
    result["llm_events"][0]["messages_sha256"] = "a" * 64
    _write_json(result_path, result)
    _rehash_manifest(cell_root)

    with pytest.raises(validator.ValidationError, match="identity differs"):
        validator.validate_run(run_root)


def test_v1_plan_cannot_be_rescued_by_v2_validator(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    plan_path = run_root / "run_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["schema"] = "paste_repro.native_prefix_causal_plan_v1"
    plan["version"] = 1
    _write_json(plan_path, plan)

    with pytest.raises(validator.ValidationError, match="run plan schema mismatch"):
        validator.validate_run(run_root)


def test_variable_completion_token_count_is_rejected(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    cell_root = run_root / "block-01/P1"
    result_path = cell_root / "evidence/result.json"
    result = json.loads(result_path.read_text())
    event = result["llm_events"][0]
    event["usage"]["completion_tokens"] = 2
    event["usage"]["total_tokens"] += 1
    result["summary"]["llm"]["completion_tokens"] += 1
    result["vllm_metric_deltas"]["vllm:generation_tokens_total"] += 1
    raw_path = cell_root / "evidence/metrics_after.prom"
    raw_path.write_text(
        raw_path.read_text().replace(
            'vllm:generation_tokens_total{engine="0",model_name="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"} 144.0',
            'vllm:generation_tokens_total{engine="0",model_name="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"} 145.0',
        ),
        encoding="utf-8",
    )
    result["raw_evidence"]["metrics_after"]["sha256"] = validator.sha256_file(
        raw_path
    )
    _write_json(result_path, result)
    _rehash_manifest(cell_root)

    with pytest.raises(validator.ValidationError, match="not exactly one token"):
        validator.validate_run(run_root)


def test_joint_patch_marker_is_rejected(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    cell_root = run_root / "block-02/P0"
    log_path = cell_root / "server/vllm_8100.log"
    log_path.write_text(
        log_path.read_text() + "[sched_policy_patch:physical_kv] active=1\n",
        encoding="utf-8",
    )
    _rehash_manifest(cell_root)

    with pytest.raises(validator.ValidationError, match="Joint patch evidence"):
        validator.validate_run(run_root)


def test_p0_nonzero_prefix_counter_is_rejected(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    cell_root = run_root / "block-01/P0"
    result_path = cell_root / "evidence/result.json"
    result = json.loads(result_path.read_text())
    result["vllm_metric_deltas"]["vllm:prefix_cache_queries_total"] = 1.0
    raw_path = cell_root / "evidence/metrics_after.prom"
    raw_path.write_text(
        raw_path.read_text().replace(
            'vllm:prefix_cache_queries_total{engine="0",model_name="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"} 0.0',
            'vllm:prefix_cache_queries_total{engine="0",model_name="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"} 1.0',
        ),
        encoding="utf-8",
    )
    result["raw_evidence"]["metrics_after"]["sha256"] = validator.sha256_file(
        raw_path
    )
    _write_json(result_path, result)
    _rehash_manifest(cell_root)

    with pytest.raises(validator.ValidationError, match="P0 produced"):
        validator.validate_run(run_root)


def test_raw_prometheus_delta_tamper_is_rejected(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    cell_root = run_root / "block-01/P1"
    raw_path = cell_root / "evidence/metrics_after.prom"
    raw_path.write_text(
        raw_path.read_text().replace(
            'vllm:request_prefill_time_seconds_sum{engine="0",model_name="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"} 70.0',
            'vllm:request_prefill_time_seconds_sum{engine="0",model_name="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"} 71.0',
        ),
        encoding="utf-8",
    )
    result_path = cell_root / "evidence/result.json"
    result = json.loads(result_path.read_text())
    result["raw_evidence"]["metrics_after"]["sha256"] = validator.sha256_file(
        raw_path
    )
    _write_json(result_path, result)
    _rehash_manifest(cell_root)

    with pytest.raises(validator.ValidationError, match="raw metric .* delta"):
        validator.validate_run(run_root)


def test_reused_os_server_process_identity_is_rejected(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    source_path = run_root / "block-01/P0/server_identity.json"
    target_root = run_root / "block-02/P1"
    target_path = target_root / "server_identity.json"
    source = json.loads(source_path.read_text())
    target = json.loads(target_path.read_text())
    for key in (
        "pid",
        "proc_start_ticks",
        "executable",
        "cmdline_sha256",
        "process_identity_sha256",
    ):
        target[key] = source[key]
    _write_json(target_path, target)
    _rehash_manifest(target_root)

    with pytest.raises(validator.ValidationError, match="OS process identity was reused"):
        validator.validate_run(run_root)


def test_valid_but_zero_latency_gain_stops_without_promotion(tmp_path: Path) -> None:
    result = validator.validate_run(
        _make_run(
            tmp_path,
            p1_task_e2e_s=6.1,
            p1_request_s=2.0,
            p1_prefill_s=100.0,
        )
    )

    assert result["valid"] is True
    assert result["promotion_passed"] is False
    assert result["selected_policy"] == "no_latency_claim"
    assert result["promotion_gates"]["p1_hit_ratio_at_least_60pct"] is True
    assert result["promotion_gates"]["prefill_reduction_at_least_15pct"] is False
    assert result["promotion_gates"]["paired_source_bootstrap_lower_above_zero"] is False
    assert result["stopping_rule"].endswith("no_optional_third_block")
