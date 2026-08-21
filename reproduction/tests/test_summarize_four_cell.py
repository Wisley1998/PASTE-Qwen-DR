from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from paste_repro.mapper import URLRankMapper, save_artifact  # noqa: E402
import summarize_four_cell as summarize_four_cell_module  # noqa: E402
from summarize_four_cell import (  # noqa: E402
    _pair_effect,
    canonical_sha256,
    file_sha256,
    load_fixed_manifest,
    repository_display_path,
    summarize_four_cell,
)


CELL_CONFIG = {
    "A": {
        "mode": "none",
        "joint": False,
        "ends": {"trace_000": 12.0, "trace_001": 9.0},
        "latencies": [4.0, 2.0, 3.0],
        "queue": 2.0,
    },
    "B": {
        "mode": "learned",
        "joint": False,
        "ends": {"trace_000": 10.0, "trace_001": 8.0},
        "latencies": [3.0, 1.5, 2.5],
        "queue": 1.5,
    },
    "C": {
        "mode": "none",
        "joint": True,
        "ends": {"trace_000": 11.0, "trace_001": 8.0},
        "latencies": [3.5, 1.8, 2.7],
        "queue": 1.6,
    },
    "D": {
        "mode": "learned",
        "joint": True,
        "ends": {"trace_000": 8.5, "trace_001": 7.0},
        "latencies": [2.5, 1.0, 2.0],
        "queue": 1.0,
    },
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request(call_index: int, mode: str, message: str, mapper_sha: str) -> dict:
    original_wait = 0.25
    saved_wait = 0.0 if mode == "none" else 0.15
    request = {
        "call_index": call_index,
        "wait_after_prev_s": original_wait - saved_wait,
        "wait_after_prev_original_s": original_wait,
        "prompt_tokens": 100 + call_index,
        "original_prompt_tokens": 100 + call_index,
        "target_output_tokens": 200 + call_index,
        "max_tokens": 128,
        "truncated": False,
        "messages": [{"role": "user", "content": message}],
        "tool_overlap_mode": mode,
        "tool_overlap_saved_s": saved_wait,
        "tool_overlap_window_s": 0.20 if mode == "learned" else 0.0,
        "tool_kind_before": "visit" if call_index else "",
        "tool_cache_hit": False,
    }
    if mode == "learned":
        request.update(
            {
                "tool_prediction_artifact_sha256": mapper_sha,
                "tool_prediction_top_k": 5,
                "tool_prediction_candidates": ["https://example.test/predicted"],
                "tool_prediction_candidate_count": 1,
                "tool_prediction_exact_hits": 1,
                "tool_prediction_waste": 0,
            }
        )
    return request


def _workload(mode: str, mapper_sha: str, role: str) -> dict:
    role_sources = {
        "calibration": [("trace_000", "calibration.jsonl", 0.0)],
        "tuning": [
            ("trace_000", "tuning-zero.jsonl", 2.0),
            ("trace_001", "tuning-one.jsonl", 1.0),
        ],
        "final": [
            ("trace_000", "session-zero.jsonl", 2.0),
            ("trace_001", "session-one.jsonl", 1.0),
        ],
    }
    metadata = {
        "target_trace_count": len(role_sources[role]),
        "max_model_len": 16384,
        "max_output_tokens_cap": 128,
        "output_token_buffer": 8,
        "min_output_tokens_floor": 64,
        "duplicate_seed": 20260417,
        "tool_overlap_efficiency": 1.0,
        "tool_overlap_mode": mode,
    }
    if mode == "learned":
        metadata.update(
            {
                "tool_prediction_artifact_sha256": mapper_sha,
                "tool_prediction_top_k": 5,
            }
        )
    traces = []
    for trace_id, source, arrival in role_sources[role]:
        request_count = 2 if trace_id == "trace_000" else 1
        traces.append(
            {
                "trace_id": trace_id,
                "source_trace": f"/fixed/{role}/{source}",
                "variant_index": int(trace_id.rsplit("_", 1)[1]),
                "duplicated": False,
                "prefix_char": "",
                "initial_delay_s": arrival,
                "requests": [
                    _request(index, mode, f"same {role} prompt {trace_id}/{index}", mapper_sha)
                    for index in range(request_count)
                ],
            }
        )
    return {"meta": metadata, "traces": traces}


def _source_sequence(workload: dict) -> list[str]:
    return [Path(trace["source_trace"]).name for trace in workload["traces"]]


def _scheduler_id(
    trace_id: str,
    call_index: int,
    request_index: int,
    mode: str,
) -> tuple[str, dict]:
    remaining = 1 if call_index == 0 else 0
    predicted_wait = (0.25 if mode == "none" else 0.10) / 10.0 if remaining else 0.0
    metadata = {
        "t": trace_id,
        "c": call_index,
        "i": request_index,
        "n": request_index + 1 + remaining,
        "rc": remaining,
        "nw": predicted_wait,
        "nwc": 0.0,
        "rtw": predicted_wait,
        "pt": 100 + call_index,
        "mt": 128,
        "ms": "online",
        "po": 128,
    }
    if remaining:
        metadata["npo"] = 128
    encoded = json.dumps(
        metadata,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8").hex()
    return f"schedx{encoded}z", metadata


def _replace_scheduler_metadata(
    event: dict,
    *,
    updates: dict | None = None,
    remove: tuple[str, ...] = (),
) -> dict:
    request_id = event["request_id"]
    encoded = request_id[len("schedx") : -len("z")]
    metadata = json.loads(bytes.fromhex(encoded).decode("utf-8"))
    metadata.update(updates or {})
    for field in remove:
        metadata.pop(field, None)
    event["request_id"] = (
        "schedx"
        + json.dumps(
            metadata,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8").hex()
        + "z"
    )
    return metadata


def _events(
    mode: str,
    mapper_sha: str,
    ends: dict[str, float],
    latencies: list[float],
) -> list[dict]:
    definitions = [
        ("trace_000", "session-zero.jsonl", 0, 0, ends["trace_000"] - 4.0),
        ("trace_000", "session-zero.jsonl", 1, 1, ends["trace_000"]),
        ("trace_001", "session-one.jsonl", 0, 0, ends["trace_001"]),
    ]
    events = []
    for (trace_id, source, call_index, request_index, end_offset), latency in zip(
        definitions, latencies, strict=True
    ):
        request_id, scheduler = _scheduler_id(
            trace_id,
            call_index,
            request_index,
            mode,
        )
        original_wait = 0.25
        saved_wait = 0.0 if mode == "none" else 0.15
        realized_wait = original_wait - saved_wait
        event = {
            "trace_id": trace_id,
            "source_trace": f"/copied/{source}",
            "duplicated": False,
            "prefix_char": "",
            "call_index": call_index,
            "prompt_tokens": 100 + call_index,
            "target_output_tokens": 200 + call_index,
            "max_tokens": 128,
            "truncated": False,
            "metadata_source": "online",
            "request_id": request_id,
            "scheduled_remaining_calls_after": scheduler["rc"],
            "scheduled_total_calls": scheduler["n"],
            "scheduled_nw": scheduler["nw"],
            "scheduled_nw_reliability": scheduler["nwc"],
            "scheduled_rtw": scheduler["rtw"],
            "nw_source": "predicted",
            "oracle_next_tool_wait_s": None,
            "oracle_remaining_tool_wait_s": None,
            "oracle_remaining_calls_after": None,
            "oracle_total_calls": None,
            "po_predicted": 128,
            "po_actual": 128,
            "ok": True,
            "http_status": 200,
            "attempts": 1,
            "attempt_history": [
                {
                    "attempt": 1,
                    "transport": "http",
                    "outcome": "success",
                    "http_status": 200,
                    "error_type": None,
                    "error": None,
                    "duration_s": latency,
                    "retryable": False,
                    "will_retry": False,
                    "retry_backoff_s": 0.0,
                    "delivery_ambiguous": False,
                }
            ],
            "latency_s": latency,
            "request_start_offset_s": end_offset - latency,
            "request_end_offset_s": end_offset,
            "tool_wait_mode": "sleep",
            "scheduled_wait_original_s": original_wait,
            "scheduled_wait_s": realized_wait if call_index == 0 else realized_wait / 10,
            "tool_overlap_saved_s": saved_wait,
            "tool_overlap_window_s": 0.20 if mode == "learned" else 0.0,
            "tool_overlap_mode": mode,
            "tool_prediction_candidate_count": 1 if mode == "learned" else 0,
            "tool_prediction_exact_hits": 1 if mode == "learned" else 0,
            "tool_prediction_waste": 0,
            "tool_prediction_artifact_sha256": mapper_sha if mode == "learned" else "",
            "tool_prediction_top_k": 5 if mode == "learned" else 0,
        }
        events.append(event)
    return events


def _make_fixed_manifest(root: Path) -> tuple[Path, dict[tuple[str, str], Path], str]:
    calibration_entries = [
        {"session_id": "calibration.jsonl", "sha256": "1" * 64}
    ]
    tuning_entries = [
        {"session_id": "tuning-zero.jsonl", "sha256": "2" * 64},
        {"session_id": "tuning-one.jsonl", "sha256": "3" * 64},
    ]
    final_entries = [
        {"session_id": "session-zero.jsonl", "sha256": "4" * 64},
        {"session_id": "session-one.jsonl", "sha256": "5" * 64},
    ]
    mapper_artifact = URLRankMapper().to_artifact(
        {
            "algorithm": "unit test",
            "train_sessions": calibration_entries,
            "calibration_sessions": calibration_entries,
            "tuning_sessions": tuning_entries,
            "final_sessions": final_entries,
        }
    )
    mapper_path = root / "mapper.json"
    save_artifact(mapper_path, mapper_artifact)
    mapper_sha = mapper_artifact["artifact_sha256"]

    split = {
        "schema": "paste_repro.fixed_three_way_split",
        "version": 1,
        "calibration_sessions": calibration_entries,
        "tuning_sessions": tuning_entries,
        "final_sessions": final_entries,
    }
    split["manifest_sha256"] = canonical_sha256(split)
    split_path = root / "split_manifest.json"
    _write_json(split_path, split)

    workload_paths: dict[tuple[str, str], Path] = {}
    records: dict[str, dict[str, dict]] = {}
    for role in ("calibration", "tuning", "final"):
        records[role] = {}
        for mode in ("none", "learned"):
            directory = root / "workloads" / role / mode
            workload_path = directory / "prepared_workload.json"
            summary_path = directory / "workload_summary.json"
            workload = _workload(mode, mapper_sha, role)
            _write_json(workload_path, workload)
            _write_json(summary_path, {"role": role, "mode": mode})
            sequence = _source_sequence(workload)
            workload_paths[(role, mode)] = workload_path
            records[role][mode] = {
                "prepared_workload": workload_path.relative_to(root).as_posix(),
                "prepared_workload_sha256": file_sha256(workload_path),
                "workload_summary": summary_path.relative_to(root).as_posix(),
                "workload_summary_sha256": file_sha256(summary_path),
                "trace_count": len(sequence),
                "unique_source_session_count": len(set(sequence)),
                "source_sequence_sha256": canonical_sha256(sequence),
                "tool_overlap_mode": mode,
                **(
                    {
                        "mapper_artifact_sha256": mapper_sha,
                        "tool_prediction_top_k": 5,
                    }
                    if mode == "learned"
                    else {}
                ),
            }

    def _cell_inputs(role: str) -> dict:
        result = {}
        for name, policy, mode in (
            ("fcfs_none", "fcfs", "none"),
            ("fcfs_learned", "fcfs", "learned"),
            ("joint_none", "online_joint_pacer_v2", "none"),
            ("joint_learned", "online_joint_pacer_v2", "learned"),
        ):
            result[name] = {
                "policy": policy,
                "tool_overlap_mode": mode,
                "evaluation_workload": records[role][mode]["prepared_workload"],
                "online_calibration_workload": records["calibration"][mode][
                    "prepared_workload"
                ],
            }
        return result

    manifest = {
        "schema": "paste_repro.fixed_workload_bundle",
        "version": 1,
        "fixed_split_manifest": split_path.relative_to(root).as_posix(),
        "fixed_split_manifest_sha256": split["manifest_sha256"],
        "calibration_only_mapper": mapper_path.relative_to(root).as_posix(),
        "calibration_only_mapper_sha256": mapper_sha,
        "parameters": {
            "speedup": 10.0,
            "tool_prediction_top_k": 5,
            "max_model_len": 16384,
            "max_output_tokens_cap": 128,
            "output_token_buffer": 8,
            "min_output_tokens_floor": 64,
            "seed": 20260417,
            "target_trace_counts": {"calibration": 1, "tuning": 2, "final": 2},
        },
        "workloads": records,
        "four_cell_inputs": {
            "tuning": _cell_inputs("tuning"),
            "final": _cell_inputs("final"),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, workload_paths, mapper_sha


def _make_heldout_manifest(
    root: Path,
) -> tuple[Path, dict[tuple[str, str], Path], str]:
    parent_path, workload_paths, mapper_sha = _make_fixed_manifest(root)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    records: dict[str, dict] = {}
    for mode in ("none", "learned"):
        tuning = json.loads(workload_paths[("tuning", mode)].read_text(encoding="utf-8"))
        final = json.loads(workload_paths[("final", mode)].read_text(encoding="utf-8"))
        source_traces = [*tuning["traces"], *final["traces"]]
        traces = []
        for index, source in enumerate(source_traces):
            trace = copy.deepcopy(source)
            trace.update(
                {
                    "trace_id": f"heldout_{index:03d}",
                    "variant_index": index,
                    "duplicated": False,
                    "prefix_char": "",
                }
            )
            traces.append(trace)
        metadata = copy.deepcopy(tuning["meta"])
        metadata.update(
            {
                "source_trace_dir": None,
                "source_roles": ["tuning", "final"],
                "evidence_role": "heldout_load_sensitivity_not_untouched_final",
                "target_trace_count": len(traces),
                "duplicates_added": 0,
                "total_truncated_calls": sum(
                    int(trace.get("truncated_calls", 0)) for trace in traces
                ),
            }
        )
        workload = {"meta": metadata, "traces": traces}
        directory = root / "workloads" / "heldout" / mode
        workload_path = directory / "prepared_workload.json"
        summary_path = directory / "workload_summary.json"
        _write_json(workload_path, workload)
        _write_json(summary_path, {"role": "heldout", "mode": mode})
        sequence = _source_sequence(workload)
        workload_paths[("heldout", mode)] = workload_path
        records[mode] = {
            "prepared_workload": workload_path.relative_to(root).as_posix(),
            "prepared_workload_sha256": file_sha256(workload_path),
            "workload_summary": summary_path.relative_to(root).as_posix(),
            "workload_summary_sha256": file_sha256(summary_path),
            "trace_count": len(sequence),
            "unique_source_session_count": len(set(sequence)),
            "source_sequence_sha256": canonical_sha256(sequence),
            "tool_overlap_mode": mode,
            **(
                {
                    "mapper_artifact_sha256": mapper_sha,
                    "tool_prediction_top_k": 5,
                }
                if mode == "learned"
                else {}
            ),
        }

    manifest = copy.deepcopy(parent)
    manifest["derived_from_manifest"] = parent_path.relative_to(root).as_posix()
    manifest["derived_from_manifest_sha256"] = parent["manifest_sha256"]
    manifest["workloads"]["heldout"] = records
    manifest["parameters"]["target_trace_counts"]["heldout"] = 4
    manifest["four_cell_inputs"]["heldout"] = {
        name: {
            "policy": policy,
            "tool_overlap_mode": mode,
            "evaluation_workload": records[mode]["prepared_workload"],
            "online_calibration_workload": parent["workloads"]["calibration"][mode][
                "prepared_workload"
            ],
        }
        for name, policy, mode in (
            ("fcfs_none", "fcfs", "none"),
            ("fcfs_learned", "fcfs", "learned"),
            ("joint_none", "online_joint_pacer_v2", "none"),
            ("joint_learned", "online_joint_pacer_v2", "learned"),
        )
    }
    manifest.setdefault("contamination_guards", {}).update(
        {
            "heldout_union_sessions": (
                "tuning plus previously inspected final; calibration excluded"
            ),
            "heldout_is_not_new_final": True,
        }
    )
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = root / "manifest_heldout.json"
    _write_json(manifest_path, manifest)
    return manifest_path, workload_paths, mapper_sha


def _write_run(
    root: Path,
    cell: str,
    replicate: int,
    workload_paths: dict[tuple[str, str], Path],
    mapper_sha: str,
) -> Path:
    config = CELL_CONFIG[cell]
    run = root / f"{cell.lower()}_r{replicate}"
    run.mkdir(parents=True)
    master_workload = workload_paths[("final", config["mode"])]
    shutil.copyfile(master_workload, run / "prepared_workload.json")
    events = _events(config["mode"], mapper_sha, config["ends"], config["latencies"])
    calibration = workload_paths[("calibration", config["mode"])].resolve()
    summary = {
        "speedup": 10.0,
        "requests_failed": 0,
        "requests_success": len(events),
        "requests_total": len(events),
        "configured_max_request_attempts": 2,
        "request_attempts_total": len(events),
        "retry_count": 0,
        "retried_request_count": 0,
        "retry_success_count": 0,
        "ambiguous_retry_count": 0,
        "final_failure_count": 0,
        "metadata_source": "online",
        "scheduler_metadata_mode": "online",
        "scheduler_calibration_workload": str(calibration),
        "scheduler_environment": {
            "VLLM_SCHED_POLICY": (
                "online_joint_pacer_v2" if config["joint"] else "fcfs"
            ),
            "VLLM_SCHED_PRED_OUT_ENABLE": "1",
            "VLLM_MAX_NUM_SEQS": "8",
        },
        "max_active_traces": 2,
        "tool_wait_mode": "sleep",
        "avg_queue_time_s": config["queue"],
        "experiment_wall_time_s": max(config["ends"].values()) + 0.5,
        "workload": {
            "trace_count": 2,
            "request_count": len(events),
            "tool_overlap_mode": config["mode"],
            **(
                {
                    "tool_prediction": {
                        "candidate_count": 3,
                        "exact_hits": 3,
                        "waste": 0,
                        "artifact_sha256": mapper_sha,
                        "top_k": 5,
                    }
                }
                if config["mode"] == "learned"
                else {}
            ),
        },
    }
    _write_json(run / "summary.json", summary)
    (run / "request_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    log = "vLLM API server version 0.10.1\n"
    if config["joint"]:
        log += (
            "[sched_policy_patch] installed policy=online_joint_pacer_v2 "
            "v0=True v1=True\n"
            "[sched_policy_patch:joint] pending_returns=2 running=4\n"
        )
    (run / "server.log").write_text(log, encoding="utf-8")
    return run


def _write_heldout_run(
    root: Path,
    cell: str,
    replicate: int,
    workload_paths: dict[tuple[str, str], Path],
    mapper_sha: str,
) -> Path:
    config = CELL_CONFIG[cell]
    mode = config["mode"]
    run = root / f"heldout_{cell.lower()}_r{replicate}"
    run.mkdir(parents=True)
    workload_path = workload_paths[("heldout", mode)]
    shutil.copyfile(workload_path, run / "prepared_workload.json")
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    events = []
    completion_by_trace: dict[str, float] = {}
    for trace_number, trace in enumerate(workload["traces"]):
        completion = 12.0 + trace_number
        completion_by_trace[trace["trace_id"]] = completion
        for request_index, request in enumerate(trace["requests"]):
            call_index = request["call_index"]
            request_id, scheduler = _scheduler_id(
                trace["trace_id"],
                call_index,
                request_index,
                mode,
            )
            end_offset = (
                completion - 2.0
                if request_index + 1 < len(trace["requests"])
                else completion
            )
            realized_wait = request["wait_after_prev_s"]
            event = {
                "trace_id": trace["trace_id"],
                "source_trace": trace["source_trace"],
                "duplicated": False,
                "prefix_char": "",
                "call_index": call_index,
                "prompt_tokens": request["prompt_tokens"],
                "target_output_tokens": request["target_output_tokens"],
                "max_tokens": request["max_tokens"],
                "truncated": request["truncated"],
                "metadata_source": "online",
                "request_id": request_id,
                "scheduled_remaining_calls_after": scheduler["rc"],
                "scheduled_total_calls": scheduler["n"],
                "scheduled_nw": scheduler["nw"],
                "scheduled_nw_reliability": scheduler["nwc"],
                "scheduled_rtw": scheduler["rtw"],
                "nw_source": "predicted",
                "oracle_next_tool_wait_s": None,
                "oracle_remaining_tool_wait_s": None,
                "oracle_remaining_calls_after": None,
                "oracle_total_calls": None,
                "po_predicted": 128,
                "po_actual": 128,
                "ok": True,
                "http_status": 200,
                "attempts": 1,
                "attempt_history": [
                    {
                        "attempt": 1,
                        "transport": "http",
                        "outcome": "success",
                        "http_status": 200,
                        "error_type": None,
                        "error": None,
                        "duration_s": 1.0,
                        "retryable": False,
                        "will_retry": False,
                        "retry_backoff_s": 0.0,
                        "delivery_ambiguous": False,
                    }
                ],
                "latency_s": 1.0,
                "request_start_offset_s": end_offset - 1.0,
                "request_end_offset_s": end_offset,
                "tool_wait_mode": "sleep",
                "scheduled_wait_original_s": request["wait_after_prev_original_s"],
                "scheduled_wait_s": (
                    realized_wait if call_index == 0 else realized_wait / 10.0
                ),
                "tool_overlap_saved_s": request["tool_overlap_saved_s"],
                "tool_overlap_window_s": request["tool_overlap_window_s"],
                "tool_overlap_mode": mode,
                "tool_prediction_candidate_count": request.get(
                    "tool_prediction_candidate_count", 0
                ),
                "tool_prediction_exact_hits": request.get(
                    "tool_prediction_exact_hits", 0
                ),
                "tool_prediction_waste": request.get("tool_prediction_waste", 0),
                "tool_prediction_artifact_sha256": request.get(
                    "tool_prediction_artifact_sha256", ""
                ),
                "tool_prediction_top_k": request.get("tool_prediction_top_k", 0),
            }
            events.append(event)
    candidate_total = sum(
        int(request.get("tool_prediction_candidate_count", 0))
        for trace in workload["traces"]
        for request in trace["requests"]
    )
    exact_hit_total = sum(
        int(request.get("tool_prediction_exact_hits", 0))
        for trace in workload["traces"]
        for request in trace["requests"]
    )
    waste_total = sum(
        int(request.get("tool_prediction_waste", 0))
        for trace in workload["traces"]
        for request in trace["requests"]
    )
    summary = {
        "speedup": 10.0,
        "requests_failed": 0,
        "requests_success": len(events),
        "requests_total": len(events),
        "configured_max_request_attempts": 2,
        "request_attempts_total": len(events),
        "retry_count": 0,
        "retried_request_count": 0,
        "retry_success_count": 0,
        "ambiguous_retry_count": 0,
        "final_failure_count": 0,
        "metadata_source": "online",
        "scheduler_metadata_mode": "online",
        "scheduler_calibration_workload": str(
            workload_paths[("calibration", mode)].resolve()
        ),
        "scheduler_environment": {
            "VLLM_SCHED_POLICY": (
                "online_joint_pacer_v2" if config["joint"] else "fcfs"
            ),
            "VLLM_SCHED_PRED_OUT_ENABLE": "1",
            "VLLM_MAX_NUM_SEQS": "8",
        },
        "max_active_traces": len(workload["traces"]),
        "tool_wait_mode": "sleep",
        "avg_queue_time_s": config["queue"],
        "experiment_wall_time_s": max(completion_by_trace.values()) + 0.5,
        "workload": {
            "trace_count": len(workload["traces"]),
            "request_count": len(events),
            "tool_overlap_mode": mode,
            **(
                {
                    "tool_prediction": {
                        "candidate_count": candidate_total,
                        "exact_hits": exact_hit_total,
                        "waste": waste_total,
                        "artifact_sha256": mapper_sha,
                        "top_k": 5,
                    }
                }
                if mode == "learned"
                else {}
            ),
        },
    }
    _write_json(run / "summary.json", summary)
    (run / "request_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    log = "vLLM API server version 0.10.1\n"
    if config["joint"]:
        log += (
            "[sched_policy_patch] installed policy=online_joint_pacer_v2 "
            "v0=True v1=True\n"
            "[sched_policy_patch:joint] pending_returns=2 running=4\n"
        )
    (run / "server.log").write_text(log, encoding="utf-8")
    return run


def _fixture(root: Path, replicates: int = 1) -> tuple[Path, dict[str, list[Path]]]:
    manifest, workload_paths, mapper_sha = _make_fixed_manifest(root)
    groups = {
        cell: [
            _write_run(root, cell, replicate, workload_paths, mapper_sha)
            for replicate in range(1, replicates + 1)
        ]
        for cell in ("A", "B", "C", "D")
    }
    return manifest, groups


def _heldout_fixture(root: Path) -> tuple[Path, dict[str, list[Path]]]:
    manifest, workload_paths, mapper_sha = _make_heldout_manifest(root)
    groups = {
        cell: [_write_heldout_run(root, cell, 1, workload_paths, mapper_sha)]
        for cell in ("A", "B", "C", "D")
    }
    return manifest, groups


def _summarize(manifest: Path, groups: dict[str, list[Path]]) -> dict:
    return summarize_four_cell(groups, manifest_path=manifest, role="final")


class FourCellSummaryTests(unittest.TestCase):
    def test_public_paths_are_repository_relative_with_absolute_fallback(self) -> None:
        repository_path = REPRODUCTION_ROOT / "artifacts" / "runs" / "example"
        rendered = repository_display_path(repository_path)
        self.assertEqual(rendered, "reproduction/artifacts/runs/example")
        self.assertFalse(Path(rendered).is_absolute())

        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / "external-run"
            self.assertEqual(
                repository_display_path(external),
                external.resolve().as_posix(),
            )

            root = Path(temporary).resolve()
            manifest, groups = _fixture(root)
            with mock.patch.object(
                summarize_four_cell_module,
                "REPOSITORY_ROOT",
                root,
            ):
                result = _summarize(manifest, groups)

        self.assertEqual(
            result["comparison_invariants"]["fixed_workload_manifest"],
            "manifest.json",
        )
        for cell in ("A", "B", "C", "D"):
            run_path = result["cells"][cell]["runs"][0]["run_path"]
            self.assertEqual(run_path, f"{cell.lower()}_r1")
            self.assertFalse(Path(run_path).is_absolute())

    def test_retry_accounting_is_recomputed_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            run = groups["A"][0]
            events_path = run / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            success = dict(events[0]["attempt_history"][0])
            success["attempt"] = 2
            events[0]["attempts"] = 2
            events[0]["attempt_history"] = [
                {
                    "attempt": 1,
                    "transport": "aiohttp_connection",
                    "outcome": "transport_error",
                    "http_status": None,
                    "error_type": "ServerDisconnectedError",
                    "error": "ServerDisconnectedError('server disconnected')",
                    "duration_s": 0.001,
                    "retryable": True,
                    "will_retry": True,
                    "retry_backoff_s": 1.0,
                    "delivery_ambiguous": True,
                },
                success,
            ]
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary.update(
                {
                    "request_attempts_total": len(events) + 1,
                    "retry_count": 1,
                    "retried_request_count": 1,
                    "retry_success_count": 1,
                    "ambiguous_retry_count": 1,
                }
            )
            _write_json(summary_path, summary)

            result = _summarize(manifest, groups)

            self.assertEqual(result["cells"]["A"]["retry_accounting"]["retry_count"], 1)
            accounting = result["comparison_invariants"]["retry_accounting"]
            self.assertEqual(accounting["requests_total"], 12)
            self.assertEqual(accounting["request_attempts_total"], 13)
            self.assertEqual(accounting["retry_count"], 1)
            self.assertEqual(accounting["ambiguous_retry_count"], 1)
            self.assertTrue(
                result["comparison_invariants"][
                    "all_requests_finally_succeeded"
                ]
            )
            self.assertFalse(
                result["comparison_invariants"][
                    "all_requests_succeeded_exactly_once"
                ]
            )

    def test_heldout_role_is_strict_union_and_summarizable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, groups = _heldout_fixture(root)
            result = summarize_four_cell(
                groups,
                manifest_path=manifest,
                role="heldout",
            )
            invariants = result["comparison_invariants"]
            self.assertEqual(invariants["fixed_role"], "heldout")
            self.assertEqual(
                invariants["evidence_role"],
                "heldout_load_sensitivity_not_untouched_final",
            )
            self.assertEqual(invariants["trace_count"], 4)
            self.assertIsNotNone(invariants["heldout_parent_manifest_sha256"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = _make_heldout_manifest(root)
            workload_path = root / "workloads/heldout/none/prepared_workload.json"
            workload = json.loads(workload_path.read_text(encoding="utf-8"))
            workload["traces"][0]["requests"][0]["messages"][0][
                "content"
            ] = "not the source trace"
            _write_json(workload_path, workload)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["workloads"]["heldout"]["none"][
                "prepared_workload_sha256"
            ] = file_sha256(workload_path)
            payload.pop("manifest_sha256")
            payload["manifest_sha256"] = canonical_sha256(payload)
            _write_json(manifest, payload)
            with self.assertRaisesRegex(ValueError, "exact retagged source trace"):
                load_fixed_manifest(manifest, "heldout")

    def test_flow_makespan_effects_and_interpolated_percentiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary), replicates=2)
            result = _summarize(manifest, groups)

        invariants = result["comparison_invariants"]
        self.assertEqual(invariants["replicate_count_per_cell"], 2)
        self.assertEqual(invariants["fixed_role"], "final")
        self.assertEqual(invariants["request_count"], 3)
        self.assertEqual(invariants["configured_max_request_attempts"], 2)
        self.assertEqual(invariants["retry_accounting"]["requests_total"], 24)
        self.assertEqual(
            invariants["retry_accounting"]["request_attempts_total"], 24
        )
        self.assertTrue(invariants["all_requests_finally_succeeded"])
        self.assertTrue(invariants["all_requests_succeeded_exactly_once"])
        self.assertEqual(result["cells"]["A"]["task_flow_time_s"]["mean"], 9.0)
        self.assertEqual(result["cells"]["A"]["task_flow_time_s"]["p50"], 9.0)
        self.assertAlmostEqual(result["cells"]["A"]["task_flow_time_s"]["p95"], 9.9)
        self.assertEqual(result["cells"]["A"]["task_makespan_s"], 12.0)
        effects = result["effects"]
        self.assertEqual(
            effects["tool_only_A_to_B"]["absolute_reduction"]["task_flow_time_s"][
                "mean"
            ],
            1.5,
        )
        self.assertEqual(
            effects["full_A_to_D"]["absolute_reduction"]["task_flow_time_s"]["mean"],
            2.75,
        )
        self.assertEqual(
            effects["interaction"]["absolute_reduction"]["task_flow_time_s"]["mean"],
            0.25,
        )

    def test_manifest_checksum_role_and_exact_workload_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["parameters"]["speedup"] = 99
            _write_json(manifest, payload)
            with self.assertRaisesRegex(ValueError, "manifest checksum mismatch"):
                _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "checksummed fixed-role workload"):
                summarize_four_cell(groups, manifest_path=manifest, role="tuning")

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            workload_path = groups["C"][0] / "prepared_workload.json"
            workload = json.loads(workload_path.read_text(encoding="utf-8"))
            workload["traces"][0]["requests"][0]["wait_after_prev_s"] = 99
            _write_json(workload_path, workload)
            with self.assertRaisesRegex(ValueError, "checksummed fixed-role workload"):
                _summarize(manifest, groups)

    def test_online_request_id_and_latency_are_verified_not_just_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            events_path = groups["A"][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[0]["request_id"] = "schedx7b7dz"
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "request_id is not online metadata"):
                _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            events_path = groups["A"][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            request_id = events[0]["request_id"]
            encoded = request_id[len("schedx") : -len("z")]
            metadata = json.loads(bytes.fromhex(encoded).decode("utf-8"))
            metadata["nw"] = 99.0
            events[0]["scheduled_nw"] = 99.0
            events[0]["request_id"] = (
                "schedx"
                + json.dumps(
                    metadata,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8").hex()
                + "z"
            )
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "calibration-only causal prediction"):
                _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            events_path = groups["A"][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[0]["latency_s"] += 1
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "end minus start"):
                _summarize(manifest, groups)

    def test_online_next_wait_reliability_is_strict_and_legacy_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            events_path = groups["A"][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[0]["scheduled_nw_reliability"] = 0.5
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "event/reliability mismatch"):
                _summarize(manifest, groups)

        for invalid in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                manifest, groups = _fixture(Path(temporary))
                events_path = groups["A"][0] / "request_events.jsonl"
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                _replace_scheduler_metadata(events[0], updates={"nwc": invalid})
                events[0]["scheduled_nw_reliability"] = invalid
                events_path.write_text(
                    "".join(json.dumps(event) + "\n" for event in events),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "scheduler nwc"):
                    _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            events_path = groups["A"][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            _replace_scheduler_metadata(events[0], updates={"nwc": 0.5})
            events[0]["scheduled_nw_reliability"] = 0.5
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "calibration-only causal prediction"):
                _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            events_path = groups["A"][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            _replace_scheduler_metadata(events[0], remove=("nwc",))
            events[0].pop("scheduled_nw_reliability")
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            result = _summarize(manifest, groups)
            self.assertEqual(
                result["status"],
                "functional_four_cell_not_full_paper_reproduction",
            )

    def test_oracle_request_id_cannot_smuggle_online_reliability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            events_path = groups["A"][0] / "request_events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            _replace_scheduler_metadata(events[0], updates={"ms": "oracle", "nwc": 0.5})
            events[0]["scheduled_nw_reliability"] = 0.5
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "request_id is not online metadata"):
                _summarize(manifest, groups)

    def test_learned_fields_cannot_be_missing_even_when_manifest_is_rechecksummed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, groups = _fixture(root)
            master = root / "workloads/final/learned/prepared_workload.json"
            workload = json.loads(master.read_text(encoding="utf-8"))
            del workload["traces"][0]["requests"][0][
                "tool_prediction_artifact_sha256"
            ]
            _write_json(master, workload)
            for cell in ("B", "D"):
                shutil.copyfile(master, groups[cell][0] / "prepared_workload.json")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["workloads"]["final"]["learned"][
                "prepared_workload_sha256"
            ] = file_sha256(master)
            payload.pop("manifest_sha256")
            payload["manifest_sha256"] = canonical_sha256(payload)
            _write_json(manifest, payload)
            with self.assertRaisesRegex(ValueError, "mapper checksum mismatch"):
                _summarize(manifest, groups)

    def test_fcfs_and_joint_v1_policy_evidence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            (groups["A"][0] / "server.log").write_text(
                "vLLM API server version 0.10.1\n"
                "[sched_policy_patch:joint] pending_returns=1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "FCFS run contains"):
                _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            (groups["D"][0] / "server.log").write_text(
                "vLLM API server version 0.10.1\n"
                "[sched_policy_patch] installed policy=online_joint_pacer_v2 "
                "v0=True v1=False\n"
                "[sched_policy_patch:joint] pending_returns=1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "clean v1 install/runtime"):
                _summarize(manifest, groups)

    def test_duplicate_identity_config_mismatch_and_zero_relative_fail_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            run = groups["A"][0]
            lines = (run / "request_events.jsonl").read_text().splitlines()
            (run / "request_events.jsonl").write_text(
                "\n".join([*lines, lines[0]]) + "\n", encoding="utf-8"
            )
            summary = json.loads((run / "summary.json").read_text())
            summary["requests_total"] = 4
            summary["requests_success"] = 4
            summary["request_attempts_total"] = 4
            summary["workload"]["request_count"] = 4
            _write_json(run / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "duplicate request event identity"):
                _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            summary_path = groups["B"][0] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["scheduler_environment"]["VLLM_MAX_NUM_SEQS"] = "99"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "run configuration mismatch"):
                _summarize(manifest, groups)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, groups = _fixture(Path(temporary))
            summary_path = groups["B"][0] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["configured_max_request_attempts"] = 3
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(
                ValueError,
                "run configuration mismatch for configured_max_request_attempts",
            ):
                _summarize(manifest, groups)

        baseline = {
            "task_flow_time_s": {name: 0.0 for name in ("mean", "p50", "p95", "max")},
            "task_makespan_s": 0.0,
            "request_latency_s": {name: 0.0 for name in ("mean", "p50", "p95", "max")},
            "mean_queue_time_s": 0.0,
            "instrumentation_wall_time_s": 0.0,
        }
        effect = _pair_effect(baseline, baseline, definition="zero")
        self.assertIsNone(effect["relative_reduction"]["task_makespan_s"])


if __name__ == "__main__":
    unittest.main()
