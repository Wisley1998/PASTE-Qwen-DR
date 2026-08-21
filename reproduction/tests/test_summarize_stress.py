from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_ROOT = REPRODUCTION_ROOT / "scripts"
RUNNER_SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
for import_root in (REPRODUCTION_ROOT, SCRIPT_ROOT, RUNNER_SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_stress_duplicate_workloads import build_stress_bundle  # noqa: E402
from online_session_predictor import OnlineSessionPredictor  # noqa: E402
from reproduction.tests.test_build_stress_duplicate_workloads import (  # noqa: E402
    _build_heldout_fixture,
)
from summarize_four_cell import (  # noqa: E402
    CELL_SPECS,
    canonical_sha256,
    file_sha256,
    load_fixed_manifest,
    parse_args as parse_four_args,
    summarize_four_cell,
)
from summarize_paired_ad import (  # noqa: E402
    parse_args as parse_paired_args,
    summarize_pairs,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resign_manifest(path: Path, payload: dict) -> None:
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(payload)
    _write_json(path, payload)


def _refresh_stress_record(
    manifest_path: Path,
    manifest: dict,
    *,
    mode: str,
    workload: dict,
) -> None:
    record = manifest["workloads"]["stress"][mode]
    workload_path = (manifest_path.parent / record["prepared_workload"]).resolve()
    _write_json(workload_path, workload)
    sequence = [Path(trace["source_trace"]).name for trace in workload["traces"]]
    load_rows = [
        {
            "trace_id": trace["trace_id"],
            "source_session": Path(trace["source_trace"]).name,
            "variant_index": trace["variant_index"],
            "duplicated": trace["duplicated"],
            "prefix_char": trace["prefix_char"],
        }
        for trace in workload["traces"]
    ]
    record["prepared_workload_sha256"] = file_sha256(workload_path)
    record["source_sequence_sha256"] = canonical_sha256(sequence)
    record["source_set_sha256"] = canonical_sha256(sorted(set(sequence)))
    record["unique_source_session_count"] = len(set(sequence))
    record["load_identity_sha256"] = canonical_sha256(load_rows)


def _request_id(metadata: dict) -> str:
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8").hex()
    return f"schedx{encoded}z"


def _write_stress_run(
    root: Path,
    *,
    cell: str,
    workload_path: Path,
    calibration_path: Path,
) -> Path:
    spec = CELL_SPECS[cell]
    mode = spec["tool_overlap_mode"]
    workload = _load(workload_path)
    trace_count = len(workload["traces"])
    predictor = OnlineSessionPredictor.from_workload(calibration_path)
    speedup = 10.0
    events: list[dict] = []
    completion_by_trace: dict[str, float] = {}
    for trace_number, trace in enumerate(workload["traces"]):
        observed_waits: list[float] = []
        base_completion = 10.0 + trace_number * 0.01
        for request_index, request in enumerate(trace["requests"]):
            call_index = int(request["call_index"])
            if call_index > 0:
                observed_waits.append(float(request["wait_after_prev_s"]))
            prediction = predictor.predict(
                current_call_index=call_index,
                past_tool_waits_s=observed_waits,
            )
            predicted_output = int(request["max_tokens"])
            metadata = {
                "t": trace["trace_id"],
                "c": call_index,
                "i": request_index,
                "n": request_index + 1 + prediction.remaining_calls,
                "rc": prediction.remaining_calls,
                "nw": prediction.next_tool_wait_s / speedup,
                "nwc": predictor.next_tool_wait_reliability,
                "rtw": prediction.remaining_tool_wait_s / speedup,
                "pt": request["prompt_tokens"],
                "mt": request["max_tokens"],
                "ms": "online",
                "po": predicted_output,
            }
            if prediction.remaining_calls > 0:
                metadata["npo"] = predicted_output
            end_offset = base_completion + request_index
            completion_by_trace[trace["trace_id"]] = end_offset
            event = {
                "trace_id": trace["trace_id"],
                "source_trace": trace["source_trace"],
                "duplicated": trace["duplicated"],
                "prefix_char": trace["prefix_char"],
                "call_index": call_index,
                "prompt_tokens": request["prompt_tokens"],
                "target_output_tokens": request["target_output_tokens"],
                "max_tokens": request["max_tokens"],
                "truncated": request["truncated"],
                "metadata_source": "online",
                "request_id": _request_id(metadata),
                "scheduled_remaining_calls_after": metadata["rc"],
                "scheduled_total_calls": metadata["n"],
                "scheduled_nw": metadata["nw"],
                "scheduled_nw_reliability": metadata["nwc"],
                "scheduled_rtw": metadata["rtw"],
                "nw_source": "predicted",
                "oracle_next_tool_wait_s": None,
                "oracle_remaining_tool_wait_s": None,
                "oracle_remaining_calls_after": None,
                "oracle_total_calls": None,
                "po_predicted": predicted_output,
                "po_actual": predicted_output,
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
                    request["wait_after_prev_s"]
                    if call_index == 0
                    else request["wait_after_prev_s"] / speedup
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

    prediction_summary = {
        "candidate_count": sum(
            int(request.get("tool_prediction_candidate_count", 0))
            for trace in workload["traces"]
            for request in trace["requests"]
        ),
        "exact_hits": sum(
            int(request.get("tool_prediction_exact_hits", 0))
            for trace in workload["traces"]
            for request in trace["requests"]
        ),
        "waste": sum(
            int(request.get("tool_prediction_waste", 0))
            for trace in workload["traces"]
            for request in trace["requests"]
        ),
        "artifact_sha256": workload["meta"].get(
            "tool_prediction_artifact_sha256"
        ),
        "top_k": workload["meta"].get("tool_prediction_top_k"),
    }
    run = root / f"stress_{cell.lower()}"
    run.mkdir(parents=True)
    shutil.copyfile(workload_path, run / "prepared_workload.json")
    summary = {
        "speedup": speedup,
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
        "scheduler_calibration_workload": str(calibration_path.resolve()),
        "scheduler_environment": {
            "VLLM_SCHED_POLICY": spec["policy"],
            "VLLM_SCHED_PRED_OUT_ENABLE": "1",
            "VLLM_MAX_NUM_SEQS": "64",
        },
        "max_active_traces": trace_count,
        "tool_wait_mode": "sleep",
        "avg_queue_time_s": 0.5,
        "experiment_wall_time_s": max(completion_by_trace.values()) + 0.5,
        "workload": {
            "trace_count": trace_count,
            "request_count": len(events),
            "tool_overlap_mode": mode,
            **({"tool_prediction": prediction_summary} if mode == "learned" else {}),
        },
    }
    _write_json(run / "summary.json", summary)
    (run / "request_events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    log = "vLLM API server version 0.10.1\n"
    if spec["policy"] != "fcfs":
        log += (
            "[sched_policy_patch] installed policy=online_joint_pacer_v2 "
            "v0=True v1=True\n"
            "[sched_policy_patch:joint] pending_returns=2 running=4\n"
        )
    (run / "server.log").write_text(log, encoding="utf-8")
    return run


def _stress_fixture(
    root: Path,
    *,
    with_runs: bool,
    load_instance_count: int = 120,
) -> tuple[Path, dict[str, Path]]:
    heldout_manifest, tokenizer = _build_heldout_fixture(root)
    stress_manifest = root / f"manifest_stress{load_instance_count}.json"
    payload = build_stress_bundle(
        manifest_path=heldout_manifest,
        output_root=root / f"stress{load_instance_count}",
        output_manifest=stress_manifest,
        tokenizer=tokenizer,
        load_instance_count=load_instance_count,
    )
    if not with_runs:
        return stress_manifest, {}
    run_paths: dict[str, Path] = {}
    for cell, spec in CELL_SPECS.items():
        mode = spec["tool_overlap_mode"]
        workload_path = (
            stress_manifest.parent
            / payload["workloads"]["stress"][mode]["prepared_workload"]
        ).resolve()
        calibration_path = (
            stress_manifest.parent
            / payload["workloads"]["calibration"][mode]["prepared_workload"]
        ).resolve()
        run_paths[cell] = _write_stress_run(
            root,
            cell=cell,
            workload_path=workload_path,
            calibration_path=calibration_path,
        )
    return stress_manifest, run_paths


class StressSummaryTests(unittest.TestCase):
    def test_stress_manifest_four_cell_and_paired_outputs_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, runs = _stress_fixture(Path(temporary), with_runs=True)
            verified = load_fixed_manifest(manifest, "stress")
            self.assertEqual(verified["load_instance_count"], 120)
            self.assertEqual(verified["independent_source_session_count"], 60)
            self.assertEqual(verified["instances_per_source"], 2)
            self.assertTrue(verified["duplicates_are_not_independent"])
            self.assertFalse(verified["is_final_evaluation"])
            self.assertEqual(verified["prefix_marker_mode"], "break_prefix")

            four = summarize_four_cell(
                {cell: [path] for cell, path in runs.items()},
                manifest_path=manifest,
                role="stress",
            )
            invariants = four["comparison_invariants"]
            self.assertEqual(
                four["status"],
                "stress120_four_cell_load_sensitivity_not_independent_not_final",
            )
            self.assertEqual(invariants["trace_count"], 120)
            self.assertEqual(invariants["source_session_count"], 60)
            self.assertEqual(invariants["independent_source_session_count"], 60)
            self.assertTrue(invariants["duplicates_are_not_independent"])
            self.assertFalse(invariants["is_final_evaluation"])
            self.assertIn("not independent", four["interpretation"])

            paired = summarize_pairs(
                [(runs["A"], runs["D"])],
                manifest_path=manifest,
                role="stress",
            )
            paired_invariants = paired["comparison_invariants"]
            self.assertEqual(
                paired["status"],
                "paired_stress120_ad_load_sensitivity_not_independent_not_final",
            )
            self.assertEqual(paired_invariants["load_instance_count"], 120)
            self.assertEqual(paired_invariants["source_session_count"], 60)
            source_aggregate = paired["aggregate"]["paired_task_flow"]
            self.assertEqual(source_aggregate["independent_session_count"], 60)
            self.assertEqual(source_aggregate["load_instance_count_per_replicate"], 120)
            self.assertEqual(source_aggregate["raw_paired_load_observation_count"], 120)
            self.assertTrue(source_aggregate["duplicates_are_not_independent"])
            self.assertEqual(len(source_aggregate["sessions"]), 60)
            bootstrap = source_aggregate[
                "independent_source_mean_bootstrap_95_ci_s"
            ]
            self.assertEqual(bootstrap["sample_size"], 60)
            self.assertEqual(
                bootstrap["sampling_unit"], "independent_source_session_mean"
            )
            self.assertEqual(
                source_aggregate["effective_independent_sample_size"], 60
            )
            self.assertTrue(
                source_aggregate[
                    "duplicates_and_replicates_do_not_increase_independent_sample_size"
                ]
            )

    def test_stress180_manifest_validates_and_four_cell_summary_is_dynamic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, runs = _stress_fixture(
                Path(temporary),
                with_runs=True,
                load_instance_count=180,
            )
            verified = load_fixed_manifest(manifest, "stress")
            self.assertEqual(verified["load_instance_count"], 180)
            self.assertEqual(verified["independent_source_session_count"], 60)
            self.assertEqual(verified["instances_per_source"], 3)
            self.assertEqual(verified["minimum_instances_per_source"], 3)
            self.assertEqual(verified["maximum_instances_per_source"], 3)
            self.assertEqual(verified["sources_with_one_extra_instance"], 0)
            self.assertTrue(verified["source_instances_are_balanced"])
            self.assertEqual(
                verified["evidence_role"],
                "stress180_load_sensitivity_not_independent_not_final",
            )

            four = summarize_four_cell(
                {cell: [path] for cell, path in runs.items()},
                manifest_path=manifest,
                role="stress",
            )
            self.assertEqual(
                four["status"],
                "stress180_four_cell_load_sensitivity_not_independent_not_final",
            )
            invariants = four["comparison_invariants"]
            self.assertEqual(invariants["trace_count"], 180)
            self.assertEqual(invariants["load_instance_count"], 180)
            self.assertEqual(invariants["source_session_count"], 60)
            self.assertEqual(invariants["instances_per_source"], 3)
            self.assertEqual(invariants["minimum_instances_per_source"], 3)
            self.assertEqual(invariants["maximum_instances_per_source"], 3)
            self.assertEqual(invariants["sources_with_one_extra_instance"], 0)
            self.assertTrue(invariants["source_instances_are_balanced"])
            self.assertIn("180 load instances", four["interpretation"])
            self.assertIn("3 deterministic instances per source", four["interpretation"])

            paired = summarize_pairs(
                [(runs["A"], runs["D"])],
                manifest_path=manifest,
                role="stress",
            )
            self.assertEqual(
                paired["status"],
                "paired_stress180_ad_load_sensitivity_not_independent_not_final",
            )
            paired_invariants = paired["comparison_invariants"]
            self.assertEqual(paired_invariants["load_instance_count"], 180)
            self.assertEqual(paired_invariants["instances_per_source"], 3)
            self.assertEqual(paired_invariants["minimum_instances_per_source"], 3)
            self.assertEqual(paired_invariants["maximum_instances_per_source"], 3)
            source_aggregate = paired["aggregate"]["paired_task_flow"]
            self.assertEqual(source_aggregate["independent_session_count"], 60)
            self.assertEqual(source_aggregate["load_instance_count_per_replicate"], 180)
            self.assertEqual(source_aggregate["raw_paired_load_observation_count"], 180)
            self.assertEqual(source_aggregate["instances_per_source"], 3)
            self.assertEqual(
                source_aggregate["independent_source_mean_bootstrap_95_ci_s"][
                    "sample_size"
                ],
                60,
            )
            self.assertTrue(
                all(
                    row["load_instances_per_replicate"] == 3
                    for row in source_aggregate["sessions"]
                )
            )
            replicate_flow = paired["replicates"][0]["paired_task_flow"]
            self.assertEqual(replicate_flow["load_instance_count"], 180)
            self.assertEqual(replicate_flow["instances_per_source"], 3)
            self.assertTrue(
                all(
                    row["load_instance_count"] == 3
                    for row in replicate_flow["source_sessions"]
                )
            )
            self.assertIn("Stress180", paired["interpretation"])

    def test_stress_manifest_semantic_tampering_fails_closed(self) -> None:
        cases = (
            ("parent_checksum", "parent heldout60 manifest checksum mismatch"),
            ("not_final_guard", "invalid guard: stress_is_not_final"),
            ("prefix_mode", "metadata mismatch: prefix_marker_mode"),
            ("marker_position", "exact leading system message"),
            ("source_leak", "sessions do not match split role"),
            ("source_multiplicity", "source order is not deterministic"),
        )
        for case, expected_error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, _ = _stress_fixture(root, with_runs=False)
                payload = _load(manifest)
                if case == "parent_checksum":
                    payload["stress_derived_from_manifest_sha256"] = "0" * 64
                elif case == "not_final_guard":
                    payload["contamination_guards"]["stress_is_not_final"] = False
                else:
                    none_record = payload["workloads"]["stress"]["none"]
                    workload_path = (
                        manifest.parent / none_record["prepared_workload"]
                    ).resolve()
                    workload = _load(workload_path)
                    if case == "prefix_mode":
                        workload["meta"]["prefix_marker_mode"] = "preserve_prefix"
                        _refresh_stress_record(
                            manifest, payload, mode="none", workload=workload
                        )
                    elif case == "marker_position":
                        duplicate = workload["traces"][60]
                        for request in duplicate["requests"]:
                            request["messages"].append(request["messages"].pop(0))
                        _refresh_stress_record(
                            manifest, payload, mode="none", workload=workload
                        )
                    elif case == "source_leak":
                        split = _load(
                            (manifest.parent / payload["fixed_split_manifest"]).resolve()
                        )
                        calibration_id = split["calibration_sessions"][0]["session_id"]
                        calibration_path = next(
                            Path(trace["source_trace"]).resolve()
                            for trace in _load(
                                (
                                    manifest.parent
                                    / payload["workloads"]["calibration"]["none"][
                                        "prepared_workload"
                                    ]
                                ).resolve()
                            )["traces"]
                            if Path(trace["source_trace"]).name == calibration_id
                        )
                        for mode in ("none", "learned"):
                            record = payload["workloads"]["stress"][mode]
                            mode_workload = _load(
                                (manifest.parent / record["prepared_workload"]).resolve()
                            )
                            mode_workload["traces"][0]["source_trace"] = str(
                                calibration_path
                            )
                            _refresh_stress_record(
                                manifest,
                                payload,
                                mode=mode,
                                workload=mode_workload,
                            )
                        payload["stress_definition"]["load_identity_sha256"] = payload[
                            "workloads"
                        ]["stress"]["none"]["load_identity_sha256"]
                    else:
                        for mode in ("none", "learned"):
                            record = payload["workloads"]["stress"][mode]
                            mode_workload = _load(
                                (manifest.parent / record["prepared_workload"]).resolve()
                            )
                            mode_workload["traces"][60]["source_trace"] = (
                                mode_workload["traces"][61]["source_trace"]
                            )
                            _refresh_stress_record(
                                manifest,
                                payload,
                                mode=mode,
                                workload=mode_workload,
                            )
                        payload["stress_definition"]["load_identity_sha256"] = payload[
                            "workloads"
                        ]["stress"]["none"]["load_identity_sha256"]
                _resign_manifest(manifest, payload)
                with self.assertRaisesRegex(ValueError, expected_error):
                    load_fixed_manifest(manifest, "stress")

    def test_stress_cli_choices(self) -> None:
        four = parse_four_args(
            [
                "--manifest",
                "manifest.json",
                "--role",
                "stress",
                "--a",
                "a",
                "--b",
                "b",
                "--c",
                "c",
                "--d",
                "d",
            ]
        )
        paired = parse_paired_args(
            [
                "--manifest",
                "manifest.json",
                "--role",
                "stress",
                "--pair",
                "a",
                "d",
            ]
        )
        self.assertEqual(four.role, "stress")
        self.assertEqual(paired.role, "stress")

    def test_run_four_cell_accepts_stress_and_propagates_validator_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_manifest = root / "invalid.json"
            _write_json(invalid_manifest, {})
            environment = os.environ.copy()
            environment.update(
                {
                    "PASTE_ENV_PREFIX": str(Path(sys.executable).resolve().parents[1]),
                    "PASTE_FIXED_WORKLOAD_MANIFEST": str(invalid_manifest),
                    "PASTE_SERVER_URL": "http://127.0.0.1:1",
                    "PASTE_RUN_ROOT": str(root / "runs"),
                }
            )
            completed = subprocess.run(
                ["bash", str(SCRIPT_ROOT / "run_four_cell.sh"), "stress"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn(
                "fixed workload manifest validation failed", completed.stderr
            )
            self.assertNotIn("Usage: run_four_cell.sh", completed.stderr)


if __name__ == "__main__":
    unittest.main()
