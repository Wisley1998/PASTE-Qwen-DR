from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPRODUCTION_ROOT / "scripts"
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_fixed_three_way_split import (  # noqa: E402
    build_fixed_bundle,
    file_sha256,
)
from paste_repro.mapper import URLRankMapper, load_artifact, save_artifact  # noqa: E402
from paste_repro.traces import (  # noqa: E402
    count_tool_calls,
    load_trace,
    transitions_from_sessions,
)
from prepare_fixed_workloads import (  # noqa: E402
    build_prepare_command,
    build_workload_manifest,
)


def _trace_events(index: int) -> list[dict]:
    chosen_rank = index % 5 + 1
    links = "\n".join(
        f"{rank}. [result {rank}](https://example.test/{index}/{rank})"
        for rank in range(1, 6)
    )
    return [
        {
            "event_type": "tool_call",
            "call_index": 0,
            "timestamp": 1.0,
            "tool_name": "search",
            "tool_args": {"query": [f"query-{index}"]},
        },
        {
            "event_type": "llm_call",
            "call_index": 1,
            "timestamp": 2.0,
            "total_time_ms": 500.0,
            "inference_time_ms": 500.0,
            "messages": [
                {
                    "role": "user",
                    "content": f"<tool_response>\n{links}\n</tool_response>",
                }
            ],
            "response": "visit decision",
        },
        {
            "event_type": "tool_call",
            "call_index": 1,
            "timestamp": 2.1,
            "tool_name": "visit",
            "tool_args": {"url": f"https://example.test/{index}/{chosen_rank}"},
        },
        {
            "event_type": "llm_call",
            "call_index": 2,
            "timestamp": 4.0,
            "total_time_ms": 100.0,
            "inference_time_ms": 100.0,
            "messages": [{"role": "user", "content": "completion"}],
            "response": "done",
        },
    ]


def _write_trace(path: Path, index: int) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in _trace_events(index)),
        encoding="utf-8",
    )


def _make_legacy_fixture(root: Path) -> tuple[Path, Path]:
    trace_directory = root / "traces"
    trace_directory.mkdir(parents=True)
    train_entries: list[dict[str, str]] = []
    held_out_entries: list[dict[str, str]] = []
    for index in range(100):
        is_train = index < 70
        if is_train:
            # The legacy train pool intentionally has 36 CJK and 34 non-CJK names.
            language = "中文" if index < 36 else "english"
            name = f"legacy_train_{index:03d}_{language}.jsonl"
        else:
            language = "中文" if index < 84 else "english"
            name = f"legacy_held_out_{index:03d}_{language}.jsonl"
        path = trace_directory / name
        _write_trace(path, index)
        entry = {"session_id": name, "sha256": file_sha256(path)}
        (train_entries if is_train else held_out_entries).append(entry)

    # Deliberately save an empty legacy mapper. The fixed builder must discard it
    # and learn non-empty counts only from the new calibration role.
    legacy_mapper = URLRankMapper().fit((), searches_seen=0)
    legacy_artifact = legacy_mapper.to_artifact(
        {
            "algorithm": "unit-test legacy 70/30 registry",
            "seed": "legacy",
            "train_ratio": 0.70,
            "train_sessions": train_entries,
            "held_out_sessions": held_out_entries,
        }
    )
    legacy_path = root / "legacy_mapper.json"
    save_artifact(legacy_path, legacy_artifact)
    return trace_directory, legacy_path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class FixedThreeWaySplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_mapper_is_calibration_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_directory, legacy_path = _make_legacy_fixture(root)
            first = build_fixed_bundle(
                legacy_artifact_path=legacy_path,
                trace_directory=trace_directory,
                output_root=root / "split-one",
                salt="fixed-salt",
            )
            second = build_fixed_bundle(
                legacy_artifact_path=legacy_path,
                trace_directory=trace_directory,
                output_root=root / "split-two",
                salt="fixed-salt",
            )

            self.assertEqual(first["split_manifest_sha256"], second["split_manifest_sha256"])
            self.assertEqual(first["mapper_artifact_sha256"], second["mapper_artifact_sha256"])
            manifest = _load_json(Path(first["split_manifest_path"]))
            self.assertEqual(manifest["counts"], {
                "total": 100,
                "calibration": 40,
                "tuning": 30,
                "final": 30,
            })
            role_ids = {
                role: {entry["session_id"] for entry in manifest[f"{role}_sessions"]}
                for role in ("calibration", "tuning", "final")
            }
            self.assertFalse(role_ids["calibration"] & role_ids["tuning"])
            self.assertFalse(role_ids["calibration"] & role_ids["final"])
            self.assertFalse(role_ids["tuning"] & role_ids["final"])
            self.assertEqual(len(set().union(*role_ids.values())), 100)
            self.assertTrue(all("legacy_held_out" in item for item in role_ids["tuning"]))
            self.assertEqual(
                manifest["selection"]["strata"],
                {
                    "cjk_filename": {"legacy_train": 36, "calibration": 21, "final": 15},
                    "non_cjk_filename": {"legacy_train": 34, "calibration": 19, "final": 15},
                },
            )

            mapper, artifact = load_artifact(Path(first["mapper_artifact_path"]))
            training_split = artifact["training_split"]
            self.assertEqual(
                {item["session_id"] for item in training_split["train_sessions"]},
                role_ids["calibration"],
            )
            self.assertEqual(
                {item["session_id"] for item in training_split["calibration_sessions"]},
                role_ids["calibration"],
            )
            self.assertEqual(
                {item["session_id"] for item in training_split["final_sessions"]},
                role_ids["final"],
            )
            calibration_sessions = tuple(
                load_trace(trace_directory / session_id)
                for session_id in sorted(role_ids["calibration"])
            )
            independently_fit = URLRankMapper().fit(
                transitions_from_sessions(calibration_sessions),
                searches_seen=count_tool_calls(calibration_sessions, "search"),
            )
            self.assertEqual(mapper.rank_counts, independently_fit.rank_counts)
            self.assertEqual(mapper.transitions_seen, independently_fit.transitions_seen)
            self.assertGreater(mapper.mapped_targets, 0)

            changed_salt = build_fixed_bundle(
                legacy_artifact_path=legacy_path,
                trace_directory=trace_directory,
                output_root=root / "split-three",
                salt="different-fixed-salt",
            )
            changed_manifest = _load_json(Path(changed_salt["split_manifest_path"]))
            changed_final_ids = {
                item["session_id"] for item in changed_manifest["final_sessions"]
            }
            self.assertNotEqual(role_ids["final"], changed_final_ids)

    def test_checksum_and_legacy_shape_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_directory, legacy_path = _make_legacy_fixture(root)
            first_trace = next(trace_directory.glob("*.jsonl"))
            first_trace.write_text(
                first_trace.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "trace checksum mismatch"):
                build_fixed_bundle(
                    legacy_artifact_path=legacy_path,
                    trace_directory=trace_directory,
                    output_root=root / "tampered-trace",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_directory, legacy_path = _make_legacy_fixture(root)
            corrupt = _load_json(legacy_path)
            corrupt["mapper"]["searches_seen"] = 999
            corrupt_path = root / "corrupt.json"
            corrupt_path.write_text(json.dumps(corrupt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
                build_fixed_bundle(
                    legacy_artifact_path=corrupt_path,
                    trace_directory=trace_directory,
                    output_root=root / "corrupt-artifact",
                )
            with self.assertRaisesRegex(ValueError, "used whole as tuning"):
                build_fixed_bundle(
                    legacy_artifact_path=legacy_path,
                    trace_directory=trace_directory,
                    output_root=root / "wrong-count",
                    tuning_count=29,
                )

    def test_prepare_commands_and_manifest_bind_matching_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_directory, legacy_path = _make_legacy_fixture(root)
            fixed = build_fixed_bundle(
                legacy_artifact_path=legacy_path,
                trace_directory=trace_directory,
                output_root=root / "splits",
            )
            mapper_path = Path(fixed["mapper_artifact_path"])
            runner = root / "runner.py"
            runner.write_text("# test runner\n", encoding="utf-8")
            none_command = build_prepare_command(
                python_executable="python-test",
                runner=runner,
                trace_directory=Path(fixed["roles"]["tuning"]["absolute_directory"]),
                trace_count=30,
                output_directory=root / "out-none",
                tokenizer="tokenizer-test",
                speedup=10.0,
                max_model_len=16384,
                output_cap=128,
                output_buffer=8,
                min_output_floor=128,
                overlap_mode="none",
                mapper_artifact=mapper_path,
                top_k=5,
                seed=7,
            )
            learned_command = build_prepare_command(
                python_executable="python-test",
                runner=runner,
                trace_directory=Path(fixed["roles"]["tuning"]["absolute_directory"]),
                trace_count=30,
                output_directory=root / "out-learned",
                tokenizer="tokenizer-test",
                speedup=10.0,
                max_model_len=16384,
                output_cap=128,
                output_buffer=8,
                min_output_floor=128,
                overlap_mode="learned",
                mapper_artifact=mapper_path,
                top_k=5,
                seed=7,
            )
            self.assertIn("--prepare-only", none_command)
            self.assertNotIn("--tool-prediction-model", none_command)
            self.assertEqual(
                none_command[none_command.index("--max-output-tokens-cap") + 1],
                "128",
            )
            self.assertEqual(none_command[none_command.index("--speedup") + 1], "10.0")
            self.assertEqual(
                learned_command[learned_command.index("--tool-prediction-model") + 1],
                str(mapper_path),
            )

            split_manifest = _load_json(Path(fixed["split_manifest_path"]))
            _, mapper_artifact = load_artifact(mapper_path)
            workload_root = root / "workloads"
            target_counts = {"calibration": 40, "tuning": 30, "final": 30}
            for role in ("calibration", "tuning", "final"):
                source_ids = [
                    entry["session_id"]
                    for entry in split_manifest[f"{role}_sessions"]
                ]
                for mode in ("none", "learned"):
                    output_directory = workload_root / role / mode
                    output_directory.mkdir(parents=True)
                    metadata = {
                        "source_trace_dir": fixed["roles"][role]["absolute_directory"],
                        "target_trace_count": target_counts[role],
                        "max_model_len": 16384,
                        "max_output_tokens_cap": 128,
                        "tool_overlap_mode": mode,
                    }
                    if mode == "learned":
                        metadata.update(
                            {
                                "tool_prediction_artifact_sha256": mapper_artifact[
                                    "artifact_sha256"
                                ],
                                "tool_prediction_top_k": 5,
                            }
                        )
                    workload = {
                        "meta": metadata,
                        "traces": [
                            {"source_trace": str(trace_directory / session_id), "requests": [{}]}
                            for session_id in source_ids
                        ],
                    }
                    summary = {"trace_count": target_counts[role], "tool_overlap_mode": mode}
                    (output_directory / "prepared_workload.json").write_text(
                        json.dumps(workload), encoding="utf-8"
                    )
                    (output_directory / "workload_summary.json").write_text(
                        json.dumps(summary), encoding="utf-8"
                    )

            output_manifest = workload_root / "manifest.json"
            manifest = build_workload_manifest(
                fixed_bundle=fixed,
                workload_root=workload_root,
                output_manifest=output_manifest,
                output_cap=128,
                speedup=10.0,
                max_model_len=16384,
                output_buffer=8,
                min_output_floor=128,
                top_k=5,
                target_counts=target_counts,
                seed=7,
            )
            for eval_role in ("tuning", "final"):
                cells = manifest["four_cell_inputs"][eval_role]
                self.assertEqual(
                    cells["joint_none"]["online_calibration_workload"],
                    manifest["workloads"]["calibration"]["none"]["prepared_workload"],
                )
                self.assertEqual(
                    cells["joint_learned"]["online_calibration_workload"],
                    manifest["workloads"]["calibration"]["learned"][
                        "prepared_workload"
                    ],
                )
                self.assertIn(
                    f"{eval_role}/learned",
                    cells["joint_learned"]["evaluation_workload"],
                )
            self.assertEqual(manifest["parameters"]["max_output_tokens_cap"], 128)
            self.assertEqual(manifest["parameters"]["speedup"], 10.0)


if __name__ == "__main__":
    unittest.main()
