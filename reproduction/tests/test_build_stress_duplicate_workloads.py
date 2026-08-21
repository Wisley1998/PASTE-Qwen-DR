from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_ROOT = REPRODUCTION_ROOT / "scripts"
RUNNER_SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
for import_root in (REPRODUCTION_ROOT, SCRIPT_ROOT, RUNNER_SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import build_stress_duplicate_workloads as stress_builder  # noqa: E402
from build_fixed_three_way_split import (  # noqa: E402
    build_fixed_bundle,
    canonical_sha256,
    file_sha256,
)
from build_heldout_union_workloads import build_heldout_union  # noqa: E402
from build_stress_duplicate_workloads import (  # noqa: E402
    EVIDENCE_ROLE,
    _normalize_load_instance_count,
    _replication_metadata,
    build_stress_bundle,
)
from paste_repro.mapper import write_json_atomic  # noqa: E402
from prepare_fixed_workloads import build_workload_manifest  # noqa: E402
from reproduction.tests.test_fixed_three_way_split import (  # noqa: E402
    _make_legacy_fixture,
)
from trace_experiment_lib import (  # noqa: E402
    _build_chat_tokens,
    duplicate_variant_marker,
    prepare_trace_workload,
    summarize_workload,
)


class _FakeTokenizer:
    """Deterministic tokenizer that also exercises the truncation path."""

    @staticmethod
    def _tokens(text: str) -> list[int]:
        return list(range(max(1, (len(text.encode("utf-8")) + 3) // 4)))

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if not tokenize or not add_generation_prompt:
            raise AssertionError("test tokenizer requires generation tokenization")
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return self._tokens(rendered + "<assistant>")

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("test tokenizer does not add special tokens")
        return self._tokens(str(text))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(manifest_path: Path, relative: str) -> Path:
    return (manifest_path.parent / relative).resolve()


def _build_heldout_fixture(root: Path) -> tuple[Path, _FakeTokenizer]:
    tokenizer = _FakeTokenizer()
    trace_directory, legacy_path = _make_legacy_fixture(root)
    fixed = build_fixed_bundle(
        legacy_artifact_path=legacy_path,
        trace_directory=trace_directory,
        output_root=root / "splits",
        salt="stress-builder-test-salt",
    )
    mapper_path = Path(str(fixed["mapper_artifact_path"]))
    workload_root = root / "fixed-workloads"
    parameters = {
        "max_model_len": 96,
        "max_output_tokens_cap": 32,
        "min_output_tokens_floor": 16,
        "output_token_buffer": 4,
        "tool_prediction_top_k": 3,
        "seed": 20260417,
    }
    target_counts = {"calibration": 40, "tuning": 30, "final": 30}
    for role in ("calibration", "tuning", "final"):
        for mode in ("none", "learned"):
            workload = prepare_trace_workload(
                trace_dir=Path(fixed["roles"][role]["absolute_directory"]),
                tokenizer=tokenizer,
                target_trace_count=target_counts[role],
                max_model_len=parameters["max_model_len"],
                max_output_tokens_cap=parameters["max_output_tokens_cap"],
                min_output_tokens_floor=parameters["min_output_tokens_floor"],
                output_token_buffer=parameters["output_token_buffer"],
                duplicate_seed=parameters["seed"],
                tool_overlap_mode=mode,
                tool_prediction_model=mapper_path if mode == "learned" else None,
                tool_prediction_top_k=parameters["tool_prediction_top_k"],
            )
            output = workload_root / role / mode
            write_json_atomic(output / "prepared_workload.json", workload)
            write_json_atomic(output / "workload_summary.json", summarize_workload(workload))

    fixed_manifest_path = root / "manifest.json"
    fixed_manifest = build_workload_manifest(
        fixed_bundle=fixed,
        workload_root=workload_root,
        output_manifest=fixed_manifest_path,
        output_cap=parameters["max_output_tokens_cap"],
        speedup=10.0,
        max_model_len=parameters["max_model_len"],
        output_buffer=parameters["output_token_buffer"],
        min_output_floor=parameters["min_output_tokens_floor"],
        top_k=parameters["tool_prediction_top_k"],
        target_counts=target_counts,
        seed=parameters["seed"],
    )
    write_json_atomic(fixed_manifest_path, fixed_manifest)
    heldout_manifest_path = root / "manifest_heldout60.json"
    build_heldout_union(
        manifest_path=fixed_manifest_path,
        output_root=root / "heldout60",
        output_manifest=heldout_manifest_path,
    )
    return heldout_manifest_path, tokenizer


class StressDuplicateWorkloadTests(unittest.TestCase):
    def test_build_is_checksummed_deterministic_and_breaks_duplicate_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout_manifest_path, tokenizer = _build_heldout_fixture(root)
            output_manifest = root / "manifest_stress120.json"
            with mock.patch.object(
                stress_builder,
                "_build_chat_tokens",
                wraps=stress_builder._build_chat_tokens,
            ) as prompt_token_rebuild:
                first = build_stress_bundle(
                    manifest_path=heldout_manifest_path,
                    output_root=root / "stress120",
                    output_manifest=output_manifest,
                    tokenizer=tokenizer,
                )
                prompt_token_rebuild_count = prompt_token_rebuild.call_count

            supplied = first["manifest_sha256"]
            without_checksum = copy.deepcopy(first)
            without_checksum.pop("manifest_sha256")
            self.assertEqual(supplied, canonical_sha256(without_checksum))
            definition = first["stress_definition"]
            self.assertEqual(definition["evidence_role"], EVIDENCE_ROLE)
            self.assertEqual(definition["unique_source_session_count"], 60)
            self.assertEqual(definition["load_instance_count"], 120)
            self.assertEqual(definition["independent_sample_count"], 60)
            self.assertTrue(definition["duplicates_are_not_independent"])
            self.assertFalse(definition["is_final_evaluation"])
            self.assertTrue(definition["calibration_excluded"])
            self.assertFalse(definition["mapper_retrained"])
            self.assertEqual(definition["prefix_marker_mode"], "break_prefix")
            self.assertEqual(
                definition["source_sessions_sha256"],
                canonical_sha256(definition["source_sessions"]),
            )

            workloads: dict[str, dict] = {}
            file_checksums: dict[str, str] = {}
            for mode in ("none", "learned"):
                record = first["workloads"]["stress"][mode]
                path = _resolve(output_manifest, record["prepared_workload"])
                file_checksums[mode] = file_sha256(path)
                self.assertEqual(file_checksums[mode], record["prepared_workload_sha256"])
                workload = _load(path)
                workloads[mode] = workload
                self.assertEqual(workload["meta"]["prefix_marker_mode"], "break_prefix")
                self.assertEqual(workload["meta"]["unique_source_session_count"], 60)
                self.assertEqual(workload["meta"]["load_instance_count"], 120)
                self.assertEqual(workload["meta"]["independent_sample_count"], 60)
                self.assertTrue(workload["meta"]["duplicates_are_not_independent"])
                self.assertTrue(workload["meta"]["stress_is_not_final"])
                traces = workload["traces"]
                self.assertEqual(len(traces), 120)
                self.assertEqual(len({trace["trace_id"] for trace in traces}), 120)
                self.assertEqual(len({trace["variant_index"] for trace in traces}), 120)
                sources = Counter(Path(trace["source_trace"]).name for trace in traces)
                self.assertEqual(set(sources.values()), {2})
                originals = Counter(
                    Path(trace["source_trace"]).name
                    for trace in traces
                    if not trace["duplicated"]
                )
                duplicates = Counter(
                    Path(trace["source_trace"]).name
                    for trace in traces
                    if trace["duplicated"]
                )
                self.assertEqual(set(originals.values()), {1})
                self.assertEqual(set(duplicates.values()), {1})
                duplicate_traces = [trace for trace in traces if trace["duplicated"]]
                self.assertEqual(
                    {trace["prefix_char"] for trace in duplicate_traces},
                    {duplicate_variant_marker(index) for index in range(60)},
                )
                for trace in duplicate_traces:
                    marker = trace["prefix_char"]
                    for request in trace["requests"]:
                        self.assertEqual(
                            request["messages"][0],
                            {"role": "system", "content": marker},
                        )
                        self.assertEqual(
                            request["prompt_tokens"],
                            _build_chat_tokens(tokenizer, request["messages"]),
                        )
                        self.assertLessEqual(
                            request["prompt_tokens"] + request["max_tokens"],
                            workload["meta"]["max_model_len"],
                        )

            self.assertEqual(
                first["workloads"]["stress"]["none"]["load_identity_sha256"],
                first["workloads"]["stress"]["learned"]["load_identity_sha256"],
            )
            self.assertEqual(
                prompt_token_rebuild_count,
                sum(len(trace["requests"]) for trace in workloads["none"]["traces"]),
            )
            self.assertEqual(
                first["workloads"]["stress"]["learned"]["mapper_artifact_sha256"],
                first["calibration_only_mapper_sha256"],
            )
            calibration_ids = {
                item["session_id"]
                for item in _load(
                    _resolve(output_manifest, first["fixed_split_manifest"])
                )["calibration_sessions"]
            }
            stress_ids = {
                Path(trace["source_trace"]).name
                for trace in workloads["none"]["traces"]
            }
            self.assertFalse(calibration_ids & stress_ids)

            second = build_stress_bundle(
                manifest_path=heldout_manifest_path,
                output_root=root / "stress120",
                output_manifest=output_manifest,
                tokenizer=tokenizer,
                load_instance_count=120,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            for mode in ("none", "learned"):
                record = second["workloads"]["stress"][mode]
                self.assertEqual(
                    file_checksums[mode],
                    file_sha256(_resolve(output_manifest, record["prepared_workload"])),
                )

    def test_stress180_balances_three_instances_and_unique_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout_manifest_path, tokenizer = _build_heldout_fixture(root)
            output_manifest = root / "manifest_stress180.json"
            result = build_stress_bundle(
                manifest_path=heldout_manifest_path,
                output_root=root / "stress180",
                output_manifest=output_manifest,
                tokenizer=tokenizer,
                load_instance_count=180,
            )

            definition = result["stress_definition"]
            self.assertEqual(definition["load_instance_count"], 180)
            self.assertEqual(definition["instances_per_source"], 3)
            self.assertEqual(definition["minimum_instances_per_source"], 3)
            self.assertEqual(definition["maximum_instances_per_source"], 3)
            self.assertEqual(definition["sources_with_one_extra_instance"], 0)
            self.assertTrue(definition["source_instances_are_balanced"])
            self.assertEqual(
                definition["evidence_role"],
                "stress180_load_sensitivity_not_independent_not_final",
            )
            self.assertEqual(
                result["parameters"]["target_trace_counts"]["stress"], 180
            )

            workloads: dict[str, dict] = {}
            for mode in ("none", "learned"):
                record = result["workloads"]["stress"][mode]
                self.assertEqual(record["trace_count"], 180)
                self.assertEqual(record["instances_per_source"], 3)
                workload = _load(
                    _resolve(output_manifest, record["prepared_workload"])
                )
                workloads[mode] = workload
                traces = workload["traces"]
                self.assertEqual(len(traces), 180)
                source_counts = Counter(
                    Path(trace["source_trace"]).name for trace in traces
                )
                self.assertEqual(set(source_counts.values()), {3})
                originals = [trace for trace in traces if not trace["duplicated"]]
                duplicates = [trace for trace in traces if trace["duplicated"]]
                self.assertEqual(len(originals), 60)
                self.assertEqual(len(duplicates), 120)
                duplicate_source_sequence = [
                    Path(trace["source_trace"]).name for trace in duplicates
                ]
                self.assertEqual(
                    duplicate_source_sequence[:60],
                    duplicate_source_sequence[60:],
                )
                self.assertEqual(
                    {trace["prefix_char"] for trace in duplicates},
                    {duplicate_variant_marker(index) for index in range(120)},
                )
                for trace in duplicates:
                    marker = trace["prefix_char"]
                    self.assertTrue(
                        all(
                            request["messages"][0]
                            == {"role": "system", "content": marker}
                            for request in trace["requests"]
                        )
                    )

            self.assertEqual(
                result["workloads"]["stress"]["none"]["load_identity_sha256"],
                result["workloads"]["stress"]["learned"]["load_identity_sha256"],
            )
            self.assertEqual(
                [
                    (trace["source_trace"], trace["prefix_char"])
                    for trace in workloads["none"]["traces"]
                ],
                [
                    (trace["source_trace"], trace["prefix_char"])
                    for trace in workloads["learned"]["traces"]
                ],
            )

    def test_stress240_has_exactly_four_instances_per_heldout_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout_manifest_path, tokenizer = _build_heldout_fixture(root)
            output_manifest = root / "manifest_stress240.json"
            result = build_stress_bundle(
                manifest_path=heldout_manifest_path,
                output_root=root / "stress240",
                output_manifest=output_manifest,
                tokenizer=tokenizer,
                load_instance_count=240,
            )

            definition = result["stress_definition"]
            self.assertEqual(definition["load_instance_count"], 240)
            self.assertEqual(definition["unique_source_session_count"], 60)
            self.assertEqual(definition["independent_sample_count"], 60)
            self.assertEqual(definition["instances_per_source"], 4)
            self.assertEqual(definition["minimum_instances_per_source"], 4)
            self.assertEqual(definition["maximum_instances_per_source"], 4)
            self.assertEqual(definition["sources_with_one_extra_instance"], 0)
            self.assertTrue(definition["source_instances_are_balanced"])
            self.assertEqual(
                definition["evidence_role"],
                "stress240_load_sensitivity_not_independent_not_final",
            )

            for mode in ("none", "learned"):
                record = result["workloads"]["stress"][mode]
                workload = _load(
                    _resolve(output_manifest, record["prepared_workload"])
                )
                traces = workload["traces"]
                source_counts = Counter(
                    Path(trace["source_trace"]).name for trace in traces
                )
                self.assertEqual(len(traces), 240)
                self.assertEqual(len(source_counts), 60)
                self.assertEqual(set(source_counts.values()), {4})
                duplicates = [trace for trace in traces if trace["duplicated"]]
                self.assertEqual(len(duplicates), 180)
                self.assertEqual(
                    len({trace["prefix_char"] for trace in duplicates}),
                    180,
                )

    def test_stress300_has_exactly_five_instances_per_heldout_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout_manifest_path, tokenizer = _build_heldout_fixture(root)
            output_manifest = root / "manifest_stress300.json"
            result = build_stress_bundle(
                manifest_path=heldout_manifest_path,
                output_root=root / "stress300",
                output_manifest=output_manifest,
                tokenizer=tokenizer,
                load_instance_count=300,
            )

            definition = result["stress_definition"]
            self.assertEqual(definition["load_instance_count"], 300)
            self.assertEqual(definition["unique_source_session_count"], 60)
            self.assertEqual(definition["independent_sample_count"], 60)
            self.assertEqual(definition["instances_per_source"], 5)
            self.assertEqual(definition["minimum_instances_per_source"], 5)
            self.assertEqual(definition["maximum_instances_per_source"], 5)
            self.assertEqual(definition["sources_with_one_extra_instance"], 0)
            self.assertTrue(definition["source_instances_are_balanced"])
            self.assertEqual(
                definition["evidence_role"],
                "stress300_load_sensitivity_not_independent_not_final",
            )

            for mode in ("none", "learned"):
                record = result["workloads"]["stress"][mode]
                workload = _load(
                    _resolve(output_manifest, record["prepared_workload"])
                )
                traces = workload["traces"]
                source_counts = Counter(
                    Path(trace["source_trace"]).name for trace in traces
                )
                self.assertEqual(len(traces), 300)
                self.assertEqual(len(source_counts), 60)
                self.assertEqual(set(source_counts.values()), {5})
                duplicates = [trace for trace in traces if trace["duplicated"]]
                self.assertEqual(len(duplicates), 240)
                self.assertEqual(
                    len({trace["prefix_char"] for trace in duplicates}),
                    240,
                )

    def test_arbitrary_count_metadata_is_balanced_and_rejects_source_omission(
        self,
    ) -> None:
        self.assertEqual(_replication_metadata(120), {"instances_per_source": 2})
        self.assertEqual(
            _replication_metadata(181),
            {
                "instances_per_source": None,
                "minimum_instances_per_source": 3,
                "maximum_instances_per_source": 4,
                "sources_with_one_extra_instance": 1,
                "source_instances_are_balanced": True,
            },
        )
        with self.assertRaisesRegex(ValueError, "at least 60"):
            _normalize_load_instance_count(59)

    def test_learned_static_tampering_blocks_token_validation_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout_manifest_path, tokenizer = _build_heldout_fixture(root)
            original_prepare = stress_builder._prepare_stress_mode

            for tamper_kind in ("messages", "prompt_tokens"):
                with self.subTest(tamper_kind=tamper_kind):
                    def tampered_prepare(**kwargs):
                        workload = original_prepare(**kwargs)
                        if kwargs["mode"] == "learned":
                            request = workload["traces"][0]["requests"][0]
                            if tamper_kind == "messages":
                                request["messages"][0]["content"] += " tampered"
                            else:
                                request["prompt_tokens"] += 1
                        return workload

                    with mock.patch.object(
                        stress_builder,
                        "_prepare_stress_mode",
                        side_effect=tampered_prepare,
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "static request identities differ"
                        ):
                            build_stress_bundle(
                                manifest_path=heldout_manifest_path,
                                output_root=root / f"tampered-{tamper_kind}",
                                output_manifest=(
                                    root / f"manifest-tampered-{tamper_kind}.json"
                                ),
                                tokenizer=tokenizer,
                            )

    def test_tampered_authoritative_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout_manifest_path, tokenizer = _build_heldout_fixture(root)
            heldout = _load(heldout_manifest_path)
            none_path = _resolve(
                heldout_manifest_path,
                heldout["workloads"]["heldout"]["none"]["prepared_workload"],
            )
            source = Path(_load(none_path)["traces"][0]["source_trace"]).resolve()
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                build_stress_bundle(
                    manifest_path=heldout_manifest_path,
                    output_root=root / "stress120",
                    output_manifest=root / "manifest_stress120.json",
                    tokenizer=tokenizer,
                )


if __name__ == "__main__":
    unittest.main()
