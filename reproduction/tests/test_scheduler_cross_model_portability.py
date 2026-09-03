from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPOSITORY_ROOT
    / "reproduction/scripts/run_scheduler_cross_model_portability.py"
)
REVISION = "1" * 40
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scheduler_cross_model_portability", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SchedulerCrossModelPortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()
        cls.frozen = cls.module.formal.load_frozen_config(cls.module.BASE_CONFIG)

    def _fake_snapshot(self, root: Path) -> Path:
        snapshot = root / REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["MistralForCausalLM"],
                    "model_type": "mistral",
                    "max_position_embeddings": 32768,
                    "vocab_size": 32000,
                }
            ),
            encoding="utf-8",
        )
        (snapshot / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"}),
            encoding="utf-8",
        )
        (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"fake-local-weight")
        return snapshot

    def test_missing_snapshot_and_revision_fail_before_subprocess(self) -> None:
        missing = Path(tempfile.gettempdir()) / "definitely-missing-cross-model-snapshot"
        with mock.patch.object(
            self.module.subprocess,
            "run",
            side_effect=AssertionError("subprocess must not run"),
        ):
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "snapshot is missing"
            ):
                self.module.main(
                    [
                        "missing-model-check",
                        "--model-id",
                        MODEL_ID,
                        "--model-revision",
                        REVISION,
                        "--model-snapshot",
                        str(missing),
                        "--gpus",
                        "0,1,2,3",
                        "--check-only",
                    ]
                )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve()
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "40 lowercase hexadecimal"
            ):
                self.module._validate_identity(MODEL_ID, "abc", path)

    def test_check_only_path_never_reaches_gpu_or_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._fake_snapshot(Path(temporary))
            config = self.module._derived_config(
                self.frozen,
                model_id=MODEL_ID,
                revision=REVISION,
                gpus="0,1,2,3",
                port=8200,
            )
            with (
                mock.patch.object(
                    self.module,
                    "_full_preflight",
                    return_value=(config, Path(sys.executable), {}),
                ),
                mock.patch.object(
                    self.module, "_dependency_bindings", return_value={}
                ),
                mock.patch.object(
                    self.module,
                    "_validate_execution_hardware",
                    side_effect=AssertionError("check-only touched GPU"),
                ),
                mock.patch.object(
                    self.module,
                    "_run_cell",
                    side_effect=AssertionError("check-only executed a cell"),
                ),
                mock.patch.object(
                    self.module.subprocess,
                    "run",
                    side_effect=AssertionError("check-only ran a subprocess"),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                status = self.module.main(
                    [
                        "check-only-no-touch",
                        "--model-id",
                        MODEL_ID,
                        "--model-revision",
                        REVISION,
                        "--model-snapshot",
                        str(snapshot.resolve()),
                        "--gpus",
                        "0,1,2,3",
                        "--port",
                        "8200",
                        "--check-only",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["check_only"])
            self.assertFalse(payload["check_only_contract"]["network_touched"])
            self.assertFalse(
                payload["check_only_contract"]["gpu_or_server_touched"]
            )
            self.assertEqual(
                [cell["cell"] for cell in payload["cells"]], ["A", "E"]
            )
            self.assertTrue(all(cell["fresh_server"] for cell in payload["cells"]))
            claimed_plan_sha = payload.pop("preflight_plan_sha256")
            payload.pop("check_only")
            self.assertEqual(self.module._sha256_json(payload), claimed_plan_sha)

    def test_check_only_c5_profile_is_fully_bound_and_no_touch(self) -> None:
        fallback = self.module.PROFILES[
            self.module.CROSS_ARCHITECTURE_FALLBACK_PROFILE_ID
        ]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._fake_snapshot(Path(temporary))
            config = self.module._derived_config(
                self.frozen,
                model_id=MODEL_ID,
                revision=REVISION,
                gpus="0,1,2,3",
                port=8200,
                profile=fallback,
            )
            with (
                mock.patch.object(
                    self.module,
                    "_full_preflight",
                    return_value=(config, Path(sys.executable), {}),
                ),
                mock.patch.object(
                    self.module, "_dependency_bindings", return_value={}
                ),
                mock.patch.object(
                    self.module,
                    "_validate_execution_hardware",
                    side_effect=AssertionError("check-only touched GPU"),
                ),
                mock.patch.object(
                    self.module,
                    "_run_cell",
                    side_effect=AssertionError("check-only executed a cell"),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                status = self.module.main(
                    [
                        "granite-c5-check",
                        "--model-id",
                        MODEL_ID,
                        "--model-revision",
                        REVISION,
                        "--model-snapshot",
                        str(snapshot.resolve()),
                        "--profile",
                        fallback.profile_id,
                        "--gpus",
                        "0,1,2,3",
                        "--port",
                        "8200",
                        "--check-only",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["profile"], self.module._profile_record(fallback))
            self.assertEqual(payload["deployment"]["context_padding_tokens"], 5_000)
            self.assertIn(fallback.profile_id, payload["run_root"])
            self.assertEqual(
                [cell["label"] for cell in payload["cells"]],
                [spec.label for spec in self.module._cells(fallback)],
            )
            claimed_plan_sha = payload.pop("preflight_plan_sha256")
            payload.pop("check_only")
            self.assertEqual(self.module._sha256_json(payload), claimed_plan_sha)

    def test_one_shot_reservation_precedes_hardware_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            snapshot = self._fake_snapshot(temporary_root / "model")
            run_base = temporary_root / "runs"
            attempt_base = run_base / ".one_shot_attempts"
            config = self.module._derived_config(
                self.frozen,
                model_id=MODEL_ID,
                revision=REVISION,
                gpus="0,1,2,3",
                port=8200,
            )
            manifest = self.module._snapshot_manifest(
                snapshot, model_id=MODEL_ID, revision=REVISION
            )
            with (
                mock.patch.object(self.module, "RUN_BASE", run_base),
                mock.patch.object(self.module, "ATTEMPT_BASE", attempt_base),
                mock.patch.object(
                    self.module,
                    "_repository_relative",
                    side_effect=lambda path: str(path),
                ),
            ):
                plan = self.module._plan(
                    run_tag="hardware-failure-one-shot",
                    model_id=MODEL_ID,
                    revision=REVISION,
                    snapshot=snapshot,
                    snapshot_manifest=manifest,
                    config=config,
                    python=Path(sys.executable),
                    gpus="0,1,2,3",
                    port=8200,
                    preflight={},
                    bindings={},
                )
                hardware = mock.Mock(
                    side_effect=self.module.CrossModelProtocolError(
                        "synthetic hardware failure"
                    )
                )
                patches = (
                    mock.patch.object(
                        self.module,
                        "_full_preflight",
                        return_value=(config, Path(sys.executable), {}),
                    ),
                    mock.patch.object(
                        self.module, "_dependency_bindings", return_value={}
                    ),
                    mock.patch.object(
                        self.module, "_validate_execution_hardware", hardware
                    ),
                )
                argv = [
                    "hardware-failure-one-shot",
                    "--model-id",
                    MODEL_ID,
                    "--model-revision",
                    REVISION,
                    "--model-snapshot",
                    str(snapshot),
                    "--gpus",
                    "0,1,2,3",
                    "--port",
                    "8200",
                    "--execute-one-shot",
                    "--expected-preflight-plan-sha256",
                    plan["preflight_plan_sha256"],
                ]
                with patches[0], patches[1], patches[2]:
                    with self.assertRaisesRegex(
                        self.module.CrossModelProtocolError,
                        "synthetic hardware failure",
                    ):
                        self.module.main(argv)
                    attempt_root = attempt_base / plan["attempt_key"]
                    self.assertTrue((attempt_root / "reservation.json").is_file())
                    failure_path = run_base / argv[0] / "failure.json"
                    self.assertTrue(failure_path.is_file())
                    failure = json.loads(failure_path.read_text(encoding="utf-8"))
                    self.assertFalse(failure["rerun_allowed"])

                    with self.assertRaisesRegex(
                        self.module.CrossModelProtocolError,
                        "already consumed",
                    ):
                        self.module.main(argv)
                self.assertEqual(hardware.call_count, 1)

    def test_snapshot_manifest_detects_content_and_manifest_sha_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._fake_snapshot(Path(temporary))
            manifest = self.module._snapshot_manifest(
                snapshot, model_id=MODEL_ID, revision=REVISION
            )
            self.module._verify_snapshot_manifest(snapshot, manifest)
            self.assertEqual(manifest["file_count"], 4)
            self.assertEqual(manifest["category_counts"]["weight"], 1)

            bad_manifest = dict(manifest)
            bad_manifest["manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "manifest SHA"
            ):
                self.module._verify_snapshot_manifest(snapshot, bad_manifest)

            (snapshot / "model.safetensors").write_bytes(b"mutated-weight")
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "file changed"
            ):
                self.module._verify_snapshot_manifest(snapshot, manifest)

    def test_one_shot_key_cannot_be_evaded_by_copying_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_snapshot = self._fake_snapshot(Path(first))
            second_snapshot = Path(second) / REVISION
            shutil.copytree(first_snapshot, second_snapshot)
            first_manifest = self.module._snapshot_manifest(
                first_snapshot, model_id=MODEL_ID, revision=REVISION
            )
            second_manifest = self.module._snapshot_manifest(
                second_snapshot, model_id=MODEL_ID, revision=REVISION
            )
            self.assertNotEqual(
                first_manifest["manifest_sha256"],
                second_manifest["manifest_sha256"],
            )
            self.assertEqual(
                first_manifest["content_sha256"],
                second_manifest["content_sha256"],
            )
            config = self.module._derived_config(
                self.frozen,
                model_id=MODEL_ID,
                revision=REVISION,
                gpus="0,1,2,3",
                port=8200,
            )
            first_key = self.module._one_shot_key(
                model_id=MODEL_ID,
                revision=REVISION,
                snapshot_manifest=first_manifest,
                config=config,
            )
            second_key = self.module._one_shot_key(
                model_id=MODEL_ID,
                revision=REVISION,
                snapshot_manifest=second_manifest,
                config=config,
            )
            self.assertEqual(first_key, second_key)

    def test_profiles_preserve_legacy_key_and_bind_c5_identity(self) -> None:
        default = self.module.DEFAULT_PROFILE
        fallback = self.module.PROFILES[
            self.module.CROSS_ARCHITECTURE_FALLBACK_PROFILE_ID
        ]
        self.assertEqual(default.context_padding_tokens, 12_000)
        self.assertEqual(fallback.context_padding_tokens, 5_000)
        self.assertEqual(default.max_active_tasks, fallback.max_active_tasks)
        self.assertFalse(default.cross_architecture_fallback)
        self.assertTrue(fallback.cross_architecture_fallback)

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._fake_snapshot(Path(temporary))
            manifest = self.module._snapshot_manifest(
                snapshot, model_id=MODEL_ID, revision=REVISION
            )
            default_config = self.module._derived_config(
                self.frozen,
                model_id=MODEL_ID,
                revision=REVISION,
                gpus="0,1,2,3",
                port=8200,
                profile=default,
            )
            fallback_config = self.module._derived_config(
                self.frozen,
                model_id=MODEL_ID,
                revision=REVISION,
                gpus="0,1,2,3",
                port=8200,
                profile=fallback,
            )
            self.assertEqual(
                default_config["PASTE_LIVE_CONTEXT_PADDING_TOKENS"], "12000"
            )
            self.assertEqual(
                fallback_config["PASTE_LIVE_CONTEXT_PADDING_TOKENS"], "5000"
            )

            default_key = self.module._one_shot_key(
                model_id=MODEL_ID,
                revision=REVISION,
                snapshot_manifest=manifest,
                config=default_config,
                profile=default,
            )
            legacy_payload = {
                "protocol_version": "cross-model-ae-portability-one-shot-v1",
                "model_id": MODEL_ID,
                "revision": REVISION,
                "snapshot_content_sha256": manifest["content_sha256"],
                "workload_sha256": self.module.formal.FORMAL_WORKLOAD_SHA256,
                "fixed_order": ["A", "E"],
                "required_gpu_sku": "NVIDIA A100 40GB",
                "tensor_parallel_size": 4,
                "dtype": "bfloat16",
                "max_model_len": 16_384,
                "context_padding_tokens": 12_000,
                "max_active_tasks": 80,
                "physical_kv_target": 0.93,
                "visit_min_start_interval_s": 3.0,
                "qwen_scheduler_knobs": {
                    key: value
                    for key, value in default_config.items()
                    if key.startswith("VLLM_SCHED_")
                },
            }
            self.assertEqual(default_key, self.module._sha256_json(legacy_payload))
            fallback_key = self.module._one_shot_key(
                model_id=MODEL_ID,
                revision=REVISION,
                snapshot_manifest=manifest,
                config=fallback_config,
                profile=fallback,
            )
            self.assertNotEqual(default_key, fallback_key)
            self.assertEqual(
                [cell.context_padding_tokens for cell in self.module._cells(fallback)],
                [5_000, 5_000],
            )
            self.assertTrue(
                all(
                    "c5k-l80" in cell.label
                    for cell in self.module._cells(fallback)
                )
            )
            self.assertEqual(
                self.module._run_root("same-tag", default),
                self.module.RUN_BASE / "same-tag",
            )
            self.assertEqual(
                self.module._run_root("same-tag", fallback),
                self.module.RUN_BASE / fallback.profile_id / "same-tag",
            )
            self.assertEqual(
                self.module._attempt_root(default_key, default),
                self.module.ATTEMPT_BASE / default_key,
            )
            self.assertEqual(
                self.module._attempt_root(fallback_key, fallback),
                self.module.ATTEMPT_BASE / fallback.profile_id / fallback_key,
            )

    def test_cross_architecture_fallback_rejects_qwen_model_type(self) -> None:
        fallback = self.module.PROFILES[
            self.module.CROSS_ARCHITECTURE_FALLBACK_PROFILE_ID
        ]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen2ForCausalLM"],
                        "model_type": "qwen2",
                        "max_position_embeddings": 32768,
                    }
                ),
                encoding="utf-8",
            )
            (snapshot / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "{{ messages }}"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError,
                "rejects Qwen-family architectures",
            ):
                self.module._offline_chat_context_preflight(
                    snapshot, profile=fallback
                )

    def test_snapshot_manifest_rejects_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self._fake_snapshot(root / "model")
            external = root / "external"
            external.mkdir()
            (external / "hidden.bin").write_bytes(b"unbound")
            (snapshot / "linked-directory").symlink_to(
                external, target_is_directory=True
            )
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError,
                "directory symlinks",
            ):
                self.module._snapshot_manifest(
                    snapshot, model_id=MODEL_ID, revision=REVISION
                )

    def test_chat_template_preflight_fails_closed(self) -> None:
        class RejectingTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                raise ValueError("roles must alternate")

        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError,
            "rejects the unchanged agent message sequence",
        ):
            self.module._render_chat(
                RejectingTokenizer(),
                [{"role": "user", "content": "prompt"}],
            )

        class NestedTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return [[1, 2, 3]]

        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError,
            "flat token-ID list",
        ):
            self.module._render_chat(
                NestedTokenizer(),
                [{"role": "user", "content": "prompt"}],
            )

    def test_treatment_isolation_rejects_fcfs_leakage(self) -> None:
        config = self.module._derived_config(
            self.frozen,
            model_id=MODEL_ID,
            revision=REVISION,
            gpus="0,1,2,3",
            port=8200,
        )
        audit = self.module._environment_audit(config)
        self.assertTrue(audit["common_config_identical"])
        self.assertTrue(audit["qwen_scheduler_knobs_transferred_without_recalibration"])
        baseline = self.module.formal._cell_environment(
            config, cell="A", inherited={}
        )
        candidate = self.module.formal._cell_environment(
            config, cell="E", inherited={}
        )
        baseline["VLLM_SCHED_JOINT_V2_TOOL_BETA"] = "0.9"
        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError, "leaked scheduler treatment"
        ):
            self.module._validate_pair_environments(
                baseline, candidate, config=config
            )

        for spec in self.module.CELLS:
            recorded = self.module.formal._cell_environment(
                config, cell=spec.cell, inherited={}
            )
            runtime_audit = self.module._validate_recorded_environment(
                recorded, spec=spec, config=config
            )
            self.assertTrue(runtime_audit["full_scheduler_environment_validated"])
        candidate["VLLM_SCHED_JOINT_V2_TOOL_BETA"] = "0.9001"
        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError,
            "full scheduler environment drift",
        ):
            self.module._validate_recorded_environment(
                candidate, spec=self.module.CELLS[1], config=config
            )

    def test_transport_gate_rejects_retry_or_non_200(self) -> None:
        attempts = []
        for index in range(self.module.EXPECTED_TOOL_COMMITS):
            tool = "search" if index < 80 else "visit"
            start = float(index if tool == "search" else (index - 80) * 3.0)
            attempts.append(
                {
                    "authoritative": True,
                    "speculative": False,
                    "committed": True,
                    "outcome": "committed",
                    "http_attempts": 1,
                    "response_status": 200,
                    "http_attempt_log": [
                        {
                            "status": 200,
                            "retried": False,
                            "started_monotonic_s": start,
                        }
                    ],
                    "tool": tool,
                    "transport_identity_source": "actual",
                    "worker_pool": {
                        "tool_min_start_intervals_s": {"visit": 3.0}
                    },
                }
            )
        result = {
            "tool_attempt_records": attempts,
            "broker_final_snapshot": {
                "stats": {
                    "authoritative_requests": 160,
                    "authoritative_started": 160,
                    "authoritative_completed": 160,
                    "authoritative_executions": 160,
                    "authoritative_failures": 0,
                    "commits": 160,
                    "speculative_started": 0,
                    "speculative_completed": 0,
                    "speculative_failures": 0,
                }
            },
        }
        audit = self.module._validate_transport(result)
        self.assertEqual(audit["http_retry_count"], 0)
        self.assertEqual(audit["physical_http_attempt_count"], 160)
        self.assertEqual(audit["tool_identity_counts"], {"search": 80, "visit": 80})
        self.assertEqual(audit["minimum_observed_visit_http_start_gap_s"], 3.0)

        original = dict(result["tool_attempt_records"][9])
        result["tool_attempt_records"][9] = {
            **original,
            "http_attempts": 2,
            "response_status": 200,
            "http_attempt_log": [
                {"status": 429, "retried": True},
                {"status": 200, "retried": False},
            ],
        }
        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError, "retry-free transport gate"
        ):
            self.module._validate_transport(result)
        result["tool_attempt_records"][9] = original

        result["tool_attempt_records"][81]["http_attempt_log"][0][
            "started_monotonic_s"
        ] = 2.979
        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError, "pacing fell below"
        ):
            self.module._validate_transport(result)
        result["tool_attempt_records"][81]["http_attempt_log"][0][
            "started_monotonic_s"
        ] = 3.0

        result["broker_final_snapshot"]["stats"]["commits"] = 159
        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError, "broker ledger mismatch"
        ):
            self.module._validate_transport(result)

    def test_evidence_manifest_and_summary_sha_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.txt"
            evidence.write_text("bound\n", encoding="utf-8")
            manifest = {
                "evidence": {"evidence.txt": self.module._sha256(evidence)}
            }
            self.module._verify_relative_file_manifest(root, manifest)
            evidence.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "SHA mismatch"
            ):
                self.module._verify_relative_file_manifest(root, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            profile = self.module.DEFAULT_PROFILE
            profile_record = self.module._profile_record(profile)
            selected_cells = self.module._cells(profile)
            attempt_key = "a" * 64
            (run_root / "run_plan.json").write_text(
                json.dumps(
                    {
                        "profile": profile_record,
                        "attempt_key": attempt_key,
                        "cells": [
                            {"label": spec.label} for spec in selected_cells
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_root / "model_snapshot_manifest.json").write_text(
                "snapshot-manifest\n", encoding="utf-8"
            )
            (run_root / "execution_hardware.json").write_text(
                json.dumps({"profile": profile_record}), encoding="utf-8"
            )
            (run_root / "summary.json").write_text(
                json.dumps(
                    {
                        "profile": profile_record,
                        "cells": {spec.label: {} for spec in selected_cells},
                    }
                ),
                encoding="utf-8",
            )
            cell_manifest_paths = []
            for index, spec in enumerate(selected_cells):
                cell_root = (
                    run_root / "cells" / f"{index + 1:02d}-{spec.label}"
                )
                cell_root.mkdir(parents=True)
                evidence = cell_root / "evidence.txt"
                evidence.write_text(
                    f"{spec.label}-evidence\n", encoding="utf-8"
                )
                cell_manifest = cell_root / "cell_manifest.json"
                cell_manifest.write_text(
                    json.dumps(
                        {
                            "profile": profile_record,
                            "cell": spec.label,
                            "evidence": {
                                "evidence.txt": self.module._sha256(evidence)
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                cell_manifest_paths.append(cell_manifest)
            completion = {
                "profile": profile_record,
                "attempt_key": attempt_key,
                "dependency_bindings": {
                    "reproduction/scripts/run_scheduler_cross_model_portability.py": (
                        self.module._sha256(SCRIPT)
                    )
                },
                "run_plan": {
                    "path": "run_plan.json",
                    "sha256": self.module._sha256(run_root / "run_plan.json"),
                },
                "model_snapshot_manifest": {
                    "path": "model_snapshot_manifest.json",
                    "sha256": self.module._sha256(
                        run_root / "model_snapshot_manifest.json"
                    ),
                },
                "execution_hardware": {
                    "path": "execution_hardware.json",
                    "sha256": self.module._sha256(
                        run_root / "execution_hardware.json"
                    ),
                },
                "summary": {
                    "path": "summary.json",
                    "sha256": self.module._sha256(run_root / "summary.json"),
                },
                "cells": [
                    {
                        "label": selected_cells[0].label,
                        "path": (
                            f"cells/01-{selected_cells[0].label}/"
                            "cell_manifest.json"
                        ),
                        "sha256": self.module._sha256(cell_manifest_paths[0]),
                    },
                    {
                        "label": selected_cells[1].label,
                        "path": (
                            f"cells/02-{selected_cells[1].label}/"
                            "cell_manifest.json"
                        ),
                        "sha256": self.module._sha256(cell_manifest_paths[1]),
                    },
                ],
            }
            self.module._verify_completion_manifest(run_root, completion)
            cell_a_evidence = cell_manifest_paths[0].parent / "evidence.txt"
            cell_a_evidence.write_text("altered-cell-evidence\n", encoding="utf-8")
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "evidence manifest SHA mismatch"
            ):
                self.module._verify_completion_manifest(run_root, completion)
            cell_a_evidence.write_text(
                f"{selected_cells[0].label}-evidence\n", encoding="utf-8"
            )
            spliced = json.loads(cell_manifest_paths[1].read_text(encoding="utf-8"))
            spliced["profile"] = self.module._profile_record(
                self.module.PROFILES[
                    self.module.CROSS_ARCHITECTURE_FALLBACK_PROFILE_ID
                ]
            )
            cell_manifest_paths[1].write_text(
                json.dumps(spliced), encoding="utf-8"
            )
            completion["cells"][1]["sha256"] = self.module._sha256(
                cell_manifest_paths[1]
            )
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "profile/label mismatch"
            ):
                self.module._verify_completion_manifest(run_root, completion)
            (run_root / "summary.json").write_text("altered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError, "summary SHA mismatch"
            ):
                self.module._verify_completion_manifest(run_root, completion)

    def test_fixed_shape_port_and_command_contract(self) -> None:
        config = self.module._derived_config(
            self.frozen,
            model_id=MODEL_ID,
            revision=REVISION,
            gpus="0,1,2,3",
            port=8200,
        )
        self.assertEqual(config["VLLM_TP_SIZE"], "4")
        self.assertEqual(config["VLLM_DTYPE"], "bfloat16")
        self.assertEqual(config["VLLM_MAX_MODEL_LEN"], "16384")
        self.assertEqual(config["PASTE_LIVE_CONTEXT_PADDING_TOKENS"], "12000")
        self.assertEqual(config["PASTE_LIVE_MAX_ACTIVE_TASKS"], "80")
        self.assertEqual(config["PASTE_LIVE_VISIT_MIN_START_INTERVAL_S"], "3.0")
        self.assertEqual(
            config["VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION"],
            "0.93",
        )
        with self.assertRaisesRegex(
            self.module.CrossModelProtocolError, "reserved"
        ):
            self.module._validate_port(8100)

        snapshot = Path("/tmp") / REVISION
        command = self.module._runner_command(
            python=Path(sys.executable),
            snapshot=snapshot,
            output=Path("/tmp/cross-model-evidence"),
            spec=self.module.CELLS[0],
            block_id="test-cross-model",
            order_index=0,
            server_instance_id="test-server-instance",
            config=config,
        )
        tokenizer_index = command.index("--tokenizer") + 1
        self.assertEqual(command[tokenizer_index], str(snapshot))
        self.assertEqual(command[command.index("--model") + 1], MODEL_ID)

    def test_execution_hardware_gate_rejects_non_a100_shape(self) -> None:
        good_stdout = "\n".join(
            f"{index}, NVIDIA A100-SXM4-40GB, 40960" for index in range(4)
        )
        with mock.patch.object(
            self.module.subprocess,
            "run",
            return_value=self.module.subprocess.CompletedProcess(
                args=[], returncode=0, stdout=good_stdout, stderr=""
            ),
        ):
            audit = self.module._validate_execution_hardware("0,1,2,3")
        self.assertEqual(audit["gpu_count"], 4)
        self.assertEqual(audit["tensor_parallel_size"], 4)

        bad_stdout = good_stdout.replace(
            "3, NVIDIA A100-SXM4-40GB, 40960",
            "3, NVIDIA H100 80GB HBM3, 81920",
        )
        with mock.patch.object(
            self.module.subprocess,
            "run",
            return_value=self.module.subprocess.CompletedProcess(
                args=[], returncode=0, stdout=bad_stdout, stderr=""
            ),
        ):
            with self.assertRaisesRegex(
                self.module.CrossModelProtocolError,
                "four identical NVIDIA A100 40GB",
            ):
                self.module._validate_execution_hardware("0,1,2,3")


if __name__ == "__main__":
    unittest.main()
