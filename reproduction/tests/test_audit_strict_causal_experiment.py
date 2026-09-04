from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_strict_causal_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("strict_causal_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

sys.path.insert(0, str(SCRIPT.parent))
ANALYZER_SCRIPT = SCRIPT.parent / "analyze_strict_causal_abef.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "strict_causal_analyzer", ANALYZER_SCRIPT
)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
analyzer = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(analyzer)

MATERIALIZER_SCRIPT = SCRIPT.parent / "materialize_strict_causal_manifest.py"
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "strict_causal_materializer", MATERIALIZER_SCRIPT
)
assert MATERIALIZER_SPEC is not None and MATERIALIZER_SPEC.loader is not None
materializer = importlib.util.module_from_spec(MATERIALIZER_SPEC)
MATERIALIZER_SPEC.loader.exec_module(materializer)


HEX = "a" * 64


def _attach_gemini_legacy_compatibility(
    tmp_path: Path, manifest: dict
) -> tuple[dict, Path]:
    """Attach a small, fully signed compatibility proof to a base fixture."""

    frozen = {row["role"]: row for row in manifest["frozen_files"]}
    training_hash = audit.canonical_sha256(
        sorted(manifest["data"]["calibration_root_ids"])
    )
    historical_builder = "b" * 64
    prediction_code = _write(tmp_path / "prediction-code.py", "# prediction code\n")
    prediction_code["role"] = "prediction_code"
    manifest["frozen_files"].append(prediction_code)

    invocation_binding = manifest["predictors"]["tool_invocation"]["artifact"]
    invocation_path = tmp_path / invocation_binding["path"]
    invocation_document = json.loads(invocation_path.read_text(encoding="utf-8"))
    invocation_document.update(
        {
            "builder_code_sha256": historical_builder,
            "training": {"source_registry_sha256": "c" * 64},
        }
    )
    invocation_document["artifact_sha256"] = audit.canonical_sha256(
        {
            key: value
            for key, value in invocation_document.items()
            if key != "artifact_sha256"
        }
    )
    invocation_path.write_text(
        json.dumps(invocation_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    invocation_binding.update(
        {
            "sha256": audit.file_sha256(invocation_path),
            "identity_sha256": invocation_document["artifact_sha256"],
        }
    )

    duration_binding = manifest["predictors"]["tool_duration"]["artifact"]
    duration_path = tmp_path / duration_binding["path"]
    model_unsigned = {
        "global_s": 1.0,
        "by_tool_s": {"repo_read": 2.0},
        "by_repository_s": {"org/repo": 4.0},
    }
    duration_model_sha256 = audit.canonical_sha256(model_unsigned)
    duration_document = {
        "input_features": ["candidate_tool_name", "candidate_repository"],
        "training_root_ids_sha256": training_hash,
        "uses_evaluation_labels": False,
        "fit_code_sha256": historical_builder,
        "prediction_code_sha256": historical_builder,
        "model": {
            **model_unsigned,
            "duration_predictor_sha256": duration_model_sha256,
        },
    }
    duration_document["artifact_sha256"] = audit.canonical_sha256(
        duration_document
    )
    duration_path.write_text(
        json.dumps(duration_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    duration_binding.update(
        {
            "sha256": audit.file_sha256(duration_path),
            "identity_sha256": duration_document["artifact_sha256"],
            "input_features": duration_document["input_features"],
            "fit_code_sha256": historical_builder,
        }
    )
    manifest["predictors"]["tool_duration"]["input_features"] = list(
        duration_document["input_features"]
    )

    verifier_marker = tmp_path / "verifier-reran.marker"
    verifier_path = tmp_path / "compat-verifier.py"
    verifier_path.write_text(
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(verifier_marker)!r}).write_text('reran')\n"
        "p=pathlib.Path(sys.argv[sys.argv.index('--certificate')+1])\n"
        "c=json.loads(p.read_text())\n"
        "a=c['artifact_tuple']; b=c['behavioral_equivalence']\n"
        "print(json.dumps({'valid':True,'compatibility_sha256':c['compatibility_sha256'],"
        "'invocation_logical_sha256':a['invocation_logical_sha256'],"
        "'duration_logical_sha256':a['duration_logical_sha256'],"
        "'service_clock_logical_sha256':a['service_clock_logical_sha256'],"
        "'current_cell_runner_sha256':c['current_cell_runner_sha256'],"
        "'independent_verifier_sha256':c['independent_verifier_sha256'],"
        "'behavioral_vectors_sha256':b['behavioral_vectors_sha256'],"
        "'behavioral_vector_count':b['behavioral_vector_count']},sort_keys=True))\n",
        encoding="utf-8",
    )
    verifier_binding = {
        "role": audit.GEMINI_LEGACY_COMPATIBILITY_VERIFIER_ROLE,
        "path": str(verifier_path),
        "sha256": audit.file_sha256(verifier_path),
    }

    contract = {
        "schema": "paste_gemini.swe_strict_duration_inference_semantics.v1",
        "inputs": ["candidate_tool_name", "candidate_repository"],
        "value_encoding": "python_binary64_float_hex_v1",
        "positive_validation": (
            "all_inputs_and_output_finite_and_strictly_positive"
        ),
        "model": {
            "global_s": 1.0.hex(),
            "by_tool_s": {"repo_read": 2.0.hex()},
            "by_repository_s": {"org/repo": 4.0.hex()},
        },
        "lookup": {
            "tool": {
                "key": "candidate_tool_name",
                "map": "by_tool_s",
                "missing": "global_s",
            },
            "repository": {
                "key": "candidate_repository",
                "map": "by_repository_s",
                "missing": "global_s",
            },
        },
        "combiner": "arithmetic_mean_tool_repository_v1",
        "output_exactness": "python_binary64_(tool_s+repository_s)/2.0",
    }
    unknown_tool = "<PASTE_UNKNOWN_TOOL_V1>"
    unknown_repository = "<PASTE_UNKNOWN_REPOSITORY_V1>"
    tool_domain = ["repo_read", unknown_tool]
    repository_domain = ["org/repo", unknown_repository]
    vectors = []
    for tool in tool_domain:
        for repository in repository_domain:
            tool_exact = tool == "repo_read"
            repository_exact = repository == "org/repo"
            eta = ((2.0 if tool_exact else 1.0) + (
                4.0 if repository_exact else 1.0
            )) / 2.0
            vectors.append(
                {
                    "candidate_tool_name": tool,
                    "candidate_repository": repository,
                    "tool_lookup": "exact_key" if tool_exact else "global_fallback",
                    "repository_lookup": (
                        "exact_key" if repository_exact else "global_fallback"
                    ),
                    "service_s_hex": eta.hex(),
                }
            )
    behavior = {
        "tool_domain": tool_domain,
        "repository_domain": repository_domain,
        "unknown_tool_sentinel": unknown_tool,
        "unknown_repository_sentinel": unknown_repository,
        "vectors": vectors,
        "behavioral_vectors_sha256": audit.canonical_sha256(vectors),
        "behavioral_vector_count": len(vectors),
        "exact_bitwise_match": True,
    }
    certificate_path = tmp_path / "legacy-compatibility.json"
    certificate = {
        "schema": audit.GEMINI_LEGACY_COMPATIBILITY_SCHEMA,
        "compatibility_mode": audit.GEMINI_LEGACY_COMPATIBILITY_MODE,
        "artifact_tuple": {
            "invocation_file_sha256": invocation_binding["sha256"],
            "invocation_logical_sha256": invocation_binding["identity_sha256"],
            "duration_file_sha256": duration_binding["sha256"],
            "duration_logical_sha256": duration_binding["identity_sha256"],
            "service_clock_file_sha256": manifest["execution"][
                "physical_service_clock"
            ]["artifact"]["sha256"],
            "service_clock_logical_sha256": audit.artifact_identity_sha256(
                manifest["execution"]["physical_service_clock"]["artifact"]
            ),
            "historical_builder_code_sha256": historical_builder,
            "invocation_runtime_module_sha256": prediction_code["sha256"],
            "duration_model_sha256": duration_model_sha256,
            "calibration_root_ids_sha256": training_hash,
            "source_registry_sha256": "c" * 64,
        },
        "duration_inference_contract": contract,
        "duration_inference_contract_sha256": audit.canonical_sha256(contract),
        "behavioral_equivalence": behavior,
        "current_cell_runner_sha256": frozen["runner"]["sha256"],
        "independent_verifier_sha256": verifier_binding["sha256"],
    }
    certificate["compatibility_sha256"] = audit.canonical_sha256(certificate)
    certificate_path.write_text(
        json.dumps(certificate, sort_keys=True) + "\n", encoding="utf-8"
    )
    certificate_binding = {
        "role": audit.GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE,
        "path": str(certificate_path),
        "sha256": audit.file_sha256(certificate_path),
        "identity_sha256": certificate["compatibility_sha256"],
        "verifier_sha256": verifier_binding["sha256"],
        "schema": audit.GEMINI_LEGACY_COMPATIBILITY_SCHEMA,
        "compatibility_mode": audit.GEMINI_LEGACY_COMPATIBILITY_MODE,
    }
    manifest["frozen_files"].extend([certificate_binding, verifier_binding])

    runtime_binding = {
        "schema": audit.GEMINI_LEGACY_COMPATIBILITY_SCHEMA,
        "compatibility_mode": audit.GEMINI_LEGACY_COMPATIBILITY_MODE,
        "certificate_path": str(certificate_path.resolve()),
        "certificate_file_sha256": certificate_binding["sha256"],
        "compatibility_sha256": certificate_binding["identity_sha256"],
        "independent_verifier_path": str(verifier_path.resolve()),
        "independent_verifier_sha256": verifier_binding["sha256"],
    }
    policy_path = tmp_path / frozen["policy_bundle"]["path"]
    policy_document = {
        "schema": "paste_gemini.swe_strict_policy_plan.v1",
        "legacy_frozen_compatibility": runtime_binding,
    }
    policy_path.write_text(
        json.dumps(policy_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    frozen["policy_bundle"]["sha256"] = audit.file_sha256(policy_path)
    _reseal(manifest, tmp_path)
    return runtime_binding, verifier_marker


def test_protocol_distinguishes_broker_acceptance_from_physical_start() -> None:
    protocol = (
        Path(__file__).resolve().parents[1] / "STRICT_CAUSAL_PAPER_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "Started counts must equal the broker counter" not in protocol
    assert "broker-accepted count may be larger" in protocol
    assert (
        "unique execution-ledger candidates with a non-null physical-start timestamp"
        in protocol
    )
    assert (
        "`admitted_candidates` and `admitted_candidate_precision` are exact aliases"
        in protocol
    )
    assert "immutable first authority claim" in protocol
    assert "both registered strict result schemas" in protocol


def test_gemini_legacy_certificate_is_bound_and_verifier_is_rerun(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    runtime_binding, verifier_marker = _attach_gemini_legacy_compatibility(
        tmp_path, manifest
    )
    checked = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert checked["valid"] is True, checked["errors"]
    assert verifier_marker.read_text(encoding="utf-8") == "reran"
    provenance = audit.expected_runtime_provenance(manifest)
    assert provenance["legacy_compatibility_certificate_file_sha256"] == (
        runtime_binding["certificate_file_sha256"]
    )
    assert provenance["legacy_compatibility_certificate_sha256"] == (
        runtime_binding["compatibility_sha256"]
    )
    assert provenance["legacy_compatibility_verifier_file_sha256"] == (
        runtime_binding["independent_verifier_sha256"]
    )


def test_gemini_legacy_certificate_vector_and_eta_poison_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    runtime_binding, _marker = _attach_gemini_legacy_compatibility(
        tmp_path, manifest
    )
    errors: list[str] = []
    context = audit._audit_gemini_legacy_compatibility_manifest(
        manifest, base=tmp_path, verify_files=True, errors=errors
    )
    assert errors == []
    assert context is not None
    provenance = {
        field: HEX for field in audit.RUNTIME_PROVENANCE_FIELDS
    }
    provenance.update(
        {
            "legacy_compatibility_certificate_file_sha256": runtime_binding[
                "certificate_file_sha256"
            ],
            "legacy_compatibility_certificate_sha256": runtime_binding[
                "compatibility_sha256"
            ],
            "legacy_compatibility_verifier_file_sha256": runtime_binding[
                "independent_verifier_sha256"
            ],
        }
    )
    payload = {
        "schema": audit.GEMINI_RESULT_SCHEMA,
        "legacy_frozen_compatibility": runtime_binding,
        "provenance": provenance,
        "tasks": [{"task_id": "task-1", "repository": "org/repo"}],
        "prediction_decisions": [
            {
                "task_id": "task-1",
                "request_index": 0,
                "tool_name_hat": "repo_read",
                "tool_service_s_hat": 3.0,
                "candidates": [
                    {
                        "tool_name_hat": "repo_read",
                        "tool_eta_s_hat": 3.0,
                    }
                ],
            }
        ],
        "llm_events": [
            {
                "task_id": "task-1",
                "request_index": 0,
                "scheduler_metadata": {
                    "tool_eta_s_hat": 3.0,
                    "remaining_tool_wait_s_hat": 3.0,
                },
            }
        ],
        "tool_events": [
            {
                "task_id": "task-1",
                "request_index": 0,
                "tool": "repo_read",
                "authority_eta_hat_s": 3.0,
                "predicted_tool_service_s_hat": 3.0,
            }
        ],
    }
    eta_errors: list[str] = []
    audit._audit_gemini_legacy_compatibility_result(
        payload, context=context, label="$", errors=eta_errors
    )
    assert eta_errors == []

    poisoned = copy.deepcopy(payload)
    poisoned["prediction_decisions"][0]["tool_service_s_hat"] = math.nextafter(
        3.0, math.inf
    )
    poison_errors: list[str] = []
    audit._audit_gemini_legacy_compatibility_result(
        poisoned, context=context, label="$", errors=poison_errors
    )
    assert any("independent recomputation mismatch" in row for row in poison_errors)

    repository_context = dict(context)
    repository_context["registered_task_repository"] = {"task-1": "org/repo"}
    poisoned_repository = copy.deepcopy(payload)
    poisoned_repository["tasks"][0]["repository"] = "attacker/repository"
    repository_errors: list[str] = []
    audit._audit_gemini_legacy_compatibility_result(
        poisoned_repository,
        context=repository_context,
        label="$",
        errors=repository_errors,
    )
    assert any("repository differs from frozen policy" in row for row in repository_errors)

    certificate_binding = next(
        row
        for row in manifest["frozen_files"]
        if row["role"] == audit.GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE
    )
    certificate_path = Path(certificate_binding["path"])
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["behavioral_equivalence"]["vectors"][0]["service_s_hex"] = (
        math.nextafter(3.0, math.inf).hex()
    )
    certificate["behavioral_equivalence"]["behavioral_vectors_sha256"] = (
        audit.canonical_sha256(certificate["behavioral_equivalence"]["vectors"])
    )
    unsigned = dict(certificate)
    unsigned.pop("compatibility_sha256", None)
    certificate["compatibility_sha256"] = audit.canonical_sha256(unsigned)
    certificate_path.write_text(
        json.dumps(certificate, sort_keys=True) + "\n", encoding="utf-8"
    )
    certificate_binding["sha256"] = audit.file_sha256(certificate_path)
    certificate_binding["identity_sha256"] = certificate["compatibility_sha256"]
    vector_errors: list[str] = []
    audit._audit_gemini_legacy_compatibility_manifest(
        manifest, base=tmp_path, verify_files=True, errors=vector_errors
    )
    assert any("service_s_hex: recomputation mismatch" in row for row in vector_errors)


def test_materializer_requires_both_gemini_legacy_frozen_roles(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    _runtime_binding, _marker = _attach_gemini_legacy_compatibility(
        tmp_path, manifest
    )
    rows = [copy.deepcopy(row) for row in manifest["frozen_files"]]
    certificate = next(
        row
        for row in rows
        if row["role"] == audit.GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE
    )
    certificate.pop("identity_sha256", None)
    certificate.pop("verifier_sha256", None)
    certificate.pop("schema", None)
    certificate.pop("compatibility_mode", None)
    annotated = materializer._annotate_legacy_compatibility_bindings(rows)
    assert annotated is not None
    assert certificate["identity_sha256"] == annotated["compatibility_sha256"]

    verifier_role = audit.GEMINI_LEGACY_COMPATIBILITY_VERIFIER_ROLE
    with pytest.raises(ValueError, match="requires both"):
        materializer._annotate_legacy_compatibility_bindings(
            [row for row in rows if row["role"] != verifier_role]
        )


def test_qwen_provenance_shape_is_unchanged_without_compatibility_roles(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    assert set(audit.expected_runtime_provenance(manifest)) == set(
        audit.RUNTIME_PROVENANCE_FIELDS
    )
    assert materializer._annotate_legacy_compatibility_bindings(
        manifest["frozen_files"]
    ) is None


def test_gemini_current_artifacts_do_not_require_legacy_compatibility_roles(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    policy_binding = next(
        row for row in manifest["frozen_files"] if row["role"] == "policy_bundle"
    )
    policy_path = tmp_path / policy_binding["path"]
    policy_path.write_text(
        json.dumps(
            {
                "schema": "paste_gemini.swe_strict_policy_plan.v1",
                "legacy_frozen_compatibility": None,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    policy_binding["sha256"] = audit.file_sha256(policy_path)
    errors: list[str] = []
    assert audit._audit_gemini_legacy_compatibility_manifest(
        manifest, base=tmp_path, verify_files=True, errors=errors
    ) is None
    assert errors == []


def test_matrix_and_result_require_all_gemini_legacy_provenance_hashes(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    runtime_binding, _marker = _attach_gemini_legacy_compatibility(
        tmp_path, manifest
    )
    provenance = audit.expected_runtime_provenance(manifest)
    poisoned_provenance = copy.deepcopy(provenance)
    poisoned_provenance.pop("legacy_compatibility_certificate_sha256")
    matrix_index = tmp_path / "matrix-index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "provenance": poisoned_provenance,
                "runtime_parameters": _manifest_runtime_result(manifest),
                "cell_evidence": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="legacy_compatibility_certificate_sha256"
    ):
        materializer._bind_matrix(manifest, matrix_index=matrix_index)

    result = {
        "schema": audit.GEMINI_RESULT_SCHEMA,
        "provenance": {
            **{field: HEX for field in audit.RUNTIME_PROVENANCE_FIELDS},
            "legacy_compatibility_certificate_file_sha256": runtime_binding[
                "certificate_file_sha256"
            ],
            "legacy_compatibility_certificate_sha256": runtime_binding[
                "compatibility_sha256"
            ],
            "legacy_compatibility_verifier_file_sha256": runtime_binding[
                "independent_verifier_sha256"
            ],
        },
        "legacy_frozen_compatibility": runtime_binding,
    }
    del result["provenance"]["legacy_compatibility_verifier_file_sha256"]
    errors = audit.audit_result_payload(result)
    assert any(
        "legacy_compatibility_verifier_file_sha256" in error for error in errors
    )


def _runtime_parameters(*, workload_instances: int = 1) -> dict:
    parameters = {
        "model_id": "fixture/model",
        "model_revision": "fixture-revision",
        "server_host": "127.0.0.1",
        "server_port": 8100,
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.86,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 48,
        "cuda_graph_sizes": [32],
        "prefix_caching": True,
        "vllm_v1": True,
        "max_active_tasks": 80,
        "tool_capacity": 32,
        "configured_speculation_capacity": 8,
        "request_timeout_s": 600.0,
        "public_output_cap": 128,
        "workload_instances": workload_instances,
        "arrival_schedule_sha256": "9" * 64,
    }
    return {
        "schema": audit.RUNTIME_PARAMETERS_SCHEMA,
        "parameters": parameters,
        "runtime_parameters_sha256": audit.runtime_parameters_sha256(parameters),
    }


def _manifest_runtime_result(manifest: dict) -> dict:
    runtime = manifest["execution"]["treatment_neutral_runtime_parameters"]
    return {
        key: copy.deepcopy(runtime[key])
        for key in ("schema", "parameters", "runtime_parameters_sha256")
    }


def _write(path: Path, value: str) -> dict[str, str]:
    path.write_text(value, encoding="utf-8")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _absolute_binding(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_materializer_accepts_runner_root_artifact_aliases(tmp_path: Path) -> None:
    qwen_roots = tmp_path / "qwen-roots.json"
    qwen_roots.write_text(
        json.dumps({"source_session_ids": ["trace-a", "trace-b"]}),
        encoding="utf-8",
    )
    gemini_roots = tmp_path / "gemini-roots.json"
    gemini_roots.write_text(
        json.dumps({"root_ids": ["task-a", "task-b"]}), encoding="utf-8"
    )
    assert materializer._root_ids(qwen_roots) == ["trace-a", "trace-b"]
    assert materializer._root_ids(gemini_roots) == ["task-a", "task-b"]


def test_materializer_feature_cli_has_no_implicit_inputs() -> None:
    args = materializer.parser().parse_args(
        [
            "create",
            "manifest.json",
            "--claim-scope",
            "confirmatory",
            "--calibration-roots",
            "calibration.json",
            "--tuning-roots",
            "tuning.json",
            "--evaluation-roots",
            "evaluation.json",
            "--selection-protocol",
            "heldout_tuning_split",
            "--invocation-predictor-artifact",
            "invocation.json",
            "--duration-predictor-artifact",
            "duration.json",
            "--service-clock-artifact",
            "service.json",
            "--runtime-parameters-json",
            "runtime.json",
        ]
    )
    assert args.invocation_feature == []
    assert args.duration_feature == []


def test_public_plan_firewall_rejects_future_authority_fields(tmp_path: Path) -> None:
    public = tmp_path / "public.json"
    public.write_text(json.dumps({"traces": [{"tools_after": []}]}), encoding="utf-8")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "plans": {
                    "final": {
                        "public": {
                            "path": public.name,
                            "sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    errors: list[str] = []
    audit.audit_public_plan_firewall(
        {"path": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()},
        base=tmp_path,
        errors=errors,
    )
    assert any("future-authority fields exposed" in error for error in errors)


def _base_manifest(tmp_path: Path, *, scope: str = "confirmatory") -> dict:
    protocol = _write(tmp_path / "protocol.md", "frozen protocol\n")
    runner = _write(tmp_path / "runner.py", "# frozen runner\n")
    config = _write(tmp_path / "config.env", "STRICT=1\n")
    hook = _write(tmp_path / "scheduler_hook.py", "# frozen hook\n")
    materializer_binding = _write(
        tmp_path / "materializer.py", "# frozen materializer\n"
    )
    auditor_binding = _write(tmp_path / "auditor.py", "# frozen auditor\n")
    analyzer_binding = _write(tmp_path / "analyzer.py", "# frozen analyzer\n")
    calibration_ids = ["cal-1", "cal-2"]
    evaluation_ids = [f"eval-{index:02d}" for index in range(1, 31)]
    policy_document = {
        "schema": "paste.paper.registered_workload_contract.v1",
        "tasks": [
            {
                "task_id": f"task-{root_id}",
                "root_id": root_id,
                "release_offset_s": 0.0,
                "request_count": 1,
            }
            for root_id in evaluation_ids
        ],
    }
    policy = _write(
        tmp_path / "policy.json",
        json.dumps(policy_document, sort_keys=True) + "\n",
    )
    training_hash = audit.canonical_sha256(sorted(calibration_ids))
    near_duplicate = None
    if scope == "confirmatory":
        duplicate_document = {
            "schema": audit.NEAR_DUPLICATE_AUDIT_SCHEMA,
            "verified": True,
            "registered_root_sets_sha256": audit.registered_root_sets_sha256(
                calibration_ids, ["tune-1", "tune-2"], evaluation_ids
            ),
            "method": "fixture exact-and-semantic fingerprint audit",
            "near_duplicate_pairs_across_splits": [],
        }
        near_duplicate = _write(
            tmp_path / "near-duplicate-audit.json",
            json.dumps(duplicate_document, sort_keys=True) + "\n",
        )
        near_duplicate.update(duplicate_document)
    invocation_features = [
        "current_messages",
        "committed_tool_results",
        "broker_queue_state",
    ]
    duration_features = [
        "candidate_tool_name",
        "candidate_host",
        "completed_tool_service_times",
    ]
    invocation_body = {
        "input_features": invocation_features,
        "training_root_ids_sha256": training_hash,
        "uses_evaluation_labels": False,
        "fit_code_sha256": "4" * 64,
    }
    invocation_identity = audit.canonical_sha256(invocation_body)
    invocation = _write(
        tmp_path / "invocation.json",
        json.dumps({**invocation_body, "artifact_sha256": invocation_identity})
        + "\n",
    )
    invocation.update(
        {
            "identity_sha256": invocation_identity,
            "input_features": invocation_features,
            "training_root_ids_sha256": training_hash,
            "uses_evaluation_labels": False,
            "fit_code_sha256": "4" * 64,
        }
    )
    duration_body = {
        "input_features": duration_features,
        "training_root_ids_sha256": training_hash,
        "uses_evaluation_labels": False,
        "fit_code_sha256": "5" * 64,
    }
    duration_identity = audit.canonical_sha256(duration_body)
    duration = _write(
        tmp_path / "duration.json",
        json.dumps({**duration_body, "artifact_sha256": duration_identity})
        + "\n",
    )
    duration.update(
        {
            "identity_sha256": duration_identity,
            "input_features": duration_features,
            "training_root_ids_sha256": training_hash,
            "uses_evaluation_labels": False,
            "fit_code_sha256": "5" * 64,
        }
    )
    service_body = {
        "training_root_ids_sha256": training_hash,
        "uses_evaluation_labels": False,
        "uses_evaluation_trace_durations": False,
        "future_state_accepted_invariant": True,
    }
    service_identity = audit.canonical_sha256(service_body)
    service_clock = _write(
        tmp_path / "service-clock.json",
        json.dumps({**service_body, "artifact_sha256": service_identity})
        + "\n",
    )
    service_clock.update(
        {
            "identity_sha256": service_identity,
            "training_root_ids_sha256": training_hash,
            "uses_evaluation_labels": False,
            "uses_evaluation_trace_durations": False,
            "future_state_accepted_invariant": True,
        }
    )
    runtime_document = _runtime_parameters(workload_instances=len(evaluation_ids))
    runtime_binding = _write(
        tmp_path / "runtime-parameters.json",
        json.dumps(runtime_document, sort_keys=True) + "\n",
    )
    runtime_binding["identity_sha256"] = runtime_document[
        "runtime_parameters_sha256"
    ]
    runtime_contract = {**runtime_document, "artifact": runtime_binding}
    manifest = {
        "schema": audit.MANIFEST_SCHEMA,
        "version": 1,
        "claim_scope": scope,
        "data": {
            "calibration_root_ids": calibration_ids,
            "tuning_root_ids": ["tune-1", "tune-2"],
            "evaluation_root_ids": evaluation_ids,
            "previously_observed_evaluation_root_ids": (
                [] if scope == "confirmatory" else ["eval-01"]
            ),
            "split_unit": "root_trace_or_task",
            "exact_root_disjoint_guard": True,
            "near_duplicate_guard": (
                "verified" if near_duplicate is not None else "not_verified"
            ),
            "selection_protocol": "heldout_tuning_split",
            "evaluation_used_for_model_or_policy_selection": False,
        },
        "freeze": {
            "sealed_before_evaluation": True,
            "no_tuning_after_seal": True,
            "accept_result_regardless_of_direction": True,
            "started_marker_exclusive_create": True,
            "formal_result_used_for_optimization": False,
            "policy_bundle_sha256": HEX,
            "sealed_payload_sha256": HEX,
            "started_marker_path": "FORMAL_STARTED.json",
            "sealed_at_utc": "2026-09-03T00:00:00Z",
        },
        "frozen_files": [
            {"role": "protocol", **protocol},
            {"role": "runner", **runner},
            {"role": "policy_bundle", **policy},
            {"role": "config", **config},
            {"role": "scheduler_hook", **hook},
            {"role": "materializer", **materializer_binding},
            {"role": "auditor", **auditor_binding},
            {"role": "analyzer", **analyzer_binding},
        ],
        "predictors": {
            "tool_invocation": {
                "training_role": "calibration",
                "uses_evaluation_labels": False,
                "frozen_before_evaluation": True,
                "training_root_ids_sha256": training_hash,
                "input_features": invocation_features,
                "artifact": invocation,
            },
            "tool_duration": {
                "training_role": "calibration",
                "uses_evaluation_labels": False,
                "frozen_before_evaluation": True,
                "training_root_ids_sha256": training_hash,
                "input_features": duration_features,
                "artifact": duration,
            },
        },
        "policy": {
            "decision_schema": audit.DECISION_SCHEMA,
            "forbidden_decision_features": sorted(
                audit.FORBIDDEN_DECISION_FEATURES
            ),
            "cells": copy.deepcopy(audit.CELLS),
            "call_graph_mode": "autonomous",
            "claim_type": "closed_loop_agent",
            "offline_tool_credit_s": 0,
        },
        "execution": {
            **{key: True for key in audit.REQUIRED_EXECUTION_ATTESTATIONS},
            "physical_service_clock": {
                "mode": "calibration_hashed_empirical_v1",
                "assignment_key": "normalized_tool_arguments_environment_version",
                "policy_can_read_realization": False,
                "policy_can_reconstruct_realization": False,
                "uses_evaluation_trace_durations": False,
                "future_authority_hit_invariant": True,
                "future_state_accepted_invariant": True,
                "same_key_same_service_all_cells": True,
                "training_role": "calibration",
                "uses_evaluation_labels": False,
                "training_root_ids_sha256": training_hash,
                "artifact": service_clock,
            },
            "treatment_neutral_runtime_parameters": runtime_contract,
            "blocks": [
                {
                    "block_id": f"block-{index + 1}",
                    "order": list(order),
                    "gpu_ids": [0, 1, 2, 3] if index % 2 == 0 else [4, 5, 6, 7],
                }
                for index, order in enumerate(audit.WILLIAMS_ORDERS)
            ],
        },
        "statistics": {
            "unit": "root_trace_or_task",
            "replicas_folded_within_root": True,
            "blocks_folded_within_root": True,
            "paired": True,
            "confidence_interval_method": "paired_root_cluster_percentile_bootstrap",
            "paired_bootstrap_resamples": 10_000,
            "paired_bootstrap_seed": "strict-causal-paper-v1",
            "ci_level": 0.95,
            "primary_contrast": "A_vs_F",
            "estimand": "ratio_of_paired_root_mean_e2e",
            "speedup_threshold": 0.20,
            "pass_rule": "point_estimate_ge_0.20_and_ci_lower_gt_0",
            "strong_claim_requires_ci_lower_ge_0.20": True,
            "report_all_factorial_contrasts": True,
        },
    }
    if near_duplicate is not None:
        manifest["data"]["near_duplicate_evidence"] = near_duplicate
    manifest["freeze"]["policy_bundle_sha256"] = audit.policy_bundle_sha256(
        frozen_files=manifest["frozen_files"],
        predictors=manifest["predictors"],
        physical_service_clock=manifest["execution"]["physical_service_clock"],
        policy=manifest["policy"],
        treatment_neutral_runtime_parameters=manifest["execution"][
            "treatment_neutral_runtime_parameters"
        ],
    )
    manifest["freeze"]["sealed_payload_sha256"] = audit.sealed_payload_sha256(
        manifest
    )
    (tmp_path / "FORMAL_STARTED.json").write_text(
        json.dumps(
            {
                "schema": audit.START_MARKER_SCHEMA,
                "sealed_payload_sha256": manifest["freeze"][
                    "sealed_payload_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _reseal(manifest: dict, tmp_path: Path) -> None:
    manifest["freeze"]["policy_bundle_sha256"] = audit.policy_bundle_sha256(
        frozen_files=manifest["frozen_files"],
        predictors=manifest["predictors"],
        physical_service_clock=manifest["execution"]["physical_service_clock"],
        policy=manifest["policy"],
        treatment_neutral_runtime_parameters=manifest["execution"][
            "treatment_neutral_runtime_parameters"
        ],
    )
    manifest["freeze"]["sealed_payload_sha256"] = audit.sealed_payload_sha256(
        manifest
    )
    (tmp_path / manifest["freeze"]["started_marker_path"]).write_text(
        json.dumps(
            {
                "schema": audit.START_MARKER_SCHEMA,
                "sealed_payload_sha256": manifest["freeze"][
                    "sealed_payload_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )


def _refresh_near_duplicate_evidence(manifest: dict, tmp_path: Path) -> None:
    evidence = manifest["data"]["near_duplicate_evidence"]
    root_hash = audit.registered_root_sets_sha256(
        manifest["data"]["calibration_root_ids"],
        manifest["data"]["tuning_root_ids"],
        manifest["data"]["evaluation_root_ids"],
    )
    evidence["registered_root_sets_sha256"] = root_hash
    path = tmp_path / evidence["path"]
    document = {
        field: evidence[field]
        for field in (
            "schema",
            "verified",
            "registered_root_sets_sha256",
            "method",
            "near_duplicate_pairs_across_splits",
        )
    }
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    evidence["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata() -> dict:
    return {
        "ms": audit.DECISION_SCHEMA,
        "decision_seq": 8,
        "observed_event_seq": 7,
        "policy_sha256": HEX,
        "predictor_artifact_sha256": HEX,
        "duration_predictor_artifact_sha256": HEX,
        "t": "opaque-1",
        "c": 1,
        "i": 1,
        "pt": 100,
        "mt": 128,
        "po_hat": 64,
        "tool_service_s_hat": 1.5,
    }


def _prediction() -> dict:
    return {
        "record_type": "prediction_decision",
        "trace_id": "trace-1",
        "request_index": 0,
        "decision_seq": 5,
        "observed_event_seq": 4,
        "decided_at_monotonic_s": 10.0,
        "candidate_invocation_digest": HEX,
        "predictor_artifact_sha256": HEX,
        "input": {"current_call_index": 1},
    }


def _result(
    cell: str,
    *,
    service_clock_sha256: str = HEX,
    provenance: dict | None = None,
    workload_instances: int = 1,
) -> dict:
    is_joint = cell in {"E", "F"}
    is_spec = cell in {"B", "F"}
    return {
        "schema": audit.QWEN_RESULT_SCHEMA,
        "experiment_started_monotonic_s": 1.0,
        "experiment_ended_monotonic_s": 22.0,
        "experiment_wall_s": 21.0,
        "provenance": provenance or {
            field: HEX for field in audit.RUNTIME_PROVENANCE_FIELDS
        },
        "runtime_parameters": _runtime_parameters(
            workload_instances=workload_instances
        ),
        "settings": {
            "scheduler": audit.CELLS[cell]["scheduler"],
            "tool_mechanism": (
                "online_causal_speculation" if is_spec else "demand_only"
            ),
            "call_graph_mode": "autonomous",
        },
        "paper_protocol": {
            "cell": cell,
            **audit.CELLS[cell],
            "call_graph_mode": "autonomous",
            "claim_type": "closed_loop_agent",
            "claim_scope": "confirmatory",
            "physical_service_clock_mode": "calibration_hashed_empirical_v1",
            "service_clock_artifact_sha256": service_clock_sha256,
            "service_assignment_policy_independent": True,
            "service_assignment_future_poison_invariant": True,
            "future_state_accepted_poison_invariance_test_passed": True,
            "same_invocation_service_clock_all_cells": True,
            "evaluation_trace_duration_role": "diagnostic_only",
            "offline_credit_s": 0,
            "all_tasks_successful": True,
            "broker_drained": True,
            "physical_speculative_starts": 1 if is_spec else 0,
        },
        "llm_events": (
            [{"scheduler_metadata": _metadata()}] if is_joint else []
        ),
        "prediction_decisions": [_prediction()] if is_spec else [],
        "task_results": [
            {
                "trace_id": f"trace-{index}",
                "source_session_id": f"root-{index}",
                "release_offset_s": 0.0,
                "scheduled_release_monotonic_s": 1.0,
                "released_at_monotonic_s": 1.0,
                "task_terminal_monotonic_s": 22.0,
                "release_lag_s": 0.0,
                "flow_s": 21.0,
                "failure": None,
            }
            for index in range(1, workload_instances + 1)
        ],
    }


def _outcomes(*, estimate: float = 0.21, lower: float = 0.03) -> dict:
    result = {
        name: {
            "estimand": "ratio_of_paired_root_mean_e2e",
            "ratio_of_paired_root_mean_e2e": 0.01,
            "paired_bootstrap_95_ci": [-0.01, 0.03],
        }
        for name in ("A_vs_B", "A_vs_E", "E_vs_F", "B_vs_F")
    }
    result["A_vs_F"] = {
        "estimand": "ratio_of_paired_root_mean_e2e",
        "ratio_of_paired_root_mean_e2e": estimate,
        "paired_bootstrap_95_ci": [lower, 0.34],
    }
    result["interaction"] = {
        "mean_interaction_s": 0.1,
        "paired_bootstrap_95_ci_s": [-0.2, 0.4],
    }
    return result


def test_valid_confirmatory_manifest_and_files(tmp_path: Path) -> None:
    result = audit.audit_manifest(
        _base_manifest(tmp_path),
        base=tmp_path,
        verify_files=True,
        require_evidence=False,
    )
    assert result["valid"] is True
    assert result["confirmatory_eligible"] is True


def test_retrospective_is_valid_but_never_confirmatory(tmp_path: Path) -> None:
    result = audit.audit_manifest(
        _base_manifest(tmp_path, scope="retrospective"),
        base=tmp_path,
        verify_files=True,
        require_evidence=False,
    )
    assert result["valid"] is True
    assert result["confirmatory_eligible"] is False
    assert any("near-duplicate" in warning for warning in result["warnings"])


def test_confirmatory_requires_bound_near_duplicate_evidence(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["data"]["near_duplicate_guard"] = "not_verified"
    manifest["data"].pop("near_duplicate_evidence")
    _reseal(manifest, tmp_path)
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert result["valid"] is False
    assert any("confirmatory scope requires" in error for error in result["errors"])


def test_near_duplicate_evidence_is_bound_to_registered_roots(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["data"]["near_duplicate_evidence"][
        "registered_root_sets_sha256"
    ] = "f" * 64
    _reseal(manifest, tmp_path)
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is False
    assert any("split identity mismatch" in error for error in result["errors"])


def test_empty_tuning_requires_calibration_nested_selection(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["data"]["tuning_root_ids"] = []
    manifest["data"][
        "selection_protocol"
    ] = "nested_cross_validation_within_calibration"
    _refresh_near_duplicate_evidence(manifest, tmp_path)
    _reseal(manifest, tmp_path)
    valid = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert valid["valid"] is True, valid["errors"]

    manifest["data"]["selection_protocol"] = "heldout_tuning_split"
    _reseal(manifest, tmp_path)
    invalid = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert invalid["valid"] is False
    assert any("empty tuning" in error for error in invalid["errors"])


def test_evaluation_may_not_drive_selection(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["data"]["evaluation_used_for_model_or_policy_selection"] = True
    _reseal(manifest, tmp_path)
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is False
    assert any("evaluation_used" in error for error in result["errors"])


def test_runtime_contract_and_state_poison_attestation_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["execution"]["treatment_neutral_runtime_parameters"]["parameters"][
        "server_port"
    ] = 9999
    _reseal(manifest, tmp_path)
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is False
    assert any("canonical hash mismatch" in error for error in result["errors"])

    payload = _result("A")
    payload["paper_protocol"][
        "future_state_accepted_poison_invariance_test_passed"
    ] = False
    errors = audit.audit_result_payload(payload)
    assert any("future_state_accepted" in error for error in errors)


def test_qwen_model_inventory_identity_and_cell_environment_are_bound(
    tmp_path: Path,
) -> None:
    files = [
        {
            "relative_path": "config.json",
            "size_bytes": 17,
            "content_sha256": "1" * 64,
        },
        {
            "relative_path": "model-00001.safetensors",
            "size_bytes": 1024,
            "content_sha256": "2" * 64,
        },
    ]
    identity = audit.canonical_sha256(
        {"schema": "paste_repro.model_snapshot_inventory.v1", "files": files}
    )
    inventory = {
        "schema": "paste_repro.model_snapshot_inventory.v1",
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "inventory_sha256": identity,
    }
    bundle_path = tmp_path / "qwen-bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema": "paste_repro.strict_trace_abef_bundle.v1",
                "plans": {},
                "model_snapshot_contract": {
                    "inventory": inventory,
                    "inventory_sha256": identity,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = {
        "frozen_files": [
            {
                "role": "policy_bundle",
                "path": bundle_path.name,
                "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            }
        ]
    }
    assert audit.qwen_model_snapshot_inventory_sha256(
        manifest, base=tmp_path
    ) == identity

    contract = {
        "evidence_sha256": "3" * 64,
        "model_snapshot_inventory_sha256": identity,
        "environment_scrubbed_before_cell": True,
        "server_and_client_launched_via_env_i": True,
    }
    errors: list[str] = []
    audit._audit_runtime_environment_contract(
        contract,
        expected_model_inventory_sha256=identity,
        label="$.runtime_environment_contract",
        errors=errors,
    )
    assert errors == []
    poisoned = copy.deepcopy(contract)
    poisoned["model_snapshot_inventory_sha256"] = "4" * 64
    errors = []
    audit._audit_runtime_environment_contract(
        poisoned,
        expected_model_inventory_sha256=identity,
        label="$.runtime_environment_contract",
        errors=errors,
    )
    assert any("differs from frozen policy bundle" in error for error in errors)

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["model_snapshot_contract"]["inventory"]["files"][1][
        "content_sha256"
    ] = "5" * 64
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    manifest["frozen_files"][0]["sha256"] = hashlib.sha256(
        bundle_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="canonical hash mismatch"):
        audit.qwen_model_snapshot_inventory_sha256(manifest, base=tmp_path)


def test_materializer_rejects_missing_model_inventory_provenance() -> None:
    expected = "1" * 64
    with pytest.raises(ValueError, match="missing or differs"):
        materializer._require_model_inventory_provenance(
            {}, expected=expected, label="matrix provenance"
        )
    materializer._require_model_inventory_provenance(
        {"model_snapshot_inventory_sha256": expected},
        expected=expected,
        label="matrix provenance",
    )


def test_gemini_platform_evidence_is_semantically_verified(tmp_path: Path) -> None:
    runtime_cwd = tmp_path / "empty-runtime-cwd"
    runtime_cwd.mkdir()
    runtime_cwd.chmod(0o555)
    hook_sha = "5" * 64
    site_sha = "6" * 64
    model_sha = "7" * 64
    environment = {
        "schema": "paste_gemini.strict_runtime_environment_evidence.v1",
        "cell": "F",
        "block_id": "cycle-01-block-01",
        "order_position": 2,
        "gpu_ids": [0, 1, 2, 3],
        "server_instance_id": "server-fixture",
        "server_policy": "paste_joint",
        "runtime_cwd": str(runtime_cwd.resolve()),
        "model_snapshot_inventory_sha256": model_sha,
        "frozen_environment_sha256": "8" * 64,
        "effective_environment_sha256": "9" * 64,
        "effective_environment_keys": ["PATH", "PYTHONSAFEPATH"],
        "client_environment_sha256": "a" * 64,
        "client_environment_keys": ["PATH", "PYTHONSAFEPATH"],
        "environment_scrubbed_before_cell": True,
        "server_and_client_launched_via_env_i": True,
        "client_python_isolated": True,
        "client_cwd_removed_from_sys_path": True,
        "client_python_safe_path_supported": False,
        "client_python_no_user_site": True,
        "server": {
            "runtime_cwd": str(runtime_cwd.resolve()),
            "python_safe_path_supported": False,
            "python_safe_path_requested": False,
            "python_safe_path_effective": False,
            "python_no_user_site": True,
            "pythonpath": str((tmp_path / "runtime-hooks").resolve()),
            "runtime_cwd_empty": True,
            "runtime_cwd_mode": "0o555",
            "hook_directory_mode": "0o555",
            "model_snapshot_identity_sha256": model_sha,
            "expected_sitecustomize_file_sha256": site_sha,
            "expected_scheduler_hook_file_sha256": hook_sha,
            "loaded_sitecustomize_file_sha256": site_sha,
            "loaded_scheduler_hook_file_sha256": hook_sha,
            "scheduler_runtime_pid": 123,
            "fresh_pid": 100,
            "policy": "paste_joint",
            "process_environment_sha256": "9" * 64,
            "import_path_guard": {
                "installed": True,
                "runtime_cwd": str(runtime_cwd.resolve()),
                "blocked_top_level_paths": ["", str(runtime_cwd.resolve())],
                "effective_sys_path": ["/frozen/site-packages"],
            },
        },
    }
    environment_path = tmp_path / "runtime-environment.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    platform = {"runtime_environment": _absolute_binding(environment_path)}
    for name in (
        "server",
        "smoke",
        "protocol_response",
        "machine_before",
        "machine_after",
        "scheduler_hook_load",
        "scheduler_runtime_marker",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"{name}\n", encoding="utf-8")
        platform[name] = _absolute_binding(path)
    hook_load_path = tmp_path / "scheduler_hook_load.txt"
    hook_load_path.write_text(
        json.dumps(
            {
                "schema": "paste_gemini.loaded_scheduler_hook.v1",
                "pid": 100,
                "installed": True,
                "sitecustomize_file_sha256": site_sha,
                "scheduler_hook_file_sha256": hook_sha,
                "import_path_guard": {
                    "installed": True,
                    "runtime_cwd": str(runtime_cwd.resolve()),
                    "blocked_top_level_paths": ["", str(runtime_cwd.resolve())],
                    "effective_sys_path": ["/frozen/site-packages"],
                },
            }
        ),
        encoding="utf-8",
    )
    platform["scheduler_hook_load"] = _absolute_binding(hook_load_path)
    smoke_path = tmp_path / "smoke.txt"
    smoke_path.write_text(
        json.dumps(
            {
                "cell": "F",
                "joint_runtime_pressure_marker_seen": True,
                "joint_runtime_import_path_guard_active": True,
                "joint_runtime_scheduler_pid": 123,
            }
        ),
        encoding="utf-8",
    )
    platform["smoke"] = _absolute_binding(smoke_path)
    contract = {
        "evidence_sha256": platform["runtime_environment"]["sha256"],
        "model_snapshot_inventory_sha256": model_sha,
        "environment_scrubbed_before_cell": True,
        "server_and_client_launched_via_env_i": True,
    }
    errors: list[str] = []
    audit._audit_platform_evidence(
        platform,
        base=tmp_path,
        verify_files=True,
        expected_model_inventory_sha256=model_sha,
        runtime_environment_contract=contract,
        cell="F",
        block_id="cycle-01-block-01",
        order_position=2,
        expected_provenance={"scheduler_hook_file_sha256": hook_sha},
        gpu_ids=[0, 1, 2, 3],
        server_instance_id="server-fixture",
        label="$.platform_evidence",
        errors=errors,
    )
    assert errors == []

    missing = copy.deepcopy(platform)
    missing.pop("machine_after")
    errors = []
    audit._audit_platform_evidence(
        missing,
        base=tmp_path,
        verify_files=True,
        expected_model_inventory_sha256=model_sha,
        runtime_environment_contract=contract,
        cell="F",
        block_id="cycle-01-block-01",
        order_position=2,
        expected_provenance={"scheduler_hook_file_sha256": hook_sha},
        gpu_ids=[0, 1, 2, 3],
        server_instance_id="server-fixture",
        label="$.platform_evidence",
        errors=errors,
    )
    assert any("machine_after" in error for error in errors)


def test_qwen_scheduler_runtime_marker_is_semantically_verified(tmp_path: Path) -> None:
    hook = tmp_path / "scheduler-hook.py"
    hook.write_text("# frozen hook\n", encoding="utf-8")
    hook_sha = hashlib.sha256(hook.read_bytes()).hexdigest()
    marker = {
        "schema": "paste.vllm.scheduler_runtime_use.v1",
        "policy": "online_joint_pacer_v2",
        "scheduler_api": "v1.Scheduler.schedule",
        "scheduler_hook_sha256": hook_sha,
        "python_safe_path_enforced": True,
        "cwd_import_filter_enforced": True,
        "working_directory_importable": False,
        "safe_working_directory": str(tmp_path.resolve()),
        "working_directory": str(tmp_path.resolve()),
        "pid": 222,
    }
    documents = {}
    for name, phase in (
        ("scheduler_runtime_after_smoke", "after_standardized_smoke"),
        ("scheduler_runtime_after_cell", "after_evaluation_cell"),
    ):
        document = {
            "schema": "paste.paper.scheduler_runtime_evidence.v1",
            "cell": "F",
            "phase": phase,
            "server_pid": 111,
            "expected_policy": "online_joint_pacer_v2",
            "hook_runtime_use_expected": True,
            "patched_scheduler_invocation_verified": True,
            "no_scheduler_hook_runtime_use_verified": False,
            "scheduler_hook_path": str(hook.resolve()),
            "scheduler_hook_sha256": hook_sha,
            "runtime_marker_sha256": "b" * 64,
            "scheduler_calling_pid": 222,
            "scheduler_calling_process_relation": "server_descendant",
            "runtime_marker": marker,
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        documents[name] = _absolute_binding(path)
    contract = {
        "evidence_sha256": documents["scheduler_runtime_after_smoke"]["sha256"],
        "hook_runtime_use_expected": True,
        "patched_scheduler_invocation_verified": True,
        "no_scheduler_hook_runtime_use_verified": False,
        "expected_policy": "online_joint_pacer_v2",
        "scheduler_calling_pid": 222,
        "scheduler_calling_process_relation": "server_descendant",
        "runtime_marker_sha256": "b" * 64,
    }
    errors: list[str] = []
    audit._audit_qwen_scheduler_runtime_evidence(
        contract,
        platform=documents,
        base=tmp_path,
        verify_files=True,
        cell="F",
        expected_scheduler_hook_sha256=hook_sha,
        label="$.scheduler_runtime_contract",
        errors=errors,
    )
    assert errors == []

    bad = copy.deepcopy(documents)
    post_path = Path(bad["scheduler_runtime_after_cell"]["path"])
    post = json.loads(post_path.read_text(encoding="utf-8"))
    post["runtime_marker"]["scheduler_api"] = "fake.schedule"
    post_path.write_text(json.dumps(post), encoding="utf-8")
    bad["scheduler_runtime_after_cell"] = _absolute_binding(post_path)
    errors = []
    audit._audit_qwen_scheduler_runtime_evidence(
        contract,
        platform=bad,
        base=tmp_path,
        verify_files=True,
        cell="F",
        expected_scheduler_hook_sha256=hook_sha,
        label="$.scheduler_runtime_contract",
        errors=errors,
    )
    assert any("Scheduler.schedule" in error for error in errors)


def test_gemini_model_inventory_identity_is_recognized_and_verified(
    tmp_path: Path,
) -> None:
    inventory_body = {
        "schema": "paste.paper.model_snapshot_inventory.v1",
        "snapshot_path": str(tmp_path / "snapshot"),
        "snapshot_resolved_path": str(tmp_path / "snapshot"),
        "files": [
            {
                "relative_path": "config.json",
                "symlink_target": None,
                "resolved_path": str(tmp_path / "snapshot" / "config.json"),
                "size_bytes": 17,
                "sha256": "1" * 64,
            },
            {
                "relative_path": "model.safetensors",
                "symlink_target": "../../blobs/model",
                "resolved_path": str(tmp_path / "blobs" / "model"),
                "size_bytes": 1024,
                "sha256": "2" * 64,
            },
        ],
        "file_count": 2,
        "total_bytes": 1041,
    }
    identity = audit.canonical_sha256(inventory_body)
    policy_path = tmp_path / "gemini-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema": "paste_gemini.swe_strict_policy_plan.v1",
                "sessions": [],
                "templates": [],
                "model_snapshot_contract": {
                    **inventory_body,
                    "identity_sha256": identity,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = {
        "frozen_files": [
            {
                "role": "policy_bundle",
                "path": policy_path.name,
                "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            }
        ]
    }
    assert audit.model_snapshot_inventory_sha256(
        manifest, base=tmp_path
    ) == identity

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["model_snapshot_contract"]["files"][0]["sha256"] = "3" * 64
    policy_path.write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")
    manifest["frozen_files"][0]["sha256"] = hashlib.sha256(
        policy_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="canonical hash mismatch"):
        audit.model_snapshot_inventory_sha256(manifest, base=tmp_path)


def test_causal_trace_replay_is_a_valid_systems_claim(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["policy"]["call_graph_mode"] = "trace_replay_causal_reveal"
    manifest["policy"]["claim_type"] = "systems_trace_replay"
    _reseal(manifest, tmp_path)
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is True
    assert result["confirmatory_eligible"] is True

    manifest["policy"]["call_graph_mode"] = "frozen"
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is False


def test_causal_reveal_result_requires_ordered_raw_events() -> None:
    payload = _result("F")
    payload["schema"] = audit.GEMINI_RESULT_SCHEMA
    payload["settings"]["call_graph_mode"] = "trace_replay_causal_reveal"
    payload["paper_protocol"].update(
        {
            "call_graph_mode": "trace_replay_causal_reveal",
            "claim_type": "systems_trace_replay",
        }
    )
    payload["llm_events"][0].update(
        {
            "trace_id": "trace-1",
            "request_index": 0,
            "llm_completed_seq": 10,
            "llm_completed_at_monotonic_s": 20.0,
        }
    )
    payload["prediction_decisions"][0].update(
        {
            "prediction_id": "prediction-1",
            "speculative_start_seq": None,
            "speculative_start_at_monotonic_s": None,
            "candidates": [
                {
                    "candidate_invocation_digest": HEX,
                    "broker_accepted": True,
                    "admitted": True,
                }
            ],
        }
    )
    payload["speculation_execution_events"] = [
        {
            "prediction_id": "prediction-1",
            "trace_id": "trace-1",
            "request_index": 0,
            "candidate_invocation_digest": HEX,
            "job_id": "physical-job-1",
            "admitted_at_monotonic_s": 10.05,
            "physical_started_at_monotonic_s": 10.1,
            "terminal_at_monotonic_s": 21.0,
            "terminal_state": "completed_claimed",
            "assigned_service_s": 10.9,
            "service_s": 10.9,
            "total_worker_service_s": 10.9,
            "speculative_resource_s": 9.9,
            "demand_resource_s": 1.0,
            "authority_claimed_at_monotonic_s": 20.0,
            "claimed_by_authority": True,
            "state_transitions": [
                {
                    "event": "admitted",
                    "at_monotonic_s": 10.05,
                    "authority_claimed_at_monotonic_s": None,
                },
                {
                    "event": "physical_started",
                    "at_monotonic_s": 10.1,
                    "authority_claimed_at_monotonic_s": None,
                },
                {
                    "event": "authority_claimed_inflight",
                    "at_monotonic_s": 20.0,
                    "authority_claimed_at_monotonic_s": 20.0,
                },
                {
                    "event": "completed",
                    "at_monotonic_s": 21.0,
                    "authority_claimed_at_monotonic_s": 20.0,
                },
            ],
        }
    ]
    payload["worker_resource_accounting"] = {
        "speculative_resource_s": 9.9,
        "promoted_demand_resource_s": 1.0,
        "direct_demand_resource_s": 0.5,
        "total_worker_occupancy_s": 11.4,
    }
    payload["broker"] = {
        "metrics": {"physical_speculative_starts": 1},
        **payload["worker_resource_accounting"],
    }
    payload["tool_events"] = [
        {
            "trace_id": "trace-1",
            "request_index": 0,
            "outcome_id": "sealed-outcome-1",
            "llm_completed_seq": 10,
            "llm_completed_at_monotonic_s": 20.0,
            "authoritative_revealed_seq": 11,
            "authoritative_revealed_at_monotonic_s": 20.0,
            "tool_completed_seq": 12,
            "tool_completed_at_monotonic_s": 21.0,
            "cache_source": "executed",
            "worker_service_s": 0.5,
            "authority_key_sha256": HEX,
            "pool_authority_key_sha256": HEX,
            "physical_service_key_sha256": HEX,
            "execution_surface_service_s": 10.9,
        }
    ]
    payload["prediction_outcomes"] = [
        {
            "prediction_id": "prediction-1",
            "trace_id": "trace-1",
            "request_index": 0,
            "candidate_invocation_digest": HEX,
            "broker_accepted": True,
            "admitted": True,
            "post_authority_hit": True,
        }
    ]
    assert audit.audit_result_payload(payload) == []

    gemini_overwritten_projection = copy.deepcopy(payload)
    gemini_overwritten_projection["speculation_execution_events"][0][
        "authority_claimed_at_monotonic_s"
    ] = 20.5
    gemini_projection_errors = audit.audit_result_payload(
        gemini_overwritten_projection
    )
    assert any(
        "top-level value differs from raw transitions" in error
        for error in gemini_projection_errors
    )

    gemini_poisoned_transition = copy.deepcopy(payload)
    gemini_poisoned_transition["speculation_execution_events"][0][
        "state_transitions"
    ][-1]["authority_claimed_at_monotonic_s"] = 20.000001
    gemini_transition_errors = audit.audit_result_payload(
        gemini_poisoned_transition
    )
    assert any(
        "first authority claim changed across callbacks" in error
        for error in gemini_transition_errors
    )

    gemini_missing_claim_field = copy.deepcopy(payload)
    gemini_missing_claim_field["speculation_execution_events"][0][
        "state_transitions"
    ][0].pop("authority_claimed_at_monotonic_s")
    gemini_missing_errors = audit.audit_result_payload(gemini_missing_claim_field)
    assert any(
        "explicit null or immutable first-claim timestamp required" in error
        for error in gemini_missing_errors
    )

    gemini_missing_boundary = copy.deepcopy(payload)
    del gemini_missing_boundary["speculation_execution_events"][0][
        "state_transitions"
    ][2]
    gemini_boundary_errors = audit.audit_result_payload(gemini_missing_boundary)
    assert any(
        "no raw authority event establishes the first-claim boundary" in error
        for error in gemini_boundary_errors
    )

    admission_after_start = copy.deepcopy(payload)
    admission_after_start["speculation_execution_events"][0][
        "admitted_at_monotonic_s"
    ] = 10.2
    admission_after_start["speculation_execution_events"][0][
        "state_transitions"
    ][0]["at_monotonic_s"] = 10.2
    admission_order_errors = audit.audit_result_payload(admission_after_start)
    assert any(
        "physical start predates broker admission" in error
        for error in admission_order_errors
    )

    gemini_queued_without_start = copy.deepcopy(payload)
    gemini_execution = gemini_queued_without_start[
        "speculation_execution_events"
    ][0]
    gemini_execution.update(
        {
            "physical_started_at_monotonic_s": None,
            "terminal_at_monotonic_s": 19.0,
            "service_s": 0.0,
            "total_worker_service_s": 0.0,
            "speculative_resource_s": 0.0,
            "demand_resource_s": 0.0,
            "authority_claimed_at_monotonic_s": None,
            "claimed_by_authority": False,
            "terminal_state": "cancelled_before_start",
        }
    )
    gemini_execution["state_transitions"] = [
        {
            "event": "admitted",
            "at_monotonic_s": 10.05,
            "authority_claimed_at_monotonic_s": None,
        },
        {
            "event": "cancelled_authority_superseded",
            "at_monotonic_s": 19.0,
            "authority_claimed_at_monotonic_s": None,
        },
    ]
    gemini_queued_without_start["paper_protocol"][
        "physical_speculative_starts"
    ] = 0
    gemini_queued_without_start["worker_resource_accounting"].update(
        {
            "speculative_resource_s": 0.0,
            "promoted_demand_resource_s": 0.0,
            "total_worker_occupancy_s": 0.5,
        }
    )
    gemini_queued_without_start["broker"].update(
        {
            "speculative_resource_s": 0.0,
            "promoted_demand_resource_s": 0.0,
            "total_worker_occupancy_s": 0.5,
        }
    )
    gemini_queued_without_start["broker"]["metrics"][
        "physical_speculative_starts"
    ] = 0
    gemini_start_errors = audit.audit_result_payload(gemini_queued_without_start)
    assert any("Gemini immediate-start" in error for error in gemini_start_errors)

    mislabeled = copy.deepcopy(payload)
    mislabeled["prediction_outcomes"][0]["post_authority_hit"] = False
    label_errors = audit.audit_result_payload(mislabeled)
    assert any("raw pool-key match" in error for error in label_errors)

    qwen_resolution = copy.deepcopy(payload)
    qwen_resolution["schema"] = audit.QWEN_RESULT_SCHEMA
    qwen_resolution["tool_events"][0].pop("authority_key_sha256", None)
    qwen_resolution["tool_events"][0].pop("pool_authority_key_sha256", None)
    qwen_resolution["tool_events"][0].pop("physical_service_key_sha256", None)
    qwen_resolution["tool_events"][0].pop("execution_surface_service_s", None)
    qwen_resolution["tool_events"][0]["assigned_service_s"] = 10.9
    qwen_resolution["tool_events"][0][
        "authority_candidate_invocation_digests"
    ] = [HEX]
    qwen_resolution["tool_events"][0]["authority_invocation_digest"] = HEX
    qwen_resolution["speculation_execution_events"][0]["state_transitions"] = [
        {
            "event": "admitted",
            "at_monotonic_s": 10.05,
            "authority_claimed_at_monotonic_s": None,
        },
        {
            "event": "physical_started",
            "at_monotonic_s": 10.1,
            "authority_claimed_at_monotonic_s": None,
        },
        {
            "event": "authority_claimed_inflight",
            "at_monotonic_s": 20.0,
            "authority_claimed_at_monotonic_s": 20.0,
        },
        {
            "event": "completed",
            "at_monotonic_s": 21.0,
            "authority_claimed_at_monotonic_s": 20.0,
        },
    ]
    qwen_resolution["prediction_outcomes"] = [
        {
            "prediction_id": "prediction-1",
            "trace_id": "trace-1",
            "request_index": 0,
            "admitted_semantics": "broker_accepted_not_physical_start",
            "authoritative_invocation_digests": [HEX],
            "authoritative_candidate_invocation_digests": [HEX],
            "candidates": [
                {
                    "candidate_invocation_digest": HEX,
                    "broker_accepted": True,
                    "admitted": True,
                    "matched_authority": True,
                }
            ],
            "emitted_candidate_count": 1,
            "broker_accepted_candidate_count": 1,
            "physical_started_candidate_count": 1,
            "admitted_candidate_count": 1,
            "matched_emitted_candidate_count": 1,
            "matched_broker_accepted_candidate_count": 1,
            "matched_physical_started_candidate_count": 1,
            "matched_admitted_candidate_count": 1,
            "decision_hit": True,
        }
    ]
    assert audit.audit_result_payload(qwen_resolution) == []

    # A later reuse of the same completed cached job is not a new resource
    # boundary.  Every callback retains the immutable first claim, and the raw
    # transition ledger independently reconstructs the original split.
    repeated_completed_reuse = copy.deepcopy(qwen_resolution)
    repeated_execution = repeated_completed_reuse[
        "speculation_execution_events"
    ][0]
    repeated_execution.update(
        {
            "terminal_at_monotonic_s": 20.5,
            "service_s": 10.4,
            "total_worker_service_s": 10.4,
            "speculative_resource_s": 9.9,
            "demand_resource_s": 0.5,
        }
    )
    repeated_execution["state_transitions"][-1].update(
        {"at_monotonic_s": 20.5}
    )
    repeated_execution["state_transitions"].append(
        {
            "event": "authority_claimed_completed",
            "at_monotonic_s": 20.8,
            "authority_claimed_at_monotonic_s": 20.0,
        }
    )
    repeated_completed_reuse["worker_resource_accounting"].update(
        {
            "speculative_resource_s": 9.9,
            "promoted_demand_resource_s": 0.5,
            "total_worker_occupancy_s": 10.9,
        }
    )
    repeated_completed_reuse["broker"].update(
        repeated_completed_reuse["worker_resource_accounting"]
    )
    assert audit.audit_result_payload(repeated_completed_reuse) == []

    overwritten_projection = copy.deepcopy(repeated_completed_reuse)
    overwritten_projection["speculation_execution_events"][0][
        "authority_claimed_at_monotonic_s"
    ] = 20.8
    overwritten_errors = audit.audit_result_payload(overwritten_projection)
    assert any(
        "top-level value differs from raw transitions" in error
        for error in overwritten_errors
    )

    poisoned_transition = copy.deepcopy(repeated_completed_reuse)
    poisoned_transition["speculation_execution_events"][0]["state_transitions"][-1][
        "authority_claimed_at_monotonic_s"
    ] = 20.8
    poisoned_errors = audit.audit_result_payload(poisoned_transition)
    assert any(
        "first authority claim changed across callbacks" in error
        for error in poisoned_errors
    )

    submillisecond_poison = copy.deepcopy(repeated_completed_reuse)
    submillisecond_poison["speculation_execution_events"][0][
        "state_transitions"
    ][-1]["authority_claimed_at_monotonic_s"] = 20.000001
    submillisecond_errors = audit.audit_result_payload(submillisecond_poison)
    assert any(
        "first authority claim changed across callbacks" in error
        for error in submillisecond_errors
    )

    deleted_claim_field = copy.deepcopy(repeated_completed_reuse)
    deleted_claim_field["speculation_execution_events"][0]["state_transitions"][
        2
    ].pop("authority_claimed_at_monotonic_s")
    deleted_claim_errors = audit.audit_result_payload(deleted_claim_field)
    assert any(
        "explicit null or immutable first-claim timestamp required" in error
        for error in deleted_claim_errors
    )

    deleted_first_claim_event = copy.deepcopy(repeated_completed_reuse)
    del deleted_first_claim_event["speculation_execution_events"][0][
        "state_transitions"
    ][2]
    deleted_event_errors = audit.audit_result_payload(deleted_first_claim_event)
    assert any(
        "no raw authority event establishes the first-claim boundary" in error
        for error in deleted_event_errors
    )

    # A broker may accept work that expires in its queue.  That candidate is
    # present in the terminal execution ledger but has no physical-start clock,
    # so it must not enter the admitted/physical-start precision denominator.
    queued_not_started = copy.deepcopy(qwen_resolution)
    queued_digest = "2" * 64
    queued_not_started["prediction_decisions"][0]["candidates"].append(
        {
            "candidate_invocation_digest": queued_digest,
            "broker_accepted": True,
            "admitted": True,
        }
    )
    queued_not_started["speculation_execution_events"].append(
        {
            "prediction_id": "prediction-1",
            "trace_id": "trace-1",
            "request_index": 0,
            "candidate_invocation_digest": queued_digest,
            "job_id": "queued-job-never-started",
            "admitted_at_monotonic_s": 10.06,
            "physical_started_at_monotonic_s": None,
            "terminal_at_monotonic_s": 19.0,
            "terminal_state": "cancelled_before_start",
            "assigned_service_s": 7.0,
            "service_s": 0.0,
            "total_worker_service_s": 0.0,
            "speculative_resource_s": 0.0,
            "demand_resource_s": 0.0,
            "authority_claimed_at_monotonic_s": None,
            "claimed_by_authority": False,
            "state_transitions": [
                {
                    "event": "admitted",
                    "at_monotonic_s": 10.06,
                    "authority_claimed_at_monotonic_s": None,
                },
                {
                    "event": "cancelled_window_expired",
                    "at_monotonic_s": 19.0,
                    "authority_claimed_at_monotonic_s": None,
                },
            ],
        }
    )
    queued_outcome = queued_not_started["prediction_outcomes"][0]
    queued_outcome["candidates"].append(
        {
            "candidate_invocation_digest": queued_digest,
            "broker_accepted": True,
            "admitted": True,
            "matched_authority": False,
        }
    )
    queued_outcome.update(
        {
            "emitted_candidate_count": 2,
            "broker_accepted_candidate_count": 2,
            "physical_started_candidate_count": 1,
            "admitted_candidate_count": 2,
            "matched_emitted_candidate_count": 1,
            "matched_broker_accepted_candidate_count": 1,
            "matched_physical_started_candidate_count": 1,
            "matched_admitted_candidate_count": 1,
        }
    )
    queued_not_started["prediction_metrics"] = {
        "admitted_metric_semantics": (
            "physical_started_at_monotonic_s_is_not_null"
        ),
        "decisions_with_candidates": 1,
        "decision_hits": 1,
        "emitted_candidates": 2,
        "broker_accepted_candidates": 2,
        "physical_started_candidates": 1,
        "admitted_candidates": 1,
        "matched_emitted_candidates": 1,
        "matched_broker_accepted_candidates": 1,
        "matched_physical_started_candidates": 1,
        "matched_admitted_candidates": 1,
        "emitted_candidate_precision": 0.5,
        "broker_accepted_candidate_precision": 0.5,
        "physical_started_candidate_precision": 1.0,
        "admitted_candidate_precision": 1.0,
    }
    assert audit.audit_result_payload(queued_not_started) == []

    queue_conditioned_tamper = copy.deepcopy(queued_not_started)
    queue_conditioned_tamper["prediction_metrics"]["admitted_candidates"] = 2
    queue_conditioned_tamper["prediction_metrics"][
        "admitted_candidate_precision"
    ] = 0.5
    queue_errors = audit.audit_result_payload(queue_conditioned_tamper)
    assert any("admitted_candidates" in error for error in queue_errors)
    assert any("admitted_candidate_precision" in error for error in queue_errors)

    mislabeled_semantics = copy.deepcopy(queued_not_started)
    mislabeled_semantics["prediction_metrics"][
        "admitted_metric_semantics"
    ] = "broker_accepted"
    semantics_errors = audit.audit_result_payload(mislabeled_semantics)
    assert any("physical-start-conditioned" in error for error in semantics_errors)

    late_start = copy.deepcopy(queued_not_started)
    late_start["speculation_execution_events"][1].update(
        {
            "physical_started_at_monotonic_s": 20.0,
            "terminal_at_monotonic_s": 20.0,
        }
    )
    late_start["speculation_execution_events"][1]["state_transitions"] = [
        {
            "event": "admitted",
            "at_monotonic_s": 10.06,
            "authority_claimed_at_monotonic_s": None,
        },
        {
            "event": "physical_started",
            "at_monotonic_s": 20.0,
            "authority_claimed_at_monotonic_s": None,
        },
        {
            "event": "cancelled_window_expired",
            "at_monotonic_s": 20.0,
            "authority_claimed_at_monotonic_s": None,
        },
    ]
    late_errors = audit.audit_result_payload(late_start)
    assert any("did not start before LLM completion" in error for error in late_errors)

    post_llm_queue_acceptance = copy.deepcopy(queued_not_started)
    post_llm_queue_acceptance["speculation_execution_events"][1].update(
        {
            "admitted_at_monotonic_s": 20.1,
            "terminal_at_monotonic_s": 20.2,
        }
    )
    post_llm_admission_errors = audit.audit_result_payload(
        post_llm_queue_acceptance
    )
    assert any(
        "broker admission did not precede LLM completion" in error
        for error in post_llm_admission_errors
    )

    qwen_resolution["tool_events"][0][
        "authority_candidate_invocation_digests"
    ] = []
    qwen_errors = audit.audit_result_payload(qwen_resolution)
    assert any("raw authoritative tool-event candidates" in error for error in qwen_errors)

    deleted_authority = copy.deepcopy(qwen_resolution)
    deleted_authority["tool_events"][0].pop("authority_invocation_digest", None)
    deleted_authority["tool_events"][0].pop(
        "authority_candidate_invocation_digests", None
    )
    deleted_authority["prediction_outcomes"][0][
        "authoritative_candidate_invocation_digests"
    ] = []
    deleted_authority["prediction_outcomes"][0]["candidates"][0][
        "matched_authority"
    ] = False
    deleted_authority["prediction_outcomes"][0][
        "matched_emitted_candidate_count"
    ] = 0
    deleted_authority["prediction_outcomes"][0][
        "matched_admitted_candidate_count"
    ] = 0
    deleted_authority["prediction_outcomes"][0]["decision_hit"] = False
    deletion_errors = audit.audit_result_payload(deleted_authority)
    assert any("required for Qwen raw authority" in error for error in deletion_errors)
    assert any("required for Qwen raw precision" in error for error in deletion_errors)

    deleted_pool_key = copy.deepcopy(payload)
    deleted_pool_key["tool_events"][0].pop("authority_key_sha256", None)
    deleted_pool_key["tool_events"][0].pop("pool_authority_key_sha256", None)
    deleted_pool_key["tool_events"][0].pop("physical_service_key_sha256", None)
    deleted_pool_key["prediction_outcomes"][0]["post_authority_hit"] = False
    pool_errors = audit.audit_result_payload(deleted_pool_key)
    assert any("required for Gemini raw authority" in error for error in pool_errors)
    assert any("required for Gemini raw precision" in error for error in pool_errors)
    assert any(
        "required for Gemini policy-independent service" in error
        for error in pool_errors
    )

    duration_tamper = copy.deepcopy(payload)
    duration_tamper["tool_events"][0].update(
        {
            "authority_eta_hat_s": 10.0,
            "duration_prediction_absolute_error_s": 0.1,
        }
    )
    duration_errors = audit.audit_result_payload(duration_tamper)
    assert any("raw prediction/assigned-service pair" in error for error in duration_errors)

    inconsistent_clock = copy.deepcopy(payload)
    inconsistent_clock["speculation_execution_events"][0]["assigned_service_s"] = 9.9
    clock_errors = audit.audit_result_payload(inconsistent_clock)
    assert any("same physical invocation key" in error for error in clock_errors)

    payload["worker_resource_accounting"]["direct_demand_resource_s"] = 0.75
    payload["worker_resource_accounting"]["total_worker_occupancy_s"] = 11.65
    payload["broker"].update(payload["worker_resource_accounting"])
    direct_errors = audit.audit_result_payload(payload)
    assert any("raw executed authoritative" in error for error in direct_errors)
    payload["worker_resource_accounting"]["direct_demand_resource_s"] = 0.5
    payload["worker_resource_accounting"]["total_worker_occupancy_s"] = 11.4
    payload["broker"].update(payload["worker_resource_accounting"])

    payload["tool_events"][0]["authoritative_revealed_seq"] = 9
    payload["tool_events"][0]["authoritative_revealed_at_monotonic_s"] = 19.0
    errors = audit.audit_result_payload(payload)
    assert any("revealed before live LLM completion" in error for error in errors)

    payload["tool_events"][0]["authoritative_revealed_seq"] = 11
    payload["tool_events"][0]["authoritative_revealed_at_monotonic_s"] = 20.0
    payload["speculation_execution_events"][0]["speculative_resource_s"] = 10.9
    payload["speculation_execution_events"][0]["demand_resource_s"] = 0.0
    payload["worker_resource_accounting"]["speculative_resource_s"] = 10.9
    payload["worker_resource_accounting"]["promoted_demand_resource_s"] = 0.0
    payload["broker"].update(payload["worker_resource_accounting"])
    split_errors = audit.audit_result_payload(payload)
    assert any("start-to-claim" in error for error in split_errors)
    assert any("claim-to-terminal" in error for error in split_errors)


def test_qwen_aggregator_keeps_first_claim_across_completed_reuse() -> None:
    import run_strict_trace_abef as qwen_runner

    started = 10.1
    claimed = 14.0
    terminal = 15.0
    speculative = claimed - started
    demand = terminal - claimed
    total = terminal - started

    def transition(
        event: str,
        at: float,
        *,
        state: str,
        claim: float | None,
        speculative_s: float,
        demand_s: float,
        total_s: float,
        claimed_by_authority: bool,
    ) -> dict:
        return {
            "prediction_id": "prediction-1",
            "trace_id": "trace-1",
            "request_index": 0,
            "candidate_invocation_digest": HEX,
            "job_id": 1,
            "event": event,
            "at_monotonic_s": at,
            "state": state,
            "authority_claimed_at_monotonic_s": claim,
            "assigned_service_s": total,
            "speculative_resource_s": speculative_s,
            "demand_resource_s": demand_s,
            "total_worker_service_s": total_s,
            "service_s": total_s,
            "claimed_by_authority": claimed_by_authority,
        }

    transitions = [
        transition(
            "admitted",
            10.0,
            state="queued",
            claim=None,
            speculative_s=0.0,
            demand_s=0.0,
            total_s=0.0,
            claimed_by_authority=False,
        ),
        transition(
            "physical_started",
            started,
            state="running",
            claim=None,
            speculative_s=0.0,
            demand_s=0.0,
            total_s=0.0,
            claimed_by_authority=False,
        ),
        transition(
            "authority_claimed_inflight",
            claimed,
            state="running",
            claim=claimed,
            speculative_s=speculative,
            demand_s=0.0,
            total_s=speculative,
            claimed_by_authority=True,
        ),
        transition(
            "completed",
            terminal,
            state="completed",
            claim=claimed,
            speculative_s=speculative,
            demand_s=demand,
            total_s=total,
            claimed_by_authority=True,
        ),
        # The same result is read again after completion.  Its event time is
        # later, but its immutable resource boundary remains the first claim.
        transition(
            "authority_claimed_completed",
            20.0,
            state="completed",
            claim=claimed,
            speculative_s=speculative,
            demand_s=demand,
            total_s=total,
            claimed_by_authority=True,
        ),
    ]
    rows = qwen_runner.aggregate_speculation_execution_events(transitions)
    assert len(rows) == 1
    row = rows[0]
    assert row["authority_claimed_at_monotonic_s"] == claimed
    assert row["speculative_resource_s"] == speculative
    assert row["demand_resource_s"] == demand
    assert all(
        "authority_claimed_at_monotonic_s" in state
        for state in row["state_transitions"]
    )

    overwritten = copy.deepcopy(transitions)
    overwritten[-1]["authority_claimed_at_monotonic_s"] = 20.0
    with pytest.raises(RuntimeError, match="first authority claim changed"):
        qwen_runner.aggregate_speculation_execution_events(overwritten)

    missing_claim_evidence = copy.deepcopy(transitions)
    missing_claim_evidence[0].pop("authority_claimed_at_monotonic_s")
    with pytest.raises(RuntimeError, match="lacks explicit first authority-claim"):
        qwen_runner.aggregate_speculation_execution_events(missing_claim_evidence)


def test_service_clock_rejects_future_dependent_assignment(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["execution"]["physical_service_clock"][
        "future_authority_hit_invariant"
    ] = False
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is False
    assert any("future_authority_hit_invariant" in error for error in result["errors"])


def test_confirmatory_rejects_exposed_or_overlapping_roots(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["data"]["previously_observed_evaluation_root_ids"] = ["eval-01"]
    manifest["data"]["tuning_root_ids"].append("eval-02")
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is False
    assert result["confirmatory_eligible"] is False
    assert any("overlap" in error for error in result["errors"])
    assert any("confirmatory is impossible" in error for error in result["errors"])


def test_rejects_unbalanced_order_and_no_gpu_swap(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    for block in manifest["execution"]["blocks"]:
        block["order"] = list(audit.WILLIAMS_ORDERS[0])
        block["gpu_ids"] = [0, 1, 2, 3]
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is False
    assert any("Williams" in error for error in result["errors"])
    assert any("GPU groups" in error for error in result["errors"])


def test_result_rejects_legacy_oracle_metadata() -> None:
    payload = _result("F")
    meta = payload["llm_events"][0]["scheduler_metadata"]
    meta.update({"ms": "paste.schedx.remaining_llm_work.v1", "rlmt": 999, "eg": 4.0})
    errors = audit.audit_result_payload(payload)
    assert any("legacy future-trace fields" in error for error in errors)
    assert any("scheduler metadata schema" in error for error in errors)


def test_prediction_decision_rejects_nested_outcome_and_duration() -> None:
    payload = _result("B")
    payload["prediction_decisions"][0]["input"].update(
        {"prediction_hit": True, "duration_s": 2.0}
    )
    errors = audit.audit_result_payload(payload)
    assert any("prediction decision contains outcomes" in error for error in errors)

    payload = _result("B")
    payload["prediction_decisions"][0]["candidates"] = [
        {"candidate_invocation_digest": HEX, "predicted_service_s": 2.0}
    ]
    errors = audit.audit_result_payload(payload)
    assert any("require explicit *_hat" in error for error in errors)


def test_result_requires_normalized_evidence() -> None:
    assert any(
        "paper_protocol" in error
        for error in audit.audit_result_payload({"settings": {}})
    )


def test_result_task_e2e_is_derived_from_raw_monotonic_clocks() -> None:
    payload = _result("A")
    assert not audit._audit_task_timing_evidence(payload)

    summary_poisoned = copy.deepcopy(payload)
    summary_poisoned["task_results"][0]["flow_s"] = 0.01
    errors = audit.audit_result_payload(summary_poisoned)
    assert any("terminal-minus-scheduled" in error for error in errors)

    origin_poisoned = copy.deepcopy(payload)
    origin_poisoned["task_results"][0]["scheduled_release_monotonic_s"] = 1.5
    errors = audit.audit_result_payload(origin_poisoned)
    assert any("experiment start plus release_offset_s" in error for error in errors)

    order_poisoned = copy.deepcopy(payload)
    order_poisoned["task_results"][0]["released_at_monotonic_s"] = 0.5
    errors = audit.audit_result_payload(order_poisoned)
    assert any("released before" in error for error in errors)

    event_poisoned = copy.deepcopy(payload)
    event_poisoned["llm_events"] = [
        {
            "trace_id": "trace-1",
            "llm_completed_at_monotonic_s": 23.0,
        }
    ]
    errors = audit.audit_result_payload(event_poisoned)
    assert any("outside scheduled-to-terminal" in error for error in errors)

    nonfinite = copy.deepcopy(payload)
    nonfinite["task_results"][0]["task_terminal_monotonic_s"] = float("nan")
    errors = audit.audit_result_payload(nonfinite)
    assert any("non-negative finite" in error for error in errors)


def test_twenty_percent_and_strong_claim_rules(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["outcomes"] = _outcomes()
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["valid"] is True
    assert result["speedup_20_pass"] is True
    assert result["strong_20_claim_pass"] is False

    manifest["outcomes"]["A_vs_F"]["paired_bootstrap_95_ci"] = [0.20, 0.35]
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=False, require_evidence=False
    )
    assert result["strong_20_claim_pass"] is True


def test_complete_cell_evidence_is_hash_checked_and_scanned(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    evidence = []
    service_clock_sha256 = audit.artifact_identity_sha256(
        manifest["execution"]["physical_service_clock"]["artifact"]
    )
    provenance = audit.expected_runtime_provenance(manifest)
    for block_index, block in enumerate(manifest["execution"]["blocks"]):
        for cell in audit.CELLS:
            position = block["order"].index(cell) + 1
            started_wall_s = 1_000.0 + block_index * 100.0 + position * 10.0
            ended_wall_s = started_wall_s + 5.0
            result_path = tmp_path / f"{block['block_id']}-{cell}.json"
            result_payload = _result(
                cell,
                service_clock_sha256=service_clock_sha256,
                provenance=provenance,
                workload_instances=manifest["execution"][
                    "treatment_neutral_runtime_parameters"
                ]["parameters"]["workload_instances"],
            )
            result_payload.update(
                {
                    "block_id": block["block_id"],
                    "order_position": position,
                    "started_wall_s": started_wall_s,
                    "ended_wall_s": ended_wall_s,
                    "gpu_ids": block["gpu_ids"],
                    "server_instance_id": f"server-{block['block_id']}-{cell}",
                    "broker_instance_id": f"broker-{block['block_id']}-{cell}",
                }
            )
            result_path.write_text(
                json.dumps(result_payload, sort_keys=True),
                encoding="utf-8",
            )
            evidence.append(
                {
                    "block_id": block["block_id"],
                    "cell": cell,
                    "order_position": position,
                    "started_wall_s": started_wall_s,
                    "ended_wall_s": ended_wall_s,
                    "gpu_ids": block["gpu_ids"],
                    "server_instance_id": f"server-{block['block_id']}-{cell}",
                    "broker_instance_id": f"broker-{block['block_id']}-{cell}",
                    "policy_bundle_sha256": manifest["freeze"][
                        "policy_bundle_sha256"
                    ],
                    "service_clock_artifact_sha256": service_clock_sha256,
                    "runtime_parameters_sha256": manifest["execution"][
                        "treatment_neutral_runtime_parameters"
                    ]["runtime_parameters_sha256"],
                    "provenance": provenance,
                    "result_path": result_path.name,
                    "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                }
            )
    manifest["cell_evidence"] = evidence
    valid = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert valid["valid"] is True, valid["errors"]

    evidence[0]["result_sha256"] = "b" * 64
    invalid = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert invalid["valid"] is False
    assert any("SHA-256 mismatch" in error for error in invalid["errors"])

    result_path = tmp_path / evidence[0]["result_path"]
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload["paper_protocol"]["claim_scope"] = "retrospective"
    result_path.write_text(json.dumps(result_payload, sort_keys=True), encoding="utf-8")
    evidence[0]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    invalid_scope = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert invalid_scope["valid"] is False
    assert any("claim scope differs" in error for error in invalid_scope["errors"])


def test_signed_artifact_identity_is_verified_against_bound_json(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["predictors"]["tool_invocation"]["artifact"][
        "identity_sha256"
    ] = "9" * 64
    _reseal(manifest, tmp_path)
    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert result["valid"] is False
    assert any("signed identity" in error for error in result["errors"])


def test_result_schema_is_bound_to_frozen_policy_bundle(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    _write_analysis_matrix(tmp_path, manifest)
    first = manifest["cell_evidence"][0]
    result_path = tmp_path / first["result_path"]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema"] == audit.QWEN_RESULT_SCHEMA

    # Rebinding every mutable file hash must not allow a Gemini/Qwen schema
    # switch to select the other repository's raw-authority audit branch.
    payload["schema"] = audit.GEMINI_RESULT_SCHEMA
    result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    first["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    checked = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert checked["valid"] is False
    assert any(
        "result.schema: differs from frozen policy bundle" in error
        for error in checked["errors"]
    )


def test_signed_artifact_body_tampering_fails_after_file_hash_is_rebound(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    binding = manifest["predictors"]["tool_invocation"]["artifact"]
    artifact_path = tmp_path / binding["path"]
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    document["fit_code_sha256"] = "a" * 64
    artifact_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    binding["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    _reseal(manifest, tmp_path)

    result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert result["valid"] is False
    assert any("embedded signed hash is invalid" in error for error in result["errors"])


def test_embedded_feature_and_calibration_contracts_are_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    manifest["predictors"]["tool_duration"]["input_features"] = [
        "candidate_tool_name"
    ]
    _reseal(manifest, tmp_path)
    feature_result = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert feature_result["valid"] is False
    assert any("must exactly match" in error for error in feature_result["errors"])

    root_hash_dir = tmp_path / "root-hash"
    root_hash_dir.mkdir()
    manifest = _base_manifest(root_hash_dir)
    manifest["execution"]["physical_service_clock"]["artifact"][
        "training_root_ids_sha256"
    ] = "8" * 64
    _reseal(manifest, root_hash_dir)
    service_result = audit.audit_manifest(
        manifest,
        base=root_hash_dir,
        verify_files=True,
        require_evidence=False,
    )
    assert service_result["valid"] is False
    assert any("does not bind calibration roots" in error for error in service_result["errors"])


def _write_analysis_matrix(tmp_path: Path, manifest: dict) -> None:
    service_clock_sha256 = audit.artifact_identity_sha256(
        manifest["execution"]["physical_service_clock"]["artifact"]
    )
    provenance = audit.expected_runtime_provenance(manifest)
    cell_e2e = {"A": 10.0, "B": 9.0, "E": 8.5, "F": 7.5}
    evidence = []
    for block_index, block in enumerate(manifest["execution"]["blocks"]):
        for cell in audit.CELLS:
            position = block["order"].index(cell) + 1
            started_wall_s = 2_000.0 + block_index * 100.0 + position * 10.0
            ended_wall_s = started_wall_s + 5.0
            payload = _result(
                cell,
                service_clock_sha256=service_clock_sha256,
                provenance=provenance,
                workload_instances=manifest["execution"][
                    "treatment_neutral_runtime_parameters"
                ]["parameters"]["workload_instances"],
            )
            payload["runtime_parameters"] = _manifest_runtime_result(manifest)
            payload["model"] = "frozen-model-revision"
            experiment_started_monotonic_s = (
                10_000.0 + block_index * 100.0 + position * 20.0
            )
            task_e2e_s = cell_e2e[cell] + 0.01 * block_index
            payload.update(
                {
                    "block_id": block["block_id"],
                    "order_position": position,
                    "started_wall_s": started_wall_s,
                    "ended_wall_s": ended_wall_s,
                    "experiment_started_monotonic_s": experiment_started_monotonic_s,
                    "experiment_ended_monotonic_s": (
                        experiment_started_monotonic_s + task_e2e_s
                    ),
                    "experiment_wall_s": task_e2e_s,
                    "gpu_ids": block["gpu_ids"],
                    "server_instance_id": f"server-{block['block_id']}-{cell}",
                    "broker_instance_id": f"broker-{block['block_id']}-{cell}",
                }
            )
            qwen_summary_shape = block_index % 2 == 1
            task_rows = []
            payload["llm_events"] = []
            payload["tool_events"] = []
            payload["prediction_decisions"] = []
            payload["prediction_outcomes"] = []
            for root_id in manifest["data"]["evaluation_root_ids"]:
                task_id = f"task-{root_id}"
                task_rows.append(
                    {
                        "task_id": task_id,
                        (
                            "source_session_id"
                            if qwen_summary_shape
                            else "source_root_id"
                        ): root_id,
                        "release_offset_s": 0.0,
                        "scheduled_release_monotonic_s": experiment_started_monotonic_s,
                        "released_at_monotonic_s": experiment_started_monotonic_s,
                        "task_terminal_monotonic_s": (
                            experiment_started_monotonic_s + task_e2e_s
                        ),
                        "released_lag_s": 0.0,
                        "flow_s": task_e2e_s,
                        "e2e_s": task_e2e_s,
                        "ok": True,
                    }
                )
                event = {
                    "task_id": task_id,
                    "request_index": 0,
                    "workload_request_sha256": hashlib.sha256(
                        f"work-{root_id}".encode()
                    ).hexdigest(),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
                if qwen_summary_shape:
                    event["trace_id"] = event.pop("task_id")
                if cell in {"E", "F"}:
                    event["scheduler_metadata"] = _metadata()
                payload["llm_events"].append(event)
                payload["tool_events"].append(
                    {
                        "trace_id": task_id,
                        "request_index": 0,
                        "authoritative_revealed_seq": 1,
                        "authority_invocation_digest": hashlib.sha256(
                            f"authority-{root_id}".encode()
                        ).hexdigest(),
                        "tool_name": "search",
                        "outcome_id": hashlib.sha256(
                            f"outcome-{root_id}".encode()
                        ).hexdigest(),
                        "assigned_service_s": 1.25,
                    }
                )
            if qwen_summary_shape:
                payload["tasks"] = len(task_rows)
                payload["task_results"] = task_rows
            else:
                payload.pop("task_results", None)
                payload["tasks"] = task_rows
            result_path = tmp_path / f"analysis-{block['block_id']}-{cell}.json"
            result_path.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            evidence.append(
                {
                    "block_id": block["block_id"],
                    "cell": cell,
                    "order_position": position,
                    "started_wall_s": started_wall_s,
                    "ended_wall_s": ended_wall_s,
                    "gpu_ids": block["gpu_ids"],
                    "server_instance_id": f"server-{block['block_id']}-{cell}",
                    "broker_instance_id": f"broker-{block['block_id']}-{cell}",
                    "policy_bundle_sha256": manifest["freeze"][
                        "policy_bundle_sha256"
                    ],
                    "service_clock_artifact_sha256": service_clock_sha256,
                    "runtime_parameters_sha256": manifest["execution"][
                        "treatment_neutral_runtime_parameters"
                    ]["runtime_parameters_sha256"],
                    "provenance": provenance,
                    "result_path": result_path.name,
                    "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                }
            )
    manifest["cell_evidence"] = evidence


def test_matrix_provenance_and_runtime_artifact_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path)
    matrix = copy.deepcopy(manifest)
    _write_analysis_matrix(tmp_path, matrix)
    provenance = audit.expected_runtime_provenance(manifest)
    index_path = tmp_path / "matrix.json"

    poisoned_index_provenance = dict(provenance)
    poisoned_index_provenance["config_file_sha256"] = "8" * 64
    index_path.write_text(
        json.dumps(
            {
                "provenance": poisoned_index_provenance,
                "runtime_parameters": _manifest_runtime_result(manifest),
                "cell_evidence": matrix["cell_evidence"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matrix provenance config_file_sha256"):
        materializer._bind_matrix(copy.deepcopy(manifest), matrix_index=index_path)

    first_result = tmp_path / matrix["cell_evidence"][0]["result_path"]
    payload = json.loads(first_result.read_text(encoding="utf-8"))
    payload["provenance"]["invocation_predictor_artifact_sha256"] = "7" * 64
    first_result.write_text(json.dumps(payload), encoding="utf-8")
    index_path.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "runtime_parameters": _manifest_runtime_result(manifest),
                "cell_evidence": matrix["cell_evidence"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="invocation_predictor_artifact_sha256 differs"
    ):
        materializer._bind_matrix(copy.deepcopy(manifest), matrix_index=index_path)


def test_matrix_rejects_overlapping_williams_positions(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    matrix = copy.deepcopy(manifest)
    _write_analysis_matrix(tmp_path, matrix)
    first_block = manifest["execution"]["blocks"][0]
    second_cell = first_block["order"][1]
    row = next(
        item
        for item in matrix["cell_evidence"]
        if item["block_id"] == first_block["block_id"] and item["cell"] == second_cell
    )
    row["started_wall_s"] -= 10.0
    result_path = tmp_path / row["result_path"]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["started_wall_s"] = row["started_wall_s"]
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    index_path = tmp_path / "overlap-matrix.json"
    index_path.write_text(
        json.dumps(
            {
                "provenance": audit.expected_runtime_provenance(manifest),
                "runtime_parameters": _manifest_runtime_result(manifest),
                "cell_evidence": matrix["cell_evidence"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap or ran out of order"):
        materializer._bind_matrix(copy.deepcopy(manifest), matrix_index=index_path)


def test_analyzer_recomputes_paired_root_table_and_work_gate(tmp_path: Path) -> None:
    manifest = _base_manifest(tmp_path)
    _write_analysis_matrix(tmp_path, manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    report = analyzer.analyze_manifest(manifest_path)
    assert report["work_equivalence"]["passed"] is True
    assert report["work_equivalence"][
        "registered_workload_contract_passed"
    ] is True
    assert audit._is_sha256(
        report["work_equivalence"]["registered_workload_contract_sha256"]
    )
    assert audit._is_sha256(
        report["work_equivalence"]["authoritative_tool_work_sha256"]
    )
    assert report["folding"]["roots"] == 30
    assert report["folding"]["blocks"] == 4
    assert report["contrasts"]["A_vs_F"][
        "ratio_of_paired_root_mean_e2e"
    ] == pytest.approx(0.25, abs=0.002)
    assert report["primary_rule"]["speedup_20_pass"] is True
    descriptive = report["mechanism_and_system_descriptive"]
    assert len(descriptive["by_block_cell"]) == 16
    for row in descriptive["by_block_cell"]:
        assert set(analyzer.DESCRIPTIVE_METRIC_FIELDS).issubset(row)
        assert row["duration_predictor_mae_s"] is None
    assert descriptive["by_cell"]["A"]["pooled"]["requests"] == 120
    assert descriptive["by_cell"]["A"]["pooled"][
        "duration_predictor_mae_s"
    ] is None

    bad = manifest["cell_evidence"][-1]
    bad_path = tmp_path / bad["result_path"]
    payload = json.loads(bad_path.read_text(encoding="utf-8"))
    payload["tool_events"][0]["assigned_service_s"] = 9.0
    bad_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    bad["result_sha256"] = hashlib.sha256(bad_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(analyzer.AnalysisError, match="work"):
        analyzer.analyze_manifest(manifest_path)

    payload["tool_events"][0]["assigned_service_s"] = 1.25
    payload["llm_events"][0]["usage"]["completion_tokens"] = 11
    bad_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    bad["result_sha256"] = hashlib.sha256(bad_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(analyzer.AnalysisError, match="work"):
        analyzer.analyze_manifest(manifest_path)


@pytest.mark.parametrize("attack", ["drop_replica", "drop_request"])
def test_common_workload_deletion_across_all_cells_still_fails_after_rebinding(
    tmp_path: Path, attack: str
) -> None:
    """Cross-cell equality cannot legitimize jointly truncated formal work."""

    manifest = _base_manifest(tmp_path)
    _write_analysis_matrix(tmp_path, manifest)
    target_task = "task-eval-01"
    for binding in manifest["cell_evidence"]:
        result_path = tmp_path / binding["result_path"]
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if attack == "drop_replica":
            rows = payload.get("task_results")
            if isinstance(rows, list):
                payload["task_results"] = [
                    row
                    for row in rows
                    if row.get("task_id", row.get("trace_id")) != target_task
                ]
                if type(payload.get("tasks")) is int:
                    payload["tasks"] -= 1
            else:
                payload["tasks"] = [
                    row
                    for row in payload["tasks"]
                    if row.get("task_id", row.get("trace_id")) != target_task
                ]
        payload["llm_events"] = [
            row
            for row in payload["llm_events"]
            if row.get("task_id", row.get("trace_id")) != target_task
        ]
        payload["tool_events"] = [
            row
            for row in payload["tool_events"]
            if row.get("task_id", row.get("trace_id")) != target_task
        ]
        result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        binding["result_sha256"] = hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()

    manifest_path = tmp_path / f"{attack}-rebound-evidence.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        analyzer.AnalysisError,
        match=(
            "task count does not equal frozen workload_instances"
            if attack == "drop_replica"
            else r"task\(s\) have no live requests"
        ),
    ):
        analyzer.analyze_manifest(manifest_path)


def test_resigned_e2e_summary_poison_cannot_forge_headline(tmp_path: Path) -> None:
    """Rebinding every mutable wrapper must not legitimize a fake task speedup."""

    manifest = _base_manifest(tmp_path)
    _write_analysis_matrix(tmp_path, manifest)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    honest = analyzer.analyze_manifest(evidence_path)

    target = next(
        row
        for row in manifest["cell_evidence"]
        if row["block_id"] == manifest["execution"]["blocks"][0]["block_id"]
        and row["cell"] == "F"
    )
    result_path = tmp_path / target["result_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    tasks = result.get("task_results")
    if not isinstance(tasks, list):
        tasks = result["tasks"]
    tasks[0]["flow_s"] = 0.01
    tasks[0]["e2e_s"] = 0.01
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    # Simulate a hostile postprocessor that rebinds the modified result into
    # the matrix/evidence manifest and self-signs a fabricated 99% report.
    target["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    evidence_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    audited = audit.audit_manifest(
        manifest, base=tmp_path, verify_files=True, require_evidence=False
    )
    assert audited["valid"] is False
    assert any("terminal-minus-scheduled" in error for error in audited["errors"])
    with pytest.raises(analyzer.AnalysisError, match="terminal-minus-scheduled"):
        analyzer.analyze_manifest(evidence_path)

    forged = copy.deepcopy(honest)
    forged["manifest_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    for binding in forged["result_bindings"]:
        if binding["block_id"] == target["block_id"] and binding["cell"] == "F":
            binding["sha256"] = target["result_sha256"]
    forged["contrasts"]["A_vs_F"]["ratio_of_paired_root_mean_e2e"] = 0.99
    forged["manifest_outcomes"]["A_vs_F"][
        "ratio_of_paired_root_mean_e2e"
    ] = 0.99
    forged["primary_rule"]["speedup_20_pass"] = True
    unsigned = dict(forged)
    unsigned.pop("analysis_sha256", None)
    forged["analysis_sha256"] = analyzer._sha256(unsigned)
    forged_path = tmp_path / "forged-analysis.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="input manifest failed audit"):
        materializer.finalize_manifest(
            SimpleNamespace(
                manifest=evidence_path,
                output=tmp_path / "forged-final.json",
                matrix_index=None,
                analysis=forged_path,
            )
        )


def test_analyzer_normalizes_qwen_and_gemini_mechanism_metrics() -> None:
    common_tasks = [
        {"task_id": "task-1", "source_root_id": "root-1", "ok": True}
    ]
    common_requests = [
        {
            "task_id": "task-1",
            "request_index": 0,
            "latency_s": 2.0,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
    ]
    qwen = {
        "experiment_wall_s": 10.0,
        "prediction_decisions": [
            {
                "prediction_id": "prediction-qwen-1",
                "trace_id": "task-1",
                "request_index": 0,
                "candidates": [
                    {"candidate_invocation_digest": "1" * 64, "broker_accepted": True, "admitted": True},
                    {"candidate_invocation_digest": "2" * 64, "broker_accepted": True, "admitted": True},
                    {"candidate_invocation_digest": "3" * 64, "broker_accepted": False, "admitted": False},
                    {"candidate_invocation_digest": "4" * 64, "broker_accepted": True, "admitted": True},
                ]
            }
        ],
        "prediction_outcomes": [
            {
                "prediction_id": "prediction-qwen-1",
                "trace_id": "task-1",
                "request_index": 0,
                "candidates": [
                    {
                            "candidate_invocation_digest": "1" * 64,
                            "broker_accepted": True,
                            "admitted": True,
                        "matched_authority": True,
                    },
                    {
                            "candidate_invocation_digest": "2" * 64,
                            "broker_accepted": True,
                            "admitted": True,
                        "matched_authority": False,
                    },
                    {
                            "candidate_invocation_digest": "3" * 64,
                            "broker_accepted": False,
                            "admitted": False,
                            "matched_authority": True,
                        },
                        {
                            "candidate_invocation_digest": "4" * 64,
                            "broker_accepted": True,
                            "admitted": True,
                            "matched_authority": False,
                        },
                ]
            }
        ],
        "speculation_execution_events": [
                {
                    "prediction_id": "prediction-qwen-1",
                    "candidate_invocation_digest": "1" * 64,
                    "physical_started_at_monotonic_s": 1.0,
                "claimed_by_authority": True,
                "speculative_resource_s": 1.5,
            },
                {
                    "prediction_id": "prediction-qwen-1",
                    "candidate_invocation_digest": "2" * 64,
                    "physical_started_at_monotonic_s": 1.5,
                "claimed_by_authority": False,
                    "speculative_resource_s": 0.5,
                },
                {
                    "prediction_id": "prediction-qwen-1",
                    "candidate_invocation_digest": "4" * 64,
                    "physical_started_at_monotonic_s": None,
                    "claimed_by_authority": False,
                    "speculative_resource_s": 0.0,
                },
        ],
        "worker_resource_accounting": {
            "speculative_resource_s": 2.0,
            "promoted_demand_resource_s": 0.25,
            "direct_demand_resource_s": 3.0,
            "total_worker_occupancy_s": 5.25,
        },
        "tool_events": [
            {
                "trace_id": "task-1",
                "request_index": 0,
                "authority_candidate_invocation_digests": ["1" * 64, "3" * 64],
                "service_s": 2.0,
            }
        ],
    }
    qwen_metrics = analyzer._descriptive_metrics(
        qwen,
        tasks_raw=common_tasks,
        requests_raw=common_requests,
        label="qwen",
    )
    assert qwen_metrics["prediction_candidates_broker_accepted"] == 3
    assert qwen_metrics["prediction_candidates_physical_started"] == 2
    assert qwen_metrics["prediction_candidates_admitted"] == 2
    assert qwen_metrics["exact_post_authority_hits"] == 1
    assert qwen_metrics["prediction_precision"] == pytest.approx(0.5)
    assert qwen_metrics["admitted_prediction_precision"] == pytest.approx(0.5)
    assert qwen_metrics["physical_started_prediction_precision"] == pytest.approx(0.5)
    assert qwen_metrics["broker_accepted_prediction_precision"] == pytest.approx(1 / 3)
    assert qwen_metrics["emitted_prediction_precision"] == pytest.approx(0.5)
    assert qwen_metrics["useful_speculative_worker_s"] == pytest.approx(1.5)
    assert qwen_metrics["wasted_speculative_worker_s"] == pytest.approx(0.5)
    assert qwen_metrics["duration_predictor_mae_s"] is None

    gemini = {
        "summary": {"makespan_s": 8.0},
        "prediction_decisions": [
            {
                "prediction_id": "prediction-gemini-1",
                "trace_id": "task-1",
                "request_index": 0,
                "candidate_invocation_digest": "4" * 64,
                "broker_accepted": True,
                "admitted": True,
            }
        ],
        "prediction_outcomes": [
            {
                "prediction_id": "prediction-gemini-1",
                "trace_id": "task-1",
                "request_index": 0,
                "candidate_invocation_digest": "4" * 64,
                "broker_accepted": True,
                "admitted": True,
                "post_authority_hit": True,
            }
        ],
        "speculation_execution_events": [
            {
                "prediction_id": "prediction-gemini-1",
                "candidate_invocation_digest": "4" * 64,
                "physical_started_at_monotonic_s": 1.0,
                "claimed_by_authority": True,
                "speculative_resource_s": 0.75,
            }
        ],
        "worker_resource_accounting": {
            "speculative_resource_s": 0.75,
            "promoted_demand_resource_s": 0.5,
            "direct_demand_resource_s": 2.0,
            "total_worker_occupancy_s": 3.25,
        },
        "tool_events": [
            {
                "trace_id": "task-1",
                "request_index": 0,
                "pool_authority_key_sha256": "4" * 64,
                "authority_eta_hat_s": 1.0,
                "execution_surface_service_s": 1.4,
                "duration_prediction_absolute_error_s": 0.4,
            }
        ],
    }
    gemini_metrics = analyzer._descriptive_metrics(
        gemini,
        tasks_raw=common_tasks,
        requests_raw=common_requests,
        label="gemini",
    )
    assert gemini_metrics["makespan_s"] == pytest.approx(8.0)
    assert gemini_metrics["prediction_precision"] == pytest.approx(1.0)
    assert gemini_metrics["broker_accepted_prediction_precision"] == pytest.approx(1.0)
    assert gemini_metrics["physical_started_prediction_precision"] == pytest.approx(1.0)
    assert gemini_metrics["admitted_prediction_precision"] == pytest.approx(1.0)
    assert gemini_metrics["emitted_prediction_precision"] == pytest.approx(1.0)
    assert gemini_metrics["duration_predictor_mae_s"] == pytest.approx(0.4)
    tampered_duration = copy.deepcopy(gemini)
    tampered_duration["tool_events"][0][
        "duration_prediction_absolute_error_s"
    ] = 0.1
    with pytest.raises(analyzer.AnalysisError, match="declared duration error"):
        analyzer._descriptive_metrics(
            tampered_duration,
            tasks_raw=common_tasks,
            requests_raw=common_requests,
            label="gemini",
        )


def test_analyzer_rejects_cross_cell_same_key_service_change() -> None:
    common = {
        "model": "model",
        "task_multiset_by_root": {"root": 1},
        "tasks": {"task": {"root_id": "root"}},
        "task_work": {"task": {"requests": [], "authoritative_tools": []}},
    }
    rows = [
        {
            **copy.deepcopy(common),
            "block_id": "block-1",
            "cell": "A",
            "physical_services": {HEX: 1.0},
        },
        {
            **copy.deepcopy(common),
            "block_id": "block-1",
            "cell": "F",
            "physical_services": {HEX: 1.1},
        },
    ]
    with pytest.raises(analyzer.AnalysisError, match="same normalized physical"):
        analyzer._check_work_equivalence(rows)


def test_materializer_create_bind_analyze_and_attach(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.txt"
    tuning_path = tmp_path / "tuning.txt"
    evaluation_path = tmp_path / "evaluation.txt"
    calibration_path.write_text("cal-1\ncal-2\n", encoding="utf-8")
    tuning_path.write_text("tune-1\ntune-2\n", encoding="utf-8")
    evaluation_path.write_text(
        "".join(f"eval-{index:02d}\n" for index in range(1, 31)),
        encoding="utf-8",
    )
    integration_evaluation_ids = [f"eval-{index:02d}" for index in range(1, 31)]
    near_duplicate_path = tmp_path / "near-duplicate.json"
    near_duplicate_path.write_text(
        json.dumps(
            {
                "schema": audit.NEAR_DUPLICATE_AUDIT_SCHEMA,
                "verified": True,
                "registered_root_sets_sha256": audit.registered_root_sets_sha256(
                    ["cal-1", "cal-2"],
                    ["tune-1", "tune-2"],
                    integration_evaluation_ids,
                ),
                "method": "integration fixture semantic fingerprint audit",
                "near_duplicate_pairs_across_splits": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    frozen = []
    for role in sorted(audit.REQUIRED_FROZEN_FILE_ROLES):
        path = tmp_path / f"{role}.txt"
        if role == "policy_bundle":
            path.write_text(
                json.dumps(
                    {
                        "schema": "paste.paper.registered_workload_contract.v1",
                        "tasks": [
                            {
                                "task_id": f"task-{root_id}",
                                "root_id": root_id,
                                "release_offset_s": 0.0,
                                "request_count": 1,
                            }
                            for root_id in integration_evaluation_ids
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"{role}\n", encoding="utf-8")
        frozen.append(f"{role}={path}")
    invocation = tmp_path / "invocation.json"
    duration = tmp_path / "duration.json"
    service = tmp_path / "service.json"
    integration_training_hash = audit.canonical_sha256(["cal-1", "cal-2"])
    integration_invocation_features = [
        "last_completed_tool_name",
        "current_visible_search_result_urls",
        "current_visible_search_result_ranks",
        "current_visible_search_result_ordinals",
        "frozen_top_k",
    ]
    integration_duration_features = [
        "candidate_tool_name",
        "candidate_host",
        "completed_tool_service_times",
    ]
    invocation_body = {
        "kind": "invocation",
        "input_features": integration_invocation_features,
        "training_root_ids_sha256": integration_training_hash,
        "uses_evaluation_labels": False,
        "fit_code_sha256": "6" * 64,
    }
    invocation_identity = audit.canonical_sha256(invocation_body)
    invocation.write_text(
        json.dumps({**invocation_body, "artifact_sha256": invocation_identity})
        + "\n",
        encoding="utf-8",
    )
    duration_body = {
        "kind": "duration",
        "input_features": integration_duration_features,
        "training_root_ids_sha256": integration_training_hash,
        "uses_evaluation_labels": False,
        "fit_code_sha256": "7" * 64,
    }
    duration_identity = audit.canonical_sha256(duration_body)
    duration.write_text(
        json.dumps({**duration_body, "artifact_sha256": duration_identity})
        + "\n",
        encoding="utf-8",
    )
    service_body = {
        "kind": "service",
        "calibration_session_ids_sha256": integration_training_hash,
        "uses_evaluation_labels": False,
        "uses_evaluation_trace_durations": False,
        "future_state_accepted_invariant": True,
    }
    gemini_surface_identity = audit.canonical_sha256(service_body)
    service.write_text(
        json.dumps(
            {
                **service_body,
                "executor_service_surface_sha256": gemini_surface_identity,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(_runtime_parameters(workload_instances=30), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sealed_path = tmp_path / "sealed-manifest.json"
    materializer.create_manifest(
        SimpleNamespace(
            output=sealed_path,
            calibration_roots=calibration_path,
            tuning_roots=tuning_path,
            evaluation_roots=evaluation_path,
            exposed_roots=None,
            near_duplicate_evidence=near_duplicate_path,
            claim_scope="confirmatory",
            selection_protocol="heldout_tuning_split",
            frozen_file=frozen,
            invocation_predictor_artifact=invocation,
            duration_predictor_artifact=duration,
            service_clock_artifact=service,
            runtime_parameters_json=runtime_path,
            invocation_feature=integration_invocation_features,
            duration_feature=integration_duration_features,
            call_graph_mode="autonomous",
            gpu_groups="0,1,2,3;4,5,6,7",
            williams_cycles=1,
            bootstrap_resamples=10_000,
            bootstrap_seed="integration-test",
        )
    )
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    assert sealed["execution"]["physical_service_clock"]["artifact"][
        "identity_sha256"
    ] == gemini_surface_identity
    assert sealed["predictors"]["tool_invocation"]["artifact"][
        "identity_sha256"
    ] == invocation_identity
    assert sealed["predictors"]["tool_duration"]["artifact"][
        "identity_sha256"
    ] == duration_identity
    assert audit.audit_manifest(
        sealed, base=tmp_path, verify_files=True, require_evidence=False
    )["valid"]

    matrix_manifest = copy.deepcopy(sealed)
    _write_analysis_matrix(tmp_path, matrix_manifest)
    matrix_index = tmp_path / "matrix-index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "provenance": audit.expected_runtime_provenance(sealed),
                "runtime_parameters": _manifest_runtime_result(sealed),
                "cell_evidence": matrix_manifest["cell_evidence"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence-manifest.json"
    materializer.finalize_manifest(
        SimpleNamespace(
            manifest=sealed_path,
            output=evidence_path,
            matrix_index=matrix_index,
            analysis=None,
        )
    )
    report = analyzer.analyze_manifest(evidence_path)
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

    forged = copy.deepcopy(report)
    forged["manifest_outcomes"]["A_vs_F"][
        "ratio_of_paired_root_mean_e2e"
    ] = 0.99
    forged_unsigned = dict(forged)
    forged_unsigned.pop("analysis_sha256")
    forged["analysis_sha256"] = analyzer._sha256(forged_unsigned)
    forged_path = tmp_path / "forged-analysis.json"
    forged_path.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="trusted recomputation"):
        materializer.finalize_manifest(
            SimpleNamespace(
                manifest=evidence_path,
                output=tmp_path / "forged-final.json",
                matrix_index=None,
                analysis=forged_path,
            )
        )

    final_path = tmp_path / "final-manifest.json"
    final = materializer.finalize_manifest(
        SimpleNamespace(
            manifest=evidence_path,
            output=final_path,
            matrix_index=None,
            analysis=analysis_path,
        )
    )
    audited = audit.audit_manifest(
        final, base=tmp_path, verify_files=True, require_evidence=True
    )
    assert audited["valid"] is True, audited["errors"]
    assert audited["speedup_20_pass"] is True

    # A separately valid evidence manifest must not silently substitute a
    # different preregistered/statistical contract merely because it binds the
    # same result rows.  Recompute a genuinely signed report over such a
    # substitute to exercise the trusted-analysis boundary end to end.
    substituted_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    substituted_evidence.pop("preregistered_manifest", None)
    substituted_evidence["statistics"]["paired_bootstrap_seed"] = (
        "attacker-selected-seed"
    )
    substituted_marker_path = tmp_path / "substituted.FORMAL_STARTED.json"
    substituted_evidence["freeze"]["started_marker_path"] = str(
        substituted_marker_path
    )
    substituted_evidence["freeze"]["sealed_payload_sha256"] = (
        audit.sealed_payload_sha256(substituted_evidence)
    )
    substituted_marker_path.write_text(
        json.dumps(
            {
                "schema": audit.START_MARKER_SCHEMA,
                "sealed_payload_sha256": substituted_evidence["freeze"][
                    "sealed_payload_sha256"
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    substituted_evidence_path = tmp_path / "substituted-evidence.json"
    substituted_evidence_path.write_text(
        json.dumps(substituted_evidence, sort_keys=True), encoding="utf-8"
    )
    substituted_report = analyzer.analyze_manifest(substituted_evidence_path)
    substituted_report_path = tmp_path / "substituted-analysis.json"
    substituted_report_path.write_text(
        json.dumps(substituted_report, sort_keys=True), encoding="utf-8"
    )
    attacked_final = copy.deepcopy(final)
    attacked_final["outcomes"] = copy.deepcopy(
        substituted_report["manifest_outcomes"]
    )
    attacked_final["analysis_evidence_manifest"] = materializer._binding(
        substituted_evidence_path
    )
    attacked_final["analysis_report"] = materializer._binding(
        substituted_report_path, identity_fields=("analysis_sha256",)
    )
    attacked = audit.audit_manifest(
        attacked_final, base=tmp_path, verify_files=True, require_evidence=True
    )
    assert attacked["valid"] is False
    assert any(
        "sealed/preregistered content differs" in error
        for error in attacked["errors"]
    )
