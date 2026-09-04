#!/usr/bin/env python3
"""Create and finalize immutable manifests for the strict causal A/B/E/F run."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_strict_causal_experiment as audit
import analyze_strict_causal_abef as analyzer


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _binding(
    path: Path,
    *,
    role: str | None = None,
    identity_fields: Sequence[str] = (),
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"bound file does not exist: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": audit.file_sha256(resolved),
    }
    if role is not None:
        result["role"] = role
    if identity_fields:
        try:
            document = _read_json(resolved)
        except (OSError, json.JSONDecodeError):
            document = None
        identity = None
        if isinstance(document, Mapping):
            for field in identity_fields:
                candidate = document.get(field)
                if audit._is_sha256(candidate):
                    identity = str(candidate)
                    break
        if identity is None:
            raise ValueError(
                f"bound artifact has no signed logical SHA-256 in "
                f"{list(identity_fields)}: {resolved}"
            )
        result["identity_sha256"] = identity
    return result


def _training_root_identity(document: Mapping[str, Any]) -> Any:
    for field in (
        "training_root_ids_sha256",
        "training_session_ids_sha256",
        "calibration_session_ids_sha256",
    ):
        if field in document:
            return document.get(field)
    provenance = document.get("training_provenance")
    if isinstance(provenance, Mapping):
        return provenance.get("session_ids_sha256")
    return None


def _annotate_predictor_contract(
    binding: dict[str, Any], *, expected_features: Sequence[str]
) -> None:
    document = _read_json(Path(str(binding["path"])))
    if not isinstance(document, Mapping):
        raise ValueError(f"predictor artifact must be a JSON object: {binding['path']}")
    features = document.get("input_features")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(item, str) or not item for item in features)
    ):
        raise ValueError(f"predictor artifact lacks exact input_features: {binding['path']}")
    if list(features) != list(expected_features):
        raise ValueError(
            "declared predictor features differ from embedded input_features: "
            f"declared={list(expected_features)!r} embedded={features!r}"
        )
    training_hash = _training_root_identity(document)
    if not audit._is_sha256(training_hash):
        raise ValueError(f"predictor artifact lacks training-root hash: {binding['path']}")
    if document.get("uses_evaluation_labels") is not False:
        raise ValueError(f"predictor artifact must declare uses_evaluation_labels=false")
    binding["input_features"] = list(features)
    binding["training_root_ids_sha256"] = str(training_hash)
    binding["uses_evaluation_labels"] = False
    fit_code = document.get("fit_code_sha256")
    if fit_code is not None:
        if not audit._is_sha256(fit_code):
            raise ValueError(f"predictor artifact fit_code_sha256 is invalid")
        binding["fit_code_sha256"] = str(fit_code)


def _annotate_service_contract(binding: dict[str, Any]) -> None:
    document = _read_json(Path(str(binding["path"])))
    if not isinstance(document, Mapping):
        raise ValueError(f"service-clock artifact must be a JSON object: {binding['path']}")
    training_hash = _training_root_identity(document)
    if not audit._is_sha256(training_hash):
        raise ValueError(f"service-clock artifact lacks calibration-root hash")
    if document.get("uses_evaluation_labels") is not False:
        raise ValueError("service-clock artifact must declare uses_evaluation_labels=false")
    if document.get("future_state_accepted_invariant") is not True:
        raise ValueError(
            "service-clock artifact must declare future_state_accepted_invariant=true"
        )
    if (
        "uses_evaluation_trace_durations" in document
        and document.get("uses_evaluation_trace_durations") is not False
    ):
        raise ValueError(
            "service-clock artifact must declare uses_evaluation_trace_durations=false"
        )
    binding["training_root_ids_sha256"] = str(training_hash)
    binding["uses_evaluation_labels"] = False
    binding["future_state_accepted_invariant"] = True
    if "uses_evaluation_trace_durations" in document:
        binding["uses_evaluation_trace_durations"] = False


def _near_duplicate_binding(
    path: Path,
    *,
    calibration: Sequence[str],
    tuning: Sequence[str],
    evaluation: Sequence[str],
) -> dict[str, Any]:
    binding = _binding(path)
    document = _read_json(Path(str(binding["path"])))
    if not isinstance(document, Mapping):
        raise ValueError("near-duplicate evidence must be a JSON object")
    expected_root_hash = audit.registered_root_sets_sha256(
        calibration, tuning, evaluation
    )
    expected = {
        "schema": audit.NEAR_DUPLICATE_AUDIT_SCHEMA,
        "verified": True,
        "registered_root_sets_sha256": expected_root_hash,
        "near_duplicate_pairs_across_splits": [],
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ValueError(
                f"near-duplicate evidence {field} must be {value!r}"
            )
        binding[field] = copy.deepcopy(value)
    method = document.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("near-duplicate evidence requires a non-empty method")
    binding["method"] = method
    return binding


def _runtime_parameters_contract(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    errors: list[str] = []
    audit._audit_runtime_parameters_contract(
        document,
        label="runtime parameters",
        require_artifact=False,
        errors=errors,
    )
    if errors:
        raise ValueError("invalid runtime-parameters JSON: " + "; ".join(errors))
    assert isinstance(document, Mapping)
    binding = _binding(
        path, identity_fields=("runtime_parameters_sha256",)
    )
    return {
        "schema": document["schema"],
        "parameters": copy.deepcopy(document["parameters"]),
        "runtime_parameters_sha256": document["runtime_parameters_sha256"],
        "artifact": binding,
    }


def _root_ids(path: Path, *, allow_empty: bool = False) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(value, Mapping):
            for field in ("root_ids", "source_session_ids"):
                if field in value:
                    value = value[field]
                    break
        if not isinstance(value, list):
            raise ValueError(
                "root file must be a JSON list, an object containing root_ids "
                f"or source_session_ids, or one ID per line: {path}"
            )
        values = value
    if (not values and not allow_empty) or any(
        not isinstance(item, str) or not item for item in values
    ):
        raise ValueError(f"root IDs must be non-empty strings: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate root IDs: {path}")
    return list(values)


def _frozen_bindings(values: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError("--frozen-file must be ROLE=PATH")
        role, path_raw = raw.split("=", 1)
        role = role.strip()
        if not role or not path_raw:
            raise ValueError("--frozen-file must be ROLE=PATH")
        rows.append(_binding(Path(path_raw), role=role))
    roles = [row["role"] for row in rows]
    if len(roles) != len(set(roles)):
        raise ValueError("duplicate --frozen-file role")
    missing = sorted(audit.REQUIRED_FROZEN_FILE_ROLES - set(roles))
    if missing:
        raise ValueError(f"missing frozen file roles: {missing}")
    return rows


def _annotate_legacy_compatibility_bindings(
    rows: Sequence[dict[str, Any]],
) -> Mapping[str, Any] | None:
    """Validate and annotate the optional one-off Gemini compatibility proof.

    A certificate without its independently frozen verifier (or vice versa) is
    never meaningful.  When neither role is present this function is a strict
    no-op so the Qwen manifest payload is unchanged.
    """

    by_role = {str(row.get("role")): row for row in rows}
    certificate_binding = by_role.get(
        audit.GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE
    )
    verifier_binding = by_role.get(
        audit.GEMINI_LEGACY_COMPATIBILITY_VERIFIER_ROLE
    )
    if certificate_binding is None and verifier_binding is None:
        return None
    if certificate_binding is None or verifier_binding is None:
        raise ValueError(
            "Gemini legacy compatibility requires both the certificate and "
            "independent verifier frozen-file roles"
        )
    document = _read_json(Path(str(certificate_binding["path"])))
    if not isinstance(document, Mapping):
        raise ValueError("Gemini legacy compatibility certificate must be an object")
    if document.get("schema") != audit.GEMINI_LEGACY_COMPATIBILITY_SCHEMA:
        raise ValueError("unsupported Gemini legacy compatibility certificate schema")
    if (
        document.get("compatibility_mode")
        != audit.GEMINI_LEGACY_COMPATIBILITY_MODE
    ):
        raise ValueError("unsupported Gemini legacy compatibility mode")
    identity = document.get("compatibility_sha256")
    if not audit._is_sha256(identity):
        raise ValueError("Gemini legacy compatibility identity is invalid")
    unsigned = dict(document)
    unsigned.pop("compatibility_sha256", None)
    if audit.canonical_sha256(unsigned) != identity:
        raise ValueError("Gemini legacy compatibility certificate signature is invalid")
    verifier_sha256 = document.get("independent_verifier_sha256")
    if verifier_sha256 != verifier_binding.get("sha256"):
        raise ValueError(
            "Gemini compatibility certificate does not bind the frozen verifier"
        )
    certificate_binding.update(
        {
            "identity_sha256": str(identity),
            "verifier_sha256": str(verifier_sha256),
            "schema": audit.GEMINI_LEGACY_COMPATIBILITY_SCHEMA,
            "compatibility_mode": audit.GEMINI_LEGACY_COMPATIBILITY_MODE,
        }
    )
    return document


def _gpu_groups(raw: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for group_raw in raw.split(";"):
        try:
            group = [int(item.strip()) for item in group_raw.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("--gpu-groups must be semicolon-separated integer lists") from exc
        if not group or min(group) < 0 or len(group) != len(set(group)):
            raise ValueError("each GPU group must contain unique non-negative IDs")
        groups.append(group)
    if len(groups) < 2 or len({tuple(group) for group in groups}) != len(groups):
        raise ValueError("at least two distinct GPU groups are required")
    return groups


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    calibration = _root_ids(args.calibration_roots)
    tuning = _root_ids(args.tuning_roots, allow_empty=True)
    evaluation = _root_ids(args.evaluation_roots)
    exposed = _root_ids(args.exposed_roots) if args.exposed_roots else []
    if args.claim_scope == "confirmatory" and exposed:
        raise ValueError("confirmatory creation forbids exposed evaluation roots")
    if args.claim_scope == "retrospective" and not exposed:
        raise ValueError("retrospective creation requires an explicit non-empty --exposed-roots")
    if not set(exposed).issubset(set(evaluation)):
        raise ValueError("exposed roots must be members of the evaluation set")
    selection_protocol = args.selection_protocol
    if not tuning and selection_protocol != "nested_cross_validation_within_calibration":
        raise ValueError(
            "an empty tuning split requires "
            "--selection-protocol nested_cross_validation_within_calibration"
        )
    if tuning and selection_protocol != "heldout_tuning_split":
        raise ValueError(
            "a non-empty tuning split requires "
            "--selection-protocol heldout_tuning_split"
        )
    if not args.invocation_feature:
        raise ValueError("at least one exact --invocation-feature is required")
    if not args.duration_feature:
        raise ValueError("at least one exact --duration-feature is required")

    runtime_parameters = _runtime_parameters_contract(args.runtime_parameters_json)

    near_duplicate_path = getattr(args, "near_duplicate_evidence", None)
    if args.claim_scope == "confirmatory" and near_duplicate_path is None:
        raise ValueError(
            "confirmatory creation requires --near-duplicate-evidence bound to "
            "the registered split membership"
        )
    near_duplicate_evidence = (
        _near_duplicate_binding(
            near_duplicate_path,
            calibration=calibration,
            tuning=tuning,
            evaluation=evaluation,
        )
        if near_duplicate_path is not None
        else None
    )

    frozen_files = _frozen_bindings(args.frozen_file)
    legacy_compatibility = _annotate_legacy_compatibility_bindings(frozen_files)
    public_plan_errors: list[str] = []
    policy_bundle_binding = next(
        row for row in frozen_files if row["role"] == "policy_bundle"
    )
    policy_bundle_document = _read_json(Path(str(policy_bundle_binding["path"])))
    if not isinstance(policy_bundle_document, Mapping):
        raise ValueError("policy bundle must be a JSON object")
    policy_compatibility = policy_bundle_document.get(
        "legacy_frozen_compatibility"
    )
    if (legacy_compatibility is None) != (policy_compatibility is None):
        raise ValueError(
            "Gemini legacy compatibility must appear in both the frozen policy "
            "bundle and its certificate/verifier roles"
        )
    if legacy_compatibility is not None:
        if not isinstance(policy_compatibility, Mapping):
            raise ValueError("policy legacy_frozen_compatibility must be an object")
        frozen_by_role = {
            str(row["role"]): row for row in frozen_files if "role" in row
        }
        certificate_binding = frozen_by_role[
            audit.GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE
        ]
        verifier_binding = frozen_by_role[
            audit.GEMINI_LEGACY_COMPATIBILITY_VERIFIER_ROLE
        ]
        expected_policy_compatibility = {
            "schema": audit.GEMINI_LEGACY_COMPATIBILITY_SCHEMA,
            "compatibility_mode": audit.GEMINI_LEGACY_COMPATIBILITY_MODE,
            "certificate_file_sha256": certificate_binding["sha256"],
            "compatibility_sha256": certificate_binding["identity_sha256"],
            "independent_verifier_sha256": verifier_binding["sha256"],
        }
        for field, expected in expected_policy_compatibility.items():
            if policy_compatibility.get(field) != expected:
                raise ValueError(
                    "policy legacy compatibility differs from frozen binding "
                    f"for {field}"
                )
        for field, expected_path in (
            ("certificate_path", Path(str(certificate_binding["path"]))),
            ("independent_verifier_path", Path(str(verifier_binding["path"]))),
        ):
            path_raw = policy_compatibility.get(field)
            if (
                not isinstance(path_raw, str)
                or Path(path_raw).resolve() != expected_path.resolve()
            ):
                raise ValueError(
                    "policy legacy compatibility path differs from frozen "
                    f"binding for {field}"
                )
    audit.audit_public_plan_firewall(
        policy_bundle_binding, base=output.parent, errors=public_plan_errors
    )
    if public_plan_errors:
        raise ValueError("public-plan firewall failed: " + "; ".join(public_plan_errors))
    invocation_artifact = _binding(
        args.invocation_predictor_artifact,
        identity_fields=("artifact_sha256",),
    )
    duration_artifact = _binding(
        args.duration_predictor_artifact,
        identity_fields=("artifact_sha256",),
    )
    service_artifact = _binding(args.service_clock_artifact)
    service_identity = getattr(args, "service_clock_identity_sha256", None)
    if service_identity is None:
        try:
            service_document = _read_json(args.service_clock_artifact)
        except (OSError, json.JSONDecodeError):
            service_document = None
        if isinstance(service_document, Mapping):
            for field in (
                "service_clock_artifact_sha256",
                "tool_service_surface_sha256",
                "executor_service_surface_sha256",
                "service_surface_sha256",
                "artifact_sha256",
            ):
                candidate = service_document.get(field)
                if audit._is_sha256(candidate):
                    service_identity = str(candidate)
                    break
    if service_identity is not None:
        if not audit._is_sha256(service_identity):
            raise ValueError("--service-clock-identity-sha256 is invalid")
        service_artifact["identity_sha256"] = service_identity
    _annotate_predictor_contract(
        invocation_artifact, expected_features=args.invocation_feature
    )
    _annotate_predictor_contract(
        duration_artifact, expected_features=args.duration_feature
    )
    _annotate_service_contract(service_artifact)
    if audit.artifact_identity_sha256(duration_artifact) == audit.artifact_identity_sha256(
        service_artifact
    ):
        raise ValueError("duration predictor and physical service clock must be distinct")
    training_hash = audit.canonical_sha256(sorted(calibration))
    for label, binding in (
        ("invocation predictor", invocation_artifact),
        ("duration predictor", duration_artifact),
        ("physical service clock", service_artifact),
    ):
        if binding.get("training_root_ids_sha256") != training_hash:
            raise ValueError(f"{label} training roots differ from --calibration-roots")
    if legacy_compatibility is not None:
        artifact_tuple = legacy_compatibility.get("artifact_tuple")
        if not isinstance(artifact_tuple, Mapping):
            raise ValueError(
                "Gemini legacy compatibility certificate lacks artifact_tuple"
            )
        frozen_by_role = {
            str(row["role"]): row for row in frozen_files if "role" in row
        }
        expected_tuple = {
            "invocation_file_sha256": invocation_artifact["sha256"],
            "invocation_logical_sha256": audit.artifact_identity_sha256(
                invocation_artifact
            ),
            "duration_file_sha256": duration_artifact["sha256"],
            "duration_logical_sha256": audit.artifact_identity_sha256(
                duration_artifact
            ),
            "service_clock_file_sha256": service_artifact["sha256"],
            "service_clock_logical_sha256": audit.artifact_identity_sha256(
                service_artifact
            ),
            "calibration_root_ids_sha256": training_hash,
        }
        for field, expected in expected_tuple.items():
            if artifact_tuple.get(field) != expected:
                raise ValueError(
                    "Gemini legacy compatibility artifact tuple differs for "
                    f"{field}"
                )
        runner_binding = frozen_by_role.get("runner", {})
        if legacy_compatibility.get("current_cell_runner_sha256") != (
            runner_binding.get("sha256")
        ):
            raise ValueError(
                "Gemini compatibility certificate does not bind the frozen cell runner"
            )
        prediction_binding = frozen_by_role.get("prediction_code")
        if (
            isinstance(prediction_binding, Mapping)
            and artifact_tuple.get("invocation_runtime_module_sha256")
            != prediction_binding.get("sha256")
        ):
            raise ValueError(
                "Gemini compatibility certificate does not bind prediction code"
            )
    predictors = {
        "tool_invocation": {
            "training_role": "calibration",
            "uses_evaluation_labels": False,
            "frozen_before_evaluation": True,
            "training_root_ids_sha256": training_hash,
            "input_features": list(args.invocation_feature),
            "artifact": invocation_artifact,
        },
        "tool_duration": {
            "training_role": "calibration",
            "uses_evaluation_labels": False,
            "frozen_before_evaluation": True,
            "training_root_ids_sha256": training_hash,
            "input_features": list(args.duration_feature),
            "artifact": duration_artifact,
        },
    }
    claim_type = {
        "autonomous": "closed_loop_agent",
        "trace_replay_causal_reveal": "systems_trace_replay",
    }[args.call_graph_mode]
    policy = {
        "decision_schema": audit.DECISION_SCHEMA,
        "forbidden_decision_features": sorted(audit.FORBIDDEN_DECISION_FEATURES),
        "cells": copy.deepcopy(audit.CELLS),
        "call_graph_mode": args.call_graph_mode,
        "claim_type": claim_type,
        "offline_tool_credit_s": 0,
    }
    service_clock = {
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
        "artifact": service_artifact,
    }
    groups = _gpu_groups(args.gpu_groups)
    blocks = []
    for cycle in range(args.williams_cycles):
        for order_index, order in enumerate(audit.WILLIAMS_ORDERS):
            blocks.append(
                {
                    "block_id": f"cycle-{cycle + 1:02d}-block-{order_index + 1:02d}",
                    "order": list(order),
                    "gpu_ids": groups[(cycle + order_index) % len(groups)],
                }
            )
    marker = output.with_name(output.name + ".FORMAL_STARTED.json")
    manifest: dict[str, Any] = {
        "schema": audit.MANIFEST_SCHEMA,
        "version": 1,
        "claim_scope": args.claim_scope,
        "data": {
            "calibration_root_ids": calibration,
            "tuning_root_ids": tuning,
            "evaluation_root_ids": evaluation,
            "previously_observed_evaluation_root_ids": exposed,
            "split_unit": "root_trace_or_task",
            "exact_root_disjoint_guard": True,
            "near_duplicate_guard": (
                "verified" if near_duplicate_evidence is not None else "not_verified"
            ),
            "selection_protocol": selection_protocol,
            "evaluation_used_for_model_or_policy_selection": False,
        },
        "freeze": {
            "sealed_before_evaluation": True,
            "no_tuning_after_seal": True,
            "accept_result_regardless_of_direction": True,
            "started_marker_exclusive_create": True,
            "formal_result_used_for_optimization": False,
            "policy_bundle_sha256": "0" * 64,
            "sealed_payload_sha256": "0" * 64,
            "started_marker_path": str(marker),
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "frozen_files": frozen_files,
        "predictors": predictors,
        "policy": policy,
        "execution": {
            **{field: True for field in audit.REQUIRED_EXECUTION_ATTESTATIONS},
            "physical_service_clock": service_clock,
            "treatment_neutral_runtime_parameters": runtime_parameters,
            "blocks": blocks,
        },
        "statistics": {
            "unit": "root_trace_or_task",
            "replicas_folded_within_root": True,
            "blocks_folded_within_root": True,
            "paired": True,
            "confidence_interval_method": "paired_root_cluster_percentile_bootstrap",
            "paired_bootstrap_resamples": args.bootstrap_resamples,
            "paired_bootstrap_seed": args.bootstrap_seed,
            "ci_level": 0.95,
            "primary_contrast": "A_vs_F",
            "estimand": "ratio_of_paired_root_mean_e2e",
            "speedup_threshold": 0.20,
            "pass_rule": "point_estimate_ge_0.20_and_ci_lower_gt_0",
            "strong_claim_requires_ci_lower_ge_0.20": True,
            "report_all_factorial_contrasts": True,
        },
    }
    if near_duplicate_evidence is not None:
        manifest["data"]["near_duplicate_evidence"] = near_duplicate_evidence
    manifest["freeze"]["policy_bundle_sha256"] = audit.policy_bundle_sha256(
        frozen_files=frozen_files,
        predictors=predictors,
        physical_service_clock=service_clock,
        policy=policy,
        treatment_neutral_runtime_parameters=runtime_parameters,
    )
    manifest["freeze"]["sealed_payload_sha256"] = audit.sealed_payload_sha256(
        manifest
    )
    if legacy_compatibility is not None:
        compatibility_errors: list[str] = []
        audit._audit_gemini_legacy_compatibility_manifest(
            manifest,
            base=output.parent,
            verify_files=True,
            errors=compatibility_errors,
        )
        if compatibility_errors:
            raise ValueError(
                "Gemini legacy compatibility verification failed before formal "
                "start: "
                + "; ".join(compatibility_errors[:12])
            )
    marker_payload = {
        "schema": audit.START_MARKER_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_payload_sha256": manifest["freeze"]["sealed_payload_sha256"],
        "creator_pid": os.getpid(),
    }
    preflight = audit.audit_manifest(
        manifest, base=output.parent, verify_files=False, require_evidence=False
    )
    if not preflight["valid"]:
        raise ValueError(
            "refusing to seal an invalid manifest: "
            + "; ".join(preflight["errors"])
        )
    # The marker is the exclusive formal-start operation.  Nothing is
    # overwritten if this protocol identity was already started.
    _write_exclusive(marker, marker_payload)
    _write_exclusive(output, manifest)
    checked = audit.audit_manifest(
        manifest, base=output.parent, verify_files=True, require_evidence=False
    )
    if not checked["valid"]:
        raise ValueError("created manifest failed its own audit: " + "; ".join(checked["errors"]))
    return manifest


def _matrix_rows(
    path: Path,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(
            "matrix index must be an object containing provenance and cell_evidence"
        )
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("matrix index must contain normalized provenance")
    runtime_parameters = value.get("runtime_parameters")
    if not isinstance(runtime_parameters, Mapping):
        raise ValueError("matrix index must contain normalized runtime_parameters")
    rows = value.get("cell_evidence")
    if not isinstance(rows, list):
        raise ValueError("matrix index must contain cell_evidence")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("matrix index rows must be objects")
    return list(rows), provenance, runtime_parameters


def _normalized_platform_evidence(
    value: Any,
    *,
    base: Path,
    label: str,
    required: bool,
) -> dict[str, dict[str, str]]:
    """Resolve and hash-check every row-supplied platform artifact.

    Keeping the complete mapping in the final manifest prevents a wrapper
    from presenting machine/server/hook evidence during finalization and then
    silently dropping it from the paper artifact.
    """

    if not isinstance(value, Mapping) or not value:
        if required:
            raise ValueError(f"{label}: non-empty mapping required")
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for name, binding_raw in sorted(value.items()):
        item_label = f"{label}.{name}"
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}: evidence names must be non-empty strings")
        if not isinstance(binding_raw, Mapping):
            raise ValueError(f"{item_label}: file binding required")
        path_raw = binding_raw.get("path")
        expected_sha = binding_raw.get("sha256")
        if (
            not isinstance(path_raw, str)
            or not path_raw
            or not audit._is_sha256(expected_sha)
        ):
            raise ValueError(f"{item_label}: invalid path/SHA-256 binding")
        path = Path(path_raw)
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not path.is_file() or audit.file_sha256(path) != expected_sha:
            raise ValueError(f"{item_label}: bound file is missing or changed")
        normalized[name] = {"path": str(path), "sha256": str(expected_sha)}
    return normalized


def _require_model_inventory_provenance(
    provenance: Mapping[str, Any], *, expected: str | None, label: str
) -> None:
    if expected is not None and provenance.get(
        "model_snapshot_inventory_sha256"
    ) != expected:
        raise ValueError(
            f"{label}: model_snapshot_inventory_sha256 is missing or differs "
            "from the frozen policy bundle"
        )


def _bind_matrix(
    manifest: dict[str, Any], *, matrix_index: Path
) -> None:
    rows, index_provenance, index_runtime = _matrix_rows(matrix_index)
    expected_provenance = audit.expected_runtime_provenance(manifest)
    manifest_runtime = manifest["execution"]["treatment_neutral_runtime_parameters"]
    expected_runtime = {
        key: copy.deepcopy(manifest_runtime[key])
        for key in ("schema", "parameters", "runtime_parameters_sha256")
    }
    if index_runtime != expected_runtime:
        raise ValueError("matrix runtime parameters differ from sealed manifest")
    expected_model_inventory = audit.model_snapshot_inventory_sha256(
        manifest, base=matrix_index.parent
    )
    expected_result_schema = audit.expected_strict_result_schema(
        manifest, base=matrix_index.parent
    )
    for field, expected_value in expected_provenance.items():
        if index_provenance.get(field) != expected_value:
            raise ValueError(
                f"matrix provenance {field} differs from sealed manifest"
            )
    block_gpu = {
        str(block["block_id"]): list(block["gpu_ids"])
        for block in manifest["execution"]["blocks"]
    }
    block_order = {
        str(block["block_id"]): list(block["order"])
        for block in manifest["execution"]["blocks"]
    }
    evidence = []
    block_intervals: dict[str, dict[int, tuple[float, float]]] = {}
    for index, row in enumerate(rows):
        block_id = row.get("block_id")
        cell = row.get("cell")
        if block_id not in block_gpu or cell not in audit.CELLS:
            raise ValueError(f"matrix row {index}: invalid block/cell")
        expected_position = block_order[str(block_id)].index(str(cell)) + 1
        if row.get("order_position") != expected_position:
            raise ValueError(
                f"matrix row {index}: order_position must be {expected_position}"
            )
        row_provenance = row.get("provenance")
        if not isinstance(row_provenance, Mapping) or row_provenance != index_provenance:
            raise ValueError(
                f"matrix row {index}: complete provenance missing or differs "
                "from matrix-level binding"
            )
        path_raw = row.get("result_path")
        if not isinstance(path_raw, str) or not path_raw:
            raise ValueError(f"matrix row {index}: missing result_path")
        result_path = Path(path_raw)
        if not result_path.is_absolute():
            result_path = matrix_index.parent / result_path
        if result_path.is_dir():
            result_path = result_path / "result.json"
        result_path = result_path.resolve()
        if not result_path.is_file():
            raise ValueError(f"matrix row {index}: missing result {result_path}")
        result = _read_json(result_path)
        if not isinstance(result, Mapping) or result.get("schema") != expected_result_schema:
            raise ValueError(
                f"matrix row {index}: result schema differs from frozen policy bundle"
            )
        result_errors = audit.audit_result_payload(result)
        if result_errors:
            raise ValueError(
                f"matrix row {index}: result audit failed: " + "; ".join(result_errors[:8])
            )
        paper = result.get("paper_protocol") if isinstance(result, Mapping) else None
        if not isinstance(paper, Mapping) or paper.get("claim_scope") != manifest.get(
            "claim_scope"
        ):
            raise ValueError(
                f"matrix row {index}: result claim_scope differs from sealed manifest"
            )
        if paper.get("cell") != cell:
            raise ValueError(f"matrix row {index}: result cell differs from binding")
        result_provenance = result.get("provenance")
        if not isinstance(result_provenance, Mapping):
            raise ValueError(f"matrix row {index}: result lacks normalized provenance")
        for field, expected_value in expected_provenance.items():
            if result_provenance.get(field) != expected_value:
                raise ValueError(
                    f"matrix row {index}: result provenance {field} differs "
                    "from sealed manifest"
                )
        if dict(result_provenance) != dict(index_provenance):
            raise ValueError(
                f"matrix row {index}: complete result provenance differs from matrix index"
            )
        _require_model_inventory_provenance(
            index_provenance,
            expected=expected_model_inventory,
            label=f"matrix row {index} provenance",
        )
        if result.get("runtime_parameters") != expected_runtime:
            raise ValueError(
                f"matrix row {index}: runtime parameters differ from sealed manifest"
            )
        runtime_environment_contract: dict[str, Any] | None = None
        runtime_environment_evidence: dict[str, str] | None = None
        scheduler_runtime_contract: dict[str, Any] | None = None
        platform_evidence = _normalized_platform_evidence(
            row.get("platform_evidence"),
            base=matrix_index.parent,
            label=f"matrix row {index} platform_evidence",
            required=expected_model_inventory is not None,
        )
        if expected_model_inventory is not None:
            environment_errors: list[str] = []
            result_environment = audit._audit_runtime_environment_contract(
                result.get("runtime_environment_contract"),
                expected_model_inventory_sha256=expected_model_inventory,
                label=f"matrix row {index} result.runtime_environment_contract",
                errors=environment_errors,
            )
            row_environment = audit._audit_runtime_environment_contract(
                row.get("runtime_environment_contract"),
                expected_model_inventory_sha256=expected_model_inventory,
                label=f"matrix row {index} runtime_environment_contract",
                errors=environment_errors,
            )
            if environment_errors:
                raise ValueError("; ".join(environment_errors))
            if result_environment != row_environment:
                raise ValueError(
                    f"matrix row {index}: runtime environment differs from result"
                )
            runtime_environment_contract = copy.deepcopy(dict(result_environment))
            environment_binding = platform_evidence.get("runtime_environment")
            if not isinstance(environment_binding, Mapping):
                raise ValueError(
                    f"matrix row {index}: runtime environment evidence binding is missing"
                )
            environment_sha256 = environment_binding.get("sha256")
            if (
                not audit._is_sha256(environment_sha256)
                or environment_sha256
                != runtime_environment_contract.get("evidence_sha256")
            ):
                raise ValueError(
                    f"matrix row {index}: runtime environment evidence identity differs"
                )
            runtime_environment_evidence = {
                "path": str(environment_binding["path"]),
                "sha256": str(environment_sha256),
            }
        server = row.get("server_instance_id")
        broker = row.get("broker_instance_id")
        if not isinstance(server, str) or not server or not isinstance(broker, str) or not broker:
            raise ValueError(f"matrix row {index}: server/broker instance IDs are required")
        gpu_ids = row.get("gpu_ids", block_gpu[str(block_id)])
        if gpu_ids != block_gpu[str(block_id)]:
            raise ValueError(f"matrix row {index}: GPU IDs differ from registered block")
        if expected_model_inventory is not None:
            platform_errors: list[str] = []
            audit._audit_platform_evidence(
                platform_evidence,
                base=matrix_index.parent,
                verify_files=True,
                expected_model_inventory_sha256=expected_model_inventory,
                runtime_environment_contract=(runtime_environment_contract or {}),
                cell=str(cell),
                block_id=str(block_id),
                order_position=expected_position,
                expected_provenance=expected_provenance,
                gpu_ids=gpu_ids,
                server_instance_id=str(server),
                label=f"matrix row {index} platform_evidence",
                errors=platform_errors,
            )
            if platform_errors:
                raise ValueError("; ".join(platform_errors))
        result_scheduler_contract = result.get("scheduler_runtime_contract")
        row_scheduler_contract = row.get("scheduler_runtime_contract")
        if result_scheduler_contract is not None or row_scheduler_contract is not None:
            if (
                not isinstance(result_scheduler_contract, Mapping)
                or not isinstance(row_scheduler_contract, Mapping)
                or dict(result_scheduler_contract) != dict(row_scheduler_contract)
            ):
                raise ValueError(
                    f"matrix row {index}: scheduler runtime contract missing or "
                    "differs from result"
                )
            scheduler_errors: list[str] = []
            checked_scheduler = audit._audit_qwen_scheduler_runtime_evidence(
                row_scheduler_contract,
                platform=platform_evidence,
                base=matrix_index.parent,
                verify_files=True,
                cell=str(cell),
                expected_scheduler_hook_sha256=str(
                    expected_provenance["scheduler_hook_file_sha256"]
                ),
                label=f"matrix row {index} scheduler_runtime_contract",
                errors=scheduler_errors,
            )
            if scheduler_errors:
                raise ValueError("; ".join(scheduler_errors))
            scheduler_runtime_contract = copy.deepcopy(dict(checked_scheduler))
        started_wall = row.get("started_wall_s")
        ended_wall = row.get("ended_wall_s")
        if (
            not audit._is_number(started_wall)
            or not audit._is_number(ended_wall)
            or float(started_wall) < 0.0
            or float(ended_wall) < float(started_wall)
        ):
            raise ValueError(f"matrix row {index}: invalid wall-clock interval")
        block_intervals.setdefault(str(block_id), {})[expected_position] = (
            float(started_wall),
            float(ended_wall),
        )
        for result_field, expected in (
            ("block_id", block_id),
            ("order_position", expected_position),
            ("gpu_ids", gpu_ids),
            ("server_instance_id", server),
            ("broker_instance_id", broker),
            ("started_wall_s", started_wall),
            ("ended_wall_s", ended_wall),
        ):
            if result.get(result_field) != expected:
                raise ValueError(
                    f"matrix row {index}: result {result_field} differs from binding"
                )
        evidence_row = {
                "block_id": block_id,
                "cell": cell,
                "order_position": expected_position,
                "gpu_ids": gpu_ids,
                "server_instance_id": server,
                "broker_instance_id": broker,
                "started_wall_s": float(started_wall),
                "ended_wall_s": float(ended_wall),
                "policy_bundle_sha256": manifest["freeze"]["policy_bundle_sha256"],
                "service_clock_artifact_sha256": manifest["execution"][
                    "physical_service_clock"
                ]["artifact"].get(
                    "identity_sha256",
                    manifest["execution"]["physical_service_clock"]["artifact"]["sha256"],
                ),
                "runtime_parameters_sha256": expected_runtime[
                    "runtime_parameters_sha256"
                ],
                "provenance": copy.deepcopy(dict(index_provenance)),
                "result_path": str(result_path),
                "result_sha256": audit.file_sha256(result_path),
            }
        if runtime_environment_contract is not None:
            evidence_row["runtime_environment_contract"] = (
                runtime_environment_contract
            )
            evidence_row["runtime_environment_evidence"] = (
                runtime_environment_evidence
            )
        if platform_evidence:
            evidence_row["platform_evidence"] = platform_evidence
        if scheduler_runtime_contract is not None:
            evidence_row["scheduler_runtime_contract"] = scheduler_runtime_contract
        evidence.append(evidence_row)
    for block_id, intervals in block_intervals.items():
        if set(intervals) != {1, 2, 3, 4}:
            raise ValueError(f"matrix block {block_id}: incomplete order positions")
        for position in range(1, 4):
            if intervals[position + 1][0] < intervals[position][1]:
                raise ValueError(
                    f"matrix block {block_id}: positions {position} and "
                    f"{position + 1} overlap or ran out of order"
                )
    manifest["cell_evidence"] = evidence


def finalize_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    manifest_raw = _read_json(source)
    if not isinstance(manifest_raw, Mapping):
        raise ValueError("manifest root must be an object")
    manifest = copy.deepcopy(dict(manifest_raw))
    checked = audit.audit_manifest(
        manifest, base=source.parent, verify_files=True, require_evidence=False
    )
    if not checked["valid"]:
        raise ValueError("input manifest failed audit: " + "; ".join(checked["errors"][:8]))
    if (args.matrix_index is None) == (args.analysis is None):
        raise ValueError("finalize requires exactly one of --matrix-index or --analysis")
    if "preregistered_manifest" not in manifest:
        manifest["preregistered_manifest"] = _binding(source)
    if args.matrix_index is not None:
        if "cell_evidence" in manifest:
            raise ValueError("manifest already contains cell evidence")
        _bind_matrix(manifest, matrix_index=args.matrix_index.resolve())
        require_evidence = False
    else:
        if "cell_evidence" not in manifest:
            raise ValueError("attach analysis only after cell evidence is bound")
        analysis_path = args.analysis.resolve()
        analysis = _read_json(analysis_path)
        if not isinstance(analysis, Mapping) or analysis.get("schema") != (
            "paste.paper.strict_causal_abef_analysis.v1"
        ):
            raise ValueError("unsupported analysis report")
        if analysis.get("manifest_sha256") != audit.file_sha256(source):
            raise ValueError("analysis report is not bound to the input evidence manifest")
        analysis_unsigned = dict(analysis)
        declared_analysis_hash = analysis_unsigned.pop("analysis_sha256", None)
        if declared_analysis_hash != analyzer._sha256(analysis_unsigned):
            raise ValueError("analysis report canonical analysis_sha256 mismatch")
        trusted_analysis = analyzer.analyze_manifest(source)
        if analysis != trusted_analysis:
            raise ValueError(
                "analysis report differs from a trusted recomputation over bound evidence"
            )
        outcomes = analysis.get("manifest_outcomes")
        if not isinstance(outcomes, Mapping):
            raise ValueError("analysis report lacks manifest_outcomes")
        manifest["outcomes"] = copy.deepcopy(dict(outcomes))
        manifest["analysis_evidence_manifest"] = _binding(source)
        manifest["analysis_report"] = _binding(
            analysis_path,
            identity_fields=("analysis_sha256",),
        )
        require_evidence = True
    checked = audit.audit_manifest(
        manifest,
        base=output.parent,
        verify_files=True,
        require_evidence=require_evidence,
    )
    if not checked["valid"]:
        raise ValueError("finalized manifest failed audit: " + "; ".join(checked["errors"][:12]))
    _write_exclusive(output, manifest)
    return manifest


def _add_create_parser(commands: argparse._SubParsersAction[Any]) -> None:
    create = commands.add_parser("create", help="seal a preregistered pre-run manifest")
    create.add_argument("output", type=Path)
    create.add_argument("--claim-scope", choices=["retrospective", "confirmatory"], required=True)
    create.add_argument("--calibration-roots", type=Path, required=True)
    create.add_argument("--tuning-roots", type=Path, required=True)
    create.add_argument("--evaluation-roots", type=Path, required=True)
    create.add_argument("--exposed-roots", type=Path)
    create.add_argument(
        "--near-duplicate-evidence",
        type=Path,
        help=(
            "bound paste.paper.near_duplicate_audit.v1 JSON; required for "
            "confirmatory scope and optional for retrospective scope"
        ),
    )
    create.add_argument(
        "--selection-protocol",
        choices=[
            "heldout_tuning_split",
            "nested_cross_validation_within_calibration",
        ],
        required=True,
        help=(
            "use heldout_tuning_split with a non-empty tuning file, or "
            "nested_cross_validation_within_calibration with an explicitly empty file"
        ),
    )
    create.add_argument(
        "--frozen-file",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="repeat for protocol, runner, policy_bundle, config, scheduler_hook, and extras",
    )
    create.add_argument("--invocation-predictor-artifact", type=Path, required=True)
    create.add_argument("--duration-predictor-artifact", type=Path, required=True)
    create.add_argument("--service-clock-artifact", type=Path, required=True)
    create.add_argument(
        "--runtime-parameters-json",
        type=Path,
        required=True,
        help=(
            "signed paste.paper.treatment_neutral_runtime.v1 JSON containing "
            "the actual cross-cell server/workload/tool-capacity parameters"
        ),
    )
    create.add_argument(
        "--service-clock-identity-sha256",
        help="logical signed surface hash when it differs from the containing file hash",
    )
    create.add_argument(
        "--invocation-feature",
        action="append",
        default=[],
        help="repeat for each exact policy-visible invocation-predictor input",
    )
    create.add_argument(
        "--duration-feature",
        action="append",
        default=[],
        help="repeat for each exact policy-visible duration-predictor input",
    )
    create.add_argument(
        "--call-graph-mode",
        choices=sorted(audit.CALL_GRAPH_MODES),
        default="trace_replay_causal_reveal",
    )
    create.add_argument("--gpu-groups", default="0,1,2,3;4,5,6,7")
    create.add_argument("--williams-cycles", type=int, default=1)
    create.add_argument("--bootstrap-resamples", type=int, default=10_000)
    create.add_argument("--bootstrap-seed", default="strict-causal-paper-v1")
    create.set_defaults(func=create_manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    _add_create_parser(commands)
    finalize = commands.add_parser("finalize", help="bind results or attach analyzed outcomes")
    finalize.add_argument("manifest", type=Path)
    finalize.add_argument("output", type=Path)
    finalize.add_argument("--matrix-index", type=Path)
    finalize.add_argument("--analysis", type=Path)
    finalize.set_defaults(func=finalize_manifest)
    return result


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "williams_cycles", 1) < 1:
        print(json.dumps({"valid": False, "error": "Williams cycles must be positive"}))
        return 2
    if getattr(args, "bootstrap_resamples", 10_000) < 10_000:
        print(json.dumps({"valid": False, "error": "bootstrap resamples must be >=10000"}))
        return 2
    try:
        value = args.func(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "output": str(args.output.resolve()),
                "sealed_payload_sha256": value["freeze"]["sealed_payload_sha256"],
                "policy_bundle_sha256": value["freeze"]["policy_bundle_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
