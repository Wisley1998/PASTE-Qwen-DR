#!/usr/bin/env python3
"""Fail-closed audit for the oracle-free PASTE paper protocol.

The auditor deliberately does not infer that a value is causal from a field
comment or from a result's self-description.  A formal run must use the
explicit ``*_hat`` decision schema, bind frozen artifacts by SHA-256, execute
the preregistered A/B/E/F matrix, and keep trace outcomes outside every policy
decision object.

This file is independent of the historical Qwen and Gemini replay runners so
it can reject their retrospective outputs without changing or legitimising
those outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_SCHEMA = "paste.paper.strict_causal_experiment.v1"
DECISION_SCHEMA = "paste.schedx.causal_prediction.v1"
QWEN_RESULT_SCHEMA = "paste_repro.strict_trace_abef_result.v1"
GEMINI_RESULT_SCHEMA = "paste_gemini.swe_strict_abef_result.v1"
STRICT_RESULT_SCHEMAS = frozenset({QWEN_RESULT_SCHEMA, GEMINI_RESULT_SCHEMA})
POLICY_BUNDLE_RESULT_SCHEMAS = {
    "paste_repro.strict_trace_abef_bundle.v1": QWEN_RESULT_SCHEMA,
    "paste_gemini.swe_strict_policy_plan.v1": GEMINI_RESULT_SCHEMA,
    # Minimal shared-test workload bundle.  It deliberately uses the Qwen
    # normalized result shape while exercising repository-neutral analysis.
    "paste.paper.registered_workload_contract.v1": QWEN_RESULT_SCHEMA,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MIN_CONFIRMATORY_ROOTS = 30
CALL_GRAPH_MODES = frozenset({"autonomous", "trace_replay_causal_reveal"})
PHYSICAL_SERVICE_CLOCK_MODES = frozenset(
    {"live_isolated_tool", "calibration_hashed_empirical_v1"}
)
START_MARKER_SCHEMA = "paste.paper.strict_causal_start_marker.v1"
NEAR_DUPLICATE_AUDIT_SCHEMA = "paste.paper.near_duplicate_audit.v1"
RUNTIME_PARAMETERS_SCHEMA = "paste.paper.treatment_neutral_runtime.v1"
GEMINI_LEGACY_COMPATIBILITY_SCHEMA = (
    "paste_gemini.swe_strict_legacy_frozen_compatibility.v1"
)
GEMINI_LEGACY_COMPATIBILITY_MODE = (
    "single_allowlisted_frozen_tuple_declarative_behavior_v1"
)
GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE = (
    "legacy_compatibility_certificate"
)
GEMINI_LEGACY_COMPATIBILITY_VERIFIER_ROLE = "legacy_compatibility_verifier"
REQUIRED_FROZEN_FILE_ROLES = frozenset(
    {
        "protocol",
        "runner",
        "policy_bundle",
        "config",
        "scheduler_hook",
        "materializer",
        "auditor",
        "analyzer",
    }
)

# Runtime results bind the bytes that were actually opened by the cell, while
# signed JSON artifacts additionally bind their logical (self-authenticated)
# identity.  Keeping these namespaces separate prevents a wrapper from passing
# an unregistered artifact whose internal hash merely resembles a file hash.
RUNTIME_PROVENANCE_FIELDS = (
    "runner_file_sha256",
    "policy_bundle_file_sha256",
    "config_file_sha256",
    "scheduler_hook_file_sha256",
    "invocation_predictor_file_sha256",
    "invocation_predictor_artifact_sha256",
    "duration_predictor_file_sha256",
    "duration_predictor_artifact_sha256",
    "service_clock_file_sha256",
    "service_clock_artifact_sha256",
    "runtime_parameters_file_sha256",
    "runtime_parameters_artifact_sha256",
)

# These identities are required only when a manifest explicitly binds the
# one-off Gemini legacy-artifact compatibility proof.  Keeping them separate
# from ``RUNTIME_PROVENANCE_FIELDS`` leaves Qwen manifests and results byte-for-
# byte unchanged when the two compatibility roles are absent.
GEMINI_LEGACY_COMPATIBILITY_PROVENANCE_FIELDS = (
    "legacy_compatibility_certificate_file_sha256",
    "legacy_compatibility_certificate_sha256",
    "legacy_compatibility_verifier_file_sha256",
)

REQUIRED_TREATMENT_NEUTRAL_RUNTIME_KEYS = frozenset(
    {
        "model_id",
        "model_revision",
        "server_host",
        "server_port",
        "tensor_parallel_size",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "cuda_graph_sizes",
        "prefix_caching",
        "vllm_v1",
        "max_active_tasks",
        "tool_capacity",
        "configured_speculation_capacity",
        "request_timeout_s",
        "public_output_cap",
        "workload_instances",
        "arrival_schedule_sha256",
    }
)

FORBIDDEN_TREATMENT_NEUTRAL_RUNTIME_KEYS = frozenset(
    {
        "cell",
        "scheduler",
        "scheduler_policy",
        "speculation",
        "speculation_enabled",
        "effective_speculation_capacity",
        "gpu_ids",
        "server_instance_id",
        "broker_instance_id",
    }
)

CELLS: dict[str, dict[str, str]] = {
    "A": {"scheduler": "native_fcfs", "speculation": "off"},
    "B": {"scheduler": "native_fcfs", "speculation": "online_causal"},
    "E": {"scheduler": "causal_joint", "speculation": "off"},
    "F": {"scheduler": "causal_joint", "speculation": "online_causal"},
}

# A balanced Williams design: every cell occupies every ordinal once and every
# first-order direction is balanced over a complete four-block cycle.
WILLIAMS_ORDERS: tuple[tuple[str, ...], ...] = (
    ("A", "B", "F", "E"),
    ("B", "E", "A", "F"),
    ("E", "F", "B", "A"),
    ("F", "A", "E", "B"),
)

ALLOWED_DECISION_FEATURES = frozenset(
    {
        "current_time",
        "release_time",
        "wait_age",
        "opaque_session_id",
        "current_call_index",
        "current_prompt_tokens",
        "current_max_tokens",
        "current_generated_tokens",
        "initial_user_prompt",
        "current_messages",
        "committed_tool_names",
        "committed_tool_arguments",
        "committed_tool_results",
        "completed_tool_service_times",
        "engine_running_count",
        "engine_waiting_count",
        "kv_cache_usage",
        "prefix_cache_state",
        "broker_queue_state",
        "candidate_tool_name",
        "candidate_tool_arguments",
        "candidate_host",
        "candidate_repository",
        "candidate_input_size",
        "frozen_prediction",
        "causal_ewma",
        "current_visible_search_result_urls",
        "current_visible_search_result_ranks",
        "current_visible_search_result_ordinals",
        "frozen_top_k",
        "current_tool_name",
        "current_normalized_visit_domain",
        "completed_job_service_s_ewma",
        "last_completed_tool_name",
        "session.session_id",
        "session.attempt_id",
        "session.environment_fingerprint",
        "session.repository",
        "session.base_commit",
        "prompt.test_spec_kind",
        "prompt.exact_pytest_nodeids",
        "prompt.django_structured_tests",
        "prompt.broad_test_tokens",
        "prompt.explicit_files",
        "prompt.identifier_tokens_first_24",
        "committed_call_count",
        "committed_calls.tool_names",
        "committed_calls.arguments_canonical_json",
        "committed_calls.results_canonical_json",
        "derived_state.workspace_epoch",
        "derived_state.environment_epoch",
        "derived_state.workflow_prefix_sha256",
    }
)

FORBIDDEN_DECISION_FEATURES = frozenset(
    {
        "future_tool_duration",
        "future_tool_name",
        "future_tool_arguments",
        "future_tool_result",
        "future_output_tokens",
        "future_prompt_tokens",
        "future_call_count",
        "future_arrival",
        "eventual_prediction_hit",
        "eventual_state_accepted",
        "offline_saved_time",
        "offline_readiness",
        "expected_url",
        "trace_tail",
        "evaluation_label",
        "other_evaluation_task_outcome",
    }
)

# These compact fields were populated from full future traces by the previous
# hybrid runners.  Even when a similarly named value could be predicted, the
# strict schema requires an explicit ``*_hat`` spelling and artifact binding.
LEGACY_ORACLE_METADATA_FIELDS = frozenset(
    {"n", "rc", "rlmt", "npt", "nmt", "nw", "nwc", "rtw", "eg", "is_final"}
)

ALLOWED_SCHEDULER_METADATA_FIELDS = frozenset(
    {
        "ms",
        "decision_seq",
        "observed_event_seq",
        "policy_sha256",
        "predictor_artifact_sha256",
        "duration_predictor_artifact_sha256",
        "t",
        "c",
        "i",
        "pt",
        "mt",
        "po_hat",
        "remaining_calls_hat",
        "remaining_llm_tokens_hat",
        "next_prompt_tokens_hat",
        "next_output_tokens_hat",
        "tool_name_hat",
        "tool_hit_probability_hat",
        "tool_service_s_hat",
        "tool_eta_s_hat",
        "remaining_tool_wait_s_hat",
        "expected_gain_s_hat",
        "tqa",
        "tqs",
        "tra",
        "trs",
        "nps",
        "nrg",
        "ntc",
        "npm",
        "br",
        "brt",
        "npjid",
        "npc",
        "npq",
        "nptq",
        "nper_hat",
        "engine_running",
        "engine_waiting",
        "kv_usage",
        "prefix_cached_tokens",
    }
)

FORBIDDEN_PREDICTION_INPUT_KEYS = frozenset(
    {
        "prediction_hit",
        "hit",
        "actual_key",
        "runtime_key_sha256",
        "actual_duration_s",
        "duration_s",
        "tool_duration_ms",
        "recorded_tool_s",
        "replay_tool_s",
        "offline_saved_s",
        "offline_evidence_saved_s",
        "offline_cache_hit_urls",
        "state_accepted",
        "safe_to_speculate",
        "post_authority_hit",
        "prediction_outcome",
        "expected_url",
        "target_output_tokens",
        "fixed_completion_tokens",
    }
)

LEGACY_UNMARKED_PREDICTION_KEYS = frozenset(
    {
        "predicted_service_s",
        "predicted_duration_s",
        "predicted_tool",
        "confidence",
        "probability",
        "eta_s",
        "remaining_calls",
        "remaining_llm_tokens",
        "next_prompt_tokens",
        "next_output_tokens",
    }
)

PUBLIC_PLAN_FORBIDDEN_FIELDS = frozenset(
    {
        "requests",
        "steps",
        "messages",
        "tools_after",
        "tool_name",
        "tool_args",
        "outcome_id",
        "authority_key",
        "runtime_key",
    }
)

REQUIRED_EXECUTION_ATTESTATIONS = (
    "fresh_server_per_cell",
    "fresh_broker_per_cell",
    "empty_cache_per_cell",
    "drain_after_cell",
    "same_resource_limits_all_cells",
    "speculation_shares_tool_pool",
    "prediction_overhead_in_e2e",
    "recorded_duration_hidden_from_policy",
    "wrong_speculation_uses_real_or_sealed_counterfactual_service",
    "policy_receives_decision_view_only",
    "future_poison_invariance_test_passed",
    "future_state_accepted_poison_invariance_test_passed",
    "actual_duration_not_used_as_prediction_or_priority",
    "authoritative_and_speculative_exact_match_only",
    "result_private_until_authoritative_commit",
    "prediction_sealed_before_authoritative_reveal",
    "authoritative_call_hidden_until_live_llm_completion",
    "physical_service_assignment_policy_independent",
    "physical_service_assignment_future_poison_invariant",
    "same_invocation_service_clock_all_cells",
    "evaluation_trace_duration_diagnostic_only",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def registered_root_sets_sha256(
    calibration: Sequence[str], tuning: Sequence[str], evaluation: Sequence[str]
) -> str:
    """Identify the exact split membership covered by a duplicate audit."""

    return canonical_sha256(
        {
            "calibration_root_ids": sorted(calibration),
            "tuning_root_ids": sorted(tuning),
            "evaluation_root_ids": sorted(evaluation),
        }
    )


def runtime_parameters_sha256(parameters: Mapping[str, Any]) -> str:
    """Hash the treatment-neutral runtime mapping with the paper canonicalizer."""

    return canonical_sha256(
        {"schema": RUNTIME_PARAMETERS_SCHEMA, "parameters": dict(parameters)}
    )


def _audit_runtime_parameters_contract(
    value: Any, *, label: str, require_artifact: bool, errors: list[str]
) -> Mapping[str, Any]:
    contract = _require_mapping(value, label, errors)
    if contract.get("schema") != RUNTIME_PARAMETERS_SCHEMA:
        errors.append(f"{label}.schema: expected {RUNTIME_PARAMETERS_SCHEMA}")
    parameters = _require_mapping(contract.get("parameters"), f"{label}.parameters", errors)
    missing = sorted(REQUIRED_TREATMENT_NEUTRAL_RUNTIME_KEYS - set(parameters))
    if missing:
        errors.append(f"{label}.parameters: missing required keys: {missing}")
    forbidden = sorted(FORBIDDEN_TREATMENT_NEUTRAL_RUNTIME_KEYS & set(parameters))
    if forbidden:
        errors.append(f"{label}.parameters: treatment-specific keys are forbidden: {forbidden}")
    try:
        expected_identity = runtime_parameters_sha256(parameters)
    except (TypeError, ValueError):
        errors.append(f"{label}.parameters: values must be finite canonical JSON")
        expected_identity = None
    if contract.get("runtime_parameters_sha256") != expected_identity:
        errors.append(f"{label}.runtime_parameters_sha256: canonical hash mismatch")

    string_fields = ("model_id", "model_revision", "server_host", "dtype")
    for field in string_fields:
        if not isinstance(parameters.get(field), str) or not parameters.get(field):
            errors.append(f"{label}.parameters.{field}: non-empty string required")
    positive_int_fields = (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_active_tasks",
        "tool_capacity",
        "configured_speculation_capacity",
        "public_output_cap",
        "workload_instances",
    )
    for field in positive_int_fields:
        if type(parameters.get(field)) is not int or int(parameters[field]) <= 0:
            errors.append(f"{label}.parameters.{field}: positive integer required")
    port = parameters.get("server_port")
    if type(port) is not int or not 1 <= int(port) <= 65535:
        errors.append(f"{label}.parameters.server_port: integer in [1,65535] required")
    utilization = parameters.get("gpu_memory_utilization")
    if not _is_number(utilization) or not 0.0 < float(utilization) <= 1.0:
        errors.append(
            f"{label}.parameters.gpu_memory_utilization: number in (0,1] required"
        )
    timeout = parameters.get("request_timeout_s")
    if not _is_number(timeout) or float(timeout) <= 0.0:
        errors.append(f"{label}.parameters.request_timeout_s: positive number required")
    for field in ("prefix_caching", "vllm_v1"):
        if type(parameters.get(field)) is not bool:
            errors.append(f"{label}.parameters.{field}: boolean required")
    graphs = parameters.get("cuda_graph_sizes")
    if not isinstance(graphs, list) or any(
        type(item) is not int or item <= 0 for item in graphs
    ):
        errors.append(f"{label}.parameters.cuda_graph_sizes: positive integer list required")
    if not _is_sha256(parameters.get("arrival_schedule_sha256")):
        errors.append(f"{label}.parameters.arrival_schedule_sha256: invalid SHA-256")

    artifact = contract.get("artifact")
    if require_artifact:
        artifact_mapping = _require_mapping(artifact, f"{label}.artifact", errors)
        _audit_file_binding_shape(
            artifact_mapping, label=f"{label}.artifact", errors=errors
        )
        if artifact_mapping.get("identity_sha256") != expected_identity:
            errors.append(f"{label}.artifact.identity_sha256: logical hash mismatch")
    elif artifact is not None:
        errors.append(f"{label}.artifact: runtime results must not inject a file binding")
    return contract


def artifact_identity_sha256(binding: Mapping[str, Any]) -> Any:
    """Return a logical signed-artifact identity, falling back to file bytes."""

    return binding.get("identity_sha256", binding.get("sha256"))


def expected_runtime_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact runtime identities sealed by ``manifest``.

    The returned mapping deliberately has no paths: paths are transport detail,
    whereas the byte and signed logical hashes are the identities a cell must
    attest it actually consumed.
    """

    frozen_raw = manifest.get("frozen_files")
    frozen_by_role = {
        str(row.get("role")): row
        for row in frozen_raw
        if isinstance(row, Mapping)
    } if isinstance(frozen_raw, list) else {}
    predictors = manifest.get("predictors")
    predictors = predictors if isinstance(predictors, Mapping) else {}
    invocation = predictors.get("tool_invocation")
    invocation = invocation if isinstance(invocation, Mapping) else {}
    invocation_artifact = invocation.get("artifact")
    invocation_artifact = (
        invocation_artifact if isinstance(invocation_artifact, Mapping) else {}
    )
    duration = predictors.get("tool_duration")
    duration = duration if isinstance(duration, Mapping) else {}
    duration_artifact = duration.get("artifact")
    duration_artifact = (
        duration_artifact if isinstance(duration_artifact, Mapping) else {}
    )
    execution = manifest.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    service = execution.get("physical_service_clock")
    service = service if isinstance(service, Mapping) else {}
    service_artifact = service.get("artifact")
    service_artifact = service_artifact if isinstance(service_artifact, Mapping) else {}
    runtime = execution.get("treatment_neutral_runtime_parameters")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    runtime_artifact = runtime.get("artifact")
    runtime_artifact = runtime_artifact if isinstance(runtime_artifact, Mapping) else {}

    result = {
        "runner_file_sha256": frozen_by_role.get("runner", {}).get("sha256"),
        "policy_bundle_file_sha256": frozen_by_role.get("policy_bundle", {}).get(
            "sha256"
        ),
        "config_file_sha256": frozen_by_role.get("config", {}).get("sha256"),
        "scheduler_hook_file_sha256": frozen_by_role.get(
            "scheduler_hook", {}
        ).get("sha256"),
        "invocation_predictor_file_sha256": invocation_artifact.get("sha256"),
        "invocation_predictor_artifact_sha256": invocation_artifact.get(
            "identity_sha256"
        ),
        "duration_predictor_file_sha256": duration_artifact.get("sha256"),
        "duration_predictor_artifact_sha256": duration_artifact.get(
            "identity_sha256"
        ),
        "service_clock_file_sha256": service_artifact.get("sha256"),
        "service_clock_artifact_sha256": service_artifact.get("identity_sha256"),
        "runtime_parameters_file_sha256": runtime_artifact.get("sha256"),
        "runtime_parameters_artifact_sha256": runtime_artifact.get(
            "identity_sha256"
        ),
    }
    compatibility_certificate = frozen_by_role.get(
        GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE
    )
    compatibility_verifier = frozen_by_role.get(
        GEMINI_LEGACY_COMPATIBILITY_VERIFIER_ROLE
    )
    if compatibility_certificate is not None or compatibility_verifier is not None:
        certificate = (
            compatibility_certificate
            if isinstance(compatibility_certificate, Mapping)
            else {}
        )
        verifier = (
            compatibility_verifier
            if isinstance(compatibility_verifier, Mapping)
            else {}
        )
        result.update(
            {
                "legacy_compatibility_certificate_file_sha256": certificate.get(
                    "sha256"
                ),
                "legacy_compatibility_certificate_sha256": certificate.get(
                    "identity_sha256"
                ),
                "legacy_compatibility_verifier_file_sha256": verifier.get(
                    "sha256"
                ),
            }
        )
    return result


def _audit_runtime_provenance_shape(
    value: Any,
    *,
    label: str,
    errors: list[str],
    extra_fields: Sequence[str] = (),
) -> Mapping[str, Any]:
    provenance = _require_mapping(value, label, errors)
    for field in (*RUNTIME_PROVENANCE_FIELDS, *extra_fields):
        if not _is_sha256(provenance.get(field)):
            errors.append(f"{label}.{field}: missing or invalid SHA-256")
    return provenance


def sealed_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash the preregistered portion, excluding only post-run attachments."""

    sealed = copy.deepcopy(dict(value))
    sealed.pop("cell_evidence", None)
    sealed.pop("outcomes", None)
    sealed.pop("preregistered_manifest", None)
    sealed.pop("analysis_evidence_manifest", None)
    sealed.pop("analysis_report", None)
    freeze = sealed.get("freeze")
    if isinstance(freeze, dict):
        freeze.pop("sealed_payload_sha256", None)
    return canonical_sha256(sealed)


def policy_bundle_sha256(
    *,
    frozen_files: Sequence[Mapping[str, Any]],
    predictors: Mapping[str, Any],
    physical_service_clock: Mapping[str, Any],
    policy: Mapping[str, Any],
    treatment_neutral_runtime_parameters: Mapping[str, Any] | None = None,
) -> str:
    """Bind policy/runtime files and all decision/service artifacts."""

    file_rows = sorted(
        (
            {
                "role": row.get("role"),
                "sha256": row.get("sha256"),
            }
            for row in frozen_files
        ),
        key=lambda row: str(row["role"]),
    )
    predictor_rows = {
        name: {
            "artifact_file_sha256": (
                value.get("artifact", {}).get("sha256")
                if isinstance(value, Mapping)
                and isinstance(value.get("artifact"), Mapping)
                else None
            ),
            "artifact_identity_sha256": (
                artifact_identity_sha256(value.get("artifact", {}))
                if isinstance(value, Mapping)
                and isinstance(value.get("artifact"), Mapping)
                else None
            ),
            "input_features": (
                value.get("input_features") if isinstance(value, Mapping) else None
            ),
            "training_root_ids_sha256": (
                value.get("training_root_ids_sha256")
                if isinstance(value, Mapping)
                else None
            ),
        }
        for name, value in sorted(predictors.items())
    }
    service_artifact = physical_service_clock.get("artifact")
    return canonical_sha256(
        {
            "frozen_files": file_rows,
            "predictors": predictor_rows,
            "physical_service_clock": {
                key: value
                for key, value in physical_service_clock.items()
                if key != "artifact"
            },
            "physical_service_clock_artifact_sha256": (
                artifact_identity_sha256(service_artifact)
                if isinstance(service_artifact, Mapping)
                else None
            ),
            "physical_service_clock_file_sha256": (
                service_artifact.get("sha256")
                if isinstance(service_artifact, Mapping)
                else None
            ),
            "policy": dict(policy),
            "treatment_neutral_runtime_parameters": (
                {
                    key: value
                    for key, value in treatment_neutral_runtime_parameters.items()
                    if key != "artifact"
                }
                if isinstance(treatment_neutral_runtime_parameters, Mapping)
                else None
            ),
            "treatment_neutral_runtime_file_sha256": (
                treatment_neutral_runtime_parameters.get("artifact", {}).get("sha256")
                if isinstance(treatment_neutral_runtime_parameters, Mapping)
                and isinstance(
                    treatment_neutral_runtime_parameters.get("artifact"), Mapping
                )
                else None
            ),
        }
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _numbers_close(left: float, right: float) -> bool:
    """Tolerate sub-millisecond clock/summation noise, not missing work."""

    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-3)


def _result_task_rows(payload: Mapping[str, Any]) -> Any:
    """Return the cross-runner task ledger without accepting a count as evidence."""

    rows = payload.get("task_results")
    if isinstance(rows, list):
        return rows
    rows = payload.get("tasks")
    return rows if isinstance(rows, list) else None


def _audit_task_timing_evidence(payload: Mapping[str, Any]) -> list[str]:
    """Verify raw per-task E2E clocks and every redundant duration summary.

    The paper estimand is derived from ``terminal - scheduled_release``.  A
    runner-provided ``flow_s``/``e2e_s`` is only a redundant checksum and must
    never be usable as the source of the headline result.
    """

    errors: list[str] = []
    started = payload.get("experiment_started_monotonic_s")
    ended = payload.get("experiment_ended_monotonic_s")
    if not _is_number(started) or float(started) < 0.0:
        errors.append("$.experiment_started_monotonic_s: non-negative finite number required")
        return errors
    if not _is_number(ended) or float(ended) < float(started):
        errors.append(
            "$.experiment_ended_monotonic_s: finite value no earlier than experiment start required"
        )
        return errors
    started_s = float(started)
    ended_s = float(ended)
    experiment_s = ended_s - started_s

    if "experiment_wall_s" in payload:
        wall = payload.get("experiment_wall_s")
        if not _is_number(wall) or not _numbers_close(float(wall), experiment_s):
            errors.append(
                "$.experiment_wall_s: does not equal raw monotonic experiment interval"
            )
    summary = payload.get("summary")
    if isinstance(summary, Mapping) and "makespan_s" in summary:
        makespan = summary.get("makespan_s")
        if not _is_number(makespan) or not _numbers_close(
            float(makespan), experiment_s
        ):
            errors.append(
                "$.summary.makespan_s: does not equal raw monotonic experiment interval"
            )

    rows = _result_task_rows(payload)
    if not isinstance(rows, list) or not rows:
        errors.append("$.task_results/tasks: non-empty raw task timing ledger required")
        return errors
    runtime = payload.get("runtime_parameters")
    parameters = runtime.get("parameters") if isinstance(runtime, Mapping) else None
    expected_tasks = (
        parameters.get("workload_instances")
        if isinstance(parameters, Mapping)
        else None
    )
    if type(expected_tasks) is not int or expected_tasks <= 0:
        errors.append(
            "$.runtime_parameters.parameters.workload_instances: positive integer required for task ledger"
        )
    elif len(rows) != expected_tasks:
        errors.append(
            "$.task_results/tasks: task count does not equal frozen workload_instances "
            f"({len(rows)} != {expected_tasks})"
        )
    task_intervals: dict[str, tuple[float, float]] = {}
    for index, row_raw in enumerate(rows):
        path = f"$.task_results/tasks[{index}]"
        if not isinstance(row_raw, Mapping):
            errors.append(f"{path}: expected object")
            continue
        task_id = next(
            (
                value
                for field in ("task_id", "trace_id", "root_instance_id")
                if isinstance((value := row_raw.get(field)), str) and value
            ),
            None,
        )
        if task_id is None:
            errors.append(f"{path}: stable task_id/trace_id is required")
        elif task_id in task_intervals:
            errors.append(f"{path}: duplicate task identity {task_id!r}")
        release_offset = row_raw.get("release_offset_s")
        scheduled = row_raw.get("scheduled_release_monotonic_s")
        released = row_raw.get("released_at_monotonic_s")
        terminal = row_raw.get("task_terminal_monotonic_s")
        for field, value in (
            ("release_offset_s", release_offset),
            ("scheduled_release_monotonic_s", scheduled),
            ("released_at_monotonic_s", released),
            ("task_terminal_monotonic_s", terminal),
        ):
            if not _is_number(value) or float(value) < 0.0:
                errors.append(f"{path}.{field}: non-negative finite number required")
        if not all(
            _is_number(value)
            for value in (release_offset, scheduled, released, terminal)
        ):
            continue
        release_offset_s = float(release_offset)
        scheduled_s = float(scheduled)
        released_s = float(released)
        terminal_s = float(terminal)
        expected_scheduled = started_s + release_offset_s
        if not _numbers_close(scheduled_s, expected_scheduled):
            errors.append(
                f"{path}.scheduled_release_monotonic_s: does not equal "
                "experiment start plus release_offset_s"
            )
        if scheduled_s < started_s or scheduled_s > ended_s:
            errors.append(f"{path}: scheduled release lies outside experiment interval")
        if released_s < scheduled_s:
            errors.append(f"{path}: task was released before its scheduled release")
        if terminal_s < released_s:
            errors.append(f"{path}: task terminal precedes task release")
        if terminal_s > ended_s:
            errors.append(f"{path}: task terminal lies after experiment end")
        if task_id is not None and task_id not in task_intervals:
            task_intervals[task_id] = (scheduled_s, terminal_s)
        raw_e2e_s = terminal_s - scheduled_s
        if raw_e2e_s <= 0.0:
            errors.append(f"{path}: raw task E2E must be positive")
        summaries = [field for field in ("e2e_s", "flow_s") if field in row_raw]
        if not summaries:
            errors.append(f"{path}: redundant e2e_s or flow_s checksum required")
        for field in summaries:
            value = row_raw.get(field)
            if (
                not _is_number(value)
                or float(value) <= 0.0
                or not _numbers_close(float(value), raw_e2e_s)
            ):
                errors.append(
                    f"{path}.{field}: does not equal raw terminal-minus-scheduled E2E"
                )
        for lag_field in ("release_lag_s", "released_lag_s"):
            if lag_field not in row_raw:
                continue
            lag = row_raw.get(lag_field)
            if not _is_number(lag) or not _numbers_close(
                float(lag), released_s - scheduled_s
            ):
                errors.append(
                    f"{path}.{lag_field}: does not equal raw released-minus-scheduled lag"
                )

    event_time_fields = {
        "llm_events": ("llm_completed_at_monotonic_s",),
        "prediction_decisions": (
            "decided_at_monotonic_s",
            "speculative_start_at_monotonic_s",
        ),
        "prediction_outcomes": ("resolved_at_monotonic_s",),
        "tool_events": (
            "llm_completed_at_monotonic_s",
            "authoritative_revealed_at_monotonic_s",
            "tool_completed_at_monotonic_s",
        ),
        "speculation_execution_events": (
            "admitted_at_monotonic_s",
            "physical_started_at_monotonic_s",
            "authority_claimed_at_monotonic_s",
            "terminal_at_monotonic_s",
        ),
    }
    for collection, time_fields in event_time_fields.items():
        events = payload.get(collection, [])
        if not isinstance(events, list):
            continue
        for index, event in enumerate(events):
            path = f"$.{collection}[{index}]"
            if not isinstance(event, Mapping):
                continue
            present_fields = [field for field in time_fields if event.get(field) is not None]
            if not present_fields:
                continue
            event_task_id = next(
                (
                    value
                    for field in ("task_id", "trace_id", "root_instance_id")
                    if isinstance((value := event.get(field)), str) and value
                ),
                None,
            )
            if event_task_id not in task_intervals:
                errors.append(f"{path}: timed event has no matching task interval")
                continue
            task_started_s, task_terminal_s = task_intervals[str(event_task_id)]
            for field in present_fields:
                value = event.get(field)
                if not _is_number(value):
                    errors.append(f"{path}.{field}: finite timestamp required")
                    continue
                value_s = float(value)
                if value_s < task_started_s or value_s > task_terminal_s:
                    errors.append(
                        f"{path}.{field}: timestamp lies outside scheduled-to-terminal task interval"
                    )
    return errors


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _metadata_objects(value: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for path, node in _walk(value):
        if not isinstance(node, Mapping):
            continue
        if path.endswith(".scheduler_metadata") or node.get("ms") == DECISION_SCHEMA:
            yield path, node


def _prediction_decisions(value: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for path, node in _walk(value):
        if not isinstance(node, Mapping):
            continue
        if (
            node.get("record_type") == "prediction_decision"
            or path.endswith(".prediction_decision")
            or re.search(r"\.prediction_decisions\[\d+\]$", path)
        ):
            yield path, node


def _event_identity(
    row: Mapping[str, Any], path: str, errors: list[str]
) -> tuple[str, int] | None:
    trace_id = row.get("trace_id")
    request_index = row.get("request_index")
    if not isinstance(trace_id, str) or not trace_id:
        errors.append(f"{path}.trace_id: missing")
        return None
    if type(request_index) is not int or request_index < 0:
        errors.append(f"{path}.request_index: invalid")
        return None
    return trace_id, request_index


def _event_seq_time(
    row: Mapping[str, Any],
    *,
    seq_field: str,
    time_field: str,
    path: str,
    errors: list[str],
) -> tuple[int, float] | None:
    seq = row.get(seq_field)
    timestamp = row.get(time_field)
    if type(seq) is not int or seq < 0:
        errors.append(f"{path}.{seq_field}: invalid")
        return None
    if not _is_number(timestamp) or float(timestamp) < 0.0:
        errors.append(f"{path}.{time_field}: invalid")
        return None
    return seq, float(timestamp)


STRICT_SPECULATION_TERMINAL_TRANSITIONS = frozenset(
    {
        "completed",
        "cancelled_preempted",
        "cancelled_authority_superseded",
        "cancelled_prediction_miss",
        "cancelled_window_expired",
        "cancelled_session_close",
        "cancelled_pool_close",
    }
)
STRICT_AUTHORITY_BOUNDARY_TRANSITIONS = frozenset(
    {"authority_claimed_inflight", "authority_claimed_completed"}
)


def _audit_strict_speculation_transition_ledger(
    row: Mapping[str, Any], *, path: str, errors: list[str]
) -> dict[str, float | None] | None:
    """Reconstruct job boundaries from a strict runner's raw transitions.

    A completed cache entry may be read by authority more than once.  Every
    callback therefore carries the immutable *first* authority-claim boundary;
    a later access timestamp is not allowed to reclassify already-accounted
    worker occupancy.  Top-level timestamps are redundant projections checked
    against this raw ledger.
    """

    raw = row.get("state_transitions")
    if not isinstance(raw, list) or not raw:
        errors.append(
            f"{path}.state_transitions: non-empty strict raw ledger required"
        )
        return None

    admitted: list[float] = []
    started: list[float] = []
    terminal: list[float] = []
    claims: list[float] = []
    boundary_events: list[tuple[str, float, float | None]] = []
    previous_at: float | None = None
    structurally_valid = True
    for index, transition_raw in enumerate(raw):
        transition_path = f"{path}.state_transitions[{index}]"
        if not isinstance(transition_raw, Mapping):
            errors.append(f"{transition_path}: expected object")
            structurally_valid = False
            continue
        event = transition_raw.get("event")
        at = transition_raw.get("at_monotonic_s")
        if not isinstance(event, str) or not event:
            errors.append(f"{transition_path}.event: non-empty string required")
            structurally_valid = False
            continue
        if not _is_number(at) or float(at) < 0.0:
            errors.append(f"{transition_path}.at_monotonic_s: invalid")
            structurally_valid = False
            continue
        at_value = float(at)
        if previous_at is not None and at_value < previous_at:
            errors.append(f"{transition_path}: transition timestamps are not ordered")
        previous_at = at_value

        if "authority_claimed_at_monotonic_s" not in transition_raw:
            errors.append(
                f"{transition_path}.authority_claimed_at_monotonic_s: "
                "explicit null or immutable first-claim timestamp required"
            )
            structurally_valid = False
            transition_claim = None
        else:
            transition_claim_raw = transition_raw.get(
                "authority_claimed_at_monotonic_s"
            )
            if transition_claim_raw is None:
                transition_claim = None
            elif not _is_number(transition_claim_raw) or float(
                transition_claim_raw
            ) < 0.0:
                errors.append(
                    f"{transition_path}.authority_claimed_at_monotonic_s: invalid"
                )
                structurally_valid = False
                transition_claim = None
            else:
                transition_claim = float(transition_claim_raw)
                if transition_claim > at_value:
                    errors.append(
                        f"{transition_path}: first authority claim is later than "
                        "the callback carrying it"
                    )
                claims.append(transition_claim)

        if event == "admitted":
            admitted.append(at_value)
        elif event == "physical_started":
            started.append(at_value)
        if event in STRICT_SPECULATION_TERMINAL_TRANSITIONS:
            terminal.append(at_value)
        if event in STRICT_AUTHORITY_BOUNDARY_TRANSITIONS:
            boundary_events.append((event, at_value, transition_claim))

    if not structurally_valid:
        return None
    if len(admitted) != 1:
        errors.append(f"{path}.state_transitions: exactly one admission required")
    if len(started) > 1:
        errors.append(f"{path}.state_transitions: at most one physical start allowed")
    if len(terminal) != 1:
        errors.append(f"{path}.state_transitions: exactly one terminal event required")
    if len(admitted) != 1 or len(started) > 1 or len(terminal) != 1:
        return None

    first_claim: float | None = claims[0] if claims else None
    if first_claim is not None:
        if any(value != first_claim for value in claims[1:]):
            errors.append(
                f"{path}.state_transitions: first authority claim changed across callbacks"
            )
        if not any(
            claim is not None
            and claim == first_claim
            and at == first_claim
            for _event, at, claim in boundary_events
        ):
            errors.append(
                f"{path}.state_transitions: no raw authority event establishes "
                "the first-claim boundary"
            )
    elif boundary_events:
        errors.append(
            f"{path}.state_transitions: authority event lacks a first-claim boundary"
        )

    reconstructed = {
        "admitted_at_monotonic_s": admitted[0],
        "physical_started_at_monotonic_s": started[0] if started else None,
        "terminal_at_monotonic_s": terminal[0],
        "authority_claimed_at_monotonic_s": first_claim,
    }
    for field, expected in reconstructed.items():
        observed = row.get(field)
        if expected is None:
            if observed is not None:
                errors.append(
                    f"{path}.{field}: top-level value differs from raw transitions"
                )
        elif not _is_number(observed) or float(observed) != expected:
            errors.append(
                f"{path}.{field}: top-level value differs from raw transitions"
            )
    raw_claimed = first_claim is not None
    if row.get("claimed_by_authority") is not raw_claimed:
        errors.append(
            f"{path}.claimed_by_authority: differs from raw first-claim evidence"
        )
    return reconstructed


def _audit_causal_reveal_events(payload: Mapping[str, Any]) -> list[str]:
    """Audit normalized raw timing evidence for fixed-graph causal reveal."""

    errors: list[str] = []
    # This map is reconstructed only from raw physical execution/authority
    # rows.  A deterministic clock must give bit-identical service to the same
    # normalized invocation key regardless of cell, admission, or eventual
    # hit.  Cross-cell equality is checked by the analyzer; this catches an
    # internally inconsistent result before it can be bound into a matrix.
    assigned_service_by_key: dict[str, float] = {}

    def bind_assigned_service(key: Any, value: Any, path: str) -> None:
        if not _is_sha256(key):
            errors.append(f"{path}: physical service key is not a SHA-256")
            return
        if not _is_number(value) or float(value) < 0.0:
            errors.append(f"{path}: assigned service is not non-negative finite")
            return
        normalized_key = str(key)
        normalized_value = float(value)
        previous = assigned_service_by_key.get(normalized_key)
        if previous is not None and previous != normalized_value:
            errors.append(
                f"{path}: same physical invocation key has non-identical assigned service"
            )
        else:
            assigned_service_by_key[normalized_key] = normalized_value

    raw_llm = payload.get("llm_events")
    raw_predictions = payload.get("prediction_decisions")
    raw_tools = payload.get("tool_events")
    raw_speculation = payload.get("speculation_execution_events")
    result_schema = payload.get("schema")
    if result_schema not in STRICT_RESULT_SCHEMAS:
        errors.append(
            "$.schema: causal replay requires a registered strict result schema"
        )
    qwen_result = result_schema == QWEN_RESULT_SCHEMA
    gemini_result = result_schema == GEMINI_RESULT_SCHEMA
    if not isinstance(raw_llm, list) or not raw_llm:
        errors.append("$.llm_events: causal replay requires raw LLM completion events")
        raw_llm = []
    if not isinstance(raw_predictions, list):
        errors.append("$.prediction_decisions: causal replay requires a decision list")
        raw_predictions = []
    if not isinstance(raw_tools, list) or not raw_tools:
        errors.append("$.tool_events: causal replay requires authoritative reveal events")
        raw_tools = []
    if not isinstance(raw_speculation, list):
        errors.append(
            "$.speculation_execution_events: causal replay requires a physical execution list"
        )
        raw_speculation = []

    llm_by_identity: dict[tuple[str, int], tuple[int, float]] = {}
    for index, row_raw in enumerate(raw_llm):
        path = f"$.llm_events[{index}]"
        if not isinstance(row_raw, Mapping):
            errors.append(f"{path}: expected object")
            continue
        identity = _event_identity(row_raw, path, errors)
        completed = _event_seq_time(
            row_raw,
            seq_field="llm_completed_seq",
            time_field="llm_completed_at_monotonic_s",
            path=path,
            errors=errors,
        )
        if identity is not None and completed is not None:
            if identity in llm_by_identity:
                errors.append(f"{path}: duplicate trace/request LLM completion")
            else:
                llm_by_identity[identity] = completed

    prediction_by_id: dict[str, dict[str, Any]] = {}
    for index, row_raw in enumerate(raw_predictions):
        path = f"$.prediction_decisions[{index}]"
        if not isinstance(row_raw, Mapping):
            errors.append(f"{path}: expected object")
            continue
        identity = _event_identity(row_raw, path, errors)
        prediction_id = row_raw.get("prediction_id")
        if not isinstance(prediction_id, str) or not prediction_id:
            errors.append(f"{path}.prediction_id: missing")
        elif prediction_id in prediction_by_id:
            errors.append(f"{path}.prediction_id: duplicate")
        decision = _event_seq_time(
            row_raw,
            seq_field="decision_seq",
            time_field="decided_at_monotonic_s",
            path=path,
            errors=errors,
        )
        completed = llm_by_identity.get(identity) if identity is not None else None
        if identity is not None and completed is None:
            errors.append(f"{path}: no matching LLM completion event")
        if decision is not None and completed is not None:
            if not decision[0] < completed[0] or not decision[1] <= completed[1]:
                errors.append(f"{path}: prediction was not sealed before LLM completion")
        candidates = row_raw.get("candidates")
        candidate_digests: set[str] = set()
        # ``admitted`` is retained only as a compatibility spelling for a
        # broker-accepted/queued candidate.  It is not evidence that the job
        # occupied a worker.  Physical admission is reconstructed below from
        # the execution ledger's non-null start clocks.
        broker_accepted_digests: set[str] = set()
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"{path}.candidates: expected non-empty candidate list")
        else:
            for candidate_index, candidate in enumerate(candidates):
                candidate_path = f"{path}.candidates[{candidate_index}]"
                if not isinstance(candidate, Mapping):
                    errors.append(f"{candidate_path}: expected object")
                    continue
                digest = candidate.get("candidate_invocation_digest")
                if not _is_sha256(digest) or digest in candidate_digests:
                    errors.append(f"{candidate_path}.candidate_invocation_digest: invalid/duplicate")
                    continue
                candidate_digests.add(str(digest))
                admitted = candidate.get("admitted")
                broker_accepted = candidate.get("broker_accepted")
                if admitted not in {True, False}:
                    errors.append(f"{candidate_path}.admitted: must be boolean")
                if broker_accepted not in {True, False}:
                    errors.append(
                        f"{candidate_path}.broker_accepted: must be boolean"
                    )
                elif admitted in {True, False} and broker_accepted is not admitted:
                    errors.append(
                        f"{candidate_path}: admitted compatibility alias differs "
                        "from broker_accepted"
                    )
                if broker_accepted is True:
                    broker_accepted_digests.add(str(digest))
        if isinstance(prediction_id, str) and prediction_id and identity is not None and decision is not None:
            prediction_by_id[prediction_id] = {
                "identity": identity,
                "decision": decision,
                "llm_completed": completed,
                "candidate_digests": candidate_digests,
                "broker_accepted_digests": broker_accepted_digests,
                "path": path,
            }
        start_seq = row_raw.get("speculative_start_seq")
        start_time = row_raw.get("speculative_start_at_monotonic_s")
        if (start_seq is None) != (start_time is None):
            errors.append(f"{path}: speculative start seq/time must both be null or present")
        elif start_seq is not None:
            started = _event_seq_time(
                row_raw,
                seq_field="speculative_start_seq",
                time_field="speculative_start_at_monotonic_s",
                path=path,
                errors=errors,
            )
            if decision is not None and started is not None and (
                decision[0] > started[0] or decision[1] > started[1]
            ):
                errors.append(f"{path}: physical speculation predates its decision")
            if started is not None and completed is not None and (
                started[0] >= completed[0] or started[1] >= completed[1]
            ):
                errors.append(f"{path}: speculative start is not before LLM completion")

    execution_by_prediction: dict[str, set[str]] = defaultdict(set)
    physical_started_by_prediction: dict[str, set[str]] = defaultdict(set)
    execution_job_ids: set[str] = set()
    observed_physical_starts = 0
    observed_speculative_resource_s = 0.0
    observed_promoted_demand_resource_s = 0.0
    claim_evidence: list[tuple[str, tuple[str, int] | None, float]] = []
    for index, row_raw in enumerate(raw_speculation):
        path = f"$.speculation_execution_events[{index}]"
        if not isinstance(row_raw, Mapping):
            errors.append(f"{path}: expected object")
            continue
        identity = _event_identity(row_raw, path, errors)
        prediction_id = row_raw.get("prediction_id")
        prediction = prediction_by_id.get(str(prediction_id))
        if prediction is None:
            errors.append(f"{path}.prediction_id: no unique matching decision")
        elif identity != prediction["identity"]:
            errors.append(f"{path}: trace/request differs from prediction decision")
        digest = row_raw.get("candidate_invocation_digest")
        if not _is_sha256(digest):
            errors.append(f"{path}.candidate_invocation_digest: invalid")
        elif prediction is not None and digest not in prediction["candidate_digests"]:
            errors.append(f"{path}: execution candidate was not sealed in its decision")
        elif prediction is not None and digest not in prediction[
            "broker_accepted_digests"
        ]:
            errors.append(
                f"{path}: execution candidate was not broker-accepted by its decision"
            )
        elif isinstance(prediction_id, str):
            if digest in execution_by_prediction[prediction_id]:
                errors.append(f"{path}: duplicate execution row for candidate")
            execution_by_prediction[prediction_id].add(str(digest))
        bind_assigned_service(
            digest,
            row_raw.get("assigned_service_s"),
            f"{path}.assigned_service_s",
        )
        admitted_at = row_raw.get("admitted_at_monotonic_s")
        if not _is_number(admitted_at) or float(admitted_at) < 0.0:
            errors.append(f"{path}.admitted_at_monotonic_s: invalid")
            admitted_at_value = None
        else:
            admitted_at_value = float(admitted_at)
        if prediction is not None and admitted_at_value is not None:
            if admitted_at_value < prediction["decision"][1]:
                errors.append(f"{path}: admission predates prediction decision")
            llm_completed = prediction["llm_completed"]
            if llm_completed is None or admitted_at_value >= llm_completed[1]:
                errors.append(
                    f"{path}: broker admission did not precede LLM completion"
                )
        started_at = row_raw.get("physical_started_at_monotonic_s")
        terminal_at = row_raw.get("terminal_at_monotonic_s")
        service_s = row_raw.get("service_s")
        total_service_s = row_raw.get("total_worker_service_s")
        speculative_resource_s = row_raw.get("speculative_resource_s")
        demand_resource_s = row_raw.get("demand_resource_s")
        claimed_at = row_raw.get("authority_claimed_at_monotonic_s")
        claimed = row_raw.get("claimed_by_authority")
        transition_boundaries = (
            _audit_strict_speculation_transition_ledger(
                row_raw, path=path, errors=errors
            )
            if result_schema in STRICT_RESULT_SCHEMAS
            else None
        )
        job_id = row_raw.get("job_id")
        valid_job_id = (
            isinstance(job_id, str) and bool(job_id)
        ) or (type(job_id) is int and job_id >= 0)
        canonical_job_id = f"{type(job_id).__name__}:{job_id}"
        if not valid_job_id or canonical_job_id in execution_job_ids:
            errors.append(f"{path}.job_id: missing or reused execution job identity")
        else:
            execution_job_ids.add(canonical_job_id)
        if not _is_number(terminal_at) or (
            admitted_at_value is not None and float(terminal_at) < admitted_at_value
        ):
            errors.append(f"{path}.terminal_at_monotonic_s: invalid")
        resource_values: dict[str, float] = {}
        for field, value in (
            ("service_s", service_s),
            ("total_worker_service_s", total_service_s),
            ("speculative_resource_s", speculative_resource_s),
            ("demand_resource_s", demand_resource_s),
        ):
            if not _is_number(value) or float(value) < 0.0:
                errors.append(f"{path}.{field}: expected non-negative finite number")
            else:
                resource_values[field] = float(value)
        if set(resource_values) == {
            "service_s",
            "total_worker_service_s",
            "speculative_resource_s",
            "demand_resource_s",
        }:
            if not _numbers_close(
                resource_values["total_worker_service_s"],
                resource_values["speculative_resource_s"]
                + resource_values["demand_resource_s"],
            ):
                errors.append(f"{path}: speculative+demand service does not equal total")
            if not _numbers_close(
                resource_values["service_s"],
                resource_values["total_worker_service_s"],
            ):
                errors.append(f"{path}: service_s alias does not equal total worker service")
            observed_speculative_resource_s += resource_values[
                "speculative_resource_s"
            ]
            observed_promoted_demand_resource_s += resource_values[
                "demand_resource_s"
            ]

        if claimed not in {True, False}:
            errors.append(f"{path}.claimed_by_authority: must be boolean")
        if claimed_at is None:
            if claimed is True:
                errors.append(f"{path}: claimed work lacks an authority-claim timestamp")
            claimed_value = None
        elif not _is_number(claimed_at) or float(claimed_at) < 0.0:
            errors.append(f"{path}.authority_claimed_at_monotonic_s: invalid")
            claimed_value = None
        else:
            claimed_value = float(claimed_at)
            if claimed is not True:
                errors.append(f"{path}: authority-claim timestamp requires claimed=true")

        effective_admitted_at = admitted_at_value
        effective_started_at = started_at
        effective_terminal_at = terminal_at
        effective_claimed_at = claimed_value
        if transition_boundaries is not None:
            effective_admitted_at = transition_boundaries[
                "admitted_at_monotonic_s"
            ]
            effective_started_at = transition_boundaries[
                "physical_started_at_monotonic_s"
            ]
            effective_terminal_at = transition_boundaries[
                "terminal_at_monotonic_s"
            ]
            effective_claimed_at = transition_boundaries[
                "authority_claimed_at_monotonic_s"
            ]
        if effective_claimed_at is not None:
            claim_evidence.append((path, identity, float(effective_claimed_at)))

        if effective_started_at is None:
            if any(
                not _numbers_close(resource_values.get(field, 0.0), 0.0)
                for field in (
                    "service_s",
                    "total_worker_service_s",
                    "speculative_resource_s",
                    "demand_resource_s",
                )
            ):
                errors.append(f"{path}: never-started work must have zero service")
        else:
            observed_physical_starts += 1
            if not _is_number(effective_started_at) or float(
                effective_started_at
            ) < 0.0:
                errors.append(f"{path}.physical_started_at_monotonic_s: invalid")
                started_value = None
            else:
                started_value = float(effective_started_at)
                if isinstance(prediction_id, str) and _is_sha256(digest):
                    physical_started_by_prediction[prediction_id].add(str(digest))
            if (
                effective_admitted_at is not None
                and started_value is not None
                and float(effective_admitted_at) > started_value
            ):
                errors.append(f"{path}: physical start predates broker admission")
            if _is_number(effective_terminal_at) and (
                started_value is not None
                and float(effective_terminal_at) < started_value
            ):
                errors.append(f"{path}.terminal_at_monotonic_s: invalid")
            if prediction is not None and started_value is not None:
                if started_value < prediction["decision"][1]:
                    errors.append(f"{path}: physical start predates prediction decision")
                llm_completed = prediction["llm_completed"]
                if llm_completed is None or started_value >= llm_completed[1]:
                    errors.append(f"{path}: physical speculation did not start before LLM completion")
            if (
                started_value is not None
                and _is_number(effective_terminal_at)
                and "total_worker_service_s" in resource_values
            ):
                terminal_value = float(effective_terminal_at)
                elapsed = terminal_value - started_value
                if not _numbers_close(
                    resource_values["total_worker_service_s"], elapsed
                ):
                    errors.append(
                        f"{path}: total worker service does not equal physical start-to-terminal occupancy"
                    )
                if effective_claimed_at is None:
                    expected_speculative = elapsed
                    expected_demand = 0.0
                elif float(effective_claimed_at) <= started_value:
                    expected_speculative = 0.0
                    expected_demand = elapsed
                elif float(effective_claimed_at) >= terminal_value:
                    expected_speculative = elapsed
                    expected_demand = 0.0
                else:
                    expected_speculative = float(effective_claimed_at) - started_value
                    expected_demand = terminal_value - float(effective_claimed_at)
                if "speculative_resource_s" in resource_values and not _numbers_close(
                    resource_values["speculative_resource_s"], expected_speculative
                ):
                    errors.append(
                        f"{path}: speculative resource is not physical start-to-claim"
                    )
                if "demand_resource_s" in resource_values and not _numbers_close(
                    resource_values["demand_resource_s"], expected_demand
                ):
                    errors.append(
                        f"{path}: promoted demand resource is not claim-to-terminal"
                    )
        if not isinstance(row_raw.get("terminal_state"), str) or not row_raw.get(
            "terminal_state"
        ):
            errors.append(f"{path}.terminal_state: missing")

    for prediction_id, prediction in prediction_by_id.items():
        missing = prediction["broker_accepted_digests"] - execution_by_prediction.get(
            prediction_id, set()
        )
        if missing:
            errors.append(
                f"{prediction['path']}: broker-accepted candidate(s) lack execution "
                "accounting: "
                f"{sorted(missing)}"
            )
        if gemini_result and prediction["broker_accepted_digests"] != (
            physical_started_by_prediction.get(prediction_id, set())
        ):
            errors.append(
                f"{prediction['path']}: Gemini immediate-start broker acceptance "
                "differs from raw physical starts"
            )

    paper = payload.get("paper_protocol")
    declared_starts = paper.get("physical_speculative_starts") if isinstance(paper, Mapping) else None
    if declared_starts != observed_physical_starts:
        errors.append(
            "$.paper_protocol.physical_speculative_starts: does not equal raw physical start events"
        )
    broker_counts = []
    for path, node in _walk(payload):
        if (
            isinstance(node, Mapping)
            and path != "$.paper_protocol"
            and type(node.get("physical_speculative_starts")) is int
        ):
            broker_counts.append((path, int(node["physical_speculative_starts"])))
    if not broker_counts:
        errors.append("$: missing broker physical_speculative_starts counter")
    for path, count in broker_counts:
        if count != observed_physical_starts:
            errors.append(f"{path}.physical_speculative_starts: disagrees with raw start events")

    tool_outcome_nodes: set[int] = set()
    seen_outcomes: set[tuple[str, int, str]] = set()
    reveal_times_by_identity: dict[tuple[str, int], list[float]] = defaultdict(list)
    tool_completion_times_by_identity: dict[
        tuple[str, int], list[float]
    ] = defaultdict(list)
    authority_invocations_by_identity: dict[
        tuple[str, int], set[str]
    ] = defaultdict(set)
    authority_candidates_by_identity: dict[
        tuple[str, int], set[str]
    ] = defaultdict(set)
    pool_authority_keys_by_identity: dict[
        tuple[str, int], set[str]
    ] = defaultdict(set)
    observed_direct_demand_resource_s = 0.0
    direct_demand_evidence_complete = True
    observed_duration_errors_s: list[float] = []
    for index, row_raw in enumerate(raw_tools):
        path = f"$.tool_events[{index}]"
        if not isinstance(row_raw, Mapping):
            errors.append(f"{path}: expected object")
            continue
        tool_outcome_nodes.add(id(row_raw))
        identity = _event_identity(row_raw, path, errors)
        copied_llm = _event_seq_time(
            row_raw,
            seq_field="llm_completed_seq",
            time_field="llm_completed_at_monotonic_s",
            path=path,
            errors=errors,
        )
        reveal = _event_seq_time(
            row_raw,
            seq_field="authoritative_revealed_seq",
            time_field="authoritative_revealed_at_monotonic_s",
            path=path,
            errors=errors,
        )
        completed_tool = _event_seq_time(
            row_raw,
            seq_field="tool_completed_seq",
            time_field="tool_completed_at_monotonic_s",
            path=path,
            errors=errors,
        )
        expected_llm = llm_by_identity.get(identity) if identity is not None else None
        if identity is not None and expected_llm is None:
            errors.append(f"{path}: no matching LLM completion event")
        elif copied_llm is not None and expected_llm != copied_llm:
            errors.append(f"{path}: copied LLM completion does not match raw LLM event")
        if copied_llm is not None and reveal is not None and (
            copied_llm[0] > reveal[0] or copied_llm[1] > reveal[1]
        ):
            errors.append(f"{path}: authority was revealed before live LLM completion")
        if reveal is not None and completed_tool is not None and (
            reveal[0] >= completed_tool[0] or reveal[1] > completed_tool[1]
        ):
            errors.append(f"{path}: tool completion does not follow authority reveal")
        if identity is not None and reveal is not None:
            reveal_times_by_identity[identity].append(reveal[1])
        if identity is not None and completed_tool is not None:
            tool_completion_times_by_identity[identity].append(completed_tool[1])
        invocation_digest = row_raw.get("authority_invocation_digest")
        if qwen_result and not _is_sha256(invocation_digest):
            errors.append(
                f"{path}.authority_invocation_digest: "
                "required for Qwen raw authority reconstruction"
            )
        if invocation_digest is not None:
            if not _is_sha256(invocation_digest):
                errors.append(f"{path}.authority_invocation_digest: invalid")
            elif identity is not None:
                authority_invocations_by_identity[identity].add(
                    str(invocation_digest)
                )
        authority_candidates_raw = row_raw.get(
            "authority_candidate_invocation_digests"
        )
        if qwen_result and authority_candidates_raw is None:
            errors.append(
                f"{path}.authority_candidate_invocation_digests: "
                "required for Qwen raw precision reconstruction"
            )
        if authority_candidates_raw is not None:
            if (
                not isinstance(authority_candidates_raw, list)
                or any(not _is_sha256(item) for item in authority_candidates_raw)
                or len(authority_candidates_raw)
                != len(set(authority_candidates_raw))
            ):
                errors.append(
                    f"{path}.authority_candidate_invocation_digests: "
                    "unique SHA-256 list required"
                )
            elif identity is not None:
                authority_candidates_by_identity[identity].update(
                    str(item) for item in authority_candidates_raw
                )
        pool_authority_digest = row_raw.get("pool_authority_key_sha256")
        authority_key_digest = row_raw.get("authority_key_sha256")
        if gemini_result and not _is_sha256(authority_key_digest):
            errors.append(
                f"{path}.authority_key_sha256: "
                "required for Gemini raw authority reconstruction"
            )
        if gemini_result and not _is_sha256(pool_authority_digest):
            errors.append(
                f"{path}.pool_authority_key_sha256: required for Gemini raw precision reconstruction"
            )
        if pool_authority_digest is not None:
            if not _is_sha256(pool_authority_digest):
                errors.append(f"{path}.pool_authority_key_sha256: invalid")
            elif identity is not None:
                pool_authority_keys_by_identity[identity].add(
                    str(pool_authority_digest)
                )
        physical_service_digest = row_raw.get("physical_service_key_sha256")
        if gemini_result and not _is_sha256(physical_service_digest):
            errors.append(
                f"{path}.physical_service_key_sha256: "
                "required for Gemini policy-independent service reconstruction"
            )
        if physical_service_digest is None:
            physical_service_digest = invocation_digest
        assigned_service = row_raw.get("execution_surface_service_s")
        if assigned_service is None:
            assigned_service = row_raw.get("assigned_service_s")
        bind_assigned_service(
            physical_service_digest,
            assigned_service,
            f"{path}.assigned_service_s",
        )

        predicted_service = row_raw.get("authority_eta_hat_s")
        if predicted_service is None:
            predicted_service = row_raw.get("tool_service_s_hat")
        declared_absolute_error = row_raw.get(
            "duration_prediction_absolute_error_s"
        )
        if declared_absolute_error is not None:
            if (
                not _is_number(predicted_service)
                or float(predicted_service) < 0.0
                or not _is_number(assigned_service)
                or float(assigned_service) < 0.0
                or not _is_number(declared_absolute_error)
                or float(declared_absolute_error) < 0.0
            ):
                errors.append(
                    f"{path}.duration_prediction_absolute_error_s: "
                    "requires non-negative raw *_hat and assigned-service values"
                )
            elif not _numbers_close(
                float(declared_absolute_error),
                abs(float(predicted_service) - float(assigned_service)),
            ):
                errors.append(
                    f"{path}.duration_prediction_absolute_error_s: "
                    "differs from raw prediction/assigned-service pair"
                )
            else:
                observed_duration_errors_s.append(
                    abs(float(predicted_service) - float(assigned_service))
                )
        elif _is_number(predicted_service) and _is_number(assigned_service):
            observed_duration_errors_s.append(
                abs(float(predicted_service) - float(assigned_service))
            )
        outcome_id = row_raw.get("outcome_id")
        if not isinstance(outcome_id, str) or not outcome_id:
            errors.append(f"{path}.outcome_id: missing")
        elif identity is not None:
            outcome_identity = (*identity, outcome_id)
            if outcome_identity in seen_outcomes:
                errors.append(f"{path}.outcome_id: duplicate within trace/request")
            seen_outcomes.add(outcome_identity)

        # Gemini emits one normalized service value.  Qwen emits a direct
        # non-visit service or atomic visit rows so promoted/reused work is not
        # double-counted as direct demand.
        if "worker_service_s" in row_raw or "cache_source" in row_raw:
            worker_service = row_raw.get("worker_service_s")
            if row_raw.get("cache_source") == "executed":
                if not _is_number(worker_service) or float(worker_service) < 0.0:
                    errors.append(f"{path}.worker_service_s: invalid direct-work evidence")
                    direct_demand_evidence_complete = False
                else:
                    observed_direct_demand_resource_s += float(worker_service)
            elif worker_service is not None and (
                not _is_number(worker_service) or float(worker_service) < 0.0
            ):
                errors.append(f"{path}.worker_service_s: invalid service evidence")
                direct_demand_evidence_complete = False
        elif isinstance(row_raw.get("tool_name"), str):
            if row_raw.get("tool_name") == "visit":
                visit_rows = row_raw.get("visit_results")
                if not isinstance(visit_rows, (list, tuple)):
                    errors.append(f"{path}.visit_results: missing direct-work evidence")
                    direct_demand_evidence_complete = False
                else:
                    for visit_index, visit_raw in enumerate(visit_rows):
                        if not isinstance(visit_raw, Mapping) or not _is_number(
                            visit_raw.get("service_s")
                        ) or float(visit_raw["service_s"]) < 0.0:
                            errors.append(
                                f"{path}.visit_results[{visit_index}]: invalid service evidence"
                            )
                            direct_demand_evidence_complete = False
                        elif visit_raw.get("source") == "executed":
                            observed_direct_demand_resource_s += float(
                                visit_raw["service_s"]
                            )
            else:
                service = row_raw.get("service_s")
                if not _is_number(service) or float(service) < 0.0:
                    errors.append(f"{path}.service_s: missing direct-work evidence")
                    direct_demand_evidence_complete = False
                else:
                    observed_direct_demand_resource_s += float(service)
        else:
            errors.append(f"{path}: missing normalized direct-demand service evidence")
            direct_demand_evidence_complete = False

    for path, identity, claimed_at in claim_evidence:
        reveals = reveal_times_by_identity.get(identity, []) if identity is not None else []
        completions = (
            tool_completion_times_by_identity.get(identity, [])
            if identity is not None
            else []
        )
        if not reveals:
            errors.append(f"{path}: authority claim has no matching reveal event")
        elif claimed_at < min(reveals) and not _numbers_close(claimed_at, min(reveals)):
            errors.append(f"{path}: authority claim predates authoritative reveal")
        if not completions:
            errors.append(f"{path}: authority claim has no matching tool completion")
        elif claimed_at > max(completions) and not _numbers_close(
            claimed_at, max(completions)
        ):
            errors.append(
                f"{path}: first authority claim follows its originating tool completion"
            )

    # Post-reveal match labels are redundant checksums, not trusted precision
    # inputs.  Recompute them from sealed candidate/authority digests and bind
    # every resolution back to exactly one immutable decision.
    raw_outcomes = payload.get("prediction_outcomes", [])
    if not isinstance(raw_outcomes, list):
        errors.append("$.prediction_outcomes: causal replay requires a resolution list")
        raw_outcomes = []
    resolved_prediction_ids: set[str] = set()
    matched_emitted = 0
    matched_broker_accepted = 0
    matched_physical_started = 0
    matched_decisions = 0
    for index, row_raw in enumerate(raw_outcomes):
        path = f"$.prediction_outcomes[{index}]"
        if not isinstance(row_raw, Mapping):
            errors.append(f"{path}: expected object")
            continue
        prediction_id = row_raw.get("prediction_id")
        prediction = prediction_by_id.get(str(prediction_id))
        if (
            not isinstance(prediction_id, str)
            or not prediction_id
            or prediction is None
            or prediction_id in resolved_prediction_ids
        ):
            errors.append(f"{path}.prediction_id: no unique matching decision")
            continue
        resolved_prediction_ids.add(prediction_id)
        identity = _event_identity(row_raw, path, errors)
        if identity != prediction["identity"]:
            errors.append(f"{path}: trace/request differs from prediction decision")
        nested = row_raw.get("candidates")
        if isinstance(nested, list):
            if row_raw.get("admitted_semantics") != (
                "broker_accepted_not_physical_start"
            ):
                errors.append(
                    f"{path}.admitted_semantics: invalid broker-acceptance declaration"
                )
            authority_raw = row_raw.get(
                "authoritative_candidate_invocation_digests"
            )
            if (
                not isinstance(authority_raw, list)
                or any(not _is_sha256(item) for item in authority_raw)
                or len(authority_raw) != len(set(authority_raw))
            ):
                errors.append(
                    f"{path}.authoritative_candidate_invocation_digests: "
                    "unique SHA-256 list required"
                )
                authority_candidates: set[str] = set()
            else:
                authority_candidates = {str(item) for item in authority_raw}
            if (
                identity is not None
                and authority_candidates
                != authority_candidates_by_identity.get(identity, set())
            ):
                errors.append(
                    f"{path}.authoritative_candidate_invocation_digests: "
                    "differs from raw authoritative tool-event candidates"
                )
            full_authority_raw = row_raw.get("authoritative_invocation_digests")
            if (
                not isinstance(full_authority_raw, list)
                or any(not _is_sha256(item) for item in full_authority_raw)
                or identity is None
                or set(str(item) for item in full_authority_raw)
                != authority_invocations_by_identity.get(identity, set())
            ):
                errors.append(
                    f"{path}.authoritative_invocation_digests: "
                    "differs from raw authoritative tool events"
                )
            seen_candidate_digests: set[str] = set()
            for candidate_index, candidate in enumerate(nested):
                candidate_path = f"{path}.candidates[{candidate_index}]"
                if not isinstance(candidate, Mapping):
                    errors.append(f"{candidate_path}: expected object")
                    continue
                digest = candidate.get("candidate_invocation_digest")
                if (
                    not _is_sha256(digest)
                    or str(digest) in seen_candidate_digests
                    or str(digest) not in prediction["candidate_digests"]
                ):
                    errors.append(
                        f"{candidate_path}.candidate_invocation_digest: "
                        "invalid, duplicate, or not in decision"
                    )
                    continue
                digest = str(digest)
                seen_candidate_digests.add(digest)
                expected_broker_accepted = digest in prediction[
                    "broker_accepted_digests"
                ]
                if candidate.get("admitted") is not expected_broker_accepted:
                    errors.append(
                        f"{candidate_path}.admitted: broker-acceptance alias "
                        "differs from decision"
                    )
                if candidate.get("broker_accepted") is not expected_broker_accepted:
                    errors.append(
                        f"{candidate_path}.broker_accepted: differs from decision"
                    )
                expected_physical_started = digest in physical_started_by_prediction.get(
                    prediction_id, set()
                )
                expected_match = digest in authority_candidates
                if candidate.get("matched_authority") is not expected_match:
                    errors.append(
                        f"{candidate_path}.matched_authority: differs from raw digest match"
                    )
                matched_emitted += int(expected_match)
                matched_broker_accepted += int(
                    expected_match and expected_broker_accepted
                )
                matched_physical_started += int(
                    expected_match and expected_physical_started
                )
            if seen_candidate_digests != prediction["candidate_digests"]:
                errors.append(f"{path}.candidates: do not exactly cover decision candidates")
            expected_counts = {
                "emitted_candidate_count": len(seen_candidate_digests),
                "broker_accepted_candidate_count": sum(
                    digest in prediction["broker_accepted_digests"]
                    for digest in seen_candidate_digests
                ),
                "physical_started_candidate_count": sum(
                    digest in physical_started_by_prediction.get(prediction_id, set())
                    for digest in seen_candidate_digests
                ),
                # Resolution-row compatibility fields mirror the immutable
                # candidate label, where admitted means broker acceptance.
                # Paper precision aggregates below use the separate raw
                # physical-start universe.
                "admitted_candidate_count": sum(
                    digest in prediction["broker_accepted_digests"]
                    for digest in seen_candidate_digests
                ),
                "matched_emitted_candidate_count": sum(
                    digest in authority_candidates
                    for digest in seen_candidate_digests
                ),
                "matched_broker_accepted_candidate_count": sum(
                    digest in authority_candidates
                    and digest in prediction["broker_accepted_digests"]
                    for digest in seen_candidate_digests
                ),
                "matched_physical_started_candidate_count": sum(
                    digest in authority_candidates
                    and digest in physical_started_by_prediction.get(prediction_id, set())
                    for digest in seen_candidate_digests
                ),
                "matched_admitted_candidate_count": sum(
                    digest in authority_candidates
                    and digest in prediction["broker_accepted_digests"]
                    for digest in seen_candidate_digests
                ),
                "decision_hit": any(
                    digest in authority_candidates
                    for digest in seen_candidate_digests
                ),
            }
            for field, expected in expected_counts.items():
                if field in row_raw and row_raw.get(field) != expected:
                    errors.append(f"{path}.{field}: differs from raw digest recomputation")
            matched_decisions += int(bool(expected_counts["decision_hit"]))
        else:
            digest = row_raw.get("candidate_invocation_digest")
            if not _is_sha256(digest) or str(digest) not in prediction[
                "candidate_digests"
            ]:
                errors.append(
                    f"{path}.candidate_invocation_digest: differs from decision"
                )
                continue
            digest = str(digest)
            expected_broker_accepted = digest in prediction[
                "broker_accepted_digests"
            ]
            if row_raw.get("admitted") is not expected_broker_accepted:
                errors.append(
                    f"{path}.admitted: broker-acceptance alias differs from decision"
                )
            if row_raw.get("broker_accepted") is not expected_broker_accepted:
                errors.append(f"{path}.broker_accepted: differs from decision")
            expected_physical_started = digest in physical_started_by_prediction.get(
                prediction_id, set()
            )
            expected_hit = (
                identity is not None
                and digest in pool_authority_keys_by_identity.get(identity, set())
            )
            if row_raw.get("post_authority_hit") is not expected_hit:
                errors.append(
                    f"{path}.post_authority_hit: differs from raw pool-key match"
                )
            matched_emitted += int(expected_hit)
            matched_broker_accepted += int(
                expected_hit and expected_broker_accepted
            )
            matched_physical_started += int(
                expected_hit and expected_physical_started
            )
            matched_decisions += int(expected_hit)
    missing_resolutions = set(prediction_by_id) - resolved_prediction_ids
    if missing_resolutions:
        errors.append(
            "$.prediction_outcomes: decision(s) lack post-reveal resolution: "
            f"{sorted(missing_resolutions)}"
        )

    metrics = payload.get("prediction_metrics")
    if isinstance(metrics, Mapping):
        if metrics.get("admitted_metric_semantics") != (
            "physical_started_at_monotonic_s_is_not_null"
        ):
            errors.append(
                "$.prediction_metrics.admitted_metric_semantics: must declare "
                "physical-start-conditioned compatibility semantics"
            )
        emitted_total = sum(
            len(row["candidate_digests"]) for row in prediction_by_id.values()
        )
        broker_accepted_total = sum(
            len(row["broker_accepted_digests"]) for row in prediction_by_id.values()
        )
        physical_started_total = sum(
            len(physical_started_by_prediction.get(prediction_id, set()))
            for prediction_id in prediction_by_id
        )
        for field, expected in (
            ("decisions_with_candidates", len(prediction_by_id)),
            ("decision_hits", matched_decisions),
            ("emitted_candidates", emitted_total),
            ("broker_accepted_candidates", broker_accepted_total),
            ("physical_started_candidates", physical_started_total),
            ("admitted_candidates", physical_started_total),
            ("matched_emitted_candidates", matched_emitted),
            ("matched_broker_accepted_candidates", matched_broker_accepted),
            ("matched_physical_started_candidates", matched_physical_started),
            ("matched_admitted_candidates", matched_physical_started),
            (
                "queued_never_started_candidates",
                broker_accepted_total - physical_started_total,
            ),
        ):
            if field in metrics and metrics.get(field) != expected:
                errors.append(
                    f"$.prediction_metrics.{field}: differs from raw digest recomputation"
                )
        for field, numerator, denominator in (
            ("emitted_candidate_precision", matched_emitted, emitted_total),
            (
                "broker_accepted_candidate_precision",
                matched_broker_accepted,
                broker_accepted_total,
            ),
            (
                "physical_started_candidate_precision",
                matched_physical_started,
                physical_started_total,
            ),
            (
                "admitted_candidate_precision",
                matched_physical_started,
                physical_started_total,
            ),
        ):
            expected = numerator / denominator if denominator else None
            if field in metrics and (
                (expected is None and metrics.get(field) is not None)
                or (
                    expected is not None
                    and (
                        not _is_number(metrics.get(field))
                        or not _numbers_close(float(metrics[field]), expected)
                    )
                )
            ):
                errors.append(
                    f"$.prediction_metrics.{field}: differs from raw digest recomputation"
                )

    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        emitted_total = sum(
            len(row["candidate_digests"]) for row in prediction_by_id.values()
        )
        broker_accepted_total = sum(
            len(row["broker_accepted_digests"]) for row in prediction_by_id.values()
        )
        physical_started_total = sum(
            len(physical_started_by_prediction.get(prediction_id, set()))
            for prediction_id in prediction_by_id
        )
        for field, expected in (
            ("prediction_candidates", emitted_total),
            ("prediction_broker_accepted", broker_accepted_total),
            ("prediction_physical_started", physical_started_total),
            ("prediction_admitted", physical_started_total),
            ("emitted_post_authority_hits", matched_emitted),
            ("broker_accepted_post_authority_hits", matched_broker_accepted),
            ("physical_started_post_authority_hits", matched_physical_started),
            ("post_authority_hits", matched_physical_started),
        ):
            if field in summary and summary.get(field) != expected:
                errors.append(f"$.summary.{field}: differs from raw digest recomputation")
        if "assigned_clock_duration_mae_s" in summary:
            expected_mae = (
                statistics.fmean(observed_duration_errors_s)
                if observed_duration_errors_s
                else None
            )
            observed_mae = summary.get("assigned_clock_duration_mae_s")
            if (
                (expected_mae is None and observed_mae is not None)
                or (
                    expected_mae is not None
                    and (
                        not _is_number(observed_mae)
                        or not _numbers_close(float(observed_mae), expected_mae)
                    )
                )
            ):
                errors.append(
                    "$.summary.assigned_clock_duration_mae_s: differs from raw "
                    "prediction/assigned-service pairs"
                )
    duration_metrics = payload.get("duration_prediction_metrics")
    if isinstance(duration_metrics, Mapping):
        if duration_metrics.get("authoritative_tool_calls") != len(
            observed_duration_errors_s
        ):
            errors.append(
                "$.duration_prediction_metrics.authoritative_tool_calls: "
                "differs from raw duration evidence"
            )
        expected_mae = (
            statistics.fmean(observed_duration_errors_s)
            if observed_duration_errors_s
            else None
        )
        observed_mae = duration_metrics.get("mean_absolute_error_s")
        if (
            (expected_mae is None and observed_mae is not None)
            or (
                expected_mae is not None
                and (
                    not _is_number(observed_mae)
                    or not _numbers_close(float(observed_mae), expected_mae)
                )
            )
        ):
            errors.append(
                "$.duration_prediction_metrics.mean_absolute_error_s: "
                "differs from raw prediction/assigned-service pairs"
            )

    accounting = payload.get("worker_resource_accounting")
    accounting_values: dict[str, float] = {}
    accounting_fields = (
        "speculative_resource_s",
        "promoted_demand_resource_s",
        "direct_demand_resource_s",
        "total_worker_occupancy_s",
    )
    if not isinstance(accounting, Mapping):
        errors.append("$.worker_resource_accounting: missing normalized worker accounting")
    else:
        for field in accounting_fields:
            value = accounting.get(field)
            if not _is_number(value) or float(value) < 0.0:
                errors.append(
                    f"$.worker_resource_accounting.{field}: expected non-negative finite number"
                )
            else:
                accounting_values[field] = float(value)
        if len(accounting_values) == len(accounting_fields):
            parts = (
                accounting_values["speculative_resource_s"]
                + accounting_values["promoted_demand_resource_s"]
                + accounting_values["direct_demand_resource_s"]
            )
            if not _numbers_close(accounting_values["total_worker_occupancy_s"], parts):
                errors.append(
                    "$.worker_resource_accounting: component services do not conserve total occupancy"
                )
            if not _numbers_close(
                accounting_values["speculative_resource_s"],
                observed_speculative_resource_s,
            ):
                errors.append(
                    "$.worker_resource_accounting.speculative_resource_s: "
                    "does not equal execution ledger"
                )
            if not _numbers_close(
                accounting_values["promoted_demand_resource_s"],
                observed_promoted_demand_resource_s,
            ):
                errors.append(
                    "$.worker_resource_accounting.promoted_demand_resource_s: "
                    "does not equal execution ledger"
                )
            if direct_demand_evidence_complete and not _numbers_close(
                accounting_values["direct_demand_resource_s"],
                observed_direct_demand_resource_s,
            ):
                errors.append(
                    "$.worker_resource_accounting.direct_demand_resource_s: "
                    "does not equal raw executed authoritative tool work"
                )
    broker_accounting = []
    for path, node in _walk(payload):
        if (
            isinstance(node, Mapping)
            and path != "$.worker_resource_accounting"
            and all(field in node for field in accounting_fields)
        ):
            broker_accounting.append((path, node))
    if not broker_accounting:
        errors.append("$: broker snapshot lacks complete worker resource accounting")
    elif len(accounting_values) == len(accounting_fields):
        for path, node in broker_accounting:
            for field in accounting_fields:
                if not _is_number(node.get(field)) or not _numbers_close(
                    float(node[field]), accounting_values[field]
                ):
                    errors.append(
                        f"{path}.{field}: disagrees with normalized worker accounting"
                    )

    for path, node in _walk(payload):
        if isinstance(node, Mapping) and "outcome_id" in node and id(node) not in tool_outcome_nodes:
            errors.append(f"{path}.outcome_id: private outcome identity appeared before/tool-outside reveal")
    return errors


def audit_result_payload(
    payload: Any,
    *,
    gemini_legacy_compatibility: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return leakage/config errors from one result JSON object."""

    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["$: result root must be an object"]
    compatibility_present = payload.get("legacy_frozen_compatibility") is not None
    _audit_runtime_provenance_shape(
        payload.get("provenance"),
        label="$.provenance",
        errors=errors,
        extra_fields=(
            GEMINI_LEGACY_COMPATIBILITY_PROVENANCE_FIELDS
            if compatibility_present or gemini_legacy_compatibility is not None
            else ()
        ),
    )
    _audit_runtime_parameters_contract(
        payload.get("runtime_parameters"),
        label="$.runtime_parameters",
        require_artifact=False,
        errors=errors,
    )
    errors.extend(_audit_task_timing_evidence(payload))
    metadata_count = 0
    for path, meta in _metadata_objects(payload):
        metadata_count += 1
        if meta.get("ms") != DECISION_SCHEMA:
            errors.append(f"{path}: scheduler metadata schema is not {DECISION_SCHEMA}")
        legacy = sorted(LEGACY_ORACLE_METADATA_FIELDS & set(meta))
        if legacy:
            errors.append(f"{path}: legacy future-trace fields present: {legacy}")
        unknown = sorted(set(meta) - ALLOWED_SCHEDULER_METADATA_FIELDS)
        if unknown:
            errors.append(f"{path}: unregistered scheduler fields present: {unknown}")
        for field in (
            "decision_seq",
            "observed_event_seq",
            "policy_sha256",
        ):
            if field not in meta:
                errors.append(f"{path}: missing required field {field}")
        decision_seq = meta.get("decision_seq")
        observed_seq = meta.get("observed_event_seq")
        if (
            type(decision_seq) is not int
            or type(observed_seq) is not int
            or observed_seq < 0
            or decision_seq < observed_seq
        ):
            errors.append(f"{path}: invalid causal event/decision sequence")
        if not _is_sha256(meta.get("policy_sha256")):
            errors.append(f"{path}: policy_sha256 is not a SHA-256")
        predicted_fields = {
            key for key in meta if key.endswith("_hat")
        }
        if predicted_fields and not _is_sha256(
            meta.get("predictor_artifact_sha256")
        ):
            errors.append(
                f"{path}: predicted fields lack predictor_artifact_sha256"
            )
        if any(
            key in meta
            for key in (
                "tool_service_s_hat",
                "tool_eta_s_hat",
                "remaining_tool_wait_s_hat",
                "nper_hat",
            )
        ) and not _is_sha256(meta.get("duration_predictor_artifact_sha256")):
            errors.append(
                f"{path}: duration prediction lacks duration artifact binding"
            )

    for path, decision in _prediction_decisions(payload):
        for child_path, child in _walk(decision, path):
            if not isinstance(child, Mapping):
                continue
            leaked = sorted(FORBIDDEN_PREDICTION_INPUT_KEYS & set(child))
            if leaked:
                errors.append(
                    f"{child_path}: prediction decision contains outcomes: {leaked}"
                )
            unmarked = sorted(LEGACY_UNMARKED_PREDICTION_KEYS & set(child))
            if unmarked:
                errors.append(
                    f"{child_path}: predicted quantities require explicit *_hat names: "
                    f"{unmarked}"
                )
        for field in (
            "decision_seq",
            "observed_event_seq",
            "decided_at_monotonic_s",
            "candidate_invocation_digest",
            "predictor_artifact_sha256",
        ):
            if field not in decision:
                errors.append(f"{path}: missing prediction evidence {field}")
        decision_seq = decision.get("decision_seq")
        observed_seq = decision.get("observed_event_seq")
        if (
            type(decision_seq) is not int
            or type(observed_seq) is not int
            or observed_seq < 0
            or decision_seq < observed_seq
        ):
            errors.append(f"{path}: prediction observes a future event")
        if not _is_sha256(decision.get("candidate_invocation_digest")):
            errors.append(f"{path}: candidate invocation is not hash-bound")
        if not _is_sha256(decision.get("predictor_artifact_sha256")):
            errors.append(f"{path}: predictor artifact is not hash-bound")

    if isinstance(payload, Mapping):
        for path, node in _walk(payload):
            if not isinstance(node, Mapping):
                continue
            call_graph = node.get("call_graph_mode")
            if call_graph is not None and call_graph not in CALL_GRAPH_MODES:
                errors.append(
                    f"{path}.call_graph_mode: expected autonomous or "
                    "trace_replay_causal_reveal"
                )
            for key in ("scheduler", "tool_mechanism", "tool_replay_mode"):
                raw = node.get(key)
                if not isinstance(raw, str):
                    continue
                lowered = raw.lower()
                if any(
                    term in lowered
                    for term in ("oracle", "offline_pattern", "projection")
                ):
                    errors.append(
                        f"{path}.{key}: forbidden retrospective mechanism {raw!r}"
                    )

        settings = payload.get("settings")
        if isinstance(settings, Mapping):
            # Kept as an explicit branch so callers may add settings-only
            # checks without weakening the recursive mechanism scan above.
            pass

        paper = payload.get("paper_protocol")
        if not isinstance(paper, Mapping):
            errors.append("$.paper_protocol: missing normalized paper evidence")
        else:
            cell = paper.get("cell")
            if cell not in CELLS:
                errors.append("$.paper_protocol.cell: unknown factorial cell")
            else:
                expected = CELLS[str(cell)]
                for key, expected_value in expected.items():
                    if paper.get(key) != expected_value:
                        errors.append(
                            f"$.paper_protocol.{key}: expected {expected_value!r} for {cell}"
                        )
            if paper.get("offline_credit_s") != 0:
                errors.append("$.paper_protocol.offline_credit_s: must be exactly zero")
            if paper.get("all_tasks_successful") is not True:
                errors.append("$.paper_protocol.all_tasks_successful: must be true")
            if paper.get("broker_drained") is not True:
                errors.append("$.paper_protocol.broker_drained: must be true")
            call_graph = paper.get("call_graph_mode")
            if call_graph not in CALL_GRAPH_MODES:
                errors.append("$.paper_protocol.call_graph_mode: unsupported")
            expected_claim_type = {
                "autonomous": "closed_loop_agent",
                "trace_replay_causal_reveal": "systems_trace_replay",
            }.get(call_graph)
            if paper.get("claim_type") != expected_claim_type:
                errors.append(
                    "$.paper_protocol.claim_type: inconsistent with call graph"
                )
            if paper.get("claim_scope") not in {"retrospective", "confirmatory"}:
                errors.append(
                    "$.paper_protocol.claim_scope: expected retrospective or "
                    "confirmatory"
                )
            clock_mode = paper.get("physical_service_clock_mode")
            if clock_mode not in PHYSICAL_SERVICE_CLOCK_MODES:
                errors.append("$.paper_protocol.physical_service_clock_mode: unsupported")
            for field in (
                "service_assignment_policy_independent",
                "service_assignment_future_poison_invariant",
                "future_state_accepted_poison_invariance_test_passed",
                "same_invocation_service_clock_all_cells",
            ):
                if paper.get(field) is not True:
                    errors.append(f"$.paper_protocol.{field}: must be true")
            if call_graph == "trace_replay_causal_reveal":
                if paper.get("evaluation_trace_duration_role") != "diagnostic_only":
                    errors.append(
                        "$.paper_protocol.evaluation_trace_duration_role: "
                        "causal replay requires diagnostic_only"
                    )
                errors.extend(_audit_causal_reveal_events(payload))
            elif paper.get("evaluation_trace_duration_role") not in {
                "diagnostic_only",
                "not_applicable",
            }:
                errors.append("$.paper_protocol.evaluation_trace_duration_role: unsupported")
            if clock_mode == "calibration_hashed_empirical_v1" and not _is_sha256(
                paper.get("service_clock_artifact_sha256")
            ):
                errors.append(
                    "$.paper_protocol.service_clock_artifact_sha256: invalid"
                )
            starts = paper.get("physical_speculative_starts")
            if cell in {"A", "E"} and starts != 0:
                errors.append("$.paper_protocol: spec-off cell started speculative work")
            if cell in {"B", "F"} and (type(starts) is not int or starts < 0):
                errors.append("$.paper_protocol.physical_speculative_starts: invalid")
            if cell in {"B", "F"} and not isinstance(
                payload.get("prediction_decisions"), list
            ):
                errors.append("$: spec-on cell has no audited decision list")
            if cell in {"E", "F"} and metadata_count == 0:
                errors.append("$: causal scheduler cell has no audited metadata")

    if payload.get("schema") == GEMINI_RESULT_SCHEMA:
        _audit_gemini_legacy_compatibility_result(
            payload,
            context=gemini_legacy_compatibility,
            label="$",
            errors=errors,
        )
    elif compatibility_present:
        errors.append(
            "$.legacy_frozen_compatibility: only the Gemini strict result may bind it"
        )

    return errors


def _require_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label}: expected object")
        return {}
    return value


def _require_unique_strings(
    value: Any,
    label: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        errors.append(
            f"{label}: expected {'a list' if allow_empty else 'non-empty list'}"
        )
        return []
    if any(not isinstance(item, str) or not item for item in value):
        errors.append(f"{label}: entries must be non-empty strings")
        return []
    rows = list(value)
    if len(rows) != len(set(rows)):
        errors.append(f"{label}: duplicate root identities")
    return rows


def _verify_file_binding(
    entry: Mapping[str, Any], *, base: Path, label: str, errors: list[str]
) -> None:
    path_raw = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(path_raw, str) or not path_raw:
        errors.append(f"{label}.path: missing")
        return
    if not _is_sha256(expected):
        errors.append(f"{label}.sha256: invalid")
        return
    path = Path(path_raw)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        errors.append(f"{label}: bound file does not exist: {path}")
    elif file_sha256(path) != expected:
        errors.append(f"{label}: SHA-256 mismatch: {path}")


def _verify_embedded_identity(
    entry: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    fields: Sequence[str],
    errors: list[str],
) -> None:
    """Verify a declared logical identity against the bound JSON document."""

    identity = entry.get("identity_sha256")
    path_raw = entry.get("path")
    if not _is_sha256(identity) or not isinstance(path_raw, str) or not path_raw:
        return
    path = Path(path_raw)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        return
    try:
        document = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot verify embedded logical identity: {exc}")
        return
    if not isinstance(document, Mapping):
        errors.append(f"{label}: bound signed artifact must be a JSON object")
        return
    matching_fields = [field for field in fields if document.get(field) == identity]
    if not matching_fields:
        errors.append(
            f"{label}.identity_sha256: does not match a signed identity in the bound file"
        )
        return

    def signed_hash_matches(value: Mapping[str, Any], field: str, expected: str) -> bool:
        unsigned = dict(value)
        unsigned.pop(field, None)
        compact_utf8 = json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        compact_ascii = json.dumps(
            unsigned,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return expected in {
            hashlib.sha256(compact_utf8).hexdigest(),
            hashlib.sha256(compact_ascii).hexdigest(),
        }

    if any(signed_hash_matches(document, field, str(identity)) for field in matching_fields):
        return

    # Qwen's invocation-provenance wrapper deliberately exposes the signed
    # mapper model identity as artifact_sha256 while self-signing the wrapper
    # under provenance_sha256.  Validate both layers rather than mistaking the
    # logical model identity for a self-hash of the wrapper.
    provenance_identity = document.get("provenance_sha256")
    source_binding = document.get("source_artifact")
    if (
        "artifact_sha256" in matching_fields
        and _is_sha256(provenance_identity)
        and signed_hash_matches(
            document, "provenance_sha256", str(provenance_identity)
        )
        and isinstance(source_binding, Mapping)
    ):
        source_errors_before = len(errors)
        _verify_file_binding(
            source_binding,
            base=path.parent,
            label=f"{label}.source_artifact",
            errors=errors,
        )
        source_path_raw = source_binding.get("path")
        if len(errors) == source_errors_before and isinstance(source_path_raw, str):
            source_path = Path(source_path_raw)
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            try:
                source_document = _load_json(source_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{label}.source_artifact: unreadable JSON: {exc}")
            else:
                if (
                    isinstance(source_document, Mapping)
                    and source_document.get("artifact_sha256") == identity
                    and signed_hash_matches(
                        source_document, "artifact_sha256", str(identity)
                    )
                ):
                    return
    errors.append(f"{label}.identity_sha256: embedded signed hash is invalid")


def _embedded_training_root_identity(document: Mapping[str, Any]) -> Any:
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


def _verify_embedded_contract(
    entry: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    predictor: bool,
    errors: list[str],
) -> None:
    path_raw = entry.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        return
    path = Path(path_raw)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        return
    try:
        document = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(document, Mapping):
        errors.append(f"{label}: bound artifact must be a JSON object")
        return
    fields = (
        "training_root_ids_sha256",
        "uses_evaluation_labels",
    )
    for field in fields:
        embedded = (
            _embedded_training_root_identity(document)
            if field == "training_root_ids_sha256"
            else document.get(field)
        )
        if entry.get(field) != embedded:
            errors.append(f"{label}.{field}: differs from embedded artifact contract")
    if predictor:
        if entry.get("input_features") != document.get("input_features"):
            errors.append(f"{label}.input_features: differs from embedded artifact contract")
        if "fit_code_sha256" in document and entry.get("fit_code_sha256") != document.get(
            "fit_code_sha256"
        ):
            errors.append(f"{label}.fit_code_sha256: differs from embedded artifact contract")
    elif "uses_evaluation_trace_durations" in document and entry.get(
        "uses_evaluation_trace_durations"
    ) != document.get("uses_evaluation_trace_durations"):
        errors.append(
            f"{label}.uses_evaluation_trace_durations: differs from embedded artifact contract"
        )
    if not predictor and entry.get("future_state_accepted_invariant") != document.get(
        "future_state_accepted_invariant"
    ):
        errors.append(
            f"{label}.future_state_accepted_invariant: differs from embedded artifact contract"
        )


def _resolved_binding_path(
    binding: Mapping[str, Any], *, base: Path
) -> Path | None:
    path_raw = binding.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        return None
    path = Path(path_raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _binary64_hex(
    value: Any, *, label: str, errors: list[str]
) -> float | None:
    if not isinstance(value, str):
        errors.append(f"{label}: canonical float.hex string required")
        return None
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError):
        errors.append(f"{label}: invalid binary64 hex value")
        return None
    if not math.isfinite(number) or number <= 0.0 or number.hex() != value:
        errors.append(
            f"{label}: canonical finite strictly-positive binary64 required"
        )
        return None
    return number


def _gemini_duration_contract_model(
    contract: Any, *, label: str, errors: list[str]
) -> dict[str, Any] | None:
    contract = _require_mapping(contract, label, errors)
    expected_contract = {
        "schema": "paste_gemini.swe_strict_duration_inference_semantics.v1",
        "inputs": ["candidate_tool_name", "candidate_repository"],
        "value_encoding": "python_binary64_float_hex_v1",
        "positive_validation": (
            "all_inputs_and_output_finite_and_strictly_positive"
        ),
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
    if set(contract) != {*expected_contract, "model"}:
        errors.append(f"{label}: exact duration-contract fields required")
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            errors.append(f"{label}.{field}: expected {expected!r}")
    if contract.get("combiner") != "arithmetic_mean_tool_repository_v1":
        errors.append(f"{label}.combiner: unsupported")
    model = _require_mapping(contract.get("model"), f"{label}.model", errors)
    global_s = _binary64_hex(
        model.get("global_s"), label=f"{label}.model.global_s", errors=errors
    )
    parsed_maps: dict[str, dict[str, float]] = {}
    for field in ("by_tool_s", "by_repository_s"):
        raw = _require_mapping(model.get(field), f"{label}.model.{field}", errors)
        parsed: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                errors.append(f"{label}.model.{field}: invalid key")
                continue
            number = _binary64_hex(
                value, label=f"{label}.model.{field}.{key}", errors=errors
            )
            if number is not None:
                parsed[key] = number
        if not parsed:
            errors.append(f"{label}.model.{field}: non-empty mapping required")
        parsed_maps[field] = parsed
    if global_s is None:
        return None
    return {
        "global_s": global_s,
        "by_tool_s": parsed_maps["by_tool_s"],
        "by_repository_s": parsed_maps["by_repository_s"],
    }


def _gemini_duration_eta(
    model: Mapping[str, Any], *, repository: str, tool: str
) -> float:
    global_s = float(model["global_s"])
    tool_s = float(model["by_tool_s"].get(tool, global_s))
    repository_s = float(model["by_repository_s"].get(repository, global_s))
    return (tool_s + repository_s) / 2.0


def _audit_gemini_behavioral_vectors(
    certificate: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    label: str,
    errors: list[str],
) -> None:
    behavior = _require_mapping(
        certificate.get("behavioral_equivalence"),
        f"{label}.behavioral_equivalence",
        errors,
    )
    if behavior.get("exact_bitwise_match") is not True:
        errors.append(
            f"{label}.behavioral_equivalence.exact_bitwise_match: must be true"
        )
    tool_domain = behavior.get("tool_domain")
    repository_domain = behavior.get("repository_domain")
    if (
        not isinstance(tool_domain, list)
        or not tool_domain
        or any(not isinstance(value, str) or not value for value in tool_domain)
        or len(tool_domain) != len(set(tool_domain))
    ):
        errors.append(f"{label}.behavioral_equivalence.tool_domain: invalid")
        tool_domain = []
    if (
        not isinstance(repository_domain, list)
        or not repository_domain
        or any(
            not isinstance(value, str) or not value for value in repository_domain
        )
        or len(repository_domain) != len(set(repository_domain))
    ):
        errors.append(
            f"{label}.behavioral_equivalence.repository_domain: invalid"
        )
        repository_domain = []
    unknown_tool = behavior.get("unknown_tool_sentinel")
    unknown_repository = behavior.get("unknown_repository_sentinel")
    if (
        not isinstance(unknown_tool, str)
        or not unknown_tool
        or unknown_tool in model["by_tool_s"]
        or unknown_tool not in tool_domain
    ):
        errors.append(
            f"{label}.behavioral_equivalence.unknown_tool_sentinel: invalid"
        )
    if (
        not isinstance(unknown_repository, str)
        or not unknown_repository
        or unknown_repository in model["by_repository_s"]
        or unknown_repository not in repository_domain
    ):
        errors.append(
            f"{label}.behavioral_equivalence.unknown_repository_sentinel: invalid"
        )
    expected_tool_domain = [*sorted(model["by_tool_s"]), unknown_tool]
    expected_repository_domain = [
        *sorted(model["by_repository_s"]),
        unknown_repository,
    ]
    if tool_domain != expected_tool_domain:
        errors.append(
            f"{label}.behavioral_equivalence.tool_domain: must cover exact keys "
            "plus one unknown sentinel"
        )
    if repository_domain != expected_repository_domain:
        errors.append(
            f"{label}.behavioral_equivalence.repository_domain: must cover exact "
            "keys plus one unknown sentinel"
        )
    vectors = behavior.get("vectors")
    if not isinstance(vectors, list):
        errors.append(f"{label}.behavioral_equivalence.vectors: list required")
        return
    if behavior.get("behavioral_vector_count") != len(vectors):
        errors.append(
            f"{label}.behavioral_equivalence.behavioral_vector_count: mismatch"
        )
    if behavior.get("behavioral_vectors_sha256") != canonical_sha256(vectors):
        errors.append(
            f"{label}.behavioral_equivalence.behavioral_vectors_sha256: mismatch"
        )
    expected_pairs = [
        (tool, repository)
        for tool in tool_domain
        for repository in repository_domain
    ]
    observed_pairs: set[tuple[str, str]] = set()
    exact_fields = {
        "candidate_tool_name",
        "candidate_repository",
        "tool_lookup",
        "repository_lookup",
        "service_s_hex",
    }
    for index, raw in enumerate(vectors):
        row_label = f"{label}.behavioral_equivalence.vectors[{index}]"
        row = _require_mapping(raw, row_label, errors)
        if set(row) != exact_fields:
            errors.append(f"{row_label}: exact vector fields required")
        tool = row.get("candidate_tool_name")
        repository = row.get("candidate_repository")
        if not isinstance(tool, str) or not isinstance(repository, str):
            errors.append(f"{row_label}: string tool/repository required")
            continue
        pair = (tool, repository)
        if pair in observed_pairs:
            errors.append(f"{row_label}: duplicate behavioral input")
        observed_pairs.add(pair)
        expected_tool_lookup = (
            "exact_key" if tool in model["by_tool_s"] else "global_fallback"
        )
        expected_repository_lookup = (
            "exact_key"
            if repository in model["by_repository_s"]
            else "global_fallback"
        )
        if row.get("tool_lookup") != expected_tool_lookup:
            errors.append(f"{row_label}.tool_lookup: mismatch")
        if row.get("repository_lookup") != expected_repository_lookup:
            errors.append(f"{row_label}.repository_lookup: mismatch")
        expected_eta = _gemini_duration_eta(
            model, repository=repository, tool=tool
        )
        if row.get("service_s_hex") != expected_eta.hex():
            errors.append(f"{row_label}.service_s_hex: recomputation mismatch")
    observed_order = [
        (str(row.get("candidate_tool_name")), str(row.get("candidate_repository")))
        for row in vectors
        if isinstance(row, Mapping)
    ]
    if observed_pairs != set(expected_pairs) or observed_order != expected_pairs:
        errors.append(
            f"{label}.behavioral_equivalence.vectors: incomplete or reordered "
            "Cartesian domain"
        )


def _audit_gemini_legacy_compatibility_manifest(
    root: Mapping[str, Any],
    *,
    base: Path,
    verify_files: bool,
    errors: list[str],
) -> dict[str, Any] | None:
    """Validate and, at the final file boundary, independently re-run proof."""

    frozen_raw = root.get("frozen_files")
    frozen_by_role = {
        str(row.get("role")): row
        for row in frozen_raw
        if isinstance(row, Mapping)
    } if isinstance(frozen_raw, list) else {}
    certificate_binding = frozen_by_role.get(
        GEMINI_LEGACY_COMPATIBILITY_CERTIFICATE_ROLE
    )
    verifier_binding = frozen_by_role.get(GEMINI_LEGACY_COMPATIBILITY_VERIFIER_ROLE)
    if certificate_binding is None and verifier_binding is None:
        policy_binding = frozen_by_role.get("policy_bundle")
        policy_path = (
            _resolved_binding_path(policy_binding, base=base)
            if isinstance(policy_binding, Mapping)
            else None
        )
        if verify_files and policy_path is not None and policy_path.is_file():
            try:
                policy_document = _load_json(policy_path)
            except (OSError, json.JSONDecodeError):
                policy_document = None
            if (
                isinstance(policy_document, Mapping)
                and policy_document.get("schema")
                == "paste_gemini.swe_strict_policy_plan.v1"
                and policy_document.get("legacy_frozen_compatibility") is not None
            ):
                errors.append(
                    "$.frozen_files: Gemini strict policy requires frozen legacy "
                    "compatibility certificate and verifier roles"
                )
        return None
    if not isinstance(certificate_binding, Mapping) or not isinstance(
        verifier_binding, Mapping
    ):
        errors.append(
            "$.frozen_files: Gemini legacy compatibility requires certificate "
            "and verifier roles together"
        )
        return {}
    certificate_label = "$.frozen_files.legacy_compatibility_certificate"
    verifier_label = "$.frozen_files.legacy_compatibility_verifier"
    if not _is_sha256(certificate_binding.get("identity_sha256")):
        errors.append(f"{certificate_label}.identity_sha256: invalid")
    if not _is_sha256(certificate_binding.get("verifier_sha256")):
        errors.append(f"{certificate_label}.verifier_sha256: invalid")
    if certificate_binding.get("verifier_sha256") != verifier_binding.get("sha256"):
        errors.append(f"{certificate_label}: verifier binding mismatch")
    if certificate_binding.get("schema") != GEMINI_LEGACY_COMPATIBILITY_SCHEMA:
        errors.append(f"{certificate_label}.schema: mismatch")
    if (
        certificate_binding.get("compatibility_mode")
        != GEMINI_LEGACY_COMPATIBILITY_MODE
    ):
        errors.append(f"{certificate_label}.compatibility_mode: mismatch")
    certificate_path = _resolved_binding_path(certificate_binding, base=base)
    verifier_path = _resolved_binding_path(verifier_binding, base=base)
    if not verify_files:
        return {
            "certificate_binding": certificate_binding,
            "verifier_binding": verifier_binding,
        }
    if certificate_path is None or verifier_path is None:
        return {}
    try:
        certificate = _load_json(certificate_path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{certificate_label}: unreadable certificate: {exc}")
        return {}
    if not isinstance(certificate, Mapping):
        errors.append(f"{certificate_label}: certificate must be an object")
        return {}
    exact_certificate_fields = {
        "schema",
        "compatibility_mode",
        "artifact_tuple",
        "duration_inference_contract",
        "duration_inference_contract_sha256",
        "behavioral_equivalence",
        "current_cell_runner_sha256",
        "independent_verifier_sha256",
        "compatibility_sha256",
    }
    if set(certificate) != exact_certificate_fields:
        errors.append(f"{certificate_label}: exact certificate fields required")
    if certificate.get("schema") != GEMINI_LEGACY_COMPATIBILITY_SCHEMA:
        errors.append(f"{certificate_label}.schema: mismatch")
    if certificate.get("compatibility_mode") != GEMINI_LEGACY_COMPATIBILITY_MODE:
        errors.append(f"{certificate_label}.compatibility_mode: mismatch")
    identity = certificate.get("compatibility_sha256")
    unsigned = dict(certificate)
    unsigned.pop("compatibility_sha256", None)
    compact_ascii = json.dumps(
        unsigned,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        not _is_sha256(identity)
        or identity != hashlib.sha256(compact_ascii).hexdigest()
        or identity != certificate_binding.get("identity_sha256")
    ):
        errors.append(f"{certificate_label}.compatibility_sha256: invalid")
    if certificate.get("independent_verifier_sha256") != verifier_binding.get(
        "sha256"
    ):
        errors.append(f"{certificate_label}.independent_verifier_sha256: mismatch")

    contract = certificate.get("duration_inference_contract")
    if certificate.get("duration_inference_contract_sha256") != canonical_sha256(
        contract
    ):
        errors.append(f"{certificate_label}.duration_inference_contract_sha256: mismatch")
    model = _gemini_duration_contract_model(
        contract,
        label=f"{certificate_label}.duration_inference_contract",
        errors=errors,
    )
    if model is not None:
        _audit_gemini_behavioral_vectors(
            certificate, model=model, label=certificate_label, errors=errors
        )

    artifact_tuple = _require_mapping(
        certificate.get("artifact_tuple"),
        f"{certificate_label}.artifact_tuple",
        errors,
    )
    exact_tuple_fields = {
        "invocation_file_sha256",
        "invocation_logical_sha256",
        "duration_file_sha256",
        "duration_logical_sha256",
        "service_clock_file_sha256",
        "service_clock_logical_sha256",
        "historical_builder_code_sha256",
        "invocation_runtime_module_sha256",
        "duration_model_sha256",
        "calibration_root_ids_sha256",
        "source_registry_sha256",
    }
    if set(artifact_tuple) != exact_tuple_fields:
        errors.append(f"{certificate_label}.artifact_tuple: exact fields required")
    predictors = root.get("predictors")
    predictors = predictors if isinstance(predictors, Mapping) else {}
    invocation = predictors.get("tool_invocation")
    invocation = invocation if isinstance(invocation, Mapping) else {}
    invocation_binding = invocation.get("artifact")
    invocation_binding = (
        invocation_binding if isinstance(invocation_binding, Mapping) else {}
    )
    duration = predictors.get("tool_duration")
    duration = duration if isinstance(duration, Mapping) else {}
    duration_binding = duration.get("artifact")
    duration_binding = duration_binding if isinstance(duration_binding, Mapping) else {}
    execution = root.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    clock = execution.get("physical_service_clock")
    clock = clock if isinstance(clock, Mapping) else {}
    service_binding = clock.get("artifact")
    service_binding = service_binding if isinstance(service_binding, Mapping) else {}
    expected_tuple = {
        "invocation_file_sha256": invocation_binding.get("sha256"),
        "invocation_logical_sha256": artifact_identity_sha256(invocation_binding),
        "duration_file_sha256": duration_binding.get("sha256"),
        "duration_logical_sha256": artifact_identity_sha256(duration_binding),
        "service_clock_file_sha256": service_binding.get("sha256"),
        "service_clock_logical_sha256": artifact_identity_sha256(service_binding),
        "calibration_root_ids_sha256": canonical_sha256(
            sorted(root.get("data", {}).get("calibration_root_ids", []))
        ),
    }
    for field, expected in expected_tuple.items():
        if artifact_tuple.get(field) != expected:
            errors.append(f"{certificate_label}.artifact_tuple.{field}: mismatch")

    runner_binding = frozen_by_role.get("runner")
    prediction_binding = frozen_by_role.get("prediction_code")
    if not isinstance(runner_binding, Mapping):
        runner_binding = {}
    if certificate.get("current_cell_runner_sha256") != runner_binding.get("sha256"):
        errors.append(f"{certificate_label}.current_cell_runner_sha256: mismatch")
    if isinstance(prediction_binding, Mapping) and artifact_tuple.get(
        "invocation_runtime_module_sha256"
    ) != prediction_binding.get("sha256"):
        errors.append(
            f"{certificate_label}.artifact_tuple.invocation_runtime_module_sha256: mismatch"
        )

    invocation_path = _resolved_binding_path(invocation_binding, base=base)
    duration_path = _resolved_binding_path(duration_binding, base=base)
    service_path = _resolved_binding_path(service_binding, base=base)
    runner_path = _resolved_binding_path(runner_binding, base=base)
    try:
        invocation_document = _load_json(invocation_path) if invocation_path else None
        duration_document = _load_json(duration_path) if duration_path else None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{certificate_label}: cannot inspect frozen artifacts: {exc}")
        invocation_document = duration_document = None
    if isinstance(invocation_document, Mapping):
        training = invocation_document.get("training")
        training = training if isinstance(training, Mapping) else {}
        if artifact_tuple.get("source_registry_sha256") != training.get(
            "source_registry_sha256"
        ):
            errors.append(f"{certificate_label}.artifact_tuple.source_registry_sha256: mismatch")
        if artifact_tuple.get("historical_builder_code_sha256") != invocation_document.get(
            "builder_code_sha256"
        ):
            errors.append(
                f"{certificate_label}.artifact_tuple.historical_builder_code_sha256: mismatch"
            )
    if isinstance(duration_document, Mapping):
        historical_builder = artifact_tuple.get("historical_builder_code_sha256")
        if any(
            duration_document.get(field) != historical_builder
            for field in ("fit_code_sha256", "prediction_code_sha256")
        ):
            errors.append(
                f"{certificate_label}.artifact_tuple.historical_builder_code_sha256: "
                "does not bind duration artifact"
            )
        duration_model = duration_document.get("model")
        duration_model = duration_model if isinstance(duration_model, Mapping) else {}
        if artifact_tuple.get("duration_model_sha256") != duration_model.get(
            "duration_predictor_sha256"
        ):
            errors.append(f"{certificate_label}.artifact_tuple.duration_model_sha256: mismatch")
        if model is not None:
            for field, parsed in (
                ("global_s", model["global_s"]),
                ("by_tool_s", model["by_tool_s"]),
                ("by_repository_s", model["by_repository_s"]),
            ):
                embedded = duration_model.get(field)
                if field == "global_s":
                    matches = _is_number(embedded) and float(embedded).hex() == float(parsed).hex()
                else:
                    matches = isinstance(embedded, Mapping) and {
                        str(key): float(value).hex() for key, value in embedded.items()
                    } == {
                        str(key): float(value).hex() for key, value in parsed.items()
                    }
                if not matches:
                    errors.append(
                        f"{certificate_label}.duration_inference_contract.model.{field}: "
                        "differs from duration artifact"
                    )

    policy_binding = frozen_by_role.get("policy_bundle")
    policy_path = (
        _resolved_binding_path(policy_binding, base=base)
        if isinstance(policy_binding, Mapping)
        else None
    )
    try:
        policy_document = _load_json(policy_path) if policy_path else None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{certificate_label}: cannot inspect policy bundle: {exc}")
        policy_document = None
    registered_task_repository: dict[str, str] = {}
    expected_runtime_binding = {
        "schema": GEMINI_LEGACY_COMPATIBILITY_SCHEMA,
        "compatibility_mode": GEMINI_LEGACY_COMPATIBILITY_MODE,
        "certificate_file_sha256": certificate_binding.get("sha256"),
        "compatibility_sha256": certificate_binding.get("identity_sha256"),
        "independent_verifier_sha256": verifier_binding.get("sha256"),
    }
    if isinstance(policy_document, Mapping):
        policy_compatibility = policy_document.get("legacy_frozen_compatibility")
        if not isinstance(policy_compatibility, Mapping):
            errors.append(
                f"{certificate_label}: policy bundle lacks legacy_frozen_compatibility"
            )
        else:
            expected_binding_fields = {
                "schema",
                "compatibility_mode",
                "certificate_path",
                "certificate_file_sha256",
                "compatibility_sha256",
                "independent_verifier_path",
                "independent_verifier_sha256",
            }
            if set(policy_compatibility) != expected_binding_fields:
                errors.append(
                    f"{certificate_label}: policy compatibility exact fields required"
                )
            for field, expected in expected_runtime_binding.items():
                if policy_compatibility.get(field) != expected:
                    errors.append(
                        f"{certificate_label}: policy compatibility {field} mismatch"
                    )
            for field, expected_path in (
                ("certificate_path", certificate_path),
                ("independent_verifier_path", verifier_path),
            ):
                raw_path = policy_compatibility.get(field)
                if (
                    not isinstance(raw_path, str)
                    or Path(raw_path).resolve() != expected_path
                ):
                    errors.append(
                        f"{certificate_label}: policy compatibility {field} mismatch"
                    )
        templates = policy_document.get("templates")
        sessions = policy_document.get("sessions")
        if isinstance(templates, list) and isinstance(sessions, list):
            repository_by_template: dict[int, str] = {}
            for index, raw in enumerate(templates):
                if not isinstance(raw, Mapping):
                    errors.append(
                        f"{certificate_label}: policy template {index} is malformed"
                    )
                    continue
                template_index = raw.get("template_index")
                repository = raw.get("repository")
                if (
                    type(template_index) is not int
                    or not isinstance(repository, str)
                    or not repository
                    or template_index in repository_by_template
                ):
                    errors.append(
                        f"{certificate_label}: policy template repository map is invalid"
                    )
                    continue
                repository_by_template[template_index] = repository
            for index, raw in enumerate(sessions):
                if not isinstance(raw, Mapping):
                    errors.append(
                        f"{certificate_label}: policy session {index} is malformed"
                    )
                    continue
                task_id = raw.get("task_id")
                template_index = raw.get("template_index")
                repository = repository_by_template.get(template_index)
                if (
                    not isinstance(task_id, str)
                    or not task_id
                    or repository is None
                    or task_id in registered_task_repository
                ):
                    errors.append(
                        f"{certificate_label}: policy session/repository map is invalid"
                    )
                    continue
                registered_task_repository[task_id] = repository

    if all(
        path is not None and path.is_file()
        for path in (
            certificate_path,
            verifier_path,
            invocation_path,
            duration_path,
            service_path,
            runner_path,
        )
    ):
        repository_root = runner_path.parents[2]
        command = [
            sys.executable,
            "-I",
            str(verifier_path),
            "verify",
            "--certificate",
            str(certificate_path),
            "--invocation-artifact",
            str(invocation_path),
            "--duration-artifact",
            str(duration_path),
            "--service-clock",
            str(service_path),
            "--cell-runner",
            str(runner_path),
            "--repository-root",
            str(repository_root),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                text=True,
                capture_output=True,
                timeout=180.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{verifier_label}: independent verification failed: {exc}")
        else:
            try:
                verifier_result = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                errors.append(f"{verifier_label}: invalid verifier JSON: {exc}")
            else:
                if completed.returncode != 0 or not isinstance(
                    verifier_result, Mapping
                ) or verifier_result.get("valid") is not True:
                    errors.append(
                        f"{verifier_label}: independent verifier rejected certificate"
                    )
                else:
                    expected_verifier_result = {
                        "compatibility_sha256": certificate.get(
                            "compatibility_sha256"
                        ),
                        "invocation_logical_sha256": artifact_tuple.get(
                            "invocation_logical_sha256"
                        ),
                        "duration_logical_sha256": artifact_tuple.get(
                            "duration_logical_sha256"
                        ),
                        "service_clock_logical_sha256": artifact_tuple.get(
                            "service_clock_logical_sha256"
                        ),
                        "current_cell_runner_sha256": certificate.get(
                            "current_cell_runner_sha256"
                        ),
                        "independent_verifier_sha256": verifier_binding.get(
                            "sha256"
                        ),
                        "behavioral_vectors_sha256": certificate.get(
                            "behavioral_equivalence", {}
                        ).get("behavioral_vectors_sha256"),
                        "behavioral_vector_count": certificate.get(
                            "behavioral_equivalence", {}
                        ).get("behavioral_vector_count"),
                    }
                    if set(verifier_result) != {
                        "valid",
                        *expected_verifier_result,
                    }:
                        errors.append(
                            f"{verifier_label}: verifier result fields changed"
                        )
                    for field, expected in expected_verifier_result.items():
                        if verifier_result.get(field) != expected:
                            errors.append(
                                f"{verifier_label}: verifier result {field} mismatch"
                            )
    else:
        errors.append(f"{verifier_label}: verifier inputs are incomplete")
    return {
        "certificate": certificate,
        "certificate_binding": certificate_binding,
        "certificate_path": certificate_path,
        "verifier_binding": verifier_binding,
        "verifier_path": verifier_path,
        "duration_model": model,
        "registered_task_repository": registered_task_repository,
        "expected_runtime_binding": expected_runtime_binding,
    }


def _eta_is_exact(value: Any, expected: float) -> bool:
    return _is_number(value) and float(value).hex() == expected.hex()


def _audit_gemini_legacy_compatibility_result(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None,
    label: str,
    errors: list[str],
) -> None:
    """Bind a Gemini result to the proof and recompute every emitted ETA."""

    binding_raw = payload.get("legacy_frozen_compatibility")
    if binding_raw is None and context is None:
        return
    binding = _require_mapping(
        binding_raw, f"{label}.legacy_frozen_compatibility", errors
    )
    required_binding_fields = {
        "schema",
        "compatibility_mode",
        "certificate_path",
        "certificate_file_sha256",
        "compatibility_sha256",
        "independent_verifier_path",
        "independent_verifier_sha256",
    }
    if set(binding) != required_binding_fields:
        errors.append(
            f"{label}.legacy_frozen_compatibility: exact binding fields required"
        )
    if binding.get("schema") != GEMINI_LEGACY_COMPATIBILITY_SCHEMA:
        errors.append(f"{label}.legacy_frozen_compatibility.schema: mismatch")
    if binding.get("compatibility_mode") != GEMINI_LEGACY_COMPATIBILITY_MODE:
        errors.append(
            f"{label}.legacy_frozen_compatibility.compatibility_mode: mismatch"
        )
    for field in (
        "certificate_file_sha256",
        "compatibility_sha256",
        "independent_verifier_sha256",
    ):
        if not _is_sha256(binding.get(field)):
            errors.append(
                f"{label}.legacy_frozen_compatibility.{field}: invalid SHA-256"
            )
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    provenance_pairs = {
        "certificate_file_sha256": (
            "legacy_compatibility_certificate_file_sha256"
        ),
        "compatibility_sha256": "legacy_compatibility_certificate_sha256",
        "independent_verifier_sha256": (
            "legacy_compatibility_verifier_file_sha256"
        ),
    }
    for binding_field, provenance_field in provenance_pairs.items():
        if binding.get(binding_field) != provenance.get(provenance_field):
            errors.append(
                f"{label}.legacy_frozen_compatibility.{binding_field}: "
                "differs from normalized provenance"
            )
    if context is None:
        return
    expected_binding = context.get("expected_runtime_binding")
    expected_binding = (
        expected_binding if isinstance(expected_binding, Mapping) else {}
    )
    for field, expected in expected_binding.items():
        if binding.get(field) != expected:
            errors.append(
                f"{label}.legacy_frozen_compatibility.{field}: "
                "differs from frozen certificate"
            )
    certificate_binding = context.get("certificate_binding")
    verifier_binding = context.get("verifier_binding")
    if isinstance(certificate_binding, Mapping):
        expected_path = context.get("certificate_path")
        result_path_raw = binding.get("certificate_path")
        if (
            isinstance(expected_path, Path)
            and (
                not isinstance(result_path_raw, str)
                or Path(result_path_raw).resolve() != expected_path
            )
        ):
            errors.append(
                f"{label}.legacy_frozen_compatibility.certificate_path: mismatch"
            )
    if isinstance(verifier_binding, Mapping):
        expected_path = context.get("verifier_path")
        result_path_raw = binding.get("independent_verifier_path")
        if (
            isinstance(expected_path, Path)
            and (
                not isinstance(result_path_raw, str)
                or Path(result_path_raw).resolve() != expected_path
            )
        ):
            errors.append(
                f"{label}.legacy_frozen_compatibility.independent_verifier_path: mismatch"
            )
    model = context.get("duration_model")
    if not isinstance(model, Mapping):
        errors.append(
            f"{label}.legacy_frozen_compatibility: duration model unavailable"
        )
        return

    tasks_raw = payload.get("task_results")
    if not isinstance(tasks_raw, list):
        tasks_raw = payload.get("tasks")
    task_repository: dict[str, str] = {}
    if isinstance(tasks_raw, list):
        for index, raw in enumerate(tasks_raw):
            if not isinstance(raw, Mapping):
                continue
            task_id = raw.get("task_id", raw.get("trace_id"))
            repository = raw.get("repository")
            if not isinstance(task_id, str) or not task_id:
                continue
            if not isinstance(repository, str) or not repository:
                errors.append(
                    f"{label}.tasks[{index}].repository: required for independent ETA audit"
                )
                continue
            previous = task_repository.setdefault(task_id, repository)
            if previous != repository:
                errors.append(f"{label}.tasks[{index}].repository: task mismatch")
    registered_task_repository = context.get("registered_task_repository")
    if isinstance(registered_task_repository, Mapping) and registered_task_repository:
        if set(task_repository) != set(registered_task_repository):
            errors.append(
                f"{label}.tasks: task IDs differ from frozen policy repository map"
            )
        for task_id, expected_repository in registered_task_repository.items():
            if task_repository.get(task_id) != expected_repository:
                errors.append(
                    f"{label}.tasks: repository differs from frozen policy for {task_id}"
                )
        # ETA recomputation uses the frozen mapping, never the result-supplied
        # copy that is checked above only as redundant evidence.
        task_repository = {
            str(task_id): str(repository)
            for task_id, repository in registered_task_repository.items()
        }

    decisions_raw = payload.get("prediction_decisions")
    decisions = decisions_raw if isinstance(decisions_raw, list) else []
    decision_by_request: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, raw in enumerate(decisions):
        decision_label = f"{label}.prediction_decisions[{index}]"
        if not isinstance(raw, Mapping):
            continue
        task_id = raw.get("task_id", raw.get("trace_id"))
        request_index = raw.get("request_index")
        tool = raw.get("tool_name_hat")
        if (
            not isinstance(task_id, str)
            or type(request_index) is not int
            or not isinstance(tool, str)
            or not tool
        ):
            errors.append(
                f"{decision_label}: task/request/tool required for ETA recomputation"
            )
            continue
        repository = task_repository.get(task_id)
        if repository is None:
            errors.append(f"{decision_label}: no task repository binding")
            continue
        key = (task_id, request_index)
        if key in decision_by_request:
            errors.append(f"{decision_label}: duplicate task/request decision")
        decision_by_request[key] = raw
        expected_eta = _gemini_duration_eta(
            model, repository=repository, tool=tool
        )
        if not _eta_is_exact(raw.get("tool_service_s_hat"), expected_eta):
            errors.append(
                f"{decision_label}.tool_service_s_hat: independent recomputation mismatch"
            )
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            errors.append(f"{decision_label}.candidates: non-empty list required")
        else:
            for candidate_index, candidate_raw in enumerate(candidates):
                candidate_label = f"{decision_label}.candidates[{candidate_index}]"
                candidate = _require_mapping(candidate_raw, candidate_label, errors)
                candidate_tool = candidate.get("tool_name_hat")
                if not isinstance(candidate_tool, str) or not candidate_tool:
                    errors.append(f"{candidate_label}.tool_name_hat: missing")
                    continue
                candidate_eta = _gemini_duration_eta(
                    model, repository=repository, tool=candidate_tool
                )
                if not _eta_is_exact(candidate.get("tool_eta_s_hat"), candidate_eta):
                    errors.append(
                        f"{candidate_label}.tool_eta_s_hat: independent recomputation mismatch"
                    )

    llm_raw = payload.get("llm_events")
    if isinstance(llm_raw, list):
        for index, raw in enumerate(llm_raw):
            if not isinstance(raw, Mapping):
                continue
            metadata = raw.get("scheduler_metadata")
            if metadata is None:
                continue
            event_label = f"{label}.llm_events[{index}].scheduler_metadata"
            if not isinstance(metadata, Mapping):
                continue
            task_id = raw.get("task_id", raw.get("trace_id"))
            request_index = raw.get("request_index")
            decision = decision_by_request.get((str(task_id), request_index))
            if decision is None:
                errors.append(f"{event_label}: no matching prediction decision")
                continue
            repository = task_repository.get(str(task_id))
            tool = decision.get("tool_name_hat")
            if repository is None or not isinstance(tool, str):
                continue
            expected_eta = _gemini_duration_eta(
                model, repository=repository, tool=tool
            )
            for field in ("tool_eta_s_hat", "remaining_tool_wait_s_hat"):
                if field in metadata and not _eta_is_exact(
                    metadata.get(field), expected_eta
                ):
                    errors.append(
                        f"{event_label}.{field}: independent recomputation mismatch"
                    )

    tools_raw = payload.get("tool_events")
    if isinstance(tools_raw, list):
        for index, raw in enumerate(tools_raw):
            if not isinstance(raw, Mapping):
                continue
            tool_label = f"{label}.tool_events[{index}]"
            task_id = raw.get("task_id", raw.get("trace_id"))
            request_index = raw.get("request_index")
            repository = task_repository.get(str(task_id))
            authority_tool = raw.get("tool", raw.get("tool_name"))
            if repository is None or not isinstance(authority_tool, str):
                errors.append(f"{tool_label}: task repository/tool binding missing")
                continue
            authority_eta = _gemini_duration_eta(
                model, repository=repository, tool=authority_tool
            )
            if not _eta_is_exact(raw.get("authority_eta_hat_s"), authority_eta):
                errors.append(
                    f"{tool_label}.authority_eta_hat_s: independent recomputation mismatch"
                )
            decision = decision_by_request.get((str(task_id), request_index))
            predicted = raw.get("predicted_tool_service_s_hat")
            if decision is not None:
                predicted_tool = decision.get("tool_name_hat")
                if isinstance(predicted_tool, str):
                    predicted_eta = _gemini_duration_eta(
                        model, repository=repository, tool=predicted_tool
                    )
                    if not _eta_is_exact(predicted, predicted_eta):
                        errors.append(
                            f"{tool_label}.predicted_tool_service_s_hat: "
                            "independent recomputation mismatch"
                        )


def _audit_file_binding_shape(
    entry: Mapping[str, Any], *, label: str, errors: list[str]
) -> None:
    if not isinstance(entry.get("path"), str) or not entry.get("path"):
        errors.append(f"{label}.path: missing")
    if not _is_sha256(entry.get("sha256")):
        errors.append(f"{label}.sha256: invalid")


def _runtime_environment_document(
    binding: Mapping[str, Any], *, base: Path
) -> Mapping[str, Any] | None:
    path_raw = binding.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        return None
    path = Path(path_raw)
    if not path.is_absolute():
        path = base / path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = {}
        for line in text.splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, raw = line.split("=", 1)
            if key:
                value[key] = raw
    return value if isinstance(value, Mapping) else None


def _audit_platform_evidence(
    value: Any,
    *,
    base: Path,
    verify_files: bool,
    expected_model_inventory_sha256: str,
    runtime_environment_contract: Mapping[str, Any],
    cell: str,
    block_id: str,
    order_position: int,
    expected_provenance: Mapping[str, Any],
    gpu_ids: Sequence[int],
    server_instance_id: str,
    label: str,
    errors: list[str],
) -> Mapping[str, Any]:
    """Verify every retained platform artifact and its runtime-env semantics."""

    platform = _require_mapping(value, label, errors)
    if not platform:
        errors.append(f"{label}: non-empty mapping required")
        return platform
    for name, raw in platform.items():
        item_label = f"{label}.{name}"
        if not isinstance(name, str) or not name:
            errors.append(f"{label}: evidence names must be non-empty strings")
            continue
        binding = _require_mapping(raw, item_label, errors)
        _audit_file_binding_shape(binding, label=item_label, errors=errors)
        if verify_files:
            _verify_file_binding(binding, base=base, label=item_label, errors=errors)

    environment_binding = platform.get("runtime_environment")
    if not isinstance(environment_binding, Mapping):
        errors.append(f"{label}.runtime_environment: required")
        return platform
    if environment_binding.get("sha256") != runtime_environment_contract.get(
        "evidence_sha256"
    ):
        errors.append(
            f"{label}.runtime_environment: hash differs from runtime contract"
        )
    if not verify_files:
        return platform
    document = _runtime_environment_document(environment_binding, base=base)
    if document is None:
        errors.append(f"{label}.runtime_environment: unreadable evidence document")
        return platform
    schema = document.get("schema")
    if document.get("model_snapshot_inventory_sha256") != (
        expected_model_inventory_sha256
    ):
        errors.append(
            f"{label}.runtime_environment: model inventory identity mismatch"
        )
    if document.get("environment_scrubbed_before_cell") not in {True, "true"}:
        # Qwen records the effective env itself plus a scrub-list/hash rather
        # than repeating this boolean; its schema is handled below.
        if schema != "paste.paper.frozen_cell_environment.v1":
            errors.append(
                f"{label}.runtime_environment: environment was not scrubbed"
            )

    if schema == "paste.paper.frozen_cell_environment.v1":
        required_names = {
            "nvidia_smi_pre",
            "nvidia_smi_post",
            "protected_pid_pre",
            "protected_pid_post",
            "standardized_smoke",
            "start_vllm",
            "stop_vllm",
            "sitecustomize",
            "runtime_environment",
            "scheduler_runtime_after_smoke",
            "scheduler_runtime_after_cell",
            "server_log",
        }
        missing = sorted(required_names - set(platform))
        if missing:
            errors.append(f"{label}: Qwen platform evidence missing {missing}")
        for field in (
            "frozen_cell_environment_sha256",
            "model_snapshot_inventory_sha256",
        ):
            if not _is_sha256(document.get(field)):
                errors.append(f"{label}.runtime_environment.{field}: invalid")
        expected_policy = "online_joint_pacer_v2" if cell in {"E", "F"} else "fcfs"
        if document.get("server_scheduler_policy") != expected_policy:
            errors.append(
                f"{label}.runtime_environment.server_scheduler_policy: "
                "differs from factorial cell"
            )
        if document.get("PYTHONSAFEPATH") != "1" or document.get(
            "PYTHONNOUSERSITE"
        ) != "1":
            errors.append(
                f"{label}.runtime_environment: Python safe-path/no-user-site not active"
            )
        for field in (
            "wrapper_working_directory",
            "server_empty_working_directory",
            "server_pythonpath",
        ):
            raw = document.get(field)
            if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
                errors.append(f"{label}.runtime_environment.{field}: absolute path required")
        empty_cwd_raw = document.get("server_empty_working_directory")
        if isinstance(empty_cwd_raw, str):
            empty_cwd = Path(empty_cwd_raw)
            if (
                not empty_cwd.is_dir()
                or any(empty_cwd.iterdir())
                or empty_cwd.stat().st_mode & 0o222
            ):
                errors.append(
                    f"{label}.runtime_environment.server_empty_working_directory: "
                    "must still be an empty directory"
                )
        for evidence_name, document_field in (
            ("sitecustomize", "runtime_sitecustomize_resolves_to"),
            ("scheduler_runtime_after_smoke", "scheduler_runtime_marker"),
        ):
            binding = platform.get(evidence_name)
            expected_path = document.get(document_field)
            if evidence_name == "scheduler_runtime_after_smoke":
                # The document names the mutable engine marker; the retained
                # after-smoke JSON wraps/authenticates it, so no path equality
                # is expected here.
                continue
            if isinstance(binding, Mapping) and isinstance(expected_path, str):
                if Path(str(binding.get("path", ""))).resolve() != Path(
                    expected_path
                ).resolve():
                    errors.append(
                        f"{label}.{evidence_name}: path differs from runtime evidence"
                    )
        hook_path_raw = document.get("runtime_scheduler_hook_resolves_to")
        if (
            not isinstance(hook_path_raw, str)
            or not Path(hook_path_raw).is_file()
            or file_sha256(Path(hook_path_raw))
            != expected_provenance.get("scheduler_hook_file_sha256")
        ):
            errors.append(
                f"{label}.runtime_environment: loaded scheduler hook bytes mismatch"
            )
    elif schema == "paste_gemini.strict_runtime_environment_evidence.v1":
        base_names = {
            "runtime_environment",
            "server",
            "smoke",
            "protocol_response",
            "machine_before",
            "machine_after",
        }
        treatment_names = {"scheduler_hook_load", "scheduler_runtime_marker"}
        missing = sorted(base_names - set(platform))
        if missing:
            errors.append(f"{label}: Gemini platform evidence missing {missing}")
        if cell in {"E", "F"}:
            missing_treatment = sorted(treatment_names - set(platform))
            if missing_treatment:
                errors.append(
                    f"{label}: Gemini scheduler evidence missing {missing_treatment}"
                )
        elif treatment_names & set(platform):
            errors.append(f"{label}: A/B must not claim Gemini scheduler-hook evidence")
        for field, expected in (
            ("cell", cell),
            ("block_id", block_id),
            ("order_position", order_position),
            ("gpu_ids", list(gpu_ids)),
            ("server_instance_id", server_instance_id),
            (
                "server_policy",
                "paste_joint" if cell in {"E", "F"} else "native_fcfs",
            ),
            ("environment_scrubbed_before_cell", True),
            ("server_and_client_launched_via_env_i", True),
            ("client_python_isolated", True),
            ("client_cwd_removed_from_sys_path", True),
            ("client_python_safe_path_supported", False),
            ("client_python_no_user_site", True),
        ):
            if document.get(field) != expected:
                errors.append(
                    f"{label}.runtime_environment.{field}: expected {expected!r}"
                )
        for field in (
            "frozen_environment_sha256",
            "effective_environment_sha256",
            "client_environment_sha256",
        ):
            if not _is_sha256(document.get(field)):
                errors.append(f"{label}.runtime_environment.{field}: invalid")
        for field in ("effective_environment_keys", "client_environment_keys"):
            keys = document.get(field)
            if (
                not isinstance(keys, list)
                or any(not isinstance(item, str) or not item for item in keys)
                or keys != sorted(set(keys))
            ):
                errors.append(
                    f"{label}.runtime_environment.{field}: "
                    "sorted unique string list required"
                )
        runtime_cwd = document.get("runtime_cwd")
        if not isinstance(runtime_cwd, str) or not Path(runtime_cwd).is_absolute():
            errors.append(f"{label}.runtime_environment.runtime_cwd: absolute path required")
        elif (
            not Path(runtime_cwd).is_dir()
            or any(Path(runtime_cwd).iterdir())
            or Path(runtime_cwd).stat().st_mode & 0o222
        ):
            errors.append(
                f"{label}.runtime_environment.runtime_cwd: "
                "must still be an empty directory"
            )
        server = _require_mapping(
            document.get("server"), f"{label}.runtime_environment.server", errors
        )
        if server.get("runtime_cwd") != runtime_cwd:
            errors.append(f"{label}.runtime_environment.server.runtime_cwd: mismatch")
        if server.get("policy") != (
            "paste_joint" if cell in {"E", "F"} else "native_fcfs"
        ):
            errors.append(f"{label}.runtime_environment.server.policy: mismatch")
        if (
            type(server.get("fresh_pid")) is not int
            or int(server["fresh_pid"]) <= 0
        ):
            errors.append(f"{label}.runtime_environment.server.fresh_pid: invalid")
        if (
            not _is_sha256(server.get("process_environment_sha256"))
            or server.get("process_environment_sha256")
            != document.get("effective_environment_sha256")
        ):
            errors.append(
                f"{label}.runtime_environment.server.process_environment_sha256: "
                "invalid or differs from effective environment"
            )
        if (
            server.get("python_safe_path_supported") is not False
            or server.get("python_safe_path_requested") is not False
            or server.get("python_safe_path_effective") is not False
            or server.get("python_no_user_site") is not True
            or server.get("runtime_cwd_empty") is not True
            or server.get("runtime_cwd_mode") != "0o555"
            or server.get("hook_directory_mode") != "0o555"
        ):
            errors.append(
                f"{label}.runtime_environment.server: Python/CWD isolation proof invalid"
            )
        if server.get("model_snapshot_identity_sha256") != (
            expected_model_inventory_sha256
        ):
            errors.append(
                f"{label}.runtime_environment.server: model inventory mismatch"
            )
        expected_hook = expected_provenance.get("scheduler_hook_file_sha256")
        if server.get("expected_scheduler_hook_file_sha256") != expected_hook:
            errors.append(
                f"{label}.runtime_environment.server: expected hook hash mismatch"
            )
        if not _is_sha256(server.get("expected_sitecustomize_file_sha256")):
            errors.append(
                f"{label}.runtime_environment.server: sitecustomize hash invalid"
            )
        smoke_binding = platform.get("smoke")
        smoke_document = (
            _runtime_environment_document(smoke_binding, base=base)
            if isinstance(smoke_binding, Mapping)
            else None
        )
        if not isinstance(smoke_document, Mapping) or smoke_document.get(
            "cell"
        ) != cell:
            errors.append(f"{label}.smoke: invalid or wrong-cell evidence")
        if cell in {"E", "F"}:
            if (
                server.get("loaded_scheduler_hook_file_sha256") != expected_hook
                or server.get("loaded_sitecustomize_file_sha256")
                != server.get("expected_sitecustomize_file_sha256")
                or type(server.get("scheduler_runtime_pid")) is not int
                or int(server["scheduler_runtime_pid"]) <= 0
                or not isinstance(server.get("pythonpath"), str)
                or not server.get("pythonpath")
            ):
                errors.append(
                    f"{label}.runtime_environment.server: E/F hook/runtime proof invalid"
                )
            hook_binding = platform.get("scheduler_hook_load")
            hook_document = (
                _runtime_environment_document(hook_binding, base=base)
                if isinstance(hook_binding, Mapping)
                else None
            )
            guard = (
                server.get("import_path_guard")
                if isinstance(server.get("import_path_guard"), Mapping)
                else None
            )
            hook_guard = (
                hook_document.get("import_path_guard")
                if isinstance(hook_document, Mapping)
                else None
            )
            blocked = guard.get("blocked_top_level_paths") if isinstance(guard, Mapping) else None
            effective_path = guard.get("effective_sys_path") if isinstance(guard, Mapping) else None
            if (
                not isinstance(hook_document, Mapping)
                or hook_document.get("schema")
                != "paste_gemini.loaded_scheduler_hook.v1"
                or hook_document.get("installed") is not True
                or hook_document.get("sitecustomize_file_sha256")
                != server.get("expected_sitecustomize_file_sha256")
                or hook_document.get("scheduler_hook_file_sha256") != expected_hook
                or hook_document.get("pid") != server.get("fresh_pid")
                or not isinstance(guard, Mapping)
                or guard != hook_guard
                or guard.get("installed") is not True
                or guard.get("runtime_cwd") != runtime_cwd
                or not isinstance(blocked, list)
                or "" not in blocked
                or runtime_cwd not in blocked
                or not isinstance(effective_path, list)
                or any(item in {"", runtime_cwd} for item in effective_path)
            ):
                errors.append(
                    f"{label}.scheduler_hook_load: import-path guard proof invalid"
                )
            if (
                not isinstance(smoke_document, Mapping)
                or smoke_document.get("joint_runtime_pressure_marker_seen") is not True
                or smoke_document.get("joint_runtime_import_path_guard_active") is not True
                or smoke_document.get("joint_runtime_scheduler_pid")
                != server.get("scheduler_runtime_pid")
            ):
                errors.append(
                    f"{label}.smoke: E/F EngineCore import-guard marker invalid"
                )
        elif any(
            server.get(field) is not None
            for field in (
                "loaded_scheduler_hook_file_sha256",
                "loaded_sitecustomize_file_sha256",
                "scheduler_runtime_pid",
            )
        ) or server.get("pythonpath") != "" or server.get(
            "import_path_guard"
        ) is not None:
            errors.append(
                f"{label}.runtime_environment.server: A/B unexpectedly loaded hooks"
            )
        elif isinstance(smoke_document, Mapping) and (
            smoke_document.get("joint_runtime_pressure_marker_seen") is not False
            or smoke_document.get("joint_runtime_import_path_guard_active") is not False
            or smoke_document.get("joint_runtime_scheduler_pid") is not None
        ):
            errors.append(f"{label}.smoke: A/B unexpectedly used scheduler hook")
    else:
        errors.append(f"{label}.runtime_environment.schema: unsupported {schema!r}")
    return platform


def _audit_qwen_scheduler_runtime_evidence(
    contract_value: Any,
    *,
    platform: Mapping[str, Any],
    base: Path,
    verify_files: bool,
    cell: str,
    expected_scheduler_hook_sha256: str,
    label: str,
    errors: list[str],
) -> Mapping[str, Any]:
    """Validate the real Scheduler.schedule marker retained by Qwen cells."""

    contract = _require_mapping(contract_value, label, errors)
    expected_use = cell in {"E", "F"}
    expected_policy = "online_joint_pacer_v2" if expected_use else "fcfs"
    expected_contract = {
        "hook_runtime_use_expected": expected_use,
        "patched_scheduler_invocation_verified": expected_use,
        "no_scheduler_hook_runtime_use_verified": not expected_use,
        "expected_policy": expected_policy,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            errors.append(f"{label}.{field}: expected {expected!r}")
    if not _is_sha256(contract.get("evidence_sha256")):
        errors.append(f"{label}.evidence_sha256: invalid")

    documents: dict[str, Mapping[str, Any]] = {}
    for evidence_name, phase in (
        ("scheduler_runtime_after_smoke", "after_standardized_smoke"),
        ("scheduler_runtime_after_cell", "after_evaluation_cell"),
    ):
        binding = platform.get(evidence_name)
        item_label = f"{label}.{evidence_name}"
        if not isinstance(binding, Mapping):
            errors.append(f"{item_label}: required")
            continue
        if evidence_name == "scheduler_runtime_after_smoke" and binding.get(
            "sha256"
        ) != contract.get("evidence_sha256"):
            errors.append(f"{item_label}: hash differs from result contract")
        if not verify_files:
            continue
        path_raw = binding.get("path")
        if not isinstance(path_raw, str):
            continue
        path = Path(path_raw)
        if not path.is_absolute():
            path = base / path
        try:
            document = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{item_label}: unreadable JSON: {exc}")
            continue
        if not isinstance(document, Mapping):
            errors.append(f"{item_label}: expected JSON object")
            continue
        documents[evidence_name] = document
        expected_document = {
            "schema": "paste.paper.scheduler_runtime_evidence.v1",
            "cell": cell,
            "phase": phase,
            "expected_policy": expected_policy,
            "hook_runtime_use_expected": expected_use,
            "patched_scheduler_invocation_verified": expected_use,
            "no_scheduler_hook_runtime_use_verified": not expected_use,
            "scheduler_hook_sha256": expected_scheduler_hook_sha256,
        }
        for field, expected in expected_document.items():
            if document.get(field) != expected:
                errors.append(f"{item_label}.{field}: expected {expected!r}")
        hook_path_raw = document.get("scheduler_hook_path")
        if (
            not isinstance(hook_path_raw, str)
            or not Path(hook_path_raw).is_file()
            or file_sha256(Path(hook_path_raw)) != expected_scheduler_hook_sha256
        ):
            errors.append(f"{item_label}.scheduler_hook_path: frozen bytes mismatch")
        if type(document.get("server_pid")) is not int or int(
            document["server_pid"]
        ) <= 0:
            errors.append(f"{item_label}.server_pid: invalid")
        marker = document.get("runtime_marker")
        if expected_use:
            if (
                not isinstance(marker, Mapping)
                or marker.get("schema") != "paste.vllm.scheduler_runtime_use.v1"
                or marker.get("policy") != expected_policy
                or marker.get("scheduler_api") != "v1.Scheduler.schedule"
                or marker.get("scheduler_hook_sha256")
                != expected_scheduler_hook_sha256
                or marker.get("python_safe_path_enforced") is not True
                or marker.get("cwd_import_filter_enforced") is not True
                or marker.get("working_directory_importable") is not False
                or not isinstance(marker.get("safe_working_directory"), str)
                or not marker.get("safe_working_directory")
                or marker.get("working_directory")
                != marker.get("safe_working_directory")
                or type(marker.get("pid")) is not int
                or marker.get("pid") != document.get("scheduler_calling_pid")
                or document.get("scheduler_calling_process_relation")
                != "server_descendant"
                or not _is_sha256(document.get("runtime_marker_sha256"))
            ):
                errors.append(f"{item_label}: invalid engine Scheduler.schedule marker")
        elif any(
            document.get(field) is not None
            for field in (
                "runtime_marker",
                "runtime_marker_sha256",
                "scheduler_calling_pid",
                "scheduler_calling_process_relation",
            )
        ):
            errors.append(f"{item_label}: FCFS cell contains scheduler runtime use")
    smoke = documents.get("scheduler_runtime_after_smoke")
    if smoke is not None:
        for field in (
            "hook_runtime_use_expected",
            "patched_scheduler_invocation_verified",
            "no_scheduler_hook_runtime_use_verified",
            "expected_policy",
            "scheduler_calling_pid",
            "scheduler_calling_process_relation",
            "runtime_marker_sha256",
        ):
            if contract.get(field) != smoke.get(field):
                errors.append(f"{label}.{field}: differs from after-smoke evidence")
    return contract


def _audit_near_duplicate_evidence(
    entry: Any,
    *,
    expected_root_sets_sha256: str,
    base: Path,
    verify_files: bool,
    errors: list[str],
) -> None:
    """Validate the bound, split-specific cross-split near-duplicate report."""

    label = "$.data.near_duplicate_evidence"
    evidence = _require_mapping(entry, label, errors)
    _audit_file_binding_shape(evidence, label=label, errors=errors)
    if evidence.get("schema") != NEAR_DUPLICATE_AUDIT_SCHEMA:
        errors.append(f"{label}.schema: expected {NEAR_DUPLICATE_AUDIT_SCHEMA}")
    if evidence.get("verified") is not True:
        errors.append(f"{label}.verified: must be true")
    if evidence.get("registered_root_sets_sha256") != expected_root_sets_sha256:
        errors.append(f"{label}.registered_root_sets_sha256: split identity mismatch")
    if not isinstance(evidence.get("method"), str) or not evidence.get("method"):
        errors.append(f"{label}.method: non-empty method is required")
    pairs = evidence.get("near_duplicate_pairs_across_splits")
    if pairs != []:
        errors.append(
            f"{label}.near_duplicate_pairs_across_splits: must be an explicitly empty list"
        )
    if not verify_files:
        return
    before = len(errors)
    _verify_file_binding(evidence, base=base, label=label, errors=errors)
    if len(errors) != before:
        return
    path = Path(str(evidence["path"]))
    if not path.is_absolute():
        path = base / path
    try:
        document = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: unreadable JSON: {exc}")
        return
    if not isinstance(document, Mapping):
        errors.append(f"{label}: evidence file must contain an object")
        return
    for field in (
        "schema",
        "verified",
        "registered_root_sets_sha256",
        "method",
        "near_duplicate_pairs_across_splits",
    ):
        if document.get(field) != evidence.get(field):
            errors.append(f"{label}.{field}: differs from bound evidence file")


def audit_public_plan_firewall(
    policy_bundle: Mapping[str, Any],
    *,
    base: Path,
    errors: list[str],
) -> None:
    """Audit Qwen public-plan bindings when the policy bundle exposes them."""

    path_raw = policy_bundle.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        return
    bundle_path = Path(path_raw)
    if not bundle_path.is_absolute():
        bundle_path = base / bundle_path
    if not bundle_path.is_file():
        return
    try:
        bundle = _load_json(bundle_path)
    except (OSError, json.JSONDecodeError):
        return
    plans = bundle.get("plans") if isinstance(bundle, Mapping) else None
    if not isinstance(plans, Mapping):
        return
    for role, raw_plan in plans.items():
        if not isinstance(raw_plan, Mapping) or not isinstance(
            raw_plan.get("public"), Mapping
        ):
            errors.append(f"policy bundle plan {role!r}: missing public binding")
            continue
        binding = raw_plan["public"]
        label = f"policy bundle plan {role!r} public"
        _verify_file_binding(binding, base=bundle_path.parent, label=label, errors=errors)
        public_path_raw = binding.get("path")
        if not isinstance(public_path_raw, str):
            continue
        public_path = Path(public_path_raw)
        if not public_path.is_absolute():
            public_path = bundle_path.parent / public_path
        if not public_path.is_file():
            continue
        try:
            public = _load_json(public_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: unreadable JSON: {exc}")
            continue
        if not isinstance(public, Mapping) or raw_plan.get(
            "public_plan_sha256"
        ) != public.get("plan_sha256"):
            errors.append(f"{label}: logical public-plan identity mismatch")
            continue
        leaked = sorted(
            {
                key
                for _path, node in _walk(public)
                if isinstance(node, Mapping)
                for key in (PUBLIC_PLAN_FORBIDDEN_FIELDS & set(node))
            }
        )
        if leaked:
            errors.append(f"{label}: future-authority fields exposed: {leaked}")


def expected_strict_result_schema(
    manifest: Mapping[str, Any], *, base: Path
) -> str:
    """Resolve the only result schema authorized by the frozen policy bundle."""

    frozen = manifest.get("frozen_files")
    if not isinstance(frozen, list):
        raise ValueError("manifest lacks frozen_files")
    rows = [
        row
        for row in frozen
        if isinstance(row, Mapping) and row.get("role") == "policy_bundle"
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("path"), str):
        raise ValueError("manifest must bind exactly one policy_bundle")
    path = Path(str(rows[0]["path"]))
    if not path.is_absolute():
        path = base / path
    if not path.is_file() or file_sha256(path) != rows[0].get("sha256"):
        raise ValueError("frozen policy_bundle bytes are missing or changed")
    bundle = _load_json(path)
    if not isinstance(bundle, Mapping):
        raise ValueError("frozen policy_bundle must contain a JSON object")
    result_schema = POLICY_BUNDLE_RESULT_SCHEMAS.get(str(bundle.get("schema")))
    if result_schema is None:
        raise ValueError(
            f"unsupported frozen policy_bundle schema {bundle.get('schema')!r}"
        )
    return result_schema


def model_snapshot_inventory_sha256(
    manifest: Mapping[str, Any], *, base: Path
) -> str | None:
    """Return and verify a bound repo's signed full-model inventory identity."""

    frozen = manifest.get("frozen_files")
    if not isinstance(frozen, list):
        return None
    rows = [
        row
        for row in frozen
        if isinstance(row, Mapping) and row.get("role") == "policy_bundle"
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("path"), str):
        return None
    path = Path(str(rows[0]["path"]))
    if not path.is_absolute():
        path = base / path
    if (
        not path.is_file()
        or file_sha256(path) != rows[0].get("sha256")
    ):
        return None
    bundle = _load_json(path)
    if not isinstance(bundle, Mapping):
        return None
    contract = bundle.get("model_snapshot_contract")
    if contract is None:
        # Generic fixtures and older non-formal policy files are not silently
        # promoted to model-byte-attested evidence.
        if isinstance(bundle.get("plans"), Mapping) or (
            isinstance(bundle.get("sessions"), list)
            and isinstance(bundle.get("templates"), list)
        ):
            raise ValueError("strict policy bundle lacks model_snapshot_contract")
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("policy bundle model_snapshot_contract is malformed")

    # Qwen stores a compact content inventory inside a surrounding model
    # contract.  Its identity deliberately excludes count aliases.
    inventory = contract.get("inventory")
    identity = contract.get("inventory_sha256")
    if isinstance(inventory, Mapping):
        if not _is_sha256(identity):
            raise ValueError("Qwen model snapshot inventory binding is invalid")
        if inventory.get("inventory_sha256") != identity:
            raise ValueError("Qwen model snapshot inventory aliases disagree")
        schema = inventory.get("schema")
        files = inventory.get("files")
        if schema != "paste_repro.model_snapshot_inventory.v1" or not isinstance(
            files, list
        ) or not files:
            raise ValueError("Qwen model snapshot inventory payload is malformed")
        relative_paths: set[str] = set()
        normalized_files: list[dict[str, Any]] = []
        for index, raw in enumerate(files):
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Qwen model snapshot inventory file {index} is malformed"
                )
            relative = raw.get("relative_path")
            size = raw.get("size_bytes")
            digest = raw.get("content_sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or relative in relative_paths
                or type(size) is not int
                or size < 0
                or not _is_sha256(digest)
            ):
                raise ValueError(
                    f"Qwen model snapshot inventory file {index} is invalid"
                )
            relative_paths.add(relative)
            normalized_files.append(
                {
                    "relative_path": relative,
                    "size_bytes": size,
                    "content_sha256": digest,
                }
            )
        if normalized_files != sorted(
            normalized_files, key=lambda row: str(row["relative_path"])
        ):
            raise ValueError("Qwen model snapshot inventory is not canonically ordered")
        if canonical_sha256({"schema": schema, "files": normalized_files}) != identity:
            raise ValueError("Qwen model snapshot inventory canonical hash mismatch")
        if inventory.get("file_count") != len(normalized_files) or inventory.get(
            "total_size_bytes"
        ) != sum(int(row["size_bytes"]) for row in normalized_files):
            raise ValueError("Qwen model snapshot inventory counts disagree")
        return str(identity)

    # Gemini stores the complete self-signed inventory directly as the model
    # contract.  It also records the symlink spelling and resolved content
    # path, while the content SHA remains the authoritative byte identity.
    if contract.get("schema") != "paste.paper.model_snapshot_inventory.v1":
        raise ValueError("unsupported policy model snapshot inventory schema")
    identity = contract.get("identity_sha256")
    files = contract.get("files")
    if not _is_sha256(identity) or not isinstance(files, list) or not files:
        raise ValueError("Gemini model snapshot inventory binding is invalid")
    unsigned = dict(contract)
    unsigned.pop("identity_sha256", None)
    encodings = (
        json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        json.dumps(
            unsigned,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    if str(identity) not in {
        hashlib.sha256(encoded).hexdigest() for encoded in encodings
    }:
        raise ValueError("Gemini model snapshot inventory canonical hash mismatch")
    relative_paths: set[str] = set()
    for index, raw in enumerate(files):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Gemini model snapshot inventory file {index} is malformed"
            )
        relative = raw.get("relative_path")
        resolved = raw.get("resolved_path")
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        symlink_target = raw.get("symlink_target")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in relative_paths
            or not isinstance(resolved, str)
            or not Path(resolved).is_absolute()
            or type(size) is not int
            or size < 0
            or not _is_sha256(digest)
            or (symlink_target is not None and not isinstance(symlink_target, str))
        ):
            raise ValueError(
                f"Gemini model snapshot inventory file {index} is invalid"
            )
        relative_paths.add(relative)
    if [str(row["relative_path"]) for row in files] != sorted(relative_paths):
        raise ValueError("Gemini model snapshot inventory is not canonically ordered")
    if contract.get("file_count") != len(files) or contract.get(
        "total_bytes"
    ) != sum(int(row["size_bytes"]) for row in files):
        raise ValueError("Gemini model snapshot inventory counts disagree")
    return str(identity)


def qwen_model_snapshot_inventory_sha256(
    manifest: Mapping[str, Any], *, base: Path
) -> str | None:
    """Compatibility alias for callers predating cross-repository support."""

    return model_snapshot_inventory_sha256(manifest, base=base)


def _audit_runtime_environment_contract(
    value: Any,
    *,
    expected_model_inventory_sha256: str,
    label: str,
    errors: list[str],
) -> Mapping[str, Any]:
    contract = _require_mapping(value, label, errors)
    if not _is_sha256(contract.get("evidence_sha256")):
        errors.append(f"{label}.evidence_sha256: invalid")
    if contract.get("model_snapshot_inventory_sha256") != (
        expected_model_inventory_sha256
    ):
        errors.append(
            f"{label}.model_snapshot_inventory_sha256: differs from frozen policy bundle"
        )
    for field in (
        "environment_scrubbed_before_cell",
        "server_and_client_launched_via_env_i",
    ):
        if contract.get(field) is not True:
            errors.append(f"{label}.{field}: must be true")
    return contract


def audit_manifest(
    payload: Any,
    *,
    base: Path,
    verify_files: bool,
    require_evidence: bool,
) -> dict[str, Any]:
    """Validate preregistration, causal boundaries, matrix, and optional evidence."""

    errors: list[str] = []
    warnings: list[str] = []
    root = _require_mapping(payload, "$", errors)
    if root.get("schema") != MANIFEST_SCHEMA:
        errors.append(f"$.schema: expected {MANIFEST_SCHEMA}")
    if root.get("version") != 1:
        errors.append("$.version: expected 1")
    claim_scope = root.get("claim_scope")
    if claim_scope not in {"retrospective", "confirmatory"}:
        errors.append("$.claim_scope: expected retrospective or confirmatory")

    data = _require_mapping(root.get("data"), "$.data", errors)
    calibration = _require_unique_strings(
        data.get("calibration_root_ids"), "$.data.calibration_root_ids", errors
    )
    tuning = _require_unique_strings(
        data.get("tuning_root_ids"),
        "$.data.tuning_root_ids",
        errors,
        allow_empty=True,
    )
    evaluation = _require_unique_strings(
        data.get("evaluation_root_ids"), "$.data.evaluation_root_ids", errors
    )
    exposed_raw = data.get("previously_observed_evaluation_root_ids")
    if not isinstance(exposed_raw, list) or any(
        not isinstance(item, str) or not item for item in exposed_raw
    ):
        errors.append(
            "$.data.previously_observed_evaluation_root_ids: expected string list"
        )
        exposed: list[str] = []
    else:
        exposed = list(exposed_raw)
    for left_name, left, right_name, right in (
        ("calibration", calibration, "tuning", tuning),
        ("calibration", calibration, "evaluation", evaluation),
        ("tuning", tuning, "evaluation", evaluation),
    ):
        overlap = sorted(set(left) & set(right))
        if overlap:
            errors.append(
                f"$.data: {left_name}/{right_name} root overlap: {overlap[:5]}"
            )
    if not set(exposed).issubset(set(evaluation)):
        errors.append("$.data: previously observed IDs are not evaluation IDs")
    if data.get("split_unit") != "root_trace_or_task":
        errors.append("$.data.split_unit: must be root_trace_or_task")
    if data.get("exact_root_disjoint_guard") is not True:
        errors.append("$.data.exact_root_disjoint_guard: must be true")
    duplicate_status = data.get("near_duplicate_guard")
    if duplicate_status == "verified":
        _audit_near_duplicate_evidence(
            data.get("near_duplicate_evidence"),
            expected_root_sets_sha256=registered_root_sets_sha256(
                calibration, tuning, evaluation
            ),
            base=base,
            verify_files=verify_files,
            errors=errors,
        )
    elif duplicate_status == "not_verified":
        if data.get("near_duplicate_evidence") is not None:
            errors.append(
                "$.data.near_duplicate_evidence: must be absent when guard is not_verified"
            )
        warnings.append(
            "cross-split near-duplicate exclusion was not independently verified; "
            "only exact root-ID disjointness is established"
        )
        if claim_scope == "confirmatory":
            errors.append(
                "$.data.near_duplicate_guard: confirmatory scope requires verified "
                "bound near-duplicate evidence"
            )
    else:
        errors.append(
            "$.data.near_duplicate_guard: expected verified or not_verified"
        )
    selection_protocol = data.get("selection_protocol")
    if tuning:
        if selection_protocol != "heldout_tuning_split":
            errors.append(
                "$.data.selection_protocol: non-empty tuning requires "
                "heldout_tuning_split"
            )
    elif selection_protocol != "nested_cross_validation_within_calibration":
        errors.append(
            "$.data.selection_protocol: empty tuning requires "
            "nested_cross_validation_within_calibration"
        )
    if data.get("evaluation_used_for_model_or_policy_selection") is not False:
        errors.append(
            "$.data.evaluation_used_for_model_or_policy_selection: must be false"
        )
    if claim_scope == "confirmatory" and len(evaluation) < MIN_CONFIRMATORY_ROOTS:
        errors.append(
            "$.data.evaluation_root_ids: confirmatory paper evidence requires "
            f"at least {MIN_CONFIRMATORY_ROOTS} independent roots"
        )
    confirmatory_eligible = not exposed and not errors
    if claim_scope == "confirmatory" and exposed:
        errors.append(
            "$.claim_scope: confirmatory is impossible with previously observed evaluation roots"
        )
        confirmatory_eligible = False
    if claim_scope == "retrospective":
        confirmatory_eligible = False
        warnings.append(
            "retrospective scope is oracle-free only; it is not untouched confirmatory evidence"
        )

    freeze = _require_mapping(root.get("freeze"), "$.freeze", errors)
    for field in (
        "sealed_before_evaluation",
        "no_tuning_after_seal",
        "accept_result_regardless_of_direction",
        "started_marker_exclusive_create",
    ):
        if freeze.get(field) is not True:
            errors.append(f"$.freeze.{field}: must be true")
    if freeze.get("formal_result_used_for_optimization") is not False:
        errors.append("$.freeze.formal_result_used_for_optimization: must be false")
    if not _is_sha256(freeze.get("policy_bundle_sha256")):
        errors.append("$.freeze.policy_bundle_sha256: invalid")
    expected_seal_hash = sealed_payload_sha256(root)
    if freeze.get("sealed_payload_sha256") != expected_seal_hash:
        errors.append("$.freeze.sealed_payload_sha256: preregistration hash mismatch")
    marker_path_raw = freeze.get("started_marker_path")
    if not isinstance(marker_path_raw, str) or not marker_path_raw:
        errors.append("$.freeze.started_marker_path: missing")
    elif verify_files:
        marker_path = Path(marker_path_raw)
        if not marker_path.is_absolute():
            marker_path = base / marker_path
        if not marker_path.is_file():
            errors.append(f"$.freeze.started_marker_path: missing file: {marker_path}")
        else:
            try:
                marker = _load_json(marker_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"$.freeze.started_marker_path: unreadable marker: {exc}")
            else:
                if not isinstance(marker, Mapping) or marker.get("schema") != START_MARKER_SCHEMA:
                    errors.append("$.freeze.started_marker_path: wrong marker schema")
                elif marker.get("sealed_payload_sha256") != expected_seal_hash:
                    errors.append("$.freeze.started_marker_path: marker seal mismatch")
    sealed_at = freeze.get("sealed_at_utc")
    if not isinstance(sealed_at, str):
        errors.append("$.freeze.sealed_at_utc: missing")
    else:
        try:
            parsed = datetime.fromisoformat(sealed_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError
        except ValueError:
            errors.append("$.freeze.sealed_at_utc: must be timezone-aware ISO-8601")

    expected_result_schema: str | None = None
    frozen_files = root.get("frozen_files")
    if not isinstance(frozen_files, list) or len(frozen_files) < len(
        REQUIRED_FROZEN_FILE_ROLES
    ):
        errors.append(
            "$.frozen_files: bind protocol, runner, policy bundle, config, "
            "scheduler hook, materializer, auditor, and analyzer"
        )
    else:
        roles = [entry.get("role") for entry in frozen_files if isinstance(entry, Mapping)]
        if len(roles) != len(set(roles)):
            errors.append("$.frozen_files: duplicate roles")
        missing_roles = sorted(REQUIRED_FROZEN_FILE_ROLES - set(roles))
        if missing_roles:
            errors.append(f"$.frozen_files: missing required roles: {missing_roles}")
        for index, entry in enumerate(frozen_files):
            mapping = _require_mapping(entry, f"$.frozen_files[{index}]", errors)
            if not isinstance(mapping.get("role"), str) or not mapping.get("role"):
                errors.append(f"$.frozen_files[{index}].role: missing")
            _audit_file_binding_shape(
                mapping, label=f"$.frozen_files[{index}]", errors=errors
            )
            if verify_files:
                _verify_file_binding(
                    mapping,
                    base=base,
                    label=f"$.frozen_files[{index}]",
                    errors=errors,
                )
        if verify_files:
            policy_bundle_rows = [
                row
                for row in frozen_files
                if isinstance(row, Mapping) and row.get("role") == "policy_bundle"
            ]
            if len(policy_bundle_rows) == 1:
                audit_public_plan_firewall(
                    policy_bundle_rows[0], base=base, errors=errors
                )
            try:
                expected_result_schema = expected_strict_result_schema(
                    root, base=base
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"$.frozen_files.policy_bundle: {exc}")

    preregistered = root.get("preregistered_manifest")
    if preregistered is not None:
        preregistered_mapping = _require_mapping(
            preregistered, "$.preregistered_manifest", errors
        )
        _audit_file_binding_shape(
            preregistered_mapping, label="$.preregistered_manifest", errors=errors
        )
        if verify_files:
            _verify_file_binding(
                preregistered_mapping,
                base=base,
                label="$.preregistered_manifest",
                errors=errors,
            )
            path_raw = preregistered_mapping.get("path")
            if isinstance(path_raw, str):
                registered_path = Path(path_raw)
                if not registered_path.is_absolute():
                    registered_path = base / registered_path
                if registered_path.is_file():
                    registered = _load_json(registered_path)
                    if not isinstance(registered, Mapping) or sealed_payload_sha256(
                        registered
                    ) != expected_seal_hash:
                        errors.append(
                            "$.preregistered_manifest: sealed content differs from completed manifest"
                        )

    predictors = _require_mapping(root.get("predictors"), "$.predictors", errors)
    for name in ("tool_invocation", "tool_duration"):
        predictor = _require_mapping(
            predictors.get(name), f"$.predictors.{name}", errors
        )
        if predictor.get("training_role") != "calibration":
            errors.append(f"$.predictors.{name}.training_role: must be calibration")
        if predictor.get("uses_evaluation_labels") is not False:
            errors.append(f"$.predictors.{name}.uses_evaluation_labels: must be false")
        if predictor.get("frozen_before_evaluation") is not True:
            errors.append(f"$.predictors.{name}.frozen_before_evaluation: must be true")
        expected_training_hash = canonical_sha256(sorted(calibration))
        if predictor.get("training_root_ids_sha256") != expected_training_hash:
            errors.append(
                f"$.predictors.{name}.training_root_ids_sha256: "
                "does not bind the calibration roots"
            )
        features = predictor.get("input_features")
        if not isinstance(features, list) or not features:
            errors.append(f"$.predictors.{name}.input_features: expected non-empty list")
        else:
            feature_set = set(features)
            unknown = sorted(feature_set - ALLOWED_DECISION_FEATURES)
            forbidden = sorted(feature_set & FORBIDDEN_DECISION_FEATURES)
            if unknown:
                errors.append(f"$.predictors.{name}: unregistered features: {unknown}")
            if forbidden:
                errors.append(f"$.predictors.{name}: forbidden features: {forbidden}")
        artifact = _require_mapping(
            predictor.get("artifact"), f"$.predictors.{name}.artifact", errors
        )
        _audit_file_binding_shape(
            artifact, label=f"$.predictors.{name}.artifact", errors=errors
        )
        if not _is_sha256(artifact.get("identity_sha256")):
            errors.append(
                f"$.predictors.{name}.artifact.identity_sha256: "
                "signed logical artifact identity is required"
            )
        if artifact.get("input_features") != features:
            errors.append(
                f"$.predictors.{name}.artifact.input_features: must exactly match "
                "the registered predictor input_features"
            )
        if artifact.get("training_root_ids_sha256") != expected_training_hash:
            errors.append(
                f"$.predictors.{name}.artifact.training_root_ids_sha256: "
                "does not bind calibration roots"
            )
        if artifact.get("uses_evaluation_labels") is not False:
            errors.append(
                f"$.predictors.{name}.artifact.uses_evaluation_labels: must be false"
            )
        if "fit_code_sha256" in artifact and not _is_sha256(
            artifact.get("fit_code_sha256")
        ):
            errors.append(f"$.predictors.{name}.artifact.fit_code_sha256: invalid")
        if verify_files:
            _verify_file_binding(
                artifact,
                base=base,
                label=f"$.predictors.{name}.artifact",
                errors=errors,
            )
            _verify_embedded_identity(
                artifact,
                base=base,
                label=f"$.predictors.{name}.artifact",
                fields=("artifact_sha256",),
                errors=errors,
            )
            _verify_embedded_contract(
                artifact,
                base=base,
                label=f"$.predictors.{name}.artifact",
                predictor=True,
                errors=errors,
            )

    policy = _require_mapping(root.get("policy"), "$.policy", errors)
    if policy.get("decision_schema") != DECISION_SCHEMA:
        errors.append(f"$.policy.decision_schema: expected {DECISION_SCHEMA}")
    declared_forbidden = policy.get("forbidden_decision_features")
    if not isinstance(declared_forbidden, list) or not FORBIDDEN_DECISION_FEATURES.issubset(
        set(declared_forbidden)
    ):
        errors.append("$.policy.forbidden_decision_features: incomplete deny-list")
    cells = _require_mapping(policy.get("cells"), "$.policy.cells", errors)
    if set(cells) != set(CELLS):
        errors.append("$.policy.cells: must contain exactly A/B/E/F")
    else:
        for cell, expected in CELLS.items():
            if cells.get(cell) != expected:
                errors.append(f"$.policy.cells.{cell}: treatment mismatch")
    call_graph_mode = policy.get("call_graph_mode")
    if call_graph_mode not in CALL_GRAPH_MODES:
        errors.append(
            "$.policy.call_graph_mode: expected autonomous or "
            "trace_replay_causal_reveal"
        )
    expected_claim_type = {
        "autonomous": "closed_loop_agent",
        "trace_replay_causal_reveal": "systems_trace_replay",
    }.get(call_graph_mode)
    if policy.get("claim_type") != expected_claim_type:
        errors.append("$.policy.claim_type: inconsistent with call graph mode")
    if policy.get("offline_tool_credit_s") != 0:
        errors.append("$.policy.offline_tool_credit_s: must be exactly zero")

    execution = _require_mapping(root.get("execution"), "$.execution", errors)
    for field in REQUIRED_EXECUTION_ATTESTATIONS:
        if execution.get(field) is not True:
            errors.append(f"$.execution.{field}: must be true")
    runtime_parameters = _audit_runtime_parameters_contract(
        execution.get("treatment_neutral_runtime_parameters"),
        label="$.execution.treatment_neutral_runtime_parameters",
        require_artifact=True,
        errors=errors,
    )
    runtime_artifact = runtime_parameters.get("artifact")
    if isinstance(runtime_artifact, Mapping) and verify_files:
        runtime_label = "$.execution.treatment_neutral_runtime_parameters.artifact"
        before = len(errors)
        _verify_file_binding(
            runtime_artifact, base=base, label=runtime_label, errors=errors
        )
        if len(errors) == before:
            runtime_path = Path(str(runtime_artifact["path"]))
            if not runtime_path.is_absolute():
                runtime_path = base / runtime_path
            try:
                embedded_runtime = _load_json(runtime_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{runtime_label}: unreadable JSON: {exc}")
            else:
                expected_embedded = {
                    key: runtime_parameters.get(key)
                    for key in (
                        "schema",
                        "parameters",
                        "runtime_parameters_sha256",
                    )
                }
                if embedded_runtime != expected_embedded:
                    errors.append(
                        f"{runtime_label}: file differs from embedded runtime contract"
                    )
    service_clock = _require_mapping(
        execution.get("physical_service_clock"),
        "$.execution.physical_service_clock",
        errors,
    )
    clock_mode = service_clock.get("mode")
    if clock_mode not in PHYSICAL_SERVICE_CLOCK_MODES:
        errors.append("$.execution.physical_service_clock.mode: unsupported")
    exact_clock_contract = {
        "assignment_key": "normalized_tool_arguments_environment_version",
        "policy_can_read_realization": False,
        "policy_can_reconstruct_realization": False,
        "uses_evaluation_trace_durations": False,
        "future_authority_hit_invariant": True,
        "future_state_accepted_invariant": True,
        "same_key_same_service_all_cells": True,
    }
    for field, expected in exact_clock_contract.items():
        if service_clock.get(field) != expected:
            errors.append(
                f"$.execution.physical_service_clock.{field}: expected {expected!r}"
            )
    if clock_mode == "calibration_hashed_empirical_v1":
        if service_clock.get("training_role") != "calibration":
            errors.append(
                "$.execution.physical_service_clock.training_role: must be calibration"
            )
        if service_clock.get("uses_evaluation_labels") is not False:
            errors.append(
                "$.execution.physical_service_clock.uses_evaluation_labels: must be false"
            )
        if service_clock.get("training_root_ids_sha256") != canonical_sha256(
            sorted(calibration)
        ):
            errors.append(
                "$.execution.physical_service_clock.training_root_ids_sha256: "
                "does not bind calibration roots"
            )
    service_artifact = _require_mapping(
        service_clock.get("artifact"),
        "$.execution.physical_service_clock.artifact",
        errors,
    )
    _audit_file_binding_shape(
        service_artifact,
        label="$.execution.physical_service_clock.artifact",
        errors=errors,
    )
    if not _is_sha256(service_artifact.get("identity_sha256")):
        errors.append(
            "$.execution.physical_service_clock.artifact.identity_sha256: "
            "signed logical service identity is required"
        )
    expected_training_hash = canonical_sha256(sorted(calibration))
    if service_artifact.get("training_root_ids_sha256") != expected_training_hash:
        errors.append(
            "$.execution.physical_service_clock.artifact.training_root_ids_sha256: "
            "does not bind calibration roots"
        )
    if service_artifact.get("uses_evaluation_labels") is not False:
        errors.append(
            "$.execution.physical_service_clock.artifact.uses_evaluation_labels: "
            "must be false"
        )
    if service_artifact.get("future_state_accepted_invariant") is not True:
        errors.append(
            "$.execution.physical_service_clock.artifact."
            "future_state_accepted_invariant: must be true"
        )
    if (
        "uses_evaluation_trace_durations" in service_artifact
        and service_artifact.get("uses_evaluation_trace_durations") is not False
    ):
        errors.append(
            "$.execution.physical_service_clock.artifact."
            "uses_evaluation_trace_durations: must be false"
        )
    if verify_files:
        _verify_file_binding(
            service_artifact,
            base=base,
            label="$.execution.physical_service_clock.artifact",
            errors=errors,
        )
        _verify_embedded_identity(
            service_artifact,
            base=base,
            label="$.execution.physical_service_clock.artifact",
            fields=(
                "service_clock_artifact_sha256",
                "tool_service_surface_sha256",
                "executor_service_surface_sha256",
                "service_surface_sha256",
                "artifact_sha256",
            ),
            errors=errors,
        )
        _verify_embedded_contract(
            service_artifact,
            base=base,
            label="$.execution.physical_service_clock.artifact",
            predictor=False,
            errors=errors,
        )
    gemini_legacy_compatibility = _audit_gemini_legacy_compatibility_manifest(
        root,
        base=base,
        verify_files=verify_files,
        errors=errors,
    )
    duration_artifact = predictors.get("tool_duration", {})
    if isinstance(duration_artifact, Mapping):
        duration_artifact = duration_artifact.get("artifact", {})
    if (
        isinstance(duration_artifact, Mapping)
        and _is_sha256(duration_artifact.get("sha256"))
        and artifact_identity_sha256(duration_artifact)
        == artifact_identity_sha256(service_artifact)
    ):
        errors.append(
            "$.execution.physical_service_clock.artifact: must be distinct from "
            "the policy duration-predictor artifact"
        )
    if isinstance(frozen_files, list):
        expected_policy_bundle_hash = policy_bundle_sha256(
            frozen_files=[
                row for row in frozen_files if isinstance(row, Mapping)
            ],
            predictors=predictors,
            physical_service_clock=service_clock,
            policy=policy,
            treatment_neutral_runtime_parameters=runtime_parameters,
        )
        if freeze.get("policy_bundle_sha256") != expected_policy_bundle_hash:
            errors.append(
                "$.freeze.policy_bundle_sha256: does not bind frozen files, "
                "predictors, service clock, and policy"
            )
    blocks_raw = execution.get("blocks")
    block_by_id: dict[str, Mapping[str, Any]] = {}
    if not isinstance(blocks_raw, list) or len(blocks_raw) < 4 or len(blocks_raw) % 4:
        errors.append("$.execution.blocks: require complete four-block Williams cycles")
    else:
        orders: Counter[tuple[str, ...]] = Counter()
        gpu_groups: Counter[tuple[int, ...]] = Counter()
        for index, raw in enumerate(blocks_raw):
            block = _require_mapping(raw, f"$.execution.blocks[{index}]", errors)
            block_id = block.get("block_id")
            if not isinstance(block_id, str) or not block_id or block_id in block_by_id:
                errors.append(f"$.execution.blocks[{index}].block_id: invalid or duplicate")
            else:
                block_by_id[block_id] = block
            order_raw = block.get("order")
            order = tuple(order_raw) if isinstance(order_raw, list) else ()
            if order not in WILLIAMS_ORDERS:
                errors.append(f"$.execution.blocks[{index}].order: not preregistered Williams order")
            else:
                orders[order] += 1
            gpu_raw = block.get("gpu_ids")
            if (
                not isinstance(gpu_raw, list)
                or not gpu_raw
                or any(type(item) is not int or item < 0 for item in gpu_raw)
                or len(set(gpu_raw)) != len(gpu_raw)
            ):
                errors.append(f"$.execution.blocks[{index}].gpu_ids: invalid")
            else:
                gpu_groups[tuple(gpu_raw)] += 1
        if orders and (
            set(orders) != set(WILLIAMS_ORDERS)
            or len(set(orders.values())) != 1
        ):
            errors.append("$.execution.blocks: Williams orders are not equally replicated")
        if gpu_groups and (
            len(gpu_groups) < 2 or max(gpu_groups.values()) - min(gpu_groups.values()) > 1
        ):
            errors.append("$.execution.blocks: GPU groups must be swapped and balanced")

    stats = _require_mapping(root.get("statistics"), "$.statistics", errors)
    exact_stats = {
        "unit": "root_trace_or_task",
        "replicas_folded_within_root": True,
        "blocks_folded_within_root": True,
        "paired": True,
        "confidence_interval_method": "paired_root_cluster_percentile_bootstrap",
        "ci_level": 0.95,
        "primary_contrast": "A_vs_F",
        "estimand": "ratio_of_paired_root_mean_e2e",
        "speedup_threshold": 0.20,
        "pass_rule": "point_estimate_ge_0.20_and_ci_lower_gt_0",
        "strong_claim_requires_ci_lower_ge_0.20": True,
        "report_all_factorial_contrasts": True,
    }
    for field, expected in exact_stats.items():
        if stats.get(field) != expected:
            errors.append(f"$.statistics.{field}: expected {expected!r}")
    resamples = stats.get("paired_bootstrap_resamples")
    if type(resamples) is not int or resamples < 10_000:
        errors.append("$.statistics.paired_bootstrap_resamples: require at least 10000")
    if not isinstance(stats.get("paired_bootstrap_seed"), str) or not stats.get(
        "paired_bootstrap_seed"
    ):
        errors.append("$.statistics.paired_bootstrap_seed: must be preregistered")

    evidence_raw = root.get("cell_evidence")
    if require_evidence and not isinstance(evidence_raw, list):
        errors.append("$.cell_evidence: required for post-run audit")
    if isinstance(evidence_raw, list):
        expected_model_inventory: str | None = None
        if verify_files:
            try:
                expected_model_inventory = model_snapshot_inventory_sha256(
                    root, base=base
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"$.frozen_files.policy_bundle: {exc}")
        expected_count = len(block_by_id) * len(CELLS)
        if len(evidence_raw) != expected_count:
            errors.append(f"$.cell_evidence: expected {expected_count} rows")
        seen_pairs: set[tuple[str, str]] = set()
        server_ids: set[str] = set()
        broker_ids: set[str] = set()
        evidence_intervals: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
        expected_provenance = expected_runtime_provenance(root)
        expected_runtime_result = {
            key: copy.deepcopy(runtime_parameters.get(key))
            for key in ("schema", "parameters", "runtime_parameters_sha256")
        }
        for index, raw in enumerate(evidence_raw):
            evidence = _require_mapping(raw, f"$.cell_evidence[{index}]", errors)
            block_id = evidence.get("block_id")
            cell = evidence.get("cell")
            pair = (str(block_id), str(cell))
            if block_id not in block_by_id or cell not in CELLS or pair in seen_pairs:
                errors.append(f"$.cell_evidence[{index}]: invalid/duplicate block-cell")
            seen_pairs.add(pair)
            if block_id in block_by_id and evidence.get("gpu_ids") != block_by_id[block_id].get("gpu_ids"):
                errors.append(f"$.cell_evidence[{index}].gpu_ids: differs from block")
            if block_id in block_by_id and cell in CELLS:
                expected_position = list(block_by_id[block_id]["order"]).index(cell) + 1
                if evidence.get("order_position") != expected_position:
                    errors.append(
                        f"$.cell_evidence[{index}].order_position: expected {expected_position}"
                    )
            for field, seen in (
                ("server_instance_id", server_ids),
                ("broker_instance_id", broker_ids),
            ):
                value = evidence.get(field)
                if not isinstance(value, str) or not value or value in seen:
                    errors.append(f"$.cell_evidence[{index}].{field}: missing or reused")
                else:
                    seen.add(value)
            if evidence.get("policy_bundle_sha256") != freeze.get("policy_bundle_sha256"):
                errors.append(f"$.cell_evidence[{index}]: policy bundle mismatch")
            if evidence.get("service_clock_artifact_sha256") != artifact_identity_sha256(
                service_artifact
            ):
                errors.append(
                    f"$.cell_evidence[{index}]: physical service clock mismatch"
                )
            if evidence.get("runtime_parameters_sha256") != runtime_parameters.get(
                "runtime_parameters_sha256"
            ):
                errors.append(
                    f"$.cell_evidence[{index}]: treatment-neutral runtime mismatch"
                )
            provenance = _audit_runtime_provenance_shape(
                evidence.get("provenance"),
                label=f"$.cell_evidence[{index}].provenance",
                errors=errors,
                extra_fields=(
                    GEMINI_LEGACY_COMPATIBILITY_PROVENANCE_FIELDS
                    if gemini_legacy_compatibility is not None
                    else ()
                ),
            )
            evidence_environment: Mapping[str, Any] | None = None
            evidence_scheduler: Mapping[str, Any] | None = None
            if expected_model_inventory is not None:
                if (
                    provenance.get("model_snapshot_inventory_sha256")
                    != expected_model_inventory
                ):
                    errors.append(
                        f"$.cell_evidence[{index}].provenance."
                        "model_snapshot_inventory_sha256: differs from policy bundle"
                    )
                evidence_environment = _audit_runtime_environment_contract(
                    evidence.get("runtime_environment_contract"),
                    expected_model_inventory_sha256=expected_model_inventory,
                    label=f"$.cell_evidence[{index}].runtime_environment_contract",
                    errors=errors,
                )
                environment_binding = _require_mapping(
                    evidence.get("runtime_environment_evidence"),
                    f"$.cell_evidence[{index}].runtime_environment_evidence",
                    errors,
                )
                _audit_file_binding_shape(
                    environment_binding,
                    label=(
                        f"$.cell_evidence[{index}].runtime_environment_evidence"
                    ),
                    errors=errors,
                )
                if environment_binding.get("sha256") != evidence_environment.get(
                    "evidence_sha256"
                ):
                    errors.append(
                        f"$.cell_evidence[{index}].runtime_environment_evidence: "
                        "hash differs from runtime environment contract"
                    )
                if verify_files:
                    _verify_file_binding(
                        environment_binding,
                        base=base,
                        label=(
                            f"$.cell_evidence[{index}].runtime_environment_evidence"
                        ),
                        errors=errors,
                    )
                platform = _audit_platform_evidence(
                    evidence.get("platform_evidence"),
                    base=base,
                    verify_files=verify_files,
                    expected_model_inventory_sha256=expected_model_inventory,
                    runtime_environment_contract=evidence_environment,
                    cell=str(cell),
                    block_id=str(block_id),
                    order_position=(
                        int(evidence.get("order_position"))
                        if type(evidence.get("order_position")) is int
                        else -1
                    ),
                    expected_provenance=expected_provenance,
                    gpu_ids=(
                        list(evidence.get("gpu_ids"))
                        if isinstance(evidence.get("gpu_ids"), list)
                        else []
                    ),
                    server_instance_id=str(evidence.get("server_instance_id", "")),
                    label=f"$.cell_evidence[{index}].platform_evidence",
                    errors=errors,
                )
                platform_environment = platform.get("runtime_environment")
                if (
                    isinstance(platform_environment, Mapping)
                    and dict(platform_environment) != dict(environment_binding)
                ):
                    errors.append(
                        f"$.cell_evidence[{index}].platform_evidence."
                        "runtime_environment: differs from compatibility binding"
                    )
                if (
                    "scheduler_runtime_after_smoke" in platform
                    or "scheduler_runtime_after_cell" in platform
                ) and evidence.get("scheduler_runtime_contract") is None:
                    errors.append(
                        f"$.cell_evidence[{index}].scheduler_runtime_contract: "
                        "required for Qwen scheduler evidence"
                    )
                if evidence.get("scheduler_runtime_contract") is not None:
                    evidence_scheduler = _audit_qwen_scheduler_runtime_evidence(
                        evidence.get("scheduler_runtime_contract"),
                        platform=platform,
                        base=base,
                        verify_files=verify_files,
                        cell=str(cell),
                        expected_scheduler_hook_sha256=str(
                            expected_provenance.get("scheduler_hook_file_sha256", "")
                        ),
                        label=(
                            f"$.cell_evidence[{index}].scheduler_runtime_contract"
                        ),
                        errors=errors,
                    )
            for field, expected_value in expected_provenance.items():
                if provenance.get(field) != expected_value:
                    errors.append(
                        f"$.cell_evidence[{index}].provenance.{field}: "
                        "differs from sealed manifest"
                    )
            started_wall = evidence.get("started_wall_s")
            ended_wall = evidence.get("ended_wall_s")
            if (
                not _is_number(started_wall)
                or not _is_number(ended_wall)
                or float(started_wall) < 0.0
                or float(ended_wall) < float(started_wall)
            ):
                errors.append(
                    f"$.cell_evidence[{index}]: invalid wall-clock interval"
                )
            elif block_id in block_by_id and cell in CELLS:
                position = list(block_by_id[block_id]["order"]).index(cell) + 1
                evidence_intervals[str(block_id)][position] = (
                    float(started_wall),
                    float(ended_wall),
                )
            result_binding = {
                "path": evidence.get("result_path"),
                "sha256": evidence.get("result_sha256"),
            }
            if verify_files:
                before = len(errors)
                _verify_file_binding(
                    result_binding,
                    base=base,
                    label=f"$.cell_evidence[{index}].result",
                    errors=errors,
                )
                if len(errors) == before:
                    result_path = Path(str(evidence["result_path"]))
                    if not result_path.is_absolute():
                        result_path = base / result_path
                    result = _load_json(result_path)
                    if (
                        expected_result_schema is None
                        or not isinstance(result, Mapping)
                        or result.get("schema") != expected_result_schema
                    ):
                        errors.append(
                            f"$.cell_evidence[{index}].result.schema: differs from "
                            "frozen policy bundle"
                        )
                    result_errors = audit_result_payload(
                        result,
                        gemini_legacy_compatibility=(
                            gemini_legacy_compatibility
                            if isinstance(gemini_legacy_compatibility, Mapping)
                            and gemini_legacy_compatibility.get("certificate")
                            is not None
                            else None
                        ),
                    )
                    errors.extend(
                        f"$.cell_evidence[{index}].result{item.removeprefix('$')}"
                        for item in result_errors
                    )
                    paper = result.get("paper_protocol") if isinstance(result, Mapping) else None
                    if isinstance(paper, Mapping):
                        if paper.get("cell") != cell:
                            errors.append(
                                f"$.cell_evidence[{index}].result: cell differs from binding"
                            )
                        if paper.get("call_graph_mode") != policy.get("call_graph_mode"):
                            errors.append(
                                f"$.cell_evidence[{index}].result: call graph differs from manifest"
                            )
                        if paper.get("claim_type") != policy.get("claim_type"):
                            errors.append(
                                f"$.cell_evidence[{index}].result: claim type differs from manifest"
                            )
                        if paper.get("claim_scope") != claim_scope:
                            errors.append(
                                f"$.cell_evidence[{index}].result: claim scope differs from manifest"
                            )
                        if paper.get("service_clock_artifact_sha256") != artifact_identity_sha256(
                            service_artifact
                        ):
                            errors.append(
                                f"$.cell_evidence[{index}].result: service clock differs from manifest"
                            )
                    if isinstance(result, Mapping):
                        if result.get("runtime_parameters") != expected_runtime_result:
                            errors.append(
                                f"$.cell_evidence[{index}].result: runtime parameters "
                                "differ from sealed manifest"
                            )
                        if expected_model_inventory is not None:
                            result_environment = _audit_runtime_environment_contract(
                                result.get("runtime_environment_contract"),
                                expected_model_inventory_sha256=(
                                    expected_model_inventory
                                ),
                                label=(
                                    f"$.cell_evidence[{index}].result."
                                    "runtime_environment_contract"
                                ),
                                errors=errors,
                            )
                            if result_environment != evidence_environment:
                                errors.append(
                                    f"$.cell_evidence[{index}].result."
                                    "runtime_environment_contract: differs from evidence binding"
                                )
                            if evidence_scheduler is not None:
                                result_scheduler = result.get(
                                    "scheduler_runtime_contract"
                                )
                                if (
                                    not isinstance(result_scheduler, Mapping)
                                    or dict(result_scheduler)
                                    != dict(evidence_scheduler)
                                ):
                                    errors.append(
                                        f"$.cell_evidence[{index}].result."
                                        "scheduler_runtime_contract: differs from "
                                        "evidence binding"
                                    )
                        result_provenance = _audit_runtime_provenance_shape(
                            result.get("provenance"),
                            label=f"$.cell_evidence[{index}].result.provenance",
                            errors=errors,
                            extra_fields=(
                                GEMINI_LEGACY_COMPATIBILITY_PROVENANCE_FIELDS
                                if gemini_legacy_compatibility is not None
                                else ()
                            ),
                        )
                        if dict(result_provenance) != dict(provenance):
                            errors.append(
                                f"$.cell_evidence[{index}].result.provenance: "
                                "complete mapping differs from evidence binding"
                            )
                        for field in expected_provenance:
                            if result_provenance.get(field) != provenance.get(field):
                                errors.append(
                                    f"$.cell_evidence[{index}].result.provenance.{field}: "
                                    "differs from evidence binding"
                                )
                        for result_field, expected in (
                            ("block_id", block_id),
                            ("order_position", evidence.get("order_position")),
                            ("gpu_ids", evidence.get("gpu_ids")),
                            ("server_instance_id", evidence.get("server_instance_id")),
                            ("broker_instance_id", evidence.get("broker_instance_id")),
                            ("started_wall_s", evidence.get("started_wall_s")),
                            ("ended_wall_s", evidence.get("ended_wall_s")),
                        ):
                            if result.get(result_field) != expected:
                                errors.append(
                                    f"$.cell_evidence[{index}].result.{result_field}: "
                                    "differs from evidence binding"
                                )

        for block_id, intervals in evidence_intervals.items():
            if set(intervals) != {1, 2, 3, 4}:
                errors.append(
                    f"$.cell_evidence: block {block_id} lacks all wall-clock positions"
                )
                continue
            for position in range(1, 4):
                if intervals[position + 1][0] < intervals[position][1]:
                    errors.append(
                        f"$.cell_evidence: block {block_id} positions {position} "
                        f"and {position + 1} overlap or ran out of order"
                    )

    speedup_20_pass: bool | None = None
    strong_20_claim_pass: bool | None = None
    outcomes = root.get("outcomes")
    if require_evidence and outcomes is None:
        errors.append("$.outcomes: required for post-run audit")
    if outcomes is not None:
        outcomes_map = _require_mapping(outcomes, "$.outcomes", errors)
        required_contrasts = {"A_vs_B", "A_vs_E", "E_vs_F", "B_vs_F", "A_vs_F", "interaction"}
        if set(outcomes_map) != required_contrasts:
            errors.append("$.outcomes: must report all five contrasts and interaction")
        for name in sorted(required_contrasts - {"interaction"}):
            row = outcomes_map.get(name)
            if not isinstance(row, Mapping):
                errors.append(f"$.outcomes.{name}: expected object")
                continue
            if row.get("estimand") != "ratio_of_paired_root_mean_e2e":
                errors.append(
                    f"$.outcomes.{name}.estimand: expected "
                    "'ratio_of_paired_root_mean_e2e'"
                )
            estimate = row.get("ratio_of_paired_root_mean_e2e")
            ci = row.get("paired_bootstrap_95_ci")
            if not _is_number(estimate):
                errors.append(
                    f"$.outcomes.{name}.ratio_of_paired_root_mean_e2e: invalid"
                )
            if (
                not isinstance(ci, list)
                or len(ci) != 2
                or not all(_is_number(item) for item in ci)
                or float(ci[0]) > float(ci[1])
            ):
                errors.append(f"$.outcomes.{name}.paired_bootstrap_95_ci: invalid")
        interaction = outcomes_map.get("interaction")
        if not isinstance(interaction, Mapping):
            errors.append("$.outcomes.interaction: expected object")
        else:
            estimate = interaction.get("mean_interaction_s")
            ci = interaction.get("paired_bootstrap_95_ci_s")
            if not _is_number(estimate):
                errors.append("$.outcomes.interaction.mean_interaction_s: invalid")
            if (
                not isinstance(ci, list)
                or len(ci) != 2
                or not all(_is_number(item) for item in ci)
                or float(ci[0]) > float(ci[1])
            ):
                errors.append("$.outcomes.interaction.paired_bootstrap_95_ci_s: invalid")
        combined = outcomes_map.get("A_vs_F")
        if isinstance(combined, Mapping):
            estimate = combined.get("ratio_of_paired_root_mean_e2e")
            ci = combined.get("paired_bootstrap_95_ci")
            if not _is_number(estimate):
                errors.append(
                    "$.outcomes.A_vs_F.ratio_of_paired_root_mean_e2e: invalid"
                )
            if (
                not isinstance(ci, list)
                or len(ci) != 2
                or not all(_is_number(item) for item in ci)
                or float(ci[0]) > float(ci[1])
            ):
                errors.append("$.outcomes.A_vs_F.paired_bootstrap_95_ci: invalid")
            if _is_number(estimate) and isinstance(ci, list) and len(ci) == 2 and all(
                _is_number(item) for item in ci
            ):
                speedup_20_pass = float(estimate) >= 0.20 and float(ci[0]) > 0.0
                strong_20_claim_pass = float(ci[0]) >= 0.20

    if outcomes is not None and isinstance(evidence_raw, list):
        evidence_binding = _require_mapping(
            root.get("analysis_evidence_manifest"),
            "$.analysis_evidence_manifest",
            errors,
        )
        report_binding = _require_mapping(
            root.get("analysis_report"), "$.analysis_report", errors
        )
        _audit_file_binding_shape(
            evidence_binding, label="$.analysis_evidence_manifest", errors=errors
        )
        _audit_file_binding_shape(
            report_binding, label="$.analysis_report", errors=errors
        )
        if not _is_sha256(report_binding.get("identity_sha256")):
            errors.append("$.analysis_report.identity_sha256: invalid")
        if verify_files:
            analysis_start_errors = len(errors)
            _verify_file_binding(
                evidence_binding,
                base=base,
                label="$.analysis_evidence_manifest",
                errors=errors,
            )
            _verify_file_binding(
                report_binding,
                base=base,
                label="$.analysis_report",
                errors=errors,
            )
            if len(errors) == analysis_start_errors:
                evidence_path = Path(str(evidence_binding["path"]))
                report_path = Path(str(report_binding["path"]))
                if not evidence_path.is_absolute():
                    evidence_path = base / evidence_path
                if not report_path.is_absolute():
                    report_path = base / report_path
                try:
                    evidence_document = _load_json(evidence_path)
                    report_document = _load_json(report_path)
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"$.analysis_report: unreadable bound input: {exc}")
                else:
                    if not isinstance(evidence_document, Mapping):
                        errors.append("$.analysis_evidence_manifest: expected object")
                    elif sealed_payload_sha256(
                        evidence_document
                    ) != sealed_payload_sha256(root):
                        errors.append(
                            "$.analysis_evidence_manifest: sealed/preregistered "
                            "content differs from final manifest"
                        )
                    elif evidence_document.get("cell_evidence") != root.get(
                        "cell_evidence"
                    ):
                        errors.append(
                            "$.analysis_evidence_manifest: cell evidence differs from final manifest"
                        )
                    if not isinstance(report_document, Mapping):
                        errors.append("$.analysis_report: expected object")
                    else:
                        declared_analysis_identity = report_document.get(
                            "analysis_sha256"
                        )
                        unsigned_report = dict(report_document)
                        unsigned_report.pop("analysis_sha256", None)
                        if declared_analysis_identity != canonical_sha256(unsigned_report):
                            errors.append(
                                "$.analysis_report.analysis_sha256: canonical hash mismatch"
                            )
                        if report_binding.get(
                            "identity_sha256"
                        ) != declared_analysis_identity:
                            errors.append(
                                "$.analysis_report.identity_sha256: differs from report"
                            )
                        if report_document.get("manifest_sha256") != evidence_binding.get(
                            "sha256"
                        ):
                            errors.append(
                                "$.analysis_report.manifest_sha256: differs from bound evidence manifest"
                            )
                    if isinstance(evidence_document, Mapping) and isinstance(
                        report_document, Mapping
                    ):
                        try:
                            import analyze_strict_causal_abef as trusted_analyzer

                            trusted_report = trusted_analyzer.analyze_manifest(
                                evidence_path
                            )
                        except Exception as exc:  # fail closed at the trust boundary
                            errors.append(
                                "$.analysis_report: trusted recomputation failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        else:
                            if report_document != trusted_report:
                                errors.append(
                                    "$.analysis_report: differs from trusted recomputation"
                                )
                            if outcomes != trusted_report.get("manifest_outcomes"):
                                errors.append(
                                    "$.outcomes: differs from trusted recomputation"
                                )

    if errors:
        confirmatory_eligible = False
    return {
        "valid": not errors,
        "claim_scope": claim_scope,
        "confirmatory_eligible": confirmatory_eligible,
        "speedup_20_pass": speedup_20_pass,
        "strong_20_claim_pass": strong_20_claim_pass,
        "errors": errors,
        "warnings": warnings,
    }


def _command_manifest(args: argparse.Namespace) -> int:
    path = args.manifest.resolve()
    result = audit_manifest(
        _load_json(path),
        base=path.parent,
        verify_files=args.verify_files,
        require_evidence=args.require_evidence,
    )
    result["manifest"] = str(path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


def _command_result(args: argparse.Namespace) -> int:
    path = args.result.resolve()
    errors = audit_result_payload(_load_json(path))
    output = {"result": str(path), "valid": not errors, "errors": errors}
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 2


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="audit a frozen protocol manifest")
    manifest.add_argument("manifest", type=Path)
    manifest.add_argument("--verify-files", action="store_true")
    manifest.add_argument("--require-evidence", action="store_true")
    manifest.set_defaults(func=_command_manifest)

    result = commands.add_parser("result", help="scan one cell result for leakage")
    result.add_argument("result", type=Path)
    result.set_defaults(func=_command_result)
    return top


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
