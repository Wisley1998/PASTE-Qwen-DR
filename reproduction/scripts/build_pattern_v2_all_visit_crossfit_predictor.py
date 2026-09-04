#!/usr/bin/env python3
"""Build and validate the retrospective cross-fit Pattern V2 predictor.

This artifact is a runtime serialization of the 100-root, nested whole-session
cross-fit model used by the all-Visit Pattern V2 analysis.  It is deliberately
label-free at each held-out root: fold ``k`` contains only parameters fitted
from roots whose outer fold is not ``k``.  It is *not* an untouched-test-set
artifact, because each fold uses labels from the other evaluation roots; that
limitation is recorded explicitly in the artifact.

No trace timing, future target URL, exact-match label, or outcome is serialized.
The optional validation report is post-hoc evaluation output and is kept in a
separate file that the runtime predictor never reads.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(REPRODUCTION_ROOT))
sys.path.insert(0, str(SCRIPT.parent))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from paste_repro.pattern_v2_all_visit_online import (  # noqa: E402
    CANDIDATE_POOL_SIZE,
    DEPLOYABLE_SCHEMA,
    FEATURE_SCHEMA,
    POLICY,
    SCHEMA,
    TOP_K,
    PatternV2CrossFitPredictor,
    PatternV2DeployablePredictor,
    canonical_sha256,
    crossfit_fold,
)
from paste_repro.traces import LLMCall, ToolCall, load_sessions, load_trace  # noqa: E402
from run_pattern_v2_adaptive_load import inner_fold  # noqa: E402
from run_pattern_v2_trace_all_visit_wall import (  # noqa: E402
    DEFAULT_TRACES,
    RICH_FEATURE_NAMES,
    calibrators_by_trigger,
    executable_url,
    extract_all_visit_decisions,
    fit_generalized_rank_pattern,
    generate_raw_windows,
)


DEFAULT_OUTPUT = (
    REPRODUCTION_ROOT
    / "artifacts"
    / "pattern_v2_all_visit_crossfit"
    / "predictor.json"
)
VALIDATION_SCHEMA = "paste_repro.pattern_v2_all_visit_crossfit_validation.v1"
DEPLOYABLE_VALIDATION_SCHEMA = (
    "paste_repro.pattern_v2_all_visit_deployable_validation.v1"
)
DEFAULT_SPLIT_MANIFEST = (
    REPRODUCTION_ROOT
    / "artifacts"
    / "fixed_trace_splits"
    / "30a0cb7c58b3-1ff2b2e2feb5-c40-t30-f30"
    / "split_manifest.json"
)


def _table_rows(
    values: Mapping[Any, float], *, key_width: int
) -> list[dict[str, Any]]:
    """Encode non-string-keyed calibration tables in canonical order."""

    normalized: list[tuple[tuple[Any, ...], float]] = []
    for raw_key, raw_value in values.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        if len(key) != key_width:
            raise ValueError("calibration table key width mismatch")
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("calibration table contains an invalid probability")
        normalized.append((tuple(key), value))
    normalized.sort(key=lambda row: canonical_sha256(list(row[0])))
    return [
        {"key": list(key), "value": value}
        for key, value in normalized
    ]


def _serialize_count(model: Any) -> dict[str, Any]:
    """Serialize every count-model parameter consumed by the blend runtime."""

    return {
        "visit_global": float(model.visit_global),
        "visit_query": _table_rows(model.visit_query, key_width=1),
        "visit_detail": _table_rows(model.visit_detail, key_width=3),
    }


def _finite_vector(value: Any, *, width: int, label: str) -> list[float]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} has invalid shape or values")
    return [float(item) for item in result]


def _serialize_rich(model: Any) -> dict[str, Any]:
    width = len(FEATURE_SCHEMA)
    return {
        "mean": _finite_vector(model.mean, width=width, label="rich.mean"),
        "scale": _finite_vector(model.scale, width=width, label="rich.scale"),
        "weights": _finite_vector(
            model.weights, width=width + 1, label="rich.weights"
        ),
    }


def _serialize_pairwise(model: Any) -> dict[str, Any]:
    width = len(FEATURE_SCHEMA)
    return {
        "scale": _finite_vector(model.scale, width=width, label="pairwise.scale"),
        "weights": _finite_vector(
            model.weights, width=width, label="pairwise.weights"
        ),
    }


def _serialize_trigger_models(
    models: Mapping[str, Any], serializer: Any
) -> dict[str, Any]:
    return {str(key): serializer(value) for key, value in sorted(models.items())}


def _fit_model_payload(
    *,
    fold_marker: int,
    train_ids: set[str],
    decisions: Sequence[Any],
) -> dict[str, Any]:
    # The calibration labels are themselves generated out of fold inside the
    # supplied training roots.  Evaluation roots never participate.
    calibration_windows: list[Any] = []
    for inner in range(4):
        inner_validation = {sid for sid in train_ids if inner_fold(sid) == inner}
        inner_fit = train_ids - inner_validation
        if not inner_fit or not inner_validation:
            raise RuntimeError(
                f"model {fold_marker}, inner {inner} has an empty fit/eval split"
            )
        calibration_windows.extend(
            generate_raw_windows(
                decisions,
                fit_ids=inner_fit,
                evaluation_ids=inner_validation,
                candidate_pool_size=CANDIDATE_POOL_SIZE,
            )
        )

    (
        global_count,
        trigger_count,
        global_rich,
        trigger_rich,
        global_pairwise,
        trigger_pairwise,
    ) = calibrators_by_trigger(calibration_windows)
    rank_model = fit_generalized_rank_pattern(decisions, train_ids)
    ordered_train_ids = sorted(train_ids)
    return {
        "outer_fold": fold_marker,
        "training_session_ids": ordered_train_ids,
        "training_session_ids_sha256": canonical_sha256(ordered_train_ids),
        "rank_counts": {
            str(rank): int(count)
            for rank, count in sorted(rank_model.rank_counts.items())
            if int(count) > 0
        },
        "global_count": _serialize_count(global_count),
        "trigger_count": _serialize_trigger_models(
            trigger_count, _serialize_count
        ),
        "global_rich": _serialize_rich(global_rich),
        "trigger_rich": _serialize_trigger_models(trigger_rich, _serialize_rich),
        "global_pairwise": _serialize_pairwise(global_pairwise),
        "trigger_pairwise": _serialize_trigger_models(
            trigger_pairwise, _serialize_pairwise
        ),
    }


def _build_fold(
    *,
    outer_fold: int,
    session_ids: set[str],
    decisions: Sequence[Any],
) -> dict[str, Any]:
    train_ids = {sid for sid in session_ids if crossfit_fold(sid) != outer_fold}
    heldout_ids = session_ids - train_ids
    if not train_ids or not heldout_ids:
        raise RuntimeError(f"outer fold {outer_fold} is empty")
    return _fit_model_payload(
        fold_marker=outer_fold,
        train_ids=train_ids,
        decisions=decisions,
    )


def build_artifact(traces: Path) -> tuple[dict[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    sessions = load_sessions(traces)
    session_ids = {session.session_id for session in sessions}
    if len(session_ids) != len(sessions):
        raise RuntimeError("trace source contains duplicate session IDs")
    if tuple(FEATURE_SCHEMA) != tuple(RICH_FEATURE_NAMES):
        raise RuntimeError("online and training feature schemas differ")
    decisions = extract_all_visit_decisions(sessions)
    folds = {
        str(outer): _build_fold(
            outer_fold=outer,
            session_ids=session_ids,
            decisions=decisions,
        )
        for outer in range(5)
    }
    ordered_session_ids = sorted(session_ids)
    unsigned: dict[str, Any] = {
        "schema": SCHEMA,
        "policy": POLICY,
        "evaluation_regime": "retrospective_crossfit",
        "uses_other_evaluation_root_labels": True,
        "uses_heldout_root_labels_per_fold": False,
        "predictor_uses_trace_timing": False,
        "configuration": {
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "top_k": TOP_K,
            "selector_model": "blend",
            "candidate_ranking": "exact_probability_only_no_duration_input",
            "cache_scope": "session_url_infinite_ttl",
        },
        "feature_schema": list(FEATURE_SCHEMA),
        "training_protocol": {
            "outer_folds": 5,
            "outer_assignment": "sha256(pattern-cache-grouped-cv-v1\\0session_id)%5",
            "inner_folds": 4,
            "inner_assignment": "sha256(pattern-confidence-inner-v1\\0session_id)%4",
            "rank_fit": "all non-held-out outer-fold roots",
            "probability_fit": "inner whole-session OOF rows from non-held-out roots",
        },
        "source_inventory": {
            "trace_directory_name": traces.resolve().name,
            "source_session_count": len(sessions),
            "source_session_ids_sha256": canonical_sha256(ordered_session_ids),
        },
        "folds": folds,
    }
    artifact = {**unsigned, "artifact_sha256": canonical_sha256(unsigned)}
    # Parse immediately through the strict runtime schema before writing it.
    PatternV2CrossFitPredictor(artifact)
    return artifact, sessions, decisions


def _load_fixed_split(
    split_manifest: Path, available_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("fixed split manifest must be an object")
    supplied_hash = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(supplied_hash, str) or canonical_sha256(unsigned) != supplied_hash:
        raise ValueError("fixed split manifest checksum mismatch")

    roles: dict[str, set[str]] = {}
    for role in ("calibration", "tuning", "final"):
        rows = payload.get(f"{role}_sessions")
        if not isinstance(rows, list):
            raise ValueError(f"fixed split is missing {role}_sessions")
        ids: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("session_id"), str):
                raise ValueError(f"fixed split contains an invalid {role} row")
            ids.add(str(row["session_id"]))
        if len(ids) != len(rows):
            raise ValueError(f"fixed split contains duplicate {role} roots")
        if not ids <= available_ids:
            missing = sorted(ids - available_ids)
            raise ValueError(f"latest trace is missing {role} roots: {missing[:3]}")
        roles[role] = ids
    if (
        roles["calibration"] & roles["tuning"]
        or roles["calibration"] & roles["final"]
        or roles["tuning"] & roles["final"]
    ):
        raise ValueError("fixed split roles overlap")
    if set().union(*roles.values()) != available_ids:
        raise ValueError("fixed split does not cover the latest trace inventory")
    expected_counts = {"calibration": 40, "tuning": 30, "final": 30}
    if {role: len(ids) for role, ids in roles.items()} != expected_counts:
        raise ValueError("fixed split is not the frozen 40/30/30 partition")
    return roles, dict(payload)


def build_deployable_artifact(
    traces: Path,
    *,
    split_manifest: Path,
    training_role: str,
) -> tuple[dict[str, Any], set[str]]:
    if training_role not in {"calibration_only", "calibration_plus_tuning"}:
        raise ValueError("unsupported deployable training role")
    trace_paths = sorted(traces.glob("*.jsonl"), key=lambda path: path.name)
    if not trace_paths:
        raise FileNotFoundError(f"no JSONL traces found in {traces}")
    available_ids = {path.name for path in trace_paths}
    if len(available_ids) != len(trace_paths):
        raise RuntimeError("trace source contains duplicate session IDs")
    if tuple(FEATURE_SCHEMA) != tuple(RICH_FEATURE_NAMES):
        raise RuntimeError("online and training feature schemas differ")
    roles, split_payload = _load_fixed_split(split_manifest, available_ids)
    train_ids = set(roles["calibration"])
    if training_role == "calibration_plus_tuning":
        train_ids.update(roles["tuning"])
    evaluation_ids = set(roles["final"])
    if train_ids & evaluation_ids:
        raise RuntimeError("deployable training and final evaluation roots overlap")

    # The builder opens only frozen training roots.  Final trace contents are
    # not parsed until the artifact has been validated and written to disk.
    training_sessions = tuple(
        load_trace(traces / session_id) for session_id in sorted(train_ids)
    )
    decisions = extract_all_visit_decisions(training_sessions)
    model = _fit_model_payload(
        fold_marker=-1,
        train_ids=train_ids,
        decisions=decisions,
    )
    ordered_available_ids = sorted(available_ids)
    unsigned: dict[str, Any] = {
        "schema": DEPLOYABLE_SCHEMA,
        "policy": POLICY,
        "configuration": {
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "top_k": TOP_K,
            "selector_model": "blend",
            "candidate_ranking": "exact_probability_only_no_duration_input",
            "cache_scope": "session_url_infinite_ttl",
        },
        "feature_schema": list(FEATURE_SCHEMA),
        "evaluation_regime": "frozen_train_eval",
        "claim_scope": "retrospective_internal_holdout",
        "prior_policy_development_used_evaluation_corpus": True,
        "uses_evaluation_root_labels": False,
        "predictor_uses_trace_timing": False,
        "training_role": training_role,
        "training_provenance": {
            "fixed_split_schema": split_payload.get("schema"),
            "fixed_split_manifest_sha256": split_payload["manifest_sha256"],
            "training_session_count": len(train_ids),
            "training_session_ids_sha256": canonical_sha256(sorted(train_ids)),
            "trace_directory_name": traces.resolve().name,
            "trace_source_session_count": len(trace_paths),
            "trace_source_session_ids_sha256": canonical_sha256(
                ordered_available_ids
            ),
            "rank_fit": "all frozen training roots",
            "probability_fit": "inner whole-session OOF rows from frozen training roots",
        },
        "model": model,
    }
    artifact = {**unsigned, "artifact_sha256": canonical_sha256(unsigned)}
    PatternV2DeployablePredictor(artifact)
    return artifact, evaluation_ids


def _assert_model_payload_is_label_and_timing_free(artifact: Mapping[str, Any]) -> None:
    """Audit serialized fold payloads, excluding required disclosure fields."""

    forbidden_fragments = (
        "target",
        "exact_match",
        "authoritative",
        "outcome",
        "duration",
        "overlap",
        "service_time",
        "timing",
    )

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in forbidden_fragments):
                    raise RuntimeError(f"forbidden model field at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    if "folds" in artifact:
        walk(artifact["folds"], "folds")
    elif "model" in artifact:
        walk(artifact["model"], "model")
    else:
        raise RuntimeError("predictor artifact has no serialized model payload")


def validate_artifact(
    artifact: Mapping[str, Any], sessions: Sequence[Any], decisions: Sequence[Any]
) -> dict[str, Any]:
    _assert_model_payload_is_label_and_timing_free(artifact)
    predictor = PatternV2CrossFitPredictor(artifact)
    sessions_by_id = {session.session_id: session for session in sessions}
    decisions_by_session: dict[str, list[Any]] = defaultdict(list)
    for decision in decisions:
        decisions_by_session[decision.session_id].append(decision)

    hits = 0
    authority_urls = 0
    predictions = 0
    unique_speculative_starts = 0
    fold_hits: dict[str, int] = defaultdict(int)
    fold_targets: dict[str, int] = defaultdict(int)
    fold_sessions: dict[str, int] = defaultdict(int)
    per_session_rows: list[dict[str, Any]] = []

    for session_id in sorted(sessions_by_id):
        session = sessions_by_id[session_id]
        runtime = predictor.start_session(
            source_session_id=session_id,
            runtime_session_id=f"validation:{session_id}",
        )
        fold = str(crossfit_fold(session_id))
        fold_sessions[fold] += 1
        cache: set[str] = set()
        session_hits = 0
        session_targets = 0
        session_predictions = 0
        session_starts = 0
        ordered = sorted(
            decisions_by_session.get(session_id, ()),
            key=lambda row: row.trigger_event_index,
        )
        for decision in ordered:
            trigger = session.events[decision.trigger_event_index]
            if not isinstance(trigger, ToolCall):
                raise RuntimeError("decision trigger is not a tool call")
            if not decision.lead_llm_event_indices:
                raise RuntimeError("prediction decision has no visible lead LLM")
            lead = session.events[decision.lead_llm_event_indices[0]]
            if not isinstance(lead, LLMCall):
                raise RuntimeError("decision lead event is not an LLM call")
            emitted = runtime.predict_after_tool(
                tool_name=trigger.tool_name,
                tool_arguments=trigger.tool_args,
                current_messages=lead.messages,
            )
            urls = tuple(row.url for row in emitted)
            if len(urls) != len(set(urls)) or len(urls) > TOP_K:
                raise RuntimeError("runtime emitted an invalid Top-K prediction set")
            session_predictions += len(urls)
            for url in urls:
                if url not in cache:
                    cache.add(url)
                    session_starts += 1

            targets = tuple(
                url for url in decision.authoritative_urls if executable_url(url)
            )
            current_hits = sum(url in cache for url in targets)
            session_hits += current_hits
            session_targets += len(targets)

        hits += session_hits
        authority_urls += session_targets
        predictions += session_predictions
        unique_speculative_starts += session_starts
        fold_hits[fold] += session_hits
        fold_targets[fold] += session_targets
        per_session_rows.append(
            {
                "source_session_id": session_id,
                "outer_fold": int(fold),
                "authority_urls": session_targets,
                "cache_hit_occurrences": session_hits,
                "emitted_predictions": session_predictions,
                "unique_speculative_starts": session_starts,
            }
        )

    coverage = hits / authority_urls if authority_urls else 0.0
    summary: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "evaluation_regime": "retrospective_crossfit",
        "predictor_artifact_sha256": artifact["artifact_sha256"],
        "selection": "blend exact probability descending, fixed Top-10",
        "cache": "infinite-TTL session URL membership coverage",
        "source_sessions": len(sessions),
        "prediction_windows": len(decisions),
        "emitted_predictions": predictions,
        "unique_speculative_starts": unique_speculative_starts,
        "authority_urls": authority_urls,
        "cache_hit_occurrences": hits,
        "cache_hit_rate": coverage,
        "expected_reference": {
            "authority_urls": 499,
            "cache_hit_occurrences": 358,
            "matches": authority_urls == 499 and hits == 358,
        },
        "folds": [
            {
                "outer_fold": fold,
                "sessions": fold_sessions[str(fold)],
                "authority_urls": fold_targets[str(fold)],
                "cache_hit_occurrences": fold_hits[str(fold)],
                "cache_hit_rate": (
                    fold_hits[str(fold)] / fold_targets[str(fold)]
                    if fold_targets[str(fold)]
                    else 0.0
                ),
            }
            for fold in range(5)
        ],
        "sessions": per_session_rows,
    }
    unsigned = dict(summary)
    summary["validation_sha256"] = canonical_sha256(unsigned)
    if not summary["expected_reference"]["matches"]:
        raise RuntimeError(
            "Pattern V2 runtime equivalence failed: "
            f"observed {hits}/{authority_urls}, expected 358/499"
        )
    return summary


def validate_deployable_artifact(
    artifact: Mapping[str, Any],
    sessions: Sequence[Any],
    decisions: Sequence[Any],
    evaluation_ids: set[str],
) -> dict[str, Any]:
    """Evaluate one frozen fit on final30, with retrospective disclosure.

    The final roots do not enter this serialized fit.  They are not a new
    confirmatory corpus, however, because Pattern V2 was previously developed
    and inspected on the same 100-root collection.
    """

    _assert_model_payload_is_label_and_timing_free(artifact)
    predictor = PatternV2DeployablePredictor(artifact)
    sessions_by_id = {
        session.session_id: session
        for session in sessions
        if session.session_id in evaluation_ids
    }
    if set(sessions_by_id) != evaluation_ids:
        raise RuntimeError("deployable validation did not resolve every final root")
    decisions_by_session: dict[str, list[Any]] = defaultdict(list)
    for decision in decisions:
        if decision.session_id in evaluation_ids:
            decisions_by_session[decision.session_id].append(decision)

    hits = 0
    authority_urls = 0
    predictions = 0
    unique_speculative_starts = 0
    evaluated_windows = 0
    per_session_rows: list[dict[str, Any]] = []
    for session_id in sorted(evaluation_ids):
        session = sessions_by_id[session_id]
        runtime = predictor.start_session(
            source_session_id=session_id,
            runtime_session_id=f"final-validation:{session_id}",
        )
        cache: set[str] = set()
        session_hits = 0
        session_targets = 0
        session_predictions = 0
        session_starts = 0
        ordered = sorted(
            decisions_by_session.get(session_id, ()),
            key=lambda row: row.trigger_event_index,
        )
        evaluated_windows += len(ordered)
        for decision in ordered:
            trigger = session.events[decision.trigger_event_index]
            if not isinstance(trigger, ToolCall):
                raise RuntimeError("decision trigger is not a tool call")
            if not decision.lead_llm_event_indices:
                raise RuntimeError("prediction decision has no visible lead LLM")
            lead = session.events[decision.lead_llm_event_indices[0]]
            if not isinstance(lead, LLMCall):
                raise RuntimeError("decision lead event is not an LLM call")
            emitted = runtime.predict_after_tool(
                tool_name=trigger.tool_name,
                tool_arguments=trigger.tool_args,
                current_messages=lead.messages,
            )
            urls = tuple(row.url for row in emitted)
            if len(urls) != len(set(urls)) or len(urls) > TOP_K:
                raise RuntimeError("runtime emitted an invalid Top-K prediction set")
            session_predictions += len(urls)
            for url in urls:
                if url not in cache:
                    cache.add(url)
                    session_starts += 1
            targets = tuple(
                url for url in decision.authoritative_urls if executable_url(url)
            )
            session_hits += sum(url in cache for url in targets)
            session_targets += len(targets)

        hits += session_hits
        authority_urls += session_targets
        predictions += session_predictions
        unique_speculative_starts += session_starts
        per_session_rows.append(
            {
                "source_session_id": session_id,
                "authority_urls": session_targets,
                "cache_hit_occurrences": session_hits,
                "emitted_predictions": session_predictions,
                "unique_speculative_starts": session_starts,
            }
        )

    training_role = str(artifact["training_role"])
    expected_hits = 100 if training_role == "calibration_plus_tuning" else 97
    coverage = hits / authority_urls if authority_urls else 0.0
    summary: dict[str, Any] = {
        "schema": DEPLOYABLE_VALIDATION_SCHEMA,
        "evaluation_regime": "frozen_train_eval",
        "claim_scope": "retrospective_internal_holdout",
        "prior_policy_development_used_evaluation_corpus": True,
        "training_role": training_role,
        "evaluation_role": "final",
        "predictor_artifact_sha256": artifact["artifact_sha256"],
        "selection": "blend exact probability descending, fixed Top-10",
        "cache": "infinite-TTL session URL membership coverage",
        "source_sessions": len(evaluation_ids),
        "prediction_windows": evaluated_windows,
        "emitted_predictions": predictions,
        "unique_speculative_starts": unique_speculative_starts,
        "authority_urls": authority_urls,
        "cache_hit_occurrences": hits,
        "cache_hit_rate": coverage,
        "expected_reference": {
            "authority_urls": 129,
            "cache_hit_occurrences": expected_hits,
            "matches": authority_urls == 129 and hits == expected_hits,
        },
        "sessions": per_session_rows,
    }
    unsigned = dict(summary)
    summary["validation_sha256"] = canonical_sha256(unsigned)
    if not summary["expected_reference"]["matches"]:
        raise RuntimeError(
            "Pattern V2 deployable runtime equivalence failed: "
            f"observed {hits}/{authority_urls}, expected {expected_hits}/129"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument(
        "--mode", choices=("crossfit", "deployable"), default="crossfit"
    )
    parser.add_argument(
        "--training-role",
        choices=("calibration_only", "calibration_plus_tuning"),
        default="calibration_plus_tuning",
        help="used only in deployable mode",
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="build only; validation is enabled by default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "crossfit":
        artifact, sessions, decisions = build_artifact(args.traces)
        evaluation_ids: set[str] | None = None
        default_dir = DEFAULT_OUTPUT.parent
    else:
        artifact, evaluation_ids = build_deployable_artifact(
            args.traces,
            split_manifest=args.split_manifest,
            training_role=args.training_role,
        )
        sessions = ()
        decisions = ()
        default_dir = (
            REPRODUCTION_ROOT
            / "artifacts"
            / f"pattern_v2_all_visit_deployable_{args.training_role}"
        )
    output = args.output or (default_dir / "predictor.json")
    validation_output = args.validation_output or (default_dir / "validation.json")
    write_json_atomic(output, artifact)
    validation: dict[str, Any] | None = None
    if not args.skip_validation:
        if args.mode == "crossfit":
            validation = validate_artifact(artifact, sessions, decisions)
        else:
            assert evaluation_ids is not None
            # This is intentionally after the frozen artifact write above.
            sessions = tuple(
                load_trace(args.traces / session_id)
                for session_id in sorted(evaluation_ids)
            )
            decisions = extract_all_visit_decisions(sessions)
            validation = validate_deployable_artifact(
                artifact, sessions, decisions, evaluation_ids
            )
        write_json_atomic(validation_output, validation)
    print(f"artifact={output.resolve()}")
    print(f"artifact_sha256={artifact['artifact_sha256']}")
    if validation is not None:
        print(f"validation={validation_output.resolve()}")
        print(
            "cache_coverage="
            f"{validation['cache_hit_occurrences']}/"
            f"{validation['authority_urls']}="
            f"{validation['cache_hit_rate']:.6%}"
        )
        print(f"validation_sha256={validation['validation_sha256']}")


if __name__ == "__main__":
    main()
