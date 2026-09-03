#!/usr/bin/env python3
"""Frozen train-CV and one-shot outer evaluation for exact-URL reranking."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Protocol, Sequence

import numpy as np


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REPRODUCTION_ROOT.parent
sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.analysis import evaluate_held_out  # noqa: E402
from paste_repro.contextual_mapper import (  # noqa: E402
    DEFAULT_L2,
    ContextualURLReranker,
    save_contextual_artifact,
)
from paste_repro.mapper import (  # noqa: E402
    URLRankMapper,
    save_artifact,
    write_json_atomic,
)
from paste_repro.pipeline import build_split_manifest  # noqa: E402
from paste_repro.tool_prediction import ContextualTraceVisitPredictor  # noqa: E402
from paste_repro.traces import (  # noqa: E402
    LLMCall,
    SearchResult,
    SearchVisitTransition,
    SessionTrace,
    ToolCall,
    count_tool_calls,
    latest_tool_response,
    load_sessions,
    parse_search_results,
    split_sessions,
    transitions_from_sessions,
)


OUTER_SEED = "paste-repro-v1"
OUTER_TRAIN_RATIO = 0.70
CV_SEED = "contextual-cv-v1"
BOOTSTRAP_SEED = "contextual-session-bootstrap-v1"
TOP_KS = (1, 3, 5)
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "1bf8984620a1a6eb5c4472dce76ed5039eb37ccb28c2e03ccdf460eff0425402"
)
EXPECTED_BASELINE_ARTIFACT_SHA256 = (
    "30a0cb7c58b35a29603ea6a805e17d09fca9fa6542a3e186c461ca303902bc56"
)
DEFAULT_OUTPUT = REPRODUCTION_ROOT / "results" / "predictor_optimization"
DEFAULT_TRACES = REPOSITORY_ROOT / "traces" / "my_traces"
DEFAULT_PROTOCOL = DEFAULT_OUTPUT / "FROZEN_PROTOCOL.md"


class Mapper(Protocol):
    def predict(
        self, search_results: Sequence[SearchResult], top_k: int
    ) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class SearchDecision:
    session_id: str
    search: ToolCall
    decision: LLMCall
    search_results: tuple[SearchResult, ...]
    outcome: str
    authoritative_urls: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a fail-closed JSON marker with an atomic O_EXCL claim."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:  # pragma: no cover - fdopen owns it on normal paths
            os.close(descriptor)


def cv_fold(session_id: str) -> int:
    digest = hashlib.sha256(f"{CV_SEED}\0{session_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 5


def prediction_urls(
    mapper: Mapper,
    search_results: Sequence[SearchResult],
    top_k: int = 5,
) -> tuple[str, ...]:
    return tuple(
        str(prediction.invocation.arguments["url"])
        for prediction in mapper.predict(search_results, top_k)
    )


def evaluate_predictions(
    mapper: Mapper,
    transitions: Sequence[SearchVisitTransition],
    *,
    split: str,
    model_name: str,
    fold: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hits = {top_k: 0 for top_k in TOP_KS}
    predictions = {top_k: 0 for top_k in TOP_KS}
    example_hits = {top_k: 0 for top_k in TOP_KS}
    session_counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "targets": 0,
            "hits": {top_k: 0 for top_k in TOP_KS},
        }
    )
    rows: list[dict[str, Any]] = []
    for transition_index, transition in enumerate(transitions):
        predicted = prediction_urls(mapper, transition.search_results, max(TOP_KS))
        targets = transition.authoritative_urls
        session_counts[transition.session_id]["targets"] += len(targets)
        row: dict[str, Any] = {
            "split": split,
            "model": model_name,
            "fold": fold,
            "session_id": transition.session_id,
            "decision_id": (
                f"{transition.session_id}:search-line-{transition.search.line_number}:"
                f"{transition_index}"
            ),
            "candidate_count": len({result.url for result in transition.search_results}),
            "target_count": len(targets),
            "targets": list(targets),
            "predictions": list(predicted),
        }
        for top_k in TOP_KS:
            prediction_set = set(predicted[:top_k])
            count = sum(url in prediction_set for url in targets)
            hits[top_k] += count
            predictions[top_k] += min(top_k, len(predicted))
            example_hits[top_k] += count > 0
            session_counts[transition.session_id]["hits"][top_k] += count
            row[f"hits_at_{top_k}"] = count
        rows.append(row)

    target_count = sum(len(item.authoritative_urls) for item in transitions)
    metrics: dict[str, Any] = {
        "sessions": len(session_counts),
        "decision_windows": len(transitions),
        "targets": target_count,
        "top_k": {},
    }
    for top_k in TOP_KS:
        per_session_recall = [
            values["hits"][top_k] / values["targets"]
            for values in session_counts.values()
            if values["targets"]
        ]
        metrics["top_k"][str(top_k)] = {
            "hits": hits[top_k],
            "target_recall": hits[top_k] / target_count if target_count else 0.0,
            "example_hits": example_hits[top_k],
            "example_hit_rate": (
                example_hits[top_k] / len(transitions) if transitions else 0.0
            ),
            "predictions": predictions[top_k],
            "conditional_precision": (
                hits[top_k] / predictions[top_k] if predictions[top_k] else 0.0
            ),
            "session_macro_target_recall": (
                sum(per_session_recall) / len(per_session_recall)
                if per_session_recall
                else 0.0
            ),
        }
    return metrics, rows


def aggregate_oof(
    train_sessions: Sequence[SessionTrace],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_transitions = transitions_from_sessions(train_sessions)
    results: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for model_name in ("M0_rank_only", "M1_contextual"):
        model_rows: list[dict[str, Any]] = []
        for fold in range(5):
            fit_transitions = tuple(
                transition
                for transition in all_transitions
                if cv_fold(transition.session_id) != fold
            )
            validation_transitions = tuple(
                transition
                for transition in all_transitions
                if cv_fold(transition.session_id) == fold
            )
            mapper: Mapper
            if model_name == "M0_rank_only":
                mapper = URLRankMapper().fit(fit_transitions)
            else:
                mapper = ContextualURLReranker(l2=DEFAULT_L2).fit(fit_transitions)
            _, rows = evaluate_predictions(
                mapper,
                validation_transitions,
                split="outer_train_oof",
                model_name=model_name,
                fold=fold,
            )
            model_rows.extend(rows)
        all_rows.extend(model_rows)
        results[model_name] = aggregate_prediction_rows(model_rows)
    return results, all_rows


def aggregate_prediction_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    targets = sum(int(row["target_count"]) for row in rows)
    sessions = {str(row["session_id"]) for row in rows}
    result: dict[str, Any] = {
        "sessions": len(sessions),
        "decision_windows": len(rows),
        "targets": targets,
        "top_k": {},
    }
    for top_k in TOP_KS:
        hits = sum(int(row[f"hits_at_{top_k}"]) for row in rows)
        predictions = sum(min(top_k, len(row["predictions"])) for row in rows)
        example_hits = sum(int(row[f"hits_at_{top_k}"]) > 0 for row in rows)
        per_session: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            per_session[str(row["session_id"])][0] += int(row[f"hits_at_{top_k}"])
            per_session[str(row["session_id"])][1] += int(row["target_count"])
        macro_values = [hit / count for hit, count in per_session.values() if count]
        result["top_k"][str(top_k)] = {
            "hits": hits,
            "target_recall": hits / targets if targets else 0.0,
            "example_hits": example_hits,
            "example_hit_rate": example_hits / len(rows) if rows else 0.0,
            "predictions": predictions,
            "conditional_precision": hits / predictions if predictions else 0.0,
            "session_macro_target_recall": (
                sum(macro_values) / len(macro_values) if macro_values else 0.0
            ),
        }
    return result


def candidate_ceiling(
    transitions: Sequence[SearchVisitTransition],
    *,
    use_full_visible_search_context: bool = False,
) -> dict[str, Any]:
    coverage = 0
    oracle = {top_k: 0 for top_k in TOP_KS}
    for transition in transitions:
        if use_full_visible_search_context:
            candidate_urls: set[str] = set()
            for message in transition.decision_llm.messages:
                content = message.get("content", "")
                if not (
                    message.get("role") == "user"
                    and isinstance(content, str)
                    and "SearXNG search for" in content
                ):
                    continue
                candidate_urls.update(
                    result.url for result in parse_search_results(content)
                )
        else:
            candidate_urls = {result.url for result in transition.search_results}
        visible_targets = {
            url for url in transition.authoritative_urls if url in candidate_urls
        }
        coverage += sum(url in candidate_urls for url in transition.authoritative_urls)
        for top_k in TOP_KS:
            oracle[top_k] += min(top_k, len(visible_targets))
    total = sum(len(item.authoritative_urls) for item in transitions)
    return {
        "covered_targets": coverage,
        "target_coverage": coverage / total if total else 0.0,
        "oracle_hits": {str(top_k): oracle[top_k] for top_k in TOP_KS},
        "oracle_recall": {
            str(top_k): oracle[top_k] / total if total else 0.0
            for top_k in TOP_KS
        },
    }


def extract_search_decisions(
    sessions: Sequence[SessionTrace],
) -> tuple[SearchDecision, ...]:
    decisions: list[SearchDecision] = []
    for session in sessions:
        events = session.events
        for index in range(len(events) - 1):
            search = events[index]
            decision = events[index + 1]
            if not (
                isinstance(search, ToolCall)
                and search.tool_name == "search"
                and isinstance(decision, LLMCall)
            ):
                continue
            raw_queries = search.tool_args.get("query")
            if isinstance(raw_queries, list):
                queries = tuple(item for item in raw_queries if isinstance(item, str))
            elif isinstance(raw_queries, str):
                queries = (raw_queries,)
            else:
                queries = ()
            next_event = events[index + 2] if index + 2 < len(events) else None
            if isinstance(next_event, ToolCall) and next_event.tool_name == "visit":
                raw_urls = next_event.tool_args.get("url")
                if isinstance(raw_urls, str):
                    targets = (raw_urls,) if raw_urls else ()
                elif isinstance(raw_urls, list):
                    targets = tuple(
                        value for value in raw_urls if isinstance(value, str) and value
                    )
                else:
                    targets = ()
                outcome = "visit"
            elif isinstance(next_event, ToolCall):
                targets = ()
                outcome = next_event.tool_name
            else:
                targets = ()
                outcome = "no_next_tool"
            decisions.append(
                SearchDecision(
                    session_id=session.session_id,
                    search=search,
                    decision=decision,
                    search_results=parse_search_results(
                        latest_tool_response(decision), queries=queries
                    ),
                    outcome=outcome,
                    authoritative_urls=targets,
                )
            )
    return tuple(decisions)


def extract_visible_search_inputs(
    sessions: Sequence[SessionTrace],
) -> tuple[tuple[SearchResult, ...], ...]:
    """Extract only pre-generation search inputs, without reading next events."""

    inputs: list[tuple[SearchResult, ...]] = []
    for session in sessions:
        events = session.events
        for index in range(len(events) - 1):
            search = events[index]
            decision = events[index + 1]
            if not (
                isinstance(search, ToolCall)
                and search.tool_name == "search"
                and isinstance(decision, LLMCall)
            ):
                continue
            raw_queries = search.tool_args.get("query")
            if isinstance(raw_queries, list):
                queries = tuple(item for item in raw_queries if isinstance(item, str))
            elif isinstance(raw_queries, str):
                queries = (raw_queries,)
            else:
                queries = ()
            inputs.append(
                parse_search_results(
                    latest_tool_response(decision), queries=queries
                )
            )
    return tuple(inputs)


def all_search_decision_audit(
    mapper: Mapper,
    sessions: Sequence[SessionTrace],
) -> dict[str, Any]:
    decisions = extract_search_decisions(sessions)
    outcomes = Counter(decision.outcome for decision in decisions)
    prediction_count = 0
    hits = 0
    visit_prediction_count = 0
    no_visit_prediction_count = 0
    for decision in decisions:
        predicted = prediction_urls(mapper, decision.search_results, 5)
        prediction_count += len(predicted)
        if decision.outcome == "visit":
            visit_prediction_count += len(predicted)
            prediction_set = set(predicted)
            hits += sum(url in prediction_set for url in decision.authoritative_urls)
        else:
            no_visit_prediction_count += len(predicted)
    return {
        "search_decisions": len(decisions),
        "outcomes": dict(sorted(outcomes.items())),
        "blind_top5_predictions": prediction_count,
        "visit_window_predictions": visit_prediction_count,
        "non_visit_window_predictions": no_visit_prediction_count,
        "exact_target_hits": hits,
        "all_window_precision": hits / prediction_count if prediction_count else 0.0,
        "wasted_predictions": prediction_count - hits,
        "scope_note": (
            "A calibrated next-tool visit/abstain gate is not implemented; this "
            "audit assumes the conditional reranker is blindly invoked after every search."
        ),
    }


def benchmark_online_prediction(
    mapper: ContextualURLReranker,
    sessions: Sequence[SessionTrace],
    *,
    passes: int = 20,
) -> dict[str, Any]:
    """Benchmark the live structured-result path, excluding fit and network time."""

    if passes <= 0:
        raise ValueError("benchmark passes must be positive")
    predictor = ContextualTraceVisitPredictor(reranker=mapper, top_k=5)
    inputs = [
        {
            "tool": "search",
            "results": [
                {
                    "url": result.url,
                    "rank": result.result_rank,
                    "query_index": result.query_index,
                    "query": result.query,
                    "title": result.title,
                    "snippet": result.snippet,
                }
                for result in search_results
            ],
        }
        for search_results in extract_visible_search_inputs(sessions)
    ]
    for item in inputs:
        predictor.predict_structured_result(item)
    durations_ms: list[float] = []
    for _ in range(passes):
        for item in inputs:
            started_ns = time.perf_counter_ns()
            predictor.predict_structured_result(item)
            durations_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
    ordered = sorted(durations_ms)
    p99_ms = percentile(ordered, 0.99)
    maximum_ms = max(ordered) if ordered else 0.0
    threshold_ms = 100.0
    return {
        "scope": (
            "structured adapter + feature extraction + exact raw-URL dedup + "
            "49-dimensional scoring + sort + Top-5 output"
        ),
        "excludes": "offline fit, search execution, network, and scheduler time",
        "search_decision_inputs": len(inputs),
        "input_rows_min": min(len(item["results"]) for item in inputs) if inputs else 0,
        "input_rows_max": max(len(item["results"]) for item in inputs) if inputs else 0,
        "warmup_calls": len(inputs),
        "passes": passes,
        "measured_calls": len(ordered),
        "mean_ms": statistics.fmean(ordered) if ordered else 0.0,
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": p99_ms,
        "max_ms": maximum_ms,
        "acceptance_threshold_ms": threshold_ms,
        "p99_pass": p99_ms < threshold_ms,
        "max_pass": maximum_ms < threshold_ms,
        "accepted": p99_ms < threshold_ms and maximum_ms < threshold_ms,
        "timer": "time.perf_counter_ns",
    }


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_session_inference(
    baseline_rows: Sequence[dict[str, Any]],
    contextual_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(row["decision_id"]): row for row in baseline_rows}
    contextual_by_id = {str(row["decision_id"]): row for row in contextual_rows}
    if baseline_by_id.keys() != contextual_by_id.keys():
        raise ValueError("paired outer predictions do not cover identical decisions")
    session_values: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "targets": 0,
            "baseline": {top_k: 0 for top_k in TOP_KS},
            "contextual": {top_k: 0 for top_k in TOP_KS},
        }
    )
    for decision_id, baseline in baseline_by_id.items():
        contextual = contextual_by_id[decision_id]
        session_id = str(baseline["session_id"])
        session_values[session_id]["targets"] += int(baseline["target_count"])
        for top_k in TOP_KS:
            session_values[session_id]["baseline"][top_k] += int(
                baseline[f"hits_at_{top_k}"]
            )
            session_values[session_id]["contextual"][top_k] += int(
                contextual[f"hits_at_{top_k}"]
            )

    session_ids = sorted(session_values)
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap: dict[int, list[float]] = {top_k: [] for top_k in TOP_KS}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [rng.choice(session_ids) for _ in session_ids]
        denominator = sum(session_values[session_id]["targets"] for session_id in selected)
        for top_k in TOP_KS:
            baseline_hits = sum(
                session_values[session_id]["baseline"][top_k]
                for session_id in selected
            )
            contextual_hits = sum(
                session_values[session_id]["contextual"][top_k]
                for session_id in selected
            )
            bootstrap[top_k].append(
                (contextual_hits - baseline_hits) / denominator
                if denominator
                else 0.0
            )

    result: dict[str, Any] = {
        "unit": "whole transition-bearing held-out session",
        "session_count": len(session_ids),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "top_k": {},
    }
    for top_k in TOP_KS:
        target_total = sum(values["targets"] for values in session_values.values())
        baseline_total = sum(
            values["baseline"][top_k] for values in session_values.values()
        )
        contextual_total = sum(
            values["contextual"][top_k] for values in session_values.values()
        )
        values = sorted(bootstrap[top_k])
        deltas = [
            session_values[session_id]["contextual"][top_k]
            - session_values[session_id]["baseline"][top_k]
            for session_id in session_ids
        ]
        observed_delta_hits = sum(deltas)
        nonzero = [value for value in deltas if value]
        if nonzero:
            extreme = 0
            total_permutations = 2 ** len(nonzero)
            for signs in itertools.product((-1, 1), repeat=len(nonzero)):
                permuted = sum(sign * value for sign, value in zip(signs, nonzero))
                extreme += abs(permuted) >= abs(observed_delta_hits)
            permutation_p = extreme / total_permutations
        else:
            total_permutations = 1
            permutation_p = 1.0
        result["top_k"][str(top_k)] = {
            "baseline_hits": baseline_total,
            "contextual_hits": contextual_total,
            "delta_hits": observed_delta_hits,
            "delta_target_recall": (
                observed_delta_hits / target_total if target_total else 0.0
            ),
            "cluster_bootstrap_95_percentile_interval": [
                percentile(values, 0.025),
                percentile(values, 0.975),
            ],
            "cluster_bootstrap_probability_delta_gt_zero": (
                sum(value > 0 for value in values) / len(values)
            ),
            "paired_session_sign_flip_two_sided_p": permutation_p,
            "sign_flip_permutations": total_permutations,
        }
    return result


def write_predictions_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "model",
        "fold",
        "session_id",
        "decision_id",
        "candidate_count",
        "target_count",
        "targets_json",
        "predictions_json",
        "hits_at_1",
        "hits_at_3",
        "hits_at_5",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields if not field.endswith("_json")},
                    "targets_json": canonical_json(row["targets"]),
                    "predictions_json": canonical_json(row["predictions"]),
                }
            )


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_cell(metrics: dict[str, Any], top_k: int) -> str:
    item = metrics["top_k"][str(top_k)]
    return f"{item['hits']}/{metrics['targets']} ({percent(item['target_recall'])})"


def promotion_decision(
    baseline: dict[str, Any],
    contextual: dict[str, Any],
    overhead: dict[str, Any],
) -> dict[str, Any]:
    top1_noninferior = (
        contextual["top_k"]["1"]["hits"] >= baseline["top_k"]["1"]["hits"]
    )
    top3_noninferior = (
        contextual["top_k"]["3"]["hits"] >= baseline["top_k"]["3"]["hits"]
    )
    top5_strictly_better = (
        contextual["top_k"]["5"]["hits"] > baseline["top_k"]["5"]["hits"]
    )
    accepted = bool(
        overhead["accepted"]
        and top1_noninferior
        and top3_noninferior
        and top5_strictly_better
    )
    return {
        "rule": (
            "online p99 and observed max <100ms; outer Top-5 hits strictly "
            "greater than M0; outer Top-1 and Top-3 hits each no lower than M0"
        ),
        "latency_gate": bool(overhead["accepted"]),
        "top1_noninferior": top1_noninferior,
        "top3_noninferior": top3_noninferior,
        "top5_strictly_better": top5_strictly_better,
        "accepted": accepted,
        "selected_model": "M1_contextual" if accepted else "M0_rank_only",
    }


def render_report(payload: dict[str, Any]) -> str:
    oof = payload["outer_train_grouped_oof"]
    outer = payload["outer_heldout"]
    baseline = outer["M0_rank_only"]["conditional_metrics"]
    contextual = outer["M1_contextual"]["conditional_metrics"]
    inference = payload["paired_session_inference"]
    ceiling = payload["candidate_ceiling"]
    base_stall = outer["M0_rank_only"]["trace_latency"]
    ctx_stall = outer["M1_contextual"]["trace_latency"]
    overhead = payload["online_overhead_benchmark"]
    promotion = payload["model"]["promotion_decision"]
    target_total = baseline["targets"]
    lines = [
        "# Contextual exact-URL predictor optimization",
        "",
        "## Result",
        "",
        (
            "The frozen current-response contextual reranker improves the grouped "
            "training evidence and was evaluated once on the unchanged whole-session "
            "outer split. It remains a **conditional URL reranker**: these Top-K "
            "metrics include only search decisions whose next tool is an authoritative visit."
        ),
        "",
        f"The predeclared promotion gate **{'accepted M1' if promotion['accepted'] else 'rejected M1 and retained M0'}**.",
        "",
        f"### Fixed outer heldout ({target_total} authoritative URL targets)",
        "",
        "| Model | Top-1 | Top-3 | Top-5 |",
        "|---|---:|---:|---:|",
        f"| M0 rank-only | {metric_cell(baseline, 1)} | {metric_cell(baseline, 3)} | {metric_cell(baseline, 5)} |",
        f"| M1 contextual | {metric_cell(contextual, 1)} | {metric_cell(contextual, 3)} | {metric_cell(contextual, 5)} |",
        "",
        "Paired changes by whole held-out session:",
        "",
        "| K | Delta hits | Delta recall | Session-bootstrap 95% interval | Sign-flip p |",
        "|---:|---:|---:|---:|---:|",
    ]
    for top_k in TOP_KS:
        item = inference["top_k"][str(top_k)]
        interval = item["cluster_bootstrap_95_percentile_interval"]
        lines.append(
            f"| {top_k} | {item['delta_hits']:+d} | {item['delta_target_recall'] * 100:+.1f} pp "
            f"| [{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}] pp "
            f"| {item['paired_session_sign_flip_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "This is post-hoc method development with a mechanically isolated "
                "outer run, not a pristine confirmatory test: the old baseline and "
                "aggregate error audit were already visible. A newly collected "
                "whole-session trace set is required for confirmation."
            ),
            "",
            "## Training-only selection",
            "",
            (
                "Five-fold CV was grouped by session inside the original 70 outer-train "
                "sessions. The model and `lambda=3` were frozen before the contextual "
                "outer evaluation."
            ),
            "",
            "| Model | Top-1 | Top-3 | Top-5 |",
            "|---|---:|---:|---:|",
            f"| M0 rank-only | {metric_cell(oof['M0_rank_only'], 1)} | {metric_cell(oof['M0_rank_only'], 3)} | {metric_cell(oof['M0_rank_only'], 5)} |",
            f"| M1 contextual | {metric_cell(oof['M1_contextual'], 1)} | {metric_cell(oof['M1_contextual'], 3)} | {metric_cell(oof['M1_contextual'], 5)} |",
            "",
            (
                "M1 uses only data visible before the decision LLM starts generating: "
                "displayed rank, query block/position, duplicate appearances, title-query "
                "and path-query overlap, path shape, host multiplicity, and PDF suffix. "
                "It emits the original raw URL and keeps exact invocation confirmation unchanged."
            ),
            "",
            "## Where the remaining headroom is",
            "",
            f"- Current-response exact coverage: `{ceiling['current_response']['covered_targets']}/{target_total} = {percent(ceiling['current_response']['target_coverage'])}`.",
            f"- Current-response Top-5 oracle: `{ceiling['current_response']['oracle_hits']['5']}/{target_total} = {percent(ceiling['current_response']['oracle_recall']['5'])}`.",
            f"- All causally prior search responses in the decision input cover `{ceiling['full_visible_search_context']['covered_targets']}/{target_total} = {percent(ceiling['full_visible_search_context']['target_coverage'])}`; their Top-5 oracle is `{ceiling['full_visible_search_context']['oracle_hits']['5']}/{target_total} = {percent(ceiling['full_visible_search_context']['oracle_recall']['5'])}`.",
            (
                "- A bounded recency-aware history cache remains a separate challenger; "
                "this frozen M1 ranks only the current response."
            ),
            "",
            "## Runtime interpretation",
            "",
            f"The end-to-end local prediction path took `{overhead['p50_ms']:.3f} ms` p50, "
            f"`{overhead['p95_ms']:.3f} ms` p95, `{overhead['p99_ms']:.3f} ms` p99, "
            f"and `{overhead['max_ms']:.3f} ms` maximum over {overhead['measured_calls']:,} calls. "
            f"The frozen `<100 ms` p99-and-maximum gate therefore **{'passed' if overhead['accepted'] else 'failed'}**.",
            "",
            (
                f"At Top-5, trace-counterfactual exposed stall changes from "
                f"`{base_stall['baseline_exposed_tool_stall_s']:.3f}s → {base_stall['optimized_exposed_tool_stall_s']:.3f}s` "
                f"for M0 and `{ctx_stall['baseline_exposed_tool_stall_s']:.3f}s → {ctx_stall['optimized_exposed_tool_stall_s']:.3f}s` "
                f"for M1. The corresponding reductions are "
                f"`{percent(base_stall['stall_reduction'])}` and "
                f"`{percent(ctx_stall['stall_reduction'])}`."
            ),
            "",
            "The autonomous all-search audit is intentionally harsher:",
            "",
            "| Model | Search decisions | Blind Top-5 predictions | Exact hits | All-window precision | Non-visit predictions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, label in (("M0_rank_only", "M0"), ("M1_contextual", "M1")):
        audit = outer[model_name]["all_search_decision_audit"]
        lines.append(
            f"| {label} | {audit['search_decisions']} | {audit['blind_top5_predictions']} "
            f"| {audit['exact_target_hits']} | {percent(audit['all_window_precision'])} "
            f"| {audit['non_visit_window_predictions']} |"
        )
    lines.extend(
        [
            "",
            (
                "A next-tool `visit`/abstain gate is therefore required before deploying "
                "this as an autonomous post-search policy. The pairwise relative score is "
                "not a calibrated admission probability."
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "python reproduction/scripts/run_predictor_optimization.py \\",
            "  --output /tmp/predictor_optimization_reproduction \\",
            "  --protocol reproduction/results/predictor_optimization/FROZEN_PROTOCOL.md",
            "```",
            "",
            (
                "The model artifact, aggregate JSON, paired prediction CSV, provenance, "
                "and completion manifest are in this directory. Promotion remains exact "
                "raw `visit({url: ...})` equality; no HTTP/HTTPS or encoding equivalence is assumed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
        help="frozen protocol to bind (allows reproduction into a fresh output directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    started_path = output / "OUTER_EVALUATION_STARTED.json"
    completion_path = output / "OUTER_EVALUATION_COMPLETE.json"
    if started_path.exists() or completion_path.exists():
        raise SystemExit(
            f"outer evaluation was already started at {started_path}; "
            "use a fresh --output directory for deterministic reproduction"
        )
    generated_names = (
        "contextual_url_reranker.json",
        "baseline_url_rank_mapper.json",
        "metrics.json",
        "predictions.csv",
        "REPORT.md",
        "provenance.json",
    )
    existing_generated = [name for name in generated_names if (output / name).exists()]
    if existing_generated:
        raise SystemExit(
            f"output contains prior generated files without a valid STARTED marker: "
            f"{existing_generated}"
        )
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_file():
        raise SystemExit(f"frozen protocol is missing: {protocol_path}")

    sessions = load_sessions(args.traces)
    train_sessions, heldout_sessions = split_sessions(
        sessions,
        train_ratio=OUTER_TRAIN_RATIO,
        seed=OUTER_SEED,
    )
    train_transitions = transitions_from_sessions(train_sessions)

    oof_metrics, oof_rows = aggregate_oof(train_sessions)
    expected_oof = {
        "M0_rank_only": {"1": 33, "3": 64, "5": 81},
        "M1_contextual": {"1": 38, "3": 71, "5": 90},
    }
    for model_name, expected in expected_oof.items():
        observed = {
            top_k: oof_metrics[model_name]["top_k"][top_k]["hits"]
            for top_k in ("1", "3", "5")
        }
        if observed != expected:
            raise RuntimeError(
                f"frozen OOF guard failed for {model_name}: {observed} != {expected}"
            )

    baseline = URLRankMapper().fit(
        train_transitions,
        searches_seen=count_tool_calls(train_sessions, "search"),
    )
    contextual = ContextualURLReranker(l2=DEFAULT_L2).fit(train_transitions)
    split_manifest = build_split_manifest(
        tuple(train_sessions),
        tuple(heldout_sessions),
        seed=OUTER_SEED,
        train_ratio=OUTER_TRAIN_RATIO,
    )
    contextual_artifact = contextual.to_artifact(split_manifest)
    baseline_artifact = baseline.to_artifact(split_manifest)
    observed_split_checksum = contextual_artifact["training_split"][
        "manifest_sha256"
    ]
    if observed_split_checksum != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(
            "fixed outer split checksum changed before evaluation: "
            f"{observed_split_checksum} != {EXPECTED_SPLIT_MANIFEST_SHA256}"
        )
    if baseline_artifact["artifact_sha256"] != EXPECTED_BASELINE_ARTIFACT_SHA256:
        raise RuntimeError(
            "legacy baseline artifact changed before evaluation: "
            f"{baseline_artifact['artifact_sha256']} != "
            f"{EXPECTED_BASELINE_ARTIFACT_SHA256}"
        )
    overhead_benchmark = benchmark_online_prediction(
        contextual, train_sessions, passes=20
    )
    if not overhead_benchmark["accepted"]:
        raise RuntimeError(
            "frozen online-overhead acceptance gate failed before outer accuracy "
            f"evaluation: {overhead_benchmark}"
        )

    code_paths = [
        Path(__file__).resolve(),
        REPRODUCTION_ROOT / "paste_repro" / "analysis.py",
        REPRODUCTION_ROOT / "paste_repro" / "contextual_mapper.py",
        REPRODUCTION_ROOT / "paste_repro" / "invocation.py",
        REPRODUCTION_ROOT / "paste_repro" / "mapper.py",
        REPRODUCTION_ROOT / "paste_repro" / "online_learned_agent.py",
        REPRODUCTION_ROOT / "paste_repro" / "pipeline.py",
        REPRODUCTION_ROOT / "paste_repro" / "tool_prediction.py",
        REPRODUCTION_ROOT / "paste_repro" / "traces.py",
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "scripts" / "run_online_trace_learned_experiment.py",
        protocol_path,
    ]
    trace_files = [
        {
            "path": str(session.path.resolve()),
            "sha256": sha256_file(session.path),
        }
        for session in sessions
    ]
    def current_stable_binding() -> dict[str, Any]:
        return {
            "frozen_protocol_sha256": sha256_file(protocol_path),
            "code_and_protocol": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in code_paths
            ],
            "trace_files": [
                {
                    "path": str(session.path.resolve()),
                    "sha256": sha256_file(session.path),
                }
                for session in sessions
            ],
            "outer_split_manifest_sha256": observed_split_checksum,
            "outer_train_oof_sha256": sha256_json(oof_metrics),
            "challenger_artifact_sha256": contextual_artifact["artifact_sha256"],
            "baseline_artifact_sha256": baseline_artifact["artifact_sha256"],
        }

    stable_binding = current_stable_binding()
    started = {
        "schema": "paste_repro.predictor_outer_started",
        "version": 1,
        "status": "started_before_any_outer_label_scoring",
        "stable_binding": stable_binding,
        "pre_outer_train_input_overhead_benchmark": overhead_benchmark,
        "manifest_sha256": "",
    }
    started["manifest_sha256"] = sha256_json(
        {key: value for key, value in started.items() if key != "manifest_sha256"}
    )
    try:
        write_json_exclusive(started_path, started)
    except FileExistsError as exc:
        raise RuntimeError(
            "another process already claimed the one-shot outer evaluation"
        ) from exc

    # The one-shot marker now exists. Held-out labels are scored only below.
    heldout_transitions = transitions_from_sessions(heldout_sessions)
    baseline_metrics, baseline_rows = evaluate_predictions(
        baseline,
        heldout_transitions,
        split="outer_heldout",
        model_name="M0_rank_only",
        fold=None,
    )
    contextual_metrics, contextual_rows = evaluate_predictions(
        contextual,
        heldout_transitions,
        split="outer_heldout",
        model_name="M1_contextual",
        fold=None,
    )

    baseline_latency = evaluate_held_out(
        baseline, heldout_transitions, top_ks=TOP_KS, latency_top_k=5
    )
    contextual_latency = evaluate_held_out(
        contextual, heldout_transitions, top_ks=TOP_KS, latency_top_k=5
    )
    promotion = promotion_decision(
        baseline_metrics, contextual_metrics, overhead_benchmark
    )

    payload: dict[str, Any] = {
        "schema": "paste_repro.predictor_optimization",
        "version": 1,
        "evaluation_status": (
            "post-hoc method development with mechanically isolated outer evaluation"
        ),
        "split": {
            "seed": OUTER_SEED,
            "train_ratio": OUTER_TRAIN_RATIO,
            "total_sessions": len(sessions),
            "train_sessions": len(train_sessions),
            "heldout_sessions": len(heldout_sessions),
            "train_decision_windows": len(train_transitions),
            "heldout_decision_windows": len(heldout_transitions),
            "train_targets": sum(len(item.authoritative_urls) for item in train_transitions),
            "heldout_targets": sum(
                len(item.authoritative_urls) for item in heldout_transitions
            ),
        },
        "outer_train_grouped_oof": oof_metrics,
        "outer_heldout": {
            "M0_rank_only": {
                "conditional_metrics": baseline_metrics,
                "trace_latency": baseline_latency,
                "all_search_decision_audit": all_search_decision_audit(
                    baseline, heldout_sessions
                ),
            },
            "M1_contextual": {
                "conditional_metrics": contextual_metrics,
                "trace_latency": contextual_latency,
                "all_search_decision_audit": all_search_decision_audit(
                    contextual, heldout_sessions
                ),
            },
        },
        "candidate_ceiling": {
            "current_response": candidate_ceiling(heldout_transitions),
            "full_visible_search_context": candidate_ceiling(
                heldout_transitions, use_full_visible_search_context=True
            ),
        },
        "paired_session_inference": paired_session_inference(
            baseline_rows, contextual_rows
        ),
        "online_overhead_benchmark": overhead_benchmark,
        "model": {
            "selected": promotion["selected_model"],
            "challenger_retained": promotion["accepted"],
            "promotion_decision": promotion,
            "selection_metric": "outer-train pooled grouped-OOF exact target Recall@5",
            "eligibility_gate": "Recall@1 and Recall@3 no lower than M0",
            "baseline": {
                "name": "M0_rank_only",
                "summary": baseline.summary(),
                "artifact_sha256": baseline_artifact["artifact_sha256"],
            },
            "challenger": {
                "name": "M1_contextual",
                "summary": contextual.summary(),
                "artifact_sha256": contextual_artifact["artifact_sha256"],
            },
        },
        "scope": {
            "conditional_on_next_tool_visit": True,
            "next_tool_gate_implemented": False,
            "output": "raw URL from current visible search response",
            "confirmation": "exact raw visit invocation equality",
            "future_decision_response_used": False,
            "network_requests": False,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "contextual_url_reranker.json"
    baseline_model_path = output / "baseline_url_rank_mapper.json"
    metrics_path = output / "metrics.json"
    predictions_path = output / "predictions.csv"
    report_path = output / "REPORT.md"
    provenance_path = output / "provenance.json"
    save_contextual_artifact(model_path, contextual_artifact)
    save_artifact(baseline_model_path, baseline_artifact)
    write_json_atomic(metrics_path, payload)
    write_predictions_csv(
        predictions_path,
        [*oof_rows, *baseline_rows, *contextual_rows],
    )
    report_path.write_text(render_report(payload), encoding="utf-8")

    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "trace_directory": str(args.traces.resolve()),
        "trace_files": trace_files,
        "code_and_protocol": [
            {"path": str(path), "sha256": sha256_file(path)} for path in code_paths
        ],
        "outer_started": {
            "path": str(started_path),
            "file_sha256": sha256_file(started_path),
            "manifest_sha256": started["manifest_sha256"],
        },
        "outputs": {},
    }
    write_json_atomic(provenance_path, provenance)
    output_paths = [
        model_path,
        baseline_model_path,
        metrics_path,
        predictions_path,
        report_path,
    ]
    provenance["outputs"] = {
        path.name: sha256_file(path) for path in output_paths
    }
    write_json_atomic(provenance_path, provenance)

    if current_stable_binding() != stable_binding:
        raise RuntimeError(
            "code, protocol, or traces changed after the outer STARTED binding; "
            "refusing to write a completion marker"
        )

    completion = {
        "schema": "paste_repro.predictor_outer_completion",
        "version": 1,
        "completed": True,
        "outer_started_file_sha256": sha256_file(started_path),
        "outer_started_manifest_sha256": started["manifest_sha256"],
        "frozen_protocol_sha256": sha256_file(protocol_path),
        "challenger_model_artifact_sha256": contextual_artifact["artifact_sha256"],
        "baseline_model_artifact_sha256": baseline_artifact["artifact_sha256"],
        "promotion_decision": promotion,
        "selected_model": promotion["selected_model"],
        "selected_model_artifact_sha256": (
            contextual_artifact["artifact_sha256"]
            if promotion["accepted"]
            else baseline_artifact["artifact_sha256"]
        ),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": sha256_file(predictions_path),
        "report_sha256": sha256_file(report_path),
        "provenance_sha256": sha256_file(provenance_path),
        "outer_result_was_not_used_for_model_changes": True,
        "manifest_sha256": "",
    }
    completion["manifest_sha256"] = sha256_json(
        {key: value for key, value in completion.items() if key != "manifest_sha256"}
    )
    write_json_atomic(completion_path, completion)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
