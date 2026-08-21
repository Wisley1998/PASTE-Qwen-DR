"""Held-out, trace-driven evaluation for the tool-only core."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .mapper import URLRankMapper
from .traces import SearchVisitTransition


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_held_out(
    mapper: URLRankMapper,
    transitions: Iterable[SearchVisitTransition],
    *,
    top_ks: Sequence[int] = (1, 3, 5),
    latency_top_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate concrete URL invocations and replay-derived exposed stall.

    The optimized estimate is intentionally bounded by both the correctly
    predicted URL fraction and the preceding LLM inference window.  A miss can
    never improve (or worsen) the baseline in this tool-only calculation.
    """

    normalized_ks = tuple(sorted(set(int(value) for value in top_ks)))
    if not normalized_ks or any(value <= 0 for value in normalized_ks):
        raise ValueError("top_ks must contain positive integers")
    selected_k = max(normalized_ks) if latency_top_k is None else int(latency_top_k)
    if selected_k <= 0:
        raise ValueError("latency_top_k must be positive")
    max_k = max(max(normalized_ks), selected_k)
    transition_list = tuple(transitions)

    total_targets = sum(len(item.authoritative_urls) for item in transition_list)
    executable_targets = 0
    executable_examples = 0
    metric_counts = {
        value: {
            "example_hits": 0,
            "invocation_hits": 0,
            "predictions": 0,
        }
        for value in normalized_ks
    }
    baseline_stall_s = 0.0
    saved_time_s = 0.0
    selected_example_hits = 0

    for transition in transition_list:
        target_urls = transition.authoritative_urls
        candidate_urls = {result.url for result in transition.search_results}
        covered = sum(url in candidate_urls for url in target_urls)
        executable_targets += covered
        executable_examples += covered > 0

        predictions = mapper.predict(transition.search_results, max_k)
        predicted_urls = [prediction.invocation.arguments["url"] for prediction in predictions]
        for value in normalized_ks:
            prediction_set = set(predicted_urls[:value])
            hits = sum(url in prediction_set for url in target_urls)
            metric_counts[value]["example_hits"] += hits > 0
            metric_counts[value]["invocation_hits"] += hits
            metric_counts[value]["predictions"] += min(value, len(predicted_urls))

        selected_set = set(predicted_urls[:selected_k])
        selected_hits = sum(url in selected_set for url in target_urls)
        selected_example_hits += selected_hits > 0
        baseline_stall_s += transition.baseline_stall_s
        hit_fraction = _safe_ratio(selected_hits, len(target_urls))
        hidden_window_s = min(
            transition.baseline_stall_s, transition.overlap_window_s
        )
        saved_time_s += hidden_window_s * hit_fraction

    top_k_metrics: dict[str, Any] = {}
    for value in normalized_ks:
        counts = metric_counts[value]
        top_k_metrics[str(value)] = {
            "example_hits": counts["example_hits"],
            "example_hit_rate": _safe_ratio(
                counts["example_hits"], len(transition_list)
            ),
            "invocation_hits": counts["invocation_hits"],
            "invocation_hit_rate": _safe_ratio(
                counts["invocation_hits"], total_targets
            ),
            "concrete_predictions": counts["predictions"],
            "prediction_precision": _safe_ratio(
                counts["invocation_hits"], counts["predictions"]
            ),
        }

    optimized_stall_s = max(0.0, baseline_stall_s - saved_time_s)
    return {
        "held_out_sessions": len({item.session_id for item in transition_list}),
        "search_visit_examples": len(transition_list),
        "authoritative_url_invocations": total_targets,
        "executable": {
            "covered_invocations": executable_targets,
            "invocation_coverage": _safe_ratio(executable_targets, total_targets),
            "covered_examples": executable_examples,
            "example_coverage": _safe_ratio(executable_examples, len(transition_list)),
        },
        "top_k_concrete_invocation_hit": top_k_metrics,
        "latency_top_k": selected_k,
        "latency_hit_examples": selected_example_hits,
        "baseline_exposed_tool_stall_s": baseline_stall_s,
        "optimized_exposed_tool_stall_s": optimized_stall_s,
        "saved_time_s": saved_time_s,
        "stall_reduction": _safe_ratio(saved_time_s, baseline_stall_s),
        "latency_model": (
            "per transition: min(visit stall, preceding LLM inference) "
            "x exact URL-invocation hit fraction"
        ),
    }

