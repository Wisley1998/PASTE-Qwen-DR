"""Semantic-boundary replay for agent-oriented serving baselines.

This module deliberately does not emulate Murakkab, llm-d, or Dynamo.  It
replays the event-order contract shared by dependency-ready workflow schedulers
and arrived-request inference schedulers: an exact external-tool invocation is
not schedulable until its producing LLM call has completed.  PASTE crosses that
boundary by predicting bounded concrete invocations from already-visible state,
executing them in isolation, and committing only an exact authoritative match.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
from pathlib import Path
import subprocess
from typing import Any

from .mapper import URLRankMapper
from .pipeline import run_speculative_tool_execution, train_and_analyze
from .traces import SearchVisitTransition


SCHEMA = "paste_repro.agent_baseline_boundary"
VERSION = 1
DEFAULT_SPEEDUPS = (1.0, 1.25, 1.5, 2.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _prediction_counts(
    mapper: URLRankMapper,
    transitions: Iterable[SearchVisitTransition],
    *,
    top_k: int,
) -> tuple[int, int]:
    predictions = 0
    exact_hits = 0
    for transition in transitions:
        predicted = mapper.predict(transition.search_results, top_k)
        predicted_keys = {item.invocation.key for item in predicted}
        predictions += len(predicted)
        exact_hits += sum(
            invocation.key in predicted_keys
            for invocation in transition.authoritative_invocations
        )
    return predictions, exact_hits


def inference_speedup_counterfactual(
    mapper: URLRankMapper,
    transitions: Iterable[SearchVisitTransition],
    *,
    top_k: int,
    speedups: Sequence[float] = DEFAULT_SPEEDUPS,
) -> dict[str, Any]:
    """Hold tool times fixed while ideally scaling only decision-LLM time.

    The output is a decomposition of the direct ``search -> LLM -> visit``
    segments, not a throughput model for any named system.  It is intentionally
    favorable to an inference-serving baseline: there is no queueing penalty,
    cache miss, routing overhead, or loss of model quality.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    normalized_speedups = tuple(float(value) for value in speedups)
    if not normalized_speedups or any(value <= 0 for value in normalized_speedups):
        raise ValueError("speedups must contain positive values")

    transition_list = tuple(transitions)
    observed_decision_s = sum(item.overlap_window_s for item in transition_list)
    observed_tool_stall_s = sum(item.baseline_stall_s for item in transition_list)
    rows: list[dict[str, Any]] = []

    for speedup in normalized_speedups:
        scaled_decision_s = observed_decision_s / speedup
        hidden_tool_stall_s = 0.0
        for transition in transition_list:
            predictions = mapper.predict(transition.search_results, top_k)
            predicted_keys = {item.invocation.key for item in predictions}
            authoritative = transition.authoritative_invocations
            exact_hits = sum(item.key in predicted_keys for item in authoritative)
            hit_fraction = exact_hits / len(authoritative)
            hidden_tool_stall_s += min(
                transition.baseline_stall_s,
                transition.overlap_window_s / speedup,
            ) * hit_fraction

        demand_only_segment_s = scaled_decision_s + observed_tool_stall_s
        paste_exposed_tool_stall_s = max(
            0.0, observed_tool_stall_s - hidden_tool_stall_s
        )
        paste_segment_s = scaled_decision_s + paste_exposed_tool_stall_s
        rows.append(
            {
                "idealized_inference_speedup": speedup,
                "scaled_decision_generation_s": scaled_decision_s,
                "demand_only_exposed_external_tool_stall_s": observed_tool_stall_s,
                "demand_only_segment_s": demand_only_segment_s,
                "paste_hidden_external_tool_stall_s": hidden_tool_stall_s,
                "paste_exposed_external_tool_stall_s": paste_exposed_tool_stall_s,
                "paste_segment_s": paste_segment_s,
                "incremental_paste_segment_saving_s": hidden_tool_stall_s,
                "incremental_paste_segment_reduction": (
                    hidden_tool_stall_s / demand_only_segment_s
                    if demand_only_segment_s
                    else 0.0
                ),
            }
        )

    return {
        "scope": "sum of held-out direct search -> decision LLM -> visit segments",
        "assumptions": [
            "only decision-LLM generation time is divided by the speedup factor",
            "external-tool service and wait times are held at their observed values",
            "there is no inference routing, queueing, cache, or quality penalty",
            (
                "PASTE saves at most min(tool stall, scaled decision window) "
                "times the exact-hit fraction"
            ),
        ],
        "observed_decision_generation_s": observed_decision_s,
        "observed_demand_only_external_tool_stall_s": observed_tool_stall_s,
        "zero_decision_time_demand_only_lower_bound_s": observed_tool_stall_s,
        "rows": rows,
    }


async def run_agent_baseline_boundary(
    trace_directory: str | Path | None = None,
    *,
    seed: str = "paste-repro-v1",
    train_ratio: float = 0.70,
    top_k: int = 5,
    max_concurrency: int = 4,
    speedups: Sequence[float] = DEFAULT_SPEEDUPS,
) -> dict[str, Any]:
    """Execute the broker replay and emit the conservative boundary analysis."""

    mechanism = await run_speculative_tool_execution(
        trace_directory,
        seed=seed,
        train_ratio=train_ratio,
        top_k=top_k,
        max_concurrency=max_concurrency,
    )
    analysis, mapper, transitions, _, _ = train_and_analyze(
        trace_directory,
        seed=seed,
        train_ratio=train_ratio,
        top_k=top_k,
    )
    evaluation = analysis["held_out_evaluation"]
    prediction_count, exact_hits = _prediction_counts(
        mapper, transitions, top_k=top_k
    )
    scheduler = {
        key: value
        for key, value in mechanism["scheduler"].items()
        # This wall-clock value comes from a zero-delay executor and is not a
        # latency result.  The trace-derived counterfactual below is the timing
        # calculation that is meaningful for this experiment.
        if key != "saved_time_s"
    }

    promotion_hits = scheduler["completed_reuse"] + scheduler["inflight_promotions"]
    if exact_hits != promotion_hits:
        raise AssertionError(
            f"exact-hit reconciliation failed: {exact_hits} != {promotion_hits}"
        )
    if prediction_count != scheduler["admitted"]:
        raise AssertionError(
            f"prediction reconciliation failed: {prediction_count} != "
            f"{scheduler['admitted']}"
        )

    repository_root = Path(__file__).resolve().parents[2]
    entrypoint = (
        repository_root
        / "reproduction"
        / "scripts"
        / "run_agent_baseline_boundary_replay.py"
    )
    sources = {
        "murakkab": {
            "official_paper_page": (
                "https://www.usenix.org/conference/osdi26/presentation/chaudhry"
            ),
            "official_final_paper": "https://www.usenix.org/system/files/osdi26-chaudhry.pdf",
        },
        "llm_d": {
            "official_agentic_serving": (
                "https://llm-d.ai/docs/well-lit-paths/workloads/agentic-serving"
            ),
            "official_router": "https://llm-d.ai/docs/dev/architecture/core/router",
            "official_scheduling": "https://llm-d.ai/docs/architecture/core/router/epp/scheduling",
        },
        "nvidia_dynamo": {
            "official_agent_overview": "https://docs.nvidia.com/dynamo/dev/agents/overview",
            "official_agent_hints": "https://docs.nvidia.com/dynamo/agents/agent-hints",
            "official_agentic_inference_digest": (
                "https://docs.nvidia.com/dynamo/dev/digest/agentic-inference"
            ),
            "official_thunderagent": (
                "https://docs.nvidia.com/dynamo/dev/agents/"
                "thunder-agent-program-scheduler"
            ),
            "official_agent_simulation": (
                "https://docs.nvidia.com/dynamo/dev/agents/agent-simulation"
            ),
        },
    }

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "experiment_kind": "executed broker replay plus semantic-boundary counterfactual",
        "provenance": {
            "git_head": _git_head(repository_root),
            "module": str(Path(__file__).resolve().relative_to(repository_root)),
            "module_sha256": _sha256_file(Path(__file__)),
            "entrypoint": str(entrypoint.relative_to(repository_root)),
            "entrypoint_sha256": _sha256_file(entrypoint) if entrypoint.is_file() else None,
            "model_artifact_sha256": mechanism["model_artifact"]["artifact_sha256"],
            "trace_files": mechanism["model_artifact"]["training_split"],
        },
        "split": mechanism["split"],
        "event_order_contract": {
            "predictor_input": "current already-visible search response",
            "future_exact_invocation_producer": "the following decision LLM response",
            "ready_only_exact_invocations_eligible_before_decision_completion": 0,
            "ready_only_exact_invocations_total": evaluation[
                "authoritative_url_invocations"
            ],
            "interpretation": (
                "A dependency-ready workflow scheduler or an arrived-request inference "
                "scheduler has no exact future visit invocation to dispatch in this window."
            ),
        },
        "visible_candidate_opportunity": evaluation["executable"],
        "executed_isolated_broker_replay": {
            "replayed_examples": mechanism["replayed_examples"],
            "replayed_authoritative_invocations": mechanism[
                "replayed_authoritative_invocations"
            ],
            "top_k": top_k,
            "concrete_predictions": prediction_count,
            "exact_invocation_hits": exact_hits,
            "exact_invocation_hit_rate": (
                exact_hits / mechanism["replayed_authoritative_invocations"]
            ),
            "state_isolation_violations": mechanism[
                "state_isolation_violations"
            ],
            "scheduler": scheduler,
            "reconciliation": {
                "predictions_equal_admissions": prediction_count
                == scheduler["admitted"],
                "exact_hits_equal_completed_plus_inflight_promotions": exact_hits
                == promotion_hits,
                "commits_equal_authoritative_invocations": scheduler["commits"]
                == mechanism["replayed_authoritative_invocations"],
            },
        },
        "trace_latency_at_observed_inference": {
            "baseline_exposed_tool_stall_s": evaluation[
                "baseline_exposed_tool_stall_s"
            ],
            "paste_exposed_tool_stall_s": evaluation[
                "optimized_exposed_tool_stall_s"
            ],
            "saved_tool_stall_s": evaluation["saved_time_s"],
            "tool_stall_reduction": evaluation["stall_reduction"],
            "latency_model": evaluation["latency_model"],
        },
        "idealized_inference_substrate_counterfactual": inference_speedup_counterfactual(
            mapper,
            transitions,
            top_k=top_k,
            speedups=speedups,
        ),
        "systems_mapped_to_ready_only_boundary": {
            "murakkab": (
                "declared or per-request-composed workflow executors and their "
                "realized dependencies; this proxy does not emulate Murakkab"
            ),
            "llm_d": (
                "arrived inference requests routed using request/KV/load state; "
                "external tool control remains in the application logic layer"
            ),
            "nvidia_dynamo": (
                "arrived model requests plus agent/session hints; speculative prefill "
                "warms model state and ThunderAgent schedules at tool boundaries, "
                "but neither is an external-tool speculation executor"
            ),
        },
        "primary_sources": sources,
        "claim_boundaries": [
            "Murakkab, llm-d, and Dynamo were not installed or performance-benchmarked here.",
            (
                "The zero-ready count follows the trace event order and documented "
                "abstraction boundaries; it is not a vendor throughput result."
            ),
            (
                "The speedup factors are idealized inference-only sensitivity points, "
                "not measured performance of any named system."
            ),
            (
                "The latency calculation covers only direct search -> decision LLM -> "
                "visit segments, not end-to-end task latency."
            ),
            (
                "The result does not claim that the named systems cannot be extended; "
                "it identifies the additional predictor, isolation, exact-match "
                "promotion, and tool-side scheduler needed to obtain PASTE behavior."
            ),
        ],
    }
