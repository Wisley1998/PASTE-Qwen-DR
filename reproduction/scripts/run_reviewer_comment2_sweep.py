#!/usr/bin/env python3
"""Reproduce and stress the metrics discussed in reviewer common comment 2.

The script intentionally keeps three different metric scopes separate:

* Qwen-DR: exact URL-invocation recall on a whole-session held-out split;
* Virtual-Lab/Tongyi: exact URL-invocation recall under whole-session LOSO;
* Gemini CLI: next-tool *name* ranking on a whole-session held-out split.

It then applies one common, explicit throttling model to those frozen prediction
opportunities.  The throttling sweep is a trace replay, not a claim about GPU
throughput.  A second Qwen-only stress test invokes the repository's real
``SpeculativeScheduler`` admission and exact-confirmation implementation with
synthetic 5 ms tool service so that capacity rejection can be exercised on a
CPU-only host.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
DEFAULT_QWEN_ROOT = SCRIPT.parents[2]
DEFAULT_HOME = DEFAULT_QWEN_ROOT.parent


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(*parts: object) -> float:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) / float(2**256)


def rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class InvocationSpec:
    name: str
    arguments_json: str

    @classmethod
    def build(cls, name: str, arguments: Mapping[str, Any]) -> "InvocationSpec":
        return cls(str(name), canonical_json(dict(arguments)))

    @property
    def key(self) -> str:
        return f"{self.name}\0{self.arguments_json}"

    def arguments(self) -> dict[str, Any]:
        value = json.loads(self.arguments_json)
        if not isinstance(value, dict):
            raise ValueError("invocation arguments are not an object")
        return value


@dataclass(frozen=True)
class Opportunity:
    dataset: str
    metric_scope: str
    opportunity_id: str
    candidates: tuple[InvocationSpec, ...]
    targets: tuple[InvocationSpec, ...]
    baseline_stall_s: float | None = None
    overlap_window_s: float | None = None


def add_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def extract_qwen(
    root: Path, max_budget: int
) -> tuple[list[Opportunity], dict[str, Any], list[dict[str, str]]]:
    add_import_path(root / "reproduction")
    from paste_repro.pipeline import train_and_analyze

    report, mapper, transitions, _train, heldout = train_and_analyze(
        root / "traces" / "my_traces", top_k=max_budget
    )
    opportunities: list[Opportunity] = []
    for index, transition in enumerate(transitions):
        predictions = mapper.predict(transition.search_results, max_budget)
        candidates = tuple(
            InvocationSpec.build(
                prediction.invocation.tool_name,
                prediction.invocation.arguments,
            )
            for prediction in predictions
        )
        targets = tuple(
            InvocationSpec.build(invocation.tool_name, invocation.arguments)
            for invocation in transition.authoritative_invocations
        )
        opportunities.append(
            Opportunity(
                dataset="qwen_url_exact_heldout",
                metric_scope="exact URL invocation / authoritative URL target",
                opportunity_id=(
                    f"{transition.session_id}:search-line-{transition.search.line_number}:"
                    f"heldout-index-{index}"
                ),
                candidates=candidates,
                targets=targets,
                baseline_stall_s=transition.baseline_stall_s,
                overlap_window_s=transition.overlap_window_s,
            )
        )
    sources = [
        {"path": str(session.path.resolve()), "sha256": sha256_file(session.path)}
        for session in heldout
    ]
    return opportunities, report, sources


def _find_virtual_artifact_sources(
    root: Path, expected: Mapping[str, str]
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    by_sha: dict[str, list[Path]] = {}
    for path in sorted((root / "reproduction" / "traces").glob("**/trace.json")):
        digest = sha256_file(path)
        if digest in set(expected.values()):
            by_sha.setdefault(digest, []).append(path)
    traces: dict[str, Mapping[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for session_id, digest in sorted(expected.items()):
        matches = by_sha.get(digest, [])
        if not matches:
            raise FileNotFoundError(
                f"Virtual-Lab artifact source {digest} for {session_id} was not found"
            )
        path = matches[0]
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"Virtual-Lab trace is not an object: {path}")
        traces[session_id] = value
        sources.append({"path": str(path.resolve()), "sha256": digest})
    return traces, sources


def extract_virtual(
    root: Path, max_budget: int
) -> tuple[list[Opportunity], dict[str, Any], list[dict[str, str]]]:
    add_import_path(root / "src")
    from virtual_lab.pattern_predictor import (
        TracePatternPredictor,
        _SPECULATION_PROBABILITY_GATE,
        _call_control_state,
        _physical_record_index,
        _successful_fetch_invocations_for_call,
        _successful_outputs_for_call,
        _tool_batch,
        evaluate_leave_one_session_out,
    )

    artifact_path = root / "reproduction" / "artifacts" / "tongyi_live_predictor.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = artifact["training_split"]["session_sha256"]
    traces, sources = _find_virtual_artifact_sources(root, expected)

    official = {
        str(budget): evaluate_leave_one_session_out(traces, top_k=budget)
        for budget in sorted({1, 3, max_budget})
    }
    opportunities: list[Opportunity] = []
    for heldout_id in sorted(traces):
        predictor = TracePatternPredictor().fit(
            traces[session_id]
            for session_id in sorted(traces)
            if session_id != heldout_id
        )
        trace = traces[heldout_id]
        calls = trace.get("llm_calls", [])
        if not isinstance(calls, list):
            continue
        valid_calls = [call for call in calls if isinstance(call, Mapping)]
        physical_index = _physical_record_index(trace)
        exposed_fetch_urls: set[str] = set()
        for call_index, call in enumerate(valid_calls):
            if call_index > 0:
                for invocation in _successful_fetch_invocations_for_call(
                    trace, valid_calls, call_index - 1, physical_index
                ):
                    url = invocation.arguments.get("url")
                    if isinstance(url, str):
                        exposed_fetch_urls.add(url)
            batch = _tool_batch(call)
            search_outputs = _successful_outputs_for_call(
                trace,
                valid_calls,
                call_index,
                physical_index,
                allowed_tools={"pubmed_search", "tavily_search"},
            )
            next_control_state = (
                _call_control_state(valid_calls[call_index + 1])
                if call_index + 1 < len(valid_calls)
                else None
            )
            raw_candidates = []
            if (
                search_outputs
                and predictor.next_tool_probability(
                    batch, "web_fetch", next_control_state
                )
                >= _SPECULATION_PROBABILITY_GATE
            ):
                for _name, _call_id, output, _record in search_outputs:
                    for url in predictor.predict_urls(
                        output,
                        top_k=max_budget,
                        excluded_urls=exposed_fetch_urls,
                    ):
                        raw_candidates.append(
                            InvocationSpec.build(
                                "web_fetch",
                                predictor.predicted_web_fetch_arguments(url),
                            )
                        )
            candidates = tuple(dict.fromkeys(raw_candidates))
            actual = (
                _successful_fetch_invocations_for_call(
                    trace, valid_calls, call_index + 1, physical_index
                )
                if call_index + 1 < len(valid_calls)
                else ()
            )
            targets = tuple(
                InvocationSpec.build(invocation.tool, invocation.arguments)
                for invocation in actual
            )
            if candidates or targets:
                opportunities.append(
                    Opportunity(
                        dataset="virtual_tongyi_url_exact_loso",
                        metric_scope="exact URL invocation / physically successful fetch",
                        opportunity_id=f"{heldout_id}:llm-call-{call_index}",
                        candidates=candidates,
                        targets=targets,
                    )
                )
    return opportunities, {"artifact": artifact, "loso": official}, sources


def extract_gemini(
    root: Path, max_budget: int
) -> tuple[list[Opportunity], dict[str, Any], list[dict[str, str]]]:
    add_import_path(root / "reproduction")
    from paste_gemini.predictor import (
        SAFE_LOCAL_TOOLS,
        _name_ranker,
        evaluate_name_predictions,
    )
    from paste_gemini.traces import load_sessions, split_sessions

    trace_paths = sorted((root / "results" / "code-improved").glob("*_tool_calls.jsonl"))
    sessions = load_sessions(trace_paths)
    train, heldout, split_manifest = split_sessions(
        sessions, seed="gemini-paste-v1", train_ratio=0.7
    )
    model = _name_ranker(train, 1)
    opportunities: list[Opportunity] = []
    for session in heldout:
        history: list[str] = []
        for step in session.steps:
            if history:
                counts = model.get(tuple(history[-1:]))
                ranking = (
                    [
                        name
                        for name, _count in sorted(
                            counts.items(), key=lambda item: (-item[1], item[0])
                        )
                    ]
                    if counts
                    else []
                )
                candidates = tuple(
                    InvocationSpec.build(name, {}) for name in ranking[:max_budget]
                )
                targets = tuple(
                    InvocationSpec.build(call.name, {})
                    for call in step.calls
                    if call.name in SAFE_LOCAL_TOOLS
                )
                if candidates or targets:
                    opportunities.append(
                        Opportunity(
                            dataset="gemini_safe_tool_name_heldout",
                            metric_scope=(
                                "tool-name ranking only / safe-local target; "
                                "not exact arguments or promotion"
                            ),
                            opportunity_id=(
                                f"{session.session_id}:response-{step.response_index}"
                            ),
                            candidates=candidates,
                            targets=targets,
                        )
                    )
            history.extend(call.name for call in step.calls)
            history = history[-1:]
    evaluation = evaluate_name_predictions(
        train, heldout, max_context=1, top_k=min(3, max_budget)
    )
    sources = [
        {"path": str(session.path.resolve()), "sha256": session.sha256}
        for session in heldout
    ]
    return opportunities, {
        "evaluation": evaluation,
        "split_manifest": split_manifest,
    }, sources


def metric_row(opportunities: Sequence[Opportunity], budget: int) -> dict[str, Any]:
    target_count = sum(len(item.targets) for item in opportunities)
    target_hits = 0
    decisions_with_targets = 0
    decision_hits = 0
    predictions = 0
    useful_predictions = 0
    for item in opportunities:
        selected = item.candidates[:budget]
        selected_keys = {candidate.key for candidate in selected}
        target_keys = {target.key for target in item.targets}
        hits = sum(target.key in selected_keys for target in item.targets)
        target_hits += hits
        predictions += len(selected)
        useful_predictions += sum(
            candidate.key in target_keys for candidate in selected
        )
        if item.targets:
            decisions_with_targets += 1
            decision_hits += hits > 0
    first = opportunities[0] if opportunities else None
    return {
        "dataset": first.dataset if first else "unknown",
        "metric_scope": first.metric_scope if first else "unknown",
        "budget": budget,
        "opportunities": len(opportunities),
        "decisions_with_targets": decisions_with_targets,
        "targets": target_count,
        "target_hits": target_hits,
        "target_recall": rate(target_hits, target_count),
        "decision_hits": decision_hits,
        "decision_hit_rate": rate(decision_hits, decisions_with_targets),
        "unthrottled_selected_predictions": predictions,
        "useful_candidates": useful_predictions,
        "candidate_precision": rate(useful_predictions, predictions),
    }


def _one_throttle_replay(
    opportunities: Sequence[Opportunity],
    *,
    budget: int,
    admission_fraction: float,
    seed: int,
) -> dict[str, float]:
    tokens: list[tuple[int, float, int, int]] = []
    for opportunity_index, item in enumerate(opportunities):
        for candidate_index, _candidate in enumerate(item.candidates[:budget]):
            tokens.append(
                (
                    candidate_index,
                    stable_fraction(seed, item.opportunity_id, candidate_index),
                    opportunity_index,
                    candidate_index,
                )
            )
    # The admission envelope gives every opportunity's rank-1 prediction
    # priority over rank-2, and so on. This is the conservative, fair behavior
    # expected from a rank-aware global speculation budget.
    tokens.sort()
    quota = int(math.floor(len(tokens) * admission_fraction + 1e-12))
    admitted = {(item_index, rank) for _r, _h, item_index, rank in tokens[:quota]}

    targets = sum(len(item.targets) for item in opportunities)
    hits = 0
    target_decisions = 0
    hit_decisions = 0
    useful_candidates = 0
    saved_s = 0.0
    baseline_stall_s = 0.0
    for opportunity_index, item in enumerate(opportunities):
        admitted_candidates = [
            candidate
            for candidate_index, candidate in enumerate(item.candidates[:budget])
            if (opportunity_index, candidate_index) in admitted
        ]
        candidate_keys = {candidate.key for candidate in admitted_candidates}
        target_keys = {target.key for target in item.targets}
        item_hits = sum(target.key in candidate_keys for target in item.targets)
        hits += item_hits
        useful_candidates += sum(
            candidate.key in target_keys for candidate in admitted_candidates
        )
        if item.targets:
            target_decisions += 1
            hit_decisions += item_hits > 0
        if item.baseline_stall_s is not None:
            baseline_stall_s += item.baseline_stall_s
            if item.targets and item.overlap_window_s is not None:
                saved_s += (
                    min(item.baseline_stall_s, item.overlap_window_s)
                    * item_hits
                    / len(item.targets)
                )
    return {
        "requested_candidates": float(len(tokens)),
        "admitted_candidates": float(len(admitted)),
        "admission_ratio": rate(len(admitted), len(tokens)),
        "target_hits": float(hits),
        "target_recall": rate(hits, targets),
        "decision_hit_rate": rate(hit_decisions, target_decisions),
        "useful_candidate_fraction": rate(useful_candidates, len(admitted)),
        "saved_stall_s": saved_s,
        "stall_reduction": rate(saved_s, baseline_stall_s),
    }


def throttle_sweep(
    opportunities: Sequence[Opportunity],
    *,
    budgets: Sequence[int],
    admission_fractions: Sequence[float],
    repetitions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first = opportunities[0]
    target_count = sum(len(item.targets) for item in opportunities)
    for budget in budgets:
        full = _one_throttle_replay(
            opportunities,
            budget=budget,
            admission_fraction=1.0,
            seed=0,
        )
        for admission_fraction in admission_fractions:
            samples = [
                _one_throttle_replay(
                    opportunities,
                    budget=budget,
                    admission_fraction=admission_fraction,
                    seed=seed,
                )
                for seed in range(repetitions)
            ]
            row: dict[str, Any] = {
                "dataset": first.dataset,
                "metric_scope": first.metric_scope,
                "budget": budget,
                "admission_fraction_requested": admission_fraction,
                "throttle_percent": 100.0 * (1.0 - admission_fraction),
                "opportunities": len(opportunities),
                "targets": target_count,
                "repetitions": repetitions,
                "authoritative_priority_contract": (
                    "speculation consumes only the stated residual admission quota; "
                    "authoritative work is not charged to it"
                ),
            }
            for name in samples[0]:
                values = [float(sample[name]) for sample in samples]
                row[name] = statistics.fmean(values)
                row[f"{name}_min"] = min(values)
                row[f"{name}_max"] = max(values)
            row["retained_fraction_of_unthrottled_hits"] = rate(
                row["target_recall"], full["target_recall"]
            )
            rows.append(row)
    return rows


async def _runtime_stress_cell(
    qwen_root: Path,
    opportunities: Sequence[Opportunity],
    *,
    budget: int,
    concurrent_opportunities: int,
    repetitions: int,
    speculative_slots: int,
    tool_service_ms: float,
    lead_ms: float,
) -> dict[str, Any]:
    add_import_path(qwen_root / "reproduction")
    from paste_repro.invocation import Invocation
    from paste_repro.scheduler import SpeculativeScheduler

    async def executor(invocation: Invocation) -> dict[str, Any]:
        await asyncio.sleep(tool_service_ms / 1000.0)
        return {"key": invocation.key}

    scheduler = SpeculativeScheduler(
        executor,
        max_concurrency=speculative_slots,
        max_pending=speculative_slots,
        ttl_s=60.0,
    )
    authoritative_results = []
    requested_predictions = 0
    target_count = 0
    logical_commits = 0
    state_isolation_violations = 0
    try:
        for repetition in range(repetitions):
            ordered = sorted(
                opportunities,
                key=lambda item: stable_fraction(
                    "runtime", repetition, item.opportunity_id
                ),
            )
            # Wrap the same held-out decisions only to keep every concurrency
            # cell equally full; this changes load, not the prediction labels.
            cell_count = (
                math.ceil(len(ordered) / concurrent_opportunities)
                * concurrent_opportunities
            )
            expanded = [ordered[index % len(ordered)] for index in range(cell_count)]
            for batch_start in range(0, len(expanded), concurrent_opportunities):
                batch = expanded[
                    batch_start : batch_start + concurrent_opportunities
                ]
                committed_before_speculation = len(scheduler.authoritative_state)
                session_ids = [
                    f"runtime:{budget}:{concurrent_opportunities}:{repetition}:"
                    f"{batch_start + index}:{item.opportunity_id}"
                    for index, item in enumerate(batch)
                ]
                # Rank-major submission implements fair top-rank-first admission.
                for rank_index in range(budget):
                    for item, session_id in zip(batch, session_ids):
                        if rank_index >= len(item.candidates):
                            continue
                        spec = item.candidates[rank_index]
                        requested_predictions += 1
                        await scheduler.speculate(
                            Invocation(spec.name, spec.arguments()),
                            session_id=session_id,
                        )
                if (
                    len(scheduler.authoritative_state)
                    != committed_before_speculation
                ):
                    # No speculative completion is allowed to commit by itself.
                    state_isolation_violations += 1
                await asyncio.sleep(lead_ms / 1000.0)
                calls = []
                for item, session_id in zip(batch, session_ids):
                    for target in item.targets:
                        target_count += 1
                        calls.append(
                            scheduler.authoritative(
                                Invocation(target.name, target.arguments()),
                                session_id=session_id,
                            )
                        )
                results = await asyncio.gather(*calls)
                authoritative_results.extend(results)
                logical_commits += len(results)
                for session_id in session_ids:
                    await scheduler.sweep(now=math.inf, session_id=session_id)
    finally:
        await scheduler.close()

    hits = sum(result.source in {"reused", "promoted"} for result in authoritative_results)
    exposed_ms = [result.exposed_wait_s * 1000.0 for result in authoritative_results]
    saved_ms = [result.saved_time_s * 1000.0 for result in authoritative_results]
    stats = scheduler.stats.to_dict()
    return {
        "dataset": "qwen_url_exact_heldout",
        "metric_scope": "real bounded scheduler; exact URL confirmation",
        "budget": budget,
        "concurrent_opportunities": concurrent_opportunities,
        "unique_heldout_opportunities": len(opportunities),
        "repetitions": repetitions,
        "speculative_slots": speculative_slots,
        "max_pending": speculative_slots,
        "tool_service_ms_synthetic": tool_service_ms,
        "prediction_lead_ms_synthetic": lead_ms,
        "requested_predictions": requested_predictions,
        "admitted_predictions": stats["admitted"],
        "rejected_capacity": stats["rejected_capacity"],
        "admission_ratio": rate(stats["admitted"], requested_predictions),
        "authoritative_targets": target_count,
        "exact_hits": hits,
        "realized_target_coverage": rate(hits, target_count),
        "authoritative_executions": stats["authoritative_executions"],
        "completed_reuse": stats["completed_reuse"],
        "inflight_promotions": stats["inflight_promotions"],
        "mean_exposed_wait_ms": statistics.fmean(exposed_ms) if exposed_ms else 0.0,
        "p95_exposed_wait_ms": (
            sorted(exposed_ms)[max(0, math.ceil(0.95 * len(exposed_ms)) - 1)]
            if exposed_ms
            else 0.0
        ),
        "mean_saved_ms": statistics.fmean(saved_ms) if saved_ms else 0.0,
        "logical_commits": logical_commits,
        "state_isolation_violations": state_isolation_violations,
        "authoritative_miss_contract": (
            "authoritative misses bypass the speculative semaphore"
        ),
    }


async def runtime_stress_sweep(
    qwen_root: Path,
    opportunities: Sequence[Opportunity],
    *,
    budgets: Sequence[int],
    concurrencies: Sequence[int],
    repetitions: int,
    speculative_slots: int,
    tool_service_ms: float,
    lead_ms: float,
) -> list[dict[str, Any]]:
    rows = []
    for budget in budgets:
        for concurrency in concurrencies:
            rows.append(
                await _runtime_stress_cell(
                    qwen_root,
                    opportunities,
                    budget=budget,
                    concurrent_opportunities=concurrency,
                    repetitions=repetitions,
                    speculative_slots=speculative_slots,
                    tool_service_ms=tool_service_ms,
                    lead_ms=lead_ms,
                )
            )
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def report_markdown(
    *,
    roots: Mapping[str, Path],
    audit: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    load_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    command: str,
) -> str:
    qwen_metrics = {
        int(row["budget"]): row
        for row in metrics
        if row["dataset"] == "qwen_url_exact_heldout"
    }
    gemini_metrics = {
        int(row["budget"]): row
        for row in metrics
        if row["dataset"] == "gemini_safe_tool_name_heldout"
    }
    virtual_metrics = {
        int(row["budget"]): row
        for row in metrics
        if row["dataset"] == "virtual_tongyi_url_exact_loso"
    }
    lines = [
        "# Reviewer common comment 2: metric audit and load × budget stress",
        "",
        "## Bottom line",
        "",
        (
            "The three percentages in the review are not one internally inconsistent "
            "measurement. The exact strings `27.8%` and `43.9%` do not occur as "
            "prediction metrics in the three repositories. The closest reproducible "
            "values are Gemini safe-target tool-name Top-1 8/28="
            f"{pct(gemini_metrics[1]['target_recall'])} and Qwen exact-URL Top-3 "
            f"38/88={pct(qwen_metrics[3]['target_recall'])}; they have different "
            "targets and denominators and must not be combined."
        ),
        "",
        (
            "The literal 93.8% is a Virtual-Lab URL-prefetch **coverage** claim at "
            "N=4. The same source reports a 67.8% per-prefetch hit rate at N=4; "
            "therefore 93.8% is not an overall speculative-execution hit rate. "
            "The repository does not contain the original 33-trace/321-selection "
            "analysis script or a frozen result table, so that legacy 93.8% claim "
            "cannot be independently regenerated from the checked-in code."
        ),
        "",
        (
            "A separate, SHA-bound Virtual-Lab Tongyi artifact happens to reproduce "
            f"exact-URL Top-1 {virtual_metrics[1]['target_hits']}/"
            f"{virtual_metrics[1]['targets']}="
            f"{pct(virtual_metrics[1]['target_recall'])} under LOSO. It is reported "
            "separately and is not used as provenance for the legacy prefetch claim."
        ),
        "",
        "## Unified definitions",
        "",
        "| Name | Numerator | Denominator | Load dependent? |",
        "|---|---|---|---|",
        "| Top-k target recall | authoritative targets whose exact target (or, for Gemini only, name) is in first k | all held-out authoritative targets | no |",
        "| Unthrottled selected-prediction precision | selected predictions that match at least one target in the same decision window | all predictions selected at budget k before any load admission; these are names only for Gemini | no |",
        "| Realized target coverage | targets matched by a candidate actually admitted under the load cap | all held-out targets | yes |",
        "| Prefetch coverage (legacy 93.8%) | selection events with at least one useful prefetched URL | selection events | no, unless admission is modeled |",
        "",
        "## Reproduced source metrics",
        "",
        "| Dataset and scope | Budget | Targets hit | Target recall | Useful/selected predictions | Unthrottled selected-prediction precision | Hit/target windows | Decision hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['dataset']} ({row['metric_scope']}) | {row['budget']} | "
            f"{row['target_hits']}/{row['targets']} | {pct(row['target_recall'])} | "
            f"{row['useful_candidates']}/{row['unthrottled_selected_predictions']} | "
            f"{pct(row['candidate_precision'])} | "
            f"{row['decision_hits']}/{row['decisions_with_targets']} | "
            f"{pct(row['decision_hit_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Qwen's 34 opportunities are search-to-visit decision windows across "
            f"{audit['qwen_official']['held_out_evaluation']['held_out_sessions']} "
            "eligible held-out sessions. Batched visits expand to 88 atomic "
            "authoritative URL targets; therefore decision hit rate uses 34 windows "
            "while target recall uses 88 URL invocations.",
            "",
            "Gemini's 81 opportunities are all held-out next-tool windows, and the "
            "name ranker emits selected names in all 81. Only 28 targets are safe-local, "
            "so safe target recall and decision hit rate use 28 while selected-prediction "
            "precision counts predictions from all 81 windows. The separate all-target "
            "name diagnostic is 39/81 Top-1 and 71/81 Top-3. Gemini candidates contain "
            "no executable arguments. The repository's causal replay finds only 2/28 "
            "exact-match opportunities, so name Top-k must never be presented as "
            "committed promotion hit rate.",
            "",
            "## Low-predictability Qwen stress under throttling",
            "",
            "The Qwen held-out set is the primary low-predictability stress: exact URL "
            f"Top-1 is {pct(qwen_metrics[1]['target_recall'])} and Top-3 is "
            f"{pct(qwen_metrics[3]['target_recall'])}. The table applies a global, "
            "rank-first residual admission quota after authoritative work. A 90% "
            "throttle means only 10% of requested speculative candidates are admitted. "
            "It is a deterministic trace-replay envelope, not a GPU benchmark.",
            "",
            "| Budget | Throttle | Admitted/requested | Realized target coverage | Retained vs unthrottled | Qwen stall reduction |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    selected_load_rows = [
        row
        for row in load_rows
        if row["dataset"] == "qwen_url_exact_heldout"
        and round(float(row["admission_fraction_requested"]), 4)
        in {1.0, 0.5, 0.25, 0.1, 0.0}
    ]
    for row in selected_load_rows:
        lines.append(
            f"| {row['budget']} | {row['throttle_percent']:.0f}% | "
            f"{row['admitted_candidates']:.1f}/{row['requested_candidates']:.1f} | "
            f"{pct(row['target_recall'])} | "
            f"{pct(row['retained_fraction_of_unthrottled_hits'])} | "
            f"{pct(row['stall_reduction'])} |"
        )
    lines.extend(
        [
            "",
            "At saturation (100% throttle) realized coverage and extra work both go "
            "to zero: the mechanism falls back to authoritative execution. Thus the "
            "defensible claim is graceful degradation, not preservation of a 93.8% "
            "number under arbitrary load. Moderate residual capacity still converts "
            "some low-predictability Top-k signal into overlap; wider budgets help "
            "only while candidate admission remains available.",
            "",
            "## Real scheduler admission stress (CPU-only)",
            "",
            "This test uses `paste_repro.scheduler.SpeculativeScheduler`, exact "
            "session-scoped confirmation, eight active/pending speculative slots, "
            "synthetic 5 ms tool service, and 2.5 ms prediction lead. Authoritative "
            "misses bypass the speculative semaphore by implementation contract. "
            "Timing values validate the harness only and are not paper performance "
            "results.",
            "",
            "| Budget | Concurrent opportunities | Admission | Exact realized coverage | Capacity rejects | State-isolation violations |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runtime_rows:
        lines.append(
            f"| {row['budget']} | {row['concurrent_opportunities']} | "
            f"{pct(row['admission_ratio'])} | "
            f"{pct(row['realized_target_coverage'])} | "
            f"{row['rejected_capacity']} | {row['state_isolation_violations']} |"
        )
    lines.extend(
        [
            "",
            "## Limits and claim boundary",
            "",
            "- The Qwen replay uses real held-out targets, candidate order, and traced stall/overlap windows, but the admission envelope is analytical.",
            "- The runtime stress uses the real bounded scheduler and real prediction labels but synthetic milliseconds; it is not an end-to-end vLLM/GPU run.",
            "- This CPU sweep and the GPU/live evidence remain separate tiers. The validated Tongyi target/high results are reported in `../scheduler_robustness/REPORT.md`; the Granite portability attempt failed closed in baseline A and produced no A/E latency result.",
            "- The legacy Virtual-Lab prefetch 93.8% statement has only an in-code summary, not its original analysis artifact. It should be replaced in the rebuttal by a table with explicit numerator/denominator or removed.",
            "- Virtual-Lab's older `reproduction/artifacts/predictor.json` has 26/49 Top-3 in its frozen payload, but the current strict trainer applied to the old ScholarQA trace directory finds no physically bound fetch targets. That legacy payload should not be advertised as freshly reproduced.",
            "",
            "## Reproduction",
            "",
            "```bash",
            command,
            "```",
            "",
            "Repository revisions:",
            "",
        ]
    )
    for name, root in roots.items():
        lines.append(f"- `{name}`: `{git_revision(root)}`")
    lines.extend(
        [
            "",
            "`provenance.json` is a non-exhaustive index of direct evaluation inputs, "
            "not a claim that one file contains every training-source SHA-256. Complete "
            "Qwen and recomputed Gemini train/split manifests are referenced through "
            "`metric_audit.json`; provenance also records paths and file hashes for the "
            "frozen Gemini and Virtual-Lab lineage manifests. Full sweep cells and "
            "deterministic admission-rotation ranges are in the JSON/CSV files beside "
            "this report.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0.0 or item > 1.0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated values in [0,1]")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument(
        "--virtual-root", type=Path, default=DEFAULT_HOME / "virtual-lab-PASTE"
    )
    parser.add_argument(
        "--gemini-root", type=Path, default=DEFAULT_HOME / "gemini-cli-PASTE"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_QWEN_ROOT
        / "reproduction"
        / "results"
        / "reviewer_comment2_load_sweep",
    )
    parser.add_argument("--budgets", type=parse_ints, default=(1, 3, 5))
    parser.add_argument(
        "--admission-fractions",
        type=parse_floats,
        default=(1.0, 0.5, 0.25, 0.1, 0.0),
    )
    parser.add_argument("--admission-repetitions", type=int, default=100)
    parser.add_argument(
        "--runtime-concurrencies", type=parse_ints, default=(1, 8, 32)
    )
    parser.add_argument("--runtime-repetitions", type=int, default=3)
    parser.add_argument("--speculative-slots", type=int, default=8)
    parser.add_argument("--tool-service-ms", type=float, default=5.0)
    parser.add_argument("--prediction-lead-ms", type=float, default=2.5)
    parser.add_argument("--skip-runtime-stress", action="store_true")
    args = parser.parse_args()
    if args.admission_repetitions <= 0 or args.runtime_repetitions <= 0:
        parser.error("repetition counts must be positive")
    if args.speculative_slots <= 0:
        parser.error("--speculative-slots must be positive")
    if args.tool_service_ms <= 0 or args.prediction_lead_ms < 0:
        parser.error("service must be positive and lead non-negative")

    roots = {
        "PASTE-Qwen-DR": args.qwen_root.resolve(),
        "virtual-lab-PASTE": args.virtual_root.resolve(),
        "gemini-cli-PASTE": args.gemini_root.resolve(),
    }
    for name, root in roots.items():
        if not root.is_dir():
            parser.error(f"{name} root does not exist: {root}")
    max_budget = max(args.budgets)

    qwen, qwen_audit, qwen_sources = extract_qwen(roots["PASTE-Qwen-DR"], max_budget)
    virtual, virtual_audit, virtual_sources = extract_virtual(
        roots["virtual-lab-PASTE"], max_budget
    )
    gemini, gemini_audit, gemini_sources = extract_gemini(
        roots["gemini-cli-PASTE"], max_budget
    )
    datasets = [qwen, virtual, gemini]
    metrics = [
        metric_row(opportunities, budget)
        for opportunities in datasets
        for budget in args.budgets
    ]
    load_rows = [
        row
        for opportunities in datasets
        for row in throttle_sweep(
            opportunities,
            budgets=args.budgets,
            admission_fractions=args.admission_fractions,
            repetitions=args.admission_repetitions,
        )
    ]
    runtime_rows: list[dict[str, Any]] = []
    if not args.skip_runtime_stress:
        runtime_rows = asyncio.run(
            runtime_stress_sweep(
                roots["PASTE-Qwen-DR"],
                qwen,
                budgets=args.budgets,
                concurrencies=args.runtime_concurrencies,
                repetitions=args.runtime_repetitions,
                speculative_slots=args.speculative_slots,
                tool_service_ms=args.tool_service_ms,
                lead_ms=args.prediction_lead_ms,
            )
        )

    prefetcher_path = roots["virtual-lab-PASTE"] / "src" / "virtual_lab" / "prefetcher.py"
    legacy_virtual_path = (
        roots["virtual-lab-PASTE"]
        / "reproduction"
        / "artifacts"
        / "predictor.json"
    )
    legacy_virtual = json.loads(legacy_virtual_path.read_text(encoding="utf-8"))
    gemini_replay_path = (
        roots["gemini-cli-PASTE"]
        / "reproduction"
        / "artifacts"
        / "trace_predictor"
        / "causal_replay.json"
    )
    gemini_replay = json.loads(gemini_replay_path.read_text(encoding="utf-8"))
    audit = {
        "schema": "paste.reviewer_comment2.metric_audit.v1",
        "exact_literal_search": {
            "27.8_percent_prediction_metric": "not found in the three repositories",
            "43.9_percent_prediction_metric": "not found in the three repositories",
            "93.8_percent": {
                "found_at": str(prefetcher_path.resolve()),
                "meaning": "N=4 event coverage, paired with 67.8% per-prefetch hit rate",
                "source_analysis_artifact_present": False,
            },
        },
        "nearest_reproducible_values": {
            "gemini_safe_name_top1": gemini_audit["evaluation"][
                "safe_local_targets"
            ],
            "qwen_exact_url_top3": qwen_audit["held_out_evaluation"][
                "top_k_concrete_invocation_hit"
            ]["3"],
        },
        "qwen_official": qwen_audit,
        "virtual_tongyi_loso": virtual_audit["loso"],
        "virtual_tongyi_artifact_sha256": virtual_audit["artifact"][
            "artifact_sha256"
        ],
        "virtual_legacy_scholarqa_frozen_evaluation": legacy_virtual.get(
            "heldout_evaluation"
        ),
        "gemini_split_manifest": gemini_audit["split_manifest"],
        "gemini_name_evaluation": gemini_audit["evaluation"],
        "gemini_causal_replay": gemini_replay,
    }
    gemini_frozen_split_path = (
        roots["gemini-cli-PASTE"]
        / "reproduction"
        / "artifacts"
        / "trace_predictor"
        / "split_manifest.json"
    )
    virtual_predictor_path = (
        roots["virtual-lab-PASTE"]
        / "reproduction"
        / "artifacts"
        / "tongyi_live_predictor.json"
    )
    provenance = {
        "schema": "paste.reviewer_comment2.provenance.v1",
        "result_scope": {
            "cpu_sweep": "completed",
            "historical_gpu_results": "reported_separately",
            "comment3_gpu_run": (
                "target_and_high_completed_and_validated; see "
                "../scheduler_robustness/REPORT.md; Granite cross-model "
                "portability failed closed in baseline A and produced no A/E result"
            ),
        },
        "repositories": {
            name: {"root": str(root), "git_revision": git_revision(root)}
            for name, root in roots.items()
        },
        "source_index_scope": {
            "exhaustive": False,
            "note": (
                "Direct evaluation inputs only; do not treat this source index as "
                "an exhaustive inventory of training-source SHA-256 values."
            ),
            "lineage_references": {
                "qwen_recomputed_train_and_split": (
                    "metric_audit.json#/qwen_official/model_artifact/training_split"
                ),
                "gemini_recomputed_train_and_split": (
                    "metric_audit.json#/gemini_split_manifest"
                ),
                "gemini_frozen_split_manifest": {
                    "path": str(gemini_frozen_split_path.resolve()),
                    "sha256": sha256_file(gemini_frozen_split_path),
                },
                "virtual_frozen_predictor_manifest": {
                    "path": str(virtual_predictor_path.resolve()),
                    "sha256": sha256_file(virtual_predictor_path),
                },
            },
        },
        "sources": {
            "qwen_heldout": qwen_sources,
            "virtual_tongyi_loso": virtual_sources,
            "gemini_heldout": gemini_sources,
            "virtual_prefetcher_summary": {
                "path": str(prefetcher_path.resolve()),
                "sha256": sha256_file(prefetcher_path),
            },
        },
        "experiment_contract": {
            "admission_policy": "global rank-first, hash-rotated within rank",
            "admission_fractions": list(args.admission_fractions),
            "admission_repetitions": args.admission_repetitions,
            "budgets": list(args.budgets),
            "runtime_concurrencies": list(args.runtime_concurrencies),
            "runtime_repetitions": args.runtime_repetitions,
            "speculative_slots": args.speculative_slots,
            "synthetic_tool_service_ms": args.tool_service_ms,
            "synthetic_prediction_lead_ms": args.prediction_lead_ms,
            "network_requests": 0,
            "gpu_required": False,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metric_audit.json", audit)
    write_json(args.output_dir / "source_metrics.json", metrics)
    write_csv(args.output_dir / "source_metrics.csv", metrics)
    write_json(args.output_dir / "load_budget_sweep.json", load_rows)
    write_csv(args.output_dir / "load_budget_sweep.csv", load_rows)
    write_json(args.output_dir / "runtime_stress.json", runtime_rows)
    if runtime_rows:
        write_csv(args.output_dir / "runtime_stress.csv", runtime_rows)
    write_json(args.output_dir / "provenance.json", provenance)
    command = (
        "python3 reproduction/scripts/run_reviewer_comment2_sweep.py "
        "--qwen-root /home/aiscuser/PASTE-Qwen-DR "
        "--virtual-root /home/aiscuser/virtual-lab-PASTE "
        "--gemini-root /home/aiscuser/gemini-cli-PASTE"
    )
    report = report_markdown(
        roots=roots,
        audit=audit,
        metrics=metrics,
        load_rows=load_rows,
        runtime_rows=runtime_rows,
        command=command,
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "source_metric_rows": len(metrics),
                "load_sweep_rows": len(load_rows),
                "runtime_stress_rows": len(runtime_rows),
                "qwen_opportunities": len(qwen),
                "virtual_opportunities": len(virtual),
                "gemini_opportunities": len(gemini),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
