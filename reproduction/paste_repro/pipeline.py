"""Composable training, analysis, and tool-only replay entry points."""

from __future__ import annotations

import asyncio
import hashlib
import math
from pathlib import Path
from typing import Any

from .analysis import evaluate_held_out
from .mapper import URLRankMapper
from .scheduler import SpeculativeScheduler
from .traces import (
    SessionTrace,
    count_tool_calls,
    load_sessions,
    split_sessions,
    transitions_from_sessions,
)


def default_trace_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "traces" / "my_traces"


def _trace_directory_label(path: Path) -> str:
    """Prefer a portable repository-relative path in persisted reports."""

    resolved = path.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        return resolved.relative_to(repository_root).as_posix()
    except ValueError:
        return str(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_split_manifest(
    train_sessions: tuple[SessionTrace, ...],
    test_sessions: tuple[SessionTrace, ...],
    *,
    seed: str,
    train_ratio: float,
) -> dict[str, Any]:
    train_files = [
        {"session_id": session.session_id, "sha256": _file_sha256(session.path)}
        for session in sorted(train_sessions, key=lambda item: item.session_id)
    ]
    held_out_files = [
        {"session_id": session.session_id, "sha256": _file_sha256(session.path)}
        for session in sorted(test_sessions, key=lambda item: item.session_id)
    ]
    manifest: dict[str, Any] = {
        "algorithm": "sha256(seed + NUL + session_id), exact whole-file prefix split",
        "seed": seed,
        "train_ratio": train_ratio,
        "train_sessions": train_files,
        "held_out_sessions": held_out_files,
    }
    # URLRankMapper.to_artifact validates and adds the manifest checksum.
    return manifest


def train_and_analyze(
    trace_directory: str | Path | None = None,
    *,
    seed: str = "paste-repro-v1",
    train_ratio: float = 0.70,
    top_k: int = 5,
) -> tuple[
    dict[str, Any],
    URLRankMapper,
    tuple[Any, ...],
    tuple[SessionTrace, ...],
    tuple[SessionTrace, ...],
]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    trace_dir = default_trace_directory() if trace_directory is None else Path(trace_directory)
    sessions = load_sessions(trace_dir)
    train_sessions, test_sessions = split_sessions(
        sessions, train_ratio=train_ratio, seed=seed
    )
    train_transitions = transitions_from_sessions(train_sessions)
    test_transitions = transitions_from_sessions(test_sessions)
    mapper = URLRankMapper().fit(
        train_transitions,
        searches_seen=count_tool_calls(train_sessions, "search"),
    )
    split_manifest = build_split_manifest(
        train_sessions, test_sessions, seed=seed, train_ratio=train_ratio
    )
    artifact = mapper.to_artifact(split_manifest)
    top_ks = tuple(sorted({1, 3, top_k}))
    evaluation = evaluate_held_out(
        mapper, test_transitions, top_ks=top_ks, latency_top_k=top_k
    )
    report = {
        "schema": "paste_repro.trace_analysis",
        "version": 1,
        "trace_directory": _trace_directory_label(trace_dir),
        "split": {
            "seed": seed,
            "train_ratio": train_ratio,
            "total_sessions": len(sessions),
            "train_sessions": len(train_sessions),
            "held_out_sessions": len(test_sessions),
            "train_search_visit_examples": len(train_transitions),
            "held_out_search_visit_examples": len(test_transitions),
        },
        "model_artifact": artifact,
        "held_out_evaluation": evaluation,
    }
    return report, mapper, test_transitions, train_sessions, test_sessions


async def run_tool_only_replay(
    trace_directory: str | Path | None = None,
    *,
    seed: str = "paste-repro-v1",
    train_ratio: float = 0.70,
    top_k: int = 5,
    max_concurrency: int = 4,
    limit: int | None = None,
    simulation_delay_s: float = 0.0,
) -> dict[str, Any]:
    """Replay held-out URL-level invocations without network access."""

    report, mapper, transitions, _, _ = train_and_analyze(
        trace_directory,
        seed=seed,
        train_ratio=train_ratio,
        top_k=top_k,
    )
    selected = transitions if limit is None else transitions[: max(0, limit)]

    async def replay_executor(invocation: Any) -> dict[str, Any]:
        if simulation_delay_s > 0:
            await asyncio.sleep(simulation_delay_s)
        else:
            await asyncio.sleep(0)
        return {"replayed": invocation.to_dict()}

    scheduler = SpeculativeScheduler(
        replay_executor,
        max_concurrency=max_concurrency,
        max_pending=max(max_concurrency, top_k),
        ttl_s=30.0,
    )
    state_isolation_violations = 0
    try:
        for index, transition in enumerate(selected):
            replay_session = f"{transition.session_id}:{index}"
            predictions = mapper.predict(transition.search_results, top_k)
            committed_before = len(scheduler.authoritative_state)
            for prediction in predictions:
                await scheduler.speculate(
                    prediction.invocation, session_id=replay_session
                )
            # Let admitted work enter the executor. No result is committed here.
            await asyncio.sleep(0)
            if len(scheduler.authoritative_state) != committed_before:
                state_isolation_violations += 1
            for invocation in transition.authoritative_invocations:
                await scheduler.authoritative(invocation, session_id=replay_session)
            # End-of-transition TTL advancement discards unused isolated results.
            await scheduler.sweep(now=math.inf, session_id=replay_session)
    finally:
        await scheduler.close()

    replay_evaluation = evaluate_held_out(
        mapper,
        selected,
        top_ks=tuple(sorted({1, 3, top_k})),
        latency_top_k=top_k,
    )
    return {
        "schema": "paste_repro.tool_only_replay",
        "version": 1,
        "model_artifact": report["model_artifact"],
        "split": report["split"],
        "replayed_examples": len(selected),
        "replayed_authoritative_invocations": sum(
            len(transition.authoritative_urls) for transition in selected
        ),
        "state_isolation_violations": state_isolation_violations,
        "authoritative_state_entries": len(scheduler.authoritative_state),
        "scheduler": scheduler.stats.to_dict(),
        "held_out_evaluation": replay_evaluation,
        "executor": "deterministic trace replay (no network calls)",
    }


async def run_speculative_tool_execution(
    trace_directory: str | Path | None = None,
    *,
    seed: str = "paste-repro-v1",
    train_ratio: float = 0.70,
    top_k: int = 5,
    max_concurrency: int = 4,
    limit: int | None = None,
    simulation_delay_s: float = 0.0,
) -> dict[str, Any]:
    """Run the trace-learned tool-side experiment without LLM co-design.

    This is the first-class name for the causal component.  The legacy
    ``run_tool_only_replay`` entry point remains stable for existing scripts.
    """

    report = await run_tool_only_replay(
        trace_directory,
        seed=seed,
        train_ratio=train_ratio,
        top_k=top_k,
        max_concurrency=max_concurrency,
        limit=limit,
        simulation_delay_s=simulation_delay_s,
    )
    report["schema"] = "paste_repro.speculative_tool_execution"
    report["experiment"] = {
        "name": "speculative tool execution (without LLM co-design)",
        "llm_co_design": False,
        "training_signal": "historical selected within-query search-result rank",
        "prediction_input": "current already-visible search response only",
        "confirmation": "exact session-scoped URL invocation match",
        "authoritative_miss_policy": "execute normally",
        "speculative_state_policy": "isolated until exact authoritative match",
    }
    return report
