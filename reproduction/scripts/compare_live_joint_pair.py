#!/usr/bin/env python3
"""Strictly compare two live tool--LLM screening cells.

The comparator is intentionally fail closed.  It derives every performance
number from task, LLM, physical-tool, and queue-sample evidence rather than
accepting the summaries embedded by the runner.  Dynamic live search results
are allowed to differ between cells, but such a pair is explicitly limited to
``screen_only`` evidence instead of being presented as an identity-matched
paired result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


SCHEMA_VERSION = 1
OUTPUT_SCHEMA = "paste_repro.live_joint_pair_screen"
BOOTSTRAP_SEED = 20260816
BOOTSTRAP_RESAMPLES = 10_000
_PAIR_CONFIG_EXCLUSIONS = frozenset(
    {
        "cell_label",
        "speculation_mode",
        # This is observed live-search evidence, not a frozen design input.
        "expected_url_search_coverage",
    }
)
_CANDIDATE_SPECULATION_MODES = frozenset({"search", "visit", "search_visit"})
CONTROLLED_HTTP_MAX_ATTEMPTS = 2
CONTROLLED_HTTP_RETRY_BACKOFF_S = 1.0
CONTROLLED_HTTP_RETRY_POLICY_VERSION = "idempotent-get-v1"
HTTP_LIBRARY_RETRY_CONTROL_VERSION = "aiohttp-private-retry-connection-v1"
FORMAL_HTTP_LIBRARY_NAME = "aiohttp"
FORMAL_HTTP_LIBRARY_VERSION = "3.12.15"
CONTROLLED_HTTP_RETRYABLE_STATUSES = [429, 500, 502, 503, 504]
CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES = [
    "asyncio.TimeoutError",
    "ConnectionError",
    "aiohttp.ClientConnectionError",
    "aiohttp.ClientPayloadError",
]


@dataclass(frozen=True)
class ValidatedRun:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    config: Mapping[str, Any]
    tasks_by_key: Mapping[tuple[str, int], Mapping[str, Any]]
    tasks_by_id: Mapping[str, Mapping[str, Any]]
    llm_by_task: Mapping[str, tuple[Mapping[str, Any], ...]]
    committed_by_task_tool: Mapping[tuple[str, str], Mapping[str, Any]]
    physical_records: tuple[Mapping[str, Any], ...]
    timeline: tuple[Mapping[str, Any], ...]
    call_graph_mode: str
    search_coverage: Mapping[str, Any] | None
    summary: Mapping[str, Any]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < (sys.float_info.min if positive else 0.0):
        qualifier = "positive" if positive else "finite and non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _sha256_text(value: Any, label: str) -> str:
    result = _string(value, label)
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be lowercase SHA256")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc


def _invocation_digest(invocation: Mapping[str, Any], label: str) -> str:
    tool_name = _string(invocation.get("tool_name"), f"{label}.tool_name")
    arguments = _mapping(invocation.get("arguments"), f"{label}.arguments")
    canonical_arguments = _canonical_json(arguments)
    return hashlib.sha256(
        f"{tool_name}\0{canonical_arguments}".encode("utf-8")
    ).hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "p50": _percentile(numeric, 0.50),
        "p95": _percentile(numeric, 0.95),
        "p99": _percentile(numeric, 0.99),
        "max": max(numeric),
    }


def _latency_comparison(baseline: float, candidate: float) -> dict[str, float]:
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_reduction": baseline - candidate,
        "relative_reduction": ((baseline - candidate) / baseline if baseline else 0.0),
    }


def _numeric_comparison(baseline: float, candidate: float) -> dict[str, float]:
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_delta": candidate - baseline,
        "relative_delta": ((candidate - baseline) / baseline if baseline else 0.0),
    }


def _close(left: float, right: float, *, tolerance: float = 1e-5) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    return _mapping(value, label)


def _resolve_sidecar(
    result_path: Path,
    reference: Mapping[str, Any],
    *,
    label: str,
    override: Path | None,
) -> tuple[Path, tuple[Mapping[str, Any], ...]]:
    referenced = Path(_string(reference.get("path"), f"{label}.path"))
    if override is not None:
        path = override.resolve()
    elif referenced.is_absolute():
        path = referenced.resolve()
    else:
        path = (result_path.parent / referenced).resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    expected_sha = _sha256_text(reference.get("sha256"), f"{label}.sha256")
    if _sha256_file(path) != expected_sha:
        raise ValueError(f"{label} SHA256 mismatch")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from exc
        rows.append(_mapping(row, f"{label} line {line_number}"))
    expected_count = _integer(reference.get("sample_count"), f"{label}.sample_count")
    if len(rows) != expected_count or not rows:
        raise ValueError(f"{label} row count does not match non-empty evidence")
    return path, tuple(rows)


def _validate_retry_config(
    config: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Validate the only bounded HTTP-retry policy accepted by live runs.

    A one-attempt development run remains valid, but any enabled retry must be
    the preregistered two-attempt, one-second-backoff idempotent-GET policy.
    This is the control-plane evidence that distinguishes a bounded retry from
    an unreported client/library retry.
    """

    max_attempts = _integer(
        config.get("tool_http_max_attempts"),
        f"{label}.tool_http_max_attempts",
        positive=True,
    )
    if max_attempts not in {1, CONTROLLED_HTTP_MAX_ATTEMPTS}:
        raise ValueError(
            f"{label}.tool_http_max_attempts must be 1 or "
            f"{CONTROLLED_HTTP_MAX_ATTEMPTS}"
        )
    backoff_s = _finite(
        config.get("tool_http_retry_backoff_s"),
        f"{label}.tool_http_retry_backoff_s",
    )
    if not math.isclose(
        backoff_s, CONTROLLED_HTTP_RETRY_BACKOFF_S, abs_tol=1e-12
    ):
        raise ValueError(
            f"{label}.tool_http_retry_backoff_s must be "
            f"{CONTROLLED_HTTP_RETRY_BACKOFF_S}"
        )
    enabled = _boolean(
        config.get("controlled_http_retry"),
        f"{label}.controlled_http_retry",
    )
    if enabled is not (max_attempts > 1):
        raise ValueError(
            f"{label}.controlled_http_retry disagrees with the attempt limit"
        )
    if config.get("tool_http_retry_policy_version") != (
        CONTROLLED_HTTP_RETRY_POLICY_VERSION
    ):
        raise ValueError(f"{label} has an unsupported HTTP retry policy")
    if config.get("tool_http_retryable_statuses") != (
        CONTROLLED_HTTP_RETRYABLE_STATUSES
    ):
        raise ValueError(f"{label} has an unsupported retryable-status set")
    if config.get("tool_http_retryable_exception_types") != (
        CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES
    ):
        raise ValueError(f"{label} has an unsupported retryable-exception set")
    if _boolean(
        config.get("tool_http_library_retry_disabled"),
        f"{label}.tool_http_library_retry_disabled",
    ) is not True:
        raise ValueError(f"{label} did not disable hidden HTTP-library retry")
    if (
        config.get("tool_http_library_retry_control_version")
        != HTTP_LIBRARY_RETRY_CONTROL_VERSION
    ):
        raise ValueError(f"{label} has an unsupported library-retry control")
    library_name = _string(
        config.get("tool_http_library_name"),
        f"{label}.tool_http_library_name",
    )
    if library_name != FORMAL_HTTP_LIBRARY_NAME:
        raise ValueError(f"{label} did not use the audited aiohttp transport")
    library_version = _string(
        config.get("tool_http_library_version"),
        f"{label}.tool_http_library_version",
    )
    if library_version != FORMAL_HTTP_LIBRARY_VERSION:
        raise ValueError(
            f"{label} must freeze aiohttp {FORMAL_HTTP_LIBRARY_VERSION}"
        )
    return {
        "max_attempts": max_attempts,
        "backoff_s": backoff_s,
        "enabled": enabled,
        "policy_version": CONTROLLED_HTTP_RETRY_POLICY_VERSION,
        "retryable_statuses": list(CONTROLLED_HTTP_RETRYABLE_STATUSES),
        "retryable_exception_types": list(
            CONTROLLED_HTTP_RETRYABLE_EXCEPTION_TYPES
        ),
        "http_library_retry_disabled": True,
        "http_library_retry_control_version": (
            HTTP_LIBRARY_RETRY_CONTROL_VERSION
        ),
        "http_library_name": library_name,
        "http_library_version": library_version,
    }


def _validate_http_record(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    invocation: Mapping[str, Any],
    label: str,
) -> None:
    tool = _string(record.get("tool"), f"{label}.tool")
    if record.get("transport_identity_source") != "actual":
        raise ValueError(f"{label} lacks actual (rather than planned) HTTP evidence")
    backend = _string(record.get("backend"), f"{label}.backend")
    hosts = [item for item in _string(record.get("request_host"), f"{label}.request_host").split(",") if item]
    if not hosts:
        raise ValueError(f"{label}.request_host is empty")
    if _integer(record.get("response_status"), f"{label}.response_status") != 200:
        raise ValueError(f"{label} is not an HTTP 200 execution")
    _integer(record.get("bytes_read"), f"{label}.bytes_read", positive=True)
    attempts = _integer(record.get("http_attempts"), f"{label}.http_attempts", positive=True)
    max_attempts = _integer(
        config.get("tool_http_max_attempts"),
        "config.tool_http_max_attempts",
        positive=True,
    )
    if attempts > max_attempts:
        raise ValueError(f"{label} exceeds the controlled HTTP attempt limit")
    arguments = _mapping(invocation.get("arguments"), f"{label}.invocation.arguments")

    if tool == "search":
        mode = _string(config.get("search_mode"), "config.search_mode")
        expected_backend = {
            "bing": "bing_html_search",
            "rest": "wikipedia_rest_search",
            "action": "wikipedia_mediawiki_action",
        }.get(mode)
        if expected_backend is None or backend != expected_backend:
            raise ValueError(f"{label} search backend does not match search_mode")
        if mode == "bing":
            if set(hosts) != {"www.bing.com"}:
                raise ValueError(f"{label} Bing host evidence is invalid")
        elif any(not host.endswith(".wikipedia.org") for host in hosts):
            raise ValueError(f"{label} Wikipedia search host evidence is invalid")
        queries = arguments.get("query")
        if (
            not isinstance(queries, list)
            or len(queries) != 1
            or not isinstance(queries[0], str)
            or not queries[0]
        ):
            raise ValueError(
                f"{label} search invocation must contain exactly one query"
            )
        return

    if tool != "visit":
        raise ValueError(f"{label} has unsupported tool {tool!r}")
    mode = _string(config.get("visit_mode"), "config.visit_mode")
    urls = arguments.get("url")
    if not isinstance(urls, list) or len(urls) != 1 or not isinstance(urls[0], str):
        raise ValueError(f"{label} visit invocation must contain exactly one URL")
    if mode == "jina":
        if backend != "r.jina.ai" or set(hosts) != {"r.jina.ai"}:
            raise ValueError(f"{label} Jina transport evidence is invalid")
    elif mode == "direct":
        expected_host = urlparse(urls[0]).hostname
        if backend != "direct_http" or not expected_host or set(hosts) != {expected_host}:
            raise ValueError(f"{label} direct transport evidence is invalid")
    else:
        raise ValueError("config.visit_mode is unsupported")


def _validate_timeline(
    rows: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any], label: str
) -> dict[str, Any]:
    previous_monotonic = -math.inf
    queued_samples = 0
    llm_samples = 0
    llm_running_samples = 0
    joint_pressure_samples = 0
    max_tool_queue = 0
    max_llm_running = 0.0
    max_llm_waiting = 0.0
    tool_workers = _integer(config.get("tool_workers"), "config.tool_workers", positive=True)
    spec_workers = _integer(
        config.get("speculative_tool_workers"),
        "config.speculative_tool_workers",
    )
    if spec_workers > tool_workers:
        raise ValueError("speculative_tool_workers exceeds total tool workers")
    for index, raw in enumerate(rows):
        prefix = f"{label}[{index}]"
        monotonic = _finite(raw.get("monotonic_s"), f"{prefix}.monotonic_s")
        _finite(raw.get("wall_s"), f"{prefix}.wall_s")
        if monotonic < previous_monotonic:
            raise ValueError(f"{label} is not monotonic")
        previous_monotonic = monotonic
        counts = {
            key: _integer(raw.get(key), f"{prefix}.{key}")
            for key in (
                "tool_queued_authoritative",
                "tool_queued_speculative",
                "tool_running_authoritative",
                "tool_running_speculative",
                "tool_completed_unclaimed_speculative",
            )
        }
        if counts["tool_running_authoritative"] + counts["tool_running_speculative"] > tool_workers:
            raise ValueError(f"{prefix} exceeds shared tool worker capacity")
        if counts["tool_running_speculative"] > spec_workers:
            raise ValueError(f"{prefix} exceeds speculative worker capacity")
        queued = counts["tool_queued_authoritative"] + counts["tool_queued_speculative"]
        max_tool_queue = max(max_tool_queue, queued)
        if queued > 0:
            queued_samples += 1
        llm_running_raw = raw.get("llm_running")
        llm_waiting_raw = raw.get("llm_waiting")
        if (llm_running_raw is None) != (llm_waiting_raw is None):
            raise ValueError(f"{prefix} has partial LLM metric evidence")
        if llm_running_raw is not None:
            llm_running = _finite(llm_running_raw, f"{prefix}.llm_running")
            llm_waiting = _finite(llm_waiting_raw, f"{prefix}.llm_waiting")
            llm_samples += 1
            max_llm_running = max(max_llm_running, llm_running)
            max_llm_waiting = max(max_llm_waiting, llm_waiting)
            if llm_running > 0:
                llm_running_samples += 1
            if llm_waiting > 0 and (
                queued > 0
                or counts["tool_running_authoritative"] > 0
                or counts["tool_running_speculative"] > 0
            ):
                joint_pressure_samples += 1
    if queued_samples == 0:
        raise ValueError(f"{label} contains no real tool-queue pressure")
    if llm_samples == 0 or llm_running_samples == 0:
        raise ValueError(f"{label} contains no live LLM-serving evidence")
    return {
        "sample_count": len(rows),
        "tool_queue_sample_count": queued_samples,
        "tool_queue_sample_fraction": queued_samples / len(rows),
        "max_tool_queued": max_tool_queue,
        "llm_metric_sample_count": llm_samples,
        "llm_running_sample_count": llm_running_samples,
        "max_llm_running": max_llm_running,
        "max_llm_waiting": max_llm_waiting,
        "joint_pressure_sample_count": joint_pressure_samples,
        "joint_pressure_fraction": joint_pressure_samples / llm_samples,
    }


def _validate_run(
    path: Path,
    *,
    role: str,
    timeline_override: Path | None = None,
) -> ValidatedRun:
    result_path = path.resolve()
    if not result_path.is_file():
        raise ValueError(f"{role} result does not exist: {result_path}")
    payload = _load_json(result_path, f"{role} result")
    if _integer(payload.get("schema_version"), f"{role}.schema_version") != 1:
        raise ValueError(f"{role} result schema_version is unsupported")
    config = _mapping(payload.get("config"), f"{role}.config")
    required_true = (
        "live_tool_execution",
        "shared_bounded_tool_pool",
        "generated_tool_call_controls_next_prompt",
        "authoritative_and_speculative_share_capacity",
        "tool_metadata_is_causal",
        "tool_result_private_until_exact_commit",
    )
    for key in required_true:
        if _boolean(config.get(key), f"{role}.config.{key}") is not True:
            raise ValueError(f"{role}.config.{key} must be true")
    retry_config = _validate_retry_config(config, label=f"{role}.config")
    if _boolean(config.get("recorded_tool_sleep"), f"{role}.config.recorded_tool_sleep"):
        raise ValueError(f"{role} used recorded tool sleeps")
    if _boolean(config.get("future_trace_oracle_used"), f"{role}.config.future_trace_oracle_used"):
        raise ValueError(f"{role} used a future-trace oracle")
    _sha256_text(config.get("workload_file_sha256"), f"{role}.config.workload_file_sha256")
    _sha256_text(config.get("selected_workload_sha256"), f"{role}.config.selected_workload_sha256")
    speculation_mode = _string(config.get("speculation_mode"), f"{role}.config.speculation_mode")
    if role == "baseline" and speculation_mode != "off":
        raise ValueError("baseline speculation_mode must be 'off'")
    if role == "candidate" and speculation_mode not in _CANDIDATE_SPECULATION_MODES:
        raise ValueError("candidate must enable search and/or visit speculation")
    raw_call_graph_mode = config.get("call_graph_mode")
    if raw_call_graph_mode is None:
        # Backward compatibility for the first autonomous live-run schema.
        call_graph_mode = "autonomous"
    else:
        call_graph_mode = _string(
            raw_call_graph_mode, f"{role}.config.call_graph_mode"
        )
    if call_graph_mode not in {"autonomous", "frozen"}:
        raise ValueError(f"{role}.config.call_graph_mode is unsupported")
    frozen_input_flag = config.get("frozen_url_is_workload_input")
    if call_graph_mode == "frozen":
        if _boolean(
            frozen_input_flag,
            f"{role}.config.frozen_url_is_workload_input",
        ) is not True:
            raise ValueError(
                f"{role} frozen graph requires frozen_url_is_workload_input=true"
            )
    elif frozen_input_flag not in {None, False}:
        raise ValueError(
            f"{role} autonomous graph cannot claim a frozen workload URL"
        )

    raw_evidence = _mapping(payload.get("raw_evidence"), f"{role}.raw_evidence")
    timeline_reference = _mapping(
        raw_evidence.get("queue_timeline"), f"{role}.raw_evidence.queue_timeline"
    )
    _, timeline = _resolve_sidecar(
        result_path,
        timeline_reference,
        label=f"{role} queue_timeline",
        override=timeline_override,
    )
    # Any additional sidecars are integrity checked as well.  The current
    # runner embeds task/LLM/tool rows and emits only queue_timeline, while a
    # future runner may split those raw rows without weakening this check.
    for kind, raw_reference in raw_evidence.items():
        if kind == "queue_timeline":
            continue
        reference = _mapping(raw_reference, f"{role}.raw_evidence.{kind}")
        _resolve_sidecar(
            result_path,
            reference,
            label=f"{role} {kind}",
            override=None,
        )
    timeline_summary = _validate_timeline(timeline, config=config, label=f"{role} queue_timeline")

    raw_tasks = _sequence(payload.get("tasks"), f"{role}.tasks")
    task_count = _integer(config.get("task_count"), f"{role}.config.task_count", positive=True)
    if len(raw_tasks) != task_count:
        raise ValueError(f"{role} task_count does not match raw tasks")
    independent_sources = _integer(
        config.get("independent_source_count"),
        f"{role}.config.independent_source_count",
        positive=True,
    )
    replicas = _integer(config.get("replicas"), f"{role}.config.replicas", positive=True)
    if task_count != independent_sources * replicas:
        raise ValueError(f"{role} task/source/replica counts are inconsistent")

    tasks_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    tasks_by_id: dict[str, Mapping[str, Any]] = {}
    task_invocations: dict[tuple[str, str], tuple[str, str, Mapping[str, Any]]] = {}
    frozen_search_matches: dict[str, bool] = {}
    for index, raw_task in enumerate(raw_tasks):
        task = _mapping(raw_task, f"{role}.tasks[{index}]")
        task_id = _string(task.get("task_id"), f"{role}.tasks[{index}].task_id")
        source_id = _string(task.get("source_id"), f"{role}.tasks[{index}].source_id")
        replica = _integer(task.get("replica"), f"{role}.tasks[{index}].replica")
        if task_id != f"{source_id}__r{replica:02d}":
            raise ValueError(f"{role} task_id is not canonical: {task_id}")
        if (source_id, replica) in tasks_by_key or task_id in tasks_by_id:
            raise ValueError(f"{role} has duplicate task identity: {task_id}")
        if _boolean(task.get("ok"), f"{role}.{task_id}.ok") is not True:
            raise ValueError(f"{role} task did not succeed: {task_id}")
        _boolean(task.get("visit_canary"), f"{role}.{task_id}.visit_canary")
        e2e = _finite(task.get("e2e_s"), f"{role}.{task_id}.e2e_s", positive=True)
        started = _finite(task.get("start_wall_s"), f"{role}.{task_id}.start_wall_s", positive=True)
        ended = _finite(task.get("end_wall_s"), f"{role}.{task_id}.end_wall_s", positive=True)
        if ended < started or abs((ended - started) - e2e) > 0.25:
            raise ValueError(f"{role} task wall-clock and monotonic E2E disagree: {task_id}")
        _sha256_text(task.get("question_sha256"), f"{role}.{task_id}.question_sha256")
        _string(task.get("search_query"), f"{role}.{task_id}.search_query")
        search_urls_raw = _sequence(task.get("search_urls"), f"{role}.{task_id}.search_urls")
        search_urls = [_string(url, f"{role}.{task_id}.search_urls") for url in search_urls_raw]
        if not search_urls or len(search_urls) != len(set(search_urls)):
            raise ValueError(f"{role} task search URLs are empty or duplicated: {task_id}")
        selected_url = _string(task.get("selected_url"), f"{role}.{task_id}.selected_url")
        task_call_graph_mode = task.get("call_graph_mode")
        if call_graph_mode == "frozen":
            if task_call_graph_mode != "frozen":
                raise ValueError(
                    f"{role} frozen task lacks call_graph_mode=frozen: {task_id}"
                )
            expected_url = _string(
                task.get("expected_url"), f"{role}.{task_id}.expected_url"
            )
            parsed_expected = urlparse(expected_url)
            if (
                parsed_expected.scheme != "https"
                or not parsed_expected.hostname
                or parsed_expected.username is not None
                or parsed_expected.password is not None
            ):
                raise ValueError(
                    f"{role} frozen expected_url must be an absolute HTTPS URL: {task_id}"
                )
            if selected_url != expected_url:
                raise ValueError(
                    f"{role} frozen selected_url differs from expected_url: {task_id}"
                )
            observed_match = expected_url in search_urls
            reported_match = _boolean(
                task.get("search_result_contains_expected_url"),
                f"{role}.{task_id}.search_result_contains_expected_url",
            )
            if reported_match != observed_match:
                raise ValueError(
                    f"{role} frozen search-coverage evidence is inconsistent: {task_id}"
                )
            frozen_search_matches[task_id] = observed_match
        else:
            if task_call_graph_mode not in {None, "autonomous"}:
                raise ValueError(
                    f"{role} autonomous task has a frozen call-graph marker: {task_id}"
                )
            if task.get("expected_url") is not None or task.get(
                "search_result_contains_expected_url"
            ) is not None:
                raise ValueError(
                    f"{role} autonomous task contains frozen URL evidence: {task_id}"
                )
            if selected_url not in search_urls:
                raise ValueError(
                    f"{role} selected URL was not returned by live search: {task_id}"
                )
        answer = _mapping(task.get("answer"), f"{role}.{task_id}.answer")
        if _string(answer.get("source_url"), f"{role}.{task_id}.answer.source_url") != selected_url:
            raise ValueError(f"{role} answer cites a different URL: {task_id}")
        _string(answer.get("answer"), f"{role}.{task_id}.answer.answer")
        answer_digest = _sha256_text(
            task.get("answer_sha256"), f"{role}.{task_id}.answer_sha256"
        )
        if answer_digest != hashlib.sha256(
            _canonical_json(answer).encode("utf-8")
        ).hexdigest():
            raise ValueError(f"{role} answer digest is inconsistent: {task_id}")
        _integer(task.get("prompt_tokens"), f"{role}.{task_id}.prompt_tokens")
        _integer(task.get("completion_tokens"), f"{role}.{task_id}.completion_tokens")
        _finite(task.get("llm_duration_s"), f"{role}.{task_id}.llm_duration_s", positive=True)

        raw_tools = _sequence(task.get("tools"), f"{role}.{task_id}.tools")
        if len(raw_tools) != 2:
            raise ValueError(f"{role} task must have exactly search+visit commits: {task_id}")
        for tool_index, expected_tool in enumerate(("search", "visit")):
            tool_row = _mapping(raw_tools[tool_index], f"{role}.{task_id}.tools[{tool_index}]")
            invocation = _mapping(
                tool_row.get("invocation"),
                f"{role}.{task_id}.tools[{tool_index}].invocation",
            )
            tool_name = _string(invocation.get("tool_name"), f"{role}.{task_id}.tools[{tool_index}].tool_name")
            if tool_name != expected_tool:
                raise ValueError(f"{role} task tool order is not search then visit: {task_id}")
            digest = _invocation_digest(invocation, f"{role}.{task_id}.{tool_name}")
            result_digest = _sha256_text(
                tool_row.get("result_sha256"), f"{role}.{task_id}.{tool_name}.result_sha256"
            )
            for metric in ("queue_s", "service_s", "exposed_wait_s", "saved_service_s"):
                _finite(tool_row.get(metric), f"{role}.{task_id}.{tool_name}.{metric}")
            _string(tool_row.get("source"), f"{role}.{task_id}.{tool_name}.source")
            task_invocations[(task_id, tool_name)] = (
                digest,
                result_digest,
                invocation,
            )
        search_args = _mapping(
            _mapping(raw_tools[0], f"{role}.{task_id}.search").get("invocation"),
            f"{role}.{task_id}.search.invocation",
        ).get("arguments")
        search_args = _mapping(search_args, f"{role}.{task_id}.search.arguments")
        if search_args.get("query") != [task["search_query"]]:
            raise ValueError(f"{role} task search query and invocation differ: {task_id}")
        visit_args = _mapping(
            _mapping(raw_tools[1], f"{role}.{task_id}.visit").get("invocation"),
            f"{role}.{task_id}.visit.invocation",
        ).get("arguments")
        visit_args = _mapping(visit_args, f"{role}.{task_id}.visit.arguments")
        if visit_args.get("url") != [selected_url]:
            raise ValueError(f"{role} selected URL and visit invocation differ: {task_id}")
        tasks_by_key[(source_id, replica)] = task
        tasks_by_id[task_id] = task

    if len({source_id for source_id, _ in tasks_by_key}) != independent_sources:
        raise ValueError(f"{role} independent source count is inconsistent")
    if Counter(replica for _, replica in tasks_by_key) != Counter(
        {replica: independent_sources for replica in range(replicas)}
    ):
        raise ValueError(f"{role} replica coverage is incomplete")

    search_coverage: Mapping[str, Any] | None = None
    if call_graph_mode == "frozen":
        observed_count = len(frozen_search_matches)
        matched_count = sum(frozen_search_matches.values())
        fraction = matched_count / observed_count
        reported_coverage = _mapping(
            config.get("expected_url_search_coverage"),
            f"{role}.config.expected_url_search_coverage",
        )
        for key, expected in (
            ("eligible_task_count", task_count),
            ("observed_task_count", observed_count),
            ("matched_task_count", matched_count),
        ):
            if _integer(
                reported_coverage.get(key),
                f"{role}.config.expected_url_search_coverage.{key}",
            ) != expected:
                raise ValueError(f"{role} frozen search-coverage count is inconsistent")
        for key in ("fraction_of_eligible", "fraction_of_observed"):
            if not _close(
                _finite(
                    reported_coverage.get(key),
                    f"{role}.config.expected_url_search_coverage.{key}",
                ),
                fraction,
            ):
                raise ValueError(
                    f"{role} frozen search-coverage fraction is inconsistent"
                )
        search_coverage = {
            "eligible_task_count": task_count,
            "observed_task_count": observed_count,
            "matched_task_count": matched_count,
            "fraction_of_eligible": fraction,
            "fraction_of_observed": fraction,
            "matched_task_ids": sorted(
                task_id for task_id, matched in frozen_search_matches.items() if matched
            ),
            "unmatched_task_ids": sorted(
                task_id for task_id, matched in frozen_search_matches.items() if not matched
            ),
        }

    raw_llm = _sequence(payload.get("llm_events"), f"{role}.llm_events")
    llm_by_task_lists: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    request_ids: set[str] = set()
    for index, raw_event in enumerate(raw_llm):
        event = _mapping(raw_event, f"{role}.llm_events[{index}]")
        task_id = _string(event.get("task_id"), f"{role}.llm_events[{index}].task_id")
        if task_id not in tasks_by_id:
            raise ValueError(f"{role} LLM event references unknown task: {task_id}")
        request_id = _string(event.get("request_id"), f"{role}.{task_id}.request_id")
        if request_id in request_ids:
            raise ValueError(f"{role} duplicate LLM request_id: {request_id}")
        request_ids.add(request_id)
        call_index = _integer(event.get("call_index"), f"{role}.{task_id}.call_index")
        if call_index not in {0, 1, 2}:
            raise ValueError(f"{role} invalid LLM call_index for {task_id}")
        if _integer(event.get("attempts"), f"{role}.{task_id}.attempts") != 1:
            raise ValueError(f"{role} LLM request was not exactly once: {task_id}/{call_index}")
        if _boolean(event.get("ok"), f"{role}.{task_id}.llm.ok") is not True:
            raise ValueError(f"{role} LLM request failed: {task_id}/{call_index}")
        if _integer(event.get("http_status"), f"{role}.{task_id}.http_status") != 200:
            raise ValueError(f"{role} LLM HTTP status is not 200: {task_id}/{call_index}")
        _finite(event.get("request_start_s"), f"{role}.{task_id}.request_start_s", positive=True)
        _finite(event.get("duration_s"), f"{role}.{task_id}.duration_s", positive=True)
        _integer(event.get("prompt_tokens_estimate"), f"{role}.{task_id}.prompt_tokens_estimate", positive=True)
        usage = _mapping(event.get("usage"), f"{role}.{task_id}.usage")
        prompt_tokens = _integer(usage.get("prompt_tokens"), f"{role}.{task_id}.usage.prompt_tokens")
        completion_tokens = _integer(usage.get("completion_tokens"), f"{role}.{task_id}.usage.completion_tokens")
        total_tokens = _integer(usage.get("total_tokens"), f"{role}.{task_id}.usage.total_tokens")
        if prompt_tokens + completion_tokens != total_tokens:
            raise ValueError(f"{role} LLM token accounting is inconsistent: {task_id}/{call_index}")
        scheduler_meta = _mapping(event.get("scheduler_meta"), f"{role}.{task_id}.scheduler_meta")
        if scheduler_meta.get("t") != task_id or scheduler_meta.get("c") != call_index or scheduler_meta.get("ms") != "live_broker":
            raise ValueError(f"{role} scheduler metadata identity is invalid: {task_id}/{call_index}")
        for key in ("tqa", "tqs", "tra", "trs"):
            _integer(scheduler_meta.get(key), f"{role}.{task_id}.scheduler_meta.{key}")
        llm_by_task_lists[task_id].append(event)
    if len(raw_llm) != 3 * task_count:
        raise ValueError(f"{role} does not contain exactly three LLM requests per task")
    llm_by_task: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for task_id, task in tasks_by_id.items():
        events = sorted(llm_by_task_lists.get(task_id, []), key=lambda event: int(event["call_index"]))
        if [event["call_index"] for event in events] != [0, 1, 2]:
            raise ValueError(f"{role} LLM call set is incomplete or duplicated: {task_id}")
        prompt_total = sum(int(_mapping(event["usage"], "usage")["prompt_tokens"]) for event in events)
        completion_total = sum(int(_mapping(event["usage"], "usage")["completion_tokens"]) for event in events)
        duration_total = sum(float(event["duration_s"]) for event in events)
        if prompt_total != task["prompt_tokens"] or completion_total != task["completion_tokens"]:
            raise ValueError(f"{role} task and LLM token totals disagree: {task_id}")
        if not _close(duration_total, float(task["llm_duration_s"])):
            raise ValueError(f"{role} task and LLM duration totals disagree: {task_id}")
        llm_by_task[task_id] = tuple(events)

    raw_physical = _sequence(payload.get("tool_attempt_records"), f"{role}.tool_attempt_records")
    physical_records: list[Mapping[str, Any]] = []
    committed_lists: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    invocation_ids: set[str] = set()
    for index, raw_record in enumerate(raw_physical):
        record = _mapping(raw_record, f"{role}.tool_attempt_records[{index}]")
        invocation_id = _string(record.get("invocation_id"), f"{role}.tool[{index}].invocation_id")
        if invocation_id in invocation_ids:
            raise ValueError(f"{role} duplicate physical invocation_id: {invocation_id}")
        invocation_ids.add(invocation_id)
        session_id = _string(record.get("session_id"), f"{role}.tool[{index}].session_id")
        if session_id not in tasks_by_id:
            raise ValueError(f"{role} physical tool record references unknown task: {session_id}")
        tool_name = _string(record.get("tool"), f"{role}.tool[{index}].tool")
        if tool_name not in {"search", "visit"}:
            raise ValueError(f"{role} physical tool record has unsupported tool")
        _sha256_text(record.get("invocation_digest"), f"{role}.tool[{index}].invocation_digest")
        admitted = _boolean(record.get("admitted"), f"{role}.tool[{index}].admitted")
        speculative = _boolean(record.get("speculative"), f"{role}.tool[{index}].speculative")
        authoritative = _boolean(record.get("authoritative"), f"{role}.tool[{index}].authoritative")
        committed = _boolean(record.get("committed"), f"{role}.tool[{index}].committed")
        cancelled = _boolean(record.get("cancelled"), f"{role}.tool[{index}].cancelled")
        outcome = _string(record.get("outcome"), f"{role}.tool[{index}].outcome")
        if "failed" in outcome:
            raise ValueError(f"{role} contains a failed physical tool job")
        if not admitted and (authoritative or committed or not speculative):
            raise ValueError(f"{role} rejected record has invalid lane/commit flags")
        if not admitted and (
            record.get("started_at") is not None
            or record.get("worker_id") is not None
            or _integer(
                record.get("http_attempts"),
                f"{role}.tool[{index}].http_attempts",
            )
            != 0
            or _finite(
                record.get("service_s"), f"{role}.tool[{index}].service_s"
            )
            != 0.0
            or any(
                record.get(field) is not None
                for field in (
                    "backend",
                    "request_host",
                    "response_status",
                    "bytes_read",
                    "transport_identity_source",
                )
            )
        ):
            raise ValueError(f"{role} rejected record claims physical HTTP work")
        if record.get("result_digest") is not None:
            _sha256_text(record.get("result_digest"), f"{role}.tool[{index}].result_digest")
        if admitted and record.get("started_at") is None:
            queued = _finite(
                record.get("queue_enter_at"),
                f"{role}.tool[{index}].queue_enter_at",
            )
            finished = _finite(
                record.get("finished_at"),
                f"{role}.tool[{index}].finished_at",
            )
            queue_s = _finite(
                record.get("queue_s"), f"{role}.tool[{index}].queue_s"
            )
            service_s = _finite(
                record.get("service_s"), f"{role}.tool[{index}].service_s"
            )
            saved_s = _finite(
                record.get("saved_service_s"),
                f"{role}.tool[{index}].saved_service_s",
            )
            if (
                not speculative
                or committed
                or not cancelled
                or outcome not in {"cancelled", "expired"}
                or record.get("worker_id") is not None
                or finished < queued
                or not _close(queue_s, finished - queued, tolerance=0.02)
                or service_s != 0.0
                or saved_s != 0.0
                or _integer(
                    record.get("http_attempts"),
                    f"{role}.tool[{index}].http_attempts",
                )
                != 0
                or any(
                    record.get(field) is not None
                    for field in (
                        "backend",
                        "request_host",
                        "response_status",
                        "bytes_read",
                        "transport_identity_source",
                    )
                )
            ):
                raise ValueError(
                    f"{role} tool record {index} has invalid never-started "
                    "cancellation telemetry"
                )
        elif admitted:
            started = _finite(record.get("started_at"), f"{role}.tool[{index}].started_at")
            finished = _finite(record.get("finished_at"), f"{role}.tool[{index}].finished_at")
            queued = _finite(record.get("queue_enter_at"), f"{role}.tool[{index}].queue_enter_at")
            if finished < started:
                raise ValueError(f"{role} physical tool finish precedes start")
            queue_s = _finite(record.get("queue_s"), f"{role}.tool[{index}].queue_s")
            if not _close(queue_s, started - queued, tolerance=0.02):
                raise ValueError(f"{role} physical tool queue duration is inconsistent")
            service = _finite(record.get("service_s"), f"{role}.tool[{index}].service_s")
            if not _close(service, finished - started):
                raise ValueError(f"{role} physical tool service duration is inconsistent")
            attempts = _integer(
                record.get("http_attempts"),
                f"{role}.tool[{index}].http_attempts",
                positive=True,
            )
            if attempts > retry_config["max_attempts"]:
                raise ValueError(
                    f"{role} tool record {index} exceeds the controlled HTTP "
                    "attempt limit"
                )
            if attempts > 1 and service + 0.01 < retry_config["backoff_s"]:
                raise ValueError(
                    f"{role} tool record {index} service time omits retry backoff"
                )
            if record.get("transport_identity_source") != "actual":
                raise ValueError(
                    f"{role} started tool record {index} lacks actual final "
                    "HTTP evidence"
                )
            backend = _string(
                record.get("backend"), f"{role}.tool[{index}].backend"
            )
            hosts = [
                item
                for item in _string(
                    record.get("request_host"),
                    f"{role}.tool[{index}].request_host",
                ).split(",")
                if item
            ]
            if not hosts:
                raise ValueError(
                    f"{role} started tool record {index} has no request host"
                )
            if _integer(
                record.get("response_status"),
                f"{role}.tool[{index}].response_status",
            ) != 200:
                raise ValueError(
                    f"{role} started tool record {index} final response is not "
                    "HTTP 200"
                )
            _integer(
                record.get("bytes_read"),
                f"{role}.tool[{index}].bytes_read",
                positive=True,
            )
            if tool_name == "search":
                expected_backend = {
                    "bing": ("bing_html_search", lambda host: host == "www.bing.com"),
                    "rest": ("wikipedia_rest_search", lambda host: host.endswith(".wikipedia.org")),
                    "action": ("wikipedia_mediawiki_action", lambda host: host.endswith(".wikipedia.org")),
                }[_string(config.get("search_mode"), "config.search_mode")]
                if backend != expected_backend[0] or any(
                    not expected_backend[1](host) for host in hosts
                ):
                    raise ValueError(
                        f"{role} started search record {index} backend does not "
                        "match actual transport evidence"
                    )
            elif _string(config.get("visit_mode"), "config.visit_mode") == "jina":
                if backend != "r.jina.ai" or set(hosts) != {"r.jina.ai"}:
                    raise ValueError(
                        f"{role} started visit record {index} has invalid actual "
                        "transport evidence"
                    )
            elif backend != "direct_http":
                raise ValueError(
                    f"{role} started visit record {index} has invalid direct backend"
                )
            _string(
                record.get("request_host"),
                f"{role}.tool[{index}].request_host",
            )
        if record.get("response_status") is not None:
            if _integer(record.get("response_status"), f"{role}.tool[{index}].response_status") != 200:
                raise ValueError(f"{role} contains a non-200 physical response")
            _integer(record.get("bytes_read"), f"{role}.tool[{index}].bytes_read", positive=True)
            _string(record.get("backend"), f"{role}.tool[{index}].backend")
            _string(record.get("request_host"), f"{role}.tool[{index}].request_host")
            attempts = _integer(
                record.get("http_attempts"),
                f"{role}.tool[{index}].http_attempts",
                positive=True,
            )
            if attempts > retry_config["max_attempts"]:
                raise ValueError(
                    f"{role} tool record {index} exceeds the controlled HTTP "
                    "attempt limit"
                )
        if committed:
            if not admitted or not authoritative or outcome != "committed" or record.get("cancelled") is not False:
                raise ValueError(f"{role} committed tool record has invalid state")
            committed_lists[(session_id, tool_name)].append(record)
        physical_records.append(record)

    committed_by_task_tool: dict[tuple[str, str], Mapping[str, Any]] = {}
    for task_id, task in tasks_by_id.items():
        raw_tools = _sequence(task.get("tools"), f"{role}.{task_id}.tools")
        for tool_index, tool_name in enumerate(("search", "visit")):
            records = committed_lists.get((task_id, tool_name), [])
            if len(records) != 1:
                raise ValueError(f"{role} requires exactly one authoritative {tool_name} commit for {task_id}")
            record = records[0]
            expected_invocation, expected_result, invocation = task_invocations[(task_id, tool_name)]
            if record.get("invocation_digest") != expected_invocation or record.get("result_digest") != expected_result:
                raise ValueError(f"{role} task and physical {tool_name} identity differ: {task_id}")
            task_tool = _mapping(raw_tools[tool_index], f"{role}.{task_id}.{tool_name}")
            if record.get("source") != task_tool.get("source"):
                raise ValueError(f"{role} task and physical {tool_name} source differ: {task_id}")
            for metric in ("queue_s", "service_s", "exposed_wait_s", "saved_service_s"):
                if not _close(float(record[metric]), float(task_tool[metric])):
                    raise ValueError(f"{role} task and physical {tool_name} {metric} differ: {task_id}")
            canary = _boolean(record.get("canary"), f"{role}.{task_id}.{tool_name}.canary")
            eligible = _boolean(
                record.get("speculation_eligible"),
                f"{role}.{task_id}.{tool_name}.speculation_eligible",
            )
            if tool_name == "search" and canary:
                raise ValueError(f"{role} search call cannot be a visit canary")
            if tool_name == "visit" and (
                canary != bool(task["visit_canary"]) or eligible == canary
            ):
                raise ValueError(f"{role} visit canary evidence is inconsistent: {task_id}")
            _validate_http_record(
                record,
                config=config,
                invocation=invocation,
                label=f"{role}.{task_id}.{tool_name}",
            )
            committed_by_task_tool[(task_id, tool_name)] = record
    if sum(len(records) for records in committed_lists.values()) != 2 * task_count:
        raise ValueError(f"{role} has extra authoritative commits")

    final_snapshot = _mapping(payload.get("broker_final_snapshot"), f"{role}.broker_final_snapshot")
    counts = _mapping(final_snapshot.get("counts"), f"{role}.broker_final_snapshot.counts")
    for key in (
        "queued_authoritative",
        "queued_speculative",
        "running_authoritative",
        "running_speculative",
        "completed_unclaimed_speculative",
    ):
        if _integer(counts.get(key), f"{role}.broker_final_snapshot.counts.{key}") != 0:
            raise ValueError(f"{role} broker was not fully drained")
    for key in ("queued_by_tool", "running_by_tool"):
        if key in counts and _mapping(
            counts.get(key), f"{role}.broker_final_snapshot.counts.{key}"
        ):
            raise ValueError(f"{role} broker retained per-tool work after close")
    if _sequence(final_snapshot.get("jobs"), f"{role}.broker_final_snapshot.jobs"):
        raise ValueError(f"{role} broker retained jobs after close")
    stats = _mapping(final_snapshot.get("stats"), f"{role}.broker_final_snapshot.stats")
    if _integer(stats.get("commits"), f"{role}.broker.stats.commits") != 2 * task_count:
        raise ValueError(f"{role} broker commit count is inconsistent")
    if _integer(stats.get("authoritative_requests"), f"{role}.broker.stats.authoritative_requests") != 2 * task_count:
        raise ValueError(f"{role} authoritative request count is inconsistent")
    if _integer(stats.get("authoritative_failures"), f"{role}.broker.stats.authoritative_failures") != 0:
        raise ValueError(f"{role} contains authoritative tool failures")
    saved_recorded = sum(float(record.get("saved_service_s") or 0.0) for record in physical_records if record.get("committed") is True)
    if not _close(_finite(stats.get("saved_service_s"), f"{role}.broker.stats.saved_service_s"), saved_recorded):
        raise ValueError(f"{role} saved-service accounting is inconsistent")

    vllm = _mapping(payload.get("vllm_metric_deltas"), f"{role}.vllm_metric_deltas")
    event_prompt = sum(int(_mapping(event["usage"], "usage")["prompt_tokens"]) for event in raw_llm)
    event_completion = sum(int(_mapping(event["usage"], "usage")["completion_tokens"]) for event in raw_llm)
    metric_prompt = _finite(vllm.get("vllm:prompt_tokens_total"), f"{role}.vllm.prompt_tokens_total")
    metric_completion = _finite(vllm.get("vllm:generation_tokens_total"), f"{role}.vllm.generation_tokens_total")
    if not _close(metric_prompt, float(event_prompt)) or not _close(metric_completion, float(event_completion)):
        raise ValueError(f"{role} vLLM and raw-event token totals disagree")
    _finite(vllm.get("vllm:request_queue_time_seconds_sum"), f"{role}.vllm.queue_time")

    summary = _mapping(payload.get("summary"), f"{role}.summary")
    if _boolean(summary.get("all_tasks_succeeded"), f"{role}.summary.all_tasks_succeeded") is not True:
        raise ValueError(f"{role} summary does not report all tasks successful")
    if _integer(summary.get("task_count"), f"{role}.summary.task_count") != task_count:
        raise ValueError(f"{role} summary task count differs from raw evidence")
    if _integer(summary.get("successful_task_count"), f"{role}.summary.successful_task_count") != task_count:
        raise ValueError(f"{role} summary successful task count differs")
    if _integer(summary.get("failed_task_count"), f"{role}.summary.failed_task_count") != 0:
        raise ValueError(f"{role} summary contains failed tasks")
    summary_llm = _mapping(summary.get("llm"), f"{role}.summary.llm")
    if _integer(summary_llm.get("request_count"), f"{role}.summary.llm.request_count") != 3 * task_count:
        raise ValueError(f"{role} summary LLM count differs")
    if _integer(
        summary_llm.get("successful_request_count"),
        f"{role}.summary.llm.successful_request_count",
    ) != 3 * task_count:
        raise ValueError(f"{role} summary successful LLM count differs")
    if _boolean(summary_llm.get("exactly_one_attempt_each"), f"{role}.summary.llm.exactly_one_attempt_each") is not True:
        raise ValueError(f"{role} summary does not confirm exactly-once LLM requests")
    summary_tool = _mapping(summary.get("tool"), f"{role}.summary.tool")
    if _integer(
        summary_tool.get("authoritative_commit_count"),
        f"{role}.summary.tool.authoritative_commit_count",
    ) != 2 * task_count:
        raise ValueError(f"{role} summary authoritative commit count differs")

    derived_summary = _summarize_run_components(
        tasks_by_key=tasks_by_key,
        llm_events=tuple(_mapping(event, "llm event") for event in raw_llm),
        physical_records=tuple(physical_records),
        committed=committed_by_task_tool,
        timeline_summary=timeline_summary,
        payload=payload,
    )
    return ValidatedRun(
        path=result_path,
        sha256=_sha256_file(result_path),
        payload=payload,
        config=config,
        tasks_by_key=tasks_by_key,
        tasks_by_id=tasks_by_id,
        llm_by_task=llm_by_task,
        committed_by_task_tool=committed_by_task_tool,
        physical_records=tuple(physical_records),
        timeline=timeline,
        call_graph_mode=call_graph_mode,
        search_coverage=search_coverage,
        summary=derived_summary,
    )


def _summarize_run_components(
    *,
    tasks_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    llm_events: Sequence[Mapping[str, Any]],
    physical_records: Sequence[Mapping[str, Any]],
    committed: Mapping[tuple[str, str], Mapping[str, Any]],
    timeline_summary: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = list(tasks_by_key.values())
    task_e2e = [float(task["e2e_s"]) for task in tasks]
    source_values: dict[str, list[float]] = defaultdict(list)
    for (source_id, _), task in tasks_by_key.items():
        source_values[source_id].append(float(task["e2e_s"]))
    source_e2e = {source: statistics.fmean(values) for source, values in source_values.items()}
    llm_durations = [float(event["duration_s"]) for event in llm_events]
    prompt_tokens = sum(int(_mapping(event["usage"], "usage")["prompt_tokens"]) for event in llm_events)
    completion_tokens = sum(int(_mapping(event["usage"], "usage")["completion_tokens"]) for event in llm_events)
    committed_records = list(committed.values())
    authoritative_retried_commits = sum(
        int(record["http_attempts"]) > 1 for record in committed_records
    )
    started_records = [
        record
        for record in physical_records
        if record.get("admitted") is True and record.get("started_at") is not None
    ]

    def tool_dist(metric: str, tool: str | None = None) -> dict[str, float | int]:
        rows = [record for record in committed_records if tool is None or record["tool"] == tool]
        return _distribution([float(record[metric]) for record in rows])

    speculative = [record for record in physical_records if record.get("speculative") is True and record.get("admitted") is True]
    hits = [record for record in committed_records if record.get("speculative") is True and record.get("exact_match") is True]
    eligible = [record for record in committed_records if record.get("speculation_eligible") is True]
    cancellations = [record for record in physical_records if record.get("cancelled") is True]
    wasted_service = sum(
        float(record.get("service_s") or 0.0)
        for record in speculative
        if record.get("committed") is not True
    )
    physical_service = sum(float(record.get("service_s") or 0.0) for record in physical_records)
    broker_stats = _mapping(
        _mapping(payload["broker_final_snapshot"], "broker_final_snapshot").get("stats"),
        "broker stats",
    )
    reported_waste = _finite(
        broker_stats.get("wasted_speculative_service_s"),
        "broker_stats.wasted_speculative_service_s",
    )
    canary_records = [
        record
        for record in committed_records
        if record.get("tool") == "visit" and record.get("canary") is True
    ]
    task_started = min(float(task["start_wall_s"]) for task in tasks)
    task_ended = max(float(task["end_wall_s"]) for task in tasks)
    task_completion_makespan = _finite(
        payload.get("task_completion_makespan_s"), "task_completion_makespan_s", positive=True
    )
    if task_completion_makespan + 0.25 < task_ended - task_started:
        raise ValueError("task_completion_makespan_s is shorter than the raw task window")
    return {
        "task_e2e_s": _distribution(task_e2e),
        "source_e2e_s": {
            "by_source": dict(sorted(source_e2e.items())),
            "distribution": _distribution(list(source_e2e.values())),
        },
        "task_window_makespan_s": task_ended - task_started,
        "task_completion_makespan_s": task_completion_makespan,
        "llm": {
            "request_duration_s": _distribution(llm_durations),
            "request_count": len(llm_events),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "vllm_metric_deltas": dict(_mapping(payload["vllm_metric_deltas"], "vllm metrics")),
        },
        "tool": {
            "authoritative_commit_count": len(committed_records),
            "authoritative_retried_commit_count": authoritative_retried_commits,
            "authoritative_retry_rate": (
                authoritative_retried_commits / len(committed_records)
            ),
            "committed_http_attempt_count": sum(
                int(record["http_attempts"]) for record in committed_records
            ),
            "exposed_wait_s": tool_dist("exposed_wait_s"),
            "queue_s": tool_dist("queue_s"),
            "service_s": tool_dist("service_s"),
            "by_tool": {
                name: {
                    "exposed_wait_s": tool_dist("exposed_wait_s", name),
                    "queue_s": tool_dist("queue_s", name),
                    "service_s": tool_dist("service_s", name),
                }
                for name in ("search", "visit")
            },
            "speculative_admitted_count": len(speculative),
            "exact_hit_count": len(hits),
            "exact_hit_rate_per_eligible_commit": len(hits) / len(eligible) if eligible else 0.0,
            "queued_promotion_count": sum(record.get("source") == "promoted_from_queue" for record in committed_records),
            "running_promotion_count": sum(record.get("source") == "promoted_inflight" for record in committed_records),
            "completed_reuse_count": sum(record.get("source") == "reused" for record in committed_records),
            "saved_service_s": sum(float(record.get("saved_service_s") or 0.0) for record in committed_records),
            "cancelled_physical_count": len(cancellations),
            "expired_physical_count": sum(record.get("outcome") == "expired" for record in physical_records),
            "rejected_physical_count": sum(record.get("admitted") is False for record in physical_records),
            "physical_job_count": len(physical_records),
            # Kept for output-schema compatibility; this is a job/row count,
            # not the number of wire-level HTTP attempts.
            "physical_attempt_count": len(physical_records),
            "started_physical_job_count": len(started_records),
            "physical_http_attempt_count": sum(
                int(record["http_attempts"]) for record in started_records
            ),
            "retried_physical_job_count": sum(
                int(record["http_attempts"]) > 1 for record in started_records
            ),
            "physical_service_s": physical_service,
            "wasted_speculative_service_s_from_records": wasted_service,
            "wasted_speculative_service_s_broker": reported_waste,
            "wasted_worker_fraction": reported_waste / physical_service if physical_service else 0.0,
            "canary_visit": {
                "count": len(canary_records),
                "exposed_wait_s": _distribution([float(record["exposed_wait_s"]) for record in canary_records]) if canary_records else None,
                "queue_s": _distribution([float(record["queue_s"]) for record in canary_records]) if canary_records else None,
                "service_s": _distribution([float(record["service_s"]) for record in canary_records]) if canary_records else None,
            },
            "broker_stats": dict(broker_stats),
        },
        "queue_timeline": dict(timeline_summary),
    }


def _validate_pair_config(baseline: ValidatedRun, candidate: ValidatedRun) -> None:
    if baseline.call_graph_mode != candidate.call_graph_mode:
        raise ValueError("baseline/candidate call_graph_mode differs")
    baseline_common = {
        key: value for key, value in baseline.config.items() if key not in _PAIR_CONFIG_EXCLUSIONS
    }
    candidate_common = {
        key: value for key, value in candidate.config.items() if key not in _PAIR_CONFIG_EXCLUSIONS
    }
    if baseline_common != candidate_common:
        differing = sorted(
            key
            for key in set(baseline_common) | set(candidate_common)
            if baseline_common.get(key) != candidate_common.get(key)
        )
        raise ValueError(
            "baseline/candidate config differs outside permitted observed/design fields: "
            + ", ".join(differing)
        )


def _bootstrap_sources(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    resamples: int,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    source_ids = sorted(baseline)
    rng = random.Random(BOOTSTRAP_SEED)
    absolute_samples: list[float] = []
    relative_samples: list[float] = []
    for _ in range(resamples):
        sample = [source_ids[rng.randrange(len(source_ids))] for _ in source_ids]
        base_mean = statistics.fmean(baseline[source] for source in sample)
        cand_mean = statistics.fmean(candidate[source] for source in sample)
        absolute_samples.append(base_mean - cand_mean)
        relative_samples.append((base_mean - cand_mean) / base_mean if base_mean else 0.0)
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": resamples,
        "source_unit": "independent_source_mean_over_replicas",
        "absolute_reduction_s_95_ci": [
            _percentile(absolute_samples, 0.025),
            _percentile(absolute_samples, 0.975),
        ],
        "relative_reduction_95_ci": [
            _percentile(relative_samples, 0.025),
            _percentile(relative_samples, 0.975),
        ],
    }


def compare_live_joint_pair(
    baseline_result: Path,
    candidate_result: Path,
    *,
    baseline_timeline: Path | None = None,
    candidate_timeline: Path | None = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    baseline = _validate_run(
        baseline_result, role="baseline", timeline_override=baseline_timeline
    )
    candidate = _validate_run(
        candidate_result, role="candidate", timeline_override=candidate_timeline
    )
    _validate_pair_config(baseline, candidate)
    if set(baseline.tasks_by_key) != set(candidate.tasks_by_key):
        raise ValueError("baseline/candidate task source+replica sets differ")

    identity_rows: list[dict[str, Any]] = []
    source_pairs: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"baseline": [], "candidate": []}
    )
    for key in sorted(baseline.tasks_by_key):
        baseline_task = baseline.tasks_by_key[key]
        candidate_task = candidate.tasks_by_key[key]
        task_id = str(baseline_task["task_id"])
        if candidate_task["task_id"] != task_id:
            raise ValueError(f"paired task_id differs for {key}")
        if baseline_task["question_sha256"] != candidate_task["question_sha256"]:
            raise ValueError(f"paired question identity differs for {task_id}")
        if baseline_task["search_query"] != candidate_task["search_query"]:
            raise ValueError(f"paired search query differs for {task_id}")
        if baseline.call_graph_mode == "frozen":
            if baseline_task["expected_url"] != candidate_task["expected_url"]:
                raise ValueError(f"paired frozen expected_url differs for {task_id}")
            baseline_visit = baseline.committed_by_task_tool[(task_id, "visit")]
            candidate_visit = candidate.committed_by_task_tool[(task_id, "visit")]
            if baseline_visit["invocation_digest"] != candidate_visit["invocation_digest"]:
                raise ValueError(
                    f"paired frozen authoritative visit invocation differs for {task_id}"
                )
        source_id = key[0]
        source_pairs[source_id]["baseline"].append(float(baseline_task["e2e_s"]))
        source_pairs[source_id]["candidate"].append(float(candidate_task["e2e_s"]))
        for tool_name in ("search", "visit"):
            base_record = baseline.committed_by_task_tool[(task_id, tool_name)]
            cand_record = candidate.committed_by_task_tool[(task_id, tool_name)]
            identity_rows.append(
                {
                    "task_id": task_id,
                    "source_id": source_id,
                    "replica": key[1],
                    "tool": tool_name,
                    "invocation_match": base_record["invocation_digest"] == cand_record["invocation_digest"],
                    "result_match": base_record["result_digest"] == cand_record["result_digest"],
                }
            )

    baseline_sources = {
        source: statistics.fmean(values["baseline"])
        for source, values in source_pairs.items()
    }
    candidate_sources = {
        source: statistics.fmean(values["candidate"])
        for source, values in source_pairs.items()
    }
    source_reductions = {
        source: baseline_sources[source] - candidate_sources[source]
        for source in sorted(baseline_sources)
    }
    faster_sources = [source for source, reduction in source_reductions.items() if reduction > 0]

    def identity_summary(tool_name: str | None = None) -> dict[str, Any]:
        rows = [row for row in identity_rows if tool_name is None or row["tool"] == tool_name]
        invocation_matches = sum(bool(row["invocation_match"]) for row in rows)
        result_matches = sum(bool(row["result_match"]) for row in rows)
        both_matches = sum(bool(row["invocation_match"] and row["result_match"]) for row in rows)
        return {
            "pair_count": len(rows),
            "invocation_match_count": invocation_matches,
            "invocation_match_rate": invocation_matches / len(rows),
            "result_match_count": result_matches,
            "result_match_rate": result_matches / len(rows),
            "invocation_and_result_match_count": both_matches,
            "invocation_and_result_match_rate": both_matches / len(rows),
        }

    selected_url_match_count = sum(
        baseline.tasks_by_key[key]["selected_url"] == candidate.tasks_by_key[key]["selected_url"]
        for key in baseline.tasks_by_key
    )
    search_url_list_match_count = sum(
        baseline.tasks_by_key[key]["search_urls"] == candidate.tasks_by_key[key]["search_urls"]
        for key in baseline.tasks_by_key
    )
    overall_identity = identity_summary()
    task_pair_count = len(baseline.tasks_by_key)
    fully_transport_result_identity_matched = (
        overall_identity["invocation_and_result_match_count"] == overall_identity["pair_count"]
        and selected_url_match_count == task_pair_count
        and search_url_list_match_count == task_pair_count
    )
    screen_reasons: list[str] = []
    if baseline.call_graph_mode == "frozen":
        # The frozen workload URL and exact visit invocation are the causal
        # call-graph identity.  Live search coverage and returned bytes remain
        # diagnostics: requiring identical dynamic web content would silently
        # turn a live-tool experiment back into a replay experiment.
        paired_claim_eligible = True
        identity_basis = "frozen_workload_expected_url_and_exact_visit_invocation"
    else:
        paired_claim_eligible = fully_transport_result_identity_matched
        identity_basis = "autonomous_full_invocation_result_and_search_selection"
        if not paired_claim_eligible:
            screen_reasons.append(
                "live authoritative invocation/result identity is not identical in every paired task"
            )

    search_coverage_pair: Mapping[str, Any] | None = None
    if baseline.call_graph_mode == "frozen":
        assert baseline.search_coverage is not None
        assert candidate.search_coverage is not None
        baseline_matched = set(baseline.search_coverage["matched_task_ids"])
        candidate_matched = set(candidate.search_coverage["matched_task_ids"])
        search_coverage_pair = {
            "identity_eligibility_effect": "diagnostic_only",
            "baseline": dict(baseline.search_coverage),
            "candidate": dict(candidate.search_coverage),
            "matched_in_both_count": len(baseline_matched & candidate_matched),
            "baseline_only_count": len(baseline_matched - candidate_matched),
            "candidate_only_count": len(candidate_matched - baseline_matched),
            "matched_in_both_task_ids": sorted(baseline_matched & candidate_matched),
            "baseline_only_task_ids": sorted(baseline_matched - candidate_matched),
            "candidate_only_task_ids": sorted(candidate_matched - baseline_matched),
        }

    baseline_task_dist = _mapping(baseline.summary["task_e2e_s"], "baseline task distribution")
    candidate_task_dist = _mapping(candidate.summary["task_e2e_s"], "candidate task distribution")
    task_comparison = {
        metric: _latency_comparison(float(baseline_task_dist[metric]), float(candidate_task_dist[metric]))
        for metric in ("mean", "p50", "p95", "p99", "max")
    }
    task_comparison["task_window_makespan"] = _latency_comparison(
        float(baseline.summary["task_window_makespan_s"]),
        float(candidate.summary["task_window_makespan_s"]),
    )
    task_comparison["task_completion_makespan"] = _latency_comparison(
        float(baseline.summary["task_completion_makespan_s"]),
        float(candidate.summary["task_completion_makespan_s"]),
    )

    baseline_llm = _mapping(baseline.summary["llm"], "baseline llm")
    candidate_llm = _mapping(candidate.summary["llm"], "candidate llm")
    baseline_llm_dist = _mapping(baseline_llm["request_duration_s"], "baseline llm duration")
    candidate_llm_dist = _mapping(candidate_llm["request_duration_s"], "candidate llm duration")
    llm_comparison = {
        "request_duration_s": {
            metric: _latency_comparison(float(baseline_llm_dist[metric]), float(candidate_llm_dist[metric]))
            for metric in ("mean", "p50", "p95", "p99", "max")
        },
        "request_count": _numeric_comparison(float(baseline_llm["request_count"]), float(candidate_llm["request_count"])),
        "prompt_tokens": _numeric_comparison(float(baseline_llm["prompt_tokens"]), float(candidate_llm["prompt_tokens"])),
        "completion_tokens": _numeric_comparison(float(baseline_llm["completion_tokens"]), float(candidate_llm["completion_tokens"])),
        "total_tokens": _numeric_comparison(float(baseline_llm["total_tokens"]), float(candidate_llm["total_tokens"])),
    }

    baseline_tool = _mapping(baseline.summary["tool"], "baseline tool")
    candidate_tool = _mapping(candidate.summary["tool"], "candidate tool")
    tool_comparison: dict[str, Any] = {}
    for metric in ("exposed_wait_s", "queue_s", "service_s"):
        base_dist = _mapping(baseline_tool[metric], f"baseline tool {metric}")
        cand_dist = _mapping(candidate_tool[metric], f"candidate tool {metric}")
        tool_comparison[metric] = {
            quantile: _latency_comparison(float(base_dist[quantile]), float(cand_dist[quantile]))
            for quantile in ("mean", "p50", "p95", "p99", "max")
        }
    tool_comparison["by_tool"] = {}
    for tool_name in ("search", "visit"):
        tool_comparison["by_tool"][tool_name] = {}
        for metric in ("exposed_wait_s", "queue_s", "service_s"):
            base_dist = _mapping(_mapping(baseline_tool["by_tool"], "baseline by tool")[tool_name][metric], "base tool dist")
            cand_dist = _mapping(_mapping(candidate_tool["by_tool"], "candidate by tool")[tool_name][metric], "candidate tool dist")
            tool_comparison["by_tool"][tool_name][metric] = {
                quantile: _latency_comparison(float(base_dist[quantile]), float(cand_dist[quantile]))
                for quantile in ("mean", "p50", "p95", "p99", "max")
            }
    tool_comparison["authoritative_retry"] = {
        "baseline": {
            "retried_commit_count": baseline_tool[
                "authoritative_retried_commit_count"
            ],
            "commit_count": baseline_tool["authoritative_commit_count"],
            "rate": baseline_tool["authoritative_retry_rate"],
        },
        "candidate": {
            "retried_commit_count": candidate_tool[
                "authoritative_retried_commit_count"
            ],
            "commit_count": candidate_tool["authoritative_commit_count"],
            "rate": candidate_tool["authoritative_retry_rate"],
        },
        "absolute_rate_difference": abs(
            float(baseline_tool["authoritative_retry_rate"])
            - float(candidate_tool["authoritative_retry_rate"])
        ),
        "definition": "authoritative commits with http_attempts>1 / commits",
    }

    baseline_canary = _mapping(baseline_tool["canary_visit"], "baseline canary")
    candidate_canary = _mapping(candidate_tool["canary_visit"], "candidate canary")
    if baseline_canary["count"] != candidate_canary["count"]:
        raise ValueError("paired visit-canary counts differ")
    canary_comparison: dict[str, Any] = {"count": baseline_canary["count"]}
    if int(baseline_canary["count"]) > 0:
        for metric in ("exposed_wait_s", "queue_s", "service_s"):
            base_dist = _mapping(baseline_canary[metric], f"baseline canary {metric}")
            cand_dist = _mapping(candidate_canary[metric], f"candidate canary {metric}")
            canary_comparison[metric] = {
                quantile: _latency_comparison(float(base_dist[quantile]), float(cand_dist[quantile]))
                for quantile in ("mean", "p50", "p95", "p99", "max")
            }

    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "baseline": {"path": str(baseline.path), "sha256": baseline.sha256},
            "candidate": {"path": str(candidate.path), "sha256": candidate.sha256},
        },
        "design": {
            "call_graph_mode": baseline.call_graph_mode,
            "baseline_speculation_mode": baseline.config["speculation_mode"],
            "candidate_speculation_mode": candidate.config["speculation_mode"],
            "independent_source_count": len(baseline_sources),
            "task_pair_count": task_pair_count,
            "replicas": baseline.config["replicas"],
            "same_config_except_cell_label_speculation_mode_and_observed_search_coverage": True,
            "all_tasks_successful": True,
            "llm_exactly_once_three_calls_per_task": True,
            "authoritative_search_and_visit_exactly_once_per_task": True,
            "real_http_transport_verified": True,
            "every_started_job_actual_final_http_200": True,
            "zero_failed_physical_jobs": True,
            "controlled_http_retry_policy": _validate_retry_config(
                baseline.config, label="baseline.config"
            ),
            "raw_queue_evidence_verified": True,
        },
        "claim_scope": {
            "screen_only": not paired_claim_eligible,
            "identity_matched_paired_claim_eligible": paired_claim_eligible,
            "identity_basis": identity_basis,
            "full_transport_result_identity_matched": fully_transport_result_identity_matched,
            "reasons": screen_reasons,
        },
        "identity_pairing": {
            "call_graph_mode": baseline.call_graph_mode,
            "claim_identity_basis": identity_basis,
            "overall": overall_identity,
            "search": identity_summary("search"),
            "visit": identity_summary("visit"),
            "selected_url_match_count": selected_url_match_count,
            "selected_url_match_rate": selected_url_match_count / task_pair_count,
            "search_url_list_match_count": search_url_list_match_count,
            "search_url_list_match_rate": search_url_list_match_count / task_pair_count,
            "mismatches": [row for row in identity_rows if not (row["invocation_match"] and row["result_match"])],
            "frozen_search_coverage": search_coverage_pair,
        },
        "baseline": dict(baseline.summary),
        "candidate": dict(candidate.summary),
        "comparison": {
            "task_e2e_s": task_comparison,
            "source_paired": {
                "baseline_mean_s": statistics.fmean(baseline_sources.values()),
                "candidate_mean_s": statistics.fmean(candidate_sources.values()),
                "mean_absolute_reduction_s": statistics.fmean(source_reductions.values()),
                "aggregate_relative_reduction": (
                    (statistics.fmean(baseline_sources.values()) - statistics.fmean(candidate_sources.values()))
                    / statistics.fmean(baseline_sources.values())
                ),
                "faster_source_count": len(faster_sources),
                "faster_source_fraction": len(faster_sources) / len(baseline_sources),
                "by_source_absolute_reduction_s": source_reductions,
                "bootstrap": _bootstrap_sources(
                    baseline_sources,
                    candidate_sources,
                    resamples=bootstrap_resamples,
                ),
            },
            "llm": llm_comparison,
            "tool": tool_comparison,
            "canary_visit": canary_comparison,
            "speculation": {
                key: candidate_tool[key]
                for key in (
                    "speculative_admitted_count",
                    "exact_hit_count",
                    "exact_hit_rate_per_eligible_commit",
                    "queued_promotion_count",
                    "running_promotion_count",
                    "completed_reuse_count",
                    "saved_service_s",
                    "cancelled_physical_count",
                    "expired_physical_count",
                    "rejected_physical_count",
                    "physical_job_count",
                    "physical_attempt_count",
                    "started_physical_job_count",
                    "physical_http_attempt_count",
                    "retried_physical_job_count",
                    "physical_service_s",
                    "wasted_speculative_service_s_from_records",
                    "wasted_speculative_service_s_broker",
                    "wasted_worker_fraction",
                )
            },
            "queue_timeline": {
                "baseline": baseline.summary["queue_timeline"],
                "candidate": candidate.summary["queue_timeline"],
            },
        },
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", required=True, type=Path)
    parser.add_argument("--candidate-result", required=True, type=Path)
    parser.add_argument("--baseline-timeline", type=Path)
    parser.add_argument("--candidate-timeline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = compare_live_joint_pair(
            args.baseline_result,
            args.candidate_result,
            baseline_timeline=args.baseline_timeline,
            candidate_timeline=args.candidate_timeline,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        _write_json_atomic(args.output, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"live pair validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["comparison"]["source_paired"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
