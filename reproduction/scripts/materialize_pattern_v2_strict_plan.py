#!/usr/bin/env python3
"""Materialize the strict 100-root x 2 Qwen Pattern V2 replay plan.

Only the trace call graph, current LLM messages, and tool arguments are copied.
The loader deliberately never accesses ``timestamp``, LLM timing fields,
``timing_correction``, or recorded responses.  Physical service is instead
frozen by a separate private normalized-invocation hashed SLO clock:
Search/GoogleScholar 1--3 seconds and each executable Visit URL 2--8 seconds.

The output uses the established strict public/sealed plan schemas.  The public
plan is metadata-only; the sealed cursor releases one current request and then
reveals its following authority only after the live LLM completes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
from typing import Any
from urllib.parse import urlsplit


SCRIPT = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = SCRIPT.parents[2]
for import_root in (REPRODUCTION_ROOT, SCRIPT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from paste_repro import pattern_v2_all_visit_online as pattern_online  # noqa: E402
from paste_repro.pattern_v2_strict_adapter import (  # noqa: E402
    new_hashed_slo_clock_artifact,
    new_public_slo_duration_artifact,
)
from paste_repro.strict_trace_runtime import (  # noqa: E402
    CausalTailPredictor,
    canonical_sha256,
    signed_payload,
    validate_signed_payload,
)
from paste_repro.traces import LLMCall, SessionTrace, ToolCall  # noqa: E402
from run_strict_trace_abef import (  # noqa: E402
    PUBLIC_PLAN_SCHEMA,
    SEALED_PLAN_SCHEMA,
    _prepare_request,
    file_sha256,
    read_json,
    write_json,
)


MANIFEST_SCHEMA = "paste_repro.pattern_v2_strict_materialization.v1"
DEFAULT_TRACES = (
    REPOSITORY_ROOT
    / "traces/my_traces_tool_slo_search_uniform_1_3s_"
    "visit_serial_uniform_2_8s_llm_x0_42"
)
EXACT_SOURCE_ROOTS = 100
EXACT_REPLICAS = 2
DEPLOYABLE_SOURCE_ROOTS = 30
DEPLOYABLE_REPLICAS = 7
CROSSFIT_LOGICAL_CORPUS_SHA256 = (
    "c8eddcf9376754cc37056a1a1af7a42b5e786d7ed8c4af65d86f904431030fbc"
)
DEPLOYABLE_LOGICAL_CORPUS_SHA256 = (
    "34857c0cab48aa604db8907face0654e7b892a7a3b626cedd0188d79994030a7"
)
ROOT_IDS_SCHEMA = "paste_repro.strict_root_ids.v1"
ALLOWED_MESSAGE_KEYS = frozenset({"role", "content"})
ALLOWED_TOOL_ARGUMENT_KEYS = {
    "search": frozenset({"query"}),
    "google_scholar": frozenset({"query"}),
    "visit": frozenset({"url", "goal"}),
}


def _load_causal_source(path: Path) -> SessionTrace:
    """Parse only policy-independent content fields from a JSONL source.

    Timing keys are not even retrieved from each decoded mapping.  The zeroes
    in the dataclasses are inert placeholders needed by the common request
    preparation helper; they never enter an output artifact.
    """

    events: list[LLMCall | ToolCall] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            event_type = payload.get("event_type")
            call_index = payload.get("call_index", 0)
            if isinstance(call_index, bool) or not isinstance(call_index, int):
                raise ValueError(f"{path}:{line_number}: invalid call_index")
            if event_type == "llm_call":
                messages = payload.get("messages")
                if not isinstance(messages, list) or not all(
                    isinstance(message, Mapping) for message in messages
                ):
                    raise ValueError(f"{path}:{line_number}: invalid messages")
                for message in messages:
                    unknown = set(message) - ALLOWED_MESSAGE_KEYS
                    if unknown:
                        raise ValueError(
                            f"{path}:{line_number}: unregistered message fields: "
                            f"{sorted(unknown)}"
                        )
                    if not isinstance(message.get("role"), str) or not isinstance(
                        message.get("content"), str
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: message role/content must be strings"
                        )
                events.append(
                    LLMCall(
                        call_index=call_index,
                        timestamp_s=0.0,
                        total_time_s=0.0,
                        inference_time_s=0.0,
                        messages=tuple(dict(message) for message in messages),
                        response="",
                        line_number=line_number,
                    )
                )
            elif event_type == "tool_call":
                tool_name = payload.get("tool_name")
                tool_args = payload.get("tool_args")
                if not isinstance(tool_name, str) or not tool_name:
                    raise ValueError(f"{path}:{line_number}: invalid tool_name")
                if not isinstance(tool_args, Mapping):
                    raise ValueError(f"{path}:{line_number}: invalid tool_args")
                allowed = ALLOWED_TOOL_ARGUMENT_KEYS.get(tool_name)
                if allowed is None:
                    raise ValueError(f"{path}:{line_number}: unsupported tool {tool_name!r}")
                unknown = set(tool_args) - allowed
                if unknown:
                    raise ValueError(
                        f"{path}:{line_number}: unregistered {tool_name} arguments: "
                        f"{sorted(unknown)}"
                    )
                if tool_name in {"search", "google_scholar"}:
                    queries = tool_args.get("query")
                    query_values = [queries] if isinstance(queries, str) else queries
                    if not isinstance(query_values, list) or not all(
                        isinstance(value, str) for value in query_values
                    ):
                        raise ValueError(f"{path}:{line_number}: invalid query argument")
                else:
                    urls = tool_args.get("url")
                    url_values = [urls] if isinstance(urls, str) else urls
                    if not isinstance(url_values, list) or not all(
                        isinstance(value, str) for value in url_values
                    ):
                        raise ValueError(f"{path}:{line_number}: invalid Visit URL argument")
                    if "goal" in tool_args and not isinstance(tool_args["goal"], str):
                        raise ValueError(f"{path}:{line_number}: invalid Visit goal")
                events.append(
                    ToolCall(
                        call_index=call_index,
                        timestamp_s=0.0,
                        tool_name=tool_name,
                        tool_args=dict(tool_args),
                        line_number=line_number,
                        timing_correction=None,
                    )
                )
            # Synthetic completion/accounting rows are not part of the call graph.
    if not events or not isinstance(events[0], LLMCall):
        raise ValueError(f"{path}: causal trace must begin with an LLM call")
    return SessionTrace(path, tuple(events))


def _load_sources(directory: Path) -> tuple[SessionTrace, ...]:
    paths = sorted(directory.glob("trace_*.jsonl"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"no trace_*.jsonl sources in {directory}")
    return tuple(_load_causal_source(path) for path in paths)


def _executable_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _unique_executable_visit_urls(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    raw = arguments.get("url")
    values = [raw] if isinstance(raw, str) else list(raw or [])
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not _executable_url(value) or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return tuple(result)


def _sanitized_tool_arguments(tool: ToolCall) -> dict[str, Any]:
    arguments = json.loads(
        json.dumps(
            tool.tool_args,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if tool.tool_name == "visit":
        urls = list(_unique_executable_visit_urls(arguments))
        arguments["url"] = urls
    return arguments


def _logical_source_sha256(session: SessionTrace) -> str:
    rows: list[dict[str, Any]] = []
    for event in session.events:
        if isinstance(event, LLMCall):
            rows.append(
                {
                    "type": "llm",
                    "call_index": event.call_index,
                    "messages": list(event.messages),
                }
            )
        else:
            rows.append(
                {
                    "type": "tool",
                    "call_index": event.call_index,
                    "tool_name": event.tool_name,
                    "tool_args": _sanitized_tool_arguments(event),
                }
            )
    return canonical_sha256(rows)


def _logical_corpus_sha256(sources: tuple[SessionTrace, ...]) -> str:
    return canonical_sha256(
        [
            {
                "source_session_id": source.session_id,
                "logical_no_timing_sha256": _logical_source_sha256(source),
            }
            for source in sources
        ]
    )


def _load_predictor(path: Path) -> tuple[Any, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("predictor artifact must be an object")
    schema = payload.get("schema")
    if schema == pattern_online.SCHEMA:
        predictor = pattern_online.PatternV2CrossFitPredictor.from_path(path)
    elif schema == pattern_online.DEPLOYABLE_SCHEMA:
        predictor = pattern_online.PatternV2DeployablePredictor.from_path(path)
    else:
        raise ValueError(f"unsupported Pattern V2 predictor schema: {schema!r}")
    return predictor, dict(payload)


def _select_sources(
    sources: tuple[SessionTrace, ...],
    *,
    predictor_schema: str,
    root_ids_artifact: Path | None,
) -> tuple[tuple[SessionTrace, ...], dict[str, Any] | None, str, int, int]:
    """Select frozen evaluation roots without consulting trace outcomes."""

    if predictor_schema == pattern_online.SCHEMA:
        if root_ids_artifact is not None:
            raise ValueError("cross-fit materialization does not accept --root-ids-artifact")
        return sources, None, "crossfit", EXACT_SOURCE_ROOTS, EXACT_REPLICAS
    if predictor_schema != pattern_online.DEPLOYABLE_SCHEMA:
        raise ValueError(f"unsupported Pattern V2 predictor schema: {predictor_schema!r}")
    if root_ids_artifact is None:
        raise ValueError("deployable materialization requires --root-ids-artifact")
    root_payload = validate_signed_payload(
        read_json(root_ids_artifact),
        "artifact_sha256",
        label="deployable root IDs",
    )
    if root_payload.get("schema") != ROOT_IDS_SCHEMA:
        raise ValueError("deployable root-ID artifact has an unsupported schema")
    if root_payload.get("role") != "final":
        raise ValueError("deployable root-ID artifact must have role=final")
    raw_ids = root_payload.get("root_ids")
    if not isinstance(raw_ids, list) or not all(
        isinstance(value, str) and value for value in raw_ids
    ):
        raise ValueError("deployable root-ID artifact has invalid root_ids")
    root_ids = tuple(raw_ids)
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("deployable root-ID artifact contains duplicates")
    if root_payload.get("source_session_ids") != list(root_ids):
        raise ValueError("root_ids and source_session_ids differ")
    if root_payload.get("source_session_ids_sha256") != canonical_sha256(list(root_ids)):
        raise ValueError("root-ID session-list hash mismatch")
    source_by_id = {source.session_id: source for source in sources}
    missing = [source_id for source_id in root_ids if source_id not in source_by_id]
    if missing:
        raise ValueError(f"deployable roots are missing from trace directory: {missing[:3]}")
    selected = tuple(source_by_id[source_id] for source_id in root_ids)
    return (
        selected,
        dict(root_payload),
        "deployable_final",
        DEPLOYABLE_SOURCE_ROOTS,
        DEPLOYABLE_REPLICAS,
    )


def _tokenizer(source: str) -> Any:
    from transformers import AutoTokenizer

    resolved = Path(source)
    if "/" in source and not resolved.exists():
        cache_root = Path(os.getenv("HF_HOME", Path.home() / "hf_cache"))
        snapshots = sorted(
            (
                cache_root / f"models--{source.replace('/', '--')}" / "snapshots"
            ).glob("*")
        )
        if len(snapshots) == 1:
            resolved = snapshots[0]
    return AutoTokenizer.from_pretrained(
        str(resolved if resolved.exists() else source),
        trust_remote_code=True,
        local_files_only=True,
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite materialization: {output_dir}")
    predictor_path = args.predictor_artifact.resolve()
    predictor, predictor_payload = _load_predictor(predictor_path)
    all_sources = _load_sources(args.traces.resolve())
    root_ids_path = (
        args.root_ids_artifact.resolve()
        if args.root_ids_artifact is not None
        else None
    )
    sources, root_ids_payload, role, expected_roots, expected_replicas = _select_sources(
        all_sources,
        predictor_schema=str(predictor_payload.get("schema", "")),
        root_ids_artifact=root_ids_path,
    )
    if args.trace_limit is not None:
        sources = sources[: args.trace_limit]
    formal = len(sources) == expected_roots and args.replicas == expected_replicas
    if not formal and not args.allow_smoke_workload:
        raise ValueError(
            f"formal {role} materialization requires exactly "
            f"{expected_roots} roots x {expected_replicas} replicas"
        )
    logical_corpus_sha256 = _logical_corpus_sha256(sources)
    expected_logical_sha256 = (
        CROSSFIT_LOGICAL_CORPUS_SHA256
        if role == "crossfit"
        else DEPLOYABLE_LOGICAL_CORPUS_SHA256
    )
    if formal and logical_corpus_sha256 != expected_logical_sha256:
        raise ValueError(
            f"formal {role} logical corpus differs from the frozen Pattern V2 trace"
        )
    tail_path = args.tail_artifact.resolve()
    tail_payload = read_json(tail_path)
    tail = CausalTailPredictor(tail_payload)
    if tail_payload.get("uses_evaluation_labels") is not False:
        raise ValueError("tail predictor must declare uses_evaluation_labels=false")
    if role == "deployable_final":
        provenance = tail_payload.get("training_provenance", {})
        tail_training_ids = set(
            provenance.get("session_ids", []) if isinstance(provenance, Mapping) else []
        )
        overlap = tail_training_ids.intersection(source.session_id for source in sources)
        if overlap:
            raise ValueError(
                "tail-predictor training roots overlap deployable evaluation: "
                f"{sorted(overlap)[:3]}"
            )
    for source in sources:
        predictor.start_session(
            source_session_id=source.session_id,
            runtime_session_id=f"preflight-{source.session_id}",
        )
    tokenizer = _tokenizer(args.tokenizer)
    seed = args.clock_seed_sha256 or secrets.token_hex(32)
    service_clock = new_hashed_slo_clock_artifact(seed_sha256=seed)
    duration = new_public_slo_duration_artifact(ewma_alpha=args.duration_ewma_alpha)
    created_at = datetime.now(timezone.utc).isoformat()

    prepared_by_source: dict[str, list[dict[str, Any]]] = {}
    source_stats: dict[str, dict[str, int]] = {}
    for source in sources:
        steps: list[dict[str, Any]] = []
        raw_visit_urls = 0
        executable_visit_urls = 0
        tool_calls = 0
        events = list(source.events)
        for event_index, event in enumerate(events):
            if not isinstance(event, LLMCall):
                continue
            tools_after: list[dict[str, Any]] = []
            cursor = event_index + 1
            while cursor < len(events) and not isinstance(events[cursor], LLMCall):
                tool = events[cursor]
                if not isinstance(tool, ToolCall):
                    raise RuntimeError("causal source contains an unknown event type")
                tool_calls += 1
                arguments = _sanitized_tool_arguments(tool)
                if tool.tool_name == "visit":
                    raw = tool.tool_args.get("url")
                    raw_values = [raw] if isinstance(raw, str) else list(raw or [])
                    raw_visit_urls += sum(isinstance(value, str) for value in raw_values)
                    executable_visit_urls += len(arguments["url"])
                tools_after.append(
                    {
                        "event_index": cursor,
                        "call_index": tool.call_index,
                        "tool_name": tool.tool_name,
                        "tool_args": arguments,
                    }
                )
                cursor += 1
            if len(tools_after) > 1:
                raise ValueError(
                    f"{source.session_id}: more than one tool between LLM turns"
                )
            request = _prepare_request(
                event,
                tokenizer=tokenizer,
                max_model_len=args.max_model_len,
                output_cap=args.output_cap,
            )
            steps.append({"request": request, "tools_after": tools_after})
        prepared_by_source[source.session_id] = steps
        source_stats[source.session_id] = {
            "llm_calls": len(steps),
            "tool_calls": tool_calls,
            "raw_visit_urls": raw_visit_urls,
            "executable_visit_urls": executable_visit_urls,
        }

    public_traces: list[dict[str, Any]] = []
    sealed_steps: dict[str, list[dict[str, Any]]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    lineage: dict[str, dict[str, Any]] = {}
    instance_index = 0
    for replica_index in range(args.replicas):
        for source_root_index, source in enumerate(sources):
            source_digest = hashlib.sha256(
                source.session_id.encode("utf-8")
            ).hexdigest()[:12]
            instance_id = (
                f"{role}-r{replica_index:02d}-s{source_root_index:03d}-"
                f"{source_digest}"
            )
            instance_steps = json.loads(
                json.dumps(prepared_by_source[source.session_id], ensure_ascii=False)
            )
            for step in instance_steps:
                for descriptor in step["tools_after"]:
                    event_index = int(descriptor["event_index"])
                    outcome_id = hashlib.sha256(
                        f"{instance_id}\0{event_index}\0{descriptor['call_index']}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    descriptor["outcome_id"] = outcome_id
                    arguments = descriptor["tool_args"]
                    visit_units = (
                        [{"url": url} for url in arguments.get("url", [])]
                        if descriptor["tool_name"] == "visit"
                        else []
                    )
                    outcomes[outcome_id] = {
                        "session_id": instance_id,
                        "source_session_id": source.session_id,
                        "event_index": event_index,
                        "call_index": int(descriptor["call_index"]),
                        "tool_name": str(descriptor["tool_name"]),
                        "visit_units": visit_units,
                        "source": "hashed_uniform_SLO_no_trace_timing",
                    }
            public_traces.append(
                {
                    "trace_id": instance_id,
                    "session_id": instance_id,
                    "source_session_id": source.session_id,
                    "source_root_index": source_root_index,
                    "replica_index": replica_index,
                    "release_offset_s": 0.0,
                    "arrival": {
                        "kind": "closed_burst",
                        "arrival_index": instance_index,
                        "release_offset_s": 0.0,
                    },
                }
            )
            sealed_steps[instance_id] = instance_steps
            lineage[instance_id] = {
                "source_session_id": source.session_id,
                "source_root_index": source_root_index,
                "replica_index": replica_index,
                "raw_source_sha256": file_sha256(source.path),
                "logical_no_timing_sha256": _logical_source_sha256(source),
            }
            instance_index += 1

    disclosure = {
        key: predictor_payload.get(key)
        for key in (
            "schema",
            "evaluation_regime",
            "claim_scope",
            "uses_other_evaluation_root_labels",
            "prior_policy_development_used_evaluation_corpus",
            "predictor_uses_trace_timing",
        )
        if key in predictor_payload
    }
    public = signed_payload(
        {
            "schema": PUBLIC_PLAN_SCHEMA,
            "role": role if formal else "smoke",
            "claim_scope": predictor_payload.get("claim_scope", "retrospective_crossfit"),
            "predictor_disclosure": disclosure,
            "call_graph_mode": "trace_replay_causal_reveal",
            "output_budget_policy": "uniform_public_cap_no_trace_response_length",
            "physical_tool_service_policy": "private_hashed_uniform_SLO_no_trace_timing",
            "invalid_visit_url_policy": "non_http_url_filtered_zero_physical_units",
            "max_model_len": args.max_model_len,
            "output_cap": args.output_cap,
            "arrival_process": {
                "kind": "closed_burst",
                "tasks": len(public_traces),
                "release_span_s": 0.0,
            },
            "independent_source_roots": len(sources),
            "replicas_per_root": args.replicas,
            "replicas": len(public_traces),
            "logical_corpus_sha256": logical_corpus_sha256,
            "predictor_artifact_sha256": predictor.artifact_sha256,
            "duration_predictor_artifact_sha256": duration["artifact_sha256"],
            "tail_predictor_artifact_sha256": tail.artifact_sha256,
            "root_ids_artifact_sha256": (
                root_ids_payload.get("artifact_sha256")
                if root_ids_payload is not None
                else None
            ),
            "traces": public_traces,
        },
        "plan_sha256",
    )
    sealed = signed_payload(
        {
            "schema": SEALED_PLAN_SCHEMA,
            "role": public["role"],
            "claim_scope": public["claim_scope"],
            "public_plan_sha256": public["plan_sha256"],
            "access_contract": (
                "CausalTraceCursor releases one current request; following authority "
                "is revealed only after live LLM completion"
            ),
            "trace_steps": sealed_steps,
            "trace_lineage": lineage,
            "outcomes": outcomes,
            "service_clock_artifact_sha256": service_clock["artifact_sha256"],
        },
        "sealed_sha256",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    predictor_copy = output_dir / "pattern_v2_predictor.json"
    tail_copy = output_dir / "tail_predictor.json"
    shutil.copy2(predictor_path, predictor_copy)
    shutil.copy2(tail_path, tail_copy)
    write_json(output_dir / "public_plan.json", public)
    write_json(output_dir / "sealed_plan.json", sealed)
    write_json(output_dir / "private_service_clock.json", service_clock)
    write_json(output_dir / "public_duration_predictor.json", duration)
    manifest = signed_payload(
        {
            "schema": MANIFEST_SCHEMA,
            "created_at": created_at,
            "formal_workload": formal,
            "source_trace_directory": str(args.traces.resolve()),
            "source_roots": len(sources),
            "logical_corpus_sha256": logical_corpus_sha256,
            "expected_logical_corpus_sha256": expected_logical_sha256,
            "replicas_per_root": args.replicas,
            "tasks": len(public_traces),
            "source_totals": {
                key: sum(row[key] for row in source_stats.values())
                for key in (
                    "llm_calls",
                    "tool_calls",
                    "raw_visit_urls",
                    "executable_visit_urls",
                )
            },
            "trace_timing_fields_accessed": False,
            "trace_timing_fields_materialized": False,
            "files": {
                name: {
                    "path": path.name,
                    "sha256": file_sha256(path),
                }
                for name, path in {
                    "public_plan": output_dir / "public_plan.json",
                    "sealed_plan": output_dir / "sealed_plan.json",
                    "predictor_artifact": predictor_copy,
                    "tail_predictor": tail_copy,
                    "duration_predictor": output_dir
                    / "public_duration_predictor.json",
                    "private_service_clock": output_dir
                    / "private_service_clock.json",
                }.items()
            },
            "predictor_disclosure": disclosure,
            "workload_role": role,
            "root_ids_artifact": (
                {
                    "path": str(root_ids_path),
                    "artifact_sha256": root_ids_payload["artifact_sha256"],
                    "file_sha256": file_sha256(root_ids_path),
                }
                if root_ids_payload is not None and root_ids_path is not None
                else None
            ),
            "tail_predictor_disclosure": {
                "artifact_sha256": tail.artifact_sha256,
                "training_role": tail_payload.get("training_role"),
                "uses_evaluation_labels": tail_payload.get("uses_evaluation_labels"),
            },
        },
        "manifest_sha256",
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--predictor-artifact", type=Path, required=True)
    parser.add_argument("--tail-artifact", type=Path, required=True)
    parser.add_argument(
        "--root-ids-artifact",
        type=Path,
        help="required final-root artifact for a deployable predictor",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--output-cap", type=int, default=128)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--duration-ewma-alpha", type=float, default=0.35)
    parser.add_argument("--clock-seed-sha256")
    parser.add_argument("--trace-limit", type=int)
    parser.add_argument("--allow-smoke-workload", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    if args.replicas <= 0:
        raise ValueError("replicas must be positive")
    if args.max_model_len <= 1 or not 0 < args.output_cap < args.max_model_len:
        raise ValueError("invalid model/output token budget")
    if not 0.0 < args.duration_ewma_alpha <= 1.0:
        raise ValueError("duration EWMA alpha must be in (0, 1]")
    if args.trace_limit is not None and args.trace_limit <= 0:
        raise ValueError("trace limit must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    result = materialize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
