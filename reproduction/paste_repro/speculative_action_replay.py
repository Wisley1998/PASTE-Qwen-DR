"""Lossless Speculative Actions adapter for recorded Qwen-DR traces.

The authoritative model is never called by this module.  A small model may
predict the next tool invocation from the messages visible at the start of a
recorded LLM turn.  Evaluation only reuses a prediction when its tool name and
complete canonical JSON argument object exactly match the following recorded
tool call.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .invocation import Invocation
from .traces import LLMCall, SessionTrace, ToolCall, load_sessions


CASE_SCHEMA = "paste_repro.speculative_action_case.v1"
PREDICTION_SCHEMA = "paste_repro.speculative_action_prediction.v1"
REPORT_SCHEMA = "paste_repro.speculative_action_replay.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SpeculationCase:
    case_id: str
    session_id: str
    llm_call_index: int
    llm_line_number: int
    tool_call_index: int
    tool_line_number: int
    overlap_window_s: float
    llm_total_time_s: float
    tool_duration_s: float
    authoritative_tool_name: str
    authoritative_tool_args: dict[str, Any]
    prompt: str
    prompt_truncated: bool

    @property
    def authoritative_invocation(self) -> Invocation:
        return Invocation(self.authoritative_tool_name, self.authoritative_tool_args)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CASE_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpeculationCase":
        if payload.get("schema") != CASE_SCHEMA:
            raise ValueError(f"unsupported case schema: {payload.get('schema')!r}")
        fields = {name: payload[name] for name in cls.__dataclass_fields__}
        return cls(**fields)


def _clip_text(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    if budget < 80:
        return text[:budget], True
    head = int(budget * 0.72)
    tail = budget - head
    return (
        text[:head]
        + f"\n... <{len(text) - budget} characters omitted> ...\n"
        + text[-tail:],
        True,
    )


def build_prediction_prompt(
    call: LLMCall,
    *,
    top_k: int = 3,
    max_context_chars: int = 36_000,
) -> tuple[str, bool]:
    """Build a compact, causal prompt without exposing the recorded response."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_context_chars < 2_000:
        raise ValueError("max_context_chars must be at least 2000")

    original_user = ""
    recent: list[tuple[str, str]] = []
    for message in call.messages:
        role = str(message.get("role", "unknown"))
        content = message.get("content", "")
        if not isinstance(content, str):
            content = canonical_json(content)
        if role == "user" and not original_user and "<tool_response>" not in content:
            original_user = content
        if role != "system":
            recent.append((role, content))

    # The most recent tool response and assistant rationale carry nearly all
    # action-selection signal.  Keeping their head is important because search
    # URLs normally appear before long page bodies.
    fixed = (
        "You are the fast Speculator in a lossless Speculative Actions system. "
        "Predict the authoritative agent's NEXT tool call from only the causal "
        "context below. Available tools are search(query: list[str]), "
        "visit(url: list[str], goal: str), and google_scholar(query: list[str]). "
        "Copy URLs and argument strings exactly when they are visible. Do not "
        "answer the research question. Return strict JSON only, with this shape: "
        f'{{"predictions":[{{"tool_name":"...","tool_args":{{...}}}}]}}. '
        f"Return at most {top_k} distinct predictions, ordered most likely first. "
        "If no tool call is likely, return {\"predictions\":[]}. /no_think\n\n"
    )
    task, task_cut = _clip_text(original_user, min(4_000, max_context_chars // 6))
    remaining = max_context_chars - len(fixed) - len(task) - 64
    selected: list[str] = []
    truncated = task_cut
    for role, content in reversed(recent[-6:]):
        if remaining <= 0:
            truncated = True
            break
        per_message = max(500, remaining // max(1, min(3, len(recent))))
        clipped, cut = _clip_text(content, min(remaining, per_message))
        selected.append(f"[{role}]\n{clipped}")
        remaining -= len(selected[-1]) + 2
        truncated = truncated or cut
    selected.reverse()
    if len(recent) > len(selected):
        truncated = True
    prompt = fixed + f"Original user task:\n{task}\n\nRecent causal context:\n" + "\n\n".join(selected)
    prompt, final_cut = _clip_text(prompt, max_context_chars)
    return prompt, truncated or final_cut


def _corrected_tool_duration(call: ToolCall) -> float | None:
    correction = call.timing_correction or {}
    value = correction.get("duration_s")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    return None


def _fallback_tool_duration(session: SessionTrace, tool_position: int) -> float:
    call = session.events[tool_position]
    assert isinstance(call, ToolCall)
    for event in session.events[tool_position + 1 :]:
        if isinstance(event, LLMCall):
            return max(0.0, event.start_timestamp_s - call.timestamp_s)
    return 0.0


def build_cases(
    trace_dir: Path,
    *,
    top_k: int = 3,
    max_context_chars: int = 36_000,
    trace_limit: int | None = None,
) -> tuple[tuple[SpeculationCase, ...], tuple[SessionTrace, ...]]:
    sessions = load_sessions(trace_dir)
    if trace_limit is not None:
        if trace_limit <= 0:
            raise ValueError("trace_limit must be positive")
        sessions = sessions[:trace_limit]
    cases: list[SpeculationCase] = []
    for session in sessions:
        for position, event in enumerate(session.events[:-1]):
            next_event = session.events[position + 1]
            if not isinstance(event, LLMCall) or not isinstance(next_event, ToolCall):
                continue
            prompt, truncated = build_prediction_prompt(
                event, top_k=top_k, max_context_chars=max_context_chars
            )
            duration = _corrected_tool_duration(next_event)
            if duration is None:
                duration = _fallback_tool_duration(session, position + 1)
            identity = (
                f"{session.session_id}\0{event.line_number}\0"
                f"{next_event.line_number}\0{next_event.invocation.key}"
            )
            cases.append(
                SpeculationCase(
                    case_id=sha256_bytes(identity.encode("utf-8"))[:24],
                    session_id=session.session_id,
                    llm_call_index=event.call_index,
                    llm_line_number=event.line_number,
                    tool_call_index=next_event.call_index,
                    tool_line_number=next_event.line_number,
                    overlap_window_s=event.overlap_window_s,
                    llm_total_time_s=event.total_time_s,
                    tool_duration_s=duration,
                    authoritative_tool_name=next_event.tool_name,
                    authoritative_tool_args=next_event.tool_args,
                    prompt=prompt,
                    prompt_truncated=truncated,
                )
            )
    return tuple(cases), tuple(sessions)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def read_cases(path: Path) -> tuple[SpeculationCase, ...]:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(SpeculationCase.from_dict(json.loads(line)) for line in handle if line.strip())


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def parse_predictions(text: str, *, top_k: int) -> tuple[Invocation, ...]:
    """Parse and de-duplicate a model response; malformed entries fail closed."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    payload = _extract_json_object(text)
    if isinstance(payload, Mapping):
        raw = payload.get("predictions", payload.get("actions", []))
        if not isinstance(raw, list) and (
            "tool_name" in payload or "name" in payload
        ):
            raw = [payload]
    elif isinstance(payload, list):
        raw = payload
    else:
        raise ValueError("prediction response must be an object or list")
    if not isinstance(raw, list):
        raise ValueError("predictions must be a list")

    result: list[Invocation] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = item.get("tool_name", item.get("name"))
        arguments = item.get("tool_args", item.get("arguments"))
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            continue
        try:
            invocation = Invocation(name, arguments)
        except (TypeError, ValueError):
            continue
        if invocation.key in seen:
            continue
        seen.add(invocation.key)
        result.append(invocation)
        if len(result) >= top_k:
            break
    return tuple(result)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _prediction_invocations(row: Mapping[str, Any], top_k: int) -> tuple[Invocation, ...]:
    raw = row.get("predictions", [])
    if not isinstance(raw, list):
        return ()
    wire = canonical_json({"predictions": raw})
    try:
        return parse_predictions(wire, top_k=top_k)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()


def evaluate_predictions(
    cases: Sequence[SpeculationCase],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Evaluate exact-match speculation with measured model latency."""

    by_case = {str(row.get("case_id")): row for row in prediction_rows}
    if len(by_case) != len(prediction_rows):
        raise ValueError("duplicate case_id in prediction rows")

    tool_priors: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        tool_priors[case.authoritative_tool_name].append(case.tool_duration_s)
    median_duration = {
        name: statistics.median(values) for name, values in tool_priors.items()
    }
    global_median = statistics.median(
        [case.tool_duration_s for case in cases]
    ) if cases else 0.0

    details: list[dict[str, Any]] = []
    total_saved = 0.0
    hits = 0
    ontime_hits = 0
    predicted_count = 0
    launched_predicted_count = 0
    ontime_prediction_cases = 0
    completed_before_authority = 0
    candidate_cases = 0
    clean_empty_cases = 0
    estimated_launched_service = 0.0
    latencies: list[float] = []
    errors = 0
    per_tool_accumulator: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "cases": 0.0,
            "hits": 0.0,
            "ontime_hits": 0.0,
            "saved_s": 0.0,
            "baseline_tool_s": 0.0,
        }
    )

    for case in cases:
        row = by_case.get(case.case_id, {})
        latency_raw = row.get("latency_s", 0.0)
        latency = (
            max(0.0, float(latency_raw))
            if isinstance(latency_raw, (int, float)) and not isinstance(latency_raw, bool)
            else 0.0
        )
        if row:
            latencies.append(latency)
        if row.get("error"):
            errors += 1
        predictions = _prediction_invocations(row, top_k)
        if predictions:
            candidate_cases += 1
        elif row and not row.get("error"):
            clean_empty_cases += 1
        predicted_count += len(predictions)
        authority = case.authoritative_invocation
        hit_rank = next(
            (index + 1 for index, prediction in enumerate(predictions) if prediction == authority),
            None,
        )
        head_start = max(0.0, case.overlap_window_s - latency)
        if row and head_start > 0.0:
            completed_before_authority += 1
        prediction_on_time = bool(predictions) and head_start > 0.0
        if prediction_on_time:
            ontime_prediction_cases += 1
            launched_predicted_count += len(predictions)
            for prediction in predictions:
                estimated_launched_service += median_duration.get(
                    prediction.tool_name, global_median
                )
        saved = min(case.tool_duration_s, head_start) if hit_rank is not None else 0.0
        hit = hit_rank is not None
        if hit:
            hits += 1
        if hit and saved > 0.0:
            ontime_hits += 1
        total_saved += saved
        tool_row = per_tool_accumulator[case.authoritative_tool_name]
        tool_row["cases"] += 1
        tool_row["hits"] += float(hit)
        tool_row["ontime_hits"] += float(hit and saved > 0.0)
        tool_row["saved_s"] += saved
        tool_row["baseline_tool_s"] += case.tool_duration_s
        details.append(
            {
                "case_id": case.case_id,
                "session_id": case.session_id,
                "llm_call_index": case.llm_call_index,
                "tool_call_index": case.tool_call_index,
                "authoritative": authority.to_dict(),
                "predictions": [prediction.to_dict() for prediction in predictions],
                "hit_rank": hit_rank,
                "speculator_latency_s": latency,
                "overlap_window_s": case.overlap_window_s,
                "tool_duration_s": case.tool_duration_s,
                "available_head_start_s": head_start,
                "saved_tool_stall_s": saved,
                "exposed_tool_stall_s": case.tool_duration_s - saved,
                "prompt_truncated": case.prompt_truncated,
                "error": row.get("error"),
            }
        )

    baseline_tool = sum(case.tool_duration_s for case in cases)
    speculative_tool = baseline_tool - total_saved
    session_llm: dict[str, float] = defaultdict(float)
    session_tool: dict[str, float] = defaultdict(float)
    for case in cases:
        session_tool[case.session_id] += case.tool_duration_s
    # A case exists for every non-terminal LLM call.  The terminal calls are
    # supplied by the preparation manifest and added by the CLI before report
    # publication; this component sum is still useful for unit tests.
    for case in cases:
        session_llm[case.session_id] += case.llm_total_time_s
    component_llm = sum(session_llm.values())
    baseline_e2e = component_llm + baseline_tool
    speculative_e2e = baseline_e2e - total_saved
    # Late model outputs cannot launch a tool.  An on-time exact match replaces
    # its corresponding demand call rather than adding another physical call.
    physical_calls = len(cases) + launched_predicted_count - ontime_hits

    per_tool: dict[str, Any] = {}
    for name, values in sorted(per_tool_accumulator.items()):
        count = int(values["cases"])
        base = values["baseline_tool_s"]
        exposed = base - values["saved_s"]
        per_tool[name] = {
            "cases": count,
            "exact_hits": int(values["hits"]),
            "exact_hit_rate": values["hits"] / count if count else 0.0,
            "on_time_exact_hits": int(values["ontime_hits"]),
            "on_time_exact_hit_rate": values["ontime_hits"] / count if count else 0.0,
            "baseline_tool_stall_s": base,
            "speculative_tool_stall_s": exposed,
            "tool_stall_reduction": values["saved_s"] / base if base else 0.0,
        }

    return {
        "schema": REPORT_SCHEMA,
        "contract": {
            "authoritative_llm_replayed": False,
            "authoritative_trace_immutable": True,
            "match": "exact tool name plus canonical complete JSON arguments",
            "commit": "reuse exact matches only; all misses remain demand execution",
            "speculator_resource_isolation": "separate GPU/endpoint required",
        },
        "summary": {
            "eligible_tool_calls": len(cases),
            "prediction_rows": len(by_case),
            "prediction_errors": errors,
            "prediction_cases_with_candidates": candidate_cases,
            "clean_empty_prediction_cases": clean_empty_cases,
            "parsed_candidate_misses": candidate_cases - hits,
            "exact_hits": hits,
            "exact_misses": len(cases) - hits,
            "exact_hit_rate": hits / len(cases) if cases else 0.0,
            "on_time_exact_hits": ontime_hits,
            "effective_misses": len(cases) - ontime_hits,
            "on_time_exact_hit_rate": ontime_hits / len(cases) if cases else 0.0,
            "predicted_invocations": predicted_count,
            "requests_completed_before_authority": completed_before_authority,
            "requests_late_for_authority": len(cases) - completed_before_authority,
            "on_time_prediction_cases": ontime_prediction_cases,
            "launched_speculative_invocations": launched_predicted_count,
            "physical_tool_invocations": physical_calls,
            "tool_call_amplification": physical_calls / len(cases) if cases else 0.0,
            "estimated_launched_tool_service_s": estimated_launched_service,
            "speculator_latency_mean_s": statistics.fmean(latencies) if latencies else 0.0,
            "speculator_latency_p50_s": percentile(latencies, 0.50),
            "speculator_latency_p95_s": percentile(latencies, 0.95),
            "baseline_tool_stall_s": baseline_tool,
            "speculative_tool_stall_s": speculative_tool,
            "saved_tool_stall_s": total_saved,
            "tool_stall_reduction": total_saved / baseline_tool if baseline_tool else 0.0,
            "tool_stall_speedup": baseline_tool / speculative_tool if speculative_tool else None,
            "component_baseline_e2e_s": baseline_e2e,
            "component_speculative_e2e_s": speculative_e2e,
            "component_e2e_reduction": total_saved / baseline_e2e if baseline_e2e else 0.0,
            "component_e2e_speedup": baseline_e2e / speculative_e2e if speculative_e2e else None,
        },
        "per_tool": per_tool,
        "cases": details,
    }
