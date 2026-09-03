"""Trace loading and extraction for the repository's JSONL sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Optional, Union

from .invocation import Invocation


class TraceFormatError(ValueError):
    """Raised when a JSONL trace cannot be interpreted safely."""


def _number(value: Any, *, field: str, path: Path, line_number: int) -> float:
    if isinstance(value, bool):
        raise TraceFormatError(f"{path}:{line_number}: {field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TraceFormatError(f"{path}:{line_number}: {field} must be numeric") from exc


@dataclass(frozen=True)
class LLMCall:
    call_index: int
    timestamp_s: float
    total_time_s: float
    inference_time_s: float
    messages: tuple[dict[str, Any], ...]
    response: str
    line_number: int

    @property
    def start_timestamp_s(self) -> float:
        # Trace timestamps are completion timestamps.
        return max(0.0, self.timestamp_s - self.total_time_s)

    @property
    def overlap_window_s(self) -> float:
        # Inference is the portion during which the future call is generated.
        return self.inference_time_s if self.inference_time_s > 0 else self.total_time_s


@dataclass(frozen=True)
class ToolCall:
    call_index: int
    timestamp_s: float
    tool_name: str
    tool_args: dict[str, Any]
    line_number: int
    timing_correction: dict[str, Any] | None = None

    @property
    def invocation(self) -> Invocation:
        return Invocation(self.tool_name, self.tool_args)


@dataclass(frozen=True)
class OtherEvent:
    event_type: str
    timestamp_s: float
    payload: dict[str, Any]
    line_number: int


TraceEvent = Union[LLMCall, ToolCall, OtherEvent]


@dataclass(frozen=True)
class SessionTrace:
    path: Path
    events: tuple[TraceEvent, ...]

    @property
    def session_id(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class SearchResult:
    url: str
    result_rank: int
    ordinal: int
    query_index: int
    # These fields are already visible in the search response.  Defaults keep
    # older artifacts and callers that only recorded URL/rank fully compatible.
    title: str = ""
    query: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class SearchVisitTransition:
    """A direct search -> visit transition observed in one session.

    Agent-level batched visits are exposed as atomic URL invocations by
    :meth:`authoritative_invocations`.  This is the concrete, side-effect-free
    operation the tool-only replay speculates on.
    """

    session_id: str
    search: ToolCall
    decision_llm: LLMCall
    visit: ToolCall
    completion_llm: Optional[LLMCall]
    search_results: tuple[SearchResult, ...]
    authoritative_urls: tuple[str, ...]
    baseline_stall_s: float
    overlap_window_s: float

    @property
    def authoritative_invocations(self) -> tuple[Invocation, ...]:
        return tuple(Invocation("visit", {"url": url}) for url in self.authoritative_urls)


def load_trace(path: Union[str, Path]) -> SessionTrace:
    """Parse one JSONL file without depending on repository runtime code."""

    trace_path = Path(path)
    events: list[TraceEvent] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceFormatError(
                    f"{trace_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise TraceFormatError(f"{trace_path}:{line_number}: event must be an object")
            event_type = payload.get("event_type")
            if not isinstance(event_type, str) or not event_type:
                raise TraceFormatError(
                    f"{trace_path}:{line_number}: missing string event_type"
                )
            timestamp_s = _number(
                payload.get("timestamp", 0.0),
                field="timestamp",
                path=trace_path,
                line_number=line_number,
            )
            call_index_raw = payload.get("call_index", 0)
            if isinstance(call_index_raw, bool) or not isinstance(call_index_raw, int):
                raise TraceFormatError(
                    f"{trace_path}:{line_number}: call_index must be an integer"
                )

            if event_type == "llm_call":
                messages_raw = payload.get("messages", [])
                if not isinstance(messages_raw, list) or not all(
                    isinstance(message, Mapping) for message in messages_raw
                ):
                    raise TraceFormatError(
                        f"{trace_path}:{line_number}: messages must be a list of objects"
                    )
                total_time_s = _number(
                    payload.get("total_time_ms", 0.0),
                    field="total_time_ms",
                    path=trace_path,
                    line_number=line_number,
                ) / 1000.0
                inference_time_s = _number(
                    payload.get("inference_time_ms", 0.0),
                    field="inference_time_ms",
                    path=trace_path,
                    line_number=line_number,
                ) / 1000.0
                events.append(
                    LLMCall(
                        call_index=call_index_raw,
                        timestamp_s=timestamp_s,
                        total_time_s=max(0.0, total_time_s),
                        inference_time_s=max(0.0, inference_time_s),
                        messages=tuple(dict(message) for message in messages_raw),
                        response=str(payload.get("response", "")),
                        line_number=line_number,
                    )
                )
            elif event_type == "tool_call":
                tool_name = payload.get("tool_name")
                tool_args = payload.get("tool_args", {})
                if not isinstance(tool_name, str) or not tool_name:
                    raise TraceFormatError(
                        f"{trace_path}:{line_number}: tool_name must be a non-empty string"
                    )
                if not isinstance(tool_args, dict):
                    raise TraceFormatError(
                        f"{trace_path}:{line_number}: tool_args must be an object"
                    )
                events.append(
                    ToolCall(
                        call_index=call_index_raw,
                        timestamp_s=timestamp_s,
                        tool_name=tool_name,
                        tool_args=dict(tool_args),
                        line_number=line_number,
                        timing_correction=(
                            dict(payload["timing_correction"])
                            if isinstance(payload.get("timing_correction"), Mapping)
                            else None
                        ),
                    )
                )
            else:
                events.append(
                    OtherEvent(
                        event_type=event_type,
                        timestamp_s=timestamp_s,
                        payload=dict(payload),
                        line_number=line_number,
                    )
                )
    return SessionTrace(trace_path, tuple(events))


def load_sessions(directory: Union[str, Path]) -> tuple[SessionTrace, ...]:
    """Load every ``*.jsonl`` session in lexical filename order."""

    trace_dir = Path(directory)
    paths = sorted(trace_dir.glob("*.jsonl"), key=lambda item: item.name)
    if not paths:
        raise FileNotFoundError(f"no JSONL traces found in {trace_dir}")
    return tuple(load_trace(path) for path in paths)


def split_sessions(
    sessions: Sequence[SessionTrace],
    train_ratio: float = 0.70,
    seed: str = "paste-repro-v1",
) -> tuple[tuple[SessionTrace, ...], tuple[SessionTrace, ...]]:
    """Deterministically split whole session files, preventing event leakage."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between zero and one")
    ordered = sorted(
        sessions,
        key=lambda session: (
            hashlib.sha256(f"{seed}\0{session.session_id}".encode("utf-8")).hexdigest(),
            session.session_id,
        ),
    )
    train_count = int(len(ordered) * train_ratio)
    if len(ordered) > 1:
        train_count = min(len(ordered) - 1, max(1, train_count))
    return tuple(ordered[:train_count]), tuple(ordered[train_count:])


def latest_tool_response(call: LLMCall) -> str:
    """Extract the newest tool response embedded in an LLM request."""

    for message in reversed(call.messages):
        content = message.get("content", "")
        if message.get("role") == "user" and isinstance(content, str):
            if "<tool_response>" in content:
                return content
    return ""


_NUMBERED_LINK = re.compile(
    r"^\s*(?P<rank>\d+)\.\s+\[(?P<title>.*)\]\((?P<url>https?://.+)\)\s*$"
)
_SEARCH_QUERY_HEADER = re.compile(
    r"^\s*A\s+.+?\s+search\s+for\s+(?P<query>.+?)\s+found\s+\d+\s+results?:\s*$",
    flags=re.IGNORECASE,
)
_PLAIN_URL = re.compile(r"https?://[^\s<>\[\]\"'`]+")


def _unquote_search_query(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
        return stripped[1:-1]
    return stripped


def parse_search_results(
    tool_response: str,
    *,
    queries: Sequence[str] | None = None,
) -> tuple[SearchResult, ...]:
    """Parse ranked Markdown links from a search tool response."""

    results: list[SearchResult] = []
    query_index = 0
    previous_rank = 0
    separator_seen = False
    parsed_queries: dict[int, str] = {}
    for line in tool_response.splitlines():
        if line.strip().startswith("======="):
            query_index += 1
            previous_rank = 0
            separator_seen = True
            continue
        header_match = _SEARCH_QUERY_HEADER.match(line)
        if header_match is not None:
            parsed_queries[query_index] = _unquote_search_query(
                header_match.group("query")
            )
            continue
        match = _NUMBERED_LINK.match(line)
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank <= previous_rank and not separator_seen:
            query_index += 1
        url = match.group("url").strip().rstrip(".,;:!?")
        query = (
            queries[query_index]
            if queries is not None and query_index < len(queries)
            else parsed_queries.get(query_index, "")
        )
        results.append(
            SearchResult(
                url=url,
                result_rank=rank,
                ordinal=len(results),
                query_index=query_index,
                title=match.group("title").strip(),
                query=query,
            )
        )
        previous_rank = rank
        separator_seen = False

    if results:
        return tuple(results)

    # A small fallback keeps the parser useful for non-Markdown search tools.
    seen: set[str] = set()
    for raw_url in _PLAIN_URL.findall(tool_response):
        url = raw_url.rstrip(").,;:!?")
        if url in seen:
            continue
        seen.add(url)
        results.append(
            SearchResult(
                url=url,
                result_rank=len(results) + 1,
                ordinal=len(results),
                query_index=0,
                query=(queries[0] if queries else ""),
            )
        )
    return tuple(results)


def _visit_urls(call: ToolCall) -> tuple[str, ...]:
    value = call.tool_args.get("url")
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def extract_search_visit_transitions(
    session: SessionTrace,
) -> tuple[SearchVisitTransition, ...]:
    """Return direct search -> visit transitions from one session trace."""

    events = session.events
    transitions: list[SearchVisitTransition] = []
    for index in range(len(events) - 2):
        search = events[index]
        decision = events[index + 1]
        visit = events[index + 2]
        if not (
            isinstance(search, ToolCall)
            and search.tool_name == "search"
            and isinstance(decision, LLMCall)
            and isinstance(visit, ToolCall)
            and visit.tool_name == "visit"
        ):
            continue
        urls = _visit_urls(visit)
        if not urls:
            continue
        completion = next(
            (event for event in events[index + 3 :] if isinstance(event, LLMCall)),
            None,
        )
        if completion is None:
            baseline_stall_s = 0.0
        else:
            # The next request starts after the tool result becomes available.
            baseline_stall_s = max(
                0.0, completion.start_timestamp_s - visit.timestamp_s
            )
        raw_queries = search.tool_args.get("query")
        search_queries = (
            tuple(item for item in raw_queries if isinstance(item, str))
            if isinstance(raw_queries, list)
            else ((raw_queries,) if isinstance(raw_queries, str) else ())
        )
        transitions.append(
            SearchVisitTransition(
                session_id=session.session_id,
                search=search,
                decision_llm=decision,
                visit=visit,
                completion_llm=completion,
                search_results=parse_search_results(
                    latest_tool_response(decision), queries=search_queries
                ),
                authoritative_urls=urls,
                baseline_stall_s=baseline_stall_s,
                overlap_window_s=max(0.0, decision.overlap_window_s),
            )
        )
    return tuple(transitions)


def transitions_from_sessions(
    sessions: Iterable[SessionTrace],
) -> tuple[SearchVisitTransition, ...]:
    return tuple(
        transition
        for session in sessions
        for transition in extract_search_visit_transitions(session)
    )


def count_tool_calls(sessions: Iterable[SessionTrace], tool_name: str) -> int:
    return sum(
        isinstance(event, ToolCall) and event.tool_name == tool_name
        for session in sessions
        for event in session.events
    )
