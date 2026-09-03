"""Deterministic whole-session collection for Tongyi DeepResearch.

The collector deliberately stays outside the predictor implementation.  It
records the legacy ``llm_call``/``tool_call`` JSONL events consumed by
``paste_repro.traces`` while checkpointing every observed event atomically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from .invocation import Invocation


WORKLOAD_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_TYPE = "paste_repro.multiturn_qwen_collection_manifest"
TRACE_SCHEMA = "paste_repro.legacy_event_jsonl_v1"
PROMPT_POLICY_VERSION = "tongyi_deepresearch_wikipedia_multiturn_v1"
OUTPUT_CLAIM_FILE = ".collection.claim"
MAX_WORKLOAD_SOURCES = 9_999
MAX_SEARCH_QUERIES_PER_CALL = 10

SYSTEM_PROMPT = """You are a deep research assistant. Research the user's
question with the provided web tools before answering. Search results contain
only titles and URLs; use visit to read page content. You may search again when
the current results are insufficient. Only visit exact URLs that appeared in a
previous search result in this session. Do not repeat a visit unless it is
needed to answer the question. When the evidence is sufficient, give the final
answer inside <answer></answer> tags.

# Tools
<tools>
{"type":"function","function":{"name":"search","description":"Search Wikipedia and return ranked titles and URLs.","parameters":{"type":"object","properties":{"query":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":10}},"required":["query"]}}}
{"type":"function","function":{"name":"visit","description":"Read one to six exact URLs returned by search.","parameters":{"type":"object","properties":{"url":{"oneOf":[{"type":"string"},{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":6}]},"goal":{"type":"string"}},"required":["url","goal"]}}}
</tools>

For one tool call, emit exactly one JSON object in these tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
Do not emit an answer and a tool call in the same response.
"""

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")
_TOOL_CALL_TAG = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CONTROL_TAG = re.compile(
    r"</?(?:tool_response|tool_call|answer)>", flags=re.IGNORECASE
)


def canonical_json(value: Any) -> str:
    """Return the checksum and JSONL canonical representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _safe_id(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if _SAFE_ID.fullmatch(result) is None:
        raise ValueError(
            f"{label} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,126}}"
        )
    return result


@dataclass(frozen=True)
class WorkloadSource:
    source_id: str
    question: str
    provenance: dict[str, Any]

    @property
    def source_sha256(self) -> str:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "question": self.question,
        }
        if self.provenance:
            payload["provenance"] = self.provenance
        return sha256_json(payload)

    @property
    def question_sha256(self) -> str:
        return hashlib.sha256(self.question.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FixedWorkload:
    workload_id: str
    sources: tuple[WorkloadSource, ...]
    file_sha256: str
    file_name: str


def load_fixed_workload(path: str | Path) -> FixedWorkload:
    """Load the strict, ordered workload format used for a fresh holdout."""

    workload_path = Path(path)
    if not workload_path.is_file():
        raise FileNotFoundError(f"workload does not exist: {workload_path}")
    try:
        payload = json.loads(workload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"workload is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("workload root must be an object")
    expected_root_keys = {"schema_version", "workload_id", "sources"}
    if set(payload) != expected_root_keys:
        raise ValueError(
            "workload root must contain exactly schema_version, workload_id, sources"
        )
    if payload["schema_version"] != WORKLOAD_SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise ValueError(f"workload schema_version must be {WORKLOAD_SCHEMA_VERSION}")
    workload_id = _safe_id(payload["workload_id"], "workload_id")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("workload sources must be a non-empty array")
    if len(raw_sources) > MAX_WORKLOAD_SOURCES:
        raise ValueError(f"workload may contain at most {MAX_WORKLOAD_SOURCES} sources")

    sources: list[WorkloadSource] = []
    seen_ids: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        label = f"sources[{index}]"
        if not isinstance(raw_source, dict):
            raise ValueError(f"{label} must be an object")
        required = {"source_id", "question"}
        if not required.issubset(raw_source) or not set(raw_source).issubset(
            required | {"provenance"}
        ):
            raise ValueError(
                f"{label} must contain source_id, question, and optional provenance"
            )
        source_id = _safe_id(raw_source["source_id"], f"{label}.source_id")
        if source_id in seen_ids:
            raise ValueError(f"duplicate source_id: {source_id}")
        seen_ids.add(source_id)
        question = _nonempty_string(raw_source["question"], f"{label}.question")
        raw_provenance = raw_source.get("provenance", {})
        if not isinstance(raw_provenance, dict):
            raise ValueError(f"{label}.provenance must be an object")
        try:
            provenance = _json_copy(raw_provenance)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.provenance must be finite JSON: {exc}") from exc
        sources.append(WorkloadSource(source_id, question, provenance))
    return FixedWorkload(
        workload_id=workload_id,
        sources=tuple(sources),
        file_sha256=sha256_file(workload_path),
        file_name=workload_path.name,
    )


def normalize_chat_completion_url(endpoint: str) -> str:
    """Resolve a server/base/v1 endpoint to the OpenAI chat-completions URL."""

    value = _nonempty_string(endpoint, "endpoint").rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "endpoint must be an uncredentialed HTTP(S) URL without query or fragment"
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        final_path = path
    elif path.endswith("/v1"):
        final_path = path + "/chat/completions"
    else:
        final_path = path + "/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, final_path, "", ""))


@dataclass(frozen=True)
class CollectorConfig:
    endpoint: str
    model: str
    max_calls: int
    max_output_tokens: int = 8_192
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    request_timeout_s: float = 300.0
    search_mode: str = "rest"
    visit_mode: str = "direct"
    tool_timeout_s: float = 20.0
    max_search_results: int = 5
    max_visit_urls: int = 6
    max_http_attempts: int = 1
    retry_backoff_s: float = 1.0
    search_min_start_interval_s: float = 0.0
    visit_min_start_interval_s: float = 0.0

    def __post_init__(self) -> None:
        normalize_chat_completion_url(self.endpoint)
        _nonempty_string(self.model, "model")
        for value, label in (
            (self.max_calls, "max_calls"),
            (self.max_output_tokens, "max_output_tokens"),
            (self.max_search_results, "max_search_results"),
            (self.max_visit_urls, "max_visit_urls"),
            (self.max_http_attempts, "max_http_attempts"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.max_visit_urls > 6:
            raise ValueError("max_visit_urls may not exceed the prompt's limit of 6")
        for value, label in (
            (self.temperature, "temperature"),
            (self.top_p, "top_p"),
            (self.request_timeout_s, "request_timeout_s"),
            (self.tool_timeout_s, "tool_timeout_s"),
            (self.retry_backoff_s, "retry_backoff_s"),
            (self.search_min_start_interval_s, "search_min_start_interval_s"),
            (self.visit_min_start_interval_s, "visit_min_start_interval_s"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.request_timeout_s <= 0 or self.tool_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if (
            self.retry_backoff_s < 0
            or self.search_min_start_interval_s < 0
            or self.visit_min_start_interval_s < 0
        ):
            raise ValueError("retry backoff and minimum start intervals must be non-negative")
        if self.search_mode not in {"rest", "action", "bing"}:
            raise ValueError("search_mode must be rest, action, or bing")
        if self.visit_mode not in {"direct", "jina"}:
            raise ValueError("visit_mode must be direct or jina")

    def to_manifest(self) -> dict[str, Any]:
        attempt_intervals = {
            tool_name: interval_s
            for tool_name, interval_s in (
                ("search", self.search_min_start_interval_s),
                ("visit", self.visit_min_start_interval_s),
            )
            if interval_s > 0
        }
        return {
            "endpoint": self.endpoint.rstrip("/"),
            "chat_completion_url": normalize_chat_completion_url(self.endpoint),
            "model": self.model,
            "max_calls": self.max_calls,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "request_timeout_s": self.request_timeout_s,
            "tool": {
                "executor": "WikipediaLiveExecutor",
                "search_mode": self.search_mode,
                "visit_mode": self.visit_mode,
                "timeout_s": self.tool_timeout_s,
                "max_search_results": self.max_search_results,
                "max_visit_urls": self.max_visit_urls,
                "max_http_attempts": self.max_http_attempts,
                "retry_backoff_s": self.retry_backoff_s,
                "search_min_start_interval_s": self.search_min_start_interval_s,
                "visit_min_start_interval_s": self.visit_min_start_interval_s,
                "http_attempt_min_start_intervals_s": attempt_intervals,
            },
        }


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    duration_s: float
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class ChatClient(Protocol):
    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> ChatCompletion: ...


class ToolExecutor(Protocol):
    async def __call__(self, invocation: Invocation) -> Mapping[str, Any]: ...


class ChatCompletionError(RuntimeError):
    """A sanitized OpenAI-compatible API failure."""


class OpenAICompatibleChatClient:
    """Small, no-retry client for an OpenAI-compatible vLLM server."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float,
        api_key: str | None = None,
        session: Any | None = None,
    ) -> None:
        self.chat_completion_url = normalize_chat_completion_url(endpoint)
        self._timeout_s = float(timeout_s)
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._closed = False

    async def _ensure_session(self) -> Any:
        if self._closed:
            raise RuntimeError("chat client is closed")
        if self._session is None:
            try:
                import aiohttp
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("OpenAICompatibleChatClient requires aiohttp") from exc
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_s), headers=headers
            )
        return self._session

    async def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> ChatCompletion:
        session = await self._ensure_session()
        body = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False,
            "stop": ["<tool_response>"],
        }
        started = time.monotonic()
        try:
            async with session.post(self.chat_completion_url, json=body) as response:
                raw = await response.read()
                status = int(response.status)
        except Exception as exc:
            raise ChatCompletionError(
                f"request failed with {type(exc).__name__}"
            ) from exc
        duration_s = max(0.0, time.monotonic() - started)
        body_sha = hashlib.sha256(raw).hexdigest()
        if status < 200 or status >= 300:
            raise ChatCompletionError(
                f"HTTP {status} from chat completions; response_sha256={body_sha}"
            )
        try:
            payload = json.loads(raw)
            choice = payload["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ChatCompletionError(
                f"invalid chat-completions response; response_sha256={body_sha}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ChatCompletionError(
                f"empty chat-completions content; response_sha256={body_sha}"
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = None
        return ChatCompletion(content, duration_s, finish_reason, usage)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_session and self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> "OpenAICompatibleChatClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


@dataclass(frozen=True)
class ParsedDecision:
    kind: str
    invocation: Invocation | None = None
    answer: str | None = None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_object,
    )


def parse_model_decision(response: str, *, max_visit_urls: int) -> ParsedDecision:
    """Parse exactly one terminal answer or one validated tool invocation."""

    tool_matches = _TOOL_CALL_TAG.findall(response)
    answer_matches = _ANSWER_TAG.findall(response)
    if len(tool_matches) + len(answer_matches) != 1:
        raise ValueError("response must contain exactly one tool_call or answer block")
    if answer_matches:
        answer = answer_matches[0].strip()
        if not answer:
            raise ValueError("answer block must not be empty")
        return ParsedDecision("answer", answer=answer)

    try:
        payload = _strict_json_loads(tool_matches[0].strip())
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"tool_call is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ValueError("tool_call must contain exactly name and arguments")
    name = payload["name"]
    arguments = payload["arguments"]
    if name not in {"search", "visit"} or not isinstance(arguments, dict):
        raise ValueError("tool_call requires a supported name and object arguments")

    if name == "search":
        if set(arguments) != {"query"}:
            raise ValueError("search arguments must contain exactly query")
        raw_queries = arguments["query"]
        queries = [raw_queries] if isinstance(raw_queries, str) else raw_queries
        if (
            not isinstance(queries, list)
            or not queries
            or len(queries) > MAX_SEARCH_QUERIES_PER_CALL
            or any(not isinstance(query, str) or not query.strip() for query in queries)
        ):
            raise ValueError("search query must contain one to ten non-empty strings")
    else:
        if set(arguments) != {"url", "goal"}:
            raise ValueError("visit arguments must contain exactly url and goal")
        raw_urls = arguments["url"]
        urls = [raw_urls] if isinstance(raw_urls, str) else raw_urls
        if (
            not isinstance(urls, list)
            or not urls
            or len(urls) > max_visit_urls
            or any(not isinstance(url, str) or not url.strip() for url in urls)
            or len(set(urls)) != len(urls)
        ):
            raise ValueError("visit url must contain unique non-empty URL strings")
        if any(
            url != url.strip()
            or urlsplit(url).scheme not in {"http", "https"}
            or not urlsplit(url).netloc
            for url in urls
        ):
            raise ValueError("visit only accepts absolute HTTP(S) URLs")
        if not isinstance(arguments["goal"], str) or not arguments["goal"].strip():
            raise ValueError("visit goal must be a non-empty string")
    return ParsedDecision("tool", invocation=Invocation(name, arguments))


def _safe_tool_text(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return _CONTROL_TAG.sub("[control-tag-removed]", text)


def format_search_response(result: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    if result.get("tool") != "search":
        raise ValueError("search executor returned the wrong tool type")
    raw_queries = result.get("query")
    raw_results = result.get("results")
    if not isinstance(raw_queries, list) or not all(
        isinstance(query, str) for query in raw_queries
    ):
        raise ValueError("search executor returned invalid queries")
    if not isinstance(raw_results, list):
        raise ValueError("search executor returned invalid results")

    grouped: list[list[tuple[int, str, str]]] = [[] for _ in raw_queries]
    visible_urls: list[str] = []
    seen_urls: set[str] = set()
    for row_index, row in enumerate(raw_results):
        if not isinstance(row, Mapping):
            raise ValueError("search result row must be an object")
        query_index = row.get("query_index")
        rank = row.get("rank")
        title = row.get("title")
        url = row.get("url")
        if (
            isinstance(query_index, bool)
            or not isinstance(query_index, int)
            or not 0 <= query_index < len(grouped)
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or not isinstance(title, str)
            or not isinstance(url, str)
            or urlsplit(url).scheme not in {"http", "https"}
            or not urlsplit(url).netloc
        ):
            raise ValueError(f"search result row {row_index} is malformed")
        grouped[query_index].append((rank, _safe_tool_text(title) or "Untitled", url))
        if url not in seen_urls:
            seen_urls.add(url)
            visible_urls.append(url)

    batches: list[str] = []
    for query_index, query in enumerate(raw_queries):
        rows = grouped[query_index]
        lines = [
            f"A Wikipedia search for {json.dumps(query, ensure_ascii=False)} found {len(rows)} results:",
            "",
            "## Web Results",
        ]
        lines.extend(f"{rank}. [{title}]({url})" for rank, title, url in rows)
        batches.append("\n".join(lines))
    return "\n=======\n".join(batches), tuple(visible_urls)


def format_visit_response(result: Mapping[str, Any]) -> str:
    if result.get("tool") != "visit":
        raise ValueError("visit executor returned the wrong tool type")
    raw_pages = result.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("visit executor returned no pages")
    pages: list[str] = []
    for index, page in enumerate(raw_pages):
        if not isinstance(page, Mapping):
            raise ValueError(f"visit page {index} must be an object")
        url = page.get("url")
        content = page.get("content")
        title = page.get("title", "")
        if not isinstance(url, str) or not isinstance(content, str):
            raise ValueError(f"visit page {index} is malformed")
        safe_title = _safe_tool_text(title)
        safe_content = _CONTROL_TAG.sub("[control-tag-removed]", content)
        heading = f"Content from {url}:"
        if safe_title:
            heading += f"\n\n# {safe_title}"
        pages.append(f"{heading}\n\n{safe_content.strip()}")
    return "\n=======\n".join(pages)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    _atomic_write_bytes(path, payload.encode("utf-8"))


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_http_attempt_log(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for raw_entry in value:
        if not isinstance(raw_entry, Mapping):
            continue
        entry: dict[str, Any] = {}
        for key in ("request_index", "attempt", "status"):
            raw = raw_entry.get(key)
            if raw is None and key == "status":
                entry[key] = None
            elif isinstance(raw, int) and not isinstance(raw, bool):
                entry[key] = raw
        raw_error_type = raw_entry.get("error_type")
        entry["error_type"] = (
            raw_error_type[:300] if isinstance(raw_error_type, str) else None
        )
        raw_retried = raw_entry.get("retried")
        if isinstance(raw_retried, bool):
            entry["retried"] = raw_retried
        for key in (
            "started_monotonic_s",
            "start_gate_wait_s",
            "retry_backoff_s",
        ):
            raw = raw_entry.get(key)
            if (
                isinstance(raw, (int, float))
                and not isinstance(raw, bool)
                and math.isfinite(float(raw))
            ):
                entry[key] = float(raw)
        if (
            isinstance(entry.get("request_index"), int)
            and isinstance(entry.get("attempt"), int)
            and entry["request_index"] >= 0
            and entry["attempt"] >= 1
        ):
            result.append(entry)
    return result


def _sanitize_transport_plan(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for key in ("backend", "request_host"):
        raw = value.get(key)
        if isinstance(raw, str):
            result[key] = raw[:1_000]
    raw_attempts = value.get("http_attempts")
    if (
        isinstance(raw_attempts, int)
        and not isinstance(raw_attempts, bool)
        and raw_attempts >= 0
    ):
        result["http_attempts"] = raw_attempts
    return result or None


def _transport_plan(executor: ToolExecutor, invocation: Invocation) -> dict[str, Any] | None:
    planner = getattr(executor, "transport_plan", None)
    if not callable(planner):
        return None
    try:
        return _sanitize_transport_plan(planner(invocation))
    except Exception:
        # Planning is diagnostic-only and must not create or suppress a real
        # dispatch.  A transport exception still carries its actual ledger.
        return None


def _exception_record(
    exc: Exception, *, transport_plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    message = str(exc).replace("\r", " ").replace("\n", " ").strip()
    result: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error_message": message[:1_000],
    }
    attempt_log = _sanitize_http_attempt_log(
        getattr(exc, "paste_http_attempt_log", None)
    )
    if attempt_log:
        result["http_attempt_log"] = attempt_log
    raw_failed_indexes = getattr(exc, "paste_http_batch_failure_indexes", None)
    if isinstance(raw_failed_indexes, (list, tuple)):
        failed_indexes = [
            value
            for value in raw_failed_indexes
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ]
        if failed_indexes:
            result["http_batch_failure_indexes"] = failed_indexes
    sanitized_plan = _sanitize_transport_plan(transport_plan)
    if sanitized_plan is not None:
        result["transport_plan"] = sanitized_plan
    return result


def _source_binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"source binding does not exist: {path}")
    return {"file_name": path.name, "sha256": sha256_file(path)}


def _executor_runtime_manifest(executor: ToolExecutor) -> dict[str, Any]:
    retryable_statuses = getattr(executor, "RETRYABLE_HTTP_STATUSES", ())
    retryable_exceptions = getattr(executor, "RETRYABLE_HTTP_EXCEPTION_TYPES", ())
    return {
        "http_retry_policy_version": getattr(
            executor, "HTTP_RETRY_POLICY_VERSION", None
        ),
        "http_library_retry_control_version": getattr(
            executor, "HTTP_LIBRARY_RETRY_CONTROL_VERSION", None
        ),
        "http_attempt_start_gate_version": getattr(
            executor, "HTTP_ATTEMPT_START_GATE_VERSION", None
        ),
        "retryable_http_statuses": [
            value
            for value in retryable_statuses
            if isinstance(value, int) and not isinstance(value, bool)
        ],
        "retryable_http_exception_types": [
            value for value in retryable_exceptions if isinstance(value, str)
        ],
        "http_library": {
            "retry_control_checked": bool(
                getattr(executor, "http_library_retry_control_checked", False)
            ),
            "retry_disabled_effective": bool(
                getattr(executor, "http_library_retry_disabled_effective", False)
            ),
            "name": getattr(executor, "http_library_name", None),
            "version": getattr(executor, "http_library_version", None),
        },
    }


def _claim_output_directory(target: Path, workload: FixedWorkload) -> Path:
    """Exclusively claim an absent or empty output directory for one run."""

    target.mkdir(parents=True, exist_ok=True)
    claim_path = target / OUTPUT_CLAIM_FILE
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"output directory is already claimed: {target}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                canonical_json(
                    {
                        "workload_id": workload.workload_id,
                        "workload_sha256": workload.file_sha256,
                        "claimed_at_utc": _utc_now(),
                        "pid": os.getpid(),
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        claim_path.unlink(missing_ok=True)
        raise
    existing = [path for path in target.iterdir() if path != claim_path]
    if existing:
        claim_path.unlink(missing_ok=True)
        raise FileExistsError(f"output directory must be empty: {target}")
    return claim_path


def _message_snapshot(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]


async def collect_one_session(
    *,
    source: WorkloadSource,
    workload_id: str,
    session_id: str,
    trace_path: Path,
    config: CollectorConfig,
    client: ChatClient,
    executor: ToolExecutor,
) -> dict[str, Any]:
    """Collect one complete session, retaining all observations on failure."""

    started_wall = _utc_now()
    started = time.monotonic()
    events: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source.question},
    ]
    visible_urls: set[str] = set()
    observations: list[dict[str, Any]] = []
    llm_calls = 0
    tool_calls = 0
    final_answer_sha256: str | None = None
    failure: dict[str, Any] | None = None
    committed_tool_results = 0

    def timestamp() -> float:
        return max(0.0, time.monotonic() - started)

    def checkpoint() -> None:
        write_jsonl_atomic(trace_path, events)

    events.append(
        {
            "event_type": "session_start",
            "call_index": 0,
            "timestamp": 0.0,
            "session_id": session_id,
            "workload_id": workload_id,
            "source_id": source.source_id,
            "source_sha256": source.source_sha256,
            "question_sha256": source.question_sha256,
            "provenance": source.provenance,
            "model": config.model,
        }
    )
    checkpoint()

    for call_index in range(config.max_calls):
        request_messages = _message_snapshot(messages)
        try:
            completion = await client.complete(
                messages=request_messages,
                model=config.model,
                max_tokens=config.max_output_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
            )
        except Exception as exc:
            failure = _exception_record(exc)
            break

        llm_calls += 1
        duration_ms = max(0.0, completion.duration_s) * 1000.0
        llm_event: dict[str, Any] = {
            "event_type": "llm_call",
            "call_index": call_index,
            "timestamp": timestamp(),
            "messages": request_messages,
            "response": completion.content,
            "total_time_ms": duration_ms,
            "rtt_ms": 0.0,
            "inference_time_ms": duration_ms,
        }
        if completion.finish_reason is not None:
            llm_event["finish_reason"] = completion.finish_reason
        if completion.usage is not None:
            llm_event["usage"] = completion.usage
        events.append(llm_event)
        checkpoint()
        messages.append({"role": "assistant", "content": completion.content})

        try:
            decision = parse_model_decision(
                completion.content, max_visit_urls=config.max_visit_urls
            )
        except Exception as exc:
            failure = _exception_record(exc)
            break
        if decision.kind == "answer":
            assert decision.answer is not None
            final_answer_sha256 = hashlib.sha256(
                decision.answer.encode("utf-8")
            ).hexdigest()
            break

        invocation = decision.invocation
        assert invocation is not None
        tool_calls += 1
        events.append(
            {
                "event_type": "tool_call",
                "call_index": call_index,
                "timestamp": timestamp(),
                "tool_name": invocation.tool_name,
                "tool_args": invocation.arguments,
            }
        )
        checkpoint()

        planned_transport = _transport_plan(executor, invocation)

        if invocation.tool_name == "visit":
            raw_urls = invocation.arguments["url"]
            urls = [raw_urls] if isinstance(raw_urls, str) else raw_urls
            unseen = [url for url in urls if url not in visible_urls]
            if unseen:
                failure = {
                    "error_type": "UnseenVisitUrlError",
                    "error_message": (
                        "visit URL was not present in a prior search result: "
                        + ", ".join(unseen)
                    )[:1_000],
                    "tool_result_committed": False,
                    "tool_failure_phase": "pre_dispatch_validation",
                    "failed_tool_call_index": call_index,
                    "failed_tool_name": invocation.tool_name,
                }
                break

        try:
            raw_result = await executor(invocation)
            if not isinstance(raw_result, Mapping):
                raise ValueError("tool executor result must be an object")
        except Exception as exc:
            failure = _exception_record(exc, transport_plan=planned_transport)
            failure.update(
                {
                    "tool_result_committed": False,
                    "tool_failure_phase": "dispatch",
                    "failed_tool_call_index": call_index,
                    "failed_tool_name": invocation.tool_name,
                }
            )
            break

        try:
            raw_result_copy = _json_copy(raw_result)
            result_hash = sha256_json(raw_result_copy)
        except Exception as exc:
            failure = _exception_record(exc, transport_plan=planned_transport)
            failure.update(
                {
                    "tool_result_committed": False,
                    "tool_failure_phase": "invalid_executor_result",
                    "failed_tool_call_index": call_index,
                    "failed_tool_name": invocation.tool_name,
                }
            )
            break
        transport = raw_result_copy.get("_paste_transport")
        transport_copy = _json_copy(transport) if isinstance(transport, Mapping) else None
        tool_result_event: dict[str, Any] = {
            "event_type": "tool_result",
            "call_index": call_index,
            "timestamp": timestamp(),
            "tool_name": invocation.tool_name,
            "commit_status": "committed",
            "result_sha256": result_hash,
            "raw_result": raw_result_copy,
            "formatted_response": None,
            "transport": transport_copy,
        }
        events.append(tool_result_event)
        committed_tool_results += 1
        observation = {
            "call_index": call_index,
            "tool_name": invocation.tool_name,
            "commit_status": "committed",
            "result_sha256": result_hash,
        }
        if transport_copy is not None:
            observation["transport"] = transport_copy
        observations.append(observation)
        # This checkpoint is the commit boundary.  raw_result is sufficient to
        # reproduce formatting even if the process stops before the next LLM.
        checkpoint()

        try:
            if invocation.tool_name == "search":
                formatted, discovered = format_search_response(raw_result_copy)
                visible_urls.update(discovered)
            else:
                formatted = format_visit_response(raw_result_copy)
        except Exception as exc:
            failure = _exception_record(exc, transport_plan=planned_transport)
            failure.update(
                {
                    "tool_result_committed": True,
                    "tool_failure_phase": "formatting",
                    "failed_tool_call_index": call_index,
                    "failed_tool_name": invocation.tool_name,
                }
            )
            break
        tool_result_event["formatted_response"] = formatted
        checkpoint()
        messages.append(
            {"role": "user", "content": f"<tool_response>\n{formatted}\n</tool_response>"}
        )
    else:
        failure = {
            "error_type": "MaxCallsExhausted",
            "error_message": f"no final answer after {config.max_calls} LLM calls",
        }

    status = "succeeded" if failure is None and final_answer_sha256 else "failed"
    if status == "failed" and failure is None:
        failure = {
            "error_type": "MaxCallsExhausted",
            "error_message": f"no final answer after {config.max_calls} LLM calls",
        }
    if failure is not None:
        events.append(
            {
                "event_type": "collector_error",
                "call_index": llm_calls,
                "timestamp": timestamp(),
                **failure,
            }
        )
        checkpoint()
    events.append(
        {
            "event_type": "session_end",
            "call_index": llm_calls,
            "timestamp": timestamp(),
            "status": status,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "committed_tool_results": committed_tool_results,
            "tool_observations": observations,
            "final_answer_sha256": final_answer_sha256,
            **(failure or {}),
        }
    )
    checkpoint()

    record: dict[str, Any] = {
        "session_id": session_id,
        "source_id": source.source_id,
        "source_sha256": source.source_sha256,
        "question_sha256": source.question_sha256,
        "provenance": source.provenance,
        "trace_file": trace_path.name,
        "trace_sha256": sha256_file(trace_path),
        "status": status,
        "started_at_utc": started_wall,
        "completed_at_utc": _utc_now(),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "committed_tool_results": committed_tool_results,
        "event_count": len(events),
        "final_answer_sha256": final_answer_sha256,
    }
    if failure is not None:
        record.update(failure)
    return record


async def collect_fixed_workload(
    *,
    workload_path: str | Path,
    output_dir: str | Path,
    config: CollectorConfig,
    client: ChatClient,
    executor: ToolExecutor,
    authentication_configured: bool = False,
    collector_cli_source_path: str | Path | None = None,
    live_executor_source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect an ordered workload and atomically maintain its manifest."""

    workload = load_fixed_workload(workload_path)
    target = Path(output_dir)
    _claim_output_directory(target, workload)
    cli_source = Path(collector_cli_source_path) if collector_cli_source_path else (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "collect_multiturn_qwen_traces.py"
    )
    live_source = Path(live_executor_source_path) if live_executor_source_path else (
        Path(__file__).with_name("live_executor.py")
    )
    manifest_path = target / "manifest.json"
    started_at = _utc_now()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": MANIFEST_TYPE,
        "collection_status": "in_progress",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "trace_schema": TRACE_SCHEMA,
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "collector_source_sha256": sha256_file(Path(__file__)),
        "source_bindings": {
            "collector": _source_binding(Path(__file__)),
            "collector_cli": _source_binding(cli_source),
            "live_executor": _source_binding(live_source),
        },
        "executor_runtime": _executor_runtime_manifest(executor),
        "authentication_configured": bool(authentication_configured),
        "workload": {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "workload_id": workload.workload_id,
            "file_name": workload.file_name,
            "file_sha256": workload.file_sha256,
            "source_count": len(workload.sources),
            "ordered_source_ids": [source.source_id for source in workload.sources],
        },
        "config": config.to_manifest(),
        "sessions": [],
    }
    write_json_atomic(manifest_path, manifest)

    records: list[dict[str, Any]] = []
    for index, source in enumerate(workload.sources, 1):
        session_id = f"{index:04d}-{source.source_id}"
        record = await collect_one_session(
            source=source,
            workload_id=workload.workload_id,
            session_id=session_id,
            trace_path=target / f"{session_id}.jsonl",
            config=config,
            client=client,
            executor=executor,
        )
        records.append(record)
        manifest["sessions"] = list(records)
        manifest["executor_runtime"] = _executor_runtime_manifest(executor)
        write_json_atomic(manifest_path, manifest)

    failure_count = sum(record["status"] != "succeeded" for record in records)
    manifest["collection_status"] = (
        "complete" if failure_count == 0 else "complete_with_failures"
    )
    manifest["completed_at_utc"] = _utc_now()
    manifest["executor_runtime"] = _executor_runtime_manifest(executor)
    manifest["summary"] = {
        "session_count": len(records),
        "succeeded": len(records) - failure_count,
        "failed": failure_count,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
