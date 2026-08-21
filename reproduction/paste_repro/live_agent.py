"""Live tool/LLM closed-loop experiment primitives.

This module deliberately keeps the control loop small and auditable:

1. a live LLM emits an exact search invocation;
2. a shared :class:`LiveToolBroker` executes that invocation;
3. returned URLs may be speculatively visited while the LLM selects one;
4. only an exact, session-scoped authoritative visit can expose a result;
5. the committed page is fed back to the LLM for the final answer.

There are no recorded tool sleeps in this path.  Tool timing is wall-clock time
spent in the broker's real executor and its finite shared queue.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import statistics
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import aiohttp

from .invocation import Invocation
from .live_broker import LiveAuthoritativeResult, LiveToolBroker


FINAL_ANSWER_MAX_CHARS = 480
FINAL_ANSWER_MAX_WORDS = 60
FINAL_ANSWER_TARGET_CHARS = 360
FINAL_COMPLETION_TOKEN_COUNT = 192
GUIDED_JSON_RECOVERY_POLICY_VERSION = "escape-unescaped-string-controls-v1"
FINAL_ANSWER_CONTRACT_POLICY_VERSION = (
    "guided-json-strict-local-whitespace-bounded-prefix-v2"
)
FIXED_FINAL_ANSWER_CONTRACT_POLICY_VERSION = (
    "guided-grammar-fixed-192-token-strict-tail-local-projection-v1"
)
FINAL_ANSWER_SCHEMA_POLICY_VERSION = "xgrammar-unbounded-answer-exact-url-v1"
FINAL_ANSWER_GRAMMAR_POLICY_VERSION = (
    "xgrammar-compact-unbounded-answer-exact-url-ascii-space-tail-v1"
)
FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION = "0.1.21"
OUTPUT_CONTRACT_POLICY_VERSION = (
    "guided-tool-and-final-json-strict-local-projection-v2"
)
FIXED_OUTPUT_CONTRACT_POLICY_VERSION = (
    "guided-tool-json-and-fixed-final-grammar-strict-local-projection-v1"
)
TOOL_SIGNAL_POLICY_VERSION = "exact-session-invocation-running-completed-v1"


SYSTEM_PROMPT = f"""You are a deterministic research agent in a live-tool benchmark.
Never answer from memory. Follow exactly one of the three phases below. Never
emit markdown or <think> text.

1. For a TASK message, emit only this JSON search call with the exact
   SEARCH_QUERY supplied by the
   user: {{"name":"search","arguments":{{"query":["..."]}}}}.
2. For a search TOOL_RESULT, emit only this JSON visit call for exactly one URL
   from that result. Prefer the first result unless it is plainly irrelevant.
   Copy RESEARCH_GOAL exactly:
   {{"name":"visit","arguments":{{"url":["..."],"goal":"..."}}}}.
3. For a visit TOOL_RESULT, emit only this JSON object. The answer must be one
   concise factual sentence grounded only in that page. Copy the exact URL from
   your preceding visit call into source_url. Do not emit markdown, citations,
   <think> text, or a line break:
   {{"answer":"...","source_url":"..."}}.
   Aim for at most {FINAL_ANSWER_TARGET_CHARS} answer characters.
"""


class TokenCounter(Protocol):
    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int: ...

    def count_text(self, text: str) -> int: ...


class ApproximateTokenCounter:
    """Explicit fallback used only when a tokenizer was not requested."""

    method = "utf8_chars_div4_ceiling"

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        text = "\n".join(
            f"{item.get('role', '')}:{item.get('content', '')}" for item in messages
        )
        return self.count_text(text) + 4

    def count_text(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))


class TransformersTokenCounter:
    method = "transformers_chat_template"

    def __init__(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        rendered = self._tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True
        )
        return max(1, len(rendered))

    def count_text(self, text: str) -> int:
        return max(1, len(self._tokenizer.encode(text, add_special_tokens=False)))


@dataclass(frozen=True)
class LiveSource:
    source_id: str
    question: str
    search_query: str
    expected_url: str | None = None


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    duration_s: float
    usage: dict[str, int]
    request_id: str
    finish_reason: str | None = None


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


def _escape_unescaped_json_string_controls(
    content: str,
) -> tuple[str, list[int]]:
    """Escape JSON-forbidden control characters only while inside strings.

    This is deliberately a lexical repair, not permissive JSON parsing.  It
    preserves the decoded string value, does not touch legal whitespace
    between JSON tokens, and leaves every other syntax error for the strict
    parser to reject.
    """

    repaired: list[str] = []
    control_codepoints: list[int] = []
    in_string = False
    escaped = False
    for character in content:
        codepoint = ord(character)
        if in_string:
            if escaped:
                repaired.append(character)
                escaped = False
            elif character == "\\":
                repaired.append(character)
                escaped = True
            elif character == '"':
                repaired.append(character)
                in_string = False
            elif codepoint < 0x20:
                # json.dumps produces the shortest valid JSON escape for the
                # common controls and a \u00XX escape for the remainder.
                repaired.append(json.dumps(character)[1:-1])
                control_codepoints.append(codepoint)
            else:
                repaired.append(character)
        else:
            repaired.append(character)
            if character == '"':
                in_string = True
    return "".join(repaired), control_codepoints


def parse_guided_object(
    content: str, *, telemetry: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if telemetry is not None:
        telemetry.update(
            {
                "policy_version": GUIDED_JSON_RECOVERY_POLICY_VERSION,
                "recovery_applied": False,
                "raw_sha256": raw_sha256,
            }
        )
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        # CPython reports this exact error only for a raw U+0000--U+001F
        # character inside a quoted JSON string.  Repair no other parse error.
        if exc.msg != "Invalid control character at":
            raise ValueError(f"LLM did not return one JSON object: {exc}") from exc
        repaired, codepoints = _escape_unescaped_json_string_controls(content)
        if not codepoints or repaired == content:
            raise ValueError(f"LLM did not return one JSON object: {exc}") from exc
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError as repaired_exc:
            raise ValueError(
                "LLM did not return one JSON object after narrowly escaping "
                f"string control characters: {repaired_exc}"
            ) from repaired_exc
        if telemetry is not None:
            telemetry.update(
                {
                    "recovery_applied": True,
                    "control_character_count": len(codepoints),
                    "control_character_codepoints": sorted(set(codepoints)),
                    "repaired_sha256": hashlib.sha256(
                        repaired.encode("utf-8")
                    ).hexdigest(),
                }
            )
    if not isinstance(value, dict):
        raise ValueError("LLM output must be a JSON object")
    return value


def _validate_absolute_https_url(value: str, *, source_index: int) -> str:
    normalized = value.strip()
    parsed = urlsplit(normalized)
    if (
        normalized != value
        or parsed.scheme.lower() != "https"
        or not parsed.netloc
        or parsed.hostname is None
    ):
        raise ValueError(
            f"source {source_index} expected_url must be an absolute HTTPS URL"
        )
    return normalized


def validate_sources(
    payload: Mapping[str, Any], *, call_graph_mode: str = "autonomous"
) -> list[LiveSource]:
    if call_graph_mode not in {"autonomous", "frozen"}:
        raise ValueError(f"unsupported call graph mode: {call_graph_mode}")
    rows = payload.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("live workload must contain a non-empty sources list")
    sources: list[LiveSource] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"source {index} is not an object")
        values = {}
        for key in ("source_id", "question", "search_query"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source {index} has invalid {key}")
            values[key] = value.strip()
        if values["source_id"] in seen:
            raise ValueError(f"duplicate source_id: {values['source_id']}")
        raw_expected_url = row.get("expected_url")
        if raw_expected_url is None:
            expected_url = None
        elif not isinstance(raw_expected_url, str):
            raise ValueError(f"source {index} has invalid expected_url")
        else:
            expected_url = _validate_absolute_https_url(
                raw_expected_url, source_index=index
            )
        if call_graph_mode == "frozen" and expected_url is None:
            raise ValueError(
                f"source {index} requires expected_url in frozen call graph mode"
            )
        seen.add(values["source_id"])
        sources.append(LiveSource(**values, expected_url=expected_url))
    return sources


def search_schema(query: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": "search"},
            "arguments": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"const": query},
                    }
                },
            },
        },
    }


def visit_schema(urls: Sequence[str], goal: str) -> dict[str, Any]:
    if not urls:
        raise ValueError("visit schema requires at least one live search URL")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": "visit"},
            "arguments": {
                "type": "object",
                "additionalProperties": False,
                "required": ["url", "goal"],
                "properties": {
                    "url": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"enum": list(urls)},
                    },
                    "goal": {"const": goal},
                },
            },
        },
    }


def final_answer_schema(url: str) -> dict[str, Any]:
    """Return the strict final-answer shape without fragile length keywords.

    xgrammar 0.1.21 has produced an invalid raw control character when a
    generated string reaches a JSON-schema ``maxLength`` boundary.  The FSM is
    therefore responsible only for the JSON shape and exact committed URL.
    Length and word bounds are projected deterministically after strict parsing.
    In particular, keep ``minLength``, ``maxLength``, and ``pattern`` out of this
    schema.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "source_url"],
        "properties": {
            "answer": {"type": "string"},
            "source_url": {"const": url},
        },
    }


@lru_cache(maxsize=512)
def final_answer_fixed_completion_grammar(url: str) -> str:
    """Return the v8 final grammar: compact semantic JSON plus ASCII spaces.

    The semantic prefix is converted from :func:`final_answer_schema`, so it
    retains the unbounded answer string and exact committed-URL constraint.
    Interior JSON whitespace is fixed to the compact representation.  Once the
    object closes, at least one literal U+0020 is required and no other trailing
    byte is admitted.  Pairing this grammar with ``min_tokens == max_tokens``
    keeps generation alive in the padding branch without changing the decoded
    answer semantics.
    """

    try:
        import xgrammar
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "fixed final completion grammar requires xgrammar"
        ) from exc

    semantic = xgrammar.Grammar.from_json_schema(
        final_answer_schema(url),
        any_whitespace=False,
        separators=(",", ":"),
        strict_mode=True,
    )
    ascii_space_tail = xgrammar.Grammar.from_ebnf('root ::= " "+')
    return str(xgrammar.Grammar.concat(semantic, ascii_space_tail))


def validate_final_answer(answer: Mapping[str, Any], *, url: str) -> None:
    """Strictly validate the locally projected final-answer object."""

    answer_text = answer.get("answer")
    if (
        set(answer) != {"answer", "source_url"}
        or answer.get("source_url") != url
        or not isinstance(answer_text, str)
    ):
        raise ValueError(
            "guided final answer violates the concise answer contract or cites "
            "the wrong URL"
        )
    words = answer_text.split(" ")
    if (
        not answer_text
        or answer_text != answer_text.strip()
        or any(
            character.isspace() and character != " " for character in answer_text
        )
        or "  " in answer_text
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in answer_text
        )
        or len(answer_text) > FINAL_ANSWER_MAX_CHARS
        or len(words) > FINAL_ANSWER_MAX_WORDS
        or "http://" in answer_text.lower()
        or "https://" in answer_text.lower()
    ):
        raise ValueError("guided final answer violates the concise answer contract")


def _bounded_answer_prefix(canonical: str) -> tuple[str, dict[str, bool]]:
    """Project a canonical answer onto the frozen word and character bounds."""

    words = canonical.split(" ") if canonical else []
    word_limited = " ".join(words[:FINAL_ANSWER_MAX_WORDS])
    word_projection_applied = len(words) > FINAL_ANSWER_MAX_WORDS
    char_projection_applied = len(word_limited) > FINAL_ANSWER_MAX_CHARS
    projected = word_limited
    if char_projection_applied:
        projected = word_limited[:FINAL_ANSWER_MAX_CHARS].rstrip()
        # Avoid a partial trailing word when a prior whole-word boundary is
        # available.  Languages without ASCII spaces still receive a bounded
        # Unicode-code-point prefix.
        if (
            projected
            and len(word_limited) > FINAL_ANSWER_MAX_CHARS
            and not word_limited[FINAL_ANSWER_MAX_CHARS].isspace()
            and " " in projected
        ):
            whole_words = projected.rsplit(" ", 1)[0]
            if whole_words:
                projected = whole_words
    return projected, {
        "word_projection_applied": word_projection_applied,
        "char_projection_applied": char_projection_applied,
    }


class _StrictJSONContractError(ValueError):
    """Raised for syntax accepted by CPython but excluded from strict JSON."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJSONContractError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_nonstandard_json_constant(value: str) -> Any:
    raise _StrictJSONContractError(f"non-standard JSON constant: {value}")


def parse_guided_final_answer(
    content: str,
    *,
    url: str,
    call_index: int = 2,
    telemetry: dict[str, Any] | None = None,
    token_counter: TokenCounter | None = None,
    completion_tokens: int | None = None,
    finish_reason: str | None = None,
    fixed_completion_tokens: int | None = None,
) -> dict[str, str]:
    """Strictly parse, canonicalize, and bound one guided final answer.

    No JSON repair or permissive recovery is attempted.  The model-emitted URL
    must equal the committed selected URL, after which the returned object's URL
    is bound from the trusted local argument.  Leading/trailing whitespace is
    removed, every Unicode-whitespace run becomes one ASCII space, and an
    overlong answer is projected onto a deterministic prefix.

    The opt-in fixed-completion contract uses ``raw_decode`` to separate one
    strict semantic JSON object from its padding.  It accepts only a non-empty
    suffix of literal ASCII spaces, exactly 192 server-reported completion
    tokens, and ``finish_reason == "length"``.  Any disagreement fails closed.
    """

    if not isinstance(content, str):
        raise TypeError("guided final completion must be text")
    fixed = fixed_completion_tokens is not None
    if fixed and (
        type(fixed_completion_tokens) is not int
        or fixed_completion_tokens != FINAL_COMPLETION_TOKEN_COUNT
    ):
        raise ValueError(
            "fixed final completion tokens must equal "
            f"{FINAL_COMPLETION_TOKEN_COUNT}"
        )
    schema = final_answer_schema(url)
    grammar = final_answer_fixed_completion_grammar(url) if fixed else None
    record: dict[str, Any] = {
        "call_index": int(call_index),
        "policy_version": (
            FIXED_FINAL_ANSWER_CONTRACT_POLICY_VERSION
            if fixed
            else FINAL_ANSWER_CONTRACT_POLICY_VERSION
        ),
        "schema_policy_version": FINAL_ANSWER_SCHEMA_POLICY_VERSION,
        "schema_sha256": sha256_json(schema),
        "schema_answer_constraint": "type_only_no_length_or_pattern",
        "mode": (
            "guided_grammar_fixed_completion_strict_raw_decode_local_projection"
            if fixed
            else "guided_json_strict_local_projection"
        ),
        "guided_json_requested": not fixed,
        "guided_grammar_requested": fixed,
        "json_parse_attempted": True,
        "strict_json_parse": True,
        "strict_json_raw_decode": fixed,
        "recovery_allowed": False,
        "recovery_applied": False,
        "parse_succeeded": False,
        "local_wrap_applied": True,
        "local_projection_applied": False,
        "object_constructed_locally": True,
        "source_url_binding": "exact_committed_selected_url",
        "source_url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        "contract_succeeded": False,
        "raw_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "raw_char_count": len(content),
        "max_chars": FINAL_ANSWER_MAX_CHARS,
        "max_words": FINAL_ANSWER_MAX_WORDS,
        "target_chars": FINAL_ANSWER_TARGET_CHARS,
    }
    if fixed:
        assert grammar is not None
        record.update(
            {
                "grammar_policy_version": FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
                "grammar_xgrammar_version": FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
                "grammar_sha256": hashlib.sha256(
                    grammar.encode("utf-8")
                ).hexdigest(),
                "grammar_semantic_json_whitespace": "compact",
                "tail_policy": "one_or_more_ascii_spaces_only",
                "tail_validation_succeeded": False,
                "fixed_completion_tokens": FINAL_COMPLETION_TOKEN_COUNT,
                "min_tokens": FINAL_COMPLETION_TOKEN_COUNT,
                "max_tokens": FINAL_COMPLETION_TOKEN_COUNT,
                "total_completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "finish_reason_validated": False,
                "token_accounting_succeeded": False,
            }
        )

    def sync_telemetry() -> None:
        if telemetry is not None:
            telemetry.update(record)

    sync_telemetry()
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_nonstandard_json_constant,
    )
    try:
        if fixed:
            value, semantic_end = decoder.raw_decode(content)
        else:
            value = decoder.decode(content)
            semantic_end = len(content)
    except (json.JSONDecodeError, _StrictJSONContractError) as exc:
        sync_telemetry()
        raise ValueError(
            f"guided final answer is not one strict JSON object: {exc}"
        ) from exc
    record["parse_succeeded"] = True
    semantic_wire = content[:semantic_end]
    if fixed:
        tail = content[semantic_end:]
        record.update(
            {
                "semantic_sha256": hashlib.sha256(
                    semantic_wire.encode("utf-8")
                ).hexdigest(),
                "semantic_char_count": len(semantic_wire),
                "semantic_byte_count": len(semantic_wire.encode("utf-8")),
                "padding_sha256": hashlib.sha256(tail.encode("utf-8")).hexdigest(),
                "padding_char_count": len(tail),
                "padding_byte_count": len(tail.encode("utf-8")),
                "tail_nonempty": bool(tail),
                "tail_ascii_space_only": bool(tail)
                and all(character == " " for character in tail),
            }
        )
        if not record["tail_ascii_space_only"]:
            sync_telemetry()
            raise ValueError(
                "guided fixed final answer tail must contain only one or more "
                "ASCII spaces"
            )
        record["tail_validation_succeeded"] = True
        if token_counter is None:
            sync_telemetry()
            raise ValueError("fixed final completion requires an exact token counter")
        token_counter_method = getattr(token_counter, "method", None)
        record["token_counter_method"] = token_counter_method
        if (
            not isinstance(token_counter_method, str)
            or not token_counter_method
            or token_counter_method == ApproximateTokenCounter.method
        ):
            sync_telemetry()
            raise ValueError(
                "fixed final completion requires the exact model tokenizer"
            )
        if (
            type(completion_tokens) is not int
            or completion_tokens != FINAL_COMPLETION_TOKEN_COUNT
        ):
            sync_telemetry()
            raise ValueError(
                "guided fixed final answer did not use exactly "
                f"{FINAL_COMPLETION_TOKEN_COUNT} completion tokens"
            )
        semantic_token_count = int(token_counter.count_text(semantic_wire))
        padding_token_count = completion_tokens - semantic_token_count
        record.update(
            {
                "semantic_token_count": semantic_token_count,
                "padding_token_count": padding_token_count,
                "token_partition_method": (
                    "server_total_minus_local_semantic_tokenization"
                ),
            }
        )
        if semantic_token_count <= 0 or padding_token_count <= 0:
            sync_telemetry()
            raise ValueError(
                "guided fixed final answer has invalid semantic/padding token "
                "accounting"
            )
        record["token_accounting_succeeded"] = True
        if finish_reason != "length":
            sync_telemetry()
            raise ValueError(
                "guided fixed final answer must finish because of length"
            )
        record["finish_reason_validated"] = True
    sync_telemetry()
    if (
        not isinstance(value, dict)
        or set(value) != {"answer", "source_url"}
        or not isinstance(value.get("answer"), str)
        or value.get("source_url") != url
    ):
        sync_telemetry()
        raise ValueError(
            "guided final answer violates the object shape or exact committed URL"
        )
    raw_answer = value["answer"]
    record.update(
        {
            "model_answer_sha256": hashlib.sha256(
                raw_answer.encode("utf-8")
            ).hexdigest(),
            "model_answer_char_count": len(raw_answer),
            "model_source_url_validated": True,
        }
    )
    if any(
        (ord(character) < 0x20 and character not in "\t\n\v\f\r")
        or ord(character) == 0x7F
        for character in raw_answer
    ):
        sync_telemetry()
        raise ValueError(
            "guided final answer contains a non-whitespace control character"
        )
    canonical = " ".join(raw_answer.split())
    projected, projection = _bounded_answer_prefix(canonical)
    record.update(
        {
            "pre_projection_canonical_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
            "pre_projection_char_count": len(canonical),
            "pre_projection_word_count": (
                len(canonical.split(" ")) if canonical else 0
            ),
            "canonical_sha256": hashlib.sha256(
                projected.encode("utf-8")
            ).hexdigest(),
            "canonicalization_changed": canonical != raw_answer,
            "canonical_char_count": len(projected),
            "canonical_word_count": (
                len(projected.split(" ")) if projected else 0
            ),
            **projection,
        }
    )
    record["local_projection_applied"] = any(projection.values())
    sync_telemetry()
    answer = {"answer": projected, "source_url": url}
    try:
        validate_final_answer(answer, url=url)
    except BaseException:
        sync_telemetry()
        raise
    record["contract_succeeded"] = True
    sync_telemetry()
    return answer


def scheduler_request_id(meta: Mapping[str, Any]) -> str:
    encoded = canonical_json(dict(meta)).encode("utf-8").hex()
    return f"schedx{encoded}z"


def _broker_wait_estimate_s(
    snapshot: Mapping[str, Any], *, tool_name: str, fallback_s: float
) -> float:
    jobs = snapshot.get("jobs", [])
    if isinstance(jobs, list):
        matching = [
            job
            for job in jobs
            if isinstance(job, Mapping) and job.get("tool_name") == tool_name
        ]
        # The highest-ranked prediction is created first.  Completed work has
        # no remaining wait; running work uses the broker's causal EWMA.
        if matching:
            job = matching[0]
            if job.get("state") == "completed":
                return 0.0
            remaining = job.get("estimated_remaining_s")
            if isinstance(remaining, (int, float)) and math.isfinite(remaining):
                return max(0.0, float(remaining))

    counts = snapshot.get("counts", {})
    capacity = snapshot.get("capacity", {})
    ewmas = snapshot.get("service_ewma_s", {})
    service_s = ewmas.get(tool_name, fallback_s) if isinstance(ewmas, Mapping) else fallback_s
    if not isinstance(service_s, (int, float)) or not math.isfinite(service_s):
        service_s = fallback_s
    queued = 0
    if isinstance(counts, Mapping):
        queued = int(counts.get("queued_authoritative", 0) or 0) + int(
            counts.get("queued_speculative", 0) or 0
        )
    workers = int(capacity.get("max_workers", 1) or 1) if isinstance(capacity, Mapping) else 1
    return max(0.0, float(service_s)) * (1.0 + queued / max(1, workers))


def _broker_tool_signal(
    snapshot: Mapping[str, Any],
    *,
    invocation: Invocation | None,
    nominal_confidence: float,
    policy: str,
    eligible: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Gate the direct overlap bonus on causal physical tool progress.

    Under ``execution_aware``, a queued speculative job still contributes its
    ETA through ``nw``/``rtw`` and the pending-return KV reserve, but it cannot
    earn the direct LLM scheduling bonus solely by sitting farther back in the
    shared tool queue.  ``legacy`` preserves the original caller-provided
    confidence exactly.
    """

    bounded = max(0.0, min(1.0, float(nominal_confidence)))
    if policy == "legacy":
        return bounded, {}
    if policy != "execution_aware":
        raise ValueError(f"unsupported tool signal policy: {policy}")
    evidence: dict[str, Any] = {
        "nps": "none",
        "nrg": 0,
        "ntc": round(bounded, 6),
        "npm": 0,
        "br": int(snapshot.get("revision", 0) or 0),
    }
    observed_at = snapshot.get("observed_at_monotonic_s")
    if isinstance(observed_at, (int, float)) and math.isfinite(observed_at):
        evidence["brt"] = round(float(observed_at), 6)
    if invocation is None:
        return 0.0, evidence
    if not eligible:
        evidence["nps"] = "ineligible"
        return 0.0, evidence

    jobs = snapshot.get("jobs", [])
    if not isinstance(jobs, list):
        evidence["nps"] = "invalid_snapshot"
        return 0.0, evidence
    expected_digest = hashlib.sha256(
        f"{invocation.tool_name}\0{invocation.canonical_arguments}".encode("utf-8")
    ).hexdigest()
    matching = [
        job
        for job in jobs
        if isinstance(job, Mapping)
        and job.get("tool_name") == invocation.tool_name
        and job.get("invocation_digest") == expected_digest
    ]
    if len(matching) != 1:
        evidence["nps"] = "missing" if not matching else "ambiguous"
        return 0.0, evidence

    job = matching[0]
    state = str(job.get("state", "unknown"))
    lane = str(job.get("lane", "unknown"))
    signal_state = f"{lane}_{state}"
    job_id = job.get("job_id")
    queue_position = job.get("queue_position")
    tool_queue_position = job.get("tool_queue_position")
    evidence.update(
        {
            "nps": signal_state,
            "npm": 1,
            "npjid": -1 if job_id is None else int(job_id),
            "npc": int(job.get("confirmed") is True),
            "npq": -1 if queue_position is None else int(queue_position),
            "nptq": (
                -1 if tool_queue_position is None else int(tool_queue_position)
            ),
        }
    )
    remaining = job.get("estimated_remaining_s")
    if isinstance(remaining, (int, float)) and math.isfinite(remaining):
        evidence["nper"] = round(max(0.0, float(remaining)), 6)
    if (
        lane == "speculative"
        and job.get("confirmed") is False
        and state in {"running", "completed"}
    ):
        evidence["nrg"] = 1
        return bounded, evidence
    return 0.0, evidence


def build_scheduler_meta(
    *,
    task_id: str,
    call_index: int,
    prompt_tokens: int,
    max_tokens: int,
    predicted_output_tokens: int,
    broker_global: Mapping[str, Any],
    broker_session: Mapping[str, Any],
    next_tool_name: str | None,
    next_tool_confidence: float,
    next_prompt_tokens: int | None,
    default_tool_service_s: float,
    next_tool_signal_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    remaining_calls = max(0, 2 - call_index)
    nw = 0.0
    if next_tool_name is not None:
        nw = _broker_wait_estimate_s(
            broker_session,
            tool_name=next_tool_name,
            fallback_s=default_tool_service_s,
        )
    counts = broker_global.get("counts", {})
    meta: dict[str, Any] = {
        "t": task_id,
        "c": call_index,
        "i": call_index,
        "n": 3,
        "rc": remaining_calls,
        "nw": round(nw, 6),
        "nwc": round(max(0.0, min(1.0, next_tool_confidence)), 6),
        "rtw": round(nw, 6),
        "pt": int(prompt_tokens),
        "mt": int(max_tokens),
        "po": int(max(1, predicted_output_tokens)),
        "npo": int(max(1, predicted_output_tokens)),
        "ms": "live_broker",
    }
    if next_tool_signal_evidence:
        meta.update(dict(next_tool_signal_evidence))
    if next_prompt_tokens is not None and remaining_calls > 0:
        meta["npt"] = int(max(1, next_prompt_tokens))
        meta["nmt"] = int(max_tokens)
    if isinstance(counts, Mapping):
        # Unknown fields are ignored by older hooks but remain causal evidence
        # that every LLM request observed the live shared tool queue.
        meta.update(
            {
                "tqa": int(counts.get("queued_authoritative", 0) or 0),
                "tqs": int(counts.get("queued_speculative", 0) or 0),
                "tra": int(counts.get("running_authoritative", 0) or 0),
                "trs": int(counts.get("running_speculative", 0) or 0),
            }
        )
    return meta


class LiveLLMClient:
    """Exactly-once HTTP client for vLLM chat completions."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        server_url: str,
        model: str,
        timeout_s: float,
        temperature: float = 0.0,
    ) -> None:
        self._session = session
        self._url = f"{server_url.rstrip('/')}/v1/chat/completions"
        self._model = model
        self._timeout_s = timeout_s
        self._temperature = temperature
        self.events: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        task_id: str,
        call_index: int,
        messages: Sequence[Mapping[str, str]],
        request_id: str,
        max_tokens: int,
        schema: Mapping[str, Any] | None,
        prompt_tokens: int,
        scheduler_meta: Mapping[str, Any],
        grammar: str | None = None,
        min_tokens: int = 0,
    ) -> LLMCompletion:
        if schema is not None and grammar is not None:
            raise ValueError("schema and grammar guided decoding are mutually exclusive")
        if min_tokens < 0 or min_tokens > max_tokens:
            raise ValueError("min_tokens must be between zero and max_tokens")
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": self._temperature,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "request_id": request_id,
        }
        if schema is not None:
            payload["guided_json"] = dict(schema)
        if grammar is not None:
            payload["guided_grammar"] = grammar
        if min_tokens:
            payload["min_tokens"] = int(min_tokens)
        started_wall = time.time()
        started = time.monotonic()
        event: dict[str, Any] = {
            "task_id": task_id,
            "call_index": call_index,
            "request_id": request_id,
            "request_start_s": started_wall,
            "request_start_monotonic_s": started,
            "prompt_tokens_estimate": prompt_tokens,
            "messages_sha256": sha256_json(list(messages)),
            "scheduler_meta": dict(scheduler_meta),
            "output_mode": (
                "guided_json"
                if schema is not None
                else "guided_grammar"
                if grammar is not None
                else "plain_text"
            ),
            "guided_json_requested": schema is not None,
            "guided_grammar_requested": grammar is not None,
            "guided_grammar_sha256": (
                hashlib.sha256(grammar.encode("utf-8")).hexdigest()
                if grammar is not None
                else None
            ),
            "min_tokens": int(min_tokens),
            "max_tokens": int(max_tokens),
            "attempts": 1,
        }
        try:
            async with self._session.post(
                self._url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout_s),
            ) as response:
                event["http_status"] = response.status
                body = await response.json(content_type=None)
                if response.status != 200:
                    event.update({"ok": False, "error_body": body})
                    raise RuntimeError(
                        f"vLLM returned HTTP {response.status} for {task_id}/{call_index}"
                    )
        except BaseException as exc:
            event.update(
                {
                    "ok": False,
                    "duration_s": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )
            self.events.append(event)
            raise

        duration_s = time.monotonic() - started
        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason")
        raw_usage = body.get("usage", {})
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(raw_usage.get("total_tokens", 0) or 0),
        }
        event.update(
            {
                "ok": True,
                "duration_s": duration_s,
                "response": content,
                "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "usage": usage,
                "finish_reason": finish_reason,
            }
        )
        self.events.append(event)
        return LLMCompletion(
            content=content,
            duration_s=duration_s,
            usage=usage,
            request_id=request_id,
            finish_reason=finish_reason,
        )


def _tool_result_message(tool: str, result: Any, *, goal: str) -> str:
    return "TOOL_RESULT\n" + canonical_json(
        {"tool": tool, "research_goal": goal, "result": result}
    )


def _extract_search_urls(result: Any) -> list[str]:
    if not isinstance(result, Mapping) or result.get("tool") != "search":
        raise ValueError("live search executor returned an invalid object")
    rows = result.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("live search returned no results")
    urls: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"live search result {index} is invalid")
        url = row.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(f"live search result {index} has an invalid URL")
        if url not in urls:
            urls.append(url)
    if not urls:
        raise ValueError("live search returned no usable URLs")
    return urls


def _authoritative_record(value: LiveAuthoritativeResult) -> dict[str, Any]:
    return {
        "invocation": value.invocation.to_dict(),
        "source": value.source,
        "exposed_wait_s": value.exposed_wait_s,
        "queue_s": value.queue_s,
        "service_s": value.service_s,
        "saved_service_s": value.saved_service_s,
        "result_sha256": sha256_json(value.result),
    }


def _unique_context_padding(
    *, token_counter: TokenCounter, task_id: str, target_tokens: int
) -> tuple[str, int]:
    """Build deterministic, task-private long-context state for load studies.

    The text is deliberately unique per task instance so prefix caching cannot
    collapse the actual KV working set.  The common system prompt remains
    shared and therefore still exercises native prefix caching normally.
    """

    if target_tokens <= 0:
        return "", 0
    fingerprint = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    unit = (
        f"private-notebook-{fingerprint}: retain this task-specific evidence "
        "state; do not quote it in the answer. "
    )
    header = "PRIVATE_AGENT_HISTORY (capacity workload; ignore as evidence):\n"
    unit_tokens = max(1, token_counter.count_text(unit))
    repetitions = max(1, math.ceil(target_tokens / unit_tokens))
    for _ in range(8):
        value = header + unit * repetitions
        actual = token_counter.count_text(value)
        if target_tokens <= actual <= target_tokens + unit_tokens:
            return value, actual
        repetitions = max(1, math.ceil(repetitions * target_tokens / actual))
    value = header + unit * repetitions
    actual = token_counter.count_text(value)
    while actual < target_tokens:
        value += unit
        actual = token_counter.count_text(value)
    return value, actual


def _parse_guided_with_record(
    content: str,
    *,
    call_index: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse one completion and append an auditable success/failure record."""

    record: dict[str, Any] = {
        "call_index": int(call_index),
        "mode": "guided_json",
        "guided_json_requested": True,
        "json_parse_attempted": True,
        "local_wrap_applied": False,
        "contract_succeeded": False,
    }
    try:
        value = parse_guided_object(content, telemetry=record)
    except BaseException:
        record["parse_succeeded"] = False
        raise
    else:
        record["parse_succeeded"] = True
        return value
    finally:
        records.append(record)


def _guided_json_recovery_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "policy_version": GUIDED_JSON_RECOVERY_POLICY_VERSION,
        "parsed_call_count": len(records),
        "recovery_count": sum(
            1 for record in records if record.get("recovery_applied") is True
        ),
        "calls": [dict(record) for record in records],
    }


def _output_contract_summary(
    guided_records: Sequence[Mapping[str, Any]],
    final_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose the output mode and local contract used by every reached call."""

    calls: list[dict[str, Any]] = []
    for record in guided_records:
        calls.append(
            {
                "call_index": record.get("call_index"),
                "mode": "guided_json",
                "guided_json_requested": True,
                "json_parse_attempted": True,
                "local_wrap_applied": False,
                "parse_succeeded": record.get("parse_succeeded") is True,
                "contract_succeeded": record.get("contract_succeeded") is True,
                "recovery_applied": record.get("recovery_applied") is True,
                "raw_sha256": record.get("raw_sha256"),
            }
        )
    if final_record:
        calls.append(dict(final_record))
    return {
        "policy_version": (
            FIXED_OUTPUT_CONTRACT_POLICY_VERSION
            if final_record.get("guided_grammar_requested") is True
            else OUTPUT_CONTRACT_POLICY_VERSION
        ),
        "calls": calls,
    }


class LiveClosedLoopExperiment:
    def __init__(
        self,
        *,
        broker: LiveToolBroker,
        llm: LiveLLMClient,
        token_counter: TokenCounter,
        speculation_mode: str,
        visit_top_k: int,
        max_tokens_tool: int,
        max_tokens_answer: int,
        default_tool_service_s: float,
        predicted_visit_result_tokens: int,
        call_graph_mode: str = "autonomous",
        context_padding_tokens: int = 0,
        tool_signal_policy: str = "legacy",
        fixed_final_completion_tokens: int | None = None,
    ) -> None:
        if speculation_mode not in {"off", "search", "visit", "search_visit"}:
            raise ValueError(f"unsupported speculation mode: {speculation_mode}")
        if visit_top_k <= 0:
            raise ValueError("visit_top_k must be positive")
        if call_graph_mode not in {"autonomous", "frozen"}:
            raise ValueError(f"unsupported call graph mode: {call_graph_mode}")
        if context_padding_tokens < 0:
            raise ValueError("context_padding_tokens cannot be negative")
        if tool_signal_policy not in {"legacy", "execution_aware"}:
            raise ValueError(
                f"unsupported tool signal policy: {tool_signal_policy}"
            )
        if fixed_final_completion_tokens is not None and (
            type(fixed_final_completion_tokens) is not int
            or fixed_final_completion_tokens != FINAL_COMPLETION_TOKEN_COUNT
        ):
            raise ValueError(
                "fixed_final_completion_tokens must be None or "
                f"{FINAL_COMPLETION_TOKEN_COUNT}"
            )
        if fixed_final_completion_tokens is not None:
            token_counter_method = getattr(token_counter, "method", None)
            if (
                not isinstance(token_counter_method, str)
                or not token_counter_method
                or token_counter_method == ApproximateTokenCounter.method
            ):
                raise ValueError(
                    "fixed final completion requires the exact model tokenizer"
                )
        self.broker = broker
        self.llm = llm
        self.token_counter = token_counter
        self.speculation_mode = speculation_mode
        self.visit_top_k = visit_top_k
        self.max_tokens_tool = max_tokens_tool
        self.max_tokens_answer = max_tokens_answer
        self.default_tool_service_s = default_tool_service_s
        self.call_graph_mode = call_graph_mode
        self.context_padding_tokens = int(context_padding_tokens)
        self.tool_signal_policy = tool_signal_policy
        self.fixed_final_completion_tokens = fixed_final_completion_tokens
        self._visit_result_tokens_ema = float(predicted_visit_result_tokens)
        self._ema_lock = asyncio.Lock()

    async def _complete(
        self,
        *,
        task_id: str,
        call_index: int,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any] | None,
        max_tokens: int,
        next_tool_name: str | None,
        next_tool_confidence: float,
        next_tool_invocation: Invocation | None = None,
        next_tool_signal_eligible: bool = True,
        grammar: str | None = None,
        min_tokens: int = 0,
    ) -> LLMCompletion:
        prompt_tokens = self.token_counter.count_messages(messages)
        predicted_output = (
            max_tokens
            if min_tokens == max_tokens
            else min(max_tokens, 96 if call_index < 2 else 128)
        )
        next_prompt_tokens = None
        if call_index < 2:
            next_prompt_tokens = prompt_tokens + int(self._visit_result_tokens_ema)
        global_snapshot = self.broker.snapshot()
        session_snapshot = self.broker.snapshot(session_id=task_id)
        effective_confidence, signal_evidence = _broker_tool_signal(
            session_snapshot,
            invocation=next_tool_invocation,
            nominal_confidence=next_tool_confidence,
            policy=self.tool_signal_policy,
            eligible=next_tool_signal_eligible,
        )
        meta = build_scheduler_meta(
            task_id=task_id,
            call_index=call_index,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            predicted_output_tokens=predicted_output,
            broker_global=global_snapshot,
            broker_session=session_snapshot,
            next_tool_name=next_tool_name,
            next_tool_confidence=effective_confidence,
            next_prompt_tokens=next_prompt_tokens,
            default_tool_service_s=self.default_tool_service_s,
            next_tool_signal_evidence=signal_evidence,
        )
        return await self.llm.complete(
            task_id=task_id,
            call_index=call_index,
            messages=messages,
            request_id=scheduler_request_id(meta),
            max_tokens=max_tokens,
            schema=schema,
            prompt_tokens=prompt_tokens,
            scheduler_meta=meta,
            grammar=grammar,
            min_tokens=min_tokens,
        )

    async def run_task(
        self,
        source: LiveSource,
        *,
        replica: int = 0,
        visit_speculation_eligible: bool = True,
    ) -> dict[str, Any]:
        task_id = f"{source.source_id}__r{replica:02d}"
        started_wall = time.time()
        started = time.monotonic()
        question_message = (
            "TASK\n"
            f"RESEARCH_GOAL: {source.question}\n"
            f"SEARCH_QUERY: {source.search_query}"
        )
        context_padding, context_padding_actual_tokens = _unique_context_padding(
            token_counter=self.token_counter,
            task_id=task_id,
            target_tokens=self.context_padding_tokens,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if context_padding:
            messages.append({"role": "user", "content": context_padding})
        messages.append({"role": "user", "content": question_message})
        search_invocation = Invocation("search", {"query": [source.search_query]})
        tool_records: list[dict[str, Any]] = []
        guided_json_records: list[dict[str, Any]] = []
        final_answer_record: dict[str, Any] = {}
        search_result_contains_expected_url: bool | None = None

        try:
            if self.call_graph_mode == "frozen" and source.expected_url is None:
                raise ValueError(
                    "frozen call graph task requires an absolute HTTPS expected_url"
                )
            if self.speculation_mode in {"search", "search_visit"}:
                await self.broker.speculate(
                    search_invocation, session_id=task_id, priority=1.0
                )

            first = await self._complete(
                task_id=task_id,
                call_index=0,
                messages=messages,
                schema=search_schema(source.search_query),
                max_tokens=self.max_tokens_tool,
                next_tool_name="search",
                next_tool_confidence=(
                    1.0 if self.speculation_mode in {"search", "search_visit"} else 0.5
                ),
                next_tool_invocation=search_invocation,
            )
            search_call = _parse_guided_with_record(
                first.content, call_index=0, records=guided_json_records
            )
            # Compare the complete wire object so extra or mutated arguments
            # cannot cross the authoritative commit boundary.
            expected = {"name": "search", "arguments": search_invocation.arguments}
            if search_call != expected:
                raise ValueError(
                    f"search call differs from guided authoritative call: {search_call!r}"
                )
            guided_json_records[-1]["contract_succeeded"] = True
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        canonical_json(search_call)
                        if self.fixed_final_completion_tokens is not None
                        else first.content
                    ),
                }
            )
            search_result = await self.broker.authoritative(
                search_invocation, session_id=task_id
            )
            tool_records.append(_authoritative_record(search_result))
            urls = _extract_search_urls(search_result.result)
            if source.expected_url is not None:
                search_result_contains_expected_url = source.expected_url in urls
            if self.call_graph_mode == "frozen":
                # The defensive check at task entry narrows this at runtime;
                # keep the branch explicit so no optional URL can reach an
                # Invocation or guided schema.
                assert source.expected_url is not None
                visit_candidates = [source.expected_url]
            else:
                visit_candidates = urls

            # Canary visits are an authoritative-only negative control.  Do
            # not enqueue a prediction that is known up front to be
            # ineligible for reuse: doing so would consume shared worker/rate
            # capacity and turn the canary itself into guaranteed waste.
            if (
                visit_speculation_eligible
                and self.speculation_mode in {"visit", "search_visit"}
            ):
                for rank, url in enumerate(
                    visit_candidates[: self.visit_top_k], start=1
                ):
                    await self.broker.speculate(
                        Invocation(
                            "visit", {"url": [url], "goal": source.question}
                        ),
                        session_id=task_id,
                        priority=1.0 / rank,
                    )

            messages.append(
                {
                    "role": "user",
                    "content": _tool_result_message(
                        "search", search_result.result, goal=source.question
                    ),
                }
            )
            second = await self._complete(
                task_id=task_id,
                call_index=1,
                messages=messages,
                schema=visit_schema(visit_candidates, source.question),
                max_tokens=self.max_tokens_tool,
                next_tool_name="visit",
                next_tool_confidence=(
                    1.0
                    if self.speculation_mode in {"visit", "search_visit"}
                    else 0.5
                ),
                next_tool_invocation=(
                    Invocation(
                        "visit",
                        {"url": [visit_candidates[0]], "goal": source.question},
                    )
                    if len(visit_candidates) == 1
                    else None
                ),
                next_tool_signal_eligible=visit_speculation_eligible,
            )
            visit_call = _parse_guided_with_record(
                second.content, call_index=1, records=guided_json_records
            )
            if visit_call.get("name") != "visit" or not isinstance(
                visit_call.get("arguments"), Mapping
            ):
                raise ValueError(f"invalid guided visit call: {visit_call!r}")
            visit_invocation = Invocation("visit", visit_call["arguments"])
            selected_urls = visit_invocation.arguments.get("url")
            allowed_urls = visit_candidates
            if (
                not isinstance(selected_urls, list)
                or len(selected_urls) != 1
                or selected_urls[0] not in allowed_urls
            ):
                if self.call_graph_mode == "frozen":
                    raise ValueError("model selected a URL outside the frozen call graph")
                raise ValueError("model selected a URL outside the live search result")
            selected_url = str(selected_urls[0])
            guided_json_records[-1]["contract_succeeded"] = True
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        canonical_json(visit_call)
                        if self.fixed_final_completion_tokens is not None
                        else second.content
                    ),
                }
            )
            visit_result = await self.broker.authoritative(
                visit_invocation,
                session_id=task_id,
                speculation_eligible=visit_speculation_eligible,
            )
            tool_records.append(_authoritative_record(visit_result))
            await self.broker.cancel_predictions(session_id=task_id)

            result_tokens = self.token_counter.count_text(canonical_json(visit_result.result))
            async with self._ema_lock:
                self._visit_result_tokens_ema = (
                    0.2 * result_tokens + 0.8 * self._visit_result_tokens_ema
                )
            messages.append(
                {
                    "role": "user",
                    "content": _tool_result_message(
                        "visit", visit_result.result, goal=source.question
                    ),
                }
            )
            fixed_final = self.fixed_final_completion_tokens
            final_grammar = (
                final_answer_fixed_completion_grammar(selected_url)
                if fixed_final is not None
                else None
            )
            third = await self._complete(
                task_id=task_id,
                call_index=2,
                messages=messages,
                schema=(
                    None if fixed_final is not None else final_answer_schema(selected_url)
                ),
                grammar=final_grammar,
                max_tokens=(
                    fixed_final
                    if fixed_final is not None
                    else self.max_tokens_answer
                ),
                min_tokens=fixed_final or 0,
                next_tool_name=None,
                next_tool_confidence=1.0,
            )
            answer = parse_guided_final_answer(
                third.content,
                url=selected_url,
                call_index=2,
                telemetry=final_answer_record,
                token_counter=(
                    self.token_counter if fixed_final is not None else None
                ),
                completion_tokens=third.usage.get("completion_tokens"),
                finish_reason=third.finish_reason,
                fixed_completion_tokens=fixed_final,
            )

            return {
                "task_id": task_id,
                "source_id": source.source_id,
                "replica": replica,
                "ok": True,
                "visit_canary": not visit_speculation_eligible,
                "start_wall_s": started_wall,
                "end_wall_s": time.time(),
                "e2e_s": time.monotonic() - started,
                "question_sha256": hashlib.sha256(
                    source.question.encode("utf-8")
                ).hexdigest(),
                "search_query": source.search_query,
                "search_urls": urls,
                "call_graph_mode": self.call_graph_mode,
                "expected_url": source.expected_url,
                "search_result_contains_expected_url": (
                    search_result_contains_expected_url
                ),
                "selected_url": selected_url,
                "answer": answer,
                "answer_sha256": sha256_json(answer),
                "guided_json_recovery": _guided_json_recovery_summary(
                    guided_json_records
                ),
                "output_contract": _output_contract_summary(
                    guided_json_records, final_answer_record
                ),
                "final_answer_contract": dict(final_answer_record),
                "tools": tool_records,
                "llm_duration_s": first.duration_s
                + second.duration_s
                + third.duration_s,
                "completion_tokens": sum(
                    call.usage["completion_tokens"] for call in (first, second, third)
                ),
                "prompt_tokens": sum(
                    call.usage["prompt_tokens"] for call in (first, second, third)
                ),
                "context_padding_target_tokens": self.context_padding_tokens,
                "context_padding_actual_tokens": context_padding_actual_tokens,
            }
        except BaseException as exc:
            await self.broker.cancel_predictions(session_id=task_id)
            return {
                "task_id": task_id,
                "source_id": source.source_id,
                "replica": replica,
                "ok": False,
                "visit_canary": not visit_speculation_eligible,
                "start_wall_s": started_wall,
                "end_wall_s": time.time(),
                "e2e_s": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "call_graph_mode": self.call_graph_mode,
                "expected_url": source.expected_url,
                "search_result_contains_expected_url": (
                    search_result_contains_expected_url
                ),
                "guided_json_recovery": _guided_json_recovery_summary(
                    guided_json_records
                ),
                "output_contract": _output_contract_summary(
                    guided_json_records, final_answer_record
                ),
                "final_answer_contract": dict(final_answer_record),
                "tools": tool_records,
                "context_padding_target_tokens": self.context_padding_tokens,
                "context_padding_actual_tokens": context_padding_actual_tokens,
            }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_live_run(
    *,
    tasks: Sequence[Mapping[str, Any]],
    llm_events: Sequence[Mapping[str, Any]],
    broker_stats: Mapping[str, Any],
    started_wall_s: float,
    ended_wall_s: float,
) -> dict[str, Any]:
    successes = [task for task in tasks if task.get("ok") is True]
    e2e = [float(task["e2e_s"]) for task in successes]
    llm_ok = [event for event in llm_events if event.get("ok") is True]
    exposed_tool_wait = [
        float(tool.get("exposed_wait_s", 0.0))
        for task in successes
        for tool in task.get("tools", [])
    ]
    tool_queue = [
        float(tool.get("queue_s", 0.0))
        for task in successes
        for tool in task.get("tools", [])
    ]
    tool_service = [
        float(tool.get("service_s", 0.0))
        for task in successes
        for tool in task.get("tools", [])
    ]

    task_metrics: dict[str, Any] = {"available": bool(e2e)}
    if e2e:
        task_metrics.update(
            {
                "mean_s": statistics.fmean(e2e),
                "p50_s": percentile(e2e, 0.50),
                "p95_s": percentile(e2e, 0.95),
                "p99_s": percentile(e2e, 0.99),
                "max_s": max(e2e),
            }
        )
    return {
        "started_wall_s": started_wall_s,
        "ended_wall_s": ended_wall_s,
        "makespan_s": ended_wall_s - started_wall_s,
        "task_count": len(tasks),
        "successful_task_count": len(successes),
        "failed_task_count": len(tasks) - len(successes),
        "all_tasks_succeeded": len(successes) == len(tasks),
        "task_e2e": task_metrics,
        "llm": {
            "request_count": len(llm_events),
            "successful_request_count": len(llm_ok),
            "exactly_one_attempt_each": all(
                int(event.get("attempts", 0)) == 1 for event in llm_events
            ),
            "mean_request_s": (
                statistics.fmean(float(event["duration_s"]) for event in llm_ok)
                if llm_ok
                else None
            ),
            "completion_tokens": sum(
                int(event.get("usage", {}).get("completion_tokens", 0) or 0)
                for event in llm_ok
            ),
            "prompt_tokens": sum(
                int(event.get("usage", {}).get("prompt_tokens", 0) or 0)
                for event in llm_ok
            ),
        },
        "tool": {
            "authoritative_commit_count": len(exposed_tool_wait),
            "mean_exposed_wait_s": (
                statistics.fmean(exposed_tool_wait) if exposed_tool_wait else None
            ),
            "mean_queue_s": statistics.fmean(tool_queue) if tool_queue else None,
            "mean_service_s": (
                statistics.fmean(tool_service) if tool_service else None
            ),
            "broker_stats": dict(broker_stats),
        },
    }


def task_to_dict(source: LiveSource) -> dict[str, Any]:
    value = asdict(source)
    # Preserve autonomous-workload identity for existing workloads that do not
    # opt into a frozen authoritative URL.
    if value["expected_url"] is None:
        value.pop("expected_url")
    return value
