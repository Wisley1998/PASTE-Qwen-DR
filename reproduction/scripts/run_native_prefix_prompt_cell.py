#!/usr/bin/env python3
"""Run one single-token, local-only native-prefix causal cell against vLLM."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import aiohttp


SCHEMA_VERSION = "paste_repro.native_prefix_prompt_cell_v2"
FIXTURE_VERSION = "three-call-local-prefix-single-token-fixture-v2"
OUTPUT_CONSTRAINT = "guided_choice_singleton_v1"
DEFAULT_SENTINEL = "A"
CALL_COUNT = 3
INTERESTING_METRICS = (
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)

SYSTEM_PROMPT = """You are in a local prefix-cache benchmark. For every request,
emit exactly the one-character sentinel A and nothing else. The server applies
a singleton guided-choice constraint and a one-token generation cap. Historical
assistant/tool turns in later requests are canonical fixtures fixed before either
cache treatment starts; they are not copied from runtime model output. This
benchmark executes no external tool and uses no external network.
"""


class Tokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Sequence[int]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


@dataclass(frozen=True)
class Source:
    source_id: str
    question: str
    search_query: str
    expected_url: str


@dataclass(frozen=True)
class CallFixture:
    call_index: int
    messages: tuple[dict[str, str], ...]
    messages_sha256: str
    prompt_tokens: int
    prompt_token_ids_sha256: str
    max_tokens: int
    expected_completion: str
    expected_completion_sha256: str
    expected_completion_tokens: int
    guided_choice: tuple[str, ...]


@dataclass(frozen=True)
class TaskFixture:
    task_id: str
    source_id: str
    replica: int
    context_padding_actual_tokens: int
    calls: tuple[CallFixture, ...]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("server URL must be an uncredentialed loopback HTTP URL")
    if parsed.port is None:
        raise ValueError("server URL must include an explicit port")
    return value.rstrip("/")


def load_sources(
    path: Path,
    *,
    expected_sha256: str,
    expected_count: int,
) -> tuple[list[Source], dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("frozen tune workload SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("split_role") != "tune"
        or payload.get("formal_eligible") is not False
        or payload.get("split_id") != "live-joint-wikipedia-frozen-tune-v1"
    ):
        raise ValueError("workload is not the frozen non-formal tune split")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != expected_count:
        raise ValueError("frozen tune source count mismatch")
    sources: list[Source] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_sources):
        if not isinstance(row, Mapping):
            raise ValueError(f"source {index} is not an object")
        source_id = _nonempty_string(row.get("source_id"), f"source {index}.source_id")
        question = _nonempty_string(row.get("question"), f"source {index}.question")
        query = _nonempty_string(row.get("search_query"), f"source {index}.search_query")
        url = _nonempty_string(row.get("expected_url"), f"source {index}.expected_url")
        parsed_url = urlsplit(url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(f"source {index} expected_url must be absolute HTTPS")
        if source_id in seen:
            raise ValueError("frozen tune source IDs are not unique")
        seen.add(source_id)
        sources.append(Source(source_id, question, query, url))
    return sources, payload


def _render_message_tokens(
    tokenizer: Tokenizer, messages: Sequence[Mapping[str, str]]
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True
    )
    return [int(token) for token in rendered]


def _count_text(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def validate_single_token_sentinel(
    tokenizer: Tokenizer, sentinel: str = DEFAULT_SENTINEL
) -> dict[str, Any]:
    """Prove the singleton constraint has one exact, ordinary tokenizer token."""
    if not isinstance(sentinel, str) or not sentinel or sentinel.strip() != sentinel:
        raise ValueError("sentinel must be a non-empty string without edge whitespace")
    token_ids = [
        int(token) for token in tokenizer.encode(sentinel, add_special_tokens=False)
    ]
    if len(token_ids) != 1:
        raise ValueError("sentinel must encode to exactly one token")
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != sentinel:
        raise ValueError("sentinel token must decode byte-for-byte to the sentinel")
    special_ids = {
        int(token) for token in getattr(tokenizer, "all_special_ids", ())
    }
    if token_ids[0] in special_ids:
        raise ValueError("sentinel token must not be a special token")
    guided_choice = [sentinel]
    return {
        "contract": OUTPUT_CONSTRAINT,
        "sentinel": sentinel,
        "sentinel_utf8_sha256": hashlib.sha256(
            sentinel.encode("utf-8")
        ).hexdigest(),
        "token_id": token_ids[0],
        "token_ids_sha256": sha256_json(token_ids),
        "token_count": 1,
        "allowed_choice_count": 1,
        "guided_choice": guided_choice,
        "guided_choice_sha256": sha256_json(guided_choice),
        "max_tokens": 1,
        "round_trip_exact": True,
        "special_token": False,
    }


def _unique_context_padding(
    tokenizer: Tokenizer,
    *,
    task_id: str,
    target_tokens: int,
) -> tuple[str, int]:
    fingerprint = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    unit = (
        f"private-notebook-{fingerprint}: retain this task-specific evidence "
        "state; do not quote it in the answer. "
    )
    header = "PRIVATE_AGENT_HISTORY (capacity workload; ignore as evidence):\n"
    unit_tokens = max(1, _count_text(tokenizer, unit))
    repetitions = max(1, math.ceil(target_tokens / unit_tokens))
    for _ in range(12):
        value = header + unit * repetitions
        actual = _count_text(tokenizer, value)
        if target_tokens <= actual <= target_tokens + unit_tokens:
            return value, actual
        repetitions = max(1, math.ceil(repetitions * target_tokens / max(1, actual)))
    value = header + unit * repetitions
    actual = _count_text(tokenizer, value)
    while actual < target_tokens:
        value += unit
        actual = _count_text(tokenizer, value)
    return value, actual


def _fixed_visit_content(
    tokenizer: Tokenizer,
    *,
    task_id: str,
    target_tokens: int,
) -> tuple[str, int]:
    fingerprint = hashlib.sha256(("visit\0" + task_id).encode("utf-8")).hexdigest()[:12]
    unit = (
        f"local-page-{fingerprint}: deterministic evidence sentence retained "
        "only to reproduce the long visit-result continuation shape. "
    )
    unit_tokens = max(1, _count_text(tokenizer, unit))
    repetitions = max(1, math.ceil(target_tokens / unit_tokens))
    for _ in range(12):
        value = unit * repetitions
        actual = _count_text(tokenizer, value)
        if target_tokens <= actual <= target_tokens + unit_tokens:
            return value, actual
        repetitions = max(1, math.ceil(repetitions * target_tokens / max(1, actual)))
    value = unit * repetitions
    actual = _count_text(tokenizer, value)
    while actual < target_tokens:
        value += unit
        actual = _count_text(tokenizer, value)
    return value, actual


def _tool_result_message(tool: str, result: Mapping[str, Any], *, goal: str) -> str:
    return "TOOL_RESULT\n" + canonical_json(
        {"tool": tool, "research_goal": goal, "result": dict(result)}
    )


def _search_fixture(source: Source) -> dict[str, Any]:
    rows = []
    # Two compact rows reproduce the observed live search-turn token increment
    # without carrying any backend response across cache treatments.
    for rank in range(1, 3):
        url = source.expected_url if rank == 1 else f"{source.expected_url}#local-rank-{rank}"
        rows.append(
            {
                "rank": rank,
                "title": f"{source.search_query} local result {rank}",
                "url": url,
                "snippet": (
                    f"Deterministic local search evidence for {source.source_id}; "
                    f"rank {rank} is fixed before either cache treatment."
                ),
            }
        )
    return {
        "tool": "search",
        "backend": "deterministic-local-fixture-v1",
        "query": [source.search_query],
        "results": rows,
    }


def build_task_fixtures(
    tokenizer: Tokenizer,
    sources: Sequence[Source],
    *,
    replicas: int,
    context_padding_tokens: int,
    visit_fixture_tokens: int,
    max_tokens_by_call: Sequence[int],
    max_model_len: int,
    sentinel: str = DEFAULT_SENTINEL,
) -> tuple[list[TaskFixture], str]:
    if replicas <= 0 or context_padding_tokens <= 0 or visit_fixture_tokens <= 0:
        raise ValueError("fixture counts and token targets must be positive")
    if tuple(max_tokens_by_call) != (1, 1, 1):
        raise ValueError("v2 requires a one-token generation cap on all three calls")
    sentinel_contract = validate_single_token_sentinel(tokenizer, sentinel)

    tasks: list[TaskFixture] = []
    manifest: list[dict[str, Any]] = []
    for source in sources:
        for replica in range(replicas):
            task_id = f"{source.source_id}__r{replica:02d}"
            padding, padding_tokens = _unique_context_padding(
                tokenizer,
                task_id=task_id,
                target_tokens=context_padding_tokens,
            )
            task_message = (
                "TASK\n"
                f"RESEARCH_GOAL: {source.question}\n"
                f"SEARCH_QUERY: {source.search_query}"
            )
            base_messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": padding},
                {"role": "user", "content": task_message},
            ]
            search_call = {
                "name": "search",
                "arguments": {"query": [source.search_query]},
            }
            search_result = _search_fixture(source)
            second_messages = [
                *base_messages,
                {"role": "assistant", "content": canonical_json(search_call)},
                {
                    "role": "user",
                    "content": _tool_result_message(
                        "search", search_result, goal=source.question
                    ),
                },
            ]
            visit_call = {
                "name": "visit",
                "arguments": {
                    "url": [source.expected_url],
                    "goal": source.question,
                },
            }
            visit_content, visit_actual_tokens = _fixed_visit_content(
                tokenizer,
                task_id=task_id,
                target_tokens=visit_fixture_tokens,
            )
            visit_result = {
                "tool": "visit",
                "backend": "deterministic-local-fixture-v1",
                "url": source.expected_url,
                "goal": source.question,
                "content": visit_content,
                "content_tokens": visit_actual_tokens,
            }
            third_messages = [
                *second_messages,
                {"role": "assistant", "content": canonical_json(visit_call)},
                {
                    "role": "user",
                    "content": _tool_result_message(
                        "visit", visit_result, goal=source.question
                    ),
                },
            ]
            message_sets = (base_messages, second_messages, third_messages)
            calls: list[CallFixture] = []
            for call_index, (messages, max_tokens) in enumerate(
                zip(message_sets, max_tokens_by_call)
            ):
                prompt_token_ids = _render_message_tokens(tokenizer, messages)
                prompt_tokens = len(prompt_token_ids)
                expected_completion_tokens = sentinel_contract["token_count"]
                if prompt_tokens + max_tokens > max_model_len:
                    raise ValueError(
                        f"fixture {task_id}/{call_index} exceeds max model length"
                    )
                calls.append(
                    CallFixture(
                        call_index=call_index,
                        messages=tuple(dict(row) for row in messages),
                        messages_sha256=sha256_json(messages),
                        prompt_tokens=prompt_tokens,
                        prompt_token_ids_sha256=sha256_json(prompt_token_ids),
                        max_tokens=int(max_tokens),
                        expected_completion=sentinel,
                        expected_completion_sha256=sha256_json(sentinel),
                        expected_completion_tokens=expected_completion_tokens,
                        guided_choice=(sentinel,),
                    )
                )
            prompt_sizes = [call.prompt_tokens for call in calls]
            if not (
                context_padding_tokens <= prompt_sizes[0] <= context_padding_tokens + 768
                and prompt_sizes[0] < prompt_sizes[1] < prompt_sizes[2]
                and 64 <= prompt_sizes[1] - prompt_sizes[0] <= 768
                and 640 <= prompt_sizes[2] - prompt_sizes[1] <= 1536
            ):
                raise ValueError(
                    f"fixture {task_id} does not match the frozen prompt topology: "
                    f"{prompt_sizes}"
                )
            task = TaskFixture(
                task_id=task_id,
                source_id=source.source_id,
                replica=replica,
                context_padding_actual_tokens=padding_tokens,
                calls=tuple(calls),
            )
            tasks.append(task)
            manifest.append(
                {
                    "task_id": task_id,
                    "source_id": source.source_id,
                    "replica": replica,
                    "context_padding_actual_tokens": padding_tokens,
                    "calls": [
                        {
                            "call_index": call.call_index,
                            "messages_sha256": call.messages_sha256,
                            "prompt_tokens": call.prompt_tokens,
                            "prompt_token_ids_sha256": (
                                call.prompt_token_ids_sha256
                            ),
                            "max_tokens": call.max_tokens,
                            "expected_completion_tokens": (
                                call.expected_completion_tokens
                            ),
                            "expected_completion_sha256": (
                                call.expected_completion_sha256
                            ),
                            "guided_choice_sha256": sha256_json(
                                list(call.guided_choice)
                            ),
                        }
                        for call in calls
                    ],
                }
            )
    return tasks, sha256_json(
        {
            "fixture_version": FIXTURE_VERSION,
            "sentinel_contract": sentinel_contract,
            "tasks": manifest,
        }
    )


async def _fetch_metrics(
    session: aiohttp.ClientSession, server_url: str
) -> tuple[dict[str, float], set[str], str]:
    from prometheus_client.parser import text_string_to_metric_families

    async with session.get(
        f"{server_url}/metrics", timeout=aiohttp.ClientTimeout(total=10)
    ) as response:
        response.raise_for_status()
        payload = await response.text()
    values: dict[str, float] = {}
    present: set[str] = set()
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            present.add(sample.name)
            values[sample.name] = values.get(sample.name, 0.0) + float(sample.value)
    return values, present, payload


def _metric_deltas(
    before: Mapping[str, float],
    after: Mapping[str, float],
    before_present: set[str],
    after_present: set[str],
) -> tuple[dict[str, float], dict[str, dict[str, bool]]]:
    deltas = {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in INTERESTING_METRICS
    }
    presence = {
        key: {"before": key in before_present, "after": key in after_present}
        for key in INTERESTING_METRICS
    }
    return deltas, presence


async def _sample_queues(
    *,
    session: aiohttp.ClientSession,
    server_url: str,
    stop: asyncio.Event,
    interval_s: float,
    rows: list[dict[str, Any]],
) -> None:
    while True:
        sample: dict[str, Any] = {
            "wall_s": time.time(),
            "monotonic_s": time.monotonic(),
        }
        try:
            metrics, _present, _raw = await _fetch_metrics(session, server_url)
            sample.update(
                {
                    "ok": True,
                    "llm_running": metrics.get("vllm:num_requests_running"),
                    "llm_waiting": metrics.get("vllm:num_requests_waiting"),
                    "gpu_cache_usage": metrics.get("vllm:gpu_cache_usage_perc"),
                }
            )
        except Exception as exc:
            sample.update(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )
        rows.append(sample)
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timeline_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("ok") is True
        and isinstance(row.get("llm_running"), (int, float))
        and isinstance(row.get("llm_waiting"), (int, float))
    ]
    return {
        "sample_count": len(rows),
        "valid_sample_count": len(valid),
        "failed_sample_count": len(rows) - len(valid),
        "max_llm_running": max(
            (float(row["llm_running"]) for row in valid), default=None
        ),
        "max_llm_waiting": max(
            (float(row["llm_waiting"]) for row in valid), default=None
        ),
        "waiting_sample_fraction": (
            sum(float(row["llm_waiting"]) > 0 for row in valid) / len(valid)
            if valid
            else None
        ),
        "max_gpu_cache_usage": max(
            (
                float(row["gpu_cache_usage"])
                for row in valid
                if isinstance(row.get("gpu_cache_usage"), (int, float))
            ),
            default=None,
        ),
    }


async def run_cell(args: argparse.Namespace) -> int:
    server_url = _validate_loopback_url(args.server_url)
    if args.cell_id not in {"P0", "P1"}:
        raise ValueError("cell must be P0 or P1")
    expected_prefix_enabled = args.cell_id == "P1"
    if args.prefix_cache_enabled != expected_prefix_enabled:
        raise ValueError("cell ID and declared prefix-cache state disagree")
    if args.replicas <= 0 or args.max_active_tasks <= 0:
        raise ValueError("replicas and active-task count must be positive")
    if args.queue_sample_interval_s <= 0 or args.request_timeout_s <= 0:
        raise ValueError("timeouts and sample intervals must be positive")
    if args.max_active_tasks != args.expected_task_count:
        raise ValueError("this frozen experiment launches all 48 tasks locally")

    scheduler_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("VLLM_SCHED_")
    }
    if scheduler_environment != {"VLLM_SCHED_POLICY": "fcfs"}:
        raise ValueError(
            "native-prefix cell must expose only VLLM_SCHED_POLICY=fcfs"
        )
    native_pythonpath = Path(os.environ.get("VLLM_HOOK_DIR", "")).resolve()
    if (
        not native_pythonpath.is_dir()
        or any(native_pythonpath.iterdir())
        or os.environ.get("PYTHONPATH", "") != ""
    ):
        raise ValueError("native-prefix cell must use an empty isolated hook path")
    env_prefix = os.environ.get("VLLM_ENABLE_PREFIX_CACHING")
    if env_prefix != ("1" if expected_prefix_enabled else "0"):
        raise ValueError("effective process prefix flag disagrees with cell")

    workload = args.workload.resolve()
    sources, workload_payload = load_sources(
        workload,
        expected_sha256=args.workload_sha256,
        expected_count=args.expected_source_count,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    tasks, fixture_manifest_sha256 = build_task_fixtures(
        tokenizer,
        sources,
        replicas=args.replicas,
        context_padding_tokens=args.context_padding_tokens,
        visit_fixture_tokens=args.visit_fixture_tokens,
        max_tokens_by_call=(
            args.max_tokens_call0,
            args.max_tokens_call1,
            args.max_tokens_call2,
        ),
        max_model_len=args.max_model_len,
        sentinel=args.sentinel,
    )
    sentinel_contract = validate_single_token_sentinel(tokenizer, args.sentinel)
    if args.output_constraint != OUTPUT_CONSTRAINT:
        raise ValueError("unsupported output constraint contract")
    if len(tasks) != args.expected_task_count:
        raise ValueError("deterministic fixture task count mismatch")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    api_key = os.getenv("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=15)
    llm_events: list[dict[str, Any]] = []
    started_wall_s = time.time()
    started_monotonic_s = time.monotonic()
    queue_rows: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        before, before_present, before_raw = await _fetch_metrics(
            session, server_url
        )
        stop_sampler = asyncio.Event()
        sampler = asyncio.create_task(
            _sample_queues(
                session=session,
                server_url=server_url,
                stop=stop_sampler,
                interval_s=args.queue_sample_interval_s,
                rows=queue_rows,
            )
        )
        semaphore = asyncio.Semaphore(args.max_active_tasks)

        async def complete(task: TaskFixture, call: CallFixture) -> dict[str, Any]:
            request_id = f"prefixcausal-{task.task_id}-c{call.call_index}"
            payload = {
                "model": args.model,
                "messages": list(call.messages),
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
                "max_tokens": call.max_tokens,
                "request_id": request_id,
                "guided_choice": list(call.guided_choice),
            }
            started_wall = time.time()
            started = time.monotonic()
            event: dict[str, Any] = {
                "task_id": task.task_id,
                "source_id": task.source_id,
                "replica": task.replica,
                "call_index": call.call_index,
                "request_id": request_id,
                "request_start_wall_s": started_wall,
                "request_start_monotonic_s": started,
                "attempts": 1,
                "messages_sha256": call.messages_sha256,
                "prompt_tokens_estimate": call.prompt_tokens,
                "prompt_token_ids_sha256": call.prompt_token_ids_sha256,
                "max_tokens": call.max_tokens,
                "request_payload_sha256": sha256_json(payload),
                "guided_choice_sha256": sha256_json(list(call.guided_choice)),
                "expected_completion": call.expected_completion,
                "expected_completion_sha256": call.expected_completion_sha256,
                "expected_completion_tokens_estimate": (
                    call.expected_completion_tokens
                ),
            }
            try:
                async with session.post(
                    f"{server_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=args.request_timeout_s),
                ) as response:
                    event["http_status"] = response.status
                    body = await response.json(content_type=None)
                    if response.status != 200:
                        event["error_body"] = body
                        raise RuntimeError(f"vLLM returned HTTP {response.status}")
                content = body.get("choices", [{}])[0].get("message", {}).get(
                    "content", ""
                )
                if not isinstance(content, str):
                    raise ValueError("vLLM response content is not text")
                if content != call.expected_completion:
                    raise ValueError("guided choice differs from one-token sentinel")
                usage_raw = body.get("usage")
                if not isinstance(usage_raw, Mapping):
                    raise ValueError("vLLM response lacks usage")
                usage = {
                    "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
                    "completion_tokens": int(
                        usage_raw.get("completion_tokens", 0)
                    ),
                    "total_tokens": int(usage_raw.get("total_tokens", 0)),
                }
                if usage["prompt_tokens"] != call.prompt_tokens:
                    raise ValueError("server and local prompt token counts differ")
                if usage["completion_tokens"] != 1:
                    raise ValueError("guided sentinel did not consume exactly one token")
                event.update(
                    {
                        "ok": True,
                        "duration_s": time.monotonic() - started,
                        "response": content,
                        "response_sha256": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                        "semantic_response_sha256": sha256_json(content),
                        "usage": usage,
                    }
                )
                return event
            except BaseException as exc:
                event.update(
                    {
                        "ok": False,
                        "duration_s": time.monotonic() - started,
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    }
                )
                raise
            finally:
                llm_events.append(event)

        async def run_one(task: TaskFixture) -> dict[str, Any]:
            async with semaphore:
                task_started_wall = time.time()
                task_started = time.monotonic()
                completed_calls: list[int] = []
                try:
                    for call in task.calls:
                        await complete(task, call)
                        completed_calls.append(call.call_index)
                    return {
                        "task_id": task.task_id,
                        "source_id": task.source_id,
                        "replica": task.replica,
                        "ok": True,
                        "started_wall_s": task_started_wall,
                        "ended_wall_s": time.time(),
                        "e2e_s": time.monotonic() - task_started,
                        "completed_call_indices": completed_calls,
                        "context_padding_actual_tokens": (
                            task.context_padding_actual_tokens
                        ),
                    }
                except BaseException as exc:
                    return {
                        "task_id": task.task_id,
                        "source_id": task.source_id,
                        "replica": task.replica,
                        "ok": False,
                        "started_wall_s": task_started_wall,
                        "ended_wall_s": time.time(),
                        "e2e_s": time.monotonic() - task_started,
                        "completed_call_indices": completed_calls,
                        "context_padding_actual_tokens": (
                            task.context_padding_actual_tokens
                        ),
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    }

        task_results = await asyncio.gather(*(run_one(task) for task in tasks))
        tasks_ended_wall_s = time.time()
        stop_sampler.set()
        await sampler
        after, after_present, after_raw = await _fetch_metrics(session, server_url)

    ended_wall_s = time.time()
    metric_deltas, metric_presence = _metric_deltas(
        before, after, before_present, after_present
    )
    successes = [task for task in task_results if task.get("ok") is True]
    successful_events = [event for event in llm_events if event.get("ok") is True]
    e2e = [float(task["e2e_s"]) for task in successes]
    durations = [float(event["duration_s"]) for event in successful_events]
    summary = {
        "task_count": len(task_results),
        "successful_task_count": len(successes),
        "failed_task_count": len(task_results) - len(successes),
        "all_tasks_succeeded": len(successes) == len(task_results),
        "request_count": len(llm_events),
        "successful_request_count": len(successful_events),
        "failed_request_count": len(llm_events) - len(successful_events),
        "exactly_one_attempt_each": all(
            event.get("attempts") == 1 for event in llm_events
        ),
        "makespan_s": ended_wall_s - started_wall_s,
        "task_completion_makespan_s": tasks_ended_wall_s - started_wall_s,
        "task_e2e": {
            "mean_s": statistics.fmean(e2e) if e2e else None,
            "p50_s": _percentile(e2e, 0.50) if e2e else None,
            "p95_s": _percentile(e2e, 0.95) if e2e else None,
            "max_s": max(e2e) if e2e else None,
        },
        "llm": {
            "mean_request_s": statistics.fmean(durations) if durations else None,
            "p95_request_s": _percentile(durations, 0.95) if durations else None,
            "prompt_tokens": sum(
                int(event["usage"]["prompt_tokens"])
                for event in successful_events
            ),
            "completion_tokens": sum(
                int(event["usage"]["completion_tokens"])
                for event in successful_events
            ),
        },
    }
    result = {
        "schema": SCHEMA_VERSION,
        "version": 2,
        "config": {
            "fixture_version": FIXTURE_VERSION,
            "fixture_manifest_sha256": fixture_manifest_sha256,
            "output_constraint": OUTPUT_CONSTRAINT,
            "sentinel_contract": sentinel_contract,
            "cell_id": args.cell_id,
            "block_id": args.block_id,
            "order_index": args.order_index,
            "server_instance_id": args.server_instance_id,
            "fresh_server": True,
            "prefix_cache_enabled": args.prefix_cache_enabled,
            "scheduler_policy": "fcfs",
            "scheduler_environment": scheduler_environment,
            "native_pythonpath_isolated": True,
            "explicit_prefix_locality_enabled": False,
            "external_network_used": False,
            "external_tools_executed": False,
            "deterministic_local_fixture": True,
            "later_prompts_use_runtime_completion": False,
            "calls_per_task": CALL_COUNT,
            "source_count": len(sources),
            "replicas": args.replicas,
            "task_count": len(tasks),
            "max_active_tasks": args.max_active_tasks,
            "context_padding_tokens": args.context_padding_tokens,
            "visit_fixture_tokens": args.visit_fixture_tokens,
            "max_tokens_by_call": [
                args.max_tokens_call0,
                args.max_tokens_call1,
                args.max_tokens_call2,
            ],
            "max_model_len": args.max_model_len,
            "server_url": server_url,
            "model": args.model,
            "tokenizer_path": str(args.tokenizer.resolve()),
            "workload_path": str(workload),
            "workload_sha256": args.workload_sha256,
            "workload_split_id": workload_payload["split_id"],
            "workload_formal_eligible": workload_payload["formal_eligible"],
            "engine_environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "MODEL_ID",
                    "MODEL_REVISION",
                    "VLLM_PORT",
                    "VLLM_MAX_MODEL_LEN",
                    "VLLM_MAX_NUM_BATCHED_TOKENS",
                    "VLLM_MAX_NUM_SEQS",
                    "VLLM_ENABLE_PREFIX_CACHING",
                    "VLLM_USE_V1",
                    "VLLM_SCHED_POLICY",
                )
            },
        },
        "summary": summary,
        "tasks": task_results,
        "llm_events": llm_events,
        "vllm_metric_deltas": metric_deltas,
        "vllm_metric_presence": metric_presence,
        "queue_timeline_summary": _timeline_summary(queue_rows),
        "started_wall_s": started_wall_s,
        "started_monotonic_s": started_monotonic_s,
        "ended_wall_s": ended_wall_s,
    }
    timeline_path = output_dir / "queue_timeline.jsonl"
    metrics_before_path = output_dir / "metrics_before.prom"
    metrics_after_path = output_dir / "metrics_after.prom"
    _write_jsonl_atomic(timeline_path, queue_rows)
    _write_text_atomic(metrics_before_path, before_raw)
    _write_text_atomic(metrics_after_path, after_raw)
    result["raw_evidence"] = {
        "queue_timeline": {
            "path": str(timeline_path),
            "sha256": sha256_file(timeline_path),
            "sample_count": len(queue_rows),
        },
        "metrics_before": {
            "path": str(metrics_before_path),
            "sha256": sha256_file(metrics_before_path),
        },
        "metrics_after": {
            "path": str(metrics_after_path),
            "sha256": sha256_file(metrics_after_path),
        },
    }
    _write_json_atomic(output_dir / "result.json", result)
    print(json.dumps({"config": result["config"], "summary": summary}, indent=2))
    return 0 if summary["all_tasks_succeeded"] else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--workload-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--cell-id", choices=["P0", "P1"], required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--order-index", type=int, choices=[0, 1], required=True)
    parser.add_argument("--server-instance-id", required=True)
    parser.add_argument(
        "--prefix-cache-enabled", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--expected-source-count", type=int, required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--max-active-tasks", type=int, required=True)
    parser.add_argument("--context-padding-tokens", type=int, required=True)
    parser.add_argument("--visit-fixture-tokens", type=int, required=True)
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--output-constraint", required=True)
    parser.add_argument("--max-tokens-call0", type=int, required=True)
    parser.add_argument("--max-tokens-call1", type=int, required=True)
    parser.add_argument("--max-tokens-call2", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--request-timeout-s", type=float, required=True)
    parser.add_argument("--queue-sample-interval-s", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run_cell(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
