#!/usr/bin/env python3
"""
Utilities for preparing and replaying trace-driven vLLM experiments.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ``scripts/`` is the import root when this file is used by
# ``python scripts/run_vllm_trace_experiment.py``.  Add the repository root so
# the independently testable reproduction package can be consumed without
# installing the repository as a wheel.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reproduction.paste_repro.mapper import URLRankMapper, load_artifact
from reproduction.paste_repro.tool_prediction import TraceLearnedVisitPredictor


PREFIX_CHARS = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!$%&()*+,-./:;<=>?@[]^_{|}~"
)


def duplicate_variant_marker(duplicate_index: int) -> str:
    """Return a deterministic marker for a duplicated trace variant."""
    if duplicate_index < 0:
        raise ValueError("duplicate_index must be non-negative")
    if duplicate_index < len(PREFIX_CHARS):
        return PREFIX_CHARS[duplicate_index]
    return f"dup_variant_{duplicate_index:04d}"


@dataclass
class PreparedRequest:
    call_index: int
    wait_after_prev_s: float
    prompt_tokens: int
    target_output_tokens: int
    max_tokens: int
    truncated: bool
    original_prompt_tokens: int
    messages: List[Dict[str, str]]
    wait_after_prev_original_s: float = 0.0
    tool_overlap_saved_s: float = 0.0
    tool_overlap_window_s: float = 0.0
    tool_kind_before: str = ""
    tool_cache_hit: bool = False
    tool_overlap_mode: str = "none"
    # These fields are emitted only for learned-overlap workloads.  They are
    # removed from serialized none/native/oracle requests below to preserve
    # the legacy workload shape for those modes.
    tool_prediction_candidates: List[str] = field(default_factory=list)
    tool_prediction_candidate_count: int = 0
    tool_prediction_exact_hits: int = 0
    tool_prediction_waste: int = 0
    tool_prediction_artifact_sha256: str = ""
    tool_prediction_top_k: int = 0


@dataclass
class PreparedTrace:
    trace_id: str
    source_trace: str
    variant_index: int
    duplicated: bool
    prefix_char: str
    initial_delay_s: float
    truncated_calls: int
    requests: List[PreparedRequest]


def list_trace_files(trace_dir: str | Path) -> List[Path]:
    return sorted(Path(trace_dir).glob("*.jsonl"))


def load_trace_events(trace_file: str | Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with Path(trace_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _clone_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    cloned: List[Dict[str, str]] = []
    for msg in messages:
        cloned.append({"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))})
    return cloned


def _inject_unique_variant_marker(
    messages: Sequence[Dict[str, Any]],
    prefix_char: str,
    prefix_marker_mode: str = "preserve_prefix",
) -> List[Dict[str, str]]:
    cloned = _clone_messages(messages)
    if not prefix_char:
        return cloned

    normalized_mode = (prefix_marker_mode or "preserve_prefix").strip().lower()
    if normalized_mode in {"break_prefix", "front", "front_marker"}:
        marker = {"role": "system", "content": prefix_char}
        return [marker] + cloned

    # Keep the original leading system scaffold intact so duplicated traces can
    # still share the same prompt prefix. We only inject the marker after the
    # initial system-message prefix to avoid breaking prefix caching from token 0.
    insert_idx = 0
    while insert_idx < len(cloned) and cloned[insert_idx]["role"] == "system":
        insert_idx += 1

    if insert_idx > 0:
        cloned.insert(insert_idx, {"role": "system", "content": prefix_char})
        return cloned

    # Fallback for prompts without a leading system message: mutate the first
    # message at the end so the shared beginning stays identical.
    if cloned:
        cloned[0]["content"] = f'{cloned[0]["content"]}\n\n{prefix_char}'
        return cloned

    return [{"role": "system", "content": prefix_char}]


def _build_chat_tokens(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def _build_output_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_messages_to_fit(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    max_prompt_tokens: int,
) -> Tuple[List[Dict[str, str]], int, bool]:
    """
    Left-truncate oldest context when prompt is over the model budget.
    Keeps the earliest system messages, then trims/removes the oldest non-system
    messages until the prompt fits.
    """
    trimmed = _clone_messages(messages)
    prompt_tokens = _build_chat_tokens(tokenizer, trimmed)
    if prompt_tokens <= max_prompt_tokens:
        return trimmed, prompt_tokens, False

    # Prefer removing oldest non-system messages.
    while prompt_tokens > max_prompt_tokens:
        removable_idx = None
        for idx, msg in enumerate(trimmed):
            if msg["role"] != "system":
                removable_idx = idx
                break
        if removable_idx is None:
            break
        del trimmed[removable_idx]
        prompt_tokens = _build_chat_tokens(tokenizer, trimmed)

    if prompt_tokens <= max_prompt_tokens:
        return trimmed, prompt_tokens, True

    # Fallback: trim the oldest long message content from the left.
    for idx, msg in enumerate(trimmed):
        if msg["role"] == "system":
            continue
        content = msg["content"]
        if not content:
            continue
        left = 0
        right = len(content)
        best = content
        while left < right:
            mid = (left + right) // 2
            candidate = "[TRUNCATED]\n" + content[mid:]
            test_messages = _clone_messages(trimmed)
            test_messages[idx]["content"] = candidate
            test_tokens = _build_chat_tokens(tokenizer, test_messages)
            if test_tokens <= max_prompt_tokens:
                best = candidate
                right = mid
            else:
                left = mid + 1
        trimmed[idx]["content"] = best
        prompt_tokens = _build_chat_tokens(tokenizer, trimmed)
        if prompt_tokens <= max_prompt_tokens:
            return trimmed, prompt_tokens, True

    # Last resort: trim the most recent message content.
    last_idx = len(trimmed) - 1
    while prompt_tokens > max_prompt_tokens and last_idx >= 0:
        current = trimmed[last_idx]["content"]
        if len(current) <= 64:
            last_idx -= 1
            continue
        trimmed[last_idx]["content"] = current[-max(64, len(current) // 2) :]
        prompt_tokens = _build_chat_tokens(tokenizer, trimmed)

    return trimmed, prompt_tokens, True


def _request_start_s(llm_event: Dict[str, Any]) -> float:
    return max(0.0, float(llm_event["timestamp"]) - float(llm_event.get("total_time_ms", 0.0)) / 1000.0)


def _request_end_s(llm_event: Dict[str, Any]) -> float:
    return float(llm_event["timestamp"])


def _tool_call_between(events: Sequence[Dict[str, Any]], prev_end_s: float, next_end_s: float) -> Optional[Dict[str, Any]]:
    for event in events:
        if event.get("event_type") != "tool_call":
            continue
        ts = float(event.get("timestamp", 0.0))
        if prev_end_s <= ts <= next_end_s:
            return event
    return None


def _llm_overlap_window_s(llm_event: Dict[str, Any]) -> float:
    inference_s = max(0.0, float(llm_event.get("inference_time_ms", 0.0)) / 1000.0)
    if inference_s > 0:
        return inference_s
    return max(0.0, float(llm_event.get("total_time_ms", 0.0)) / 1000.0)


def _canonical_tool_key(tool_event: Dict[str, Any]) -> str:
    payload = {
        "tool": tool_event.get("tool_name", ""),
        "args": tool_event.get("tool_args", {}),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _visit_urls(tool_event: Dict[str, Any]) -> list[str]:
    if tool_event.get("tool_name") != "visit":
        return []
    value = tool_event.get("tool_args", {}).get("url")
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _native_can_overlap_tool(
    tool_event: Optional[Dict[str, Any]],
    previous_tool_kind: str,
) -> bool:
    if tool_event is None:
        return False
    return tool_event.get("tool_name") == "visit" and previous_tool_kind == "search"


def _latest_visible_tool_response(llm_event: Dict[str, Any]) -> str:
    """Return only tool data already present in this request's messages.

    In particular, this deliberately does not inspect the recorded LLM response,
    later events, token counts, or timing fields.  It is the causal input that
    would be available when the decision LLM starts generating.
    """

    messages = llm_event.get("messages", [])
    if not isinstance(messages, Sequence):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str) and "<tool_response>" in content:
            return content
    return ""


def _learned_visit_candidates(
    mapper: URLRankMapper,
    decision_llm: Dict[str, Any],
    top_k: int,
) -> list[str]:
    """Late-bind learned ranks to concrete, currently visible result URLs."""

    predictor = TraceLearnedVisitPredictor(mapper=mapper, top_k=top_k)
    return list(
        predictor.predict_visible_response(_latest_visible_tool_response(decision_llm))
    )


def _prediction_wait_meta(
    *,
    mode: str,
    artifact_sha256: str,
    top_k: int,
    candidates: Sequence[str] = (),
    authoritative_urls: Sequence[str] = (),
) -> Dict[str, Any]:
    """Describe one resolved speculative batch using exact URL equality."""

    if mode != "learned":
        return {
            "tool_prediction_candidates": [],
            "tool_prediction_candidate_count": 0,
            "tool_prediction_exact_hits": 0,
            "tool_prediction_waste": 0,
            "tool_prediction_artifact_sha256": "",
            "tool_prediction_top_k": 0,
        }

    concrete_candidates = list(candidates)
    candidate_set = set(concrete_candidates)
    authoritative_set = set(authoritative_urls)
    exact_hits = sum(url in candidate_set for url in authoritative_urls)
    consumed_candidates = len(candidate_set.intersection(authoritative_set))
    return {
        "tool_prediction_candidates": concrete_candidates,
        "tool_prediction_candidate_count": len(concrete_candidates),
        "tool_prediction_exact_hits": exact_hits,
        "tool_prediction_waste": max(0, len(concrete_candidates) - consumed_candidates),
        "tool_prediction_artifact_sha256": artifact_sha256,
        "tool_prediction_top_k": top_k,
    }


def _compute_wait_after_prev(
    all_events: Sequence[Dict[str, Any]],
    prev_llm: Optional[Dict[str, Any]],
    current_llm: Dict[str, Any],
    tool_jitter_ratio: float,
    initial_delay_s: float,
    previous_tool_kind: str,
    seen_tool_keys: set[str],
    seen_visit_urls: set[str],
    tool_overlap_mode: str,
    tool_overlap_efficiency: float,
    learned_mapper: Optional[URLRankMapper] = None,
    learned_artifact_sha256: str = "",
    learned_top_k: int = 0,
    pending_prediction_urls: Optional[list[str]] = None,
) -> Tuple[float, Dict[str, Any], str]:
    normalized_mode = (tool_overlap_mode or "none").strip().lower()
    prediction_meta = _prediction_wait_meta(
        mode=normalized_mode,
        artifact_sha256=learned_artifact_sha256,
        top_k=learned_top_k,
    )

    if prev_llm is None:
        wait_s = max(0.0, initial_delay_s)
        return wait_s, {
            "wait_after_prev_original_s": wait_s,
            "tool_overlap_saved_s": 0.0,
            "tool_overlap_window_s": 0.0,
            "tool_kind_before": "",
            "tool_cache_hit": False,
            "tool_overlap_mode": normalized_mode,
            **prediction_meta,
        }, ""

    prev_end_s = _request_end_s(prev_llm)
    curr_start_s = _request_start_s(current_llm)
    if curr_start_s <= prev_end_s:
        if normalized_mode == "learned" and pending_prediction_urls:
            prediction_meta = _prediction_wait_meta(
                mode=normalized_mode,
                artifact_sha256=learned_artifact_sha256,
                top_k=learned_top_k,
                candidates=pending_prediction_urls,
            )
            pending_prediction_urls.clear()
        return 0.0, {
            "wait_after_prev_original_s": 0.0,
            "tool_overlap_saved_s": 0.0,
            "tool_overlap_window_s": 0.0,
            "tool_kind_before": "",
            "tool_cache_hit": False,
            "tool_overlap_mode": normalized_mode,
            **prediction_meta,
        }, ""

    tool_event = _tool_call_between(all_events, prev_end_s, _request_end_s(current_llm))
    if tool_event is None:
        if normalized_mode == "learned" and pending_prediction_urls:
            prediction_meta = _prediction_wait_meta(
                mode=normalized_mode,
                artifact_sha256=learned_artifact_sha256,
                top_k=learned_top_k,
                candidates=pending_prediction_urls,
            )
            pending_prediction_urls.clear()
        wait_s = max(0.0, curr_start_s - prev_end_s)
        return wait_s, {
            "wait_after_prev_original_s": wait_s,
            "tool_overlap_saved_s": 0.0,
            "tool_overlap_window_s": 0.0,
            "tool_kind_before": "",
            "tool_cache_hit": False,
            "tool_overlap_mode": normalized_mode,
            **prediction_meta,
        }, ""

    tool_kind = str(tool_event.get("tool_name", ""))
    tool_start_s = float(tool_event.get("timestamp", prev_end_s))
    fixed_overhead_s = max(0.0, tool_start_s - prev_end_s)
    tool_exec_s = max(0.0, curr_start_s - tool_start_s) * tool_jitter_ratio
    original_wait_s = fixed_overhead_s + tool_exec_s
    overlap_window_s = _llm_overlap_window_s(prev_llm) * max(0.0, tool_overlap_efficiency)

    tool_key = _canonical_tool_key(tool_event)
    visit_urls = _visit_urls(tool_event)
    cached_visit_urls = sum(1 for url in visit_urls if url in seen_visit_urls)
    cache_hit = tool_key in seen_tool_keys or cached_visit_urls > 0
    saved_s = 0.0

    if normalized_mode == "native":
        if tool_key in seen_tool_keys:
            saved_s = tool_exec_s
        elif visit_urls and cached_visit_urls > 0:
            saved_s = tool_exec_s * (cached_visit_urls / len(visit_urls))
        if _native_can_overlap_tool(tool_event, previous_tool_kind):
            saved_s += min(max(0.0, tool_exec_s - saved_s), overlap_window_s)
    elif normalized_mode == "oracle":
        saved_s = min(tool_exec_s, overlap_window_s)
    elif normalized_mode == "learned":
        if learned_mapper is None or pending_prediction_urls is None:
            raise ValueError(
                "learned tool overlap requires a loaded prediction artifact"
            )
        resolved_candidates = (
            tuple(pending_prediction_urls) if previous_tool_kind == "search" else ()
        )
        prediction_meta = _prediction_wait_meta(
            mode=normalized_mode,
            artifact_sha256=learned_artifact_sha256,
            top_k=learned_top_k,
            candidates=resolved_candidates,
            authoritative_urls=visit_urls if tool_kind == "visit" else (),
        )
        if tool_kind == "visit" and visit_urls:
            exact_hit_fraction = (
                prediction_meta["tool_prediction_exact_hits"] / len(visit_urls)
            )
            saved_s = min(tool_exec_s, overlap_window_s) * exact_hit_fraction

        # A prediction batch is valid only for the immediately following tool
        # decision.  It is discarded on a miss/non-visit, and a new search uses
        # only the response visible in ``current_llm`` to create its next batch.
        pending_prediction_urls.clear()
        if tool_kind == "search":
            pending_prediction_urls.extend(
                _learned_visit_candidates(learned_mapper, current_llm, learned_top_k)
            )

    wait_s = max(0.0, fixed_overhead_s + tool_exec_s - saved_s)
    seen_tool_keys.add(tool_key)
    seen_visit_urls.update(visit_urls)
    return wait_s, {
        "wait_after_prev_original_s": original_wait_s,
        "tool_overlap_saved_s": saved_s,
        "tool_overlap_window_s": overlap_window_s,
        "tool_kind_before": tool_kind,
        "tool_cache_hit": cache_hit,
        "tool_overlap_mode": normalized_mode,
        **prediction_meta,
    }, tool_kind


def _wait_after_prev_s(
    all_events: Sequence[Dict[str, Any]],
    prev_llm: Optional[Dict[str, Any]],
    current_llm: Dict[str, Any],
    tool_jitter_ratio: float,
    initial_delay_s: float,
) -> float:
    if prev_llm is None:
        return max(0.0, initial_delay_s)

    prev_end_s = _request_end_s(prev_llm)
    curr_start_s = _request_start_s(current_llm)
    if curr_start_s <= prev_end_s:
        return 0.0

    tool_event = _tool_call_between(all_events, prev_end_s, _request_end_s(current_llm))
    if tool_event is None:
        return max(0.0, curr_start_s - prev_end_s)

    tool_start_s = float(tool_event.get("timestamp", prev_end_s))
    fixed_overhead_s = max(0.0, tool_start_s - prev_end_s)
    tool_exec_s = max(0.0, curr_start_s - tool_start_s)
    return fixed_overhead_s + tool_exec_s * tool_jitter_ratio


def _pick_duplicate_sources(trace_files: Sequence[Path], target_trace_count: int, seed: int) -> List[Path]:
    if target_trace_count <= len(trace_files):
        return []
    rng = random.Random(seed)
    duplicates_needed = target_trace_count - len(trace_files)
    shuffled = list(trace_files)
    rng.shuffle(shuffled)
    chosen: List[Path] = []
    while len(chosen) < duplicates_needed:
        chosen.extend(shuffled)
    return chosen[:duplicates_needed]


def prepare_trace_workload(
    trace_dir: str | Path,
    tokenizer: Any,
    target_trace_count: int,
    max_model_len: int,
    max_output_tokens_cap: int,
    min_output_tokens_floor: int,
    output_token_buffer: int,
    duplicate_seed: int = 20260417,
    tool_overlap_mode: str = "none",
    tool_overlap_efficiency: float = 1.0,
    prefix_marker_mode: str = "preserve_prefix",
    tool_prediction_model: str | Path | None = None,
    tool_prediction_top_k: int = 5,
) -> Dict[str, Any]:
    normalized_overlap_mode = (tool_overlap_mode or "none").strip().lower()
    if normalized_overlap_mode not in {"none", "native", "oracle", "learned"}:
        raise ValueError(f"unsupported tool overlap mode: {tool_overlap_mode!r}")

    learned_mapper: Optional[URLRankMapper] = None
    learned_artifact_sha256 = ""
    if normalized_overlap_mode == "learned":
        if tool_prediction_model is None:
            raise ValueError(
                "learned tool overlap requires --tool-prediction-model"
            )
        if tool_prediction_top_k <= 0:
            raise ValueError("tool_prediction_top_k must be positive")
        learned_mapper, artifact = load_artifact(tool_prediction_model)
        learned_artifact_sha256 = str(artifact["artifact_sha256"])

    trace_files = list_trace_files(trace_dir)
    if not trace_files:
        raise FileNotFoundError(f"No trace files found under {trace_dir}")
    if target_trace_count < 1:
        raise ValueError("target_trace_count must be positive")
    rng = random.Random(duplicate_seed)
    duplicate_sources = _pick_duplicate_sources(trace_files, target_trace_count, duplicate_seed)
    all_specs: List[Tuple[Path, bool, str, int]] = []

    for idx, trace_file in enumerate(trace_files):
        all_specs.append((trace_file, False, "", idx))

    for dup_idx, source in enumerate(duplicate_sources):
        all_specs.append((source, True, duplicate_variant_marker(dup_idx), len(trace_files) + dup_idx))

    prepared_traces: List[PreparedTrace] = []
    total_truncated_calls = 0
    max_prompt_budget = max_model_len - min_output_tokens_floor

    for source_path, duplicated, prefix_char, variant_index in all_specs:
        all_events = load_trace_events(source_path)
        llm_events = [e for e in all_events if e.get("event_type") == "llm_call"]
        llm_events.sort(key=lambda x: int(x.get("call_index", 0)))

        initial_delay_s = rng.uniform(0.0, 2.0) if duplicated else 0.0
        requests: List[PreparedRequest] = []
        prev_llm: Optional[Dict[str, Any]] = None
        previous_tool_kind = ""
        seen_tool_keys: set[str] = set()
        seen_visit_urls: set[str] = set()
        pending_prediction_urls: list[str] = []
        truncated_calls = 0

        for llm_event in llm_events:
            messages = _inject_unique_variant_marker(
                llm_event.get("messages", []),
                prefix_char,
                prefix_marker_mode=prefix_marker_mode,
            )
            original_prompt_tokens = _build_chat_tokens(tokenizer, messages)
            trimmed_messages, prompt_tokens, truncated = _truncate_messages_to_fit(
                tokenizer=tokenizer,
                messages=messages,
                max_prompt_tokens=max_prompt_budget,
            )
            if truncated:
                truncated_calls += 1

            available_output_tokens = max(1, max_model_len - prompt_tokens)
            response_text = str(llm_event.get("response", ""))
            target_output_tokens = max(1, _build_output_tokens(tokenizer, response_text))
            max_tokens = min(
                max_output_tokens_cap,
                available_output_tokens,
                max(min_output_tokens_floor, target_output_tokens + output_token_buffer),
            )

            tool_jitter_ratio = rng.uniform(0.85, 1.15) if duplicated else 1.0
            wait_s, wait_meta, previous_tool_kind = _compute_wait_after_prev(
                all_events=all_events,
                prev_llm=prev_llm,
                current_llm=llm_event,
                tool_jitter_ratio=tool_jitter_ratio,
                initial_delay_s=initial_delay_s,
                previous_tool_kind=previous_tool_kind,
                seen_tool_keys=seen_tool_keys,
                seen_visit_urls=seen_visit_urls,
                tool_overlap_mode=normalized_overlap_mode,
                tool_overlap_efficiency=tool_overlap_efficiency,
                learned_mapper=learned_mapper,
                learned_artifact_sha256=learned_artifact_sha256,
                learned_top_k=tool_prediction_top_k,
                pending_prediction_urls=pending_prediction_urls,
            )
            requests.append(
                PreparedRequest(
                    call_index=int(llm_event.get("call_index", 0)),
                    wait_after_prev_s=wait_s,
                    prompt_tokens=prompt_tokens,
                    target_output_tokens=target_output_tokens,
                    max_tokens=max_tokens,
                    truncated=truncated,
                    original_prompt_tokens=original_prompt_tokens,
                    messages=trimmed_messages,
                    wait_after_prev_original_s=wait_meta["wait_after_prev_original_s"],
                    tool_overlap_saved_s=wait_meta["tool_overlap_saved_s"],
                    tool_overlap_window_s=wait_meta["tool_overlap_window_s"],
                    tool_kind_before=wait_meta["tool_kind_before"],
                    tool_cache_hit=wait_meta["tool_cache_hit"],
                    tool_overlap_mode=wait_meta["tool_overlap_mode"],
                    tool_prediction_candidates=wait_meta[
                        "tool_prediction_candidates"
                    ],
                    tool_prediction_candidate_count=wait_meta[
                        "tool_prediction_candidate_count"
                    ],
                    tool_prediction_exact_hits=wait_meta[
                        "tool_prediction_exact_hits"
                    ],
                    tool_prediction_waste=wait_meta["tool_prediction_waste"],
                    tool_prediction_artifact_sha256=wait_meta[
                        "tool_prediction_artifact_sha256"
                    ],
                    tool_prediction_top_k=wait_meta["tool_prediction_top_k"],
                )
            )
            prev_llm = llm_event

        # If a trace ends immediately after a search result was consumed by the
        # final decision LLM, its unconfirmed speculative calls are all waste.
        if normalized_overlap_mode == "learned" and pending_prediction_urls and requests:
            terminal_request = requests[-1]
            terminal_request.tool_prediction_candidates.extend(
                pending_prediction_urls
            )
            terminal_request.tool_prediction_candidate_count += len(
                pending_prediction_urls
            )
            terminal_request.tool_prediction_waste += len(pending_prediction_urls)
            pending_prediction_urls.clear()

        total_truncated_calls += truncated_calls
        prepared_traces.append(
            PreparedTrace(
                trace_id=f"trace_{variant_index:03d}",
                source_trace=str(source_path),
                variant_index=variant_index,
                duplicated=duplicated,
                prefix_char=prefix_char,
                initial_delay_s=initial_delay_s,
                truncated_calls=truncated_calls,
                requests=requests,
            )
        )

    serialized_traces = [
        asdict(trace) for trace in prepared_traces[:target_trace_count]
    ]
    if normalized_overlap_mode != "learned":
        # Preserve the legacy request schema byte-for-byte for existing modes.
        learned_only_fields = {
            "tool_prediction_candidates",
            "tool_prediction_candidate_count",
            "tool_prediction_exact_hits",
            "tool_prediction_waste",
            "tool_prediction_artifact_sha256",
            "tool_prediction_top_k",
        }
        for trace in serialized_traces:
            for request in trace["requests"]:
                for field_name in learned_only_fields:
                    request.pop(field_name, None)

    workload = {
        "meta": {
            "source_trace_dir": str(trace_dir),
            "target_trace_count": target_trace_count,
            "max_model_len": max_model_len,
            "max_output_tokens_cap": max_output_tokens_cap,
            "min_output_tokens_floor": min_output_tokens_floor,
            "output_token_buffer": output_token_buffer,
            "duplicate_seed": duplicate_seed,
            "duplicates_added": len(duplicate_sources),
            "total_truncated_calls": total_truncated_calls,
            "tool_overlap_mode": normalized_overlap_mode,
            "tool_overlap_efficiency": tool_overlap_efficiency,
            "prefix_marker_mode": prefix_marker_mode,
            **(
                {
                    "tool_prediction_artifact_sha256": learned_artifact_sha256,
                    "tool_prediction_top_k": tool_prediction_top_k,
                }
                if normalized_overlap_mode == "learned"
                else {}
            ),
        },
        "traces": serialized_traces,
    }
    return workload


def save_workload(workload: Dict[str, Any], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(workload, f, ensure_ascii=False, indent=2)


def load_workload(workload_file: str | Path) -> Dict[str, Any]:
    with Path(workload_file).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_learned_workload_artifact(
    workload: Dict[str, Any],
    tool_prediction_model: str | Path | None,
) -> None:
    """Fail closed unless a prepared learned workload matches its artifact."""

    if tool_prediction_model is None:
        raise ValueError("learned tool overlap requires --tool-prediction-model")
    _, artifact = load_artifact(tool_prediction_model)
    metadata = workload.get("meta", {})
    if metadata.get("tool_overlap_mode") != "learned":
        raise ValueError(
            "--tool-overlap-mode learned cannot reuse a non-learned prepared workload"
        )
    expected_checksum = artifact["artifact_sha256"]
    if metadata.get("tool_prediction_artifact_sha256") != expected_checksum:
        raise ValueError(
            "prepared workload tool prediction artifact checksum mismatch"
        )
    try:
        top_k = int(metadata.get("tool_prediction_top_k", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared learned workload has invalid top_k") from exc
    if top_k <= 0:
        raise ValueError("prepared learned workload has invalid top_k")


def cap_workload_by_arrival_time(
    workload: Dict[str, Any],
    max_arrival_time_s: float,
) -> Dict[str, Any]:
    if max_arrival_time_s <= 0:
        raise ValueError("max_arrival_time_s must be positive")

    capped = copy.deepcopy(workload)
    dropped_request_count = 0
    existing_dropped_request_count = int(capped.get("meta", {}).get("dropped_late_requests", 0))

    for trace in capped["traces"]:
        kept_requests: List[Dict[str, Any]] = []
        arrival_time_s = float(trace.get("initial_delay_s", 0.0))
        for idx, request in enumerate(trace["requests"]):
            if idx == 0:
                arrival_time_s = float(trace.get("initial_delay_s", 0.0))
            else:
                arrival_time_s += float(request.get("wait_after_prev_s", 0.0))

            if arrival_time_s <= max_arrival_time_s:
                kept_requests.append(request)
            else:
                dropped_request_count += 1

        trace["requests"] = kept_requests
        trace["truncated_calls"] = sum(1 for request in kept_requests if request.get("truncated"))

    capped.setdefault("meta", {})
    capped["meta"]["arrival_cap_s"] = max_arrival_time_s
    capped["meta"]["dropped_late_requests"] = existing_dropped_request_count + dropped_request_count
    return capped


def iter_requests(workload: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for trace in workload["traces"]:
        for request in trace["requests"]:
            yield trace, request


def summarize_workload(workload: Dict[str, Any]) -> Dict[str, Any]:
    prompt_tokens = [request["prompt_tokens"] for _, request in iter_requests(workload)]
    target_output_tokens = [
        request.get("target_output_tokens", request["max_tokens"])
        for _, request in iter_requests(workload)
    ]
    max_tokens = [request["max_tokens"] for _, request in iter_requests(workload)]
    waits = [request["wait_after_prev_s"] for _, request in iter_requests(workload)]
    original_waits = [
        request.get("wait_after_prev_original_s", request["wait_after_prev_s"])
        for _, request in iter_requests(workload)
    ]
    overlap_savings = [
        request.get("tool_overlap_saved_s", 0.0)
        for _, request in iter_requests(workload)
    ]
    prediction_candidate_count = sum(
        int(request.get("tool_prediction_candidate_count", 0))
        for _, request in iter_requests(workload)
    )
    prediction_exact_hits = sum(
        int(request.get("tool_prediction_exact_hits", 0))
        for _, request in iter_requests(workload)
    )
    prediction_waste = sum(
        int(request.get("tool_prediction_waste", 0))
        for _, request in iter_requests(workload)
    )
    truncated_calls = sum(1 for _, request in iter_requests(workload) if request["truncated"])
    duplicated_count = sum(1 for trace in workload["traces"] if trace["duplicated"])
    dropped_late_request_count = int(workload.get("meta", {}).get("dropped_late_requests", 0))
    arrival_cap_s = workload.get("meta", {}).get("arrival_cap_s")

    def _percentile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(math.floor((len(ordered) - 1) * q)))
        return float(ordered[idx])

    summary = {
        "trace_count": len(workload["traces"]),
        "request_count": len(prompt_tokens),
        "duplicated_trace_count": duplicated_count,
        "truncated_call_count": truncated_calls,
        "dropped_late_request_count": dropped_late_request_count,
        "arrival_cap_s": arrival_cap_s,
        "prompt_tokens_p50": _percentile(prompt_tokens, 0.50),
        "prompt_tokens_p95": _percentile(prompt_tokens, 0.95),
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else 0,
        "target_output_tokens_p50": _percentile(target_output_tokens, 0.50),
        "target_output_tokens_p95": _percentile(target_output_tokens, 0.95),
        "target_output_tokens_max": max(target_output_tokens) if target_output_tokens else 0,
        "max_tokens_p50": _percentile(max_tokens, 0.50),
        "max_tokens_p95": _percentile(max_tokens, 0.95),
        "wait_after_prev_s_p50": _percentile(waits, 0.50),
        "wait_after_prev_s_p95": _percentile(waits, 0.95),
        "wait_after_prev_original_s_p50": _percentile(original_waits, 0.50),
        "wait_after_prev_original_s_p95": _percentile(original_waits, 0.95),
        "tool_overlap_saved_total_s": sum(float(value) for value in overlap_savings),
        "tool_overlap_saved_avg_per_request_s": (
            sum(float(value) for value in overlap_savings) / len(overlap_savings)
            if overlap_savings else 0.0
        ),
        "tool_overlap_mode": workload.get("meta", {}).get("tool_overlap_mode", "none"),
    }
    arrival_process = workload.get("meta", {}).get("arrival_process")
    if arrival_process:
        summary["arrival_process"] = copy.deepcopy(arrival_process)
    if workload.get("meta", {}).get("tool_overlap_mode") == "learned":
        summary["tool_prediction"] = {
            "candidate_count": prediction_candidate_count,
            "exact_hits": prediction_exact_hits,
            "waste": prediction_waste,
            "artifact_sha256": workload.get("meta", {}).get(
                "tool_prediction_artifact_sha256", ""
            ),
            "top_k": int(
                workload.get("meta", {}).get("tool_prediction_top_k", 0)
            ),
        }
    return summary
