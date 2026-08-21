from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import aggregate_live_joint_four_cell as aggregate_module  # noqa: E402
from aggregate_live_joint_four_cell import (  # noqa: E402
    BOOTSTRAP_SEED,
    _fixed_final_grammar_sha256,
    aggregate_live_joint_four_cell,
    parse_args,
)
from validate_live_joint_formal_workload import (  # noqa: E402
    FORMAL_V3_WORKLOAD,
    FORMAL_V4_WORKLOAD,
    FORMAL_V5_WORKLOAD,
    FORMAL_V6_WORKLOAD,
    FORMAL_V7_WORKLOAD,
    FORMAL_V8_WORKLOAD,
    FORMAL_V9_WORKLOAD,
    validate_formal_workload,
)


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _invocation_sha(tool: str, arguments: dict) -> str:
    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{tool}\0{encoded}".encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _formal_sources(workload_path: Path = FORMAL_V3_WORKLOAD) -> list[dict]:
    return json.loads(workload_path.read_text(encoding="utf-8"))["sources"]


def _make_run(
    root: Path,
    *,
    block_id: str,
    cell: str,
    order_index: int,
    e2e_base: float,
    workload_path: Path = FORMAL_V3_WORKLOAD,
) -> Path:
    formal = validate_formal_workload(workload_path)
    sources = _formal_sources(workload_path)
    is_v5 = formal["split_id"] == "live-joint-wikipedia-frozen-formal-v5"
    is_v6 = formal["split_id"] == "live-joint-wikipedia-frozen-formal-v6"
    is_v7 = formal["split_id"] == "live-joint-wikipedia-frozen-formal-v7"
    is_v8 = formal["split_id"] == "live-joint-wikipedia-frozen-formal-v8"
    is_v9 = formal["split_id"] == "live-joint-wikipedia-frozen-formal-v9"
    is_fixed_final = is_v8 or is_v9
    is_load80 = is_v8 or is_v9
    is_modern = formal["split_id"] in {
        "live-joint-wikipedia-frozen-formal-v4",
        "live-joint-wikipedia-frozen-formal-v5",
        "live-joint-wikipedia-frozen-formal-v6",
        "live-joint-wikipedia-frozen-formal-v7",
        "live-joint-wikipedia-frozen-formal-v8",
        "live-joint-wikipedia-frozen-formal-v9",
    }
    source_count = len(sources)
    context_padding = 10_000 if is_modern else 5_600
    actual_padding = context_padding + 12
    llm_prompt_tokens = 11_000 if is_modern else 6_000
    visit_capacity = 2 if is_modern else 1
    search_capacity = 3 if is_modern else 0
    canary_stride = 6 if is_modern else 10
    visit_interval = 2.5 if is_v9 else 2.1
    run_dir = root / block_id / cell
    evidence_dir = run_dir / "evidence" if is_v9 else run_dir
    timeline_path = evidence_dir / "queue_timeline.jsonl"
    speculation_mode = (
        "off"
        if cell in {"A", "E"}
        else ("visit" if is_modern else "search_visit")
    )
    scheduler = {
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "MODEL_ID": "test/model",
        "VLLM_MAX_NUM_SEQS": "96",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "2048" if is_modern else "8192",
        "VLLM_MAX_MODEL_LEN": "16384" if is_modern else None,
        "VLLM_ENABLE_PREFIX_CACHING": "1" if is_modern else None,
        "VLLM_SCHED_POLICY": (
            "fcfs" if cell in {"A", "B"} else "online_joint_pacer_v2"
        ),
        "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY": (
            None if cell in {"A", "B"} else ("0" if is_modern else "1")
        ),
    }
    timeline = []
    for index in range(20):
        pressured = index < (10 if is_load80 else 2)
        timeline.append(
            {
                "wall_s": 1000.0 + index * 0.2,
                "monotonic_s": 10.0 + index * 0.2,
                "broker_revision": index,
                "tool_queued_authoritative": 1 if pressured else 0,
                "tool_queued_speculative": (
                    1 if pressured and speculation_mode != "off" else 0
                ),
                "tool_running_authoritative": 1,
                "tool_running_speculative": (
                    1 if speculation_mode != "off" else 0
                ),
                "tool_completed_unclaimed_speculative": 0,
                "llm_running": 70.0 if is_load80 else 48.0,
                "llm_waiting": 2.0 if pressured else 0.0,
                "gpu_cache_usage": 0.4,
            }
        )
    _write_jsonl(timeline_path, timeline)

    config = {
        "cell_label": f"formal-{block_id}-{cell}",
        "server_url": "http://127.0.0.1:8100",
        "model": "test/model",
        "call_graph_mode": "frozen",
        "expected_url_search_coverage": {
            "eligible_task_count": source_count,
            "observed_task_count": source_count,
            "matched_task_count": source_count,
            "fraction_of_eligible": 1.0,
            "fraction_of_observed": 1.0,
        },
        "speculation_mode": speculation_mode,
        "visit_top_k": 1,
        "independent_source_count": source_count,
        "replicas": 1,
        "task_count": source_count,
        "max_active_tasks": source_count,
        "tool_workers": 4,
        "speculative_tool_workers": 2,
        "search_tool_capacity": search_capacity,
        "visit_tool_capacity": visit_capacity,
        "search_min_start_interval_s": 0.0,
        "visit_min_start_interval_s": visit_interval,
        "max_speculative_pending": 128,
        "speculative_ttl_s": 120.0,
        "tool_http_max_attempts": 2,
        "tool_http_retry_backoff_s": 1.0,
        "controlled_http_retry": True,
        "tool_http_retry_policy_version": "idempotent-get-v1",
        "tool_http_retryable_statuses": [429, 500, 502, 503, 504],
        "tool_http_retryable_exception_types": [
            "asyncio.TimeoutError",
            "ConnectionError",
            "aiohttp.ClientConnectionError",
            "aiohttp.ClientPayloadError",
        ],
        "tool_http_library_retry_disabled": True,
        "tool_http_library_retry_control_version": (
            "aiohttp-private-retry-connection-v1"
        ),
        "tool_http_library_name": "aiohttp",
        "tool_http_library_version": "3.12.15",
        "visit_mode": "jina",
        "search_mode": "bing",
        "search_max_results": 5,
        "visit_max_chars": 3000,
        "max_tokens_tool": 128,
        "max_tokens_answer": 256 if is_fixed_final else 160,
        "visit_canary_stride": canary_stride,
        "context_padding_tokens": context_padding,
        "queue_sample_interval_s": 0.2,
        "token_count_method": (
            "transformers_chat_template" if is_fixed_final else "test"
        ),
        "live_tool_execution": True,
        "recorded_tool_sleep": False,
        "shared_bounded_tool_pool": True,
        "generated_tool_call_controls_next_prompt": True,
        "authoritative_and_speculative_share_capacity": True,
        "workload_path": str(workload_path),
        "workload_file_sha256": formal["file_sha256"],
        "selected_workload_sha256": formal["canonical_sources_sha256"],
        "scheduler_environment": scheduler,
        "tool_metadata_is_causal": True,
        "tool_result_private_until_exact_commit": True,
        "future_trace_oracle_used": False,
        "frozen_url_is_workload_input": True,
        "workload_split_id": formal["split_id"],
        "workload_split_role": "formal_heldout",
        "workload_formal_eligible": True,
        "formal_run": {
            "block_id": block_id,
            "cell_id": cell,
            "order_index": order_index,
            "server_instance_id": f"server-{block_id}-{cell}",
            "fresh_server": True,
            "result_cache_empty": True,
            "broker_drained": True,
        },
    }
    if is_modern:
        live_agent_sha = {
            "live-joint-wikipedia-frozen-formal-v4": (
                "d523800ff6caa06e5727b28294b2041b7e44f4856b5ebb67e159057709d66be3"
            ),
            "live-joint-wikipedia-frozen-formal-v5": (
                "678864a738084076bb21a181cf15baa24c5839599fc5547303b269bb9e8c8455"
            ),
            "live-joint-wikipedia-frozen-formal-v6": (
                "719b34c36b5bf4f30d2a6bd4c47e37fe23fdea66a6ad7a5ea8128bdfbb50c28f"
            ),
            "live-joint-wikipedia-frozen-formal-v7": (
                "6fa736aa4e56657874834841c8a60b18c53e31f48ffbe741cc2e93f1c750432f"
            ),
            "live-joint-wikipedia-frozen-formal-v8": hashlib.sha256(
                (
                    REPOSITORY_ROOT
                    / "reproduction"
                    / "paste_repro"
                    / "live_agent.py"
                ).read_bytes()
            ).hexdigest(),
            "live-joint-wikipedia-frozen-formal-v9": hashlib.sha256(
                (
                    REPOSITORY_ROOT
                    / "reproduction"
                    / "paste_repro"
                    / "live_agent.py"
                ).read_bytes()
            ).hexdigest(),
        }[formal["split_id"]]
        config.update(
            {
                "tool_signal_policy": "execution_aware",
                "tool_signal_policy_version": (
                    "exact-session-invocation-running-completed-v1"
                ),
                "tool_signal_policy_module_sha256": live_agent_sha,
                "min_speculative_tool_workers": 0,
                "tool_http_attempt_start_gate_enabled": True,
                "tool_http_attempt_start_gate_policy_version": (
                    "shared-per-tool-monotonic-v1"
                ),
                "tool_http_attempt_min_start_intervals_s": {
                    "visit": visit_interval
                },
            }
        )
        if is_fixed_final:
            config.update(
                {
                    "fixed_final_completion_tokens": 192,
                    "fixed_final_completion_enabled": True,
                    "final_answer_contract_policy_version": (
                        "guided-grammar-fixed-192-token-strict-tail-local-projection-v1"
                    ),
                    "final_answer_schema_policy_version": (
                        "xgrammar-unbounded-answer-exact-url-v1"
                    ),
                    "final_answer_grammar_policy_version": (
                        "xgrammar-compact-unbounded-answer-exact-url-ascii-space-tail-v1"
                    ),
                    "final_answer_grammar_xgrammar_version": "0.1.21",
                    "output_contract_policy_version": (
                        "guided-tool-json-and-fixed-final-grammar-strict-local-projection-v1"
                    ),
                    "live_agent_sha256": live_agent_sha,
                    "tool_call_prompt_encoding": "canonical_json_sort_keys_compact",
                }
            )
    tasks: list[dict] = []
    llm_events: list[dict] = []
    tool_records: list[dict] = []
    saved_total = 0.0
    speculative_count = 0
    for source_index, source_row in enumerate(sources):
        source_id = source_row["source_id"]
        task_id = f"{source_id}__r00"
        canary = source_index % canary_stride == 0
        e2e = e2e_base + source_index * 0.01
        url = source_row["expected_url"]
        tools: list[dict] = []
        for tool_index, tool in enumerate(("search", "visit")):
            arguments = (
                {"query": [source_row["search_query"]]}
                if tool == "search"
                else {"url": [url], "goal": source_row["question"]}
            )
            invocation = {"tool_name": tool, "arguments": arguments}
            result_digest = hashlib.sha256(
                f"{block_id}:{cell}:{source_id}:{tool}:live-result".encode()
            ).hexdigest()
            speculative = speculation_mode != "off" and (
                (not is_modern and tool == "search")
                or (tool == "visit" and not canary)
            )
            source = "promoted_inflight" if speculative else "executed"
            saved = 0.1 if speculative else 0.0
            saved_total += saved
            speculative_count += int(speculative)
            queue_s = 0.2 if speculative else 0.5
            service_s = 0.5
            exposed_s = (
                6.0
                if is_fixed_final and cell == "E" and (tool == "search" or not canary)
                else 0.4
            )
            tools.append(
                {
                    "invocation": invocation,
                    "source": source,
                    "exposed_wait_s": exposed_s,
                    "queue_s": queue_s,
                    "service_s": service_s,
                    "saved_service_s": saved,
                    "result_sha256": result_digest,
                }
            )
            started_at = 100.0 + source_index * 5.0 + tool_index * 2.0
            interval = visit_interval if tool == "visit" else 0.0
            tool_records.append(
                {
                    "job_id": len(tool_records) + 1,
                    "invocation_id": (
                        f"{block_id}-{cell}-tool-{len(tool_records) + 1}"
                    ),
                    "session_id": task_id,
                    "tool": tool,
                    "invocation_digest": _invocation_sha(tool, arguments),
                    "speculative": speculative,
                    "authoritative": True,
                    "admitted": True,
                    "queue_enter": started_at - queue_s,
                    "queue_enter_at": started_at - queue_s,
                    "admitted_at": started_at - queue_s,
                    "start": started_at,
                    "started_at": started_at,
                    "confirmation": started_at + 0.1,
                    "authoritative_confirmation_at": started_at + 0.1,
                    "finish": started_at + service_s,
                    "finished_at": started_at + service_s,
                    "outcome": "committed",
                    "result_digest": result_digest,
                    "exact_match": speculative,
                    "source": source,
                    "cancelled": False,
                    "speculation_eligible": not (tool == "visit" and canary),
                    "canary": tool == "visit" and canary,
                    "worker_id": 0,
                    "tool_capacity": (
                        visit_capacity
                        if tool == "visit"
                        else (search_capacity or 4)
                    ),
                    "worker_pool": {
                        "max_workers": 4,
                        "max_speculative_workers": 2,
                        "max_speculative_pending": 128,
                        "tool_capacities": {
                            **({"search": search_capacity} if search_capacity else {}),
                            "visit": visit_capacity,
                        },
                        "tool_min_start_intervals_s": {
                            "visit": visit_interval
                        },
                    },
                    "queue_s": queue_s,
                    "service_s": service_s,
                    "exposed_wait_s": exposed_s,
                    "saved_service_s": saved,
                    "committed": True,
                    "response_status": 200,
                    "bytes_read": 1024,
                    "backend": (
                        "bing_html_search" if tool == "search" else "r.jina.ai"
                    ),
                    "request_host": (
                        "www.bing.com" if tool == "search" else "r.jina.ai"
                    ),
                    "http_attempts": 1,
                    "http_attempt_log": [
                        {
                            "request_index": 0,
                            "attempt": 1,
                            "status": 200,
                            "error_type": None,
                            "retried": False,
                            "started_monotonic_s": started_at,
                            "start_gate_wait_s": 0.0,
                            "retry_backoff_s": 0.0,
                        }
                    ],
                    "transport_identity_source": "actual",
                    "tool_min_start_interval_s": interval,
                    "rate_limit_eligible_at": started_at,
                    "rate_limit_next_eligible_at": started_at + interval,
                    "rate_limit_wait_s": 0.0,
                }
            )
        model_answer_text = f"answer for {source_id}"
        if is_v7 and source_index == 0:
            model_answer_text = " ".join(["x"] * 61)
            answer_text = " ".join(["x"] * 60)
        else:
            answer_text = model_answer_text
        answer = {"answer": answer_text, "source_url": url}
        extra_task_evidence: dict = {}
        fixed_final_response: str | None = None
        if is_v5 or is_v6 or is_v7 or is_fixed_final:
            guided_count = 2 if (is_v6 or is_v7 or is_fixed_final) else 3
            guided_calls = [
                {
                    "call_index": call_index,
                    **(
                        {
                            "mode": "guided_json",
                            "guided_json_requested": True,
                            "json_parse_attempted": True,
                            "local_wrap_applied": False,
                            "contract_succeeded": True,
                        }
                        if (is_v7 or is_fixed_final)
                        else {}
                    ),
                    "policy_version": "escape-unescaped-string-controls-v1",
                    "recovery_applied": False,
                    "parse_succeeded": True,
                    "raw_sha256": hashlib.sha256(
                        f"{task_id}:{call_index}:guided".encode()
                    ).hexdigest(),
                }
                for call_index in range(guided_count)
            ]
            extra_task_evidence["guided_json_recovery"] = {
                "policy_version": "escape-unescaped-string-controls-v1",
                "parsed_call_count": guided_count,
                "recovery_count": 0,
                "calls": guided_calls,
            }
        if is_v6:
            answer_text = str(answer["answer"])
            canonical_sha = hashlib.sha256(answer_text.encode()).hexdigest()
            final_contract = {
                "call_index": 2,
                "policy_version": "plain-text-unicode-whitespace-local-wrap-v1",
                "mode": "plain_text_local_wrap",
                "guided_json_requested": False,
                "json_parse_attempted": False,
                "local_wrap_applied": True,
                "object_constructed_locally": True,
                "source_url_binding": "exact_committed_selected_url",
                "source_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                "contract_succeeded": True,
                "raw_sha256": canonical_sha,
                "raw_char_count": len(answer_text),
                "max_chars": 480,
                "max_words": 60,
                "canonical_sha256": canonical_sha,
                "canonicalization_changed": False,
                "canonical_char_count": len(answer_text),
                "canonical_word_count": len(answer_text.split(" ")),
            }
            extra_task_evidence["output_contract"] = {
                "policy_version": "guided-tool-json-plain-final-local-wrap-v1",
                "calls": [
                    {
                        "call_index": call_index,
                        "mode": "guided_json",
                        "guided_json_requested": True,
                        "json_parse_attempted": True,
                        "local_wrap_applied": False,
                        "parse_succeeded": True,
                        "contract_succeeded": True,
                        "recovery_applied": False,
                        "raw_sha256": guided_calls[call_index]["raw_sha256"],
                    }
                    for call_index in range(2)
                ]
                + [dict(final_contract)],
            }
            extra_task_evidence["final_answer_contract"] = final_contract
        if is_v7:
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer", "source_url"],
                "properties": {
                    "answer": {"type": "string"},
                    "source_url": {"const": url},
                },
            }
            raw_object = {"answer": model_answer_text, "source_url": url}
            raw_text = json.dumps(
                raw_object,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            pre_projection_sha = hashlib.sha256(
                model_answer_text.encode()
            ).hexdigest()
            final_sha = hashlib.sha256(answer_text.encode()).hexdigest()
            word_projection = source_index == 0
            final_contract = {
                "call_index": 2,
                "policy_version": (
                    "guided-json-strict-local-whitespace-bounded-prefix-v2"
                ),
                "schema_policy_version": (
                    "xgrammar-unbounded-answer-exact-url-v1"
                ),
                "schema_sha256": _canonical_sha(schema),
                "schema_answer_constraint": "type_only_no_length_or_pattern",
                "mode": "guided_json_strict_local_projection",
                "guided_json_requested": True,
                "json_parse_attempted": True,
                "strict_json_parse": True,
                "recovery_allowed": False,
                "recovery_applied": False,
                "parse_succeeded": True,
                "local_wrap_applied": True,
                "local_projection_applied": word_projection,
                "object_constructed_locally": True,
                "source_url_binding": "exact_committed_selected_url",
                "source_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                "contract_succeeded": True,
                "raw_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                "raw_char_count": len(raw_text),
                "max_chars": 480,
                "max_words": 60,
                "target_chars": 360,
                "model_answer_sha256": pre_projection_sha,
                "model_answer_char_count": len(model_answer_text),
                "model_source_url_validated": True,
                "pre_projection_canonical_sha256": pre_projection_sha,
                "pre_projection_char_count": len(model_answer_text),
                "pre_projection_word_count": len(model_answer_text.split(" ")),
                "canonical_sha256": final_sha,
                "canonicalization_changed": False,
                "canonical_char_count": len(answer_text),
                "canonical_word_count": len(answer_text.split(" ")),
                "word_projection_applied": word_projection,
                "char_projection_applied": False,
            }
            extra_task_evidence["output_contract"] = {
                "policy_version": (
                    "guided-tool-and-final-json-strict-local-projection-v2"
                ),
                "calls": [
                    {
                        "call_index": call_index,
                        "mode": "guided_json",
                        "guided_json_requested": True,
                        "json_parse_attempted": True,
                        "local_wrap_applied": False,
                        "parse_succeeded": True,
                        "contract_succeeded": True,
                        "recovery_applied": False,
                        "raw_sha256": guided_calls[call_index]["raw_sha256"],
                    }
                    for call_index in range(2)
                ]
                + [dict(final_contract)],
            }
            extra_task_evidence["final_answer_contract"] = final_contract
        if is_fixed_final:
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer", "source_url"],
                "properties": {
                    "answer": {"type": "string"},
                    "source_url": {"const": url},
                },
            }
            semantic_wire = json.dumps(
                answer,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            padding = " " * 200
            fixed_final_response = semantic_wire + padding
            answer_sha = hashlib.sha256(answer_text.encode()).hexdigest()
            final_contract = {
                "call_index": 2,
                "policy_version": (
                    "guided-grammar-fixed-192-token-strict-tail-local-projection-v1"
                ),
                "schema_policy_version": (
                    "xgrammar-unbounded-answer-exact-url-v1"
                ),
                "schema_sha256": _canonical_sha(schema),
                "schema_answer_constraint": "type_only_no_length_or_pattern",
                "mode": (
                    "guided_grammar_fixed_completion_strict_raw_decode_"
                    "local_projection"
                ),
                "guided_json_requested": False,
                "guided_grammar_requested": True,
                "json_parse_attempted": True,
                "strict_json_parse": True,
                "strict_json_raw_decode": True,
                "recovery_allowed": False,
                "recovery_applied": False,
                "parse_succeeded": True,
                "local_wrap_applied": True,
                "local_projection_applied": False,
                "object_constructed_locally": True,
                "source_url_binding": "exact_committed_selected_url",
                "source_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                "contract_succeeded": True,
                "raw_sha256": hashlib.sha256(
                    fixed_final_response.encode()
                ).hexdigest(),
                "raw_char_count": len(fixed_final_response),
                "max_chars": 480,
                "max_words": 60,
                "target_chars": 360,
                "grammar_policy_version": (
                    "xgrammar-compact-unbounded-answer-exact-url-"
                    "ascii-space-tail-v1"
                ),
                "grammar_xgrammar_version": "0.1.21",
                "grammar_sha256": _fixed_final_grammar_sha256(url),
                "grammar_semantic_json_whitespace": "compact",
                "tail_policy": "one_or_more_ascii_spaces_only",
                "tail_validation_succeeded": True,
                "fixed_completion_tokens": 192,
                "min_tokens": 192,
                "max_tokens": 192,
                "total_completion_tokens": 192,
                "finish_reason": "length",
                "finish_reason_validated": True,
                "token_accounting_succeeded": True,
                "semantic_sha256": hashlib.sha256(
                    semantic_wire.encode()
                ).hexdigest(),
                "semantic_char_count": len(semantic_wire),
                "semantic_byte_count": len(semantic_wire.encode()),
                "padding_sha256": hashlib.sha256(padding.encode()).hexdigest(),
                "padding_char_count": len(padding),
                "padding_byte_count": len(padding.encode()),
                "tail_nonempty": True,
                "tail_ascii_space_only": True,
                "token_counter_method": "transformers_chat_template",
                "semantic_token_count": 40,
                "padding_token_count": 152,
                "token_partition_method": (
                    "server_total_minus_local_semantic_tokenization"
                ),
                "model_answer_sha256": answer_sha,
                "model_answer_char_count": len(answer_text),
                "model_source_url_validated": True,
                "pre_projection_canonical_sha256": answer_sha,
                "pre_projection_char_count": len(answer_text),
                "pre_projection_word_count": len(answer_text.split(" ")),
                "canonical_sha256": answer_sha,
                "canonicalization_changed": False,
                "canonical_char_count": len(answer_text),
                "canonical_word_count": len(answer_text.split(" ")),
                "word_projection_applied": False,
                "char_projection_applied": False,
            }
            extra_task_evidence["output_contract"] = {
                "policy_version": (
                    "guided-tool-json-and-fixed-final-grammar-strict-"
                    "local-projection-v1"
                ),
                "calls": [
                    {
                        "call_index": call_index,
                        "mode": "guided_json",
                        "guided_json_requested": True,
                        "json_parse_attempted": True,
                        "local_wrap_applied": False,
                        "parse_succeeded": True,
                        "contract_succeeded": True,
                        "recovery_applied": False,
                        "raw_sha256": guided_calls[call_index]["raw_sha256"],
                    }
                    for call_index in range(2)
                ]
                + [dict(final_contract)],
            }
            extra_task_evidence["final_answer_contract"] = final_contract
        tasks.append(
            {
                "task_id": task_id,
                "source_id": source_id,
                "replica": 0,
                "ok": True,
                "visit_canary": canary,
                "start_wall_s": 2000.0,
                "end_wall_s": 2000.0 + e2e,
                "e2e_s": e2e,
                "question_sha256": hashlib.sha256(
                    source_row["question"].encode("utf-8")
                ).hexdigest(),
                "search_query": source_row["search_query"],
                "search_urls": [url],
                "selected_url": url,
                "call_graph_mode": "frozen",
                "expected_url": url,
                "search_result_contains_expected_url": True,
                "answer": answer,
                "answer_sha256": _canonical_sha(answer),
                **extra_task_evidence,
                "tools": tools,
                "llm_duration_s": 1.2,
                "prompt_tokens": 3 * llm_prompt_tokens,
                "completion_tokens": 196 if is_fixed_final else 6,
                "context_padding_target_tokens": context_padding,
                "context_padding_actual_tokens": actual_padding,
            }
        )
        for call_index in range(3):
            completion_tokens = 192 if is_fixed_final and call_index == 2 else 2
            semantic_response = fixed_final_response or ""
            llm_events.append(
                {
                    "task_id": task_id,
                    "call_index": call_index,
                    "request_id": f"{block_id}-{cell}-{source_id}-{call_index}",
                    "request_start_s": 2000.0 + call_index,
                    "duration_s": 0.4,
                    "prompt_tokens_estimate": llm_prompt_tokens,
                    "attempts": 1,
                    "ok": True,
                    "http_status": 200,
                    **(
                        {
                            "response": semantic_response,
                            "response_sha256": hashlib.sha256(
                                semantic_response.encode()
                            ).hexdigest(),
                            "finish_reason": "length",
                            "output_mode": "guided_grammar",
                            "guided_json_requested": False,
                            "guided_grammar_requested": True,
                            "guided_grammar_sha256": _fixed_final_grammar_sha256(
                                url
                            ),
                            "min_tokens": 192,
                            "max_tokens": 192,
                        }
                        if is_fixed_final and call_index == 2
                        else {}
                    ),
                    **(
                        {
                            "output_mode": (
                                "guided_grammar"
                                if is_fixed_final and call_index == 2
                                else "guided_json"
                                if (is_v7 or call_index < 2)
                                else "plain_text"
                            ),
                            "guided_json_requested": (
                                not (is_fixed_final and call_index == 2)
                                and (is_v7 or call_index < 2)
                            ),
                        }
                        if (is_v6 or is_v7 or is_fixed_final)
                        else {}
                    ),
                    **(
                        {
                            "guided_grammar_requested": False,
                            "guided_grammar_sha256": None,
                            "min_tokens": 0,
                            "max_tokens": 128,
                        }
                        if is_fixed_final and call_index < 2
                        else {}
                    ),
                    "usage": {
                        "prompt_tokens": llm_prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": llm_prompt_tokens + completion_tokens,
                    },
                    "scheduler_meta": {
                        "t": task_id,
                        "c": call_index,
                        "ms": "live_broker",
                        "tqa": 1,
                        "tqs": int(speculation_mode != "off"),
                        "tra": 1,
                        "trs": int(speculation_mode != "off"),
                    },
                }
            )

    result = {
        "schema_version": 1,
        "config": config,
        "summary": {
            "all_tasks_succeeded": True,
            "task_count": source_count,
            "successful_task_count": source_count,
            "failed_task_count": 0,
            "llm": {
                "request_count": 3 * source_count,
                "successful_request_count": 3 * source_count,
                "exactly_one_attempt_each": True,
            },
            "tool": {"authoritative_commit_count": 2 * source_count},
        },
        "task_completion_makespan_s": max(task["e2e_s"] for task in tasks) + 0.1,
        "tasks": tasks,
        "llm_events": llm_events,
        "tool_attempt_records": tool_records,
        "broker_final_snapshot": {
            "counts": {
                "queued_authoritative": 0,
                "queued_speculative": 0,
                "running_authoritative": 0,
                "running_speculative": 0,
                "completed_unclaimed_speculative": 0,
                "queued_by_tool": {},
                "running_by_tool": {},
            },
            "jobs": [],
            "stats": {
                "commits": 2 * source_count,
                "authoritative_requests": 2 * source_count,
                "authoritative_failures": 0,
                "saved_service_s": saved_total,
                "wasted_speculative_service_s": 0.0,
                "speculative_admitted": speculative_count,
                "queued_promotions": 0,
                "running_promotions": speculative_count,
                "completed_reuse": 0,
            },
        },
        "vllm_metric_deltas": {
            "vllm:prompt_tokens_total": float(source_count * 3 * llm_prompt_tokens),
            "vllm:generation_tokens_total": float(
                source_count * (196 if is_fixed_final else 6)
            ),
            "vllm:request_queue_time_seconds_sum": 10.0,
            "vllm:num_preemptions_total": 0.0,
        },
        "raw_evidence": {
            "queue_timeline": {
                "path": str(timeline_path.resolve()),
                "sha256": _sha(timeline_path),
                "sample_count": len(timeline),
            }
        },
    }
    result_path = evidence_dir / "result.json"
    _write_json(result_path, result)
    if is_v9:
        relative = lambda path: str(path.resolve().relative_to(REPOSITORY_ROOT))
        selection_provenance = {
            "completed_screen": {
                "path": (
                    "reproduction/artifacts/live_joint/development/v9_screen/"
                    "v9-screen-r1/completed_screen.json"
                ),
                "sha256": (
                    "40b4a8033529883f26c1f298d54a92a69e4fcfb6cb942a8d5f70c98fc86481f3"
                ),
            },
            "strict_development_selection": {
                "path": (
                    "reproduction/artifacts/live_joint/development/v9_screen/"
                    "v9-screen-r1/strict_development_selection.json"
                ),
                "sha256": (
                    "7f7c9de71f341741192de78ab8596b9cb01721fe211ec3faed79ee33bd7dc7cc"
                ),
            },
            "selected_transport": {
                "path": (
                    "reproduction/artifacts/live_joint/development/v9_screen/"
                    "v9-screen-r1/stage-0/selected_transport.json"
                ),
                "sha256": (
                    "3c44458963c65deb55b35dfa5a2ff888d5e1ec4cb6c0ff350ebe41e53612dc0d"
                ),
            },
            "selected_policy": "F0",
            "selected_visit_interval_s": 2.5,
            "selected_min_speculative_tool_workers": 0,
            "maximum_observed_http_retries_per_cell": 0,
            "zero_wasted_speculative_service_required": True,
            "live_broker_sha256": (
                "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27"
            ),
            "workload": {
                "path": relative(workload_path),
                "raw_sha256": formal["file_sha256"],
                "canonical_sha256": formal["canonical_json_sha256"],
                "sources_sha256": formal["canonical_sources_sha256"],
                "source_count": 80,
            },
        }
        effective_path = run_dir / "effective_config.json"
        effective = {
            "schema": "paste_repro.live_joint_formal_cell_config",
            "version": 1,
            "formal_generation": "v9",
            "block_id": block_id,
            "block_number": int(block_id.rsplit("-", 1)[-1]),
            "cell_id": cell,
            "order_index": order_index,
            "server_instance_id": f"server-{block_id}-{cell}",
            "llm_scheduler": scheduler["VLLM_SCHED_POLICY"],
            "speculation_mode": speculation_mode,
            "call_graph_mode": "frozen",
            "min_speculative_tool_workers": 0,
            "workload": {
                "path": relative(workload_path),
                "sha256": formal["file_sha256"],
            },
            "formal_v9_selection": selection_provenance,
        }
        _write_json(effective_path, effective)
        manifest_path = run_dir / "cell_manifest.json"
        manifest = {
            "schema": "paste_repro.live_joint_formal_cell_evidence",
            "version": 1,
            "block_id": block_id,
            "cell_id": cell,
            "order_index": order_index,
            "server_instance_id": f"server-{block_id}-{cell}",
            "evidence": {
                relative(effective_path): _sha(effective_path),
                relative(result_path): _sha(result_path),
                relative(timeline_path): _sha(timeline_path),
            },
        }
        _write_json(manifest_path, manifest)
    return result_path


def _make_blocks(
    root: Path,
    *,
    workload_path: Path = FORMAL_V3_WORKLOAD,
) -> list[tuple[str, Path, Path, Path, Path]]:
    orders = {
        "block-1": ["A", "B", "E", "F"],
        "block-2": ["B", "A", "F", "E"],
        "block-3": ["E", "A", "F", "B"],
    }
    durations = {"A": 100.0, "B": 80.0, "E": 75.0, "F": 65.0}
    blocks = []
    for block_number, (block_id, order) in enumerate(orders.items()):
        paths = {}
        for cell in ("A", "B", "E", "F"):
            paths[cell] = _make_run(
                root,
                block_id=block_id,
                cell=cell,
                order_index=order.index(cell),
                e2e_base=durations[cell] + block_number,
                workload_path=workload_path,
            )
        blocks.append(
            (block_id, paths["A"], paths["B"], paths["E"], paths["F"])
        )
    return blocks


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class AggregateLiveJointFourCellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.blocks = _make_blocks(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict:
        return aggregate_live_joint_four_cell(
            self.blocks, bootstrap_resamples=200
        )

    def rewrite(self, path: Path, mutate) -> None:
        value = _read(path)
        mutate(value)
        _write_json(path, value)

    def mark_committed_search_retries(self, path: Path, count: int) -> None:
        def mutate(payload: dict) -> None:
            rows = [
                row
                for row in payload["tool_attempt_records"]
                if row["tool"] == "search" and row["committed"] is True
            ][:count]
            self.assertEqual(len(rows), count)
            tasks = {task["task_id"]: task for task in payload["tasks"]}
            for row in rows:
                row["http_attempts"] = 2
                row["service_s"] += 1.0
                row["finished_at"] += 1.0
                row["finish"] = row["finished_at"]
                tasks[row["session_id"]]["tools"][0]["service_s"] = row[
                    "service_s"
                ]

        self.rewrite(path, mutate)

    def test_valid_formal_four_cell_aggregation_passes(self) -> None:
        result = self.aggregate()
        formal = validate_formal_workload(FORMAL_V3_WORKLOAD)
        self.assertTrue(result["formal_promotion_passed"])
        self.assertEqual(result["failed_gate_names"], [])
        self.assertEqual(result["design"]["block_count"], 3)
        self.assertEqual(result["design"]["independent_source_count"], 60)
        self.assertEqual(result["design"]["replicas_per_source"], 1)
        self.assertEqual(result["design"]["tasks_per_cell_per_block"], 60)
        self.assertEqual(
            result["design"]["logical_llm_requests_per_cell_per_block"], 180
        )
        self.assertEqual(
            result["design"]["authoritative_tool_commits_per_cell_per_block"],
            120,
        )
        self.assertEqual(result["design"]["effective_bootstrap_sample_size"], 60)
        self.assertTrue(result["design"]["replicas_are_not_independent_samples"])
        self.assertEqual(
            result["design"]["formal_load"],
            {
                "max_active_tasks": 60,
                "vllm_max_num_seqs": 96,
                "context_padding_target_tokens": 5600,
                "visit_tool_capacity": 1,
                "visit_min_start_interval_s": 2.1,
            },
        )
        self.assertEqual(
            result["design"]["identity_validation"]["task_identity_count"], 60
        )
        self.assertEqual(
            result["design"]["identity_validation"]["split_id"],
            formal["split_id"],
        )
        self.assertEqual(
            result["design"]["identity_validation"]["workload_file_sha256"],
            formal["file_sha256"],
        )
        self.assertEqual(
            result["design"]["identity_validation"][
                "selected_workload_sha256"
            ],
            formal["canonical_sources_sha256"],
        )
        self.assertEqual(
            result["design"]["identity_validation"][
                "context_padding_actual_tokens"
            ]["observation_count"],
            3 * 4 * 60,
        )
        self.assertEqual(result["design"]["unique_fresh_server_instance_count"], 12)
        self.assertEqual(result["design"]["A_B_forward_count"], 2)
        self.assertEqual(result["design"]["E_F_reverse_count"], 1)
        self.assertEqual(result["effects"]["E_to_F"]["faster_source_count"], 60)
        self.assertGreater(
            result["effects"]["A_to_F"]["aggregate_relative_reduction"],
            0.25,
        )
        self.assertEqual(
            result["effects"]["E_to_F"]["bootstrap"]["seed"],
            BOOTSTRAP_SEED,
        )
        self.assertEqual(
            result["effects"]["E_to_F"]["bootstrap"]["sample_size"], 60
        )
        self.assertEqual(
            result["aggregate_cells"]["A"]["source_distribution_s"]["count"],
            60,
        )
        self.assertEqual(
            result["aggregate_cells"]["A"]["task_e2e_s"]["count"], 180
        )
        self.assertEqual(
            result["blocks"]["block-1"]["A"]["task_e2e_s"]["count"], 60
        )
        self.assertEqual(
            result["blocks"]["block-1"]["A"]["llm_request_duration_s"][
                "count"
            ],
            180,
        )
        self.assertEqual(
            result["blocks"]["block-1"]["A"]["tool"][
                "authoritative_commit_count"
            ],
            120,
        )
        self.assertLess(result["interaction"]["mean_interaction_s"], 0.0)
        self.assertEqual(
            result["interaction"]["acceptance_effect"], "reported_only"
        )

    def test_controlled_authoritative_retry_below_threshold_is_accepted(self) -> None:
        self.mark_committed_search_retries(self.blocks[0][4], 1)
        result = self.aggregate()
        self.assertTrue(result["formal_promotion_passed"])
        retry = result["aggregate_cells"]["F"]["authoritative_retry"]
        self.assertEqual(retry["retried_commit_count"], 1)
        self.assertEqual(retry["commit_count"], 360)
        self.assertAlmostEqual(retry["rate"], 1 / 360)
        self.assertEqual(
            result["aggregate_cells"]["F"]["physical_http_attempt_count"],
            361,
        )
        self.assertTrue(
            result["diagnostics"]["authoritative_retry"][
                "service_and_waste_include_attempts_and_fixed_backoff"
            ]
        )

    def test_authoritative_retry_rate_above_two_percent_fails_gate(self) -> None:
        self.mark_committed_search_retries(self.blocks[0][4], 3)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "all_cells_authoritative_retry_rate_at_most_2pct",
            result["failed_gate_names"],
        )

    def test_retry_rate_imbalance_above_one_pp_fails_EF_and_AF_gates(self) -> None:
        for block in self.blocks:
            self.mark_committed_search_retries(block[4], 2)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertNotIn(
            "all_cells_authoritative_retry_rate_at_most_2pct",
            result["failed_gate_names"],
        )
        self.assertIn(
            "E_to_F_authoritative_retry_rate_difference_at_most_1pp",
            result["failed_gate_names"],
        )
        self.assertIn(
            "A_to_F_authoritative_retry_rate_difference_at_most_1pp",
            result["failed_gate_names"],
        )

    def test_reused_server_instance_fails_closed(self) -> None:
        duplicate = _read(self.blocks[0][1])["config"]["formal_run"][
            "server_instance_id"
        ]
        self.rewrite(
            self.blocks[0][2],
            lambda payload: payload["config"]["formal_run"].update(
                server_instance_id=duplicate
            ),
        )
        with self.assertRaisesRegex(ValueError, "server_instance_id is reused"):
            self.aggregate()

    def test_formal_workload_identity_mismatch_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[1][4],
            lambda payload: payload["tasks"][0].update(question_sha256="0" * 64),
        )
        with self.assertRaisesRegex(ValueError, "question differs"):
            self.aggregate()

    def test_r00_only_replica_identity_fails_closed(self) -> None:
        target = self.blocks[0][1]
        self.rewrite(
            target,
            lambda payload: payload["tasks"][0].update(replica=1),
        )
        with self.assertRaisesRegex(ValueError, "task_id is not canonical"):
            self.aggregate()

    def test_context_padding_actual_fails_closed(self) -> None:
        target = self.blocks[0][1]
        self.rewrite(
            target,
            lambda payload: payload["tasks"][0].update(
                context_padding_actual_tokens=5599
            ),
        )
        with self.assertRaisesRegex(ValueError, "padding actual is invalid"):
            self.aggregate()

    def test_non_factor_config_difference_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[2][2],
            lambda payload: payload["config"].update(visit_max_chars=999),
        )
        with self.assertRaisesRegex(ValueError, "outside the two factorial"):
            self.aggregate()

    def test_unbalanced_within_pair_order_fails_closed(self) -> None:
        for block in self.blocks:
            for order_index, path in enumerate(block[1:]):
                self.rewrite(
                    path,
                    lambda payload, index=order_index: payload["config"][
                        "formal_run"
                    ].update(order_index=index),
                )
        with self.assertRaisesRegex(ValueError, "A/B forward and reverse"):
            self.aggregate()

    def test_physical_visit_start_interval_violation_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def violate(payload: dict) -> None:
            visits = [
                row
                for row in payload["tool_attempt_records"]
                if row["tool"] == "visit"
            ]
            first_start = visits[0]["started_at"]
            second = visits[1]
            second["started_at"] = first_start + 1.0
            second["start"] = second["started_at"]
            second["queue_enter_at"] = second["started_at"] - second["queue_s"]
            second["queue_enter"] = second["queue_enter_at"]
            second["admitted_at"] = second["queue_enter_at"]
            second["authoritative_confirmation_at"] = second["started_at"] + 0.1
            second["confirmation"] = second["authoritative_confirmation_at"]
            second["finished_at"] = second["started_at"] + second["service_s"]
            second["finish"] = second["finished_at"]
            second["rate_limit_eligible_at"] = second["started_at"]
            second["rate_limit_next_eligible_at"] = second["started_at"] + 2.1

        self.rewrite(target, violate)
        with self.assertRaisesRegex(ValueError, "minimum start interval"):
            self.aggregate()

    def test_started_physical_job_requires_actual_final_http_200(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tool_attempt_records"][0].update(
                transport_identity_source="planned"
            ),
        )
        with self.assertRaisesRegex(ValueError, "actual final HTTP evidence"):
            self.aggregate()

    def test_never_started_cancellation_requires_zero_attempt_telemetry(self) -> None:
        target = self.blocks[0][4]

        def add_cancelled(payload: dict) -> None:
            row = copy.deepcopy(payload["tool_attempt_records"][0])
            finished = row["queue_enter_at"] + 3.0
            row.update(
                {
                    "job_id": 9999,
                    "invocation_id": "never-started-cancelled",
                    "speculative": True,
                    "authoritative": False,
                    "committed": False,
                    "confirmation": None,
                    "authoritative_confirmation_at": None,
                    "start": None,
                    "started_at": None,
                    "finish": finished,
                    "finished_at": finished,
                    "outcome": "cancelled",
                    "result_digest": None,
                    "exact_match": False,
                    "source": "cancelled",
                    "cancelled": True,
                    "worker_id": None,
                    "queue_s": 3.0,
                    "service_s": 0.0,
                    "exposed_wait_s": None,
                    "saved_service_s": 0.0,
                    "response_status": None,
                    "bytes_read": None,
                    "backend": None,
                    "request_host": None,
                    "http_attempts": 0,
                    "transport_identity_source": None,
                }
            )
            payload["tool_attempt_records"].append(row)

        self.rewrite(target, add_cancelled)
        self.aggregate()

        self.rewrite(
            target,
            lambda payload: payload["tool_attempt_records"][-1].update(
                http_attempts=None
            ),
        )
        with self.assertRaisesRegex(ValueError, "http_attempts must be an integer"):
            self.aggregate()

    def test_performance_failure_is_reported_as_failed_gates(self) -> None:
        for block in self.blocks:
            f_path = block[4]

            def regress(payload: dict) -> None:
                for task in payload["tasks"]:
                    task["e2e_s"] = 99.0
                    task["end_wall_s"] = task["start_wall_s"] + 99.0
                payload["task_completion_makespan_s"] = 99.1

            self.rewrite(f_path, regress)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn("E_to_F_mean_reduction", result["failed_gate_names"])
        self.assertIn("A_to_F_mean_reduction", result["failed_gate_names"])
        self.assertTrue(result["effects"]["A_to_B"]["every_block_mean_reduction_positive"])

    def test_controlled_speculative_retry_is_counted_in_attempts_and_waste(self) -> None:
        target = self.blocks[0][4]

        def add_retry(payload: dict) -> None:
            row = copy.deepcopy(payload["tool_attempt_records"][0])
            row.update(
                {
                    "job_id": 9999,
                    "invocation_id": "uncommitted-retried-speculation",
                    "authoritative": False,
                    "committed": False,
                    "outcome": "completed",
                    "source": "speculative",
                    "exact_match": False,
                    "saved_service_s": 0.0,
                    "exposed_wait_s": None,
                    "started_at": 10_000.0,
                    "start": 10_000.0,
                    "queue_enter_at": 10_000.0,
                    "queue_enter": 10_000.0,
                    "admitted_at": 10_000.0,
                    "finished_at": 10_001.5,
                    "finish": 10_001.5,
                    "queue_s": 0.0,
                    "service_s": 1.5,
                    "http_attempts": 2,
                    "tool_min_start_interval_s": 0.0,
                    "rate_limit_eligible_at": 10_000.0,
                    "rate_limit_next_eligible_at": 10_000.0,
                }
            )
            payload["tool_attempt_records"].append(row)
            payload["broker_final_snapshot"]["stats"][
                "wasted_speculative_service_s"
            ] = 1.5

        self.rewrite(target, add_retry)
        result = self.aggregate()
        self.assertTrue(result["formal_promotion_passed"])
        self.assertNotIn("zero_uncontrolled_http_retries", result["failed_gate_names"])
        self.assertEqual(
            result["aggregate_cells"]["F"]["uncontrolled_retry_count"], 0
        )
        self.assertEqual(
            result["aggregate_cells"]["F"]["retried_physical_job_count"], 1
        )

    def test_uncontrolled_retry_configuration_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["config"].update(
                tool_http_max_attempts=3,
                controlled_http_retry=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
            self.aggregate()

    def test_formal_http_library_version_is_exactly_frozen(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["config"].update(
                tool_http_library_version="3.13.0"
            ),
        )
        with self.assertRaisesRegex(ValueError, "must freeze aiohttp 3.12.15"):
            self.aggregate()

    def test_per_block_completion_token_imbalance_fails_gate(self) -> None:
        target = self.blocks[0][4]

        def add_tokens(payload: dict) -> None:
            event = payload["llm_events"][0]
            event["usage"]["completion_tokens"] += 20
            event["usage"]["total_tokens"] += 20
            payload["tasks"][0]["completion_tokens"] += 20
            payload["vllm_metric_deltas"]["vllm:generation_tokens_total"] += 20

        self.rewrite(target, add_tokens)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "E_to_F_completion_token_difference_below_1pct",
            result["failed_gate_names"],
        )
        observed = result["formal_gates"][
            "E_to_F_completion_token_difference_below_1pct"
        ]["observed"]
        self.assertGreater(observed["by_block"]["block-1"], 0.01)

    def test_cli_requires_block_quintuples(self) -> None:
        parsed = parse_args(
            [
                "--block",
                "b1",
                "a.json",
                "b.json",
                "e.json",
                "f.json",
                "--formal-workload",
                "formal.json",
                "--output",
                "aggregate.json",
            ]
        )
        self.assertEqual(parsed.block[0][0], "b1")
        self.assertEqual(parsed.block[0][4], "f.json")
        self.assertEqual(parsed.formal_workload, Path("formal.json"))


class AggregateLiveJointFourCellV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.blocks = _make_blocks(self.root, workload_path=FORMAL_V4_WORKLOAD)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self, *, explicit_workload: bool = True) -> dict:
        return aggregate_live_joint_four_cell(
            self.blocks,
            bootstrap_resamples=200,
            formal_workload=FORMAL_V4_WORKLOAD if explicit_workload else None,
        )

    def rewrite(self, path: Path, mutate) -> None:
        value = _read(path)
        mutate(value)
        _write_json(path, value)

    def test_v4_profile_auto_detects_and_passes_strict_live_evidence(self) -> None:
        result = self.aggregate(explicit_workload=False)
        formal = validate_formal_workload(FORMAL_V4_WORKLOAD)
        self.assertTrue(result["formal_promotion_passed"])
        self.assertEqual(result["design"]["formal_profile"], "formal-v4")
        self.assertEqual(
            result["design"]["formal_workload"]["file_sha256"],
            formal["file_sha256"],
        )
        self.assertEqual(
            result["design"]["config_validation"]["speculation_mode"],
            "visit",
        )
        self.assertEqual(
            result["design"]["formal_load"],
            {
                "max_active_tasks": 60,
                "vllm_max_num_seqs": 96,
                "context_padding_target_tokens": 10000,
                "visit_tool_capacity": 2,
                "visit_min_start_interval_s": 2.1,
                "vllm_max_model_len": 16384,
                "vllm_max_num_batched_tokens": 2048,
                "visit_canary_stride": 6,
                "expected_canary_count": 10,
            },
        )
        self.assertTrue(
            result["formal_gates"][
                "formal_v4_execution_aware_policy_and_code_binding"
            ]["passed"]
        )
        self.assertTrue(
            result["formal_gates"][
                "formal_v4_http_attempt_gate_and_success_ledgers"
            ]["passed"]
        )
        self.assertTrue(
            result["formal_gates"][
                "formal_v4_ten_canaries_skip_visit_speculation_before_enqueue"
            ]["passed"]
        )
        for block in result["diagnostics"]["canary_pre_enqueue_skip"].values():
            for row in block.values():
                self.assertEqual(row["canary_task_count"], 10)
                self.assertEqual(row["authoritative_canary_visit_commit_count"], 10)
                self.assertEqual(row["canary_speculative_record_count"], 0)
                self.assertEqual(row["canary_speculative_visit_record_count"], 0)

    def test_v4_requires_visit_only_speculation(self) -> None:
        for block in self.blocks:
            for path in (block[2], block[4]):
                self.rewrite(
                    path,
                    lambda payload: payload["config"].update(
                        speculation_mode="search_visit"
                    ),
                )
        with self.assertRaisesRegex(ValueError, "B/F speculation treatments differ"):
            self.aggregate()

    def test_v4_execution_policy_module_sha_drift_fails_closed(self) -> None:
        for block in self.blocks:
            for path in block[1:]:
                self.rewrite(
                    path,
                    lambda payload: payload["config"].update(
                        tool_signal_policy_module_sha256="0" * 64
                    ),
                )
        with self.assertRaisesRegex(ValueError, "runtime contract differs"):
            self.aggregate()

    def test_v4_missing_http_attempt_ledger_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tool_attempt_records"][0].pop(
                "http_attempt_log"
            ),
        )
        with self.assertRaisesRegex(ValueError, "HTTP-attempt ledger"):
            self.aggregate()

    def test_v4_retry_attempt_must_obey_physical_start_gate(self) -> None:
        target = self.blocks[0][4]

        def violate_attempt_gate(payload: dict) -> None:
            visit = next(
                row
                for row in reversed(payload["tool_attempt_records"])
                if row["tool"] == "visit"
            )
            started = visit["started_at"]
            visit["http_attempts"] = 2
            visit["http_attempt_log"] = [
                {
                    "request_index": 0,
                    "attempt": 1,
                    "status": 429,
                    "error_type": "aiohttp.ClientResponseError",
                    "retried": True,
                    "started_monotonic_s": started,
                    "start_gate_wait_s": 0.0,
                    "retry_backoff_s": 1.0,
                },
                {
                    "request_index": 0,
                    "attempt": 2,
                    "status": 200,
                    "error_type": None,
                    "retried": False,
                    "started_monotonic_s": started + 1.0,
                    "start_gate_wait_s": 0.0,
                    "retry_backoff_s": 0.0,
                },
            ]
            visit["service_s"] = 2.5
            visit["finished_at"] = started + 2.5
            visit["finish"] = visit["finished_at"]
            task = next(
                task
                for task in payload["tasks"]
                if task["task_id"] == visit["session_id"]
            )
            task["tools"][1]["service_s"] = 2.5

        self.rewrite(target, violate_attempt_gate)
        with self.assertRaisesRegex(ValueError, "physical HTTP-attempt start gate"):
            self.aggregate()

    def test_v4_frozen_vllm_batch_profile_drift_fails_closed(self) -> None:
        for block in self.blocks:
            for path in block[1:]:
                self.rewrite(
                    path,
                    lambda payload: payload["config"][
                        "scheduler_environment"
                    ].update(VLLM_MAX_NUM_BATCHED_TOKENS="4096"),
                )
        with self.assertRaisesRegex(ValueError, "vLLM profile differs"):
            self.aggregate()

    def test_v4_canary_visit_prediction_enqueue_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def add_canary_prediction(payload: dict) -> None:
            canary_task_id = payload["tasks"][0]["task_id"]
            authoritative = next(
                row
                for row in payload["tool_attempt_records"]
                if row["session_id"] == canary_task_id and row["tool"] == "visit"
            )
            prediction = copy.deepcopy(authoritative)
            prediction.update(
                {
                    "job_id": 9999,
                    "invocation_id": "forbidden-canary-visit-prediction",
                    "speculative": True,
                    "authoritative": False,
                    "committed": False,
                    "outcome": "completed",
                    "source": "speculative",
                    "exact_match": False,
                    "canary": False,
                    "saved_service_s": 0.0,
                    "started_at": 10_000.0,
                    "start": 10_000.0,
                    "queue_enter_at": 10_000.0,
                    "queue_enter": 10_000.0,
                    "admitted_at": 10_000.0,
                    "finished_at": 10_000.5,
                    "finish": 10_000.5,
                    "queue_s": 0.0,
                    "service_s": 0.5,
                    "exposed_wait_s": None,
                    "http_attempt_log": [
                        {
                            "request_index": 0,
                            "attempt": 1,
                            "status": 200,
                            "error_type": None,
                            "retried": False,
                            "started_monotonic_s": 10_000.0,
                            "start_gate_wait_s": 0.0,
                            "retry_backoff_s": 0.0,
                        }
                    ],
                    "rate_limit_eligible_at": 10_000.0,
                    "rate_limit_next_eligible_at": 10_002.1,
                }
            )
            payload["tool_attempt_records"].append(prediction)
            payload["broker_final_snapshot"]["stats"][
                "wasted_speculative_service_s"
            ] = 0.5

        self.rewrite(target, add_canary_prediction)
        with self.assertRaisesRegex(ValueError, "canary prediction"):
            self.aggregate()


class AggregateLiveJointFourCellV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.blocks = _make_blocks(self.root, workload_path=FORMAL_V5_WORKLOAD)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict:
        return aggregate_live_joint_four_cell(
            self.blocks,
            bootstrap_resamples=200,
            formal_workload=FORMAL_V5_WORKLOAD,
        )

    def rewrite(self, path: Path, mutate) -> None:
        value = _read(path)
        mutate(value)
        _write_json(path, value)

    def test_v5_profile_passes_and_binds_all_pinned_workload_hashes(self) -> None:
        result = self.aggregate()
        self.assertTrue(result["formal_promotion_passed"])
        self.assertEqual(result["design"]["formal_profile"], "formal-v5")
        self.assertEqual(
            result["design"]["formal_workload"],
            {
                "path": str(FORMAL_V5_WORKLOAD.resolve()),
                "split_id": "live-joint-wikipedia-frozen-formal-v5",
                "file_sha256": (
                    "6b11193c8a0dbbd70f9ae4bc2c72b56737893b4d45dacd1d9970e01ca019ae31"
                ),
                "canonical_json_sha256": (
                    "7e89dea02bf2dfc5bf2b7dd2669c0d753097d5e2e351b26f018eb3df02268fbe"
                ),
                "canonical_sources_sha256": (
                    "478310accbd16ce623a4684465dd029a01efa80bfd299f3522943e90bf2cba46"
                ),
            },
        )
        self.assertEqual(
            result["design"]["config_validation"]["speculation_mode"], "visit"
        )
        self.assertTrue(
            result["formal_gates"]["formal_v5_zero_guided_json_recovery"][
                "passed"
            ]
        )
        self.assertTrue(
            result["formal_gates"][
                "formal_v5_http_attempt_gate_and_success_ledgers"
            ]["passed"]
        )
        for block in result["diagnostics"]["guided_json_recovery"].values():
            for row in block.values():
                self.assertEqual(row["task_count"], 60)
                self.assertEqual(row["parsed_call_count"], 180)
                self.assertEqual(row["recovery_count"], 0)

    def test_v5_any_reported_guided_json_recovery_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tasks"][0]["guided_json_recovery"].update(
                recovery_count=1
            ),
        )
        with self.assertRaisesRegex(ValueError, "recovery_count must be zero"):
            self.aggregate()

    def test_v5_hidden_call_recovery_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tasks"][0]["guided_json_recovery"][
                "calls"
            ][0].update(recovery_applied=True),
        )
        with self.assertRaisesRegex(ValueError, "strict-parse-only"):
            self.aggregate()

    def test_v5_guided_json_policy_version_drift_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tasks"][0]["guided_json_recovery"].update(
                policy_version="unfrozen"
            ),
        )
        with self.assertRaisesRegex(ValueError, "policy version differs"):
            self.aggregate()


class AggregateLiveJointFourCellV6Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.blocks = _make_blocks(self.root, workload_path=FORMAL_V6_WORKLOAD)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict:
        return aggregate_live_joint_four_cell(
            self.blocks,
            bootstrap_resamples=200,
            formal_workload=FORMAL_V6_WORKLOAD,
        )

    def rewrite(self, path: Path, mutate) -> None:
        value = _read(path)
        mutate(value)
        _write_json(path, value)

    def test_v6_profile_binds_code_workload_and_plain_final_contract(self) -> None:
        result = self.aggregate()
        self.assertTrue(result["formal_promotion_passed"])
        self.assertEqual(result["design"]["formal_profile"], "formal-v6")
        workload = result["design"]["formal_workload"]
        self.assertEqual(
            workload["file_sha256"],
            "44122877db66b1df4a985316c2a96b71d91d13c4e8be84affb73d405490bd43f",
        )
        self.assertEqual(
            workload["canonical_json_sha256"],
            "019fbc5177e45b4cc8cb752ccc28a7070ae1c70a1faeded787a1989dc262a96b",
        )
        self.assertEqual(
            workload["canonical_sources_sha256"],
            "e07a94c9485205e2fb864d65a6339ac5885b0821d0b2123113107bfed988f4e0",
        )
        runtime = result["design"]["v6_runtime_validation"]
        self.assertEqual(
            runtime["tool_signal_policy_module_sha256"],
            "719b34c36b5bf4f30d2a6bd4c47e37fe23fdea66a6ad7a5ea8128bdfbb50c28f",
        )
        self.assertNotEqual(
            runtime["current_tool_signal_policy_module_sha256"],
            runtime["tool_signal_policy_module_sha256"],
        )
        self.assertFalse(runtime["requires_current_module_sha_match"])
        self.assertTrue(
            result["formal_gates"]["formal_v6_zero_guided_json_recovery"][
                "passed"
            ]
        )
        self.assertTrue(
            result["formal_gates"]["formal_v6_plain_final_output_contract"][
                "passed"
            ]
        )
        for block in result["diagnostics"]["guided_json_recovery"].values():
            for row in block.values():
                self.assertEqual(row["parsed_call_count"], 120)
                self.assertEqual(row["recovery_count"], 0)
        for block in result["diagnostics"]["output_contract"].values():
            for row in block.values():
                self.assertEqual(row["output_call_count"], 180)
                self.assertEqual(row["guided_json_output_call_count"], 120)
                self.assertEqual(row["plain_text_local_wrap_call_count"], 60)

    def test_v6_recovery_or_third_guided_parse_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tasks"][0]["guided_json_recovery"].update(
                recovery_count=1
            ),
        )
        with self.assertRaisesRegex(ValueError, "recovery_count must be zero"):
            self.aggregate()

    def test_v6_call2_must_be_plain_local_wrap(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tasks"][0]["output_contract"]["calls"][
                2
            ].update(guided_json_requested=True),
        )
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            self.aggregate()

    def test_v6_final_answer_exact_url_binding_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def corrupt_url_binding(payload: dict) -> None:
            task = payload["tasks"][0]
            wrong = "0" * 64
            task["final_answer_contract"]["source_url_sha256"] = wrong
            task["output_contract"]["calls"][2]["source_url_sha256"] = wrong

        self.rewrite(target, corrupt_url_binding)
        with self.assertRaisesRegex(ValueError, "URL binding differs"):
            self.aggregate()

    def test_v6_final_answer_sha_and_count_evidence_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def corrupt_count(payload: dict) -> None:
            task = payload["tasks"][0]
            task["final_answer_contract"]["canonical_char_count"] += 1
            task["output_contract"]["calls"][2]["canonical_char_count"] += 1

        self.rewrite(target, corrupt_count)
        with self.assertRaisesRegex(ValueError, "contract counts differ"):
            self.aggregate()

    def test_v6_live_agent_sha_drift_fails_closed(self) -> None:
        for block in self.blocks:
            for path in block[1:]:
                self.rewrite(
                    path,
                    lambda payload: payload["config"].update(
                        tool_signal_policy_module_sha256="0" * 64
                    ),
                )
        with self.assertRaisesRegex(ValueError, "runtime contract differs"):
            self.aggregate()


class AggregateLiveJointFourCellV7Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.blocks = _make_blocks(self.root, workload_path=FORMAL_V7_WORKLOAD)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict:
        return aggregate_live_joint_four_cell(
            self.blocks,
            bootstrap_resamples=200,
            formal_workload=FORMAL_V7_WORKLOAD,
        )

    def rewrite(self, path: Path, mutate) -> None:
        value = _read(path)
        mutate(value)
        _write_json(path, value)

    def test_v7_profile_binds_strict_guided_final_and_allows_projection(self) -> None:
        result = self.aggregate()
        self.assertTrue(result["formal_promotion_passed"])
        self.assertEqual(result["design"]["formal_profile"], "formal-v7")
        workload = result["design"]["formal_workload"]
        self.assertEqual(
            workload["file_sha256"],
            "cbf143f59f4d2a05650df68d8fa6f00d7471964a4b257d26dd092ba90c40e6c8",
        )
        self.assertEqual(
            workload["canonical_json_sha256"],
            "09e88d67f4aeb1994a566e11678fceb8f374f3b86f667da112f901209e0ef393",
        )
        self.assertEqual(
            workload["canonical_sources_sha256"],
            "710cc4f8d62f6c2b8ab78ec3d61d79be1ba7db25f47559accd407e7d0ddc810c",
        )
        runtime = result["design"]["v7_runtime_validation"]
        self.assertEqual(
            runtime["tool_signal_policy_module_sha256"],
            "6fa736aa4e56657874834841c8a60b18c53e31f48ffbe741cc2e93f1c750432f",
        )
        self.assertNotEqual(
            runtime["current_tool_signal_policy_module_sha256"],
            runtime["tool_signal_policy_module_sha256"],
        )
        self.assertFalse(runtime["requires_current_module_sha_match"])
        self.assertTrue(
            result["formal_gates"]["formal_v7_zero_guided_json_recovery"][
                "passed"
            ]
        )
        self.assertTrue(
            result["formal_gates"][
                "formal_v7_strict_guided_final_output_contract"
            ]["passed"]
        )
        for block in result["diagnostics"][
            "strict_guided_final_contract"
        ].values():
            for row in block.values():
                self.assertEqual(row["output_call_count"], 180)
                self.assertEqual(row["guided_tool_call_count"], 120)
                self.assertEqual(row["strict_guided_final_call_count"], 60)
                self.assertEqual(row["recovery_applied_count"], 0)
                self.assertEqual(row["local_projection_count"], 1)
                self.assertEqual(row["word_projection_count"], 1)
                self.assertTrue(row["projection_allowed"])

    def test_v7_tool_guided_recovery_fails_closed(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tasks"][0]["guided_json_recovery"].update(
                recovery_count=1
            ),
        )
        with self.assertRaisesRegex(ValueError, "recovery_count must be zero"):
            self.aggregate()

    def test_v7_tool_recovery_record_must_exactly_mirror_output(self) -> None:
        self.rewrite(
            self.blocks[0][4],
            lambda payload: payload["tasks"][0]["guided_json_recovery"][
                "calls"
            ][0].pop("mode"),
        )
        with self.assertRaisesRegex(ValueError, "recovery telemetry differs"):
            self.aggregate()

    def test_v7_final_recovery_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def add_recovery(payload: dict) -> None:
            task = payload["tasks"][0]
            task["final_answer_contract"]["recovery_applied"] = True
            task["output_contract"]["calls"][2]["recovery_applied"] = True

        self.rewrite(target, add_recovery)
        with self.assertRaisesRegex(ValueError, "strict final contract differs"):
            self.aggregate()

    def test_v7_schema_sha_drift_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def corrupt_schema(payload: dict) -> None:
            task = payload["tasks"][0]
            task["final_answer_contract"]["schema_sha256"] = "0" * 64
            task["output_contract"]["calls"][2]["schema_sha256"] = "0" * 64

        self.rewrite(target, corrupt_schema)
        with self.assertRaisesRegex(ValueError, "final schema SHA differs"):
            self.aggregate()

    def test_v7_final_policy_version_drift_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def corrupt_policy(payload: dict) -> None:
            task = payload["tasks"][0]
            task["final_answer_contract"]["schema_policy_version"] = "wrong-v1"
            task["output_contract"]["calls"][2][
                "schema_policy_version"
            ] = "wrong-v1"

        self.rewrite(target, corrupt_policy)
        with self.assertRaisesRegex(ValueError, "strict final contract differs"):
            self.aggregate()

    def test_v7_exact_url_binding_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def corrupt_url(payload: dict) -> None:
            task = payload["tasks"][0]
            task["final_answer_contract"]["source_url_sha256"] = "0" * 64
            task["output_contract"]["calls"][2]["source_url_sha256"] = "0" * 64

        self.rewrite(target, corrupt_url)
        with self.assertRaisesRegex(ValueError, "final URL SHA differs"):
            self.aggregate()

    def test_v7_projection_invariant_drift_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def corrupt_projection(payload: dict) -> None:
            task = payload["tasks"][0]
            task["final_answer_contract"]["local_projection_applied"] = False
            task["output_contract"]["calls"][2][
                "local_projection_applied"
            ] = False

        self.rewrite(target, corrupt_projection)
        with self.assertRaisesRegex(ValueError, "projection invariants differ"):
            self.aggregate()

    def test_v7_agent_sha_drift_fails_closed(self) -> None:
        for block in self.blocks:
            for path in block[1:]:
                self.rewrite(
                    path,
                    lambda payload: payload["config"].update(
                        tool_signal_policy_module_sha256="0" * 64
                    ),
                )
        with self.assertRaisesRegex(ValueError, "runtime contract differs"):
            self.aggregate()


class AggregateLiveJointFourCellV8Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.blocks = _make_blocks(self.root, workload_path=FORMAL_V8_WORKLOAD)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict:
        return aggregate_live_joint_four_cell(
            self.blocks,
            bootstrap_resamples=200,
            formal_workload=FORMAL_V8_WORKLOAD,
        )

    def rewrite(self, path: Path, mutate) -> None:
        value = _read(path)
        mutate(value)
        _write_json(path, value)

    def rewrite_timeline(self, result_path: Path, mutate) -> None:
        payload = _read(result_path)
        evidence = payload["raw_evidence"]["queue_timeline"]
        timeline_path = Path(evidence["path"])
        rows = [
            json.loads(line)
            for line in timeline_path.read_text(encoding="utf-8").splitlines()
        ]
        mutate(rows)
        _write_jsonl(timeline_path, rows)
        evidence["sha256"] = _sha(timeline_path)
        _write_json(result_path, payload)

    def test_v8_profile_passes_with_80_above_64_and_fixed_final_tokens(self) -> None:
        result = self.aggregate()
        self.assertTrue(result["formal_promotion_passed"])
        self.assertEqual(result["failed_gate_names"], [])
        self.assertEqual(result["design"]["formal_profile"], "formal-v8")
        self.assertEqual(result["design"]["independent_source_count"], 80)
        self.assertEqual(result["design"]["tasks_per_cell_per_block"], 80)
        self.assertEqual(
            result["design"]["logical_llm_requests_per_cell_per_block"], 240
        )
        self.assertEqual(
            result["design"]["authoritative_tool_commits_per_cell_per_block"],
            160,
        )
        load = result["design"]["formal_load"]
        self.assertEqual(load["max_active_tasks"], 80)
        self.assertGreater(load["max_active_tasks"], 64)
        self.assertLess(load["max_active_tasks"], load["vllm_max_num_seqs"])
        self.assertEqual(load["max_dual_queue_adjacent_sample_gap_s"], 0.5)
        self.assertTrue(
            result["formal_gates"][
                "formal_v8_offered_concurrency_above_64_below_max_num_seqs"
            ]["passed"]
        )
        for block in result["blocks"].values():
            proof = block["A"]["load_qualification"]
            self.assertGreaterEqual(
                proof["native_waiting_below_cap_fraction"], 0.05
            )
            self.assertGreaterEqual(
                proof["authoritative_tool_queue_sample_fraction"], 0.05
            )
            self.assertGreaterEqual(proof["dual_queue_pressure_sample_count"], 10)
            self.assertGreaterEqual(
                proof["longest_consecutive_dual_queue_pressure_elapsed_s"], 1.0
            )
        self.assertTrue(
            result["formal_gates"][
                "formal_v8_fourteen_canaries_skip_visit_speculation_before_enqueue"
            ]["passed"]
        )
        for block in result["diagnostics"]["canary_pre_enqueue_skip"].values():
            self.assertEqual(block["A"]["speculative_visit_record_count"], 0)
            self.assertEqual(block["E"]["speculative_visit_record_count"], 0)
            self.assertEqual(block["B"]["speculative_visit_record_count"], 66)
            self.assertEqual(block["F"]["speculative_visit_record_count"], 66)
        fixed = result["diagnostics"]["fixed_final_completion_contract"]
        for block in fixed.values():
            for cell in block.values():
                self.assertEqual(cell["task_count"], 80)
                self.assertEqual(cell["exact_completion_token_task_count"], 80)
                self.assertTrue(cell["all_completion_tokens_exact"])
                self.assertEqual(cell["semantic_json_object_count"], 80)
                self.assertEqual(cell["ascii_space_tail_count"], 80)
        for effect in ("A_to_B", "E_to_F", "A_to_F"):
            self.assertTrue(
                result["formal_gates"][
                    f"{effect}_completion_token_difference_below_1pct"
                ]["passed"]
            )
        self.assertEqual(result["effects"]["E_to_F"]["faster_source_count"], 80)
        self.assertTrue(
            result["formal_gates"][
                "E_to_F_LLM_component_not_more_than_1pct_faster"
            ]["passed"]
        )
        self.assertTrue(
            result["formal_gates"][
                "E_to_F_tool_exposed_wait_explains_net_saving"
            ]["passed"]
        )

    def test_v8_one_non_192_final_fails_exact_final_gate(self) -> None:
        target = self.blocks[0][4]

        def mutate(payload: dict) -> None:
            task_id = payload["tasks"][0]["task_id"]
            event = next(
                row
                for row in payload["llm_events"]
                if row["task_id"] == task_id and row["call_index"] == 2
            )
            event["usage"]["completion_tokens"] = 191
            event["usage"]["total_tokens"] -= 1
            task = payload["tasks"][0]
            task["completion_tokens"] -= 1
            task["final_answer_contract"]["total_completion_tokens"] = 191
            task["final_answer_contract"]["padding_token_count"] = 151
            task["output_contract"]["calls"][2][
                "total_completion_tokens"
            ] = 191
            task["output_contract"]["calls"][2]["padding_token_count"] = 151
            payload["vllm_metric_deltas"]["vllm:generation_tokens_total"] -= 1

        self.rewrite(target, mutate)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "formal_v8_call2_completion_tokens_exact", result["failed_gate_names"]
        )
        self.assertNotIn(
            "E_to_F_completion_token_difference_below_1pct",
            result["failed_gate_names"],
        )

    def test_v8_non_ascii_space_tail_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def mutate(payload: dict) -> None:
            event = next(
                row for row in payload["llm_events"] if row["call_index"] == 2
            )
            event["response"] = event["response"][:-1] + "\n"
            event["response_sha256"] = hashlib.sha256(
                event["response"].encode()
            ).hexdigest()
            task = payload["tasks"][0]
            semantic_end = event["response"].index("}") + 1
            tail = event["response"][semantic_end:]
            for contract in (
                task["final_answer_contract"],
                task["output_contract"]["calls"][2],
            ):
                contract["raw_sha256"] = event["response_sha256"]
                contract["padding_sha256"] = hashlib.sha256(
                    tail.encode()
                ).hexdigest()

        self.rewrite(target, mutate)
        with self.assertRaisesRegex(ValueError, "ASCII spaces only"):
            self.aggregate()

    def test_v8_dynamic_grammar_sha_drift_fails_closed(self) -> None:
        target = self.blocks[0][4]

        def mutate(payload: dict) -> None:
            task = payload["tasks"][0]
            task["final_answer_contract"]["grammar_sha256"] = "0" * 64
            task["output_contract"]["calls"][2]["grammar_sha256"] = "0" * 64

        self.rewrite(target, mutate)
        with self.assertRaisesRegex(ValueError, "schema/grammar/URL SHA differs"):
            self.aggregate()

    def test_v8_fixed_final_config_policy_drift_fails_closed(self) -> None:
        for block in self.blocks:
            for target in block[1:]:
                self.rewrite(
                    target,
                    lambda payload: payload["config"].update(
                        final_answer_grammar_policy_version="unfrozen-v1"
                    ),
                )
        with self.assertRaisesRegex(ValueError, "runtime contract differs"):
            self.aggregate()

    def test_v8_dual_pressure_must_have_ten_samples_and_one_second_streak(self) -> None:
        target = self.blocks[0][1]

        def split_streak(rows: list[dict]) -> None:
            rows[5]["tool_queued_authoritative"] = 0
            rows[5]["llm_waiting"] = 0.0
            rows[10]["tool_queued_authoritative"] = 1
            rows[10]["llm_waiting"] = 2.0

        self.rewrite_timeline(target, split_streak)
        result = self.aggregate()
        proof = result["blocks"]["block-1"]["A"]["load_qualification"]
        self.assertEqual(proof["dual_queue_pressure_sample_count"], 10)
        self.assertLess(
            proof["longest_consecutive_dual_queue_pressure_elapsed_s"], 1.0
        )
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "all_A_blocks_have_native_llm_and_authoritative_tool_queue",
            result["failed_gate_names"],
        )

    def test_v8_dual_pressure_streak_resets_across_sample_gap(self) -> None:
        target = self.blocks[0][1]

        def insert_gap(rows: list[dict]) -> None:
            for row in rows[5:]:
                row["monotonic_s"] += 0.4
                row["wall_s"] += 0.4

        self.rewrite_timeline(target, insert_gap)
        result = self.aggregate()
        proof = result["blocks"]["block-1"]["A"]["load_qualification"]
        self.assertEqual(proof["dual_queue_pressure_sample_count"], 10)
        self.assertAlmostEqual(
            proof["maximum_adjacent_simultaneous_monotonic_gap_s"], 0.6
        )
        self.assertAlmostEqual(
            proof["maximum_adjacent_simultaneous_wall_gap_s"], 0.6
        )
        self.assertEqual(proof["simultaneous_gap_reset_count"], 1)
        self.assertLess(proof["longest_continuous_dual_span_s"], 1.0)
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "all_A_blocks_have_native_llm_and_authoritative_tool_queue",
            result["failed_gate_names"],
        )

    def test_v8_llm_component_speedup_above_one_percent_fails_gate(self) -> None:
        for block in self.blocks:
            target = block[4]

            def speed_up_llm(payload: dict) -> None:
                for task in payload["tasks"]:
                    task["llm_duration_s"] = 0.9
                for event in payload["llm_events"]:
                    event["duration_s"] = 0.3

            self.rewrite(target, speed_up_llm)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "E_to_F_LLM_component_not_more_than_1pct_faster",
            result["failed_gate_names"],
        )

    def test_v8_tool_wait_must_explain_all_net_e2e_saving(self) -> None:
        for block in self.blocks:
            target = block[3]

            def remove_tool_saving(payload: dict) -> None:
                for task in payload["tasks"]:
                    for tool in task["tools"]:
                        tool["exposed_wait_s"] = 0.4
                for record in payload["tool_attempt_records"]:
                    if record["committed"] is True:
                        record["exposed_wait_s"] = 0.4

            self.rewrite(target, remove_tool_saving)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "E_to_F_tool_exposed_wait_explains_net_saving",
            result["failed_gate_names"],
        )


class AggregateLiveJointFourCellV9Tests(unittest.TestCase):
    def setUp(self) -> None:
        parent = REPOSITORY_ROOT / "reproduction/artifacts/live_joint"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".formal-v9-aggregate-test-",
            dir=parent,
        )
        self.root = Path(self.temporary.name)
        self.blocks = _make_blocks(self.root, workload_path=FORMAL_V9_WORKLOAD)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def aggregate(self) -> dict:
        return aggregate_live_joint_four_cell(
            self.blocks,
            bootstrap_resamples=200,
            formal_workload=FORMAL_V9_WORKLOAD,
        )

    @staticmethod
    def _relative(path: Path) -> str:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))

    def rewrite_result(self, path: Path, mutate) -> None:
        payload = _read(path)
        mutate(payload)
        _write_json(path, payload)
        manifest_path = path.parent.parent / "cell_manifest.json"
        manifest = _read(manifest_path)
        manifest["evidence"][self._relative(path)] = _sha(path)
        _write_json(manifest_path, manifest)

    def rewrite_effective(self, result_path: Path, mutate) -> None:
        effective_path = result_path.parent.parent / "effective_config.json"
        effective = _read(effective_path)
        mutate(effective)
        _write_json(effective_path, effective)
        manifest_path = result_path.parent.parent / "cell_manifest.json"
        manifest = _read(manifest_path)
        manifest["evidence"][self._relative(effective_path)] = _sha(effective_path)
        _write_json(manifest_path, manifest)

    def test_v9_profile_passes_with_exact_selection_and_zero_transport_noise(self) -> None:
        result = self.aggregate()
        self.assertTrue(result["formal_promotion_passed"])
        self.assertEqual(result["failed_gate_names"], [])
        self.assertEqual(result["design"]["formal_profile"], "formal-v9")
        self.assertEqual(result["design"]["bootstrap_seed"], 20260817)
        workload = result["design"]["formal_workload"]
        self.assertEqual(
            workload["file_sha256"],
            "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20",
        )
        self.assertEqual(
            workload["canonical_json_sha256"],
            "de588fcbd46c1181156f5a6e49e0264c785c00c43e0d8c2a62698fb6217e3ce7",
        )
        self.assertEqual(
            workload["canonical_sources_sha256"],
            "750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c",
        )
        load = result["design"]["formal_load"]
        self.assertEqual(load["visit_min_start_interval_s"], 2.5)
        self.assertEqual(load["min_speculative_tool_workers"], 0)
        self.assertEqual(load["max_active_tasks"], 80)
        self.assertLess(64, load["max_active_tasks"])
        self.assertLess(load["max_active_tasks"], load["vllm_max_num_seqs"])
        selection = result["design"]["v9_development_selection"]
        self.assertEqual(selection["selected_policy"], "F0")
        self.assertEqual(selection["selected_visit_interval_s"], 2.5)
        self.assertEqual(selection["selected_min_speculative_tool_workers"], 0)
        self.assertEqual(selection["maximum_observed_http_retries_per_cell"], 0)
        self.assertIs(
            selection["zero_wasted_speculative_service_required"], True
        )
        self.assertEqual(
            selection["live_broker_sha256"],
            "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27",
        )
        for gate in (
            "formal_v9_frozen_development_selection_and_cell_provenance",
            "formal_v9_physical_visit_attempt_gate_2p5s",
            "formal_v9_every_block_cell_zero_http_retry",
            "formal_v9_every_block_cell_zero_wasted_speculative_service",
            "formal_v9_offered_concurrency_above_64_below_max_num_seqs",
            "E_to_F_mean_reduction",
            "A_to_F_mean_reduction",
            "E_to_F_LLM_component_not_more_than_1pct_faster",
            "E_to_F_tool_exposed_wait_explains_net_saving",
        ):
            self.assertTrue(result["formal_gates"][gate]["passed"], gate)
        provenance = result["design"]["v9_cell_provenance"]
        self.assertEqual(sum(len(block) for block in provenance.values()), 12)
        for block in result["blocks"].values():
            for cell in block.values():
                physical = cell["physical_validation"]
                self.assertEqual(physical["retried_physical_job_count"], 0)
                self.assertEqual(physical["failed_physical_job_count"], 0)
                self.assertEqual(physical["wasted_speculative_worker_s"], 0.0)

    def test_v9_effective_selection_provenance_drift_fails_closed(self) -> None:
        self.rewrite_effective(
            self.blocks[0][4],
            lambda effective: effective["formal_v9_selection"].update(
                live_broker_sha256="0" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "effective config differs"):
            self.aggregate()

    def test_v9_development_f0_minimum_must_be_observed_in_both_blocks(self) -> None:
        original_selection = _read(
            aggregate_module.V9_STRICT_DEVELOPMENT_SELECTION
        )
        f0_key = next(
            key
            for key in original_selection["common_code_and_config_identity"][
                "cells"
            ]
            if key.endswith("/F0")
        )
        original_selection["common_code_and_config_identity"]["cells"][f0_key][
            "min_speculative_tool_workers"
        ] = 1
        selection_path = self.root / "mutated_strict_selection.json"
        _write_json(selection_path, original_selection)

        completed = _read(aggregate_module.V9_COMPLETED_SCREEN)
        completed["strict_development_selection"] = {
            "path": self._relative(selection_path),
            "sha256": _sha(selection_path),
        }
        completed_path = self.root / "mutated_completed_screen.json"
        _write_json(completed_path, completed)
        with (
            mock.patch.object(
                aggregate_module,
                "V9_STRICT_DEVELOPMENT_SELECTION",
                selection_path,
            ),
            mock.patch.object(
                aggregate_module,
                "V9_STRICT_DEVELOPMENT_SELECTION_SHA256",
                _sha(selection_path),
            ),
            mock.patch.object(
                aggregate_module,
                "V9_COMPLETED_SCREEN",
                completed_path,
            ),
            mock.patch.object(
                aggregate_module,
                "V9_COMPLETED_SCREEN_SHA256",
                _sha(completed_path),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "strict development selection is not F0/2.5s"
            ):
                self.aggregate()

    def test_v9_manifest_result_sha_drift_fails_closed(self) -> None:
        target = self.blocks[0][4]
        manifest_path = target.parent.parent / "cell_manifest.json"
        manifest = _read(manifest_path)
        manifest["evidence"][self._relative(target)] = "0" * 64
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "manifest evidence SHA differs"):
            self.aggregate()

    def test_v9_any_actual_retry_fails_exact_zero_gate(self) -> None:
        target = self.blocks[0][1]

        def add_controlled_retry(payload: dict) -> None:
            record = next(
                row
                for row in payload["tool_attempt_records"]
                if row["tool"] == "search" and row["committed"] is True
            )
            started = record["started_at"]
            record["http_attempts"] = 2
            record["http_attempt_log"] = [
                {
                    "request_index": 0,
                    "attempt": 1,
                    "status": 429,
                    "error_type": None,
                    "retried": True,
                    "started_monotonic_s": started,
                    "start_gate_wait_s": 0.0,
                    "retry_backoff_s": 1.0,
                },
                {
                    "request_index": 0,
                    "attempt": 2,
                    "status": 200,
                    "error_type": None,
                    "retried": False,
                    "started_monotonic_s": started + 1.0,
                    "start_gate_wait_s": 0.0,
                    "retry_backoff_s": 0.0,
                },
            ]
            record["service_s"] = 2.0
            record["finished_at"] = started + 2.0
            record["finish"] = record["finished_at"]
            task = next(
                row for row in payload["tasks"]
                if row["task_id"] == record["session_id"]
            )
            task["tools"][0]["service_s"] = 2.0

        self.rewrite_result(target, add_controlled_retry)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "formal_v9_every_block_cell_zero_http_retry",
            result["failed_gate_names"],
        )

    def test_v9_any_wasted_speculative_service_fails_exact_zero_gate(self) -> None:
        target = self.blocks[0][4]

        def add_wasted_speculative_visit(payload: dict) -> None:
            records = payload["tool_attempt_records"]
            original = next(
                row
                for row in records
                if row["tool"] == "visit"
                and row["speculative"] is True
                and row["canary"] is False
            )
            wasted = copy.deepcopy(original)
            start = max(
                row["finished_at"]
                for row in records
                if row.get("finished_at") is not None
            ) + 2.5
            wasted.update(
                {
                    "job_id": max(row["job_id"] for row in records) + 1,
                    "invocation_id": original["invocation_id"] + "-wasted",
                    "authoritative": False,
                    "committed": False,
                    "cancelled": False,
                    "outcome": "expired",
                    "exact_match": False,
                    "source": "completed_unclaimed_expired",
                    "queue_enter": start - 0.2,
                    "queue_enter_at": start - 0.2,
                    "admitted_at": start - 0.2,
                    "start": start,
                    "started_at": start,
                    "confirmation": None,
                    "authoritative_confirmation_at": None,
                    "finish": start + 0.5,
                    "finished_at": start + 0.5,
                    "queue_s": 0.2,
                    "service_s": 0.5,
                    "exposed_wait_s": 0.0,
                    "saved_service_s": 0.0,
                    "http_attempts": 1,
                    "http_attempt_log": [
                        {
                            "request_index": 0,
                            "attempt": 1,
                            "status": 200,
                            "error_type": None,
                            "retried": False,
                            "started_monotonic_s": start,
                            "start_gate_wait_s": 0.0,
                            "retry_backoff_s": 0.0,
                        }
                    ],
                    "rate_limit_eligible_at": start,
                    "rate_limit_next_eligible_at": start + 2.5,
                    "rate_limit_wait_s": 0.0,
                }
            )
            records.append(wasted)
            payload["broker_final_snapshot"]["stats"][
                "wasted_speculative_service_s"
            ] = 0.5

        self.rewrite_result(target, add_wasted_speculative_visit)
        result = self.aggregate()
        self.assertFalse(result["formal_promotion_passed"])
        self.assertIn(
            "formal_v9_every_block_cell_zero_wasted_speculative_service",
            result["failed_gate_names"],
        )

    def test_v9_interval_or_f0_minimum_drift_fails_closed(self) -> None:
        for block in self.blocks:
            for target in block[1:]:
                self.rewrite_result(
                    target,
                    lambda payload: payload["config"].update(
                        min_speculative_tool_workers=1
                    ),
                )
        with self.assertRaisesRegex(ValueError, "runtime contract differs"):
            self.aggregate()


if __name__ == "__main__":
    unittest.main()
