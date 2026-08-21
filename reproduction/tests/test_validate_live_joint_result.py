from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from validate_live_joint_result import (  # noqa: E402
    PROTOCOL_PATH,
    validate_live_joint_result,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _evidence_ref(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": _sha(path)}


def _make_cell(
    root: Path,
    *,
    cell_id: str,
    source_ids: list[str],
    blocks: list[str],
    copies: int,
    duration_s: float,
    prefix_policy: str,
    prefix_hit: float,
) -> dict:
    llm_policy = {
        "A": "fcfs_native",
        "B": "fcfs_native",
        "E": "joint_physical_kv",
        "F": "joint_physical_kv",
    }[cell_id]
    tool_policy = {
        "A": "demand_only",
        "B": "resource_aware_speculation",
        "E": "demand_only",
        "F": "resource_aware_speculation",
    }[cell_id]
    spec_on = tool_policy == "resource_aware_speculation"
    cell_dir = root / "runs" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict] = []
    llm_rows: list[dict] = []
    tool_rows: list[dict] = []
    prefix_rows: list[dict] = []
    resource_rows: list[dict] = []
    ordinal = 0
    for block_id in blocks:
        prefix_rows.append(
            {"block_id": block_id, "timestamp": float(len(prefix_rows)), "gpu_prefix_hit_ratio": prefix_hit}
        )
        for sample_index in range(10):
            under_pressure = sample_index == 0
            resource_rows.append(
                {
                    "block_id": block_id,
                    "timestamp": float(sample_index),
                    "llm_running": max(0, len(source_ids) * copies - 1),
                    "llm_waiting": 1 if under_pressure else 0,
                    "tool_running_authoritative": 2,
                    "tool_running_speculative": 0,
                    "tool_queued_authoritative": 1 if under_pressure else 0,
                    "tool_queued_speculative": 0,
                }
            )
        for source_id in source_ids:
            for copy in range(copies):
                task_id = f"{cell_id}:{block_id}:{source_id}:{copy}"
                task_start = ordinal * 0.01
                task_rows.append(
                    {
                        "task_instance_id": task_id,
                        "source_id": source_id,
                        "block_id": block_id,
                        "started_at": task_start,
                        "finished_at": task_start + duration_s,
                        "success": True,
                        "logical_llm_requests": 3,
                        "logical_tool_calls": 2,
                    }
                )
                for call_index in range(3):
                    submitted = task_start + call_index * 0.2
                    llm_rows.append(
                        {
                            "request_id": f"llm:{task_id}:{call_index}",
                            "task_instance_id": task_id,
                            "source_id": source_id,
                            "call_index": call_index,
                            "submitted_at": submitted,
                            "started_at": submitted + 0.01,
                            "finished_at": submitted + 0.11,
                            "http_status": 200,
                            "success": True,
                            "attempts": 1,
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "prefix_cached_tokens": int(100 * prefix_hit),
                        }
                    )
                base = ordinal * 10.0
                for tool_index, tool_name in enumerate(("search", "visit")):
                    canary = tool_index == 0
                    speculative = spec_on and not canary
                    queue_enter = base + tool_index * 2.0
                    started = queue_enter + 0.1
                    finished = started + 1.0
                    confirmation = queue_enter if not speculative else started + 0.6
                    digest = f"digest:{block_id}:{source_id}:{copy}:{tool_index}"
                    tool_rows.append(
                        {
                            "job_id": f"job:{task_id}:{tool_index}",
                            "logical_call_id": f"logical:{task_id}:{tool_index}",
                            "invocation_id": f"invocation:{task_id}:{tool_index}",
                            "invocation_digest": digest,
                            "result_digest": f"result:{digest}",
                            "task_instance_id": task_id,
                            "session_id": task_id,
                            "source_id": source_id,
                            "tool": tool_name,
                            "admitted": True,
                            "speculative": speculative,
                            "authoritative": True,
                            "committed": True,
                            "speculation_eligible": not canary,
                            "canary": canary,
                            "admitted_at": queue_enter,
                            "queue_enter_at": queue_enter,
                            "started_at": started,
                            "authoritative_confirmation_at": confirmation,
                            "finished_at": finished,
                            "outcome": "success",
                            # Broker-native exact_match denotes a speculative
                            # reuse, not correctness of a direct execution.
                            "exact_match": speculative,
                            "source": "promoted" if speculative else "executed",
                            "cancelled": False,
                            "cross_session_commit": False,
                            "worker_pool": f"pool:{cell_id}:{block_id}",
                            "worker_id": 0,
                            "queue_s": 0.1,
                            "service_s": 1.0,
                            "saved_service_s": 0.6 if speculative else 0.0,
                            "response_status": 200,
                            "bytes_read": 1024,
                            "http_attempts": 1,
                            "backend": (
                                "bing_html_search"
                                if tool_name == "search"
                                else "r.jina.ai"
                            ),
                            "request_host": (
                                "www.bing.com"
                                if tool_name == "search"
                                else "r.jina.ai"
                            ),
                            "transport_identity_source": "actual",
                        }
                    )
                ordinal += 1

    event_files = {
        "task_events": task_rows,
        "llm_events": llm_rows,
        "tool_events": tool_rows,
        "resource_samples": resource_rows,
        "prefix_samples": prefix_rows,
    }
    evidence: dict[str, dict[str, str]] = {}
    for kind, rows in event_files.items():
        path = cell_dir / f"{kind}.jsonl"
        _write_jsonl(path, rows)
        evidence[kind] = _evidence_ref(root, path)
    for kind in ("frozen_config", "server_log", "tool_server_log"):
        path = cell_dir / f"{kind}.txt"
        path.write_text(f"{cell_id} {kind}\n", encoding="utf-8")
        evidence[kind] = _evidence_ref(root, path)
    return {
        "policy": {
            "llm_scheduler": llm_policy,
            "tool_scheduler": tool_policy,
            "prefix_policy": prefix_policy,
            "speculation_scope": "visit_only" if spec_on else "none",
        },
        "fresh_server_block_ids": blocks,
        "server_instance_by_block": {
            block: f"server:{cell_id}:{block}" for block in blocks
        },
        "result_cache_warm_start": False,
        "engine": {
            "max_num_seqs": len(source_ids) * copies + 1,
            "max_active_sessions": len(source_ids) * copies,
        },
        "tool_runtime": {
            "worker_pool_by_block": {
                block: f"pool:{cell_id}:{block}" for block in blocks
            },
            "worker_capacity": 4,
            "per_tool_capacity": {"search": 4, "visit": 2},
            "max_speculative_workers": 2,
            "max_speculative_pending": 32,
            "speculative_ttl_s": 60.0,
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
        },
        "evidence": evidence,
    }


def _make_prefix_ablation(root: Path, formal_sources: list[str]) -> dict:
    tune_sources = [f"prefix-tune-{index:02d}" for index in range(12)]
    blocks = ["prefix-block-1"]
    cells = {}
    for cell_id, policy, duration, hit in (
        ("P0", "disabled", 100.0, 0.0),
        ("P1", "native", 90.0, 0.20),
        ("P2", "explicit_affinity", 87.0, 0.25),
    ):
        cell_dir = root / "prefix" / cell_id
        evidence = {}
        task_rows = []
        for source_index, source_id in enumerate(tune_sources):
            started = float(source_index)
            task_rows.append(
                {
                    "task_instance_id": f"{cell_id}:prefix-block-1:{source_id}:0",
                    "source_id": source_id,
                    "block_id": "prefix-block-1",
                    "started_at": started,
                    "finished_at": started + duration,
                    "success": True,
                    "logical_llm_requests": 3,
                    "logical_tool_calls": 2,
                }
            )
        prefix_rows = [
            {
                "block_id": "prefix-block-1",
                "timestamp": 0.0,
                "gpu_prefix_hit_ratio": hit,
            }
        ]
        for kind, rows in (
            ("task_events", task_rows),
            ("prefix_samples", prefix_rows),
        ):
            path = cell_dir / f"{kind}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(path, rows)
            evidence[kind] = _evidence_ref(root, path)
        for kind in ("frozen_config", "server_log"):
            path = cell_dir / f"{kind}.txt"
            path.write_text(f"{cell_id} {kind}\n", encoding="utf-8")
            evidence[kind] = _evidence_ref(root, path)
        cells[cell_id] = {
            "prefix_policy": policy,
            "fresh_server_block_ids": blocks,
            "server_instance_by_block": {
                "prefix-block-1": f"prefix-server:{cell_id}:prefix-block-1"
            },
            "evidence": evidence,
        }
    return {
        "source_ids": tune_sources,
        "block_ids": blocks,
        "copies_per_source": 1,
        "formal_source_overlap_count": len(set(tune_sources) & set(formal_sources)),
        "selected_policy": "explicit_affinity",
        "cells": cells,
    }


def _make_payload(root: Path, *, stage: str, weak_gain: bool = False) -> dict:
    if stage == "formal":
        source_ids = [f"heldout-{index:02d}" for index in range(60)]
        block_specs = [
            ("block-1", ["A", "B", "E", "F"]),
            ("block-2", ["B", "A", "F", "E"]),
            ("block-3", ["A", "B", "F", "E"]),
        ]
        durations = {"A": 100.0, "B": 98.0, "E": 80.0, "F": 72.0}
        prefix_policy = "explicit_affinity"
    else:
        source_ids = [f"tune-{index:02d}" for index in range(12)]
        block_specs = [("screen-1", ["A", "B", "E", "F"])]
        durations = {"A": 100.0, "B": 98.0, "E": 90.0, "F": 89.0 if weak_gain else 85.0}
        prefix_policy = "native"
    blocks = [block for block, _ in block_specs]
    workload_path = root / "workload.json"
    split_role = "heldout" if stage == "formal" else "tune"
    workload_sources = []
    for source_id in source_ids:
        workload_sources.append(
            {
                "source_id": source_id,
                "question": f"Question for {source_id}?",
                "language": "en",
                "prefix_group_id": "system-v1",
                "system_prompt_sha256": hashlib.sha256(b"system-v1").hexdigest(),
                "steps": [
                    {
                        "step_index": 0,
                        "kind": "llm",
                        "request_template_sha256": hashlib.sha256(
                            f"{source_id}:llm:0".encode()
                        ).hexdigest(),
                    },
                    {
                        "step_index": 1,
                        "kind": "search",
                        "arguments": {"query": source_id},
                    },
                    {
                        "step_index": 2,
                        "kind": "llm",
                        "request_template_sha256": hashlib.sha256(
                            f"{source_id}:llm:1".encode()
                        ).hexdigest(),
                    },
                    {
                        "step_index": 3,
                        "kind": "visit",
                        "url_from": {
                            "search_step_index": 1,
                            "heldout_result_rank": 1,
                        },
                    },
                    {
                        "step_index": 4,
                        "kind": "llm",
                        "request_template_sha256": hashlib.sha256(
                            f"{source_id}:llm:2".encode()
                        ).hexdigest(),
                    },
                ],
            }
        )
    workload_path.write_text(
        json.dumps(
            {
                "schema": "paste_repro.live_joint_workload",
                "version": 1,
                "split_id": f"test-{stage}-v1",
                "split_role": split_role,
                "formal_eligible": split_role == "heldout",
                "sources": workload_sources,
            }
        ),
        encoding="utf-8",
    )
    cells = {
        cell_id: _make_cell(
            root,
            cell_id=cell_id,
            source_ids=source_ids,
            blocks=blocks,
            copies=1,
            duration_s=durations[cell_id],
            prefix_policy="native" if cell_id in {"A", "B"} else prefix_policy,
            prefix_hit=0.20 if cell_id in {"A", "B"} else 0.25,
        )
        for cell_id in ("A", "B", "E", "F")
    }
    payload = {
        "schema": "paste_repro.live_joint_experiment",
        "version": 1,
        "stage": stage,
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "runtime": {
            "backend_mode": "external_live",
            "search_backend": "bing_html_search",
            "visit_backend": "r_jina_ai",
            "live_llm_http": True,
            "live_search_http": True,
            "live_visit_http": True,
            "shared_process_wide_tool_pool": True,
            "exact_invocation_matching": True,
            "frozen_call_graph": True,
            "baseline_only_load_selection": True,
            "recorded_wait_replay": False,
            "synthetic_tool_sleep": False,
            "future_information_used": False,
            "cross_cell_tool_cache": False,
            "generated_text_changes_tool_plan": False,
        },
        "workload": {
            "manifest": _evidence_ref(root, workload_path),
            "source_ids": source_ids,
            "split_role": split_role,
            "tuning_source_overlap_count": 0,
            "copies_per_source": 1,
            "max_active_sessions": len(source_ids),
        },
        "blocks": [
            {"block_id": block_id, "cell_order": order}
            for block_id, order in block_specs
        ],
        "cells": cells,
    }
    if stage == "formal":
        payload["prefix_ablation"] = _make_prefix_ablation(root, source_ids)
    return payload


def _mark_tool_retries(
    root: Path,
    payload: dict,
    *,
    cell_id: str,
    count: int,
    block_id: str | None = None,
) -> None:
    entry = payload["cells"][cell_id]["evidence"]["tool_events"]
    path = root / entry["path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [
        row
        for row in rows
        if row["tool"] == "search"
        and row["committed"] is True
        and (block_id is None or f":{block_id}:" in row["task_instance_id"])
    ][:count]
    if len(selected) != count:
        raise AssertionError("fixture does not contain enough committed search rows")
    for row in selected:
        row["http_attempts"] = 2
        row["service_s"] += 1.0
        row["finished_at"] += 1.0
    _write_jsonl(path, rows)
    entry["sha256"] = _sha(path)


class LiveJointValidatorTests(unittest.TestCase):
    def test_screening_passes_with_incremental_live_gain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = validate_live_joint_result(
                _make_payload(root, stage="screening"),
                stage="screening",
                repository_root=root,
            )
        self.assertTrue(result["promotion_passed"])
        self.assertGreater(
            result["effects"]["live_speculation_E_vs_F"]["relative_reduction"],
            0.05,
        )

    def test_screening_effect_below_gate_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = validate_live_joint_result(
                _make_payload(root, stage="screening", weak_gain=True),
                stage="screening",
                repository_root=root,
            )
        self.assertFalse(result["promotion_passed"])
        self.assertFalse(result["live_speculation_gates"]["mean_task_e2e_reduction"])

    def test_controlled_authoritative_retry_below_formal_threshold_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="formal")
            _mark_tool_retries(root, payload, cell_id="F", count=1)
            result = validate_live_joint_result(
                payload, stage="formal", repository_root=root
            )
        self.assertTrue(result["promotion_passed"])
        self.assertAlmostEqual(
            result["comparability"]["E_vs_F"][
                "authoritative_retry_rate_difference"
            ],
            1 / 360,
        )
        self.assertEqual(
            result["tool_retry_accounting"]["F"][
                "physical_http_attempt_count"
            ],
            361,
        )

    def test_per_block_retry_rate_above_two_percent_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="formal")
            _mark_tool_retries(
                root,
                payload,
                cell_id="F",
                count=3,
                block_id="block-1",
            )
            result = validate_live_joint_result(
                payload, stage="formal", repository_root=root
            )
        self.assertFalse(result["promotion_passed"])
        self.assertFalse(
            result["live_speculation_gates"][
                "all_block_cells_authoritative_retry_rate"
            ]
        )

    def test_retry_rate_imbalance_above_one_pp_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="formal")
            for block_id in ("block-1", "block-2", "block-3"):
                _mark_tool_retries(
                    root,
                    payload,
                    cell_id="F",
                    count=2,
                    block_id=block_id,
                )
            result = validate_live_joint_result(
                payload, stage="formal", repository_root=root
            )
        self.assertFalse(result["promotion_passed"])
        self.assertTrue(
            result["live_speculation_gates"][
                "all_cells_authoritative_retry_rate"
            ]
        )
        self.assertFalse(
            result["live_speculation_gates"][
                "authoritative_retry_rate_balance"
            ]
        )
        self.assertFalse(
            result["overall_system_gates"][
                "authoritative_retry_rate_balance_A_F"
            ]
        )

    def test_recorded_wait_attestation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            payload["runtime"]["recorded_wait_replay"] = True
            with self.assertRaisesRegex(ValueError, "runtime attestation failed"):
                validate_live_joint_result(payload, stage="screening", repository_root=root)

    def test_nonexact_authoritative_commit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            entry = payload["cells"]["F"]["evidence"]["tool_events"]
            path = root / entry["path"]
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            speculative_row = next(row for row in rows if row["speculative"])
            speculative_row["exact_match"] = False
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "non-exact speculative result"):
                validate_live_joint_result(payload, stage="screening", repository_root=root)

    def test_tampered_evidence_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            payload["cells"]["A"]["evidence"]["server_log"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_live_joint_result(payload, stage="screening", repository_root=root)

    def test_raw_resource_samples_control_queue_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            entry = payload["cells"]["A"]["evidence"]["resource_samples"]
            path = root / entry["path"]
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["llm_waiting"] = 0
                row["tool_queued_authoritative"] = 0
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "dual-queue load-selection proof failed"):
                validate_live_joint_result(payload, stage="screening", repository_root=root)

    def test_unfrozen_search_backend_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            entry = payload["cells"]["A"]["evidence"]["tool_events"]
            path = root / entry["path"]
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            search_row = next(row for row in rows if row["tool"] == "search")
            search_row["backend"] = "wikipedia_rest_search"
            search_row["request_host"] = "en.wikipedia.org"
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "not a live Bing HTML request"):
                validate_live_joint_result(payload, stage="screening", repository_root=root)

    def test_started_tool_requires_actual_final_http_200(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            entry = payload["cells"]["F"]["evidence"]["tool_events"]
            path = root / entry["path"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["transport_identity_source"] = "planned"
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "actual final HTTP evidence"):
                validate_live_joint_result(
                    payload, stage="screening", repository_root=root
                )

    def test_tool_attempt_above_controlled_bound_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            entry = payload["cells"]["F"]["evidence"]["tool_events"]
            path = root / entry["path"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["http_attempts"] = 3
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "controlled range 1..2"):
                validate_live_joint_result(
                    payload, stage="screening", repository_root=root
                )

    def test_hidden_http_library_retry_control_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            payload["cells"]["F"]["tool_runtime"][
                "tool_http_library_retry_disabled"
            ] = False
            with self.assertRaisesRegex(ValueError, "hidden HTTP-library retry"):
                validate_live_joint_result(
                    payload, stage="screening", repository_root=root
                )

    def test_prestart_cancellation_requires_zero_attempt_and_no_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _make_payload(root, stage="screening")
            entry = payload["cells"]["F"]["evidence"]["tool_events"]
            path = root / entry["path"]
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            waste = dict(next(row for row in rows if row["tool"] == "visit"))
            waste.update(
                {
                    "job_id": "never-started-waste",
                    "logical_call_id": None,
                    "invocation_id": "never-started-waste",
                    "result_digest": None,
                    "speculative": True,
                    "authoritative": False,
                    "committed": False,
                    "started_at": None,
                    "authoritative_confirmation_at": None,
                    "finished_at": waste["queue_enter_at"] + 4.0,
                    "outcome": "cancelled",
                    "exact_match": False,
                    "source": "cancelled",
                    "cancelled": True,
                    "worker_id": None,
                    "queue_s": 4.0,
                    "service_s": 0.0,
                    "saved_service_s": 0.0,
                    "response_status": None,
                    "bytes_read": None,
                    "http_attempts": 0,
                    "backend": None,
                    "request_host": None,
                    "transport_identity_source": None,
                }
            )
            rows.append(waste)
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            result = validate_live_joint_result(
                payload, stage="screening", repository_root=root
            )
            self.assertTrue(result["promotion_passed"])

            waste["http_attempts"] = None
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "http_attempts must be an integer"):
                validate_live_joint_result(
                    payload, stage="screening", repository_root=root
                )

            waste["http_attempts"] = 0
            waste["bytes_read"] = 1
            _write_jsonl(path, rows)
            entry["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "claims HTTP evidence"):
                validate_live_joint_result(
                    payload, stage="screening", repository_root=root
                )

    def test_formal_passes_prefix_and_system_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = validate_live_joint_result(
                _make_payload(root, stage="formal"),
                stage="formal",
                repository_root=root,
            )
        self.assertTrue(result["promotion_passed"])
        self.assertTrue(result["prefix_ablation"]["selection_valid"])
        self.assertFalse(result["thirty_percent_claim_permitted"])
        self.assertEqual(result["independent_source_count"], 60)
        self.assertEqual(result["block_count"], 3)


if __name__ == "__main__":
    unittest.main()
