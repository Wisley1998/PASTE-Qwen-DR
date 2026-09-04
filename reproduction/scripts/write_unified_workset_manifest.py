#!/usr/bin/env python3
"""Validate a paired Qwen run and atomically write its provenance manifest.

The manifest intentionally does not snapshot the process environment.  The
profile is pinned by file hash and exported variable *names* only, while vLLM
launch evidence is restricted to an explicit allowlist parsed from each
server log.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "paste_repro.dr_trace_hybrid_plan.v1"
RESULT_SCHEMA = "paste_repro.dr_trace_hybrid_result.v1"
MANIFEST_SCHEMA = "paste_repro.unified_workset_manifest.v1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = Path(__file__).with_name("run_dr_trace_hybrid_pair.py")
DEFAULT_LAUNCHER = Path(__file__).with_name("start_vllm.sh")
DEFAULT_SITECUSTOMIZE = (
    REPOSITORY_ROOT / "scripts" / "pythonhooks" / "sitecustomize.py"
)

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROFILE_NAME_RE = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE
)
_SAFE_VLLM_ARG_NAMES = frozenset(
    {
        "host",
        "port",
        "model",
        "served_model_name",
        "revision",
        "dtype",
        "max_model_len",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "disable_custom_all_reduce",
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "max_num_batched_tokens",
        "max_num_seqs",
        "cuda_graph_sizes",
        "enable_chunked_prefill",
    }
)
_REQUIRED_VLLM_ARG_NAMES = frozenset(
    {
        "model",
        "served_model_name",
        "dtype",
        "max_model_len",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_prefix_caching",
        "cuda_graph_sizes",
    }
)


def canonical_hash(value: Any) -> str:
    wire = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def embedded_hash_valid(value: Mapping[str, Any], field: str) -> bool:
    expected = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    return isinstance(expected, str) and expected == canonical_hash(unsigned)


def artifact_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"regular file required: {path}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _resolve_recorded_path(recorded: object, result_path: Path) -> Path:
    if not isinstance(recorded, str) or not recorded:
        raise ValueError(f"result has no usable plan path: {result_path}")
    path = Path(recorded)
    return path if path.is_absolute() else result_path.parent / path


def _plan_counts(plan: Mapping[str, Any]) -> dict[str, int]:
    traces = plan.get("traces")
    if not isinstance(traces, list):
        raise ValueError("plan.traces must be a list")
    requests = 0
    tools = 0
    prompt_tokens = 0
    completion_tokens = 0
    for trace in traces:
        steps = trace["steps"]
        requests += len(steps)
        for step in steps:
            request = step["request"]
            prompt_tokens += int(request["prompt_tokens"])
            completion_tokens += int(request["fixed_completion_tokens"])
            tools += len(step["tools_after"])
    return {
        "sessions": len(traces),
        "requests": requests,
        "tools": tools,
        "prompt_tokens": prompt_tokens,
        "fixed_completion_tokens": completion_tokens,
    }


def _result_counts(result: Mapping[str, Any]) -> dict[str, int]:
    tasks = result.get("tasks")
    llm_events = result.get("llm_events")
    tool_events = result.get("tool_events")
    if not isinstance(tasks, list):
        raise ValueError("result.tasks must be a list")
    if not isinstance(llm_events, list):
        raise ValueError("result.llm_events must be a list")
    if not isinstance(tool_events, list):
        raise ValueError("result.tool_events must be a list")
    return {
        "tasks": len(tasks),
        "successful_tasks": sum(row.get("ok") is True for row in tasks),
        "llm_requests": len(llm_events),
        "tool_calls": len(tool_events),
        "prompt_tokens": sum(
            int(row["usage"]["prompt_tokens"]) for row in llm_events
        ),
        "completion_tokens": sum(
            int(row["usage"]["completion_tokens"]) for row in llm_events
        ),
    }


def _summary_counts_match(
    result: Mapping[str, Any], counts: Mapping[str, int]
) -> bool:
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return False
    return all(
        summary.get(key) == counts[key]
        for key in ("tasks", "successful_tasks", "llm_requests", "tool_calls")
    )


def _plan_summary_matches(plan: Mapping[str, Any], counts: Mapping[str, int]) -> bool:
    summary = plan.get("summary")
    if not isinstance(summary, Mapping):
        return False
    return all(summary.get(key) == value for key, value in counts.items())


def _plan_task_ids(plan: Mapping[str, Any]) -> list[str]:
    return sorted(str(trace["task_id"]) for trace in plan["traces"])


def _result_task_ids(result: Mapping[str, Any]) -> list[str]:
    return sorted(str(row["task_id"]) for row in result["tasks"])


def _plan_llm_signature(plan: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for trace in plan["traces"]:
        task_id = str(trace["task_id"])
        for request_index, step in enumerate(trace["steps"]):
            request = step["request"]
            rows.append(
                (
                    task_id,
                    request_index,
                    int(request["call_index"]),
                    int(request["prompt_tokens"]),
                    int(request["fixed_completion_tokens"]),
                    200,
                )
            )
    return sorted(rows)


def _result_llm_signature(result: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(row["task_id"]),
            int(row["request_index"]),
            int(row["call_index"]),
            int(row["usage"]["prompt_tokens"]),
            int(row["usage"]["completion_tokens"]),
            int(row["http_status"]),
        )
        for row in result["llm_events"]
    )


def _tool_result_digest(tool: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "tool_name": tool["tool_name"],
            "call_index": tool["call_index"],
            "visit_units": tool.get("visit_units", []),
        }
    )


def _plan_tool_signature(plan: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for trace in plan["traces"]:
        task_id = str(trace["task_id"])
        for step in trace["steps"]:
            for tool in step["tools_after"]:
                rows.append(
                    (
                        task_id,
                        int(tool["event_index"]),
                        int(tool["call_index"]),
                        str(tool["tool_name"]),
                        float(tool["duration_s"]),
                        _tool_result_digest(tool),
                    )
                )
    return sorted(rows)


def _result_tool_signature(result: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(row["task_id"]),
            int(row["event_index"]),
            int(row["call_index"]),
            str(row["tool_name"]),
            float(row["full_service_s"]),
            str(row["result_sha256"]),
        )
        for row in result["tool_events"]
    )


def _strip_ansi(value: str) -> str:
    return _ANSI_RE.sub("", value)


def parse_server_log(path: Path) -> dict[str, Any]:
    text = _strip_ansi(path.read_text(encoding="utf-8", errors="replace"))
    args_match = re.search(r"non-default args:\s*(\{[^\n]*\})", text)
    safe_args: dict[str, Any] | None = None
    args_line_sha256: str | None = None
    if args_match:
        args_line_sha256 = hashlib.sha256(
            args_match.group(0).encode("utf-8")
        ).hexdigest()
        try:
            parsed = ast.literal_eval(args_match.group(1))
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            safe_args = {
                str(key): value
                for key, value in parsed.items()
                if key in _SAFE_VLLM_ARG_NAMES
            }
    running = [int(value) for value in re.findall(r"Running: (\d+)", text)]
    waiting = [int(value) for value in re.findall(r"Waiting: (\d+)", text)]
    gpu_mappings = sorted(
        {
            (int(cuda_device), int(nvml_device), bus_id.lower())
            for cuda_device, nvml_device, bus_id in re.findall(
                r"cudaDev (\d+) nvmlDev (\d+) busId ([0-9A-Fa-f]+)", text
            )
        }
    )
    return {
        "vllm_non_default_args_found": safe_args is not None,
        "vllm_non_default_args_line_sha256": args_line_sha256,
        "vllm_non_default_args_allowlisted": safe_args,
        "http_200_chat_completions": len(
            re.findall(r'POST /v1/chat/completions HTTP/1\.1" 200', text)
        ),
        "joint_hook_installations": text.count(
            "[sched_policy_patch] installed policy=online_joint_pacer_v2"
        ),
        "fail_open_markers": text.count("fail_open"),
        "max_running": max(running, default=0),
        "max_waiting": max(waiting, default=0),
        "nccl_local_to_nvml_gpu_mapping": [
            {
                "cuda_device": cuda_device,
                "nvml_device": nvml_device,
                "nccl_bus_id": bus_id,
            }
            for cuda_device, nvml_device, bus_id in gpu_mappings
        ],
    }


def _comparable_server_args(evidence: Mapping[str, Any]) -> dict[str, Any]:
    args = evidence.get("vllm_non_default_args_allowlisted")
    if not isinstance(args, Mapping):
        return {}
    # Endpoint placement is intentionally cell-local; all engine/model knobs
    # must remain equal.
    return {str(key): value for key, value in args.items() if key not in {"port"}}


def _required_server_args(evidence: Mapping[str, Any]) -> dict[str, Any]:
    args = evidence.get("vllm_non_default_args_allowlisted")
    if not isinstance(args, Mapping):
        return {}
    return {str(key): args[key] for key in _REQUIRED_VLLM_ARG_NAMES if key in args}


def _served_model_matches(result: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    args = evidence.get("vllm_non_default_args_allowlisted")
    if not isinstance(args, Mapping):
        return False
    served = args.get("served_model_name")
    names = served if isinstance(served, list) else [served]
    return result.get("model") in names


def _runtime_gpu_binding(
    evidence: Mapping[str, Any], gpu_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    mappings = evidence.get("nccl_local_to_nvml_gpu_mapping")
    if not isinstance(mappings, list):
        mappings = []
    uuids_by_index = {
        row.get("index"): row.get("uuid")
        for row in gpu_inventory.get("gpus", [])
        if isinstance(row, Mapping)
    }
    return {
        "cuda_visible_devices_environment_value": None,
        "cuda_visible_devices_environment_status": "unknown_not_emitted_by_log",
        "local_to_nvml_device": mappings,
        "gpu_uuids_correlated_at_manifest_write": [
            uuids_by_index.get(row.get("nvml_device")) for row in mappings
        ],
        "mapping_evidence": (
            "NCCL cudaDev-to-nvmlDev records in server log"
            if mappings
            else "unavailable"
        ),
    }


def _run(
    argv: Sequence[str], *, cwd: Path, timeout_s: float = 10.0
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )


def collect_git_provenance(repo_root: Path) -> dict[str, Any]:
    commit = _run(("git", "rev-parse", "HEAD"), cwd=repo_root)
    if commit.returncode != 0:
        raise ValueError(f"not a readable git worktree: {repo_root}")
    status = _run(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        cwd=repo_root,
    )
    diff = _run(
        ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"),
        cwd=repo_root,
    )
    if status.returncode != 0 or diff.returncode != 0:
        raise ValueError("could not fingerprint git dirty state")
    # Only hashes are retained.  This avoids copying either a diff or values
    # from an accidentally tracked local configuration into the manifest.
    return {
        "repository_root": str(repo_root.resolve()),
        "commit": commit.stdout.decode("ascii", errors="strict").strip(),
        "dirty": bool(status.stdout),
        "dirty_state_scope": "immediately before manifest write",
        "dirty_status_sha256": hashlib.sha256(status.stdout).hexdigest(),
        "dirty_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "dirty_diff_bytes": len(diff.stdout),
    }


def collect_gpu_inventory() -> dict[str, Any]:
    try:
        query = _run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,pci.bus_id,driver_version",
                "--format=csv,noheader,nounits",
            ),
            cwd=REPOSITORY_ROOT,
        )
        topology = _run(("nvidia-smi", "topo", "-m"), cwd=REPOSITORY_ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return {
            "captured": False,
            "error": "nvidia-smi inventory unavailable",
        }
    if query.returncode != 0 or topology.returncode != 0:
        return {
            "captured": False,
            "error": "nvidia-smi inventory unavailable",
        }
    gpus: list[dict[str, Any]] = []
    for line in query.stdout.decode("utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, uuid, name, memory_mib, pci_bus_id, driver = fields
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(memory_mib),
                "pci_bus_id": pci_bus_id,
                "driver_version": driver,
            }
        )
    return {
        "captured": True,
        "scope": "manifest_write_time_read_only_inventory",
        "gpus": gpus,
        "topology": topology.stdout.decode("utf-8", errors="replace"),
    }


def profile_export_names(path: Path) -> list[str]:
    # Values are deliberately omitted from the output.
    return sorted(set(_PROFILE_NAME_RE.findall(path.read_text(encoding="utf-8"))))


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path = args.baseline.resolve(strict=True)
    full_path = args.full.resolve(strict=True)
    baseline_log_path = args.baseline_server_log.resolve(strict=True)
    full_log_path = args.full_server_log.resolve(strict=True)
    profile_path = args.profile_config.resolve(strict=True)
    hook_path = args.hook.resolve(strict=True)
    runner_path = args.runner.resolve(strict=True)
    launcher_path = args.launcher.resolve(strict=True)
    sitecustomize_path = args.sitecustomize.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)

    baseline = read_json_object(baseline_path)
    full = read_json_object(full_path)
    recorded_plan = _resolve_recorded_path(baseline.get("plan"), baseline_path)
    plan_path = (
        args.plan.resolve(strict=True) if args.plan is not None
        else recorded_plan.resolve(strict=True)
    )
    plan = read_json_object(plan_path)

    plan_counts = _plan_counts(plan)
    baseline_counts = _result_counts(baseline)
    full_counts = _result_counts(full)
    plan_llm = _plan_llm_signature(plan)
    baseline_llm = _result_llm_signature(baseline)
    full_llm = _result_llm_signature(full)
    plan_tools = _plan_tool_signature(plan)
    baseline_tools = _result_tool_signature(baseline)
    full_tools = _result_tool_signature(full)
    plan_tasks = _plan_task_ids(plan)
    baseline_tasks = _result_task_ids(baseline)
    full_tasks = _result_task_ids(full)

    baseline_log = parse_server_log(baseline_log_path)
    full_log = parse_server_log(full_log_path)

    plan_summary_expected = {
        "sessions": plan_counts["sessions"],
        "requests": plan_counts["requests"],
        "tools": plan_counts["tools"],
        "prompt_tokens": plan_counts["prompt_tokens"],
        "fixed_completion_tokens": plan_counts["fixed_completion_tokens"],
    }
    checks = {
        "supported_plan_schema": plan.get("schema") == PLAN_SCHEMA,
        "supported_result_schemas": (
            baseline.get("schema") == RESULT_SCHEMA
            and full.get("schema") == RESULT_SCHEMA
        ),
        "embedded_plan_hash_valid": embedded_hash_valid(plan, "plan_sha256"),
        "embedded_result_hashes_valid": (
            embedded_hash_valid(baseline, "result_sha256")
            and embedded_hash_valid(full, "result_sha256")
        ),
        "baseline_and_full_roles": (
            baseline.get("system") == "baseline" and full.get("system") == "full"
        ),
        "same_frozen_plan": (
            baseline.get("plan_sha256")
            == full.get("plan_sha256")
            == plan.get("plan_sha256")
        ),
        "same_result_model": baseline.get("model") == full.get("model"),
        # Compare the recorded provenance strings, but allow ``--plan`` to
        # point at a relocated byte-identical plan on another machine.
        "recorded_plan_paths_agree": baseline.get("plan") == full.get("plan"),
        "plan_counts_match_summary": _plan_summary_matches(
            plan, plan_summary_expected
        ),
        "baseline_counts_match_summary": _summary_counts_match(
            baseline, baseline_counts
        ),
        "full_counts_match_summary": _summary_counts_match(full, full_counts),
        "task_ids_match_plan": (
            plan_tasks == baseline_tasks == full_tasks
            and len(plan_tasks) == len(set(plan_tasks))
        ),
        "all_tasks_successful": (
            baseline_counts["tasks"] == baseline_counts["successful_tasks"]
            and full_counts["tasks"] == full_counts["successful_tasks"]
        ),
        "llm_token_work_and_status_match_plan": (
            plan_llm == baseline_llm == full_llm
        ),
        "tool_trace_work_and_results_match_plan": (
            plan_tools == baseline_tools == full_tools
        ),
        "server_http_counts_match_results": (
            baseline_log["http_200_chat_completions"]
            == baseline_counts["llm_requests"]
            and full_log["http_200_chat_completions"]
            == full_counts["llm_requests"]
        ),
        "server_argv_evidence_present": (
            baseline_log["vllm_non_default_args_found"]
            and full_log["vllm_non_default_args_found"]
        ),
        "server_required_argv_keys_present": (
            set(_required_server_args(baseline_log)) == _REQUIRED_VLLM_ARG_NAMES
            and set(_required_server_args(full_log)) == _REQUIRED_VLLM_ARG_NAMES
        ),
        "server_required_argv_equal": (
            _required_server_args(baseline_log) == _required_server_args(full_log)
        ),
        "server_engine_argv_equal": (
            _comparable_server_args(baseline_log)
            == _comparable_server_args(full_log)
        ),
        "result_models_match_served_model_argv": (
            _served_model_matches(baseline, baseline_log)
            and _served_model_matches(full, full_log)
        ),
        "baseline_joint_hook_absent": baseline_log["joint_hook_installations"] == 0,
        "full_joint_hook_installed": full_log["joint_hook_installations"] > 0,
        "server_logs_fail_open_free": (
            baseline_log["fail_open_markers"] == 0
            and full_log["fail_open_markers"] == 0
        ),
    }
    invalid_checks = [name for name, passed in checks.items() if passed is not True]

    artifacts = {
        "plan": artifact_ref(plan_path),
        "baseline_result": artifact_ref(baseline_path),
        "full_result": artifact_ref(full_path),
        "baseline_server_log": artifact_ref(baseline_log_path),
        "full_server_log": artifact_ref(full_log_path),
        "profile_config": artifact_ref(profile_path),
        "scheduler_hook": artifact_ref(hook_path),
        "sitecustomize": artifact_ref(sitecustomize_path),
        "vllm_launcher": artifact_ref(launcher_path),
        "runner": artifact_ref(runner_path),
    }
    gpu_inventory = (
        {"captured": False, "reason": "disabled by command line"}
        if args.skip_gpu_inventory
        else collect_gpu_inventory()
    )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": not invalid_checks,
        "invalid_checks": invalid_checks,
        "validation": {
            "checks": checks,
            "counts": {
                "plan": plan_counts,
                "baseline": baseline_counts,
                "full": full_counts,
            },
            "workload_digests": {
                "plan_llm_token_work_and_status_sha256": canonical_hash(plan_llm),
                "baseline_llm_token_work_and_status_sha256": canonical_hash(
                    baseline_llm
                ),
                "full_llm_token_work_and_status_sha256": canonical_hash(full_llm),
                "plan_tool_trace_and_result_sha256": canonical_hash(plan_tools),
                "baseline_tool_trace_and_result_sha256": canonical_hash(
                    baseline_tools
                ),
                "full_tool_trace_and_result_sha256": canonical_hash(full_tools),
            },
        },
        "artifacts": artifacts,
        "provenance": {
            "git": collect_git_provenance(repo_root),
            "profile_export_names": profile_export_names(profile_path),
            "profile_values_recorded": False,
            "process_environment_recorded": False,
            "server_argv_evidence": {
                "source": "vllm server log effective non-default args (explicit allowlist)",
                "completeness": "partial_effective_argv_not_exact_process_argv",
                "baseline": baseline_log,
                "full": full_log,
            },
            "exact_command_evidence": {
                "client_argv": {
                    "status": "unknown",
                    "reason": "result schema does not persist the invoking process argv",
                },
                "server_argv": {
                    "status": "partial",
                    "reason": "vLLM log persists effective non-default args, not exact process argv",
                },
            },
            "runtime_scheduler_environment": {
                "status": "unknown",
                "vllm_sched_snapshot": None,
                "reason": (
                    "legacy server logs do not emit a complete VLLM_SCHED_* "
                    "snapshot; the profile hash proves intended configuration "
                    "bytes only, not the runtime environment"
                ),
            },
            "runtime_gpu_binding": {
                "baseline": _runtime_gpu_binding(baseline_log, gpu_inventory),
                "full": _runtime_gpu_binding(full_log, gpu_inventory),
            },
            "gpu_inventory": gpu_inventory,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wire = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(wire)
            stream.flush()
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--baseline-server-log", type=Path, required=True)
    parser.add_argument("--full-server-log", type=Path, required=True)
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--hook", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--sitecustomize", type=Path, default=DEFAULT_SITECUSTOMIZE)
    parser.add_argument(
        "--plan",
        type=Path,
        help="Optional relocated plan; defaults to the path embedded in the result",
    )
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--skip-gpu-inventory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    write_json_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "valid": manifest["valid"],
                "invalid_checks": manifest["invalid_checks"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0 if manifest["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
