#!/usr/bin/env python3
"""Run the frozen A/B/E/F external-live formal matrix, one fresh server per cell."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_formal_v8_matrix.env.example"
)
FORMAL_VALIDATOR = (
    REPOSITORY_ROOT / "reproduction/scripts/validate_live_joint_formal_workload.py"
)
RUNNER = REPOSITORY_ROOT / "scripts/run_live_tool_llm_experiment.py"
START_SERVER = REPOSITORY_ROOT / "reproduction/scripts/start_vllm.sh"
STOP_SERVER = REPOSITORY_ROOT / "reproduction/scripts/stop_vllm.sh"
PROTOCOL = (
    REPOSITORY_ROOT
    / "reproduction/results/live_joint/LIVE_TOOL_LLM_PROTOCOL.md"
)
LIVE_AGENT = REPOSITORY_ROOT / "reproduction/paste_repro/live_agent.py"

FORMAL_WORKLOAD_SHA256 = (
    "780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4"
)
FORMAL_CANONICAL_SHA256 = (
    "93b8cfad78b76c42101f7d0f23583911b01bc8c075260ae3d85bce45456a9ec7"
)
FORMAL_SOURCES_SHA256 = (
    "01b029c3427f5f04d4f1b83b4f9b13e5decd705e773ffdeaeebb15970150f0df"
)
GUIDED_JSON_RECOVERY_POLICY_VERSION = "escape-unescaped-string-controls-v1"
OUTPUT_CONTRACT_POLICY_VERSION = (
    "guided-tool-json-and-fixed-final-grammar-strict-local-projection-v1"
)
FINAL_ANSWER_SCHEMA_POLICY_VERSION = "xgrammar-unbounded-answer-exact-url-v1"
FINAL_ANSWER_CONTRACT_POLICY_VERSION = (
    "guided-grammar-fixed-192-token-strict-tail-local-projection-v1"
)
FINAL_ANSWER_GRAMMAR_POLICY_VERSION = (
    "xgrammar-compact-unbounded-answer-exact-url-ascii-space-tail-v1"
)
FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION = "0.1.21"
VLLM_LIBRARY_VERSION = "0.10.1"
TRANSFORMERS_LIBRARY_VERSION = "4.56.1"
FIXED_FINAL_COMPLETION_TOKENS = 192
LIVE_AGENT_SHA256 = (
    "6dab494fa65749b1d60a5b5cbfbb4d0eed3c804b91b3646e0388c707cb7ade8f"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_CELL_SET = frozenset({"A", "B", "E", "F"})
CELL_POLICY = {
    "A": ("fcfs", "off"),
    "B": ("fcfs", "visit"),
    "E": ("online_joint_pacer_v2", "off"),
    "F": ("online_joint_pacer_v2", "visit"),
}
EXPORT_RE = re.compile(r'export ([A-Z][A-Z0-9_]*)="([^"\\]*)"\Z')

EXPECTED_CONFIG: dict[str, str] = {
    "PASTE_LIVE_FORMAL_PROFILE": (
        "live_joint_wikipedia_frozen_formal_v8_"
        "context10000_visitonly_execaware_attemptgate_retry2_"
        "jsonrecovery0_fixedfinal192_stricttail_load80"
    ),
    "PASTE_LIVE_FORMAL_WORKLOAD": (
        "reproduction/workloads/live_joint_wikipedia_frozen_formal_v8.json"
    ),
    "PASTE_LIVE_FORMAL_WORKLOAD_SHA256": FORMAL_WORKLOAD_SHA256,
    "PASTE_LIVE_FORMAL_CANONICAL_SHA256": FORMAL_CANONICAL_SHA256,
    "PASTE_LIVE_FORMAL_SOURCES_SHA256": FORMAL_SOURCES_SHA256,
    "PASTE_LIVE_FORMAL_SOURCE_COUNT": "80",
    "PASTE_LIVE_FORMAL_DEFAULT_ORDERS": "A,B,E,F;B,A,F,E;A,B,F,E",
    "PASTE_LIVE_FORMAL_RUN_BASE": "reproduction/artifacts/live_joint/formal",
    "PASTE_LIVE_GUIDED_JSON_RECOVERY_POLICY_VERSION": (
        GUIDED_JSON_RECOVERY_POLICY_VERSION
    ),
    "PASTE_LIVE_OUTPUT_CONTRACT_POLICY_VERSION": OUTPUT_CONTRACT_POLICY_VERSION,
    "PASTE_LIVE_FINAL_ANSWER_SCHEMA_POLICY_VERSION": (
        FINAL_ANSWER_SCHEMA_POLICY_VERSION
    ),
    "PASTE_LIVE_FINAL_ANSWER_CONTRACT_POLICY_VERSION": (
        FINAL_ANSWER_CONTRACT_POLICY_VERSION
    ),
    "PASTE_LIVE_FINAL_ANSWER_GRAMMAR_POLICY_VERSION": (
        FINAL_ANSWER_GRAMMAR_POLICY_VERSION
    ),
    "PASTE_LIVE_FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION": (
        FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION
    ),
    "PASTE_LIVE_FIXED_FINAL_COMPLETION_TOKENS": (
        str(FIXED_FINAL_COMPLETION_TOKENS)
    ),
    "PASTE_LIVE_VLLM_LIBRARY_VERSION": VLLM_LIBRARY_VERSION,
    "PASTE_LIVE_TRANSFORMERS_LIBRARY_VERSION": TRANSFORMERS_LIBRARY_VERSION,
    "PASTE_LIVE_GUIDED_JSON_RECOVERY_MODULE_SHA256": LIVE_AGENT_SHA256,
    "PASTE_LIVE_FORMAL_MAX_GUIDED_JSON_RECOVERIES": "0",
    "PASTE_ENV_PREFIX": "/home/aiscuser/.conda/envs/paste",
    "HF_HOME": "/home/aiscuser/hf_cache",
    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
    "VLLM_HOST": "127.0.0.1",
    "VLLM_PROBE_HOST": "127.0.0.1",
    "VLLM_PORT": "8100",
    "VLLM_TP_SIZE": "4",
    "VLLM_DTYPE": "bfloat16",
    "VLLM_MAX_MODEL_LEN": "16384",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.86",
    "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
    "VLLM_MAX_NUM_SEQS": "96",
    "VLLM_CUDA_GRAPH_SIZES": "32",
    "VLLM_ENABLE_PREFIX_CACHING": "1",
    "VLLM_USE_V1": "1",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
    "VLLM_READY_TIMEOUT": "3600",
    "VLLM_SHUTDOWN_TIMEOUT": "60",
    "VLLM_SCHED_PRED_OUT_ENABLE": "1",
    "VLLM_SCHED_PRED_OUT_EMA_ALPHA": "0.5",
    "VLLM_SCHED_DEFAULT_PRED_OUT": "128",
    "VLLM_SCHED_AVG_CALL_SERVICE_S": "2.0",
    "VLLM_SCHED_PREFILL_TOKENS_PER_S_V2": "38112",
    "VLLM_SCHED_DECODE_TOKENS_PER_S_V2": "113.7",
    "VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S": "6000",
    "VLLM_SCHED_TIME_AGING_ALPHA": "0.2",
    "VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS": "96",
    "VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING": "48",
    "VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S": "40",
    "VLLM_SCHED_JOINT_V2_FINAL_LANE": "1",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE": "1",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES": "0",
    "VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S": "0",
    "VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING": "48",
    "VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING": "96",
    "VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING": "96",
    "VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION": "0",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION": "1",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION": "0.93",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S": "40",
    "VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S": "1",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY": "0",
    "VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S": "0",
    "VLLM_SCHED_JOINT_V2_TAIL_BETA": "0.25",
    "VLLM_SCHED_JOINT_V2_TOOL_BETA": "0.9",
    "VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S": "80",
    "VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT": "0.35",
    "VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA": "1.4",
    "VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS": "8000",
    "VLLM_SCHED_JOINT_V2_FINAL_BONUS_S": "12",
    "VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S": "8",
    "VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S": "4",
    "VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S": "120",
    "VLLM_SCHED_HBM_MIN_RUNNING_REQS": "16",
    "VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP": "16",
    "VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS": "16000",
    "VLLM_SCHED_HBM_MAX_LONG_RUNNING": "16",
    "VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS": "786432",
    "VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS": "524288",
    "VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS": "1048576",
    "VLLM_SCHED_HBM_LOW_PRESSURE": "0.82",
    "VLLM_SCHED_HBM_HIGH_PRESSURE": "1.02",
    "VLLM_SCHED_HBM_BUDGET_INCREASE": "1.02",
    "VLLM_SCHED_HBM_BUDGET_DECREASE": "0.97",
    "VLLM_SCHED_HBM_CONTROL_INTERVAL_S": "5",
    "VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO": "0.96",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY": "0",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_WEIGHT": "1",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_REFRESH_S": "0.25",
    "VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY_LOG_INTERVAL_S": "1",
    "PASTE_LIVE_MAX_ACTIVE_TASKS": "80",
    "PASTE_LIVE_REPLICAS": "1",
    "PASTE_LIVE_TOOL_SIGNAL_POLICY": "execution_aware",
    "PASTE_LIVE_TOOL_WORKERS": "4",
    "PASTE_LIVE_SPECULATIVE_TOOL_WORKERS": "2",
    "PASTE_LIVE_MIN_SPECULATIVE_TOOL_WORKERS": "0",
    "PASTE_LIVE_SEARCH_TOOL_CAPACITY": "3",
    "PASTE_LIVE_VISIT_TOOL_CAPACITY": "2",
    "PASTE_LIVE_SEARCH_MIN_START_INTERVAL_S": "0",
    "PASTE_LIVE_VISIT_MIN_START_INTERVAL_S": "2.1",
    "PASTE_LIVE_MAX_SPECULATIVE_PENDING": "128",
    "PASTE_LIVE_SPECULATIVE_TTL_S": "120",
    "PASTE_LIVE_TOOL_TIMEOUT_S": "60",
    "PASTE_LIVE_TOOL_HTTP_MAX_ATTEMPTS": "2",
    "PASTE_LIVE_TOOL_HTTP_RETRY_BACKOFF_S": "1.0",
    "PASTE_LIVE_TOOL_HTTP_ATTEMPT_START_GATE": "1",
    "PASTE_LIVE_TOOL_HTTP_LIBRARY_NAME": "aiohttp",
    "PASTE_LIVE_TOOL_HTTP_LIBRARY_VERSION": "3.12.15",
    "PASTE_LIVE_TOOL_HTTP_LIBRARY_RETRY_DISABLED": "1",
    "PASTE_LIVE_TOOL_HTTP_LIBRARY_RETRY_CONTROL_VERSION": (
        "aiohttp-private-retry-connection-v1"
    ),
    "PASTE_LIVE_TOOL_SERVICE_HINT_S": "2.0",
    "PASTE_LIVE_SEARCH_MAX_RESULTS": "5",
    "PASTE_LIVE_VISIT_MAX_CHARS": "3000",
    "PASTE_LIVE_REQUEST_TIMEOUT_S": "300",
    "PASTE_LIVE_MAX_TOKENS_TOOL": "128",
    "PASTE_LIVE_MAX_TOKENS_ANSWER": "256",
    "PASTE_LIVE_PREDICTED_VISIT_RESULT_TOKENS": "1600",
    "PASTE_LIVE_CONTEXT_PADDING_TOKENS": "10000",
    "PASTE_LIVE_QUEUE_SAMPLE_INTERVAL_S": "0.2",
    "PASTE_LIVE_VISIT_CANARY_STRIDE": "6",
}

# The A/B baselines must be native FCFS processes.  These are the only
# scheduler-extension variables allowed into E/F; _cell_environment first
# removes every inherited VLLM_SCHED_* variable so the launch is deterministic.
FROZEN_JOINT_SCHEDULER_ENV_KEYS = frozenset(
    key for key in EXPECTED_CONFIG if key.startswith("VLLM_SCHED_")
)

BOUND_CODE_PATHS = (
    Path(__file__).resolve(),
    RUNNER,
    START_SERVER,
    STOP_SERVER,
    FORMAL_VALIDATOR,
    PROTOCOL,
    LIVE_AGENT,
    REPOSITORY_ROOT / "reproduction/paste_repro/live_broker.py",
    REPOSITORY_ROOT / "reproduction/paste_repro/live_executor.py",
    REPOSITORY_ROOT / "scripts/pythonhooks/sched_policy_patch.py",
)


class FormalRunError(RuntimeError):
    """Fail-closed formal-matrix error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FormalRunError(f"path is outside the repository: {path}") from exc


def load_frozen_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FormalRunError(f"frozen config does not exist: {path}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = EXPORT_RE.fullmatch(line)
        if match is None:
            raise FormalRunError(
                f"frozen config line {line_number} is not a literal export"
            )
        name, value = match.groups()
        if name in values:
            raise FormalRunError(f"frozen config repeats {name}")
        values[name] = value
    missing = sorted(set(EXPECTED_CONFIG) - set(values))
    extra = sorted(set(values) - set(EXPECTED_CONFIG))
    changed = sorted(
        key for key in EXPECTED_CONFIG.keys() & values.keys()
        if values[key] != EXPECTED_CONFIG[key]
    )
    if missing or extra or changed:
        raise FormalRunError(
            "frozen config mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return values


def _cell_environment(
    config: Mapping[str, str],
    *,
    cell: str,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a deterministic launch environment for one registered cell.

    An invoking shell may contain stale scheduler-experiment variables.  Clear
    all of them before applying the frozen profile.  A/B then clear the frozen
    Joint variables as well, leaving only ``VLLM_SCHED_POLICY=fcfs``; E/F keep
    the complete registered Joint profile.
    """

    if cell not in CELL_POLICY:
        raise FormalRunError(f"unknown formal cell: {cell}")
    values = dict(os.environ if inherited is None else inherited)
    for key in tuple(values):
        if key.startswith("VLLM_SCHED_"):
            values.pop(key)
    values.update(config)
    policy, _speculation = CELL_POLICY[cell]
    if policy == "fcfs":
        for key in FROZEN_JOINT_SCHEDULER_ENV_KEYS:
            values.pop(key, None)
    values["VLLM_SCHED_POLICY"] = policy
    return values


def validate_orders(raw: str, *, baseline_only: bool) -> list[list[str]]:
    orders = [part.split(",") for part in raw.split(";") if part]
    if baseline_only:
        if not orders or orders[0][0:1] != ["A"]:
            raise FormalRunError("baseline-only order must begin with A")
        return [["A"]]
    if len(orders) != 3:
        raise FormalRunError("formal matrix requires exactly three server blocks")
    for index, order in enumerate(orders, 1):
        if len(order) != 4 or frozenset(order) != EXPECTED_CELL_SET:
            raise FormalRunError(
                f"block {index} must contain A,B,E,F exactly once"
            )
    if orders[0][0] != "A":
        raise FormalRunError(
            "block 1 must start with baseline A before any candidate is observed"
        )
    for left, right in (("A", "B"), ("E", "F")):
        forward = sum(order.index(left) < order.index(right) for order in orders)
        reverse = len(orders) - forward
        if min(forward, reverse) < 1 or abs(forward - reverse) > 1:
            raise FormalRunError(
                f"{left}/{right} order is not balanced across three blocks"
            )
    return orders


def validate_formal_workload(
    *, python: Path, workload: Path
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python), str(FORMAL_VALIDATOR), "--workload", str(workload)],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise FormalRunError(
            "formal workload validator failed: " + completed.stderr.strip()
        )
    try:
        validation = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FormalRunError("formal workload validator emitted invalid JSON") from exc
    expected = {
        "valid": True,
        "source_count": 80,
        "formal_eligible": True,
        "file_sha256": FORMAL_WORKLOAD_SHA256,
        "canonical_json_sha256": FORMAL_CANONICAL_SHA256,
        "canonical_sources_sha256": FORMAL_SOURCES_SHA256,
    }
    if any(validation.get(key) != value for key, value in expected.items()):
        raise FormalRunError("formal workload validation does not match frozen SHAs")
    return validation


def validate_entrypoints(*, python: Path) -> None:
    if sha256_file(LIVE_AGENT) != LIVE_AGENT_SHA256:
        raise FormalRunError(
            "live_agent.py does not match the frozen output-contract code"
        )
    aiohttp_version = subprocess.run(
        [
            str(python),
            "-c",
            "import aiohttp; print(aiohttp.__version__)",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        aiohttp_version.returncode != 0
        or aiohttp_version.stdout.strip()
        != EXPECTED_CONFIG["PASTE_LIVE_TOOL_HTTP_LIBRARY_VERSION"]
    ):
        raise FormalRunError(
            "formal live-tool aiohttp version does not match the frozen "
            "no-hidden-retry contract"
        )
    runner_help = subprocess.run(
        [str(python), str(RUNNER), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if runner_help.returncode != 0:
        raise FormalRunError("live runner --help failed: " + runner_help.stderr.strip())
    required_runner_flags = {
        "--formal-block-id",
        "--formal-cell-id",
        "--formal-order-index",
        "--server-instance-id",
        "--fresh-server",
        "--result-cache-empty",
        "--call-graph-mode",
        "--tool-signal-policy",
        "--min-speculative-tool-workers",
        "--search-min-start-interval-s",
        "--visit-min-start-interval-s",
        "--tool-http-max-attempts",
        "--tool-http-retry-backoff-s",
        "--tool-http-attempt-start-gate",
        "--fixed-final-completion-tokens",
    }
    missing = sorted(flag for flag in required_runner_flags if flag not in runner_help.stdout)
    if missing:
        raise FormalRunError(f"live runner is missing formal CLI flags: {missing}")
    guided_stack_versions = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from importlib.metadata import version; "
                "print(version('xgrammar'), version('vllm'), "
                "version('transformers'))"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        guided_stack_versions.returncode != 0
        or guided_stack_versions.stdout.strip().split()
        != [
            FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
            VLLM_LIBRARY_VERSION,
            TRANSFORMERS_LIBRARY_VERSION,
        ]
    ):
        raise FormalRunError(
            "formal fixed-final guided-decoding stack versions do not match"
        )
    for script in (START_SERVER, STOP_SERVER):
        completed = subprocess.run(
            [str(script), "--help"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise FormalRunError(f"{script.name} --help failed: {completed.stderr.strip()}")


def validate_fixed_final_grammar_feasibility(
    *,
    workload: Path,
    model_snapshot: Path,
    expected_source_count: int = 80,
) -> dict[str, Any]:
    """Compile every v8 URL grammar with the pinned tokenizer, entirely offline."""

    reproduction_root = REPOSITORY_ROOT / "reproduction"
    if str(reproduction_root) not in sys.path:
        sys.path.insert(0, str(reproduction_root))
    try:
        import xgrammar
        from transformers import AutoTokenizer
        from paste_repro.live_agent import (
            canonical_json,
            final_answer_fixed_completion_grammar,
        )
    except ImportError as exc:
        raise FormalRunError(
            "fixed-final grammar feasibility dependencies are unavailable"
        ) from exc

    payload = json.loads(workload.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if (
        type(expected_source_count) is not int
        or expected_source_count <= 0
        or not isinstance(sources, list)
        or len(sources) != expected_source_count
    ):
        raise FormalRunError(
            "fixed-final preflight source count does not match its frozen workload"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_snapshot),
        trust_remote_code=True,
        local_files_only=True,
    )
    model_config = json.loads(
        (model_snapshot / "config.json").read_text(encoding="utf-8")
    )
    vocab_size = model_config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise FormalRunError("pinned model config lacks a valid vocabulary size")
    tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=vocab_size,
    )
    compiler = xgrammar.GrammarCompiler(
        tokenizer_info,
        max_threads=8,
        cache_enabled=False,
    )
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    if (
        not isinstance(space_ids, list)
        or len(space_ids) != 1
        or tokenizer.decode(space_ids) != " "
    ):
        raise FormalRunError("pinned tokenizer lacks one exact ASCII-space token")
    eos_token_id = tokenizer.eos_token_id
    if type(eos_token_id) is not int:
        raise FormalRunError("pinned tokenizer lacks an EOS token")

    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise FormalRunError(f"formal source {source_index} is invalid")
        source_id = source.get("source_id")
        url = source.get("expected_url")
        if not isinstance(source_id, str) or not isinstance(url, str):
            raise FormalRunError(f"formal source {source_index} lacks its URL")
        grammar_text = final_answer_fixed_completion_grammar(url)
        compiled = compiler.compile_grammar(
            xgrammar.Grammar.from_ebnf(grammar_text)
        )
        # xgrammar's compact JSON grammar emits escaped forward slashes for a
        # const URL. This wire parses back to the exact canonical URL.
        semantic = canonical_json(
            {"answer": "Grounded factual answer.", "source_url": url}
        ).replace("/", "\\/")
        semantic_ids = tokenizer.encode(semantic, add_special_tokens=False)
        padding_tokens = FIXED_FINAL_COMPLETION_TOKENS - len(semantic_ids)
        if not semantic_ids or padding_tokens <= 0:
            raise FormalRunError(
                f"{source_id} has no positive 192-token grammar padding path"
            )
        matcher = xgrammar.GrammarMatcher(compiled)
        wire_ids = [*semantic_ids, *([space_ids[0]] * padding_tokens)]
        for token_index, token_id in enumerate(wire_ids):
            if not matcher.accept_token(token_id):
                raise FormalRunError(
                    f"{source_id} grammar rejected token {token_index}"
                )
        if matcher.is_terminated():
            raise FormalRunError(
                f"{source_id} grammar terminated before the server length bound"
            )
        if not matcher.accept_token(eos_token_id) or not matcher.is_terminated():
            raise FormalRunError(f"{source_id} grammar cannot terminate with EOS")
        rows.append(
            {
                "source_id": source_id,
                "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                "grammar_sha256": hashlib.sha256(
                    grammar_text.encode("utf-8")
                ).hexdigest(),
                "semantic_wire_sha256": hashlib.sha256(
                    semantic.encode("utf-8")
                ).hexdigest(),
                "semantic_token_count": len(semantic_ids),
                "padding_token_count": padding_tokens,
                "total_completion_tokens": len(wire_ids),
                "eos_termination_succeeded": True,
            }
        )
    return {
        "schema": "paste_repro.fixed_final_grammar_feasibility",
        "version": 1,
        "valid": True,
        "offline_only": True,
        "gpu_or_server_touched": False,
        "network_touched": False,
        "source_count": len(rows),
        "fixed_final_completion_tokens": FIXED_FINAL_COMPLETION_TOKENS,
        "grammar_policy_version": FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
        "xgrammar_version": FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
        "vllm_version": VLLM_LIBRARY_VERSION,
        "transformers_version": TRANSFORMERS_LIBRARY_VERSION,
        "tokenizer_method": "transformers_chat_template",
        "space_token_id": space_ids[0],
        "minimum_semantic_token_count": min(
            row["semantic_token_count"] for row in rows
        ),
        "maximum_semantic_token_count": max(
            row["semantic_token_count"] for row in rows
        ),
        "rows_sha256": _sha256_json(rows),
        "rows": rows,
    }


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormalRunError(
                f"queue timeline line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, Mapping):
            raise FormalRunError(f"queue timeline line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise FormalRunError("queue timeline is empty")
    return rows


def evaluate_baseline_gate(
    result_path: Path,
    timeline_path: Path,
    *,
    block_id: str,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("config", {})
    summary = result.get("summary", {})
    formal = config.get("formal_run", {})
    broker = summary.get("tool", {}).get("broker_stats", {})
    rows = load_jsonl(timeline_path)
    valid_llm = [
        row
        for row in rows
        if isinstance(row.get("llm_running"), (int, float))
        and not isinstance(row.get("llm_running"), bool)
        and isinstance(row.get("llm_waiting"), (int, float))
        and not isinstance(row.get("llm_waiting"), bool)
    ]
    max_num_seqs = int(config.get("scheduler_environment", {}).get(
        "VLLM_MAX_NUM_SEQS", 0
    ) or 0)
    scheduler = config.get("scheduler_environment", {})
    max_active = int(config.get("max_active_tasks", 0) or 0)
    llm_native_wait = [
        row
        for row in valid_llm
        if float(row["llm_waiting"]) > 0
        and float(row["llm_running"]) < max_num_seqs
    ]
    tool_auth_wait = [
        row for row in rows if int(row.get("tool_queued_authoritative", 0) or 0) > 0
    ]
    simultaneous = [
        row
        for row in valid_llm
        if float(row["llm_waiting"]) > 0
        and int(row.get("tool_queued_authoritative", 0) or 0) > 0
    ]
    dual_flags = [
        (
            isinstance(row.get("llm_running"), (int, float))
            and not isinstance(row.get("llm_running"), bool)
            and isinstance(row.get("llm_waiting"), (int, float))
            and not isinstance(row.get("llm_waiting"), bool)
            and float(row["llm_waiting"]) > 0
            and int(row.get("tool_queued_authoritative", 0) or 0) > 0
        )
        for row in rows
    ]
    longest_dual_count = 0
    longest_dual_span_s = 0.0
    streak_start: float | None = None
    streak_count = 0
    previous_dual_monotonic: float | None = None
    previous_dual_wall: float | None = None
    maximum_adjacent_dual_monotonic_gap_s = 0.0
    maximum_adjacent_dual_wall_gap_s = 0.0
    dual_gap_reset_count = 0
    for row, pressured in zip(rows, dual_flags):
        if not pressured:
            streak_start = None
            streak_count = 0
            previous_dual_monotonic = None
            previous_dual_wall = None
            continue
        timestamp = row.get("monotonic_s")
        wall_timestamp = row.get("wall_s")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or isinstance(wall_timestamp, bool)
            or not isinstance(wall_timestamp, (int, float))
            or not math.isfinite(float(wall_timestamp))
        ):
            raise FormalRunError(
                "dual-queue timeline lacks monotonic/wall timestamps"
            )
        monotonic_value = float(timestamp)
        wall_value = float(wall_timestamp)
        if previous_dual_monotonic is not None and previous_dual_wall is not None:
            monotonic_gap = monotonic_value - previous_dual_monotonic
            wall_gap = wall_value - previous_dual_wall
            if monotonic_gap < 0.0 or wall_gap < 0.0:
                raise FormalRunError("dual-queue timeline timestamps are not ordered")
            maximum_adjacent_dual_monotonic_gap_s = max(
                maximum_adjacent_dual_monotonic_gap_s, monotonic_gap
            )
            maximum_adjacent_dual_wall_gap_s = max(
                maximum_adjacent_dual_wall_gap_s, wall_gap
            )
            if monotonic_gap > 0.5 or wall_gap > 0.5:
                streak_start = None
                streak_count = 0
                dual_gap_reset_count += 1
        if streak_start is None:
            streak_start = monotonic_value
            streak_count = 1
        else:
            streak_count += 1
        longest_dual_count = max(longest_dual_count, streak_count)
        longest_dual_span_s = max(
            longest_dual_span_s, monotonic_value - streak_start
        )
        previous_dual_monotonic = monotonic_value
        previous_dual_wall = wall_value
    llm_fraction = len(llm_native_wait) / len(valid_llm) if valid_llm else 0.0
    tool_fraction = len(tool_auth_wait) / len(rows)
    checks = {
        "cell_is_frozen_fcfs_demand_baseline": (
            config.get("call_graph_mode") == "frozen"
            and config.get("speculation_mode") == "off"
            and config.get("scheduler_environment", {}).get("VLLM_SCHED_POLICY")
            == "fcfs"
            and formal.get("block_id") == block_id
            and formal.get("cell_id") == "A"
            and isinstance(formal.get("order_index"), int)
            and formal.get("order_index") in range(4)
            and formal.get("fresh_server") is True
            and formal.get("result_cache_empty") is True
            and formal.get("broker_drained") is True
        ),
        "native_fcfs_has_no_joint_scheduler_knobs": all(
            value is None
            for key, value in scheduler.items()
            if key.startswith("VLLM_SCHED_") and key != "VLLM_SCHED_POLICY"
        ),
        "real_context_profile_is_frozen": (
            config.get("context_padding_tokens") == 10000
            and scheduler.get("VLLM_MAX_MODEL_LEN") == "16384"
            and scheduler.get("VLLM_MAX_NUM_BATCHED_TOKENS") == "2048"
        ),
        "all_tasks_and_requests_exactly_once": (
            summary.get("all_tasks_succeeded") is True
            and summary.get("task_count") == 80
            and summary.get("successful_task_count") == 80
            and summary.get("failed_task_count") == 0
            and summary.get("llm", {}).get("request_count") == 240
            and summary.get("llm", {}).get("successful_request_count") == 240
            and summary.get("llm", {}).get("exactly_one_attempt_each") is True
            and broker.get("authoritative_requests") == 160
            and broker.get("commits") == 160
            and broker.get("authoritative_failures") == 0
        ),
        "offered_concurrency_exceeds_64_but_native_ceiling_is_nonbinding": (
            max_active == 80
            and max_active > 64
            and max_num_seqs == 96
            and max_active < max_num_seqs
        ),
        "llm_waiting_below_native_ceiling_fraction_at_least_005": (
            llm_fraction >= 0.05
        ),
        "tool_authoritative_queue_fraction_at_least_005": tool_fraction >= 0.05,
        "simultaneous_llm_and_tool_queue_sample_count_at_least_10": (
            len(simultaneous) >= 10
        ),
        "continuous_simultaneous_llm_and_tool_queue_span_at_least_1s": (
            longest_dual_span_s >= 1.0
        ),
    }
    return {
        "schema": "paste_repro.live_joint_baseline_gate",
        "version": 1,
        "accepted": all(checks.values()),
        "selection_uses_candidate_performance": False,
        "block_id": block_id,
        "cell_id": "A",
        "thresholds": {
            "max_active_sessions_strictly_below_max_num_seqs": True,
            "minimum_llm_waiting_sample_fraction": 0.05,
            "minimum_tool_authoritative_queue_sample_fraction": 0.05,
            "minimum_simultaneous_queue_samples": 10,
            "minimum_consecutive_simultaneous_queue_span_s": 1.0,
            "maximum_adjacent_simultaneous_sample_gap_s": 0.5,
            "offered_concurrency_must_exceed": 64,
        },
        "observed": {
            "max_active_sessions": max_active,
            "max_num_seqs": max_num_seqs,
            "resource_sample_count": len(rows),
            "llm_metric_sample_count": len(valid_llm),
            "llm_waiting_below_native_ceiling_sample_count": len(llm_native_wait),
            "llm_waiting_below_native_ceiling_sample_fraction": llm_fraction,
            "tool_authoritative_queue_sample_count": len(tool_auth_wait),
            "tool_authoritative_queue_sample_fraction": tool_fraction,
            "simultaneous_llm_and_tool_queue_sample_count": len(simultaneous),
            "longest_consecutive_simultaneous_queue_sample_count": (
                longest_dual_count
            ),
            "longest_consecutive_simultaneous_queue_span_s": (
                longest_dual_span_s
            ),
            "maximum_adjacent_simultaneous_monotonic_gap_s": (
                maximum_adjacent_dual_monotonic_gap_s
            ),
            "maximum_adjacent_simultaneous_wall_gap_s": (
                maximum_adjacent_dual_wall_gap_s
            ),
            "simultaneous_gap_reset_count": dual_gap_reset_count,
        },
        "checks": checks,
        "evidence": {
            "result": {
                "path": repository_relative(result_path),
                "sha256": sha256_file(result_path),
            },
            "resource_samples": {
                "path": repository_relative(timeline_path),
                "sha256": sha256_file(timeline_path),
            },
        },
    }


def _validate_started_tool_record(
    record: Mapping[str, Any],
    *,
    label: str,
    max_http_attempts: int,
    retry_backoff_s: float,
) -> list[float]:
    """Fail fast on the final transport evidence used by the aggregator."""

    attempts = record.get("http_attempts")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 1
        or attempts > max_http_attempts
    ):
        raise FormalRunError(f"{label} has invalid HTTP attempts")
    if "failed" in str(record.get("outcome")):
        raise FormalRunError(f"{label} has a failed physical outcome")
    if record.get("transport_identity_source") != "actual":
        raise FormalRunError(f"{label} lacks actual final HTTP evidence")
    if record.get("response_status") != 200:
        raise FormalRunError(f"{label} final response is not HTTP 200")
    bytes_read = record.get("bytes_read")
    if (
        isinstance(bytes_read, bool)
        or not isinstance(bytes_read, int)
        or bytes_read <= 0
    ):
        raise FormalRunError(f"{label} has invalid response bytes")
    service_raw = record.get("service_s")
    if (
        isinstance(service_raw, bool)
        or not isinstance(service_raw, (int, float))
        or not math.isfinite(float(service_raw))
        or float(service_raw) < 0.0
    ):
        raise FormalRunError(f"{label} has invalid service time")
    if attempts > 1 and float(service_raw) + 0.01 < retry_backoff_s:
        raise FormalRunError(f"{label} service time omits retry backoff")
    expected_transport = {
        "search": ("bing_html_search", "www.bing.com"),
        "visit": ("r.jina.ai", "r.jina.ai"),
    }.get(record.get("tool"))
    actual_transport = (record.get("backend"), record.get("request_host"))
    if actual_transport != expected_transport:
        raise FormalRunError(f"{label} used a non-frozen backend")
    attempt_log = record.get("http_attempt_log")
    if not isinstance(attempt_log, list) or len(attempt_log) != attempts:
        raise FormalRunError(f"{label} lacks exact physical-attempt ledger")
    starts: list[float] = []
    for attempt_index, entry in enumerate(attempt_log):
        if not isinstance(entry, Mapping):
            raise FormalRunError(f"{label} HTTP attempt {attempt_index} is invalid")
        started = entry.get("started_monotonic_s")
        gate_wait = entry.get("start_gate_wait_s")
        backoff = entry.get("retry_backoff_s")
        if (
            isinstance(started, bool)
            or not isinstance(started, (int, float))
            or not math.isfinite(float(started))
        ):
            raise FormalRunError(
                f"{label} HTTP attempt {attempt_index} lacks monotonic start"
            )
        for field, value in (
            ("start_gate_wait_s", gate_wait),
            ("retry_backoff_s", backoff),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise FormalRunError(
                    f"{label} HTTP attempt {attempt_index} has invalid {field}"
                )
        starts.append(float(started))
    return starts


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=512)
def _fixed_final_grammar_sha256(url: str) -> str:
    reproduction_root = REPOSITORY_ROOT / "reproduction"
    if str(reproduction_root) not in sys.path:
        sys.path.insert(0, str(reproduction_root))
    from paste_repro.live_agent import final_answer_fixed_completion_grammar

    grammar = final_answer_fixed_completion_grammar(url)
    return hashlib.sha256(grammar.encode("utf-8")).hexdigest()


def _validate_task_output_contract(
    task: Mapping[str, Any], *, label: str
) -> None:
    """Validate two guided tool calls and one strict guided final object."""

    recovery = task.get("guided_json_recovery")
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("policy_version")
        != GUIDED_JSON_RECOVERY_POLICY_VERSION
        or recovery.get("recovery_count") != 0
        or recovery.get("parsed_call_count") != 2
    ):
        raise FormalRunError(f"{label} used guided-JSON recovery")
    recovery_calls = recovery.get("calls")
    if not isinstance(recovery_calls, list) or len(recovery_calls) != 2:
        raise FormalRunError(f"{label} has inconsistent guided-JSON recovery evidence")
    for call_index, call in enumerate(recovery_calls):
        if (
            not isinstance(call, Mapping)
            or call.get("call_index") != call_index
            or call.get("policy_version")
            != GUIDED_JSON_RECOVERY_POLICY_VERSION
            or call.get("mode") != "guided_json"
            or call.get("guided_json_requested") is not True
            or call.get("json_parse_attempted") is not True
            or call.get("local_wrap_applied") is not False
            or call.get("parse_succeeded") is not True
            or call.get("contract_succeeded") is not True
            or call.get("recovery_applied") is not False
            or not _is_sha256(call.get("raw_sha256"))
        ):
            raise FormalRunError(
                f"{label} has inconsistent guided-JSON recovery evidence"
            )

    output_contract = task.get("output_contract")
    output_calls = (
        output_contract.get("calls")
        if isinstance(output_contract, Mapping)
        else None
    )
    if (
        not isinstance(output_contract, Mapping)
        or output_contract.get("policy_version") != OUTPUT_CONTRACT_POLICY_VERSION
        or not isinstance(output_calls, list)
        or len(output_calls) != 3
    ):
        raise FormalRunError(f"{label} violates the frozen output contract")
    for call_index in range(2):
        call = output_calls[call_index]
        recovery_call = recovery_calls[call_index]
        if (
            not isinstance(call, Mapping)
            or call.get("call_index") != call_index
            or call.get("mode") != "guided_json"
            or call.get("guided_json_requested") is not True
            or call.get("json_parse_attempted") is not True
            or call.get("local_wrap_applied") is not False
            or call.get("parse_succeeded") is not True
            or call.get("contract_succeeded") is not True
            or call.get("recovery_applied") is not False
            or not _is_sha256(call.get("raw_sha256"))
            or call.get("raw_sha256") != recovery_call.get("raw_sha256")
        ):
            raise FormalRunError(f"{label} violates the guided-call output contract")

    final_contract = task.get("final_answer_contract")
    final_output_call = output_calls[2]
    if (
        not isinstance(final_contract, Mapping)
        or not isinstance(final_output_call, Mapping)
        or dict(final_output_call) != dict(final_contract)
    ):
        raise FormalRunError(f"{label} final-answer contract is not mirrored")

    selected_url = task.get("selected_url")
    expected_url = task.get("expected_url")
    answer = task.get("answer")
    answer_text = answer.get("answer") if isinstance(answer, Mapping) else None
    if (
        not isinstance(selected_url, str)
        or not selected_url.startswith("https://")
        or selected_url != expected_url
        or not isinstance(answer, Mapping)
        or set(answer) != {"answer", "source_url"}
        or answer.get("source_url") != selected_url
        or not isinstance(answer_text, str)
    ):
        raise FormalRunError(f"{label} final answer is not bound to the committed URL")

    tool_rows = task.get("tools")
    committed_visit_urls: list[Any] = []
    if isinstance(tool_rows, list):
        for tool_row in tool_rows:
            invocation = (
                tool_row.get("invocation")
                if isinstance(tool_row, Mapping)
                else None
            )
            if (
                isinstance(invocation, Mapping)
                and invocation.get("tool_name") == "visit"
            ):
                arguments = invocation.get("arguments")
                committed_visit_urls.append(
                    arguments.get("url") if isinstance(arguments, Mapping) else None
                )
    if committed_visit_urls != [[selected_url]]:
        raise FormalRunError(f"{label} lacks one exact committed visit URL")

    expected_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "source_url"],
        "properties": {
            "answer": {"type": "string"},
            "source_url": {"const": selected_url},
        },
    }
    expected_final_fields = {
        "call_index": 2,
        "policy_version": FINAL_ANSWER_CONTRACT_POLICY_VERSION,
        "schema_policy_version": FINAL_ANSWER_SCHEMA_POLICY_VERSION,
        "schema_sha256": _sha256_json(expected_schema),
        "schema_answer_constraint": "type_only_no_length_or_pattern",
        "mode": (
            "guided_grammar_fixed_completion_strict_raw_decode_local_projection"
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
        "object_constructed_locally": True,
        "model_source_url_validated": True,
        "source_url_binding": "exact_committed_selected_url",
        "contract_succeeded": True,
        "max_chars": 480,
        "max_words": 60,
        "target_chars": 360,
        "grammar_policy_version": FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
        "grammar_xgrammar_version": FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
        "grammar_semantic_json_whitespace": "compact",
        "tail_policy": "one_or_more_ascii_spaces_only",
        "tail_validation_succeeded": True,
        "tail_nonempty": True,
        "tail_ascii_space_only": True,
        "fixed_completion_tokens": FIXED_FINAL_COMPLETION_TOKENS,
        "min_tokens": FIXED_FINAL_COMPLETION_TOKENS,
        "max_tokens": FIXED_FINAL_COMPLETION_TOKENS,
        "total_completion_tokens": FIXED_FINAL_COMPLETION_TOKENS,
        "finish_reason": "length",
        "finish_reason_validated": True,
        "token_accounting_succeeded": True,
        "token_counter_method": "transformers_chat_template",
        "token_partition_method": (
            "server_total_minus_local_semantic_tokenization"
        ),
    }
    if any(
        final_contract.get(key) != expected
        for key, expected in expected_final_fields.items()
    ):
        raise FormalRunError(f"{label} violates the strict guided-final contract")
    if (
        final_contract.get("source_url_sha256")
        != hashlib.sha256(selected_url.encode("utf-8")).hexdigest()
        or final_contract.get("grammar_sha256")
        != _fixed_final_grammar_sha256(selected_url)
        or not _is_sha256(final_contract.get("semantic_sha256"))
        or not _is_sha256(final_contract.get("padding_sha256"))
        or not _is_sha256(final_contract.get("raw_sha256"))
        or not _is_sha256(final_contract.get("model_answer_sha256"))
        or not _is_sha256(final_contract.get("pre_projection_canonical_sha256"))
        or not _is_sha256(final_contract.get("canonical_sha256"))
        or task.get("answer_sha256") != _sha256_json(answer)
    ):
        raise FormalRunError(f"{label} has invalid final-answer SHA evidence")

    raw_char_count = final_contract.get("raw_char_count")
    semantic_char_count = final_contract.get("semantic_char_count")
    semantic_byte_count = final_contract.get("semantic_byte_count")
    padding_char_count = final_contract.get("padding_char_count")
    padding_byte_count = final_contract.get("padding_byte_count")
    semantic_token_count = final_contract.get("semantic_token_count")
    padding_token_count = final_contract.get("padding_token_count")
    model_answer_char_count = final_contract.get("model_answer_char_count")
    pre_projection_char_count = final_contract.get("pre_projection_char_count")
    pre_projection_word_count = final_contract.get("pre_projection_word_count")
    canonical_char_count = final_contract.get("canonical_char_count")
    canonical_word_count = final_contract.get("canonical_word_count")
    canonicalization_changed = final_contract.get("canonicalization_changed")
    local_projection_applied = final_contract.get("local_projection_applied")
    word_projection_applied = final_contract.get("word_projection_applied")
    char_projection_applied = final_contract.get("char_projection_applied")
    canonical_sha256 = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
    canonical_words = answer_text.split(" ") if answer_text else []
    if (
        not _is_nonnegative_int(raw_char_count)
        or not _is_nonnegative_int(semantic_char_count)
        or not _is_nonnegative_int(semantic_byte_count)
        or not _is_nonnegative_int(padding_char_count)
        or not _is_nonnegative_int(padding_byte_count)
        or not _is_nonnegative_int(semantic_token_count)
        or not _is_nonnegative_int(padding_token_count)
        or not _is_nonnegative_int(model_answer_char_count)
        or not _is_nonnegative_int(pre_projection_char_count)
        or not _is_nonnegative_int(pre_projection_word_count)
        or not _is_nonnegative_int(canonical_char_count)
        or not _is_nonnegative_int(canonical_word_count)
        or not isinstance(canonicalization_changed, bool)
        or not isinstance(local_projection_applied, bool)
        or not isinstance(word_projection_applied, bool)
        or not isinstance(char_projection_applied, bool)
        or semantic_char_count <= 0
        or semantic_byte_count < semantic_char_count
        or padding_char_count <= 0
        or padding_byte_count != padding_char_count
        or raw_char_count != semantic_char_count + padding_char_count
        or semantic_token_count <= 0
        or padding_token_count <= 0
        or semantic_token_count + padding_token_count
        != FIXED_FINAL_COMPLETION_TOKENS
        or semantic_char_count < model_answer_char_count
        or model_answer_char_count < pre_projection_char_count
        or pre_projection_char_count < canonical_char_count
        or pre_projection_word_count < canonical_word_count
        or canonical_char_count == 0
        or canonical_word_count == 0
        or canonical_char_count != len(answer_text)
        or canonical_word_count != len(canonical_words)
        or final_contract.get("canonical_sha256") != canonical_sha256
        or answer_text != answer_text.strip()
        or "  " in answer_text
        or any(
            character.isspace() and character != " "
            for character in answer_text
        )
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in answer_text
        )
        or len(answer_text) > 480
        or len(canonical_words) > 60
        or "http://" in answer_text.lower()
        or "https://" in answer_text.lower()
    ):
        raise FormalRunError(f"{label} has invalid final-answer count evidence")

    model_answer_sha256 = final_contract["model_answer_sha256"]
    pre_projection_sha256 = final_contract["pre_projection_canonical_sha256"]
    if canonicalization_changed:
        if model_answer_sha256 == pre_projection_sha256:
            raise FormalRunError(f"{label} has inconsistent final canonicalization")
    elif (
        model_answer_char_count != pre_projection_char_count
        or model_answer_sha256 != pre_projection_sha256
    ):
        raise FormalRunError(f"{label} has inconsistent final canonicalization")

    if (
        local_projection_applied
        != (word_projection_applied or char_projection_applied)
        or word_projection_applied != (pre_projection_word_count > 60)
        or canonical_word_count > 60
        or canonical_char_count > 480
    ):
        raise FormalRunError(f"{label} has inconsistent final projection evidence")
    if local_projection_applied:
        if (
            pre_projection_sha256 == canonical_sha256
            or pre_projection_char_count <= canonical_char_count
        ):
            raise FormalRunError(
                f"{label} has inconsistent final projection evidence"
            )
    elif (
        pre_projection_char_count != canonical_char_count
        or pre_projection_word_count != canonical_word_count
        or pre_projection_sha256 != canonical_sha256
    ):
        raise FormalRunError(f"{label} has inconsistent final projection evidence")
    if (
        char_projection_applied
        and pre_projection_char_count <= 480
    ):
        raise FormalRunError(f"{label} has inconsistent final projection evidence")
    if (
        word_projection_applied
        and not char_projection_applied
        and canonical_word_count != 60
    ):
        raise FormalRunError(f"{label} has inconsistent final projection evidence")

    expected_padding_sha256 = hashlib.sha256(
        (" " * padding_char_count).encode("utf-8")
    ).hexdigest()
    if final_contract.get("padding_sha256") != expected_padding_sha256:
        raise FormalRunError(f"{label} has inconsistent ASCII-space tail evidence")


def _validate_fixed_final_llm_events(
    result: Mapping[str, Any],
    *,
    tasks: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    """Bind all 240 exactly-once requests to the v8 fixed-final wire contract."""

    events = result.get("llm_events")
    if not isinstance(events, list) or len(events) != 240:
        raise FormalRunError(f"{label} requires exactly 240 LLM event records")
    tasks_by_id: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in tasks_by_id:
            raise FormalRunError(f"{label} has invalid or duplicate task IDs")
        tasks_by_id[task_id] = task
    seen: set[tuple[str, int]] = set()
    event_usage_by_task: dict[str, dict[str, int]] = {
        task_id: {"prompt_tokens": 0, "completion_tokens": 0}
        for task_id in tasks_by_id
    }
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise FormalRunError(f"{label} LLM event {event_index} is invalid")
        task_id = event.get("task_id")
        call_index = event.get("call_index")
        identity = (task_id, call_index)
        usage = event.get("usage")
        if (
            not isinstance(task_id, str)
            or task_id not in tasks_by_id
            or type(call_index) is not int
            or call_index not in range(3)
            or identity in seen
            or event.get("ok") is not True
            or event.get("attempts") != 1
            or event.get("http_status") != 200
            or not isinstance(event.get("request_id"), str)
            or not isinstance(usage, Mapping)
        ):
            raise FormalRunError(
                f"{label} LLM event {event_index} violates exactly-once evidence"
            )
        seen.add(identity)
        for token_field in ("prompt_tokens", "completion_tokens"):
            token_value = usage.get(token_field)
            if (
                type(token_value) is not int
                or token_value <= 0
            ):
                raise FormalRunError(
                    f"{label} LLM event {event_index} has invalid token usage"
                )
            event_usage_by_task[task_id][token_field] += token_value

        if call_index < 2:
            if (
                event.get("output_mode") != "guided_json"
                or event.get("guided_json_requested") is not True
                or event.get("guided_grammar_requested") is not False
                or event.get("guided_grammar_sha256") is not None
                or event.get("min_tokens") != 0
                or event.get("max_tokens") != 128
                or usage.get("completion_tokens") > 128
            ):
                raise FormalRunError(
                    f"{label} tool-call event {event_index} is not compact guided JSON"
                )
            continue

        final_contract = tasks_by_id[task_id].get("final_answer_contract")
        if not isinstance(final_contract, Mapping):
            raise FormalRunError(f"{label}/{task_id} lacks final contract evidence")
        if (
            event.get("output_mode") != "guided_grammar"
            or event.get("guided_json_requested") is not False
            or event.get("guided_grammar_requested") is not True
            or event.get("guided_grammar_sha256")
            != final_contract.get("grammar_sha256")
            or event.get("min_tokens") != FIXED_FINAL_COMPLETION_TOKENS
            or event.get("max_tokens") != FIXED_FINAL_COMPLETION_TOKENS
            or usage.get("completion_tokens") != FIXED_FINAL_COMPLETION_TOKENS
            or event.get("finish_reason") != "length"
            or event.get("response_sha256") != final_contract.get("raw_sha256")
        ):
            raise FormalRunError(
                f"{label}/{task_id} violates the exact fixed-final LLM event contract"
            )

    expected_identities = {
        (task_id, call_index)
        for task_id in tasks_by_id
        for call_index in range(3)
    }
    if seen != expected_identities:
        raise FormalRunError(f"{label} LLM event identity matrix is incomplete")
    for task_id, task in tasks_by_id.items():
        usage = event_usage_by_task[task_id]
        if (
            task.get("prompt_tokens") != usage["prompt_tokens"]
            or task.get("completion_tokens") != usage["completion_tokens"]
        ):
            raise FormalRunError(f"{label}/{task_id} aggregate token usage mismatch")


def validate_cell_result(
    result: Mapping[str, Any],
    *,
    cell: str,
    block_id: str,
    order_index: int,
    server_instance_id: str,
) -> None:
    policy, speculation = CELL_POLICY[cell]
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise FormalRunError(f"{block_id}/{cell} result config is missing")
    scheduler = config.get("scheduler_environment")
    if not isinstance(scheduler, Mapping):
        raise FormalRunError(f"{block_id}/{cell} scheduler evidence is missing")
    expected_config = {
        "call_graph_mode": "frozen",
        "speculation_mode": speculation,
        "tool_signal_policy": "execution_aware",
        "tool_signal_policy_version": (
            "exact-session-invocation-running-completed-v1"
        ),
        "tool_signal_policy_module_sha256": LIVE_AGENT_SHA256,
        "independent_source_count": 80,
        "replicas": 1,
        "task_count": 80,
        "max_active_tasks": 80,
        "tool_workers": 4,
        "speculative_tool_workers": 2,
        "min_speculative_tool_workers": 0,
        "search_tool_capacity": 3,
        "visit_tool_capacity": 2,
        "search_min_start_interval_s": 0.0,
        "visit_min_start_interval_s": 2.1,
        "max_speculative_pending": 128,
        "speculative_ttl_s": 120.0,
        "tool_http_max_attempts": 2,
        "tool_http_retry_backoff_s": 1.0,
        "tool_http_attempt_start_gate_enabled": True,
        "tool_http_attempt_start_gate_policy_version": (
            "shared-per-tool-monotonic-v1"
        ),
        "tool_http_attempt_min_start_intervals_s": {"visit": 2.1},
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
        "visit_canary_stride": 6,
        "context_padding_tokens": 10000,
        "fixed_final_completion_tokens": FIXED_FINAL_COMPLETION_TOKENS,
        "fixed_final_completion_enabled": True,
        "final_answer_contract_policy_version": (
            FINAL_ANSWER_CONTRACT_POLICY_VERSION
        ),
        "final_answer_schema_policy_version": (
            FINAL_ANSWER_SCHEMA_POLICY_VERSION
        ),
        "final_answer_grammar_policy_version": (
            FINAL_ANSWER_GRAMMAR_POLICY_VERSION
        ),
        "final_answer_grammar_xgrammar_version": (
            FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION
        ),
        "output_contract_policy_version": OUTPUT_CONTRACT_POLICY_VERSION,
        "live_agent_sha256": LIVE_AGENT_SHA256,
        "tool_call_prompt_encoding": "canonical_json_sort_keys_compact",
        "token_count_method": "transformers_chat_template",
        "live_tool_execution": True,
        "recorded_tool_sleep": False,
        "controlled_http_retry": True,
        "shared_bounded_tool_pool": True,
        "authoritative_and_speculative_share_capacity": True,
        "tool_metadata_is_causal": True,
        "tool_result_private_until_exact_commit": True,
        "future_trace_oracle_used": False,
        "frozen_url_is_workload_input": True,
        "workload_file_sha256": FORMAL_WORKLOAD_SHA256,
        "workload_split_id": "live-joint-wikipedia-frozen-formal-v8",
        "workload_split_role": "formal_heldout",
        "workload_formal_eligible": True,
    }
    changed = sorted(
        key for key, expected in expected_config.items()
        if config.get(key) != expected
    )
    if changed:
        raise FormalRunError(
            f"{block_id}/{cell} frozen runner config mismatch: {changed}"
        )
    expected_scheduler = {
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
        "VLLM_PORT": "8100",
        "VLLM_MAX_MODEL_LEN": "16384",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_MAX_NUM_SEQS": "96",
        "VLLM_ENABLE_PREFIX_CACHING": "1",
        "VLLM_USE_V1": "1",
        "VLLM_SCHED_POLICY": policy,
    }
    if policy == "fcfs":
        leaked_joint = sorted(
            key
            for key, value in scheduler.items()
            if key.startswith("VLLM_SCHED_")
            and key != "VLLM_SCHED_POLICY"
            and value is not None
        )
        if leaked_joint:
            raise FormalRunError(
                f"{block_id}/{cell} native FCFS leaked Joint knobs: {leaked_joint}"
            )
    else:
        expected_scheduler.update(
            {
                key: EXPECTED_CONFIG[key]
                for key in FROZEN_JOINT_SCHEDULER_ENV_KEYS
            }
        )
    changed_scheduler = sorted(
        key for key, expected in expected_scheduler.items()
        if scheduler.get(key) != expected
    )
    if changed_scheduler:
        raise FormalRunError(
            f"{block_id}/{cell} scheduler environment mismatch: {changed_scheduler}"
        )
    expected_formal = {
        "block_id": block_id,
        "cell_id": cell,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "fresh_server": True,
        "result_cache_empty": True,
        "broker_drained": True,
    }
    if config.get("formal_run") != expected_formal:
        raise FormalRunError(f"{block_id}/{cell} formal runner evidence mismatch")
    summary = result.get("summary", {})
    if summary.get("all_tasks_succeeded") is not True:
        raise FormalRunError(f"{block_id}/{cell} did not complete every task")
    broker_stats = summary.get("tool", {}).get("broker_stats", {})
    exact_counts = {
        "task_count": 80,
        "successful_task_count": 80,
        "failed_task_count": 0,
    }
    llm_counts = {
        "request_count": 240,
        "successful_request_count": 240,
        "exactly_one_attempt_each": True,
    }
    broker_counts = {
        "authoritative_requests": 160,
        "commits": 160,
        "authoritative_failures": 0,
    }
    if (
        any(summary.get(key) != value for key, value in exact_counts.items())
        or any(
            summary.get("llm", {}).get(key) != value
            for key, value in llm_counts.items()
        )
        or any(broker_stats.get(key) != value for key, value in broker_counts.items())
    ):
        raise FormalRunError(f"{block_id}/{cell} exact completion counts mismatch")
    tasks = result.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 80:
        raise FormalRunError(f"{block_id}/{cell} task evidence is incomplete")
    for task_index, task in enumerate(tasks):
        if (
            not isinstance(task, Mapping)
            or task.get("ok") is not True
            or task.get("visit_canary") != (task_index % 6 == 0)
            or task.get("context_padding_target_tokens") != 10000
        ):
            raise FormalRunError(
                f"{block_id}/{cell} task {task_index} violates frozen canary/padding"
            )
        _validate_task_output_contract(
            task, label=f"{block_id}/{cell} task {task_index}"
        )
    _validate_fixed_final_llm_events(
        result,
        tasks=tasks,
        label=f"{block_id}/{cell}",
    )
    records = result.get("tool_attempt_records")
    if not isinstance(records, list) or not records:
        raise FormalRunError(f"{block_id}/{cell} physical tool evidence is empty")
    visit_attempt_starts: list[float] = []
    canary_records = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FormalRunError(f"{block_id}/{cell} tool record {index} is invalid")
        attempts = record.get("http_attempts")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
            raise FormalRunError(
                f"{block_id}/{cell} tool record {index} lacks HTTP-attempt evidence"
            )
        if attempts == 0:
            queued = record.get("queue_enter_at")
            finished = record.get("finished_at")
            queue_s = record.get("queue_s")
            numeric = (queued, finished, queue_s)
            if (
                record.get("admitted") is not True
                or record.get("speculative") is not True
                or record.get("committed") is not False
                or record.get("cancelled") is not True
                or record.get("outcome") not in {"cancelled", "expired"}
                or record.get("started_at") is not None
                or record.get("start") is not None
                or record.get("worker_id") is not None
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in numeric
                )
                or float(finished) < float(queued)
                or not math.isclose(
                    float(queue_s),
                    float(finished) - float(queued),
                    rel_tol=0.02,
                    abs_tol=0.01,
                )
                or record.get("service_s") != 0.0
                or record.get("saved_service_s") != 0.0
                or any(
                    record.get(field) is not None
                    for field in (
                        "backend",
                        "request_host",
                        "response_status",
                        "bytes_read",
                        "transport_identity_source",
                    )
                )
            ):
                raise FormalRunError(
                    f"{block_id}/{cell} tool record {index} has invalid "
                    "never-started cancellation telemetry"
                )
            continue
        if record.get("canary") is True:
            if (
                record.get("tool") != "visit"
                or record.get("speculative") is not False
                or record.get("speculation_eligible") is not False
            ):
                raise FormalRunError(
                    f"{block_id}/{cell} tool record {index} has invalid canary semantics"
                )
            canary_records += 1
        if record.get("started_at") is None or record.get("start") is None:
            raise FormalRunError(
                f"{block_id}/{cell} tool record {index} claims HTTP without starting"
            )
        starts = _validate_started_tool_record(
            record,
            label=f"{block_id}/{cell} tool record {index}",
            max_http_attempts=int(config["tool_http_max_attempts"]),
            retry_backoff_s=float(config["tool_http_retry_backoff_s"]),
        )
        if record.get("tool") == "visit":
            visit_attempt_starts.extend(starts)
    if canary_records != 14:
        raise FormalRunError(
            f"{block_id}/{cell} expected exactly 14 authoritative-only canary visits"
        )
    ordered_visit_starts = sorted(visit_attempt_starts)
    if len(ordered_visit_starts) < 80:
        raise FormalRunError(f"{block_id}/{cell} lacks live visit GET evidence")
    if any(
        right - left < 2.08
        for left, right in zip(ordered_visit_starts, ordered_visit_starts[1:])
    ):
        raise FormalRunError(
            f"{block_id}/{cell} physical visit starts violated the 2.1s attempt gate"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag")
    parser.add_argument(
        "--cells",
        choices=["A", "A,B,E,F"],
        default="A,B,E,F",
        help="Run the baseline gate only, or the complete three-block matrix.",
    )
    parser.add_argument(
        "--orders",
        help=(
            "Semicolon-separated cell orders for three blocks; quote the shell "
            "value. Both A/B and E/F directions must be balanced."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all frozen inputs without creating output or touching GPUs.",
    )
    return parser.parse_args(argv)


def _relative_bindings(paths: Sequence[Path]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FormalRunError(f"bound code/evidence file is missing: {path}")
        bindings[repository_relative(path)] = sha256_file(path)
    return bindings


def _verify_bindings(bindings: Mapping[str, str]) -> None:
    for relative, expected in bindings.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise FormalRunError(f"bound input changed during formal run: {relative}")


def _runner_command(
    *,
    python: Path,
    workload: Path,
    output: Path,
    cell: str,
    block_id: str,
    order_index: int,
    server_instance_id: str,
    config: Mapping[str, str],
) -> list[str]:
    _policy, speculation = CELL_POLICY[cell]
    return [
        str(python),
        str(RUNNER),
        "--workload", str(workload),
        "--output-dir", str(output),
        "--server-url", f"http://127.0.0.1:{config['VLLM_PORT']}",
        "--model", config["MODEL_ID"],
        "--tokenizer", str(_model_snapshot(config)),
        "--cell-label", f"{block_id}-{cell}",
        "--formal-block-id", block_id,
        "--formal-cell-id", cell,
        "--formal-order-index", str(order_index),
        "--server-instance-id", server_instance_id,
        "--fresh-server",
        "--result-cache-empty",
        "--call-graph-mode", "frozen",
        "--speculation-mode", speculation,
        "--tool-signal-policy", config["PASTE_LIVE_TOOL_SIGNAL_POLICY"],
        "--visit-top-k", "1",
        "--replicas", config["PASTE_LIVE_REPLICAS"],
        "--max-active-tasks", config["PASTE_LIVE_MAX_ACTIVE_TASKS"],
        "--tool-workers", config["PASTE_LIVE_TOOL_WORKERS"],
        "--speculative-tool-workers",
        config["PASTE_LIVE_SPECULATIVE_TOOL_WORKERS"],
        "--min-speculative-tool-workers",
        config["PASTE_LIVE_MIN_SPECULATIVE_TOOL_WORKERS"],
        "--search-tool-capacity", config["PASTE_LIVE_SEARCH_TOOL_CAPACITY"],
        "--visit-tool-capacity", config["PASTE_LIVE_VISIT_TOOL_CAPACITY"],
        "--search-min-start-interval-s",
        config["PASTE_LIVE_SEARCH_MIN_START_INTERVAL_S"],
        "--visit-min-start-interval-s",
        config["PASTE_LIVE_VISIT_MIN_START_INTERVAL_S"],
        "--max-speculative-pending",
        config["PASTE_LIVE_MAX_SPECULATIVE_PENDING"],
        "--speculative-ttl-s", config["PASTE_LIVE_SPECULATIVE_TTL_S"],
        "--tool-timeout-s", config["PASTE_LIVE_TOOL_TIMEOUT_S"],
        "--tool-http-max-attempts",
        config["PASTE_LIVE_TOOL_HTTP_MAX_ATTEMPTS"],
        "--tool-http-retry-backoff-s",
        config["PASTE_LIVE_TOOL_HTTP_RETRY_BACKOFF_S"],
        "--tool-http-attempt-start-gate",
        "--tool-service-hint-s", config["PASTE_LIVE_TOOL_SERVICE_HINT_S"],
        "--visit-mode", "jina",
        "--search-mode", "bing",
        "--search-max-results", config["PASTE_LIVE_SEARCH_MAX_RESULTS"],
        "--visit-max-chars", config["PASTE_LIVE_VISIT_MAX_CHARS"],
        "--request-timeout-s", config["PASTE_LIVE_REQUEST_TIMEOUT_S"],
        "--max-tokens-tool", config["PASTE_LIVE_MAX_TOKENS_TOOL"],
        "--max-tokens-answer", config["PASTE_LIVE_MAX_TOKENS_ANSWER"],
        "--fixed-final-completion-tokens",
        config["PASTE_LIVE_FIXED_FINAL_COMPLETION_TOKENS"],
        "--predicted-visit-result-tokens",
        config["PASTE_LIVE_PREDICTED_VISIT_RESULT_TOKENS"],
        "--context-padding-tokens",
        config["PASTE_LIVE_CONTEXT_PADDING_TOKENS"],
        "--queue-sample-interval-s",
        config["PASTE_LIVE_QUEUE_SAMPLE_INTERVAL_S"],
        "--visit-canary-stride", config["PASTE_LIVE_VISIT_CANARY_STRIDE"],
    ]


def _model_snapshot(config: Mapping[str, str]) -> Path:
    hf_home = Path(config["HF_HOME"])
    key = "models--" + config["MODEL_ID"].replace("/", "--")
    return hf_home / key / "snapshots" / config["MODEL_REVISION"]


def _run_logged(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        completed = subprocess.run(
            list(command),
            cwd=REPOSITORY_ROOT,
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return completed.returncode


def _run_cell(
    *,
    run_root: Path,
    block_id: str,
    block_number: int,
    cell: str,
    order_index: int,
    config: Mapping[str, str],
    workload: Path,
    python: Path,
    bindings: Mapping[str, str],
    config_path: Path,
    accepted_gate: Mapping[str, str] | None,
) -> Path:
    _verify_bindings(bindings)
    cell_root = run_root / f"block-{block_number:02d}" / cell
    cell_root.mkdir(parents=True, exist_ok=False)
    server_dir = cell_root / "server"
    state_dir = cell_root / "state"
    server_dir.mkdir()
    state_dir.mkdir()
    result_dir = cell_root / "evidence"
    lifecycle_stdout = cell_root / "server_lifecycle.stdout.log"
    lifecycle_stderr = cell_root / "server_lifecycle.stderr.log"
    runner_stdout = cell_root / "runner.stdout.log"
    runner_stderr = cell_root / "runner.stderr.log"
    server_instance_id = str(uuid.uuid4())
    policy, speculation = CELL_POLICY[cell]
    cell_env = _cell_environment(config, cell=cell)
    cell_env.update(
        {
            "VLLM_REQUIRE_NEW": "1",
            "VLLM_STATE_DIR": str(state_dir),
            "VLLM_LOG_DIR": str(server_dir),
            "VLLM_HOOK_DIR": str(REPOSITORY_ROOT / "scripts/pythonhooks"),
            "MODEL_SNAPSHOT": str(_model_snapshot(config)),
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = _runner_command(
        python=python,
        workload=workload,
        output=result_dir,
        cell=cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
        config=config,
    )
    effective = {
        "schema": "paste_repro.live_joint_formal_cell_config",
        "version": 1,
        "block_id": block_id,
        "block_number": block_number,
        "cell_id": cell,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "fresh_server_required": True,
        "result_cache_empty_required": True,
        "llm_scheduler": policy,
        "speculation_mode": speculation,
        "tool_signal_policy": config["PASTE_LIVE_TOOL_SIGNAL_POLICY"],
        "call_graph_mode": "frozen",
        "live_backends": {"search": "bing_html_search", "visit": "r_jina_ai"},
        "recorded_tool_sleep": False,
        "controlled_http_retry": True,
        "tool_http_max_attempts": int(
            config["PASTE_LIVE_TOOL_HTTP_MAX_ATTEMPTS"]
        ),
        "tool_http_retry_backoff_s": float(
            config["PASTE_LIVE_TOOL_HTTP_RETRY_BACKOFF_S"]
        ),
        "tool_http_attempt_start_gate": True,
        "tool_http_attempt_start_gate_policy_version": (
            "shared-per-tool-monotonic-v1"
        ),
        "min_speculative_tool_workers": int(
            config["PASTE_LIVE_MIN_SPECULATIVE_TOOL_WORKERS"]
        ),
        "visit_canary_policy": {
            "stride": int(config["PASTE_LIVE_VISIT_CANARY_STRIDE"]),
            "known_ineligible_prediction": "skip_before_enqueue",
        },
        "guided_json_recovery_gate": {
            "policy_version": config[
                "PASTE_LIVE_GUIDED_JSON_RECOVERY_POLICY_VERSION"
            ],
            "module_sha256": config[
                "PASTE_LIVE_GUIDED_JSON_RECOVERY_MODULE_SHA256"
            ],
            "maximum_recovery_count_per_task": int(
                config["PASTE_LIVE_FORMAL_MAX_GUIDED_JSON_RECOVERIES"]
            ),
            "required_parsed_call_count_per_task": 2,
        },
        "output_contract_gate": {
            "policy_version": config[
                "PASTE_LIVE_OUTPUT_CONTRACT_POLICY_VERSION"
            ],
            "guided_json_call_indices": [0, 1],
            "guided_grammar_fixed_completion_call_index": 2,
            "final_answer_schema_policy_version": config[
                "PASTE_LIVE_FINAL_ANSWER_SCHEMA_POLICY_VERSION"
            ],
            "final_answer_schema_constraint": (
                "type_only_no_length_or_pattern"
            ),
            "final_answer_policy_version": config[
                "PASTE_LIVE_FINAL_ANSWER_CONTRACT_POLICY_VERSION"
            ],
            "final_answer_grammar_policy_version": config[
                "PASTE_LIVE_FINAL_ANSWER_GRAMMAR_POLICY_VERSION"
            ],
            "final_answer_grammar_xgrammar_version": config[
                "PASTE_LIVE_FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION"
            ],
            "fixed_final_completion_tokens": int(
                config["PASTE_LIVE_FIXED_FINAL_COMPLETION_TOKENS"]
            ),
            "min_tokens_equals_max_tokens": True,
            "strict_ascii_space_tail_required": True,
            "strict_json_parse_required": True,
            "guided_final_recovery_allowed": False,
            "model_source_url_validation_required": True,
            "local_object_construction_required": True,
            "local_projection_allowed": True,
            "projection_evidence_required": True,
            "exact_committed_url_binding_required": True,
        },
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
        "future_information_used": False,
        "workload": {
            "path": repository_relative(workload),
            "sha256": FORMAL_WORKLOAD_SHA256,
        },
        "frozen_config": {
            "path": repository_relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "accepted_block1_baseline_gate": accepted_gate,
        "environment": {
            key: cell_env.get(key)
            for key in sorted(EXPECTED_CONFIG)
            if key.startswith(("CUDA_", "MODEL_", "VLLM_"))
        },
        "runner_arguments": command[2:],
    }
    effective["environment"].update(
        {
            "HF_HOME": cell_env["HF_HOME"],
            "MODEL_SNAPSHOT": cell_env["MODEL_SNAPSHOT"],
            "VLLM_HOOK_DIR": cell_env["VLLM_HOOK_DIR"],
            "VLLM_REQUIRE_NEW": cell_env["VLLM_REQUIRE_NEW"],
            "VLLM_SCHED_POLICY": cell_env["VLLM_SCHED_POLICY"],
        }
    )
    write_json_atomic(cell_root / "effective_config.json", effective)

    print(
        f"[{block_id}] starting cell {cell} ({order_index + 1}/4): "
        f"policy={policy}, speculation={speculation}",
        flush=True,
    )
    started = False
    primary_error: FormalRunError | None = None
    try:
        start_code = _run_logged(
            [str(START_SERVER)],
            env=cell_env,
            stdout_path=lifecycle_stdout,
            stderr_path=lifecycle_stderr,
        )
        if start_code != 0:
            raise FormalRunError(
                f"{block_id}/{cell} fresh vLLM start failed; see lifecycle logs"
            )
        started = True
        runner_code = _run_logged(
            command,
            env=cell_env,
            stdout_path=runner_stdout,
            stderr_path=runner_stderr,
        )
        if runner_code != 0:
            raise FormalRunError(
                f"{block_id}/{cell} runner failed; see runner logs"
            )
    except FormalRunError as exc:
        primary_error = exc
    finally:
        if started:
            stop_code = _run_logged(
                [str(STOP_SERVER)],
                env=cell_env,
                stdout_path=lifecycle_stdout,
                stderr_path=lifecycle_stderr,
            )
            if stop_code != 0 and primary_error is None:
                primary_error = FormalRunError(
                    f"{block_id}/{cell} vLLM did not stop cleanly"
                )
    if primary_error is not None:
        raise primary_error

    result_path = result_dir / "result.json"
    timeline_path = result_dir / "queue_timeline.jsonl"
    server_log = server_dir / f"vllm_{config['VLLM_PORT']}.log"
    for required in (result_path, timeline_path, server_log):
        if not required.is_file():
            raise FormalRunError(f"{block_id}/{cell} evidence is missing: {required}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_cell_result(
        result,
        cell=cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
    )
    cell_manifest = {
        "schema": "paste_repro.live_joint_formal_cell_evidence",
        "version": 1,
        "block_id": block_id,
        "cell_id": cell,
        "order_index": order_index,
        "server_instance_id": server_instance_id,
        "accepted_block1_baseline_gate": accepted_gate,
        "evidence": {
            repository_relative(path): sha256_file(path)
            for path in (
                cell_root / "effective_config.json",
                result_path,
                timeline_path,
                server_log,
                lifecycle_stdout,
                lifecycle_stderr,
                runner_stdout,
                runner_stderr,
            )
        },
    }
    write_json_atomic(cell_root / "cell_manifest.json", cell_manifest)
    _verify_bindings(bindings)
    print(f"[{block_id}] cell {cell} completed and server stopped", flush=True)
    return cell_root


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_tag) is None:
        raise FormalRunError(
            "RUN_TAG must use only letters, digits, dot, underscore, and dash"
        )
    config_path = args.config.resolve()
    repository_relative(config_path)
    config = load_frozen_config(config_path)
    orders = validate_orders(
        args.orders or config["PASTE_LIVE_FORMAL_DEFAULT_ORDERS"],
        baseline_only=args.cells == "A",
    )
    python = Path(config["PASTE_ENV_PREFIX"]) / "bin/python"
    if not python.is_file():
        raise FormalRunError(f"reproduction Python is missing: {python}")
    validate_entrypoints(python=python)
    workload = (REPOSITORY_ROOT / config["PASTE_LIVE_FORMAL_WORKLOAD"]).resolve()
    if repository_relative(workload) != config["PASTE_LIVE_FORMAL_WORKLOAD"]:
        raise FormalRunError("formal workload path is not the frozen repository path")
    if not workload.is_file() or sha256_file(workload) != FORMAL_WORKLOAD_SHA256:
        raise FormalRunError("formal workload raw-file SHA256 mismatch")
    workload_validation = validate_formal_workload(python=python, workload=workload)
    model_snapshot = _model_snapshot(config)
    if not model_snapshot.is_dir() or not (model_snapshot / "config.json").is_file():
        raise FormalRunError(f"pinned model snapshot is missing: {model_snapshot}")
    grammar_feasibility = validate_fixed_final_grammar_feasibility(
        workload=workload,
        model_snapshot=model_snapshot,
    )
    offered_concurrency = int(config["PASTE_LIVE_MAX_ACTIVE_TASKS"])
    native_sequence_ceiling = int(config["VLLM_MAX_NUM_SEQS"])
    if offered_concurrency <= 64:
        raise FormalRunError("formal v8 offered concurrency must exceed 64")
    if offered_concurrency >= native_sequence_ceiling:
        raise FormalRunError("max-num-seqs must be strictly above offered concurrency")

    run_base = (REPOSITORY_ROOT / config["PASTE_LIVE_FORMAL_RUN_BASE"]).resolve()
    repository_relative(run_base)
    run_root = run_base / args.run_tag
    lock_path = run_base / f".{args.run_tag}.lock"
    if run_root.exists() or lock_path.exists():
        raise FormalRunError(f"run output or lock already exists: {run_root}")

    bound_paths = (config_path, workload, *BOUND_CODE_PATHS)
    bindings = _relative_bindings(bound_paths)
    if args.check_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "check_only": True,
                    "gpu_or_server_touched": False,
                    "run_tag": args.run_tag,
                    "cells": args.cells,
                    "orders": orders,
                    "config_sha256": bindings[repository_relative(config_path)],
                    "workload_validation": workload_validation,
                    "fixed_final_grammar_feasibility": grammar_feasibility,
                    "guided_json_recovery_gate": {
                        "policy_version": GUIDED_JSON_RECOVERY_POLICY_VERSION,
                        "module_sha256": LIVE_AGENT_SHA256,
                        "maximum_recovery_count_per_task": 0,
                        "required_parsed_call_count_per_task": 2,
                    },
                    "output_contract_gate": {
                        "policy_version": OUTPUT_CONTRACT_POLICY_VERSION,
                        "guided_json_call_indices": [0, 1],
                        "guided_grammar_fixed_completion_call_index": 2,
                        "final_answer_schema_policy_version": (
                            FINAL_ANSWER_SCHEMA_POLICY_VERSION
                        ),
                        "final_answer_schema_constraint": (
                            "type_only_no_length_or_pattern"
                        ),
                        "final_answer_policy_version": (
                            FINAL_ANSWER_CONTRACT_POLICY_VERSION
                        ),
                        "final_answer_grammar_policy_version": (
                            FINAL_ANSWER_GRAMMAR_POLICY_VERSION
                        ),
                        "final_answer_grammar_xgrammar_version": (
                            FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION
                        ),
                        "fixed_final_completion_tokens": (
                            FIXED_FINAL_COMPLETION_TOKENS
                        ),
                        "min_tokens_equals_max_tokens": True,
                        "strict_ascii_space_tail_required": True,
                        "strict_json_parse_required": True,
                        "guided_final_recovery_allowed": False,
                        "model_source_url_validation_required": True,
                        "local_object_construction_required": True,
                        "local_projection_allowed": True,
                        "projection_evidence_required": True,
                        "exact_committed_url_binding_required": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    run_base.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise FormalRunError(f"another process reserved run tag {args.run_tag}") from exc
    try:
        run_root.mkdir()
        shutil.copy2(config_path, run_root / "frozen_config.env")
        write_json_atomic(run_root / "workload_validation.json", workload_validation)
        write_json_atomic(
            run_root / "fixed_final_grammar_feasibility.json",
            grammar_feasibility,
        )
        plan = {
            "schema": "paste_repro.live_joint_formal_matrix_plan",
            "version": 1,
            "created_wall_s": time.time(),
            "run_tag": args.run_tag,
            "baseline_only": args.cells == "A",
            "cells": args.cells,
            "orders": orders,
            "baseline_gate_uses_candidate_performance": False,
            "fresh_server_per_cell": True,
            "cross_cell_result_cache": False,
            "bindings": bindings,
        }
        write_json_atomic(run_root / "run_plan.json", plan)

        completed_cells: list[dict[str, Any]] = []
        gate_ref: dict[str, str] | None = None
        baseline_gate_refs: list[dict[str, str]] = []
        for block_number, order in enumerate(orders, 1):
            block_id = f"{args.run_tag}-block-{block_number}"
            if block_number > 1:
                if gate_ref is None:
                    raise FormalRunError("later block has no accepted block-1 baseline gate")
                block_root = run_root / f"block-{block_number:02d}"
                block_root.mkdir(parents=True, exist_ok=False)
                write_json_atomic(
                    block_root / "accepted_block1_baseline_gate.json", gate_ref
                )
                # _run_cell creates only the cell directory under an existing block.
            for order_index, cell in enumerate(order):
                cell_root = _run_cell(
                    run_root=run_root,
                    block_id=block_id,
                    block_number=block_number,
                    cell=cell,
                    order_index=order_index,
                    config=config,
                    workload=workload,
                    python=python,
                    bindings=bindings,
                    config_path=config_path,
                    accepted_gate=gate_ref,
                )
                completed_cells.append(
                    {
                        "block_id": block_id,
                        "cell_id": cell,
                        "order_index": order_index,
                        "path": repository_relative(cell_root),
                    }
                )
                if cell == "A":
                    gate_path = (
                        run_root
                        / f"block-{block_number:02d}"
                        / "baseline_gate.json"
                    )
                    gate = evaluate_baseline_gate(
                        cell_root / "evidence/result.json",
                        cell_root / "evidence/queue_timeline.jsonl",
                        block_id=block_id,
                    )
                    write_json_atomic(gate_path, gate)
                    current_gate_ref = {
                        "path": repository_relative(gate_path),
                        "sha256": sha256_file(gate_path),
                    }
                    baseline_gate_refs.append(current_gate_ref)
                    if block_number == 1:
                        gate_ref = current_gate_ref
                    if not gate["accepted"]:
                        raise FormalRunError(
                            f"block-{block_number} A failed the preregistered "
                            "80-offered native dual-queue gate"
                        )
                    print(f"[{block_id}] baseline resource gate accepted", flush=True)

        _verify_bindings(bindings)
        write_json_atomic(
            run_root / "completed_matrix.json",
            {
                "schema": "paste_repro.live_joint_formal_matrix_completion",
                "version": 1,
                "completed_wall_s": time.time(),
                "run_tag": args.run_tag,
                "baseline_only": args.cells == "A",
                "orders": orders,
                "accepted_block1_baseline_gate": gate_ref,
                "accepted_baseline_gates": baseline_gate_refs,
                "completed_cells": completed_cells,
                "bindings": bindings,
            },
        )
        print(f"Formal run completed: {run_root}", flush=True)
        return 0
    except BaseException as exc:
        if run_root.is_dir():
            failure = {
                "schema": "paste_repro.live_joint_formal_matrix_failure",
                "version": 1,
                "failed_wall_s": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_json_atomic(run_root / "failure.json", failure)
        raise
    finally:
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
