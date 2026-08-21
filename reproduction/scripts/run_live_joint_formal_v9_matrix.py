#!/usr/bin/env python3
"""Run the prospective v9 A/B/E/F matrix after the SHA-bound F0 screen.

This module intentionally wraps, rather than edits, the frozen v8 execution
kernel.  The completed v9 development screen binds that kernel byte-for-byte;
keeping it unchanged preserves the causal chain from development selection to
the untouched v9 formal workload.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "reproduction/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_live_joint_formal_matrix as formal  # type: ignore


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "reproduction/configs/live_joint_formal_v9_matrix.env.example"
)
V9_PROTOCOL = (
    REPOSITORY_ROOT
    / "reproduction/results/live_joint/V9_FORMAL_MATRIX_PROTOCOL.md"
)
V9_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v9.json"
)
DEVELOPMENT_ROOT = (
    REPOSITORY_ROOT
    / "reproduction/artifacts/live_joint/development/v9_screen/v9-screen-r1"
)
COMPLETED_SCREEN = DEVELOPMENT_ROOT / "completed_screen.json"
STRICT_DEVELOPMENT_SELECTION = (
    DEVELOPMENT_ROOT / "strict_development_selection.json"
)
SELECTED_TRANSPORT = DEVELOPMENT_ROOT / "stage-0/selected_transport.json"

FORMAL_WORKLOAD_SHA256 = (
    "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20"
)
FORMAL_CANONICAL_SHA256 = (
    "de588fcbd46c1181156f5a6e49e0264c785c00c43e0d8c2a62698fb6217e3ce7"
)
FORMAL_SOURCES_SHA256 = (
    "750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c"
)
COMPLETED_SCREEN_SHA256 = (
    "40b4a8033529883f26c1f298d54a92a69e4fcfb6cb942a8d5f70c98fc86481f3"
)
STRICT_DEVELOPMENT_SELECTION_SHA256 = (
    "7f7c9de71f341741192de78ab8596b9cb01721fe211ec3faed79ee33bd7dc7cc"
)
SELECTED_TRANSPORT_SHA256 = (
    "3c44458963c65deb55b35dfa5a2ff888d5e1ec4cb6c0ff350ebe41e53612dc0d"
)
SELECTED_POLICY = "F0"
SELECTED_VISIT_INTERVAL_S = 2.5
SELECTED_MIN_SPECULATIVE_TOOL_WORKERS = 0
FORMAL_SPLIT_ID = "live-joint-wikipedia-frozen-formal-v9"

# These are prospective constants, not hashes sampled at run time.  A changed
# dependency therefore fails before a server can be started.  The wrapper,
# v9 config, and v9 protocol are additionally recorded in each run's dynamic
# immutable binding map.
EXPECTED_FROZEN_RUNTIME_SHA256: dict[str, str] = {
    "reproduction/paste_repro/live_agent.py": (
        "6dab494fa65749b1d60a5b5cbfbb4d0eed3c804b91b3646e0388c707cb7ade8f"
    ),
    "reproduction/paste_repro/live_broker.py": (
        "a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27"
    ),
    "reproduction/paste_repro/live_executor.py": (
        "1605c6a3f0002979d11e70c765684b38c6228bf5d69316cd223436aae7179956"
    ),
    "reproduction/results/live_joint/LIVE_TOOL_LLM_PROTOCOL.md": (
        "5ffb2b20582d798a7350f78c42e975e2e516b890486c76148f0edd3ab2c295b6"
    ),
    "reproduction/scripts/run_live_joint_formal_matrix.py": (
        "e735970a86483c17dceb7e79255ea0c11d8144c1b09b141566b8bcb2185f41db"
    ),
    "reproduction/scripts/start_vllm.sh": (
        "45154b12d870e319781f153c588a9944bfdbb655999e3139f394c7f656eb6a40"
    ),
    "reproduction/scripts/stop_vllm.sh": (
        "90f174e526c26190e927597ee5ff7c32f1f89a62760a937d8b619cebee34f7dd"
    ),
    "reproduction/scripts/validate_live_joint_formal_workload.py": (
        "88d75a7f00d8c8495e0612f93321a26b2193ba6b7bdf89a0674567f0581b0ff4"
    ),
    "scripts/pythonhooks/sched_policy_patch.py": (
        "9acd2316dddddd6a879614336550d2097c47958ef0b56a7da786b55ecf7b8791"
    ),
    "scripts/run_live_tool_llm_experiment.py": (
        "2672bd58a06de204e0a6a92622b688c453cfca36422660c49e32afae5b70afa3"
    ),
}
EXPECTED_FORMAL_REGISTRATION_SHA256: dict[str, str] = {
    "reproduction/configs/live_joint_formal_v9_matrix.env.example": (
        "946db6793569d6d9c33215d515318a4fffaf869d8a2def65daa43f2e798c09ac"
    ),
    "reproduction/results/live_joint/V9_FORMAL_MATRIX_PROTOCOL.md": (
        "fb9a626e4b9181560de35403b833c26c838297e6a1ae243cc2e8b02440bcf4d6"
    ),
}


class FormalV9RunError(formal.FormalRunError):
    """Fail-closed error raised before the frozen formal kernel is entered."""


_LEGACY_VALIDATE_CELL_RESULT = formal.validate_cell_result
_LEGACY_WRITE_JSON_ATOMIC = formal.write_json_atomic


def _relative(path: Path) -> str:
    return formal.repository_relative(path.resolve())


def _load_exact_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if not path.is_file() or formal.sha256_file(path) != expected_sha256:
        raise FormalV9RunError(f"frozen evidence SHA256 mismatch: {_relative(path)}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalV9RunError(f"frozen evidence is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FormalV9RunError(f"frozen evidence is not a JSON object: {path}")
    return value


def validate_frozen_runtime() -> dict[str, str]:
    """Verify the prospective runtime allowlist without sampling new hashes."""

    frozen = {
        **EXPECTED_FROZEN_RUNTIME_SHA256,
        **EXPECTED_FORMAL_REGISTRATION_SHA256,
    }
    for relative, expected in frozen.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or formal.sha256_file(path) != expected:
            raise FormalV9RunError(
                f"frozen formal-v9 runtime SHA256 mismatch: {relative}"
            )
    return dict(sorted(frozen.items()))


def validate_development_selection() -> dict[str, Any]:
    """Replay the exact development decision that permits formal v9."""

    completed = _load_exact_json(COMPLETED_SCREEN, COMPLETED_SCREEN_SHA256)
    selection = _load_exact_json(
        STRICT_DEVELOPMENT_SELECTION,
        STRICT_DEVELOPMENT_SELECTION_SHA256,
    )
    transport = _load_exact_json(SELECTED_TRANSPORT, SELECTED_TRANSPORT_SHA256)

    if (
        completed.get("schema")
        != "paste_repro.live_joint_v9_development_screen_completion"
        or completed.get("version") != 1
        or completed.get("development_only") is not True
        or completed.get("formal_eligible") is not False
        or completed.get("formal_evidence_eligible") is not False
        or completed.get("development_selection_passed") is not True
        or completed.get("selected_policy") != SELECTED_POLICY
        or completed.get("selected_transport")
        != {
            "path": _relative(SELECTED_TRANSPORT),
            "sha256": SELECTED_TRANSPORT_SHA256,
        }
        or completed.get("strict_development_selection")
        != {
            "path": _relative(STRICT_DEVELOPMENT_SELECTION),
            "sha256": STRICT_DEVELOPMENT_SELECTION_SHA256,
        }
    ):
        raise FormalV9RunError("completed development screen is not the F0 winner")

    if (
        selection.get("schema") != "paste_repro.live_joint_v9_development_screen"
        or selection.get("version") != 1
        or selection.get("valid") is not True
        or selection.get("development_only") is not True
        or selection.get("formal_eligible") is not False
        or selection.get("formal_evidence_eligible") is not False
        or selection.get("development_selection_passed") is not True
        or selection.get("selected_policy") != SELECTED_POLICY
        or selection.get("selected_visit_interval_s")
        != SELECTED_VISIT_INTERVAL_S
    ):
        raise FormalV9RunError("strict development selection is not accepted F0/2.5")
    candidate_passed = selection.get("candidate_passed")
    if (
        not isinstance(candidate_passed, Mapping)
        or candidate_passed.get("F0") is not True
        or selection.get("F1_incremental_passed") is not False
    ):
        raise FormalV9RunError("strict selection does not causally select F0 over F1")
    identity = selection.get("common_code_and_config_identity")
    identity_cells = identity.get("cells") if isinstance(identity, Mapping) else None
    if not isinstance(identity_cells, Mapping):
        raise FormalV9RunError("strict selection lacks treatment identity evidence")
    f0_rows = {
        key: value
        for key, value in identity_cells.items()
        if isinstance(key, str) and key.endswith("/F0")
    }
    if len(f0_rows) != 2 or any(
        not isinstance(value, Mapping)
        or value.get("min_speculative_tool_workers")
        != SELECTED_MIN_SPECULATIVE_TOOL_WORKERS
        or value.get("speculation_mode") != "visit"
        for value in f0_rows.values()
    ):
        raise FormalV9RunError("strict selection does not prove F0 min-spec=0 twice")

    if (
        transport.get("schema")
        != "paste_repro.live_joint_v9_development_transport_selection"
        or transport.get("version") != 1
        or transport.get("valid") is not True
        or transport.get("development_only") is not True
        or transport.get("formal_eligible") is not False
        or transport.get("formal_evidence_eligible") is not False
        or transport.get("selected_visit_interval_s")
        != SELECTED_VISIT_INTERVAL_S
        or transport.get("candidate_performance_observed_or_used") is not False
        or transport.get("selection_input_cells") != ["A"]
    ):
        raise FormalV9RunError("selected transport is not causal baseline-only 2.5s")

    screen_bindings = completed.get("bindings")
    if not isinstance(screen_bindings, Mapping):
        raise FormalV9RunError("completed screen lacks its immutable bindings")
    mismatched = sorted(
        relative
        for relative, expected in EXPECTED_FROZEN_RUNTIME_SHA256.items()
        if screen_bindings.get(relative) != expected
    )
    if mismatched:
        raise FormalV9RunError(
            "development screen/runtime causal binding mismatch: "
            + ", ".join(mismatched)
        )

    return {
        "valid": True,
        "completed_screen": {
            "path": _relative(COMPLETED_SCREEN),
            "sha256": COMPLETED_SCREEN_SHA256,
        },
        "strict_development_selection": {
            "path": _relative(STRICT_DEVELOPMENT_SELECTION),
            "sha256": STRICT_DEVELOPMENT_SELECTION_SHA256,
        },
        "selected_transport": {
            "path": _relative(SELECTED_TRANSPORT),
            "sha256": SELECTED_TRANSPORT_SHA256,
        },
        "selected_policy": SELECTED_POLICY,
        "selected_visit_interval_s": SELECTED_VISIT_INTERVAL_S,
        "selected_min_speculative_tool_workers": (
            SELECTED_MIN_SPECULATIVE_TOOL_WORKERS
        ),
        "candidate_performance_used_for_transport_selection": False,
    }


def _validate_v9_physical_visit_gate(result: Mapping[str, Any], *, label: str) -> None:
    records = result.get("tool_attempt_records")
    if not isinstance(records, list):
        raise FormalV9RunError(f"{label} lacks physical tool-attempt records")
    starts: list[float] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        attempts = record.get("http_attempts")
        attempt_records = record.get("http_attempt_log")
        if (
            type(attempts) is not int
            or attempts not in {0, 1}
            or not isinstance(attempt_records, list)
            or len(attempt_records) != attempts
        ):
            raise FormalV9RunError(f"{label} violates the zero-retry formal gate")
        if attempts == 1 and (
            not isinstance(attempt_records[0], Mapping)
            or attempt_records[0].get("attempt") != 1
            or attempt_records[0].get("retried") is not False
        ):
            raise FormalV9RunError(f"{label} has inconsistent zero-retry telemetry")
        if (
            attempts == 1
            and record.get("speculative") is True
            and record.get("committed") is not True
        ):
            raise FormalV9RunError(
                f"{label} violates the zero-wasted-speculative-service gate"
            )
        if record.get("tool") != "visit":
            continue
        for attempt in attempt_records:
            if not isinstance(attempt, Mapping):
                continue
            start = attempt.get("started_monotonic_s")
            if (
                isinstance(start, (int, float))
                and not isinstance(start, bool)
                and math.isfinite(float(start))
            ):
                starts.append(float(start))
    starts.sort()
    if len(starts) < 80:
        raise FormalV9RunError(f"{label} lacks at least 80 live visit GET starts")
    # Preserve the frozen runner's 20 ms telemetry tolerance at the selected
    # 2.5 second transport interval.
    if any(right - left < 2.48 for left, right in zip(starts, starts[1:])):
        raise FormalV9RunError(f"{label} violated the selected 2.5s visit gate")


def validate_v9_cell_result(
    result: Mapping[str, Any],
    *,
    cell: str,
    block_id: str,
    order_index: int,
    server_instance_id: str,
) -> None:
    """Add v9 invariants, then reuse every frozen v8 evidence check.

    The compatibility copy changes only the two fields hard-coded as v8/2.1
    in the frozen validator.  Their original v9 values are checked first, and
    the stronger 2.5 second physical-start invariant is checked afterwards.
    The evidence object on disk is never changed.
    """

    config = result.get("config")
    if not isinstance(config, Mapping):
        raise FormalV9RunError(f"{block_id}/{cell} result config is missing")
    expected_v9 = {
        "visit_min_start_interval_s": SELECTED_VISIT_INTERVAL_S,
        "tool_http_attempt_min_start_intervals_s": {
            "visit": SELECTED_VISIT_INTERVAL_S
        },
        "workload_split_id": FORMAL_SPLIT_ID,
        "workload_file_sha256": FORMAL_WORKLOAD_SHA256,
        "min_speculative_tool_workers": SELECTED_MIN_SPECULATIVE_TOOL_WORKERS,
    }
    changed = sorted(
        key for key, expected in expected_v9.items() if config.get(key) != expected
    )
    if changed:
        raise FormalV9RunError(
            f"{block_id}/{cell} formal-v9 treatment mismatch: {changed}"
        )

    compatible = dict(result)
    compatible_config = copy.deepcopy(dict(config))
    compatible_config["visit_min_start_interval_s"] = 2.1
    compatible_config["tool_http_attempt_min_start_intervals_s"] = {
        "visit": 2.1
    }
    compatible_config["workload_split_id"] = (
        "live-joint-wikipedia-frozen-formal-v8"
    )
    compatible["config"] = compatible_config
    _LEGACY_VALIDATE_CELL_RESULT(
        compatible,
        cell=cell,
        block_id=block_id,
        order_index=order_index,
        server_instance_id=server_instance_id,
    )
    _validate_v9_physical_visit_gate(result, label=f"{block_id}/{cell}")


def _formal_v9_provenance() -> dict[str, Any]:
    return {
        "completed_screen": {
            "path": _relative(COMPLETED_SCREEN),
            "sha256": COMPLETED_SCREEN_SHA256,
        },
        "strict_development_selection": {
            "path": _relative(STRICT_DEVELOPMENT_SELECTION),
            "sha256": STRICT_DEVELOPMENT_SELECTION_SHA256,
        },
        "selected_transport": {
            "path": _relative(SELECTED_TRANSPORT),
            "sha256": SELECTED_TRANSPORT_SHA256,
        },
        "selected_policy": SELECTED_POLICY,
        "selected_visit_interval_s": SELECTED_VISIT_INTERVAL_S,
        "selected_min_speculative_tool_workers": (
            SELECTED_MIN_SPECULATIVE_TOOL_WORKERS
        ),
        "maximum_observed_http_retries_per_cell": 0,
        "zero_wasted_speculative_service_required": True,
        "live_broker_sha256": EXPECTED_FROZEN_RUNTIME_SHA256[
            "reproduction/paste_repro/live_broker.py"
        ],
        "workload": {
            "path": _relative(V9_WORKLOAD),
            "raw_sha256": FORMAL_WORKLOAD_SHA256,
            "canonical_sha256": FORMAL_CANONICAL_SHA256,
            "sources_sha256": FORMAL_SOURCES_SHA256,
            "source_count": 80,
        },
    }


def write_json_atomic_v9(path: Path, value: Any) -> None:
    """Attach selection provenance to coordinator-owned formal metadata."""

    if isinstance(value, Mapping) and value.get("schema") in {
        "paste_repro.live_joint_formal_matrix_plan",
        "paste_repro.live_joint_formal_cell_config",
        "paste_repro.live_joint_formal_matrix_completion",
    }:
        value = dict(value)
        value["formal_generation"] = "v9"
        value["formal_v9_selection"] = _formal_v9_provenance()
    _LEGACY_WRITE_JSON_ATOMIC(path, value)


def _configure_frozen_kernel() -> None:
    """Install prospective v9 constants into the unchanged execution kernel."""

    expected = dict(formal.EXPECTED_CONFIG)
    expected.update(
        {
            "PASTE_LIVE_FORMAL_PROFILE": (
                "live_joint_wikipedia_frozen_formal_v9_context10000_"
                "visitonly_execaware_attemptgate_retry2_jsonrecovery0_"
                "fixedfinal192_stricttail_load80_interval2p5_f0min0"
            ),
            "PASTE_LIVE_FORMAL_WORKLOAD": _relative(V9_WORKLOAD),
            "PASTE_LIVE_FORMAL_WORKLOAD_SHA256": FORMAL_WORKLOAD_SHA256,
            "PASTE_LIVE_FORMAL_CANONICAL_SHA256": FORMAL_CANONICAL_SHA256,
            "PASTE_LIVE_FORMAL_SOURCES_SHA256": FORMAL_SOURCES_SHA256,
            "PASTE_LIVE_VISIT_MIN_START_INTERVAL_S": "2.5",
            "PASTE_LIVE_MIN_SPECULATIVE_TOOL_WORKERS": "0",
            "PASTE_LIVE_FORMAL_COMPLETED_SCREEN": _relative(COMPLETED_SCREEN),
            "PASTE_LIVE_FORMAL_COMPLETED_SCREEN_SHA256": (
                COMPLETED_SCREEN_SHA256
            ),
            "PASTE_LIVE_FORMAL_STRICT_DEVELOPMENT_SELECTION": _relative(
                STRICT_DEVELOPMENT_SELECTION
            ),
            "PASTE_LIVE_FORMAL_STRICT_DEVELOPMENT_SELECTION_SHA256": (
                STRICT_DEVELOPMENT_SELECTION_SHA256
            ),
            "PASTE_LIVE_FORMAL_SELECTED_TRANSPORT": _relative(
                SELECTED_TRANSPORT
            ),
            "PASTE_LIVE_FORMAL_SELECTED_TRANSPORT_SHA256": (
                SELECTED_TRANSPORT_SHA256
            ),
            "PASTE_LIVE_FORMAL_SELECTED_POLICY": SELECTED_POLICY,
            "PASTE_LIVE_FORMAL_MAX_OBSERVED_HTTP_RETRIES": "0",
        }
    )
    formal.DEFAULT_CONFIG = DEFAULT_CONFIG
    formal.PROTOCOL = V9_PROTOCOL
    formal.FORMAL_WORKLOAD_SHA256 = FORMAL_WORKLOAD_SHA256
    formal.FORMAL_CANONICAL_SHA256 = FORMAL_CANONICAL_SHA256
    formal.FORMAL_SOURCES_SHA256 = FORMAL_SOURCES_SHA256
    formal.EXPECTED_CONFIG = expected
    formal.FROZEN_JOINT_SCHEDULER_ENV_KEYS = frozenset(
        key for key in expected if key.startswith("VLLM_SCHED_")
    )
    formal.validate_cell_result = validate_v9_cell_result
    formal.write_json_atomic = write_json_atomic_v9
    formal.BOUND_CODE_PATHS = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "reproduction/scripts/run_live_joint_formal_matrix.py",
        formal.RUNNER,
        formal.START_SERVER,
        formal.STOP_SERVER,
        formal.FORMAL_VALIDATOR,
        V9_PROTOCOL,
        formal.LIVE_AGENT,
        REPOSITORY_ROOT / "reproduction/paste_repro/live_broker.py",
        REPOSITORY_ROOT / "reproduction/paste_repro/live_executor.py",
        REPOSITORY_ROOT / "scripts/pythonhooks/sched_policy_patch.py",
        COMPLETED_SCREEN,
        STRICT_DEVELOPMENT_SELECTION,
        SELECTED_TRANSPORT,
    )


def _validate_v9_cli(arguments: Sequence[str]) -> None:
    """Forbid ad-hoc formal config or order changes at invocation time."""

    parsed = formal.parse_args(arguments)
    if parsed.config.resolve() != DEFAULT_CONFIG.resolve():
        raise FormalV9RunError("formal-v9 requires its registered config path")
    registered_orders = formal.EXPECTED_CONFIG[
        "PASTE_LIVE_FORMAL_DEFAULT_ORDERS"
    ]
    if parsed.orders is not None and parsed.orders != registered_orders:
        raise FormalV9RunError("formal-v9 cell orders are preregistered and immutable")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    runtime = validate_frozen_runtime()
    selection = validate_development_selection()
    _configure_frozen_kernel()
    _validate_v9_cli(arguments)

    if "--check-only" not in arguments:
        return formal.main(arguments)

    output = io.StringIO()
    with redirect_stdout(output):
        return_code = formal.main(arguments)
    try:
        result = json.loads(output.getvalue())
    except json.JSONDecodeError as exc:
        raise FormalV9RunError("frozen kernel emitted invalid check-only JSON") from exc
    result.update(
        {
            "schema": "paste_repro.live_joint_formal_v9_check",
            "version": 1,
            "formal_generation": "v9",
            "development_selection": selection,
            "frozen_runtime_sha256": runtime,
            "formal_workload": {
                "path": _relative(V9_WORKLOAD),
                "raw_sha256": FORMAL_WORKLOAD_SHA256,
                "canonical_sha256": FORMAL_CANONICAL_SHA256,
                "sources_sha256": FORMAL_SOURCES_SHA256,
                "source_count": 80,
                "untouched_by_development_screen": True,
            },
            "registered_treatment": {
                "cells": {
                    "A": {"scheduler": "fcfs", "speculation": "off"},
                    "B": {
                        "scheduler": "fcfs",
                        "speculation": "visit",
                    },
                    "E": {
                        "scheduler": "online_joint_pacer_v2",
                        "speculation": "off",
                    },
                    "F": {
                        "scheduler": "online_joint_pacer_v2",
                        "speculation": "visit",
                    },
                },
                "visit_min_start_interval_s": SELECTED_VISIT_INTERVAL_S,
                "min_speculative_tool_workers": (
                    SELECTED_MIN_SPECULATIVE_TOOL_WORKERS
                ),
                "maximum_observed_http_retries_per_cell": 0,
                "zero_wasted_speculative_service_required": True,
                "offered_concurrency": 80,
                "former_threshold": 64,
                "native_max_num_seqs": 96,
                "fixed_final_completion_tokens": 192,
            },
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except formal.FormalRunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
