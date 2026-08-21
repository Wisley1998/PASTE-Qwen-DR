#!/usr/bin/env python3
"""Strictly compare one reused FCFS A cell with one Joint D screening cell.

Inputs are validated against the fixed-workload manifest and are never
modified.  Scheduler-environment differences must equal an explicit key
allowlist, with exact expected values (or explicit absence) on both sides.
The derived JSON is written atomically as a direct child of the D run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (REPRODUCTION_ROOT, SCRIPT_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from paste_repro.mapper import write_json_atomic  # noqa: E402
from summarize_candidate_d import (  # noqa: E402
    _cell_metrics,
    _comparison,
    _saving_decomposition,
    _source_pairing,
)
from summarize_four_cell import load_fixed_manifest, load_run  # noqa: E402
from summarize_natural_queue_probe import summarize_probe  # noqa: E402
from summarize_paired_ad import (  # noqa: E402
    _load_raw_execution_accounting,
    _task_flow_by_trace,
    _validate_source_multiplicity,
)


SCHEMA = "paste_repro.strict_screening_ad"
VERSION = 1
DEFAULT_ENGINE_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MODEL_ID",
    "MODEL_REVISION",
    "VLLM_HOST",
    "VLLM_PROBE_HOST",
    "VLLM_PORT",
    "VLLM_TP_SIZE",
    "VLLM_DTYPE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_CUDA_GRAPH_SIZES",
    "VLLM_USE_V1",
)
_MISSING = object()


def _parse_key_value(items: Sequence[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"{option} requires KEY=VALUE, got {item!r}")
        if key in result:
            raise ValueError(f"{option} repeats key {key}")
        result[key] = value
    return result


def _exact_config_guard(
    a_config: Mapping[str, Any],
    d_config: Mapping[str, Any],
    *,
    allowed_differences: set[str],
    expected_a: Mapping[str, str],
    expected_d: Mapping[str, str],
    expected_a_missing: set[str],
    expected_d_missing: set[str],
) -> dict[str, Any]:
    if set(expected_a) & expected_a_missing or set(expected_d) & expected_d_missing:
        raise ValueError("a config key cannot be both expected present and missing")
    if (set(expected_a) | expected_a_missing) != allowed_differences:
        raise ValueError("every allowed config difference needs an exact A expectation")
    if (set(expected_d) | expected_d_missing) != allowed_differences:
        raise ValueError("every allowed config difference needs an exact D expectation")

    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(a_config) | set(d_config)):
        a_value = a_config.get(key, _MISSING)
        d_value = d_config.get(key, _MISSING)
        if a_value == d_value:
            continue
        differences[key] = {
            "a_present": a_value is not _MISSING,
            "a_value": None if a_value is _MISSING else a_value,
            "d_present": d_value is not _MISSING,
            "d_value": None if d_value is _MISSING else d_value,
        }
    actual = set(differences)
    if actual != allowed_differences:
        raise ValueError(
            "A/D scheduler configuration diff does not exactly match allowlist; "
            f"unexpected={sorted(actual - allowed_differences)}, "
            f"unused={sorted(allowed_differences - actual)}"
        )

    def validate_side(
        config: Mapping[str, Any],
        expected: Mapping[str, str],
        expected_missing: set[str],
        label: str,
    ) -> None:
        for key, value in expected.items():
            if config.get(key, _MISSING) != value:
                raise ValueError(
                    f"{label} scheduler configuration {key}="
                    f"{config.get(key, _MISSING)!r}; expected {value!r}"
                )
        for key in expected_missing:
            if key in config:
                raise ValueError(f"{label} scheduler configuration unexpectedly has {key}")

    validate_side(a_config, expected_a, expected_a_missing, "A")
    validate_side(d_config, expected_d, expected_d_missing, "D")
    return {
        "exact_allowlist_match": True,
        "allowed_difference_keys": sorted(allowed_differences),
        "actual_difference_keys": sorted(actual),
        "differences": differences,
        "all_nonwhitelisted_keys_identical": True,
        "expected_a_values": dict(sorted(expected_a.items())),
        "expected_d_values": dict(sorted(expected_d.items())),
        "expected_a_missing": sorted(expected_a_missing),
        "expected_d_missing": sorted(expected_d_missing),
    }


def _engine_shape_guard(
    a_config: Mapping[str, Any],
    d_config: Mapping[str, Any],
    *,
    required_keys: Sequence[str],
    allowed_differences: set[str],
) -> dict[str, Any]:
    if not required_keys or len(set(required_keys)) != len(required_keys):
        raise ValueError("required engine keys must be non-empty and unique")
    overlap = set(required_keys) & allowed_differences
    if overlap:
        raise ValueError(f"engine keys cannot be allowed to differ: {sorted(overlap)}")
    shape: dict[str, Any] = {}
    for key in required_keys:
        if key not in a_config or key not in d_config:
            raise ValueError(f"engine-shape key is missing: {key}")
        if a_config[key] != d_config[key]:
            raise ValueError(f"engine-shape mismatch: {key}")
        shape[key] = a_config[key]
    return {
        "required_keys": list(required_keys),
        "all_required_keys_present_and_identical": True,
        "values": shape,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_config(run_path: Path, recorded_sha256: Any) -> dict[str, Any]:
    if not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        raise ValueError(f"invalid recorded frozen-config SHA: {run_path}")
    config_path = run_path.parent / "frozen_config.env"
    sidecar_path = run_path.parent / "frozen_config.sha256"
    if not config_path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"frozen config evidence is missing: {run_path.parent}")
    actual = _sha256_file(config_path)
    sidecar_fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar_fields) != 2 or sidecar_fields[0] != actual:
        raise ValueError(f"frozen config checksum sidecar mismatch: {run_path.parent}")
    if recorded_sha256 != actual:
        raise ValueError(f"summary/frozen config checksum mismatch: {run_path}")
    return {
        "path": config_path.resolve().as_posix(),
        "sha256": actual,
        "summary_matches_file": True,
        "sidecar_matches_file": True,
    }


def _execution_comparison(
    a_metrics: Mapping[str, Any], d_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    a_execution = a_metrics["execution_accounting"]
    d_execution = d_metrics["execution_accounting"]
    a_tokens = a_execution["completion_tokens"]["total"]
    d_tokens = d_execution["completion_tokens"]["total"]
    token_effect = None
    if a_tokens is not None and d_tokens is not None:
        token_effect = {
            "a_total": a_tokens,
            "d_total": d_tokens,
            "d_minus_a": d_tokens - a_tokens,
            "d_relative_to_a": (d_tokens - a_tokens) / a_tokens if a_tokens else None,
        }
    a_preempt = a_execution["preemption"]["num_preemptions_total"]
    d_preempt = d_execution["preemption"]["num_preemptions_total"]
    preemption_effect = {
        "a_total": a_preempt,
        "d_total": d_preempt,
        "a_minus_d": (
            a_preempt - d_preempt
            if a_preempt is not None and d_preempt is not None
            else None
        ),
    }
    return {
        "completion_tokens": token_effect,
        "retry_accounting": {
            "A": a_metrics["retry_accounting"],
            "D": d_metrics["retry_accounting"],
        },
        "preemption": preemption_effect,
        "swap": {
            "A": a_execution["swap"],
            "D": d_execution["swap"],
        },
    }


def summarize_strict_screening(
    *,
    manifest_path: Path,
    role: str,
    a_run: Path,
    d_run: Path,
    allowed_config_differences: set[str],
    expected_a_config: Mapping[str, str],
    expected_d_config: Mapping[str, str],
    expected_a_config_missing: set[str],
    expected_d_config_missing: set[str],
    expected_a_policy: str,
    expected_d_policy: str,
    expected_a_overlap: str,
    expected_d_overlap: str,
    required_engine_keys: Sequence[str] = DEFAULT_ENGINE_KEYS,
    include_natural_queue_evidence: bool = True,
    require_natural_queue: bool = True,
    verify_frozen_configs: bool = True,
) -> dict[str, Any]:
    a_path = a_run.resolve()
    d_path = d_run.resolve()
    if a_path == d_path:
        raise ValueError("A and D run directories must be distinct")
    manifest = load_fixed_manifest(manifest_path, role)
    a = load_run(a_path, "A", manifest["bindings"]["A"])
    d = load_run(d_path, "D", manifest["bindings"]["D"])

    if a["identity_rows"] != d["identity_rows"]:
        raise ValueError("A/D request identity, prompts, or messages mismatch")
    if a["source_mapping"] != d["source_mapping"]:
        raise ValueError("A/D source-session mapping mismatch")
    source_counts = Counter(a["source_mapping"].values())
    _validate_source_multiplicity(
        source_counts, workload_invariants=manifest, replicate=1
    )
    for field in (
        "speedup",
        "max_active_traces",
        "tool_wait_mode",
        "configured_max_request_attempts",
    ):
        if a["public"][field] != d["public"][field]:
            raise ValueError(f"A/D replay configuration mismatch: {field}")
    expected_modes = {
        "A": (expected_a_policy, expected_a_overlap),
        "D": (expected_d_policy, expected_d_overlap),
    }
    for label, run in (("A", a), ("D", d)):
        policy, overlap = expected_modes[label]
        if run["public"]["policy"] != policy:
            raise ValueError(f"{label} policy differs from explicit expectation")
        if run["public"]["tool_overlap_mode"] != overlap:
            raise ValueError(f"{label} overlap mode differs from explicit expectation")

    a_config = a["public"]["scheduler_configuration"]
    d_config = d["public"]["scheduler_configuration"]
    config_guard = _exact_config_guard(
        a_config,
        d_config,
        allowed_differences=allowed_config_differences,
        expected_a=expected_a_config,
        expected_d=expected_d_config,
        expected_a_missing=expected_a_config_missing,
        expected_d_missing=expected_d_config_missing,
    )
    engine_guard = _engine_shape_guard(
        a_config,
        d_config,
        required_keys=required_engine_keys,
        allowed_differences=allowed_config_differences,
    )

    frozen_evidence: dict[str, Any] | None = None
    if verify_frozen_configs:
        frozen_evidence = {
            "A": _verify_frozen_config(
                a_path, a_config.get("PASTE_FROZEN_CONFIG_SHA256")
            ),
            "D": _verify_frozen_config(
                d_path, d_config.get("PASTE_FROZEN_CONFIG_SHA256")
            ),
        }

    a_flows = _task_flow_by_trace(a_path, a)
    d_flows = _task_flow_by_trace(d_path, d)
    if set(a_flows) != set(d_flows):
        raise ValueError("A/D task identities do not exactly match")
    for trace_id in a_flows:
        if (
            a_flows[trace_id]["source_session"]
            != d_flows[trace_id]["source_session"]
            or a_flows[trace_id]["initial_delay_s"]
            != d_flows[trace_id]["initial_delay_s"]
        ):
            raise ValueError(f"A/D task pairing mismatch: {trace_id}")

    cells: dict[str, Any] = {}
    for label, path, run, flows in (
        ("A", a_path, a, a_flows),
        ("D", d_path, d, d_flows),
    ):
        metrics = _cell_metrics(path, run, flows)
        metrics["execution_accounting"] = _load_raw_execution_accounting(
            path, run["public"]
        )
        cells[label] = metrics

    if include_natural_queue_evidence:
        queue_cells = {"A": summarize_probe(a_path), "D": summarize_probe(d_path)}
        all_proven = all(
            evidence["sequence_capacity"]["natural_vllm_queue_proven"]
            for evidence in queue_cells.values()
        )
        if require_natural_queue and not all_proven:
            failed = [
                label
                for label, evidence in queue_cells.items()
                if not evidence["sequence_capacity"]["natural_vllm_queue_proven"]
            ]
            raise ValueError(f"natural vLLM queue requirement failed for {failed}")
        natural_queue: dict[str, Any] = {
            "all_cells_proven": all_proven,
            "cells": queue_cells,
        }
    else:
        if require_natural_queue:
            raise ValueError("cannot require natural queue when evidence is disabled")
        natural_queue = {"available": False, "reason": "disabled by caller"}

    source_pairing = _source_pairing(a_flows, d_flows, a["source_mapping"])
    if source_pairing["independent_source_session_count"] != manifest[
        "independent_source_session_count"
    ]:
        raise AssertionError("source-folded sample count differs from manifest")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "screening_reuses_previous_a_not_fresh_server_pair",
        "comparison_invariants": {
            "fixed_role": role,
            "fixed_workload_manifest": manifest["path"].as_posix(),
            "fixed_workload_manifest_sha256": manifest["manifest_sha256"],
            "load_instance_count": manifest["load_instance_count"],
            "independent_source_session_count": manifest[
                "independent_source_session_count"
            ],
            "instances_per_source": manifest["instances_per_source"],
            "duplicates_are_not_independent": manifest[
                "duplicates_are_not_independent"
            ],
            "request_identity_exact_match": True,
            "request_identity_sha256": a["public"]["request_identity_sha256"],
            "source_mapping_exact_match": True,
            "source_sessions_sha256": a["public"]["source_sessions_sha256"],
            "engine_shape_guard": engine_guard,
            "scheduler_configuration_guard": config_guard,
            "mode_expectations": {
                "A": {"policy": expected_a_policy, "tool_overlap": expected_a_overlap},
                "D": {"policy": expected_d_policy, "tool_overlap": expected_d_overlap},
            },
            "mode_specific_hashes_validated_by_manifest": {
                "A": {
                    "prepared_workload_sha256": a["public"][
                        "prepared_workload_sha256"
                    ],
                    "scheduler_calibration_workload_sha256": a["public"][
                        "scheduler_calibration_workload_sha256"
                    ],
                    "mapper_artifact_sha256": a["public"][
                        "mapper_artifact_sha256"
                    ],
                },
                "D": {
                    "prepared_workload_sha256": d["public"][
                        "prepared_workload_sha256"
                    ],
                    "scheduler_calibration_workload_sha256": d["public"][
                        "scheduler_calibration_workload_sha256"
                    ],
                    "mapper_artifact_sha256": d["public"][
                        "mapper_artifact_sha256"
                    ],
                },
            },
            "frozen_config_evidence": frozen_evidence,
        },
        "cells": cells,
        "comparison": {
            **_comparison(cells["A"], cells["D"], baseline_label="A", candidate_label="D"),
            "execution": _execution_comparison(cells["A"], cells["D"]),
        },
        "source_pairing": source_pairing,
        "task_saving_decomposition": _saving_decomposition(cells["A"], cells["D"]),
        "natural_queue_evidence": natural_queue,
        "interpretation": (
            "A and D have identical deterministic request identities and engine shape, "
            "but A is reused from an earlier fresh server. This is strict screening "
            "evidence, not a fresh-server paired replicate. Bootstrap inference uses "
            "60 independent source-session means after folding four deterministic "
            "load copies per source."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("final", "heldout", "stress"), default="stress")
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--d-run", type=Path, required=True)
    parser.add_argument("--expect-a-policy", required=True)
    parser.add_argument("--expect-d-policy", required=True)
    parser.add_argument("--expect-a-overlap", required=True)
    parser.add_argument("--expect-d-overlap", required=True)
    parser.add_argument("--allow-config-diff", action="append", default=[], metavar="KEY")
    parser.add_argument("--expect-a-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--expect-d-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--expect-a-config-missing", action="append", default=[], metavar="KEY")
    parser.add_argument("--expect-d-config-missing", action="append", default=[], metavar="KEY")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    allowed = set(args.allow_config_diff)
    if len(allowed) != len(args.allow_config_diff):
        raise ValueError("--allow-config-diff contains duplicate keys")
    a_missing = set(args.expect_a_config_missing)
    d_missing = set(args.expect_d_config_missing)
    if len(a_missing) != len(args.expect_a_config_missing):
        raise ValueError("--expect-a-config-missing contains duplicate keys")
    if len(d_missing) != len(args.expect_d_config_missing):
        raise ValueError("--expect-d-config-missing contains duplicate keys")
    d_path = args.d_run.resolve()
    output = args.output.resolve()
    if output.parent != d_path or output.suffix != ".json":
        raise ValueError("--output must be a JSON file directly under the D run root")
    result = summarize_strict_screening(
        manifest_path=args.manifest,
        role=args.role,
        a_run=args.a_run,
        d_run=args.d_run,
        allowed_config_differences=allowed,
        expected_a_config=_parse_key_value(args.expect_a_config, "--expect-a-config"),
        expected_d_config=_parse_key_value(args.expect_d_config, "--expect-d-config"),
        expected_a_config_missing=a_missing,
        expected_d_config_missing=d_missing,
        expected_a_policy=args.expect_a_policy,
        expected_d_policy=args.expect_d_policy,
        expected_a_overlap=args.expect_a_overlap,
        expected_d_overlap=args.expect_d_overlap,
        include_natural_queue_evidence=True,
        require_natural_queue=True,
        verify_frozen_configs=True,
    )
    write_json_atomic(output, result)
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
