#!/usr/bin/env python3
"""Build a contamination-aware 40/30/30 split and calibration-only mapper.

The legacy checksummed mapper artifact is used only as the authoritative 70/30
session registry.  Its learned mapper state is deliberately discarded.  All
legacy held-out sessions become the tuning set.  A salted, checksum-bound,
language-stratified hash selects the final set from the legacy training pool;
the remainder is calibration data.  A fresh mapper is then fit exclusively on
the calibration sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPOSITORY_ROOT / "reproduction"
if str(REPRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_ROOT))

from paste_repro.mapper import (  # noqa: E402
    URLRankMapper,
    load_artifact,
    save_artifact,
    write_json_atomic,
)
from paste_repro.traces import (  # noqa: E402
    count_tool_calls,
    load_trace,
    transitions_from_sessions,
)


SPLIT_SCHEMA = "paste_repro.fixed_three_way_split"
SPLIT_VERSION = 1
BUNDLE_SCHEMA = "paste_repro.fixed_three_way_bundle"
BUNDLE_VERSION = 1
DEFAULT_SALT = "paste-repro-fixed-three-way-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_entries(raw: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"source artifact {label} must be a non-empty list")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f"source artifact {label} entry must be an object")
        session_id = item.get("session_id")
        checksum = item.get("sha256")
        if (
            not isinstance(session_id, str)
            or not session_id
            or Path(session_id).name != session_id
        ):
            raise ValueError(f"unsafe session id in source artifact {label}: {session_id!r}")
        if session_id in seen:
            raise ValueError(f"duplicate session id in source artifact {label}: {session_id}")
        if not isinstance(checksum, str) or _SHA256_RE.fullmatch(checksum) is None:
            raise ValueError(f"invalid SHA-256 for source session {session_id}")
        seen.add(session_id)
        entries.append({"session_id": session_id, "sha256": checksum})
    return entries


def _verify_sources(
    trace_directory: Path,
    entries: Iterable[dict[str, str]],
) -> dict[str, Path]:
    verified: dict[str, Path] = {}
    for entry in entries:
        source = trace_directory / entry["session_id"]
        if not source.is_file():
            raise FileNotFoundError(f"trace from source artifact is missing: {source}")
        actual = file_sha256(source)
        if actual != entry["sha256"]:
            raise ValueError(
                f"trace checksum mismatch for {entry['session_id']}: "
                f"{actual} != {entry['sha256']}"
            )
        verified[entry["session_id"]] = source.resolve()
    return verified


def _stratum(session_id: str) -> str:
    """Use only the filename language marker, never trace outcomes, for strata."""

    return "cjk_filename" if _CJK_RE.search(session_id) else "non_cjk_filename"


def _selection_sha256(
    entry: Mapping[str, str],
    *,
    salt: str,
    source_artifact_sha256: str,
) -> str:
    material = "\0".join(
        (
            salt,
            source_artifact_sha256,
            entry["session_id"],
            entry["sha256"],
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _largest_remainder_quotas(
    stratum_sizes: Mapping[str, int],
    target_count: int,
) -> dict[str, int]:
    total = sum(stratum_sizes.values())
    if target_count <= 0:
        raise ValueError("final_count must be positive")
    if target_count >= total:
        raise ValueError("final_count must leave at least one calibration session")

    quotas = {
        name: target_count * size // total
        for name, size in stratum_sizes.items()
    }
    remaining = target_count - sum(quotas.values())
    order = sorted(
        stratum_sizes,
        key=lambda name: (
            -(target_count * stratum_sizes[name] % total),
            name,
        ),
    )
    for name in order[:remaining]:
        quotas[name] += 1
    if sum(quotas.values()) != target_count:
        raise AssertionError("largest-remainder apportionment did not reach target")
    return quotas


def _plain_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    return {"session_id": str(entry["session_id"]), "sha256": str(entry["sha256"])}


def _sorted_plain_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    return sorted((_plain_entry(entry) for entry in entries), key=lambda item: item["session_id"])


def plan_fixed_split(
    source_artifact: Mapping[str, Any],
    *,
    salt: str = DEFAULT_SALT,
    calibration_count: int = 40,
    tuning_count: int = 30,
    final_count: int = 30,
) -> dict[str, Any]:
    """Return a stable split manifest without reading trace outcomes."""

    if not salt:
        raise ValueError("salt must be non-empty")
    source_checksum = source_artifact.get("artifact_sha256")
    if not isinstance(source_checksum, str) or _SHA256_RE.fullmatch(source_checksum) is None:
        raise ValueError("source artifact is missing a valid artifact_sha256")
    training_split = source_artifact.get("training_split")
    if not isinstance(training_split, Mapping):
        raise ValueError("source artifact is missing training_split")

    legacy_train = _validate_entries(training_split.get("train_sessions"), "train_sessions")
    legacy_held_out = _validate_entries(
        training_split.get("held_out_sessions"), "held_out_sessions"
    )
    train_ids = {entry["session_id"] for entry in legacy_train}
    held_out_ids = {entry["session_id"] for entry in legacy_held_out}
    overlap = sorted(train_ids & held_out_ids)
    if overlap:
        raise ValueError(f"legacy train and held-out sessions overlap: {overlap}")
    if len(legacy_held_out) != tuning_count:
        raise ValueError(
            "legacy held-out pool must be used whole as tuning: "
            f"expected {tuning_count}, found {len(legacy_held_out)}"
        )
    if len(legacy_train) != calibration_count + final_count:
        raise ValueError(
            "legacy train pool does not match requested calibration/final counts: "
            f"{len(legacy_train)} != {calibration_count} + {final_count}"
        )

    grouped: dict[str, list[dict[str, str]]] = {}
    for entry in legacy_train:
        grouped.setdefault(_stratum(entry["session_id"]), []).append(entry)
    quotas = _largest_remainder_quotas(
        {name: len(entries) for name, entries in grouped.items()},
        final_count,
    )

    final_ids: set[str] = set()
    stratum_summary: dict[str, dict[str, int]] = {}
    selection_hashes: dict[str, str] = {}
    for name, entries in grouped.items():
        ordered = sorted(
            entries,
            key=lambda entry: (
                _selection_sha256(
                    entry,
                    salt=salt,
                    source_artifact_sha256=source_checksum,
                ),
                entry["session_id"],
            ),
        )
        quota = quotas[name]
        chosen = ordered[:quota]
        final_ids.update(entry["session_id"] for entry in chosen)
        selection_hashes.update(
            {
                entry["session_id"]: _selection_sha256(
                    entry,
                    salt=salt,
                    source_artifact_sha256=source_checksum,
                )
                for entry in entries
            }
        )
        stratum_summary[name] = {
            "legacy_train": len(entries),
            "calibration": len(entries) - quota,
            "final": quota,
        }

    calibration = [entry for entry in legacy_train if entry["session_id"] not in final_ids]
    final = [entry for entry in legacy_train if entry["session_id"] in final_ids]
    if len(calibration) != calibration_count or len(final) != final_count:
        raise AssertionError("fixed split counts do not match requested counts")

    def selected_entry(entry: Mapping[str, str], origin: str) -> dict[str, str]:
        result = {
            "session_id": entry["session_id"],
            "sha256": entry["sha256"],
            "origin": origin,
            "stratum": _stratum(entry["session_id"]),
        }
        if origin == "legacy_train":
            result["selection_sha256"] = selection_hashes[entry["session_id"]]
        return result

    manifest: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "version": SPLIT_VERSION,
        "source_mapper_artifact_sha256": source_checksum,
        "source_training_manifest_sha256": training_split.get("manifest_sha256"),
        "selection": {
            "algorithm": (
                "language filename strata; largest-remainder quotas; within each "
                "stratum sort sha256(salt + NUL + source artifact sha256 + NUL + "
                "session_id + NUL + trace sha256)"
            ),
            "salt": salt,
            "outcome_fields_used": [],
            "legacy_held_out_policy": "all sessions assigned to tuning",
            "strata": dict(sorted(stratum_summary.items())),
        },
        "counts": {
            "total": calibration_count + tuning_count + final_count,
            "calibration": calibration_count,
            "tuning": tuning_count,
            "final": final_count,
        },
        "calibration_sessions": sorted(
            (selected_entry(entry, "legacy_train") for entry in calibration),
            key=lambda item: item["session_id"],
        ),
        "tuning_sessions": sorted(
            (selected_entry(entry, "legacy_held_out") for entry in legacy_held_out),
            key=lambda item: item["session_id"],
        ),
        "final_sessions": sorted(
            (selected_entry(entry, "legacy_train") for entry in final),
            key=lambda item: item["session_id"],
        ),
        "directories": {
            "calibration": "calibration",
            "tuning": "tuning",
            "final": "final",
        },
        "contamination_guards": {
            "roles_are_whole-session_and_disjoint": True,
            "legacy_mapper_weights_are_discarded": True,
            "mapper_training_role": "calibration",
            "online_predictor_training_role": "calibration workload only",
            "tuning_role_was_previously_live_evaluated": True,
            "final_role_must_not_be_used_until_configuration_freeze": True,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _ensure_symlink_directory(
    directory: Path,
    entries: Sequence[Mapping[str, Any]],
    verified_sources: Mapping[str, Path],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected = {str(entry["session_id"]) for entry in entries}
    for entry in entries:
        name = str(entry["session_id"])
        source = verified_sources[name]
        destination = directory / name
        if destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"existing symlink points elsewhere: {destination}")
            continue
        if destination.exists():
            raise RuntimeError(f"refusing to replace existing path: {destination}")
        destination.symlink_to(source)
    unexpected = sorted(path.name for path in directory.iterdir() if path.name not in expected)
    if unexpected:
        raise RuntimeError(
            f"split directory contains unexpected entries: {directory}: {unexpected}"
        )


def _role_sessions(
    entries: Sequence[Mapping[str, Any]],
    verified_sources: Mapping[str, Path],
) -> tuple[Any, ...]:
    return tuple(load_trace(verified_sources[str(entry["session_id"])]) for entry in entries)


def build_fixed_bundle(
    *,
    legacy_artifact_path: Path,
    trace_directory: Path,
    output_root: Path,
    salt: str = DEFAULT_SALT,
    calibration_count: int = 40,
    tuning_count: int = 30,
    final_count: int = 30,
    result_out: Path | None = None,
) -> dict[str, Any]:
    """Verify, materialize, and retrain the fixed split bundle."""

    # load_artifact validates both artifact and embedded split checksums.
    _, source_artifact = load_artifact(legacy_artifact_path)
    manifest = plan_fixed_split(
        source_artifact,
        salt=salt,
        calibration_count=calibration_count,
        tuning_count=tuning_count,
        final_count=final_count,
    )
    all_entries = [
        *manifest["calibration_sessions"],
        *manifest["tuning_sessions"],
        *manifest["final_sessions"],
    ]
    verified_sources = _verify_sources(trace_directory.resolve(), all_entries)
    if len(verified_sources) != manifest["counts"]["total"]:
        raise AssertionError("verified source count does not cover the fixed split")

    split_name = (
        f"{source_artifact['artifact_sha256'][:12]}-"
        f"{manifest['manifest_sha256'][:12]}-"
        f"c{calibration_count}-t{tuning_count}-f{final_count}"
    )
    split_root = output_root.resolve() / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    for role in ("calibration", "tuning", "final"):
        _ensure_symlink_directory(
            split_root / manifest["directories"][role],
            manifest[f"{role}_sessions"],
            verified_sources,
        )

    split_manifest_path = split_root / "split_manifest.json"
    write_json_atomic(split_manifest_path, manifest)

    calibration_sessions = _role_sessions(
        manifest["calibration_sessions"], verified_sources
    )
    calibration_transitions = transitions_from_sessions(calibration_sessions)
    mapper = URLRankMapper().fit(
        calibration_transitions,
        searches_seen=count_tool_calls(calibration_sessions, "search"),
    )
    calibration_entries = _sorted_plain_entries(manifest["calibration_sessions"])
    tuning_entries = _sorted_plain_entries(manifest["tuning_sessions"])
    final_entries = _sorted_plain_entries(manifest["final_sessions"])
    mapper_training_split = {
        "algorithm": "fixed three-way whole-session split; mapper fit on calibration only",
        "seed": salt,
        "train_ratio": calibration_count / manifest["counts"]["total"],
        "source_artifact_sha256": source_artifact["artifact_sha256"],
        "fixed_split_manifest_sha256": manifest["manifest_sha256"],
        # Keep conventional names for compatibility with existing artifact readers.
        "train_sessions": calibration_entries,
        "held_out_sessions": sorted(
            [*tuning_entries, *final_entries], key=lambda item: item["session_id"]
        ),
        "calibration_sessions": calibration_entries,
        "tuning_sessions": tuning_entries,
        "final_sessions": final_entries,
    }
    mapper_artifact = mapper.to_artifact(mapper_training_split)
    mapper_artifact_path = split_root / "url_rank_mapper_calibration_only.json"
    save_artifact(mapper_artifact_path, mapper_artifact)
    # Read back from disk so a partial/corrupt write cannot enter the bundle.
    _, persisted_mapper_artifact = load_artifact(mapper_artifact_path)

    roles: dict[str, Any] = {}
    for role in ("calibration", "tuning", "final"):
        entries = _sorted_plain_entries(manifest[f"{role}_sessions"])
        roles[role] = {
            "directory": manifest["directories"][role],
            "session_count": len(entries),
            "sessions_sha256": canonical_sha256(entries),
        }
    bundle: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "version": BUNDLE_VERSION,
        "split_manifest": split_manifest_path.name,
        "split_manifest_sha256": manifest["manifest_sha256"],
        "source_mapper_artifact_sha256": source_artifact["artifact_sha256"],
        "mapper_artifact": mapper_artifact_path.name,
        "mapper_artifact_sha256": persisted_mapper_artifact["artifact_sha256"],
        "mapper_training_role": "calibration",
        "online_predictor_training_role": "calibration workload only",
        "roles": roles,
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    bundle_path = split_root / "bundle.json"
    write_json_atomic(bundle_path, bundle)

    result = {
        **bundle,
        "split_root": str(split_root),
        "bundle_path": str(bundle_path),
        "split_manifest_path": str(split_manifest_path),
        "mapper_artifact_path": str(mapper_artifact_path),
        "roles": {
            role: {
                **payload,
                "absolute_directory": str(split_root / payload["directory"]),
            }
            for role, payload in roles.items()
        },
    }
    if result_out is not None:
        write_json_atomic(result_out, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a fixed 40 calibration / 30 tuning / 30 final split from a "
            "checksummed legacy 70/30 mapper artifact, then retrain from calibration only."
        )
    )
    parser.add_argument(
        "--legacy-artifact",
        type=Path,
        default=REPRODUCTION_ROOT / "results" / "tool_only" / "url_rank_mapper.json",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=REPOSITORY_ROOT / "traces" / "my_traces",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPRODUCTION_ROOT / "artifacts" / "fixed_trace_splits",
    )
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--calibration-count", type=int, default=40)
    parser.add_argument("--tuning-count", type=int, default=30)
    parser.add_argument("--final-count", type=int, default=30)
    parser.add_argument(
        "--result-out",
        type=Path,
        help="optional machine-local JSON containing resolved bundle paths",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_fixed_bundle(
        legacy_artifact_path=args.legacy_artifact,
        trace_directory=args.trace_dir,
        output_root=args.output_root,
        salt=args.salt,
        calibration_count=args.calibration_count,
        tuning_count=args.tuning_count,
        final_count=args.final_count,
        result_out=args.result_out,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
