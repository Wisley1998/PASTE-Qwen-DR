#!/usr/bin/env python3
"""Offline validator for untouched frozen live-joint formal workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata
from urllib.parse import quote, unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v2.json"
)
FORMAL_V3_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v3.json"
)
FORMAL_V4_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v4.json"
)
FORMAL_V5_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v5.json"
)
FORMAL_V6_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v6.json"
)
FORMAL_V7_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v7.json"
)
FORMAL_V8_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v8.json"
)
FORMAL_V9_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_formal_v9.json"
)
FROZEN_TUNE_WORKLOAD = (
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_tune_v1.json"
)
DEFAULT_DEVELOPMENT = (
    REPOSITORY_ROOT / "reproduction/workloads/live_joint_wikipedia_v1.json",
    REPOSITORY_ROOT / "reproduction/workloads/live_joint_wikipedia_tune_v1.json",
    REPOSITORY_ROOT
    / "reproduction/workloads/live_joint_wikipedia_frozen_dev_v1.json",
)
EXPECTED_DEVELOPMENT_NAMES = {path.name for path in DEFAULT_DEVELOPMENT}
FORMAL_V3_EXCLUSIONS = (*DEFAULT_DEVELOPMENT, DEFAULT_WORKLOAD)
FORMAL_V4_EXCLUSIONS = (
    *DEFAULT_DEVELOPMENT,
    FROZEN_TUNE_WORKLOAD,
    DEFAULT_WORKLOAD,
    FORMAL_V3_WORKLOAD,
)
FORMAL_V5_EXCLUSIONS = (*FORMAL_V4_EXCLUSIONS, FORMAL_V4_WORKLOAD)
FORMAL_V6_EXCLUSIONS = (*FORMAL_V5_EXCLUSIONS, FORMAL_V5_WORKLOAD)
FORMAL_V7_EXCLUSIONS = (*FORMAL_V6_EXCLUSIONS, FORMAL_V6_WORKLOAD)
FORMAL_V8_EXCLUSIONS = (*FORMAL_V7_EXCLUSIONS, FORMAL_V7_WORKLOAD)
FORMAL_V9_EXCLUSIONS = (*FORMAL_V8_EXCLUSIONS, FORMAL_V8_WORKLOAD)
EXPECTED_SOURCE_COUNT = 60
FORMAL_V8_EXPECTED_SOURCE_COUNT = 80
FORMAL_V9_EXPECTED_SOURCE_COUNT = 80
EXPECTED_CREATED_DATE = "2026-08-16"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")

FORMAL_V8_EXCLUSION_SHA256 = {
    "live_joint_wikipedia_v1.json": (
        "d3829e9162d5c46c9c12e4d6162c2d5d114b903d9fe22d3e91bf86a631d03f52"
    ),
    "live_joint_wikipedia_tune_v1.json": (
        "38bb9e2d53a4557fd60ec3ac7a447a4fcbc535a0bd3a38edb008b9e9d691a26e"
    ),
    "live_joint_wikipedia_frozen_dev_v1.json": (
        "ae2df033e872631f517b2cd36664282dca0f0c821bbfa48d82d4a3b8ad3f2e8e"
    ),
    "live_joint_wikipedia_frozen_tune_v1.json": (
        "e9f63f75bb80c840fbc59f2aa9a581527669c10fc761a4649f50a1bc03eaf1ea"
    ),
    "live_joint_wikipedia_frozen_formal_v2.json": (
        "4c71ce9bf72b3cbec8ddc077f7e58270493f10e63f3a45e107e39faff3b1bb76"
    ),
    "live_joint_wikipedia_frozen_formal_v3.json": (
        "a8f5de832e7e04e3cbd1b7bb71629207201f99285a0d9f95fbc1e7246f0b6366"
    ),
    "live_joint_wikipedia_frozen_formal_v4.json": (
        "e965317225ed0f2d4aec9e8e1a444abd0949521205e705c4daae5e786ce092d5"
    ),
    "live_joint_wikipedia_frozen_formal_v5.json": (
        "6b11193c8a0dbbd70f9ae4bc2c72b56737893b4d45dacd1d9970e01ca019ae31"
    ),
    "live_joint_wikipedia_frozen_formal_v6.json": (
        "44122877db66b1df4a985316c2a96b71d91d13c4e8be84affb73d405490bd43f"
    ),
    "live_joint_wikipedia_frozen_formal_v7.json": (
        "cbf143f59f4d2a05650df68d8fa6f00d7471964a4b257d26dd092ba90c40e6c8"
    ),
}
FORMAL_V9_EXCLUSION_SHA256 = {
    **FORMAL_V8_EXCLUSION_SHA256,
    "live_joint_wikipedia_frozen_formal_v8.json": (
        "780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4"
    ),
}

SPLIT_PROFILES: dict[str, dict[str, Any]] = {
    "live-joint-wikipedia-frozen-formal-v2": {
        "source_id_re": re.compile(r"formal-v2-(\d{3})\Z"),
        "excluded_paths": DEFAULT_DEVELOPMENT,
    },
    "live-joint-wikipedia-frozen-formal-v3": {
        "source_id_re": re.compile(r"formal-v3-(\d{3})\Z"),
        "excluded_paths": FORMAL_V3_EXCLUSIONS,
    },
    "live-joint-wikipedia-frozen-formal-v4": {
        "source_id_re": re.compile(r"formal-v4-(\d{3})\Z"),
        "excluded_paths": FORMAL_V4_EXCLUSIONS,
    },
    "live-joint-wikipedia-frozen-formal-v5": {
        "source_id_re": re.compile(r"formal-v5-(\d{3})\Z"),
        "excluded_paths": FORMAL_V5_EXCLUSIONS,
    },
    "live-joint-wikipedia-frozen-formal-v6": {
        "source_id_re": re.compile(r"formal-v6-(\d{3})\Z"),
        "excluded_paths": FORMAL_V6_EXCLUSIONS,
    },
    "live-joint-wikipedia-frozen-formal-v7": {
        "source_id_re": re.compile(r"formal-v7-(\d{3})\Z"),
        "excluded_paths": FORMAL_V7_EXCLUSIONS,
    },
    "live-joint-wikipedia-frozen-formal-v8": {
        "source_id_re": re.compile(r"formal-v8-(\d{3})\Z"),
        "excluded_paths": FORMAL_V8_EXCLUSIONS,
        "expected_source_count": FORMAL_V8_EXPECTED_SOURCE_COUNT,
        "expected_exclusion_sha256": FORMAL_V8_EXCLUSION_SHA256,
    },
    "live-joint-wikipedia-frozen-formal-v9": {
        "source_id_re": re.compile(r"formal-v9-(\d{3})\Z"),
        "excluded_paths": FORMAL_V9_EXCLUSIONS,
        "expected_source_count": FORMAL_V9_EXPECTED_SOURCE_COUNT,
        "expected_exclusion_sha256": FORMAL_V9_EXCLUSION_SHA256,
    },
}

EXPECTED_EXCLUSION_METADATA = {
    "live_joint_wikipedia_v1.json": (
        "live-joint-wikipedia-development-v1",
        "development",
        False,
    ),
    "live_joint_wikipedia_tune_v1.json": (
        "live-joint-wikipedia-tune-v1",
        "tune",
        False,
    ),
    "live_joint_wikipedia_frozen_dev_v1.json": (
        "live-joint-wikipedia-frozen-development-v1",
        "development",
        False,
    ),
    "live_joint_wikipedia_frozen_formal_v2.json": (
        "live-joint-wikipedia-frozen-formal-v2",
        "formal_heldout",
        True,
    ),
    "live_joint_wikipedia_frozen_formal_v3.json": (
        "live-joint-wikipedia-frozen-formal-v3",
        "formal_heldout",
        True,
    ),
    "live_joint_wikipedia_frozen_formal_v4.json": (
        "live-joint-wikipedia-frozen-formal-v4",
        "formal_heldout",
        True,
    ),
    "live_joint_wikipedia_frozen_formal_v5.json": (
        "live-joint-wikipedia-frozen-formal-v5",
        "formal_heldout",
        True,
    ),
    "live_joint_wikipedia_frozen_formal_v6.json": (
        "live-joint-wikipedia-frozen-formal-v6",
        "formal_heldout",
        True,
    ),
    "live_joint_wikipedia_frozen_formal_v7.json": (
        "live-joint-wikipedia-frozen-formal-v7",
        "formal_heldout",
        True,
    ),
    "live_joint_wikipedia_frozen_formal_v8.json": (
        "live-joint-wikipedia-frozen-formal-v8",
        "formal_heldout",
        True,
    ),
    "live_joint_wikipedia_frozen_tune_v1.json": (
        "live-joint-wikipedia-frozen-tune-v1",
        "tune",
        False,
    ),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    return _mapping(value, label)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_topic(value: str) -> str:
    decoded = unquote(value)
    normalized = unicodedata.normalize("NFKC", decoded).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _topic_tokens(value: str) -> frozenset[str]:
    return frozenset(normalize_topic(value).split())


def _canonical_wikipedia_topic(value: Any, label: str) -> tuple[str, str]:
    url = _nonempty_string(value, label)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "en.wikipedia.org"
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/wiki/")
    ):
        raise ValueError(f"{label} must be canonical HTTPS en.wikipedia.org /wiki URL")
    encoded_title = parsed.path[len("/wiki/") :]
    if not encoded_title or "/" in encoded_title or any(ch.isspace() for ch in encoded_title):
        raise ValueError(f"{label} has an invalid article title")
    decoded_title = unquote(encoded_title)
    if not decoded_title or any(ch.isspace() for ch in decoded_title):
        raise ValueError(f"{label} must use underscores rather than spaces")
    canonical_title = quote(decoded_title, safe="()_,-.")
    canonical_url = f"https://en.wikipedia.org/wiki/{canonical_title}"
    if url != canonical_url:
        raise ValueError(f"{label} is not canonically encoded: expected {canonical_url}")
    return canonical_url, normalize_topic(decoded_title)


def _excluded_semantics(
    paths: Sequence[Path],
    *,
    expected_names: set[str],
    expected_sha256: Mapping[str, str] | None = None,
) -> tuple[set[str], set[str], list[dict[str, str]]]:
    if {path.name for path in paths} != expected_names or len(paths) != len(expected_names):
        raise ValueError("excluded workload set does not match the frozen split binding")
    if expected_sha256 is not None and set(expected_sha256) != expected_names:
        raise ValueError("frozen exclusion SHA binding set does not match exclusions")
    topics: set[str] = set()
    questions: set[str] = set()
    bindings: list[dict[str, str]] = []
    for path in sorted(paths, key=lambda item: item.name):
        file_sha256 = _sha256_bytes(path.read_bytes()) if path.is_file() else ""
        if expected_sha256 is not None and file_sha256 != expected_sha256[path.name]:
            raise ValueError(f"excluded workload {path.name} has the wrong frozen SHA256")
        payload = _read_json(path, f"excluded workload {path.name}")
        expected = EXPECTED_EXCLUSION_METADATA.get(path.name)
        if expected is None:
            raise ValueError(f"excluded workload {path.name} is not a frozen input")
        expected_split_id, expected_role, expected_formal_eligible = expected
        if payload.get("split_id") != expected_split_id:
            raise ValueError(f"excluded workload {path.name} has the wrong split ID")
        if payload.get("split_role") != expected_role:
            raise ValueError(f"excluded workload {path.name} has an invalid split role")
        if (
            _boolean(payload.get("formal_eligible"), f"{path.name}.formal_eligible")
            is not expected_formal_eligible
        ):
            raise ValueError(
                f"excluded workload {path.name} has the wrong formal eligibility"
            )
        for index, raw in enumerate(_sequence(payload.get("sources"), f"{path.name}.sources")):
            row = _mapping(raw, f"{path.name}.sources[{index}]")
            query = _nonempty_string(
                row.get("search_query"), f"{path.name}.sources[{index}].search_query"
            )
            topics.add(normalize_topic(query))
            question = _nonempty_string(
                row.get("question"), f"{path.name}.sources[{index}].question"
            )
            questions.add(normalize_topic(question))
            expected_url = row.get("expected_url")
            if expected_url is not None:
                _, url_topic = _canonical_wikipedia_topic(
                    expected_url, f"{path.name}.sources[{index}].expected_url"
                )
                topics.add(url_topic)
        bindings.append(
            {
                "name": path.name,
                "file_sha256": file_sha256,
            }
        )
    return topics, questions, bindings


def _semantic_overlap(topic: str, development_topics: set[str]) -> str | None:
    tokens = _topic_tokens(topic)
    for development in sorted(development_topics):
        development_tokens = _topic_tokens(development)
        if topic == development or tokens <= development_tokens or development_tokens <= tokens:
            return development
    return None


def validate_formal_workload(
    workload_path: Path = DEFAULT_WORKLOAD,
    *,
    development_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    payload = _read_json(workload_path, "formal workload")
    if payload.get("schema_version") != 2:
        raise ValueError("formal workload schema_version must be 2")
    split_id = payload.get("split_id")
    profile = SPLIT_PROFILES.get(split_id)
    if profile is None:
        raise ValueError("formal workload split_id is not a supported frozen split")
    if payload.get("split_role") != "formal_heldout":
        raise ValueError("formal workload split_role must be formal_heldout")
    if _boolean(payload.get("formal_eligible"), "formal_eligible") is not True:
        raise ValueError("formal workload is not marked formal eligible")
    for key in ("used_for_tuning", "used_for_parameter_selection"):
        if _boolean(payload.get(key), key) is not False:
            raise ValueError(f"formal workload attestation failed: {key}")
    for key in ("untouched_at_freeze", "frozen_call_graph"):
        if _boolean(payload.get(key), key) is not True:
            raise ValueError(f"formal workload attestation failed: {key}")
    created_date = _nonempty_string(payload.get("created_date_utc"), "created_date_utc")
    if not DATE_RE.fullmatch(created_date) or created_date != EXPECTED_CREATED_DATE:
        raise ValueError("formal workload created_date_utc is not the frozen date")
    if payload.get("language") != "en":
        raise ValueError("formal workload language must be en")
    if payload.get("search_backend") != "bing_html_search":
        raise ValueError("formal workload search backend is not frozen Bing HTML")
    if payload.get("visit_backend") != "r.jina.ai":
        raise ValueError("formal workload visit backend is not frozen r.jina.ai")
    expected_source_count = int(profile.get("expected_source_count", EXPECTED_SOURCE_COUNT))
    if payload.get("source_count") != expected_source_count:
        raise ValueError(
            f"formal workload source_count must be {expected_source_count}"
        )
    expected_exclusion_paths = tuple(profile["excluded_paths"])
    expected_exclusion_names = {path.name for path in expected_exclusion_paths}
    excluded_values = [
        _nonempty_string(value, f"excluded_workloads[{index}]")
        for index, value in enumerate(
            _sequence(payload.get("excluded_workloads"), "excluded_workloads")
        )
    ]
    if (
        len(excluded_values) != len(expected_exclusion_names)
        or set(excluded_values) != expected_exclusion_names
    ):
        raise ValueError("formal workload does not bind all frozen exclusions")

    selected_exclusion_paths = (
        expected_exclusion_paths
        if development_paths is None
        else tuple(Path(path) for path in development_paths)
    )
    excluded_topics, excluded_questions, exclusion_bindings = _excluded_semantics(
        selected_exclusion_paths,
        expected_names=expected_exclusion_names,
        expected_sha256=profile.get("expected_exclusion_sha256"),
    )
    excluded_semantics = excluded_topics | excluded_questions
    rows = _sequence(payload.get("sources"), "sources")
    if len(rows) != expected_source_count:
        raise ValueError(
            f"formal workload must contain exactly {expected_source_count} sources"
        )

    source_ids: set[str] = set()
    queries: set[str] = set()
    urls: set[str] = set()
    topics: set[str] = set()
    questions: set[str] = set()
    for index, raw in enumerate(rows, 1):
        prefix = f"sources[{index - 1}]"
        row = _mapping(raw, prefix)
        if set(row) != {"source_id", "question", "search_query", "expected_url"}:
            raise ValueError(f"{prefix} fields are not frozen exactly")
        source_id = _nonempty_string(row.get("source_id"), f"{prefix}.source_id")
        match = profile["source_id_re"].fullmatch(source_id)
        if match is None or int(match.group(1)) != index:
            raise ValueError(f"{prefix}.source_id is not the expected sequential ID")
        question = _nonempty_string(row.get("question"), f"{prefix}.question")
        query = _nonempty_string(row.get("search_query"), f"{prefix}.search_query")
        if not question.isascii() or not query.isascii() or not question.endswith("?"):
            raise ValueError(f"{prefix} is not a frozen English question/query")
        if len(question.split()) < 6:
            raise ValueError(f"{prefix}.question is too short")
        canonical_url, url_topic = _canonical_wikipedia_topic(
            row.get("expected_url"), f"{prefix}.expected_url"
        )
        query_topic = normalize_topic(query)
        if query_topic != url_topic:
            raise ValueError(f"{prefix} query and canonical URL identify different topics")
        question_topic = normalize_topic(question)
        for semantic_label, semantic_value in (
            ("query", query_topic),
            ("URL topic", url_topic),
            ("question", question_topic),
        ):
            overlap = _semantic_overlap(semantic_value, excluded_semantics)
            if overlap is not None:
                raise ValueError(
                    f"{prefix} {semantic_label} overlaps excluded semantics: {overlap}"
                )
        if source_id in source_ids or query_topic in queries or canonical_url in urls:
            raise ValueError(f"{prefix} duplicates a source ID, query, or URL")
        if query_topic in topics or question_topic in questions:
            raise ValueError(f"{prefix} duplicates a formal topic or question")
        source_ids.add(source_id)
        queries.add(query_topic)
        urls.add(canonical_url)
        topics.add(url_topic)
        questions.add(question_topic)

    canonical_payload = canonical_json_bytes(payload)
    canonical_sources = canonical_json_bytes(list(rows))
    return {
        "schema": "paste_repro.live_joint_formal_workload_validation",
        "version": 1,
        "valid": True,
        "split_id": split_id,
        "source_count": len(rows),
        "formal_eligible": True,
        "created_date_utc": created_date,
        "development_topic_count": len(excluded_topics),
        "development_bindings": exclusion_bindings,
        "excluded_topic_count": len(excluded_topics),
        "excluded_question_count": len(excluded_questions),
        "excluded_semantic_value_count": len(excluded_semantics),
        "exclusion_bindings": exclusion_bindings,
        "file_sha256": _sha256_bytes(workload_path.read_bytes()),
        "canonical_json_sha256": _sha256_bytes(canonical_payload),
        "canonical_sources_sha256": _sha256_bytes(canonical_sources),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--development", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    development = tuple(args.development) if args.development else None
    result = validate_formal_workload(args.workload, development_paths=development)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
