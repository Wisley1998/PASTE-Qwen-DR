"""A trace-learned search-result rank mapper."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .invocation import Invocation
from .traces import SearchResult, SearchVisitTransition


ARTIFACT_SCHEMA = "paste_repro.url_rank_mapper"
ARTIFACT_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Prediction:
    invocation: Invocation
    source_rank: int
    source_ordinal: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation": self.invocation.to_dict(),
            "source_rank": self.source_rank,
            "source_ordinal": self.source_ordinal,
            "confidence": self.confidence,
        }


class URLRankMapper:
    """Learn which within-query search ranks are selected for visits.

    Search responses contain multiple query blocks whose displayed ranks reset
    (normally 1--5).  Training counts the displayed rank for every historical
    URL that flowed into the immediately following visit.  At inference time
    the learned frequency orders URLs from the *current* search response;
    URLs themselves are never memorized or guessed.
    """

    def __init__(self) -> None:
        self._rank_counts: Counter[int] = Counter()
        self.transitions_seen = 0
        self.targets_seen = 0
        self.mapped_targets = 0
        self.searches_seen = 0

    def fit(
        self,
        transitions: Iterable[SearchVisitTransition],
        *,
        searches_seen: int | None = None,
    ) -> "URLRankMapper":
        self._rank_counts.clear()
        self.transitions_seen = 0
        self.targets_seen = 0
        self.mapped_targets = 0
        transition_list = tuple(transitions)
        for transition in transition_list:
            self.transitions_seen += 1
            # First occurrence is deterministic when query variants duplicate a URL.
            url_to_result: dict[str, SearchResult] = {}
            for result in transition.search_results:
                url_to_result.setdefault(result.url, result)
            for url in transition.authoritative_urls:
                self.targets_seen += 1
                source = url_to_result.get(url)
                if source is None:
                    continue
                self._rank_counts[source.result_rank] += 1
                self.mapped_targets += 1
        self.searches_seen = (
            self.transitions_seen if searches_seen is None else max(0, int(searches_seen))
        )
        return self

    @property
    def learned_rank_order(self) -> tuple[int, ...]:
        return tuple(
            rank
            for rank, _ in sorted(
                self._rank_counts.items(), key=lambda item: (-item[1], item[0])
            )
        )

    @property
    def rank_counts(self) -> dict[int, int]:
        return dict(sorted(self._rank_counts.items()))

    @property
    def transition_confidence(self) -> float:
        if not self.searches_seen:
            return 0.0
        return self.transitions_seen / self.searches_seen

    def predict(
        self, search_results: Sequence[SearchResult], top_k: int
    ) -> tuple[Prediction, ...]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not self._rank_counts or self.mapped_targets == 0:
            return ()

        # A concrete URL is emitted once, even if several query variants returned it.
        unique_results: dict[str, SearchResult] = {}
        for result in search_results:
            unique_results.setdefault(result.url, result)
        eligible = [
            result
            for result in unique_results.values()
            if self._rank_counts[result.result_rank] > 0
        ]
        eligible.sort(
            key=lambda result: (
                -self._rank_counts[result.result_rank],
                result.ordinal,
                result.url,
            )
        )
        predictions = []
        for result in eligible[:top_k]:
            predictions.append(
                Prediction(
                    invocation=Invocation("visit", {"url": result.url}),
                    source_rank=result.result_rank,
                    source_ordinal=result.ordinal,
                    confidence=(
                        self._rank_counts[result.result_rank] / self.mapped_targets
                    ),
                )
            )
        return tuple(predictions)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "learned_within_query_url_rank",
            "transitions_seen": self.transitions_seen,
            "searches_seen": self.searches_seen,
            "transition_confidence": self.transition_confidence,
            "targets_seen": self.targets_seen,
            "mapped_targets": self.mapped_targets,
            "training_executable_coverage": (
                self.mapped_targets / self.targets_seen if self.targets_seen else 0.0
            ),
            "rank_counts": {str(rank): count for rank, count in self.rank_counts.items()},
            "learned_rank_order": list(self.learned_rank_order),
        }

    def to_artifact(self, training_split: Mapping[str, Any]) -> dict[str, Any]:
        """Build a stable, checksummed model artifact."""

        split = dict(training_split)
        supplied_split_checksum = split.pop("manifest_sha256", None)
        computed_split_checksum = _sha256_json(split)
        if supplied_split_checksum not in (None, computed_split_checksum):
            raise ValueError("training split manifest checksum does not match its content")
        split["manifest_sha256"] = computed_split_checksum
        artifact: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "version": ARTIFACT_VERSION,
            "mapper": self.summary(),
            "training_split": split,
        }
        artifact["artifact_sha256"] = _sha256_json(artifact)
        return artifact

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "URLRankMapper":
        """Validate and restore a mapper without silently retraining it."""

        raw = dict(artifact)
        checksum = raw.pop("artifact_sha256", None)
        if not isinstance(checksum, str) or checksum != _sha256_json(raw):
            raise ValueError("model artifact checksum mismatch")
        if raw.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported model artifact schema: {raw.get('schema')!r}")
        if raw.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"unsupported model artifact version: {raw.get('version')!r}")
        split_raw = raw.get("training_split")
        if not isinstance(split_raw, Mapping):
            raise ValueError("model artifact is missing training_split")
        split = dict(split_raw)
        split_checksum = split.pop("manifest_sha256", None)
        if not isinstance(split_checksum, str) or split_checksum != _sha256_json(split):
            raise ValueError("training split manifest checksum mismatch")
        mapper_raw = raw.get("mapper")
        if not isinstance(mapper_raw, Mapping):
            raise ValueError("model artifact is missing mapper")
        rank_counts_raw = mapper_raw.get("rank_counts")
        if not isinstance(rank_counts_raw, Mapping):
            raise ValueError("model artifact rank_counts must be an object")

        restored = cls()
        try:
            restored._rank_counts.update(
                {
                    int(rank): int(count)
                    for rank, count in rank_counts_raw.items()
                    if int(rank) > 0 and int(count) > 0
                }
            )
            restored.transitions_seen = int(mapper_raw.get("transitions_seen", 0))
            restored.searches_seen = int(mapper_raw.get("searches_seen", 0))
            restored.targets_seen = int(mapper_raw.get("targets_seen", 0))
            restored.mapped_targets = int(mapper_raw.get("mapped_targets", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("model artifact contains invalid numeric counts") from exc
        if min(
            restored.transitions_seen,
            restored.searches_seen,
            restored.targets_seen,
            restored.mapped_targets,
        ) < 0:
            raise ValueError("model artifact counts must be non-negative")
        if sum(restored._rank_counts.values()) != restored.mapped_targets:
            raise ValueError("rank_counts do not sum to mapped_targets")
        return restored


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write stable, human-readable JSON in the target directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def save_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Validate, then atomically write a deterministic UTF-8 artifact."""

    URLRankMapper.from_artifact(artifact)
    write_json_atomic(path, artifact)


def load_artifact(path: str | Path) -> tuple[URLRankMapper, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model artifact root must be an object")
    mapper = URLRankMapper.from_artifact(raw)
    return mapper, raw
