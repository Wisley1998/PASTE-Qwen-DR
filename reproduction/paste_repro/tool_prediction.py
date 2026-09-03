"""Causal trace-learned predictions for speculative visit execution.

The predictor deliberately has a narrow contract: it observes only a search
response that is already available, maps historically learned displayed ranks
to URLs in that response, and returns concrete visit candidates.  It neither
chooses the authoritative URL nor reads a future trace event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

from .contextual_mapper import (
    CONTEXTUAL_ARTIFACT_SCHEMA,
    CONTEXTUAL_POLICY_VERSION,
    ContextualURLReranker,
    load_contextual_artifact,
)
from .mapper import ARTIFACT_SCHEMA, Prediction, URLRankMapper, load_artifact
from .traces import SearchResult, parse_search_results


TRACE_LEARNED_VISIT_POLICY_VERSION = "visible-search-learned-rank-v1"


class VisitPredictor(Protocol):
    """Runtime interface shared by legacy and contextual visit predictors."""

    top_k: int
    artifact_sha256: str

    @property
    def policy(self) -> str: ...

    def predict_structured_result(self, result: Any) -> tuple[str, ...]: ...

    def metadata(self) -> dict[str, Any]: ...


def structured_search_results(result: Any) -> tuple[SearchResult, ...]:
    """Convert a live executor search result to the mapper's stable input.

    Older test executors only supplied ``url``.  Rank and query index therefore
    have deterministic fallbacks while the production executor's explicit
    fields are preserved.
    """

    if not isinstance(result, Mapping) or result.get("tool") != "search":
        raise ValueError("structured search result must be a search tool object")
    rows = result.get("results")
    if not isinstance(rows, list):
        raise ValueError("structured search result rows must be a list")

    converted: list[SearchResult] = []
    next_rank_by_query: dict[int, int] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"structured search result {ordinal} is invalid")
        url = row.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(f"structured search result {ordinal} has an invalid URL")

        raw_query_index = row.get("query_index", 0)
        query_index = (
            raw_query_index
            if isinstance(raw_query_index, int) and not isinstance(raw_query_index, bool)
            else 0
        )
        query_index = max(0, query_index)
        fallback_rank = next_rank_by_query.get(query_index, 1)
        raw_rank = row.get("rank")
        result_rank = (
            raw_rank
            if isinstance(raw_rank, int)
            and not isinstance(raw_rank, bool)
            and raw_rank > 0
            else fallback_rank
        )
        next_rank_by_query[query_index] = max(fallback_rank, result_rank) + 1
        converted.append(
            SearchResult(
                url=url,
                result_rank=result_rank,
                ordinal=ordinal,
                query_index=query_index,
                title=str(row.get("title") or ""),
                query=str(row.get("query") or ""),
                snippet=str(row.get("snippet") or ""),
            )
        )
    return tuple(converted)


@dataclass(frozen=True)
class TraceLearnedVisitPredictor:
    """Reusable adapter around a checksummed :class:`URLRankMapper` artifact."""

    mapper: URLRankMapper
    top_k: int = 5
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    @classmethod
    def from_artifact(
        cls, path: str | Path, *, top_k: int = 5
    ) -> "TraceLearnedVisitPredictor":
        mapper, artifact = load_artifact(path)
        checksum = artifact.get("artifact_sha256")
        if not isinstance(checksum, str) or not checksum:
            # load_artifact already validates this; keep the invariant local.
            raise ValueError("model artifact is missing artifact_sha256")
        return cls(mapper=mapper, top_k=top_k, artifact_sha256=checksum)

    @property
    def policy(self) -> str:
        return TRACE_LEARNED_VISIT_POLICY_VERSION

    def predict(self, search_results: Sequence[SearchResult]) -> tuple[Prediction, ...]:
        return self.mapper.predict(search_results, self.top_k)

    def predict_urls(self, search_results: Sequence[SearchResult]) -> tuple[str, ...]:
        urls: list[str] = []
        for prediction in self.predict(search_results):
            url = prediction.invocation.arguments.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
        return tuple(urls)

    def predict_visible_response(self, tool_response: str) -> tuple[str, ...]:
        """Predict from trace text already present in the decision request."""

        return self.predict_urls(parse_search_results(tool_response))

    def predict_structured_result(self, result: Any) -> tuple[str, ...]:
        """Predict from the current live executor response, never future state."""

        return self.predict_urls(structured_search_results(result))

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "top_k": self.top_k,
            "artifact_sha256": self.artifact_sha256 or None,
            "mapper": self.mapper.summary(),
        }


@dataclass(frozen=True)
class ContextualTraceVisitPredictor:
    """Artifact-backed adapter for the current-visible contextual reranker."""

    reranker: ContextualURLReranker
    top_k: int = 5
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    @classmethod
    def from_artifact(
        cls, path: str | Path, *, top_k: int = 5
    ) -> "ContextualTraceVisitPredictor":
        reranker, artifact = load_contextual_artifact(path)
        checksum = artifact.get("artifact_sha256")
        if not isinstance(checksum, str) or not checksum:
            raise ValueError("contextual model artifact is missing artifact_sha256")
        return cls(
            reranker=reranker,
            top_k=top_k,
            artifact_sha256=checksum,
        )

    @property
    def policy(self) -> str:
        return CONTEXTUAL_POLICY_VERSION

    def predict(self, search_results: Sequence[SearchResult]) -> tuple[Prediction, ...]:
        return self.reranker.predict(search_results, self.top_k)

    def predict_urls(self, search_results: Sequence[SearchResult]) -> tuple[str, ...]:
        return tuple(
            str(prediction.invocation.arguments["url"])
            for prediction in self.predict(search_results)
        )

    def predict_visible_response(self, tool_response: str) -> tuple[str, ...]:
        return self.predict_urls(parse_search_results(tool_response))

    def predict_structured_result(self, result: Any) -> tuple[str, ...]:
        return self.predict_urls(structured_search_results(result))

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "top_k": self.top_k,
            "artifact_sha256": self.artifact_sha256 or None,
            "reranker": self.reranker.summary(),
        }


def load_visit_predictor(
    path: str | Path, *, top_k: int = 5
) -> TraceLearnedVisitPredictor | ContextualTraceVisitPredictor:
    """Load a checksummed predictor artifact by its explicit schema."""

    artifact_path = Path(path)
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("visit predictor artifact root must be an object")
    schema = raw.get("schema")
    if schema == ARTIFACT_SCHEMA:
        return TraceLearnedVisitPredictor.from_artifact(
            artifact_path, top_k=top_k
        )
    if schema == CONTEXTUAL_ARTIFACT_SCHEMA:
        return ContextualTraceVisitPredictor.from_artifact(
            artifact_path, top_k=top_k
        )
    raise ValueError(f"unsupported visit predictor artifact schema: {schema!r}")
