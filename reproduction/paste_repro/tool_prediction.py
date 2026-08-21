"""Causal trace-learned predictions for speculative visit execution.

The predictor deliberately has a narrow contract: it observes only a search
response that is already available, maps historically learned displayed ranks
to URLs in that response, and returns concrete visit candidates.  It neither
chooses the authoritative URL nor reads a future trace event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mapper import Prediction, URLRankMapper, load_artifact
from .traces import SearchResult, parse_search_results


TRACE_LEARNED_VISIT_POLICY_VERSION = "visible-search-learned-rank-v1"


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
