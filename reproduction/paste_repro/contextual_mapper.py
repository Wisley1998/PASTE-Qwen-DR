"""Causal contextual reranking for exact-URL visit speculation.

The legacy mapper intentionally learns only a global displayed-rank prior.
This module keeps the same safety boundary (raw URLs from the current visible
search response only) while using query position, duplicate appearances, and
fixed lexical similarity features.  Training is a deterministic, regularized
same-transition pairwise logistic objective.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np

from .invocation import Invocation
from .mapper import Prediction, write_json_atomic
from .traces import SearchResult, SearchVisitTransition


CONTEXTUAL_ARTIFACT_SCHEMA = "paste_repro.contextual_url_reranker"
CONTEXTUAL_ARTIFACT_VERSION = 1
CONTEXTUAL_POLICY_VERSION = "visible-search-contextual-pairwise-v1"
DEFAULT_L2 = 3.0
DEFAULT_MAX_ITERATIONS = 60
DEFAULT_TOLERANCE = 1e-7


FEATURE_SCHEMA = (
    "intercept",
    *(f"rank_is_{rank}" for rank in range(1, 6)),
    *(f"query_index_is_{index}_capped" for index in range(7)),
    *(
        f"rank_{rank}_x_query_{query_index}_capped"
        for rank in range(1, 6)
        for query_index in range(4)
    ),
    "log1p_url_occurrences",
    "unique_query_fraction",
    "normalized_first_query_position",
    "reciprocal_first_rank",
    "first_query_title_unigram_coverage",
    "first_query_title_bigram_coverage",
    "first_query_title_token_jaccard",
    "first_query_decoded_path_bigram_coverage",
    "best_query_title_unigram_coverage",
    "best_query_title_bigram_coverage",
    "best_query_title_token_jaccard",
    "best_query_decoded_path_bigram_coverage",
    "scaled_log1p_path_length",
    "scaled_log1p_host_candidate_count",
    "path_has_pdf_suffix",
    "normalized_query_is_title_substring",
)


_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", flags=re.IGNORECASE)


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


def _normalized_alnum(value: str) -> str:
    return "".join(
        character.lower()
        for character in unicodedata.normalize("NFKC", value)
        if character.isalnum()
    )


def _ngrams(value: str, width: int) -> set[str]:
    normalized = _normalized_alnum(value)
    if len(normalized) < width:
        return set()
    return {
        normalized[index : index + width]
        for index in range(len(normalized) - width + 1)
    }


def _query_coverage(query: str, text: str, width: int) -> float:
    query_grams = _ngrams(query, width)
    if not query_grams:
        return 0.0
    return len(query_grams & _ngrams(text, width)) / len(query_grams)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_TOKEN.findall(unicodedata.normalize("NFKC", left).lower()))
    right_tokens = set(_TOKEN.findall(unicodedata.normalize("NFKC", right).lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


@dataclass(frozen=True)
class ContextualCandidate:
    """One raw-URL candidate and its fixed feature vector."""

    url: str
    features: tuple[float, ...]
    source_rank: int
    source_ordinal: int
    source_query_index: int


def contextual_candidates(
    search_results: Sequence[SearchResult],
) -> tuple[ContextualCandidate, ...]:
    """Build stable current-response candidates without reading future state."""

    grouped: dict[str, list[SearchResult]] = {}
    for result in search_results:
        grouped.setdefault(result.url, []).append(result)
    if not grouped:
        return ()

    query_count = max(
        1,
        max(result.query_index for result in search_results) + 1,
    )
    host_candidate_count = Counter(
        urlsplit(url).netloc.lower() for url in grouped
    )
    candidates: list[ContextualCandidate] = []
    for url, occurrences in grouped.items():
        first = occurrences[0]
        parsed = urlsplit(url)
        decoded_path = unquote(f"{parsed.path} {parsed.query}")

        occurrence_similarities: list[tuple[float, float, float, float, int]] = []
        for occurrence in occurrences:
            occurrence_similarities.append(
                (
                    _query_coverage(occurrence.query, occurrence.title, 1),
                    _query_coverage(occurrence.query, occurrence.title, 2),
                    _token_jaccard(occurrence.query, occurrence.title),
                    _query_coverage(occurrence.query, decoded_path, 2),
                    occurrence.ordinal,
                )
            )
        first_similarity = occurrence_similarities[0]
        best_similarity = max(
            occurrence_similarities,
            key=lambda values: (values[1], values[0], -values[4]),
        )

        rank = min(5, max(1, first.result_rank))
        query_index = min(6, max(0, first.query_index))
        interaction_query_index = min(3, max(0, first.query_index))
        normalized_query = _normalized_alnum(first.query)
        normalized_title = _normalized_alnum(first.title)
        features: list[float] = [1.0]
        features.extend(float(rank == value) for value in range(1, 6))
        features.extend(float(query_index == value) for value in range(7))
        features.extend(
            float(rank == rank_value and interaction_query_index == query_value)
            for rank_value in range(1, 6)
            for query_value in range(4)
        )
        features.extend(
            (
                math.log1p(len(occurrences)),
                len({item.query_index for item in occurrences}) / query_count,
                first.query_index / max(1, query_count - 1),
                1.0 / first.result_rank,
                *first_similarity[:4],
                *best_similarity[:4],
                math.log1p(len(parsed.path)) / 8.0,
                math.log1p(host_candidate_count[parsed.netloc.lower()]) / 4.0,
                float(parsed.path.lower().endswith(".pdf")),
                float(bool(normalized_query) and normalized_query in normalized_title),
            )
        )
        if len(features) != len(FEATURE_SCHEMA):  # pragma: no cover - invariant
            raise AssertionError("contextual feature schema length mismatch")
        candidates.append(
            ContextualCandidate(
                url=url,
                features=tuple(features),
                source_rank=first.result_rank,
                source_ordinal=first.ordinal,
                source_query_index=first.query_index,
            )
        )
    return tuple(candidates)


class ContextualURLReranker:
    """Pairwise reranker over raw URLs in the current visible response."""

    def __init__(
        self,
        *,
        l2: float = DEFAULT_L2,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        if not math.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be a finite positive number")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be a finite positive number")
        self.l2 = float(l2)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self._weights = np.zeros(len(FEATURE_SCHEMA), dtype=float)
        self.transitions_seen = 0
        self.targets_seen = 0
        self.mapped_targets = 0
        self.candidates_seen = 0
        self.pair_count = 0
        self.weighted_pair_mass = 0.0
        self.optimizer_iterations = 0
        self.optimizer_converged = False

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._weights)

    @property
    def policy(self) -> str:
        return CONTEXTUAL_POLICY_VERSION

    def fit(
        self, transitions: Iterable[SearchVisitTransition]
    ) -> "ContextualURLReranker":
        transition_list = tuple(transitions)
        pair_rows: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        pair_weights: list[float] = []
        self.transitions_seen = len(transition_list)
        self.targets_seen = sum(
            len(transition.authoritative_urls) for transition in transition_list
        )
        self.mapped_targets = 0
        self.candidates_seen = 0

        for transition in transition_list:
            candidates = contextual_candidates(transition.search_results)
            self.candidates_seen += len(candidates)
            target_urls = set(transition.authoritative_urls)
            positives = [candidate for candidate in candidates if candidate.url in target_urls]
            negatives = [candidate for candidate in candidates if candidate.url not in target_urls]
            self.mapped_targets += len(positives)
            if not positives or not negatives:
                continue
            negative_weight = 1.0 / len(negatives)
            for positive in positives:
                positive_features = np.asarray(positive.features, dtype=float)
                for negative in negatives:
                    pair_rows.append(
                        positive_features
                        - np.asarray(negative.features, dtype=float)
                    )
                    pair_weights.append(negative_weight)

        self.pair_count = len(pair_rows)
        self.weighted_pair_mass = float(sum(pair_weights))
        self._weights = np.zeros(len(FEATURE_SCHEMA), dtype=float)
        self.optimizer_iterations = 0
        self.optimizer_converged = not pair_rows
        if not pair_rows:
            return self

        matrix = np.asarray(pair_rows, dtype=float)
        sample_weights = np.asarray(pair_weights, dtype=float)
        regularized = np.ones(len(FEATURE_SCHEMA), dtype=float)
        regularized[0] = 0.0

        def objective(weights: np.ndarray[Any, np.dtype[np.float64]]) -> float:
            margins = np.clip(matrix @ weights, -50.0, 50.0)
            return float(
                np.sum(sample_weights * np.logaddexp(0.0, -margins))
                + self.l2 * 0.5 * np.sum((weights * regularized) ** 2)
            )

        for iteration in range(1, self.max_iterations + 1):
            margins = np.clip(matrix @ self._weights, -50.0, 50.0)
            probabilities = 1.0 / (1.0 + np.exp(-margins))
            gradient = (
                matrix.T @ (sample_weights * (probabilities - 1.0))
                + self.l2 * regularized * self._weights
            )
            curvature = sample_weights * probabilities * (1.0 - probabilities)
            hessian = (
                matrix.T @ (matrix * curvature[:, None])
                + np.diag(self.l2 * regularized + 1e-8)
            )
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive fallback
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            base_objective = objective(self._weights)
            directional_derivative = float(gradient @ step)
            scale = 1.0
            while scale > 1e-5:
                proposed = self._weights - scale * step
                if objective(proposed) <= (
                    base_objective - 1e-4 * scale * directional_derivative
                ):
                    break
                scale *= 0.5
            update = scale * step
            self._weights -= update
            self.optimizer_iterations = iteration
            if float(np.max(np.abs(update))) < self.tolerance:
                self.optimizer_converged = True
                break
        return self

    def score_candidates(
        self, search_results: Sequence[SearchResult]
    ) -> tuple[tuple[ContextualCandidate, float], ...]:
        scored = [
            (candidate, float(np.asarray(candidate.features) @ self._weights))
            for candidate in contextual_candidates(search_results)
        ]
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].source_ordinal,
                item[0].url,
            )
        )
        return tuple(scored)

    def predict(
        self, search_results: Sequence[SearchResult], top_k: int
    ) -> tuple[Prediction, ...]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or self.pair_count == 0:
            return ()
        scored = self.score_candidates(search_results)
        if not scored:
            return ()
        # Pairwise training identifies score differences, not an absolute
        # calibrated probability.  Expose a stable softmax-relative weight;
        # admission/abstention must use a separately calibrated next-tool gate.
        maximum = max(score for _, score in scored)
        exponentials = [math.exp(max(-50.0, score - maximum)) for _, score in scored]
        denominator = sum(exponentials)
        return tuple(
            Prediction(
                invocation=Invocation("visit", {"url": candidate.url}),
                source_rank=candidate.source_rank,
                source_ordinal=candidate.source_ordinal,
                confidence=(exponentials[index] / denominator),
            )
            for index, (candidate, _score) in enumerate(scored[:top_k])
        )

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "current-visible contextual pairwise exact-URL reranker",
            "policy": self.policy,
            "feature_schema": list(FEATURE_SCHEMA),
            "feature_count": len(FEATURE_SCHEMA),
            "l2": self.l2,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "transitions_seen": self.transitions_seen,
            "targets_seen": self.targets_seen,
            "mapped_targets": self.mapped_targets,
            "training_executable_coverage": (
                self.mapped_targets / self.targets_seen if self.targets_seen else 0.0
            ),
            "candidates_seen": self.candidates_seen,
            "pair_count": self.pair_count,
            "weighted_pair_mass": self.weighted_pair_mass,
            "optimizer_iterations": self.optimizer_iterations,
            "optimizer_converged": self.optimizer_converged,
            "confidence_semantics": (
                "softmax-relative candidate score; not a calibrated next-tool "
                "or hit probability"
            ),
        }

    def to_artifact(self, training_split: Mapping[str, Any]) -> dict[str, Any]:
        split = dict(training_split)
        supplied_split_checksum = split.pop("manifest_sha256", None)
        computed_split_checksum = _sha256_json(split)
        if supplied_split_checksum not in (None, computed_split_checksum):
            raise ValueError("training split manifest checksum does not match its content")
        split["manifest_sha256"] = computed_split_checksum
        artifact: dict[str, Any] = {
            "schema": CONTEXTUAL_ARTIFACT_SCHEMA,
            "version": CONTEXTUAL_ARTIFACT_VERSION,
            "model": self.summary(),
            "weights": list(self.weights),
            "training_split": split,
        }
        artifact["artifact_sha256"] = _sha256_json(artifact)
        return artifact

    @classmethod
    def from_artifact(
        cls, artifact: Mapping[str, Any]
    ) -> "ContextualURLReranker":
        raw = dict(artifact)
        checksum = raw.pop("artifact_sha256", None)
        if not isinstance(checksum, str) or checksum != _sha256_json(raw):
            raise ValueError("contextual model artifact checksum mismatch")
        if raw.get("schema") != CONTEXTUAL_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported contextual artifact schema: {raw.get('schema')!r}")
        if raw.get("version") != CONTEXTUAL_ARTIFACT_VERSION:
            raise ValueError(f"unsupported contextual artifact version: {raw.get('version')!r}")
        model_raw = raw.get("model")
        if not isinstance(model_raw, Mapping):
            raise ValueError("contextual model artifact is missing model metadata")
        if tuple(model_raw.get("feature_schema", ())) != FEATURE_SCHEMA:
            raise ValueError("contextual model feature schema mismatch")
        split_raw = raw.get("training_split")
        if not isinstance(split_raw, Mapping):
            raise ValueError("contextual model artifact is missing training_split")
        split = dict(split_raw)
        split_checksum = split.pop("manifest_sha256", None)
        if not isinstance(split_checksum, str) or split_checksum != _sha256_json(split):
            raise ValueError("contextual training split manifest checksum mismatch")
        weights_raw = raw.get("weights")
        if not isinstance(weights_raw, list) or len(weights_raw) != len(FEATURE_SCHEMA):
            raise ValueError("contextual model weight vector has the wrong length")
        try:
            weights = np.asarray([float(value) for value in weights_raw], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("contextual model weights must be numeric") from exc
        if not np.all(np.isfinite(weights)):
            raise ValueError("contextual model weights must be finite")

        restored = cls(
            l2=float(model_raw.get("l2", DEFAULT_L2)),
            max_iterations=int(
                model_raw.get("max_iterations", DEFAULT_MAX_ITERATIONS)
            ),
            tolerance=float(model_raw.get("tolerance", DEFAULT_TOLERANCE)),
        )
        restored._weights = weights
        for field in (
            "transitions_seen",
            "targets_seen",
            "mapped_targets",
            "candidates_seen",
            "pair_count",
            "optimizer_iterations",
        ):
            value = int(model_raw.get(field, 0))
            if value < 0:
                raise ValueError(f"contextual model {field} must be non-negative")
            setattr(restored, field, value)
        restored.weighted_pair_mass = float(model_raw.get("weighted_pair_mass", 0.0))
        restored.optimizer_converged = bool(model_raw.get("optimizer_converged", False))
        return restored


def save_contextual_artifact(
    path: str | Path, artifact: Mapping[str, Any]
) -> None:
    ContextualURLReranker.from_artifact(artifact)
    write_json_atomic(path, artifact)


def load_contextual_artifact(
    path: str | Path,
) -> tuple[ContextualURLReranker, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("contextual model artifact root must be an object")
    return ContextualURLReranker.from_artifact(raw), raw
