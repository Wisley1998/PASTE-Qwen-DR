"""Strict runtime adapter for the Qwen all-Visit Pattern V2 policy.

This module joins three already-separated pieces without reintroducing the
future-timing inputs used by the historical analytical replay:

* :class:`PatternV2CrossFitPredictor` sees only a completed tool and the next
  currently-visible request;
* :class:`HashedUniformSLOClock` assigns physical service from a frozen hash of
  the normalized invocation, independently of the trace and policy; and
* :class:`PersistentPatternV2ToolExecutor` deliberately exposes no
  per-decision expiry/resolution API.  Wrong predictions remain useful as a
  session URL cache and are preempted only by the authority-first Visit pool
  when real demand needs capacity.

The intended evaluation is a causal-reveal systems trace replay.  The live LLM
does not choose the recorded authoritative call graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Protocol

from .pattern_v2_all_visit_online import (
    PatternV2Prediction,
    TOP_K,
    canonical_sha256,
)
from .strict_trace_runtime import (
    CausalSessionState,
    CausalTailPredictor,
    DurationEstimate,
    POLICY_NAME as SCHEDULER_METADATA_SCHEMA,
    SealedTraceToolExecutor,
    ToolExecutionObservation,
    normalized_tool_arguments,
    normalize_url,
)
from .trace_coscheduler import AsyncPreemptibleVisitPool


SLO_CLOCK_SCHEMA = "paste_repro.hashed_uniform_slo_clock.v1"
DURATION_PREDICTOR_SCHEMA = "paste_repro.public_slo_duration_predictor.v1"
TOOL_POLICY_NAME = "paste.pattern_v2_all_visit.strict_probability_top10.v1"
DEFAULT_SLO_RANGES_S: dict[str, tuple[float, float]] = {
    "search": (1.0, 3.0),
    "google_scholar": (1.0, 3.0),
    "visit": (2.0, 8.0),
}


def _signed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def _checked_signed(
    payload: Mapping[str, Any], *, schema: str, label: str
) -> dict[str, Any]:
    result = dict(payload)
    supplied = result.pop("artifact_sha256", None)
    if not isinstance(supplied, str) or canonical_sha256(result) != supplied:
        raise ValueError(f"{label} checksum mismatch")
    if result.get("schema") != schema:
        raise ValueError(f"unsupported {label} schema")
    result["artifact_sha256"] = supplied
    return result


def _validate_ranges(raw: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, Mapping) or set(raw) != set(DEFAULT_SLO_RANGES_S):
        raise ValueError("SLO ranges must contain search, google_scholar, and visit")
    result: dict[str, tuple[float, float]] = {}
    for name, expected in DEFAULT_SLO_RANGES_S.items():
        value = raw[name]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"invalid SLO range: {name}")
        lower, upper = value
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) <= 0.0
            or float(lower) > float(upper)
        ):
            raise ValueError(f"invalid SLO range: {name}")
        pair = (float(lower), float(upper))
        if pair != expected:
            raise ValueError(
                f"{name} SLO range {pair!r} differs from frozen {expected!r}"
            )
        result[name] = pair
    return result


def new_hashed_slo_clock_artifact(*, seed_sha256: str) -> dict[str, Any]:
    """Freeze the private, policy-independent physical service surface.

    The artifact enumerates no evaluation invocation.  The seed should live in
    the sealed experiment bundle rather than in any policy-facing document.
    """

    if (
        not isinstance(seed_sha256, str)
        or len(seed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in seed_sha256)
    ):
        raise ValueError("clock seed must be a lowercase SHA-256 value")
    return _signed(
        {
            "schema": SLO_CLOCK_SCHEMA,
            "physical_service_clock_mode": "normalized_invocation_hashed_uniform_v1",
            "seed_sha256": seed_sha256,
            "ranges_s": {
                name: [lower, upper]
                for name, (lower, upper) in DEFAULT_SLO_RANGES_S.items()
            },
            "canonicalization": (
                "canonical-json({tool,normalized_arguments}); visit URL normalization "
                "is defined by strict_trace_runtime.normalized_tool_arguments"
            ),
            "selection_rule": "first 64 SHA-256 bits / (2^64-1), affine uniform range",
            "enumerates_evaluation_invocations": False,
            "uses_trace_timing": False,
            "uses_future_authority": False,
            "policy_visible": False,
        }
    )


class HashedUniformSLOClock:
    """Deterministic physical service keyed only by normalized invocation."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        checked = _checked_signed(
            artifact, schema=SLO_CLOCK_SCHEMA, label="hashed SLO clock artifact"
        )
        expected_flags = {
            "enumerates_evaluation_invocations": False,
            "uses_trace_timing": False,
            "uses_future_authority": False,
            "policy_visible": False,
        }
        if any(checked.get(key) is not value for key, value in expected_flags.items()):
            raise ValueError("hashed SLO clock violates its isolation contract")
        seed = checked.get("seed_sha256")
        if (
            not isinstance(seed, str)
            or len(seed) != 64
            or any(character not in "0123456789abcdef" for character in seed)
        ):
            raise ValueError("hashed SLO clock seed is invalid")
        self._ranges = _validate_ranges(checked.get("ranges_s"))
        self._seed = seed
        self._artifact = checked

    @property
    def artifact_sha256(self) -> str:
        return str(self._artifact["artifact_sha256"])

    def service_s(
        self, *, tool_name: str, tool_arguments: Mapping[str, Any]
    ) -> float:
        name = str(tool_name)
        if name not in self._ranges:
            raise ValueError(f"tool has no frozen physical SLO: {name}")
        invocation = {
            "tool": name,
            "arguments": normalized_tool_arguments(name, tool_arguments),
        }
        wire = json.dumps(
            invocation,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(f"{self._seed}\0{wire}".encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
        lower, upper = self._ranges[name]
        return lower + (upper - lower) * unit


def new_public_slo_duration_artifact(*, ewma_alpha: float = 0.35) -> dict[str, Any]:
    """Freeze the policy's causal estimate without including the clock seed."""

    if not 0.0 < float(ewma_alpha) <= 1.0:
        raise ValueError("duration EWMA alpha must be in (0, 1]")
    return _signed(
        {
            "schema": DURATION_PREDICTOR_SCHEMA,
            "prior_source": "public_experiment_SLO_midpoint",
            "ranges_s": {
                name: [lower, upper]
                for name, (lower, upper) in DEFAULT_SLO_RANGES_S.items()
            },
            "ewma_alpha": float(ewma_alpha),
            "runtime_update": "completed_physical_job_only",
            "input_features": ["current_tool_name", "completed_job_service_s_ewma"],
            "uses_clock_seed": False,
            "uses_trace_timing": False,
            "uses_future_authority": False,
        }
    )


class PublicSLODurationPredictor:
    """Public midpoint prior plus observations from already completed jobs."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        checked = _checked_signed(
            artifact,
            schema=DURATION_PREDICTOR_SCHEMA,
            label="public SLO duration artifact",
        )
        if (
            checked.get("prior_source") != "public_experiment_SLO_midpoint"
            or checked.get("runtime_update") != "completed_physical_job_only"
            or checked.get("input_features")
            != ["current_tool_name", "completed_job_service_s_ewma"]
            or checked.get("uses_clock_seed") is not False
            or checked.get("uses_trace_timing") is not False
            or checked.get("uses_future_authority") is not False
        ):
            raise ValueError("public SLO duration predictor contract mismatch")
        alpha = checked.get("ewma_alpha")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not 0.0 < float(alpha) <= 1.0
        ):
            raise ValueError("duration EWMA alpha must be in (0, 1]")
        self._ranges = _validate_ranges(checked.get("ranges_s"))
        self._alpha = float(alpha)
        self._observed_ewma: dict[str, float] = {}
        self._observed_count: dict[str, int] = {}
        self._artifact = checked

    @property
    def artifact_sha256(self) -> str:
        return str(self._artifact["artifact_sha256"])

    def frozen_estimate(
        self, tool_name: str, url: str | None = None
    ) -> DurationEstimate:
        del url
        name = str(tool_name)
        if name not in self._ranges:
            raise ValueError(f"tool has no public duration prior: {name}")
        lower, upper = self._ranges[name]
        return DurationEstimate((lower + upper) / 2.0, "public_SLO_midpoint")

    def estimate(self, tool_name: str, url: str | None = None) -> DurationEstimate:
        del url
        name = str(tool_name)
        frozen = self.frozen_estimate(name)
        observed = self._observed_ewma.get(name)
        if observed is None:
            return frozen
        return DurationEstimate(
            0.5 * frozen.service_s + 0.5 * observed,
            "public_SLO_midpoint+completed_job_ewma",
        )

    def observe_completed(
        self, tool_name: str, service_s: float, url: str | None = None
    ) -> None:
        del url
        name = str(tool_name)
        if name not in self._ranges:
            raise ValueError(f"tool has no public duration prior: {name}")
        if (
            isinstance(service_s, bool)
            or not isinstance(service_s, (int, float))
            or not math.isfinite(float(service_s))
            or float(service_s) < 0.0
        ):
            raise ValueError("completed service must be finite and non-negative")
        value = float(service_s)
        previous = self._observed_ewma.get(name)
        self._observed_ewma[name] = (
            value
            if previous is None
            else self._alpha * value + (1.0 - self._alpha) * previous
        )
        self._observed_count[name] = self._observed_count.get(name, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "observed_ewma_s": dict(sorted(self._observed_ewma.items())),
            "observed_counts": dict(sorted(self._observed_count.items())),
        }


@dataclass(frozen=True)
class PatternV2StrictCandidate:
    url: str
    confidence: float
    predicted_service_s: float
    prediction_source: str
    source_position: int
    trigger_tool: str

    @property
    def admission_score(self) -> float:
        """The frozen policy ranks and preempts by probability only."""

        return self.confidence


class PatternV2SessionLike(Protocol):
    """Small runtime surface shared by cross-fit and deployable predictors."""

    predictor_artifact_sha256: str

    def predict_after_tool(
        self,
        *,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
        current_messages: Sequence[Mapping[str, Any]],
    ) -> tuple[PatternV2Prediction, ...]: ...

    def snapshot(self) -> dict[str, Any]: ...


class PatternV2PredictorLike(Protocol):
    artifact_sha256: str

    def start_session(
        self, *, source_session_id: str, runtime_session_id: str
    ) -> PatternV2SessionLike: ...


class PatternV2StrictSession:
    """Causal per-session adapter over one held-out cross-fit model."""

    def __init__(
        self,
        *,
        session: PatternV2SessionLike,
        duration_predictor: PublicSLODurationPredictor,
    ) -> None:
        self._session = session
        self._duration_predictor = duration_predictor

    @property
    def predictor_artifact_sha256(self) -> str:
        return self._session.predictor_artifact_sha256

    def predict_after_completed_tool(
        self,
        *,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
        current_messages: Sequence[Mapping[str, Any]],
    ) -> tuple[PatternV2StrictCandidate, ...]:
        predictions = self._session.predict_after_tool(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            current_messages=current_messages,
        )
        if len(predictions) > TOP_K:
            raise RuntimeError("Pattern V2 predictor exceeded the frozen Top-10 cap")
        estimate = self._duration_predictor.estimate("visit")
        return tuple(
            PatternV2StrictCandidate(
                url=row.url,
                confidence=row.confidence,
                predicted_service_s=estimate.service_s,
                prediction_source=estimate.source,
                source_position=row.source_position,
                trigger_tool=row.trigger_tool,
            )
            for row in predictions
        )

    def snapshot(self) -> dict[str, Any]:
        return self._session.snapshot()


class PatternV2StrictPolicy:
    """Frozen cross-fit Pattern V2 policy plus optional causal LLM metadata."""

    def __init__(
        self,
        *,
        predictor: PatternV2PredictorLike,
        duration_predictor: PublicSLODurationPredictor,
        tail_predictor: CausalTailPredictor | None = None,
    ) -> None:
        self._predictor = predictor
        self.duration_predictor = duration_predictor
        self.tail_predictor = tail_predictor
        self.predictor_artifact_sha256 = predictor.artifact_sha256
        self.policy_sha256 = canonical_sha256(
            {
                "policy": TOOL_POLICY_NAME,
                "predictor_artifact_sha256": self.predictor_artifact_sha256,
                "duration_predictor_artifact_sha256": (
                    duration_predictor.artifact_sha256
                ),
                "top_k": TOP_K,
                "candidate_ranking": "exact_probability_only_no_duration_input",
                "cache_scope": "session_url_infinite_ttl",
                "visit_pool": "authority_first_adaptive_idle_fill_capacity_64",
                "wrong_candidate_retirement": "authority_capacity_pressure_or_session_close",
            }
        )

    def start_session(
        self, *, source_session_id: str, runtime_session_id: str
    ) -> PatternV2StrictSession:
        return PatternV2StrictSession(
            session=self._predictor.start_session(
                source_session_id=source_session_id,
                runtime_session_id=runtime_session_id,
            ),
            duration_predictor=self.duration_predictor,
        )

    def scheduler_metadata(
        self,
        *,
        trace_id: str,
        request_index: int,
        current_call_index: int,
        prompt_tokens: int,
        max_tokens: int,
        state: CausalSessionState,
        observed_event_seq: int,
        decision_seq: int,
    ) -> dict[str, Any]:
        if self.tail_predictor is None:
            raise RuntimeError("causal joint scheduling requires a tail predictor")
        if decision_seq < observed_event_seq:
            raise ValueError("decision sequence precedes observed event sequence")
        tail = self.tail_predictor.predict(
            current_call_index=current_call_index,
            completed_tool_group_waits_s=state.completed_tool_group_waits_s or (),
        )
        return {
            "t": str(trace_id),
            "c": int(current_call_index),
            "i": int(request_index),
            "pt": max(0, int(prompt_tokens)),
            "mt": max(1, int(max_tokens)),
            "po_hat": max(
                1,
                min(int(max_tokens), int(round(state.predicted_output_tokens))),
            ),
            "tool_eta_s_hat": tail.next_tool_wait_s,
            # This field is the calibrated event probability consumed by the
            # server's expected tool-gain term.  ``tail.reliability`` measures
            # a different quantity (duration-regression MAE skill versus a
            # global median), so multiplying the two would incorrectly turn a
            # valid nonzero event probability into zero whenever the duration
            # model falls back to its population prior.
            "tool_hit_probability_hat": tail.next_tool_probability,
            "tool_eta_reliability_hat": tail.reliability,
            "remaining_tool_wait_s_hat": tail.remaining_tool_wait_s,
            # ``ms`` is a wire-schema discriminator, not the tool-policy name.
            # The server intentionally ignores all ``*_hat`` fields unless this
            # exact fail-closed causal schema is present.
            "ms": SCHEDULER_METADATA_SCHEMA,
            "decision_seq": int(decision_seq),
            "observed_event_seq": int(observed_event_seq),
            "policy_sha256": self.policy_sha256,
            "predictor_artifact_sha256": self.predictor_artifact_sha256,
            "duration_predictor_artifact_sha256": (
                self.duration_predictor.artifact_sha256
            ),
        }


class PersistentPatternV2ToolExecutor:
    """Pattern V2 executor whose API cannot retire candidates per decision.

    It delegates authoritative execution and sealed outcome validation to the
    established strict executor.  Speculative submission goes directly to the
    shared pool so its score is exactly the model probability, independent of
    both actual and predicted duration.
    """

    def __init__(
        self,
        *,
        sealed_outcomes: Mapping[str, Mapping[str, Any]],
        service_clock: HashedUniformSLOClock,
        duration_predictor: PublicSLODurationPredictor,
        visit_pool: AsyncPreemptibleVisitPool,
    ) -> None:
        if visit_pool.capacity != 64 or visit_pool.speculative_cap != 64:
            raise ValueError(
                "strict Pattern V2 requires Visit capacity=64 and speculative_cap=64"
            )
        self._pool = visit_pool
        self._clock = service_clock
        self._delegate = SealedTraceToolExecutor(
            sealed_outcomes=sealed_outcomes,
            service_clock=service_clock,  # runtime-compatible service_s API
            duration_predictor=duration_predictor,  # completed-only observer API
            visit_pool=visit_pool,
        )

    async def speculate(
        self,
        *,
        session_id: str,
        candidates: Sequence[PatternV2StrictCandidate],
        decision_id: str,
    ) -> tuple[bool, ...]:
        rows = [
            (
                str(session_id),
                normalize_url(row.url),
                self._clock.service_s(
                    tool_name="visit", tool_arguments={"url": row.url}
                ),
                row.admission_score,
                str(decision_id),
            )
            for row in candidates
        ]
        return await self._pool.speculate_batch(rows)

    async def execute_authoritative(
        self, *, session_id: str, descriptor: Mapping[str, Any]
    ) -> ToolExecutionObservation:
        return await self._delegate.execute_authoritative(
            session_id=session_id, descriptor=descriptor
        )

    async def close_session(self, session_id: str) -> None:
        await self._delegate.close_session(session_id)

    def snapshot(self) -> dict[str, Any]:
        result = self._delegate.snapshot()
        result["cache_scope"] = "session_url_infinite_ttl"
        result["speculation_admission"] = "adaptive_idle_fill"
        result["wrong_candidate_retirement"] = (
            "authority_capacity_pressure_or_session_close"
        )
        result["candidate_priority"] = "exact_probability_only"
        result["physical_service_clock_sha256"] = self._clock.artifact_sha256
        return result

    async def close(self) -> None:
        await self._delegate.close()


def prediction_evidence(
    candidate: PatternV2StrictCandidate, *, admitted: bool
) -> dict[str, Any]:
    """Serialize policy-facing hats without leaking the physical clock."""

    return {
        **asdict(candidate),
        "admitted": bool(admitted),
        "candidate_priority_hat": candidate.admission_score,
    }
