"""Causal runtime primitives for the strict Qwen trace A/B/E/F experiment.

This module deliberately separates three kinds of information:

* policy inputs contain only the current LLM request and completed history;
* calibration artifacts contain statistics fitted from the fixed calibration
  sessions; and
* sealed outcomes contain replay-only authoritative result structure, while a
  separate calibration-only private clock assigns every physical service.

The separation is an audit boundary, not a claim that trace replay is an
autonomous agent evaluation.  Recorded tool calls are revealed only after the
corresponding live LLM turn completes.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import statistics
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .mapper import URLRankMapper
from .trace_coscheduler import AsyncPreemptibleVisitPool, VisitResult
from .traces import LLMCall, SessionTrace, ToolCall, parse_search_results


DURATION_SCHEMA = "paste_repro.strict_duration_predictor.v1"
TAIL_SCHEMA = "paste_repro.strict_session_tail_predictor.v1"
SERVICE_CLOCK_SCHEMA = "paste_repro.calibration_hashed_service_clock.v1"
POLICY_NAME = "paste.schedx.causal_prediction.v1"


def canonical_sha256(payload: Any) -> str:
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def signed_payload(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(checksum_field, None)
    result[checksum_field] = canonical_sha256(result)
    return result


def validate_signed_payload(
    payload: Mapping[str, Any], checksum_field: str, *, label: str
) -> dict[str, Any]:
    result = dict(payload)
    supplied = result.pop(checksum_field, None)
    expected = canonical_sha256(result)
    if not isinstance(supplied, str) or supplied != expected:
        raise ValueError(f"{label} checksum mismatch")
    result[checksum_field] = supplied
    return result


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _finite_positive(value: Any, *, label: str) -> float:
    result = _finite_nonnegative(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _fit_code_sha256(callable_object: Any) -> str:
    """Hash the exact checked-in source of an artifact-fitting entry point."""

    return hashlib.sha256(
        inspect.getsource(callable_object).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _median(values: Sequence[float], fallback: float) -> float:
    return statistics.median(values) if values else fallback


def _domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def normalize_url(url: str) -> str:
    """Canonicalize stable HTTP URL equivalences used by cache/service keys."""

    value = str(url).strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        return value
    netloc = parsed.netloc.lower()
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def normalized_tool_arguments(
    tool_name: str, tool_arguments: Mapping[str, Any]
) -> dict[str, Any]:
    result = json.loads(
        json.dumps(
            dict(tool_arguments),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if str(tool_name) != "visit":
        return result
    raw = result.get("url")
    if isinstance(raw, str):
        result["url"] = normalize_url(raw)
    elif isinstance(raw, list):
        result["url"] = [
            normalize_url(value) if isinstance(value, str) else value
            for value in raw
        ]
    return result


def visit_urls(call: ToolCall) -> tuple[str, ...]:
    raw = call.tool_args.get("url")
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, list):
        return tuple(value for value in raw if isinstance(value, str) and value)
    return ()


def sealed_tool_key(
    session_id: str, tool_name: str, tool_arguments: Mapping[str, Any]
) -> str:
    """Opaque key for a pre-sealed, policy-independent tool outcome."""

    return canonical_sha256(
        {
            "session_id": str(session_id),
            "tool": str(tool_name),
            "arguments": normalized_tool_arguments(tool_name, tool_arguments),
        }
    )


def sealed_visit_key(session_id: str, url: str) -> str:
    """Backward-compatible convenience wrapper for atomic Visit keys."""

    return sealed_tool_key(session_id, "visit", {"url": str(url)})


def corrected_tool_outcome(call: ToolCall) -> dict[str, Any]:
    """Extract corrected timing for calibration-only fitting.

    This strict parser is intentionally suitable only for calibration traces.
    Evaluation plan construction must use authoritative tool arguments for its
    invocation graph and treat recorded timing through a non-throwing diagnostic
    parser; neither the online policy nor the physical service clock calls this.
    """

    correction = call.timing_correction or {}
    duration = correction.get("duration_s")
    if duration is None:
        return {
            "duration_s": None,
            "visit_units": [],
            "source": "missing_calibration_fallback",
        }
    duration_s = _finite_nonnegative(duration, label="tool duration")
    units_raw = correction.get("unit_duration_s", [])
    if not isinstance(units_raw, list):
        raise ValueError("unit_duration_s must be a list")
    unit_durations = [
        _finite_nonnegative(value, label="visit unit duration")
        for value in units_raw
    ]
    urls = visit_urls(call) if call.tool_name == "visit" else ()
    visit_units: list[dict[str, Any]] = []
    if call.tool_name == "visit":
        if len(urls) != len(unit_durations):
            raise ValueError(
                f"visit URL/duration mismatch at line {call.line_number}: "
                f"{len(urls)} != {len(unit_durations)}"
            )
        visit_units = [
            {"url": url, "duration_s": unit_duration}
            for url, unit_duration in zip(urls, unit_durations, strict=True)
        ]
    return {
        "duration_s": duration_s,
        "visit_units": visit_units,
        "source": "corrected_trace_sealed_outcome",
    }


@dataclass(frozen=True)
class DurationEstimate:
    service_s: float
    source: str


class CalibrationHashedServiceClock:
    """Policy-independent physical clock frozen solely from calibration data."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        checked = validate_signed_payload(
            artifact, "artifact_sha256", label="calibration service clock artifact"
        )
        if checked.get("schema") != SERVICE_CLOCK_SCHEMA:
            raise ValueError("unsupported calibration service clock schema")
        if checked.get("training_role") != "calibration":
            raise ValueError("service clock was not fitted on calibration")
        if checked.get("uses_evaluation_labels") is not False:
            raise ValueError("service clock must not use evaluation labels")
        if checked.get("future_state_accepted_invariant") is not True:
            raise ValueError("service clock must be invariant to future acceptance labels")
        seed = checked.get("seed_sha256")
        if (
            not isinstance(seed, str)
            or len(seed) != 64
            or any(character not in "0123456789abcdef" for character in seed)
        ):
            raise ValueError("service clock seed is invalid")
        raw_samples = checked.get("samples_by_tool_s")
        if not isinstance(raw_samples, Mapping):
            raise ValueError("service clock samples are invalid")
        samples: dict[str, tuple[float, ...]] = {}
        for tool_name, raw_values in raw_samples.items():
            if not isinstance(raw_values, list) or not raw_values:
                raise ValueError(f"service clock samples are empty: {tool_name}")
            samples[str(tool_name)] = tuple(
                _finite_positive(value, label=f"service sample {tool_name}")
                for value in raw_values
            )
        if "__global__" not in samples:
            raise ValueError("service clock lacks a global fallback")
        minimum_pool_size = checked.get("minimum_selection_pool_size")
        if type(minimum_pool_size) is not int or minimum_pool_size < 3:
            raise ValueError("service clock minimum selection pool must be at least 3")
        if len(samples["__global__"]) < minimum_pool_size:
            raise ValueError("service clock global fallback pool is too small")
        self._artifact = checked
        self._seed = seed
        self._samples = samples
        self._minimum_pool_size = minimum_pool_size

    @property
    def artifact_sha256(self) -> str:
        return str(self._artifact["artifact_sha256"])

    def service_s(
        self, *, tool_name: str, tool_arguments: Mapping[str, Any]
    ) -> float:
        normalized = {
            "tool": str(tool_name),
            "arguments": normalized_tool_arguments(tool_name, tool_arguments),
        }
        digest = hashlib.sha256(
            (
                f"{self._seed}\0"
                + json.dumps(
                    normalized,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ).encode("utf-8")
        ).digest()
        tool_samples = self._samples.get(str(tool_name), ())
        samples = (
            tool_samples
            if len(tool_samples) >= self._minimum_pool_size
            else self._samples["__global__"]
        )
        return samples[int.from_bytes(digest[:8], "big") % len(samples)]


class CausalDurationPredictor:
    """Calibration priors plus EWMA of already-completed executions."""

    def __init__(
        self,
        *,
        tool_medians: Mapping[str, float],
        domain_medians: Mapping[str, float],
        global_median_s: float,
        ewma_alpha: float = 0.35,
        artifact: Mapping[str, Any] | None = None,
    ) -> None:
        if not 0.0 < float(ewma_alpha) <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self._tool_medians = {
            str(key): _finite_nonnegative(value, label=f"tool median {key}")
            for key, value in tool_medians.items()
        }
        self._domain_medians = {
            str(key): _finite_nonnegative(value, label=f"domain median {key}")
            for key, value in domain_medians.items()
        }
        self._global_median_s = _finite_nonnegative(
            global_median_s, label="global duration median"
        )
        self._ewma_alpha = float(ewma_alpha)
        self._observed_ewma: dict[str, float] = {}
        self._observed_counts: dict[str, int] = defaultdict(int)
        self._artifact = dict(artifact or {})

    @staticmethod
    def _key(tool_name: str, url: str | None) -> str:
        domain = _domain(url) if tool_name == "visit" else ""
        return f"{tool_name}:{domain}" if domain else tool_name

    @classmethod
    def fit(
        cls,
        sessions: Sequence[SessionTrace],
        *,
        training_provenance: Mapping[str, Any],
        ewma_alpha: float = 0.35,
    ) -> tuple["CausalDurationPredictor", dict[str, Any]]:
        training_root_ids_sha256 = training_provenance.get("session_ids_sha256")
        if not _is_sha256(training_root_ids_sha256):
            raise ValueError("duration fit requires a calibration root-ID SHA-256")
        tool_values: dict[str, list[float]] = defaultdict(list)
        domain_values: dict[str, list[float]] = defaultdict(list)
        all_values: list[float] = []
        missing = 0
        for session in sessions:
            for event in session.events:
                if not isinstance(event, ToolCall):
                    continue
                outcome = corrected_tool_outcome(event)
                if outcome["duration_s"] is None:
                    missing += 1
                    continue
                if event.tool_name == "visit" and outcome["visit_units"]:
                    for unit in outcome["visit_units"]:
                        value = float(unit["duration_s"])
                        tool_values["visit"].append(value)
                        domain = _domain(str(unit["url"]))
                        if domain:
                            domain_values[domain].append(value)
                        all_values.append(value)
                else:
                    value = float(outcome["duration_s"])
                    tool_values[event.tool_name].append(value)
                    all_values.append(value)
        if not all_values:
            raise ValueError("calibration traces contain no corrected tool durations")
        global_median = statistics.median(all_values)
        body = {
            "schema": DURATION_SCHEMA,
            "training_role": "calibration",
            "training_provenance": dict(training_provenance),
            "training_root_ids_sha256": training_root_ids_sha256,
            "uses_evaluation_labels": False,
            "input_features": [
                "current_tool_name",
                "current_normalized_visit_domain",
                "completed_job_service_s_ewma",
            ],
            "fit_code_sha256": _fit_code_sha256(cls.fit),
            "ewma_alpha": float(ewma_alpha),
            "global_median_s": global_median,
            "tool_statistics": {
                key: {"count": len(values), "median_s": statistics.median(values)}
                for key, values in sorted(tool_values.items())
            },
            "visit_domain_statistics": {
                key: {"count": len(values), "median_s": statistics.median(values)}
                for key, values in sorted(domain_values.items())
            },
            "missing_calibration_outcomes": missing,
            "runtime_update": "completed-job EWMA only",
        }
        artifact = signed_payload(body, "artifact_sha256")
        predictor = cls.from_artifact(artifact)
        return predictor, artifact

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "CausalDurationPredictor":
        checked = validate_signed_payload(
            artifact, "artifact_sha256", label="duration predictor artifact"
        )
        if checked.get("schema") != DURATION_SCHEMA:
            raise ValueError("unsupported duration predictor schema")
        if checked.get("training_role") != "calibration":
            raise ValueError("duration predictor was not trained on calibration")
        if checked.get("uses_evaluation_labels") is not False:
            raise ValueError("duration predictor may not use evaluation labels")
        provenance = checked.get("training_provenance")
        if (
            not isinstance(provenance, Mapping)
            or not _is_sha256(checked.get("training_root_ids_sha256"))
            or checked.get("training_root_ids_sha256")
            != provenance.get("session_ids_sha256")
        ):
            raise ValueError("duration predictor training-root binding is invalid")
        if checked.get("input_features") != [
            "current_tool_name",
            "current_normalized_visit_domain",
            "completed_job_service_s_ewma",
        ]:
            raise ValueError("duration predictor input feature schema is invalid")
        if checked.get("fit_code_sha256") != _fit_code_sha256(cls.fit):
            raise ValueError("duration predictor fit-code binding is invalid")
        tool_statistics = checked.get("tool_statistics", {})
        domain_statistics = checked.get("visit_domain_statistics", {})
        if not isinstance(tool_statistics, Mapping) or not isinstance(
            domain_statistics, Mapping
        ):
            raise ValueError("invalid duration predictor statistics")
        return cls(
            tool_medians={
                str(key): float(value["median_s"])
                for key, value in tool_statistics.items()
            },
            domain_medians={
                str(key): float(value["median_s"])
                for key, value in domain_statistics.items()
            },
            global_median_s=float(checked["global_median_s"]),
            ewma_alpha=float(checked["ewma_alpha"]),
            artifact=checked,
        )

    @property
    def artifact_sha256(self) -> str | None:
        value = self._artifact.get("artifact_sha256")
        return str(value) if value else None

    def frozen_estimate(self, tool_name: str, url: str | None = None) -> DurationEstimate:
        domain = _domain(url) if tool_name == "visit" else ""
        if domain and domain in self._domain_medians:
            return DurationEstimate(self._domain_medians[domain], "calibration_domain_median")
        if tool_name in self._tool_medians:
            return DurationEstimate(self._tool_medians[tool_name], "calibration_tool_median")
        return DurationEstimate(self._global_median_s, "calibration_global_median")

    def estimate(self, tool_name: str, url: str | None = None) -> DurationEstimate:
        frozen = self.frozen_estimate(tool_name, url)
        key = self._key(tool_name, url)
        observed = self._observed_ewma.get(key)
        if observed is None and tool_name == "visit":
            observed = self._observed_ewma.get("visit")
        if observed is None:
            return frozen
        return DurationEstimate(
            0.5 * frozen.service_s + 0.5 * observed,
            f"{frozen.source}+completed_job_ewma",
        )

    def observe_completed(
        self, tool_name: str, service_s: float, url: str | None = None
    ) -> None:
        value = _finite_nonnegative(service_s, label="completed tool service")
        keys = [self._key(tool_name, url)]
        if tool_name == "visit" and keys[0] != "visit":
            keys.append("visit")
        for key in keys:
            previous = self._observed_ewma.get(key)
            self._observed_ewma[key] = (
                value
                if previous is None
                else self._ewma_alpha * value + (1.0 - self._ewma_alpha) * previous
            )
            self._observed_counts[key] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "observed_ewma_s": dict(sorted(self._observed_ewma.items())),
            "observed_counts": dict(sorted(self._observed_counts.items())),
        }


@dataclass(frozen=True)
class TailPrediction:
    next_tool_wait_s: float
    remaining_tool_wait_s: float
    next_tool_probability: float
    reliability: float


class CausalTailPredictor:
    """Call-index population estimates fitted exclusively on calibration."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        checked = validate_signed_payload(
            artifact, "artifact_sha256", label="tail predictor artifact"
        )
        if checked.get("schema") != TAIL_SCHEMA:
            raise ValueError("unsupported tail predictor schema")
        if checked.get("training_role") != "calibration":
            raise ValueError("tail predictor was not trained on calibration")
        if checked.get("uses_evaluation_labels") is not False:
            raise ValueError("tail predictor may not use evaluation labels")
        provenance = checked.get("training_provenance")
        if (
            not isinstance(provenance, Mapping)
            or not _is_sha256(checked.get("training_root_ids_sha256"))
            or checked.get("training_root_ids_sha256")
            != provenance.get("session_ids_sha256")
        ):
            raise ValueError("tail predictor training-root binding is invalid")
        if checked.get("input_features") != [
            "current_call_index",
            "completed_tool_group_waits_s",
        ]:
            raise ValueError("tail predictor input feature schema is invalid")
        if checked.get("fit_code_sha256") != _fit_code_sha256(type(self).fit):
            raise ValueError("tail predictor fit-code binding is invalid")
        self._artifact = checked
        raw = checked.get("per_call", {})
        if not isinstance(raw, Mapping):
            raise ValueError("tail predictor per_call must be an object")
        self._per_call = {int(key): dict(value) for key, value in raw.items()}
        self._reliability = min(
            1.0, max(0.0, float(checked.get("next_tool_wait_reliability", 0.0)))
        )

    @staticmethod
    def _tool_duration_for_training(call: ToolCall, fallback_s: float) -> float:
        outcome = corrected_tool_outcome(call)
        value = outcome["duration_s"]
        return fallback_s if value is None else float(value)

    @classmethod
    def _session_rows(
        cls, session: SessionTrace, fallback_s: float
    ) -> list[dict[str, float | int]]:
        events = list(session.events)
        rows: list[dict[str, float | int]] = []
        for index, event in enumerate(events):
            if not isinstance(event, LLMCall):
                continue
            cursor = index + 1
            next_wait = 0.0
            while cursor < len(events) and not isinstance(events[cursor], LLMCall):
                tool = events[cursor]
                if isinstance(tool, ToolCall):
                    next_wait += cls._tool_duration_for_training(tool, fallback_s)
                cursor += 1
            rows.append({"call_index": event.call_index, "next_wait_s": next_wait})
        suffix = 0.0
        for row in reversed(rows):
            suffix += float(row["next_wait_s"])
            row["remaining_wait_s"] = suffix
        return rows

    @staticmethod
    def _fit_rows(rows_by_session: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[int, dict[str, float]]:
        buckets: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"next": [], "remaining": [], "has_next": []}
        )
        for rows in rows_by_session.values():
            for row in rows:
                bucket = buckets[int(row["call_index"])]
                next_wait = float(row["next_wait_s"])
                bucket["next"].append(next_wait)
                bucket["remaining"].append(float(row["remaining_wait_s"]))
                bucket["has_next"].append(float(next_wait > 0.0))
        return {
            call_index: {
                "samples": float(len(values["next"])),
                "next_tool_wait_s": statistics.median(values["next"]),
                "remaining_tool_wait_s": statistics.median(values["remaining"]),
                "next_tool_probability": statistics.fmean(values["has_next"]),
            }
            for call_index, values in buckets.items()
        }

    @classmethod
    def fit(
        cls,
        sessions: Sequence[SessionTrace],
        *,
        training_provenance: Mapping[str, Any],
        duration_predictor: CausalDurationPredictor,
    ) -> tuple["CausalTailPredictor", dict[str, Any]]:
        training_root_ids_sha256 = training_provenance.get("session_ids_sha256")
        if not _is_sha256(training_root_ids_sha256):
            raise ValueError("tail fit requires a calibration root-ID SHA-256")
        fallback_s = duration_predictor.frozen_estimate("unknown").service_s
        rows_by_session = {
            session.session_id: cls._session_rows(session, fallback_s)
            for session in sessions
        }
        fitted = cls._fit_rows(rows_by_session)

        actual: list[float] = []
        model_error: list[float] = []
        baseline_error: list[float] = []
        session_ids = sorted(rows_by_session)
        for held_out in session_ids:
            training = {
                key: rows_by_session[key] for key in session_ids if key != held_out
            }
            fold = cls._fit_rows(training)
            population = [
                float(row["next_wait_s"])
                for rows in training.values()
                for row in rows
            ]
            if not population:
                continue
            baseline = statistics.median(population)
            for row in rows_by_session[held_out]:
                target = float(row["next_wait_s"])
                estimate = fold.get(int(row["call_index"]))
                predicted = float(estimate["next_tool_wait_s"]) if estimate else baseline
                actual.append(target)
                model_error.append(abs(predicted - target))
                baseline_error.append(abs(baseline - target))
        reliability = 0.0
        if len(actual) >= 2 and statistics.pvariance(actual) > 1e-12:
            model_mae = statistics.fmean(model_error)
            baseline_mae = statistics.fmean(baseline_error)
            if baseline_mae > 1e-12 and model_mae < baseline_mae:
                reliability = min(1.0, max(0.0, 1.0 - model_mae / baseline_mae))

        body = {
            "schema": TAIL_SCHEMA,
            "training_role": "calibration",
            "training_provenance": dict(training_provenance),
            "training_root_ids_sha256": training_root_ids_sha256,
            "uses_evaluation_labels": False,
            "input_features": [
                "current_call_index",
                "completed_tool_group_waits_s",
            ],
            "fit_code_sha256": _fit_code_sha256(cls.fit),
            "per_call": {
                str(key): value for key, value in sorted(fitted.items())
            },
            "next_tool_wait_reliability": reliability,
            "reliability_method": "leave-one-calibration-session-out MAE skill vs global median",
            "runtime_inputs": ["current_call_index", "completed_tool_group_waits"],
        }
        artifact = signed_payload(body, "artifact_sha256")
        return cls(artifact), artifact

    @property
    def artifact_sha256(self) -> str:
        return str(self._artifact["artifact_sha256"])

    def predict(
        self,
        *,
        current_call_index: int,
        completed_tool_group_waits_s: Sequence[float] = (),
    ) -> TailPrediction:
        row = self._per_call.get(int(current_call_index))
        if row is None:
            return TailPrediction(0.0, 0.0, 0.0, 0.0)
        next_wait = max(0.0, float(row["next_tool_wait_s"]))
        remaining = max(next_wait, float(row["remaining_tool_wait_s"]))
        observed = [
            _finite_nonnegative(value, label="completed tool group wait")
            for value in completed_tool_group_waits_s
        ]
        if observed:
            observed_mean = statistics.fmean(observed)
            weight = 1.0 - math.exp(-len(observed) / 3.0)
            base_per_group = next_wait if next_wait > 0.0 else remaining
            scale = observed_mean / max(base_per_group, 1e-9)
            scale = min(4.0, max(0.25, scale))
            adjustment = (1.0 - weight) + weight * scale
            next_wait *= adjustment
            remaining *= adjustment
        return TailPrediction(
            next_tool_wait_s=next_wait,
            remaining_tool_wait_s=remaining,
            next_tool_probability=min(
                1.0, max(0.0, float(row["next_tool_probability"]))
            ),
            reliability=self._reliability,
        )


@dataclass(frozen=True)
class StrictCandidate:
    url: str
    confidence: float
    predicted_service_s: float
    prediction_source: str

    @property
    def priority(self) -> float:
        return self.confidence * self.predicted_service_s


@dataclass
class CausalSessionState:
    predicted_output_tokens: float = 128.0
    completed_tool_group_waits_s: list[float] | None = None
    last_completed_tool_name: str = ""
    last_completed_event_index: int = -1

    def __post_init__(self) -> None:
        if self.completed_tool_group_waits_s is None:
            self.completed_tool_group_waits_s = []

    def observe_llm_completion(self, completion_tokens: int, *, alpha: float = 0.5) -> None:
        if completion_tokens <= 0:
            return
        self.predicted_output_tokens = (
            alpha * float(completion_tokens)
            + (1.0 - alpha) * self.predicted_output_tokens
        )

    def observe_tool_group(
        self, *, tool_name: str, event_index: int, exposed_wait_s: float
    ) -> None:
        assert self.completed_tool_group_waits_s is not None
        self.completed_tool_group_waits_s.append(
            _finite_nonnegative(exposed_wait_s, label="completed tool group wait")
        )
        self.last_completed_tool_name = str(tool_name)
        self.last_completed_event_index = int(event_index)


def latest_visible_tool_response(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content", "")
        if (
            message.get("role") == "user"
            and isinstance(content, str)
            and "<tool_response>" in content
        ):
            return content
    return ""


class StrictOnlinePolicy:
    """Policy API whose arguments cannot carry trace suffixes or outcomes."""

    def __init__(
        self,
        *,
        mapper: URLRankMapper,
        mapper_artifact_sha256: str,
        duration_predictor: CausalDurationPredictor,
        tail_predictor: CausalTailPredictor,
        top_k: int,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.mapper = mapper
        self.mapper_artifact_sha256 = str(mapper_artifact_sha256)
        self.duration_predictor = duration_predictor
        self.tail_predictor = tail_predictor
        self.top_k = int(top_k)
        self.predictor_artifact_sha256 = canonical_sha256(
            {
                "invocation_predictor": self.mapper_artifact_sha256,
                "tail_predictor": self.tail_predictor.artifact_sha256,
            }
        )
        self.policy_sha256 = canonical_sha256(
            {
                "policy": POLICY_NAME,
                "top_k": self.top_k,
                "predictor_artifact_sha256": self.predictor_artifact_sha256,
                "duration_predictor_artifact_sha256": (
                    self.duration_predictor.artifact_sha256
                ),
                "decision_inputs": [
                    "current_messages",
                    "current_call_index",
                    "current_prompt_tokens",
                    "current_public_max_tokens",
                    "completed_tool_service_times",
                ],
            }
        )

    def materialize_candidates(
        self,
        *,
        current_messages: Sequence[Mapping[str, Any]],
        last_completed_tool_name: str,
    ) -> tuple[StrictCandidate, ...]:
        # The tool kind is already observed.  Requiring an observed search
        # prevents arbitrary URLs in a Visit response from entering the parser's
        # legacy plain-URL fallback.
        if last_completed_tool_name != "search":
            return ()
        visible = latest_visible_tool_response(current_messages)
        if not visible:
            return ()
        predictions = self.mapper.predict(parse_search_results(visible), self.top_k)
        result: list[StrictCandidate] = []
        for prediction in predictions:
            url = str(prediction.invocation.arguments["url"])
            estimate = self.duration_predictor.estimate("visit", url)
            result.append(
                StrictCandidate(
                    url=url,
                    confidence=min(1.0, max(0.0, float(prediction.confidence))),
                    predicted_service_s=estimate.service_s,
                    prediction_source=estimate.source,
                )
            )
        return tuple(result)

    def scheduler_metadata(
        self,
        *,
        trace_id: str,
        request_index: int,
        current_call_index: int,
        prompt_tokens: int,
        max_tokens: int,
        state: CausalSessionState,
        observed_event_seq: int | None = None,
        decision_seq: int | None = None,
    ) -> dict[str, Any]:
        history = state.completed_tool_group_waits_s or []
        tail = self.tail_predictor.predict(
            current_call_index=current_call_index,
            completed_tool_group_waits_s=history,
        )
        observed_seq = (
            max(0, int(state.last_completed_event_index) + 1)
            if observed_event_seq is None
            else max(0, int(observed_event_seq))
        )
        decided_seq = observed_seq + 1 if decision_seq is None else int(decision_seq)
        if decided_seq < observed_seq:
            raise ValueError("decision sequence precedes observed event sequence")
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
            "tool_hit_probability_hat": (
                tail.next_tool_probability * tail.reliability
            ),
            "remaining_tool_wait_s_hat": tail.remaining_tool_wait_s,
            "ms": POLICY_NAME,
            "decision_seq": decided_seq,
            "observed_event_seq": observed_seq,
            "policy_sha256": self.policy_sha256,
            "predictor_artifact_sha256": self.predictor_artifact_sha256,
            "duration_predictor_artifact_sha256": (
                self.duration_predictor.artifact_sha256
            ),
        }


class CausalTraceCursor:
    """Reveal current input, then authoritative calls, in causal order."""

    def __init__(self, steps: Sequence[Mapping[str, Any]]) -> None:
        self.__steps = tuple(dict(step) for step in steps)
        self.__index = 0
        self.__llm_completed = False

    @property
    def done(self) -> bool:
        return self.__index >= len(self.__steps)

    @property
    def request_index(self) -> int:
        return self.__index

    def current_request(self) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("trace cursor is exhausted")
        if self.__llm_completed:
            raise RuntimeError("authoritative tools must be consumed before advancing")
        return json.loads(json.dumps(self.__steps[self.__index]["request"]))

    def mark_llm_completed(self) -> None:
        if self.done or self.__llm_completed:
            raise RuntimeError("invalid LLM completion transition")
        self.__llm_completed = True

    def reveal_authoritative_tools(self) -> tuple[dict[str, Any], ...]:
        if self.done or not self.__llm_completed:
            raise RuntimeError("authoritative tools are hidden until the LLM completes")
        return tuple(
            json.loads(json.dumps(row))
            for row in self.__steps[self.__index].get("tools_after", [])
        )

    def advance(self) -> None:
        if self.done or not self.__llm_completed:
            raise RuntimeError("cannot advance before LLM completion")
        self.__index += 1
        self.__llm_completed = False


@dataclass(frozen=True)
class ToolExecutionObservation:
    tool_name: str
    event_index: int
    exposed_wait_s: float
    service_s: float
    saved_service_s: float
    visit_results: tuple[VisitResult, ...] = ()


class SealedTraceToolExecutor:
    """The sole reader of a policy-independent, pre-sealed service surface."""

    def __init__(
        self,
        *,
        sealed_outcomes: Mapping[str, Mapping[str, Any]],
        service_clock: CalibrationHashedServiceClock,
        duration_predictor: CausalDurationPredictor,
        visit_pool: AsyncPreemptibleVisitPool,
        clock: Any = time.monotonic,
    ) -> None:
        self.__outcomes = {
            str(key): dict(value) for key, value in sealed_outcomes.items()
        }
        self.__service_clock = service_clock
        self.__duration_predictor = duration_predictor
        self.__visit_pool = visit_pool
        self.__clock = clock
        self.__non_visit_direct_demand_s = 0.0

    def __sealed_tool_service(
        self,
        *,
        session_id: str,
        tool_name: str,
        tool_arguments: Mapping[str, Any],
    ) -> float:
        del session_id
        return self.__service_clock.service_s(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )

    async def speculate(
        self,
        *,
        session_id: str,
        candidates: Sequence[StrictCandidate],
        after_event_index: int,
        decision_id: str,
    ) -> tuple[bool, ...]:
        rows = [
            (
                session_id,
                normalize_url(candidate.url),
                self.__sealed_tool_service(
                    session_id=session_id,
                    tool_name="visit",
                    tool_arguments={"url": normalize_url(candidate.url)},
                ),
                candidate.priority,
                decision_id,
            )
            for candidate in candidates
        ]
        return await self.__visit_pool.speculate_batch(rows)

    def __outcome(self, descriptor: Mapping[str, Any]) -> dict[str, Any]:
        outcome_id = str(descriptor.get("outcome_id", ""))
        outcome = self.__outcomes.get(outcome_id)
        if outcome is None:
            raise ValueError(f"sealed tool outcome is missing: {outcome_id}")
        if (
            str(outcome.get("tool_name")) != str(descriptor.get("tool_name"))
            or int(outcome.get("event_index", -1))
            != int(descriptor.get("event_index", -2))
        ):
            raise ValueError(f"sealed cursor/outcome descriptor mismatch: {outcome_id}")
        return outcome

    async def execute_authoritative(
        self, *, session_id: str, descriptor: Mapping[str, Any]
    ) -> ToolExecutionObservation:
        outcome = self.__outcome(descriptor)
        tool_name = str(descriptor["tool_name"])
        event_index = int(descriptor["event_index"])
        started = self.__clock()
        visit_results: list[VisitResult] = []
        saved = 0.0
        service = 0.0
        if tool_name == "visit":
            public_urls = descriptor.get("tool_args", {}).get("url")
            urls = (
                [public_urls]
                if isinstance(public_urls, str)
                else list(public_urls or [])
            )
            sealed_units = list(outcome.get("visit_units", []))
            if [str(row["url"]) for row in sealed_units] != [str(url) for url in urls]:
                raise ValueError(f"sealed cursor/outcome visit URL mismatch: {descriptor['outcome_id']}")
            for unit in sealed_units:
                url = str(unit["url"])
                result = await self.__visit_pool.authoritative(
                    session_id=session_id,
                    url=normalize_url(url),
                    duration_s=self.__sealed_tool_service(
                        session_id=session_id,
                        tool_name="visit",
                        tool_arguments={"url": url},
                    ),
                )
                visit_results.append(result)
                service += result.service_s
                saved += result.saved_service_s
                self.__duration_predictor.observe_completed(
                    "visit", result.service_s, url
                )
        else:
            tool_arguments = descriptor.get("tool_args", {})
            if not isinstance(tool_arguments, Mapping):
                raise ValueError("sealed authority tool arguments must be an object")
            duration_s = self.__sealed_tool_service(
                session_id=session_id,
                tool_name=tool_name,
                tool_arguments=tool_arguments,
            )
            await asyncio.sleep(duration_s)
            service = max(0.0, self.__clock() - started)
            self.__non_visit_direct_demand_s += service
            self.__duration_predictor.observe_completed(tool_name, service)
        exposed = max(0.0, self.__clock() - started)
        return ToolExecutionObservation(
            tool_name=tool_name,
            event_index=event_index,
            exposed_wait_s=exposed,
            service_s=service,
            saved_service_s=saved,
            visit_results=tuple(visit_results),
        )

    async def reveal_prediction_outcome(
        self,
        *,
        decision_id: str,
        authoritative_descriptors: Sequence[Mapping[str, Any]],
    ) -> None:
        """Resolve queued/running wrong work after, never before, reveal."""

        authoritative_urls: set[str] = set()
        for descriptor in authoritative_descriptors:
            if str(descriptor.get("tool_name")) != "visit":
                continue
            arguments = descriptor.get("tool_args", {})
            if not isinstance(arguments, Mapping):
                continue
            raw_urls = arguments.get("url")
            urls = [raw_urls] if isinstance(raw_urls, str) else list(raw_urls or [])
            authoritative_urls.update(normalize_url(str(url)) for url in urls)
        await self.__visit_pool.resolve_prediction(
            decision_id=decision_id,
            authoritative_urls=authoritative_urls,
        )

    def expire_prediction_window(self, decision_id: str) -> None:
        """Prevent not-yet-started work from crossing the LLM completion edge."""

        self.__visit_pool.expire_queued_decision(decision_id)

    async def close_session(self, session_id: str) -> None:
        await self.__visit_pool.close_session(session_id)

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.__visit_pool.snapshot()
        non_visit = self.__non_visit_direct_demand_s
        snapshot["non_visit_direct_demand_resource_s"] = non_visit
        snapshot["direct_demand_resource_s"] += non_visit
        snapshot["demand_resource_s"] += non_visit
        snapshot["ledger_demand_service_s"] += non_visit
        snapshot["total_worker_service_s"] += non_visit
        snapshot["total_worker_occupancy_s"] += non_visit
        return snapshot

    async def close(self) -> None:
        await self.__visit_pool.close()


def serialize_observation(observation: ToolExecutionObservation) -> dict[str, Any]:
    # dataclasses.asdict recursively converts VisitResult children as well.
    return asdict(observation)
