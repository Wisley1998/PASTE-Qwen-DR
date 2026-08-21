"""Causal session-tail estimates for trace replay scheduler metadata.

The predictor is fitted only from a separate calibration workload.  At replay
time it accepts the current call index and already-observed tool waits; it has
no API through which the replay trace's future requests can be supplied.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


_SCHEMA_VERSION = 1
_HISTORY_WEIGHT_TAU = 3.0
_VARIANCE_EPSILON = 1e-12
_LOSS_EPSILON = 1e-12


@dataclass(frozen=True)
class OnlineSessionPrediction:
    """Predicted scheduler signals after the current LLM call."""

    next_tool_wait_s: float
    remaining_tool_wait_s: float
    remaining_calls: int


@dataclass(frozen=True)
class _CallIndexEstimate:
    samples: int
    next_tool_wait_s: float
    remaining_tool_wait_s: float
    remaining_calls: float


class OnlineSessionPredictor:
    """Small, serializable population predictor indexed by LLM call index.

    Calibration computes the median next wait, total remaining wait, and number
    of remaining calls for every observed call index.  Medians keep a single
    pathological trace from dominating all replay sessions.  Past waits from
    the live session provide a gradually increasing task-specific adjustment.
    Session length is always estimated from calibration and never from the
    replay trace being scheduled.
    """

    def __init__(
        self,
        per_call: Mapping[int, _CallIndexEstimate],
        *,
        next_tool_wait_reliability: float = 1.0,
    ) -> None:
        self._per_call = {int(index): estimate for index, estimate in per_call.items()}
        reliability = float(next_tool_wait_reliability)
        self._next_tool_wait_reliability = (
            min(1.0, max(0.0, reliability)) if math.isfinite(reliability) else 0.0
        )

    @property
    def next_tool_wait_reliability(self) -> float:
        """Calibration-only out-of-sample skill in the closed interval [0, 1]."""

        return self._next_tool_wait_reliability

    @classmethod
    def from_workload(cls, workload_path: str | Path) -> "OnlineSessionPredictor":
        workload = json.loads(Path(workload_path).read_text(encoding="utf-8"))
        return cls.from_workload_data(workload)

    @classmethod
    def from_workload_data(cls, workload: Mapping[str, Any]) -> "OnlineSessionPredictor":
        traces = list(workload.get("traces", []))
        per_call = cls._fit_per_call(traces)
        reliability = cls._backtest_next_tool_wait_reliability(traces)
        return cls(
            per_call,
            next_tool_wait_reliability=reliability,
        )

    @staticmethod
    def _fit_per_call(
        traces: Sequence[Mapping[str, Any]],
    ) -> Dict[int, _CallIndexEstimate]:
        observations: Dict[int, Dict[str, list[float]]] = {}
        for trace in traces:
            requests = list(trace.get("requests", []))
            for request_index, request in enumerate(requests):
                call_index = int(request.get("call_index", request_index))
                future = requests[request_index + 1 :]
                next_wait_s = (
                    max(0.0, float(future[0].get("wait_after_prev_s", 0.0)))
                    if future
                    else 0.0
                )
                remaining_wait_s = sum(
                    max(0.0, float(item.get("wait_after_prev_s", 0.0)))
                    for item in future
                )
                bucket = observations.setdefault(
                    call_index,
                    {"next": [], "remaining_wait": [], "remaining_calls": []},
                )
                bucket["next"].append(next_wait_s)
                bucket["remaining_wait"].append(remaining_wait_s)
                bucket["remaining_calls"].append(float(len(future)))

        per_call: Dict[int, _CallIndexEstimate] = {}
        for call_index, values in observations.items():
            per_call[call_index] = _CallIndexEstimate(
                samples=len(values["remaining_calls"]),
                next_tool_wait_s=statistics.median(values["next"]),
                remaining_tool_wait_s=statistics.median(values["remaining_wait"]),
                remaining_calls=statistics.median(values["remaining_calls"]),
            )
        return per_call

    @staticmethod
    def _session_key(trace: Mapping[str, Any], trace_index: int) -> str:
        """Group duplicated calibration traces into one held-out source session."""

        for key in ("source_trace", "trace_id"):
            value = trace.get(key)
            if value is not None and str(value).strip():
                return f"{key}:{value}"
        return f"trace_index:{trace_index}"

    @classmethod
    def _backtest_next_tool_wait_reliability(
        cls,
        traces: Sequence[Mapping[str, Any]],
    ) -> float:
        """Return leave-one-session-out MAE skill over a call-index median.

        Each fold fits exclusively on other calibration source sessions.  The
        candidate prediction may use waits already observed in the held-out
        session, matching the online API, while the baseline uses only the
        fold's call-index median.  Only requests with a following tool wait are
        scored.  This avoids terminal zeroes manufacturing apparent skill.

        Reliability is ``1 - candidate_mae / baseline_mae``, clipped to [0, 1].
        A constant target, an exact/degenerate baseline, or no positive skill
        fails closed to zero.
        """

        grouped: Dict[str, list[Mapping[str, Any]]] = {}
        for trace_index, trace in enumerate(traces):
            grouped.setdefault(cls._session_key(trace, trace_index), []).append(trace)
        if len(grouped) < 2:
            return 0.0

        actual_waits: list[float] = []
        candidate_errors: list[float] = []
        baseline_errors: list[float] = []
        groups = list(grouped.items())
        for held_out_key, held_out_traces in groups:
            training_traces = [
                trace
                for session_key, session_traces in groups
                if session_key != held_out_key
                for trace in session_traces
            ]
            fold_predictor = cls(cls._fit_per_call(training_traces))

            for trace in held_out_traces:
                requests = list(trace.get("requests", []))
                past_tool_waits_s: list[float] = []
                for request_index, request in enumerate(requests[:-1]):
                    if int(request.get("call_index", request_index)) > 0:
                        past_tool_waits_s.append(
                            max(0.0, float(request.get("wait_after_prev_s", 0.0)))
                        )

                    call_index = int(request.get("call_index", request_index))
                    actual_wait_s = max(
                        0.0,
                        float(requests[request_index + 1].get("wait_after_prev_s", 0.0)),
                    )
                    candidate_wait_s = fold_predictor.predict(
                        current_call_index=call_index,
                        past_tool_waits_s=past_tool_waits_s,
                    ).next_tool_wait_s
                    estimate = fold_predictor._per_call.get(call_index)
                    baseline_wait_s = (
                        max(0.0, estimate.next_tool_wait_s)
                        if estimate is not None
                        else 0.0
                    )

                    actual_waits.append(actual_wait_s)
                    candidate_errors.append(abs(candidate_wait_s - actual_wait_s))
                    baseline_errors.append(abs(baseline_wait_s - actual_wait_s))

        if len(actual_waits) < 2 or statistics.pvariance(actual_waits) <= _VARIANCE_EPSILON:
            return 0.0

        candidate_mae = statistics.fmean(candidate_errors)
        baseline_mae = statistics.fmean(baseline_errors)
        if baseline_mae <= _LOSS_EPSILON or candidate_mae >= baseline_mae:
            return 0.0
        return min(1.0, max(0.0, 1.0 - candidate_mae / baseline_mae))

    def predict(
        self,
        *,
        current_call_index: int,
        past_tool_waits_s: Sequence[float] = (),
    ) -> OnlineSessionPrediction:
        """Return a causal estimate using calibration plus observed history.

        ``past_tool_waits_s`` contains waits completed before the current LLM
        request, including the wait that led into the current call.  It must not
        contain a wait following the current request.
        """

        estimate = self._per_call.get(int(current_call_index))
        if estimate is None:
            return OnlineSessionPrediction(0.0, 0.0, 0)

        remaining_calls = max(0, int(math.floor(estimate.remaining_calls + 0.5)))
        if remaining_calls == 0:
            return OnlineSessionPrediction(0.0, 0.0, 0)

        next_wait_s = max(0.0, estimate.next_tool_wait_s)
        remaining_wait_s = max(next_wait_s, estimate.remaining_tool_wait_s)

        observed = [max(0.0, float(wait_s)) for wait_s in past_tool_waits_s]
        if observed:
            history_mean_s = statistics.fmean(observed)
            history_weight = 1.0 - math.exp(-len(observed) / _HISTORY_WEIGHT_TAU)
            population_per_call_s = remaining_wait_s / max(
                estimate.remaining_calls,
                1.0,
            )
            next_wait_s = (
                history_weight * history_mean_s
                + (1.0 - history_weight) * next_wait_s
            )
            adjusted_per_call_s = (
                history_weight * history_mean_s
                + (1.0 - history_weight) * population_per_call_s
            )
            remaining_wait_s = max(
                next_wait_s,
                adjusted_per_call_s * max(estimate.remaining_calls, 1.0),
            )

        return OnlineSessionPrediction(
            next_tool_wait_s=max(0.0, next_wait_s),
            remaining_tool_wait_s=max(0.0, remaining_wait_s),
            remaining_calls=remaining_calls,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "predictor_type": "online_session_by_call_index",
            "next_tool_wait_reliability": self.next_tool_wait_reliability,
            "per_call": {
                str(index): asdict(estimate)
                for index, estimate in sorted(self._per_call.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OnlineSessionPredictor":
        if int(payload.get("schema_version", -1)) != _SCHEMA_VERSION:
            raise ValueError("unsupported online session predictor schema")
        if payload.get("predictor_type") != "online_session_by_call_index":
            raise ValueError("invalid online session predictor type")
        per_call = {
            int(index): _CallIndexEstimate(
                samples=int(values["samples"]),
                next_tool_wait_s=float(values["next_tool_wait_s"]),
                remaining_tool_wait_s=float(values["remaining_tool_wait_s"]),
                remaining_calls=float(values["remaining_calls"]),
            )
            for index, values in payload.get("per_call", {}).items()
        }
        # Predictor artifacts written before reliability gating do not contain
        # this field.  Preserve their historical scheduling behavior.
        reliability = payload.get("next_tool_wait_reliability", 1.0)
        return cls(per_call, next_tool_wait_reliability=float(reliability))

    def save(self, output_path: str | Path) -> None:
        Path(output_path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, predictor_path: str | Path) -> "OnlineSessionPredictor":
        payload = json.loads(Path(predictor_path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
