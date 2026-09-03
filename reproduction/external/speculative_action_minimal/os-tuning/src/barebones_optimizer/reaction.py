from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .benchmark import WindowExecution
from .config import ReplicationConfig
from .parameter_manager import SchedulerParameterManager


def actor_replay_schedule(history_data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    start_time = float(history_data.get("started_at") or history_data["initial_apply_timestamp"])
    events: List[Dict[str, Any]] = []
    for request in history_data.get("requests", []):
        if request.get("request_type") != "actor":
            continue
        if not request.get("applied") or request.get("apply_timestamp") is None:
            continue
        proposed = request.get("proposed_parameters") or {}
        if "min_granularity_ns" not in proposed:
            continue
        events.append(
            {
                "request_id": request.get("request_id"),
                "request_type": "actor_replay",
                "role": "actor",
                "model": request.get("model"),
                "apply_offset_s": max(0.0, float(request["apply_timestamp"]) - start_time),
                "min_granularity_ns": int(proposed["min_granularity_ns"]),
                "source_call_timestamp": request.get("call_timestamp"),
                "source_response_timestamp": request.get("response_timestamp"),
                "source_apply_timestamp": request.get("apply_timestamp"),
            }
        )
    events.sort(key=lambda item: item["apply_offset_s"])
    return events


class ReactionReplayRunner:
    def __init__(
        self,
        *,
        config: ReplicationConfig,
        benchmark,
        parameter_manager: SchedulerParameterManager,
        run_dir: Path,
        setup_mode: str,
        request_type: Optional[str],
        model_name: Optional[str],
        scheduled_events: Iterable[Mapping[str, Any]],
        source_history: Optional[str] = None,
    ):
        self.config = config
        self.benchmark = benchmark
        self.parameter_manager = parameter_manager
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.setup_mode = setup_mode
        self.request_type = request_type
        self.model_name = model_name
        self.scheduled_events = [dict(event) for event in scheduled_events]
        self.source_history = source_history
        self.history: List[Dict[str, Any]] = []
        self.requests: List[Dict[str, Any]] = []
        self.default_parameters: Dict[str, int] = {}
        self.current_parameters: Dict[str, int] = {}
        self.started_at = 0.0

    def run(self) -> Dict[str, Any]:
        self.default_parameters = self.parameter_manager.snapshot()
        initial_min = self.config.initial_min_granularity_ns or self.default_parameters["min_granularity_ns"]
        self.current_parameters = {
            "latency_ns": self.config.latency_ns,
            "min_granularity_ns": initial_min,
        }
        initial_apply_time = self.parameter_manager.apply(
            latency_ns=self.config.latency_ns,
            min_granularity_ns=initial_min,
        )
        self.started_at = initial_apply_time

        pending_events = list(self.scheduled_events)
        try:
            for iteration in range(1, self.config.iterations + 1):
                window_start_parameters = deepcopy(self.current_parameters)
                execution = self._execute_window_with_schedule(iteration, pending_events)
                self.history.append(
                    {
                        "iteration": iteration,
                        "timestamp": execution.window_end_time,
                        "window_start_time": execution.window_start_time,
                        "window_end_time": execution.window_end_time,
                        "parameters": deepcopy(self.current_parameters),
                        "parameters_at_window_start": window_start_parameters,
                        "benchmark_metrics": execution.benchmark_metrics.to_dict(),
                        "system_metrics": execution.system_metrics.to_dict(),
                        "log_file": execution.log_file,
                    }
                )
        finally:
            self.parameter_manager.restore(self.default_parameters)

        history_path = self._save_history(initial_apply_time)
        return {"history_file": str(history_path)}

    def _execute_window_with_schedule(
        self,
        iteration: int,
        pending_events: List[Dict[str, Any]],
    ) -> WindowExecution:
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, BaseException] = {}

        def run_window() -> None:
            try:
                result_holder["execution"] = self.benchmark.execute_window(iteration, self.config.window_duration_s)
            except BaseException as exc:  # pragma: no cover - surfaced in main thread
                error_holder["error"] = exc

        worker = threading.Thread(target=run_window, daemon=True)
        worker.start()

        while worker.is_alive():
            self._apply_due_events(pending_events)
            time.sleep(0.01)

        worker.join()
        self._apply_due_events(pending_events)

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["execution"]

    def _apply_due_events(self, pending_events: List[Dict[str, Any]]) -> None:
        elapsed = time.time() - self.started_at
        while pending_events and elapsed >= float(pending_events[0]["apply_offset_s"]):
            event = pending_events.pop(0)
            apply_time = self.parameter_manager.apply(
                min_granularity_ns=int(event["min_granularity_ns"]),
            )
            self.current_parameters["min_granularity_ns"] = int(event["min_granularity_ns"])
            self.requests.append(
                {
                    "request_id": event.get("request_id") or f"{self.request_type}_{len(self.requests) + 1:04d}",
                    "request_type": self.request_type,
                    "role": "actor" if self.request_type else None,
                    "model": self.model_name,
                    "call_timestamp": event.get("source_call_timestamp"),
                    "response_timestamp": event.get("source_response_timestamp"),
                    "apply_timestamp": apply_time,
                    "scheduled_apply_offset_s": float(event["apply_offset_s"]),
                    "proposed_parameters": {"min_granularity_ns": int(event["min_granularity_ns"])},
                    "justification": "Replayed actor action from dual-loop trace." if self.request_type else None,
                    "token_metrics": None,
                    "applied": True,
                    "error": None,
                }
            )
            elapsed = time.time() - self.started_at

    def _save_history(self, initial_apply_time: float) -> Path:
        history_path = self.run_dir / "history.json"
        ended_at = time.time()
        payload = {
            "schema_version": 1,
            "setup_mode": self.setup_mode,
            "run_kind": self.config.run_kind,
            "benchmark": self.config.benchmark,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "started_at_iso": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "ended_at_iso": datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
            "run_duration_s": self.config.run_duration_s,
            "window_duration_s": self.config.window_duration_s,
            "iterations": self.config.iterations,
            "models": {"actor": self.model_name, "speculator": None},
            "latency_ns": self.config.latency_ns,
            "min_granularity_range_ns": list(self.config.min_granularity_range_ns),
            "os_defaults": deepcopy(self.default_parameters),
            "initial_parameters": {
                "latency_ns": self.config.latency_ns,
                "min_granularity_ns": self.config.initial_min_granularity_ns or self.default_parameters["min_granularity_ns"],
            },
            "initial_apply_timestamp": initial_apply_time,
            "config": self.config.to_dict(),
            "best_window": min(
                self.history,
                key=lambda item: float(item["benchmark_metrics"]["latency_p95"]),
                default=None,
            ),
            "history": deepcopy(self.history),
            "requests": deepcopy(self.requests),
            "source_history": self.source_history,
        }
        history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return history_path
