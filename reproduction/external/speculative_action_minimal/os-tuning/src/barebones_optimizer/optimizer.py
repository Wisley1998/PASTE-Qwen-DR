from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .benchmark import WindowExecution
from .config import ReplicationConfig
from .parameter_manager import SchedulerParameterManager
from .tuners import GeminiTuner
from .tuners.llm import PromptContext

logger = logging.getLogger(__name__)
ACTOR_MIN_RESPONSE_SECONDS = 10.0
PROGRESS_REPORT_INTERVAL_WINDOWS = 10


class ReplicationOptimizer:
    def __init__(
        self,
        *,
        config: ReplicationConfig,
        benchmark,
        parameter_manager: SchedulerParameterManager,
        tuners: Dict[str, GeminiTuner],
        run_dir: Path,
    ):
        self.config = config
        self.benchmark = benchmark
        self.parameter_manager = parameter_manager
        self.tuners = tuners
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[Dict[str, Any]] = []
        self.requests: list[Dict[str, Any]] = []
        self.best_window: Optional[Dict[str, Any]] = None
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self.current_parameters: Dict[str, int] = {}
        self.default_parameters: Dict[str, int] = {}
        self.baseline_index = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._request_counter = 0
        self.started_at = time.time()

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
        self._emit_run_start(initial_min)

        try:
            for iteration in range(1, self.config.iterations + 1):
                window_start_parameters = deepcopy(self.current_parameters)
                execution = self._execute_window_with_monitoring(iteration)
                window_entry = self._record_window(iteration, execution, window_start_parameters)
                if self._is_better(window_entry):
                    self.best_window = deepcopy(window_entry)
                self._schedule_requests()
                self._emit_progress(iteration)
        finally:
            self._finalize()

        history_path = self._save_history(initial_apply_time)
        self._emit_run_complete(history_path)
        return {"history_file": str(history_path)}

    def _execute_window_with_monitoring(self, iteration: int) -> WindowExecution:
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
            self._poll_requests()
            time.sleep(0.01)

        worker.join()
        self._poll_requests()

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["execution"]

    def _record_window(
        self,
        iteration: int,
        execution: WindowExecution,
        window_start_parameters: Dict[str, int],
    ) -> Dict[str, Any]:
        window_entry = {
            "iteration": iteration,
            "timestamp": execution.window_end_time,
            "window_start_time": execution.window_start_time,
            "window_end_time": execution.window_end_time,
            "parameters_at_window_start": window_start_parameters,
            "parameters": deepcopy(self.current_parameters),
            "benchmark_metrics": execution.benchmark_metrics.to_dict(),
            "system_metrics": execution.system_metrics.to_dict(),
            "log_file": execution.log_file,
        }
        self.history.append(window_entry)
        return window_entry

    def _is_better(self, window_entry: Dict[str, Any]) -> bool:
        candidate = float(window_entry["benchmark_metrics"]["latency_p95"])
        if self.best_window is None:
            return True
        return candidate < float(self.best_window["benchmark_metrics"]["latency_p95"])

    def _start_request(self, role: str) -> None:
        if role in self.pending_requests:
            return

        self._request_counter += 1
        request_type = role
        call_timestamp = time.time()
        call_number = len(self.history) + 1
        tuner = self.tuners[role]
        prompt_context = PromptContext(
            call_number=call_number,
            current_parameters=deepcopy(self.current_parameters),
            best_window=deepcopy(self.best_window),
            history=deepcopy(self.history),
            baseline_index=self.baseline_index,
            role=role,
            mode=self.config.mode,
        )
        request_entry = {
            "request_id": f"{role}_{self._request_counter:04d}",
            "request_type": request_type,
            "role": role,
            "model": tuner.model_name,
            "call_timestamp": call_timestamp,
            "response_timestamp": None,
            "apply_timestamp": None,
            "earliest_apply_timestamp": (
                call_timestamp + ACTOR_MIN_RESPONSE_SECONDS if role == "actor" else call_timestamp
            ),
            "iteration_context": call_number,
            "proposed_parameters": None,
            "justification": None,
            "token_metrics": None,
            "applied": False,
            "error": None,
        }
        future = self.executor.submit(tuner.suggest, prompt_context)
        self.pending_requests[role] = {"future": future, "request": request_entry}

    def _poll_requests(self) -> None:
        for role in list(self.pending_requests):
            pending = self.pending_requests[role]
            future = pending["future"]
            if not future.done():
                continue
            request_entry = pending["request"]
            if time.time() < float(request_entry["earliest_apply_timestamp"]):
                continue
            response_timestamp = time.time()
            try:
                response = future.result()
                request_entry["response_timestamp"] = response_timestamp
                request_entry["proposed_parameters"] = deepcopy(response.proposed_parameters)
                request_entry["justification"] = response.justification
                request_entry["token_metrics"] = deepcopy(response.token_metrics)

                proposed_min = response.proposed_parameters.get("min_granularity_ns")
                if proposed_min is not None:
                    apply_time = self.parameter_manager.apply(
                        min_granularity_ns=int(proposed_min),
                    )
                    self.current_parameters["min_granularity_ns"] = int(proposed_min)
                    request_entry["apply_timestamp"] = apply_time
                    request_entry["applied"] = True
                    latest_p95 = None
                    if self.history:
                        latest_p95 = float(self.history[-1]["benchmark_metrics"]["latency_p95"])
                    self._emit_apply_event(role, int(proposed_min), apply_time, latest_p95)
                    if role == "actor" and self.config.mode == "dual":
                        self.baseline_index = len(self.history)
            except BaseException as exc:  # pragma: no cover - depends on API/network
                request_entry["response_timestamp"] = response_timestamp
                request_entry["error"] = str(exc)
                self._emit(
                    f"[t={self._elapsed_s(response_timestamp):5.1f}s] {role} request failed: {exc}"
                )

            self.requests.append(request_entry)
            self.pending_requests.pop(role, None)

    def _schedule_requests(self) -> None:
        if not self.history:
            return
        if self.config.mode == "dual":
            if "speculator" not in self.pending_requests:
                self._start_request("speculator")
            if "actor" not in self.pending_requests:
                self._start_request("actor")
            return
        role = self.config.active_roles()[0]
        if role not in self.pending_requests:
            self._start_request(role)

    def _finalize(self) -> None:
        shutdown_time = time.time()
        for pending in self.pending_requests.values():
            pending["future"].cancel()
            request_entry = pending["request"]
            request_entry["error"] = request_entry["error"] or "cancelled_at_shutdown"
            request_entry["response_timestamp"] = request_entry["response_timestamp"] or shutdown_time
            self.requests.append(request_entry)
        self.pending_requests.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.parameter_manager.restore(self.default_parameters)

    def _save_history(self, initial_apply_time: float) -> Path:
        history_path = self.run_dir / "history.json"
        ended_at = time.time()
        payload = {
            "schema_version": 1,
            "setup_mode": self.config.mode,
            "run_kind": self.config.run_kind,
            "benchmark": self.config.benchmark,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "started_at_iso": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "ended_at_iso": datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
            "run_duration_s": self.config.run_duration_s,
            "window_duration_s": self.config.window_duration_s,
            "iterations": self.config.iterations,
            "models": {
                "actor": self.config.actor_model if self.config.mode in {"actor_only", "dual"} else None,
                "speculator": self.config.speculator_model if self.config.mode in {"speculator_only", "dual"} else None,
            },
            "latency_ns": self.config.latency_ns,
            "min_granularity_range_ns": list(self.config.min_granularity_range_ns),
            "os_defaults": deepcopy(self.default_parameters),
            "initial_parameters": {
                "latency_ns": self.config.latency_ns,
                "min_granularity_ns": self.config.initial_min_granularity_ns or self.default_parameters["min_granularity_ns"],
            },
            "initial_apply_timestamp": initial_apply_time,
            "config": self.config.to_dict(),
            "best_window": deepcopy(self.best_window),
            "history": deepcopy(self.history),
            "requests": deepcopy(self.requests),
        }
        history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return history_path

    def _emit(self, message: str) -> None:
        print(message, flush=True)

    def _elapsed_s(self, timestamp: Optional[float] = None) -> float:
        reference = timestamp if timestamp is not None else time.time()
        return max(0.0, float(reference) - float(self.started_at))

    @staticmethod
    def _format_min_granularity(value_ns: int) -> str:
        return f"{value_ns:,} ns ({value_ns / 1_000_000.0:.3f} ms)"

    def _emit_run_start(self, initial_min_granularity_ns: int) -> None:
        models = []
        for role in self.config.active_roles():
            models.append(f"{role}={self.config.model_for_role(role)}")
        self._emit("Run start")
        self._emit(
            f"  mode: {self.config.mode} | kind: {self.config.run_kind} | benchmark: {self.config.benchmark}"
        )
        self._emit(
            f"  duration: {self.config.run_duration_s}s total ({self.config.iterations} windows x {self.config.window_duration_s}s)"
        )
        self._emit(
            "  scheduler: "
            f"latency_ns={self.config.latency_ns:,}, "
            f"min_granularity_ns=[{self.config.min_granularity_range_ns[0]:,}, {self.config.min_granularity_range_ns[1]:,}]"
        )
        self._emit(f"  initial min_granularity_ns: {self._format_min_granularity(initial_min_granularity_ns)}")
        self._emit(f"  models: {', '.join(models)}")
        self._emit(f"  run dir: {self.run_dir}")

    def _emit_apply_event(
        self,
        role: str,
        proposed_min_granularity_ns: int,
        apply_time: float,
        latest_p95_ms: Optional[float],
    ) -> None:
        p95_text = f"{latest_p95_ms:.2f} ms" if latest_p95_ms is not None else "n/a"
        self._emit(
            f"[t={self._elapsed_s(apply_time):5.1f}s] {role} applied "
            f"min_granularity_ns={self._format_min_granularity(proposed_min_granularity_ns)} | latest p95={p95_text}"
        )

    def _emit_progress(self, iteration: int) -> None:
        if iteration != self.config.iterations and iteration % PROGRESS_REPORT_INTERVAL_WINDOWS != 0:
            return
        if not self.history:
            return
        best = self.best_window or self.history[-1]
        pending = ",".join(sorted(self.pending_requests)) or "-"
        self._emit(
            f"[t={self._elapsed_s(self.history[-1]['window_end_time']):5.1f}s] "
            f"progress iter {iteration}/{self.config.iterations} | "
            f"current={self._format_min_granularity(self.current_parameters['min_granularity_ns'])} | "
            f"best p95={float(best['benchmark_metrics']['latency_p95']):.2f} ms @ iter {best['iteration']} | "
            f"pending={pending}"
        )

    def _emit_run_complete(self, history_path: Path) -> None:
        request_counts = Counter(request["request_type"] for request in self.requests if request.get("request_type"))
        counts_text = ", ".join(f"{name}={count}" for name, count in sorted(request_counts.items())) or "none"
        self._emit("Run complete")
        if self.best_window is not None:
            self._emit(
                "  best: "
                f"iter {self.best_window['iteration']} at t={self._elapsed_s(self.best_window['window_end_time']):.1f}s -> "
                f"p95={float(self.best_window['benchmark_metrics']['latency_p95']):.2f} ms, "
                f"min_granularity_ns={self._format_min_granularity(int(self.best_window['parameters']['min_granularity_ns']))}"
            )
        self._emit(
            f"  final min_granularity_ns: {self._format_min_granularity(self.current_parameters['min_granularity_ns'])}"
        )
        self._emit(f"  requests: {counts_text}")
        self._emit(f"  history: {history_path}")
