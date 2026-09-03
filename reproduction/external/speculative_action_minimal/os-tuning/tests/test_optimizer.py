from __future__ import annotations

import json
import time
from pathlib import Path

import barebones_optimizer.optimizer as optimizer_module
from barebones_optimizer.benchmark import BenchmarkMetrics, SystemMetrics, WindowExecution
from barebones_optimizer.config import ReplicationConfig
from barebones_optimizer.optimizer import ReplicationOptimizer
from barebones_optimizer.parameter_manager import SchedulerParameterManager
from barebones_optimizer.tuners.base import TunerResponse


class FakeBenchmark:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def execute_window(self, iteration: int, duration_s: int) -> WindowExecution:
        start = time.time()
        time.sleep(0.02)
        end = time.time()
        return WindowExecution(
            benchmark_metrics=BenchmarkMetrics(
                throughput=1000.0,
                goodput=1000.0,
                latency_avg=30.0 + iteration,
                latency_p95=40.0 - iteration,
                latency_p99=None,
                events_total=1000 + iteration,
                total_time_s=end - start,
            ),
            system_metrics=SystemMetrics(
                window_start_time=start,
                window_end_time=end,
                power_socket0_watts=50.0,
                power_ram_watts=5.0,
                cstate_poll_pct=0.0,
                cstate_c1_pct=10.0,
                cstate_c1e_pct=0.0,
                cstate_c6_pct=90.0,
                cpu_load_cores_pct=99.0,
                cpu_load_socket0_pct=50.0,
            ),
            window_start_time=start,
            window_end_time=end,
            log_file=str(self.run_dir / f"window_{iteration}.log"),
        )


class FakeTuner:
    def __init__(self, model_name: str, values):
        self.model_name = model_name
        self._values = iter(values)

    def suggest(self, context):
        time.sleep(0.01)
        value = next(self._values)
        return TunerResponse(
            proposed_parameters={"min_granularity_ns": value},
            justification=f"set to {value}",
            raw_text="Analysis: test\nConfig: {}",
            token_metrics={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def test_optimizer_writes_normalized_history(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(optimizer_module, "ACTOR_MIN_RESPONSE_SECONDS", 0.0)
    latency = tmp_path / "latency_ns"
    min_gran = tmp_path / "min_granularity_ns"
    latency.write_text("1000\n", encoding="utf-8")
    min_gran.write_text("3000000\n", encoding="utf-8")

    config = ReplicationConfig(
        mode="dual",
        benchmark="sysbench_cpu",
        run_duration_s=2,
        window_duration_s=1,
        results_dir=str(tmp_path / "results"),
    )
    config.validate()

    run_dir = tmp_path / "run"
    optimizer = ReplicationOptimizer(
        config=config,
        benchmark=FakeBenchmark(run_dir),
        parameter_manager=SchedulerParameterManager(
            {
                "latency_ns": str(latency),
                "min_granularity_ns": str(min_gran),
            }
        ),
        tuners={
            "actor": FakeTuner("gemini-2.5-flash", [250000, 200000]),
            "speculator": FakeTuner("gemini-2.5-flash-lite", [500000, 300000]),
        },
        run_dir=run_dir,
    )

    result = optimizer.run()
    output = capsys.readouterr().out
    history_path = Path(result["history_file"])
    payload = json.loads(history_path.read_text(encoding="utf-8"))

    assert payload["setup_mode"] == "dual"
    assert len(payload["history"]) == 2
    assert payload["requests"]
    assert {request["request_type"] for request in payload["requests"]} == {"actor", "speculator"}
    assert payload["requests"][0]["token_metrics"]["total_tokens"] == 15
    assert int(latency.read_text(encoding="utf-8").strip()) == 1000
    assert "Run start" in output
    assert "progress iter" in output
    assert "Run complete" in output
