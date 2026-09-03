from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class BenchmarkMetrics:
    throughput: float
    goodput: float
    latency_avg: float
    latency_p95: float
    latency_p99: Optional[float]
    events_total: int
    total_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemMetrics:
    window_start_time: float
    window_end_time: float
    power_socket0_watts: Optional[float]
    power_ram_watts: Optional[float]
    cstate_poll_pct: Optional[float]
    cstate_c1_pct: Optional[float]
    cstate_c1e_pct: Optional[float]
    cstate_c6_pct: Optional[float]
    cpu_load_cores_pct: Optional[float]
    cpu_load_socket0_pct: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WindowExecution:
    benchmark_metrics: BenchmarkMetrics
    system_metrics: SystemMetrics
    window_start_time: float
    window_end_time: float
    log_file: str
