from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..benchmark import BenchmarkMetrics, SystemMetrics, WindowExecution

logger = logging.getLogger(__name__)


class SysbenchCpuBenchmark:
    """1 Hz sysbench CPU runner with direct start/end system metric sampling."""

    def __init__(self, config, run_dir: Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.windows_dir = self.run_dir / "windows"
        self.windows_dir.mkdir(parents=True, exist_ok=True)

    def execute_window(self, iteration: int, duration_s: int) -> WindowExecution:
        window_dir = self.windows_dir / f"window_{iteration:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        log_file = window_dir / "sysbench.log"

        command = self._build_command(duration_s)
        start_time = time.time()
        snapshot_before = self._capture_system_snapshot()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            check=False,
        )
        end_time = time.time()
        log_file.write_text(result.stdout, encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(f"sysbench failed for window {iteration}: {result.stdout}")

        benchmark_metrics = self._parse_metrics(result.stdout)
        system_metrics = self._compute_system_metrics(start_time, end_time, snapshot_before)

        return WindowExecution(
            benchmark_metrics=benchmark_metrics,
            system_metrics=system_metrics,
            window_start_time=start_time,
            window_end_time=end_time,
            log_file=str(log_file),
        )

    def _build_command(self, duration_s: int) -> List[str]:
        command = [
            "sysbench",
            "cpu",
            f"--threads={self.config.sysbench_threads}",
            f"--cpu-max-prime={self.config.sysbench_cpu_max_prime}",
            f"--time={duration_s}",
            "--percentile=95",
            "run",
        ]
        if self.config.pin_to_cores:
            return ["taskset", "-c", self.config.pin_to_cores, *command]
        return command

    def _parse_metrics(self, output: str) -> BenchmarkMetrics:
        def first_float(pattern: str) -> Optional[float]:
            match = re.search(pattern, output, flags=re.MULTILINE)
            return float(match.group(1)) if match else None

        def first_int(pattern: str) -> Optional[int]:
            match = re.search(pattern, output, flags=re.MULTILINE)
            if not match:
                return None
            return int(match.group(1).replace(",", ""))

        throughput = first_float(r"events per second:\s+([0-9.]+)") or 0.0
        total_time = first_float(r"total time:\s+([0-9.]+)s") or 0.0
        events_total = first_int(r"total number of events:\s+([0-9,]+)") or 0
        latency_avg = first_float(r"avg:\s+([0-9.]+)") or 0.0
        latency_p95 = first_float(r"95th percentile:\s+([0-9.]+)") or 0.0
        latency_p99 = first_float(r"99th percentile:\s+([0-9.]+)")

        return BenchmarkMetrics(
            throughput=throughput,
            goodput=throughput,
            latency_avg=latency_avg,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            events_total=events_total,
            total_time_s=total_time,
        )

    def _capture_system_snapshot(self) -> Dict[str, object]:
        return {
            "rapl": self._read_rapl(),
            "cpu": self._read_cpu_stats(),
            "cstate": self._read_cstate_residency(self._parse_cores_spec(self.config.pin_to_cores)),
        }

    def _compute_system_metrics(
        self,
        start_time: float,
        end_time: float,
        before: Dict[str, object],
    ) -> SystemMetrics:
        after = self._capture_system_snapshot()
        elapsed = max(1e-6, end_time - start_time)

        before_rapl = before["rapl"]
        after_rapl = after["rapl"]
        package_watts = self._compute_power(before_rapl.get("package"), after_rapl.get("package"), elapsed)
        dram_watts = self._compute_power(before_rapl.get("dram"), after_rapl.get("dram"), elapsed)

        cpu_before = before["cpu"]
        cpu_after = after["cpu"]
        cores_filter = self._parse_cores_spec(self.config.pin_to_cores)

        cstate_before = before["cstate"]
        cstate_after = after["cstate"]
        cstate_percentages = self._compute_cstate_percentages(cstate_before, cstate_after)

        return SystemMetrics(
            window_start_time=start_time,
            window_end_time=end_time,
            power_socket0_watts=package_watts,
            power_ram_watts=dram_watts,
            cstate_poll_pct=cstate_percentages.get("POLL"),
            cstate_c1_pct=cstate_percentages.get("C1"),
            cstate_c1e_pct=cstate_percentages.get("C1E"),
            cstate_c6_pct=cstate_percentages.get("C6"),
            cpu_load_cores_pct=self._calculate_cpu_load(cpu_before, cpu_after, cores_filter=cores_filter),
            cpu_load_socket0_pct=self._calculate_cpu_load(cpu_before, cpu_after),
        )

    def _read_rapl(self) -> Dict[str, Optional[int]]:
        result = {"package": None, "dram": None}
        base = Path("/sys/class/powercap")
        if not base.exists():
            return result

        for zone in base.rglob("energy_uj"):
            try:
                name_file = zone.parent / "name"
                name = name_file.read_text(encoding="utf-8").strip().lower()
                value = int(zone.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue

            if "dram" in name and result["dram"] is None:
                result["dram"] = value
            elif ("package" in name or "pkg" in name) and result["package"] is None:
                result["package"] = value

        return result

    @staticmethod
    def _compute_power(before: Optional[int], after: Optional[int], elapsed: float) -> Optional[float]:
        if before is None or after is None:
            return None
        delta = after - before
        if delta < 0:
            return None
        return delta / elapsed / 1_000_000.0

    def _read_cpu_stats(self) -> Dict[str, Tuple[int, ...]]:
        stats: Dict[str, Tuple[int, ...]] = {}
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("cpu"):
                    continue
                parts = line.split()
                label = parts[0]
                values = tuple(int(part) for part in parts[1:])
                stats[label] = values
        return stats

    @staticmethod
    def _cpu_utilization(before: Sequence[int], after: Sequence[int]) -> Optional[float]:
        total_before = sum(before)
        total_after = sum(after)
        idle_before = before[3] + (before[4] if len(before) > 4 else 0)
        idle_after = after[3] + (after[4] if len(after) > 4 else 0)
        total_delta = total_after - total_before
        idle_delta = idle_after - idle_before
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100.0))

    def _calculate_cpu_load(
        self,
        before: Dict[str, Tuple[int, ...]],
        after: Dict[str, Tuple[int, ...]],
        *,
        cores_filter: Optional[Set[int]] = None,
    ) -> Optional[float]:
        labels: Iterable[str]
        if cores_filter is None:
            labels = ("cpu",)
        else:
            labels = [f"cpu{core}" for core in sorted(cores_filter)]

        values = []
        for label in labels:
            if label not in before or label not in after:
                continue
            value = self._cpu_utilization(before[label], after[label])
            if value is not None:
                values.append(value)

        if not values:
            return None
        return sum(values) / len(values)

    def _read_cstate_residency(self, cores_filter: Optional[Set[int]]) -> Dict[str, int]:
        residency: Dict[str, int] = {}
        cpu_base = Path("/sys/devices/system/cpu")
        cores = cores_filter
        if cores is None:
            cores = {
                int(path.name[3:])
                for path in cpu_base.glob("cpu[0-9]*")
                if path.name[3:].isdigit()
            }

        for core in cores:
            cpuidle_dir = cpu_base / f"cpu{core}" / "cpuidle"
            if not cpuidle_dir.exists():
                continue
            for state_dir in cpuidle_dir.glob("state*"):
                try:
                    name = (state_dir / "name").read_text(encoding="utf-8").strip()
                    value = int((state_dir / "time").read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    continue
                residency[name] = residency.get(name, 0) + value
        return residency

    @staticmethod
    def _compute_cstate_percentages(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, float]:
        deltas: Dict[str, int] = {}
        for key in set(before) | set(after):
            delta = after.get(key, 0) - before.get(key, 0)
            if delta > 0:
                deltas[key] = delta
        total = sum(deltas.values())
        if total <= 0:
            return {}
        return {key: value / total * 100.0 for key, value in deltas.items()}

    @staticmethod
    def _parse_cores_spec(spec: Optional[str]) -> Optional[Set[int]]:
        if not spec:
            return None

        result: Set[int] = set()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                result.update(range(start, end + 1))
            else:
                result.add(int(part))
        return result
