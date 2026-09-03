from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Mapping, Optional


DEFAULT_PARAMETER_PATHS = {
    "latency_ns": "/sys/kernel/debug/sched/latency_ns",
    "min_granularity_ns": "/sys/kernel/debug/sched/min_granularity_ns",
}


class SchedulerParameterManager:
    """Minimal parameter manager for the sysbench CPU use case."""

    def __init__(self, parameter_paths: Optional[Mapping[str, str]] = None):
        self.parameter_paths = dict(parameter_paths or DEFAULT_PARAMETER_PATHS)

    def _path_for(self, name: str) -> Path:
        try:
            return Path(self.parameter_paths[name])
        except KeyError as exc:
            raise ValueError(f"Unsupported parameter: {name}") from exc

    def read_parameter(self, name: str) -> int:
        path = self._path_for(name)
        return int(self._read_text(path).strip())

    def write_parameter(self, name: str, value: int) -> None:
        path = self._path_for(name)
        self._write_text(path, f"{int(value)}\n")

    def apply(self, *, latency_ns: Optional[int] = None, min_granularity_ns: Optional[int] = None) -> float:
        if latency_ns is not None:
            self.write_parameter("latency_ns", latency_ns)
        if min_granularity_ns is not None:
            self.write_parameter("min_granularity_ns", min_granularity_ns)
        return time.time()

    def snapshot(self) -> Dict[str, int]:
        return {
            "latency_ns": self.read_parameter("latency_ns"),
            "min_granularity_ns": self.read_parameter("min_granularity_ns"),
        }

    def restore(self, snapshot: Mapping[str, int]) -> float:
        return self.apply(
            latency_ns=int(snapshot["latency_ns"]),
            min_granularity_ns=int(snapshot["min_granularity_ns"]),
        )

    def access_mode(self, name: str) -> str:
        path = self._path_for(name)
        if os.access(path, os.W_OK):
            return "direct"
        if self._sudo_test("-w", path):
            return "sudo"
        if path.exists() or self._sudo_test("-e", path):
            return "unavailable"
        return "missing"

    def access_summary(self) -> Dict[str, str]:
        return {name: self.access_mode(name) for name in self.parameter_paths}

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            result = subprocess.run(
                ["sudo", "-n", "cat", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            subprocess.run(
                ["sudo", "-n", "tee", str(path)],
                check=True,
                input=content,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

    @staticmethod
    def _sudo_test(flag: str, path: Path) -> bool:
        try:
            result = subprocess.run(
                ["sudo", "-n", "test", flag, str(path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False
        return result.returncode == 0
