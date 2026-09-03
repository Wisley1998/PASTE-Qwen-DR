from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


SUPPORTED_MODES = {"actor_only", "speculator_only", "dual"}
SUPPORTED_BENCHMARK = "sysbench_cpu"
SUPPORTED_RUN_KINDS = {"convergence", "reaction"}


@dataclass
class ReplicationConfig:
    mode: str
    benchmark: str = SUPPORTED_BENCHMARK
    latency_ns: int = 1000
    min_granularity_range_ns: Tuple[int, int] = (50_000, 50_000_000)
    window_duration_s: int = 1
    run_duration_s: int = 200
    pin_to_cores: str = "0-9"
    sysbench_threads: int = 40
    sysbench_cpu_max_prime: int = 50_000
    results_dir: str = "results/use_case"
    gemini_api_key: Optional[str] = None
    perturbation_min_granularity_ns: int = 10_000_000
    run_kind: str = "convergence"
    actor_model: str = "gemini-2.5-flash"
    speculator_model: str = "gemini-2.5-flash-lite"
    request_timeout_s: float = 30.0

    @classmethod
    def load(cls, file_path: str) -> "ReplicationConfig":
        with open(file_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if "min_granularity_range_ns" in raw:
            raw["min_granularity_range_ns"] = tuple(raw["min_granularity_range_ns"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.benchmark != SUPPORTED_BENCHMARK:
            raise ValueError(f"Unsupported benchmark: {self.benchmark}")
        if self.run_kind not in SUPPORTED_RUN_KINDS:
            raise ValueError(f"Unsupported run_kind: {self.run_kind}")
        if self.latency_ns <= 0:
            raise ValueError("latency_ns must be positive")
        lo, hi = self.min_granularity_range_ns
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError("min_granularity_range_ns must be an increasing positive pair")
        if self.window_duration_s <= 0:
            raise ValueError("window_duration_s must be positive")
        if self.run_duration_s <= 0:
            raise ValueError("run_duration_s must be positive")
        if self.run_duration_s % self.window_duration_s != 0:
            raise ValueError("run_duration_s must be divisible by window_duration_s")
        if self.sysbench_threads <= 0:
            raise ValueError("sysbench_threads must be positive")
        if self.sysbench_cpu_max_prime <= 0:
            raise ValueError("sysbench_cpu_max_prime must be positive")
        if self.perturbation_min_granularity_ns <= 0:
            raise ValueError("perturbation_min_granularity_ns must be positive")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if not self.gemini_api_key and not os.getenv("GEMINI_API_KEY"):
            raise ValueError("Set gemini_api_key in config or GEMINI_API_KEY in the environment")

    @property
    def iterations(self) -> int:
        return self.run_duration_s // self.window_duration_s

    @property
    def api_key(self) -> str:
        key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Missing Gemini API key")
        return key

    @property
    def initial_min_granularity_ns(self) -> Optional[int]:
        if self.run_kind == "reaction":
            return self.perturbation_min_granularity_ns
        return None

    def model_for_role(self, role: str) -> str:
        if role == "actor":
            return self.actor_model
        if role == "speculator":
            return self.speculator_model
        raise ValueError(f"Unsupported role: {role}")

    def active_roles(self) -> Tuple[str, ...]:
        if self.mode == "actor_only":
            return ("actor",)
        if self.mode == "speculator_only":
            return ("speculator",)
        return ("actor", "speculator")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["min_granularity_range_ns"] = list(self.min_granularity_range_ns)
        return data

    def make_run_dir(self) -> Path:
        root = Path(self.results_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root
