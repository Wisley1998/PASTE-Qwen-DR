from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from google import genai

from .base import TunerResponse


PRICE_MAP_USD_PER_MILLION = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}


@dataclass
class PromptContext:
    call_number: int
    current_parameters: Dict[str, int]
    best_window: Optional[Dict[str, Any]]
    history: Sequence[Dict[str, Any]]
    baseline_index: int
    role: str
    mode: str


class GeminiTuner:
    """Appendix-style Gemini prompt builder and response parser."""

    def __init__(
        self,
        *,
        role: str,
        mode: str,
        model_name: str,
        min_granularity_range_ns: Sequence[int],
        latency_ns: int,
        api_key: str,
        client: Optional[Any] = None,
    ):
        self.role = role
        self.mode = mode
        self.model_name = model_name
        self.min_granularity_range_ns = (int(min_granularity_range_ns[0]), int(min_granularity_range_ns[1]))
        self.latency_ns = latency_ns
        self.client = client or genai.Client(api_key=api_key)

    def suggest(self, context: PromptContext) -> TunerResponse:
        prompt = self.build_prompt(context)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.build_base_prompt(), prompt],
        )
        text = getattr(response, "text", "") or ""
        parsed = self.parse_response(text)
        usage = getattr(response, "usage_metadata", None)
        token_metrics = None
        if usage is not None:
            token_metrics = {
                "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
            }
        return TunerResponse(
            proposed_parameters=parsed["config"],
            justification=parsed["analysis"],
            raw_text=text,
            token_metrics=token_metrics,
        )

    def build_base_prompt(self) -> str:
        role_text = {
            ("actor", "dual"): (
                "MULTI-AGENT ROLE: You are part of a MULTI-AGENT System.\n"
                "[For Actor] You are the Actor. Your role is to provide thoughtful, well-analyzed "
                "parameter recommendations. You work alongside a Speculator that explores the parameter "
                "space rapidly. You will receive accumulated results from multiple agent calls to perform "
                "deeper analysis and identify trends."
            ),
            ("speculator", "dual"): (
                "MULTI-AGENT ROLE: You are part of a MULTI-AGENT System.\n"
                "[For Speculator] You are the Speculator. Your role is to provide immediate, intuitive "
                "parameter recommendations for each window. You work alongside an Actor that performs "
                "deeper analysis."
            ),
            ("actor", "actor_only"): (
                "ROLE: You are the Actor-only baseline. Provide thoughtful, well-analyzed parameter "
                "recommendations using the observed history."
            ),
            ("speculator", "speculator_only"): (
                "ROLE: You are the Speculator-only baseline. Provide immediate, intuitive parameter "
                "recommendations every window using the latest result."
            ),
        }[(self.role, self.mode)]
        return (
            "You are a Linux kernel scheduler tuning expert with deep knowledge of the Completely Fair Scheduler (CFS).\n"
            f"{role_text}\n"
            "Your goal is to MINIMIZE p95 latency for a CPU-bound workload. "
            "The workload performance metrics might be NOISY, so look for consistent trends across configurations.\n"
            "Tunable CFS parameter:\n"
            "- min granularity ns: Minimum time slice before preemption. Lower values increase responsiveness but also overhead. Higher values improve throughput but can worsen latency.\n"
            "Parameter Range:\n"
            f"- min granularity ns: {self.min_granularity_range_ns[0]:,} to {self.min_granularity_range_ns[1]:,} nanoseconds\n"
            "Fixed CFS parameter:\n"
            f"- latency ns: {self.latency_ns:,} nanoseconds\n"
            "Performance data will be provided in future calls. Respond ONLY in the format shown below:\n"
            "Analysis: <Your one or two-sentence decision reasoning>\n"
            'Config: { "min granularity ns": <int> }'
        )

    def build_prompt(self, context: PromptContext) -> str:
        if not context.history:
            return (
                f"CURRENT BEST: p95 latency=N/A at call #0\n"
                f"No prior performance data has been observed yet.\n"
                f"Current config: min granularity ns={context.current_parameters['min_granularity_ns']:,}\n"
                f"Please provide your analysis and the next configuration for call #{context.call_number}."
            )

        best_line = self._best_line(context.best_window)
        older_windows = list(context.history[: context.baseline_index])
        recent_windows = list(context.history[context.baseline_index :])
        latest = context.history[-1]
        latest_min = latest["parameters"]["min_granularity_ns"]
        latest_p95 = latest["benchmark_metrics"]["latency_p95"]

        sections: List[str] = [best_line]
        compressed = self._compressed_history(older_windows)
        if compressed:
            sections.append("[Compressed history]")
            sections.append(compressed)

        if self.role == "actor" and self.mode == "dual":
            raw_recent = self._raw_recent_speculator_results(recent_windows)
            if raw_recent:
                sections.append(raw_recent)
            sections.append(
                f"RESULT for call #{latest['iteration']} [LATEST]: min granularity ns={latest_min:,} ↑ p95 latency={latest_p95:.2f}"
            )
            sections.append(
                f"Please provide your analysis of the trend and the next configuration for call #{context.call_number}."
            )
        else:
            sections.append(
                f"Latest Result for call #{latest['iteration']}:\n"
                f'Config: {{ "min granularity ns": {latest_min} }} ↑ p95 latency={latest_p95:.2f}'
            )
            sections.append(
                f"Please provide your analysis and the next configuration for call #{context.call_number}."
            )

        return "\n".join(section for section in sections if section)

    @staticmethod
    def _best_line(best_window: Optional[Dict[str, Any]]) -> str:
        if not best_window:
            return "CURRENT BEST: p95 latency=N/A at call #0"
        return (
            f"CURRENT BEST: p95 latency={best_window['benchmark_metrics']['latency_p95']:.2f} "
            f"at call #{best_window['iteration']}"
        )

    def _compressed_history(self, windows: Sequence[Dict[str, Any]]) -> str:
        if not windows:
            return ""
        if len(windows) <= 12:
            return "\n".join(self._window_summary(window) for window in windows)

        lines: List[str] = []
        chunk_size = 10
        for start in range(0, len(windows), chunk_size):
            chunk = windows[start : start + chunk_size]
            best = min(chunk, key=lambda item: item["benchmark_metrics"]["latency_p95"])
            last = chunk[-1]
            lines.append(
                "Calls "
                f"{chunk[0]['iteration']}-{chunk[-1]['iteration']}: "
                f"best p95={best['benchmark_metrics']['latency_p95']:.2f} at min granularity "
                f"{best['parameters']['min_granularity_ns']:,}; "
                f"last p95={last['benchmark_metrics']['latency_p95']:.2f} at "
                f"{last['parameters']['min_granularity_ns']:,}"
            )
        return "\n".join(lines)

    @staticmethod
    def _window_summary(window: Dict[str, Any]) -> str:
        return (
            f"call #{window['iteration']}: min granularity ns={window['parameters']['min_granularity_ns']:,} "
            f"↑ p95 latency={window['benchmark_metrics']['latency_p95']:.2f}"
        )

    @staticmethod
    def _raw_recent_speculator_results(windows: Sequence[Dict[str, Any]]) -> str:
        if not windows:
            return ""
        return "\n".join(
            (
                f"RESULT for call #{window['iteration']} [SPECULATOR]: "
                f"min granularity ns={window['parameters']['min_granularity_ns']:,} "
                f"↑ p95 latency={window['benchmark_metrics']['latency_p95']:.2f}"
            )
            for window in windows
        )

    def parse_response(self, text: str) -> Dict[str, Any]:
        config_match = re.search(r"Config:\s*(\{.*?\})", text, flags=re.IGNORECASE | re.DOTALL)
        if not config_match:
            raise ValueError(f"Could not parse Config block from response: {text}")

        config_text = config_match.group(1)
        normalized = config_text.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            value_match = re.search(r"min\s*granularity\s*ns\s*[:=]\s*([0-9]+)", normalized, flags=re.IGNORECASE)
            if not value_match:
                raise ValueError(f"Could not parse min granularity value from response: {text}")
            parsed = {"min granularity ns": int(value_match.group(1))}

        value = parsed.get("min granularity ns")
        if value is None:
            for key, candidate in parsed.items():
                if key.replace("_", " ").strip().lower() == "min granularity ns":
                    value = candidate
                    break
        if value is None:
            raise ValueError(f"Response missing min granularity ns: {text}")

        value = int(value)
        lo, hi = self.min_granularity_range_ns
        value = max(lo, min(hi, value))

        analysis_match = re.search(r"Analysis:\s*(.*?)(?:\nConfig:|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided."
        return {"analysis": analysis, "config": {"min_granularity_ns": value}}

    @staticmethod
    def token_cost_usd(model_name: str, token_metrics: Optional[Dict[str, int]]) -> Optional[float]:
        if not token_metrics:
            return None
        rates = PRICE_MAP_USD_PER_MILLION.get(model_name)
        if not rates:
            return None
        input_tokens = max(0, int(token_metrics.get("input_tokens", 0)))
        output_tokens = max(0, int(token_metrics.get("output_tokens", 0)))
        return (
            input_tokens / 1_000_000.0 * rates["input"]
            + output_tokens / 1_000_000.0 * rates["output"]
        )
