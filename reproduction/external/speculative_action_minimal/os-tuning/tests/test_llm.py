from __future__ import annotations

from barebones_optimizer.tuners.llm import GeminiTuner, PromptContext


def make_tuner() -> GeminiTuner:
    return GeminiTuner(
        role="actor",
        mode="actor_only",
        model_name="gemini-2.5-flash",
        min_granularity_range_ns=(50_000, 50_000_000),
        latency_ns=1000,
        api_key="dummy",
        client=object(),
    )


def test_base_prompt_contains_appendix_language():
    tuner = make_tuner()
    prompt = tuner.build_base_prompt()
    assert "You are a Linux kernel scheduler tuning expert" in prompt
    assert 'Config: { "min granularity ns": <int> }' in prompt


def test_parse_response_extracts_analysis_and_config():
    tuner = make_tuner()
    parsed = tuner.parse_response(
        'Analysis: Narrow around the best region.\nConfig: { "min granularity ns": 250000 }'
    )
    assert parsed["analysis"] == "Narrow around the best region."
    assert parsed["config"]["min_granularity_ns"] == 250000


def test_build_prompt_uses_current_best():
    tuner = make_tuner()
    context = PromptContext(
        call_number=3,
        current_parameters={"latency_ns": 1000, "min_granularity_ns": 300000},
        best_window={
            "iteration": 2,
            "benchmark_metrics": {"latency_p95": 31.5},
        },
        history=[
            {
                "iteration": 1,
                "parameters": {"min_granularity_ns": 500000},
                "benchmark_metrics": {"latency_p95": 40.0},
            },
            {
                "iteration": 2,
                "parameters": {"min_granularity_ns": 300000},
                "benchmark_metrics": {"latency_p95": 31.5},
            },
        ],
        baseline_index=0,
        role="actor",
        mode="actor_only",
    )
    prompt = tuner.build_prompt(context)
    assert "CURRENT BEST: p95 latency=31.50 at call #2" in prompt
    assert "Please provide your analysis" in prompt
