from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from paste_repro.pattern_v2_strict_adapter import (
    HashedUniformSLOClock,
    new_hashed_slo_clock_artifact,
)
from paste_repro.traces import ToolCall


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pattern_v2_strict_slo_simulation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_pattern_v2_strict_slo_simulation_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _clock() -> HashedUniformSLOClock:
    return HashedUniformSLOClock(
        new_hashed_slo_clock_artifact(seed_sha256="d" * 64)
    )


def test_environment_tool_service_ignores_trace_timing_and_timestamp() -> None:
    first = ToolCall(
        call_index=1,
        timestamp_s=1.0,
        tool_name="visit",
        tool_args={"url": ["https://example.test/a"]},
        line_number=2,
        timing_correction={
            "duration_s": 0.001,
            "unit_duration_s": [0.001],
        },
    )
    poisoned = ToolCall(
        call_index=999,
        timestamp_s=1e12,
        tool_name="visit",
        tool_args={"url": ["https://example.test/a"]},
        line_number=999,
        timing_correction={
            "duration_s": 1e12,
            "unit_duration_s": [1e12],
        },
    )
    assert runner._tool_service_s(_clock(), first) == runner._tool_service_s(
        _clock(), poisoned
    )


def test_multi_url_visit_service_is_serial_hashed_unit_sum() -> None:
    clock = _clock()
    event = ToolCall(
        call_index=1,
        timestamp_s=0.0,
        tool_name="visit",
        tool_args={
            "url": [
                "HTTPS://Example.Test:443/a#fragment",
                "https://example.test/b",
            ]
        },
        line_number=1,
        timing_correction=None,
    )
    expected = sum(
        clock.service_s(tool_name="visit", tool_arguments={"url": url})
        for url in ("https://example.test/a", "https://example.test/b")
    )
    assert runner._tool_service_s(clock, event) == expected
