from __future__ import annotations

import subprocess

import pytest

from barebones_optimizer.parameter_manager import SchedulerParameterManager


def test_parameter_manager_snapshot_apply_restore(tmp_path):
    latency = tmp_path / "latency_ns"
    min_gran = tmp_path / "min_granularity_ns"
    latency.write_text("1000\n", encoding="utf-8")
    min_gran.write_text("3000000\n", encoding="utf-8")

    manager = SchedulerParameterManager(
        {
            "latency_ns": str(latency),
            "min_granularity_ns": str(min_gran),
        }
    )

    snapshot = manager.snapshot()
    assert snapshot == {"latency_ns": 1000, "min_granularity_ns": 3000000}

    manager.apply(latency_ns=2000, min_granularity_ns=250000)
    assert manager.snapshot() == {"latency_ns": 2000, "min_granularity_ns": 250000}

    manager.restore(snapshot)
    assert manager.snapshot() == snapshot

    assert manager.access_summary() == {
        "latency_ns": "direct",
        "min_granularity_ns": "direct",
    }


def test_parameter_manager_uses_sudo_fallback(monkeypatch, tmp_path):
    latency = tmp_path / "latency_ns"
    min_gran = tmp_path / "min_granularity_ns"

    manager = SchedulerParameterManager(
        {
            "latency_ns": str(latency),
            "min_granularity_ns": str(min_gran),
        }
    )

    def fail_read(*args, **kwargs):
        raise PermissionError("denied")

    def fail_write(*args, **kwargs):
        raise PermissionError("denied")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["sudo", "-n", "cat"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="123\n", stderr="")
        if cmd[:3] == ["sudo", "-n", "tee"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr("pathlib.Path.read_text", fail_read)
    monkeypatch.setattr("pathlib.Path.write_text", fail_write)
    monkeypatch.setattr("subprocess.run", fake_run)

    assert manager.read_parameter("latency_ns") == 123
    manager.write_parameter("min_granularity_ns", 250000)
    assert calls[0][:3] == ["sudo", "-n", "cat"]
    assert calls[1][:3] == ["sudo", "-n", "tee"]
