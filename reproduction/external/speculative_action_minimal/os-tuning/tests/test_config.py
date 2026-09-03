from __future__ import annotations

import json

import pytest

from barebones_optimizer.config import ReplicationConfig


def test_config_load_and_validate(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "dual",
                "benchmark": "sysbench_cpu",
                "run_duration_s": 20,
                "window_duration_s": 1,
            }
        ),
        encoding="utf-8",
    )
    config = ReplicationConfig.load(str(config_path))
    assert config.mode == "dual"
    assert config.iterations == 20


def test_config_rejects_unsupported_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    config = ReplicationConfig(mode="fixed")
    with pytest.raises(ValueError):
        config.validate()
