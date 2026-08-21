from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY_ROOT / "reproduction/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_native_prefix_causal_dev as matrix_runner  # noqa: E402
import run_native_prefix_prompt_cell as cell_runner  # noqa: E402


class FakeTokenizer:
    all_special_ids = []

    def encode(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        if text == "A":
            return [32]
        return list(range(max(1, math.ceil(len(text) / 4))))

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "A" if list(token_ids) == [32] else "invalid"

    def apply_chat_template(
        self, conversation, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize and add_generation_prompt
        count = 5 + sum(
            2 + math.ceil(len(row["content"]) / 4) for row in conversation
        )
        return list(range(count))


def test_default_config_and_check_only_are_frozen_and_nonexecuting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = matrix_runner.load_frozen_config(matrix_runner.DEFAULT_CONFIG)
    assert matrix_runner._matrix_from_config(config) == matrix_runner.EXACT_MATRIX
    assert matrix_runner._thresholds_from_config(config) == matrix_runner.EXACT_THRESHOLDS

    def forbidden(*_args, **_kwargs):
        raise AssertionError("check-only must not launch a process")

    monkeypatch.setattr(matrix_runner, "_run_logged", forbidden)
    assert matrix_runner.main(["unit_check_only", "--check-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["gpu_or_server_touched"] is False
    assert payload["external_network_touched"] is False
    assert payload["output_created"] is False
    assert payload["orders"] == [["P0", "P1"], ["P1", "P0"]]
    assert payload["matrix"]["task_count"] == 48
    assert payload["matrix"]["max_tokens_by_call"] == [1, 1, 1]
    assert payload["fixture_preflight"]["sentinel_contract"]["token_id"] == 32
    assert payload["contract_bindings"]["prior_r1_disposition"].startswith(
        "rejected_diagnostic"
    )
    assert payload["engine"]["VLLM_MAX_NUM_SEQS"] == "96"


@pytest.mark.parametrize(("cell_id", "flag"), [("P0", "0"), ("P1", "1")])
def test_cell_environment_clears_every_joint_or_trace_knob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_id: str,
    flag: str,
) -> None:
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY", "1")
    monkeypatch.setenv("VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS", "1")
    monkeypatch.setenv("VLLM_TRACE_SWAP_PATCH", "1")
    monkeypatch.setenv("VLLM_FAKE_HIDDEN_CAP", "1")
    monkeypatch.setenv("PYTHONPATH", "/tmp/host-sitecustomize")
    config = matrix_runner.load_frozen_config(matrix_runner.DEFAULT_CONFIG)
    native_pythonpath = tmp_path / "native_pythonpath"
    native_pythonpath.mkdir()
    env = matrix_runner._cell_environment(
        config,
        cell_id=cell_id,
        state_dir=tmp_path / "state",
        server_dir=tmp_path / "server",
        model_snapshot=tmp_path / "model",
        native_pythonpath=native_pythonpath,
    )

    assert {
        key for key in env if key.startswith("VLLM_SCHED_")
    } == {"VLLM_SCHED_POLICY"}
    assert env["VLLM_SCHED_POLICY"] == "fcfs"
    assert env["VLLM_ENABLE_PREFIX_CACHING"] == flag
    assert not any(key.startswith("VLLM_TRACE_") for key in env)
    assert "VLLM_FAKE_HIDDEN_CAP" not in env
    assert env["VLLM_HOOK_DIR"] == str(native_pythonpath)
    assert env["PYTHONPATH"] == ""


def test_cell_command_is_loopback_local_and_has_only_prefix_treatment(
    tmp_path: Path,
) -> None:
    config = matrix_runner.load_frozen_config(matrix_runner.DEFAULT_CONFIG)
    command = matrix_runner._cell_command(
        python=Path(config["PASTE_ENV_PREFIX"]) / "bin/python",
        config=config,
        workload=REPOSITORY_ROOT / config["PASTE_PREFIX_CAUSAL_WORKLOAD"],
        model_snapshot=tmp_path / "model",
        output_dir=tmp_path / "out",
        block_id="unit-block-1",
        order_index=0,
        cell_id="P0",
        server_instance_id="server-1",
    )
    rendered = " ".join(str(value) for value in command)
    assert "--server-url http://127.0.0.1:8100" in rendered
    assert "--no-prefix-cache-enabled" in command
    assert "--expected-task-count 48" in rendered
    assert "--max-active-tasks 48" in rendered
    assert "bing" not in rendered.lower()
    assert "jina" not in rendered.lower()
    assert "speculation" not in rendered.lower()


def test_fixture_builder_has_three_fixed_long_context_calls_and_unique_tasks() -> None:
    source = cell_runner.Source(
        "source-1",
        "What fixed fact is being tested?",
        "fixed query",
        "https://en.wikipedia.org/wiki/Fixed",
    )
    tasks, manifest_sha = cell_runner.build_task_fixtures(
        FakeTokenizer(),
        [source],
        replicas=2,
        context_padding_tokens=10000,
        visit_fixture_tokens=900,
        max_tokens_by_call=(1, 1, 1),
        max_model_len=16384,
    )

    assert len(tasks) == 2
    assert len({task.task_id for task in tasks}) == 2
    assert len(manifest_sha) == 64
    assert all(len(task.calls) == 3 for task in tasks)
    for task in tasks:
        prompts = [call.prompt_tokens for call in task.calls]
        assert 10000 <= prompts[0] <= 10768
        assert 64 <= prompts[1] - prompts[0] <= 768
        assert 640 <= prompts[2] - prompts[1] <= 1536
        assert prompts[2] + 1 < 16384
        assert all(
            call.expected_completion_sha256
            == cell_runner.sha256_json(call.expected_completion)
            for call in task.calls
        )
        assert all(call.expected_completion == "A" for call in task.calls)
        assert all(call.expected_completion_tokens == 1 for call in task.calls)
        assert all(call.max_tokens == 1 for call in task.calls)
        assert all(call.guided_choice == ("A",) for call in task.calls)
    assert tasks[0].calls[0].messages_sha256 != tasks[1].calls[0].messages_sha256


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8100",
        "http://example.com:8100",
        "http://user@127.0.0.1:8100",
        "http://127.0.0.1",
    ],
)
def test_cell_driver_rejects_nonlocal_or_ambiguous_server_urls(url: str) -> None:
    with pytest.raises(ValueError):
        cell_runner._validate_loopback_url(url)


def test_cell_driver_accepts_explicit_loopback() -> None:
    assert (
        cell_runner._validate_loopback_url("http://127.0.0.1:8100/")
        == "http://127.0.0.1:8100"
    )


def test_single_token_sentinel_preflight_rejects_non_roundtrip() -> None:
    tokenizer = FakeTokenizer()
    assert cell_runner.validate_single_token_sentinel(tokenizer) == (
        validator_contract := {
            "contract": "guided_choice_singleton_v1",
            "sentinel": "A",
            "sentinel_utf8_sha256": cell_runner.hashlib.sha256(b"A").hexdigest(),
            "token_id": 32,
            "token_ids_sha256": cell_runner.sha256_json([32]),
            "token_count": 1,
            "allowed_choice_count": 1,
            "guided_choice": ["A"],
            "guided_choice_sha256": cell_runner.sha256_json(["A"]),
            "max_tokens": 1,
            "round_trip_exact": True,
            "special_token": False,
        }
    )
    assert validator_contract["allowed_choice_count"] == 1

    class BrokenTokenizer(FakeTokenizer):
        def decode(self, *_args, **_kwargs):
            return " A"

    with pytest.raises(ValueError, match="decode byte-for-byte"):
        cell_runner.validate_single_token_sentinel(BrokenTokenizer())
