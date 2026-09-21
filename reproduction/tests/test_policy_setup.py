import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/policy_setup.py"
spec = importlib.util.spec_from_file_location("policy_setup", path)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def lock():
    config = json.loads(setup.CONFIG.read_text())
    return {"config": config, "profiles": {p: setup.profile_environment(
        setup.ROOT / config[p]["repository"] / config[p]["profile"]) for p in ("dr", "coding")}}


@pytest.mark.parametrize("project", ["dr", "coding"])
def test_tool_information_comparison_changes_only_engine_ranker(project, tmp_path):
    frozen = lock()
    le, lc = setup.cell_command(frozen, project, "test", "length_spec", tmp_path)
    fe, fc = setup.cell_command(frozen, project, "test", "full", tmp_path)
    assert lc == fc
    assert le.pop("VLLM_SCHED_JOINT_V2_RANKER") == "length-aware"
    assert fe.pop("VLLM_SCHED_JOINT_V2_RANKER") == "tool-aware"
    assert le == fe


def test_native_control_has_same_tool_capacity_and_no_speculation(tmp_path):
    frozen = lock()
    e, native = setup.cell_command(frozen, "dr", "test", "native_fcfs", tmp_path)
    _, full = setup.cell_command(frozen, "dr", "test", "full", tmp_path)
    assert e["VLLM_SCHED_POLICY"] == "fcfs"
    assert native[native.index("--speculation-capacity") + 1] == "0"
    assert native[native.index("--tool-capacity") + 1] == full[full.index("--tool-capacity") + 1]
    assert "--session-result-cache" in full and "--preempt-speculation" in full


def test_coding_tune_and_test_never_mix_sources(tmp_path):
    frozen = lock()
    for split in ("tune", "test"):
        _, command = setup.cell_command(frozen, "coding", split, "full", tmp_path)
        sources = {command[i + 1] for i, value in enumerate(command) if value == "--source-session"}
        assert sources == set(frozen["config"]["coding"][f"{split}_sources"])
    assert not set(frozen["config"]["coding"]["tune_sources"]) & set(frozen["config"]["coding"]["test_sources"])


def test_profile_ignores_ambient_scheduler_alias(monkeypatch):
    monkeypatch.setenv("VLLM_SCHED_DECODE_TOKENS_PER_S_V2", "1")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_FINAL_LANE", "999")
    e = setup.profile_environment(setup.DR / "reproduction/configs/unified_workset_v1.env.example")
    assert e["VLLM_SCHED_DECODE_TOKENS_PER_S_V2"] == "113.7"
    assert e["VLLM_SCHED_JOINT_V2_FINAL_LANE"] == "1"


def test_client_owned_working_set_disables_duplicate_server_gate(tmp_path):
    frozen = lock()
    frozen["config"]["coding"].update(working_set=40, server_working_set=2147483647)
    env, cmd = setup.cell_command(frozen, "coding", "tune", "full", tmp_path)
    assert env["VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS"] == "2147483647"
    assert cmd[cmd.index("--max-active-tasks") + 1] == "40"


def test_coding_entry_imports_own_package_without_pythonpath():
    import os
    import subprocess
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    runner = setup.ROOT / "gemini-cli-PASTE/reproduction/scripts/run_swe_trace_live_pair.py"
    code = ('import runpy; runpy.run_path(' + repr(str(runner)) + ', run_name="entry_import_check"); '
            'from paste_gemini.trace_tool_pool import AsyncPreemptibleToolPool')
    subprocess.run([str(setup.PYTHON), "-c", code], env=env, check=True,
                   capture_output=True, text=True)
