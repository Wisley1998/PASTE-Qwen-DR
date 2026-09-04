from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from paste_repro.pattern_v2_strict_adapter import SCHEDULER_METADATA_SCHEMA


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load(
    "qwen_dr_trace_hybrid_runner_for_test",
    REPOSITORY_ROOT / "reproduction" / "scripts" / "run_dr_trace_hybrid_pair.py",
)
policy = _load(
    "qwen_sched_policy_patch_for_test",
    REPOSITORY_ROOT / "scripts" / "pythonhooks" / "sched_policy_patch.py",
)


def test_pattern_v2_adapter_and_server_share_strict_causal_wire_schema() -> None:
    meta = {
        "ms": SCHEDULER_METADATA_SCHEMA,
        "tool_eta_s_hat": 7.0,
        "tool_hit_probability_hat": 0.75,
    }

    assert SCHEDULER_METADATA_SCHEMA == policy._STRICT_CAUSAL_METADATA_SCHEMA
    assert policy._strict_causal_metadata(meta)
    assert policy._causal_meta_float(
        meta,
        predicted_key="tool_eta_s_hat",
        legacy_key="nw",
        default=-1.0,
    ) == 7.0


def test_v1_schedule_call_atomically_records_runtime_policy_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.waiting = []

        def schedule(self) -> str:
            return "scheduled"

    modules = {
        name: ModuleType(name)
        for name in (
            "vllm",
            "vllm.v1",
            "vllm.v1.core",
            "vllm.v1.core.sched",
            "vllm.v1.core.sched.scheduler",
        )
    }
    modules["vllm.v1.core.sched.scheduler"].Scheduler = FakeScheduler
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    marker = tmp_path / "scheduler-runtime.json"
    monkeypatch.setenv("VLLM_SCHED_POLICY", "online_joint_pacer_v2")
    monkeypatch.setenv("VLLM_SCHEDULER_RUNTIME_EVIDENCE", str(marker))
    policy._runtime_policy_evidence_emitted.clear()

    assert policy._install_v1("online_joint_pacer_v2") is True
    assert not marker.exists(), "installation alone is not execution evidence"
    assert FakeScheduler().schedule() == "scheduled"
    first_bytes = marker.read_bytes()
    evidence = json.loads(first_bytes)
    assert evidence["schema"] == "paste.vllm.scheduler_runtime_use.v1"
    assert evidence["pid"] == policy.os.getpid()
    assert evidence["policy"] == "online_joint_pacer_v2"
    assert evidence["scheduler_api"] == "v1.Scheduler.schedule"
    assert evidence["scheduler_hook_sha256"] == policy.hashlib.sha256(
        Path(policy.__file__).resolve().read_bytes()
    ).hexdigest()

    FakeScheduler().schedule()
    assert marker.read_bytes() == first_bytes
    policy._runtime_policy_evidence_emitted.clear()


def _trace() -> dict[str, Any]:
    completions = (100, 200, 50)
    prompts = (10, 20, 30)
    return {
        "task_id": "dr-test",
        "steps": [
            {
                "request": {
                    "call_index": index,
                    "prompt_tokens": prompt,
                    "fixed_completion_tokens": completion,
                },
                "tools_after": [
                    {
                        "duration_s": float(index + 2),
                        "offline_saved_s": float(index + 1),
                    }
                ] if index < 2 else [],
            }
            for index, (prompt, completion) in enumerate(
                zip(prompts, completions, strict=True)
            )
        ],
    }


def test_trace_metadata_remaining_llm_work_is_exact_and_monotone() -> None:
    trace = _trace()
    rows = [
        runner.build_scheduler_metadata(trace, index, full=True, po_ema=128.0)
        for index in range(3)
    ]

    assert [row["rlmt"] for row in rows] == [350, 250, 50]
    assert [row["npt"] for row in rows] == [20, 30, 0]
    assert [row["nmt"] for row in rows] == [200, 50, 0]
    assert [row["rc"] for row in rows] == [2, 1, 0]
    assert [row["eg"] for row in rows] == [3.0, 2.0, 0.0]


def test_session_admission_features_and_score_combine_work_gain_and_aging() -> None:
    trace = _trace()
    baseline = runner.session_admission_features(trace, full=False)
    full = runner.session_admission_features(trace, full=True)

    assert baseline == runner.SessionAdmissionFeatures(350, 30, 0.0)
    assert full == runner.SessionAdmissionFeatures(350, 30, 3.0)
    options = {
        "pressure": 0.75,
        "prefill_tokens_per_s": 10_000.0,
        "decode_tokens_per_s": 500.0,
        "pressure_weight": 1.0,
        "tool_gain_beta": 1.0,
        "aging_alpha": 0.1,
    }
    short = runner.SessionAdmissionFeatures(100, 100, 0.0)
    long = runner.SessionAdmissionFeatures(1_000, 1_000, 0.0)
    gained = runner.SessionAdmissionFeatures(1_000, 1_000, 5.0)

    short_score = runner.session_admission_score(short, wait_s=0.0, **options)
    long_score = runner.session_admission_score(long, wait_s=0.0, **options)
    gained_score = runner.session_admission_score(gained, wait_s=0.0, **options)
    aged_score = runner.session_admission_score(long, wait_s=20.0, **options)
    assert short_score < long_score
    assert gained_score < long_score
    assert aged_score < long_score


def test_priority_pool_coalesces_and_holds_a_whole_session_slot() -> None:
    async def scenario() -> tuple[list[str], int, int]:
        pool = runner.AsyncSessionAdmissionPool(
            capacity=1,
            coalesce_s=0.02,
            prefill_tokens_per_s=10_000.0,
            decode_tokens_per_s=500.0,
            pressure_weight=1.0,
            tool_gain_beta=0.0,
            aging_alpha=0.0,
        )
        order: list[str] = []

        async def run(task_id: str, tokens: int) -> None:
            await pool.acquire(
                task_id,
                runner.SessionAdmissionFeatures(tokens, 100, 0.0),
            )
            order.append(task_id)
            assert pool.active == 1
            await asyncio.sleep(0.005)
            # The other task stays pending until this simulated whole session
            # releases its persistent slot.
            assert pool.active == 1
            await pool.release(task_id)

        await asyncio.gather(run("long", 1_000), run("short", 10))
        return order, pool.active, pool.pending

    order, active, pending = asyncio.run(scenario())
    assert order == ["short", "long"]
    assert active == 0
    assert pending == 0


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("coalesce_s", float("inf")),
        ("prefill_tokens_per_s", float("nan")),
        ("decode_tokens_per_s", float("inf")),
        ("pressure_weight", float("nan")),
        ("tool_gain_beta", float("-inf")),
        ("aging_alpha", float("nan")),
    ),
)
def test_priority_pool_rejects_nonfinite_options(
    option: str,
    value: float,
) -> None:
    options = {
        "capacity": 1,
        "coalesce_s": 0.0,
        "prefill_tokens_per_s": 10_000.0,
        "decode_tokens_per_s": 500.0,
        "pressure_weight": 1.0,
        "tool_gain_beta": 1.0,
        "aging_alpha": 0.1,
    }
    options[option] = value

    with pytest.raises(ValueError, match="must be finite"):
        runner.AsyncSessionAdmissionPool(**options)


def test_priority_pool_propagates_dispatch_failure_to_all_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_score(*_: Any, **__: Any) -> float:
        raise RuntimeError("injected admission score failure")

    monkeypatch.setattr(runner, "session_admission_score", fail_score)

    async def scenario() -> tuple[list[BaseException], int, int]:
        pool = runner.AsyncSessionAdmissionPool(
            capacity=2,
            coalesce_s=0.0,
            prefill_tokens_per_s=10_000.0,
            decode_tokens_per_s=500.0,
            pressure_weight=1.0,
            tool_gain_beta=1.0,
            aging_alpha=0.1,
        )
        waiters = [
            asyncio.create_task(
                pool.acquire(
                    task_id,
                    runner.SessionAdmissionFeatures(100, 100, 0.0),
                )
            )
            for task_id in ("first", "second")
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*waiters, return_exceptions=True),
            timeout=0.5,
        )
        return results, pool.active, pool.pending

    results, active, pending = asyncio.run(scenario())
    assert len(results) == 2
    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("injected admission score failure" in str(result) for result in results)
    assert active == 0
    assert pending == 0


def test_fifo_preengine_policy_is_backward_compatible_default() -> None:
    args = runner.parser().parse_args(
        [
            "run-cell",
            "--plan",
            "plan.json",
            "--system",
            "baseline",
            "--output",
            "result.json",
        ]
    )

    assert args.preengine_policy == "fifo"
    assert args.max_active_tasks == 80
    assert args.preengine_coalesce_s == pytest.approx(0.25)


def _score_feature(*, rlmt: int, realized_gain_s: float = 0.0) -> dict[str, Any]:
    return {
        "meta": {
            "t": "task",
            "c": 1,
            "rc": 1,
            "pt": 100,
            "mt": 10,
            "po": 10,
            "rtw": 0.0,
            "rlmt": rlmt,
            "eg": realized_gain_s,
        },
        "prompt_tokens": 100,
        "kv_tokens": 110,
        "cached_tokens": 0,
        "marginal_kv_tokens": 110,
        "next_tool_wait": 0.0,
        "prompt_len": 100,
        "max_tokens": 10,
        "waited_s": 0.0,
    }


def _score(feature: dict[str, Any]) -> float:
    score, _ = policy._joint_v2_score_s(
        feature,
        live_tokens=0.0,
        virtual_tokens=0.0,
        live_long_count=0,
        virtual_long_count=0,
        is_new_session=False,
    )
    return score


def test_remaining_llm_score_cost_is_opt_in_and_monotone(monkeypatch: pytest.MonkeyPatch) -> None:
    short = _score_feature(rlmt=100)
    long = _score_feature(rlmt=1000)

    monkeypatch.delenv("VLLM_SCHED_JOINT_V2_REMAINING_LLM_WEIGHT", raising=False)
    assert _score(short) == pytest.approx(_score(long))

    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REMAINING_LLM_WEIGHT", "1")
    assert _score(short) < _score(long)


def test_realized_tool_gain_bonus_survives_zero_next_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REMAINING_LLM_WEIGHT", "0")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REALIZED_GAIN_WEIGHT", "1")
    no_gain = _score_feature(rlmt=100, realized_gain_s=0.0)
    preserved_gain = _score_feature(rlmt=100, realized_gain_s=25.0)

    assert no_gain["next_tool_wait"] == preserved_gain["next_tool_wait"] == 0.0
    assert _score(preserved_gain) == pytest.approx(_score(no_gain) - 25.0)


class _Request:
    def __init__(
        self,
        metadata: dict[str, Any],
        *,
        arrival_time: float = 99.0,
        num_tokens: int = 100,
    ) -> None:
        self.request_id = runner.schedx_id(metadata)
        self.arrival_time = arrival_time
        self.num_tokens = num_tokens
        self.num_prompt_tokens = int(metadata.get("pt", 100))
        self.max_tokens = int(metadata.get("mt", 10))


class _Scheduler:
    def __init__(self, native_cap: int = 48) -> None:
        self.max_num_running_reqs = native_cap
        self.cache_config = SimpleNamespace(
            num_gpu_blocks=10_000,
            block_size=16,
        )
        self.kv_cache_manager = SimpleNamespace(usage=0.0)


def _request(task: str, *, call_index: int = 1) -> _Request:
    return _Request(
        {
            "t": task,
            "c": call_index,
            "i": call_index,
            "rc": 1,
            "pt": 100,
            "mt": 10,
            "po": 10,
            "nw": 0.0,
            "rtw": 0.0,
            "rlmt": 100,
        }
    )


def _enable_combined_physical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION", "1")
    monkeypatch.setenv(
        "VLLM_SCHED_JOINT_V2_PHYSICAL_RESPECT_JOINT_LIMITS", "1"
    )
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION", "1")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S", "0")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S", "0")


def test_physical_admission_respects_decode_band(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_combined_physical(monkeypatch)
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING", "24")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING", "25")
    scheduler = _Scheduler(native_cap=48)
    running = [_request(f"running-{index}") for index in range(20)]
    ordered = [_request(f"waiting-{index}") for index in range(10)]

    decision = policy._apply_joint_v2_physical_kv_admission(
        scheduler,
        ordered=ordered,
        admissible_count=10,
        running_items=running,
        prompt_len_fn=lambda item: item.num_prompt_tokens,
        reserved_kv=0.0,
        now_s=100.0,
    )

    assert decision["joint_limit_slots"] == 4
    assert decision["admit"] == 4
    assert decision["effective_cap"] == 24
    assert scheduler.max_num_running_reqs == 24


def test_foreground_gate_cannot_be_bypassed_by_physical_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_combined_physical(monkeypatch)
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS", "2")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING", "1")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING", "24")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING", "25")
    monkeypatch.setenv("VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS", "1000000")
    old_started = set(policy._v2_started_sessions)
    old_completed = set(policy._v2_completed_sessions)
    try:
        policy._v2_started_sessions.clear()
        policy._v2_started_sessions.update({"active-a", "active-b"})
        policy._v2_completed_sessions.clear()
        running = [_request("active-a")]
        cold = [_request("cold-a", call_index=0), _request("cold-b", call_index=0)]

        ordered, admissible_count, _ = policy._order_joint_pacer_v2_waiting(
            waiting_items=cold,
            running_items=running,
            now_s=100.0,
            prompt_len_fn=lambda item: item.num_prompt_tokens,
        )
        assert admissible_count == 0

        scheduler = _Scheduler(native_cap=48)
        decision = policy._apply_joint_v2_physical_kv_admission(
            scheduler,
            ordered=ordered,
            admissible_count=admissible_count,
            running_items=running,
            prompt_len_fn=lambda item: item.num_prompt_tokens,
            reserved_kv=0.0,
            now_s=100.0,
        )
        assert decision["reason"] == "joint_limit"
        assert decision["admit"] == 0
        assert decision["effective_cap"] == 1
        assert scheduler.max_num_running_reqs == 1
    finally:
        policy._v2_started_sessions.clear()
        policy._v2_started_sessions.update(old_started)
        policy._v2_completed_sessions.clear()
        policy._v2_completed_sessions.update(old_completed)


def _forecast_request(
    task: str,
    *,
    predicted_output_tokens: int,
    num_tokens: int,
    arrival_time: float,
) -> _Request:
    return _Request(
        {
            "t": task,
            "c": 1,
            "i": 1,
            "rc": 1,
            "pt": 100,
            "mt": predicted_output_tokens,
            "po": predicted_output_tokens,
            "nw": 0.0,
            "rtw": 0.0,
            "rlmt": predicted_output_tokens,
        },
        arrival_time=arrival_time,
        num_tokens=num_tokens,
    )


def test_aged_physical_rescue_counts_running_forecast_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_combined_physical(monkeypatch)
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION", "0.5")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S", "1")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING", "2")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING", "3")
    scheduler = _Scheduler(native_cap=48)
    scheduler.cache_config.num_gpu_blocks = 20
    scheduler.kv_cache_manager.usage = 0.35
    running = _forecast_request(
        "running-growth",
        predicted_output_tokens=100,
        num_tokens=112,
        arrival_time=99.0,
    )
    due = _forecast_request(
        "unsafe-aged",
        predicted_output_tokens=100,
        num_tokens=100,
        arrival_time=50.0,
    )

    decision = policy._apply_joint_v2_physical_kv_admission(
        scheduler,
        ordered=[due],
        admissible_count=1,
        running_items=[running],
        prompt_len_fn=lambda item: item.num_prompt_tokens,
        reserved_kv=0.0,
        now_s=100.0,
    )

    assert decision["capacity_tokens"] == 320
    assert decision["committed_tokens"] == 208
    assert decision["admit"] == 0
    assert decision["reason"] == "forecast_hold"
    assert scheduler.max_num_running_reqs == 1


def test_aged_physical_rescue_can_use_utilization_reserve_within_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_combined_physical(monkeypatch)
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION", "0.5")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S", "1")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING", "2")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING", "3")
    scheduler = _Scheduler(native_cap=48)
    scheduler.cache_config.num_gpu_blocks = 20
    scheduler.kv_cache_manager.usage = 0.35
    running = _forecast_request(
        "running-no-growth",
        predicted_output_tokens=10,
        num_tokens=112,
        arrival_time=99.0,
    )
    due = _forecast_request(
        "safe-aged",
        predicted_output_tokens=50,
        num_tokens=100,
        arrival_time=50.0,
    )

    decision = policy._apply_joint_v2_physical_kv_admission(
        scheduler,
        ordered=[due],
        admissible_count=1,
        running_items=[running],
        prompt_len_fn=lambda item: item.num_prompt_tokens,
        reserved_kv=0.0,
        now_s=100.0,
    )

    assert decision["budget_tokens"] == 160
    assert decision["committed_tokens"] == 112
    assert decision["predicted_admit_tokens"] == 160
    assert decision["committed_tokens"] + decision["predicted_admit_tokens"] <= 320
    assert decision["admit"] == 1
    assert decision["reason"] == "aged_rescue"
    assert decision["rescue"] == 1
    assert scheduler.max_num_running_reqs == 2


def test_idle_progress_never_exceeds_physical_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_combined_physical(monkeypatch)
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION", "0.5")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING", "2")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING", "3")
    scheduler = _Scheduler(native_cap=48)
    scheduler.cache_config.num_gpu_blocks = 20
    too_large = _forecast_request(
        "too-large-idle",
        predicted_output_tokens=230,
        num_tokens=100,
        arrival_time=99.0,
    )

    decision = policy._apply_joint_v2_physical_kv_admission(
        scheduler,
        ordered=[too_large],
        admissible_count=1,
        running_items=[],
        prompt_len_fn=lambda item: item.num_prompt_tokens,
        reserved_kv=0.0,
        now_s=100.0,
    )

    assert decision["capacity_tokens"] == 320
    assert decision["admit"] == 0
    assert decision["reason"] == "forecast_hold"
    assert scheduler.max_num_running_reqs == 0


def test_idle_progress_counts_committed_return_reservations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_combined_physical(monkeypatch)
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION", "0.5")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING", "2")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING", "3")
    scheduler = _Scheduler(native_cap=48)
    scheduler.cache_config.num_gpu_blocks = 20
    waiting = _forecast_request(
        "idle-with-return-reserve",
        predicted_output_tokens=50,
        num_tokens=100,
        arrival_time=99.0,
    )

    decision = policy._apply_joint_v2_physical_kv_admission(
        scheduler,
        ordered=[waiting],
        admissible_count=1,
        running_items=[],
        prompt_len_fn=lambda item: item.num_prompt_tokens,
        reserved_kv=176.0,
        now_s=100.0,
    )

    assert decision["capacity_tokens"] == 320
    assert decision["committed_tokens"] == 176
    assert decision["admit"] == 0
    assert decision["reason"] == "forecast_hold"
    assert scheduler.max_num_running_reqs == 0


def test_compare_rejects_bad_hook_evidence_without_success_claim(
    tmp_path: Path,
) -> None:
    summary = {
        "tasks": 1,
        "successful_tasks": 1,
        "llm_requests": 1,
        "tool_calls": 0,
        "mean_e2e_s": 2.0,
        "p50_e2e_s": 2.0,
        "p95_e2e_s": 2.0,
        "makespan_s": 2.0,
        "mean_llm_request_s": 1.0,
        "offline_url_hits": 0,
        "saved_tool_service_s": 0.0,
        "executed_tool_service_s": 0.0,
    }
    common = {
        "schema": runner.RESULT_SCHEMA,
        "plan_sha256": "frozen-plan",
        "model": "model",
        "summary": summary,
        "tasks": [{"task_id": "task", "e2e_s": 2.0}],
        "llm_events": [
            {
                "task_id": "task",
                "request_index": 0,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                "http_status": 200,
            }
        ],
        "tool_events": [],
    }
    baseline = {**common, "system": "baseline"}
    full = {
        **common,
        "system": "full",
        "settings": {"tool_capacity": 1},
    }
    baseline_path = tmp_path / "baseline.json"
    full_path = tmp_path / "full.json"
    runner.write_json(baseline_path, baseline)
    runner.write_json(full_path, full)
    http_line = 'POST /v1/chat/completions HTTP/1.1" 200\n'
    install_line = "[sched_policy_patch] installed policy=online_joint_pacer_v2\n"
    baseline_log = tmp_path / "baseline.log"
    full_log = tmp_path / "full.log"
    baseline_log.write_text(
        "engine {'max_num_seqs': 48}\n" + http_line + install_line,
        encoding="utf-8",
    )
    full_log.write_text(
        "engine {'max_num_seqs': 48}\n" + http_line + install_line + "fail_open\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"
    markdown = tmp_path / "report.md"

    result = runner.compare(
        SimpleNamespace(
            baseline=baseline_path,
            full=full_path,
            baseline_server_log=baseline_log,
            full_server_log=full_log,
            output=output,
            markdown=markdown,
        )
    )

    report = runner.read_json(output)
    rendered = markdown.read_text(encoding="utf-8").lower()
    assert result == 2
    assert report["valid"] is False
    assert report["validity"]["baseline_joint_hook_absent"] is False
    assert report["validity"]["full_fail_open_free"] is False
    assert "validation: **fail**" in rendered
    assert "checks passed" not in rendered


def test_joint_v2_score_snapshot_is_bitwise_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REMAINING_LLM_WEIGHT", "0.73")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REALIZED_GAIN_WEIGHT", "1.25")
    feature = _score_feature(rlmt=987, realized_gain_s=13.5)
    feature.update(
        {
            "cached_tokens": 48,
            "marginal_kv_tokens": 72,
            "waited_s": 7.25,
        }
    )
    dynamic = {
        "live_tokens": 12_345.0,
        "virtual_tokens": 6_789.0,
        "live_long_count": 1,
        "virtual_long_count": 2,
        "is_new_session": True,
    }

    implicit = policy._joint_v2_score_s(feature, **dynamic)
    snapshot = policy._joint_v2_score_config_snapshot()
    explicit = policy._joint_v2_score_s(
        feature,
        score_config=snapshot,
        **dynamic,
    )

    assert explicit == implicit


def test_joint_v2_reads_score_knobs_once_per_scheduler_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked = (
        "_joint_v2_prefix_locality_weight",
        "_joint_v2_remaining_call_soft_weight_s",
        "_hbm_target_context_tokens",
        "_hbm_virtual_fill_ratio",
        "_prefill_tokens_per_s_v2",
        "_decode_tokens_per_s_v2",
        "_default_predicted_output_tokens",
        "_oas_v3_context_tokens_per_s",
        "_joint_v2_context_alpha",
        "_joint_v2_tool_wait_cap_s",
        "_avg_call_service_s",
        "_joint_v2_remaining_tool_weight",
        "_joint_v2_context_ref_tokens",
        "_joint_v2_final_bonus_s",
        "_joint_v2_progress_bonus_s",
        "_joint_v2_tool_beta",
        "_joint_v2_remaining_llm_weight",
        "_joint_v2_realized_gain_weight",
        "_hbm_long_context_tokens",
        "_hbm_max_long_running",
        "_joint_v2_over_budget_penalty_s",
        "_joint_v2_new_session_penalty_s",
        "_joint_v2_tail_beta",
        "_time_aging_alpha",
    )
    calls = {name: 0 for name in tracked}
    for name in tracked:
        original = getattr(policy, name)

        def counted(
            *,
            _name: str = name,
            _original: Any = original,
        ) -> Any:
            calls[_name] += 1
            return _original()

        monkeypatch.setattr(policy, name, counted)

    waiting = [_request(f"waiting-{index}") for index in range(48)]
    policy._order_joint_pacer_v2_waiting(
        waiting_items=waiting,
        running_items=[],
        now_s=100.0,
        prompt_len_fn=lambda item: item.num_prompt_tokens,
    )

    assert calls == {name: 1 for name in tracked}


def test_joint_v2_skips_unused_legacy_tool_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_tool_key(**_: Any) -> float:
        raise AssertionError("Joint-v2 must not build the unused legacy key")

    monkeypatch.setattr(policy, "_joint_tool_queue_key_s", unexpected_tool_key)
    waiting = [_request(f"waiting-{index}") for index in range(8)]

    ordered, admissible_count, _ = policy._order_joint_pacer_v2_waiting(
        waiting_items=waiting,
        running_items=[],
        now_s=100.0,
        prompt_len_fn=lambda item: item.num_prompt_tokens,
    )

    assert len(ordered) == len(waiting)
    assert admissible_count > 0


def test_joint_v2_score_config_is_refreshed_between_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_TOOL_BETA", "0.25")
    first = policy._joint_v2_score_config_snapshot()
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_TOOL_BETA", "1.75")
    second = policy._joint_v2_score_config_snapshot()

    assert first.tool_beta == 0.25
    assert second.tool_beta == 1.75


def test_strict_causal_schema_ignores_poisoned_legacy_oracle_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict decisions must be invariant to every old trace-tail spelling."""

    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S", "1")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REMAINING_LLM_WEIGHT", "1")
    monkeypatch.setenv("VLLM_SCHED_JOINT_V2_REALIZED_GAIN_WEIGHT", "1")
    causal = {
        "ms": policy._STRICT_CAUSAL_METADATA_SCHEMA,
        "t": "strict-task",
        "c": 1,
        "i": 1,
        "pt": 100,
        "mt": 32,
        "po_hat": 12,
        "remaining_calls_hat": 2,
        "remaining_llm_tokens_hat": 30,
        "tool_eta_s_hat": 3.0,
        "remaining_tool_wait_s_hat": 5.0,
        "tool_hit_probability_hat": 0.4,
        "expected_gain_s_hat": 0.5,
    }
    poison = {
        **causal,
        "n": 0,
        "rc": 0,
        "rlmt": 10**9,
        "npt": 10**9,
        "nmt": 10**9,
        "nw": 10**9,
        "nwc": 1.0,
        "rtw": 10**9,
        "eg": 10**9,
        "is_final": True,
    }

    def feature(meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "meta": meta,
            "prompt_tokens": 100,
            "kv_tokens": 132,
            "cached_tokens": 0,
            "marginal_kv_tokens": 132,
            "next_tool_wait": policy._causal_meta_float(
                meta,
                predicted_key="tool_eta_s_hat",
                legacy_key="nw",
                default=0.0,
            ),
            "prompt_len": 100,
            "max_tokens": 32,
            "waited_s": 0.0,
        }

    clean_score, clean_over = policy._joint_v2_score_s(
        feature(causal),
        live_tokens=0.0,
        virtual_tokens=0.0,
        live_long_count=0,
        virtual_long_count=0,
        is_new_session=False,
    )
    poison_score, poison_over = policy._joint_v2_score_s(
        feature(poison),
        live_tokens=0.0,
        virtual_tokens=0.0,
        live_long_count=0,
        virtual_long_count=0,
        is_new_session=False,
    )

    assert policy._joint_v2_soft_remaining_calls(poison) == 2
    assert poison_score == pytest.approx(clean_score)
    assert poison_over is clean_over
