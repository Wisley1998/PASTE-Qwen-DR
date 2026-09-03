"""Constrained Murakkab planning and evidence checks for the fixed PASTE setup.

This module intentionally contains no alternative model, hardware, scaling,
workflow, or SLO choice.  It validates a typed workflow and a registry, proves
that the candidate set is a singleton, and checks that the existing live
closed-loop runner executed the selected workflow dependency by dependency.

The live implementation remains :mod:`paste_repro.live_agent`; this module is
an outside-the-timed-path control plane and evidence layer.  In particular it
does not reinterpret PASTE's speculation width as a Murakkab optimization.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCHEMA = "paste_repro.murakkab_fixed_runtime"
VERSION = 1
WORKFLOW_ID = "tongyi_deepresearch_fixed_linear_v1"
SELECTED_CANDIDATE_ID = "tongyi-30b-tp4-a100x4-singleton"


class MurakkabFixedError(RuntimeError):
    """A fixed-setup plan or live result violated its registered contract."""


@dataclass(frozen=True)
class FixedWorkflowNode:
    node_id: str
    executor: str
    depends_on: tuple[str, ...]
    input_types: dict[str, str]
    output_type: str


@dataclass(frozen=True)
class FixedTypedWorkflow:
    """Minimal typed-DAG representation used only by the fixed M path."""

    workflow_id: str
    description: str
    nodes: tuple[FixedWorkflowNode, ...]
    topological_order: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FixedTypedWorkflow":
        workflow_id = raw.get("id")
        description = raw.get("description")
        nodes_raw = raw.get("nodes")
        if not isinstance(workflow_id, str) or not workflow_id:
            raise MurakkabFixedError("workflow id must be a non-empty string")
        if not isinstance(description, str):
            raise MurakkabFixedError("workflow description must be a string")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise MurakkabFixedError("workflow nodes must be a non-empty list")
        nodes: list[FixedWorkflowNode] = []
        for raw_node in nodes_raw:
            if not isinstance(raw_node, Mapping):
                raise MurakkabFixedError("workflow node must be an object")
            node_id = raw_node.get("id")
            executor = raw_node.get("executor")
            dependencies = raw_node.get("depends_on")
            inputs = raw_node.get("input_types")
            output = raw_node.get("output_type")
            if not isinstance(node_id, str) or not node_id:
                raise MurakkabFixedError("workflow node id must be a string")
            if not isinstance(executor, str) or not executor:
                raise MurakkabFixedError(f"{node_id}: executor must be a string")
            if not isinstance(dependencies, list) or not all(
                isinstance(value, str) and value for value in dependencies
            ):
                raise MurakkabFixedError(f"{node_id}: dependencies are invalid")
            if not isinstance(inputs, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in inputs.items()
            ):
                raise MurakkabFixedError(f"{node_id}: input types are invalid")
            if not isinstance(output, str) or not output:
                raise MurakkabFixedError(f"{node_id}: output type is invalid")
            nodes.append(
                FixedWorkflowNode(
                    node_id=node_id, executor=executor,
                    depends_on=tuple(dependencies), input_types=dict(inputs),
                    output_type=output,
                )
            )
        by_id = {node.node_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise MurakkabFixedError("workflow node ids must be unique")
        for node in nodes:
            if set(node.input_types) != set(node.depends_on):
                raise MurakkabFixedError(
                    f"{node.node_id}: dependency and input-type keys differ"
                )
            for dependency in node.depends_on:
                source = by_id.get(dependency)
                if source is None:
                    raise MurakkabFixedError(
                        f"{node.node_id}: unknown dependency {dependency!r}"
                    )
                expected = node.input_types[dependency]
                if source.output_type != expected:
                    raise MurakkabFixedError(
                        f"{node.node_id}: {dependency!r} emits {source.output_type!r}, "
                        f"expected {expected!r}"
                    )
        remaining = {node.node_id: set(node.depends_on) for node in nodes}
        order: list[str] = []
        while remaining:
            ready = sorted(node_id for node_id, deps in remaining.items() if not deps)
            if not ready:
                raise MurakkabFixedError("workflow contains a dependency cycle")
            for node_id in ready:
                order.append(node_id)
                remaining.pop(node_id)
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        return cls(workflow_id, description, tuple(nodes), tuple(order))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.workflow_id,
            "description": self.description,
            "nodes": [asdict(node) for node in self.nodes],
            "topological_order": list(self.topological_order),
            "type_checked": True,
        }


@dataclass(frozen=True)
class ExecutorRegistration:
    executor_id: str
    kind: str
    implementation: str
    output_type: str


@dataclass(frozen=True)
class FixedExecutableConfiguration:
    candidate_id: str
    model: str
    model_revision: str
    serving_engine: str
    dtype: str
    tensor_parallelism: int
    gpu_count: int
    gpu_type: str
    replicas: int
    scheduler: str
    tool_execution: str


FIXED_WORKFLOW: dict[str, Any] = {
    "id": WORKFLOW_ID,
    "description": (
        "Fixed Tongyi DeepResearch workflow; every node becomes ready only "
        "after its authoritative predecessor has completed."
    ),
    "nodes": [
        {
            "id": "initial_llm",
            "executor": "tongyi_guided_search_call",
            "depends_on": [],
            "input_types": {},
            "output_type": "search_invocation",
        },
        {
            "id": "search",
            "executor": "bing_html_search",
            "depends_on": ["initial_llm"],
            "input_types": {"initial_llm": "search_invocation"},
            "output_type": "search_results",
        },
        {
            "id": "decision_llm",
            "executor": "tongyi_guided_visit_call",
            "depends_on": ["search"],
            "input_types": {"search": "search_results"},
            "output_type": "visit_invocation",
        },
        {
            "id": "visit",
            "executor": "jina_visit",
            "depends_on": ["decision_llm"],
            "input_types": {"decision_llm": "visit_invocation"},
            "output_type": "visit_result",
        },
        {
            "id": "synthesis_llm",
            "executor": "tongyi_fixed_final_answer",
            "depends_on": ["visit"],
            "input_types": {"visit": "visit_result"},
            "output_type": "grounded_answer",
        },
    ],
}

FIXED_REGISTRY: tuple[ExecutorRegistration, ...] = (
    ExecutorRegistration(
        "tongyi_guided_search_call", "llm", "LiveClosedLoopExperiment.call[0]",
        "search_invocation",
    ),
    ExecutorRegistration(
        "bing_html_search", "tool", "WikipediaLiveExecutor.search(bing)",
        "search_results",
    ),
    ExecutorRegistration(
        "tongyi_guided_visit_call", "llm", "LiveClosedLoopExperiment.call[1]",
        "visit_invocation",
    ),
    ExecutorRegistration(
        "jina_visit", "tool", "WikipediaLiveExecutor.visit(jina)",
        "visit_result",
    ),
    ExecutorRegistration(
        "tongyi_fixed_final_answer", "llm", "LiveClosedLoopExperiment.call[2]",
        "grounded_answer",
    ),
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MurakkabFixedError(f"{label} must be an object")
    return value


def _validate_registry(workflow: FixedTypedWorkflow) -> list[dict[str, Any]]:
    by_id = {entry.executor_id: entry for entry in FIXED_REGISTRY}
    if len(by_id) != len(FIXED_REGISTRY):
        raise MurakkabFixedError("executor registry contains duplicate ids")
    used: set[str] = set()
    for node in workflow.nodes:
        registration = by_id.get(node.executor)
        if registration is None:
            raise MurakkabFixedError(
                f"workflow node {node.node_id!r} has no registered executor"
            )
        if registration.output_type != node.output_type:
            raise MurakkabFixedError(
                f"executor {node.executor!r} emits {registration.output_type!r}, "
                f"but node {node.node_id!r} declares {node.output_type!r}"
            )
        used.add(node.executor)
    unused = sorted(set(by_id) - used)
    if unused:
        raise MurakkabFixedError(f"executor registry has unused entries: {unused}")
    return [asdict(entry) for entry in FIXED_REGISTRY]


def build_singleton_plan(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the constrained protocol and return its sole executable plan."""

    shared = _require_mapping(protocol.get("shared_setup"), "shared_setup")
    constrained = _require_mapping(
        protocol.get("constrained_murakkab"), "constrained_murakkab"
    )
    singleton_fields = (
        "workflow_configuration_candidates",
        "model_candidates",
        "hardware_candidates",
        "parallelism_candidates",
        "replica_count_candidates",
        "slo_tiers",
    )
    changed = {
        key: constrained.get(key)
        for key in singleton_fields
        if constrained.get(key) != 1
    }
    if changed:
        raise MurakkabFixedError(
            f"constrained Murakkab candidate dimensions are not singleton: {changed}"
        )
    if constrained.get("scale_actions_allowed") is not False:
        raise MurakkabFixedError("scale actions must be disabled")
    if constrained.get("reconfiguration_allowed") is not False:
        raise MurakkabFixedError("reconfiguration must be disabled")

    gpu = _require_mapping(shared.get("gpu"), "shared_setup.gpu")
    candidate = FixedExecutableConfiguration(
        candidate_id=SELECTED_CANDIDATE_ID,
        model=str(shared.get("model")),
        model_revision=str(shared.get("model_revision")),
        serving_engine=str(shared.get("serving_engine")),
        dtype=str(shared.get("dtype")),
        tensor_parallelism=int(shared.get("tensor_parallelism")),
        gpu_count=int(gpu.get("count")),
        gpu_type=str(gpu.get("type")),
        replicas=int(shared.get("replicas")),
        scheduler="native_fcfs",
        tool_execution="demand_only",
    )
    expected_candidate = FixedExecutableConfiguration(
        candidate_id=SELECTED_CANDIDATE_ID,
        model="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        model_revision="4b0ac5767427a55d08a254f0367e2934976598e0",
        serving_engine="vLLM 0.10.1",
        dtype="bf16",
        tensor_parallelism=4,
        gpu_count=4,
        gpu_type="NVIDIA A100-SXM4-40GB",
        replicas=1,
        scheduler="native_fcfs",
        tool_execution="demand_only",
    )
    if candidate != expected_candidate:
        raise MurakkabFixedError(
            "protocol shared setup is not the registered fixed PASTE deployment"
        )

    workflow = FixedTypedWorkflow.from_mapping(FIXED_WORKFLOW)
    expected_order = (
        "initial_llm", "search", "decision_llm", "visit", "synthesis_llm"
    )
    if workflow.topological_order != expected_order:
        raise MurakkabFixedError("fixed workflow is not the registered linear DAG")
    registry = _validate_registry(workflow)
    candidate_dict = asdict(candidate)
    registry_sha = canonical_sha256(registry)
    workflow_dict = workflow.to_dict()
    workflow_sha = canonical_sha256(workflow_dict)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "planner": "singleton_constrained_selection",
        "optimizer_outside_timed_path": True,
        "candidate_count": 1,
        "candidate_dimensions": {key: 1 for key in singleton_fields},
        "objective_evaluated": False,
        "selection_reason": "the registered candidate set contains exactly one item",
        "selected_candidate_id": SELECTED_CANDIDATE_ID,
        "selected_candidate": candidate_dict,
        "workflow": workflow_dict,
        "workflow_sha256": workflow_sha,
        "executor_registry": registry,
        "registry_sha256": registry_sha,
        "typed_dag_validated": True,
        "dependency_ready_dispatch": True,
        "forbidden_optimizations_enabled": [],
    }


EXPECTED_RUNTIME_CONFIG: dict[str, Any] = {
    "model": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "call_graph_mode": "autonomous",
    "speculation_mode": "off",
    "tool_signal_policy": "execution_aware",
    "visit_top_k": 1,
    "independent_source_count": 80,
    "replicas": 1,
    "task_count": 80,
    "max_active_tasks": 80,
    "tool_workers": 4,
    "speculative_tool_workers": 2,
    "min_speculative_tool_workers": 0,
    "search_tool_capacity": 3,
    "visit_tool_capacity": 2,
    "search_min_start_interval_s": 0.0,
    "visit_min_start_interval_s": 2.5,
    "max_speculative_pending": 128,
    "speculative_ttl_s": 120.0,
    "tool_http_max_attempts": 2,
    "tool_http_retry_backoff_s": 1.0,
    "tool_http_attempt_start_gate_enabled": True,
    "visit_mode": "jina",
    "search_mode": "bing",
    "search_max_results": 5,
    "visit_max_chars": 3000,
    "max_tokens_tool": 128,
    "max_tokens_answer": 256,
    "fixed_final_completion_tokens": 192,
    "context_padding_tokens": 10000,
    "live_tool_execution": True,
    "recorded_tool_sleep": False,
    "controlled_http_retry": True,
    "shared_bounded_tool_pool": True,
    "generated_tool_call_controls_next_prompt": True,
    "authoritative_and_speculative_share_capacity": True,
    "future_trace_oracle_used": False,
    "frozen_url_is_workload_input": False,
}

EXPECTED_SCHEDULER_ENV: dict[str, str] = {
    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
    "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
    "VLLM_PORT": "8100",
    "VLLM_TP_SIZE": "4",
    "VLLM_DTYPE": "bfloat16",
    "VLLM_MAX_MODEL_LEN": "16384",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.86",
    "VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
    "VLLM_MAX_NUM_SEQS": "96",
    "VLLM_CUDA_GRAPH_SIZES": "32",
    "VLLM_ENABLE_PREFIX_CACHING": "1",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
    "VLLM_USE_V1": "1",
    "VLLM_SCHED_POLICY": "fcfs",
}


def _as_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MurakkabFixedError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise MurakkabFixedError(f"{label} must be finite")
    return value


def validate_dependency_dispatch(result: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the observed calls respected the registered linear dependencies."""

    tasks = result.get("tasks")
    events = result.get("llm_events")
    records = result.get("tool_attempt_records")
    if not isinstance(tasks, list) or not isinstance(events, list) or not isinstance(records, list):
        raise MurakkabFixedError("live result lacks task, LLM, or tool evidence")

    events_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, Mapping) or not isinstance(event.get("task_id"), str):
            raise MurakkabFixedError("LLM event has no task id")
        events_by_task[str(event["task_id"])].append(event)
    records_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("session_id"), str):
            raise MurakkabFixedError("tool attempt has no session id")
        if record.get("admitted") is True:
            records_by_task[str(record["session_id"])].append(record)

    minimum_slack_s = math.inf
    checked = 0
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("ok") is not True:
            raise MurakkabFixedError("dependency proof requires every task to succeed")
        task_id = task.get("task_id")
        if not isinstance(task_id, str):
            raise MurakkabFixedError("successful task has no task id")
        task_events = sorted(
            events_by_task.get(task_id, []), key=lambda row: int(row.get("call_index", -1))
        )
        if [row.get("call_index") for row in task_events] != [0, 1, 2]:
            raise MurakkabFixedError(f"{task_id}: expected exactly LLM calls 0,1,2")
        task_records = records_by_task.get(task_id, [])
        if len(task_records) != 2:
            raise MurakkabFixedError(f"{task_id}: expected exactly two tool executions")
        by_tool = {str(row.get("tool")): row for row in task_records}
        if set(by_tool) != {"search", "visit"}:
            raise MurakkabFixedError(f"{task_id}: expected search and visit tools")
        for tool_name, row in by_tool.items():
            if row.get("speculative") is not False:
                raise MurakkabFixedError(
                    f"{task_id}/{tool_name}: M cell executed speculative work"
                )
            if (
                row.get("authoritative") is not True
                or row.get("source") != "executed"
                or row.get("committed") is not True
            ):
                raise MurakkabFixedError(f"{task_id}/{tool_name}: tool was not committed")

        event_end = [
            _as_finite_float(row.get("request_start_monotonic_s"), "LLM start")
            + _as_finite_float(row.get("duration_s"), "LLM duration")
            for row in task_events
        ]
        event_start = [
            _as_finite_float(row.get("request_start_monotonic_s"), "LLM start")
            for row in task_events
        ]
        search_start = _as_finite_float(by_tool["search"].get("queue_enter_at"), "search queue")
        search_finish = _as_finite_float(by_tool["search"].get("finished_at"), "search finish")
        visit_start = _as_finite_float(by_tool["visit"].get("queue_enter_at"), "visit queue")
        visit_finish = _as_finite_float(by_tool["visit"].get("finished_at"), "visit finish")
        slacks = (
            search_start - event_end[0],
            event_start[1] - search_finish,
            visit_start - event_end[1],
            event_start[2] - visit_finish,
        )
        if min(slacks) < -1e-6:
            raise MurakkabFixedError(
                f"{task_id}: observed execution violated a DAG dependency: {slacks}"
            )
        minimum_slack_s = min(minimum_slack_s, *slacks)
        checked += 1

    return {
        "validated": True,
        "task_count": checked,
        "expected_topological_order": [
            "initial_llm", "search", "decision_llm", "visit", "synthesis_llm"
        ],
        "observed_llm_calls_per_task": 3,
        "observed_authoritative_tools_per_task": 2,
        "minimum_dependency_slack_s": minimum_slack_s if checked else None,
        "speculative_tool_execution_observed": False,
    }


def validate_live_result(
    result: Mapping[str, Any], *, call_graph_mode: str = "autonomous",
    expected_task_count: int = 80,
) -> dict[str, Any]:
    """Fail closed unless a raw run is exactly the fixed M treatment."""

    config = _require_mapping(result.get("config"), "config")
    expected = dict(EXPECTED_RUNTIME_CONFIG)
    if call_graph_mode not in {"autonomous", "frozen"}:
        raise MurakkabFixedError("call_graph_mode must be autonomous or frozen")
    expected["call_graph_mode"] = call_graph_mode
    expected["frozen_url_is_workload_input"] = call_graph_mode == "frozen"
    if isinstance(expected_task_count, bool) or expected_task_count <= 0:
        raise MurakkabFixedError("expected_task_count must be positive")
    expected["independent_source_count"] = expected_task_count
    expected["task_count"] = expected_task_count
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise MurakkabFixedError(f"fixed M runtime mismatch: {mismatches}")
    fixed_workload_fields = {
        "workload_file_sha256": (
            "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20"
        ),
        "workload_split_id": "live-joint-wikipedia-frozen-formal-v9",
        "workload_formal_eligible": True,
    }
    workload_mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in fixed_workload_fields.items()
        if config.get(key) != value
    }
    if workload_mismatches:
        raise MurakkabFixedError(
            f"fixed-v9 workload identity mismatch: {workload_mismatches}"
        )
    if (
        expected_task_count == 80
        and config.get("selected_workload_sha256")
        != "750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c"
    ):
        raise MurakkabFixedError("fixed-v9 selected-workload SHA256 mismatch")

    scheduler = _require_mapping(config.get("scheduler_environment"), "scheduler_environment")
    scheduler_mismatches = {
        key: {"expected": value, "observed": scheduler.get(key)}
        for key, value in EXPECTED_SCHEDULER_ENV.items()
        if scheduler.get(key) != value
    }
    extension_leaks = {
        key: value
        for key, value in scheduler.items()
        if key.startswith("VLLM_SCHED_")
        and key != "VLLM_SCHED_POLICY"
        and value is not None
    }
    if scheduler_mismatches or extension_leaks:
        raise MurakkabFixedError(
            f"native-FCFS environment mismatch: expected={scheduler_mismatches}, "
            f"extension_leaks={extension_leaks}"
        )

    summary = _require_mapping(result.get("summary"), "summary")
    llm = _require_mapping(summary.get("llm"), "summary.llm")
    tool = _require_mapping(summary.get("tool"), "summary.tool")
    broker = _require_mapping(tool.get("broker_stats"), "broker_stats")
    required_counts = {
        "task_count": expected_task_count,
        "successful_task_count": expected_task_count,
        "failed_task_count": 0,
    }
    count_mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in required_counts.items()
        if summary.get(key) != value
    }
    if (
        count_mismatches
        or summary.get("all_tasks_succeeded") is not True
        or llm.get("request_count") != 3 * expected_task_count
        or llm.get("successful_request_count") != 3 * expected_task_count
        or llm.get("exactly_one_attempt_each") is not True
        or broker.get("authoritative_requests") != 2 * expected_task_count
        or broker.get("commits") != 2 * expected_task_count
        or broker.get("authoritative_failures") != 0
    ):
        raise MurakkabFixedError("M cell completion/integrity gate failed")
    speculative_nonzero = {
        key: broker.get(key)
        for key in (
            "speculative_admitted", "speculative_started", "speculative_completed",
            "speculative_failures", "queued_promotions", "running_promotions",
            "completed_reuse", "wasted_speculative_service_s",
        )
        if broker.get(key) not in {0, 0.0}
    }
    if speculative_nonzero:
        raise MurakkabFixedError(
            f"M demand-only cell has speculative activity: {speculative_nonzero}"
        )
    records = result.get("tool_attempt_records")
    if not isinstance(records, list) or len(records) != 2 * expected_task_count:
        raise MurakkabFixedError("M cell physical tool record count is not 2 per task")
    invalid_attempts = [
        index
        for index, record in enumerate(records)
        if not isinstance(record, Mapping)
        or record.get("http_attempts") != 1
        or record.get("speculative") is not False
        or record.get("authoritative") is not True
        or record.get("source") != "executed"
        or record.get("committed") is not True
    ]
    if invalid_attempts:
        raise MurakkabFixedError(
            f"M physical tool/retry contract failed for {len(invalid_attempts)} records"
        )
    snapshot = result.get("broker_final_snapshot")
    counts = snapshot.get("counts") if isinstance(snapshot, Mapping) else None
    drained_keys = (
        "completed_unclaimed_speculative", "queued_authoritative",
        "queued_speculative", "running_authoritative", "running_speculative",
    )
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("jobs") != []
        or not isinstance(counts, Mapping)
        or any(counts.get(key) != 0 for key in drained_keys)
        or counts.get("queued_by_tool") != {}
        or counts.get("running_by_tool") != {}
    ):
        raise MurakkabFixedError("M cell ended with a non-drained tool broker")
    return validate_dependency_dispatch(result)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise MurakkabFixedError("cannot summarize an empty metric")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    cleaned = [float(value) for value in values]
    if not cleaned or any(not math.isfinite(value) for value in cleaned):
        raise MurakkabFixedError("distribution requires finite observations")
    return {
        "count": len(cleaned),
        "mean_s": statistics.fmean(cleaned),
        "p50_s": percentile(cleaned, 0.50),
        "p95_s": percentile(cleaned, 0.95),
        "p99_s": percentile(cleaned, 0.99),
        "max_s": max(cleaned),
    }


def compute_fixed_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Compute task throughput, latency, completion time, and components."""

    config = _require_mapping(result.get("config"), "config")
    mode = str(config.get("call_graph_mode"))
    expected_task_count = int(config.get("task_count", 0))
    dependency_evidence = validate_live_result(
        result, call_graph_mode=mode, expected_task_count=expected_task_count
    )
    tasks = [dict(row) for row in result["tasks"]]
    events = [dict(row) for row in result["llm_events"]]
    records = [dict(row) for row in result["tool_attempt_records"]]
    makespan = _as_finite_float(
        result.get("task_completion_makespan_s"), "task completion makespan"
    )
    if makespan <= 0:
        raise MurakkabFixedError("task completion makespan must be positive")
    e2e = [_as_finite_float(task.get("e2e_s"), "task e2e") for task in tasks]
    release = min(_as_finite_float(task.get("start_wall_s"), "task start") for task in tasks)
    completion_offsets = [
        _as_finite_float(task.get("end_wall_s"), "task end") - release
        for task in tasks
    ]
    llm_by_call: dict[int, list[float]] = defaultdict(list)
    for event in events:
        llm_by_call[int(event["call_index"])].append(
            _as_finite_float(event.get("duration_s"), "LLM duration")
        )
    tools_by_name: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"exposed_wait": [], "queue": [], "service": []}
    )
    task_residuals: list[float] = []
    for task in tasks:
        task_tool_wait = 0.0
        for row in task.get("tools", []):
            invocation = _require_mapping(row.get("invocation"), "tool invocation")
            name = str(invocation.get("tool_name"))
            for source, target in (
                ("exposed_wait_s", "exposed_wait"),
                ("queue_s", "queue"),
                ("service_s", "service"),
            ):
                value = _as_finite_float(row.get(source, 0.0), source)
                tools_by_name[name][target].append(value)
                if source == "exposed_wait_s":
                    task_tool_wait += value
        llm_s = _as_finite_float(task.get("llm_duration_s"), "task LLM duration")
        task_residuals.append(max(0.0, float(task["e2e_s"]) - llm_s - task_tool_wait))
    physical_attempts = sum(int(row.get("http_attempts", 0) or 0) for row in records)
    physical_service = sum(
        _as_finite_float(row.get("service_s", 0.0), "physical service")
        for row in records
    )
    completion_tokens = sum(
        int(event.get("usage", {}).get("completion_tokens", 0) or 0)
        for event in events
    )
    prompt_tokens = sum(
        int(event.get("usage", {}).get("prompt_tokens", 0) or 0)
        for event in events
    )
    return {
        "schema": "paste_repro.murakkab_fixed_live_metrics",
        "version": 1,
        "cell_id": "M",
        "call_graph_mode": mode,
        "task_count": len(tasks),
        "successful_task_count": len(tasks),
        "task_completion_makespan_s": makespan,
        "throughput": {
            "completed_tasks_per_s": len(tasks) / makespan,
            "completed_tasks_per_min": 60.0 * len(tasks) / makespan,
            "llm_requests_per_s": len(events) / makespan,
            "completion_tokens_per_s": completion_tokens / makespan,
        },
        "task_e2e": distribution(e2e),
        "task_completion_offset_from_first_release": distribution(completion_offsets),
        "llm": {
            "request_count": len(events),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "request_latency_by_call": {
                str(index): distribution(llm_by_call[index]) for index in sorted(llm_by_call)
            },
            "task_total_llm_latency": distribution(
                [_as_finite_float(task.get("llm_duration_s"), "task LLM duration") for task in tasks]
            ),
        },
        "tool": {
            "authoritative_commit_count": sum(len(task.get("tools", [])) for task in tasks),
            "physical_execution_count": len(records),
            "physical_http_attempt_count": physical_attempts,
            "physical_worker_service_s": physical_service,
            "by_tool": {
                name: {metric: distribution(values) for metric, values in groups.items()}
                for name, groups in sorted(tools_by_name.items())
            },
            "speculative_execution_count": 0,
            "wasted_speculative_service_s": 0.0,
        },
        "unattributed_per_task_overhead": distribution(task_residuals),
        "dependency_evidence": dependency_evidence,
        "resource_boundary": {
            "provisioned_gpu_count": 4,
            "gpu_count_saving_claimed": False,
            "energy_measured": False,
            "cost_measured": False,
        },
    }
