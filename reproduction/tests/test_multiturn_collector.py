from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, Mapping, Sequence

from paste_repro.invocation import Invocation
from paste_repro.multiturn_collector import (
    ChatCompletion,
    CollectorConfig,
    WORKLOAD_SCHEMA_VERSION,
    collect_fixed_workload,
    load_fixed_workload,
    parse_model_decision,
)
from paste_repro.traces import load_trace, parse_search_results


URL = "https://en.wikipedia.org/wiki/Alpha"


class FakeChatClient:
    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    async def complete(self, **kwargs: Any) -> ChatCompletion:
        self.requests.append([dict(message) for message in kwargs["messages"]])
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return ChatCompletion(value, 0.01, "stop", {"completion_tokens": 4})


class FakeExecutor:
    HTTP_RETRY_POLICY_VERSION = "fake-retry-v1"
    HTTP_LIBRARY_RETRY_CONTROL_VERSION = "fake-library-control-v1"
    HTTP_ATTEMPT_START_GATE_VERSION = "fake-start-gate-v1"
    RETRYABLE_HTTP_STATUSES = (429, 503)
    RETRYABLE_HTTP_EXCEPTION_TYPES = ("FakeTimeout",)

    def __init__(self, *, fail_tool: str | None = None) -> None:
        self.fail_tool = fail_tool
        self.invocations: list[Invocation] = []

    @property
    def http_library_retry_control_checked(self) -> bool:
        return True

    @property
    def http_library_retry_disabled_effective(self) -> bool:
        return True

    @property
    def http_library_name(self) -> str:
        return "fake-http"

    @property
    def http_library_version(self) -> str:
        return "1.2.3"

    def transport_plan(self, invocation: Invocation) -> Mapping[str, Any]:
        return {
            "backend": f"fake_{invocation.tool_name}",
            "request_host": "fake.example",
            "http_attempts": 1,
        }

    async def __call__(self, invocation: Invocation) -> Mapping[str, Any]:
        self.invocations.append(invocation)
        if invocation.tool_name == self.fail_tool:
            error = RuntimeError("deterministic fake tool failure")
            error.paste_http_attempt_log = (  # type: ignore[attr-defined]
                {
                    "request_index": 0,
                    "attempt": 1,
                    "status": 503,
                    "error_type": "FakeHTTP503",
                    "retried": False,
                    "started_monotonic_s": 12.0,
                    "start_gate_wait_s": 0.5,
                    "retry_backoff_s": 0.0,
                },
            )
            raise error
        if invocation.tool_name == "search":
            return {
                "tool": "search",
                "query": invocation.arguments["query"],
                "results": [
                    {
                        "query": "alpha",
                        "query_index": 0,
                        "rank": 1,
                        "title": "Alpha",
                        "url": URL,
                        "snippet": "not exposed to the model",
                    }
                ],
                "_paste_transport": {"backend": "fake_wikipedia"},
            }
        return {
            "tool": "visit",
            "goal": invocation.arguments["goal"],
            "pages": [{"url": URL, "title": "Alpha", "content": "Evidence."}],
            "_paste_transport": {"backend": "fake_visit"},
        }


class MultiturnCollectorTests(unittest.TestCase):
    def _write_workload(self, root: Path, *, duplicate: bool = False) -> Path:
        sources = [
            {
                "source_id": "scientific-0001",
                "question": "What is alpha?",
                "provenance": {
                    "source_file": "scientific_benchmark_50_tasks.jsonl",
                    "source_line": 1,
                    "source_file_sha256": "a" * 64,
                },
            }
        ]
        if duplicate:
            sources.append(dict(sources[0]))
        path = root / "workload.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": WORKLOAD_SCHEMA_VERSION,
                    "workload_id": "fresh-scientific-holdout-v1",
                    "sources": sources,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _config() -> CollectorConfig:
        return CollectorConfig(
            endpoint="http://127.0.0.1:8100/v1",
            model="Tongyi-DeepResearch-30B-A3B",
            max_calls=4,
        )

    def test_success_trace_is_legacy_compatible_and_whole_session(self) -> None:
        responses = [
            '<think>search</think><tool_call>{"name":"search","arguments":{"query":["alpha"]}}</tool_call>',
            f'<think>visit</think><tool_call>{{"name":"visit","arguments":{{"url":"{URL}","goal":"find evidence"}}}}</tool_call>',
            "<answer>Alpha is documented.</answer>",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = self._write_workload(root)
            output = root / "output"
            client = FakeChatClient(responses)
            manifest = asyncio.run(
                collect_fixed_workload(
                    workload_path=workload,
                    output_dir=output,
                    config=self._config(),
                    client=client,
                    executor=FakeExecutor(),
                )
            )

            self.assertEqual(manifest["collection_status"], "complete")
            trace_path = output / manifest["sessions"][0]["trace_file"]
            trace = load_trace(trace_path)
            self.assertEqual(len(trace.events), 9)
            raw_events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(
                [event["event_type"] for event in raw_events],
                [
                    "session_start",
                    "llm_call",
                    "tool_call",
                    "tool_result",
                    "llm_call",
                    "tool_call",
                    "tool_result",
                    "llm_call",
                    "session_end",
                ],
            )
            search_commit = raw_events[3]
            self.assertEqual(search_commit["commit_status"], "committed")
            self.assertEqual(search_commit["tool_name"], "search")
            self.assertEqual(search_commit["raw_result"]["results"][0]["url"], URL)
            self.assertEqual(
                parse_search_results(search_commit["formatted_response"])[0].url,
                URL,
            )
            search_observation = client.requests[1][-1]["content"]
            self.assertIn(f"[Alpha]({URL})", search_observation)
            self.assertNotIn("not exposed to the model", search_observation)
            self.assertEqual(manifest["sessions"][0]["provenance"]["source_line"], 1)
            self.assertEqual(manifest["sessions"][0]["committed_tool_results"], 2)
            self.assertEqual(
                manifest["executor_runtime"]["http_library"],
                {
                    "retry_control_checked": True,
                    "retry_disabled_effective": True,
                    "name": "fake-http",
                    "version": "1.2.3",
                },
            )
            self.assertEqual(
                manifest["executor_runtime"]["http_retry_policy_version"],
                "fake-retry-v1",
            )
            for binding in manifest["source_bindings"].values():
                self.assertEqual(len(binding["sha256"]), 64)
            self.assertFalse(list(output.glob(".*.tmp-*")))

    def test_parse_failure_retains_session_and_finalizes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = self._write_workload(root)
            output = root / "output"
            manifest = asyncio.run(
                collect_fixed_workload(
                    workload_path=workload,
                    output_dir=output,
                    config=self._config(),
                    client=FakeChatClient(["<tool_call>{bad json}</tool_call>"]),
                    executor=FakeExecutor(),
                )
            )
            self.assertEqual(manifest["collection_status"], "complete_with_failures")
            record = manifest["sessions"][0]
            self.assertEqual(record["status"], "failed")
            events = [
                json.loads(line)
                for line in (output / record["trace_file"]).read_text().splitlines()
            ]
            self.assertEqual(events[-2]["event_type"], "collector_error")
            self.assertEqual(events[-1]["event_type"], "session_end")
            self.assertEqual(events[-1]["status"], "failed")
            on_disk_manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(on_disk_manifest["summary"]["failed"], 1)

    def test_tool_failure_retains_emitted_tool_call(self) -> None:
        response = '<tool_call>{"name":"search","arguments":{"query":["alpha"]}}</tool_call>'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = self._write_workload(root)
            output = root / "output"
            manifest = asyncio.run(
                collect_fixed_workload(
                    workload_path=workload,
                    output_dir=output,
                    config=self._config(),
                    client=FakeChatClient([response]),
                    executor=FakeExecutor(fail_tool="search"),
                )
            )
            record = manifest["sessions"][0]
            events = [
                json.loads(line)
                for line in (output / record["trace_file"]).read_text().splitlines()
            ]
            self.assertEqual(
                [event["event_type"] for event in events],
                [
                    "session_start",
                    "llm_call",
                    "tool_call",
                    "collector_error",
                    "session_end",
                ],
            )
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertFalse(record["tool_result_committed"])
            self.assertEqual(record["tool_failure_phase"], "dispatch")
            self.assertEqual(record["http_attempt_log"][0]["status"], 503)
            self.assertEqual(record["transport_plan"]["backend"], "fake_search")
            self.assertNotIn("tool_result", [event["event_type"] for event in events])

    def test_committed_result_survives_following_llm_failure(self) -> None:
        response = '<tool_call>{"name":"search","arguments":{"query":["alpha"]}}</tool_call>'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = self._write_workload(root)
            output = root / "output"
            manifest = asyncio.run(
                collect_fixed_workload(
                    workload_path=workload,
                    output_dir=output,
                    config=self._config(),
                    client=FakeChatClient([response, RuntimeError("next LLM failed")]),
                    executor=FakeExecutor(),
                )
            )
            record = manifest["sessions"][0]
            events = [
                json.loads(line)
                for line in (output / record["trace_file"]).read_text().splitlines()
            ]
            self.assertEqual(
                [event["event_type"] for event in events],
                [
                    "session_start",
                    "llm_call",
                    "tool_call",
                    "tool_result",
                    "collector_error",
                    "session_end",
                ],
            )
            self.assertEqual(events[3]["commit_status"], "committed")
            self.assertIsNotNone(events[3]["formatted_response"])
            self.assertEqual(record["committed_tool_results"], 1)

    def test_rejects_duplicate_source_and_parses_valid_visit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "duplicate source_id"):
                load_fixed_workload(self._write_workload(root, duplicate=True))

        decision = parse_model_decision(
            f'<tool_call>{{"name":"visit","arguments":{{"url":"{URL}","goal":"x"}}}}</tool_call>',
            max_visit_urls=6,
        )
        self.assertEqual(decision.invocation.tool_name, "visit")

    def test_unseen_visit_is_recorded_but_not_dispatched(self) -> None:
        response = (
            f'<tool_call>{{"name":"visit","arguments":{{"url":"{URL}",'
            '"goal":"x"}}</tool_call>'
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = self._write_workload(root)
            output = root / "output"
            executor = FakeExecutor()
            manifest = asyncio.run(
                collect_fixed_workload(
                    workload_path=workload,
                    output_dir=output,
                    config=self._config(),
                    client=FakeChatClient([response]),
                    executor=executor,
                )
            )
            record = manifest["sessions"][0]
            self.assertEqual(record["error_type"], "UnseenVisitUrlError")
            self.assertFalse(record["tool_result_committed"])
            self.assertEqual(record["tool_failure_phase"], "pre_dispatch_validation")
            self.assertEqual(executor.invocations, [])
            events = [
                json.loads(line)
                for line in (output / record["trace_file"]).read_text().splitlines()
            ]
            self.assertEqual(events[2]["event_type"], "tool_call")
            self.assertEqual(events[-1]["status"], "failed")

    def test_raw_commit_survives_formatter_failure(self) -> None:
        class MalformedSearchExecutor(FakeExecutor):
            async def __call__(self, invocation: Invocation) -> Mapping[str, Any]:
                self.invocations.append(invocation)
                return {
                    "tool": "search",
                    "query": invocation.arguments["query"],
                    "results": "malformed-but-finite",
                    "_paste_transport": {"backend": "fake_wikipedia"},
                }

        response = '<tool_call>{"name":"search","arguments":{"query":["alpha"]}}</tool_call>'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = self._write_workload(root)
            output = root / "output"
            manifest = asyncio.run(
                collect_fixed_workload(
                    workload_path=workload,
                    output_dir=output,
                    config=self._config(),
                    client=FakeChatClient([response]),
                    executor=MalformedSearchExecutor(),
                )
            )
            record = manifest["sessions"][0]
            events = [
                json.loads(line)
                for line in (output / record["trace_file"]).read_text().splitlines()
            ]
            committed = next(
                event for event in events if event["event_type"] == "tool_result"
            )
            self.assertEqual(committed["commit_status"], "committed")
            self.assertIsNone(committed["formatted_response"])
            self.assertEqual(committed["raw_result"]["results"], "malformed-but-finite")
            self.assertTrue(record["tool_result_committed"])
            self.assertEqual(record["tool_failure_phase"], "formatting")

    def test_search_batch_boundary_allows_gate_relevant_ten_queries(self) -> None:
        ten_queries = [f"query-{index}" for index in range(10)]
        response = (
            "<tool_call>"
            + json.dumps(
                {"name": "search", "arguments": {"query": ten_queries}},
                separators=(",", ":"),
            )
            + "</tool_call>"
        )
        decision = parse_model_decision(response, max_visit_urls=6)
        self.assertEqual(decision.invocation.arguments["query"], ten_queries)

        response_with_eleven = (
            "<tool_call>"
            + json.dumps(
                {
                    "name": "search",
                    "arguments": {"query": ten_queries + ["query-10"]},
                },
                separators=(",", ":"),
            )
            + "</tool_call>"
        )
        with self.assertRaisesRegex(ValueError, "one to ten"):
            parse_model_decision(response_with_eleven, max_visit_urls=6)

    def test_transport_retry_and_start_interval_config_is_frozen(self) -> None:
        config = CollectorConfig(
            endpoint="http://127.0.0.1:8100",
            model="Tongyi-DeepResearch-30B-A3B",
            max_calls=8,
            search_mode="bing",
            max_http_attempts=2,
            retry_backoff_s=5.0,
            search_min_start_interval_s=1.0,
            visit_min_start_interval_s=1.0,
        )
        tool = config.to_manifest()["tool"]
        self.assertEqual(tool["max_http_attempts"], 2)
        self.assertEqual(tool["retry_backoff_s"], 5.0)
        self.assertEqual(
            tool["http_attempt_min_start_intervals_s"],
            {"search": 1.0, "visit": 1.0},
        )

        defaults = self._config().to_manifest()["tool"]
        self.assertEqual(defaults["max_http_attempts"], 1)
        self.assertEqual(defaults["retry_backoff_s"], 1.0)
        self.assertEqual(defaults["http_attempt_min_start_intervals_s"], {})

    def test_transport_config_rejects_invalid_values(self) -> None:
        base = {
            "endpoint": "http://127.0.0.1:8100",
            "model": "Tongyi-DeepResearch-30B-A3B",
            "max_calls": 8,
        }
        for field, value in (
            ("max_http_attempts", 0),
            ("max_http_attempts", True),
            ("retry_backoff_s", -1.0),
            ("retry_backoff_s", float("nan")),
            ("search_min_start_interval_s", -0.1),
            ("search_min_start_interval_s", float("inf")),
            ("visit_min_start_interval_s", -0.1),
            ("visit_min_start_interval_s", float("-inf")),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    CollectorConfig(**base, **{field: value})

    def test_output_directory_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workload = self._write_workload(root)
            output = root / "output"
            output.mkdir()
            (output / ".collection.claim").write_text("already claimed\n")
            with self.assertRaisesRegex(FileExistsError, "already claimed"):
                asyncio.run(
                    collect_fixed_workload(
                        workload_path=workload,
                        output_dir=output,
                        config=self._config(),
                        client=FakeChatClient([]),
                        executor=FakeExecutor(),
                    )
                )


if __name__ == "__main__":
    unittest.main()
