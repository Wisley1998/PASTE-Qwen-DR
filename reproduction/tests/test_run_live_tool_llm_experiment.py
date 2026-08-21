from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest import mock
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts/run_live_tool_llm_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_live_tool_llm_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _never_started() -> dict[str, object]:
    return {
        "job_id": 1,
        "admitted": True,
        "speculative": True,
        "authoritative": False,
        "committed": False,
        "cancelled": True,
        "outcome": "cancelled",
        "source": "cancelled",
        "queue_enter": 10.0,
        "queue_enter_at": 10.0,
        "start": None,
        "started_at": None,
        "finish": 13.5,
        "finished_at": 13.5,
        "worker_id": None,
        "queue_s": None,
        "service_s": None,
        "saved_service_s": None,
        "http_attempts": None,
        "backend": None,
        "request_host": None,
        "response_status": None,
        "bytes_read": None,
        "transport_identity_source": None,
    }


class ToolAttemptNormalizationTests(unittest.TestCase):
    def test_derives_only_never_started_cancellation_without_mutating_input(self) -> None:
        raw = _never_started()
        normalized = runner._normalize_tool_attempt_records([raw])[0]
        self.assertIsNone(raw["http_attempts"])
        self.assertIsNone(raw["queue_s"])
        self.assertEqual(normalized["http_attempts"], 0)
        self.assertEqual(normalized["queue_s"], 3.5)
        self.assertEqual(normalized["service_s"], 0.0)
        self.assertEqual(normalized["saved_service_s"], 0.0)
        for field in (
            "backend",
            "request_host",
            "response_status",
            "bytes_read",
            "transport_identity_source",
        ):
            self.assertIsNone(normalized[field])

    def test_started_cancellation_retains_its_real_attempt(self) -> None:
        row = _never_started()
        row.update(
            start=11.0,
            started_at=11.0,
            finish=12.0,
            finished_at=12.0,
            worker_id=0,
            queue_s=1.0,
            service_s=1.0,
            saved_service_s=0.0,
            http_attempts=1,
            backend="r.jina.ai",
            request_host="r.jina.ai",
            response_status=200,
            bytes_read=123,
            transport_identity_source="actual",
        )
        normalized = runner._normalize_tool_attempt_records([row])[0]
        self.assertEqual(normalized, row)
        self.assertEqual(normalized["http_attempts"], 1)

    def test_ambiguous_attempt_or_transport_evidence_fails_closed(self) -> None:
        started = _never_started()
        started.update(start=11.0, started_at=11.0)
        with self.assertRaisesRegex(RuntimeError, "positive HTTP attempts"):
            runner._normalize_tool_attempt_records([started])

        never_started = _never_started()
        never_started["backend"] = "r.jina.ai"
        with self.assertRaisesRegex(RuntimeError, "claims transport evidence"):
            runner._normalize_tool_attempt_records([never_started])


class ControlledHttpRetryCliTests(unittest.TestCase):
    def test_retry_controls_are_explicit_and_default_off(self) -> None:
        base = [
            "run_live_tool_llm_experiment.py",
            "--workload",
            "workload.json",
            "--output-dir",
            "output",
            "--cell-label",
            "cell",
            "--speculation-mode",
            "off",
        ]
        with mock.patch("sys.argv", base):
            defaults = runner.parse_args()
        self.assertEqual(defaults.tool_http_max_attempts, 1)
        self.assertEqual(defaults.tool_http_retry_backoff_s, 1.0)
        self.assertFalse(defaults.tool_http_attempt_start_gate)
        self.assertEqual(runner._http_attempt_start_intervals(defaults), {})

        with mock.patch(
            "sys.argv",
            base
            + [
                "--tool-http-max-attempts",
                "2",
                "--tool-http-retry-backoff-s",
                "1.0",
                "--tool-http-attempt-start-gate",
                "--search-min-start-interval-s",
                "0.25",
                "--visit-min-start-interval-s",
                "2.1",
            ],
        ):
            controlled = runner.parse_args()
        self.assertEqual(controlled.tool_http_max_attempts, 2)
        self.assertEqual(controlled.tool_http_retry_backoff_s, 1.0)
        self.assertTrue(controlled.tool_http_attempt_start_gate)
        self.assertEqual(
            runner._http_attempt_start_intervals(controlled),
            {"search": 0.25, "visit": 2.1},
        )

    def test_attempt_start_gate_requires_an_explicit_positive_interval(self) -> None:
        argv = [
            "run_live_tool_llm_experiment.py",
            "--workload",
            "missing.json",
            "--output-dir",
            "unused-output",
            "--cell-label",
            "cell",
            "--speculation-mode",
            "off",
            "--tool-http-attempt-start-gate",
        ]
        with mock.patch("sys.argv", argv):
            args = runner.parse_args()
        with self.assertRaisesRegex(ValueError, "requires at least one positive"):
            asyncio.run(runner.async_main(args))

    def test_real_context_padding_requires_exact_tokenizer(self) -> None:
        argv = [
            "run_live_tool_llm_experiment.py",
            "--workload",
            "missing.json",
            "--output-dir",
            "unused-output",
            "--cell-label",
            "cell",
            "--speculation-mode",
            "off",
            "--context-padding-tokens",
            "5600",
        ]
        with mock.patch("sys.argv", argv):
            args = runner.parse_args()
        with self.assertRaisesRegex(ValueError, "requires --tokenizer"):
            asyncio.run(runner.async_main(args))


if __name__ == "__main__":
    unittest.main()
