from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest import mock
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_online_trace_learned_experiment.py"
SPEC = importlib.util.spec_from_file_location(
    "run_online_trace_learned_experiment", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class OnlineTraceLearnedRunnerTests(unittest.TestCase):
    def _base_argv(self) -> list[str]:
        return [
            "run_online_trace_learned_experiment.py",
            "--workload",
            "missing.json",
            "--output-dir",
            "unused-output",
            "--cell-label",
            "cell",
            "--speculation-mode",
            "off",
        ]

    def test_learned_model_option_is_available(self) -> None:
        with mock.patch("sys.argv", self._base_argv()):
            args = runner.parse_args()
        self.assertIsNone(args.visit_prediction_model)

    def test_learned_model_requires_visit_speculation(self) -> None:
        with mock.patch(
            "sys.argv",
            self._base_argv() + ["--visit-prediction-model", "mapper.json"],
        ):
            args = runner.parse_args()
        with self.assertRaisesRegex(ValueError, "requires visit or search_visit"):
            asyncio.run(runner.async_main(args))


if __name__ == "__main__":
    unittest.main()
