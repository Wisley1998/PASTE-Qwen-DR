from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPOSITORY_ROOT
    / "reproduction"
    / "scripts"
    / "run_online_speculative_execution.py"
)
SPEC = importlib.util.spec_from_file_location("run_online_speculative_execution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class OnlineSpeculativeExecutionEntryPointTests(unittest.TestCase):
    def test_command_fixes_causal_online_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = runner.build_parser().parse_args(
                ["--output-dir", str(Path(directory) / "result"), "--source-limit", "2"]
            )
            command = runner.build_command(args)

        joined = " ".join(command)
        self.assertIn("--call-graph-mode autonomous", joined)
        self.assertIn("--speculation-mode visit", joined)
        self.assertIn("--visit-prediction-model", command)
        self.assertIn("--visit-top-k 5", joined)
        self.assertIn("--source-limit 2", joined)
        self.assertNotIn("frozen", command)

    def test_invalid_top_k_fails_before_server_contact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = runner.build_parser().parse_args(
                ["--output-dir", str(Path(directory) / "result"), "--top-k", "0"]
            )
            with self.assertRaisesRegex(ValueError, "top-k must be positive"):
                runner.build_command(args)


if __name__ == "__main__":
    unittest.main()
