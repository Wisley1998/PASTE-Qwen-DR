from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
START_VLLM = REPOSITORY_ROOT / "reproduction" / "scripts" / "start_vllm.sh"


class StartVllmScriptTests(unittest.TestCase):
    def _capture_command(
        self, graph_sizes: str | None, *, prefix_caching: str | None = None
    ) -> list[str]:
        with tempfile.TemporaryDirectory(
            prefix=".start-vllm-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            environment_root = root / "env"
            fake_python = environment_root / "bin" / "python"
            fake_python.parent.mkdir(parents=True)
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\0' \"$@\" > \"${FAKE_VLLM_ARGS}\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            model_snapshot = root / "model"
            model_snapshot.mkdir()
            (model_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            captured_args = root / "args.bin"

            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]

            environment = os.environ.copy()
            environment.update(
                {
                    "PASTE_ENV_PREFIX": str(environment_root),
                    "MODEL_ID": "test/model",
                    "MODEL_SNAPSHOT": str(model_snapshot),
                    "VLLM_HOST": "127.0.0.1",
                    "VLLM_PROBE_HOST": "127.0.0.1",
                    "VLLM_PORT": str(port),
                    "VLLM_STATE_DIR": str(root / "state"),
                    "VLLM_LOG_DIR": str(root / "logs"),
                    "VLLM_READY_TIMEOUT": "1",
                    "VLLM_START_CLEANUP_TIMEOUT": "1",
                    "VLLM_REQUIRE_NEW": "1",
                    "FAKE_VLLM_ARGS": str(captured_args),
                }
            )
            if graph_sizes is None:
                environment.pop("VLLM_CUDA_GRAPH_SIZES", None)
            else:
                environment["VLLM_CUDA_GRAPH_SIZES"] = graph_sizes
            if prefix_caching is None:
                environment.pop("VLLM_ENABLE_PREFIX_CACHING", None)
            else:
                environment["VLLM_ENABLE_PREFIX_CACHING"] = prefix_caching

            completed = subprocess.run(
                ["bash", str(START_VLLM)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertTrue(captured_args.is_file(), completed.stdout)
            return [
                value.decode("utf-8")
                for value in captured_args.read_bytes().split(b"\0")
                if value
            ]

    def test_cuda_graph_sizes_are_omitted_when_unset(self) -> None:
        args = self._capture_command(None)
        self.assertNotIn("--cuda-graph-sizes", args)

    def test_single_cuda_graph_size_is_forwarded(self) -> None:
        args = self._capture_command("256")
        index = args.index("--cuda-graph-sizes")
        self.assertEqual(args[index + 1 :], ["256"])

    def test_comma_separated_cuda_graph_sizes_become_separate_arguments(self) -> None:
        args = self._capture_command("64,128,256")
        index = args.index("--cuda-graph-sizes")
        self.assertEqual(args[index + 1 :], ["64", "128", "256"])

    def test_prefix_caching_is_enabled_by_default(self) -> None:
        args = self._capture_command(None)
        self.assertIn("--enable-prefix-caching", args)

    def test_prefix_caching_can_be_disabled_for_ablation(self) -> None:
        args = self._capture_command(None, prefix_caching="0")
        self.assertNotIn("--enable-prefix-caching", args)
        self.assertIn("--no-enable-prefix-caching", args)

    def test_invalid_cuda_graph_sizes_fail_before_startup(self) -> None:
        for value in ("0", "-1", "1,,2", "1, 2", "1,", "abc"):
            with self.subTest(value=value):
                environment = os.environ.copy()
                environment["VLLM_CUDA_GRAPH_SIZES"] = value
                completed = subprocess.run(
                    ["bash", str(START_VLLM)],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(completed.returncode, 1, completed.stdout)
                self.assertIn(
                    "VLLM_CUDA_GRAPH_SIZES must be a positive integer",
                    completed.stdout,
                )

    def test_invalid_prefix_caching_flag_fails_before_startup(self) -> None:
        environment = os.environ.copy()
        environment["VLLM_ENABLE_PREFIX_CACHING"] = "yes"
        completed = subprocess.run(
            ["bash", str(START_VLLM)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 1, completed.stdout)
        self.assertIn(
            "VLLM_ENABLE_PREFIX_CACHING must be 0 or 1", completed.stdout
        )


if __name__ == "__main__":
    unittest.main()
