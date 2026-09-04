from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
START_VLLM = REPOSITORY_ROOT / "reproduction" / "scripts" / "start_vllm.sh"
PASTE_PYTHON = Path("/home/aiscuser/.conda/envs/paste/bin/python")
PYTHON_HOOKS = REPOSITORY_ROOT / "scripts" / "pythonhooks"


class StartVllmScriptTests(unittest.TestCase):
    def test_python310_sitecustomize_filters_late_inserted_cwd_path(self) -> None:
        """The pinned Python lacks -P, so the bound hook enforces its equivalent."""

        self.assertTrue(PASTE_PYTHON.is_file())
        with tempfile.TemporaryDirectory(
            prefix=".strict-safe-path-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            safe_cwd = Path(temporary)
            safe_cwd.chmod(0o500)
            environment = {
                "PATH": "/home/aiscuser/.conda/envs/paste/bin:/usr/bin:/bin",
                "PYTHONPATH": str(PYTHON_HOOKS),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "VLLM_SAFE_WORKING_DIR": str(safe_cwd),
                "VLLM_SCHED_POLICY": "fcfs",
            }
            program = (
                "import importlib.util,json,os,sys;"
                "os.chmod(os.getcwd(),0o700);"
                "open('cwd_poison.py','w').write('raise RuntimeError');"
                "print(json.dumps({"
                "'enforced':os.getenv('PASTE_STRICT_SAFE_PATH_ENFORCED'),"
                "'filter':os.getenv('PASTE_STRICT_CWD_IMPORT_FILTER_ENFORCED'),"
                "'path0':sys.path[0],"
                "'poison_spec':importlib.util.find_spec('cwd_poison') is not None"
                "}))"
            )
            completed = subprocess.run(
                [str(PASTE_PYTHON), "-c", program],
                cwd=safe_cwd,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
                check=False,
            )
            safe_cwd.chmod(0o700)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["enforced"], "1")
            self.assertEqual(payload["filter"], "1")
            # Python 3.10 inserts this after sitecustomize.  The persistent
            # PathFinder wrapper still prevents the injected module loading.
            self.assertEqual(payload["path0"], "")
            self.assertFalse(payload["poison_spec"])

    def test_python310_sitecustomize_preserves_metadata_but_filters_cwd_dist_info(
        self,
    ) -> None:
        """Installed metadata remains visible while CWD metadata is excluded."""

        self.assertTrue(PASTE_PYTHON.is_file())
        with tempfile.TemporaryDirectory(
            prefix=".strict-metadata-path-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            safe_cwd = Path(temporary)
            safe_cwd.chmod(0o500)
            environment = {
                "PATH": "/home/aiscuser/.conda/envs/paste/bin:/usr/bin:/bin",
                "PYTHONPATH": str(PYTHON_HOOKS),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "VLLM_SAFE_WORKING_DIR": str(safe_cwd),
                "VLLM_SCHED_POLICY": "fcfs",
            }
            program = """
import importlib.metadata as metadata
import json
import os
from pathlib import Path
from pydantic import BaseModel, EmailStr


class EmailFixture(BaseModel):
    address: EmailStr

cwd = Path.cwd()
os.chmod(cwd, 0o700)
for directory, name, version in (
    ("email_validator-999.0.dist-info", "email-validator", "999.0"),
    ("cwd_poison_metadata-999.0.dist-info", "cwd-poison-metadata", "999.0"),
):
    root = cwd / directory
    root.mkdir()
    (root / "METADATA").write_text(
        f"Metadata-Version: 2.1\\nName: {name}\\nVersion: {version}\\n",
        encoding="utf-8",
    )

try:
    poison_default = metadata.version("cwd-poison-metadata")
except metadata.PackageNotFoundError:
    poison_default = None

def versions_for(path):
    return sorted(
        row.version
        for row in metadata.distributions(
            name="cwd-poison-metadata", path=path
        )
    )

print(json.dumps({
    "installed": metadata.version("email-validator"),
    "pydantic_email": str(EmailFixture(address="strict@example.com").address),
    "poison_default": poison_default,
    "poison_explicit_absolute": versions_for([str(cwd)]),
    "poison_explicit_empty": versions_for([""]),
    "poison_explicit_pathlike": versions_for([cwd]),
}))
"""
            try:
                completed = subprocess.run(
                    [str(PASTE_PYTHON), "-c", program],
                    cwd=safe_cwd,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10,
                    check=False,
                )
            finally:
                safe_cwd.chmod(0o700)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["installed"], "2.3.0")
            self.assertEqual(payload["pydantic_email"], "strict@example.com")
            self.assertIsNone(payload["poison_default"])
            self.assertEqual(payload["poison_explicit_absolute"], [])
            self.assertEqual(payload["poison_explicit_empty"], [])
            self.assertEqual(payload["poison_explicit_pathlike"], [])

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
                "if [[ \"${1:-}\" == \"-I\" && \"${2:-}\" == \"-c\" "
                "&& \"${3:-}\" == *importlib.metadata* ]]; then\n"
                "  printf '%s\\n' \"${FAKE_VLLM_VERSION:-0.10.1}\"\n"
                "  exit \"${FAKE_VLLM_VERSION_STATUS:-0}\"\n"
                "fi\n"
                "printf '%s\\0' \"$@\" > \"${FAKE_VLLM_ARGS}\"\n"
                "printf '%s\\n' \"${VLLM_SCHEDULER_RUNTIME_EVIDENCE:-}\" "
                "> \"${FAKE_RUNTIME_EVIDENCE_CAPTURE}\"\n"
                "pwd -P > \"${FAKE_WORKING_DIRECTORY_CAPTURE}\"\n"
                "printf '%s\\n' \"${PYTHONPATH:-}\" > \"${FAKE_PYTHONPATH_CAPTURE}\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            model_revision = "fixture-revision"
            hf_home = root / "hf"
            model_snapshot = (
                hf_home
                / "models--test--model"
                / "snapshots"
                / model_revision
            )
            model_snapshot.mkdir(parents=True)
            (model_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            captured_args = root / "args.bin"
            captured_runtime_evidence = root / "runtime-evidence-path.txt"
            captured_working_directory = root / "working-directory.txt"
            captured_pythonpath = root / "pythonpath.txt"
            safe_working_directory = root / "empty-python-cwd"
            safe_working_directory.mkdir()
            malicious_working_directory = root / "malicious-caller-cwd"
            malicious_working_directory.mkdir()
            poison_marker = root / "malicious-cwd-imported"
            (malicious_working_directory / "sitecustomize.py").write_text(
                f"from pathlib import Path\nPath({str(poison_marker)!r}).touch()\n",
                encoding="utf-8",
            )
            poison_vllm = malicious_working_directory / "vllm"
            poison_vllm.mkdir()
            (poison_vllm / "__init__.py").write_text(
                f"from pathlib import Path\nPath({str(poison_marker)!r}).touch()\n",
                encoding="utf-8",
            )

            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]

            environment = os.environ.copy()
            environment.pop("MODEL_SNAPSHOT", None)
            environment.update(
                {
                    "PASTE_ENV_PREFIX": str(environment_root),
                    "HF_HOME": str(hf_home),
                    "MODEL_ID": "test/model",
                    "MODEL_REVISION": model_revision,
                    "VLLM_HOST": "127.0.0.1",
                    "VLLM_PROBE_HOST": "127.0.0.1",
                    "VLLM_PORT": str(port),
                    "VLLM_STATE_DIR": str(root / "state"),
                    "VLLM_LOG_DIR": str(root / "logs"),
                    "VLLM_READY_TIMEOUT": "1",
                    "VLLM_START_CLEANUP_TIMEOUT": "1",
                    "VLLM_REQUIRE_NEW": "1",
                    "VLLM_SAFE_WORKING_DIR": str(safe_working_directory),
                    "FAKE_VLLM_ARGS": str(captured_args),
                    "FAKE_RUNTIME_EVIDENCE_CAPTURE": str(
                        captured_runtime_evidence
                    ),
                    "FAKE_WORKING_DIRECTORY_CAPTURE": str(
                        captured_working_directory
                    ),
                    "FAKE_PYTHONPATH_CAPTURE": str(captured_pythonpath),
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
                cwd=malicious_working_directory,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertTrue(captured_args.is_file(), completed.stdout)
            self.assertEqual(
                captured_runtime_evidence.read_text(encoding="utf-8").strip(),
                str(root / "state" / f"vllm_{port}.scheduler_runtime.json"),
            )
            self.assertEqual(
                captured_working_directory.read_text(encoding="utf-8").strip(),
                str(safe_working_directory),
            )
            self.assertEqual(
                captured_pythonpath.read_text(encoding="utf-8").strip(),
                str(REPOSITORY_ROOT / "scripts" / "pythonhooks"),
            )
            self.assertFalse(poison_marker.exists())
            return [
                value.decode("utf-8")
                for value in captured_args.read_bytes().split(b"\0")
                if value
            ]

    def test_cuda_graph_sizes_are_omitted_when_unset(self) -> None:
        args = self._capture_command(None)
        self.assertEqual(args[:2], ["-m", "vllm.entrypoints.openai.api_server"])
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

    def test_wrong_vllm_version_fails_before_server_launch(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".start-vllm-version-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            environment_root = root / "env"
            fake_python = environment_root / "bin" / "python"
            fake_python.parent.mkdir(parents=True)
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == \"-I\" && \"${2:-}\" == \"-c\" "
                "&& \"${3:-}\" == *importlib.metadata* ]]; then\n"
                "  printf '0.10.0\\n'\n"
                "  exit 0\n"
                "fi\n"
                "touch \"${FAKE_SERVER_LAUNCHED}\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            model_revision = "fixture-revision"
            hf_home = root / "hf"
            model_snapshot = (
                hf_home
                / "models--test--model"
                / "snapshots"
                / model_revision
            )
            model_snapshot.mkdir(parents=True)
            (model_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            launched = root / "server-launched"
            environment = os.environ.copy()
            environment.pop("MODEL_SNAPSHOT", None)
            environment.update(
                {
                    "PASTE_ENV_PREFIX": str(environment_root),
                    "HF_HOME": str(hf_home),
                    "MODEL_ID": "test/model",
                    "MODEL_REVISION": model_revision,
                    "VLLM_STATE_DIR": str(root / "state"),
                    "VLLM_LOG_DIR": str(root / "logs"),
                    "FAKE_SERVER_LAUNCHED": str(launched),
                }
            )
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
                "vLLM 0.10.1 is required exactly (found 0.10.0)",
                completed.stdout,
            )
            self.assertFalse(launched.exists())

    def test_model_snapshot_environment_override_is_rejected(self) -> None:
        environment = os.environ.copy()
        environment["MODEL_SNAPSHOT"] = "/tmp/unregistered-model-snapshot"
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
        self.assertIn("MODEL_SNAPSHOT is not a registered input", completed.stdout)

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
