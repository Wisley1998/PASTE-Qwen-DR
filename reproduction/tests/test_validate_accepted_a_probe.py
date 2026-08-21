from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from summarize_natural_queue_probe import summarize_probe  # noqa: E402
from validate_accepted_a_probe import (  # noqa: E402
    ENGINE_SHAPE_KEYS,
    validate_accepted_probe,
)
from reproduction.tests.test_summarize_natural_queue_probe import (  # noqa: E402
    _write_probe_fixture,
)


ENGINE_SHAPE = {
    "MODEL_ID": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "MODEL_REVISION": "4b0ac5767427a55d08a254f0367e2934976598e0",
    "VLLM_TP_SIZE": "4",
    "VLLM_DTYPE": "bfloat16",
    "VLLM_MAX_MODEL_LEN": "16384",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.86",
    "VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
    "VLLM_MAX_NUM_SEQS": "256",
    "VLLM_CUDA_GRAPH_SIZES": "256",
    "VLLM_USE_V1": "1",
}
EXPECTED_A_PROFILE = "stress240_native256_g256_u86_a_probe"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_accepted_probe_fixture(
    run_root: Path,
    *,
    profile: str = EXPECTED_A_PROFILE,
    engine_overrides: dict[str, str] | None = None,
    retry: bool = False,
    preemptions: int = 0,
    swap_events: int = 0,
) -> Path:
    cell = _write_probe_fixture(
        run_root,
        retry=retry,
        preemptions=preemptions,
        swap_events=swap_events,
    )
    accepted_cell = run_root / "accepted_a_fcfs_none"
    cell.rename(accepted_cell)
    summary_path = accepted_cell / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["max_active_traces"] = 240
    summary["workload"]["trace_count"] = 240
    environment = summary["scheduler_environment"]
    environment["PASTE_STRESS_PROFILE"] = profile
    environment.update(ENGINE_SHAPE)
    if engine_overrides:
        environment.update(engine_overrides)
    _write_json(summary_path, summary)

    probe_path = run_root / "natural_queue_probe.json"
    _write_json(probe_path, summarize_probe(accepted_cell))
    return probe_path


def _validate(probe_path: Path, **overrides: object) -> dict:
    arguments: dict[str, object] = {
        "repository_root": REPOSITORY_ROOT,
        "expected_profile": EXPECTED_A_PROFILE,
        "expected_load": 240,
        "expected_max_num_seqs": 256,
        "minimum_waiting_fraction": 0.50,
        "minimum_queue_fraction": 0.20,
        "maximum_preemptions_per_request": 0.25,
        "expected_engine_shape": ENGINE_SHAPE,
    }
    arguments.update(overrides)
    return validate_accepted_probe(probe_path, **arguments)  # type: ignore[arg-type]


class AcceptedAProbeTests(unittest.TestCase):
    def test_accepts_recomputed_stress240_probe_and_exact_engine_shape(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".accepted-a-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            probe = _write_accepted_probe_fixture(Path(temporary))
            result = _validate(probe)

        self.assertEqual(tuple(result["engine_shape"]), ENGINE_SHAPE_KEYS)
        self.assertEqual(result["profile"], EXPECTED_A_PROFILE)
        self.assertEqual(result["load"], 240)
        self.assertEqual(result["max_num_seqs"], 256)
        self.assertEqual(result["waiting_below_cap_sample_fraction"], 0.5)
        self.assertEqual(result["queue_time_fraction_of_request_latency"], 0.25)

    def test_missing_or_non_json_probe_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".accepted-a-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "missing or incomplete"):
                _validate(root / "natural_queue_probe.json")
            invalid = root / "natural_queue_probe.json"
            invalid.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                _validate(invalid)

    def test_detached_or_edited_assertion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".accepted-a-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            probe = _write_accepted_probe_fixture(Path(temporary))
            payload = json.loads(probe.read_text(encoding="utf-8"))
            payload["queueing"]["queue_time_fraction_of_request_latency"] = 0.99
            _write_json(probe, payload)
            with self.assertRaisesRegex(ValueError, "fresh validation"):
                _validate(probe)

    def test_wrong_profile_load_or_engine_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".accepted-a-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            wrong_profile = _write_accepted_probe_fixture(
                root / "profile", profile="stress180_native256_g256_u86"
            )
            with self.assertRaisesRegex(ValueError, "scheduler profile"):
                _validate(wrong_profile)

            accepted = _write_accepted_probe_fixture(root / "load")
            with self.assertRaisesRegex(ValueError, "does not match load 239"):
                _validate(accepted, expected_load=239)

            wrong_engine = _write_accepted_probe_fixture(
                root / "engine", engine_overrides={"VLLM_CUDA_GRAPH_SIZES": "128"}
            )
            with self.assertRaisesRegex(
                ValueError, "engine shape mismatch for VLLM_CUDA_GRAPH_SIZES"
            ):
                _validate(wrong_engine)

    def test_exactly_once_swap_wait_queue_and_preemption_gates_are_enforced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".accepted-a-test-", dir=REPOSITORY_ROOT
        ) as temporary:
            root = Path(temporary)
            cases = (
                (
                    _write_accepted_probe_fixture(root / "retry", retry=True),
                    {},
                    "exactly-once",
                ),
                (
                    _write_accepted_probe_fixture(root / "swap", swap_events=1),
                    {},
                    "no-CPU-KV-swap",
                ),
                (
                    _write_accepted_probe_fixture(root / "wait"),
                    {"minimum_waiting_fraction": 0.51},
                    "waiting-below-cap",
                ),
                (
                    _write_accepted_probe_fixture(root / "queue"),
                    {"minimum_queue_fraction": 0.26},
                    "queue-time",
                ),
                (
                    _write_accepted_probe_fixture(root / "preempt", preemptions=1),
                    {},
                    "preemption",
                ),
            )
            for probe, overrides, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        _validate(probe, **overrides)


if __name__ == "__main__":
    unittest.main()
