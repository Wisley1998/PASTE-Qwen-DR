from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_heldout_union_workloads import _merge_mode  # noqa: E402


def workload(role: str, mode: str, count: int = 30) -> dict:
    meta = {
        "max_model_len": 16384,
        "max_output_tokens_cap": 512,
        "min_output_tokens_floor": 64,
        "output_token_buffer": 8,
        "tool_overlap_mode": mode,
        "tool_overlap_efficiency": 1.0,
        "prefix_marker_mode": "preserve_prefix",
    }
    if mode == "learned":
        meta.update(
            {
                "tool_prediction_artifact_sha256": "a" * 64,
                "tool_prediction_top_k": 5,
            }
        )
    return {
        "meta": meta,
        "traces": [
            {
                "trace_id": f"trace_{index:03d}",
                "source_trace": f"/trace/{role}_{index:03d}.jsonl",
                "variant_index": index,
                "duplicated": False,
                "prefix_char": "",
                "initial_delay_s": 0.0,
                "truncated_calls": index % 2,
                "requests": [{"call_index": 0, "max_tokens": 64}],
            }
            for index in range(count)
        ],
    }


class HeldoutUnionTests(unittest.TestCase):
    def test_union_has_sixty_unique_nonduplicated_sessions(self) -> None:
        merged = _merge_mode(
            tuning=workload("tuning", "learned"),
            final=workload("final", "learned"),
            mode="learned",
        )
        traces = merged["traces"]
        self.assertEqual(len(traces), 60)
        self.assertEqual([trace["trace_id"] for trace in traces], [
            f"heldout_{index:03d}" for index in range(60)
        ])
        self.assertEqual(len({trace["source_trace"] for trace in traces}), 60)
        self.assertTrue(all(not trace["duplicated"] for trace in traces))
        self.assertEqual(merged["meta"]["source_roles"], ["tuning", "final"])
        self.assertEqual(
            merged["meta"]["evidence_role"],
            "heldout_load_sensitivity_not_untouched_final",
        )

    def test_source_overlap_fails_closed(self) -> None:
        tuning = workload("tuning", "none")
        final = workload("final", "none")
        final["traces"][0]["source_trace"] = tuning["traces"][0]["source_trace"]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            _merge_mode(tuning=tuning, final=final, mode="none")

    def test_metadata_or_mode_mismatch_fails_closed(self) -> None:
        tuning = workload("tuning", "none")
        final = copy.deepcopy(tuning)
        final["meta"]["max_output_tokens_cap"] = 128
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            _merge_mode(tuning=tuning, final=final, mode="none")
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            _merge_mode(
                tuning=workload("tuning", "none"),
                final=workload("final", "none"),
                mode="learned",
            )


if __name__ == "__main__":
    unittest.main()
