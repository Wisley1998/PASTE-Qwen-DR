from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "reproduction/scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from validate_live_joint_formal_workload import (  # noqa: E402
    DEFAULT_WORKLOAD,
    FORMAL_V3_EXCLUSIONS,
    FORMAL_V3_WORKLOAD,
    FORMAL_V4_EXCLUSIONS,
    FORMAL_V4_WORKLOAD,
    FORMAL_V5_EXCLUSIONS,
    FORMAL_V5_WORKLOAD,
    FORMAL_V6_EXCLUSIONS,
    FORMAL_V6_WORKLOAD,
    FORMAL_V7_EXCLUSIONS,
    FORMAL_V7_WORKLOAD,
    FORMAL_V8_EXCLUSIONS,
    FORMAL_V8_EXCLUSION_SHA256,
    FORMAL_V8_WORKLOAD,
    FORMAL_V9_EXCLUSIONS,
    FORMAL_V9_EXCLUSION_SHA256,
    FORMAL_V9_WORKLOAD,
    validate_formal_workload,
)


EXPECTED_CANONICAL_SHA256 = (
    "53bcc19229c9a67052518e2fe6b2c5762621bd81be80e1f461d590bdba13db77"
)
EXPECTED_FILE_SHA256 = (
    "4c71ce9bf72b3cbec8ddc077f7e58270493f10e63f3a45e107e39faff3b1bb76"
)
EXPECTED_V3_CANONICAL_SHA256 = (
    "73ca2e13e825da8a3ad6715bab4100707a3e74ffbcbf363709d2cdd455700580"
)
EXPECTED_V3_SOURCES_SHA256 = (
    "b363d73b1f1a1180e8d8f3748221e01e26e4b2b2302dd7c714b95b7db8b98beb"
)
EXPECTED_V3_FILE_SHA256 = (
    "a8f5de832e7e04e3cbd1b7bb71629207201f99285a0d9f95fbc1e7246f0b6366"
)
EXPECTED_V4_CANONICAL_SHA256 = (
    "8fd754b87065986832f2c03850652d83cbe25dd0ceaa66788719dff7ebc28ab3"
)
EXPECTED_V4_SOURCES_SHA256 = (
    "04bcbd0cf5eb1ad2d11beb1869ad9cd806d8255648e92a811a14004b3b635fe6"
)
EXPECTED_V4_FILE_SHA256 = (
    "e965317225ed0f2d4aec9e8e1a444abd0949521205e705c4daae5e786ce092d5"
)
EXPECTED_V5_CANONICAL_SHA256 = (
    "7e89dea02bf2dfc5bf2b7dd2669c0d753097d5e2e351b26f018eb3df02268fbe"
)
EXPECTED_V5_SOURCES_SHA256 = (
    "478310accbd16ce623a4684465dd029a01efa80bfd299f3522943e90bf2cba46"
)
EXPECTED_V5_FILE_SHA256 = (
    "6b11193c8a0dbbd70f9ae4bc2c72b56737893b4d45dacd1d9970e01ca019ae31"
)
EXPECTED_V6_CANONICAL_SHA256 = (
    "019fbc5177e45b4cc8cb752ccc28a7070ae1c70a1faeded787a1989dc262a96b"
)
EXPECTED_V6_SOURCES_SHA256 = (
    "e07a94c9485205e2fb864d65a6339ac5885b0821d0b2123113107bfed988f4e0"
)
EXPECTED_V6_FILE_SHA256 = (
    "44122877db66b1df4a985316c2a96b71d91d13c4e8be84affb73d405490bd43f"
)
EXPECTED_V7_CANONICAL_SHA256 = (
    "09e88d67f4aeb1994a566e11678fceb8f374f3b86f667da112f901209e0ef393"
)
EXPECTED_V7_SOURCES_SHA256 = (
    "710cc4f8d62f6c2b8ab78ec3d61d79be1ba7db25f47559accd407e7d0ddc810c"
)
EXPECTED_V7_FILE_SHA256 = (
    "cbf143f59f4d2a05650df68d8fa6f00d7471964a4b257d26dd092ba90c40e6c8"
)
EXPECTED_V8_CANONICAL_SHA256 = (
    "93b8cfad78b76c42101f7d0f23583911b01bc8c075260ae3d85bce45456a9ec7"
)
EXPECTED_V8_SOURCES_SHA256 = (
    "01b029c3427f5f04d4f1b83b4f9b13e5decd705e773ffdeaeebb15970150f0df"
)
EXPECTED_V8_FILE_SHA256 = (
    "780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4"
)
EXPECTED_V9_CANONICAL_SHA256 = (
    "de588fcbd46c1181156f5a6e49e0264c785c00c43e0d8c2a62698fb6217e3ce7"
)
EXPECTED_V9_SOURCES_SHA256 = (
    "750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c"
)
EXPECTED_V9_FILE_SHA256 = (
    "c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20"
)


def _payload(path: Path = DEFAULT_WORKLOAD) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_mutation(payload: dict) -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "mutated_formal_workload.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return validate_formal_workload(path)


class FormalWorkloadValidatorTests(unittest.TestCase):
    def test_frozen_v2_passes_with_expected_digest(self) -> None:
        result = validate_formal_workload()
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 60)
        self.assertEqual(result["canonical_json_sha256"], EXPECTED_CANONICAL_SHA256)
        self.assertEqual(result["file_sha256"], EXPECTED_FILE_SHA256)

    def test_frozen_v3_passes_with_expected_digests_and_bindings(self) -> None:
        result = validate_formal_workload(FORMAL_V3_WORKLOAD)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 60)
        self.assertEqual(
            result["split_id"], "live-joint-wikipedia-frozen-formal-v3"
        )
        self.assertEqual(
            result["canonical_json_sha256"], EXPECTED_V3_CANONICAL_SHA256
        )
        self.assertEqual(
            result["canonical_sources_sha256"], EXPECTED_V3_SOURCES_SHA256
        )
        self.assertEqual(result["file_sha256"], EXPECTED_V3_FILE_SHA256)
        self.assertEqual(result["excluded_topic_count"], 137)
        self.assertEqual(result["excluded_question_count"], 136)
        self.assertEqual(result["excluded_semantic_value_count"], 273)
        self.assertEqual(
            {binding["name"] for binding in result["exclusion_bindings"]},
            {path.name for path in FORMAL_V3_EXCLUSIONS},
        )

    def test_frozen_v4_passes_with_expected_digests_and_bindings(self) -> None:
        result = validate_formal_workload(FORMAL_V4_WORKLOAD)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 60)
        self.assertEqual(
            result["split_id"], "live-joint-wikipedia-frozen-formal-v4"
        )
        self.assertEqual(
            result["canonical_json_sha256"], EXPECTED_V4_CANONICAL_SHA256
        )
        self.assertEqual(
            result["canonical_sources_sha256"], EXPECTED_V4_SOURCES_SHA256
        )
        self.assertEqual(result["file_sha256"], EXPECTED_V4_FILE_SHA256)
        self.assertEqual(result["excluded_topic_count"], 200)
        self.assertEqual(result["excluded_question_count"], 196)
        self.assertEqual(result["excluded_semantic_value_count"], 396)
        self.assertEqual(
            {binding["name"] for binding in result["exclusion_bindings"]},
            {path.name for path in FORMAL_V4_EXCLUSIONS},
        )

    def test_frozen_v5_passes_with_expected_digests_and_bindings(self) -> None:
        result = validate_formal_workload(FORMAL_V5_WORKLOAD)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 60)
        self.assertEqual(
            result["split_id"], "live-joint-wikipedia-frozen-formal-v5"
        )
        self.assertEqual(
            result["canonical_json_sha256"], EXPECTED_V5_CANONICAL_SHA256
        )
        self.assertEqual(
            result["canonical_sources_sha256"], EXPECTED_V5_SOURCES_SHA256
        )
        self.assertEqual(result["file_sha256"], EXPECTED_V5_FILE_SHA256)
        self.assertEqual(result["excluded_topic_count"], 260)
        self.assertEqual(result["excluded_question_count"], 256)
        self.assertEqual(result["excluded_semantic_value_count"], 516)
        self.assertEqual(
            {binding["name"] for binding in result["exclusion_bindings"]},
            {path.name for path in FORMAL_V5_EXCLUSIONS},
        )

    def test_frozen_v6_passes_with_expected_digests_and_bindings(self) -> None:
        result = validate_formal_workload(FORMAL_V6_WORKLOAD)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 60)
        self.assertEqual(
            result["split_id"], "live-joint-wikipedia-frozen-formal-v6"
        )
        self.assertEqual(
            result["canonical_json_sha256"], EXPECTED_V6_CANONICAL_SHA256
        )
        self.assertEqual(
            result["canonical_sources_sha256"], EXPECTED_V6_SOURCES_SHA256
        )
        self.assertEqual(result["file_sha256"], EXPECTED_V6_FILE_SHA256)
        self.assertEqual(result["excluded_topic_count"], 320)
        self.assertEqual(result["excluded_question_count"], 316)
        self.assertEqual(result["excluded_semantic_value_count"], 636)
        self.assertEqual(
            {binding["name"] for binding in result["exclusion_bindings"]},
            {path.name for path in FORMAL_V6_EXCLUSIONS},
        )

    def test_frozen_v7_passes_with_expected_digests_and_bindings(self) -> None:
        result = validate_formal_workload(FORMAL_V7_WORKLOAD)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 60)
        self.assertEqual(
            result["split_id"], "live-joint-wikipedia-frozen-formal-v7"
        )
        self.assertEqual(
            result["canonical_json_sha256"], EXPECTED_V7_CANONICAL_SHA256
        )
        self.assertEqual(
            result["canonical_sources_sha256"], EXPECTED_V7_SOURCES_SHA256
        )
        self.assertEqual(result["file_sha256"], EXPECTED_V7_FILE_SHA256)
        self.assertEqual(result["excluded_topic_count"], 380)
        self.assertEqual(result["excluded_question_count"], 376)
        self.assertEqual(result["excluded_semantic_value_count"], 756)
        self.assertEqual(
            {binding["name"] for binding in result["exclusion_bindings"]},
            {path.name for path in FORMAL_V7_EXCLUSIONS},
        )

    def test_frozen_v8_passes_with_exact_80_sources_and_sha_bindings(self) -> None:
        result = validate_formal_workload(FORMAL_V8_WORKLOAD)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 80)
        self.assertEqual(
            result["split_id"], "live-joint-wikipedia-frozen-formal-v8"
        )
        self.assertEqual(
            result["canonical_json_sha256"], EXPECTED_V8_CANONICAL_SHA256
        )
        self.assertEqual(
            result["canonical_sources_sha256"], EXPECTED_V8_SOURCES_SHA256
        )
        self.assertEqual(result["file_sha256"], EXPECTED_V8_FILE_SHA256)
        self.assertEqual(result["excluded_topic_count"], 440)
        self.assertEqual(result["excluded_question_count"], 436)
        self.assertEqual(result["excluded_semantic_value_count"], 876)
        self.assertEqual(
            {binding["name"]: binding["file_sha256"] for binding in result["exclusion_bindings"]},
            FORMAL_V8_EXCLUSION_SHA256,
        )
        self.assertEqual(
            {binding["name"] for binding in result["exclusion_bindings"]},
            {path.name for path in FORMAL_V8_EXCLUSIONS},
        )

    def test_frozen_v9_passes_with_exact_80_sources_and_sha_bindings(self) -> None:
        result = validate_formal_workload(FORMAL_V9_WORKLOAD)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_count"], 80)
        self.assertEqual(
            result["split_id"], "live-joint-wikipedia-frozen-formal-v9"
        )
        self.assertEqual(
            result["canonical_json_sha256"], EXPECTED_V9_CANONICAL_SHA256
        )
        self.assertEqual(
            result["canonical_sources_sha256"], EXPECTED_V9_SOURCES_SHA256
        )
        self.assertEqual(result["file_sha256"], EXPECTED_V9_FILE_SHA256)
        self.assertEqual(result["excluded_topic_count"], 520)
        self.assertEqual(result["excluded_question_count"], 516)
        self.assertEqual(result["excluded_semantic_value_count"], 1036)
        self.assertEqual(
            {
                binding["name"]: binding["file_sha256"]
                for binding in result["exclusion_bindings"]
            },
            FORMAL_V9_EXCLUSION_SHA256,
        )
        self.assertEqual(
            {binding["name"] for binding in result["exclusion_bindings"]},
            {path.name for path in FORMAL_V9_EXCLUSIONS},
        )

    def test_duplicate_query_and_url_fail_closed(self) -> None:
        payload = _payload()
        payload["sources"][1]["search_query"] = payload["sources"][0]["search_query"]
        payload["sources"][1]["expected_url"] = payload["sources"][0]["expected_url"]
        with self.assertRaisesRegex(ValueError, "duplicates a source ID, query, or URL"):
            _validate_mutation(payload)

    def test_development_topic_overlap_fails_closed(self) -> None:
        payload = _payload()
        payload["sources"][0].update(
            {
                "question": "When did Apollo 11 land on the Moon, and who walked there?",
                "search_query": "Apollo 11",
                "expected_url": "https://en.wikipedia.org/wiki/Apollo_11",
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps excluded semantics"):
            _validate_mutation(payload)

    def test_v3_excludes_dev_tune_frozen_url_and_formal_v2_topics(self) -> None:
        cases = (
            ("Apollo 11", "https://en.wikipedia.org/wiki/Apollo_11"),
            ("Ada Lovelace", "https://en.wikipedia.org/wiki/Ada_Lovelace"),
            ("DNA", "https://en.wikipedia.org/wiki/DNA"),
            (
                "French Revolution",
                "https://en.wikipedia.org/wiki/French_Revolution",
            ),
        )
        for query, expected_url in cases:
            with self.subTest(query=query):
                payload = _payload(FORMAL_V3_WORKLOAD)
                payload["sources"][0].update(
                    {
                        "question": f"What key facts distinguish the historical topic called {query}?",
                        "search_query": query,
                        "expected_url": expected_url,
                    }
                )
                with self.assertRaisesRegex(
                    ValueError, "overlaps excluded semantics"
                ):
                    _validate_mutation(payload)

    def test_v3_question_only_overlap_fails_closed(self) -> None:
        payload = _payload(FORMAL_V3_WORKLOAD)
        payload["sources"][0]["question"] = (
            "How did Apollo 11 complete its mission and return safely to Earth?"
        )
        with self.assertRaisesRegex(ValueError, "question overlaps excluded semantics"):
            _validate_mutation(payload)

    def test_noncanonical_or_non_https_url_fails_closed(self) -> None:
        payload = _payload()
        payload["sources"][0]["expected_url"] = (
            "http://en.wikipedia.org/wiki/French_Revolution"
        )
        with self.assertRaisesRegex(ValueError, "canonical HTTPS"):
            _validate_mutation(payload)

    def test_tuning_attestation_fails_closed(self) -> None:
        payload = _payload()
        payload["used_for_tuning"] = True
        with self.assertRaisesRegex(ValueError, "attestation failed: used_for_tuning"):
            _validate_mutation(payload)

    def test_query_url_topic_mismatch_fails_closed(self) -> None:
        payload = _payload()
        payload["sources"][0]["expected_url"] = (
            "https://en.wikipedia.org/wiki/Roman_Empire"
        )
        with self.assertRaisesRegex(ValueError, "identify different topics"):
            _validate_mutation(payload)

    def test_exact_development_bindings_are_required(self) -> None:
        payload = copy.deepcopy(_payload())
        payload["excluded_workloads"].pop()
        with self.assertRaisesRegex(ValueError, "does not bind all frozen exclusions"):
            _validate_mutation(payload)

    def test_v3_requires_formal_v2_in_exclusion_binding(self) -> None:
        payload = _payload(FORMAL_V3_WORKLOAD)
        payload["excluded_workloads"].remove(
            "live_joint_wikipedia_frozen_formal_v2.json"
        )
        with self.assertRaisesRegex(ValueError, "does not bind all frozen exclusions"):
            _validate_mutation(payload)

    def test_v3_rejects_incomplete_exclusion_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the frozen split binding"):
            validate_formal_workload(
                FORMAL_V3_WORKLOAD,
                development_paths=FORMAL_V3_EXCLUSIONS[:-1],
            )

    def test_v4_excludes_frozen_tune_and_both_prior_formal_splits(self) -> None:
        cases = (
            ("Ada Lovelace", "https://en.wikipedia.org/wiki/Ada_Lovelace"),
            (
                "French Revolution",
                "https://en.wikipedia.org/wiki/French_Revolution",
            ),
            ("Renaissance", "https://en.wikipedia.org/wiki/Renaissance"),
        )
        for query, expected_url in cases:
            with self.subTest(query=query):
                payload = _payload(FORMAL_V4_WORKLOAD)
                payload["sources"][0].update(
                    {
                        "question": f"What key facts distinguish the historical topic called {query}?",
                        "search_query": query,
                        "expected_url": expected_url,
                    }
                )
                with self.assertRaisesRegex(
                    ValueError, "overlaps excluded semantics"
                ):
                    _validate_mutation(payload)

    def test_v4_rejects_incomplete_exclusion_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the frozen split binding"):
            validate_formal_workload(
                FORMAL_V4_WORKLOAD,
                development_paths=FORMAL_V4_EXCLUSIONS[:-1],
            )

    def test_v5_excludes_formal_v4_topics(self) -> None:
        payload = _payload(FORMAL_V5_WORKLOAD)
        payload["sources"][0].update(
            {
                "question": "What key facts distinguish the ancient civilization called Sumer?",
                "search_query": "Sumer",
                "expected_url": "https://en.wikipedia.org/wiki/Sumer",
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps excluded semantics"):
            _validate_mutation(payload)

    def test_v5_rejects_incomplete_exclusion_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the frozen split binding"):
            validate_formal_workload(
                FORMAL_V5_WORKLOAD,
                development_paths=FORMAL_V5_EXCLUSIONS[:-1],
            )

    def test_v6_excludes_formal_v5_topics(self) -> None:
        payload = _payload(FORMAL_V6_WORKLOAD)
        payload["sources"][0].update(
            {
                "question": "Where did Ancient Egypt develop, and how was its kingdom governed?",
                "search_query": "Ancient Egypt",
                "expected_url": "https://en.wikipedia.org/wiki/Ancient_Egypt",
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps excluded semantics"):
            _validate_mutation(payload)

    def test_v6_rejects_incomplete_exclusion_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the frozen split binding"):
            validate_formal_workload(
                FORMAL_V6_WORKLOAD,
                development_paths=FORMAL_V6_EXCLUSIONS[:-1],
            )

    def test_v7_excludes_formal_v6_topics(self) -> None:
        payload = _payload(FORMAL_V7_WORKLOAD)
        payload["sources"][0].update(
            {
                "question": "Where did the Akkadian Empire arise, and how was its territory governed?",
                "search_query": "Akkadian Empire",
                "expected_url": "https://en.wikipedia.org/wiki/Akkadian_Empire",
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps excluded semantics"):
            _validate_mutation(payload)

    def test_v7_rejects_incomplete_exclusion_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the frozen split binding"):
            validate_formal_workload(
                FORMAL_V7_WORKLOAD,
                development_paths=FORMAL_V7_EXCLUSIONS[:-1],
            )

    def test_v8_excludes_formal_v7_topics(self) -> None:
        payload = _payload(FORMAL_V8_WORKLOAD)
        payload["sources"][0].update(
            {
                "question": "Where did the Hittites establish their kingdom and political authority?",
                "search_query": "Hittites",
                "expected_url": "https://en.wikipedia.org/wiki/Hittites",
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps excluded semantics"):
            _validate_mutation(payload)

    def test_v8_requires_exactly_80_sources(self) -> None:
        payload = _payload(FORMAL_V8_WORKLOAD)
        payload["sources"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 80 sources"):
            _validate_mutation(payload)

    def test_v8_requires_unique_questions(self) -> None:
        payload = _payload(FORMAL_V8_WORKLOAD)
        payload["sources"][1]["question"] = payload["sources"][0]["question"]
        with self.assertRaisesRegex(ValueError, "duplicates a formal topic or question"):
            _validate_mutation(payload)

    def test_v8_rejects_incomplete_exclusion_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the frozen split binding"):
            validate_formal_workload(
                FORMAL_V8_WORKLOAD,
                development_paths=FORMAL_V8_EXCLUSIONS[:-1],
            )

    def test_v8_rejects_duplicate_exclusion_names(self) -> None:
        payload = _payload(FORMAL_V8_WORKLOAD)
        payload["excluded_workloads"].append(payload["excluded_workloads"][0])
        with self.assertRaisesRegex(ValueError, "does not bind all frozen exclusions"):
            _validate_mutation(payload)

    def test_v8_rejects_tampered_historical_exclusion_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tampered_v7 = Path(temporary) / FORMAL_V7_WORKLOAD.name
            tampered_v7.write_text(
                FORMAL_V7_WORKLOAD.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            exclusions = [
                tampered_v7 if path == FORMAL_V7_WORKLOAD else path
                for path in FORMAL_V8_EXCLUSIONS
            ]
            with self.assertRaisesRegex(ValueError, "wrong frozen SHA256"):
                validate_formal_workload(
                    FORMAL_V8_WORKLOAD,
                    development_paths=exclusions,
                )

    def test_v9_excludes_formal_v8_topics(self) -> None:
        payload = _payload(FORMAL_V9_WORKLOAD)
        payload["sources"][0].update(
            {
                "question": "Where did the Olmecs flourish, and what cultural legacy did they leave?",
                "search_query": "Olmecs",
                "expected_url": "https://en.wikipedia.org/wiki/Olmecs",
            }
        )
        with self.assertRaisesRegex(ValueError, "overlaps excluded semantics"):
            _validate_mutation(payload)

    def test_v9_requires_exactly_80_sources(self) -> None:
        payload = _payload(FORMAL_V9_WORKLOAD)
        payload["sources"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 80 sources"):
            _validate_mutation(payload)

    def test_v9_requires_unique_questions(self) -> None:
        payload = _payload(FORMAL_V9_WORKLOAD)
        payload["sources"][1]["question"] = payload["sources"][0]["question"]
        with self.assertRaisesRegex(ValueError, "duplicates a formal topic or question"):
            _validate_mutation(payload)

    def test_v9_rejects_incomplete_exclusion_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the frozen split binding"):
            validate_formal_workload(
                FORMAL_V9_WORKLOAD,
                development_paths=FORMAL_V9_EXCLUSIONS[:-1],
            )

    def test_v9_rejects_duplicate_exclusion_names(self) -> None:
        payload = _payload(FORMAL_V9_WORKLOAD)
        payload["excluded_workloads"].append(payload["excluded_workloads"][0])
        with self.assertRaisesRegex(ValueError, "does not bind all frozen exclusions"):
            _validate_mutation(payload)

    def test_v9_rejects_tampered_formal_v8_exclusion_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tampered_v8 = Path(temporary) / FORMAL_V8_WORKLOAD.name
            tampered_v8.write_text(
                FORMAL_V8_WORKLOAD.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            exclusions = [
                tampered_v8 if path == FORMAL_V8_WORKLOAD else path
                for path in FORMAL_V9_EXCLUSIONS
            ]
            with self.assertRaisesRegex(ValueError, "wrong frozen SHA256"):
                validate_formal_workload(
                    FORMAL_V9_WORKLOAD,
                    development_paths=exclusions,
                )


if __name__ == "__main__":
    unittest.main()
