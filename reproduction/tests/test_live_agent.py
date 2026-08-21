from __future__ import annotations

import asyncio
import hashlib
from importlib.metadata import version
import json
import unittest

from paste_repro.invocation import Invocation
from paste_repro.live_agent import (
    ApproximateTokenCounter,
    FINAL_ANSWER_MAX_CHARS,
    FINAL_ANSWER_MAX_WORDS,
    FINAL_ANSWER_TARGET_CHARS,
    FINAL_COMPLETION_TOKEN_COUNT,
    FINAL_ANSWER_CONTRACT_POLICY_VERSION,
    FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
    FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION,
    FINAL_ANSWER_SCHEMA_POLICY_VERSION,
    FIXED_FINAL_ANSWER_CONTRACT_POLICY_VERSION,
    FIXED_OUTPUT_CONTRACT_POLICY_VERSION,
    GUIDED_JSON_RECOVERY_POLICY_VERSION,
    OUTPUT_CONTRACT_POLICY_VERSION,
    LLMCompletion,
    LiveClosedLoopExperiment,
    LiveLLMClient,
    LiveSource,
    SYSTEM_PROMPT,
    build_scheduler_meta,
    canonical_json,
    final_answer_fixed_completion_grammar,
    final_answer_schema,
    parse_guided_final_answer,
    parse_guided_object,
    search_schema,
    scheduler_request_id,
    summarize_live_run,
    task_to_dict,
    validate_final_answer,
    validate_sources,
    visit_schema,
    _broker_tool_signal,
    _unique_context_padding,
)
from paste_repro.live_broker import LiveToolBroker
from paste_repro.online_learned_agent import (
    LiveClosedLoopExperiment as OnlineLearnedClosedLoopExperiment,
)

from scripts.run_live_tool_llm_experiment import _timeline_summary


class ExactTestTokenCounter(ApproximateTokenCounter):
    method = "exact_test_fixture"


class LiveAgentUnitTests(unittest.TestCase):
    def test_execution_aware_tool_signal_gates_queued_predictions(self) -> None:
        queued = {
            "jobs": [
                {
                    "tool_name": "visit",
                    "invocation_digest": hashlib.sha256(
                        b'visit\0{"url":["https://example.test/u"]}'
                    ).hexdigest(),
                    "lane": "speculative",
                    "state": "queued",
                    "confirmed": False,
                }
            ]
        }
        running = {
            "jobs": [
                {
                    "tool_name": "visit",
                    "invocation_digest": hashlib.sha256(
                        b'visit\0{"url":["https://example.test/u"]}'
                    ).hexdigest(),
                    "lane": "speculative",
                    "state": "running",
                    "confirmed": False,
                    "job_id": 7,
                    "queue_position": 0,
                    "tool_queue_position": 0,
                    "estimated_remaining_s": 3.5,
                }
            ]
        }
        invocation = Invocation("visit", {"url": ["https://example.test/u"]})
        self.assertEqual(
            _broker_tool_signal(
                queued,
                invocation=invocation,
                nominal_confidence=1.0,
                policy="execution_aware",
            ),
            (
                0.0,
                {
                    "nps": "speculative_queued",
                    "nrg": 0,
                    "ntc": 1.0,
                    "npm": 1,
                    "br": 0,
                    "npjid": -1,
                    "npc": 0,
                    "npq": -1,
                    "nptq": -1,
                },
            ),
        )

        confirmed = dict(running)
        confirmed["jobs"] = [dict(running["jobs"][0], confirmed=True)]
        confidence, evidence = _broker_tool_signal(
            confirmed,
            invocation=invocation,
            nominal_confidence=1.0,
            policy="execution_aware",
        )
        self.assertEqual(confidence, 0.0)
        self.assertEqual(evidence["nrg"], 0)
        self.assertEqual(evidence["npc"], 1)

        confidence, evidence = _broker_tool_signal(
            running,
            invocation=invocation,
            nominal_confidence=1.0,
            policy="execution_aware",
            eligible=False,
        )
        self.assertEqual(confidence, 0.0)
        self.assertEqual(evidence["nps"], "ineligible")
        self.assertEqual(
            _broker_tool_signal(
                running,
                invocation=invocation,
                nominal_confidence=0.75,
                policy="execution_aware",
            ),
            (
                0.75,
                {
                    "nps": "speculative_running",
                    "nrg": 1,
                    "ntc": 0.75,
                    "npm": 1,
                    "br": 0,
                    "npjid": 7,
                    "npc": 0,
                    "npq": 0,
                    "nptq": 0,
                    "nper": 3.5,
                },
            ),
        )

    def test_execution_aware_tool_signal_fails_closed_without_exact_job(self) -> None:
        self.assertEqual(
            _broker_tool_signal(
                {"jobs": []},
                invocation=Invocation("search", {"query": ["q"]}),
                nominal_confidence=0.5,
                policy="execution_aware",
            ),
            (
                0.0,
                {"nps": "missing", "nrg": 0, "ntc": 0.5, "npm": 0, "br": 0},
            ),
        )
        self.assertEqual(
            _broker_tool_signal(
                {"jobs": []},
                invocation=Invocation("search", {"query": ["q"]}),
                nominal_confidence=0.5,
                policy="legacy",
            ),
            (0.5, {}),
        )

    def test_final_answer_contract_is_bounded_and_mirrored_locally(self) -> None:
        url = "https://example.test/source"
        self.assertIn(
            f"aim for at most {FINAL_ANSWER_TARGET_CHARS} answer characters",
            SYSTEM_PROMPT.lower(),
        )
        self.assertIn('{"answer":"...","source_url":"..."}', SYSTEM_PROMPT)

        schema = final_answer_schema(url)
        self.assertEqual(schema["properties"]["answer"], {"type": "string"})
        self.assertEqual(schema["properties"]["source_url"], {"const": url})
        encoded_schema = json.dumps(schema)
        for forbidden in ("minLength", "maxLength", "pattern"):
            self.assertNotIn(forbidden, encoded_schema)

        validate_final_answer({"answer": "bounded", "source_url": url}, url=url)
        with self.assertRaisesRegex(ValueError, "concise answer contract"):
            validate_final_answer(
                {"answer": "x" * (FINAL_ANSWER_MAX_CHARS + 1), "source_url": url},
                url=url,
            )

    def test_guided_final_canonicalizes_and_projects_with_versioned_evidence(
        self,
    ) -> None:
        url = "https://example.test/source"
        model_answer = "  Apollo\n\tlanded\u2003in July 1969.  "
        raw = json.dumps({"answer": model_answer, "source_url": url})
        telemetry: dict = {}
        answer = parse_guided_final_answer(raw, url=url, telemetry=telemetry)
        self.assertEqual(
            answer,
            {"answer": "Apollo landed in July 1969.", "source_url": url},
        )
        self.assertEqual(
            telemetry["policy_version"], FINAL_ANSWER_CONTRACT_POLICY_VERSION
        )
        self.assertEqual(
            telemetry["schema_policy_version"],
            FINAL_ANSWER_SCHEMA_POLICY_VERSION,
        )
        self.assertEqual(telemetry["mode"], "guided_json_strict_local_projection")
        self.assertTrue(telemetry["guided_json_requested"])
        self.assertTrue(telemetry["json_parse_attempted"])
        self.assertTrue(telemetry["strict_json_parse"])
        self.assertFalse(telemetry["recovery_allowed"])
        self.assertFalse(telemetry["recovery_applied"])
        self.assertTrue(telemetry["parse_succeeded"])
        self.assertTrue(telemetry["local_wrap_applied"])
        self.assertTrue(telemetry["object_constructed_locally"])
        self.assertTrue(telemetry["canonicalization_changed"])
        self.assertFalse(telemetry["local_projection_applied"])
        self.assertTrue(telemetry["model_source_url_validated"])
        self.assertTrue(telemetry["contract_succeeded"])
        self.assertEqual(
            telemetry["raw_sha256"], hashlib.sha256(raw.encode()).hexdigest()
        )
        self.assertEqual(
            telemetry["canonical_sha256"],
            hashlib.sha256(answer["answer"].encode()).hexdigest(),
        )
        self.assertEqual(telemetry["canonical_word_count"], 5)

    def test_guided_final_projects_both_frozen_bounds_deterministically(self) -> None:
        url = "https://example.test/source"
        raw_answer = " ".join(
            ["abcdefghij"] * (FINAL_ANSWER_MAX_WORDS + 5)
        )
        raw = json.dumps({"answer": raw_answer, "source_url": url})
        first_telemetry: dict = {}
        second_telemetry: dict = {}
        first = parse_guided_final_answer(raw, url=url, telemetry=first_telemetry)
        second = parse_guided_final_answer(raw, url=url, telemetry=second_telemetry)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first["answer"]), FINAL_ANSWER_MAX_CHARS)
        self.assertLessEqual(
            len(first["answer"].split(" ")), FINAL_ANSWER_MAX_WORDS
        )
        self.assertTrue(first_telemetry["word_projection_applied"])
        self.assertTrue(first_telemetry["char_projection_applied"])
        self.assertTrue(first_telemetry["local_projection_applied"])
        self.assertEqual(
            first_telemetry["canonical_sha256"],
            second_telemetry["canonical_sha256"],
        )

        # A language without ASCII word separators still receives a hard,
        # deterministic Unicode-code-point prefix.
        cjk = "界" * (FINAL_ANSWER_MAX_CHARS + 7)
        projected = parse_guided_final_answer(
            json.dumps({"answer": cjk, "source_url": url}), url=url
        )
        self.assertEqual(projected["answer"], cjk[:FINAL_ANSWER_MAX_CHARS])

    def test_guided_final_fails_closed_on_malformed_or_unbound_json(self) -> None:
        url = "https://example.test/source"
        malformed = '{"answer":"alpha\nbeta","source_url":"' + url + '"}'
        telemetry: dict = {}
        with self.assertRaisesRegex(ValueError, "strict JSON object"):
            parse_guided_final_answer(malformed, url=url, telemetry=telemetry)
        self.assertFalse(telemetry["parse_succeeded"])
        self.assertFalse(telemetry["recovery_allowed"])
        self.assertFalse(telemetry["recovery_applied"])
        self.assertFalse(telemetry["contract_succeeded"])

        duplicate = (
            '{"answer":"first","answer":"second","source_url":"'
            + url
            + '"}'
        )
        with self.assertRaisesRegex(ValueError, "strict JSON object"):
            parse_guided_final_answer(duplicate, url=url)
        nonstandard = '{"answer":NaN,"source_url":"' + url + '"}'
        with self.assertRaisesRegex(ValueError, "strict JSON object"):
            parse_guided_final_answer(nonstandard, url=url)

        wrong_url = json.dumps(
            {"answer": "bounded", "source_url": "https://example.test/wrong"}
        )
        with self.assertRaisesRegex(ValueError, "exact committed URL"):
            parse_guided_final_answer(wrong_url, url=url)
        extra = json.dumps(
            {"answer": "bounded", "source_url": url, "unexpected": True}
        )
        with self.assertRaisesRegex(ValueError, "object shape"):
            parse_guided_final_answer(extra, url=url)
        control = json.dumps({"answer": "answer\x00injection", "source_url": url})
        with self.assertRaisesRegex(ValueError, "control character"):
            parse_guided_final_answer(control, url=url)

    def test_final_schema_compiles_offline_with_installed_xgrammar(self) -> None:
        try:
            import xgrammar
        except ImportError:
            self.skipTest("xgrammar is not installed in this test environment")

        url = "https://en.wikipedia.org/wiki/A_(B)"
        schema = final_answer_schema(url)
        # A byte-level stand-in vocabulary keeps this compiler regression test
        # independent of a model download or GPU.  It exercises the installed
        # xgrammar JSON-schema converter and token grammar compiler.
        encoded_vocab = [bytes([value]) for value in range(256)] + [b"<eos>"]
        tokenizer_info = xgrammar.TokenizerInfo(
            encoded_vocab,
            vocab_size=len(encoded_vocab),
            stop_token_ids=[256],
        )
        compiled = xgrammar.GrammarCompiler(
            tokenizer_info, max_threads=1, cache_enabled=False
        ).compile_json_schema(schema, strict_mode=True)

        # This deliberately crosses the former maxLength=480 failure boundary:
        # the FSM accepts it and the local contract is solely responsible for
        # projecting it to the hard answer bound.
        raw = json.dumps(
            {"answer": "x" * (FINAL_ANSWER_MAX_CHARS + 1), "source_url": url},
            separators=(",", ":"),
        ).replace("/", "\\/")
        matcher = xgrammar.GrammarMatcher(
            compiled, terminate_without_stop_token=True
        )
        self.assertTrue(matcher.accept_string(raw))
        self.assertTrue(matcher.is_terminated())
        projected = parse_guided_final_answer(raw, url=url)
        self.assertEqual(len(projected["answer"]), FINAL_ANSWER_MAX_CHARS)

    def test_fixed_final_grammar_compiles_on_xgrammar_0_1_21(self) -> None:
        try:
            import xgrammar
        except ImportError:
            self.skipTest("xgrammar is not installed in this test environment")

        self.assertEqual(version("xgrammar"), FINAL_ANSWER_GRAMMAR_XGRAMMAR_VERSION)
        url = "https://en.wikipedia.org/wiki/A_(B)"
        grammar_text = final_answer_fixed_completion_grammar(url)
        grammar = xgrammar.Grammar.from_ebnf(grammar_text)
        encoded_vocab = [bytes([value]) for value in range(256)] + [b"<eos>"]
        tokenizer_info = xgrammar.TokenizerInfo(
            encoded_vocab,
            vocab_size=len(encoded_vocab),
            stop_token_ids=[256],
        )
        compiled = xgrammar.GrammarCompiler(
            tokenizer_info, max_threads=1, cache_enabled=False
        ).compile_grammar(grammar)

        semantic = canonical_json(
            {
                "answer": "x" * (FINAL_ANSWER_MAX_CHARS + 1),
                "source_url": url,
            }
        ).replace("/", "\\/")

        def accepts(raw: str) -> bool:
            matcher = xgrammar.GrammarMatcher(
                compiled, terminate_without_stop_token=True
            )
            return matcher.accept_string(raw) and matcher.is_terminated()

        self.assertTrue(accepts(semantic + "   "))
        self.assertFalse(accepts(semantic))
        self.assertFalse(accepts(semantic + " \t"))
        self.assertFalse(accepts(semantic + " \n"))
        self.assertFalse(accepts(semantic + " x"))
        self.assertFalse(accepts(semantic.replace("https", "http", 1) + " "))
        self.assertFalse(accepts(semantic.replace(":", ": ", 1) + " "))

    def test_fixed_final_strict_tail_and_token_evidence(self) -> None:
        url = "https://example.test/source"
        semantic = canonical_json(
            {"answer": "  Apollo\nlanded in July 1969.  ", "source_url": url}
        )
        padding = " " * 73
        raw = semantic + padding
        counter = ExactTestTokenCounter()
        telemetry: dict = {}
        answer = parse_guided_final_answer(
            raw,
            url=url,
            telemetry=telemetry,
            token_counter=counter,
            completion_tokens=FINAL_COMPLETION_TOKEN_COUNT,
            finish_reason="length",
            fixed_completion_tokens=FINAL_COMPLETION_TOKEN_COUNT,
        )

        self.assertEqual(
            answer,
            {"answer": "Apollo landed in July 1969.", "source_url": url},
        )
        self.assertEqual(
            telemetry["policy_version"],
            FIXED_FINAL_ANSWER_CONTRACT_POLICY_VERSION,
        )
        self.assertEqual(
            telemetry["grammar_policy_version"],
            FINAL_ANSWER_GRAMMAR_POLICY_VERSION,
        )
        self.assertEqual(
            telemetry["grammar_sha256"],
            hashlib.sha256(
                final_answer_fixed_completion_grammar(url).encode()
            ).hexdigest(),
        )
        self.assertEqual(
            telemetry["semantic_sha256"],
            hashlib.sha256(semantic.encode()).hexdigest(),
        )
        self.assertEqual(telemetry["padding_char_count"], len(padding))
        self.assertTrue(telemetry["tail_validation_succeeded"])
        self.assertTrue(telemetry["tail_ascii_space_only"])
        self.assertEqual(
            telemetry["total_completion_tokens"], FINAL_COMPLETION_TOKEN_COUNT
        )
        self.assertEqual(
            telemetry["semantic_token_count"]
            + telemetry["padding_token_count"],
            FINAL_COMPLETION_TOKEN_COUNT,
        )
        self.assertGreater(telemetry["padding_token_count"], 0)
        self.assertEqual(telemetry["finish_reason"], "length")
        self.assertTrue(telemetry["finish_reason_validated"])
        self.assertTrue(telemetry["token_accounting_succeeded"])
        self.assertTrue(telemetry["contract_succeeded"])
        self.assertFalse(telemetry["recovery_allowed"])
        self.assertFalse(telemetry["recovery_applied"])

    def test_fixed_final_contract_fails_closed(self) -> None:
        url = "https://example.test/source"
        semantic = canonical_json({"answer": "bounded", "source_url": url})
        common = {
            "url": url,
            "token_counter": ExactTestTokenCounter(),
            "completion_tokens": FINAL_COMPLETION_TOKEN_COUNT,
            "finish_reason": "length",
            "fixed_completion_tokens": FINAL_COMPLETION_TOKEN_COUNT,
        }
        for suffix in ("", "\t", " \n", " x", "\u00a0"):
            with self.subTest(suffix=repr(suffix)):
                with self.assertRaisesRegex(ValueError, "tail must contain"):
                    parse_guided_final_answer(semantic + suffix, **common)

        with self.assertRaisesRegex(ValueError, "exactly 192 completion tokens"):
            parse_guided_final_answer(
                semantic + " ",
                **{**common, "completion_tokens": 191},
            )
        with self.assertRaisesRegex(ValueError, "finish because of length"):
            parse_guided_final_answer(
                semantic + " ",
                **{**common, "finish_reason": "stop"},
            )
        with self.assertRaisesRegex(ValueError, "exact model tokenizer"):
            parse_guided_final_answer(
                semantic + " ",
                **{**common, "token_counter": ApproximateTokenCounter()},
            )
        with self.assertRaisesRegex(ValueError, "must equal 192"):
            parse_guided_final_answer(
                semantic + " ",
                **{**common, "fixed_completion_tokens": 191},
            )
        with self.assertRaisesRegex(ValueError, "strict JSON object"):
            parse_guided_final_answer('{"answer":NaN} ', **common)

        class OversizedSemanticCounter(ExactTestTokenCounter):
            def count_text(self, text: str) -> int:
                return FINAL_COMPLETION_TOKEN_COUNT

        with self.assertRaisesRegex(ValueError, "token accounting"):
            parse_guided_final_answer(
                semantic + " ",
                **{**common, "token_counter": OversizedSemanticCounter()},
            )

    def test_private_context_padding_reaches_target_and_is_task_unique(self) -> None:
        counter = ApproximateTokenCounter()
        left, left_tokens = _unique_context_padding(
            token_counter=counter, task_id="source__r00", target_tokens=128
        )
        right, right_tokens = _unique_context_padding(
            token_counter=counter, task_id="source__r01", target_tokens=128
        )
        self.assertGreaterEqual(left_tokens, 128)
        self.assertGreaterEqual(right_tokens, 128)
        self.assertNotEqual(left, right)

    def test_workload_validation_rejects_duplicate_sources(self) -> None:
        payload = {
            "sources": [
                {"source_id": "x", "question": "q", "search_query": "s"},
                {"source_id": "x", "question": "q2", "search_query": "s2"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            validate_sources(payload)

    def test_frozen_workload_requires_absolute_https_expected_url(self) -> None:
        base = {"source_id": "x", "question": "q", "search_query": "s"}
        with self.assertRaisesRegex(ValueError, "requires expected_url"):
            validate_sources({"sources": [base]}, call_graph_mode="frozen")
        for invalid in ("http://example.test/x", "/relative", "not a url"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "absolute HTTPS"):
                    validate_sources(
                        {"sources": [{**base, "expected_url": invalid}]},
                        call_graph_mode="frozen",
                    )

        sources = validate_sources(
            {
                "sources": [
                    {**base, "expected_url": "https://example.test/frozen"}
                ]
            },
            call_graph_mode="frozen",
        )
        self.assertEqual(sources[0].expected_url, "https://example.test/frozen")

    def test_autonomous_workload_does_not_require_expected_url(self) -> None:
        sources = validate_sources(
            {"sources": [{"source_id": "x", "question": "q", "search_query": "s"}]}
        )
        self.assertIsNone(sources[0].expected_url)
        self.assertEqual(
            task_to_dict(sources[0]),
            {"source_id": "x", "question": "q", "search_query": "s"},
        )

    def test_visit_schema_only_allows_live_result_urls(self) -> None:
        schema = visit_schema(["https://example/a", "https://example/b"], "goal")
        items = schema["properties"]["arguments"]["properties"]["url"]["items"]
        self.assertEqual(items["enum"], ["https://example/a", "https://example/b"])
        self.assertEqual(
            schema["properties"]["arguments"]["properties"]["goal"]["const"],
            "goal",
        )

    def test_request_id_round_trips_live_queue_metadata(self) -> None:
        global_snapshot = {
            "counts": {
                "queued_authoritative": 2,
                "queued_speculative": 3,
                "running_authoritative": 4,
                "running_speculative": 1,
            },
            "capacity": {"max_workers": 8},
            "service_ewma_s": {"visit": 2.0},
            "jobs": [],
        }
        meta = build_scheduler_meta(
            task_id="t",
            call_index=1,
            prompt_tokens=100,
            max_tokens=64,
            predicted_output_tokens=30,
            broker_global=global_snapshot,
            broker_session=global_snapshot,
            next_tool_name="visit",
            next_tool_confidence=0.75,
            next_prompt_tokens=500,
            default_tool_service_s=1.0,
        )
        self.assertEqual(meta["tqa"], 2)
        self.assertEqual(meta["tqs"], 3)
        self.assertAlmostEqual(meta["nw"], 3.25)
        request_id = scheduler_request_id(meta)
        self.assertTrue(request_id.startswith("schedx"))
        self.assertTrue(request_id.endswith("z"))

    def test_guided_parser_rejects_non_object(self) -> None:
        telemetry: dict = {}
        self.assertEqual(
            parse_guided_object('{"a":1}', telemetry=telemetry), {"a": 1}
        )
        self.assertEqual(
            telemetry,
            {
                "policy_version": GUIDED_JSON_RECOVERY_POLICY_VERSION,
                "recovery_applied": False,
                "raw_sha256": hashlib.sha256(b'{"a":1}').hexdigest(),
            },
        )
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_guided_object("[]")

    def test_guided_parser_narrowly_recovers_controls_inside_strings(self) -> None:
        content = '{"answer":"alpha\nbeta\tgamma","source_url":"https://x"}'
        telemetry: dict = {}
        self.assertEqual(
            parse_guided_object(content, telemetry=telemetry),
            {
                "answer": "alpha\nbeta\tgamma",
                "source_url": "https://x",
            },
        )
        self.assertTrue(telemetry["recovery_applied"])
        self.assertEqual(telemetry["control_character_count"], 2)
        self.assertEqual(telemetry["control_character_codepoints"], [9, 10])
        self.assertEqual(
            telemetry["raw_sha256"], hashlib.sha256(content.encode()).hexdigest()
        )
        self.assertNotEqual(telemetry["raw_sha256"], telemetry["repaired_sha256"])

    def test_guided_parser_does_not_repair_other_json_errors(self) -> None:
        telemetry: dict = {}
        with self.assertRaisesRegex(ValueError, "one JSON object"):
            parse_guided_object('{"a":1,}', telemetry=telemetry)
        self.assertFalse(telemetry["recovery_applied"])
        self.assertNotIn("repaired_sha256", telemetry)

    def test_control_recovery_does_not_bypass_answer_length_validation(self) -> None:
        url = "https://example.test/source"
        content = (
            '{"answer":"'
            + "x" * FINAL_ANSWER_MAX_CHARS
            + '\n","source_url":"'
            + url
            + '"}'
        )
        telemetry: dict = {}
        answer = parse_guided_object(content, telemetry=telemetry)
        self.assertTrue(telemetry["recovery_applied"])
        with self.assertRaisesRegex(ValueError, "concise answer contract"):
            validate_final_answer(answer, url=url)

    def test_summary_counts_real_tool_commits(self) -> None:
        tasks = [
            {
                "ok": True,
                "e2e_s": 10.0,
                "tools": [
                    {"exposed_wait_s": 1.0, "queue_s": 0.2, "service_s": 0.8},
                    {"exposed_wait_s": 0.1, "queue_s": 0.0, "service_s": 0.9},
                ],
            }
        ]
        events = [{"ok": True, "attempts": 1, "duration_s": 3.0, "usage": {}}]
        summary = summarize_live_run(
            tasks=tasks,
            llm_events=events,
            broker_stats={"commits": 2},
            started_wall_s=1.0,
            ended_wall_s=11.0,
        )
        self.assertTrue(summary["all_tasks_succeeded"])
        self.assertEqual(summary["tool"]["authoritative_commit_count"], 2)
        self.assertAlmostEqual(summary["tool"]["mean_exposed_wait_s"], 0.55)

    def test_timeline_summary_requires_simultaneous_llm_and_tool_pressure(self) -> None:
        rows = [
            {
                "tool_queued_authoritative": 1,
                "tool_queued_speculative": 0,
                "tool_running_authoritative": 0,
                "llm_running": 2,
                "llm_waiting": 0,
            },
            {
                "tool_queued_authoritative": 0,
                "tool_queued_speculative": 3,
                "tool_running_authoritative": 1,
                "llm_running": 4,
                "llm_waiting": 5,
            },
        ]
        summary = _timeline_summary(rows)
        self.assertEqual(summary["tool_authoritative_queue_sample_count"], 1)
        self.assertEqual(
            summary["joint_llm_wait_and_live_tool_pressure_sample_count"], 1
        )
        self.assertEqual(summary["max_llm_waiting"], 5)


class LiveAgentClosedLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_learned_predictor_only_controls_speculative_candidates(
        self,
    ) -> None:
        first_url = "https://example.test/rank-one"
        learned_url = "https://example.test/rank-two"
        calls: list[Invocation] = []

        async def tool_executor(invocation: Invocation):
            calls.append(invocation)
            await asyncio.sleep(0)
            if invocation.tool_name == "search":
                return {
                    "tool": "search",
                    "query": ["q"],
                    "results": [
                        {"url": first_url, "rank": 1, "query_index": 0},
                        {"url": learned_url, "rank": 2, "query_index": 0},
                    ],
                }
            return {"tool": "visit", "pages": [{"url": learned_url, "content": "a"}]}

        class FakePredictor:
            policy = "visible-search-learned-rank-v1"
            artifact_sha256 = "artifact"
            top_k = 1

            def predict_structured_result(self, result):
                # The predictor sees the current result and selects rank two;
                # it does not alter the URL enum passed to the LLM.
                return (result["results"][1]["url"],)

        class FakeLLM:
            events = []

            async def complete(self, **kwargs):
                outputs = [
                    {"name": "search", "arguments": {"query": ["q"]}},
                    {
                        "name": "visit",
                        "arguments": {"url": [learned_url], "goal": "g"},
                    },
                    {"answer": "a", "source_url": learned_url},
                ]
                return LLMCompletion(
                    json.dumps(outputs[kwargs["call_index"]]),
                    0.01,
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    kwargs["request_id"],
                )

        broker = LiveToolBroker(tool_executor, max_workers=2, max_speculative_workers=1)
        experiment = OnlineLearnedClosedLoopExperiment(
            broker=broker,
            llm=FakeLLM(),
            token_counter=ApproximateTokenCounter(),
            speculation_mode="visit",
            visit_top_k=1,
            max_tokens_tool=32,
            max_tokens_answer=32,
            default_tool_service_s=1,
            predicted_visit_result_tokens=10,
            visit_predictor=FakePredictor(),
        )
        result = await experiment.run_task(LiveSource("s", "g", "q"))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["search_urls"], [first_url, learned_url])
        self.assertEqual(result["visit_prediction"]["candidate_urls"], [learned_url])
        self.assertTrue(result["visit_prediction"]["selected_url_exact_hit"])
        visit_rows = [row for row in broker.tool_records() if row["tool"] == "visit"]
        self.assertTrue(any(row["speculative"] for row in visit_rows))
        self.assertEqual(sum(call.tool_name == "visit" for call in calls), 1)
        await broker.close()

    async def test_llm_client_sends_guided_json_for_final_call(self) -> None:
        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def json(self, *, content_type=None):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"answer":"a","source_url":"https:\\/\\/x"}'
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }

        class FakeSession:
            def __init__(self) -> None:
                self.payloads: list[dict] = []

            def post(self, url, *, json, timeout):
                self.payloads.append(dict(json))
                return FakeResponse()

        session = FakeSession()
        client = LiveLLMClient(
            session, server_url="http://vllm.test", model="model", timeout_s=5
        )
        final_schema = final_answer_schema("https://x")
        await client.complete(
            task_id="task",
            call_index=2,
            messages=[{"role": "user", "content": "visit result"}],
            request_id="request",
            max_tokens=256,
            schema=final_schema,
            prompt_tokens=10,
            scheduler_meta={},
        )
        self.assertEqual(session.payloads[0]["guided_json"], final_schema)
        self.assertEqual(session.payloads[0]["max_tokens"], 256)
        self.assertEqual(client.events[0]["output_mode"], "guided_json")
        self.assertTrue(client.events[0]["guided_json_requested"])

        schema = search_schema("q")
        await client.complete(
            task_id="task",
            call_index=0,
            messages=[{"role": "user", "content": "task"}],
            request_id="request-guided",
            max_tokens=64,
            schema=schema,
            prompt_tokens=10,
            scheduler_meta={},
        )
        self.assertEqual(session.payloads[1]["guided_json"], schema)
        self.assertTrue(client.events[1]["guided_json_requested"])

    async def test_llm_client_sends_fixed_final_grammar_and_token_bounds(self) -> None:
        url = "https://example.test/source"
        semantic = canonical_json({"answer": "bounded", "source_url": url})

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def json(self, *, content_type=None):
                return {
                    "choices": [
                        {
                            "message": {"content": semantic + " " * 20},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": FINAL_COMPLETION_TOKEN_COUNT,
                        "total_tokens": 10 + FINAL_COMPLETION_TOKEN_COUNT,
                    },
                }

        class FakeSession:
            def __init__(self) -> None:
                self.payloads: list[dict] = []

            def post(self, url, *, json, timeout):
                self.payloads.append(dict(json))
                return FakeResponse()

        session = FakeSession()
        client = LiveLLMClient(
            session, server_url="http://vllm.test", model="model", timeout_s=5
        )
        grammar = final_answer_fixed_completion_grammar(url)
        completion = await client.complete(
            task_id="task",
            call_index=2,
            messages=[{"role": "user", "content": "visit result"}],
            request_id="request",
            max_tokens=FINAL_COMPLETION_TOKEN_COUNT,
            min_tokens=FINAL_COMPLETION_TOKEN_COUNT,
            schema=None,
            grammar=grammar,
            prompt_tokens=10,
            scheduler_meta={},
        )

        payload = session.payloads[0]
        self.assertEqual(payload["guided_grammar"], grammar)
        self.assertNotIn("guided_json", payload)
        self.assertEqual(payload["min_tokens"], FINAL_COMPLETION_TOKEN_COUNT)
        self.assertEqual(payload["max_tokens"], FINAL_COMPLETION_TOKEN_COUNT)
        self.assertEqual(completion.finish_reason, "length")
        self.assertEqual(client.events[0]["output_mode"], "guided_grammar")
        self.assertTrue(client.events[0]["guided_grammar_requested"])
        self.assertEqual(
            client.events[0]["guided_grammar_sha256"],
            hashlib.sha256(grammar.encode()).hexdigest(),
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            await client.complete(
                task_id="task",
                call_index=2,
                messages=[],
                request_id="invalid",
                max_tokens=FINAL_COMPLETION_TOKEN_COUNT,
                min_tokens=FINAL_COMPLETION_TOKEN_COUNT,
                schema=final_answer_schema(url),
                grammar=grammar,
                prompt_tokens=1,
                scheduler_meta={},
            )

    async def test_frozen_call_graph_visits_expected_url_after_live_search(self) -> None:
        expected_url = "https://example.test/frozen"
        observed_url = "https://example.test/current-search-result"
        calls: list[Invocation] = []

        async def tool_executor(invocation: Invocation):
            calls.append(invocation)
            await asyncio.sleep(0)
            if invocation.tool_name == "search":
                return {
                    "tool": "search",
                    "query": ["q"],
                    "results": [{"url": observed_url}],
                }
            return {
                "tool": "visit",
                "pages": [{"url": expected_url, "content": "fresh live page"}],
            }

        class FakeLLM:
            events = []

            def __init__(self) -> None:
                self.prompts: list[list[dict[str, str]]] = []
                self.schemas: list[dict | None] = []

            async def complete(self, **kwargs):
                self.prompts.append([dict(item) for item in kwargs["messages"]])
                self.schemas.append(kwargs["schema"])
                outputs = [
                    {"name": "search", "arguments": {"query": ["q"]}},
                    {
                        "name": "visit",
                        "arguments": {"url": [expected_url], "goal": "g"},
                    },
                    {"answer": "a", "source_url": expected_url},
                ]
                output = outputs[kwargs["call_index"]]
                return LLMCompletion(
                    json.dumps(output) if isinstance(output, dict) else output,
                    0.01,
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    kwargs["request_id"],
                )

        broker = LiveToolBroker(
            tool_executor, max_workers=2, max_speculative_workers=1
        )
        llm = FakeLLM()
        experiment = LiveClosedLoopExperiment(
            broker=broker,
            llm=llm,
            token_counter=ExactTestTokenCounter(),
            speculation_mode="visit",
            visit_top_k=1,
            max_tokens_tool=32,
            max_tokens_answer=32,
            default_tool_service_s=1,
            predicted_visit_result_tokens=10,
            call_graph_mode="frozen",
        )
        result = await experiment.run_task(
            LiveSource("s", "g", "q", expected_url=expected_url)
        )
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["search_result_contains_expected_url"])
        self.assertEqual(result["selected_url"], expected_url)
        self.assertEqual(
            calls[-1], Invocation("visit", {"url": [expected_url], "goal": "g"})
        )
        visit_items = llm.schemas[1]["properties"]["arguments"]["properties"][
            "url"
        ]["items"]
        self.assertEqual(visit_items["enum"], [expected_url])
        self.assertEqual(llm.schemas[2], final_answer_schema(expected_url))
        self.assertIn(observed_url, llm.prompts[1][-1]["content"])
        self.assertIn("fresh live page", llm.prompts[2][-1]["content"])
        self.assertEqual(result["answer"], {"answer": "a", "source_url": expected_url})
        await broker.close()

    async def test_closed_loop_fixed_final_is_opt_in_and_tool_calls_are_compact(
        self,
    ) -> None:
        selected_url = "https://example.test/source"

        async def tool_executor(invocation: Invocation):
            await asyncio.sleep(0)
            if invocation.tool_name == "search":
                return {
                    "tool": "search",
                    "query": ["q"],
                    "results": [{"url": selected_url}],
                }
            return {
                "tool": "visit",
                "pages": [{"url": selected_url, "content": "fresh page"}],
            }

        class FakeLLM:
            def __init__(self) -> None:
                self.events = []
                self.requests: list[dict] = []

            async def complete(self, **kwargs):
                request = dict(kwargs)
                request["messages"] = [dict(row) for row in kwargs["messages"]]
                self.requests.append(request)
                index = kwargs["call_index"]
                if index == 0:
                    content = json.dumps(
                        {"name": "search", "arguments": {"query": ["q"]}},
                        indent=2,
                    )
                    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    finish_reason = "stop"
                elif index == 1:
                    content = json.dumps(
                        {
                            "name": "visit",
                            "arguments": {"url": [selected_url], "goal": "g"},
                        },
                        indent=2,
                    )
                    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    finish_reason = "stop"
                else:
                    content = canonical_json(
                        {"answer": "grounded answer", "source_url": selected_url}
                    ) + " " * 64
                    usage = {
                        "prompt_tokens": 1,
                        "completion_tokens": FINAL_COMPLETION_TOKEN_COUNT,
                        "total_tokens": 1 + FINAL_COMPLETION_TOKEN_COUNT,
                    }
                    finish_reason = "length"
                return LLMCompletion(
                    content,
                    0.01,
                    usage,
                    kwargs["request_id"],
                    finish_reason,
                )

        invalid_broker = LiveToolBroker(tool_executor, max_workers=1)
        with self.assertRaisesRegex(ValueError, "None or 192"):
            LiveClosedLoopExperiment(
                broker=invalid_broker,
                llm=FakeLLM(),
                token_counter=ApproximateTokenCounter(),
                speculation_mode="off",
                visit_top_k=1,
                max_tokens_tool=32,
                max_tokens_answer=64,
                default_tool_service_s=1,
                predicted_visit_result_tokens=10,
                fixed_final_completion_tokens=191,
            )
        with self.assertRaisesRegex(ValueError, "exact model tokenizer"):
            LiveClosedLoopExperiment(
                broker=invalid_broker,
                llm=FakeLLM(),
                token_counter=ApproximateTokenCounter(),
                speculation_mode="off",
                visit_top_k=1,
                max_tokens_tool=32,
                max_tokens_answer=64,
                default_tool_service_s=1,
                predicted_visit_result_tokens=10,
                fixed_final_completion_tokens=FINAL_COMPLETION_TOKEN_COUNT,
            )
        await invalid_broker.close()

        broker = LiveToolBroker(tool_executor, max_workers=1)
        llm = FakeLLM()
        experiment = LiveClosedLoopExperiment(
            broker=broker,
            llm=llm,
            token_counter=ExactTestTokenCounter(),
            speculation_mode="off",
            visit_top_k=1,
            max_tokens_tool=32,
            max_tokens_answer=64,
            default_tool_service_s=1,
            predicted_visit_result_tokens=10,
            fixed_final_completion_tokens=FINAL_COMPLETION_TOKEN_COUNT,
        )
        result = await experiment.run_task(LiveSource("s", "g", "q"))

        self.assertTrue(result["ok"], result)
        self.assertEqual(len(llm.requests), 3)
        for request in llm.requests[:2]:
            self.assertIsNotNone(request["schema"])
            self.assertIsNone(request["grammar"])
            self.assertEqual(request["min_tokens"], 0)
            self.assertEqual(request["max_tokens"], 32)
        final_request = llm.requests[2]
        self.assertIsNone(final_request["schema"])
        self.assertEqual(
            final_request["grammar"],
            final_answer_fixed_completion_grammar(selected_url),
        )
        self.assertEqual(
            final_request["min_tokens"], FINAL_COMPLETION_TOKEN_COUNT
        )
        self.assertEqual(
            final_request["max_tokens"], FINAL_COMPLETION_TOKEN_COUNT
        )
        expected_search = canonical_json(
            {"name": "search", "arguments": {"query": ["q"]}}
        )
        expected_visit = canonical_json(
            {
                "name": "visit",
                "arguments": {"url": [selected_url], "goal": "g"},
            }
        )
        self.assertEqual(llm.requests[1]["messages"][-2]["content"], expected_search)
        assistant_messages = [
            row["content"]
            for row in final_request["messages"]
            if row["role"] == "assistant"
        ]
        self.assertEqual(assistant_messages, [expected_search, expected_visit])
        self.assertEqual(
            result["output_contract"]["policy_version"],
            FIXED_OUTPUT_CONTRACT_POLICY_VERSION,
        )
        self.assertEqual(
            result["final_answer_contract"]["total_completion_tokens"],
            FINAL_COMPLETION_TOKEN_COUNT,
        )
        self.assertEqual(result["final_answer_contract"]["finish_reason"], "length")
        await broker.close()

    async def test_generated_visit_claims_real_speculation_and_drives_final_prompt(
        self,
    ) -> None:
        calls: list[Invocation] = []

        async def tool_executor(invocation: Invocation):
            calls.append(invocation)
            await asyncio.sleep(0)
            if invocation.tool_name == "search":
                return {
                    "tool": "search",
                    "query": ["Apollo 11"],
                    "results": [
                        {
                            "rank": 1,
                            "title": "Apollo 11",
                            "url": "https://example.test/apollo",
                        }
                    ],
                }
            return {
                "tool": "visit",
                "goal": "When did it land?",
                "pages": [
                    {
                        "url": "https://example.test/apollo",
                        "title": "Apollo 11",
                        "content": "Apollo 11 landed in July 1969.",
                    }
                ],
            }

        class FakeLLM:
            def __init__(self) -> None:
                self.events = []
                self.prompts = []

            async def complete(self, **kwargs):
                self.prompts.append(kwargs["messages"])
                index = kwargs["call_index"]
                values = [
                    {
                        "name": "search",
                        "arguments": {"query": ["Apollo 11"]},
                    },
                    {
                        "name": "visit",
                        "arguments": {
                            "url": ["https://example.test/apollo"],
                            "goal": "When did it land?",
                        },
                    },
                ]
                if index == 2:
                    values.append(
                        {
                            "answer": "  July\n1969  ",
                            "source_url": "https://example.test/apollo",
                        }
                    )
                content = json.dumps(values[index])
                return LLMCompletion(
                    content=content,
                    duration_s=0.01,
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                    request_id=kwargs["request_id"],
                )

        broker = LiveToolBroker(
            tool_executor,
            max_workers=2,
            max_speculative_workers=1,
            ttl_s=5,
        )
        llm = FakeLLM()
        experiment = LiveClosedLoopExperiment(
            broker=broker,
            llm=llm,
            token_counter=ApproximateTokenCounter(),
            speculation_mode="search_visit",
            visit_top_k=1,
            max_tokens_tool=32,
            max_tokens_answer=32,
            default_tool_service_s=1,
            predicted_visit_result_tokens=50,
        )
        result = await experiment.run_task(
            LiveSource(
                source_id="apollo",
                question="When did it land?",
                search_query="Apollo 11",
            )
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual([record["source"] for record in result["tools"]], [
            "promoted_inflight",
            "promoted_inflight",
        ])
        self.assertEqual(len(calls), 2)
        self.assertIn("July 1969", llm.prompts[2][-1]["content"])
        self.assertEqual(result["answer"]["answer"], "July 1969")
        recovery = result["guided_json_recovery"]
        self.assertEqual(
            recovery["policy_version"], GUIDED_JSON_RECOVERY_POLICY_VERSION
        )
        self.assertEqual(recovery["parsed_call_count"], 2)
        self.assertEqual(recovery["recovery_count"], 0)
        self.assertEqual([row["call_index"] for row in recovery["calls"]], [0, 1])
        contract = result["output_contract"]
        self.assertEqual(contract["policy_version"], OUTPUT_CONTRACT_POLICY_VERSION)
        self.assertEqual([row["call_index"] for row in contract["calls"]], [0, 1, 2])
        self.assertEqual(
            contract["calls"][2]["mode"],
            "guided_json_strict_local_projection",
        )
        self.assertTrue(contract["calls"][2]["guided_json_requested"])
        self.assertTrue(contract["calls"][2]["json_parse_attempted"])
        self.assertFalse(contract["calls"][2]["recovery_allowed"])
        self.assertFalse(contract["calls"][2]["recovery_applied"])
        self.assertTrue(contract["calls"][2]["contract_succeeded"])
        final_contract = result["final_answer_contract"]
        self.assertEqual(
            final_contract["policy_version"], FINAL_ANSWER_CONTRACT_POLICY_VERSION
        )
        self.assertTrue(final_contract["canonicalization_changed"])
        self.assertTrue(final_contract["object_constructed_locally"])
        self.assertEqual(broker.stats.authoritative_executions, 0)
        self.assertEqual(broker.stats.commits, 2)
        await broker.close()

    async def test_canary_bypasses_exact_visit_prediction(self) -> None:
        async def tool_executor(invocation: Invocation):
            await asyncio.sleep(0)
            if invocation.tool_name == "search":
                return {
                    "tool": "search",
                    "query": ["q"],
                    "results": [{"url": "https://example.test/u"}],
                }
            return {"tool": "visit", "pages": [{"content": "answer"}]}

        class FakeLLM:
            events = []
            outputs = [
                {"name": "search", "arguments": {"query": ["q"]}},
                {
                    "name": "visit",
                    "arguments": {"url": ["https://example.test/u"], "goal": "g"},
                },
                {
                    "answer": "a",
                    "source_url": "https://example.test/u",
                },
            ]

            async def complete(self, **kwargs):
                output = self.outputs[kwargs["call_index"]]
                return LLMCompletion(
                    json.dumps(output) if isinstance(output, dict) else output,
                    0.01,
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    kwargs["request_id"],
                )

        broker = LiveToolBroker(tool_executor, max_workers=2, max_speculative_workers=1)
        experiment = LiveClosedLoopExperiment(
            broker=broker,
            llm=FakeLLM(),
            token_counter=ApproximateTokenCounter(),
            speculation_mode="visit",
            visit_top_k=1,
            max_tokens_tool=32,
            max_tokens_answer=32,
            default_tool_service_s=1,
            predicted_visit_result_tokens=10,
        )
        result = await experiment.run_task(
            LiveSource("s", "g", "q"), visit_speculation_eligible=False
        )
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["visit_canary"])
        visit_records = [
            row for row in broker.tool_records() if row["tool"] == "visit"
        ]
        self.assertFalse(any(row["speculative"] for row in visit_records))
        self.assertTrue(
            any(row["speculation_eligible"] is False for row in visit_records)
        )
        await broker.close()

    async def test_search_only_speculation_does_not_prefetch_visit(self) -> None:
        calls: list[str] = []

        async def tool_executor(invocation: Invocation):
            calls.append(invocation.tool_name)
            await asyncio.sleep(0)
            if invocation.tool_name == "search":
                return {
                    "tool": "search",
                    "query": ["q"],
                    "results": [{"url": "https://example.test/u"}],
                }
            return {"tool": "visit", "pages": [{"content": "answer"}]}

        class FakeLLM:
            events = []
            outputs = [
                {"name": "search", "arguments": {"query": ["q"]}},
                {
                    "name": "visit",
                    "arguments": {"url": ["https://example.test/u"], "goal": "g"},
                },
                {
                    "answer": "a",
                    "source_url": "https://example.test/u",
                },
            ]

            async def complete(self, **kwargs):
                output = self.outputs[kwargs["call_index"]]
                return LLMCompletion(
                    json.dumps(output) if isinstance(output, dict) else output,
                    0.01,
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    kwargs["request_id"],
                )

        broker = LiveToolBroker(tool_executor, max_workers=2, max_speculative_workers=1)
        experiment = LiveClosedLoopExperiment(
            broker=broker,
            llm=FakeLLM(),
            token_counter=ApproximateTokenCounter(),
            speculation_mode="search",
            visit_top_k=1,
            max_tokens_tool=32,
            max_tokens_answer=32,
            default_tool_service_s=1,
            predicted_visit_result_tokens=10,
        )
        result = await experiment.run_task(LiveSource("s", "g", "q"))
        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, ["search", "visit"])
        search_rows = [row for row in broker.tool_records() if row["tool"] == "search"]
        visit_rows = [row for row in broker.tool_records() if row["tool"] == "visit"]
        self.assertTrue(any(row["speculative"] for row in search_rows))
        self.assertFalse(any(row["speculative"] for row in visit_rows))
        await broker.close()


if __name__ == "__main__":
    unittest.main()
