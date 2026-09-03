# Frozen protocol: bounded pattern cache and visit-abstain gate

Freeze date: 2026-08-31 UTC. This protocol was finalized before any model-generated
trace for `new-whole-session-holdout-v1` existed. The workload questions and their
provenance were fixed in advance; no holdout outcome, search result, visit, or model
response was available for predictor selection.

## Question and policies

The prospective comparison is between:

- `M0`: the legacy current-response, displayed-rank URL mapper, invoked blindly.
- `Pattern-v2`: an exact-URL, non-neural pattern matcher with a session-local bounded
  cache and a deterministic `next_tool=visit` abstain gate.

Both policies use the same rank-count artifact fitted on the 100 historical sessions.
No Transformer embedding, neural network, gradient descent, or backpropagation is
used.

The frozen Pattern-v2 runtime is:

- Policy: `rank-recency-visited-cache-gate-v2`; artifact version 2.
- URL identity: exact raw HTTP(S) strings; no normalization or semantic matching.
- Current response: unbounded for the current decision.
- History cache: session-local LRU 64; only search ages 1 and 2 are eligible.
- Visited cache: independent session-local LRU 64, updated only after a committed
  visit result. A visited URL is penalized, not prohibited.
- Score: `log(rank_count + 0.5) - 1.5 * search_age - was_visited`.
- Top-1: the legacy M0 current-response Top-1 is preserved exactly.
- Remaining positions: deterministic Top-5 selection over the current/history union.
- Gate: abstain on no executable candidates; also abstain exactly when
  `query_count >= 10` and `consecutive_search_streak == 2`; otherwise admit.
- Missing/unmatched gate patterns fail open; malformed API inputs fail closed.

The age coefficient was revised from 1.0 to 1.5 using historical data only, before
this freeze. With 1.0, one grouped fold sat on a fragile score boundary and lost
three targets. The 1.5 value lies inside an old-data equivalence plateau: across 20
alternative whole-session fold seeds its worst result was no lower than M0. This is
an explicitly recorded old-only post-hoc robustness revision, not holdout tuning.

## Historical development evidence

All development uses `/home/aiscuser/PASTE-Qwen-DR/traces/my_traces` only.

- Strict historical 70-session grouped OOF, 148 visit targets:
  M0 `33 / 64 / 81`; Pattern-v2 `33 / 65 / 81` at Top-1/3/5.
- All-100 grouped OOF, 236 visit targets:
  M0 `50 / 102 / 131`; Pattern-v2 `50 / 105 / 134`.
- All-100 gate confusion: TP/FP/TN/FN = `116 / 198 / 26 / 0`.
- Local 6,800-call benchmark: p50 `0.308 ms`, p95 `0.611 ms`, p99
  `0.716 ms`, maximum `15.678 ms`.

The historical results are development evidence, not prospective confirmation.

## Fixed new workload

- Workload ID: `new-whole-session-holdout-v1`.
- Workload file SHA-256:
  `88d15dfea2f6e1abbce20086f608bc0f324ef8549a47359b425f60cba0ac7f87`.
- Exactly 30 ordered, unique sources selected without replacement by SHA-256 ordering
  with seed `new-whole-session-holdout-v1` from the pre-existing 50-task source.
- Source file SHA-256:
  `79e288e1f6ab719512035d3a074ff96752130cac85539656f4113c4489e72c53`.
- Exact and case/whitespace-normalized question overlap with the old 100 sessions: 0.
- All 30 offered sessions are retained. There is no replacement, replenishment,
  result-based resampling, or rerun of a failed session.

The scientific prompts are intentionally difficult and some omit inputs mentioned in
their task description. `max_calls=9` remains fixed to match the historical session
budget. Collection failures and max-call exhaustion remain in the denominator and
manifest; they are not repaired by collecting substitutes.

## Fixed collection runtime

- Model: `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`.
- Model revision:
  `4b0ac5767427a55d08a254f0367e2934976598e0`.
- Local OpenAI-compatible endpoint: `http://127.0.0.1:8200`.
- vLLM: tensor parallel 4 on GPUs 0-3, bfloat16, context 114688,
  `max_num_batched_tokens=8192`, `max_num_seqs=2`, FCFS, prefix caching enabled,
  CUDA graph size 1, GPU memory utilization 0.84.
- Sessions: sequential in workload order.
- Generation: maximum 9 LLM calls/session, maximum 8192 output tokens/call,
  temperature 0, top-p 1, seed 0, request timeout 300 seconds.
- Search: Bing HTML restricted by the executor to Wikipedia results, maximum 5
  results/query.
- Visit: Jina fetch of the exact URL selected from a prior search, maximum 6
  URLs/call.
- HTTP: tool timeout 20 seconds, at most 2 explicit attempts, fixed 5-second retry
  backoff, and a shared per-tool minimum request-start interval of 1 second for both
  search and visit.
- aiohttp's hidden connection retry must be verified disabled in the final manifest.

The final non-holdout integration smoke is
`reproduction/results/pattern_cache_collector_smoke/run3`: one complete synthetic
session, all requested tools committed, and two committed search decisions replayed
from explicit raw `tool_result` events. It is not evaluation eligible.

Collection writes a claim with `O_EXCL`, checkpoints every event atomically, persists
each successful raw tool result before the next LLM request, and retains a complete
HTTP attempt ledger on failure. The predictor is not run during collection, so it
cannot influence the model's tool sequence.

## One-shot evaluation semantics

Before reading any collected trace, the evaluator must restore the checksummed v2
artifact and create `NEW_HOLDOUT_EVALUATION_STARTED.json` with `O_EXCL`. It then
validates the collection manifest fail-closed against the fixed workload: status,
30 ordered records, source/question/provenance bindings, trace names and SHA-256,
session start/end records, event counts, committed-result counts, and absence of
extra JSONL files.

Only `tool_result(commit_status=committed)` events enter the primary evaluation:

- A committed search raw result creates a decision and its candidate state.
- A committed next visit supplies exact authoritative HTTP(S) target URLs and updates
  visited state.
- Requested but uncommitted tools are reported separately and never enter cache state
  or the primary exact-URL denominator.
- Failed sessions and sessions with zero evaluable decisions remain among all 30
  whole-session bootstrap units.

Metrics are exact target recall at Top-1/3/5, all-window prediction precision/waste,
gate confusion and recall, candidate/dispatch counts, requested-versus-committed tool
counts, and predictor-only wall-clock latency. Uncertainty uses 10,000 paired
bootstrap replicates with the whole session as the resampling unit. The bootstrap
estimand is explicitly gated Pattern-v2 minus M0; a resample containing no visit
target has undefined conditional recall and is replaced, never imputed as zero.

Because the new Bing/Jina/Wikipedia transport differs from the historical SearXNG
transport, new absolute percentages are not compared directly with the old
19.3%/43.2%/55.7%. Only paired M0-versus-Pattern-v2 differences on the same new
sessions are confirmatory.

## Frozen acceptance rules

Data adequacy requires at least 80 committed executable visit-target URLs and at
least 20 of the 30 sessions to contain a committed search decision. If either fails,
accuracy and gate conclusions are labeled `inconclusive`; all observed results are
still reported.

Subject to data adequacy:

- Ranker success: Pattern-v2 exact Top-1/3/5 hit counts are each no lower than M0,
  and at least one of Top-3 or Top-5 is strictly higher.
- Gate success: committed-visit recall is at least 95%, gated candidate dispatches
  are strictly fewer than the same cache ranker without the gate, and gated exact
  Top-1/3/5 counts are not lower than its ungated counts. If the mined gate never
  fires, its result is `inconclusive`, not accepted.
- Runtime success: predictor-only wall latency has both p99 and observed maximum
  below 100 ms.

Promotion requires all applicable rules. Ranker, gate, data adequacy, and runtime are
also reported independently so a failed ranker cannot be hidden by reduced waste.
No rule, parameter, workload member, or failed session may be changed after the
collection-start marker is created.

## Frozen bindings

- Predictor source SHA-256:
  `be1ead62e99c30c5df9ad8c07cb6013847f4bf705e9d7ddfc973a38625d3f967`.
- Collector source SHA-256:
  `75b651dbb9d778cc7126e532218ba05333448bb7a873b4d790f3c37e4012dfa3`.
- Collector CLI SHA-256:
  `ccd393a72dbafdb41d205b6a467e936346356b53af9491eebac00915486127bb`.
- Live executor SHA-256:
  `dbe600338efbc19d2f9e884dc0f1f5dc3baa2d776d4f67d54c8d99bbb6a9ad39`.
- Evaluation source SHA-256:
  `90bf7a1d42c6b60b54dd2247a750245a201ca7133410038800e079d5120e2823`.
- Artifact internal SHA-256:
  `4365827d65bb0a4e5396197989e5ee93387a058f87e141ec28059b83193e6732`.
- Artifact file SHA-256:
  `78278eec8d60b5527e0954c38f193227459acd136af87a1c1d22b863bd1c029b`.
- Development metrics file SHA-256:
  `3fb355809146b7e36f3aa1106d908586c12684dfa8abd807db44843e0bc9775c`.
- Final smoke manifest SHA-256:
  `c259b322ad694698bfa518c623702f1a98dcaa2748bd60b05c557211178e177f`.
- Final smoke trace SHA-256:
  `dfa6c9dfae7b965391e869be2eba42cdc95eca54eb6ae6061290054d0e900774`.
