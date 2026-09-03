# PASTE Live Reproduction

PASTE reduces agent end-to-end latency by overlapping predictable tool
execution with LLM generation and coordinating the sessions that return to the
LLM engine. This reproduction evaluates the design in a live Qwen DeepResearch
loop with real vLLM serving, Bing search, Jina visit, bounded tool workers, and
a shared tool queue.

## Supported Paths

The reproduction keeps five complementary entry points:

| Path | What it establishes | Entry point |
|---|---|---|
| Trace-learned speculative tools | Held-out prediction, exact confirmation, bounded overlap, no LLM co-design | `scripts/run_speculative_tool_execution.sh` |
| Agent-oriented baseline boundary | Executed broker replay plus a conservative Murakkab/llm-d/Dynamo abstraction-boundary sensitivity | `scripts/run_agent_baseline_boundary_replay.py` |
| Adaptive-width ablation | Profile-guided selection of PASTE speculation intensity; not a Murakkab system comparison | `scripts/run_murakkab_paste_comparison.py` |
| Online speculative tools | Live Qwen-DR generation plus live search/visit using the same learned predictor | `scripts/run_online_speculative_execution.py` |
| Existing full PASTE system | Native prefix plus live A/B/E/F tool and Joint scheduling experiments | `scripts/run_native_prefix_causal_dev.py`, `scripts/run_live_joint_formal_v9_matrix.py` |

The trace path is the controlled component experiment; the baseline-boundary
path maps its event order to current agent-serving abstractions without
pretending to benchmark those systems; the adaptive-width path is retained as
a PASTE-only offline ablation and not as a Murakkab result; the online path
shows that the component can execute causally in a live Qwen agent loop; the
existing full-system path remains the end-to-end reproduction.

## External Speculative Action reference

A source-only snapshot of the external Speculative Action workloads is kept
under
[`external/speculative_action_minimal/`](external/speculative_action_minimal/PASTE_INTEGRATION.md).
It pins the upstream commit and retains the required licenses, runners, and
small fixtures while excluding generated results, bulk trajectories, and the
full HotPotQA dataset. The code is comparison material and is not imported by
the PASTE-Qwen-DR runtime.

## Reviewer Evidence Bundle

The consolidated response to common comments 2, 3, and 5 is in
[results/reviewer_comments_2_3_5/REPORT.md](results/reviewer_comments_2_3_5/REPORT.md).
Its source audits are the
[metric/load report](results/reviewer_comment2_load_sweep/REPORT.md),
[scheduler robustness report](results/scheduler_robustness/REPORT.md),
[Granite fail-closed portability audit](results/scheduler_cross_model_portability/FAILED_GRANITE_C5K_R1_AUDIT.md),
and [agent-baseline boundary report](results/agent_baseline_boundary/REPORT.md).
The Granite one-shot stopped in baseline A on six model-output contract
failures, so it supplies no cross-model A/E latency result and is not rerunnable.

## System Design

The implementation follows the three mechanisms in the paper:

1. **Pattern-aware tool speculation.** After search, the runtime instantiates a
   concrete visit prediction and places it in the shared tool queue while the
   LLM continues generation.
2. **Non-interference speculation lifecycle.** Speculative results remain
   isolated until an exact authoritative invocation arrives. A match reuses a
   completed result or promotes queued/in-flight work.
3. **Joint LLM–Tool Co-Scheduling.** Tool state, session progress, context
   length, and physical-KV pressure guide LLM admission and ordering.

```text
agent session ──► LLM generation ──► authoritative tool call
      │                 │                       │
      │                 └── predicted visit ───┤
      │                                         ▼
      └──────── Joint session state ◄── shared tool queue
                           │             Bing / Jina
                           ▼
                    vLLM scheduler
```

The live workload uses an execution-aware signal. Exact running or completed
predictions provide readiness information; queued predictions contribute
through queue pre-positioning and authoritative promotion. A shared 2.5-second
visit attempt gate covers initial requests and retries.

Native prefix caching complements the closed loop by reusing repeated context
across the three LLM turns. Its causal effect is measured separately with
native FCFS and a P0/P1 reverse-block design.

## Implementation

| Component | Implementation |
|---|---|
| Agent loop and prediction contract | `paste_repro/live_agent.py` |
| Tool speculation scheduler and shared queue | `paste_repro/live_broker.py` |
| Bing/Jina execution and HTTP attempt control | `paste_repro/live_executor.py` |
| Live experiment driver | `../scripts/run_live_tool_llm_experiment.py` |
| LLM–Tool Co-Scheduler | `../scripts/pythonhooks/sched_policy_patch.py` |
| Native prefix experiment | `scripts/run_native_prefix_causal_dev.py` |
| Four-cell experiment | `scripts/run_live_joint_formal_v9_matrix.py` |
| Shared trace-learned visit predictor | `paste_repro/tool_prediction.py` |
| Agent-baseline boundary replay | `paste_repro/baseline_boundary.py`, `scripts/run_agent_baseline_boundary_replay.py` |
| Murakkab-inspired typed workflow and optimizer | `paste_repro/murakkab_optimizer.py`, `scripts/run_murakkab_paste_comparison.py` |
| Online learned-speculation entry point | `scripts/run_online_speculative_execution.py` |
| Isolated online learned agent/driver | `paste_repro/online_learned_agent.py`, `../scripts/run_online_trace_learned_experiment.py` |

## Speculative Tool Execution (without LLM co-design)

The trace-learned tool-side component is also reproducible on its own. It is
separate from the frozen-URL formal matrix and does not modify LLM scheduling
or constrain the URL selected by the LLM.

Its causal contract is:

1. split whole trace sessions into deterministic 70/30 train and held-out sets;
2. learn which displayed within-query result ranks historically flow into the
   immediately following `visit`;
3. late-bind those learned ranks to URLs in the current, already-visible search
   response;
4. keep speculative results isolated and reuse/promote them only when the
   authoritative session-scoped URL invocation matches exactly.

Run the CPU-only experiment with:

```bash
bash reproduction/scripts/run_speculative_tool_execution.sh
```

It writes a checksummed mapper and a unified report to
`reproduction/artifacts/speculative_tool_execution/`. On the repository trace
snapshot with `top_k=5`, the held-out result is:

| Metric | Result |
|---|---:|
| Transition/example hit rate | `76.47%` |
| Authoritative URL invocation hit rate | `55.68%` |
| Exposed tool stall | `38.514 → 19.647 s` |
| Stall reduction | `48.99%` |

The recorded stall reduction is a trace-timestamp counterfactual, bounded by
the preceding LLM decision window; the scheduler replay itself performs no
network requests. See the [standalone report](results/speculative_tool_execution/REPORT.md)
for the interpretation boundary and exact counts.

### Contextual predictor optimization audit

An offline 49-feature linear pairwise reranker was also tested. It is not a
neural network: inference is fixed string/position feature extraction, one
linear score per visible URL, and a stable sort. Session-grouped training OOF
improved, and the measured structured-result prediction path was well below the
100 ms gate (`p99=11.35 ms`, observed maximum `24.57 ms`). On the unchanged
outer sessions, however, Top-1/3/5 changed from
`19.3% / 43.2% / 55.7%` to `18.2% / 40.9% / 56.8%`. The predeclared gate
therefore rejected it and retained the legacy rank-only mapper. A deterministic
slot-5 backoff also failed (`54.5%` Top-5).

See the [frozen optimization report](results/predictor_optimization/REPORT.md)
and [backoff diagnostic](results/predictor_hybrid_backoff/REPORT.md). The audit
also identifies the next prospective direction: earlier causally visible
search results raise the exact Top-5 oracle from `78.4%` to `89.8%`, but need a
recency-aware matcher and a new whole-session holdout.

For the agent-oriented baseline comparison, run
`python reproduction/scripts/run_agent_baseline_boundary_replay.py`. It adds an
idealized 1–2× inference-only sensitivity and an explicit capability/composition
audit for Murakkab, llm-d, and NVIDIA Dynamo. It is a semantic boundary replay,
not a throughput benchmark of those systems; see the
[baseline report](results/agent_baseline_boundary/REPORT.md).

### Murakkab-inspired adaptive-width ablation

No runnable official Murakkab artifact is linked from the paper, USENIX page,
or author publication pages as of 2026-08-31. The repository therefore includes
an idea-level reproduction scoped to the configuration surface that can be
tested fairly here:

```bash
python reproduction/scripts/run_murakkab_paste_comparison.py
```

The runner deterministically materializes disjoint 40/30/30 whole-session
calibration, tuning, and final roles. It fits the URL-rank mapper on calibration,
profiles `top_k=0..5` on calibration and tuning, then selects the minimum
conservative admitted-tool request units that satisfy each latency SLO plus its
declared margin. The final role is evaluated with exact-match isolated broker
replays.

| Equal-weight SLO mix | Stall reduction | Tool request units / authoritative call | Aggregate SLO tiers met |
|---|---:|---:|---:|
| Demand only (`k=0`) | `0.00%` | `1.000` | `1/4` |
| Static PASTE (`k=5`) | `15.83%` | `3.188` | `4/4` |
| Murakkab-inspired PASTE (`k=0/3/4/5`) | `10.10%` | `2.268` | `4/4` |

These numbers are retained only as an adaptive-`top_k` PASTE ablation. They do
not compare Murakkab with PASTE: the resource metric is an admission-count
proxy, the SLO tiers and weights are synthetic, and this path does not run the
fixed Tongyi/vLLM/4×A100 deployment. The prior system-comparison interpretation
is superseded by the [fixed-model same-setup protocol](results/murakkab_paste/FIXED_MODEL_SAME_SETUP_PROTOCOL.md).
See the [superseded report](results/murakkab_paste/REPORT.md),
[machine-readable evidence](results/murakkab_paste/comparison.json), and
[configuration](configs/murakkab_paste_trace.json).

The same predictor drives the existing bounded live broker through the isolated
online runner. First create the artifact above, then use these flags with
`../scripts/run_online_trace_learned_experiment.py`:

```text
--call-graph-mode autonomous
--speculation-mode visit
--visit-top-k 5
--visit-prediction-model reproduction/artifacts/speculative_tool_execution/url_rank_mapper.json
```

In this mode all current search URLs remain available to the LLM. The learned
model only chooses which concrete visits enter the speculative tool queue, and
task output records the candidate URLs, artifact checksum, and exact-match hit.
Use a native/unchanged LLM scheduler when measuring this component without LLM
co-design.

### Online Qwen-DR path

With an OpenAI-compatible Qwen DeepResearch vLLM server running, the explicit
online entry point is:

```bash
python reproduction/scripts/run_online_speculative_execution.py \
  --output-dir reproduction/artifacts/online_trace_learned_smoke \
  --source-limit 2
```

It defaults to the checked-in checksummed mapper at
`reproduction/results/tool_only/url_rank_mapper.json` and the autonomous tune
workload. The wrapper fixes `call_graph_mode=autonomous` and visit speculation.
After each live search response, learned ranks are bound to current URLs and
submitted to the bounded broker while Qwen generates its authoritative visit.
All returned URLs remain in Qwen's allowed choice set; a miss executes normally.

Use `--dry-run` to inspect the complete delegated live-runner command without
starting network or model requests.

## Standalone PASTE-Qwen-DR Repository

The supported reproduction can be exported without the upstream/vendor agent
tree or machine-local generated artifacts:

```bash
python reproduction/scripts/export_standalone_repo.py \
  --output ../PASTE-Qwen-DR
```

The exporter preserves the current `reproduction/` layout and includes the
required root live/trace drivers, vLLM scheduler hook, 100 trace files, pinned
requirements, license, tests, workloads, protocols, and checked-in evidence.
It excludes model weights, credentials, logs, runtime state, and the roughly
5 GB of machine-local generated artifacts. The destination is validated with
CPU smoke tests and initialized as a Git repository by default.

An exported checkout can be revalidated independently with:

```bash
python reproduction/scripts/validate_standalone_repo.py \
  --repository-root . --require-manifest --smoke
```

The vLLM integration uses a startup hook and request metadata; the serving API
and native token execution path remain unchanged.

## Evaluation Methodology

### Experimental setup

| Component | Setting |
|---|---|
| Model | `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` |
| Revision | `4b0ac5767427a55d08a254f0367e2934976598e0` |
| Serving engine | vLLM 0.10.1, TP=4, BF16 |
| GPU | 4 × NVIDIA A100-SXM4-40GB |
| Context / native capacity | 16K / 96 sequences |
| Workload | 80 concurrent three-turn tasks, 10K context padding |
| Tool resources | 4 workers, search capacity 3, visit capacity 2 |
| Tool backends | Bing HTML search, `r.jina.ai` visit |
| Output control | 192 completion tokens for every final answer |

The formal workload contains 80 held-out sources. Each cell starts with a fresh
vLLM server, empty prefix cache, and empty broker. The call graph is fixed;
search and visit execute over the live network, while the predicted visit URL
comes from the workload's `expected_url`.

### Ablation matrix

All cells enable native prefix caching and use the same model, workload, output
contract, tool capacity, and HTTP policy.

| Cell | LLM serving | Tool execution |
|---|---|---|
| A | Native FCFS | Demand only |
| B | Native FCFS | Visit speculation |
| E | Joint physical-KV scheduling | Demand only |
| F | Joint physical-KV scheduling | Visit speculation |

This matrix isolates tool-side acceleration (A→B), LLM-side scheduling (A→E),
the speculation increment under Joint scheduling (E→F), and the full system
(A→F).

The three blocks use the registered orders `A-B-E-F`, `B-A-F-E`, and
`A-B-F-E`. Results are folded by source across blocks and evaluated with a
10,000-sample paired bootstrap over 80 source means.

## Reproducing the Experiments

### Environment

```bash
cd /path/to/Qwen-DeepResearch-PASTE
bash reproduction/scripts/setup_env.sh

HF_HOME="$HOME/hf_cache" \
  "$HOME/.conda/envs/paste/bin/python" \
  reproduction/scripts/download_model.py

PASTE_PY=/home/aiscuser/.conda/envs/paste/bin/python
```

The registered profile uses GPUs `4,5,6,7`, port `8100`, and the model cache at
`/home/aiscuser/hf_cache`. Bing and Jina are accessed through their public HTTP
interfaces.

### Preflight

The preflight validates model, tokenizer, workload, configuration, and code
bindings without starting a server:

```bash
"$PASTE_PY" reproduction/scripts/run_native_prefix_causal_dev.py \
  prefix-preflight --check-only

"$PASTE_PY" reproduction/scripts/run_live_joint_formal_v9_matrix.py \
  formal-v9-preflight --check-only
```

### Native prefix

```bash
"$PASTE_PY" reproduction/scripts/run_native_prefix_causal_dev.py \
  native-prefix-v2-r2
```

This runs two reverse blocks, `P0→P1` and `P1→P0`, with a fresh native FCFS
server for every cell. Results are written to
`artifacts/live_joint/prefix_native_causal_dev_v2/<run-tag>/strict_validation.json`.

### Full live matrix

```bash
"$PASTE_PY" reproduction/scripts/run_live_joint_formal_v9_matrix.py \
  formal-v9-context10k-live-r2
```

The runner executes 12 fresh-server cells: 960 tasks, 2,880 LLM requests,
1,920 authoritative tool commits, and 1,920 live HTTP requests.

Aggregate a completed matrix with:

```bash
RUN_TAG=formal-v9-context10k-live-r2
RUN_ROOT=reproduction/artifacts/live_joint/formal/$RUN_TAG

"$PASTE_PY" reproduction/scripts/aggregate_live_joint_four_cell.py \
  --block "$RUN_TAG-block-1" "$RUN_ROOT"/block-01/{A,B,E,F}/evidence/result.json \
  --block "$RUN_TAG-block-2" "$RUN_ROOT"/block-02/{A,B,E,F}/evidence/result.json \
  --block "$RUN_TAG-block-3" "$RUN_ROOT"/block-03/{A,B,E,F}/evidence/result.json \
  --formal-workload reproduction/workloads/live_joint_wikipedia_frozen_formal_v9.json \
  --output "$RUN_ROOT/strict_four_cell_aggregate.json"
```

The aggregate contains source-paired effects, confidence intervals, latency
tails, time decomposition, HTTP telemetry, and all registered gates.

## Evaluation Results

### End-to-end latency

| Comparison | Mean E2E | Reduction | 95% relative CI | Faster sources |
|---|---:|---:|---:|---:|
| A→B | `161.8274 → 139.8429 s` | `13.59%` | `[10.76%, 15.95%]` | `55/80` |
| A→E | `161.8274 → 120.7134 s` | `25.41%` | `[23.42%, 27.70%]` | `80/80` |
| E→F | `120.7134 → 115.8396 s` | `4.04%` | `[3.49%, 4.65%]` | `77/80` |
| A→F | `161.8274 → 115.8396 s` | `28.42%` | `[26.31%, 30.79%]` | `80/80` |

The full system reduces mean task E2E latency by 28.42%. Joint scheduling
provides the largest component, and visit speculation adds 4.04% under the
Joint policy. Thirty-nine of the forty registered gates are satisfied; the
E→F effect is 4.04% against the registered 5% threshold.

### Time breakdown

| Per-task component | E | F | E−F |
|---|---:|---:|---:|
| LLM | `42.9922 s` | `44.0700 s` | `-1.0778 s` |
| Exposed tool wait | `77.5882 s` | `71.6376 s` | `+5.9506 s` |
| E2E | `120.7134 s` | `115.8396 s` | `+4.8738 s` |

Speculation reduces exposed tool wait by 5.95 seconds per task while the LLM
component increases by 1.08 seconds, leaving a net E2E saving of 4.87 seconds.

Among F's 198 exact hits, 189 are queued promotions, one is an in-flight
promotion, and eight reuse completed results. Queue pre-positioning is the
dominant source of tool-side overlap in this workload.

### Native prefix

| Metric | Cache off | Cache on | Change |
|---|---:|---:|---:|
| Mean task E2E | `54.9726 s` | `24.3167 s` | `-55.77%` |
| Task P95 | `65.4460 s` | `25.6025 s` | `-60.88%` |
| Prefill time | `158.5928 s` | `86.5087 s` | `-45.45%` |
| Prefix hit ratio | `0%` | `64.53%` | `+64.53 pp` |

## Artifacts and Tests

The main records are:

- [final report](results/live_joint/PREFIX_AND_LIVE_CLOSED_LOOP_FINAL_REPORT.md)
- [native prefix protocol](results/live_joint/NATIVE_PREFIX_CAUSAL_DEV_PROTOCOL.md)
- [live tool–LLM protocol](results/live_joint/LIVE_TOOL_LLM_PROTOCOL.md)
- [formal-v9 protocol](results/live_joint/V9_FORMAL_MATRIX_PROTOCOL.md)

Run the core CPU suite with:

```bash
PYTHONPATH=reproduction \
  "$PASTE_PY" -m pytest -q \
  reproduction/tests/test_live_agent.py \
  reproduction/tests/test_live_broker.py \
  reproduction/tests/test_run_live_tool_llm_experiment.py \
  reproduction/tests/test_run_native_prefix_causal_dev.py \
  reproduction/tests/test_run_live_joint_formal_v9_matrix.py \
  reproduction/tests/test_aggregate_live_joint_four_cell.py \
  reproduction/tests/test_validate_live_joint_result.py
```
