# Consolidated reviewer response for common comments 2, 3, and 5

Status: evidence-backed draft. The fresh Comment 3 Tongyi target/high runs are
complete; the separately frozen Granite portability attempt failed closed in A
and therefore produced no cross-model A/E result.

This response consolidates, without extending, the verified conclusions in:

- [Comment 2 metric and load audit](../reviewer_comment2_load_sweep/REPORT.md)
- [Comment 3 co-scheduler and robustness audit](../scheduler_robustness/REPORT.md)
- [Comment 3 Granite fail-closed portability audit](../scheduler_cross_model_portability/FAILED_GRANITE_C5K_R1_AUDIT.md)
- [Comment 5 agent-baseline boundary audit](../agent_baseline_boundary/REPORT.md)

## 1. Metric correction and load degradation

### Reviewer-response text (English)

We agree that the earlier prediction percentages were underspecified. We now
report every prediction result with its target identity, numerator, denominator,
budget, and admission condition. The strings `27.8%` and `43.9%` do not occur as
prediction metrics in the audited repositories. The closest reproducible values
are Gemini safe-target tool-name Top-1, `8/28 = 28.6%`, and Qwen exact-URL Top-3,
`38/88 = 43.2%`; they use different targets and denominators and must not be
combined. Qwen has 34 held-out decision windows across 19 eligible sessions,
which expand to 88 atomic authoritative URL invocations, so decision-window hit
rate and exact-URL target recall are also reported separately.

The literal `93.8%` in the older Virtual-Lab source is a selection-event
prefetch-coverage claim at `N=4`, not an overall speculative-execution hit rate;
the same source reports `67.8%` useful prefetched URLs per prefetch at `N=4`.
The original 33-trace/321-selection analysis script and frozen result table are
not present, so the legacy `93.8%` claim is not independently regenerable from
the checked-in code. A separate SHA-bound Tongyi LOSO artifact happens to obtain
exact-URL Top-1 `15/16 = 93.8%`, but it is a different experiment and is not
provenance for the legacy coverage claim.

The corrected source metrics are:

| Scope | Top-1 | Top-3 | Top-5 | Interpretation boundary |
|---|---:|---:|---:|---|
| Qwen exact URL, held out | 17/88 (19.3%) | 38/88 (43.2%) | 49/88 (55.7%) | Exact atomic URL targets; 34 decision windows are a separate denominator |
| Virtual-Lab Tongyi exact URL, LOSO | 15/16 (93.8%) | 16/16 (100.0%) | 16/16 (100.0%) | Separate physically successful-fetch artifact, not legacy-prefetch provenance |
| Gemini safe-local tool name, held out | 8/28 (28.6%) | 24/28 (85.7%) | 27/28 (96.4%) | Name ranking only; candidates lack executable arguments and are not committed promotions |

We also added an explicitly low-predictability Qwen stress test. It applies a
global rank-first residual quota only after authoritative work. For budget
`k=5`, the deterministic trace envelope degrades as follows:

| Requested speculative work throttled | Admitted/requested | Realized exact-target coverage | Trace-derived stall reduction |
|---:|---:|---:|---:|
| 0% | 170/170 | 55.7% | 49.0% |
| 50% | 85/170 | 39.3% | 39.7% |
| 75% | 42/170 | 23.2% | 26.7% |
| 90% | 17/170 | 9.9% | 12.1% |
| 100% | 0/170 | 0.0% | 0.0% |

At saturation the mechanism falls back to authoritative execution: both extra
work and speculative coverage go to zero. The supported conclusion is graceful
degradation, not preservation of a `93.8%` number under arbitrary load. A
separate CPU-only run of the real bounded scheduler reaches the same qualitative
boundary: capacity rejection rises with concurrent opportunities, authoritative
misses bypass the speculative semaphore, and all nine budget/concurrency cells
have zero state-isolation violations. Its synthetic 5 ms service time validates
the harness, not paper or GPU latency.

## 2. Co-scheduler exact mapping, selection, and robustness limits

### Reviewer-response text (English)

We agree that the paper-level scheduler notation did not exactly identify the
deployed Qwen policy. The registered formal path is not the literal ratio
`ExposedToolGain / LLMPressure + Aging`, nor does it use a literal
`DecodeLoad + gamma * KVLoad` controller. With prefix locality disabled,
candidates are ordered lower-is-better by the following additive cost, preceded
by lexicographic final-call and exact-remaining-call lanes:

```text
C_i = P_llm,i - G_tool,i - G_progress,i - A_i

service_i = pt_i / prefill_rate + po_i / decode_rate
G_tool,i = beta * confidence_i * min(next_tool_wait_i, 80 s)
           / (1 + projected_KV_pressure_i * pt_i / context_ref)
A_i = aging_alpha * scheduler_wait_i
```

`P_llm` adds service time, nonlinear context/KV contention, task-tail cost,
over-budget cost, and a cold-session penalty. `G_progress` contains the final
call and reciprocal-progress bonuses. The audit independently decomposes these
terms and matches production `_joint_v2_score_s` with maximum absolute error
`1.421e-14 s` over the replayed candidate states.

The exact terminology mapping is:

| Paper/reviewer term | Registered Qwen implementation | Formal physical-KV status |
|---|---|---|
| ExposedToolGain | `nwc * min(nw, 80)` surrogate, confidence-weighted and pressure-damped | Active surrogate |
| LLMPressure | Additive service/context/task-tail/over-budget/cold-session cost | Active surrogate |
| DecodeLoad | Running-request band helper | Inactive; bypassed by formal physical-KV admission |
| KVLoad | Logical projected tokens for ranking; physical blocks and forecast footprint for admission | Active |
| Pressure band | Legacy `.82–1.02` controller | Inactive; formal target is physical utilization `.93` |
| gamma | No literal Joint-v2 gamma; context alpha `1.4` is only a non-equivalent analogue | No literal parameter |
| aging | `0.2 * wait_seconds` plus a 40 s physical rescue deadline | Active |
| tool speculation budget | 4 global workers, at most 2 speculative workers, visit cap 2, pending cap 128, TTL 120 s, minimum reservation 0 | Tool-side broker |

The prefill/decode rates (`38112` and `113.7` token/s) are calibration constants,
not universal model/GPU constants. The physical target `.93` is the registered
value; a single-factor replay changes physical admissions from
14 at `.85` to 18 at `.90`, `.93`, and `.97`. The legacy pressure-band sweep
leaves admissions unchanged at 18, confirming that it is inactive in this path.
The minimum speculative reservation was frozen at zero because the development
`min=1` cell improved only `0.2079%`, only 6/16 sources were faster, and its
bootstrap interval crossed zero. The maximum of two speculative workers and
pending cap 128 were not independently hardware-swept and are not claimed to be
universally optimal.

The 108-state CPU replay crosses three throughput/KV proxies, four workload
mixes, three context scales, and three physical-load ratios. Relative to the
registered A100-shape proxy, full-policy pairwise agreement is `0.935`, `1.000`,
and `0.997` for the small-slow, registered, and large-fast proxies; continuous
score agreement is `0.900`, `1.000`, and `0.998`. These are scheduler-decision
sensitivities, not E2E latency measurements, and the proxy names do not identify
measured GPU SKUs.

Checked-in Qwen/A100 results provide load sensitivity but not a single causal
load curve: older configurations range from regressions to gains. The current
physical-KV development screens at 240 and 300 offered sessions show mean-task
reductions of `25.385%` and `23.852%`, but each is a single screen and the source
reports disclose request-P95 regressions. The separate real-trace center A/E run
executes matching 264-request sets and changes task mean `91.1907 → 88.1054 s`
(`3.383%`) and task P95 `100.6761 → 99.2065 s` (`1.460%`); maximum physical
usage is only `.271`, so target `.93` is not binding. It is functional evidence,
not replicated target sensitivity or cross-model/cross-GPU validation.

### Fresh live target/shape results

The SHA-bound one-shot suites completed on Tongyi-DeepResearch-30B-A3B and four
A100-SXM4-40GB GPUs. Each cell completed 80/80 tasks, 240/240 LLM requests, 80
searches plus 80 visits, one status-200 transport attempt per tool invocation,
zero retry/429 events, 160 broker commits, and zero broker failures.

| Suite / shape | Physical target | A mean (s) | E mean (s) | Mean reduction | A P95 (s) | E P95 (s) | P95 reduction | Faster tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target / c10k-l80 | .85 | 209.225 | 148.418 | 29.063% | 300.221 | 248.202 | 17.327% | 78/80 |
| target / c10k-l80 | .93 | 209.225 | 180.549 | 13.706% | 300.221 | 256.475 | 14.571% | 69/80 |
| target / c10k-l80 | .97 | 209.225 | 175.970 | 15.894% | 300.221 | 252.363 | 15.941% | 74/80 |
| high / c12k-l80 | .93 | 249.026 | 218.009 | 12.455% | 313.284 | 273.431 | 12.721% | 78/80 |

All four observed E cells are directionally faster than their corresponding A
observation. This is single-run, fixed-order development evidence with no
confidence intervals and live external-HTTP timing drift; it neither identifies
`.85` as optimal nor isolates the target value as the cause. The high pair does
show that the positive direction persists at the separately registered
12k-context/80-task shape.

Physical-admission fields and safety equations recompute exactly, with zero
semantic required-field malformations and zero controller fail-closed events.
Concurrent stdout did concatenate the tail of 3/253, 1/313, 0/311, and 2/364
physical-marker lines in the `.85`, `.93`, `.97`, and high cells respectively.
The affected `rescue` tails are therefore not strict-parser-v2 clean; the report
does not call those raw logs malformed-free. The `.97` cell, with zero raw-line
interleavings, is strict-parser-v2 clean. The already completed `center093`
functional run remains a separate evidence tier and is not renamed as a target-
or shape-sensitivity cell.

### Cross-model portability boundary

A separately frozen, development-only
[Granite-3.3-8B-Instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)
profile tested the same 80-source call graph on the same A100 family at
c5k-l80. The lower
context shape was selected before live execution because Granite's native
tokenizer deterministically failed the registered c12k/16k context gate; it is
not comparable to, or pooled with, the Tongyi c10k/c12k cells. The offline gate
bound the exact 40-character model revision and 15-file snapshot, rendered all
80 three-phase prompts, compiled all 80 final grammars, and left at least 831
tokens of modeled headroom.

The one-shot live attempt then failed closed in baseline A: 74/80 tasks passed,
all 239 issued local LLM HTTP requests succeeded, and all 159 issued tool
invocations used one status-200 HTTP attempt with no retry, but the unchanged
output contract rejected six model generations. Four final answers ended as
unterminated JSON, one final answer violated the required ASCII-space tail, and
one second tool call was not a valid JSON object. That last failure explains the missing visit
and request relative to the 240/160 completion gates. E was never launched, the
attempt key is permanently consumed, and the fresh server stopped cleanly.
Consequently there is no Granite A/E latency result and no cross-model scheduler
generalization claim. The failure instead shows that offline grammar
compilability and prompt fit are insufficient to establish end-to-end model
contract portability.

The exact model/snapshot identities, failure classification, lifecycle checks,
and raw SHA-256 manifest are in the
[fail-closed portability audit](../scheduler_cross_model_portability/FAILED_GRANITE_C5K_R1_AUDIT.md).

## 3. Agent-baseline positioning

### Reviewer-response text (English)

We agree that ORION and SpecFaaS alone were insufficiently agent-specific. We
added Murakkab, llm-d, and NVIDIA Dynamo, and distinguish their scheduling units
rather than treating them as interchangeable PASTE implementations.

- Murakkab schedules logical/executable workflows and executors, and selects
  workflow configuration, model/tool implementation, hardware, and resource
  allocation. It supports request-agnostic DAGs, per-request dynamic
  composition, and known fan-out. The primary sources are the
  [OSDI '26 page](https://www.usenix.org/conference/osdi26/presentation/chaudhry)
  and [final paper](https://www.usenix.org/system/files/osdi26-chaudhry.pdf),
  especially Table 1 (p. 572), Sections 3.2–3.4 (pp. 573–574), Section 4.4
  (p. 577), and Section 4.6 (p. 578).
- llm-d's deployed contract routes an arrived `InferenceRequest` using request,
  KV, load, queue, LoRA, and SLO state. Its program-aware session graph,
  lifecycle, and proactive placement are documented as a direction, while agent
  control flow remains in the logic layer. See the official
  [Agentic Serving](https://llm-d.ai/docs/well-lit-paths/workloads/agentic-serving),
  [Router](https://llm-d.ai/docs/dev/architecture/core/router), and
  [Request Scheduler](https://llm-d.ai/docs/architecture/core/router/epp/scheduling)
  documentation.
- Dynamo explicitly provides agent-aware serving while the harness retains
  prompts, tools, subagents, and reasoning state. Its speculative prefill warms
  next-turn model KV; it does not execute or store an external-tool result. Its
  ThunderAgent program scheduler is experimental and unreleased, and performs
  program working-set accounting plus logical admission pause/resume at realized
  tool boundaries. See the official
  [agent overview](https://docs.nvidia.com/dynamo/dev/agents/overview),
  [agent hints](https://docs.nvidia.com/dynamo/agents/agent-hints),
  [agentic-inference description](https://docs.nvidia.com/dynamo/dev/digest/agentic-inference),
  [ThunderAgent scheduler](https://docs.nvidia.com/dynamo/dev/agents/thunder-agent-program-scheduler),
  and [agent trace replay](https://docs.nvidia.com/dynamo/dev/agents/agent-simulation).

PASTE addresses a different temporal edge: while the producing LLM is still
deciding, it predicts bounded concrete external-tool invocations from visible
state, executes them in an isolated lane, and promotes/reuses only an exact
authoritative match. In the held-out trace ordering, zero of 88 authoritative
`visit(URL)` invocation events are emitted before the producing decision LLM
completes. This is only an event-order count: no Murakkab, llm-d, or Dynamo code
was executed. The URL string itself is visible among earlier search candidates
for 70/88 invocations.

The separately executed PASTE broker admits 170 bounded candidates and obtains
49/88 exact hits, decomposed into 23 completed reuses and 26 in-flight
promotions. All 88 authoritative calls commit, ordinary misses execute normally,
and state-isolation violations are zero. The trace-derived exposed tool stall is
`38.514 → 19.647 s`; this is a broker/trace replay, not a throughput, latency, or
correctness benchmark of the three named systems. Under an ideal inference-only
2× sensitivity, `12.227 s` remains hideable, while the opportunity converges to
zero as decision time approaches zero.

The defensible relationship is complementarity. A PASTE-aware broker behind a
Murakkab tool executor, llm-d serving PASTE's model calls, or Dynamo serving its
LLM turns are proposed and unevaluated compositions. Dynamo speculative prefill
can overlap the tail of the opposite tool-to-LLM edge when the harness knows a
tool is about to return; it does not replace PASTE's external-result isolation
and exact-match promotion.

## 4. Manuscript changes

### Reviewer-response text (English)

We will make the following concrete revisions:

1. Replace standalone prediction percentages with a metric table containing
   target type, numerator, denominator, budget, admission condition, and whether
   the value is decision-window coverage, atomic-target recall, or selected-
   prediction precision. Remove or explicitly relabel the non-regenerable legacy
   `93.8%` coverage statement.
2. Add the Qwen low-predictability load × budget table and describe saturation
   as graceful fallback to authoritative execution. Keep analytical admission,
   CPU scheduler stress, and GPU/live evidence in separate evidence tiers.
3. Replace the abstract scheduler equations with the exact additive Qwen policy,
   its lexicographic lanes, and the separate physical-KV admission rule. Add the
   reviewer-term mapping, calibration/selection evidence, and the explicitly
   inactive or non-literal terms.
4. Add the validated live target/high table with its fixed-order, single-run,
   external-HTTP, and raw-log parser qualifications. Do not infer unexecuted
   cells, pool the excluded partial shape run, or claim an optimal target.
5. Add the consumed Granite portability attempt as a fail-closed compatibility
   result, not a latency row: disclose the 74/80 contract pass count, six model-
   output failures, absent E cell, and no-rerun rule.
6. Add an agent-baseline abstraction table for Murakkab, llm-d, Dynamo, and
   PASTE; retain the official primary-source URLs and mark all proposed
   integrations as unevaluated.
7. Add the held-out event-order and broker replay with its explicit non-vendor-
   benchmark label, plus the idealized 1×–2× inference sensitivity.

## 5. Claim boundary

### Reviewer-response text (English)

The consolidated evidence supports corrected metric definitions, graceful
degradation of speculative coverage as residual capacity disappears, an exact
mapping from the deployed Qwen scheduler to its implementation, bounded
parameter-sensitivity evidence, and a documented abstraction boundary between
PASTE and current agent-serving systems. It does not establish:

- that the legacy `93.8%` coverage number is an overall hit rate or remains
  valid under arbitrary load;
- E2E GPU latency from the analytical admission envelope, synthetic CPU
  scheduler stress, or 108-state policy replay;
- universal optimality of the Qwen calibration rates, aging coefficient,
  physical target, speculative-worker cap, or pending cap;
- cross-model scheduler generalization; the Granite attempt failed the
  unchanged model-output contract in A, so E was not executed;
- cross-GPU generalization; every completed live scheduler cell uses the same
  A100-SXM4-40GB family;
- a realized-completed-tool-gain feedback side channel in the current Joint-v2
  score;
- vendor performance for Murakkab, llm-d, or Dynamo from the `0/88` event-order
  audit or the PASTE broker replay; or
- an implemented Murakkab/llm-d/Dynamo integration with PASTE.

Percentages from different reports or evidence tiers are not additive. Every
submitted table should retain its experiment type—source-metric replay,
analytical envelope, synthetic scheduler run, checked-in A100 evidence, fresh
live result, or semantic agent-boundary replay—next to the number.

## 中文行动建议

1. **先修指标口径。** 在 rebuttal 和正文中删除孤立百分比；逐项写明目标、
   分子、分母、Top-k、是否经过负载准入。Qwen 的 34 个 decision windows 与
   88 个 atomic URLs 必须分列；Gemini 的 name ranking 不得写成 exact-argument
   promotion。
2. **处理旧 `93.8%`。** 最稳妥做法是删除；若保留，必须标为旧版 N=4
   selection-event coverage，并同时披露无法从当前仓库独立重算。不得用
   Tongyi LOSO 的 15/16 为旧 claim 补 provenance。
3. **正文公式对齐代码。** 用 additive cost、lexicographic lanes、physical-KV
   admission 三部分重写；明确 DecodeLoad band 和旧 pressure band 在 formal
   路径中不生效，`gamma` 不是 literal parameter。
4. **写入已完成的 Comment 3 live 结果。** target 三个 E cells 与 high A/E
   均已通过 completeness、transport、broker、SHA/provenance 和独立聚合检查；
   正文必须同时保留 fixed-order、single-run、无 CI、外部 HTTP 漂移与 raw
   telemetry 尾字段拼接限制，不得把 `.85` 写成最优值，也不得复用失败的
   shape-r1 前四格。
5. **严格分层呈现 robustness。** CPU proxy 只支持决策敏感性；历史 A100
   只支持同一 Qwen/A100 family 的开发期负载敏感性；center093 只支持单点
   functional direction，且 `.93` 未绑定；fresh target/high 才是本轮正式
   live evidence，但仍不是 cross-GPU proof。
6. **如实报告 cross-model fail-closed。** Granite c5k 的 A 仅 74/80 tasks
   通过未修改的 structured-output contract；E 未启动、one-shot 已消耗，不得
   用 74 个成功 task 估算或补造 A/E latency，也不得据此声称 scheduler 已跨
   模型泛化。该结果只能说明 chat/context/grammar 离线预检不足以保证真实
   generation contract portability。
7. **补 agent baseline，但避免贬低。** 明确 Dynamo 已 agent-aware、Murakkab
   支持 dynamic workflows、llm-d 有 program-aware north-star；差异写成
   documented scheduling-unit boundary，而不是“这些系统不能扩展”。
8. **固定非 benchmark 声明。** `0/88` 只表示权威调用事件在 trace 中尚未
   发出，并非 URL 字符串不可见，也未执行 vendor scheduler。PASTE 的
   49/88 与 stall 数字仅属于独立 broker/trace replay。
9. **提交前做交叉检查。** 保留各子报告的相对链接和官方 URL；逐表标注
   evidence tier；禁止跨实验相加百分比，并确认正文、rebuttal、附录中的
   公式、默认值和限制完全一致。
