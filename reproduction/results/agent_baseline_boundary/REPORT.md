# Agent-oriented baseline boundary: Murakkab, llm-d, and NVIDIA Dynamo

Date of source audit: 2026-08-30

## Reviewer-facing conclusion

The reviewer is right that ORION and SpecFaaS alone do not establish the
agent-system boundary. Murakkab, llm-d, and Dynamo are stronger and more
relevant comparisons. They should not, however, be described as interchangeable
PASTE implementations:

- Murakkab optimizes workflow configuration, executor assignment, model/tool
  selection, hardware/resource provisioning, and multi-workflow SLO efficiency.
- llm-d optimizes the serving of model requests and model state, with an
  explicit direction toward program-aware scheduling and KV lifecycle control.
- Dynamo's official documentation exposes agent-aware request metadata,
  KV-aware routing, speculative prefill, and tracing/replay, and describes an
  experimental, unreleased program scheduler at tool boundaries.
- PASTE addresses an earlier edge in the loop: while an LLM is still deciding
  the next external-tool call, it predicts bounded concrete invocations from
  already-visible state, executes them in an isolated tool lane, and only
  promotes/reuses an exact authoritative match.

The fair claim is therefore **complementarity plus a documented
abstraction-boundary distinction**, not that PASTE replaces these systems or
beats their published throughput.

## What each system actually schedules

| System | Primary abstraction and scheduling granularity | Agent/tool awareness | Reuse or speculative mechanism | Not documented by the cited mechanism |
|---|---|---|---|---|
| Murakkab (OSDI '26) | Logical/executable workflow, executor, workflow–SLO configuration, model/tool choice, hardware and resource allocation; per-request dispatch after composition | Workflow tasks map to executors that include LLMs, structured compositions, and tools; supports declared DAGs and per-request dynamic composition | Known workflow fan-out, profile-guided reconfiguration, executor colocation/multiplexing, autoscaling | The paper's documented mechanisms do not include predicting a future exact external-tool invocation before its producing LLM returns, isolating candidate results, or exact-match promotion |
| llm-d v0.9 | An arrived `InferenceRequest`; Filter → Score → Pick selects a model endpoint from request, KV, load, queue, LoRA, and SLO state | The deployable stack targets agent traffic; the program-aware session graph is documented as the direction, while agent control flow and cognitive decisions remain in the logic layer | Deployed recipe: prefix/KV-aware routing, tiered KV offload, precise cache state, P/D disaggregation. North-star directions: program-level state lifecycle and proactive placement | The compared official contract leaves agent/tool control in the logic layer and does not document an external-tool executor or future exact-tool-request generator |
| NVIDIA Dynamo | An arrived model request with session/agent hints; KV-aware worker routing; experimental ThunderAgent groups the whole `LLM turn → tool call → next turn` program and pauses/resumes at tool boundaries | Yes. Official docs explicitly call it agent-aware while leaving prompts, tools, subagents, and reasoning state in the harness | KV reuse/routing, priority, expected output length, speculative **prefill** of a predicted next-turn prefix; experimental tool-boundary program pause/resume | Speculative prefill warms model KV, not an external-tool result. ThunderAgent performs program working-set accounting and admission pause/resume at realized tool boundaries; the cited mechanisms do not describe predicting, executing, isolating, and promoting the future tool call |
| PASTE | A candidate concrete external-tool invocation plus a joint LLM/tool readiness signal | Dynamic model-emitted tool batches, session/generation identity, exact arguments, tool safety/freshness policy | Bounded tool execution before the authoritative call exists; exact session/name/arguments promotion or completed reuse; misses fall back normally | It does not replace model serving, fleet-level KV management, workflow configuration, GPU placement, or autoscaling |

This reading is grounded in primary sources:

- The [Murakkab OSDI '26 page](https://www.usenix.org/conference/osdi26/presentation/chaudhry)
  describes the declarative abstraction, profile-guided optimizer, and adaptive
  runtime. The [final paper](https://www.usenix.org/system/files/osdi26-chaudhry.pdf)
  says the logical DAG is request-agnostic (Section 3.2, p. 573), but it also
  supports per-request dynamic composition (Section 3.4, p. 574) and evaluates a
  request-specific dynamic coding pipeline (Section 4.4, p. 577). Table 1
  (p. 572) and Sections 3.3–3.4 place model/tool/hardware decisions at
  optimization epochs and dispatch at request time; Section 4.6 (p. 578)
  executes two known DAG fan-out sub-tasks in parallel. Thus, “Murakkab only
  supports static DAGs” would be false.
- The [llm-d agentic-serving guide](https://llm-d.ai/docs/well-lit-paths/workloads/agentic-serving)
  distinguishes today's request-centric default from its program-aware
  direction (sections “Deploy” and “Direction”). It lists the deployable
  baseline (prefix/load routing, tiered KV, precise prefix state, and P/D
  disaggregation) and explicitly leaves agent control flow and cognitive
  decisions in the logic layer. The
  [Router](https://llm-d.ai/docs/dev/architecture/core/router) acts when an
  inference request arrives (“How it Works”), and the
  [request scheduler](https://llm-d.ai/docs/architecture/core/router/epp/scheduling)
  selects an endpoint for that request (“Architecture Overview”) and exposes
  plugin interfaces (“Extension Points”).
- The [Dynamo agent overview](https://docs.nvidia.com/dynamo/dev/agents/overview)
  says in “Agents” that Dynamo is agent-aware without owning the agent loop:
  the harness still manages tools. Its
  [agent-hints contract](https://docs.nvidia.com/dynamo/agents/agent-hints)
  defines in “Agent Hints” and “Request Flow” `speculative_prefill` as warming
  the predicted next-turn prefix after a turn completes. NVIDIA's
  [agentic-inference description](https://docs.nvidia.com/dynamo/dev/digest/agentic-inference)
  gives the temporal direction more precisely: a harness can request prefill
  when it knows a tool is about to return. The
  [ThunderAgent program scheduler](https://docs.nvidia.com/dynamo/dev/agents/thunder-agent-program-scheduler)
  is explicitly experimental/unreleased and documents under “The Scheduler”
  and “Tool-Boundary Pause/Resume” logical pause/resume at tool boundaries,
  without decode preemption. Finally, Dynamo's own
  [agent trace replay](https://docs.nvidia.com/dynamo/dev/agents/agent-simulation)
  says under “What This Captures” that tools are not executed again and, by
  default, tool arguments are not stored. It is a model-serving workload replay,
  not a speculative-tool replay. NVIDIA calls that serving replay a reusable
  benchmark; the `0 / 88` event-order count below was not produced by or run
  through it.

No claim here depends on an absence-of-keyword search. The boundary follows
from each documented input/output contract and scheduling unit.

## The boundary made concrete

The direct transition present in the Qwen traces is:

```text
search result becomes visible
          |
          +--> PASTE ranks concrete visit(URL) candidates and starts bounded,
          |    isolated external work
          v
   decision LLM is running --------> exact authoritative visit(URL) is emitted
                                             |
                                             +--> the workflow can mark the node
                                                  ready and the application can
                                                  launch ordinary demand execution
                                             |
                                             +--> after the tool result, llm-d or
                                                  Dynamo can optimize the next
                                                  model request/KV state
```

A dependency-respecting workflow scheduler may parallelize *ready* nodes,
and an ideal inference substrate can make the decision LLM arbitrarily fast.
Neither operation emits the missing authoritative invocation decision before
the decision LLM does. Adding a predictor alone is also insufficient: wrong
results must be isolated, bounded, expired/cancelled, and reconciled by exact
identity.

## Executed minimal experiment

### Command

From the Qwen repository root:

```bash
python -m unittest reproduction.tests.test_baseline_boundary -v

python reproduction/scripts/run_agent_baseline_boundary_replay.py \
  --top-k 5 \
  --max-concurrency 4 \
  --inference-speedups 1 1.25 1.5 2 \
  --output reproduction/results/agent_baseline_boundary/replay.json
```

The machine-readable result is [replay.json](replay.json). It records the source
commit, runner/module SHA-256 values, model-artifact checksum, and SHA-256 for
every train/held-out trace. The run performs two operations:

1. It **executes** the existing PASTE broker over the deterministic held-out
   split with a zero-delay local tool executor. This validates admission,
   isolation, exact promotion/reuse, miss fallback, expiry, and reconciliation;
   it does not claim network latency.
2. It replays the observed decision-window and tool-stall durations while
   ideally dividing only LLM time by 1×–2×. This is deliberately favorable to a
   model-serving substrate: no routing/cache/queue/quality penalty is charged.
   It is a sensitivity analysis, not an implementation benchmark of any named
   baseline.

### Held-out result

| Quantity | Result |
|---|---:|
| Total trace sessions; deterministic train/held-out split | 100; 70 / 30 |
| Held-out direct `search → decision LLM → visit` examples | 34 (from 19 sessions) |
| Held-out authoritative URL invocations | 88 |
| Trace event-order count (no vendor code executed): authoritative `visit(URL)` invocation events emitted before decision completion | **0 / 88** |
| Authoritative URLs already present among visible search candidates | 70 / 88 (79.55%) |
| Bounded top-5 candidate submissions | 170 |
| Exact invocation hits | 49 / 88 (55.68%) |
| Hit lifecycle | 23 completed reuse + 26 in-flight promotion |
| Ordinary authoritative misses | 39 |
| Expired unused candidates | 121 |
| Authoritative commits | 88 / 88 |
| State-isolation violations | **0** |
| Trace-derived exposed tool stall, demand-only → PASTE | 38.514 → 19.647 s |
| Trace-derived hidden stall | 18.866 s (48.99% of exposed tool stall) |

The `0 / 88` is solely a trace event-order/eligibility count; no Murakkab,
llm-d, or Dynamo code was run to obtain it. In each selected transition, the
authoritative `visit(URL)` invocation event follows its producing decision-LLM
event. The URL string itself was already visible among prior search candidates
for 70/88 invocations; those strings are non-authoritative candidates rather
than emitted next-call decisions. The `0 / 88` count is **not** a throughput,
latency, or correctness measurement of any named system. The separately
executed PASTE broker result shows the additional predict/execute/reconcile
mechanism: its 49 exact hits reconcile exactly with 23 completed reuses plus 26
in-flight promotions; all 88 authoritative calls commit, misses execute
normally, and no speculative result leaks into authoritative state.

### Inference-substrate sensitivity

Only the decision-LLM duration is ideally accelerated; observed external-tool
times are held fixed. “Segment” means only the summed decision-LLM plus exposed
`visit` stall for the 34 transitions.

| Ideal decision-LLM speedup | Demand-only segment | PASTE segment | Tool stall hidden by PASTE |
|---:|---:|---:|---:|
| 1.00× | 146.623 s | 127.757 s | 18.866 s |
| 1.25× | 125.001 s | 108.539 s | 16.462 s |
| 1.50× | 110.587 s | 95.782 s | 14.805 s |
| 2.00× | 92.568 s | 80.342 s | 12.227 s |

The demand-only external-tool component remains 38.514 s at every point by
construction. PASTE's opportunity decreases as faster inference shortens the
overlap window—an important limitation rather than a hidden assumption—but a 2×
ideal model substrate still leaves 12.227 s of tool stall hideable by the
top-5 policy. In the zero-decision-time limit, this form of PASTE also has no
window and both policies converge to the same 38.514 s tool-only lower bound.
Thus the result is not “PASTE always adds a fixed speedup”; it is “inference
acceleration alone does not execute the external tool, and the two mechanisms
compose when a nonzero decision window exists.”

## Audit of the three PASTE repositories

The audit was read-only before this report; unrelated working-tree changes were
left untouched. Commit IDs below identify the inspected snapshots.

| Repository / inspected HEAD | Concrete temporal edge and safety boundary | Executable entry points |
|---|---|---|
| `PASTE-Qwen-DR` / `83e018557566c78e5d499dae5bfd1a877b66eef2` | `reproduction/paste_repro/live_agent.py:1411–1423` submits concrete `visit` predictions before the decision LLM at 1433–1454; the exact model-selected invocation is issued authoritatively at 1485–1489. `live_broker.py:115–131` gives both lanes shared bounded tool capacity, and its authoritative path performs exact promotion. `traces.py:330–367` extracts exactly this event order and timing window. | `reproduction/scripts/run_speculative_tool_execution.sh`; `reproduction/scripts/run_online_speculative_execution.py`; `reproduction/scripts/run_agent_baseline_boundary_replay.py`; full live A/B/E/F commands in `reproduction/README.md` |
| `virtual-lab-PASTE` / `15f3bc3227892cf0d5d96c3b9e5ed1d63ca74a8f` | `src/virtual_lab/run_meeting.py:1830–1851` obtains and resolves a model-emitted batch; 1865–1880 executes exact authoritative invocations. After visible search output, 1950–2023 causally predicts and submits bounded `web_fetch` work before the next LLM decision. `speculative_broker.py:256–263,386–476` isolates by session+canonical invocation and promotes queued/running/completed exact matches. | `reproduction/qwen_tool_only_phase1.py`; `reproduction/qwen_joint_tool_phase2.py`; `reproduction/scripts/run_trace_abef.sh`; `reproduction/scripts/run_live_abef.sh`. `src/virtual_lab/tool_only_replay.py` keeps `learned_rank` causal evidence separate from the explicitly noncausal `trace_exact` mechanism upper bound. |
| `gemini-cli-PASTE` / `0cbb8bb05910f3fa3d0d0cb29630af47c871b98d` | `packages/core/src/core/speculativeToolRuntime.ts:2503–2533,2755–2805` learns from a completed authoritative batch and starts next-call predictions; 2811–2865 resolves them when the next model batch is known; 3224 onward permits reuse only for exact session/name/arguments and otherwise falls back. Lines 41–50 only permit completed local reuse for `read_file`, whose freshness can be proven; other reads need a snapshot. `coreToolScheduler.ts:1105,1270` is the authoritative integration point. | `reproduction/scripts/reproduce_trace_artifacts.py`; `run_causal_replay.py`; `run_node_central_abef.py`; `run_abef_benchmark.py` |

The three implementations deliberately have different safe speculative
surfaces—remote read-only visits/fetches, scientific web fetch, and
freshness-proven local file reads/opt-in anonymous HTTP preparation. This is
evidence that PASTE is not merely “mark any future tool as high priority”: the
store-buffer/freshness policy is part of its required semantics.

## What composes, and what does not substitute

### Murakkab + PASTE

Murakkab's paper documents executor assignment, model/tool selection,
CPU/GPU-resource provisioning, executor colocation/model multiplexing, and
workflow/SLO configuration. On that interface, we **propose—without
implementing or evaluating it here**—placing a PASTE-aware broker behind a tool
executor. Murakkab's dynamic workflow support avoids any requirement that the
whole agent graph be static, but its documented dispatch of a selected workflow
does not itself describe PASTE's prediction and commit protocol before a
dynamic executor invocation becomes exact.

### llm-d + PASTE

As a **proposed, unevaluated composition**, llm-d could serve PASTE's model
calls, retain/offload/reuse their KV, disaggregate prefill/decode, and route
using request/session/cache state. A future integration could translate
tool-readiness or critical-path metadata into a Router extension; the cited
scheduler documentation establishes Filter/Scorer/Picker extension points, not
a PASTE metadata integration. Reproducing PASTE's effect would still require an
application-side component to predict and safely execute the external tool,
because the documented llm-d contract leaves agent control flow in that layer.
KV “zero recompute” and tool-result exact reuse are different state objects and
require different validity rules.

### Dynamo + PASTE

The temporal directions are complementary:

- PASTE overlaps **decision LLM → predicted next external tool**.
- When the harness knows a tool is about to return, Dynamo speculative prefill
  can overlap **the tail of the current external tool → next-LLM prefill/KV
  readiness**.
- ThunderAgent regulates **whole-program admission and working-set pressure at
  realized tool boundaries**.

A plausible but **unevaluated** integration would use Dynamo for LLM turns and
PASTE for the bounded external-tool broker. Session identity and priority hints
could be mapped across the two, but the integration would need separate
accounting for GPU/KV capacity and external-tool API/rate-limit capacity.
Dynamo tracing or `speculative_prefill=true` does not, by the cited contracts,
provide PASTE's exact external-result isolation/promotion.

### Existing composability evidence, with limits

The checked-in Qwen live matrix is not a Murakkab/llm-d/Dynamo benchmark, but it
does show that tool speculation remains incremental under a stronger joint
scheduling policy. On 80 paired live sources, Joint+demand-only (E) to
Joint+visit-speculation (F) changes mean E2E `120.7134 → 115.8396 s` (4.04%) and
exposed tool wait `77.5882 → 71.6376 s` (5.9506 s/task), while F's LLM time is
actually 1.0778 s higher. The result is statistically stable but misses its
preregistered 5% promotion threshold (39/40 gates pass), so it must be described
as positive diagnostic/composability evidence, not a formally promoted claim.
The same repository separately shows native prefix-cache benefits, but those
percentages are not additive with the A/B/E/F matrix.

## Evidence boundary and proposed response

This artifact did **not** deploy Murakkab, llm-d, or Dynamo, and it makes no
throughput, cost, or superiority claim about them. We did not locate an official
runnable Murakkab artifact from the USENIX paper page. A stock llm-d/Dynamo
deployment comparison would require a controlled accelerator/model-serving
topology (and Kubernetes for the cited llm-d recipe) and would primarily measure
model-serving/KV behavior. The cited contracts do not document the disputed
external-tool path; exercising that path would require adding an
application-side predictor/broker such as PASTE. The executed experiment
therefore tests the minimal semantic difference with the baseline-favorable
assumptions stated above. The documentation is living; the versions and access
date at the top of this report should be retained.

A concise reviewer response can read:

> We agree that ORION and SpecFaaS were insufficiently agent-specific. We added
> Murakkab, llm-d, and NVIDIA Dynamo and now distinguish their scheduling units.
> Murakkab optimizes realized/declarative workflow executors, configurations,
> hardware, and SLO/resource allocation; llm-d optimizes arrived inference
> requests and KV state while leaving agent control in the logic layer; Dynamo
> is already agent-aware and adds session hints and speculative prefill, while
> documenting an experimental, unreleased tool-boundary program scheduler. These
> capabilities are complementary to PASTE, but none of their documented
> mechanisms predicts, isolates, executes, and exact-match-promotes a
> not-yet-emitted external-tool invocation. We added a conservative held-out
> boundary replay. A trace
> event-order audit—without executing vendor code—found that 0/88 authoritative
> `visit(URL)` invocation events were emitted before the producing LLM completed,
> whereas the separately executed PASTE broker admitted 170 bounded candidates,
> obtained 49/88 exact hits, had zero state-isolation violations, and reduced
> trace-derived exposed tool stall from 38.514 to 19.647 s. This is explicitly a
> semantic/trace replay rather than a throughput benchmark of the three systems.
> An ideal 2× inference-only sensitivity still leaves 12.227 s hideable by PASTE.
> We also clarify a proposed, unevaluated composition path: Murakkab could
> configure/provision a PASTE-aware tool executor, llm-d or Dynamo could serve
> its LLM calls, and Dynamo speculative prefill can overlap the tail of the
> opposite tool→LLM edge.

For the manuscript, the defensible revision is to add the abstraction table,
the executed boundary replay (with its non-benchmark label), the 1×–2×
sensitivity, and the composition paragraph. It should avoid “these systems
cannot support dynamic agents,” “Dynamo is not agent-aware,” and any claim that
`0 / 88` is measured baseline performance.
