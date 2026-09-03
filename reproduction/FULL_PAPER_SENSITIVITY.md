# Conservative FULL and paper-aligned sensitivity

> Paper-facing knob definitions and the final presentation table are frozen in
> `reproduction/results/full_paper_sensitivity_quick/PAPER_PRESENTATION.md`.
> In that notation, `ExposedToolGain=p_hit*predicted_tool_time`,
> `LLMPressure=predicted_LLM_time*KVUsage`, `Aging` is proportional to waiting
> time, and `EnginePressure=DecodeLoad+gamma*KVLoad`. The lower-level mappings
> below are retained only as implementation provenance.

## Decision

The sensitivity experiment measures only one complete system, `FULL`, while
varying one paper-level parameter at a time. It is not an ablation matrix and
does not use the recently added client-side `GainPressureAdmissionController`.
Ready LLM turns go directly to vLLM's waiting queue, reproducing the old Joint
serving setup in which scheduler queue pressure is present and observable.

`FULL-center` contains only mechanisms with positive supporting evidence:

1. Native vLLM prefix caching is enabled. The controlled three-stage prefix
   test reduced mean flow from 54.9726 s to 24.3167 s. Explicit prefix-affinity
   reordering remains disabled because it did not improve prefix hit rate or
   LLM latency and its apparent end-to-end gain was not causally supported.
2. Joint-v2 stage-aware waiting-queue ordering is enabled. The registered
   formal-v9 result reduced mean task flow from 161.8274 s to 120.7134 s
   (25.4061%).
3. Forecast-aware physical-KV admission is enabled at target 0.93 with a
   40-second rescue. Development screens showed an additional 10.426% and
   7.204% improvement at stress 240 and 300 and removed native preemptions.
4. The new all-Visit predictor/executor is enabled with the conservative
   fixed-half shared-pool policy (`capacity=16`, speculative cap `8`). The
   resource-tight replay reports 12.28% end-to-end speedup, 13.29% mean-flow
   speedup, and 2.812x call amplification for this point. The executor's latest
   live run reduced exposed Visit time from 24.84 s to 11.18 s per task, while
   the external Python admission gate erased the system-level benefit; that
   gate is therefore excluded.

The component evidence above justifies the conservative composition. It does
not substitute for measuring the newly composed end-to-end FULL center; that
measurement is the first cell in the new matrix.

## Paper abstraction to implementation

The paper equations describe abstract control quantities. A quantity can map
to several concrete signals as long as the mapping is frozen before the run
and every sensitivity point changes the complete bundle consistently.

| Paper quantity | Concrete FULL implementation |
|---|---|
| `ExposedToolGain` | next-tool gain (`TOOL_BETA`), remaining-tool value, progress bonus, final-turn bonus, and the always-on causal final/remaining-call lanes |
| `LLMPressure` | predicted prefill/decode service, context-pressure cost, logical projected KV, tail cost, over-budget cost, and cold-session cost |
| `Aging` | continuous wait credit plus the physical-KV rescue deadline |
| `DecodeLoad` | native running-request/batch state and native `max_num_seqs`; normal vLLM token scheduling is retained inside the admitted set |
| `KVLoad` | logical projected KV in ranking plus physical blocks, running growth, and request footprint forecasts in admission |
| `gamma` | relative weight of context/logical-KV pressure in the continuous `LLMPressure` score |
| `P_low` | invariant work-conserving progress rule: do not leave an empty engine idle when a physically feasible request exists |
| `P_high` | forecast-aware physical-KV utilization ceiling, centered at 0.93 |

This is a behavioral correspondence, not a claim that the paper equation is
written as one literal source-code expression.

## Metrics

The paper-facing sensitivity figure should remain small. Its x-axis is the
paper knob and its primary y-axis is task completion time for FULL:

| Level | Metric | Exact definition | What it establishes |
|---|---|---|---|
| primary | mean task flow time (s) | mean release-to-final-completion time over all sessions, including the client task gate, vLLM waiting/running time, tools, and speculation | average end-to-end sensitivity |
| primary | task-flow p95 (s) | empirical 95th percentile of the same per-session flow times | tail sensitivity/fairness |
| normalized | change vs `FULL-center` (%) | `(variant - center) / center`; negative is faster | shape of each knob curve without turning the experiment into an ablation |
| guard | completed sessions / failures | exact completed count and zero-failure requirement | comparable, valid cells |

The following are diagnostic metrics. They explain *why* a paper knob changes
task time and can be reported in the text or appendix, but they are not extra
sensitivity axes:

| Paper mechanism | Diagnostic metrics |
|---|---|
| `ExposedToolGain` | exposed Visit seconds/task, saved Visit seconds/task, realized Visit hit rate, speculative call amplification |
| `LLMPressure` / `DecodeLoad` | mean and p95 LLM-turn latency, native vLLM waiting time, running-request count, queue depth, batched-token pressure |
| `KVLoad` / `gamma` / `P_high` | physical-KV usage, predicted admitted tokens, effective admission cap, budget-truncated ticks, preemptions |
| `Aging` | oldest/p95 scheduler wait, continuous waiting credit, rescue count, task-flow p95/max |
| native prefix cache guard | prefix-cache queries, hits and hit ratio; it is held on and is not itself swept |

The headline result table contains only the paper knob, its value, mean task
flow, p95 task flow, and change from FULL center. Internal environment-variable
names stay in the machine-readable contract for reproducibility, not in the
paper figure.

## FULL-only sensitivity matrix

The center is the registered Joint setting. Every off-center point keeps all
FULL mechanisms enabled and changes one abstract paper quantity only.

### Exact center values

The paper-level value `1x` is a normalization around the following concrete
registered configuration; it does not mean that every implementation
coefficient equals one.

| Paper quantity | Paper-level center | Concrete center values |
|---|---:|---|
| `ExposedToolGain` | `beta_G=0.9` | paper term `beta_G * p_hit * predicted_tool_time`; exact final/remaining-call lanes remain enabled in FULL |
| `LLMPressure` | `1x` fixed composite | prefill rate `38,112 token/s`; decode rate `113.7 token/s`; context reference `8,000 tokens`; context coefficient `1.4`; task-tail coefficient `0.25`; new-session penalty `4 s`; over-budget penalty `120 s` |
| `Aging` | `alpha_A=0.2` | paper term `alpha_A * waiting_time`; physical-KV rescue deadline remains an implementation fairness guard |
| `gamma` | `1.0` | paper term `DecodeLoad + gamma * KVLoad` |
| `P_low` | work-conserving, operational floor `48/96 = 0.50` decode load | `gate_min_running=48`, `deadline_min_running=48`, native `max_num_seqs=96`; an empty engine always admits one physically feasible request |
| `P_high` | `0.93` | physical-KV forecast budget is `93%` of profiled KV capacity; the remaining `7%` is headroom and may be used only by the aging rescue without crossing physical 100% |

For this 80-session quick workload, `foreground_max_sessions=96`, so the cold-
session-cap part of the `P_low` bundle is normally non-binding. The 48-request
value still defines the deadline/eligibility floor, while empty-engine progress
is the hard work-conserving lower-bound behavior.

Other fixed FULL-center values are: native prefix caching on, explicit prefix
locality off, `max_active_tasks=80`, all-Visit policy `budget_w5_cap10`, Visit
pool capacity `16`, and speculative cap `8`.

| Axis shown in paper | Values | Bundled implementation change | Expected consequence |
|---|---|---|---|
| `ExposedToolGain` coefficient `beta_G` | 0.45, 0.9, 1.8 | multiply predicted `p_hit * tool_time` | larger values expose/consume tool progress earlier; excessive values can delay high-cost turns |
| `Aging` coefficient `alpha_A` | 0.1, 0.2, 0.4 | multiply request waiting time | larger values improve fairness sooner but weaken gain-efficient ordering |
| `gamma` | 0.5, 1, 2 | multiply normalized physical `KVLoad` in EnginePressure | larger values protect against KV pressure but can underfill the running batch |
| `P_low/P_high` | work-conserving/0.85, /0.93, /0.97 | keep lower progress rule fixed and sweep physical-KV ceiling | lower ceilings preserve headroom but may underfill; higher ceilings permit larger batches but increase overload risk |

The runner defaults to three repetitions, fresh vLLM server per cell, 80 active
tasks, native prefix caching on, and explicit prefix locality off. Task flow is
measured from workload release, including time in the client concurrency gate.
Cell order is independently and deterministically shuffled within each
repetition to reduce wall-clock drift bias; the seed and all orders are written
to `run_plan.json` before any server starts.

## Entry point

```bash
python reproduction/scripts/run_trace_all_visit_coscheduling_matrix.py \
  full-paper-sensitivity-r1 --suite sensitivity --repetitions 3
```

Use `--check-only` to emit the frozen cell-to-implementation mapping without
launching servers.
