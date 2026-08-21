# Queued stress180 scheduler search and verification report

Date: 2026-08-15

> **Status: completed three-replicate load-sensitivity verification.** The
> frozen, fresh-server A/D aggregate gives a 31.419% mean task-flow reduction.
> This is a tuned queued stress result, not an untouched final evaluation or a
> reproduction of the paper's complete system.

## Why the historical “near 30%” result is not the current baseline

The repository did previously record effects around 30%, but none was the same
strict causal, fixed-workload, full-drain comparison as the current A/D
protocol. The detailed provenance audit is retained in
[`docs/legacy.md`](../../../docs/legacy.md#历史近-30-为何不可与当前-strict-causal-full-drain-比较).

| Historical number | Actual measurement scope | Why it is not directly comparable |
|---|---|---|
| OAS-Aging `27–32%` | C112 runs stopped by a 900 s timebox; the reported means used only completed fair/intersection subsets, such as 29/36 tasks for the `32.1%` result. | Incomplete tasks were absent from mean E2E. Later healthy-C48 checks were about 18–19%, and the larger C112 effect was documented as an overload/drop-policy signal. |
| Joint Pacer V2 `35.5%` | N=350, 900 s timebox, oracle overlap and future trajectory fields; the comparison contained only the 47 tasks completed by both `oracle+fcfs` and V2. | The two policies completed 47/350 and 180/350 tasks respectively. This is a survivorship-selected common set and a non-causal oracle upper-bound setting. |
| Full-completion `oracle_critical` `24.54%` | All 128 tasks drained; mean E2E fell from 1871.064 s to 1411.878 s. | Completion-set bias was removed, but the key used final-call, future tool-wait, remaining-call and total-trajectory metadata. |

Thus the archived results establish that ordering has substantial headroom
under queueing, but they do not show that this repository previously achieved
30% under the present strict causal full-drain protocol. The earlier public
stress120 result remains retained as a 10.040% lower-load reference. The
completed stress180 result below is a separate, stronger, explicitly tuned
load-sensitivity verification—not a retroactive validation of the historical
timeboxed or oracle numbers.

## Fixed queued workload

The queued screen raises load without changing the independent source set:

- 60 held-out source sessions, each represented by one original and two
  deterministic `break_prefix` variants: 180 load instances but only 60
  independent sources.
- 1,557 logical requests per cell, no arrival cap, `max_active_traces=180`, and
  full drain rather than a timebox.
- Recorded tool waits replayed as sleeps at 10× speedup. No live web tool or
  shared tool-side queue is executed.
- `max_tokens` is capped at 512 with a 64-token floor; model context is 16K.
- vLLM 0.10.1, TP=4 BF16 on 4 × A100-SXM4-40GB,
  `max-num-seqs=64`, and `max-num-batched-tokens=8192`.

The fixed workload manifest is
[`manifest_stress180.json`](../../artifacts/workloads/fixed_three_way_cap512_floor64/manifest_stress180.json).
Its canonical manifest SHA-256 is
`9dc47fe8fe0134cfbbe762165592f5b9852feb959cd1597425fdf230e9a46c91`;
the calibration-only learned mapper remains
`d4ac5ee9cebcb328ec153192fe4d78508cafd9dcff09cea5d025fb35f5818394`.
The duplicate variants are load generators, not additional statistical
samples, and stress180 remains development/load-sensitivity evidence rather
than a new untouched final evaluation.

## Baseline-only load selection

The operating point was selected using FCFS+none only, before comparing Joint
candidates. This avoids choosing the workload because a treatment happened to
look favorable.

| FCFS probe | Mean request | Mean queue | Queue share | Avg running / waiting | Max timeline KV | Preemptions / recorded swap events | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| GPU util `0.83` | 62.503 s | 35.944 s | 57.51% | 61.129 / 84.115 | 99.955% | 4 / 0 | Reject |
| GPU util `0.86` | 62.781 s | 36.185 s | 57.64% | 61.128 / 84.520 | 96.484% | 0 / 0 | Accept |

Both probes created the desired sustained queue. The `0.83` probe nevertheless
hit the KV boundary and incurred four preemptions; its derived
`kv_swap_happened` flag is therefore true even though the separate swap parser
recorded zero swap events. It is retained as a rejected load probe, not used as
evidence for a scheduler effect. See its
[`summary.json`](../../artifacts/queue180_baseline_r1/queue180_c512_m64_baseline_r1_fcfs_none/summary.json)
and
[`swap_summary.json`](../../artifacts/queue180_baseline_r1/queue180_c512_m64_baseline_r1_fcfs_none/swap_summary.json).

The accepted `0.86` baseline fully completed 1,557/1,557 requests with no retry,
preemption, or swap. Waiting was positive in 92.29% of timeline samples; average
waiting was 1.32× the native 64-sequence capacity. Only 0.40% of samples were
above 95% KV utilization. This retains meaningful queue pressure without making
memory eviction a competing explanation. The accepted raw evidence is
[`summary.json`](../../artifacts/queue180_u86_baseline_r1/queue180_c512_m64_u86_baseline_r1_fcfs_none/summary.json)
and
[`timeline.json`](../../artifacts/queue180_u86_baseline_r1/queue180_c512_m64_u86_baseline_r1_fcfs_none/timeline.json).

The accepted screening reference was:

| Mean task flow | P50 | P95 | Max | Makespan | Completion tokens |
|---:|---:|---:|---:|---:|---:|
| 555.650 s | 578.189 s | 660.710 s | 665.492 s | 667.456 s | 495,535 |

## One-run scheduler search ledger

These are development screens against the one accepted FCFS run, not fresh
paired replicates and not inferential results. Positive percentages in
parentheses are reductions versus FCFS; a negative percentage is a regression.
The label `fair120` refers to a 120-second eligibility deadline on the same
stress180 workload, not a 120-instance workload.

| Screen | Mean task flow | P50 | P95 | Makespan | Faster independent sources |
|---|---:|---:|---:|---:|---:|
| FCFS+none reference | 555.650 s | 578.189 s | 660.710 s | 667.456 s | — |
| Aggressive score/gate, no stage lanes | 467.615 s (**15.844%**) | 511.199 s (**11.586%**) | 661.256 s (**−0.083%**) | 671.091 s (**−0.545%**) | 53/60 |
| Stage-aware causal candidate | 385.506 s (**30.621%**) | 389.252 s (**32.677%**) | 585.839 s (**11.332%**) | 621.621 s (**6.867%**) | 58/60 |
| Stage-aware + `fair120` deadline | 393.150 s (**29.245%**) | 413.921 s (**28.411%**) | 606.190 s (**8.252%**) | 632.557 s (**5.229%**) | 58/60 |
| Non-causal `oracle_critical` diagnostic | 345.508 s (**37.819%**) | 333.254 s (**42.363%**) | 567.587 s (**14.094%**) | 612.953 s (**8.166%**) | 58/60 |

Service-side and integrity checks expose the tradeoff hidden by task means:

| Screen | Mean request | P95 request | Mean queue | Prefix-hit ratio | Completion-token delta | Retry / preempt / swap |
|---|---:|---:|---:|---:|---:|---:|
| FCFS+none reference | 62.781 s | 107.309 s | 36.185 s | 18.64% | reference | 0 / 0 / 0 |
| Aggressive, no stage lanes | 52.611 s (**16.200%**) | 106.684 s (**0.582%**) | 26.900 s (**25.659%**) | 37.10% | +0.703% | 0 / 0 / 0 |
| Stage-aware causal candidate | 43.119 s (**31.320%**) | 176.326 s (**−64.317%**) | 20.912 s (**42.208%**) | 58.05% | +0.368% | 1 / 0 / 0 |
| Stage-aware + `fair120` deadline | 44.002 s (**29.912%**) | 161.861 s (**−50.837%**) | 21.165 s (**41.509%**) | 57.65% | −0.111% | 0 / 0 / 0 |
| Non-causal `oracle_critical` diagnostic | 38.488 s (**38.696%**) | 201.643 s (**−87.909%**) | 17.696 s (**51.094%**) | 58.00% | −0.470% | 0 / 0 / 0 |

All screens fully drained 1,557/1,557 logical requests. The stage-aware screen
had one ambiguous transport disconnect followed by a successful explicit retry;
its final failure count is zero. The other listed screens had no retry. The raw
identity- and invariant-checked one-pair screening summaries for the first two
causal candidates are
[`aggressive`](../../artifacts/queue180_u86_candidate_aggr_r1/paired_vs_baseline.json)
and
[`stage-aware`](../../artifacts/queue180_u86_candidate_stage_r1/paired_vs_baseline.json).

The `fair120` values are a read-only, same-identity raw screen against the
accepted baseline. A formal paired JSON was intentionally not emitted because
the candidate records the expanded runtime-evidence schema while the older
baseline lacks several of those keys, so the validator fails closed on the
configuration record. Its primary raw evidence is
[`summary.json`](../../artifacts/queue180_u86_candidate_fair120_r1/queue180_c512_m64_u86_fair120_r1_joint_learned/summary.json)
and
[`request_events.jsonl`](../../artifacts/queue180_u86_candidate_fair120_r1/queue180_c512_m64_u86_fair120_r1_joint_learned/request_events.jsonl).
The oracle row is explicitly non-causal and comes from its
[`summary.json`](../../artifacts/queue180_u86_oracle_critical_r1/oracle_critical_stress180_r1/summary.json)
and
[`request_events.jsonl`](../../artifacts/queue180_u86_oracle_critical_r1/oracle_critical_stress180_r1/request_events.jsonl).

The stage-aware candidate was selected for formal replication because it was
the best causal mean-task screen while also improving task p95 and makespan.
The bounded-fairness variant is retained rather than hidden: it gives up 1.38
percentage points of mean-task reduction and 3.08 points of task-p95 reduction,
while reducing—but not eliminating—the request-p95 penalty.

## What changed, and what the screen says about mechanism

### Stage-aware admission is the largest new signal

The aggressive and stage-aware screens share the same continuous scoring,
foreground gate, aging, HBM controls and target64 capacity. The stage candidate
adds two causal ordering lanes: predicted final calls first, then fewer
predicted remaining calls. Mandatory deadline/fairness handling remains ahead
of those lanes, followed by the existing budget, score and arrival tie-breaks.
In the single-run screen this addition moved mean task flow from 467.615 s to
385.506 s and changed task p95 from a 0.083% regression to an 11.332%
improvement. This is strong screening evidence for completion-aware ordering,
but not yet a replicated ablation.

The `fair120` screen adds a bounded one-request fairness release at a running
floor of 63 after 120 seconds. It demonstrates a real objective tradeoff rather
than a free improvement: request p95 is less severe than under the selected
stage candidate, but task mean, task p95 and makespan are all weaker.

### Most of the request gain is queue reduction

In the formal aggregate, mean request latency falls by 19.901 s and mean queue
time falls by 15.338 s. Thus 77.1% of the request-latency reduction is the
measured queue component; the remaining 4.563 s is the approximate
inference/service component. In the earlier selected screen, prefix-hit ratio
rose from 18.64% to 58.05%, consistent with better session/prefix locality and
a less contentious running set. Prefix hit is post-hoc screening evidence,
however, not an isolated causal contribution or a formal aggregate endpoint.

### Learned overlap is not the source of a 30% direct saving here

The stress180 learned workload removes 103.685 seconds of recorded wait before
the 10× replay scaling. At runtime that is only 10.368 seconds across 180 tasks,
or approximately **0.0576 s per task**. Against the screen's 170.143 s and the
formal aggregate's 172.202 s mean task savings, the direct recorded-wait
subtraction is below 0.04%. The A/D effect can include interactions, but it
cannot accurately be described as a 30% pacing or tool-prefetch saving. The
dominant observed path is task-aware LLM admission, queue reduction and
prefix/HBM locality.

### Weak next-wait prediction is gated off, and target64 is not throttling

The calibration-only leave-one-source-out reliability for the current
next-tool-wait predictor is zero. Online request metadata therefore records
`scheduled_nw_reliability=0.0`, and the scheduler multiplies the next-wait
bonus by zero rather than trusting a weak estimate. Stage/final progress and
remaining-call predictions still use only calibration and information observed
up to the current call; no oracle fields are populated in the causal screens.

Both decode target and decode maximum are 64, equal to native
`max-num-seqs=64`. Consequently this profile does not reduce the decode band
below native capacity and must not be presented as a pacing-only result.

The implementation is in
[`sched_policy_patch.py`](../../../scripts/pythonhooks/sched_policy_patch.py),
and the frozen candidate configuration is
[`joint_stress180_u86_stage.env.example`](../../configs/joint_stress180_u86_stage.env.example).

## Formal three-replicate fresh-server A/D result

All six cells completed and passed strict validation. The machine-readable
aggregate is
[`paired_stress180_stage_3x.json`](../../artifacts/stress180_u86_stage_formal/paired_stress180_stage_3x.json).
A compact committed view of the same validated fields is
[`summary_stress180_u86_stage_3x.json`](summary_stress180_u86_stage_3x.json).
Its evidence status is
`paired_stress180_ad_load_sensitivity_not_independent_not_final`: “formal” here
means a frozen, repeated verification, not an untouched final evaluation.

Each cell value below is the arithmetic mean of its three replicate-level
metrics, and each relative effect is computed after that aggregation. The
percentiles are therefore means of replicate percentiles, not percentiles from
pooling 540 non-independent load instances.

| Cell | vLLM scheduler | Tool-overlap mode |
|---|---|---|
| A | `fcfs` | `none` |
| D | `online_joint_pacer_v2` with stage lanes | checksummed `learned` top-5 mapper |

| Formal metric | A: FCFS+none | D: stage-aware Joint+learned | Reduction |
|---|---:|---:|---:|
| Mean task flow | 548.080 s | 375.878 s | **31.419%** |
| P50 task flow | 571.970 s | 380.614 s | **33.456%** |
| P95 task flow | 651.891 s | 576.942 s | **11.497%** |
| Max task flow | 656.648 s | 607.353 s | **7.507%** |
| Task makespan | 658.054 s | 608.554 s | **7.522%** |
| Instrumentation wall time | 658.581 s | 609.085 s | **7.516%** |
| Mean request latency | 61.907 s | 42.006 s | **32.147%** |
| P50 request latency | 59.094 s | 22.656 s | **61.660%** |
| P95 request latency | 106.180 s | 171.518 s | **−61.536%** (regression) |
| Max request latency | 117.601 s | 352.036 s | **−199.348%** (regression) |
| Mean queue time | 35.626 s | 20.288 s | **43.053%** |

Of the 19.901 s formal mean-request saving, 15.338 s (77.1%) is the measured
queue component; the residual 4.563 s is the approximate service component.
The per-replicate mean prefix-hit ratio rises from 18.89% under A to 57.87%
under D. This is consistent with the screening mechanism diagnosis, while
remaining observational rather than an isolated prefix-locality ablation.

The task-level effect was consistent across fresh-server pairs:

| Replicate | Mean reduction | P50 reduction | P95 reduction | Makespan reduction |
|---|---:|---:|---:|---:|
| 1 | 32.076% | 33.634% | 10.911% | 7.456% |
| 2 | 30.169% | 32.153% | 10.925% | 6.541% |
| 3 | 32.015% | 34.563% | 12.642% | 8.565% |

At the correct independent unit, D was faster for 58/60 source sessions. The
source-level mean saving was 172.202 s. A fixed-seed (`20260815`),
10,000-resample nonparametric percentile bootstrap over the 60 independent
source-session means gives a 95% interval of **[151.366, 191.673] s**. Each
source mean first averages its three deterministic variants within a replicate
and then its three repeated measurements; neither variants nor replicates
increase the independent sample size beyond 60.

All 9,342/9,342 logical requests succeeded exactly once: total attempts were
also 9,342, with zero retry, final failure, preemption, or swap event. Mean
completion tokens per replicate were 496,318.33 under A and 497,368.33 under D,
so D generated **0.212% more**, within the declared 1% comparability guard. The
gain is therefore not explained by less generated output.

All three frozen configuration files have SHA-256
`e4417bdd8e609feeb576f1c277be942dcffbb4e5b2572f0572b212611eac2551`;
the validator also confirms the canonical workload-manifest SHA-256
`9dc47fe8fe0134cfbbe762165592f5b9852feb959cd1597425fdf230e9a46c91`,
identical scheduler configuration within each A/D pair, online metadata, and no
oracle trajectory fields. The formal task endpoint improves, but the request
p95 and maximum regress materially because completion-aware ordering delays
some individual requests. That tradeoff is part of the result, not an omitted
outlier.

## Limitations and claim boundary

- This workload is deliberately tuned after examining earlier results. It is a
  queued load-sensitivity experiment, not a new untouched final benchmark.
- The 180 instances contain only 60 independent sources. Any uncertainty
  analysis must collapse the three deterministic variants within each source.
- Screening reused one accepted baseline and was used for candidate selection.
  The reported aggregate uses three new fresh-server A/D pairs, but candidate
  and workload selection still make this tuned development evidence.
- Recorded sleeps are not live tools. There is no shared tool executor, tool
  queue, network variability, or feedback from generated text into later tool
  calls. This does not evaluate the paper's complete joint tool/LLM system.
- The current next-wait predictor contributes no score because its calibration
  reliability is zero. A richer live tool-side state model remains a separate
  engineering direction, not an implemented explanation for this result.
- Completion-aware ordering intentionally favors task completion. In the formal
  aggregate, request p95 regresses 61.536% and request maximum regresses
  199.348%; both must accompany any task-mean claim.
- The non-causal oracle screen provides a one-run upper-bound diagnostic above
  30%; it is not a deployable policy or part of the causal A/D.
- The hardware and runtime are a 4 × A100-40GB, 16K-context minimal replay, not
  the paper's larger evaluation system. Absolute numbers must not be compared
  as a paper reproduction.

Subject to these boundaries, the completed fresh-server verification answers
the immediate engineering question: under sustained queueing, the causal
completion-aware candidate reduces mean task flow by **31.419%**, with all three
replicates between 30.169% and 32.076%. Task p95 and makespan also improve in
every replicate, and 58/60 independent sources are faster. This is a task-level
win for the fixed recorded-sleep stress replay, accompanied by a substantial
individual-request tail tradeoff; it is not a claim about live-tool execution
or the paper's complete system.
