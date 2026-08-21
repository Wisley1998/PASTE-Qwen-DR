# Dynamic physical-KV admission: stress300 one-screen report

Date: 2026-08-16

## Result and scope

The strict stress300 screens pass every frozen promotion boundary.  A is vLLM
FCFS with no tool overlap and native admission; B adds Joint ordering and
learned overlap while leaving admission entirely native; C holds B's ordering
and overlap fixed and replaces native admission with adaptive physical-KV
admission.

Across the fixed workload, mean task E2E falls from 816.393 s in A to
669.925 s in B and 621.664 s in C.  Thus A to B saves 146.468 s
(**17.941%**), B to C saves a further 48.261 s (**7.204%**), and the full A to
C bundle saves 194.729 s (**23.852%**).  At the independent-source level, the
incremental B-to-C saving has a 10,000-resample bootstrap 95% interval of
**[13.501, 83.502] s**, with C faster for 43/60 sources.  The full A-to-C
interval is **[150.616, 238.413] s**, with C faster for 54/60 sources.

These are retained **single-screen load-sensitivity results**, not a formal
fresh-server paired replication.  Each cell was run on a fresh server, but A,
C, and the later B screen are noncontemporaneous: A was reused from its
independently accepted A-only probe, and B was compared with an immutable
historical C.  B/C isolates the configured native-to-physical admission change
far better than A/C, but one nonrandomized cell per mode is not repeated-run
causal evidence.

## Primary evidence

- [Strict A/C screen](../../artifacts/stress300_u86_native320_g256_physical093_exact_rescue120/stress300_c_physical093_r1/strict_a_vs_c_physical_v2.json),
  SHA-256
  `906df1cd484311c3acbf701720d49cc3c0f516f5b48bf78e9e51ec1b5fcc7771`.
- [Strict B/C incremental screen](../../artifacts/stress300_u86_native320_g256_native_exact_rescue120_b_screen/stress300_b_native_r1/strict_b_vs_c_physical_v2.json),
  SHA-256
  `8e9db08d1fa2558ff3fe2a5d8a4de4988ae059470bc375e6dbced1e60a686d4b`.
- [B native-admission zero-write validation](../../artifacts/stress300_u86_native320_g256_native_exact_rescue120_b_screen/stress300_b_native_r1/native_admission_zero_write_v2.json),
  SHA-256
  `6138577e44a5eba666877fdd4be4e3e409d8840f5aa5cfdcf4975b853f278977`.
- [Fresh parser-v2 physical-KV validation](../../artifacts/stress300_u86_native320_g256_physical093_exact_rescue120/stress300_c_physical093_r1/physical_kv_validation_v2.json),
  SHA-256
  `b292c04f0bdaf53ec9bea4ff290a8517f19cdc277d2eca908eb055c24dbf252e`.
- [Accepted A-only natural-queue probe](../../artifacts/stress300_u86_native320_g256_keepalive60_a_probe/stress300_a_probe_r3/natural_queue_probe.json),
  SHA-256
  `c2a5b098a178e7e9d899ea88995f0f591bb24ec70380c2d5242bc734d2c247bd`.
- [Accepted-A fresh revalidation snapshot](../../artifacts/stress300_u86_native320_g256_physical093_exact_rescue120/stress300_c_physical093_r1/accepted_a_probe_validation.json),
  SHA-256
  `610067f45ad773bb3172a3ec9c76a75cdb28f73aa4afa6f708b934dca2fc95d5`.
- [Fixed stress300 manifest](../../artifacts/workloads/fixed_three_way_cap512_floor64/manifest_stress300.json),
  file SHA-256
  `43f6d9dee3f12c4d31f7195e1616fa0ffd21ac98e8a7bdbffe3089be378318fa`.

The frozen [A configuration](../../configs/joint_stress300_u86_native320_keepalive60_a_probe.env.example),
[B configuration](../../configs/joint_stress300_u86_native320_native_exact_rescue120_b_screen.env.example),
and [C configuration](../../configs/joint_stress300_u86_native320_physical093_exact_rescue120.env.example)
hashes are, respectively,
`c1c043836601203c4f49284daf8b7e925bab450747482e486eed83897dda2d06`,
`e024ab17e6b08c1c1cd3246e4b74b253b681af152138af762bc536f7b513908e`,
and
`1ee7dfe9f5831223fb4ff14c1e86154827d32d7835d11b2749c8e07863321d43`.
The strict screens verify identical engine shape, exact request and source
identity, fixed workload/calibration/mapper identities, the exact nine-key A/C
allowlist, and the exact seven-key B/C scheduler allowlist.

## No fixed 64-request ceiling

This screen directly tests the configuration requested after the earlier
target-64 runs:

- 300 trace instances are offered concurrently, with at most one outstanding
  LLM request per trace;
- `VLLM_MAX_NUM_SEQS=320`, leaving 20 requests of structural headroom over the
  offered-concurrency upper bound;
- no timeline sample reaches 320 running requests;
- `VLLM_CUDA_GRAPH_SIZES=256` is a graph-capture shape, not an admission cap;
- A and B use native vLLM admission.  Joint does not install a private
  running-count limit in either cell.

A reaches 300 running requests while also showing waiting requests below the
320 sequence cap in 1,735/1,890 samples (91.799%).  Waiting peaks at 203 and
averages 127.781 requests; GPU KV utilization averages 93.522% and reaches
100%.  Mean vLLM queue time is 49.321 s, or 53.076% of mean request latency.
The A probe therefore establishes a real native vLLM resource queue with a
nonbinding sequence-count setting.  It does not, by itself, distinguish the
instantaneous contributions of the batched-token budget, physical KV, or
other native serving resources.

B provides a stronger implementation-level check for the reorder-only case.
Across 193 Joint-cap observations, its cap is always the native value 320
(minimum, maximum, and sole distinct value); physical-capacity writes,
physical write tokens, and physical telemetry markers are all exactly zero.
Nevertheless, vLLM reports maximum running/waiting populations of 300/157,
and B observes running above 64.  B therefore changes waiting order while
vLLM itself retains admission and forms the serving queue.

C intentionally does **not** claim native admission: adaptive physical-KV
admission is part of the treatment.  Its dynamic effective cap spans 2--300
with 238 distinct values, and 1,950 validated samples have both pressure and
an effective cap above 64.  Thus C is neither restricted to 64 nor governed by
a fixed 300-request target; the cap follows observed and forecast KV demand.

## Workload and cells

| Cell | Scheduler/order | Tool overlap | Admission |
|---|---|---|---|
| A | vLLM `fcfs` | `none` | native vLLM |
| B | `online_joint_pacer_v2` | checksummed `learned` | native vLLM; reorder only, zero cap writes |
| C | `online_joint_pacer_v2` | checksummed `learned` | adaptive physical KV at target 0.93 |

All three cells use the same pinned model revision, TP=4, BF16, 16K context,
8,192 batched-token budget, GPU-memory utilization 0.86, 512 output-token cap,
vLLM v1, and HTTP keepalive 60 s.  The workload contains 300 concurrent trace
instances and 2,595 logical requests per cell.  It is built from 60 held-out
source sessions, each represented by five deterministic `break_prefix`
copies.  Those copies create serving load but do not create 300 independent
samples; source-level inference first averages the five copies for each of the
60 sources.

## Latency results

Lower is better.  Percentage columns use the left cell as denominator; a
negative reduction denotes a regression.

| Metric | A: FCFS native | B: Joint native | C: Joint physical | A to B | B to C | A to C |
|---|---:|---:|---:|---:|---:|---:|
| Mean task E2E | 816.393 s | 669.925 s | 621.664 s | **17.941%** | **7.204%** | **23.852%** |
| P50 task E2E | 865.008 s | 716.020 s | 672.105 s | **17.224%** | **6.133%** | **22.301%** |
| P95 task E2E | 994.634 s | 1,012.837 s | 972.899 s | **-1.830%** | **3.943%** | **2.185%** |
| P99 task E2E | 1,002.722 s | 1,025.596 s | 992.413 s | **-2.281%** | **3.236%** | **1.028%** |
| Max task E2E | 1,004.288 s | 1,026.480 s | 993.387 s | **-2.210%** | **3.224%** | **1.085%** |
| Makespan | 1,004.642 s | 1,027.375 s | 993.880 s | **-2.263%** | **3.260%** | **1.071%** |
| Mean request latency | 92.925 s | 76.000 s | 70.420 s | **18.214%** | **7.342%** | **24.219%** |
| P50 request latency | 94.228 s | 51.943 s | 54.243 s | **44.875%** | **-4.427%** | **42.435%** |
| P95 request latency | 155.982 s | 196.826 s | 187.513 s | **-26.185%** | **4.732%** | **-20.214%** |
| P99 request latency | 162.445 s | 215.294 s | 198.480 s | **-32.533%** | **7.810%** | **-22.183%** |
| Max request latency | 171.899 s | 225.011 s | 201.949 s | **-30.897%** | **10.249%** | **-17.481%** |
| Mean request queue time | 49.321 s | 31.256 s | 31.163 s | **36.627%** | **0.299%** | **36.817%** |
| Mean nonqueue request time | 43.604 s | 44.744 s | 39.258 s | **-2.613%** | **12.261%** | **9.969%** |

For B/C, C is faster for 214/300 load instances and 43/60 folded source
sessions.  The source-folded mean saving is 48.261 s with bootstrap 95%
interval **[13.501, 83.502] s**.  For the full A/C bundle, C is faster for
264/300 instances and 54/60 sources, with mean saving 194.729 s and interval
**[150.616, 238.413] s**.

The exact same-identity source rows also permit the algebraic A/B contrast
`(A-C) - (B-C)`: B is faster for 46/60 sources, with mean saving 146.468 s and
fixed-seed bootstrap interval **[106.409, 186.389] s**.  This is derived from
the two strict source tables, not a third independent run or an additional
replicate.  All intervals use the independent source-session mean as the
sampling unit, seed 20260815, and 10,000 nonparametric percentile resamples.
They describe these fixed screens, not run-to-run uncertainty from fresh
replicated blocks.

## Where the observed saving appears

The strict accounting identity is:

```text
task mean = queue + nonqueue request + noninitial recorded tool wait
          + residual harness/timing
```

Because the same B cell forms the endpoint of A-to-B and the start of B-to-C,
the absolute component savings add exactly:

| Mean-task component | A to B saving | B to C saving | A to C saving |
|---|---:|---:|---:|
| Queue | **+156.261 s** | +0.809 s | +157.070 s |
| Nonqueue request | **-9.855 s** | **+47.455 s** | +37.600 s |
| Noninitial recorded tool wait | +0.058 s | ~0.000 s | +0.058 s |
| Residual harness/timing | +0.004 s | -0.002 s | +0.001 s |
| Total | **+146.468 s** | **+48.261 s** | **+194.729 s** |

A-to-B keeps admission native and is a queue/order-dominated effect: queue
saving is 156.261 s, more than the net saving, while nonqueue request time
regresses by 9.855 s.  Recomputation preemptions barely change, from 506 to
496.  Because A-to-B changes both FCFS/none to Joint/learned, this leg does not
separate task-aware ordering from learned overlap.

B-to-C holds Joint ordering, learned overlap, engine shape, workload, and
request identities fixed.  Its exact seven-key configuration difference is
native reorder-only versus adaptive physical-KV admission.  Queue contributes
only 0.809 s (1.676%) of the 48.261 s saving, whereas nonqueue request time
contributes 47.455 s (98.329%); mean per-request queue time is nearly unchanged
at 31.256 s versus 31.163 s.  Concurrently, preemptions fall from 496 to zero.
This is direct evidence that the additional B-to-C gain is not ordinary queue
reordering: it is associated with physical-KV admission and elimination of
recomputation pressure.

Across A-to-C, queue and nonqueue components contribute 157.070 s (80.661%)
and 37.600 s (19.309%).  The accounting identity is exact, but causal language
still needs the single-screen qualification: B was run later than the frozen C,
and there is no fresh randomized replication of any leg.

## Dynamic physical-KV evidence

C derives physical capacity from vLLM's GPU block allocation and applies a
0.93 soft utilization target.  It forecasts the footprint of running growth,
tool-return reserve, and waiting candidates in Joint order, admits the ordered
fit set, and writes a dynamic effective cap.  A request aged beyond 120 s may
use the rescue path above the soft reserve while remaining within physical
capacity.

| Validated physical-KV fact | Value |
|---|---:|
| GPU blocks / block size | 44,178 / 16 tokens |
| Minimum rank capacity used by controller | 706,848 tokens |
| Raw rank capacities | 719,136 / 706,848 / 706,848 / 719,136 tokens |
| Soft budget at target 0.93 | 657,360 tokens |
| Experiment telemetry samples | 2,539 |
| Effective-cap range / distinct values | 2--300 / 238 |
| Cap increases / decreases | 1,123 / 1,300 |
| Pressure samples above 64 | 1,950 |
| Over-soft zero-admit forecast holds | 455 |
| Aged-request rescue samples | 329 |
| Capacity-write counter range | 2--6,763, strictly increasing |
| Malformed / fail-closed decisions | 0 / 0 |

Parser-v2 accounts for all 2,540 full-lifecycle raw markers, excludes only the
single warmup marker, and exactly matches the 2,539 stored experiment samples.
Capacity is stable and equals the minimum reported tensor-parallel rank
capacity.  The validation observes both positive and zero fit/admit decisions,
cap increases and decreases, no preemption, and no CPU-KV swap.  This proves
that physical telemetry actively drove a dynamic controller; it does not
prove that target 0.93 is globally optimal.

The canonical [C server log](../../artifacts/stress300_u86_native320_g256_physical093_exact_rescue120/stress300_c_physical093_r1/stress300_c_physical093_r1_joint_learned/server/vllm_8100.log)
has SHA-256
`c2eb67a5f6bb737991e485487fe08124a630a4c2f1d57db6e19ac37c34d9a17e`;
the validator binds this raw log and the parser/validator implementations by
hash.

## Reliability, token accounting, and tail trade-off

| Evidence | A | B | C |
|---|---:|---:|---:|
| Successful logical requests | 2,595 | 2,595 | 2,595 |
| HTTP attempts | 2,595 | 2,595 | 2,595 |
| Retries / ambiguous retries / failures | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Completion tokens | 827,133 | 828,310 | 827,855 |
| Recomputation preemptions | 506 | 496 | 0 |
| CPU-KV swap events | 0 | 0 | 0 |
| Requests over 120 s | 610 | 636 | 579 |
| Requests over 240 s | 0 | 0 | 0 |

C produces 455 fewer completion tokens than B (-0.0549%) but 722 more than A
(+0.0873%); neither comparison approaches the frozen 1% guard.  The full
bundle benefit is therefore not obtained by generating fewer output tokens.
Exact-once completion and zero retries in all cells rule out transport-retry
imbalance.  Preemptions move only 506 to 496 in the queue-dominated A-to-B leg,
then fall to zero in the nonqueue-dominated B-to-C leg; CPU-KV swap remains
zero throughout.

The tail picture also separates the mechanisms.  A-to-B improves mean and
median request latency but worsens request P95/P99/max by
26.185%/32.533%/30.897%, task P95 by 1.830%, and makespan by 2.263%.
B-to-C improves request P95/P99/max by 4.732%/7.810%/10.249%, task P95 by
3.943%, and makespan by 3.260%, although request P50 regresses by 4.427%;
requests over 120 s fall from 636 to 579.
However, C does not fully undo A-to-B's individual-request tail trade-off:
relative to A, C request P95/P99/max remain 20.214%/22.183%/17.481% worse.
Task P95/P99/max and makespan finish 1.0--2.2% better than A.  Reporting the
mean-task gain without both sides of this trade-off would be incomplete.

## Promotion boundaries

All frozen classification gates pass.  For the full A/C promotion:

| Boundary | Observed | Required |
|---|---:|---:|
| Mean task E2E reduction | 23.852% | at least 15% |
| Independent sources faster | 54/60 | at least 48/60 |
| Source-bootstrap 95% lower bound | 150.616 s | above 0 s |
| Completion-token absolute difference | 0.0873% | below 1% |
| C/A request-P99 ratio | 1.2218 | at most 1.5 |
| Requests over 240 s | 0 vs 0 | must not increase |
| Task P95 | 972.899 vs 994.634 s | must not regress |
| C/A makespan ratio | 0.9893 | at most 1.03 |

For the incremental B/C physical-admission classification:

| Boundary | Observed | Required |
|---|---:|---:|
| Mean task E2E reduction | 7.204% | above 0% |
| Independent sources faster | 43/60 | strict majority, at least 31/60 |
| Source-bootstrap 95% lower bound | 13.501 s | above 0 s |
| Completion-token absolute difference | 0.0549% | below 1% |
| C/B request-P99 ratio | 0.9219 | at most 1.5 |
| Requests over 240 s | 0 vs 0 | must not increase |
| Task P95 | 972.899 vs 1,012.837 s | must not regress |
| C/B makespan ratio | 0.9674 | at most 1.03 |

The separately preregistered 0.95 follow-up is **not permitted**: it was
allowed only if the validated-safe 0.93 candidate made makespan or throughput
strictly more than 3% worse than A, whereas C improves both in this screen.
The gates classify the already completed run; they do not turn this one screen
into a replicated estimate.

## Claim boundaries and next evidence

- **Single cells and noncontemporaneous comparisons.** A is an earlier
  accepted fresh-server A-only probe, C is one later fresh-server candidate,
  and B was run later still against the immutable historical C.  These are not
  randomized contemporaneous pairs and not a three-block replication.
- **60 independent sources, not 300.** Five deterministic copies of each of
  60 sources create concurrency.  Copies are folded before inference.
- **Recorded-wait replay, not live tools.** Tool waits are recorded waits
  replayed at 10x speed.  No live tool/web service executes, and generated
  text does not determine the recorded next request.  This is not the paper's
  full live system.
- **Bundle versus incremental effect.** A-to-C simultaneously changes FCFS to
  Joint ordering, none to learned overlap, and native to physical-KV
  admission.  B-to-C holds Joint/learned fixed and its exact configuration
  allowlist isolates native versus physical admission, but remains a
  noncontemporaneous single-screen estimate rather than replicated causality.
- **Derived A/B source inference.** The A/B source deltas are an exact
  algebraic contrast of same-identity rows in the strict A/C and B/C tables;
  they are not an extra run or an additional independent dataset.
- **Load-sensitivity screen.** stress300 deliberately increases pressure; it
  should not be pooled with stress240 or presented as an untouched final
  evaluation.  The 23.852% result is below the stress240 one-screen 25.385%
  relative reduction, despite a larger 194.729 s absolute saving.
- **Tail qualification.** Reporting the mean-task gain without the request
  P95/P99 regression and small task-tail/makespan gains would be incomplete.

The defensible next step is a prospectively frozen, balanced fresh-server
A/B/C replication under one fixed load, with at least three blocks and no
retuning from the observed cells.  It should keep the nonbinding sequence-cap
proof, source-folded inference, exact retry/token/preemption/swap accounting,
parser-v2 physical telemetry, and simultaneous request-tail guardrails.
