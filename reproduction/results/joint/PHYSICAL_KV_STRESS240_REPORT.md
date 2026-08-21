# Dynamic physical-KV admission: stress240 one-screen report

Date: 2026-08-16

This report records the first strict stress240 screen with a nonbinding native
sequence limit and dynamic physical-KV admission. It answers two separate
questions:

1. How much does the full FCFS-to-Joint bundle improve task latency when the
   workload is allowed to form a real vLLM resource queue?
2. Holding Joint ordering and learned overlap fixed, what changes when native
   admission is replaced by adaptive physical-KV admission?

The answers are **25.385% A→C mean-task reduction for the full bundle** and an
additional **10.426% B→C reduction for physical-KV admission relative to the
Joint native-admission anchor**. These are one-screen development results, not
fresh-server replicated estimates. In particular, the 25.385% result must not
be attributed to physical admission alone.

## Evidence and checksums

The primary statistical artifacts are:

- [strict A→C screen](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/stress240_c_physical093_r1_joint_learned/strict_a_vs_c_physical093.json),
  SHA-256
  `eba78f2481841630bd9dd14aa9e7b105d95798963a2ee675815a1ea64e59aa56`;
- [strict B→C incremental screen](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/stress240_c_physical093_r1_joint_learned/strict_b_vs_c_physical093.json),
  SHA-256
  `573a73c1d407adf800d0b85911995de37ada51dc843f2f7db2ad2472019ec8af`;
- [raw-log physical-KV revalidation](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/physical_kv_revalidation.json),
  SHA-256
  `b9baeaa4f2d7792ba9803b618a0db98ccc922b1ab11d5ad5e9be86d2e8102574`;
- [strict A→B reference screen](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/reference_b_strict_screening.json),
  SHA-256
  `234f117467c2cb3fa1e6068551c27364961c89aef48405c2fecb15639b5f5509`;
- [accepted natural-queue A evidence](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/accepted_a_probe.json),
  SHA-256
  `5876f1c37849d0c6c1643f8e1950b042bca914e4a7132fb106eb4a97a8da0e3e`.

The fixed [stress240 manifest](../../artifacts/workloads/fixed_three_way_cap512_floor64/manifest_stress240.json)
has file SHA-256
`cd790a2f96a947198f1c8e23e0c1e0c3beb6c715ca0278dc3c551d66aab277dd`.
The A, B, and C frozen-config hashes are respectively
`1bd0073e891083efeed328c5d0e925832772d73ee25946f0a18f80d504989102`,
`d0f9f486f9cdd14aa3fd970086682b31220b4666b03af91cc77a75951d1065b0`,
and `00f4064146b4e88f90bca340f91ec9ec8ca6beafe807813b03fc462f5f30c54d`.

## Why the fixed 64 limit was removed

The earlier stress120 result used both native `max-num-seqs=64` and Joint
target/max settings at 64. That configuration was valid for its original
load-sensitivity question, but it could not demonstrate what vLLM would do
above 64 or whether a dynamic controller would choose a substantially
different running population. A fixed cap also mixes two distinct effects:
resource pressure created by vLLM itself and pressure manufactured by an
external running-count ceiling.

Stress240 therefore uses:

- 240 offered trace instances;
- `VLLM_MAX_NUM_SEQS=256` and CUDA graph size 256;
- no fixed physical-admission running target, minimum-running floor, or
  maximum-admit count;
- native admission in A and B, and physical-KV-derived admission in C.

The offered concurrency is 16 below the configured sequence limit. A reached
240 running requests without ever reaching 256, while waiting requests were
present below the sequence cap in all 1,306 waiting timeline samples. Thus the
queue was formed by native vLLM resource pressure rather than a configured
64-request ceiling. C's effective cap subsequently ranged from 2 to 240; its
maximum was bounded by the offered workload, not a hidden 64 or a binding 256.

## Cells and workload

| Cell | Ordering / scheduler | Tool overlap | Admission |
|---|---|---|---|
| A | vLLM `fcfs` | `none` | native |
| B | `online_joint_pacer_v2` | checksummed `learned` | native; reorder only |
| C | `online_joint_pacer_v2` | the same checksummed `learned` | adaptive physical KV, target 0.93 |

All cells use the same model revision, TP=4 BF16 engine shape, 16K context,
8,192 batched-token limit, deterministic request identities, source mapping,
and stress240 workload. The 60 unique held-out source sessions are each
represented by four deterministic `break_prefix` load copies: 240 concurrent
instances and 2,076 logical requests per cell. The copies create load but are
not independent observations; inference folds them into 60 source means.

The strict B→C comparator requires its seven actual configuration differences
to equal the exact native-to-physical allowlist. The A→C comparator likewise
requires all ten scheduler-configuration differences to match its explicit
allowlist, while checking `fcfs+none` and `Joint+learned` separately as exact
mode expectations. All required engine, request, source, workload,
calibration, and mapper checks pass.

## A/B/C results

Lower is better. Percentage columns use the left cell as denominator.

| Metric | A: FCFS native | B: Joint native | C: Joint physical | A→C | B→C |
|---|---:|---:|---:|---:|---:|
| Mean task E2E | 627.575 s | 522.765 s | 468.262 s | **25.385%** | **10.426%** |
| P50 task E2E | 654.055 s | 570.834 s | 507.626 s | **22.388%** | **11.073%** |
| P95 task E2E | 787.281 s | 770.893 s | 720.608 s | **8.469%** | **6.523%** |
| P99 task E2E | 792.046 s | 777.440 s | 742.529 s | **6.252%** | **4.490%** |
| Max task E2E | 792.816 s | 778.936 s | 743.846 s | **6.177%** | **4.505%** |
| Makespan | 793.160 s | 779.290 s | 744.166 s | **6.177%** | **4.507%** |
| Mean request latency | 71.099 s | 58.987 s | 52.687 s | **25.897%** | **10.681%** |
| P50 request latency | 69.807 s | 38.354 s | 37.865 s | **45.757%** | **1.275%** |
| P95 request latency | 129.310 s | 186.931 s | 169.726 s | **−31.255%** | **9.204%** |
| P99 request latency | 135.806 s | 210.937 s | 185.571 s | **−36.644%** | **12.026%** |
| Max request latency | 140.597 s | 219.319 s | 196.038 s | **−39.433%** | **10.615%** |
| Requests over 120 s | 203 | 258 | 255 | 52 more | 3 fewer |
| Requests over 240 s | 0 | 0 | 0 | unchanged | unchanged |

At the correct independent unit, C is faster than A for 57/60 source sessions
and 222/240 load instances. The source-folded mean A→C saving is 159.313 s;
a fixed-seed 10,000-resample bootstrap over 60 source means gives a 95%
interval of **[131.685, 187.368] s**. For B→C, C is faster for 48/60 sources
and 189/240 instances; mean saving is 54.503 s with interval
**[32.565, 77.359] s**. These are descriptive intervals for this fixed screen,
not repeated-run uncertainty estimates.

## How the two mechanisms add

Because the strict A→B and B→C screens share the exact same B cell, their
absolute task-mean savings add arithmetically:

```text
A → B: 627.575 - 522.765 = 104.810 s
B → C: 522.765 - 468.262 =  54.503 s
A → C: 627.575 - 468.262 = 159.313 s
```

The relative percentages do not add: 16.701% followed by 10.426% gives
`1 - (522.765 / 627.575) × (468.262 / 522.765) = 25.385%`.

The component identity is task mean = queue + nonqueue request + noninitial
recorded tool wait + residual harness/timing. Positive entries below contribute
to the right-hand cell's saving.

| Mean-task component | A→B saving | B→C saving | A→C saving |
|---|---:|---:|---:|
| Queue | +102.368 s | **−8.080 s** | +94.288 s |
| Nonqueue request | +2.400 s | **+62.579 s** | +64.980 s |
| Noninitial recorded tool wait | +0.058 s | ~0.000 s | +0.058 s |
| Residual harness/timing | −0.016 s | +0.004 s | −0.012 s |
| Total | **+104.810 s** | **+54.503 s** | **+159.313 s** |

This is the central mechanism result:

- A→B is almost entirely a queue/task-ordering effect under the combined
  Joint-ordering and learned-overlap treatment.
- B→C is not a queue reduction. C's mean per-request queue time is slightly
  worse than B's (18.005 s versus 17.071 s), while mean nonqueue request time
  falls from 41.917 s to 34.682 s and preemptions fall from 317 to zero.
- Across the full A→C bundle, queue savings contribute 94.288 s (59.18%) and
  nonqueue request savings contribute 64.980 s (40.79%).

The decomposition describes observed time accounting; it is not a standalone
causal proof for an internal kernel or scheduler subroutine. A→C changes the
whole FCFS/overlap/Joint/admission bundle. Only B→C holds Joint ordering and
learned overlap fixed, and even that is one screen rather than a replicated
ablation.

## Native queue evidence

The [accepted A probe](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/accepted_a_probe.json)
establishes a native vLLM resource queue with a nonbinding sequence cap:

- configured `max-num-seqs`: 256;
- offered and observed maximum running requests: 240;
- configured and observed sequence headroom: 16;
- waiting samples: 1,306/1,496 (87.299%);
- maximum and mean waiting requests: 138 and 76.129;
- mean/max GPU KV usage: 92.369%/100%;
- mean request queue time: 28.905 s;
- sequence-cap-reached samples: zero.

B independently satisfies the same native-queue criterion with maximum running
240, maximum waiting 120, and mean waiting 46.241. C intentionally does not
claim native admission: it is the adaptive-admission treatment. Its safety and
dynamic behavior are established by the independent B→C comparator and the
raw-log revalidation rather than by relabeling C as native.

This evidence proves that A and B queued below a nonbinding sequence count. It
does not by itself identify whether token budget, physical KV availability, or
another native resource was the dominant instantaneous cause.

## Dynamic physical-KV admission and telemetry

At each scheduler decision, C derives physical capacity from
`num_gpu_blocks × block_size`, observes live KV usage, and forecasts the
additional footprint of already-running growth, tool-return reserve, and
waiting candidates in Joint order. It admits the largest ordered fit set under
the 0.93 soft physical-KV budget and writes
`effective_cap = min(native_cap, running + admit)`. There is no fixed running
target in this branch.

An aged-request rescue may cross the 0.93 reserve to preserve progress, but it
may not cross 100% physical capacity. An empty-running fallback likewise
ensures progress. The controller records capacity, budget, live and logical
live tokens, predicted growth/reserve/admission, running/waiting counts,
fit/admit counts, effective/native caps, rescue state, reason, write source,
and a monotonic write counter. See the canonical
[C vLLM log](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/stress240_c_physical093_r1_joint_learned/server/vllm_8100.log)
and the [revalidation sidecar](../../artifacts/stress240_u86_native256_g256_physical093_exact_rescue120/stress240_c_physical093_r1/physical_kv_revalidation.json).

Observed experiment-scope telemetry:

| Physical-KV fact | Value |
|---|---:|
| Physical capacity | 721,904 tokens |
| Target utilization | 0.93 |
| Native safety cap | 256 |
| Parsed experiment samples | 1,988 |
| Effective-cap range | 2–240 |
| Distinct effective caps | 208 |
| Cap increases / decreases | 876 / 1,045 |
| Zero / positive fit-admit samples | 1,051 / 937 |
| Pressure samples with running and cap above 64 | 1,689 |
| Rescue samples | 141 |
| Capacity-write counter range | 2–4,994, strictly increasing |
| Fail-closed decisions | 0 |

The original post-run parser stored 1,776 samples, classified 212 as malformed,
and therefore recorded an overall failed validation under that predicate.
Those original files and the failed status remain unchanged. Parser-v2
revalidation binds the full raw log, parser, validator, original summary, and
log summary by path and SHA. It accounts for all 1,989 full-lifecycle markers,
excludes only warmup write count 1, and recovers 1,988 experiment samples.

All 212 former rejections have the one frozen safe shape:

```text
decision=admit reason=forecast_hold rescue=0 admit=0 fit_admit=0
predicted_admit_tokens=0
```

They add no KV exposure when the forecast for existing work is already above
the soft reserve. The raw revalidation reports zero truly malformed samples,
zero fail-closed decisions, exact marker accounting, strictly increasing write
counts, and all per-sample physical-capacity and soft-budget invariants passing.
It is a narrow correction of an old validation predicate, not permission to
ignore arbitrary malformed telemetry.

## Request-tail trade-off

The task objective and individual-request fairness move differently. Relative
to A, C improves mean and median request latency by 25.897% and 45.757%, but
request P95, P99, and maximum are 31.255%, 36.644%, and 39.433% worse. Requests
over 120 s increase from 203 to 255. At the same time, task P95, P99, maximum,
and makespan all improve by 8.469%, 6.252%, 6.177%, and 6.177%.

B→C improves every reported request percentile relative to B, including P95
by 9.204% and P99 by 12.026%, but it does not fully undo the request-tail
trade-off introduced between A and B. Any summary that reports only mean task
latency would therefore be incomplete. A fresh replication should treat
request P95/P99 and counts over 120 s as explicit guardrails rather than
post-hoc diagnostics.

## Tokens and reliability

| Evidence | A | B | C |
|---|---:|---:|---:|
| Successful logical requests | 2,076 | 2,076 | 2,076 |
| Completion tokens | 667,145 | 664,137 | 663,733 |
| Retries / final failures | 0 / 0 | 0 / 0 | 0 / 0 |
| Preemptions | 404 | 317 | 0 |
| CPU-KV swap events | 0 | 0 | 0 |

C generates 3,412 fewer tokens than A (−0.511%) and 404 fewer than B
(−0.061%). The difference is small but nonzero, so this screen must not claim
bit-identical generation or categorically exclude every token-volume effect.
The much larger discontinuity in B→C is the removal of 317 recomputation
preemptions with no swap. All request outcomes are exactly-once successes, so
retry or failure imbalance does not explain the latency differences.

## Claim boundaries

- **One screen, not a replicated result.** A is an accepted natural-queue cell
  reused from an earlier fresh-server run. B and C also contribute only one
  completed cell each. The strict artifacts validate pairing and identity, but
  do not manufacture run-to-run replication.
- **60 independent sources, not 240.** Four deterministic copies per source
  create stress. Bootstrap sampling uses the 60 folded source means.
- **Recorded-wait replay.** Tool waits are recorded waits replayed at 10×
  speedup. No live web/tool service runs, and generated text does not determine
  the recorded next call. This is not the paper's full live system.
- **Bundle versus increment.** A→C is the complete FCFS+none to
  Joint+learned+physical treatment. B→C is the narrower incremental
  native-to-physical comparison. Neither result is a pacing-only claim.
- **Development/load-sensitivity evidence.** stress240 was constructed to
  expose native queue pressure and dynamic admission. It is not an untouched
  final evaluation and should not be selected or reshaped merely to reach a
  desired percentage.
- **Tail qualification.** The positive task-level result coexists with a
  material individual-request tail regression versus A.

## Next step: preregistered fresh replication

The next defensible performance claim should use at least three fresh-server
A/B/C blocks under the already frozen stress240 workload:

1. A = FCFS+none+native, B = Joint+learned+native, and C =
   Joint+learned+physical at the fixed 0.93 target.
2. Keep `max-num-seqs=256` nonbinding and retain the exact engine, request,
   source, calibration, mapper, and configuration guards used here.
3. Randomize or balance cell order within each block and restart the server for
   every cell. Do not retune 0.93 from these repetitions.
4. Preregister mean task E2E as primary, with task P95/makespan and request
   P95/P99/over-120-s counts as simultaneous tail gates.
5. Require exact success/retry/token/preemption/swap accounting and parser-v2
   physical telemetry with no malformed or fail-closed samples in every C
   replicate.
6. Report per-replicate A→B, B→C, and A→C effects, then fold the four load
   copies before source-level uncertainty calculations. Do not pool the 240
   copies as independent observations.

Only after that replication should 25.385% be promoted beyond a strong
one-screen observation. A subsequent frozen stress300 run can test load
sensitivity, but it should be reported separately rather than pooled with
stress240.
