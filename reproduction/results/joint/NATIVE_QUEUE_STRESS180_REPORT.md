# Native-admission stress180 exploration report

Date: 2026-08-16

> **Status: natural vLLM queue established; scheduler search still in
> progress.** The 64-sequence ceiling has been removed from this profile. A
> completed FCFS probe proves that requests wait while the configured
> 256-sequence limit is non-binding. The first open A/D pair reduces mean task
> flow by 11.593%, but regresses request maximum and makespan; it is a
> development screen, not the selected result or a replicated performance
> claim. Three completed fairness screens improve the catastrophic request
> maximum but reduce mean task flow by only 6.285%, 4.789%, and 4.449%, while
> regressing important task and request tails; all three are rejected. A fourth
> exact+rescue120 screen reaches a 10.699% mean reduction with nearly neutral
> task P99 and improved makespan. It is the current best balanced task-level
> Pareto candidate, but still fails the strict request-tail gate and is not a
> promoted or replicated result. A separate stress240 one-pair screen raises
> genuine native queue pressure and reaches a 16.339% mean-task reduction, but
> request P95/P99 regress by 45.178%/59.034%; that result is also screening
> evidence rather than a promoted claim.

## What “removing 64” means

The earlier stress180 profile used `max-num-seqs=64` and Joint target/max values
of 64. Although Joint did not lower concurrency below that native setting, 64
was still an explicit experiment ceiling. The native-admission profile changes
the relevant limits as follows:

| Setting | Native-admission value | Why it cannot impose a 64-request queue |
|---|---:|---|
| Maximum simultaneously active traces | 180 | Offered workload concurrency |
| vLLM `max-num-seqs` | 256 | 76 above the workload's concurrency upper bound |
| Joint foreground/decode target/decode max | 256 | Also above the offered upper bound |
| CUDA graph size | 256 | Capture shape only; it is not an admission limit |
| vLLM batched-token budget | 8,192 | A real native serving-resource constraint |

`VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION=1` makes the Joint hook reorder the
waiting list but bypasses its capacity controller. In this mode the hook does
not write `max_num_running_reqs`, allocate or free KV blocks, reserve a private
decode band, or directly preempt a request. Native vLLM remains responsible for
admission using its token budget and physical KV-cache availability. The
implementation and no-cap-write regression tests are in
[`sched_policy_patch.py`](../../../scripts/pythonhooks/sched_policy_patch.py)
and
[`test_joint_scheduler_gate.py`](../../../tests/test_joint_scheduler_gate.py).

There is still a finite native safety/configuration value of 256, as there is
for any real deployment. The important experimental condition is that it is
strictly above both the offered maximum of 180 and every observed running
count. It therefore does not form the measured queue.

The workload itself is unchanged from the fixed stress180 development set: 60
independent source sessions, each represented by one original and two
deterministic `break_prefix` variants, for 180 load instances and 1,557 logical
requests per cell. Its canonical manifest SHA-256 is
`9dc47fe8fe0134cfbbe762165592f5b9852feb959cd1597425fdf230e9a46c91`.
The variants generate load but do not increase the independent sample size
beyond 60.

## Direct proof that vLLM forms the queue

The baseline-only FCFS probe used `max-num-seqs=256`, graph256, GPU-memory
utilization 0.86, and no Joint scheduling treatment. The read-only validator
cross-checks its summary, timeline, request events, metrics, swap sidecar, and
server-log evidence, then fails unless waiting is observed below a structurally
and empirically non-binding sequence cap.

| FCFS natural-queue diagnostic | Observed value |
|---|---:|
| Logical requests finally successful / total | 1,557 / 1,557 |
| Retry count | 0 |
| Timeline samples | 1,064 |
| Peak running requests | 180 / 256 |
| Samples reaching 256 running | 0 |
| Samples with waiting while running < 256 | 777 / 1,064 (73.026%) |
| Mean / maximum waiting requests | 28.328 / 74 |
| Mean / maximum GPU KV usage | 87.351% / 100.000% |
| Mean request latency / queue time | 47.868 s / 10.046 s |
| Queue share of mean request latency | 20.988% |
| Native recompute preemptions | 238 |
| CPU swap events | 0 |

This establishes a **native vLLM resource queue**, not a sequence-count queue.
The server reaches 100% KV usage and performs recompute preemptions, so KV
pressure materially participates. The native 8,192-token scheduling budget is
also finite; these artifacts do not isolate which of token budget versus
physical KV availability is dominant at every step. The accurate attribution
is therefore “vLLM token/KV resource checks,” not “a hidden 64 cap” and not a
single exclusively identified resource.

The immutable probe evidence starts at its
[`summary.json`](../../artifacts/stress180_u86_native256/native256_fcfs_probe_r1/native256_fcfs_probe_r1_fcfs_none/summary.json)
and
[`timeline.json`](../../artifacts/stress180_u86_native256/native256_fcfs_probe_r1/native256_fcfs_probe_r1_fcfs_none/timeline.json).
It can be checked with
[`summarize_natural_queue_probe.py`](../../scripts/summarize_natural_queue_probe.py)
using `--require-natural-queue`.

## First open native256 A/D pair

This pair compares A=`fcfs+none` with D=`Joint+learned` under the same
native256/graph256/u0.86 shape. Its immutable configuration is
[`frozen_config.env`](../../artifacts/stress180_u86_native256_g256/native_open_pair_r1/frozen_config.env),
SHA-256
`2ed7ca0d1df8ab657133bc6710b388952fe446465845f5ca28bb868e63999e33`.
The invariant-checked pair summary is
[`paired_summary.json`](../../artifacts/stress180_u86_native256_g256/native_open_pair_r1/paired_summary.json),
SHA-256
`9b5da4a1cb8ba2344043d1565f9374ceb1f5ef6a02551c272dc3bb1e3b43cbaa`.

Positive reductions below mean D is faster. Negative values are regressions.

| Metric | A: FCFS+none | D: Joint+learned | Reduction |
|---|---:|---:|---:|
| Mean task flow | 436.521 s | 385.913 s | **11.593%** |
| P50 task flow | 463.991 s | 397.769 s | **14.272%** |
| P95 task flow | 566.214 s | 540.605 s | **4.523%** |
| Maximum task flow | 572.034 s | 586.178 s | **-2.473%** |
| Task makespan | 572.389 s | 586.532 s | **-2.471%** |
| Mean request latency | 49.010 s | 43.166 s | **11.924%** |
| P50 request latency | 40.978 s | 31.267 s | **23.698%** |
| P95 request latency | 98.381 s | 95.288 s | **3.143%** |
| Maximum request latency | 114.234 s | 332.307 s | **-190.901%** |
| Mean queue time | 10.288 s | 5.389 s | **47.616%** |

D was faster for 55/60 independent source sessions and 158/180 load
instances. The source-level mean saving was 50.608 s; a fixed-seed 10,000-draw
bootstrap over only the 60 independent source means gives a descriptive 95%
interval of [41.220, 59.408] s. This is one pair over a development workload,
not three replicated pairs and not a new held-out evaluation.

Both cells finally completed 1,557/1,557 logical requests. A had two ambiguous
transport disconnects followed by successful retries; D had none. Thus the
pair has 3,116 attempts for 3,114 logical requests and is not an exactly-once,
retry-symmetric formal result. D generated 0.234% more completion tokens, so
its mean gain was not obtained by generating less text.

Both cells again pass the natural-queue condition: peak running was 180, no
sample reached 256, and all positive-wait samples occurred below 256. A had
215 native recompute preemptions and D had 168, a 21.86% reduction. Neither
cell recorded a CPU KV-swap event. Older summaries used a legacy boolean that
conflated any preemption with “swap”; the validator normalizes that field to
the current CPU-swap-only meaning rather than silently reporting 215 or 168
swap events.

## Where the 11.593% mean gain comes from

The gain is **not only a direct tool-overlap saving**, and in this open-native
pair it is **mostly—but not entirely—queue reduction**. An additive diagnostic
converts per-request means to a per-task basis using 1,557 requests / 180
tasks:

| Approximate contribution to 50.608 s mean task saving | Saving | Share |
|---|---:|---:|
| Measured vLLM queue-time reduction | 42.374 s/task | 83.73% |
| Non-queue request-latency reduction | 8.175 s/task | 16.15% |
| Direct 10x-replayed learned-wait subtraction | 0.058 s/task | 0.11% |

This is accounting, not an isolated causal ablation. It says what changed in
the measured latency components; the A/D treatment still combines scheduler
ordering with learned overlap.

The scheduler uses causal task-progress/final-call and remaining-call metadata
to choose which *waiting* request vLLM sees first. Its score also contains
context/HBM and prefix/session-locality terms. Better ordering can therefore
reduce queue time directly and improve service indirectly by presenting a less
contentious, more prefix-local running mix. Consistent post-hoc signals are:

- mean queue time fell 47.616%;
- native recompute preemptions fell from 215 to 168;
- mean sampled GPU KV usage fell from 86.865% to 77.785%; and
- mean sampled prefix-hit ratio rose from 22.223% to 24.800%.

Those latter three observations are mechanism evidence, not separate causal
effect estimates. There is no kernel-level decode optimization in this
treatment, and native-admission mode performs no Joint capacity pacing. The
learned overlap path subtracts only about 0.058 seconds per task after 10x
replay scaling, so it cannot directly explain a 50.608-second task saving.

## Why mean improves while the tail fails

The open-pair ranking used an exact remaining-call lane: after final calls,
requests with fewer predicted remaining calls are lexicographically preferred.
That completes many tasks earlier, but under native KV pressure it can defer a
small set of early-stage requests for a long time.

The raw-event diagnostic makes this visible. A had no request above 120
seconds. D had 49, including 19 above 240 seconds; all 19 were call index 2
with six scheduled calls remaining. Consequently D still improves task P95,
but its maximum request latency grows to 332.307 s and its final stragglers
extend makespan by 2.471%. This candidate therefore fails a balanced
mean-and-tail objective even though its mean result is positive.

This also explains why the replicated 31.419% target64 result does not carry
over automatically. In the native256 A cell, accumulated mean queue time is
only about 20.4% of mean task flow, so eliminating queue time completely would
still not yield a 30% task improvement at this operating point. Removing the
64 ceiling makes the FCFS baseline substantially less queued and removes much
of the old ordering headroom. That target64/native256 contrast is a
cross-profile diagnostic, not a paired causal estimate. Reaching 30% honestly
now requires either a more strongly but naturally loaded workload or
additional service/locality/tool gains; reintroducing a Joint-only admission
cap would answer a different question. The target64 result remains documented
separately in
[`QUEUED_STRESS180_REPORT.md`](QUEUED_STRESS180_REPORT.md).

## Rejected running-priority diagnostic

An opt-in experiment also reordered the already-running set, with the intent
of serving near-completion tasks first. It was stopped early because recompute
churn accumulated too quickly. The surviving partial artifact contains 860 of
1,557 completed logical requests; its last metrics sample already records 177
native recompute preemptions, 99.69% KV usage, 115 running requests, and 29
waiting requests.

Because it did not full-drain, it has no valid task-E2E or makespan comparison
and must not appear in a performance table as a result. It is retained only as
negative engineering evidence that manipulating the running order/victim
choice under high KV pressure is unsafe in this form. The current native
profile leaves `VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY=0`. The partial evidence
is its
[`request_events.jsonl`](../../artifacts/stress180_u86_native256_g256/running_priority_screen_r1/running_priority_screen_r1_joint_learned/request_events.jsonl)
and
[`metrics_samples.jsonl`](../../artifacts/stress180_u86_native256_g256/running_priority_screen_r1/running_priority_screen_r1_joint_learned/metrics_samples.jsonl).

## Completed waiting-queue candidate screens

Four follow-up D cells preserve native admission and leave running-set order
untouched. The deadline-only candidate lowers the deadline guard from 256 to
48, allowing at most one request older than the existing 40-second threshold
to receive priority. The coarse candidate additionally replaces exact
non-final remaining-call order with near (`rc=1–2`) and far/unknown (`rc>=3`)
buckets, while retaining continuous score and arrival tie-breaks within them.
The soft4 candidate retains the deadline guard but disables both exact and
coarse non-final lanes, replacing them with a 4.0-second soft remaining-call
score weight. Exact+rescue120 restores the exact remaining-call lane and uses
the same 48-running guard, but delays its single expired-waiter rescue until
120 seconds.

All four candidate cells fully drained 1,557/1,557 requests exactly once,
recorded zero CPU swap events, and passed the natural-queue proof: peak running
remained 180/256 and no timeline sample reached the sequence cap. Their strict
configuration guards found exactly the permitted candidate differences and no
others. Deadline generated 0.470% more completion tokens than A and coarse
generated 0.203% more; soft4 generated 0.395% more and rescue120 generated
0.408% more. None of their gains came from producing less text. The
machine-readable comparisons are:

- [`deadline candidate comparison`](../../artifacts/stress180_u86_native256_g256/deadline_screen_r1/candidate_comparison.json),
  SHA-256
  `d52e48800220fd45276e62399c84175ba2e59f39c4747cfaa5915511b0dccfec`;
  candidate configuration SHA-256
  `05fe145bb18b58691e3c1ee11c4ed8b0e3545f29e2aee7c9a3d6345f946902ad`.
- [`coarse candidate comparison`](../../artifacts/stress180_u86_native256_g256/coarse_screen_r1/candidate_comparison.json),
  SHA-256
  `c1d8b7ef3c8276d825d26512f568052a22a726d0c421103fdcd89698ec3c1ce5`;
  candidate configuration SHA-256
  `fa413b3e516aead16b6c134d49710507301d784ec2581decd4ea3582283ab632`.
- [`soft4 candidate comparison`](../../artifacts/stress180_u86_native256_g256_soft4/soft4_screen_r1/candidate_comparison.json),
  SHA-256
  `2eec21f9273ff692fe7e9b041030dbd18cf03a63a22c57e31b3b890d9652ced8`;
  candidate configuration SHA-256
  `e0132902ca9f19f8725192eaa19b1136ce03ea5f6a45d4bf46e79c4db5504a5e`.
- [`exact+rescue120 candidate comparison`](../../artifacts/stress180_u86_native256_g256_exact_rescue120/rescue120_screen_r1/candidate_comparison.json),
  SHA-256
  `93832a7ed706e27e5396b7b3909cab3f57a8506472c367a187c5402ac347a418`;
  candidate configuration SHA-256
  `51f98c40ef816969becca8be9808cd1f1093ee71bf898fe1b1c97d0949cfd637`.

The percentages below compare each completed candidate D with the same reused
open-pair A. Positive is an improvement; negative is a regression.

| Metric | Reused A | Deadline D (vs A) | Coarse D (vs A) | Soft4 D (vs A) | Rescue120 D (vs A) |
|---|---:|---:|---:|---:|---:|
| Mean task flow | 436.521 s | 409.086 s (**6.285%**) | 415.617 s (**4.789%**) | 417.098 s (**4.449%**) | 389.817 s (**10.699%**) |
| P50 task flow | 463.991 s | 434.992 s (**6.250%**) | 445.413 s (**4.004%**) | 448.248 s (**3.393%**) | 408.145 s (**12.036%**) |
| P95 task flow | 566.214 s | 571.389 s (**-0.914%**) | 581.570 s (**-2.712%**) | 584.320 s (**-3.198%**) | 550.393 s (**2.794%**) |
| P99 task flow | 568.772 s | 578.301 s (**-1.675%**) | 587.906 s (**-3.364%**) | 590.348 s (**-3.793%**) | 569.619 s (**-0.149%**) |
| Task makespan | 572.389 s | 580.029 s (**-1.335%**) | 591.723 s (**-3.378%**) | 591.909 s (**-3.410%**) | 571.161 s (**0.214%**) |
| Mean request latency | 49.010 s | 45.844 s (**6.459%**) | 46.599 s (**4.918%**) | 46.771 s (**4.569%**) | 43.617 s (**11.003%**) |
| P95 request latency | 98.381 s | 117.842 s (**-19.782%**) | 114.238 s (**-16.119%**) | 116.916 s (**-18.840%**) | 105.444 s (**-7.179%**) |
| P99 request latency | 109.625 s | 134.005 s (**-22.239%**) | 122.526 s (**-11.768%**) | 127.588 s (**-16.385%**) | 194.055 s (**-77.017%**) |
| Maximum request latency | 114.234 s | 135.653 s (**-18.750%**) | 136.268 s (**-19.289%**) | 131.786 s (**-15.365%**) | 199.478 s (**-74.622%**) |
| Mean queue time | 10.288 s | 6.327 s (**38.499%**) | 6.571 s (**36.125%**) | 6.471 s (**37.097%**) | 5.683 s (**44.761%**) |
| Mean non-queue request time | 38.722 s | 39.517 s (**-2.054%**) | 40.028 s (**-3.373%**) | 40.299 s (**-4.074%**) | 37.934 s (**2.034%**) |
| Native recompute preemptions | 215 | 219 (**-1.860%**) | 222 (**-3.256%**) | 204 (**5.116%**) | 163 (**24.186%**) |

The deadline candidate was faster for 43/60 independent source sessions; its
mean source saving was 27.435 s with a descriptive bootstrap 95% interval of
[18.498, 36.445] s. The coarse candidate was faster for only 33/60 sources;
its mean source saving was 20.904 s with interval [11.297, 30.687] s. Soft4
was faster for 39/60 sources; its mean saving was 19.423 s with interval
[10.077, 28.943] s. Rescue120 was faster for 54/60 sources; its mean saving
was 46.704 s with interval [37.741, 55.623] s. The positive intervals show a
mean effect on this reused development set, not that any candidate passes the
full acceptance gate.

All four candidates eliminate requests above 240 seconds and keep maximum
request latency below 200 seconds, versus 332.307 seconds for the exact-lane
reference D. They do so by spreading delay more broadly rather than beating
FCFS tails: deadline has 62 requests above 120 seconds, coarse has 39, soft4
has 44, and rescue120 has 63, while reused A has none. The deadline candidate's
27.435-second task saving is approximately +34.260 s from queue reduction,
+0.058 s from replayed learned wait, and **-6.879 s** from a non-queue
regression. For coarse, the corresponding values are +32.148 s, +0.058 s, and
**-11.298 s**; for soft4 they are +33.012 s, +0.058 s, and **-13.645 s**.
Queue improvement is therefore partially consumed by worse service/non-queue
time in those three variants. Soft4 shows that lowering preemptions alone is
not sufficient when non-queue service time deteriorates more sharply.

Rescue120 is different: its 46.704-second task saving decomposes into
+39.832 s of queue reduction, +6.813 s of non-queue improvement, and +0.058 s
of replayed learned wait. It strictly dominates deadline, coarse, and soft4 on
all five reported task metrics: mean, P50, P95, P99, and makespan. Against the
original exact-lane D it gives up only 0.894 percentage points of mean-task
reduction (11.593% to 10.699%), while shrinking the task-P99 regression from
2.677% to 0.149% and changing makespan from a 2.471% regression to a 0.214%
improvement. It is therefore the **current best balanced task-level Pareto
candidate** among the completed native256 screens.

The deadline candidate is rejected because its mean gain falls to 6.285%, task
P99 and makespan regress by 1.675% and 1.335%, request P95/P99 regress by
19.782%/22.239%, and preemptions rise rather than fall. Coarse is rejected more
strongly: mean gain is only 4.789%, task P95/P99/makespan regress by
2.712%/3.364%/3.378%, request tails remain worse than A, and preemptions rise
to 222. Soft4 is also rejected: despite lowering preemptions to 204, mean gain
falls to 4.449%, task P95/P99/makespan regress by 3.198%/3.793%/3.410%, and
request P95/P99 regress by 18.840%/16.385%.

Rescue120 is retained as the task-Pareto best, but it still does **not** pass
the strict end-to-end acceptance gate. Request P95 regresses by 7.179%, P99 by
77.017%, maximum by 74.622%, and 63 requests exceed 120 seconds. Its zero
requests above 240 seconds, 163 preemptions, zero retries, and zero CPU swaps
are genuine improvements over the exact-lane failure mode, but they do not
erase the remaining request-tail violation. It is not promoted to a formal
result or fresh-server replication on this screen alone.

These are strict, completed **candidate comparisons**, but not fresh A/D
pairs. All four reuse the earlier A cell, which had two ambiguous successful
retries, while each candidate D had zero retries. Reusing A is appropriate for
tuning screens with identical deterministic workload identity; it does not
support a formal paired-replicate claim or remove run-to-run server variance.

The evolving profile entry point is
[`joint_stress180_u86_native256.env.example`](../../configs/joint_stress180_u86_native256.env.example),
and the strict wrapper is
[`run_joint_stress_pair.sh`](../../scripts/run_joint_stress_pair.sh). The
immutable `frozen_config.env` linked above, not the evolving example, is the
configuration of the reported open pair.

## Stress240 one-pair screening result

> **Status: strict one-pair screening; not a fresh-server paired replicate.**
> The stress240 result strengthens the natural-load signal, but it still fails
> the request-tail objective and must not be presented as a formal replicated
> result.

### Baseline-only load selection and strict comparison

Stress240 uses the same 60 independent source sessions with four deterministic
instances per source: 240 load instances and 2,076 logical requests per cell.
The copies create load but do not increase the independent sample size beyond
60. The canonical fixed-manifest SHA-256 is
`9c2c190aaecea4570de23adc81a7ed56469c8d53776c2837fa383c323c1bfcd6`;
the manifest is
[`manifest_stress240.json`](../../artifacts/workloads/fixed_three_way_cap512_floor64/manifest_stress240.json).

The FCFS A-only probe was run and accepted before D was observed. Its frozen
gates required at least 50% waiting-below-cap samples, at least a 20% request
queue share, no CPU swap, and no more than 0.25 recompute preemptions per
logical request. A passed with 87.299%, 40.654%, zero swap, and 0.1946
preemptions/request. This avoids selecting the offered load because D happened
to look favorable.

The completed strict comparison is
[`strict_a_vs_d_screening.json`](../../artifacts/stress240_u86_native256_g256_exact_rescue120/stress240_d_screen_r1/stress240_d_screen_r1_joint_learned/strict_a_vs_d_screening.json),
SHA-256
`af7b7bef7c037af409ae8560ec3323f26e50af867a9b09b8be43cbee8b6f46b7`.
It proves exact request identity and source mapping, validates all 14 required
engine-shape keys as identical, and finds exactly the five allowed
configuration differences—profile/config hashes plus the three expected
exact-rescue policy fields—with no other drift. A's frozen configuration is
[`frozen A config`](../../artifacts/stress240_u86_native256_g256_a_probe/stress240_a_probe_r1/frozen_config.env),
SHA-256
`1bd0073e891083efeed328c5d0e925832772d73ee25946f0a18f80d504989102`;
D's is
[`frozen D config`](../../artifacts/stress240_u86_native256_g256_exact_rescue120/stress240_d_screen_r1/frozen_config.env),
SHA-256
`d0f9f486f9cdd14aa3fd970086682b31220b4666b03af91cc77a75951d1065b0`.

The expected policy-field differences are inactive under A's FCFS policy but
material under D: D uses the exact remaining-call lane, disables coarse/soft
lanes, and rescues at most one expired waiter after 120 seconds. The custom
strict screening schema validates that design explicitly; the ordinary paired
summarizer's shared-scheduler-configuration invariant is intentionally not
relaxed or misreported as passing.

### Non-binding sequence cap and native queue proof

Both cells used `max-num-seqs=256`, while offered concurrency and observed peak
running were 240. The 16-request headroom is small but strict: no timeline
sample reached 256, and every positive-wait sample occurred below 256 in both
cells. Thus the heavier queue is still formed by vLLM's token/KV resource
checks, not by a 64- or 240-request experiment ceiling.

| Natural-queue diagnostic | A: FCFS+none | D: exact+rescue120+learned |
|---|---:|---:|
| Peak running / configured maximum | 240 / 256 | 240 / 256 |
| Waiting below cap samples | 1,306 / 1,496 (87.299%) | 1,125 / 1,493 (75.352%) |
| Mean / maximum waiting | 76.129 / 138 | 44.331 / 113 |
| Mean / maximum GPU KV usage | 92.369% / 100.000% | 88.943% / 99.996% |
| Request queue share | 40.654% | 27.960% |
| Recompute preemptions | 404 | 304 |
| CPU KV-swap events | 0 | 0 |

The committed A-side proof is
[`natural_queue_probe.json`](../../artifacts/stress240_u86_native256_g256_a_probe/stress240_a_probe_r1/natural_queue_probe.json);
the strict comparison independently recomputes the proof for both A and D.

### Task gain and request-tail tradeoff

Positive reductions below mean D is lower/faster; negative values are
regressions.

| Metric | A: FCFS+none | D: exact+rescue120+learned | Reduction |
|---|---:|---:|---:|
| Mean task flow | 627.575 s | 525.034 s | **16.339%** |
| P50 task flow | 654.055 s | 555.317 s | **15.096%** |
| P95 task flow | 787.281 s | 782.487 s | **0.609%** |
| P99 task flow | 792.046 s | 790.784 s | **0.159%** |
| Maximum task flow | 792.816 s | 791.895 s | **0.116%** |
| Task makespan | 793.160 s | 792.271 s | **0.112%** |
| Mean request latency | 71.099 s | 59.250 s | **16.666%** |
| P50 request latency | 69.807 s | 38.601 s | **44.703%** |
| P95 request latency | 129.310 s | 187.731 s | **-45.178%** |
| P99 request latency | 135.806 s | 215.978 s | **-59.034%** |
| Maximum request latency | 140.597 s | 224.236 s | **-59.489%** |
| Mean queue time | 28.905 s | 16.566 s | **42.686%** |
| Mean non-queue request time | 42.194 s | 42.683 s | **-1.158%** |

D was faster for 53/60 independent sources and 204/240 deterministic load
instances. Mean source saving was 102.542 s. A fixed-seed 10,000-resample
bootstrap over the 60 source means gives a descriptive 95% interval of
[79.854, 125.343] s. This interval reflects paired variation over this fixed
development set; four copies per source remain one independent source.

The 102.542-second mean-task saving decomposes into **+106.728 s from measured
queue reduction**, **-4.228 s from worse non-queue request time**, +0.058 s
from direct replayed learned-wait subtraction, and -0.016 s residual. Queue
reduction is 104.08% of the net saving because the non-queue component offsets
4.12%. The direct overlap subtraction is only 0.056% of the total. Consistent
post-hoc signals are a 24.752% reduction in recompute preemptions, a 41.769%
reduction in mean sampled waiting, a 3.425-percentage-point reduction in mean
KV usage, and a small prefix-hit increase from 19.472% to 20.511%. These are
mechanism diagnostics, not isolated causal ablations.

The near-zero makespan change alongside the 16.339% mean reduction shows that
D primarily **front-loads task completion** rather than raising full-drain
throughput. A raw completion-curve diagnostic finds 16 versus 72 tasks complete
by 400 seconds, 91 versus 137 by 600 seconds, and 148 versus 190 by 700 seconds
for A versus D; both finally drain near 793 seconds.

The request tail shows where that front-loading comes from. Requests above 120
seconds increase from 203 to 261, although neither cell has a request above 240
seconds. In D, all 117 requests above 180 seconds occur at early call indices:
56 at call 2, 57 at call 3, and 4 at call 4. Meanwhile calls 5–8 become much
faster. This is consistent with exact remaining-call ordering deferring early
stages to finish nearer-completion tasks, while rescue120 bounds the extreme
starvation observed in the stress180 exact-lane screen. This call-index result
is a post-hoc diagnostic from the immutable
[`A request events`](../../artifacts/stress240_u86_native256_g256_a_probe/stress240_a_probe_r1/stress240_a_probe_r1_fcfs_none/request_events.jsonl)
and
[`D request events`](../../artifacts/stress240_u86_native256_g256_exact_rescue120/stress240_d_screen_r1/stress240_d_screen_r1_joint_learned/request_events.jsonl),
not an additional primary endpoint.

### Integrity and claim boundary

All 4,152 logical requests finally succeeded. A completed exactly once; D had
one approximately 1.4 ms delivery-ambiguous connection-write failure, waited
one second, and succeeded on its explicit retry. The comparison therefore has
4,153 attempts and is not exactly-once or retry-symmetric. Both cells record
zero CPU swap. D has 665,447 completion tokens versus A's 667,145, a 0.255%
decrease. That small output-volume difference is disclosed rather than called
zero; it is far smaller than the 16.339% task-mean effect but prevents claiming
perfect output-volume equality.

Stress240 is a **single A-reused screening comparison**. A was a previously
accepted fresh-server load probe and D was run later; this is not a matched
fresh-server replicate and contains run-to-run variance. It also reuses the
same 60 development sources as stress180, with deterministic fourth copies,
and still replays recorded tool waits at 10x rather than executing live tools.
The result is consistent with stronger, naturally formed queue pressure
exposing more scheduling headroom: baseline accumulated queue time is 39.84%
of mean task flow here, versus about 20.4% in stress180 native256. That
cross-profile contrast is descriptive, not a causal load ablation. Stress240
does **not** support a 16.339% formal claim until fresh-server replication, and
the 45.178%/59.034% request-P95/P99 regressions fail the current strict tail
objective in any case.

## Scope and claim boundaries

- The stress180 screens use 60 independent source sessions with deterministic
  triplication, not 180 independent samples. Stress240 uses the same 60
  sources with four instances each, not 240 independent samples.
- Recorded waits are replayed as sleeps at 10x speedup. There is no live tool,
  shared tool-side queue, or response-dependent future tool trajectory.
- The open A/D pair changes both scheduler (`fcfs` to Joint) and overlap mode
  (`none` to learned); it does not isolate either component.
- The deadline, coarse, soft4, and rescue120 screens reuse the prior A rather
  than starting a new matched A server. Their intervals are descriptive tuning
  evidence only.
- The stress240 screen also reuses a previously accepted A-only load probe;
  despite stricter identity/configuration validation, it is not a matched
  fresh-server pair or a replicated result.
- “Current task-level Pareto candidate” compares completed native256 screens;
  it is not an acceptance claim. Rescue120 still fails the strict request-tail
  gate.
- Native-admission proves that the sequence cap is non-binding, but does not
  make physical KV capacity or the 8,192-token batch budget disappear.
- The 11.593% number is a one-pair screen with asymmetric retries and a clear
  tail regression. It must not replace the repository's replicated result
  unless a frozen candidate later passes formal replication.
