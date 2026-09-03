# Authority-first Pattern-v2 under low predictability and high load

## Result

The quoted `Top-1 ≈27.8% / hit rate 93.8%` pair is not reproduced under the frozen whole-session grouped-OOF protocol. Exact-URL Top-1 is 21.2% and Top-5 is 56.8%. The nearby 92.8% figure is an evaluation-only candidate-union oracle over 15486 candidates, not a realizable hit rate for a bounded runtime policy.

This experiment replaces per-task candidate allocation with a global non-neural empirical-count confidence table and expected-utility allocator. Authoritative work is dispatch-prioritized, one of the two visit slots is reserved for it, speculative admission is batched, and the utility policy abstains as forecast authoritative pressure rises.

The `safe_global_benefit` row uses a stricter lexicographic policy: it first requires an isolated-capacity certificate and then ranks all visible sessions by expected saved latency. With K=0 it follows the demand-only fast path; with K>0 the original authority worker/tool caps are preserved and a running exact hit races a protected fresh authority backup.

Nested whole-session grouped OOF candidate AP is 0.1424, versus 0.1158 for rank alone; Brier is 0.07087 versus 0.07158. These are development OOF estimates, not a new confirmatory holdout.

The online path is deterministic Pattern-v2 state/feature update plus empirical table lookup: measured Pattern-v2 feature runtime is 0.442 ms/decision on average and 1.110 ms at p99. A cheap fold-trained calls/window gate runs first; when forecast pressure exceeds 2x visit capacity, the utility policy skips candidate generation and probability lookup entirely.

## Observed-label closed-loop burst replay

Shared pool: 4 workers, visit capacity 2, isolated speculative slots 0, service 20.0 ms, lead 10.0 ms. Positive net means lower latency than demand-only after charging pattern feature, confidence lookup, and selection overhead.

| Policy | Offered / realized C | Exact / overlap hit | Wrong starts | Call amp. | Mean authority wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap authority regression mean / p95 | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| safe_global_benefit | 1 / 1 | 0.0% / 0.0% | 0 | 1.00x | 30.85→30.75 | +0.104 / +0.047 (4/4) | -0.104 / +0.240 | +0.3% |
| safe_global_benefit | 8 / 8 | 0.0% / 0.0% | 0 | 1.00x | 52.16→52.24 | -0.075 / -0.095 (1/4) | +0.075 / +0.792 | -0.3% |
| safe_global_benefit | 32 / 32 | 0.0% / 0.0% | 0 | 1.00x | 133.31→133.08 | +0.233 / +0.151 (4/4) | -0.233 / +1.411 | +0.2% |
| safe_global_benefit | 98 / 98 | 0.0% / 0.0% | 0 | 1.00x | 291.59→291.25 | +0.333 / +0.545 (2/4) | -0.333 / +7.419 | +0.2% |

`Exact hit` includes queued promotion; `overlap hit` counts only completed reuse and inflight promotion. Non-overlap regression is measured only on authority targets that obtained no speculative overlap. Admission, deadline, source, p95, and per-repeat data are in `metrics.json`. Repetitions vary scheduling/order only and are not independent accuracy samples.

## Deterministic all-wrong worst case

Every authoritative URL is replaced by a guaranteed non-candidate URL, while gates, scores, ordering, and load stay fixed.

| Policy | Offered / realized C | Selected / wrong-started | Call amp. | Mean wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap regression p95 / max | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|
| safe_global_benefit | 1 / 1 | 0 / 0 | 1.00x | 30.72→30.81 | -0.091 / +0.000 (2/4) | +0.349 / +20.318 | -0.1% |
| safe_global_benefit | 8 / 8 | 0 / 0 | 1.00x | 52.23→52.25 | -0.020 / -0.040 (1/4) | +0.754 / +1.325 | -0.0% |
| safe_global_benefit | 32 / 32 | 0 / 0 | 1.00x | 133.43→133.24 | +0.198 / +0.077 (2/4) | +1.755 / +3.760 | +0.1% |
| safe_global_benefit | 98 / 98 | 0 / 0 | 1.00x | 292.72→292.20 | +0.526 / +0.126 (2/4) | +4.004 / +5.258 | +0.2% |

If `Selected=0`, any nonzero paired latency difference is measurement noise from separate asyncio runs. In particular, the all-wrong C=128 pooled value is not a causal effect.

## What authority-first can and cannot guarantee

The broker now removes a whole cancellation set atomically before dispatch, so queued siblings cannot start one-by-one during cleanup. Batch admission also removes the prior O(N)-sweep-per-candidate control path, and the per-tool reserve prevents speculation from occupying both visit slots.

With fixed shared capacity and non-preemptible tool calls, strict zero interference is impossible whenever any wrong speculation is running: a future burst can still need every slot. Absolute isolation requires extra/dedicated capacity, genuinely preemptible calls, or abstention. Therefore the intended saturated behavior is graceful fallback to demand-only, not forced positive speculation at every load.

`safe_global_benefit` implements the extra-capacity case explicitly: authority is capped at its original baseline envelope, speculation is capped at K added slots, and K=0 admits nothing. This is a structural worker/tool-capacity guarantee; it excludes unisolated rate limits, predictor CPU, memory bandwidth, and event-loop noise.

This table is a deterministic synthetic-service, closed-loop burst experiment. It diagnoses scheduling and resource allocation; it does not claim production network latency. A sustained/open-loop replay and a new whole-session confirmatory holdout are separate validation requirements.
