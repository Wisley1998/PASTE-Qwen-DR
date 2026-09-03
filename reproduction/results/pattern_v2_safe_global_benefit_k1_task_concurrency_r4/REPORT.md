# Authority-first Pattern-v2 under low predictability and high load

## Result

The quoted `Top-1 ≈27.8% / hit rate 93.8%` pair is not reproduced under the frozen whole-session grouped-OOF protocol. Exact-URL Top-1 is 21.2% and Top-5 is 56.8%. The nearby 92.8% figure is an evaluation-only candidate-union oracle over 15486 candidates, not a realizable hit rate for a bounded runtime policy.

This experiment replaces per-task candidate allocation with a global non-neural empirical-count confidence table and expected-utility allocator. Authoritative work is dispatch-prioritized, one of the two visit slots is reserved for it, speculative admission is batched, and the utility policy abstains as forecast authoritative pressure rises.

The `safe_global_benefit` row uses a stricter lexicographic policy: it first requires an isolated-capacity certificate and then ranks all visible sessions by expected saved latency. With K=0 it follows the demand-only fast path; with K>0 the original authority worker/tool caps are preserved and a running exact hit races a protected fresh authority backup.

Nested whole-session grouped OOF candidate AP is 0.1424, versus 0.1158 for rank alone; Brier is 0.07087 versus 0.07158. These are development OOF estimates, not a new confirmatory holdout.

The online path is deterministic Pattern-v2 state/feature update plus empirical table lookup: measured Pattern-v2 feature runtime is 0.418 ms/decision on average and 1.085 ms at p99. A cheap fold-trained calls/window gate runs first; when forecast pressure exceeds 2x visit capacity, the utility policy skips candidate generation and probability lookup entirely.

## Observed-label closed-loop burst replay

Shared pool: 4 workers, visit capacity 2, isolated speculative slots 1, service 20.0 ms, lead 10.0 ms. Positive net means lower latency than demand-only after charging pattern feature, confidence lookup, and selection overhead.

| Policy | Offered / realized C | Exact / overlap hit | Wrong starts | Call amp. | Mean authority wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap authority regression mean / p95 | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| safe_global_benefit | 1 / 1 | 16.2% / 16.2% | 693 | 1.89x | 30.83→28.90 | +1.061 / +1.069 (4/4) | -0.067 / +0.823 | -0.3% |
| safe_global_benefit | 2 / 2 | 11.1% / 11.1% | 390 | 1.52x | 34.57→33.31 | +0.445 / +0.376 (4/4) | +0.106 / +0.839 | -2.3% |
| safe_global_benefit | 4 / 4 | 6.8% / 6.8% | 228 | 1.30x | 39.55→38.68 | +0.098 / -0.009 (2/4) | +0.258 / +0.946 | -4.7% |
| safe_global_benefit | 8 / 8 | 3.8% / 3.8% | 138 | 1.18x | 52.16→51.24 | +0.185 / +0.173 (2/4) | -0.074 / +1.154 | -5.6% |
| safe_global_benefit | 16 / 16 | 2.3% / 2.3% | 76 | 1.09x | 81.02→79.57 | +0.715 / +0.645 (2/4) | -0.551 / +2.078 | -5.6% |
| safe_global_benefit | 32 / 32 | 1.5% / 1.5% | 44 | 1.05x | 133.89→132.34 | +0.822 / +1.447 (3/4) | -0.559 / +2.426 | -5.4% |
| safe_global_benefit | 64 / 64 | 1.0% / 1.0% | 29 | 1.03x | 222.19→221.97 | -0.516 / -0.761 (2/4) | +0.422 / +4.353 | -5.7% |
| safe_global_benefit | 98 / 98 | 0.4% / 0.4% | 28 | 1.03x | 293.53→292.24 | +0.571 / +0.833 (2/4) | -1.139 / +3.156 | -5.6% |

`Exact hit` includes queued promotion; `overlap hit` counts only completed reuse and inflight promotion. Non-overlap regression is measured only on authority targets that obtained no speculative overlap. Admission, deadline, source, p95, and per-repeat data are in `metrics.json`. Repetitions vary scheduling/order only and are not independent accuracy samples.

## Deterministic all-wrong worst case

Every authoritative URL is replaced by a guaranteed non-candidate URL, while gates, scores, ordering, and load stay fixed.

| Policy | Offered / realized C | Selected / wrong-started | Call amp. | Mean wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap regression p95 / max | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|
| safe_global_benefit | 1 / 1 | 1256 / 845 | 1.90x | 30.80→31.04 | -1.103 / -1.075 (0/4) | +0.864 / +3.898 | -4.3% |
| safe_global_benefit | 2 / 2 | 678 / 493 | 1.52x | 34.71→34.92 | -1.011 / -1.029 (0/4) | +1.347 / +3.835 | -5.0% |
| safe_global_benefit | 4 / 4 | 346 / 292 | 1.31x | 39.81→39.96 | -0.905 / -0.906 (0/4) | +0.819 / +3.155 | -5.4% |
| safe_global_benefit | 8 / 8 | 182 / 174 | 1.19x | 52.48→52.85 | -1.109 / -1.125 (0/4) | +1.300 / +20.390 | -6.3% |
| safe_global_benefit | 16 / 16 | 103 / 98 | 1.10x | 81.34→81.87 | -1.270 / -1.229 (0/4) | +2.004 / +2.856 | -6.6% |
| safe_global_benefit | 32 / 32 | 61 / 58 | 1.06x | 134.26→135.63 | -2.098 / -1.202 (0/4) | +6.243 / +27.228 | -6.3% |
| safe_global_benefit | 64 / 64 | 42 / 40 | 1.04x | 224.49→225.17 | -1.411 / -1.764 (1/4) | +2.994 / +5.023 | -6.2% |
| safe_global_benefit | 98 / 98 | 32 / 32 | 1.03x | 293.64→296.06 | -3.137 / -1.935 (0/4) | +20.526 / +25.256 | -6.3% |

If `Selected=0`, any nonzero paired latency difference is measurement noise from separate asyncio runs. In particular, the all-wrong C=128 pooled value is not a causal effect.

## What authority-first can and cannot guarantee

The broker now removes a whole cancellation set atomically before dispatch, so queued siblings cannot start one-by-one during cleanup. Batch admission also removes the prior O(N)-sweep-per-candidate control path, and the per-tool reserve prevents speculation from occupying both visit slots.

With fixed shared capacity and non-preemptible tool calls, strict zero interference is impossible whenever any wrong speculation is running: a future burst can still need every slot. Absolute isolation requires extra/dedicated capacity, genuinely preemptible calls, or abstention. Therefore the intended saturated behavior is graceful fallback to demand-only, not forced positive speculation at every load.

`safe_global_benefit` implements the extra-capacity case explicitly: authority is capped at its original baseline envelope, speculation is capped at K added slots, and K=0 admits nothing. This is a structural worker/tool-capacity guarantee; it excludes unisolated rate limits, predictor CPU, memory bandwidth, and event-loop noise.

This table is a deterministic synthetic-service, closed-loop burst experiment. It diagnoses scheduling and resource allocation; it does not claim production network latency. A sustained/open-loop replay and a new whole-session confirmatory holdout are separate validation requirements.
