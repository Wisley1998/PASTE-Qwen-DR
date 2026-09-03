# Authority-first Pattern-v2 under low predictability and high load

## Result

The quoted `Top-1 ≈27.8% / hit rate 93.8%` pair is not reproduced under the frozen whole-session grouped-OOF protocol. Exact-URL Top-1 is 21.2% and Top-5 is 56.8%. The nearby 92.8% figure is an evaluation-only candidate-union oracle over 15486 candidates, not a realizable hit rate for a bounded runtime policy.

This experiment replaces per-task candidate allocation with a global non-neural empirical-count confidence table and expected-utility allocator. Authoritative work is dispatch-prioritized, one of the two visit slots is reserved for it, speculative admission is batched, and the utility policy abstains as forecast authoritative pressure rises.

The `safe_global_benefit` row uses a stricter lexicographic policy: it first requires an isolated-capacity certificate and then ranks all visible sessions by expected saved latency. With K=0 it follows the demand-only fast path; with K>0 the original authority worker/tool caps are preserved and a running exact hit races a protected fresh authority backup.

Nested whole-session grouped OOF candidate AP is 0.1424, versus 0.1158 for rank alone; Brier is 0.07087 versus 0.07158. These are development OOF estimates, not a new confirmatory holdout.

The online path is deterministic Pattern-v2 state/feature update plus empirical table lookup: measured Pattern-v2 feature runtime is 0.412 ms/decision on average and 1.022 ms at p99. A cheap fold-trained calls/window gate runs first; when forecast pressure exceeds 2x visit capacity, the utility policy skips candidate generation and probability lookup entirely.

## Observed-label closed-loop burst replay

Shared pool: 4 workers, visit capacity 2, isolated speculative slots 1, service 20.0 ms, lead 10.0 ms. Positive net means lower latency than demand-only after charging pattern feature, confidence lookup, and selection overhead.

| Policy | Offered / realized C | Exact / overlap hit | Wrong starts | Call amp. | Mean authority wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap authority regression mean / p95 | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| safe_global_benefit | 1 / 1 | 16.1% / 16.1% | 1388 | 1.89x | 30.75→28.90 | +0.992 / +0.961 (8/8) | +0.021 / +0.819 | -0.4% |
| safe_global_benefit | 2 / 2 | 10.6% / 10.6% | 787 | 1.52x | 34.16→32.87 | +0.500 / +0.374 (8/8) | +0.082 / +0.868 | -2.2% |
| safe_global_benefit | 4 / 4 | 6.2% / 6.2% | 465 | 1.30x | 39.63→38.75 | +0.124 / +0.313 (6/8) | +0.121 / +1.124 | -4.6% |
| safe_global_benefit | 8 / 8 | 3.6% / 3.6% | 275 | 1.17x | 53.57→52.64 | +0.196 / +0.049 (4/8) | -0.101 / +0.923 | -5.3% |
| safe_global_benefit | 16 / 16 | 2.3% / 2.3% | 154 | 1.09x | 81.39→80.44 | +0.221 / +0.163 (5/8) | -0.211 / +2.190 | -5.6% |
| safe_global_benefit | 32 / 32 | 1.4% / 1.4% | 91 | 1.05x | 130.31→129.06 | +0.525 / +0.724 (6/8) | -0.482 / +2.284 | -5.3% |
| safe_global_benefit | 64 / 64 | 0.9% / 0.9% | 61 | 1.03x | 220.16→219.43 | +0.024 / -1.080 (3/8) | -0.018 / +4.886 | -5.4% |
| safe_global_benefit | 98 / 98 | 0.4% / 0.4% | 56 | 1.03x | 291.42→291.25 | -0.534 / -0.185 (4/8) | -0.026 / +2.675 | -5.7% |

`Exact hit` includes queued promotion; `overlap hit` counts only completed reuse and inflight promotion. Non-overlap regression is measured only on authority targets that obtained no speculative overlap. Admission, deadline, source, p95, and per-repeat data are in `metrics.json`. Repetitions vary scheduling/order only and are not independent accuracy samples.

## Deterministic all-wrong worst case

Every authoritative URL is replaced by a guaranteed non-candidate URL, while gates, scores, ordering, and load stay fixed.

| Policy | Offered / realized C | Selected / wrong-started | Call amp. | Mean wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap regression p95 / max | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|
| safe_global_benefit | 1 / 1 | 2512 / 1690 | 1.90x | 30.65→30.92 | -1.109 / -1.067 (0/8) | +0.843 / +3.566 | -4.5% |
| safe_global_benefit | 2 / 2 | 1357 / 987 | 1.52x | 34.30→34.50 | -0.991 / -1.019 (0/8) | +0.979 / +2.886 | -5.1% |
| safe_global_benefit | 4 / 4 | 692 / 580 | 1.31x | 39.76→40.03 | -1.024 / -1.024 (0/8) | +0.945 / +3.241 | -5.7% |
| safe_global_benefit | 8 / 8 | 363 / 342 | 1.18x | 53.54→53.99 | -1.206 / -1.214 (0/8) | +1.169 / +3.322 | -6.6% |
| safe_global_benefit | 16 / 16 | 203 / 197 | 1.10x | 81.59→81.84 | -0.993 / -1.050 (0/8) | +1.731 / +3.092 | -6.3% |
| safe_global_benefit | 32 / 32 | 122 / 116 | 1.06x | 131.19→131.90 | -1.445 / -1.379 (0/8) | +3.038 / +11.087 | -6.2% |
| safe_global_benefit | 64 / 64 | 83 / 78 | 1.04x | 221.47→223.07 | -2.313 / -2.738 (1/8) | +6.599 / +16.219 | -6.5% |
| safe_global_benefit | 98 / 98 | 64 / 64 | 1.03x | 293.04→293.53 | -1.207 / -1.556 (1/8) | +4.316 / +6.788 | -5.7% |

A cell is a timing-only no-op only when `Selected=0`. An all-wrong cell with positive speculative starts has no latency benefit, but still measures real scheduler/control-plane and cleanup overhead in addition to paired-run timing noise.

## What authority-first can and cannot guarantee

The broker now removes a whole cancellation set atomically before dispatch, so queued siblings cannot start one-by-one during cleanup. Batch admission also removes the prior O(N)-sweep-per-candidate control path, and the per-tool reserve prevents speculation from occupying both visit slots.

With fixed shared capacity and non-preemptible tool calls, strict zero interference is impossible whenever any wrong speculation is running: a future burst can still need every slot. Absolute isolation requires extra/dedicated capacity, genuinely preemptible calls, or abstention. Therefore the intended saturated behavior is graceful fallback to demand-only, not forced positive speculation at every load.

`safe_global_benefit` implements the extra-capacity case explicitly: authority is capped at its original baseline envelope, speculation is capped at K added slots, and K=0 admits nothing. This is a structural worker/tool-capacity guarantee; it excludes unisolated rate limits, predictor CPU, memory bandwidth, and event-loop noise.

This table is a deterministic synthetic-service, closed-loop burst experiment. It diagnoses scheduling and resource allocation; it does not claim production network latency. A sustained/open-loop replay and a new whole-session confirmatory holdout are separate validation requirements.
