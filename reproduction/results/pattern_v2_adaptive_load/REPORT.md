# Authority-first Pattern-v2 under low predictability and high load

## Result

The quoted `Top-1 ≈27.8% / hit rate 93.8%` pair is not reproduced under the frozen whole-session grouped-OOF protocol. Exact-URL Top-1 is 21.2% and Top-5 is 56.8%. The nearby 92.8% figure is an evaluation-only candidate-union oracle over 15486 candidates, not a realizable hit rate for a bounded runtime policy.

This experiment replaces per-task candidate allocation with a global non-neural empirical-count confidence table and expected-utility allocator. Authoritative work is dispatch-prioritized, one of the two visit slots is reserved for it, speculative admission is batched, and the utility policy abstains as forecast authoritative pressure rises.

Nested whole-session grouped OOF candidate AP is 0.1424, versus 0.1158 for rank alone; Brier is 0.07087 versus 0.07158. These are development OOF estimates, not a new confirmatory holdout.

The online path is deterministic Pattern-v2 state/feature update plus empirical table lookup: measured Pattern-v2 feature runtime is 0.383 ms/decision on average and 0.782 ms at p99. A cheap fold-trained calls/window gate runs first; when forecast pressure exceeds 2x visit capacity, the utility policy skips candidate generation and probability lookup entirely.

For the risk-limited utility allocator, conservative net is -0.461 ms/target at concurrency 1 and +0.109 ms/target at concurrency 128; the latter selected zero candidates and is timing noise around a demand-only fallback, not a speedup.

No C>=8 utility cell met the repeat-stability positive rule (found 0). A pooled positive estimate exists at C=32, but its paired repetitions change sign. The defensible high-load result is bounded harm and graceful abstention, not demonstrated positive latency benefit.

## Observed-label closed-loop burst replay

Shared pool: 4 workers, visit capacity 2, service 5.0 ms, lead 2.5 ms. Positive net means lower latency than demand-only after charging pattern feature, confidence lookup, and selection overhead.

| Policy | Offered / realized C | Exact / overlap hit | Wrong starts | Call amp. | Mean authority wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap authority regression mean / p95 | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rank5_sequential_unreserved | 1 / 1 | 35.7% / 35.7% | 2176 | 3.31x | 7.96→7.20 | +0.196 / +0.189 (4/4) | +0.776 / +3.257 | -33.2% |
| rank_budgeted_round_robin_reserved | 1 / 1 | 21.3% / 21.3% | 1056 | 2.12x | 7.97→7.28 | +0.107 / +0.131 (4/4) | +0.044 / +2.406 | -27.1% |
| confidence_global_reserved | 1 / 1 | 18.3% / 18.3% | 832 | 1.89x | 7.96→7.35 | +0.019 / +0.020 (4/4) | +0.023 / +2.395 | -22.2% |
| utility_global_risk_limited | 1 / 1 | 4.3% / 4.3% | 140 | 1.15x | 8.01→7.87 | -0.461 / -0.499 (0/4) | -0.014 / +0.228 | -9.6% |
| rank5_sequential_unreserved | 8 / 8 | 6.6% / 6.6% | 302 | 1.32x | 13.70→17.52 | -4.377 / -4.456 (0/4) | +4.134 / +5.622 | -37.7% |
| rank_budgeted_round_robin_reserved | 8 / 8 | 2.6% / 2.6% | 158 | 1.17x | 13.72→14.27 | -1.124 / -1.113 (0/4) | +0.844 / +2.500 | -22.3% |
| confidence_global_reserved | 8 / 8 | 3.9% / 3.9% | 145 | 1.15x | 13.70→14.13 | -1.014 / -0.967 (0/4) | +0.722 / +2.537 | -21.7% |
| utility_global_risk_limited | 8 / 8 | 0.1% / 0.1% | 4 | 1.00x | 13.69→13.68 | -0.005 / -0.013 (1/4) | -0.010 / +0.135 | -0.0% |
| rank5_sequential_unreserved | 32 / 32 | 2.1% / 2.1% | 102 | 1.11x | 36.10→39.51 | -3.966 / -4.093 (0/4) | +3.506 / +6.048 | -31.3% |
| rank_budgeted_round_robin_reserved | 32 / 32 | 1.2% / 1.2% | 50 | 1.05x | 36.13→35.95 | -0.393 / -0.365 (0/4) | +0.082 / +1.346 | -19.3% |
| confidence_global_reserved | 32 / 32 | 1.5% / 1.5% | 47 | 1.05x | 36.16→35.82 | -0.245 / -0.249 (1/4) | -0.063 / +1.234 | -19.4% |
| utility_global_risk_limited | 32 / 32 | 0.1% / 0.1% | 4 | 1.00x | 36.48→36.24 | +0.228 / -0.174 (1/4) | -0.234 / +0.492 | -0.3% |
| rank5_sequential_unreserved | 64 / 64 | 1.7% / 1.7% | 66 | 1.07x | 62.66→64.46 | -2.358 / -2.458 (0/4) | +1.873 / +5.996 | -28.0% |
| rank_budgeted_round_robin_reserved | 64 / 64 | 0.2% / 0.2% | 40 | 1.04x | 62.10→62.53 | -1.002 / -0.999 (0/4) | +0.479 / +1.141 | -19.2% |
| confidence_global_reserved | 64 / 64 | 1.0% / 1.0% | 33 | 1.04x | 62.09→62.34 | -0.829 / -0.951 (0/4) | +0.427 / +1.092 | -19.7% |
| utility_global_risk_limited | 64 / 64 | 0.2% / 0.2% | 3 | 1.00x | 62.10→62.55 | -0.467 / -0.162 (1/4) | +0.466 / +0.704 | -1.2% |
| rank5_sequential_unreserved | 128 / 98 | 1.5% / 1.5% | 42 | 1.04x | 83.82→86.21 | -2.954 / -2.952 (0/4) | +2.451 / +6.510 | -25.4% |
| rank_budgeted_round_robin_reserved | 128 / 98 | 0.0% / 0.0% | 32 | 1.03x | 83.90→85.00 | -1.668 / -1.037 (0/4) | +1.100 / +1.816 | -19.6% |
| confidence_global_reserved | 128 / 98 | 0.4% / 0.4% | 28 | 1.03x | 83.94→84.41 | -1.053 / -1.056 (0/4) | +0.520 / +1.131 | -19.5% |
| utility_global_risk_limited | 128 / 98 | 0.0% / 0.0% | 0 | 1.00x | 83.98→83.86 | +0.109 / +0.151 (3/4) | -0.115 / +0.425 | -0.3% |

`Exact hit` includes queued promotion; `overlap hit` counts only completed reuse and inflight promotion. Non-overlap regression is measured only on authority targets that obtained no speculative overlap. Admission, deadline, source, p95, and per-repeat data are in `metrics.json`. Repetitions vary scheduling/order only and are not independent accuracy samples.

## Deterministic all-wrong worst case

Every authoritative URL is replaced by a guaranteed non-candidate URL, while gates, scores, ordering, and load stay fixed.

| Policy | Offered / realized C | Selected / wrong-started | Call amp. | Mean wait baseline→policy | Conservative pooled / repeat-median net ms/target (+reps) | Non-overlap regression p95 / max | Drained wall benefit |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank5_sequential_unreserved | 1 / 1 | 6280 / 2512 | 3.67x | 7.99→10.33 | -2.902 / -2.896 (0/4) | +3.270 / +3.469 | -51.2% |
| rank_budgeted_round_robin_reserved | 1 / 1 | 1256 / 1256 | 2.34x | 7.97→8.79 | -1.400 / -1.399 (0/4) | +2.450 / +2.615 | -34.4% |
| confidence_global_reserved | 1 / 1 | 1004 / 1004 | 2.07x | 7.96→8.74 | -1.378 / -1.377 (0/4) | +2.451 / +18.277 | -29.1% |
| utility_global_risk_limited | 1 / 1 | 180 / 180 | 1.19x | 7.96→8.07 | -0.712 / -0.712 (0/4) | +0.284 / +2.541 | -11.3% |
| rank5_sequential_unreserved | 8 / 8 | 6280 / 364 | 1.39x | 13.68→18.76 | -5.634 / -5.647 (0/4) | +5.584 / +24.656 | -42.8% |
| rank_budgeted_round_robin_reserved | 8 / 8 | 182 / 182 | 1.19x | 13.69→14.80 | -1.673 / -1.665 (0/4) | +2.524 / +2.799 | -24.1% |
| confidence_global_reserved | 8 / 8 | 182 / 182 | 1.19x | 13.69→14.82 | -1.705 / -1.701 (0/4) | +2.518 / +2.772 | -23.5% |
| utility_global_risk_limited | 8 / 8 | 5 / 5 | 1.01x | 13.69→13.69 | -0.019 / -0.016 (1/4) | +0.167 / +2.425 | -0.9% |
| rank5_sequential_unreserved | 32 / 32 | 6280 / 122 | 1.13x | 36.38→39.82 | -3.997 / -3.678 (0/4) | +5.884 / +6.388 | -30.6% |
| rank_budgeted_round_robin_reserved | 32 / 32 | 61 / 61 | 1.06x | 36.15→36.54 | -0.962 / -0.981 (0/4) | +1.566 / +2.875 | -20.2% |
| confidence_global_reserved | 32 / 32 | 61 / 61 | 1.06x | 36.16→36.67 | -1.087 / -0.893 (0/4) | +1.634 / +19.778 | -21.4% |
| utility_global_risk_limited | 32 / 32 | 5 / 5 | 1.01x | 36.10→36.18 | -0.095 / -0.047 (1/4) | +0.318 / +3.787 | -1.1% |
| rank5_sequential_unreserved | 64 / 64 | 6280 / 84 | 1.09x | 61.94→66.49 | -5.102 / -4.921 (0/4) | +6.595 / +7.856 | -31.3% |
| rank_budgeted_round_robin_reserved | 64 / 64 | 42 / 42 | 1.04x | 61.98→62.44 | -1.029 / -0.998 (0/4) | +1.304 / +2.949 | -19.4% |
| confidence_global_reserved | 64 / 64 | 42 / 42 | 1.04x | 61.92→62.43 | -1.090 / -1.130 (0/4) | +1.197 / +2.579 | -21.2% |
| utility_global_risk_limited | 64 / 64 | 5 / 5 | 1.01x | 61.90→62.02 | -0.133 / -0.143 (0/4) | +0.500 / +2.546 | -0.8% |
| rank5_sequential_unreserved | 128 / 98 | 6280 / 54 | 1.06x | 83.83→86.31 | -3.037 / -3.451 (0/4) | +6.458 / +7.252 | -26.3% |
| rank_budgeted_round_robin_reserved | 128 / 98 | 32 / 32 | 1.03x | 83.85→84.49 | -1.210 / -1.205 (0/4) | +1.286 / +2.530 | -19.0% |
| confidence_global_reserved | 128 / 98 | 32 / 32 | 1.03x | 83.87→84.74 | -1.454 / -1.411 (0/4) | +2.575 / +3.719 | -20.0% |
| utility_global_risk_limited | 128 / 98 | 0 / 0 | 1.00x | 83.86→83.91 | -0.051 / -0.008 (2/4) | +0.483 / +0.685 | -0.9% |

If `Selected=0`, any nonzero paired latency difference is measurement noise from separate asyncio runs. In particular, the all-wrong C=128 pooled value is not a causal effect.

## What authority-first can and cannot guarantee

The broker now removes a whole cancellation set atomically before dispatch, so queued siblings cannot start one-by-one during cleanup. Batch admission also removes the prior O(N)-sweep-per-candidate control path, and the per-tool reserve prevents speculation from occupying both visit slots.

With fixed shared capacity and non-preemptible tool calls, strict zero interference is impossible whenever any wrong speculation is running: a future burst can still need every slot. Absolute isolation requires extra/dedicated capacity, genuinely preemptible calls, or abstention. Therefore the intended saturated behavior is graceful fallback to demand-only, not forced positive speculation at every load.

This table is a deterministic synthetic-service, closed-loop burst experiment. It diagnoses scheduling and resource allocation; it does not claim production network latency. A sustained/open-loop replay and a new whole-session confirmatory holdout are separate validation requirements.
