# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay scales per-decision LLM overlap while preserving the observed
visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K sweep=`[1, 2, 4, 8, 16, 32]`, scheduling seeds=`32`, coordination cost=`50.0 ms/start`.
LLM duration scale=`0.7`; selection uses outer-fold OOF empirical atomic-service distributions.

## Global K=1

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.530 s | 0.25% | 0.03% | 13.19% | 14.93% | 1.970x |
| 2 | 0.636 s | 0.30% | 0.15% | 8.90% | 16.25% | 1.582x |
| 4 | 0.737 s | 0.35% | 0.17% | 5.52% | 18.74% | 1.299x |
| 8 | 0.859 s | 0.41% | 0.17% | 3.23% | 20.87% | 1.152x |
| 16 | 0.896 s | 0.43% | 0.20% | 1.72% | 22.03% | 1.082x |
| 32 | 0.650 s | 0.31% | 0.13% | 0.94% | 23.50% | 1.048x |
| 64 | 0.256 s | 0.12% | 0.06% | 0.68% | 27.18% | 1.030x |
| 128 | -0.037 s | -0.02% | 0.06% | 0.43% | 25.00% | 1.026x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.530 s | 0.25% | 0.03% | 13.19% | 14.93% | 1.970x |
| 2 | 0.530 s | 0.25% | 0.24% | 13.19% | 14.93% | 1.970x |
| 4 | 0.544 s | 0.26% | 0.23% | 9.49% | 16.85% | 1.597x |
| 8 | 1.079 s | 0.51% | 0.23% | 5.98% | 19.37% | 1.305x |
| 16 | 0.947 s | 0.45% | 0.22% | 3.66% | 22.04% | 1.160x |
| 32 | 0.828 s | 0.39% | 0.19% | 1.95% | 22.04% | 1.095x |
| 64 | 0.743 s | 0.35% | 0.14% | 1.26% | 22.69% | 1.061x |
| 128 | 0.785 s | 0.37% | 0.03% | 1.28% | 25.00% | 1.051x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.530 s | 0.25% | 0.03% | 13.19% | 14.93% | 1.970x |
| 2 | 0.530 s | 0.25% | 0.24% | 13.19% | 14.93% | 1.970x |
| 4 | 0.530 s | 0.25% | 0.33% | 13.19% | 14.93% | 1.970x |
| 8 | 0.413 s | 0.20% | 0.33% | 9.41% | 16.23% | 1.616x |
| 16 | 0.649 s | 0.31% | 0.28% | 6.00% | 18.91% | 1.323x |
| 32 | 0.586 s | 0.28% | 0.22% | 3.62% | 20.40% | 1.186x |
| 64 | 0.576 s | 0.27% | 0.15% | 2.25% | 20.81% | 1.119x |
| 128 | 0.111 s | 0.05% | 0.03% | 1.70% | 23.33% | 1.098x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.530 s | 0.25% | 0.03% | 13.19% | 14.93% | 1.970x |
| 2 | 0.530 s | 0.25% | 0.24% | 13.19% | 14.93% | 1.970x |
| 4 | 0.530 s | 0.25% | 0.33% | 13.19% | 14.93% | 1.970x |
| 8 | 0.530 s | 0.25% | 0.41% | 13.19% | 14.93% | 1.970x |
| 16 | 0.053 s | 0.03% | 0.39% | 9.63% | 16.05% | 1.634x |
| 32 | 0.530 s | 0.25% | 0.28% | 6.53% | 19.20% | 1.354x |
| 64 | 0.265 s | 0.13% | 0.15% | 4.32% | 21.06% | 1.220x |
| 128 | -0.498 s | -0.24% | 0.03% | 2.98% | 19.64% | 1.191x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.530 s | 0.25% | 0.03% | 13.19% | 14.93% | 1.970x |
| 2 | 0.530 s | 0.25% | 0.24% | 13.19% | 14.93% | 1.970x |
| 4 | 0.530 s | 0.25% | 0.33% | 13.19% | 14.93% | 1.970x |
| 8 | 0.530 s | 0.25% | 0.41% | 13.19% | 14.93% | 1.970x |
| 16 | 0.530 s | 0.25% | 0.49% | 13.19% | 14.93% | 1.970x |
| 32 | -0.185 s | -0.09% | 0.36% | 9.83% | 16.09% | 1.664x |
| 64 | 0.084 s | 0.04% | 0.16% | 7.02% | 19.08% | 1.408x |
| 128 | 1.037 s | 0.49% | 0.00% | 6.38% | 20.39% | 1.349x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.530 s | 0.25% | 0.03% | 13.19% | 14.93% | 1.970x |
| 2 | 0.530 s | 0.25% | 0.24% | 13.19% | 14.93% | 1.970x |
| 4 | 0.530 s | 0.25% | 0.33% | 13.19% | 14.93% | 1.970x |
| 8 | 0.530 s | 0.25% | 0.41% | 13.19% | 14.93% | 1.970x |
| 16 | 0.530 s | 0.25% | 0.49% | 13.19% | 14.93% | 1.970x |
| 32 | 0.530 s | 0.25% | 0.45% | 13.19% | 14.93% | 1.970x |
| 64 | -0.355 s | -0.17% | 0.16% | 10.29% | 15.98% | 1.726x |
| 128 | -2.432 s | -1.16% | 0.00% | 8.51% | 15.82% | 1.634x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
