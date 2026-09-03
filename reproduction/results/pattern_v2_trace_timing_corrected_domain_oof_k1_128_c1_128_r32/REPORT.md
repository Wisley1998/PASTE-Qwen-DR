# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay scales per-decision LLM overlap while preserving the observed
visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K sweep=`[1, 2, 4, 8, 16, 32, 64, 128]`, scheduling seeds=`32`, coordination cost=`1.0 ms/start`.
LLM duration scale=`0.7`; selection uses outer-fold OOF empirical atomic-service distributions.

## Global K=1

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 8.860 s | 4.22% | 0.53% | 9.38% | 16.41% | 1.604x |
| 4 | 4.977 s | 2.37% | 0.35% | 5.52% | 18.68% | 1.300x |
| 8 | 3.067 s | 1.46% | 0.27% | 3.23% | 20.86% | 1.152x |
| 16 | 2.105 s | 1.00% | 0.26% | 1.72% | 22.17% | 1.082x |
| 32 | 1.366 s | 0.65% | 0.18% | 0.94% | 24.42% | 1.048x |
| 64 | 0.729 s | 0.35% | 0.10% | 0.68% | 28.71% | 1.030x |
| 128 | 0.355 s | 0.17% | 0.10% | 0.43% | 25.00% | 1.026x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 8.995 s | 4.28% | 0.49% | 9.77% | 16.95% | 1.607x |
| 8 | 5.443 s | 2.59% | 0.37% | 6.02% | 19.49% | 1.305x |
| 16 | 3.317 s | 1.58% | 0.31% | 3.67% | 22.24% | 1.161x |
| 32 | 2.224 s | 1.06% | 0.25% | 1.95% | 22.51% | 1.095x |
| 64 | 1.653 s | 0.79% | 0.19% | 1.26% | 23.81% | 1.062x |
| 128 | 1.569 s | 0.75% | 0.09% | 1.28% | 25.00% | 1.051x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 8.938 s | 4.26% | 0.52% | 9.63% | 16.43% | 1.621x |
| 16 | 5.281 s | 2.52% | 0.39% | 6.13% | 19.37% | 1.324x |
| 32 | 3.289 s | 1.57% | 0.29% | 3.64% | 20.80% | 1.187x |
| 64 | 2.304 s | 1.10% | 0.21% | 2.25% | 21.65% | 1.120x |
| 128 | 1.580 s | 0.75% | 0.09% | 1.70% | 25.81% | 1.098x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 8.844 s | 4.21% | 0.54% | 9.89% | 16.34% | 1.643x |
| 32 | 5.628 s | 2.68% | 0.38% | 6.66% | 19.65% | 1.359x |
| 64 | 3.478 s | 1.66% | 0.22% | 4.34% | 21.86% | 1.224x |
| 128 | 2.244 s | 1.07% | 0.09% | 2.98% | 20.69% | 1.196x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 15.589 s | 7.42% | 0.66% | 16.60% | 15.61% | 2.128x |
| 32 | 9.180 s | 4.37% | 0.49% | 10.33% | 16.54% | 1.681x |
| 64 | 6.061 s | 2.89% | 0.25% | 7.30% | 19.83% | 1.417x |
| 128 | 6.081 s | 2.90% | 0.09% | 6.38% | 20.75% | 1.357x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 15.589 s | 7.42% | 0.66% | 16.60% | 15.61% | 2.128x |
| 32 | 15.589 s | 7.42% | 0.58% | 16.60% | 15.61% | 2.128x |
| 64 | 9.986 s | 4.76% | 0.27% | 11.10% | 16.70% | 1.751x |
| 128 | 6.757 s | 3.22% | 0.09% | 9.36% | 16.85% | 1.651x |

## Global K=64

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 15.589 s | 7.42% | 0.66% | 16.60% | 15.61% | 2.128x |
| 32 | 15.589 s | 7.42% | 0.58% | 16.60% | 15.61% | 2.128x |
| 64 | 15.589 s | 7.42% | 0.29% | 16.60% | 15.61% | 2.128x |
| 128 | 12.022 s | 5.73% | 0.09% | 14.47% | 15.71% | 2.004x |

## Global K=128

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 15.589 s | 7.42% | 0.66% | 16.60% | 15.61% | 2.128x |
| 32 | 15.589 s | 7.42% | 0.58% | 16.60% | 15.61% | 2.128x |
| 64 | 15.589 s | 7.42% | 0.29% | 16.60% | 15.61% | 2.128x |
| 128 | 15.589 s | 7.42% | 0.09% | 16.60% | 15.61% | 2.128x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
