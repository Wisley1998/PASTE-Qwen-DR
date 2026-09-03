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
| 2 | 8.668 s | 4.13% | 0.53% | 9.60% | 16.63% | 1.602x |
| 4 | 4.906 s | 2.34% | 0.35% | 5.68% | 18.57% | 1.301x |
| 8 | 2.974 s | 1.42% | 0.25% | 3.39% | 20.37% | 1.153x |
| 16 | 2.079 s | 0.99% | 0.26% | 1.72% | 18.39% | 1.086x |
| 32 | 1.458 s | 0.69% | 0.21% | 0.90% | 17.26% | 1.052x |
| 64 | 1.523 s | 0.73% | 0.18% | 0.96% | 25.24% | 1.032x |
| 128 | 1.552 s | 0.74% | 0.10% | 0.85% | 25.00% | 1.026x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 8.954 s | 4.26% | 0.50% | 10.11% | 17.26% | 1.605x |
| 8 | 5.302 s | 2.53% | 0.36% | 6.26% | 19.63% | 1.305x |
| 16 | 3.163 s | 1.51% | 0.30% | 3.43% | 20.19% | 1.166x |
| 32 | 2.090 s | 1.00% | 0.24% | 1.72% | 17.97% | 1.101x |
| 64 | 1.660 s | 0.79% | 0.19% | 1.24% | 18.56% | 1.066x |
| 128 | 1.569 s | 0.75% | 0.09% | 1.28% | 18.75% | 1.055x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 8.919 s | 4.25% | 0.53% | 10.04% | 16.92% | 1.618x |
| 16 | 5.253 s | 2.50% | 0.40% | 6.61% | 20.20% | 1.321x |
| 32 | 3.191 s | 1.52% | 0.28% | 3.54% | 19.51% | 1.190x |
| 64 | 2.031 s | 0.97% | 0.21% | 1.94% | 16.62% | 1.128x |
| 128 | 1.580 s | 0.75% | 0.09% | 1.70% | 22.58% | 1.102x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 8.930 s | 4.25% | 0.55% | 10.43% | 17.00% | 1.638x |
| 32 | 5.328 s | 2.54% | 0.35% | 7.01% | 20.18% | 1.357x |
| 64 | 3.168 s | 1.51% | 0.22% | 4.07% | 19.91% | 1.230x |
| 128 | 1.812 s | 0.86% | 0.09% | 2.13% | 13.79% | 1.213x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 15.589 s | 7.42% | 0.66% | 16.60% | 15.61% | 2.128x |
| 32 | 9.331 s | 4.44% | 0.49% | 10.84% | 17.03% | 1.677x |
| 64 | 6.047 s | 2.88% | 0.25% | 7.85% | 20.60% | 1.413x |
| 128 | 6.086 s | 2.90% | 0.09% | 6.81% | 21.70% | 1.353x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.589 s | 7.42% | 0.93% | 16.60% | 15.61% | 2.128x |
| 2 | 15.589 s | 7.42% | 0.81% | 16.60% | 15.61% | 2.128x |
| 4 | 15.589 s | 7.42% | 0.69% | 16.60% | 15.61% | 2.128x |
| 8 | 15.589 s | 7.42% | 0.65% | 16.60% | 15.61% | 2.128x |
| 16 | 15.589 s | 7.42% | 0.66% | 16.60% | 15.61% | 2.128x |
| 32 | 15.589 s | 7.42% | 0.58% | 16.60% | 15.61% | 2.128x |
| 64 | 10.199 s | 4.86% | 0.27% | 11.32% | 17.00% | 1.748x |
| 128 | 7.042 s | 3.35% | 0.09% | 9.36% | 17.39% | 1.647x |

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
| 128 | 11.744 s | 5.59% | 0.09% | 14.89% | 16.07% | 2.000x |

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
