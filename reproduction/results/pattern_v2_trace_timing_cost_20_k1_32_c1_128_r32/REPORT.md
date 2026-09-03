# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay scales per-decision LLM overlap while preserving the observed
visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K sweep=`[1, 2, 4, 8, 16, 32]`, scheduling seeds=`32`, coordination cost=`20.0 ms/start`.
LLM duration scale=`0.7`; selection uses outer-fold OOF empirical atomic-service distributions.

## Global K=1

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.623 s | 4.58% | 0.57% | 16.60% | 15.61% | 2.128x |
| 2 | 5.636 s | 2.68% | 0.38% | 9.38% | 16.41% | 1.604x |
| 4 | 3.327 s | 1.58% | 0.28% | 5.52% | 18.68% | 1.300x |
| 8 | 2.207 s | 1.05% | 0.23% | 3.23% | 20.86% | 1.152x |
| 16 | 1.634 s | 0.78% | 0.24% | 1.72% | 22.17% | 1.082x |
| 32 | 1.084 s | 0.52% | 0.16% | 0.94% | 24.42% | 1.048x |
| 64 | 0.541 s | 0.26% | 0.08% | 0.68% | 28.71% | 1.030x |
| 128 | 0.203 s | 0.10% | 0.08% | 0.43% | 25.00% | 1.026x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.623 s | 4.58% | 0.57% | 16.60% | 15.61% | 2.128x |
| 2 | 9.623 s | 4.58% | 0.59% | 16.60% | 15.61% | 2.128x |
| 4 | 5.731 s | 2.73% | 0.39% | 9.77% | 16.95% | 1.607x |
| 8 | 3.749 s | 1.79% | 0.31% | 6.02% | 19.49% | 1.305x |
| 16 | 2.390 s | 1.14% | 0.28% | 3.67% | 22.24% | 1.161x |
| 32 | 1.676 s | 0.80% | 0.22% | 1.95% | 22.51% | 1.095x |
| 64 | 1.292 s | 0.62% | 0.17% | 1.26% | 23.81% | 1.062x |
| 128 | 1.265 s | 0.60% | 0.07% | 1.28% | 25.00% | 1.051x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.623 s | 4.58% | 0.57% | 16.60% | 15.61% | 2.128x |
| 2 | 9.623 s | 4.58% | 0.59% | 16.60% | 15.61% | 2.128x |
| 4 | 9.623 s | 4.58% | 0.55% | 16.60% | 15.61% | 2.128x |
| 8 | 5.618 s | 2.68% | 0.44% | 9.63% | 16.43% | 1.621x |
| 16 | 3.485 s | 1.66% | 0.35% | 6.13% | 19.37% | 1.324x |
| 32 | 2.232 s | 1.06% | 0.26% | 3.64% | 20.80% | 1.187x |
| 64 | 1.618 s | 0.77% | 0.19% | 2.25% | 21.65% | 1.120x |
| 128 | 0.991 s | 0.47% | 0.07% | 1.70% | 25.81% | 1.098x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.623 s | 4.58% | 0.57% | 16.60% | 15.61% | 2.128x |
| 2 | 9.623 s | 4.58% | 0.59% | 16.60% | 15.61% | 2.128x |
| 4 | 9.623 s | 4.58% | 0.55% | 16.60% | 15.61% | 2.128x |
| 8 | 9.623 s | 4.58% | 0.56% | 16.60% | 15.61% | 2.128x |
| 16 | 5.414 s | 2.58% | 0.48% | 9.89% | 16.34% | 1.643x |
| 32 | 3.633 s | 1.73% | 0.34% | 6.66% | 19.65% | 1.359x |
| 64 | 2.198 s | 1.05% | 0.19% | 4.34% | 21.86% | 1.224x |
| 128 | 1.142 s | 0.54% | 0.07% | 2.98% | 20.69% | 1.196x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.623 s | 4.58% | 0.57% | 16.60% | 15.61% | 2.128x |
| 2 | 9.623 s | 4.58% | 0.59% | 16.60% | 15.61% | 2.128x |
| 4 | 9.623 s | 4.58% | 0.55% | 16.60% | 15.61% | 2.128x |
| 8 | 9.623 s | 4.58% | 0.56% | 16.60% | 15.61% | 2.128x |
| 16 | 9.623 s | 4.58% | 0.59% | 16.60% | 15.61% | 2.128x |
| 32 | 5.539 s | 2.64% | 0.44% | 10.33% | 16.54% | 1.681x |
| 64 | 3.737 s | 1.78% | 0.22% | 7.30% | 19.83% | 1.417x |
| 128 | 4.067 s | 1.94% | 0.06% | 6.38% | 20.75% | 1.357x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.623 s | 4.58% | 0.57% | 16.60% | 15.61% | 2.128x |
| 2 | 9.623 s | 4.58% | 0.59% | 16.60% | 15.61% | 2.128x |
| 4 | 9.623 s | 4.58% | 0.55% | 16.60% | 15.61% | 2.128x |
| 8 | 9.623 s | 4.58% | 0.56% | 16.60% | 15.61% | 2.128x |
| 16 | 9.623 s | 4.58% | 0.59% | 16.60% | 15.61% | 2.128x |
| 32 | 9.623 s | 4.58% | 0.53% | 16.60% | 15.61% | 2.128x |
| 64 | 5.962 s | 2.84% | 0.23% | 11.10% | 16.70% | 1.751x |
| 128 | 3.261 s | 1.55% | 0.06% | 9.36% | 16.85% | 1.651x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
