# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay scales per-decision LLM overlap while preserving the observed
visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K sweep=`[1, 2, 4, 8, 16, 32]`, scheduling seeds=`32`, coordination cost=`5.2 ms/start`.
LLM duration scale=`0.7`; selection uses outer-fold OOF empirical atomic-service distributions.

## Global K=1

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.270 s | 6.80% | 0.85% | 16.60% | 15.61% | 2.128x |
| 2 | 8.147 s | 3.88% | 0.49% | 9.38% | 16.41% | 1.604x |
| 4 | 4.612 s | 2.20% | 0.33% | 5.52% | 18.68% | 1.300x |
| 8 | 2.877 s | 1.37% | 0.26% | 3.23% | 20.86% | 1.152x |
| 16 | 2.001 s | 0.95% | 0.25% | 1.72% | 22.17% | 1.082x |
| 32 | 1.304 s | 0.62% | 0.17% | 0.94% | 24.42% | 1.048x |
| 64 | 0.687 s | 0.33% | 0.10% | 0.68% | 28.71% | 1.030x |
| 128 | 0.321 s | 0.15% | 0.09% | 0.43% | 25.00% | 1.026x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.270 s | 6.80% | 0.85% | 16.60% | 15.61% | 2.128x |
| 2 | 14.270 s | 6.80% | 0.76% | 16.60% | 15.61% | 2.128x |
| 4 | 8.274 s | 3.94% | 0.46% | 9.77% | 16.95% | 1.607x |
| 8 | 5.069 s | 2.41% | 0.35% | 6.02% | 19.49% | 1.305x |
| 16 | 3.112 s | 1.48% | 0.30% | 3.67% | 22.24% | 1.161x |
| 32 | 2.103 s | 1.00% | 0.24% | 1.95% | 22.51% | 1.095x |
| 64 | 1.573 s | 0.75% | 0.18% | 1.26% | 23.81% | 1.062x |
| 128 | 1.502 s | 0.72% | 0.09% | 1.28% | 25.00% | 1.051x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.270 s | 6.80% | 0.85% | 16.60% | 15.61% | 2.128x |
| 2 | 14.270 s | 6.80% | 0.76% | 16.60% | 15.61% | 2.128x |
| 4 | 14.270 s | 6.80% | 0.66% | 16.60% | 15.61% | 2.128x |
| 8 | 8.204 s | 3.91% | 0.50% | 9.63% | 16.43% | 1.621x |
| 16 | 4.884 s | 2.33% | 0.38% | 6.13% | 19.37% | 1.324x |
| 32 | 3.055 s | 1.45% | 0.29% | 3.64% | 20.80% | 1.187x |
| 64 | 2.152 s | 1.02% | 0.21% | 2.25% | 21.65% | 1.120x |
| 128 | 1.450 s | 0.69% | 0.09% | 1.70% | 25.81% | 1.098x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.270 s | 6.80% | 0.85% | 16.60% | 15.61% | 2.128x |
| 2 | 14.270 s | 6.80% | 0.76% | 16.60% | 15.61% | 2.128x |
| 4 | 14.270 s | 6.80% | 0.66% | 16.60% | 15.61% | 2.128x |
| 8 | 14.270 s | 6.80% | 0.63% | 16.60% | 15.61% | 2.128x |
| 16 | 8.086 s | 3.85% | 0.52% | 9.89% | 16.34% | 1.643x |
| 32 | 5.187 s | 2.47% | 0.37% | 6.66% | 19.65% | 1.359x |
| 64 | 3.195 s | 1.52% | 0.21% | 4.34% | 21.86% | 1.224x |
| 128 | 2.000 s | 0.95% | 0.09% | 2.98% | 20.69% | 1.196x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.270 s | 6.80% | 0.85% | 16.60% | 15.61% | 2.128x |
| 2 | 14.270 s | 6.80% | 0.76% | 16.60% | 15.61% | 2.128x |
| 4 | 14.270 s | 6.80% | 0.66% | 16.60% | 15.61% | 2.128x |
| 8 | 14.270 s | 6.80% | 0.63% | 16.60% | 15.61% | 2.128x |
| 16 | 14.270 s | 6.80% | 0.64% | 16.60% | 15.61% | 2.128x |
| 32 | 8.375 s | 3.99% | 0.48% | 10.33% | 16.54% | 1.681x |
| 64 | 5.547 s | 2.64% | 0.24% | 7.30% | 19.83% | 1.417x |
| 128 | 5.636 s | 2.68% | 0.09% | 6.38% | 20.75% | 1.357x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.270 s | 6.80% | 0.85% | 16.60% | 15.61% | 2.128x |
| 2 | 14.270 s | 6.80% | 0.76% | 16.60% | 15.61% | 2.128x |
| 4 | 14.270 s | 6.80% | 0.66% | 16.60% | 15.61% | 2.128x |
| 8 | 14.270 s | 6.80% | 0.63% | 16.60% | 15.61% | 2.128x |
| 16 | 14.270 s | 6.80% | 0.64% | 16.60% | 15.61% | 2.128x |
| 32 | 14.270 s | 6.80% | 0.57% | 16.60% | 15.61% | 2.128x |
| 64 | 9.096 s | 4.33% | 0.27% | 11.10% | 16.70% | 1.751x |
| 128 | 5.984 s | 2.85% | 0.09% | 9.36% | 16.85% | 1.651x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
