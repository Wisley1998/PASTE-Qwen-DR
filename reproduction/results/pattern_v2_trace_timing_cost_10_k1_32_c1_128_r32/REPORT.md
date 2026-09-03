# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay scales per-decision LLM overlap while preserving the observed
visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K sweep=`[1, 2, 4, 8, 16, 32]`, scheduling seeds=`32`, coordination cost=`10.0 ms/start`.
LLM duration scale=`0.7`; selection uses outer-fold OOF empirical atomic-service distributions.

## Global K=1

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.763 s | 6.08% | 0.76% | 16.60% | 15.61% | 2.128x |
| 2 | 7.333 s | 3.49% | 0.46% | 9.38% | 16.41% | 1.604x |
| 4 | 4.195 s | 2.00% | 0.31% | 5.52% | 18.68% | 1.300x |
| 8 | 2.660 s | 1.27% | 0.25% | 3.23% | 20.86% | 1.152x |
| 16 | 1.882 s | 0.90% | 0.25% | 1.72% | 22.17% | 1.082x |
| 32 | 1.233 s | 0.59% | 0.17% | 0.94% | 24.42% | 1.048x |
| 64 | 0.640 s | 0.30% | 0.09% | 0.68% | 28.71% | 1.030x |
| 128 | 0.283 s | 0.13% | 0.09% | 0.43% | 25.00% | 1.026x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.763 s | 6.08% | 0.76% | 16.60% | 15.61% | 2.128x |
| 2 | 12.763 s | 6.08% | 0.71% | 16.60% | 15.61% | 2.128x |
| 4 | 7.449 s | 3.55% | 0.44% | 9.77% | 16.95% | 1.607x |
| 8 | 4.641 s | 2.21% | 0.34% | 6.02% | 19.49% | 1.305x |
| 16 | 2.878 s | 1.37% | 0.29% | 3.67% | 22.24% | 1.161x |
| 32 | 1.964 s | 0.94% | 0.24% | 1.95% | 22.51% | 1.095x |
| 64 | 1.482 s | 0.71% | 0.18% | 1.26% | 23.81% | 1.062x |
| 128 | 1.425 s | 0.68% | 0.08% | 1.28% | 25.00% | 1.051x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.763 s | 6.08% | 0.76% | 16.60% | 15.61% | 2.128x |
| 2 | 12.763 s | 6.08% | 0.71% | 16.60% | 15.61% | 2.128x |
| 4 | 12.763 s | 6.08% | 0.62% | 16.60% | 15.61% | 2.128x |
| 8 | 7.365 s | 3.51% | 0.48% | 9.63% | 16.43% | 1.621x |
| 16 | 4.431 s | 2.11% | 0.37% | 6.13% | 19.37% | 1.324x |
| 32 | 2.788 s | 1.33% | 0.28% | 3.64% | 20.80% | 1.187x |
| 64 | 1.979 s | 0.94% | 0.20% | 2.25% | 21.65% | 1.120x |
| 128 | 1.301 s | 0.62% | 0.08% | 1.70% | 25.81% | 1.098x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.763 s | 6.08% | 0.76% | 16.60% | 15.61% | 2.128x |
| 2 | 12.763 s | 6.08% | 0.71% | 16.60% | 15.61% | 2.128x |
| 4 | 12.763 s | 6.08% | 0.62% | 16.60% | 15.61% | 2.128x |
| 8 | 12.763 s | 6.08% | 0.61% | 16.60% | 15.61% | 2.128x |
| 16 | 7.219 s | 3.44% | 0.51% | 9.89% | 16.34% | 1.643x |
| 32 | 4.683 s | 2.23% | 0.36% | 6.66% | 19.65% | 1.359x |
| 64 | 2.872 s | 1.37% | 0.20% | 4.34% | 21.86% | 1.224x |
| 128 | 1.722 s | 0.82% | 0.08% | 2.98% | 20.69% | 1.196x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.763 s | 6.08% | 0.76% | 16.60% | 15.61% | 2.128x |
| 2 | 12.763 s | 6.08% | 0.71% | 16.60% | 15.61% | 2.128x |
| 4 | 12.763 s | 6.08% | 0.62% | 16.60% | 15.61% | 2.128x |
| 8 | 12.763 s | 6.08% | 0.61% | 16.60% | 15.61% | 2.128x |
| 16 | 12.763 s | 6.08% | 0.63% | 16.60% | 15.61% | 2.128x |
| 32 | 7.455 s | 3.55% | 0.46% | 10.33% | 16.54% | 1.681x |
| 64 | 4.960 s | 2.36% | 0.23% | 7.30% | 19.83% | 1.417x |
| 128 | 5.127 s | 2.44% | 0.08% | 6.38% | 20.75% | 1.357x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.763 s | 6.08% | 0.76% | 16.60% | 15.61% | 2.128x |
| 2 | 12.763 s | 6.08% | 0.71% | 16.60% | 15.61% | 2.128x |
| 4 | 12.763 s | 6.08% | 0.62% | 16.60% | 15.61% | 2.128x |
| 8 | 12.763 s | 6.08% | 0.61% | 16.60% | 15.61% | 2.128x |
| 16 | 12.763 s | 6.08% | 0.63% | 16.60% | 15.61% | 2.128x |
| 32 | 12.763 s | 6.08% | 0.56% | 16.60% | 15.61% | 2.128x |
| 64 | 8.080 s | 3.85% | 0.26% | 11.10% | 16.70% | 1.751x |
| 128 | 5.101 s | 2.43% | 0.08% | 9.36% | 16.85% | 1.651x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
