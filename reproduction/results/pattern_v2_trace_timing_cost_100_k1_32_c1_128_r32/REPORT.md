# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay scales per-decision LLM overlap while preserving the observed
visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K sweep=`[1, 2, 4, 8, 16, 32]`, scheduling seeds=`32`, coordination cost=`100.0 ms/start`.
LLM duration scale=`0.7`; selection uses outer-fold OOF empirical atomic-service distributions.

## Global K=1

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -2.212 s | -1.05% | -0.13% | 3.83% | 20.37% | 1.183x |
| 2 | -1.945 s | -0.93% | -0.09% | 3.55% | 20.35% | 1.170x |
| 4 | -1.486 s | -0.71% | -0.01% | 3.02% | 20.66% | 1.144x |
| 8 | -0.722 s | -0.34% | 0.08% | 2.42% | 21.72% | 1.108x |
| 16 | -0.100 s | -0.05% | 0.15% | 1.48% | 22.80% | 1.067x |
| 32 | 0.130 s | 0.06% | 0.11% | 0.76% | 24.19% | 1.038x |
| 64 | -0.025 s | -0.01% | 0.05% | 0.55% | 31.06% | 1.022x |
| 128 | -0.337 s | -0.16% | 0.04% | 0.43% | 28.57% | 1.021x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -2.212 s | -1.05% | -0.13% | 3.83% | 20.37% | 1.183x |
| 2 | -2.212 s | -1.05% | -0.10% | 3.83% | 20.37% | 1.183x |
| 4 | -2.135 s | -1.02% | -0.02% | 3.80% | 20.55% | 1.180x |
| 8 | -1.739 s | -0.83% | 0.07% | 3.46% | 20.64% | 1.162x |
| 16 | -0.900 s | -0.43% | 0.15% | 2.77% | 21.91% | 1.118x |
| 32 | -0.156 s | -0.07% | 0.15% | 1.65% | 22.64% | 1.072x |
| 64 | 0.270 s | 0.13% | 0.12% | 1.00% | 25.12% | 1.042x |
| 128 | 0.460 s | 0.22% | 0.02% | 0.85% | 27.27% | 1.034x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -2.212 s | -1.05% | -0.13% | 3.83% | 20.37% | 1.183x |
| 2 | -2.212 s | -1.05% | -0.10% | 3.83% | 20.37% | 1.183x |
| 4 | -2.212 s | -1.05% | -0.02% | 3.83% | 20.37% | 1.183x |
| 8 | -2.191 s | -1.04% | 0.05% | 3.82% | 20.34% | 1.182x |
| 16 | -1.982 s | -0.94% | 0.14% | 3.66% | 20.57% | 1.172x |
| 32 | -1.122 s | -0.53% | 0.17% | 2.85% | 21.07% | 1.130x |
| 64 | -0.344 s | -0.16% | 0.13% | 1.62% | 21.34% | 1.078x |
| 128 | -0.215 s | -0.10% | 0.02% | 1.28% | 22.22% | 1.060x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -2.212 s | -1.05% | -0.13% | 3.83% | 20.37% | 1.183x |
| 2 | -2.212 s | -1.05% | -0.10% | 3.83% | 20.37% | 1.183x |
| 4 | -2.212 s | -1.05% | -0.02% | 3.83% | 20.37% | 1.183x |
| 8 | -2.212 s | -1.05% | 0.05% | 3.83% | 20.37% | 1.183x |
| 16 | -2.205 s | -1.05% | 0.14% | 3.83% | 20.39% | 1.183x |
| 32 | -2.059 s | -0.98% | 0.16% | 3.74% | 20.65% | 1.176x |
| 64 | -1.211 s | -0.58% | 0.13% | 2.78% | 21.11% | 1.124x |
| 128 | -1.389 s | -0.66% | 0.02% | 1.70% | 16.67% | 1.106x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -2.212 s | -1.05% | -0.13% | 3.83% | 20.37% | 1.183x |
| 2 | -2.212 s | -1.05% | -0.10% | 3.83% | 20.37% | 1.183x |
| 4 | -2.212 s | -1.05% | -0.02% | 3.83% | 20.37% | 1.183x |
| 8 | -2.212 s | -1.05% | 0.05% | 3.83% | 20.37% | 1.183x |
| 16 | -2.212 s | -1.05% | 0.14% | 3.83% | 20.37% | 1.183x |
| 32 | -2.212 s | -1.05% | 0.16% | 3.83% | 20.37% | 1.183x |
| 64 | -1.983 s | -0.94% | 0.12% | 3.80% | 21.14% | 1.173x |
| 128 | -1.398 s | -0.67% | 0.02% | 3.40% | 22.73% | 1.145x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -2.212 s | -1.05% | -0.13% | 3.83% | 20.37% | 1.183x |
| 2 | -2.212 s | -1.05% | -0.10% | 3.83% | 20.37% | 1.183x |
| 4 | -2.212 s | -1.05% | -0.02% | 3.83% | 20.37% | 1.183x |
| 8 | -2.212 s | -1.05% | 0.05% | 3.83% | 20.37% | 1.183x |
| 16 | -2.212 s | -1.05% | 0.14% | 3.83% | 20.37% | 1.183x |
| 32 | -2.212 s | -1.05% | 0.16% | 3.83% | 20.37% | 1.183x |
| 64 | -2.212 s | -1.05% | 0.12% | 3.83% | 20.37% | 1.183x |
| 128 | -2.212 s | -1.05% | 0.02% | 3.83% | 20.37% | 1.183x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
