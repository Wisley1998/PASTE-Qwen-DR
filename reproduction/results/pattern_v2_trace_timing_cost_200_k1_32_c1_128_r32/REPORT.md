# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay scales per-decision LLM overlap while preserving the observed
visit stall. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K sweep=`[1, 2, 4, 8, 16, 32]`, scheduling seeds=`32`, coordination cost=`200.0 ms/start`.
LLM duration scale=`0.7`; selection uses outer-fold OOF empirical atomic-service distributions.

## Global K=1

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 2 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 4 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 8 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 16 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 32 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 64 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 128 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |

## Global K=2

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 2 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 4 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 8 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 16 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 32 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 64 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 128 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |

## Global K=4

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 2 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 4 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 8 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 16 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 32 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 64 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 128 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |

## Global K=8

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 2 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 4 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 8 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 16 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 32 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 64 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 128 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |

## Global K=16

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 2 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 4 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 8 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 16 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 32 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 64 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 128 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |

## Global K=32

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 2 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 4 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 8 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 16 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 32 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 64 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |
| 128 | 0.000 s | 0.00% | 0.00% | 0.00% | 0.00% | 1.000x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
