# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay uses the original per-decision LLM overlap and visit-stall
timestamps. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K=`1`, scheduling seeds=`32`, coordination cost=`1.0 ms/start`.

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.025 s | 9.06% | 0.83% | 16.60% | 15.61% | 2.128x |
| 4 | 6.913 s | 3.29% | 0.18% | 5.17% | 18.72% | 1.300x |
| 8 | 3.601 s | 1.71% | 0.13% | 2.77% | 20.17% | 1.154x |
| 16 | 2.342 s | 1.12% | 0.14% | 1.89% | 22.42% | 1.082x |
| 32 | 1.182 s | 0.56% | 0.09% | 1.22% | 23.58% | 1.048x |
| 64 | 0.297 s | 0.14% | 0.01% | 0.76% | 22.40% | 1.033x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
