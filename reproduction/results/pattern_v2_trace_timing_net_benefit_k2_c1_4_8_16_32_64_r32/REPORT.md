# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay uses the original per-decision LLM overlap and visit-stall
timestamps. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K=`2`, scheduling seeds=`32`, coordination cost=`1.0 ms/start`.

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.025 s | 9.06% | 0.83% | 16.60% | 15.61% | 2.128x |
| 4 | 12.464 s | 5.94% | 0.30% | 9.92% | 17.72% | 1.602x |
| 8 | 7.290 s | 3.47% | 0.20% | 5.20% | 18.51% | 1.309x |
| 16 | 4.749 s | 2.26% | 0.22% | 3.24% | 20.64% | 1.165x |
| 32 | 2.892 s | 1.38% | 0.20% | 2.25% | 23.59% | 1.094x |
| 64 | 1.532 s | 0.73% | 0.07% | 1.52% | 24.47% | 1.061x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
