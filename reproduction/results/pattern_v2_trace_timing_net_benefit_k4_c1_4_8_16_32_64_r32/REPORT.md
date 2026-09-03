# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay uses the original per-decision LLM overlap and visit-stall
timestamps. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K=`4`, scheduling seeds=`32`, coordination cost=`1.0 ms/start`.

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.025 s | 9.06% | 0.83% | 16.60% | 15.61% | 2.128x |
| 4 | 19.025 s | 9.06% | 0.49% | 16.60% | 15.61% | 2.128x |
| 8 | 12.670 s | 6.03% | 0.25% | 10.16% | 17.88% | 1.611x |
| 16 | 8.137 s | 3.88% | 0.25% | 5.49% | 18.58% | 1.328x |
| 32 | 5.674 s | 2.70% | 0.23% | 3.72% | 21.64% | 1.185x |
| 64 | 4.190 s | 2.00% | 0.16% | 2.89% | 25.45% | 1.114x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
