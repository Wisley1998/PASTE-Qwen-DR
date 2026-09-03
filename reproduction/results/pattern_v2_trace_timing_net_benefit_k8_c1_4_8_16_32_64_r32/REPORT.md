# Pattern-v2 Real-Trace Timing Net-Benefit Replay

This replay uses the original per-decision LLM overlap and visit-stall
timestamps. Exact hits suppress one matching AUTH URL call; no 20 ms
synthetic service or shadow AUTH is used.

Global K=`8`, scheduling seeds=`32`, coordination cost=`1.0 ms/start`.

| Task C | Net saved tool stall / replay | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Prediction precision | Physical-call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.025 s | 9.06% | 0.83% | 16.60% | 15.61% | 2.128x |
| 4 | 19.025 s | 9.06% | 0.49% | 16.60% | 15.61% | 2.128x |
| 8 | 19.025 s | 9.06% | 0.38% | 16.60% | 15.61% | 2.128x |
| 16 | 12.891 s | 6.14% | 0.26% | 10.48% | 17.83% | 1.631x |
| 32 | 8.648 s | 4.12% | 0.24% | 5.90% | 18.61% | 1.364x |
| 64 | 6.598 s | 3.14% | 0.18% | 4.14% | 21.35% | 1.225x |

Multi-URL visits use a conservative proportional-credit rule. Visits
without a following LLM timestamp receive zero benefit. Repetitions are
deterministic scheduling-order sensitivity runs, not independent traces.
