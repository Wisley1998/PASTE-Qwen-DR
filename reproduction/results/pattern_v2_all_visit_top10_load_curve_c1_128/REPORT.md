# One-policy load curve: Top-10 + full session cache

Every concurrency uses the same policy: Top-10 candidates, infinite-TTL zero-read-cost session URL cache, adaptive idle-fill, and immediate authority preemption in one fixed shared Visit pool. There is no per-load policy selection.

Unconstrained policy coverage is 359/499 = 71.94%. The fixed replay contains 200 tasks (2 replicas) and the Visit pool has 64 slots.

`Tool stall reduction` is the reduction in summed authority-visible Visit wait (queue + remaining service). `Overall E2E` is closed-loop makespan reduction over the complete 0.42x-LLM sessions.

`Spec calls / auth call` counts all physically started speculative tool executions. `Unused spec calls / auth call` counts the subset whose result is never consumed by an authority call. Both use the number of authority tool calls as the denominator.

| C | Realized hit | Spec calls / auth call | Unused spec calls / auth call | Tool stall reduction | Overall E2E |
|---:|---:|---:|---:|---:|---:|
| 1 | 71.94% | 5.423 | 4.741 | 58.25% | 24.32% |
| 2 | 71.94% | 5.423 | 4.741 | 58.25% | 24.25% |
| 4 | 71.94% | 5.423 | 4.741 | 58.25% | 24.25% |
| 8 | 71.94% | 5.423 | 4.742 | 58.25% | 23.83% |
| 16 | 70.92% | 5.412 | 4.739 | 56.88% | 22.00% |
| 32 | 55.02% | 3.821 | 3.299 | 42.21% | 15.22% |
| 48 | 43.46% | 2.716 | 2.301 | 32.35% | 9.94% |
| 64 | 35.24% | 2.144 | 1.807 | 25.13% | 6.44% |
| 96 | 22.65% | 1.531 | 1.313 | 14.66% | 5.19% |
| 128 | 14.77% | 1.183 | 1.041 | 10.19% | 5.41% |

## Mostly-wrong high-load negative control

Only the maximum-load C=128 cell is used. For the negative control, 75% or 100% of authority URLs are replaced after selection by guaranteed non-candidate URLs; prediction scores and admission remain unchanged.

| C=128 scenario | Realized hit | Spec calls / auth call | Unused spec calls / auth call | Tool stall reduction | Overall E2E |
|---|---:|---:|---:|---:|---:|
| observed | 14.77% | 1.183 | 1.041 | 10.19% | 5.41% |
| mostly wrong (75%) | 3.38% | 1.190 | 1.157 | 2.39% | 0.02% |
| all wrong (100%) | 0.00% | 1.201 | 1.201 | 0.00% | 0.00% |

At 75% corruption the optimization retains only the overlap from the remaining correct predictions. In the deterministic all-wrong case, running speculation is preempted immediately for authority: realized hit and latency benefit both become zero, no wrong result is committed, and the cost is limited to wasted speculative calls on otherwise idle tool capacity.

Realized hit falls only when the fixed pool loses speculative slack. Wrong running calls are preempted before authority dispatch, so the curve charges resource waste without allowing speculation to sit in front of a real Visit.
