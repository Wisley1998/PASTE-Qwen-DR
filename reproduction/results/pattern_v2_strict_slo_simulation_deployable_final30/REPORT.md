# Strict Pattern V2 hashed-SLO CPU sensitivity

This is a deterministic systems sensitivity replay, not a live-LLM or autonomous-agent result. Candidate inference is online and causal. Tool service comes only from the sealed normalized-invocation hashed SLO clock; recorded trace tool durations are never read. Recorded 0.42x LLM durations are environment-only proxy service and never enter the policy.

Policy coverage: 100/129 = 77.52%.

| C | Realized hit | Tool stall reduction | E2E speedup | Mean-flow speedup | Call amp. |
|---:|---:|---:|---:|---:|---:|
| 1 | 77.52% | 63.47% | 26.33% | 26.33% | 6.876x |
| 2 | 77.52% | 63.47% | 26.38% | 26.33% | 6.876x |
| 4 | 77.52% | 63.47% | 26.56% | 26.33% | 6.876x |
| 8 | 77.51% | 63.45% | 26.88% | 26.32% | 6.876x |
| 16 | 75.55% | 60.18% | 26.56% | 24.96% | 6.701x |
| 32 | 54.31% | 37.49% | 19.50% | 15.55% | 4.359x |
| 48 | 40.50% | 25.06% | 16.09% | 10.40% | 3.362x |
| 64 | 29.91% | 16.76% | 12.46% | 6.95% | 2.870x |
| 96 | 19.37% | 10.12% | 9.81% | 4.20% | 2.352x |
| 128 | 12.77% | 6.45% | 7.89% | 2.70% | 2.114x |

All loads use the same probability Top-10 policy, session URL infinite cache, 64-slot shared Visit pool, adaptive idle fill, exact in-flight promotion, and lowest-probability preemption for authority.
