# Strict Pattern V2 hashed-SLO CPU sensitivity

This is a deterministic systems sensitivity replay, not a live-LLM or autonomous-agent result. Candidate inference is online and causal. Tool service comes only from the sealed normalized-invocation hashed SLO clock; recorded trace tool durations are never read. Recorded 0.42x LLM durations are environment-only proxy service and never enter the policy.

Policy coverage: 358/499 = 71.74%.

| C | Realized hit | Tool stall reduction | E2E speedup | Mean-flow speedup | Call amp. |
|---:|---:|---:|---:|---:|---:|
| 1 | 71.74% | 58.54% | 24.55% | 24.55% | 5.752x |
| 2 | 71.74% | 58.54% | 24.45% | 24.55% | 5.752x |
| 4 | 71.74% | 58.54% | 24.44% | 24.55% | 5.752x |
| 8 | 71.74% | 58.54% | 23.86% | 24.54% | 5.752x |
| 16 | 70.32% | 57.12% | 22.58% | 23.95% | 5.731x |
| 32 | 53.69% | 42.11% | 14.66% | 17.66% | 4.185x |
| 48 | 42.17% | 32.19% | 9.51% | 13.50% | 3.205x |
| 64 | 33.47% | 25.01% | 6.22% | 10.49% | 2.723x |
| 96 | 20.70% | 14.56% | 3.65% | 6.11% | 2.269x |
| 128 | 13.34% | 9.14% | 3.44% | 3.84% | 2.045x |

All loads use the same probability Top-10 policy, session URL infinite cache, 64-slot shared Visit pool, adaptive idle fill, exact in-flight promotion, and lowest-probability preemption for authority.
