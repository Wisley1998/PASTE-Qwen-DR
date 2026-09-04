# Pattern V2 strict A/B/E/F live analysis

Scope: `retrospective_internal_holdout_live_pilot`; confirmatory claim allowed: **no**.

| Cell | Makespan (s) | Mean flow (s) | P95 flow (s) | Visit hit | Call amp. |
|---|---:|---:|---:|---:|---:|
| A | 812.657 | 420.779 | 757.929 | 0.00% | 1.000x |
| B | 646.441 | 347.857 | 629.030 | 76.97% | 6.888x |
| E | 831.374 | 434.217 | 780.580 | 0.00% | 1.000x |
| F | 652.296 | 351.576 | 631.689 | 77.19% | 6.884x |

| Contrast | Makespan speedup | Paired-root mean-flow speedup (95% CI) |
|---|---:|---:|
| B_vs_A_tool_only | 20.45% | 17.33% ([16.42%, 18.31%]) |
| E_vs_A_scheduler_only | -2.30% | -3.19% ([-3.25%, -3.14%]) |
| F_vs_E_tool_incremental | 21.54% | 19.03% ([18.17%, 19.96%]) |
| F_vs_A_combined | 19.73% | 16.45% ([15.51%, 17.40%]) |
| F_vs_B_scheduler_incremental | -0.91% | -1.07% ([-1.12%, -1.02%]) |

All cells executed identical 210 tasks, 1,785 LLM requests, and 1,302 authoritative tool calls.
