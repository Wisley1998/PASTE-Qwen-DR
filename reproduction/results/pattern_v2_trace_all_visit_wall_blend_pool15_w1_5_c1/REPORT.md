# Pattern-v2 all-visit causal wall replay

Prediction windows are created after every measurable search or visit result.
Visit continuations reuse the causal search cache and update visited state before prediction.
Wrong-call contention is not modeled; speculative slots remain isolated.

## Coverage inventory

| Trigger | Windows | Next visit windows | Visit URLs | Top-15 candidate-pool hits |
|---|---:|---:|---:|---:|
| search | 340 | 140 | 369 | 277 |
| visit | 188 | 89 | 130 | 59 |

## Wall results

| Width | C | Full wall speedup | Mean flow reduction | Eligible visit reduction | Authority recall | Spec precision | Call amp. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.55% | 2.55% | 7.91% | 15.03% | 14.15% | 1.912x |
| 2 | 1 | 4.65% | 4.65% | 14.45% | 24.65% | 11.60% | 2.878x |
| 3 | 1 | 6.36% | 6.36% | 19.73% | 32.46% | 10.20% | 3.860x |
| 4 | 1 | 8.07% | 8.07% | 25.05% | 39.08% | 9.21% | 4.854x |
| 5 | 1 | 9.15% | 9.15% | 28.42% | 43.49% | 8.20% | 5.870x |
