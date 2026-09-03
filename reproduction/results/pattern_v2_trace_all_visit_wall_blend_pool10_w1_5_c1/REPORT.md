# Pattern-v2 all-visit causal wall replay

Prediction windows are created after every measurable search or visit result.
Visit continuations reuse the causal search cache and update visited state before prediction.
Wrong-call contention is not modeled; speculative slots remain isolated.

## Coverage inventory

| Trigger | Windows | Next visit windows | Visit URLs | Top-10 candidate-pool hits |
|---|---:|---:|---:|---:|
| search | 340 | 140 | 369 | 244 |
| visit | 188 | 89 | 130 | 52 |

## Wall results

| Width | C | Full wall speedup | Mean flow reduction | Eligible visit reduction | Authority recall | Spec precision | Call amp. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.65% | 2.65% | 8.23% | 15.63% | 14.72% | 1.906x |
| 2 | 1 | 4.43% | 4.43% | 13.77% | 24.25% | 11.42% | 2.882x |
| 3 | 1 | 5.98% | 5.98% | 18.57% | 31.86% | 10.01% | 3.866x |
| 4 | 1 | 7.70% | 7.70% | 23.92% | 37.88% | 8.92% | 4.866x |
| 5 | 1 | 9.29% | 9.29% | 28.85% | 44.49% | 8.39% | 5.858x |
