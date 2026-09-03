# Pattern-v2 all-visit causal wall replay

Prediction windows are created after every measurable search or visit result.
Visit continuations reuse the causal search cache and update visited state before prediction.
Wrong-call contention is not modeled; speculative slots remain isolated.

## Coverage inventory

| Trigger | Windows | Next visit windows | Visit URLs | Top-5 candidate hits |
|---|---:|---:|---:|---:|
| search | 340 | 140 | 369 | 299 |
| visit | 188 | 89 | 130 | 62 |

## Wall results

| Width | C | Full wall speedup | Mean flow reduction | Eligible visit reduction | Authority recall | Spec precision | Call amp. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 8.52% | 8.52% | 26.45% | 42.28% | 8.37% | 5.627x |
| 10 | 1 | 11.36% | 11.36% | 35.26% | 53.51% | 5.30% | 10.565x |
| 15 | 1 | 14.10% | 14.10% | 43.77% | 63.93% | 4.26% | 15.385x |
| 20 | 1 | 15.73% | 15.73% | 48.83% | 69.34% | 3.55% | 19.860x |
