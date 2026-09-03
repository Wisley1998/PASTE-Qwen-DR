# Pattern-v2 all-visit causal wall replay

Prediction windows are created after every measurable search or visit result.
Visit continuations reuse the causal search cache and update visited state before prediction.
Selector: `blend`; allocation: `cross_fold_budget`.
Effective LLM duration scale: `0.42` (materialized `0.42` × runtime `1.0`).
Wrong-call contention is not modeled; speculative slots remain isolated.

## Coverage inventory

| Trigger | Windows | Next visit windows | Visit URLs | Top-20 candidate-pool hits |
|---|---:|---:|---:|---:|
| search | 340 | 140 | 369 | 299 |
| visit | 188 | 89 | 130 | 62 |

## Wall results

| Budget | Burst cap | C | Selected | No-opt wall | Optimized wall | Full wall speedup | Mean flow reduction | Eligible visit reduction | Authority recall | Spec precision | Call amp. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1 | 244 | 5947.659 s | 5781.338 s | 2.80% | 2.80% | 6.70% | 8.82% | 18.03% | 1.401x |
| 1 | 2 | 8 | 244 | 788.621 s | 765.282 s | 2.96% | 2.80% | 6.70% | 8.82% | 18.03% | 1.401x |
| 1 | 2 | 16 | 244 | 436.482 s | 424.455 s | 2.76% | 2.80% | 6.70% | 8.82% | 18.03% | 1.401x |
| 1 | 2 | 32 | 244 | 283.649 s | 281.196 s | 0.86% | 2.80% | 6.70% | 8.82% | 18.03% | 1.401x |
| 1 | 2 | 64 | 244 | 238.750 s | 238.663 s | 0.04% | 2.80% | 6.70% | 8.82% | 18.03% | 1.401x |
| 1 | 2 | 128 | 244 | 230.837 s | 230.838 s | -0.00% | 2.80% | 6.70% | 8.82% | 18.03% | 1.401x |
| 2 | 4 | 1 | 602 | 5947.659 s | 5576.139 s | 6.25% | 6.25% | 14.96% | 18.24% | 15.12% | 2.024x |
| 2 | 4 | 8 | 602 | 788.621 s | 737.540 s | 6.48% | 6.25% | 14.96% | 18.24% | 15.12% | 2.024x |
| 2 | 4 | 16 | 602 | 436.482 s | 412.368 s | 5.52% | 6.25% | 14.96% | 18.24% | 15.12% | 2.024x |
| 2 | 4 | 32 | 602 | 283.649 s | 279.232 s | 1.56% | 6.25% | 14.96% | 18.24% | 15.12% | 2.024x |
| 2 | 4 | 64 | 602 | 238.750 s | 238.457 s | 0.12% | 6.25% | 14.96% | 18.24% | 15.12% | 2.024x |
| 2 | 4 | 128 | 602 | 230.837 s | 230.840 s | -0.00% | 6.25% | 14.96% | 18.24% | 15.12% | 2.024x |
| 3 | 6 | 1 | 1032 | 5947.659 s | 5416.694 s | 8.93% | 8.93% | 21.38% | 27.66% | 13.37% | 2.792x |
| 3 | 6 | 8 | 1032 | 788.621 s | 716.068 s | 9.20% | 8.93% | 21.38% | 27.66% | 13.37% | 2.792x |
| 3 | 6 | 16 | 1032 | 436.482 s | 401.184 s | 8.09% | 8.93% | 21.38% | 27.66% | 13.37% | 2.792x |
| 3 | 6 | 32 | 1032 | 283.649 s | 277.612 s | 2.13% | 8.93% | 21.38% | 27.66% | 13.37% | 2.792x |
| 3 | 6 | 64 | 1032 | 238.750 s | 238.359 s | 0.16% | 8.93% | 21.38% | 27.66% | 13.37% | 2.792x |
| 3 | 6 | 128 | 1032 | 230.837 s | 230.841 s | -0.00% | 8.93% | 21.38% | 27.66% | 13.37% | 2.792x |
| 4 | 8 | 1 | 1551 | 5947.659 s | 5309.948 s | 10.72% | 10.72% | 25.68% | 34.47% | 11.09% | 3.764x |
| 4 | 8 | 8 | 1551 | 788.621 s | 702.725 s | 10.89% | 10.72% | 25.68% | 34.47% | 11.09% | 3.764x |
| 4 | 8 | 16 | 1551 | 436.482 s | 394.970 s | 9.51% | 10.72% | 25.68% | 34.47% | 11.09% | 3.764x |
| 4 | 8 | 32 | 1551 | 283.649 s | 276.682 s | 2.46% | 10.72% | 25.68% | 34.47% | 11.09% | 3.764x |
| 4 | 8 | 64 | 1551 | 238.750 s | 238.245 s | 0.21% | 10.72% | 25.68% | 34.47% | 11.09% | 3.764x |
| 4 | 8 | 128 | 1551 | 230.837 s | 230.845 s | -0.00% | 10.72% | 25.68% | 34.47% | 11.09% | 3.764x |
| 5 | 10 | 1 | 2093 | 5947.659 s | 5176.266 s | 12.97% | 12.97% | 31.06% | 41.88% | 9.99% | 4.776x |
| 5 | 10 | 8 | 2093 | 788.621 s | 685.264 s | 13.11% | 12.97% | 31.06% | 41.88% | 9.99% | 4.776x |
| 5 | 10 | 16 | 2093 | 436.482 s | 384.926 s | 11.81% | 12.97% | 31.06% | 41.88% | 9.99% | 4.776x |
| 5 | 10 | 32 | 2093 | 283.649 s | 267.675 s | 5.63% | 12.97% | 31.06% | 41.88% | 9.99% | 4.776x |
| 5 | 10 | 64 | 2093 | 238.750 s | 229.534 s | 3.86% | 12.97% | 31.06% | 41.88% | 9.99% | 4.776x |
| 5 | 10 | 128 | 2093 | 230.837 s | 222.483 s | 3.62% | 12.97% | 31.06% | 41.88% | 9.99% | 4.776x |

The no-optimization wall is the same 0.42x-LLM trace replayed without speculative visits. Authority recall and speculative precision are selection metrics, so they remain constant across task concurrency; concurrency changes only the closed-loop makespan schedule.
Authority multi-URL visits execute serially using corrected per-URL service durations. Selected speculative visits start concurrently in isolated slots.
