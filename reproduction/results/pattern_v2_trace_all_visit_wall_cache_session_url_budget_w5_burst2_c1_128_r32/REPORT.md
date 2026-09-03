# Pattern-v2 all-visit causal wall replay

Prediction windows are created after every measurable search or visit result.
Visit continuations reuse the causal search cache and update visited state before prediction.
Selector: `blend`; allocation: `cross_fold_budget`.
Speculative result cache: `session_url`.
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
| 5 | 10 | 1 | 1502 | 5947.659 s | 4784.645 s | 19.55% | 19.55% | 46.83% | 55.51% | 18.44% | 3.455x |
| 5 | 10 | 8 | 1502 | 788.621 s | 636.180 s | 19.33% | 19.55% | 46.83% | 55.51% | 18.44% | 3.455x |
| 5 | 10 | 16 | 1502 | 436.482 s | 364.099 s | 16.58% | 19.55% | 46.83% | 55.51% | 18.44% | 3.455x |
| 5 | 10 | 32 | 1502 | 283.649 s | 263.429 s | 7.13% | 19.55% | 46.83% | 55.51% | 18.44% | 3.455x |
| 5 | 10 | 64 | 1502 | 238.750 s | 229.055 s | 4.06% | 19.55% | 46.83% | 55.51% | 18.44% | 3.455x |
| 5 | 10 | 128 | 1502 | 230.837 s | 222.468 s | 3.63% | 19.55% | 46.83% | 55.51% | 18.44% | 3.455x |

## Persistent speculative cache

| Budget | Policy selections | Physical starts | Deduplicated | Cache hits | Ready | In-flight | Earlier-decision hits | Incremental future hits | Wait tail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2093 | 1502 | 591 | 277 | 211 | 66 | 116 | 68 | 201.371 s |

The cache is URL-keyed within one session, has infinite TTL, zero read cost, and no content expiration. Running jobs are singleflight-claimed; completed speculative results persist across later decisions. Authority results do not populate this cache.

The no-optimization wall is the same 0.42x-LLM trace replayed without speculative visits. Authority recall and speculative precision are fixed replay outcomes, so they remain constant across task concurrency; concurrency changes only the closed-loop makespan schedule.
Authority multi-URL visits execute serially using corrected per-URL service durations. Selected speculative visits start concurrently in isolated slots.
