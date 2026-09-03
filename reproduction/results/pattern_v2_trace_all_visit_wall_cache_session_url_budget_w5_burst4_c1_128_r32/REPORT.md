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
| 5 | 20 | 1 | 1881 | 5947.659 s | 4650.733 s | 21.81% | 21.81% | 52.22% | 60.52% | 16.06% | 4.164x |
| 5 | 20 | 8 | 1881 | 788.621 s | 619.883 s | 21.40% | 21.81% | 52.22% | 60.52% | 16.06% | 4.164x |
| 5 | 20 | 16 | 1881 | 436.482 s | 357.250 s | 18.15% | 21.81% | 52.22% | 60.52% | 16.06% | 4.164x |
| 5 | 20 | 32 | 1881 | 283.649 s | 262.800 s | 7.35% | 21.81% | 52.22% | 60.52% | 16.06% | 4.164x |
| 5 | 20 | 64 | 1881 | 238.750 s | 228.962 s | 4.10% | 21.81% | 52.22% | 60.52% | 16.06% | 4.164x |
| 5 | 20 | 128 | 1881 | 230.837 s | 222.469 s | 3.62% | 21.81% | 52.22% | 60.52% | 16.06% | 4.164x |

## Persistent speculative cache

| Budget | Policy selections | Physical starts | Deduplicated | Cache hits | Ready | In-flight | Earlier-decision hits | Incremental future hits | Wait tail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2648 | 1881 | 767 | 302 | 239 | 63 | 132 | 68 | 193.201 s |

The cache is URL-keyed within one session, has infinite TTL, zero read cost, and no content expiration. Running jobs are singleflight-claimed; completed speculative results persist across later decisions. Authority results do not populate this cache.

The no-optimization wall is the same 0.42x-LLM trace replayed without speculative visits. Authority recall and speculative precision are fixed replay outcomes, so they remain constant across task concurrency; concurrency changes only the closed-loop makespan schedule.
Authority multi-URL visits execute serially using corrected per-URL service durations. Selected speculative visits start concurrently in isolated slots.
