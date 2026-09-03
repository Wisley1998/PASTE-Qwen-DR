# Pattern-v2 all-visit causal wall replay

Prediction windows are created after every measurable search or visit result.
Visit continuations reuse the causal search cache and update visited state before prediction.
Selector: `blend`; allocation: `per_decision`.
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
| 10 | 10 | 1 | 2706 | 5947.659 s | 4503.712 s | 24.28% | 24.28% | 58.14% | 71.94% | 13.27% | 5.703x |
| 10 | 10 | 8 | 2706 | 788.621 s | 599.937 s | 23.93% | 24.28% | 58.14% | 71.94% | 13.27% | 5.703x |
| 10 | 10 | 16 | 2706 | 436.482 s | 347.525 s | 20.38% | 24.28% | 58.14% | 71.94% | 13.27% | 5.703x |
| 10 | 10 | 32 | 2706 | 283.649 s | 256.287 s | 9.65% | 24.28% | 58.14% | 71.94% | 13.27% | 5.703x |
| 10 | 10 | 64 | 2706 | 238.750 s | 223.797 s | 6.26% | 24.28% | 58.14% | 71.94% | 13.27% | 5.703x |
| 10 | 10 | 128 | 2706 | 230.837 s | 217.590 s | 5.74% | 24.28% | 58.14% | 71.94% | 13.27% | 5.703x |

## Persistent speculative cache

| Budget | Policy selections | Physical starts | Deduplicated | Cache hits | Ready | In-flight | Earlier-decision hits | Incremental future hits | Wait tail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5245 | 2706 | 2539 | 359 | 257 | 102 | 159 | 62 | 330.044 s |

The cache is URL-keyed within one session, has infinite TTL, zero read cost, and no content expiration. Running jobs are singleflight-claimed; completed speculative results persist across later decisions. Authority results do not populate this cache.

The no-optimization wall is the same 0.42x-LLM trace replayed without speculative visits. Authority recall and speculative precision are fixed replay outcomes, so they remain constant across task concurrency; concurrency changes only the closed-loop makespan schedule.
Authority multi-URL visits execute serially using corrected per-URL service durations. Selected speculative visits start concurrently in isolated slots.
