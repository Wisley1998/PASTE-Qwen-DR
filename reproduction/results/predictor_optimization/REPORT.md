# Contextual exact-URL predictor optimization

## Result

The frozen current-response contextual reranker improves the grouped training evidence and was evaluated once on the unchanged whole-session outer split. It remains a **conditional URL reranker**: these Top-K metrics include only search decisions whose next tool is an authoritative visit.

The predeclared promotion gate **rejected M1 and retained M0**.

### Fixed outer heldout (88 authoritative URL targets)

| Model | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| M0 rank-only | 17/88 (19.3%) | 38/88 (43.2%) | 49/88 (55.7%) |
| M1 contextual | 16/88 (18.2%) | 36/88 (40.9%) | 50/88 (56.8%) |

Paired changes by whole held-out session:

| K | Delta hits | Delta recall | Session-bootstrap 95% interval | Sign-flip p |
|---:|---:|---:|---:|---:|
| 1 | -1 | -1.1 pp | [-4.7, +2.6] pp | 1.0000 |
| 3 | -2 | -2.3 pp | [-6.5, +2.1] pp | 0.6250 |
| 5 | +1 | +1.1 pp | [-4.5, +7.5] pp | 1.0000 |

This is post-hoc method development with a mechanically isolated outer run, not a pristine confirmatory test: the old baseline and aggregate error audit were already visible. A newly collected whole-session trace set is required for confirmation.

## Training-only selection

Five-fold CV was grouped by session inside the original 70 outer-train sessions. The model and `lambda=3` were frozen before the contextual outer evaluation.

| Model | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| M0 rank-only | 33/148 (22.3%) | 64/148 (43.2%) | 81/148 (54.7%) |
| M1 contextual | 38/148 (25.7%) | 71/148 (48.0%) | 90/148 (60.8%) |

M1 uses only data visible before the decision LLM starts generating: displayed rank, query block/position, duplicate appearances, title-query and path-query overlap, path shape, host multiplicity, and PDF suffix. It emits the original raw URL and keeps exact invocation confirmation unchanged.

## Where the remaining headroom is

- Current-response exact coverage: `70/88 = 79.5%`.
- Current-response Top-5 oracle: `69/88 = 78.4%`.
- All causally prior search responses in the decision input cover `84/88 = 95.5%`; their Top-5 oracle is `79/88 = 89.8%`.
- A bounded recency-aware history cache remains a separate challenger; this frozen M1 ranks only the current response.

## Runtime interpretation

The end-to-end local prediction path took `3.257 ms` p50, `7.920 ms` p95, `11.347 ms` p99, and `24.569 ms` maximum over 4,900 calls. The frozen `<100 ms` p99-and-maximum gate therefore **passed**.

At Top-5, trace-counterfactual exposed stall changes from `38.514s → 19.647s` for M0 and `38.514s → 20.174s` for M1. The corresponding reductions are `49.0%` and `47.6%`.

The autonomous all-search audit is intentionally harsher:

| Model | Search decisions | Blind Top-5 predictions | Exact hits | All-window precision | Non-visit predictions |
|---|---:|---:|---:|---:|---:|
| M0 | 95 | 475 | 49 | 10.3% | 305 |
| M1 | 95 | 475 | 50 | 10.5% | 305 |

A next-tool `visit`/abstain gate is therefore required before deploying this as an autonomous post-search policy. The pairwise relative score is not a calibrated admission probability.

## Reproduce

```bash
python reproduction/scripts/run_predictor_optimization.py \
  --output /tmp/predictor_optimization_reproduction \
  --protocol reproduction/results/predictor_optimization/FROZEN_PROTOCOL.md
```

The model artifact, aggregate JSON, paired prediction CSV, provenance, and completion manifest are in this directory. Promotion remains exact raw `visit({url: ...})` equality; no HTTP/HTTPS or encoding equivalence is assumed.
