# Frozen protocol: contextual exact-URL predictor

Frozen before the first evaluation of the contextual model on the fixed outer
held-out sessions.  The legacy outer baseline numbers and a diagnostic error
audit were already known, so this is a mechanically isolated post-hoc
evaluation, not an untouched confirmatory test.  A new trace collection is
required for fully prospective confirmation.

## Data boundary

- Outer split: the existing deterministic whole-session split,
  `seed=paste-repro-v1`, `train_ratio=0.70` (70 train sessions, 30 held-out
  sessions).
- Selection: deterministic five-fold grouped cross-validation inside the 70
  outer-train sessions.  Fold assignment is
  `int(sha256("contextual-cv-v1\\0" + session_id), 16) % 5`.
- All labels, priors, coefficients, and preprocessing statistics for a fold
  are fit from that fold's training sessions only.
- The contextual model is evaluated once on all fixed outer-held-out sessions.
  The result is retained regardless of direction and is not used to change
  features, regularization, split, or tie-breaking.

## Frozen candidates and exactness

- Both models may emit only a raw URL present in the current, already-visible
  search response.
- Duplicate appearances of the same raw URL are one candidate; aggregate
  duplicate features may use all its current-response appearances.
- URL normalization is used only to compute lexical features.  It does not
  alter the emitted URL, deduplication key, or exact promotion rule.
- The generated LLM decision response, future visit call, future tool result,
  and held-out labels are unavailable to feature extraction and scoring.

## Models

- M0: the existing `URLRankMapper`, trained on outer-train sessions.
- M1: a deterministic same-transition pairwise logistic reranker with L2
  coefficient `lambda=3.0`; the intercept is unpenalized.  Each visible
  positive is paired with every current-response negative, and each pair has
  weight `1 / negative_count`, matching atomic-target evaluation.
- Optimization is deterministic damped Newton iteration (maximum 60
  iterations, fixed tolerance and line search).
- Score ties are resolved by current-response ordinal and then raw URL.

M1's frozen feature schema contains only causal current-response values:

1. intercept;
2. displayed-rank one-hot (ranks 1--5, capped);
3. query-index one-hot (indices 0--6, capped);
4. displayed-rank x query-index interaction (query index capped at 3);
5. log URL occurrence count, fraction of query blocks containing the URL,
   normalized first query position, and reciprocal displayed rank;
6. fixed unigram coverage, bigram coverage, token Jaccard, and decoded-path
   bigram coverage between title/URL and its own query, for both the first and
   best current occurrence;
7. path length, current-response host multiplicity, PDF suffix, and normalized
   query-in-title indicator.

No learned vocabulary, foundation model, external request, or future trace
state is used.

## Selection rule already applied on outer-train only

Primary: pooled out-of-fold exact invocation Recall@5, with an eligibility
gate requiring Recall@1 and Recall@3 to be no lower than M0.  Ties are broken
by Recall@3, Recall@1, stronger regularization, then fewer features.

Frozen pooled OOF results over 148 authoritative URL targets:

| Model | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| M0 rank-only | 33/148 (22.3%) | 64/148 (43.2%) | 81/148 (54.7%) |
| M1 contextual, lambda=3 | 38/148 (25.7%) | 71/148 (48.0%) | 90/148 (60.8%) |

M1 is frozen as the selected challenger; final online retention is governed by
the predeclared outer promotion gate below.

## Online-overhead acceptance gate

- Offline fitting time is excluded; the deployed artifact contains the 49
  coefficients.
- The measured path is end-to-end local prediction from the already-returned
  structured search object: adapter conversion, fixed feature extraction,
  49-dimensional scoring, exact-URL deduplication, sorting, and Top-5 output.
- The acceptance benchmark covers all 245 outer-train search-decision input
  shapes, warms each input once, then runs 20 deterministic passes (4,900
  calls).  It performs no network request and its input extractor does not
  inspect the next event, future labels, or the generated decision response.
- Acceptance requires both observed p99 and observed maximum latency below
  100 ms on the current CPU environment.
- M1 is retained for online use only if the latency gate passes, outer
  Recall@5 is strictly greater than M0, and outer Recall@1 and Recall@3 are
  each no lower than M0.  The challenger artifact and all results are retained
  even if this predeclared promotion gate rejects it; no feature or weight is
  changed after seeing the result.

## Outer metrics

- Primary: paired whole-session change in exact target hits and Recall@5.
- Secondary: Recall@1/3, example hit rate, precision, session-macro recall,
  exact visible coverage, budget-specific oracle ceiling, and replay-derived
  exposed-stall reduction.
- Uncertainty: deterministic session-cluster bootstrap over transition-bearing
  held-out sessions.  Atomic URLs are not treated as independent samples.
- Candidate-cache and URL-alias generation are reported separately as future
  or ablation opportunities; they cannot change M1 after this freeze.
- Before any held-out label is scored, the runner atomically writes a STARTED
  manifest binding the frozen protocol, code, original split-manifest checksum
  (`1bf8984620a1a6eb5c4472dce76ed5039eb37ccb28c2e03ccdf460eff0425402`),
  all trace hashes, the OOF result, and the challenger artifact.  A STARTED
  marker blocks an unacknowledged rerun even if the first process fails before
  writing final outputs.
