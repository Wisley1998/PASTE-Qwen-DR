# Frozen protocol: slot-5 contextual backoff

This follow-up was designed after the separate M1 outer evaluation showed the
expected ranking trade-off (slightly higher Top-5, lower Top-1/3).  Therefore
any result on the same outer sessions is an adaptive post-hoc diagnostic, not
independent confirmation or a basis for a paper claim.  New whole-session
traces are required before online promotion.

## Fixed rule

- Fit no new parameters and do not change either child artifact.
- Positions 1--4 are exactly the legacy rank-only mapper's first four raw URLs.
- Position 5 is the contextual reranker's highest-scored raw URL not already
  in those four.  If none exists, use the legacy fifth URL.
- For K <= 4, return the legacy prefix unchanged.  Exact raw invocation
  equality remains the only hit/promotion rule.
- Inputs remain limited to the current, already-visible search response.

This is a deterministic pattern/backoff composition: the stable global-rank
pattern owns the high-confidence prefix; the contextual matcher is used only
for the residual fifth slot.

## Training-only selection record

The prefix length was selected from `{1,2,3,4}` using the already saved
outer-train grouped-OOF predictions.  Top-5 hits were respectively
`89, 88, 85, 86` out of 148, but only prefixes 3 and 4 structurally preserve
both the legacy Top-1 and Top-3 lists.  Prefix 4 wins the predeclared protected
metrics plus Top-5 rule:

| Policy | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Legacy | 33/148 (22.3%) | 64/148 (43.2%) | 81/148 (54.7%) |
| Prefix-4 backoff | 33/148 (22.3%) | 64/148 (43.2%) | 86/148 (58.1%) |

## Diagnostic gates

- Outer Top-1 and Top-3 must be bit-identical to legacy by construction.
- Outer Top-5 must be strictly higher than legacy.
- End-to-end local prediction p99 and observed maximum must remain below
  100 ms on the current CPU environment.
- All outcomes are retained.  Even if all gates pass, status is
  `prospective_confirmation_required`, not automatically promoted, because
  these outer sessions have already been evaluated.
