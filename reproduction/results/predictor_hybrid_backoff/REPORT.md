# Slot-5 contextual backoff diagnostic

The deterministic hybrid is rejected. It preserves the legacy first four
positions and uses the contextual reranker only for the fifth slot.

| Policy | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Legacy rank-only | 17/88 (19.3%) | 38/88 (43.2%) | 49/88 (55.7%) |
| Prefix-4 contextual backoff | 17/88 (19.3%) | 38/88 (43.2%) | 48/88 (54.5%) |

The protected Top-1/3 metrics are bit-identical by construction, but Top-5
loses one exact target. This follow-up used outer sessions that had already
been evaluated, so it is adaptive post-hoc evidence only; even a positive
result would have required prospective confirmation.

No child model or exact raw-URL confirmation rule was changed. The deployed
choice remains the legacy rank-only mapper.
