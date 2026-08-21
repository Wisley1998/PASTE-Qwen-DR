# Speculative Tool Execution (without LLM co-design)

Date: 2026-08-15

This is the isolated, causal reproduction of trace-learned speculative tool
execution. It does not use oracle future calls, frozen authoritative URLs, or
an LLM scheduling policy.

## Protocol

- Source: 100 session files in `traces/my_traces`.
- Split: deterministic whole-session 70/30 split with seed
  `paste-repro-v1`; the model artifact records every file SHA-256.
- Training signal: the displayed search-result rank selected by the direct
  next `visit` in training sessions.
- Inference input: only URLs in the current, already-visible search response.
- Prediction: learned ranks are late-bound to those current URLs (`top_k=5`).
- Confirmation: only an exact authoritative URL invocation can reuse or
  promote isolated speculative work; misses execute normally.
- Latency bound: a hit hides at most the smaller of the recorded visit stall
  and preceding LLM inference window, scaled by the exact hit fraction for
  batched visits.

## Held-out result

| Metric | Result |
|---|---:|
| Train / held-out sessions | 70 / 30 |
| Train / held-out direct search→visit examples | 82 / 34 |
| Held-out authoritative URL invocations | 88 |
| Search-response executable coverage | 70 / 88 (79.55%) |
| Top-5 transition/example hits | 26 / 34 (76.47%) |
| Top-5 authoritative invocation hits | 49 / 88 (55.68%) |
| Baseline exposed tool stall | 38.514 s |
| Optimized exposed tool stall | 19.647 s |
| Saved stall | 18.866 s (48.99%) |
| Authoritative state-isolation violations | 0 |

The bounded replay admitted 170 predictions: 49 exact matches (23 completed
result reuses and 26 in-flight promotions), 39 authoritative misses, and 121
expired unused predictions. Waste is retained in the report because it is part
of the capacity tradeoff.

The checked-in machine-readable snapshot remains available as
[analysis.json](../tool_only/analysis.json),
[replay.json](../tool_only/replay.json), and the checksummed
[url_rank_mapper.json](../tool_only/url_rank_mapper.json). The first-class
runner regenerates the equivalent model and evaluation under
`reproduction/artifacts/speculative_tool_execution/` using schema
`paste_repro.speculative_tool_execution`.

## Interpretation boundary

The 48.99% result is computed from recorded trace timestamps; it does not
re-fetch the held-out pages. This deliberately isolates prediction, exact-match
confirmation, bounded overlap, miss fallback, and state isolation from network
variance and all LLM-side co-design. The optional live integration uses the
same predictor against a structured live search response, while leaving the
authoritative URL choice to the LLM.
