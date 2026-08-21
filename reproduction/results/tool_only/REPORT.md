# Tool-only functional result

Date: 2026-08-15

This is the small, causal acceptance test for the trace analyzer and tool
speculation scheduler. It is not an exhaustive paper-result reproduction.

## Protocol

- Source: the 100 session files in `traces/my_traces`.
- Split: deterministic whole-session 70/30 split with seed
  `paste-repro-v1`; every file SHA-256 is stored in the model artifact.
- Training signal: which displayed search-result ranks flowed into the direct
  next `visit` call. No evaluation URL is memorized.
- Prediction: learned ranks are late-bound to URLs in the current search
  response; top-k is 5.
- Confirmation: only exact URL invocation matches can reuse or promote a
  speculative result. Misses run the authoritative invocation.
- Latency model: for a confirmed URL, hide at most the smaller of recorded
  visit stall and the preceding decision-LLM inference window, scaled by the
  exact hit fraction of a batched visit.

## Result

| Metric | Result |
|---|---:|
| Train / held-out sessions | 70 / 30 |
| Held-out sessions with measured direct transitions | 19 / 30 |
| Train / held-out direct search→visit examples | 82 / 34 |
| Held-out authoritative URL invocations | 88 |
| Search-response executable coverage | 70 / 88 (79.55%) |
| Top-5 example hit rate | 26 / 34 (76.47%) |
| Top-5 invocation hit rate | 49 / 88 (55.68%) |
| Baseline exposed tool stall | 38.514 s |
| Optimized exposed tool stall | 19.647 s |
| Saved stall | 18.866 s (48.99%) |
| Authoritative state-isolation violations | 0 |

The bounded scheduler replay admitted 170 predictions. It confirmed 49 exact
matches (23 completed-result reuses and 26 in-flight promotions), executed 39
misses authoritatively, and expired 121 unused predictions. The latter is
reported explicitly because speculation has a real waste/capacity tradeoff.

Machine-readable inputs and results are in
[`url_rank_mapper.json`](url_rank_mapper.json), [`analysis.json`](analysis.json),
and [`replay.json`](replay.json).

## Interpretation boundary

The latency numbers are calculated from recorded trace timestamps and do not
re-fetch 88 web pages. This isolates the tool-side mechanism from network
variance. The repository's current `visit` executor ignores `goal`, so this
test treats one URL fetch as the execution-affecting atomic invocation. If that
executor changes, its canonical argument boundary must change with it.
