# Policy experiments: recorded-duration revision v9

The user changed policy ablations to simulated tool execution on 2026-09-20. This supersedes the manual's requirement to execute physical tools for the policy matrix. Correctness evidence and physical tool-resource measurements remain separately labelled v8/earlier artifacts. No smoke runs or environment restoration are part of v9.

Both original entrypoints still execute real vLLM requests with the frozen token workload. Tools wait asynchronously for the plan's duration under the existing shared worker and speculative budgets. There is no additional clock scaling or instantaneous virtual-time jump: skipping tool time altogether would remove the overlap and GPU requeueing effect being measured. The scheduler receives only current request inputs, completed history, current predictions, and observed submission-time readiness. It never receives assigned future durations, trace suffix lengths, future tool calls, or hit labels. Public generation caps are unchanged from the corrected v8 path.

DR uses its existing learned predictor and broker. A predicted URL receives its first recorded per-session service duration if available; an unrecorded candidate receives the fixed two-second prior. The executor alone reads this timing catalog. Exact invocation/session match, completion state, existing TTL and simulated stable-result validity determine adoption. This does not test physical HTTP freshness or equality of real outputs.

Coding uses its existing timer pool and freezes the existing result-path predictor's emitted candidates from the complete v8 length-aware tool audit. Candidates become available only after their originating authoritative tool completes; neither recorded wall-clock release offsets nor hit/outcome labels are read. Task epochs advance only after the corresponding authoritative tool completes. The pool models completed path+epoch cache reuse, no running-prefetch adoption, no preemption, and retains the authoritative residual timer. Credit is bounded by the recorded source-I/O component minus cache lookup, never the entire subprocess duration. Of 280 candidates in 64 tasks, four have positive credit under this conservative recorded-component model. This timing model is not a new correctness implementation or a prediction-accuracy benchmark.

Models/revisions, arrival scales, plans, Coding 2 tune / 8 test source split, gate settings, LLM estimator, predictor, top-k and budgets are retained. Three rankers × three fresh-server blocks are queued first, with rotated order. Failed/incomplete cells remain visible and cannot enter a paired speedup. Other policy budget/load ablations must use these same timed backends; do not combine timed and physical cells in a pair.

Resource fields are simulated worker occupancy and simulated executed-call counts, with physical CPU/network fields null. Real GPU makespan and per-task E2E are measured. These data can estimate a service-demand tradeoff but cannot establish actual CPU/network cost. The correctness and real-resource reports must retain that distinction.

Frozen source: `source_manifest_v9.json`, SHA256 `2e3f5acff6c81e7eb6cd3ad8539b19c087e36cc5c023b3f9635c99fbff138e88`.
Protocol and original-entrypoint commands: `/home/aiscuser/resubmission-runs/frozen_20260920/timed_policy/`.
Validation: DR 107 tests + 23 subtests; Coding 37 tests; all 64 frozen task audits and 280 candidate prefix bindings validated.

Historical status at launch: matrix running; no v9 performance result yet. The completed real DR v8 pair is retained separately (mean 358.626 → 337.077 s, 6.01% reduction, one descriptive block). Coding v8 tool-aware has only 63/64 successful tasks after an env_ensure timeout and has no valid speedup comparison; its later retry was cancelled when the user switched execution modes.

Additional legacy compatibility suite: 68 passed (including existing timer-pool tests), 21 failed because historical predictor/calibration artifacts and the old batch manifest are absent. This is recorded separately in `timed_policy/validation.json`; the missing legacy experiment framework was not reconstructed.

## Current status after checking completed runs

Coding ordering: all 9 cells (3 rankers × 3 fresh-server blocks) completed, every cell 64/64 tasks and 1,240 LLM requests. Paired checks across all blocks passed for workload, common causal LLM metadata, server gate/configuration, predictor and budget, with no fail-open. Three-block mean E2E: FCFS 115.786 s; length-aware 110.736 s; tool-aware 113.062 s. Tool-aware is 2.35% faster than FCFS and 2.10% slower than length-aware in this timed setup; this does not establish an advantage over length-aware. Detailed aggregate: `coding/three_block_summary.json`.

DR v9 failed before saving its first result because the timed path called close on an absent physical executor. Raw logs are preserved in `excluded/timed_v9_dr_cleanup_failure`; there is no usable saved latency result for that cell. v10 adds only the executor-is-not-None cleanup guard, plus a result-persistence regression test (108 tests + 23 subtests passed). The entire DR ordering matrix has been restarted through the same original entrypoint. Coding source and its v9 results are unchanged.

Budget/load and tuned-cap comparisons under the timed backend are not yet complete. The nine completed Coding ordering cells are not the complete paper experiment matrix.
