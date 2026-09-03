# Reviewer common comment 2: metric audit and load × budget stress

## Bottom line

The three percentages in the review are not one internally inconsistent measurement. The exact strings `27.8%` and `43.9%` do not occur as prediction metrics in the three repositories. The closest reproducible values are Gemini safe-target tool-name Top-1 8/28=28.6% and Qwen exact-URL Top-3 38/88=43.2%; they have different targets and denominators and must not be combined.

The literal 93.8% is a Virtual-Lab URL-prefetch **coverage** claim at N=4. The same source reports a 67.8% per-prefetch hit rate at N=4; therefore 93.8% is not an overall speculative-execution hit rate. The repository does not contain the original 33-trace/321-selection analysis script or a frozen result table, so that legacy 93.8% claim cannot be independently regenerated from the checked-in code.

A separate, SHA-bound Virtual-Lab Tongyi artifact happens to reproduce exact-URL Top-1 15/16=93.8% under LOSO. It is reported separately and is not used as provenance for the legacy prefetch claim.

## Unified definitions

| Name | Numerator | Denominator | Load dependent? |
|---|---|---|---|
| Top-k target recall | authoritative targets whose exact target (or, for Gemini only, name) is in first k | all held-out authoritative targets | no |
| Unthrottled selected-prediction precision | selected predictions that match at least one target in the same decision window | all predictions selected at budget k before any load admission; these are names only for Gemini | no |
| Realized target coverage | targets matched by a candidate actually admitted under the load cap | all held-out targets | yes |
| Prefetch coverage (legacy 93.8%) | selection events with at least one useful prefetched URL | selection events | no, unless admission is modeled |

## Reproduced source metrics

| Dataset and scope | Budget | Targets hit | Target recall | Useful/selected predictions | Unthrottled selected-prediction precision | Hit/target windows | Decision hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen_url_exact_heldout (exact URL invocation / authoritative URL target) | 1 | 17/88 | 19.3% | 17/34 | 50.0% | 17/34 | 50.0% |
| qwen_url_exact_heldout (exact URL invocation / authoritative URL target) | 3 | 38/88 | 43.2% | 38/102 | 37.3% | 25/34 | 73.5% |
| qwen_url_exact_heldout (exact URL invocation / authoritative URL target) | 5 | 49/88 | 55.7% | 49/170 | 28.8% | 26/34 | 76.5% |
| virtual_tongyi_url_exact_loso (exact URL invocation / physically successful fetch) | 1 | 15/16 | 93.8% | 15/16 | 93.8% | 15/16 | 93.8% |
| virtual_tongyi_url_exact_loso (exact URL invocation / physically successful fetch) | 3 | 16/16 | 100.0% | 16/31 | 51.6% | 16/16 | 100.0% |
| virtual_tongyi_url_exact_loso (exact URL invocation / physically successful fetch) | 5 | 16/16 | 100.0% | 16/31 | 51.6% | 16/16 | 100.0% |
| gemini_safe_tool_name_heldout (tool-name ranking only / safe-local target; not exact arguments or promotion) | 1 | 8/28 | 28.6% | 8/81 | 9.9% | 8/28 | 28.6% |
| gemini_safe_tool_name_heldout (tool-name ranking only / safe-local target; not exact arguments or promotion) | 3 | 24/28 | 85.7% | 24/243 | 9.9% | 24/28 | 85.7% |
| gemini_safe_tool_name_heldout (tool-name ranking only / safe-local target; not exact arguments or promotion) | 5 | 27/28 | 96.4% | 27/382 | 7.1% | 27/28 | 96.4% |

Qwen's 34 opportunities are search-to-visit decision windows across 19 eligible held-out sessions. Batched visits expand to 88 atomic authoritative URL targets; therefore decision hit rate uses 34 windows while target recall uses 88 URL invocations.

Gemini's 81 opportunities are all held-out next-tool windows, and the name ranker emits selected names in all 81. Only 28 targets are safe-local, so safe target recall and decision hit rate use 28 while selected-prediction precision counts predictions from all 81 windows. The separate all-target name diagnostic is 39/81 Top-1 and 71/81 Top-3. Gemini candidates contain no executable arguments. The repository's causal replay finds only 2/28 exact-match opportunities, so name Top-k must never be presented as committed promotion hit rate.

## Low-predictability Qwen stress under throttling

The Qwen held-out set is the primary low-predictability stress: exact URL Top-1 is 19.3% and Top-3 is 43.2%. The table applies a global, rank-first residual admission quota after authoritative work. A 90% throttle means only 10% of requested speculative candidates are admitted. It is a deterministic trace-replay envelope, not a GPU benchmark.

| Budget | Throttle | Admitted/requested | Realized target coverage | Retained vs unthrottled | Qwen stall reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 0% | 34.0/34.0 | 19.3% | 100.0% | 23.5% |
| 1 | 50% | 17.0/34.0 | 9.9% | 51.0% | 12.1% |
| 1 | 75% | 8.0/34.0 | 4.7% | 24.1% | 5.7% |
| 1 | 90% | 3.0/34.0 | 1.7% | 9.0% | 2.0% |
| 1 | 100% | 0.0/34.0 | 0.0% | 0.0% | 0.0% |
| 3 | 0% | 102.0/102.0 | 43.2% | 100.0% | 42.1% |
| 3 | 50% | 51.0/102.0 | 27.3% | 63.3% | 30.3% |
| 3 | 75% | 25.0/102.0 | 14.1% | 32.7% | 17.6% |
| 3 | 90% | 10.0/102.0 | 5.8% | 13.5% | 7.3% |
| 3 | 100% | 0.0/102.0 | 0.0% | 0.0% | 0.0% |
| 5 | 0% | 170.0/170.0 | 55.7% | 100.0% | 49.0% |
| 5 | 50% | 85.0/170.0 | 39.3% | 70.7% | 39.7% |
| 5 | 75% | 42.0/170.0 | 23.2% | 41.6% | 26.7% |
| 5 | 90% | 17.0/170.0 | 9.9% | 17.7% | 12.1% |
| 5 | 100% | 0.0/170.0 | 0.0% | 0.0% | 0.0% |

At saturation (100% throttle) realized coverage and extra work both go to zero: the mechanism falls back to authoritative execution. Thus the defensible claim is graceful degradation, not preservation of a 93.8% number under arbitrary load. Moderate residual capacity still converts some low-predictability Top-k signal into overlap; wider budgets help only while candidate admission remains available.

## Real scheduler admission stress (CPU-only)

This test uses `paste_repro.scheduler.SpeculativeScheduler`, exact session-scoped confirmation, eight active/pending speculative slots, synthetic 5 ms tool service, and 2.5 ms prediction lead. Authoritative misses bypass the speculative semaphore by implementation contract. Timing values validate the harness only and are not paper performance results.

| Budget | Concurrent opportunities | Admission | Exact realized coverage | Capacity rejects | State-isolation violations |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100.0% | 19.3% | 0 | 0 |
| 1 | 8 | 100.0% | 18.8% | 0 | 0 |
| 1 | 32 | 25.0% | 5.4% | 144 | 0 |
| 3 | 1 | 100.0% | 43.2% | 0 | 0 |
| 3 | 8 | 33.3% | 18.8% | 240 | 0 |
| 3 | 32 | 8.3% | 5.4% | 528 | 0 |
| 5 | 1 | 100.0% | 55.7% | 0 | 0 |
| 5 | 8 | 20.0% | 18.8% | 480 | 0 |
| 5 | 32 | 5.0% | 5.4% | 912 | 0 |

## Limits and claim boundary

- The Qwen replay uses real held-out targets, candidate order, and traced stall/overlap windows, but the admission envelope is analytical.
- The runtime stress uses the real bounded scheduler and real prediction labels but synthetic milliseconds; it is not an end-to-end vLLM/GPU run.
- This CPU sweep and the GPU/live evidence remain separate tiers. The validated Tongyi target/high results are reported in `../scheduler_robustness/REPORT.md`; the Granite portability attempt failed closed in baseline A and produced no A/E latency result.
- The legacy Virtual-Lab prefetch 93.8% statement has only an in-code summary, not its original analysis artifact. It should be replaced in the rebuttal by a table with explicit numerator/denominator or removed.
- Virtual-Lab's older `reproduction/artifacts/predictor.json` has 26/49 Top-3 in its frozen payload, but the current strict trainer applied to the old ScholarQA trace directory finds no physically bound fetch targets. That legacy payload should not be advertised as freshly reproduced.

## Reproduction

```bash
python3 reproduction/scripts/run_reviewer_comment2_sweep.py --qwen-root /home/aiscuser/PASTE-Qwen-DR --virtual-root /home/aiscuser/virtual-lab-PASTE --gemini-root /home/aiscuser/gemini-cli-PASTE
```

Repository revisions:

- `PASTE-Qwen-DR`: `83e018557566c78e5d499dae5bfd1a877b66eef2`
- `virtual-lab-PASTE`: `15f3bc3227892cf0d5d96c3b9e5ed1d63ca74a8f`
- `gemini-cli-PASTE`: `0cbb8bb05910f3fa3d0d0cb29630af47c871b98d`

`provenance.json` is a non-exhaustive index of direct evaluation inputs, not a claim that one file contains every training-source SHA-256. Complete Qwen and recomputed Gemini train/split manifests are referenced through `metric_audit.json`; provenance also records paths and file hashes for the frozen Gemini and Virtual-Lab lineage manifests. Full sweep cells and deterministic admission-rotation ranges are in the JSON/CSV files beside this report.
