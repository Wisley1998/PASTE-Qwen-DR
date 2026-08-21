# Prefix development audit

This report is a read-only audit of `prefix_dev_block2_p0_nativeoff`,
`prefix_dev_block2_p1_nativeon`, and `prefix_dev_block2_p2_affinity`.  It uses
development artifacts only.  It does not read or evaluate the formal workload.
The machine-recomputable source is
`reproduction/scripts/audit_live_prefix_development.py`.

## Integrity and comparability

All three cells completed 24/24 tasks, 72/72 single-attempt LLM requests, and
48/48 authoritative tool commits.  Their independent source sets are identical
(SHA256 `b140015ff95503b92237a3ba9e22d2c98d63831be8cb53b5dbe7cdacc07482f4`),
and their frozen workload identities match.

After excluding `cell_label` and the runtime-observed Bing URL coverage field,
the declared P0/P1 configuration differs only at
`VLLM_ENABLE_PREFIX_CACHING`; the declared P1/P2 configuration differs only at
`VLLM_SCHED_JOINT_V2_PREFIX_LOCALITY`.  URL coverage was 17/24, 16/24, and
17/24 respectively and is an observed live-backend outcome, not a controlled
input.

The block2 P0 is nevertheless **not a valid native-cache-off cell**.  Its result declares
`VLLM_ENABLE_PREFIX_CACHING=0`, but its startup log records
`enable_prefix_caching=True`.  It also records 18,400 native cache hits.  P1
and P2 likewise report effective `enable_prefix_caching=True`.  Consequently,
block2 P0/P1 cannot measure native cache benefit, regardless of its E2E point
estimate.  It is retained as rejected engineering evidence.

The subsequent `prefix_dev_block3_p0_nativeoff` is a valid native-off cell.  Its
API and engine startup records both state `enable_prefix_caching=False`; its
native hit/query counters are 0/0 and every rounded server-log hit-rate sample
is 0.0%.  Its declared and effective configurations therefore agree.

Artifact bindings:

| Cell | Result SHA256 | Server-log SHA256 |
|---|---|---|
| P0 | `b14d3c6c45e10f74bbd1a1c4fdae2a45ddee5bdf5a68068545b0ddedaa706ca6` | `a9bec44b516da500df1130efcdfa6e76e68ddb966675340bba70ac58ceee42c4` |
| P1 | `38b14201432cc113ef994d0e7ddc029cea549fa85815cc86a7ee4734e3e77cb0` | `abaf2481d5570fe7ded874b965153ffe019f5f73e1b990297cda07687dc205bf` |
| P2 | `89fa67d80ab909418b6d7084a78836416e9e8d5978bec1d5b61a908af6c686ca` | `02cc4d46dd95da788ffe930daeb00e2846c6e266d59b24980fc4e28e0dd34f03` |
| Corrected P0 (block3) | `0970fa6d6d52badc34cb97be5b2b9531a12fd47b9a3b6561414cbec62f09b48d` | `121048d95965ed5b72058341e18474bd0c2b5cf6ce46af5a78b9c08783135d36` |

## Recomputed metrics

| Metric | P0 declared off | P1 native | P2 native + affinity |
|---|---:|---:|---:|
| Mean task E2E | 52.319369 s | 35.311316 s | 31.883164 s |
| P50 task E2E | 51.538962 s | 35.574089 s | 31.472406 s |
| P95 task E2E | 90.302691 s | 57.297655 s | 52.765033 s |
| Makespan | 93.954847 s | 59.644771 s | 56.127860 s |
| Mean LLM request | 1.639982 s | 1.737287 s | 1.806230 s |
| Server prefill sum | 6.392945 s | 5.439506 s | 5.786466 s |
| Server inference sum | 79.809530 s | 80.989691 s | 86.956001 s |
| Server LLM queue sum | 1.848343 s | 2.898331 s | 2.650302 s |
| Prompt / completion tokens | 47,802 / 3,324 | 47,851 / 3,371 | 48,130 / 3,531 |
| Native hits / queries | 18,400 / 47,802 | 18,432 / 47,851 | 18,512 / 48,130 |
| Native hit ratio | 38.492113% | 38.519571% | 38.462497% |
| Mean tool queue | 21.903328 s | 14.388856 s | 12.637373 s |
| Mean tool service | 1.787604 s | 0.653120 s | 0.586890 s |
| Mean tool exposed wait | 23.691816 s | 15.042330 s | 13.224598 s |

### Corrected native-off diagnostic

| Metric | Corrected P0, block3 | Native P1, block2 |
|---|---:|---:|
| Mean task E2E | 31.918438 s | 35.311316 s |
| P50 / P95 task E2E | 32.504445 / 53.559977 s | 35.574089 / 57.297655 s |
| Makespan | 56.172884 s | 59.644771 s |
| Mean LLM request | 1.728140 s | 1.737287 s |
| Server prefill sum | 6.036895 s | 5.439506 s |
| Server inference sum | 81.171012 s | 80.989691 s |
| Server LLM queue sum | 2.958373 s | 2.898331 s |
| Prompt / completion tokens | 47,607 / 3,352 | 47,851 / 3,371 |
| Native hits / queries | 0 / 0 (disabled) | 18,432 / 47,851 |
| Mean tool queue / service | 12.779828 / 0.578384 s | 14.388856 / 0.653120 s |
| Mean tool exposed wait | 13.359502 s | 15.042330 s |

The corrected P0 and P1 have identical sources and workload identity, and their
declared controlled inputs differ only by the native-cache flag.  They are not,
however, a balanced pair: P1 belongs to block2 and corrected P0 was run later in
block3.  Descriptively, P1 reduces aggregate prefill time by 9.8956% and
inference time by 0.2234%, but mean LLM request time is 0.5293% higher.  P1 mean
task E2E is 3.392878 seconds (10.6298%) slower; only 10/24 sources are faster and
the paired-source saving interval is `[-10.190330, 3.619413]` seconds.

The cross-run task result is again dominated by tool timing: relative to
corrected P0, P1 adds 0.027442 seconds of LLM time and 3.365657 seconds of tool
exposed wait per task.  It is a descriptive diagnostic, not a causal estimate
of native caching.  A balanced, counterordered P0/P1 block would still be
required for such a claim.

P2 emitted 26 rate-limited explicit-locality observations.  Every observation
had exactly one waiting request.  Their counters sum to 26 lookups, zero reused
lookups, 25 hit requests, 9,504 cached tokens, 33,318 prompt tokens, and 23,814
marginal-prefill tokens.  The logged cached/prompt fraction is 28.525122%.
Both `head_changed` and direct input/output-head differences are zero in all 26
observations.  These are one-second-rate-limited snapshots rather than an
exhaustive trace, but they contain no direct evidence of an affinity reorder.

## P1 to P2 attribution

P2's descriptive mean E2E improvement is 3.428152 s or 9.7084%.  P50 improves
11.5300%, P95 improves 7.9107%, and makespan improves 5.8964%.  Sixteen of 24
sources are faster.  The paired-source bootstrap 95% interval for mean saving is
`[-2.427072, 9.388169]` seconds, so its lower bound is not positive.  This
bootstrap also cannot capture shared live-tool or one-block run-order noise.

The native hit ratio changes by **-0.057074 percentage points**, rather than
increasing by the required three points.  LLM mean request time regresses
3.9684%, prefill sum regresses 6.3785%, and inference sum regresses 7.3668%.
Only LLM queue sum improves, by 8.5577%.  P2 also generates 4.7464% more
completion tokens.

The raw event accounting reconstructs mean task E2E almost exactly:

| Per-task component | P1 | P2 | P2 minus P1 |
|---|---:|---:|---:|
| Three LLM calls | 5.211862 s | 5.418691 s | +0.206829 s |
| Two tool exposed waits | 30.084660 s | 26.449195 s | -3.635464 s |
| Residual | 0.014794 s | 0.015277 s | +0.000483 s |

Thus the complete 3.428152-second point improvement is accounted for by lower
live-tool wait while the LLM component becomes slower.  This is consistent with
tool-queue variation, not a measured prefix-affinity gain.  In addition, the
runs are sequential P0/P1/P2 and visit service improves monotonically: P0 visit
service mean/max is 3.236955/12.070674 seconds, versus 0.981720/4.993728 for P1
and 0.818087/1.812069 for P2.  The shared visit capacity is one, so live service
variation is amplified through queue head-of-line blocking.

## Frozen protocol decision

| P2 selection gate relative to P1 | Result |
|---|---|
| Mean task E2E improves at least 2% | Pass (9.7084%) |
| Native prefix-hit ratio rises at least 3 pp | **Fail (-0.0571 pp)** |
| Paired bootstrap lower bound is positive | **Fail (-2.4271 s)** |
| Task P95 remains within 3% of P1 | Pass (it improves 7.9107%) |

The gates are conjunctive.  The selected policy remains **P1: native vLLM
prefix cache without explicit affinity**.  P2 must not be selected or credited
with a prefix gain.  Corrected P0 proves that native-off execution is now real,
but its cross-block comparison is insufficient for a causal native-cache claim.
