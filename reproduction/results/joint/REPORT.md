# Joint + learned-overlap stress120 load-sensitivity result

Date: 2026-08-15

This is the repository's original stress120 Joint reference result. It is a
three-replicate, fresh-server, fixed-workload A/D **load-sensitivity**
experiment—not an untouched final evaluation and not a reproduction of the
paper's full system. The machine executes a live vLLM server, but recorded tool
waits are replayed; no live web tool is called.

A later, more strongly queued stress180 profile reaches a replicated 31.419%
mean-task reduction with a material request-tail tradeoff. It is reported
separately in [`QUEUED_STRESS180_REPORT.md`](QUEUED_STRESS180_REPORT.md); the
stress120 protocol and numbers below remain unchanged for provenance.

## What is compared

| Cell | vLLM scheduler | Tool-overlap mode |
|---|---|---|
| A | `fcfs` | `none` |
| D | `online_joint_pacer_v2` | checksummed `learned` top-5 mapper |

The reported effect is therefore the combined A/D treatment. “Causal” below
means the scheduler only receives calibration and completed-call information
available at its decision time; it does **not** mean this two-cell comparison
identifies a causal contribution from each component.

## Frozen protocol

- Model/runtime: `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` revision
  `4b0ac5767427a55d08a254f0367e2934976598e0`, vLLM 0.10.1, PyTorch
  2.7.1+cu126, TP=4 BF16 on 4 × A100-SXM4-40GB, 16K context.
- Configuration: [`../../configs/joint_stress.env.example`](../../configs/joint_stress.env.example), including `max-num-seqs=64` and target64 scheduler settings.
  The fixed stress manifest SHA-256 is
  `f3c179a2e5a84b928dc1ffd17254173e9363d5a0ab78bf70fdd21f85dccd3399`;
  the learned mapper SHA-256 is
  `d4ac5ee9cebcb328ec153192fe4d78508cafd9dcff09cea5d025fb35f5818394`.
- Workload: 60 held-out source sessions, each represented by one original and
  one deterministic `break_prefix` duplicate: 120 simultaneous load instances
  and 1,038 logical requests per cell per replicate. The duplicates raise
  contention but are not independent samples.
- Repetition: three matched A/D replicates (six fresh server cells; 6,228
  logical requests total). The source-session paired analysis averages each
  source's two load instances within each replicate, then summarizes 60 unique
  source means.
- Tool timing: recorded waits are replayed as `sleep` at 10× speedup. Generated
  text cannot change the trace's recorded next call; no oracle scheduler
  metadata is exposed.
- Reliability: all 6,228 logical requests finally succeeded; no preemption or
  swap was observed. There were two transient approximately 1 ms
  `ServerDisconnected` events, one in A and one in D, both successful on the
  explicit retry. Final failure count is zero, but the global exactly-once
  invariant is consequently false (6,230 total request attempts).

The detailed, invariant-checked aggregate is
[`paired_stress120_cap512_m64_t64.json`](paired_stress120_cap512_m64_t64.json).

## Aggregate result

Lower is better except for completion-token volume. “Reduction” is
`(A - D) / A`; positive is faster under D.

| Metric | A: FCFS+none | D: Joint+learned | Reduction |
|---|---:|---:|---:|
| Mean task flow time | 312.790 s | 281.387 s | **10.040%** |
| Median task flow time | 325.037 s | 292.794 s | **9.920%** |
| p95 task flow time | 388.322 s | 374.318 s | **3.606%** |
| Max task flow time | 397.756 s | 387.350 s | **2.616%** |
| Task makespan | 398.110 s | 387.705 s | **2.614%** |
| Instrumentation wall time | 398.636 s | 388.230 s | **2.610%** |
| Mean request latency | 34.710 s | 31.086 s | **10.439%** |
| p95 request latency | 71.150 s | 60.927 s | **14.368%** |
| Mean request queue time | 12.601 s | 10.253 s | **18.633%** |
| Aggregate completion tokens | reference | 0.163% higher | — |

Per-replicate mean task-flow reductions were 9.3449%, 9.6182%, and 11.1237%;
every replicate also had a positive p95 task-flow reduction (2.8179%, 3.5015%,
and 4.4875%).

At the correct independent unit, D was faster for 57/60 source sessions. The
source-level mean saving is 31.403 s. A fixed-seed nonparametric bootstrap over
the 60 per-source means gives a 95% interval of **[25.949, 37.104] s**. This is
descriptive paired uncertainty for this fixed stress workload; it does not turn
the duplicated 120 instances into 120 independent observations.

## Post-hoc server-log diagnostics (not primary metrics)

After the experiment, a read-only parser processed the six original
`server.log` files. These quantities are derived from the logs rather than
back-filled from the request summary, were not the primary endpoint, and were
not used to select the result.

| Log-derived average across the three replicates | A: FCFS+none | D: Joint+learned |
|---|---:|---:|
| Prefix-hit rate | 40.924% | 50.438% |
| Waiting requests | 33.209 | 27.671 |
| GPU KV usage | 50.181% | 46.484% |
| Running requests | 56.544 | 54.825 |

All three replicates changed in the same direction for these diagnostics. They
are consistent with an explanation involving better prefix locality and
task-aware ordering under the combined treatment. They are not a causal proof,
do not establish an isolated effect for any one mechanism, and should not be
read as additional primary performance endpoints.

## Interpretation and boundaries

At target64, both `VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING` and
`VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING` equal native `max-num-seqs=64`; the
decode band therefore does not restrict decode concurrency below the native
capacity. The observed A/D gain is attributable only to the implemented
**combination** of causal task-aware reordering, HBM/prefix-locality controls,
and learned overlap. It must not be described as a pacing-only gain, nor as an
ablation of the tool and LLM components.

The stress120 duplicate workload is intentionally a development/load-sensitivity
test: it is not a final held-out benchmark, does not execute live tools, and
does not let model responses determine future tool calls. It also runs on
4 × A100-40GB rather than the paper's 32 × A100-80GB evaluation. This repository
is consequently a minimal trace-replay implementation of selected PASTE
mechanisms, not the complete paper system or a claim to reproduce its absolute
numbers.

## Development history retained

We do not hide the tuning failures that led to target64. The target32, target48,
and target56 development configurations produced negative or diagnostic results
under their respective setups; the target56 stress aggregate is retained in
[`paired_stress120_cap512_m64_t56.json`](paired_stress120_cap512_m64_t56.json).
The old one-pair 30-session functional smoke reported a 10.29% mean reduction
but regressed tail/makespan metrics. It is no longer the public performance
claim and is not pooled with this strict three-replicate result. Those records
remain available as development evidence. Target64 was selected after this
development work; it is not presented as an untouched final configuration, and
contrary runs have not been erased.
