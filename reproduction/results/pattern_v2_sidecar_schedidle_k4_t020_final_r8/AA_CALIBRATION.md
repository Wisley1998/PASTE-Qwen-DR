# K=0 / K=0 paired A/A calibration

> Provenance limitation: this immediate diagnostic run was not persisted as a
> raw JSON artifact. The table is retained only as descriptive scale
> calibration; it is not independently auditable matched evidence and is not
> embedded in `metrics.json` (whose `noise_calibration` status remains
> `missing`).

This calibration used the same frozen trace set, host, CPU affinity, synthetic
service (20 ms), lead (10 ms), concurrency values, and AB/BA pairing as the
main sidecar experiment. Both arms called `_run_sample(..., sidecar_slots=0)`;
there was no predictor selection, child process, IPC, or speculative call.
Eight paired repetitions were used at each concurrency.

The independent inference unit is one paired repetition. Intervals are the
same repeat-level two-sided 90% / one-sided 95% intervals used by the main
runner.

| C | Authority difference ms/target (mean, 90% CI) | Authority wall geometric difference (90% CI) |
|---:|---:|---:|
| 1 | +0.008 `[-0.127, +0.142]` | +0.135% `[-0.135%, +0.406%]` |
| 16 | +0.046 `[-0.327, +0.419]` | -0.013% `[-0.270%, +0.244%]` |
| 64 | -0.438 `[-1.571, +0.695]` | -0.063% `[-0.443%, +0.319%]` |

## Interpretation

The measurement design cannot resolve a 0.10 ms/target and 0.1% wall
equivalence margin at R=8: even two identical K=0 arms are `inconclusive` at
all three concurrency values. The main experiment's all-wrong point estimates
therefore must be compared with this noise floor; an inconclusive sidecar cell
is not evidence that speculation caused a regression.

This A/A result is an informal measurement calibration, not a margin
adjustment or a formal comparison with the treatment. It does not turn an
uncertain result into an equivalence pass. A publishable
sub-millisecond equivalence claim would need more paired blocks, a lower-noise
open-loop harness, raw repeat vectors with provenance, or all three.
