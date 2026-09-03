# Pattern-v2 isolated speculative sidecar

Authority uses an unchanged demand-only broker on the parent loop. Predictor scoring and selection are precomputed in the parent before the timed wall; only admission, speculative execution, finite-lease cleanup, and drain use the dedicated sidecar control plane with batched non-blocking ingress (`process`). A bounded pull is performed in the pre-authority guard window and then the result epoch is sealed; no parent bridge or socket read runs during authority, and exact confirmation is an O(1) parent-local lookup. Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority lane regression ms/target | Authority observed regression ms/target | Logical benefit ms/target | Benefit evidence | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|
| observed_nested_oof | 1 | 4 | 0.000 | +0.102 | +0.102 | +3.847 | improvement | +3.2% | 19.0% | 14.2% | 2.335x | inconclusive |
| observed_nested_oof | 4 | 4 | 0.000 | +0.137 | +0.137 | +3.832 | improvement | +3.5% | 19.0% | 14.2% | 2.334x | inconclusive |
| observed_nested_oof | 8 | 4 | 0.000 | +0.108 | +0.108 | +2.534 | improvement | +1.0% | 12.6% | 17.0% | 1.740x | inconclusive |
| observed_nested_oof | 16 | 4 | 0.000 | +0.356 | +0.356 | +1.199 | improvement | +0.2% | 7.3% | 18.0% | 1.405x | inconclusive |
| observed_nested_oof | 32 | 4 | 0.000 | +0.594 | +0.594 | +0.478 | improvement | +0.3% | 5.0% | 20.8% | 1.240x | regression |
| observed_nested_oof | 64 | 4 | 0.000 | +2.022 | +2.022 | -1.383 | inconclusive | -0.5% | 2.9% | 19.0% | 1.154x | regression |

## Metric semantics

- Authority lane regression compares scheduled-to-terminal time inside the always-executed demand-only path. Authority observed regression additionally includes return handoff to the control loop.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Benefit evidence is an improvement only when the repeat-level one-sided 95% lower bound on saved latency is above zero.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- No-regression inference treats one paired AB/BA repetition—not individual targets—as the independent unit. A cell needs at least eight repetitions and one-sided 95% upper bounds no larger than 0.10 ms/target and 0.1% authority wall. Otherwise it is reported as regression, inconclusive, or insufficient rather than a binary point-estimate failure.
- The authority-control burst circuit breaker sets the safe start budget to zero once a synchronized authority batch exceeds the host-calibrated limit; a zero limit means no positive resource certificate was supplied. The latch remains closed for the rest of the replay, and a fully abstained treatment never starts a sidecar process.

## Scope

This synthetic replay establishes control-plane behavior, not shared backend quota isolation. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar. A formal sub-millisecond equivalence claim additionally requires a matched A/A noise calibration; absent that, repeat-level inference may remain inconclusive. All-wrong cells that do not confirm a practical regression still do not establish equivalence.
