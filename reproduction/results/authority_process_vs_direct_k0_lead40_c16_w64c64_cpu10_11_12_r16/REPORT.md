# Pattern-v2 isolated speculative sidecar

The K=0 baseline uses the original in-process authority path; the treatment uses a private authority process/GIL/CPU. This is an end-to-end topology-migration comparison, not an incremental sidecar-only comparison. Predictor scoring and selection are precomputed in the parent before the timed wall; only admission, speculative execution, finite-lease cleanup, and drain use the dedicated sidecar control plane with batched non-blocking ingress and a blocking result bridge started during untimed setup when the sidecar is activated (`process`). Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority lane regression ms/target | Authority observed regression ms/target | Logical benefit ms/target | Benefit evidence | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|
| all_wrong_counterfactual | 16 | 1 | 0.220 | -0.743 | -0.126 | +0.126 | inconclusive | -1.0% | 0.0% | 0.0% | 1.000x | regression |

## Metric semantics

- Authority lane regression compares scheduled-to-terminal time inside the always-executed demand-only path. Authority observed regression additionally includes return handoff to the control loop.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Benefit evidence is an improvement only when the repeat-level one-sided 95% lower bound on saved latency is above zero.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- No-regression inference treats one paired AB/BA repetition—not individual targets—as the independent unit. A cell needs at least eight repetitions and one-sided 95% upper bounds no larger than 0.10 ms/target and 0.1% authority wall. Otherwise it is reported as regression, inconclusive, or insufficient rather than a binary point-estimate failure.
- The authority-control burst circuit breaker sets the safe start budget to zero once a synchronized authority batch exceeds the host-calibrated limit; a zero limit means no positive resource certificate was supplied. The latch remains closed for the rest of the replay, and a fully abstained treatment never starts a sidecar process.

## Scope

This synthetic replay establishes control-plane behavior, not shared backend quota isolation. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar. A formal sub-millisecond equivalence claim additionally requires a matched A/A noise calibration; absent that, repeat-level inference may remain inconclusive. All-wrong cells that do not confirm a practical regression still do not establish equivalence.
