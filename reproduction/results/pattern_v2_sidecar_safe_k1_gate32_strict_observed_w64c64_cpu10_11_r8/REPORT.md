# Pattern-v2 isolated speculative sidecar

Authority uses an unchanged demand-only broker. Predictor scoring and selection are precomputed in the parent before the timed wall; only admission, speculative execution, finite-lease cleanup, and drain use the dedicated sidecar control plane with batched non-blocking ingress and a lazy result bridge (`process`). Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority regression ms/target | Logical benefit ms/target | Benefit evidence | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|
| observed_nested_oof | 1 | 1 | 0.200 | +0.094 | +0.220 | improvement | -0.4% | 4.1% | 24.7% | 1.168x | regression |
| observed_nested_oof | 16 | 1 | 0.200 | +0.392 | -0.192 | regression | -1.2% | 1.9% | 22.9% | 1.084x | regression |
| observed_nested_oof | 64 | 1 | 0.200 | +0.282 | -0.282 | inconclusive | -0.6% | 0.0% | 0.0% | 1.000x | inconclusive |

## Metric semantics

- Authority regression compares scheduled-to-terminal time for the always-executed demand-only backbone calls.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Benefit evidence is an improvement only when the repeat-level one-sided 95% lower bound on saved latency is above zero.
- Strict shadow barrier is enabled: a later batch is not admitted until the current batch's shadow-authority calls have drained. This prevents speculative early return from increasing protected-broker overlap across batches.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- No-regression inference treats one paired AB/BA repetition—not individual targets—as the independent unit. A cell needs at least eight repetitions and one-sided 95% upper bounds no larger than 0.10 ms/target and 0.1% authority wall. Otherwise it is reported as regression, inconclusive, or insufficient rather than a binary point-estimate failure.
- The authority-control burst circuit breaker sets the safe start budget to zero once a synchronized authority batch exceeds the host-calibrated limit. The latch remains closed for the rest of the replay, and a fully abstained treatment never starts a sidecar process.

## Scope

This synthetic replay establishes control-plane behavior, not shared backend quota isolation. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar. A formal sub-millisecond equivalence claim additionally requires a matched A/A noise calibration; absent that, repeat-level inference may remain inconclusive. All-wrong cells that do not confirm a practical regression still do not establish equivalence.
