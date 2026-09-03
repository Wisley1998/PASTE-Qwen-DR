# Pattern-v2 isolated speculative sidecar

Authority uses an unchanged demand-only broker. Predictor scoring and selection are precomputed in the parent before the timed wall; only admission, speculative execution, finite-lease cleanup, and drain use the dedicated sidecar control plane with batched non-blocking ingress and a lazy result bridge (`process`). Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority regression ms/target | Logical benefit ms/target | Benefit evidence | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|
| observed_nested_oof | 1 | 4 | 0.200 | +0.048 | +0.281 | improvement | +0.1% | 4.1% | 21.4% | 1.191x | inconclusive |
| observed_nested_oof | 16 | 4 | 0.200 | +0.249 | +1.110 | improvement | -0.2% | 4.2% | 22.4% | 1.188x | inconclusive |
| observed_nested_oof | 64 | 4 | 0.200 | +0.561 | +3.317 | improvement | -0.1% | 3.0% | 26.6% | 1.114x | inconclusive |
| all_wrong_counterfactual | 1 | 4 | 0.200 | +0.041 | -0.041 | inconclusive | -0.2% | 0.0% | 0.0% | 1.190x | inconclusive |
| all_wrong_counterfactual | 16 | 4 | 0.200 | +0.348 | -0.348 | regression | -0.2% | 0.0% | 0.0% | 1.188x | regression |
| all_wrong_counterfactual | 64 | 4 | 0.200 | -0.172 | +0.172 | inconclusive | -0.1% | 0.0% | 0.0% | 1.107x | inconclusive |

## Metric semantics

- Authority regression compares scheduled-to-terminal time for the always-executed demand-only backbone calls.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Benefit evidence is an improvement only when the repeat-level one-sided 95% lower bound on saved latency is above zero.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- With strict shadow barrier enabled, a later batch is not admitted until the current batch's shadow-authority calls have drained. This prevents speculative early return from increasing protected-broker overlap across batches.
- No-regression inference treats one paired AB/BA repetition—not individual targets—as the independent unit. A cell needs at least eight repetitions and one-sided 95% upper bounds no larger than 0.10 ms/target and 0.1% authority wall. Otherwise it is reported as regression, inconclusive, or insufficient rather than a binary point-estimate failure.

## Scope

This synthetic replay establishes control-plane behavior, not shared backend quota isolation. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar. A formal sub-millisecond equivalence claim additionally requires a matched A/A noise calibration; absent that, repeat-level inference may remain inconclusive. All-wrong cells that do not confirm a practical regression still do not establish equivalence.
