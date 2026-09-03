# Pattern-v2 isolated speculative sidecar

Authority uses an unchanged demand-only broker. Predictor, admission, execution, finite-lease cleanup, and drain use a dedicated sidecar control plane with batched non-blocking ingress and a lazy result bridge (`process`). Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority regression ms/target | Logical benefit ms/target | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| observed_nested_oof | 1 | 4 | 0.200 | +0.013 | +0.409 | +1.1% | 4.3% | 22.2% | 1.191x | inconclusive |
| observed_nested_oof | 16 | 4 | 0.200 | +0.626 | +0.828 | -0.1% | 4.3% | 22.7% | 1.188x | regression |
| observed_nested_oof | 64 | 4 | 0.200 | +1.673 | +2.253 | -0.1% | 3.0% | 26.6% | 1.114x | regression |
| all_wrong_counterfactual | 1 | 4 | 0.200 | +0.075 | -0.075 | +0.1% | 0.0% | 0.0% | 1.191x | inconclusive |
| all_wrong_counterfactual | 16 | 4 | 0.200 | +0.426 | -0.426 | -0.3% | 0.0% | 0.0% | 1.188x | inconclusive |
| all_wrong_counterfactual | 64 | 4 | 0.200 | +0.312 | -0.312 | -0.2% | 0.0% | 0.0% | 1.114x | inconclusive |

## Metric semantics

- Authority regression compares scheduled-to-terminal time for the always-executed demand-only backbone calls.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- No-regression inference treats one paired AB/BA repetition—not individual targets—as the independent unit. A cell needs at least eight repetitions and one-sided 95% upper bounds no larger than 0.10 ms/target and 0.1% authority wall. Otherwise it is reported as regression, inconclusive, or insufficient rather than a binary point-estimate failure.

## Scope

This synthetic replay establishes control-plane behavior, not shared backend quota isolation. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar. A formal sub-millisecond equivalence claim additionally requires a matched A/A noise calibration; absent that, repeat-level inference may remain inconclusive even when structural invariants and all-wrong wall results show no resource regression.
