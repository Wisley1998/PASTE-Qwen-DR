# Pattern-v2 isolated speculative sidecar

Authority uses an unchanged demand-only broker. Predictor scoring and selection were precomputed in the parent before the timed wall; admission, speculative execution, finite-lease cleanup, and drain use a dedicated sidecar control plane with batched non-blocking ingress and a lazy result bridge (`process`). Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority regression ms/target | Logical benefit ms/target | Benefit evidence | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|
| observed_nested_oof | 1 | 4 | 0.200 | +0.045 | +0.307 | improvement | +1.0% | 4.0% | 20.9% | 1.190x | inconclusive |
| observed_nested_oof | 16 | 4 | 0.200 | +0.597 | +0.823 | improvement | -0.2% | 4.3% | 22.8% | 1.187x | regression |
| observed_nested_oof | 64 | 4 | 0.200 | +1.313 | +2.347 | improvement | +0.1% | 2.9% | 25.7% | 1.114x | regression |
| all_wrong_counterfactual | 1 | 4 | 0.200 | +0.043 | -0.043 | inconclusive | -0.1% | 0.0% | 0.0% | 1.190x | inconclusive |
| all_wrong_counterfactual | 16 | 4 | 0.200 | +0.219 | -0.219 | inconclusive | -0.1% | 0.0% | 0.0% | 1.188x | inconclusive |
| all_wrong_counterfactual | 64 | 4 | 0.200 | -0.350 | +0.350 | inconclusive | -0.0% | 0.0% | 0.0% | 1.112x | inconclusive |

## Metric semantics

- Authority regression compares scheduled-to-terminal time for the always-executed demand-only backbone calls.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Benefit evidence is an improvement only when the repeat-level one-sided 95% lower bound on saved latency is above zero.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- No-regression inference treats one paired AB/BA repetition—not individual targets—as the independent unit. A cell needs at least eight repetitions and one-sided 95% upper bounds no larger than 0.10 ms/target and 0.1% authority wall. Otherwise it is reported as regression, inconclusive, or insufficient rather than a binary point-estimate failure.

## Scope

This synthetic replay establishes conditional tool-call behavior, not shared backend quota isolation or end-to-end predictor cost. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar. All all-wrong cells remain statistically inconclusive: they did not confirm a practical regression, but they do not establish equivalence. The separate A/A note has no raw run artifact and is descriptive rather than formal matched evidence.
