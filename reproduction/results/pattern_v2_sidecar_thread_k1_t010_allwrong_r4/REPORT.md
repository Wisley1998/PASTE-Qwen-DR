# Pattern-v2 isolated speculative sidecar

Authority uses an unchanged demand-only broker. Predictor, admission, TTL, execution, and drain use a dedicated sidecar event-loop thread. Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority regression ms/target | Logical benefit ms/target | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| all_wrong_counterfactual | 1 | 1 | 0.100 | +0.055 | -0.055 | -2.0% | 0.0% | 0.0% | 1.732x | yes |
| all_wrong_counterfactual | 16 | 1 | 0.100 | -0.086 | +0.086 | -0.3% | 0.0% | 0.0% | 1.104x | yes |
| all_wrong_counterfactual | 64 | 1 | 0.100 | -0.857 | +0.857 | -0.1% | 0.0% | 0.0% | 1.041x | yes |

## Metric semantics

- Authority regression compares scheduled-to-terminal time for the always-executed demand-only backbone calls.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- The practical no-regression gate is a paired pooled and repeat-median authority regression no larger than 0.10 ms/target.

## Scope

This synthetic replay establishes control-plane behavior, not shared backend quota isolation. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar.
