# Pattern-v2 isolated speculative sidecar

Authority uses an unchanged demand-only broker. Predictor, admission, TTL, execution, and drain use a dedicated sidecar event-loop thread. Every target still submits one shadow authority attempt; speculative success can only shorten the logical completion.

| Scenario | C | K | Threshold | Authority regression ms/target | Logical benefit ms/target | Logical wall speedup | Visible coverage | Started precision | Call amp. | No-regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| observed_nested_oof | 1 | 1 | 0.100 | +0.405 | +1.489 | +1.7% | 14.7% | 20.3% | 1.724x | no |
| observed_nested_oof | 16 | 1 | 0.100 | -0.024 | +0.966 | -0.2% | 2.3% | 22.4% | 1.104x | no |
| observed_nested_oof | 64 | 1 | 0.100 | +0.601 | +0.047 | -0.1% | 1.0% | 22.5% | 1.043x | no |

## Metric semantics

- Authority regression compares scheduled-to-terminal time for the always-executed demand-only backbone calls.
- Logical benefit is agent-visible scheduled-to-first-valid-result latency, with authority winning ties.
- Logical wall stops after all agent-visible results. Authority wall then waits for every shadow authority call; drained wall additionally waits for the isolated sidecar.
- The practical no-regression gate is a paired pooled and repeat-median authority regression no larger than 0.10 ms/target.

## Scope

This synthetic replay establishes control-plane behavior, not shared backend quota isolation. Production use still requires an independent connection/rate/concurrency entitlement for the sidecar.
