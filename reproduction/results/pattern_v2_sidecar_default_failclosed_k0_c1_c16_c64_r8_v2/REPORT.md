# Process-sidecar fixed-arrival no-interference supplement

Each paired repeat replays the same precomputed absolute authority arrival deadlines against K=0 and process-sidecar treatment. The scenario is forced all-wrong, so any treatment difference is interference rather than speculative benefit.

| C | R | Authority regression ms/target | Latency 90% CI | Makespan regression | Makespan 90% CI | Decision | Safety |
|---:|---:|---:|:---:|---:|:---:|:---:|:---:|
| 1 | 8 | -0.0267 | [-0.1603, +0.1069] | +0.013% | [-0.003%, +0.029%] | inconclusive | pass |
| 16 | 8 | +0.1377 | [-0.0790, +0.3543] | -0.008% | [-0.075%, +0.059%] | inconclusive | pass |
| 64 | 8 | +0.2991 | [+0.0124, +0.5857] | -0.136% | [-0.243%, -0.028%] | inconclusive | pass |

## Protocol

- C uses the existing source-session batching definition. The batch cadence is frozen from modeled lead/service waves before either paired replay; observed completions never schedule future arrivals.
- Authority scheduled latency includes timer-release lateness, broker queueing, and service. Repeat—not target—is the inference unit; AB/BA order is counterbalanced.
- Treatment requests K=1. The resource gate abstained in every sample, so no sidecar process, preload, bridge, or IPC was created; the following isolation details are therefore vacuous: topology-aware dedicated CPU placement, and SCHED_IDLE. All future batches are handed off in one bounded packet before the timed origin; `timed_parent_submit_packets=0` is enforced. No exact claims, result packets, terminal packets, or parent tombstone packets are permitted before the safety gate.
- A configured speculation phase guard delays only epoch 2+ sidecar releases beyond the preceding modeled authority completion boundary. Authority arrivals remain unchanged, and admission uses the resulting shorter effective lead.
- The authority-control burst gate makes the certified start budget zero for an epoch whose synchronized authority arrivals exceed the configured host-calibrated limit. A zero limit means that no positive resource certificate was supplied. This protects the single authority event loop even when tool slots are plentiful.
- Makespan inference charges the measured one-time parent preload cost even though that work is outside the fixed-arrival origin.
- The no-interference margins match the main runner: 0.10 ms/target and 0.1% trace makespan, with one-sided 95% bounds and at least eight paired repeats.

## Scope

This isolates modeled executor capacity, Python GIL, and logical CPU placement for lightweight synthetic tools. It does not certify physical-core/LLC/NUMA isolation or independent network, connection-pool, and remote-service quotas. A statistically inconclusive result is not evidence of equivalence.

Raw repeat vectors are stored in `raw_repeat_vectors.json`; configuration and source hashes are stored in both JSON outputs.
