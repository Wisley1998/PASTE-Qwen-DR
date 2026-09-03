# Process-sidecar fixed-arrival no-interference supplement

Each paired repeat replays the same precomputed absolute authority arrival deadlines against K=0 and process-sidecar treatment. The scenario is forced all-wrong, so any treatment difference is interference rather than speculative benefit.

| C | R | Authority regression ms/target | Latency 90% CI | Makespan regression | Makespan 90% CI | Decision | Safety |
|---:|---:|---:|:---:|---:|:---:|:---:|:---:|
| 1 | 8 | -0.0369 | [-0.1237, +0.0499] | +0.008% | [-0.010%, +0.026%] | pass | pass |
| 16 | 8 | +0.0510 | [-0.2333, +0.3354] | +0.005% | [-0.060%, +0.070%] | inconclusive | pass |
| 64 | 8 | -0.2218 | [-0.4276, -0.0160] | -0.200% | [-0.376%, -0.024%] | pass | pass |

## Protocol

- C uses the existing source-session batching definition. The batch cadence is frozen from modeled lead/service waves before either paired replay; observed completions never schedule future arrivals.
- Authority scheduled latency includes timer-release lateness, broker queueing, and service. Repeat—not target—is the inference unit; AB/BA order is counterbalanced.
- Treatment requests K=1, but the zero resource-certificate limit abstained in every sample. No sidecar process, preload, bridge, or IPC was created; `timed_parent_submit_packets=0`, `selected=started=0`, and physical call amplification is exactly 1.0.
- A configured speculation phase guard delays only epoch 2+ sidecar releases beyond the preceding modeled authority completion boundary. Authority arrivals remain unchanged, and admission uses the resulting shorter effective lead.
- The authority-control burst gate makes the certified start budget zero for an epoch whose synchronized authority arrivals exceed the configured host-calibrated limit. A zero limit means that no positive resource certificate was supplied. This protects the single authority event loop even when tool slots are plentiful.
- Makespan inference charges the measured one-time parent preload cost even though that work is outside the fixed-arrival origin.
- The no-interference margins match the main runner: 0.10 ms/target and 0.1% trace makespan, with one-sided 95% bounds and at least eight paired repeats.

## Scope

This isolates modeled executor capacity, Python GIL, and logical CPU placement for lightweight synthetic tools. It does not certify physical-core/LLC/NUMA isolation or independent network, connection-pool, and remote-service quotas. A statistically inconclusive result is not evidence of equivalence.

Raw repeat vectors are stored in `raw_repeat_vectors.json`; configuration and source hashes are stored in both JSON outputs.
