# Process-sidecar fixed-arrival no-interference supplement

Each paired repeat replays the same precomputed absolute authority arrival deadlines against K=0 and process-sidecar treatment. The scenario is forced all-wrong, so any treatment difference is interference rather than speculative benefit.

| C | R | Authority regression ms/target | Latency 90% CI | Makespan regression | Makespan 90% CI | Decision | Safety |
|---:|---:|---:|:---:|---:|:---:|:---:|:---:|
| 1 | 8 | -0.0378 | [-0.2070, +0.1314] | +0.041% | [+0.028%, +0.054%] | inconclusive | pass |
| 16 | 8 | +0.1939 | [+0.0662, +0.3215] | +0.273% | [+0.170%, +0.376%] | regression | pass |
| 64 | 8 | +0.2086 | [-0.2232, +0.6403] | +0.272% | [+0.112%, +0.431%] | regression | pass |

## Protocol

- C uses the existing source-session batching definition. The batch cadence is frozen from modeled lead/service waves before either paired replay; observed completions never schedule future arrivals.
- Authority scheduled latency includes timer-release lateness, broker queueing, and service. Repeat—not target—is the inference unit; AB/BA order is counterbalanced.
- Treatment uses a forked process sidecar, K=4 by default, finite leases, lazy result bridge, disjoint logical CPU affinity, and SCHED_IDLE. All future batches are handed off in one bounded packet before the timed origin; `timed_parent_submit_packets=0` is enforced. No exact claims, result packets, terminal packets, or parent tombstone packets are permitted before the safety gate.
- Makespan inference charges the measured one-time parent preload cost even though that work is outside the fixed-arrival origin.
- The no-interference margins match the main runner: 0.10 ms/target and 0.1% trace makespan, with one-sided 95% bounds and at least eight paired repeats.

## Scope

This isolates modeled executor capacity, Python GIL, and logical CPU placement for lightweight synthetic tools. It does not certify physical-core/LLC/NUMA isolation or independent network, connection-pool, and remote-service quotas. A statistically inconclusive result is not evidence of equivalence.

Raw repeat vectors are stored in `raw_repeat_vectors.json`; configuration and source hashes are stored in both JSON outputs.
