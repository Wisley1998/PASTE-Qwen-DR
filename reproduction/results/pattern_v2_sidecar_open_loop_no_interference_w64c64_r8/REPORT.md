# Process-sidecar fixed-arrival no-interference supplement

Each paired repeat replays the same precomputed absolute authority arrival deadlines against K=0 and process-sidecar treatment. The scenario is forced all-wrong, so any treatment difference is interference rather than speculative benefit.

| C | R | Authority regression ms/target | Latency 90% CI | Makespan regression | Makespan 90% CI | Decision | Safety |
|---:|---:|---:|:---:|---:|:---:|:---:|:---:|
| 1 | 8 | +0.0470 | [-0.0779, +0.1719] | +0.003% | [-0.012%, +0.017%] | inconclusive | pass |
| 16 | 8 | +0.4403 | [+0.2940, +0.5865] | -0.003% | [-0.110%, +0.104%] | regression | pass |
| 64 | 8 | +0.3054 | [-0.3636, +0.9744] | +0.274% | [-0.067%, +0.617%] | inconclusive | pass |

## Protocol

- C uses the existing source-session batching definition. The batch cadence is frozen from modeled lead/service waves before either paired replay; observed completions never schedule future arrivals.
- Authority scheduled latency includes timer-release lateness, broker queueing, and service. Repeat—not target—is the inference unit; AB/BA order is counterbalanced.
- Treatment uses a forked process sidecar, K=4 by default, finite leases, lazy result bridge, disjoint logical CPU affinity, and SCHED_IDLE. No exact claims, result packets, terminal packets, or parent tombstone packets are permitted before the safety gate.
- The no-interference margins match the main runner: 0.10 ms/target and 0.1% trace makespan, with one-sided 95% bounds and at least eight paired repeats.

## Scope

This isolates modeled executor capacity, Python GIL, and logical CPU placement for lightweight synthetic tools. It does not certify physical-core/LLC/NUMA isolation or independent network, connection-pool, and remote-service quotas. A statistically inconclusive result is not evidence of equivalence.

Raw repeat vectors are stored in `raw_repeat_vectors.json`; configuration and source hashes are stored in both JSON outputs.
