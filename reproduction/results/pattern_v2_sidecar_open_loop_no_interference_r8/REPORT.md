# Process-sidecar fixed-arrival no-interference supplement

Each paired repeat replays the same precomputed absolute authority arrival deadlines against K=0 and process-sidecar treatment. The scenario is forced all-wrong, so any treatment difference is interference rather than speculative benefit.

| C | R | Authority regression ms/target | Latency 90% CI | Makespan regression | Makespan 90% CI | Decision | Safety |
|---:|---:|---:|:---:|---:|:---:|:---:|:---:|
| 1 | 8 | +0.0085 | [-0.0819, +0.0988] | +0.004% | [-0.008%, +0.015%] | pass | pass |
| 16 | 8 | +0.5224 | [-0.6107, +1.6556] | -0.000% | [-0.022%, +0.021%] | inconclusive | pass |
| 64 | 8 | +0.1577 | [-3.9964, +4.3117] | +0.011% | [-0.188%, +0.209%] | inconclusive | pass |

## Protocol

- C uses the existing source-session batching definition. The batch cadence is frozen from modeled lead/service waves before either paired replay; observed completions never schedule future arrivals.
- Authority scheduled latency includes timer-release lateness, broker queueing, and service. Repeat—not target—is the inference unit; AB/BA order is counterbalanced.
- Treatment uses a forked process sidecar, K=4 by default, finite leases, lazy result bridge, disjoint logical CPU affinity, and SCHED_IDLE. No exact claims, result packets, terminal packets, or parent tombstone packets are permitted before the safety gate.
- The no-interference margins match the main runner: 0.10 ms/target and 0.1% trace makespan, with one-sided 95% bounds and at least eight paired repeats.

## Scope

This isolates modeled executor capacity, Python GIL, and logical CPU placement for lightweight synthetic tools. It does not certify physical-core/LLC/NUMA isolation or independent network, connection-pool, and remote-service quotas. A statistically inconclusive result is not evidence of equivalence.

Raw repeat vectors are stored in `raw_repeat_vectors.json`; configuration and source hashes are stored in both JSON outputs.
