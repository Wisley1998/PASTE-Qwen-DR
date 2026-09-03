# Process-sidecar fixed-arrival no-interference supplement

Each paired repeat replays the same precomputed absolute authority arrival deadlines against K=0 and process-sidecar treatment. The scenario is forced all-wrong, so any treatment difference is interference rather than speculative benefit.

| C | R | Authority regression ms/target | Latency 90% CI | Makespan regression | Makespan 90% CI | Decision | Safety |
|---:|---:|---:|:---:|---:|:---:|:---:|:---:|
| 16 | 32 | +0.1254 | [-0.0205, +0.2714] | +0.349% | [+0.169%, +0.530%] | regression | pass |

## Protocol

- C uses the existing source-session batching definition. The batch cadence is frozen from modeled lead/service waves before either paired replay; observed completions never schedule future arrivals.
- Authority scheduled latency includes timer-release lateness, broker queueing, and service. Repeat—not target—is the inference unit; AB/BA order is counterbalanced.
- Treatment uses a forked process sidecar, K=4 by default, finite leases, lazy result bridge, disjoint logical CPU affinity, and SCHED_IDLE. All future batches are handed off in one bounded packet before the timed origin; `timed_parent_submit_packets=0` is enforced. No exact claims, result packets, terminal packets, or parent tombstone packets are permitted before the safety gate.
- Makespan inference charges the measured one-time parent preload cost even though that work is outside the fixed-arrival origin.
- The no-interference margins match the main runner: 0.10 ms/target and 0.1% trace makespan, with one-sided 95% bounds and at least eight paired repeats.

## Scope

This isolates modeled executor capacity, Python GIL, and logical CPU placement for lightweight synthetic tools. It does not certify physical-core/LLC/NUMA isolation or independent network, connection-pool, and remote-service quotas. A statistically inconclusive result is not evidence of equivalence.

Raw repeat vectors are stored in `raw_repeat_vectors.json`; configuration and source hashes are stored in both JSON outputs.
