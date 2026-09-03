# Lightweight Web Speculation Capacity Profile

AUTH concurrency `64`, local service `20.0 ms`, payload `65536 B`, paired R=`64`.

Selected maximum safe speculative parallelism: **K=0**.

| K | Decision | Mean diff ms (UCB) | p95 diff ms (UCB) | Batch wall diff % (UCB) | Failures |
|---:|:---|---:|---:|---:|---:|
| 0 | fail | -2.7982 (-0.1672) | -2.7790 (+0.0816) | -0.9662 (+0.3121) | 0 |
| 1 | fail | +1.6866 (+3.7444) | +2.0048 (+4.4485) | +1.0317 (+2.1733) | 0 |
| 2 | fail | -0.1139 (+1.7855) | +0.0298 (+2.2965) | +0.2479 (+1.3018) | 0 |

This is a host-specific all-wrong loopback profile; it does not certify remote backend quotas.
