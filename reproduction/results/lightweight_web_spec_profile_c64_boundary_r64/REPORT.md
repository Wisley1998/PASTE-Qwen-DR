# Lightweight Web Speculation Capacity Profile

AUTH concurrency `64`, local service `20.0 ms`, payload `65536 B`, paired R=`64`.

Selected maximum safe speculative parallelism: **K=0**.

| K | Decision | Mean diff ms (UCB) | p95 diff ms (UCB) | Batch wall diff % (UCB) | Failures |
|---:|:---|---:|---:|---:|---:|
| 0 | fail | -1.1597 (+1.6864) | -1.0924 (+2.2670) | -0.0869 (+1.4482) | 0 |
| 1 | fail | +3.0636 (+5.3771) | +2.7966 (+5.5824) | +1.5096 (+2.8315) | 0 |
| 2 | fail | +6.1764 (+8.2616) | +6.9932 (+9.4146) | +3.4705 (+4.6181) | 0 |

This is a host-specific all-wrong loopback profile; it does not certify remote backend quotas.
