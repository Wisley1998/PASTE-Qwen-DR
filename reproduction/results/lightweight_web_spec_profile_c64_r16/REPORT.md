# Lightweight Web Speculation Capacity Profile

AUTH concurrency `64`, local service `20.0 ms`, payload `65536 B`, paired R=`16`.

Selected maximum safe speculative parallelism: **K=0**.

| K | Decision | Mean diff ms (UCB) | p95 diff ms (UCB) | Batch wall diff % (UCB) | Failures |
|---:|:---|---:|---:|---:|---:|
| 0 | fail | +0.6078 (+16.4909) | +3.1056 (+30.7766) | +1.8491 (+4.8575) | 0 |
| 1 | fail | +29.2338 (+45.0451) | +37.1918 (+65.5301) | +4.9550 (+7.8504) | 0 |
| 2 | fail | +60.4995 (+77.3137) | +71.1805 (+98.6230) | +8.0079 (+11.9289) | 0 |
| 4 | fail | +109.7196 (+133.7575) | +124.4096 (+159.2623) | +12.0760 (+16.7490) | 0 |
| 8 | fail | +213.7964 (+232.3500) | +254.1254 (+281.5829) | +23.6967 (+27.5667) | 0 |
| 16 | fail | +400.5851 (+423.8096) | +480.8675 (+511.8903) | +41.1983 (+44.9521) | 0 |
| 32 | fail | +741.3958 (+781.3544) | +1009.9022 (+1063.0462) | +84.5863 (+88.6392) | 0 |
| 64 | fail | +1058.1189 (+1115.6688) | +1873.8226 (+1959.2912) | +125.5505 (+131.5337) | 0 |

This is a host-specific all-wrong loopback profile; it does not certify remote backend quotas.
