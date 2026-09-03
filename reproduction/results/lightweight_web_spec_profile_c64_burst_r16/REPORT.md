# Lightweight Web Speculation Capacity Profile

AUTH concurrency `64`, local service `20.0 ms`, payload `65536 B`, paired R=`16`.

Selected maximum safe speculative parallelism: **K=0**.

| K | Decision | Mean diff ms (UCB) | p95 diff ms (UCB) | Batch wall diff % (UCB) | Failures |
|---:|:---|---:|---:|---:|---:|
| 0 | fail | -6.1930 (+0.3913) | -6.3932 (+0.8597) | -2.4677 (+0.7290) | 0 |
| 1 | fail | -0.7686 (+2.5344) | -1.2800 (+2.8913) | -0.5414 (+1.5406) | 0 |
| 2 | fail | +8.0197 (+11.1903) | +7.9479 (+12.7201) | +3.9307 (+6.3526) | 0 |
| 4 | fail | +11.4537 (+15.0997) | +9.7252 (+14.0586) | +4.9389 (+7.2180) | 0 |
| 8 | fail | +23.7994 (+27.3114) | +22.2839 (+26.8005) | +10.9111 (+13.2628) | 0 |
| 16 | fail | +51.5464 (+56.8617) | +51.5263 (+57.5769) | +25.9692 (+28.8465) | 0 |
| 32 | fail | +97.6383 (+102.0127) | +97.2830 (+102.8209) | +47.3689 (+50.8844) | 0 |
| 64 | fail | +192.2420 (+208.6962) | +226.3076 (+238.3646) | +109.5466 (+115.4874) | 0 |

This is a host-specific all-wrong loopback profile; it does not certify remote backend quotas.
