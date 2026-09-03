# Robustness under low predictability and high load

## Metric audit: Top-1 versus firing many candidates

The quoted 27.8% Top-1 and 93.8% hit rate are not reproduced as one same-scope metric by the frozen all-visit trace. The table below reports exact URL targets on the current 0.42x-LLM trace. `Immediate recall` uses the candidate set selected at that decision; `persistent coverage` also credits an earlier completed or in-flight session-cache prediction.

| Candidate budget | Immediate exact hits | Immediate recall | Persistent cache hits | Persistent coverage | Policy selections | Physical starts | Hits/start |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_top1 | 78/499 | 15.63% | 102/499 | 20.44% | 530 | 411 | 24.82% |
| fixed_top5 | 222/499 | 44.49% | 280/499 | 56.11% | 2643 | 1608 | 17.41% |
| budget_w5_cap10 | 209/499 | 41.88% | 277/499 | 55.51% | 2093 | 1502 | 18.44% |
| fixed_top10 | 297/499 | 59.52% | 359/499 | 71.94% | 5245 | 2706 | 13.27% |
| fixed_top20 | 360/499 | 72.14% | 405/499 | 81.16% | 9784 | 4403 | 9.20% |

## Observed labels: concurrency and load

Top-10 with preemptible adaptive idle-fill is shown below. Pool capacity scales with active Agent concurrency. Wasted work includes both cancelled partial calls and completed speculative calls that never produce a cache hit.

| Slots/Agent | C | Policy coverage | Realized hit | Wasted calls/auth | Wasted seconds/auth | Call amp. | Net latency benefit | E2E speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | 8 | 71.94% | 30.04% | 1.574 | 6.247 s | 2.594x | +68.25 s | 8.74% |
| 1.0x | 16 | 71.94% | 32.16% | 1.651 | 6.595 s | 2.660x | +33.18 s | 7.50% |
| 1.0x | 32 | 71.94% | 34.04% | 1.804 | 7.288 s | 2.808x | +19.19 s | 6.23% |
| 1.5x | 8 | 71.94% | 43.29% | 2.503 | 10.830 s | 3.516x | +101.35 s | 13.00% |
| 1.5x | 16 | 71.94% | 45.72% | 2.504 | 10.996 s | 3.508x | +46.56 s | 10.55% |
| 1.5x | 32 | 71.94% | 46.17% | 2.615 | 11.424 s | 3.611x | +24.06 s | 7.69% |
| 2.0x | 8 | 71.94% | 53.18% | 3.331 | 14.980 s | 4.328x | +125.32 s | 16.09% |
| 2.0x | 16 | 71.94% | 54.76% | 3.311 | 14.986 s | 4.306x | +54.71 s | 12.43% |
| 2.0x | 32 | 71.94% | 54.46% | 3.338 | 15.096 s | 4.333x | +26.11 s | 8.30% |

## Candidate breadth at representative high load

This cell uses C=16 and 1.5 Visit slots per active Agent. It shows whether wider firing remains worthwhile after charging wasted execution.

| Candidates | Scheduler | Policy coverage | Realized hit | Wasted calls/auth | Wasted seconds/auth | Call amp. | E2E speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| budget_w5_cap10 | adaptive_idle_fill | 55.51% | 45.82% | 2.262 | 10.050 s | 3.263x | 9.69% |
| budget_w5_cap10 | fixed_reserve_one | 55.51% | 45.54% | 2.267 | 10.035 s | 3.267x | 9.35% |
| fixed_top1 | adaptive_idle_fill | 20.44% | 20.44% | 0.627 | 2.951 s | 1.619x | 3.07% |
| fixed_top1 | fixed_reserve_one | 20.44% | 20.44% | 0.627 | 2.951 s | 1.619x | 3.07% |
| fixed_top10 | adaptive_idle_fill | 71.94% | 45.72% | 2.504 | 10.996 s | 3.508x | 10.55% |
| fixed_top10 | fixed_reserve_one | 71.94% | 45.47% | 2.513 | 10.986 s | 3.515x | 10.20% |

## Degrading predictability

For this deterministic negative control, 50%, 75%, or 100% of authority URLs are replaced by guaranteed non-candidate URLs after selection. Scores and admission are unchanged. The representative cell is C=16, 1.5 slots/Agent, adaptive idle-fill.

| Candidates | Scenario | Realized hit | Wasted calls/auth | Wasted seconds/auth | Waste fraction | Call amp. | E2E speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| budget_w5_cap10 | observed | 45.82% | 2.262 | 10.050 s | 82.59% | 3.263x | 9.69% |
| budget_w5_cap10 | wrong_50pct | 24.60% | 2.545 | 11.129 s | 90.74% | 3.555x | 5.12% |
| budget_w5_cap10 | mostly_wrong_75pct | 13.43% | 2.689 | 11.681 s | 94.77% | 3.710x | 3.00% |
| budget_w5_cap10 | all_wrong | 0.00% | 2.863 | 12.537 s | 100.00% | 3.864x | 0.00% |
| fixed_top10 | observed | 45.72% | 2.504 | 10.996 s | 83.85% | 3.508x | 10.55% |
| fixed_top10 | wrong_50pct | 24.20% | 2.817 | 12.145 s | 91.55% | 3.827x | 5.55% |
| fixed_top10 | mostly_wrong_75pct | 14.10% | 2.958 | 12.677 s | 94.87% | 3.976x | 2.98% |
| fixed_top10 | all_wrong | 0.00% | 3.136 | 13.541 s | 100.00% | 4.138x | 0.00% |

## Deterministic all-wrong worst case

Top-10 adaptive idle-fill is forced to miss every authority URL. Because running speculation is synchronously preempted for authority, the latency path falls back to baseline; the failure cost is wasted idle resource rather than authority delay or incorrect state commit.

| Slots/Agent | C | Realized hit | Wasted calls/auth | Wasted seconds/auth | Waste fraction | Call amp. | E2E speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | 8 | 0.00% | 1.871 | 7.280 s | 100.00% | 2.877x | 0.00% |
| 1.0x | 16 | 0.00% | 1.962 | 7.739 s | 100.00% | 2.966x | 0.00% |
| 1.0x | 32 | 0.00% | 2.158 | 8.599 s | 100.00% | 3.159x | 0.00% |
| 1.5x | 8 | 0.00% | 3.106 | 13.244 s | 100.00% | 4.112x | 0.00% |
| 1.5x | 16 | 0.00% | 3.136 | 13.541 s | 100.00% | 4.138x | 0.00% |
| 1.5x | 32 | 0.00% | 3.248 | 14.062 s | 100.00% | 4.249x | 0.00% |
| 2.0x | 8 | 0.00% | 4.237 | 18.877 s | 100.00% | 5.241x | 0.00% |
| 2.0x | 16 | 0.00% | 4.222 | 19.007 s | 100.00% | 5.224x | 0.00% |
| 2.0x | 32 | 0.00% | 4.224 | 18.913 s | 100.00% | 5.226x | 0.00% |

## Interpretation

A wide candidate budget raises static coverage but also lowers useful work per start. Under load, the relevant quantity is realized hit after admission, not the unthrottled candidate-union coverage. Preemption makes the all-wrong latency behavior fail-safe, but it does not make wrong work free: resource amplification and wasted slot-seconds remain, so a runtime confidence/load gate should shrink to W5 or abstain as predicted utility falls.
