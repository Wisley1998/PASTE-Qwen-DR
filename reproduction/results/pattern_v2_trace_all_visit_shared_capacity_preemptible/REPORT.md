# Resource-tight all-visit replay with preemptible speculation

Speculation shares a bounded Visit pool with authority. On authority arrival, an exact in-flight job is promoted with progress preserved; otherwise the lowest-score running speculations are cancelled immediately until authority can dispatch. Multi-URL authoritative Visits remain serial within a session.

`Policy hit` is the resource-unconstrained session-cache coverage of the unchanged selector (W5=55.51%, Top-10=71.94%). `Realized hit` is the part that obtained execution under the shared-capacity scheduler and was reusable by authority; capacity affects only the latter.

| Candidates | Scheduler | Pool/C | Spec cap | Policy hit | Realized hit | E2E speedup | Mean-flow speedup | Call amp. | Preempted | Wasted spec seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| budget_w5_cap10 | fixed_half | 8/8 | 4 | 55.51% | 24.90% | 7.11% | 7.52% | 2.025x | 90.8 | 2273.2 |
| budget_w5_cap10 | fixed_reserve_one | 8/8 | 7 | 55.51% | 29.93% | 8.52% | 9.12% | 2.499x | 230.6 | 3003.6 |
| budget_w5_cap10 | adaptive_idle_fill | 8/8 | 8 | 55.51% | 29.88% | 8.44% | 9.13% | 2.525x | 239.1 | 3017.7 |
| fixed_top10 | fixed_half | 8/8 | 4 | 71.94% | 24.45% | 7.26% | 7.60% | 2.056x | 97.0 | 2306.4 |
| fixed_top10 | fixed_reserve_one | 8/8 | 7 | 71.94% | 30.09% | 8.58% | 9.00% | 2.572x | 244.8 | 3094.9 |
| fixed_top10 | adaptive_idle_fill | 8/8 | 8 | 71.94% | 30.04% | 8.76% | 9.15% | 2.594x | 255.0 | 3117.1 |
| budget_w5_cap10 | fixed_half | 12/8 | 6 | 55.51% | 33.99% | 9.97% | 10.84% | 2.423x | 63.5 | 3411.1 |
| budget_w5_cap10 | fixed_reserve_one | 12/8 | 11 | 55.51% | 43.69% | 12.98% | 14.24% | 3.233x | 207.9 | 4966.1 |
| budget_w5_cap10 | adaptive_idle_fill | 12/8 | 12 | 55.51% | 44.16% | 13.38% | 14.40% | 3.241x | 217.9 | 4953.4 |
| fixed_top10 | fixed_half | 12/8 | 6 | 71.94% | 33.47% | 10.04% | 10.73% | 2.497x | 78.4 | 3495.6 |
| fixed_top10 | fixed_reserve_one | 12/8 | 11 | 71.94% | 43.59% | 13.03% | 13.82% | 3.476x | 258.5 | 5358.9 |
| fixed_top10 | adaptive_idle_fill | 12/8 | 12 | 71.94% | 43.29% | 13.00% | 13.68% | 3.516x | 275.2 | 5404.2 |
| budget_w5_cap10 | fixed_half | 16/8 | 8 | 55.51% | 40.86% | 12.28% | 13.29% | 2.812x | 67.9 | 4378.1 |
| budget_w5_cap10 | fixed_reserve_one | 16/8 | 15 | 55.51% | 51.15% | 16.20% | 17.34% | 3.462x | 134.8 | 5822.4 |
| budget_w5_cap10 | adaptive_idle_fill | 16/8 | 16 | 55.51% | 51.30% | 15.85% | 17.42% | 3.465x | 138.5 | 5818.7 |
| fixed_top10 | fixed_half | 16/8 | 8 | 71.94% | 39.73% | 12.16% | 12.75% | 2.945x | 93.6 | 4589.1 |
| fixed_top10 | fixed_reserve_one | 16/8 | 15 | 71.94% | 53.28% | 15.86% | 17.06% | 4.300x | 262.1 | 7471.3 |
| fixed_top10 | adaptive_idle_fill | 16/8 | 16 | 71.94% | 53.18% | 16.08% | 17.19% | 4.328x | 278.0 | 7475.2 |
| budget_w5_cap10 | fixed_half | 16/16 | 8 | 55.51% | 27.23% | 5.94% | 8.46% | 2.023x | 69.8 | 2370.5 |
| budget_w5_cap10 | fixed_reserve_one | 16/16 | 15 | 55.51% | 32.41% | 6.64% | 9.84% | 2.524x | 227.8 | 3052.3 |
| budget_w5_cap10 | adaptive_idle_fill | 16/16 | 16 | 55.51% | 32.84% | 6.88% | 9.94% | 2.524x | 230.9 | 3049.7 |
| fixed_top10 | fixed_half | 16/16 | 8 | 71.94% | 27.53% | 6.57% | 8.36% | 2.105x | 82.4 | 2495.2 |
| fixed_top10 | fixed_reserve_one | 16/16 | 15 | 71.94% | 32.89% | 7.35% | 9.79% | 2.654x | 250.2 | 3273.4 |
| fixed_top10 | adaptive_idle_fill | 16/16 | 16 | 71.94% | 32.16% | 7.46% | 9.65% | 2.660x | 254.0 | 3290.8 |
| budget_w5_cap10 | fixed_half | 24/16 | 12 | 55.51% | 36.35% | 7.70% | 11.35% | 2.421x | 60.9 | 3421.3 |
| budget_w5_cap10 | fixed_reserve_one | 24/16 | 23 | 55.51% | 45.54% | 9.20% | 14.80% | 3.267x | 217.5 | 5007.4 |
| budget_w5_cap10 | adaptive_idle_fill | 24/16 | 24 | 55.51% | 45.82% | 9.52% | 14.83% | 3.263x | 214.8 | 5015.2 |
| fixed_top10 | fixed_half | 24/16 | 12 | 71.94% | 35.90% | 8.32% | 11.09% | 2.557x | 86.8 | 3648.6 |
| fixed_top10 | fixed_reserve_one | 24/16 | 23 | 71.94% | 45.47% | 10.10% | 14.24% | 3.515x | 256.1 | 5482.0 |
| fixed_top10 | adaptive_idle_fill | 24/16 | 24 | 71.94% | 45.72% | 10.47% | 14.37% | 3.508x | 252.9 | 5487.2 |
| budget_w5_cap10 | fixed_half | 32/16 | 16 | 55.51% | 41.81% | 9.03% | 13.51% | 2.809x | 66.8 | 4391.7 |
| budget_w5_cap10 | fixed_reserve_one | 32/16 | 31 | 55.51% | 52.56% | 10.53% | 17.91% | 3.479x | 127.1 | 5893.0 |
| budget_w5_cap10 | adaptive_idle_fill | 32/16 | 32 | 55.51% | 52.58% | 10.75% | 17.93% | 3.479x | 128.4 | 5894.5 |
| fixed_top10 | fixed_half | 32/16 | 16 | 71.94% | 41.51% | 9.63% | 13.18% | 2.999x | 102.6 | 4724.4 |
| fixed_top10 | fixed_reserve_one | 32/16 | 31 | 71.94% | 54.51% | 12.45% | 17.46% | 4.308x | 270.4 | 7471.5 |
| fixed_top10 | adaptive_idle_fill | 32/16 | 32 | 71.94% | 54.76% | 12.30% | 17.40% | 4.306x | 270.2 | 7478.3 |
| budget_w5_cap10 | fixed_half | 32/32 | 16 | 55.51% | 28.08% | 4.37% | 8.37% | 2.106x | 69.8 | 2598.8 |
| budget_w5_cap10 | fixed_reserve_one | 32/32 | 31 | 55.51% | 33.82% | 4.67% | 10.15% | 2.580x | 219.6 | 3238.7 |
| budget_w5_cap10 | adaptive_idle_fill | 32/32 | 32 | 55.51% | 33.69% | 4.95% | 10.18% | 2.585x | 222.9 | 3240.6 |
| fixed_top10 | fixed_half | 32/32 | 16 | 71.94% | 28.38% | 5.62% | 8.32% | 2.240x | 92.0 | 2813.6 |
| fixed_top10 | fixed_reserve_one | 32/32 | 31 | 71.94% | 34.04% | 6.25% | 9.96% | 2.802x | 257.2 | 3643.6 |
| fixed_top10 | adaptive_idle_fill | 32/32 | 32 | 71.94% | 34.04% | 6.27% | 9.96% | 2.808x | 258.1 | 3636.8 |
| budget_w5_cap10 | fixed_half | 48/32 | 24 | 55.51% | 36.70% | 5.43% | 11.39% | 2.474x | 62.4 | 3573.4 |
| budget_w5_cap10 | fixed_reserve_one | 48/32 | 47 | 55.51% | 45.24% | 6.07% | 14.73% | 3.257x | 210.9 | 4998.4 |
| budget_w5_cap10 | adaptive_idle_fill | 48/32 | 48 | 55.51% | 45.29% | 5.76% | 14.74% | 3.260x | 211.2 | 4992.6 |
| fixed_top10 | fixed_half | 48/32 | 24 | 71.94% | 36.25% | 6.96% | 11.01% | 2.694x | 101.6 | 3973.3 |
| fixed_top10 | fixed_reserve_one | 48/32 | 47 | 71.94% | 45.67% | 7.50% | 14.30% | 3.622x | 263.4 | 5716.3 |
| fixed_top10 | adaptive_idle_fill | 48/32 | 48 | 71.94% | 46.17% | 7.87% | 14.41% | 3.611x | 262.0 | 5700.6 |
| budget_w5_cap10 | fixed_half | 64/32 | 32 | 55.51% | 42.26% | 5.91% | 13.62% | 2.825x | 74.0 | 4443.5 |
| budget_w5_cap10 | fixed_reserve_one | 64/32 | 63 | 55.51% | 52.53% | 4.62% | 17.73% | 3.494x | 126.9 | 5893.7 |
| budget_w5_cap10 | adaptive_idle_fill | 64/32 | 64 | 55.51% | 52.61% | 4.62% | 17.77% | 3.493x | 126.5 | 5896.9 |
| fixed_top10 | fixed_half | 64/32 | 32 | 71.94% | 41.93% | 7.60% | 13.12% | 3.108x | 116.0 | 4984.0 |
| fixed_top10 | fixed_reserve_one | 64/32 | 63 | 71.94% | 54.56% | 8.46% | 17.37% | 4.339x | 265.9 | 7532.5 |
| fixed_top10 | adaptive_idle_fill | 64/32 | 64 | 71.94% | 54.46% | 8.54% | 17.36% | 4.333x | 265.5 | 7532.9 |

`fixed_half` and `fixed_reserve_one` leave a fixed speculative ceiling even while authority is idle. `adaptive_idle_fill` may use the entire idle pool and shrinks immediately through preemption when authority arrives. Authority-to-authority queueing is retained in both baseline and treatment; wrong speculation is never allowed to add queueing in front of authority.
