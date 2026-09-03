# Pattern-v2 sustained open-loop authority stress

Every decision release and authoritative confirmation is fixed on an exogenous timeline shared by demand-only and treatment. Results use a non-preemptible synthetic executor; no vLLM or network is used.

| Scenario | Policy | Load / peak authority C | Overlap hit | Wrong starts | Waste/target | Physical amp | Miss p95 regression | Response p99 regression | Net exposed / scheduled per target | Scheduled repeat median / min | Throughput ratio | Top-10% start share | Backlog / coarse abstain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observed_nested_oof | rank5_unreserved | 0.50 / 25 | 7.2% | 432 | 3.07 ms | 1.46x | +29.88 ms | +37.14 ms | -4.97 / -6.07 ms | -5.57 / -8.39 ms | 0.988 | 54.5% | 0 / 0 |
| observed_nested_oof | confidence_reserved | 0.50 / 21 | 7.0% | 307 | 1.88 ms | 1.33x | +4.41 ms | +5.44 ms | -0.45 / -0.56 ms | -0.50 / -1.16 ms | 1.000 | 36.5% | 0 / 0 |
| observed_nested_oof | utility_risk_limited | 0.50 / 21 | 1.8% | 68 | 0.41 ms | 1.07x | +2.38 ms | +3.51 ms | -0.30 / -0.31 ms | -0.37 / -0.49 ms | 1.000 | 100.0% | 35 / 1885 |
| observed_nested_oof | rank5_unreserved | 0.90 / 104 | 1.0% | 50 | 0.32 ms | 1.05x | +204.49 ms | +210.70 ms | -81.79 / -84.39 ms | -77.70 / -119.90 ms | 0.827 | 100.0% | 0 / 0 |
| observed_nested_oof | confidence_reserved | 0.90 / 51 | 0.6% | 35 | 0.22 ms | 1.04x | +19.30 ms | +20.10 ms | -7.56 / -7.70 ms | -7.73 / -8.73 ms | 0.977 | 100.0% | 0 / 0 |
| observed_nested_oof | utility_risk_limited | 0.90 / 45 | 0.1% | 13 | 0.08 ms | 1.01x | +2.47 ms | +5.17 ms | +2.10 / +2.09 ms | +2.15 / -0.12 ms | 1.006 | 100.0% | 11 / 5775 |
| observed_nested_oof | rank5_unreserved | 1.20 / 143 | 0.3% | 23 | 0.15 ms | 1.02x | +191.25 ms | +194.65 ms | -112.08 / -114.95 ms | -115.50 / -129.86 ms | 0.814 | 100.0% | 0 / 0 |
| observed_nested_oof | confidence_reserved | 1.20 / 98 | 0.3% | 17 | 0.10 ms | 1.02x | +37.33 ms | +42.20 ms | -13.73 / -14.12 ms | -15.45 / -20.46 ms | 0.966 | 100.0% | 0 / 0 |
| observed_nested_oof | utility_risk_limited | 1.20 / 90 | 0.0% | 4 | 0.03 ms | 1.00x | +11.55 ms | +13.33 ms | -0.87 / -0.85 ms | -0.16 / -6.66 ms | 0.995 | 75.0% | 7 / 6050 |
| all_wrong_counterfactual | rank5_unreserved | 0.50 / 25 | 0.0% | 444 | 3.09 ms | 1.47x | +36.04 ms | +62.10 ms | -8.17 / -9.38 ms | -8.80 / -13.22 ms | 0.978 | 55.2% | 0 / 0 |
| all_wrong_counterfactual | confidence_reserved | 0.50 / 23 | 0.0% | 349 | 2.14 ms | 1.37x | +5.08 ms | +6.47 ms | -1.44 / -1.57 ms | -1.46 / -2.24 ms | 0.998 | 39.0% | 0 / 0 |
| all_wrong_counterfactual | utility_risk_limited | 0.50 / 21 | 0.0% | 87 | 0.54 ms | 1.09x | +2.28 ms | +4.04 ms | -0.26 / -0.28 ms | -0.28 / -0.44 ms | 1.000 | 100.0% | 31 / 1875 |
| all_wrong_counterfactual | rank5_unreserved | 0.90 / 104 | 0.0% | 51 | 0.34 ms | 1.05x | +191.65 ms | +193.87 ms | -94.32 / -96.41 ms | -97.38 / -112.30 ms | 0.807 | 100.0% | 0 / 0 |
| all_wrong_counterfactual | confidence_reserved | 0.90 / 49 | 0.0% | 39 | 0.24 ms | 1.04x | +23.60 ms | +28.98 ms | -8.69 / -8.89 ms | -8.26 / -13.78 ms | 0.973 | 100.0% | 0 / 0 |
| all_wrong_counterfactual | utility_risk_limited | 0.90 / 45 | 0.0% | 13 | 0.08 ms | 1.01x | +7.81 ms | +9.83 ms | -1.20 / -1.24 ms | -2.46 / -3.98 ms | 0.997 | 100.0% | 13 / 5805 |
| all_wrong_counterfactual | rank5_unreserved | 1.20 / 145 | 0.0% | 24 | 0.16 ms | 1.03x | +199.68 ms | +207.31 ms | -120.21 / -123.74 ms | -123.38 / -139.91 ms | 0.804 | 100.0% | 0 / 0 |
| all_wrong_counterfactual | confidence_reserved | 1.20 / 106 | 0.0% | 22 | 0.14 ms | 1.02x | +55.50 ms | +61.98 ms | -23.16 / -23.62 ms | -21.61 / -36.66 ms | 0.947 | 100.0% | 0 / 0 |
| all_wrong_counterfactual | utility_risk_limited | 1.20 / 88 | 0.0% | 3 | 0.02 ms | 1.00x | +0.47 ms | +1.12 ms | +2.77 / +2.77 ms | +2.53 / +1.23 ms | 1.005 | 75.0% | 7 / 6065 |

## Interpretation boundaries

- Offered load is authoritative service demand divided by visit capacity; speculative work is extra.
- Every scored prefix is cloned as an independent task. Peak authority C is not original-source-session concurrency.
- Each policy has its own paired demand-only runs, counterbalanced AB/BA by repetition.
- Scheduled response includes event-loop arrival lateness plus broker exposed wait. It is the primary authority metric.
- Peak authority C is the maximum number of confirmed calls outstanding at once; offered load is exogenous URL-call utilization.
- A queued promotion is not an overlap-producing hit.
- rank5 policies are full-fire legacy controls; rank_budgeted_reserved is the equal one-start rank-only control.
- Top-10% start share measures priority concentration across decision sessions; Jain allocation breadth is retained in metrics.json.
- Reserve bounds simultaneous speculative visit work but cannot preempt a call that already started.
- All-wrong preserves candidates, probabilities, and the arrival plan while replacing every authoritative URL; causal backlog feedback can still change later admissions.
- Any positive zero-overlap or all-wrong net estimate is timing noise, not a latency-benefit claim; inspect repeat median/min.
- Net latency is scheduler-marginal: Pattern feature extraction and OOF probability lookup are precomputed and excluded.
- This is development evidence from nested grouped OOF traces, not the untouched confirmatory holdout.

## Reproduction

```bash
PYTHONPATH=reproduction:reproduction/scripts python reproduction/scripts/run_pattern_v2_open_loop_stress.py --traces /home/aiscuser/PASTE-Qwen-DR/traces/my_traces --output reproduction/results/pattern_v2_open_loop_stress --loads 0.5,0.9,1.2 --policies rank5_unreserved,confidence_reserved,utility_risk_limited --repetitions 4 --cycles 1 --workers 4 --visit-capacity 2 --max-speculative-pending 128 --service-ms 5.0 --lead-ms 2.5
```
