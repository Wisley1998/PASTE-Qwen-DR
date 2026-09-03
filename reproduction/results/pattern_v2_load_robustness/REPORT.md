# Pattern-v2 robustness under low predictability and high load

## Bottom line

The premise `Top-1 ≈27.8% / hit rate 93.8%` is not a reproducible Pattern-v2 metric pair. On the 100-session whole-session grouped-OOF replay, exact target Top-1 is 21.2%; grouped-OOF recall at the frozen runtime width Top-5 is 56.8%.

The nearby 92.8% number is an evaluation-only bounded-pool target oracle (visit-window coverage 94.8%). Firing every admitted candidate union would issue 15486 candidates and expose this ceiling, with only 1.4% candidate precision. The delivered v2 runtime never does this: it is frozen at Top-5.

The repository's shared-capacity broker implementation was then stressed with synthetic service. At the widest tested runtime prefix, conservative exposed-wait benefit is +0.17 ms/target at burst width 1 and -32.40 ms/target at width 128. The drained workload-time result is reported separately because wrong-candidate cleanup can hurt throughput even when confirmation-to-result wait improves.

## Static Pattern-v2 quality and logical waste

| K | Exact target recall | Visit-window coverage | Candidates | Candidate precision | Logical waste | Logical invocation-equivalent upper envelope |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50/236 (21.2%) | 50/116 (43.1%) | 314 | 15.9% | 264 (84.1%) | 2.12x |
| 2 | 84/236 (35.6%) | 71/116 (61.2%) | 628 | 13.4% | 544 (86.6%) | 3.31x |
| 3 | 105/236 (44.5%) | 77/116 (66.4%) | 942 | 11.1% | 837 (88.9%) | 4.55x |
| 4 | 124/236 (52.5%) | 85/116 (73.3%) | 1256 | 9.9% | 1132 (90.1%) | 5.80x |
| 5 | 134/236 (56.8%) | 86/116 (74.1%) | 1570 | 8.5% | 1436 (91.5%) | 7.08x |

`Candidates` includes all gated search windows, including windows whose next tool was not `visit`; they are selected candidate demand, although capacity rejection or cancellation can prevent physical work. The final column is a logical upper envelope on the historical-label denominator (which contains one non-executable label), not a measured physical-call ratio. It assumes every selected candidate completes before unused work is cancelled.

## Closed-loop shared-pool burst sweep (CPU-only synthetic service)

Configuration: 4 shared workers, at most 2 speculative workers, visit capacity 2, pending cap 128, executor-requested sleep 5.0 ms, decision deadline 2.5 ms from batch start. Candidate-submission time consumes that deadline; observed service also includes event-loop scheduling delay and is reported explicitly. This is a closed-loop drained-burst stress, not sustained open-loop traffic. The denominator is executable HTTP(S) targets only; invalid trace labels are not dispatched.

| K | Burst width | Admission / deadline misses | Admitted exact match | Overlap-producing hit | Wrong starts (pooled) | Observed wrong service/start | Physical-call amp. | Mean exposed wait baseline→v2 | Conservative exposed net | Conservative drained wall incl. predictor baseline→v2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100.0% / 0/1020 | 21.3% | 21.3% | 792 | 5.36 ms | 2.12x | 7.94→7.22 ms | +0.24 ms (+3.0%) | 1.964→2.464 s (-25.4%) |
| 1 | 8 | 100.0% / 0/129 | 21.3% | 7.1% | 235 | 6.09 ms | 1.33x | 13.70→15.15 ms | -1.93 ms (-14.1%) | 0.837→1.048 s (-25.3%) |
| 1 | 32 | 100.0% / 0/33 | 21.3% | 1.6% | 82 | 7.34 ms | 1.12x | 36.36→38.03 ms | -2.15 ms (-5.9%) | 0.701→0.869 s (-24.0%) |
| 1 | 64 | 100.0% / 15/18 | 21.3% | 0.9% | 55 | 14.80 ms | 1.08x | 67.14→79.36 ms | -12.71 ms (-18.9%) | 0.693→0.912 s (-31.7%) |
| 1 | 128 | 100.0% / 9/9 | 21.3% | 0.6% | 35 | 25.54 ms | 1.05x | 125.20→152.32 ms | -27.60 ms (-22.0%) | 0.698→0.953 s (-36.7%) |
| 3 | 1 | 100.0% / 0/1020 | 44.7% | 35.7% | 2511 | 5.45 ms | 4.56x | 7.94→7.07 ms | +0.38 ms (+4.7%) | 1.967→3.891 s (-97.9%) |
| 3 | 8 | 100.0% / 0/129 | 44.7% | 7.1% | 385 | 6.33 ms | 1.55x | 13.61→16.23 ms | -3.11 ms (-22.8%) | 0.832→1.234 s (-48.4%) |
| 3 | 32 | 100.0% / 33/33 | 44.7% | 1.6% | 152 | 11.47 ms | 1.22x | 36.76→45.39 ms | -9.11 ms (-24.8%) | 0.708→1.091 s (-54.1%) |
| 3 | 64 | 73.8% / 18/18 | 37.2% | 0.9% | 87 | 19.20 ms | 1.12x | 67.06→84.40 ms | -17.83 ms (-26.6%) | 0.691→1.084 s (-56.9%) |
| 3 | 128 | 40.8% / 9/9 | 24.1% | 0.6% | 37 | 36.36 ms | 1.05x | 124.95→156.84 ms | -32.37 ms (-25.9%) | 0.696→1.051 s (-50.9%) |
| 5 | 1 | 100.0% / 0/1020 | 57.0% | 35.7% | 4308 | 5.46 ms | 7.11x | 7.93→7.27 ms | +0.17 ms (+2.2%) | 1.962→5.555 s (-183.1%) |
| 5 | 8 | 100.0% / 104/129 | 57.0% | 7.1% | 550 | 6.92 ms | 1.78x | 13.62→18.55 ms | -5.42 ms (-39.8%) | 0.833→1.494 s (-79.5%) |
| 5 | 32 | 87.4% / 33/33 | 54.3% | 1.6% | 193 | 13.92 ms | 1.27x | 36.29→49.49 ms | -13.69 ms (-37.7%) | 0.699→1.277 s (-82.7%) |
| 5 | 64 | 46.6% / 18/18 | 37.9% | 0.9% | 93 | 22.24 ms | 1.13x | 67.17→86.02 ms | -19.34 ms (-28.8%) | 0.692→1.165 s (-68.3%) |
| 5 | 128 | 24.5% / 9/9 | 24.1% | 0.6% | 37 | 44.72 ms | 1.05x | 125.18→157.09 ms | -32.40 ms (-25.9%) | 0.697→1.124 s (-61.1%) |

`Admitted exact match` includes queued promotion; only completed reuse and inflight promotion count as an overlap-producing hit. A positive net value means Pattern-v2 reduced exposed authoritative wait; a negative value means contention cost more than overlap saved. Requested-but-rejected candidates do no physical work. The JSON also separates admitted-never-started waste from wrong calls that actually started and records p95/p99 waits and every paired repetition. Started call counts are pooled over 3 repetitions. Conservative columns charge the full local predictor runtime serially; in practice it may overlap. `Drained wall` additionally includes candidate admission and cancellation tails, so it is the appropriate closed-loop throughput check rather than a per-session latency claim. Its percentage is an unclamped benefit: a negative value means wall time increased by that magnitude. A deadline miss means serial candidate admission itself exhausted the batch-start-to-confirmation lead budget; exact confirmation offsets and submission distributions remain in `metrics.json`.

## Mostly-wrong worst case

This counterfactual keeps the exact Pattern-v2 gate, number/order of candidates, arrival batches, service time, and authoritative target multiplicity, but deterministically replaces every target so no candidate can match.

| Burst width | Admission / deadline misses | Exact / overlap hits | Wrong starts (pooled) | Observed wrong service/start | Physical-call amp. | Mean exposed wait baseline→v2 | Conservative exposed net | Conservative drained wall incl. predictor baseline→v2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% / 0/1020 | 0 / 0 | 4710 | 5.48 ms | 7.68x | 8.15→10.46 ms | -2.79 ms (-34.3%) | 1.973→5.775 s (-192.6%) |
| 8 | 100.0% / 110/129 | 0 / 0 | 822 | 6.70 ms | 2.17x | 13.60→20.30 ms | -7.18 ms (-52.8%) | 0.831→1.745 s (-109.9%) |
| 32 | 87.4% / 33/33 | 0 / 0 | 336 | 11.52 ms | 1.48x | 36.29→51.47 ms | -15.66 ms (-43.2%) | 0.699→1.428 s (-104.2%) |
| 64 | 46.6% / 18/18 | 0 / 0 | 133 | 20.13 ms | 1.19x | 67.13→88.32 ms | -21.67 ms (-32.3%) | 0.692→1.215 s (-75.7%) |
| 128 | 24.5% / 9/9 | 0 / 0 | 50 | 45.95 ms | 1.07x | 125.33→159.88 ms | -35.03 ms (-28.0%) | 0.698→1.149 s (-64.6%) |

Worst-case behavior is fail-safe for correctness, not free for latency: speculative results never commit without an exact same-session authoritative claim. The configured speculative and visit caps are 2 and 2 within 4 global workers, and at most 128 predictions can remain pending. Those caps bound instantaneous occupancy, not cumulative waste across arriving batches. In the current broker, bulk cancellation waits for jobs one by one; while it waits for a non-preemptive wrong call, queued wrong siblings may start and must also drain. Thus even burst width 1 can execute the whole selected wrong set. Pending-cap saturation rejects new candidates while full, but cumulative wrong work grows again as slots drain and later batches arrive.

If an external visit hangs, the finite worst-case latency bound comes from the visit timeout or backend service bound, not from Pattern-v2. Without such a timeout there is no finite predictor-only bound. The all-wrong drained-wall column captures this cleanup tail for the bounded synthetic executor used here.

## Scope and reproducibility

- Prediction evidence is development-only grouped OOF over the existing 100 sessions; no genuinely unseen confirmatory trace set remains.
- Queue results use the repository's real `LiveToolBroker` and exact session-scoped confirmation. The executor requests a fixed sleep, but observed service includes event-loop scheduling delay and is reported. This is a scheduler experiment, not an end-to-end GPU/network claim.
- Load is closed-loop drained visit-window bursts only: there is no open-loop sustained arrival process, mixed search/LLM traffic, or tool start-rate gate. Exposed authoritative wait and drained workload wall answer different latency and throughput questions.
- Each paired repetition runs baseline first and Pattern-v2 second. Three repetitions are descriptive and no confidence interval is claimed.
- The 27.8% and legacy 93.8% values are not substituted into this run. All reported Pattern-v2 numerators and denominators are regenerated from the checked-in Qwen traces.
- No vLLM server, model inference, or network request is used.

Reproduce with:

```bash
PYTHONPATH=reproduction python reproduction/scripts/run_pattern_v2_load_robustness.py --traces /home/aiscuser/PASTE-Qwen-DR/traces/my_traces --artifact /home/aiscuser/PASTE-Qwen-DR/reproduction/results/pattern_cache_development/pattern_cache_policy.json --output /home/aiscuser/PASTE-Qwen-DR/reproduction/results/pattern_v2_load_robustness --widths 1,3,5 --concurrencies 1,8,32,64,128 --repetitions 3 --workers 4 --speculative-workers 2 --visit-capacity 2 --max-speculative-pending 128 --service-ms 5.0 --lead-ms 2.5
```
