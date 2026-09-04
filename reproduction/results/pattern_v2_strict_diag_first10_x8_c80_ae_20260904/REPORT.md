# Pattern V2 + causal LLM/tool co-scheduling: small diagnostic

Date: 2026-09-04

This is a single-run engineering diagnostic, not a formal or confirmatory
result. It uses the first 10 roots in the already-frozen final-root order; no
root was selected by its outcome. Each root has eight deterministic load
replicas, so there are 80 tasks but only 10 independent source roots.

## Why the earlier C16 pilot could not evaluate co-scheduling

The earlier Pattern V2 adapter used the tool-policy name as the metadata wire
schema. The vLLM hook consequently ignored every `*_hat` field and fell back to
legacy defaults. In addition, the tail predictor's nonzero next-tool
probability was multiplied by an unrelated duration-MAE skill value of zero,
which made the tool-aware bonus identically zero.

The C16 load also offered at most 16 simultaneous LLM requests to a server with
a native 48-sequence limit. The pilot's joint cell reached only 13.49% physical
KV usage and all 1,299 recorded physical admission decisions admitted every
candidate. There was essentially no queue to optimize.

The adapter now sends the fail-closed causal wire schema expected by the
server. `tool_hit_probability_hat` is the calibration estimate of the event
probability; the separate duration-regression reliability is retained only as
`tool_eta_reliability_hat` and no longer zeros that probability.

## Diagnostic setup

- Workload: first 10 frozen final roots, eight load replicas per root, 80 tasks.
- Arrival: simultaneous closed burst; client active-task limit 80.
- vLLM: native `max-num-seqs=48`, so the load creates a real waiting queue.
- Tool pool: Visit capacity 64 and speculative capacity 64.
- Tool service: the same normalized-invocation hashed SLO clock in all cells;
  Search/Scholar use 1--3 seconds and each Visit URL uses 2--8 seconds.
- Tool ETA: calibration-only population prediction from current call index and
  completed tool waits. Evaluation trace execution time is not a runtime input.
- Output work: fixed public cap of 128 tokens for every LLM request.

All four selected cells completed the same 640 semantic LLM requests,
3,288,176 prompt tokens, 81,920 completion tokens, and 592 authoritative tool
events, with zero failed tasks. B required two recorded transient transport
retries and F required four; those extra attempts are included in wall time.

## Valid cells

| Cell | Scheduler | Pattern V2 | Wall/makespan (s) | Mean task flow (s) | P95 flow (s) | Mean LLM latency (s) | Tool exposed/task (s) | Visit hit | Call amp. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | FCFS | off | 138.115 | 117.391 | 134.455 | 9.688 | 39.860 | 0.00% | 1.000x |
| B | FCFS | on | 134.125 | 113.832 | 131.998 | 10.976 | 25.923 | 49.78% | 4.152x |
| E | causal joint | off | 119.492 | 91.149 | 116.728 | 6.407 | 39.856 | 0.00% | 1.000x |
| F | causal joint | on | 116.272 | 87.066 | 112.243 | 7.084 | 30.287 | 37.72% | 3.759x |

The selected paths are `A`, `B_r2`, `E_r2`, and `F`. The original `B` and `E`
directories each contain one incomplete task after a transient
`ServerDisconnectedError`; they are retained as failed diagnostics and are not
used in the table.

## Contrasts

Positive values mean lower latency.

| Contrast | Meaning | Wall speedup | Mean-flow speedup | P95 speedup |
|---|---|---:|---:|---:|
| B vs A | Pattern V2 only | 2.89% | 3.03% | 1.83% |
| E vs A | causal co-scheduler only | 13.48% | 22.35% | 13.18% |
| F vs A | complete method | 15.82% | **25.83%** | **16.52%** |
| F vs E | Pattern increment under joint scheduling | 2.69% | 4.48% | 3.84% |
| F vs B | scheduler increment with Pattern enabled | 13.31% | 23.51% | 14.97% |

Pattern V2's component-level effect is larger than its high-load E2E contrast:
B reduces exposed tool time by 34.96% and hides 13.98 seconds of tool service
per task, but the earlier returns raise FCFS LLM latency by 13.30%. Joint
scheduling absorbs most of that feedback in F. Relative to A, F reduces mean
LLM latency by 26.88% and exposed tool time by 24.02% simultaneously.

At the source-root aggregation level, E and F are faster than A for all 10/10
roots. B is faster than A for 7/10 roots, and F is faster than E for 7/10.
These counts are descriptive because the diagnostic has only 10 roots and one
execution per cell.

## Tool-time signal ablation

An additional `E_no_tool` cell keeps the same LLM service/context/KV scheduler
but sets both tool-time score weights to zero. It completed all 640 requests
with mean flow 95.582 seconds, p95 119.898 seconds, and wall time 127.824
seconds.

Re-enabling the non-oracle tool-time predictions (`E` vs `E_no_tool`) improves
mean flow by 4.64%, p95 by 2.64%, and wall time by 6.52%. Thus the E/A result is
not solely a generic LLM ordering effect: the predicted tool-time signal has a
separate positive contribution in this diagnostic.

## Interpretation and boundary

The failure in the earlier pilot was a setup failure, not evidence that the
LLM/tool scheduler is ineffective. Once the causal metadata is actually
consumed and the offered load creates a queue, both ablations are positive and
the complete method exceeds 20% on mean end-to-end task latency.

This test validates predicted next-tool-duration prioritization. It does not
yet claim that the scheduler consumes the live admitted/running state of each
Pattern speculative job, and the return-reservation knobs remain disabled.
Adding that feedback is a separate enhancement; it is not needed for the
positive A/B/E/F diagnostic reported here.
