# New-trace live LLM/tool co-scheduling: quick result

## Result

The requested end-to-end gain is **not present in the current implementation**.
The new all-Visit speculative executor works and removes about 55% of exposed
Visit time, but every tested co-scheduler pressure setting is slower than the
original vLLM FCFS baseline on mean task completion time.

This is a one-run quick scan, per the instruction to avoid repetitions and get
an early conclusion. It is an engineering result, not a confidence-interval
claim.

| Mode | `P_low–P_high` | Mean task flow | p95 flow | Change vs FCFS | Mean LLM | Mean admission/turn | Visit exposed/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original vLLM FCFS | — | **373.24 s** | **632.47 s** | — | 39.06 s | 0.00 s | 24.84 s |
| PASTE, active gate | 24–40 | 399.18 s | 709.82 s | **+6.95% slower** | 25.62 s | 17.97 s | 11.18 s |
| PASTE, active gate | 32–48 | 509.27 s | 875.90 s | +36.44% slower | 37.76 s | 18.44 s | 11.18 s |
| PASTE, active gate | 40–56 | 462.24 s | 836.82 s | +23.84% slower | 40.18 s | 10.63 s | 11.18 s |
| PASTE, active gate | 48–64 | 446.84 s | 817.70 s | +19.72% slower | 41.78 s | 7.27 s | 11.18 s |
| PASTE, gate effectively inactive | 48–80 | 383.30 s | 689.37 s | **+2.69% slower** | 41.77 s | 0.00 s | 11.18 s |

The best treatment is the effectively ungated 48–80 point, but it is still
2.69% slower on mean E2E and 9.00% slower at p95 than FCFS. Among settings that
actually gate, 24–40 is best, but remains 6.95% slower on mean E2E.

## What actually ran

- Workload: the frozen materialized `0.42x` trace, 100 sessions, 873 real LLM
  requests, and 599 tool events. Every valid cell completed with zero failures.
- Baseline: unmodified vLLM FCFS, no speculative Visit work and no PASTE
  admission queue.
- Search and non-Visit tools: real `asyncio.sleep` for the corrected trace
  service time.
- Visit: a real asynchronous shared wall-clock pool. Exact in-flight predictions
  are promoted without losing work; completed predictions use a session-scoped
  URL cache; authority can preempt the lowest-score wrong prediction.
- Selector: causal nested out-of-fold all-Visit candidates, `W=5`, `cap=10`.
  The frozen plan contains 530 decisions and 2,093 selected candidates.
- Treatment execution: 1,502 physical speculative Visit jobs and 222 physical
  authority jobs served 499 authority calls. There were 277 cache hits
  (55.51%), all ready before authority, and 13.74 seconds of saved Visit service
  per task. Call amplification was 3.455x.
- Shared Visit capacity was 128 with at most 64 active sessions, so no tool-side
  preemption occurred in these cells. The promotion/preemption paths are covered
  by unit tests, but only ready-result reuse was exercised by this workload.

## Why the system-level gain disappears

The executor consistently reduces exposed Visit time from 24.84 to 11.18
seconds per task. The loss is on the LLM side:

- At 24–40, decoding gets faster, but the scheduler adds 17.97 seconds of
  pre-engine wait to each LLM turn. Across 8.73 turns per task, that cost is
  larger than the saved Visit time.
- At 48–64, admission wait falls to 7.27 seconds, but mean LLM latency rises to
  41.78 seconds, so saved tool time is again converted into LLM-side delay.
- At 48–80, admission is effectively inactive (`max_running=64 < P_high=80`).
  It has the best treatment E2E, yet the Joint-v2 LLM path is 2.72 seconds slower
  per turn than FCFS, enough to consume the 13.74-second per-task Visit saving.

This directly demonstrates the mechanism described in the requested Figure 9
text, but with a negative outcome: avoiding both under-utilization and overload
has not yet been achieved for this trace.

## Co-scheduler knob definitions

The implemented pre-engine controller uses:

`priority(i) = gain_weight × ExposedToolGain(i) / LLMPressure(i, load) + aging_weight × wait(i)`

and the concrete pressure definition:

`EnginePressure(B) = active_decode_requests + γ × Σ(predicted_KV_tokens / context_ref_tokens)`

The band controller admits only when projected pressure is at most `P_high`,
opens admission aggressively below `P_low`, softly limits cold sessions with
`cold_session_cap`, and always admits one physically feasible request from an
empty engine to guarantee progress. vLLM's physical-KV target remains a second
safety boundary.

The executable matrix defines one-factor-at-a-time axes for pressure band,
`cold_session_cap`, `gain_weight`, `γ`, and physical-KV target. In this quick
run only the pressure band was swept because the first completed cells showed
it dominated the result; the other axes were intentionally stopped rather than
reported as if measured.

Recommended provisional defaults for further development are `P_low=24`,
`P_high=40`, `cold_session_cap=48`, `gain_weight=1`, `aging_weight=0.02`,
`γ=1`, and physical-KV target `0.93`. This is the least-bad **active**
co-scheduler point, not a claimed improvement over FCFS.

## Validity notes

- The exact same frozen plan and prompt-token inputs were used in every cell;
  its SHA-256 is
  `4844d8eeaa5922bd015b3dd935c9070802460f3fc881a58e820b6ebc89eac753`.
- Actual generation tokens varied from 485,195 to 499,800 across online cells
  despite deterministic request settings. With no repeats, small differences
  should not be over-interpreted. The observed negative E2E gaps are reported
  without significance claims.
- Two canceled cells never produced a valid summary and are excluded: one
  duplicate center requested before the no-repeat instruction, and an unneeded
  56–72 point stopped after 48–64 was already on the overload side.
- Full machine-readable values and evidence paths are in `metrics.json`.

## Artifacts

- Live runner: `reproduction/scripts/run_trace_all_visit_live.py`
- Runtime scheduler and Visit executor:
  `reproduction/paste_repro/trace_coscheduler.py`
- Prospective matrix and all knob definitions:
  `reproduction/scripts/run_trace_all_visit_coscheduling_matrix.py`
- Frozen workload plan:
  `reproduction/artifacts/trace_all_visit_coscheduling/plan/prepared_plan.json`
- Consolidated machine-readable result: this directory's `metrics.json`
