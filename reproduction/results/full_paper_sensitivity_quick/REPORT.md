# FULL-only paper-knob sensitivity: quick single run

Date: 2026-09-02

## Result

The requested quick sensitivity run is complete. Every point is the same FULL
system; there are no component-off ablations. Each point ran once on the same
frozen first 80 sessions (696 LLM turns and 523 tool events), with a fresh vLLM
server. All nine points completed 80/80 tasks with zero failures and zero vLLM
preemptions.

The useful conclusion is **robustness, not an optimum**. Mean task flow spans
319.04--338.63 s (within -3.84% to +2.07% of the one-shot FULL center), while
task p95 spans 508.48--559.08 s. Because there are no repetitions and the two
four-GPU groups ran in parallel, small differences must remain descriptive.

In the paper-facing presentation, the concrete center coefficients are
`beta_G=0.9`, `alpha_A=0.2`, and `gamma=1`. `ExposedToolGain` is
`beta_G * p_hit * predicted_tool_time`;
`LLMPressure` is predicted LLM time multiplied by current KV-cache usage;
`Aging` is proportional to request waiting time; and `gamma` is the KVLoad
weight in `DecodeLoad + gamma * KVLoad`. See `PAPER_PRESENTATION.md` for the
complete notation and paper-ready table.

| Paper knob | Value | Mean task flow | Change vs FULL center | Task p95 | P95 change |
|---|---:|---:|---:|---:|---:|
| center | beta_G=.9 / alpha_A=.2 / gamma=1 | 331.763 s | +0.000% | 545.470 s | +0.000% |
| ExposedToolGain coefficient beta_G | 0.45 | 319.888 s | -3.579% | 517.809 s | -5.071% |
| ExposedToolGain coefficient beta_G | 1.8 | 319.041 s | -3.835% | 530.386 s | -2.765% |
| Aging coefficient alpha_A | 0.1 | 326.882 s | -1.471% | 525.667 s | -3.631% |
| Aging coefficient alpha_A | 0.4 | 332.684 s | +0.278% | 559.079 s | +2.495% |
| gamma | 0.5 | 328.429 s | -1.005% | 528.625 s | -3.088% |
| gamma | 2.0 | 319.119 s | -3.811% | 511.206 s | -6.282% |
| P_low/P_high | work-conserving / 0.85 | 338.632 s | +2.071% | 534.292 s | -2.049% |
| P_low/P_high | work-conserving / 0.97 | 321.395 s | -3.125% | 508.485 s | -6.780% |

Negative change means faster than the single FULL-center observation. These
percentages are not confidence intervals and should not be used to select a
new center post hoc.

## Interpretation by paper knob

### ExposedToolGain

In paper notation, this axis changes the concrete coefficient `beta_G` applied
to `p_hit * predicted_tool_time`: 0.45, 0.9, and 1.8. The two endpoints have almost identical means
(319.89 and 319.04 s). This
supports the modest claim that FULL is not fragile to a 4x gain-scale range.
It does not support a monotonic trend: both endpoints happened to outperform
the one center observation, which is consistent with center-run drift.

### Aging

In paper notation, Aging is `alpha_A * waiting_time`, with concrete values
0.1, 0.2, and 0.4. The observed direction is mild: 0.1 is 1.47% faster in mean and 3.63% faster
at p95, while 2x is 0.28% slower in mean and 2.50% slower at p95. This is
consistent with aggressive fairness weakening gain-efficient order, but the
hard rescue path never fired, so it remains a descriptive observation rather
than causal evidence for changing the default.

### gamma

In paper notation, `gamma` is the KVLoad weight in
`EnginePressure = DecodeLoad + gamma * KVLoad`. Both 0.5 and 2.0 completed
cleanly. The 2.0 point has the lowest observed mean within this axis, but the
absence of repeats prevents an optimum claim. The supported statement is that
changing the in-engine KV-load weight from 0.5 to 2 did not destabilize FULL.

### P_low/P_high

This quick workload did not activate the physical-KV boundary. Across cells,
maximum physical usage was only 0.522--0.598; every physical-admission trace
had zero budget-truncated ticks and zero rescue events. Consequently the
observed .85/.93/.97 differences cannot be attributed to the pressure ceiling.
The correct paper statement is that no sensitivity was observed *while the
band was non-binding*, not that .97 is better than .85.

## Mechanism guards

- Native prefix-cache hit ratio was stable at 66.218%--66.223% in every cell.
- Realized Visit hit rate was 28.54%--30.98%; call amplification was
  2.444x--2.544x.
- Mean exposed Visit time was 18.17--18.70 s/task, with 7.06--7.62 s/task of
  saved Visit service.
- Mean vLLM queue time was only 0.046--0.073 s/request even though instantaneous
  waiting depth reached 46--48. This reduced workload is sufficient for a fast
  robustness screen, but not for a strong overload-boundary claim.
- Generation tokens varied from 337,270 to 349,445 across online cells despite
  deterministic request settings; this is another reason not to over-interpret
  small one-shot gaps.

## FULL configuration

- Native vLLM prefix caching: on.
- Explicit prefix-affinity reorder: off.
- Joint-v2 stage/gain/pressure waiting-queue ordering: on.
- Forecast-aware physical-KV admission: on; center target 0.93.
- External Python gain-pressure admission queue: off.
- All-Visit speculation: `budget_w5_cap10`, authority-first preemptible pool,
  capacity 16 and speculative cap 8.
- Client task limit: 80; task flow includes time from workload release, so no
  client-semaphore waiting is omitted.

### Paper-facing center knob values

| Paper quantity | Center definition |
|---|---|
| `ExposedToolGain` | `beta_G * p_hit * predicted_tool_time`, with `beta_G=0.9` |
| `LLMPressure` | `predicted_LLM_time * (used_KV_blocks / total_KV_blocks)`; predicted LLM time uses context and predicted output length |
| `Aging` | `alpha_A * waiting_time`, with `alpha_A=0.2` |
| `DecodeLoad` | `running_requests / 96` |
| `KVLoad` | `used_KV_blocks / total_KV_blocks` |
| `gamma` | `1`, giving `EnginePressure = DecodeLoad + KVLoad` at center |

The numerator, denominator, and engine-pressure measurements are dynamic per
request or scheduling tick. The concrete sensitivity coefficients are
`beta_G in {0.45,0.9,1.8}`, `alpha_A in {0.1,0.2,0.4}`, and
`gamma in {0.5,1,2}`.

## Validity boundary

This run intentionally follows the request for speed: one repetition and a
reduced 80-session workload. It is suitable for a compact paper sensitivity
figure showing the observed curve and FULL's stability. It is not a statistical
significance experiment, an ablation, or evidence that the descriptive best
point is universally optimal. The initial A-group server startup failed before
any request because memory from an earlier smoke had not fully released; it is
excluded. The clean replacement launched once and is the only A-group data used.

Machine-readable values and source evidence paths are in `metrics.json`.
