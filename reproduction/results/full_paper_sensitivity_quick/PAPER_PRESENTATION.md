# Paper-facing sensitivity presentation

## Knob definitions

For request `i`, define the predicted exposed-tool-time reduction as

```text
ExposedToolGain_i = p_hit,i * T_tool,i,
```

where `p_hit,i` is the predicted probability that speculation produces the
next required tool result, and `T_tool,i` is the predicted execution time of
that tool call. The concrete gain coefficient `beta_G` gives

```text
ExposedToolGain_i(beta_G) = beta_G * p_hit,i * T_tool,i.
```

The predicted LLM time and pressure are

```text
T_LLM,i = context_length_i / r_prefill
          + predicted_output_length_i / r_decode,

KVUsage_B = used_KV_blocks_B / total_KV_blocks,

LLMPressure(i,B) = T_LLM,i * KVUsage_B.
```

The center calibration uses `r_prefill=38,112 token/s` and
`r_decode=113.7 token/s`. `T_LLM,i`, `KVUsage_B`, and therefore
`LLMPressure(i,B)` are dynamic measurements, not fixed center knobs.

Aging is

```text
Aging_i(alpha_A) = alpha_A * waiting_time_i,
```

where the concrete center coefficient is `alpha_A=0.2`. Thus pre-engine
priority is written as

```text
priority(i) = ExposedToolGain_i(beta_G) / LLMPressure(i,B)
              + Aging_i(alpha_A).
```

For running batch `B`, define

```text
DecodeLoad(B) = number_of_running_requests_B / 96,
KVLoad(B)     = used_KV_blocks_B / total_KV_blocks,

EnginePressure(B) = DecodeLoad(B) + gamma * KVLoad(B).
```

Here 96 is the configured maximum engine concurrency. The paper-facing center
is therefore

```text
beta_G = 0.9,  alpha_A = 0.2,  gamma = 1.
```

`ExposedToolGain`, `LLMPressure`, `DecodeLoad`, and `KVLoad` are runtime
variables and do not have one constant center value. The sensitivity knobs do:

```text
beta_G  in {0.45, 0.9, 1.8},
alpha_A in {0.1, 0.2, 0.4},
gamma   in {0.5, 1.0, 2.0}.
```

## Sensitivity analysis

| Control point | Knob | Setting | Mean task latency | Delta vs. center |
|---|---|---:|---:|---:|
| center | -- | `beta_G=0.9, alpha_A=0.2, gamma=1` | 331.76 s | -- |
| Pre-engine admission | ExposedToolGain coefficient `beta_G` | 0.45 (0.5x) | 319.89 s | -3.58% |
|  |  | 1.8 (2x) | 319.04 s | -3.84% |
| Pre-engine admission | Aging coefficient `alpha_A` | 0.1 (0.5x) | 326.88 s | -1.47% |
|  |  | 0.4 (2x) | 332.68 s | +0.28% |
| In-engine load shaping | `gamma` (KVLoad weight) | 0.5 | 328.43 s | -1.01% |
|  |  | 2 | 319.12 s | -3.81% |

Each configuration runs the same 80-task FULL workload once. All other system
components, including speculation, prefix caching, workload, and engine
capacity, remain enabled and fixed.

## Paper-ready interpretation

Across the tested range, FULL is insensitive to moderate changes in the
pre-engine gain, fairness, and in-engine KV-pressure weights: mean task latency
changes by at most 3.84% relative to center in this single-run screen. Scaling
`beta_G` from 0.45 to 1.8 gives nearly identical mean task latency at the two
endpoints. Increasing `alpha_A` from 0.2 to 0.4 slightly increases mean latency,
consistent with stronger fairness gradually trading away gain-efficient
ordering. Increasing `gamma` gives lower descriptive latency in this run,
although the lack of repetitions means this should be presented as sensitivity
rather than an optimal-parameter claim.

The internal scheduler uses several concrete signals to realize each abstract
paper term. This document defines the paper-space quantities and presentation;
machine-readable contracts retain lower-level implementation values for
reproducibility.
