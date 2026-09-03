# Constrained Murakkab-style emulation: M-only live result

Evidence class: `fixed-v9-setup-engineering` (not a formal-v9 matrix result).
Validated repetitions: 3; independent sources: 80.
Artifact roots used only for completion-manifest key mapping: `/home/aiscuser/PASTE-Qwen-DR`, `/home/aiscuser/PASTE-Qwen-DR-murakkab-frozen-v9`.
These are absolute system-level measurements with the registered ResNet co-load present on GPUs 4,5,6,7 at both endpoint snapshots; they are not isolated-Qwen capacity measurements.

| Metric | Value |
|---|---:|
| Runner-window throughput | 0.296089 tasks/s |
| Release-window throughput | 0.296146 tasks/s |
| LLM throughput | 0.888267 requests/s |
| Tool throughput | 0.592178 commits/s |
| Source-mean E2E | 185.178 s |
| Source E2E p50 | 179.001 s |
| Source E2E p95 | 256.609 s |
| Source E2E p99 | 263.417 s |
| Per-run task-completion makespan mean | 270.189 s |
| Physical HTTP attempts | 480 |
| Tool worker service | 569.556 s |
| Successful tasks | 100% |
| Speculative jobs | 0 |

## Per-repetition measurements

| Rep | Tasks/s | Release tasks/s | LLM req/s | Tool commits/s | Makespan (s) | E2E mean | p50 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.267677 | 0.267733 | 0.803030 | 0.535353 | 298.868 | 232.986 | 224.604 | 284.411 | 290.087 | 290.967 |
| 2 | 0.309540 | 0.309600 | 0.928620 | 0.619080 | 258.448 | 163.060 | 158.941 | 245.931 | 252.945 | 253.004 |
| 3 | 0.315892 | 0.315946 | 0.947677 | 0.631785 | 253.251 | 159.487 | 153.462 | 241.189 | 247.219 | 248.634 |

## Pooled task-level decomposition

| Component | Mean (s) | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| LLM request time per task | 108.596 | 87.042 | 184.797 | 191.583 |
| Search exposed wait | 0.183 | 0.171 | 0.246 | 0.296 |
| Visit exposed wait | 76.228 | 75.081 | 148.344 | 158.756 |
| Unattributed residual | 0.171 | 0.129 | 0.291 | 0.355 |

## Supplementary operationally excluded attempts

These measurements are disclosed for transparency but are not included in any primary aggregate above.

| Run | Rep | Classification | Tasks/s | E2E mean (s) | API overlap (s) | Worker overlap (s) |
|---|---:|---|---:|---:|---:|---:|
| m-fixed-v9-engineering-r2-20260831a | 2 | `host_co_load_contaminated` | 0.264739 | 230.735 | 104.556 | 70.836 |

The exclusion is a post-run operational decision based on independently validated external-host vLLM start/worker timestamps overlapping the timed window. Performance values had already been inspected, which is disclosed; the rule did not use a throughput or latency threshold.

All values are recomputed from raw task, LLM, physical-tool, hardware, and bound sidecar evidence. Across three repetitions, mean/median/range are descriptive only; no significance test is claimed.

This is a constrained Murakkab-style emulation with A-equivalent runtime semantics, not official Murakkab code or runtime. Singleton planning ran outside the timed path, and its overhead was not measured. This M-only run does not estimate a PASTE speedup or GPU, energy, or cost saving.

For each repetition, the before/after snapshots verify the same registered ResNet PID, process start time, boot ID, executable, argv, working directory, script SHA, and one positive-memory application row on each selected GPU. No continuous in-run background monitor was added, so this establishes endpoint process/code identity—not constant or historically equivalent utilization, power, or training intensity. The user reports that historical PASTE used the same registered ResNet setup; that remains retrospective context rather than a fresh causal comparison.
