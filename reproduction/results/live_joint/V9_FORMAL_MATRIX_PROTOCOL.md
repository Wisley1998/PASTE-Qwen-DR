# Prospective v9 external-live four-cell formal protocol

Status: frozen before any v9 formal GPU, network, or performance observation.
This protocol confirms the development-selected F0 policy on 80 untouched
held-out sources.  Development evidence is selection evidence only and is not
pooled with the formal estimator.

## Frozen causal chain

The formal runner must fail before starting a server unless these exact files
exist and match SHA256:

- completed development screen: `40b4a8033529883f26c1f298d54a92a69e4fcfb6cb942a8d5f70c98fc86481f3`;
- strict development selection: `7f7c9de71f341741192de78ab8596b9cb01721fe211ec3faed79ee33bd7dc7cc`;
- baseline-only selected transport: `3c44458963c65deb55b35dfa5a2ff888d5e1ec4cb6c0ff350ebe41e53612dc0d`;
- live broker, including its causal reservation ledger:
  `a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27`.

The strict selection must say that F0 passed, F1's incremental preference gate
failed, the selected policy is F0, the selected visit interval is 2.5 seconds,
and both development blocks identify F0 as visit speculation with minimum
speculative workers zero.  The transport selection must have used only A and
must attest that candidate performance was neither observed nor used.

The untouched workload is
`live_joint_wikipedia_frozen_formal_v9.json`: 80 independent sources, one task
per source per cell, raw SHA
`c15314f470d25beb709bace748357b09815a5971413de985e38beb901100ed20`,
canonical SHA
`de588fcbd46c1181156f5a6e49e0264c785c00c43e0d8c2a62698fb6217e3ce7`,
and canonical-source SHA
`750df4d7a441dc9e65fb3d32ee7594f13f14c83e281a875d08029156826e259c`.
It is disjoint from every development/tune source.  No v9 formal observation
may alter the workload, treatment, order, transport, estimator, or gates.

## Registered matrix

Run exactly three blocks, with one new vLLM process, one new broker, and an
empty result cache for every cell:

1. `A, B, E, F`
2. `B, A, F, E`
3. `A, B, F, E`

The treatment is a two-by-two factorial:

| Cell | LLM scheduler | Tool execution |
|---|---|---|
| A | native FCFS | demand only |
| B | native FCFS | exact visit speculation, F0 |
| E | Joint physical-KV | demand only |
| F | Joint physical-KV | exact visit speculation, F0 |

Every cell uses native prefix caching and disables explicit prefix-locality
reordering.  Every cell also freezes: 80 offered tasks, `max-num-seqs=96`
(`80 > 64` and `80 < 96`), 10,000 private padding tokens, three exactly-once
LLM calls, a 192-token fixed guided-final completion, four shared tool workers,
search capacity three, visit capacity two, visit start interval 2.5 seconds,
maximum speculative workers two, and minimum speculative workers zero.  B and
F use `visit`; A and E use `off`.

Bing search, Jina visit, the shared finite broker queue, and vLLM serving are
all live.  The call graph and expected visit URL are frozen; therefore this is
a true live execution/queue/serving latency experiment, not an autonomous
search-planning claim.

## Cell validity and transport gate

Each cell must contain exactly 80 successful tasks, 240 successful LLM
requests (one request and one HTTP-200 attempt for each task/call index), and
160 exact authoritative tool commits.  Each fixed-final call must use exactly
192 completion tokens and terminate by length.  Guided-JSON recovery, failed
physical jobs, non-exact/cross-session commits, recorded wait, simulated tool
sleep, leaked Joint variables in A/B, and speculative work in A/E are forbidden.

The selected transport is fail-closed: every physical tool record has zero or
one HTTP attempt; every started record has exactly attempt 1 with
`retried=false`; thus every cell has zero retry.  Every physical visit-attempt
start must be at least 2.48 seconds after the preceding global visit start (the
registered 2.5-second interval with the frozen 20 ms telemetry tolerance).
There must be zero failed job and zero wasted speculative service.  Canary
visits are skipped before speculative enqueue.

Each A block must independently reproduce real joint pressure: native LLM
waiting below the 96-sequence ceiling in at least 5% of samples,
authoritative-tool queueing in at least 5%, at least ten simultaneous
LLM-wait/tool-queue samples, and at least one continuous simultaneous interval
lasting one second with adjacent samples no more than 0.5 seconds apart.

## Estimator and promotion gates

The source is the statistical unit.  For each source and cell, average its
three block observations.  Compute paired source differences and a 10,000-draw
paired bootstrap with seed 20260817.  Server blocks and tasks are not treated
as independent replicates.

F versus E is the primary incremental live-tool effect and passes only if all
of the following hold:

- aggregate mean task-E2E reduction is at least 5%, every block is positive,
  at least 56 of 80 sources are faster, and the paired-bootstrap 95% absolute
  saving lower bound is above zero;
- combined task P95 does not regress and mean makespan is at most 1.03 times E;
- completion-token relative difference is below 1%, aggregate and per block;
- F's LLM component is not more than 1% faster than E, while exposed-tool-wait
  saving is at least the net E2E saving;
- speculative hit rate is at least 20%, wasted speculative-worker fraction is
  at most 30% (the cell-level zero-waste rule is stricter), and canary mean/P95
  ratios are at most 1.03/1.05.

A versus F is the overall system effect and passes only if mean task-E2E
reduction is at least 25%, every block is positive, at least 64 of 80 sources
are faster, and F request P99 is at most 1.25 times A.  A/B, A/E, B/F, the
factorial interaction, component decomposition, prefix telemetry, actual HTTP
attempt counts, and all external error rates must be reported even when they
are not promotion gates.  A headline of “30% improvement” additionally
requires the A-to-F point estimate to be at least 30%.

## Commands

Offline preflight (must report `gpu_or_server_touched=false`):

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_live_joint_formal_v9_matrix.py \
  formal-v9-context10k-live-r1 --check-only
```

The optional A-only boundary uses the same untouched workload and therefore is
formal evidence, not tuning.  Do not use its latency to change the matrix:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_live_joint_formal_v9_matrix.py \
  formal-v9-context10k-live-r1-baseline --cells A
```

Run the registered matrix once:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_live_joint_formal_v9_matrix.py \
  formal-v9-context10k-live-r1
```
