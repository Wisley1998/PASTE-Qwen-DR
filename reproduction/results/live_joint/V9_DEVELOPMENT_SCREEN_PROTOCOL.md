# Prospective v9 development transport/reservation screen

Status: preregistered development-only mechanism screen.  It is not formal
evidence and may not consume a formal workload.  The only admitted workload is
`live_joint_wikipedia_frozen_tune_v1.json` (16 independent sources, five
replicas per source, 80 simultaneous tasks per cell).

## Invariants

Every cell uses a fresh vLLM process on GPUs 4--7 and port 8100, native prefix
caching on, explicit prefix-locality reordering off, `max-num-seqs=96`, 10,000
private context-padding tokens, three exactly-once LLM calls, and a fixed
192-token call-2 guided grammar.  Bing search and Jina visit are real HTTP GETs
through one shared broker with four global workers, search capacity three,
visit capacity two, and a shared physical HTTP-attempt start gate.  Calls use a
frozen expected-URL graph: network execution and queueing are live, but URL
prediction is deliberately perfect/frozen.  Canary stride six gives 14 tasks
whose visit prediction is skipped before enqueue.

The exact required count per cell is 80 successful tasks, 240 successful
exactly-once LLM requests, and 160 committed search/visit calls.  Every call-2
usage must equal 192 and finish by length.  Every cell requires zero failed
physical jobs, zero wasted speculative service, and, after transport
selection, one HTTP attempt per physical record with zero retry.  All evidence
and all runner/config/code bindings are checked again between fresh servers.
The fair-reservation broker is frozen at SHA256
`a1e844d439aefa75fc5a1538f4fc23de0d9408603c99784ab7a925bec26efd27`.

No v9 formal workload, prior formal result, or treatment performance may
affect this screen's transport choice.  Every produced plan, validation, and
aggregate records `development_only=true`, `formal_eligible=false`, and
`formal_evidence_eligible=false`.

## Stage 0: baseline-only transport ladder

The transport ladder is fixed before the first live observation:

1. Run one fresh native-FCFS, speculation-off A at a 2.5 second minimum visit
   HTTP-attempt start interval.
2. Accept 2.5 seconds only if all correctness and load gates pass with zero
   retries.
3. A fresh A-only 3.0 second attempt is permitted if and only if the 2.5 second
   attempt passed every non-transport gate and its sole failed gate is the
   composite zero-retry/at-most-2%-authoritative-retry transport gate.
4. Accept 3.0 seconds only if it passes every gate with zero retries.  If 2.5
   seconds fails any other gate, or 3.0 seconds still fails, stop.  Never run a
   treatment cell.

The A load gate requires offered load 80 strictly below native limit 96,
native LLM waiting while running below the ceiling in at least 5% of LLM
samples, an authoritative tool queue in at least 5% of timeline samples, at
least 10 simultaneous LLM/tool-queue samples, and at least one continuous
one-second simultaneous interval whose adjacent monotonic and wall-clock
sample gaps are at most 0.5 seconds.

The selected value is the first zero-retry, load-qualified A interval.  Stage
1 freezes that value in every cell.  No candidate latency is observed during
selection.

## Stage 1: reverse-order mechanism screen

Only an accepted Stage-0 selection unlocks Stage 1.  Each of the following six
cells starts from a fresh server and an empty tool-result cache:

- Block 1: `E, F0, F1`
- Block 2: `F1, F0, E`

`E` is the frozen Joint LLM scheduler with demand-only tools and reservation
minimum zero.  `F0` adds exact visit speculation with reservation minimum zero.
`F1` is identical to F0 except for the bounded fair reservation minimum one.
The common-config comparison permits only cell label, formal cell attestation,
observed search coverage, speculation mode, and minimum speculative workers to
differ.  All other result configuration, scheduler configuration, selected
workload SHA, and bound code SHA must be identical.

Canary tasks must have no speculative record at all.  Each F cell must have 66
exact eligible visit hits.  E and F0 must contain no reservation dispatch or
repayment.  F1 reservation evidence is replayed in physical start order from
the broker's per-dispatch causal ledger; every row must agree with the raw
record flags and recorded before/after state.  Per-tool debt must remain in
`{0,1}`, each contested reserved start must be followed by a same-tool
authoritative repayment, and final debt must be zero.

## Estimator and gates

The statistical unit is the source, not a task replica or server block.  For
each source/cell/block, first average its five replicas.  Then average the two
block estimates for that source.  The paired bootstrap samples those 16 source
means with replacement (seed 20260817, 10,000 resamples).

F0 and F1 are independently eligible against E only if all gates pass:

- aggregate mean E2E reduction at least 5%;
- positive mean reduction in both blocks;
- at least 13 of 16 source estimates faster;
- paired-bootstrap 95% absolute-reduction lower bound above zero;
- combined task P95 no worse than E;
- mean makespan no more than 1.03 times E;
- total completion-token relative difference below 1%, both aggregate and in
  each block;
- candidate LLM component not more than 1% faster than E, preventing an LLM
  speedup from being mislabeled as a tool-overlap win;
- exposed-tool-wait saving at least as large as net E2E saving.

F1 is selected ahead of F0 only if F1 passes every E gate, improves mean E2E
over F0 by at least another 2%, improves in both blocks, and in each block has
at least six causally replayed contested reserved dispatches, all corresponding
authoritative repayments, and at least six completed-reuse ready hits.  If F1
does not meet every preference gate, select F0 only if F0 passes every E gate.
Otherwise report `no winner`.  This result remains development-only and must
be confirmed on untouched held-out v9 formal evidence before promotion.

## Commands

Offline preflight (does not touch a GPU, server, or network):

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_live_joint_v9_development_screen.py \
  v9-screen-r1 --check-only
```

Run Stage 0 only (useful as a transport go/no-go boundary):

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_live_joint_v9_development_screen.py \
  v9-screen-r1 --stage0-only
```

Resume the bound run after an accepted Stage 0, or run both stages in one
invocation:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_live_joint_v9_development_screen.py \
  v9-screen-r1 --resume-stage1

/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_live_joint_v9_development_screen.py \
  v9-screen-r2
```

The standalone aggregator accepts exactly two `--block` groups, each ordered
as `BLOCK_ID E_RESULT F0_RESULT F1_RESULT`, plus the Stage-0-selected interval.
The orchestrating runner invokes the same aggregation function and records its
SHA-bound output under the run root.
