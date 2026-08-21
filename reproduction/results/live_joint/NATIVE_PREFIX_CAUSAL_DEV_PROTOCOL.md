# Native prefix causal development protocol v2

Protocol ID: `native-prefix-causal-v2`. Plan schema:
`paste_repro.native_prefix_causal_plan_v2`. Cell schema:
`paste_repro.native_prefix_prompt_cell_v2`. Validator schema:
`paste_repro.native_prefix_causal_validation_v2`.

The earlier v1/r1 guided-JSON run is a **rejected diagnostic** because output
whitespace and completion-token counts were not treatment-invariant. It cannot
be rescued, revalidated, pooled, or promoted under v2. V2 uses a new artifact
root and incompatible schemas. Every v2 plan SHA-binds this protocol, the cell
runner, and the validator in `contract_bindings`; strict validation reports the
immutable `run_plan_sha256` as well as those code/protocol SHA256 values.

## Question and scope

This development-only benchmark answers one narrow question: with the current
three-call, approximately 10k-token agent prompt topology, does vLLM's native
prefix cache causally reduce prefill work and user-visible LLM flow time?

It is deliberately separate from the live-tool formal matrix.  It does not
read formal-v5 outputs, contact Bing/Jina, execute a tool, reuse a formal
server, or change the live broker, formal runner, agent, or scheduler hook.
The frozen tune workload is not formal evidence.

## Fixed matrix

The benchmark uses 16 frozen tune sources with three task-private replicas,
for 48 tasks and 144 real vLLM calls per cell.  Each task has three sequential
turns:

1. a task prompt after a deterministic, task-private 10k-token history;
2. the same prefix plus a fixed search call and local search result;
3. the same prefix plus a fixed visit call and roughly 900 tokens of local
   page content.

Every model request has exactly one allowed output: the sentinel `A`. Before a
server can start, the pinned local tokenizer must prove that `A` encodes to one
non-special token (token ID 32), decodes byte-for-byte to `A`, and that the
choice set has cardinality one. The OpenAI-compatible request uses
`guided_choice=["A"]` and `max_tokens=1`; the validator requires exactly one
completion token for every one of the 144 calls in every cell.

Later prompts still append the canonical search/visit fixture objects rather
than runtime text. Consequently the complete prompt bytes are identical across
treatments and the causal measurement retains the three-stage prefix topology,
while guided-JSON whitespace, schema serialization, free-form reasoning, and
variable decode length cannot enter the latency comparison. No sleeps or tool
queues are present.

| Cell | Native prefix cache | Scheduler |
|---|---:|---|
| P0 | off (`--no-enable-prefix-caching`) | native FCFS |
| P1 | on (`--enable-prefix-caching`) | native FCFS |

Run exactly two fresh-server blocks in reverse order: `P0,P1` and `P1,P0`.
Every inherited `VLLM_*` variable is removed and an exact allowlist is rebuilt;
only `VLLM_SCHED_POLICY=fcfs` remains among scheduler variables.  The server is
started with an empty cell-private `PYTHONPATH`, so `sitecustomize` and the
repository scheduler hook cannot load.  `max-num-seqs=96` is strictly above
the 48 offered tasks, so this experiment does not introduce a 48/64 admission
cap.  Explicit prefix-locality scheduling is absent and recorded as off.

## Integrity checks

The validator rejects the run unless all of the following hold:

- four distinct runner IDs plus four distinct OS PID/start-clock/command
  identities, all SHA-bound into cell evidence;
- the pinned vLLM 0.10.1 API startup line and V1 engine line independently
  agree on the effective cache flag;
- the server log records max model length 16384, batched tokens 2048, and max
  sequences 96;
- no scheduler-patch marker or Joint scheduler environment variable exists;
- raw before/after Prometheus snapshots and labels are preserved; the
  validator selects the single `engine=0` model series, rejects resets, and
  recomputes every delta;
- under pinned vLLM 0.10.1's token-counter contract, P0 exposes zero native
  cache queries/hits, while P1's query counter equals exact prompt tokens and
  its hit ratio is at least 60%;
- every cell completes exactly 48 tasks and 144 one-attempt requests;
- messages SHA256, rendered chat-template token-ID SHA256, prompt-token count,
  request/singleton-choice SHA256, sentinel, raw completion SHA256, semantic
  completion SHA256, and one-token completion count match exactly for each
  task/call across all four cells;
- every server-reported prompt-token count equals the local tokenizer count;
- prompt sizes follow the frozen three-stage long-context shape and never
  approach the 16384-token model limit after the configured generation cap;
- there are no failed calls or preemptions, and observed running concurrency
  remains below max-num-seqs.

## Promotion gates and stopping rule

All gates are conjunctive:

- aggregate prefill-time reduction is at least 15%, and P1 is lower in both
  blocks;
- aggregate mean request-time reduction is at least 3%;
- aggregate mean task-flow reduction is at least 3%, and P1 is lower in both
  blocks;
- the paired-source bootstrap 95% lower bound for P0 minus P1 is positive;
- P1 task P95 is at most 1.03 times P0;
- completion-token relative difference is below 1% (the strict 144 tokens per
  cell invariant makes it zero).

If all gates pass, native caching is retained and prefix exploration stops for
this workload.  If cache hits are present but latency gates fail, the result
supports only a prefill-work claim.  There is no optional third block and no
post-hoc threshold change.  Explicit prefix affinity remains disabled; a
future scheduler experiment would first require independent shadow evidence
of heterogeneous, actionable cache locality among at least two waiting
same-stage requests.

## Commands

The preflight creates no benchmark artifact and does not touch a server/GPU or
network. It locally loads the pinned tokenizer, builds all 144 prompts, binds
their manifest, checks prompt topology/context length, proves the singleton
sentinel contract, and prints all protocol/runner/validator SHA bindings:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_native_prefix_causal_dev.py native-prefix-v2-r1 --check-only
```

After any live formal run has fully stopped, execute the frozen development
matrix with:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_native_prefix_causal_dev.py native-prefix-v2-r1
```

The wrapper validates the completed matrix automatically.  It can also be
revalidated without starting a server:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/validate_native_prefix_causal_dev.py \
  --run-root reproduction/artifacts/live_joint/prefix_native_causal_dev_v2/native-prefix-v2-r1
```
