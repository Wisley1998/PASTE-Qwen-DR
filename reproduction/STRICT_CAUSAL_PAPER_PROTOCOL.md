# Strict causal, no-oracle paper protocol

Status: protocol and audit contract.  Historical hybrid results are not
promoted by this document.  Freeze a concrete policy bundle and data split
before starting a confirmatory run.

This protocol applies to both:

- `/home/aiscuser/PASTE-Qwen-DR` (DeepResearch traces); and
- `/home/aiscuser/gemini-cli-PASTE` (Gemini/SWE traces).

The primary question is whether the complete causal system reduces mean task
end-to-end latency by at least 20% relative to no speculation plus native FCFS
vLLM.  Reaching 20% is an evaluation outcome, not a stopping rule: once the
formal split is opened, no parameter may be changed and the result is retained
regardless of direction.

## 1. Two independent validity axes

`oracle_free` and `confirmatory` are different properties.

1. **Oracle-free execution** means every decision is a function only of events
   visible by that decision's timestamp.  A previously studied trace can still
   be replayed oracle-free.
2. **Confirmatory evidence** additionally requires that the evaluation roots
   were not used to fit predictors, select hyperparameters, choose load, or
   inspect performance before the policy was sealed.

The current 50%/40% hybrid artifacts fail the first property because their
policies consume future trace values.  They also cannot serve as a new
untouched confirmatory set:

- the Qwen unified 80-session run overlaps the existing fixed split in 33/40
  calibration, 24/30 tuning, and 23/30 nominal-final sessions;
- all ten Gemini held-out templates have already been evaluated repeatedly,
  and the 80 load instances are eight replicas of those ten roots.

Consequently, a repaired run on the existing traces must be labelled
**oracle-free retrospective feasibility evidence**.  A paper-confirmatory claim
requires newly collected root traces/tasks (at least 30 per repository), or an
outer split that was genuinely sealed before any outcome inspection.  Replicas
never turn an observed root into a new independent root.

## 2. Data and freeze sequence

Use whole root sessions/tasks, not individual turns.  Exact root-ID
disjointness is mandatory and machine-checked.  Near-duplicate disjointness is
a separate claim: the current retrospective cohorts have no independently
bound near-duplicate audit and are therefore recorded as `not_verified`.

1. **Calibration:** fit tool-invocation, tool-duration, output-length, and
   optional remaining-work predictors.  Tool-duration labels may be used here.
2. **Tuning:** choose Top-K/threshold, active working-set size, tool capacity,
   aging, launch TTL, and every scheduler coefficient.  Run arbitrary screens
   only on this split.
3. **Seal:** write an immutable manifest binding split identities and hashes,
   policy/config, predictor artifacts, fit and runtime code, model/tokenizer
   revision, container/packages, arrival schedule, cell orders, GPU mapping,
   estimator, bootstrap seed, and acceptance rules.  Atomically create a
   `FORMAL_STARTED` marker using exclusive creation.
4. **Evaluation:** execute the complete registered matrix once and retain all
   cells even if the effect is negative or below 20%.  Formal observations may
   never feed back into this policy.  Further optimization needs a new outer
   evaluation set.

Both predictors must record `training_role=calibration`, the SHA-256 of the
sorted calibration root IDs, exact input-feature schema, artifact hash, fit-code
hash, and `uses_evaluation_labels=false`.  Each signed predictor and service
artifact is bound twice: `sha256` covers the exact file bytes and
`identity_sha256` is the signed logical hash embedded in that JSON.  A timestamp
or a filename containing "frozen" is not sufficient provenance.

## 3. Decision-time information contract

At a decision timestamp `t`, the policy may use only:

- the task's already observed release time, wait age, opaque session ID, initial
  user input, current messages, and number of calls already committed;
- the current request's measured prompt tokens, public `max_tokens`, tokens
  generated so far, and a frozen/past-only output-length estimate;
- committed tool names, arguments, results, and service times whose completion
  events precede `t`;
- current vLLM running/waiting counts, physical KV use, prefix-cache state, and
  current tool-broker queue/running state;
- concrete candidate invocations derived from the visible prefix; and
- outputs of SHA-bound predictors trained without evaluation labels.  A
  preregistered online EWMA is causal only over already completed calls and must
  reset identically in every cell.

It may not use:

- any future call count, prompt/output length, tool name/arguments/result,
  actual tool duration, finality, or not-yet-released arrival;
- eventual hit, state-acceptance, success, readiness, or saved-time labels;
- another evaluation task's eventual outcome or a replica's prior outcome;
- an `expected_url`, authoritative next-call descriptor, or target answer from
  the sealed benchmark/executor; or
- any formal-set metric when choosing or stopping the policy.

This restriction is on the policy side of the firewall.  A
`trace_replay_causal_reveal` executor may hold a frozen authoritative graph and
recorded outcomes privately; neither the policy nor its predictors may receive
them before the corresponding reveal event.  Merely hiding the future value is
not enough: before reveal, the executor must not query whether a candidate will
match a future authority and then choose its physical service time/result.  A
candidate launch is keyed only by its normalized invocation and a policy-
independent service environment sealed before the cells.

For Qwen replay, the public plan is metadata-only (opaque trace/root identity
and arrival information).  Its recursively audited representation may not
contain `requests`, `steps`, `messages`, `tools_after`, tool names/arguments,
outcome IDs, or authority/runtime keys.  Current request contents and the
following authoritative calls live only in the sealed executor document and
cross the firewall one causal step at a time.

Use a narrow `DecisionView`, not a full trace/template object.  Every decision
records `observed_event_seq <= decision_seq`, input digest, predictor hashes,
candidate invocation digest, and monotonic timestamp.  A required poisoning
test mutates every future field and proves that all decisions up to the mutation
boundary remain byte-identical.

### Scheduler metadata

Use schema `paste.schedx.causal_prediction.v1`.  Predicted quantities have an
explicit `_hat` suffix and artifact binding.  The prior ambiguous compact fields
`n`, `rc`, `rlmt`, `npt`, `nmt`, `nw`, `nwc`, `rtw`, `eg`, and `is_final` are
forbidden in formal request metadata.

Current request values such as `pt` and public `mt` are permitted.  Examples of
permitted predictions are `po_hat`, `remaining_calls_hat`,
`remaining_llm_tokens_hat`, `tool_hit_probability_hat`,
`tool_service_s_hat`, `tool_eta_s_hat`, `remaining_tool_wait_s_hat`, and
`expected_gain_s_hat`.  The safest
first causal scheduler is the observed-pressure working-set controller with
FIFO/aging; add predicted tail terms only after calibrating them outside the
evaluation split.

For a fixed synthetic three-call application, the protocol-defined phase may
be known.  That exception does not permit reading the variable length of a real
DR or SWE trace.  A recorded realized completion length may define the current
request's deterministic replay budget, but it must not be exposed before that
request arrives or summed across future requests.  Such a run is a serving
trace replay, not an autonomous answer-quality experiment.

## 4. Fields that must leave the previous runtime path

### Qwen DeepResearch hybrid runner

The following are evaluation labels and may remain only in a sealed executor or
post-run evaluator:

- `offline_saved_s`, `offline_cache_hit_urls`, and trace-provided
  `speculation.candidates`;
- session-wide sums/maxima of `fixed_completion_tokens`, future prompt tokens,
  or saved tool time;
- scheduler values `n/rc/rlmt/npt/nmt/nw/rtw/eg`; and
- subtracting saved seconds before `asyncio.sleep`.

The old implementation constructs these at
`run_dr_trace_hybrid_pair.py:59-90`, `:359-394`, `:470-542`, and applies the
credit at `:681-688`.  Replace it with runtime invocation prediction and a real
shared broker.  The existing `tool_prediction.py` current-search-response URL
mapper and `online_learned_agent.py` live broker are useful causal components.
A formal run may use either registered scope below:

- `call_graph_mode=autonomous`, `claim_type=closed_loop_agent`: the live model
  produces the next authoritative invocation; or
- `call_graph_mode=trace_replay_causal_reveal`,
  `claim_type=systems_trace_replay`: the benchmark graph is private to a sealed
  executor, the policy seals its prediction first, and the corresponding
  authoritative descriptor/key/result/duration is revealed only after the live
  LLM request completes.

The historical mode that hands `expected_url` to the policy remains forbidden.
The causal replay scope may support a confirmatory *systems* claim on new,
previously unobserved sealed roots, but cannot be reported as autonomous agent
quality.

### Gemini/SWE hybrid runner

The current `online_speculation` mode is also not formal evidence.  It must not:

- precompute `prediction_hit`, `state_accepted`, `safe_to_speculate`, or the
  authoritative `runtime_key_sha256` from the next trace call;
- choose the true key on a hit and synthesize `wrong:*` after already knowing a
  miss;
- use true `replay_duration_s` for a hit but the predicted duration for a miss;
- preload `task_start_preparations` from all future calls; or
- calibrate `prediction_confidence` from the same held-out test decisions.

Those paths currently occur in `run_swe_trace_live_pair.py:458-485`,
`:508-578`, and `:825-880`.  Serialize the training-only pattern model, invoke
it at runtime from the initial prompt plus committed prefix, and emit a concrete
safe invocation.  The later authoritative call independently produces its key;
only canonical equality may turn it into a hit.  Unsafe write/edit tools must
abstain unless a separately audited isolated transactional executor exists.

## 5. Real speculation and tool-time prediction

The speculation policy must make a physical decision before the corresponding
authoritative invocation is revealed:

1. predict/abstain from the current visible prefix;
2. record the candidate and predicted service distribution;
3. admit, queue, and start it in the same finite worker pool used by demand;
4. keep the result private;
5. on a later authoritative call, reuse or promote only an exact canonical
   `(tool, arguments, environment/version, session-scope)` match; and
6. count every wrong, expired, cancelled-after-start, failed, and retried
   attempt as real worker/network cost.

There is no `offline_saved_s` subtraction.  Saved latency is observed from the
wall-clock race.  Prediction CPU time and admission delay are inside task E2E.
Task E2E ends at successful terminal completion, while run makespan and worker
cost include cancellation and complete broker drain.

The tool-time predictor may be a simple robust per-tool median or a richer
model, but it is trained/calibrated outside evaluation and sees only launch-time
features (tool name, visible argument shape/size/host, current queue state, and
past completed durations).  For a running job, estimated remaining time may use
elapsed wall time plus that prediction.  Actual evaluation duration becomes
visible only in the completion event.

The strongest trace-replay clock is either a real isolated tool execution or a
separate calibration-only service environment.  For the latter, precommit a
salted, deterministic empirical draw keyed by normalized
`(tool, arguments, environment/version)`.  The same key must receive the same
physical service assignment in A/B/E/F whether it arrives speculatively or on
demand.  The service artifact is private to the executor, distinct from the
duration-predictor artifact, and its assignment must remain byte-identical when
future authority keys, hit labels, `state_accepted`, and evaluation trace
durations are poisoned.  The service artifact itself declares
`future_state_accepted_invariant=true`, and the frozen runner test must actually
mutate that label and compare assignment digests; a generic attestation is not
enough.
Evaluation trace durations are diagnostic only in this mode.

### Code-only repairs after a policy freeze

If a completed matrix is invalidated by an evidence-accounting defect rather
than by the frozen policy itself, its replacement run must retain the exact
previous invocation predictor, duration predictor, tuning selection, public
plans, runtime parameters, and private service clock.  In particular, a
code-only repair may not draw a new service-clock salt or resample the
calibration pool; the preparation command must use its explicit
`--reuse-service-clock` path, and the reused signed document must regenerate
exactly from the current calibration sources when the retained salt is
supplied.  The invalid run remains quarantined and is never an input to policy
or hyperparameter selection.

Gemini's one historical frozen tuple predates the first-authority-claim ledger
repair and embeds the then-current monolithic runner hash in its artifact
builder provenance.  It may be reused only through the signed
`paste_gemini.swe_strict_legacy_frozen_compatibility.v1` certificate and its
separately frozen independent verifier.  This is a single exact tuple
allowlist, not a generic code-hash waiver.  The verifier must, before opening
evaluation data and again in every cell/final audit: verify all three artifact
file and logical hashes; reconstruct the calibration model and retained-salt
physical clock from the bound raw calibration files; require the duration
model to equal the invocation artifact's service model; and compare the live
duration implementation bit-for-bit against the declarative
`(tool_estimate + repository_estimate) / 2` rule over the full fitted-key plus
unknown-sentinel Cartesian product.  The certificate/verifier file hashes and
certificate logical hash are carried by the manifest, matrix index, and every
cell result, and the auditor independently recomputes every emitted tool-time
estimate from the frozen model.

The historical monolithic Gemini runner bytes were not retained.  Therefore
this certificate supports only explicitly disclosed declarative and exhaustive
finite-domain behavioral compatibility for duration inference; it does not
claim source-level equivalence to those unavailable bytes.  Any other legacy
tuple, missing proof, changed feature order/model/source, resigned artifact,
fresh salt, changed sample ordering, or one-bit ETA discrepancy fails closed.

A wrong predicted invocation must therefore be physically executed or use
that independently sealed counterfactual service/result.  Using the duration
predictor's estimate as the wrong job's actual sleep is invalid because it
removes prediction error and biases waste downward.  Likewise, looking ahead
to use a recorded duration only when the candidate will hit is an execution-
side oracle even when that fact is never returned to the policy.

## 6. Registered two-by-two experiment

| Cell | LLM scheduler | Tool policy |
|---|---|---|
| A | native vLLM FCFS | demand only |
| B | native vLLM FCFS | online causal speculation |
| E | causal Joint/working-set scheduler | demand only |
| F | causal Joint/working-set scheduler | online causal speculation |

All non-treatment settings are byte-identical: model revision, tensor
parallelism, prefix caching, prompt/task set, arrival offsets, current-request
budgets, tool workers/rate limits, retry policy, and quality contract.  B and F
use the same invocation/duration artifacts and launch policy.  A/E contain zero
speculative admission/start.  Every cell uses a fresh vLLM process, broker,
workspace snapshot, and empty broker/result/evaluation-workload cache, and
drains before teardown.  The manifest attestation `empty_cache_per_cell` has
exactly that meaning.  It does not claim an empty vLLM prefix cache after the
standardized startup smoke request; prefix caching remains identically enabled
and the smoke request plus its hash are recorded for every cell.

The Qwen launcher accepts only five non-runtime path controls (config, bundle,
output root, validate-only mode, and scheduler-hook path), immediately
re-executes itself through `env -i`, and launches every server, probe, smoke,
client, teardown, and evidence process through the same clean-environment
wrapper.  `BASH_ENV` is rejected.  The formal config is a closed allowlist of
quoted literal assignments; inherited `MODEL_SNAPSHOT`, `PYTHONPATH`,
`PYTORCH_CUDA_ALLOC_CONF`, and unknown `VLLM_*`, `PASTE_*`, CUDA, NCCL, Python,
or Torch variables cannot reach a cell.  Control-plane Python processes use
Python 3.10's `-I` isolated mode.  The vLLM process and its explicitly enabled
V1 multiprocessing children instead run from a per-cell empty, non-writable
working directory with `PYTHONNOUSERSITE=1`, `PYTHONSAFEPATH=1`, and
`PYTHONPATH` set to a per-cell directory containing only symlinks to the two
SHA-bound hook files.  Because the pinned Python 3.10 predates `-P` and ignores
`PYTHONSAFEPATH`, the bound `sitecustomize` additionally replaces the standard
filesystem finder with a wrapper that filters this working directory on every
import (including after CPython inserts the late `-m` path entry).  This
preserves automatic hook loading in the spawned EngineCore while excluding
caller-CWD and unregistered `PYTHONPATH` imports; the allocator setting is
assigned by the frozen server launcher.  The
patched `Scheduler.schedule` writes an exclusive first-call marker containing
its process identity, policy, scheduler API, and hook SHA-256.  E/F must produce
that marker from a live descendant of the managed server during the standardized
smoke request, while A/B must lack it after both smoke and the full cell.  Thus
an install-time log alone is not accepted as proof that the EngineCore used the
registered policy, and caller directories containing shadow `vllm`, `aiohttp`,
or `sitecustomize` modules are ignored.

The model directory is derived exactly from canonical
`HF_HOME/models--MODEL_ID/snapshots/MODEL_REVISION`.  Before evaluation opens,
the policy freeze records a content inventory for every file in that snapshot:
relative path, byte size, and SHA-256 of the symlink-resolved contents.  Thus
weights, indexes, tokenizer files, templates, and custom code are bound rather
than trusting a revision label or `config.json` alone.  The matrix verifies the
same inventory before every server start, and the cell runner verifies it at
start and end; its identity is retained in per-cell environment evidence.

The frozen policy bundle also preregisters the complete workload-instance
contract.  For every instance it binds the stable task ID, source-root ID,
release offset, and number of live LLM requests.  Qwen obtains the first three
from its metadata-only public plan and obtains only the request count from the
signed sealed plan during post-run auditing.  Gemini obtains the instance map
from its policy plan and the request count from the policy-bound signed sealed
trace.  Future tool authorities remain absent from the analysis manifest.  A
result must contain exactly `runtime_parameters.parameters.workload_instances`
task rows, and the analyzer must match every task and contiguous request index
to this frozen contract.  Cross-cell equality alone is insufficient: deleting
the same replica or request from all 16 cells is a protocol failure.

Report all contrasts:

- `A->B`: speculation under FCFS;
- `A->E`: scheduler without speculation;
- `E->F`: incremental speculation under Joint;
- `B->F`: scheduler with speculation;
- `A->F`: combined system (primary); and
- interaction in seconds: `(E-F) - (A-B)`.

Do not relabel `A->F` as the tool-only effect.

## 7. Repetitions, card exchange, and statistics

Run at least one complete four-block Williams cycle, with a fresh server for
every cell:

1. `A, B, F, E`
2. `B, E, A, F`
3. `E, F, B, A`
4. `F, A, E, B`

Use GPUs 0-3 for blocks 1/3 and 4-7 for blocks 2/4 (or an equally balanced,
predeclared mapping).  Do not run paired cells concurrently.  Record GPU UUID,
driver, clocks, memory, utilization, and every unrelated process before and
after each cell.  The existing ResNet process remains alive and becomes a
recorded common background condition; it is never terminated for this study.

The statistical unit is the independent root trace/task.  First average load
replicas within root and block, then average blocks within root.  Bootstrap
paired root vectors (at least 10,000 deterministic draws); never bootstrap
turns, requests, 80 copied Gemini instances, or server cells as independent
tasks.  Report the 95% CI, per-block effects, fraction of roots faster, p50/p95,
makespan, request latency, throughput, failures, emitted-candidate precision,
broker-acceptance precision, physical-start-conditioned candidate precision,
abstention, and speculative worker waste.  Gemini needs at least 30 new independent task roots;
ten roots do not become `n=80` through replication.

For relative contrast `X->Y`, the registered headline estimand is the ratio of
paired root means, `(mean_root(X)-mean_root(Y)) / mean_root(X)`, after both
folding steps.  Each bootstrap draw resamples roots and recomputes that ratio.
Also report the mean of root-specific ratios as a clearly marked secondary
descriptive quantity; do not silently substitute it for the headline.  The
interaction is the mean seconds per root of `(E-F) - (A-B)`.  Before estimating anything,
require an identical source-root multiset, replica identities, release schedule,
request digests (covering prompt and public max-token work), successful-task
set, and actual completion-token work in all bound cells.  A work mismatch is a
validity failure, not a covariate adjustment.

The preregistered practical pass rule for each repository is:

```text
A->F mean relative E2E reduction >= 0.20
and paired-root bootstrap 95% CI lower bound > 0
```

The cross-repository headline passes only if both repositories pass separately.
This supports “the point estimate exceeds 20% with evidence of positive gain.”
The stronger literal claim “the true speedup is at least 20%” requires the 95%
CI lower bound itself to be at least `0.20`.  Always print the actual estimate
and CI even when a gate fails.

## 8. Fail-closed evidence and conservation checks

Before launch:

- prove exact calibration/tuning/evaluation root disjointness; for a
  confirmatory claim, additionally bind an independently generated
  near-duplicate audit for those exact registered root sets;
- verify every frozen file and predictor artifact hash;
- verify no prior evaluation root exposure for a confirmatory label;
- verify exact A/B/E/F config diffs, Williams orders, and balanced GPU mapping;
- scan source/request metadata for the legacy oracle fields; and
- run future-poison invariance and predictor re-execution tests.

During/post-run:

- every policy input event precedes its decision; prediction precedes
  authoritative reveal and speculative start;
- each speculative hit is an exact canonical match, with no cross-session
  result leak and no result visibility before authoritative commit;
- measured `service_s = finish-start` includes retries/backoff, and all started
  jobs have physical transport/executor evidence;
- demand and speculation obey the same global/per-tool capacity; wrong work and
  partial cancellation consume worker seconds;
- physical service assignment is policy-independent, equal for identical
  normalized invocation keys across all cells, and invariant to poisoning
  future authority/hit/evaluation-duration fields;
- the number and arguments of physical tool invocations are reconstructed only
  from the authoritative tool name/arguments; missing, malformed, non-finite,
  zero, or length-mismatched evaluation `duration_s`/`unit_duration_s` values
  may change only the owner-readable diagnostic sidecar;
- all tasks use the same release schedule and root IDs; failures are not
  silently dropped; all-success is a validity gate unless an ITT failure rule
  was preregistered;
- server HTTP counts, model/token work contract, fresh instance IDs, empty
  broker/result/evaluation-workload cache, standardized smoke-warmed prefix
  state, and full broker drain agree with raw logs; and
- summaries and bootstrap results are recomputed from immutable raw events.

Each result records `experiment_started_monotonic_s` and
`experiment_ended_monotonic_s`.  Every task records
`scheduled_release_monotonic_s`, `released_at_monotonic_s`, and
`task_terminal_monotonic_s`, with the scheduled release equal to the cell
origin plus the preregistered `release_offset_s`.  All task-scoped prediction,
LLM-completion, authority-reveal, tool-completion, and physical-speculation
timestamps must lie between that task's scheduled release and terminal event.
`flow_s` and `e2e_s` are redundant checksums only: the primary analysis always
recomputes task E2E as `task_terminal_monotonic_s -
scheduled_release_monotonic_s`.  The auditor rejects any disagreement, including
after result and analysis hashes have been regenerated.

For successful one-to-one execution, started physical jobs partition into the
unique jobs that satisfy authoritative commits plus started speculative waste;
queued-never-started cancellations are recorded separately with zero service.
Retries are HTTP attempts within a job, not extra logical calls.

For `trace_replay_causal_reveal`, raw evidence additionally contains prediction
seals, an independent `speculation_execution_events` ledger, live-LLM
completions, and authoritative reveals.  Do not mutate a sealed prediction row
later to add asynchronous outcomes.  Every execution ledger row joins a unique
prediction/candidate/job, and every actual speculative start occurs after its
decision and before the corresponding LLM completion/reveal.  The count of
unique execution-ledger candidates with a non-null physical-start timestamp
must equal every declared physical-start aggregate, including the broker's
`physical_speculative_starts` counter; broker-accepted count may be larger.
Candidate-level `admitted` is a legacy alias for `broker_accepted`, not proof
of worker occupancy; broker-accepted but
never-started work terminates with zero service.  For a running prediction promoted by authority, charge
`physical_start -> authority_claim` to `speculative_resource_s` and
`authority_claim -> terminal` to `demand_resource_s`; a prediction completed
before claim remains entirely speculative.  Here `authority_claim` is the
immutable first authority claim on that physical job: later callers that reuse
or join the same cached job cannot move the resource boundary.  Each raw state
transition in both registered strict result schemas carries that first-claim
timestamp (or explicit null), and the auditor reconstructs it independently
rather than trusting the top-level job projection.  The per-job parts must sum to
`total_worker_service_s`, and the broker must separately conserve speculative,
promoted-demand, direct-demand, and total worker occupancy across the cell.
The shared auditor does not trust those four summaries alone: it reconstructs
direct demand from raw Qwen tool events as non-`Visit` `service_s` plus each
`VisitResult` whose `source=executed`, while the independent speculation ledger
reconstructs speculative and promoted work.  It then checks the full sum.
The authority's key, arguments, result, and recorded duration
may first appear at or after reveal.  A future-field poisoning test must change
the private graph/outcomes and leave every earlier policy decision, admission,
and physical-service assignment digest invariant.

Prediction accuracy is reconstructed the same way.  For Qwen, the auditor
unions `tool_events[].authority_candidate_invocation_digests` for each
`(trace_id, request_index)` and compares every sealed candidate digest against
that raw post-reveal set.  For Gemini, it compares the candidate digest against
the raw `pool_authority_key_sha256`.  The emitted precision denominator is every
sealed candidate.  Broker-accepted precision is a queue-conditioned diagnostic.
The paper's `physical_started_candidate_precision` denominator is reconstructed
only from unique execution-ledger candidates having a non-null
`physical_started_at_monotonic_s`; legacy aggregate names
`admitted_candidates` and `admitted_candidate_precision` are exact aliases of
that physical-start-conditioned metric.  Thus queued-never-started candidates
enter neither its numerator nor denominator.  Outcome booleans, hit counters,
and precision summaries are redundant checksums only.  Duration MAE likewise uses
only the pre-authority `authority_eta_hat_s`/`tool_service_s_hat` and the
independent executor's `execution_surface_service_s`/`assigned_service_s`.
Recorded evaluation-trace duration and its diagnostic error never enter this
metric.

Any failed validity check exits nonzero and labels latency numbers diagnostic
only.  There is no fail-open fallback to a historical result.

## 9. Machine-checkable audit

Create the pre-run seal from explicit root lists and content-bound files.  For
a retrospective run, also pass a non-empty `--exposed-roots`; the materializer
will not permit that manifest to claim confirmatory eligibility.

```bash
python reproduction/scripts/materialize_strict_causal_manifest.py create \
  /run/sealed-manifest.json --claim-scope retrospective \
  --calibration-roots calibration.txt --tuning-roots tuning.txt \
  --evaluation-roots evaluation.txt --exposed-roots evaluation.txt \
  --selection-protocol heldout_tuning_split \
  --frozen-file protocol=reproduction/STRICT_CAUSAL_PAPER_PROTOCOL.md \
  --frozen-file runner=reproduction/scripts/run_strict_trace_abef.py \
  --frozen-file strict_runtime=reproduction/paste_repro/strict_trace_runtime.py \
  --frozen-file tool_pool=reproduction/paste_repro/trace_coscheduler.py \
  --frozen-file mapper_code=reproduction/paste_repro/mapper.py \
  --frozen-file matrix_wrapper=reproduction/scripts/run_strict_trace_abef_matrix.sh \
  --frozen-file smoke_script=reproduction/scripts/smoke_vllm.py \
  --frozen-file start_vllm=reproduction/scripts/start_vllm.sh \
  --frozen-file stop_vllm=reproduction/scripts/stop_vllm.sh \
  --frozen-file sitecustomize=scripts/pythonhooks/sitecustomize.py \
  --frozen-file materializer=reproduction/scripts/materialize_strict_causal_manifest.py \
  --frozen-file auditor=reproduction/scripts/audit_strict_causal_experiment.py \
  --frozen-file analyzer=reproduction/scripts/analyze_strict_causal_abef.py \
  --frozen-file policy_bundle=/run/frozen-bundle.json \
  --frozen-file config=/run/formal.env \
  --frozen-file scheduler_hook=scripts/pythonhooks/sched_policy_patch.py \
  --invocation-predictor-artifact /run/invocation_predictor_provenance.json \
  --duration-predictor-artifact /run/duration-predictor.json \
  --service-clock-artifact /run/sealed-service-clock.json \
  --runtime-parameters-json /run/runtime_parameters.json \
  --invocation-feature last_completed_tool_name \
  --invocation-feature current_visible_search_result_urls \
  --invocation-feature current_visible_search_result_ranks \
  --invocation-feature current_visible_search_result_ordinals \
  --invocation-feature frozen_top_k \
  --duration-feature current_tool_name \
  --duration-feature current_normalized_visit_domain \
  --duration-feature completed_job_service_s_ewma \
  --call-graph-mode trace_replay_causal_reveal \
  --gpu-groups '0,1,2,3;4,5,6,7' \
  --williams-cycles 1

python reproduction/scripts/audit_strict_causal_experiment.py \
  manifest /path/to/sealed_manifest.json --verify-files
```

If no honest independent tuning split exists, pass an explicitly empty tuning
file and `--selection-protocol nested_cross_validation_within_calibration`.
The manifest then binds all selection to calibration-only folds and records
`evaluation_used_for_model_or_policy_selection=false`; the materializer rejects
an empty tuning split under any other label.  Never invent tuning roots merely
to satisfy a schema.

The command above is intentionally retrospective for the already exposed
traces.  For a future untouched confirmatory collection, use
`--claim-scope confirmatory`, omit `--exposed-roots`, and register at least 30
new independent evaluation roots before opening any result.  It must also pass
`--near-duplicate-evidence near-duplicate-audit.json`; that JSON has schema
`paste.paper.near_duplicate_audit.v1`, `verified=true`, a
`registered_root_sets_sha256` covering all three split lists, a non-empty
method, and an explicitly empty `near_duplicate_pairs_across_splits`.  The file
and its SHA-256 are sealed.  Without it a retrospective manifest remains valid
with a warning, but the materializer will not create a confirmatory manifest.

The `create` operation writes the content-addressed manifest and an exclusive
`FORMAL_STARTED` marker; it refuses to overwrite either.  After all registered
blocks finish, use a JSON matrix index object with top-level `provenance`,
`runtime_parameters`, and `cell_evidence`.  Every cell row contains `block_id`, `cell`, one-based
`order_position`, `started_wall_s`, `ended_wall_s`, `result_path`, fresh
`server_instance_id`/`broker_instance_id`, and registered `gpu_ids`.  Within a
block, adjacent wall intervals must follow the Williams order without overlap.
The top-level provenance and every result must exactly match the manifest for:

- runner, policy-bundle, config, and scheduler-hook file SHA-256;
- invocation-predictor file SHA-256 and signed artifact SHA-256;
- duration-predictor file SHA-256 and signed artifact SHA-256; and
- physical service-clock file SHA-256 and signed artifact SHA-256; and
- treatment-neutral runtime-parameter file SHA-256 and canonical artifact
  SHA-256.

When a full model inventory exists, top-level, row, and result provenance must
also contain the exact `model_snapshot_inventory_sha256`; omission is a hard
failure.  Every matrix row retains its complete `platform_evidence` mapping in
the final manifest, and post-run audit rehashes every file rather than keeping
only the runtime-environment entry.  The runtime-environment document is parsed
semantically: it must bind cell/block/order/GPU/server identity, the clean
environment, the non-writable empty runtime CWD, model inventory, and the
repo-specific import-path guard.  Qwen additionally retains and parses both
after-smoke and after-cell scheduler evidence; E/F must show a
`v1.Scheduler.schedule` marker from a managed EngineCore descendant with the
frozen hook hash, while A/B must show its absence.  Gemini E/F must bind the
loaded-sitecustomize/hook record and persistent PathFinder guard; A/B must have
an empty `PYTHONPATH` and no hook/runtime marker.

The runtime artifact uses schema `paste.paper.treatment_neutral_runtime.v1`.
Its logical hash is SHA-256 of UTF-8 compact, key-sorted, no-NaN JSON
`{"schema": schema, "parameters": parameters}`.  The exact mapping binds the
model/revision, host/port, TP/dtype/model length/GPU utilization, batched-token
and sequence limits, CUDA graph sizes, prefix/V1 modes, task/tool/configured
speculation capacities, timeout, public output cap, workload-instance count,
and arrival-schedule hash.  Cell scheduler/speculation, effective disabled cap,
and GPU IDs are deliberately excluded because they are treatment/block fields.

Thus a wrapper cannot seal one artifact and silently pass another to the cell.

```bash
python reproduction/scripts/materialize_strict_causal_manifest.py finalize \
  /run/sealed-manifest.json /run/evidence-manifest.json \
  --matrix-index /run/matrix-index.json

python reproduction/scripts/analyze_strict_causal_abef.py \
  /run/evidence-manifest.json --output /run/paper-analysis.json

python reproduction/scripts/materialize_strict_causal_manifest.py finalize \
  /run/evidence-manifest.json /run/final-manifest.json \
  --analysis /run/paper-analysis.json

python reproduction/scripts/audit_strict_causal_experiment.py \
  manifest /run/final-manifest.json --verify-files --require-evidence

python reproduction/scripts/audit_strict_causal_experiment.py \
  result /path/to/cell/result.json
```

It validates the split/exposure label, predictor feature allowlist and hashes,
freeze attestations, exact factorial cells, Williams/card balance, root-cluster
statistics, optional cell file hashes/instance uniqueness, the 20% and strong
20% rules, scheduler metadata, and decision/outcome separation.  A formal cell
result supplies a normalized `paper_protocol` object plus raw evidence; E/F
must contain the strict scheduler metadata and B/F must contain physical
prediction decisions.  Historical unified results intentionally fail this
checker.

The analyzer verifies all bound result hashes and the complete work-equivalence
contract, folds replicas and blocks in the registered order, recomputes all
five relative contrasts plus the interaction, and performs the deterministic
paired-root bootstrap.  Its `manifest_outcomes` object is the exact normalized
payload.  Headline E2E comes exclusively from the raw per-task monotonic clock
endpoints described above; runner summaries never enter the estimator.  Work
equivalence includes each task's ordered authoritative tool
sequence: request/ordinal, invocation digest, tool, fixed result/acceptance,
and policy-independent assigned service.  Cache source and worker accounting
are excluded because those are treatment outcomes.  Finalization does not
trust a supplied report: it verifies `analysis_sha256`, reruns this analyzer on
the bound evidence manifest, exact-compares the full report (including result
bindings and primary rule), and binds both input and report files.  The post-run
auditor repeats that recomputation, so a self-consistent forged speedup fails.
It also emits explicitly descriptive per-block and
per-cell system/mechanism metrics: makespan and throughput, request latency and
tokens, failures, prediction emission/broker-acceptance/physical-start,
exact-match precision separately over all emitted candidates, broker-accepted
candidates, and candidates that actually obtained a physical worker,
abstention, speculative/promoted/direct/total worker occupancy, useful/wasted
speculative occupancy, and duration-predictor MAE when raw tool events carry
both an ETA and realized service.  Unavailable metrics are JSON `null`, never
silently omitted or imputed.  These diagnostics do not alter the registered
root-level E2E primary estimand or its pass rule.

The checker cannot cryptographically prove that a human never viewed a trace.
That fact comes from collection and access-control records and must be stated
honestly in `previously_observed_evaluation_root_ids`.  It can ensure that a
declared retrospective run is never upgraded to `confirmatory_eligible=true`.

## 10. Claim boundary

The manifest and every result must use exactly one coherent pair:

| `call_graph_mode` | `claim_type` | Permitted claim |
|---|---|---|
| `trace_replay_causal_reveal` | `systems_trace_replay` | trace-conditioned serving, queueing, and physical speculation latency |
| `autonomous` | `closed_loop_agent` | closed-loop agent latency, provided task quality is also non-inferior |

Either mode may be retrospective or confirmatory; that axis depends on prior
root exposure and the freeze boundary, not on who supplies the call graph.  A
fresh sealed causal replay can therefore be confirmatory systems evidence, but
it does not establish autonomous agent quality.  A run with generated calls,
real isolated tools, and task correctness/non-inferiority establishes the
stronger closed-loop result and should be reported separately.  Gemini's opaque
token envelopes in particular must not be described as SWE solve quality, even
when their latency protocol is strictly causal.
