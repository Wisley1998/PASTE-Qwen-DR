# Live tool–LLM closed-loop experiment protocol

Status: prospective protocol.  No result may be called a live tool–LLM result
unless it passes the companion validator in
`reproduction/scripts/validate_live_joint_result.py`.

## 1. Claim boundary

The primary estimand is the incremental benefit of resource-aware speculative
tool execution while holding the LLM scheduler and prefix policy fixed:

```text
live-speculation effect = E mean task E2E - F mean task E2E
```

`E` and `F` both use Joint physical-KV scheduling and the same selected prefix
policy.  `E` uses demand-only tool execution; `F` adds resource-aware
speculation.  This comparison prevents existing queue reordering, KV admission,
or prefix gains from being relabelled as a tool-speculation gain.

The overall system comparison is `A` versus `F`.  The tool/LLM interaction is:

```text
interaction seconds = (E - F) - (A - B)
```

A positive interaction means speculation helps more under the Joint LLM policy
than under FCFS.  A combined system may pass without a positive interaction;
in that case it must be described as additive, not synergistic.

## 2. What qualifies as live

Every accepted cell must satisfy all of the following:

- LLM requests are executed by the live vLLM HTTP server.
- Search calls execute as live Bing HTML requests (`bing_html_search` at
  `www.bing.com`) and retain only Wikipedia result URLs.  This path needs no
  private credential.  Wikimedia Action and REST probes were rejected after
  stable HTTP 429 robot-policy failures at the intended concurrency; see
  `ENGINEERING_RUN_LEDGER.md`.
- Visit calls perform a fresh HTTP request through `r.jina.ai`.  Direct fetch,
  fallback, or a mixture of visit backends is not allowed inside a frozen
  comparison block.
- Demand and speculative calls enter one finite, process-wide tool worker pool.
  Per-agent schedulers, unbounded default executors, or a separate free
  speculative pool do not qualify.
- There is no recorded-wait replay, tool-side `sleep`, pre-recorded tool result,
  oracle future URL, or future authoritative invocation visible to the broker.
- A speculative result crosses the commit boundary only after an exact
  canonical invocation match.  Result state is task-isolated; same-domain or
  fuzzy-URL cache reuse is forbidden.
- Live tool HTTP uses the frozen `idempotent-get-v1` policy: at most two GET
  attempts, one fixed 1.0-second backoff, retryable statuses exactly
  `429/500/502/503/504`, and retryable exceptions exactly
  `asyncio.TimeoutError`, `ConnectionError`,
  `aiohttp.ClientConnectionError`, and `aiohttp.ClientPayloadError`.  No
  HTTP-library, proxy, wrapper, cell, or whole-run retry outside this policy is
  allowed.  A retry remains inside one physical broker job and one logical
  call; its failed first attempt and backoff remain part of that job's service
  time and task E2E.
- The formal transport is frozen to `aiohttp==3.12.15`.  Its private
  `_retry_connection` behavior is disabled and verified fail-closed before any
  request, with control version
  `aiohttp-private-retry-connection-v1`.  Consequently one counted outer
  attempt is one physical GET; an aiohttp-internal resend cannot hide behind a
  single `http_attempts` value.
- The frozen trace determines the sequence of logical calls for the primary
  causal experiment.  Generated text is still produced by vLLM and tool results
  release the next LLM request, but generated text does not change the call
  graph.  A separate autonomous-agent confirmation may be reported, but it
  cannot replace the trace-controlled latency result.

A loopback HTTP service backed by frozen payloads is useful for unit and load
testing, but is labelled `controlled_http`, not `external_live`, and cannot be
the final live-web headline.

## 3. Workload contract

The workload is immutable JSON with this shape:

```json
{
  "schema": "paste_repro.live_joint_workload",
  "version": 1,
  "split_id": "live-joint-tune-v1",
  "split_role": "tune",
  "formal_eligible": false,
  "sources": [
    {
      "source_id": "source-000",
      "question": "...",
      "language": "en",
      "prefix_group_id": "system-prompt-v1",
      "system_prompt_sha256": "<64 lowercase hex characters>",
      "steps": [
        {
          "step_index": 0,
          "kind": "llm",
          "request_template_sha256": "<64 lowercase hex characters>"
        },
        {
          "step_index": 1,
          "kind": "search",
          "arguments": {"query": "resource scheduling"}
        },
        {
          "step_index": 2,
          "kind": "llm",
          "request_template_sha256": "<64 lowercase hex characters>"
        },
        {
          "step_index": 3,
          "kind": "visit",
          "url_from": {"search_step_index": 1, "heldout_result_rank": 2}
        }
      ]
    }
  ]
}
```

The runner owns `heldout_result_rank`; the broker cannot read it until the
authoritative visit is revealed.  Predictions may use the question, completed
history, current search response, queue snapshots, live service-time estimates,
and LLM slack.  They may not use a later step, an authoritative URL, a future
response, or a trace-derived oracle wait.

Each source must contain at least one search, one visit, three LLM requests, and
one authoritative canary tool call marked `speculation_eligible=false`.  Canary
calls use the same invocation set in spec-off and spec-on cells and measure
authoritative slowdown without selection bias.

Tune and formal source IDs are disjoint.  Repeating a source raises offered load
but not the independent sample count.  Source copies receive unique session and
invocation IDs while preserving identical logical content.

`reproduction/workloads/live_joint_wikipedia_tune_v1.json` is a disjoint,
16-source tuning set for the autonomous live runner.  It may be used to screen
capacity and speculation parameters, but its simpler schema and model-selected
visit URL do not make it a frozen-call-graph primary workload.  In particular,
the first 12–20 entries of the formal 60-source file must not be reused for
tuning.

### 3.1 Validator input manifest

The validator consumes one aggregate manifest.  Counts and latency statistics
are recomputed from the evidence files; the manifest supplies policies, frozen
capacity, run ordering, and evidence bindings:

```json
{
  "schema": "paste_repro.live_joint_experiment",
  "version": 1,
  "stage": "screening",
  "protocol_sha256": "<SHA256 of this protocol>",
  "runtime": {
    "backend_mode": "external_live",
    "search_backend": "bing_html_search",
    "visit_backend": "r_jina_ai",
    "live_llm_http": true,
    "live_search_http": true,
    "live_visit_http": true,
    "shared_process_wide_tool_pool": true,
    "exact_invocation_matching": true,
    "frozen_call_graph": true,
    "baseline_only_load_selection": true,
    "recorded_wait_replay": false,
    "synthetic_tool_sleep": false,
    "future_information_used": false,
    "cross_cell_tool_cache": false,
    "generated_text_changes_tool_plan": false
  },
  "workload": {
    "manifest": {"path": "relative/path.json", "sha256": "<sha256>"},
    "source_ids": ["source-000"],
    "split_role": "tune",
    "tuning_source_overlap_count": 0,
    "copies_per_source": 10,
    "max_active_sessions": 120
  },
  "blocks": [
    {"block_id": "screen-1", "cell_order": ["A", "B", "E", "F"]}
  ],
  "cells": {
    "A": {
      "policy": {
        "llm_scheduler": "fcfs_native",
        "tool_scheduler": "demand_only",
        "prefix_policy": "native",
        "speculation_scope": "none"
      },
      "fresh_server_block_ids": ["screen-1"],
      "server_instance_by_block": {
        "screen-1": "unique-vllm-instance-id"
      },
      "result_cache_warm_start": false,
      "engine": {
        "max_num_seqs": 128,
        "max_active_sessions": 120
      },
      "tool_runtime": {
        "worker_pool_by_block": {
          "screen-1": "unique-broker-instance-id"
        },
        "worker_capacity": 16,
        "per_tool_capacity": {"search": 16, "visit": 4},
        "max_speculative_workers": 8,
        "max_speculative_pending": 128,
        "speculative_ttl_s": 60.0,
        "tool_http_max_attempts": 2,
        "tool_http_retry_backoff_s": 1.0,
        "controlled_http_retry": true,
        "tool_http_retry_policy_version": "idempotent-get-v1",
        "tool_http_retryable_statuses": [429, 500, 502, 503, 504],
        "tool_http_retryable_exception_types": [
          "asyncio.TimeoutError",
          "ConnectionError",
          "aiohttp.ClientConnectionError",
          "aiohttp.ClientPayloadError"
        ],
        "tool_http_library_retry_disabled": true,
        "tool_http_library_retry_control_version": "aiohttp-private-retry-connection-v1",
        "tool_http_library_name": "aiohttp",
        "tool_http_library_version": "3.12.15"
      },
      "evidence": {
        "frozen_config": {"path": "relative/path", "sha256": "<sha256>"},
        "task_events": {"path": "relative/path", "sha256": "<sha256>"},
        "llm_events": {"path": "relative/path", "sha256": "<sha256>"},
        "tool_events": {"path": "relative/path", "sha256": "<sha256>"},
        "resource_samples": {"path": "relative/path", "sha256": "<sha256>"},
        "prefix_samples": {"path": "relative/path", "sha256": "<sha256>"},
        "server_log": {"path": "relative/path", "sha256": "<sha256>"},
        "tool_server_log": {"path": "relative/path", "sha256": "<sha256>"}
      }
    }
  }
}
```

The example abbreviates `cells`: A/B/E/F are mandatory, and every listed block
must contain every supplied cell exactly once.  `controlled_http` may be used
for plumbing screens with both backend labels set to `controlled_http`, but the
validator never promotes it as an external-live result.

## 4. Staged exploration

### 4.1 Preflight and baseline-only load selection

Use only demand-only FCFS cell `A`.  Candidate offered loads and tool capacities
may be evaluated, but no spec-on or Joint result may be inspected until one load
is accepted.  Select the first preregistered load satisfying:

- all logical tasks, LLM requests, and tool commits complete exactly once
  (bounded HTTP attempts within one tool job do not create another logical
  call or commit);
- `VLLM_MAX_NUM_SEQS` is strictly greater than offered session concurrency;
- waiting is observed while running is below `VLLM_MAX_NUM_SEQS`;
- at least 5% of LLM timeline samples show that native queue;
- the finite tool pool has a non-empty authoritative queue in at least 5% of
  tool timeline samples; and
- both queues are simultaneously non-empty in at least one sample.

This makes both queues native consequences of resource pressure.  Neither a
fixed 64-request gate nor a private speculative capacity cap may manufacture
the comparison.

Backend protection may impose frozen search/visit sub-limits inside the one
shared global worker pool.  Each sub-limit must be no larger than the global
capacity, must apply equally to authoritative and speculative jobs, and must
be identical across A/B/E/F.  It is a concurrency guard for the real service,
not an additional pool; the validator reconstructs global and per-tool
physical concurrency from job intervals.

### 4.2 Screening

Screen with 12–20 independent tune sources, one fresh block, and enough
deterministic copies to reproduce the accepted offered load.  The minimum
four-cell matrix is:

| Cell | LLM scheduler | Tool policy | Prefix policy |
|---|---|---|---|
| A | FCFS native | demand only | native vLLM cache |
| B | FCFS native | resource-aware speculation | native vLLM cache |
| E | Joint physical-KV | demand only | selected policy |
| F | Joint physical-KV | resource-aware speculation | selected policy |

Add `C=Joint native+demand` and `D=Joint native+speculation` when decomposing
physical-KV admission.  Screening promotion requires at least 3% mean task-E2E
reduction for `F` versus `E`, at least 60% of sources faster, real speculative
hits, and the screening resource-safety gates in the validator.

The tune-time speculation scope is one of `search_only`, `visit_only`, or
`search_visit`.  Search-only is genuine speculation: the exact supplied query
is executed during the first live LLM call and remains private until that call
authoritatively emits the same invocation.  B and F must use the same selected
scope; A and E use `none`.  The scope is frozen before formal execution.

All tuning of global/per-tool worker capacity, prediction threshold, maximum
pending work, TTL, cancellation, and scheduling weights ends after screening.
An unsuccessful screen may lead to a new prospectively named screen, never to
retroactive gate changes.

### 4.3 Prefix exploration

Run these cells on tune sources with demand-only tools:

| Cell | vLLM prefix cache | Explicit prefix affinity |
|---|---:|---:|
| P0 | off | off |
| P1 | on | off |
| P2 | on | on |

`P0→P1` measures native automatic prefix caching.  `P1→P2` measures the
incremental scheduling policy.  P2 may be selected only when it improves mean
task E2E by at least 2%, raises GPU prefix-hit ratio by at least 3 percentage
points, has a positive paired-source bootstrap lower bound, and keeps task P95
within 3% of P1.  Otherwise P1 is selected.  P0 is diagnostic and is never the
headline baseline.

The selected prefix policy is then frozen identically in E and F.  This order
ensures the live-speculation headline remains an exact one-factor comparison.

The aggregate `prefix_ablation` object binds its tune `source_ids`, one or more
`block_ids`, `copies_per_source`, and `selected_policy`.  Each P0/P1/P2 cell
records `fresh_server_block_ids`, `server_instance_by_block`, and SHA-bound
`frozen_config`, `task_events`, `prefix_samples`, and `server_log` evidence.
The validator recomputes source E2E and hit ratios from those raw files; summary
values supplied by the experiment are ignored.

### 4.4 Formal promotion

Formal promotion uses exactly 60 untouched independent sources and three fresh,
paired server blocks.  Every source appears the same number of times in every
cell.  At minimum, run A/B/E/F in every block.  C/D are optional diagnostic
cells.  Within each of the A/B and E/F pairs, run order is reversed in at least
one block and the number of forward versus reverse orders differs by no more
than one.

Each cell starts a fresh vLLM server and fresh broker, has an empty result cache,
drains all admitted work, and records a unique server instance ID.  External
tool responses must not be carried from one cell to another.

## 5. Raw evidence contract

Every cell binds repository-relative paths and SHA256 values for:

- `frozen_config`;
- `task_events`;
- `llm_events`;
- `tool_events`;
- `resource_samples`;
- `prefix_samples`;
- `server_log`; and
- `tool_server_log`.

`tool_events.jsonl` contains one final row per physical job with at least:

```text
job_id, logical_call_id, invocation_id, task_instance_id, source_id,
session_id, tool, admitted, speculative, authoritative, committed,
speculation_eligible, canary, admitted_at, queue_enter_at, started_at,
authoritative_confirmation_at, finished_at, outcome, result_digest,
invocation_digest, exact_match, source, cancelled, worker_pool,
worker_id, cross_session_commit, queue_s, service_s, saved_service_s,
response_status, bytes_read, backend, request_host, http_attempts,
transport_identity_source
```

Only admitted physical jobs belong in this file; rejected prediction decisions
are recorded separately.  `committed`, rather than the broker's transient
`authoritative` lane flag, is the logical commit boundary.  For a direct
demand execution `exact_match` may be false; a committed speculative row must
have `exact_match=true`.  A queued cancellation has no worker/backend/HTTP
attempt, while every started row must have `transport_identity_source=actual`,
final `response_status=200`, positive response bytes, its actual backend and
host, and `http_attempts` in `1..2`.  A row with two attempts is accepted only
when its frozen config records `tool_http_max_attempts=2`,
`tool_http_retry_backoff_s=1.0`, `controlled_http_retry=true`, and the policy
and retryable-set values above.  It must additionally attest the exact
aiohttp library/version and successful hidden-retry disablement described
above.  A started row with a failed final outcome,
planned-only transport identity, or an unbounded/implicit retry invalidates the
cell.

`service_s` is always `finished_at-started_at`; therefore it includes every
HTTP attempt and the fixed retry backoff.  Speculative worker seconds and waste
are calculated from this inclusive service time.  Reports retain both physical
job counts and actual HTTP-attempt counts.

`task_events.jsonl` has one final row per task instance with `task_instance_id`,
`source_id`, `block_id`, `started_at`, `finished_at`, `success`,
`logical_llm_requests`, and `logical_tool_calls`.  `llm_events.jsonl` has one
final row per logical request with `request_id`, `task_instance_id`, `source_id`,
`call_index`, `submitted_at`, `started_at`, `finished_at`, `http_status`,
`success`, `attempts`, `prompt_tokens`, `completion_tokens`, and
`prefix_cached_tokens`.  The validator recomputes task and request metrics from
these raw rows rather than trusting reported means.

`resource_samples.jsonl` contains fixed-cadence shared-state samples with
`timestamp`, `block_id`, `llm_running`, `llm_waiting`,
`tool_running_authoritative`, `tool_running_speculative`,
`tool_queued_authoritative`, and `tool_queued_speculative`.  Native queue
fractions, peak queue, and simultaneous dual pressure are recomputed from this
file; self-reported summary counters are not accepted.

The aggregator must additionally prove one authoritative commit per logical
call, exact invocation/result identity for every reused or promoted result,
bounded worker concurrency, queue occupancy, cancellations, wasted worker
seconds, and the canary latency distribution.  Spec-off cells must contain zero
speculative jobs.  For a successful, one-to-one experiment:

```text
physical jobs = logical authoritative calls + speculative wastes
```

## 6. Formal acceptance gates

The validator uses fixed thresholds rather than thresholds supplied by a result
file.

Correctness and comparability:

- 100% tasks and LLM requests succeed once, with exactly one authoritative
  commit per logical tool call;
- zero LLM retry, failed physical tool job, non-exact commit, cross-session
  commit, recorded wait, or simulated tool sleep;
- every started tool job has actual final HTTP-200 evidence and one or two
  attempts under the frozen controlled policy; zero non-controlled retry;
- identical logical LLM/tool counts across paired cells;
- completion-token difference below 1% within A/B, E/F, and A/F;
- all evidence files exist under the repository and match their SHA256 values.

Live tool resource safety for F versus E:

- speculative hit rate at least 20%;
- wasted speculative worker seconds at most 30% of speculative worker seconds;
- canary authoritative mean latency no more than 1.03× E;
- canary authoritative P95 no more than 1.05× E;
- authoritative retry rate no more than 2% in every A/B/E/F cell.  The rate is
  recomputed from raw committed jobs as
  `count(commits with http_attempts > 1) / count(authoritative commits)`;
- E/F authoritative retry rates differ by at most one percentage point;
- task P95 does not regress and makespan regresses by no more than 3%.

Effective live-speculation gain:

- F versus E mean task E2E reduction at least 5%;
- paired-source bootstrap 95% lower bound strictly above zero; and
- at least 42 of 60 independent sources are faster.

Overall system promotion:

- A versus F mean task E2E reduction at least 25%;
- at least 48 of 60 sources are faster; and
- request P99 is no more than 1.25× A; and
- A/F authoritative retry rates differ by at most one percentage point (the
  all-cell 2% ceiling above also applies).

An A→F point estimate of at least 30% may be described as “30% improvement.”
Anything below it must be reported with its actual value.  Positive interaction
and its confidence interval are reported separately; they are not silently
assumed from the combined gain.

## 7. Required limitations in any report

The report must state whether the primary call graph was frozen or autonomous,
whether the backend was `external_live` or `controlled_http`, the independent
source count versus copied load-instance count, the external failure and retry
rates, actual HTTP-attempt counts, and whether C/D were run.  It must also state
that retry attempts/backoff remain in tool service, E2E, and speculative-waste
accounting.  A one-block screen is a candidate-selection
result, not formal evidence.  A Bing-search/Jina-visit experiment establishes
a live web closed loop for those tool classes; it does not establish every tool
in the paper system.
