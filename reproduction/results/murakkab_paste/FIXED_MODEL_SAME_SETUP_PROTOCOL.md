# Fixed-model Murakkab–PASTE comparison protocol

Status: the constrained M-only engineering execution is complete. Its primary
result is in [`M_FIXED_V9_CLEAN_ENGINEERING_REPORT.md`](M_FIXED_V9_CLEAN_ENGINEERING_REPORT.md),
with the interpretation and historical comparison in
[`MURAKKAB_FIXED_V9_COMPARISON_CN.md`](MURAKKAB_FIXED_V9_COMPARISON_CN.md).

This protocol replaces the earlier adaptive-`top_k` trace experiment as the
design for any Murakkab-versus-PASTE system comparison. The earlier experiment
remains reproducible only as a PASTE configuration-width ablation.

## Question

Under the exact fixed deployment used by PASTE, how does a constrained
Murakkab-style reactive DAG executor compare with PASTE's fine-grained
execution policy? Neither side may receive a different model, hardware,
workflow, capacity, input, or request information.

This deliberately removes the model switching, heterogeneous hardware,
autoscaling, workflow-quality changes, and cross-workflow multiplexing used by
the full Murakkab paper. If Murakkab has no useful optimization freedom in this
deployment, the experiment must report that null result rather than introduce
additional knobs.

## Shared setup

Every cell fixes:

- `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` at revision
  `4b0ac5767427a55d08a254f0367e2934976598e0`;
- vLLM 0.10.1, BF16, TP=4, one replica, 4×A100-SXM4-40GB;
- 16K context, GPU-memory utilization 0.86, 2,048 max batched tokens,
  `max-num-seqs=96`, native prefix caching on, explicit prefix-locality
  reordering off;
- 80 offered/active tasks, 10K private padding, three exactly-once LLM calls,
  and a fixed 192-token final completion;
- the same linear `LLM → search → LLM → visit → LLM` workflow and prompts;
- four physical tool workers, search capacity three, visit capacity two, two
  maximum and zero reserved speculative workers, and a 2.5-second visit start
  gate;
- the same Bing/Jina policies, timeout/retry contract, fresh server, fresh
  broker, and empty result cache.

All cells must use one shared typed-DAG frontend and registry. Workflow
onboarding and the singleton optimizer run outside the timed request path.

## What Murakkab is allowed to do

For the primary setup, the candidate counts for workflow, model, hardware,
parallelism, replica count, and SLO tier are all exactly one. The constrained
optimizer must therefore select the sole executable workflow. Runtime dispatch
submits a node only after its dependencies and authoritative inputs are ready.

It may not switch models or GPUs, resize the deployment, change TP, prune the
workflow, lower answer quality, multiplex another workflow, invent SLO tiers,
or use PASTE's `top_k` as if it were a Murakkab mechanism.

This scope retains Murakkab's declarative DAG, type checking, registry, and
dependency-respecting dispatch. It does not claim to evaluate Murakkab's full
cloud-resource optimizer.

## Cells

| Cell | Shared DAG frontend | LLM scheduler | Tool execution |
|---|---|---|---|
| M: Murakkab-fixed | on | native FCFS | demand only |
| S: PASTE speculation-only | on | native FCFS | bounded exact-match visit speculation |
| J: PASTE Joint-only | on | Joint physical-KV | demand only |
| P: PASTE-full | on | Joint physical-KV | bounded exact-match visit speculation |

Two contrasts must appear together in every headline table: M→S is the clean
same-FCFS estimate of PASTE's causal lookahead, while M→P is the constrained
fixed-DAG versus full-PASTE system endpoint. M→J, J→P, and the complete
factorial interaction are mandatory additional decompositions. Reporting only
the most favorable contrast is forbidden.

## Realistic and controlled tracks

The primary result must be autonomous: the fixed Tongyi model chooses the
authoritative URL from the current search response. PASTE may use only a
checksummed rank predictor trained on a disjoint role and late-bound to that
same visible response. Its width is fixed at one before the experiment, matching
the single predicted visit in the controlled PASTE setup. The workload's
`expected_url` may not be used for prediction or authoritative selection.

A second frozen-URL track may use the predeclared `expected_url`, but it is a
perfect-prediction mechanism upper bound and is not eligible for the headline.
The existing formal-v9 result belongs only to this retrospective upper-bound
category.

## Estimands

The source is the statistical unit. Each cell runs with a fresh server in a
four-block Latin square: `M,S,J,P`; `S,M,P,J`; `J,P,M,S`; `P,J,S,M`.
Source observations are averaged across blocks before paired differences and a
10,000-draw paired bootstrap are computed.

Required outputs are actual task mean/P50/P95/P99 E2E, makespan, one absolute
deadline's violation rate, LLM/tool-wait decomposition, HTTP attempt counts,
tool-worker service seconds, speculative waste, success rate, and authoritative
commit integrity.

All cells provision four GPUs, so GPU-count saving is zero/not identifiable.
No admission-count proxy may be called resource usage. Energy may be reported
only after integrating NVML power over an identical task-release-to-last-task
completion window. Cost requires a charging model frozen before execution.

## Evidence discipline

Formal-v9 was observed before this protocol. Its A/B/E/F data can inform the
implementation and may be shown as retrospective context, but A cannot be
renamed Murakkab and those observations cannot validate the new comparison.

Before a confirmatory run, the common frontend, runner, estimator, cell order,
code hashes, and a new source set disjoint from development and formal-v2-v9
must be frozen. Null, negative, failed, and incomplete cells remain reportable.

The machine-readable draft is
[`murakkab_paste_fixed_model_protocol.json`](../../configs/murakkab_paste_fixed_model_protocol.json).
