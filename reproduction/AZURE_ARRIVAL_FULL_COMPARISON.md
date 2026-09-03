# Azure arrival traces: vLLM baseline vs FULL

## Frozen comparison

This experiment changes only the top-level Agent-session release process. The
recorded Qwen-DR messages, multi-turn call graph, token budgets, corrected tool
service times, all-Visit OOF predictions, and candidate selection remain
byte-for-byte identical within each materialized plan.

The two systems are:

- `vllm_baseline`: native vLLM V1 FCFS with prefix caching enabled; no tool
  speculation and no custom scheduler.
- `full`: native prefix caching, Joint-v2 stage/gain/pressure ordering,
  forecast-aware physical-KV admission (`P_high=0.93`, 40-second rescue), and
  the all-Visit preemptible shared pool (`capacity=16`, speculative cap `8`).

The rejected client-side gain/pressure queue and explicit prefix-affinity
reordering are disabled. These choices match `FULL-center` in
`FULL_PAPER_SENSITIVITY.md`.

Both systems reuse one checksummed plan for each arrival trace. Flow time starts
at the external release timestamp and includes any delay behind the client
concurrency cap. The report records mean/p95 task flow, experiment wall time,
LLM latency, Visit hit rate, call amplification, all environment settings, and
the exact execution order.

## Arrival processes

### Azure LLM Inference Trace 2024

Each selected CSV invocation releases one complete Agent session. Azure token
counts are provenance only. The first selected invocation is normalized to
time zero; inter-arrival gaps are divided by the declared arrival speedup.

### Azure Functions Dataset 2019

The selected real `invocations_per_function` window is aggregated across
functions. Exactly 100 raw invocations are sampled without replacement from
the observed per-minute count mass. Because the public dataset does not expose
sub-minute timestamps, sampled events are placed uniformly inside their
observed minute. The seed, archive hash, member, day, window, raw count, sample
fraction, and time compression are written into the plan.

This is a real trace-shaped arrival process, not the synthetic sine generator
in the older virtual-lab runner.

## Prepare the two immutable plans

```bash
python scripts/prepare_azure_trace_plans.py \
  --base-plan reproduction/artifacts/trace_all_visit_coscheduling/plan/prepared_plan.json \
  --azure-llm-trace datasets/azure_llm_2024/AzureLLMInferenceTrace_conv_1week.csv \
  --azure-llm-variant conversation \
  --azure-llm-arrival-speedup 10 \
  --azure-functions-trace /home/aiscuser/virtual-lab-PASTE/data/azure_traces/azurefunctions-dataset2019.tar.xz \
  --azure-functions-day 1 \
  --azure-functions-start-minute 480 \
  --azure-functions-duration-minutes 20 \
  --azure-functions-time-compression 20 \
  --session-count 100 \
  --mapping-seed 20260417 \
  --functions-sampling-seed 20260903 \
  --output-dir reproduction/artifacts/azure_arrival_comparison/plans
```

## Run the paired matrix

The two frozen client caps are 96 and 72. Both satisfy the requested ceiling of
100, and the second is strictly below 80. Cell order is deterministically
shuffled to reduce wall-clock drift bias. Every cell starts a fresh server and
the runner stops that server in a `finally` block.

```bash
python reproduction/scripts/run_azure_arrival_comparison.py \
  azure-arrivals-c96-c72-r1 \
  --plan azure_llm=reproduction/artifacts/azure_arrival_comparison/plans/azure_llm_conversation_plan.json \
  --plan azure_functions=reproduction/artifacts/azure_arrival_comparison/plans/azure_functions_plan.json \
  --concurrencies 96 72 \
  --repetitions 1 \
  --gpus 4,5,6,7
```

Use `--check-only` to validate and print the complete contract without starting
vLLM. Final artifacts are `run_plan.json`, one `cell_contract.json` and evidence
directory per cell, `aggregate.json`, and `REPORT.md` under the run directory.

