# Speculative Actions baseline on the fixed Qwen-DR trace

This adapter applies the core lossless mechanism from the local
`/home/aiscuser/speculative-action` repository to the same recorded Qwen-DR
workload used by the latest PASTE trace runner. It does not regenerate the
authoritative LLM output.

## Fixed choices

- Trace snapshot:
  `traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42`
- Speculator: official `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`, BF16 on one isolated A100
- Serving: the repository's vLLM 0.10.1 environment, TP=1, native 32K context,
  FCFS, port 8200
- Prediction width: Top-3, matching the default width used in the
  Speculative Actions HotPotQA implementation
- Decode: non-thinking, deterministic temperature 0, at most 512 output tokens
- Verification: exact tool name plus exact complete canonical JSON arguments

The 8B model is preferable to 14B here because the speculator must finish
inside a recorded Qwen-DR generation window (the median window in this trace is
about one second). A larger model can be added later as a quality/latency
ablation without changing the protocol.

## Fairness contract

For every recorded `LLM call → tool call` boundary, the speculator receives
only the messages visible at the start of that LLM call. The recorded LLM
response and following tool label are never included in its request. The
speculator runs on a separate GPU. Its measured HTTP + queue + inference
latency is subtracted from the recorded LLM overlap window before any saving is
credited.

A prediction is isolated until the recorded authoritative call arrives. An
exact match reuses the predicted call; every malformed, late, or mismatched
prediction falls back to the recorded demand execution. Consequently the
answer and tool-call semantics are unchanged. Extra predicted calls are
reported as tool-call amplification.

## Deployment (no experiment)

```bash
cd /home/aiscuser/PASTE-Qwen-DR

# Already completed on this machine; this is idempotent.
HF_HOME=/home/aiscuser/hf_cache \
  /home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/download_speculative_action_model.py

set -a
source reproduction/configs/speculative_action_qwen3_8b.env.example
set +a
bash reproduction/scripts/start_speculative_action_model.sh
```

Stopping the isolated endpoint does not affect the authoritative server:

```bash
bash reproduction/scripts/stop_speculative_action_model.sh
```

## Experiment commands (do not run during deployment)

First pin and audit the inputs. This command executes neither model nor tools:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_speculative_action_trace_replay.py prepare
```

Then collect the small-model predictions. This contacts only port 8200 and
caches every raw prediction and measured latency:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_speculative_action_trace_replay.py collect \
  --server-url http://127.0.0.1:8200 \
  --model Qwen/Qwen3-8B --top-k 3 --concurrency 1
```

Finally, replay the cached predictions offline:

```bash
/home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/run_speculative_action_trace_replay.py evaluate --top-k 3
```

Outputs land under `reproduction/artifacts/speculative_action_qwen3_8b/`:

- `prepare_manifest.json`: trace and case checksums plus the no-execution audit
- `cases.jsonl`: causal speculator prompts and hidden authoritative labels
- `collection_manifest.json` / `predictions.jsonl`: deployment and raw model evidence
- `report.json` / `REPORT.md`: exact-hit, tool overhead, tool-stall, and end-to-end metrics

Use `collect --dry-run` to inspect the model request configuration without
contacting the endpoint. `--case-limit` is intended only for an explicitly
labelled smoke run, never for the final result.
