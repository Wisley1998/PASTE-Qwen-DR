# Unified workset v1: Qwen DeepResearch reproduction

This profile compares the frozen 80-session DeepResearch replay under the
same Tongyi checkpoint and vLLM engine shape:

- **Baseline:** no speculative tool savings, FIFO session admission, native
  vLLM FCFS scheduling.
- **FULL:** frozen Pattern-v2 exact-hit tool savings, a 40-session persistent
  gain/pressure admission pool, and Joint-v2 physical-KV ∩ decode-pressure
  admission with a 32-request target (33 only for the fairness lane).

The server profile is
[`configs/unified_workset_v1.env.example`](configs/unified_workset_v1.env.example).
It pins TP=4, `max_num_seqs=48`, CUDA graph size 32, prefix caching, the
40/32 working set, remaining-LLM weight 1, and realized-gain weight 0.25.

## Validity boundary

This is a systems trace replay, not a causal production trial or a new
answer-quality evaluation.  All chat requests execute live with the recorded
messages and fixed completion-token counts.  The following signals are
privileged, frozen trace-derived scheduling labels:

- `rlmt` is the exact sum of fixed completion tokens from the current request
  through the end of that task.  It is used as a remaining-LLM-work hint.
- `eg` is the exact remaining tool service removed by frozen Pattern-v2 URL
  hits.  It is zero in baseline and is weighted by 0.25 in FULL.
- FULL tool hits/readiness come from the already frozen, session-grouped OOF
  Pattern-v2 projection with exact URL confirmation.  LLM queueing and the
  residual shared tool clock remain online.

Consequently the result demonstrates the engine/co-scheduling opportunity
when those estimates are available.  A deployable claim requires replacing
`rlmt` and `eg` with causal predictors and separately checking answer quality.
The comparison command verifies the same plan, model, request/tool counts,
prompt and completion tokens, HTTP status, and raw tool result digests.

## Run the paired cells

Run from the repository root.  Set `PASTE_RUN_ROOT` to a new directory; do not
reuse or overwrite a previous evidence directory.

```bash
cd /home/aiscuser/PASTE-Qwen-DR
source reproduction/configs/unified_workset_v1.env.example

export PASTE_PLAN="/home/aiscuser/PASTE-Qwen-DR/reproduction/artifacts/dr_real_trace_hybrid_v1/plan.json"
export PASTE_RUN_ROOT="/absolute/path/to/a/new/unified-workset-v1-qwen-run"
export PASTE_PYTHON="${PASTE_ENV_PREFIX}/bin/python"

mkdir -p \
  "${PASTE_RUN_ROOT}/baseline/state" \
  "${PASTE_RUN_ROOT}/baseline/server" \
  "${PASTE_RUN_ROOT}/full/state" \
  "${PASTE_RUN_ROOT}/full/server"
```

Confirm the pinned plan before spending GPU time:

```bash
"${PASTE_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema"] == "paste_repro.dr_trace_hybrid_plan.v1"; assert p["summary"]["sessions"] == 80; print(p["plan_sha256"])' "${PASTE_PLAN}"
```

Start two isolated TP=4 servers.  Baseline changes only the scheduling policy;
all model and engine-shape variables come from the same sourced profile.

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" \
VLLM_PORT="8200" \
VLLM_STATE_DIR="${PASTE_RUN_ROOT}/baseline/state" \
VLLM_LOG_DIR="${PASTE_RUN_ROOT}/baseline/server" \
VLLM_SCHED_POLICY="fcfs" \
bash reproduction/scripts/start_vllm.sh

CUDA_VISIBLE_DEVICES="4,5,6,7" \
VLLM_PORT="8300" \
VLLM_STATE_DIR="${PASTE_RUN_ROOT}/full/state" \
VLLM_LOG_DIR="${PASTE_RUN_ROOT}/full/server" \
VLLM_SCHED_POLICY="online_joint_pacer_v2" \
bash reproduction/scripts/start_vllm.sh
```

Launch both clients close together so each cell retains its assigned four GPUs
while seeing the same host conditions:

```bash
"${PASTE_PYTHON}" reproduction/scripts/run_dr_trace_hybrid_pair.py run-cell \
  --plan "${PASTE_PLAN}" \
  --system baseline \
  --output "${PASTE_RUN_ROOT}/baseline/result.json" \
  --base-url http://127.0.0.1:8200/v1 \
  --model "${MODEL_ID}" \
  --max-active-tasks 80 \
  --tool-capacity 16 \
  --request-timeout-s 900 \
  >"${PASTE_RUN_ROOT}/baseline/client.log" 2>&1 &
PASTE_BASELINE_CLIENT_PID=$!

"${PASTE_PYTHON}" reproduction/scripts/run_dr_trace_hybrid_pair.py run-cell \
  --plan "${PASTE_PLAN}" \
  --system full \
  --output "${PASTE_RUN_ROOT}/full/result.json" \
  --base-url http://127.0.0.1:8300/v1 \
  --model "${MODEL_ID}" \
  --max-active-tasks 40 \
  --preengine-policy gain-pressure \
  --preengine-coalesce-s 3.2 \
  --preengine-prefill-tokens-per-s 38112 \
  --preengine-decode-tokens-per-s 113.7 \
  --preengine-pressure-weight 1 \
  --preengine-tool-gain-beta 0.25 \
  --preengine-aging-alpha 0.05 \
  --tool-capacity 16 \
  --request-timeout-s 900 \
  >"${PASTE_RUN_ROOT}/full/client.log" 2>&1 &
PASTE_FULL_CLIENT_PID=$!

wait "${PASTE_BASELINE_CLIENT_PID}"
wait "${PASTE_FULL_CLIENT_PID}"
```

Stop only the two managed vLLM services.  `stop_vllm.sh` checks the recorded
PID, executable, module, model, host, and port before sending `SIGTERM`.

```bash
VLLM_PORT="8200" \
VLLM_STATE_DIR="${PASTE_RUN_ROOT}/baseline/state" \
bash reproduction/scripts/stop_vllm.sh

VLLM_PORT="8300" \
VLLM_STATE_DIR="${PASTE_RUN_ROOT}/full/state" \
bash reproduction/scripts/stop_vllm.sh
```

Never terminate the pre-existing ResNet process.  It is intentionally outside
the vLLM PID/state ownership chain and may remain resident on all eight GPUs.

## Validate and compare

```bash
"${PASTE_PYTHON}" reproduction/scripts/run_dr_trace_hybrid_pair.py compare \
  --baseline "${PASTE_RUN_ROOT}/baseline/result.json" \
  --full "${PASTE_RUN_ROOT}/full/result.json" \
  --baseline-server-log "${PASTE_RUN_ROOT}/baseline/server/vllm_8200.log" \
  --full-server-log "${PASTE_RUN_ROOT}/full/server/vllm_8300.log" \
  --output "${PASTE_RUN_ROOT}/comparison.json" \
  --markdown "${PASTE_RUN_ROOT}/REPORT.md"

"${PASTE_PYTHON}" -c 'import json,sys; v=json.load(open(sys.argv[1]))["validity"]; bad=[k for k,ok in v.items() if not ok]; assert not bad, bad; print("all validity checks passed")' "${PASTE_RUN_ROOT}/comparison.json"
```

Bind the result files to the exact runner, hook, launcher, profile, plan, logs,
repository state, and GPU inventory in a fail-closed provenance manifest:

```bash
"${PASTE_PYTHON}" reproduction/scripts/write_unified_workset_manifest.py \
  --baseline "${PASTE_RUN_ROOT}/baseline/result.json" \
  --full "${PASTE_RUN_ROOT}/full/result.json" \
  --baseline-server-log "${PASTE_RUN_ROOT}/baseline/server/vllm_8200.log" \
  --full-server-log "${PASTE_RUN_ROOT}/full/server/vllm_8300.log" \
  --profile-config reproduction/configs/unified_workset_v1.env.example \
  --hook scripts/pythonhooks/sched_policy_patch.py \
  --sitecustomize scripts/pythonhooks/sitecustomize.py \
  --launcher reproduction/scripts/start_vllm.sh \
  --runner reproduction/scripts/run_dr_trace_hybrid_pair.py \
  --output "${PASTE_RUN_ROOT}/provenance_manifest.json"
```

The manifest marks evidence that old logs cannot prove (such as the original
client argv or a complete process-environment snapshot) as `unknown` or
`partial`; it does not silently treat missing provenance as verified.

## Validated result

The completed fresh pair is in
`/home/aiscuser/PASTE-Qwen-DR/reproduction/artifacts/unified_workset_v1/qwen`.
Its `comparison.json` has `valid=true` and every hard validity field is true:
mean task E2E is 514.967s for baseline and 310.755s for FULL, a 39.66%
reduction.  The p50, p95, makespan, and mean LLM-request reductions are
50.22%, 12.07%, 6.68%, and 54.11%, respectively; all 80 paired tasks are
faster.  Use the artifact's `REPORT.md` and `comparison.json` as the result of
record rather than the earlier development screens.  Its
`provenance_manifest.json` is valid with no failed checks.
