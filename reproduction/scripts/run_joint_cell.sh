#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: run_joint_cell.sh {fcfs|joint} RUN_NAME" >&2
}

if (( $# != 2 )); then
  usage
  exit 2
fi
POLICY_LABEL="$1"
RUN_NAME="$2"
if [[ "${POLICY_LABEL}" != "fcfs" && "${POLICY_LABEL}" != "joint" ]]; then
  usage
  exit 2
fi
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: RUN_NAME contains unsupported characters" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPRO_ROOT="${REPO_ROOT}/reproduction"
ENV_PREFIX="${PASTE_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
ARTIFACT_ROOT="${PASTE_ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}"
EVAL_WORKLOAD="${PASTE_EVAL_WORKLOAD:-${ARTIFACT_ROOT}/workloads/eval_learned/prepared_workload.json}"
CALIBRATION_WORKLOAD="${PASTE_CALIBRATION_WORKLOAD:-${ARTIFACT_ROOT}/workloads/calibration_learned/prepared_workload.json}"
MAPPER="${PASTE_TOOL_MODEL:-${REPRO_ROOT}/results/tool_only/url_rank_mapper.json}"
RUN_DIR="${PASTE_RUN_ROOT:-${ARTIFACT_ROOT}/runs}/${RUN_NAME}"
SERVER_URL="${PASTE_SERVER_URL:-http://127.0.0.1:8000}"
SERVER_LOG="${PASTE_VLLM_LOG_FILE:-${REPRO_ROOT}/logs/vllm_8000.log}"
MAX_ACTIVE="${PASTE_MAX_ACTIVE_TRACES:-30}"
SPEEDUP="${PASTE_TRACE_SPEEDUP:-10}"
MAX_MODEL_LEN="${PASTE_MAX_MODEL_LEN:-16384}"
MAX_REQUEST_ATTEMPTS="${PASTE_MAX_REQUEST_ATTEMPTS:-2}"
TOOL_OVERLAP_MODE="${PASTE_TOOL_OVERLAP_MODE:-learned}"
SCHEDULER_METADATA_MODE="${PASTE_SCHEDULER_METADATA_MODE:-online}"

if [[ ! "${MAX_REQUEST_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: PASTE_MAX_REQUEST_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

for required in "${EVAL_WORKLOAD}" "${CALIBRATION_WORKLOAD}"; do
  if [[ ! -f "${required}" ]]; then
    echo "error: required artifact is missing: ${required}" >&2
    exit 1
  fi
done
if [[ "${TOOL_OVERLAP_MODE}" != "none" && "${TOOL_OVERLAP_MODE}" != "learned" ]]; then
  echo "error: PASTE_TOOL_OVERLAP_MODE must be none or learned" >&2
  exit 2
fi
if [[ "${SCHEDULER_METADATA_MODE}" != "online" && "${SCHEDULER_METADATA_MODE}" != "oracle" ]]; then
  echo "error: PASTE_SCHEDULER_METADATA_MODE must be online or oracle" >&2
  exit 2
fi
if [[ "${TOOL_OVERLAP_MODE}" == "learned" && ! -f "${MAPPER}" ]]; then
  echo "error: learned tool mapper is missing: ${MAPPER}" >&2
  exit 1
fi
if ! curl --fail --silent --show-error --max-time 3 "${SERVER_URL}/health" >/dev/null; then
  echo "error: vLLM is not healthy at ${SERVER_URL}" >&2
  exit 1
fi
if [[ "${POLICY_LABEL}" == "joint" ]]; then
  grep -F "[sched_policy_patch] installed policy=online_joint_pacer_v2 " "${SERVER_LOG}" \
    | grep -F "v1=True" >/dev/null || {
      echo "error: joint v1 scheduler install evidence is missing from ${SERVER_LOG}" >&2
      exit 1
    }
fi

export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_SCHED_PRED_OUT_ENABLE=1
TOOL_ARGS=(--tool-overlap-mode "${TOOL_OVERLAP_MODE}")
if [[ "${TOOL_OVERLAP_MODE}" == "learned" ]]; then
  TOOL_ARGS+=(
    --tool-prediction-model "${MAPPER}"
    --tool-prediction-top-k "${PASTE_TOOL_PREDICTION_TOP_K:-5}"
  )
fi
# VLLM_API_KEY, when set in the private shell environment, is inherited by
# both the server and replay client. Its value is never serialized.
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_vllm_trace_experiment.py" \
  --prepared-workload "${EVAL_WORKLOAD}" \
  --output-dir "${RUN_DIR}" \
  --server-url "${SERVER_URL}" \
  --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --speedup "${SPEEDUP}" \
  --scheduler-metadata-mode "${SCHEDULER_METADATA_MODE}" \
  --scheduler-calibration-workload "${CALIBRATION_WORKLOAD}" \
  "${TOOL_ARGS[@]}" \
  --tool-wait-mode sleep \
  --max-model-len "${MAX_MODEL_LEN}" \
  --temperature 0 \
  --top-p 1 \
  --presence-penalty 0 \
  --request-timeout-s 600 \
  --max-request-attempts "${MAX_REQUEST_ATTEMPTS}" \
  --metrics-scrape-interval-s 0.5 \
  --max-active-traces "${MAX_ACTIVE}" \
  --vllm-log-file "${SERVER_LOG}" \
  --swap-events-file "${PASTE_SWAP_EVENTS_FILE:-${REPRO_ROOT}/logs/vllm_8000_swap_events.jsonl}"

cp -- "${SERVER_LOG}" "${RUN_DIR}/server.log"
echo "Completed ${POLICY_LABEL} cell: ${RUN_DIR}"
