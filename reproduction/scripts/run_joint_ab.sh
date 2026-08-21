#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPRO_ROOT="${REPO_ROOT}/reproduction"
RUN_ROOT="${PASTE_RUN_ROOT:-${REPRO_ROOT}/artifacts/runs}"
BASELINE_NAME="${PASTE_BASELINE_RUN_NAME:-fcfs}"
JOINT_NAME="${PASTE_JOINT_RUN_NAME:-joint}"

validate_run_name() {
  local name="$1"
  local label="$2"
  if [[ ! "${name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: ${label} run name contains unsupported characters" >&2
    exit 2
  fi
}

validate_run_name "${BASELINE_NAME}" baseline
validate_run_name "${JOINT_NAME}" joint
if [[ "${BASELINE_NAME}" == "${JOINT_NAME}" ]]; then
  echo "error: baseline and joint run names must be different" >&2
  exit 2
fi

if curl --fail --silent --max-time 2 "${PASTE_SERVER_URL:-http://127.0.0.1:8000}/health" >/dev/null 2>&1; then
  echo "error: a server is already running; stop it before the isolated A/B" >&2
  exit 1
fi

if [[ "${PASTE_SKIP_PREPARE:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/prepare_joint_workloads.sh"
fi

export VLLM_USE_V1=1
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
export VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS="${VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS:-6}"
export VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING="${VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING:-4}"
export VLLM_SCHED_HBM_MIN_RUNNING_REQS="${VLLM_SCHED_HBM_MIN_RUNNING_REQS:-4}"
export VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP="${VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP:-4}"
export VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS="${VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS:-12000}"
export VLLM_SCHED_HBM_MAX_LONG_RUNNING="${VLLM_SCHED_HBM_MAX_LONG_RUNNING:-2}"
export VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS="${VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS:-196608}"
export VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS="${VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS:-131072}"
export VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS="${VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS:-262144}"

SERVER_IS_MANAGED=0
cleanup() {
  if (( SERVER_IS_MANAGED == 1 )); then
    "${SCRIPT_DIR}/stop_vllm.sh" || true
  fi
}
trap cleanup EXIT

run_cell() {
  local policy="$1"
  local label="$2"
  export VLLM_SCHED_POLICY="${policy}"
  export VLLM_REQUIRE_NEW=1
  export VLLM_LOG_DIR="${RUN_ROOT}/${label}/server"
  export PASTE_VLLM_LOG_FILE="${VLLM_LOG_DIR}/vllm_${VLLM_PORT:-8000}.log"
  "${SCRIPT_DIR}/start_vllm.sh"
  SERVER_IS_MANAGED=1
  "${SCRIPT_DIR}/smoke_vllm.py" --max-tokens 64
  "${SCRIPT_DIR}/run_joint_cell.sh" \
    "$([[ "${policy}" == "fcfs" ]] && echo fcfs || echo joint)" "${label}"
  "${SCRIPT_DIR}/stop_vllm.sh"
  SERVER_IS_MANAGED=0
}

run_cell fcfs "${BASELINE_NAME}"
run_cell online_joint_pacer_v2 "${JOINT_NAME}"

"${SCRIPT_DIR}/summarize_joint.py" \
  --baseline "${RUN_ROOT}/${BASELINE_NAME}" \
  --joint "${RUN_ROOT}/${JOINT_NAME}" \
  --output "${RUN_ROOT}/summary.json"

echo "Matched A/B summary: ${RUN_ROOT}/summary.json"
