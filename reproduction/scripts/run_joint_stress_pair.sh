#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: run_joint_stress_pair.sh RUN_TAG [options]

Run one fresh-server A/D stress pair, a validated A-only load probe, or a
validated D-only screening cell.

Options:
  --cells A|D|A,D|D,A                     cell selection/order (default: A,D)
  --gpus ID,ID,ID,ID                      four distinct GPU IDs (default: 0,1,2,3)
  --port PORT                             localhost vLLM port (default: 8000)
  --config PATH                           shell config file (default: stress120)
  --check-only                            validate without creating output or using GPUs
  -h, --help                              show this help

A means fcfs_none; D means joint_learned.  Stress240/stress300 A-probe profiles
require --cells A so load selection cannot inspect D.  D-only screens require a
completed accepted A probe via PASTE_ACCEPTED_A_PROBE.  Stress240 physical093
also requires a completed native reference B; stress300 physical093 instead
uses the frozen retry-clean A-r3 probe.  The stress300 native B causal screen
requires the frozen completed physical093 C via PASTE_REFERENCE_C_RUN.  A
partial or completed RUN_TAG is never overwritten: choose a new tag after an
interrupted run.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi
case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  -*)
    usage >&2
    exit 2
    ;;
esac

RUN_TAG="$1"
shift
CELL_ORDER="A,D"
GPU_IDS="0,1,2,3"
PORT="8000"
CONFIG_PATH=""
CHECK_ONLY=0
A_ONLY_PROBE=0
D_ONLY_SCREEN=0
REQUIRE_ACCEPTED_A_PROBE=0
REQUIRE_REFERENCE_B_RUN=0
REQUIRE_PHYSICAL_KV_TELEMETRY=0
REQUIRE_PHYSICAL_KV_TELEMETRY_V2=0
REQUIRE_HTTP_KEEPALIVE60=0
REQUIRE_REFERENCE_C_RUN=0
REQUIRE_NATIVE_ZERO_WRITE_V2=0

while (( $# > 0 )); do
  case "$1" in
    --cells|--cell-order)
      (( $# >= 2 )) || die "--cells requires a value"
      CELL_ORDER="$2"
      shift 2
      ;;
    --gpus)
      (( $# >= 2 )) || die "--gpus requires a value"
      GPU_IDS="$2"
      shift 2
      ;;
    --port)
      (( $# >= 2 )) || die "--port requires a value"
      PORT="$2"
      shift 2
      ;;
    --config)
      (( $# >= 2 )) || die "--config requires a value"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unsupported argument: $1"
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPRO_ROOT="${REPO_ROOT}/reproduction"
if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${REPRO_ROOT}/configs/joint_stress.env.example"
elif [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="$(realpath -m -- "${PWD}/${CONFIG_PATH}")"
fi

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "RUN_TAG must use only letters, digits, dot, underscore, and dash"
[[ -f "${CONFIG_PATH}" ]] || die "config file is missing: ${CONFIG_PATH}"

# shellcheck disable=SC1090
source "${CONFIG_PATH}"
STRESS_PROFILE="${PASTE_STRESS_PROFILE:-stress120_target64}"
export PASTE_STRESS_PROFILE="${STRESS_PROFILE}"
CONFIG_SHA256_BEFORE="$(sha256sum -- "${CONFIG_PATH}" | awk '{print $1}')"
export PASTE_FROZEN_CONFIG_SHA256="${CONFIG_SHA256_BEFORE}"

require_exact() {
  local name="$1"
  local expected="$2"
  local actual="${!name-}"
  if [[ "${actual}" != "${expected}" ]]; then
    die "${name} must be ${expected@Q} for ${STRESS_PROFILE} (got ${actual@Q})"
  fi
  export "${name}"
}

require_nonempty() {
  local name="$1"
  local actual="${!name-}"
  [[ -n "${actual}" ]] || die "${name} must be explicitly set for ${STRESS_PROFILE}"
  export "${name}"
}

require_unset() {
  local name="$1"
  local actual
  if declare -p "${name}" >/dev/null 2>&1; then
    actual="${!name}"
    die "${name} must be absent for ${STRESS_PROFILE} (got ${actual@Q})"
  fi
}

validate_common_profile() {
  require_exact PASTE_TRACE_SPEEDUP "10"
  require_exact PASTE_MAX_REQUEST_ATTEMPTS "2"
  require_exact PASTE_TOOL_PREDICTION_TOP_K "5"
  require_exact PASTE_SCHEDULER_METADATA_MODE "online"
  require_exact MODEL_ID "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
  require_exact MODEL_REVISION "4b0ac5767427a55d08a254f0367e2934976598e0"
  require_exact VLLM_HOST "127.0.0.1"
  require_exact VLLM_PROBE_HOST "127.0.0.1"
  require_exact VLLM_TP_SIZE "4"
  require_exact VLLM_DTYPE "bfloat16"
  require_exact VLLM_MAX_MODEL_LEN "16384"
  require_exact VLLM_MAX_NUM_BATCHED_TOKENS "8192"
  require_exact VLLM_USE_V1 "1"
  require_exact VLLM_SCHED_PRED_OUT_ENABLE "1"
}

validate_complete_scheduler_profile() {
  local name
  local -a required_names=(
    VLLM_SCHED_PRED_OUT_EMA_ALPHA
    VLLM_SCHED_DEFAULT_PRED_OUT
    VLLM_SCHED_AVG_CALL_SERVICE_S
    VLLM_SCHED_PREFILL_TOKENS_PER_S_V2
    VLLM_SCHED_DECODE_TOKENS_PER_S_V2
    VLLM_SCHED_TIME_AGING_ALPHA
    VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS
    VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING
    VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S
    VLLM_SCHED_JOINT_V2_FINAL_LANE
    VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE
    VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING
    VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING
    VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING
    VLLM_SCHED_JOINT_V2_TAIL_BETA
    VLLM_SCHED_JOINT_V2_TOOL_BETA
    VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S
    VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT
    VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA
    VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS
    VLLM_SCHED_JOINT_V2_FINAL_BONUS_S
    VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S
    VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S
    VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S
    VLLM_SCHED_HBM_MIN_RUNNING_REQS
    VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP
    VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS
    VLLM_SCHED_HBM_MAX_LONG_RUNNING
    VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS
    VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS
    VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS
    VLLM_SCHED_HBM_LOW_PRESSURE
    VLLM_SCHED_HBM_HIGH_PRESSURE
    VLLM_SCHED_HBM_BUDGET_INCREASE
    VLLM_SCHED_HBM_BUDGET_DECREASE
    VLLM_SCHED_HBM_CONTROL_INTERVAL_S
    VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO
  )
  for name in "${required_names[@]}"; do
    require_nonempty "${name}"
  done
}

validate_stress120_target64_profile() {
  validate_common_profile
  require_exact VLLM_MAX_NUM_SEQS "64"
  require_exact PASTE_MAX_ACTIVE_TRACES "120"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.83"
  require_exact VLLM_SCHED_PRED_OUT_EMA_ALPHA "0.5"
  require_exact VLLM_SCHED_DEFAULT_PRED_OUT "357"
  require_exact VLLM_SCHED_AVG_CALL_SERVICE_S "3.3"
  require_exact VLLM_SCHED_PREFILL_TOKENS_PER_S_V2 "38112"
  require_exact VLLM_SCHED_DECODE_TOKENS_PER_S_V2 "113.7"
  require_exact VLLM_SCHED_TIME_AGING_ALPHA "1.2"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "64"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "6"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "64"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "64"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "64"
  require_exact VLLM_SCHED_JOINT_V2_TAIL_BETA "0.8"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_BETA "0.9"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S "80"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT "0.35"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA "1.4"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS "16000"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_BONUS_S "10"
  require_exact VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S "5"
  require_exact VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S "0"
  require_exact VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S "240"
  require_exact VLLM_SCHED_HBM_MIN_RUNNING_REQS "24"
  require_exact VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP "16"
  require_exact VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS "16000"
  require_exact VLLM_SCHED_HBM_MAX_LONG_RUNNING "16"
  require_exact VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS "786432"
  require_exact VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS "524288"
  require_exact VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS "1048576"
  require_exact VLLM_SCHED_HBM_LOW_PRESSURE "0.82"
  require_exact VLLM_SCHED_HBM_HIGH_PRESSURE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_INCREASE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_DECREASE "0.97"
  require_exact VLLM_SCHED_HBM_CONTROL_INTERVAL_S "5"
  require_exact VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO "0.96"
}

validate_stress180_target64_u86_profile() {
  validate_common_profile
  require_exact VLLM_MAX_NUM_SEQS "64"
  require_exact PASTE_MAX_ACTIVE_TRACES "180"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  validate_complete_scheduler_profile
}

validate_stress180_native256_g256_u86_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "180"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "256"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "256"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "40"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "1"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  validate_complete_scheduler_profile
}

validate_stress180_native256_g256_u86_exact_rescue120_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "180"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "256"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_SCHED_PRED_OUT_EMA_ALPHA "0.5"
  require_exact VLLM_SCHED_DEFAULT_PRED_OUT "357"
  require_exact VLLM_SCHED_AVG_CALL_SERVICE_S "3.3"
  require_exact VLLM_SCHED_PREFILL_TOKENS_PER_S_V2 "38112"
  require_exact VLLM_SCHED_DECODE_TOKENS_PER_S_V2 "113.7"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_TIME_AGING_ALPHA "0.2"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "256"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "120"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_TAIL_BETA "0.25"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_BETA "0.9"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S "80"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT "0.35"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA "1.4"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS "16000"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_BONUS_S "28"
  require_exact VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S "18"
  require_exact VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S "8"
  require_exact VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S "240"
  require_exact VLLM_SCHED_HBM_MIN_RUNNING_REQS "24"
  require_exact VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP "16"
  require_exact VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS "16000"
  require_exact VLLM_SCHED_HBM_MAX_LONG_RUNNING "16"
  require_exact VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS "786432"
  require_exact VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS "524288"
  require_exact VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS "1048576"
  require_exact VLLM_SCHED_HBM_LOW_PRESSURE "0.82"
  require_exact VLLM_SCHED_HBM_HIGH_PRESSURE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_INCREASE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_DECREASE "0.97"
  require_exact VLLM_SCHED_HBM_CONTROL_INTERVAL_S "5"
  require_exact VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO "0.96"
  validate_complete_scheduler_profile
}

validate_stress180_native256_g256_u86_soft4_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "180"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "256"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "256"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "40"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S "4.0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  validate_complete_scheduler_profile
}

validate_stress240_native256_g256_u86_a_probe_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "240"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "256"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "256"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "40"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "1"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  require_exact PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION "0.50"
  require_exact PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION "0.20"
  require_exact PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST "0.25"
  validate_complete_scheduler_profile
}

validate_stress300_native320_g256_u86_a_probe_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "300"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "320"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_SCHED_PRED_OUT_EMA_ALPHA "0.5"
  require_exact VLLM_SCHED_DEFAULT_PRED_OUT "357"
  require_exact VLLM_SCHED_AVG_CALL_SERVICE_S "3.3"
  require_exact VLLM_SCHED_PREFILL_TOKENS_PER_S_V2 "38112"
  require_exact VLLM_SCHED_DECODE_TOKENS_PER_S_V2 "113.7"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_TIME_AGING_ALPHA "0.2"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "320"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "40"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "320"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "320"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_TAIL_BETA "0.25"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_BETA "0.9"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S "80"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT "0.35"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA "1.4"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS "16000"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_BONUS_S "28"
  require_exact VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S "18"
  require_exact VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S "8"
  require_exact VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S "240"
  require_exact VLLM_SCHED_HBM_MIN_RUNNING_REQS "24"
  require_exact VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP "16"
  require_exact VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS "16000"
  require_exact VLLM_SCHED_HBM_MAX_LONG_RUNNING "16"
  require_exact VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS "786432"
  require_exact VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS "524288"
  require_exact VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS "1048576"
  require_exact VLLM_SCHED_HBM_LOW_PRESSURE "0.82"
  require_exact VLLM_SCHED_HBM_HIGH_PRESSURE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_INCREASE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_DECREASE "0.97"
  require_exact VLLM_SCHED_HBM_CONTROL_INTERVAL_S "5"
  require_exact VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO "0.96"
  require_exact PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION "0.50"
  require_exact PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION "0.20"
  require_exact PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST "0.25"
  validate_complete_scheduler_profile
}

validate_stress300_native320_g256_u86_keepalive60_a_probe_profile() {
  validate_stress300_native320_g256_u86_a_probe_profile
  require_exact VLLM_HTTP_TIMEOUT_KEEP_ALIVE "60"
}

validate_stress300_native320_g256_u86_physical093_exact_rescue120_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "300"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "320"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_HTTP_TIMEOUT_KEEP_ALIVE "60"
  require_exact VLLM_SCHED_PRED_OUT_EMA_ALPHA "0.5"
  require_exact VLLM_SCHED_DEFAULT_PRED_OUT "357"
  require_exact VLLM_SCHED_AVG_CALL_SERVICE_S "3.3"
  require_exact VLLM_SCHED_PREFILL_TOKENS_PER_S_V2 "38112"
  require_exact VLLM_SCHED_DECODE_TOKENS_PER_S_V2 "113.7"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_TIME_AGING_ALPHA "0.2"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "320"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "120"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "320"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "320"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "0"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION "0.93"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S "120"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_TAIL_BETA "0.25"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_BETA "0.9"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S "80"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT "0.35"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA "1.4"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS "16000"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_BONUS_S "28"
  require_exact VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S "18"
  require_exact VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S "8"
  require_exact VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S "240"
  require_exact VLLM_SCHED_HBM_MIN_RUNNING_REQS "24"
  require_exact VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP "16"
  require_exact VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS "16000"
  require_exact VLLM_SCHED_HBM_MAX_LONG_RUNNING "16"
  require_exact VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS "786432"
  require_exact VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS "524288"
  require_exact VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS "1048576"
  require_exact VLLM_SCHED_HBM_LOW_PRESSURE "0.82"
  require_exact VLLM_SCHED_HBM_HIGH_PRESSURE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_INCREASE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_DECREASE "0.97"
  require_exact VLLM_SCHED_HBM_CONTROL_INTERVAL_S "5"
  require_exact VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO "0.96"
  require_exact PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION "0.50"
  require_exact PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION "0.20"
  require_exact PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST "0.25"
  require_exact PASTE_ACCEPTED_A_PROBE_SHA256 \
    "c2a5b098a178e7e9d899ea88995f0f591bb24ec70380c2d5242bc734d2c247bd"
  require_exact PASTE_ACCEPTED_A_CONFIG_SHA256 \
    "c1c043836601203c4f49284daf8b7e925bab450747482e486eed83897dda2d06"
  require_nonempty PASTE_ACCEPTED_A_PROBE
  validate_complete_scheduler_profile
}

validate_stress300_native320_g256_u86_native_exact_rescue120_b_screen_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "300"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "320"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_HTTP_TIMEOUT_KEEP_ALIVE "60"
  require_exact VLLM_SCHED_PRED_OUT_EMA_ALPHA "0.5"
  require_exact VLLM_SCHED_DEFAULT_PRED_OUT "357"
  require_exact VLLM_SCHED_AVG_CALL_SERVICE_S "3.3"
  require_exact VLLM_SCHED_PREFILL_TOKENS_PER_S_V2 "38112"
  require_exact VLLM_SCHED_DECODE_TOKENS_PER_S_V2 "113.7"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_TIME_AGING_ALPHA "0.2"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "320"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "120"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "320"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "320"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "1"
  require_unset VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION
  require_unset VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION
  require_unset VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S
  require_unset VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_TAIL_BETA "0.25"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_BETA "0.9"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S "80"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT "0.35"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA "1.4"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS "16000"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_BONUS_S "28"
  require_exact VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S "18"
  require_exact VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S "8"
  require_exact VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S "240"
  require_exact VLLM_SCHED_HBM_MIN_RUNNING_REQS "24"
  require_exact VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP "16"
  require_exact VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS "16000"
  require_exact VLLM_SCHED_HBM_MAX_LONG_RUNNING "16"
  require_exact VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS "786432"
  require_exact VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS "524288"
  require_exact VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS "1048576"
  require_exact VLLM_SCHED_HBM_LOW_PRESSURE "0.82"
  require_exact VLLM_SCHED_HBM_HIGH_PRESSURE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_INCREASE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_DECREASE "0.97"
  require_exact VLLM_SCHED_HBM_CONTROL_INTERVAL_S "5"
  require_exact VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO "0.96"
  require_exact PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION "0.50"
  require_exact PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION "0.20"
  require_exact PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST "0.25"
  require_exact PASTE_REFERENCE_C_CONFIG_SHA256 \
    "1ee7dfe9f5831223fb4ff14c1e86154827d32d7835d11b2749c8e07863321d43"
  require_exact PASTE_REFERENCE_C_PHYSICAL_V2_SHA256 \
    "b292c04f0bdaf53ec9bea4ff290a8517f19cdc277d2eca908eb055c24dbf252e"
  require_exact PASTE_REFERENCE_C_AC_SCREENING_SHA256 \
    "906df1cd484311c3acbf701720d49cc3c0f516f5b48bf78e9e51ec1b5fcc7771"
  require_exact PASTE_REFERENCE_C_SUMMARY_SHA256 \
    "15f42aa950ce16e0a40a114ce0e70fee52f32f7d70402c5c3a7a554d70d06742"
  require_exact PASTE_REFERENCE_C_RAW_LOG_SHA256 \
    "c2eb67a5f6bb737991e485487fe08124a630a4c2f1d57db6e19ac37c34d9a17e"
  require_nonempty PASTE_REFERENCE_C_RUN
  validate_complete_scheduler_profile
}

validate_stress240_native256_g256_u86_exact_rescue120_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "240"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "256"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_SCHED_PRED_OUT_EMA_ALPHA "0.5"
  require_exact VLLM_SCHED_DEFAULT_PRED_OUT "357"
  require_exact VLLM_SCHED_AVG_CALL_SERVICE_S "3.3"
  require_exact VLLM_SCHED_PREFILL_TOKENS_PER_S_V2 "38112"
  require_exact VLLM_SCHED_DECODE_TOKENS_PER_S_V2 "113.7"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_TIME_AGING_ALPHA "0.2"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "256"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "120"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_TAIL_BETA "0.25"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_BETA "0.9"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S "80"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT "0.35"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA "1.4"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS "16000"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_BONUS_S "28"
  require_exact VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S "18"
  require_exact VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S "8"
  require_exact VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S "240"
  require_exact VLLM_SCHED_HBM_MIN_RUNNING_REQS "24"
  require_exact VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP "16"
  require_exact VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS "16000"
  require_exact VLLM_SCHED_HBM_MAX_LONG_RUNNING "16"
  require_exact VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS "786432"
  require_exact VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS "524288"
  require_exact VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS "1048576"
  require_exact VLLM_SCHED_HBM_LOW_PRESSURE "0.82"
  require_exact VLLM_SCHED_HBM_HIGH_PRESSURE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_INCREASE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_DECREASE "0.97"
  require_exact VLLM_SCHED_HBM_CONTROL_INTERVAL_S "5"
  require_exact VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO "0.96"
  require_exact PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION "0.50"
  require_exact PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION "0.20"
  require_exact PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST "0.25"
  require_nonempty PASTE_ACCEPTED_A_PROBE
  validate_complete_scheduler_profile
}

validate_stress240_native256_g256_u86_physical093_exact_rescue120_profile() {
  validate_common_profile
  require_exact PASTE_MAX_ACTIVE_TRACES "240"
  require_exact VLLM_GPU_MEMORY_UTILIZATION "0.86"
  require_exact VLLM_MAX_NUM_SEQS "256"
  require_exact VLLM_CUDA_GRAPH_SIZES "256"
  require_exact VLLM_SCHED_PRED_OUT_EMA_ALPHA "0.5"
  require_exact VLLM_SCHED_DEFAULT_PRED_OUT "357"
  require_exact VLLM_SCHED_AVG_CALL_SERVICE_S "3.3"
  require_exact VLLM_SCHED_PREFILL_TOKENS_PER_S_V2 "38112"
  require_exact VLLM_SCHED_DECODE_TOKENS_PER_S_V2 "113.7"
  require_exact VLLM_SCHED_OAS_V3_CONTEXT_TOKENS_PER_S "6000"
  require_exact VLLM_SCHED_TIME_AGING_ALPHA "0.2"
  require_exact VLLM_SCHED_JOINT_V2_FOREGROUND_MAX_SESSIONS "256"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_GATE_MAX_WAIT_S "120"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_LANE "1"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_COARSE_LANES "0"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_CALL_SOFT_WEIGHT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_DEADLINE_MIN_RUNNING "48"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_TARGET_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_DECODE_MAX_RUNNING "256"
  require_exact VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION "0"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_ADMISSION "1"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_TARGET_UTILIZATION "0.93"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_RESCUE_WAIT_S "120"
  require_exact VLLM_SCHED_JOINT_V2_PHYSICAL_KV_LOG_INTERVAL_S "1"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY "0"
  require_exact VLLM_SCHED_JOINT_V2_RUNNING_PRIORITY_MAX_WAIT_S "0"
  require_exact VLLM_SCHED_JOINT_V2_TAIL_BETA "0.25"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_BETA "0.9"
  require_exact VLLM_SCHED_JOINT_V2_TOOL_WAIT_CAP_S "80"
  require_exact VLLM_SCHED_JOINT_V2_REMAINING_TOOL_WEIGHT "0.35"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_ALPHA "1.4"
  require_exact VLLM_SCHED_JOINT_V2_CONTEXT_REF_TOKENS "16000"
  require_exact VLLM_SCHED_JOINT_V2_FINAL_BONUS_S "28"
  require_exact VLLM_SCHED_JOINT_V2_PROGRESS_BONUS_S "18"
  require_exact VLLM_SCHED_JOINT_V2_NEW_SESSION_PENALTY_S "8"
  require_exact VLLM_SCHED_JOINT_V2_OVER_BUDGET_PENALTY_S "240"
  require_exact VLLM_SCHED_HBM_MIN_RUNNING_REQS "24"
  require_exact VLLM_SCHED_HBM_MAX_ADMIT_PER_STEP "16"
  require_exact VLLM_SCHED_HBM_LONG_CONTEXT_TOKENS "16000"
  require_exact VLLM_SCHED_HBM_MAX_LONG_RUNNING "16"
  require_exact VLLM_SCHED_HBM_TARGET_CONTEXT_TOKENS "786432"
  require_exact VLLM_SCHED_HBM_MIN_CONTEXT_TOKENS "524288"
  require_exact VLLM_SCHED_HBM_MAX_CONTEXT_TOKENS "1048576"
  require_exact VLLM_SCHED_HBM_LOW_PRESSURE "0.82"
  require_exact VLLM_SCHED_HBM_HIGH_PRESSURE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_INCREASE "1.02"
  require_exact VLLM_SCHED_HBM_BUDGET_DECREASE "0.97"
  require_exact VLLM_SCHED_HBM_CONTROL_INTERVAL_S "5"
  require_exact VLLM_SCHED_HBM_VIRTUAL_FILL_RATIO "0.96"
  require_exact PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION "0.50"
  require_exact PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION "0.20"
  require_exact PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST "0.25"
  require_exact PASTE_REFERENCE_B_CONFIG_SHA256 \
    "d0f9f486f9cdd14aa3fd970086682b31220b4666b03af91cc77a75951d1065b0"
  require_exact PASTE_REFERENCE_B_SCREENING_SHA256 \
    "234f117467c2cb3fa1e6068551c27364961c89aef48405c2fecb15639b5f5509"
  require_nonempty PASTE_ACCEPTED_A_PROBE
  require_nonempty PASTE_REFERENCE_B_RUN
  validate_complete_scheduler_profile
}

require_repo_relative_path() {
  local name="$1"
  local raw="${!name-}"
  local resolved
  [[ -n "${raw}" ]] || die "${name} is required"
  [[ "${raw}" != /* ]] || die "${name} must be repository-relative"
  resolved="$(realpath -m -- "${REPO_ROOT}/${raw}")"
  [[ "${resolved}" == "${REPO_ROOT}/"* ]] \
    || die "${name} must stay inside the repository"
  printf '%s\n' "${resolved}"
}

case "${CELL_ORDER}" in
  A|fcfs_none)
    CELLS="fcfs_none"
    A_ONLY_PROBE=1
    ;;
  D|joint_learned)
    CELLS="joint_learned"
    D_ONLY_SCREEN=1
    ;;
  A,D|fcfs_none,joint_learned)
    CELLS="fcfs_none,joint_learned"
    ;;
  D,A|joint_learned,fcfs_none)
    CELLS="joint_learned,fcfs_none"
    ;;
  *)
    die "--cells must contain exactly A and D in either order, or select a profile-specific A or D single cell"
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
(( ${#GPU_ARRAY[@]} == 4 )) || die "--gpus must list exactly four GPU IDs"
declare -A GPU_SEEN=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^(0|[1-9][0-9]*)$ ]] || die "invalid GPU ID: ${gpu}"
  [[ -z "${GPU_SEEN[${gpu}]+x}" ]] || die "GPU IDs must be distinct"
  GPU_SEEN["${gpu}"]=1
done
[[ "${PORT}" =~ ^[1-9][0-9]*$ ]] || die "--port must be an integer in 1..65535"
(( PORT <= 65535 )) || die "--port must be an integer in 1..65535"

case "${STRESS_PROFILE}" in
  stress120_target64)
    EXPECTED_LOAD_INSTANCE_COUNT=120
    PAIR_LABEL="stress120 target64"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=0
    validate_stress120_target64_profile
    ;;
  stress180_target64_u86)
    EXPECTED_LOAD_INSTANCE_COUNT=180
    PAIR_LABEL="stress180 target64/u86"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    validate_stress180_target64_u86_profile
    ;;
  stress180_native256_g256_u86)
    EXPECTED_LOAD_INSTANCE_COUNT=180
    PAIR_LABEL="stress180 native256/graph256/u86"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    validate_stress180_native256_g256_u86_profile
    ;;
  stress180_native256_g256_u86_exact_rescue120)
    EXPECTED_LOAD_INSTANCE_COUNT=180
    PAIR_LABEL="stress180 native256/graph256/u86 exact-stage rescue120 D-only screen"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    validate_stress180_native256_g256_u86_exact_rescue120_profile
    (( D_ONLY_SCREEN == 1 )) \
      || die "stress180_native256_g256_u86_exact_rescue120 requires --cells D; it is a D-only screening profile"
    ;;
  stress180_native256_g256_u86_soft4)
    EXPECTED_LOAD_INSTANCE_COUNT=180
    PAIR_LABEL="stress180 native256/graph256/u86 soft4"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    validate_stress180_native256_g256_u86_soft4_profile
    ;;
  stress240_native256_g256_u86_a_probe)
    EXPECTED_LOAD_INSTANCE_COUNT=240
    PAIR_LABEL="stress240 native256/graph256/u86 A-only load probe"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    validate_stress240_native256_g256_u86_a_probe_profile
    (( A_ONLY_PROBE == 1 )) \
      || die "stress240_native256_g256_u86_a_probe requires --cells A; D is intentionally unavailable during load selection"
    ;;
  stress300_native320_g256_u86_a_probe)
    EXPECTED_LOAD_INSTANCE_COUNT=300
    PAIR_LABEL="stress300 native320/graph256/u86 A-only load probe"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    validate_stress300_native320_g256_u86_a_probe_profile
    (( A_ONLY_PROBE == 1 )) \
      || die "stress300_native320_g256_u86_a_probe requires --cells A; D is intentionally unavailable during load selection"
    ;;
  stress300_native320_g256_u86_keepalive60_a_probe)
    EXPECTED_LOAD_INSTANCE_COUNT=300
    PAIR_LABEL="stress300 native320/graph256/u86 keepalive60 A-only retry-clean load probe"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    REQUIRE_HTTP_KEEPALIVE60=1
    validate_stress300_native320_g256_u86_keepalive60_a_probe_profile
    (( A_ONLY_PROBE == 1 )) \
      || die "stress300_native320_g256_u86_keepalive60_a_probe requires --cells A; D is intentionally unavailable during load selection"
    ;;
  stress300_native320_g256_u86_native_exact_rescue120_b_screen)
    EXPECTED_LOAD_INSTANCE_COUNT=300
    PAIR_LABEL="stress300 native320/graph256/u86 keepalive60 native reorder-only exact-rescue120 B-only causal screen"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    REQUIRE_REFERENCE_C_RUN=1
    REQUIRE_NATIVE_ZERO_WRITE_V2=1
    REQUIRE_HTTP_KEEPALIVE60=1
    validate_stress300_native320_g256_u86_native_exact_rescue120_b_screen_profile
    [[ "${CONFIG_SHA256_BEFORE}" == \
      "e024ab17e6b08c1c1cd3246e4b74b253b681af152138af762bc536f7b513908e" ]] \
      || die "stress300 native B-screen config SHA drifted"
    [[ "${GPU_IDS}" == "4,5,6,7" ]] \
      || die "stress300 native B-screen must match completed C GPUs 4,5,6,7"
    [[ "${PORT}" == "8100" ]] \
      || die "stress300 native B-screen must match completed C port 8100"
    (( D_ONLY_SCREEN == 1 )) \
      || die "stress300_native320_g256_u86_native_exact_rescue120_b_screen requires --cells D and the frozen completed C reference"
    ;;
  stress300_native320_g256_u86_physical093_exact_rescue120)
    EXPECTED_LOAD_INSTANCE_COUNT=300
    PAIR_LABEL="stress300 native320/graph256/u86 keepalive60 physical093 exact-rescue120 accepted-A C-only screen"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    REQUIRE_ACCEPTED_A_PROBE=1
    REQUIRE_PHYSICAL_KV_TELEMETRY_V2=1
    REQUIRE_HTTP_KEEPALIVE60=1
    validate_stress300_native320_g256_u86_physical093_exact_rescue120_profile
    [[ "${CONFIG_SHA256_BEFORE}" == \
      "1ee7dfe9f5831223fb4ff14c1e86154827d32d7835d11b2749c8e07863321d43" ]] \
      || die "stress300 physical093 config SHA drifted"
    [[ "${GPU_IDS}" == "4,5,6,7" ]] \
      || die "stress300 physical093 must match accepted A-r3 GPUs 4,5,6,7"
    [[ "${PORT}" == "8100" ]] \
      || die "stress300 physical093 must match accepted A-r3 port 8100"
    (( D_ONLY_SCREEN == 1 )) \
      || die "stress300_native320_g256_u86_physical093_exact_rescue120 requires --cells D and the frozen accepted A-r3 probe"
    ;;
  stress240_native256_g256_u86_exact_rescue120)
    EXPECTED_LOAD_INSTANCE_COUNT=240
    PAIR_LABEL="stress240 native256/graph256/u86 exact-stage rescue120 accepted-A D-only screen"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    REQUIRE_ACCEPTED_A_PROBE=1
    validate_stress240_native256_g256_u86_exact_rescue120_profile
    (( D_ONLY_SCREEN == 1 )) \
      || die "stress240_native256_g256_u86_exact_rescue120 requires --cells D and an accepted A probe"
    ;;
  stress240_native256_g256_u86_physical093_exact_rescue120)
    EXPECTED_LOAD_INSTANCE_COUNT=240
    PAIR_LABEL="stress240 native256/graph256/u86 physical093 exact-stage rescue120 accepted-A/reference-B C-only screen"
    REQUIRE_IDENTICAL_SCHEDULER_CONFIG=1
    REQUIRE_ACCEPTED_A_PROBE=1
    REQUIRE_REFERENCE_B_RUN=1
    REQUIRE_PHYSICAL_KV_TELEMETRY=1
    validate_stress240_native256_g256_u86_physical093_exact_rescue120_profile
    (( D_ONLY_SCREEN == 1 )) \
      || die "stress240_native256_g256_u86_physical093_exact_rescue120 requires --cells D, an accepted A probe, and a completed reference B"
    ;;
  *)
    die "unsupported PASTE_STRESS_PROFILE: ${STRESS_PROFILE}"
    ;;
esac

if [[ "${STRESS_PROFILE}" != "stress240_native256_g256_u86_a_probe" \
  && "${STRESS_PROFILE}" != "stress300_native320_g256_u86_a_probe" \
  && "${STRESS_PROFILE}" != "stress300_native320_g256_u86_keepalive60_a_probe" ]] \
  && (( A_ONLY_PROBE == 1 )); then
  die "--cells A is reserved for a validated A-only load-selection profile"
fi
if [[ "${STRESS_PROFILE}" != "stress180_native256_g256_u86_exact_rescue120" \
  && "${STRESS_PROFILE}" != "stress240_native256_g256_u86_exact_rescue120" \
  && "${STRESS_PROFILE}" != "stress240_native256_g256_u86_physical093_exact_rescue120" \
  && "${STRESS_PROFILE}" != "stress300_native320_g256_u86_physical093_exact_rescue120" \
  && "${STRESS_PROFILE}" != "stress300_native320_g256_u86_native_exact_rescue120_b_screen" ]] \
  && (( D_ONLY_SCREEN == 1 )); then
  die "--cells D is reserved for an exact-rescue120 D-only screening profile"
fi

MANIFEST="$(require_repo_relative_path PASTE_FIXED_WORKLOAD_MANIFEST)"
RUN_BASE="$(require_repo_relative_path PASTE_STRESS_RUN_BASE)"
[[ -f "${MANIFEST}" ]] || {
  echo "error: ${STRESS_PROFILE} manifest is missing: ${MANIFEST}" >&2
  echo "Build the matching fixed stress workload bundle first." >&2
  exit 1
}
ENV_PREFIX="${PASTE_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
[[ -x "${PYTHON_BIN}" ]] || die "reproduction environment is missing: ${PYTHON_BIN}"
MANIFEST_LOAD_INSTANCE_COUNT="$(
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = manifest.get("stress_definition", {}).get("load_instance_count")
if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise SystemExit("manifest has no positive stress load_instance_count")
print(value)
PY
)" || die "could not read stress load_instance_count from ${MANIFEST}"
[[ "${MANIFEST_LOAD_INSTANCE_COUNT}" == "${EXPECTED_LOAD_INSTANCE_COUNT}" ]] \
  || die "${STRESS_PROFILE} requires ${EXPECTED_LOAD_INSTANCE_COUNT} load instances (manifest has ${MANIFEST_LOAD_INSTANCE_COUNT})"
if [[ "${STRESS_PROFILE}" == "stress240_native256_g256_u86_a_probe" \
  || "${STRESS_PROFILE}" == "stress240_native256_g256_u86_exact_rescue120" \
  || "${STRESS_PROFILE}" == "stress240_native256_g256_u86_physical093_exact_rescue120" ]]; then
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY' || \
    die "stress240 manifest must contain exactly four balanced instances of each of 60 heldout sources"
import json
from pathlib import Path
import sys

definition = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get(
    "stress_definition", {}
)
expected = {
    "source_role": "heldout",
    "unique_source_session_count": 60,
    "load_instance_count": 240,
    "instances_per_source": 4,
    "minimum_instances_per_source": 4,
    "maximum_instances_per_source": 4,
    "sources_with_one_extra_instance": 0,
    "source_instances_are_balanced": True,
    "calibration_excluded": True,
    "mapper_retrained": False,
    "duplicates_are_not_independent": True,
}
if any(definition.get(name) != value for name, value in expected.items()):
    raise SystemExit(1)
PY
fi
if [[ "${STRESS_PROFILE}" == "stress300_native320_g256_u86_a_probe" \
  || "${STRESS_PROFILE}" == "stress300_native320_g256_u86_keepalive60_a_probe" \
  || "${STRESS_PROFILE}" == "stress300_native320_g256_u86_physical093_exact_rescue120" \
  || "${STRESS_PROFILE}" == "stress300_native320_g256_u86_native_exact_rescue120_b_screen" ]]; then
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY' || \
    die "stress300 manifest must contain exactly five balanced instances of each of 60 heldout sources"
import json
from pathlib import Path
import sys

definition = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get(
    "stress_definition", {}
)
expected = {
    "source_role": "heldout",
    "unique_source_session_count": 60,
    "load_instance_count": 300,
    "instances_per_source": 5,
    "minimum_instances_per_source": 5,
    "maximum_instances_per_source": 5,
    "sources_with_one_extra_instance": 0,
    "source_instances_are_balanced": True,
    "calibration_excluded": True,
    "mapper_retrained": False,
    "duplicates_are_not_independent": True,
}
if any(definition.get(name) != value for name, value in expected.items()):
    raise SystemExit(1)
PY
fi

ACCEPTED_A_PROBE=""
ACCEPTED_A_PROBE_SHA256_BEFORE=""
ACCEPTED_A_VALIDATION_BEFORE=""
ACCEPTED_A_CELL=""
ACCEPTED_A_FROZEN_CONFIG=""
ACCEPTED_A_FROZEN_SIDECAR=""
ACCEPTED_A_FROZEN_CONFIG_SHA256_BEFORE=""
if (( REQUIRE_ACCEPTED_A_PROBE == 1 )); then
  ACCEPTED_A_PROBE="$(require_repo_relative_path PASTE_ACCEPTED_A_PROBE)"
  [[ -f "${ACCEPTED_A_PROBE}" ]] \
    || die "accepted A probe is missing or incomplete: ${ACCEPTED_A_PROBE}"
  if [[ "${STRESS_PROFILE}" == \
    "stress300_native320_g256_u86_physical093_exact_rescue120" ]]; then
    ACCEPTED_A_EXPECTED_PROFILE="stress300_native320_g256_u86_keepalive60_a_probe"
    ACCEPTED_A_EXPECTED_LOAD="300"
    ACCEPTED_A_EXPECTED_MAX_NUM_SEQS="320"
  else
    ACCEPTED_A_EXPECTED_PROFILE="stress240_native256_g256_u86_a_probe"
    ACCEPTED_A_EXPECTED_LOAD="240"
    ACCEPTED_A_EXPECTED_MAX_NUM_SEQS="256"
  fi
  ACCEPTED_A_ARGS=(
    "${ACCEPTED_A_PROBE}"
    --repository-root "${REPO_ROOT}"
    --expected-profile "${ACCEPTED_A_EXPECTED_PROFILE}"
    --expected-load "${ACCEPTED_A_EXPECTED_LOAD}"
    --expected-max-num-seqs "${ACCEPTED_A_EXPECTED_MAX_NUM_SEQS}"
    --min-waiting-fraction
      "${PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION}"
    --min-queue-fraction
      "${PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION}"
    --max-preemptions-per-request
      "${PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST}"
  )
  for name in \
    MODEL_ID \
    MODEL_REVISION \
    VLLM_TP_SIZE \
    VLLM_DTYPE \
    VLLM_MAX_MODEL_LEN \
    VLLM_GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_MAX_NUM_SEQS \
    VLLM_CUDA_GRAPH_SIZES \
    VLLM_USE_V1; do
    ACCEPTED_A_ARGS+=(--expect-engine-shape "${name}=${!name}")
  done
  if ! ACCEPTED_A_VALIDATION_BEFORE="$(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_accepted_a_probe.py" \
      "${ACCEPTED_A_ARGS[@]}" 2>&1
  )"; then
    echo "${ACCEPTED_A_VALIDATION_BEFORE}" >&2
    die "accepted A probe validation failed: ${ACCEPTED_A_PROBE}"
  fi
  ACCEPTED_A_PROBE_SHA256_BEFORE="$(
    sha256sum -- "${ACCEPTED_A_PROBE}" | awk '{print $1}'
  )"
  if [[ "${STRESS_PROFILE}" == \
    "stress300_native320_g256_u86_physical093_exact_rescue120" ]]; then
    [[ "${ACCEPTED_A_PROBE_SHA256_BEFORE}" == \
      "${PASTE_ACCEPTED_A_PROBE_SHA256}" ]] \
      || die "accepted stress300 A probe SHA mismatch"
    ACCEPTED_A_CELL="$(
      "${PYTHON_BIN}" -c \
        'import json, sys; print(json.load(sys.stdin)["cell_dir"])' \
        <<< "${ACCEPTED_A_VALIDATION_BEFORE}"
    )" || die "could not resolve accepted stress300 A cell"
    [[ -d "${ACCEPTED_A_CELL}" ]] \
      || die "accepted stress300 A cell is missing: ${ACCEPTED_A_CELL}"
    ACCEPTED_A_FROZEN_CONFIG="$(dirname -- "${ACCEPTED_A_CELL}")/frozen_config.env"
    ACCEPTED_A_FROZEN_SIDECAR="$(dirname -- "${ACCEPTED_A_CELL}")/frozen_config.sha256"
    [[ -f "${ACCEPTED_A_FROZEN_CONFIG}" ]] \
      || die "accepted stress300 A frozen config is missing"
    [[ -f "${ACCEPTED_A_FROZEN_SIDECAR}" ]] \
      || die "accepted stress300 A frozen-config checksum sidecar is missing"
    ACCEPTED_A_FROZEN_CONFIG_SHA256_BEFORE="$(
      sha256sum -- "${ACCEPTED_A_FROZEN_CONFIG}" | awk '{print $1}'
    )"
    [[ "${ACCEPTED_A_FROZEN_CONFIG_SHA256_BEFORE}" == \
      "${PASTE_ACCEPTED_A_CONFIG_SHA256}" ]] \
      || die "accepted stress300 A frozen config SHA mismatch"
    [[ "$(< "${ACCEPTED_A_FROZEN_SIDECAR}")" == \
      "${PASTE_ACCEPTED_A_CONFIG_SHA256}  frozen_config.env" ]] \
      || die "accepted stress300 A frozen-config checksum sidecar mismatch"
    grep -Fqx -- 'export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="60"' \
      "${ACCEPTED_A_FROZEN_CONFIG}" \
      || die "accepted stress300 A frozen config does not prove keepalive60"
  fi
fi

REFERENCE_B_RUN=""
REFERENCE_B_SCREENING=""
REFERENCE_B_SCREENING_SHA256_BEFORE=""
REFERENCE_B_SUMMARY_SHA256_BEFORE=""
REFERENCE_B_VALIDATION_BEFORE=""
if (( REQUIRE_REFERENCE_B_RUN == 1 )); then
  REFERENCE_B_RUN="$(require_repo_relative_path PASTE_REFERENCE_B_RUN)"
  [[ -d "${REFERENCE_B_RUN}" ]] \
    || die "reference B cell is missing or incomplete: ${REFERENCE_B_RUN}"
  REFERENCE_B_SCREENING="${REFERENCE_B_RUN}/strict_a_vs_b_screening.json"
  [[ -f "${REFERENCE_B_SCREENING}" ]] \
    || die "reference B strict A/B evidence is missing: ${REFERENCE_B_SCREENING}"
  REFERENCE_B_SCREENING_SHA256_BEFORE="$(
    sha256sum -- "${REFERENCE_B_SCREENING}" | awk '{print $1}'
  )"
  [[ "${REFERENCE_B_SCREENING_SHA256_BEFORE}" == "${PASTE_REFERENCE_B_SCREENING_SHA256}" ]] \
    || die "reference B strict A/B evidence SHA mismatch"
  REFERENCE_B_ARGS=(
    "${REFERENCE_B_RUN}"
    --reference-b
    --manifest "${MANIFEST}"
    --expected-load 240
    --expected-requests 2076
    --expected-config-sha256 "${PASTE_REFERENCE_B_CONFIG_SHA256}"
  )
  for name in \
    MODEL_ID \
    MODEL_REVISION \
    VLLM_HOST \
    VLLM_PROBE_HOST \
    VLLM_TP_SIZE \
    VLLM_DTYPE \
    VLLM_MAX_MODEL_LEN \
    VLLM_GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_MAX_NUM_SEQS \
    VLLM_CUDA_GRAPH_SIZES \
    VLLM_USE_V1; do
    REFERENCE_B_ARGS+=(--expect-engine-shape "${name}=${!name}")
  done
  REFERENCE_B_ARGS+=(
    --expect-engine-shape "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
    --expect-engine-shape "VLLM_PORT=${PORT}"
  )
  if ! REFERENCE_B_VALIDATION_BEFORE="$(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_physical_kv_admission.py" \
      "${REFERENCE_B_ARGS[@]}" 2>&1
  )"; then
    echo "${REFERENCE_B_VALIDATION_BEFORE}" >&2
    die "reference B validation failed: ${REFERENCE_B_RUN}"
  fi
  REFERENCE_B_SUMMARY_SHA256_BEFORE="$(
    sha256sum -- "${REFERENCE_B_RUN}/summary.json" | awk '{print $1}'
  )"
fi

REFERENCE_C_RUN=""
REFERENCE_C_ROOT=""
REFERENCE_C_PHYSICAL_V2=""
REFERENCE_C_AC_SCREENING=""
REFERENCE_C_SUMMARY=""
REFERENCE_C_RAW_LOG=""
REFERENCE_C_VALIDATION_BEFORE=""
REFERENCE_C_PHYSICAL_V2_SHA256_BEFORE=""
REFERENCE_C_AC_SCREENING_SHA256_BEFORE=""
REFERENCE_C_SUMMARY_SHA256_BEFORE=""
REFERENCE_C_RAW_LOG_SHA256_BEFORE=""
if (( REQUIRE_REFERENCE_C_RUN == 1 )); then
  REFERENCE_C_RUN="$(require_repo_relative_path PASTE_REFERENCE_C_RUN)"
  EXPECTED_REFERENCE_C_RUN="${REPO_ROOT}/reproduction/artifacts/stress300_u86_native320_g256_physical093_exact_rescue120/stress300_c_physical093_r1/stress300_c_physical093_r1_joint_learned"
  [[ "${REFERENCE_C_RUN}" == "${EXPECTED_REFERENCE_C_RUN}" ]] \
    || die "PASTE_REFERENCE_C_RUN must point to the frozen completed stress300 C cell"
  [[ -d "${REFERENCE_C_RUN}" ]] \
    || die "reference C cell is missing or incomplete: ${REFERENCE_C_RUN}"
  REFERENCE_C_ROOT="$(dirname -- "${REFERENCE_C_RUN}")"
  REFERENCE_C_PHYSICAL_V2="${REFERENCE_C_ROOT}/physical_kv_validation_v2.json"
  REFERENCE_C_AC_SCREENING="${REFERENCE_C_ROOT}/strict_a_vs_c_physical_v2.json"
  REFERENCE_C_SUMMARY="${REFERENCE_C_RUN}/summary.json"
  REFERENCE_C_RAW_LOG="${REFERENCE_C_RUN}/server/vllm_8100.log"
  for path in \
    "${REFERENCE_C_PHYSICAL_V2}" \
    "${REFERENCE_C_AC_SCREENING}" \
    "${REFERENCE_C_SUMMARY}" \
    "${REFERENCE_C_RAW_LOG}"; do
    [[ -f "${path}" ]] || die "reference C evidence is missing: ${path}"
  done
  REFERENCE_C_PHYSICAL_V2_SHA256_BEFORE="$(
    sha256sum -- "${REFERENCE_C_PHYSICAL_V2}" | awk '{print $1}'
  )"
  REFERENCE_C_AC_SCREENING_SHA256_BEFORE="$(
    sha256sum -- "${REFERENCE_C_AC_SCREENING}" | awk '{print $1}'
  )"
  REFERENCE_C_SUMMARY_SHA256_BEFORE="$(
    sha256sum -- "${REFERENCE_C_SUMMARY}" | awk '{print $1}'
  )"
  REFERENCE_C_RAW_LOG_SHA256_BEFORE="$(
    sha256sum -- "${REFERENCE_C_RAW_LOG}" | awk '{print $1}'
  )"
  [[ "${REFERENCE_C_PHYSICAL_V2_SHA256_BEFORE}" == \
    "${PASTE_REFERENCE_C_PHYSICAL_V2_SHA256}" ]] \
    || die "reference C parser-v2 validation SHA mismatch"
  [[ "${REFERENCE_C_AC_SCREENING_SHA256_BEFORE}" == \
    "${PASTE_REFERENCE_C_AC_SCREENING_SHA256}" ]] \
    || die "reference C strict A/C evidence SHA mismatch"
  [[ "${REFERENCE_C_SUMMARY_SHA256_BEFORE}" == \
    "${PASTE_REFERENCE_C_SUMMARY_SHA256}" ]] \
    || die "reference C summary SHA mismatch"
  [[ "${REFERENCE_C_RAW_LOG_SHA256_BEFORE}" == \
    "${PASTE_REFERENCE_C_RAW_LOG_SHA256}" ]] \
    || die "reference C canonical raw-log SHA mismatch"
  REFERENCE_C_ARGS=(
    "${REFERENCE_C_RUN}"
    --expected-profile stress300_native320_g256_u86_physical093_exact_rescue120
    --expected-load 300
    --expected-requests 2595
    --expected-config-sha256 "${PASTE_REFERENCE_C_CONFIG_SHA256}"
    --expected-num-gpu-blocks 44178
    --expected-block-size 16
    --expected-target-utilization 0.93
    --expected-keepalive-s 60
    --expected-preemptions 0
  )
  for name in \
    MODEL_ID \
    MODEL_REVISION \
    VLLM_HOST \
    VLLM_PROBE_HOST \
    VLLM_TP_SIZE \
    VLLM_DTYPE \
    VLLM_MAX_MODEL_LEN \
    VLLM_GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_MAX_NUM_SEQS \
    VLLM_CUDA_GRAPH_SIZES \
    VLLM_USE_V1; do
    REFERENCE_C_ARGS+=(--expect-engine-shape "${name}=${!name}")
  done
  REFERENCE_C_ARGS+=(
    --expect-engine-shape "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
    --expect-engine-shape "VLLM_PORT=${PORT}"
  )
  if ! REFERENCE_C_VALIDATION_BEFORE="$(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_physical_kv_admission_v2.py" \
      "${REFERENCE_C_ARGS[@]}" 2>&1
  )"; then
    echo "${REFERENCE_C_VALIDATION_BEFORE}" >&2
    die "reference C parser-v2 revalidation failed: ${REFERENCE_C_RUN}"
  fi
  [[ "${REFERENCE_C_VALIDATION_BEFORE}" == "$(< "${REFERENCE_C_PHYSICAL_V2}")" ]] \
    || die "reference C saved parser-v2 evidence differs from fresh revalidation"
fi

RUN_ROOT="${RUN_BASE}/${RUN_TAG}"
LOCK_DIR="${RUN_ROOT}.lock"
[[ ! -e "${RUN_ROOT}" && ! -e "${LOCK_DIR}" ]] \
  || die "output or lock already exists for RUN_TAG ${RUN_TAG}: ${RUN_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export VLLM_PORT="${PORT}"
export PASTE_SERVER_URL="http://127.0.0.1:${PORT}"
export PASTE_FIXED_WORKLOAD_MANIFEST="${MANIFEST}"
export PASTE_RUN_ROOT="${RUN_ROOT}"
export PASTE_RUN_PREFIX="${RUN_TAG}"
export PASTE_CELLS="${CELLS}"
export VLLM_STATE_DIR="${RUN_ROOT}/state"
export PASTE_SWAP_EVENTS_FILE="${RUN_ROOT}/swap_events.jsonl"
export PASTE_VALIDATE_ONLY="0"

if (( A_ONLY_PROBE == 1 || D_ONLY_SCREEN == 1 )); then
  echo "${PAIR_LABEL}"
else
  echo "${PAIR_LABEL} pair"
fi
echo "  run tag: ${RUN_TAG}"
echo "  profile: ${STRESS_PROFILE}"
echo "  config:  ${CONFIG_PATH} (${CONFIG_SHA256_BEFORE})"
echo "  cells:   ${CELLS}"
echo "  GPUs:    ${CUDA_VISIBLE_DEVICES} (TP=${VLLM_TP_SIZE})"
echo "  server:  ${PASTE_SERVER_URL}"
echo "  output:  ${RUN_ROOT}"
if (( REQUIRE_HTTP_KEEPALIVE60 == 1 )); then
  echo "  HTTP keep-alive: ${VLLM_HTTP_TIMEOUT_KEEP_ALIVE}s (frozen server setting)"
fi
if (( REQUIRE_ACCEPTED_A_PROBE == 1 )); then
  echo "  A probe: ${ACCEPTED_A_PROBE} (accepted)"
  echo "  A SHA:   ${ACCEPTED_A_PROBE_SHA256_BEFORE}"
fi
if (( REQUIRE_REFERENCE_B_RUN == 1 )); then
  echo "  B cell:  ${REFERENCE_B_RUN} (validated native reference)"
  echo "  B SHA:   ${REFERENCE_B_SUMMARY_SHA256_BEFORE}"
  echo "  B proof: ${REFERENCE_B_SCREENING_SHA256_BEFORE}"
fi
if (( REQUIRE_REFERENCE_C_RUN == 1 )); then
  echo "  C cell:  ${REFERENCE_C_RUN} (frozen physical093 reference)"
  echo "  C SHA:   ${REFERENCE_C_SUMMARY_SHA256_BEFORE}"
  echo "  C proof: ${REFERENCE_C_PHYSICAL_V2_SHA256_BEFORE}"
fi

if (( CHECK_ONLY == 1 )); then
  PASTE_VALIDATE_ONLY=1 "${SCRIPT_DIR}/run_four_cell.sh" stress
  echo "Check-only validation completed; no output was created."
  exit 0
fi

mkdir -p -- "${RUN_BASE}"
if ! mkdir -- "${LOCK_DIR}"; then
  die "another process reserved RUN_TAG ${RUN_TAG}"
fi
cleanup_lock() {
  rmdir -- "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_lock EXIT

mkdir -- "${RUN_ROOT}"
cp -- "${CONFIG_PATH}" "${RUN_ROOT}/frozen_config.env"
printf '%s  %s\n' \
  "${CONFIG_SHA256_BEFORE}" \
  "frozen_config.env" > "${RUN_ROOT}/frozen_config.sha256"
if (( REQUIRE_ACCEPTED_A_PROBE == 1 )); then
  cp -- "${ACCEPTED_A_PROBE}" "${RUN_ROOT}/accepted_a_probe.json"
  printf '%s  %s\n' \
    "${ACCEPTED_A_PROBE_SHA256_BEFORE}" \
    "accepted_a_probe.json" > "${RUN_ROOT}/accepted_a_probe.sha256"
  printf '%s\n' \
    "${ACCEPTED_A_VALIDATION_BEFORE}" \
    > "${RUN_ROOT}/accepted_a_probe_validation.json"
fi
if (( REQUIRE_REFERENCE_B_RUN == 1 )); then
  cp -- "${REFERENCE_B_SCREENING}" \
    "${RUN_ROOT}/reference_b_strict_screening.json"
  cp -- "${REFERENCE_B_RUN}/summary.json" \
    "${RUN_ROOT}/reference_b_summary.json"
  printf '%s  %s\n%s  %s\n' \
    "${REFERENCE_B_SCREENING_SHA256_BEFORE}" \
    "reference_b_strict_screening.json" \
    "${REFERENCE_B_SUMMARY_SHA256_BEFORE}" \
    "reference_b_summary.json" \
    > "${RUN_ROOT}/reference_b_evidence.sha256"
  printf '%s\n' \
    "${REFERENCE_B_VALIDATION_BEFORE}" \
    > "${RUN_ROOT}/reference_b_validation.json"
fi
if (( REQUIRE_REFERENCE_C_RUN == 1 )); then
  cp -- "${REFERENCE_C_PHYSICAL_V2}" \
    "${RUN_ROOT}/reference_c_physical_kv_validation_v2.json"
  cp -- "${REFERENCE_C_AC_SCREENING}" \
    "${RUN_ROOT}/reference_c_strict_a_vs_c_physical_v2.json"
  cp -- "${REFERENCE_C_SUMMARY}" \
    "${RUN_ROOT}/reference_c_summary.json"
  printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
    "${REFERENCE_C_PHYSICAL_V2_SHA256_BEFORE}" \
    "reference_c_physical_kv_validation_v2.json" \
    "${REFERENCE_C_AC_SCREENING_SHA256_BEFORE}" \
    "reference_c_strict_a_vs_c_physical_v2.json" \
    "${REFERENCE_C_SUMMARY_SHA256_BEFORE}" \
    "reference_c_summary.json" \
    "${REFERENCE_C_RAW_LOG_SHA256_BEFORE}" \
    "${REFERENCE_C_RAW_LOG#${REPO_ROOT}/}" \
    > "${RUN_ROOT}/reference_c_evidence.sha256"
fi

"${SCRIPT_DIR}/run_four_cell.sh" stress
if (( A_ONLY_PROBE == 1 )); then
  PROBE_CELL="${RUN_ROOT}/${RUN_TAG}_fcfs_none"
  PROBE_SUMMARY="${RUN_ROOT}/natural_queue_probe.json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_natural_queue_probe.py" \
    "${PROBE_CELL}" > "${PROBE_SUMMARY}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_natural_queue_probe.py" \
    "${PROBE_CELL}" \
    --require-natural-queue \
    --require-exactly-once \
    --require-no-kv-swap \
    --min-waiting-below-cap-sample-fraction \
      "${PASTE_NATURAL_QUEUE_MIN_WAITING_SAMPLE_FRACTION}" \
    --min-queue-time-fraction \
      "${PASTE_NATURAL_QUEUE_MIN_QUEUE_TIME_FRACTION}" \
    --max-preemptions-per-request \
      "${PASTE_NATURAL_QUEUE_MAX_PREEMPTIONS_PER_REQUEST}" >/dev/null
elif (( D_ONLY_SCREEN == 0 )); then
  SUMMARY_ARGS=(
    --manifest "${MANIFEST}"
    --role stress
    --pair
      "${RUN_ROOT}/${RUN_TAG}_fcfs_none"
      "${RUN_ROOT}/${RUN_TAG}_joint_learned"
    --output "${RUN_ROOT}/paired_summary.json"
  )
  if (( REQUIRE_IDENTICAL_SCHEDULER_CONFIG == 1 )); then
    SUMMARY_ARGS+=(--require-identical-scheduler-config)
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_paired_ad.py" \
    "${SUMMARY_ARGS[@]}" >/dev/null
fi

if (( REQUIRE_NATIVE_ZERO_WRITE_V2 == 1 )); then
  NATIVE_B_CELL="${RUN_ROOT}/${RUN_TAG}_joint_learned"
  NATIVE_ZERO_WRITE_ARGS=(
    "${NATIVE_B_CELL}"
    --expected-profile "${STRESS_PROFILE}"
    --expected-load 300
    --expected-requests 2595
    --expected-config-sha256 "${CONFIG_SHA256_BEFORE}"
    --expected-keepalive-s 60
    --output "${RUN_ROOT}/native_admission_zero_write_v2.json"
  )
  for name in \
    MODEL_ID \
    MODEL_REVISION \
    VLLM_HOST \
    VLLM_PROBE_HOST \
    VLLM_TP_SIZE \
    VLLM_DTYPE \
    VLLM_MAX_MODEL_LEN \
    VLLM_GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_MAX_NUM_SEQS \
    VLLM_CUDA_GRAPH_SIZES \
    VLLM_USE_V1; do
    NATIVE_ZERO_WRITE_ARGS+=(--expect-engine-shape "${name}=${!name}")
  done
  NATIVE_ZERO_WRITE_ARGS+=(
    --expect-engine-shape "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
    --expect-engine-shape "VLLM_PORT=${PORT}"
  )
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_native_admission_zero_write_v2.py" \
    "${NATIVE_ZERO_WRITE_ARGS[@]}" >/dev/null; then
    die "native reorder-only zero-capacity-write validation failed: ${NATIVE_B_CELL}"
  fi
fi

if (( REQUIRE_PHYSICAL_KV_TELEMETRY == 1 )); then
  PHYSICAL_KV_CELL="${RUN_ROOT}/${RUN_TAG}_joint_learned"
  if ! PHYSICAL_KV_VALIDATION="$(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_physical_kv_admission.py" \
      "${PHYSICAL_KV_CELL}" \
      --expected-profile "${STRESS_PROFILE}" \
      --expected-load 240 \
      --expected-config-sha256 "${CONFIG_SHA256_BEFORE}" 2>&1
  )"; then
    echo "${PHYSICAL_KV_VALIDATION}" >&2
    die "physical-KV telemetry validation failed: ${PHYSICAL_KV_CELL}"
  fi
  printf '%s\n' "${PHYSICAL_KV_VALIDATION}" \
    > "${RUN_ROOT}/physical_kv_validation.json"
fi

if (( REQUIRE_PHYSICAL_KV_TELEMETRY_V2 == 1 )); then
  PHYSICAL_KV_CELL="${RUN_ROOT}/${RUN_TAG}_joint_learned"
  PHYSICAL_KV_V2_ARGS=(
    "${PHYSICAL_KV_CELL}"
    --expected-profile "${STRESS_PROFILE}"
    --expected-load 300
    --expected-requests 2595
    --expected-config-sha256 "${CONFIG_SHA256_BEFORE}"
    --expected-num-gpu-blocks 44178
    --expected-block-size 16
    --expected-target-utilization 0.93
    --expected-keepalive-s 60
    --expected-preemptions 0
    --output "${RUN_ROOT}/physical_kv_validation_v2.json"
  )
  for name in \
    MODEL_ID \
    MODEL_REVISION \
    VLLM_HOST \
    VLLM_PROBE_HOST \
    VLLM_TP_SIZE \
    VLLM_DTYPE \
    VLLM_MAX_MODEL_LEN \
    VLLM_GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_MAX_NUM_SEQS \
    VLLM_CUDA_GRAPH_SIZES \
    VLLM_USE_V1; do
    PHYSICAL_KV_V2_ARGS+=(--expect-engine-shape "${name}=${!name}")
  done
  PHYSICAL_KV_V2_ARGS+=(
    --expect-engine-shape "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
    --expect-engine-shape "VLLM_PORT=${PORT}"
  )
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_physical_kv_admission_v2.py" \
    "${PHYSICAL_KV_V2_ARGS[@]}" >/dev/null; then
    die "parser-v2 physical-KV validation failed: ${PHYSICAL_KV_CELL}"
  fi
fi

if (( REQUIRE_ACCEPTED_A_PROBE == 1 )); then
  if ! ACCEPTED_A_VALIDATION_AFTER="$(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_accepted_a_probe.py" \
      "${ACCEPTED_A_ARGS[@]}" 2>&1
  )"; then
    echo "${ACCEPTED_A_VALIDATION_AFTER}" >&2
    die "accepted A probe failed post-run revalidation: ${ACCEPTED_A_PROBE}"
  fi
  [[ "${ACCEPTED_A_VALIDATION_AFTER}" == "${ACCEPTED_A_VALIDATION_BEFORE}" ]] \
    || die "accepted A probe or source cell changed during the D screen"
  ACCEPTED_A_PROBE_SHA256_AFTER="$(
    sha256sum -- "${ACCEPTED_A_PROBE}" | awk '{print $1}'
  )"
  [[ "${ACCEPTED_A_PROBE_SHA256_AFTER}" == "${ACCEPTED_A_PROBE_SHA256_BEFORE}" ]] \
    || die "accepted A probe changed during the D screen"
  if (( REQUIRE_PHYSICAL_KV_TELEMETRY_V2 == 1 )); then
    ACCEPTED_A_FROZEN_CONFIG_SHA256_AFTER="$(
      sha256sum -- "${ACCEPTED_A_FROZEN_CONFIG}" | awk '{print $1}'
    )"
    [[ "${ACCEPTED_A_FROZEN_CONFIG_SHA256_AFTER}" == \
      "${ACCEPTED_A_FROZEN_CONFIG_SHA256_BEFORE}" ]] \
      || die "accepted stress300 A frozen config changed during the C screen"
  fi
fi

if (( REQUIRE_PHYSICAL_KV_TELEMETRY_V2 == 1 )); then
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_strict_screening_ac_physical_v2.py" \
    --manifest "${MANIFEST}" \
    --a-run "${ACCEPTED_A_CELL}" \
    --c-run "${PHYSICAL_KV_CELL}" \
    --accepted-a-probe "${ACCEPTED_A_PROBE}" \
    --expected-a-probe-sha256 "${PASTE_ACCEPTED_A_PROBE_SHA256}" \
    --expected-a-profile stress300_native320_g256_u86_keepalive60_a_probe \
    --expected-c-profile \
      stress300_native320_g256_u86_physical093_exact_rescue120 \
    --expected-a-config-sha256 "${PASTE_ACCEPTED_A_CONFIG_SHA256}" \
    --expected-c-config-sha256 "${CONFIG_SHA256_BEFORE}" \
    --expected-load 300 \
    --expected-requests 2595 \
    --expected-num-gpu-blocks 44178 \
    --expected-block-size 16 \
    --expected-max-num-seqs 320 \
    --expected-keepalive-s 60 \
    --output "${RUN_ROOT}/strict_a_vs_c_physical_v2.json" >/dev/null; then
    die "strict stress300 A/C physical-KV comparison failed"
  fi
fi

if (( REQUIRE_REFERENCE_B_RUN == 1 )); then
  if ! REFERENCE_B_VALIDATION_AFTER="$(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_physical_kv_admission.py" \
      "${REFERENCE_B_ARGS[@]}" 2>&1
  )"; then
    echo "${REFERENCE_B_VALIDATION_AFTER}" >&2
    die "reference B failed post-run revalidation: ${REFERENCE_B_RUN}"
  fi
  [[ "${REFERENCE_B_VALIDATION_AFTER}" == "${REFERENCE_B_VALIDATION_BEFORE}" ]] \
    || die "reference B evidence changed during the physical-KV screen"
  REFERENCE_B_SCREENING_SHA256_AFTER="$(
    sha256sum -- "${REFERENCE_B_SCREENING}" | awk '{print $1}'
  )"
  REFERENCE_B_SUMMARY_SHA256_AFTER="$(
    sha256sum -- "${REFERENCE_B_RUN}/summary.json" | awk '{print $1}'
  )"
  [[ "${REFERENCE_B_SCREENING_SHA256_AFTER}" == "${REFERENCE_B_SCREENING_SHA256_BEFORE}" ]] \
    || die "reference B strict A/B evidence changed during the physical-KV screen"
  [[ "${REFERENCE_B_SUMMARY_SHA256_AFTER}" == "${REFERENCE_B_SUMMARY_SHA256_BEFORE}" ]] \
    || die "reference B summary changed during the physical-KV screen"
fi

if (( REQUIRE_REFERENCE_C_RUN == 1 )); then
  if ! REFERENCE_C_VALIDATION_AFTER="$(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_physical_kv_admission_v2.py" \
      "${REFERENCE_C_ARGS[@]}" 2>&1
  )"; then
    echo "${REFERENCE_C_VALIDATION_AFTER}" >&2
    die "reference C failed post-run parser-v2 revalidation: ${REFERENCE_C_RUN}"
  fi
  [[ "${REFERENCE_C_VALIDATION_AFTER}" == "${REFERENCE_C_VALIDATION_BEFORE}" ]] \
    || die "reference C or its derived parser-v2 validation changed during the B screen"
  [[ "$(sha256sum -- "${REFERENCE_C_PHYSICAL_V2}" | awk '{print $1}')" == \
    "${REFERENCE_C_PHYSICAL_V2_SHA256_BEFORE}" ]] \
    || die "reference C parser-v2 evidence changed during the B screen"
  [[ "$(sha256sum -- "${REFERENCE_C_AC_SCREENING}" | awk '{print $1}')" == \
    "${REFERENCE_C_AC_SCREENING_SHA256_BEFORE}" ]] \
    || die "reference C strict A/C evidence changed during the B screen"
  [[ "$(sha256sum -- "${REFERENCE_C_SUMMARY}" | awk '{print $1}')" == \
    "${REFERENCE_C_SUMMARY_SHA256_BEFORE}" ]] \
    || die "reference C summary changed during the B screen"
  [[ "$(sha256sum -- "${REFERENCE_C_RAW_LOG}" | awk '{print $1}')" == \
    "${REFERENCE_C_RAW_LOG_SHA256_BEFORE}" ]] \
    || die "reference C canonical raw log changed during the B screen"
fi

if (( REQUIRE_NATIVE_ZERO_WRITE_V2 == 1 && REQUIRE_REFERENCE_C_RUN == 1 )); then
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_strict_screening_bc_physical_v2.py" \
    --manifest "${MANIFEST}" \
    --b-run "${NATIVE_B_CELL}" \
    --c-run "${REFERENCE_C_RUN}" \
    --output "${RUN_ROOT}/strict_b_vs_c_physical_v2.json" >/dev/null; then
    die "strict stress300 B/C incremental physical-admission comparison failed"
  fi
fi

CONFIG_SHA256_AFTER="$(sha256sum -- "${CONFIG_PATH}" | awk '{print $1}')"
[[ "${CONFIG_SHA256_AFTER}" == "${CONFIG_SHA256_BEFORE}" ]] \
  || die "configuration file changed during the run: ${CONFIG_PATH}"

trap - EXIT
cleanup_lock
if (( A_ONLY_PROBE == 1 )); then
  echo "Completed ${PAIR_LABEL} under ${RUN_ROOT}"
  echo "Accepted natural-queue probe: ${RUN_ROOT}/natural_queue_probe.json"
elif (( D_ONLY_SCREEN == 1 )); then
  echo "Completed ${PAIR_LABEL} under ${RUN_ROOT}"
  echo "Candidate cell: ${RUN_ROOT}/${RUN_TAG}_joint_learned"
  if (( REQUIRE_PHYSICAL_KV_TELEMETRY == 1 )); then
    echo "Validated physical-KV telemetry: ${RUN_ROOT}/physical_kv_validation.json"
  elif (( REQUIRE_PHYSICAL_KV_TELEMETRY_V2 == 1 )); then
    echo "Validated physical-KV telemetry: ${RUN_ROOT}/physical_kv_validation_v2.json"
    echo "Strict A/C screening: ${RUN_ROOT}/strict_a_vs_c_physical_v2.json"
  elif (( REQUIRE_NATIVE_ZERO_WRITE_V2 == 1 )); then
    echo "Validated native zero-write evidence: ${RUN_ROOT}/native_admission_zero_write_v2.json"
    echo "Strict B/C screening: ${RUN_ROOT}/strict_b_vs_c_physical_v2.json"
  fi
else
  echo "Completed ${PAIR_LABEL} pair under ${RUN_ROOT}"
  echo "Validated pair summary: ${RUN_ROOT}/paired_summary.json"
fi
