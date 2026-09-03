#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: start_speculative_action_model.sh

Start the isolated, pinned Qwen3-8B Speculative Actions endpoint. Configure it
with SPEC_ACTION_* variables; see configs/speculative_action_qwen3_8b.env.example.
EOF
  exit 0
fi

export PASTE_ENV_PREFIX="${SPEC_ACTION_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
export HF_HOME="${SPEC_ACTION_HF_HOME:-${HOME}/hf_cache}"
export MODEL_ID="${SPEC_ACTION_MODEL_ID:-Qwen/Qwen3-8B}"
export MODEL_REVISION="${SPEC_ACTION_MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
export MODEL_SNAPSHOT="${SPEC_ACTION_MODEL_SNAPSHOT:-${HF_HOME}/models--Qwen--Qwen3-8B/snapshots/${MODEL_REVISION}}"
export CUDA_VISIBLE_DEVICES="${SPEC_ACTION_GPU:-0}"
export VLLM_HOST="${SPEC_ACTION_HOST:-127.0.0.1}"
export VLLM_PROBE_HOST="${SPEC_ACTION_PROBE_HOST:-127.0.0.1}"
export VLLM_PORT="${SPEC_ACTION_PORT:-8200}"
export VLLM_TP_SIZE="1"
export VLLM_DTYPE="bfloat16"
export VLLM_MAX_MODEL_LEN="${SPEC_ACTION_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEMORY_UTILIZATION="${SPEC_ACTION_GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${SPEC_ACTION_MAX_NUM_BATCHED_TOKENS:-16384}"
export VLLM_MAX_NUM_SEQS="${SPEC_ACTION_MAX_NUM_SEQS:-16}"
export VLLM_ENABLE_PREFIX_CACHING="1"
export VLLM_USE_V1="1"
export VLLM_SCHED_POLICY="fcfs"
export VLLM_READY_TIMEOUT="${SPEC_ACTION_READY_TIMEOUT:-1800}"
export VLLM_STATE_DIR="${SPEC_ACTION_STATE_DIR:-${REPO_ROOT}/reproduction/run/speculative_action}"
export VLLM_LOG_DIR="${SPEC_ACTION_LOG_DIR:-${REPO_ROOT}/reproduction/logs/speculative_action}"
if [[ -n "${SPEC_ACTION_API_KEY:-}" ]]; then
  export VLLM_API_KEY="${SPEC_ACTION_API_KEY}"
fi

exec "${SCRIPT_DIR}/start_vllm.sh" "$@"
