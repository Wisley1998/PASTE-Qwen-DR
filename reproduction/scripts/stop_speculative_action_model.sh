#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: stop_speculative_action_model.sh

Stop only the managed Qwen3-8B endpoint on SPEC_ACTION_PORT (default 8200).
EOF
  exit 0
fi
export VLLM_HOST="${SPEC_ACTION_HOST:-127.0.0.1}"
export VLLM_PROBE_HOST="${SPEC_ACTION_PROBE_HOST:-127.0.0.1}"
export VLLM_PORT="${SPEC_ACTION_PORT:-8200}"
export PASTE_ENV_PREFIX="${SPEC_ACTION_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
export HF_HOME="${SPEC_ACTION_HF_HOME:-${HOME}/hf_cache}"
export MODEL_ID="${SPEC_ACTION_MODEL_ID:-Qwen/Qwen3-8B}"
export MODEL_REVISION="${SPEC_ACTION_MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
export MODEL_SNAPSHOT="${SPEC_ACTION_MODEL_SNAPSHOT:-${HF_HOME}/models--Qwen--Qwen3-8B/snapshots/${MODEL_REVISION}}"
export VLLM_STATE_DIR="${SPEC_ACTION_STATE_DIR:-${REPO_ROOT}/reproduction/run/speculative_action}"
export VLLM_LOG_DIR="${SPEC_ACTION_LOG_DIR:-${REPO_ROOT}/reproduction/logs/speculative_action}"
export VLLM_SHUTDOWN_TIMEOUT="${SPEC_ACTION_SHUTDOWN_TIMEOUT:-60}"
exec "${SCRIPT_DIR}/stop_vllm.sh" "$@"
