#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: start_vllm.sh

Start the pinned local Tongyi checkpoint with the verified vLLM settings.
Configuration is provided through environment variables; see
reproduction/configs/model.env.example.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    echo "error: start_vllm.sh does not accept positional arguments" >&2
    usage >&2
    exit 2
    ;;
esac
if (( $# > 1 )); then
  echo "error: start_vllm.sh does not accept positional arguments" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

ENV_PREFIX="${PASTE_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
ENV_PYTHON="${ENV_PREFIX}/bin/python"
HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
MODEL_ID="${MODEL_ID:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
MODEL_REVISION="${MODEL_REVISION:-4b0ac5767427a55d08a254f0367e2934976598e0}"
MODEL_CACHE_KEY="models--${MODEL_ID//\//--}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-${HF_HOME}/${MODEL_CACHE_KEY}/snapshots/${MODEL_REVISION}}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.83}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_CUDA_GRAPH_SIZES="${VLLM_CUDA_GRAPH_SIZES:-}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_SCHED_POLICY="${VLLM_SCHED_POLICY:-fcfs}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-3600}"
VLLM_START_CLEANUP_TIMEOUT="${VLLM_START_CLEANUP_TIMEOUT:-60}"
VLLM_REQUIRE_NEW="${VLLM_REQUIRE_NEW:-0}"
VLLM_PROBE_HOST="${VLLM_PROBE_HOST:-${VLLM_HOST}}"
VLLM_HOOK_DIR="${VLLM_HOOK_DIR:-${REPO_ROOT}/scripts/pythonhooks}"
VLLM_STATE_DIR="${VLLM_STATE_DIR:-${REPO_ROOT}/reproduction/run}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-${REPO_ROOT}/reproduction/logs}"
PID_FILE="${VLLM_STATE_DIR}/vllm_${VLLM_PORT}.pid"
POLICY_FILE="${VLLM_STATE_DIR}/vllm_${VLLM_PORT}.policy"
LOG_FILE="${VLLM_LOG_DIR}/vllm_${VLLM_PORT}.log"
LOCK_DIR="${PID_FILE}.lock"

if [[ "${VLLM_HOST}" == "0.0.0.0" ]]; then
  VLLM_PROBE_HOST="${VLLM_PROBE_HOST/0.0.0.0/127.0.0.1}"
fi

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${name} must be a positive integer (got ${value})" >&2
    exit 1
  fi
}

require_positive_integer VLLM_PORT "${VLLM_PORT}"
require_positive_integer VLLM_TP_SIZE "${VLLM_TP_SIZE}"
require_positive_integer VLLM_MAX_MODEL_LEN "${VLLM_MAX_MODEL_LEN}"
require_positive_integer VLLM_MAX_NUM_BATCHED_TOKENS "${VLLM_MAX_NUM_BATCHED_TOKENS}"
require_positive_integer VLLM_MAX_NUM_SEQS "${VLLM_MAX_NUM_SEQS}"
require_positive_integer VLLM_READY_TIMEOUT "${VLLM_READY_TIMEOUT}"
require_positive_integer VLLM_START_CLEANUP_TIMEOUT "${VLLM_START_CLEANUP_TIMEOUT}"
VLLM_CUDA_GRAPH_SIZE_ARGS=()
if [[ -n "${VLLM_CUDA_GRAPH_SIZES}" ]]; then
  if [[ ! "${VLLM_CUDA_GRAPH_SIZES}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "error: VLLM_CUDA_GRAPH_SIZES must be a positive integer or a comma-separated list of positive integers (got ${VLLM_CUDA_GRAPH_SIZES})" >&2
    exit 1
  fi
  IFS=',' read -r -a VLLM_CUDA_GRAPH_SIZE_ARGS <<< "${VLLM_CUDA_GRAPH_SIZES}"
fi
if (( VLLM_PORT > 65535 )); then
  echo "error: VLLM_PORT must be at most 65535" >&2
  exit 1
fi
if [[ ! "${VLLM_GPU_MEMORY_UTILIZATION}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  echo "error: VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]" >&2
  exit 1
fi
if [[ ! "${VLLM_SCHED_POLICY}" =~ ^[a-z0-9_]+$ ]]; then
  echo "error: VLLM_SCHED_POLICY contains unsupported characters" >&2
  exit 1
fi
if [[ "${VLLM_USE_V1}" != "0" && "${VLLM_USE_V1}" != "1" ]]; then
  echo "error: VLLM_USE_V1 must be 0 or 1" >&2
  exit 1
fi
if [[ "${VLLM_ENABLE_PREFIX_CACHING}" != "0" && "${VLLM_ENABLE_PREFIX_CACHING}" != "1" ]]; then
  echo "error: VLLM_ENABLE_PREFIX_CACHING must be 0 or 1" >&2
  exit 1
fi
if [[ "${VLLM_REQUIRE_NEW}" != "0" && "${VLLM_REQUIRE_NEW}" != "1" ]]; then
  echo "error: VLLM_REQUIRE_NEW must be 0 or 1" >&2
  exit 1
fi

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "error: environment Python not found: ${ENV_PYTHON}" >&2
  echo "Run reproduction/scripts/setup_env.sh first." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required for vLLM readiness checks" >&2
  exit 1
fi
if [[ ! -d "${MODEL_SNAPSHOT}" ]]; then
  echo "error: pinned local model snapshot not found: ${MODEL_SNAPSHOT}" >&2
  echo "Run reproduction/scripts/download_model.py first." >&2
  exit 1
fi
MODEL_SNAPSHOT="$(cd -- "${MODEL_SNAPSHOT}" && pwd -P)"
if [[ ! -f "${MODEL_SNAPSHOT}/config.json" ]]; then
  echo "error: model snapshot has no config.json: ${MODEL_SNAPSHOT}" >&2
  exit 1
fi
if [[ "${VLLM_SCHED_POLICY}" != "fcfs" ]]; then
  if [[ ! -f "${VLLM_HOOK_DIR}/sitecustomize.py" || ! -f "${VLLM_HOOK_DIR}/sched_policy_patch.py" ]]; then
    echo "error: scheduler hook files are missing from ${VLLM_HOOK_DIR}" >&2
    exit 1
  fi
fi

mkdir -p -- "${VLLM_STATE_DIR}" "${VLLM_LOG_DIR}"
if ! mkdir -- "${LOCK_DIR}" 2>/dev/null; then
  echo "error: another start/stop operation holds ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir -- "${LOCK_DIR}" 2>/dev/null || true' EXIT

cmdline_matches() {
  local pid="$1"
  local expected_python actual_python
  local -a argv=()
  local saw_port=0 saw_model=0 saw_served_model=0 saw_host=0
  local index

  [[ -r "/proc/${pid}/cmdline" && -e "/proc/${pid}/exe" ]] || return 1
  expected_python="$(readlink -f -- "${ENV_PYTHON}")"
  actual_python="$(readlink -f -- "/proc/${pid}/exe")"
  [[ "${actual_python}" == "${expected_python}" ]] || return 1
  mapfile -d '' -t argv < "/proc/${pid}/cmdline"
  [[ "${argv[1]:-}" == "-m" ]] || return 1
  [[ "${argv[2]:-}" == "vllm.entrypoints.openai.api_server" ]] || return 1
  for (( index = 0; index < ${#argv[@]}; index++ )); do
    case "${argv[index]}" in
      --port)
        [[ "${argv[index + 1]:-}" == "${VLLM_PORT}" ]] || return 1
        (( saw_port += 1 ))
        ;;
      --model)
        [[ "${argv[index + 1]:-}" == "${MODEL_SNAPSHOT}" ]] || return 1
        (( saw_model += 1 ))
        ;;
      --served-model-name)
        [[ "${argv[index + 1]:-}" == "${MODEL_ID}" ]] || return 1
        (( saw_served_model += 1 ))
        ;;
      --host)
        [[ "${argv[index + 1]:-}" == "${VLLM_HOST}" ]] || return 1
        (( saw_host += 1 ))
        ;;
    esac
  done
  (( saw_port == 1 && saw_model == 1 && saw_served_model == 1 && saw_host == 1 ))
}

endpoint_ready() {
  local -a auth_header=()
  if [[ -n "${VLLM_API_KEY:-}" ]]; then
    auth_header=(--header "Authorization: Bearer ${VLLM_API_KEY}")
  fi
  curl --fail --silent --show-error --max-time 3 \
    "${auth_header[@]}" \
    "http://${VLLM_PROBE_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1 &&
  curl --fail --silent --show-error --max-time 3 \
    "${auth_header[@]}" \
    "http://${VLLM_PROBE_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1
}

tcp_port_open() {
  (exec 3<>"/dev/tcp/${VLLM_PROBE_HOST}/${VLLM_PORT}") 2>/dev/null
}

process_is_running() {
  local pid="$1"
  local state
  kill -0 "${pid}" >/dev/null 2>&1 || return 1
  state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
  [[ "${state}" != "Z" ]]
}

cleanup_started_server() {
  local pid="$1"
  local deadline state_pid

  echo "Rolling back vLLM pid ${pid} because startup validation failed." >&2
  if process_is_running "${pid}"; then
    kill -TERM "${pid}" >/dev/null 2>&1 || true
    deadline=$((SECONDS + VLLM_START_CLEANUP_TIMEOUT))
    while (( SECONDS < deadline )) && process_is_running "${pid}"; do
      sleep 1
    done
    if process_is_running "${pid}"; then
      echo "vLLM pid ${pid} ignored SIGTERM; sending SIGKILL to this newly started process." >&2
      kill -KILL "${pid}" >/dev/null 2>&1 || true
      deadline=$((SECONDS + VLLM_START_CLEANUP_TIMEOUT))
      while (( SECONDS < deadline )) && process_is_running "${pid}"; do
        sleep 1
      done
    fi
  fi
  if process_is_running "${pid}"; then
    echo "error: newly started vLLM pid ${pid} could not be terminated" >&2
    return 1
  fi
  wait "${pid}" 2>/dev/null || true

  deadline=$((SECONDS + VLLM_START_CLEANUP_TIMEOUT))
  while (( SECONDS < deadline )) && tcp_port_open; do
    sleep 1
  done
  if tcp_port_open; then
    echo "error: ${VLLM_PROBE_HOST}:${VLLM_PORT} remains open after startup rollback" >&2
    return 1
  fi

  state_pid=""
  if [[ -f "${PID_FILE}" ]]; then
    IFS= read -r state_pid < "${PID_FILE}" || true
  fi
  if [[ "${state_pid}" == "${pid}" ]]; then
    rm -f -- "${PID_FILE}" "${POLICY_FILE}"
  fi
}

hook_ready() {
  [[ "${VLLM_SCHED_POLICY}" == "fcfs" ]] && return 0
  grep -F "[sched_policy_patch] installed policy=${VLLM_SCHED_POLICY} " "${LOG_FILE}" \
    | grep -F "v1=True" >/dev/null
}

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID=""
  IFS= read -r EXISTING_PID < "${PID_FILE}" || true
  if [[ ! "${EXISTING_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: malformed PID file; inspect it manually: ${PID_FILE}" >&2
    exit 1
  elif process_is_running "${EXISTING_PID}"; then
    if ! cmdline_matches "${EXISTING_PID}"; then
      echo "error: PID ${EXISTING_PID} is alive but is not the expected vLLM process" >&2
      echo "Refusing to overwrite ${PID_FILE}." >&2
      exit 1
    fi
    if [[ "${VLLM_REQUIRE_NEW}" == "1" ]]; then
      echo "error: a managed vLLM process already exists (pid ${EXISTING_PID})" >&2
      echo "This caller requires a fresh server and will not reuse it." >&2
      exit 1
    fi
    EXISTING_POLICY=""
    if [[ -f "${POLICY_FILE}" ]]; then
      IFS= read -r EXISTING_POLICY < "${POLICY_FILE}" || true
    fi
    if [[ "${EXISTING_POLICY}" != "${VLLM_SCHED_POLICY}" ]]; then
      echo "error: managed vLLM policy is ${EXISTING_POLICY:-unknown}, requested ${VLLM_SCHED_POLICY}" >&2
      echo "Stop the existing service before changing scheduler policy." >&2
      exit 1
    fi
    if endpoint_ready && hook_ready; then
      echo "vLLM is already healthy (pid ${EXISTING_PID})."
      exit 0
    fi
    echo "error: the managed vLLM process ${EXISTING_PID} is alive but not healthy" >&2
    echo "Log: ${LOG_FILE}" >&2
    exit 1
  else
    rm -f -- "${PID_FILE}"
    rm -f -- "${POLICY_FILE}"
  fi
fi

if endpoint_ready || tcp_port_open; then
  echo "error: an unmanaged service already occupies ${VLLM_PROBE_HOST}:${VLLM_PORT}" >&2
  echo "Refusing to start a second server." >&2
  exit 1
fi

export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_V1
export VLLM_SCHED_POLICY
export PYTHONPATH="${VLLM_HOOK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VLLM_COMMAND=(
  "${ENV_PYTHON}" -m vllm.entrypoints.openai.api_server
  --model "${MODEL_SNAPSHOT}"
  --served-model-name "${MODEL_ID}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --tensor-parallel-size "${VLLM_TP_SIZE}"
  --disable-custom-all-reduce
  --dtype "${VLLM_DTYPE}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --enable-chunked-prefill
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --disable-log-requests
)
if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "1" ]]; then
  VLLM_COMMAND+=(--enable-prefix-caching)
else
  # vLLM V1 may resolve its effective default to enabled even when the
  # positive flag is omitted.  The explicit negative flag is required for a
  # real prefix-cache ablation.
  VLLM_COMMAND+=(--no-enable-prefix-caching)
fi
if (( ${#VLLM_CUDA_GRAPH_SIZE_ARGS[@]} > 0 )); then
  VLLM_COMMAND+=(--cuda-graph-sizes "${VLLM_CUDA_GRAPH_SIZE_ARGS[@]}")
fi
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  VLLM_COMMAND+=(--api-key "${VLLM_API_KEY}")
fi

: > "${LOG_FILE}"
nohup "${VLLM_COMMAND[@]}" >> "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" > "${PID_FILE}.tmp"
mv -f -- "${PID_FILE}.tmp" "${PID_FILE}"
printf '%s\n' "${VLLM_SCHED_POLICY}" > "${POLICY_FILE}.tmp"
mv -f -- "${POLICY_FILE}.tmp" "${POLICY_FILE}"

echo "Started vLLM pid ${SERVER_PID}; waiting for /health and /v1/models."
echo "Log: ${LOG_FILE}"

START_TIME="${SECONDS}"
while (( SECONDS - START_TIME < VLLM_READY_TIMEOUT )); do
  if ! process_is_running "${SERVER_PID}"; then
    cleanup_started_server "${SERVER_PID}" || true
    echo "error: vLLM exited before becoming ready; inspect ${LOG_FILE}" >&2
    exit 1
  fi
  if endpoint_ready; then
    if ! hook_ready; then
      echo "error: endpoints are ready, but the scheduler hook did not report v1=True" >&2
      echo "Policy: ${VLLM_SCHED_POLICY}; log: ${LOG_FILE}" >&2
      cleanup_started_server "${SERVER_PID}" || true
      exit 1
    fi
    echo "vLLM is ready at http://${VLLM_PROBE_HOST}:${VLLM_PORT} (pid ${SERVER_PID})."
    exit 0
  fi
  sleep 2
done

echo "error: timed out after ${VLLM_READY_TIMEOUT}s while starting pid ${SERVER_PID}" >&2
echo "Inspect ${LOG_FILE}. The failed startup is being rolled back." >&2
cleanup_started_server "${SERVER_PID}" || true
exit 1
