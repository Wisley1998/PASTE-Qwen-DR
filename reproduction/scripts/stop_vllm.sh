#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: stop_vllm.sh

Gracefully stop only the vLLM process whose executable, module, model, host,
and port exactly match this reproduction configuration.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    echo "error: stop_vllm.sh does not accept positional arguments" >&2
    usage >&2
    exit 2
    ;;
esac
if (( $# > 1 )); then
  echo "error: stop_vllm.sh does not accept positional arguments" >&2
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
if [[ ${MODEL_SNAPSHOT+x} ]]; then
  echo "error: MODEL_SNAPSHOT is not a registered input; it must be derived from HF_HOME/MODEL_ID/MODEL_REVISION" >&2
  exit 1
fi
if [[ "${HF_HOME}" != /* || ! -d "${HF_HOME}" ]]; then
  echo "error: HF_HOME must be an existing absolute directory: ${HF_HOME}" >&2
  exit 1
fi
HF_HOME="$(cd -- "${HF_HOME}" && pwd -P)"
MODEL_SNAPSHOT="${HF_HOME}/${MODEL_CACHE_KEY}/snapshots/${MODEL_REVISION}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PROBE_HOST="${VLLM_PROBE_HOST:-${VLLM_HOST}}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_SHUTDOWN_TIMEOUT="${VLLM_SHUTDOWN_TIMEOUT:-60}"
VLLM_STATE_DIR="${VLLM_STATE_DIR:-${REPO_ROOT}/reproduction/run}"
PID_FILE="${VLLM_STATE_DIR}/vllm_${VLLM_PORT}.pid"
POLICY_FILE="${VLLM_STATE_DIR}/vllm_${VLLM_PORT}.policy"
LOCK_DIR="${PID_FILE}.lock"

if [[ "${VLLM_HOST}" == "0.0.0.0" ]]; then
  VLLM_PROBE_HOST="${VLLM_PROBE_HOST/0.0.0.0/127.0.0.1}"
fi

if [[ ! "${VLLM_PORT}" =~ ^[1-9][0-9]*$ ]] || (( VLLM_PORT > 65535 )); then
  echo "error: VLLM_PORT must be in 1..65535" >&2
  exit 1
fi
if [[ ! "${VLLM_SHUTDOWN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: VLLM_SHUTDOWN_TIMEOUT must be a positive integer" >&2
  exit 1
fi
if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "error: expected environment Python not found: ${ENV_PYTHON}" >&2
  exit 1
fi
if [[ -d "${MODEL_SNAPSHOT}" ]]; then
  MODEL_SNAPSHOT="$(cd -- "${MODEL_SNAPSHOT}" && pwd -P)"
fi
if [[ "${MODEL_SNAPSHOT}" != "${HF_HOME}/${MODEL_CACHE_KEY}/snapshots/${MODEL_REVISION}" ]]; then
  echo "error: pinned model snapshot resolves outside its exact revision path" >&2
  exit 1
fi

mkdir -p -- "${VLLM_STATE_DIR}"
if ! mkdir -- "${LOCK_DIR}" 2>/dev/null; then
  echo "error: another start/stop operation holds ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir -- "${LOCK_DIR}" 2>/dev/null || true' EXIT

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

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No managed vLLM PID file exists at ${PID_FILE}."
  exit 0
fi

PID=""
IFS= read -r PID < "${PID_FILE}" || true
if [[ ! "${PID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: malformed PID file; refusing to act: ${PID_FILE}" >&2
  exit 1
fi

if ! process_is_running "${PID}"; then
  if tcp_port_open; then
    echo "error: managed PID ${PID} exited, but ${VLLM_PROBE_HOST}:${VLLM_PORT} is still open" >&2
    echo "State files are retained for manual inspection." >&2
    exit 1
  fi
  rm -f -- "${PID_FILE}"
  rm -f -- "${POLICY_FILE}"
  echo "Removed stale PID file for exited process ${PID}."
  exit 0
fi

EXPECTED_PYTHON="$(readlink -f -- "${ENV_PYTHON}")"
ACTUAL_PYTHON="$(readlink -f -- "/proc/${PID}/exe" 2>/dev/null || true)"
if [[ "${ACTUAL_PYTHON}" != "${EXPECTED_PYTHON}" ]]; then
  echo "error: PID ${PID} executable does not match ${EXPECTED_PYTHON}; refusing to stop it" >&2
  exit 1
fi

ARGV=()
mapfile -d '' -t ARGV < "/proc/${PID}/cmdline"
SAW_MODEL=0
SAW_SERVED_MODEL=0
SAW_HOST=0
SAW_PORT=0
if [[ "${ARGV[1]:-}" != "-m" || "${ARGV[2]:-}" != "vllm.entrypoints.openai.api_server" ]]; then
  echo "error: PID ${PID} is not running the expected vLLM API-server module" >&2
  exit 1
fi
for (( INDEX = 0; INDEX < ${#ARGV[@]}; INDEX++ )); do
  case "${ARGV[INDEX]}" in
    --model)
      [[ "${ARGV[INDEX + 1]:-}" == "${MODEL_SNAPSHOT}" ]] || {
        echo "error: PID ${PID} model path does not match; refusing to stop it" >&2
        exit 1
      }
      (( SAW_MODEL += 1 ))
      ;;
    --served-model-name)
      [[ "${ARGV[INDEX + 1]:-}" == "${MODEL_ID}" ]] || {
        echo "error: PID ${PID} served-model name does not match; refusing to stop it" >&2
        exit 1
      }
      (( SAW_SERVED_MODEL += 1 ))
      ;;
    --host)
      [[ "${ARGV[INDEX + 1]:-}" == "${VLLM_HOST}" ]] || {
        echo "error: PID ${PID} host does not match; refusing to stop it" >&2
        exit 1
      }
      (( SAW_HOST += 1 ))
      ;;
    --port)
      [[ "${ARGV[INDEX + 1]:-}" == "${VLLM_PORT}" ]] || {
        echo "error: PID ${PID} port does not match; refusing to stop it" >&2
        exit 1
      }
      (( SAW_PORT += 1 ))
      ;;
  esac
done
if ! (( SAW_MODEL == 1 && SAW_SERVED_MODEL == 1 && SAW_HOST == 1 && SAW_PORT == 1 )); then
  echo "error: PID ${PID} command line does not exactly match this vLLM configuration" >&2
  echo "Refusing to send a signal; PID file retained at ${PID_FILE}." >&2
  exit 1
fi

DESCENDANT_PIDS=()
mapfile -t DESCENDANT_PIDS < <(
  ps -eo pid=,ppid= | awk -v root="${PID}" '
    { parent[$1] = $2 }
    END {
      for (pid in parent) {
        current = pid
        hops = 0
        while ((current in parent) && current != root && hops < 10000) {
          current = parent[current]
          hops++
        }
        if (pid != root && current == root) print pid
      }
    }
  ' | sort -n
)

descendants_are_running() {
  local child
  for child in "${DESCENDANT_PIDS[@]}"; do
    if process_is_running "${child}"; then
      return 0
    fi
  done
  return 1
}

clear_owned_state() {
  local state_pid=""
  if [[ -f "${PID_FILE}" ]]; then
    IFS= read -r state_pid < "${PID_FILE}" || true
  fi
  if [[ "${state_pid}" != "${PID}" ]]; then
    echo "error: PID state changed while stop held its lock; refusing to remove it" >&2
    return 1
  fi
  rm -f -- "${PID_FILE}" "${POLICY_FILE}"
}

kill -TERM "${PID}"
echo "Sent SIGTERM to vLLM pid ${PID}; waiting up to ${VLLM_SHUTDOWN_TIMEOUT}s."
START_TIME="${SECONDS}"
while (( SECONDS - START_TIME < VLLM_SHUTDOWN_TIMEOUT )); do
  if ! process_is_running "${PID}" && ! descendants_are_running && ! tcp_port_open; then
    clear_owned_state
    echo "vLLM pid ${PID} stopped cleanly."
    exit 0
  fi
  sleep 1
done

echo "error: pid ${PID}, a recorded descendant, or its port did not stop after ${VLLM_SHUTDOWN_TIMEOUT}s" >&2
echo "No force-kill was sent; PID file retained at ${PID_FILE}." >&2
exit 1
