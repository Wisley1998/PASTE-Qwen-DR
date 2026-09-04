#!/bin/bash

# A non-interactive Bash may source $BASH_ENV before executing this file.  Such
# an invocation is rejected; the documented launcher uses env -i.  Every other
# invocation is immediately re-executed under a minimal environment, carrying
# forward only path controls (never runtime knobs).
if [[ ${BASH_ENV+x} ]]; then
  echo "error: BASH_ENV is forbidden for the strict matrix wrapper" >&2
  exit 1
fi
if [[ "${PASTE_WRAPPER_CLEAN_REEXEC:-0}" != "1" ]]; then
  WRAPPER_ENTRY="$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}")"
  CLEAN_LAUNCH_ENV=(
    "HOME=/home/aiscuser"
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    "PASTE_WRAPPER_CLEAN_REEXEC=1"
  )
  for LAUNCH_CONTROL in \
    PASTE_STRICT_CONFIG PASTE_STRICT_BUNDLE PASTE_RUN_ROOT \
    PASTE_VALIDATE_ONLY PASTE_SCHEDULER_HOOK_FILE; do
    if [[ -v "${LAUNCH_CONTROL}" ]]; then
      CLEAN_LAUNCH_ENV+=("${LAUNCH_CONTROL}=${!LAUNCH_CONTROL}")
    fi
  done
  exec /usr/bin/env -i "${CLEAN_LAUNCH_ENV[@]}" \
    /bin/bash --noprofile --norc "${WRAPPER_ENTRY}" "$@"
fi
unset PASTE_WRAPPER_CLEAN_REEXEC

set -Eeuo pipefail

# Do not resolve any helper through a caller-controlled PATH.  The server and
# client later run from an even narrower `env -i` environment frozen in the
# formal config.
BOOTSTRAP_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PATH="${BOOTSTRAP_PATH}"
export PATH
hash -r

usage() {
  cat <<'EOF' >&2
Usage: run_strict_trace_abef_matrix.sh {tuning|final}

Run one complete four-block Williams cycle. Every block/cell starts a fresh
vLLM server and a fresh in-process tool broker. Set PASTE_VALIDATE_ONLY=1 to
validate inputs and print the preregistered order without starting a GPU.

Required after bundle preparation:
  PASTE_STRICT_BUNDLE=/path/to/strict/bundle.json

Optional:
  PASTE_STRICT_CONFIG=/path/to/frozen.env
  PASTE_RUN_ROOT=/path/to/output

GPU groups, protected PID, capacities, model paths, and all server/client knobs
come only from the frozen config; inherited overrides are discarded.
EOF
}

if (( $# != 1 )) || [[ "$1" != "tuning" && "$1" != "final" ]]; then
  usage
  exit 2
fi
ROLE="$1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd -- "${REPO_ROOT}"
CONFIG="${PASTE_STRICT_CONFIG:-${REPO_ROOT}/reproduction/configs/strict_trace_abef.env.example}"
BUNDLE="${PASTE_STRICT_BUNDLE:-${REPO_ROOT}/reproduction/artifacts/strict_trace_abef/bundle.json}"
RUN_ROOT="${PASTE_RUN_ROOT:-${REPO_ROOT}/reproduction/artifacts/strict_trace_abef/runs/${ROLE}}"
VALIDATE_ONLY="${PASTE_VALIDATE_ONLY:-0}"
SCHEDULER_HOOK="${PASTE_SCHEDULER_HOOK_FILE:-${REPO_ROOT}/scripts/pythonhooks/sched_policy_patch.py}"
SITECUSTOMIZE="${REPO_ROOT}/scripts/pythonhooks/sitecustomize.py"
SMOKE_SCRIPT="${SCRIPT_DIR}/smoke_vllm.py"
START_SCRIPT="${SCRIPT_DIR}/start_vllm.sh"
STOP_SCRIPT="${SCRIPT_DIR}/stop_vllm.sh"
MATRIX_WRAPPER="${BASH_SOURCE[0]}"

# Launcher path controls have already been copied into local shell variables.
# Remove every inherited variable family known to affect Python, vLLM, CUDA,
# collectives, tokenization, or this harness before sourcing the frozen config.
# None of these inherited values is forwarded to a cell process.
SCRUBBED_RUNTIME_VARIABLES=()
mapfile -t inherited_environment_names < <(compgen -e)
for variable_name in "${inherited_environment_names[@]}"; do
  case "${variable_name}" in
    PASTE_*|VLLM_*|MODEL_*|HF_*|TRANSFORMERS_*|PYTHON*|PYTORCH_*|TORCH_*|\
    CUDA_*|CUBLAS_*|CUDNN_*|NCCL_*|RAY_*|FLASHINFER_*|XFORMERS_*|TRITON_*|\
    TOKENIZERS_*|OMP_*|MKL_*|ORACLE_*|LD_PRELOAD|LD_LIBRARY_PATH)
      SCRUBBED_RUNTIME_VARIABLES+=("${variable_name}")
      unset "${variable_name}"
      ;;
  esac
done

if [[ "${VALIDATE_ONLY}" != "0" && "${VALIDATE_ONLY}" != "1" ]]; then
  echo "error: PASTE_VALIDATE_ONLY must be 0 or 1" >&2
  exit 2
fi
for required in \
  "${CONFIG}" "${BUNDLE}" "${SCHEDULER_HOOK}" "${SITECUSTOMIZE}" \
  "${SMOKE_SCRIPT}" "${START_SCRIPT}" "${STOP_SCRIPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "error: required frozen input is missing: ${required}" >&2
    exit 1
  fi
done

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

CONFIG_SHA256="$(sha256_file "${CONFIG}")"
BUNDLE_FILE_SHA256="$(sha256_file "${BUNDLE}")"
SCHEDULER_HOOK_SHA256="$(sha256_file "${SCHEDULER_HOOK}")"
SITECUSTOMIZE_SHA256="$(sha256_file "${SITECUSTOMIZE}")"
SMOKE_SCRIPT_SHA256="$(sha256_file "${SMOKE_SCRIPT}")"
START_SCRIPT_SHA256="$(sha256_file "${START_SCRIPT}")"
STOP_SCRIPT_SHA256="$(sha256_file "${STOP_SCRIPT}")"
MATRIX_WRAPPER_SHA256="$(sha256_file "${MATRIX_WRAPPER}")"

frozen_inputs_match() {
  [[ "$(sha256_file "${CONFIG}")" == "${CONFIG_SHA256}" \
    && "$(sha256_file "${BUNDLE}")" == "${BUNDLE_FILE_SHA256}" \
    && "$(sha256_file "${SCHEDULER_HOOK}")" == "${SCHEDULER_HOOK_SHA256}" \
    && "$(sha256_file "${SITECUSTOMIZE}")" == "${SITECUSTOMIZE_SHA256}" \
    && "$(sha256_file "${SMOKE_SCRIPT}")" == "${SMOKE_SCRIPT_SHA256}" \
    && "$(sha256_file "${START_SCRIPT}")" == "${START_SCRIPT_SHA256}" \
    && "$(sha256_file "${STOP_SCRIPT}")" == "${STOP_SCRIPT_SHA256}" \
    && "$(sha256_file "${MATRIX_WRAPPER}")" == "${MATRIX_WRAPPER_SHA256}" ]]
}

# Reject executable shell constructs before sourcing even a caller-selected
# config.  Full path/hash binding to the bundle is checked immediately after
# the literal exports are loaded.
BOOTSTRAP_PYTHON="/home/aiscuser/.conda/envs/paste/bin/python"
if [[ ! -x "${BOOTSTRAP_PYTHON}" ]]; then
  echo "error: frozen bootstrap Python is missing: ${BOOTSTRAP_PYTHON}" >&2
  exit 1
fi
/usr/bin/env -i HOME="/home/aiscuser" PATH="${BOOTSTRAP_PATH}" \
  "${BOOTSTRAP_PYTHON}" -I - "${SCRIPT_DIR}" "${CONFIG}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from run_strict_trace_abef import (
    _formal_config_exports,
    _validate_formal_environment_contract,
)
exports = _formal_config_exports(Path(sys.argv[2]).resolve())
_validate_formal_environment_contract(exports)
PY

# shellcheck source=/dev/null
source "${CONFIG}"
: "${PASTE_GPU_GROUPS:?formal config must export PASTE_GPU_GROUPS}"
: "${PASTE_PROTECTED_PID:?formal config must export PASTE_PROTECTED_PID}"
: "${PASTE_MAX_ACTIVE_TASKS:?formal config must export PASTE_MAX_ACTIVE_TASKS}"
: "${PASTE_VISIT_CAPACITY:?formal config must export PASTE_VISIT_CAPACITY}"
: "${PASTE_SPECULATIVE_CAP:?formal config must export PASTE_SPECULATIVE_CAP}"
: "${PASTE_REQUEST_TIMEOUT_S:?formal config must export PASTE_REQUEST_TIMEOUT_S}"
: "${PASTE_DEFAULT_PREDICTED_OUTPUT_TOKENS:?formal config must export PASTE_DEFAULT_PREDICTED_OUTPUT_TOKENS}"
: "${VLLM_HOOK_DIR:?formal config must export VLLM_HOOK_DIR}"
: "${PASTE_RUNTIME_HOME:?formal config must export PASTE_RUNTIME_HOME}"
: "${PASTE_RUNTIME_PATH:?formal config must export PASTE_RUNTIME_PATH}"
: "${PASTE_RUNTIME_LD_LIBRARY_PATH:?formal config must export PASTE_RUNTIME_LD_LIBRARY_PATH}"
: "${PASTE_RUNTIME_TMPDIR:?formal config must export PASTE_RUNTIME_TMPDIR}"
: "${PASTE_RUNTIME_LANG:?formal config must export PASTE_RUNTIME_LANG}"
: "${PASTE_RUNTIME_TZ:?formal config must export PASTE_RUNTIME_TZ}"
GPU_GROUPS="${PASTE_GPU_GROUPS}"
PROTECTED_PID="${PASTE_PROTECTED_PID}"
if [[ ! "${PROTECTED_PID}" =~ ^[1-9][0-9]*$ \
   || ! "${PASTE_MAX_ACTIVE_TASKS}" =~ ^[1-9][0-9]*$ \
   || ! "${PASTE_VISIT_CAPACITY}" =~ ^[1-9][0-9]*$ \
   || ! "${PASTE_SPECULATIVE_CAP}" =~ ^[0-9]+$ ]]; then
  echo "error: frozen PID/client/tool capacities are malformed" >&2
  exit 2
fi
if (( PASTE_SPECULATIVE_CAP > PASTE_VISIT_CAPACITY )); then
  echo "error: frozen speculation capacity exceeds tool capacity" >&2
  exit 2
fi
PYTHON_BIN="${PASTE_ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: reproduction Python is missing: ${PYTHON_BIN}" >&2
  exit 1
fi

mapfile -t FORMAL_EXPORT_NAMES < <(
  sed -nE 's/^export ([A-Z][A-Z0-9_]*)=.*/\1/p' "${CONFIG}"
)
FROZEN_CELL_ENV=(
  "HOME=${PASTE_RUNTIME_HOME}"
  "PATH=${PASTE_RUNTIME_PATH}"
  "LD_LIBRARY_PATH=${PASTE_RUNTIME_LD_LIBRARY_PATH}"
  "TMPDIR=${PASTE_RUNTIME_TMPDIR}"
  "LANG=${PASTE_RUNTIME_LANG}"
  "LC_ALL=${PASTE_RUNTIME_LANG}"
  "TZ=${PASTE_RUNTIME_TZ}"
)
for variable_name in "${FORMAL_EXPORT_NAMES[@]}"; do
  FROZEN_CELL_ENV+=("${variable_name}=${!variable_name}")
done
run_in_frozen_environment() {
  /usr/bin/env -i "${FROZEN_CELL_ENV[@]}" "$@"
}
run_server_in_frozen_environment() {
  local safe_cwd="$1"
  shift
  (
    cd -- "${safe_cwd}"
    run_in_frozen_environment "$@"
  )
}
FROZEN_CELL_ENV_SHA256="$({
  printf '%s\n' "${FROZEN_CELL_ENV[@]}"
} | LC_ALL=C sort | sha256sum | awk '{print $1}')"
mapfile -t SCRUBBED_RUNTIME_VARIABLES_SORTED < <(
  printf '%s\n' "${SCRUBBED_RUNTIME_VARIABLES[@]}" | LC_ALL=C sort -u
)

MODEL_CACHE_KEY="models--${MODEL_ID//\//--}"
PINNED_MODEL_SNAPSHOT="${HF_HOME}/${MODEL_CACHE_KEY}/snapshots/${MODEL_REVISION}"
if [[ ! -d "${PINNED_MODEL_SNAPSHOT}" \
   || "$(cd -- "${PINNED_MODEL_SNAPSHOT}" && pwd -P)" != "${PINNED_MODEL_SNAPSHOT}" \
   || ! -f "${PINNED_MODEL_SNAPSHOT}/config.json" ]]; then
  echo "error: exact frozen model snapshot is absent or resolves elsewhere: ${PINNED_MODEL_SNAPSHOT}" >&2
  exit 1
fi
MODEL_CONFIG_SHA256="$(sha256_file "${PINNED_MODEL_SNAPSHOT}/config.json")"
model_inventory_sha256() {
  run_in_frozen_environment "${PYTHON_BIN}" -I - \
    "${SCRIPT_DIR}" "${PINNED_MODEL_SNAPSHOT}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from run_strict_trace_abef import _model_snapshot_inventory
print(_model_snapshot_inventory(Path(sys.argv[2]))["inventory_sha256"])
PY
}
MODEL_INVENTORY_SHA256="$(model_inventory_sha256)"

write_scheduler_runtime_evidence() {
  local marker_file="$1"
  local output_file="$2"
  local policy="$3"
  local cell="$4"
  local server_pid="$5"
  local phase="$6"
  run_in_frozen_environment "${PYTHON_BIN}" -I - \
    "${marker_file}" "${output_file}" "${policy}" "${cell}" \
    "${server_pid}" "${phase}" "${SCHEDULER_HOOK}" \
    "${SCHEDULER_HOOK_SHA256}" "${PYTHON_BIN}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

marker_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
policy = sys.argv[3]
cell = sys.argv[4]
server_pid = int(sys.argv[5])
phase = sys.argv[6]
hook_path = Path(sys.argv[7]).resolve(strict=True)
hook_sha256 = sys.argv[8]
python_path = Path(sys.argv[9]).resolve(strict=True)
expected_use = cell in {"E", "F"}
if expected_use != (policy == "online_joint_pacer_v2"):
    raise RuntimeError("cell/policy scheduler-runtime expectation is inconsistent")
if output_path.exists():
    raise FileExistsError(f"refusing to overwrite scheduler evidence: {output_path}")

def proc_fields(pid: int) -> tuple[int, int]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    tail = raw.rsplit(") ", 1)[1].split()
    return int(tail[1]), int(tail[19])

marker = None
marker_sha256 = None
caller_relation = None
if expected_use:
    if not marker_path.is_file():
        raise RuntimeError("patched scheduler did not emit runtime evidence during smoke")
    marker_bytes = marker_path.read_bytes()
    marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
    marker = json.loads(marker_bytes)
    required = {
        "schema": "paste.vllm.scheduler_runtime_use.v1",
        "policy": policy,
        "scheduler_api": "v1.Scheduler.schedule",
        "scheduler_hook_path": str(hook_path),
        "scheduler_hook_sha256": hook_sha256,
        "python_safe_path_enforced": True,
        "cwd_import_filter_enforced": True,
        "working_directory_importable": False,
    }
    for name, expected in required.items():
        if marker.get(name) != expected:
            raise RuntimeError(
                f"scheduler runtime marker {name} mismatch: "
                f"{marker.get(name)!r} != {expected!r}"
            )
    safe_working_directory = marker.get("safe_working_directory")
    if (
        not isinstance(safe_working_directory, str)
        or not safe_working_directory
        or not Path(safe_working_directory).is_absolute()
        or marker.get("working_directory") != safe_working_directory
    ):
        raise RuntimeError("scheduler process did not attest its frozen safe working directory")
    caller_pid = marker.get("pid")
    if not isinstance(caller_pid, int) or caller_pid <= 0:
        raise RuntimeError("scheduler runtime marker has invalid PID")
    actual_python = Path(f"/proc/{caller_pid}/exe").resolve(strict=True)
    if actual_python != python_path:
        raise RuntimeError("scheduler runtime marker PID uses the wrong Python")
    current_pid = caller_pid
    visited = set()
    while current_pid > 1 and current_pid not in visited:
        visited.add(current_pid)
        if current_pid == server_pid:
            caller_relation = (
                "server_process" if caller_pid == server_pid else "server_descendant"
            )
            break
        current_pid, _ = proc_fields(current_pid)
    if caller_relation is None:
        raise RuntimeError("scheduler runtime marker PID is not the managed server or its descendant")
    if caller_relation != "server_descendant":
        raise RuntimeError("vLLM V1 scheduler runtime marker was not emitted by an engine child")
    _, current_start_ticks = proc_fields(caller_pid)
    if marker.get("process_start_ticks") != current_start_ticks:
        raise RuntimeError("scheduler runtime marker PID identity changed")
else:
    if marker_path.exists():
        raise RuntimeError("FCFS cell unexpectedly executed the scheduler hook")

payload = {
    "schema": "paste.paper.scheduler_runtime_evidence.v1",
    "cell": cell,
    "phase": phase,
    "server_pid": server_pid,
    "expected_policy": policy,
    "hook_runtime_use_expected": expected_use,
    "patched_scheduler_invocation_verified": expected_use,
    "no_scheduler_hook_runtime_use_verified": not expected_use,
    "scheduler_hook_path": str(hook_path),
    "scheduler_hook_sha256": hook_sha256,
    "runtime_marker_path": str(marker_path),
    "runtime_marker_sha256": marker_sha256,
    "scheduler_calling_pid": marker.get("pid") if marker else None,
    "scheduler_calling_process_relation": caller_relation,
    "runtime_marker": marker,
}
encoded = (
    json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
).encode("utf-8")
temporary = output_path.with_name(output_path.name + ".tmp")
fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    written = os.write(fd, encoded)
    if written != len(encoded):
        raise RuntimeError("short write for scheduler verification evidence")
    os.fsync(fd)
finally:
    os.close(fd)
os.replace(temporary, output_path)
PY
}

# Validate all checksums and role constraints before touching a GPU.
run_in_frozen_environment "${PYTHON_BIN}" -I - \
  "${SCRIPT_DIR}" "${BUNDLE}" "${ROLE}" "${CONFIG}" \
  "${SCHEDULER_HOOK}" "${SITECUSTOMIZE}" "${START_SCRIPT}" "${STOP_SCRIPT}" \
  "${VLLM_HOOK_DIR}" "${PASTE_MAX_ACTIVE_TASKS}" "${PASTE_VISIT_CAPACITY}" \
  "${PASTE_SPECULATIVE_CAP}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from run_strict_trace_abef import load_strict_bundle, validate_matrix_execution_contract
loaded = load_strict_bundle(Path(sys.argv[2]).resolve(), sys.argv[3])
validate_matrix_execution_contract(
    loaded=loaded,
    config_path=Path(sys.argv[4]),
    scheduler_hook_path=Path(sys.argv[5]),
    sitecustomize_path=Path(sys.argv[6]),
    start_vllm_path=Path(sys.argv[7]),
    stop_vllm_path=Path(sys.argv[8]),
    hook_dir=Path(sys.argv[9]),
    max_active_tasks=int(sys.argv[10]),
    visit_capacity=int(sys.argv[11]),
    speculative_cap=int(sys.argv[12]),
)
print(
    f"validated strict bundle={loaded['bundle']['bundle_sha256']} "
    f"role={sys.argv[3]} replicas={len(loaded['public']['traces'])}"
)
PY

IFS=';' read -r -a GROUP_ARRAY <<< "${GPU_GROUPS}"
if (( ${#GROUP_ARRAY[@]} < 2 )); then
  echo "error: PASTE_GPU_GROUPS must contain at least two semicolon-separated groups" >&2
  exit 2
fi
ORDERS=("A B F E" "B E A F" "E F B A" "F A E B")
for index in "${!ORDERS[@]}"; do
  group="${GROUP_ARRAY[index % ${#GROUP_ARRAY[@]}]}"
  echo "cycle-01-block-$(printf '%02d' "$((index + 1))") gpu=${group} order=${ORDERS[index]}"
done
if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  exit 0
fi

protected_start_tick() {
  local line tail
  local -a fields=()
  [[ -r "/proc/${PROTECTED_PID}/stat" ]] || return 1
  IFS= read -r line < "/proc/${PROTECTED_PID}/stat"
  tail="${line##*) }"
  read -r -a fields <<< "${tail}"
  [[ -n "${fields[19]:-}" ]] || return 1
  printf '%s\n' "${fields[19]}"
}

if ! PROTECTED_START_TICK="$(protected_start_tick)"; then
  echo "error: protected ResNet PID ${PROTECTED_PID} is not live" >&2
  exit 1
fi
PROTECTED_INITIAL_COMMAND="$(tr '\0' ' ' < "/proc/${PROTECTED_PID}/cmdline")"
if [[ "${PROTECTED_INITIAL_COMMAND,,}" != *resnet* ]]; then
  echo "error: protected PID ${PROTECTED_PID} is not the expected ResNet process" >&2
  exit 1
fi

capture_platform_snapshot() {
  local cell_root="$1"
  local phase="$2"
  local current_tick command_summary
  if ! current_tick="$(protected_start_tick)" || [[ "${current_tick}" != "${PROTECTED_START_TICK}" ]]; then
    echo "error: protected ResNet PID ${PROTECTED_PID} exited or changed identity" >&2
    return 1
  fi
  run_in_frozen_environment nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,clocks.current.graphics,clocks.current.sm,clocks.current.memory,pstate \
    --format=csv > "${cell_root}/nvidia_smi_${phase}.csv"
  command_summary="$(tr '\0' ' ' < "/proc/${PROTECTED_PID}/cmdline")"
  {
    printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'pid=%s\n' "${PROTECTED_PID}"
    printf 'proc_start_tick=%s\n' "${current_tick}"
    printf 'comm=%s\n' "$(< "/proc/${PROTECTED_PID}/comm")"
    printf 'exe=%s\n' "$(readlink -f -- "/proc/${PROTECTED_PID}/exe")"
    printf 'cmdline_sha256=%s\n' "$(sha256_file "/proc/${PROTECTED_PID}/cmdline")"
    printf 'cmdline_summary=%s\n' "${command_summary:0:512}"
  } > "${cell_root}/protected_pid_${phase}.txt"
}

if run_in_frozen_environment curl --fail --silent --max-time 2 \
  "http://${VLLM_PROBE_HOST:-127.0.0.1}:${VLLM_PORT:-8100}/health" >/dev/null 2>&1; then
  echo "error: a server is already running; strict cells require fresh servers" >&2
  exit 1
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "error: refusing to reuse matrix run root: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p -- "${RUN_ROOT}"
STARTED_MARKER="${RUN_ROOT}/$(printf '%s' "${ROLE}" | tr '[:lower:]' '[:upper:]')_STARTED"
if ! (set -o noclobber; printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STARTED_MARKER}") 2>/dev/null; then
  echo "error: started marker already exists: ${STARTED_MARKER}" >&2
  exit 1
fi

SERVER_MANAGED=0
CURRENT_GPU_GROUP=""
CURRENT_POLICY=""
CURRENT_STATE_DIR=""
CURRENT_LOG_DIR=""
cleanup() {
  if (( SERVER_MANAGED == 1 )); then
    run_in_frozen_environment \
      "CUDA_VISIBLE_DEVICES=${CURRENT_GPU_GROUP}" \
      "VLLM_SCHED_POLICY=${CURRENT_POLICY}" \
      "VLLM_STATE_DIR=${CURRENT_STATE_DIR}" \
      "VLLM_LOG_DIR=${CURRENT_LOG_DIR}" \
      "${STOP_SCRIPT}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for block_index in "${!ORDERS[@]}"; do
  block_id="cycle-01-block-$(printf '%02d' "$((block_index + 1))")"
  gpu_group="${GROUP_ARRAY[block_index % ${#GROUP_ARRAY[@]}]}"
  read -r -a cells <<< "${ORDERS[block_index]}"
  for cell_index in "${!cells[@]}"; do
    cell="${cells[cell_index]}"
    order_position="$((cell_index + 1))"
    case "${cell}" in
      A|B) policy="fcfs" ;;
      E|F) policy="online_joint_pacer_v2" ;;
      *) echo "error: bad cell ${cell}" >&2; exit 2 ;;
    esac
    cell_root="${RUN_ROOT}/${block_id}/${cell}"
    current_state_dir="${cell_root}/state"
    current_log_dir="${cell_root}/server"
    CURRENT_GPU_GROUP="${gpu_group}"
    CURRENT_POLICY="${policy}"
    CURRENT_STATE_DIR="${current_state_dir}"
    CURRENT_LOG_DIR="${current_log_dir}"
    mkdir -p -- "${cell_root}"
    safe_python_cwd="${cell_root}/empty_python_cwd"
    runtime_hook_dir="${cell_root}/runtime_python_hook"
    scheduler_runtime_marker="${current_state_dir}/vllm_${VLLM_PORT}.scheduler_runtime.json"
    scheduler_runtime_evidence_pre="${cell_root}/scheduler_runtime_after_smoke.json"
    scheduler_runtime_evidence_post="${cell_root}/scheduler_runtime_after_cell.json"
    mkdir -- "${safe_python_cwd}" "${runtime_hook_dir}"
    ln -s -- "${SITECUSTOMIZE}" "${runtime_hook_dir}/sitecustomize.py"
    ln -s -- "${SCHEDULER_HOOK}" "${runtime_hook_dir}/sched_policy_patch.py"
    chmod 0500 -- "${safe_python_cwd}" "${runtime_hook_dir}"
    if find "${safe_python_cwd}" -mindepth 1 -print -quit | grep -q .; then
      echo "error: per-cell Python working directory is not empty" >&2
      exit 1
    fi
    runtime_environment_evidence="${cell_root}/runtime_environment.txt"
    current_model_inventory_sha256="$(model_inventory_sha256)"
    if [[ "${current_model_inventory_sha256}" != "${MODEL_INVENTORY_SHA256}" ]]; then
      echo "error: pinned model snapshot inventory changed before ${block_id}/${cell}" >&2
      exit 1
    fi
    {
      printf 'schema=paste.paper.frozen_cell_environment.v1\n'
      printf 'frozen_cell_environment_sha256=%s\n' "${FROZEN_CELL_ENV_SHA256}"
      printf 'scrubbed_external_variables=%s\n' "$({
        printf '%s\n' "${SCRUBBED_RUNTIME_VARIABLES_SORTED[@]}"
      } | paste -sd, -)"
      printf 'cuda_visible_devices=%s\n' "${gpu_group}"
      printf 'server_scheduler_policy=%s\n' "${policy}"
      printf 'server_state_dir=%s\n' "${current_state_dir}"
      printf 'server_log_dir=%s\n' "${current_log_dir}"
      printf 'model_snapshot=%s\n' "${PINNED_MODEL_SNAPSHOT}"
      printf 'model_config_sha256=%s\n' "${MODEL_CONFIG_SHA256}"
      printf 'model_snapshot_inventory_sha256=%s\n' "${MODEL_INVENTORY_SHA256}"
      printf 'server_python=%s\n' "${PYTHON_BIN}"
      printf 'wrapper_working_directory=%s\n' "${REPO_ROOT}"
      printf 'server_empty_working_directory=%s\n' "${safe_python_cwd}"
      printf 'server_pythonpath=%s\n' "${runtime_hook_dir}"
      printf 'runtime_sitecustomize_resolves_to=%s\n' "$(readlink -f -- "${runtime_hook_dir}/sitecustomize.py")"
      printf 'runtime_scheduler_hook_resolves_to=%s\n' "$(readlink -f -- "${runtime_hook_dir}/sched_policy_patch.py")"
      printf 'scheduler_runtime_marker=%s\n' "${scheduler_runtime_marker}"
      printf 'scheduler_runtime_use_expected=%s\n' "$([[ "${cell}" == "E" || "${cell}" == "F" ]] && printf true || printf false)"
      printf 'pytorch_cuda_alloc_conf=expandable_segments:True\n'
      printf '%s\n' "${FROZEN_CELL_ENV[@]}" | LC_ALL=C sort
    } > "${runtime_environment_evidence}"
    runtime_environment_evidence_sha256="$(sha256_file "${runtime_environment_evidence}")"
    if ! frozen_inputs_match; then
      echo "error: a frozen matrix input changed after preflight" >&2
      exit 1
    fi
    capture_platform_snapshot "${cell_root}" pre
    server_start_uuid="$(< /proc/sys/kernel/random/uuid)"
    run_server_in_frozen_environment "${safe_python_cwd}" \
      "CUDA_VISIBLE_DEVICES=${gpu_group}" \
      "VLLM_SCHED_POLICY=${policy}" \
      "VLLM_STATE_DIR=${current_state_dir}" \
      "VLLM_LOG_DIR=${current_log_dir}" \
      "VLLM_RUNTIME_HOOK_DIR=${runtime_hook_dir}" \
      "VLLM_SAFE_WORKING_DIR=${safe_python_cwd}" \
      "${START_SCRIPT}"
    SERVER_MANAGED=1
    server_pid_file="${current_state_dir}/vllm_${VLLM_PORT}.pid"
    server_pid="$(< "${server_pid_file}")"
    server_instance_id="server-${server_start_uuid}-pid-${server_pid}"
    installed_vllm_version="$(run_in_frozen_environment "${PYTHON_BIN}" -I -c \
      'from importlib.metadata import version; print(version("vllm"))')"
    if [[ "${installed_vllm_version}" != "0.10.1" ]]; then
      echo "error: post-start vLLM version evidence is not 0.10.1: ${installed_vllm_version}" >&2
      exit 1
    fi
    smoke_evidence="${cell_root}/smoke_vllm.txt"
    {
      printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'contract=fresh process plus identical standardized smoke-warmed prefix state; no evaluation workload cache\n'
      printf 'vllm_distribution_version=%s\n' "${installed_vllm_version}"
      printf 'vllm_version_requirement=exactly-0.10.1\n'
      printf 'model_snapshot=%s\n' "${PINNED_MODEL_SNAPSHOT}"
      printf 'model_config_sha256=%s\n' "${MODEL_CONFIG_SHA256}"
      printf 'model_snapshot_inventory_sha256=%s\n' "${MODEL_INVENTORY_SHA256}"
      printf 'frozen_cell_environment_sha256=%s\n' "${FROZEN_CELL_ENV_SHA256}"
      printf 'script_sha256=%s\n' "${SMOKE_SCRIPT_SHA256}"
      printf 'command=smoke_vllm.py --max-tokens 64\n'
      run_in_frozen_environment \
        "CUDA_VISIBLE_DEVICES=${gpu_group}" \
        "VLLM_SCHED_POLICY=${policy}" \
        "${PYTHON_BIN}" -I "${SMOKE_SCRIPT}" --max-tokens 64
      printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${smoke_evidence}" 2>&1
    write_scheduler_runtime_evidence \
      "${scheduler_runtime_marker}" "${scheduler_runtime_evidence_pre}" \
      "${policy}" "${cell}" "${server_pid}" "after_standardized_smoke"
    scheduler_runtime_evidence_pre_sha256="$(sha256_file "${scheduler_runtime_evidence_pre}")"
    {
      printf 'scheduler_runtime_evidence_sha256=%s\n' "${scheduler_runtime_evidence_pre_sha256}"
      printf 'scheduler_runtime_marker=%s\n' "${scheduler_runtime_marker}"
    } >> "${smoke_evidence}"
    smoke_evidence_sha256="$(sha256_file "${smoke_evidence}")"
    run_in_frozen_environment \
      "CUDA_VISIBLE_DEVICES=${gpu_group}" \
      "VLLM_SCHED_POLICY=${policy}" \
      "${PYTHON_BIN}" -I "${SCRIPT_DIR}/run_strict_trace_abef.py" run-cell \
      --bundle "${BUNDLE}" \
      --role "${ROLE}" \
      --cell "${cell}" \
      --output-dir "${cell_root}/client" \
      --server-url "http://${VLLM_PROBE_HOST:-127.0.0.1}:${VLLM_PORT:-8100}" \
      --server-policy-file "${current_state_dir}/vllm_${VLLM_PORT}.policy" \
      --server-pid-file "${server_pid_file}" \
      --server-instance-id "${server_instance_id}" \
      --block-id "${block_id}" \
      --order-position "${order_position}" \
      --gpu-ids "${gpu_group}" \
      --config-file "${CONFIG}" \
      --config-file-sha256 "${CONFIG_SHA256}" \
      --scheduler-hook-file "${SCHEDULER_HOOK}" \
      --scheduler-hook-file-sha256 "${SCHEDULER_HOOK_SHA256}" \
      --smoke-evidence-file "${smoke_evidence}" \
      --smoke-evidence-sha256 "${smoke_evidence_sha256}" \
      --runtime-environment-evidence-file "${runtime_environment_evidence}" \
      --runtime-environment-evidence-sha256 "${runtime_environment_evidence_sha256}" \
      --scheduler-runtime-evidence-file "${scheduler_runtime_evidence_pre}" \
      --scheduler-runtime-evidence-sha256 "${scheduler_runtime_evidence_pre_sha256}" \
      --scheduler-runtime-marker-file "${scheduler_runtime_marker}" \
      --max-active-tasks "${PASTE_MAX_ACTIVE_TASKS}" \
      --visit-capacity "${PASTE_VISIT_CAPACITY}" \
      --speculative-cap "${PASTE_SPECULATIVE_CAP}" \
      --default-predicted-output-tokens "${PASTE_DEFAULT_PREDICTED_OUTPUT_TOKENS}" \
      --request-timeout-s "${PASTE_REQUEST_TIMEOUT_S}"
    write_scheduler_runtime_evidence \
      "${scheduler_runtime_marker}" "${scheduler_runtime_evidence_post}" \
      "${policy}" "${cell}" "${server_pid}" "after_evaluation_cell"
    if ! frozen_inputs_match; then
      echo "error: a frozen matrix input changed during the cell" >&2
      exit 1
    fi
    run_in_frozen_environment \
      "CUDA_VISIBLE_DEVICES=${gpu_group}" \
      "VLLM_SCHED_POLICY=${policy}" \
      "VLLM_STATE_DIR=${current_state_dir}" \
      "VLLM_LOG_DIR=${current_log_dir}" \
      "${STOP_SCRIPT}"
    SERVER_MANAGED=0
    if find "${safe_python_cwd}" -mindepth 1 -print -quit | grep -q .; then
      echo "error: vLLM wrote an unregistered file into its empty working directory" >&2
      exit 1
    fi
    capture_platform_snapshot "${cell_root}" post
    server_log="${current_log_dir}/vllm_${VLLM_PORT}.log"
    run_in_frozen_environment "${PYTHON_BIN}" -I - \
      "${cell_root}/client/result.json" \
      "${cell_root}/matrix_evidence.json" \
      "${cell_root}/nvidia_smi_pre.csv" \
      "${cell_root}/nvidia_smi_post.csv" \
      "${cell_root}/protected_pid_pre.txt" \
      "${cell_root}/protected_pid_post.txt" \
      "${smoke_evidence}" \
      "${START_SCRIPT}" \
      "${STOP_SCRIPT}" \
      "${SITECUSTOMIZE}" \
      "${runtime_environment_evidence}" \
      "${scheduler_runtime_evidence_pre}" \
      "${scheduler_runtime_evidence_post}" \
      "${server_log}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

result_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
result = json.loads(result_path.read_text(encoding="utf-8"))

def binding(raw: str) -> dict:
    path = Path(raw).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

row = {
    "block_id": result["block_id"],
    "cell": result["paper_protocol"]["cell"],
    "order_position": result["order_position"],
    "started_wall_s": result["started_wall_s"],
    "ended_wall_s": result["ended_wall_s"],
    "gpu_ids": result["gpu_ids"],
    "server_instance_id": result["server_instance_id"],
    "broker_instance_id": result["broker_instance_id"],
    "service_clock_artifact_sha256": result["paper_protocol"]["service_clock_artifact_sha256"],
    "provenance": result["provenance"],
    "runtime_parameters": result["runtime_parameters"],
    "runtime_parameters_sha256": result["runtime_parameters"]["runtime_parameters_sha256"],
    "runtime_environment_contract": result["runtime_environment_contract"],
    "scheduler_runtime_contract": result["scheduler_runtime_contract"],
    "result_path": str(result_path),
    "platform_evidence": {
        "nvidia_smi_pre": binding(sys.argv[3]),
        "nvidia_smi_post": binding(sys.argv[4]),
        "protected_pid_pre": binding(sys.argv[5]),
        "protected_pid_post": binding(sys.argv[6]),
        "standardized_smoke": binding(sys.argv[7]),
        "start_vllm": binding(sys.argv[8]),
        "stop_vllm": binding(sys.argv[9]),
        "sitecustomize": binding(sys.argv[10]),
        "runtime_environment": binding(sys.argv[11]),
        "scheduler_runtime_after_smoke": binding(sys.argv[12]),
        "scheduler_runtime_after_cell": binding(sys.argv[13]),
        "server_log": binding(sys.argv[14]),
    },
    "cache_state_contract": result["cache_state_contract"],
}
if output_path.exists():
    raise FileExistsError(f"refusing to overwrite cell evidence: {output_path}")
output_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  done
done

run_in_frozen_environment "${PYTHON_BIN}" -I - "${RUN_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

run_root = Path(sys.argv[1]).resolve()
rows = []
for path in sorted(run_root.glob("cycle-01-block-??/?/matrix_evidence.json")):
    rows.append(json.loads(path.read_text(encoding="utf-8")))
if len(rows) != 16:
    raise RuntimeError(f"strict Williams matrix requires 16 cell rows, found {len(rows)}")
provenance = rows[0].get("provenance")
if not isinstance(provenance, dict) or any(row.get("provenance") != provenance for row in rows):
    raise RuntimeError("all matrix cells must have identical frozen provenance")
runtime_parameters = rows[0].get("runtime_parameters")
if not isinstance(runtime_parameters, dict) or any(
    row.get("runtime_parameters") != runtime_parameters for row in rows
):
    raise RuntimeError("all matrix cells must have identical runtime parameters")
output = run_root / "matrix_index.json"
if output.exists():
    raise FileExistsError(f"refusing to overwrite matrix index: {output}")
output.write_text(
    json.dumps(
        {
            "provenance": provenance,
            "runtime_parameters": runtime_parameters,
            "cell_evidence": rows,
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

trap - EXIT
echo "Completed strict ${ROLE} Williams matrix under ${RUN_ROOT}"
