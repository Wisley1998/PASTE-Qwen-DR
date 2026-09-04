#!/bin/bash
set -Eeuo pipefail

usage() {
  echo "Usage: run_pattern_v2_strict_abef_matrix.sh {validate|pilot|formal}" >&2
}

if (( $# != 1 )) || [[ "$1" != "validate" && "$1" != "pilot" && "$1" != "formal" ]]; then
  usage
  exit 2
fi
MODE="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd -- "${REPO_ROOT}"

CONFIG="${PASTE_PATTERN_V2_CONFIG:-${REPO_ROOT}/reproduction/configs/pattern_v2_strict_abef_c16.env}"
PLAN_DIR="${PASTE_PATTERN_V2_PLAN_DIR:-${REPO_ROOT}/reproduction/artifacts/pattern_v2_strict_deployable_final30_20260904_v2}"
RUN_ROOT="${PASTE_PATTERN_V2_RUN_ROOT:-${REPO_ROOT}/reproduction/results/pattern_v2_strict_live_deployable_final30_c16_${MODE}_20260904}"
RUNNER="${SCRIPT_DIR}/run_pattern_v2_strict_abef.py"
START_SCRIPT="${SCRIPT_DIR}/start_vllm.sh"
STOP_SCRIPT="${SCRIPT_DIR}/stop_vllm.sh"
SMOKE_SCRIPT="${SCRIPT_DIR}/smoke_vllm.py"
SCHEDULER_HOOK="${REPO_ROOT}/scripts/pythonhooks/sched_policy_patch.py"
SITECUSTOMIZE="${REPO_ROOT}/scripts/pythonhooks/sitecustomize.py"
PYTHON_BIN="/home/aiscuser/.conda/envs/paste/bin/python"

PUBLIC_PLAN="${PLAN_DIR}/public_plan.json"
SEALED_PLAN="${PLAN_DIR}/sealed_plan.json"
PREDICTOR="${PLAN_DIR}/pattern_v2_predictor.json"
DURATION="${PLAN_DIR}/public_duration_predictor.json"
CLOCK="${PLAN_DIR}/private_service_clock.json"
TAIL="${PLAN_DIR}/tail_predictor.json"
PLAN_MANIFEST="${PLAN_DIR}/manifest.json"

FROZEN_INPUTS=(
  "${CONFIG}" "${PUBLIC_PLAN}" "${SEALED_PLAN}" "${PREDICTOR}"
  "${DURATION}" "${CLOCK}" "${TAIL}" "${PLAN_MANIFEST}"
  "${RUNNER}" "${START_SCRIPT}" "${STOP_SCRIPT}" "${SMOKE_SCRIPT}"
  "${SCHEDULER_HOOK}" "${SITECUSTOMIZE}"
  "${REPO_ROOT}/reproduction/paste_repro/pattern_v2_all_visit_online.py"
  "${REPO_ROOT}/reproduction/paste_repro/pattern_v2_strict_adapter.py"
  "${REPO_ROOT}/reproduction/paste_repro/strict_trace_runtime.py"
  "${REPO_ROOT}/reproduction/paste_repro/trace_coscheduler.py"
)
for path in "${FROZEN_INPUTS[@]}"; do
  [[ -f "${path}" ]] || { echo "error: missing frozen input: ${path}" >&2; exit 1; }
done
[[ -x "${PYTHON_BIN}" ]] || { echo "error: missing Python: ${PYTHON_BIN}" >&2; exit 1; }

"${PYTHON_BIN}" -I - "${SCRIPT_DIR}" "${CONFIG}" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from run_strict_trace_abef import _formal_config_exports, _validate_formal_environment_contract
exports = _formal_config_exports(Path(sys.argv[2]).resolve())
_validate_formal_environment_contract(exports)
PY

# The preceding parser proves this file contains literal exports only.
# shellcheck source=/dev/null
source "${CONFIG}"
if [[ "${PASTE_STRICT_SESSIONS}" != "210" \
   || "${PASTE_MAX_ACTIVE_TASKS}" != "16" \
   || "${PASTE_VISIT_CAPACITY}" != "64" \
   || "${PASTE_SPECULATIVE_CAP}" != "64" ]]; then
  echo "error: Pattern V2 matrix requires tasks=210, C=16, Visit=64, Spec=64" >&2
  exit 1
fi

mapfile -t CONFIG_EXPORT_NAMES < <(
  sed -nE 's/^export ([A-Z][A-Z0-9_]*)=.*/\1/p' "${CONFIG}"
)
FROZEN_ENV=(
  "HOME=${PASTE_RUNTIME_HOME}"
  "PATH=${PASTE_RUNTIME_PATH}"
  "LD_LIBRARY_PATH=${PASTE_RUNTIME_LD_LIBRARY_PATH}"
  "TMPDIR=${PASTE_RUNTIME_TMPDIR}"
  "LANG=${PASTE_RUNTIME_LANG}"
  "LC_ALL=${PASTE_RUNTIME_LANG}"
  "TZ=${PASTE_RUNTIME_TZ}"
)
for name in "${CONFIG_EXPORT_NAMES[@]}"; do
  FROZEN_ENV+=("${name}=${!name}")
done
run_frozen() {
  /usr/bin/env -i "${FROZEN_ENV[@]}" "$@"
}

declare -A INPUT_SHA256=()
for path in "${FROZEN_INPUTS[@]}"; do
  INPUT_SHA256["${path}"]="$(sha256sum -- "${path}" | awk '{print $1}')"
done
verify_frozen_inputs() {
  local path current
  for path in "${FROZEN_INPUTS[@]}"; do
    current="$(sha256sum -- "${path}" | awk '{print $1}')"
    [[ "${current}" == "${INPUT_SHA256[${path}]}" ]] || {
      echo "error: frozen input changed during matrix: ${path}" >&2
      return 1
    }
  done
}

run_frozen "${PYTHON_BIN}" -I - \
  "${SCRIPT_DIR}" "${PUBLIC_PLAN}" "${SEALED_PLAN}" "${PREDICTOR}" \
  "${DURATION}" "${CLOCK}" "${TAIL}" <<'PY'
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, sys.argv[1])
import run_pattern_v2_strict_abef as runner
args = SimpleNamespace(
    public_plan=Path(sys.argv[2]), sealed_plan=Path(sys.argv[3]),
    predictor_artifact=Path(sys.argv[4]), duration_artifact=Path(sys.argv[5]),
    service_clock_artifact=Path(sys.argv[6]), tail_artifact=Path(sys.argv[7]),
    allow_smoke_workload=False, cell="F",
)
loaded = runner.load_runtime_inputs(args)
assert loaded.formal_workload and loaded.workload_contract == (
    "retrospective_internal_holdout_30_roots_x7"
)
print(f"validated plan: {len(loaded.public['traces'])} tasks, {loaded.workload_contract}")
PY

IFS=';' read -r -a GPU_GROUPS <<< "${PASTE_GPU_GROUPS}"
if (( ${#GPU_GROUPS[@]} != 2 )); then
  echo "error: the frozen matrix requires exactly two GPU groups" >&2
  exit 1
fi
ORDERS=("A B F E" "B E A F" "E F B A" "F A E B")
BLOCK_COUNT=0
case "${MODE}" in
  validate) BLOCK_COUNT=0 ;;
  pilot) BLOCK_COUNT=1 ;;
  formal) BLOCK_COUNT=4 ;;
esac
for ((block=0; block < (BLOCK_COUNT == 0 ? 4 : BLOCK_COUNT); block++)); do
  echo "block-$((block + 1)) gpu=${GPU_GROUPS[block % 2]} order=${ORDERS[block]}"
done
if [[ "${MODE}" == "validate" ]]; then
  exit 0
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "error: refusing to reuse run root: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p -- "${RUN_ROOT}"
(set -o noclobber; printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_ROOT}/${MODE^^}_STARTED")

protected_start_tick() {
  local line tail
  local -a fields=()
  [[ -r "/proc/${PASTE_PROTECTED_PID}/stat" ]] || return 1
  IFS= read -r line < "/proc/${PASTE_PROTECTED_PID}/stat"
  tail="${line##*) }"
  read -r -a fields <<< "${tail}"
  printf '%s\n' "${fields[19]}"
}
PROTECTED_START_TICK="$(protected_start_tick)" || {
  echo "error: protected PID ${PASTE_PROTECTED_PID} is absent" >&2
  exit 1
}
PROTECTED_COMMAND="$(tr '\0' ' ' < "/proc/${PASTE_PROTECTED_PID}/cmdline")"
[[ "${PROTECTED_COMMAND,,}" == *resnet* ]] || {
  echo "error: protected PID is not ResNet" >&2
  exit 1
}

CURRENT_DYNAMIC_ENV=()
SERVER_MANAGED=0
cleanup() {
  if (( SERVER_MANAGED == 1 )); then
    run_frozen "${CURRENT_DYNAMIC_ENV[@]}" "${STOP_SCRIPT}" || true
  fi
}
trap cleanup EXIT

for ((block=0; block < BLOCK_COUNT; block++)); do
  block_id="block-$(printf '%02d' "$((block + 1))")"
  gpu_group="${GPU_GROUPS[block % 2]}"
  read -r -a cells <<< "${ORDERS[block]}"
  for position in "${!cells[@]}"; do
    cell="${cells[position]}"
    case "${cell}" in
      A|B) policy="fcfs" ;;
      E|F) policy="online_joint_pacer_v2" ;;
    esac
    cell_root="${RUN_ROOT}/${block_id}/${cell}"
    state_dir="${cell_root}/state"
    log_dir="${cell_root}/server"
    safe_dir="${cell_root}/empty_python_cwd"
    hook_dir="${cell_root}/runtime_python_hook"
    mkdir -p -- "${state_dir}" "${log_dir}"
    mkdir -- "${safe_dir}" "${hook_dir}"
    ln -s -- "${SITECUSTOMIZE}" "${hook_dir}/sitecustomize.py"
    ln -s -- "${SCHEDULER_HOOK}" "${hook_dir}/sched_policy_patch.py"
    chmod 0500 -- "${safe_dir}" "${hook_dir}"
    CURRENT_DYNAMIC_ENV=(
      "CUDA_VISIBLE_DEVICES=${gpu_group}"
      "VLLM_SCHED_POLICY=${policy}"
      "VLLM_STATE_DIR=${state_dir}"
      "VLLM_LOG_DIR=${log_dir}"
      "VLLM_RUNTIME_HOOK_DIR=${hook_dir}"
      "VLLM_SAFE_WORKING_DIR=${safe_dir}"
    )
    verify_frozen_inputs
    [[ "$(protected_start_tick)" == "${PROTECTED_START_TICK}" ]] || {
      echo "error: protected PID identity changed" >&2
      exit 1
    }
    run_frozen nvidia-smi \
      --query-gpu=timestamp,index,uuid,memory.total,memory.used,utilization.gpu \
      --format=csv > "${cell_root}/nvidia_smi_pre.csv"
    run_frozen "${CURRENT_DYNAMIC_ENV[@]}" "${START_SCRIPT}"
    SERVER_MANAGED=1
    server_pid="$(< "${state_dir}/vllm_${VLLM_PORT}.pid")"
    run_frozen "${PYTHON_BIN}" -I "${SMOKE_SCRIPT}" \
      --base-url "http://${VLLM_PROBE_HOST}:${VLLM_PORT}" --max-tokens 64 \
      > "${cell_root}/standardized_smoke.json"
    runtime_marker="${state_dir}/vllm_${VLLM_PORT}.scheduler_runtime.json"
    run_frozen "${PYTHON_BIN}" -I - \
      "${runtime_marker}" "${policy}" "${server_pid}" "${SCHEDULER_HOOK}" \
      "${safe_dir}" <<'PY'
import hashlib, json
from pathlib import Path
import sys
marker_path, policy, server_pid, hook_path, safe_dir = sys.argv[1:]
server_pid = int(server_pid)
expected = policy == "online_joint_pacer_v2"
path = Path(marker_path)
if not expected:
    if path.exists():
        raise RuntimeError("FCFS unexpectedly emitted scheduler runtime evidence")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
required = {
    "schema": "paste.vllm.scheduler_runtime_use.v1",
    "policy": policy,
    "scheduler_api": "v1.Scheduler.schedule",
    "scheduler_hook_path": str(Path(hook_path).resolve()),
    "scheduler_hook_sha256": hashlib.sha256(Path(hook_path).read_bytes()).hexdigest(),
    "safe_working_directory": str(Path(safe_dir).resolve()),
    "python_safe_path_enforced": True,
    "cwd_import_filter_enforced": True,
    "working_directory_importable": False,
}
for key, value in required.items():
    if payload.get(key) != value:
        raise RuntimeError(f"scheduler evidence mismatch for {key}")
pid = int(payload["pid"])
seen = set()
while pid > 1 and pid not in seen and pid != server_pid:
    seen.add(pid)
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    pid = int(raw.rsplit(") ", 1)[1].split()[1])
if pid != server_pid:
    raise RuntimeError("scheduler marker is not from a server descendant")
PY
    run_frozen "${PYTHON_BIN}" -I "${RUNNER}" \
      --public-plan "${PUBLIC_PLAN}" \
      --sealed-plan "${SEALED_PLAN}" \
      --predictor-artifact "${PREDICTOR}" \
      --duration-artifact "${DURATION}" \
      --service-clock-artifact "${CLOCK}" \
      --tail-artifact "${TAIL}" \
      --cell "${cell}" \
      --output-dir "${cell_root}/client" \
      --server-url "http://${VLLM_PROBE_HOST}:${VLLM_PORT}" \
      --server-policy-file "${state_dir}/vllm_${VLLM_PORT}.policy" \
      --model "${MODEL_ID}" \
      --max-active-tasks "${PASTE_MAX_ACTIVE_TASKS}" \
      --visit-capacity "${PASTE_VISIT_CAPACITY}" \
      --speculative-cap "${PASTE_SPECULATIVE_CAP}" \
      --default-predicted-output-tokens "${PASTE_DEFAULT_PREDICTED_OUTPUT_TOKENS}" \
      --request-timeout-s "${PASTE_REQUEST_TIMEOUT_S}"
    verify_frozen_inputs
    run_frozen "${CURRENT_DYNAMIC_ENV[@]}" "${STOP_SCRIPT}"
    SERVER_MANAGED=0
    [[ "$(protected_start_tick)" == "${PROTECTED_START_TICK}" ]] || {
      echo "error: protected PID identity changed" >&2
      exit 1
    }
    run_frozen nvidia-smi \
      --query-gpu=timestamp,index,uuid,memory.total,memory.used,utilization.gpu \
      --format=csv > "${cell_root}/nvidia_smi_post.csv"
    {
      printf 'schema=paste.pattern_v2.matrix_cell_evidence.v1\n'
      printf 'mode=%s\nblock=%s\norder_position=%s\ncell=%s\n' \
        "${MODE}" "${block_id}" "$((position + 1))" "${cell}"
      printf 'gpu_ids=%s\nserver_pid=%s\npolicy=%s\n' \
        "${gpu_group}" "${server_pid}" "${policy}"
      printf 'protected_pid=%s\nprotected_start_tick=%s\n' \
        "${PASTE_PROTECTED_PID}" "${PROTECTED_START_TICK}"
      for path in "${FROZEN_INPUTS[@]}"; do
        printf 'input_sha256=%s  %s\n' "${INPUT_SHA256[${path}]}" "${path}"
      done
      find "${cell_root}/client" -maxdepth 1 -type f -print0 \
        | sort -z | xargs -0 sha256sum
      sha256sum "${cell_root}/standardized_smoke.json" \
        "${cell_root}/nvidia_smi_pre.csv" "${cell_root}/nvidia_smi_post.csv" \
        "${log_dir}/vllm_${VLLM_PORT}.log"
      if [[ -f "${runtime_marker}" ]]; then sha256sum "${runtime_marker}"; fi
    } > "${cell_root}/cell_evidence.txt"
  done
done

run_frozen "${PYTHON_BIN}" -I - "${RUN_ROOT}" "${MODE}" "${BLOCK_COUNT}" <<'PY'
import hashlib, json
from pathlib import Path
import sys
root, mode, blocks = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3])
rows = []
for summary_path in sorted(root.glob("block-??/?/client/summary.json")):
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = summary_path.parents[1] / "cell_evidence.txt"
    rows.append({
        "block": summary_path.parents[2].name,
        "cell": payload["cell"],
        "result_sha256": payload["result_sha256"],
        "summary_file_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "cell_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    })
expected = blocks * 4
if len(rows) != expected:
    raise RuntimeError(f"matrix expected {expected} cells, found {len(rows)}")
result = {"schema": "paste.pattern_v2.matrix_index.v1", "mode": mode, "cells": rows}
encoded = json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
result["matrix_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
(root / "matrix_index.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_ROOT}/${MODE^^}_COMPLETED"
echo "completed Pattern V2 ${MODE} matrix: ${RUN_ROOT}"
