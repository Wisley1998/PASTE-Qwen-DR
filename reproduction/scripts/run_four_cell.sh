#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF' >&2
Usage: run_four_cell.sh {tuning|final|heldout|stress}

Run fresh-server cells from the checksummed fixed workload manifest.
PASTE_CELLS may select a comma-separated subset of:
  fcfs_none,fcfs_learned,joint_none,joint_learned

Set PASTE_VALIDATE_ONLY=1 to validate the manifest and requested cells without
creating output directories or starting a server.
EOF
}

if (( $# != 1 )) || [[ "$1" != "tuning" && "$1" != "final" && "$1" != "heldout" && "$1" != "stress" ]]; then
  usage
  exit 2
fi
ROLE="$1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPRO_ROOT="${REPO_ROOT}/reproduction"
ENV_PREFIX="${PASTE_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
MANIFEST="${PASTE_FIXED_WORKLOAD_MANIFEST:-${REPRO_ROOT}/artifacts/workloads/fixed_three_way/manifest.json}"
RUN_ROOT="${PASTE_RUN_ROOT:-${REPRO_ROOT}/artifacts/four_cell_${ROLE}}"
RUN_PREFIX="${PASTE_RUN_PREFIX:-${ROLE}}"
CELL_LIST="${PASTE_CELLS:-fcfs_none,fcfs_learned,joint_none,joint_learned}"
VALIDATE_ONLY="${PASTE_VALIDATE_ONLY:-0}"

if [[ "${VALIDATE_ONLY}" != "0" && "${VALIDATE_ONLY}" != "1" ]]; then
  echo "error: PASTE_VALIDATE_ONLY must be 0 or 1" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: reproduction environment is missing: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: fixed workload manifest is missing: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! "${RUN_PREFIX}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: PASTE_RUN_PREFIX contains unsupported characters" >&2
  exit 2
fi
CELL_OUTPUT="$(
  "${PYTHON_BIN}" - "${MANIFEST}" "${ROLE}" "${CELL_LIST}" "${SCRIPT_DIR}" <<'PY'
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1]).resolve()
role = sys.argv[2]
requested = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
sys.path.insert(0, sys.argv[4])
from summarize_four_cell import CELL_SPECS, load_fixed_manifest

allowed = ("fcfs_none", "fcfs_learned", "joint_none", "joint_learned")
if not requested or len(requested) != len(set(requested)):
    raise SystemExit("PASTE_CELLS must be a non-empty list without duplicates")
unknown = sorted(set(requested) - set(allowed))
if unknown:
    raise SystemExit(f"unsupported PASTE_CELLS entries: {unknown}")
verified = load_fixed_manifest(manifest_path, role)
cell_by_name = {spec["name"]: cell for cell, spec in CELL_SPECS.items()}
mapper = verified["mapper_artifact"]
for name in requested:
    binding = verified["bindings"][cell_by_name[name]]
    evaluation = binding["evaluation_workload"]
    calibration = binding["calibration_workload"]
    for required in (evaluation, calibration, mapper):
        if not required.is_file():
            raise SystemExit(f"fixed cell artifact is missing: {required}")
    print("\t".join((
        name,
        binding["policy"],
        binding["tool_overlap_mode"],
        str(evaluation),
        str(calibration),
        str(mapper),
        str(binding["trace_count"]),
        str(binding["speedup"]),
        str(binding["tool_prediction_top_k"] or 0),
    )))
PY
)" || {
  echo "error: fixed workload manifest validation failed" >&2
  exit 1
}
if [[ -z "${CELL_OUTPUT}" ]]; then
  echo "error: fixed workload manifest produced no runnable cells" >&2
  exit 1
fi
readarray -t CELL_ROWS <<< "${CELL_OUTPUT}"
unset CELL_OUTPUT

if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  echo "Validated fixed ${ROLE} manifest and ${#CELL_ROWS[@]} requested cell(s)."
  exit 0
fi
if curl --fail --silent --max-time 2 "${PASTE_SERVER_URL:-http://127.0.0.1:8000}/health" >/dev/null 2>&1; then
  echo "error: a server is already running; stop it before isolated cells" >&2
  exit 1
fi

mkdir -p -- "${RUN_ROOT}"
SERVER_IS_MANAGED=0
cleanup() {
  if (( SERVER_IS_MANAGED == 1 )); then
    "${SCRIPT_DIR}/stop_vllm.sh" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for row in "${CELL_ROWS[@]}"; do
  IFS=$'\t' read -r cell_name policy overlap evaluation calibration mapper trace_count manifest_speedup manifest_top_k <<< "${row}"
  run_name="${RUN_PREFIX}_${cell_name}"
  export VLLM_SCHED_POLICY="${policy}"
  export VLLM_REQUIRE_NEW=1
  export VLLM_LOG_DIR="${RUN_ROOT}/${run_name}/server"
  export PASTE_VLLM_LOG_FILE="${VLLM_LOG_DIR}/vllm_${VLLM_PORT:-8000}.log"
  export PASTE_RUN_ROOT="${RUN_ROOT}"
  export PASTE_EVAL_WORKLOAD="${evaluation}"
  export PASTE_CALIBRATION_WORKLOAD="${calibration}"
  export PASTE_TOOL_MODEL="${mapper}"
  export PASTE_TOOL_OVERLAP_MODE="${overlap}"
  export PASTE_MAX_ACTIVE_TRACES="${PASTE_MAX_ACTIVE_TRACES:-${trace_count}}"
  export PASTE_TRACE_SPEEDUP="${PASTE_TRACE_SPEEDUP:-${manifest_speedup}}"
  if [[ "${overlap}" == "learned" ]]; then
    export PASTE_TOOL_PREDICTION_TOP_K="${PASTE_TOOL_PREDICTION_TOP_K:-${manifest_top_k}}"
  fi

  "${SCRIPT_DIR}/start_vllm.sh"
  SERVER_IS_MANAGED=1
  "${SCRIPT_DIR}/smoke_vllm.py" --max-tokens 64
  policy_label="fcfs"
  if [[ "${policy}" != "fcfs" ]]; then
    policy_label="joint"
  fi
  "${SCRIPT_DIR}/run_joint_cell.sh" "${policy_label}" "${run_name}"
  "${SCRIPT_DIR}/stop_vllm.sh"
  SERVER_IS_MANAGED=0
done

trap - EXIT
echo "Completed fixed ${ROLE} cells under ${RUN_ROOT}"
