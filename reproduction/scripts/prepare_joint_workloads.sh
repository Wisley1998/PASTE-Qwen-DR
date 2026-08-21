#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REPRO_ROOT="${REPO_ROOT}/reproduction"
ENV_PREFIX="${PASTE_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
MODEL_REVISION="${MODEL_REVISION:-4b0ac5767427a55d08a254f0367e2934976598e0}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-${HF_HOME}/models--Alibaba-NLP--Tongyi-DeepResearch-30B-A3B/snapshots/${MODEL_REVISION}}"
MAPPER="${PASTE_TOOL_MODEL:-${REPRO_ROOT}/results/tool_only/url_rank_mapper.json}"
EVAL_LIMIT="${PASTE_EVAL_SESSION_LIMIT:-30}"
SPEEDUP="${PASTE_TRACE_SPEEDUP:-10}"
MAX_MODEL_LEN="${PASTE_MAX_MODEL_LEN:-16384}"
MAX_OUTPUT_TOKENS_CAP="${PASTE_MAX_OUTPUT_TOKENS_CAP:-128}"
OUTPUT_TOKEN_BUFFER="${PASTE_OUTPUT_TOKEN_BUFFER:-8}"
MIN_OUTPUT_TOKENS_FLOOR="${PASTE_MIN_OUTPUT_TOKENS_FLOOR:-64}"
ARTIFACT_ROOT="${PASTE_ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}"
SPLIT_JSON="${ARTIFACT_ROOT}/materialized_split.json"
CALIBRATION_DIR="${PASTE_CALIBRATION_DIR:-${ARTIFACT_ROOT}/workloads/calibration_learned}"
EVAL_DIR="${PASTE_EVAL_WORKLOAD_DIR:-${ARTIFACT_ROOT}/workloads/eval_learned}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: reproduction environment is missing: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_SNAPSHOT}" ]]; then
  echo "error: pinned model snapshot is missing: ${MODEL_SNAPSHOT}" >&2
  exit 1
fi
if [[ ! -f "${MAPPER}" ]]; then
  echo "error: learned tool mapper is missing: ${MAPPER}" >&2
  echo "Run reproduction/scripts/run_tool_only.sh first." >&2
  exit 1
fi
if [[ ! "${EVAL_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "error: PASTE_EVAL_SESSION_LIMIT must be a non-negative integer" >&2
  exit 1
fi

mkdir -p -- "${ARTIFACT_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/materialize_trace_split.py" \
  --artifact "${MAPPER}" \
  --held-out-limit "${EVAL_LIMIT}" > "${SPLIT_JSON}"

readarray -t SPLIT_VALUES < <(
  "${PYTHON_BIN}" - "${SPLIT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["train_directory"])
print(payload["held_out_directory"])
print(len(payload["train_sessions"]))
print(len(payload["held_out_sessions"]))
PY
)
TRAIN_TRACE_DIR="${SPLIT_VALUES[0]}"
EVAL_TRACE_DIR="${SPLIT_VALUES[1]}"
TRAIN_COUNT="${SPLIT_VALUES[2]}"
EVAL_COUNT="${SPLIT_VALUES[3]}"

COMMON=(
  --tokenizer "${MODEL_SNAPSHOT}"
  --speedup "${SPEEDUP}"
  --prepare-only
  --max-model-len "${MAX_MODEL_LEN}"
  --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}"
  --output-token-buffer "${OUTPUT_TOKEN_BUFFER}"
  --min-output-tokens-floor "${MIN_OUTPUT_TOKENS_FLOOR}"
  --tool-overlap-mode learned
  --tool-prediction-model "${MAPPER}"
  --tool-prediction-top-k 5
  --tool-overlap-efficiency 1.0
  --temperature 0
  --top-p 1
  --presence-penalty 0
  --seed 20260417
)

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_vllm_trace_experiment.py" \
  --trace-dir "${TRAIN_TRACE_DIR}" \
  --trace-count "${TRAIN_COUNT}" \
  --output-dir "${CALIBRATION_DIR}" \
  "${COMMON[@]}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_vllm_trace_experiment.py" \
  --trace-dir "${EVAL_TRACE_DIR}" \
  --trace-count "${EVAL_COUNT}" \
  --output-dir "${EVAL_DIR}" \
  "${COMMON[@]}"

echo "Calibration workload: ${CALIBRATION_DIR}/prepared_workload.json"
echo "Held-out workload:    ${EVAL_DIR}/prepared_workload.json"
