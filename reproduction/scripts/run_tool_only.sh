#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPRO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PASTE_ANALYSIS_PYTHON:-python3}"
OUTPUT_DIR="${PASTE_TOOL_OUTPUT_DIR:-${REPRO_ROOT}/artifacts/tool_only}"

mkdir -p -- "${OUTPUT_DIR}"
cd -- "${REPRO_ROOT}"

"${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" -m paste_repro.cli analyze \
  --top-k "${PASTE_TOOL_TOP_K:-5}" \
  --model-out "${OUTPUT_DIR}/url_rank_mapper.json" \
  --report-out "${OUTPUT_DIR}/analysis.json"
"${PYTHON_BIN}" -m paste_repro.cli run-tool-only \
  --top-k "${PASTE_TOOL_TOP_K:-5}" \
  --max-concurrency "${PASTE_TOOL_MAX_CONCURRENCY:-4}" \
  --report-out "${OUTPUT_DIR}/replay.json"

echo "Tool-only artifacts: ${OUTPUT_DIR}"

