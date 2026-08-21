#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPRO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PASTE_ANALYSIS_PYTHON:-python3}"
OUTPUT_DIR="${PASTE_SPEC_TOOL_OUTPUT_DIR:-${REPRO_ROOT}/artifacts/speculative_tool_execution}"

mkdir -p -- "${OUTPUT_DIR}"
cd -- "${REPRO_ROOT}"

"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_mapper.py' -v
"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_tool_prediction.py' -v
"${PYTHON_BIN}" -m unittest discover -s tests -p 'test_scheduler.py' -v
"${PYTHON_BIN}" -m paste_repro.cli run-speculative-tools \
  --top-k "${PASTE_SPEC_TOOL_TOP_K:-5}" \
  --max-concurrency "${PASTE_SPEC_TOOL_MAX_CONCURRENCY:-4}" \
  --model-out "${OUTPUT_DIR}/url_rank_mapper.json" \
  --report-out "${OUTPUT_DIR}/result.json"

echo "Speculative-tool artifacts: ${OUTPUT_DIR}"
