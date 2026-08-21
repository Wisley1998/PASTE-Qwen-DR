#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: setup_env.sh

Create the reproduction environment and install the repository's pinned
requirements in one pip transaction.

Environment overrides:
  PASTE_ENV_PREFIX   Conda prefix (default: ~/.conda/envs/paste)
  CONDA_EXE          Conda executable (otherwise resolved from PATH)
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    echo "error: setup_env.sh does not accept positional arguments" >&2
    usage >&2
    exit 2
    ;;
esac
if (( $# > 1 )); then
  echo "error: setup_env.sh does not accept positional arguments" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements.txt"
ENV_PREFIX="${PASTE_ENV_PREFIX:-${HOME}/.conda/envs/paste}"
MARKER_FILE="${ENV_PREFIX}/.paste-requirements.sha256"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "error: requirements file not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  echo "error: conda is required but was not found" >&2
  exit 1
fi

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "error: conda is not executable: ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "Creating Python 3.10 environment at ${ENV_PREFIX}"
  "${CONDA_BIN}" create --yes --prefix "${ENV_PREFIX}" python=3.10 pip
fi

ENV_PYTHON="${ENV_PREFIX}/bin/python"
if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "error: ${ENV_PREFIX} exists but does not contain an executable Python" >&2
  exit 1
fi

PYTHON_SERIES="$("${ENV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_SERIES}" != "3.10" ]]; then
  echo "error: expected Python 3.10 in ${ENV_PREFIX}, found ${PYTHON_SERIES}" >&2
  exit 1
fi

REQUIREMENTS_HASH="$("${ENV_PYTHON}" - "${REQUIREMENTS_FILE}" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

print(sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

requirements_are_exact() {
  "${ENV_PYTHON}" - "${REQUIREMENTS_FILE}" <<'PY'
from importlib import metadata
from pathlib import Path
import re
import sys


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


required: dict[str, tuple[str, str]] = {}
for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if "==" not in line:
        raise SystemExit(1)
    name, expected = line.split("==", 1)
    required[normalize(name.strip())] = (name.strip(), expected.strip())

installed = {
    normalize(dist.metadata["Name"]): dist.version
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
wrong = [
    name
    for key, (name, expected) in required.items()
    if installed.get(key) != expected
]
raise SystemExit(1 if wrong else 0)
PY
}

MARKER_HASH=""
if [[ -f "${MARKER_FILE}" ]]; then
  IFS= read -r MARKER_HASH < "${MARKER_FILE}" || true
fi

if [[ "${MARKER_HASH}" == "${REQUIREMENTS_HASH}" ]] && requirements_are_exact; then
  echo "Pinned requirements are already installed; nothing to do."
elif requirements_are_exact; then
  echo "Pinned requirements are already present; recording their fingerprint."
  printf '%s\n' "${REQUIREMENTS_HASH}" > "${MARKER_FILE}.tmp"
  mv -f -- "${MARKER_FILE}.tmp" "${MARKER_FILE}"
else
  echo "Installing ${REQUIREMENTS_FILE} in one pinned transaction."
  "${ENV_PYTHON}" -m pip install --requirement "${REQUIREMENTS_FILE}"
  "${ENV_PYTHON}" -m pip check
  if ! requirements_are_exact; then
    echo "error: installed packages do not exactly match requirements.txt" >&2
    exit 1
  fi
  printf '%s\n' "${REQUIREMENTS_HASH}" > "${MARKER_FILE}.tmp"
  mv -f -- "${MARKER_FILE}.tmp" "${MARKER_FILE}"
fi

"${ENV_PYTHON}" -m pip check
echo "Environment ready: ${ENV_PREFIX}"
