#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

pytest "$ROOT/tests"
