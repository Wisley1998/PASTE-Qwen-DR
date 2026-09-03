#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PHASE="${1:-all}"
RESULTS_ROOT="${2:-$ROOT/results/use_case_$(date -u +%Y%m%d_%H%M%S)}"

case "$PHASE" in
  convergence|reaction|all) ;;
  *)
    echo "Usage: $0 [convergence|reaction|all] [results_root]" >&2
    exit 1
    ;;
esac

mkdir -p "$RESULTS_ROOT"

newest_history() {
  local dir="$1"
  python3 - "$dir" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
candidates = sorted(path.glob("*/history.json"))
if not candidates:
    raise SystemExit(1)
print(max(candidates, key=lambda item: item.stat().st_mtime).resolve())
PY
}

run_one() {
  local template="$1"
  local phase="$2"
  local mode="$3"
  local perturbation="${4:-}"
  local target_dir="$RESULTS_ROOT/$phase/$mode"
  local tmp_config
  tmp_config="$(mktemp)"
  mkdir -p "$target_dir"
  echo "[run] phase=$phase mode=$mode results=$target_dir"

  python3 - "$template" "$tmp_config" "$target_dir" "$perturbation" <<'PY'
import json
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
target_dir = Path(sys.argv[3])
perturbation = sys.argv[4]

with template_path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)

data["results_dir"] = str(target_dir)
if perturbation:
    data["perturbation_min_granularity_ns"] = int(perturbation)

with output_path.open("w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
PY

  python3 -m barebones_optimizer.main --config "$tmp_config"
  rm -f "$tmp_config"
}

if [[ "$PHASE" == "convergence" || "$PHASE" == "all" ]]; then
  run_one "$ROOT/config/use_case/actor_only_convergence.json" convergence actor_only
  run_one "$ROOT/config/use_case/speculator_only_convergence.json" convergence speculator_only
  run_one "$ROOT/config/use_case/dual_convergence.json" convergence dual
fi

if [[ "$PHASE" == "reaction" || "$PHASE" == "all" ]]; then
  CALIBRATION_PATH="$RESULTS_ROOT/reaction_calibration.json"
  python3 "$ROOT/scripts/calibrate_reaction_baseline.py" \
    --config-template "$ROOT/config/use_case/dual_reaction.json" \
    --output "$CALIBRATION_PATH"
  PERTURBATION_NS="$(python3 - "$CALIBRATION_PATH" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(payload['selected_min_granularity_ns'])
PY
)"

  run_one "$ROOT/config/use_case/actor_only_reaction.json" reaction actor_only "$PERTURBATION_NS"
  run_one "$ROOT/config/use_case/speculator_only_reaction.json" reaction speculator_only "$PERTURBATION_NS"
  run_one "$ROOT/config/use_case/dual_reaction.json" reaction dual "$PERTURBATION_NS"

  DUAL_HISTORY="$(newest_history "$RESULTS_ROOT/reaction/dual")"
  python3 "$ROOT/scripts/build_paper_reaction_controls.py" \
    --dual-history "$DUAL_HISTORY" \
    --output-root "$RESULTS_ROOT/paper_reaction"
fi

manifest_path="$RESULTS_ROOT/manifest.json"

python3 - "$RESULTS_ROOT" "$manifest_path" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

results_root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()

def newest_history(path: Path):
    candidates = sorted(path.glob("*/history.json"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item.stat().st_mtime).resolve())

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "results_root": str(results_root),
    "convergence": {
        "actor_only": newest_history(results_root / "convergence" / "actor_only"),
        "speculator_only": newest_history(results_root / "convergence" / "speculator_only"),
        "dual": newest_history(results_root / "convergence" / "dual"),
    },
    "reaction": {
        "actor_only": newest_history(results_root / "reaction" / "actor_only"),
        "speculator_only": newest_history(results_root / "reaction" / "speculator_only"),
        "dual": newest_history(results_root / "reaction" / "dual"),
    },
    "paper_reaction": {
        "untuned": newest_history(results_root / "paper_reaction" / "untuned"),
        "actor_only": newest_history(results_root / "paper_reaction" / "actor_only"),
        "dual": newest_history(results_root / "reaction" / "dual"),
    },
    "reaction_calibration": str((results_root / "reaction_calibration.json").resolve()) if (results_root / "reaction_calibration.json").exists() else None,
}

manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(manifest_path)
PY

python3 "$ROOT/scripts/plot_use_case.py" --manifest "$manifest_path"

echo "Manifest written to $manifest_path"
