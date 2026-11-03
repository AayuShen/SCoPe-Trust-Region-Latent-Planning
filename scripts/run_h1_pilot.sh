#!/usr/bin/env bash
# H1 pilot on Push-T (see scripts/h1_pilot.py). Extra arguments go to the pilot, e.g.
#   bash scripts/run_h1_pilot.sh --episodes 2 --record 1:8,10:8 --mc 2 --bank-trajs 20   # smoke test
# Like the Phase 0 runners, launch long runs with scripts/launch_detached.ps1.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate scope
export SDL_VIDEODRIVER=dummy WANDB_MODE=disabled TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1
LOG_DIR="$REPO_ROOT/outputs/runs/h1_pilot"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_TAG:-h1}_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"
cd "$REPO_ROOT/external/dino_wm"
set +e
python "$REPO_ROOT/scripts/h1_pilot.py" --out "$LOG_DIR/${RUN_TAG:-h1}" "$@" > "$LOG" 2>&1
code=$?
echo "[h1] finished with exit code $code" >> "$LOG"
exit $code
