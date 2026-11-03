#!/usr/bin/env bash
# Phase 0, full Push-T evaluation: upstream plan_pusht.yaml (MPC-CEM, horizon 5, 300
# samples, top-30, 30 CEM steps, 5 actions per round), 50 episodes, seed 99, capped at 5
# MPC rounds. Solved episodes are skipped in later rounds (SCOPE_SKIP_SOLVED=1).
#
# Meant to be launched detached from Windows (scripts/launch_detached.ps1): WSL stops the
# whole VM, and every process in it, as soon as the last wsl.exe session ends, so this
# script must be the foreground process of a session that stays open.
#   bash scripts/phase0_pusht_full.sh [n_evals] [max_iter] [hydra overrides...]
set -euo pipefail

N_EVALS="${1:-50}"
MAX_ITER="${2:-5}"
shift $(( $# < 2 ? $# : 2 ))

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate scope
export SCOPE_SKIP_SOLVED=1 SCOPE_PRECISION="${SCOPE_PRECISION:-tf32}"

LOG_DIR="$REPO_ROOT/outputs/runs/phase0/pusht"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_TAG:-full}_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"

cd "$REPO_ROOT"
set +e
timeout 10h bash scripts/reproduce_dino_wm.sh pusht "$N_EVALS" "planner.max_iter=$MAX_ITER" "$@" > "$LOG" 2>&1
code=$?
echo "[phase0] finished with exit code $code" >> "$LOG"
exit $code
