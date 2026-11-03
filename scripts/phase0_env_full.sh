#!/usr/bin/env bash
# Phase 0 evaluation of the released DINO-WM checkpoint on Wall or PointMaze, using the
# upstream planning config (plan_wall.yaml / plan_point_maze.yaml: MPC-CEM, horizon 5,
# 300 samples, top-30, 10 CEM steps, 5 actions per round) with the runtime patches of
# scripts/plan_dino_wm.py. Like phase0_pusht_full.sh, it must be the foreground process of
# a wsl.exe session that stays open (launch with scripts/launch_detached.ps1).
#
#   bash scripts/phase0_env_full.sh <wall|point_maze> [n_evals] [max_iter] [hydra overrides...]
#
# Smoke test (5 episodes, 1 round, 2 CEM steps):
#   bash scripts/phase0_env_full.sh wall 5 1 planner.sub_planner.opt_steps=2
set -euo pipefail

ENV_NAME="${1:?usage: phase0_env_full.sh <wall|point_maze> [n_evals] [max_iter] [overrides...]}"
case "$ENV_NAME" in wall|point_maze) ;; *) echo "env must be wall or point_maze" >&2; exit 2 ;; esac
N_EVALS="${2:-50}"
MAX_ITER="${3:-3}"
shift $(( $# < 3 ? $# : 3 ))

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate scope
export SCOPE_SKIP_SOLVED=1 SCOPE_PRECISION="${SCOPE_PRECISION:-tf32}" PYTHONUNBUFFERED=1

LOG_DIR="$REPO_ROOT/outputs/runs/phase0/$ENV_NAME"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_TAG:-$ENV_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"

cd "$REPO_ROOT"
set +e
timeout 10h bash scripts/reproduce_dino_wm.sh "$ENV_NAME" "$N_EVALS" "planner.max_iter=$MAX_ITER" "$@" > "$LOG" 2>&1
code=$?
echo "[phase0] finished with exit code $code" >> "$LOG"
exit $code
