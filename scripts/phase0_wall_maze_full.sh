#!/usr/bin/env bash
# Phase 0: full Wall and PointMaze evaluations, one after the other (two jobs at once would
# not fit the 8 GB GPU). 50 episodes, up to 3 MPC rounds, seed 99 (the upstream default).
#   bash scripts/phase0_wall_maze_full.sh [max_iter]
# Launch detached: powershell -File scripts\launch_detached.ps1 -Script scripts/phase0_wall_maze_full.sh
set -uo pipefail

MAX_ITER="${1:-3}"
cd "$(dirname "$0")/.."

for env in wall point_maze; do
  echo "[wall_maze] $(date +%T) starting $env"
  RUN_TAG="full_${env}" bash scripts/phase0_env_full.sh "$env" 50 "$MAX_ITER"
  echo "[wall_maze] $(date +%T) $env exited with code $?"
done
echo "[wall_maze] all done"
