#!/usr/bin/env bash
# Phase 0: repeat the full Push-T evaluation for several seeds, one after another (two
# jobs at once would not fit the 8 GB GPU). Each seed re-draws the 50 start/goal pairs.
#   bash scripts/phase0_pusht_seeds.sh [max_iter] seed [seed ...]
# Launch detached: powershell -File scripts\launch_detached.ps1 -Script scripts/phase0_pusht_seeds.sh -ScriptArgs "3 100 101"
set -uo pipefail

MAX_ITER="$1"
shift
cd "$(dirname "$0")/.."

for seed in "$@"; do
  echo "[seeds] $(date +%T) starting seed $seed"
  RUN_TAG="seed${seed}" bash scripts/phase0_pusht_full.sh 50 "$MAX_ITER" "seed=$seed"
  echo "[seeds] $(date +%T) seed $seed exited with code $?"
done
echo "[seeds] all done"
