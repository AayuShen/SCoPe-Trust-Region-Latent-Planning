#!/usr/bin/env bash
# Phase 0: evaluate DINO-WM's released checkpoints with its own planning configs.
#   bash scripts/reproduce_dino_wm.sh [pusht|wall|point_maze] [n_evals] [hydra overrides...]
# Extra arguments are passed to plan.py, e.g. a smoke test that caps replanning:
#   bash scripts/reproduce_dino_wm.sh pusht 5 planner.max_iter=2
# Upstream sets planner.max_iter=null (replan until success, no cap), so full runs
# should be launched under `timeout`.
set -euo pipefail

ENV_NAME="${1:-pusht}"
N_EVALS="${2:-50}"
shift $(( $# < 2 ? $# : 2 ))
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${SCOPE_DATA_DIR:-$HOME/scope_data}"
OUT_DIR="$REPO_ROOT/outputs/runs/phase0/$ENV_NAME"
case "$ENV_NAME" in  # released checkpoint folder names
  wall) MODEL_NAME=wall_single ;;
  *)    MODEL_NAME="$ENV_NAME" ;;
esac
mkdir -p "$OUT_DIR"

export SDL_VIDEODRIVER=dummy  # Push-T renders with pygame; no display in WSL
# plan.py always calls wandb.init; even when disabled, wandb 0.13 inspects git, which
# refuses repos on /mnt/e ("dubious ownership"). Skip the git lookup entirely.
export WANDB_MODE="${WANDB_MODE:-disabled}" WANDB_DISABLE_GIT=true WANDB_PROGRAM_RELPATH=plan.py
# DINO-WM checkpoints pickle whole modules, which torch>=2.6 rejects by default
# (weights_only=True). Restore torch 2.3's behaviour for these runs only. This trusts
# the files: they are the authors' official OSF release, verified against its sha256.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# plan.py resolves env, dataset and planning modules relative to its own directory.
cd "$REPO_ROOT/external/dino_wm"
# Same as `python plan.py ...`, but with rollouts chunked to fit 8 GB of VRAM.
python "$REPO_ROOT/scripts/plan_dino_wm.py" --config-name "plan_${ENV_NAME}.yaml" \
  model_name="$MODEL_NAME" \
  ckpt_base_path="$DATA_DIR/checkpoints" \
  n_evals="$N_EVALS" \
  hydra.run.dir="$OUT_DIR/\${now:%Y%m%d_%H%M%S}" \
  "$@" \
  2>&1 | tee "$OUT_DIR/last_run.log"
