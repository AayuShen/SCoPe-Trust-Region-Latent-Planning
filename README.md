# scope-planning

Code for **SCoPe: Trust-Region Latent Planning with JEPA World Models**: latent MPC that rejects
imagined trajectories the world model should not be trusted on.

New to the project? Read the overview first: **[scope-project](https://github.com/AayuShen/scope-project)**.
The deck and speaker script are in [scope-mid-review](https://github.com/AayuShen/scope-mid-review).

## What is here

| Path | What it does |
|---|---|
| `src/scope/world_model/interface.py` | Backbone-agnostic world-model interface (DINO-WM now, V-JEPA 2-AC later) |
| `src/scope/trust/interface.py` | Interface every trust scorer implements |
| `src/scope/world_model/dino_wm_patches.py` | Runtime patches that fit DINO-WM planning in 8 GB and make it 3.8× faster without changing CEM's choices |
| `scripts/plan_dino_wm.py`, `scripts/reproduce_dino_wm.sh` | Run DINO-WM's CEM-MPC planner with the patches |
| `scripts/phase0_pusht_full.sh`, `phase0_pusht_seeds.sh`, `phase0_env_full.sh` | The Phase 0 reproduction runs (Push-T, Wall, PointMaze) |
| `scripts/h1_pilot.py`, `scripts/run_h1_pilot.sh` | H1 pilot: logs imagined vs real latents for CEM's candidates and scores untrained trust signals against the real error |
| `scripts/benchmark_dino_wm.py` | Speed and fidelity benchmark of the patches |
| `scripts/summarize_phase0.py` | Success per MPC round from the run logs |
| `scripts/launch_detached.ps1` | Starts a long WSL job that survives the launching shell (Windows host) |
| `external/dino_wm` | The DINO-WM authors' code, pinned at commit 0a9492f (git submodule, MIT licence) |
| `docs/setup_and_downloads.md` | Datasets and checkpoints: what to download and where it goes |

Datasets, checkpoints and run outputs are not in git (see `.gitignore`).

## Setup (WSL2 Ubuntu, NVIDIA GPU)

```bash
git clone --recurse-submodules https://github.com/AayuShen/scope-planning.git
cd scope-planning
bash scripts/setup_wsl.sh            # conda env "scope" (PyTorch cu128)
conda activate scope
bash scripts/download_dino_wm_assets.sh
bash scripts/reproduce_dino_wm.sh pusht 5   # 5-episode smoke test, about 4 minutes
```

Full runs and the pilot:

```bash
bash scripts/phase0_pusht_full.sh                  # Push-T, 50 episodes, seed 99
bash scripts/phase0_env_full.sh wall 50 3 seed=100 # Wall, 50 episodes, up to 3 MPC rounds
RUN_TAG=pilot40 bash scripts/run_h1_pilot.sh --episodes 40 --seed 1
```

Results and how they were obtained: [scope-project/docs](https://github.com/AayuShen/scope-project/tree/main/docs).
