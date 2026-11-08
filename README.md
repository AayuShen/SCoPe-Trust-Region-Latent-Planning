# SCoPe: Trust-Region Latent Planning with JEPA World Models

Source code for our 22AIE401 Reinforcement Learning term project
(Amrita School of Computing, Amritapuri).

**Team:** Aayush Shenday (AM.SC.U4AIE23004), Rajashekar Kasarla (AM.SC.U4AIE23041)
**Guide:** Dr. Geeta

---

## The problem

A robot can plan without trial and error by **imagining** what its actions will do. It uses a learned
**world model** to predict the future, tries many plans in its imagination, and carries out the one that
gets closest to the goal.

The catch: the planner **trusts every imagined step equally**. When the world model predicts something
wrong, the planner still believes it, picks a plan that only looks good in imagination, and then fails in
the real environment. Nothing in the planner asks *"can I actually believe this prediction?"*

We work with **DINO-WM**, a recent world model that plans from camera images towards a goal image. It is a
JEPA-style model: it predicts compact image features, not pixels. Like other world-model planners, it has
no reliability check.

## What we did

1. **Reproduced DINO-WM on a laptop GPU (8 GB).** We ran the authors' released model and planner on their
   three tasks (Push-T, Wall, PointMaze) and matched the published success rates.
2. **Made planning 3.8× faster**, with runtime patches that fit the model into 8 GB. We checked that the
   patches do not change any of the planner's decisions.
3. **Added a trust score for every imagined step.** We tested five ways to score how trustworthy an
   imagined step is, plus two controls. None of them needs any training:

   | Score | Idea |
   |---|---|
   | **MC-dropout** | Predict the same step 4 times with dropout on. If the answers disagree, the model is unsure. |
   | Action perturbation | Nudge the actions slightly (Gaussian noise 0.1). A big change in the prediction means it is fragile. |
   | Image perturbation | Add slight noise (0.1) to the start image. Same idea. |
   | Distance to training data | How far the imagined state is from 4,500 frames the model was trained on. |
   | Imagined cost | The planner's own score. Does it already reveal bad predictions? |
   | *Controls:* distance moved, random | To make sure a score is not winning for a trivial reason or by luck. |

4. **Checked every score against reality.** For each decision we recorded 192 of the plans the planner
   imagined. We then **reset the simulator to the same start state and ran each plan for real**, encoded
   what really happened, and measured the true error of every imagined step. That gives
   **80 decisions × 192 plans × 5 steps = 76,800 imagined steps**, each with a known true error.
5. **Tested a trust gate offline.** Using the recorded plans, we removed the 30% least-trusted ones before
   the planner chose, and checked whether the chosen plan then succeeded.

## Why it matters

- It shows that a **frozen world model already contains a usable reliability signal**. Its own dropout
  tells you where its imagination is wrong, without retraining anything.
- It shows that the **planner's own cost cannot tell** good predictions from bad ones (AUROC 0.51, the same
  as guessing). So a separate trust check is needed.
- It shows **where the check has to act**. Removing untrusted plans only at the final choice changes almost
  nothing, even with perfect knowledge of the error. The check has to steer the search itself, which is the
  next step.

## Results

**Planning success** (50 episodes per seed, up to 3 rounds of plan → act → look again):

| Task | Ours | Published DINO-WM |
|---|---|---|
| Push-T (3 seeds) | **0.90** | 0.90 |
| Wall (3 seeds) | **0.95** | 0.96 |
| PointMaze (1 seed) | **1.00** | 0.98 |

With a single plan and no replanning we get 0.87 / 0.52 / 0.86. Wall is lower than the published 0.74 on
every seed, and the cause is unknown. Rollouts take 6.9 s instead of 26.2 s per 300 plans (3.8× faster).

**Trust scores** (Push-T, 76,800 imagined steps). A step counts as "wrong" if its true error is in the top
10% for its step number. AUROC is computed per step number and averaged, so 0.5 means guessing and 1.0 is
perfect:

| Score | AUROC |
|---|---|
| **MC-dropout** | **0.86** (95% CI 0.84–0.88; 0.87 / 0.84 / 0.88 on the three seeds) |
| Distance moved (control) | 0.79 |
| Action perturbation | 0.66 |
| Imagined cost | 0.51 |
| Random (control) | 0.50 |

Rejecting the 30% least-trusted steps by MC-dropout catches **81%** of the wrong steps.

**Offline gate test** (80 decisions, one decision = 1.25 points): no gate 71.3%, MC-dropout gate 70%,
random gate 70%, oracle gate (knows the true error) 72.5%. The differences are within one or two
decisions.

## What is not done yet

To be clear about the current state:

- The **trust gate inside the live planning loop** is designed, but not built. The interface is in
  `src/scope/trust/interface.py`. All closed-loop runs used the plain DINO-WM planner, and the gate was
  only evaluated offline, on recorded plans.
- The **fallback ladder** (what to do when every plan is rejected) is designed, but not built.
- The trust test was run on **Push-T only**, not on Wall or PointMaze, and not under visual changes.
- **No model is trained** anywhere in this project, by design. We use DINO-WM's released checkpoints.

There is **no reward signal** in the data or in the planner. The planner minimises the distance between
the imagined end state and the goal image. "Reward = 1 at the goal" is used only to score success.

## How it works

```
camera image + goal image
      │
      ▼
 DINOv2 encoder (frozen) ──► current state, goal (image features)
      │
      ▼
 CEM planner: 300 random action plans, keep the best 30, repeat (30× Push-T, 10× Wall/PointMaze)
      │
      ▼
 DINO-WM predictor (frozen) imagines the next 5 steps of every plan
      │
      ├──► imagined cost: how far the end is from the goal
      └──► trust score (SCoPe): can each imagined step be believed?
      │
      ▼
 choose the plan (lowest cost; the trust gate would drop low-trust plans)
      │
      ▼
 run the first 5 actions in the simulator ──► new camera image ──► plan again (up to 3 rounds)
```

## Repository layout

| Path | What it is |
|---|---|
| `src/scope/world_model/dino_wm_patches.py` | Our runtime patches for DINO-WM: fused attention, encoding the shared start frame once, chunked rollouts, TF32, gradient-free encode/decode, skipping solved episodes. This is where the 3.8× speed-up comes from. |
| `src/scope/world_model/interface.py` | A standard interface for any latent world model (`encode`, `rollout`, `latent_distance`), so the planner and the trust code do not depend on DINO-WM directly. |
| `src/scope/trust/interface.py` | The interface for trust scorers (one score per imagined step) and for the trust gate. |
| `scripts/h1_pilot.py` | **The main experiment.** It records the candidate plans, computes all trust scores, runs each plan in the real simulator to get the true error, then computes AUROC and the offline gate test. ("H1" is hypothesis 1: *does a trust score predict the real error?*) |
| `scripts/h1_aggregate.py` | Pools the seeds: AUROC, Spearman, bootstrap confidence intervals, offline gate, recall versus rejection rate. |
| `scripts/run_h1_pilot.sh` | Runs `h1_pilot.py` with the right environment variables. |
| `scripts/plan_dino_wm.py` | Runs DINO-WM's own `plan.py` (CEM + MPC) with our patches applied. |
| `scripts/reproduce_dino_wm.sh` | Phase 0: evaluates the released checkpoints with DINO-WM's planning configs. |
| `scripts/phase0_pusht_full.sh`, `phase0_pusht_seeds.sh`, `phase0_env_full.sh`, `phase0_wall_maze_full.sh` | The full Phase 0 runs (Push-T, Wall, PointMaze, several seeds). "Phase 0" means reproducing the base model before adding anything new. |
| `scripts/summarize_phase0.py` | Reads the run logs and prints the success rate after each MPC round. |
| `scripts/benchmark_dino_wm.py` | Compares patched and unpatched rollouts: same elite plans, and how much faster. |
| `scripts/setup_wsl.sh` | Builds the `scope` conda environment (staged, resumable, checksum-verified downloads). |
| `scripts/download_dino_wm_assets.sh` | Downloads DINO-WM's released checkpoints and datasets (about 6 GB). |
| `scripts/extract_wall_val.py` | Extracts only the Wall validation episodes (the full Wall set unpacks to 55 GB). |
| `scripts/launch_detached.ps1` | Windows helper: starts a long WSL job that keeps running after the launching window closes. |
| `environment.yml`, `requirements.txt`, `pyproject.toml` | Dependencies (conda half, pip half) and our `scope` package. |
| `external/dino_wm/` | **Not included.** DINO-WM's original code goes here (see setup step 2). |

Datasets, checkpoints and run outputs are not in the repository. They are large, and the scripts download
or create them.

## Dependencies

- Linux, or **WSL2 Ubuntu** on Windows. We used Windows 11 + WSL2 Ubuntu.
- An NVIDIA GPU. We used an RTX 5060 Laptop with 8 GB.
- Python 3.10 and **PyTorch 2.7.1 with CUDA 12.8**. DINO-WM's original pins (PyTorch 2.3, Python 3.9) do
  not support RTX 50-series GPUs.
- MuJoCo 2.1 and mujoco-py (for PointMaze), pymunk (Push-T), gym 0.23, hydra-core, transformers. The full
  list is in `environment.yml` and `requirements.txt`.
- About 15 GB for the environment and about 7 GB for the checkpoints and datasets.

## Setup and how to run

**1. Get this repository**

```bash
git clone https://github.com/AayuShen/scope-planning.git
cd scope-planning
```

**2. Get the original DINO-WM code** (not included here). Download it from the authors' repository,
**https://github.com/gaoyuezhou/dino_wm**, into `external/dino_wm`, at the commit we used:

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git external/dino_wm
git -C external/dino_wm checkout 0a9492f
```

**3. Build the environment** (inside WSL2 Ubuntu):

```bash
bash scripts/setup_wsl.sh      # creates the conda env "scope"
conda activate scope
```

The script keeps its download caches in folders set by `SCOPE_INSTALLERS_DIR` and `SCOPE_WHEELS_DIR`.
Point them at any folder with enough space.

**4. Download the released checkpoints and datasets** (about 6 GB, into `~/scope_data`, or set
`SCOPE_DATA_DIR`; archives are kept in `SCOPE_ARCHIVE_DIR`):

```bash
bash scripts/download_dino_wm_assets.sh checkpoints pusht    # add wall point_maze when needed
```

**5. Quick check** (5 Push-T episodes, a few minutes):

```bash
bash scripts/reproduce_dino_wm.sh pusht 5 planner.max_iter=2
```

**6. Reproduce the planning results** (hours per run):

```bash
bash scripts/phase0_pusht_full.sh                 # Push-T, 50 episodes, seed 99
bash scripts/phase0_env_full.sh wall 50 3         # Wall, 50 episodes, up to 3 MPC rounds
bash scripts/phase0_env_full.sh point_maze 50 3   # PointMaze
python scripts/summarize_phase0.py --env pusht --paper 0.90 outputs/runs/phase0/pusht/*.log
```

**7. Run the trust-score experiment** (Push-T):

```bash
# smoke test, 2 decisions
bash scripts/run_h1_pilot.sh --episodes 2 --record 1:8,10:8 --mc 2 --bank-trajs 20
# the runs used in the report: 40 + 20 + 20 decisions
RUN_TAG=pilot40    bash scripts/run_h1_pilot.sh --episodes 40 --seed 1
RUN_TAG=pilot20_s2 bash scripts/run_h1_pilot.sh --episodes 20 --seed 2
RUN_TAG=pilot20_s3 bash scripts/run_h1_pilot.sh --episodes 20 --seed 3
# pool the three seeds
python scripts/h1_aggregate.py outputs/runs/h1_pilot/pilot40 outputs/runs/h1_pilot/pilot20_s2 \
       outputs/runs/h1_pilot/pilot20_s3 --out outputs/runs/h1_pilot/pooled_3seeds
```

Main settings of `h1_pilot.py`: 300 CEM samples, top 30, 10 CEM iterations, 192 recorded plans per
decision (`--record 1:48,5:48,10:96`), 4 MC-dropout passes, 30% rejection in the gate test.

## Things to know

- Run everything inside WSL or Linux, not native Windows.
- Long runs must stay in an open WSL session. On Windows, `scripts/launch_detached.ps1` keeps them alive.
- With about 7.5 GB of RAM in WSL, keep `SCOPE_SERIAL_ENVS=1`. DINO-WM otherwise starts one simulator
  process per episode, and 50 of them run out of memory.
- GPU memory is capped at 75% (`SCOPE_GPU_MEM_FRACTION`). On Windows, overflowing the GPU spills into system
  RAM and can freeze the machine.
- Logs are buffered, so a log that has not updated for a while does not mean the job has stopped.

## Credits

- **DINO-WM:** G. Zhou, H. Pan, Y. LeCun, L. Pinto. *DINO-WM: World Models on Pre-trained Visual Features
  enable Zero-shot Planning*, ICML 2025. Code: https://github.com/gaoyuezhou/dino_wm (MIT licence).
  The model, planner, simulators, checkpoints and datasets are theirs.
- **DINOv2:** M. Oquab et al., TMLR 2024.
- **MC-dropout:** Y. Gal, Z. Ghahramani. *Dropout as a Bayesian Approximation*, ICML 2016.
- **PETS:** K. Chua et al., NeurIPS 2018. Uncertainty-aware planning with CEM.
