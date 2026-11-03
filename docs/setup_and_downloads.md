# Environment setup and downloads: what, how big, and what went wrong

Everything here lives inside Ubuntu (WSL2), whose entire filesystem is one file on
E: — `E:\DevTools\WSL\Ubuntu-26.04\ext4.vhdx`. E: is a 100 GB partition (~84 GB free
after the fresh Ubuntu install). Nothing is written to C: except brief installer
temp files.

## 1. What `scripts/setup_wsl.sh` installs

Sizes are what was actually measured on the first (successful) install on 2026-10-02.

| Stage | What it downloads / installs | Why the project needs it | Size on disk |
|---|---|---|---|
| `system` | Checks Ubuntu packages: build-essential (gcc), OSMesa/OpenGL, GLEW, GLFW, patchelf, unzip, curl. Installs nothing itself; prints one `sudo apt-get install` command if any are missing. | C compiler for mujoco-py's one-time build; headless rendering for MuJoCo | ~0.3 GB (apt) |
| `mujoco` | MuJoCo 2.1 physics engine → `~/.mujoco` (installer kept in `E:\DevTools\WSLCache\installers`) | DINO-WM's PointMaze environment; every DINO-WM env imports it | 9 MB |
| `conda` | Miniforge (conda) → `~/miniforge3` (installer kept in `E:\DevTools\WSLCache\installers`), then a `scope` environment with Python 3.10, ffmpeg, gcc 12 | Isolated Python matching DINO-WM's needs | ~2.5 GB |
| `wheels` | **119 Python packages** downloaded into `E:\DevTools\WSLCache\pip-wheels\cp310-cu128`, each checksum-verified | See below | **4.1 GB** |
| `install` | Installs those 119 packages offline into the `scope` env, plus our `scope` package | — | ~6 GB more inside `~/miniforge3` |
| `activate`, `verify` | Environment variables; one-time compile of mujoco-py; GPU check | — | negligible |
| | | **Total after setup** | **~15 GB** |

### Where the 4.1 GB of packages comes from

| Package group | Approx. size | Purpose |
|---|---|---|
| PyTorch 2.7.1 + CUDA 12.8 | ~1 GB | Runs DINO-WM on the RTX 5060 (older PyTorch cannot use this GPU) |
| NVIDIA CUDA libraries (cuDNN, cuBLAS, NCCL, cuSPARSE, cuFFT, …) | ~2.5 GB | Required by PyTorch's GPU build; cuDNN alone is ~0.7 GB |
| Everything else (torchvision, transformers, OpenCV, d4rl, gym, pymunk, scipy, …) | ~0.6 GB | DINO-WM's code, simulators and data loading |

The large single files (0.6–1 GB each) are the ones that kept failing at ~600 MB.
Root cause: PyTorch's package server publishes no separate metadata, so pip downloaded
every NVIDIA package *in full, non-resumably* just to read its dependency list, before
installing anything. The script now fetches only torch/torchvision from PyTorch's
server (resumable, checksummed) and resolves everything else against PyPI, which
publishes metadata separately, so no large file is downloaded during resolution.

### Space that can be reclaimed after setup (optional)

| Item | Size | Cost of deleting |
|---|---|---|
| `E:\DevTools\WSLCache\pip-wheels\cp310-cu128` | 4.1 GB | Rebuilding this env (or another py3.10 + CUDA 12.8 project) re-downloads them |
| conda package cache (`conda clean --all`) | ~2 GB | Same |

## 2. What `scripts/download_dino_wm_assets.sh` downloads

All from DINO-WM's official OSF storage. Verified zips are kept in
`E:\Libs\Datasets\dino_wm` for reuse; extracted copies go inside Ubuntu
(`~/scope_data`) for fast reads. Exact sizes from OSF:

| File | Size | What it is | Needed for |
|---|---|---|---|
| `outputs.zip` | 0.95 GB | DINO-WM's **pretrained world-model checkpoints** for PointMaze, Push-T, Wall | All Phase 0 runs |
| `pusht_noise.zip` | 2.79 GB | Push-T dataset (videos + actions) | Push-T eval (first env) |
| `wall_single.zip` | 1.67 GB | Wall dataset | Wall eval (second env) |
| `point_maze.zip` | 0.72 GB | PointMaze dataset | PointMaze eval (third env) |
| | **6.13 GB** | | |
| *Not downloaded:* `deformable.*` | *15.8 GB* | Rope / Granular datasets | *Out of scope (needs PyFleX)* |

Planning needs the datasets because start and goal states are drawn from them.

**Peak space while downloading:** each archive and its extracted copy exist together
until the archive is deleted, so the peak is roughly double the largest archive on top
of everything already downloaded: **~13 GB peak, ~7 GB after extraction.**

## 3. Why E: filled up (2026-10-02)

The datasets above total 6 GB. The drive filled with ~72 GB because of a bug in my
download script, not because of the data:

1. The script used `wget -c -O <file>` to resume interrupted downloads.
2. OSF **ignores resume requests** — asked for "bytes 100–200" it sends the whole
   953 MB file again (verified afterwards).
3. With `-O`, wget then **appended** that full copy to the partial file instead of
   replacing it. Every dropped connection and retry added another ~1 GB copy.
4. The download never got past the first file (`outputs.zip`, the checkpoints), so the
   growth was almost certainly that one file repeated many times. (Ubuntu could not
   boot afterwards, so this could not be inspected directly.)
5. Ubuntu's disk file grew 19 GB → 91 GB in ~50 minutes until E: had 0 bytes free;
   Ubuntu's filesystem then went read-only and WSL could not start.
6. The script's space check would not have helped: `df` inside Ubuntu reports the
   virtual 1 TB disk, not E:'s real free space.

Recovery: Ubuntu was unregistered (freeing the space) and reinstalled fresh.

## 4. Safeguards now in the scripts (written, not yet run)

| Risk | Safeguard |
|---|---|
| Server ignores resume → file grows | OSF downloads never resume: each attempt overwrites from zero. Other servers use `curl -C -`, which errors instead of appending if resume is unsupported, and the script then restarts that file from zero. |
| A file grows past its real size | OSF downloads are capped at the published byte size (`--max-filesize`) and must match it exactly. |
| Corrupt or partial file kept | Every file must match its published SHA-256 before it is kept; partial files are `.part` and are never used. |
| Drive fills silently | Free space is checked **on E: itself** (`/mnt/e`), not inside Ubuntu. Downloads stop if they would leave less than 5 GB free. |
| Endless retries | At most 10 attempts per dataset file, then a clear `FAILED` and exit. |
| Failures reported as success | Every stage exits non-zero and prints `FAILED` on error. |

## 5. Disk budget on E:

| | Size |
|---|---|
| Fresh Ubuntu | 1.4 GB |
| Environment (setup) | ~15 GB |
| Datasets + checkpoints (after extraction) | ~7 GB |
| Download peak headroom | ~6 GB |
| **Ubuntu disk file at its largest** | **~29 GB** |
| E: free now | 83.8 GB |

**Important:** Ubuntu's disk file grows but **does not shrink** on its own. Deleting
files inside Ubuntu frees space for Ubuntu, not for E:. Expect the disk file to stay
near its peak (~29 GB) unless it is compacted, or unless sparse mode is enabled first
(`wsl --manage Ubuntu-26.04 --set-sparse true`, which returns freed space to E:
automatically; this is a WSL setting, so only with your OK).

## 6. Proposed order of work (each step only with your go-ahead)

1. **You:** create your Ubuntu user (`wsl -d Ubuntu-26.04`), then run the one `sudo apt-get install …`
   line the `system` stage prints.
2. *(Optional, recommended)* enable sparse mode so deletions give space back to E:.
3. Setup stages one at a time, reporting E: free space after each:
   `mujoco` → `conda` → `wheels` → `install` → `activate` → `verify`.
4. *(Optional)* delete the 4.1 GB wheelhouse and run `conda clean --all`.
5. Download **only the checkpoints (0.95 GB)** first (`download_dino_wm_assets.sh checkpoints`), confirm size + checksum + E: space.
6. Download **only Push-T (2.79 GB)** (`download_dino_wm_assets.sh pusht`), then run the 5-episode smoke test.
7. Wall and PointMaze datasets later, when those environments are actually needed.

Expected E: free space at the end: ~55 GB (or ~60 GB with steps 2 and 4).
