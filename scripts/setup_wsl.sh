#!/usr/bin/env bash
# Staged, resumable environment setup inside WSL2 Ubuntu. Run from the repo root:
#   bash scripts/setup_wsl.sh                  # all stages
#   bash scripts/setup_wsl.sh wheels install   # selected stages
#
# Large files are never fetched by pip directly: pip only resolves versions, then
# every package is downloaded with curl (resuming only where the server supports it), checked against its sha256, and
# installed offline from the local wheelhouse. Rerunning any stage is safe.
#
# Downloads are kept on E: as ordinary Windows folders so other projects can reuse
# them and deleting them really frees space; installed software stays inside Ubuntu.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DL="$HOME/scope_downloads"            # small working files: pip report, pip cache, temp
INSTALLERS="${SCOPE_INSTALLERS_DIR:-/mnt/e/DevTools/WSLCache/installers}"
WHEELS="${SCOPE_WHEELS_DIR:-/mnt/e/DevTools/WSLCache/pip-wheels/cp310-cu128}"
CONDA_HOME="$HOME/miniforge3"
ENV_NAME="scope"
PY="$CONDA_HOME/envs/$ENV_NAME/bin/python"
HOST_DRIVE="${SCOPE_HOST_DRIVE:-/mnt/e}"   # Windows drive holding Ubuntu's ext4.vhdx
APT_PACKAGES=(build-essential libosmesa6-dev libgl1 libglew-dev libglfw3 patchelf unzip curl)

export TMPDIR="$DL/tmp"               # WSL's /tmp is RAM-backed; keep temp files on disk
export PIP_CACHE_DIR="$DL/pip-cache"
mkdir -p "$INSTALLERS" "$WHEELS" "$TMPDIR"

verify() {  # verify <file> [sha256]: checksum when the index publishes one, else archive integrity
  local f="$1" sha="${2:-}"
  if [ -n "$sha" ]; then
    echo "$sha  $f" | sha256sum -c --quiet - >/dev/null 2>&1
  else
    case "$f" in
      *.whl|*.zip|*.whl.part|*.zip.part) unzip -tq "$f" >/dev/null 2>&1 ;;
      *.tar.gz|*.tgz|*.tar.gz.part|*.tgz.part) tar -tzf "$f" >/dev/null 2>&1 ;;
      *) [ -s "$f" ] ;;
    esac
  fi
}

fetch() {  # fetch <url> <dest> [sha256]: resumes on interruption, renames only when verified
  local url="$1" dest="$2" sha="${3:-}"
  if [ -f "$dest" ]; then
    verify "$dest" "$sha" && return 0
    echo "  verification failed, re-downloading $(basename "$dest")" >&2
    rm -f "$dest"
  fi
  local free; free=$(df -B1 --output=avail "$HOST_DRIVE" | tail -1)
  if [ "$free" -lt $((5 * 1024 ** 3)) ]; then
    echo "Less than 5 GB free on $HOST_DRIVE (holds Ubuntu's disk image); stopping" >&2; exit 1
  fi
  for attempt in $(seq 1 30); do
    # curl -C - resumes only when the server honours Range; otherwise it fails with
    # code 33 and we restart from zero rather than appending a second copy.
    local rc=0
    curl -fL -C - --connect-timeout 30 --speed-limit 10000 --speed-time 120 -s -S -o "$dest.part" "$url" || rc=$?
    [ "$rc" = 33 ] && { rm -f "$dest.part"; echo "  server cannot resume, restarting $(basename "$dest")" >&2; continue; }
    if [ "$rc" = 0 ]; then
      if ! verify "$dest.part" "$sha"; then
        echo "  verification failed after download, restarting $(basename "$dest")" >&2
        rm -f "$dest.part"; continue
      fi
      mv "$dest.part" "$dest"; return 0
    fi
    echo "  interrupted ($attempt/30), resuming $(basename "$dest")" >&2
    sleep 5
  done
  echo "FAILED: $url" >&2; return 1
}

stage_system() {
  local missing=()
  for p in "${APT_PACKAGES[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing system packages. Run, then rerun this script:"
    echo "  sudo apt-get update && sudo apt-get install -y ${missing[*]}"
    exit 1
  fi
  echo "[system] ok"
}

stage_mujoco() {
  if [ -f "$HOME/.mujoco/mujoco210/bin/libmujoco210.so" ]; then echo "[mujoco] already installed"; return; fi
  fetch https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz "$INSTALLERS/mujoco210-linux-x86_64.tar.gz"
  mkdir -p "$HOME/.mujoco"
  rm -rf "$HOME/.mujoco/mujoco210" "$HOME/.mujoco/.staging" && mkdir "$HOME/.mujoco/.staging"
  tar -xzf "$INSTALLERS/mujoco210-linux-x86_64.tar.gz" -C "$HOME/.mujoco/.staging"
  mv "$HOME/.mujoco/.staging/mujoco210" "$HOME/.mujoco/" && rmdir "$HOME/.mujoco/.staging"
  echo "[mujoco] installed"
}

stage_conda() {
  if [ ! -x "$CONDA_HOME/bin/conda" ]; then
    fetch https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh "$INSTALLERS/Miniforge3-Linux-x86_64.sh"
    bash "$INSTALLERS/Miniforge3-Linux-x86_64.sh" -b -u -p "$CONDA_HOME"
  fi
  if [ -x "$PY" ]; then
    "$CONDA_HOME/bin/conda" env update -q -n "$ENV_NAME" -f "$REPO_ROOT/environment.yml"
  else
    "$CONDA_HOME/bin/conda" env create -q -n "$ENV_NAME" -f "$REPO_ROOT/environment.yml"
  fi
  # Build tools for the offline install (old setup.py packages are built without isolation).
  "$PY" -m pip install -q "setuptools>=64,<70" wheel "cython==0.29.37"
  echo "[conda] env '$ENV_NAME' ready: $("$PY" --version)"
}

TORCH_WHEELS=(  # CUDA 12.8 builds, which only PyTorch's own server hosts
  "https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl|d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3|torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
  "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl|538f4db667286d939b4eee0a66d31ed21b51186668006b0e0ffe20338ecc7e00|torchvision-0.22.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
)

stage_wheels() {
  # PyTorch's server publishes no standalone metadata, so letting pip resolve against
  # it downloads every NVIDIA wheel in full (non-resumably) just to read dependencies.
  # Fetch the two torch wheels ourselves, then resolve the rest against PyPI, which
  # serves metadata separately: no large downloads happen during resolution.
  echo "[wheels] fetching torch + torchvision (CUDA 12.8 builds)"
  local entry url sha name
  for entry in "${TORCH_WHEELS[@]}"; do
    IFS='|' read -r url sha name <<< "$entry"
    fetch "$url" "$WHEELS/$name" "$sha" || { echo "[wheels] FAILED on $name" >&2; exit 1; }
  done

  echo "[wheels] resolving remaining versions against PyPI (metadata only)..."
  "$PY" -m pip install --dry-run --ignore-installed --quiet --find-links "$WHEELS" \
    --report "$DL/pip-report.json" -r "$REPO_ROOT/requirements.txt"
  "$PY" - "$DL/pip-report.json" > "$DL/downloads.tsv" <<'EOF'
import json, sys, os, urllib.parse
for item in json.load(open(sys.argv[1]))["install"]:
    info = item["download_info"]
    if info["url"].startswith("file:"):
        continue  # already in the local wheelhouse (the prefetched torch wheels)
    if "vcs_info" in info:
        sys.exit(f"cannot prefetch non-archive source: {info['url']}")
    url = info["url"].split("#")[0]
    archive = info.get("archive_info", {})
    sha = archive.get("hashes", {}).get("sha256", "")
    if not sha and archive.get("hash", "").startswith("sha256="):
        sha = archive["hash"].split("=", 1)[1]
    name = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path))
    print(f"{url}|{sha or '-'}|{name}")  # '-' placeholder: empty fields would collapse
EOF
  local n; n=$(wc -l < "$DL/downloads.tsv")
  echo "[wheels] $n packages to fetch into $WHEELS"
  while IFS='|' read -r url sha name; do
    [ "$sha" = "-" ] && sha=""
    fetch "$url" "$WHEELS/$name" "$sha" || { echo "[wheels] FAILED on $name" >&2; exit 1; }
  done < "$DL/downloads.tsv"
  echo "[wheels] all $n packages downloaded and verified"
}

stage_install() {
  "$PY" -m pip install --no-index --find-links "$WHEELS" --no-build-isolation \
    -r "$REPO_ROOT/requirements.txt"
  "$PY" -m pip install --no-index --no-build-isolation --no-deps -e "$REPO_ROOT"
  echo "[install] done"
}

stage_activate() {
  local d="$CONDA_HOME/envs/$ENV_NAME/etc/conda/activate.d"
  mkdir -p "$d"
  cat > "$d/scope.sh" <<'EOF'
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/wsl/lib"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export SCOPE_DATA_DIR="${SCOPE_DATA_DIR:-$HOME/scope_data}"
export DATASET_DIR="$SCOPE_DATA_DIR/datasets"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TMPDIR="$HOME/scope_downloads/tmp"
# mujoco-py compiles a C extension against Ubuntu's OSMesa headers, which conda's
# compilers cannot see; use the system gcc with the pre-gcc-14 C defaults its
# 2021-era generated code was written against (C17, pointer mismatches as warnings).
export CC=/usr/bin/gcc
export CFLAGS="-std=gnu17 -Wno-error=incompatible-pointer-types -Wno-error=int-conversion -Wno-error=implicit-function-declaration"
EOF
  echo "[activate] env vars installed"
}

stage_verify() {
  # Also triggers mujoco-py's one-time Cython build so failures surface here.
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  python - <<'EOF' || { echo "[verify] FAILED" >&2; exit 1; }
import torch, mujoco_py, gym, d4rl, pymunk, decord, scope
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda"); print("gpu matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
print("mujoco_py, gym, d4rl, pymunk, decord, scope: ok")
EOF
  echo "[verify] ok"
}

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(system mujoco conda wheels install activate verify)
for s in "${STAGES[@]}"; do "stage_$s"; done
echo "[setup] finished: ${STAGES[*]}"
