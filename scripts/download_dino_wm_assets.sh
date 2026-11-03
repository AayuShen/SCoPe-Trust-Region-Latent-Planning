#!/usr/bin/env bash
# Download DINO-WM's released checkpoints and the datasets for Phase 0 (~6 GB total).
# Deformable datasets (~16 GB, need PyFleX) are deliberately excluded.
#
# OSF ignores HTTP Range requests, so downloads cannot resume: each attempt starts
# from zero and overwrites, is capped at the published size, and must match the
# published sha256 before it is kept. Free space is checked on the Windows drive
# that holds Ubuntu's disk image, because df inside WSL reports the virtual disk.
set -euo pipefail

DATA_DIR="${SCOPE_DATA_DIR:-$HOME/scope_data}"            # extracted, inside Ubuntu (fast reads)
ARCHIVES="${SCOPE_ARCHIVE_DIR:-/mnt/e/Libs/Datasets/dino_wm}"  # verified zips, kept on E: for reuse
HOST_DRIVE="${SCOPE_HOST_DRIVE:-/mnt/e}"   # Windows drive holding ext4.vhdx
Q="?view_only=a56a296ce3b24cceaf408383a175ce28"
MARGIN=$((5 * 1024 ** 3))                  # always leave 5 GB free on the host drive
mkdir -p "$DATA_DIR/datasets" "$DATA_DIR/checkpoints" "$ARCHIVES"

host_free() { df -B1 --output=avail "$HOST_DRIVE" | tail -1; }

fetch() {  # fetch <osf_id> <dest_dir> <name> <bytes> <sha256>
  local id="$1" dir="$2" name="$3" bytes="$4" sha="$5"
  local zip="$ARCHIVES/$name.zip" part="$ARCHIVES/$name.zip.part" done="$dir/.$name.done" stage="$dir/.staging_$name"
  [ -f "$done" ] && { echo "$name: already done"; return; }

  # The archive (kept on E:) and its extracted copy (Ubuntu's disk, also on E:) both
  # stay, so require room for both plus margin; a reused archive needs only the copy.
  local need=$((bytes * 2 + MARGIN)) free; free=$(host_free)
  [ -f "$zip" ] && need=$((bytes + MARGIN))
  if [ "$free" -lt "$need" ]; then
    echo "$name: needs $((need / 1024 ** 3)) GB free on $HOST_DRIVE, only $((free / 1024 ** 3)) GB; stopping" >&2
    exit 1
  fi

  if ! { [ -f "$zip" ] && echo "$sha  $zip" | sha256sum -c --quiet - >/dev/null 2>&1; }; then
    rm -f "$zip"
    local attempt ok=0
    for attempt in $(seq 1 10); do
      rm -f "$part"
      if curl -fL --max-filesize "$bytes" --connect-timeout 30 --speed-limit 10000 --speed-time 120 \
          --progress-bar -o "$part" "https://osf.io/download/$id/$Q" \
         && [ "$(stat -c %s "$part")" -eq "$bytes" ] \
         && echo "$sha  $part" | sha256sum -c --quiet - >/dev/null 2>&1; then
        mv "$part" "$zip"; ok=1; break
      fi
      echo "  $name: attempt $attempt/10 failed or incomplete, restarting from zero" >&2
      sleep 10
    done
    rm -f "$part"
    [ "$ok" = 1 ] || { echo "$name: FAILED after 10 attempts" >&2; exit 1; }
  fi

  [ -n "${WALL_ONLY_DOWNLOAD:-}" ] && { echo "$name: archive verified"; return; }

  # The zip's size says little about what it unpacks to (wall_single: 1.7 GB -> 55 GB),
  # so check the unpacked size before writing a byte. This is what filled E: once.
  local unpacked free_now; unpacked=$(unzip -l "$zip" | awk 'END{print $1}'); free_now=$(host_free)
  if [ "$unpacked" -gt $((free_now - MARGIN)) ]; then
    echo "$name: unpacks to $((unpacked / 1024 ** 3)) GB but only $((free_now / 1024 ** 3)) GB are free" \
         "on $HOST_DRIVE (margin $((MARGIN / 1024 ** 3)) GB); not extracting" >&2
    exit 1
  fi

  rm -rf "$stage" && mkdir -p "$stage"
  unzip -q "$zip" -d "$stage"
  for entry in "$stage"/* "$stage"/.[!.]*; do
    [ -e "$entry" ] || continue
    rm -rf "$dir/$(basename "$entry")"
    mv "$entry" "$dir/"
  done
  rm -rf "$stage"
  touch "$done"
  echo "$name: done"
}

#   bash scripts/download_dino_wm_assets.sh [checkpoints] [pusht] [wall] [point_maze]
# No arguments downloads all four.
ITEMS=("$@")
[ ${#ITEMS[@]} -eq 0 ] && ITEMS=(checkpoints pusht wall point_maze)
for item in "${ITEMS[@]}"; do
  case "$item" in
    checkpoints) fetch xvzs4 "$DATA_DIR/checkpoints" outputs     953204628  425b0d8c2a4194d3ec996b553575f69409aa5714176b063d8401789db9179482 ;;
    pusht)       fetch k2d8w "$DATA_DIR/datasets"    pusht_noise 2785304515 442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08 ;;
    wall)        # 55 GB unpacked; planning needs only the 192 validation episodes (5.5 GB).
                 # Download + verify here, then extract just those (scripts/extract_wall_val.py).
                 WALL_ONLY_DOWNLOAD=1 fetch 49rnx "$DATA_DIR/datasets" wall_single 1668205895 2b4ae4ed0ad03b337efac637f17752e7e7e27f864fec39dc25b51fef490c980d
                 "${PYTHON:-python}" "$(dirname "$0")/extract_wall_val.py" "$ARCHIVES/wall_single.zip" "$DATA_DIR/datasets" ;;
    point_maze)  fetch vr5gy "$DATA_DIR/datasets"    point_maze  718363945  6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc ;;
    *) echo "unknown item '$item' (use: checkpoints pusht wall point_maze)" >&2; exit 1 ;;
  esac
done
echo "E: free: $(( $(host_free) / 1024 ** 3 )) GB"
