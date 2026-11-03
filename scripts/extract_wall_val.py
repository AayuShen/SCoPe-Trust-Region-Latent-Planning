"""Extract only what DINO-WM planning reads from the Wall dataset zip.

wall_single.zip unpacks to 55 GB: ~1,800 episode files (obses/episode_XXX.pth, ~30 MB
each) plus small metadata. plan.py only uses the validation split, which wall_dset
draws as a seeded random 10% of the episodes (conf/env/wall.yaml: split_mode random,
split_ratio 0.9; get_train_val_sliced: seed 42). This reproduces that split exactly
(torch.randperm with a seed-42 generator, validation = everything after the first
int(0.9 * N) indices) and extracts the metadata plus those episodes: ~5.5 GB.
Training would need the other 90%; planning does not touch them.

  python extract_wall_val.py <wall_single.zip> <datasets_dir> [--split-ratio 0.9] [--seed 42]
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("zip_path", type=Path)
    p.add_argument("datasets_dir", type=Path)
    p.add_argument("--split-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--host-drive", default="/mnt/e", help="drive that holds Ubuntu's disk image")
    p.add_argument("--margin-gb", type=float, default=8.0)
    args = p.parse_args()

    final = args.datasets_dir / "wall_single"
    stage = args.datasets_dir / ".staging_wall_single"
    done = args.datasets_dir / ".wall_single.done"
    if done.exists():
        print("wall_single: already done")
        return
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    zf = zipfile.ZipFile(args.zip_path)
    members = [
        i for i in zf.infolist()
        if not i.is_dir() and not i.filename.startswith("__MACOSX") and "/.huggingface/" not in i.filename
    ]
    meta = [i for i in members if "/obses/" not in i.filename]
    obses = {int(Path(i.filename).stem.split("_")[1]): i for i in members if "/obses/" in i.filename}

    for i in meta:
        zf.extract(i, stage)
    states = torch.load(stage / "wall_single" / "states.pth")
    n = len(states)
    if sorted(obses) != list(range(n)):
        sys.exit(f"zip has {len(obses)} episode files but states.pth has {n} rows; refusing to guess")

    n_train = int(args.split_ratio * n)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    val = sorted(order[n_train:])
    need = sum(obses[k].file_size for k in val)
    free = shutil.disk_usage(args.host_drive).free
    print(f"{n} episodes; validation = {len(val)} episodes, {need / 2**30:.1f} GiB to write; "
          f"{free / 2**30:.1f} GiB free on {args.host_drive}", flush=True)
    if need + args.margin_gb * 2**30 > free:
        shutil.rmtree(stage, ignore_errors=True)
        sys.exit("not enough free space on the host drive; nothing extracted")

    for k, idx in enumerate(val, 1):
        zf.extract(obses[idx], stage)
        if k % 40 == 0:
            print(f"  {k}/{len(val)} episodes", flush=True)

    shutil.rmtree(final, ignore_errors=True)
    shutil.move(str(stage / "wall_single"), str(final))
    shutil.rmtree(stage, ignore_errors=True)
    (final / "VAL_ONLY.txt").write_text(
        f"Only validation episodes were extracted (seed {args.seed}, split_ratio {args.split_ratio}).\n"
        f"Valid for planning/evaluation only; training needs the full dataset.\n"
        f"Episode indices: {val}\n"
    )
    done.touch()
    print(f"wall_single: done ({len(val)} validation episodes)")


if __name__ == "__main__":
    main()
