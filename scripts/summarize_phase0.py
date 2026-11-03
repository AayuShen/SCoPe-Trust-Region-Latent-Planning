"""Summarise DINO-WM reproduction runs (Phase 0): success per MPC round, per seed.

Reads the plain-text log of each run, which prints `self.is_success: [...]` (a numpy
array, per-episode, accumulated over rounds) after every MPC round, and prints a
markdown table plus the unsolved episodes. Standard library only.

  python summarize_phase0.py --env pusht --paper 0.90 LOG [LOG ...]
  python summarize_phase0.py --env pusht --paper 0.90 --glob "outputs/runs/phase0/pusht/seed*.log"

A run that crashed part-way is still summarised for the rounds it completed.
"""

import argparse
import glob
import re
import statistics
from pathlib import Path

ROUND = re.compile(r"MPC iter (\d+) Eval")
SUCCESS = re.compile(r"self\.is_success:\s*\[([^\]]*)\]")
SEED = re.compile(r"seed=?(\d+)")  # matches both "seed=100" and the "seed100_<time>.log" file names
SKIP = re.compile(r"planning (\d+) of (\d+) episodes")
CRASH = re.compile(r"Traceback|OutOfMemoryError|Killed")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    rounds, flags = [], []
    for m in SUCCESS.finditer(text):
        flags.append([tok == "True" for tok in m.group(1).split()])
    n_rounds_started = len(ROUND.findall(text))
    for f in flags:
        rounds.append(sum(f) / len(f))
    seed = SEED.search(path.name)
    return {
        "log": path.name,
        "seed": int(seed.group(1)) if seed else 99,  # upstream default seed; seed-less logs are the base run
        "n_evals": len(flags[0]) if flags else 0,
        "rounds": rounds,
        "flags": flags,
        "rounds_started": n_rounds_started,
        "crashed": bool(CRASH.search(text)),
        "finished": "[phase0] finished with exit code 0" in text,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*")
    p.add_argument("--glob", default=None)
    p.add_argument("--env", default="pusht")
    p.add_argument("--paper", type=float, default=None, help="success rate reported in the paper")
    args = p.parse_args()

    paths = [Path(x) for x in args.logs]
    if args.glob:
        paths += [Path(x) for x in sorted(glob.glob(args.glob))]
    runs = [parse_log(x) for x in paths]
    runs = [r for r in runs if r["rounds"]]
    if not runs:
        raise SystemExit("no completed MPC round found in the given logs")
    runs.sort(key=lambda r: r["seed"])

    width = max(len(r["rounds"]) for r in runs)
    head = "| seed | episodes | " + " | ".join(f"after round {i + 1}" for i in range(width)) + " | status |"
    print(f"### {args.env}: success rate per MPC round\n")
    print(head)
    print("|" + "---|" * (3 + width))
    for r in runs:
        cells = [f"{x:.2f}" for x in r["rounds"]] + ["-"] * (width - len(r["rounds"]))
        status = "complete" if r["finished"] else (
            f"crashed in round {len(r['rounds']) + 1}" if r["crashed"] else "incomplete")
        print(f"| {r['seed']} | {r['n_evals']} | " + " | ".join(cells) + f" | {status} |")

    print()
    for i in range(width):
        vals = [r["rounds"][i] for r in runs if len(r["rounds"]) > i]
        if len(vals) < 2:
            continue
        spread = f"mean {statistics.mean(vals):.3f}, std {statistics.stdev(vals):.3f}, " \
                 f"range {min(vals):.2f} to {max(vals):.2f} over {len(vals)} seeds"
        print(f"- after round {i + 1}: {spread}")
    if args.paper is not None:
        first = [r["rounds"][0] for r in runs]
        print(f"- paper: {args.paper:.2f}; round-1 results: {', '.join(f'{x:.2f}' for x in first)}")

    print("\nUnsolved episodes after the last completed round:")
    for r in runs:
        last = r["flags"][-1]
        bad = [i for i, ok in enumerate(last) if not ok]
        print(f"- seed {r['seed']}: {len(bad)} of {len(last)} unsolved {bad}")


if __name__ == "__main__":
    main()
