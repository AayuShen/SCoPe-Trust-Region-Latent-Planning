"""Pool several H1 pilot runs (scripts/h1_pilot.py) into one result.

    python scripts/h1_aggregate.py outputs/runs/h1_pilot/pilot40 outputs/runs/h1_pilot/pilot40_s2 ... --out <dir>

Reads each run's per_candidate.npz, labels errors exactly as the pilot does (top 10% per imagined step,
threshold taken within each run), and reports on the pooled decision points:
- stratified AUROC and Spearman per score, with 95% bootstrap intervals over decision points,
- the same AUROC per run (seed), to show the spread,
- one pre-specified combined score, the rank average of MC-dropout and "distance moved" (no fitting),
- the offline gate replay (reject 30%, pick the lowest imagined cost) on all decision points.
Numpy only; no GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

COMBO = ("mc_dropout", "dist_start")


# ---------------------------------------------------------------- metrics (same as scripts/h1_pilot.py,
# copied because importing it pulls in DINO-WM's environments)
def ranks(x):
    return rankdata(x)  # average ranks for ties, as in the pilot


def auroc(score, label):
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = ranks(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def stratified(fn, score, err, label):
    """Average a per-step metric over imagined steps 1..5 (removes the trivial step effect)."""
    return float(np.nanmean([fn(score[:, k], (label if label is not None else err)[:, k]) for k in range(err.shape[1])]))


def load(run):
    z = np.load(Path(run) / "per_candidate.npz")
    E = len({k.split("_", 1)[0] for k in z.files})
    names = sorted(k[len("ep0_score_"):] for k in z.files if k.startswith("ep0_score_"))
    rows = []
    for i in range(E):
        rows.append({"err": z[f"ep{i}_err"], "cost": z[f"ep{i}_cost"], "success": z[f"ep{i}_success"],
                     "scores": {n: z[f"ep{i}_score_{n}"] for n in names}})
    thr = np.percentile(np.concatenate([r["err"] for r in rows]), 90, axis=0)  # as in the pilot
    for r in rows:
        r["label"] = r["err"] >= thr
    return rows, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--reject", type=float, default=0.3)
    a = p.parse_args()

    rows, per_run, names = [], [], None
    for run in a.runs:
        r, names = load(run)
        per_run.append((Path(run).name, len(rows), len(rows) + len(r)))
        rows += r
    E = len(rows)

    # pre-specified combination: rank average within each imagined step over all pooled candidates
    steps = rows[0]["err"].shape[1]
    sizes = [len(r["err"]) for r in rows]
    combo = np.zeros((sum(sizes), steps))
    for n in COMBO:
        s = np.concatenate([r["scores"][n] for r in rows])
        combo += np.stack([ranks(s[:, k]) for k in range(steps)], 1) / len(s)
    off = np.cumsum([0] + sizes)
    for i, r in enumerate(rows):
        r["scores"]["combo_mc_dist"] = combo[off[i]:off[i + 1]] / len(COMBO)
    names = list(names) + ["combo_mc_dist"]

    def summarize(sel):
        e = np.concatenate([rows[i]["err"] for i in sel])
        lab = np.concatenate([rows[i]["label"] for i in sel])
        return {n: {"auroc": stratified(auroc, np.concatenate([rows[i]["scores"][n] for i in sel]), e, lab),
                    "spearman": stratified(spearman, np.concatenate([rows[i]["scores"][n] for i in sel]), e, None)}
                for n in names}

    res = summarize(range(E))
    boot = {n: [] for n in names}
    rng = np.random.default_rng(0)
    for _ in range(500):
        b = summarize(rng.integers(0, E, E))
        for n in names:
            boot[n].append(b[n]["auroc"])
    for n in names:
        lo, hi = np.nanpercentile(boot[n], [2.5, 97.5])
        res[n]["auroc_ci"] = [float(lo), float(hi)]
    by_run = {tag: {n: v["auroc"] for n, v in summarize(range(s, t)).items()} for tag, s, t in per_run}

    def pick(r, trust):
        keep = np.ones(len(r["cost"]), bool) if trust is None else trust <= np.quantile(trust, 1 - a.reject)
        return r["success"][np.flatnonzero(keep)[np.argmin(r["cost"][keep])]]

    grng = np.random.default_rng(1)
    gates = {"plain (no gate)": lambda r: None, "oracle (true error)": lambda r: r["err"].max(1),
             "random gate": lambda r: grng.random(len(r["cost"]))}
    for n in ["mc_dropout", "act_perturb", "knn", "dist_start", "combo_mc_dist"]:
        gates[f"gate: {n}"] = (lambda nn: (lambda r: r["scores"][nn].max(1)))(n)
    gate = {g: float(np.mean([pick(r, fn(r)) for r in rows])) for g, fn in gates.items()}

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return None if isinstance(x, float) and np.isnan(x) else x

    summary = clean({
        "setup": {"runs": [t for t, _, _ in per_run], "episodes": E, "pairs": int(sum(r["err"].size for r in rows)),
                  "reject_fraction": a.reject, "label": "real latent error in the top 10% for its imagined step (threshold per run)",
                  "combo": "rank average of " + " and ".join(COMBO) + " within each imagined step; pre-specified, no fitting"},
        "auroc_stratified": res,
        "auroc_by_run": by_run,
        "offline_gate_success": gate,
        "candidate_success_rate": float(np.mean(np.concatenate([r["success"] for r in rows]))),
    })
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
