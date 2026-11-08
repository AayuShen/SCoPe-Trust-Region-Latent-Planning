# Results

The measured results behind every number in the main README. All of them come from the scripts in
`scripts/`; nothing here was edited by hand.

| File | What it holds |
|---|---|
| `phase0_metrics.json` | Planning success after each MPC round, per task and seed (Push-T and Wall: seeds 99, 100, 101; PointMaze: seed 99), parsed from the Phase 0 run logs. |
| `h1_trust/pilot40/` | Trust-score run, seed 1, 40 decisions. |
| `h1_trust/pilot20_s2/` | Trust-score run, seed 2, 20 decisions. |
| `h1_trust/pilot20_s3/` | Trust-score run, seed 3, 20 decisions. |
| `h1_trust/pooled_3seeds/summary.json` | All 80 decisions pooled: the AUROC table with 95% bootstrap intervals, AUROC per seed, the offline gate test, and candidate success rates. |

Each trust-score run folder has two files:

- `per_candidate.npz` is the labelled data. For every recorded plan it holds the true error of each of its
  5 imagined steps, the value of every trust score at every step, the imagined cost, the real cost, and
  whether the plan really succeeded in the simulator. In total that is 80 decisions × 192 plans × 5 steps =
  76,800 labelled imagined steps.
- `summary.json` holds that run's own metrics, as written by `scripts/h1_pilot.py`.

## Check the numbers yourself (no GPU needed)

The pooled results can be recomputed from the `.npz` files in a few seconds, with NumPy only:

```bash
python scripts/h1_aggregate.py results/h1_trust/pilot40 results/h1_trust/pilot20_s2 \
       results/h1_trust/pilot20_s3 --out /tmp/check
```

`/tmp/check/summary.json` comes out identical to `h1_trust/pooled_3seeds/summary.json`.

Main values in `pooled_3seeds/summary.json`:

- `auroc_stratified` gives the AUROC of each score: MC-dropout 0.86, distance moved 0.79, imagined cost 0.51, random 0.50.
- `auroc_by_run` gives AUROC per seed (MC-dropout 0.87, 0.84, 0.88).
- `offline_gate_success` gives the gate test results (no gate 71.3%, MC-dropout 70%, random 70%, oracle 72.5%).

The `"step"` entry is the step index used as a score. It is constant within each step, so its stratified
AUROC is exactly 0.5 by construction.
