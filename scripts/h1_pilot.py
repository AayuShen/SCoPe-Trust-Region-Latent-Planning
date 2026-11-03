"""H1 pilot: do trust scores computed on imagined latents predict the real latent error?

For each Push-T decision point (a start state and a goal 25 simulator steps later, drawn from the
validation split the way DINO-WM's planner draws them) we run CEM in the released world model and
record a subsample of the candidate plans it evaluates. Every recorded candidate is then executed
in the simulator from the same start state (reset-and-execute), the real frames are encoded, and
the real latent error of each imagined step is compared with trust scores that need no training:

  step          the imagined step index k (trivial baseline: errors grow with k)
  dist_start    distance of the imagined latent from the start latent (trivial baseline)
  mc_dropout    variance of the imagined latent over stochastic passes, predictor dropout on
                (the released predictor was trained with dropout 0.1)
  obs_perturb   divergence of the rollout when the start image gets small noise (ACPC-style)
  act_perturb   divergence of the rollout when the actions get small noise
  knn           distance of the imagined latent to the nearest encoded training frames
  cost          the candidate's imagined cost to the goal (does the planner pick bad predictions?)
  random        control

Label: "unreliable" = real latent error in the top 10% for that imagined step. Metrics: AUROC
(stratified by step), Spearman correlation, bootstrap 95% CIs over decision points, and an
offline gate check: on each decision point's recorded candidates, which plan would plain
selection, a score-gated selection, a random gate and an oracle gate pick, and does it succeed?

Run through scripts/run_h1_pilot.sh (it activates the env and sets the working directory).
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, os.getcwd())  # external/dino_wm

import gym  # noqa: E402
import hydra  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import env  # noqa: E402,F401  (registers the gym environments)
from plan import load_model  # noqa: E402
from planning.objectives import create_objective_fn  # noqa: E402
from preprocessor import Preprocessor  # noqa: E402
from utils import seed as seed_all  # noqa: E402

from scope.world_model import dino_wm_patches  # noqa: E402

FRAMES = [0, 5, 10, 15, 20, 25]  # simulator steps that match imagined steps 0..5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cem-iters", type=int, default=10)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--record", default="1:48,5:48,10:96", help="iteration:count of candidates to record")
    p.add_argument("--mc", type=int, default=4, help="stochastic passes for MC-dropout")
    p.add_argument("--bank-trajs", type=int, default=300)
    p.add_argument("--bank-frames", type=int, default=15)
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--reject", type=float, default=0.3, help="fraction rejected by each gate")
    p.add_argument("--out", default=str(REPO / "outputs" / "runs" / "h1_pilot"))
    return p.parse_args()


# ---------------------------------------------------------------- metrics (numpy only)
def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return r


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


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("SCOPE_GPU_MEM_FRACTION", "0.75")))
    dino_wm_patches.apply(chunk=50, precision="tf32", skip_solved=False)
    device = torch.device("cuda")
    record = {int(i): int(n) for i, n in (t.split(":") for t in a.record.split(","))}

    ckpt = Path(os.environ.get("SCOPE_DATA_DIR", str(Path.home() / "scope_data"))) / "checkpoints" / "outputs" / "pusht"
    model_cfg = OmegaConf.load(ckpt / "hydra.yaml")
    seed_all(a.seed)
    _, traj = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist, num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    dset = traj["valid"]
    # load on the CPU first: the checkpoint payload would otherwise spike GPU memory while unpickling
    wm = load_model(ckpt / "checkpoints" / "model_latest.pth", model_cfg, model_cfg.num_action_repeat, device=torch.device("cpu"))
    wm = wm.to(device)
    # the ViT attention mask is a plain tensor attribute, not a buffer, so .to() leaves it behind
    for m in wm.modules():
        b = getattr(m, "bias", None)
        if isinstance(b, torch.Tensor) and not isinstance(b, torch.nn.Parameter):
            m.bias = b.to(device)
    wm.eval()  # VWorldModel.eval() returns None, so it cannot be chained
    fs = model_cfg.frameskip
    H = 5
    A = dset.action_dim * fs
    pre = Preprocessor(action_mean=dset.action_mean, action_std=dset.action_std, state_mean=dset.state_mean,
                       state_std=dset.state_std, proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
                       transform=dset.transform)
    objective = create_objective_fn(alpha=1, base=2, mode="last")
    sim = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    to_dev = lambda d: {k: v.to(device) for k, v in d.items()}  # noqa: E731

    @torch.no_grad()
    def rollout(obs0, actions):
        z, _ = wm.rollout(obs_0={k: v.expand(actions.shape[0], *v.shape[1:]) for k, v in obs0.items()}, act=actions)
        return z  # dict: visual (C, 6, 196, 384), proprio (C, 6, p)

    @torch.no_grad()
    def encode(obs):
        return wm.encode_obs(to_dev(pre.transform_obs(obs)))

    # -------- kNN bank: random projection of patch tokens of encoded training frames
    t0 = time.time()
    train = traj["train"]
    base = train.dataset if hasattr(train, "indices") else train
    ids = list(train.indices) if hasattr(train, "indices") else list(range(len(train)))
    rng = np.random.default_rng(a.seed)
    bank = []
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    proj = (torch.randn(196 * 384, a.proj_dim, generator=g) / np.sqrt(a.proj_dim)).to(device, torch.float16)
    for idx in rng.choice(ids, size=min(a.bank_trajs, len(ids)), replace=False):
        n = int(base.seq_lengths[idx])
        frames = sorted(rng.choice(n, size=min(a.bank_frames, n), replace=False).tolist())
        obs, _, _, _ = base.get_frames(int(idx), frames)
        with torch.no_grad():
            vis = wm.encode_obs({"visual": obs["visual"][None].to(device), "proprio": obs["proprio"][None].to(device)})["visual"][0]
        bank.append((vis.flatten(1).half() @ proj).float())
    bank = torch.cat(bank)
    print(f"[h1] kNN bank: {bank.shape[0]} training frames in {time.time() - t0:.0f} s", flush=True)

    def knn(vis):  # vis (N, 196, 384) -> mean distance to 5 nearest bank entries
        q = (vis.flatten(1).half() @ proj).float()
        return torch.cdist(q, bank).topk(5, largest=False).values.mean(1)

    def step_mse(x, y):  # (C, 6, P, D) -> (C, 5) for imagined steps 1..5
        return ((x[:, 1:] - y[:, 1:]) ** 2).mean(dim=(2, 3))

    rows = []
    for ep in range(a.episodes):
        t_ep = time.time()
        eval_seed = a.seed * ep + 1
        # ---- decision point: start state and goal 25 steps later, as plan.py does (goal_source dset)
        while True:
            tid = random.randint(0, len(dset) - 1)
            obs, act, state, info = dset[tid]
            if obs["visual"].shape[0] >= fs * H + 1:
                break
        off = random.randint(0, obs["visual"].shape[0] - (fs * H + 1))
        init_state = state.numpy()[off]
        sim.update_env(info)
        replay = pre.denormalize_actions(act[off: off + fs * H]).numpy()
        r_obs, r_states = sim.rollout(eval_seed, init_state, replay)
        state_g = r_states[-1]
        obs0 = to_dev(pre.transform_obs({k: v[None, None, 0] for k, v in r_obs.items()}))
        z_g = encode({k: v[None, None, -1] for k, v in r_obs.items()})

        # ---- CEM (upstream algorithm), recording candidates at chosen iterations
        mu = torch.zeros(H, A, device=device)
        sigma = torch.ones(H, A, device=device)
        recorded, rec_iter = [], []
        for it in range(1, a.cem_iters + 1):
            cand = torch.randn(a.samples, H, A, device=device) * sigma + mu
            cand[0] = mu
            z = rollout(obs0, cand)
            loss = objective(z, {k: v.expand(a.samples, *v.shape[1:]) for k, v in z_g.items()})
            elite = cand[torch.argsort(loss)[: a.topk]]
            mu, sigma = elite.mean(0), elite.std(0)
            if it in record:
                pick = torch.randperm(a.samples, device=device)[: record[it]]
                recorded.append(cand[pick])
                rec_iter += [it] * record[it]
        cand = torch.cat(recorded)
        C = cand.shape[0]

        # ---- imagined rollouts and trust scores
        zi = rollout(obs0, cand)
        cost = objective(zi, {k: v.expand(C, *v.shape[1:]) for k, v in z_g.items()})
        scores = {"step": np.tile(np.arange(1, H + 1, dtype=float), (C, 1))}
        scores["dist_start"] = ((zi["visual"][:, 1:] - zi["visual"][:, :1]) ** 2).mean(dim=(2, 3)).cpu().numpy()
        wm.predictor.train()  # dropout on: MC-dropout passes, variance accumulated without stacking
        s1 = s2 = 0
        for _ in range(a.mc):
            v = rollout(obs0, cand)["visual"][:, 1:]
            s1, s2 = s1 + v, s2 + v * v
        wm.predictor.eval()
        scores["mc_dropout"] = ((s2 - s1 * s1 / a.mc) / (a.mc - 1)).mean(dim=(2, 3)).cpu().numpy()
        del s1, s2, v
        noisy = {"visual": obs0["visual"] + 0.1 * torch.randn_like(obs0["visual"]), "proprio": obs0["proprio"]}
        scores["obs_perturb"] = step_mse(rollout(noisy, cand)["visual"], zi["visual"]).cpu().numpy()
        scores["act_perturb"] = step_mse(rollout(obs0, cand + 0.1 * torch.randn_like(cand))["visual"], zi["visual"]).cpu().numpy()
        scores["knn"] = knn(zi["visual"][:, 1:].reshape(C * H, 196, 384)).reshape(C, H).cpu().numpy()
        scores["cost"] = np.tile(cost.cpu().numpy()[:, None], (1, H))
        scores["random"] = np.random.rand(C, H)

        # ---- reset-and-execute every candidate, encode the real frames
        exec_acts = pre.denormalize_actions(rearrange(cand.cpu(), "c t (f d) -> c (t f) d", f=fs)).numpy()
        frames, success, dist = [], [], []
        for c in range(C):
            o, s = sim.rollout(eval_seed, init_state, exec_acts[c])
            frames.append({k: v[FRAMES] for k, v in o.items()})
            ev = sim.eval_state(state_g, s[-1])
            success.append(bool(ev["success"]))
            dist.append(float(ev["state_dist"]))
        zr_vis = []
        for i in range(0, C, 32):
            chunk = {k: np.stack([f[k] for f in frames[i:i + 32]]) for k in frames[0]}
            zr_vis.append(encode(chunk)["visual"])
        zr_vis = torch.cat(zr_vis)
        err = step_mse(zi["visual"], zr_vis).cpu().numpy()
        real_cost = ((zr_vis[:, -1:] - z_g["visual"]) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

        # final CEM mean plan, executed for real (sanity check against the Phase 0 curve)
        mean_exec = pre.denormalize_actions(rearrange(mu[None].cpu(), "c t (f d) -> c (t f) d", f=fs)).numpy()[0]
        _, s_mu = sim.rollout(eval_seed, init_state, mean_exec)
        mean_success = bool(sim.eval_state(state_g, s_mu[-1])["success"])

        rows.append({"err": err, "scores": scores, "cost": cost.cpu().numpy(), "real_cost": real_cost,
                     "success": np.array(success), "dist": np.array(dist), "iter": np.array(rec_iter), "mean_success": mean_success})
        print(f"[h1] episode {ep + 1}/{a.episodes}: {C} candidates, real success {np.mean(success):.2f}, "
              f"mean-plan success {mean_success}, median err {np.median(err):.4f}, {time.time() - t_ep:.0f} s", flush=True)
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- analysis
    names = list(rows[0]["scores"])
    E = len(rows)
    err_all = np.concatenate([r["err"] for r in rows])
    thr = np.percentile(err_all, 90, axis=0)  # per-step top-10% threshold

    def summarize(sel):
        e = np.concatenate([rows[i]["err"] for i in sel])
        lab = e >= thr
        res = {}
        for n in names:
            s = np.concatenate([rows[i]["scores"][n] for i in sel])
            res[n] = {"auroc": stratified(auroc, s, e, lab), "spearman": stratified(spearman, s, e, None)}
        return res

    main_res = summarize(range(E))
    boot = {n: [] for n in names}
    brng = np.random.default_rng(0)
    for _ in range(500):
        res = summarize(brng.integers(0, E, E))
        for n in names:
            boot[n].append(res[n]["auroc"])
    for n in names:
        lo, hi = np.nanpercentile(boot[n], [2.5, 97.5])
        main_res[n]["auroc_ci"] = [float(lo), float(hi)]

    # does the model rank candidates correctly? (imagined vs real cost, per decision point)
    rank_agree = float(np.mean([spearman(r["cost"], r["real_cost"]) for r in rows]))

    # offline gate check: pick the lowest imagined cost among accepted candidates
    def pick(r, trust):
        keep = np.ones(len(r["cost"]), bool) if trust is None else trust <= np.quantile(trust, 1 - a.reject)
        idx = np.flatnonzero(keep)[np.argmin(r["cost"][keep])]
        return r["success"][idx], r["dist"][idx]

    grng = np.random.default_rng(1)
    gates = {"plain (no gate)": lambda r: None, "oracle (true error)": lambda r: r["err"].max(1),
             "random gate": lambda r: grng.random(len(r["cost"]))}
    for n in ["mc_dropout", "obs_perturb", "act_perturb", "knn", "dist_start"]:
        gates[f"gate: {n}"] = (lambda nn: (lambda r: r["scores"][nn].max(1)))(n)
    gate_res = {}
    for g, fn in gates.items():
        picks = [pick(r, fn(r)) for r in rows]
        gate_res[g] = {"success": float(np.mean([p[0] for p in picks])), "state_dist": float(np.mean([p[1] for p in picks]))}

    summary = {
        "setup": {"episodes": E, "seed": a.seed, "cem_iters": a.cem_iters, "samples": a.samples, "topk": a.topk,
                  "recorded_per_episode": int(sum(record.values())), "pairs": int(err_all.size), "reject_fraction": a.reject,
                  "label": "real latent error in the top 10% for its imagined step"},
        "auroc_stratified": main_res,
        "rank_agreement_imagined_vs_real_cost": rank_agree,
        "mean_plan_success": float(np.mean([r["mean_success"] for r in rows])),
        "candidate_success_rate": float(np.mean(np.concatenate([r["success"] for r in rows]))),
        "error_by_step_median": [float(x) for x in np.median(err_all, axis=0)],
        "offline_gate": gate_res,
    }
    def clean(x):  # NaN (e.g. Spearman of a constant score) is not valid JSON
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return None if isinstance(x, float) and np.isnan(x) else x

    summary = clean(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / "per_candidate.npz", **{f"ep{i}_{k}": v for i, r in enumerate(rows) for k, v in
                                                     [("err", r["err"]), ("cost", r["cost"]), ("real_cost", r["real_cost"]),
                                                      ("success", r["success"]), ("dist", r["dist"]), ("iter", r["iter"])] +
                                                     [(f"score_{n}", r["scores"][n]) for n in names]})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
