"""Benchmark and validate the DINO-WM runtime patches on a released checkpoint.

Answers, before the patches are trusted for evaluation runs:
  1. Is dropout active in the predictor as loaded (plan.py never calls .eval())?
  2. How far do patched rollouts drift from the original ones? Compared in eval mode,
     against a noise floor: the original code against itself at another batch size.
     Drift is reported both in latent space and where it matters, in CEM's decision:
     overlap of the 30 elite candidates and change of the elite-mean action, using a
     real Push-T start frame, the frame 25 steps later as goal, and upstream's cost.
  3. How fast is a 300-candidate CEM rollout, and how much GPU memory does it need?

Run from external/dino_wm inside the `scope` env, e.g.
  python ../../scripts/benchmark_dino_wm.py --ckpt-dir ~/scope_data/checkpoints/outputs/pusht \
      --precisions fp32 tf32 bf16 --chunks 150

Results on the RTX 5060 Laptop (8 GB) are recorded in
docs/experiments/phase0_dino_wm_reproduction.md.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # see scripts/reproduce_dino_wm.sh
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scope.world_model import dino_wm_patches as patches  # noqa: E402


def load(ckpt_dir: Path, device):
    from plan import load_model

    cfg = OmegaConf.load(ckpt_dir / "hydra.yaml")
    model = load_model(ckpt_dir / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat, device)
    return model, cfg


def inputs(model, cfg, batch, horizon, device, seed=0):
    """A real Push-T start frame (repeated per candidate, as CEM does), the frame
    frameskip * horizon steps later as goal, and N(0, 1) candidate actions, which is
    CEM's first-iteration sampling distribution."""
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import PushTDataset

    ds = PushTDataset(
        n_rollout=1,
        transform=default_transform(img_size=cfg.img_size),
        data_path=os.path.join(os.environ["DATASET_DIR"], "pusht_noise", "val"),
        normalize_action=cfg.get("normalize_action", True),
    )
    frames, *_ = ds.get_frames(0, [0, cfg.frameskip * horizon])
    start = {k: v[None, :1].expand(batch, *([-1] * v.dim())) for k, v in frames.items()}
    goal = {k: v[None, 1:2] for k, v in frames.items()}
    to = lambda d: {k: v.float().to(device) for k, v in d.items()}  # noqa: E731
    return to(start), to(goal)


def candidates(model, batch, horizon, sigma, device, seed=0):
    """CEM-style samples mu + sigma * N(0, 1). sigma=1 matches CEM's first iteration;
    small sigma mimics late iterations, where candidates are near-identical and the
    ranking is most sensitive to numerical drift."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dim = model.action_encoder.patch_embed.in_channels
    mu = torch.randn(1, horizon, dim, generator=g)
    act = mu + sigma * torch.randn(batch, horizon, dim, generator=g)
    act[0] = mu[0]  # upstream also keeps the current mean as candidate 0
    return act.to(device)


def cem_decision(z_pred, z_goal, act, objective, topk=30):
    """Upstream CEM's cost per candidate, its elite set, and its elite-mean action."""
    goal = {k: v.expand(z_pred[k].shape[0], *v.shape[1:]) for k, v in z_goal.items()}
    cost = objective(z_pred, goal)
    elite = torch.argsort(cost)[:topk]
    return set(elite.tolist()), act[elite].mean(dim=0)


def decision_drift(ref, new, topk=30):
    overlap = len(ref[0] & new[0]) / topk
    shift = float((ref[1] - new[1]).norm() / ref[1].norm())
    return overlap, shift


def fresh_gpu():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def rollout_cpu(model, obs, act):
    fresh_gpu()
    with torch.no_grad():
        z_obses, _ = model.rollout(obs, act)
    return {k: v.float().cpu() for k, v in z_obses.items()}  # keep GPU memory free between tests


def drift(ref, new):
    """Largest absolute and relative difference across all predicted latents."""
    worst_abs = worst_rel = 0.0
    for k in ref:
        d = float((ref[k] - new[k]).abs().max())
        worst_abs = max(worst_abs, d)
        worst_rel = max(worst_rel, d / float(ref[k].abs().max()))
    return worst_abs, worst_rel


def timed(model, obs, act, reps):
    fresh_gpu()
    with torch.no_grad():
        model.rollout(obs, act)  # warm-up
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            model.rollout(obs, act)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps, torch.cuda.max_memory_allocated() / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, default=300)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--precisions", nargs="+", default=["fp32"], choices=["fp32", "tf32", "bf16"])
    p.add_argument("--chunks", nargs="+", type=int, default=[50, 100, 150, 300])
    p.add_argument("--sigmas", nargs="+", type=float, default=[1.0],
                   help="candidate spread(s) for the equivalence check; 1.0 = CEM's first iteration")
    p.add_argument("--skip-timing", action="store_true", help="equivalence checks only")
    p.add_argument("--mem-fraction", type=float, default=0.85,
                   help="cap PyTorch below physical VRAM so overflow raises OOM instead of "
                        "silently spilling into (much slower) shared system memory on Windows")
    args = p.parse_args()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    from planning.objectives import create_objective_fn

    model, cfg = load(args.ckpt_dir.expanduser(), device)
    obs, goal = inputs(model, cfg, args.batch, args.horizon, device)
    objective = create_objective_fn(alpha=1, base=2, mode="last")  # conf/plan_pusht.yaml

    active = [n for n, m in model.named_modules()
              if isinstance(m, torch.nn.Dropout) and m.p > 0 and m.training]
    print(f"[dropout] predictor.training={model.predictor.training}; "
          f"active dropout layers: {len(active)}", flush=True)

    flags = {n: m.training for n, m in model.named_modules()}
    model.eval()
    patches.apply(chunk=50, attention=False, dedup=False)
    with torch.no_grad():
        z_goal = {k: v.cpu() for k, v in model.encode_obs(goal).items()}

    # Equivalence: reference is the unmodified model (only chunked to fit), fp32.
    variants = [("noise floor: original, chunk 25", dict(chunk=25, attention=False, dedup=False))]
    variants += [(f"patched {p}, chunk {c}", dict(chunk=c, precision=p))
                 for p in args.precisions for c in args.chunks]
    for sigma in args.sigmas:
        act = candidates(model, args.batch, args.horizon, sigma, device)
        act_cpu = act.cpu()
        patches.apply(chunk=50, attention=False, dedup=False)
        ref = rollout_cpu(model, obs, act)
        ref_decision = cem_decision(ref, z_goal, act_cpu, objective)
        for name, kwargs in variants:
            patches.apply(**kwargs)
            try:
                z = rollout_cpu(model, obs, act)
            except torch.OutOfMemoryError:
                fresh_gpu()
                print(f"[sigma {sigma}] {name}: out of memory", flush=True)
                continue
            _, d_rel = drift(ref, z)
            overlap, shift = decision_drift(ref_decision, cem_decision(z, z_goal, act_cpu, objective))
            print(f"[sigma {sigma}] {name}: latent drift {d_rel:.2e}; elite overlap {overlap:.0%}; "
                  f"elite-mean action change {shift:.3%}", flush=True)
        del ref

    # Speed and memory.
    if not args.skip_timing:
        act = candidates(model, args.batch, args.horizon, 1.0, device)
        for name, kwargs in [("original, chunk 50", dict(chunk=50, attention=False, dedup=False))] + variants[1:]:
            patches.apply(**kwargs)
            try:
                sec, mib = timed(model, obs, act, args.reps)
            except torch.OutOfMemoryError:
                fresh_gpu()
                print(f"[speed] {name}: out of memory", flush=True)
                continue
            print(f"[speed] {name}: {sec * 1000:.0f} ms per {args.batch}-candidate rollout, "
                  f"peak {mib:.0f} MiB", flush=True)

    for n, m in model.named_modules():
        m.training = flags[n]
    patches.restore()


if __name__ == "__main__":
    main()
