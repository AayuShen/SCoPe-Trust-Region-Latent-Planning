"""Runtime speed/memory patches for DINO-WM that leave its predictions unchanged.

external/dino_wm is never edited; these replace methods on its classes at runtime
(used by scripts/plan_dino_wm.py and scripts/benchmark_dino_wm.py). Must be applied
from inside external/dino_wm so its `models` package is importable.

1. attention   The predictor computes attention by hand (matmul -> softmax -> matmul),
               materialising a 588x588 matrix per head per candidate. Swapped for torch's
               fused scaled_dot_product_attention with the same frame-causal mask, scale
               and attention dropout: equal up to float rounding, far less memory.
2. dedup       CEM gives every candidate the same start image, so the DINOv2 encoder runs
               on hundreds of identical frames. When every row of a batch is identical,
               encode one row and repeat it.
3. chunking    Split the candidate batch so peak memory fits the GPU. Candidates are
               independent, so outputs are unchanged. When chunking, rollout() returns
               (z_obses, None): the second value is unused by planning (see the comment
               in patch_chunked_rollout).

4. skip_solved MPC keeps running CEM for episodes that already succeeded, then discards
               their actions (zeroed by MPCPlanner._apply_success_mask). Skipping them
               changes no unsolved episode's planning problem; it only changes which
               random numbers the remaining episodes draw. Early-stop treats solved
               episodes as successes.

Optional, and NOT result-preserving (off by default): `precision="tf32"` lets fp32
matmuls use TF32 tensor cores; `precision="bf16"` runs rollouts under bf16 autocast.
Measure their effect with scripts/benchmark_dino_wm.py before using them.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from einops import rearrange

_originals: dict[str, object] = {}


def _classes():
    from models.visual_world_model import VWorldModel
    from models.vit import Attention

    return VWorldModel, Attention


def _sdpa_forward(self, x):
    _, t, _ = x.size()
    x = self.norm(x)
    q, k, v = (
        rearrange(z, "b n (h d) -> b h n d", h=self.heads)
        for z in self.to_qkv(x).chunk(3, dim=-1)
    )
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=self.bias[:, :, :t, :t].bool(),  # upstream: positions where bias == 0 get -inf
        dropout_p=self.dropout.p if self.dropout.training else 0.0,
        scale=self.scale,
    )
    return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


def patch_attention() -> None:
    _, Attention = _classes()
    _originals.setdefault("Attention.forward", Attention.forward)
    Attention.forward = _sdpa_forward


def patch_encoder_dedup() -> None:
    VWorldModel, _ = _classes()
    original = _originals.setdefault("VWorldModel.encode_obs", VWorldModel.encode_obs)

    def encode_obs(self, obs):
        batch = obs["visual"].shape[0]
        identical = batch > 1 and all(
            torch.equal(v, v[:1].expand_as(v)) for v in obs.values()
        )
        if not identical:
            return original(self, obs)
        one = original(self, {k: v[:1] for k, v in obs.items()})
        return {k: v.expand(batch, *v.shape[1:]) for k, v in one.items()}

    VWorldModel.encode_obs = encode_obs


def patch_inference_memory(decode_frames: int = 60) -> None:
    """encode_obs / decode_obs without autograd, and decode_obs in groups of ~60 frames.

    Upstream calls both with gradient tracking on although nothing differentiates
    through them in CEM/MPC planning (the planner only differentiates w.r.t. actions, and
    only for the GD planner, which reaches actions through encode_act). The evaluator
    encodes all n_evals end-of-round frames and decodes n_evals x (horizon+1) predicted
    frames to plot them, keeping every intermediate activation: harmless at 5 episodes,
    several GB at 50. Values are unchanged. decode_obs returns the decoder's second
    output from the first group only; the evaluator ignores it.
    """
    VWorldModel, _ = _classes()
    encode = _originals.setdefault("VWorldModel.encode_obs", VWorldModel.encode_obs)
    decode = _originals.setdefault("VWorldModel.decode_obs", VWorldModel.decode_obs)
    current_encode = VWorldModel.encode_obs  # may already be the dedup wrapper

    @torch.no_grad()
    def encode_obs(self, obs):
        return current_encode(self, obs)

    @torch.no_grad()
    def decode_obs(self, z_obs):
        batch, frames = z_obs["visual"].shape[:2]
        # The evaluator replays the whole action history every MPC round, so `frames`
        # grows by horizon+... each round (6 in round 1, 21 in round 4, 26 in round 5).
        # Group by frame count, not episodes, or a late round makes one huge batch.
        per_group = max(1, decode_frames // frames)
        if batch <= per_group:
            return decode(self, z_obs)
        parts = [
            decode(self, {k: v[i : i + per_group] for k, v in z_obs.items()})
            for i in range(0, batch, per_group)
        ]
        obs = {k: torch.cat([p[0][k] for p in parts]) for k in parts[0][0]}
        return obs, parts[0][1]

    VWorldModel.encode_obs = encode_obs
    VWorldModel.decode_obs = decode_obs


def patch_chunked_rollout(chunk: int, bf16: bool = False) -> None:
    VWorldModel, _ = _classes()
    original = _originals.setdefault("VWorldModel.rollout", VWorldModel.rollout)

    def run(self, obs_0, act):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            z_obses, z = original(self, obs_0, act)
        if bf16:  # hand fp32 back to the planner, as upstream does
            z_obses, z = {k: v.float() for k, v in z_obses.items()}, z.float()
        return z_obses, z

    def rollout(self, obs_0, act):
        batch = act.shape[0]
        if batch <= chunk:
            return run(self, obs_0, act)
        # Chunked path: results are written straight into preallocated outputs (no list
        # of parts plus a concatenated copy), and the second return value, the full
        # latent with action embeddings tiled in, is not built: CEM and the evaluator
        # discard it, and at 300 candidates it is ~0.75 GB that would otherwise stay
        # alive next to the previous step's outputs.
        z_obses = None
        for i in range(0, batch, chunk):
            part, _ = run(self, {k: v[i : i + chunk] for k, v in obs_0.items()}, act[i : i + chunk])
            if z_obses is None:
                z_obses = {k: v.new_empty((batch, *v.shape[1:])) for k, v in part.items()}
            for k, v in part.items():
                z_obses[k][i : i + chunk] = v
            del part
        return z_obses, None

    VWorldModel.rollout = rollout


def patch_skip_solved() -> None:
    import numpy as np
    from einops import repeat
    from planning.cem import CEMPlanner
    from planning.mpc import MPCPlanner
    from utils import move_to_device

    _originals.setdefault("CEMPlanner.plan", CEMPlanner.plan)
    _originals.setdefault("MPCPlanner.__init__", MPCPlanner.__init__)
    mpc_init = MPCPlanner.__init__

    def init(self, *args, **kwargs):
        mpc_init(self, *args, **kwargs)
        self.sub_planner.parent_mpc = self  # lets CEM see which episodes are solved

    def plan(self, obs_0, obs_g, actions=None):
        # Upstream CEMPlanner.plan (external/dino_wm/planning/cem.py), plus `solved`.
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        # Upstream encodes the goals with autograd on, so the graph (every DINOv2
        # activation of all n_evals goal images) stays alive for the whole plan: ~3 GB at
        # 50 episodes. CEM never differentiates through it; values are identical.
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        parent = getattr(self, "parent_mpc", None)
        solved = (
            np.asarray(parent.is_success, dtype=bool).copy()
            if parent is not None and parent.is_success is not None
            else np.zeros(n_evals, dtype=bool)
        )
        print(f"[scope] CEM: planning {int((~solved).sum())} of {n_evals} episodes "
              f"({int(solved.sum())} already solved, skipped); GPU holds "
              f"{torch.cuda.memory_allocated() / 2**20:.0f} MiB before the first rollout", flush=True)

        for i in range(self.opt_steps):
            step_start = time.perf_counter()
            losses = []
            for traj in range(n_evals):
                if solved[traj]:
                    continue
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(obs_0=cur_trans_obs_0, act=action)

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                del i_z_obses, i_zs, cur_trans_obs_0, cur_z_obs_g  # free before the next rollout
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1})
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                print(f"[scope] CEM step {i + 1}/{self.opt_steps}: "
                      f"{time.perf_counter() - step_start:.0f} s, "
                      f"GPU peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
                if np.all(np.logical_or(successes, solved)):
                    break  # terminate planning if every episode is solved

        return mu, np.full(n_evals, np.inf)  # all actions are valid

    MPCPlanner.__init__ = init
    CEMPlanner.plan = plan


def apply(chunk: int, attention: bool = True, dedup: bool = True, precision: str = "fp32",
          skip_solved: bool = False) -> None:
    if precision not in ("fp32", "tf32", "bf16"):
        raise ValueError(f"unknown precision {precision!r}")
    restore()
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    if attention:
        patch_attention()
    if dedup:
        patch_encoder_dedup()
    patch_inference_memory()
    patch_chunked_rollout(chunk, bf16=precision == "bf16")
    if skip_solved:
        patch_skip_solved()


def restore() -> None:
    from planning.cem import CEMPlanner
    from planning.mpc import MPCPlanner

    VWorldModel, Attention = _classes()
    owners = {"Attention": Attention, "VWorldModel": VWorldModel,
              "CEMPlanner": CEMPlanner, "MPCPlanner": MPCPlanner}
    for key, fn in _originals.items():
        cls, name = key.split(".")
        setattr(owners[cls], name, fn)
    _originals.clear()
    torch.backends.cuda.matmul.allow_tf32 = False  # torch default; upstream never enables it
