"""Run DINO-WM's plan.py with the runtime patches from scope.world_model.dino_wm_patches.

Same as `python plan.py ...` (same Hydra arguments), but rollouts use fused attention,
encode a repeated start frame once, and run in chunks that fit an 8 GB GPU. Validated
against the unpatched model with scripts/benchmark_dino_wm.py; results are recorded in
docs/experiments/phase0_dino_wm_reproduction.md.

Run from external/dino_wm. Environment overrides:
  SCOPE_ROLLOUT_CHUNK     candidates per rollout chunk (default 50: same speed as 150 in
                          the benchmark, ~1 GB less peak, which the full planner needs)
  SCOPE_PRECISION         fp32 | tf32 | bf16 (default tf32: identical CEM decisions in
                          the benchmark, 3.8x faster; fp32 for strictest fidelity)
  SCOPE_GPU_MEM_FRACTION  cap on PyTorch's share of VRAM (default 0.75), so overflow
                          raises OOM instead of Windows silently spilling it into system
                          RAM, which on this 15 GB laptop can hang WSL and Windows
  SCOPE_SKIP_SOLVED       1 (default) = MPC stops running CEM for episodes that already
                          succeeded; 0 = upstream behaviour
  SCOPE_SERIAL_ENVS       1 (default) = run the simulators one after another in this
                          process (upstream does this for Wall) instead of one forked
                          process each. Fifty forked Push-T simulators take ~130 MB each
                          and exhaust the 7.5 GB WSL VM within seconds, killing the run.
                          Simulation results are unchanged; only wall-clock differs.
"""

import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())

from scope.world_model import dino_wm_patches  # noqa: E402


def main() -> None:
    chunk = int(os.environ.get("SCOPE_ROLLOUT_CHUNK", "50"))
    precision = os.environ.get("SCOPE_PRECISION", "tf32")
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("SCOPE_GPU_MEM_FRACTION", "0.75")))
    skip_solved = os.environ.get("SCOPE_SKIP_SOLVED", "1") == "1"
    dino_wm_patches.apply(chunk=chunk, precision=precision, skip_solved=skip_solved)
    print(f"[scope] DINO-WM patches: fused attention, encoder dedup, chunk {chunk}, {precision}, "
          f"skip_solved={skip_solved}", flush=True)
    if os.environ.get("SCOPE_SERIAL_ENVS", "1") == "1":
        import env.venv
        from env.serial_vector_env import SerialVectorEnv

        def serial_vector_env(env_fns, **_):
            return SerialVectorEnv([make() for make in env_fns])

        env.venv.SubprocVectorEnv = serial_vector_env  # plan.py imports this name at start-up
        print("[scope] simulators: serial, in-process", flush=True)
    sys.argv[0] = "plan.py"
    runpy.run_path("plan.py", run_name="__main__")


if __name__ == "__main__":
    main()
