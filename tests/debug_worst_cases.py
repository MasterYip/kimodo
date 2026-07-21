#!/usr/bin/env python3
"""Test: generate the worst-case velocities from the orchestrator using
the clean single-sample method to determine if jitter is velocity-specific."""

import numpy as np
import os
import sys
import time
from pathlib import Path
from scipy.ndimage import uniform_filter1d

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from locomotion_framework.constraints import build_root2d_constraint

FPS = 30
STEPS = 100
DEVICE = "cuda:0"
MODEL = "kimodo-g1-rp"

# Worst-case velocities from the orchestrator demo test
WORST_CASES = [
    {"name": "debug_walk_fwd_027", "prompt": "a robot walks normally",
     "vel": {"vx": 0.5714, "vy": 0.2887, "wz": 0.405}, "dur": 5.71},
    {"name": "debug_walk_lat_321", "prompt": "a robot walks normally",
     "vel": {"vx": 0.4598, "vy": -0.3681, "wz": -0.3972}, "dur": 5.27},
    {"name": "debug_run_fwd_005", "prompt": "a robot runs at a brisk pace normally",
     "vel": {"vx": 1.0392, "vy": 0.0948, "wz": 0.0588}, "dur": 5.89},
]


def jitter_score(vel):
    smooth = uniform_filter1d(np.asarray(vel, dtype=np.float64), 5)
    return float(np.sqrt(np.mean((vel - smooth) ** 2)))


def analyze(posed, fps, label):
    root = posed[:, 0, :]
    vel = np.linalg.norm(np.diff(root, axis=0), axis=-1) * fps
    T = len(vel)
    third = T // 3
    jh = jitter_score(vel[:third])
    jt = jitter_score(vel[2 * third:])
    return {
        "label": label,
        "head_v": float(np.mean(vel[:30])),
        "mid_v": float(np.mean(vel[third: 2 * third])),
        "tail_v": float(np.mean(vel[-30:])),
        "t_m": float(np.mean(vel[-30:]) / (np.mean(vel[third: 2 * third]) + 1e-6)),
        "jit_h": jh,
        "jit_t": jt,
        "jit_th": jt / (jh + 1e-6),
    }


def main():
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst

    print("=" * 70)
    print("Testing worst-case velocities via clean single-sample method")
    print("=" * 70)

    model, _ = load_model(MODEL, device=DEVICE, return_resolved_name=True)

    for tc in WORST_CASES:
        print(f"\n-- {tc['name']}: vx={tc['vel']['vx']:.3f} "
              f"vy={tc['vel']['vy']:.3f} wz={tc['vel']['wz']:.3f} "
              f"dur={tc['dur']:.1f} --")

        nf = int(tc["dur"] * FPS)
        constraint = build_root2d_constraint(tc["vel"], tc["dur"], fps=FPS, stride=1)
        kc = load_constraints_lst([constraint], model.skeleton, device=DEVICE)

        # Single, no CFG
        t0 = time.time()
        out = model(
            [tc["prompt"]], [nf],
            constraint_lst=[kc],
            num_denoising_steps=STEPS,
            return_numpy=True,
        )
        dt = time.time() - t0
        posed = np.asarray(out["posed_joints"][0])
        r = analyze(posed, FPS, f"{tc['name']}_nocfg")
        print(f"  single no-CFG: T/M={r['t_m']:.2f} JitT/H={r['jit_th']:.1f} "
              f"head={r['head_v']:.3f} mid={r['mid_v']:.3f} tail={r['tail_v']:.3f} ({dt:.1f}s)")

        # Single, CFG [2,2] (full demo method)
        t0 = time.time()
        out2 = model(
            [tc["prompt"]], [nf],
            constraint_lst=[kc],
            num_denoising_steps=STEPS,
            cfg_weight=[2.0, 2.0],
            return_numpy=True,
        )
        dt2 = time.time() - t0
        posed2 = np.asarray(out2["posed_joints"][0])
        r2 = analyze(posed2, FPS, f"{tc['name']}_cfg")
        print(f"  single CFG:   T/M={r2['t_m']:.2f} JitT/H={r2['jit_th']:.1f} "
              f"head={r2['head_v']:.3f} mid={r2['mid_v']:.3f} tail={r2['tail_v']:.3f} ({dt2:.1f}s)")

    del model
    import torch
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
