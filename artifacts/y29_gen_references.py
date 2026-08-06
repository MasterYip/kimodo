#!/usr/bin/env python3
"""Y29 stage 1: generate the 8 per-direction reference motions deterministically.

Each reference is used ONLY to derive the per-direction hand EndEffector
constraint (hand target positions + root trajectory). References are generated
with the same prompts as Y28, same seed base 20260805 + direction index, same
6s/30fps/100 steps / separated cfg 2.0/2.0. Root2D disabled (no constraint).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kimodo import load_model
from kimodo.tools import seed_everything

OUT_DIR = Path("/data/masteryip/kimodo/kimodo-agent-worktrees/DATA-KIMODO-LOCOMOTION-YAML-004/artifacts/y29_references")
DURATION_S = 6.0
FPS = 30
SEED_BASE = 20260805
DIFFUSION_STEPS = 100
CFG = [2.0, 2.0]

DIRECTIONS = [
    ("d00_forward", "A person walks forward with a normal walking gait."),
    ("d01_forward_right", "A person walks diagonally forward and right with a normal walking gait."),
    ("d02_right", "A person walks to the right with a normal walking gait."),
    ("d03_backward_right", "A person walks diagonally backward and right with a normal walking gait."),
    ("d04_backward", "A person walks backward with a normal walking gait."),
    ("d05_backward_left", "A person walks diagonally backward and left with a normal walking gait."),
    ("d06_left", "A person walks to the left with a normal walking gait."),
    ("d07_forward_left", "A person walks diagonally forward and left with a normal walking gait."),
]

# Kimodo world frame: Y = up. Lateral/left in rest pelvis frame = +X, forward = +Z.
PELVIS = 0
L_WRIST = 24
R_WRIST = 32


def main() -> None:
    model, name = load_model("kimodo-g1-rp", device="cuda:0", return_resolved_name=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = int(round(DURATION_S * FPS))
    meta_rows = []
    for i, (dname, prompt) in enumerate(DIRECTIONS):
        out_npz = OUT_DIR / f"{dname}.npz"
        seed = SEED_BASE + i
        seed_everything(seed, deterministic=True)
        out = model(
            [prompt],
            [frames],
            num_denoising_steps=DIFFUSION_STEPS,
            cfg_type="separated",
            cfg_weight=CFG,
            return_numpy=True,
        )
        posed = np.asarray(out["posed_joints"])[0]
        local_rot = np.asarray(out["local_rot_mats"])[0]
        global_rot = np.asarray(out["global_rot_mats"])[0]
        root_pos = np.asarray(out["root_positions"])[0]
        np.savez_compressed(
            out_npz,
            posed_joints=posed.astype(np.float32),
            local_rot_mats=local_rot.astype(np.float32),
            global_rot_mats=global_rot.astype(np.float32),
            root_positions=root_pos.astype(np.float32),
        )
        pelvis = posed[:, PELVIS]
        disp = float(np.linalg.norm(root_pos[-1, [0, 2]] - root_pos[0, [0, 2]]))
        ph = float(np.median(pelvis[:, 1]))
        meta_rows.append({
            "direction": dname, "index": i, "prompt": prompt, "seed": seed,
            "frames": frames, "root_2d_travel_m": disp, "pelvis_height_median_m": ph,
        })
        print(f"[{i}] {dname} seed={seed} frames={frames} root_travel={disp:.3f} pelvis_h={ph:.3f} -> {out_npz.name}")
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "model": name, "duration_s": DURATION_S, "fps": FPS,
        "diffusion_steps": DIFFUSION_STEPS, "cfg": CFG, "seed_base": SEED_BASE,
        "note": "Per-direction references for Y29 hand-constraint derivation only.",
        "directions": meta_rows,
    }, indent=2) + "\n")
    print(f"ALL_REFERENCES_SAVED to {OUT_DIR}")


if __name__ == "__main__":
    main()
