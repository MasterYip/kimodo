#!/usr/bin/env python3
"""Y29 step 1: generate an arms-out reference motion with the kimodo-g1-rp model.

The reference motion is used ONLY to derive the hand EndEffector constraint arm
pose (local joint rotations) for the Y29 batch. It is not a Y29 sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kimodo import load_model
from kimodo.tools import seed_everything

OUT_DIR = Path("/data/masteryip/kimodo/kimodo-agent-worktrees/DATA-KIMODO-LOCOMOTION-YAML-004/artifacts/y29_reference")
PROMPT = ("A person walks forward at a normal walking gait with both arms stretched "
          "straight out horizontally to the sides at shoulder height, elbows locked, "
          "hands open, palms facing down, like a T-pose walk.")
DURATION_S = 6.0
FPS = 30
SEED = 20260805
DIFFUSION_STEPS = 100
CFG = [2.0, 2.0]


def main() -> None:
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else PROMPT
    # Launched with CUDA_VISIBLE_DEVICES=1 so physical GPU 1 is cuda:0.
    model, name = load_model("kimodo-g1-rp", device="cuda:0", return_resolved_name=True)
    seed_everything(SEED, deterministic=True)
    frames = int(round(DURATION_S * FPS))
    out = model(
        [prompt],
        [frames],
        num_denoising_steps=DIFFUSION_STEPS,
        cfg_type="separated",
        cfg_weight=CFG,
        return_numpy=True,
    )
    # G1Skeleton34 indices
    PELVIS = 0
    L_WRIST = 24   # left_wrist_yaw_skel
    R_WRIST = 32   # right_wrist_yaw_skel
    posed = np.asarray(out["posed_joints"])[0]          # (T, 34, 3)
    local_rot = np.asarray(out["local_rot_mats"])[0]    # (T, 34, 3, 3)
    global_rot = np.asarray(out["global_rot_mats"])[0]  # (T, 34, 3, 3)
    root_pos = np.asarray(out["root_positions"])[0]     # (T, 3)
    fps = float(np.asarray(out.get("fps", [[FPS]])).reshape(-1)[0])
    T = posed.shape[0]
    print(f"model={name} frames={T} fps={fps}")

    # Pelvis-frame hand |y| (top-down proxy)
    pelvis = posed[:, PELVIS]
    lw = posed[:, L_WRIST]
    rw = posed[:, R_WRIST]
    l_abs_y = np.abs(lw[:, 1] - pelvis[:, 1])
    r_abs_y = np.abs(rw[:, 1] - pelvis[:, 1])
    median = float(np.median(np.concatenate([l_abs_y, r_abs_y])))
    print(f"left wrist |y| median={float(np.median(l_abs_y)):.4f} p05={float(np.quantile(l_abs_y,0.05)):.4f}")
    print(f"right wrist |y| median={float(np.median(r_abs_y)):.4f} p05={float(np.quantile(r_abs_y,0.05)):.4f}")
    print(f"combined |y| median={median:.4f}")
    # height
    print(f"pelvis height median={float(np.median(pelvis[:,1])):.4f}")
    # root travel
    disp = np.linalg.norm(root_pos[-1, [0, 2]] - root_pos[0, [0, 2]])
    print(f"root 2D travel={disp:.3f} m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "reference.npz",
        posed_joints=posed.astype(np.float32),
        local_rot_mats=local_rot.astype(np.float32),
        global_rot_mats=global_rot.astype(np.float32),
        root_positions=root_pos.astype(np.float32),
    )
    meta = {
        "prompt": prompt,
        "duration_s": DURATION_S,
        "fps": FPS,
        "seed": SEED,
        "diffusion_steps": DIFFUSION_STEPS,
        "cfg": CFG,
        "model": name,
        "hand_y_median_m": median,
        "left_wrist_y_median_m": float(np.median(l_abs_y)),
        "right_wrist_y_median_m": float(np.median(r_abs_y)),
        "pelvis_height_median_m": float(np.median(pelvis[:, 1])),
        "root_2d_travel_m": float(disp),
        "note": ("Reference motion used only to derive the Y29 hand-constraint "
                 "arm pose; not a Y29 sample."),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved reference to {OUT_DIR / 'reference.npz'}")
    ok = median > 0.20
    print(f"HAND_CLEARANCE={'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
