#!/usr/bin/env python3
"""Probe: find the local-frame rotation(s) that extend the G1 arm laterally.

Loads the saved reference local_rot_mats, applies candidate extra rotations to
arm joints, FKs, and reports hand positions relative to pelvis.
"""

from __future__ import annotations

import numpy as np
import torch

from kimodo import load_model


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


AXES = {"x": rot_x, "y": rot_y, "z": rot_z}

# G1Skeleton34 indices
PELVIS = 0
L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW = 18, 19, 20
L_ELBOW = 21
L_WRIST_YAW, L_HAND_ROLL = 24, 25
R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW = 26, 27, 28
R_ELBOW = 29
R_WRIST_YAW, R_HAND_ROLL = 32, 33

REF = ("/data/masteryip/kimodo/kimodo-agent-worktrees/DATA-KIMODO-LOCOMOTION-"
       "YAML-004/artifacts/y29_reference/reference.npz")


def main() -> None:
    model, _ = load_model("kimodo-g1-rp", device="cpu", return_resolved_name=True)
    sk = model.skeleton
    ref = np.load(REF)
    local_rot = ref["local_rot_mats"]  # (T,34,3,3)
    root_pos = ref["root_positions"]   # (T,3)
    t = local_rot.shape[0] // 2
    L = torch.tensor(local_rot, dtype=torch.float32)
    R = torch.tensor(root_pos, dtype=torch.float32)
    base_pose = L[t : t + 1]   # (1,34,3,3)
    base_root = R[t : t + 1]

    def fk_report(pose, root, tag):
        gr, gp, _ = sk.fk(pose, root)  # gp: (1,34,3)
        g = gp[0].numpy()
        pelvis = g[PELVIS]
        for side, wi, hi in (("L", L_WRIST_YAW, L_HAND_ROLL), ("R", R_WRIST_YAW, R_HAND_ROLL)):
            wy = abs(g[wi, 1] - pelvis[1])
            hz = g[hi, 2] - pelvis[2]
            wx = g[wi, 0] - pelvis[0]
            wz = g[wi, 2] - pelvis[2]
            print(f"  {tag} {side} wrist |y|={wy:.3f} wrist(z-pelvis)={wz:.3f} "
                  f"wrist(x-pelvis)={wx:.3f} hand(z-pelvis)={hz:.3f}")
        return g

    # Baseline (no modification)
    print("baseline:")
    fk_report(base_pose, base_root, "base")
    print()

    # Candidate single-joint rotations
    cands = [
        ("L_shoulder_pitch", L_SHOULDER_PITCH, "x", 0.5), ("L_shoulder_pitch", L_SHOULDER_PITCH, "x", 1.0),
        ("L_shoulder_pitch", L_SHOULDER_PITCH, "y", 0.5), ("L_shoulder_pitch", L_SHOULDER_PITCH, "y", 1.0),
        ("L_shoulder_pitch", L_SHOULDER_PITCH, "z", 0.5), ("L_shoulder_pitch", L_SHOULDER_PITCH, "z", 1.0),
        ("L_shoulder_roll", L_SHOULDER_ROLL, "x", 0.5), ("L_shoulder_roll", L_SHOULDER_ROLL, "x", 1.0),
        ("L_shoulder_roll", L_SHOULDER_ROLL, "y", 0.5), ("L_shoulder_roll", L_SHOULDER_ROLL, "y", 1.0),
        ("L_shoulder_roll", L_SHOULDER_ROLL, "z", 0.5), ("L_shoulder_roll", L_SHOULDER_ROLL, "z", 1.0),
        ("L_shoulder_yaw", L_SHOULDER_YAW, "x", 0.5), ("L_shoulder_yaw", L_SHOULDER_YAW, "y", 0.5),
        ("L_shoulder_yaw", L_SHOULDER_YAW, "z", 0.5),
        ("L_elbow", L_ELBOW, "x", 0.5), ("L_elbow", L_ELBOW, "y", 0.5), ("L_elbow", L_ELBOW, "z", 0.5),
        ("L_shoulder_pitch", L_SHOULDER_PITCH, "x", -0.5), ("L_shoulder_roll", L_SHOULDER_ROLL, "y", -0.5),
    ]
    for name, jidx, ax, ang in cands:
        pose = base_pose.clone()
        extra = AXES[ax](ang)
        pose[0, jidx] = pose[0, jidx] @ torch.tensor(extra, dtype=torch.float32)
        print(f"candidate {name} {ax}+{ang:.2f}:")
        fk_report(pose, base_root, f"{name}{ax}{ang:+.2f}")


if __name__ == "__main__":
    main()
