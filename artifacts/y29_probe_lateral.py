#!/usr/bin/env python3
"""Grid-search arm-chain rotations that put the hand laterally out (rest frame).

Rest frame convention: +X = left (lateral), +Y = up, +Z = forward.
Goal: hand lateral |x| > 0.25 m, hand height y in [0.15, 0.50] m above pelvis.
"""

from __future__ import annotations

import numpy as np
import torch

from kimodo import load_model


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


AXES = {"x": rot_x, "y": rot_y, "z": rot_z}

PELVIS = 0
LSP, LSR, LSY, LEL = 18, 19, 20, 21
LWW, LHR = 24, 25


def main() -> None:
    model, _ = load_model("kimodo-g1-rp", device="cpu", return_resolved_name=True)
    sk = model.skeleton
    N = sk.neutral_joints.shape[0]
    ident = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).unsqueeze(0)
    root0 = torch.zeros(1, 3)

    def hand_rel(joint_rots):
        gr, gp, _ = sk.fk(joint_rots, root0)
        g = gp[0].numpy()
        p = g[PELVIS]
        return g[LWW], g[LHR], p

    best = []
    for sp_ax in ("x", "y", "z"):
        for sp_ang in np.arange(-1.6, 1.61, 0.2):
            for sr_ax in ("x", "y", "z"):
                for sr_ang in np.arange(-1.6, 1.61, 0.2):
                    pose = ident.clone()
                    if abs(sp_ang) > 1e-6:
                        pose[0, LSP] = pose[0, LSP] @ torch.tensor(AXES[sp_ax](sp_ang), dtype=torch.float32)
                    if abs(sr_ang) > 1e-6:
                        pose[0, LSR] = pose[0, LSR] @ torch.tensor(AXES[sr_ax](sr_ang), dtype=torch.float32)
                    w, h, p = hand_rel(pose)
                    lat = w[0] - p[0]      # lateral (left positive)
                    up = w[1] - p[1]       # height above pelvis
                    fwd = w[2] - p[2]      # forward
                    if abs(lat) > 0.25 and 0.15 <= up <= 0.50 and abs(fwd) < 0.15:
                        best.append((abs(lat), lat, up, fwd, sp_ax, sp_ang, sr_ax, sr_ang))
    best.sort(key=lambda r: (-r[0], abs(r[3])))
    print(f"found {len(best)} lateral-out configs")
    for b in best[:15]:
        print(f"  |lat|={b[0]:.3f} lat={b[1]:+.3f} up={b[2]:+.3f} fwd={b[3]:+.3f}  "
              f"shoulder_pitch_{b[4]}={b[5]:+.2f}  shoulder_roll_{b[6]}={b[7]:+.2f}")


if __name__ == "__main__":
    main()
