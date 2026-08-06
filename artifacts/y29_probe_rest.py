#!/usr/bin/env python3
"""Probe the G1 skeleton rest pose and find clean arm abduction."""

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
RSP, RSR, RSY, REL = 26, 27, 28, 29
RWW, RHR = 32, 33

ARM_L = [18, 19, 20, 21, 22, 23, 24, 25]
ARM_R = [26, 27, 28, 29, 30, 31, 32, 33]


def main() -> None:
    model, _ = load_model("kimodo-g1-rp", device="cpu", return_resolved_name=True)
    sk = model.skeleton
    nj = sk.neutral_joints.cpu().numpy() if torch.is_tensor(sk.neutral_joints) else np.asarray(sk.neutral_joints)
    print("neutral_joints shape", nj.shape)
    print("has rest_local_rots:", getattr(sk, "rest_local_rots", None) is not None)
    print("arm-chain neutral positions (x,y,z):")
    for i in ARM_L + ARM_R + [PELVIS]:
        print(f"  {i:2d} {sk.bone_order_names[i]:24s} {nj[i]}")
    print("root_idx", sk.root_idx)
    print()

    N = nj.shape[0]
    ident = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).unsqueeze(0)  # (1,N,3,3)
    root0 = torch.zeros(1, 3)

    def report(pose, root, tag):
        gr, gp, _ = sk.fk(pose, root)
        g = gp[0].numpy()
        p = g[PELVIS]
        for s, w, h in (("L", LWW, LHR), ("R", RWW, RHR)):
            print(f"  {tag} {s} wrist_rel=(x={g[w,0]-p[0]:+.3f} y={g[w,1]-p[1]:+.3f} z={g[w,2]-p[2]:+.3f}) "
                  f"hand_z={g[h,2]-p[2]:+.3f} |wrist_y|={abs(g[w,1]-p[1]):.3f}")
        return g

    print("rest pose (identity local):")
    report(ident, root0, "rest")
    print()

    # Try abduction by rotating shoulder_pitch/roll about each axis, both signs, big angle
    for jidx, name in ((LSP, "L_shoulder_pitch"), (LSR, "L_shoulder_roll"), (LSY, "L_shoulder_yaw")):
        for ax in ("x", "y", "z"):
            for ang in (-1.4, -0.7, 0.7, 1.4):
                pose = ident.clone()
                extra = AXES[ax](ang)
                pose[0, jidx] = pose[0, jidx] @ torch.tensor(extra, dtype=torch.float32)
                gr, gp, _ = sk.fk(pose, root0)
                g = gp[0].numpy()
                p = g[PELVIS]
                wy = abs(g[LWW, 1] - p[1])
                wz = g[LWW, 2] - p[2]
                if wy > 0.15 and wz > 0.1:
                    print(f"PROMISING {name} {ax} {ang:+.2f} -> |wy|={wy:.3f} wz={wz:.3f}")
    print("done")


if __name__ == "__main__":
    main()
