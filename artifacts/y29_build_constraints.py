#!/usr/bin/env python3
"""Y29 stage 2: build per-direction hand EndEffector constraint JSONs.

Loads the per-direction references (stage 1), and for each keyframe orients a
straight arm (shoulder_pitch_z, shoulder_roll_y) so the wrist lands at a
pelvis-frame target that is laterally clear of the hips/crotch:
    target_wrist = pelvis_pos + R_pelvis @ (sign*LAT_TARGET, UP_TARGET, FWD_TARGET)

A per-keyframe batched grid search over (shoulder_pitch_z, shoulder_roll_y)
picks the angles whose FK wrist position is closest to the target. The chosen
pose's full local rotations are FK'd and serialized into the native
left-hand / right-hand constraint dicts consumed by load_constraints_lst.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.geometry import matrix_to_axis_angle

REF_DIR = Path("/data/masteryip/kimodo/kimodo-agent-worktrees/DATA-KIMODO-LOCOMOTION-YAML-004/artifacts/y29_references")
OUT_DIR = Path("/data/masteryip/kimodo/kimodo-agent-worktrees/DATA-KIMODO-LOCOMOTION-YAML-004/artifacts/y29_constraints")
KEYFRAME_STRIDE = 10
KEYFRAME_MAX = 180
MIN_LATERAL = 0.20

LAT_TARGET = 0.32   # pelvis-frame lateral offset target (m)
UP_TARGET = 0.30    # pelvis-frame height target (m, above pelvis -> above crotch)
FWD_TARGET = 0.02   # pelvis-frame forward target (m)

PELVIS = 0
LSP, LSR = 18, 19
LW_Y = 24
RSP, RSR = 26, 27
RW_Y = 32
ARM_L = [18, 19, 20, 21, 22, 23, 24, 25]
ARM_R = [26, 27, 28, 29, 30, 31, 32, 33]

# Candidate grid: shoulder_pitch about local z, shoulder_roll about local y.
SP_GRID = np.linspace(-1.2, 1.2, 25)
SR_GRID = np.linspace(-2.4, 2.4, 25)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main() -> None:
    model, name = load_model("kimodo-g1-rp", device="cpu", return_resolved_name=True)
    sk = model.skeleton

    # Pre-build candidate local-rotation patches for the arm chain.
    eye = np.eye(3)
    n_cand = len(SP_GRID) * len(SR_GRID)
    cand_l = np.zeros((n_cand, 34, 3, 3), dtype=np.float32)  # left-arm candidate patches
    cand_r = np.zeros((n_cand, 34, 3, 3), dtype=np.float32)
    idx = 0
    for sp in SP_GRID:
        for sr in SR_GRID:
            patch_l = np.stack([eye.copy() for _ in range(34)])
            patch_r = np.stack([eye.copy() for _ in range(34)])
            patch_l[LSP] = rot_z(sp)
            patch_l[LSR] = rot_y(sr)
            patch_r[RSP] = rot_z(sp)
            patch_r[RSR] = rot_y(-sr)
            cand_l[idx] = patch_l
            cand_r[idx] = patch_r
            idx += 1

    keyframes = list(range(0, KEYFRAME_MAX, KEYFRAME_STRIDE))
    if keyframes[-1] != KEYFRAME_MAX - 1:
        keyframes.append(KEYFRAME_MAX - 1)
    n_kf = len(keyframes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    all_ok = True

    for npz in sorted(REF_DIR.glob("d*.npz")):
        dname = npz.stem
        ref = np.load(npz)
        local_rot = torch.tensor(ref["local_rot_mats"], dtype=torch.float32)   # (T,34,3,3)
        root_pos = torch.tensor(ref["root_positions"], dtype=torch.float32)    # (T,3)

        # FK base pose (reference, unmodified) once for pelvis positions/orientations.
        gr_base, gp_base, _ = sk.fk(local_rot, root_pos)
        gp_base_n = gp_base.numpy()   # (T,34,3)
        gr_base_n = gr_base.numpy()   # (T,34,3,3)

        # Per-keyframe IK: batch candidates for all keyframes.
        # Build (n_kf * n_cand, 34, 3, 3) pose batch for left and right arms.
        base_kf = local_rot[keyframes]                       # (K,34,3,3)
        root_kf = root_pos[keyframes]                        # (K,3)
        base_kf_n = base_kf.numpy()

        # Left-arm search
        poses_l = np.repeat(base_kf_n, n_cand, axis=0)       # (K*C,34,3,3)
        for j in ARM_L:
            poses_l[:, j] = np.repeat(cand_l[:, j][None], n_kf, axis=0).reshape(n_kf * n_cand, 3, 3)
        # Right-arm search
        poses_r = np.repeat(base_kf_n, n_cand, axis=0)
        for j in ARM_R:
            poses_r[:, j] = np.repeat(cand_r[:, j][None], n_kf, axis=0).reshape(n_kf * n_cand, 3, 3)

        root_batch = np.repeat(root_kf.numpy(), n_cand, axis=0)  # (K*C,3)

        def search(poses, wrist_idx, side_sign):
            gr, gp, _ = sk.fk(torch.tensor(poses), torch.tensor(root_batch))
            gp_n = gp.numpy()   # (K*C,34,3)
            gr_n = gr.numpy()
            # Target per keyframe: pelvis + R_p @ (side_sign*LAT, UP, FWD)
            pp = gp_n.reshape(n_kf, n_cand, 34, 3)
            pr = gr_n.reshape(n_kf, n_cand, 34, 3, 3)
            target = np.zeros((n_kf, 3))
            for k in range(n_kf):
                Rp = gr_base_n[keyframes[k], PELVIS]
                target[k] = gp_base_n[keyframes[k], PELVIS] + Rp @ np.array(
                    [side_sign * LAT_TARGET, UP_TARGET, FWD_TARGET]
                )
            wpos = pp[:, :, wrist_idx, :]                     # (K,C,3)
            d = np.linalg.norm(wpos - target[:, None, :], axis=-1)  # (K,C)
            best = np.argmin(d, axis=1)                       # (K,)
            return best, d, pp, pr

        best_l, d_l, pp_l, pr_l = search(poses_l, LW_Y, +1.0)
        best_r, d_r, pp_r, pr_r = search(poses_r, RW_Y, -1.0)

        # Assemble optimized keyframe poses.
        opt_local = base_kf_n.copy()
        for k in range(n_kf):
            opt_local[k, ARM_L] = cand_l[best_l[k]][ARM_L]
            opt_local[k, ARM_R] = cand_r[best_r[k]][ARM_R]

        # FK optimized keyframe poses.
        gr_kf, gp_kf, _ = sk.fk(torch.tensor(opt_local), torch.tensor(root_kf.numpy()))
        gp_kf_n = gp_kf.numpy()   # (K,34,3)
        gr_kf_n = gr_kf.numpy()   # (K,34,3,3)

        # Verify pelvis-frame lateral clearance.
        lat_l = np.zeros(n_kf)
        lat_r = np.zeros(n_kf)
        up_l = np.zeros(n_kf)
        up_r = np.zeros(n_kf)
        for k in range(n_kf):
            Rp = gr_kf_n[k, PELVIS]
            off_l = gp_kf_n[k, LW_Y] - gp_kf_n[k, PELVIS]
            off_r = gp_kf_n[k, RW_Y] - gp_kf_n[k, PELVIS]
            lat_l[k] = abs(off_l @ Rp[:, 0])
            lat_r[k] = abs(off_r @ Rp[:, 0])
            up_l[k] = off_l @ Rp[:, 1]
            up_r[k] = off_r @ Rp[:, 1]

        ok_l = lat_l.min() >= MIN_LATERAL and up_l.min() > 0.0
        ok_r = lat_r.min() >= MIN_LATERAL and up_r.min() > 0.0
        ok = ok_l and ok_r
        all_ok = all_ok and ok

        # Serialize native constraint dicts (same schema as get_save_info:
        # local_joints_rot = global->local then axis-angle).
        local_rot_kf = sk.global_rots_to_local_rots(gr_kf)
        local_rot_aa = matrix_to_axis_angle(local_rot_kf).numpy()
        root_kf_np = gp_kf_n[:, PELVIS]
        smooth_root_2d = root_kf_np[:, [0, 2]].tolist()
        constraints = [
            {
                "type": "left-hand",
                "frame_indices": keyframes,
                "local_joints_rot": local_rot_aa.tolist(),
                "root_positions": root_kf_np.tolist(),
                "smooth_root_2d": smooth_root_2d,
            },
            {
                "type": "right-hand",
                "frame_indices": keyframes,
                "local_joints_rot": local_rot_aa.tolist(),
                "root_positions": root_kf_np.tolist(),
                "smooth_root_2d": smooth_root_2d,
            },
        ]
        out_file = OUT_DIR / f"{dname}.constraints.json"
        out_file.write_text(json.dumps(constraints) + "\n")

        # Round-trip validation via load_constraints_lst + FK.
        loaded = load_constraints_lst(constraints, sk, device="cpu")
        assert len(loaded) == 2, f"{dname}: expected 2 constraint objects"
        lh = loaded[0].global_joints_positions.numpy()
        rh = loaded[1].global_joints_positions.numpy()
        max_err = float(max(np.abs(lh - gp_kf_n).max(), np.abs(rh - gp_kf_n).max()))
        rt_ok = max_err < 1e-4
        all_ok = all_ok and rt_ok

        report[dname] = {
            "constraint_file": str(out_file),
            "keyframes": keyframes,
            "n_keyframes": n_kf,
            "left_min_lateral_m": float(lat_l.min()),
            "left_median_lateral_m": float(np.median(lat_l)),
            "left_min_up_m": float(up_l.min()),
            "right_min_lateral_m": float(lat_r.min()),
            "right_median_lateral_m": float(np.median(lat_r)),
            "right_min_up_m": float(up_r.min()),
            "roundtrip_max_abs_err_m": max_err,
            "pelvis_frame_check_pass": bool(ok),
            "roundtrip_fk_pass": bool(rt_ok),
            "target": {"lateral_m": LAT_TARGET, "up_m": UP_TARGET, "fwd_m": FWD_TARGET},
        }
        print(f"[{dname}] L lat min={lat_l.min():.3f} med={np.median(lat_l):.3f} up_min={up_l.min():.3f} "
              f"R lat min={lat_r.min():.3f} med={np.median(lat_r):.3f} up_min={up_r.min():.3f} "
              f"rtErr={max_err:.2e} OK={ok and rt_ok}")

    (OUT_DIR / "report.json").write_text(json.dumps({
        "model": name,
        "keyframe_stride": KEYFRAME_STRIDE,
        "min_lateral_m": MIN_LATERAL,
        "target": {"lateral_m": LAT_TARGET, "up_m": UP_TARGET, "fwd_m": FWD_TARGET},
        "note": "Per-keyframe straight-arm IK; native EndEffector hand constraint schema.",
        "directions": report,
    }, indent=2) + "\n")
    print(f"ALL_OK={all_ok}")
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    main()
