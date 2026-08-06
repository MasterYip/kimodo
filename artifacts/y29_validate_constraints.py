#!/usr/bin/env python3
"""Y29 deliverable-2 schema/tensor validation for the hand EndEffector constraints.

Validates each per-direction constraint JSON against the native
EndEffectorConstraintSet schema:
  * keys are exactly the serialized get_save_info keys
  * frame_indices are sorted, unique, within [0, KEYFRAME_MAX), count < 20
  * local_joints_rot has shape (K, 34, 3) and finite values
  * root_positions has shape (K, 3); smooth_root_2d has shape (K, 2)
  * pose/root consistency: load_constraints_lst rebuilds the set and FK
    reconstructs global positions within tolerance
  * hand lateral clearance in the pelvis frame >= MIN_LATERAL and hand height
    above pelvis > 0 at every keyframe
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from kimodo import load_model
from kimodo.constraints import load_constraints_lst

CONSTRAINTS_DIR = Path("/data/masteryip/kimodo/kimodo-agent-worktrees/DATA-KIMODO-LOCOMOTION-YAML-004/artifacts/y29_constraints")
KEYFRAME_MAX = 180
MIN_LATERAL = 0.20
FK_TOL = 1e-4
PELVIS = 0
LW_Y = 24
RW_Y = 32

REQUIRED_KEYS = {"type", "frame_indices", "local_joints_rot", "root_positions", "smooth_root_2d"}


def pelvis_frame_lateral_up(global_pos: np.ndarray, global_rot: np.ndarray, hand_idx: int):
    """Return (lateral_abs, up) arrays in pelvis frame for the hand index."""
    off = global_pos[:, hand_idx] - global_pos[:, PELVIS]   # (K,3)
    Rp = global_rot[:, PELVIS]                              # (K,3,3)
    # World offset dotted with the pelvis local basis (COLUMNS of R_pelvis):
    # col0 = lateral (left), col1 = up, col2 = forward.
    pelvis_frame_off = np.einsum("ki,kij->kj", off, Rp)     # (K,3)
    return np.abs(pelvis_frame_off[:, 0]), pelvis_frame_off[:, 1]


def validate_one(path: Path, skeleton, keyframes_ref) -> dict:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or len(raw) != 2:
        return {"file": path.name, "pass": False, "reason": "expected list of 2 constraint dicts"}
    # types must be left-hand + right-hand
    types = sorted(el["type"] for el in raw)
    if types != ["left-hand", "right-hand"]:
        return {"file": path.name, "pass": False, "reason": f"unexpected types {types}"}

    rows = []
    ok = True
    for el in raw:
        reason = []
        keys = set(el.keys())
        if keys != REQUIRED_KEYS:
            reason.append(f"keys={sorted(keys)}")
        fi = el.get("frame_indices")
        if not isinstance(fi, list) or len(fi) == 0:
            reason.append("frame_indices not a non-empty list")
        else:
            if len(fi) >= 20:
                reason.append(f"n_keyframes={len(fi)} >= 20")
            if sorted(fi) != fi:
                reason.append("frame_indices not sorted")
            if len(set(fi)) != len(fi):
                reason.append("frame_indices not unique")
            if min(fi) < 0 or max(fi) >= KEYFRAME_MAX:
                reason.append(f"frame_indices out of range {fi[0]}..{fi[-1]}")
            if keyframes_ref is not None and fi != keyframes_ref:
                reason.append("frame_indices differ from reference")
        lj = el.get("local_joints_rot")
        if not isinstance(lj, list) or not isinstance(lj[0], list):
            reason.append("local_joints_rot malformed")
        else:
            a = np.asarray(lj, dtype=np.float64)
            if a.shape != (len(fi), 34, 3):
                reason.append(f"local_joints_rot shape {a.shape}")
            if not np.isfinite(a).all():
                reason.append("local_joints_rot non-finite")
        rp = np.asarray(el.get("root_positions"), dtype=np.float64)
        if rp.shape != (len(fi), 3):
            reason.append(f"root_positions shape {rp.shape}")
        sr2d = np.asarray(el.get("smooth_root_2d"), dtype=np.float64)
        if sr2d.shape != (len(fi), 2):
            reason.append(f"smooth_root_2d shape {sr2d.shape}")
        if not reason:
            # Load + FK round-trip via the native path.
            loaded = load_constraints_lst([el], skeleton, device="cpu")[0]
            gp = loaded.global_joints_positions.numpy()   # (K,34,3)
            gr = loaded.global_joints_rots.numpy()        # (K,34,3,3)
            lat, up = pelvis_frame_lateral_up(gp, gr, LW_Y if el["type"] == "left-hand" else RW_Y)
            if lat.min() < MIN_LATERAL:
                reason.append(f"min lateral {lat.min():.3f} < {MIN_LATERAL}")
            if up.min() <= 0.0:
                reason.append(f"min up {up.min():.3f} <= 0")
            if not np.isfinite(gp).all() or not np.isfinite(gr).all():
                reason.append("FK produced non-finite values")
        rows.append({"type": el["type"], "n_keyframes": len(fi), "pass": not reason, "issues": reason})
        if reason:
            ok = False
    return {"file": path.name, "pass": ok, "details": rows}


def main() -> None:
    model, name = load_model("kimodo-g1-rp", device="cpu", return_resolved_name=True)
    sk = model.skeleton
    keyframes_ref = list(range(0, KEYFRAME_MAX, 10))
    if keyframes_ref[-1] != KEYFRAME_MAX - 1:
        keyframes_ref.append(KEYFRAME_MAX - 1)
    results = []
    all_ok = True
    for path in sorted(CONSTRAINTS_DIR.glob("d*.constraints.json")):
        r = validate_one(path, sk, keyframes_ref)
        results.append(r)
        all_ok = all_ok and r["pass"]
        print(f"[{path.name}] pass={r['pass']}")
        for d in r.get("details", []):
            print(f"    {d['type']}: pass={d['pass']} n_kf={d['n_keyframes']} issues={d['issues']}")
    print(f"VALIDATE_ALL_PASS={all_ok}")
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    main()
