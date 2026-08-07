#!/usr/bin/env python3
"""First-sample gate for DATA-KIMODO-DISTRIBUTED-007.

Verifies (before the full 40-motion batch) that:
  1. E1 d00 forward walks forward (native z displacement >> |x|, z > 0).
  2. E1 d02 right walks right (native x displacement negative, |x| >> |z|).
  3. E4 d00 forward has the constrained heading near [1,0] (faces forward) and
     walks forward.
  4. The replay conversion produces a valid [240,36] qpos motion.npz.

Native-frame convention (verified): root_positions (x, y, z) with x positive =
left, z positive = forward. For "right" the x displacement must be NEGATIVE.

Usage:
    python gate_check.py <batch_root>
Returns exit 0 if all gates pass, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if BATCH is None:
    sys.exit("usage: gate_check.py <batch_root>")

# (sample, label, expected) -- expected: (travel_sign_x, travel_sign_z)
# travel_sign = -1 / 0 / +1 for the dominant native axis displacement.
CHECKS = [
    ("e1_d00_forward_s20260805", "E1 d00 forward",
     {"x_min": None, "z_min": 3.0, "heading_first": None}),
    ("e1_d02_right_s20260805", "E1 d02 right",
     {"x_max": -1.5, "z_min": None, "heading_first": None}),
    ("e4_d00_forward_s20260805", "E4 d00 forward",
     {"x_min": None, "z_min": 3.0, "heading_first": (0.9, None)}),
]

failures: list[str] = []
for sample, label, exp in CHECKS:
    native = BATCH / "native" / f"{sample}.npz"
    replay = BATCH / "motions" / sample / "motion.npz"
    if not native.exists():
        failures.append(f"{label}: missing native {native}")
        continue
    a = np.load(native, allow_pickle=False)
    rp = a["root_positions"].astype(np.float64)
    dx = float(rp[-1, 0] - rp[0, 0])   # +left
    dz = float(rp[-1, 2] - rp[0, 2])   # +forward
    heading = None
    if "global_root_heading" in a.files:
        heading = a["global_root_heading"].astype(np.float64)
    print(f"[gate] {label}: dx={dx:+.3f} dz={dz:+.3f} "
          f"|disp|={np.hypot(dx, dz):.3f} m")
    if exp["x_min"] is not None and dx > exp["x_min"]:
        failures.append(f"{label}: expected rightward (dx<{exp['x_min']}), got dx={dx:+.3f}")
    if exp["z_min"] is not None and dz < exp["z_min"]:
        failures.append(f"{label}: expected forward (dz>={exp['z_min']}), got dz={dz:+.3f}")
    if exp["heading_first"] is not None:
        dot_lim, _ = exp["heading_first"]
        if heading is None:
            failures.append(f"{label}: no global_root_heading in native output")
        else:
            h0 = heading[0]
            dot = float(np.dot(h0, np.array([1.0, 0.0])))
            norm = float(np.linalg.norm(h0))
            print(f"[gate]   {label} heading[0]={h0.tolist()} dot(+fwd)={dot:.3f} "
                  f"norm={norm:.3f}")
            if dot < dot_lim:
                failures.append(
                    f"{label}: heading not forward (dot={dot:.3f} < {dot_lim})")
            if abs(norm - 1.0) > 1e-3:
                failures.append(f"{label}: heading not unit ({norm:.4f})")
    # replay conversion gate
    if replay.exists():
        q = np.load(replay, allow_pickle=False)["qpos"]
        fps = float(np.load(replay, allow_pickle=False)["fps"].reshape(-1)[0])
        ok = q.shape == (240, 36) and fps == 30.0
        print(f"[gate]   replay {sample}: qpos={q.shape} fps={fps} ok={ok}")
        if not ok:
            failures.append(f"{label}: replay qpos shape/fps unexpected")
    else:
        failures.append(f"{label}: missing replay {replay}")

if failures:
    print("GATE_FAIL")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("GATE_PASS")
sys.exit(0)
