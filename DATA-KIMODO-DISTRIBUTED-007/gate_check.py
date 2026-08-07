#!/usr/bin/env python3
"""First-sample gate for DATA-KIMODO-DISTRIBUTED-007.

Verifies (before the full 40-motion batch) that:
  1. E1 d00 forward walks forward (native z displacement > 3.0 m).
  2. E1 d02 right walks right (native x displacement < -1.5 m).
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

# (sample, label, expected) -- optional thresholds on native dx/dz and heading.
CHECKS = [
    ("e1_d00_forward_s20260805", "E1 d00 forward", dict(dz_gt=3.0)),
    ("e1_d02_right_s20260805", "E1 d02 right", dict(dx_lt=-1.5)),
    ("e4_d00_forward_s20260805", "E4 d00 forward",
     dict(dz_gt=3.0, heading_forward=True)),
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
    dx_lt = exp.get("dx_lt")
    if dx_lt is not None and not dx < dx_lt:
        failures.append(f"{label}: expected rightward (dx<{dx_lt}), got dx={dx:+.3f}")
    dz_gt = exp.get("dz_gt")
    if dz_gt is not None and not dz > dz_gt:
        failures.append(f"{label}: expected forward (dz>{dz_gt}), got dz={dz:+.3f}")
    if exp.get("heading_forward"):
        if heading is None:
            failures.append(f"{label}: no global_root_heading in native output")
        else:
            h0 = heading[0]
            dot = float(np.dot(h0, np.array([1.0, 0.0])))
            norm = float(np.linalg.norm(h0))
            print(f"[gate]   {label} heading[0]={h0.tolist()} dot(+fwd)={dot:.3f} "
                  f"norm={norm:.3f}")
            if dot < 0.9:
                failures.append(f"{label}: heading not forward (dot={dot:.3f})")
            # The model's float32 (cos,sin) heading is naturally ~1.00-1.002
            # (the accepted CONSTRAINT-003 C0 native heading was also ~1.002).
            # A loose sanity bound only catches grossly-wrong vectors.
            if abs(norm - 1.0) > 0.1:
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
