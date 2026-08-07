#!/usr/bin/env python3
"""Author native Kimodo Root2D constraint JSONs for DATA-KIMODO-DISTRIBUTED-007.

This is a STANDALONE constraint-authoring tool (numpy only). It does NOT import
the locomotion_framework and is not part of any generation path; it writes the
native `Root2DConstraintSet` JSON files that `python -m kimodo.scripts.generate`
consumes via `--constraints`.

Native schema (kimodo/constraints.py Root2DConstraintSet):
  {
    "type": "root2d",
    "frame_indices": [int, ...],            # length K
    "smooth_root_2d": [[x, z], ...],        # length K, x=lateral z=forward
    # optional:
    "global_root_heading": [[cx, sy], ...]  # length K, [cos, sin] model convention
  }

Verified native axis convention (from CONSTRAINT-003 c1_forward/c1_right and
global_root_heading): smooth_root_2d x is positive toward the model's LEFT and
negative toward RIGHT; z is positive FORWARD. A native compass angle theta is
measured from +z (forward) toward +x (left), i.e. theta=+90 deg is a LEFT turn
and theta=-90 deg is a RIGHT turn. Equivalently, for a world command
(vx_eff=forward, vy_eff=right):
    x = -vy_eff*t ,  z = vx_eff*t.
The Y-series direction NAMES (d00..d07) are preserved, but each name maps to
its native-frame angle (e.g. d02_right = -90 deg, d06_left = +90 deg).

Cells (8 directions d00..d07, seed 20260805, 8 s / 240 f / 30 fps / 100 steps):
  E1: straight-line Root2D, DENSE (all 240 frames), varied speed per direction.
  E2: curved-arc Root2D, SPARSE stride-8 (frames 0,8,...,232,239), no heading.
  E3: staggered-waypoint Root2D, SPARSE 3 keyframes (0,120,239), no heading.
  E4: straight-line Root2D, DENSE + paired global_root_heading (natural facing).
E0 is text-only (no constraint) and is produced by the job loop with no
`--constraints` argument.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Frozen generation invariants
# ---------------------------------------------------------------------------
SEED = 20260805
DURATION_S = 8.0
FPS = 30
NUM_FRAMES = int(DURATION_S * FPS)  # 240
DT = 1.0 / FPS

# Native-frame compass angles (degrees): 0 = forward (+z), positive toward
# native +x = LEFT. A "right" command is therefore a NEGATIVE angle (toward
# native -x). Verified against CONSTRAINT-003 c1_right which walked RIGHT at
# native x = -2.65 (via qpos y = -2.65 and +y = left in the G1 URDF frame).
# The Y-series direction NAMES (d00..d07) are preserved; only the native angle
# is listed here.
DIRECTIONS = [
    ("d00_forward", 0.0),
    ("d01_forward_right", -45.0),   # toward -x (right), +z (forward)
    ("d02_right", -90.0),           # toward -x (right)
    ("d03_backward_right", -135.0), # toward -x (right), -z (backward)
    ("d04_backward", 180.0),        # toward -z (backward)
    ("d05_backward_left", 135.0),   # toward +x (left), -z (backward)
    ("d06_left", 90.0),             # toward +x (left)
    ("d07_forward_left", 45.0),     # toward +x (left), +z (forward)
]

PROMPTS = {
    "d00_forward": "A person walks forward.",
    "d01_forward_right": "A person walks forward to the right.",
    "d02_right": "A person walks to the right.",
    "d03_backward_right": "A person walks backward to the right.",
    "d04_backward": "A person walks backward.",
    "d05_backward_left": "A person walks backward to the left.",
    "d06_left": "A person walks to the left.",
    "d07_forward_left": "A person walks forward to the left.",
}

# Speed (m/s) per direction for the straight cells (E1/E4). Varied on purpose:
# the CONSTRAINT-003 narrow envelope used one fixed speed for all directions;
# "distributed" means the envelope should vary travel distance/cadence.
E1_SPEED = {
    "d00_forward": 0.86,   # matches accepted CONSTRAINT-003 forward baseline (6.88 m in 8 s)
    "d01_forward_right": 0.60,
    "d02_right": 0.45,
    "d03_backward_right": 0.45,
    "d04_backward": 0.50,
    "d05_backward_left": 0.45,
    "d06_left": 0.45,
    "d07_forward_left": 0.60,
}

# Curved cell (E2): each direction walks a constant-speed arc turning +20 deg
# (toward LEFT/native +x) on even cells and -20 deg on odd cells, to widen the
# envelope across turn directions.
E2_TURN_DEG = {
    "d00_forward": 20.0,
    "d01_forward_right": -20.0,
    "d02_right": 20.0,
    "d03_backward_right": -20.0,
    "d04_backward": 20.0,
    "d05_backward_left": -20.0,
    "d06_left": 20.0,
    "d07_forward_left": -20.0,
}
E2_SPEED = {
    "d00_forward": 0.80,
    "d01_forward_right": 0.60,
    "d02_right": 0.45,
    "d03_backward_right": 0.45,
    "d04_backward": 0.50,
    "d05_backward_left": 0.45,
    "d06_left": 0.45,
    "d07_forward_left": 0.60,
}
E2_STRIDE = 8

# Waypoint cell (E3): sparse 3-keyframe polylines in native (x, z); x>0 = left,
# x<0 = right, z>0 = forward. Frames at the vertices.
E3_WAYPOINTS = {
    "d00_forward": [(0.0, 0.0), (0.5, 3.2), (-0.3, 6.8)],       # gentle S-weave
    "d01_forward_right": [(0.0, 0.0), (-1.4, 2.6), (-3.4, 3.4)],  # two-segment diagonal
    "d02_right": [(0.0, 0.0), (-1.6, 0.5), (-3.6, 0.0)],        # right w/ slight forward
    "d03_backward_right": [(0.0, 0.0), (-1.3, -1.3), (-2.5, -2.5)],
    "d04_backward": [(0.0, 0.0), (-0.4, -2.0), (0.4, -4.0)],    # S-weave backward
    "d05_backward_left": [(0.0, 0.0), (1.3, -1.3), (2.5, -2.5)],
    "d06_left": [(0.0, 0.0), (1.6, 0.5), (3.6, 0.0)],          # left w/ slight forward
    "d07_forward_left": [(0.0, 0.0), (1.4, 2.6), (3.4, 3.4)],
}
E3_KEYFRAMES = [0, 120, 239]

# E4: straight dense Root2D + paired heading. Heading follows natural facing:
# forward/backward/diagonals face the travel direction; pure lateral (right/
# left) face FORWARD (side-step, mirrors CONSTRAINT-003 C5 avoidance of forcing
# tangent yaw on side-steps).
E4_SPEED = E1_SPEED


def direction_angle_rad(theta_deg: float) -> float:
    return math.radians(theta_deg)


def compass_command(theta_deg: float, speed: float) -> tuple[float, float]:
    """World command (vx=forward, vy=right) for compass theta (deg, 0=forward,
    +90=LEFT in the native convention)."""
    a = direction_angle_rad(theta_deg)
    # theta is measured toward native +x (left). A command that moves the body
    # toward +x (left) has vy_eff = -speed*sin(theta) in (forward, right) terms
    # because native x = -vy_eff*t. We instead return (vx_eff, vy_eff) with the
    # +right vy_eff, then map x = -vy_eff*t below. For theta measured toward
    # native left: forward comp vx_eff = speed*cos(theta), lateral vy_eff:
    #   native x_end = -vy_eff*T and x_end = speed*T*sin(theta) -> vy_eff=-speed*sin(theta).
    vx_eff = speed * math.cos(a)
    vy_eff = -speed * math.sin(a)  # +right
    return vx_eff, vy_eff


def straight_path(theta_deg: float, speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Return native (X, Z) arrays for a constant-speed straight path."""
    vx_eff, vy_eff = compass_command(theta_deg, speed)
    t = np.arange(NUM_FRAMES, dtype=np.float64) * DT
    x = -vy_eff * t  # native x: negative for right
    z = vx_eff * t
    return x, z


def arc_path(theta_deg: float, speed: float, turn_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return native (X, Z) for a constant-speed arc turning turn_deg over 8 s.

    World velocity direction angle phi(t) = theta_0 + wz*t where theta_0 is the
    compass angle and the turn is toward native +x (left) for positive turn_deg.
    Native x = -v_right_effective integrated; z = v_forward integrated.
    """
    a0 = direction_angle_rad(theta_deg)
    wz = math.radians(turn_deg) / DURATION_S  # rad/s
    t = np.arange(NUM_FRAMES, dtype=np.float64) * DT
    phi = a0 + wz * t
    # In (native x, z): velocity = (speed*sin(phi), speed*cos(phi)) because phi
    # is measured from +z toward +x (left). Equivalent to:
    #   v_forward = speed*cos(phi), v_right = -speed*sin(phi) -> x=-v_right = speed*sin(phi).
    v_x = speed * np.sin(phi)
    v_z = speed * np.cos(phi)
    x = np.cumsum(v_x) * DT
    z = np.cumsum(v_z) * DT
    return x, z


def natural_heading(theta_deg: float, t: np.ndarray) -> np.ndarray:
    """Global root heading (model convention) for natural facing.

    Model convention (verified): a robot facing forward has heading = [1, 0];
    the heading vector points along the robot's facing, expressed as
    [cos(facing_angle), sin(facing_angle)] with facing_angle measured from +z
    toward +x (native x positive = left). For pure lateral directions the body
    faces forward (side-step). Otherwise it faces the travel direction.
    """
    theta = theta_deg % 360.0
    if theta in (90.0, 270.0):
        # pure lateral: face forward
        facing = np.full(len(t), 0.0)
    elif theta == 180.0:
        facing = np.full(len(t), math.pi)
    else:
        # travel direction angle from +z toward +x
        a0 = direction_angle_rad(theta_deg)
        facing = np.full(len(t), a0)
    return np.stack([np.cos(facing), np.sin(facing)], axis=-1)


def root2d_dict(frame_indices: list[int], x: np.ndarray, z: np.ndarray,
                heading: np.ndarray | None = None) -> dict:
    x = np.asarray(x, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    if len(x) != len(frame_indices):
        raise ValueError("smooth_root_2d length must match frame_indices")
    out = {
        "type": "root2d",
        "frame_indices": [int(i) for i in frame_indices],
        "smooth_root_2d": np.stack([x, z], axis=-1).tolist(),
    }
    if heading is not None:
        h = np.asarray(heading, dtype=np.float32)
        if len(h) != len(frame_indices):
            raise ValueError("global_root_heading length must match frame_indices")
        out["global_root_heading"] = h.tolist()
    return out


def dense_frames() -> list[int]:
    return list(range(NUM_FRAMES))


def stride_frames(stride: int) -> list[int]:
    f = list(range(0, NUM_FRAMES, stride))
    if f[-1] != NUM_FRAMES - 1:
        f.append(NUM_FRAMES - 1)
    return f


def build_e1() -> dict[str, dict]:
    out = {}
    for name, theta in DIRECTIONS:
        x, z = straight_path(theta, E1_SPEED[name])
        out[name] = root2d_dict(dense_frames(), x, z)
    return out


def build_e2() -> dict[str, dict]:
    out = {}
    for name, theta in DIRECTIONS:
        x, z = arc_path(theta, E2_SPEED[name], E2_TURN_DEG[name])
        fi = stride_frames(E2_STRIDE)
        out[name] = root2d_dict(fi, x[fi], z[fi])
    return out


def build_e3() -> dict[str, dict]:
    out = {}
    for name, theta in DIRECTIONS:
        wps = E3_WAYPOINTS[name]
        fi = list(E3_KEYFRAMES)
        assert len(wps) == len(fi), (name, len(wps), len(fi))
        x = np.array([wp[0] for wp in wps], dtype=np.float32)
        z = np.array([wp[1] for wp in wps], dtype=np.float32)
        out[name] = root2d_dict(fi, x, z)
    return out


def build_e4() -> dict[str, dict]:
    out = {}
    t = np.arange(NUM_FRAMES, dtype=np.float64)
    for name, theta in DIRECTIONS:
        x, z = straight_path(theta, E4_SPEED[name])
        heading = natural_heading(theta, t)
        out[name] = root2d_dict(dense_frames(), x, z, heading)
    return out


CELLS = {"E1": build_e1, "E2": build_e2, "E3": build_e3, "E4": build_e4}

# E0 = text-only control (no constraint). Same prompt/seed/duration/diffusion as
# every other cell so the only factor vs E1-E4 is the Root2D envelope.
CELL_ORDER = ["E0", "E1", "E2", "E3", "E4"]


def sample_name(cell: str, direction: str) -> str:
    return f"{cell.lower()}_{direction}_s{SEED}"


def build_manifest() -> list[dict]:
    rows = []
    for cell in CELL_ORDER:
        for direction, _ in DIRECTIONS:
            if cell == "E0":
                constraint = "none"
            else:
                constraint = f"constraints/{cell}_{direction}.constraints.json"
            rows.append({
                "cell": cell,
                "sample": sample_name(cell, direction),
                "direction": direction,
                "prompt": PROMPTS[direction],
                "duration_s": f"{DURATION_S:.1f}",
                "seed": SEED,
                "diffusion_steps": 100,
                "constraint": constraint,
                "native_stem": f"native/{sample_name(cell, direction)}",
            })
    return rows


def validate_schema(constraints: list[dict]) -> None:
    for c in constraints:
        assert c["type"] == "root2d", c["type"]
        fi = c["frame_indices"]
        sr = c["smooth_root_2d"]
        assert len(fi) == len(sr), (len(fi), len(sr))
        assert all(isinstance(i, int) for i in fi)
        assert fi == sorted(fi) and len(set(fi)) == len(fi), "indices sorted+unique"
        assert 0 <= fi[0] and fi[-1] < NUM_FRAMES
        if "global_root_heading" in c:
            gh = c["global_root_heading"]
            assert len(gh) == len(fi)
            for cx, sy in gh:
                n = math.hypot(cx, sy)
                assert n > 1e-6, "zero-length heading vector"
        for x, z in sr:
            assert math.isfinite(x) and math.isfinite(z)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "constraints"
    out_dir.mkdir(exist_ok=True)
    for cell, builder in CELLS.items():
        for name, constraint in builder().items():
            path = out_dir / f"{cell}_{name}.constraints.json"
            lst = [constraint]
            validate_schema(lst)
            path.write_text(json.dumps(lst, indent=2) + "\n")
            fi = constraint["frame_indices"]
            first = constraint["smooth_root_2d"][0]
            last = constraint["smooth_root_2d"][-1]
            has_h = "global_root_heading" in constraint
            print(f"{path.name:44s} K={len(fi):3d} first={first} last={last} heading={has_h}")

    manifest_path = Path(__file__).resolve().parent / "sample_manifest.csv"
    rows = build_manifest()
    fieldnames = list(rows[0])
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    n_constrained = sum(1 for r in rows if r["constraint"] != "none")
    print(f"wrote {manifest_path} with {len(rows)} rows "
          f"({n_constrained} constrained, {len(rows) - n_constrained} text-only)")
    print("CONSTRAINT_GENERATION_DONE")


if __name__ == "__main__":
    main()
