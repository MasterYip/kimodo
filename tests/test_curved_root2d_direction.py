"""DATA-KIMODO-NATURAL-LOCO-014 — regression test for the curved-path fix.

The framework's ``build_root2d_constraint`` curved branch previously rotated the
ALREADY-world-frame velocity components by θ = θ₀ + wz·t, which double-rotated
the initial travel direction to 2·θ₀ for any non-forward heading.  Only forward
arcs (θ₀=0) were correct, which is why PORT-008's forward-curve checks passed.

This test pins the corrected geometry: the initial tangent of a curved Root2D
path must equal the commanded native-compass direction (θ₀) for forward,
backward AND lateral arcs, and the emitted global root heading must start at
θ₀ (body faces the travel direction) — unchanged for forward arcs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from locomotion_framework.constraints import build_root2d_constraint  # noqa: E402

FPS = 30


def _initial_tangent_deg(constraint: dict) -> float:
    pts = np.asarray(constraint["smooth_root_2d"], dtype=np.float64)
    d = pts[1] - pts[0]
    return float(np.degrees(np.arctan2(d[0], d[1])))  # x=left, z=forward


def _heading_deg(constraint: dict) -> list[float]:
    h = np.asarray(constraint["global_root_heading"], dtype=np.float64)
    return [float(np.degrees(np.arctan2(v[1], v[0]))) for v in h]


def _angle_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("direction_deg,expect_deg", [
    (0.0, 0.0),     # forward
    (180.0, 180.0), # backward
    (90.0, 90.0),   # left
    (-90.0, -90.0), # right
    (45.0, 45.0),   # forward-left
    (-45.0, -45.0), # forward-right
    (135.0, 135.0), # backward-left
])
def test_curved_initial_tangent_matches_direction(direction_deg: float, expect_deg: float):
    theta = np.deg2rad(direction_deg)
    vx = float(np.cos(theta))
    vy = float(np.sin(theta))
    c = build_root2d_constraint(
        {"vx": vx, "vy": vy, "wz": 1.0}, 8.0, fps=FPS, stride=1,
    )
    tan = _initial_tangent_deg(c)
    assert abs(_angle_diff(tan, expect_deg)) <= 2.0
    h0 = _heading_deg(c)[0]
    assert abs(_angle_diff(h0, expect_deg)) <= 1.0


def test_curved_forward_heading_emission_backward_compatible():
    """Forward arc (θ₀=0): heading emission = wz·t exactly as before the fix."""
    c = build_root2d_constraint(
        {"vx": 1.0, "vy": 0.0, "wz": 1.0}, 8.0, fps=FPS, stride=1,
    )
    h = _heading_deg(c)
    t = np.arange(len(h), dtype=np.float64) / FPS
    # heading(t) = 0 + 1.0*t (rad), so the vector angle should equal degrees(t)
    for i in range(0, len(h), 30):
        assert abs(_angle_diff(h[i], np.degrees(1.0 * t[i]))) <= 0.5


def test_curved_heading_rotates_at_wz_from_theta0():
    """Backward arc: heading starts at 180 and rotates at +wz."""
    c = build_root2d_constraint(
        {"vx": -1.0, "vy": 0.0, "wz": 0.5}, 8.0, fps=FPS, stride=1,
    )
    h = _heading_deg(c)
    t = np.arange(len(h), dtype=np.float64) / FPS
    for i in range(0, len(h), 30):
        expect = np.degrees(np.pi + 0.5 * t[i])
        got = h[i]
        diff = (got - expect + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.5
