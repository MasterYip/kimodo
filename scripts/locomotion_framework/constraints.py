"""Convert sampled velocity commands to Kimodo Root2D constraints.

Velocity → (x,z) path via integration of a cos-eased speed profile:

    Phase 1 (accel, 10%):   0 → target speed
    Phase 2 (cruise, 80%):  constant target speed
    Phase 3 (decel, 10%):   target speed → 0

The path is built for a **stretched** duration (3× the actual generation
duration) so that only the accel + cruise phases fall within the
constrained frame range.  The model never sees the deceleration tail,
eliminating the constraint-vs-motion-prior conflict that causes root
jitter at the end of the motion.
"""

import numpy as np

from .sampler import SampledMotion

# Stretch factor: the path is built for this many times the generation
# duration.  With accel/decel each at 10% of the STRETCHED path, the
# actual generated frames see only accel (first 3.3% of stretched) +
# cruise (remaining 30% of stretched).  The deceleration begins at frame
# 0.9 * stretch * N_gen, which is far beyond the last generated frame.
_PATH_STRETCH = 3.0


def _speed_profile(
    num_frames_total: int,
    accel_frac: float = 0.10,
    decel_frac: float = 0.10,
) -> np.ndarray:
    """C¹-smooth speed multiplier ∈ [0, 1] for the **full** stretched path."""
    s = np.ones(num_frames_total, dtype=np.float64)
    n_accel = max(1, int(num_frames_total * accel_frac))
    n_decel = max(1, int(num_frames_total * decel_frac))

    if n_accel > 1:
        t = np.linspace(0, 1, n_accel)
        s[:n_accel] = 0.5 * (1.0 - np.cos(np.pi * t))

    if n_decel > 1:
        t = np.linspace(0, 1, n_decel)
        s[-n_decel:] = 0.5 * (1.0 + np.cos(np.pi * t))

    return s.astype(np.float32)


def build_root2d_constraint(
    vel: dict[str, float],
    duration: float,
    fps: int = 30,
    heading: float = 0.0,
) -> dict:
    """Build a Root2D constraint from a velocity command.

    The velocity profile is cosine-eased at both ends, then integrated
    frame-by-frame to produce smooth (x,z) positions.  The path is built
    for ``_PATH_STRETCH × duration`` but only the first ``duration``'s
    worth of frames are constrained — the robot never reaches the
    deceleration phase.
    """
    num_frames_gen = int(duration * fps)
    num_frames_path = int(duration * _PATH_STRETCH * fps)
    dt = 1.0 / fps

    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    wz = vel.get("wz", 0.0)

    speed = _speed_profile(num_frames_path)

    t = np.arange(num_frames_path, dtype=np.float64) * dt

    # ── Integrate speed profile over the STRETCHED path ─────────────
    if abs(wz) < 1e-6:
        # Straight: (x, z) = ∫ (vy*s, vx*s) dt
        #   vy (lateral)  → Kimodo X
        #   vx (forward)  → Kimodo Z
        x_full = np.cumsum(vy * speed) * dt
        z_full = np.cumsum(vx * speed) * dt
        headings = None
    else:
        # Curved (circular arc).  Body-frame velocity (vx,vy) with
        # rotation rate wz traces a circular arc in world coordinates.
        # We integrate numerically to support the eased speed profile:
        #
        #   θ(t) = atan2(vy, vx) + wz·t
        #   vx_world = vx·cos θ − vy·sin θ
        #   vz_world = vx·sin θ + vy·cos θ
        #   Kimodo X = ∫ lateral_world · speed  = ∫ vz_world · speed
        #   Kimodo Z = ∫ forward_world · speed  = ∫ vx_world · speed
        theta_0 = np.arctan2(vy, vx)
        theta = theta_0 + wz * t

        vx_world = vx * np.cos(theta) - vy * np.sin(theta)
        vz_world = vx * np.sin(theta) + vy * np.cos(theta)

        # Kimodo X = lateral component = vz_world
        # Kimodo Z = forward component = vx_world
        x_full = np.cumsum(vz_world * speed) * dt
        z_full = np.cumsum(vx_world * speed) * dt

        # Heading per frame (constrain only gen frames)
        frame_headings = heading + wz * (np.arange(num_frames_gen) * dt)
        headings = np.stack(
            [np.cos(frame_headings), np.sin(frame_headings)], axis=-1
        ).astype(np.float32)

    # ── Constrain only the generation-length prefix ──────────────────
    smooth_root_2d = np.stack(
        [x_full[:num_frames_gen], z_full[:num_frames_gen]], axis=-1
    ).astype(np.float32).tolist()

    constraint = {
        "type": "root2d",
        "frame_indices": list(range(num_frames_gen)),
        "smooth_root_2d": smooth_root_2d,
    }

    if headings is not None:
        constraint["global_root_heading"] = headings.tolist()

    return constraint


def build_constraints_json(sample: SampledMotion, fps: int = 30) -> list[dict]:
    """Build the full constraints list for a sampled motion."""
    constraints = []

    if sample.vel and any(abs(v) > 1e-6 for v in sample.vel.values()):
        root_constraint = build_root2d_constraint(
            sample.vel, sample.duration, fps=fps
        )
        constraints.append(root_constraint)

    return constraints


def build_sampled_constraints_list(
    samples: list[SampledMotion], fps: int = 30
) -> list[dict]:
    """Build a constraint list from multiple sampled motions."""
    all_constraints = []
    for sample in samples:
        constraints = build_constraints_json(sample, fps=fps)
        all_constraints.append(constraints)
    return all_constraints
