"""Convert sampled velocity commands to Kimodo Root2D constraints."""

import numpy as np

from .sampler import SampledMotion


def build_root2d_constraint(
    vel: dict[str, float],
    duration: float,
    fps: int = 30,
    heading: float = 0.0,
) -> dict:
    """Convert a velocity command to a Root2D constraint dict.

    Args:
        vel: {"vx": float, "vy": float, "wz": float}
            vx = forward velocity (m/s)
            vy = lateral velocity (m/s)
            wz = angular velocity (rad/s)
        duration: motion duration in seconds
        fps: frames per second (G1 = 30)
        heading: initial heading angle (radians, 0 = facing +z)

    Returns:
        Dict with "type", "frame_indices", "smooth_root_2d",
        and optionally "global_root_heading", ready for Kimodo.
    """
    num_frames = int(duration * fps)
    dt = 1.0 / fps
    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    wz = vel.get("wz", 0.0)

    t = np.arange(num_frames) * dt

    # Build trajectory in the (x,z) plane
    # x = forward axis, z maps to what Kimodo uses
    # Without turning (wz=0): simple linear path
    # With turning: path follows a circular arc
    if abs(wz) < 1e-6:
        # Straight line
        x = vx * t
        z = vy * t
        headings = None
    else:
        # Circular arc: radius = v/wz
        # Position on circle: (R * sin(wt), R * (1 - cos(wt))) then rotate
        radius = np.sqrt(vx**2 + vy**2) / abs(wz)
        angle = heading + wz * t
        # Starting direction
        start_dir = np.arctan2(vy, vx)
        # Arc positions in local frame then rotate:
        # For small wz, use exact integration
        x = np.where(
            vx != 0,
            (vx * np.sin(wz * t) + vy * (1 - np.cos(wz * t))) / wz,
            vx * t,
        )
        z = np.where(
            vy != 0,
            (vy * np.sin(wz * t) - vx * (1 - np.cos(wz * t))) / wz,
            vy * t,
        )
        # Compute heading at each frame
        frame_headings = heading + wz * t
        headings = np.stack(
            [np.cos(frame_headings), np.sin(frame_headings)], axis=-1
        )

    # smooth_root_2d is (x, z) — Kimodo uses smooth_root_pos in (x,z) plane
    smooth_root_2d = np.stack([x, z], axis=-1).tolist()
    frame_indices = list(range(num_frames))

    constraint = {
        "type": "root2d",
        "frame_indices": frame_indices,
        "smooth_root_2d": smooth_root_2d,
    }

    if headings is not None:
        constraint["global_root_heading"] = headings.tolist()

    return constraint


def build_constraints_json(sample: SampledMotion, fps: int = 30) -> list[dict]:
    """Build the full constraints list for a sampled motion.

    Args:
        sample: SampledMotion with velocity parameters.
        fps: Frames per second.

    Returns:
        List of constraint dicts (ready for JSON serialization).
    """
    constraints = []

    # Add root2d constraint if we have velocity commands
    if sample.vel and any(abs(v) > 1e-6 for v in sample.vel.values()):
        root_constraint = build_root2d_constraint(
            sample.vel, sample.duration, fps=fps
        )
        constraints.append(root_constraint)

    return constraints


def build_sampled_constraints_list(
    samples: list[SampledMotion], fps: int = 30
) -> list[dict]:
    """Build a constraint list from multiple sampled motions (one constraint per motion).

    Each constraint will be cropped appropriately by Kimodo during multi-prompt generation.
    For single-motion batches, just pass the constraint directly.
    """
    all_constraints = []
    for sample in samples:
        constraints = build_constraints_json(sample, fps=fps)
        all_constraints.append(constraints)
    return all_constraints
