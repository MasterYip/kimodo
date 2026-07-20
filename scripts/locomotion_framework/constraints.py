"""Convert sampled velocity commands to Kimodo Root2D constraints."""

import numpy as np

from .sampler import SampledMotion


def _ease_out_cosine(t: np.ndarray) -> np.ndarray:
    """Smooth ease-out: 1 → 0 over t ∈ [0, 1].  Derivative is 0 at both ends."""
    return 0.5 * (1.0 + np.cos(np.pi * t))


def build_root2d_constraint(
    vel: dict[str, float],
    duration: float,
    fps: int = 30,
    heading: float = 0.0,
    decel_frac: float = 0.15,
) -> dict:
    """Convert a velocity command to a Root2D constraint dict.

    The path is built at constant velocity for the first ``1 - decel_frac``
    of the duration, then smoothly eases to a stop over the remaining
    ``decel_frac``.  This prevents the constraint-vs-diffusion conflict at
    the tail that causes root jitter.

    Args:
        vel: {"vx": float, "vy": float, "wz": float}
            vx = forward velocity (m/s)
            vy = lateral velocity (m/s)
            wz = angular velocity (rad/s)
        duration: motion duration in seconds.
        fps: frames per second.
        heading: initial heading angle (radians, 0 = facing +z).
        decel_frac: fraction of the duration used for deceleration (0 = none).

    Returns:
        Dict with "type", "frame_indices", "smooth_root_2d",
        and optionally "global_root_heading".
    """
    num_frames = int(duration * fps)
    dt = 1.0 / fps
    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    wz = vel.get("wz", 0.0)

    # Number of constant-velocity frames and deceleration frames
    if decel_frac > 0 and num_frames > 5:
        n_decel = max(1, int(num_frames * decel_frac))
        n_const = num_frames - n_decel
    else:
        n_decel = 0
        n_const = num_frames

    t = np.arange(num_frames) * dt
    t_const = t[:n_const]  # constant-velocity time

    # ── Build base positions (constant-velocity path) ──────────────
    if abs(wz) < 1e-6:
        t_full = np.arange(n_const) * dt
        x_base = vx * t_full
        z_base = vy * t_full
        headings_raw = None
    else:
        t_full = np.arange(n_const) * dt
        # Exact integration for circular arc
        x_base = np.where(
            abs(vx) > 1e-6,
            (vx * np.sin(wz * t_full) + vy * (1 - np.cos(wz * t_full))) / wz,
            vx * t_full,
        )
        z_base = np.where(
            abs(vy) > 1e-6,
            (vy * np.sin(wz * t_full) - vx * (1 - np.cos(wz * t_full))) / wz,
            vy * t_full,
        )
        frame_headings = heading + wz * t_full
        headings_raw = np.stack(
            [np.cos(frame_headings), np.sin(frame_headings)], axis=-1
        )

    # ── Deceleration easing ────────────────────────────────────────
    if n_decel > 0:
        # End position of constant-velocity segment
        x_end = float(x_base[-1]) if n_const > 0 else 0.0
        z_end = float(z_base[-1]) if n_const > 0 else 0.0

        # If the motion continued at constant speed, where would it end?
        t_total = num_frames * dt
        if abs(wz) < 1e-6:
            x_final = vx * t_total
            z_final = vy * t_total
        else:
            # position at total time (scalar)
            x_final = (
                (vx * np.sin(wz * t_total) + vy * (1 - np.cos(wz * t_total))) / wz
            )
            z_final = (
                (vy * np.sin(wz * t_total) - vx * (1 - np.cos(wz * t_total))) / wz
            )

        # Build decel positions: linear interpolation between x_end and x_final,
        # then apply ease-out warping so velocity goes to 0 smoothly.
        frac = np.linspace(0.0, 1.0, n_decel + 1)[1:]  # (n_decel,)
        ease = _ease_out_cosine(frac)

        # The ease factor tells us how far along the "remaining path" we are.
        # At frac=0 → ease=1 (full speed), at frac=1 → ease=0 (stopped).
        # Position = final_pos - ease * remaining_distance
        x_decel = x_final - ease * (x_final - x_end)
        z_decel = z_final - ease * (z_final - z_end)

        x = np.concatenate([x_base, x_decel])
        z = np.concatenate([z_base, z_decel])

        if headings_raw is not None:
            # For turning: heading keeps rotating but at reduced rate in decel
            heading_end_rad = heading + wz * t_total
            # Ease the heading change rate
            heading_decel_frac = _ease_out_cosine(
                np.linspace(0, 1, n_decel + 1)[1:]
            )
            # Heading at each decel frame: interpolate angular position
            heading_const_end = heading + wz * (n_const * dt)
            # Eased heading: gradually stop rotating
            heading_decel = heading_const_end + wz * dt * np.cumsum(heading_decel_frac)
            all_headings_rad = np.concatenate([
                heading + wz * t_const,
                heading_decel,
            ])
            headings = np.stack(
                [np.cos(all_headings_rad), np.sin(all_headings_rad)], axis=-1
            )
        else:
            headings = None
    else:
        x = x_base
        z = z_base
        headings = headings_raw
        if headings_raw is not None:
            heading_arr = heading + wz * t
            headings = np.stack(
                [np.cos(heading_arr), np.sin(heading_arr)], axis=-1
            )

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
