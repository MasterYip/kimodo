"""Convert sampled velocity commands to Kimodo Root2D constraints.

Velocity → (x,z) path via constant-velocity integration.

For **pure locomotion clips** (no start/end phase), the path is built at
constant speed for the exact generation duration.  No acceleration or
deceleration — the constraint and the diffusion motion prior are perfectly
aligned, eliminating the soft-constraint conflict that causes root jitter.

Optionally, only every ``stride``-th frame is constrained (sparse mode),
giving the diffusion model room for natural COM oscillation while
following the overall trajectory.
"""

import numpy as np

from .sampler import SampledMotion

# Constrain EVERY frame (dense, like demo 05_root_path).
# Dense inpainting at every DDIM step ensures the root follows the path
# while the diffusion model naturally produces kinematics consistent with
# that root position — no post-processing needed, feet stay planted.
_CONSTRAINT_STRIDE = 1


def build_root2d_constraint(
    vel: dict[str, float],
    duration: float,
    fps: int = 30,
    heading: float = 0.0,
    stride: int = _CONSTRAINT_STRIDE,
) -> dict:
    """Build a Root2D constraint from a velocity command.

    The path is pure constant-velocity — no cosine easing, no stretch.
    The model sees root moving at steady speed from the first constrained
    frame to the last, matching its training prior for walk/run/squat loops.

    Args:
        vel:      {"vx": float, "vy": float, "wz": float}  body-frame
        duration: motion duration in seconds.
        fps:      frames per second.
        heading:  initial world heading (radians).
        stride:   constrain every ``stride``-th frame (1 = all frames).
    """
    num_frames = int(duration * fps)
    dt = 1.0 / fps

    vx = vel.get("vx", 0.0)   # forward  (body-frame)
    vy = vel.get("vy", 0.0)   # lateral  (body-frame)  +right
    wz = vel.get("wz", 0.0)   # angular  (rad/s)        +right turn

    t = np.arange(num_frames, dtype=np.float64) * dt

    # ── Constant-speed integration ──────────────────────────────────
    if abs(wz) < 1e-6:
        # Straight path.
        # Kimodo X = lateral  (world) ← vy  (body lateral)
        # Kimodo Z = forward  (world) ← vx  (body forward)
        x = vy * t
        z = vx * t
        headings = None
    else:
        # Curved path.
        # Body-frame velocity (vx, vy) rotates at rate wz.
        # At each frame the body heading is θ(t) = θ₀ + wz·t.
        #
        # World-frame velocity from body-frame (vx, vy):
        #     v_X_world =  vy·cos θ + vx·sin θ    (lateral in world)
        #     v_Z_world =  vx·cos θ − vy·sin θ    (forward in world)
        #
        # Positions obtained by integration (cumulative sum).
        theta_0 = np.arctan2(vy, vx)
        theta = theta_0 + wz * t

        v_X = vy * np.cos(theta) + vx * np.sin(theta)
        v_Z = vx * np.cos(theta) - vy * np.sin(theta)

        x = np.cumsum(v_X) * dt
        z = np.cumsum(v_Z) * dt

        # Global root heading per frame
        frame_headings = heading + wz * t
        headings = np.stack(
            [np.cos(frame_headings), np.sin(frame_headings)], axis=-1
        ).astype(np.float32)

    # ── Frame selection ─────────────────────────────────────────
    stride = max(1, stride)
    constrain_indices = list(range(0, num_frames, stride))

    # Always include the first and last frame
    if constrain_indices[-1] != num_frames - 1:
        constrain_indices.append(num_frames - 1)

    smooth_root_2d = np.stack([x, z], axis=-1).astype(np.float32)
    smooth_root_2d = smooth_root_2d[constrain_indices].tolist()

    constraint = {
        "type": "root2d",
        "frame_indices": constrain_indices,
        "smooth_root_2d": smooth_root_2d,
    }

    if headings is not None:
        constraint["global_root_heading"] = (
            headings[constrain_indices].tolist()
        )

    return constraint


def build_constraints_json(
    sample: SampledMotion,
    fps: int = 30,
    duration_override: float | None = None,
    stride: int | None = None,
    enabled: bool = True,
) -> list[dict]:
    """Build the full constraints list for a sampled motion.

    Args:
        sample:            The sampled motion spec.
        fps:               Frames per second.
        duration_override: If set, build constraints for this duration
                           instead of ``sample.duration`` (used for
                           generate-and-truncate mode).
        stride:            Override constraint stride (None = use default).
                           stride=1 for dense, stride=3+ for sparse.
        enabled:           False emits no Root2D while preserving ``sample.vel``.
    """
    constraints = []
    duration = duration_override if duration_override is not None else sample.duration
    kw = {"stride": stride} if stride is not None else {}

    if enabled and sample.vel and any(abs(v) > 1e-6 for v in sample.vel.values()):
        root_constraint = build_root2d_constraint(
            sample.vel, duration, fps=fps, **kw
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
