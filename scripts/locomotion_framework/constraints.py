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
from pathlib import Path

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
    emit_heading: bool | None = None,
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
        emit_heading: ``global_root_heading`` emission control.
            - True  → always emit (straight paths use the constant ``heading``;
                      arcs use the rotating ``heading + wz·t``).
            - False → never emit.
            - None  → original behaviour (straight paths: none; arcs: emit).
    """
    num_frames = int(duration * fps)
    dt = 1.0 / fps

    vx = vel.get("vx", 0.0)   # forward  (world frame; polar native compass)
    vy = vel.get("vy", 0.0)   # lateral  (world frame; +left in native compass)
    wz = vel.get("wz", 0.0)   # angular  (rad/s; +left / CCW)

    t = np.arange(num_frames, dtype=np.float64) * dt

    # ── Constant-speed integration ──────────────────────────────────
    if abs(wz) < 1e-6:
        # Straight path.
        # Kimodo X = lateral  (world) ← vy  (body lateral)
        # Kimodo Z = forward  (world) ← vx  (body forward)
        x = vy * t
        z = vx * t
        if emit_heading is True:
            headings = np.stack(
                [np.cos(heading), np.sin(heading)], axis=-1
            ).astype(np.float32)
            headings = np.tile(headings, (num_frames, 1))
        else:
            headings = None
    else:
        # Curved path.
        # (vx, vy) are the WORLD-frame planar velocity components (native
        # compass: vx = +forward, vy = +left, from polar_to_cartesian).  The
        # travel direction rotates at rate wz: θ(t) = θ₀ + wz·t, where
        # θ₀ = atan2(vy, vx) is the INITIAL travel direction (= the prompt
        # heading).  The world-frame velocity at time t is therefore
        #     v_X_world =  s·sin θ    (left)
        #     v_Z_world =  s·cos θ    (forward)
        # with s = |(vx, vy)|.
        #
        # FIX (DATA-KIMODO-NATURAL-LOCO-014): the previous code rotated the
        # already-world-frame components by θ (vy·cosθ + vx·sinθ, ...), which
        # double-rotated the initial direction to 2·θ₀ for any non-forward
        # heading (e.g. a "backward" arc started moving world-forward).  Only
        # forward arcs (θ₀=0) were unaffected, which is why the original code
        # passed PORT-008's forward-curve checks.  For θ₀=0 the two forms are
        # identical (v_X = s·sinθ, v_Z = s·cosθ), so this is backward-compatible
        # for all previously-validated arc configs.
        speed = float(np.hypot(vx, vy))
        theta_0 = np.arctan2(vy, vx)
        theta = theta_0 + wz * t

        v_X = speed * np.sin(theta)
        v_Z = speed * np.cos(theta)

        x = np.cumsum(v_X) * dt
        z = np.cumsum(v_Z) * dt

        # Global root heading per frame (suppressed when emit_heading is False).
        # The body faces the travel direction: initial heading θ₀ (+ any base
        # `heading` offset), rotating at wz.  For θ₀=0 this is unchanged from
        # the original `heading + wz·t`.
        if emit_heading is not False:
            frame_headings = theta_0 + heading + wz * t
            headings = np.stack(
                [np.cos(frame_headings), np.sin(frame_headings)], axis=-1
            ).astype(np.float32)
        else:
            headings = None

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


def build_keyframe_constraint(
    frame_indices: list[int],
    smooth_root_2d: list[list[float]],
    headings: list[list[float]] | None = None,
) -> dict:
    """Build a Root2D constraint dict from explicit keyframes/waypoints.

    This is the low-level builder behind the config-level ``keyframes`` and
    ``constraint_path`` mechanisms (DATA-KIMODO-FRAMEWORK-PORT-008).  The
    values are taken verbatim — the caller is responsible for the native
    coordinate convention (x +left, z +forward).

    Args:
        frame_indices: Sorted, unique frame indices within ``[0, num_frames)``.
        smooth_root_2d: ``K`` native ``[x, z]`` waypoints (one per frame index).
        headings: Optional ``K`` ``[cos, sin]`` heading vectors. If ``None`` no
            ``global_root_heading`` is emitted.
    """
    n = len(frame_indices)
    if len(smooth_root_2d) != n:
        raise ValueError("smooth_root_2d length must match frame_indices")
    if headings is not None and len(headings) != n:
        raise ValueError("global_root_heading length must match frame_indices")
    constraint: dict = {
        "type": "root2d",
        "frame_indices": [int(i) for i in frame_indices],
        "smooth_root_2d": [[float(x), float(z)] for x, z in smooth_root_2d],
    }
    if headings is not None:
        constraint["global_root_heading"] = [
            [float(c), float(s)] for c, s in headings
        ]
    return constraint


def load_constraint_path(path: str | Path) -> list[dict]:
    """Load a DISTRIBUTED-007-style Root2D constraint JSON verbatim.

    The file is a JSON list of constraint dicts, each with ``type: root2d``,
    ``frame_indices``, ``smooth_root_2d`` and optionally ``global_root_heading``
    (the same schema written by ``DATA-KIMODO-DISTRIBUTED-007/
    generate_constraints.py`` and consumed by ``kimodo --constraints``).
    Loading verbatim guarantees exact geometry parity with the native batch.

    Args:
        path: Path to the JSON file.
    """
    import json

    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"constraint path {path}: expected a non-empty JSON list")
    out = []
    for c in raw:
        if not isinstance(c, dict) or c.get("type") != "root2d":
            raise ValueError(f"constraint path {path}: only root2d constraints supported")
        fi = c["frame_indices"]
        sr = c["smooth_root_2d"]
        if len(fi) != len(sr):
            raise ValueError(f"constraint path {path}: frame_indices/smooth_root_2d length mismatch")
        entry: dict = {
            "type": "root2d",
            "frame_indices": [int(i) for i in fi],
            "smooth_root_2d": [[float(x), float(z)] for x, z in sr],
        }
        if "global_root_heading" in c:
            entry["global_root_heading"] = [
                [float(cx), float(sy)] for cx, sy in c["global_root_heading"]
            ]
        out.append(entry)
    return out


def build_constraints_json(
    sample: SampledMotion,
    fps: int = 30,
    duration_override: float | None = None,
    stride: int | None = None,
    enabled: bool = True,
) -> list[dict]:
    """Build the full constraints list for a sampled motion.

    Resolution order (first match wins):
      1. ``sample.constraint_path``  — load the DISTRIBUTED-007 constraint JSON
         verbatim (exact geometry parity).
      2. ``sample.keyframes``        — explicit ``[[frame, x, z], ...]`` waypoints.
      3. velocity path               — ``build_root2d_constraint`` from
         ``sample.vel``, honouring ``sample.heading_deg`` / ``sample.stride`` /
         ``sample.emit_heading``.

    Args:
        sample:            The sampled motion spec.
        fps:               Frames per second.
        duration_override: If set, build constraints for this duration
                           instead of ``sample.duration`` (used for
                           generate-and-truncate mode).
        stride:            Override constraint stride (None = use sample.stride,
                           then the framework default 1).
                           stride=1 for dense, stride=3+ for sparse.
        enabled:           False emits no Root2D while preserving ``sample.vel``.
    """
    # 1. Exact constraint file.
    if sample.constraint_path:
        return load_constraint_path(sample.constraint_path)

    # 2. Explicit waypoints.
    if sample.keyframes:
        fi = [int(k[0]) for k in sample.keyframes]
        sr = [[float(k[1]), float(k[2])] for k in sample.keyframes]
        return [build_keyframe_constraint(fi, sr)]

    # 3. Velocity-derived path.
    constraints = []
    duration = duration_override if duration_override is not None else sample.duration
    if stride is None:
        stride = sample.stride if sample.stride is not None else _CONSTRAINT_STRIDE
    heading_rad = (
        np.radians(sample.heading_deg) if sample.heading_deg is not None else 0.0
    )

    if enabled and sample.vel and any(abs(v) > 1e-6 for v in sample.vel.values()):
        root_constraint = build_root2d_constraint(
            sample.vel,
            duration,
            fps=fps,
            heading=heading_rad,
            stride=stride,
            emit_heading=sample.emit_heading,
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
