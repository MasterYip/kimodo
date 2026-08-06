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


def _load_heading_override(path: str, num_frames: int) -> np.ndarray:
    """Load a per-frame ``global_root_heading`` array from a native npz/npy.

    Mirrors the CONSTRAINT-003 C5 recipe: the reference array follows the
    Kimodo convention ``(cos θ, sin θ)`` per frame and is taken from a paired
    native motion. Each source row is normalized as ``h / max(||h||, 1e-8)``
    before being returned. Only the first ``num_frames`` rows are kept.

    Args:
        path:       Path to a ``.npz`` (key ``global_root_heading``) or a
                    raw ``.npy`` array of shape ``(T, 2)``.
        num_frames: Number of frames the current motion needs.

    Returns:
        float32 array of shape ``(num_frames, 2)`` with unit rows.
    """
    _data = np.load(path, allow_pickle=False)
    if isinstance(_data, np.lib.npyio.NpzFile):
        if "global_root_heading" not in _data.files:
            raise ValueError(
                f"heading file {path} has no 'global_root_heading' key (got {_data.files})"
            )
        arr = np.asarray(_data["global_root_heading"], dtype=np.float64)
    else:
        arr = np.asarray(_data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"heading file {path}: expected shape (T,2), got {arr.shape}")
    if arr.shape[0] < num_frames:
        raise ValueError(
            f"heading file {path}: {arr.shape[0]} frames < required {num_frames}"
        )
    arr = arr[:num_frames]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-8)
    return arr.astype(np.float32)


def build_root2d_constraint(
    vel: dict[str, float],
    duration: float,
    fps: int = 30,
    heading: float = 0.0,
    stride: int = _CONSTRAINT_STRIDE,
    density: str | None = None,
    heading_override: np.ndarray | None = None,
) -> dict:
    """Build a Root2D constraint from a velocity command.

    The path is pure constant-velocity — no cosine easing, no stretch.
    The model sees root moving at steady speed from the first constrained
    frame to the last, matching its training prior for walk/run/squat loops.

    Args:
        vel:              {"vx": float, "vy": float, "wz": float}  body-frame
        duration:         motion duration in seconds.
        fps:              frames per second.
        heading:          initial world heading (radians).
        stride:           constrain every ``stride``-th frame (1 = all frames).
                          Used when ``density`` is None.
        density:          "dense" (all frames), "stride_N" (every Nth frame +
                          last), "endpoint" (first + last only), or None to
                          derive from ``stride``.
        heading_override: optional (T,2) ``global_root_heading`` array (unit
                          rows) to inject per constrained frame, exactly as the
                          C5 recipe. Overrides the wz-derived heading.
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

    # ── Frame selection (density) ───────────────────────────────
    if density == "endpoint":
        # First + last frame only (CONSTRAINT-003 C4)
        constrain_indices = [0, num_frames - 1]
    elif density == "dense":
        # Every frame (CONSTRAINT-003 C1)
        constrain_indices = list(range(0, num_frames, 1))
    elif density is not None and density.startswith("stride_"):
        # Every Nth frame + last (CONSTRAINT-003 C2/C3)
        _s = max(1, int(density.split("_", 1)[1]))
        constrain_indices = list(range(0, num_frames, _s))
        if constrain_indices[-1] != num_frames - 1:
            constrain_indices.append(num_frames - 1)
    else:
        # Global stride (backward compatible)
        stride = max(1, stride)
        constrain_indices = list(range(0, num_frames, stride))
        if constrain_indices[-1] != num_frames - 1:
            constrain_indices.append(num_frames - 1)

    # C5-style reference-heading injection overrides the wz-derived heading.
    if heading_override is not None:
        if heading_override.shape[0] != num_frames:
            raise ValueError(
                f"heading_override has {heading_override.shape[0]} frames, "
                f"expected {num_frames}"
            )
        headings = heading_override[constrain_indices]
    elif headings is not None:
        # Slice the wz-derived full-frame heading to the constrained frames.
        headings = headings[constrain_indices]

    smooth_root_2d = np.stack([x, z], axis=-1).astype(np.float32)
    smooth_root_2d = smooth_root_2d[constrain_indices].tolist()

    constraint = {
        "type": "root2d",
        "frame_indices": constrain_indices,
        "smooth_root_2d": smooth_root_2d,
    }

    if headings is not None:
        constraint["global_root_heading"] = headings.tolist()

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
        density = getattr(sample, "root2d_density", None)
        heading_file = getattr(sample, "root2d_heading_file", None)
        heading_override = None
        if heading_file:
            heading_override = _load_heading_override(
                heading_file, int(duration * fps)
            )
        root_constraint = build_root2d_constraint(
            sample.vel,
            duration,
            fps=fps,
            density=density,
            heading_override=heading_override,
            **kw,
        )
        constraints.append(root_constraint)

    # Optional native left-hand/right-hand EndEffector constraint set.
    # Sample-level path to a JSON list of constraint dicts (schema emitted by
    # EndEffectorConstraintSet.get_save_info). Loaded and validated downstream
    # by kimodo.constraints.load_constraints_lst.
    hand_file = getattr(sample, "hand_constraints_file", None)
    if hand_file:
        import json as _json

        with open(hand_file) as _f:
            hand_constraints = _json.load(_f)
        if not isinstance(hand_constraints, list) or not hand_constraints:
            raise ValueError(
                f"hand_constraints_file {hand_file} must be a non-empty JSON list"
            )
        constraints.extend(hand_constraints)

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
