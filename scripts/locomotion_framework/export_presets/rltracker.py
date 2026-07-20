"""RLTrackerDataset export preset.

Converts Kimodo output NPZ keys into the RLTracker motion dataset format
matching the structure of:

    selected_motions/
      {motion_name}__{actor_id}[_M]/
        motion.npz

The target NPZ has 7 keys (all float32):
    fps               (1,)       – frames per second
    joint_pos         (T, 29)    – 29 hinge joint angles (radians)
    joint_vel         (T, 29)    – joint angular velocities
    body_pos_w        (T, 30, 3) – 30 body world positions (m, MuJoCo: z-up x-fwd)
    body_quat_w       (T, 30, 4) – 30 body world quaternions (w,x,y,z)
    body_lin_vel_w    (T, 30, 3) – body linear velocities
    body_ang_vel_w    (T, 30, 3) – body angular velocities

All conversion is done from Kimodo FK output — no MuJoCo Python bindings needed.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from . import register, ExportPreset

# Kimodo → MuJoCo coordinate transform (from mujoco.py L77-79)
# Kimodo: y-up, z-forward  →  MuJoCo: z-up, x-forward
_KIMODO_TO_MUJOCO = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
_MUJOCO_TO_KIMODO = _KIMODO_TO_MUJOCO.T  # inverse


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _compute_trajectory_type(vel: dict[str, float]) -> str:
    """Classify trajectory from velocity command."""
    vx = abs(vel.get("vx", 0.0))
    vy = abs(vel.get("vy", 0.0))
    wz = abs(vel.get("wz", 0.0))

    if max(vx, vy, wz) < 0.03:
        return "still"
    if wz > 0.15 and wz > max(vx, vy) * 1.5:
        return "turn"
    if vy > vx * 1.2 and vy > 0.05:
        return "lat"
    if vel.get("vx", 0) < -0.05:
        return "bwd"
    return "fwd"


def _compute_heading_deg(vel: dict[str, float]) -> int:
    """Compute heading in degrees [0, 360) from velocity direction."""
    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    deg = int(round(np.arctan2(vy, vx) * 180.0 / np.pi))
    return deg % 360


def _compute_pace(vel: dict[str, float]) -> str:
    """Map speed to qualitative pace descriptor."""
    vx = abs(vel.get("vx", 0.0))
    vy = abs(vel.get("vy", 0.0))
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 0.03:
        return "still"
    elif speed < 0.30:
        return "slow"
    elif speed < 0.80:
        return "norm"
    elif speed < 1.50:
        return "fast"
    else:
        return "sprt"


def build_motion_name(
    motion_type: str,
    vel: dict[str, float],
    variant_idx: int,
    seed: int,
) -> str:
    """Build RLTracker-style directory name from motion parameters.

    Returns a name like ``walk_fwd_000_norm_0001__K42``.
    """
    # Map internal type names to RLTracker abbreviations
    type_map = {
        "walk": "walk",
        "run": "run",
        "stand": "stand",
        "squat_move": "squat",
    }
    type_abbr = type_map.get(motion_type, motion_type)

    traj = _compute_trajectory_type(vel)
    heading = _compute_heading_deg(vel)
    pace = _compute_pace(vel)

    return f"{type_abbr}_{traj}_{heading:03d}_{pace}_{variant_idx:04d}__K{seed}"


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _mat_to_quat_wxyz(rot_mats: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (..., 3, 3) to quaternions (..., 4) w,x,y,z.

    Uses the standard algorithm that avoids division-by-zero for near-zero
    trace.
    """
    shape = rot_mats.shape[:-2]
    r = rot_mats.reshape(-1, 3, 3)
    n = r.shape[0]

    q = np.zeros((n, 4), dtype=np.float32)

    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]

    # Case 1: trace > 0
    mask0 = trace > 0.0
    if mask0.any():
        s = 0.5 / np.sqrt(trace[mask0] + 1.0)
        q[mask0, 0] = 0.25 / s
        q[mask0, 1] = (r[mask0, 2, 1] - r[mask0, 1, 2]) * s
        q[mask0, 2] = (r[mask0, 0, 2] - r[mask0, 2, 0]) * s
        q[mask0, 3] = (r[mask0, 1, 0] - r[mask0, 0, 1]) * s

    # Case 2: r00 is max diagonal
    mask1 = ~mask0 & (r[:, 0, 0] >= r[:, 1, 1]) & (r[:, 0, 0] >= r[:, 2, 2])
    if mask1.any():
        s = 2.0 * np.sqrt(
            np.maximum(
                1.0 + r[mask1, 0, 0] - r[mask1, 1, 1] - r[mask1, 2, 2], 0.0
            )
        )
        q[mask1, 1] = 0.25 * s
        q[mask1, 0] = (r[mask1, 2, 1] - r[mask1, 1, 2]) / s
        q[mask1, 2] = (r[mask1, 0, 1] + r[mask1, 1, 0]) / s
        q[mask1, 3] = (r[mask1, 0, 2] + r[mask1, 2, 0]) / s

    # Case 3: r11 is max diagonal
    mask2 = ~mask0 & ~mask1 & (r[:, 1, 1] >= r[:, 2, 2])
    if mask2.any():
        s = 2.0 * np.sqrt(
            np.maximum(
                1.0 + r[mask2, 1, 1] - r[mask2, 0, 0] - r[mask2, 2, 2], 0.0
            )
        )
        q[mask2, 2] = 0.25 * s
        q[mask2, 0] = (r[mask2, 0, 2] - r[mask2, 2, 0]) / s
        q[mask2, 1] = (r[mask2, 0, 1] + r[mask2, 1, 0]) / s
        q[mask2, 3] = (r[mask2, 1, 2] + r[mask2, 2, 1]) / s

    # Case 4: r22 is max diagonal
    mask3 = ~mask0 & ~mask1 & ~mask2
    if mask3.any():
        s = 2.0 * np.sqrt(
            np.maximum(
                1.0 + r[mask3, 2, 2] - r[mask3, 0, 0] - r[mask3, 1, 1], 0.0
            )
        )
        q[mask3, 3] = 0.25 * s
        q[mask3, 0] = (r[mask3, 1, 0] - r[mask3, 0, 1]) / s
        q[mask3, 1] = (r[mask3, 0, 2] + r[mask3, 2, 0]) / s
        q[mask3, 2] = (r[mask3, 1, 2] + r[mask3, 2, 1]) / s

    # Normalize
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)

    return q.reshape(*shape, 4)


def _quat_angular_velocity(
    quats: np.ndarray, dt: float
) -> np.ndarray:
    """Compute angular velocity from a sequence of quaternions (T, N, 4) wxyz.

    Uses finite-difference of the quaternion derivative:
        omega = 2 * dq/dt * q^{-1}
    taking the vector part.
    """
    T = quats.shape[0]
    if T < 2:
        return np.zeros_like(quats[..., :3])

    # q(t+1) * q(t)^-1  where q^-1 = (w, -x, -y, -z) for unit quaternions
    q_next = quats[1:]   # (T-1, N, 4)
    q_curr = quats[:-1]  # (T-1, N, 4)
    q_curr_inv = q_curr.copy()
    q_curr_inv[..., 1:] *= -1.0

    # Quaternion multiply: q_next * q_curr^{-1}
    w0, x0, y0, z0 = (
        q_next[..., 0], q_next[..., 1], q_next[..., 2], q_next[..., 3]
    )
    w1, x1, y1, z1 = (
        q_curr_inv[..., 0],
        q_curr_inv[..., 1],
        q_curr_inv[..., 2],
        q_curr_inv[..., 3],
    )
    dq_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    dq_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    dq_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    dq_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # omega = (2/dt) * vector_part(dq), scaled by sign of scalar part
    sign = np.sign(dq_w)[..., None]
    omega = (2.0 / dt) * sign * np.stack([dq_x, dq_y, dq_z], axis=-1)

    # Duplicate first/last frame for same-length output
    result = np.zeros_like(quats[..., :3])
    result[0] = omega[0]
    result[-1] = omega[-1]
    result[1:-1] = 0.5 * (omega[:-1] + omega[1:])

    return result


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

@register("rltracker")
class RLTrackerExporter:
    """Export Kimodo-generated motions in RLTracker dataset format."""

    name = "rltracker"
    description = (
        "RLTracker dataset format: flat dirs with motion.npz containing "
        "joint_pos (29), joint_vel (29), body_pos_w (30), body_quat_w (30), "
        "body_lin_vel_w (30), body_ang_vel_w (30), plus fps."
    )

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        """Lazy-load the MujocoQposConverter (cached per skeleton)."""
        if self._converter is None:
            from kimodo.exports.mujoco import MujocoQposConverter
            from kimodo.skeleton.registry import build_skeleton

            skeleton = build_skeleton(34)  # G1Skeleton34
            self._converter = MujocoQposConverter(skeleton)
        return self._converter

    def _kimodo_output_to_rltracker(
        self,
        posed_joints: np.ndarray,       # (T, 34, 3)
        global_rot_mats: np.ndarray,    # (T, 34, 3, 3)
        local_rot_mats: np.ndarray,     # (T, 34, 3, 3)
        root_positions: np.ndarray,     # (T, 3)
        fps: float,
    ) -> dict[str, np.ndarray]:
        """Convert Kimodo FK output to RLTracker NPZ dict."""
        import torch

        converter = self._get_converter()

        T = posed_joints.shape[0]
        dt = 1.0 / fps

        # -- joint_pos: 29 hinge angles via MujocoQposConverter ----------
        local_t = torch.from_numpy(local_rot_mats).unsqueeze(0)  # (1, T, 34, 3, 3)
        root_t = torch.from_numpy(root_positions).unsqueeze(0)    # (1, T, 3)
        qpos = converter.to_qpos(local_t, root_t, root_quat_w_first=True)
        qpos_np = qpos.squeeze(0).cpu().numpy()  # (T, 36)
        joint_pos = qpos_np[:, 7:].astype(np.float32)  # (T, 29)

        # -- joint_vel: finite differences ---------------------------------
        joint_vel = np.gradient(joint_pos, axis=0) / dt
        joint_vel = joint_vel.astype(np.float32)

        # -- body_pos_w: map 34 Kimodo joints → 30 MuJoCo bodies ----------
        # Converter mapping arrays (numpy copies)
        kim_to_mj = (
            converter._kimodo_indices_to_mujoco_indices.cpu().numpy()
        )  # (34,)  mujoco body index or -1
        mj_to_kim = (
            converter._mujoco_indices_to_kimodo_indices.cpu().numpy()
        )  # (29,)  kimodo joint index for each hinge

        body_pos_w = np.zeros((T, 30, 3), dtype=np.float32)

        # Body 0 = pelvis (root)
        body_pos_w[:, 0, :] = root_positions @ _MUJOCO_TO_KIMODO.T
        # Alternative: use qpos root translation directly
        # qpos_np[:, :3] is root in Mujoco space — use this instead since it's already correct
        body_pos_w[:, 0, :] = qpos_np[:, :3].astype(np.float32)

        # Bodies 1..29 = hinge joints
        # posed_joints is in Kimodo space; transform to MuJoCo space
        posed_mj = posed_joints @ _MUJOCO_TO_KIMODO.T  # (T, 34, 3)
        for mj_hinge_idx, kim_joint_idx in enumerate(mj_to_kim):
            if kim_joint_idx >= 0:
                body_pos_w[:, mj_hinge_idx + 1, :] = posed_mj[:, int(kim_joint_idx), :]

        # -- body_quat_w: global rotations → MuJoCo space → quaternion -----
        # R_mujoco = kimodo_to_mujoco @ R_kimodo @ mujoco_to_kimodo
        body_quat_w = np.zeros((T, 30, 4), dtype=np.float32)

        # Root quaternion (from qpos for accuracy)
        body_quat_w[:, 0, :] = qpos_np[:, 3:7].astype(np.float32)  # (w,x,y,z)

        # Hinge joint bodies
        for mj_hinge_idx, kim_joint_idx in enumerate(mj_to_kim):
            if kim_joint_idx < 0:
                continue
            Rk = global_rot_mats[:, int(kim_joint_idx), :, :]  # (T, 3, 3)
            # Transform to MuJoCo space
            Rm = _KIMODO_TO_MUJOCO @ Rk @ _MUJOCO_TO_KIMODO  # (T, 3, 3)
            q = _mat_to_quat_wxyz(Rm)  # (T, 4) w,x,y,z
            body_quat_w[:, mj_hinge_idx + 1, :] = q

        # Normalize quaternions
        norms = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
        body_quat_w = body_quat_w / np.maximum(norms, 1e-12)

        # -- body_lin_vel_w: finite differences of body_pos_w --------------
        body_lin_vel_w = np.gradient(body_pos_w, axis=0) / dt
        body_lin_vel_w = body_lin_vel_w.astype(np.float32)

        # -- body_ang_vel_w: from quaternion changes ----------------------
        body_ang_vel_w = _quat_angular_velocity(body_quat_w, dt).astype(np.float32)

        return {
            "fps": np.array([fps], dtype=np.float32),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
        }

    def export(
        self,
        *,
        posed_joints: np.ndarray,
        global_rot_mats: np.ndarray,
        local_rot_mats: np.ndarray,
        root_positions: np.ndarray,
        fps: float,
        sample_idx: int,
        motion_type: str,
        vel: dict[str, float],
        torso_height: float,
        style: str,
        seed: int,
        output_base: Path,
    ) -> Path:
        """Export one Kimodo-generated motion to RLTracker format.

        Returns the output directory path.
        """
        name = build_motion_name(motion_type, vel, sample_idx, seed)
        out_dir = output_base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        rl_dict = self._kimodo_output_to_rltracker(
            posed_joints=posed_joints,
            global_rot_mats=global_rot_mats,
            local_rot_mats=local_rot_mats,
            root_positions=root_positions,
            fps=fps,
        )

        np.savez_compressed(str(out_dir / "motion.npz"), **rl_dict)

        return out_dir
