"""RLTrackerDataset export preset.

Converts Kimodo output NPZ keys into the RLTracker motion dataset format
matching the structure of:

    selected_motions/
      {motion_name}__{actor_id}[_M]/
        motion.npz

The target NPZ has 7 keys (all float32):
    fps               (1,)       – frames per second  (50 Hz)
    joint_pos         (T, 29)    – 29 hinge joint angles in ISAACLAB DOF order
    joint_vel         (T, 29)    – joint angular velocities
    body_pos_w        (T, 30, 3) – 30 body world positions in ISAACLAB body order
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
# Joint / body name tables and reordering permutations
#
# Kimodo's MujocoQposConverter uses the G1 MuJoCo XML, which orders joints
# and bodies in **document order**.  The RLTracker reference dataset uses
# **Isaac Lab order** (alphabetical from the URDF, left/right interleaved).
#
# We build in MuJoCo order first, then permute both joint_pos and body
# arrays to Isaac Lab order.
# ---------------------------------------------------------------------------

MUJOCO_DOF_NAMES = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",       "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",      "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ISAACLAB_DOF_NAMES = [
    "left_hip_pitch_joint",  "right_hip_pitch_joint",  "waist_yaw_joint",
    "left_hip_roll_joint",   "right_hip_roll_joint",   "waist_roll_joint",
    "left_hip_yaw_joint",    "right_hip_yaw_joint",    "waist_pitch_joint",
    "left_knee_joint",       "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint","right_ankle_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint","right_shoulder_yaw_joint",
    "left_elbow_joint",      "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint","right_wrist_pitch_joint",
    "left_wrist_yaw_joint",  "right_wrist_yaw_joint",
]

# Permutation:  joint_pos_il = joint_pos_mj[:, _MJ_DOF_TO_IL]
# _MJ_DOF_TO_IL[i] = MuJoCo index of the joint that goes to IsaacLab position i
_MJ_DOF_TO_IL: list[int] = [MUJOCO_DOF_NAMES.index(name) for name in ISAACLAB_DOF_NAMES]

# MuJoCo XML document-order body names (same order as worldbody children)
MUJOCO_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",  "left_hip_roll_link",  "left_hip_yaw_link",
    "left_knee_link",       "left_ankle_pitch_link","left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link",  "right_hip_yaw_link",
    "right_knee_link",      "right_ankle_pitch_link","right_ankle_roll_link",
    "waist_yaw_link",       "waist_roll_link",       "torso_link",
    "left_shoulder_pitch_link","left_shoulder_roll_link","left_shoulder_yaw_link",
    "left_elbow_link",       "left_wrist_roll_link",  "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link","right_shoulder_roll_link","right_shoulder_yaw_link",
    "right_elbow_link",      "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

ISAACLAB_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",  "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",   "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",       "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link","right_ankle_pitch_link",
    "left_shoulder_roll_link",  "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link","right_shoulder_yaw_link",
    "left_elbow_link",      "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link","right_wrist_pitch_link",
    "left_wrist_yaw_link",  "right_wrist_yaw_link",
]

# Permutation:  body_xxx_il = body_xxx_mj[:, _MJ_BODY_TO_IL, ...]
# _MJ_BODY_TO_IL[i] = MuJoCo body index for the IsaacLab body at position i
_MJ_BODY_TO_IL: list[int] = [MUJOCO_BODY_NAMES.index(name) for name in ISAACLAB_BODY_NAMES]

N_BODIES = 30
N_DOFS = 29


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _compute_trajectory_type(vel: dict[str, float]) -> str:
    """Classify trajectory from velocity command.

    Rules (evaluated in order):
        still  – all velocity components are negligible
        turn   – in-place rotation (|wz| dominates, translation near zero)
        lat    – primarily sideways
        bwd    – primarily backward
        fwd    – everything else (including forward with incidental turning)
    """
    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    wz = abs(vel.get("wz", 0.0))
    speed = np.sqrt(vx ** 2 + vy ** 2)

    if max(speed, wz) < 0.03:
        return "still"

    # Only "turn" when the robot rotates while barely translating
    # → angular velocity is high AND translational speed is low
    if wz > 0.20 and speed < 0.15:
        return "turn"

    # Lateral when sideways velocity is the dominant translation component
    if abs(vy) > abs(vx) * 0.8 and abs(vy) > 0.05:
        return "lat"

    # Backward (vx is negative and dominates vy)
    if vx < -0.05 and abs(vx) >= abs(vy):
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
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------

def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (..., 3, 3) to quaternions (..., 4) w,x,y,z.

    Numerically stable Shepperd method.
    """
    *batch, _, _ = R.shape
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    qw = np.zeros(batch, dtype=np.float32)
    qx = np.zeros_like(qw)
    qy = np.zeros_like(qw)
    qz = np.zeros_like(qw)

    # Case 1: trace > 0
    c1 = trace > 0
    s1 = np.sqrt(np.maximum(trace[c1] + 1.0, 0.0)) * 2.0
    qw[c1] = 0.25 * s1
    qx[c1] = (R[..., 2, 1][c1] - R[..., 1, 2][c1]) / s1
    qy[c1] = (R[..., 0, 2][c1] - R[..., 2, 0][c1]) / s1
    qz[c1] = (R[..., 1, 0][c1] - R[..., 0, 1][c1]) / s1

    # Case 2: R[0,0] > R[1,1], R[0,0] > R[2,2]
    c2 = ~c1 & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s2 = np.sqrt(np.maximum(1.0 + R[..., 0, 0][c2] - R[..., 1, 1][c2] - R[..., 2, 2][c2], 0.0)) * 2.0
    qx[c2] = 0.25 * s2
    qw[c2] = (R[..., 2, 1][c2] - R[..., 1, 2][c2]) / s2
    qy[c2] = (R[..., 0, 1][c2] + R[..., 1, 0][c2]) / s2
    qz[c2] = (R[..., 0, 2][c2] + R[..., 2, 0][c2]) / s2

    # Case 3: R[1,1] > R[2,2]
    c3 = ~c1 & ~c2 & (R[..., 1, 1] > R[..., 2, 2])
    s3 = np.sqrt(np.maximum(1.0 + R[..., 1, 1][c3] - R[..., 0, 0][c3] - R[..., 2, 2][c3], 0.0)) * 2.0
    qy[c3] = 0.25 * s3
    qw[c3] = (R[..., 0, 2][c3] - R[..., 2, 0][c3]) / s3
    qx[c3] = (R[..., 0, 1][c3] + R[..., 1, 0][c3]) / s3
    qz[c3] = (R[..., 1, 2][c3] + R[..., 2, 1][c3]) / s3

    # Case 4: default
    c4 = ~c1 & ~c2 & ~c3
    s4 = np.sqrt(np.maximum(1.0 + R[..., 2, 2][c4] - R[..., 0, 0][c4] - R[..., 1, 1][c4], 0.0)) * 2.0
    qz[c4] = 0.25 * s4
    qw[c4] = (R[..., 1, 0][c4] - R[..., 0, 1][c4]) / s4
    qx[c4] = (R[..., 0, 2][c4] + R[..., 2, 0][c4]) / s4
    qy[c4] = (R[..., 1, 2][c4] + R[..., 2, 1][c4]) / s4

    q = np.stack([qw, qx, qy, qz], axis=-1)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)
    return q


def _quat_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    """Compute per-body angular velocity from quaternion sequence.

    Uses central-difference SO(3) derivative:  ω = log(R_rel) / (2*dt)
    where R_rel = R(t+1) * R(t-1)^T, matching the reference converter.

    Args:
        quats:  [T, N, 4]  wxyz body quaternions in world frame.
        dt:     Time step in seconds.

    Returns:
        [T, N, 3] angular velocities in world frame (rad/s).
    """
    T = quats.shape[0]
    if T < 3:
        return np.zeros((T, quats.shape[1], 3), dtype=np.float32)

    # Central differences on SO(3):  q_rel = q(t+1) * q(t-1)^-1
    q_prev = quats[:-2]  # [T-2, N, 4]
    q_next = quats[2:]   # [T-2, N, 4]

    # Conjugate: q^-1 = (w, -x, -y, -z)
    q_prev_inv = q_prev.copy()
    q_prev_inv[..., 1:] *= -1.0

    # Multiply: q_rel = q_next * q_prev^-1
    w0, x0, y0, z0 = q_next[..., 0], q_next[..., 1], q_next[..., 2], q_next[..., 3]
    w1, x1, y1, z1 = q_prev_inv[..., 0], q_prev_inv[..., 1], q_prev_inv[..., 2], q_prev_inv[..., 3]
    q_rel_w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    q_rel_x = w0*x1 + x0*w1 + y0*z1 - z0*y1
    q_rel_y = w0*y1 - x0*z1 + y0*w1 + z0*x1
    q_rel_z = w0*z1 + x0*y1 - y0*x1 + z0*w1

    # axis-angle:  θ = 2*acos(|w|),  axis = vec / |vec|
    # ω = axis * θ / (2*dt)
    # Short arc: ensure w >= 0
    sign = np.sign(q_rel_w)
    q_rel_w_abs = np.abs(q_rel_w)
    n = np.sqrt(np.maximum(q_rel_x**2 + q_rel_y**2 + q_rel_z**2, 1e-12))
    theta = 2.0 * np.arctan2(n, q_rel_w_abs)
    omega = (theta / (2.0 * dt))[..., None] * np.stack([
        sign * q_rel_x / n, sign * q_rel_y / n, sign * q_rel_z / n
    ], axis=-1)

    # Pad to full T
    result = np.zeros((T, quats.shape[1], 3), dtype=np.float32)
    result[0] = omega[0]
    result[-1] = omega[-1]
    result[1:-1] = omega

    return result


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


def _sanitize_dict(rl_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace NaN/Inf in output arrays with safe fallback values.

    NaN can appear in wrist-joint angles and their derived FK positions
    when rotation matrices hit degenerate cases during resampling or
    joint-angle extraction.  We fill NaN with 0 (joints/velocities),
    identity quaternion, or nearest-valid-frame (body positions).
    """
    for key in list(rl_dict.keys()):
        arr = rl_dict[key]
        if not np.isnan(arr).any() and not np.isinf(arr).any():
            continue

        if key == "fps":
            continue

        if key in ("joint_pos", "joint_vel"):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        elif key == "body_quat_w":
            nan_mask = np.isnan(arr).any(axis=-1) | np.isinf(arr).any(axis=-1)
            arr[nan_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            zero_mask = np.abs(arr).sum(axis=-1) < 1e-8
            arr[zero_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
            arr = arr.copy()
            T = arr.shape[0]
            for t in range(T):
                frame_mask = (
                    np.isnan(arr[t]).any(axis=-1) | np.isinf(arr[t]).any(axis=-1)
                )
                if frame_mask.any():
                    arr[t, frame_mask] = (
                        arr[t - 1, frame_mask] if t > 0 else 0.0
                    )

        rl_dict[key] = arr.astype(np.float32)

    return rl_dict


@register("rltracker")
class RLTrackerExporter:
    """Export Kimodo-generated motions in RLTracker dataset format.

    Converts Kimodo FK output (y-up, z-forward, 34 joints) to the
    RLTracker dataset convention (z-up, x-forward, 30 bodies, 29 DOFs,
    IsaacLab ordering).
    """

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
        posed_joints: np.ndarray,       # (T, 34, 3)  Kimodo world-frame joint positions
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

        # ── joint_pos: 29 hinge angles via MujocoQposConverter ──
        # to_qpos with mujoco_rest_zero=False gives raw joint DOFs (Euler extraction,
        # numerically stable). We then manually subtract the rest DOFs to get
        # T-pose-relative angles, avoiding the axis-angle NaN path that
        # mujoco_rest_zero=True triggers for certain wrist rotations.
        local_t = torch.from_numpy(local_rot_mats).unsqueeze(0)  # (1, T, 34, 3, 3)
        root_t = torch.from_numpy(root_positions).unsqueeze(0)    # (1, T, 3)
        qpos = converter.to_qpos(
            local_t, root_t, root_quat_w_first=True, mujoco_rest_zero=False
        )
        qpos_np = qpos.squeeze(0).cpu().numpy()  # (T, 36)

        # Columns 7:36 = 29 hinge angles in MJ order
        joint_dofs_raw = qpos_np[:, 7:].astype(np.float32)  # (T, 29)

        # Subtract rest DOFs manually (stable Euler path, same values but no NaN)
        rest_dofs = converter._rest_dofs.cpu().numpy().astype(np.float32)  # (29,)
        joint_pos_mj = joint_dofs_raw - rest_dofs[np.newaxis, :]  # (T, 29) MJ order

        # Permute to IsaacLab order
        joint_pos = joint_pos_mj[:, _MJ_DOF_TO_IL]             # (T, 29) IL order

        # ── joint_vel: finite differences ──
        joint_vel = np.gradient(joint_pos, axis=0) / dt
        joint_vel = joint_vel.astype(np.float32)

        # ── Kimodo → MuJooco coordinate transform ──
        # mj_to_kim[i] = kimodo_joint_idx for MuJoCo hinge i (bodies i+1)
        mj_to_kim = converter._mujoco_indices_to_kimodo_indices.cpu().numpy()  # (29,)

        # posed_joints in MuJoCo space:  (T, 34, 3)
        posed_mj = posed_joints @ _MUJOCO_TO_KIMODO.T

        # ── body_pos_w in MuJoCo body order ──
        body_pos_mj = np.zeros((T, N_BODIES, 3), dtype=np.float32)

        # Body 0 = pelvis (root) — use qpos translation (already in MuJoCo space)
        body_pos_mj[:, 0, :] = qpos_np[:, :3].astype(np.float32)

        # Bodies 1..29 — hinge joints: body = hinge_idx + 1
        for hinge_idx, kimodo_joint_idx in enumerate(mj_to_kim):
            if kimodo_joint_idx >= 0:
                body_pos_mj[:, hinge_idx + 1, :] = posed_mj[:, int(kimodo_joint_idx), :]

        # Permute to IsaacLab body order
        body_pos_w = body_pos_mj[:, _MJ_BODY_TO_IL, :]

        # ── body_quat_w in MuJoCo body order ──
        # R_mujoco = kimodo_to_mujoco @ R_kimodo @ mujoco_to_kimodo
        body_quat_mj = np.zeros((T, N_BODIES, 4), dtype=np.float32)

        # Root quaternion from qpos
        body_quat_mj[:, 0, :] = qpos_np[:, 3:7].astype(np.float32)

        for hinge_idx, kimodo_joint_idx in enumerate(mj_to_kim):
            if kimodo_joint_idx >= 0:
                Rk = global_rot_mats[:, int(kimodo_joint_idx), :, :]
                Rm = _KIMODO_TO_MUJOCO @ Rk @ _MUJOCO_TO_KIMODO
                body_quat_mj[:, hinge_idx + 1, :] = _mat_to_quat_wxyz(Rm)

        norms = np.linalg.norm(body_quat_mj, axis=-1, keepdims=True)
        body_quat_mj = body_quat_mj / np.maximum(norms, 1e-12)

        # Permute to IsaacLab body order
        body_quat_w = body_quat_mj[:, _MJ_BODY_TO_IL, :]

        # ── body_lin_vel_w ──
        body_lin_vel_w = np.gradient(body_pos_w, axis=0) / dt
        body_lin_vel_w = body_lin_vel_w.astype(np.float32)

        # ── body_ang_vel_w (SO(3) central differences) ──
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

        # Sanitize: replace NaN/Inf with safe values
        rl_dict = _sanitize_dict(rl_dict)

        np.savez_compressed(str(out_dir / "motion.npz"), **rl_dict)

        return out_dir
