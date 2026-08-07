"""Deterministic locomotion metrics and pelvis-frame wrist diagnostics.

Coordinate convention: RLTracker world uses +X/+Y horizontal and +Z up.
Pelvis-local points subtract pelvis translation then apply inverse pelvis yaw,
so local +X is pelvis forward, +Y is pelvis left/lateral, and +Z is up.
Distance diagnostics use body/link centers and are not mesh-collision evidence.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

PELVIS, TORSO = 0, 9
LEFT_SHOULDER, RIGHT_SHOULDER = 22, 23
LEFT_WRIST, RIGHT_WRIST = 28, 29
LEFT_ARM, RIGHT_ARM = (22, 24, 26, 28), (23, 25, 27, 29)
HIP_LINKS = (0, 1, 2, 4, 5, 7, 8, 10, 11)
WAIST_PITCH = 8
REQUIRED = {
    "fps": (1,), "joint_pos": (None, 29), "body_pos_w": (None, 30, 3),
    "body_quat_w": (None, 30, 4),
}


def load_motion(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a native RLTracker motion NPZ."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    for key, shape in REQUIRED.items():
        if key not in data:
            raise ValueError(f"{path}: missing {key}")
        expected = tuple(data[key].shape[0] if x is None else x for x in shape)
        if data[key].shape != expected:
            raise ValueError(f"{path}: {key} shape {data[key].shape}, expected {expected}")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{path}: non-finite {key}")
    if len(data["body_pos_w"]) != len(data["joint_pos"]):
        raise ValueError(f"{path}: inconsistent frame counts")
    return data


def yaw_wxyz(quat: np.ndarray) -> np.ndarray:
    """Return yaw radians for scalar-first quaternions."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.unwrap(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))


def pitch_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True); out[..., 1:] *= -1
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), axis=-1)


def pelvis_local(points_w: np.ndarray, pelvis_pos_w: np.ndarray,
                 pelvis_quat_w: np.ndarray) -> np.ndarray:
    """Transform world points to pelvis-local +forward/+left/+up coordinates."""
    points = np.asarray(points_w, dtype=np.float64)
    origin = np.asarray(pelvis_pos_w, dtype=np.float64)
    delta = points - origin[..., None, :] if points.ndim == origin.ndim + 1 else points - origin
    yaw = yaw_wxyz(pelvis_quat_w)
    c, s = np.cos(yaw), np.sin(yaw)
    while c.ndim < delta[..., 0].ndim:
        c, s = c[..., None], s[..., None]
    x = c * delta[..., 0] + s * delta[..., 1]
    y = -s * delta[..., 0] + c * delta[..., 1]
    return np.stack((x, y, delta[..., 2]), axis=-1)


def expected_path(vx: float, vy: float, wz: float, frames: int, fps: float) -> np.ndarray:
    """Integrate framework body-frame commands into world horizontal XY."""
    out = np.zeros((frames, 2), dtype=np.float64)
    yaw = 0.0
    dt = 1.0 / fps
    for i in range(1, frames):
        yaw += wz * dt
        out[i] = out[i-1] + [vx*math.cos(yaw)-vy*math.sin(yaw),
                              vx*math.sin(yaw)+vy*math.cos(yaw)] * np.array(dt)
    return out


def _alignment_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return float("nan")
    cosine = float(np.clip(abs(np.dot(a, b) / (na*nb)), 0, 1))
    return float(np.degrees(np.arccos(cosine)))


def _nearest_fraction(a: np.ndarray, b: np.ndarray, threshold: float) -> tuple[float, float]:
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    nearest = np.min(distances, axis=1)
    return float(np.min(distances)), float(np.mean(nearest < threshold))


def evaluate_motion(path: str | Path, provenance: dict | None = None) -> tuple[dict, dict]:
    """Evaluate one motion; return scalar row and wrist-trajectory arrays."""
    provenance = dict(provenance or {})
    data = load_motion(path)
    pos = data["body_pos_w"].astype(np.float64); quat = data["body_quat_w"].astype(np.float64)
    joints = data["joint_pos"].astype(np.float64); frames = len(pos); fps = float(data["fps"][0])
    core = slice(max(0, int(.8*fps)), min(frames, frames-int(.8*fps)))
    local = pelvis_local(pos, pos[:, PELVIS], quat[:, PELVIS])
    left, right = local[:, LEFT_WRIST], local[:, RIGHT_WRIST]
    shoulder_sign = np.sign(local[:, LEFT_SHOULDER, 1] - local[:, RIGHT_SHOULDER, 1])
    wrist_sign = np.sign(left[:, 1] - right[:, 1])
    order_cross = shoulder_sign*wrist_sign < 0
    midline_cross = ((np.sign(left[:, 1])*np.sign(local[:, LEFT_SHOULDER, 1]) < 0) |
                     (np.sign(right[:, 1])*np.sign(local[:, RIGHT_SHOULDER, 1]) < 0))
    both = np.concatenate((left[core, :2], right[core, :2]), axis=0)
    cov = np.cov(both.T); values, vectors = np.linalg.eigh(cov); dominant = vectors[:, np.argmax(values)]
    forward_amp = float(np.mean([np.ptp(left[core, 0]), np.ptp(right[core, 0])]))
    lateral_amp = float(np.mean([np.ptp(left[core, 1]), np.ptp(right[core, 1])]))
    dominant_axis = "forward" if forward_amp >= lateral_amp else "lateral"
    vx, vy, wz = (float(provenance.get(k, 0.0) or 0.0) for k in ("vx", "vy", "wz"))
    target = expected_path(vx, vy, wz, frames, fps)
    measured = pos[:, PELVIS, :2] - pos[0, PELVIS, :2]
    error = measured - target
    travel = np.array([vx, vy], dtype=float)
    heading_delta = float(np.degrees(yaw_wxyz(quat[:, PELVIS])[-1] - yaw_wxyz(quat[:, PELVIS])[0]))
    overlap_min, overlap_fraction = _nearest_fraction(left[core, :2], right[core, :2], .10)
    armhip = np.linalg.norm(pos[:, LEFT_ARM + RIGHT_ARM, None, :] - pos[:, None, HIP_LINKS, :], axis=-1)
    armtorso = np.linalg.norm(pos[:, LEFT_ARM + RIGHT_ARM, :] - pos[:, TORSO, None, :], axis=-1)
    pelvis_pitch = np.degrees(pitch_wxyz(quat[:, PELVIS]))
    torso_rel = quat_multiply(quat_conjugate(quat[:, PELVIS]), quat[:, TORSO])
    torso_pitch = np.degrees(pitch_wxyz(torso_rel))
    displacement = measured[-1]
    row = {
        **provenance, "motion_path": str(Path(path)), "frames": frames, "fps": fps,
        "endpoint_error_m": float(np.linalg.norm(error[-1])),
        "path_rmse_m": float(np.sqrt(np.mean(np.sum(error*error, axis=1)))),
        "mean_speed_mps": float(np.mean(np.linalg.norm(np.diff(measured, axis=0), axis=1))*fps),
        "direction_alignment_deg": _alignment_deg(displacement, travel),
        "heading_change_deg": heading_delta,
        "pelvis_height_mean_m": float(np.mean(pos[core, PELVIS, 2])),
        "pelvis_height_std_m": float(np.std(pos[core, PELVIS, 2])),
        "pelvis_pitch_mean_deg": float(np.mean(pelvis_pitch[core])),
        "pelvis_pitch_abs_mean_deg": float(np.mean(np.abs(pelvis_pitch[core]))),
        "torso_relative_pitch_mean_deg": float(np.mean(torso_pitch[core])),
        "waist_pitch_mean_deg": float(np.degrees(np.mean(joints[core, WAIST_PITCH]))),
        "wrist_midline_cross_fraction": float(np.mean(midline_cross[core])),
        "wrist_order_cross_fraction": float(np.mean(order_cross[core])),
        "wrist_trajectory_overlap_min_m": overlap_min,
        "wrist_trajectory_overlap_fraction_10cm": overlap_fraction,
        "dominant_swing_axis": dominant_axis,
        "dominant_swing_range_m": max(forward_amp, lateral_amp),
        "swing_alignment_to_command_deg": _alignment_deg(dominant, travel),
        "wrist_forward_amplitude_m": forward_amp,
        "wrist_lateral_amplitude_m": lateral_amp,
        "arm_hip_center_min_m": float(np.min(armhip[core])),
        "arm_torso_center_min_m": float(np.min(armtorso[core])),
        "proxy_warning": "center/link proxies; not mesh-collision evidence",
    }
    trajectories = {"left": left, "right": right, "travel": travel, "fps": fps}
    return row, trajectories


def load_manifest_rows(root: str | Path) -> list[tuple[Path, dict]]:
    """Resolve generation manifests and motion/config/prompt provenance."""
    root = Path(root)
    rows = []
    for manifest in sorted(root.rglob("manifest.csv")):
        with manifest.open(newline="") as stream:
            for item in csv.DictReader(stream):
                motion = manifest.parent / item["path"]
                if motion.is_file():
                    provenance = dict(item); provenance["config"] = manifest.parent.name
                    rows.append((motion, provenance))
    if not rows:
        rows = [(path, {"config": path.parent.parent.name}) for path in sorted(root.rglob("motion.npz"))]
    return rows


def plot_wrist_trajectory(trajectories: dict, row: dict, output: str | Path,
                          limit: float = .65, title: str | None = None) -> None:
    """Write a headless, stable-scale top-down pelvis-frame wrist plot."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    left, right = trajectories["left"], trajectories["right"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    ax.add_patch(Ellipse((0, 0), .28, .38, facecolor="#d9dde3", edgecolor="#535b66", alpha=.55))
    ax.axhline(0, color="#8b929b", lw=.8, ls="--"); ax.axvline(0, color="#8b929b", lw=.8, ls=":")
    ax.plot(left[:,0], left[:,1], color="#087e8b", lw=1.4, label="left wrist")
    ax.plot(right[:,0], right[:,1], color="#d1495b", lw=1.4, label="right wrist")
    for curve, color in ((left,"#087e8b"),(right,"#d1495b")):
        ax.scatter(curve[0,0],curve[0,1],marker="o",s=28,color=color)
        ax.scatter(curve[-1,0],curve[-1,1],marker="X",s=34,color=color)
        step=max(1,len(curve)//8); ax.scatter(curve[::step,0],curve[::step,1],s=8,color=color,alpha=.45)
    travel=np.asarray(trajectories["travel"],float)
    if np.linalg.norm(travel)>1e-9:
        arrow=.28*travel/np.linalg.norm(travel); ax.arrow(0,0,arrow[0],arrow[1],width=.008,color="#2f4858",length_includes_head=True)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect="equal",
           xlabel="pelvis-local forward (+X) [m]",ylabel="pelvis-local left/lateral (+Y) [m]")
    ax.set_title(title or Path(row["motion_path"]).parent.name, fontsize=10)
    note=(f"midline {row['wrist_midline_cross_fraction']:.1%} | overlap<10cm "
          f"{row['wrist_trajectory_overlap_fraction_10cm']:.1%}\n"
          f"dominant {row['dominant_swing_axis']} {row['dominant_swing_range_m']:.2f}m | "
          f"align {row['swing_alignment_to_command_deg']:.1f} deg")
    ax.text(.02,.02,note,transform=ax.transAxes,fontsize=7,va="bottom")
    ax.legend(loc="upper right",fontsize=7); ax.grid(alpha=.15)
    fig.savefig(output,dpi=180); plt.close(fig)


def plot_comparison(items: list[tuple[dict, dict]], output: str | Path, limit: float = .65) -> None:
    """Write a compact stable-scale wrist comparison panel."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    cols = min(3, len(items)); rows_n = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.0*cols, 3.8*rows_n), squeeze=False, constrained_layout=True)
    for ax, (row, trajectories) in zip(axes.flat, items):
        left, right = trajectories["left"], trajectories["right"]
        ax.add_patch(Ellipse((0,0),.28,.38,facecolor="#d9dde3",edgecolor="#535b66",alpha=.55))
        ax.axhline(0,color="#8b929b",lw=.7,ls="--"); ax.axvline(0,color="#8b929b",lw=.7,ls=":")
        ax.plot(left[:,0],left[:,1],color="#087e8b",lw=1.1,label="left")
        ax.plot(right[:,0],right[:,1],color="#d1495b",lw=1.1,label="right")
        travel=np.asarray(trajectories["travel"],float)
        if np.linalg.norm(travel)>1e-9:
            arrow=.25*travel/np.linalg.norm(travel); ax.arrow(0,0,arrow[0],arrow[1],width=.008,color="#2f4858",length_includes_head=True)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect="equal",title=str(row.get("config",Path(row["motion_path"]).parent.name)),xlabel="local forward +X [m]",ylabel="local left +Y [m]")
        ax.text(.02,.02,f"mid {row['wrist_midline_cross_fraction']:.1%} | {row['dominant_swing_axis']} {row['dominant_swing_range_m']:.2f}m",transform=ax.transAxes,fontsize=7)
    for ax in axes.flat[len(items):]: ax.set_visible(False)
    fig.savefig(output,dpi=180); plt.close(fig)


def evaluate_output_root(root: str | Path, output_dir: str | Path,
                         per_motion_plots: bool = False,
                         comparison_patterns: Iterable[str] = ()) -> list[dict]:
    """Evaluate all manifest motions and write deterministic CSV/JSON/plots."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows=[]; comparison=[]; patterns=tuple(comparison_patterns); plot_dir=output/"wrist_plots"
    if per_motion_plots: plot_dir.mkdir(exist_ok=True)
    for index,(motion, provenance) in enumerate(load_manifest_rows(root)):
        row, trajectories=evaluate_motion(motion, provenance); rows.append(row)
        if patterns and any(pattern in str(row.get("config", "")) or pattern in str(motion) for pattern in patterns):
            comparison.append((row, trajectories))
        if per_motion_plots:
            plot_wrist_trajectory(trajectories,row,plot_dir/f"{index:04d}_{motion.parent.name}.png")
    rows.sort(key=lambda x:(str(x.get("config","")),str(x.get("motion_type","")),str(x.get("motion_path",""))))
    (output/"evaluation.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n")
    if rows:
        fields=sorted({key for row in rows for key in row})
        with (output/"evaluation.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    if comparison:
        plot_comparison(comparison, output/"wrist_comparison.png")
    return rows
