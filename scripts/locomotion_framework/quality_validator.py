#!/usr/bin/env python3
"""Deterministic quality validation and visual evidence for RLTracker motions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REQUIRED = {
    "fps": (1,),
    "joint_pos": (None, 29),
    "joint_vel": (None, 29),
    "body_pos_w": (None, 30, 3),
    "body_quat_w": (None, 30, 4),
    "body_lin_vel_w": (None, 30, 3),
    "body_ang_vel_w": (None, 30, 3),
}
PELVIS, TORSO = 0, 9
LEFT_FOOT, RIGHT_FOOT = 18, 19
HIP_PITCH = (0, 1)
SHOULDER_PITCH = (11, 12)
SHOULDER_ROLL = (15, 16)
ELBOW = (21, 22)
SKELETON_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10), (10, 14), (14, 18),
    (0, 2), (2, 5), (5, 8), (8, 11), (11, 15), (15, 19),
    (0, 3), (3, 6), (6, 9),
    (9, 12), (12, 16), (16, 20), (20, 22), (22, 24), (24, 26), (26, 28),
    (9, 13), (13, 17), (17, 21), (21, 23), (23, 25), (25, 27), (27, 29),
]
LOCOMOTION = {
    "forward_walk_straight", "forward_walk_turn", "backward_walk",
    "jog", "run", "crouched_walk", "side_step",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def periodicity(signal: np.ndarray, fps: float) -> float:
    x = np.asarray(signal, dtype=np.float64)
    x -= np.mean(x)
    denom = float(np.dot(x, x))
    if len(x) < 8 or denom < 1e-10:
        return 0.0
    lo = max(1, int(round(0.25 * fps)))
    hi = min(len(x) - 2, int(round(1.5 * fps)))
    if hi <= lo:
        return 0.0
    return float(max(np.dot(x[:-lag], x[lag:]) / denom for lag in range(lo, hi + 1)))


def quat_roll_pitch(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = (quat[..., i] for i in range(4))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    return roll, np.arcsin(sinp)


def shape_matches(shape: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    return len(shape) == len(expected) and all(e is None or e == a for a, e in zip(shape, expected))


def read_manifest(dataset: Path) -> list[dict]:
    manifest = dataset / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    with manifest.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_one(dataset: Path, row: dict, thresholds: dict) -> dict:
    relpath = row.get("path", "")
    npz_path = dataset / relpath
    motion_type = row.get("motion_type", "unknown")
    result = {
        "dataset_id": dataset.name,
        "motion_id": f"{dataset.name}:{Path(relpath).parent.name}",
        "motion_type": motion_type,
        "prompt": row.get("prompt", ""),
        "seed": dataset.name.rsplit("s", 1)[-1].replace("_raw", ""),
        "source_path": str(npz_path),
        "relative_path": relpath,
        "cmd_vx": finite_float(row.get("vx")),
        "cmd_vy": finite_float(row.get("vy")),
        "cmd_wz": finite_float(row.get("wz")),
        "requested_torso_height": finite_float(row.get("torso_height"), float("nan")),
    }
    reject, quarantine = [], []
    if not npz_path.is_file():
        result.update(quality_tier="rejected", reason_codes="missing_file")
        return result
    result["sha256"] = sha256(npz_path)
    try:
        archive = np.load(npz_path, allow_pickle=False)
    except Exception as exc:
        result.update(quality_tier="rejected", reason_codes=f"load_error:{type(exc).__name__}")
        return result

    missing = sorted(set(REQUIRED) - set(archive.files))
    if missing:
        reject.append("missing_keys:" + "+".join(missing))
    arrays = {}
    for key, expected in REQUIRED.items():
        if key not in archive:
            continue
        value = np.asarray(archive[key])
        arrays[key] = value
        if not shape_matches(value.shape, expected):
            reject.append(f"shape_{key}")
        if value.dtype != np.float32:
            reject.append(f"dtype_{key}")
        if not np.isfinite(value).all():
            reject.append(f"nonfinite_{key}")
    if reject or "fps" not in arrays:
        result.update(quality_tier="rejected", reason_codes=";".join(sorted(set(reject))))
        return result

    fps = float(arrays["fps"].reshape(-1)[0])
    result["fps"] = fps
    if abs(fps - float(thresholds["expected_fps"])) > 1e-6:
        reject.append("fps_mismatch")
    t_values = [value.shape[0] for key, value in arrays.items() if key != "fps" and value.ndim]
    if not t_values or len(set(t_values)) != 1 or t_values[0] < 10:
        reject.append("frame_count_mismatch")
    if reject:
        result.update(quality_tier="rejected", reason_codes=";".join(sorted(set(reject))))
        return result

    joint_pos = arrays["joint_pos"].astype(np.float64)
    joint_vel = arrays["joint_vel"].astype(np.float64)
    pos = arrays["body_pos_w"].astype(np.float64)
    quat = arrays["body_quat_w"].astype(np.float64)
    lin_vel = arrays["body_lin_vel_w"].astype(np.float64)
    ang_vel = arrays["body_ang_vel_w"].astype(np.float64)
    frames = pos.shape[0]
    core = slice(max(1, frames // 10), max(2, frames - frames // 10))
    duration = frames / fps
    quat_err = float(np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1.0)))
    torso_roll, torso_pitch = quat_roll_pitch(quat[:, TORSO])
    measured = np.mean(lin_vel[core, PELVIS, :], axis=0)
    measured_wz = float(np.mean(ang_vel[core, PELVIS, 2]))

    foot_z = pos[:, [LEFT_FOOT, RIGHT_FOOT], 2]
    foot_xy_speed = np.linalg.norm(lin_vel[:, [LEFT_FOOT, RIGHT_FOOT], :2], axis=-1)
    contact_level = np.quantile(foot_z, 0.20, axis=0) + 0.025
    contact_mask = foot_z <= contact_level[None, :]
    contact_speeds = foot_xy_speed[contact_mask]
    foot_slip = float(np.mean(contact_speeds)) if len(contact_speeds) else float("nan")
    gait_signal = foot_z[:, 0] - foot_z[:, 1]

    shoulder_pitch = joint_pos[core, list(SHOULDER_PITCH)]
    shoulder_roll = joint_pos[core, list(SHOULDER_ROLL)]
    arm_excursion = float(np.mean(np.ptp(shoulder_pitch, axis=0)))
    arm_abduction = float(np.max(np.abs(shoulder_roll)))
    phase_strength = float(np.mean([
        corr_abs(joint_pos[core, SHOULDER_PITCH[0]], joint_pos[core, HIP_PITCH[1]]),
        corr_abs(joint_pos[core, SHOULDER_PITCH[1]], joint_pos[core, HIP_PITCH[0]]),
    ]))
    arm_symmetry = corr_abs(shoulder_pitch[:, 0], shoulder_pitch[:, 1])
    arm_signal = np.mean(joint_pos[:, list(SHOULDER_PITCH + ELBOW)], axis=1)
    arm_jerk = np.diff(arm_signal, n=3) * fps ** 3 if frames >= 4 else np.zeros(1)

    result.update({
        "frames": frames,
        "duration_s": duration,
        "quat_norm_max_error": quat_err,
        "pelvis_z_min": float(np.min(pos[:, PELVIS, 2])),
        "pelvis_z_mean": float(np.mean(pos[core, PELVIS, 2])),
        "torso_z_min": float(np.min(pos[:, TORSO, 2])),
        "torso_roll_abs_p95": float(np.quantile(np.abs(torso_roll), 0.95)),
        "torso_pitch_abs_p95": float(np.quantile(np.abs(torso_pitch), 0.95)),
        "body_z_min": float(np.min(pos[..., 2])),
        "joint_abs_max": float(np.max(np.abs(joint_pos))),
        "joint_velocity_abs_max": float(np.max(np.abs(joint_vel))),
        "joint_accel_rms": float(np.sqrt(np.mean(np.diff(joint_vel, axis=0) ** 2)) * fps),
        "measured_vx": float(measured[0]),
        "measured_vy": float(measured[1]),
        "measured_wz": measured_wz,
        "error_vx": float(abs(measured[0] - result["cmd_vx"])),
        "error_vy": float(abs(measured[1] - result["cmd_vy"])),
        "error_wz": float(abs(measured_wz - result["cmd_wz"])),
        "foot_slip_mean": foot_slip,
        "step_width_mean": float(np.mean(np.abs(pos[core, LEFT_FOOT, 1] - pos[core, RIGHT_FOOT, 1]))),
        "gait_periodicity": periodicity(gait_signal[core], fps),
        "arm_excursion": arm_excursion,
        "arm_abduction_abs_max": arm_abduction,
        "arm_phase_strength": phase_strength,
        "arm_lr_correlation_abs": arm_symmetry,
        "arm_periodicity": periodicity(shoulder_pitch[:, 0] - shoulder_pitch[:, 1], fps),
        "arm_jerk_rms": float(np.sqrt(np.mean(arm_jerk ** 2))),
    })

    hard, soft = thresholds["hard"], thresholds["soft"]
    hard_checks = {
        "quaternion_norm": quat_err > hard["quaternion_norm_error"],
        "pelvis_too_low": result["pelvis_z_min"] < hard["pelvis_height_min"],
        "torso_too_low": result["torso_z_min"] < hard["torso_height_min"],
        "ground_penetration": result["body_z_min"] < hard["ground_penetration_min"],
        "joint_range_extreme": result["joint_abs_max"] > hard["joint_abs_max"],
        "joint_velocity_extreme": result["joint_velocity_abs_max"] > hard["joint_velocity_abs_max"],
        "arm_abduction_extreme": arm_abduction > hard["arm_abduction_abs_max"],
    }
    reject.extend(name for name, failed in hard_checks.items() if failed)

    torso_soft = soft["crouched_torso_height_min"] if motion_type == "crouched_walk" else soft["torso_height_min"]
    soft_checks = {
        "pelvis_low_review": result["pelvis_z_min"] < soft["pelvis_height_min"],
        "torso_low_review": result["torso_z_min"] < torso_soft,
        "ground_contact_review": result["body_z_min"] < soft["ground_penetration_min"],
        "joint_velocity_review": result["joint_velocity_abs_max"] > soft["joint_velocity_abs_max"],
        "foot_slip_review": math.isfinite(foot_slip) and foot_slip > soft["foot_slip_mean_max"],
        "arm_abduction_review": arm_abduction > soft["arm_abduction_abs_max"],
    }
    if motion_type in LOCOMOTION:
        soft_checks.update({
            "frozen_arm_review": arm_excursion < soft["arm_excursion_min"],
            "arm_phase_review": phase_strength < soft["arm_phase_strength_min"],
            "gait_periodicity_review": result["gait_periodicity"] < soft["gait_periodicity_min"],
        })
    quarantine.extend(name for name, failed in soft_checks.items() if failed)

    tolerances = thresholds["command_tolerance"].get(
        motion_type, thresholds["command_tolerance"]["default"])
    for axis in ("vx", "vy", "wz"):
        if result[f"error_{axis}"] > tolerances[axis]:
            quarantine.append(f"command_{axis}_review")
    if motion_type in thresholds.get("research_quarantine_types", []):
        quarantine.append("research_tier_manual_review")
    if motion_type != "crouched_walk" and any(word in result["prompt"].lower() for word in ("crouch", "squat")):
        reject.append("nominal_prompt_contamination")

    tier = "rejected" if reject else "quarantine" if quarantine else "accepted"
    reasons = reject if reject else quarantine
    result.update(quality_tier=tier, reason_codes=";".join(sorted(set(reasons))) or "none")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["motion_id"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quality_score(row: dict) -> float:
    return (
        finite_float(row.get("error_vx")) + finite_float(row.get("error_vy"))
        + finite_float(row.get("error_wz")) + finite_float(row.get("foot_slip_mean"))
        + 0.2 * finite_float(row.get("arm_abduction_abs_max"))
        + max(0.0, 0.15 - finite_float(row.get("arm_phase_strength")))
    )


def select_visuals(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.RandomState(seed)
    selected = []
    grouped = defaultdict(list)
    for row in rows:
        if "frames" in row:
            grouped[row["motion_type"]].append(row)
    for motion_type, group in sorted(grouped.items()):
        ordered = sorted(group, key=quality_score)
        roles = {
            "random": group[int(rng.randint(len(group)))],
            "median": ordered[(len(ordered) - 1) // 2],
            "worst": ordered[-1],
        }
        for role, row in roles.items():
            selected.append({
                "motion_id": row["motion_id"], "motion_type": motion_type,
                "role": role, "quality_tier": row["quality_tier"],
                "source_path": row["source_path"], "reason_codes": row["reason_codes"],
            })
    return selected


def plot_summary(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    types = sorted({row["motion_type"] for row in rows})
    tiers = ["accepted", "quarantine", "rejected"]
    colors = {"accepted": "#2ca02c", "quarantine": "#ffbf00", "rejected": "#d62728"}
    counts = {tier: [sum(r["motion_type"] == typ and r["quality_tier"] == tier for r in rows) for typ in types]
              for tier in tiers}
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bottom = np.zeros(len(types))
    for tier in tiers:
        ax.bar(types, counts[tier], bottom=bottom, label=tier, color=colors[tier])
        bottom += counts[tier]
    ax.set_ylabel("motion count")
    ax.set_title("DATA-KIMODO-001 pilot: generated motion types and quality composition")
    ax.tick_params(axis="x", rotation=28)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "motion_type_composition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, key in zip(axes, ("cmd_vx", "cmd_vy", "cmd_wz")):
        for idx, typ in enumerate(types):
            values = [finite_float(r.get(key)) for r in rows if r["motion_type"] == typ]
            axis.scatter([idx] * len(values), values, s=32, alpha=0.8)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_title(key.replace("cmd_", "commanded "))
        axis.set_xticks(range(len(types)), types, rotation=55, ha="right", fontsize=7)
    fig.suptitle("Requested conservative command coverage (lateral tier remains separate)")
    fig.tight_layout()
    path = out_dir / "velocity_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    valid = [r for r in rows if "frames" in r]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for row in valid:
        color = colors[row["quality_tier"]]
        axes[0].scatter(row["arm_excursion"], row["arm_abduction_abs_max"], c=color, s=28)
        axes[1].scatter(row["arm_phase_strength"], row["arm_periodicity"], c=color, s=28)
        axes[2].scatter(row["foot_slip_mean"], row["gait_periodicity"], c=color, s=28)
    axes[0].set(xlabel="shoulder pitch excursion (rad)", ylabel="max shoulder roll |rad|", title="Arm excursion / abduction")
    axes[1].set(xlabel="contralateral phase strength", ylabel="arm periodicity", title="Arm timing")
    axes[2].set(xlabel="contact-foot slip proxy (m/s)", ylabel="gait periodicity", title="Gait quality proxies")
    fig.suptitle("Automatic arm and gait diagnostics (pilot thresholds are provisional)")
    fig.tight_layout()
    path = out_dir / "arm_gait_quality.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    return figures


def render_frame(pos: np.ndarray, label: str, size=(720, 320)):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    center = pos[PELVIS].copy()
    centered = pos.copy()
    centered[:, :2] -= center[:2]
    scale, floor_y = 145.0, size[1] - 30
    panels = [(size[0] // 4, 0, "side"), (3 * size[0] // 4, 1, "front")]
    for cx, horizontal_axis, title in panels:
        projected = np.column_stack([
            cx + centered[:, horizontal_axis] * scale,
            floor_y - centered[:, 2] * scale,
        ])
        for a, b in SKELETON_EDGES:
            draw.line((*projected[a], *projected[b]), fill=(40, 70, 120), width=3)
        for point in projected:
            x, y = point
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(200, 50, 45))
        draw.line((cx - 150, floor_y, cx + 150, floor_y), fill=(100, 100, 100), width=1)
        draw.text((cx - 20, 8), title, fill=(0, 0, 0))
    draw.text((8, size[1] - 18), label[:100], fill=(0, 0, 0))
    return image


def render_visuals(selection: list[dict], out_dir: Path, video_stride: int = 2) -> list[dict]:
    import imageio.v2 as imageio
    from PIL import Image

    media_dir = out_dir / "media"
    media_dir.mkdir(exist_ok=True)
    by_motion = defaultdict(list)
    for item in selection:
        by_motion[item["motion_id"]].append(item["role"])
    index = []
    selected_map = {item["motion_id"]: item for item in selection}
    for motion_id, roles in sorted(by_motion.items()):
        item = selected_map[motion_id]
        archive = np.load(item["source_path"], allow_pickle=False)
        pos = np.asarray(archive["body_pos_w"])
        fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        safe_id = motion_id.replace(":", "__").replace("/", "_")
        frame_ids = np.linspace(0, len(pos) - 1, 8).round().astype(int)
        tiles = [render_frame(pos[i], f"{motion_id} frame={i}", size=(640, 280)) for i in frame_ids]
        sheet = Image.new("RGB", (1280, 1120), "white")
        for tile_idx, tile in enumerate(tiles):
            sheet.paste(tile, ((tile_idx % 2) * 640, (tile_idx // 2) * 280))
        sheet_path = media_dir / f"{safe_id}_contact.png"
        sheet.save(sheet_path)
        video_path = media_dir / f"{safe_id}.mp4"
        with imageio.get_writer(video_path, fps=max(1, int(round(fps / video_stride))), codec="libx264",
                                quality=7, macro_block_size=16) as writer:
            for frame_idx in range(0, len(pos), video_stride):
                writer.append_data(np.asarray(render_frame(pos[frame_idx], motion_id)))
        index.append({
            **item,
            "roles": ";".join(sorted(roles)),
            "contact_sheet": str(sheet_path),
            "video": str(video_path),
            "frame_indices": ";".join(map(str, frame_ids)),
        })
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--selection-seed", type=int, default=20260803)
    args = parser.parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(args.thresholds) as stream:
        thresholds = yaml.safe_load(stream)
    rows = []
    for dataset_arg in args.datasets:
        dataset = Path(dataset_arg).resolve()
        rows.extend(validate_one(dataset, row, thresholds) for row in read_manifest(dataset))
    rows.sort(key=lambda row: row["motion_id"])
    write_csv(out_dir / "per_motion.csv", rows)
    (out_dir / "per_motion.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    for tier in ("accepted", "quarantine", "rejected"):
        write_csv(out_dir / f"{tier}_manifest.csv", [row for row in rows if row["quality_tier"] == tier])
    selection = select_visuals(rows, args.selection_seed)
    write_csv(out_dir / "visual_selection.csv", selection)
    figures = plot_summary(rows, out_dir)
    media_index = render_visuals(selection, out_dir) if args.render else []
    write_csv(out_dir / "visualization_index.csv", media_index if args.render else selection)
    summary = {
        "validator_version": thresholds["version"],
        "datasets": [str(Path(item).resolve()) for item in args.datasets],
        "thresholds_sha256": sha256(Path(args.thresholds)),
        "total": len(rows),
        "composition": dict(Counter(row["quality_tier"] for row in rows)),
        "by_type": {
            typ: dict(Counter(row["quality_tier"] for row in rows if row["motion_type"] == typ))
            for typ in sorted({row["motion_type"] for row in rows})
        },
        "figures": [str(path) for path in figures],
        "rendered_media": len(media_index),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
