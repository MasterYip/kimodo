#!/usr/bin/env python3
"""Audit Kimodo RLTracker clips around the supported model horizon.

This tool intentionally reads the exported ``motion.npz`` files used by the
downstream viewer.  It therefore distinguishes generator defects from media
rendering defects and produces tail-only evidence for human inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _jitter(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 5:
        return 0.0
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(signal, kernel, mode="same")
    return float(np.sqrt(np.mean((signal - smooth) ** 2)))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-8))


def _segment_metrics(data: dict[str, np.ndarray], start: int, end: int, fps: float) -> dict[str, float]:
    pos = np.asarray(data["body_pos_w"], dtype=np.float64)[start:end]
    joints = np.asarray(data["joint_pos"], dtype=np.float64)[start:end]
    quat = np.asarray(data["body_quat_w"], dtype=np.float64)[start:end]
    if len(pos) < 3:
        return {key: math.nan for key in (
            "root_speed_mean", "root_speed_jitter", "body_step_rms",
            "joint_velocity_rms", "joint_velocity_max", "quat_step_rms",
        )}
    root_speed = np.linalg.norm(np.diff(pos[:, 0], axis=0), axis=-1) * fps
    body_step = np.diff(pos, axis=0) * fps
    joint_velocity = np.diff(joints, axis=0) * fps
    dots = np.abs(np.sum(quat[1:] * quat[:-1], axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    quat_step = 2.0 * np.arccos(dots) * fps
    return {
        "root_speed_mean": float(np.mean(root_speed)),
        "root_speed_jitter": _jitter(root_speed),
        "body_step_rms": float(np.sqrt(np.mean(body_step ** 2))),
        "joint_velocity_rms": float(np.sqrt(np.mean(joint_velocity ** 2))),
        "joint_velocity_max": float(np.max(np.abs(joint_velocity))),
        "quat_step_rms": float(np.sqrt(np.mean(quat_step ** 2))),
    }


def _read_manifest(dataset: Path) -> list[dict[str, str]]:
    with (dataset / "manifest.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_tail_media(rows: list[dict], out_dir: Path, seconds: float) -> list[dict]:
    import imageio.v2 as imageio
    from PIL import Image
    from locomotion_framework.quality_validator import render_frame

    media_dir = out_dir / "tail_media"
    media_dir.mkdir(exist_ok=True)
    selected: dict[str, dict] = {}
    for row in rows:
        selected.setdefault(row["motion_type"], row)

    index = []
    for motion_type, row in sorted(selected.items()):
        source = Path(row["source_path"])
        with np.load(source, allow_pickle=False) as archive:
            pos = np.asarray(archive["body_pos_w"])
            fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
        start = max(0, len(pos) - int(round(seconds * fps)))
        frame_ids = np.linspace(start, len(pos) - 1, 8).round().astype(int)
        tiles = [render_frame(pos[i], f"{motion_type} tail frame={i}", size=(640, 280)) for i in frame_ids]
        sheet = Image.new("RGB", (1280, 1120), "white")
        for tile_idx, tile in enumerate(tiles):
            sheet.paste(tile, ((tile_idx % 2) * 640, (tile_idx // 2) * 280))
        sheet_path = media_dir / f"{motion_type}__final_{seconds:g}s_contact.png"
        sheet.save(sheet_path)
        video_path = media_dir / f"{motion_type}__final_{seconds:g}s.mp4"
        with imageio.get_writer(
            video_path,
            fps=max(1, int(round(fps / 2))),
            codec="libx264",
            quality=7,
            macro_block_size=16,
        ) as writer:
            for frame_idx in range(start, len(pos), 2):
                writer.append_data(np.asarray(render_frame(pos[frame_idx], f"{motion_type} frame={frame_idx}")))
        index.append({
            "motion_type": motion_type,
            "source_path": str(source),
            "tail_start_frame": start,
            "tail_end_frame": len(pos) - 1,
            "contact_sheet": str(sheet_path),
            "video": str(video_path),
        })
    _write_csv(out_dir / "tail_media_index.csv", index)
    return index


def _plot(rows: list[dict], out_dir: Path, native_fps: float, max_seconds: float) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = sorted({row["motion_type"] for row in rows})
    metrics = (
        ("root_speed_ratio_post_pre", "Root speed post/pre"),
        ("root_jitter_ratio_post_pre", "Root jitter post/pre"),
        ("joint_velocity_ratio_post_pre", "Joint velocity RMS post/pre"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (key, title) in zip(axes, metrics):
        values = [[float(row[key]) for row in rows if row["motion_type"] == typ] for typ in types]
        axis.boxplot(values, labels=types, showmeans=True)
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=55, labelsize=7)
        axis.set_yscale("log")
    fig.suptitle(
        f"Tail failure after official model horizon ({max_seconds:g}s × {native_fps:g} FPS = "
        f"{int(round(max_seconds * native_fps))} frames)"
    )
    fig.tight_layout()
    ratio_path = out_dir / "tail_horizon_ratios_by_motion_type.png"
    fig.savefig(ratio_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for row in rows:
        ax.scatter(
            float(row["frames"]),
            float(row["root_jitter_ratio_post_pre"]),
            label=row["motion_type"],
            alpha=0.75,
        )
    horizon = max_seconds * native_fps
    ax.axvline(horizon, color="#d62728", linewidth=2, label=f"supported horizon: {horizon:g} frames")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set(xlabel="exported frame count", ylabel="post/pre root jitter ratio", title="All rejected pilots cross the model horizon")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    fig.tight_layout()
    boundary_path = out_dir / "tail_horizon_boundary.png"
    fig.savefig(boundary_path, dpi=180)
    plt.close(fig)
    return [str(ratio_path), str(boundary_path)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-fps", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--visual-verdict", default="pending_human_review")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(round(args.native_fps * args.max_seconds))
    rows: list[dict] = []
    for dataset_arg in args.datasets:
        dataset = Path(dataset_arg).resolve()
        for manifest_row in _read_manifest(dataset):
            source = dataset / manifest_row["path"]
            with np.load(source, allow_pickle=False) as archive:
                data = {key: np.asarray(archive[key]) for key in archive.files}
            fps = float(data["fps"].reshape(-1)[0])
            frames = int(data["body_pos_w"].shape[0])
            tail_frames = int(round(args.tail_seconds * fps))
            if frames > horizon:
                pre_start = max(0, horizon - tail_frames)
                pre_end = horizon
                post_start = horizon
                post_end = min(frames, horizon + tail_frames)
                comparison_basis = "before_vs_after_supported_horizon"
            else:
                post_start = max(0, frames - tail_frames)
                post_end = frames
                pre_start = max(0, post_start - tail_frames)
                pre_end = post_start
                comparison_basis = "penultimate_vs_final_tail_window"
            pre = _segment_metrics(data, pre_start, pre_end, fps)
            post = _segment_metrics(data, post_start, post_end, fps)
            row = {
                "dataset_id": dataset.name,
                "motion_type": manifest_row["motion_type"],
                "motion_path": manifest_row["path"],
                "source_path": str(source),
                "cmd_vx": manifest_row.get("vx", ""),
                "cmd_vy": manifest_row.get("vy", ""),
                "cmd_wz": manifest_row.get("wz", ""),
                "reported_fps": fps,
                "native_fps": args.native_fps,
                "frames": frames,
                "reported_duration_s": frames / fps,
                "native_duration_s": frames / args.native_fps,
                "supported_horizon_frame": horizon,
                "pre_frames": f"{pre_start}:{pre_end}",
                "post_frames": f"{post_start}:{post_end}",
                "comparison_basis": comparison_basis,
                "crosses_supported_horizon": frames > horizon,
                "visual_verdict": args.visual_verdict,
            }
            row.update({f"pre_{key}": value for key, value in pre.items()})
            row.update({f"post_{key}": value for key, value in post.items()})
            row.update({
                "root_speed_ratio_post_pre": _ratio(post["root_speed_mean"], pre["root_speed_mean"]),
                "root_jitter_ratio_post_pre": _ratio(post["root_speed_jitter"], pre["root_speed_jitter"]),
                "joint_velocity_ratio_post_pre": _ratio(post["joint_velocity_rms"], pre["joint_velocity_rms"]),
                "body_step_ratio_post_pre": _ratio(post["body_step_rms"], pre["body_step_rms"]),
                "quat_step_ratio_post_pre": _ratio(post["quat_step_rms"], pre["quat_step_rms"]),
            })
            rows.append(row)

    _write_csv(out_dir / "tail_horizon_per_motion.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["motion_type"]].append(row)
    per_type = {
        motion_type: {
            "count": len(group),
            "root_speed_ratio_post_pre_mean": float(np.mean([float(r["root_speed_ratio_post_pre"]) for r in group])),
            "root_jitter_ratio_post_pre_mean": float(np.mean([float(r["root_jitter_ratio_post_pre"]) for r in group])),
            "joint_velocity_ratio_post_pre_mean": float(np.mean([float(r["joint_velocity_ratio_post_pre"]) for r in group])),
        }
        for motion_type, group in sorted(grouped.items())
    }
    figures = _plot(rows, out_dir, args.native_fps, args.max_seconds)
    media = _render_tail_media(rows, out_dir, args.tail_seconds) if args.render else []
    summary = {
        "status": "rejected" if args.visual_verdict.startswith("rejected") else "pending_review",
        "visual_verdict": args.visual_verdict,
        "checkpoint_native_fps": args.native_fps,
        "official_max_duration_s": args.max_seconds,
        "supported_horizon_frames": horizon,
        "motion_count": len(rows),
        "motion_types": per_type,
        "overall": {
            "root_speed_ratio_post_pre_mean": float(np.mean([float(r["root_speed_ratio_post_pre"]) for r in rows])),
            "root_jitter_ratio_post_pre_mean": float(np.mean([float(r["root_jitter_ratio_post_pre"]) for r in rows])),
            "joint_velocity_ratio_post_pre_mean": float(np.mean([float(r["joint_velocity_ratio_post_pre"]) for r in rows])),
        },
        "figures": figures,
        "tail_media_count": len(media),
    }
    (out_dir / "tail_horizon_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
