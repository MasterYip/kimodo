#!/usr/bin/env python3
"""Validate native prompt batches and emit review-oriented center proxies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED = {"forward": np.array((1.0, 0.0)), "right": np.array((0.0, -1.0)),
            "left": np.array((0.0, 1.0))}
LEFT_HAND, RIGHT_HAND = 25, 33
HIP_IDS = (0, 1, 2, 3, 8, 9, 10)
LEFT_LEG, RIGHT_LEG = (1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14)
LEFT_FOOT, RIGHT_FOOT = (6, 7), (13, 14)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pitch_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))


def evaluate(dataset: Path) -> list[dict]:
    with (dataset / "generation_manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    with (dataset / "best_practices_preflight.json").open() as stream:
        preflight = json.load(stream)
    launched = [row for row in preflight["rows"] if row["status"] == "launch"]
    if len(manifest) != len(launched):
        raise RuntimeError(f"manifest/preflight count mismatch: {len(manifest)} != {len(launched)}")
    rows = []
    for item in manifest:
        native_path = dataset / item["native_path"]
        motion_path = dataset / item["motion_path"]
        with np.load(native_path, allow_pickle=False) as native:
            posed = np.asarray(native["posed_joints"])
            rotations = np.asarray(native["local_rot_mats"])
        with np.load(motion_path, allow_pickle=False) as motion:
            qpos = np.asarray(motion["qpos"])
            fps = float(motion["fps"])
        if posed.ndim == 4: posed = posed[0]
        if rotations.ndim == 5: rotations = rotations[0]
        schema_ok = (posed.shape == (240, 34, 3) and rotations.shape == (240, 34, 3, 3)
                     and qpos.shape == (240, 36) and qpos.dtype == np.float32
                     and abs(fps - 30.0) < 1e-6)
        finite = bool(np.isfinite(posed).all() and np.isfinite(rotations).all()
                      and np.isfinite(qpos).all())
        quat_error = float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)))
        ortho_error = float(np.max(np.abs(
            rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3))))
        if not (schema_ok and finite and quat_error < 1e-4 and ortho_error < 1e-4):
            raise RuntimeError(f"structural failure: {motion_path}")

        core = slice(24, 216)
        wrists = posed[:, (LEFT_HAND, RIGHT_HAND)]
        hips = posed[:, HIP_IDS]
        wrist_hip = np.linalg.norm(
            wrists[:, :, None] - hips[:, None], axis=-1).min(axis=2)
        separation = np.linalg.norm(wrists[:, 0] - wrists[:, 1], axis=1)
        lo = hips.min(axis=1) - np.array((.10, .10, .04))
        hi = hips.max(axis=1) + np.array((.10, .10, .04))
        in_zone = ((wrists >= lo[:, None]) & (wrists <= hi[:, None])).all(axis=2)
        leg_distance = np.linalg.norm(
            posed[:, LEFT_LEG, None] - posed[:, None, RIGHT_LEG], axis=-1)
        foot_distance = np.linalg.norm(
            posed[:, LEFT_FOOT, None] - posed[:, None, RIGHT_FOOT], axis=-1)

        displacement = qpos[-1, :2] - qpos[0, :2]
        distance = float(np.linalg.norm(displacement))
        expected = EXPECTED.get(item["direction"])
        direction_cos = (float(np.dot(displacement, expected) / max(distance, 1e-9))
                         if expected is not None else float("nan"))
        pitch = np.degrees(pitch_wxyz(qpos[:, 3:7]))
        hand_rel = wrists - posed[:, 0, None]
        first, last = slice(0, 120), slice(180, 240)
        forward_amp_first = float(np.mean(np.ptp(hand_rel[first, :, 2], axis=0)))
        forward_amp_last = float(np.mean(np.ptp(hand_rel[last, :, 2], axis=0)))
        tail_rms = float(np.sqrt(np.mean(np.diff(qpos[-60:], axis=0) ** 2)) /
                         max(np.sqrt(np.mean(np.diff(qpos[-120:-60], axis=0) ** 2)), 1e-9))
        rows.append({
            "index": int(item["index"]), "prompt_family": item["prompt_family"],
            "direction": item["direction"], "prompt": item["prompt"],
            "seed": int(item["seed"]), "frames": len(qpos), "fps": fps,
            "schema_ok": schema_ok, "finite": finite,
            "quat_norm_max_error": quat_error, "rotation_ortho_max_error": ortho_error,
            "root_dx_m": float(displacement[0]), "root_dy_m": float(displacement[1]),
            "root_distance_m": distance, "direction_cosine": direction_cos,
            "achieved_angle_deg": float(math.degrees(math.atan2(displacement[1], displacement[0]))),
            "pelvis_height_mean_m": float(qpos[core, 2].mean()),
            "pelvis_pitch_mean_deg": float(pitch[core].mean()),
            "pelvis_pitch_first_half_deg": float(pitch[first].mean()),
            "pelvis_pitch_last_quarter_deg": float(pitch[last].mean()),
            "hand_forward_amplitude_first_half_m": forward_amp_first,
            "hand_forward_amplitude_last_quarter_m": forward_amp_last,
            "hand_hip_center_min_m": float(wrist_hip[core].min()),
            "hand_hip_center_p05_m": float(np.quantile(wrist_hip[core], .05)),
            "hip_crotch_zone_either_fraction": float(in_zone[core].any(axis=1).mean()),
            "wrist_separation_p05_m": float(np.quantile(separation[core], .05)),
            "inter_leg_center_min_m": float(leg_distance[core].min()),
            "inter_foot_center_min_m": float(foot_distance[core].min()),
            "tail_motion_rms_ratio": tail_rms,
            "best_practices_pass": item["best_practices_pass"] == "True",
            "proxy_warning": "joint-center diagnostics; human mesh review authoritative",
            "motion_sha256": sha256(motion_path), "native_sha256": sha256(native_path),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    rows = evaluate(args.dataset)
    with (args.dataset / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.dataset / "evaluation.json").write_text(json.dumps({
        "candidate_review_policy": "Human review required; proxies never accept motions.",
        "rows": rows}, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"evaluated": len(rows), "all_structural_checks_pass": True}))


if __name__ == "__main__":
    main()
