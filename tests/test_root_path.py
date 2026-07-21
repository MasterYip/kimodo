#!/usr/bin/env python3
"""Test: compare dense (05_root_path-style) vs sparse+projection root constraints.

Generates the same velocity command using two approaches and compares
velocity stability / frame-to-frame jitter.

Approach A: **Dense** — constrain every frame (stride=1), no post-processing.
            This matches the Kimodo demo's 05_root_path example.
Approach B: **Sparse+Projection** — constrain every 3rd frame (stride=3)
            then post-process with root projection (rigid shift).

Usage (on 4090-3):
    cd /data/masteryip/kimodo/kimodo/scripts
    source ./env.sh
    PYTHONPATH=. python3 locomotion_framework/test_root_path.py

Output:
    /data/masteryip/kimodo/kimodo/tests/output/normal_loco_test/
      dense_v05_fwd/    → stride=1, no projection (like 05_root_path)
      sparse_v06_fwd/   → stride=3 + root projection (our approach)
"""

import numpy as np
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on path (same pattern as orchestrator.py)
# File is at:  kimodo/scripts/locomotion_framework/test_root_path.py
# Repo root:   kimodo/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from locomotion_framework.constraints import (
    build_root2d_constraint,
    build_constraints_json,
)
from locomotion_framework.sampler import SampledMotion


# ── Test cases ──────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "walk_fwd",
        "prompt": "a person walks forward at a steady pace",
        "vel": {"vx": 0.5, "vy": 0.0, "wz": 0.0},
        "duration": 6.0,
        "torso_height": 0.7,
    },
    {
        "name": "run_fwd",
        "prompt": "a person runs forward steadily",
        "vel": {"vx": 1.5, "vy": 0.0, "wz": 0.0},
        "duration": 6.0,
        "torso_height": 0.7,
    },
    {
        "name": "walk_curve",
        "prompt": "a person walks in a smooth counterclockwise arc",
        "vel": {"vx": 0.5, "vy": 0.1, "wz": 0.3},
        "duration": 6.0,
        "torso_height": 0.7,
    },
    {
        "name": "walk_lateral",
        "prompt": "a person walks sideways to the right",
        "vel": {"vx": 0.0, "vy": 0.4, "wz": 0.0},
        "duration": 6.0,
        "torso_height": 0.7,
    },
]

FPS = 30
DIFFUSION_STEPS = 100
SEED = 42
DEVICE = "cuda:0"
MODEL_NAME = "kimodo-g1-rp"
OUTPUT_BASE = "/data/masteryip/kimodo/kimodo/tests/output/normal_loco_test"


# ── Post-processing: root projection ────────────────────────────────

def project_root_to_path(
    posed_joints: np.ndarray,
    root_idx: int,
    vel: dict,
    fps: int,
) -> np.ndarray:
    """Rigidly shift all joints so the root follows the exact constant-velocity path."""
    posed = posed_joints.copy()
    num_frames = posed.shape[0]
    dt = 1.0 / fps

    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    wz = vel.get("wz", 0.0)

    t = np.arange(num_frames, dtype=np.float64) * dt

    if abs(wz) < 1e-6:
        desired_x = vy * t
        desired_z = vx * t
    else:
        theta_0 = np.arctan2(vy, vx)
        theta = theta_0 + wz * t
        v_X = vy * np.cos(theta) + vx * np.sin(theta)
        v_Z = vx * np.cos(theta) - vy * np.sin(theta)
        desired_x = np.cumsum(v_X) * dt
        desired_z = np.cumsum(v_Z) * dt

    desired_xy = np.stack([desired_x, desired_z], axis=-1).astype(np.float32)
    current_root_xy = posed[:, root_idx, [0, 2]]
    offset = desired_xy - current_root_xy
    posed[:, :, [0, 2]] += offset[:, np.newaxis, :]

    return posed


# ── Metrics ──────────────────────────────────────────────────────────

def compute_velocity(body_pos_w: np.ndarray, fps: float) -> np.ndarray:
    """Frame-to-frame root speed."""
    return np.linalg.norm(np.diff(body_pos_w[:, 0, :], axis=0), axis=-1) * fps


def jitter_score(vel: np.ndarray) -> float:
    """RMS deviation from 5-frame moving average."""
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(vel, 5)
    return float(np.sqrt(np.mean((vel - smooth) ** 2)))


def analyze(name: str, body_pos_w: np.ndarray, fps: float) -> dict:
    """Compute standard quality metrics."""
    vel = compute_velocity(body_pos_w, fps)
    T = len(vel)
    if T < 90:
        return {"name": name, "error": f"too short ({T} vel frames)"}

    third = T // 3
    head_v = float(np.mean(vel[:30]))
    mid_v = float(np.mean(vel[third: 2 * third]))
    tail_v = float(np.mean(vel[-30:]))
    jh = jitter_score(vel[:third])
    jm = jitter_score(vel[third: 2 * third])
    jt = jitter_score(vel[2 * third:])

    # Check path deviation: how much does root deviate from a straight
    # line between start and end in the (X,Z) plane?
    root_xy = body_pos_w[:, 0, [0, 2]]
    start = root_xy[0]
    end = root_xy[-1]
    direction = end - start
    path_len = float(np.linalg.norm(direction))
    if path_len > 0.01:
        direction /= path_len
        # Signed perpendicular deviation from the start→end line
        deviations = []
        for frame in range(len(root_xy)):
            vec = root_xy[frame] - start
            proj = np.dot(vec, direction)
            perp = vec - proj * direction
            deviations.append(float(np.linalg.norm(perp)))
        max_deviation = float(np.max(deviations))
        rms_deviation = float(np.sqrt(np.mean(np.array(deviations) ** 2)))
    else:
        max_deviation = 0.0
        rms_deviation = 0.0

    return {
        "name": name,
        "frames": len(vel),
        "head_v": head_v,
        "mid_v": mid_v,
        "tail_v": tail_v,
        "t_m_ratio": tail_v / (mid_v + 1e-6),
        "jit_h": jh,
        "jit_m": jm,
        "jit_t": jt,
        "jit_t_h_ratio": jt / (jh + 1e-6),
        "path_len": path_len,
        "max_path_deviation_m": max_deviation,
        "rms_path_deviation_m": rms_deviation,
    }


# ── Main test ────────────────────────────────────────────────────────

def main():
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("=" * 70)
    print("Root Path Constraint Comparison Test")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}  |  Device: {DEVICE}  |  FPS: {FPS}")
    print()

    # Load model once
    print("Loading model...")
    model, _ = load_model(MODEL_NAME, device=DEVICE, return_resolved_name=True)
    root_idx = model.skeleton.root_idx
    print(f"  Skeleton root_idx: {root_idx}\n")

    all_results = []

    for tc in TEST_CASES:
        print(f"── {tc['name']} ──")
        print(f"    vel={tc['vel']}  dur={tc['duration']}s")
        vel = tc["vel"]
        duration = tc["duration"]
        num_frames = int(duration * FPS)

        # ── Approach A: Dense (stride=1) ──
        print("  [A] Dense (stride=1, like 05_root_path)...", end=" ", flush=True)
        constraint_dense = build_root2d_constraint(vel, duration, fps=FPS, stride=1)
        dense_frames = len(constraint_dense["frame_indices"])
        kimodo_constraint = load_constraints_lst(
            [constraint_dense], model.skeleton, device=DEVICE
        )

        t0 = time.time()
        output = model(
            [tc["prompt"]],
            [num_frames],
            constraint_lst=[kimodo_constraint],
            num_denoising_steps=DIFFUSION_STEPS,
            return_numpy=True,
        )
        elapsed = time.time() - t0

        posed_dense = np.asarray(output["posed_joints"][0])
        result_a = analyze(f"{tc['name']}_dense", posed_dense, FPS)
        result_a["approach"] = "dense (stride=1)"
        result_a["constraint_frames"] = dense_frames
        result_a["gen_time_s"] = elapsed
        print(f"✓ {elapsed:.1f}s  ({dense_frames}/{num_frames} frames constrained)")

        # ── Approach B: Sparse (stride=3) + root projection ──
        print("  [B] Sparse (stride=3) + projection...", end=" ", flush=True)
        constraint_sparse = build_root2d_constraint(vel, duration, fps=FPS, stride=3)
        sparse_frames = len(constraint_sparse["frame_indices"])
        kimodo_constraint_b = load_constraints_lst(
            [constraint_sparse], model.skeleton, device=DEVICE
        )

        t0 = time.time()
        output_b = model(
            [tc["prompt"]],
            [num_frames],
            constraint_lst=[kimodo_constraint_b],
            num_denoising_steps=DIFFUSION_STEPS,
            return_numpy=True,
        )
        elapsed_b = time.time() - t0

        posed_sparse = np.asarray(output_b["posed_joints"][0])
        posed_proj = project_root_to_path(posed_sparse, root_idx, vel, FPS)
        result_b = analyze(f"{tc['name']}_sparse+proj", posed_proj, FPS)
        result_b["approach"] = "sparse (stride=3) + projection"
        result_b["constraint_frames"] = sparse_frames
        result_b["gen_time_s"] = elapsed_b
        print(f"✓ {elapsed_b:.1f}s ({sparse_frames}/{num_frames} frames constrained)")

        # Compare
        print()
        print(f"    {'Metric':<25} {'A (dense)':>12} {'B (sparse+proj)':>16}")
        print(f"    {'-'*53}")
        for key, fmt in [
            ("head_v", "12.4f"),
            ("mid_v", "12.4f"),
            ("tail_v", "12.4f"),
            ("t_m_ratio", "12.2f"),
            ("jit_h", "12.4f"),
            ("jit_m", "12.4f"),
            ("jit_t", "12.4f"),
            ("jit_t_h_ratio", "12.1f"),
            ("max_path_deviation_m", "12.4f"),
        ]:
            a_val = result_a.get(key, float("nan"))
            b_val = result_b.get(key, float("nan"))
            label = key.replace("_", " ")
            print(f"    {label:<25} {a_val:{fmt}} {b_val:{fmt}}")
        print()

        all_results.extend([result_a, result_b])

    # ── Save results ─────────────────────────────────────────────────
    import json

    results_path = os.path.join(OUTPUT_BASE, "comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Summary table ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY: Dense (05_root_path) vs Sparse+Projection")
    print("=" * 70)
    print(f"{'Test':<20} {'A T/M':>7} {'A JitT/H':>9} {'B T/M':>7} {'B JitT/H':>9}")
    print("-" * 55)
    for tc in TEST_CASES:
        a = [r for r in all_results if r["name"] == f"{tc['name']}_dense"][0]
        b = [r for r in all_results if r["name"] == f"{tc['name']}_sparse+proj"][0]
        print(
            f"{tc['name']:<20} {a['t_m_ratio']:>7.2f} {a['jit_t_h_ratio']:>9.1f}"
            f" {b['t_m_ratio']:>7.2f} {b['jit_t_h_ratio']:>9.1f}"
        )

    # ✅ / ❌ verdict
    print()
    print("Verdict:")
    dense_better = 0
    sparse_better = 0
    for tc in TEST_CASES:
        a = [r for r in all_results if r["name"] == f"{tc['name']}_dense"][0]
        b = [r for r in all_results if r["name"] == f"{tc['name']}_sparse+proj"][0]
        a_score = abs(a["t_m_ratio"] - 1.0) + a["jit_t_h_ratio"]
        b_score = abs(b["t_m_ratio"] - 1.0) + b["jit_t_h_ratio"]

        if b_score < a_score:
            verdict = "B (sparse+proj) better ✓"
            sparse_better += 1
        else:
            verdict = "A (dense) better"
            dense_better += 1
        print(f"  {tc['name']:<20} → {verdict}")

    print(f"\n  Sparse+projection wins: {sparse_better}/{len(TEST_CASES)}")

    # Cleanup
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
