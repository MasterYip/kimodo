#!/usr/bin/env python3
"""Test: does batch generation (list API) produce same quality as single-sample (demo)?

Compares 4 generation modes for the same 4 test cases:
  1. Single + no CFG  (like "Generate" button, cfg off)
  2. Single + CFG      (like "Generate" button, cfg [2.0, 2.0])
  3. Batch + no CFG    (our orchestrator list API)
  4. Batch + CFG       (our orchestrator list API + cfg)

All use the SAME constraints (dense stride=1, with global_root_heading for curves).

Usage:
    cd /data/masteryip/kimodo/kimodo/scripts
    source ./env.sh
    PYTHONPATH=. python3 locomotion_framework/test_batch_vs_demo.py
"""

import json
import numpy as np
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from locomotion_framework.constraints import build_root2d_constraint

FPS = 30
DIFFUSION_STEPS = 100
SEED = 42
DEVICE = "cuda:0"
MODEL = "kimodo-g1-rp"
OUTPUT = "/data/masteryip/kimodo/kimodo/tests/output/batch_vs_demo"

TEST_CASES = [
    # ── Mid-range (safe zone) ──
    {"name": "walk_fwd_safe",  "prompt": "a person walks forward",
     "vel": {"vx": 0.5, "vy": 0.0, "wz": 0.0}, "dur": 5.0},
    {"name": "run_fwd_safe",   "prompt": "a person runs forward",
     "vel": {"vx": 1.5, "vy": 0.0, "wz": 0.0}, "dur": 5.0},

    # ── Edge cases (matching full LHS distribution ranges) ──
    # Walk edges: vx up to 0.80, vy up to 0.40, wz up to 0.50
    {"name": "walk_fast_fwd",  "prompt": "a person walks forward",
     "vel": {"vx": 0.80, "vy": 0.0, "wz": 0.0}, "dur": 5.0},
    {"name": "walk_fast_bwd",  "prompt": "a person walks backward",
     "vel": {"vx": -0.60, "vy": 0.0, "wz": 0.0}, "dur": 5.0},
    {"name": "walk_lat_edge",  "prompt": "a person walks sideways",
     "vel": {"vx": 0.0, "vy": 0.40, "wz": 0.0}, "dur": 5.0},
    {"name": "walk_turn_fast", "prompt": "a person walks turning",
     "vel": {"vx": 0.3, "vy": 0.0, "wz": 0.50}, "dur": 5.0},

    # Run edges: vx up to 2.00
    {"name": "run_sprint",     "prompt": "a person runs at high speed",
     "vel": {"vx": 2.00, "vy": 0.0, "wz": 0.0}, "dur": 5.0},
    {"name": "run_turn_slow",  "prompt": "a person runs turning slowly",
     "vel": {"vx": 1.0, "vy": 0.1, "wz": 0.15}, "dur": 5.0},
]


def jitter_score(vel):
    from scipy.ndimage import uniform_filter1d
    return float(np.sqrt(np.mean((vel - uniform_filter1d(vel, 5)) ** 2)))


def analyze(body_pos_w, fps, label):
    vel = np.linalg.norm(np.diff(body_pos_w[:, 0, :], axis=0), axis=-1) * fps
    T = len(vel)
    if T < 90:
        return {"label": label, "error": "too short"}
    th = T // 3
    return {
        "label": label,
        "head_v": float(np.mean(vel[:30])),
        "mid_v": float(np.mean(vel[th:2*th])),
        "tail_v": float(np.mean(vel[-30:])),
        "t_m": float(np.mean(vel[-30:]) / (np.mean(vel[th:2*th]) + 1e-6)),
        "jit_h": jitter_score(vel[:th]),
        "jit_t": jitter_score(vel[2*th:]),
        "jit_th": jitter_score(vel[2*th:]) / (jitter_score(vel[:th]) + 1e-6),
    }


def save_npz(name, suffix, output_sample, out_dir):
    import numpy as np
    d = os.path.join(out_dir, f"{name}__{suffix}")
    os.makedirs(d, exist_ok=True)
    save = {}
    for k in ("posed_joints", "global_rot_mats", "local_rot_mats",
              "root_positions", "foot_contacts"):
        v = output_sample.get(k)
        if v is not None:
            save[k] = np.asarray(v).astype(np.float32)
    np.savez_compressed(os.path.join(d, "motion.npz"), **save)


def main():
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst

    os.makedirs(OUTPUT, exist_ok=True)

    print("=" * 70)
    print("Batch API vs Single-Sample Comparison")
    print("=" * 70)

    model, _ = load_model(MODEL, device=DEVICE, return_resolved_name=True)
    root_idx = model.skeleton.root_idx

    all_results = []

    for tc in TEST_CASES:
        print(f"\n── {tc['name']} vel={tc['vel']} dur={tc['dur']}s ──")

        nf = int(tc["dur"] * FPS)
        constraint = build_root2d_constraint(tc["vel"], tc["dur"], fps=FPS, stride=1)
        kc = load_constraints_lst([constraint], model.skeleton, device=DEVICE)
        n_constrained = len(constraint["frame_indices"])

        # ── Mode 1: Single, no CFG (like demo Generate without CFG checkbox) ──
        print("  [1] Single, no CFG...", end=" ", flush=True)
        t0 = time.time()
        out1 = model(
            [tc["prompt"]], [nf],
            constraint_lst=[kc],
            num_denoising_steps=DIFFUSION_STEPS,
            return_numpy=True,
        )
        dt1 = time.time() - t0
        p1 = np.asarray(out1["posed_joints"][0])
        r1 = analyze(p1, FPS, f"{tc['name']}_single_nocfg")
        r1["mode"] = "single_nocfg"
        r1["constrained"] = n_constrained
        r1["time_s"] = dt1
        save_npz(tc["name"], "1_single_nocfg", {k: v[0] for k, v in out1.items()}, OUTPUT)
        print(f"✓ {dt1:.1f}s")

        # ── Mode 2: Single, CFG (like demo Generate with CFG) ──
        print("  [2] Single, CFG [2,2]...", end=" ", flush=True)
        t0 = time.time()
        try:
            out2 = model(
                [tc["prompt"]], [nf],
                constraint_lst=[kc],
                num_denoising_steps=DIFFUSION_STEPS,
                cfg_weight=[2.0, 2.0],
                return_numpy=True,
            )
            dt2 = time.time() - t0
            p2 = np.asarray(out2["posed_joints"][0])
            r2 = analyze(p2, FPS, f"{tc['name']}_single_cfg")
            r2["mode"] = "single_cfg"
            r2["constrained"] = n_constrained
            r2["time_s"] = dt2
            save_npz(tc["name"], "2_single_cfg", {k: v[0] for k, v in out2.items()}, OUTPUT)
            print(f"✓ {dt2:.1f}s")
        except Exception as e:
            print(f"✗ {e}")
            r2 = {"label": f"{tc['name']}_single_cfg", "mode": "single_cfg", "error": str(e)}

        # ── Mode 3: Mixed batch (different per sample) = orchestrator sim ──
        # Skip the 5x same-constraint batch; already proved identical to single.

        all_results.extend([r1, r2])

    # ── Mode 5+6: Mixed batch (different constraint per sample) ───────
    # This is what our orchestrator does: 50 DIFFERENT velocities in one batch.
    # Generate all 4 test cases in a single batch call.

    print(f"\n── Mode 3: Mixed batch (4 different constraints), no CFG ──")
    all_prompts = [tc["prompt"] for tc in TEST_CASES]
    all_frames = [int(tc["dur"] * FPS) for tc in TEST_CASES]
    all_kc = []
    for tc in TEST_CASES:
        c = build_root2d_constraint(tc["vel"], tc["dur"], fps=FPS, stride=1)
        all_kc.append(load_constraints_lst([c], model.skeleton, device=DEVICE))
    try:
        t0 = time.time()
        out5 = model(
            all_prompts, all_frames,
            constraint_lst=all_kc,
            num_denoising_steps=DIFFUSION_STEPS,
            return_numpy=True,
        )
        dt5 = time.time() - t0
        for bi, tc in enumerate(TEST_CASES):
            pi = np.asarray(out5["posed_joints"][bi])
            ri = analyze(pi, FPS, f"{tc['name']}_mixed_nocfg")
            ri["mode"] = "mixed_nocfg"
            ri["time_s"] = dt5 / len(TEST_CASES)
            all_results.append(ri)
            save_npz(tc["name"], "3_mixed_nocfg",
                     {k: v[bi] for k, v in out5.items()}, OUTPUT)
        print(f"  ✓ {dt5:.1f}s ({dt5/len(TEST_CASES):.1f}s/sample)")
    except Exception as e:
        print(f"  ✗ {e}")
        for tc in TEST_CASES:
            all_results.append({"label": f"{tc['name']}_mixed_nocfg", "mode": "mixed_nocfg", "error": str(e)})

    print(f"\n── Mode 4: Mixed batch (4 different constraints), CFG [2,2] ──")
    try:
        t0 = time.time()
        out6 = model(
            all_prompts, all_frames,
            constraint_lst=all_kc,
            num_denoising_steps=DIFFUSION_STEPS,
            cfg_weight=[2.0, 2.0],
            return_numpy=True,
        )
        dt6 = time.time() - t0
        for bi, tc in enumerate(TEST_CASES):
            pi = np.asarray(out6["posed_joints"][bi])
            ri = analyze(pi, FPS, f"{tc['name']}_mixed_cfg")
            ri["mode"] = "mixed_cfg"
            ri["time_s"] = dt6 / len(TEST_CASES)
            all_results.append(ri)
            save_npz(tc["name"], "4_mixed_cfg",
                     {k: v[bi] for k, v in out6.items()}, OUTPUT)
        print(f"  ✓ {dt6:.1f}s ({dt6/len(TEST_CASES):.1f}s/sample)")
    except Exception as e:
        print(f"  ✗ {e}")
        for tc in TEST_CASES:
            all_results.append({"label": f"{tc['name']}_mixed_cfg", "mode": "mixed_cfg", "error": str(e)})

    # ── Aggregate ────────────────────────────────────────────────────
    # Group batch results by mode
    batch_modes = {}
    for r in all_results:
        mode = r["mode"]
        if mode not in batch_modes:
            batch_modes[mode] = {"t_m": [], "jit_th": []}
        if "t_m" in r:
            batch_modes[mode]["t_m"].append(r["t_m"])
            batch_modes[mode]["jit_th"].append(r["jit_th"])

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS: Batch vs Single, CFG vs no-CFG")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n{tc['name']}:")
        print(f"  {'Mode':<20} {'T/M':>7} {'JitT/H':>8}")
        print(f"  {'-'*35}")
        for mode_label, mode_key in [
            ("1. single no-CFG", "single_nocfg"),
            ("2. single CFG", "single_cfg"),
            ("3. mixed no-CFG", "mixed_nocfg"),
            ("4. mixed CFG", "mixed_cfg"),
        ]:
            mode_results = [r for r in all_results
                          if r["mode"] == mode_key
                          and r["label"].startswith(tc["name"])]
            for r in mode_results:
                if "t_m" in r:
                    print(f"  {mode_label:<20} {r['t_m']:>7.2f} {r['jit_th']:>8.1f}")

    # Summary by mode
    print("\n" + "=" * 70)
    print("AGGREGATE (mean across all tests)")
    print("=" * 70)
    for mode_label, mode_key in [
        ("1. single no-CFG", "single_nocfg"),
        ("2. single CFG", "single_cfg"),
        ("3. mixed no-CFG", "mixed_nocfg"),
        ("4. mixed CFG", "mixed_cfg"),
    ]:
        tms = [r["t_m"] for r in all_results
               if r["mode"] == mode_key and "t_m" in r]
        jits = [r["jit_th"] for r in all_results
               if r["mode"] == mode_key and "jit_th" in r]
        if tms:
            print(f"  {mode_label:<20} T/M={np.mean(tms):.2f}±{np.std(tms):.2f}  "
                  f"JitT/H={np.mean(jits):.1f}±{np.std(jits):.1f}  (n={len(tms)})")

    # ── Safe vs Edge breakdown (the key question) ───────────────────
    print("\n" + "=" * 70)
    print("SAFE vs EDGE: Does quality degrade at velocity extremes?")
    print("=" * 70)

    safe_names = {"walk_fwd_safe", "run_fwd_safe"}
    edge_names = {tc["name"] for tc in TEST_CASES} - safe_names

    for group_label, group_names in [("SAFE (mid-range)", safe_names),
                                      ("EDGE (full range extremes)", edge_names)]:
        for mode_label, mode_key in [
            ("single no-CFG", "single_nocfg"),
            ("single CFG", "single_cfg"),
            ("mixed no-CFG", "mixed_nocfg"),
            ("mixed CFG", "mixed_cfg"),
        ]:
            tms = [r["t_m"] for r in all_results
                   if r["mode"] == mode_key and "t_m" in r
                   and any(r["label"].startswith(n) for n in group_names)]
            jits = [r["jit_th"] for r in all_results
                   if r["mode"] == mode_key and "jit_th" in r
                   and any(r["label"].startswith(n) for n in group_names)]
            if tms:
                print(f"  {group_label:<28} {mode_label:<20} "
                      f"T/M={np.mean(tms):.2f}±{np.std(tms):.2f}  "
                      f"JitT/H={np.mean(jits):.1f}±{np.std(jits):.1f}  n={len(tms)}")

    # Save
    with open(os.path.join(OUTPUT, "batch_vs_demo.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults: {OUTPUT}/batch_vs_demo.json")
    print(f"Motions: {OUTPUT}/")

    del model
    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
