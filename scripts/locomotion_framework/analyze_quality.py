#!/usr/bin/env python3
"""Analyze quality of all motions in an RLTracker dataset output directory."""
import numpy as np
import json
import sys
import csv
from pathlib import Path
from scipy.ndimage import uniform_filter1d


def jitter_score(vel):
    smooth = uniform_filter1d(np.asarray(vel, dtype=np.float64), 5)
    return float(np.sqrt(np.mean((vel - smooth) ** 2)))


def analyze_one(npz_path, manifest_map=None):
    data = np.load(npz_path, allow_pickle=True)
    name = npz_path.parent.name

    for key in ["body_pos_w", "posed_joints", "root_positions"]:
        if key not in data:
            continue
        body_pos = np.asarray(data[key])
        fps = float(np.asarray(data.get("fps", np.array([30]))).reshape(-1)[0])
        T = body_pos.shape[0]
        root = body_pos[:, 0, :] if body_pos.ndim >= 3 else body_pos
        vel = np.linalg.norm(np.diff(root, axis=0), axis=-1) * fps

        if len(vel) < 60:
            return {"name": name, "error": f"too short ({len(vel)} vel frames)"}

        third = len(vel) // 3
        jh = jitter_score(vel[:third])
        jm = jitter_score(vel[third: 2 * third])
        jt = jitter_score(vel[2 * third:])

        # Get expected info from manifest if available
        info = manifest_map.get(name, {}) if manifest_map else {}
        return {
            "name": name,
            "dur_s": T / fps,
            "head_v": float(np.mean(vel[:20])),
            "mid_v": float(np.mean(vel[third: 2 * third])),
            "tail_v": float(np.mean(vel[-20:])),
            "t_m": float(np.mean(vel[-20:]) / (np.mean(vel[third: 2 * third]) + 1e-6)),
            "jit_h": jh,
            "jit_m": jm,
            "jit_t": jt,
            "jit_th": jt / (jh + 1e-6),
            "motion_type": info.get("motion_type", ""),
        }

    return {"name": name, "error": "no position data"}


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    manifest_path = out_dir / "manifest.csv"

    manifest_map = {}
    if manifest_path.exists():
        with manifest_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                mname = Path(row["path"]).parent.name
                manifest_map[mname] = {
                    "motion_type": row["motion_type"],
                    "vx": float(row.get("vx", 0)),
                    "vy": float(row.get("vy", 0)),
                    "wz": float(row.get("wz", 0)),
                }

    results = []
    for npz_path in sorted(out_dir.rglob("motion.npz")):
        r = analyze_one(npz_path, manifest_map)
        results.append(r)

    # Group by type
    by_type = {}
    for r in results:
        mtype = r.get("motion_type", "unknown") or "unknown"
        by_type.setdefault(mtype, []).append(r)

    print("=" * 70)
    print(f"Quality Analysis: {out_dir} ({len(results)} motions)")
    print("=" * 70)

    all_good = [r for r in results if "jit_th" in r]
    all_bad = [r for r in results if "error" in r]

    # Per-type summary
    for mtype in sorted(by_type.keys()):
        rs = [r for r in by_type[mtype] if "jit_th" in r]
        re = [r for r in by_type[mtype] if "error" in r]
        if not rs:
            continue
        jits = [r["jit_th"] for r in rs]
        tms = [r["t_m"] for r in rs]
        tails = [r["tail_v"] for r in rs]
        mids = [r["mid_v"] for r in rs]
        print(f"\n{mtype} ({len(rs)} ok + {len(re)} err):")
        print(f"  T/M:      {np.mean(tms):.2f} ± {np.std(tms):.2f}")
        print(f"  JitT/H:   {np.mean(jits):.1f} ± {np.std(jits):.1f}  "
              f"[{np.min(jits):.1f}, {np.max(jits):.1f}]")
        print(f"  Tail decel: mid={np.mean(mids):.3f} → tail={np.mean(tails):.3f} "
              f"(ratio={np.mean(tails)/(np.mean(mids)+1e-6):.2f})")

    # Overall
    print("\n" + "=" * 70)
    print(f"OVERALL: {len(all_good)} good, {len(all_bad)} errors")
    if all_good:
        jits = [r["jit_th"] for r in all_good]
        tms = [r["t_m"] for r in all_good]
        print(f"  T/M:    {np.mean(tms):.2f} ± {np.std(tms):.2f}")
        print(f"  JitT/H: {np.mean(jits):.1f} ± {np.std(jits):.1f}  "
              f"[{np.min(jits):.1f}, {np.max(jits):.1f}]")

        # Count categories
        counts = {"excellent (x<2)": 0, "good (2-3x)": 0,
                  "ok (3-5x)": 0, "bad (>=5x)": 0, "unknown": 0}
        for j in jits:
            if j < 2: counts["excellent (x<2)"] += 1
            elif j < 3: counts["good (2-3x)"] += 1
            elif j < 5: counts["ok (3-5x)"] += 1
            else: counts["bad (>=5x)"] += 1
        print(f"\n  Distribution:")
        for label, cnt in counts.items():
            if cnt:
                print(f"    {label}: {cnt} ({100*cnt/len(all_good):.0f}%)")

    # Save
    with open(str(out_dir / "quality_analysis.json"), "w") as f:
        json.dump({
            "overall_t_m": float(np.mean(tms)) if all_good else None,
            "overall_jit_th": float(np.mean(jits)) if all_good else None,
            "per_type": {mtype: {
                "n": len([r for r in rs if "jit_th" in r]),
                "mean_jit_th": float(np.mean([r["jit_th"] for r in rs if "jit_th" in r])) if rs else None,
                "mean_t_m": float(np.mean([r["t_m"] for r in rs if "t_m" in r])) if rs else None,
            } for mtype, rs in by_type.items()},
            "all": [r for r in results],
            "errors": [e["name"] for e in all_bad],
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'quality_analysis.json'}")


if __name__ == "__main__":
    main()
