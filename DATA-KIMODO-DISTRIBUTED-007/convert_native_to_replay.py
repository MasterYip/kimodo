#!/usr/bin/env python3
"""Convert a native kimodo output to an RLTracker replay motion.npz.

CONSTRAINT-003 convention (verified against the CONSTRAINT-003 raw archive):
replay `motion.npz` has keys `qpos` [T, 36] float32 (MuJoCo free-root qpos:
root xyz + wxyz quat + 29 joint angles) and `fps` float32 = 30.0.

Native `python -m kimodo.scripts.generate --output <stem>` writes `<stem>.csv`
for the g1 model — the qpos sidecar (240 lines, 36 comma-separated floats, NO
header). This helper packs that CSV into the replay npz. A native `.npz` source
is also accepted via kimodo's MujocoQposConverter (best-effort).

Usage:
    python convert_native_to_replay.py <source.csv> <output_dir>/motion.npz [--fps 30.0]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

DEFAULT_FPS = 30.0


def csv_to_qpos(csv_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with open(csv_path, newline="") as stream:
        for line in csv.reader(stream):
            if not line:
                continue
            rows.append([float(v) for v in line])
    qpos = np.asarray(rows, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"expected qpos [T,36], got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{csv_path}: non-finite qpos")
    quat_err = float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0)))
    if quat_err > 1e-3:
        raise ValueError(f"{csv_path}: root quaternion norm error {quat_err:.3g}")
    return qpos


def npz_to_qpos(npz_path: Path) -> np.ndarray:
    # Best-effort path via kimodo's MujocoQposConverter (requires the kimodo
    # venv). Not used by the default job flow (which uses the CSV sidecar).
    import torch
    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter

    model, _ = load_model("kimodo-g1-rp", device="cpu", return_resolved_name=True)
    converter = MujocoQposConverter(model.skeleton)
    data = dict(np.load(npz_path, allow_pickle=False))
    batched: dict = {}
    for k, v in data.items():
        if v.ndim == 0:
            batched[k] = v
        else:
            batched[k] = torch.from_numpy(v)[None]
    qpos = np.asarray(converter.dict_to_qpos(batched, "cpu"), dtype=np.float32)
    if qpos.ndim == 3:
        qpos = qpos[0]
    return qpos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="native .csv (qpos) or .npz")
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = parser.parse_args()

    if args.source.suffix == ".csv":
        qpos = csv_to_qpos(args.source)
        source_desc = f"csv({args.source.name})"
    elif args.source.suffix == ".npz":
        qpos = npz_to_qpos(args.source)
        source_desc = f"npz({args.source.name})"
    else:
        raise ValueError("source must be .csv or .npz")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, qpos=qpos, fps=np.float32(args.fps))
    print(
        f"wrote {args.output_npz} qpos={qpos.shape} fps={args.fps} "
        f"src={source_desc} frames={len(qpos)}"
    )


if __name__ == "__main__":
    main()
