"""Single-GPU worker subprocess for distributed generation.

Each worker loads a sub-config, calls generate_batch() (the same function
used by single-GPU mode), and pickles the raw tensor results.  This ensures
multi-GPU output is identical in format to single-GPU output — ready for 3D
visualization without any format conversion.

Usage:
    python3 worker.py --config /tmp/gpu0.yaml --gpu 5 --output /tmp/gpu0_results.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Single-GPU generation worker")
    parser.add_argument("--config", required=True, help="Path to per-GPU YAML config")
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU ID")
    parser.add_argument("--output", required=True, help="Pickle file path for results")
    parser.add_argument("--max-batch-size", type=int, default=50, help="Sub-batch size")
    args = parser.parse_args()

    # ── Set up paths before imports ──────────────────────────────
    # worker.py lives at: .../scripts/locomotion_framework/editor/worker.py
    # We need scripts/ on sys.path so "import locomotion_framework" works.
    # this_file:     .../scripts/locomotion_framework/editor/worker.py
    # editor_dir:    .../scripts/locomotion_framework/editor/      [parent]
    # pkg_dir:       .../scripts/locomotion_framework/             [parent^2]
    # scripts_dir:   .../scripts/                                  [parent^3]
    # repo_root:     .../kimodo/                                    [parent^4]
    this_file = Path(__file__).resolve()
    scripts_dir = str(this_file.parent.parent.parent)  # editor → lf → scripts
    repo_root = str(this_file.parent.parent.parent.parent)  # scripts → repo

    for p in [scripts_dir, repo_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # ── Restrict to assigned GPU ─────────────────────────────────
    # CUDA_VISIBLE_DEVICES remaps: our "cuda:0" → physical GPU <args.gpu>
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    # Suppress GPU messages from other libraries
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    t0 = time.time()
    print(f"[worker] GPU {args.gpu} | config={args.config} | device={device} | "
          f"max_batch={args.max_batch_size}", flush=True)

    # ── Imports (after CUDA_VISIBLE_DEVICES is set) ──────────────
    from locomotion_framework.config import load_config
    from locomotion_framework.editor.generation import generate_batch

    # ── Generate ─────────────────────────────────────────────────
    config = load_config(args.config)
    total = sum(
        mt.get("num_samples", getattr(mt, "num_samples", 0))
        if hasattr(mt, "get") else getattr(mt, "num_samples", 0)
        for mt in (
            config.motion_types.values()
            if hasattr(config.motion_types, "values")
            else config.motion_types
        )
    )
    print(f"[worker] GPU {args.gpu}: {len(config.motion_types)} types, "
          f"{total} samples", flush=True)

    result = generate_batch(
        config=config,
        output_base=config.global_.output_dir,
        device=device,
        return_tensors=True,
        max_batch_size=args.max_batch_size,
    )

    elapsed = time.time() - t0
    print(f"[worker] GPU {args.gpu}: DONE — {result['total_motions']} motions "
          f"in {elapsed:.1f}s ({elapsed/max(result['total_motions'],1):.2f}s/sample)",
          flush=True)

    # ── Save results ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    print(f"[worker] GPU {args.gpu}: results saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
