"""Batch orchestrator for G1 locomotion motion generation.

Usage:
    cd /data/masteryip/kimodo/kimodo
    source scripts/env.sh
    PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
        --config scripts/locomotion_framework/configs/g1_locomotion.yaml
    PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
        --config scripts/locomotion_framework/configs/g1_locomotion.yaml --dry-run
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .config import LocomotionConfig, MotionSpec, VelRange, load_config
from .sampler import MotionSampler, SampledMotion
from .constraints import build_constraints_json


# ── output helpers ───────────────────────────────────────────────


def _save_metadata(out_dir: Path, samples: list[SampledMotion]) -> None:
    """Save a metadata JSON summarizing the generated samples."""
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "motion_type": samples[0].motion_type if samples else "mixed",
        "num_samples": len(samples),
        "parameters": [],
    }
    for s in samples:
        metadata["parameters"].append({
            "motion_type": s.motion_type,
            "prompt": s.prompt,
            "duration_s": round(s.duration, 2),
            "vel": {k: round(v, 4) for k, v in s.vel.items()},
            "torso_height": round(s.torso_height, 3),
            "style": s.style,
            "diffusion_steps": s.diffusion_steps,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _save_manifest(csv_path: Path, all_samples: list[SampledMotion]) -> None:
    """Save a manifest CSV with one row per generated motion."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "motion_type", "prompt", "duration_s",
            "vx", "vy", "wz", "torso_height", "style",
            "npz_path", "csv_path",
        ])
        for i, s in enumerate(all_samples):
            writer.writerow([
                i,
                s.motion_type,
                s.prompt,
                round(s.duration, 2),
                round(s.vel.get("vx", 0), 4),
                round(s.vel.get("vy", 0), 4),
                round(s.vel.get("wz", 0), 4),
                round(s.torso_height, 3),
                s.style,
                f"{s.motion_type}/{s.motion_type}_{i:04d}.npz",
                f"{s.motion_type}/{s.motion_type}_{i:04d}.csv",
            ])
    print(f"Manifest saved: {csv_path} ({len(all_samples)} rows)")


# ── main orchestrator ────────────────────────────────────────────


def run_generation(
    config: LocomotionConfig,
    output_base: Path,
    dry_run: bool = False,
    gpu: int = 0,
):
    """Run the full locomotion batch generation pipeline.

    Args:
        config: Parsed LocomotionConfig.
        output_base: Base directory for outputs.
        dry_run: If True, only print what would be generated.
        gpu: CUDA device index.
    """
    device = f"cuda:{gpu}"
    sampler = MotionSampler(config, seed=config.global_.seed)

    # Generate batch specs
    batch_specs = sampler.generate_batch_specs()
    total_motions = sum(len(s) for s in batch_specs.values())
    print(f"=== Locomotion Batch Generation ===")
    print(f"Model: {config.global_.model}")
    print(f"Types: {len(batch_specs)} | Total motions: {total_motions}")
    print(f"Output: {output_base.resolve()}")
    if dry_run:
        print("DRY RUN — sampling only\n")
    else:
        print(f"Device: {device}\n")

    # Collect all samples for manifest
    all_samples: list[SampledMotion] = []

    for type_name, samples in batch_specs.items():
        n = len(samples)
        spec = config.motion_types[type_name]
        out_dir = output_base / type_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"[{type_name}] {n} samples")
        print(f"  Prompt:  {spec.description}")
        vx_range = spec.vel_cmd.get("vx", VelRange(0.0, 0.0))
        vy_range = spec.vel_cmd.get("vy", VelRange(0.0, 0.0))
        wz_range = spec.vel_cmd.get("wz", VelRange(0.0, 0.0))
        print(f"  Vel:     vx=[{vx_range.min:.2f}, {vx_range.max:.2f}] "
              f"vy=[{vy_range.min:.2f}, {vy_range.max:.2f}] "
              f"wz=[{wz_range.min:.2f}, {wz_range.max:.2f}]")
        print(f"  Torso:   {spec.torso_height_range}")
        print(f"  Styles:  {spec.styles}")

        if dry_run:
            for i, s in enumerate(samples[:3]):
                print(f"  [{i:03d}] {s.prompt}")
                if s.vel:
                    print(f"        vel={s.vel} torso={s.torso_height:.2f}")
            if n > 3:
                print(f"  ... and {n - 3} more")
            all_samples.extend(samples)
            continue

        # Build the prompt (shared across num_samples for this type)
        # Use the first sample's prompt as base; variations come from diffusion
        base_prompt = samples[0].prompt
        num_frames = int(samples[0].duration * config.global_.fps)

        # Build constraints (Root2D path for velocity control)
        constraints = build_constraints_json(samples[0], fps=config.global_.fps)

        print(f"  Prompt:  {base_prompt}")
        print(f"  Frames:  {num_frames} ({samples[0].duration:.1f}s)")
        if constraints:
            print(f"  Constraints: {len(constraints)} set(s)")
            for c in constraints:
                nf = len(c.get("frame_indices", []))
                has_heading = "global_root_heading" in c
                print(f"    - {c['type']}: {nf} frames"
                      f"{' + heading' if has_heading else ''}")

        # Generate
        t0 = time.time()
        _generate_batch(
            config=config,
            samples=samples,
            prompt=base_prompt,
            num_frames=num_frames,
            constraint_lst=constraints,
            out_dir=out_dir,
            device=device,
        )
        elapsed = time.time() - t0
        print(f"  ✓ Generated in {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

        all_samples.extend(samples)

    # Save manifest
    _save_manifest(output_base / "manifest.csv", all_samples)

    if not dry_run:
        print(f"\n{'=' * 60}")
        print(f"Total: {total_motions} motions in {output_base.resolve()}")
        print(f"Manifest: {output_base / 'manifest.csv'}")


def _generate_batch(
    config: LocomotionConfig,
    samples: list[SampledMotion],
    prompt: str,
    num_frames: int,
    constraint_lst: list[dict],
    out_dir: Path,
    device: str,
):
    """Call Kimodo Python API to generate one batch of motions.

    One Kimodo model() call with num_samples=N produces N variations
    from the same prompt and constraints.
    """
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.exports.mujoco import MujocoQposConverter

    # Load model once per batch
    model, resolved_name = load_model(
        config.global_.model,
        device=device,
        return_resolved_name=True,
    )

    n = len(samples)

    # Parse constraints into Kimodo objects
    kimodo_constraints = []
    if constraint_lst:
        # constraint_lst is a list of dicts; convert to Kimodo objects
        kimodo_constraints = load_constraints_lst(
            constraint_lst, model.skeleton, device=device
        )

    seed_val = None
    if config.global_.seed is not None:
        seed_val = config.global_.seed

    # Generate
    output = model(
        prompt,
        num_frames,
        constraint_lst=kimodo_constraints,
        num_denoising_steps=samples[0].diffusion_steps,
        num_samples=n,
        return_numpy=True,
        seed=seed_val,
    )

    # Save outputs
    for i, sample in enumerate(samples):
        single = {
            k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0
                and v.shape[0] == n else v)
            for k, v in output.items()
        }

        stem = f"{sample.motion_type}_{i:04d}"

        # NPZ
        npz_path = out_dir / f"{stem}.npz"
        save_kimodo_npz(str(npz_path), single)

        # CSV (MuJoCo qpos)
        if "g1" in resolved_name.lower():
            converter = MujocoQposConverter(model.skeleton)
            qpos = converter.dict_to_qpos(single, device)
            csv_path = out_dir / f"{stem}.csv"
            np.savetxt(str(csv_path), qpos.cpu().numpy(), delimiter=",")

    # Cleanup
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


# ── entry point ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="G1 Locomotion Batch Generation Framework"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sampled specs without generating",
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "--num-total", "-n", type=int, default=None,
        help="Override: generate exactly N random motions across all types",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.gpu is not None:
        config.global_.gpu = args.gpu

    output_base = Path(args.output or config.global_.output_dir)

    # Override: random weighted sampling instead of fixed per-type counts
    if args.num_total is not None:
        sampler = MotionSampler(config, seed=config.global_.seed)
        all_samples = sampler.generate_random_specs(args.num_total)

        print(f"=== Random Sample Mode ===")
        print(f"Total: {args.num_total} motions across "
              f"{len(set(s.motion_type for s in all_samples))} types")
        if args.dry_run:
            for i, s in enumerate(all_samples[:10]):
                print(f"  [{i:03d}] [{s.motion_type}] {s.prompt}")
            print(f"  ... and {args.num_total - 10} more")
            sys.exit(0)

        # Group by type for generation
        by_type = defaultdict(list)
        for s in all_samples:
            by_type[s.motion_type].append(s)

        for type_name, samples in by_type.items():
            print(f"\n[{type_name}] {len(samples)} samples")
            out_dir = output_base / type_name
            out_dir.mkdir(parents=True, exist_ok=True)

            constraints = build_constraints_json(samples[0])
            num_frames = int(samples[0].duration * config.global_.fps)

            _generate_batch(
                config=config,
                samples=samples,
                prompt=samples[0].prompt,
                num_frames=num_frames,
                constraint_lst=constraints,
                out_dir=out_dir,
                device=f"cuda:{config.global_.gpu}",
            )

        _save_manifest(output_base / "manifest.csv", all_samples)
        return

    run_generation(
        config=config,
        output_base=output_base,
        dry_run=args.dry_run,
        gpu=config.global_.gpu,
    )


if __name__ == "__main__":
    main()
