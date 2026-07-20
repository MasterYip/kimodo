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

# Import export preset registry (populates _registry)
from .export_presets import get_preset, list_presets  # noqa: E402


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


def _save_manifest(csv_path: Path, all_samples: list[SampledMotion],
                   preset: str = "kimodo", seed: int = 0) -> None:
    """Save a manifest CSV with one row per generated motion."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "motion_type", "prompt", "duration_s",
            "vx", "vy", "wz", "torso_height", "style",
            "path",
        ])
        for i, s in enumerate(all_samples):
            if preset == "rltracker":
                from .export_presets.rltracker import build_motion_name
                name = build_motion_name(s.motion_type, s.vel, i, seed)
                path_str = f"{name}/motion.npz"
            else:
                path_str = f"{s.motion_type}/{s.motion_type}_{i:04d}.npz"
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
                path_str,
            ])
    print(f"Manifest saved: {csv_path} ({len(all_samples)} rows)")


# ── main orchestrator ────────────────────────────────────────────


def run_generation(
    config: LocomotionConfig,
    output_base: Path,
    dry_run: bool = False,
    gpu: int = 0,
    preset: str = "kimodo",
):
    """Run the full locomotion batch generation pipeline.

    Args:
        config: Parsed LocomotionConfig.
        output_base: Base directory for outputs.
        dry_run: If True, only print what would be generated.
        gpu: CUDA device index.
        preset: Export preset name ("kimodo" or "rltracker").
    """
    device = f"cuda:{gpu}"
    sampler = MotionSampler(config, seed=config.global_.seed)

    # Generate batch specs
    method = config.global_.sampling_method
    batch_specs = sampler.generate_batch_specs(method=method)
    total_motions = sum(len(s) for s in batch_specs.values())
    print(f"=== Locomotion Batch Generation ===")
    print(f"Model: {config.global_.model} | Sampling: {method} | Preset: {preset}")
    print(f"Types: {len(batch_specs)} | Total motions: {total_motions}")
    print(f"Output: {output_base.resolve()}")
    if dry_run:
        print("DRY RUN — sampling only\n")
    else:
        print(f"Device: {device}\n")

    # In dry-run mode with rltracker, show naming preview
    if dry_run and preset == "rltracker":
        from .export_presets.rltracker import build_motion_name
        seed = config.global_.seed or 0
        global_idx = 0
        for type_name, samples in batch_specs.items():
            for s in samples[:3]:
                name = build_motion_name(s.motion_type, s.vel, global_idx, seed)
                print(f"  [{global_idx:03d}] {name}")
                print(f"        prompt: {s.prompt}")
                if s.vel:
                    print(f"        vel={s.vel} torso={s.torso_height:.2f}")
                global_idx += 1
            global_idx += max(0, len(samples) - 3)
        print()

    # Collect all samples for manifest
    all_samples: list[SampledMotion] = []

    for type_name, samples in batch_specs.items():
        n = len(samples)
        spec = config.motion_types[type_name]
        out_dir = output_base / type_name

        if not (dry_run and preset == "rltracker"):
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
            if preset == "rltracker":
                # Naming preview already shown above; just extend samples
                all_samples.extend(samples)
            else:
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
            out_dir=output_base,  # rltracker: flat output; kimodo: per-type subdir
            device=device,
            preset=preset,
        )
        elapsed = time.time() - t0
        print(f"  ✓ Generated in {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

        all_samples.extend(samples)

    # Save manifest
    _save_manifest(output_base / "manifest.csv", all_samples,
                   preset=preset, seed=config.global_.seed or 0)

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
    preset: str = "kimodo",
):
    """Call Kimodo Python API to generate one batch of motions.

    One Kimodo model() call with num_samples=N produces N variations
    from the same prompt and constraints.

    Args:
        preset: "kimodo" for default NPZ+CSV output, "rltracker" for
                RLTracker dataset format (flat dirs with motion.npz).
    """
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst

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
    )

    fps = config.global_.fps

    # Export per-sample
    for i, sample in enumerate(samples):
        single = {
            k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0
                and v.shape[0] == n else v)
            for k, v in output.items()
        }

        if preset == "rltracker":
            _export_rltracker(
                single=single,
                model=model,
                fps=fps,
                sample_idx=i,
                sample=sample,
                seed=seed_val or 0,
                output_base=out_dir,
                device=device,
            )
        else:
            _export_kimodo(
                single=single,
                model=model,
                resolved_name=resolved_name,
                sample=sample,
                out_dir=out_dir / sample.motion_type,
                device=device,
            )

    # Cleanup
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def _export_kimodo(
    single: dict,
    model,
    resolved_name: str,
    sample: SampledMotion,
    out_dir: Path,
    device: str,
):
    """Default Kimodo export: NPZ + optional MuJoCo CSV."""
    import numpy as np
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.exports.mujoco import MujocoQposConverter

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sample.motion_type}_{sample.motion_type}_{0:04d}"  # fallback
    # Use proper index from sample if available
    idx = getattr(sample, '_global_idx', 0)
    stem = f"{sample.motion_type}_{idx:04d}"

    # NPZ
    npz_path = out_dir / f"{stem}.npz"
    save_kimodo_npz(str(npz_path), single)

    # CSV (MuJoCo qpos)
    if "g1" in resolved_name.lower():
        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(single, device)
        csv_path = out_dir / f"{stem}.csv"
        np.savetxt(str(csv_path), qpos.cpu().numpy(), delimiter=",")


def _export_rltracker(
    single: dict,
    model,
    fps: float,
    sample_idx: int,
    sample: SampledMotion,
    seed: int,
    output_base: Path,
    device: str,
):
    """Export one motion in RLTracker dataset format.

    Uses Kimodo FK + MujocoQposConverter to build the 7-key NPZ
    (joint_pos, joint_vel, body_pos_w, body_quat_w,
     body_lin_vel_w, body_ang_vel_w, fps).
    """
    import numpy as np
    import torch

    # Run FK to get posed_joints + global_rot_mats
    local_rot_mats_t = torch.from_numpy(
        np.asarray(single["local_rot_mats"])
    ).float().to(device)
    root_positions_t = torch.from_numpy(
        np.asarray(single["root_positions"])
    ).float().to(device)

    if local_rot_mats_t.dim() == 3:
        local_rot_mats_t = local_rot_mats_t.unsqueeze(0)
    if root_positions_t.dim() == 1:
        root_positions_t = root_positions_t.unsqueeze(0)

    global_rot_mats, posed_joints, _ = model.skeleton.fk(
        local_rot_mats_t, root_positions_t
    )

    # Convert to numpy
    posed_joints_np = posed_joints.squeeze(0).cpu().numpy().astype(np.float32)
    global_rot_mats_np = global_rot_mats.squeeze(0).cpu().numpy().astype(np.float32)
    local_rot_mats_np = local_rot_mats_t.squeeze(0).cpu().numpy().astype(np.float32)
    root_positions_np = root_positions_t.squeeze(0).cpu().numpy().astype(np.float32)

    from .export_presets import get_preset
    exporter = get_preset("rltracker")
    exporter.export(
        posed_joints=posed_joints_np,
        global_rot_mats=global_rot_mats_np,
        local_rot_mats=local_rot_mats_np,
        root_positions=root_positions_np,
        fps=float(fps),
        sample_idx=sample_idx,
        motion_type=sample.motion_type,
        vel=sample.vel,
        torso_height=sample.torso_height,
        style=sample.style,
        seed=seed,
        output_base=output_base,
    )


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
    parser.add_argument(
        "--method", "-m", type=str, default=None,
        choices=["uniform", "lhs"],
        help="Sampling method (overrides config). 'lhs' = Latin Hypercube Sampling.",
    )
    parser.add_argument(
        "--preset", "-p", type=str, default=None,
        choices=["kimodo", "rltracker"],
        help="Export preset (overrides config). 'rltracker' = flat dirs with motion.npz.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.gpu is not None:
        config.global_.gpu = args.gpu
    if args.method is not None:
        config.global_.sampling_method = args.method
    if args.preset is not None:
        config.global_.export_preset = args.preset

    preset = config.global_.export_preset
    output_base = Path(args.output or config.global_.output_dir)

    # Override: random weighted sampling instead of fixed per-type counts
    if args.num_total is not None:
        sampler = MotionSampler(config, seed=config.global_.seed)
        method = config.global_.sampling_method
        all_samples = sampler.generate_random_specs(args.num_total, method=method)

        print(f"=== Random Sample Mode (method={method}, preset={preset}) ===")
        print(f"Total: {args.num_total} motions across "
              f"{len(set(s.motion_type for s in all_samples))} types")
        if args.dry_run:
            show_n = min(len(all_samples), 10)
            for i, s in enumerate(all_samples[:show_n]):
                if preset == "rltracker":
                    from .export_presets.rltracker import build_motion_name
                    name = build_motion_name(
                        s.motion_type, s.vel, i, config.global_.seed or 0
                    )
                    print(f"  [{i:03d}] {name}  |  [{s.motion_type}] {s.prompt}")
                else:
                    print(f"  [{i:03d}] [{s.motion_type}] {s.prompt}")
            if len(all_samples) > show_n:
                print(f"  ... and {len(all_samples) - show_n} more")
            sys.exit(0)

        # Group by type for generation
        by_type = defaultdict(list)
        for s in all_samples:
            by_type[s.motion_type].append(s)

        for type_name, samples in by_type.items():
            print(f"\n[{type_name}] {len(samples)} samples")
            constraints = build_constraints_json(samples[0])
            num_frames = int(samples[0].duration * config.global_.fps)

            _generate_batch(
                config=config,
                samples=samples,
                prompt=samples[0].prompt,
                num_frames=num_frames,
                constraint_lst=constraints,
                out_dir=output_base,
                device=f"cuda:{config.global_.gpu}",
                preset=preset,
            )

        _save_manifest(output_base / "manifest.csv", all_samples,
                       preset=preset, seed=config.global_.seed or 0)
        return

    run_generation(
        config=config,
        output_base=output_base,
        dry_run=args.dry_run,
        gpu=config.global_.gpu,
        preset=preset,
    )


if __name__ == "__main__":
    main()
