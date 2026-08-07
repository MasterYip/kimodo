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


def _polar_components(vel: dict[str, float]) -> tuple[float, float]:
    """Return planar speed and direction in degrees from sampled vx/vy.

    Direction uses the native compass (0 = +forward, +90 = +left,
    -90 = +right) — the same convention as ``polar_to_cartesian``.
    """
    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    return float(np.hypot(vx, vy)), float(np.degrees(np.arctan2(vy, vx)))


def _save_metadata(out_dir: Path, samples: list[SampledMotion]) -> None:
    """Save a metadata JSON summarizing the generated samples."""
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "motion_type": samples[0].motion_type if samples else "mixed",
        "num_samples": len(samples),
        "parameters": [],
    }
    for s in samples:
        speed_mps, direction_deg = _polar_components(s.vel)
        metadata["parameters"].append({
            "motion_type": s.motion_type,
            "prompt": s.prompt,
            "duration_s": round(s.duration, 2),
            "vel": {k: round(v, 4) for k, v in s.vel.items()},
            "speed_mps": round(speed_mps, 4),
            "direction_deg": round(direction_deg, 4),
            "torso_height": round(s.torso_height, 3) if s.torso_height is not None else None,
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
            "vx", "vy", "wz", "speed_mps", "direction_deg", "torso_height", "style",
            "path",
        ])
        for i, s in enumerate(all_samples):
            speed_mps, direction_deg = _polar_components(s.vel)
            if preset == "rltracker":
                if getattr(s, "output_name", None):
                    name = s.output_name
                else:
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
                round(speed_mps, 4),
                round(direction_deg, 4),
                round(s.torso_height, 3) if s.torso_height is not None else "",
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
    batch_specs = sampler.generate_batch_specs(method=method, rerank=config.global_.rerank)
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
            if spec.polar_vel_cmd is not None:
                print(f"  Vel(polar): speed=[{spec.polar_vel_cmd.speed.min:.2f}, "
                      f"{spec.polar_vel_cmd.speed.max:.2f}] m/s  direction_deg="
                      f"[{spec.polar_vel_cmd.direction_deg.min:.1f}, "
                      f"{spec.polar_vel_cmd.direction_deg.max:.1f}] (0 fwd, +90 left)")
                wz_range = spec.vel_cmd.get("wz", VelRange(0.0, 0.0))
                print(f"  Vel:     wz=[{wz_range.min:.2f}, {wz_range.max:.2f}]")
            else:
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
        # Generate: per-sample prompts, frames, constraints via list API
        base_prompt = samples[0].prompt  # for display only
        num_frames = int(samples[0].duration * config.global_.fps)

        print(f"  Prompt:  {base_prompt}")
        print(f"  Dur:     {samples[0].duration:.1f}s, {num_frames}f")
        print(f"  Constraint per sample (e.g.): vx={samples[0].vel.get('vx',0):.2f} "
              f"vy={samples[0].vel.get('vy',0):.2f} wz={samples[0].vel.get('wz',0):.2f}")
        print(f"  Per-sample mode: {n} unique constraints + prompts + durations")

        t0 = time.time()
        _generate_batch(
            config=config,
            samples=samples,
            out_dir=output_base,
            device=device,
            preset=preset,
        )
        elapsed = time.time() - t0
        print(f"  ✓ Generated in {elapsed:.1f}s ({elapsed/n:.2f}s/sample)")

        all_samples.extend(samples)

    if dry_run:
        print(f"Dry run complete: {total_motions} sampled motions; no files written.")
        return

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
    out_dir: Path,
    device: str,
    preset: str = "kimodo",
):
    """Call Kimodo Python API to generate one batch of motions.

    Uses Kimodo's **list API**: passing ``prompts`` as a list of strings,
    ``num_frames`` as a list of ints, and ``constraint_lst`` as a list of
    per-sample constraint lists causes each sample to get its OWN prompt,
    duration and Root2D path — the velocity distribution from the sampler
    actually takes effect.

    Uses the demo ``05_root_path`` method: dense stride=1 constraints,
    exact-duration generation (no margin/truncation).  The diffusion model
    with dense inpainting at every DDIM step produces clean motions without
    tail jitter — no post-processing needed.

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
    fps = config.global_.fps
    seed_val = config.global_.seed if config.global_.seed is not None else 0

    # ── Build per-sample lists (demo 05_root_path method) ────────────
    per_prompts: list[str] = []
    per_frames: list[int] = []
    per_constraints_raw: list[list[dict]] = []

    for s in samples:
        per_prompts.append(s.prompt)
        per_frames.append(int(round(s.duration * fps)))
        per_constraints_raw.append(
            build_constraints_json(s, fps=fps)
        )

    # Convert to Kimodo constraint objects
    per_kimodo_constraints: list[list] = []
    for raw_lst in per_constraints_raw:
        if raw_lst:
            per_kimodo_constraints.append(
                load_constraints_lst(raw_lst, model.skeleton, device=device)
            )
        else:
            per_kimodo_constraints.append([])

    # Use list-of-constraints to signal per-sample mode
    kimodo_constraints = per_kimodo_constraints  # list[list]

    # ── Generate: list API → each sample gets its own prompt + constraint ──
    # CFG passed explicitly for parity with native generation
    # (``--cfg_type separated --cfg_weight 2.0 2.0``). The model default is the
    # same separated [2.0, 2.0], but the config now controls it.
    output = model(
        per_prompts,               # list[str] → num_samples = len(prompts)
        per_frames,                # list[int] → per-sample durations
        constraint_lst=kimodo_constraints,  # list[list] → per-sample
        num_denoising_steps=samples[0].diffusion_steps,
        cfg_type=config.global_.cfg_type,
        cfg_weight=list(config.global_.cfg_weight),
        return_numpy=True,
    )

    # ── Export per-sample (trim to actual duration — model pads to max in batch) ──
    for i, sample in enumerate(samples):
        single = {
            k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0
                and v.shape[0] == n else v)
            for k, v in output.items()
        }

        # Trim to actual sample duration (model pads all samples to max_frames)
        actual_frames = int(round(sample.duration * fps))
        for k in list(single.keys()):
            arr = single[k]
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim >= 1 and arr.shape[0] > actual_frames:
                single[k] = arr[:actual_frames].copy()

        if preset == "rltracker":
            _export_rltracker(
                single=single,
                model=model,
                fps=fps,
                sample_idx=i,
                sample=sample,
                seed=seed_val,
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


def _trim_motion(single: dict, start: int, end: int | None) -> dict:
    """Slice the temporal dimension of every array in ``single``.

    Args:
        single: Per-sample motion dict.
        start:  Frames to drop from the beginning.
        end:    Frames to drop from the end (negative index or None).

    Returns:
        A new dict with truncated arrays.  Scalars and arrays whose
        first dimension doesn't match are passed through unchanged.
    """
    import numpy as np

    trimmed = {}
    # Determine the expected temporal length from root_positions
    ref_len = 0
    for k in ("root_positions", "posed_joints", "joint_pos"):
        arr = single.get(k)
        if arr is not None and hasattr(arr, "shape") and len(arr.shape) >= 1:
            ref_len = arr.shape[0]
            break

    for k, v in single.items():
        if v is None:
            trimmed[k] = v
            continue
        arr = np.asarray(v)
        if len(arr.shape) >= 1 and arr.shape[0] == ref_len:
            if end is not None:
                trimmed[k] = arr[start:end].copy()
            else:
                trimmed[k] = arr[start:].copy()
        else:
            trimmed[k] = arr

    return trimmed


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

    Output at Kimodo's **native** 30 fps.  No temporal resampling is applied —
    resampling artefacts at wrist joints were causing body flickering in the
    simulator.  The RLTracker viewer handles the fps mismatch.
    """
    import numpy as np

    # Model output already has all 7 keys as numpy arrays (return_numpy=True)
    posed_joints_np = np.asarray(single["posed_joints"]).astype(np.float32)
    global_rot_mats_np = np.asarray(single["global_rot_mats"]).astype(np.float32)
    local_rot_mats_np = np.asarray(single["local_rot_mats"]).astype(np.float32)
    root_positions_np = np.asarray(single["root_positions"]).astype(np.float32)

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
        output_name=getattr(sample, "output_name", None),
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
        all_samples = sampler.generate_random_specs(args.num_total, method=method, rerank=config.global_.rerank)

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

            _generate_batch(
                config=config,
                samples=samples,
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
