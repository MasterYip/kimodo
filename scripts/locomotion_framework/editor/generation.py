"""In-process generation pipeline for the locomotion editor.

Reuses the same sampler, constraint builder, prompts, and model API as the
batch orchestrator, but runs synchronously in-process so we can track
progress and return tensors directly for 3D visualisation.
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import yaml

from locomotion_framework.config import LocomotionConfig, VelRange, load_config
from locomotion_framework.sampler import MotionSampler, SampledMotion
from locomotion_framework.constraints import build_constraints_json
from locomotion_framework.prompts import build_motion_prompt
from locomotion_framework.editor.serializers import config_to_yaml_str


def _ensure_repo_path():
    """Ensure the kimodo repo root is importable."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def generate_batch(
    *,
    config: LocomotionConfig,
    output_base: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    stop_event: Optional[threading.Event] = None,
    device: str = "cuda:0",
    return_tensors: bool = True,
) -> dict[str, Any]:
    """Run the full locomotion generation pipeline in-process.

    Args:
        config: Parsed LocomotionConfig.
        output_base: Base directory for saved outputs.
        progress_callback: Called as ``callback(type_name, done, total)``
            for per-type progress reporting.
        stop_event: If set, generation is aborted at the next checkpoint.
        device: CUDA device string.
        return_tensors: If True, include raw model output tensors in the
            returned dict for 3D visualisation.

    Returns:
        Dict with keys:
            - ``samples``: flat list of all SampledMotion objects
            - ``results``: list of per-sample dicts with ``{motion_type, name,
              posed_joints, global_rot_mats, foot_contacts, path}``
            - ``elapsed_s``: total wall time
    """
    _ensure_repo_path()

    from kimodo import load_model

    sampler = MotionSampler(config, seed=config.global_.seed)
    method = config.global_.sampling_method
    batch_specs = sampler.generate_batch_specs(method=method, rerank=config.global_.rerank)

    total_motions = sum(len(s) for s in batch_specs.values())
    output_path = Path(output_base)

    # ── Load model once ────────────────────────────────────────────
    model, resolved_name = load_model(
        config.global_.model,
        device=device,
        return_resolved_name=True,
    )

    fps = config.global_.fps
    seed_val = config.global_.seed if config.global_.seed is not None else 0
    all_samples: list[SampledMotion] = []
    all_results: list[dict] = []
    global_idx = 0

    t_start = time.time()

    for type_name, samples in batch_specs.items():
        if stop_event and stop_event.is_set():
            break

        n = len(samples)
        if progress_callback:
            progress_callback(type_name, 0, n)

        # ── Build per-sample prompts / frames / constraints ────────
        per_prompts: list[str] = []
        per_frames: list[int] = []
        per_constraints_raw: list[list[dict]] = []

        for s in samples:
            per_prompts.append(s.prompt)
            per_frames.append(int(round(s.duration * fps)))
            per_constraints_raw.append(build_constraints_json(s, fps=fps))

        # Convert to Kimodo constraint objects
        from kimodo.constraints import load_constraints_lst

        per_kimodo_constraints: list[list] = []
        for raw_lst in per_constraints_raw:
            if raw_lst:
                per_kimodo_constraints.append(
                    load_constraints_lst(raw_lst, model.skeleton, device=device)
                )
            else:
                per_kimodo_constraints.append([])

        # ── Generate ───────────────────────────────────────────────
        output = model(
            per_prompts,
            per_frames,
            constraint_lst=per_kimodo_constraints,
            num_denoising_steps=samples[0].diffusion_steps,
            return_numpy=True,
        )

        # ── Extract per-sample results ─────────────────────────────
        for i, sample in enumerate(samples):
            single = {}
            for k, v in output.items():
                arr = np.asarray(v)
                if arr.ndim >= 1 and arr.shape[0] == n:
                    single[k] = arr[i].copy()
                else:
                    single[k] = arr

            # Trim to actual duration
            actual_frames = int(round(sample.duration * fps))
            for k in list(single.keys()):
                arr = single[k]
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.ndim >= 1 and arr.shape[0] > actual_frames:
                    single[k] = arr[:actual_frames].copy()

            # Save to disk
            saved_path = _save_sample(
                single=single,
                model=model,
                resolved_name=resolved_name,
                fps=fps,
                sample=sample,
                sample_idx=global_idx,
                seed=seed_val,
                output_base=output_path,
                preset=config.global_.export_preset,
                device=device,
            )

            result_entry = {
                "motion_type": sample.motion_type,
                "sample_idx": i,
                "global_idx": global_idx,
                "name": saved_path.name if saved_path else f"{sample.motion_type}_{global_idx:04d}",
                "prompt": sample.prompt,
                "duration": sample.duration,
                "vel": sample.vel,
                "torso_height": sample.torso_height,
                "path": str(saved_path) if saved_path else None,
            }

            if return_tensors:
                result_entry["posed_joints"] = np.asarray(single.get("posed_joints"))
                result_entry["global_rot_mats"] = np.asarray(single.get("global_rot_mats"))
                result_entry["foot_contacts"] = np.asarray(single.get("foot_contacts", None))

            all_results.append(result_entry)
            all_samples.append(sample)
            global_idx += 1

            if progress_callback:
                progress_callback(type_name, i + 1, n)

    # ── Save manifest ──────────────────────────────────────────────
    from locomotion_framework.orchestrator import (
        _save_manifest as save_manifest,
    )
    save_manifest(output_path / "manifest.csv", all_samples,
                  preset=config.global_.export_preset, seed=seed_val)

    # ── Save YAML snapshot ─────────────────────────────────────────
    yaml_str = config_to_yaml_str(config)
    (output_path / "config.yaml").write_text(yaml_str)

    # ── Cleanup ────────────────────────────────────────────────────
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    elapsed = time.time() - t_start

    return {
        "samples": all_samples,
        "results": all_results,
        "elapsed_s": elapsed,
        "total_motions": total_motions,
    }


def _save_sample(
    single: dict,
    model,
    resolved_name: str,
    fps: float,
    sample: SampledMotion,
    sample_idx: int,
    seed: int,
    output_base: Path,
    preset: str,
    device: str,
) -> Optional[Path]:
    """Save one sample to disk using the configured export preset."""
    import numpy as np

    if preset == "rltracker":
        from locomotion_framework.export_presets.rltracker import RLTrackerExporter

        posed_joints = np.asarray(single["posed_joints"]).astype(np.float32)
        global_rot_mats = np.asarray(single["global_rot_mats"]).astype(np.float32)
        local_rot_mats = np.asarray(single["local_rot_mats"]).astype(np.float32)
        root_positions = np.asarray(single["root_positions"]).astype(np.float32)

        exporter = RLTrackerExporter()
        return exporter.export(
            posed_joints=posed_joints,
            global_rot_mats=global_rot_mats,
            local_rot_mats=local_rot_mats,
            root_positions=root_positions,
            fps=fps,
            sample_idx=sample_idx,
            motion_type=sample.motion_type,
            vel=sample.vel,
            torso_height=sample.torso_height,
            style=sample.style,
            seed=seed,
            output_base=output_base,
        )
    else:
        from kimodo.exports.motion_io import save_kimodo_npz

        out_dir = output_base / sample.motion_type
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{sample.motion_type}_{sample_idx:04d}"
        npz_path = out_dir / f"{stem}.npz"
        save_kimodo_npz(str(npz_path), single)
        return npz_path
