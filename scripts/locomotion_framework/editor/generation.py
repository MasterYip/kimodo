"""In-process batch generation pipeline for the Loco Editor.

Uses the same model API pattern as orchestrator.py:
  - Per-sample prompts, num_frames, and constraints via list API
  - Each motion type batch is generated in ONE model call
  - Outputs are trimmed to actual duration (model pads to max in batch)
"""

import time
from typing import Any, Callable, Optional

import numpy as np


def generate_batch(
    config: Any,  # LocomotionConfig
    output_base: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    stop_event: Optional[Any] = None,
    device: str = "cuda:0",
    return_tensors: bool = False,
) -> dict:
    """Run the locomotion generation pipeline in-process.

    Uses the exact same pattern as orchestrator._generate_batch():
      - sampler.generate_batch_specs() → per-type SampledMotion lists
      - build_constraints_json() → raw constraint dicts
      - load_constraints_lst() → Kimodo constraint objects
      - model(prompts, num_frames, constraint_lst=...) → batch output
      - Trim batch-padding frames for each sample

    Args:
        config: LocomotionConfig with global settings and motion type specs.
        output_base: Base output directory (unused in editor mode).
        progress_callback: Called as progress_callback(type_name, done, total).
        stop_event: threading.Event for cancellation.
        device: CUDA device string.
        return_tensors: If True, include raw tensors in results for 3D viz.

    Returns:
        dict with keys: "results" (list of sample dicts), "total_motions",
        "elapsed_s", "output_dir".
    """
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.tools import seed_everything

    from locomotion_framework.constraints import build_constraints_json
    from locomotion_framework.sampler import MotionSampler

    t0 = time.time()

    # Load model once (pattern from orchestrator._generate_batch)
    print(f"Loading model {config.global_.model} ...")
    model, resolved_name = load_model(
        config.global_.model, device=device, return_resolved_name=True
    )

    # Seed
    seed_val = config.global_.seed if config.global_.seed is not None else 42
    seed_everything(seed_val)

    # Sampler — generate all specs upfront
    sampler = MotionSampler(config, seed=seed_val)
    method = config.global_.sampling_method
    rerank = config.global_.rerank
    batch_specs = sampler.generate_batch_specs(method=method, rerank=rerank)
    # batch_specs: dict[str, list[SampledMotion]]

    fps = config.global_.fps
    results = []
    total_motions = sum(len(s) for s in batch_specs.values())

    for type_name, samples in batch_specs.items():
        if stop_event is not None and stop_event.is_set():
            break

        n = len(samples)
        if n == 0:
            continue

        if progress_callback is not None:
            progress_callback(type_name, 0, n)

        try:
            # ── Build per-sample lists (same pattern as orchestrator) ──
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

            # Check stop before GPU call
            if stop_event is not None and stop_event.is_set():
                break

            # ── Generate batch (one call for all samples of this type) ──
            diffusion_steps = samples[0].diffusion_steps
            output = model(
                per_prompts,
                per_frames,
                constraint_lst=per_kimodo_constraints,
                num_denoising_steps=diffusion_steps,
                return_numpy=True,
            )

            # ── Extract per-sample outputs (trim to actual duration) ──
            for i, sample in enumerate(samples):
                if stop_event is not None and stop_event.is_set():
                    break

                actual_frames = int(round(sample.duration * fps))

                # Extract and trim i-th sample
                posed_joints = output.get("posed_joints")
                global_rot_mats = output.get("global_rot_mats")
                foot_contacts = output.get("foot_contacts")

                def _to_numpy(x):
                    """Safely convert tensor or numpy to numpy on CPU."""
                    if x is None:
                        return None
                    if isinstance(x, np.ndarray):
                        return x
                    if hasattr(x, "cpu"):  # torch tensor (possibly on GPU)
                        return x.detach().cpu().numpy()
                    return np.asarray(x)

                pj_i = _to_numpy(posed_joints[i]) if posed_joints is not None else None
                grm_i = _to_numpy(global_rot_mats[i]) if global_rot_mats is not None else None
                fc_i = _to_numpy(foot_contacts[i]) if foot_contacts is not None else None

                # Trim to actual duration
                if pj_i is not None and pj_i.shape[0] > actual_frames:
                    pj_i = pj_i[:actual_frames].copy()
                if grm_i is not None and grm_i.shape[0] > actual_frames:
                    grm_i = grm_i[:actual_frames].copy()
                if fc_i is not None and fc_i is not None and fc_i.shape[0] > actual_frames:
                    fc_i = fc_i[:actual_frames].copy()

                sample_dict = {
                    "motion_type": type_name,
                    "sample_idx": i,
                    "name": f"{type_name}_{i:04d}",
                    "posed_joints": pj_i,
                    "global_rot_mats": grm_i,
                    "foot_contacts": fc_i,
                }
                results.append(sample_dict)

            if progress_callback is not None:
                progress_callback(type_name, n, n)

        except Exception as e:
            import traceback
            print(f"Error generating {type_name}: {e}")
            traceback.print_exc()
            if progress_callback is not None:
                progress_callback(type_name, 0, n)

    elapsed = time.time() - t0

    return {
        "results": results,
        "total_motions": len(results),
        "elapsed_s": elapsed,
        "output_dir": output_base,
    }
