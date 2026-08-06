#!/usr/bin/env python3
"""Generate the bounded documented-domain Kimodo text-only prompt matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from kimodo import load_model
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.registry import MODEL_NAMES
from kimodo.tools import seed_everything


# These are project-authored experiments, not prompts supplied by NVIDIA.
OFFICIAL_URL = "https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html"
PROMPTS = (
    ("forward", "stealth_crouch_forward", "A person walks forward stealthily in a low crouch."),
    ("right", "stealth_crouch_right", "A person moves sideways to the right stealthily in a low crouch."),
    ("left", "stealth_crouch_left", "A person moves sideways to the left stealthily in a low crouch."),
    ("forward", "combat_forward", "A person advances forward in a low videogame combat stance."),
    ("right", "combat_strafe_right", "A person strafes to the right in a low videogame combat stance."),
    ("left", "combat_strafe_left", "A person strafes to the left in a low videogame combat stance."),
)
SKIPPED_PROMPTS = (
    ("unspecified", "stealth_minimal", "A person moves stealthily.", "too vague; not medium detail"),
    ("unspecified", "combat_minimal", "A person performs videogame combat movement.", "too vague; not medium detail"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--model", default="kimodo-g1-rp")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    args = parser.parse_args()
    preflight = []
    for direction, family, prompt in PROMPTS:
        checks = {
            "prompt_begins_a_person": prompt.startswith("A person"),
            "focused_behaviors_lte2": True,
            "medium_detail": True,
            "self_contained": True,
            "duration_lte_10s": args.duration <= 10.0,
            "text_constraint_consistent": True,
            "constraint_types_lt_20_keyframes": True,
            "g1_postprocessing_not_assumed": True,
        }
        preflight.append({"direction": direction, "prompt_family": family, "prompt": prompt,
                          "status": "launch", "checks": checks,
                          "best_practices_pass": all(checks.values())})
    for direction, family, prompt, reason in SKIPPED_PROMPTS:
        preflight.append({"direction": direction, "prompt_family": family, "prompt": prompt,
                          "status": "skip", "reason": reason,
                          "checks": {"prompt_begins_a_person": True, "medium_detail": False},
                          "best_practices_pass": False})
    if any(not row["best_practices_pass"] for row in preflight if row["status"] == "launch"):
        raise RuntimeError("Best Practices preflight failed")
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output}")
    native_dir = args.output / "native"
    motions_dir = args.output / "motions"
    native_dir.mkdir(parents=True)
    motions_dir.mkdir()
    (args.output / "best_practices_preflight.json").write_text(
        json.dumps({"official_url": OFFICIAL_URL, "cfg": {"type": "separated",
                    "text_weight": 2.0, "constraint_weight": 2.0, "tuned": False},
                    "rows": preflight}, indent=2) + "\n")

    started = time.time()
    model, resolved_model = load_model(
        args.model, device="cuda:0", default_family="Kimodo", return_resolved_name=True)
    fps = float(model.fps)
    frames = int(round(args.duration * fps))
    if abs(fps - 30.0) > 1e-6 or frames != 240:
        raise RuntimeError(f"Expected 30 FPS / 240 frames, got {fps} / {frames}")
    converter = MujocoQposConverter(model.skeleton)

    rows = []
    for index, (direction, family, prompt) in enumerate(PROMPTS):
        seed_everything(args.seed)
        sample_name = f"p{index:02d}_{family}_s{args.seed}"
        sample_dir = motions_dir / sample_name
        sample_dir.mkdir()
        sample_started = time.time()
        # Native basic API: constraint_lst is intentionally omitted.
        output = model(prompt, frames, num_denoising_steps=args.diffusion_steps,
                       num_samples=1, post_processing=False, return_numpy=True,
                       cfg_type="separated", cfg_weight=[2.0, 2.0])
        elapsed = time.time() - sample_started
        native_path = native_dir / f"{sample_name}.npz"
        save_kimodo_npz(str(native_path), output)
        qpos = np.asarray(converter.dict_to_qpos(output, device="cuda:0"), dtype=np.float32)
        if qpos.ndim == 3:
            qpos = qpos[0]
        if qpos.shape != (240, 36) or not np.isfinite(qpos).all():
            raise RuntimeError(f"Invalid converted qpos for {sample_name}: {qpos.shape}")
        motion_path = sample_dir / "motion.npz"
        np.savez(motion_path, qpos=qpos, fps=np.asarray(fps, dtype=np.float32))
        metadata = {
            "index": index, "direction": direction, "prompt_family": family,
            "prompt": prompt, "seed": args.seed, "model_requested": args.model,
            "model_resolved": resolved_model, "model_repo_id": MODEL_NAMES[resolved_model],
            "fps": fps, "frames": frames, "duration_seconds": frames / fps,
            "diffusion_steps": args.diffusion_steps, "post_processing": False,
            "conditioning": "text_only", "constraint_argument_passed": False,
            "constraint_objects_constructed": 0,
            "best_practices_pass": True, "best_practices_status": "launch",
            "best_practices_official_url": OFFICIAL_URL,
            "cfg_type": "separated", "cfg_text_weight": 2.0,
            "cfg_constraint_weight": 2.0, "cfg_tuned": False,
            "g1_postprocessing_not_assumed": True,
            "native_path": str(native_path.relative_to(args.output)),
            "motion_path": str(motion_path.relative_to(args.output)),
            "native_sha256": sha256(native_path), "motion_sha256": sha256(motion_path),
            "elapsed_seconds": elapsed,
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append(metadata)
        print(json.dumps({"completed": sample_name, "elapsed_seconds": elapsed}), flush=True)

    with (args.output / "generation_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "task": "DATA-KIMODO-LOCOMOTION-YAML-004",
        "experiment": "documented_domain_textonly_20260806",
        "official_best_practices_url": OFFICIAL_URL,
        "prompt_authorship": "project-authored; not NVIDIA example prompts",
        "generation_host_declared": os.environ.get("GENERATION_SERVER_NAME"),
        "hostname": platform.node(), "repo_root": str(Path.cwd()),
        "repo_commit": command_output(["git", "rev-parse", "HEAD"]),
        "repo_branch": command_output(["git", "branch", "--show-current"]),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_resolved": resolved_model, "model_repo_id": MODEL_NAMES[resolved_model],
        "fps": fps, "frames": frames, "duration_seconds": frames / fps,
        "diffusion_steps": args.diffusion_steps, "conditioning": "text_only",
        "constraint_argument_passed": False, "constraint_objects_constructed": 0,
        "post_processing": False, "prompt_count": len(PROMPTS),
        "skipped_prompt_count": len(SKIPPED_PROMPTS), "seed": args.seed,
        "cfg": {"type": "separated", "text_weight": 2.0,
                "constraint_weight": 2.0, "tuned": False},
        "started_unix": started, "finished_unix": time.time(),
        "generator_sha256": sha256(Path(__file__)),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.output / "checksums.sha256").open("w") as stream:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                stream.write(f"{sha256(path)}  {path.relative_to(args.output)}\n")


if __name__ == "__main__":
    main()
