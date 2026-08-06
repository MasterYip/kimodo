#!/usr/bin/env python3
"""Generate a preflighted native Kimodo prompt manifest without constraints."""

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
import yaml

from kimodo import load_model
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.model.registry import MODEL_NAMES
from kimodo.tools import seed_everything


OFFICIAL_URL = "https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html"
REQUIRED_TRUE = (
    "prompt_begins_a_person", "focused_behaviors_lte2", "medium_detail",
    "self_contained", "duration_lte_10s", "text_constraint_consistent",
    "constraint_types_lt_20_keyframes", "g1_postprocessing_not_assumed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def validate_manifest(path: Path, duration: float) -> tuple[dict, list[dict]]:
    manifest = yaml.safe_load(path.read_text())
    if manifest.get("official_url") != OFFICIAL_URL:
        raise ValueError("Manifest official_url mismatch")
    rows = []
    for index, item in enumerate(manifest.get("prompts", [])):
        checks = dict(item.get("best_practices_check", {}))
        failures = [key for key in REQUIRED_TRUE if checks.get(key) is not True]
        prompt = item.get("prompt", "")
        if not prompt.startswith("A person"):
            failures.append("actual_prompt_prefix")
        if duration > 10.0:
            failures.append("actual_duration")
        count = checks.get("focused_behavior_count")
        if not isinstance(count, int) or not 1 <= count <= 2:
            failures.append("focused_behavior_count")
        status = item.get("status")
        passed = not failures and status == "launch"
        rows.append({**item, "index": index, "failures": failures,
                     "best_practices_pass": passed})
        if status == "launch" and not passed:
            raise ValueError(f"Best Practices preflight failed: {item.get('family')}: {failures}")
        if status not in {"launch", "skip"}:
            raise ValueError(f"Invalid status: {status}")
    if not rows:
        raise ValueError("No prompt rows")
    return manifest, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="kimodo-g1-rp")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    args = parser.parse_args()
    manifest, preflight = validate_manifest(args.manifest, args.duration)
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output}")
    native_dir = args.output / "native"
    motions_dir = args.output / "motions"
    native_dir.mkdir(parents=True)
    motions_dir.mkdir()
    cfg = {"type": "separated", "text_weight": 2.0,
           "constraint_weight": 2.0, "tuned": False}
    (args.output / "best_practices_preflight.json").write_text(json.dumps({
        "official_url": OFFICIAL_URL, "manifest_sha256": sha256(args.manifest),
        "cfg": cfg, "rows": preflight}, indent=2) + "\n")

    started = time.time()
    model, resolved_model = load_model(
        args.model, device="cuda:0", default_family="Kimodo", return_resolved_name=True)
    fps = float(model.fps)
    frames = int(round(args.duration * fps))
    if abs(fps - 30.0) > 1e-6 or frames != 240:
        raise RuntimeError(f"Expected 30 FPS / 240 frames, got {fps} / {frames}")
    converter = MujocoQposConverter(model.skeleton)
    generated = []
    for item in preflight:
        if item["status"] != "launch":
            continue
        index = item["index"]
        prompt = item["prompt"]
        family = item["family"]
        seed = int(item["seed"])
        seed_everything(seed)
        sample_name = f"p{index:02d}_{family}_s{seed}"
        sample_dir = motions_dir / sample_name
        sample_dir.mkdir()
        sample_started = time.time()
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
            raise RuntimeError(f"Invalid qpos: {sample_name}: {qpos.shape}")
        motion_path = sample_dir / "motion.npz"
        np.savez(motion_path, qpos=qpos, fps=np.asarray(fps, dtype=np.float32))
        row = {
            "index": index, "direction": item["direction"], "prompt_family": family,
            "prompt": prompt, "seed": seed, "model_resolved": resolved_model,
            "model_repo_id": MODEL_NAMES[resolved_model], "fps": fps, "frames": frames,
            "duration_seconds": frames / fps, "diffusion_steps": args.diffusion_steps,
            "conditioning": "text_only", "constraint_argument_passed": False,
            "constraint_objects_constructed": 0, "post_processing": False,
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
        (sample_dir / "metadata.json").write_text(json.dumps(row, indent=2) + "\n")
        generated.append(row)
        print(json.dumps({"completed": sample_name, "elapsed_seconds": elapsed}), flush=True)
    with (args.output / "generation_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(generated[0]))
        writer.writeheader()
        writer.writerows(generated)
    provenance = {
        "task": "DATA-KIMODO-LOCOMOTION-YAML-004", "experiment": manifest["experiment"],
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
        "post_processing": False, "generated_count": len(generated),
        "skipped_count": sum(row["status"] == "skip" for row in preflight),
        "cfg": cfg, "started_unix": started, "finished_unix": time.time(),
        "generator_sha256": sha256(Path(__file__)),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.output / "checksums.sha256").open("w") as stream:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                stream.write(f"{sha256(path)}  {path.relative_to(args.output)}\n")


if __name__ == "__main__":
    main()
