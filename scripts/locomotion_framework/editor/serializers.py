"""Bidirectional conversion between viser GUI values and LocomotionConfig."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import yaml

from locomotion_framework.config import (
    GlobalConfig,
    LocomotionConfig,
    MotionSpec,
    VelRange,
)


# ── Config → YAML ──────────────────────────────────────────────────


def config_to_yaml_str(config: LocomotionConfig) -> str:
    """Serialize a LocomotionConfig to a YAML string.

    Uses custom representers for clean, readable output matching the
    hand-authored config style.
    """
    def _vel_range_representer(dumper, vr: VelRange):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", [vr.min, vr.max], flow_style=True
        )

    yaml.add_representer(VelRange, _vel_range_representer)

    # Build dict matching the config file format
    global_dict = {
        "model": config.global_.model,
        "diffusion_steps": config.global_.diffusion_steps,
        "seed": config.global_.seed,
        "output_dir": config.global_.output_dir,
        "sampling_method": config.global_.sampling_method,
        "export_preset": config.global_.export_preset,
    }
    if config.global_.rerank:
        global_dict["rerank"] = config.global_.rerank
    if config.global_.fps != 30:
        global_dict["fps"] = config.global_.fps

    motion_dict = {}
    for name, spec in config.motion_types.items():
        entry: dict[str, Any] = {
            "description": spec.description,
            "duration": list(spec.duration_range),
            "vel_cmd": {},
            "torso_height": list(spec.torso_height_range),
        }
        for key in ("vx", "vy", "wz"):
            if key in spec.vel_cmd:
                entry["vel_cmd"][key] = spec.vel_cmd[key]
        if spec.styles:
            entry["styles"] = spec.styles
        if spec.weight != 1.0:
            entry["weight"] = spec.weight
        if spec.num_samples != 10:
            entry["num_samples"] = spec.num_samples
        if spec.diffusion_steps != config.global_.diffusion_steps:
            entry["diffusion_steps"] = spec.diffusion_steps
        motion_dict[name] = entry

    out = {"global": global_dict, "motion_types": motion_dict}

    return yaml.dump(
        out,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )


# ── YAML → Config ──────────────────────────────────────────────────


def yaml_str_to_config(yaml_str: str) -> LocomotionConfig:
    """Parse a YAML string into a LocomotionConfig.

    Delegates to the existing ``load_config`` but via an in-memory file.
    """
    from locomotion_framework.config import load_config as _load_config_file

    # Write to a temp file so we can reuse load_config (which reads a path)
    import tempfile
    import os

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="loco_editor_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(yaml_str)
        return _load_config_file(tmp_path)
    finally:
        os.unlink(tmp_path)


# ── Config → default widgets (populate viser GUI) ───────────────────


def make_default_yaml() -> str:
    """Return a minimal example YAML string to seed the editor."""
    return """# Locomotion Config — edit this panel, or load from a file below
global:
  model: kimodo-g1-rp
  diffusion_steps: 100
  seed: 42
  output_dir: outputs/loco_editor
  sampling_method: lhs
  rerank: speed
  export_preset: rltracker

motion_types:
  walk:
    description: "a person walks"
    duration: [5.0, 8.0]
    vel_cmd:
      vx: [-0.60, 0.80]
      vy: [-0.50, 0.50]
      wz: [-0.80, 0.80]
    torso_height: [0.70, 0.85]
    styles:
      - "normally"
      - "at a steady pace"
    weight: 0.50
    num_samples: 50

  run:
    description: "a person runs"
    duration: [5.0, 8.0]
    vel_cmd:
      vx: [-1.00, 2.00]
      vy: [-1.00, 1.00]
      wz: [-0.50, 0.50]
    torso_height: [0.65, 0.80]
    styles:
      - "normally"
      - "steadily"
    weight: 0.20
    num_samples: 50

  stand:
    description: "a person stands"
    duration: [5.0, 8.0]
    vel_cmd:
      vx: [-0.10, 0.10]
      vy: [-0.10, 0.10]
      wz: [-0.30, 0.30]
    torso_height: [0.80, 0.90]
    styles:
      - "steadily"
      - "normally"
    weight: 0.30
    num_samples: 50

  squat_move:
    description: "a person squats and moves"
    duration: [5.0, 8.0]
    vel_cmd:
      vx: [-0.20, 0.30]
      vy: [-0.10, 0.10]
      wz: [-0.20, 0.20]
    torso_height: [0.55, 0.70]
    styles:
      - "normally"
      - "while staying low"
    weight: 0.15
    num_samples: 50
"""


def compute_total_motions(config: LocomotionConfig) -> int:
    """Return total number of motions across all types."""
    return sum(s.num_samples for s in config.motion_types.values())


def config_summary_md(config: LocomotionConfig) -> str:
    """Generate a markdown summary of the current config."""
    total = compute_total_motions(config)
    lines = [
        "### Config Summary",
        "",
        f"**Model:** {config.global_.model} | **Seed:** {config.global_.seed}",
        f"**Sampling:** {config.global_.sampling_method} | **Rerank:** {config.global_.rerank or 'none'}",
        f"**Export:** {config.global_.export_preset} | **FPS:** {config.global_.fps}",
        f"**Output:** `{config.global_.output_dir}`",
        f"**Total motions:** {total} across {len(config.motion_types)} types",
        "",
        "| Type | Description | Duration | vx | vy | wz | Torso | N |",
        "|------|-------------|----------|----|----|----|-------|---|",
    ]
    for name, s in config.motion_types.items():
        def _vr(vr: VelRange | None) -> str:
            if vr is None:
                return "—"
            return f"[{vr.min:.2f}, {vr.max:.2f}]"

        desc = s.description[:30] + ("…" if len(s.description) > 30 else "")
        dur = f"[{s.duration_range[0]:.1f}, {s.duration_range[1]:.1f}]"
        torso = f"[{s.torso_height_range[0]:.2f}, {s.torso_height_range[1]:.2f}]"
        lines.append(
            f"| `{name}` | {desc} | {dur} | {_vr(s.vel_cmd.get('vx'))} | "
            f"{_vr(s.vel_cmd.get('vy'))} | {_vr(s.vel_cmd.get('wz'))} | "
            f"{torso} | {s.num_samples} |"
        )

    return "\n".join(lines)
