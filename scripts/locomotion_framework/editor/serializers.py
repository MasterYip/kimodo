"""YAML ↔ LocomotionConfig serialization and config summary."""

import io
from pathlib import Path

import yaml

from locomotion_framework.config import (
    GlobalConfig,
    LocomotionConfig,
    MotionSpec,
    VelRange,
    load_config,
)


# ── Default YAML ───────────────────────────────────────────────────

def make_default_yaml() -> str:
    """Return a minimal default YAML config string."""
    return """global:
  model: kimodo-g1-rp
  diffusion_steps: 100
  seed: 42
  output_dir: outputs/loco_editor
  sampling_method: uniform
  export_preset: rltracker
  rerank: ""

motion_types:
  walk:
    description: "a robot walks"
    duration: [3.0, 8.0]
    vel_cmd:
      vx: [0.2, 1.0]
      vy: [-0.3, 0.3]
      wz: [-0.5, 0.5]
    torso_height: [0.70, 0.85]
    styles:
      - "normally"
      - "at a steady pace"
    weight: 0.5
    num_samples: 10
  stand:
    description: "a robot stands still"
    duration: [2.0, 6.0]
    vel_cmd:
      vx: [0.0, 0.0]
      vy: [0.0, 0.0]
      wz: [0.0, 0.0]
    torso_height: [0.65, 0.85]
    styles:
      - "upright"
      - "relaxed"
    weight: 0.5
    num_samples: 5
"""


# ── Config → YAML ──────────────────────────────────────────────────

def config_to_yaml_str(config: LocomotionConfig) -> str:
    """Serialize a LocomotionConfig to a YAML string."""
    buf = io.StringIO()

    # Global section
    g = config.global_
    buf.write("global:\n")
    buf.write(f"  model: {g.model}\n")
    buf.write(f"  diffusion_steps: {g.diffusion_steps}\n")
    if g.seed is not None:
        buf.write(f"  seed: {g.seed}\n")
    buf.write(f"  output_dir: {g.output_dir}\n")
    buf.write(f"  sampling_method: {g.sampling_method}\n")
    buf.write(f"  export_preset: {g.export_preset}\n")
    buf.write(f"  rerank: \"{g.rerank}\"\n")
    buf.write("\n")

    # Motion types
    buf.write("motion_types:\n")
    for name, spec in config.motion_types.items():
        buf.write(f"  {name}:\n")
        buf.write(f"    description: \"{spec.description}\"\n")
        buf.write(f"    duration: [{spec.duration_range[0]}, {spec.duration_range[1]}]\n")
        buf.write("    vel_cmd:\n")
        for key in ("vx", "vy", "wz"):
            if key in spec.vel_cmd:
                vr = spec.vel_cmd[key]
                buf.write(f"      {key}: [{vr.min}, {vr.max}]\n")
        buf.write(f"    torso_height: [{spec.torso_height_range[0]}, {spec.torso_height_range[1]}]\n")
        if spec.styles:
            buf.write("    styles:\n")
            for s in spec.styles:
                buf.write(f"      - \"{s}\"\n")
        buf.write(f"    weight: {spec.weight}\n")
        buf.write(f"    num_samples: {spec.num_samples}\n")

    return buf.getvalue()


# ── YAML → Config ──────────────────────────────────────────────────

def yaml_str_to_config(yaml_str: str) -> LocomotionConfig:
    """Deserialize a YAML string into a LocomotionConfig."""
    raw = yaml.safe_load(yaml_str)

    global_raw = raw.get("global", {})
    global_config = GlobalConfig(
        model=global_raw.get("model", "kimodo-g1-rp"),
        diffusion_steps=global_raw.get("diffusion_steps", 100),
        seed=global_raw.get("seed"),
        output_dir=global_raw.get("output_dir", "outputs/loco_editor"),
        gpu=global_raw.get("gpu", 0),
        fps=global_raw.get("fps", 30),
        sampling_method=global_raw.get("sampling_method", "uniform"),
        export_preset=global_raw.get("export_preset", "kimodo"),
        rerank=global_raw.get("rerank", ""),
    )

    motion_types = {}
    for name, spec_raw in raw.get("motion_types", {}).items():
        vel_cmd = {}
        for key in ("vx", "vy", "wz"):
            if key in spec_raw.get("vel_cmd", {}):
                vel_cmd[key] = VelRange.from_list(spec_raw["vel_cmd"][key])

        spec = MotionSpec(
            name=name,
            description=spec_raw.get("description", ""),
            duration_range=tuple(spec_raw.get("duration", [3.0, 8.0])),
            vel_cmd=vel_cmd,
            torso_height_range=tuple(spec_raw.get("torso_height", [0.70, 0.85])),
            styles=spec_raw.get("styles", []),
            weight=spec_raw.get("weight", 1.0),
            num_samples=spec_raw.get("num_samples", 10),
            diffusion_steps=spec_raw.get(
                "diffusion_steps", global_config.diffusion_steps
            ),
        )
        motion_types[name] = spec

    return LocomotionConfig(global_=global_config, motion_types=motion_types)


# ── Helpers ─────────────────────────────────────────────────────────

def compute_total_motions(config: LocomotionConfig) -> int:
    """Sum num_samples across all motion types."""
    return sum(spec.num_samples for spec in config.motion_types.values())


def config_summary_md(config: LocomotionConfig) -> str:
    """Render a markdown summary table of the current config."""
    lines = [
        "| Motion Type | Description | Duration | vx | vy | wz | Torso H | Styles | # |",
        "|------------|-------------|----------|----|----|----|---------|--------|---|",
    ]
    for name, spec in config.motion_types.items():
        vx = spec.vel_cmd.get("vx")
        vy = spec.vel_cmd.get("vy")
        wz = spec.vel_cmd.get("wz")
        vx_s = f"[{vx.min:.1f}, {vx.max:.1f}]" if vx else "—"
        vy_s = f"[{vy.min:.1f}, {vy.max:.1f}]" if vy else "—"
        wz_s = f"[{wz.min:.1f}, {wz.max:.1f}]" if wz else "—"
        lines.append(
            f"| **{name}** | {spec.description[:20]} | "
            f"[{spec.duration_range[0]:.0f}, {spec.duration_range[1]:.0f}]s | "
            f"{vx_s} | {vy_s} | {wz_s} | "
            f"[{spec.torso_height_range[0]:.2f}, {spec.torso_height_range[1]:.2f}] | "
            f"{', '.join(spec.styles[:2])} | {spec.num_samples} |"
        )
    total = compute_total_motions(config)
    lines.append(f"\n**Total motions: {total}**")
    return "\n".join(lines)
