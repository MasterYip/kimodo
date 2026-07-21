"""Motion specification dataclasses and YAML config loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VelRange:
    """Uniform distribution range for a velocity component."""
    min: float
    max: float

    def sample(self, rng) -> float:
        return rng.uniform(self.min, self.max)

    @classmethod
    def from_list(cls, lst: list[float]) -> "VelRange":
        return cls(min=float(lst[0]), max=float(lst[1]))


@dataclass
class MotionSpec:
    """Defines a distribution over one motion type (walk, run, stand, etc.)."""
    name: str
    description: str                       # base text prompt template
    duration_range: tuple[float, float]    # (min, max) seconds
    vel_cmd: dict[str, VelRange]           # {"vx": ..., "vy": ..., "wz": ...}
    torso_height_range: tuple[float, float]  # (min, max) normalized
    styles: list[str]                      # ["casually", "briskly", ...]
    weight: float = 1.0                    # relative sampling probability
    num_samples: int = 10                  # how many motions from this type
    diffusion_steps: int = 100


@dataclass
class GlobalConfig:
    """Global generation settings."""
    model: str = "kimodo-g1-rp"
    diffusion_steps: int = 100
    seed: Optional[int] = 42
    output_dir: str = "outputs/locomotion"
    gpu: int = 0
    fps: int = 30
    sampling_method: str = "uniform"  # "uniform" or "lhs"
    export_preset: str = "kimodo"     # "kimodo" (default NPZ+CSV) or "rltracker"
    generate_margin: float = 0.0      # DEPRECATED: demo method uses exact duration, no margin


@dataclass
class LocomotionConfig:
    """Top-level config with global settings and motion type definitions."""
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    motion_types: dict[str, MotionSpec] = field(default_factory=dict)


def _parse_vel_cmd(raw: dict) -> dict[str, VelRange]:
    """Parse raw velocity command dict from YAML into VelRange objects."""
    parsed = {}
    for key in ("vx", "vy", "wz"):
        if key in raw:
            parsed[key] = VelRange.from_list(raw[key])
    return parsed


def load_config(path: str | Path) -> LocomotionConfig:
    """Load a locomotion config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        LocomotionConfig with parsed motion specs.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    global_raw = raw.get("global", {})
    global_config = GlobalConfig(
        model=global_raw.get("model", "kimodo-g1-rp"),
        diffusion_steps=global_raw.get("diffusion_steps", 100),
        seed=global_raw.get("seed"),
        output_dir=global_raw.get("output_dir", "outputs/locomotion"),
        gpu=global_raw.get("gpu", 0),
        fps=global_raw.get("fps", 30),
        sampling_method=global_raw.get("sampling_method", "uniform"),
        export_preset=global_raw.get("export_preset", "kimodo"),
        generate_margin=global_raw.get("generate_margin", 2.0),
    )

    motion_types = {}
    for name, spec_raw in raw.get("motion_types", {}).items():
        spec = MotionSpec(
            name=name,
            description=spec_raw["description"],
            duration_range=tuple(spec_raw["duration"]),
            vel_cmd=_parse_vel_cmd(spec_raw.get("vel_cmd", {})),
            torso_height_range=tuple(spec_raw["torso_height"]),
            styles=spec_raw.get("styles", []),
            weight=spec_raw.get("weight", 1.0),
            num_samples=spec_raw.get("num_samples", 10),
            diffusion_steps=spec_raw.get("diffusion_steps", global_config.diffusion_steps),
        )
        motion_types[name] = spec

    return LocomotionConfig(global_=global_config, motion_types=motion_types)
