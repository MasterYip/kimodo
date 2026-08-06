"""Motion specification dataclasses and YAML config loading."""

import math
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
        return float(rng.uniform(self.min, self.max))

    @classmethod
    def from_list(cls, lst: list[float]) -> "VelRange":
        if not isinstance(lst, (list, tuple)) or len(lst) != 2:
            raise ValueError("velocity ranges must be two-element [min, max] lists")
        low, high = float(lst[0]), float(lst[1])
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("velocity range bounds must be finite")
        if low > high:
            raise ValueError(f"velocity range min {low} exceeds max {high}")
        return cls(min=low, max=high)


@dataclass
class PolarVelRange:
    """Planar speed and direction ranges."""

    speed: VelRange
    direction_deg: VelRange

    def __post_init__(self) -> None:
        if self.speed.min < 0.0:
            raise ValueError("polar speed range must be non-negative")
        if self.direction_deg.max - self.direction_deg.min > 360.0:
            raise ValueError("polar direction range must span at most 360 degrees")


@dataclass
class MotionSpec:
    """Defines a distribution over one motion type (walk, run, stand, etc.)."""
    name: str
    description: str = ""                  # base text prompt template; empty = unconstrained
    prompt_override: Optional[str] = None   # exact final prompt; bypasses composition
    arm_swing: str = "none"                 # text intent: auto/sagittal/lateral/none
    duration_range: tuple[float, float] = (3.0, 8.0)  # (min, max) seconds
    vel_cmd: dict[str, VelRange] = field(default_factory=dict)  # {"vx": ..., "vy": ..., "wz": ...}
    torso_height_range: Optional[tuple[float, float]] = None  # (min, max) normalized; None = unconstrained
    styles: list[str] = field(default_factory=list)  # ["casually", "briskly", ...]
    polar_vel_cmd: Optional[PolarVelRange] = None
    weight: float = 1.0                    # relative sampling probability
    num_samples: int = 10                  # how many motions from this type
    diffusion_steps: int = 100


@dataclass
class Root2DConstraintConfig:
    """Controls Root2D injection without changing sampled command metadata."""
    enabled: bool = True
    stride: int = 1


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
    rerank: str = ""                  # sort samples before generation: "vx", "vy", "wz",
                                      #   "speed" (|v|), "torso", "duration", or "" (no sort)
    root2d_constraint: Root2DConstraintConfig = field(default_factory=Root2DConstraintConfig)


@dataclass
class LocomotionConfig:
    """Top-level config with global settings and motion type definitions."""
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    motion_types: dict[str, MotionSpec] = field(default_factory=dict)


def _parse_vel_cmd(raw: dict) -> dict[str, VelRange]:
    """Parse raw velocity command dict from YAML into VelRange objects."""
    if not isinstance(raw, dict):
        raise ValueError("vel_cmd must be a mapping")
    if "polar" in raw and any(key in raw for key in ("vx", "vy")):
        raise ValueError("vel_cmd.polar cannot be combined with Cartesian vx/vy")
    parsed = {}
    for key in ("vx", "vy", "wz"):
        if key in raw:
            parsed[key] = VelRange.from_list(raw[key])
    return parsed


def _parse_polar_vel_cmd(raw: dict) -> Optional[PolarVelRange]:
    """Parse optional polar planar velocity ranges."""
    polar_raw = raw.get("polar")
    if polar_raw is None:
        return None
    if not isinstance(polar_raw, dict):
        raise ValueError("vel_cmd.polar must be a mapping")
    missing = {"speed", "direction_deg"} - set(polar_raw)
    if missing:
        raise ValueError(
            "vel_cmd.polar is missing required field(s): " + ", ".join(sorted(missing))
        )
    unknown = set(polar_raw) - {"speed", "direction_deg"}
    if unknown:
        raise ValueError(
            "unknown vel_cmd.polar field(s): " + ", ".join(sorted(unknown))
        )
    return PolarVelRange(
        speed=VelRange.from_list(polar_raw["speed"]),
        direction_deg=VelRange.from_list(polar_raw["direction_deg"]),
    )


def _parse_root2d_constraint(raw: dict | None) -> Root2DConstraintConfig:
    """Parse and validate the additive Root2D settings."""
    if raw is None:
        return Root2DConstraintConfig()
    if not isinstance(raw, dict):
        raise ValueError("global.root2d_constraint must be a mapping")
    enabled = raw.get("enabled", True)
    stride = raw.get("stride", 1)
    if not isinstance(enabled, bool):
        raise ValueError("global.root2d_constraint.enabled must be boolean")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("global.root2d_constraint.stride must be an integer >= 1")
    return Root2DConstraintConfig(enabled=enabled, stride=stride)


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
        rerank=global_raw.get("rerank", ""),
        root2d_constraint=_parse_root2d_constraint(global_raw.get("root2d_constraint")),
    )

    motion_types = {}
    for name, spec_raw in raw.get("motion_types", {}).items():
        vel_raw = spec_raw.get("vel_cmd", {})
        prompt_override = spec_raw.get("prompt")
        if prompt_override is not None and (
            not isinstance(prompt_override, str) or not prompt_override.strip()
        ):
            raise ValueError(f"motion_types.{name}.prompt must be a non-empty string")
        arm_swing = spec_raw.get("arm_swing", "none")
        if arm_swing not in {"auto", "sagittal", "lateral", "none"}:
            raise ValueError(
                f"motion_types.{name}.arm_swing must be auto/sagittal/lateral/none"
            )
        spec = MotionSpec(
            name=name,
            description=spec_raw.get("description", ""),
            prompt_override=prompt_override.strip() if prompt_override is not None else None,
            arm_swing=arm_swing,
            duration_range=tuple(spec_raw["duration"]) if "duration" in spec_raw else (3.0, 8.0),
            vel_cmd=_parse_vel_cmd(vel_raw),
            torso_height_range=tuple(spec_raw["torso_height"]) if "torso_height" in spec_raw else None,
            styles=spec_raw.get("styles", []),
            polar_vel_cmd=_parse_polar_vel_cmd(vel_raw),
            weight=spec_raw.get("weight", 1.0),
            num_samples=spec_raw.get("num_samples", 10),
            diffusion_steps=spec_raw.get("diffusion_steps", global_config.diffusion_steps),
        )
        motion_types[name] = spec

    return LocomotionConfig(global_=global_config, motion_types=motion_types)
