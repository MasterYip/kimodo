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
    description: str = ""                  # base text prompt template; empty = unconstrained
    prompt_override: Optional[str] = None   # exact final prompt; bypasses composition
    arm_swing: str = "none"                 # text intent: auto/sagittal/lateral/none
    duration_range: tuple[float, float] = (3.0, 8.0)  # (min, max) seconds
    vel_cmd: dict[str, VelRange] = field(default_factory=dict)  # {"vx": ..., "vy": ..., "wz": ...}
    torso_height_range: Optional[tuple[float, float]] = None  # (min, max) normalized; None = unconstrained
    styles: list[str] = field(default_factory=list)  # ["casually", "briskly", ...]
    weight: float = 1.0                    # relative sampling probability
    num_samples: int = 10                  # how many motions from this type
    diffusion_steps: int = 100
    # ── Distributed-Root2D extensions (DATA-KIMODO-FRAMEWORK-PORT-008) ──
    # All optional and backward-compatible: default None keeps the previous
    # velocity-only behaviour.
    prompt: Optional[str] = None          # exact prompt override (used verbatim,
                                          #   skipping style/torso/speed hints)
    heading_deg: Optional[float] = None   # initial world heading for arc paths
                                          #   (converted to radians internally)
    stride: Optional[int] = None          # Root2D keyframe stride (1 = dense,
                                          #   8 = sparse). None = framework default 1.
    keyframes: Optional[list] = None      # explicit waypoints as [[frame, x, z], ...]
                                          #   (native x +left, z +forward). None = off.
    constraint_path: Optional[str] = None # path to a DISTRIBUTED-007-style Root2D
                                          #   constraint JSON (list of dicts). When set,
                                          #   the exact frame_indices/smooth_root_2d/
                                          #   global_root_heading are loaded verbatim.
    emit_heading: Optional[bool] = None   # global_root_heading emission control:
                                          #   True = emit (straight paths use the
                                          #     constant ``heading_deg``; arcs the
                                          #     rotating heading),
                                          #   False = never emit,
                                          #   None = default (straight: none;
                                          #     arc: emitted — original behaviour).
    output_name: Optional[str] = None     # exact output directory/name for this motion
                                          #   (overrides the rltracker build_motion_name)
    speed_hint: bool = True               # append the "at a brisk pace"/"slowly" hint
                                          #   from vel magnitude (False for exact-prompt
                                          #   parity batches)


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
    cfg_type: str = "separated"       # CFG strategy passed to the kimodo model call
                                      #   ("nocfg", "regular", or "separated")
    cfg_weight: tuple[float, float] = (2.0, 2.0)  # separated (text, constraint) CFG weights
    root2d_constraint: Root2DConstraintConfig = field(default_factory=Root2DConstraintConfig)


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
        cfg_type=global_raw.get("cfg_type", "separated"),
        cfg_weight=tuple(global_raw.get("cfg_weight", [2.0, 2.0])),
        root2d_constraint=_parse_root2d_constraint(global_raw.get("root2d_constraint")),
    )

    config_dir = Path(path).resolve().parent

    def _resolve_constraint_path(cp: Optional[str]) -> Optional[str]:
        """Resolve a relative constraint_path against the config file's dir."""
        if cp is None:
            return None
        p = Path(cp)
        if p.is_absolute():
            return str(p)
        return str((config_dir / p).resolve())

    motion_types = {}
    for name, spec_raw in raw.get("motion_types", {}).items():
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
            vel_cmd=_parse_vel_cmd(spec_raw.get("vel_cmd", {})),
            torso_height_range=tuple(spec_raw["torso_height"]) if "torso_height" in spec_raw else None,
            styles=spec_raw.get("styles", []),
            weight=spec_raw.get("weight", 1.0),
            num_samples=spec_raw.get("num_samples", 10),
            diffusion_steps=spec_raw.get("diffusion_steps", global_config.diffusion_steps),
            prompt=spec_raw.get("prompt"),
            heading_deg=spec_raw.get("heading_deg"),
            stride=spec_raw.get("stride"),
            keyframes=spec_raw.get("keyframes"),
            constraint_path=_resolve_constraint_path(spec_raw.get("constraint_path")),
            emit_heading=spec_raw.get("emit_heading"),
            output_name=spec_raw.get("output_name"),
            speed_hint=spec_raw.get("speed_hint", True),
        )
        motion_types[name] = spec

    return LocomotionConfig(global_=global_config, motion_types=motion_types)
