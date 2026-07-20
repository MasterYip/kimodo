"""Distribution-based sampling for locomotion motion parameters."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import LocomotionConfig, MotionSpec


@dataclass
class SampledMotion:
    """A single sampled motion specification ready for generation."""
    motion_type: str          # e.g. "walk", "run"
    prompt: str               # final text prompt
    duration: float           # seconds
    vel: dict[str, float]     # {"vx": 0.5, "vy": 0.0, "wz": 0.1}
    torso_height: float       # normalized
    style: str
    diffusion_steps: int


class MotionSampler:
    """Samples motion parameters from LocomotionConfig distributions."""

    def __init__(self, config: LocomotionConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self._specs = list(config.motion_types.values())
        self._weights = [s.weight for s in self._specs]

    def sample_params(self, spec: MotionSpec) -> SampledMotion:
        """Sample one motion from a specific MotionSpec distribution."""
        from .prompts import build_motion_prompt

        duration = self.rng.uniform(*spec.duration_range)
        style = spec.styles[self.rng.randint(len(spec.styles))] if spec.styles else ""

        vel = {}
        for key, vr in spec.vel_cmd.items():
            vel[key] = vr.sample(self.rng)

        torso_height = self.rng.uniform(*spec.torso_height_range)

        prompt = build_motion_prompt(
            spec.description,
            style,
            torso_height=torso_height if spec.name != "stand" else None,
        )

        return SampledMotion(
            motion_type=spec.name,
            prompt=prompt,
            duration=duration,
            vel=vel,
            torso_height=torso_height,
            style=style,
            diffusion_steps=spec.diffusion_steps,
        )

    def sample_weighted_type(self) -> MotionSpec:
        """Pick a motion type weighted by its weight field."""
        total = sum(self._weights)
        probs = [w / total for w in self._weights]
        idx = self.rng.choice(len(self._specs), p=probs)
        return self._specs[idx]

    def generate_batch_specs(self) -> dict[str, list[SampledMotion]]:
        """For each motion type, sample `num_samples` motions.

        Returns:
            Dict mapping type_name → list of SampledMotion objects.
        """
        batch: dict[str, list[SampledMotion]] = {}
        for spec in self._specs:
            samples = [self.sample_params(spec) for _ in range(spec.num_samples)]
            batch[spec.name] = samples
        return batch

    def generate_random_specs(self, total: int) -> list[SampledMotion]:
        """Generate `total` motions by randomly selecting types (weighted).

        Returns:
            Flat list of SampledMotion objects.
        """
        specs = []
        for _ in range(total):
            spec = self.sample_weighted_type()
            specs.append(self.sample_params(spec))
        return specs
