"""Distribution-based sampling for locomotion motion parameters.

Supports two sampling methods:
  - "uniform": independent random draws from each dimension (default)
  - "lhs": Latin Hypercube Sampling for better coverage of high-D parameter space
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import LocomotionConfig, MotionSpec
from .prompts import build_motion_prompt


@dataclass
class SampledMotion:
    """A single sampled motion specification ready for generation."""
    motion_type: str          # e.g. "walk", "run"
    prompt: str               # final text prompt
    duration: float           # seconds
    vel: dict[str, float]     # {"vx": 0.5, "vy": 0.0, "wz": 0.1}
    torso_height: Optional[float] = None  # normalized; None = unconstrained
    style: str = ""
    diffusion_steps: int = 100
    # ── Distributed-Root2D extensions (DATA-KIMODO-FRAMEWORK-PORT-008) ──
    heading_deg: Optional[float] = None
    stride: Optional[int] = None
    keyframes: Optional[list] = None
    constraint_path: Optional[str] = None
    emit_heading: Optional[bool] = None
    output_name: Optional[str] = None
    speed_hint: bool = True


def _prompt_for_spec(spec: MotionSpec, style: str, torso_height: Optional[float],
                     vel: dict[str, float]) -> str:
    """Build the final prompt for a spec.

    If the spec declares an exact ``prompt`` override it is used verbatim
    (normalised to end with a single period).  Otherwise the description is
    composed with style/torso/speed hints (speed hint suppressed when
    ``spec.speed_hint`` is False).
    """
    if spec.prompt:
        return spec.prompt.rstrip(".") + "."
    return build_motion_prompt(
        spec.description,
        style,
        torso_height=torso_height if (torso_height is not None and spec.name != "stand") else None,
        vel=vel,
        speed_hint=spec.speed_hint,
    )


def _extra_kwargs(spec: MotionSpec) -> dict:
    """Return the Distributed-Root2D extension kwargs copied from a spec."""
    return dict(
        heading_deg=spec.heading_deg,
        stride=spec.stride,
        keyframes=spec.keyframes,
        constraint_path=spec.constraint_path,
        emit_heading=spec.emit_heading,
        output_name=spec.output_name,
        speed_hint=spec.speed_hint,
    )


def _sort_key_for(sample: SampledMotion, by: str) -> float:
    """Return a scalar sort key for a sampled motion.

    Args:
        sample: A SampledMotion instance.
        by: Sort dimension — "vx", "vy", "wz", "speed" (|v|),
            "torso", or "duration".

    Returns:
        Float sort key (lower = earlier).
    """
    if by == "vx":
        return sample.vel.get("vx", 0.0)
    elif by == "vy":
        return sample.vel.get("vy", 0.0)
    elif by == "wz":
        return sample.vel.get("wz", 0.0)
    elif by == "speed":
        vx = sample.vel.get("vx", 0.0)
        vy = sample.vel.get("vy", 0.0)
        return (vx ** 2 + vy ** 2) ** 0.5
    elif by == "torso":
        return sample.torso_height
    elif by == "duration":
        return sample.duration
    else:
        return 0.0


def lhs_sample(rng: np.random.RandomState, ranges: list[tuple[float, float]], n: int) -> np.ndarray:
    """Latin Hypercube Sampling over D continuous dimensions.

    Divides each dimension into ``n`` equal-probability strata and draws
    one sample per stratum.  Intra-stratum positions are randomised
    (``U(0,1)`` within the cell), so repeated seeds produce identical grids
    while individual draws remain stochastic.

    Args:
        rng: numpy RandomState for reproducibility.
        ranges: list of ``(low, high)`` bounds, one per dimension.  Shape ``(D,)``.
        n: number of samples.

    Returns:
        ``(n, D)`` array of sampled values.
    """
    d = len(ranges)
    # stratified grid: each column is a random permutation of (0, 1, ..., n-1)
    grid = np.empty((n, d))
    for j in range(d):
        grid[:, j] = rng.permutation(n)

    # intra-stratum jitter  U ~ (0, 1)
    jitter = rng.uniform(size=(n, d))

    # normalise to [0, 1), then map to actual ranges
    samples = (grid + jitter) / float(n)

    for j, (lo, hi) in enumerate(ranges):
        samples[:, j] = lo + samples[:, j] * (hi - lo)

    return samples


def _spec_param_ranges(spec: MotionSpec) -> list[tuple[float, float]]:
    """Return ordered continuous-parameter ranges for a MotionSpec.

    Order: duration, vx, vy, wz, torso_height.
    Dimensions present are determined by ``vel_cmd`` keys.
    """
    ranges = [spec.duration_range]
    for key in ("vx", "vy", "wz"):
        if key in spec.vel_cmd:
            ranges.append((spec.vel_cmd[key].min, spec.vel_cmd[key].max))
    if spec.torso_height_range is not None:
        ranges.append(spec.torso_height_range)
    return ranges


def _lhs_sample_from_spec(
    rng: np.random.RandomState, spec: MotionSpec
) -> list[SampledMotion]:
    """Generate ``spec.num_samples`` motions with LHS over the spec's ranges."""
    n = spec.num_samples
    ranges = _spec_param_ranges(spec)
    lhs_values = lhs_sample(rng, ranges, n)

    vel_keys_present = [k for k in ("vx", "vy", "wz") if k in spec.vel_cmd]

    samples = []
    for row in range(n):
        vals = iter(lhs_values[row])
        duration = float(next(vals))

        vel = {}
        for k in vel_keys_present:
            vel[k] = float(next(vals))

        torso_height = float(next(vals))

        style = spec.styles[rng.randint(len(spec.styles))] if spec.styles else ""

        prompt = _prompt_for_spec(spec, style, torso_height, vel)

        samples.append(SampledMotion(
            motion_type=spec.name,
            prompt=prompt,
            duration=duration,
            vel=vel,
            torso_height=torso_height,
            style=style,
            diffusion_steps=spec.diffusion_steps,
            **_extra_kwargs(spec),
        ))

    return samples


class MotionSampler:
    """Samples motion parameters from LocomotionConfig distributions."""

    def __init__(self, config: LocomotionConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self._specs = list(config.motion_types.values())
        self._weights = [s.weight for s in self._specs]

    def sample_params(self, spec: MotionSpec) -> SampledMotion:
        """Sample one motion from a specific MotionSpec distribution (uniform)."""
        duration = self.rng.uniform(*spec.duration_range)
        style = spec.styles[self.rng.randint(len(spec.styles))] if spec.styles else ""

        vel = {}
        for key, vr in spec.vel_cmd.items():
            vel[key] = vr.sample(self.rng)

        torso_height = self.rng.uniform(*spec.torso_height_range) if spec.torso_height_range is not None else None

        prompt = _prompt_for_spec(spec, style, torso_height, vel)

        return SampledMotion(
            motion_type=spec.name,
            prompt=prompt,
            duration=duration,
            vel=vel,
            torso_height=torso_height,
            style=style,
            diffusion_steps=spec.diffusion_steps,
            **_extra_kwargs(spec),
        )

    def sample_weighted_type(self) -> MotionSpec:
        """Pick a motion type weighted by its weight field."""
        total = sum(self._weights)
        probs = [w / total for w in self._weights]
        idx = self.rng.choice(len(self._specs), p=probs)
        return self._specs[idx]

    def generate_batch_specs(self, method: str = "uniform",
                             rerank: str = "") -> dict[str, list[SampledMotion]]:
        """For each motion type, sample ``num_samples`` motions.

        Args:
            method: "uniform" (default) or "lhs" (Latin Hypercube Sampling).
            rerank: If set, sort samples within each type by this dimension
                    before returning ("vx", "vy", "wz", "speed", "torso",
                    "duration").  Empty string = no sort (random order).

        Returns:
            Dict mapping type_name → list of SampledMotion objects.
        """
        batch: dict[str, list[SampledMotion]] = {}
        for spec in self._specs:
            if method == "lhs":
                batch[spec.name] = _lhs_sample_from_spec(self.rng, spec)
            else:
                batch[spec.name] = [self.sample_params(spec) for _ in range(spec.num_samples)]

            # Rerank: sort samples within this type for regularised output
            if rerank:
                batch[spec.name] = sorted(
                    batch[spec.name],
                    key=lambda s: _sort_key_for(s, rerank),
                )

        return batch

    def generate_random_specs(self, total: int, method: str = "uniform",
                              rerank: str = "") -> list[SampledMotion]:
        """Generate ``total`` motions by randomly selecting types (weighted).

        Args:
            total: number of motions desired.
            method: "uniform" (default) or "lhs".
            rerank: If set, sort samples within each type by this dimension
                    before returning.

        Returns:
            Flat list of SampledMotion objects.
        """
        if method == "lhs":
            # pick n from each type weighted, then LHS in each group
            counts = np.floor(np.array(self._weights) / sum(self._weights) * total).astype(int)
            # distribute remainder
            deficit = total - counts.sum()
            order = self.rng.permutation(len(self._specs))
            for i in range(deficit):
                counts[order[i % len(self._specs)]] += 1

            results: list[SampledMotion] = []
            for spec, n in zip(self._specs, counts):
                if n == 0:
                    continue
                spec_copy = MotionSpec(
                    name=spec.name,
                    description=spec.description,
                    duration_range=spec.duration_range,
                    vel_cmd=dict(spec.vel_cmd),
                    torso_height_range=spec.torso_height_range,
                    styles=spec.styles,
                    weight=spec.weight,
                    num_samples=n,
                    diffusion_steps=spec.diffusion_steps,
                    prompt=spec.prompt,
                    **_extra_kwargs(spec),
                )
                results.extend(_lhs_sample_from_spec(self.rng, spec_copy))

            # Rerank within each type group, keep type groups ordered
            if rerank:
                results.sort(key=lambda s: (s.motion_type, _sort_key_for(s, rerank)))
            else:
                self.rng.shuffle(results)

            return results

        specs = []
        for _ in range(total):
            spec = self.sample_weighted_type()
            specs.append(self.sample_params(spec))
        return specs
