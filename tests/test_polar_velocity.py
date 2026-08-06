"""Tests for polar velocity configuration and sampling."""

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from locomotion_framework.config import load_config
from locomotion_framework.prompts import _speed_hint
from locomotion_framework.sampler import MotionSampler, polar_to_cartesian


def _write_config(tmp_path: Path, vel_cmd: str, method: str = "lhs") -> Path:
    path = tmp_path / "polar.yaml"
    path.write_text(
        f"""
global:
  seed: 20260806
  sampling_method: {method}
motion_types:
  walk:
    description: "A person walks naturally"
    duration: [8.0, 8.0]
    vel_cmd:
{vel_cmd}
    torso_height: [0.77, 0.77]
    styles: []
    num_samples: 24
""",
        encoding="utf-8",
    )
    return path


def test_polar_conversion_uses_forward_right_convention():
    assert polar_to_cartesian(0.5, 0.0) == pytest.approx({"vx": 0.5, "vy": 0.0})
    assert polar_to_cartesian(0.5, 90.0) == pytest.approx({"vx": 0.0, "vy": 0.5})
    assert polar_to_cartesian(0.5, 180.0) == pytest.approx({"vx": -0.5, "vy": 0.0})


def test_lhs_stratifies_speed_and_direction(tmp_path):
    config = load_config(_write_config(
        tmp_path,
        """      polar:
        speed: [0.35, 0.85]
        direction_deg: [-180.0, 180.0]
      wz: [0.0, 0.0]""",
    ))
    samples = MotionSampler(config, seed=config.global_.seed).generate_batch_specs("lhs")["walk"]

    speeds = np.array([np.hypot(s.vel["vx"], s.vel["vy"]) for s in samples])
    directions = np.array([
        np.degrees(np.arctan2(s.vel["vy"], s.vel["vx"])) for s in samples
    ])
    assert np.all((0.35 <= speeds) & (speeds <= 0.85))
    assert np.all((-180.0 <= directions) & (directions <= 180.0))

    n = len(samples)
    speed_cells = np.floor((speeds - 0.35) / (0.85 - 0.35) * n).astype(int)
    direction_cells = np.floor((directions + 180.0) / 360.0 * n).astype(int)
    assert sorted(speed_cells.tolist()) == list(range(n))
    assert sorted(direction_cells.tolist()) == list(range(n))


def test_uniform_sampling_preserves_planar_speed_in_prompt(tmp_path):
    config = load_config(_write_config(
        tmp_path,
        """      polar:
        speed: [0.5, 0.5]
        direction_deg: [90.0, 90.0]
      wz: [0.0, 0.0]""",
        method="uniform",
    ))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk"])
    assert sample.vel == pytest.approx({"vx": 0.0, "vy": 0.5, "wz": 0.0})
    assert "very slowly" not in sample.prompt
    assert _speed_hint(sample.vel) == ""


def test_rejects_mixed_cartesian_and_polar(tmp_path):
    path = _write_config(
        tmp_path,
        """      vx: [0.0, 1.0]
      polar:
        speed: [0.35, 0.85]
        direction_deg: [-180.0, 180.0]""",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        load_config(path)


def test_rejects_negative_speed(tmp_path):
    path = _write_config(
        tmp_path,
        """      polar:
        speed: [-0.1, 0.85]
        direction_deg: [-180.0, 180.0]""",
    )
    with pytest.raises(ValueError, match="non-negative"):
        load_config(path)
