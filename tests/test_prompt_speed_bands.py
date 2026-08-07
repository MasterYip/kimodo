"""Tests for the per-speed-band exact-prompt mechanism."""

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from locomotion_framework.config import load_config
from locomotion_framework.sampler import MotionSampler


def _write_config(tmp_path: Path, speed: str, num_samples: int = 8,
                  torso: str = "0.77, 0.77") -> Path:
    path = tmp_path / "bands.yaml"
    path.write_text(
        f"""
global:
  seed: 20260807
  sampling_method: lhs
motion_types:
  walk_fwd:
    description: "A person walks forward."
    duration: [8.0, 8.0]
    vel_cmd:
      polar:
        speed: [{speed}]
        direction_deg: [-15.0, 15.0]
      wz: [0.0, 0.0]
    torso_height: [{torso}]
    styles: []
    prompt_speed_bands:
      - [0.15, "A person stands still."]
      - [0.45, "A person walks slowly forward."]
      - [0.85, "A person walks forward."]
      - [1.20, "A person walks briskly forward."]
      - [1.51, "A person jogs forward."]
    num_samples: {num_samples}
""",
        encoding="utf-8",
    )
    return path


def _prompts(samples) -> list[str]:
    return [s.prompt for s in samples]


def test_band_selected_from_sampled_speed(tmp_path):
    config = load_config(_write_config(tmp_path, "0.0, 0.0", num_samples=1))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.vel == pytest.approx({"vx": 0.0, "vy": 0.0, "wz": 0.0})
    assert sample.prompt == "A person stands still."

    config = load_config(_write_config(tmp_path, "0.6, 0.6", num_samples=1))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.prompt == "A person walks forward."

    config = load_config(_write_config(tmp_path, "1.4, 1.4", num_samples=1))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.prompt == "A person jogs forward."


def test_full_spectrum_batch_uses_all_bands(tmp_path):
    config = load_config(_write_config(tmp_path, "0.0, 1.5", num_samples=40))
    batch = MotionSampler(config, seed=config.global_.seed).generate_batch_specs("lhs")["walk_fwd"]
    prompts = set(_prompts(batch))
    assert "A person stands still." in prompts
    assert "A person walks slowly forward." in prompts
    assert "A person walks forward." in prompts
    assert "A person walks briskly forward." in prompts
    assert "A person jogs forward." in prompts
    # prompts always track the sampled speed (no fixed sentence)
    for s in batch:
        speed = np.hypot(s.vel["vx"], s.vel["vy"])
        if speed < 0.15:
            assert s.prompt == "A person stands still."
        elif speed < 0.45:
            assert "walks slowly" in s.prompt
        elif speed < 0.85:
            assert "walks forward." in s.prompt and "slowly" not in s.prompt
        elif speed < 1.2:
            assert "walks briskly" in s.prompt
        else:
            assert "jogs forward" in s.prompt


def test_torso_hint_appended_to_band_prompt(tmp_path):
    # low torso -> crouch descriptor appended to the speed-band prompt
    config = load_config(_write_config(tmp_path, "0.6, 0.6", num_samples=1,
                                       torso="0.45, 0.45"))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.prompt == "A person walks forward, crouching very low."

    # deep crouch (<0.50) on a jog-band prompt
    config = load_config(_write_config(tmp_path, "1.4, 1.4", num_samples=1,
                                       torso="0.45, 0.45"))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.prompt == "A person jogs forward, crouching very low."

    # mid crouch (0.50-0.60) -> "crouching"
    config = load_config(_write_config(tmp_path, "1.4, 1.4", num_samples=1,
                                       torso="0.55, 0.55"))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.prompt == "A person jogs forward, crouching."

    # neutral torso -> band prompt verbatim (backward compatibility, byte-equal)
    config = load_config(_write_config(tmp_path, "0.6, 0.6", num_samples=1,
                                       torso="0.77, 0.77"))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.prompt == "A person walks forward."

    # near-neutral 0.80 (>=0.70) -> no hint, prompt unchanged
    config = load_config(_write_config(tmp_path, "0.6, 0.6", num_samples=1,
                                       torso="0.80, 0.80"))
    sample = MotionSampler(config, seed=1).sample_params(config.motion_types["walk_fwd"])
    assert sample.prompt == "A person walks forward."


def test_torso_hint_stratifies_in_lhs_batch(tmp_path):
    """2-D LHS: speed and torso are both stratified; hints track torso strata."""
    config = load_config(_write_config(tmp_path, "0.0, 1.5", num_samples=8,
                                       torso="0.45, 0.8"))
    batch = MotionSampler(config, seed=config.global_.seed).generate_batch_specs(
        "lhs")["walk_fwd"]
    torsos = sorted(round(s.torso_height, 3) for s in batch)
    # 8 torso cells over [0.45, 0.8] -> ~uniform, one per stratum
    assert torsos[0] < 0.50 and torsos[-1] > 0.75
    hints = [s.prompt for s in batch]
    assert any("crouching very low" in p for p in hints)          # low strata
    assert any("slightly crouching" in p for p in hints)          # mid-low strata
    assert any("walks forward." in p and "crouching" not in p
               for p in hints)                                    # neutral strata


def test_rejects_unsorted_bands(tmp_path):
    path = _write_config(tmp_path, "0.5, 1.5")
    text = path.read_text().replace(
        "[0.15, \"A person stands still.\"]",
        "[0.85, \"A person stands still.\"]",  # out of ascending order
    )
    path.write_text(text)
    with pytest.raises(ValueError, match="ascending"):
        load_config(path)
