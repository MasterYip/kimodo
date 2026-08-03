import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from locomotion_framework.quality_validator import validate_one
from locomotion_framework.orchestrator import _assign_global_indices, _validate_generation_timing
from locomotion_framework.sampler import SampledMotion


def _thresholds():
    import yaml
    path = Path(__file__).resolve().parents[1] / "scripts/locomotion_framework/configs/g1_quality_thresholds_v2_fps30.yaml"
    return yaml.safe_load(path.read_text())


def _write_motion(root: Path, *, fps=30.0, corrupt_quat=False):
    motion_dir = root / "stand_still_000_still_0000__K1"
    motion_dir.mkdir(parents=True)
    frames = 100
    pos = np.zeros((frames, 30, 3), np.float32)
    pos[..., 2] = 0.8
    pos[:, 0, 2] = 0.8
    pos[:, 9, 2] = 1.1
    quat = np.zeros((frames, 30, 4), np.float32)
    quat[..., 0] = 2.0 if corrupt_quat else 1.0
    zeros_29 = np.zeros((frames, 29), np.float32)
    zeros_30 = np.zeros((frames, 30, 3), np.float32)
    np.savez_compressed(
        motion_dir / "motion.npz", fps=np.array([fps], np.float32),
        joint_pos=zeros_29, joint_vel=zeros_29, body_pos_w=pos,
        body_quat_w=quat, body_lin_vel_w=zeros_30, body_ang_vel_w=zeros_30,
    )
    return {
        "motion_type": "stand", "prompt": "a robot stands upright", "vx": "0", "vy": "0", "wz": "0",
        "torso_height": "0.85", "path": f"{motion_dir.name}/motion.npz",
    }


def test_known_good_fixture_is_accepted(tmp_path):
    row = _write_motion(tmp_path)
    result = validate_one(tmp_path, row, _thresholds())
    assert result["quality_tier"] == "accepted"


def test_bad_quaternion_is_rejected(tmp_path):
    row = _write_motion(tmp_path, corrupt_quat=True)
    result = validate_one(tmp_path, row, _thresholds())
    assert result["quality_tier"] == "rejected"
    assert "quaternion_norm" in result["reason_codes"]


def test_wrong_fps_is_rejected(tmp_path):
    row = _write_motion(tmp_path, fps=50.0)
    result = validate_one(tmp_path, row, _thresholds())
    assert result["quality_tier"] == "rejected"
    assert "fps_mismatch" in result["reason_codes"]


def test_export_indices_match_manifest_order():
    samples = [
        SampledMotion("stand", "stand", 1.0, {}),
        SampledMotion("walk", "walk", 1.0, {"vx": 0.3}),
        SampledMotion("run", "run", 1.0, {"vx": 1.2}),
    ]
    _assign_global_indices(samples)
    assert [sample._global_idx for sample in samples] == [0, 1, 2]


def test_generation_timing_accepts_native_fps_within_horizon():
    _validate_generation_timing(30.0, [2.0, 8.0, 10.0], 30.0)


def test_generation_timing_rejects_relabelled_fps():
    import pytest
    with pytest.raises(ValueError, match="native"):
        _validate_generation_timing(50.0, [8.0], 30.0)


def test_generation_timing_rejects_unsupported_horizon():
    import pytest
    with pytest.raises(ValueError, match="supported"):
        _validate_generation_timing(30.0, [10.01], 30.0)
