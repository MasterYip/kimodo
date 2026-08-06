"""Regression tests for the DATA-KIMODO-CONSTRAINT-003 Root2D framework extension.

Covers the per-motion ``root2d_density`` (dense / stride_N / endpoint) and the
optional C5-style ``global_root_heading`` reference-heading injection added for
the Y30 batch. The legacy global-stride velocity->Root2D path must remain
unchanged (density=None falls back to the configured stride).
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from locomotion_framework.config import load_config
from locomotion_framework.constraints import (
    _load_heading_override,
    build_constraints_json,
    build_root2d_constraint,
)
from locomotion_framework.sampler import MotionSampler, SampledMotion


def _heading_npy(path: Path, t: int, unit: bool = True) -> None:
    """Write a raw (T,2) global_root_heading array."""
    arr = np.stack([np.cos(np.linspace(0, 1.0, t)),
                    np.sin(np.linspace(0, 1.0, t))], axis=-1)
    if not unit:
        arr *= 3.0
    np.save(path, arr.astype(np.float64))


class Root2DDensityTest(unittest.TestCase):

    def _vel(self):
        return {"vx": 0.35, "vy": 0.0, "wz": 0.0}

    def test_dense_covers_all_frames(self):
        c = build_root2d_constraint(self._vel(), 2.0, fps=30, density="dense")
        n = 60
        self.assertEqual(len(c["smooth_root_2d"]), n)
        self.assertEqual(c["frame_indices"], list(range(n)))
        # straight path (wz=0) and no override -> no heading key (existing behavior)
        self.assertNotIn("global_root_heading", c)

    def test_endpoint_covers_first_and_last(self):
        c = build_root2d_constraint(self._vel(), 2.0, fps=30, density="endpoint")
        self.assertEqual(c["frame_indices"], [0, 59])
        self.assertEqual(len(c["smooth_root_2d"]), 2)

    def test_stride2_includes_last(self):
        c = build_root2d_constraint(self._vel(), 2.0, fps=30, density="stride_2")
        self.assertEqual(c["frame_indices"], list(range(0, 60, 2)) + [59])
        # legacy stride kwarg path must still work (backward compat)
        c2 = build_root2d_constraint(self._vel(), 2.0, fps=30, stride=2)
        self.assertEqual(c["frame_indices"], c2["frame_indices"])
        # unknown density falls back to the legacy global-stride path
        c3 = build_root2d_constraint(self._vel(), 2.0, fps=30, density="bogus")
        self.assertEqual(c3["frame_indices"], list(range(0, 60, 1)))

    def test_curved_path_emits_heading(self):
        c = build_root2d_constraint(
            {"vx": 0.35, "vy": 0.0, "wz": 0.5}, 2.0, fps=30, density="endpoint"
        )
        self.assertIn("global_root_heading", c)
        self.assertEqual(len(c["global_root_heading"]), 2)

    def test_heading_override_injected_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heading.npy"
            _heading_npy(path, 60, unit=False)  # rows have norm 3
            c = build_root2d_constraint(
                self._vel(), 2.0, fps=30, density="dense",
                heading_override=_load_heading_override(str(path), 60),
            )
            rows = np.asarray(c["global_root_heading"])
            self.assertEqual(rows.shape, (60, 2))
            norms = np.linalg.norm(rows, axis=1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_heading_override_short_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "heading.npy"
            _heading_npy(path, 30)
            with self.assertRaises(ValueError):
                build_root2d_constraint(
                    self._vel(), 2.0, fps=30,
                    heading_override=_load_heading_override(str(path), 60),
                )

    def test_load_heading_npz_key_and_zero_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.npz"
            arr = np.stack([np.cos(np.linspace(0, 1, 5)),
                            np.sin(np.linspace(0, 1, 5))], axis=-1)
            arr[2] = [0.0, 0.0]
            np.savez(path, global_root_heading=arr.astype(np.float64))
            out = _load_heading_override(str(path), 5)
            self.assertEqual(out.shape, (5, 2))
            norms = np.linalg.norm(out, axis=1)
            self.assertTrue(np.allclose(norms[np.any(out, axis=1)], 1.0))
            self.assertEqual(out[2].tolist(), [0.0, 0.0])

    def test_build_constraints_json_density_and_heading_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            heading = Path(tmp) / "h.npy"
            _heading_npy(heading, 60)
            sample = SampledMotion(
                motion_type="forward",
                prompt="A person walks forward.",
                duration=2.0,
                vel=self._vel(),
                root2d_density="endpoint",
                root2d_heading_file=str(heading),
            )
            cons = build_constraints_json(sample, fps=30, enabled=True)
            self.assertEqual(len(cons), 1)
            self.assertEqual(cons[0]["frame_indices"], [0, 59])
            self.assertIn("global_root_heading", cons[0])

    def test_build_constraints_json_enabled_false_and_legacy(self):
        # enabled=False emits no Root2D constraint
        sample = SampledMotion(
            motion_type="forward", prompt="p.", duration=2.0, vel=self._vel(),
        )
        self.assertEqual(build_constraints_json(sample, fps=30, enabled=False), [])

    def test_yaml_density_parses_and_flows_to_sample(self):
        text = """
global:
  seed: 7
  root2d_constraint: {enabled: true, stride: 1}
motion_types:
  forward:
    description: A person walks forward
    prompt: A person walks forward.
    duration: [2.0, 2.0]
    vel_cmd: {vx: [0.35, 0.35], vy: [0.0, 0.0], wz: [0.0, 0.0]}
    num_samples: 1
    root2d_density: endpoint
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(text)
            cfg = load_config(path)
            sample = MotionSampler(cfg, seed=7).generate_batch_specs()["forward"][0]
            self.assertEqual(sample.root2d_density, "endpoint")
            # invalid density is rejected at load time
            path.write_text(text.replace("endpoint", "every_other"))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
