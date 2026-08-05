from pathlib import Path
import tempfile
import unittest

from locomotion_framework.config import load_config
from locomotion_framework.prompts import build_motion_prompt, build_multi_prompt, resolve_arm_swing
from locomotion_framework.sampler import MotionSampler


class PromptControlsTest(unittest.TestCase):
    def test_auto_direction_classes(self):
        sagittal = "with natural alternating forward and backward arm swing"
        lateral = "with natural alternating left and right arm swing and slight fore-aft clearance"
        self.assertEqual(resolve_arm_swing("auto", {"vx": 0.5, "vy": 0.0}), sagittal)
        self.assertEqual(resolve_arm_swing("auto", {"vx": -0.5, "vy": 0.0}), sagittal)
        self.assertEqual(resolve_arm_swing("auto", {"vx": 0.0, "vy": 0.35}), lateral)
        self.assertEqual(resolve_arm_swing("auto", {"vx": 0.0, "vy": -0.35}), lateral)
        self.assertEqual(resolve_arm_swing("auto", {"vx": 0.0, "vy": 0.0}), "")

    def test_exact_prompt_bypasses_composition(self):
        result = build_motion_prompt(
            "ignored", "briskly", 0.42, {"vx": 2.0, "vy": 0.0},
            exact_prompt="A person walks naturally", arm_swing="auto",
        )
        self.assertEqual(result, "A person walks naturally.")

    def test_composed_prompts_and_multi_segment(self):
        fwd = build_motion_prompt(
            "A person walks forward", torso_height=0.69,
            vel={"vx": 0.5, "vy": 0.0}, arm_swing="auto",
        )
        right = build_motion_prompt(
            "A person walks to the right", torso_height=0.69,
            vel={"vx": 0.0, "vy": 0.35}, arm_swing="auto",
        )
        self.assertEqual(fwd, "A person walks forward slightly crouching with natural alternating forward and backward arm swing.")
        self.assertEqual(right, "A person walks to the right slightly crouching with natural alternating left and right arm swing and slight fore-aft clearance.")
        class Segment:
            def __init__(self, prompt, duration): self.prompt, self.duration = prompt, duration
        prompts, durations = build_multi_prompt([Segment(fwd, 2.0), Segment(right, 3.0)])
        self.assertIn("forward and backward arm swing", prompts)
        self.assertIn("left and right arm swing", prompts)
        self.assertEqual(durations, "2.0 3.0")

    def test_yaml_validation_and_sampler(self):
        text = """
global:
  seed: 7
motion_types:
  forward:
    description: A person walks forward
    prompt: A frozen prompt
    arm_swing: auto
    duration: [1.0, 1.0]
    vel_cmd: {vx: [0.5, 0.5], vy: [0.0, 0.0], wz: [0.0, 0.0]}
    num_samples: 1
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(text)
            cfg = load_config(path)
            sample = MotionSampler(cfg, seed=7).generate_batch_specs()["forward"][0]
            self.assertEqual(sample.prompt, "A frozen prompt.")
            path.write_text(text.replace("arm_swing: auto", "arm_swing: diagonal"))
            with self.assertRaisesRegex(ValueError, "arm_swing"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
