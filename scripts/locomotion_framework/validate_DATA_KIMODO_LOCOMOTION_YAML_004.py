#!/usr/bin/env python3
"""Dependency-light static validation for DATA-KIMODO-LOCOMOTION-YAML-004."""

from __future__ import annotations

import argparse
from pathlib import Path

from locomotion_framework.config import load_config
from locomotion_framework.constraints import build_constraints_json
from locomotion_framework.sampler import MotionSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path)
    args = parser.parse_args()

    paths = sorted(args.configs.glob("*.yaml"))
    assert len(paths) == 12, len(paths)
    checked = 0
    for path in paths:
        cfg = load_config(path)
        root = cfg.global_.root2d_constraint
        samples = MotionSampler(cfg, seed=cfg.global_.seed).generate_batch_specs(
            method=cfg.global_.sampling_method
        )
        assert set(samples) == {"forward", "right"}
        for direction, group in samples.items():
            assert len(group) == 1
            sample = group[0]
            expected = (
                "A person walks forward."
                if direction == "forward"
                else "A person walks to the right."
            )
            assert sample.prompt == expected, (path, sample.prompt)
            constraints = build_constraints_json(
                sample, fps=cfg.global_.fps,
                enabled=root.enabled, stride=root.stride,
            )
            if not root.enabled:
                assert constraints == []
            else:
                assert len(constraints) == 1
                constraint = constraints[0]
                expected_count = len(range(0, 240, root.stride))
                if (239 % root.stride) != 0:
                    expected_count += 1
                assert len(constraint["frame_indices"]) == expected_count
                has_heading = "global_root_heading" in constraint
                assert has_heading == ("heading" in path.stem)
            checked += 1
    print(f"PASS: {checked} direction/config cells across {len(paths)} YAML files")


if __name__ == "__main__":
    main()

