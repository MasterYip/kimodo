"""Locomotion Framework: distribution-based G1 motion generation with Kimodo.

Usage:
    python3 -m locomotion_framework.orchestrator --config configs/g1_locomotion.yaml
    python3 -m locomotion_framework.orchestrator --config configs/g1_locomotion.yaml --dry-run
"""

from .config import MotionSpec, LocomotionConfig, load_config
from .sampler import MotionSampler
from .constraints import build_root2d_constraint
from .prompts import build_motion_prompt
