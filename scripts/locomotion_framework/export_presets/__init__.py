"""Export preset registry.

Each preset converts Kimodo model output into a specific dataset format.
Register new presets with the ``@register`` decorator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ExportPreset(Protocol):
    """Interface for export presets."""

    name: str
    description: str

    def export(
        self,
        *,
        posed_joints: np.ndarray,       # (T, J, 3)  world joint positions
        global_rot_mats: np.ndarray,    # (T, J, 3, 3)  world rotation matrices
        local_rot_mats: np.ndarray,     # (T, J, 3, 3)  local rotation matrices
        root_positions: np.ndarray,     # (T, 3)  root position
        fps: float,
        sample_idx: int,
        motion_type: str,
        vel: dict[str, float],
        torso_height: float,
        style: str,
        seed: int,
        output_base: Path,
    ) -> Path:
        """Export one motion sample.  Returns the output directory path."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, type] = {}


def register(name: str):
    """Decorator to register an export preset class."""

    def _decorator(cls: type) -> type:
        _registry[name] = cls
        return cls

    return _decorator


def get_preset(name: str):
    """Look up an export preset by name.

    Raises ``KeyError`` if not found.
    """
    if name not in _registry:
        raise KeyError(
            f"Unknown export preset {name!r}. Available: {list(_registry)}"
        )
    return _registry[name]()


def list_presets() -> list[str]:
    """Return names of all registered presets."""
    return sorted(_registry)


# Import presets so they self-register
from . import rltracker  # noqa: E402, F401
