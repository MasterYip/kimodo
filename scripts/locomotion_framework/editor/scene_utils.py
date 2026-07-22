"""3D scene helpers wrapping kimodo.viz classes for the locomotion editor."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

import viser
from kimodo.skeleton import SkeletonBase
from kimodo.viz.scene import Character
from kimodo.viz.playback import CharacterMotion

# Define local theme dicts with grid colors (kimodo.viz.scene themes only have "mesh")
LIGHT_THEME = {
    "grid": (180, 180, 180),
    "mesh": (152, 189, 255),
}

DARK_THEME = {
    "grid": (105, 105, 110),
    "mesh": (100, 135, 195),
}


def create_character(
    client: viser.ClientHandle,
    skeleton: SkeletonBase,
    model_name: str,
    dark_mode: bool = False,
) -> Character:
    """Create a Character with skinned mesh and skeleton for the given model.

    Args:
        client: Viser client handle.
        skeleton: Skeleton from the loaded model.
        model_name: Model name (e.g. "kimodo-g1-rp") for mesh mode selection.
        dark_mode: Whether to use dark theme colors.

    Returns:
        A ``Character`` instance ready for motion playback.
    """
    if "g1" in model_name.lower():
        mesh_mode = "g1_stl"
    elif "smplx" in model_name.lower():
        mesh_mode = "smplx_skin"
    elif "soma" in model_name.lower():
        mesh_mode = "soma_skin"
    else:
        raise ValueError(f"Unrecognised model name for skinning: {model_name}")

    return Character(
        "loco_character",
        client,
        skeleton,
        create_skeleton_mesh=True,
        create_skinned_mesh=True,
        visible_skeleton=False,
        visible_skinned_mesh=True,
        dark_mode=dark_mode,
        mesh_mode=mesh_mode,
    )


def set_motion_on_character(
    character: Character,
    posed_joints: np.ndarray,       # (T, J, 3)
    global_rot_mats: np.ndarray,    # (T, J, 3, 3)
    foot_contacts: Optional[np.ndarray] = None,  # (T, C)
    model_fps: float = 30.0,
) -> CharacterMotion:
    """Load motion data onto a Character and return the CharacterMotion handle.

    The motion data is converted to torch tensors and placed on CPU for
    Viser rendering.  The character is positioned so the root is at origin
    at frame 0.
    """
    # Convert to torch on CPU
    jpos = torch.from_numpy(np.asarray(posed_joints)).float()
    jrot = torch.from_numpy(np.asarray(global_rot_mats)).float()
    fc = None
    if foot_contacts is not None:
        fc = torch.from_numpy(np.asarray(foot_contacts)).float()

    # Center root at origin (frame 0)
    root_offset = jpos[0, character.skeleton.root_idx].clone()
    root_offset[1] = 0.0  # keep y offset
    jpos = jpos - root_offset

    return CharacterMotion(character, jpos, jrot, fc)


def set_rest_pose(character: Character) -> CharacterMotion:
    """Put the character in rest (T-pose) with a single frame."""
    init_jpos, init_jrot = character.get_pose()
    jpos = init_jpos.unsqueeze(0)
    jrot = init_jrot.unsqueeze(0)
    return CharacterMotion(character, jpos, jrot)


def setup_scene(
    client: viser.ClientHandle,
    dark_mode: bool = False,
    floor_size: float = 40.0,
) -> viser.GridHandle:
    """Set up the 3D scene: grid, camera, theme.

    Returns the grid handle for later theme updates.
    """
    theme = DARK_THEME if dark_mode else LIGHT_THEME

    client.camera.position = np.array(
        [3.0, 2.0, 8.0], dtype=np.float64
    )
    client.camera.look_at = np.array([0.0, 0.5, 0.0], dtype=np.float64)
    client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    client.camera.fov = np.deg2rad(45.0)

    grid = client.scene.add_grid(
        "/grid",
        width=floor_size,
        height=floor_size,
        wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
        position=(0.0, 0.0001, 0.0),
        fade_distance=3 * floor_size,
        section_color=theme["grid"],
        infinite_grid=True,
    )

    client.scene.world_axes.visible = False
    client.scene.set_up_direction("+y")

    return grid


def configure_theme(client: viser.ClientHandle, dark_mode: bool = False) -> None:
    """Apply light/dark theme to the viser GUI."""
    client.gui.set_panel_label("Loco Editor")
    client.gui.configure_theme(
        control_layout="floating",
        control_width="large",
        dark_mode=dark_mode,
        show_logo=False,
        show_share_button=False,
        brand_color=(100, 180, 255),
    )


# ── Camera presets ──────────────────────────────────────────────────

CAMERA_PRESETS = {
    "Front": {
        "position": [0.0, 1.5, 6.0],
        "look_at": [0.0, 0.6, 0.0],
    },
    "Side": {
        "position": [6.0, 1.5, 0.0],
        "look_at": [0.0, 0.6, 0.0],
    },
    "Top": {
        "position": [0.0, 8.0, 0.1],
        "look_at": [0.0, 0.6, 0.0],
    },
    "Orbit": {
        "position": [4.0, 3.0, 5.0],
        "look_at": [0.0, 0.6, 0.0],
    },
    "Close-up": {
        "position": [0.0, 1.2, 2.5],
        "look_at": [0.0, 0.6, 0.0],
    },
}


def apply_camera_preset(client: viser.ClientHandle, preset_name: str) -> None:
    """Move camera to a named preset position."""
    preset = CAMERA_PRESETS.get(preset_name)
    if preset is None:
        return
    client.camera.position = np.array(preset["position"], dtype=np.float64)
    client.camera.look_at = np.array(preset["look_at"], dtype=np.float64)
    client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
