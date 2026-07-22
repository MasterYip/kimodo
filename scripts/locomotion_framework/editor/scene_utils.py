"""Scene utilities for the Loco Editor.

Adapted from the Kimodo Demo class (kimodo/demo/app.py) to reuse proven patterns
for Character creation, scene setup, and camera presets.
"""

from typing import Optional

import numpy as np
import torch

import viser
from kimodo.viz.viser_utils import Character, CharacterMotion

# Theme colours (copied from kimodo.demo.config)
LIGHT_THEME = {"grid": (180, 180, 180), "mesh": (152, 189, 255)}
DARK_THEME = {"grid": (105, 105, 110), "mesh": (100, 135, 195)}

# Camera presets (front, side, top, orbit)
CAMERA_PRESETS = {
    "Front": {
        "position": np.array([0.0, 1.5, 5.0], dtype=np.float64),
        "look_at": np.array([0.0, 0.8, 0.0], dtype=np.float64),
    },
    "Side": {
        "position": np.array([6.0, 1.5, 0.0], dtype=np.float64),
        "look_at": np.array([0.0, 0.8, 0.0], dtype=np.float64),
    },
    "Top": {
        "position": np.array([0.0, 8.0, 0.1], dtype=np.float64),
        "look_at": np.array([0.0, 0.8, 0.0], dtype=np.float64),
    },
    "Orbit": {
        "position": np.array([2.74, 1.88, 7.68], dtype=np.float64),
        "look_at": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    },
}


def apply_camera_preset(client: viser.ClientHandle, preset_name: str) -> None:
    """Move the client camera to a named preset."""
    preset = CAMERA_PRESETS.get(preset_name)
    if preset is None:
        return
    client.camera.position = preset["position"]
    client.camera.look_at = preset["look_at"]
    # Keep up-direction consistent
    client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def configure_theme(
    client: viser.ClientHandle, dark_mode: bool = False
) -> None:
    """Configure viser GUI theme (panel label, control width, dark mode).

    Simplified from Demo.configure_theme — no titlebar branding.
    """
    client.gui.set_panel_label("Loco Editor")
    client.gui.configure_theme(
        control_layout="floating",
        control_width="large",
        dark_mode=dark_mode,
        show_logo=False,
        show_share_button=False,
    )


def setup_scene(
    client: viser.ClientHandle, dark_mode: bool = False
) -> viser.GridHandle:
    """Set up the 3D scene with grid and default camera.

    Pattern from Demo.setup_scene.
    """
    floor_len = 20.0
    theme = DARK_THEME if dark_mode else LIGHT_THEME

    # Camera (same initial position as Demo)
    client.camera.position = np.array(
        [2.74, 1.88, 7.68], dtype=np.float64
    )
    client.camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    client.camera.fov = np.deg2rad(45.0)

    # Grid (same pattern as Demo)
    grid_handle = client.scene.add_grid(
        "/grid",
        width=floor_len,
        height=floor_len,
        wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
        position=(0.0, 0.0001, 0.0),
        fade_distance=3 * floor_len,
        section_color=theme["grid"],
        infinite_grid=True,
    )
    return grid_handle


def create_character(
    client: viser.ClientHandle,
    skeleton,
    model_name: str,
    dark_mode: bool = False,
) -> Character:
    """Create a Character instance for the given skeleton.

    Pattern from Demo.add_character_motion — Character creation part.
    """
    # Determine mesh mode based on model name (same as Demo)
    if "g1" in model_name:
        mesh_mode = "g1_stl"
    elif "smplx" in model_name:
        mesh_mode = "smplx_skin"
    elif "soma" in model_name:
        mesh_mode = "soma_skin"
    else:
        mesh_mode = "g1_stl"

    character = Character(
        "character0",
        client,
        skeleton,
        create_skeleton_mesh=True,
        create_skinned_mesh=True,
        visible_skeleton=True,
        visible_skinned_mesh=True,
        skinned_mesh_opacity=0.9,
        show_foot_contacts=False,
        dark_mode=dark_mode,
        mesh_mode=mesh_mode,
        gui_use_soma_layer_checkbox=None,
    )
    return character


def set_rest_pose(character: Character) -> CharacterMotion:
    """Set character to its default rest pose (T-pose or equivalent).

    Returns a CharacterMotion so playback controls can reference it.
    """
    # Get the character's default pose
    init_joints_pos, init_joints_rot = character.get_pose()
    # Create a 1-frame motion
    joints_pos = init_joints_pos[None].repeat(1, 1, 1)
    joints_rot = init_joints_rot[None].repeat(1, 1, 1, 1)
    motion = CharacterMotion(character, joints_pos, joints_rot, None)
    motion.set_frame(0)
    return motion


def set_motion_on_character(
    character: Character,
    posed_joints: torch.Tensor,      # [T, J, 3]
    global_rot_mats: torch.Tensor,   # [T, J, 3, 3]
    foot_contacts: Optional[torch.Tensor] = None,  # [T, F]
) -> CharacterMotion:
    """Set motion data on a Character, returning a CharacterMotion for playback.

    Pattern from Demo.add_character_motion and Demo.generate.
    """
    motion = CharacterMotion(character, posed_joints, global_rot_mats, foot_contacts)
    motion.set_frame(0)
    return motion
