"""Scene utilities for the Loco Editor.

Adapted from the Kimodo Demo class (kimodo/demo/app.py) to reuse proven patterns
for Character creation, scene setup, camera presets, and theme configuration.
"""

import base64
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage
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
        "position": np.array([3.0, 2.0, 5.0], dtype=np.float64),
        "look_at": np.array([0.0, 0.8, 0.0], dtype=np.float64),
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
    client: viser.ClientHandle,
    dark_mode: bool = False,
    titlebar_dark_mode_checkbox_uuid: str | None = None,
) -> None:
    """Configure viser GUI theme with a Kimodo-style titlebar.

    Pattern from Demo.configure_theme — titlebar with SVG logo, buttons,
    and built-in dark/light mode toggle synced to a checkbox UUID.
    """
    # ── Titlebar buttons (top-right) ────────────────────────────
    buttons = (
        TitlebarButton(
            text="GitHub",
            icon="GitHub",
            href="https://github.com/MasterYip/kimodo",
        ),
        TitlebarButton(
            text="Documentation",
            icon="Description",
            href="https://research.nvidia.com/labs/sil/projects/kimodo/docs/index.html",
        ),
        TitlebarButton(
            text="Project Page",
            icon=None,
            href="https://research.nvidia.com/labs/sil/projects/kimodo/",
        ),
    )

    # ── Kimodo banner logo (top-left, dual-scheme) ──────────────
    assets_candidates = [
        Path("/data/masteryip/kimodo/kimodo/assets"),
        Path(__file__).resolve().parent.parent.parent.parent / "assets",
    ]
    image = None
    for assets_dir in assets_candidates:
        dark_path = assets_dir / "KimodoLocoMoGenBlackBkg.svg"   # dark mode
        light_path = assets_dir / "KimodoLocoMoGenWhiteBkg.svg"  # light mode
        if dark_path.exists() and light_path.exists():
            dark_b64 = base64.standard_b64encode(
                dark_path.read_bytes()
            ).decode("ascii")
            light_b64 = base64.standard_b64encode(
                light_path.read_bytes()
            ).decode("ascii")
            image = TitlebarImage(
                image_url_light=f"data:image/svg+xml;base64,{light_b64}",
                image_url_dark=f"data:image/svg+xml;base64,{dark_b64}",
                image_alt="Kimodo",
                href="https://github.com/MasterYip/kimodo",
            )
            break

    # ── Titlebar config ──────────────────────────────────────────
    titlebar = TitlebarConfig(
        buttons=buttons,
        image=image,
        title_text="Fine-Grained Locomotion Generator",
    )

    client.gui.set_panel_label("Kimodo — Fine-Grained Motion Generator")
    client.gui.configure_theme(
        titlebar_content=titlebar,
        control_layout="floating",
        control_width="large",
        dark_mode=dark_mode,
        show_logo=False,
        show_share_button=False,
        titlebar_dark_mode_checkbox_uuid=titlebar_dark_mode_checkbox_uuid,
        brand_color=(152, 189, 255),
    )


def setup_scene(
    client: viser.ClientHandle, dark_mode: bool = False
) -> viser.GridHandle:
    """Set up the 3D scene with grid and default camera.

    Pattern from Demo.setup_scene.
    """
    floor_len = 20.0
    theme = DARK_THEME if dark_mode else LIGHT_THEME

    # Camera — focused on the robot at origin (~0.8 m torso height)
    client.camera.position = np.array(
        [4.0, 2.0, 6.0], dtype=np.float64
    )
    client.camera.look_at = np.array([0.0, 0.8, 0.0], dtype=np.float64)
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


def _mesh_mode(model_name: str) -> str:
    if "g1" in model_name:
        return "g1_stl"
    elif "smplx" in model_name:
        return "smplx_skin"
    elif "soma" in model_name:
        return "soma_skin"
    return "g1_stl"


def create_character(
    client: viser.ClientHandle,
    skeleton,
    model_name: str,
    dark_mode: bool = False,
    name: str = "character0",
) -> Character:
    """Create a Character instance for the given skeleton.

    Pattern from Demo.add_character_motion — Character creation part.
    """
    mesh_mode = _mesh_mode(model_name)
    character = Character(
        name,
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


def grid_position(index: int, cols: int, rows: int, spacing: float = 2.5):
    """Compute (x, z) offset for a character in a grid layout, centered at origin."""
    col = index % cols
    row = index // cols
    x_off = (col - (cols - 1) / 2.0) * spacing
    z_off = -(row - (rows - 1) / 2.0) * spacing
    return x_off, z_off


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


def _ensure_torch_device(x, device):
    """Convert numpy array or tensor to float32 torch tensor on the target device."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    if hasattr(x, "cpu"):
        return x.detach().to(device).float()
    return torch.tensor(x, device=device).float()


def set_motion_on_character(
    character: Character,
    posed_joints,                    # np.ndarray | torch.Tensor  [T, J, 3]
    global_rot_mats,                 # np.ndarray | torch.Tensor  [T, J, 3, 3]
    foot_contacts = None,            # np.ndarray | torch.Tensor | None  [T, F]
) -> CharacterMotion:
    """Set motion data on a Character, returning a CharacterMotion for playback.

    Accepts both numpy arrays and torch tensors (CPU or CUDA).
    Converts to the same device as the character's skeleton to avoid
    device-mismatch errors in global_rots_to_local_rots (which indexes
    with skeleton buffers that live on the model's device).
    """
    device = character.skeleton.joint_parents.device
    posed_joints = _ensure_torch_device(posed_joints, device)
    global_rot_mats = _ensure_torch_device(global_rot_mats, device)
    foot_contacts = _ensure_torch_device(foot_contacts, device)
    motion = CharacterMotion(character, posed_joints, global_rot_mats, foot_contacts)
    motion.set_frame(0)
    return motion
