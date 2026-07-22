"""Editor state dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Optional

from locomotion_framework.config import LocomotionConfig


@dataclass
class EditorState:
    """Mutable state for the locomotion editor session."""

    # Config
    config: Optional[LocomotionConfig] = None
    config_yaml: str = ""  # raw YAML text shown in the editor

    # Model
    model_bundle: Any = None  # ModelBundle from demo pattern
    skeleton: Any = None

    # 3D scene
    character: Any = None  # Character instance
    current_motion: Any = None  # CharacterMotion instance

    # Playback
    frame_idx: int = 0
    playing: bool = False
    playback_speed: float = 1.0
    max_frame_idx: int = 0
    model_fps: float = 30.0

    # Generation
    generation_running: bool = False
    stop_requested: bool = False
    generated_samples: list[dict] = field(default_factory=list)
    # Each dict: {"motion_type": str, "sample_idx": int, "posed_joints": np.array,
    #              "global_rot_mats": np.array, "foot_contacts": np.array, "name": str}
    current_sample_idx: int = 0
    generation_log: str = ""
    output_dir: str = ""

    # GUI element refs (set by panels.py)
    gui_config_text: Any = None
    gui_progress_bar: Any = None
    gui_progress_text: Any = None
    gui_generate_button: Any = None
    gui_stop_button: Any = None
    gui_log_md: Any = None
    gui_sample_label: Any = None
    gui_frame_slider: Any = None
    gui_speed_slider: Any = None
    gui_mesh_checkbox: Any = None
    gui_skeleton_checkbox: Any = None
    gui_dark_mode_checkbox: Any = None
    gui_foot_contacts_checkbox: Any = None
    gui_opacity_slider: Any = None
    gui_camera_dropdown: Any = None
    gui_play_button: Any = None
    gui_output_dir_text: Any = None
    gui_total_motions_text: Any = None
    gui_yaml_path_text: Any = None
    gui_load_button: Any = None
    gui_save_button: Any = None
