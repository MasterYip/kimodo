"""Editor state dataclass."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EditorState:
    """Mutable state for the locomotion editor session."""

    # Config
    config: Any = None  # LocomotionConfig
    config_yaml: str = ""

    # Model
    skeleton: Any = None
    device: str = "cuda:0"  # GPU device for model + generation

    # 3D scene — multi-character grid (like demo's session.motions)
    characters: dict[str, Any] = field(default_factory=dict)   # name -> Character
    motions: dict[str, Any] = field(default_factory=dict)       # name -> CharacterMotion
    character: Any = None    # convenience: first/default character
    current_motion: Any = None  # convenience: first/default motion

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
    current_sample_idx: int = 0
    generation_log: str = ""
    output_dir: str = ""

    # ── GUI element refs (set by panels.py) ──

    # Config tab
    gui_yaml_text: Any = None  # The large YAML text area
    gui_preset_dropdown: Any = None
    gui_save_path_text: Any = None
    gui_config_md: Any = None
    global_widgets: dict[str, Any] = field(default_factory=dict)

    # Generate tab
    gui_progress_bar: Any = None
    gui_progress_text: Any = None
    gui_generate_button: Any = None
    gui_stop_button: Any = None
    gui_log_md: Any = None
    gui_total_motions_text: Any = None

    # Visualize tab
    gui_play_button: Any = None
    gui_speed_slider: Any = None
    gui_mesh_checkbox: Any = None
    gui_skeleton_checkbox: Any = None
    gui_dark_mode_checkbox: Any = None
    gui_foot_contacts_checkbox: Any = None
    gui_opacity_slider: Any = None
    gui_camera_dropdown: Any = None
    gui_sample_label: Any = None
    gui_prev_frame_button: Any = None
    gui_next_frame_button: Any = None
    gui_prev_sample_button: Any = None
    gui_next_sample_button: Any = None
    gui_type_filter_dropdown: Any = None  # Filter by motion type
    selected_type_filter: str = ""  # Current type filter value ("" = show all)
    gui_max_batch_size: Any = None
    max_batch_size: int = 50  # Sub-batch size for generation to avoid OOM
