"""Viser GUI panels for the locomotion editor.

Creates three tabs:
  - Config:   YAML load/save, global settings, per-motion-type fine-grained controls
  - Generate: generate button, progress, log, download
  - Visualize: playback, camera, visibility toggles
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import viser

from locomotion_framework.config import (
    GlobalConfig,
    LocomotionConfig,
    MotionSpec,
    VelRange,
    load_config,
)
from .serializers import (
    config_summary_md,
    config_to_yaml_str,
    compute_total_motions,
    make_default_yaml,
    yaml_str_to_config,
)
from .state import EditorState


# ── Constants ──────────────────────────────────────────────────────

MODEL_OPTIONS = ["kimodo-g1-rp", "kimodo-soma-rp", "kimodo-smplx-rp"]
SAMPLING_OPTIONS = ["uniform", "lhs"]
RERANK_OPTIONS = ["", "vx", "vy", "wz", "speed", "torso", "duration"]
PRESET_OPTIONS = ["rltracker", "kimodo"]
MAX_MOTION_TYPES = 12  # sane limit for the GUI


# ── Global settings widgets ─────────────────────────────────────────


def _build_global_settings(
    client: viser.ClientHandle,
    config: LocomotionConfig,
) -> dict[str, Any]:
    """Build global-config viser widgets.

    Returns a dict mapping field name → widget handle.
    """
    g = config.global_
    return {
        "model": client.gui.add_dropdown(
            "Model",
            options=MODEL_OPTIONS,
            initial_value=g.model,
        ),
        "seed": client.gui.add_number(
            "Seed",
            initial_value=g.seed if g.seed is not None else 42,
            min=0,
            max=2**31 - 1,
            step=1,
        ),
        "sampling_method": client.gui.add_dropdown(
            "Sampling method",
            options=SAMPLING_OPTIONS,
            initial_value=g.sampling_method,
        ),
        "rerank": client.gui.add_dropdown(
            "Rerank by",
            options=RERANK_OPTIONS,
            initial_value=g.rerank or "",
        ),
        "export_preset": client.gui.add_dropdown(
            "Export preset",
            options=PRESET_OPTIONS,
            initial_value=g.export_preset,
        ),
        "diffusion_steps": client.gui.add_number(
            "Diffusion steps",
            initial_value=g.diffusion_steps,
            min=25,
            max=250,
            step=5,
        ),
        "fps": client.gui.add_number(
            "FPS",
            initial_value=g.fps,
            min=10,
            max=120,
            step=5,
        ),
        "output_dir": client.gui.add_text(
            "Output directory",
            initial_value=g.output_dir,
        ),
    }


def _read_global_settings(handles: dict[str, Any]) -> GlobalConfig:
    """Read widget values back into a GlobalConfig."""
    return GlobalConfig(
        model=handles["model"].value,
        seed=int(handles["seed"].value),
        sampling_method=handles["sampling_method"].value,
        rerank=handles["rerank"].value,
        export_preset=handles["export_preset"].value,
        diffusion_steps=int(handles["diffusion_steps"].value),
        fps=int(handles["fps"].value),
        output_dir=handles["output_dir"].value,
    )


# ── Motion-type widgets ────────────────────────────────────────────


class MotionTypeWidgets:
    """Widget handles for one motion type in the config panel."""

    def __init__(
        self,
        client: viser.ClientHandle,
        name: str,
        spec: MotionSpec,
    ):
        self.name = name
        self.folder = client.gui.add_folder(
            f"📐 {name}",
            expand_by_default=False,
        )
        with self.folder:
            self.name_input = client.gui.add_text("Name", initial_value=name)
            self.description = client.gui.add_text(
                "Description", initial_value=spec.description
            )
            self.dur_min = client.gui.add_number(
                "Duration min (s)", initial_value=spec.duration_range[0],
                min=0.5, max=60.0, step=0.5,
            )
            self.dur_max = client.gui.add_number(
                "Duration max (s)", initial_value=spec.duration_range[1],
                min=0.5, max=60.0, step=0.5,
            )

            # Velocity ranges
            self.vx_min = client.gui.add_number(
                "vx min", initial_value=spec.vel_cmd.get("vx", VelRange(0, 0)).min,
                min=-5.0, max=5.0, step=0.05,
            )
            self.vx_max = client.gui.add_number(
                "vx max", initial_value=spec.vel_cmd.get("vx", VelRange(0, 0)).max,
                min=-5.0, max=5.0, step=0.05,
            )
            self.vy_min = client.gui.add_number(
                "vy min", initial_value=spec.vel_cmd.get("vy", VelRange(0, 0)).min,
                min=-5.0, max=5.0, step=0.05,
            )
            self.vy_max = client.gui.add_number(
                "vy max", initial_value=spec.vel_cmd.get("vy", VelRange(0, 0)).max,
                min=-5.0, max=5.0, step=0.05,
            )
            self.wz_min = client.gui.add_number(
                "wz min", initial_value=spec.vel_cmd.get("wz", VelRange(0, 0)).min,
                min=-5.0, max=5.0, step=0.05,
            )
            self.wz_max = client.gui.add_number(
                "wz max", initial_value=spec.vel_cmd.get("wz", VelRange(0, 0)).max,
                min=-5.0, max=5.0, step=0.05,
            )

            # Torso height
            self.torso_min = client.gui.add_number(
                "Torso height min", initial_value=spec.torso_height_range[0],
                min=0.20, max=1.20, step=0.01,
            )
            self.torso_max = client.gui.add_number(
                "Torso height max", initial_value=spec.torso_height_range[1],
                min=0.20, max=1.20, step=0.01,
            )

            # Styles
            self.styles = client.gui.add_text(
                "Styles (comma-separated)",
                initial_value=", ".join(spec.styles) if spec.styles else "",
            )

            # Weight and count
            self.weight = client.gui.add_number(
                "Weight", initial_value=spec.weight,
                min=0.01, max=100.0, step=0.01,
            )
            self.num_samples = client.gui.add_number(
                "Num samples", initial_value=spec.num_samples,
                min=1, max=2000, step=1,
            )

    def remove(self) -> None:
        """Remove all widgets for this motion type."""
        self.folder.remove()

    def to_spec(self, default_diffusion_steps: int = 100) -> MotionSpec:
        """Build a MotionSpec from current widget values."""
        styles_str = self.styles.value.strip()
        styles = (
            [s.strip() for s in styles_str.split(",") if s.strip()]
            if styles_str else []
        )

        vel_cmd = {}
        vx_min = float(self.vx_min.value)
        vx_max = float(self.vx_max.value)
        if abs(vx_max - vx_min) > 1e-8:
            vel_cmd["vx"] = VelRange(min=min(vx_min, vx_max), max=max(vx_min, vx_max))
        vy_min = float(self.vy_min.value)
        vy_max = float(self.vy_max.value)
        if abs(vy_max - vy_min) > 1e-8:
            vel_cmd["vy"] = VelRange(min=min(vy_min, vy_max), max=max(vy_min, vy_max))
        wz_min = float(self.wz_min.value)
        wz_max = float(self.wz_max.value)
        if abs(wz_max - wz_min) > 1e-8:
            vel_cmd["wz"] = VelRange(min=min(wz_min, wz_max), max=max(wz_min, wz_max))

        dur_min = max(0.5, float(self.dur_min.value))
        dur_max = max(dur_min, float(self.dur_max.value))
        torso_min = max(0.1, float(self.torso_min.value))
        torso_max = max(torso_min, float(self.torso_max.value))

        return MotionSpec(
            name=self.name_input.value.strip() or self.name,
            description=self.description.value.strip(),
            duration_range=(dur_min, dur_max),
            vel_cmd=vel_cmd,
            torso_height_range=(torso_min, torso_max),
            styles=styles,
            weight=float(self.weight.value),
            num_samples=int(self.num_samples.value),
            diffusion_steps=default_diffusion_steps,
        )


# ── Full panel builder ─────────────────────────────────────────────


def build_all_panels(
    client: viser.ClientHandle,
    state: EditorState,
    on_generate: Callable[[], None],
    on_stop: Callable[[], None],
    on_load_yaml: Callable[[str], None],
    on_save_yaml: Callable[[str], None],
    on_download: Callable[[], None],
    on_camera_preset: Callable[[str], None],
) -> None:
    """Create all three tab panels and populate initial state.

    Must be called once per client connect.
    """
    tab_group = client.gui.add_tab_group()

    _build_config_tab(client, tab_group, state, on_load_yaml, on_save_yaml)
    _build_generate_tab(client, tab_group, state, on_generate, on_stop, on_download)
    _build_visualize_tab(client, tab_group, state, on_camera_preset)


def _build_config_tab(
    client: viser.ClientHandle,
    tab_group: Any,
    state: EditorState,
    on_load_yaml: Callable[[str], None],
    on_save_yaml: Callable[[str], None],
) -> None:
    """Build the Config tab."""

    with tab_group.add_tab("Config", icon=None):
        # ── Global Settings ─────────────────────────────────────
        with client.gui.add_folder("Global Settings", expand_by_default=True):
            config = state.config
            if config is None:
                # Use default YAML to seed
                default_yaml = make_default_yaml()
                config = yaml_str_to_config(default_yaml)
                state.config = config
                state.config_yaml = default_yaml

            state.global_widgets = _build_global_settings(client, config)

        # ── Motion Types ────────────────────────────────────────
        with client.gui.add_folder("Motion Types", expand_by_default=True):
            state.motion_type_widgets = _build_motion_type_widgets(client, config)

            # Add / Remove buttons
            client.gui.add_button("➕ Add Motion Type").on_click(
                lambda _: _add_motion_type(state, client)
            )
            client.gui.add_button("🔄 Rebuild from widgets").on_click(
                lambda _: _rebuild_config_from_widgets(state)
            )

        # ── YAML file ops ───────────────────────────────────────
        with client.gui.add_folder("YAML File Operations", expand_by_default=False):
            state.gui_yaml_path_text = client.gui.add_text(
                "Server file path",
                initial_value="/data/masteryip/kimodo/kimodo/scripts/locomotion_framework/configs/g1_normal_loco.yaml",
            )
            state.gui_load_button = client.gui.add_button("📂 Load YAML from server")
            state.gui_save_button = client.gui.add_button("💾 Save YAML to server")

            state.gui_load_button.on_click(
                lambda _: on_load_yaml(state.gui_yaml_path_text.value)
            )
            state.gui_save_button.on_click(
                lambda _: on_save_yaml(state.gui_yaml_path_text.value)
            )

        # ── Summary ─────────────────────────────────────────────
        with client.gui.add_folder("Config Summary", expand_by_default=True):
            state.gui_config_md = client.gui.add_markdown(
                config_summary_md(config)
            )


def _build_motion_type_widgets(
    client: viser.ClientHandle, config: LocomotionConfig
) -> list[MotionTypeWidgets]:
    """Build widget groups for each motion type in the config."""
    widgets = []
    for name, spec in config.motion_types.items():
        w = MotionTypeWidgets(client, name, spec)
        widgets.append(w)
    return widgets


def _add_motion_type(state: EditorState, client: viser.ClientHandle) -> None:
    """Add a new default motion type."""
    if state.motion_type_widgets is None:
        return
    idx = len(state.motion_type_widgets) + 1
    new_name = f"type_{idx}"
    default_spec = MotionSpec(
        name=new_name,
        description="a person walks",
        duration_range=(5.0, 8.0),
        vel_cmd={
            "vx": VelRange(-0.5, 0.8),
            "vy": VelRange(-0.3, 0.3),
            "wz": VelRange(-0.5, 0.5),
        },
        torso_height_range=(0.7, 0.85),
        styles=["normally"],
        weight=0.2,
        num_samples=10,
    )
    w = MotionTypeWidgets(client, new_name, default_spec)
    state.motion_type_widgets.append(w)
    _rebuild_config_from_widgets(state)


def _rebuild_config_from_widgets(state: EditorState) -> None:
    """Read all widget values and rebuild the LocomotionConfig + summary."""
    if state.global_widgets is None or state.motion_type_widgets is None:
        return

    global_cfg = _read_global_settings(state.global_widgets)

    motion_types = {}
    for w in state.motion_type_widgets:
        name = w.name_input.value.strip()
        if not name:
            name = w.name
        spec = w.to_spec(default_diffusion_steps=global_cfg.diffusion_steps)
        motion_types[name] = spec

    state.config = LocomotionConfig(global_=global_cfg, motion_types=motion_types)
    state.config_yaml = config_to_yaml_str(state.config)

    # Update summary
    if state.gui_config_md is not None:
        state.gui_config_md.content = config_summary_md(state.config)

    # Update total motions display
    total = compute_total_motions(state.config)
    if state.gui_total_motions_text is not None:
        state.gui_total_motions_text.content = f"**Total motions:** {total} across {len(motion_types)} types"


def _rebuild_motion_type_panels(
    state: EditorState, client: viser.ClientHandle
) -> None:
    """Remove and recreate all motion type widgets from the current config."""
    if state.motion_type_widgets is not None:
        for w in state.motion_type_widgets:
            w.remove()
        state.motion_type_widgets.clear()

    if state.config is not None:
        state.motion_type_widgets = _build_motion_type_widgets(client, state.config)


def repopulate_from_config(
    state: EditorState, client: viser.ClientHandle
) -> None:
    """Reload all widgets from the current config (after loading a new YAML)."""
    config = state.config
    if config is None:
        return

    # Update global widget values
    if state.global_widgets is not None:
        g = config.global_
        state.global_widgets["model"].value = g.model
        state.global_widgets["seed"].value = g.seed if g.seed is not None else 42
        state.global_widgets["sampling_method"].value = g.sampling_method
        state.global_widgets["rerank"].value = g.rerank or ""
        state.global_widgets["export_preset"].value = g.export_preset
        state.global_widgets["diffusion_steps"].value = g.diffusion_steps
        state.global_widgets["fps"].value = g.fps
        state.global_widgets["output_dir"].value = g.output_dir

    # Rebuild motion type panels
    _rebuild_motion_type_panels(state, client)

    # Update summary
    if state.gui_config_md is not None:
        state.gui_config_md.content = config_summary_md(config)

    total = compute_total_motions(config)
    if state.gui_total_motions_text is not None:
        state.gui_total_motions_text.content = (
            f"**Total motions:** {total} across {len(config.motion_types)} types"
        )


# ── Generate tab ───────────────────────────────────────────────────


def _build_generate_tab(
    client: viser.ClientHandle,
    tab_group: Any,
    state: EditorState,
    on_generate: Callable[[], None],
    on_stop: Callable[[], None],
    on_download: Callable[[], None],
) -> None:
    """Build the Generate tab."""

    with tab_group.add_tab("Generate", icon=None):
        # ── Controls ───────────────────────────────────────────
        total = compute_total_motions(state.config) if state.config else 0
        state.gui_total_motions_text = client.gui.add_markdown(
            f"**Total motions:** {total} across "
            f"{len(state.config.motion_types) if state.config else 0} types"
        )

        state.gui_generate_button = client.gui.add_button(
            "🚀 Generate All Motions"
        )
        state.gui_generate_button.on_click(lambda _: on_generate())

        state.gui_stop_button = client.gui.add_button("⏹ Stop")
        state.gui_stop_button.on_click(lambda _: on_stop())

        client.gui.add_button("⬇ Download Last Output (ZIP)").on_click(
            lambda _: on_download()
        )

        # ── Progress ───────────────────────────────────────────
        state.gui_progress_bar = client.gui.add_progress_bar()
        state.gui_progress_bar.value = 0.0
        state.gui_progress_text = client.gui.add_markdown("*Ready.*")

        # ── Log ────────────────────────────────────────────────
        with client.gui.add_folder("Generation Log", expand_by_default=False):
            state.gui_log_md = client.gui.add_markdown(
                "*No generations yet.*"
            )


# ── Visualize tab ──────────────────────────────────────────────────


def _build_visualize_tab(
    client: viser.ClientHandle,
    tab_group: Any,
    state: EditorState,
    on_camera_preset: Callable[[str], None],
) -> None:
    """Build the Visualize tab."""

    with tab_group.add_tab("Visualize", icon=None):
        # ── Playback ───────────────────────────────────────────
        with client.gui.add_folder("Playback", expand_by_default=True):
            state.gui_play_button = client.gui.add_button("▶ Play / Pause")
            state.gui_frame_slider = client.gui.add_slider(
                "Frame",
                min=0,
                max=max(state.max_frame_idx, 1),
                step=1,
                initial_value=0,
            )
            state.gui_speed_slider = client.gui.add_slider(
                "Speed",
                min=0.1,
                max=4.0,
                step=0.1,
                initial_value=1.0,
            )
            state.gui_sample_label = client.gui.add_markdown(
                "*No motion loaded.*"
            )

            # Navigation buttons for samples
            with client.gui.add_folder("Sample Selector", expand_by_default=False):
                client.gui.add_button("◀ Previous Sample").on_click(
                    lambda _: _prev_sample(state)
                )
                client.gui.add_button("Next Sample ▶").on_click(
                    lambda _: _next_sample(state)
                )

        # ── Display ────────────────────────────────────────────
        with client.gui.add_folder("Display", expand_by_default=True):
            state.gui_mesh_checkbox = client.gui.add_checkbox(
                "Show Mesh", initial_value=True
            )
            state.gui_skeleton_checkbox = client.gui.add_checkbox(
                "Show Skeleton", initial_value=False
            )
            state.gui_foot_contacts_checkbox = client.gui.add_checkbox(
                "Show Foot Contacts", initial_value=True
            )
            state.gui_dark_mode_checkbox = client.gui.add_checkbox(
                "Dark Mode", initial_value=False
            )
            state.gui_opacity_slider = client.gui.add_slider(
                "Mesh Opacity",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=1.0,
            )

        # ── Camera ─────────────────────────────────────────────
        with client.gui.add_folder("Camera", expand_by_default=True):
            from .scene_utils import CAMERA_PRESETS
            preset_names = list(CAMERA_PRESETS.keys())
            state.gui_camera_dropdown = client.gui.add_dropdown(
                "Preset",
                options=preset_names,
                initial_value=preset_names[0] if preset_names else "",
            )
            client.gui.add_button("Go to preset").on_click(
                lambda _: on_camera_preset(state.gui_camera_dropdown.value)
            )


def _prev_sample(state: EditorState) -> None:
    """Switch to the previous generated sample."""
    if not state.generated_samples:
        return
    state.current_sample_idx = max(0, state.current_sample_idx - 1)
    _load_sample_into_scene(state, state.current_sample_idx)


def _next_sample(state: EditorState) -> None:
    """Switch to the next generated sample."""
    if not state.generated_samples:
        return
    state.current_sample_idx = min(
        len(state.generated_samples) - 1, state.current_sample_idx + 1
    )
    _load_sample_into_scene(state, state.current_sample_idx)


def _load_sample_into_scene(state: EditorState, idx: int) -> None:
    """Load one generated sample into the 3D scene."""
    from .scene_utils import set_motion_on_character

    if not state.generated_samples or state.character is None:
        return

    sample = state.generated_samples[idx]
    posed = sample.get("posed_joints")
    rot = sample.get("global_rot_mats")
    fc = sample.get("foot_contacts")

    if posed is None or rot is None:
        return

    state.current_motion = set_motion_on_character(
        state.character,
        posed,
        rot,
        fc,
        model_fps=state.model_fps,
    )
    state.max_frame_idx = posed.shape[0] - 1
    state.frame_idx = 0

    if state.gui_frame_slider is not None:
        state.gui_frame_slider.max = state.max_frame_idx
        state.gui_frame_slider.value = 0

    name = sample.get("name", f"sample_{idx}")
    mtype = sample.get("motion_type", "?")
    prompt = sample.get("prompt", "?")
    dur = sample.get("duration", 0)
    total = len(state.generated_samples)

    if state.gui_sample_label is not None:
        state.gui_sample_label.content = (
            f"**[{idx+1}/{total}]** `{mtype}` — {name}\n\n"
            f"*{prompt}*  ({dur:.1f}s)"
        )


# ── Public helpers called by app.py ─────────────────────────────────


def update_progress(state: EditorState, type_name: str, done: int, total: int) -> None:
    """Update the progress bar and text during generation."""
    if state.gui_progress_bar is not None:
        # Calculate overall progress — rough estimate
        all_total = compute_total_motions(state.config) if state.config else 1
        done_so_far = sum(
            s.num_samples
            for name, s in state.config.motion_types.items()
            # This is approximate; a more precise version would track the
            # global index.  For now just update the bar proportionally.
        ) if state.config else 1
        fraction = min(1.0, done / max(total, 1))
        state.gui_progress_bar.value = fraction

    if state.gui_progress_text is not None:
        state.gui_progress_text.content = (
            f"**Generating:** `{type_name}` — {done}/{total}"
        )


def append_log(state: EditorState, message: str) -> None:
    """Append a line to the generation log."""
    state.generation_log += message + "\n"
    if state.gui_log_md is not None:
        state.gui_log_md.content = state.generation_log


def set_generating(state: EditorState, active: bool) -> None:
    """Enable/disable the Generate button and show/hide Stop."""
    state.generation_running = active
    if state.gui_generate_button is not None:
        state.gui_generate_button.disabled = active


def set_frame_slider(state: EditorState, frame: int) -> None:
    """Update the frame slider without triggering its callback."""
    if state.gui_frame_slider is not None:
        state.gui_frame_slider.value = frame
