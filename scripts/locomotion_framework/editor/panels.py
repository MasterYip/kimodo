"""GUI panels for the Loco Editor — Config, Generate, and Visualize tabs.

Design decisions to avoid viser folder-context scoping issues:
  - Global settings use fine-grained widgets (fixed set, never rebuilt).
  - Motion types are edited as YAML text in a text area (dynamic count handled
    by the YAML format, not by dynamic widgets — avoids the "can't add widgets
    to a folder after leaving its with: block" viser limitation).
  - Load Preset updates the YAML text area AND global widget values.
  - Generate parses the YAML text area as the source of truth for the config.

Uses viser GUI patterns from the original Kimodo demo (kimodo/demo/ui.py):
  - tab_group = client.gui.add_tab_group()
  - with tab_group.add_tab("Name", viser.Icon.XXX):
  - client.gui.add_*() for all widget creation
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import viser

from locomotion_framework.config import LocomotionConfig, MotionSpec, VelRange
from .serializers import (
    compute_total_motions,
    config_summary_md,
    config_to_yaml_str,
    make_default_yaml,
    yaml_str_to_config,
)

# ── Constants ──────────────────────────────────────────────────────

_PRESET_DIR = "/data/masteryip/kimodo/kimodo/scripts/locomotion_framework/configs"

SAMPLING_OPTIONS = ["uniform", "lhs"]
EXPORT_OPTIONS = ["rltracker", "kimodo"]
RERANK_OPTIONS = ["", "vx", "vy", "wz", "speed", "torso", "duration"]
CAMERA_OPTIONS = ["Front", "Side", "Top", "Orbit"]
PANEL_WIDTH_OPTIONS = ["small", "medium", "large"]


# ── Public API ─────────────────────────────────────────────────────

def build_all_panels(
    client: viser.ClientHandle,
    state: Any,  # EditorState
    on_generate: Callable[[], None],
    on_stop: Callable[[], None],
    on_load_yaml: Callable[[str], None],
    on_save_yaml: Callable[[str], None],
    on_camera_preset: Callable[[str], None],
) -> None:
    """Create all GUI tabs and populate them with widgets.

    Called once when a client connects. All widgets are created inside
    their respective tab contexts at startup. Dynamic updates happen
    via value changes on existing widgets, not by creating new ones.
    """
    tab_group = client.gui.add_tab_group()

    # ── Config Tab ─────────────────────────────────────────────────
    with tab_group.add_tab("Config", viser.Icon.SETTINGS):
        _build_config_tab(client, state, on_load_yaml, on_save_yaml)

    # ── Generate Tab ───────────────────────────────────────────────
    with tab_group.add_tab("Generate", viser.Icon.WALK):
        _build_generate_tab(client, state, on_generate, on_stop)

    # ── Visualize Tab ──────────────────────────────────────────────
    with tab_group.add_tab("Visualize", viser.Icon.EYE):
        _build_visualize_tab(client, state, on_camera_preset)


def repopulate_from_config(state: Any, client: viser.ClientHandle) -> None:
    """Update all widget values to reflect the current config.

    Called after loading a YAML preset. Updates:
      - Global widget values (model, seed, sampling, etc.)
      - YAML text area (full config serialized)
      - Config summary markdown
      - Total motions counter

    Does NOT create or remove any widgets — only modifies existing values.
    This avoids the viser folder context scoping issue entirely.
    """
    if state.config is None:
        return

    g = state.config.global_
    gw = state.global_widgets
    if gw.get("model") is not None:
        gw["model"].value = g.model
    if gw.get("seed") is not None:
        gw["seed"].value = g.seed if g.seed is not None else 42
    if gw.get("sampling_method") is not None:
        gw["sampling_method"].value = g.sampling_method
    if gw.get("rerank") is not None:
        gw["rerank"].value = g.rerank or ""
    if gw.get("export_preset") is not None:
        gw["export_preset"].value = g.export_preset
    if gw.get("diffusion_steps") is not None:
        gw["diffusion_steps"].value = g.diffusion_steps
    if gw.get("output_dir") is not None:
        gw["output_dir"].value = g.output_dir

    # Update YAML text area with serialized config
    if state.gui_yaml_text is not None:
        state.gui_yaml_text.value = config_to_yaml_str(state.config)
    state.config_yaml = config_to_yaml_str(state.config)

    # Update the config summary and total motions
    _update_summary_and_total(state)


def flush_widgets_to_config(state: Any) -> None:
    """Read widget values and write them back into state.config.

    Priority: YAML text area is the source of truth. Global widget values
    are merged on top (they override any differences in the YAML).

    Called before generation or saving YAML.
    """
    # Parse the YAML text area as the primary config source
    if state.gui_yaml_text is not None:
        yaml_str = state.gui_yaml_text.value
        try:
            state.config = yaml_str_to_config(yaml_str)
            state.config_yaml = yaml_str
        except Exception:
            # If YAML parsing fails, keep current config and update from widgets
            pass

    if state.config is None:
        return

    # Override global settings from fine-grained widgets (they take precedence)
    g = state.config.global_
    gw = state.global_widgets
    if gw.get("model") is not None:
        g.model = gw["model"].value
    if gw.get("seed") is not None:
        g.seed = gw["seed"].value
    if gw.get("sampling_method") is not None:
        g.sampling_method = gw["sampling_method"].value
    if gw.get("rerank") is not None:
        g.rerank = gw["rerank"].value
    if gw.get("export_preset") is not None:
        g.export_preset = gw["export_preset"].value
    if gw.get("diffusion_steps") is not None:
        g.diffusion_steps = gw["diffusion_steps"].value
    if gw.get("output_dir") is not None:
        g.output_dir = gw["output_dir"].value

    # Sync YAML text area with merged config
    state.config_yaml = config_to_yaml_str(state.config)
    if state.gui_yaml_text is not None:
        try:
            state.gui_yaml_text.value = state.config_yaml
        except Exception:
            pass


def set_frame_slider(state: Any, idx: int) -> None:
    """Update the frame slider value without triggering its callback."""
    if state.gui_frame_slider is not None:
        try:
            state.gui_frame_slider.value = idx
        except Exception:
            pass


def set_generating(state: Any, running: bool) -> None:
    """Toggle generate/stop button states."""
    state.generation_running = running
    if state.gui_generate_button is not None:
        state.gui_generate_button.disabled = running
    if state.gui_stop_button is not None:
        state.gui_stop_button.disabled = not running


def update_progress(state: Any, type_name: str, done: int, total: int) -> None:
    """Update progress bar and text during generation."""
    if state.gui_progress_bar is not None:
        try:
            state.gui_progress_bar.value = done / max(total, 1)
        except Exception:
            pass
    if state.gui_progress_text is not None:
        total_all = compute_total_motions(state.config) if state.config else 1
        state.gui_progress_text.content = (
            f"**{type_name}**: {done}/{total}  |  *Generating...*"
        )


def append_log(state: Any, text: str) -> None:
    """Append text to the generation log markdown."""
    state.generation_log += text + "\n"
    if state.gui_log_md is not None:
        state.gui_log_md.content = state.generation_log[-5000:]


def load_sample_into_scene(state: Any, idx: int) -> None:
    """Load a generated sample into the 3D scene for playback."""
    if not state.generated_samples or idx >= len(state.generated_samples):
        return
    sample = state.generated_samples[idx]
    state.current_sample_idx = idx

    if state.gui_sample_label is not None:
        total = len(state.generated_samples)
        name = sample.get("name", f"sample_{idx}")
        state.gui_sample_label.content = f"**Sample {idx+1}/{total}**: `{name}`"

    from .scene_utils import set_motion_on_character
    try:
        state.current_motion = set_motion_on_character(
            state.character,
            sample["posed_joints"],
            sample["global_rot_mats"],
            sample.get("foot_contacts"),
        )
        state.max_frame_idx = sample["posed_joints"].shape[0] - 1
        state.frame_idx = 0
        if state.gui_frame_slider is not None:
            state.gui_frame_slider.max = state.max_frame_idx
            state.gui_frame_slider.value = 0
    except Exception as e:
        append_log(state, f"❌ Error loading sample {idx}: {e}")


# ── Internal: Config Tab ───────────────────────────────────────────

def _build_config_tab(
    client: viser.ClientHandle,
    state: Any,
    on_load_yaml: Callable[[str], None],
    on_save_yaml: Callable[[str], None],
) -> None:
    """Build the Config tab with global settings widgets and a YAML text area.

    All widgets are created once here and never added/removed. Dynamic
    changes (Load Preset, Save) only update widget VALUES.
    """
    gw = state.global_widgets
    config = state.config
    g = config.global_ if config else None
    initial_yaml = state.config_yaml or (config_to_yaml_str(config) if config else make_default_yaml())

    # ── Presets folder ──
    with client.gui.add_folder("Presets", expand_by_default=True):
        # Scan for preset YAML files
        preset_options = ["(none)"]
        if os.path.isdir(_PRESET_DIR):
            preset_options += sorted([
                f for f in os.listdir(_PRESET_DIR)
                if f.endswith(".yaml") or f.endswith(".yml")
            ])

        preset_dropdown = client.gui.add_dropdown(
            "Preset", options=preset_options, initial_value="(none)",
            hint="Select a config preset to load"
        )
        state.gui_preset_dropdown = preset_dropdown

        client.gui.add_button(
            "Load Preset",
            hint="Load the selected preset config"
        ).on_click(
            lambda _event: _load_preset(preset_dropdown.value, state, client, on_load_yaml)
        )

        state.gui_save_path_text = client.gui.add_text(
            "Save Path",
            initial_value=f"{_PRESET_DIR}/my_config.yaml",
            hint="Server path to save the current config"
        )
        client.gui.add_button(
            "Save Config",
            hint="Save current config as YAML to the server"
        ).on_click(
            lambda _event: on_save_yaml(state.gui_save_path_text.value)
        )

        # Panel width
        client.gui.add_dropdown(
            "Panel Width", options=PANEL_WIDTH_OPTIONS, initial_value="large",
            hint="Adjust the GUI panel width"
        ).on_update(
            lambda event: client.gui.configure_theme(control_width=event.target.value)
        )

    # ── Global Settings folder ──
    with client.gui.add_folder("Global Settings", expand_by_default=True):
        gw["model"] = client.gui.add_text(
            "Model", initial_value=g.model if g else "kimodo-g1-rp",
            hint="Kimodo model name (e.g. kimodo-g1-rp)"
        )
        gw["seed"] = client.gui.add_number(
            "Seed", initial_value=g.seed if g and g.seed else 42,
            min=0, max=999999, step=1,
            hint="Random seed for reproducibility"
        )
        gw["sampling_method"] = client.gui.add_dropdown(
            "Sampling", options=SAMPLING_OPTIONS,
            initial_value=g.sampling_method if g else "uniform",
            hint="Uniform or Latin Hypercube Sampling"
        )
        gw["rerank"] = client.gui.add_dropdown(
            "Rerank", options=RERANK_OPTIONS,
            initial_value=g.rerank if g and g.rerank else "",
            hint="Sort samples before generation"
        )
        gw["export_preset"] = client.gui.add_dropdown(
            "Export Preset", options=EXPORT_OPTIONS,
            initial_value=g.export_preset if g else "rltracker",
            hint="rltracker = flat dirs, kimodo = per-type dirs"
        )
        gw["diffusion_steps"] = client.gui.add_slider(
            "Diffusion Steps", min=10, max=500, step=10,
            initial_value=g.diffusion_steps if g else 100,
            hint="Number of denoising steps"
        )
        gw["output_dir"] = client.gui.add_text(
            "Output Dir", initial_value=g.output_dir if g else "outputs/loco_editor",
            hint="Base directory for generated motions"
        )

    # ── Motion Types YAML Editor ──
    with client.gui.add_folder("Motion Types (YAML)", expand_by_default=True):
        # Use a multiline text area for YAML editing
        state.gui_yaml_text = client.gui.add_text(
            "YAML Config",
            initial_value=initial_yaml,
            multiline=True,
            hint="Edit motion types here as YAML. Use Load Preset to populate from server configs."
        )

        # Button to refresh config from YAML
        client.gui.add_button(
            "🔄 Parse YAML & Update Summary",
            hint="Re-parse the YAML text and update the summary table"
        ).on_click(
            lambda _event: _on_parse_yaml(state)
        )

    # ── Config summary ──
    state.gui_config_md = client.gui.add_markdown(
        content=config_summary_md(config) if config else "*No config loaded*"
    )


def _load_preset(
    preset_name: str, state: Any, client: viser.ClientHandle,
    on_load_yaml: Callable[[str], None]
) -> None:
    """Callback for the Load Preset button."""
    if preset_name == "(none)" or not preset_name:
        return
    path = os.path.join(_PRESET_DIR, preset_name)
    on_load_yaml(path)


def _on_parse_yaml(state: Any) -> None:
    """Parse the YAML text area content and update the config + summary."""
    if state.gui_yaml_text is None:
        return
    try:
        yaml_str = state.gui_yaml_text.value
        state.config = yaml_str_to_config(yaml_str)
        state.config_yaml = yaml_str

        # Update global widget values too
        g = state.config.global_
        gw = state.global_widgets
        if gw.get("model") is not None:
            gw["model"].value = g.model
        if gw.get("seed") is not None:
            gw["seed"].value = g.seed if g.seed is not None else 42
        if gw.get("sampling_method") is not None:
            gw["sampling_method"].value = g.sampling_method
        if gw.get("rerank") is not None:
            gw["rerank"].value = g.rerank or ""
        if gw.get("export_preset") is not None:
            gw["export_preset"].value = g.export_preset
        if gw.get("diffusion_steps") is not None:
            gw["diffusion_steps"].value = g.diffusion_steps
        if gw.get("output_dir") is not None:
            gw["output_dir"].value = g.output_dir

        _update_summary_and_total(state)
        append_log(state, f"✅ YAML parsed: {len(state.config.motion_types)} types, "
                    f"{compute_total_motions(state.config)} total motions")
    except Exception as e:
        append_log(state, f"❌ YAML parse error: {e}")


def _update_summary_and_total(state: Any) -> None:
    """Update the config summary markdown and total motions text."""
    if state.config is None:
        return
    if state.gui_config_md is not None:
        state.gui_config_md.content = config_summary_md(state.config)
    if state.gui_total_motions_text is not None:
        state.gui_total_motions_text.content = (
            f"**Total motions:** {compute_total_motions(state.config)}"
        )


# ── Internal: Generate Tab ─────────────────────────────────────────

def _build_generate_tab(
    client: viser.ClientHandle,
    state: Any,
    on_generate: Callable[[], None],
    on_stop: Callable[[], None],
) -> None:
    """Build the Generate tab content."""
    total = compute_total_motions(state.config) if state.config else 0

    state.gui_total_motions_text = client.gui.add_markdown(
        content=f"**Total motions:** {total}"
    )

    with client.gui.add_folder("Generation", expand_by_default=True):
        state.gui_generate_button = client.gui.add_button(
            "🚀 Generate All Motions",
            hint="Generate all motions from the current config"
        )
        state.gui_generate_button.on_click(lambda _event: on_generate())

        state.gui_stop_button = client.gui.add_button(
            "⏹ Stop", hint="Stop the current generation", disabled=True
        )
        state.gui_stop_button.on_click(lambda _event: on_stop())

        state.gui_progress_text = client.gui.add_markdown(content="*Ready.*")
        state.gui_progress_bar = client.gui.add_progress_bar(value=0.0)

    state.gui_log_md = client.gui.add_markdown(content="*Log will appear here...*")


# ── Internal: Visualize Tab ────────────────────────────────────────

def _build_visualize_tab(
    client: viser.ClientHandle,
    state: Any,
    on_camera_preset: Callable[[str], None],
) -> None:
    """Build the Visualize tab content."""
    # ── Playback ──
    with client.gui.add_folder("Playback", expand_by_default=True):
        state.gui_play_button = client.gui.add_button("▶ Play")
        state.gui_frame_slider = client.gui.add_slider(
            "Frame", min=0, max=max(state.max_frame_idx, 1), step=1,
            initial_value=0,
            hint="Current frame"
        )
        state.gui_speed_slider = client.gui.add_slider(
            "Speed", min=0.1, max=5.0, step=0.1,
            initial_value=1.0,
            hint="Playback speed multiplier"
        )

    # ── Display ──
    with client.gui.add_folder("Display", expand_by_default=True):
        state.gui_mesh_checkbox = client.gui.add_checkbox(
            "Show Mesh", initial_value=True
        )
        state.gui_skeleton_checkbox = client.gui.add_checkbox(
            "Show Skeleton", initial_value=True
        )
        state.gui_dark_mode_checkbox = client.gui.add_checkbox(
            "Dark Mode", initial_value=False
        )
        state.gui_opacity_slider = client.gui.add_slider(
            "Mesh Opacity", min=0.1, max=1.0, step=0.05,
            initial_value=0.9,
            hint="Skinned mesh opacity"
        )

    # ── Camera ──
    with client.gui.add_folder("Camera", expand_by_default=True):
        cam_dd = client.gui.add_dropdown(
            "Preset", options=CAMERA_OPTIONS, initial_value="Orbit",
            hint="Move camera to a preset position"
        )
        state.gui_camera_dropdown = cam_dd
        cam_dd.on_update(lambda event: on_camera_preset(event.target.value))

    # ── Sample Navigator ──
    with client.gui.add_folder("Sample Navigator", expand_by_default=True):
        state.gui_sample_label = client.gui.add_markdown(
            content="*No samples generated yet*"
        )
        state.gui_prev_sample_button = client.gui.add_button(
            "◀ Prev Sample", disabled=True
        )
        state.gui_next_sample_button = client.gui.add_button(
            "Next Sample ▶", disabled=True
        )

        def _on_prev(_event):
            if state.generated_samples:
                new_idx = (state.current_sample_idx - 1) % len(state.generated_samples)
                load_sample_into_scene(state, new_idx)

        def _on_next(_event):
            if state.generated_samples:
                new_idx = (state.current_sample_idx + 1) % len(state.generated_samples)
                load_sample_into_scene(state, new_idx)

        state.gui_prev_sample_button.on_click(_on_prev)
        state.gui_next_sample_button.on_click(_on_next)
