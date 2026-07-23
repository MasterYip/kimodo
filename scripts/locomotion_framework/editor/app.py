"""LocoEditor v2 — Locomotion config editor + batch generation + 3D visualization.

Derived from the original Kimodo Demo class (kimodo/demo/app.py) to reuse proven
viser patterns for Character rendering, motion playback, scene setup, and theme.

Architecture:
  - Single-user Viser server on port 7861 (separate from the Kimodo demo on 7860)
  - Three tabs: Config (YAML-equivalent widgets), Generate (batch with progress),
    Visualize (multi-character 3D robot playback)
  - Generation reuses the locomotion_framework pipeline (sampler + constraints + prompts)
  - 3D rendering uses kimodo.viz Character + CharacterMotion (same as the demo)
  - Multi-character grid layout: all generated samples play simultaneously
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import viser
from viser.theme import TitlebarConfig

# Ensure repo root is on path for locomotion_framework imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from locomotion_framework.config import load_config
from .state import EditorState
from .serializers import (
    compute_total_motions,
    config_to_yaml_str,
    make_default_yaml,
    yaml_str_to_config,
)
from . import panels
from .scene_utils import (
    apply_camera_preset,
    configure_theme,
    create_character,
    grid_position,
    set_motion_on_character,
    set_rest_pose,
    setup_scene,
    DARK_THEME,
    LIGHT_THEME,
)

# Max characters to show in the grid (to avoid GPU overload)
MAX_GRID_CHARS = 20
DEFAULT_CHAR_SPACING = 3.0


class LocoEditor:
    """Main application class for the locomotion editor GUI.

    Follows the same patterns as kimodo.demo.app.Demo:
      - ViserServer with on_client_connect / on_client_disconnect
      - Model loading via kimodo.load_model()
      - Scene setup with grid, camera, theme
      - Character + CharacterMotion for 3D rendering
      - Playback loop advancing frames at model FPS
      - Keyboard controls on client.scene
      - Multi-character simultaneous playback via motions dict
    """

    def __init__(
        self,
        model_name: str = "kimodo-g1-rp",
        port: int = 7861,
        config_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.port = port

        # State
        self.state = EditorState()
        self.client: Optional[viser.ClientHandle] = None

        # GPU generation lock
        self._generation_lock = threading.Lock()
        self._stop_event: Optional[threading.Event] = None

        # Suppress slider re-entrancy during programmatic updates
        self._setting_frame = False

        # Load initial config
        if config_path and os.path.exists(config_path):
            print(f"Loading config: {config_path}")
            self.state.config = load_config(config_path)
            self.state.config_yaml = config_to_yaml_str(self.state.config)
        else:
            default_yaml = make_default_yaml()
            self.state.config = yaml_str_to_config(default_yaml)
            self.state.config_yaml = default_yaml

        # Resolve GPU device from config
        g = self.state.config.global_
        self.state.device = f"cuda:{g.gpu}" if g.gpu is not None else "cuda:0"

        # Model
        self.model = None
        self.skeleton = None
        self.model_fps = 30.0

        # Viser server (pattern from Demo.__init__)
        self.server = viser.ViserServer(
            host="0.0.0.0",
            port=port,
            label="Loco Editor",
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")

        self.server.on_client_connect(self._on_client_connect)
        self.server.on_client_disconnect(self._on_client_disconnect)

        # Grid handle for theme switching
        self.grid_handle: Optional[viser.GridHandle] = None
        self._dark_mode = False

    # ── Client lifecycle ──────────────────────────────────────────

    def _on_client_connect(self, client: viser.ClientHandle) -> None:
        """Set up scene and GUI when a browser connects.

        Pattern derived from Demo.on_client_connect + Demo._setup_demo_for_client.
        """
        import traceback
        print(f"Client connected: {client.client_id}", flush=True)
        self.client = client

        try:
            # Load model (lazy, once)
            if self.model is None:
                self._load_model()

            # Scene setup (pattern from Demo.setup_scene)
            self.grid_handle = setup_scene(client, dark_mode=self._dark_mode)
            configure_theme(client, dark_mode=self._dark_mode)

            # Default character in rest pose (before generation)
            default_char = create_character(
                client, self.skeleton, self.model_name,
                dark_mode=self._dark_mode, name="preview",
            )
            self.state.character = default_char
            self.state.current_motion = set_rest_pose(default_char)
            self.state.characters["preview"] = default_char
            self.state.motions["preview"] = self.state.current_motion
            self.state.max_frame_idx = 1
            self.state.model_fps = self.model_fps

            # Set up viser timeline (replaces custom slider)
            client.timeline.set_defaults(
                fps=self.model_fps,
                default_num_frames_zoom=300,
                max_frames_zoom=2000,
            )
            client.timeline.set_current_frame(0)

            # Build GUI panels (locomotion-specific tabs)
            panels.build_all_panels(
                client=client,
                state=self.state,
                on_generate=self._on_generate,
                on_stop=self._on_stop,
                on_load_yaml=self._on_load_yaml,
                on_save_yaml=self._on_save_yaml,
                on_camera_preset=self._on_camera_preset,
                on_type_filter=self._on_type_filter,
            )

            # Wire playback controls (pattern from Demo + keyboard)
            self._wire_playback_controls(client)

            print(f"Client setup complete for {client.client_id}", flush=True)

        except Exception:
            print(f"FATAL: error setting up client {client.client_id}:", flush=True)
            traceback.print_exc()
            try:
                client.gui.add_markdown(
                    f"### Error loading Loco Editor\n\n"
                    f"Something went wrong during setup. Check the server log.\n\n"
                    f"```\n{traceback.format_exc()[-500:]}\n```"
                )
            except Exception:
                pass

    def _on_client_disconnect(self, client: viser.ClientHandle) -> None:
        print(f"Client disconnected: {client.client_id}")

    # ── Model loading ─────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the Kimodo model and extract skeleton + fps.

        Uses state.device so the GPU can be changed via the Config tab GUI.
        Pattern from Demo.load_model.
        """
        from kimodo import load_model as kimodo_load

        device = self.state.device
        print(f"Loading model: {self.model_name} on {device} ...")
        model, resolved = kimodo_load(
            self.model_name,
            device=device,
            return_resolved_name=True,
        )
        self.model = model
        self.skeleton = model.skeleton
        self.model_fps = float(model.motion_rep.fps)
        self.state.model_fps = self.model_fps
        self.model_name = resolved
        print(f"Model loaded: {resolved} (fps={self.model_fps})")

    # ── Character grid management ─────────────────────────────────

    def _clear_characters(self) -> None:
        """Remove all existing characters and their motions from the scene."""
        s = self.state
        for name, motion in list(s.motions.items()):
            try:
                motion.clear()
            except Exception:
                pass
        s.motions.clear()
        s.characters.clear()
        s.character = None
        s.current_motion = None

    def _setup_character_grid(self) -> None:
        """Create Characters + CharacterMotions for generated samples in a grid.

        Filters by s.selected_type_filter if set (non-empty, not "(all)").
        All characters in the grid play simultaneously at the same frame index.
        """
        s = self.state
        samples = s.generated_samples
        if not samples or self.client is None:
            return

        client = self.client

        # Filter by selected motion type
        type_filter = s.selected_type_filter
        if type_filter and type_filter != "(all)":
            display_samples = [s for s in samples if s.get("motion_type") == type_filter]
        else:
            display_samples = list(samples)

        if not display_samples:
            panels.append_log(s, f"⚠️ No samples match type filter: {type_filter}")
            return

        # Limit to MAX_GRID_CHARS
        display_samples = display_samples[:MAX_GRID_CHARS]

        # Clear existing characters
        self._clear_characters()

        n = len(display_samples)
        cols = min(n, 5)
        rows = math.ceil(n / cols)
        spacing = DEFAULT_CHAR_SPACING

        type_label = f" [{type_filter}]" if type_filter and type_filter != "(all)" else ""
        print(f"[DEBUG] Creating character grid: {n} chars ({cols}x{rows}){type_label}", flush=True)

        for i, sample in enumerate(display_samples):
            name = sample.get("name", f"sample_{i}")

            # Create character
            char = create_character(
                client, self.skeleton, self.model_name,
                dark_mode=self._dark_mode, name=name,
            )

            # Translate motion data for grid position
            pj = sample["posed_joints"]
            if hasattr(pj, "cpu"):
                pj = pj.detach().cpu().numpy()
            pj = pj.astype(np.float64).copy()
            x_off, z_off = grid_position(i, cols, rows, spacing)
            pj[..., 0] += x_off
            pj[..., 2] += z_off

            grm = sample.get("global_rot_mats")
            if hasattr(grm, "cpu"):
                grm = grm.detach().cpu().numpy()

            fc = sample.get("foot_contacts")
            if hasattr(fc, "cpu"):
                fc = fc.detach().cpu().numpy()

            # Create motion
            motion = set_motion_on_character(char, pj, grm, fc)

            # Store
            s.characters[name] = char
            s.motions[name] = motion

            # Track first character as the "convenience" ref
            if i == 0:
                s.character = char
                s.current_motion = motion

        # Set max frame to longest motion
        def _n_frames(arr) -> int:
            if arr is None:
                return 0
            return arr.shape[0] if hasattr(arr, "shape") else 0

        s.max_frame_idx = max(
            _n_frames(sample.get("posed_joints")) - 1 for sample in display_samples
        )
        s.frame_idx = 0
        s.current_sample_idx = 0

        # Update timeline range
        if client is not None:
            client.timeline.set_defaults(
                fps=self.model_fps,
                default_duration=max(1, s.max_frame_idx),
                max_duration=max(1, s.max_frame_idx),
                default_num_frames_zoom=min(300, s.max_frame_idx + 30),
                max_frames_zoom=max(300, s.max_frame_idx + 30),
            )

        # Update type filter dropdown options (when samples first arrive)
        panels.update_type_filter_options(s)

        # Apply display settings to all characters
        self._apply_display_settings()

        # Force frame 0
        self._set_frame(0)

        # Update sample label
        if s.gui_sample_label is not None:
            total_all = len(samples)
            type_info = f" | type: {type_filter}" if type_filter and type_filter != "(all)" else ""
            s.gui_sample_label.content = (
                f"**{n} characters** in grid ({cols}×{rows}){type_info}"
                f"\n{total_all} total samples across all types"
            )
        panels.append_log(s, f"🎭 Showing {n} characters in grid{type_label}")

    def _apply_display_settings(self) -> None:
        """Apply current display settings to all characters."""
        s = self.state
        show_mesh = True
        show_skel = True
        opacity = 0.9
        try:
            if s.gui_mesh_checkbox is not None:
                show_mesh = s.gui_mesh_checkbox.value
        except Exception:
            pass
        try:
            if s.gui_skeleton_checkbox is not None:
                show_skel = s.gui_skeleton_checkbox.value
        except Exception:
            pass
        try:
            if s.gui_opacity_slider is not None:
                opacity = s.gui_opacity_slider.value
        except Exception:
            pass

        for char in s.characters.values():
            char.set_skinned_mesh_visibility(show_mesh)
            if char.skeleton_mesh is not None:
                char.skeleton_mesh.set_visibility(show_skel)
            char.set_skinned_mesh_opacity(opacity)

    # ── Playback controls ─────────────────────────────────────────

    def _wire_playback_controls(self, client: viser.ClientHandle) -> None:
        """Connect viser widget events and timeline to playback state.

        Uses client.timeline for frame display/scrubbing (matching the Demo pattern),
        which eliminates the slider callback feedback loop.
        """
        s = self.state

        # Play/pause button
        if s.gui_play_button is not None:
            @s.gui_play_button.on_click
            def _(event: viser.GuiEvent) -> None:
                s.playing = not s.playing
                s.gui_play_button.text = "⏹ Stop" if s.playing else "▶ Play"

        # Frame navigation buttons
        if s.gui_prev_frame_button is not None:
            @s.gui_prev_frame_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self._set_frame(max(0, s.frame_idx - 1))

        if s.gui_next_frame_button is not None:
            @s.gui_next_frame_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self._set_frame(min(s.max_frame_idx, s.frame_idx + 1))

        # Speed slider
        if s.gui_speed_slider is not None:
            @s.gui_speed_slider.on_update
            def _(event: viser.GuiEvent) -> None:
                s.playback_speed = event.target.value

        # Timeline scrubbing (user drags the timeline handle)
        @client.timeline.on_frame_change
        def _(event: viser.TimelineFrameEvent) -> None:
            self._set_frame(event.frame)

        # --- Type filter dropdown ---
        if s.gui_type_filter_dropdown is not None:
            @s.gui_type_filter_dropdown.on_update
            def _(event: viser.GuiEvent) -> None:
                self._on_type_filter(event.target.value)

        # --- Display toggles (iterate all characters) ---

        if s.gui_mesh_checkbox is not None:
            @s.gui_mesh_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                for char in s.characters.values():
                    char.set_skinned_mesh_visibility(event.target.value)

        if s.gui_skeleton_checkbox is not None:
            @s.gui_skeleton_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                for char in s.characters.values():
                    if char.skeleton_mesh is not None:
                        char.skeleton_mesh.set_visibility(event.target.value)

        if s.gui_dark_mode_checkbox is not None:
            @s.gui_dark_mode_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                self._dark_mode = event.target.value
                configure_theme(client, dark_mode=event.target.value)
                for char in s.characters.values():
                    char.change_theme(event.target.value)
                if self.grid_handle is not None:
                    theme = DARK_THEME if event.target.value else LIGHT_THEME
                    self.grid_handle.section_color = theme["grid"]

        if s.gui_opacity_slider is not None:
            @s.gui_opacity_slider.on_update
            def _(event: viser.GuiEvent) -> None:
                for char in s.characters.values():
                    char.set_skinned_mesh_opacity(event.target.value)

        # --- Keyboard controls (pattern from Demo) ---

        @client.scene.on_keyboard_event("keydown", debounce_ms=100)
        def _(event: viser.KeyboardEvent) -> None:
            key = event.key
            if key == " ":
                s.playing = not s.playing
                if s.gui_play_button is not None:
                    s.gui_play_button.text = "⏹ Stop" if s.playing else "▶ Play"
            elif key == "ArrowRight":
                self._set_frame(min(s.max_frame_idx, s.frame_idx + 1))
            elif key == "ArrowLeft":
                self._set_frame(max(0, s.frame_idx - 1))

    def _set_frame(self, idx: int) -> None:
        """Set current frame on ALL motions and update the timeline.

        Pattern from Demo.set_frame — iterate session.motions.
        Uses a guard flag to prevent re-entrant updates from the timeline callback.
        """
        if self._setting_frame:
            return
        self._setting_frame = True

        try:
            s = self.state
            idx = max(0, min(s.max_frame_idx, idx))
            s.frame_idx = idx

            # Update all motions simultaneously (Demo pattern)
            for motion in list(s.motions.values()):
                try:
                    motion.set_frame(idx)
                except Exception:
                    pass

            # Sync timeline display
            if self.client is not None:
                self.client.timeline.set_current_frame(idx)
        finally:
            self._setting_frame = False

    # ── Callbacks ──────────────────────────────────────────────────

    def _on_generate(self) -> None:
        """Called when user clicks Generate All Motions."""
        print("[DEBUG] _on_generate CALLED", flush=True)
        if self.state.generation_running:
            print("[DEBUG] _on_generate: already running, returning", flush=True)
            return

        # Rebuild config from widgets first
        panels.flush_widgets_to_config(self.state)

        config = self.state.config
        if config is None:
            print("[DEBUG] _on_generate: config is None, returning", flush=True)
            return

        # Resolve GPU device from config (may differ from current state if user changed it)
        g = config.global_
        device = f"cuda:{g.gpu}" if g.gpu is not None else "cuda:0"
        self.state.device = device

        total = compute_total_motions(config)
        print(f"[DEBUG] _on_generate: total={total}, device={device}, starting thread", flush=True)
        output_base = g.output_dir

        # Start generation in background thread
        self._stop_event = threading.Event()
        self.state.stop_requested = False
        self.state.generation_log = ""
        if self.state.gui_log_md is not None:
            self.state.gui_log_md.content = "*Starting generation...*"

        panels.set_generating(self.state, True)
        panels.append_log(
            self.state,
            f"### Generation started\n"
            f"Model: {config.global_.model} | "
            f"Sampling: {config.global_.sampling_method} | "
            f"Rerank: {config.global_.rerank or 'none'}\n"
            f"Types: {len(config.motion_types)} | "
            f"Total motions: {total}\n"
            f"Output: `{output_base}`\n",
        )

        def _run():
            from .generation import generate_batch

            locked = self._generation_lock.acquire(blocking=True)
            try:
                result = generate_batch(
                    config=config,
                    output_base=output_base,
                    progress_callback=self._on_progress,
                    stop_event=self._stop_event,
                    device=self.state.device,
                    return_tensors=True,
                )

                self.state.generated_samples = result["results"]
                self.state.current_sample_idx = 0
                self.state.output_dir = output_base

                panels.append_log(
                    self.state,
                    f"\n✅ **Done!** {result['total_motions']} motions "
                    f"in {result['elapsed_s']:.1f}s "
                    f"({result['elapsed_s']/max(result['total_motions'],1):.2f}s/sample)\n",
                )

                # Setup multi-character grid for simultaneous playback
                if self.state.generated_samples:
                    self._setup_character_grid()

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                panels.append_log(self.state, f"\n❌ **Error:** {e}\n```\n{tb}\n```")
                print(tb, flush=True)
            finally:
                self._generation_lock.release()
                panels.set_generating(self.state, False)
                if self.state.gui_progress_bar is not None:
                    self.state.gui_progress_bar.value = 0.0
                if self.state.gui_progress_text is not None:
                    self.state.gui_progress_text.content = "*Ready.*"

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _on_stop(self) -> None:
        """Called when user clicks Stop."""
        self.state.stop_requested = True
        if self._stop_event is not None:
            self._stop_event.set()
        panels.append_log(self.state, "\n⏹ **Stop requested...**\n")

    def _on_progress(self, type_name: str, done: int, total: int) -> None:
        """Progress callback from generation pipeline."""
        panels.update_progress(self.state, type_name, done, total)
        if done == total:
            panels.append_log(self.state, f"  ✓ `{type_name}` — {total} samples")

    def _on_load_yaml(self, path: str) -> None:
        """Load YAML from a server file path and repopulate all widgets."""
        if not path:
            return
        try:
            if not os.path.exists(path):
                if self.client is not None:
                    self.client.add_notification(
                        "File not found",
                        f"Path does not exist: {path}",
                        auto_close_seconds=3.0,
                    )
                return

            self.state.config = load_config(path)
            self.state.config_yaml = config_to_yaml_str(self.state.config)
            panels.append_log(
                self.state,
                f"📂 Loaded config from `{path}`\n"
                f"Types: {len(self.state.config.motion_types)} | "
                f"Total: {compute_total_motions(self.state.config)} motions",
            )

            # Repopulate GUI
            if self.client is not None:
                panels.repopulate_from_config(self.state, self.client)

            if self.client is not None:
                self.client.add_notification(
                    "Config loaded",
                    f"Loaded: {os.path.basename(path)}",
                    auto_close_seconds=2.0,
                )
        except Exception as e:
            panels.append_log(self.state, f"❌ Load error: {e}")
            if self.client is not None:
                self.client.add_notification(
                    "Load failed", str(e), auto_close_seconds=5.0
                )

    def _on_save_yaml(self, path: str) -> None:
        """Save current config as YAML to a server file path."""
        try:
            panels.flush_widgets_to_config(self.state)

            if self.state.config is None:
                return

            yaml_str = config_to_yaml_str(self.state.config)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(yaml_str)

            self.state.config_yaml = yaml_str
            panels.append_log(self.state, f"💾 Saved config to `{path}`")

            if self.client is not None:
                self.client.add_notification(
                    "Config saved",
                    f"Saved to: {path}",
                    auto_close_seconds=2.0,
                )
        except Exception as e:
            panels.append_log(self.state, f"❌ Save error: {e}")
            if self.client is not None:
                self.client.add_notification(
                    "Save failed", str(e), auto_close_seconds=5.0
                )

    def _on_type_filter(self, type_name: str) -> None:
        """Called when user selects a motion type filter."""
        s = self.state
        s.selected_type_filter = type_name
        if s.generated_samples:
            self._setup_character_grid()

    def _on_camera_preset(self, preset_name: str) -> None:
        """Move camera to a named preset."""
        if self.client is not None:
            apply_camera_preset(self.client, preset_name)

    # ── Main loop ─────────────────────────────────────────────────

    def run(self) -> None:
        """Start the editor main loop (playback + event processing).

        Pattern from Demo.run — single update_counter loop over all client sessions.
        Since we have a single-user editor, we process just our one state.
        """
        print(f"\n{'='*60}")
        print(f"Loco Editor running at http://127.0.0.1:{self.port}")
        print(f"Model: {self.model_name}")
        print(f"GPU:  {self.state.device}")
        print(f"{'='*60}\n")
        print("Access from your machine:")
        print(f"  ssh -L {self.port}:127.0.0.1:{self.port} user@<server-ip> -p 22222 -N")
        print(f"  Open http://127.0.0.1:{self.port}\n")

        update_counter = 0
        playback_fps = self.model_fps * 2.0  # supports up to 2x speed

        while True:
            last_update_time = time.time()
            s = self.state

            # Advance playback (pattern from Demo.run — per-session frame advance)
            if s.playing and s.motions:
                update_interval = int(playback_fps / (s.playback_speed * s.model_fps))
                if update_counter % max(update_interval, 1) == 0:
                    next_frame = s.frame_idx + 1
                    if next_frame > s.max_frame_idx:
                        next_frame = 0
                    self._set_frame(next_frame)

            elapsed = time.time() - last_update_time
            time.sleep(max(0, 1.0 / playback_fps - elapsed))
            update_counter = (update_counter + 1) % int(playback_fps)
