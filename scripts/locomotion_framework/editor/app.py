"""LocoEditor v2 — Locomotion config editor + batch generation + 3D visualization.

Derived from the original Kimodo Demo class (kimodo/demo/app.py) to reuse proven
viser patterns for Character rendering, motion playback, scene setup, and theme.

Architecture:
  - Single-user Viser server on port 7861 (separate from the Kimodo demo on 7860)
  - Three tabs: Config (YAML-equivalent widgets), Generate (batch with progress),
    Visualize (3D robot playback)
  - Generation reuses the locomotion_framework pipeline (sampler + constraints + prompts)
  - 3D rendering uses kimodo.viz Character + CharacterMotion (same as the demo)
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

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
    set_motion_on_character,
    set_rest_pose,
    setup_scene,
    DARK_THEME,
    LIGHT_THEME,
)


class LocoEditor:
    """Main application class for the locomotion editor GUI.

    Follows the same patterns as kimodo.demo.app.Demo:
      - ViserServer with on_client_connect / on_client_disconnect
      - Model loading via kimodo.load_model()
      - Scene setup with grid, camera, theme
      - Character + CharacterMotion for 3D rendering
      - Playback loop advancing frames at model FPS
      - Keyboard controls on client.scene
    """

    def __init__(
        self,
        model_name: str = "kimodo-g1-rp",
        port: int = 7861,
        config_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.port = port
        self.device = "cuda:0"

        # State
        self.state = EditorState()
        self.client: Optional[viser.ClientHandle] = None

        # GPU generation lock
        self._generation_lock = threading.Lock()
        self._stop_event: Optional[threading.Event] = None

        # Load initial config
        if config_path and os.path.exists(config_path):
            print(f"Loading config: {config_path}")
            self.state.config = load_config(config_path)
            self.state.config_yaml = config_to_yaml_str(self.state.config)
        else:
            default_yaml = make_default_yaml()
            self.state.config = yaml_str_to_config(default_yaml)
            self.state.config_yaml = default_yaml

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
            self.grid_handle = setup_scene(client, dark_mode=False)
            configure_theme(client, dark_mode=False)

            # Character (pattern from Demo.add_character_motion)
            self.state.character = create_character(
                client, self.skeleton, self.model_name, dark_mode=False
            )
            self.state.current_motion = set_rest_pose(self.state.character)
            self.state.max_frame_idx = 1
            self.state.model_fps = self.model_fps

            # Build GUI panels (locomotion-specific tabs)
            panels.build_all_panels(
                client=client,
                state=self.state,
                on_generate=self._on_generate,
                on_stop=self._on_stop,
                on_load_yaml=self._on_load_yaml,
                on_save_yaml=self._on_save_yaml,
                on_camera_preset=self._on_camera_preset,
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

        Pattern from Demo.load_model.
        """
        from kimodo import load_model as kimodo_load

        print(f"Loading model: {self.model_name} ...")
        model, resolved = kimodo_load(
            self.model_name,
            device=self.device,
            return_resolved_name=True,
        )
        self.model = model
        self.skeleton = model.skeleton
        self.model_fps = float(model.motion_rep.fps)
        self.state.model_fps = self.model_fps
        self.model_name = resolved
        print(f"Model loaded: {resolved} (fps={self.model_fps})")

    # ── Playback controls ─────────────────────────────────────────

    def _wire_playback_controls(self, client: viser.ClientHandle) -> None:
        """Connect viser widget events to playback state.

        Pattern from Demo GUI callbacks + keyboard handling on client.scene.
        """
        s = self.state

        if s.gui_play_button is not None:
            @s.gui_play_button.on_click
            def _(event: viser.GuiEvent) -> None:
                s.playing = not s.playing
                s.gui_play_button.text = "⏸ Pause" if s.playing else "▶ Play"

        if s.gui_frame_slider is not None:
            @s.gui_frame_slider.on_update
            def _(event: viser.GuiEvent) -> None:
                s.frame_idx = int(event.target.value)
                self._set_frame(s.frame_idx)

        if s.gui_speed_slider is not None:
            @s.gui_speed_slider.on_update
            def _(event: viser.GuiEvent) -> None:
                s.playback_speed = event.target.value

        # Display toggles
        if s.gui_mesh_checkbox is not None:
            @s.gui_mesh_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                if s.character is not None:
                    s.character.set_skinned_mesh_visibility(event.target.value)

        if s.gui_skeleton_checkbox is not None:
            @s.gui_skeleton_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                if s.character is not None and s.character.skeleton_mesh is not None:
                    s.character.skeleton_mesh.set_visibility(event.target.value)

        if s.gui_dark_mode_checkbox is not None:
            @s.gui_dark_mode_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                configure_theme(client, dark_mode=event.target.value)
                if s.character is not None:
                    s.character.change_theme(event.target.value)
                if self.grid_handle is not None:
                    theme = DARK_THEME if event.target.value else LIGHT_THEME
                    self.grid_handle.section_color = theme["grid"]

        if s.gui_opacity_slider is not None:
            @s.gui_opacity_slider.on_update
            def _(event: viser.GuiEvent) -> None:
                if s.character is not None:
                    s.character.set_skinned_mesh_opacity(event.target.value)

        # Keyboard controls (pattern from Demo — on client.scene, not client)
        @client.scene.on_keyboard_event("keydown", debounce_ms=100)
        def _(event: viser.KeyboardEvent) -> None:
            key = event.key
            if key == " ":
                s.playing = not s.playing
                if s.gui_play_button is not None:
                    s.gui_play_button.text = "⏸ Pause" if s.playing else "▶ Play"
            elif key == "ArrowRight":
                s.frame_idx = min(s.max_frame_idx, s.frame_idx + 1)
                self._set_frame(s.frame_idx)
            elif key == "ArrowLeft":
                s.frame_idx = max(0, s.frame_idx - 1)
                self._set_frame(s.frame_idx)

    def _set_frame(self, idx: int) -> None:
        """Set current frame on motion and update slider.

        Pattern from Demo.set_frame.
        """
        s = self.state
        idx = max(0, min(s.max_frame_idx, idx))
        s.frame_idx = idx
        if s.current_motion is not None:
            s.current_motion.set_frame(idx)
        panels.set_frame_slider(s, idx)

    # ── Callbacks ──────────────────────────────────────────────────

    def _on_generate(self) -> None:
        """Called when user clicks Generate All Motions."""
        if self.state.generation_running:
            return

        # Rebuild config from widgets first
        panels.flush_widgets_to_config(self.state)

        config = self.state.config
        if config is None:
            return

        total = compute_total_motions(config)
        output_base = config.global_.output_dir

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
                    device=self.device,
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

                # Load first sample into scene
                if self.state.generated_samples:
                    panels.load_sample_into_scene(self.state, 0)

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

    def _on_camera_preset(self, preset_name: str) -> None:
        """Move camera to a named preset."""
        if self.client is not None:
            apply_camera_preset(self.client, preset_name)

    # ── Main loop ─────────────────────────────────────────────────

    def run(self) -> None:
        """Start the editor main loop (playback + event processing).

        Pattern from Demo.run.
        """
        print(f"\n{'='*60}")
        print(f"Loco Editor running at http://127.0.0.1:{self.port}")
        print(f"Model: {self.model_name}")
        print(f"GPU:  {self.device}")
        print(f"{'='*60}\n")
        print("Access from your machine:")
        print(f"  ssh -L {self.port}:127.0.0.1:{self.port} user@<server-ip> -p 22222 -N")
        print(f"  Open http://127.0.0.1:{self.port}\n")

        update_counter = 0
        playback_fps = self.model_fps * 2.0

        while True:
            last_time = time.time()
            s = self.state

            # Advance playback (pattern from Demo.run)
            if s.playing and s.current_motion is not None:
                interval = int(playback_fps / (s.playback_speed * s.model_fps))
                if update_counter % max(interval, 1) == 0:
                    next_frame = s.frame_idx + 1
                    if next_frame > s.max_frame_idx:
                        next_frame = 0
                    s.frame_idx = next_frame
                    s.current_motion.set_frame(s.frame_idx)
                    panels.set_frame_slider(s, s.frame_idx)

            elapsed = time.time() - last_time
            time.sleep(max(0, 1.0 / playback_fps - elapsed))
            update_counter = (update_counter + 1) % int(playback_fps)
