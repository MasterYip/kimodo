<p align="center">
  <img src="./assets/Kimodo.svg" alt="Kimodo Fine-Grained Batch Generation" width="60%">
</p>

<p align="center">
  <a href="https://github.com/MasterYip/kimodo"><img src="https://img.shields.io/badge/Built_on-Kimodo-76B900.svg" alt="Built on Kimodo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/CUDA-12.1+-green.svg" alt="CUDA"></a>
</p>

---

## Overview

**Kimodo Fine-Grained Batch Motion Generation** is a high-throughput locomotion generation pipeline built on top of the [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) motion diffusion model. It enables **dataset-scale generation** of physics-plausible robot motions with per-sample velocity constraints, multiple motion types, and a rich Web GUI for interactive authoring and 3D visualization.

> 🎯 **Use case**: generate thousands of precisely-parameterized locomotion clips for training robot control policies (RL trackers, imitation learning), with fine-grained control over velocity ranges, torso heights, styles, and sampling strategies.

<!-- TODO: insert demo GIF showing full workflow
<div align="center">
  <img src="./assets/demo_workflow.gif" width="1280">
</div>
-->

### Key Features

| Feature | Description |
|---------|-------------|
| 📝 **YAML-Driven Config** | Declare motion types (walk, run, stand, squat…) with rich parameter ranges per type |
| 🎲 **Flexible Sampling** | Uniform random or Latin Hypercube Sampling (LHS) for coverage-optimized parameter sweeps |
| 🔀 **Reranking** | Sort samples pre-generation by speed magnitude, direction, torso height, or duration |
| 🖥️ **Web GUI (LocoEditor v2)** | Three-tab Viser interface: Config editor, live generation monitor, multi-character 3D viewer |
| 🚀 **Multi-GPU Distributed** | Sample-level splitting across arbitrary GPU counts with automatic load balancing |
| 📦 **Sub-Batching** | Configurable batch chunking to avoid CUDA OOM on consumer GPUs |
| 🎭 **Paginated 3D Viz** | 20-characters-per-page grid with skeleton/mesh rendering, playback controls, and type filter |
| 💾 **Dual Export** | Kimodo NPZ (per-type dirs) or RLTracker format (flat dirs with `motion.npz`) |
| 🐍 **Python API** | Importable `generate_batch()` for scripting and CI/CD pipelines |

---

## News

- **[2026-07-24]** Added lazy STL mesh creation — page transitions are now near-instant, with meshes created on-demand when toggled
- **[2026-07-23]** Pagination support for 3D visualization grid — handle hundreds of samples without GPU overload
- **[2026-07-23]** Multi-GPU distributed generation via per-GPU Python workers — sample-level splitting with pickle-based result merging
- **[2026-07-22]** Initial release: YAML config system, LHS/uniform sampling, Web GUI with 3 tabs, single-GPU in-process generation

---

## Installation

### Prerequisites

- **Python** 3.10+
- **CUDA** 12.1+ with a supported NVIDIA GPU (tested on RTX 3090, 4090, A100)
- **Kimodo** installed and working (this pipeline runs on top of Kimodo)

### 1. Install Kimodo

Follow the [Kimodo installation guide](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html). Verify:

```bash
kimodo_demo  # should start the interactive demo on port 7860
```

### 2. Clone & Configure

```bash
cd /path/to/kimodo/scripts
git clone <this-repo> locomotion_framework
```

Ensure the repo root is on `PYTHONPATH`:

```bash
# In your .bashrc or env.sh:
export PYTHONPATH="/path/to/kimodo/scripts:$PYTHONPATH"
```

### 3. Verify Installation

```bash
cd /path/to/kimodo
source scripts/env.sh
PYTHONPATH=scripts python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_walk.yaml --dry-run
```

You should see a sampling plan printed without errors.

<!-- TODO: insert installation verification screenshot
![Install verify](./assets/install_verify.png)
-->

---

## Quick Start

### CLI — Generate 200 Walk Motions in 30 Seconds

```bash
cd /path/to/kimodo
source scripts/env.sh

# Single-GPU: 200 walk motions on GPU 5
PYTHONPATH=scripts python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_walk.yaml \
    -n 200 -g 5
```

### Multi-GPU — Split Across 4 GPUs

```bash
# 400 motions across GPUs 0,5,6,7
bash scripts/locomotion_framework/run_distributed.sh \
    -c scripts/locomotion_framework/configs/g1_normal_loco.yaml \
    -n 400 -g 0,5,6,7
```

### Web GUI — Interactive Authoring

```bash
cd /path/to/kimodo
source scripts/env.sh
python3 -m locomotion_framework.editor --model kimodo-g1-rp --port 7861
```

Then open `http://127.0.0.1:7861` in your browser (SSH tunnel if remote):

```bash
ssh -L 7861:127.0.0.1:7861 user@<server-ip> -p 22222 -N
```

<!-- TODO: insert Web GUI screenshot
<div align="center">
  <img src="./assets/gui_overview.png" width="1000">
</div>
-->

---

## Configuration Reference

### YAML Format

Motion generation is controlled by a single YAML file with two sections:

```yaml
global:
  model: kimodo-g1-rp          # Kimodo model variant
  diffusion_steps: 100          # Denoising steps (50–500)
  seed: 42                      # Random seed (reproducibility)
  output_dir: outputs/normal_loco
  sampling_method: lhs          # "uniform" or "lhs" (Latin Hypercube)
  rerank: speed                 # Pre-sort: "", "vx", "vy", "wz", "speed", "torso"
  export_preset: rltracker      # "rltracker" (flat dirs) or "kimodo" (per-type)
  gpu: 0                        # Single GPU ID or comma-separated list "5,6,7"
  fps: 30                       # Output framerate

motion_types:
  walk:
    description: "a person walks"          # Text prompt for the diffusion model
    duration: [5.0, 8.0]                  # Min/max duration (seconds)
    vel_cmd:                               # Velocity command ranges
      vx: [-0.60, 0.80]                   # Forward (+) / backward (-) m/s
      vy: [-0.50, 0.50]                   # Lateral (right + / left -) m/s
      wz: [-0.80, 0.80]                   # Angular velocity (rad/s)
    torso_height: [0.70, 0.85]            # Min/max torso height (meters)
    styles:                                # Style phrases (injected into prompt)
      - "normally"
      - "at a steady pace"
    weight: 0.50                           # Sample weight (relative to other types)
    num_samples: 50                        # How many variations to generate

  run:
    description: "a person runs"
    duration: [5.0, 8.0]
    vel_cmd:
      vx: [1.0, 2.00]
      vy: [-1.0, 1.00]
      wz: [-0.5, 0.5]
    torso_height: [0.65, 0.80]
    styles: ["normally", "steadily"]
    weight: 0.20
    num_samples: 50
```

### Sampling Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `uniform` | Independent uniform draws per parameter | Quick exploration, prototyping |
| `lhs` | Latin Hypercube Sampling — space-filling with guaranteed coverage | Dataset generation, uniform parameter coverage |

### Reranking Strategies

Before generation, samples are sorted by a configurable criterion so the most important ones generate first:

| Strategy | Description |
|----------|-------------|
| `speed` | Sort by translational speed magnitude |
| `vx` / `vy` / `wz` | Sort by specific velocity component |
| `torso` | Sort by torso height |
| `duration` | Sort by motion duration |
| *(empty)* | No reranking (original sampling order) |

### Velocity Command Semantics

Velocities are specified in the **robot's local frame** (y-up coordinate system):

| Field | Axis | Positive | Negative |
|-------|------|----------|----------|
| `vx` | Forward | Moves forward | Moves backward |
| `vy` | Lateral | Moves right | Moves left |
| `wz` | Yaw rate | Turns left (CCW) | Turns right (CW) |

---

## Web GUI — LocoEditor v2

The editor runs as a standalone Viser server on port **7861** with three tabs:

### 📝 Config Tab

- **Presets**: Load/save YAML configs from the server's `configs/` directory
- **Global Settings**: Model selection, seed, sampling method, reranking, export preset, diffusion steps, GPU ID(s), output directory
- **Motion Types Editor**: YAML text area for editing motion type definitions — parse and preview as a summary table
- **Live Summary**: Markdown table showing all motion types with parameter ranges and sample counts

<!-- TODO: Config tab screenshot
![Config Tab](./assets/gui_config.png)
-->

### 🚀 Generate Tab

- **Generate All Motions**: Single-click launch — automatically detects single vs. multi-GPU mode from the GPU field
- **Stop**: Graceful cancellation with worker termination
- **Progress Bar**: Per-motion-type progress with live sample counts
- **Max Batch Size**: Configurable sub-batch size (5–500) to tune GPU memory usage
- **Live Log**: Scrollable HTML log panel showing per-worker stdout, timing, and errors

<!-- TODO: Generate tab screenshot
![Generate Tab](./assets/gui_generate.png)
-->

### 🎭 Visualize Tab

- **Motion Type Filter**: Dropdown to show only one motion type at a time
- **Playback Controls**: Play/Stop, frame stepping, speed multiplier (0.1×–5×), timeline scrubbing
- **Page Navigation**: 20 characters per page with Prev/Next controls — supports hundreds of samples
- **Display Toggles**: Mesh visibility, skeleton visibility, mesh opacity, dark/light theme
- **Camera Presets**: Front, Side, Top, Orbit views
- **Scene Info**: Grid layout summary and sample count

<!-- TODO: Visualize tab screenshot
![Visualize Tab](./assets/gui_visualize.png)
-->

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Stop |
| `←` | Previous frame |
| `→` | Next frame |

---

## Multi-GPU Distributed Generation

When the GPU field contains a comma-separated list (e.g., `5,6,7`), the editor launches **per-GPU Python worker subprocesses** that call the same `generate_batch()` function as single-GPU mode. Results are pickled and merged, producing identical-format output across all modes.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LocoEditor                       │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ Worker   │   │ Worker   │   │ Worker   │  ...   │
│  │ GPU 5    │   │ GPU 6    │   │ GPU 7    │        │
│  │ 10       │   │ 10       │   │ 10       │        │
│  │ samples  │   │ samples  │   │ samples  │        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       │ pickle        │ pickle       │ pickle        │
│       ▼               ▼              ▼               │
│  ┌──────────────────────────────────────────┐      │
│  │        Merge & deduplicate names         │      │
│  │    walk_0000 ... walk_0029 (unique)      │      │
│  └──────────────────────────────────────────┘      │
│                       │                             │
│                       ▼                             │
│            _setup_character_grid()                  │
│            (paginated 3D scene)                     │
└─────────────────────────────────────────────────────┘
```

### Split Strategy

| Condition | Strategy |
|-----------|----------|
| `n_gpus ≤ n_types` | Type-level: round-robin assignment of motion types to GPUs |
| `n_gpus > n_types` | Sample-level: proportional split of each type's samples across GPUs |

### Command-Line Distributed Mode

You can also run distributed generation without the GUI:

```bash
bash scripts/locomotion_framework/run_distributed.sh \
    -c scripts/locomotion_framework/configs/g1_normal_loco.yaml \
    -g 0,5,6,7 \
    -o outputs/my_dataset \
    -n 400

# Dry-run to preview the split plan:
bash scripts/locomotion_framework/run_distributed.sh \
    -c scripts/locomotion_framework/configs/g1_normal_loco.yaml \
    -g 0,5,6,7 --dry-run
```

---

## Output Formats

### Kimodo Format (`export_preset: kimodo`)

```
outputs/normal_loco/
├── manifest.csv
├── walk/
│   ├── walk_0000.npz
│   ├── walk_0001.npz
│   └── ...
├── run/
│   └── ...
└── stand/
    └── ...
```

Each NPZ contains raw model output:
- `posed_joints` — global joint positions `[T, J, 3]`
- `global_rot_mats` — global rotation matrices `[T, J, 3, 3]`
- `local_rot_mats` — local (parent-relative) rotations `[T, J, 3, 3]`
- `foot_contacts` — foot contact labels `[T, 4]`
- `smooth_root_pos` — smoothed root trajectory `[T, 3]`
- `root_positions` — raw root joint trajectory `[T, 3]`
- `global_root_heading` — heading direction `[T, 2]`

### RLTracker Format (`export_preset: rltracker`)

```
outputs/normal_loco/
├── manifest.csv
├── walk_fwd_000_norm_0000__K42/
│   └── motion.npz
├── walk_fwd_000_norm_0001__K42/
│   └── motion.npz
└── ...
```

Each `motion.npz` contains RLTracker-dataset-compatible keys (all `float32`):
- `fps` — frames per second (50 Hz)
- `joint_pos` — 29 hinge joint angles `[T, 29]` (IsaacLab DOF order)
- `joint_vel` — joint angular velocities `[T, 29]`
- `body_pos_w` — 30 body world positions `[T, 30, 3]`
- `body_quat_w` — 30 body world quaternions `[T, 30, 4]`
- `body_lin_vel_w` — body linear velocities `[T, 30, 3]`
- `body_ang_vel_w` — body angular velocities `[T, 30, 3]`

> **Note**: The RLTracker format uses 30 MuJoCo bodies reordered to IsaacLab convention. For 3D visualization in the editor, use the Kimodo export preset.

### Naming Convention (RLTracker)

```
{motion_type}_{trajectory}_{heading}_{pace}_{variant:04d}__K{seed}

Examples:
  walk_fwd_000_norm_0042__K42    → walk forward, heading 0°, normal pace, variant 42
  run_fwd_025_fast_0007__K42     → run forward at 25°, fast pace, variant 7
  stand_still_000_still_0001__K42 → standing still
```

Trajectory types are auto-classified: `fwd`, `bwd`, `lat`, `turn`, `still`.

---

## Python API

### In-Process Generation

```python
from locomotion_framework.config import load_config
from locomotion_framework.editor.generation import generate_batch

config = load_config("configs/g1_walk.yaml")
result = generate_batch(
    config=config,
    output_base="outputs/walk_batch",
    device="cuda:5",
    return_tensors=True,       # Include raw tensors for 3D viz
    max_batch_size=50,         # Sub-batch to avoid OOM
    progress_callback=my_callback,  # Optional: (type_name, done, total)
    stop_event=my_stop_event,       # Optional: threading.Event
)

# result = {
#     "results": [{"name": "walk_0000", "posed_joints": ndarray, ...}, ...],
#     "total_motions": 50,
#     "elapsed_s": 45.2,
#     "output_dir": "outputs/walk_batch",
# }
```

### YAML Serialization

```python
from locomotion_framework.editor.serializers import (
    config_to_yaml_str, yaml_str_to_config, compute_total_motions
)

# Config → YAML
yaml_text = config_to_yaml_str(config, motion_types_only=True)

# YAML → Config
config = yaml_str_to_config(yaml_text)

# Count
total = compute_total_motions(config)
```

---

## GPU Memory Management

### Requirements

| Component | VRAM |
|-----------|------|
| Kimodo model (G1-RP-v1) | ~10 GB |
| Text encoder (LLM2Vec) | ~7 GB |
| Generation buffer | Variable (sub-batch controlled) |
| **Minimum recommended** | **17–20 GB** |

### Sub-Batching

Use `max_batch_size` to cap the number of samples per model call. This is critical when generating many samples at once:

```python
# In the GUI: set "Max Batch Size" in the Generate tab
# In Python:
generate_batch(config, output_base, max_batch_size=25)  # Split into 25-sample chunks
# In the worker script:
python3 worker.py --config gpu0.yaml --gpu 5 --max-batch-size 25
```

### CPU Text Encoding

For GPUs with < 17 GB VRAM, offload the text encoder to CPU:

```bash
export TEXT_ENCODER_DEVICE=cpu
```

This reduces GPU memory to ~3 GB but slightly increases generation time.

---

## Directory Structure

```
scripts/locomotion_framework/
├── config/                     # Kimodo YAML config loader
│   └── ...
├── configs/                    # Preset config files
│   ├── g1_walk.yaml
│   ├── g1_normal_loco.yaml     # Full coverage: walk, run, stand, squat
│   └── debug_squat_height.yaml
├── constraints.py              # Constraint building for Kimodo model API
├── sampler.py                  # MotionSampler: LHS + uniform + reranking
├── prompts.py                  # Prompt composition
├── export_presets/             # Output format converters
│   ├── __init__.py             # Registry + ExportPreset protocol
│   └── rltracker.py            # Kimodo FK → RLTracker NPZ converter
├── orchestrator.py             # CLI batch orchestrator
├── run_distributed.sh          # Multi-GPU launcher (shell → Python)
├── editor/                     # LocoEditor v2 Web GUI
│   ├── __init__.py             # Entry point + argparse
│   ├── __main__.py             # `python3 -m locomotion_framework.editor`
│   ├── app.py                  # LocoEditor class (Viser server)
│   ├── panels.py               # GUI widget creation (3 tabs)
│   ├── state.py                # EditorState dataclass
│   ├── serializers.py          # YAML ↔ Config conversion
│   ├── generation.py           # generate_batch() in-process pipeline
│   ├── worker.py               # Per-GPU subprocess worker
│   ├── scene_utils.py          # Character creation + mesh laziness
│   └── multi_gpu_gen.py        # Legacy multi-GPU helpers
└── README.md                   # ← You are here
```

---

## Comparison: CLI vs Web GUI vs Python API

| Feature | CLI (`orchestrator`) | Web GUI (`editor`) | Python API |
|---------|---------------------|-------------------|------------|
| Interactive config editing | ❌ | ✅ (YAML + widgets) | ❌ |
| Real-time progress | Text stdout | Progress bar + live log | Callbacks |
| 3D visualization | ❌ | ✅ (paginated grid) | ❌ |
| Multi-GPU | Shell script | Auto-detected from GPU field | Manual |
| Scripting / CI | ✅ | ❌ | ✅ |
| Result format | NPZ files | In-memory tensors + NPZ | Configurable |

---

## Common Workflows

### Dataset Generation for RL Training

```bash
# 1. Edit config with desired parameter ranges
vim scripts/locomotion_framework/configs/my_dataset.yaml

# 2. Dry-run to verify
PYTHONPATH=scripts python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/my_dataset.yaml --dry-run

# 3. Generate across 4 GPUs
bash scripts/locomotion_framework/run_distributed.sh \
    -c scripts/locomotion_framework/configs/my_dataset.yaml \
    -g 0,5,6,7 -n 2000
```

### Interactive Prototyping

```bash
# 1. Launch the editor
python3 -m locomotion_framework.editor --model kimodo-g1-rp --port 7861

# 2. In the browser:
#    - Config tab: tweak velocity ranges, styles, sample counts
#    - Generate tab: click "Generate All Motions"
#    - Visualize tab: filter by type, play back, check quality

# 3. Save the config when satisfied
#    Config tab → Save Path → "my_tuned_config.yaml" → Save Config
```

### Debugging a Single Motion Type

```python
from locomotion_framework.config import GlobalConfig, MotionSpec, VelRange, LocomotionConfig
from locomotion_framework.editor.generation import generate_batch

config = LocomotionConfig(
    global_=GlobalConfig(model="kimodo-g1-rp", diffusion_steps=100, gpu=5),
    motion_types={
        "test_walk": MotionSpec(
            name="test_walk",
            description="a robot walks briskly",
            duration_range=(3.0, 5.0),
            vel_cmd={"vx": VelRange(0.5, 0.8)},
            torso_height_range=(0.75, 0.80),
            styles=["briskly"],
            num_samples=3,
        )
    }
)

result = generate_batch(config, "outputs/debug", device="cuda:5", return_tensors=True)
for r in result["results"]:
    print(f"{r['name']}: {r['posed_joints'].shape}")
```

---

## Limitations & Best Practices

- **VRAM**: Full pipeline needs ~17 GB. Use `max_batch_size=10` if you hit OOM.
- **Multi-GPU visualization**: Each worker generates independently — sample names are auto-deduplicated during merge. Verify results are correct after large distributed runs.
- **Export preset**: The RLTracker format adds ~2–3 seconds per sample for MuJoCo coordinate conversion. Use `kimodo` preset for faster generation when you don't need the RLTracker NPZ keys.
- **Character count**: The 3D viewer supports 20 characters per page for performance. Mesh creation is deferred (lazy) to keep page transitions fast.
- **Model compatibility**: Tested with `kimodo-g1-rp`. SOMA and SMPL-X models may work but export formats differ.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: locomotion_framework` | PYTHONPATH not set | `export PYTHONPATH=/path/to/kimodo/scripts:$PYTHONPATH` |
| CUDA Out of Memory | Batch too large | Reduce `max_batch_size` (try 10–25) |
| Multi-GPU workers crash immediately | Worker path math wrong | Ensure `worker.py` is deployed; check paths in worker stdout |
| Only 10 robots in 3D view | Name collision across workers | Fixed in 2026-07-24; redeploy latest `app.py` |
| Generation produces 0 motions | Silent OOM or config error | Check the log panel for tracebacks |
| `Connection refused` on port 7861 | Editor not running | `ps aux \| grep locomotion_framework.editor` |

---

## Related Projects

- **[Kimodo](https://github.com/MasterYip/kimodo)** — The underlying motion diffusion model
- **[ARDY](https://github.com/nv-tlabs/ardy)** — Real-time controllable motion generation
- **[ProtoMotions](https://github.com/NVlabs/ProtoMotions)** — Physics-based humanoid simulation
- **[RLTracker (PegasusMoDye)](https://github.com/...)** — RL whole-body controller for G1 (downstream consumer of this pipeline's output)

---

## Citation

If you use this pipeline in your research, please cite both this work and Kimodo:

```bibtex
@article{Kimodo2026,
  title={Kimodo: Scaling Controllable Human Motion Generation},
  author={Rempe, Davis and Petrovich, Mathis and Yuan, Ye and Zhang, Haotian and
          Peng, Xue Bin and Jiang, Yifeng and Wang, Tingwu and Iqbal, Umar and
          Minor, David and de Ruyter, Michael and Li, Jiefeng and Tessler, Chen and
          Lim, Edy and Jeong, Eugene and Wu, Sam and Hassani, Ehsan and
          Huang, Michael and Yu, Jin-Bey and Chung, Chaeyeon and Song, Lina and
          Dionne, Olivier and Kautz, Jan and Yuen, Simon and Fidler, Sanja},
  journal={arXiv:2603.15546},
  year={2026}
}
```

---

## License

This codebase is licensed under [Apache-2.0](LICENSE). Model checkpoints and data are licensed separately as indicated on their Hugging Face download pages.

## Acknowledgments

Built on top of excellent open-source projects:
- [Kimodo](https://github.com/MasterYip/kimodo) — motion diffusion model
- [Viser](https://github.com/nerdstudio-project/viser) — 3D visualization framework

---

<!-- TODO: Add these assets before publishing -->
<!--
- ./assets/banner.png — Project banner image
- ./assets/demo_workflow.gif — Full workflow demo (edit config → generate → visualize)
- ./assets/gui_overview.png — All three tabs side by side
- ./assets/gui_config.png — Config tab with populated YAML
- ./assets/gui_generate.png — Generate tab during active generation
- ./assets/gui_visualize.png — Visualize tab with 20-character grid
- ./assets/mujoco_result.gif — Sample generated motion in MuJoCo
-->
