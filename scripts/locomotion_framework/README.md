# Locomotion Framework — G1 Motion Generation

Distribution-based batch generation for G1 humanoid robot locomotion motions
using Kimodo. Controls velocity (vx, vy, wz), torso height, style, and
motion type via configurable distributions.

---

## Quick Start

```bash
cd /data/masteryip/kimodo/kimodo
source scripts/env.sh

# Dry run — see what will be generated
PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_nromal_loco.yaml --dry-run

# Generate full batch (all motion types)
PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_nromal_loco.yaml

# Generate exactly 100 motions (random weighted selection)
PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_nromal_loco.yaml -n 100

# Use a specific GPU
PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_nromal_loco.yaml -g 1
```

---

## Architecture

```
locomotion_framework/
├── configs/
│   └── g1_nromal_loco.yaml    ← Distribution definitions
├── config.py                 ← MotionSpec dataclass + YAML loader
├── sampler.py                ← Distribution sampling
├── constraints.py            ← Velocity → Root2D path builder
├── prompts.py                ← Text prompt composition
├── orchestrator.py           ← Batch runner + manifest output
└── README.md
```

---

## Config File Format

```yaml
global:
  model: kimodo-g1-rp
  diffusion_steps: 100
  seed: 42
  output_dir: outputs/locomotion

motion_types:
  walk:
    description: "a robot walks forward"      # base prompt
    duration: [3.0, 8.0]                     # uniform range (seconds)
    vel_cmd:                                  # velocity command ranges
      vx: [0.2, 0.8]  # forward (m/s)
      vy: [-0.1, 0.1] # lateral (m/s)
      wz: [-0.1, 0.1] # angular (rad/s)
    torso_height: [0.65, 0.85]               # normalized range
    styles:                                    # style modifiers
      - "casually"
      - "briskly"
    weight: 0.35                              # sampling probability
    num_samples: 40                           # generated count
```

### Velocity Command Semantics

| Key | Axis | Range | Meaning |
|-----|------|-------|---------|
| `vx` | Forward | positive = forward | Walking/running speed |
| `vy` | Lateral | positive = right | Sideways drift |
| `wz` | Angular | positive = counter-clockwise | Turning rate |

---

## How It Works

### 1. Distribution Sampling

For each motion type, the sampler draws from uniform distributions:
- Duration: `U(duration[0], duration[1])`
- Velocity: `U(vx[0], vx[1])`, `U(vy[0], vy[1])`, `U(wz[0], wz[1])`
- Torso height: `U(torso_height[0], torso_height[1])`
- Style: uniform random from `styles` list

### 2. Velocity → Root2D Path

Sampled velocity commands are converted to a 2D root trajectory:
- **Straight**: `x(t) = vx · t · dt`, `z(t) = vy · t · dt`
- **Turning**: circular arc + heading constraints via `global_root_heading`
- Passed to Kimodo as `Root2DConstraintSet`

### 3. Text Prompt Composition

Parameters are composed into a single prompt:
```
"a robot walks forward briskly."
"a robot with crouching walks forward carefully."
```

Torso height is mapped to qualitative hints:
- `< 0.50`: "crouching very low"
- `0.50–0.60`: "crouching"
- `0.60–0.70`: "slightly crouching"
- `0.70–0.85`: (normal — no hint)
- `> 0.85`: "standing tall"

### 4. Batch Generation

The orchestrator loads Kimodo once, then for each motion type:
1. Builds the prompt + Root2D constraints
2. Calls `model(prompt, num_frames, constraint_lst=..., num_samples=N)`
3. Saves `.npz` + `.csv` per sample with metadata

Kimodo processes all `num_samples` in parallel through one denoising loop.

---

## Output

```
outputs/locomotion/
├── manifest.csv              ← All samples: type, vel, torso, style, paths
├── walk/
│   ├── metadata.json         ← Per-batch parameter summary
│   ├── walk_0000.npz         ← Kimodo format (posed_joints, rot_mats, ...)
│   ├── walk_0000.csv         ← MuJoCo qpos format
│   ├── walk_0001.npz
│   └── ...
├── crouch_walk/
│   ├── metadata.json
│   ├── crouch_walk_0000.npz
│   └── ...
└── ...
```

### Manifest CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `index` | int | Sequential sample ID |
| `motion_type` | str | e.g. "walk", "run" |
| `prompt` | str | Full generated text prompt |
| `duration_s` | float | Motion duration |
| `vx`, `vy`, `wz` | float | Sampled velocity command |
| `torso_height` | float | Sampled normalized height |
| `style` | str | Sampled style modifier |
| `npz_path`, `csv_path` | str | Relative output paths |

---

## Customizing

### Adding a new motion type

Add an entry under `motion_types`:

```yaml
jump:
  description: "a robot jumps forward"
  duration: [0.5, 2.0]
  vel_cmd:
    vx: [0.3, 1.5]
    vy: [0.0, 0.0]
    wz: [0.0, 0.0]
  torso_height: [0.60, 0.95]
  styles: ["energetically"]
  weight: 0.05
  num_samples: 15
```

### Disabling velocity constraints

Set all vel_cmd ranges to `[0.0, 0.0]` — no Root2D constraint will be generated,
and the motion will be purely text-guided.

### Using the Python API directly

```python
from locomotion_framework.config import load_config
from locomotion_framework.sampler import MotionSampler

config = load_config("configs/g1_nromal_loco.yaml")
sampler = MotionSampler(config, seed=42)

# Generate 50 random motions
samples = sampler.generate_random_specs(50)
for s in samples[:5]:
    print(f"[{s.motion_type}] {s.prompt} | vel={s.vel}")
```
