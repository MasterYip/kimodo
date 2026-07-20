# G1 Locomotion Motion Generation Framework

## Context

Kimodo is installed on 4090-3 with the NF4 text encoder working.
The goal is a distribution-based batch generation framework for G1
locomotion motions with fine-grained velocity, torso height, and style control.

Kimodo's constraint system supports:
- **Root2DConstraintSet**: fix (x,z) root trajectory + heading at any frames
- **FullBodyConstraintSet**: fix full joint poses at keyframes
- **EndEffectorConstraintSet**: fix hand/foot positions
- **Text prompts**: control action type, style, qualitative modifiers

### Key Finding: Velocity → Root2D Path

For a sampled `(vx, vy, wz)` and duration `T`:
- Build a straight/curved path `(x(t), z(t))` for `t in 0..N-1` (N = T × 30fps)
- For straight: `x(t) = vx*t*dt, z(t) = vy*t*dt`
- For turning: add heading rotation via `global_root_heading`
- Create `Root2DConstraintSet` with the computed path

### Key Constraint: One Batch = One Motion Type Per Call

Kimodo's `num_samples` applies one prompt+constraint set to N samples in parallel.
Different motion types require separate calls. Multi-prompt mode chains segments
within one motion.

---

## Architecture

```
kimodo/                           # repo root on 4090-3
└── scripts/
    └── locomotion_framework/     # NEW directory
        ├── __init__.py
        ├── config.py             # MotionSpec dataclass + YAML loading
        ├── sampler.py            # Distribution sampling
        ├── constraints.py        # Root2D path builder from velocities
        ├── prompts.py            # Text prompt builder (action + style)
        ├── orchestrator.py       # Batch generation runner + metadata
        ├── configs/              # Example YAML configuration files
        │   └── g1_locomotion.yaml
        └── README.md
```

### 1. Motion Specification (config.py)

```python
@dataclass
class VelRange:
    """Uniform distribution over [min, max]."""
    min: float
    max: float

@dataclass
class MotionSpec:
    """Defines a distribution for one motion type."""
    name: str
    description: str             # base text prompt
    duration: tuple[float, float]  # (min, max) seconds
    vel_cmd: dict                 # {"vx": VelRange, "vy": VelRange, "wz": VelRange}
    torso_height: tuple[float, float]  # (min, max) normalized
    styles: list[str]             # ["casually", "briskly", ...]
    weight: float                 # sampling probability
    num_samples: int              # how many from this type per batch
```

### 2. Sampler (sampler.py)

- `sample_type(config, rng)` → picks motion type weighted by `weight`
- `sample_params(spec, rng)` → samples concrete vel, duration, style, torso_height
- `generate_batch_specs(config)` → creates list of (type, params) for a batch

### 3. Path Builder (constraints.py)

```python
def vel_to_root2d_constraint(vx, vy, wz, duration, fps=30):
    """Convert velocity command to Root2D path constraint.
    
    Builds (x,z) trajectory frame-by-frame. For wz (angular vel),
    the path curves and global_root_heading is set.
    Returns a dict ready for Kimodo's constraint format.
    """
    
def build_constraints_json(sampled_params, fps=30):
    """Build full constraints.json from sampled parameters."""
```

### 4. Prompt Builder (prompts.py)

```python
def build_motion_prompt(action, style, torso_height=None):
    """Compose text prompt: 'a robot {action} {style}. [{height hint}]'
    
    Torso height is mapped to qualitative hints:
    - low (< 0.7): "crouching low"
    - normal (0.7-0.85): ""  
    - high (> 0.85): "standing tall"
    """
```

### 5. Orchestrator (orchestrator.py)

```python
def generate_locomotion_batch(config_path, output_dir, seed=None):
    """Full pipeline:
    1. Load YAML config
    2. Sample N specs from distributions
    3. For each spec: build prompt + constraints
    4. Call kimodo_gen for each spec (or use Python API for shared model)
    5. Save .npz + .csv + metadata.json per sample
    6. Save manifest.csv summarizing all samples
    """
```

### 6. Config File Format (configs/g1_locomotion.yaml)

```yaml
global:
  model: kimodo-g1-rp
  diffusion_steps: 100
  seed: 42
  output_dir: outputs/locomotion

motion_types:
  stand:
    description: "a robot stands still"
    duration: [2.0, 5.0]
    vel_cmd:
      vx: [-0.05, 0.05]
      vy: [-0.02, 0.02] 
      wz: [-0.05, 0.05]
    torso_height: [0.70, 0.90]
    styles: ["idle", "at attention", "looking around"]
    weight: 0.15
    num_samples: 20

  walk:
    description: "a robot walks forward"
    duration: [3.0, 8.0]
    vel_cmd:
      vx: [0.2, 0.8]
      vy: [-0.1, 0.1]
      wz: [-0.15, 0.15]
    torso_height: [0.65, 0.85]
    styles: ["casually", "briskly", "with long strides"]
    weight: 0.35
    num_samples: 50

  run:
    description: "a robot runs forward"
    duration: [2.0, 5.0]
    vel_cmd:
      vx: [0.8, 2.0]
      vy: [-0.05, 0.05]
      wz: [-0.1, 0.1]
    torso_height: [0.60, 0.80]
    styles: ["quickly", "sprinting"]
    weight: 0.10
    num_samples: 20

  walk_wave:
    description: "a robot walks forward while waving its right hand"
    duration: [3.0, 6.0]
    vel_cmd:
      vx: [0.2, 0.6]
      vy: [-0.1, 0.1]
      wz: [-0.1, 0.1]
    torso_height: [0.65, 0.85]
    styles: ["enthusiastically", "casually"]
    weight: 0.15
    num_samples: 25

  crouch_walk:
    description: "a robot crouches and walks forward carefully"
    duration: [2.0, 5.0]
    vel_cmd:
      vx: [0.1, 0.4]
      vy: [-0.05, 0.05]
      wz: [-0.1, 0.1]
    torso_height: [0.45, 0.65]
    styles: ["stealthily", "carefully"]
    weight: 0.10
    num_samples: 15

  sidestep:
    description: "a robot steps sideways"
    duration: [1.0, 3.0]
    vel_cmd:
      vx: [-0.1, 0.1]
      vy: [0.1, 0.5]
      wz: [-0.05, 0.05]
    torso_height: [0.65, 0.85]
    styles: ["carefully", "quickly"]
    weight: 0.15
    num_samples: 20
```

### Output Structure

```
outputs/locomotion/
├── manifest.csv            # all samples: type, vx, vy, wz, torso, style, path
├── stand/
│   ├── stand_00.npz
│   ├── stand_00.csv
│   ├── ...
│   └── metadata.json       # per-type parameter summary
├── walk/
│   ├── walk_00.npz
│   ...
└── walk_wave/
    ├── walk_wave_00.npz
    ...
```

---

## Key Design Decisions

### 1. Shared model loading (not CLI per call)

The orchestrator loads the Kimodo model once via Python API, then calls
`model(prompt, num_frames, constraint_lst=..., num_samples=N)` for each
motion type. This avoids reloading the model for each type and keeps
the text encoder in GPU memory.

### 2. Constraint file per call (not one massive file)

Each motion type gets its own set of Root2D constraints built from sampled
velocity parameters. The constraints are passed as Python objects directly
to the model API (no intermediate JSON file).

### 3. Torso height via text (not full-body constraints)

Kimodo doesn't have direct torso height control. We map height ranges to
text modifiers ("crouching low", "standing tall") which Kimodo interprets.
For stricter control in the future, we can add FullBodyConstraintSet at
the start frame.

### 4. Batch parallelism

Within one motion type, `num_samples` handles parallel generation.
For multi-GPU across types, we use Python `multiprocessing` with one
GPU per worker.

---

## Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/locomotion_framework/__init__.py` | 5 | Package init |
| `scripts/locomotion_framework/config.py` | 100 | MotionSpec, load YAML |
| `scripts/locomotion_framework/sampler.py` | 80 | Distribution sampling |
| `scripts/locomotion_framework/constraints.py` | 120 | Velocity→Root2D path |
| `scripts/locomotion_framework/prompts.py` | 60 | Text prompt composition |
| `scripts/locomotion_framework/orchestrator.py` | 200 | Batch runner + metadata |
| `scripts/locomotion_framework/configs/g1_locomotion.yaml` | 80 | Default config |
| `scripts/locomotion_framework/README.md` | 100 | Usage docs |

## Dependencies

All already installed in the venv: `numpy`, `torch`, `pyyaml`, `kimodo`

## Verification

1. `python3 -m locomotion_framework.orchestrator --config configs/g1_locomotion.yaml --dry-run`
   → prints sampled specs without generating
2. `python3 -m locomotion_framework.orchestrator --config configs/g1_locomotion.yaml --num-total 10`
   → generates 10 total motions across types, verifies outputs
3. `python3 -m locomotion_framework.orchestrator --config configs/g1_locomotion.yaml`
   → full batch generation (150 motions by default), check manifest.csv
