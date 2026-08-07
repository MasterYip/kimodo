# Locomotion Framework — G1 Motion Generation

Distribution-based batch generation for G1 humanoid robot locomotion motions
using Kimodo. Controls velocity (vx, vy, wz), torso height, style, and
motion type via configurable distributions.

---

## Quick Start

```bash
cd /data/masteryip/kimodo/kimodo
source scripts/env.sh

# Dry run — see what will be generated with naming preview
PYTHONPATH=scripts python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_normal_loco.yaml --dry-run --preset rltracker

# Generate full batch (all motion types, single GPU)
PYTHONPATH=scripts python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_normal_loco.yaml --preset rltracker

# Generate exactly 100 motions (random weighted selection)
PYTHONPATH=scripts python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_normal_loco.yaml -n 100

# Use a specific GPU
PYTHONPATH=scripts python3 -m locomotion_framework.orchestrator \
    -c scripts/locomotion_framework/configs/g1_normal_loco.yaml -g 1
```

---

## Distributed Multi-GPU Generation

For large datasets, generation can be split across multiple GPUs to cut
wall-clock time by **N_GPUs×**. Motion types are distributed round-robin
across GPUs, and each GPU launches an independent orchestrator process.

```bash
# All 8 GPUs (default), full config
bash scripts/locomotion_framework/run_distributed.sh

# 4 specific GPUs
bash scripts/locomotion_framework/run_distributed.sh -g 0,1,2,3

# 4 GPUs, 200 random motions
bash scripts/locomotion_framework/run_distributed.sh -g 0,1,2,3 -n 200

# Dry run — print GPU assignment without generating
bash scripts/locomotion_framework/run_distributed.sh --dry-run
```

### How It Works

1. **Config splitting**: The launcher parses the YAML config and distributes
   motion types round-robin across GPUs (e.g. with 4 types on 4 GPUs, each
   GPU gets one type).

2. **Per-GPU processes**: Each GPU runs `CUDA_VISIBLE_DEVICES=<gpu>` with an
   independent `orchestrator.py` process using a temporary config containing
   only its assigned motion types.

3. **Parallel execution**: All processes run concurrently. Since the Kimodo
   model takes ~6 GB VRAM, one process per GPU works well.

4. **Shared output**: All processes write to the same output directory.
   Outputs are independent per motion (named by type + index), so there are
   no write conflicts.

### Performance

| GPUs | Motion types | Wall time (200 motions) |
|------|:-----------:|------------------------:|
| 1 | 4 | ~18 min |
| 4 | 1 each | ~5 min |
| 8 | ½ each | ~3 min |

Single-GPU generation is the bottleneck because the orchestrator processes
motion types sequentially within one process. Distributed mode parallelizes
across types.

### Manual GPU Assignment

To control exactly which types go to which GPU, create multiple config files
and launch them manually:

```bash
# GPU 0 — walk + run
CUDA_VISIBLE_DEVICES=0 python3 -m locomotion_framework.orchestrator \
    -c configs/walk_run.yaml --preset rltracker &

# GPU 1 — stand + squat
CUDA_VISIBLE_DEVICES=1 python3 -m locomotion_framework.orchestrator \
    -c configs/stand_squat.yaml --preset rltracker &

wait
```

---

## Architecture

```
locomotion_framework/
├── configs/
│   └── g1_normal_loco.yaml    ← Distribution definitions
├── config.py                  ← MotionSpec dataclass + YAML loader
├── sampler.py                 ← Distribution sampling (LHS + uniform)
├── constraints.py             ← Velocity → Root2D path builder (dense stride=1)
├── prompts.py                 ← Text prompt composition
├── orchestrator.py            ← Batch runner + manifest output
├── run.sh                     ← Single-GPU quick launch
├── run_full.sh                ← Full 200-motion preset launcher
├── run_distributed.sh         ← Multi-GPU distributed launcher
├── analyze_quality.py         ← Post-generation quality analysis
└── README.md
```

---

## Config File Format

```yaml
global:
  model: kimodo-g1-rp
  diffusion_steps: 100
  seed: 42
  output_dir: outputs/normal_loco
  sampling_method: lhs          # "uniform" or "lhs" (Latin Hypercube Sampling)
  export_preset: rltracker      # "kimodo" or "rltracker" (flat dirs + motion.npz)
  root2d_constraint:
    enabled: true               # legacy default; false keeps command metadata but emits no Root2D
    stride: 1                   # legacy dense path; 2+ constrains every Nth frame plus endpoints

motion_types:
  walk:
    description: "a robot walks"            # base prompt
    duration: [5.0, 8.0]                    # uniform range (seconds)
    vel_cmd:                                # velocity command ranges
      vx: [-0.60, 0.80]   # forward (m/s)
      vy: [-0.50, 0.50]   # lateral (m/s)
      wz: [-0.80, 0.80]   # angular (rad/s)
    torso_height: [0.70, 0.85]              # normalized range
    styles:                                  # style modifiers
      - "normally"
      - "at a steady pace"
    weight: 0.50                             # sampling probability (for -n mode)
    num_samples: 50                          # generated count (for full mode)
```

### Velocity Command Semantics

| Key | Axis | Range | Meaning |
|-----|------|-------|---------|
| `vx` | Forward | positive = forward | Walking/running speed |
| `vy` | Lateral | positive = **left** | Sideways drift |
| `wz` | Angular | positive = counter-clockwise | Turning rate |

> **Sign convention (corrected).** The emitted Root2D constraint is
> `x = vy·t`, `z = vx·t` in the **native Kimodo world frame** where
> `smooth_root_2d` `x` is **+left**, `z` is **+forward**. So positive `vy`
> makes the robot travel +left and negative `vy` makes it travel +right.
> Earlier README text ("positive = right") was the Y30 lateral-sign defect:
> a "right" prompt with a constraint pushing +x (native +left) fights the
> prompt and the model collapses by ignoring the violated dense constraint.
> PORT-008 verified empirically from generated qpos that `vy > 0` == +left.
> Use the **polar** form below to specify directions in the native compass.

### Polar Velocity (native compass)

Planar velocity can be specified in polar coordinates:

```yaml
motion_types:
  walk_left:
    description: "A person walks to the left."
    duration: [8.0, 8.0]
    vel_cmd:
      polar:
        speed: [0.0, 1.5]            # planar speed in m/s
        direction_deg: [75.0, 105.0] # native compass: 0 fwd, +90 LEFT, -90 right
      wz: [0.0, 0.0]
    torso_height: [0.77, 0.77]
    styles: []
    num_samples: 5
```

The sampler converts each draw using

```text
vx = speed * cos(direction_deg)
vy = speed * sin(direction_deg)
```

`direction_deg` uses the **native compass**: 0 = +forward, +90 = **+left**,
-90 = **+right**, 180 = backward. This matches the PORT-008 verified sign
convention. (An older comment "+90 = right" was the Y30 lateral-sign defect;
see above.)

Polar and Cartesian planar fields are mutually exclusive: a `polar` block
cannot be combined with `vx` or `vy`. `wz` remains independent and may be
used with either representation. Direction intervals may use values outside
`[-180, 180]` for sectors crossing the wrap boundary (e.g. `[165, 195]` for
backward), but their span must not exceed 360 degrees. With
`sampling_method: lhs`, speed and direction are stratified independently,
avoiding the radial and angular bias of sampling a Cartesian rectangle.

### Per-speed-band exact prompts (`prompt_speed_bands`)

A single direction group can track the whole speed spectrum with an exact
prompt chosen per sample from the sampled planar speed:

```yaml
    prompt_speed_bands:
      - [0.15, "A person stands still."]
      - [0.45, "A person walks slowly forward."]
      - [0.85, "A person walks forward."]
      - [1.20, "A person walks briskly forward."]
      - [1.51, "A person jogs forward."]
```

The first band whose `max_speed` is strictly above the sampled planar speed
provides the exact prompt (no style/torso/speed hints are appended). This is
how the `g1_natural_distributed_velocity.yaml` config keeps "A person walks
<direction>." natural at 0 m/s (standing) and at >1 m/s (jogging).

---

## Best validated configs

These are the natural-recipe configs that were generated on 4090-3 and
evaluated end-to-end (HHOP = hand-hip-overlap proxy, 0–100 lower = better,
from the pelvis-frame URDF-FK eval CSVs). All share the **frozen natural
recipe**:

- `duration ≥ 8 s` (all `[8.0, 8.0]`)
- exact per-speed prompts (`prompt_speed_bands`) that track the speed spectrum
- native-compass sign (0° = forward, +90° = left, −90° = right — the
  PORT-008-verified convention; see `PolarVelRange`)
- separated CFG `[2.0, 2.0]` (`cfg_type: separated`, `cfg_weight: [2.0, 2.0]`)
- a fixed seed, `styles: []`, `model: kimodo-g1-rp`, `fps: 30`,
  `diffusion_steps: 100`, `export_preset: rltracker`
- **constraint channel**: the framework's Root2D constraint is strictly 2-D
  (`smooth_root_2d[..., [0, 1]]` — height is dropped), so **posture reaches
  the model only through the text hint** (`_torso_height_hint` in `prompts.py`
  maps normalized torso → "crouching very low"/"crouching"/"slightly
  crouching"/none). A `root_y_pos` height channel exists in native Kimodo but
  is emitted only by the heavy `fullbody` constraint type; the validated
  configs below do **not** use it.

### `g1_normal_loco.yaml` — baseline grouped config (reference)

The canonical grouped config (`global` + `motion_types:` groups). Sample
`g1_normal_loco.yaml` with LHS for a mixed-type reference batch (e.g. 200
samples). It is the structural template for every config below; not a
distribution-targeted batch.

### `g1_eight_direction_port_008.yaml` — PORT-008 natural recipe

The FRAMEWORK-PORT-008 reproduction of the DISTRIBUTED-007 40-motion envelope
(5 cells × 8 directions, seed 20260805, 8 s). Native compass, exact prompts,
separated CFG. User verdict after playback: **"more natural"**. Batch HHOP
**mean 0.8, 0/40 flagged** (vs native DISTRIBUTED 3.1); per-cell E0 1.4 /
E1 0.7 / E2 0.8 / E3 1.1 / E4 0.0. Use it to reproduce the PORT-008 evaluation
envelope exactly.

### `g1_natural_distributed_velocity.yaml` — VELOCITY-DISTRIBUTION-009

Distributes **planar velocity** only: 8 direction groups × 5 LHS speed samples
over `[0.0, 1.5]` m/s = **40 motions**. Per-speed-band exact prompts track
stand → slow → walk → brisk → jog. Natural recipe with seed 20260807. Batch
HHOP **mean 2.79, median 0.00, 26/40 zero** (eval CSV
`velocity_distribution_eval.csv`); all 10 speed histogram bins over `[0,1.5]`
populated. Use it when you want a full speed-spectrum velocity distribution in
all 8 directions with a single fixed torso.

### `g1_natural_distributed_torso_velocity.yaml` — TORSO-HEIGHT-DISTRIBUTION-013

**2-D distribution**: planar velocity `[0.0, 1.5]` m/s **×** torso_height
`[0.45, 0.8]` (normalized; 0.77 = neutral standing, 0.45 ≈ deep crouch), 8
direction groups × 8 **2-D LHS** samples = **64 motions**. Seed 20260808.
`prompt_speed_bands` (same per-speed exact prompts) **+ the torso-hint
composition**: when the sampled torso is off-neutral, `_torso_height_hint` is
appended to the band prompt (e.g. `"A person walks forward, crouching very
low."`); neutral 0.77 configs emit no hint and are byte-identical to
VELOCITY-009. Batch HHOP **mean 1.04, median 0.00, 56/64 zero**; low-torso band
(commanded ≤ 0.55) HHOP **0.00 (18/18)**; head_z **min 0.340 m** (no prone
collapse — the falsifiable test of the old 0.45–0.55 concern is **not**
reproduced). **Achieved root/pelvis height** (from qpos, normalized) **spreads
0.352–0.785** with **r = 0.728 vs commanded** — the distribution contract is
met (not bunched at neutral). Eval CSV `torso_velocity_eval.csv`. Use it when
you want the velocity AND torso_height both well-distributed.

> Note: achieved heights at the low end (commanded 0.45–0.51) land around
> 0.35–0.60 — the model crouches from the text hint but does not mechanically
> hit the commanded value. The spread is real and reproducible; a precise
> commanded-height mapping would require a `root_y_pos` constraint channel
> (Path B, PM-reviewed).

---


### Root2D Constraint Settings

`global.root2d_constraint.enabled` decouples sampled velocity metadata and
prompt construction from Root2D injection. It defaults to `true`, preserving
legacy behavior. `stride` defaults to `1`; values above one retain the first
and last frame and constrain every Nth frame between them. The G1 skeleton
root is the pelvis, so the locomotion framework does not expose separate
root-only and pelvis-only factors. It also injects no upper-body constraint.
## How It Works

### 1. Distribution Sampling

For each motion type, the sampler draws from the configured distributions:

- **Latin Hypercube Sampling (LHS)**: Stratified sampling across all 5
  dimensions (vx, vy, wz, torso_height, duration) for uniform coverage.
  Recommended for full coverage datasets.

- **Uniform random**: Independent uniform draws per parameter. Used when
  `sampling_method: uniform` is set.

Sampled parameters:
- Duration: `U(duration[0], duration[1])`
- Velocity: `U(vx[0], vx[1])`, `U(vy[0], vy[1])`, `U(wz[0], wz[1])`
- Polar velocity: `U(speed[0], speed[1])` and
  `U(direction_deg[0], direction_deg[1])`, converted to `vx/vy`
- Torso height: `U(torso_height[0], torso_height[1])`
- Style: uniform random from `styles` list

### 2. Velocity → Root2D Path

Sampled velocity commands are converted to a 2D root trajectory using the
demo `05_root_path` method:

- **Dense stride=1 constraints**: Every frame is constrained, matching the
  Kimodo demo approach. The diffusion model with hard inpainting at each
  DDIM step produces clean motions without tail jitter.

- **Straight-line**: `x(t) = vy · t · dt` (lateral, +left), `z(t) = vx · t · dt`
  (forward) in the native Kimodo world frame

- **Arc turning**: Circular arc with heading constraints via
  `global_root_heading`

- **Exact duration**: Motions are generated at exactly the sampled duration
  (no margin/truncation needed), since dense inpainting doesn't require
  settling time.

Constraints are passed to Kimodo as `Root2DConstraintSet` with per-sample
list API (`constraint_lst=list[list]`).

### 3. Text Prompt Composition

Parameters are composed into a single prompt:

```
"a robot walks slowly at a steady pace."
"a robot runs at a brisk pace normally slightly crouching."
```

Torso height is mapped to qualitative hints:
- `< 0.50`: "crouching very low"
- `0.50–0.60`: "crouching"
- `0.60–0.70`: "slightly crouching"
- `0.70–0.85`: (normal — no hint)
- `> 0.85`: "standing tall"

### 4. Batch Generation

The orchestrator loads Kimodo once per motion type, then:
1. Builds per-sample prompts, frames, and dense Root2D constraints via the
   Kimodo **list API** (each sample gets its own unique velocity path)
2. Calls `model(prompts, num_frames, constraint_lst=..., num_denoising_steps=N)`
3. Trims batch-padding frames (Kimodo pads all outputs to `max(num_frames)`)
4. Exports each sample in the selected preset format

Kimodo processes all samples of one motion type in a single denoising loop.

---

## Export Presets

### `rltracker` (default)

Flat directory structure, one `motion.npz` per motion:

```
outputs/normal_loco/
├── manifest.csv
├── walk_fwd_028_norm_0003__K42/
│   └── motion.npz          ← 7 keys
├── walk_lat_295_norm_0000__K42/
│   └── motion.npz
├── run_fwd_001_fast_0001__K42/
│   └── motion.npz
├── stand_still_000_still_0000__K42/
│   └── motion.npz
└── ...
```

Naming: `{type}_{traj}_{heading:03d}_{pace}_{var:04d}__K{seed}`

| Component | Values | Source |
|-----------|--------|--------|
| `type` | `walk`, `run`, `stand`, `squat` | `motion_type` from config |
| `traj` | `fwd`, `bwd`, `lat`, `turn`, `still` | computed from (vx, vy, wz) |
| `heading` | `000`–`359` | `int(round(atan2(vy, vx)*180/pi)) % 360` |
| `pace` | `slow`, `norm`, `fast`, `sprt`, `still` | from `|vx|` magnitude |
| `var` | `0000`–`9999` | sequential per type |
| `seed` | global config seed | `K{seed}` (K = Kimodo origin) |

NPZ keys (float32, Kimodo native 30 fps):
| Key | Shape | Description |
|-----|-------|-------------|
| `fps` | `(1,)` | Frames per second |
| `joint_pos` | `(T, 29)` | 29 hinge joint angles (radians) |
| `joint_vel` | `(T, 29)` | Joint angular velocities |
| `body_pos_w` | `(T, 30, 3)` | 30 body world positions (m, MuJoCo: z-up x-forward) |
| `body_quat_w` | `(T, 30, 4)` | 30 body world quaternions (w,x,y,z) |
| `body_lin_vel_w` | `(T, 30, 3)` | Body linear velocities |
| `body_ang_vel_w` | `(T, 30, 3)` | Body angular velocities |

### `kimodo` (legacy)

Per-type directories with NPZ + CSV:

```
outputs/locomotion/
├── manifest.csv
├── walk/
│   ├── metadata.json
│   ├── walk_0000.npz         ← Kimodo format (posed_joints, rot_mats, ...)
│   ├── walk_0000.csv         ← MuJoCo qpos format
│   └── ...
└── ...
```

---

## Manifest CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `index` | int | Sequential sample ID |
| `motion_type` | str | e.g. "walk", "run" |
| `prompt` | str | Full generated text prompt |
| `duration_s` | float | Motion duration |
| `vx`, `vy`, `wz` | float | Sampled velocity command |
| `torso_height` | float | Sampled normalized height |
| `style` | str | Sampled style modifier |
| `path` | str | Relative path to `motion.npz` |

---

## Quality Analysis

```bash
# Run on generated output
PYTHONPATH=scripts python3 scripts/locomotion_framework/analyze_quality.py outputs/normal_loco
```

Metrics computed:
- **T/M**: Tail/Mid velocity ratio — should be ~1.0 (no deceleration at tail)
- **JitT/H**: Tail/Head jitter ratio — < 2x is excellent, 2-3x is good, > 5x is bad
- Per-type summaries and distribution breakdowns saved to `quality_analysis.json`

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

Set all vel_cmd ranges to `[0.0, 0.0]` — no Root2D constraint will be
generated, and the motion will be purely text-guided.

### Using the Python API directly

```python
from locomotion_framework.config import load_config
from locomotion_framework.sampler import MotionSampler

config = load_config("configs/g1_normal_loco.yaml")
sampler = MotionSampler(config, seed=42)

# Generate 50 random motions
samples = sampler.generate_random_specs(50)
for s in samples[:5]:
    print(f"[{s.motion_type}] {s.prompt} | vel={s.vel}")
```

## Exact prompts, directional arm intent, and evaluation

Each motion type may set `prompt` to a non-empty literal final prompt. It
bypasses speed, style, torso-height, and arm-swing text composition. Otherwise,
`arm_swing: auto|sagittal|lateral|none` adds prompt-only intent. `auto` maps
forward/backward (`|vx| >= |vy|`) to alternating forward/backward swing and
left/right travel to alternating left/right swing with slight fore-aft
clearance. These fields are text intent, not physical arm constraints.

Evaluate any native `rltracker` output root deterministically:

```bash
PYTHONPATH=scripts python -m locomotion_framework.evaluation.cli \
  /path/to/framework/output \
  --output-dir /path/to/evaluation \
  --per-motion-plots \
  --comparison y0_disabled --comparison y2_sparse4
```

The evaluator writes CSV/JSON, optional per-motion wrist plots, and a compact
comparison panel. Coordinates are pelvis-local `+X` forward, `+Y` left/lateral,
`+Z` up after inverse pelvis yaw. Arm/hip/torso and wrist-overlap distances use
body/link centers; they are diagnostic proxies, not mesh-collision evidence.

CSV/JSON evaluation requires only NumPy. Plotting is an optional feature and
requires Matplotlib; when the generation environment omits it, run the same CLI
against the checksum-verified output on an analysis environment with
Matplotlib installed.
