# G1 Locomotion Fine-Grained Motion Generation — Development Report

**Date**: 2026-07-20  
**Author**: Kimodo G1 Locomotion Framework  
**Location**: `/data/masteryip/kimodo/kimodo/docs/dev_log/20260720 FineGrainedMoGen/`

---

## 1. Summary

We deployed a distribution-based batch generation framework (Latin Hypercube Sampling + Kimodo diffusion) producing 200 G1 locomotion motions across 4 types (walk / run / stand / squat). After visualization, two issues were identified:

| # | Symptom | Root Cause |
|---|---------|-----------|
| A | Walk motions drift sideways/backward instead of going forward | **Axis swap in constraint builder**: `vx` (forward velocity) is mapped to Kimodo's X-axis instead of Z-axis |
| B | Root trajectory jitters at motion tail | **Constraint–diffusion conflict**: cosine-eased deceleration in the Root2D path forces the root position against the model's walking-prior joint motion |

---

## 2. Principle of Kimodo Constraints

### 2.1 How Root2D constraints work

Kimodo's `Root2DConstraintSet` specifies desired **world-frame (x, z) root positions** at a subset of frames, with optional `global_root_heading` for orientation.

```
constraint = {
    "type": "root2d",
    "frame_indices": [0, 1, 2, ..., N-1],    # which frames are constrained
    "smooth_root_2d": [[x0,z0], [x1,z1], ...], # (x,z) world position per frame
    "global_root_heading": [[cos_θ0,sin_θ0], ...],  # optional: heading per frame
}
```

Key design properties:
- **Sparse or dense**: can constrain any subset of frames. Unconstrained frames are free.
- **Not hard post-processing**: constraints are encoded into the diffusion latent via masking (`observed_motion` + `motion_mask`), acting as **inpainting-style conditioning**. The diffusion process tries to satisfy both the constraint AND the learned motion prior simultaneously.
- **Conflict resolution**: when constraint and prior disagree, the result is a compromise — often unstable (jitter, foot sliding).

### 2.2 Kimodo coordinate system

```
Kimodo:   Y-up, Z-forward, X-right    (right-handed)
MuJoCo:   Z-up, X-forward, Y-left     (right-handed)

Transformation matrix (Kimodo → MuJoCo):
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]]

Kimodo (x, y, z)  →  MuJoCo (y_kim, z_kim, x_kim)
                     = MuJoCo (y, z, x)    where x=fwd, y=left, z=up
```

**`smooth_root_2d` uses the Kimodo (x, z) plane**: x = lateral, z = forward.

### 2.3 Velocity → Trajectory mapping

For a sampled velocity command `(vx_fwd, vy_lat, wz_ang)` with duration T at 30 fps:

- **Straight** (`wz ≈ 0`): simple integration
  ```
  x_K(t) = ∫ v_lat · speed(t) dt       ← Kimodo X (lateral)
  z_K(t) = ∫ v_fwd · speed(t) dt       ← Kimodo Z (forward)
  ```
  `smooth_root_2d = [[x_K(0), z_K(0)], [x_K(1), z_K(1)], ...]`

- **Curved** (`wz ≠ 0`): velocity rotates at rate wz
  ```
  θ(t) = θ₀ + wz·t                      ← world heading at frame t
  v_x(t) = v_fwd·sin(θ) + v_lat·cos(θ) ← world-frame X velocity
  v_z(t) = v_fwd·cos(θ) - v_lat·sin(θ) ← world-frame Z velocity
  x_K(t) = ∫ v_x(t)·speed(t) dt
  z_K(t) = ∫ v_z(t)·speed(t) dt
  ```
  `global_root_heading = [[cos(θ(t)), sin(θ(t))], ...]`

---

## 3. Bug A: Axis Swap — Walk Drifts Left

### 3.1 What went wrong

In `constraints.py:build_root2d_constraint()`, the forward velocity `vx` was mapped to Kimodo's X-axis, and lateral velocity `vy` to Kimodo's Z-axis — the **opposite** of the correct mapping.

**Buggy code** (lines ~90–91, now fixed):
```python
x = np.cumsum(vx * speed) * dt    # vx (forward) → Kimodo X (lateral) ← WRONG
z = np.cumsum(vy * speed) * dt    # vy (lateral) → Kimodo Z (forward) ← WRONG
```

**Verification** for `walk_fwd_006_norm_0004` (vx=+0.684, vy=+0.078, dur=6.1s):

| Metric | Buggy (vx→X) | Fixed (vx→Z) | Actual root |
|--------|-------------|-------------|-------------|
| MJ X (forward, m) | 0.47 | 4.13 | -1.34 |
| MJ Y (left, m) | 4.13 | 0.47 | 3.12 |
| Interpretation | Goes LEFT 4.1m | Goes FORWARD 4.1m | Goes LEFT |

The actual root went 3.1m left and 1.3m backward — the buggy constraint ordered the robot to go left, and the model obeyed.

### 3.2 Impact on curved paths

The curved-path code (`wz ≠ 0` branch) had the same axis swap in the velocity-decomposition step:
```python
vx_world = vx * np.cos(theta) - vy * np.sin(theta)   # forward → X (wrong)
vz_world = vx * np.sin(theta) + vy * np.cos(theta)   # lateral → Z (wrong)
```

Every motion with `|wz| > 1e-6` (approximately 75% of walk samples) had its primary direction rotated 90° from intended.

### 3.3 Impact on heading

The world heading `θ(t) = atan2(vy, vx) + wz·t` used `arctan2(lateral, forward)`, which computes heading from the **body-frame velocity direction**. With the axis swap, this heading was effectively 90° offset, causing `global_root_heading` constraints to point in wrong directions for all curved trajectories.

### 3.4 Fix

Swap the assignment:
```python
x = np.cumsum(vy * speed) * dt    # vy (lateral) → Kimodo X ✓
z = np.cumsum(vx * speed) * dt    # vx (forward) → Kimodo Z ✓
```

For curved paths, correct the rotation decomposition to use proper forward/lateral roles.

---

## 4. Bug B: Tail Root Jitter

### 4.1 What was observed

In the viewer, the robot's root position oscillates and drifts vertically at the last ~15% of each motion. Example from `run_fwd_000_sprt_0027`:

```
Frame  velocity(m/s)  height(m)
223    0.92           0.781
224    1.16           0.769
225    2.31           0.769
226    2.90           0.772
227   17.90 ***SPIKE  0.768   ← single-frame 18 m/s jump
228    2.00           0.793   ← height spike
229    0.98           0.781
...
237    1.72           0.740   ← 4cm height drift in 14 frames
```

### 4.2 Why it happens: soft-constraint conflict

Kimodo constraints are **soft conditioning** via diffusion inpainting, not hard kinematics.

The pipeline:
```
Constraint path (x,z,t)  →  encoded as observed_motion + motion_mask
                          →  diffusion denoising conditioned on mask
                          →  generated joints + root must reconcile:
                              (a) constraint says "root is at position P" 
                              (b) motion prior says "walking at speed V needs 
                                  foot placements at distance D"
```

During **constant-velocity cruise** (frames 0–85%), (a) and (b) agree: root moves at steady speed, joints generate walking at that speed. Signal is dominated by the valid-conditioned diffusion.

During **deceleration** (frames 85–100%): constraint says root velocity → 0, but the learned walking prior from Kimodo's training data (which contains mostly loop/walk motions, not stopping motions) generates limb kinematics appropriate for the cruising speed. The diffusion process receives conflicting signals:

- **Constraint dimension**: root must be near-constant position
- **Free dimensions** (joint angles, foot contacts): diffusion continues generating walking-appropriate values based on the prompt "a robot walks forward"

The 7-dimensional motion representation (root position + local rotations + foot contacts + heading) cannot simultaneously satisfy a stopped root and walking limb kinematics — the latent space has no representation for a robot that "walks in place." The compromise is a root that oscillates as the model alternates between satisfying the constraint and the motion prior.

### 4.3 Why deceleration specifically triggers it

Kimodo was trained on motion-capture data that predominantly contains:
- **Loop motions** (continuous walking/running cycles)
- **Start motions** (stand → walk transition)
- **Stop motions** (walk → stand transition) — but fewer of these

The model's learned manifold has a well-defined "walking at speed V" region but a poorly-defined "decelerating from V to 0" region. When the constraint forces traversal through this under-represented region, the diffusion process lacks a strong prior to regularize the output, leading to instability.

### 4.4 Why the 85%-constrained approach didn't work

Constraining only frames 0–85% and leaving the tail free creates a **sharp transition at the constraint boundary**: at frame N_constraint, the root must match the constraint; at frame N_constraint+1, no constraint applies. The diffusion model, conditioned on the full sequence, must interpolate between the constrained and unconstrained regions — but the transition is too abrupt for the model to resolve smoothly.

### 4.5 Recommended fix

**Constrain the constant-velocity phase only, zero constraint on deceleration**:

Instead of running constrained frames 0–85% with a 15% free tail, omit all tail frames from the constraint and let the prompt describe the ending behavior:

```yaml
# Add a "walk_stop" motion type to the config
walk_stop:
  description: "a robot walks forward then comes to a stop"
  duration: [5.0, 8.0]
  vel_cmd:
    vx: [0.20, 0.80]
    vy: [-0.10, 0.10]
    wz: [-0.10, 0.10]
  torso_height: [0.60, 0.85]
  styles:
    - "normally"
    - "smoothly coming to a halt"
  num_samples: 20
```

The constraint should cover the **walking phase** and let the text prompt guide the stop. Alternatively, use Kimodo's **multi-prompt mode**: generate two segments (walk + stop) with smooth transitions enabled.

### 4.6 Alternative: abandon tail constraints entirely

For maximum quality at the cost of exact velocity control, constrain only a sparse set of waypoints (every 10–20 frames) rather than every frame. This gives the diffusion process more freedom to generate natural motion while still guiding the overall trajectory.

---

## 5. Current Status & Next Steps

### 5.1 What works
- LHS sampling for uniform parameter coverage across 5 continuous dimensions
- Per-sample constraint generation via Kimodo's list API
- IsaacLab DOF/body order permutation matching reference dataset
- Native 30fps output (no resampling artifacts)
- RLTracker-compatible NPZ format

### 5.2 Requires fix
- **Axis swap** (Section 3): simple coordinate mapping fix in `constraints.py`
- **Tail jitter** (Section 4): replace frame-level tail constraints with: (a) prompt-guided ending, or (b) sparse waypoint constraints, or (c) unconstrained tail region

### 5.3 Design recommendations
1. Always constrain fewer frames than generated (constrain_frac ≤ 0.80)
2. Add "stop" and "start" motion types with appropriate text prompts
3. Consider adding `multi_prompt=True` mode for composite motions (stand→walk→stop)
4. Validate constraint paths against actual root trajectories with automated checks
