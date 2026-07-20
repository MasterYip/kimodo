# G1 Locomotion Fine-Grained Motion Generation — Development Report

**Date**: 2026-07-20
**Author**: Kimodo G1 Locomotion Framework

---

## 1. How the Root Constraint Path Is Built

### 1.1 Sampled velocity command

The sampler (LHS) draws 5 continuous parameters from configured ranges:

```
vx   = forward velocity (m/s, body-frame)
vy   = lateral velocity (m/s, body-frame, positive = right)
wz   = angular velocity (rad/s, positive = counterclockwise / turning right)
duration = motion length (s)
torso_height = normalized (0.35–0.90)
```

### 1.2 Speed profile — pure constant velocity

For **pure locomotion clips** (no start/end phase), the constraint uses constant velocity from frame 0 to frame N−1 with no cosine easing:

```
s[t] = 1.0    for all frames t ∈ [0, N−1]
```

No acceleration phase. No deceleration phase. The model sees root moving at steady speed from the first frame to the last.

### 1.3 Integration — straight paths (|wz| < 1e-6)

```
Kimodo X (lateral)  = vy · t · dt
Kimodo Z (forward)  = vx · t · dt
smooth_root_2d      = [[X₀,Z₀], [X₁,Z₁], ..., [Xₙ₋₁,Zₙ₋₁]]
```

Kimodo coordinate convention: Y-up, Z-forward, X-right.

#### Axis mapping proof

Code at `kimodo_motionrep.py:246-248`:
```python
f_sliced = observed_motion[:, slice("smooth_root_pos")]   # (T, 3)
f_sliced[indices, 0] = smooth_root_2d[:, 0]               # dim 0 = Kimodo X = lateral
f_sliced[indices, 2] = smooth_root_2d[:, 1]               # dim 2 = Kimodo Z = forward
```

Therefore:
- `smooth_root_2d[:, 0]` must be **lateral** position → driven by `vy` ✓
- `smooth_root_2d[:, 1]` must be **forward** position → driven by `vx` ✓

### 1.4 Integration — curved paths (|wz| > 1e-6)

Body-frame velocity `(vx, vy)` with rotation rate `wz` traces a circular arc. Rotating body → world at each frame:

```
θ(t)     = atan2(vy, vx) + wz·t
v_X_w(t) = vy·cos θ + vx·sin θ      [world-frame lateral velocity]
v_Z_w(t) = vx·cos θ − vy·sin θ      [world-frame forward velocity]

Kimodo X (lateral)  = Σ v_X_w · dt   [cumulative sum]
Kimodo Z (forward)  = Σ v_Z_w · dt

global_root_heading = [[cos θ, sin θ], ...] at constrained frames
```

#### Derivation of the rotation formulas

Kimodo uses Y-up, Z-forward, X-right (right-handed). Heading θ is the angle of body Z-axis from world Z-axis, measured toward world X-axis.

Rotation around Y-axis by θ maps body → world:
```
| cos θ   0   sin θ |   |X_body|     |cos θ·X_body + sin θ·Z_body|
|   0     1     0   | × |Y_body|  =  |           Y_body           |
| −sin θ  0   cos θ |   |Z_body|     |−sin θ·X_body + cos θ·Z_body|
```

Substituting body_X = vy (lateral), body_Z = vx (forward):
```
X_world = vy·cos θ + vx·sin θ         (matches code ✓)
Z_world = vx·cos θ − vy·sin θ         (matches code ✓)
```

### 1.5 Which frames are constrained

Only every 3rd frame is constrained (stride=3, ~34% of frames), plus the first and last frames always. This gives the diffusion model freedom to add natural COM oscillation between waypoints while following the overall trajectory:

```
frame_indices = [0, 3, 6, 9, ..., N-3, N-1]
```

---

## 2. How Constraints Enter Kimodo

### 2.1 Orchestrator → Kimodo API

```python
# orchestrator.py:_generate_batch, line 276
output = model(
    per_prompts,                # list[str] — one text prompt per sample
    per_frames,                 # list[int] — one duration (in frames) per sample
    constraint_lst=kimodo_constraints,  # list[list[KimodoConstraint]]
    num_denoising_steps=100,    # DDIM steps
    return_numpy=True,          # return numpy arrays (not torch tensors)
)
```

`constraint_lst` is a **list of lists**: outer list = per sample, inner list = constraints for that sample. Kimodo detects per-sample mode when `isinstance(constraints_lst[0], list)` is True (`base.py:275`).

### 2.2 Constraint → motion representation encoding

Kimodo calls `create_conditions_from_constraints_batched()` → `create_conditions()` in `kimodo_motionrep.py:222`.

The function builds two tensors:
- `observed_motion` (T, motion_rep_dim): filled with constraint values at constrained positions, zero elsewhere
- `motion_mask` (T, motion_rep_dim, bool): True at constrained positions

**Root2D (smooth_root_2d) encoding** (lines 242–251):
```python
observed_motion = zeros(T, motion_rep_dim)
motion_mask     = zeros(T, motion_rep_dim, dtype=bool)

# Write (X, Z) positions at constrained frame indices
f_sliced = observed_motion[:, slice("smooth_root_pos")]   # (T, 3) slice
f_sliced[indices, 0] = smooth_root_2d[:, 0]               # Kimodo X (lateral)
f_sliced[indices, 2] = smooth_root_2d[:, 1]               # Kimodo Z (forward)
m_sliced = motion_mask[:, slice("smooth_root_pos")]
m_sliced[indices, 0] = True   # X is constrained
m_sliced[indices, 2] = True   # Z is constrained
```

Key: **Root Y (height) is NEVER constrained** by Root2D — only X and Z. Diffusion freely generates COM height.

**Heading (global_root_heading) encoding** (lines 262–269):
```python
f_sliced = observed_motion[:, slice("global_root_heading")]   # (T, 2)
f_sliced[indices] = global_root_heading                       # [cos θ, sin θ]
m_sliced = motion_mask[:, slice("global_root_heading")]
m_sliced[indices] = True
```

Only included when `|wz| > 1e-6` (curved or standing paths).

### 2.3 Diffusion conditioning mechanism

At each of the 100 DDIM denoising steps:

1. The diffusion latent is passed through the model (UNet-style)
2. At constrained positions: the latent value is **substituted** with `observed_motion` (inpainting)
3. At unconstrained positions: the diffusion process freely generates

This is **soft conditioning via inpainting**, NOT hard post-processing. The model must simultaneously satisfy constraint values AND the learned motion prior. When they agree (constant-velocity walk → "a robot walks"), the result is coherent. When they conflict, the result oscillates.

### 2.4 Constraint coverage summary

| Component | Constrained? | Notes |
|-----------|-------------|-------|
| Root (X, Z) | Yes | Via smooth_root_2d, every 3rd frame (34% coverage) |
| Root Y (height) | No | Free — diffusion decides COM height |
| Root heading | Conditional | Only when |wz| > 1e-6 |
| Joint angles | No | Free — diffusion generates from motion prior |
| Foot contacts | No | Free |

---

## 3. Previous Bug: Axis Swap

### 3.1 Symptom

Walk motions labeled "forward" actually drifted left/backward. Example from earlier generation run:

```
walk_fwd_006_norm_0004: vx=+0.684, vy=+0.078
  Actual root: Δx_fwd=-1.337m, Δy_left=+3.124m
  → Robot went LEFT 3.1m and BACKWARD 1.3m instead of forward
```

### 3.2 Root cause

In the original constraint code, `vx` (forward velocity) was assigned to `smooth_root_2d[:, 0]` (Kimodo X = lateral axis), and `vy` (lateral velocity) to `smooth_root_2d[:, 1]` (Kimodo Z = forward axis). The axes were swapped.

**Buggy code** (now fixed):
```python
x = vx * t    # forward velocity → lateral axis ✗
z = vy * t    # lateral velocity → forward axis ✗
```

**Fixed code**:
```python
x = vy * t    # lateral velocity → Kimodo X (lateral) ✓
z = vx * t    # forward velocity → Kimodo Z (forward) ✓
```

### 3.3 Impact

- All straight paths had the primary motion direction rotated 90°
- All curved paths had wrong rotation decomposition
- Heading constraints for curved paths pointed 90° off

---

## 4. Previous Bug: Acceleration/Deceleration Phases

### 4.1 Symptom

Root trajectory showed velocity ramps at clip start (~2s of acceleration) and occasional end-of-clip jitter.

### 4.2 Root cause

The previous constraint code built a 3× stretched cosine-eased speed profile. For a 7s walk at 30fps (210 generated frames):

```
Stretched path (630 frames):
  Accel: frames   0– 62 (10%)    speed: 0 → target
  Cruise: frames 63–566 (80%)    speed: target
  Decel: frames 567–629 (10%)    speed: target → 0

Constrained frames: 0–209  → includes ALL accel (63 frames) + 147 cruise frames
```

30% of constrained frames had an accelerating root path. The model's "walking at speed V" prior conflicted with the constraint's "root speed increasing from 0 to V."

### 4.3 Fix

Removed the speed profile entirely. The constraint is now pure **constant velocity** for the exact generation duration. The model sees "root moves at speed V for all frames" and generates walking kinematics at speed V — perfect alignment between constraint and motion prior.

---

## 5. Current Implementation (Final)

### 5.1 Constraint construction

`constraints.py:build_root2d_constraint(vel, duration, fps=30, stride=3)`:

1. Build constant-velocity path for `num_frames = duration × fps` frames
2. For straight paths: `X = vy·t, Z = vx·t`
3. For curved paths: body velocity rotated frame-by-frame via `wz`, integrated
4. Select every `stride`-th frame (default: 3) as constraint indices
5. Always include first and last frame
6. Include `global_root_heading` only when `|wz| > 1e-6`

### 5.2 Generation pipeline

`orchestrator.py:_generate_batch()`:

1. Build per-sample lists: `per_prompts`, `per_frames`, `per_constraints_raw`
2. Convert raw constraint dicts → `KimodoConstraint` objects via `load_constraints_lst()`
3. Call `model(per_prompts, per_frames, constraint_lst=per_kimodo_constraints)` — Kimodo list API
4. Each sample gets its own prompt, duration, and constraint → velocity distribution takes effect
5. Export each sample via `_export_rltracker()` → 7-key NPZ with IsaacLab DOF/body ordering

### 5.3 Output format

```
normal_loco/
  {type}_{traj}_{heading:03d}_{pace}_{variant:04d}__K{seed}/
    motion.npz    — 7 keys: fps, joint_pos(29), joint_vel(29),
                    body_pos_w(30,3), body_quat_w(30,4),
                    body_lin_vel_w(30,3), body_ang_vel_w(30,3)
  manifest.csv    — per-sample parameter index
```

NPZ keys match RLTracker reference dataset exactly (IsaacLab DOF/body order, 30fps native, MuJoCo z-up/x-forward coordinates).
