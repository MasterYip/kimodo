# Kimodo G1 Robot Motion Generation — Quick Guide

**Server:** 4090-3 (8× RTX 4090 24 GB)  
**Install path:** `/data/masteryip/kimodo/`  
**Venv:** `/data/masteryip/kimodo/.venv`  

> **Note:** All downloads use `hf-mirror.com` instead of the local proxy.
> The text encoder uses `Aero-Ex/KIMODO-Meta3_llm2vec_NF4` (standalone NF4, no gated Llama dependency).

---

## 1. Activate Environment

```bash
cd /data/masteryip/kimodo/kimodo
source scripts/env.sh
```

This sets:
- `HF_ENDPOINT=https://hf-mirror.com` — HF mirror for downloads
- `TEXT_ENCODER=llm2vec-nf4` — standalone NF4 text encoder
- `HF_HOME=/data/masteryip/kimodo/.cache/huggingface`
- Activates the virtual environment automatically

### Available Scripts

All utilities live under `scripts/`. See `scripts/README.md` for full details.

| Script | Purpose |
|--------|---------|
| `scripts/env.sh` | Set environment + activate venv |
| `scripts/batch_generate.sh` | Batch generation with sensible defaults |
| `scripts/launch_demo.sh` | Start interactive web UI |
| `scripts/mujoco_view.sh` | View a CSV in MuJoCo |
| `scripts/install_nf4_encoder.sh` | Download NF4 text encoder |
| `scripts/test_hf_token.py` | Test HF token access |

---

## 3. Available G1 Models

| Short Key | Model Name | Training Data | Use |
|-----------|-----------|---------------|-----|
| `kimodo-g1-rp` | Kimodo-G1-RP-v1 | Bones Rigplay 1 (700 hrs) | **Best quality — default** |
| `kimodo-g1-seed` | Kimodo-G1-SEED-v1 | BONES-SEED (288 hrs) | Public-data comparison |

Models auto-download from Hugging Face on first use (~10-15 GB each).

---

## 4. Basic Generation

### Single motion
```bash
TEXT_ENCODER_DEVICE=cpu kimodo_gen \
  "a robot walks forward." \
  --model kimodo-g1-rp \
  --duration 5.0 \
  --output /data/masteryip/outputs/walk
```

### Multiple variations (10 samples)
```bash
TEXT_ENCODER_DEVICE=cpu kimodo_gen \
  "a robot waves its right hand." \
  --model kimodo-g1-rp \
  --duration 3.0 \
  --num_samples 10 \
  --seed 42 \
  --output /data/masteryip/outputs/wave
```

---

## 5. Fine-Grained Motion Control

### Multi-prompt timeline (speed + direction + gestures)
```bash
TEXT_ENCODER_DEVICE=cpu kimodo_gen \
  "walk forward quickly. turn left slowly. crouch and wave right hand." \
  --model kimodo-g1-rp \
  --duration "3.0 2.0 3.0" \
  --num_transition_frames 10 \
  --output /data/masteryip/outputs/multi_prompt
```

| What it does | How |
|-------------|-----|
| **Speed control** | "quickly", "slowly" in prompt + duration per segment |
| **Direction** | "turn left", "turn right", "turn around", "walk backward" |
| **Style** | "aggressively", "gracefully", "casually", "carefully" |
| **Gestures** | "wave right hand", "point forward", "raise both hands" |

### CFG tuning (text vs constraint balance)
```bash
# Stronger text adherence
TEXT_ENCODER_DEVICE=cpu kimodo_gen \
  "a robot dances energetically." \
  --model kimodo-g1-rp \
  --duration 5.0 \
  --cfg_type separated --cfg_weight 4.0 2.0 \
  --output /data/masteryip/outputs/dance_strong_text
```

- `--cfg_type separated` — independent control over text + constraints
- `--cfg_weight <text_weight> <constraint_weight>` — higher text weight = stronger prompt adherence
- Default: `2.0 2.0` | Recommended range: `1.5–5.0`

### Diffusion quality
```bash
# Faster (lower quality): --diffusion_steps 50
# Default: --diffusion_steps 100
# Higher quality (slower): --diffusion_steps 200
```

---

## 6. Batch / Parallel Generation

Kimodo processes all `--num_samples` in a **single parallel forward pass** (one denoising loop, batch dim = num_samples).

### Large batch (128 samples in one shot)
```bash
TEXT_ENCODER_DEVICE=cpu CUDA_VISIBLE_DEVICES=0 kimodo_gen \
  "a robot walks forward." \
  --model kimodo-g1-rp \
  --duration 5.0 \
  --num_samples 128 \
  --diffusion_steps 100 \
  --output /data/masteryip/outputs/batch128/walk
```

### Massive batch (4×128 = 512 total, spread across GPUs)
```bash
for gpu in 0 1 2 3; do
  TEXT_ENCODER_DEVICE=cpu CUDA_VISIBLE_DEVICES=$gpu kimodo_gen \
    "a robot performs various actions." \
    --model kimodo-g1-rp \
    --duration 5.0 \
    --num_samples 128 \
    --seed $((gpu * 1000)) \
    --output /data/masteryip/outputs/batch512/gpu${gpu}_ &
done
wait
```

### Batch size guidelines (RTX 4090 24 GB, `TEXT_ENCODER_DEVICE=cpu`)

| Batch Size | VRAM | Time (100 steps, 5s) | Notes |
|-----------|------|----------------------|-------|
| 32 | ~3 GB | ~10s | Very safe |
| 64 | ~4 GB | ~15s | Safe |
| 128 | ~6 GB | ~25s | Recommended max per GPU |
| 256 | ~10 GB | ~45s | Works but diminishing returns |
| 512 | ~18 GB | ~90s | Tight — monitor VRAM |

---

## 7. Output Formats

### G1 model outputs

| Flag | Format | File | Use Case |
|------|--------|------|----------|
| (default) | Kimodo NPZ | `*.npz` | Full motion data (posed_joints, rot_mats, foot_contacts) |
| (auto for G1) | MuJoCo CSV | `*.csv` | Direct load into MuJoCo simulation |
| `--save_example_dir` | Demo dir | `*_example/` | Reload into Kimodo web demo |

### Output NPZ structure
```python
import numpy as np
data = np.load("output.npz")
data["posed_joints"]      # [T, 34, 3] — joint positions
data["global_rot_mats"]   # [T, 34, 3, 3] — rotation matrices
data["local_rot_mats"]    # [T, 34, 3, 3] — parent-relative rotations
data["foot_contacts"]     # [T, 4] — heel/toe contacts
data["root_positions"]    # [T, 3] — root trajectory
data["smooth_root_pos"]   # [T, 3] — smoothed root
data["global_root_heading"]  # [T, 2] — heading (cos, sin)
```

### Convert to BVH
```bash
# SOMA models only → add --bvh
# For G1, use CSV directly in MuJoCo
```

---

## 8. Visualizing G1 Motions in MuJoCo

```bash
source /data/masteryip/kimodo/.venv/bin/activate

# Generate motion as MuJoCo CSV
TEXT_ENCODER_DEVICE=cpu kimodo_gen \
  "a robot walks forward." \
  --model kimodo-g1-rp \
  --duration 5.0 \
  --output /data/masteryip/outputs/mujoco_test

# Then edit kimodo/scripts/mujoco_load.py to point to your CSV
python -m kimodo.scripts.mujoco_load
```

---

## 9. Interactive Demo (Web UI)

```bash
cd /data/masteryip/kimodo/kimodo
source scripts/env.sh

# Start the demo
bash scripts/launch_demo.sh

# Opens at http://127.0.0.1:7860
# For remote access, set up SSH tunnel first:
#   ssh -L 7860:127.0.0.1:7860 user@<server-host> -N
```

The web demo provides:
- Timeline-based text prompt editing
- Full-body keyframe constraints
- End-effector (hands/feet) positioning
- 2D root path drawing
- Real-time 3D preview
- Export constraints as JSON for CLI reuse

---

## 10. Constraint-Based Generation

### Save constraints from demo → use in CLI

```bash
# After exporting constraints.json from the web demo:
TEXT_ENCODER_DEVICE=cpu kimodo_gen \
  "a robot walks forward." \
  --model kimodo-g1-rp \
  --constraints /path/to/constraints.json \
  --output /data/masteryip/outputs/constrained
```

### Batch with constraints from folder
```bash
TEXT_ENCODER_DEVICE=cpu kimodo_gen \
  --input_folder /data/masteryip/inputs/scene1 \
  --model kimodo-g1-rp \
  --num_samples 32 \
  --output /data/masteryip/outputs/scene1
```

The `--input_folder` should contain:
- `meta.json` — with `"text"`, `"duration"`, `"num_samples"`, etc.
- `constraints.json` (optional) — keyframes, paths, end-effector targets

---

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| `OSError: ...Meta-Llama-3-8B-Instruct...` | Ensure `source env.sh` to use NF4 text encoder |
| Out of memory (CUDA OOM) | Reduce `--num_samples`; use `CUDA_VISIBLE_DEVICES` for a free GPU |
| Network errors downloading | Ensure `HF_ENDPOINT=https://hf-mirror.com` (set in `env.sh`) |
| Slow text encoding | First run loads NF4 model (~4.7 GB to GPU); subsequent runs faster |
| `401/403 GatedRepoError` | NF4 model is ungated; should not happen with `TEXT_ENCODER=llm2vec-nf4` |
| Proxy `127.0.0.1:7890` unreachable | Not needed — all downloads use `hf-mirror.com` directly |

---

## 12. Quick Reference Card

```bash
# === ENV (REQUIRED before any command) ===
cd /data/masteryip/kimodo/kimodo
source scripts/env.sh

# === SINGLE ===
kimodo_gen "prompt" --model kimodo-g1-rp -d 5.0 -o out

# === BATCH (via script) ===
bash scripts/batch_generate.sh "prompt" outputs/dir -n 128

# === MULTI-PROMPT ===
kimodo_gen "walk. turn. wave." --model kimodo-g1-rp \
  --duration "3.0 2.0 2.0" --output out

# === CFG TUNING ===
bash scripts/batch_generate.sh "prompt" outputs/dir \
  --cfg_type separated --cfg_weight 3.0 2.0

# === DEMO ===
bash scripts/launch_demo.sh
```

| Short flag | Long flag |
|-----------|----------|
| `-d` | `--duration` |
| `-o` | `--output` |
| `-m` | `--model` |
| `-n` | `--num_samples` |

---

## 13. Text Encoder — NF4 Substitution

The standard Kimodo text encoder (`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`)
depends on gated `meta-llama/Meta-Llama-3-8B-Instruct`. Since this repo rejected access for
`MasterYip`, we use a **standalone NF4 substitute**:

| | Original | Substitution |
|---|---|---|
| **Model** | `McGill-NLP/LLM2Vec-...-mntp` + `-supervised` | `Aero-Ex/KIMODO-Meta3_llm2vec_NF4` |
| **Size** | 16 GB + 168 MB PEFT | 4.7 GB (NF4 quantized) |
| **Gated** | Yes (blocked) | No (ungated) |
| **Architecture** | `LlamaBiModel`, 4096 dim | Same |
| **Pooling** | mean, 512 max tokens | Same |

Selection is automatic via `TEXT_ENCODER=llm2vec-nf4` in `env.sh`.
If the Llama 3 gate is ever approved, switch back with `TEXT_ENCODER=llm2vec`.
