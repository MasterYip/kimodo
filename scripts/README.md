# Kimodo Utility Scripts

All scripts live under `scripts/` in the repository root.
Source `env.sh` first (the other scripts do this automatically).

| Script | Purpose |
|--------|---------|
| `env.sh` | Set environment variables (HF mirror, text encoder, venv) |
| `launch_demo.sh` | Start the interactive motion authoring web UI |
| `batch_generate.sh` | Convenient batch generation with sensible defaults |
| `mujoco_view.sh` | Open a generated CSV in the MuJoCo physics viewer |
| `install_nf4_encoder.sh` | Download the standalone NF4 text encoder |
| `test_hf_token.py` | Test whether a HF token has access to Llama 3 |

---

## Quick Start

```bash
cd /data/masteryip/kimodo/kimodo   # repo root

# Set up environment
source scripts/env.sh

# Generate 32 walking motions
bash scripts/batch_generate.sh "a robot walks forward." outputs/walk

# Launch the web demo
bash scripts/launch_demo.sh

# View a motion in MuJoCo
bash scripts/mujoco_view.sh outputs/walk/motion_00.csv
```

---

## Script Details

### `env.sh`

Sets all required environment variables. **Source this first** or the other scripts will do it for you.

```bash
source scripts/env.sh
```

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_ENDPOINT` | `https://hf-mirror.com` | HF mirror (direct access, no proxy) |
| `TEXT_ENCODER` | `llm2vec-nf4` | Standalone NF4 text encoder |
| `HF_HOME` | `../.cache/huggingface` | Model cache location |
| `CUDA_VISIBLE_DEVICES` | `0` (default) | Which GPU to use |

Override with environment:
```bash
CUDA_VISIBLE_DEVICES=2 source scripts/env.sh
```

---

### `launch_demo.sh`

Starts the Kimodo interactive web demo (Gradio + Viser 3D viewer).

```bash
# G1 robot model (default)
bash scripts/launch_demo.sh

# Alternative skeleton
bash scripts/launch_demo.sh kimodo-soma-rp
```

Access at **http://127.0.0.1:7860**.

For remote servers, set up SSH tunnel:
```bash
ssh -L 7860:127.0.0.1:7860 user@host -p 22222 -N
```

**Loading pre-generated motions:** In the demo UI, expand the "Save / Load" panel,
paste a `.npz` path in the "Load Path" field, and click "Load Motion".

---

### `batch_generate.sh`

Wrapper around `kimodo_gen` with defaults tuned for batch G1 generation.

```bash
bash scripts/batch_generate.sh <PROMPT> <OUTPUT_DIR> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | `kimodo-g1-rp` | Kimodo model variant |
| `-d, --duration` | `5.0` | Duration per segment in seconds |
| `-n, --num-samples` | `32` | Number of parallel samples |
| `-g, --gpu` | `0` | GPU device index |
| `-s, --seed` | random | Random seed for reproducibility |
| `--diffusion-steps` | `100` | Denoising steps (fewer = faster) |

Examples:
```bash
# 128 samples on GPU 0
bash scripts/batch_generate.sh "walk." outputs/batch128 -n 128

# Multi-prompt on GPU 1
bash scripts/batch_generate.sh "walk. turn. wave." outputs/multi \
    -d "3.0 2.0 2.0" -g 1 -n 64 -s 42

# Aggressive text guidance
bash scripts/batch_generate.sh "dance." outputs/dance \
    --cfg_type separated --cfg_weight 3.0 2.0
```

---

### `mujoco_view.sh`

Opens a generated G1 CSV file in the MuJoCo physics viewer.
Requires a display (X11/desktop).

```bash
bash scripts/mujoco_view.sh outputs/batch128/walk/walk_00.csv
```

---

### `install_nf4_encoder.sh`

Downloads the NF4 text encoder and installs `bitsandbytes`.
Only needed once per installation.

```bash
bash scripts/install_nf4_encoder.sh
```

---

### `test_hf_token.py`

Tests whether a Hugging Face token has access to the Llama 3 gated model.
Token is read from stdin — never appears in command line or logs.

```bash
python3 scripts/test_hf_token.py
# Paste token when prompted
```
