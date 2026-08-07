#!/bin/bash
# =============================================================================
# Kimodo Environment Setup — source this before any kimodo command
#
# Usage:
#   source scripts/env.sh
#   kimodo_gen "a robot walks." --model kimodo-g1-rp -d 5.0 -o output
# =============================================================================

# ---- install paths (edit if you moved the repo) ----
KIMODO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/data/masteryip/kimodo/.venv"

# ---- activate virtual environment ----
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
fi

# ---- Hugging Face mirror (direct access — no proxy needed) ----
export HF_ENDPOINT=https://hf-mirror.com

# ---- Text encoder: standalone NF4 model (no gated Llama dependency) ----
export TEXT_ENCODER=llm2vec-nf4

# ---- Configurable cache / tmp root ----
# Set KIMODO_CACHE_DIR or KIMODO_TMP_DIR before sourcing to override. Defaults
# live under <kimodo parent>/.cache on the same volume as the repo (e.g. /data),
# so generation never needs free space on the root filesystem.
KIMODO_CACHE_DIR="${KIMODO_CACHE_DIR:-${KIMODO_ROOT}/../.cache}"
KIMODO_TMP_DIR="${KIMODO_TMP_DIR:-${KIMODO_CACHE_DIR}/tmp}"

# ---- Cache paths ----
export HF_HOME="${KIMODO_CACHE_DIR}/huggingface"
export TORCHINDUCTOR_CACHE_DIR="${KIMODO_CACHE_DIR}/torchinductor"
export XDG_CACHE_HOME="${KIMODO_CACHE_DIR}/xdg"

# ---- Temp paths ----
# Redirect TMPDIR (python tempfile, bvh export, loco editor) and per-tool caches
# (matplotlib ~/.config, cuda ~/.nv) off the root filesystem. Without this a full
# root FS (e.g. /tmp on the same device) hangs any subprocess that writes.
export TMPDIR="${KIMODO_TMP_DIR}"
export TMP="${KIMODO_TMP_DIR}"
export TEMP="${KIMODO_TMP_DIR}"
export MPLCONFIGDIR="${KIMODO_CACHE_DIR}/matplotlib"
export CUDA_CACHE_PATH="${KIMODO_CACHE_DIR}/cuda"

mkdir -p "${KIMODO_TMP_DIR}" \
         "${KIMODO_CACHE_DIR}/torchinductor" \
         "${KIMODO_CACHE_DIR}/xdg" \
         "${KIMODO_CACHE_DIR}/matplotlib" \
         "${KIMODO_CACHE_DIR}/cuda"

# ---- GPU defaults ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[kimodo] HF_ENDPOINT=${HF_ENDPOINT}  TEXT_ENCODER=${TEXT_ENCODER}  TMPDIR=${TMPDIR}"
