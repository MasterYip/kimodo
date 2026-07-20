#!/bin/bash
# =============================================================================
# Batch motion generation for G1 humanoid robot
#
# Usage:
#   bash scripts/batch_generate.sh <PROMPT> <OUTPUT_DIR> [OPTIONS]
#
# Examples:
#   # Quick: 32 samples, GPU 0
#   bash scripts/batch_generate.sh "a robot walks forward." outputs/walk
#
#   # Large batch: 128 samples, GPU 0
#   bash scripts/batch_generate.sh "a robot waves." outputs/wave -n 128
#
#   # Multi-prompt with custom duration and GPU
#   bash scripts/batch_generate.sh "walk. turn. crouch." outputs/multi -g 1 \
#       -d "3.0 2.0 2.0" -n 64
#
#   # With CFG tuning
#   bash scripts/batch_generate.sh "dance energetically." outputs/dance \
#       --cfg_type separated --cfg_weight 3.0 2.0
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# ---- defaults ----
MODEL="kimodo-g1-rp"
DURATION="5.0"
NUM_SAMPLES=32
GPU=0
DIFFUSION_STEPS=100
SEED="${SEED:-}"
EXTRA_ARGS=()

# ---- usage ----
usage() {
    head -30 "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

# ---- parse args ----
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        -d|--duration) DURATION="$2"; shift 2 ;;
        -n|--num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        -g|--gpu) GPU="$2"; shift 2 ;;
        -s|--seed) SEED="$2"; shift 2 ;;
        --diffusion-steps) DIFFUSION_STEPS="$2"; shift 2 ;;
        --cfg_type|--cfg_weight)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *)
            if [ -z "${PROMPT:-}" ]; then
                PROMPT="$1"
            elif [ -z "${OUTPUT_DIR:-}" ]; then
                OUTPUT_DIR="$1"
            else
                echo "ERROR: unexpected argument: $1"
                usage
            fi
            shift ;;
    esac
done

if [ -z "${PROMPT:-}" ] || [ -z "${OUTPUT_DIR:-}" ]; then
    echo "ERROR: PROMPT and OUTPUT_DIR are required"
    echo ""
    usage
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Kimodo G1 Batch Generation ==="
echo "Prompt:  ${PROMPT}"
echo "Samples: ${NUM_SAMPLES}"
echo "GPU:     ${GPU}"
echo "Output:  ${OUTPUT_DIR}"
echo ""

SEED_ARG=()
if [ -n "${SEED}" ]; then
    SEED_ARG=(--seed "${SEED}")
fi

cd "${KIMODO_ROOT}"
kimodo_gen "${PROMPT}" \
    --model "${MODEL}" \
    --duration "${DURATION}" \
    --num_samples "${NUM_SAMPLES}" \
    --diffusion_steps "${DIFFUSION_STEPS}" \
    --output "${OUTPUT_DIR}/motion" \
    "${SEED_ARG[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done: $(ls "${OUTPUT_DIR}"/*.npz 2>/dev/null | wc -l) NPZ files ==="
ls -lh "${OUTPUT_DIR}"/*.csv 2>/dev/null | head -5
