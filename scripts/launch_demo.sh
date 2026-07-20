#!/bin/bash
# =============================================================================
# Launch the Kimodo Interactive Motion Authoring Demo
#
# Usage:
#   bash scripts/launch_demo.sh [MODEL]
#
#   MODEL defaults to kimodo-g1-rp (Unitree G1 humanoid robot).
#   Other options: kimodo-soma-rp, kimodo-smplx-rp
#
# Access:
#   Open http://127.0.0.1:7860 in your browser.
#   If running on a remote server, set up an SSH tunnel first:
#       ssh -L 7860:127.0.0.1:7860 user@host -p 22222 -N
#
# Example:
#   bash scripts/launch_demo.sh kimodo-g1-rp
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="${1:-kimodo-g1-rp}"

echo "=== Kimodo Interactive Demo ==="
echo "Model:  ${MODEL}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Access: http://127.0.0.1:7860"
echo ""

# Kill any previous demo
pkill -f "kimodo.demo" 2>/dev/null || true
sleep 1

cd "${KIMODO_ROOT}"
python3 -m kimodo.demo --model "${MODEL}"
