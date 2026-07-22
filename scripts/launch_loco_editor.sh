#!/bin/bash
# =============================================================================
# Launch the Kimodo Locomotion Editor
#
# Usage:
#   bash scripts/launch_loco_editor.sh [MODEL] [PORT] [CONFIG]
#
#   MODEL  defaults to kimodo-g1-rp (Unitree G1 humanoid robot).
#   PORT   defaults to 7861.
#   CONFIG optional path to initial YAML config file.
#
# Access:
#   Open http://127.0.0.1:7861 in your browser.
#   If running on a remote server, set up an SSH tunnel first:
#       ssh -L 7861:127.0.0.1:7861 user@host -p 22222 -N
#
# Example:
#   bash scripts/launch_loco_editor.sh kimodo-g1-rp 7861 configs/g1_normal_loco.yaml
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="${1:-kimodo-g1-rp}"
PORT="${2:-7861}"
CONFIG="${3:-}"

echo "=== Kimodo Locomotion Editor ==="
echo "Model:  ${MODEL}"
echo "Port:   ${PORT}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Access: http://127.0.0.1:${PORT}"
echo ""

# Kill any previous editor instance
pkill -f "locomotion_framework.editor" 2>/dev/null || true
sleep 1

cd "${KIMODO_ROOT}/scripts"

ARGS="--model ${MODEL} --port ${PORT}"
if [ -n "${CONFIG:-}" ] && [ -f "${CONFIG}" ]; then
    ARGS="${ARGS} --config ${CONFIG}"
    echo "Config: ${CONFIG}"
fi

PYTHONPATH=. python3 -m locomotion_framework.editor ${ARGS}
