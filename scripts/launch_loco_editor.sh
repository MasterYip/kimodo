#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/env.sh"
MODEL="${1:-kimodo-g1-rp}"
PORT="${2:-7861}"
CONFIG="${3:-}"
ARGS="--model ${MODEL} --port ${PORT}"
[ -n "${CONFIG:-}" ] && ARGS="${ARGS} --config ${CONFIG}"
EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP "pid=\K[0-9]+" | head -1 || true)
if [ -n "${EXISTING_PID:-}" ]; then
    echo "Killing existing editor on port ${PORT} (pid ${EXISTING_PID})"
    kill "${EXISTING_PID}" 2>/dev/null || true
    sleep 2
fi
cd "${SCRIPT_DIR}"
echo "Starting Loco Editor: model=${MODEL} port=${PORT}"
PYTHONPATH=. exec python3 -m locomotion_framework.editor ${ARGS}
