#!/bin/bash
# =============================================================================
# View a generated G1 motion in MuJoCo
#
# Usage:
#   bash scripts/mujoco_view.sh <CSV_FILE>
#
# Example:
#   bash scripts/mujoco_view.sh outputs/batch128/walk/walk_00.csv
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

CSV_FILE="${1:-}"
if [ -z "${CSV_FILE}" ]; then
    echo "Usage: bash scripts/mujoco_view.sh <CSV_FILE>"
    echo "Example: bash scripts/mujoco_view.sh outputs/walk.csv"
    exit 1
fi

if [ ! -f "${CSV_FILE}" ]; then
    echo "ERROR: file not found: ${CSV_FILE}"
    exit 1
fi

echo "=== MuJoCo Viewer: ${CSV_FILE} ==="

cat << PYEOF > /tmp/_kimodo_mujoco_view.py
import mujoco, mujoco.viewer, numpy as np, time
from kimodo.assets import skeleton_asset_path

qpos = np.loadtxt("${CSV_FILE}", delimiter=",")
model = mujoco.MjModel.from_xml_path(
    str(skeleton_asset_path("g1skel34", "xml", "g1.xml"))
)
data = mujoco.MjData(model)
print(f"Frames: {len(qpos)}, Joints: {qpos.shape[1]}")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 30.0)
PYEOF

cd "${KIMODO_ROOT}"
python3 /tmp/_kimodo_mujoco_view.py
rm -f /tmp/_kimodo_mujoco_view.py
