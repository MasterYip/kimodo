#!/bin/bash
# Full 200-motion generation with RLTracker preset
set -euo pipefail
cd /data/masteryip/kimodo/kimodo/scripts
source ./env.sh
rm -rf /data/masteryip/kimodo/kimodo/outputs/normal_loco
PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
    -c locomotion_framework/configs/g1_normal_loco.yaml \
    --preset rltracker \
    -o /data/masteryip/kimodo/kimodo/outputs/normal_loco
