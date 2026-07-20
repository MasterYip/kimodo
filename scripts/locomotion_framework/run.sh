#!/bin/bash
# Launch the locomotion batch generation framework
#
# Usage:
#   bash scripts/locomotion_framework/run.sh [--dry-run] [-n TOTAL] [-g GPU] [-c CONFIG]
#
# Default config: g1_normal_loco.yaml (normal style, LHS sampling, 50 samples/type)
# Legacy config:  g1_locomotion.yaml (mixed styles, uniform sampling)
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env.sh

# Default config
CONFIG="locomotion_framework/configs/g1_normal_loco.yaml"

# Parse args: if -c is given, use that config; pass everything through
PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
    -c "$CONFIG" \
    "$@"
