#!/bin/bash
# Launch the locomotion batch generation framework
# Usage:
#   bash scripts/locomotion_framework/run.sh [--dry-run] [-n TOTAL] [-g GPU]
set -euo pipefail

cd "$(dirname "$0")/.."
source ./env.sh
PYTHONPATH=. python3 -m locomotion_framework.orchestrator \
    -c locomotion_framework/configs/g1_locomotion.yaml \
    "$@"
