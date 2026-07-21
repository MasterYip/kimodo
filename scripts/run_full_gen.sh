#!/bin/bash
cd /data/masteryip/kimodo/kimodo/scripts
source ./env.sh
exec python3 -m locomotion_framework.orchestrator \
  -c locomotion_framework/configs/g1_normal_loco.yaml \
  -o /data/masteryip/kimodo/kimodo/outputs/normal_loco \
  > /data/masteryip/kimodo/kimodo/outputs/normal_loco/run.log 2>&1
