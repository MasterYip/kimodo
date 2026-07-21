#!/bin/bash
# =============================================================================
# Distributed multi-GPU motion generation
#
# Splits motion types from a config across multiple GPUs and launches
# parallel orchestrator processes — cutting wall-clock time by N_GPUs×.
#
# Usage:
#   bash scripts/locomotion_framework/run_distributed.sh [options]
#
# Options:
#   -c CONFIG    Config file (default: locomotion_framework/configs/g1_normal_loco.yaml)
#   -g GPUS      Comma-separated GPU indices (default: 0,1,2,3,4,5,6,7)
#   -o OUTDIR    Output directory (default: from config)
#   -n TOTAL     Total motions across all GPUs (overrides per-type counts)
#   --dry-run    Print GPU assignment without generating
#
# Examples:
#   # All 8 GPUs, full config
#   bash scripts/locomotion_framework/run_distributed.sh
#
#   # 4 GPUs, 200 random motions
#   bash scripts/locomotion_framework/run_distributed.sh -g 0,1,2,3 -n 200
#
#   # Dry run to see assignment
#   bash scripts/locomotion_framework/run_distributed.sh --dry-run
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"

# ---- defaults ----
CONFIG="$CONFIG_DIR/g1_normal_loco.yaml"
GPUS="0,1,2,3,4,5,6,7"
OUTDIR=""
N_TOTAL=""
DRY_RUN=""

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) CONFIG="$2"; shift 2 ;;
        -g) GPUS="$2"; shift 2 ;;
        -o) OUTDIR="$2"; shift 2 ;;
        -n) N_TOTAL="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---- resolve paths ----
CONFIG_REL="$(realpath --relative-to="$REPO_ROOT" "$CONFIG" 2>/dev/null || echo "$CONFIG")"

# ---- source env ----
cd "$REPO_ROOT"
source scripts/env.sh

# ---- split config by GPU (Python helper inline) ----
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
N_GPUS=${#GPU_ARRAY[@]}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Distributed Generation — $N_GPUS GPU(s): ${GPUS}  ║"
echo "╠══════════════════════════════════════════════════════════════╣"

# Generate per-GPU configs and launch scripts via Python
PYTHONPATH="$REPO_ROOT" python3 - "$CONFIG" "$GPUS" "$OUTDIR" "$N_TOTAL" "$DRY_RUN" <<'PYEOF'
import sys, os, yaml, subprocess, time
from pathlib import Path

config_path = sys.argv[1]
gpu_list = sys.argv[2].split(',')
outdir_override = sys.argv[3] or ''
n_total = sys.argv[4] or ''
dry_run = sys.argv[5] or ''

repo_root = os.environ.get('REPO_ROOT', str(Path(__file__).resolve().parent.parent.parent))

# Load config
with open(config_path) as f:
    cfg = yaml.safe_load(f)

motion_types = list(cfg['motion_types'].keys())
n_gpus = len(gpu_list)

# Distribute motion types round-robin across GPUs
gpu_assignments = {gpu: [] for gpu in gpu_list}
for i, mtype in enumerate(motion_types):
    gpu = gpu_list[i % n_gpus]
    gpu_assignments[gpu].append(mtype)

# Print assignment
for gpu, types in gpu_assignments.items():
    print(f"║  GPU {gpu}: {types}")

if dry_run:
    print(f"║  DRY RUN — no generation")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    for gpu, types in gpu_assignments.items():
        print(f"\n[GPU {gpu}] motion types: {types}")
        for t in types:
            spec = cfg['motion_types'][t]
            print(f"  {t}: {spec.get('num_samples', '?')} samples, "
                  f"vx={spec['vel_cmd'].get('vx',[0,0])}, "
                  f"vy={spec['vel_cmd'].get('vy',[0,0])}, "
                  f"wz={spec['vel_cmd'].get('wz',[0,0])}")
    sys.exit(0)

# Build per-GPU configs
tmpdir = Path(cfg.get('global', {}).get('output_dir', 'outputs/normal_loco')).parent / '.tmp_configs'
tmpdir.mkdir(parents=True, exist_ok=True)

procs = []
for gpu, types in gpu_assignments.items():
    if not types:
        continue

    # Subset config
    sub_cfg = dict(cfg)
    sub_cfg['motion_types'] = {t: cfg['motion_types'][t] for t in types}

    tmp_config = tmpdir / f"gpu{gpu}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(sub_cfg, f, default_flow_style=False)

    outdir = outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco')
    outdir = Path(outdir)

    # Build command
    cmd = [
        sys.executable, '-m', 'locomotion_framework.orchestrator',
        '-c', str(tmp_config),
        '-g', '0',  # CUDA_VISIBLE_DEVICES maps local 0 → physical GPU
        '-o', str(outdir),
    ]
    if n_total:
        cmd += ['-n', n_total]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONPATH'] = repo_root
    env['PYTHONUNBUFFERED'] = '1'

    print(f"║  Launching GPU {gpu}: {' '.join(cmd)}")

    logfile = tmpdir / f"gpu{gpu}.log"
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(logfile, 'w'),
        stderr=subprocess.STDOUT,
        cwd=repo_root,
    )
    procs.append((gpu, p, logfile))

print(f"╚══════════════════════════════════════════════════════════════╝")
print(f"\nWaiting for {len(procs)} GPU workers...\n")

# Wait for all
for gpu, p, logfile in procs:
    rc = p.wait()
    status = "✓ DONE" if rc == 0 else f"✗ FAILED (exit={rc})"
    print(f"  GPU {gpu}: {status}  (log: {logfile})")

# Check results
total_ok = all(p.wait() == 0 for _, p, _ in procs)
print()

if total_ok:
    print("All GPUs completed successfully.")

    # Count generated motions
    outdir = Path(outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco'))
    n_dirs = len([d for d in outdir.iterdir() if d.is_dir()]) if outdir.exists() else 0
    print(f"Output: {outdir} ({n_dirs} motion dirs)")

    # Cleanup temp configs
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
else:
    print("Some GPUs failed — check logs in", tmpdir)
    sys.exit(1)
PYEOF
