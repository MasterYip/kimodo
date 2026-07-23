#!/bin/bash
# =============================================================================
# Distributed multi-GPU motion generation (sample-level splitting)
#
# Splits motion types AND samples across multiple GPUs.
# When there are fewer types than GPUs, samples within each type are split.
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

CONFIG="$CONFIG_DIR/g1_normal_loco.yaml"
GPUS="0,1,2,3,4,5,6,7"
OUTDIR=""
N_TOTAL=""
DRY_RUN=""

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

CONFIG_REL="$(realpath --relative-to="$REPO_ROOT" "$CONFIG" 2>/dev/null || echo "$CONFIG")"

cd "$REPO_ROOT"
source scripts/env.sh

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
N_GPUS=${#GPU_ARRAY[@]}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Distributed Generation — $N_GPUS GPU(s): ${GPUS}  ║"
echo "╠══════════════════════════════════════════════════════════════╣"

PYTHONPATH="$REPO_ROOT" python3 - "$CONFIG" "$GPUS" "$OUTDIR" "$N_TOTAL" "$DRY_RUN" <<'PYEOF'
import sys, os, yaml, subprocess, time, math, copy
from pathlib import Path

config_path = sys.argv[1]
gpu_list = sys.argv[2].split(',')
outdir_override = sys.argv[3] or ''
n_total_str = sys.argv[4] or ''
dry_run = sys.argv[5] or ''

repo_root = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent.parent)) + "/scripts"

with open(config_path) as f:
    cfg = yaml.safe_load(f)

motion_types = list(cfg['motion_types'].keys())
n_gpus = len(gpu_list)
n_types = len(motion_types)

# Compute total samples from config
config_total = sum(cfg['motion_types'][t].get('num_samples', 0) for t in motion_types)
total_motions = int(n_total_str) if n_total_str else config_total

# --- Determine distribution strategy ---
# If more GPUs than types, split each type's samples across GPUs (sample-level)
use_sample_split = n_gpus > n_types

if use_sample_split:
    # Sample-level splitting: each GPU gets every type, but with proportional samples
    samples_per_gpu_total = math.ceil(total_motions / n_gpus)
    
    gpu_samples = {}
    remaining = total_motions
    for i, gpu in enumerate(gpu_list):
        n = min(samples_per_gpu_total, remaining)
        gpu_samples[gpu] = n
        remaining -= n
    
    # Distribute per-type samples proportionally
    type_weights = {}
    type_samples = {}
    for t in motion_types:
        spec = cfg['motion_types'][t]
        ns = spec.get('num_samples', 0)
        type_weights[t] = ns / max(config_total, 1)
    
    gpu_assignments = {}
    for i, gpu in enumerate(gpu_list):
        gpu_total = gpu_samples[gpu]
        assignments = {}
        cumulative = 0
        for j, t in enumerate(motion_types):
            if j == len(motion_types) - 1:
                # Last type gets the remainder
                n = gpu_total - cumulative
            else:
                n = int(round(gpu_total * type_weights[t]))
            n = max(0, n)
            assignments[t] = n
            cumulative += n
        gpu_assignments[gpu] = assignments
    
    # Print
    for gpu in gpu_list:
        parts = [f"{t}:{gpu_assignments[gpu][t]}" for t in motion_types]
        print(f"║  GPU {gpu}: {{{', '.join(parts)}}}  (total={gpu_samples[gpu]})")
else:
    # Type-level splitting (original behavior)
    gpu_assignments = {gpu: {} for gpu in gpu_list}
    type_sample_map = {t: cfg['motion_types'][t].get('num_samples', 0) for t in motion_types}
    for i, mtype in enumerate(motion_types):
        gpu = gpu_list[i % n_gpus]
        gpu_assignments[gpu][mtype] = type_sample_map[mtype]
    
    for gpu in gpu_list:
        parts = [f"{t}:{gpu_assignments[gpu][t]}" for t in gpu_assignments[gpu]]
        total = sum(gpu_assignments[gpu].values())
        print(f"║  GPU {gpu}: {{{', '.join(parts)}}}  (total={total})")

if dry_run:
    print(f"║  DRY RUN — no generation")
    print(f"║  Strategy: {'sample-level' if use_sample_split else 'type-level'} split")
    print(f"║  Total motions: {total_motions} across {n_gpus} GPUs")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    for gpu in gpu_list:
        ga = gpu_assignments[gpu]
        if isinstance(ga, dict) and all(isinstance(v, int) for v in ga.values()):
            total_g = sum(ga.values())
            parts = [f"{t}={n}" for t, n in ga.items() if n > 0]
            print(f"\n[GPU {gpu}] {total_g} samples: {', '.join(parts)}")
        else:
            types = list(ga.keys())
            print(f"\n[GPU {gpu}] motion types: {types}")
            for t in types:
                spec = cfg['motion_types'][t]
                print(f"  {t}: {spec.get('num_samples', '?')} samples")
    sys.exit(0)

# --- Launch generation ---
tmpdir = Path(cfg.get('global', {}).get('output_dir', 'outputs/normal_loco')).parent / '.tmp_configs'
tmpdir.mkdir(parents=True, exist_ok=True)

procs = []
for gpu in gpu_list:
    ga = gpu_assignments[gpu]
    
    if isinstance(ga, dict) and all(isinstance(v, int) for v in ga.values()):
        # Sample-level split: create config with per-type sample counts
        sub_cfg = copy.deepcopy(cfg)
        for t in motion_types:
            if t in ga:
                sub_cfg['motion_types'][t]['num_samples'] = ga[t]
            else:
                sub_cfg['motion_types'][t]['num_samples'] = 0
        
        # Remove types with 0 samples
        sub_cfg['motion_types'] = {t: s for t, s in sub_cfg['motion_types'].items() if s.get('num_samples', 0) > 0}
        
        if not sub_cfg['motion_types']:
            print(f"║  GPU {gpu}: no samples — skipping")
            continue
        
        # Use per-GPU output subdirectory to avoid file conflicts
        outdir = Path(outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco'))
        
        # Each GPU uses different seed for diversity
        sub_cfg['global']['seed'] = cfg.get('global', {}).get('seed', 42) + int(gpu) * 1000
    else:
        # Type-level split
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg['motion_types'] = {t: cfg['motion_types'][t] for t in ga}
        outdir = Path(outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco'))
    
    tmp_config = tmpdir / f"gpu{gpu}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(sub_cfg, f, default_flow_style=False)
    
    cmd = [
        sys.executable, '-m', 'locomotion_framework.orchestrator',
        '-c', str(tmp_config),
        '-g', '0',
        '-o', str(outdir),
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONPATH'] = repo_root
    env['PYTHONUNBUFFERED'] = '1'
    
    gpu_total = sum(sub_cfg['motion_types'][t].get('num_samples', 0) for t in sub_cfg['motion_types'])
    print(f"║  Launching GPU {gpu}: {gpu_total} samples — {list(sub_cfg['motion_types'].keys())}")
    
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

for gpu, p, logfile in procs:
    rc = p.wait()
    status = "✓ DONE" if rc == 0 else f"✗ FAILED (exit={rc})"
    print(f"  GPU {gpu}: {status}  (log: {logfile})")

# Verify
failed = [(gpu, p, logfile) for gpu, p, logfile in procs if p.returncode != 0]
print()

if not failed:
    print("All GPUs completed successfully.")
    outdir = Path(outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco'))
    n_dirs = len([d for d in outdir.iterdir() if d.is_dir()]) if outdir.exists() else 0
    print(f"Output: {outdir} ({n_dirs} motion dirs)")
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
else:
    print("Some GPUs failed — check logs in", tmpdir)
    for gpu, p, logfile in failed:
        print(f"  GPU {gpu}: exit={p.returncode}  log={logfile}")
    sys.exit(1)
PYEOF
