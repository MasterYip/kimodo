#!/bin/bash
# =============================================================================
# Distributed multi-GPU motion generation (v2 — sample-level splitting)
#
# Splits motion types AND samples across multiple GPUs.
# When there are fewer types than GPUs, samples within each type are split.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"

CONFIG="$SCRIPTS_DIR/locomotion_framework/configs/g1_normal_loco.yaml"
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

cd "$REPO_ROOT"
source scripts/env.sh

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
N_GPUS=${#GPU_ARRAY[@]}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Distributed Generation — $N_GPUS GPU(s): ${GPUS}  ║"
echo "╠══════════════════════════════════════════════════════════════╣"

# Pass REPO_ROOT explicitly as env
export REPO_ROOT
export SCRIPTS_DIR

PYTHONPATH="$SCRIPTS_DIR" python3 - "$CONFIG" "$GPUS" "$OUTDIR" "$N_TOTAL" "$DRY_RUN" "$SCRIPTS_DIR" <<'PYEOF'
import sys, os, yaml, subprocess, time, math, copy
from pathlib import Path

config_path = sys.argv[1]
gpu_list = sys.argv[2].split(',')
outdir_override = sys.argv[3] or ''
n_total_str = sys.argv[4] or ''
dry_run = sys.argv[5] or ''
scripts_dir = sys.argv[6]  # /data/.../scripts

with open(config_path) as f:
    cfg = yaml.safe_load(f)

motion_types = list(cfg['motion_types'].keys())
n_gpus = len(gpu_list)
n_types = len(motion_types)
config_total = sum(cfg['motion_types'][t].get('num_samples', 0) for t in motion_types)
total_motions = int(n_total_str) if n_total_str else config_total

use_sample_split = (n_gpus > n_types) and n_types > 0

if use_sample_split:
    samples_per_gpu_total = math.ceil(total_motions / n_gpus)
    type_weights = {t: cfg['motion_types'][t].get('num_samples', 0) / max(config_total, 1)
                    for t in motion_types}

    gpu_samples = {}
    remaining = total_motions
    for i, gpu in enumerate(gpu_list):
        n = min(samples_per_gpu_total, remaining)
        gpu_samples[gpu] = n
        remaining -= n

    gpu_assignments = {}
    for i, gpu in enumerate(gpu_list):
        gpu_total = gpu_samples[gpu]
        assignments = {}
        cumulative = 0
        for j, t in enumerate(motion_types):
            if j == len(motion_types) - 1:
                n = max(0, gpu_total - cumulative)
            else:
                n = max(0, int(round(gpu_total * type_weights[t])))
            assignments[t] = n
            cumulative += n
        gpu_assignments[gpu] = assignments

    for gpu in gpu_list:
        parts = [f"{t}:{gpu_assignments[gpu][t]}" for t in motion_types]
        print(f"║  GPU {gpu}: {{{', '.join(parts)}}}  (total={gpu_samples[gpu]})")
else:
    gpu_assignments = {gpu: {} for gpu in gpu_list}
    for i, mtype in enumerate(motion_types):
        gpu = gpu_list[i % n_gpus]
        gpu_assignments[gpu][mtype] = cfg['motion_types'][mtype].get('num_samples', 0)
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
        if ga and all(isinstance(v, int) for v in ga.values()):
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
outbase = outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco')
tmpdir = Path(outbase).parent / '.tmp_configs'
tmpdir.mkdir(parents=True, exist_ok=True)

procs = []
for gpu in gpu_list:
    ga = gpu_assignments[gpu]

    sub_cfg = copy.deepcopy(cfg)
    if isinstance(ga, dict) and all(isinstance(v, int) for v in ga.values()):
        # sample-level split
        for t in motion_types:
            sub_cfg['motion_types'][t]['num_samples'] = ga.get(t, 0)
        sub_cfg['motion_types'] = {t: s for t, s in sub_cfg['motion_types'].items()
                                    if s.get('num_samples', 0) > 0}
    else:
        # type-level split
        sub_cfg['motion_types'] = {t: cfg['motion_types'][t] for t in ga}

    if not sub_cfg['motion_types']:
        print(f"║  GPU {gpu}: no samples — skipping")
        continue

    sub_cfg['global']['seed'] = cfg.get('global', {}).get('seed', 42) + int(gpu) * 1000

    tmp_config = (tmpdir / f"gpu{gpu}.yaml").resolve()
    with open(tmp_config, 'w') as f:
        yaml.dump(sub_cfg, f, default_flow_style=False)

    outdir = Path(outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco')).resolve()

    cmd = [
        sys.executable, '-m', 'locomotion_framework.orchestrator',
        '-c', str(tmp_config),
        '-g', '0',
        '-o', str(outdir),
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONPATH'] = scripts_dir
    env['PYTHONUNBUFFERED'] = '1'

    gpu_total = sum(sub_cfg['motion_types'][t].get('num_samples', 0)
                    for t in sub_cfg['motion_types'])
    print(f"║  Launching GPU {gpu}: {gpu_total} samples — {list(sub_cfg['motion_types'].keys())}")

    logfile = tmpdir / f"gpu{gpu}.log"
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(logfile, 'w'),
        stderr=subprocess.STDOUT,
        cwd=scripts_dir,
    )
    procs.append((gpu, p, logfile))

print(f"╚══════════════════════════════════════════════════════════════╝")
print(f"\nWaiting for {len(procs)} GPU workers...\n")

for gpu, p, logfile in procs:
    rc = p.wait()
    status = "DONE" if rc == 0 else f"FAILED (exit={rc})"
    print(f"  GPU {gpu}: {status}  (log: {logfile})")

failed = [(gpu, p, logfile) for gpu, p, logfile in procs if p.returncode != 0]
print()

if not failed:
    print("All GPUs completed successfully.")
    outpath = Path(outdir_override or cfg.get('global', {}).get('output_dir', 'outputs/normal_loco'))
    n_dirs = len([d for d in outpath.iterdir() if d.is_dir()]) if outpath.exists() else 0
    print(f"Output: {outpath} ({n_dirs} motion dirs)")
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
else:
    print("Some GPUs failed — check logs in", tmpdir)
    for gpu, p, logfile in failed:
        print(f"  GPU {gpu}: exit={p.returncode}  log={logfile}")
    sys.exit(1)
PYEOF
