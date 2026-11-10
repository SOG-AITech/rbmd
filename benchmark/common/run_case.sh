#!/bin/bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/env.sh"
ENGINE=${1:?rbmd or lammps}; CASE=$(realpath "${2:?case directory}")
test -n "${SLURM_JOB_ID:-}" || { echo 'Run inside a Slurm allocation; no MD on login node' >&2; exit 2; }
test "$ENGINE" = rbmd || test "$ENGINE" = lammps
export NB_LAMMPS_ROLE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("lammps_role","production"))' "$CASE/case.json")
if [[ "$ENGINE" == lammps ]]; then
  source "$NB_LAMMPS_RUNTIME"
else
  source "$NB_RUNTIME"
fi
unset HIP_LAUNCH_BLOCKING CUDA_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY
unset RBMD_DEBUG_FORCE_STAGE_SYNC RBMD_FORCE_OVERLAP
cd "$CASE"
test ! -e run_receipt.json && test ! -e stdout.log || { echo 'Existing run; use a fresh case directory' >&2; exit 2; }
python3 "$ROOT/00_preflight/audit.py" --case "$CASE" --engine "$ENGINE" --output preflight.json
WARMUP=$(python3 -c 'import json;print(json.load(open("case.json"))["warmup"])')
SYNC=$(python3 -c 'import json;print(json.load(open("case.json"))["launch_blocking"])')
if [[ "$SYNC" == on ]]; then export HIP_LAUNCH_BLOCKING=1; fi
export RBMD_TIMING_WARMUP_STEPS=$WARMUP
export NB_ENGINE=$ENGINE
python3 "$ROOT/common/execute.py" "$CASE"
