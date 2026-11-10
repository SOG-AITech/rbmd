#!/bin/bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/env.sh"
export NB_ENGINE=${1:?engine}
CASE=$(realpath "${2:?case}")
test -n "${SLURM_JOB_ID:-}" || { echo 'Compute allocation required'; exit 2; }
cd "$CASE"
test ! -e run_receipt.json && test ! -e stdout.log
if [[ "$NB_ENGINE" == lammps ]]; then source "$NB_LAMMPS_RUNTIME"; else source "$NB_RUNTIME"; fi
# Validate immutable binaries and grid wrapper before enabling either engine.
python3 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.load(open('case.json'))
for path,sha in [(m['binary'],m['binary_sha256']),(m.get('preload'),m.get('preload_sha256'))]:
 if path:
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha, f'Asset mismatch: {path}'
assert Path(m['data']).is_file(),m['data']
PY
export RBMD_BIN=$(python3 -c 'import json;print(json.load(open("case.json"))["binary"])')
if [[ "$NB_ENGINE" == lammps ]]; then export LAMMPS_BIN="$RBMD_BIN"; fi
unset LD_PRELOAD RBMD_DEBUG_FORCE_STAGE_SYNC RBMD_FORCE_OVERLAP CUDA_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY
export HIP_LAUNCH_BLOCKING=1 RBMD_TIMING_WARMUP_STEPS=0
if [[ "$NB_ENGINE" == rbmd ]]; then
 export RBMD_MPI_GRID=$(python3 -c 'import json;print("x".join(map(str,json.load(open("case.json"))["mpi_grid"])))')
 PRELOAD=$(python3 -c 'import json;print(json.load(open("case.json")).get("preload") or "")')
 if [[ -n "$PRELOAD" ]]; then export LD_PRELOAD="$PRELOAD"; export RBMD_PAIR_KERNEL=optimized RBMD_PAIR_VALIDATE=1; fi
fi
python3 "$ROOT/common/execute.py" "$CASE"
