#!/bin/bash
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=dcu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=${NB_ROOT:?Set NB_ROOT to the NewBenchmark directory}
CASE=$(realpath "${1:?postprocessing directory}")
test -n "${SLURM_JOB_ID:-}"
source "$ROOT/env.sh"
source "$NB_LAMMPS_RUNTIME"
cd "$CASE"
test ! -e stdout.log && test ! -e receipt.json
python3 - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('postprocess.json').read_text())
p=Path(m['source_case']).parents[3]/'checks-round1'/f"{m['system']}-{m['source_engine']}.json"
c=json.loads(p.read_text())
assert c['status'] in ('PASS','MOLECULAR_GEOMETRY_FAILED')
assert c['frames']==1001 and c['finite_values'] and c['complete_unique_ids'] and c['identity_matches_source'] and c['box_constant']
Path('trajectory_check.json').write_text(json.dumps(c,indent=2)+'\n')
PY
set +e
mpirun -np 4 --bind-to none "$LAMMPS_BIN" -in run.in > stdout.log 2>&1
code=$?
set -e
python3 - "$code" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
Path('receipt.json').write_text(json.dumps(dict(exit_code=int(sys.argv[1]),job=os.environ['SLURM_JOB_ID'],
    binary=os.environ['LAMMPS_BIN'],binary_sha256=sha(os.environ['LAMMPS_BIN']),
    input_sha256=sha('run.in'),ranks=4,mode='LAMMPS CPU native computes; no dynamics'),indent=2)+'\n')
PY
test "$code" -eq 0
python3 "$ROOT/04_dynamics_rerun/reduce_native.py" "$CASE" > reduction.log 2>&1
