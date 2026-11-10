#!/bin/bash
#SBATCH --job-name=NB-reference
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=dcu:1
#SBATCH --time=01:00:00
set -euo pipefail
CASE=${1:?CPU reference case directory}
python3 - "$CASE/case.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['lammps_role']=='reference'
PY
exec bash "$CASE/run.sh"
