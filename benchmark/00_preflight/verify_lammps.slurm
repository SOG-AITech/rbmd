#!/bin/bash
#SBATCH --job-name=NB-device-mpi
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --time=00:10:00
set -euo pipefail
ROOT=${NB_ROOT:-/public/home/acymkxdx47/ReviewRBMD/NewBenchmark}
source "$ROOT/env.sh"
source "$NB_LAMMPS_RUNTIME"
OUT="$ROOT/validation/device-mpi-$SLURM_JOB_ID"
mkdir -p "$OUT"
cd "$OUT"
"$LAMMPS_BIN" -h > binary-help.txt
python3 - <<'PY'
import re
s=open('binary-help.txt').read()
assert re.search(r'KOKKOS FFT library\s*=\s*hipfft',s,re.I)
assert re.search(r'FFT precision\s*=\s*single',s)
assert 'shake/kk' in s
PY
cat > run.in <<'IN'
units lj
atom_style charge
lattice sc 0.125
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1
set group all charge 0
variable charge atom 2*(id%2)-1
set group all charge v_charge
pair_style lj/cut/coul/long 3
pair_coeff * * 1 1
kspace_style pppm 1e-4
velocity all create 1 162731
fix thermostat all nvt temp 1 1 0.1
thermo 1
run 5
IN
mpirun --bind-to none -np 2 --map-by ppr:1:node "$LAMMPS_BIN" -k on g 1 -sf kk -pk kokkos gpu/aware on -in run.in -log log.lammps > stdout.log 2>&1
python3 - "$ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'common'))
from read_thermo import read
rows=read(Path('stdout.log'))
assert rows[-1]['Step']==5
PY
sha256sum "$LAMMPS_BIN" > binary.sha256
mpirun --version > mpi-version.txt
ldd "$LAMMPS_BIN" > libraries.txt
printf '%s\n' 'PASS: two-node PPPM device-buffer MPI smoke; water SHAKE and production scaling still require separate runs.' > result.txt
