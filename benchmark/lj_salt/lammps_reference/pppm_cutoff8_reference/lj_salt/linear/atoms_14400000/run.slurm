#!/bin/bash
#SBATCH -J lj8c_l14
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export HIP_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
ulimit -c 0

cd "$SLURM_SUBMIT_DIR"

base_atoms=$(awk '/^[[:space:]]*[0-9]+[[:space:]]+atoms[[:space:]]*$/ {print $1; exit}' "/public/home/acymkxdx47/RBMD2/benchmark/data/lj_salt.data")
expected_atoms=$((base_atoms * 4 * 6 * 3))
if [[ "$base_atoms" -ne 200000 || "$expected_atoms" -ne 14400000 ]]; then
  printf 'PRECHECK_FAILED base_atoms=%s replicate=4x6x3 expected=%s target=14400000
'     "$base_atoms" "$expected_atoms" >&2
  exit 42
fi
if grep -Eq '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$0"; then
  echo 'PRECHECK_FAILED: --exclusive is forbidden' >&2
  exit 43
fi

printf 'job_id=%s
nodes=%s
tasks=%s
base_atoms=%s
replicate=4x6x3
requested_atoms=14400000
expected_atoms=%s
cutoff=8/8
exclusive=no
start_epoch=%s
'   "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$base_atoms"   "$expected_atoms" "$(date +%s)" > runtime_metadata.txt

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   -x HIP_LAUNCH_BLOCKING -x OMP_NUM_THREADS   "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp" -k on g 4 -sf kk   -pk kokkos newton off neigh full gpu/aware on   -var DATA_FILE "/public/home/acymkxdx47/RBMD2/benchmark/data/lj_salt.data" -var RX "4" -var RY "6" -var RZ "3"   -var RUNSTEPS "2000" -in "/public/home/acymkxdx47/rbmd/build/lammps_kokkos_pppm_lj8_corrected_noexclusive_20260813/in.lj_salt" -log log.lammps
rc=$?
set -e

actual_atoms=$(awk '/Loop time of/ {value=$12} END {print value}' log.lammps 2>/dev/null)
printf 'actual_atoms=%s
end_epoch=%s
exit_code=%s
'   "${actual_atoms:-missing}" "$(date +%s)" "$rc" >> runtime_metadata.txt
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
if [[ "${actual_atoms:-}" != "14400000" ]]; then
  printf 'POSTCHECK_FAILED actual_atoms=%s target=14400000
' "${actual_atoms:-missing}" >&2
  exit 44
fi
touch VALIDATED_COMPLETE
