#!/bin/bash
#SBATCH --job-name=lw64_shape336
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --time=03:00:00
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

cd "${SLURM_SUBMIT_DIR}"
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/lammps_water_weak_shape_aligned_8_16_64_20260816/cases/weak_64dcu_shape3x3x6)"
test -x "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp"
test -s "/public/home/acymkxdx47/RBMD2/benchmark/data/h2o.data"
test -s "/public/home/acymkxdx47/rbmd/build/new_gyf_test/lammps_water_weak_shape_aligned_8_16_64_20260816/in.water"
rm -f VALIDATED_COMPLETE

start_epoch=$(date +%s)
printf 'job_id=%s
nodes=%s
tasks=%s
dcus_per_node=4
replicate=12x12x24
processor_grid=4x4x4
expected_atoms=252222336
steps=1000
start_epoch=%s
' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$start_epoch" > runtime_metadata.txt

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x HIP_LAUNCH_BLOCKING -x OMP_NUM_THREADS \
  "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp" \
  -k on g 4 -sf kk \
  -pk kokkos newton off neigh full gpu/aware on \
  -var DATA_FILE "/public/home/acymkxdx47/RBMD2/benchmark/data/h2o.data" \
  -var RX "12" -var RY "12" -var RZ "24" \
  -var PX "4" -var PY "4" -var PZ "4" \
  -var RUNSTEPS "1000" \
  -in "/public/home/acymkxdx47/rbmd/build/new_gyf_test/lammps_water_weak_shape_aligned_8_16_64_20260816/in.water" -log log.lammps
rc=$?
set -e

end_epoch=$(date +%s)
printf 'end_epoch=%s
elapsed_s=%s
exit_code=%s
' \
  "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi

actual_atoms=$(awk '/Loop time of/ {value=$12} END {print value}' log.lammps 2>/dev/null)
actual_steps=$(awk '/Loop time of/ {value=$9} END {print value}' log.lammps 2>/dev/null)
printf 'actual_atoms=%s
actual_steps=%s
' \
  "${actual_atoms:-missing}" "${actual_steps:-missing}" >> runtime_metadata.txt
if [[ "${actual_atoms:-}" != "252222336" || "${actual_steps:-}" != "1000" ]]; then
  printf 'POSTCHECK FAILED actual_atoms=%s target=252222336 actual_steps=%s target_steps=1000
' \
    "${actual_atoms:-missing}" "${actual_steps:-missing}" >&2
  exit 44
fi
touch VALIDATED_COMPLETE
