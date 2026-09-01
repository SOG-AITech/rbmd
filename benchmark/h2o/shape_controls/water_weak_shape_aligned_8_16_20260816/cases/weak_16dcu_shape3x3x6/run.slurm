#!/bin/bash
#SBATCH --job-name=rw16_shape336
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

RBMD_BIN=/public/home/acymkxdx47/rbmd/build/rbmd
ROCM_PATH=/public/software/compiler/dtk-25.04.4
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN=0
export OMP_NUM_THREADS=1
ulimit -c 0

cd "${SLURM_SUBMIT_DIR}"
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/water_weak_shape_aligned_8_16_20260816/cases/weak_16dcu_shape3x3x6)"
test -x "$RBMD_BIN"
test -s ./run.json

start_epoch=$(date +%s)
printf 'job_id=%s
nodes=%s
tasks=%s
dcus_per_node=4
replicate=12x6x12
expected_atoms=63055584
steps=1000
start_epoch=%s
' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$start_epoch" > runtime_metadata.txt

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN -x OMP_NUM_THREADS \
  "$RBMD_BIN" -j ./run.json
rc=$?
set -e

end_epoch=$(date +%s)
printf 'end_epoch=%s
elapsed_s=%s
exit_code=%s
' \
  "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
exit "$rc"
