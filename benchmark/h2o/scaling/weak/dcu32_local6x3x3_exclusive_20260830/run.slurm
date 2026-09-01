#!/bin/bash
#SBATCH -J h2o_w32_sync
#SBATCH -p hx1hdnormal01
#SBATCH -N 8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=00:30:00

set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RBMD_BIN="${RBMD_BIN:-$CASE_DIR/../../../../../build/rbmd}"
ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN=0
export OMP_NUM_THREADS=1
export RBMD_DETAILED_PAIR_TIMING=1
export RBMD_MPI_GRID=4x4x2
ulimit -c 0
cd "$CASE_DIR"
test -x "$RBMD_BIN"

printf 'job_id=%s
nodes=%s
tasks=%s
gpus_per_node=4
reserved_cpus_per_rank=4
compute_threads_per_rank=1
requested_mem_per_node=60G
start_epoch=%s
' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$(date +%s)" \
  > runtime_metadata.txt
start_epoch=$(date +%s)
set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN -x OMP_NUM_THREADS -x RBMD_MPI_GRID -x RBMD_DETAILED_PAIR_TIMING \
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
