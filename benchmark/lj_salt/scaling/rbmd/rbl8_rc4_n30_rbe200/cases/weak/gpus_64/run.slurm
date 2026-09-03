#!/bin/bash
#SBATCH -J ng_lj_w64
#SBATCH -p hx1hdnormal01
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:8
#SBATCH --mem=120G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=02:00:00

set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

RBMD_BIN="/public/home/acymkxdx47/rbmd/build/rbmd"
ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN=0
export OMP_NUM_THREADS=1
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"

printf 'job_id=%s
nodes=%s
tasks=%s
gpus_per_node=8
reserved_cpus_per_rank=4
compute_threads_per_rank=1
requested_mem_per_node=120G
start_epoch=%s
' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$(date +%s)" \
  > runtime_metadata.txt
start_epoch=$(date +%s)
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
