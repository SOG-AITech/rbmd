#!/bin/bash
#SBATCH -J rbmd-strong-64gpu
#SBATCH -p hx1hdnormal
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:8
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --time=05:00:00



set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

RBMD_BIN="/public/home/acymkxdx47/rbmd/build/rbmd"
ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN="${RBMD_DEBUG_NEIGHBOR_SKIN:-1}"
export RBMD_DEBUG_NEIGHBOR_SKIN_EVERY="${RBMD_DEBUG_NEIGHBOR_SKIN_EVERY:-100}"
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"

printf 'job_id=%s\nnodes=%s\ntasks=%s\nhost=%s\nstart_epoch=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$(hostname)" "$(date +%s)" \
  > runtime_metadata.txt

start_epoch=$(date +%s)
set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN -x RBMD_DEBUG_NEIGHBOR_SKIN_EVERY \
  "$RBMD_BIN" -j ../test.json
rc=$?
set -e
end_epoch=$(date +%s)
printf 'end_epoch=%s\nelapsed_s=%s\nexit_code=%s\n' \
  "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
exit "$rc"
