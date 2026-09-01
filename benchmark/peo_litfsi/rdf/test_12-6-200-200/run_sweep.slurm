#!/bin/bash
#SBATCH -J rbsweep
#SBATCH -p hx1hdnormal
#SBATCH -N 4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --time=08:00:00
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.out
# NOTE: --array is NOT set here on purpose. Set it at submit time based on
# how many combos gen_sweep.py generated, e.g.:
#   sbatch --array=0-23%6 run_sweep.slurm
# (gen_sweep.py prints the exact command for you at the end of its output.)
# The %6 caps how many array tasks run at once, so you don't hog the whole
# partition; drop it (or change the number) if you want more/fewer concurrent.

set -euo pipefail

mkdir -p logs

module purge && module load compiler/gcc/9.3.0  sghpc-mpi-gcc/26.3    compiler/cmake/3.24.1     compiler/dtk/25.04.4

RBMD_BIN="/public/home/acymkxdx47/rbmd/build/rbmd"
ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"

export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN="${RBMD_DEBUG_NEIGHBOR_SKIN:-1}"
export RBMD_DEBUG_NEIGHBOR_SKIN_EVERY="${RBMD_DEBUG_NEIGHBOR_SKIN_EVERY:-100}"
ulimit -c 0

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this script must be submitted with --array (see comment at top)." >&2
  exit 1
fi

TABLE="sweep_table.tsv"
TAG=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" '$1==idx {print $2}' "$TABLE")
RUNDIR="runs/${TAG}"

if [[ -z "$TAG" || ! -d "$RUNDIR" ]]; then
  echo "ERROR: no run directory for array index $SLURM_ARRAY_TASK_ID (tag='$TAG'). Did you run gen_sweep.py first?" >&2
  exit 1
fi

echo "array task $SLURM_ARRAY_TASK_ID -> combo: $TAG"
cd "$RUNDIR"

mpirun -np 16 --mca pml ucx \
  -x UCX_TLS \
  -x UCX_MEMTYPE_CACHE \
  -x LD_LIBRARY_PATH \
  -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN \
  -x RBMD_DEBUG_NEIGHBOR_SKIN_EVERY \
  "$RBMD_BIN" -j test_rshell.json

echo "RBMD_EXIT=0 tag=${TAG}"
