#!/bin/bash

set -euo pipefail

: "${CASE_DIR:?CASE_DIR must be set by the case Slurm script}"
: "${RBMD_MPI_GRID:?RBMD_MPI_GRID must be set by the case Slurm script}"

module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

H2O_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RBMD_BIN="${RBMD_BIN:-$H2O_ROOT/../../build/rbmd}"
ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"

export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN=0
export RBMD_DETAILED_PAIR_TIMING=1
export OMP_NUM_THREADS=1

ulimit -c 0
cd "$CASE_DIR"
test -x "$RBMD_BIN"
test -s ./run.json

mpirun -np "$SLURM_NTASKS" --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN -x RBMD_DETAILED_PAIR_TIMING \
  -x RBMD_MPI_GRID -x OMP_NUM_THREADS \
  "$RBMD_BIN" -j ./run.json
