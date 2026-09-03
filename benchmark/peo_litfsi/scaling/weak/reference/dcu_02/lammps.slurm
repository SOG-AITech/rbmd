#!/bin/bash
#SBATCH -J lammps-weak-2m
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:2
#SBATCH --time=08:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.out
#SBATCH --exclusive

set -euo pipefail



module purge && module load compiler/gcc/9.3.0  sghpc-mpi-gcc/26.3    compiler/cmake/3.24.1     compiler/dtk/25.04.4

LAMMPS_BIN="/public/home/acymkxdx47/software/lammps/build-kokkos/lmp"
ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"

export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export HIP_LAUNCH_BLOCKING=1
ulimit -c 0

# One rank per DCU, matching --ntasks-per-node=4 --gres=dcu:4.
# Without "-k on ... -sf kk" the run silently falls back to the CPU path and
# the timings are meaningless, so these flags are not optional.
mpirun -np "${SLURM_NTASKS}" --mca pml ucx \
  -x UCX_TLS \
  -x UCX_MEMTYPE_CACHE \
  -x LD_LIBRARY_PATH \
  -x ROCM_PATH \
  "$LAMMPS_BIN" -k on g "${SLURM_GPUS_ON_NODE}" -sf kk -pk kokkos newton off neigh full gpu/aware on -in run.in
