#!/bin/bash
#SBATCH -J match_water_weak_n4
#SBATCH -p hx1hdnormal
#SBATCH -N 4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
module load compiler/gcc/9.3.0             sghpc-mpi-gcc/26.3             compiler/cmake/3.24.1             compiler/dtk/25.04.4

ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export HIP_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
ulimit -c 0

cd "$SLURM_SUBMIT_DIR"
printf 'job_id=%s\nnodes=%s\ntasks=%s\nhost=%s\nstart_epoch=%s\n'   "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS"   "$(hostname)" "$(date +%s)" > runtime_metadata.txt

mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS   -x UCX_MEMTYPE_CACHE   -x LD_LIBRARY_PATH   -x ROCM_PATH   -x HIP_LAUNCH_BLOCKING   -x OMP_NUM_THREADS   "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp"   -k on g 4   -sf kk   -pk kokkos newton off neigh full gpu/aware on   -var DATA_FILE "/public/home/acymkxdx47/RBMD2/benchmark/data/h2o.data"   -var RX "12"   -var RY "12"   -var RZ "6"   -var RUNSTEPS "2000"   -in "/public/home/acymkxdx47/rbmd/build/lammps_kokkos_pppm_water9_complete64_noexclusive_20260814/in.water"   -log log.lammps

rc=$?
printf 'end_epoch=%s\nelapsed_s=%s\nexit_code=%s\n'   "$(date +%s)" "$(($(date +%s)-$(awk -F= '/start_epoch/{print $2}' runtime_metadata.txt)))"   "$rc" >> runtime_metadata.txt
exit "$rc"
