#!/bin/bash
#SBATCH --job-name=ws_l32_kiss
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

export ROCM_PATH=/public/software/compiler/dtk-25.04.4
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export OMP_NUM_THREADS=1
ulimit -c 0

cd "$SLURM_SUBMIT_DIR"
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_kiss_matched_recheck_20260820/cases/lammps_32dcu_grid4x4x2)"
rm -f VALIDATED_COMPLETE RUN_EXIT_ZERO runtime_metadata.txt rank_map.tsv slurm_job_record.txt
scontrol show job -o "$SLURM_JOB_ID" > slurm_job_record.txt
grep -F "WorkDir=$(pwd -P)" slurm_job_record.txt >/dev/null

start_epoch=$(date +%s)
hostlist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
printf 'job_id=%s
method=LAMMPS-Kokkos-KISS
nodes=8
tasks=32
dcus_per_node=4
replicate=6x6x6
expected_atoms=15763896
steps=1000
processor_grid=4x4x2
local_replica_ratio=1.5x1.5x3
hostlist=%s
start_epoch=%s
'   "$SLURM_JOB_ID" "$hostlist" "$start_epoch" > runtime_metadata.txt

mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   /bin/bash -lc 'printf "%s\t%s\t%s\n" "$OMPI_COMM_WORLD_RANK" "$(hostname)" "$OMPI_COMM_WORLD_LOCAL_RANK"'   | sort -n > rank_map.tsv
test "$(wc -l < rank_map.tsv)" -eq "$SLURM_NTASKS"

export HIP_LAUNCH_BLOCKING=1
printf 'cutoff=9/9
pppm_accuracy=1e-4
newton=off
timer=full_sync
hip_launch_blocking=1
fft_library=KISS
lammps_sha256=1638cd0091394daa8f61940f5f33f39535df632676766114060b3717eeb435f8
input_sha256=0a4fd276086ae1bf13061f8e4f35eaeb5d528c2edfeab276948b5671d034d11a
' >> runtime_metadata.txt
test -x "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp"
test -s "/public/home/acymkxdx47/RBMD2/benchmark/data/h2o.data"
test -s "/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_kiss_matched_recheck_20260820/in.water"

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   -x HIP_LAUNCH_BLOCKING -x OMP_NUM_THREADS   "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp" -k on g 4 -sf kk   -pk kokkos newton off neigh full gpu/aware on   -var DATA_FILE "/public/home/acymkxdx47/RBMD2/benchmark/data/h2o.data"   -var RX 6 -var RY 6 -var RZ 6   -var PX 4 -var PY 4 -var PZ 2   -var RUNSTEPS 1000   -in "/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_kiss_matched_recheck_20260820/in.water" -log log.lammps
rc=$?
set -e

end_epoch=$(date +%s)
printf 'end_epoch=%s
elapsed_s=%s
exit_code=%s
'   "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
if [[ "$rc" == 0 ]]; then
  touch RUN_EXIT_ZERO
fi
exit "$rc"
