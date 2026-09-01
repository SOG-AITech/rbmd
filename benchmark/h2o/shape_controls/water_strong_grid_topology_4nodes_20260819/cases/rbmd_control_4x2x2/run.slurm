#!/bin/bash
#SBATCH --job-name=wrg_control
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4
ROCM_PATH=/public/software/compiler/dtk-25.04.4
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN=0
export RBMD_MPI_GRID=4x2x2
export LD_PRELOAD="/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/build/libforce_mpi_dims.so${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=1
ulimit -c 0

cd "$SLURM_SUBMIT_DIR"
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/cases/rbmd_control_4x2x2)"
test -x "/public/home/acymkxdx47/rbmd/build/rbmd"
test -s "/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/build/libforce_mpi_dims.so"
test -s ./run.json
rm -f VALIDATED_COMPLETE rbmd.log

start_epoch=$(date +%s)
hostlist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
printf 'job_id=%s
method=RBMD
nodes=4
tasks=16
dcus_per_node=4
replicate=6x6x6
expected_atoms=15763896
steps=1000
forced_mpi_grid=4x2x2
cutoff_core=9/5
rbl_sample=30
rbe_sample=200
hostlist=%s
rbmd_sha256=33d1616445a2b7d911bb1709da7bd4122afde9de917d4f6ede14cb02219cb75f
preload_sha256=%s
start_epoch=%s
'   "$SLURM_JOB_ID" "$hostlist" "$(sha256sum "/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/build/libforce_mpi_dims.so" | awk '{print $1}')"   "$start_epoch" > runtime_metadata.txt

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   -x RBMD_DEBUG_NEIGHBOR_SKIN -x RBMD_MPI_GRID -x LD_PRELOAD   -x OMP_NUM_THREADS   "/public/home/acymkxdx47/rbmd/build/rbmd" -j ./run.json
rc=$?
set -e

end_epoch=$(date +%s)
printf 'end_epoch=%s
elapsed_s=%s
exit_code=%s
'   "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
[[ "$rc" == 0 ]]
grep -q 'Finish' rbmd.log
grid_intercept_count=$(grep -h 'RBMD_MPI_GRID applied nnodes=16 dims=4x2x2' \
  "slurm-${SLURM_JOB_ID}.out" "slurm-${SLURM_JOB_ID}.err" 2>/dev/null | wc -l)
printf 'grid_intercept_count=%s
' "$grid_intercept_count" >> runtime_metadata.txt
(( grid_intercept_count > 0 ))
grep -q 'Short-Range' rbmd.log
grep -q 'Long-Range' rbmd.log
touch VALIDATED_COMPLETE
