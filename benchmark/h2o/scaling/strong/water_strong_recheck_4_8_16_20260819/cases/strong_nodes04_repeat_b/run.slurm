#!/bin/bash
#SBATCH --job-name=wat_s04_b
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
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_recheck_4_8_16_20260819/cases/strong_nodes04_repeat_b)"
test -x "$RBMD_BIN"
test -s ./run.json

start_epoch=$(date +%s)
hostlist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
printf 'job_id=%s
nodes=%s
tasks=%s
dcus_per_node=4
replicate=6x6x6
expected_atoms=15763896
steps=1000
expected_mpi_grid=4x2x2
local_box_angstrom=135x270x270
hostlist=%s
rbmd_sha256=33d1616445a2b7d911bb1709da7bd4122afde9de917d4f6ede14cb02219cb75f
source_run_sha256=dbd7f0658f143df5d57a300a90fa83f3ea7a09f3a0ddb293b56ad278ac4cf96c
start_epoch=%s
' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" \
  "$hostlist" "$start_epoch" > runtime_metadata.txt

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
