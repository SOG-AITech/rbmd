#!/bin/bash
#SBATCH --job-name=ws_r64_kiss
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
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
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_kiss_matched_recheck_20260820/cases/rbmd_64dcu_grid4x4x4)"
rm -f VALIDATED_COMPLETE RUN_EXIT_ZERO runtime_metadata.txt rank_map.tsv slurm_job_record.txt
scontrol show job -o "$SLURM_JOB_ID" > slurm_job_record.txt
grep -F "WorkDir=$(pwd -P)" slurm_job_record.txt >/dev/null

start_epoch=$(date +%s)
hostlist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
printf 'job_id=%s
method=RBMD
nodes=16
tasks=64
dcus_per_node=4
replicate=6x6x6
expected_atoms=15763896
steps=1000
processor_grid=4x4x4
local_replica_ratio=1.5x1.5x1.5
hostlist=%s
start_epoch=%s
'   "$SLURM_JOB_ID" "$hostlist" "$start_epoch" > runtime_metadata.txt

mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   /bin/bash -lc 'printf "%s\t%s\t%s\n" "$OMPI_COMM_WORLD_RANK" "$(hostname)" "$OMPI_COMM_WORLD_LOCAL_RANK"'   | sort -n > rank_map.tsv
test "$(wc -l < rank_map.tsv)" -eq "$SLURM_NTASKS"

export RBMD_DEBUG_NEIGHBOR_SKIN=0
printf 'cutoff_core=9/5
rbl_sample=30
rbe_sample=200
rbmd_sha256=33d1616445a2b7d911bb1709da7bd4122afde9de917d4f6ede14cb02219cb75f
run_sha256=dbd7f0658f143df5d57a300a90fa83f3ea7a09f3a0ddb293b56ad278ac4cf96c
' >> runtime_metadata.txt
test -x "/public/home/acymkxdx47/rbmd/build/rbmd"
test -s ./run.json

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   -x RBMD_DEBUG_NEIGHBOR_SKIN -x OMP_NUM_THREADS   "/public/home/acymkxdx47/rbmd/build/rbmd" -j ./run.json
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
