#!/bin/bash
#SBATCH --job-name=peo_w4_b
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --time=01:00:00
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
export RBMD_DEBUG_NEIGHBOR_SKIN="${RBMD_DEBUG_NEIGHBOR_SKIN:--1}"
export RBMD_DEBUG_NEIGHBOR_SKIN_EVERY="${RBMD_DEBUG_NEIGHBOR_SKIN_EVERY:--100}"
export OMP_NUM_THREADS=1
ulimit -c 0

cd "$SLURM_SUBMIT_DIR"
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/peo_weak_firstpoint_repeat_20260819/cases/weak_4dcu_repeat_b)"
test -x "$RBMD_BIN"
test -s ./test.json
test -s "/public/home/acymkxdx47/RBMD2/benchmark/data/peo_litfsi.data"
rm -f VALIDATED_COMPLETE rbmd.log

base_atoms=$(awk '{gsub(/\r/,""); if($2=="atoms"){print $1; exit}}' "/public/home/acymkxdx47/RBMD2/benchmark/data/peo_litfsi.data")
[[ "$base_atoms" == "3904800" ]]
grep -Eq '"cut_off"[[:space:]]*:[[:space:]]*12' test.json
grep -Eq '"r_core"[[:space:]]*:[[:space:]]*6' test.json
grep -Eq '"neighbor_sample_num"[[:space:]]*:[[:space:]]*200' test.json
grep -Eq '"coulomb_sample_num"[[:space:]]*:[[:space:]]*200' test.json
grep -Eq '"num_steps"[[:space:]]*:[[:space:]]*2000' test.json

start_epoch=$(date +%s)
hostlist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
printf 'job_id=%s
method=RBMD
nodes=%s
tasks=%s
dcus_per_node=4
atoms=%s
steps=2000
cutoff_core=12/6
rbl_sample=200
rbe_sample=200
hostlist=%s
rbmd_sha256=33d1616445a2b7d911bb1709da7bd4122afde9de917d4f6ede14cb02219cb75f
source_sha256=3a23c025826b842e521497ba0e40243747a3dda97a7fec8bbbc3ba275a61a1c3
start_epoch=%s
'   "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$base_atoms"   "$hostlist" "$start_epoch" > runtime_metadata.txt

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   -x RBMD_DEBUG_NEIGHBOR_SKIN -x RBMD_DEBUG_NEIGHBOR_SKIN_EVERY   -x OMP_NUM_THREADS   "$RBMD_BIN" -j ./test.json
rc=$?
set -e

end_epoch=$(date +%s)
printf 'end_epoch=%s
elapsed_s=%s
exit_code=%s
'   "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
[[ "$rc" == 0 ]]
grep -q 'Finish' rbmd.log
grep -q 'Short-Range' rbmd.log
grep -q 'Long-Range' rbmd.log
touch VALIDATED_COMPLETE
