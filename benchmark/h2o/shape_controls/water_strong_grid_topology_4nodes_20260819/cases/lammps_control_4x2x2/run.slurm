#!/bin/bash
#SBATCH --job-name=wlg_control
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
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
export HIP_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
ulimit -c 0

cd "$SLURM_SUBMIT_DIR"
test "$(pwd -P)" = "$(readlink -f /public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/cases/lammps_control_4x2x2)"
test -x "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp"
test -s "/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/in.water"
rm -f VALIDATED_COMPLETE log.lammps

start_epoch=$(date +%s)
hostlist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
printf 'job_id=%s
method=LAMMPS-Kokkos
nodes=4
tasks=16
dcus_per_node=4
replicate=6x6x6
expected_atoms=15763896
steps=1000
processor_grid=4x2x2
cutoff=9/9
pppm_accuracy=1e-4
newton=off
timer=full_sync
hostlist=%s
lammps_sha256=1638cd0091394daa8f61940f5f33f39535df632676766114060b3717eeb435f8
start_epoch=%s
'   "$SLURM_JOB_ID" "$hostlist" "$start_epoch" > runtime_metadata.txt

set +e
mpirun -np "$SLURM_NTASKS" --mca pml ucx   -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH   -x HIP_LAUNCH_BLOCKING -x OMP_NUM_THREADS   "/public/home/acymkxdx47/software/lammps/build-kokkos/lmp" -k on g 4 -sf kk   -pk kokkos newton off neigh full gpu/aware on   -var DATA_FILE "/public/home/acymkxdx47/RBMD2/benchmark/data/h2o.data"   -var RX 6 -var RY 6 -var RZ 6   -var PX 4 -var PY 2 -var PZ 2   -var RUNSTEPS 1000   -in "/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/in.water" -log log.lammps
rc=$?
set -e

end_epoch=$(date +%s)
printf 'end_epoch=%s
elapsed_s=%s
exit_code=%s
'   "$end_epoch" "$((end_epoch-start_epoch))" "$rc" >> runtime_metadata.txt
[[ "$rc" == 0 ]]
actual_atoms=$(awk '/Loop time of/{value=$12} END{print value}' log.lammps)
actual_steps=$(awk '/Loop time of/{value=$9} END{print value}' log.lammps)
printf 'actual_atoms=%s
actual_steps=%s
' "$actual_atoms" "$actual_steps" >> runtime_metadata.txt
[[ "$actual_atoms" == "15763896" ]]
[[ "$actual_steps" == "1000" ]]
grep -q '^Pair ' log.lammps
grep -q '^Kspace ' log.lammps
touch VALIDATED_COMPLETE
