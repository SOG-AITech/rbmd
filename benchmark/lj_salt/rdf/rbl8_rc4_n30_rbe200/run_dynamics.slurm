#!/bin/bash
#SBATCH -J ls_rdf_n30p200
#SBATCH -p hx1hdnormal01
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --mem=60G
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.out
#SBATCH --time=02:00:00

set -euo pipefail
trap 'rc=$?; echo "ERROR rc=$rc line=$LINENO command=$BASH_COMMAND"' ERR
echo "SLURM_JOB_START job=${SLURM_JOB_ID:-unknown} host=$(hostname) time=$(date -Iseconds)"

module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3 compiler/cmake/3.24.1 compiler/dtk/25.04.4

RUN_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RBMD_BIN="${RBMD_BIN:-/public/home/acymkxdx47/rbmd/build/rbmd}"
ROCM_PATH="${ROCM_PATH:-/public/software/compiler/dtk-25.04.4}"
export ROCM_PATH
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export UCX_TLS="${UCX_TLS:-^cma,rocm_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export RBMD_DEBUG_NEIGHBOR_SKIN=0
ulimit -c 0

cd "$RUN_DIR"
test -x "$RBMD_BIN"
test -s /public/home/acymkxdx47/RBMD2/benchmark/data/lj_salt.data
test -s ./rdf_lj_salt_cut8.csv

echo "CASE=LJSALT_CUT8_ALIGNED EQ=20000 PROD=20000 RDF=800bins_rmax8_sample10"
echo "PARAMETERS=RBL_8_4_30,RBE_P200,seed_20260811,dt_0.001,NVT_1_1_1,tloop_1,tchain_3"

mpirun -np 4 --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN "$RBMD_BIN" -j ./equil.json

mv rbmd.trj equil.trj
mv rbmd.log equil_rbmd.log
mv temperature.txt equil_temperature.txt
mv thermo.txt equil_thermo.txt
python3 ./trajectory_to_data.py --trajectory ./equil.trj --output /public/home/acymkxdx47/RBMD2/benchmark/data/lj_salt.data

mpirun -np 4 --mca pml ucx \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x LD_LIBRARY_PATH -x ROCM_PATH \
  -x RBMD_DEBUG_NEIGHBOR_SKIN "$RBMD_BIN" -j ./prod.json

echo "RBMD_EQ_PROD_EXIT=0 time=$(date -Iseconds)"
