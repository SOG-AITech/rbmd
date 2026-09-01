#!/bin/bash
#SBATCH -J ls_rdf_n30p200_a
#SBATCH -p hx1hdnormal01
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=64G
#SBATCH --output=rdf_parallel_%j.out
#SBATCH --error=rdf_parallel_%j.out
#SBATCH --time=04:00:00

set -euo pipefail
trap 'rc=$?; echo "ERROR rc=$rc line=$LINENO command=$BASH_COMMAND"' ERR

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
test -s ./rbmd.trj
test -s ./rdf_lj_salt_cut8.csv

python3 ./parallel_rdf.py \
  --trajectory ./rbmd.trj \
  --reference ./rdf_lj_salt_cut8.csv \
  --outdir ./analysis \
  --workers 32 \
  --frames-per-task 8 \
  --start-step 10 \
  --end-step 20000 \
  --interval 10 \
  --threshold-percent 3.0

echo "RDF_PARALLEL_EXIT=0 time=$(date -Iseconds)"
