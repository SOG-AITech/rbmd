#!/bin/bash
#SBATCH --job-name=h2o_weak_64
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CASE_DIR
export RBMD_MPI_GRID=4x4x4
source "$CASE_DIR/../../../../run_common.sh"
