#!/bin/bash
#SBATCH --job-name=h2o_rdf
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CASE_DIR
export RBMD_MPI_GRID=2x2x1
source "$CASE_DIR/../run_common.sh"
