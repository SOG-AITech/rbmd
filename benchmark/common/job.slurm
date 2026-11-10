#!/bin/bash
#SBATCH --job-name=NewBenchmark
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:4
#SBATCH --time=01:00:00
#SBATCH --output=NewBenchmark-%j.out
set -euo pipefail
CASE=${1:?Usage: sbatch common/job.slurm /absolute/case/directory}
exec bash "$CASE/run.sh"
