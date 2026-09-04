#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/peo_weak_firstpoint_grid_2x1x2_20260819
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75
[[ $(wc -l < "$ROOT/submitted_jobs.tsv") -eq 1 ]] || exit 0
test -s "$ROOT/build/libforce_mpi_dims.so"
active=$(grep -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases/peo_4dcu_grid_2x1x2/run.slurm" || true)
[[ "$active" == 0 ]]
case_dir="$ROOT/cases/peo_4dcu_grid_2x1x2"
job=$(cd "$case_dir" && sbatch --parsable ./run.slurm)
printf '%s	%s	%s	%s	%s
' "$job" peo_4dcu_grid_2x1x2 "$case_dir" hx1hdnormal "$(date -Iseconds)" >> "$ROOT/submitted_jobs.tsv"
echo "$job"
