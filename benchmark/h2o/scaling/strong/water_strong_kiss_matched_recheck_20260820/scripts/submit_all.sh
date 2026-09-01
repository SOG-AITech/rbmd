#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_kiss_matched_recheck_20260820
MANIFEST="$ROOT/manifest.tsv"
SUBMITTED="$ROOT/submitted_jobs.tsv"
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75
test "$(grep -R -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include=run.slurm | awk -F: '{s+=$2} END{print s+0}')" = 0
test "$(awk 'END{print NR-1}' "$SUBMITTED")" = 0

previous=""
while IFS=$'\t' read -r order method name nodes dcus atoms replicate steps grid ratio case_dir; do
  [[ "$order" == "order" ]] && continue
  dependency=()
  if [[ -n "$previous" ]]; then
    dependency=(--dependency="afterany:$previous")
  fi
  job=$(cd "$case_dir" && sbatch --parsable "${dependency[@]}" ./run.slurm)
  printf '%s\t%s\t%s\t%s\t%s\n' "$job" "$name" "$case_dir" hx1hdnormal "$(date -Iseconds)" >> "$SUBMITTED"
  echo "submitted $job $name dependency=${previous:-none}"
  previous="$job"
done < "$MANIFEST"
