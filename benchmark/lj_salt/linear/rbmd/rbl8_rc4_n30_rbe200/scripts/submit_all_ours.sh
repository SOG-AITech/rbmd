#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/rbmd_linear_subnode_20260815/corrected_ljsalt_RBE200_RBL30
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75
test "$(grep -R -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include='run.slurm' | awk -F: '{s+=$2} END{print s+0}')" = 0
dep=""
while IFS=$'\t' read -r system mode name gpus rep atoms steps case_dir; do
  [[ "$system" == "system" ]] && continue
  if awk -F'\t' -v d="$case_dir" 'NR>1 && $5==d{f=1} END{exit !f}' "$ROOT/submitted_jobs.tsv"; then
    echo "skip recorded $case_dir"
    continue
  fi
  args=()
  [[ -n "$dep" ]] && args=(--dependency="afterany:$dep")
  job=$(cd "$case_dir" && sbatch --parsable "${args[@]}" ./run.slurm)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$system" "$mode" "$name" "$case_dir" "${dep:-none}" "$(date -Iseconds)" >> "$ROOT/submitted_jobs.tsv"
  echo "submitted $job $case_dir dependency=${dep:-none}"
  dep=$job
done < "$ROOT/manifest.tsv"
