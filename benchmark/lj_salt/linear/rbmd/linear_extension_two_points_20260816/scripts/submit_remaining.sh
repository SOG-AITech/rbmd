#!/bin/bash
set -u
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/linear_extension_two_points_20260816
MANIFEST="$ROOT/manifest.tsv"
SUBMITTED="$ROOT/submitted_jobs.tsv"
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75
test "$(grep -R -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include=run.slurm | awk -F: '{s+=$2} END{print s+0}')" = 0 || exit 2

while true; do
  latest=$(awk -F'\t' 'NR>1{id=$1} END{print id}' "$SUBMITTED")
  if [[ -n "$latest" ]] && squeue -h -j "$latest" | grep -q .; then
    sleep 60
    continue
  fi
  case_dir=$(awk -F'\t' '
    NR==FNR {if(FNR>1) seen[$4]=1; next}
    FNR>1 && !seen[$13] {print $13; exit}
  ' "$SUBMITTED" "$MANIFEST")
  if [[ -z "$case_dir" ]]; then
    echo "all four extension cases left the queue"
    exit 0
  fi
  system=$(basename "$(dirname "$case_dir")")
  name=$(basename "$case_dir")
  if job=$(cd "$case_dir" && sbatch --parsable ./run.slurm); then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$job" "$system" "$name" "$case_dir" "hx1hdnormal" "$(date -Iseconds)" >> "$SUBMITTED"
    echo "submitted $job $system/$name"
  else
    echo "submission deferred for $system/$name" >&2
  fi
  sleep 60
done
