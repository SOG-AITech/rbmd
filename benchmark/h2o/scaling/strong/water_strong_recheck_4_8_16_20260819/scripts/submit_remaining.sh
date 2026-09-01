#!/bin/bash
set -u
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_recheck_4_8_16_20260819
MANIFEST="$ROOT/manifest.tsv"
SUBMITTED="$ROOT/submitted_jobs.tsv"
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75
test "$(grep -R -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include=run.slurm | awk -F: '{s+=$2} END{print s+0}')" = 0 || exit 2

while true; do
  latest=$(awk -F'\t' 'NR>1{id=$1} END{print id}' "$SUBMITTED")
  if [[ -n "$latest" ]] && squeue -h -j "$latest" 2>/dev/null | grep -q .; then
    sleep 60
    continue
  fi
  case_dir=$(awk -F'\t' '
    NR==FNR {if(FNR>1) seen[$3]=1; next}
    FNR>1 && !seen[$16] {print $16; exit}
  ' "$SUBMITTED" "$MANIFEST")
  [[ -n "$case_dir" ]] || { echo "all recheck cases left the queue"; exit 0; }
  name=$(basename "$case_dir")
  if job=$(cd "$case_dir" && sbatch --parsable ./run.slurm); then
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$job" "$name" "$case_dir" hx1hdnormal "$(date -Iseconds)" >> "$SUBMITTED"
    echo "submitted $job $name"
  else
    echo "submission failed for $name; stopping without retry" >&2
    exit 1
  fi
  sleep 60
done
