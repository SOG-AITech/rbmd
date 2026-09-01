#!/bin/bash
set -u
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/peo_weak_firstpoint_repeat_20260819
MANIFEST="$ROOT/manifest.tsv"
SUBMITTED="$ROOT/submitted_jobs.tsv"
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75

active_exclusive=$(grep -R -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include=run.slurm 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
[[ "$active_exclusive" == 0 ]] || { echo "active --exclusive found" >&2; exit 2; }

while true; do
  latest_id=$(awk -F'	' 'NR>1{id=$1} END{print id}' "$SUBMITTED")
  latest_dir=$(awk -F'	' 'NR>1{d=$3} END{print d}' "$SUBMITTED")
  if [[ -n "$latest_id" ]] && squeue -h -j "$latest_id" 2>/dev/null | grep -q .; then
    sleep 60
    continue
  fi
  if [[ -n "$latest_dir" && ! -f "$latest_dir/VALIDATED_COMPLETE" ]]; then
    echo "STOP: latest case failed validation: $latest_id $latest_dir" >&2
    exit 41
  fi

  case_dir=$(awk -F'	' '
    NR==FNR {if(FNR>1) seen[$3]=1; next}
    FNR>1 && !seen[$3] {print $3; exit}
  ' "$SUBMITTED" "$MANIFEST")
  if [[ -z "$case_dir" ]]; then
    echo "all peo_weak_firstpoint_repeat cases validated"
    exit 0
  fi

  name=$(basename "$case_dir")
  if job=$(cd "$case_dir" && sbatch --parsable ./run.slurm); then
    printf '%s	%s	%s	%s	%s
'       "$job" "$name" "$case_dir" "hx1hdnormal" "$(date -Iseconds)" >> "$SUBMITTED"
    echo "submitted $job $name"
  else
    echo "submission deferred for $name" >&2
  fi
  sleep 60
done
