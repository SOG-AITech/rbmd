#!/bin/bash
set -u
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/ljsalt_scaling_RBE200_RBL30_20260815
MANIFEST="$ROOT/manifest.tsv"
SUBMITTED="$ROOT/submitted_jobs.tsv"
PAID_SWITCH="$ROOT/PAID_PARTITION"
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75

active_exclusive=$(grep -R -nE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include=run.slurm || true)
if [[ -n "$active_exclusive" ]]; then
  printf '%s\n' "$active_exclusive" >&2
  exit 2
fi

submit_next() {
  local mode=$1 latest case_dir case_name job partition args
  latest=$(awk -F'\t' -v m="$mode" 'NR>1 && $2==m{id=$1} END{print id}' "$SUBMITTED")
  if [[ -n "$latest" ]] && squeue -h -j "$latest" | grep -q .; then
    return 0
  fi

  case_dir=$(awk -F'\t' -v m="$mode" '
    NR==FNR {if (FNR>1 && $2==m) seen[$4]=1; next}
    FNR>1 && $2==m && !seen[$16] {print $16; exit}
  ' "$SUBMITTED" "$MANIFEST")
  [[ -z "$case_dir" ]] && return 0
  case_name=$(basename "$case_dir")
  partition=hx1hdnormal
  args=()
  if [[ -s "$PAID_SWITCH" ]]; then
    partition=$(tr -d '[:space:]' < "$PAID_SWITCH")
    [[ "$partition" == "hx1hdnormal01" ]] || {
      echo "invalid paid partition switch: $partition" >&2
      return 0
    }
    args=(--partition="$partition")
  fi
  if job=$(cd "$case_dir" && sbatch --parsable "${args[@]}" ./run.slurm); then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$job" "$mode" "$case_name" "$case_dir" "$partition" "$(date -Iseconds)" >> "$SUBMITTED"
    echo "submitted $job $mode/$case_name partition=$partition"
  else
    echo "submission deferred for $mode/$case_name" >&2
  fi
}

while true; do
  submit_next strong
  submit_next weak
  recorded=$(awk 'END{print NR-1}' "$SUBMITTED")
  active=0
  while IFS=$'\t' read -r job_id mode case case_dir partition submitted_at; do
    [[ "$job_id" == "job_id" ]] && continue
    squeue -h -j "$job_id" | grep -q . && active=$((active+1))
  done < "$SUBMITTED"
  if [[ "$recorded" -eq 12 && "$active" -eq 0 ]]; then
    echo "all 12 campaign cases left the queue"
    exit 0
  fi
  sleep 60
done
