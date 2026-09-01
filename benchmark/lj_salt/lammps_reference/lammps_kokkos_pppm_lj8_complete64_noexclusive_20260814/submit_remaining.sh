#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
TARGETS="$ROOT/target_cases.tsv"
MANIFEST="$ROOT/submitted_jobs.tsv"
touch "$MANIFEST"
exec 9>"$ROOT/submit_remaining.lock"
flock -n 9 || exit 0
echo "START $(date -Is) pid=$$" >> "$ROOT/submit_remaining.log"

tail -n +2 "$TARGETS" | while IFS=$'\t' read -r label system mode nodes gpus atoms rx ry rz case_dir; do
  if awk -F '\t' -v d="$case_dir" 'NF >= 4 && $3 == d {found=1} END {exit !found}' "$MANIFEST"; then
    echo "SKIP $(date -Is) $case_dir already submitted" >> "$ROOT/submit_remaining.log"
    continue
  fi
  while true; do
    if output=$(cd "$case_dir" && sbatch --parsable run.slurm 2>&1); then
      job_id=${output%%;*}
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$label" "$case_dir" "$job_id" "$nodes" "$gpus" "$atoms" >> "$MANIFEST"
      echo "SUBMITTED $(date -Is) job=$job_id case=$case_dir" >> "$ROOT/submit_remaining.log"
      break
    fi
    echo "WAIT $(date -Is) case=$case_dir reason=$output" >> "$ROOT/submit_remaining.log"
    if ! grep -Eq 'AssocGrpSubmitJobsLimit|AssocGrpCpuLimit|QOS|temporarily unavailable|Socket timed out' <<<"$output"; then
      echo "FATAL $(date -Is) case=$case_dir" >> "$ROOT/submit_remaining.log"
      break
    fi
    sleep 300
  done
done
echo "COMPLETE $(date -Is)" >> "$ROOT/submit_remaining.log"
