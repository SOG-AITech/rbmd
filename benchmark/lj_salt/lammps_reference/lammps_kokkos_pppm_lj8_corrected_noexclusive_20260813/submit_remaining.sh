#!/bin/bash
set -euo pipefail

ROOT=/public/home/acymkxdx47/rbmd/build/lammps_kokkos_pppm_lj8_corrected_noexclusive_20260813
TARGETS="$ROOT/all_target_cases.tsv"
MANIFEST="$ROOT/submitted_jobs.tsv"
LOG="$ROOT/submit_remaining.log"

exec 9>"$ROOT/submit_remaining.lock"
flock -n 9 || exit 0
exec >>"$LOG" 2>&1
echo "START $(date -Is) pid=$$"

tail -n +2 "$TARGETS" | while IFS=$'	' read -r cutoff mode nodes gpus atoms rx ry rz case_dir; do
  if awk -F '	' -v d="$case_dir" 'NR>1 && $9==d {found=1} END {exit !found}' "$MANIFEST"; then
    echo "SKIP $(date -Is) $case_dir already submitted"
    continue
  fi
  while true; do
    if output=$(cd "$case_dir" && sbatch --parsable run.slurm 2>&1); then
      job_id=${output%%;*}
      printf '%s	%s	%s	%s	%s	%s	%s	%s	%s	%s
'         "$cutoff" "$mode" "$nodes" "$gpus" "$atoms" "$rx" "$ry" "$rz" "$case_dir" "$job_id" >> "$MANIFEST"
      echo "SUBMITTED $(date -Is) job=$job_id case=$case_dir"
      break
    fi
    echo "WAIT $(date -Is) case=$case_dir reason=$output"
    if ! grep -Eq 'AssocGrpSubmitJobsLimit|AssocGrpCpuLimit|job submit limit|violates accounting/QOS policy' <<<"$output"; then
      echo "FATAL $(date -Is) case=$case_dir"
      exit 1
    fi
    sleep 300
  done
done
echo "COMPLETE $(date -Is)"
