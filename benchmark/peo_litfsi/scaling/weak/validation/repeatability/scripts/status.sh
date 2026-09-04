#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/peo_weak_firstpoint_repeat_20260819
printf 'manifest=%s submitted=%s validated=%s
'   "$(( $(wc -l < "$ROOT/manifest.tsv") - 1 ))"   "$(( $(wc -l < "$ROOT/submitted_jobs.tsv") - 1 ))"   "$(find "$ROOT/cases" -name VALIDATED_COMPLETE -type f | wc -l)"
awk -F'	' 'NR>1{print $1}' "$ROOT/submitted_jobs.tsv" | paste -sd, - | xargs -r squeue -o '%.18i %.10T %.10M %.6D %R' -j
