#!/bin/bash
set -u
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819
MANIFEST="$ROOT/manifest.tsv"
SUBMITTED="$ROOT/submitted_jobs.tsv"
exec 9>"$ROOT/logs/submit.lock"
flock -n 9 || exit 75
test -s "$ROOT/build/libforce_mpi_dims.so" || exit 3
active=$(grep -R -cE '^#SBATCH[[:space:]]+--exclusive([[:space:]]|$)' "$ROOT/cases" --include=run.slurm | awk -F: '{s+=$2} END{print s+0}')
[[ "$active" == 0 ]] || exit 2
while true; do
  id=$(awk -F'	' 'NR>1{x=$1} END{print x}' "$SUBMITTED")
  dir=$(awk -F'	' 'NR>1{x=$3} END{print x}' "$SUBMITTED")
  if [[ -n "$id" ]] && squeue -h -j "$id" | grep -q .; then sleep 60; continue; fi
  if [[ -n "$dir" && ! -f "$dir/VALIDATED_COMPLETE" ]]; then echo "STOP invalid $id $dir" >&2; exit 41; fi
  next=$(awk -F'	' 'NR==FNR{if(FNR>1) seen[$3]=1;next} FNR>1&&!seen[$3]{print $3;exit}' "$SUBMITTED" "$MANIFEST")
  if [[ -z "$next" ]]; then echo "all grid cases validated"; exit 0; fi
  name=$(basename "$next")
  if job=$(cd "$next" && sbatch --parsable ./run.slurm); then
    printf '%s	%s	%s	%s	%s
' "$job" "$name" "$next" hx1hdnormal "$(date -Iseconds)" >> "$SUBMITTED"
    echo "submitted $job $name"
  fi
  sleep 60
done
