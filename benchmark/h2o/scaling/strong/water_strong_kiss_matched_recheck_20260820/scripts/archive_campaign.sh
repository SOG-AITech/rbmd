#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_kiss_matched_recheck_20260820
cd "$ROOT"
python3 scripts/collect_results.py
test "$(awk -F'\t' 'NR>1 && $12=="validated_complete"{n++} END{print n+0}' results/results.tsv)" = 6
stamp=$(date +%Y%m%dT%H%M%S)
archive="water_strong_kiss_matched_recheck_${stamp}.tgz"
tar -czf "$archive" campaign.json manifest.tsv submitted_jobs.tsv in.water cases scripts logs results
sha256sum "$archive" > "$archive.sha256"
sha256sum -c "$archive.sha256"
printf '%s\n' "$archive"
