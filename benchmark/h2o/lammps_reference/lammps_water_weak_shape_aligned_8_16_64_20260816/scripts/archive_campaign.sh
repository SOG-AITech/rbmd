#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/lammps_water_weak_shape_aligned_8_16_64_20260816
OUT="$ROOT/archive/lammps_water_weak_shape_aligned_8_16_64_$(date +%Y%m%dT%H%M%S).tgz"
tar -czf "$OUT" --exclude='./archive' -C "$ROOT" .
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
