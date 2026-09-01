#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_linear_same_sizes_shape_control_20260818
OUT="$ROOT/archive/water_linear_same_sizes_shape_control_$(date +%Y%m%dT%H%M%S).tgz"
tar -czf "$OUT" --exclude='./archive' -C "$ROOT" .
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
