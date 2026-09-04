#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/peo_weak_firstpoint_grid_2x1x2_20260819
OUT="$ROOT/archive/peo_weak_firstpoint_grid_2x1x2_$(date +%Y%m%dT%H%M%S).tgz"
tar -czf "$OUT" --exclude='./archive' -C "$ROOT" .
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
