#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/linear_extension_two_points_20260816
OUT="$ROOT/archive/linear_extension_two_points_$(date +%Y%m%dT%H%M%S).tgz"
tar -czf "$OUT" --exclude='./archive' -C "$ROOT" .
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
