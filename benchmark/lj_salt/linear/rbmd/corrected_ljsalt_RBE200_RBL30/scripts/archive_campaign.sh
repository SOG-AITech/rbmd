#!/bin/bash
set -euo pipefail
ROOT=/public/home/acymkxdx47/rbmd/build/new_gyf_test/rbmd_linear_subnode_20260815/corrected_ljsalt_RBE200_RBL30
OUT="$ROOT/archive/corrected_ljsalt_RBE200_RBL30_$(date +%Y%m%dT%H%M%S).tgz"
tar -czf "$OUT" --exclude='./archive' -C "$ROOT" .
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
