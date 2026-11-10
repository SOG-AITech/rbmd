#!/bin/bash
#SBATCH --job-name=NB-single-build
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --time=01:00:00
#SBATCH --output=NewBenchmark-build-%j.out
set -euo pipefail
ROOT=${NB_ROOT:-/public/home/acymkxdx47/ReviewRBMD/NewBenchmark}
source "$ROOT/env.sh"
source "$NB_RUNTIME"
export HIPCC_COMPILE_FLAGS_APPEND=--gcc-toolchain=/public/software/compiler/gcc-11.4.0
export HIPCC_LINK_FLAGS_APPEND=--gcc-toolchain=/public/software/compiler/gcc-11.4.0
# Separate build directory; the existing executable is never overwritten.
BUILD="$ROOT/software/rbmd-single"
test ! -e "$BUILD/rbmd" || { echo 'Existing binary: choose a new build directory before rebuilding'; exit 2; }
SOURCE="$ROOT/software/source-timing"
if [[ ! -d "$SOURCE" ]]; then
  python3 "$ROOT/00_preflight/prepare_timing_source.py" "$RBMD_SOURCE" "$SOURCE"
fi
test -f "$SOURCE/timing_source_hashes.json"
cmake -S "$SOURCE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$HIP_PATH/bin/hipcc" -DUSE_ROCM=ON -DUSE_MPI=ON \
  -DENABLE_DOUBLE=OFF -DENABLE_64BIT_IDS=ON -DWARP_SIZE_64=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx936
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-4}"
sha256sum "$BUILD/rbmd" > "$BUILD/binary.sha256"
git -C "$RBMD_SOURCE" rev-parse HEAD > "$BUILD/source.commit"
git -C "$RBMD_SOURCE" diff > "$BUILD/source.uncommitted.patch"
