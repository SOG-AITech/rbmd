#!/bin/bash
#SBATCH --job-name=NB-lammps-build
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=${NB_ROOT:-/public/home/acymkxdx47/ReviewRBMD/NewBenchmark}
source "$ROOT/env.sh"
source "$NB_LAMMPS_RUNTIME"
case "${1:?reference or production}" in
  reference)
    BUILD=$(dirname "$LAMMPS_REFERENCE_BIN")
    cmake -S "$LAMMPS_SOURCE/cmake" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/public/software/compiler/gcc-12.2.0/bin/g++ \
      -DBUILD_MPI=ON -DPKG_KOKKOS=OFF -DPKG_KSPACE=ON -DPKG_MOLECULE=ON \
      -DPKG_RIGID=ON -DFFT=KISS -DFFT_SINGLE=OFF
    ;;
  production)
    BUILD=$(dirname "$LAMMPS_BIN")
    test -f "$BUILD/CMakeCache.txt"
    # Preserve the existing HIP compiler and launcher; rebuild the selected binary in place.
    cmake -S "$LAMMPS_SOURCE/cmake" -B "$BUILD" -DPKG_KOKKOS=ON -DPKG_KSPACE=ON \
      -DPKG_MOLECULE=ON -DPKG_RIGID=ON -DKOKKOS_PREC=single -DFFT_SINGLE=ON -DFFT_KOKKOS=HIPFFT
    ;;
  *) echo 'Expected reference or production' >&2; exit 2;;
esac
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-8}"
"$BUILD/lmp" -h > "$BUILD/binary-help.txt"
sha256sum "$BUILD/lmp" > "$BUILD/binary.sha256"
