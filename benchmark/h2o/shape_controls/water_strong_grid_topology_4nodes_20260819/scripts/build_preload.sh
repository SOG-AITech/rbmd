#!/bin/bash
set -euo pipefail
module purge
module load compiler/gcc/9.3.0 sghpc-mpi-gcc/26.3
cd "/public/home/acymkxdx47/rbmd/build/new_gyf_test/water_strong_grid_topology_4nodes_20260819/build"
mpicc -O2 -fPIC -shared ./force_mpi_dims.c -o ./libforce_mpi_dims.so
test -s ./libforce_mpi_dims.so
ldd ./libforce_mpi_dims.so > ./ldd.txt
sha256sum ./force_mpi_dims.c ./libforce_mpi_dims.so > ./SHA256SUMS
cat ./SHA256SUMS
