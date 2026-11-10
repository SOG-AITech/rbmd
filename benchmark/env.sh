#!/bin/bash
# Source on a Zhengzhou compute node; LAMMPS is fixed to the selected rbm26 build.
NB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NB_ROOT
export RBMD_SOURCE=${RBMD_SOURCE:-/public/home/acymkxdx47/ReviewRBMD/rbmd}
export RBMD_BIN=${RBMD_BIN:-/public/home/acymkxdx47/ReviewRBMD/rbmd/build/rbmd}
export LAMMPS_BIN=/public/home/acymkxdx47/software/rbm26/build-hip-single/lmp
export LAMMPS_REFERENCE_BIN=${LAMMPS_REFERENCE_BIN:-$NB_ROOT/software/lammps-reference-double/lmp}
export LAMMPS_SOURCE=${LAMMPS_SOURCE:-/public/home/acymkxdx47/software/rbm26/lammps-30Sep2026}
export NB_RUNTIME=${NB_RUNTIME:-/public/home/acymkxdx47/RBMD2_revision/experiment_assets/aligned_20261007/software/runtime.env.sh}
export NB_LAMMPS_RUNTIME="$NB_ROOT/common/lammps-runtime.env.sh"
export OMP_NUM_THREADS=1
export UCX_TLS=${UCX_TLS:-^cma,rocm_ipc}
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx
