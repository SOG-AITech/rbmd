#!/bin/bash
set -euo pipefail
if [[ "$NB_ENGINE" != lammps || "${NB_LAMMPS_ROLE:-production}" != reference ]]; then
  LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE:?OpenMPI local size missing}
  # Both engines select local_rank from the SAME scheduler-visible device pool.
  python3 "$(dirname -- "${BASH_SOURCE[0]}")/gpu_binding.py"
  unset RBMD_GPU_LIST RBMD_DEVICE_LIST RBMD_GPU_FIRST_ID RBMD_DEVICE_FIRST_ID RBMD_GPU_COUNT RBMD_DEVICE_COUNT
  export RBMD_GPU_BINDING_STRICT=1 RBMD_GPU_BINDING_VERBOSE=1
fi
if [[ "$NB_ENGINE" == rbmd ]]; then
  COMMAND=("$RBMD_BIN" -j rbmd.json)
elif [[ "${NB_LAMMPS_ROLE:-production}" == reference ]]; then
  COMMAND=("$LAMMPS_REFERENCE_BIN" -in run.in -log log.lammps)
else
  COMMAND=("$LAMMPS_BIN" -k on g "$LOCAL_SIZE" -sf kk -pk kokkos gpu/aware on -in run.in -log log.lammps)
fi
RANK=${OMPI_COMM_WORLD_RANK:?}
if [[ "${NB_PROFILE:-off}" == all || ( "${NB_PROFILE:-off}" == rank0 && "$RANK" == 0 ) ]]; then
  command -v hipprof >/dev/null
  exec hipprof --hip-trace --mpi-trace --barrier -o "profile.rank$RANK" "${COMMAND[@]}"
fi
exec "${COMMAND[@]}"
