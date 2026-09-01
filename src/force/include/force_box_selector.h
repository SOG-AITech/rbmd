#pragma once

#include "model/box.h"

#ifdef USE_MPI
#include "rbmd_parallel_until_locator.h"
#endif

// LAMMPS-style contract:
// - local_box drives domain decomposition and linked-cell ownership
// - periodic distance evaluation uses the global periodic box under MPI
inline Box GetPeriodicBoxForNeighborAndForce(const Box& local_box) {
#ifdef USE_MPI
  return GET_RBMD_PARALLEL->_global_structure_info.global_box;
#else
  return local_box;
#endif
}

inline Box GetPeriodicBoxForKspace(const Box& local_box) {
#ifdef USE_MPI
  return GET_RBMD_PARALLEL->_global_structure_info.global_box;
#else
  return local_box;
#endif
}
