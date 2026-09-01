#pragma once

#include <cstddef>

#include "rbmd_define.h"
#include "types.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

struct GlobalTemperatureStats {
  rbmd::Real global_temp_sum = 0.0;
  rbmd::Id global_num_atoms = 0;
};

inline GlobalTemperatureStats GetGlobalTemperatureStats(
    rbmd::Real local_temp_sum, rbmd::Id local_num_atoms) {
  GlobalTemperatureStats stats{local_temp_sum, local_num_atoms};
#ifdef USE_MPI
  MPI_Allreduce(&local_temp_sum, &stats.global_temp_sum, 1, MPI_RBMD_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_num_atoms, &stats.global_num_atoms, 1, MPI_RBMD_ID,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  return stats;
}

inline rbmd::Real GetGlobalRealSum(rbmd::Real local_sum) {
  rbmd::Real global_sum = local_sum;
#ifdef USE_MPI
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_RBMD_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  return global_sum;
}

inline rbmd::Id GetGlobalIdSum(rbmd::Id local_count) {
  rbmd::Id global_count = local_count;
#ifdef USE_MPI
  MPI_Allreduce(&local_count, &global_count, 1, MPI_RBMD_ID, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  return global_count;
}

inline void AllReduceRealBufferInPlace(rbmd::Real* data, std::size_t count) {
#ifdef USE_MPI
  if (data == nullptr || count == 0) {
    return;
  }
  MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(count), MPI_RBMD_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#else
  (void)data;
  (void)count;
#endif
}
