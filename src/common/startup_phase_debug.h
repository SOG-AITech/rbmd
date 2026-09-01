#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include "common/mpi_root_guard.hpp"

#if defined(USE_MPI) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <mpi.h>
#define RBMD_HOST_MPI_AVAILABLE 1
#else
#define RBMD_HOST_MPI_AVAILABLE 0
#endif

namespace rbmd::debug {

inline bool StartupPhaseDebugEnabled() {
  const char* env = std::getenv("RBMD_DEBUG_STARTUP_PHASES");
  return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

inline int StartupPhaseWorldSize() {
#if RBMD_HOST_MPI_AVAILABLE
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      int world_size = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      return world_size;
    }
  }
#endif
  return 1;
}

inline void StartupPhaseLog(const std::string& stage,
                            const std::string& detail = std::string()) {
  if (!StartupPhaseDebugEnabled() || stage.empty()) {
    return;
  }

  std::cerr << "[startup_phase] rank=" << rbmd::mpi::CurrentRank()
            << " size=" << StartupPhaseWorldSize() << " stage=" << stage;
  if (!detail.empty()) {
    std::cerr << " " << detail;
  }
  std::cerr << std::endl;
}

}  // namespace rbmd::debug
#undef RBMD_HOST_MPI_AVAILABLE
