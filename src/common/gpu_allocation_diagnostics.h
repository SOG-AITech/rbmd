#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/rbmd_define.h"

extern rbmd::Id test_current_step;

namespace rbmd {

inline std::size_t CheckedAllocationElementCount(rbmd::Id count,
                                                 const char* context) {
  if (count < 0) {
    std::ostringstream oss;
    oss << "GPU allocation negative element count"
        << " context=" << (context ? context : "unknown")
        << " step=" << test_current_step
        << " count=" << count;
    throw std::runtime_error(oss.str());
  }
  return static_cast<std::size_t>(count);
}

inline std::size_t CheckedAllocationElementProduct(
    std::size_t lhs, std::size_t rhs, const char* context) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    std::ostringstream oss;
    oss << "GPU allocation size overflow"
        << " context=" << (context ? context : "unknown")
        << " step=" << test_current_step
        << " lhs=" << lhs
        << " rhs=" << rhs;
    throw std::runtime_error(oss.str());
  }
  return lhs * rhs;
}

inline int GpuAllocationDiagnosticRank() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) return 0;

  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized != 0) return 0;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

[[noreturn]] inline void ThrowGpuAllocationFailure(
    const char* context, const std::bad_alloc& error,
    std::size_t requested_elements = 0, std::size_t element_size = 0,
    rbmd::Id native_atoms = -1, rbmd::Id ghost_atoms = -1,
    rbmd::Id total_atoms = -1) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const auto mem_status = MEMGETINFO(&free_bytes, &total_bytes);

  std::ostringstream oss;
  oss << "GPU allocation failure"
      << " context=" << (context ? context : "unknown")
      << " step=" << test_current_step
      << " rank=" << GpuAllocationDiagnosticRank()
      << " requested_elements=" << requested_elements
      << " element_size=" << element_size;
  if (requested_elements > 0 && element_size > 0 &&
      requested_elements <=
          std::numeric_limits<std::size_t>::max() / element_size) {
    oss << " requested_bytes=" << requested_elements * element_size;
  } else {
    oss << " requested_bytes=unknown";
  }
  oss << " native_atoms=" << native_atoms
      << " ghost_atoms=" << ghost_atoms
      << " total_atoms=" << total_atoms
      << " free_bytes="
      << (mem_status == SUCCESS ? std::to_string(free_bytes) : "unknown")
      << " total_gpu_bytes="
      << (mem_status == SUCCESS ? std::to_string(total_bytes) : "unknown")
      << " cause=" << error.what();
  throw std::runtime_error(oss.str());
}

template <typename Callable>
void RunWithGpuAllocationDiagnostics(
    const char* context, Callable&& callable, std::size_t requested_elements = 0,
    std::size_t element_size = 0, rbmd::Id native_atoms = -1,
    rbmd::Id ghost_atoms = -1, rbmd::Id total_atoms = -1) {
  try {
    std::forward<Callable>(callable)();
  } catch (const std::bad_alloc& error) {
    ThrowGpuAllocationFailure(context, error, requested_elements, element_size,
                              native_atoms, ghost_atoms, total_atoms);
  }
}

template <typename DeviceVector>
void ResizeDeviceVectorWithDiagnostics(
    DeviceVector& vector, std::size_t requested_elements, const char* context,
    rbmd::Id native_atoms = -1, rbmd::Id ghost_atoms = -1,
    rbmd::Id total_atoms = -1) {
  using Value = typename DeviceVector::value_type;
  RunWithGpuAllocationDiagnostics(
      context, [&]() { vector.resize(requested_elements); }, requested_elements,
      sizeof(Value), native_atoms, ghost_atoms, total_atoms);
}

}  // namespace rbmd
