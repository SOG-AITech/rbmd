#pragma once
#include <cstddef>
#include "types.h"
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace rbmd {
// Non-owning view of sorted local/ghost IDs. Equal IDs resolve to the first
// local index (owned atoms precede ghosts). No allocation depends on max ID.
struct AtomIndexMapView {
  const Id* keys = nullptr;
  const LocalAtomIndex* indices = nullptr;
  std::size_t count = 0;
#if defined(__CUDACC__) || defined(__HIPCC__)
  __host__ __device__
#endif
  LocalAtomIndex operator[](Id id) const {
    std::size_t lo = 0, hi = count;
    while (lo < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (keys[mid] < id) lo = mid + 1;
      else hi = mid;
    }
    return lo < count && keys[lo] == id ? indices[lo] : LocalAtomIndex(-1);
  }
};
}  // namespace rbmd
