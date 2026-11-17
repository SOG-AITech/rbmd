#pragma once
#if defined(__ROCM) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
#include "types.h"
namespace rbmd {
// GPU integer atomics use unsigned storage for portable 32/64-bit overloads.
// These helpers are for nonnegative counts, capacities and bit flags only.
template<class T, class V>
__device__ inline T AtomicMaxCount(T* address, V value) {
  if constexpr (sizeof(T) == 8)
    return static_cast<T>(atomicMax(reinterpret_cast<unsigned long long*>(address),
                                    static_cast<unsigned long long>(value)));
  else
    return static_cast<T>(atomicMax(reinterpret_cast<unsigned int*>(address),
                                    static_cast<unsigned int>(value)));
}
template<class T, class V>
__device__ inline T AtomicOrCount(T* address, V value) {
  if constexpr (sizeof(T) == 8)
    return static_cast<T>(atomicOr(reinterpret_cast<unsigned long long*>(address),
                                   static_cast<unsigned long long>(value)));
  else
    return static_cast<T>(atomicOr(reinterpret_cast<unsigned int*>(address),
                                   static_cast<unsigned int>(value)));
}
__device__ inline Id AtomicAddCount(Id* address, Id value) {
#if USE_64BIT_IDS
  return static_cast<Id>(atomicAdd(reinterpret_cast<unsigned long long*>(address),
                                   static_cast<unsigned long long>(value)));
#else
  return atomicAdd(address, value);
#endif
}
}  // namespace rbmd
